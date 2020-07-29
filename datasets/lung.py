import logging
import sys
from pathlib import Path
from typing import Union, List, Tuple, Callable, Any, Dict

import SimpleITK as sitk
import numpy as np
import pandas
# from edipy.dicom.metadata import ImageTag, Tag
# from edipy.sitk import loading, resampling
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import paths, pytorch, stats, plots
from utils.data import pad_or_truncate, Split, HistogramNormalisation
from utils.dicom import company_formatted_id
from utils.logging import configure_logging
from utils.pytorch import NumpyDataset

logger = logging.getLogger(__name__)


def dataset_name() -> str:
    return 'Lung'


def debug_root() -> Path:
    return paths.debug_output_root() / dataset_name()


def data_root() -> Path:
    return paths.input_data_root() / dataset_name()


def histogram_root() -> Path:
    return data_root() / 'training_histograms'


def dataset_file() -> Path:
    return data_root() / f'{dataset_name()}_data.csv'


def raw_data_root() -> Path:
    return data_root() / 'raw'


def preprocessed_data_root() -> Path:
    return data_root() / 'preprocessed'


def preprocessed_dicom_root() -> Path:
    return preprocessed_data_root() / 'dicom'


def preprocessed_gt_root() -> Path:
    return preprocessed_data_root() / 'gt'


def sub_folder(root: Path, patient_id: Union[str, int], series_number: int) -> Path:
    return root / company_formatted_id(patient_id) / str(series_number)


def padding_value() -> int:
    return -2048


def valid_image_range() -> Tuple[int, int]:
    return padding_value(), 3000


def clipped_image_range() -> Tuple[int, int]:
    return -500, 400


def ltrc_info(all_data: pandas.DataFrame, ltrc_to_edidicom: pandas.DataFrame) -> pandas.DataFrame:
    clinical_data_root = raw_data_root() / 'LTRC_R026_Clinical_data'
    ltrc_data_file = clinical_data_root / 'LTRC_CT_Data_Information.csv'
    ltrc_data_info = pandas.read_csv(ltrc_data_file)

    ltrc_data = all_data[all_data.origin == 'LTRC']

    unique_ltrc = ltrc_data_info.drop_duplicates(subset='ID')
    manufacturers = list()

    data_iter = tqdm(zip(ltrc_data.edidicom_id, ltrc_data.edidicom_series), file=sys.stdout)
    for edidicom_id, edidicom_series in data_iter:
        data_iter.set_description(f'Finding manufacturer info for {edidicom_id}, {edidicom_series}')
        ltrc_id = ltrc_to_edidicom.query(f'edidicom_id=={edidicom_id}')['ltrc_id'].item()
        manufacturer = unique_ltrc.query(f'ID=={ltrc_id}')['Manufacturer'].item()
        manufacturers.append(manufacturer)

    ltrc_data = ltrc_data.assign(manufacturer=manufacturers)

    return ltrc_data


def get_dicom_info(edidicom_id_and_series_numbers: List[Tuple[int, int]],
                   labels_and_tags: Dict[str, Tag]) -> Dict[str, List[Any]]:
    patient_iter = tqdm(edidicom_id_and_series_numbers, file=sys.stdout)
    values = {label: list() for label in labels_and_tags.keys()}
    for edidicom_id, series_number in patient_iter:
        patient_iter.set_description(f'Gathering DICOM info for {edidicom_id}, {series_number}')
        patient_id = company_formatted_id(patient_id=edidicom_id)

        image = loading.load_dicom_from_library(patient_id=patient_id, series_number=series_number,
                                                force_cache_consistent=False)
        for label, tag in labels_and_tags.items():
            dicom_value = None
            if image.HasMetaDataKey(tag.to_tag_string()):
                dicom_value = image.GetMetaData(tag.to_tag_string())
            values[label].append(dicom_value)

    return values


def generate_dataset_file(output_file: Path):
    logger.info(f'Generating dataset file at {output_file}')

    lung_segmentation_data_root = raw_data_root() / 'LungSegmentation'

    training_split_file = lung_segmentation_data_root / 'master_training.csv'
    final_evaluation_split_file = lung_segmentation_data_root / 'master_final_evaluation.csv'

    training_info = pandas.read_csv(training_split_file)
    final_evaluation_info = pandas.read_csv(final_evaluation_split_file)

    all_data = pandas.concat([training_info, final_evaluation_info])

    ltrc_to_edidicom_file = lung_segmentation_data_root / 'ltrc_to_edidicom_map.csv'
    ltrc_to_edidicom = pandas.read_csv(ltrc_to_edidicom_file)
    ltrc_data = ltrc_info(all_data=all_data, ltrc_to_edidicom=ltrc_to_edidicom)

    labels_and_tags = {'manufacturer': ImageTag.Manufacturer}
    other_data = all_data[all_data.origin != 'LTRC']
    ids_and_series_numbers = list(zip(other_data.edidicom_id, other_data.edidicom_series))
    dicom_values = get_dicom_info(edidicom_id_and_series_numbers=ids_and_series_numbers,
                                  labels_and_tags=labels_and_tags)

    other_data = other_data.assign(**dicom_values)

    output_data = ltrc_data.append(other_data)
    output_data.to_csv(output_file, index=False)


def load_raw_gt(gt_folder: Path) -> Union[sitk.Image, None]:
    gt_file_paths = [file for file in gt_folder.glob('*.nrrd') if 'pleural' not in file.stem.lower()]
    if len(gt_file_paths) == 0:
        return None

    cumulative_gt = None
    for file in gt_file_paths:
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NrrdImageIO")
        reader.SetOutputPixelType(sitk.sitkUInt8)
        reader.SetFileName(str(file))
        gt = reader.Execute()

        if cumulative_gt is None:
            cumulative_gt = gt
        else:
            cumulative_gt = sitk.Add(cumulative_gt, gt)

    assert cumulative_gt is not None, "No ground truth found!"

    cumulative_gt = sitk.BinaryThreshold(cumulative_gt, lowerThreshold=0.0, upperThreshold=0.0, insideValue=0,
                                         outsideValue=1)

    return cumulative_gt


def preprocess_volumes(image: sitk.Image, gt: sitk.Image) -> Tuple[np.ndarray, np.ndarray]:
    target_mm_per_pixel = 1.0
    try:
        dicom_padding_value = int(image.GetMetaData("PixelPaddingValue"))
    except Exception:
        dicom_padding_value = padding_value()
    gt_padding_value = 0
    target_image_shape = (512, 512)

    # Use the space transforms from the input image
    resampler = resampling.get_identity_resampler(image)
    resampler = resampling.isotropic_resampler(target_mm_per_pixel, resampler)
    resampler = resampling.encapsulate_old_bounds_resampler(image, resampler)
    resampler.SetInterpolator(sitk.sitkLinear)

    image_array = sitk.GetArrayFromImage(image)
    gt_array = sitk.GetArrayFromImage(gt)

    assert image_array.shape == gt_array.shape, \
        f'Mismatch in DICOM and GT array shapes: DICOM = {image.shape}, GT = {gt.shape}'

    resampler.SetDefaultPixelValue(dicom_padding_value)
    resampled_image = resampling.execute_resampler(image, resampler, padding_aware=True, smooth_before=True)

    resampler.SetDefaultPixelValue(gt_padding_value)
    resampled_gt = resampling.execute_resampler(gt, resampler, padding_aware=False, smooth_before=False)

    resampled_image = sitk.GetArrayFromImage(resampled_image)
    resampled_gt = sitk.GetArrayFromImage(resampled_gt)

    assert resampled_image.shape == resampled_gt.shape, \
        f'Mismatch in resampled DICOM and GT array shapes: ' \
        f'DICOM = {resampled_image.shape}, GT = {resampled_gt.shape}'

    num_slices = resampled_image.shape[0]
    resampled_image = pad_or_truncate(input_data=resampled_image,
                                      output_shape=(num_slices, *target_image_shape),
                                      padding_value=dicom_padding_value)

    resampled_gt = pad_or_truncate(input_data=resampled_gt,
                                   output_shape=(num_slices, *target_image_shape),
                                   padding_value=gt_padding_value)

    resampled_gt = np.where(resampled_gt >= 0.5, 1.0, 0.0).astype(np.uint8)
    resampled_image = np.around(resampled_image).astype(np.int16)

    return resampled_image, resampled_gt


def preprocess(patient_info: List[Tuple[int, int]], output_images: bool = False):
    gt_root = raw_data_root() / 'LungSegmentation'

    patient_iter = tqdm(patient_info, file=sys.stdout)
    for patient_id, series_number in patient_iter:
        patient_id = company_formatted_id(patient_id)
        patient_iter.set_description(f'Running preprocessing for {patient_id}, {series_number}')

        dicom_output_folder = sub_folder(preprocessed_dicom_root(), patient_id, series_number)
        gt_output_folder = sub_folder(preprocessed_gt_root(), patient_id, series_number)

        if dicom_output_folder.exists() and gt_output_folder.exists():
            continue

        try:
            image = loading.load_dicom_from_library(patient_id=patient_id, series_number=series_number,
                                                    use_calculated_padding=True, force_cache_consistent=False)
        except Exception as e:
            logger.info(f'Failed to load {patient_id}, {series_number}: {e}')
            continue

        patient_gt_folder = gt_root / patient_id
        gt = load_raw_gt(gt_folder=patient_gt_folder)

        if gt is None:
            logger.warning(f'Could not find any GT for {patient_id}, {series_number} in {patient_gt_folder}')
            continue

        preprocessed_image, preprocessed_gt = preprocess_volumes(image=image, gt=gt)

        dicom_output_folder.mkdir(parents=True, exist_ok=True)
        gt_output_folder.mkdir(parents=True, exist_ok=True)
        for slice_number, (dicom_slice, gt_slice) in enumerate(zip(preprocessed_image, preprocessed_gt)):
            file_name = f'{slice_number}.npy'
            np.save(dicom_output_folder / file_name, dicom_slice)
            np.save(gt_output_folder / file_name, gt_slice)

        if output_images:
            plots.volume_mips({'Image': preprocessed_image, 'GT': preprocessed_gt},
                              output_file=debug_root() / 'preprocessed' / f'{patient_id}_{series_number}.png',
                              title=f'Preprocessed images for {patient_id}, {series_number}')


def create_data_filter(splits: Union[Split, List[Split]], filters: Dict[str, Union[str, List[str]]]) -> str:
    data_filter = None
    if splits is not None:
        if not isinstance(splits, list):
            splits = [splits]
        split_filter = ' | '.join(f'split == "{split}"' for split in splits)
        data_filter = f'({split_filter})'

    if filters is not None and len(filters) > 0:
        queries = list()
        for name, values in filters.items():
            if not isinstance(values, list):
                values = [values]
            query = ' | '.join(f'{name} == "{value}"' for value in values)
            queries.append(query)
        other_filters = ' & '.join(f'({query})' for query in queries)
        data_filter = other_filters if data_filter is None else f'{data_filter} & ({other_filters})'

    return data_filter


def patient_ids_and_series_numbers(splits: Union[Split, List[Split]] = None,
                                   filters: Dict[str, Union[str, List[str]]] = None,
                                   dataset_file_path: Path = None) -> List[Tuple[int, int]]:
    if not dataset_file_path:
        dataset_file_path = dataset_file()
    df = pandas.read_csv(dataset_file_path)

    data_filter = create_data_filter(splits=splits, filters=filters)

    if data_filter:
        logger.debug(f'Filtering dataset using {data_filter}')
        df = df.query(data_filter)

    # The GT for these datasets is not being preprocessed correctly so we just exclude them
    exclude = [(9499, 5), (9500, 4), (9501, 5), (9502, 4), (9504, 6), (9507, 6), (9512, 5), (9513, 9)]
    patients = [patient for patient in zip(df.edidicom_id, df.edidicom_series) if patient not in exclude]
    logger.debug(f'Found {len(patients)} patient datasets')
    return patients


def single_volume_datasets(patient_info: List[Tuple[int, int]],
                           image_transform: Union[Callable[[np.ndarray], np.ndarray], None] = None,
                           gt_transform: Union[Callable[[np.ndarray], np.ndarray], None] = None) \
        -> List[Dataset]:
    datasets = list()

    for patient_id, series_number in patient_info:
        dicom_folder = sub_folder(root=preprocessed_dicom_root(), patient_id=patient_id,
                                  series_number=series_number)
        gt_folder = sub_folder(root=preprocessed_gt_root(), patient_id=patient_id,
                               series_number=series_number)

        try:
            dataset = NumpyDataset(input_folder=dicom_folder, gt_folder=gt_folder,
                                   input_transform=image_transform, gt_transform=gt_transform,
                                   label=f'{patient_id}_{series_number}')
        except Exception as e:
            logger.info(f'Failed to create dataset for {patient_id}, {series_number}: {e}')
            continue

        datasets.append(dataset)
    return datasets


def compute_training_histograms():
    min_image_value, max_image_value = valid_image_range()

    patient_info = patient_ids_and_series_numbers(splits=Split.training_splits())
    datasets = single_volume_datasets(patient_info=patient_info)

    data_iter = tqdm(zip(patient_info, datasets), file=sys.stdout, desc='Computing training histograms')
    for (patient_id, series_number), dataset in data_iter:
        output_file = histogram_root() / f'{patient_id}_{series_number}.npy'
        if output_file.exists():
            continue
        histogram = pytorch.dataset_histogram(datasets=dataset, data_min=min_image_value,
                                              data_max=max_image_value)
        stats.save_histogram(histogram=histogram, file_path=output_file)


def data_transforms(filters: Dict[str, Union[str, List[str]]]) \
        -> Tuple[Union[Callable[[np.ndarray], np.ndarray], None],
                 Union[Callable[[np.ndarray], np.ndarray], None]]:
    min_image_value, max_image_value = clipped_image_range()

    logger.debug('Creating histogram normalisation transform from training data')

    patient_info = patient_ids_and_series_numbers(splits=Split.training_splits(), filters=filters)

    histogram = None
    for patient_id, series_number in patient_info:
        histogram_path = histogram_root() / f'{patient_id}_{series_number}.npy'
        patient_histogram = np.load(histogram_path)
        histogram = stats.update_histogram(current_histogram=histogram, new_histogram=patient_histogram)

    image_transform = HistogramNormalisation(histogram=histogram, min_value=min_image_value,
                                             max_value=max_image_value)

    logger.debug(f'Applying normalisation of data clipped to [{min_image_value}, {max_image_value}] using: '
                 f'(x - {image_transform.subtractor}) / {image_transform.divisor}')

    gt_transform = None

    return image_transform, gt_transform


def get_datasets(splits: Union[Split, List[Split]], normalised: bool,
                 filters: Dict[str, Union[str, List[str]]] = None) -> List[Dataset]:
    patient_info = patient_ids_and_series_numbers(splits=splits, filters=filters)

    image_transform = None
    gt_transform = None
    if normalised:
        image_transform, gt_transform = data_transforms(filters=filters)
    return single_volume_datasets(patient_info=patient_info, image_transform=image_transform,
                                  gt_transform=gt_transform)


def verify_volumes(patient_info: List[Tuple[int, int]]):
    datasets = single_volume_datasets(patient_info=patient_info, image_transform=None, gt_transform=None)
    failed_datasets = list()
    volume_iter = tqdm(datasets, file=sys.stdout)
    for dataset in volume_iter:
        volume, gt, metadata = pytorch.dataset_to_volume(dataset=dataset)
        volume_iter.set_description(f'Verifying {metadata[0]}')

        if not np.any(gt):
            failed_datasets.append(metadata[0])
    if len(failed_datasets) > 0:
        logger.warning(f'{len(failed_datasets)} (out of {len(datasets)}) with an empty GT volume!')
        for info in failed_datasets:
            logger.warning(f'    ({info}),')
    else:
        logger.info(f'Verified {len(datasets)} volumes')


def main():
    verbose = False
    debug_mode = False

    configure_logging(log_debug=verbose)

    dataset_file_path = dataset_file()
    if not dataset_file_path.exists():
        generate_dataset_file(dataset_file_path)

    patient_info = patient_ids_and_series_numbers(dataset_file_path=dataset_file_path)

    preprocess(patient_info=patient_info, output_images=debug_mode)

    compute_training_histograms()

    verify_volumes(patient_info=patient_info)


if __name__ == '__main__':
    main()

import os
import platform
from pathlib import Path
from typing import List, Tuple

# import bcolz
import numpy as np
import pandas as pd
import torch
# from edipy.io import nrrd
from torch.utils.data import Dataset

from utils import paths, dicom
from utils.data import Split

# DATA PREPROCESSING
annotator = 4  # there are 4 annotator for multi organ task on LTRC and NLST
watermark = 7000  # the white square burned in on slices with annotations
ignor_datasets = []


def data_root_slices() -> Path:
    if platform.system() == "Linux":
        return Path("/home/anli@local.tmvse.com/data/multiO/")
    else:
        return paths.nutfiles_data_root() / "ToAnnotate" / "Annotated_MultiObservers"


def data_root_vol() -> Path:
    if platform.system() == "Linux":
        return Path("/home/anli@local.tmvse.com/data/multiOL/")
    else:
        return paths.nutfiles_data_root() / "Labelled"


def multi_organ_data_root() -> Path:
    multio_path = paths.input_data_root() / 'MultiOrgan'
    if not os.path.exists(str(multio_path)):
        os.mkdir(str(multio_path))
    return multio_path


def preprocessed_data_root() -> Path:
    preproces_multio_path = multi_organ_data_root() / 'preprocessed'
    if not os.path.exists(str(preproces_multio_path)):
        os.mkdir(str(preproces_multio_path))
    return preproces_multio_path


def get_ground_truth_filelist(patient_id: str, path: Path, annotator: int) -> List:
    """

    :param patient_id:
    :param path:
    :return:a list of absolute filepaths to GT seg.nrrd segmentations
    """
    gt_folder = Path(path) / patient_id / "AnatomySegmentation"
    if annotator == 1:
        gt_files = [Path("{0}/Segmentation.seg.nrrd".format(gt_folder))]

    else:
        gt_files = [Path("{0}/Segmentation{1}.seg.nrrd".format(gt_folder, annotator))]
    print(gt_files)

    assert 0 < len(gt_files) <= 2, f"Expected 1 or 2 GT files only, found {len(gt_files)}."
    absolute_filepaths = sorted(gt_files)  # to get newest annotations
    return absolute_filepaths


def load_ground_truth(patient_id: str, path: Path, annotator: int) -> Tuple[np.ndarray, dict]:
    """

    :param patient_id:
    :param path
    :return:
    """
    gt_files = get_ground_truth_filelist(patient_id, path, annotator)

    for file in gt_files:
        gt, header = nrrd.reader.read(file)

    return gt, header


def convert_gt_seg_to_vol(gt: np.ndarray, h: dict, vol_shape: tuple) -> List:
    """

    :param gt:
    :param h:
    :param vol_shape:
    :return:
    """
    h = h['keyvaluepairs']
    gt_list = []
    z, y, x, organs = gt.shape

    for organ in range(0, organs):
        offsets = list(map(int, h['Segmentation_ReferenceImageExtentOffset'].split(' ')))
        ox, oy, oz = offsets
        selector = [slice(o, o + s) for o, s in zip([oz, oy, ox], gt.shape[:-1])]
        image = np.zeros(vol_shape)
        image[tuple(selector)] = gt[..., organ]
        gt_list.append(image)
    return gt_list


def load_gt_list_with_names(patient_id: str, vol_shape: tuple, path: Path, annotator: int) -> Tuple[List, List]:
    """

    :param patient_id:
    :param vol_shape:
    :param path: from where to load gt data
    :return:
    """
    gt, h = load_ground_truth(patient_id, path, annotator)
    gt_list = convert_gt_seg_to_vol(gt, h, vol_shape)
    name_list = get_gt_name_list(gt, h)

    return gt_list, name_list


def get_gt_name_list(gt: np.ndarray, h: dict) -> List:
    """

    :param gt:
    :param h: header
    :return:
    """
    _, _, _, num_seg = gt.shape
    names_list = []
    for i in range(0, num_seg):
        names_list.append(h['keyvaluepairs'][f'Segment{i}_Name'])
    return names_list


def slices_with_gt(gt: np.ndarray) -> List:
    """
    :param gt:
    :return: list of indices for slices that have annotations
    """
    indicies = []
    for i in range(len(gt)):
        if np.max(gt[i, ...]) > 0:
            indicies.append(i)
    return indicies


def get_all_gt_indicies(gt_list: list) -> np.ndarray:
    """
    :param gt_list:  list with gt volumes for each task
    :return: slices that have at least one task annotated
    """
    idx_cummulative_list = []
    for gt in gt_list:
        indices = slices_with_gt(gt)
        idx_cummulative_list = idx_cummulative_list + indices
    return np.unique(idx_cummulative_list)


def name_mapping(name: str) -> str:
    """

    :param name: name in the header as given by annotator
    :return: unified name
    """
    name = name.lower()

    if "right" in name:
        name = "r_lung"
    elif "left" in name:
        name = "l_lung"
    if "r_lung" in name:
        name = "r_lung"
    elif "l_lung" in name:
        name = "l_lung"
    elif "spinal" in name:
        name = "spinal_cord"
    elif "trachea" in name:
        name = "trachea"
    elif "oesophagus" in name:
        name = "oesophagus"

    return name


def all_organs() -> List[str]:
    return ['l_lung', 'r_lung', 'oesophagus']#, "spinal_cord", "trachea"]


def patient_path(patient_id: str) -> Path:
    """
    Given patient id return path to the preprocessed bcolz folder
    :param patient_id:
    :return:
    """
    patient_path = preprocessed_data_root() / patient_id
    if not os.path.exists(str(patient_path)):
        os.mkdir(str(patient_path))
    return patient_path


def experiment_data_folder_path(split: str) -> Path:
    """

    :param split:
    :return:
    """

    folder_path = preprocessed_data_root() / split
    if not os.path.exists(str(folder_path)):
        os.mkdir(str(folder_path))
    return folder_path


def organ_data_folder(path: Path, organ: str) -> Path:
    organ_path = path / organ
    if not os.path.exists(str(organ_path)):
        os.mkdir(str(organ_path))
    return organ_path


def preprocess_patient_10_slice_annotation(patient_id: str, annotator=annotator):
    """
    Takes patient id, loads corresponding ct and gt. This is for datasets that have only 10 slices annotated,
    but each slices should have gt for each organ.
     Saves annotated ct and gt slices for the patient as bcolz
    :param annotator:
    :param patient_id:
    :return:
    """

    filepath = data_root_slices() / patient_id / "CT"
    ct, spacing = dicom.load_image_volume(filepath)
    gt_list, gt_name_list = load_gt_list_with_names(patient_id, ct.shape, data_root_slices(), annotator)
    indices = get_all_gt_indicies(gt_list)

    bcolz.carray(np.expand_dims(ct[indices], axis=-1),
                 rootdir="{0}/ct_slices".format(patient_path(patient_id)),
                 mode='w')
    for gt, name in zip(gt_list, gt_name_list):
        organ = name_mapping(name)
        xb = bcolz.carray(np.expand_dims(gt[indices], axis=-1),
                          rootdir="{0}/gt_{1}".format(patient_path(patient_id), organ),
                          mode='w')
        xb.flush()
        xb = None


def preprocess_patient_multi_slice_annotation(patient_id: str, annotator: int = 1):
    """
    Takes patient id, loads corresponding ct and gt. Saves annotated ct and gt slices for the patient as bzolz
    :param annotator:
    :param patient_id:
    :return:
    """

    filepath = data_root_vol() / patient_id / "CT"
    ct, spacing = dicom.load_image_volume(filepath)
    gt_list, gt_name_list = load_gt_list_with_names(patient_id, ct.shape, data_root_vol(), annotator)
    indices = get_all_gt_indicies(gt_list)  # all slices with annotations

    bcolz.carray(np.expand_dims(ct[indices], axis=-1),
                 rootdir="{0}/ct_slices".format(patient_path(patient_id)),
                 mode='w')

    for gt, name in zip(gt_list, gt_name_list):
        annotated = slices_with_gt(gt)  # annotated slices for the organ
        organ = name_mapping(name)
        missing = list(set(indices).difference(set(annotated)))
        print("Before", np.min(gt))
        gt = missing_slices(gt, missing)
        print("After", np.min(gt))
        xb = bcolz.carray(np.expand_dims(gt[indices], axis=-1),
                          rootdir="{0}/gt_{1}".format(patient_path(patient_id), organ),
                          mode='w')
        xb.flush()
        xb = None


def missing_slices(gt: np.ndarray, indices_of_slices_with_missing_gt: List) -> np.ndarray:
    """
    set slices with missing annotation to -1
    :return:
    """
    missing_annotation = np.zeros_like(gt[0]) - 1
    for i in indices_of_slices_with_missing_gt:
        gt[i] = missing_annotation

    return gt


def preprocess_slice_dataset(patient_list: list):
    """
    :param patient_list:
    :return:
    """
    for patient in patient_list:
        preprocess_patient_10_slice_annotation(patient)


def preprocess_vol_dataset(patient_list: list):
    """
    :param patient_list:
    :return:
    """
    for patient in patient_list:
        preprocess_patient_multi_slice_annotation(patient)


def get_all_valid_raw_data(path: Path) -> List:
    patient_in_folder = [p for p in os.listdir(str(path)) if ".csv" not in p]
    patient_list = [x for x in patient_in_folder if x not in ignor_datasets]

    return patient_list


def get_preprocessed_patient(patient_id: str, organ: str) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param patient_id:
    :param organ: the name of the organ which annotation we want to retriev
    :return: x and y
    """
    # ct_slice_path = "{0}/ct_slices_{1}".format(patient_path(patient_id), organ)
    # if not os.path.exists(str(ct_slice_path)):
    ct_slice_path = "{0}/ct_slices".format(patient_path(patient_id))
    x = np.array(bcolz.open(rootdir=ct_slice_path))
    y = np.array(bcolz.open(rootdir="{0}/gt_{1}".format(patient_path(patient_id), organ)))

    return x, y


def get_all_preprocessed_patients(split: str) -> List:
    """

    :return: list of preprocessed patients
    """
    assert split is not None
    patients = set(os.listdir(str(preprocessed_data_root())))
    df = pd.read_csv(data_root_slices() / "multi_organ_data.csv")
    subset_df = df.loc[df['split'] == split]
    patients_of_intrest = list(patients.intersection(set(subset_df.patient_id)))

    return patients_of_intrest


def normalize_hu_values(vol, v_min=-1000, v_max=500) -> np.ndarray:
    """
    Normalise images. Assumes the input np.array contains h.u. values.
    :param vol: The volume to be normalised.
    :param v_min: The minimum h.u. value to clamp to.
    :param v_max: The maximum h.u. value to clamp to.
    :return: the normalised volume.
    """
    # strip out high padding values. It is a bit of bodge this being performed here.
    vol = np.clip(vol, v_min, v_max)
    vol = vol - v_min
    vol = vol / (v_max - v_min)

    return vol


def get_patients(patient_list, organ) -> Tuple[List, List]:
    X, Y = [], []
    for patient in patient_list:
        x, y = get_preprocessed_patient(patient, organ)
        X.append(x)
        Y.append(y)

    return X, Y


def create_per_patient_dataset(patient_id: str, experiment_data_folder: Path, organ: str):
    patient_path = experiment_data_folder / patient_id
    if not os.path.exists(str(patient_path)):
        os.mkdir(str(patient_path))

    x, y = get_preprocessed_patient(patient_id, organ)
    norm_X = normalize_hu_values(np.squeeze(x))
    samples = [(x, y) for x, y in zip(norm_X, np.squeeze(y))]

    for i, s in enumerate(samples):
        bcolz.carray(s[0], rootdir="{0}/ct_slice_{1}".format(patient_path, i), mode='w')
        bcolz.carray(s[1], rootdir="{0}/gt_{1}_{2}".format(patient_path, organ, i), mode='w')


def prepare_slice_dataset(split: str, organ: str):
    patient_list = get_all_preprocessed_patients(split)
    path = experiment_data_folder_path(split)

    for patient in patient_list:
        create_per_patient_dataset(patient, path, organ)


def process_item(x):
    x = torch.tensor(x).squeeze()
    x = x.view(1, x.shape[0], x.shape[1]).permute([0, 1, 2]).float()

    return x


class MultiOrganSinglePatientDataset(Dataset):

    def __init__(self, split=None, patient_id=None, organ_list=None, organ_with_present_gt=None,
                 dataset_file_path: Path = None, transform=None):
        self.organ_with_present_gt = organ_with_present_gt if organ_with_present_gt else all_organs()
        self.organ_list = organ_list if organ_list else all_organs()
        self.split = split
        self.patient = patient_id
        self.patient_path = dataset_file_path if dataset_file_path else \
            experiment_data_folder_path(self.split) / patient_id
        self.len = len([s for s in os.listdir(self.patient_path) if "ct" in s])
        self.transform = transform

        # Getter

    def __getitem__(self, index):
        if index >= self.len:
            raise IndexError
        x_path = "{0}/ct_slice_{1}".format(self.patient_path, index)
        x = bcolz.open(rootdir=x_path)
        y_list = []
        for organ in self.organ_list:
            if organ in self.organ_with_present_gt:
                y = bcolz.open(rootdir="{0}/gt_{1}_{2}".format(self.patient_path, organ, index))
            else:
                y = np.zeros_like(x) - 1
            y_list.append(y)
        # If there is any transform method, apply it onto the image and gt
        if self.transform:
            x = self.transform(x)
            y_list = [self.transform(y) for y in y_list]

        metadata = index  # to match the pattern from domain adaptation

        return x, y_list, metadata

    # Get Length
    def __len__(self):
        return self.len


def get_multi_organ_dataset(split: str, organ_list: list, present_organs: list, debug_mode: bool) -> List[Dataset]:
    datasets = []
    patient_list = get_all_preprocessed_patients(split)
    if debug_mode:
        patient_list = patient_list[:2]
    for patient in patient_list:
        d = MultiOrganSinglePatientDataset(split=split, organ_list=organ_list, organ_with_present_gt=present_organs,
                                           patient_id=patient, transform=process_item)
        datasets.append(d)

    return datasets


def main():
    # preprocess_slice_dataset(get_all_valid_raw_data(data_root_slices())[:1])
    # preprocess_vol_dataset(get_all_valid_raw_data(data_root_vol()))
    splits = Split.training_splits() + Split.test_splits()
    for split in splits:
        for organ in all_organs():
            prepare_slice_dataset(str(split), organ)


if __name__ == '__main__':
    main()
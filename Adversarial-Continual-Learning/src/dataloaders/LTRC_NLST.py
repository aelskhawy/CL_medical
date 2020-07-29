
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import platform
from pathlib import Path
from typing import List, Tuple

# import bcolz
import numpy as np
import pandas as pd
import torch
# from edipy.io import nrrd
from torch.utils.data import Dataset

# from utils.data import Split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from distutils.dir_util import copy_tree

# DATA PREPROCESSING
annotator = 4  # there are 4 annotator for multi organ task on LTRC and NLST
watermark = 7000  # the white square burned in on slices with annotations

def all_organs() -> List[str]:
    return ['l_lung', 'r_lung', "spinal_cord", "trachea",'oesophagus' ] # OrderC
    # return ['oesophagus']
    # return ['spinal_cord']
    # return ['trachea']
    # return ['oesophagus', 'spinal_cord', 'trachea' ]
    # return ['l_lung', 'r_lung']
    # return ['l_lung', 'r_lung', 'oesophagus',]
    # return ['l_lung', 'r_lung', 'spinal_cord', 'trachea', 'oesophagus' ]
    # return ['spinal_cord', 'r_lung', 'l_lung', 'trachea', 'oesophagus']  # OrderA
    # return ['oesophagus', 'trachea',  'l_lung', 'r_lung', 'spinal_cord'  ]  # OrderB

def data_root_slices() -> Path:
    # return Path("C:\skhawy\Canon\Code\LTRC_NLST_raw_data\Annotated_MultiObservers") # local
    return Path("/home/abel@local.tmvse.com/skhawy/Canon/Code/ADP_ContinualLearning/data/MultiOrgan_LTRC_NLST")
    # return Path("/c/skhawy/Canon/Code/LTRC_NLST_raw_data")

def data_root_vol() -> Path:
    return Path("C:\skhawy\Canon\Code\LTRC_NLST_raw_data\Labelled")

def multi_organ_data_root() -> Path:
    multio_path = data_root_slices() / 'MultiOrgan'
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
    # print(gt_files)

    assert 0 < len(gt_files) <= 2, f"Expected 1 or 2 GT files only, found {len(gt_files)}."
    absolute_filepaths = sorted(gt_files)  # to get newest annotations
    return absolute_filepaths


def load_ground_truth(patient_id: str, path: Path, annotator: int) -> Tuple[np.ndarray, dict]:

    gt_files = get_ground_truth_filelist(patient_id, path, annotator)

    for file in gt_files:
        gt, header = nrrd.reader.read(file)

    return gt, header


def convert_gt_seg_to_vol(gt: np.ndarray, h: dict, vol_shape: tuple) -> List:
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


def name_mapping(name: str) -> str:
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


def patient_path(patient_id: str) -> Path:
    patient_path = preprocessed_data_root() / patient_id
    if not os.path.exists(str(patient_path)):
        os.mkdir(str(patient_path))
        os.mkdir(os.path.join(str(patient_path), 'PNG'))
    return patient_path


def experiment_data_folder_path(split: str) -> Path:
    folder_path = preprocessed_data_root() / split
    if not os.path.exists(str(folder_path)):
        os.mkdir(str(folder_path))
    return folder_path



def visualize_annotated_slices(anno_slices, gt, organ, path):
    nrows = anno_slices.shape[0] if anno_slices.shape[0]  > 1 else 2
    fig, ax = plt.subplots(nrows,1, figsize=[20, 50])
    for slice_num in range(anno_slices.shape[0]):
        ax[slice_num].set_title('x+gt')
        ax[slice_num].imshow(anno_slices[slice_num], cmap='gray')
        ax[slice_num].imshow(gt[slice_num], alpha = 0.8)
        ax[slice_num].contour(gt[slice_num])
        ax[slice_num].axis('off')

    path_name = os.path.join(path, "PNG")
    paths_name = path_name + "/{0}.png".format(organ)
    plt.savefig(paths_name)
    plt.close(fig)


def preprocess_patient_10_slice_annotation(patient_id: str, annotator=annotator):
    """
    Takes patient id, loads corresponding ct and gt. This is for datasets that have only 10 slices annotated,
    but each slices should have gt for each organ.
     Saves annotated ct and gt slices for the patient as bcolz
    :param annotator:
    :param patient_id:
    :return:
    """
    print("Now processing {}".format(patient_id))
    save_dir = data_root_slices() / patient_path(patient_id)
    filepath = data_root_slices() / patient_id / "CT"
    ct, spacing = dicom.load_image_volume(filepath)
    # print(np.min(ct), np.max(ct))
    ct = normalize_hu_values(ct)
    # print(np.min(ct), np.max(ct))
    gt_list, gt_name_list = load_gt_list_with_names(patient_id, ct.shape, data_root_slices(), annotator)
    # indices = get_all_gt_indicies(gt_list)


    for name, gt in zip(gt_name_list, gt_list):
        nnz_idx = (np.max(gt, axis=(1, 2)) > 0)
        one_gt = gt[nnz_idx]
        anno_slices = ct[nnz_idx]
        organ = name_mapping(name)
        # print(anno_slices.shape, one_gt.shape, organ)
        if anno_slices.shape[0] !=0:
            np.savez_compressed(os.path.join(save_dir, "{}_ct.npz".format(organ)), anno_slices)
            np.savez_compressed(os.path.join(save_dir, "{}_gt.npz".format(organ)), one_gt)
            visualize_annotated_slices(anno_slices, one_gt, organ, save_dir)
        else:
            print("======> Patient {} doesn't have annotation for {} <======".format(patient_id, organ))
    # exit()
# LTRC00408698_1 , LTRC00407195_1, LTRC00404734_2, LTRC00403705_2 # switched left and right lung annotations

def preprocess_patient_multi_slice_annotation(patient_id: str, annotator: int = 1):
    """
    Takes patient id, loads corresponding ct and gt. Saves annotated ct and gt slices for the patient as bzolz
    :param annotator:
    :param patient_id:
    :return:
    """
    print("Now processing {}".format(patient_id))
    save_dir = data_root_vol() / patient_path(patient_id)
    filepath = data_root_vol() / patient_id / "CT"
    ct, spacing = dicom.load_image_volume(filepath)
    gt_list, gt_name_list = load_gt_list_with_names(patient_id, ct.shape, data_root_vol(), annotator)
    ct = normalize_hu_values(ct)

    for name, gt in zip(gt_name_list, gt_list):
        nnz_idx = (np.max(gt, axis=(1, 2)) > 0)
        one_gt = gt[nnz_idx]
        anno_slices = ct[nnz_idx]
        organ = name_mapping(name)
        # print(anno_slices.shape, one_gt.shape, organ)
        if anno_slices.shape[0] !=0:
            np.savez_compressed(os.path.join(save_dir, "{}_ct.npz".format(organ)), anno_slices)
            np.savez_compressed(os.path.join(save_dir, "{}_gt.npz".format(organ)), one_gt)
            visualize_annotated_slices(anno_slices, one_gt, organ, save_dir)
        else:
            print("======> Patient {} doesn't have annotation for {} <======".format(patient_id, organ))

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



def get_all_valid_raw_data(path: Path, dataset_name='LTRC') -> List:
    patient_in_folder = [p for p in os.listdir(str(path))  if dataset_name in p] #if ".csv" not in p]
    # print("all patient", patient_in_folder)
    # print("len", len(patient_in_folder))

    patient_list = [x for x in patient_in_folder if x not in ignor_datasets]
    # print("after filter", patient_list)
    return patient_list


def get_all_preprocessed_patients(split: str) -> List:
    """

    :return: list of preprocessed patients
    """
    assert split is not None
    patients = set(os.listdir(str(preprocessed_data_root())))
    df = pd.read_csv(data_root_slices() / "LTRC_NLST_multi_organ_data.csv")
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

def move_patient_to_split_folder(patient_id: str, split_path: Path):
    patient_file_path = preprocessed_data_root() / patient_id
    split_path = split_path / patient_id
    # shutil.copy2(str(patient_file_path) , split_path)
    # print(patient_file_path, split_path)
    copy_tree(str(patient_file_path) , str(split_path))

def prepare_split_dataset(split: str):
    # get a list of patients in a certain split
    patient_list = get_all_preprocessed_patients(split)
    # Create the specific split directory
    split_path = experiment_data_folder_path(split)
    for patient in patient_list:
        print("moving {} to {}".format(patient, split_path))
        move_patient_to_split_folder(patient, split_path)

ignor_datasets = ["NLST200516", "LTRC103094_h", "NLST104707",
                  "NLST218499", "LTRC00403120_2",
                  # # scans that have been preprocessed already
                  # "LTRC00403351_0",
                  # "LTRC00406587_0",
                  # "LTRC0080369_4", "LTRC103094_2", "LTRC203881", "LTRC309379", "LTRC406656", "LTRC408440",
                  # "LTRC802171", "LTRC802688", "LTRC802790", "LTRC802954", "LTRC100934",
                  # "LTRC101971", "LTRC103094",
                  #
                  # # defected
                  # "LTRC100023", "LTRC100058", "LTRC100066", "LTRC401121", "LTRC100239",
                  # "LTRC100650",
                  ]

# LTRC309379 might have issues with oesophagus naming
# LTRC00403120_2 doesn't have segmentation available

# the two fully annotated NLST volumes would require different loading function as slices might be not ordered


class LTRCNLSTSinglePatient(Dataset):

    def __init__(self, split=None, patient_id=None, requested_organ_list=None,
                 dataset_file_path: Path = None, transform=None):

        if not isinstance(requested_organ_list, list):
            self.requested_organ_list = [requested_organ_list]
        else:
            self.requested_organ_list = requested_organ_list
        self.organ = requested_organ_list[0]  # handles one organ for now
        self.split = split
        self.patient = patient_id
        self.patient_path = dataset_file_path if dataset_file_path else \
            experiment_data_folder_path(self.split) / patient_id
        self.prepare_files()
        self.len = self.patient_vol.shape[0]
        self.transform = transform
        self.tt = -1  # task label
        self.td = -1  # disc label

        self.set_tt_td()

    def set_tt_td(self):
        "to set the task label and disc label for this dataset"
        # all_organs have the organs in the order i wanna train them, tt should not relate to
        # the true label of the organ cuz the training order might change
        self.tt = all_organs().index(self.requested_organ_list[0])
        self.td = self.tt + 1

    def prepare_files(self):
        try:
            patient_vol = np.load(os.path.join(self.patient_path, "{}_ct.npz".format(self.organ)))['arr_0']
            patient_gt = np.load(os.path.join(self.patient_path,  "{}_gt.npz".format(self.organ)))['arr_0']
        except:
            # in case one organ doesn't exist for a certain patient which is valid for
            # LTRC00402607_2 no trachea, and LTRC0031197_3 no spinal cord
            print("returning empty slice")
            patient_vol = np.zeros((1,512,512))
            patient_gt = np.zeros((1,512,512))

        # idx = np.max(patient_gt, axis=(1,2)) > 0
        # print("non empty", len(idx))
        # self.patient_vol = patient_vol[idx]
        # self.patient_gt  = patient_gt[idx]
        self.patient_vol = patient_vol
        self.patient_gt  = patient_gt

        # print(self.patient_vol.shape, self.patient_gt.shape)

    def __getitem__(self, index):
        x = self.patient_vol[index]
        y = self.patient_gt[index]
        # print("max x in dataset {} min x in dataset {}".format(np.max(x), np.min(x)))
        assert self.requested_organ_list is not None
        # If there is any transform method, apply it onto the image and gt
        if self.transform:
            x, y = self.transform(x, y)

        # print("tt {}  td {}".format(self.tt, self.td))
        return x, y, self.tt, self.td

    def __len__(self):
        return self.len


def get_ltrc_nlst_dataset(split: str, requested_organ_list: list, debug_mode: bool,
                          transforms=None, opt=None) -> List[Dataset]:
    datasets = []
    patient_list = get_all_preprocessed_patients(split)
    # print("patient list", patient_list)
    if debug_mode:
        patient_list = patient_list[:2]
    for patient in patient_list:
        d = LTRCNLSTSinglePatient(split=split, requested_organ_list=requested_organ_list,
                                  patient_id=patient, transform=transforms)
        datasets.append(d)

    return datasets
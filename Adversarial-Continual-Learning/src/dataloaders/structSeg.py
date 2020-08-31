import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import shutil


# left lung 1, right lung 2, heart 3, oesophagus 4?, trachea 5, spinal_cord 6

def data_root_structSeg():
    # return Path("/home/skhawy/thesis/structseg")
    return Path("/data/structseg_preprocessed")  #polyaxon

def all_organs() -> List[str]:
    # return ["l_lung", "r_lung", "heart", "oesophagus", "trachea", "spinal_cord"]
    # return ["l_lung", "r_lung", "heart", "oesophagus", "trachea", "spinal_cord"]
    return ['spinal_cord', 'r_lung', 'l_lung', 'heart', 'oesophagus', 'trachea']
    # return ["l_lung",] # 'r_lung']
    # return ["spinal_cord"]

def roi_order() -> List[str]:
    return ['background', "l_lung", "r_lung", "heart", "oesophagus", "trachea", "spinal_cord"]


def get_all_preprocessed_patients(split):
    assert split is not None
    csv_path = data_root_structSeg() / "structseg_split.csv"

    df = pd.read_csv(csv_path)
    patients = set(df.patient_id)  # for DGX
    subset_df = df.loc[df['split'] == split]
    patients_of_intrest = sorted(list(patients.intersection(set(subset_df.patient_id))))

    return patients_of_intrest




def get_multi_organ_dataset(split: str, debug_mode: bool, requested_organ_list=None,
                            transforms=None, opt=None) -> List[Dataset]:
    datasets = []
    patient_list = get_all_preprocessed_patients(split)
    if debug_mode:
        patient_list = patient_list[:2]
    # print("len patient list", len(patient_list))
    for patient in patient_list:
        d = StructSegSinglePatient(split=split, patient_id=patient, requested_organ_list=requested_organ_list,
                              transform=transforms, opt=opt)
        # print("=req organ in get multi organ<================", d.requested_organ_list)
        datasets.append(d)

    return datasets


def experiment_data_folder_path(split: str) -> Path:
    folder_path = data_root_structSeg() / split
    os.makedirs(str(folder_path), exist_ok=True)
    return folder_path


# class StructSegSinglePatient(Dataset):
#     def __init__(self, split=None, patient_id=None, requested_organ_list=None,
#                  dataset_file_path: Path = None, transform=None, opt=None):
#         # Just making sure its a list
#         if not isinstance(requested_organ_list, list):
#             self.requested_organ_list = [requested_organ_list]
#         else:
#             self.requested_organ_list = requested_organ_list
#         # print("init Requested organs", self.requested_organ_list)
#         self.split = split
#         self.patient = patient_id
#         self.patient_path = dataset_file_path if dataset_file_path else \
#             experiment_data_folder_path(self.split)
#         self.organs_names = ["l_lung", "r_lung", "heart", "oesophagus", "trachea", "spinal_cord"]
#         self.prepare_files()
#         self.len = self.patient_vol.shape[0]
#         self.transform = transform
#         self.input_dict = {}
#
#     def prepare_files(self):
#         patient_vol = np.load(os.path.join(self.patient_path, str(self.patient) + ".npz"))['arr_0']
#         patient_gt = np.load(os.path.join(self.patient_path, str(self.patient) + "_gt.npz"))['arr_0']
#
#         # Return one organ or a list of organs as requested
#         if len(self.requested_organ_list) == 1:  # return only one organ
#             organ_idx = self.organ_label_mapping(self.requested_organ_list[0])  # a list of 1 item
#             patient_gt[patient_gt != organ_idx] = 0
#         else:
#             zero_out_organs = list(set(self.organs_names).difference(self.requested_organ_list))
#             for organ in zero_out_organs:
#                 organ_idx = self.organ_label_mapping(organ)
#                 patient_gt[patient_gt == organ_idx] = 0
#
#         # remove slices that have no gt for the requested labels
#         # if max = 0, then >0 will evaluate to false
#         idx = np.max(patient_gt, axis=(1, 2)) > 0
#         self.patient_vol = patient_vol[idx]
#         self.patient_gt = patient_gt[idx]
#
#     def organ_label_mapping(self, organ):
#         # returns the corresponding label for a certain organ
#         return self.organs_names.index(organ) + 1
#
#     def __getitem__(self, index):
#         x = self.patient_vol[index]
#         y = self.patient_gt[index]
#
#         assert self.requested_organ_list is not None
#         # If there is any transform method, apply it onto the image and gt
#         if self.transform:
#             x = self.transform(x)
#             y = self.transform(y)
#
#         # Changing the output to fit other functions
#         return x, y, dict()
#
#     def __len__(self):
#         return self.len

class StructSegSinglePatient(Dataset):
    def __init__(self, split=None, patient_id=None, requested_organ_list=None,
                 dataset_file_path: Path = None, transform=None, opt=None):

        self.args = opt
        # Just making sure its a list
        if not isinstance(requested_organ_list, list):
            self.requested_organ_list = [requested_organ_list]
        else:
            self.requested_organ_list = requested_organ_list
        # print("init Requested organs", self.requested_organ_list)
        self.split = split
        self.patient = patient_id
        self.patient_path = dataset_file_path if dataset_file_path else \
            experiment_data_folder_path(self.split)
        self.organs_names = ["l_lung", "r_lung", "heart", "oesophagus", "trachea", "spinal_cord"]
        self.prepare_files()
        self.len = self.patient_vol.shape[0]
        self.transform = transform
        self.input_dict = {}

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
        patient_vol = np.load(os.path.join(self.patient_path, str(self.patient) + ".npz"))['arr_0']
        patient_gt = np.load(os.path.join(self.patient_path, str(self.patient) + "_gt.npz"))['arr_0']

        if self.args.vis_seg:
            # i check the scan outside to get the indices where we have all the organs
            # this is just for visualizing purposes
            print("Returning only slices where all organs exist")
            patient_vol = patient_vol[70:97, :, :]
            patient_gt = patient_gt[70:97, :, :]
            print("unique in patient scan {}".format(np.unique(patient_gt)))

        # Return one organ or a list of organs as requested
        if len(self.requested_organ_list) == 1:  # return only one organ
            organ_idx = self.organ_label_mapping(self.requested_organ_list[0])  # a list of 1 item
            # for 1st organ for ex tt = 0, td = 1
            patient_gt[patient_gt != organ_idx] = 0
        else:
            zero_out_organs = list(set(self.organs_names).difference(self.requested_organ_list))
            for organ in zero_out_organs:
                organ_idx = self.organ_label_mapping(organ)
                patient_gt[patient_gt == organ_idx] = 0

        # if self.mode != "val":
            # remove slices that have no gt for the requested labels
            # if max = 0, then >0 will evaluate to false
        #
        idx = np.max(patient_gt, axis=(1,2)) > 0
        # print("non zero spine in patient {} is {} out of {}".format(self.patient, np.sum(idx), patient_gt.shape[0]))
        self.patient_vol = patient_vol[idx]
        self.patient_gt  = patient_gt[idx]
        # self.patient_vol = patient_vol
        # self.patient_gt  = patient_gt
        # print("non zero idx size {}".format(len(idx)))

    def organ_label_mapping(self, organ):
        # returns the corresponding label for a certain organ
        return self.organs_names.index(organ) + 1

    def __getitem__(self, index):
        x = self.patient_vol[index]
        y = self.patient_gt[index]

        assert self.requested_organ_list is not None
        # If there is any transform method, apply it onto the image and gt
        if self.transform:
            x, y = self.transform(x, y)

        # Changing the output to fit other functions
        return x, y, self.tt, self.td

    def __len__(self):
        return self.len

def move_patient_to_split_folder(patient_id, split_path):
    patient_file_path = data_root_structSeg() / "preprocessed"
    shutil.copy2(str(patient_file_path / str(patient_id)) + ".npz", split_path)
    shutil.copy2(str(patient_file_path / str(patient_id)) + "_gt.npz", split_path)

def prepare_split_dataset(split: str):
    # get a list of patients in a certain split
    patient_list = get_all_preprocessed_patients(split)
    # Create the specific split directory
    split_path = experiment_data_folder_path(split)
    for patient in patient_list:
        move_patient_to_split_folder(patient, split_path)

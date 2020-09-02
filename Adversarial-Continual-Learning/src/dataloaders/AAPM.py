import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import List
import shutil
# import pydicom
from skimage.draw import polygon
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
# DATA PREPROCESSING
annotator = 4  # there are 4 annotator for multi organ task on LTRC and NLST
watermark = 7000 # the white square burned in on slices with annotations
ignor_datasets = []



def all_organs() -> List[str]:
    # return ['r_lung'] #, 'oesophagus', 'heart', 'spinal_cord', 'l_lung']
    return ['spinal_cord', 'r_lung', 'l_lung', 'heart', 'oesophagus']  # OrderA
    # return ['oesophagus', 'heart', 'l_lung', 'r_lung', 'spinal_cord'] # OrderB
    # return [ 'l_lung', 'r_lung', 'spinal_cord',  'heart', 'oesophagus' ]  # OrderC
    # return ['oesophagus']
    # return ['r_lung', 'l_lung', 'heart']  # just for prototyping
    # return ['heart']
    # return [  'l_lung', 'spinal_cord', 'r_lung' , 'oesophagus', 'heart',]
    # return ['heart',  'oesophagus', 'l_lung', 'spinal_cord', 'r_lung'  ]

def roi_order() -> List[str]:
    return ['background', 'spinal_cord', 'r_lung', 'l_lung', 'heart', 'oesophagus']

def project_root() -> Path:
    return Path(__file__).parent.parent


def input_data_root() -> Path:
    # Set the environment variable "ADP_CL_DATA_ROOT" to data location.  Defaults to "<project_repo>/data".
    # path_root = os.environ.get("ADP_CL_DATA_ROOT", project_root() / 'data')
    # return Path("/home/abel@local.tmvse.com/skhawy/Canon/Code/ADP_ContinualLearning/data/")
    # return Path("/ADP_ContinualLearning/data/")  # DGX
    return Path("/home/skhawy/thesis/CL_medical/data")

def multi_organ_data_root() -> Path:
    multio_path = input_data_root() / 'MultiOrgan'
    if not os.path.exists(str(multio_path)):
        os.mkdir(str(multio_path))
    return multio_path


def preprocessed_data_root() -> Path:
    preproces_multio_path = multi_organ_data_root() / 'preprocessed'
    if not os.path.exists(str(preproces_multio_path)):
        os.mkdir(str(preproces_multio_path))
    return preproces_multio_path




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


def get_all_valid_raw_data(path: Path) -> List:
    patient_in_folder = [p for p in os.listdir(str(path)) if ".csv" not in p]
    patient_list = [x for x in patient_in_folder if x not in ignor_datasets]

    return patient_list




def normalize_hu_values(vol, v_min=-1000, v_max=500) ->np.ndarray:
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
    shutil.copy2(str(patient_file_path/ patient_id) + ".npz", split_path)
    shutil.copy2(str(patient_file_path/ patient_id) + "_gt.npz", split_path)

def prepare_split_dataset(split: str, opt):
    # get a list of patients in a certain split
    patient_list = get_all_preprocessed_patients(split, opt)
    # Create the specific split directory
    split_path = experiment_data_folder_path(split)
    for patient in patient_list:
        move_patient_to_split_folder(patient, split_path)




class AAPMPreprocess:
    def __init__(self, data_root: Path=None):
        self.data_root = Path(data_root)
        self.patients_list = get_all_valid_raw_data(data_root)
        self.patients_list = [p for p in self.patients_list if "LCTSC" in p]
        print("n patients", len(self.patients_list))
        self.ROI_ORDER = ['SpinalCord', 'Lung_R', 'Lung_L', 'Heart', 'Esophagus']
        self.organs_names = ["spinal_cord", "r_lung", "l_lung", "heart", "oesophagus" ]

    def get_all_valid_raw_data(path: Path) -> List:
        patient_in_folder = [p for p in os.listdir(str(path)) if ".csv" not in p]
        patient_list = [x for x in patient_in_folder if x not in ignor_datasets]

        return patient_list

    def read_contour_structure(self, structure):
        """ reads the dicom label file and returns the contours"""
        contours = []
        for i in range(len(structure.ROIContourSequence)):
            contour = dict()
            contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
            contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber  # RefdROINumber
            contour['name'] = structure.StructureSetROISequence[i].ROIName
            assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
            contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
            contours.append(contour)
        colors = tuple(np.array([con['color'] for con in contours]) / 255.0)

        return contours, colors

    def create_label_map(self, contours, shape, slices):
        z = [np.around(s.ImagePositionPatient[2], 1) for s in slices]
        pos_r = slices[0].ImagePositionPatient[1]
        spacing_r = slices[0].PixelSpacing[1]
        pos_c = slices[0].ImagePositionPatient[0]
        spacing_c = slices[0].PixelSpacing[0]

        label_map = np.zeros(shape, dtype=np.float32)
        for con in contours:
            # +1 because 0 is reserved for background
            num = self.ROI_ORDER.index(con['name'].replace(" ", "_")) + 1
            for c in con['contours']:
                # triplets describing points of contour
                nodes = np.array(c).reshape((-1, 3))
                assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
                z_index = z.index(np.around(nodes[0, 2], 1))
                r = (nodes[:, 1] - pos_r) / spacing_r
                c = (nodes[:, 0] - pos_c) / spacing_c
                rr, cc = polygon(r, c)
                label_map[z_index, rr, cc] = num

        return label_map

    def load_scan(self, path):
        for subdir, dirs, files in os.walk(path):
            dcms = glob.glob(os.path.join(subdir, '*.dcm'))
            if len(dcms) == 1:
                # logging.info("Reading contours")
                structure = pydicom.read_file(dcms[0])
                contours, colors = self.read_contour_structure(structure)
            if len(dcms) > 1:  # elif
                slices = [pydicom.read_file(dcm) for dcm in dcms]
                slices.sort(key=lambda x: int(x.InstanceNumber))
                try:
                    slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
                except:
                    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

                for s in slices:
                    s.SliceThickness = slice_thickness
        images_stacked = np.stack([s.pixel_array for s in slices], axis=0)

        labels = self.create_label_map(contours, images_stacked.shape, slices)
        labels = labels.astype(np.int8)
        return slices, labels, colors

    def normalize_hu_values(self, vol, v_min=-1000, v_max=500) -> np.ndarray:
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
    def preprocess(self):
        for patient_id in tqdm(self.patients_list, desc="Preprocessing..."):
            patient_vol, gt_vol, colors = self.load_scan(self.data_root / patient_id)
            patient_vol = dicom.get_pixels_hu(patient_vol)
            patient_vol = self.normalize_hu_values(patient_vol)
            # print(np.unique(patient_vol), np.unique(gt_vol))
            # # make padding values consistent between volumes, at -1024
            # patient_vol = np.clip(patient_vol, a_min=-1024, a_max=np.max(patient_vol))
            save_npz_dir = patient_path(patient_id)
            np.savez_compressed(str(save_npz_dir / patient_id)  + ".npz", patient_vol)
            np.savez_compressed(str(save_npz_dir / patient_id)  + "_gt.npz", gt_vol)

def get_all_preprocessed_patients(split: str, opt=None, requested_organ_list=None) -> List:
    """

    :return: list of preprocessed patients
    """
    assert split is not None
    # a small change for DGX
    # patients = set(os.listdir(str(preprocessed_data_root())))
    # patients = set([p for p in list(patients) if "LCTSC" in p])
    csv_path = input_data_root() / "multi_organ_data.csv"

    df = pd.read_csv(csv_path)
    patients = set(df.patient_id) # for DGX
    subset_df = df.loc[df['split'] == split]
    patients_of_intrest = sorted(list(patients.intersection(set(subset_df.patient_id))))
    # print(patients)
    # print(csv_path,df)
    # print("patients of interests are", patients_of_intrest)
    # pick only the organ specific scans as per the sheet and just in case of training
    # if split == 'Training':
    #     organ_scans = df.loc[(df['Organ'] == requested_organ_list[0]) & (df['split'] == split)].patient_id
    #     patients_of_intrest = list(organ_scans)
    #     logger.info(patients_of_intrest)
    if opt.vis_seg:
        patients_of_intrest = ["LCTSC-Test-S2-104"] #["LCTSC-Test-S1-203"]  # for visualization purpose
        print("patients of interest {}".format(patients_of_intrest))
    return patients_of_intrest


class AAPMSinglePatient(Dataset):
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
        self.organs_names = ["spinal_cord", "r_lung", "l_lung", "heart", "oesophagus"]
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
        patient_vol = np.load(os.path.join(self.patient_path, self.patient + ".npz"))['arr_0']
        patient_gt = np.load(os.path.join(self.patient_path, self.patient + "_gt.npz"))['arr_0']

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

    def slices_with_gt(self, gt: np.ndarray) -> List:
        """
        :param gt:
        :return: list of indices for slices that have annotations
        """
        indicies = []
        for i in range(len(gt)):
            if np.max(gt[i, ...]) > 0:
                indicies.append(i)
        return indicies

    def organ_label_mapping(self, organ):
        # returns the corresponding label for a certain organ
        return self.organs_names.index(organ) + 1

    def __getitem__(self, index):
        x = self.patient_vol[index]
        y = self.patient_gt[index]

        assert self.requested_organ_list is not None
        # If there is any transform method, apply it onto the image and gt
        if self.transform:
            # x = self.transform(x)
            # y = self.transform(y)
            x, y = self.transform(x, y)

        # Changing the output to fit other functions
        return x, y, self.tt, self.td
        # self.input_dict['slice'] = x
        # self.input_dict['gt'] = y
        # self.input_dict['metadata'] = dict()
        # return self.input_dict

    def __len__(self):
        return self.len



def get_multi_organ_dataset(split: str, debug_mode: bool, requested_organ_list=None,
                            transforms=None, opt=None) -> List[Dataset]:
    datasets = []
    patient_list = get_all_preprocessed_patients(split, opt, requested_organ_list)
    if debug_mode:
        patient_list = patient_list[:2]
    # print("len patient list", len(patient_list))
    # print(patient_list)
    for patient in patient_list:
        d = AAPMSinglePatient(split=split, patient_id=patient, requested_organ_list=requested_organ_list,
                              transform=transforms, opt=opt)
        # print("=req organ in get multi organ<================", d.requested_organ_list)
        datasets.append(d)

    return datasets

def main():
    opt_obj = Options()
    opt = opt_obj.parse()
    opt_obj.print_options(opt)
    # aapm_processor = AAPMPreprocess(opt.data_root_aapm)
    # aapm_processor.preprocess()
    #
    # splits = Split.training_splits() + Split.test_splits()
    # print("Preparing slice datasets")
    # for split in splits:
    #     print("Split: {}".format(split))
    #     prepare_split_dataset(str(split), opt=opt)


    # transforms = Transformer(opt=opt, subset="train")
    # datasets = get_multi_organ_dataset(split='Training', requested_organ_list=['oesophagus'],
    #                                   opt=opt, transforms=transforms, debug_mode=True)
    # # print("final len of dataset", len(datasets))
    #
    # from torch.utils.data import DataLoader
    # # from torch.utils.data.dataloader import default_collate
    #
    # # def my_collate(batch):
    # #     # print(batch)
    # #     batch = list(filter(lambda data: data['gt'] is not None, batch))
    # #     # print(batch)
    # #     return default_collate(batch)
    #
    # cat_datasets = torch.utils.data.ConcatDataset(datasets)
    # print(len(cat_datasets))
    # loader = DataLoader(cat_datasets,
    #                       batch_size=4 ,
    #                       shuffle=True, )#collate_fn=my_collate)
    #
    # print("len of loader",len(loader))
    # for i, d in enumerate(loader):
    #     # print(i, d)
    #     input, target = d['slice'], d['gt']
    #     # print(input.size(), torch.unique(input))
    #     print(target.size(), torch.unique(target))
    #     # break
if __name__ == '__main__':
    main()

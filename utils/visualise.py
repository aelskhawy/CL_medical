import os
import sys
from typing import List

# import bcolz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.multi_organ_mix import normalize_hu_values, patient_path
from experiments.metrics import dice_coefficient
from models.squeezeUnet import SqueezeUnetMultiTask
from utils import paths, pytorch, plots
from pathlib import Path
from utils.pytorch import predict

def patient_prediction_path(patient_id: str) -> Path:
    patient_path = paths.evaluation_output_root() / patient_id
    if not os.path.exists(str(patient_path)):
        os.mkdir(str(patient_path))
    return patient_path


def visualise_one_patient(patient_id: str, model: torch.nn.Module, organ: str):
    """
    Visualise prediction per patient at a slingle slice
    :param organ:
    :param patient_id:
    :param model: trained pytorch master model
    """

    x = np.array(bcolz.open(rootdir="{0}/ct_slices".format(patient_path(patient_id))))
    y_target = np.array(bcolz.open(rootdir="{0}/gt_{1}".format(patient_path(patient_id), organ)))
    x = normalize_hu_values(x)
    print(x.shape)
    # x = process_item(x)
    x = np.moveaxis(x,3,1)
    print(x.shape)
    y_pred_list = predict(model, x, 2)
    print(len(y_pred_list))
    visualise_models_pred_results(patient_id, x, np.squeeze(y_target), np.squeeze(y_pred_list[1]), organ)


def find_worse_slice(diff: np.array) -> int:
    """

    :param diff:
    :return:
    """
    max = 0
    index = 0
    for i, d in enumerate(diff):
        if np.mean(d) > max:
            index = i
            max = np.mean(d)
    return index


def find_best_slice(diff: np.array) -> int:
    """

    :param diff:
    :return:
    """
    min = 1000000
    index = 0
    for i, d in enumerate(diff):
        if np.mean(d) < min:
            min = np.mean(d)
            index = i
    return index


# def visualise_models_pred_results(patient: str, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray, organ: str):
#     """
#
#     :param patient:
#     :param y_pred:
#     :param y:
#     :param x:
#     :return:
#     """
#
#     width_axis = 2
#     height_axis = 3
#
#     diff = np.subtract(y, y_pred)
#     y_pred = (y_pred > 0.5)
#     slice_num = find_best_slice(diff)
#     rows = 1
#     columns = 4
#     fig = plt.figure()
#     a = fig.add_subplot(rows, columns, 1)
#     plt.imshow(x[slice_num].reshape((x.shape[width_axis], x.shape[height_axis])), cmap='gray')
#     plt.axis('off')
#     a.set_title(patient)
#     b = fig.add_subplot(rows, columns, 2)
#     plt.imshow(y[slice_num].reshape((x.shape[width_axis], x.shape[height_axis])), cmap='gray')
#     gt_title = "GT_{0}".format(organ)
#     b.set_title(gt_title)
#     plt.axis('off')
#     c = fig.add_subplot(rows, columns, 3)
#     plt.imshow(y_pred[slice_num].reshape((x.shape[width_axis], x.shape[height_axis])), cmap='gray')
#     dice = dice_coefficient(y, y_pred)
#     pred_dice = "Pred. Dice:{0:.2f}".format(dice)
#     c.set_title(pred_dice)
#     plt.axis('off')
#     d = fig.add_subplot(rows, columns, 4)
#     plt.imshow(diff[slice_num].reshape((x.shape[width_axis], x.shape[height_axis])), cmap='gray')
#     d.set_title("Diff")
#     plt.axis('off')
#     paths_name = "{0}/images/{1}_{2}_{3}.png".format(paths.output_data_root(), patient, str(slice_num), organ)
#     plt.savefig(paths_name)
#     plt.show()


def plot_volume_datasets(datasets: List[Dataset], output_folder: Path):
    volume_iter = tqdm(datasets, file=sys.stdout)
    for dataset in volume_iter:
        volume, gt, metadata = pytorch.dataset_to_volume(dataset=dataset)
        volume_iter.set_description(f'Visualising {metadata[0]}')

        output_file = output_folder / f'{metadata[0]}.png'
        plots.volume_mips(volumes={'input': volume, 'GT': gt}, output_file=output_file,
                          image_ranges=[None, (0, 1)])


if __name__ == '__main__':
    # Example
    initial_model_path = "/home/anli@local.tmvse.com/projects/continual_learning/output/models/l_lung,r_lung,oesophagus,spinal_cord,trachea,ideal/naive_multi_head_fullyTune/r_lung/multioutput_dice_lr_1.0e-04_wd_1.0e-03_Adam_batch_4/model.pt"
    model = SqueezeUnetMultiTask(task_list=['task_1_l_lung', 'l_lung,r_lung'], all_task=True)

    model.load_state_dict(torch.load(initial_model_path))#, strict=False)

    visualise_one_patient('LTRC00403386_2', model, 'r_lung')


def visualise_models_pred_results(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray,
                                  organ: str, path=None, iter=0):
    """

    :param patient:
    :param y_pred:
    :param y:
    :param x:
    :return:
    """
    if not isinstance(x, np.ndarray):
        x = x.squeeze(1).cpu().detach().numpy()
        y = y.squeeze(1).cpu().detach().numpy().astype(np.uint8)
        y_pred = y_pred.squeeze(1).cpu().detach().numpy().astype(np.uint8)

    print("*" * 80)
    colors = [(255, 255, 0), (0, 255, 0), (255, 132, 0), (0, 247, 255), (128, 0, 0)]
    true_label = np.zeros((x.shape[0], 256, 256, 3))
    pred_label = np.zeros((x.shape[0], 256, 256, 3))
    # to make sure label color stays consistent
    unique_true = list(np.unique(y))[1:]
    # print("unique true {}".format(unique_true))
    for label in unique_true:
        true_label[y == label] = colors[label - 1]

    unique_pred = list(np.unique(y_pred))[1:]
    # print("unique pred {}".format(unique_pred))
    for label in unique_pred:
        pred_label[y_pred == label] = colors[label - 1]

    # print(" final unique values in true label {}".format(np.unique(true_label)))
    # print(" final unique values in pred label {}".format(np.unique(pred_label)))
    nrows = x.shape[0] if x.shape[0] > 1 else 2
    fig, ax = plt.subplots(nrows, 3, figsize=[20, 50])
    for slice_num in range(x.shape[0]):
        ax[slice_num, 0].set_title('x+gt')
        ax[slice_num, 0].imshow(x[slice_num], cmap='gray', interpolation='none')
        ax[slice_num, 0].imshow(true_label[slice_num] / 255, alpha=0.5, interpolation='none', )
        # ax[slice_num, 0].contour(y[slice_num], alpha = 0.8, colors=["C{}".format(int(i)) for i in np.unique(y)[1:]])
        # ax[slice_num, 0].contour(y[slice_num])
        ax[slice_num, 0].axis('off')

        ax[slice_num, 1].set_title('x+pred')
        ax[slice_num, 1].imshow(x[slice_num], cmap='gray', interpolation='none')
        ax[slice_num, 1].imshow(pred_label[slice_num] / 255, alpha=0.5, interpolation='none', )
        # ax[slice_num, 1].contour(y_pred[slice_num].astype(np.uint8), alpha=0.8, interpolation='none', cmap='Greens')
        # ax[slice_num, 1].contour(y_pred[slice_num])
        ax[slice_num, 1].axis('off')

        ax[slice_num, 2].set_title('original slice')
        ax[slice_num, 2].imshow(x[slice_num], cmap='gray')
        ax[slice_num, 2].axis('off')

    # path_name = os.path.join(path, "visuals")
    paths_name = path + "/{0}_model_{1}.png".format(iter, organ)
    plt.savefig(paths_name)
    # plt.show()
    plt.close(fig)
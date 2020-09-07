# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from copy import deepcopy
import pickle
import time
import uuid
from subprocess import call
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

########################################################################################################################

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])


def report_tr(res, e, sbatch, clock0, clock1, path):
    # Training performance
    message ='| Epoch {:3d}, | Train losses={:.3f} | Task: loss={:.3f}, dice={}, surfd={}, | ' \
             'Disc: loss={:.3f}, acc={:5.1f}%,' \
             ' ''Diff loss:{:.3f} |'.format(e + 1,
                                            res['loss_tot'],
                                            res['loss_t'],
                                            res['dice'],
                                            res['surfd'],
                                            res['loss_a'],
                                            res['acc_d'],
                                            res['loss_d'])
    # print(message, end='')
    # print(message, file=open(os.path.join(path, "log.log"), "a"))
    logger.info(message)

def report_val(res, path):
    # Validation performance
    message = ' Valid losses={:.3f} | Task: loss={:.6f}, dice={}, surfd={} | ' \
              'Disc: loss={:.3f}, acc={:5.2f}%,' \
              ' Diff loss={:.3f} |'.format(res['loss_tot'],
                                           res['loss_t'],
                                           res['dice'],
                                           res['surfd'],
                                           res['loss_a'],
                                           res['acc_d'],
                                           res['loss_d'])

    logger.info(message)
    # print(message, end='')
    # print(message, file=open(os.path.join(path, "log.log"), "a"))


########################################################################################################################
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
        y = y.squeeze(1).cpu().detach().numpy()
        y_pred = y_pred.squeeze(1).cpu().detach().numpy()

    y_pred = (y_pred > 0.5)
    # if np.sum(y, axis=)
    empty_gt = np.max(y, axis=(1, 2)) < 1
    # print(empty_gt)
    if np.sum(empty_gt) != 0:
        print("empty gt at iter {}".format(iter))
    nrows = x.shape[0] if x.shape[0] > 1 else 2
    fig, ax = plt.subplots(nrows, 3, figsize=[20, 50])
    for slice_num in range(x.shape[0]):
        ax[slice_num, 0].set_title('x+gt')
        ax[slice_num, 0].imshow(x[slice_num], cmap='gray')
        ax[slice_num, 0].imshow(y[slice_num], alpha = 0.8)
        ax[slice_num, 0].contour(y[slice_num])
        ax[slice_num, 0].axis('off')

        ax[slice_num, 1].set_title('x+pred')
        ax[slice_num, 1].imshow(x[slice_num], cmap='gray')
        ax[slice_num, 1].imshow(y_pred[slice_num], alpha=0.8)
        ax[slice_num, 1].contour(y_pred[slice_num])
        ax[slice_num, 1].axis('off')

        ax[slice_num, 2].set_title('original slice')
        ax[slice_num, 2].imshow(x[slice_num], cmap='gray')
        ax[slice_num, 2].axis('off')

    path_name = os.path.join(path, "visuals")
    paths_name = path_name + "/{0}_{1}.png".format(iter, organ)
    plt.savefig(paths_name)
    # plt.show()
    plt.close(fig)


def visualize_seg_sequentially(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray,
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

    y = y.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)

    y_pred[y_pred  > 5] = 0
    # print("*"*80)
    colors = [(255, 255, 0), (0, 255, 0), (255, 132, 0), (0, 247, 255), (128, 0, 0)]
    true_label = np.zeros((x.shape[0], 256, 256, 3))
    pred_label = np.zeros((x.shape[0], 256, 256, 3))
    # to make sure label color stays consistent
    unique_true = list(np.unique(y))[1:]
    # print("unique true {}".format(unique_true))
    for label in unique_true:
        true_label[y == label] = colors[label-1]

    unique_pred = list(np.unique(y_pred))[1:]
    # print("unique pred {}".format(unique_pred))
    for label in unique_pred:
        pred_label[y_pred==label] = colors[label-1]

    # print(" final unique values in true label {}".format(np.unique(true_label)))
    # print(" final unique values in pred label {}".format(np.unique(pred_label)))
    nrows = x.shape[0] if x.shape[0] > 1 else 2
    fig, ax = plt.subplots(nrows, 3, figsize=[20, 50])
    for slice_num in range(x.shape[0]):
        ax[slice_num, 0].set_title('x+gt')
        ax[slice_num, 0].imshow(x[slice_num], cmap='gray', interpolation='none')
        ax[slice_num, 0].imshow(true_label[slice_num]/255, alpha = 0.5, interpolation='none',)
        # ax[slice_num, 0].contour(y[slice_num], alpha = 0.8, colors=["C{}".format(int(i)) for i in np.unique(y)[1:]])
        # ax[slice_num, 0].contour(y[slice_num])
        ax[slice_num, 0].axis('off')

        ax[slice_num, 1].set_title('x+pred')
        ax[slice_num, 1].imshow(x[slice_num], cmap='gray', interpolation='none')
        ax[slice_num, 1].imshow(pred_label[slice_num]/255, alpha = 0.5, interpolation='none',)
        # ax[slice_num, 1].contour(y_pred[slice_num].astype(np.uint8), alpha=0.8, interpolation='none', cmap='Greens')
        # ax[slice_num, 1].contour(y_pred[slice_num])
        ax[slice_num, 1].axis('off')

        ax[slice_num, 2].set_title('original slice')
        ax[slice_num, 2].imshow(x[slice_num], cmap='gray')
        ax[slice_num, 2].axis('off')

    path_name = os.path.join(path, "visuals")
    paths_name = path_name + "/{0}_model_{1}.png".format(iter, organ)
    plt.savefig(paths_name)
    # plt.show()
    plt.close(fig)
########################################################################################################################
def get_model(model):
    return deepcopy(model.state_dict())

########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################

def save_print_log(taskcla, acc, lss, output_path):

    print('*'*100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t',end=',')
        for j in range(acc.shape[1]):
            print('{:5.4f}% '.format(acc[i,j]),end=',')
        print()
    print ('ACC: {:5.4f}%'.format((np.mean(acc[acc.shape[0]-1,:]))))
    print()

    print ('BWD Transfer = ')

    print ()
    print ("Diagonal R_ii")
    for i in range(acc.shape[0]):
        print('\t',end='')
        print('{:5.2f}% '.format(np.diag(acc)[i]), end=',')


    print()
    print ("Last row")
    for i in range(acc.shape[0]):
        print('\t', end=',')
        print('{:5.2f}% '.format(acc[-1][i]), end=',')

    print()
    # BWT calculated based on GEM paper (https://arxiv.org/abs/1706.08840)
    gem_bwt = sum(acc[-1]-np.diag(acc))/ (len(acc[-1])-1)
    # BWT calculated based on our UCB paper (https://openreview.net/pdf?id=HklUCCVKDB)
    ucb_bwt = (acc[-1] - np.diag(acc)).mean()
    print ('BWT: {:5.2f}%'.format(gem_bwt))
    # print ('BWT (UCB paper): {:5.2f}%'.format(ucb_bwt))

    print('*'*100)
    print('Done!')


    logs = {}
    # save results
    logs['name'] = output_path
    logs['taskcla'] = taskcla
    logs['acc'] = acc
    logs['loss'] = lss
    logs['gem_bwt'] = gem_bwt
    logs['ucb_bwt'] = ucb_bwt
    logs['rii'] = np.diag(acc)
    logs['rij'] = acc[-1]

    # pickle
    with open(os.path.join(output_path, 'logs.p'), 'wb') as output:
        pickle.dump(logs, output)

    print ("Log file saved in ", os.path.join(output_path, 'logs.p'))


def print_log_acc_bwt(taskcla, acc, lss, output_path, run_id):

    print('*'*100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t',end=',')
        for j in range(acc.shape[1]):
            print('{:5.4f}% '.format(acc[i,j]),end=',')
        print()

    avg_acc = np.mean(acc[acc.shape[0]-1,:])
    print ('ACC: {:5.4f}%'.format(avg_acc))
    print()
    print()
    # BWT calculated based on GEM paper (https://arxiv.org/abs/1706.08840)
    gem_bwt = sum(acc[-1]-np.diag(acc))/ (len(acc[-1])-1)
    # BWT calculated based on UCB paper (https://arxiv.org/abs/1906.02425)
    ucb_bwt = (acc[-1] - np.diag(acc)).mean()
    print ('BWT: {:5.2f}%'.format(gem_bwt))
    # print ('BWT (UCB paper): {:5.2f}%'.format(ucb_bwt))

    print('*'*100)
    print('Done!')


    logs = {}
    # save results
    logs['name'] = output_path
    logs['taskcla'] = taskcla
    logs['acc'] = acc
    logs['loss'] = lss
    logs['gem_bwt'] = gem_bwt
    logs['ucb_bwt'] = ucb_bwt
    logs['rii'] = np.diag(acc)
    logs['rij'] = acc[-1]

    # pickle
    path = os.path.join(output_path, 'logs_run_id_{}.p'.format(run_id))
    with open(path, 'wb') as output:
        pickle.dump(logs, output)

    print ("Log file saved in ", path)
    return avg_acc, gem_bwt


def print_running_acc_bwt(acc, task_num):
    print()
    acc = acc[:task_num+1,:task_num+1]
    avg_acc = np.mean(acc[acc.shape[0] - 1, :])
    gem_bwt = sum(acc[-1] - np.diag(acc)) / (len(acc[-1]) - 1)
    print('ACC: {:5.4f}%  || BWT: {:5.2f}% '.format(avg_acc, gem_bwt))
    print()


def make_dirs_polyaxon():
    from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

    output_path = get_outputs_path()
    CHECKPOINTS = os.path.join(output_path)
    POLYAXON = True
    print("CHECKPOINTS dir is", CHECKPOINTS)
    return CHECKPOINTS


def make_directories(args):
    # uid = uuid.uuid4().hex
    if args.checkpoint is None:
        os.mkdir('checkpoints')
        args.checkpoint = os.path.join('./checkpoints/',args.name)
        os.makedirs(args.checkpoint, exist_ok=True)
    else:
        if not os.path.exists(args.checkpoint):
            os.mkdir(args.checkpoint)
        args.checkpoint = os.path.join(args.checkpoint, args.name)
        os.makedirs(args.checkpoint, exist_ok=True)

    os.makedirs(os.path.join(args.checkpoint, "visuals"), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint, "scores"), exist_ok=True)



def some_sanity_checks(args):
    # Making sure the chosen experiment matches with the number of tasks performed in the paper:
    datasets_tasks = {}
    datasets_tasks['mnist5']=[5]
    datasets_tasks['pmnist']=[10,20,30,40]
    datasets_tasks['cifar100']=[20]
    datasets_tasks['miniimagenet']=[20]
    datasets_tasks['multidatasets']=[5]


    if not args.ntasks in datasets_tasks[args.experiment]:
        raise Exception("Chosen number of tasks ({}) does not match with {} experiment".format(args.ntasks,args.experiment))

    # Making sure if memory usage is happenning:
    if args.use_memory == 'yes' and not args.samples > 0:
        raise Exception("Flags required to use memory: --use_memory yes --samples n where n>0")

    if args.use_memory == 'no' and args.samples > 0:
        raise Exception("Flags required to use memory: --use_memory yes --samples n where n>0")



def save_code(args):
    cwd = os.getcwd()
    des = os.path.join(args.checkpoint, 'code') + '/'
    if not os.path.exists(des):
        os.mkdir(des)

    def get_folder(folder):
        return os.path.join(cwd,folder)

    folders = [get_folder(item) for item in ['dataloaders', 'networks', 'configs', 'main.py', 'acl.py', 'utils.py']]

    for folder in folders:
        call('cp -rf {} {}'.format(folder, des),shell=True)


def print_time():
    from datetime import datetime

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Job finished at =", dt_string)


def proposed_cl_metric(model_scores, ideal_scores):
    omega_base, omega_new, omega_all = 0, 0, 0
    for task_id in range(1, len(model_scores)):
        current_model_scores = model_scores[task_id]
        # print(current_model_scores, ideal_scores)
        omega_base += current_model_scores[0] / ideal_scores[0]
        omega_new += current_model_scores[task_id] / ideal_scores[task_id]
        omega_all += np.mean(current_model_scores) / np.mean(ideal_scores[:task_id + 1])
        # print(np.mean(current_model_scores), np.mean(ideal_scores[:task_id + 1]))

    number_of_tasks = len(model_scores)
    denominator = number_of_tasks - 1
    omega_base /= denominator
    omega_new /= denominator
    omega_all /= denominator

    return omega_base, omega_new, omega_all
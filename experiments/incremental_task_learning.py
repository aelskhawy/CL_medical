from typing import Tuple, List, Dict, Any

from torch.optim import Adam
from torch.utils.data import Dataset

from continual_learning.model_base import CLModel
from continual_learning.naive import NaiveModelMultiHead, LwFMultiHead
from datasets import multi_organ_mix
from datasets import AAPM, structSeg
from datasets.all_data import DataQuery
from datasets.transforms import Transformer
from experiments import metrics, evaluation
from models.squeezeUnet import SqueezeUnetMultiTask
from models.unet_multihead import UNetMultiHead
from nn.losses import to_multioutput, dice_loss_with_missing_values
from utils.data import Split
from utils.options import Options


def continual_learning_tasks(options: Options) -> List[DataQuery]:
    if options.dataset == "AAPM":
        tasks = aapm_data_queries(options=options)
    else: # structseg
        tasks = structseg_data_queries(options=options) #multi_organ_data_queries()
    return tasks

# TODO: Return a proper data class to hold parameters, not just a dictionary!
def training_parameters(options: Options) -> Dict[str, Any]:
    num_epochs = 1 if options.debug_mode else options.num_epochs

    return {
        'optimiser': Adam,
        'learning_rate': options.lr,
        'weight_decay': options.weight_decay,
        'batch_size': options.batch_size,
        'num_epochs': num_epochs,
        'loss': to_multioutput(dice_loss_with_missing_values),
        'num_workers': options.num_workers
    }


def continual_learning_model(options: Options) -> CLModel:
    if options.dataset == 'AAPM' or options.dataset =="structseg":
        if options.replay_mode == "LwF":
            model = UNetMultiHead(in_channels=1, init_features=options.nfc,
                              task_list=['background'])
        else:
            #ideal case
            model = UNetMultiHead(in_channels=1, init_features=options.nfc,
                                  task_list= ['background'] + AAPM.all_organs())

        model.print_network()
        cl_model = LwFMultiHead(initial_model=model, options=options)
    else:
        model = SqueezeUnetMultiTask(channels=1, nclass=1, task_list=[])
        cl_model = NaiveModelMultiHead(initial_model=model, options=options)


    return cl_model


# TODO: Return a proper data class to hold parameters, not just a dictionary!
def evaluation_parameters(options: Options) -> Dict[str, Any]:
    return {
        'evaluation_fn': metrics.masked_dice_coefficient if options.mask_dataset else metrics.dice_coefficient,
        'evaluation_mode': evaluation.EvaluationMode.PerVolume,
        'prediction_threshold': 0.5
    }

def aapm_data_queries(options=None) -> List[DataQuery]:
    if options.replay_mode == "ideal":
        return [DataQuery(tasks=AAPM.all_organs())]
    else:
        return [DataQuery(tasks=organ) for organ in AAPM.all_organs()]




def structseg_data_queries(options=None) -> List[DataQuery]:
    # logger.info("Training Order is {}".format(structSeg.all_organs()))
    if options.replay_mode == "ideal":
        return [DataQuery(tasks=structSeg.all_organs())]
    else:
        return [DataQuery(tasks=organ) for organ in structSeg.all_organs()]


def multi_organ_data_queries() -> List[DataQuery]:
    return [DataQuery(tasks=organ) for organ in multi_organ_mix.all_organs()]


def multi_organ_data(debug_mode=False, ideal: bool = False) -> List[Tuple[str, Dict[Split, Dataset]]]:
    """

    :param debug_mode:
    :return:
    """
    tasks = list()
    cumulative_organ_list = []
    for organ in multi_organ_mix.all_organs():
        cumulative_organ_list.append(organ)
        if ideal:
            data_splits = data_per_split(debug_mode, present_organs=[organ], organ_list=cumulative_organ_list)
            task_name = "ideal_{0}".format(cumulative_organ_list)
        else:
            data_splits = data_per_split(debug_mode, present_organs=cumulative_organ_list,
                                         organ_list=cumulative_organ_list)
            task_name = organ
        tasks.append((task_name, data_splits))

    return tasks


def data_per_split(debug_mode, present_organs: list, organ_list: list):
    """
    Training with all task id the upper bound (ideal)
    :param debug_mode:
    :return:
    """
    splits = Split.training_splits() + Split.test_splits()
    data_splits = dict()
    for split in splits:
        data_splits[split] = multi_organ_mix.get_multi_organ_dataset(split=str(split), organ_list=organ_list,
                                                                     present_organs=present_organs,
                                                                     debug_mode=debug_mode)

    return data_splits


def aapm_multi_organ_data(debug_mode=False, transforms=None, opt=None) -> List[
    Tuple[str, Dict[Split, Dataset]]]:
    tasks = list()
    splits = Split.training_splits() + Split.test_splits()
    data_splits = dict()
    # TODO: handle the case where you need more than one organ in the scan or u need em all
    for organ in AAPM.all_organs():
        data_splits = dict()
        for split in splits:
            data_splits[split] = AAPM.get_multi_organ_dataset(split=str(split),
                                                              requested_organ_list=organ,
                                                              opt=opt, transforms=transforms,
                                                              debug_mode=debug_mode)
        tasks.append((organ, data_splits))

    return tasks
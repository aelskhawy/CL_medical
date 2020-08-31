import enum
import copy
import logging
from typing import List, Union, Dict

from torch.utils.data import Dataset
from dataloaders import AAPM, LTRC_NLST, structSeg
from dataloaders.transforms import Transformer

logger = logging.getLogger(__name__)

class Split(enum.Enum):
    Training = 'Training'
    EarlyStopping = 'EarlyStopping'
    DevelopmentTest = 'DevelopmentTest'
    FinalEvaluation = 'FinalEvaluation'

    def __str__(self):
        return str(self.value)

    @staticmethod
    def training_splits() -> List['Split']:
        return [Split.Training, Split.EarlyStopping]

    @staticmethod
    def test_splits() -> List['Split']:
        return [Split.DevelopmentTest]

    @staticmethod
    def evaluation_splits() -> List['Split']:
        logger.warning(f'Only use the {Split.FinalEvaluation} split if you are done with development!')
        return [Split.FinalEvaluation]


class DataQuery:
    def __init__(self, tasks: Union[str, List[str]], gt_tasks: List[str] = None,
                 domains: Dict[str, Union[str, List[str], int, List[int]]] = None):
        self.tasks = [tasks] if not isinstance(tasks, list) else tasks
        self.domains = domains if domains is not None else dict()
        self.gt_tasks = gt_tasks

    def __str__(self):
        task_str = '+'.join(self.gt_tasks) if self.gt_tasks else '+'.join(self.tasks)
        domain_strings = list()
        for key, values in self.domains.items():
            if isinstance(values, list):
                value_str = ','.join(str(value) for value in values)
            elif isinstance(values, int):
                value_str = f'{key}={values}'
            else:
                value_str = str(values)
            domain_strings.append(value_str)
        result = task_str
        if len(domain_strings) > 0:
            domain_str = ','.join(domain_strings)
            result = f'{task_str}_{domain_str}'
        return result


def get_data(query: DataQuery, split: Split, debug_mode: bool = False, options=None) -> List[Dataset]:
    multi_organ_mix_tasks = AAPM.all_organs()
    # print(multi_organ_mix_tasks, query.tasks)
    # if all(task in multi_organ_mix_tasks for task in query.tasks):
    transforms = Transformer(opt=options, subset=str(split))
    if options.dataset == 'AAPM':
        datasets = AAPM.get_multi_organ_dataset(split=str(split), debug_mode=options.debug_mode,
                                                requested_organ_list=query.tasks,
                                                transforms=transforms, opt=options)
    else:  # structseg
        datasets = structSeg.get_multi_organ_dataset(split=str(split), debug_mode=options.debug_mode,
                                                requested_organ_list=query.tasks,
                                                transforms=transforms, opt=options)
    if debug_mode:
        datasets = [datasets[0]]
        
    return datasets


def aapm_data_queries(options=None) -> List[DataQuery]:
    logger.info("Training Order is {}".format(AAPM.all_organs()))
    if options.replay_mode == "ideal":
        return [DataQuery(tasks=AAPM.all_organs())]
    else:
        return [DataQuery(tasks=organ) for organ in AAPM.all_organs()]


def structseg_data_queries(options=None) -> List[DataQuery]:
    logger.info("Training Order is {}".format(structSeg.all_organs()))
    if options.replay_mode == "ideal":
        return [DataQuery(tasks=structSeg.all_organs())]
    else:
        return [DataQuery(tasks=organ) for organ in structSeg.all_organs()]


def ltrc_nlst_data_queries() -> List[DataQuery]:
    logger.info("Training Order is {}".format(LTRC_NLST.all_organs()))
    return [DataQuery(tasks=organ) for organ in LTRC_NLST.all_organs()]



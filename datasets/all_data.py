import copy
import logging
from typing import List, Union, Dict

from torch.utils.data import Dataset

# from datasets import lung, multi_organ_mix
from datasets import multi_organ_mix
from utils.data import Split
from utils.options import  Options
from datasets import AAPM
from datasets.transforms import Transformer

logger = logging.getLogger(__name__)


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


def get_data(query: DataQuery, split: Split, debug_mode: bool = False, options: Options=None) -> List[Dataset]:
    multi_organ_mix_tasks = multi_organ_mix.all_organs() if not options.dataset =="AAPM" \
                            else AAPM.all_organs()
    lung_tasks = ['lung', 'lung_healthy', 'lung_pathological']

    if all(task in multi_organ_mix_tasks for task in query.tasks):
        if not options.dataset =="AAPM":
            datasets = multi_organ_mix.get_multi_organ_dataset(split=str(split), organ_list=query.tasks,
                                                               present_organs=query.gt_tasks,
                                                               debug_mode=debug_mode)
        else:
            transforms = Transformer(opt=options)
            datasets = AAPM.get_multi_organ_dataset(split=str(split), debug_mode=options.debug_mode,
                                                    requested_organ_list=query.tasks,
                                                    transforms=transforms, opt=options)

    elif all(task in lung_tasks for task in query.tasks):
        filters = copy.deepcopy(query.domains)

        healthy_filter = list()
        if 'lung_healthy' in query.tasks:
            healthy_filter.append(1)
        if 'lung_pathological' in query.tasks:
            healthy_filter.append(0)
        if len(healthy_filter) > 0:
            filters['Healthy'] = healthy_filter

        datasets = lung.get_datasets(splits=split, filters=filters, normalised=True)
    else:
        raise ValueError(f'Invalid combination of tasks: {query.tasks}.  All specified tasks must either be '
                         f'in {multi_organ_mix_tasks} or {lung_tasks}')

    if debug_mode:
        datasets = [datasets[0]]

    return datasets

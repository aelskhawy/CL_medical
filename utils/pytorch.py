import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Any, Callable, Union

import numpy as np
# import pynvml
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from utils import stats

logger = logging.getLogger(__name__)


class MaskedDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, output_indices: Union[int, List[int]],
                 num_outputs: int = 1):
        """Converts a Dataset to a version with some GT targets "masked".
        If the Dataset outputs a single GT target, the output targets are padded with masked copies of the
        original GT target to produce num_outputs or max(output_indices)+1 outputs, whichever is bigger.
        If the Dataset outputs multiple GT targets, the new Dataset will output the same number of GT targets
        with valid GT at each index specified by output_indices.
        :param dataset: A pytorch Dataset
        :param output_indices: The indices of the output GT targets that provide valid, unmasked GT.
        :param num_outputs: The total number of output GT targets in the Dataset.  This only affects the case
        where the input Dataset only provides a single GT target.
        """
        self.dataset = dataset
        self.output_indices = [output_indices] if not isinstance(output_indices, list) else output_indices
        self.num_outputs = max(num_outputs, max(self.output_indices) + 1)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def to_mask(array: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(array, torch.Tensor):
            return -torch.ones_like(array)
        if isinstance(array, np.ndarray):
            return -np.ones_like(array)
        raise ValueError(f'Unknown array type: {type(array)}')

    def __getitem__(self, index):
        input, gt, metadata = self.dataset.__getitem__(index)

        if not isinstance(gt, list) and not isinstance(gt, tuple):
            # Make sure the gt represents a list of GT targets (even if there is only one)
            gt = [gt]

        current_number_of_outputs = len(gt)
        if current_number_of_outputs == 1:
            # Map single output to specific index of outputs
            if len(self.output_indices) > 1:
                raise ValueError(
                    f'Must only specify one output index when adding masks to a dataset with a single output')
            unmasked_gt = gt[0]
            masked_gt = self.to_mask(unmasked_gt)
            new_gt = [masked_gt] * self.num_outputs
            index = self.output_indices[0]
            new_gt[index] = unmasked_gt
        else:
            # map multiple outputs to multiple masked outputs
            new_gt = [self.to_mask(unmasked_gt) for unmasked_gt in gt]

            for index in self.output_indices:
                if index >= len(gt):
                    raise ValueError(
                        f'valid_indices specifies an invalid index ({index}) for the number of GT outputs '
                        f'({len(gt)})')
                new_gt[index] = gt[index]

        return input, new_gt, metadata


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, input_folder: Path, gt_folder: Path, label: str,
                 input_transform: Callable[[np.ndarray], np.ndarray] = None,
                 gt_transform: Callable[[np.ndarray], np.ndarray] = None):
        self.input_folder = input_folder
        self.gt_folder = gt_folder
        self.label = label
        self.file_extension = '.npy'
        self.samples = list()

        self.input_transform = input_transform
        self.gt_transform = gt_transform

        self._init_dataset()

    def _init_dataset(self):
        logger.debug(
            f'Creating dataset for files with extension {self.file_extension} '
            f'from input folder: {self.input_folder} and GT folder: {self.gt_folder}')
        input_files = sorted(self.input_folder.glob(f'*{self.file_extension}'), key=lambda x: int(x.stem))
        gt_files = sorted(self.gt_folder.glob(f'*{self.file_extension}'), key=lambda x: int(x.stem))

        num_input = len(input_files)
        num_gt = len(gt_files)

        assert num_input == num_gt, f'Must have the same number of files in input ({self.input_folder}) ' \
                                    f'and GT ({self.gt_folder}) folders'
        assert num_input > 0 and num_gt > 0, f'Cannot create a dataset with empty input or GT folders.  ' \
                                             f'Found {num_input} files in {self.input_folder} ' \
                                             f'and {num_gt} files in {self.gt_folder}'

        relative_input_files = [file.relative_to(self.input_folder) for file in input_files]
        relative_gt_files = [file.relative_to(self.gt_folder) for file in gt_files]

        assert all(input_file in relative_gt_files for input_file in relative_input_files), \
            f'{self.input_folder} and {self.gt_folder} must have the same setup of files'

        for i, (input_file, gt_file) in enumerate(zip(input_files, gt_files)):
            self.samples.append((input_file, gt_file, f'{self.label}_{i}'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        input_file, gt_file, metadata = self.samples[item]

        input = np.load(input_file)
        gt = np.load(gt_file)

        input = input.astype(np.float32)
        gt = gt.astype(np.float32)

        if self.input_transform:
            input = self.input_transform(input)

        if self.gt_transform:
            gt = self.gt_transform(gt)

        input = np.expand_dims(input, axis=0)
        gt = np.expand_dims(gt, axis=0)

        return input, gt, metadata


def to_subsets(dataset: Dataset, samples_per_subset: int) -> List[Dataset]:
    data_subsets = list()
    for idx in range(0, len(dataset), samples_per_subset):
        subset = torch.utils.data.Subset(dataset, range(idx, idx + samples_per_subset))
        data_subsets.append(subset)
    return data_subsets


def map_subset(dataset: Subset, indices: List[int] = None) -> Tuple[Dataset, List[int]]:
    """
    Maps from a subset to it's underlying dataset and indices.  Only really useful if the subset has more than
    one level of indirection i.e. it is a Subset of a Subset
    """
    if not indices:
        indices = dataset.indices
    if isinstance(dataset.dataset, Subset):
        subset_indices = [dataset.dataset.indices[i] for i in indices]
        return map_subset(dataset.dataset, subset_indices)
    return dataset.dataset, indices


def dataset_to_volume(dataset: Dataset) -> Tuple[np.ndarray, List[np.ndarray], List[Any]]:
    inputs = list()
    targets = list()
    metadata = list()

    for sample_input, sample_target, sample_metadata in dataset:
        inputs.append(sample_input)
        metadata.append(sample_metadata)

        if not isinstance(sample_target, list) and not isinstance(sample_target, tuple):
            # Single output so make a list to look like multiple outputs
            sample_target = [sample_target]

        if len(targets) == 0:
            targets = [list() for _ in range(len(sample_target))]

        # > 0: in case the gt are still in the original form (1,2,3,4,5). This won't work if we have multiple labels
        # in one slice
        for i, target in enumerate(sample_target):
            targets[i].append(target > 0)

    input_volume = np.stack(inputs)
    target_volumes = [np.stack(target) for target in targets]
    # print("input vol {} target[0] vols {}".format(input_volume.shape, target_volumes[0].shape)) # ???
    # print("len targets", len(target_volumes)) # ???
    return input_volume, target_volumes, metadata


def dataset_histogram(datasets: Union[Dataset, List[Dataset]], data_min: float, data_max: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(datasets, Dataset):
        datasets = to_subsets(dataset=datasets, samples_per_subset=400)

    dataset_iter = datasets
    if len(datasets) > 1:
        dataset_iter = tqdm(datasets, file=sys.stdout, desc=f'Computing data histogram')

    histogram = None
    for dataset in dataset_iter:
        volume, _, _ = dataset_to_volume(dataset=dataset)
        data_histogram = stats.compute_histogram(data=volume, data_min=data_min, data_max=data_max)
        histogram = stats.update_histogram(current_histogram=histogram, new_histogram=data_histogram)

    return histogram


def predict(model: torch.nn.Module, slices: np.array) -> List[np.ndarray]:
    device = get_device()
    batch_size = max_batch_size(model=model, input_size=slices.shape[1:], device=device)

    model.to(device)

    predictions = None

    model.eval()  # To disable dropout/freeze batch norm.
    for start_idx in range(0, len(slices), batch_size):
        if start_idx + batch_size > len(slices):
            batch_size = len(slices) - start_idx
        batch = slices[start_idx: start_idx + batch_size, ...]
        tensor_batch = torch.from_numpy(batch).float().to(device)

        with torch.no_grad():
            outputs = model(tensor_batch)

        if isinstance(outputs, dict):
            # In case the model output is a dict (UnetMultihead), unstack and convert to a list
            ## TODO: pass the outputs through activation in the model
            outputs = outputs['softmaxed_seg_logits']
            outputs = list(torch.unbind(outputs, dim=1))
            outputs = [out.unsqueeze(1) for out in outputs[1:]]

        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]

        if not predictions:
            predictions = [np.zeros_like(slices) for _ in range(len(outputs))]

        for i, out in enumerate(outputs):
            predictions[i][start_idx: start_idx + batch_size, ...] = out.cpu().numpy()

        start_idx += batch_size

    if device.type != 'cpu':
        # Free up any GPU memory once we're done
        torch.cuda.empty_cache()

    return predictions


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def device_memory(device: torch.device = None) -> int:
    """Returns free GPU device memory in bytes or -1 if device is cpu"""
    if not device:
        device = get_device()

    if device.type == 'cpu':
        return -1

    pynvml.nvmlInit()

    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    device_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
    denominator = 1024 ** 3
    unit = 'GB'

    logger.debug(f'GPU: {device_name}')
    logger.debug(f'    - Total memory: {info.total / denominator:.2f}{unit}')
    logger.debug(f'    - Free memory: {info.free / denominator:.2f}{unit}')
    logger.debug(f'    - Used memory: {info.used / denominator:.2f}{unit}')

    pynvml.nvmlShutdown()

    return info.free


def max_batch_size(model: torch.nn.Module, input_size: Tuple, device: torch.device = None) -> int:
    if not device:
        device = get_device()
    if device.type == 'cpu':
        return 16

    available_memory = device_memory(device=device)

    # TODO: something more intelligent than this!
    #  I.e. actually use the model and input size to compute expected memory usage

    # available_memory -= expected_model_size
    gb_per_input = 0.36
    total_gb = available_memory / 1024 ** 3
    return int(0.8 * total_gb / gb_per_input)


def check_batch_size(batch_size: int, input_size: Tuple, model: torch.nn.Module,
                     device: torch.device) -> None:
    max_batch = max_batch_size(model=model, device=device, input_size=input_size)

    if batch_size > max_batch:
        logger.warning(f'Input batch size {batch_size} is more than the max ({max_batch}) given current '
                       f'available GPU memory.  You should reduce the batch size.')
    elif batch_size < max_batch:
        logger.warning(f'Input batch size {batch_size} is less than the max ({max_batch}) given current '
                       f'available GPU memory.  You could increase the batch size.')


class EarlyStopping:
    """ Modified from version at https://github.com/Bjarten/early-stopping-pytorch"""
    """Early stops the training if the given loss doesn't improve after a given patience."""

    def __init__(self, model_file_path: Path, loss_to_monitor: str, initial_losses: Dict[str, float] = None,
                 patience: int = 7, verbose: bool = False, delta: float = 0):
        """
        Args:
            model_file_path: Path for saving the best model
            loss_to_monitor: value to track in loss for early stopping
            initial_losses: If provided, used to set the current best loss
            patience:   How long to wait after last time validation loss improved.
                        Default: 7
            verbose:    If True, prints a message for each validation loss improvement.
                        Default: False
            delta:      Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0
        """
        if initial_losses:
            assert loss_to_monitor in initial_losses, \
                f'Could not find {loss_to_monitor} in losses ({initial_losses})'

            min_loss = initial_losses[loss_to_monitor]
            self.best_score = -min_loss
            self.loss_min = min_loss
        else:
            self.best_score = None
            self.loss_min = np.Inf
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.loss_to_monitor = loss_to_monitor
        self.delta = delta
        self.model_file_path = model_file_path

    def __call__(self, losses: Dict[str, float], model: torch.nn.Module):

        assert self.loss_to_monitor in losses, f'Could not find {self.loss_to_monitor} in losses ({losses})'

        current_loss = losses[self.loss_to_monitor]

        score = -current_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # logger.info(f'Loading best model ({self.loss_to_monitor} = {self.loss_min})'
                #             f' from {self.model_file_path}')
                self.save_bestval(current_loss, model)
                # model.load_state_dict(torch.load(self.model_file_path).state_dict())
                return True
        else:
            self.best_score = score
            self.save_checkpoint(current_loss, model)
            self.counter = 0

    def save_checkpoint(self, current_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(f'{self.loss_to_monitor} decreased ({self.loss_min:.6f} --> {current_loss:.6f}).  '
                        f'Saving model to {self.model_file_path} ...')
        self.model_file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, self.model_file_path)
        torch.save(model.state_dict(), str(self.model_file_path)+".state_dict")
        self.loss_min = current_loss

    def save_bestval(self, current_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info("saving best validation model")
        self.model_file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, str(self.model_file_path)+'_bestval')
        torch.save(model.state_dict(), str(self.model_file_path)+"_bestval.state_dict")
        self.loss_min = current_loss

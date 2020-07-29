import enum
import logging
from pathlib import Path
from typing import List, Union, Tuple, Dict

import numpy as np
import pandas

from utils import stats

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


class ClippedNormalisation(object):
    """Normalisation of image that accounts for clipping of image range.  mean and/or std cannot be provided
    on their own.  Either both must be provided or neither of them.  If mean and std are not provided, the
    image is normalised using (image - min_value)/(max_value - min_value)
    """

    def __init__(self, min_value: int, max_value: int, mean: float = None, std: float = None):
        assert max_value > min_value, f'max_value ({max_value} must be greater than min_value ({min_value})'
        assert mean and std or (not mean and not std), \
            f'If mean ({mean}) or std ({std}) are provided, the other must also be provided'

        self.min_value = min_value
        self.max_value = max_value

        self.subtractor = mean if mean else min_value
        self.divisor = std if std else (max_value - min_value)

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        sample = np.clip(sample, a_min=self.min_value, a_max=self.max_value)

        sample -= self.subtractor
        sample /= self.divisor

        return sample


class HistogramNormalisation(ClippedNormalisation):
    def __init__(self, histogram: Union[Path, Tuple[np.ndarray, np.ndarray]], min_value: int, max_value: int):
        if isinstance(histogram, Path):
            histogram = np.load(histogram)

        mean, std = stats.mean_and_std_from_histogram(histogram=histogram, min_value=min_value,
                                                      max_value=max_value)
        super(HistogramNormalisation, self).__init__(min_value=min_value, max_value=max_value, mean=mean,
                                                     std=std)


def pad_or_truncate(input_data: np.ndarray, output_shape: Union[Tuple[int, int], Tuple[int, int, int]],
                    padding_value: int):
    """
    Pads or truncates in each dimension so that the input data has the shape specified.
    Uses the given constant value when padding.

    :param input_data: Numpy array to be reshaped
    :param output_shape: Desired shape of output
    :param padding_value: Single value which will be used for padding
    :return: input_data but same shape as output_shape via zero padding or truncation
    """
    assert len(np.shape(input_data)) == len(output_shape)
    resized_data = np.array(input_data, copy=True)
    ndims = len(output_shape)
    # pad_amounts and slice_amounts are identity transformations for the padding and slicing operations respectively
    pad_amounts = [(0, 0), ] * ndims
    slice_amounts = [slice(None), ] * ndims
    # Loop over each dimension and determine if padding or truncation is necessary
    for i, (dim_in, dim_out) in enumerate(zip(np.shape(resized_data), output_shape)):
        if dim_in < dim_out:
            # pad
            diff = dim_out - dim_in
            diff_before = diff - diff // 2
            diff_after = diff - diff_before
            pad_amounts[i] = (diff_before, diff_after)
        elif dim_in > dim_out:
            # truncate
            diff = dim_in - dim_out
            diff_before = diff - diff // 2
            diff_after = diff - diff_before
            slice_amounts[i] = slice(diff_before, dim_in - diff_after)

    resized_data = np.pad(resized_data, tuple(pad_amounts), 'constant', constant_values=padding_value)
    resized_data = resized_data[tuple(slice_amounts)]
    return resized_data

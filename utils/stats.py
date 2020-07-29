import math
from pathlib import Path
from typing import Tuple, Union

import numpy as np


def compute_histogram(data: np.ndarray, data_min: Union[float, int] = None,
                      data_max: Union[float, int] = None) -> Tuple[np.ndarray, np.ndarray]:
    data_min = data_min if data_min else np.min(data)
    data_max = data_max if data_max else np.max(data)

    clipped_data = np.clip(data, data_min, data_max)

    return np.histogram(clipped_data, bins=data_max - data_min, range=(data_min, data_max))


def update_histogram(current_histogram: Tuple[np.ndarray, np.ndarray],
                     new_histogram: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if current_histogram is None:
        return new_histogram

    assert np.array_equal(current_histogram[1], new_histogram[1]), "Bin edges are not identical!"

    return current_histogram[0] + new_histogram[0], current_histogram[1]


def save_histogram(histogram: Tuple[np.ndarray, np.ndarray], file_path: Path) -> None:
    hist, bins = histogram
    np.save(file_path, np.stack([hist, bins[:-1].astype(np.int32)]))


def clip_histogram(histogram: Tuple[np.ndarray, np.ndarray], min_value: Union[float, int],
                   max_value: Union[float, int]) -> Tuple[np.ndarray, np.ndarray]:
    counts, bins = histogram
    lower_index = int(min_value - bins[0])
    upper_index = int(lower_index + (max_value - min_value))
    counts[lower_index] = sum(counts[:lower_index + 1])
    counts[upper_index] = sum(counts[upper_index:])
    return counts[lower_index:upper_index + 1], bins[lower_index:upper_index + 1]


def mean_and_std_from_histogram(histogram: Tuple[np.ndarray, np.ndarray], min_value: Union[float, int],
                                max_value: Union[float, int]) -> Tuple[float, float]:
    counts, bins = clip_histogram(histogram=histogram, min_value=min_value, max_value=max_value)
    mean = np.average(bins, weights=counts)
    variance = np.average((bins - mean) ** 2, weights=counts)
    std = math.sqrt(variance)
    return mean, std

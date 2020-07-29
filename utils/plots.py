from pathlib import Path
from typing import Any, List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def loss_history(history: Dict[str, List[Any]], file_path: Path):
    fig, ax = plt.subplots()
    min_value = np.inf
    max_value = -np.inf
    for name, values in history.items():
        min_value = min(min_value, np.min(values))
        max_value = max(max_value, np.max(values))
        x = np.arange(len(values))
        ax.plot(x, values, label=name)
        legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    min_value = np.floor(min_value)
    max_value = np.ceil(max_value)
    ax.set_ylim(min_value, max_value)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(file_path)
    plt.close(fig)


def slice_images(images: Dict[str, List[np.ndarray]], output_file: Path, title: str = None,
                 image_ranges: List[Tuple[float, float]] = None):
    num_images = len(images)
    assert num_images > 0

    if image_ranges:
        assert len(image_ranges) == num_images, f'The number of image ranges ({len(image_ranges)}) must ' \
                                                f'match the number of input images ({num_images})'
    else:
        image_ranges = [None] * num_images

    if not title:
        title = output_file.stem

    output_file.parent.mkdir(parents=True, exist_ok=True)

    plot_width = 5
    plot_height = 5
    fig_size = (plot_width * num_images, plot_height)

    fig, ax = plt.subplots(nrows=1, ncols=num_images, squeeze=False, figsize=fig_size)
    plt.suptitle(title)
    for i, (name, image) in enumerate(images.items()):
        if image_ranges[i]:
            min_value, max_value = image_ranges[i]
        else:
            min_value, max_value = np.min(image), np.max(image)
        current_axes = ax[0][i]
        im = current_axes.imshow(np.squeeze(image), cmap='gray', origin='lower', vmin=min_value,
                                 vmax=max_value)
        plt.colorbar(im, ax=current_axes)
        current_axes.set_title(f'{name}')
    plt.savefig(output_file)
    plt.close(fig)


def volume_mips(volumes: Dict[str, np.ndarray], output_file: Path, title: str = None,
                image_ranges: List[Tuple[float, float]] = None):
    num_volumes = len(volumes)
    assert num_volumes > 0

    if image_ranges:
        assert len(image_ranges) == num_volumes, f'The number of image ranges ({len(image_ranges)}) must ' \
                                                 f'match the number of input volumes ({num_volumes})'
    else:
        image_ranges = [None] * num_volumes

    if not title:
        title = output_file.stem

    output_file.parent.mkdir(parents=True, exist_ok=True)

    axes = {'axial': 0, 'coronal': 1, 'sagittal': 2}

    num_axes = len(axes)

    plot_width = 5
    plot_height = 5
    fig_size = (plot_width * num_axes, plot_height * num_volumes)

    fig, ax = plt.subplots(nrows=num_volumes, ncols=num_axes, squeeze=False, figsize=fig_size)
    plt.suptitle(title)
    for i, (name, volume) in enumerate(volumes.items()):
        if image_ranges[i]:
            min_value, max_value = image_ranges[i]
        else:
            min_value, max_value = np.min(volume), np.max(volume)
        for view, axis in axes.items():
            im = ax[i][axis].imshow(np.max(np.squeeze(volume), axis=axis), cmap='gray', origin='lower',
                                    vmin=min_value, vmax=max_value)
            plt.colorbar(im, ax=ax[i][axis])
            ax[i][axis].set_title(f'{name} - {view}')
    plt.savefig(output_file)
    plt.close(fig)

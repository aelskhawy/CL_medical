import logging
from pathlib import Path
from typing import List, Dict, Any

from torch.optim import Adam

from continual_learning.model_base import CLModel
from continual_learning.naive import NaiveModel
# from datasets import lung
from datasets.all_data import DataQuery
from experiments import metrics, evaluation
from models.unet import UNet
from nn.losses import to_multioutput, dice_loss_with_missing_values
from utils import visualise
from utils.data import Split
from utils.logging import configure_logging
from utils.options import Options

logger = logging.getLogger(__name__)


def continual_learning_tasks(options: Options) -> List[DataQuery]:
    # TODO: Use the options to actually change what we are working on!
    return ltrc_manufacturer_tasks()


# TODO: Return a proper data class to hold parameters, not just a dictionary!
def training_parameters(options: Options) -> Dict[str, Any]:
    num_epochs = 1 if options.debug_mode else 100
    return {
        'optimiser': Adam,
        'learning_rate': 0.0001,
        'weight_decay': 1e-3,
        'batch_size': 16,
        'num_epochs': num_epochs,
        'loss': to_multioutput(dice_loss_with_missing_values),
        'num_workers': 2
    }


def continual_learning_model(options: Options) -> CLModel:
    # TODO: Use the options to actually change what we are working on!
    model = UNet(n_channels=1, n_classes=1, n_base_filters=16)
    cl_model = NaiveModel(initial_model=model)

    return cl_model


# TODO: Return a proper data class to hold parameters, not just a dictionary!
def evaluation_parameters(options: Options) -> Dict[str, Any]:
    # TODO: Use the options to actually change what we are working on!
    return {
        'evaluation_fn': metrics.masked_dice_coefficient,
        'evaluation_mode': evaluation.EvaluationMode.PerVolume,
        'prediction_threshold': 0.5
    }


def ltrc_manufacturer_tasks() -> List[DataQuery]:
    return [
        DataQuery(tasks='lung', domains={'manufacturer': 'SIEMENS', 'origin': 'LTRC'}),
        DataQuery(tasks='lung', domains={'manufacturer': 'GE MEDICAL SYSTEMS', 'origin': 'LTRC'})
    ]


def lung_pathology_tasks() -> List[DataQuery]:
    return [
        DataQuery(tasks='lung_healthy'),
        DataQuery(tasks='lung_pathological')
    ]


def output_images(tasks: List[DataQuery], output_root: Path):
    splits = Split.training_splits() + Split.test_splits()
    for query in tasks:
        task_name = str(query)
        for split in splits:
            output_folder = output_root / task_name / str(split)
            output_folder.mkdir(parents=True, exist_ok=True)

            data = lung.get_datasets(splits=split, normalised=True, filters=query.domains)
            visualise.plot_volume_datasets(datasets=data, output_folder=output_folder)


def main():
    configure_logging()

    lung_tasks = ltrc_manufacturer_tasks() + lung_pathology_tasks()
    output_images(tasks=lung_tasks, output_root=lung.debug_root())


if __name__ == '__main__':
    main()

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).parent.parent


def input_data_root() -> Path:
    # Set the environment variable "ADP_CL_DATA_ROOT" to data location.  Defaults to "<project_repo>/data".
    path_root = os.environ.get("ADP_CL_DATA_ROOT", project_root() / 'data')
    return Path(path_root)
    # return Path("/ADP_ContinualLearning/data/")  # DGX

def nutfiles_data_root() -> Path:
    return Path(r"\\nutfiles\ClinicalData\AI_Data_Library\ActiveLearning")


def output_data_root() -> Path:
    return project_root() / 'output'


def model_output_root() -> Path:
    return output_data_root() / 'models'


def training_output_root() -> Path:
    return output_data_root() / 'training'


def evaluation_output_root() -> Path:
    return output_data_root() / 'evaluation'


def debug_output_root() -> Path:
    return output_data_root() / 'debug'

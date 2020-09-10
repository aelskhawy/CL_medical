import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
import os, sys
np.random.seed(0)

import logging

logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
       )
logger = logging.getLogger(__name__)

from continual_learning.experiment import CLExperiment
# from experiments import domain_adaptation, incremental_task_learning
from experiments import incremental_task_learning
from utils import paths
from utils.logging import configure_logging
from utils.options import Options
from pathlib import Path



def parse_options() -> Options:
    opt_obj = Options()
    opt = opt_obj.parse()
    opt_obj.print_options(opt)
    return opt


def main():
    options = parse_options()
    verbose = options.verbose  # False
    debug_mode = options.debug_mode  # True
    run_training = options.run_training
    run_evaluation = options.run_evaluation
    output_predictions = options.output_predictions
    overwrite = options.overwrite
    domain_learning = options.domain_learning  # True

    if not options.supress_logging:
        configure_logging(log_debug=verbose)

    if domain_learning:
        tasks = domain_adaptation.continual_learning_tasks(options=options)
        training_parameters = domain_adaptation.training_parameters(options=options)
        evaluation_parameters = domain_adaptation.evaluation_parameters(options=options)
        cl_model = domain_adaptation.continual_learning_model(options=options)
    else:
        tasks = incremental_task_learning.continual_learning_tasks(options=options)
        training_parameters = incremental_task_learning.training_parameters(options=options)
        evaluation_parameters = incremental_task_learning.evaluation_parameters(options=options)
        cl_model = incremental_task_learning.continual_learning_model(options=options)

    start_task = options.start_task

    experiment_name = str(options.name) #';'.join([str(task) for task in tasks])
    # if debug_mode:
    #     experiment_name = f'{experiment_name}_debug'
    algorithm_name = str(cl_model)

    ROOT_DIR = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
    if ROOT_DIR == "/code":
        from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
        output_path = get_outputs_path()
        output_folder = Path(output_path) / experiment_name / algorithm_name

    else:
        output_folder = paths.output_data_root() / experiment_name / algorithm_name

    print(">>>>>>>>>>>Output folder is {}".format(output_folder))

    if run_training:
        experiment = CLExperiment(cl_model=cl_model, tasks=tasks, output_folder=output_folder, options=options)
        experiment.train(training_parameters=training_parameters, starting_task=start_task,
                         overwrite=overwrite, debug_mode=debug_mode)

    if run_evaluation:
        file_path = output_folder / CLExperiment.file_name()
        experiment = CLExperiment.load(file_path=file_path)
        experiment.evaluate(evaluation_parameters=evaluation_parameters, debug_mode=debug_mode)

    if output_predictions:
        file_path = output_folder / CLExperiment.file_name()
        experiment = CLExperiment.load(file_path=file_path)
        experiment.output_predictions(debug_mode=debug_mode)


if __name__ == '__main__':
    main()

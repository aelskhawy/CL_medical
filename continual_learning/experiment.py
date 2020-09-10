import logging
import pickle
import sys
from pathlib import Path, PurePath
from typing import List, Dict, Any, Tuple
import time
import pandas
import torch
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

from continual_learning.model_base import CLModel
from datasets.all_data import get_data, DataQuery
from experiments.evaluation import continual_learning_metrics, output_results_to_csv, \
    compute_model_task_scores
from utils import plots, paths
from utils.data import Split
from utils.options import Options
logger = logging.getLogger(__name__)


class CLExperiment:

    def __init__(self, cl_model: CLModel, tasks: List[DataQuery], output_folder: Path, options: Options=None):
        self.cl_model = cl_model
        self.tasks = tasks
        self.polyaxon_out_folder = output_folder
        # Make relative so that when we save it, the experiment can be reloaded from a different location
        # relative_output_folder = output_folder.relative_to(paths.project_root())
        # self._output_folder = PurePath(str(relative_output_folder))
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.options = options
        self.verify_tasks()

    @property
    def output_folder(self):
        # return paths.project_root() / Path(self._output_folder)
        return Path(self.polyaxon_out_folder)
    @property
    def file_path(self):
        return self.output_folder / self.file_name()

    @classmethod
    def file_name(cls) -> str:
        return 'experiment.bin'

    @classmethod
    def load(cls, file_path: Path):
        logger.info(f'Loading experiment from {file_path}')
        with open(str(file_path), 'rb') as f:
            return pickle.load(f)

    def save(self, file_path: Path):
        logger.info(f'Saving experiment to {file_path}')
        with open(str(file_path), 'wb') as f:
            pickle.dump(self, f)

    def verify_tasks(self):
        """Verifies this experiment's own task list against any trained tasks already in the CLModel"""
        known_tasks = self.cl_model.trained_tasks()
        # TODO: Maybe CLModel could also store these as DataQuery objects as well, to save conversion to
        #  string in order to compare them?
        task_names = [str(task) for task in self.tasks]
        trained_tasks = [task for task in task_names if task in known_tasks]
        tasks_to_train = [task for task in task_names if task not in known_tasks]

        if len(trained_tasks) == 0:
            # No overlap between tasks in self.tasks and those already known by the CLModel
            return

        # Checks that the order of the trained tasks in the experiment's task list matches the last trained
        # tasks in the CLModel
        last_tasks = known_tasks[-len(trained_tasks):]
        assert trained_tasks == last_tasks, \
            f'The last trained tasks ({last_tasks}) do not match the order of the tasks requested to ' \
            f'train ({task_names})'

        assert trained_tasks + tasks_to_train == task_names, \
            f'Task list ({task_names}) has a different order to already trained tasks ' \
            f'({trained_tasks}) and tasks that will be trained ({tasks_to_train})'

    def define_task_list(self, starting_task: int, overwrite: bool) -> List[DataQuery]:
        known_tasks = self.cl_model.trained_tasks()
        trained_tasks = [task for task in self.tasks if str(task) in known_tasks]
        tasks_to_train = [task for task in self.tasks if str(task) not in known_tasks]

        if len(trained_tasks) == 0:
            return tasks_to_train

        start_index = 0
        if starting_task:
            start_index = starting_task

        if not overwrite:
            # Don't overwrite any trained tasks
            new_index = max(start_index, len(trained_tasks))
            if new_index > start_index:
                logger.warning(f'Overwrite = False so skipping known tasks: '
                               f'{self.tasks[start_index:new_index]}')
            start_index = new_index

        if start_index > len(trained_tasks):
            # Starting task is in the untrained tasks, start at the beginning of these tasks
            logger.warning(f'Tasks earlier than {self.tasks[start_index]} have not been trained, starting at '
                           f'the earliest untrained task: {self.tasks[len(trained_tasks)]}')
            start_index = len(trained_tasks)

        return self.tasks[start_index:]

    @staticmethod
    def get_training_data(data_query: DataQuery, debug_mode: bool = False, options: Options=None) \
            -> Tuple[Dataset, Dataset]:
        training_data_volumes = get_data(query=data_query, split=Split.Training,
                                         debug_mode=debug_mode, options=options)
        early_stopping_data_volumes = get_data(query=data_query, split=Split.EarlyStopping,
                                               debug_mode=debug_mode, options=options)

        # Combine all volume samples for slice-wise training
        training_data = ConcatDataset(training_data_volumes)
        early_stopping_data = ConcatDataset(early_stopping_data_volumes)

        return training_data, early_stopping_data

    @staticmethod
    def get_test_data(data_query: DataQuery, debug_mode: bool = False, options: Options=None) -> List[Dataset]:
        return get_data(query=data_query, split=Split.FinalEvaluation, debug_mode=debug_mode, options=options)

    def train(self, training_parameters: Dict[str, Any], starting_task: int = None,
              overwrite: bool = False, debug_mode: bool = False):
        task_list = self.define_task_list(starting_task=starting_task, overwrite=overwrite)

        for data_query in task_list:
            task_name = str(data_query)
            model_file_path = self.output_folder / f'{task_name}.pt'

            if model_file_path.exists() and not overwrite:
                logger.info(f'Overwrite = False and {model_file_path} exists.  Loading model for {task_name}')
                model = torch.load(model_file_path)
                self.cl_model.add_task(task_name=task_name, model=model)
            else:
                logger.info(f'Training {self.cl_model} for task {task_name}')
                print("========> Fetching training data <===========") # ????
                st = time.time() #?????
                training_data, early_stopping_data = self.get_training_data(data_query=data_query,
                                                                            debug_mode=debug_mode,
                                                                            options=self.options)
                end = time.time()
                print("Time to get data", (end-st)/60) ##????
                loss_history = self.cl_model.train(training_data=training_data,
                                                   validation_data=early_stopping_data,
                                                   training_parameters=training_parameters,
                                                   model_file_path=model_file_path, task_name=task_name)

                # plots.loss_history(history=loss_history,
                #                    file_path=self.output_folder / f'{task_name}_loss.png')

            self.save(self.file_path)

    def output_predictions(self, debug_mode: bool = False):
        prediction_output_root = self.output_folder / 'predictions'
        threshold = 0.5

        for i in range(len(self.tasks)):
            model_task_name = str(self.tasks[i])
            model_name = '+'.join(str(task) for task in self.tasks[0:i+1])

            predictions_output_folder = prediction_output_root / model_name

            for data_query in self.tasks[0:i+1]:
                test_data = self.get_test_data(data_query=data_query, debug_mode=debug_mode)
                current_task = str(data_query)
                data_iter = tqdm(test_data, file=sys.stdout,
                                 desc=f'Outputting predictions of {model_name} model for {data_query}')
                for data in data_iter:
                    _, _, sample_name = next(iter(data))
                    prediction, target = self.cl_model.prediction_and_target(dataset=data,
                                                                             model_task_name=model_task_name,
                                                                             task_to_predict=current_task)
                    diff = target - prediction

                    import numpy as np
                    thresholded_prediction = np.where(prediction >= threshold, 1.0, 0.0)
                    thresholded_diff = target - thresholded_prediction

                    volumes = {'GT': target, 'Prediction': prediction, 'Diff. (GT - Prediction)': diff,
                               f'Prediction (threshold={threshold})': thresholded_prediction,
                               f'Diff. (GT - Prediction, threshold={threshold})': thresholded_diff}

                    image_ranges = [(0, 1)] * len(volumes)
                    output_file = predictions_output_folder / f'{current_task}_{sample_name}.png'
                    plots.volume_mips(volumes=volumes,
                                      output_file=output_file, image_ranges=image_ranges)

    # def evaluate(self, evaluation_parameters: Dict[str, Any], debug_mode: bool = False):
    #     # TODO: Use a proper data class, not just a dictionary!
    #     evaluation_fn = evaluation_parameters['evaluation_fn']
    #     evaluation_mode = evaluation_parameters['evaluation_mode']
    #     task_threshold = evaluation_parameters['prediction_threshold']
    #
    #     task_test_data = [
    #         self.get_test_data(data_query=data_query, debug_mode=debug_mode, options=self.options) for
    #         data_query in self.tasks
    #     ]
    #
    #     model_scores = compute_model_task_scores(cl_model=self.cl_model, task_test_data=task_test_data,
    #                                              score_fn=evaluation_fn, threshold=task_threshold,
    #                                              evaluation_mode=evaluation_mode)
    #
    #     task_names = [str(q) for q in self.tasks]
    #     for i, task in enumerate(self.tasks):
    #         task_name = str(task)
    #         output_file = self.output_folder / f'{task_name}_scores.csv'
    #         df = pandas.DataFrame()
    #         for j in range(i, len(model_scores)):
    #             current_model_scores = model_scores[j]
    #             model_name = '+'.join(task_names[0:j + 1])
    #             task_scores = current_model_scores[i]
    #             df[model_name] = task_scores
    #         df.to_csv(output_file)
    #
    #     # TODO: compute the ideal scores for each task!
    #     ideal_scores = [1.0] * len(model_scores)
    #
    #     omega_base, omega_new, omega_all = continual_learning_metrics(model_scores=model_scores,
    #                                                                   ideal_scores=ideal_scores)
    #
    #     logger.info(f'Omega base: {omega_base}, Omega New: {omega_new}, Omega All: {omega_all}')
    #
    #     final_scores = dict()
    #     final_model_scores = model_scores[-1]
    #     for task_name, task_scores in zip(self.tasks, final_model_scores):
    #         final_scores[task_name] = {task_threshold: task_scores}
    #
    #     # TODO: provide score labels
    #     output_results_to_csv(scores=final_scores, output_path=self.output_folder)

    def evaluate(self, evaluation_parameters: Dict[str, Any], debug_mode: bool = False):
        # TODO: Use a proper data class, not just a dictionary!
        evaluation_fn = evaluation_parameters['evaluation_fn']
        evaluation_mode = evaluation_parameters['evaluation_mode']
        task_threshold = evaluation_parameters['prediction_threshold']

        # print("debug mode {}".format(debug_mode))
        # task_test_data = [
        #     self.get_test_data(data_query=data_query, debug_mode=debug_mode, options=self.options) for
        #     data_query in self.tasks
        # ]

        ### For vis purpose
        available_tasks = self.cl_model.trained_tasks()
        # fetches spinal, spinal+ rlung, spinal+rlung+llung ...
        data_queries_multilable = [DataQuery(tasks=available_tasks[:i+1]) for i in range(len(available_tasks))]

        task_test_data = [
            self.get_test_data(data_query=data_query, debug_mode=debug_mode, options=self.options) for
            data_query in data_queries_multilable ]
        # print("task_test_data {}".format(task_test_data))

        model_scores = compute_model_task_scores(cl_model=self.cl_model, task_test_data=task_test_data,
                                                 score_fn=evaluation_fn, threshold=task_threshold,
                                                 evaluation_mode=evaluation_mode)

        # TODO: change the order of the ideal scores according to the training order!
        # ideal_scores = [0.76, 0.92, 0.93, 0.925, 0.764]  # orderA
        ideal_scores = [ 0.764 , 0.925,  0.92, 0.93, 0.76]  # orderB

        omega_base, omega_new, omega_all = continual_learning_metrics(model_scores=model_scores,
                                                                      ideal_scores=ideal_scores)

        logger.info(f'Omega base: {omega_base}, Omega New: {omega_new}, Omega All: {omega_all}')

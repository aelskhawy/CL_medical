import enum
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Callable, Dict, Union, Any

import numpy as np
import pandas as pd
import torch
from torch.nn import Module
from torch.utils.data import Dataset
from tqdm import tqdm

from continual_learning.model_base import CLModel
from utils import pytorch, plots

logger = logging.getLogger(__name__)


class EvaluationMode(enum.Enum):
    PerSlice = 'PerSlice'
    PerVolume = 'PerVolume'

    def __str__(self):
        return str(self.value)


# def compute_model_task_scores(cl_model: CLModel, task_test_data: List[List[Dataset]],
#                               score_fn: Callable[[np.ndarray, np.ndarray], float], threshold: float,
#                               evaluation_mode: EvaluationMode):
#     model_scores = list()
#     task_names = cl_model.trained_tasks()
#     task_iter = tqdm(task_names, file=sys.stdout)
#     for i, current_task_name in enumerate(task_iter):
#         task_iter.set_description(desc=f'Evaluating model for {current_task_name}')
#
#         # Only evaluates one task at a time.  This could be more efficient if we could say that a Dataset has
#         # GT for all tasks up to this point and evaluate all simultaneously.  Unfortunately this is not
#         # possible if we want to mix different inputs for different tasks, so we have to evaluate per-task.
#         all_task_scores = list()
#         for j in range(0, i + 1):
#             test_data = task_test_data[j]
#             other_task_name = task_names[j]
#             task_scores = list()
#             for data in test_data:
#                 prediction, target = cl_model.prediction_and_target(dataset=data,
#                                                                     model_task_name=current_task_name,
#                                                                     task_to_predict=other_task_name)
#
#                 scores = evaluate_prediction(prediction=prediction, target=target, threshold=threshold,
#                                              score_fn=score_fn, mode=evaluation_mode)
#                 task_scores.extend(scores)
#             all_task_scores.append(task_scores)
#
#         model_scores.append(all_task_scores)
#     return model_scores
def compute_model_task_scores(cl_model: CLModel, task_test_data: List[List[Dataset]],
                              score_fn: Callable[[np.ndarray, np.ndarray], float], threshold: float,
                              evaluation_mode: EvaluationMode):
    all_models_scores = list()
    task_names = cl_model.trained_tasks()
    task_labels = [cl_model.roi_order.index(task) for task in task_names]  # original labels
    print("task_names {}".format(task_names))
    # task_iter = tqdm(task_names, file=sys.stdout)
    for i, current_task_name in enumerate(task_names):

        # task_iter.set_description(desc=f'Evaluating model {current_task_name}')
        print("#" * 80)
        print("Evaluating model {}".format(current_task_name))
        print("#" * 80)

        if current_task_name != "oesophagus":
            continue  # for visualization purpose evaluating only last model

        active_classes = [0]  # background
        one_model_scores = list()
        for j in range(0, i + 1):
            test_data = task_test_data[j]
            # test_data_multi_label = test_data_multi_label
            logger.info("############# Evaluating {} task ################".format(j))
            active_classes.append(task_labels[j])
            one_task_scores = cl_model.prediction_and_target(dataset=test_data, model_task_name=current_task_name,
                                                             active_classes=active_classes)
            one_model_scores.append(one_task_scores)

        logger.info("one_model_scores {}".format(one_model_scores))
        all_models_scores.append(one_model_scores)
        print("all models scores {}".format(all_models_scores))
    return all_models_scores

def continual_learning_metrics(model_scores: List[List[float]], ideal_scores: List[float]) \
        -> Tuple[float, float, float]:
    """Calculates metrics proposed in 'Measuring Catastrophic Forgetting in Neural Networks' Kemker et al. 2017

    Computes the following metrics:
        omega_base: Measures a model’s retention of the first session, after learning in later study sessions
        omega_new: Measures a model’s ability to immediately recall new tasks
        omega_all: Measures how well a model both retains prior knowledge and acquires new information

    model_scores should be the same length as the number of tasks trained.  Each element of model_scores
    should be a list of scores for each task trained for that model.  E.g. model_scores[0] should contain the
    scores for model 0 on task 0, model_scores[1] should contain the scores for model 1 on tasks 0 and 1 etc.
    This method assumes that the model_scores[0] represents the baseline model's scores

    :param model_scores: Per-model scores for each task, can be computed using compute_model_task_scores.
    :param ideal_scores: Score for the equivalent model trained on all data
    :return: omega_base, omega_new, omega_all metrics
    """

    omega_base, omega_new, omega_all = 0, 0, 0
    for task_id in range(1, len(model_scores)):
        current_model_scores = model_scores[task_id]
        # print(current_model_scores, ideal_scores)
        omega_base += current_model_scores[0] / ideal_scores[0]
        omega_new += current_model_scores[task_id] / ideal_scores[task_id]
        omega_all += np.mean(current_model_scores) / np.mean(ideal_scores[:task_id + 1])
        # print(np.mean(current_model_scores), np.mean(ideal_scores[:task_id + 1]))

    number_of_tasks = len(model_scores)
    denominator = number_of_tasks - 1
    omega_base /= denominator
    omega_new /= denominator
    omega_all /= denominator

    return omega_base, omega_new, omega_all


def output_predictions(model: Module, data: List[Dataset], thresholds: Union[float, List[float]],
                       mode: EvaluationMode, output_root: Path):
    image_ranges = [None, (0, 1), None] + [(0, 1)] * len(thresholds)
    sub_folder = str(mode)
    data_iter = tqdm(data, file=sys.stdout, desc=f'Outputting {mode.value} predictions')
    for dataset in data_iter:
        input, targets, metadata = pytorch.dataset_to_volume(dataset)
        predictions = pytorch.predict(model=model, slices=input)
        for task, (prediction, target) in enumerate(zip(predictions, targets)):
            thresholded_predictions = {
                f'Threshold = {t}': np.where(prediction >= t, 1.0, 0.0) for t in thresholds
            }

            if mode == EvaluationMode.PerVolume:
                # TODO: Check consistency of metadata
                data_id = str(metadata[0])
                title = f'{data_id}'
                volumes = {'input': input, 'GT': target, 'Raw Prediction': prediction}
                volumes.update(thresholded_predictions)
                output_file = output_root / sub_folder / f'{task}_{data_id}.png'
                plots.volume_mips(volumes=volumes, output_file=output_file, title=title,
                                  image_ranges=image_ranges)
            elif mode == EvaluationMode.PerSlice:
                for i, (input_slice, target_slice, prediction_slice) in enumerate(
                        zip(input, target, prediction)):
                    data_id = str(metadata[i])
                    title = f'{data_id}'
                    images = {'input': input_slice, 'GT': target_slice, 'Raw Prediction': prediction_slice}
                    thresholded_images = {
                        key: volume[i, ...] for key, volume in thresholded_predictions.items()
                    }
                    images.update(thresholded_images)
                    output_file = output_root / sub_folder / f'{data_id}.png'
                    plots.slice_images(images=images, output_file=output_file, title=title,
                                       image_ranges=image_ranges)


def evaluate_prediction(prediction: np.ndarray, target: np.ndarray, threshold: float,
                        score_fn: Callable[[np.ndarray, np.ndarray], float], mode: EvaluationMode) \
        -> List[float]:
    thresholded_prediction = np.where(prediction >= threshold, 1.0, 0.0)
    if mode == EvaluationMode.PerSlice:
        prediction_scores = list()
        for prediction_slice, target_slice in zip(thresholded_prediction, target):
            slice_score = score_fn(prediction_slice, target_slice)
            prediction_scores.append(slice_score)
    else:
        volume_score = score_fn(thresholded_prediction, target)
        prediction_scores = [volume_score]

    return prediction_scores


def evaluate_model(model: Module, data: Union[Dataset, List[Dataset]], thresholds: Union[float, List[float]],
                   score_fn: Callable[[np.ndarray, np.ndarray], float], mode: EvaluationMode) \
        -> Union[Dict[int, Dict[float, List[float]]], Dict[float, List[float]]]:
    """
    Assumes that the number of targets in the dataset is equal to the number of outputs of the model.
    If 'mode' == Evaluation.PerSlice, each dataset in 'data' is treated as multiple image slices.
    If 'mode' == Evaluation.PerVolume, each dataset in 'data' is treated as a single volume.
    """

    if mode != EvaluationMode.PerSlice and mode != EvaluationMode.PerVolume:
        raise ValueError(f'Unknown evaluation mode: {mode}')

    if not isinstance(data, list):
        data = [data]

    if not isinstance(thresholds, list):
        thresholds = [thresholds]

    if mode == EvaluationMode.PerSlice:
        # Batch up the data to load roughly the same number of slices as a volume per iteration
        data = pytorch.to_subsets(dataset=torch.utils.data.ConcatDataset(data), samples_per_subset=400)

    scores = dict()
    data_iter = tqdm(data, file=sys.stdout, desc=f'Evaluating {mode}')
    for dataset in data_iter:
        inputs, targets, _ = pytorch.dataset_to_volume(dataset)
        predictions = pytorch.predict(model=model, slices=inputs)

        assert len(predictions) >= len(targets), f'Not enough model outputs ({len(predictions)}) for the ' \
                                                 f'number of GT targets ({len(targets)})'

        assert len(targets) == len(predictions), f'Mismatch in number of predictions ({len(predictions)}) ' \
                                                 f'and number of target outputs ({len(targets)})'

        for output_id, (prediction, target) in enumerate(zip(predictions, targets)):
            if output_id not in scores:
                scores[output_id] = dict()
            for threshold in thresholds:
                prediction_scores = evaluate_prediction(prediction=prediction, target=target,
                                                        threshold=threshold, mode=mode, score_fn=score_fn)

                if threshold not in scores[output_id]:
                    scores[output_id][threshold] = list()

                scores[output_id][threshold].extend(prediction_scores)

    return scores


def output_results_to_csv(scores: Dict[Any, Dict[float, List[float]]], output_path: Path,
                          score_labels: List[str] = None):
    results = pd.DataFrame()
    for output, threshold_scores in scores.items():
        threshold_scores = {f'Threshold {threshold}': values for threshold, values in
                            threshold_scores.items()}
        combined = pd.concat({f'Output {output}': pd.DataFrame(threshold_scores)}, axis=1)
        results = pd.concat([results, combined], axis=1)

    if score_labels:
        results.index = score_labels

    results.to_csv(output_path / "results.csv")
    return results

import copy
from pathlib import Path
from typing import List, Union, Dict, Any, Tuple
import os
import numpy as np
from torch.nn import Module
from torch.utils.data import Dataset

from utils import pytorch

import torch
import logging

logger = logging.getLogger(__name__)
from utils import visualise

class CLModel:
    def __init__(self, initial_model: Module, initial_task_name: str = None):
        """
        If initial_task_name is specified then this assumes the model is trained and adds it to the set of
        trained tasks for this model.
        """
        self.initial_model = initial_model
        self.task_models = list()
        if initial_task_name is not None:
            self.add_task(model=initial_model, task_name=initial_task_name)

    def __str__(self):
        return f'{self.__name__}_{self.initial_model.__name__}'

    def add_task(self, task_name: str, model: Module):
        self.task_models.append((task_name, model))

    def previous_model(self) -> Module:
        """You could do something more sophisticated in a subclass"""
        if len(self.task_models) > 0:
            return self.task_models[-1][1]
        # Nothing has been trained so far so just return the initial model
        return self.initial_model

    def trained_tasks(self) -> List[str]:
        return [name for name, _ in self.task_models]

    def trained_models(self) -> List[Module]:
        return [model for _, model in self.task_models]

    def task_index(self, task_name: str) -> Union[int, None]:
        try:
            return self.trained_tasks().index(task_name)
        except ValueError:
            raise ValueError(f'Unknown task name: {task_name}.  Available task names: {self.trained_tasks()}')

    def task_model(self, task_name: str) -> Union[Module, None]:
        index = self.task_index(task_name=task_name)
        return self.task_models[index][1]

    def task_output_index(self, task_name: str) -> Union[int, None]:
        raise NotImplementedError('CLModel is a base class intended for inheritance by CL model '
                                  'implementations and should not be used directly')

    def number_of_model_outputs(self, task_name: str) -> Union[int, None]:
        raise NotImplementedError('CLModel is a base class intended for inheritance by CL model '
                                  'implementations and should not be used directly')

    def prepare_single_task_data(self, dataset: Dataset, task_index: int, num_outputs: int = 1) -> Dataset:
        raise NotImplementedError('CLModel is a base class intended for inheritance by CL model '
                                  'implementations and should not be used directly')

    # def prediction_and_target(self, dataset: Dataset, model_task_name: str, task_to_predict: str) \
    #         -> Tuple[np.ndarray, np.ndarray]:
    #     task_model = self.task_model(task_name=model_task_name)
    #
    #     output_index = self.task_output_index(task_name=task_to_predict)
    #     num_model_outputs = self.number_of_model_outputs(task_name=model_task_name)
    #     data = self.prepare_single_task_data(dataset=dataset, task_index=output_index,num_outputs=num_model_outputs)
    #
    #     inputs, targets, _ = pytorch.dataset_to_volume(data)
    #     predictions = pytorch.predict(model=task_model, slices=inputs)
    #
    #     output_index = self.task_output_index(task_name=task_to_predict)
    #
    #     return predictions[output_index], targets[0] #[output_index] ??????? This needs to be fixed

    def prediction_and_target(self, dataset: Dataset, model_task_name: str, active_classes) \
            -> Tuple[np.ndarray, np.ndarray]:
        task_model = self.task_model(task_name=model_task_name)
        self.active_classes = active_classes
        one_task_data = torch.utils.data.ConcatDataset(dataset)
        data_loader = torch.utils.data.DataLoader(one_task_data, batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0)

        one_task_scores = self.get_model_predictions(task_model, data_loader)
        logger.info("one task scores {}".format(one_task_scores))
        return one_task_scores

    # def prediction_and_target(self, dataset: Dataset, model_task_name: str, active_classes) \
    #         -> Tuple[np.ndarray, np.ndarray]:
    #     task_model = self.task_model(task_name=model_task_name)
    #     self.active_classes = active_classes
    #     # one_task_data = torch.utils.data.ConcatDataset(dataset)
    #
    #     # per vol evaluation
    #     for one_scan_dataset in dataset:
    #         data_loader = torch.utils.data.DataLoader(one_scan_dataset, batch_size=1,
    #                                                   shuffle=False,
    #                                                   num_workers=0)
    #         one_scan_scores = self.get_model_predictions(task_model, data_loader)
    #         logger.info("scan {} score is {}".format(one_scan_dataset.patient, one_scan_scores))
    #     # one_task_scores = self.get_model_predictions(task_model, data_loader)
    #     # logger.info("one task scores {}".format(one_task_scores))
    #     return 0 #one_task_scores

    def get_model_predictions(self, model, dataloader):
        model.eval()
        self.device = pytorch.get_device()
        with torch.no_grad():
            # data_iter = tqdm(dataloader, file=sys.stdout, desc='LwF: Evaluating all active...')

            one_task_scores = []
            iter = 0
            # for input, target in data_iter:
            for input, target, _ in dataloader:

                input, target = input.to(self.device), target.to(self.device)
                target_multi_label = target.clone()
                present_classes = list(torch.unique(target).cpu().detach().numpy())
                if len(present_classes) == 1:
                    continue  # contains only background

                target[target != present_classes[-1]] = 0  # just returning the organ i want to eval
                model_output = model(input)
                y_pred, logits, logsigma = model_output["softmaxed_seg_logits"], \
                                           model_output["seg_logits"], \
                                           model_output["logsigma"]
                dsc_score = self.compute_dsc_scores(y_pred, target)[1]  # to ignore the background score
                print("Dice score of slice {} is {}".format(iter, dsc_score))
                one_task_scores.append(dsc_score.item())

                # visualize
                # pred = torch.sigmoid(logits)
                # pred = (pred > 0.5)
                # _, pred = pred.max(1)
                _, pred = y_pred.max(1)

                # zero out the predictions that are not in the present classes  (for vis)
                zero_out = list(set(torch.unique(pred).cpu().detach().numpy()).difference(present_classes))
                for label in zero_out:
                    pred[pred == label] = 0
                # print("unique pred {}".format(torch.unique(pred)))
                # print("unique pred  and count ",torch.unique(pred,return_counts=True))
                # print("present classes {}".format(present_classes))
                path = "/home/abel@local.tmvse.com/skhawy/Canon/Code/ADP_ContinualLearning/" \
                       "Adversarial-Continual-Learning/src/checkpoints/adp_visuals"
                os.makedirs(path, exist_ok=True)
                # print("input size {}".format(input.size()))
                visualise.visualise_models_pred_results(input, target_multi_label,
                                                        pred, present_classes[-1], path, iter )
                iter += 1
            one_task_scores = np.asarray(one_task_scores).mean(axis=0)

        return one_task_scores

    def compute_dsc_scores(self, y_pred, y_true):
        """
        Evaluates the dice score for one class at a time (+ background)
        :param y_pred: softmaxed logits for the specific class
        :param y_true: target for the specific class (just 1 class at a time)
        :return: dice scores
        """
        self.smooth = 1
        y_true_one_hot = self.make_one_hot(y_true, num_classes=6)
        # if self.replay_mode == "LwF":
        present_class = list(torch.unique(y_true).detach().cpu().numpy())  # [0, the other class]
        present_class = [int(x) for x in present_class]
        head_ids = [self.active_classes.index(label) for label in present_class]
        # slicing only the channels correspondng to active class
        # present class in y_true_one hot, as it is 6 channels and when the order change for ex llung as 1st task
        # in y_true_one_hot, it will be in channel 3, but in y_pred it will be channel 1 (1st output of the model)
        y_true = y_true_one_hot[:, present_class, :, :].to(self.device)
        y_pred = y_pred[:, head_ids, :, :]

        assert y_pred.size() == y_true.size()

        if len(y_pred.size()) < 4:
            # make 4d tensor of 3d tensor by adding the channel dim
            y_pred = y_pred.unsqueeze(1)
            y_true = y_true.unsqueeze(1)

        intersection = (y_pred * y_true).sum(dim=(2, 3))
        # print(intersection)
        dsc = (2. * intersection + self.smooth) / (
                y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) + self.smooth)
        dsc = dsc.detach()
        # print("mean dsc for background and the other class", dsc.mean(dim=0))
        return dsc.mean(dim=0)

    def make_one_hot(self, tensor, num_classes=1):
        bs, _, h, w = tensor.size()
        tensor = tensor.type(torch.LongTensor)
        y_true_one_hot = torch.FloatTensor(bs, num_classes, h, w).zero_()
        y_true_one_hot = y_true_one_hot.scatter_(1, tensor, 1.0)

        return y_true_one_hot

    def train(self, training_data: Dataset, validation_data: Dataset, training_parameters: Dict[str, Any],
              model_file_path: Path, task_name: str) -> Dict[str, List[float]]:
        previous_model = self.previous_model()
        model = copy.deepcopy(previous_model)

        training_label = ';'.join(self.trained_tasks() + [task_name])

        task_index = len(self.trained_tasks())

        training_data = self.prepare_single_task_data(dataset=training_data, task_index=task_index)
        validation_data = self.prepare_single_task_data(dataset=validation_data, task_index=task_index)

        model, loss_history = self._training_imp(model=model, training_data=training_data,
                                                 validation_data=validation_data,
                                                 training_parameters=training_parameters,
                                                 model_file_path=model_file_path, task_name=task_name,
                                                 training_label=training_label)

        self.add_task(model=model, task_name=task_name)
        return loss_history

    def _training_imp(self, model: Module, training_data: Dataset, validation_data: Dataset,
                      training_parameters: Dict[str, Any], model_file_path: Path, task_name: str,
                      training_label: str) -> Tuple[Module, Dict[str, List[float]]]:
        """Override this method in subclasses with training implementation"""
        raise NotImplementedError('CLModel is a base class intended for inheritance by CL model '
                                  'implementations and should not be used directly')


def main():
    pytorch_model = Module()
    model = CLModel(initial_model=pytorch_model, initial_task_name='Task1')

    assert model.task_index('Task1') == 0
    assert model.task_model('Task1') == pytorch_model


if __name__ == '__main__':
    main()

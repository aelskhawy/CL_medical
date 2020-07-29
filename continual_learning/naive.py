import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple
from models.unet_multihead import UNetMultiHead, SSegHead
from torch.nn import Module
from torch.utils.data import Dataset
from nn.Lwf_Trainer import LwFTrainer

from continual_learning.model_base import CLModel
from models.squeezeUnet import SegHead
from nn import training
from utils.options import Options
from utils.pytorch import MaskedDatasetWrapper

logger = logging.getLogger(__name__)


class NaiveModel(CLModel):
    def __init__(self, initial_model: Module, initial_task_name: str = None):
        super(NaiveModel, self).__init__(initial_model=initial_model, initial_task_name=initial_task_name)
        self.__name__ = 'NaiveModel'

    def task_output_index(self, task_name: str) -> Union[int, None]:
        # TODO: add to general "SingleHeadModel" class?
        return 0

    def number_of_model_outputs(self, task_name: str) -> Union[int, None]:
        # TODO: add to general "SingleHeadModel" class?
        return 1

    def prepare_single_task_data(self, dataset: Dataset, task_index: int, num_outputs: int = 1) -> Dataset:
        # TODO: add to general "SingleHeadModel" class?
        # TODO: should probably check if the dataset only gives one GT output
        return dataset

    def _training_imp(self, model: Module, training_data: Dataset, validation_data: Dataset,
                      training_parameters: Dict[str, Any], model_file_path: Path, task_name: str,
                      training_label: str) -> Tuple[Module, Dict[str, List[float]]]:
        loss_history = training.train_model(model=model, training_data=training_data,
                                            validation_data=validation_data,
                                            training_parameters=training_parameters,
                                            model_file_path=model_file_path, run_label=training_label)
        return model, loss_history


class NaiveModelMultiHead(CLModel):
    def __init__(self, initial_model: Module, initial_task_name: str = None, options: Options = None):
        super(NaiveModelMultiHead, self).__init__(initial_model=initial_model,
                                                  initial_task_name=initial_task_name)
        self.__name__ = 'NaiveModelMultiHead'
        self.options = options

    def task_output_index(self, task_name: str) -> Union[int, None]:
        # TODO: add to general "MultiHeadModel" class?
        return self.task_index(task_name=task_name)

    def number_of_model_outputs(self, task_name: str) -> Union[int, None]:
        # TODO: add to general "MultiHeadModel" class?
        return self.task_index(task_name=task_name) + 1

    def prepare_single_task_data(self, dataset: Dataset, task_index: int, num_outputs: int = 1) -> Dataset:
        # TODO: add to general "MultiHeadModel" class?
        return MaskedDatasetWrapper(dataset=dataset, output_indices=[task_index], num_outputs=num_outputs)

    def _training_imp(self, model: Module, training_data: Dataset, validation_data: Dataset,
                      training_parameters: Dict[str, Any], model_file_path: Path, task_name: str,
                      training_label: str) -> Tuple[Module, Dict[str, List[float]]]:

        # I'm not 100% sure if we should handle "ideal" training within the CL class so for now I'm commenting
        # this out
        # if "ideal" in training_label:
        #     model = SqueezeUnetMultiTask(task_list=self.trained_tasks() + [task_name])

        model = self.freeze_heads(model)
        model = self.add_head(model, task_name)
        model.task_list.append(task_name)

        # Had to add it here, or add a separate train routine in training.py to call my trainer
        if self.options.replay_mode == "LwF":
            trainer = LwFTrainer(model_file_path=model_file_path, opt=self.options, label=training_label)
            loss_history = trainer.train(model=model, training_data=training_data,
                                         validation_data=validation_data,
                                         num_epochs=self.options.num_epochs,
                                         num_workers=self.options.num_workers)
            return model, loss_history

        loss_history = training.train_model(model=model, training_data=training_data,
                                            validation_data=validation_data,
                                            training_parameters=training_parameters,
                                            model_file_path=model_file_path, run_label=training_label)
        return model, loss_history

    @staticmethod
    def freeze_base(model: Module) -> Module:
        for param in model.base_model.parameters():
            param.requires_grad = False
        return model

    @staticmethod
    def freeze_heads(model: Module) -> Module:
        for head in model.seg_heads:
            for param in head.parameters():
                param.requires_grad = False
        return model

    @staticmethod
    def add_head(model: Module, task: str) -> Module:

        new_head = SegHead(head_task=task)
        model.seg_heads.append(new_head)

        return model

class LwFMultiHead(CLModel):
    def __init__(self, initial_model: Module, initial_task_name: str = None, options: Options = None):
        super(LwFMultiHead, self).__init__(initial_model=initial_model,
                                                  initial_task_name=initial_task_name)
        self.__name__ = 'LwfMultiHead'
        self.options = options
        self.roi_order = ['background', 'spinal_cord', 'r_lung', 'l_lung', 'heart', 'oesophagus']

    def task_output_index(self, task_name: str) -> Union[int, None]:
        # TODO: add to general "MultiHeadModel" class?
        return self.task_index(task_name=task_name)

    def number_of_model_outputs(self, task_name: str) -> Union[int, None]:
        # TODO: add to general "MultiHeadModel" class?
        return self.task_index(task_name=task_name) + 1

    def prepare_single_task_data(self, dataset: Dataset, task_index: int, num_outputs: int = 1) -> Dataset:
        # print("len of dataset", len(dataset))
        return dataset

    def _training_imp(self, model: Module, training_data: Dataset, validation_data: Dataset,
                      training_parameters: Dict[str, Any], model_file_path: Path, task_name: str,
                      training_label: str) -> Tuple[Module, Dict[str, List[float]]]:

        if self.options.replay_mode == "LwF":
            # model = self.freeze_all(model)
            model = self.freeze_heads(model)
            model = self.add_head(model, task_name)
            model.task_list.append(task_name)
            active_tasks_idx = [self.roi_order.index(task) for task in model.task_list]
        else:
            # ideal case
            active_tasks_idx = [0]

        # self.active_classes = active_tasks_idx
        prev_model = self.previous_model()
        print("current model tasks", model.task_list)
        print("previous model tasks", prev_model.task_list)
        # model = self.prepare_new_model(prev_model=model, freeze_prev_heads=True)

        trainer = LwFTrainer(model_file_path=model_file_path, opt=self.options, label=training_label)
        loss_history = trainer.train(model=model, prev_model=prev_model,
                                     training_data=training_data,
                                     validation_data=validation_data,
                                     num_epochs=self.options.num_epochs,
                                     active_classes=active_tasks_idx)
        return model, loss_history

    # @staticmethod
    # def freeze_all(model: Module) -> Module:
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     return model

    @staticmethod
    def freeze_heads(model: Module) -> Module:
        # As my model starts with a background head, i don't wanna freeze this before starting any training
        if model.task_list[-1] == 'background':
            return model
        for head in model.seg_heads:
            print("========> Freezing", head) # ?????
            for param in head.parameters():
                param.requires_grad = False
        return model

    @staticmethod
    def add_head(model: Module, task: str) -> Module:

        new_head = SSegHead(head_task=task)
        for p in new_head.parameters():
            p.requires_grad = True
        model.seg_heads.append(new_head)

        return model

    def get_trainable_params(self, model):
        model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        return model_params

    # def prepare_new_model(self, prev_model, freeze_prev_heads=True):
    #     """
    #         CL function
    #         This function loads the prev model into the new model (which has more heads to accommodate
    #         new classes) and freezes the prev heads if needed
    #     :param new_model:
    #     :param freeze_prev_head: bool
    #     :return:
    #     """
    #     print("task list", prev_model.task_list)
    #     new_model = UNetMultiHead(in_channels=1, init_features=self.options.nfc, task_list=prev_model.task_list,
    #                               activation=None)
    #     new_model = new_model.to('cuda')
    #     missing, unexp = new_model.load_state_dict(prev_model.state_dict(), strict=False)
    #     if freeze_prev_heads:
    #         for name, p in new_model.named_parameters():
    #             if "head" in name:
    #                 # to avoid freezing the recently added head
    #                 if "head_" + str(prev_model.task_list[-1]) not in name:
    #                     print("freezing layer", name)
    #                     p.requires_grad = False
    #
    #     return new_model
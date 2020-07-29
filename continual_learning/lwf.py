import copy
import logging
from pathlib import Path
from typing import Callable, Dict

import torch

from nn.training import Trainer
from utils import pytorch

logger = logging.getLogger(__name__)


class LwfModel:
    def __init__(self, original_model: torch.nn.Module, loss_fn: Callable, batch_size: int,
                 learning_rate: float = 0.0001, device=None):
        self.original_model = original_model
        self.previous_models = [original_model]
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device if device else pytorch.get_device()

    @staticmethod
    def id() -> str:
        return 'lwf'

    def compute_losses(self, prediction, target) -> Dict[str, float]:
        return {'loss': self.loss_fn(prediction, target)}

    @staticmethod
    def update_model(losses: Dict[str, torch.Tensor]) -> None:
        losses['loss'].backward()

    def train(self, training_data: torch.utils.data.Dataset, validation_data: torch.utils.data.Dataset,
              num_epochs: int, model_file_path: Path, task_name: str):

        # TODO: Add new heads/freeze appropriate weights for CL
        model = copy.deepcopy(self.previous_models[-1])

        # TODO: Use loss and updates for continual learning
        compute_loss_fn = self.compute_losses
        update_model_fn = self.update_model

        optimiser = torch.optim.Adam(params=model.parameters(), lr=self.learning_rate, weight_decay=1e-3)

        trainer = Trainer(compute_loss_fn=compute_loss_fn, update_model_fn=update_model_fn,
                          batch_size=self.batch_size, optimiser=optimiser, model_file_path=model_file_path,
                          device=self.device, label=task_name)

        loss_history = trainer.train(model=model, training_data=training_data,
                                     validation_data=validation_data, num_epochs=num_epochs)

        self.previous_models.append(model)
        return loss_history

import logging
import sys
import argparse
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Callable, List
from nn.Lwf_Trainer import LwFTrainer
import torch
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import pytorch, paths

logger = logging.getLogger(__name__)


def basic_training_fn(loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], model: torch.nn.Module,
                      input: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    prediction = model(input)

    losses = {'loss': loss_fn(prediction, target)}

    for loss in losses.values():
        loss.backward()

    return losses


def basic_validation_fn(loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], model: torch.nn.Module,
                        input: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        prediction = model(input)

        losses = {'loss': loss_fn(prediction, target)}

    return losses


class Trainer:
    def __init__(self, training_fn: Callable[
        [torch.nn.Module, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]],
                 validation_fn: Callable[
                     [torch.nn.Module, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]],
                 batch_size: int, optimiser: torch.optim.Optimizer, model_file_path: Path, label: str = None,
                 device=None):
        self.training_fn = training_fn
        self.validation_fn = validation_fn
        self.batch_size = batch_size
        self.optimiser = optimiser
        self.model_file_path = model_file_path
        self.device = device if device else pytorch.get_device()
        self.label = label

    @staticmethod
    def _combine_losses(training_losses: Dict[str, float], validation_losses: Dict[str, float]) -> \
            Dict[str, float]:
        training_losses = {f'training_{name}': value for name, value in training_losses.items()}
        validation_losses = {f'validation_{name}': value for name, value in validation_losses.items()}

        return {**training_losses, **validation_losses}

    @staticmethod
    def _output_losses(training_losses: Dict[str, float], validation_losses: Dict[str, float],
                       # writer: SummaryWriter,
                       epoch: int):
        common_losses = set(training_losses.keys()).intersection(set(validation_losses.keys()))
        for loss_name in common_losses:
            scalars = {'training': training_losses[loss_name], 'validation': validation_losses[loss_name]}
            writer.add_scalars(loss_name.capitalize(), scalars, epoch)

        for loss_name in training_losses.keys():
            if loss_name not in common_losses:
                writer.add_scalar(loss_name.capitalize(), training_losses[loss_name], epoch)

        for loss_name in validation_losses.keys():
            if loss_name not in common_losses:
                writer.add_scalar(loss_name.capitalize(), validation_losses[loss_name], epoch)

    def train(self, model: torch.nn.Module, training_data: torch.utils.data.Dataset,
              validation_data: torch.utils.data.Dataset, num_epochs: int, num_workers : int) -> Dict[str, List[float]]:


        logger.info(f'Training on {self.device}, batch size = {self.batch_size}, num_workers = {num_workers}')

        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_folder = f'{self.label}_{current_time}' if self.label else current_time
        writer = SummaryWriter(log_dir=paths.output_data_root() / 'runs' / run_folder)

        training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=self.batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers)
        validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=self.batch_size,
                                                             shuffle=False,
                                                             num_workers=num_workers)
        model = model.to(self.device)

        early_stopping_callback = pytorch.EarlyStopping(loss_to_monitor='validation_loss', verbose=True,
                                                        model_file_path=self.model_file_path)

        loss_history = dict()
        for epoch in range(1, num_epochs + 1):
            logger.info(f'Epoch {epoch} - Training')
            training_losses = self.train_one_epoch(model=model, data_loader=training_data_loader)
            logger.info(f'Epoch {epoch} - Validation')
            validation_losses = self.test(model=model, data_loader=validation_data_loader)

            # self._output_losses(training_losses=training_losses, validation_losses=validation_losses,
            #                     writer=writer, epoch=epoch)

            losses = self._combine_losses(training_losses=training_losses,
                                          validation_losses=validation_losses)

            loss_strings = [f'{name} = {value:.6f}' for name, value in losses.items()]
            logger.info(f'Epoch {epoch} - {", ".join(loss_strings)}')

            for name, value in losses.items():
                if name not in loss_history:
                    loss_history[name] = list()
                loss_history[name].append(value)

            # if hasattr(writer, 'flush'):
            #     writer.flush()
            if early_stopping_callback(losses, model):
                logger.info(
                    f'No improvement seen in {early_stopping_callback.patience} epochs.  Stopping training')
                break

        # writer.close()

        if self.device.type != 'cpu':
            # Free up any GPU memory once we're done
            torch.cuda.empty_cache()

        return loss_history

    def train_one_epoch(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> \
            Dict[str, float]:
        model.train()
        data_iter = tqdm(data_loader, file=sys.stdout)
        total_losses = dict()
        for input, target, metadata in data_iter:
            self.optimiser.zero_grad()

            input = input.to(self.device)
            if isinstance(target, list):
                target = [t.to(self.device) for t in target]
            else:
                target = target.to(self.device)

            # Model-specific
            losses = self.training_fn(model=model, input=input, target=target)

            with torch.no_grad():
                for name, loss in losses.items():
                    if name not in total_losses:
                        total_losses[name] = 0.0
                    total_losses[name] += loss.item()

            self.optimiser.step()

        average_losses = {name: total_loss / len(data_loader) for name, total_loss in total_losses.items()}

        return average_losses

    def test(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        model.eval()
        data_iter = tqdm(data_loader, file=sys.stdout, desc='Testing model')
        total_losses = dict()
        with torch.no_grad():
            for input, target, metadata in data_iter:
                input = input.to(self.device)
                if isinstance(target, list):
                    target = [t.to(self.device) for t in target]
                else:
                    target = target.to(self.device)

                # Model-specific
                losses = self.validation_fn(model=model, input=input, target=target)

                for name, loss in losses.items():
                    if name not in total_losses:
                        total_losses[name] = 0.0
                    total_losses[name] += loss.item()

        average_losses = {name: total_loss / len(data_iter) for name, total_loss in total_losses.items()}

        return average_losses



def train_model(model: torch.nn.Module, training_data: torch.utils.data.Dataset,
                validation_data: torch.utils.data.Dataset, training_parameters, model_file_path: Path,
                run_label: str) -> Dict[str, List[float]]:
    """

    :param model:
    :param training_data:
    :param validation_data:
    :param training_parameters:
    :param model_file_path:
    :param run_label:
    :return:
    """
    optimiser_fn = training_parameters['optimiser']
    learning_rate = training_parameters['learning_rate']
    weight_decay = training_parameters['weight_decay']
    batch_size = training_parameters['batch_size']
    num_epochs = training_parameters['num_epochs']
    loss_fn = training_parameters['loss']
    num_workers = training_parameters['num_workers']
    optimiser = optimiser_fn(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    training_fn = partial(basic_training_fn, loss_fn=loss_fn)
    validation_fn = partial(basic_validation_fn, loss_fn=loss_fn)

    trainer = Trainer(training_fn=training_fn, validation_fn=validation_fn,
                      batch_size=batch_size, optimiser=optimiser,
                      model_file_path=model_file_path, label=run_label)

    return trainer.train(model=model, training_data=training_data,
                         validation_data=validation_data,
                         num_epochs=num_epochs, num_workers=num_workers)

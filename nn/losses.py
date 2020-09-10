import logging
import sys
from typing import List, Callable

import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from datasets import AAPM
logger = logging.getLogger(__name__)


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = pred * target
    return 1.0 - (2.0 * intersection.sum()) / (pred.sum() + target.sum() + sys.float_info.epsilon)


def dice_loss_with_missing_values(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target == -1
    pred = pred[~mask]
    target = target[~mask]
    return dice_loss(pred=pred, target=target)


def bc_loss(y_pred, y_true):
    """

    :param y_pred:
    :param y_true:
    :return:
    """
    return F.binary_cross_entropy(y_pred, y_true)  # because sigmoid output not logit


def combined(y_pred, y_true):
    return dice_loss(y_pred, y_true) + bc_loss(y_pred, y_true)


class ClLoss(nn.Module):
    def __init__(self):
        super(ClLoss, self).__init__()
        self.KD_temp = 1e3
        self.smooth = 1.0
        self.device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.kd_damping_factor = 1 #0.1
        self.bce_factor = 1
        print("kd damping {} , bce factor {}".format(self.kd_damping_factor, self.bce_factor))

    def forward(self, y_true, current_scores, previous_score, loss_weights, mode):
        """

        :param y_true: the ground Truth, (Batch x num_classes x H x W)  (not 1 hot) for now
        :param current_scores: logits from the current active model
        :param previous_score: logits from the previous model
        :param loss_weights:
        :param mode: offline or LwF or DGR
        :return: losses and dsc scores
        """
        #
        # assert y_pred.size() == y_true.size()
        losses = dict()
        weighted_bce_loss = self.get_bce_loss(current_scores, y_true, loss_weights, mode)

        # Not none, means in reply mode
        if previous_score is not None:
            # weighted_KD_loss = self.get_kD_loss(current_scores, previous_score, loss_weights, active_classes)
            criterion = nn.MSELoss()
            # compute KD loss for all channels except the last one which represents the newly added class
            # ignore background in kd loss calculation
            weighted_KD_loss = criterion(current_scores[:, 1:-1, :, :], previous_score[:, 1:, :, :])
            total_loss = weighted_bce_loss * self.bce_factor + weighted_KD_loss * self.kd_damping_factor
        else:
            weighted_KD_loss = torch.tensor([0]).type(torch.FloatTensor).to(self.device)
            total_loss = weighted_bce_loss

        losses['bce_loss'] = weighted_bce_loss * self.bce_factor
        losses["kd_loss"] = weighted_KD_loss * self.kd_damping_factor
        losses['total_loss'] = total_loss
        return losses


    def get_bce_loss(self, current_scores, y_true, loss_weights, mode):
        """

        :param current_scores: prediction logits
        :param y_true: true label
        :param loss_weights: weight the losses (just in sequential offline training)
        :param active_classes: list of the active heads' ids
        :param mode: offline or LwF
        :return:
        """
        active_classes = list(torch.unique(y_true).cpu().detach().numpy())
        n_model_heads = current_scores.size()[1]
        # That means y_pred is either 2 channels if 1st task or more starting from 2nd task
        if mode != "ideal":
            if n_model_heads == 2:  # 1st task + background
                loss_weights = [0.0002, 1]
            else:  # only the last head will be used for bce loss and the rest for KD loss
                loss_weights = [1]
                active_classes = active_classes[-1]

        else:
            active_classes = list(np.arange(len(AAPM.all_organs())+1)) # this will fail if the task order has changed

        loss_weights = torch.tensor(loss_weights, dtype=torch.float, requires_grad=True).to(self.device)
        if not isinstance(active_classes, list):
            active_classes = [active_classes]

        loss = 0
        # print("active classes loss func {} , loss_weights {}".format(active_classes, loss_weights))
        for i, label in enumerate(active_classes):  # loop over the number of heads
            true_label = y_true == label
            # i need only the last head for bce in case of LwF, otherwise i'll computer over all
            # the heads if offline or task 1
            if len(active_classes) == 1:
                # Not task 1 or offline, take the last head
                head_idx = -1
            else:
                head_idx = i

            one_loss = self.bce_criterion(current_scores[:,head_idx, :, :].squeeze(),
                                          true_label.type(torch.FloatTensor).squeeze().to(self.device))

            loss += one_loss * loss_weights[i]

        return loss


def to_multioutput(loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) \
        -> Callable[[List[torch.Tensor], List[torch.Tensor]], torch.Tensor]:
    def multioutput_loss(predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> torch.Tensor:
        if not isinstance(predictions, list):
            # Handle the single prediction case
            predictions = [predictions]

        if not isinstance(targets, list):
            # Handle the single target case
            targets = [targets]

        loss = Variable(torch.zeros(1).cuda())
        if len(predictions) != len(targets):
            min_index = min(len(predictions), len(targets))
            logger.warning(
                f'Mismatch in number of predictions ({len(predictions)}) and targets ({len(targets)}).  Loss '
                f'will only be computed for predictions {",".join([str(i) for i in range(0, min_index)])}.')
        for prediction, target in zip(predictions, targets):
            loss += loss_fn(prediction, target)

        return loss

    return multioutput_loss

# if __name__ == '__main__':
#
#     ##Test masking loss
#     import numpy as np
#     target = np.zeros((3,10,10))
#     target[1] = np.ones((10, 10))
#     target[2] = np.zeros((10, 10)) - 1
#     pred =np.zeros((3,10,10))
#     pred[1] = np.ones((10, 10)) - 0.2
#     pred[2] =  np.ones((10, 10)) - 0.2
#
#
#     target = torch.from_numpy(target)
#
#     pred = torch.from_numpy(pred)
#
#     d = dice_loss_with_missing_values(pred, target)
#
#     print("Dice", d)

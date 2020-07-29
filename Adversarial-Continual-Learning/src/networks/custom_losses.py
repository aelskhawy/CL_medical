import torch
import sys
import torch.nn as nn


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = pred * target
    return 1.0 - (2.0 * intersection.sum()) / (pred.sum() + target.sum() + sys.float_info.epsilon)

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-6
        # ROI_ORDER = ['Background', 'SpinalCord', 'Lung_R', 'Lung_L', 'Heart', 'Esophagus']
        # self.loss_weights = loss_weights
        self.device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')

    def forward(self, y_pred, y_true):

        assert y_pred.size() == y_true.size()
        intersection = (y_pred * y_true).sum(dim=(2, 3))  # torch.Size([Batch, n_classes])
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) + self.smooth
        )

        # Get the mean over the batch
        loss = (1.0 - dsc).mean(dim=0)  #torch.Size([n_classes])
        # weighted_sum = (loss * torch.tensor(loss_weights, dtype=torch.float, requires_grad=True).to(self.device)).sum()  # scalar
        return loss.sum()

    # def forward(self, y_pred, y_true):
    #     assert y_pred.size() == y_true.size()
    #     intersection = y_pred * y_true
    #     return 1.0 - (2.0 * intersection.sum()) / (y_pred.sum() + y_true.sum() + sys.float_info.epsilon)

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

        self.device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')
        self.criterion = nn.BCEWithLogitsLoss()
        self.alpah =0.25
        self.gamma=2.0
    def forward(self, y_pred, y_true):

        # print("test self loss weights", loss_weights)
        # loss_list = []
        loss = 0
        true_label = y_true
        true_label = true_label.type(torch.FloatTensor).squeeze().to(self.device)
        # print(i, true_label.shape, y_pred[i].shape)
        # print(true_label, y_pred[i])
        per_ent_ce = self.criterion(y_pred.squeeze(), true_label)
        prediction_probabilities = torch.sigmoid(y_pred.squeeze())
        p_t = ((true_label * prediction_probabilities) +
               ((1 - true_label) * (1 - prediction_probabilities)))
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)

        focal_cross_entropy_loss = (modulating_factor * per_ent_ce)

        loss = focal_cross_entropy_loss.sum(dim=(1,2)).mean() #.mean(dim=0).sum()

        return loss
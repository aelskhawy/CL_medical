
# from torch.utils.tensorboard import SummaryWriter
import logging
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Callable, List
from nn.losses import ClLoss
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy
from utils import pytorch, paths
import numpy as np
from collections import OrderedDict
from datasets.all_data import get_data, DataQuery
from utils.data import Split
from datasets.all_data import DataQuery

logger = logging.getLogger(__name__)


class LwFTrainer:
    def __init__(self,
                 model_file_path: Path,
                 opt=None,
                 label: str = None):
        self.opt = opt
        self.device = pytorch.get_device()
        self.model_file_path = model_file_path
        self.replay_mode = self.opt.replay_mode
        self.label = label

        self.batch_size = self.opt.batch_size
        self.lr = self.opt.lr
        self.loss_func = ClLoss()

        self.fine_tune = self.opt.fine_tune
        self.prev_model = None

        #### TO BE REMVOED LATER
        self.train_loss = []
        self.loss_names = ["Total_loss", "bce_loss", "KD_loss"]
        self.all_tasks = ['background', 'spinal_cord', 'r_lung', 'l_lung', 'heart', 'oesophagus']
        if self.opt.dataset == 'structseg':
            self.all_tasks = ['background', "l_lung", "r_lung", "heart", "oesophagus", "trachea", "spinal_cord"]

        self.val_dsc_scores_list = []
        self.val_loss = []
        self.epoch = 0
        self.loss_weights = [0.00026391335252852157, 0.34153133637858485, 0.011513810817807897,
                             0.014717132000521804, 0.038526956519617024,0.59344685093094]

        # used for model saving
        self.best_val_dice = 0

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

    def model_on_cuda(self):
        self.model.to(self.device)

    def get_trainable_parameters(self):
        model_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        return model_params

    def _init_optimizer(self):
        trainable_model_params = self.get_trainable_parameters()
        print("trainable parameters len", len(trainable_model_params))
        self.optimizer = optim.Adam(trainable_model_params, lr=self.lr, weight_decay=self.opt.weight_decay)

    def create_logger(self):
        dirname = paths.training_output_root() / "LwF" / str(self.opt.name)
        dirname.mkdir(parents=True, exist_ok=True)

        log_file = dirname / "logging.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler("{}".format(log_file)),
                logging.StreamHandler()
            ])
        return logging.getLogger()

    def _init_logger(self):

        self.logger = create_logger(self.opt)
        self.logger.info("Using {}".format(self.device))

        # The reported values here for the specific channels are the dsc scores
        ROIs = ["Background", "Spine", "RLung", "LLung", "Heart", "Esophagus"]

        logging_names = "Learning_mode | Task_ID | global_step | "
        logging_names += "epoch | iter | lr | "
        logging_names += ''.join("{} | ".format(loss) for loss in self.loss_names)
        logging_names += ''.join("{} | ".format(roi) for roi in ROIs)
        logging_names += "Dsc_mean_noBG | "
        logging_names += "flag"
        # The reported values here for the specific channels are the dsc scores
        print(logging_names)
        shosh_logger = DFLogger(logging_names, sep="|", log_root_dir=self.log_dir)
        self.shosh_logger = shosh_logger


    def get_replay_batch(self, x):
        """
        Receives the model (with the extra head) and get the response of the previous model (a freezed copy
        of the current model before starting training)
        :param x:
        :return:
        """
        # For the 1st task (n_head=2) or offline mode, we don't need replay batch
        if (len(self.model.seg_heads) == 2) or (self.replay_mode == 'ideal'):
            y_r_scores = None
            return y_r_scores

        elif (self.replay_mode == 'LwF') and (len(self.model.seg_heads) > 2):
            # print("====== Obtaining Replay batch =====")
            previous_model = self.prev_model
            previous_model.eval()
            with torch.no_grad():
                model_output = previous_model(x.to(self.device))
                y_r_scores = model_output["seg_logits"]

        # # Modifying the output (as the prev model is a copy of the new model with the new untrained head)
        # y_r_scores = y_r_scores[:, :-1, :, :]
        return y_r_scores

    def train_one_epoch(self):
        self.model.train()
        data_iter = tqdm(self.training_data_loader, file=sys.stdout)
        total_losses = dict()

        # sanity check
        # print("len seg heads", len(self.model.seg_heads))
        # for p in self.model.seg_heads[-1].parameters():
        #     print(p.requires_grad)

        for input, target, _ in data_iter:
            self.optimizer.zero_grad()
            self.model.zero_grad()
            input, target = input.to(self.device), target.to(self.device)
            # print("unique target", torch.unique(target))
            self.iteration += 1
            y_replay_scores = self.get_replay_batch(x=input)

            with torch.set_grad_enabled(True):
                model_output = self.model(input)
                y_pred, logits, logsigma = model_output["softmaxed_seg_logits"], \
                                           model_output["seg_logits"], \
                                           model_output["logsigma"]

                losses = self.loss_func(target,      # true labels not one hot
                                       logits,      # w/o softmax
                                       y_replay_scores,  # previous scores,
                                       self.loss_weights,
                                       self.replay_mode)

                # Backprop only the BCE Loss corresponds to fine tuning
                if self.fine_tune:
                    # print("=========> Fine Tuning <===========")
                    losses['bce_loss'].backward()
                else:

                    # losses['total_loss'] = 1 / (2 * torch.exp(logsigma[0])) * losses['bce_loss'] + \
                    #                        1 / (2 * torch.exp(logsigma[1])) * losses['kd_loss'] + \
                    #                        logsigma[0] + logsigma[1]

                    losses['total_loss'].backward()
                self.optimizer.step()

                # self.train_loss.append([losses['total_loss'].item(), losses['bce_loss'].item(), losses['kd_loss'].item()])
                # self.log_train_stats()

            with torch.no_grad():
                for name, loss in losses.items():
                    if name not in total_losses:
                        total_losses[name] = 0.0
                    total_losses[name] += loss.item()



        average_losses = {name: total_loss / len(self.training_data_loader)
                          for name, total_loss in total_losses.items()}

        # if (self.iteration + 1) % 10 == 0:
        #     print("bce weight {} , kd weight {}".format(torch.exp(logsigma[0]), torch.exp(logsigma[1])))
        return average_losses


    def eval_one_epoch(self):
        self.model.eval()
        data_iter = tqdm(self.validation_data_loader, file=sys.stdout, desc='LwF: Evaluating model...')
        total_losses = dict()

        with torch.no_grad():
            for input, target, _ in data_iter:
                # input, target = data['slice'], data['gt']
                input, target = input.to(self.device), target.to(self.device)
                if len(list(torch.unique(target).cpu().detach().numpy())) == 1:
                    continue  # contains only background
                # print("torch unique in eval ", torch.unique(target))
                y_replay_scores = self.get_replay_batch(x=input)
                model_output = self.model(input)
                y_pred, logits, logsigma = model_output["softmaxed_seg_logits"], \
                                           model_output["seg_logits"], \
                                           model_output["logsigma"]

                losses = self.loss_func(target,  # true labels not one hot
                                        logits,  # w/o softmax
                                        y_replay_scores,  # previous scores,
                                        self.loss_weights,  # no loss weights for LwF or the 1st task
                                        self.replay_mode)
                # exit()
                val_dsc_scores = self.compute_dsc_scores(y_pred, target)

                self.val_loss.append([losses['total_loss'].item(), losses['bce_loss'].item(), losses['kd_loss'].item()])
                self.val_dsc_scores_list.append(val_dsc_scores)

                for name, loss in losses.items():
                    if name not in total_losses:
                        total_losses[name] = 0.0
                    total_losses[name] += loss.item()

            average_losses = {name: total_loss / len(data_iter) for name, total_loss in total_losses.items()}

            self.val_dsc_scores_list = np.asarray([(tensor.detach().cpu().numpy()) for
                                                   tensor in self.val_dsc_scores_list]).mean(axis=0)

            self.best_val_dice = self.val_dsc_scores_list[1]  # 0 for background
            self.log_val_stats()

            return average_losses

    def eval_active_tasks(self):
        self.model.eval()
        active_tasks = [self.all_tasks[id] for id in self.active_classes]
        active_tasks = [DataQuery(tasks=organ) for organ in active_tasks[1:]] # to ignore the background

        with torch.no_grad():
            for data_query  in active_tasks:
                one_task_volumes = get_data(query=data_query, split=Split.EarlyStopping,
                                                       debug_mode=self.opt.debug_mode, options=self.opt)
                one_task_data = torch.utils.data.ConcatDataset(one_task_volumes)
                one_task_dataloader = torch.utils.data.DataLoader(one_task_data, batch_size=self.batch_size,
                                                                 shuffle=False,
                                                                 num_workers=self.opt.num_workers)

                data_iter = tqdm(one_task_dataloader, file=sys.stdout, desc='LwF: Evaluating all active...')

                one_task_scores = []
                for input, target, _ in data_iter:
                    input, target = input.to(self.device), target.to(self.device)
                    if len(list(torch.unique(target).cpu().detach().numpy())) == 1:
                        continue  # contains only background
                    # print("unique in eval active tasks", torch.unique(target))
                    model_output = self.model(input)
                    y_pred, logits, logsigma = model_output["softmaxed_seg_logits"], \
                                               model_output["seg_logits"], \
                                               model_output["logsigma"]
                    val_dsc_score = self.compute_dsc_scores(y_pred, target)[1]  # to ignore the background score
                    one_task_scores.append(val_dsc_score.item())
                one_task_scores = np.asarray(one_task_scores).mean(axis=0)

                self.val_dsc_scores_list.append(one_task_scores)

        self.log_val_stats()

    def train(self, model: torch.nn.Module, prev_model: torch.nn.Module, training_data: torch.utils.data.Dataset,
              validation_data: torch.utils.data.Dataset, num_epochs: int, active_classes: list) -> Dict[str, List[float]]:
        """
            Responsible for the main training flow
        :return:
        """
        num_workers = self.opt.num_workers #min(self.batch_size, 2)
        logger.info(f'Training on {self.device}, batch size = {self.batch_size}, num_workers = {num_workers}')
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_folder = f'{self.label}_{current_time}' if self.label else current_time

        # writer = SummaryWriter(log_dir=paths.output_data_root() / 'runs' / run_folder)

        self.training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=self.batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers)

        self.validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=self.batch_size,
                                                             shuffle=False,
                                                             num_workers=num_workers)
        print("Len training data loader", len(self.training_data_loader))
        print("Len val data loader", len(self.validation_data_loader))

        self.model = model.to(self.device)
        self.prev_model = prev_model.to(self.device)
        self._init_optimizer()
        self.active_classes = active_classes
        print("active_classes input", active_classes)

        early_stopping_callback = pytorch.EarlyStopping(loss_to_monitor='validation_total_loss', verbose=True,
                                                        model_file_path=self.model_file_path, patience=15)

        loss_history = dict()
        self.iteration = 0
        self.eval_active_tasks()
        local_best_val_dice = 0
        # only 10 epochs for the 1st organ to avoid the model getting stuck in the space of the 1st model's weights
        num_epochs = num_epochs if len(active_classes) > 2 else 30 #10 for all except oesophagus
        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            logger.info(f'Epoch {epoch} - Training')
            training_losses = self.train_one_epoch()
            logger.info(f'Epoch {epoch} - Validation')
            validation_losses = self.eval_one_epoch()

            # save the model that gives the highest dice, just for reference later
            if self.best_val_dice > local_best_val_dice:
                local_best_val_dice = self.best_val_dice
                self.save_best_val_dice(self.model, epoch)
                print("saving best validation dice model at epoch {}".format(epoch))

            if not self.replay_mode == 'ideal' and epoch% 5 == 0:
                self.eval_active_tasks()
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
            if early_stopping_callback(losses, self.model):
                # logger.info(
                #     f'No improvement seen in {early_stopping_callback.patience} epochs.  Stopping training')
                logger.info(
                    f'No improvement seen in {early_stopping_callback.patience} epochs.')
                # break

            # save the model each epoch - for debugging purposes
            # self.save_each_epoch(self.model, epoch)
        # writer.close()



        if self.device.type != 'cpu':
            # Free up any GPU memory once we're done
            torch.cuda.empty_cache()

        return loss_history

    def save_best_val_dice(self, model, epoch):
        '''Saves model when validation loss decrease.'''
        self.model_file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, str(self.model_file_path) + "_best_dice")
        torch.save(model.state_dict(), str(self.model_file_path) + "_best_dice.state_dict")

    # def save_each_epoch(self, model, epoch):
    #     '''Saves model when validation loss decrease.'''
    #     self.model_file_path.parent.mkdir(parents=True, exist_ok=True)
    #     torch.save(model, str(self.model_file_path)+"_{}".format(epoch))
    #     torch.save(model.state_dict(), str(self.model_file_path)+"_{}.state_dict".format(epoch))


    def get_current_losses(self, errors):
        """Return training losses / errors"""
        errors_ret = OrderedDict()
        for i, name in enumerate(self.loss_names):
            errors_ret[name] = errors[i]
        return errors_ret

    def log_train_stats(self):
        """
            Uses the self.train_loss list to extract the following values
            total_loss, dsc_loss, kd_loss
            also reports the mean in self.train_dsc_scores_list
        :return: Noting
        """
        # if (self.iteration + 1) % 10 == 0:
        train_stats = np.asarray(self.train_loss).mean(axis=0)
        train_errors = self.get_current_losses(train_stats)
        self.print_current_train_stats(train_errors)
        self.train_loss = []

    def print_current_train_stats(self, errors):
        """
            prints the train stats to the stdout and a file
        :param stats:
        :return:
        """
        message = 'train, epoch: %d' % (self.epoch)
        for k, v in errors.items():
            message += '%s: %.3f, ' % (k, v)
        print(message)
        print(message, file=open("./mylogging.log", "a"))

    def make_one_hot(self, tensor, num_classes=1):
        bs, _, h, w = tensor.size()
        tensor = tensor.type(torch.LongTensor)
        y_true_one_hot = torch.FloatTensor(bs, num_classes, h, w).zero_()
        y_true_one_hot = y_true_one_hot.scatter_(1, tensor, 1.0)

        return y_true_one_hot

    def compute_dsc_scores(self, y_pred, y_true):
        """
        Evaluates the dice score for one class at a time (+ background)
        :param y_pred: softmaxed logits for the specific class
        :param y_true: target for the specific class (just 1 class at a time)
        :return: dice scores
        """
        self.smooth = 1
        y_true_one_hot = self.make_one_hot(y_true, num_classes=self.opt.num_classes)  # 6 for appm , 7 for structseg
        # if self.replay_mode == "LwF":
        present_class = list(torch.unique(y_true).detach().cpu().numpy())  # [0, the other class]
        present_class = [int(x) for x in present_class]
        # for order B for ex active classes = [0,5], head_ids = [0,1]
        head_ids = [self.active_classes.index(label) for label in present_class]
        # slicing only the channels correspondng to active class
        # present class in y_true_one hot, as it is 6 channels and when the order change for ex llung as 1st task
        # in y_true_one_hot, it will be in channel 3, but in y_pred it will be channel 1 (1st output of the model)
        y_true = y_true_one_hot[:, present_class, :, :].to(self.device)
        y_pred = y_pred[:, head_ids, :, :]

        # print("present class {} , head ids {}".format(present_class, head_ids))
        # print("unique y_true {}, unique y_pred {}".format(torch.unique(y_true), torch.unique(y_pred)))

        # for i in range(6):
        #     print("unique true in channel {} is {}".format(i, torch.unique(y_true_one_hot[:, i, :, :])))
        #
        # for i in range(y_true.size()[1]):
        #     print("unique in my selection channel {}  is  {}".format(i, torch.unique(y_true[:, i, : ,:])))
        # exit()
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

    def log_val_stats(self):
        """
           Uses the self.val_loss list to extract the following values
           total_loss, dsc_loss, kd_loss
           also reports the mean in self.val_dsc_scores_list
        :param dataset_key: a key (current, previous, combined) for the dataset reported in val
       :return: Noting
       """

        if self.val_loss == []:
            val_errors = {"t1": 0, "t2":0, "t3":0}
        else:
            val_stats = np.asarray(self.val_loss).mean(axis=0)
            val_errors = self.get_current_losses(val_stats)
        val_dsc_scores = np.asarray(self.val_dsc_scores_list)

        self.print_current_val_stats(val_errors, scores=val_dsc_scores)
        self.val_loss = []
        self.val_dsc_scores_list = []

    def print_current_val_stats(self, errors, scores):

        message = 'val, epoch: %d ' % (self.epoch)
        # Add the losses to message
        for k, v in errors.items():
            message += '%s: %.3f, ' % (k, v)

        # Add the scores to message
        message += 'mean_dsc/channel {}, '.format(scores)
        message += 'mean_dsc_no_bg {} '.format(scores.mean())
        print("###"*50)
        print(message)
        print("###" * 50)
        print(message, file=open("./mylogging.log", "a"))

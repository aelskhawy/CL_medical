# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import sys, time, os
import numpy as np
import pandas as pd
import torch
import copy
import utils

from copy import deepcopy
from tqdm import tqdm

sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F
from networks.custom_losses import DiceLoss, FocalLoss

from networks.discriminator import Discriminator
from dataloaders.all_data import get_data, DataQuery, Split
from typing import List, Union, Dict, Tuple
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
import logging
logger = logging.getLogger(__name__)

class ACL(object):

    def __init__(self, model, tasks, args, network):
        self.args=args
        self.nepochs=args.nepochs
        self.batch_size = args.batch_size
        self.tasks = tasks #[0]  # One Dataquery containing all organs

        self.ROI_order = ['spinal_cord', 'r_lung', 'l_lung', 'heart', 'oesophagus']
        # optimizer & adaptive lr
        self.e_lr=args.e_lr
        self.d_lr=args.d_lr


        self.lr_min=args.lr_min
        self.lr_factor=args.lr_factor
        self.lr_patience=args.lr_patience

        self.device=args.device
        self.checkpoint=args.checkpoint

        self.adv_loss_reg=args.adv
        self.diff_loss_reg=args.orth
        self.s_steps=args.s_step
        self.d_steps=args.d_step

        self.diff=args.diff #????

        self.network=network
        self.inputsize=args.inputsize
        self.taskcla=args.taskcla
        self.num_tasks=args.ntasks

        # Initialize generator and discriminator
        self.model=model
        self.discriminator=self.get_discriminator()
        self.discriminator.get_size()

        self.latent_dim=args.latent_dim

        # self.task_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.task_loss=  JointBceLoss().to(self.device)
        # self.dice_loss = DiceLoss().to(self.device)
        # self.focal_loss = FocalLoss() #
        self.adversarial_loss_d=torch.nn.CrossEntropyLoss().to(self.device)
        self.adversarial_loss_s=torch.nn.CrossEntropyLoss().to(self.device)
        self.diff_loss=JointDiffLoss().to(self.device)

        self.optimizer_S=self.get_S_optimizer()
        self.optimizer_D=self.get_D_optimizer()

        self.create_summary_writer()

        self.task_encoded={}

        self.mu=0.0
        self.sigma=1.0

        self.epoch = 0

    def create_summary_writer(self):
        self.eval_summary_writer = SummaryWriter()

    def get_discriminator(self):
        discriminator=Discriminator(self.args).to(self.args.device)
        # print(discriminator)
        return discriminator

    def get_S_optimizer(self, e_lr=None):
        if e_lr is None: e_lr=self.e_lr
        # optimizer_S=torch.optim.SGD(self.model.parameters(), momentum=self.args.mom,
        #                             weight_decay=self.args.e_wd, lr=e_lr)
        optimizer_S = torch.optim.Adam(self.model.parameters(), lr=e_lr, weight_decay=self.args.e_wd)
        return optimizer_S

    def get_D_optimizer(self, d_lr=None):
        if d_lr is None: d_lr=self.d_lr
        optimizer_D=torch.optim.SGD(self.discriminator.parameters(), weight_decay=self.args.d_wd, lr=d_lr)
        # optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=d_lr, weight_decay=0)
        return optimizer_D

    @staticmethod
    def get_training_data(data_query: DataQuery, debug_mode: bool = False, options=None) \
            -> Tuple[Dataset, Dataset]:
        training_data_volumes = get_data(query=data_query, split=Split.Training,
                                         debug_mode=debug_mode, options=options)
        # print(training_data_volumes)
        # This is just here cuz i was trying to find what is different about the train data when evaluating on them
        if options.per_vol_eval:
            training_data = training_data_volumes
        else:
            # Combine all volume samples for slice-wise training
            training_data = ConcatDataset(training_data_volumes)
        # Combine all volume samples for slice-wise training
        # training_data = ConcatDataset(training_data_volumes)
        # print(training_data)
        return training_data

    def get_validation_data(self, data_query: DataQuery, debug_mode: bool = False, options=None) \
            -> Tuple[Dataset, Dataset]:
        early_stopping_data_volumes = get_data(query=data_query, split=Split.EarlyStopping,
                                               debug_mode=debug_mode, options=options)

        if options.per_vol_eval:
            early_stopping_data = early_stopping_data_volumes
        else:
            # Combine all volume samples for slice-wise training
            early_stopping_data = ConcatDataset(early_stopping_data_volumes)

        return early_stopping_data

    def get_test_data(self, data_query: DataQuery, debug_mode: bool = False, options=None) \
            -> Tuple[Dataset, Dataset]:
        test_data_volumes = get_data(query=data_query, split=Split.FinalEvaluation,
                                     debug_mode=debug_mode, options=options)

        if options.per_vol_eval:
            test_data = test_data_volumes
        else:
            # Combine all volume samples for slice-wise training
            test_data = ConcatDataset(test_data_volumes)

        return test_data

    # @staticmethod
    # def get_test_data(data_query: DataQuery, debug_mode: bool = False, options=None) -> List[Dataset]:
    #     return get_data(query=data_query, split=Split.DevelopmentTest, debug_mode=debug_mode, options=options)

    def organ_label_mapping(self, organ):
        # returns the corresponding label for a certain organ
        return self.ROI_order.index(organ) + 1

    def train(self):

        # Before training a task check if there is a checkpoint for it, and load it
        model_file = os.path.join(self.checkpoint, 'model.pth.tar')
        if os.path.exists(model_file):
            self.model = self.load_checkpoint()

            if not self.args.continue_train:
                return
        #     if not self.organ in  ["oesophagus"]: # "heart", "oesophagus"
        #         return
        #
        # self.discriminator=self.get_discriminator(task_id)

        best_loss=np.inf
        best_val_score = 0
        best_model=utils.get_model(self.model)


        best_loss_d=np.inf
        best_model_d=utils.get_model(self.discriminator)

        dis_lr_update=True  #????
        d_lr=self.d_lr
        patience_d=self.lr_patience
        self.optimizer_D=self.get_D_optimizer(d_lr)

        e_lr=self.e_lr
        patience=self.lr_patience
        self.optimizer_S=self.get_S_optimizer(e_lr)

        task_query = self.tasks[0]  # one query containing 5 organs
        # print(task_query)
        # print(type(task_query[0]))
        # exit()
        # get the respective task data
        # task_query = self.tasks[task_id]
        # self.original_label = self.organ_label_mapping(task_query.tasks[0]) if self.args.dataset == 'AAPM' else 1

        st=time.time()
        training_data = self.get_training_data(data_query=task_query,
                                                debug_mode=self.args.debug_mode,
                                                options=self.args)
        print("len training data", len(training_data))
        validation_data = self.get_validation_data(data_query=task_query,
                                                   debug_mode=self.args.debug_mode,
                                                   options=self.args)
        en = time.time()

        logger.info("Time for fetching the data {} min ".format((en-st)/60))
        # Create dataloaders
        training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=self.batch_size,
                                                           # drop last true will cause a crash in LTRC datasets cuz it is less than batchsize
                                                           # so it is dropped and len dataloader = 0
                                                           shuffle=True, drop_last=True,
                                                           num_workers=self.args.workers)
        print("train data loader len ", len(training_data_loader))
        validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=self.batch_size,
                                                             shuffle=False, drop_last=False,
                                                             num_workers=self.args.workers)


        logger.info("*"*80)
        logger.info("Start Training task {}".format(task_query.tasks))
        logger.info("*" * 80)
        for e in range(self.nepochs):
            self.epoch = e
            # Train
            clock0=time.time()
            # print("ddata loader len before", len(training_data_loader))
            self.train_epoch(training_data_loader)
            # print("ddata loader len after", len(training_data_loader))
            clock1=time.time()
            logger.info("Epoch time {} ".format((clock1-clock0)/60))
            train_res=self.eval_(training_data_loader)
            utils.report_tr(train_res, e, self.batch_size, clock0, clock1, self.checkpoint)


            # Valid
            valid_res=self.eval_(validation_data_loader) #
            logger.info(" \n ")
            # logger.info(" \n ")
            utils.report_val(valid_res, self.checkpoint)
            logger.info(" \n ")
            # logger.info(" \n ")

            # Adapt lr for S and D
            # if valid_res['loss_t'] < best_loss:  # old:  loss_tot  TODO: change this to total loss if needed
            #     best_loss=valid_res['loss_t']    #  old: loss_tot
            #     best_model=utils.get_model(self.model)
            #     patience=self.lr_patience
            #     logger.info(' *')
            if valid_res['dice'].mean() > best_val_score:
                best_val_score = valid_res['dice'].mean()
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                logger.info(' *')
            else:
                patience-=1
                if patience <= 0:
                    e_lr/=self.lr_factor
                    logger.info('=========> Decreasing Shard mod. lr={:.1e}'.format(e_lr))
                    if e_lr < self.lr_min:
                        logger.info(" \n")
                        break
                    patience=self.lr_patience
                    self.optimizer_S=self.get_S_optimizer(e_lr)

            if train_res['loss_a'] < best_loss_d:
                best_loss_d=train_res['loss_a']
                best_model_d=utils.get_model(self.discriminator)
                patience_d=self.lr_patience
            else:
                patience_d-=1
                if patience_d <= 0 and dis_lr_update:
                    d_lr/=self.lr_factor
                    logger.info('====> Decreasing Disc lr={:.1e}'.format(d_lr))
                    if d_lr < self.lr_min:
                        dis_lr_update=False
                        logger.info("========= > Dis lr reached minimum value")
                        logger.info( "\n")
                    patience_d=self.lr_patience
                    self.optimizer_D=self.get_D_optimizer(d_lr)
            logger.info(" \n")

            # Saving the model each epoch just in case
            self.save_all_models()

        # Restore best validation model (early-stopping) after training the task is over
        logger.info("===== > Restoring best validation model")
        self.model.load_state_dict(copy.deepcopy(best_model))
        self.discriminator.load_state_dict(copy.deepcopy(best_model_d))

        self.save_all_models()

    def train_epoch(self, training_data_loader):

        self.model.train()
        self.discriminator.train()

        # For per a set of iterations reporting
        report_iter, r_adv_loss, r_diff_loss, fake_dis, real_dis, acc_in_s, acc_in_d = 0, 0, 0, 0, 0, 0, 0
        correct_s, correct_d_real, correct_d_fake = 0, 0, 0

        for data, target, tt, td in training_data_loader:
            x=data.to(device=self.device)
            y= target #(target == self.original_label).type(torch.FloatTensor).to(device=self.device)
            tt=tt.to(device=self.device)

            # print("min input {} max input {}".format(torch.min(x), torch.max(x)))
            # print("unique target {} , unique y {}".format(torch.unique(target), torch.unique(y)))
            # continue
            # print("unique target", torch.unique(target))
            # exit()
            # print("unique y", torch.unique(y))
            # Detaching samples in the batch which do not belong to the current task before feeding them to P
            # Thats because she uses memory sometimes that contains samples from previous tasks
            # t_current=task_id * torch.ones_like(tt)
            # body_mask=torch.eq(t_current, tt).cpu().numpy()
            # # x_task_module=data.to(device=self.device)
            # x_task_module=data.clone()
            # for index in range(x.size(0)):
            #     if body_mask[index] == 0:
            #         x_task_module[index]=x_task_module[index].detach()
            # x_task_module=x_task_module.to(device=self.device)

            # Discriminator's real and fake task labels
            t_real_D=td.to(self.device)
            t_fake_D=torch.zeros_like(t_real_D).to(self.device)


            # ================================================================== #
            #                        Train Shared Module                          #
            # ================================================================== #
            # training S for s_steps
            # print("Training Shared Module")
            for s_step in range(self.s_steps):
                self.optimizer_S.zero_grad()
                self.model.zero_grad()

                #x_task_module goes through P modules and contains only the samples that belongs to the current task
                output, shared_encoded, task_encoded=self.model(x, x)  # x_task_module
                # print(output.type(), y.type())

                task_loss = self.task_loss(output, y, [0]) #+ 0.0 * self.dice_loss(output, y)
                # uses only the shared_encoded_input, check the implementation of the disc
                dis_out_gen_training=self.discriminator.forward(shared_encoded) #, t_real_D, task_id)
                # print("max in input {}".format(torch.max(x)))
                # print("max in pvt encoded {}".format(torch.max(task_encoded)))
                # print("max in shared encoded {}".format(torch.max(shared_encoded)))
                # print("max in disc out gen training", torch.max(dis_out_gen_training))

                adv_loss=self.adversarial_loss_s(dis_out_gen_training, t_real_D)  #CrossEntropyLoss

                if self.diff == 'yes':
                    diff_loss=self.diff_loss(shared_encoded, task_encoded)  # Diffloss
                else:
                    diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
                    self.diff_loss_reg=0  # regularizer

                total_loss=task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss

                total_loss.backward(retain_graph=True)

                self.optimizer_S.step()  ## optimizes the whole net (shared and private and heads)

                # For reporting, to be removed after debugging
                _, pred_d = dis_out_gen_training.max(1)  # Batch x n_outputs
                correct_s += pred_d.eq(t_real_D.view_as(pred_d)).sum().item()
                r_adv_loss+=adv_loss.item() * self.adv_loss_reg
                r_diff_loss+=diff_loss.item()

            # ================================================================== #
            #                          Train Discriminator                       #
            # ================================================================== #
            # training discriminator for d_steps
            for d_step in range(self.d_steps):
                self.optimizer_D.zero_grad()
                self.discriminator.zero_grad()

                # training discriminator on real data
                output, shared_encoded, task_out=self.model(x, x)  #x_task_module
                # shared_encoded, task_out=self.model.get_encoded_ftrs(x, x, task_id)  #x_task_module
                dis_real_out=self.discriminator.forward(shared_encoded.detach()) #.detach(), t_real_D, task_id)
                dis_real_loss=self.adversarial_loss_d(dis_real_out, t_real_D)

                # TODO: check if you need to multiply by the regularizer
                # if self.args.experiment == 'miniimagenet':
                dis_real_loss*=self.adv_loss_reg
                dis_real_loss.backward(retain_graph=True)

                # training discriminator on fake data
                z_fake=torch.as_tensor(np.random.normal(self.mu, self.sigma, (x.size(0), self.latent_dim)),dtype=torch.float32, device=self.device)
                # uses only the firts input, check the implementation of the disc, no use of t_real_d and taskid
                dis_fake_out=self.discriminator.forward(z_fake) #, t_real_D, task_id)
                dis_fake_loss=self.adversarial_loss_d(dis_fake_out, t_fake_D)
                dis_fake_loss*=self.adv_loss_reg
                dis_fake_loss.backward(retain_graph=True)

                self.optimizer_D.step()

                # For debugging, to be removed later
                _, pred_d = dis_real_out.max(1)  # Batch x n_outputs
                correct_d_real += pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

                _, pred_d = dis_fake_out.max(1)  # Batch x n_outputs
                correct_d_fake += pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

                fake_dis+=dis_fake_loss.item()
                real_dis+=dis_real_loss.item()

            report_iter += 1
            if report_iter % 1 == 0:
                denom = report_iter * self.batch_size
                logger.info("*****s_adv_loss: {} ,"
                          " diff_loss: {:.3f}, fake_loss_dis: {:.3f}, real_loss_dis: {:.3f}, "
                          "acc_disc_inS: {:.3f}% , acc_dis_real {:.3f}%, acc_dis_fake {:.3f}%"
                          " ****** ".format(r_adv_loss/(report_iter*self.s_steps),
                                            r_diff_loss/(report_iter*self.s_steps),
                                            fake_dis/report_iter,  # loss is averaged over batch in loss function
                                            real_dis/report_iter,  # loss is averaged over batch in loss function
                                            100*correct_s/(denom*self.s_steps),
                                            100*correct_d_real/denom,
                                            100*correct_d_fake/denom,
                                            ))
                report_iter, r_adv_loss, r_diff_loss, fake_dis, real_dis, acc_in_s, acc_in_d = 0, 0, 0, 0, 0, 0, 0
                correct_s, correct_d_real, correct_d_fake = 0, 0, 0
            #########

        return


    def eval_(self, data_loader, vis_organ="None"):
        loss_a, loss_t, loss_d, loss_total=0, 0, 0, 0
        correct_d, correct_t = 0, 0
        num=0
        batch=0

        self.model.eval()
        self.discriminator.eval()

        res={}
        dice_scores=[]  # list of scores, one corresponding to each threshold
        iter = 0
        with torch.no_grad():
            for batch, (data, target, tt, td) in enumerate(data_loader):
                x=data.to(device=self.device)
                y= target
                tt=tt.to(device=self.device)
                t_real_D=td.to(self.device)

                # print(torch.unique(y), torch.unique(target))
                # Forward
                output, shared_out, task_out=self.model(x, x)
                # Discriminator's performance:
                # probably don't need to add sigmoid or softmax because we choose the location of the max output
                # and in anycase the node that gives the max output will be the same after a sigmoid or softmax
                output_d = self.discriminator.forward(shared_out)  # , t_real_D, task_id)
                # tensor.max returns the max and max_indices, we need the max_indicies (argmax)
                _, pred_d = output_d.max(1)  # Batch x n_outputs
                correct_d += pred_d.eq(t_real_D.view_as(pred_d)).sum().item()
                # # Loss values
                task_loss = self.task_loss(output, y, [0]) # + 0.0 * self.dice_loss(output, y)
                adv_loss = self.adversarial_loss_d(output_d, t_real_D)

                if self.diff == 'yes':
                    diff_loss = self.diff_loss(shared_out, task_out)
                else:
                    diff_loss = torch.tensor(0).to(device=self.device, dtype=torch.float32)
                    self.diff_loss_reg = 0

                total_loss = task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss
                # total_loss += focal_loss * 0.5

                # total_loss = task_loss
                # compute dice scores and save some samples of the segmentation for sanity check
                output = torch.sigmoid(output)
                for i, t in enumerate([0.5]):
                    final_pred = (output >= t).type(torch.FloatTensor).to(self.device)
                    dice_score = self.compute_dice_score(final_pred, y)
                    # print(dice_score)
                    dice_scores.append(dice_score) # .item()

                if self.args.vis_flag:
                    # TODO: limit the number of samples you save each epoch
                    utils.visualise_models_pred_results(x, y, output, vis_organ, self.checkpoint, iter)
                    iter += 1

                loss_t+=task_loss
                loss_a+=adv_loss
                loss_d+=diff_loss
                loss_total+=total_loss

                # loss_t+=task_loss
                # loss_a+= torch.tensor([0])
                # loss_d+= torch.tensor([0])
                # loss_total+=total_loss

                num+=x.size(0)

        dice_scores_to_report = np.asarray(dice_scores).mean(axis=0)
        # print(dice_scores_to_report.shape)
        # exit()
        res['loss_t'], res['dice']=loss_t.item() / (batch + 1), dice_scores_to_report #np.inf #100 * correct_t / num
        res['loss_a'], res['acc_d']=loss_a.item() / (batch + 1), 100 * correct_d / num
        res['loss_d']=loss_d.item() / (batch + 1)
        res['loss_tot']=loss_total.item() / (batch + 1)
        res['size']=self.loader_size(data_loader)

        return res

    #
    def eval_all(self):
        """
        This function is being called after training is done, to evaluate the current task and all the previous ones
        """
        print("########### EVALUATION#######################")

        model_file = os.path.join(self.checkpoint, 'model.pth.tar')
        print("searching for the model in {}".format(self.checkpoint))
        if not os.path.exists(model_file):
            raise ValueError("No model found for {}".format(self.organ))

        self.model = self.load_checkpoint()
        self.model.eval()

        st=time.time()
        task_query = self.tasks[0]  # one query containing 5 organs
        if self.args.eval_split == "val":
            evaluation_data = self.get_validation_data(data_query=task_query,
                                                       debug_mode=self.args.debug_mode,
                                                       options=self.args)
        else:
            evaluation_data = self.get_test_data(data_query=task_query,
                                                       debug_mode=self.args.debug_mode,
                                                       options=self.args)

                # evaluation_data = self.get_training_data(data_query=task_query,
                #                                      debug_mode=self.args.debug_mode,
                #                                      options=self.args)
        en = time.time()

        logger.info("Time for fetching the data {} min ".format((en-st)/60))
    # Create dataloaders

        eval_data_loader = torch.utils.data.DataLoader(evaluation_data, batch_size=self.batch_size,
                                                             shuffle=False, drop_last=True,
                                                             num_workers=self.args.workers)

        logger.info("Len of validation dataloader {}, batch size {}".format(len(eval_data_loader), self.batch_size))
        logger.info("*"*80)
        valid_res=self.eval_(eval_data_loader)
        utils.report_val(valid_res, self.checkpoint)
        logger.info(" \n")
        # save organ eval to file
        # print(scores)
        # scores_df = pd.DataFrame.from_dict(scores)
        # file_name = os.path.join(os.path.join(self.checkpoint, 'scores'),
        #                          "{}_model_all_organs_avg_{}.csv".format(self.organ, self.args.eval_split))
        # scores_df.T.to_csv(file_name)

    def eval_all_per_vol(self, task_id):
        """
        This function is being called after training is done, to evaluate the current task and all the previous ones
        """
        tasks_to_evaluate = self.tasks[:task_id+1]

        self.organ = self.tasks[task_id].tasks[0]
        model_file = os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(self.organ))
        if not os.path.exists(model_file):
            raise ValueError("No model found for {}".format(self.organ))

        if not self.organ in ['oesophagus']:
            logger.info("skipping evaluation for {}".format(self.organ))
            return
        self.model = self.load_checkpoint(self.organ)
        self.model.eval()
        # NOTE: Task id changes in the loop to cover all trained tasks
        for task_id, task_query in enumerate(tasks_to_evaluate):
            # get the respective task data
            self.original_label = self.organ_label_mapping(task_query.tasks[0]) #if self.args.dataset == 'AAPM' else 1

            logger.info("========> Task ID {} organ {} <========".format(task_id, task_query.tasks[0]))

            st=time.time()
            if self.args.eval_split == "val":
                evaluation_data = self.get_validation_data(data_query=task_query,
                                                           debug_mode=self.args.debug_mode,
                                                           options=self.args)
            else:
                evaluation_data = self.get_test_data(data_query=task_query,
                                                           debug_mode=self.args.debug_mode,
                                                           options=self.args)

                # evaluation_data = self.get_training_data(data_query=task_query,
                #                                      debug_mode=self.args.debug_mode,
                #                                      options=self.args)
            en = time.time()

            logger.info("Time for fetching the data {} min ".format((en-st)/60))
            logger.info("*" * 80)
            logger.info("Start evaluating model {} on task {}".format(self.organ, task_query.tasks))
            logger.info("*" * 80)
            scores = {}
            for eval_data in evaluation_data:
                # if eval_data.patient not in ["LCTSC-Test-S2-103"]:
                #     continue
                logger.info("Evaluating one scan {}".format(eval_data.patient))
                validation_data_loader = torch.utils.data.DataLoader(eval_data, batch_size=self.batch_size,
                                                                     shuffle=False, drop_last=False,
                                                                     num_workers=self.args.workers)

                # logger.info("Len of validation dataloader {}, batch size {}".format(len(validation_data_loader), self.batch_size))
                valid_res=self.eval_(validation_data_loader, task_id, phase="val",
                                     vis_organ=task_query.tasks[0], vis_flag=self.args.vis_flag)
                utils.report_val(valid_res, self.checkpoint)
                logger.info(" \n")

                #save to dict
                scores[eval_data.patient] = valid_res['dice']

            # save organ eval to file
            scores_df = pd.DataFrame.from_dict(scores)
            file_name = os.path.join(os.path.join(self.checkpoint, 'scores'),
                                     "{}_eval_{}.csv".format(task_query.tasks[0], self.args.eval_split))
            scores_df.T.to_csv(file_name)
            # print(scores_df.T)

    def make_one_hot(self, tensor, num_classes=7): # 7 structseg
        bs, _, h, w = tensor.size()
        tensor = tensor.type(torch.LongTensor)
        y_true_one_hot = torch.FloatTensor(bs, num_classes, h, w).zero_()
        y_true_one_hot = y_true_one_hot.scatter_(1, tensor, 1.0)

        return y_true_one_hot

    def compute_dice_score(self, pred, target, smooth=1e-6):
        '''
            pred : Bxn_headsxwxh
            target: Bx1xwxh
        '''
        # convert target to one hot
        target = self.make_one_hot(target)
        target = target[:, 1:, :, :] # to skip the background
        assert pred.size() == target.size()
        output = pred.to(self.device)
        target = target.to(self.device)
        if len(pred.size()) < 4:
            # make 4d tensor of 3d tensor by adding the channel dim
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)

        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        # print(torch.unique(pred, return_counts=True), torch.unique(target, return_counts=True))
        # print(output.sum(dim=(2, 3)) , target.sum(dim=(2, 3)))
        # print(intersection, union)
        dsc = (2. * intersection + smooth) / ( union + smooth)
        dsc = dsc.detach()
        return dsc.mean(dim=0).cpu().numpy()
    # def compute_dice_score(self, prediction, target) -> float:
    #     """
    #     Computes the Dice coefficient.
    #     Returns:
    #         Dice coefficient (0 = no overlap, 1 = perfect overlap)
    #     """
    #     if not isinstance(prediction, np.ndarray):
    #         prediction = prediction.cpu().detach().numpy()
    #         target = target.cpu().detach().numpy()
    #
    #     assert prediction.shape == target.shape
    #
    #     prediction_bool = prediction.astype(np.bool)
    #     target_bool = target.astype(np.bool)
    #
    #     if not np.any(prediction_bool) and not np.any(target_bool):
    #         # Matching empty sets is valid so return 1
    #         return 1.0
    #
    #     intersection = np.logical_and(prediction_bool, target_bool)
    #
    #     if not np.any(intersection):
    #         # Avoid divide by zero
    #         return 0.0
    #
    #     return 2.0 * intersection.sum() / (prediction_bool.sum() + target_bool.sum())

    def save_all_models(self):

        dis=utils.get_model(self.discriminator)
        torch.save({'model_state_dict': dis,
                    }, os.path.join(self.checkpoint, 'discriminator.pth.tar'))

        model=utils.get_model(self.model)
        torch.save({'model_state_dict': model,
                    }, os.path.join(self.checkpoint, 'model.pth.tar'))

        logger.info("Saved all models for task...")


    def load_checkpoint(self):
        logger.info("Loading checkpoint from {}".format(self.checkpoint))

        # Load a previous model
        net=self.network.Net(self.args, tasks=self.tasks)
        checkpoint=torch.load(os.path.join(self.checkpoint, 'model.pth.tar'))
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        net=net.to(self.args.device)
        return net


    def loader_size(self, data_loader):
        return data_loader.dataset.__len__()



    def get_tsne_embeddings(self, task_id):
        from tensorboardX import SummaryWriter
        self.task_id = task_id
        tasks_to_evaluate = self.tasks[:task_id+1]
        self.organ = self.tasks[task_id].tasks[0]
        model_file = os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(self.organ))
        if not os.path.exists(model_file):
            raise ValueError("No model found for {}".format(self.organ))

        if not self.organ in ['oesophagus']: #oesophagus
            logger.info("skipping embeddings from {}".format(self.organ))
            return
        self.model = self.load_checkpoint(self.organ)
        self.model.eval()

        print("*"*80)
        print("Extracting tsne embeddings from model {}".format(self.organ))
        print("*" * 80)

        tag_ = '_diff_{}'.format(self.args.diff)
        all_images, all_targets, all_shared, all_private = [], [], [], []

        # Test final model on first 10 tasks:
        writer = SummaryWriter("runs/{}".format(self.args.name))
        with torch.no_grad():
            for task_id, task_query in enumerate(tasks_to_evaluate):
                # fetch the data for this specific task
                evaluation_data = self.get_test_data(data_query=task_query,
                                                     debug_mode=self.args.debug_mode,
                                                     options=self.args)
                eval_data_loader = torch.utils.data.DataLoader(evaluation_data,
                                                               batch_size=self.batch_size,
                                                               shuffle=True,
                                                               drop_last=True,
                                                               num_workers=self.args.workers)

                for itr, (data, target, tt, td) in enumerate(eval_data_loader):
                    # limiting the number of samples to make it faster 50*12 = 600
                    if itr > 50:
                        break
                    x = data.to(device=self.device)
                    tt = tt.to(device=self.device)
                    target = target.to(self.device)
                    # print("target size", target.size())
                    # output = self.model.forward(x, x, tt, task_id)
                    shared_out, private_out = self.model.get_encoded_ftrs(x, x, task_id)
                    # print("sahred size {} private size {}".format(shared_out.size(), private_out.size()) )
                    all_shared.append(shared_out.cpu().detach())
                    all_private.append(private_out.cpu().detach())
                    # all_images.append(x)
                    all_targets.append(target.cpu().detach())


        shared = torch.stack(all_shared,dim=1).view(-1, 256).data
        private =  torch.stack(all_private,dim=1).view(-1, 256).data
        # label_img = torch.stack(all_images,dim=1).view(-1,1,256,256).data
        metadata = torch.stack(all_targets, dim=1).view(-1,1,256,256).data
        print(shared.size(), private.size(), metadata.size())
        print(torch.unique(metadata))
        tag = ['Shared_{}_{}'.format(tag_,i) for i in range(1, len(tasks_to_evaluate))]
        writer.add_embedding(mat=shared, #label_img=label_img,
                             metadata=metadata, tag="shared")

        tag = ['Private_{}_{}'.format(tag_, i) for i in range(1, len(tasks_to_evaluate))]
        writer.add_embedding(mat=private, #label_img=label_img,
                             metadata=metadata, tag="private")
        writer.close()


    def get_tsne_embeddings_last_three_tasks(self, dataset, model):
        from tensorboardX import SummaryWriter

        # Test final model on last 3 tasks:
        model.eval()
        tag = '_diff_{}'.format(self.args.diff)

        for t in [17,18,19]:
            all_images, all_labels, all_shared, all_private = [], [], [], []
            writer = SummaryWriter()
            for itr, (data, target, tt, td) in enumerate(dataset[t]['tsne']):
                x = data.to(device=self.device)
                y = target.to(device=self.device, dtype=torch.long)
                tt = tt.to(device=self.device)
                output = model.forward(x, x, tt, t)
                shared_out, private_out = model.get_encoded_ftrs(x, x, t)
                # print (shared_out.size())

                all_shared.append(shared_out)
                all_private.append(private_out)
                all_images.append(x)
                all_labels.append(y)

            writer.add_embedding(mat=torch.stack(all_shared,dim=1).data, label_img=torch.stack(all_images,dim=1).data,
                                 metadata=list(range(1,6)), tag='Shared_{}_{}'.format(t, tag))
                                 # ,metadata_header=list(range(1,6)))
            writer.add_embedding(mat=torch.stack(all_private,dim=1).data, label_img=torch.stack(all_images,dim=1).data,
                                 metadata=list(range(1,6)), tag='Private_{}_{}'.format(t, tag))
                                 # ,metadata_header=list(range(1,6)))

        writer.close()

    # def maximize_neuron_output(self):
    #     model_file = os.path.join(self.checkpoint, 'model_{}.pth.tar'.format("oesophagus"))
    #     if not os.path.exists(model_file):
    #         raise ValueError("No model found for {}".format("oesophagus"))
    #
    #     net=self.network.Net(self.args, tasks=self.tasks)
    #     checkpoint=torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format("oesophagus")))
    #     net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    #     net=net.to(self.args.device)
    #
    #     # print(net)
    #     shared = net.shared
    #     # print(shared)
    #     input_img_data = torch.tensor(np.random.randint(0, 255, (1, 1, 256, 256))).to(self.device)
    #
    #
    #     grads = .gradients(loss, model.input)[0]
    #
    #     # normalization trick: we normalize the gradient
    #     grads /= K.std(grads) + 1e-8
    #
    #     # this function returns the loss and grads given the input picture
    #     iterate = K.function([model.input], [loss, grads])

        #
class JointDiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(JointDiffLoss, self).__init__()

    def forward(self, shared_embedding, stacked_pvt_embeddings):
        shared_embedding=shared_embedding.view(shared_embedding.size(0), -1)
        shared_embedding_norm=torch.norm(shared_embedding, p=2, dim=1, keepdim=True).detach()
        shared_embedding_norm=shared_embedding.div(shared_embedding_norm.expand_as(shared_embedding) + 1e-6)

        diff_loss = 0
        for pvt_embedding in stacked_pvt_embeddings:
            pvt_embedding=pvt_embedding.view(pvt_embedding.size(0), -1)
            pvt_embedding_norm=torch.norm(pvt_embedding, p=2, dim=1, keepdim=True).detach()
            pvt_embedding_norm=pvt_embedding.div(pvt_embedding_norm.expand_as(pvt_embedding) + 1e-6)
            diff_loss += torch.mean((shared_embedding_norm.mm(pvt_embedding_norm.t()).pow(2)))
        # return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
        return diff_loss


class JointBceLoss(torch.nn.Module):
    def __init__(self):
        super(JointBceLoss, self).__init__()

        self.device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda')
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true, loss_weights):
        loss = 0
        for i in range(6):
            true_label = y_true == i+1  # no background head or whatsoever

            one_loss = self.criterion(y_pred[:,i,:,:].squeeze(), true_label.type(torch.FloatTensor).squeeze().to(self.device))
            loss += one_loss #* loss_weights[i]
        return loss
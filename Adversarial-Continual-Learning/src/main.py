# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse,time
import numpy as np
from omegaconf import OmegaConf

import torch

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import utils
import logging
from dataloaders.logging import configure_logging

tstart=time.time()

logger = logging.getLogger(__name__)

print("I'm in the main ")
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Arguments
parser = argparse.ArgumentParser(description='Adversarial Continual Learning...')
# Load the config file
parser.add_argument('--config',  type=str, default='./configs/config_mnist5.yml')
flags =  parser.parse_args()
args = OmegaConf.load(flags.config)

print()

from acl import ACL as approach
if args.which_network == 'old':
    from networks import old_Unet_acl as network
elif args.which_network == 'normal':
    from networks import Unet_acl as network
else:
    from networks import Aspp_Unet_acl as network

from dataloaders import AAPM, all_data

########################################################################################################################

def run(args, run_id):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

        # Faster run but not deterministic:
        # torch.backends.cudnn.benchmark = True
        # To get deterministic results that match with paper at cost of lower speed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # if args.joint = true, this will return one query with all tasks included

    if args.dataset == 'AAPM':
        tasks = all_data.aapm_data_queries(options=args)
    elif args.dataset == 'structseg':
        tasks = all_data.structseg_data_queries(options=args)
    else:
        tasks = all_data.ltrc_nlst_data_queries()

    # Model
    # Net contains both shared, private modules and the task specific heads
    net = network.Net(args, tasks=tasks)
    net = net.to(args.device)
    logger.info(net)
    net.print_model_size()


    # Approach

    appr=approach(net,tasks, args,network=network)
        # Evaluate only
    if args.evaluate_only:
        if args.per_vol_eval:
            appr.eval_all_per_vol()
        else:
            appr.eval_all()
            # accumalted_scores.append(model_score)

    else:
        # Train
        appr.train()
    print('-'*250)
    print()

    # compute the Omega scores

    return [], [] #accumalted_scores, omega_scores



#######################################################################################################################
def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    logger.info(message)
    # print(message)

def main(args):

    accum_scores_list, omega_scores_list = list(), list()
    orig_exp_name = args.name
    orig_checkpoint = args.checkpoint
    for n in range(args.num_runs):
        args.checkpoint = orig_checkpoint
        args.name = str(orig_exp_name) + '/run_{}'.format(n)

        ROOT_DIR = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
        if ROOT_DIR == "/code":
            checkpoint_dir_polyaxon = utils.make_dirs_polyaxon()
            args.checkpoint = checkpoint_dir_polyaxon
        else:
            utils.make_directories(args)

        configure_logging(args)
        print_options(args)

        logger.info("Experiement dir {}/{}".format(args.checkpoint, args.name))
        # You can use whatever seed you want!
        args.seed = 123
        logger.info(" >>>> Run # {} ".format(n))
        _, _ = run(args, n)

    print ("All Done! ")
    print('[Elapsed time = {:.1f} min]'.format((time.time()-tstart)/(60)))
    utils.print_time()


#######################################################################################################################

if __name__ == '__main__':
    main(args)

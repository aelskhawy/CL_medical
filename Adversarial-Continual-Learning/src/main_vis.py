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

# Arguments
parser = argparse.ArgumentParser(description='Adversarial Continual Learning...')
# Load the config file
parser.add_argument('--config',  type=str, default='./configs/config_mnist5.yml')
flags =  parser.parse_args()
args = OmegaConf.load(flags.config)

print()

from acl import ACL as approach
from networks import vis_Unet_acl as network
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

    tasks_names = AAPM.all_organs()
    tasks = all_data.aapm_data_queries(options=args)

    # Model
    # Net contains both shared, private modules and the task specific heads
    net = network.Net(args, tasks=tasks)
    net = net.to(args.device)
    net.print_model_size()

    # Approach

    appr=approach(net,tasks, args,network=network)


    # Loop tasks
    acc=np.zeros((len(tasks),len(tasks)),dtype=np.float32)
    lss=np.zeros((len(tasks),len(tasks)),dtype=np.float32)
    # exit()
    # visualise only
    appr.maximize_neuron_output()

    return 0, 0



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
    utils.make_directories(args)
    # utils.some_sanity_checks(args)
    # utils.save_code(args)

    # print('=' * 100)
    # print('Arguments =')
    # for arg in vars(args):
    #     print('\t' + arg + ':', getattr(args, arg))
    # print('=' * 100)
    configure_logging(args)
    print_options(args)

    accuracies, forgetting = [], []
    for n in range(args.num_runs):
        # You can use whatever seed you want!
        args.seed = 123
        args.output = '{}_{}_tasks_seed_{}.txt'.format(args.experiment, args.ntasks, args.seed)
        print ("args.output: ", args.output)

        print (" >>>> Run #", n)
        acc, bwt = run(args, n)
        accuracies.append(acc)
        forgetting.append(bwt)


    print('*' * 100)
    print ("Average over {} runs: ".format(args.num_runs))
    print ('AVG ACC: {:5.4f}% \pm {:5.4f}'.format(np.array(accuracies).mean(), np.array(accuracies).std()))
    print ('AVG BWT: {:5.2f}% \pm {:5.4f}'.format(np.array(forgetting).mean(), np.array(forgetting).std()))


    print ("All Done! ")
    print('[Elapsed time = {:.1f} min]'.format((time.time()-tstart)/(60)))
    utils.print_time()


#######################################################################################################################

if __name__ == '__main__':
    main(args)

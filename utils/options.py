import os
import argparse
import yaml
from utils import paths


class Options:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser = argparse.ArgumentParser(description="Options from Multi organ CL training.")
        parser.add_argument("--name", default="", type=str, help="Name/number of the experiment", )
        parser.add_argument("-m", "--message", default="", type=str, required=False,
                            help="What is different in this trial", )

        parser.add_argument("--yaml_file", dest='yaml_file', type=argparse.FileType(mode='r'),
                            help="yaml file containing the run configs")

        ########## For data Preprocessing ######
        parser.add_argument('--dataset', type=str, default='LTRC_NLST',
                            help='Dataset to work on LTRC_NLTS | AAPM')
        parser.add_argument('--data_root_aapm', type=str,
                            default="./",
                            help='aapm data root')

        ########### FLAGS ########################
        parser.add_argument("--verbose", action='store_true', help="")
        parser.add_argument("--debug_mode", action='store_true', help="")
        parser.add_argument("--run_training", action='store_true', help="")
        parser.add_argument("--run_evaluation", action='store_true', help="")
        parser.add_argument("--output_predictions", action='store_true', help="")
        parser.add_argument("--domain_learning", action='store_true', help="")
        parser.add_argument('--overwrite', action='store_true', help="")
        parser.add_argument('--mask_dataset', action='store_true', help="")
        parser.add_argument('--supress_logging', action='store_true', help="less clutter logging")

        ########### Training options ###############

        parser.add_argument("--num_epochs", type=int, default=100, help="(default: 100)", )
        parser.add_argument("--warmup_epochs", type=int, default=10, help="(default: 10)", )
        parser.add_argument("--batch_size", type=int, default=16, help="input batch size for training", )
        parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate", )
        parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay", )
        parser.add_argument("--num_workers", type=int, default=0, help="", )

        ########### UNET  ################
        parser.add_argument("--nfc", type=int, default=16, help="n conv filter in 1st layer", )

        ####### Augmentation Options ######
        parser.add_argument("--load_size", type=int, default=256, help="target input image size")
        parser.add_argument("--resize", action='store_true', help="If specified, resize the image", )
        parser.add_argument("--gauss_noise", action='store_true',
                            help="If specified, add gauss noise to the image", )
        parser.add_argument("--aug_scale", type=float, default=None,
                            help="scale factor range for augmentation (default: 0.2)", )  # 0.05
        parser.add_argument("--aug_angle", type=int, default=None,
                            help="rotation angle range in degrees for augmentation (default: 15)")  ### 15
        parser.add_argument("--flip_prob", type=float, default=None,
                            help="Flip probability for data augmentation")  # 0.5
        parser.add_argument("--crop", type=float, default=None, help="crop for data augmentation")  # 0.5

        ############# LWF Trainer options ##################
        parser.add_argument("--replay_mode", type=str, default=None, help="LwF| ideal | DGR")
        parser.add_argument("--fine_tune", type=int, default=0, help="0 | 1")

        parser.add_argument("--start_task", type=int, default=None, help="")

        self.initialized = True
        return parser

    def parse_args_from_yaml(self, parser):
        """
        Reads the options from yaml file, if exists, and set the args attributes in the parser
        :param parser: parser object
        :return:
        """
        args = parser.parse_args()
        # print("yaml file", args.yaml_file)
        if args.yaml_file:
            data = yaml.load(args.yaml_file, Loader=yaml.FullLoader)
            # print(data)
            delattr(args, 'yaml_file')
            # if yaml file is empty
            if data is None:
                return args
            for key, value in data.items():
                if isinstance(value, list):
                    for v in value:
                        getattr(args, key, []).extend(v)
                else:
                    setattr(args, key, value)
        return args

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

            opt = self.parse_args_from_yaml(parser)  # parser.parse_args()
            self.parser = parser
            return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):

        opt = self.gather_options()
        # opt.isTrain = self.isTrain   # train or test

        # self.print_options(opt)
        self.opt = opt
        return self.opt

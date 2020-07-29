from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

import logging
logger = logging.getLogger(__name__)
from networks.net_modules import downsample_conv_block, _block, ResBlock

class TaskHead(nn.Module):
    def __init__(self, organ=''):
        super(TaskHead, self).__init__()
        feat = 32
        self.convnorm1 = TaskHead._convnormrelu(in_channels=2, out_channels=feat)  #TaskHead._convnormrelu
        self.convnorm2 = TaskHead._convnormrelu(in_channels=feat, out_channels=feat*2)
        # self.convnorm3 = TaskHead._convnormrelu(in_channels=feat*2, out_channels=feat*2)
        # self.convnorm4 = TaskHead._convnormrelu(in_channels=feat*2, out_channels=feat*2)
        self.lastconvnorm = nn.Conv2d(feat*2,1, kernel_size=3, padding=1, bias=True)

        self.up = nn.Upsample(scale_factor=4) #, mode="bilinear", align_corners=True)  ## TODO: Default nearest, probably use bilinear ???
    def forward(self, f_s_p):
        # f_s_p size = B*2*256
        f_s_p = f_s_p.view(-1, 2, 16, 16)
        f_s_p = self.up(self.convnorm1(f_s_p))
        f_s_p = self.up(self.convnorm2(f_s_p))
        # f_s_p = self.up(self.convnorm3(f_s_p))
        # f_s_p = self.up(self.convnorm4(f_s_p))
        out = self.lastconvnorm(f_s_p)
        # print(out.size())
        return out

    @staticmethod
    def _convnormrelu(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

class PrivateModule(nn.Module):
    def __init__(self, in_channels=1, init_features=64, organ=None, args=None):
        super(PrivateModule, self).__init__()
        self.__name__ = "Private_" + str(organ)
        features = init_features
        self.args = args
        self.latent_dim = args.latent_dim

        self.enc1 = downsample_conv_block(in_channels, features, "Private_enc1", True)  #downsample_conv_block
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.enc2 = downsample_conv_block(features, features * 2, "Private_enc2", True)
        self.lastconv = nn.Conv2d(features * 2, 1, kernel_size=3, padding=1)
        # self.outputlayer = nn.Sequential(
        #     # nn.Linear(16 * 16, self.latent_dim),
        #     # nn.ReLU(),
        #     # nn.Dropout(0.5),
        # )
        # self.dropout1 = nn.Dropout(0.2)

    def forward(self, x):
        pvt_feat = self.pool(self.enc1(x))  # # B*64*64*64
        pvt_feat = self.pool(self.enc2(pvt_feat))  # B*128*16*16
        pvt_feat = self.lastconv(pvt_feat)  # B*1*16*16
        pvt_feat = pvt_feat.view(x.size(0), -1)  # B*256
        # pvt_feat = self.outputlayer(pvt_feat)
        return pvt_feat

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (self.__name__, num_params / 1000000))


class Shared(nn.Module):
    def __init__(self, in_channels=1, init_features=32, args=None):
        super(Shared, self).__init__()
        self.__name__ = "Shared"
        features = init_features
        self.latent_dim = args.latent_dim

        self.encoder1 = _block(in_channels, features, name="Shared_enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder2 = _block(features, features * 2, name="Shared_enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder3 = _block(features * 2, features * 4, name="Shared_enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder4 = _block(features * 4, features * 8, name="Shared_enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 256*16*16

        self.bottleneck = _block(features * 8, features * 16, name="Shared_bottleneck") # 512*16*16
        # in the other scenario we would add ASPP module here
        self.lastconv   = nn.Conv2d(features * 16, 1, kernel_size=3, padding=1)  # bias = False ???

        #
        self.outputlayer  = nn.Sequential(
            nn.Linear(16*16, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.dropout1 = nn.Dropout(0.2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        shared_ftrs = self.lastconv(bottleneck)  # Batch*16*16
        shared_ftrs = shared_ftrs.view(x.size(0), -1) # batch * 256
        # print(shared_ftrs.size())
        shared_ftrs = self.outputlayer(shared_ftrs)

        # print(bottleneck.size(), shared_ftrs.size())
        # shared_ftrs = shared_ftrs.view(x.size(0), -1) # batch * 256
        return shared_ftrs


    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (self.__name__, num_params / 1000000))


class Net(torch.nn.Module):

    def __init__(self, args=None, tasks=None):
        super(Net, self).__init__()
        # tasks is a list of data queries, tasks[0] to access the organ in the query
        self.organs = [task.tasks[0] for task in tasks]
        self.args = args
        # print(self.organs)
        self.num_tasks = len(self.organs)
        self.shared = Shared(args=self.args)

        self.private = torch.nn.ModuleList()
        self.head = torch.nn.ModuleList()

        for organ in self.organs:
            self.private.append(PrivateModule(organ=organ, args=self.args))
            self.head.append(TaskHead(organ=organ))

    def forward(self, x_s, x_p, tt, task_id):

        x_s = x_s.view_as(x_s)
        x_p = x_p.view_as(x_p)

        x_shared = self.shared(x_s)

        # In case of joint training which i'm skipping for now
        # task id will be a list in case of joint training
        # if not isinstance(task_id, list):
        #     task_id = [task_id]
        # prvt_out = []
        # taskhead_out = []
        # for t in task_id:
        #     x_prvt = self.private[t](x_p)
        #     concat_frts = torch.cat([x_prvt, x_shared], dim=1)
        #     x_head = self.head[task_id](concat_frts)
        #
        #     prvt_out.append(x_prvt)
        #     taskhead_out.append(x_head)

        x_prvt = self.private[task_id](x_p)
        # logger.info("both full val")

        # print(x_shared.size(), x_prvt.size())
        concat_frts = torch.cat([x_prvt, x_shared], dim=1)
        # logger.info("shared only full val")
        # concat_frts = torch.cat([x_shared, x_shared], dim=1)  # to test the effect for removing private module
        # logger.info("pvt only full val ")
        # concat_frts = torch.cat([x_prvt, x_prvt], dim=1)
        # print(x_s.size(), x_shared.size(), x_prvt.size(), concat_frts.size())
        x_head = self.head[task_id](concat_frts)

        return x_head


    def get_encoded_ftrs(self, x_s, x_p, task_id):

        return self.shared(x_s), self.private[task_id](x_p)

    def print_model_size(self):
        count_P = sum(p.numel() for p in self.private.parameters() if p.requires_grad)
        count_S = sum(p.numel() for p in self.shared.parameters() if p.requires_grad)
        count_H = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        print('Num parameters in S       = %s ' % (self.pretty_print(count_S)))
        print('Num parameters in P       = %s,  per task = %s ' % (self.pretty_print(count_P),self.pretty_print(count_P/self.num_tasks)))
        print('Num parameters in p       = %s,  per task = %s ' % (self.pretty_print(count_H),self.pretty_print(count_H/self.num_tasks)))
        print('Num parameters in P+p    = %s ' % self.pretty_print(count_P+count_H))
        print('-------------------------->   Architecture size: %s parameters (%sB)' % (self.pretty_print(count_S + count_P + count_H),
                                                                    self.pretty_print(4*(count_S + count_P + count_H))))

        print("------------------------------------------------------------------------------")
        print("                               TOTAL:  %sB" % self.pretty_print(4*(count_S + count_P + count_H)))

    def pretty_print(self, num):
        magnitude=0
        while abs(num) >= 1000:
            magnitude+=1
            num/=1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])






if __name__ == "__main__":
    shared = Shared(in_channels=1, init_features=32)
    pvt = PrivateModule(in_channels=1, init_features=64)
    shared.print_network()
    pvt.print_network()

    net = Net()
    net.print_model_size()
    # shared.to('cuda')
    # # print(unet)
    # from torchsummary import summary
    # #
    # summary(shared, batch_size= 2,  input_size=(1, 256, 256))

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

import logging
logger = logging.getLogger(__name__)
from networks.net_modules import downsample_conv_block, _block, ResBlock, ASPP, ASPPConv, ASPPPooling

class TaskHead(nn.Module):
    def __init__(self, organ=''):
        super(TaskHead, self).__init__()
        feat = 32
        self.convnorm1 = TaskHead._convnormrelu(in_channels=4, out_channels=feat)  # TODO: changed from 4 to 2 for LTRC
        self.convnorm2 = TaskHead._convnormrelu(in_channels=1, out_channels=feat)
        self.before_ps1 = TaskHead._convnormrelu(in_channels=feat, out_channels=feat//2)
        self.before_ps2 = TaskHead._convnormrelu(in_channels=feat, out_channels=feat//2)
        self.lastconvnorm = nn.Conv2d(1,1, kernel_size=3, padding=1, bias=True)

        # self.up = nn.Upsample(scale_factor=4) #, mode="bilinear", align_corners=True)  ## TODO: Default nearest, probably use bilinear ???
        self.pixelshuffle_1 = nn.PixelShuffle(4)
        self.pixelshuffle_2 = nn.PixelShuffle(4)
    def forward(self, f_s_p):

        f_s_p = f_s_p.view(-1, 2, 16, 16)
        mult = (f_s_p[:, 0, :, :] * f_s_p[:, 1, :, :]).unsqueeze(1) # TODO: commented for LTRC
        add = (f_s_p[:, 0, :, :] + f_s_p[:, 1, :, :]).unsqueeze(1) # TODO: commented for LTRC
        # print(f_s_p.size(), mult.size(), add.size())
        f_s_p = torch.cat([f_s_p, mult, add], dim=1)
        f_s_p = self.pixelshuffle_1(self.before_ps1(self.convnorm1(f_s_p)))

        f_s_p = self.pixelshuffle_2(self.before_ps2(self.convnorm2(f_s_p)))

        out = self.lastconvnorm(f_s_p)
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
    def __init__(self, in_channels=1, init_features=32, organ=None, args=None):
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

        self.encoder4 = _block(features * 4, features * 4, name="Shared_enc4") # 256*32*32
        # self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 256*16*16
        self.aspp = ASPP(features * 4, out_channels= features*4)
        # self.bottleneck = _block(features * 8, features * 8, name="Shared_bottleneck") # 256*16*16
        self.lastconv   = nn.Conv2d(32+128, 1, kernel_size=3, padding=1)  # bias = False ???
        #
        self.outputlayer  = nn.Sequential(
            nn.Linear(16*16, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # to down sample the final output to 16*16
        self.last_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._init_weight()

    def forward(self, x):
        enc1 = self.pool1(self.encoder1(x))  # 32*128*128
        enc2 = self.pool2(self.encoder2((enc1)))
        enc3 = self.pool3(self.encoder3(enc2))  # B, 128, 32, 32
        enc4 = self.encoder4(enc3) # B*256*32*32
        aspp_out = self.aspp(enc4)
        low_level_features = enc1
        # downsampling the low level features to concat with aspp_out
        low_level_features =  F.interpolate(low_level_features, size=aspp_out.shape[2:], mode='bilinear',
                                       align_corners=False)

        combined_features = torch.cat([low_level_features, aspp_out], dim=1)
        combined_features = self.last_pool(self.lastconv(combined_features))
        # To pass to linear layer
        shared_ftrs = combined_features.view(x.size(0), -1) # batch * 256
        shared_ftrs = self.outputlayer(shared_ftrs)

        # print(bottleneck.size(), shared_ftrs.size())
        # shared_ftrs = shared_ftrs.view(x.size(0), -1) # batch * 256
        return shared_ftrs

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def print_network(self):
    #     if isinstance(self, list):
    #         self = self[0]
    #     num_params = 0
    #     for param in self.parameters():
    #         num_params += param.numel()
    #     print('Network [%s] was created. Total number of parameters: %.1f million. '
    #           'To see the architecture, do print(network).'
    #           % (self.__name__, num_params / 1000000))


class Net(torch.nn.Module):

    def __init__(self, args=None, tasks=None):
        super(Net, self).__init__()
        # tasks is a list of data queries, tasks[0] to access the organ in the query
        self.organs = tasks[0].tasks  #[task.tasks[0] for task in tasks] if not args.joint else tasks[0].tasks  # else for joint training
        # print("organs in net", self.organs)
        self.args = args
        # print(self.organs)
        self.num_tasks = len(self.organs)
        self.shared = Shared(args=self.args)

        self.private = torch.nn.ModuleList()
        self.head = torch.nn.ModuleList()

        for organ in self.organs:
            # by experiment, heart and oesophagus didn't go well with just 32 filter
            nf = 64 # if organ in ['heart', 'oesophagus'] else 32
            self.private.append(PrivateModule(organ=organ, init_features=nf, args=self.args))
            self.head.append(TaskHead(organ=organ))

    def forward(self, x_s, x_p):

        x_s = x_s.view_as(x_s)
        x_p = x_p.view_as(x_p)

        x_shared = self.shared(x_s)

        pvt_mods_out = []
        task_heads_out = []

        for pvt_mod, task_head in zip(self.private, self.head):
            x_prvt = pvt_mod(x_p)
            concat_frts = torch.cat([x_prvt, x_shared], dim=1)
            # concat_frts = torch.cat([x_shared, x_shared], dim=1)
            # print("shared only")
            # concat_frts = torch.cat([x_prvt, x_prvt], dim=1)
            # print("private only")
            x_head = task_head(concat_frts)
            pvt_mods_out.append(x_prvt)
            task_heads_out.append(x_head)

        # # This should contain channels = number of heads
        # pvt_stacked_out = torch.stack(pvt_mods_out)
        # pvt_stacked_out = pvt_stacked_out.squeeze() # n_heads x bs x latent_dim

        final_stacked_out = torch.stack(task_heads_out)  # [n_heads, bs, 1, h,w]
        final_stacked_out = final_stacked_out.squeeze(2).permute(1, 0, 2, 3)  # [bs, n_heads, h, w]

        return final_stacked_out, x_shared, pvt_mods_out


    # def get_encoded_ftrs(self, x_s, x_p, task_id):
    #
    #     return self.shared(x_s), self.private[task_id](x_p)

    def print_model_size(self):
        count_P = sum(p.numel() for p in self.private.parameters() if p.requires_grad)
        count_S = sum(p.numel() for p in self.shared.parameters() if p.requires_grad)
        count_H = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        logger.info('Num parameters in S       = %s ' % (self.pretty_print(count_S)))
        logger.info('Num parameters in P       = %s,  per task = %s ' % (self.pretty_print(count_P),self.pretty_print(count_P/self.num_tasks)))
        logger.info('Num parameters in p       = %s,  per task = %s ' % (self.pretty_print(count_H),self.pretty_print(count_H/self.num_tasks)))
        logger.info('Num parameters in P+p    = %s ' % self.pretty_print(count_P+count_H))
        logger.info('-------------------------->   Architecture size: %s parameters (%sB)' % (self.pretty_print(count_S + count_P + count_H),
                                                                    self.pretty_print(4*(count_S + count_P + count_H))))

        logger.info("------------------------------------------------------------------------------")
        logger.info("                               TOTAL:  %sB" % self.pretty_print(4*(count_S + count_P + count_H)))

    def pretty_print(self, num):
        magnitude=0
        while abs(num) >= 1000:
            magnitude+=1
            num/=1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])






# if __name__ == "__main__":
    # shared = Shared(in_channels=1, init_features=32)
    # # pvt = PrivateModule(in_channels=1, init_features=64)
    # shared.print_network()
    # pvt.print_network()
    #
    # net = Net()
    # net.print_model_size()
    # shared.to('cuda')
    # # print(unet)
    # from torchsummary import summary
    # #
    # summary(shared, batch_size= 2,  input_size=(1, 256, 256))
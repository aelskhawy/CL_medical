from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

import logging
logger = logging.getLogger(__name__)
from networks.net_modules import downsample_conv_block, _block, ResBlock, ASPP, ASPPConv, ASPPPooling

# Different processing for private and shared
# class TaskHead(nn.Module):
#     def __init__(self, organ=''):
#         super(TaskHead, self).__init__()
#         feat = 32
#         self.pvt_feat_conv = TaskHead._convnormrelu(in_channels=1, out_channels=64)
#         self.shared_feat_conv = TaskHead._convnormrelu(in_channels=1, out_channels=64)
#         self.both_feat_conv = TaskHead._convnormrelu(in_channels=128, out_channels=256)
#         # self.before_ps1 = TaskHead._convnormrelu(in_channels=128, out_channels=64)
#         self.before_ps2 = TaskHead._convnormrelu(in_channels=16, out_channels=128)
#         self.lastconvnorm = nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=True)
#
#         self.pixelshuffle_1 = nn.PixelShuffle(4)
#         self.pixelshuffle_2 = nn.PixelShuffle(4)
#     def forward(self, f_s_p):
#         # f_s_p size = B*2*256
#         # print(f_s_p.size())
#         pvt_feat = f_s_p[:, 0, :].view(-1, 1, 16,16)
#         shared_feat = f_s_p[:, 1, :].view(-1, 1, 16,16)
#
#         pvt_feat = self.pvt_feat_conv(pvt_feat)  # B*64*16*16
#         shared_feat = self.shared_feat_conv(shared_feat)  ## B*64*16*16
#
#         both_feat = torch.cat([pvt_feat, shared_feat], dim=1)  # B*128*16*16
#         both_feat = self. both_feat_conv(both_feat) # B*256*16*16
#         both_feat = self.pixelshuffle_1(both_feat)  # B*16*64*64
#
#         both_feat = self.before_ps2(both_feat) # B*128*64*64
#         both_feat = self.pixelshuffle_2(both_feat)  # B*8*256*256
#
#         out = self.lastconvnorm(both_feat)  # B*1*256*256
#         return out

# old task head
# class TaskHead(nn.Module):
#     def __init__(self, organ=''):
#         super(TaskHead, self).__init__()
#         feat = 32
#         self.convnorm1 = TaskHead._convnormrelu(in_channels=2, out_channels=feat)  # TaskHead._convnormrelu
#         self.convnorm2 = TaskHead._convnormrelu(in_channels=1, out_channels=feat)
#         self.before_ps1 = TaskHead._convnormrelu(in_channels=feat, out_channels=feat // 2)
#         self.before_ps2 = TaskHead._convnormrelu(in_channels=feat, out_channels=feat // 2)
#         self.lastconvnorm = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
#
#         self.pixelshuffle_1 = nn.PixelShuffle(4)
#         self.pixelshuffle_2 = nn.PixelShuffle(4)
#
#     def forward(self, f_s_p):
#         # f_s_p size = B*2*256
#         f_s_p = f_s_p.view(-1, 2, 16, 16)
#         f_s_p = self.pixelshuffle_1(self.before_ps1(self.convnorm1(f_s_p)))
#         # print(f_s_p.size())
#         f_s_p = self.pixelshuffle_2(self.before_ps2(self.convnorm2(f_s_p)))
#         # print(f_s_p.size())
#         # f_s_p = self.up(self.convnorm3(f_s_p))
#         # f_s_p = self.up(self.convnorm4(f_s_p))
#         out = self.lastconvnorm(f_s_p)
#         # print(out.size())
#         return out
#
#     @staticmethod
#     def _convnormrelu(in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=out_channels),
#             nn.ReLU(inplace=True),
#         )

class TaskHead(nn.Module):
    ## SPADE IDEA
    def __init__(self, organ=''):
        super(TaskHead, self).__init__()
        feat = 32
        # self.convnorm1 = TaskHead._convnormrelu(in_channels=2, out_channels=feat)  # TaskHead._convnormrelu
        # self.convnorm2 = TaskHead._convnormrelu(in_channels=1, out_channels=feat)
        # self.before_ps1 = TaskHead._convnormrelu(in_channels=feat, out_channels=feat // 2)
        # self.before_ps2 = TaskHead._convnormrelu(in_channels=feat, out_channels=feat // 2)


        self.conv1 = nn.Conv2d(1,feat, kernel_size=3, padding=1, bias=False)
        self.spade1 = SPADE(feat)
        self.conv2 = nn.Conv2d(feat,feat*2, kernel_size=3, padding=1, bias=False)
        self.spade2 = SPADE(feat*2)

        self.process_up1 = nn.Conv2d(4,feat*2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feat*2)
        self.conv3 = nn.Conv2d(feat*2, feat*2, kernel_size=3, padding=1, bias=False)
        self.spade3 = SPADE(feat*2)
        self.conv4 = nn.Conv2d(feat*2, feat * 2, kernel_size=3, padding=1, bias=False)
        self.spade4 = SPADE(feat * 2)
        self.process_up2 = nn.Conv2d(4, feat * 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(feat * 2)

        self.pixelshuffle_1 = nn.PixelShuffle(4)
        self.pixelshuffle_2 = nn.PixelShuffle(4)

        self.lastconvnorm = nn.Conv2d(feat*2, 1, kernel_size=3, padding=1, bias=True)
    def forward(self, f_s_p):
        # f_s_p size = B*2*256
        pvt_feat = f_s_p[:, 0, :].view(-1, 1, 16,16)
        shared_feat = f_s_p[:, 1, :].view(-1, 1, 16,16)

        pvt_out = self.actvn(self.spade1(self.conv1(shared_feat), pvt_feat))
        pvt_out = self.actvn(self.spade2(self.conv2(pvt_out), pvt_feat))  # b*64*16*16
        first_up = self.actvn(self.bn1(self.process_up1(self.pixelshuffle_1(pvt_out))))  # b*4*64*64

        pvt_out2 = self.actvn(self.spade3(self.conv3(first_up), pvt_feat))  # b*64*64*64
        pvt_out2 = self.actvn(self.spade4(self.conv4(pvt_out2), pvt_feat))  # b*64*64*64

        scnd_up =self.actvn(self.bn2(self.process_up2(self.pixelshuffle_2(pvt_out2)))) # b*4*256*256
        out = self.lastconvnorm(scnd_up)
        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

## SPADE block receives output of a conv layer and does the conditioned normalization then after it you have to pass it
# by an activation
class SPADE(nn.Module):
    def __init__(self, norm_nc):
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        nhidden = 32

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, cond):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on some condition
        cond = F.interpolate(cond, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(cond)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class PrivateModule(nn.Module):
    def __init__(self, in_channels=1, init_features=32, organ=None, args=None):
        super(PrivateModule, self).__init__()
        self.__name__ = "Private_" + str(organ)
        features = init_features
        self.args = args
        self.latent_dim = args.latent_dim

        self.enc1 = downsample_conv_block(in_channels, features, "Private_enc1", True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.enc2 = downsample_conv_block(features, features * 2, "Private_enc2", True)
        self.lastconv = nn.Conv2d(features * 2, 1, kernel_size=3, padding=1, bias=False)
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

        self.bottleneck = _block(features * 8, features * 8, name="Shared_bottleneck") # 256*16*16
        # in the other scenario we would add ASPP module here
        self.lastconv   = nn.Conv2d(features * 8, 1, kernel_size=3, padding=1, bias=False)  # bias = False ???

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

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
            # by experiment, heart and oesophagus didn't go well with just 32 filter
            nf = 64 # if organ in ['heart', 'oesophagus'] else 32
            self.private.append(PrivateModule(organ=organ, init_features=nf, args=self.args))
            self.head.append(TaskHead(organ=organ))

    def forward(self, x_s, x_p, tt, task_id):

        x_s = x_s.view_as(x_s)
        x_p = x_p.view_as(x_p)

        x_shared = self.shared(x_s).unsqueeze(1)

        x_prvt = self.private[task_id](x_p).unsqueeze(1)
        # logger.info("both ")
        concat_frts = torch.cat([x_prvt, x_shared], dim=1)
        # logger.info("shared only")
        # concat_frts = torch.cat([x_shared, x_shared], dim=1)  # to test the effect for removing private module
        # logger.info("pvt only ")
        # concat_frts = torch.cat([x_prvt, x_prvt], dim=1)
        x_head = self.head[task_id](concat_frts)

        return x_head


    def get_encoded_ftrs(self, x_s, x_p, task_id):

        return self.shared(x_s), self.private[task_id](x_p)

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

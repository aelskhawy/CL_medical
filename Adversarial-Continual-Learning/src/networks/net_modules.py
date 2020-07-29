
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

import logging
logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, name="", downsample=False):
        super().__init__()
        # Attributes
        self.learned_shortcut = (in_channels != out_channels)
        fmiddle = min(in_channels, out_channels)

        # create conv layers
        self.convblock = _block(in_channels, out_channels, name) if downsample == False else \
            downsample_conv_block(in_channels, out_channels, name)
        if self.learned_shortcut and downsample == False:
            self.learn_shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,bias=False), # might wanna choose ks =1 ),
                nn.BatchNorm2d(num_features=out_channels),  # InstanceNorm2d
                nn.ReLU(inplace=True),
            )
        else:
            self.learn_shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False),
                # might wanna choose ks =1 ),
                nn.BatchNorm2d(num_features=out_channels),  # InstanceNorm2d
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.convblock(x)
        # print(x_s.size(), dx.size())
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.learn_shortcut(x)
        else:
            x_s = x
        return x_s


def downsample_conv_block(in_channels, features, name="", useless_flag=False):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=5,
                        padding=2,
                        stride=2,
                        bias=False,  ### change that???
                    ),
                ),
                (name + "norm1", nn.BatchNorm2d(num_features=features)),  # InstanceNorm2d
                (name + "relu1", nn.ReLU(inplace=True)),  # if not self.use_lrelu else
                # (name + "LRelu1", nn.LeakyReLU(0.2, inplace=True)),
                (
                    name + "conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,  ### change that???
                    ),
                ),
                (name + "norm2", nn.BatchNorm2d(num_features=features)),
                (name + "relu2", nn.ReLU(inplace=True)),  # if not self.use_lrelu else
                # (name + "LRelu1", nn.LeakyReLU(0.2, inplace=True)),
            ]
        )
    )


def _block(in_channels, features, name=""):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,  ### change that???
                    ),
                ),
                (name + "norm1", nn.BatchNorm2d(num_features=features)), # InstanceNorm2d
                (name + "relu1", nn.ReLU(inplace=True)), #if not self.use_lrelu else
                # (name + "LRelu1", nn.LeakyReLU(0.2, inplace=True)),
                (
                    name + "conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False, ### change that???
                    ),
                ),
                (name + "norm2", nn.BatchNorm2d(num_features=features)),
                (name + "relu2", nn.ReLU(inplace=True)), #if not self.use_lrelu else
                # (name + "LRelu1", nn.LeakyReLU(0.2, inplace=True)),
            ]
        )
    )

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels= 256, atrous_rates = [6, 12, 18]):
        super(ASPP, self).__init__()
        out_channels =  out_channels #256
        modules = []

        # this is skipped cuz i'm inputting 128*32*32
        # modules.append(nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)))

        # rate1, rate2, rate3 = tuple(atrous_rates)
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # modules.append(ASPPConv(in_channels, out_channels, rate2))
        # modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        nf_aspp_out = (len(atrous_rates)+2) * out_channels
        self.project = nn.Sequential(
            nn.Conv2d(nf_aspp_out, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        res.append(x)
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# class DeepLabHeadV3Plus(nn.Module):
#     def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[6, 12, 18]):
#         super(DeepLabHeadV3Plus, self).__init__()
#         self.project = nn.Sequential(
#             nn.Conv2d(low_level_channels, 48, 1, bias=False),
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True),
#         )
#
#         self.aspp = ASPP(in_channels, aspp_dilate)
#
#         self.classifier = nn.Sequential(
#             nn.Conv2d(304, 256, 3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, num_classes, 1)
#         )
#         self._init_weight()
#
#     def forward(self, feature):
#         low_level_feature = self.project(feature['low_level'])
#         output_feature = self.aspp(feature['out'])
#         output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
#                                        align_corners=False)
#         return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

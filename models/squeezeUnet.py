#  Copyright (C) 2019 Canon Medical Systems Corporation. All rights reserved.

import torch

from collections import OrderedDict


class FireModule2D(torch.nn.Module):
    """
    Single fire module of a squeeze net
    :param x: input
    :param squeeze: number of kernels in squeeze block (normally expand/4)
    :param expand: number of kernels in EACH of the pathways (normally 4*squeeze)
    :return: concatenated output from 1x1x1 and 3x3x3 pathways
    """

    def __init__(self, channels_in: int, squeeze: int = 16, expand: int = 64):
        super(FireModule2D, self).__init__()

        self.trunk = torch.nn.Sequential(OrderedDict([
            ("squeeze1x1", torch.nn.Conv2d(channels_in, squeeze, kernel_size=1)),
            ("squeeze1x1 relu", torch.nn.ReLU()),
            ("squeeze1x1 batchnorm", torch.nn.BatchNorm2d(squeeze))]))

        self.branch_left = torch.nn.Sequential(OrderedDict([
            ("expand1x1", torch.nn.Conv2d(squeeze, expand, kernel_size=1)),
            ("expand1x1 relu", torch.nn.ReLU())]))

        self.branch_right = torch.nn.Sequential(OrderedDict([
            ("expand3x3", torch.nn.Conv2d(squeeze, expand, kernel_size=3, padding=1)),
            ("expand3x3 relu", torch.nn.ReLU())]))

    def forward(self, x):
        x = self.trunk(x)
        l = self.branch_left(x)
        r = self.branch_right(x)
        return torch.cat([l, r], dim=1)


class SqueezeUNet2DSmall(torch.nn.Module):

    @staticmethod
    def _check_dims(x):
        assert len(x.shape) == 4, "number of dims != 4"
        assert x.shape[2] % 32 == 0, 'height not mod 32'
        assert x.shape[3] % 32 == 0, 'width not mod 32'

    def __init__(self, nclass: int, input_height: int = 320, input_width: int = 448, channels: int = 1,
                 activation=None):
        super(SqueezeUNet2DSmall, self).__init__()

        self.nclass = nclass
        self.input_height = input_height
        self.input_width = input_width
        self.channels = channels

        self.stage1 = torch.nn.Sequential(OrderedDict([
            ("conv1", torch.nn.Conv2d(channels, 24, kernel_size=3, stride=2, padding=1)),
            ("conv1 relu", torch.nn.ReLU())]))

        self.stage2 = torch.nn.MaxPool2d(3, stride=2, padding=1)

        self.stage3 = torch.nn.Sequential(OrderedDict([
            ("fire2", FireModule2D(24, squeeze=16, expand=64)),
            ("fire21", FireModule2D(128, squeeze=16, expand=64)),
            ("fire3", FireModule2D(128, squeeze=16, expand=64)),
            ("pool3", torch.nn.MaxPool2d(3, stride=2, padding=1)),
        ]))

        self.stage4 = torch.nn.Sequential(OrderedDict([
            ("fire4", FireModule2D(128, squeeze=32, expand=128)),
            ("fire41", FireModule2D(256, squeeze=32, expand=128)),
            ("fire5", FireModule2D(256, squeeze=32, expand=128)),
            ("pool5", torch.nn.MaxPool2d(3, stride=2, padding=1)),
        ]))

        self.stage5 = torch.nn.Sequential(OrderedDict([
            ("fire6", FireModule2D(256, squeeze=48, expand=192)),
            ("fire61", FireModule2D(384, squeeze=48, expand=192)),
            ("fire7", FireModule2D(384, squeeze=48, expand=192)),
            ("pool7", torch.nn.MaxPool2d(3, stride=2, padding=1)),
        ]))

        self.stage6 = torch.nn.Sequential(OrderedDict([
            ("fire8", FireModule2D(384, squeeze=64, expand=256)),
            ("fire81", FireModule2D(512, squeeze=64, expand=256)),
            ("fire9", FireModule2D(512, squeeze=64, expand=256))
        ]))

        self.convtrans1A = torch.nn.ConvTranspose2d(512, 192, kernel_size=3, padding=1)
        self.up1 = FireModule2D(384 + 192, squeeze=48, expand=192)

        self.convtrans1B = torch.nn.ConvTranspose2d(384, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = FireModule2D(256 + 128, squeeze=32, expand=128)

        self.convtrans2 = torch.nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = FireModule2D(128 + 64, squeeze=16, expand=64)

        self.convtrans3 = torch.nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up4 = FireModule2D(24 + 32, squeeze=16, expand=32)

        self.convtrans4a = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsampling1 = torch.nn.Upsample(scale_factor=2)

        self.conv_decoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(32 + 64 + 24, 24, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

        self.convtrans4 = torch.nn.ConvTranspose2d(24, 12, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsampling2 = torch.nn.Upsample(scale_factor=2)

        self.conv_decoder2 = torch.nn.Conv2d(12 + 24, nclass, kernel_size=1)

        self.activation = activation if activation is not None else (
            torch.nn.Sigmoid() if nclass == 1 else torch.nn.Softmax2d())

    def forward(self, x, activate=True, check_dims=False):

        if check_dims:
            self._check_dims(x)

        x01 = self.stage1(x)
        x02 = self.stage2(x01)
        x05 = self.stage3(x02)
        x08 = self.stage4(x05)
        x11 = self.stage5(x08)
        x13 = self.stage6(x11)

        up1 = self.up1(torch.cat([self.convtrans1A(x13), x11], dim=1))
        up2 = self.up2(torch.cat([self.convtrans1B(up1), x08], dim=1))
        up3 = self.up3(torch.cat([self.convtrans2(up2), x05], dim=1))
        up4_ = self.up4(torch.cat([self.convtrans3(up3), x02], dim=1))
        up4 = torch.cat([self.convtrans4a(up4_), self.upsampling1(up4_)], dim=1)

        x = self.conv_decoder1(torch.cat([up4, x01], dim=1))
        x = self.conv_decoder2(torch.cat([self.convtrans4(x), self.upsampling2(x)], dim=1))

        if activate:
            x = self.activation(x)

        return x


class SegHead(torch.nn.Module):
    def __init__(self, head_task='', out=1, activation=None):
        super(SegHead, self).__init__()
        feat = 16 + 32
        self.layers = torch.nn.Sequential(OrderedDict([
            ("head_" + str(head_task) + "_conv1",
             torch.nn.Conv2d(in_channels=feat, out_channels=feat, kernel_size=3, padding=1, bias=False)),
            ("relu1", torch.nn.ReLU(inplace=True)),
            ("head_" + str(head_task) + "_conv2",
             torch.nn.Conv2d(in_channels=feat, out_channels=out, kernel_size=1, bias=True)),
        ]))
        self.output = activation if activation else torch.nn.Sigmoid()


    def forward(self, x):
        logits = self.layers(x)
        output = self.output(logits)
        return output


class SqueezeUnet(torch.nn.Module):
    def __init__(self, nclass: int, channels: int = 1, task: str = '_', base_model: torch.nn.Module = None,
                 activation=None, tune=False):
        super(SqueezeUnet, self).__init__()
        self.nclass = nclass
        self.channels = channels
        self.task = task
        self.activation = activation
        self.base_model = base_model if base_model else SqueezeUNet2DLargeBase(channels=self.channels)
        self.seg_head = SegHead(head_task=self.task, out=self.nclass, activation=self.activation)
        self.tune = tune
        if self.tune:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward_without_head(self, x):
        x = self.base_model(x)
        return x

    def forward(self, x):
        x = self.forward_without_head(x)
        return [self.seg_head(x)]


class SqueezeUnetMultiTask(torch.nn.Module):
    def __init__(self, nclass: int=1, channels: int = 1, task_list: list = None,
                 base_model: torch.nn.Module = None, activation=None):
        super(SqueezeUnetMultiTask, self).__init__()
        self.nclass = nclass
        self.channels = channels
        self.task_list = task_list
        self.activation = activation
        self.base_model = base_model if base_model else SqueezeUNet2DLargeBase(channels=self.channels)
        self.seg_heads =\
            torch.nn.ModuleList([SegHead(head_task=task, out=self.nclass, activation=self.activation)
                                 for task in self.task_list])
        self.__name__ = 'SqueezeUnetMultiTask'

    def forward_without_head(self, x):
        x = self.base_model(x)
        return x

    def forward(self, x):
        x = self.forward_without_head(x)
        seg_heads_out = []
        for head in self.seg_heads:
            seg_out = head(x)
            seg_heads_out.append(seg_out)
        return seg_heads_out



class SqueezeUNet2DLarge(torch.nn.Module):

    @staticmethod
    def _check_dims(x):
        assert len(x.shape) == 4, "number of dims != 4"
        assert x.shape[2] % 32 == 0, 'height not mod 32'
        assert x.shape[3] % 32 == 0, 'width not mod 32'

    def __init__(self, nclass: int, input_height: int = 320, input_width: int = 448, channels: int = 1,
                 dropout: float = 0.5, activation=None):
        super(SqueezeUNet2DLarge, self).__init__()

        self.dropout = dropout > 0

        # VGG style stage1 and stage2
        self.stage1 = torch.nn.Sequential(OrderedDict([
            ("conv1a", torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)),
            ("conv1a relu", torch.nn.ReLU()),
            ("conv1", torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            ("conv1 relu", torch.nn.ReLU())
        ]))

        self.stage2 = torch.nn.Sequential(OrderedDict([
            ("fire101", FireModule2D(64, squeeze=32, expand=128)),
            ("fire102", FireModule2D(256, squeeze=32, expand=128)),
            ("pool1", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        self.stage3 = torch.nn.Sequential(OrderedDict([
            ("fire2", FireModule2D(256, squeeze=64, expand=256)),
            ("fire21", FireModule2D(512, squeeze=64, expand=256)),
            ("fire3", FireModule2D(512, squeeze=64, expand=256)),
            ("pool3", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        self.stage4 = torch.nn.Sequential(OrderedDict([
            ("fire4", FireModule2D(512, squeeze=128, expand=512)),
            ("fire41", FireModule2D(1024, squeeze=128, expand=512)),
            ("fire5", FireModule2D(1024, squeeze=128, expand=512)),
            ("pool5", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        self.dropout = torch.nn.Dropout(p=dropout)

        self.up2 = FireModule2D(1024, squeeze=64, expand=256)

        self.up3 = FireModule2D(512 + 64, squeeze=32, expand=128)
        self.convtrans2 = torch.nn.ConvTranspose2d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.up4 = FireModule2D(256 + 32, squeeze=16, expand=64)
        self.convtrans3 = torch.nn.ConvTranspose2d(256, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsampling1 = torch.nn.Upsample(scale_factor=2)
        self.convtrans4a = torch.nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_decoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(32 + 128 + 64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.convtrans4 = torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsampling2 = torch.nn.Upsample(scale_factor=2)

        self.conv_decoder2 = torch.nn.Conv2d(16 + 32, nclass, kernel_size=1)

        self.activation = activation if activation is not None else (
            torch.nn.Sigmoid() if nclass == 1 else torch.nn.Softmax2d())

    def forward(self, x, activate=True, check_dims=False):

        if check_dims:
            self._check_dims(x)

        x01 = self.stage1(x)
        x02 = self.stage2(x01)
        x05 = self.stage3(x02)
        x08 = self.stage4(x05)

        if self.dropout:
            x08 = self.dropout(x08)

        up2 = self.up2(x08)
        up3 = self.up3(torch.cat([self.convtrans2(up2), x05], dim=1))
        up4_ = self.up4(torch.cat([self.convtrans3(up3), x02], dim=1))
        up4 = torch.cat([self.convtrans4a(up4_), self.upsampling1(up4_)], dim=1)

        x = torch.cat([up4, x01], dim=1)
        x = self.conv_decoder1(x)
        x = torch.cat([self.convtrans4(x), self.upsampling2(x)], dim=1)
        x = self.conv_decoder2(x)
        if activate:
            x = self.activation(x)

        return x


class SqueezeUNet2DLargeBase(torch.nn.Module):

    @staticmethod
    def _check_dims(x):
        assert len(x.shape) == 4, "number of dims != 4"
        assert x.shape[2] % 32 == 0, 'height not mod 32'
        assert x.shape[3] % 32 == 0, 'width not mod 32'

    def __init__(self, channels: int = 1):
        super(SqueezeUNet2DLargeBase, self).__init__()

        # VGG style stage1 and stage2
        self.stage1 = torch.nn.Sequential(OrderedDict([
            ("conv1a", torch.nn.Conv2d(channels, 64, kernel_size=3, padding=1)),
            ("conv1a relu", torch.nn.ReLU()),
            ("conv1", torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            ("conv1 relu", torch.nn.ReLU())
        ]))

        self.stage2 = torch.nn.Sequential(OrderedDict([
            ("fire101", FireModule2D(64, squeeze=32, expand=128)),
            ("fire102", FireModule2D(256, squeeze=32, expand=128)),
            ("pool1", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        self.stage3 = torch.nn.Sequential(OrderedDict([
            ("fire2", FireModule2D(256, squeeze=64, expand=256)),
            ("fire21", FireModule2D(512, squeeze=64, expand=256)),
            ("fire3", FireModule2D(512, squeeze=64, expand=256)),
            ("pool3", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        self.stage4 = torch.nn.Sequential(OrderedDict([
            ("fire4", FireModule2D(512, squeeze=128, expand=512)),
            ("fire41", FireModule2D(1024, squeeze=128, expand=512)),
            ("fire5", FireModule2D(1024, squeeze=128, expand=512)),
            ("pool5", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        self.up2 = FireModule2D(1024, squeeze=64, expand=256)

        self.up3 = FireModule2D(512 + 64, squeeze=32, expand=128)
        self.convtrans2 = torch.nn.ConvTranspose2d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.up4 = FireModule2D(256 + 32, squeeze=16, expand=64)
        self.convtrans3 = torch.nn.ConvTranspose2d(256, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsampling1 = torch.nn.Upsample(scale_factor=2)
        self.convtrans4a = torch.nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_decoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(32 + 128 + 64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.convtrans4 = torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsampling2 = torch.nn.Upsample(scale_factor=2)

    def forward(self, x, check_dims=False):
        if check_dims:
            self._check_dims(x)

        x01 = self.stage1(x)
        x02 = self.stage2(x01)
        x05 = self.stage3(x02)
        x08 = self.stage4(x05)
        up2 = self.up2(x08)
        up3 = self.up3(torch.cat([self.convtrans2(up2), x05], dim=1))
        up4_ = self.up4(torch.cat([self.convtrans3(up3), x02], dim=1))
        up4 = torch.cat([self.convtrans4a(up4_), self.upsampling1(up4_)], dim=1)

        x = torch.cat([up4, x01], dim=1)
        x = self.conv_decoder1(x)
        x = torch.cat([self.convtrans4(x), self.upsampling2(x)], dim=1)

        return x


class FireModule3D(torch.nn.Module):
    """
    Single fire module of a squeeze net
    :param x: input
    :param squeeze: number of kernels in squeeze block (normally expand/4)
    :param expand: number of kernels in EACH of the pathways (normally 4*squeeze)
    :return: concatenated output from 1x1x1 and 3x3x3 pathways
    """

    def __init__(self, channels_in: int, squeeze: int = 16, expand: int = 64):
        super(FireModule3D, self).__init__()

        self.trunk = torch.nn.Sequential(OrderedDict([
            ("squeeze1x1", torch.nn.Conv3d(channels_in, squeeze, kernel_size=1)),
            ("squeeze1x1 relu", torch.nn.ReLU()),
            ("squeeze1x1 batchnorm", torch.nn.BatchNorm3d(squeeze))]))

        self.branch_left = torch.nn.Sequential(OrderedDict([
            ("expand1x1", torch.nn.Conv3d(squeeze, expand, kernel_size=1)),
            ("expand1x1 relu", torch.nn.ReLU())]))

        self.branch_right = torch.nn.Sequential(OrderedDict([
            ("expand3x3", torch.nn.Conv3d(squeeze, expand, kernel_size=3, padding=1)),
            ("expand3x3 relu", torch.nn.ReLU())]))

    def forward(self, x):
        x = self.trunk(x)
        l = self.branch_left(x)
        r = self.branch_right(x)
        return torch.cat([l, r], dim=1)


class SqueezeUNet3D(torch.nn.Module):
    """
    Implements the volumetric U-net with Squeeze Excitation modules
    With default arguments it corresponds to SqueezeNet5 from YokoTenkai
    With second_stride=2 and magic_upsampling=True it corresponds to SqueezeNet12
    The fire module is a 3D version of a layer proposed in:
    Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016).
    SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size. ARxiv.
    """

    def __init__(self, class_count: int = 1,
                 input_channels: int = 1,
                 deconv_ksize: int = 3,
                 activation=None,
                 first_stride: int = 1,
                 second_stride: int = 1,
                 depth: int = 5,
                 first_width: int = 24,
                 width_multiplier: int = 32,
                 magic_upsampling: bool = False):
        """
        :param input_channels: number of channels in the input volume.
        :param depth: total number of compressing layers (maxpools and conv with stride 2), min 3, default 5
        :param first_width: number of filters in the first and last conv layer, divisible by 2,  min 8, default 24
        :param width_multiplier: multiplier for the number of filters, divisible by 4, min 8, default 32
        :param class_count: number of output classes (default 1)
        :param deconv_ksize: size of the deconvolution kernels (default 3)
        :param activation: activation function in the last layer (default sigmoid for 1 class, softmax for more)
        :param first_stride: stride in z axis for the first compressing layer (default 1)
        :param second_stride: stride in z axis for the second compressing layer (default 1)
        :param magic_upsampling: mix of trans convolutions and upsampling (default False)
        magic_upsampling is a compromise between simple upsampling which is a straight-through parameter-less operation
        and transposed convolution where parameters of upsampling are learned from data.
        Upsampling brings quick initial learning (gradient passed straight through during backprop)
        and even fill of areas (transposed convs struggle here); transposed convs refine the upsampling slightly
        """
        super(SqueezeUNet3D, self).__init__()

        self.depth = depth
        self.magic_upsampling = magic_upsampling
        second_pool_size = 1 if second_stride == 1 else 3

        self.conv1 = torch.nn.Sequential(OrderedDict([
            ("conv1", torch.nn.Conv3d(input_channels, first_width,
                                      kernel_size=3,
                                      stride=(first_stride, 2, 2),
                                      padding=1)),
            ("conv1 relu", torch.nn.ReLU())]))

        self.pool1 = torch.nn.MaxPool3d(kernel_size=(second_pool_size, 3, 3),
                                        stride=(second_stride, 2, 2),
                                        padding=(second_pool_size // 2, 1, 1))

        self.initial_fires = torch.nn.Sequential(OrderedDict([
            ("fire1", FireModule3D(first_width, squeeze=width_multiplier // 2, expand=2 * width_multiplier)),
            ("fire2", FireModule3D(4 * width_multiplier, squeeze=width_multiplier // 2, expand=2 * width_multiplier)),
            ("fire3", FireModule3D(4 * width_multiplier, squeeze=width_multiplier // 2, expand=2 * width_multiplier))
        ]))

        self.down_blocks = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                f"pool{2 * dep + 1}": torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                "fires": torch.nn.Sequential(OrderedDict([
                    (f"fire{dep * 3 + 1}", FireModule3D(4 * dep * width_multiplier,
                                                        squeeze=int((dep + 1) / 2 * width_multiplier),
                                                        expand=2 * (dep + 1) * width_multiplier)),
                    (f"fire{dep * 3 + 2}", FireModule3D(4 * (dep + 1) * width_multiplier,
                                                        squeeze=int((dep + 1) / 2 * width_multiplier),
                                                        expand=2 * (dep + 1) * width_multiplier)),
                    (f"fire{dep * 3 + 3}", FireModule3D(4 * (dep + 1) * width_multiplier,
                                                        squeeze=int((dep + 1) / 2 * width_multiplier),
                                                        expand=2 * (dep + 1) * width_multiplier))
                ]))
            })
            for dep in range(1, depth - 1)])

        self.convtrans_bridge = torch.nn.ConvTranspose3d(4 * (self.depth - 1) * width_multiplier,
                                                         (depth - 2) * 2 * width_multiplier,
                                                         kernel_size=deconv_ksize,
                                                         padding=deconv_ksize // 2)

        self.up_blocks = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                f"fire{100 + dep + 1}": FireModule3D((depth - 2) * 2 * width_multiplier + 4 * (
                        depth - 2) * width_multiplier if dep == depth - 3 else max(2 * (dep + 1),
                                                                                   1) * width_multiplier + (4 * (
                        dep + 1) * width_multiplier),
                                                     squeeze=int((dep + 1) / 2 * width_multiplier),
                                                     expand=2 * (dep + 1) * width_multiplier),
                f"convtrans{100 + dep + 1}": torch.nn.ConvTranspose3d(4 * (dep + 1) * width_multiplier,
                                                                      max(2 * dep, 1) * width_multiplier,
                                                                      kernel_size=deconv_ksize,
                                                                      stride=2,
                                                                      padding=deconv_ksize // 2,
                                                                      output_padding=1)
            })
            for dep in range(depth - 3, -1, -1)
        ])

        self.fire100 = FireModule3D(width_multiplier + first_width, squeeze=width_multiplier // 2,
                                    expand=width_multiplier)

        self.upsampling1 = torch.nn.Upsample(scale_factor=(first_stride, 2, 2))
        self.convtrans_secondlast = torch.nn.ConvTranspose3d(2 * width_multiplier, width_multiplier // 2,
                                                             kernel_size=deconv_ksize,
                                                             stride=2,
                                                             padding=deconv_ksize // 2,
                                                             output_padding=1)

        upsampling2_channels = (
                width_multiplier // 2 + 2 * width_multiplier) if magic_upsampling else 2 * width_multiplier
        self.conv_secondlast = torch.nn.Sequential(
            torch.nn.Conv3d(first_width + upsampling2_channels, first_width,
                            kernel_size=3,
                            padding=1),
            torch.nn.ReLU())

        self.upsampling2 = torch.nn.Upsample(scale_factor=(first_stride, 2, 2))
        self.convtrans_last = torch.nn.ConvTranspose3d(2 * width_multiplier, width_multiplier // 2,
                                                       kernel_size=deconv_ksize,
                                                       stride=2,
                                                       padding=deconv_ksize // 2,
                                                       output_padding=1)

        upsampling2_channels = first_width // 2 + first_width if magic_upsampling else first_width
        self.conv_last = torch.nn.Conv3d(upsampling2_channels, class_count,
                                         kernel_size=1)

        self.activation = activation if activation is not None else (
            torch.nn.Sigmoid if class_count == 1 else torch.nn.Softmax(dim=1))

    def forward(self, x, activate=True):

        skips = []

        x = self.conv1(x)
        skips.append(x)

        x = self.pool1(x)
        skips.append(x)

        x = self.initial_fires(x)

        for block, dep in zip(self.down_blocks, range(1, self.depth - 1)):
            x = block[f"pool{2 * dep + 1}"](x)
            skips.append(x)

            x = block["fires"](x)

        y = torch.cat([self.convtrans_bridge(x), skips[-1]], dim=1)
        print("skips[-1].shape", skips[-1].shape)

        print("depth", self.depth)
        for block, dep in zip(self.up_blocks, range(self.depth - 3, -1, -1)):
            print(dep)
            y = block[f"fire{100 + dep + 1}"](y)

            y = torch.cat([block[f"convtrans{100 + dep + 1}"](y), skips[dep + 1]], dim=1)

        y = self.fire100(y)

        if self.magic_upsampling:
            y = torch.cat([self.convtrans_secondlast(y), self.upsampling1(y)], dim=1)
        else:
            y = self.upsampling1(y)

        y = torch.cat([y, skips[0]], dim=1)
        y = self.conv_secondlast(y)

        if self.magic_upsampling:
            y = torch.cat([self.convtrans_last(y), self.upsampling2(y)], dim=1)
        else:
            y = self.upsampling2(y)

        y = self.conv_last(y)

        if activate:
            y = self.activation(y)

        return y


if __name__ == '__main__':
    from torchsummary import summary

    model = SqueezeUnetMultiTask(1, 1, task_list=["test1", "test2", "test3"])

    summary(model, input_size=(1, 512, 512), device='cpu')

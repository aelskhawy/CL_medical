from collections import OrderedDict

import torch
import torch.nn as nn


class SSegHead(nn.Module):
    def __init__(self, head_task=''):
        super(SSegHead, self).__init__()
        feat = 32
        self.layers = nn.Sequential(OrderedDict([
            ("head_" + str(head_task) + "_conv1",
             nn.Conv2d(in_channels=feat, out_channels=feat, kernel_size=3, padding=1, bias=False)),
            ("head_" + str(head_task) + "_norm1", nn.BatchNorm2d(num_features=feat)),
            ("relu1", nn.ReLU(inplace=True)),
            ("head_" + str(head_task) + "_conv2",
             nn.Conv2d(in_channels=feat, out_channels=1, kernel_size=1, bias=True)),
        ]))
        # self.output = activation if activation else torch.nn.Sigmoid()


    def forward(self, x):
        # simple forward through the layers of the head
        return self.layers(x)


class UNetMultiHead(nn.Module):

    def __init__(self, in_channels=1, init_features=32, task_list: list = None, activation=None, use_lrelu=False):
        super(UNetMultiHead, self).__init__()
        self.__name__ = "UNetMultiHead"
        self.task_list = task_list
        self.activation = activation
        features = init_features
        self.use_lrelu = use_lrelu


        self.encoder1 = UNetMultiHead._block(self, in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetMultiHead._block(self, features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNetMultiHead._block(self, features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNetMultiHead._block(self, features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256*16*16

        self.bottleneck = UNetMultiHead._block(self, features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNetMultiHead._block(self, (features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNetMultiHead._block(self, (features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNetMultiHead._block(self, (features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNetMultiHead._block(self, features * 2, features, name="dec1")

        self.seg_heads = nn.ModuleList([SSegHead(head_task=i) for i in self.task_list])

        # self.lastconv = nn.Conv2d(
        #     in_channels=self.n_heads, out_channels=self.n_heads, kernel_size=3, padding=1)
        self.logsigma = nn.Parameter(torch.FloatTensor([0, float("-inf")]))


    def forward(self, x):
        enc1 = self.encoder1(x)
        # print("enc1".format(enc1.size()))
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        # dec1 = self.conv(dec1)

        # Collecting the outputs from the different heads

        seg_heads_out = []
        for head in self.seg_heads:
            seg_out = head(dec1)
            # print(head)
            seg_heads_out.append(seg_out)


        # This should contain channels = number of heads
        logits = torch.stack(seg_heads_out)  # [n_heads, bs, 1, h,w]
        logits = logits.squeeze(2).permute(1, 0, 2, 3)  # [bs, n_heads, h, w]

        # logits = self.lastconv(logits)
        return {"softmaxed_seg_logits": torch.softmax(logits, dim=1),
                "seg_logits": logits,
                "logsigma": self.logsigma}
        #return torch.softmax(logits, dim=1), logits  # softmax along the channel dimension torch.nn.functional.softmax

    # @staticmethod
    def _block(self, in_channels, features, name):
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
                    (name + "relu1", nn.ReLU(inplace=True)) if not self.use_lrelu else
                    (name + "LRelu1", nn.LeakyReLU(0.2, inplace=True)),
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
                    (name + "relu2", nn.ReLU(inplace=True)) if not self.use_lrelu else
                    (name + "LRelu1", nn.LeakyReLU(0.2, inplace=True)),
                ]
            )
        )

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (self.__name__, num_params / 1000000))

if __name__ == "__main__":
    unet = UNet(in_channels=3, out_channels=6, init_features=32, use_lrelu=True)
    unet.print_network()
    unet.to('cuda')
    # print(unet)
    from torchsummary import summary

    summary(unet, batch_size= 2,  input_size=(3, 256, 256))

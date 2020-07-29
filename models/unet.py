import torch


# Taken from https://github.com/milesial/Pytorch-UNet

class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = torch.nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(torch.nn.Module):
    def __init__(self, n_channels: int, n_classes: int, n_base_filters: int = 64,
                 activation: torch.nn.Module = None):
        super(UNet, self).__init__()
        self.__name__ = 'UNet'

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, n_base_filters)
        self.down1 = Down(n_base_filters, n_base_filters * 2)
        self.down2 = Down(n_base_filters * 2, n_base_filters * 2 ** 2)
        self.down3 = Down(n_base_filters * 2 ** 2, n_base_filters * 2 ** 3)
        self.down4 = Down(n_base_filters * 2 ** 3, n_base_filters * 2 ** 4)
        self.down5 = Down(n_base_filters * 2 ** 4, n_base_filters * 2 ** 4)
        self.up1 = Up(n_base_filters * 2 ** 5, n_base_filters * 2 ** 3)
        self.up2 = Up(n_base_filters * 2 ** 4, n_base_filters * 2 ** 2)
        self.up3 = Up(n_base_filters * 2 ** 3, n_base_filters * 2)
        self.up4 = Up(n_base_filters * 2 ** 2, n_base_filters)
        self.up5 = Up(n_base_filters * 2, n_base_filters)
        self.outc = OutConv(n_base_filters, n_classes)
        self.output = activation if activation else torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        output = self.output(logits)
        return output


if __name__ == '__main__':
    from torchsummary import summary

    model = UNet(1, 1, n_base_filters=16)

    summary(model, input_size=(1, 512, 512), device='cpu')

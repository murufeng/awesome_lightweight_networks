import torch.nn as nn
import torch
__all__ = ['mbv1']

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class DepthSepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1):
        super(DepthSepConv, self).__init__()

        self.conv_dw = nn.Conv2d(
            int(in_channels * channel_multiplier), int(in_channels * channel_multiplier), kernel_size,
            stride=stride, groups=int(in_channels * channel_multiplier), dilation=dilation, padding=padding)

        self.conv_pw = nn.Conv2d(int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding,
                                 bias=bias)

        self.relu = nn.ReLU(inplace=True)

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        x = self.relu(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )

        self.model = nn.Sequential(
            DepthSepConv(32, 64, stride=1),
            DepthSepConv(64, 128, stride=2),
            DepthSepConv(128, 128, stride=1),
            DepthSepConv(128, 256, stride=2),
            DepthSepConv(256, 256, stride=1),
            DepthSepConv(256, 512, stride=2),
            DepthSepConv(512, 512, stride=1),
            DepthSepConv(512, 512, stride=1),
            DepthSepConv(512, 512, stride=1),
            DepthSepConv(512, 512, stride=1),
            DepthSepConv(512, 512, stride=1),
            DepthSepConv(512, 1024, stride=2),
            DepthSepConv(1024, 1024, stride=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.model(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        #x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def mbv1(**kwargs):

    return MobileNetV1(**kwargs)

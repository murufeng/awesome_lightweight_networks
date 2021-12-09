'''
Inception Convolution with Efficient Dilation Search.
arxiv: https://arxiv.org/abs/2012.13587
code：https://github.com/yifan123/IC-Conv
'''

import torch
import torch.nn as nn
import re
import math
import json

pattern = None
pattern_index = -1

class ICConv2d(nn.Module):
    def __init__(self, pattern_dist, inplanes, planes, kernel_size, stride=1, groups=1, bias=False):
        super(ICConv2d, self).__init__()
        self.conv_list = nn.ModuleList()
        self.planes = planes
        for pattern in pattern_dist:
            channel = pattern_dist[pattern]
            pattern_trans = re.findall(r"\d+\.?\d*", pattern)
            pattern_trans[0] = int(pattern_trans[0]) + 1
            pattern_trans[1] = int(pattern_trans[1]) + 1
            if channel > 0:
                padding = [0, 0]
                padding[0] = (kernel_size + 2 * (pattern_trans[0] - 1)) // 2
                padding[1] = (kernel_size + 2 * (pattern_trans[1] - 1)) // 2
                self.conv_list.append(nn.Conv2d(inplanes, channel, kernel_size=kernel_size, stride=stride,
                                                padding=padding, bias=bias, groups=groups, dilation=pattern_trans))

    def forward(self, x):
        out = []
        for conv in self.conv_list:
            out.append(conv(x))
        out = torch.cat(out, dim=1)
        assert out.shape[1] == self.planes
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=3):
        super(BasicBlock, self).__init__()
        global pattern, pattern_index
        pattern_index = pattern_index + 1
        self.conv1 = ICConv2d(pattern[pattern_index], inplanes,
                              planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        pattern_index = pattern_index + 1
        self.conv2 = ICConv2d(
            pattern[pattern_index], planes, planes, kernel_size=kernel_size, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=3):
        super(Bottleneck, self).__init__()
        global pattern, pattern_index
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        pattern_index = pattern_index + 1
        self.conv2 = ICConv2d(
            pattern[pattern_index], planes, planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, kernel_size=3, pattern_path=None):

        super(ResNet, self).__init__()

        global pattern
        with open(pattern_path, 'r') as fin:
            pattern = json.load(fin)

        self.inplanes = 64
        self.kernel_size = kernel_size
        global pattern_index

        # ResNet
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        '''
        ResNet-D
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

        assert len(pattern) == pattern_index + 1

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, kernel_size=self.kernel_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                kernel_size=self.kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ic_resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def ic_resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def ic_resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def ic_resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def ic_resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


if __name__ == "__main__":
    patter = './pattern_zoo/detection/ic_resnet50_k9.json'
    model = ic_resnet50(pattern_path=patter)
    model.eval()
    print(model)
    input = torch.randn(1, 3, 224, 224)

    from thop import profile
    flops, params = profile(model=model, inputs=(input,))
    print('Model:{:.2f} GFLOPs and {:.2f}M parameters'.format(flops / 1e9, params / 1e6))
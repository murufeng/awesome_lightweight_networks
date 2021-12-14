import torch
import torch.nn as nn

def conv_bn(in_channels,out_channels,kernel_size, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, stride=stride,
                  padding=kernel_size // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels)
    )

class GlobalAveragePool2D():
    def __init__(self, keepdim=True):
        self.keepdim = keepdim

    def __call__(self, inputs):
        return torch.mean(inputs, axis=[2, 3], keepdim=self.keepdim)


class SSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSEBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.norm = nn.BatchNorm2d(self.in_channels)
        self.globalAvgPool = GlobalAveragePool2D()
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        bn = self.norm(inputs)
        x = self.globalAvgPool(bn)
        x = self.conv(x)
        x = self.sigmoid(x)

        z = torch.mul(bn, x)
        return z

class Downsampling_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsampling_block, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels

        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv1 = conv_bn(self.in_channels, self.out_channels, kernel_size=1)
        self.conv2 = conv_bn(self.in_channels, self.out_channels, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.globalAvgPool = GlobalAveragePool2D()
        self.act = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.avgpool(inputs)
        x = self.conv1(x)

        y = self.conv2(inputs)

        z = self.globalAvgPool(inputs)
        z = self.conv3(z)
        z = self.sigmoid(z)

        a = x + y
        b = torch.mul(a, z)
        out = self.act(b)
        return out

class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = 2 * self.in_channels
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv1 = conv_bn(self.mid_channels, self.out_channels, kernel_size=1, stride=1, groups=2)
        self.conv2 = conv_bn(self.mid_channels, self.out_channels, kernel_size=3, stride=2, groups=2)
        self.conv3 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.out_channels, kernel_size=1, groups=2)
        self.globalAvgPool = GlobalAveragePool2D()
        self.act = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.group = in_channels

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def forward(self, input1, input2):

        a = torch.cat([self.bn(input1), self.bn(input2)], dim=1)

        a = self.channel_shuffle(a)

        x = self.avgpool(a)

        x = self.conv1(x)

        y = self.conv2(a)

        z = self.globalAvgPool(a)

        z = self.conv3(z)
        z = self.sigmoid(z)

        a = x + y

        b = torch.mul(a, z)
        out = self.act(b)
        return out

class Stream(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sse = nn.Sequential(SSEBlock(self.in_channels, self.out_channels))
        self.fuse = nn.Sequential(FuseBlock(self.in_channels, self.out_channels))
        self.act = nn.SiLU(inplace=True)

    def forward(self, inputs):
        a = self.sse(inputs)
        b = self.fuse(inputs)
        c = a + b

        d = self.act(c)
        return d


class FuseBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = conv_bn(self.in_channels, self.out_channels, kernel_size=1)
        self.conv2 = conv_bn(self.in_channels, self.out_channels, kernel_size=3, stride=1)

    def forward(self, inputs):
        a = self.conv1(inputs)
        b = self.conv2(inputs)

        c = a + b
        return c


class ParNetEncoder(nn.Module):
    def __init__(self, in_channels, block_channels, depth) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.block_channels = block_channels
        self.depth = depth
        self.d1 = Downsampling_block(self.in_channels, self.block_channels[0])
        self.d2 = Downsampling_block(self.block_channels[0], self.block_channels[1])
        self.d3 = Downsampling_block(self.block_channels[1], self.block_channels[2])
        self.d4 = Downsampling_block(self.block_channels[2], self.block_channels[3])
        self.d5 = Downsampling_block(self.block_channels[3], self.block_channels[4])
        self.stream1 = nn.Sequential(
            *[Stream(self.block_channels[1], self.block_channels[1]) for _ in range(self.depth[0])]
        )

        self.stream1_downsample = Downsampling_block(self.block_channels[1], self.block_channels[2])

        self.stream2 = nn.Sequential(
            *[Stream(self.block_channels[2], self.block_channels[2]) for _ in range(self.depth[1])]
        )

        self.stream3 = nn.Sequential(
            *[Stream(self.block_channels[3], self.block_channels[3]) for _ in range(self.depth[2])]
        )

        self.stream2_fusion = Fusion(self.block_channels[2], self.block_channels[3])
        self.stream3_fusion = Fusion(self.block_channels[3], self.block_channels[3])

    def forward(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)

        y = self.stream1(x)
        y = self.stream1_downsample(y)

        x = self.d3(x)

        z = self.stream2(x)
        z = self.stream2_fusion(y, z)

        x = self.d4(x)

        a = self.stream3(x)
        b = self.stream3_fusion(z, a)

        x = self.d5(b)
        return x


class ParNetDecoder(nn.Module):
    def __init__(self, in_channels, n_classes) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_channels, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return self.softmax(x)


class ParNet(nn.Module):
    def __init__(self, in_channels, n_classes, block_channels=[64, 128, 256, 512, 2048], depth=[4, 5, 5]) -> None:
        super().__init__()
        self.encoder = ParNetEncoder(in_channels, block_channels, depth)
        self.decoder = ParNetDecoder(block_channels[-1], n_classes)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)

        return x

def parnet_s(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_channels=[64, 96, 192, 384, 1280])


def parnet_m(in_channels, n_classes):
    model = ParNet(in_channels, n_classes, block_channels=[64, 128, 256, 512, 2048])
    return model


def parnet_l(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_channels=[64, 160, 320, 640, 2560])


def parnet_xl(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_channels=[64, 200, 400, 800, 3200])


if __name__ == '__main__':
    model = parnet_s(3, 1000)
    model.eval()
    print(model)
    input = torch.randn(1, 3, 256, 256)
    y = model(input)
    print(y.size())





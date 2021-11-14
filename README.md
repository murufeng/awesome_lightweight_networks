# awesome_lightweight_networks

![](https://img.shields.io/badge/awesome_lightweight_networks-v0.1.0-brightgreen)
![](https://img.shields.io/badge/python->=v3.0-blue)
![](https://img.shields.io/badge/pytorch->=v1.4-red)

[![GitHub stars](https://img.shields.io/github/stars/murufeng/awesome_lightweight_networks.svg?style=social&label=Stars)](https://github.com/murufeng/awesome_lightweight_networks)
[![GitHub forks](https://img.shields.io/github/forks/murufeng/awesome_lightweight_networks.svg?style=social&label=Forks)](https://github.com/murufeng/awesome_lightweight_networks)

![](./figures/view.jpg)

目前在深度学习领域主要分为两类，一派为学院派(Researcher)，研究强大、复杂的模型网络和实验方法，旨在追求更高的性能；
另一派为工程派(Engineer)，旨在将算法更稳定、更高效的落地部署在不同硬件平台上。因此，针对这些移动端的算力设备，
如何去设计一种高效且精简的网络架构就显得尤为重要。从2017年以来，已出现了很多优秀实用的轻量级网络架构，但是还没有一个通用的项目把这些网络架构进行集成起来。**本项目可以作为一个即插即用的工具包，通过直接调用就可以直接训练各种类型的网络**。
目前该项目暂时支持在Cifar10/100,ImageNet数据集上进行实验。**后续会持续针对每个不同的具体任务，更新工业界比较实用的SOTA的网络架构模型**。
本项目主要提供一个移动端网络架构的基础性工具，避免重复造轮子，后续我们将针对具体视觉任务集成更多的网络架构。希望本项目既能**让深度学习初学者快速入门**，又能**服务科研和工业社区**。（同时也非常欢迎各位科研工作者将自己的工作的核心代码整理到本项目中，推动科研社区的发展，会在readme中注明代码的作者~）

## Table of Contents
### [MobileNets系列](#MobileNet)
  
### [ShuffleNet系列](#ShuffleNet)

### [华为诺亚轻量级网络系列](#noah)

### [轻量级注意力网络架构](#attention)

### [Transfor轻量级网络结构](#vit)

### [移动端部署CPU网络架构](#cpu)

### [CondenseNet系列](#condense)

### [轻量级检测网络结构](#det)

### [轻量级图像分割网络架构](#seg)

### [轻量级图像去噪网络架构](#denoise)

### [轻量级图像超分网络架构](#super)

### [轻量级人体姿态估计网络架构](#hrnet)

### [轻量级的卷积操作](#conv)

### [模型压缩方法汇总](#compress)

### [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187v2)

#### 网络结构
![](./figures/resnet_d.jpg)

#### Code

```python
import torch
from light_cnns import resnet50_v1b
model = resnet50_v1b()
model.eval()
print(model)
input = torch.randn(1, 3, 224, 224)
y = model(input)
print(y.size())
```

<a name="MobileNet"></a>
### MobileNets系列
   - [MobileNetV1](#mbv1)
   - [MobileNetV2](#mbv2)
   - [MobileNetV3](#mbv3)
   - [MobileNeXt](#mbnext)

<a name="mbv1"></a>  
#### MobileNetv1 网络模块
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

![](./figures/mbv1.jpg)

#### Code
```python
import torch
from light_cnns import mbv1
model = mbv1()
model.eval()
print(model)
input = torch.randn(1, 3, 224, 224)
y = model(input)
print(y.size())
```
<a name="mbv2"></a>  
#### MobileNetv2 网络模块
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

![](./figures/mbv2.jpg)

#### Code
```python
import torch
from light_cnns import mbv2
model = mbv2()
model.eval()
print(model)
input = torch.randn(1, 3, 224, 224)
y = model(input)
print(y.size())
```
<a name="mbnext"></a>  
#### MobileNeXt 网络模块

- [Rethinking Bottleneck Structure for Efficient Mobile Network Design](https://arxiv.org/abs/2007.02269)
![](./figures/mbnext.jpg)
#### Code
```python
import torch
from light_cnns import mobilenext
model = mobilenext()
model.eval()
print(model)
input = torch.randn(1, 3, 224, 224)
y = model(input)
print(y.size())
```

<a name="ShuffleNet"></a>
#### ShuffleNet
- [ShuffleNetv1](#shffv1)
- [ShuffleNetV2](#shffv2)

<a name="shffv1"></a>  
#### ShuffleNetv1 网络模块

- [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)

![](./figures/shuffv1.jpg)

#### Code
```python
import torch
from light_cnns import shufflenetv1
model = shufflenetv1()
model.eval()
print(model)
input = torch.randn(1, 3, 224, 224)
y = model(input)
print(y.size())
```

<a name="shffv2"></a>  
#### ShuffleNetv2 网络模块
- [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164v1)

![](./figures/shuffv2.jpg)

#### Code
```python
import torch
from light_cnns import shufflenetv2
model = shufflenetv2()
model.eval()
print(model)
input = torch.randn(1, 3, 224, 224)
y = model(input)
print(y.size())
```

<a name="noah"></a>
### 华为诺亚方舟系列
- [AdderNet](#add)
- [GhostNet](#ghost)

<a name="add"></a>
#### AdderNet（加法网络)
- [AdderNet and its Minimalist Hardware Design for Energy-Efficient Artificial Intelligence](https://arxiv.org/abs/2101.10015)

![](./figures/adder.jpg)

#### Code
```python
import torch
from light_cnns import resnet20
model = resnet20()
model.eval()
print(model)
input = torch.randn(1, 3, 224, 224)
y = model(input)
print(y.size())
```
<a name="ghost"></a>
#### GhostNet
- [GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907)
![](./figures/ghost.jpg)

#### Code
```python
import torch
from light_cnns import ghostnet
model = ghostnet()
model.eval()
print(model)
input = torch.randn(1, 3, 224, 224)
y = model(input)
print(y.size())
```

<a name="attention"></a>
### 注意力系列

#### CANet
- [Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907)

![](./figures/canet.jpg)

#### Code
```python
import torch
from light_cnns import mbv2_ca
model = mbv2_ca()
model.eval()
print(model)
input = torch.randn(1, 3, 224, 224)
y = model(input)
print(y.size())
```

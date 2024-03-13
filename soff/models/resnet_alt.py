'''ResNet (Alt) in PyTorch

from: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

from functools import partial
from torch import nn
from torch.nn import init
import torch.nn.functional as F


def _weights_init(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(module.weight)


class _LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, batch):
        return self.lambd(batch)


class _BasicBlock(nn.Module):
    """ResNet basic block"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # For CIFAR10 ResNet paper uses option A.
            if option == 'A':
                self.shortcut = _LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, self.expansion * planes,
                        kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(
                        self.expansion * planes, track_running_stats=False))

    def forward(self, batch):
        out = F.relu(self.bn1(self.conv1(batch)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(batch)
        out = F.relu(out)
        return out


class _Resnet(nn.Module):
    """The ResNet class"""

    def __init__(self, block, num_blocks, _, dataset):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, track_running_stats=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, dataset.num_classes())
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stri in strides:
            layers.append(block(self.in_planes, planes, stri))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, batch):
        out = F.relu(self.bn1(self.conv1(batch)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Resnet20(_Resnet):
    def __init__(self, cfg, dataset):
        super().__init__(_BasicBlock, [3, 3, 3], cfg, dataset)


class Resnet32(_Resnet):
    def __init__(self, cfg, dataset):
        super().__init__(_BasicBlock, [5, 5, 5], cfg, dataset)


class Resnet44(_Resnet):
    def __init__(self, cfg, dataset):
        super().__init__(_BasicBlock, [7, 7, 7], cfg, dataset)


class Resnet56(_Resnet):
    def __init__(self, cfg, dataset):
        super().__init__(_BasicBlock, [9, 9, 9], cfg, dataset)


class Resnet110(_Resnet):
    def __init__(self, cfg, dataset):
        super().__init__(_BasicBlock, [18, 18, 18], cfg, dataset)


class Resnet1202(_Resnet):
    def __init__(self, cfg, dataset):
        super().__init__(_BasicBlock, [200, 200, 200], cfg, dataset)

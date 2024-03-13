'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch.nn.functional as F
from torch import nn
from ..utils.arg_parser import ArgParseOption, options, require


class _BasicBlock(nn.Module):
    """Small ResNet block"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, track_running_stats=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.batchnorm1 = nn.BatchNorm2d(
            planes, track_running_stats=track_running_stats)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(
            planes, track_running_stats=track_running_stats)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1,
                    stride=stride, bias=False),
                nn.BatchNorm2d(
                    self.expansion * planes,
                    track_running_stats=track_running_stats))

    def forward(self, batch):
        """Overrides Module.forward"""
        out = F.relu(self.batchnorm1(self.conv1(batch)))
        out = self.batchnorm2(self.conv2(out))
        out += self.shortcut(batch)
        out = F.relu(out)
        return out


class _Bottleneck(nn.Module):
    """Large ResNet block"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, track_running_stats=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(
            planes, track_running_stats=track_running_stats)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.batchnorm2 = nn.BatchNorm2d(
            planes, track_running_stats=track_running_stats)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(
            self.expansion * planes, track_running_stats=track_running_stats)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1,
                    stride=stride, bias=False),
                nn.BatchNorm2d(
                    self.expansion * planes,
                    track_running_stats=track_running_stats))

    def forward(self, batch):
        """Overrides Module.forward"""
        out = F.relu(self.batchnorm1(self.conv1(batch)))
        out = F.relu(self.batchnorm2(self.conv2(out)))
        out = self.batchnorm3(self.conv3(out))
        out += self.shortcut(batch)
        out = F.relu(out)
        return out


@require('model.resnet.batchnorm_runstat')
@options(
    "ResNet Model Configs",
    ArgParseOption(
        'res.br', 'resnet.batchnorm-runstat',
        action='store_true',
        help="Track the running statistics of batchnorm layer"))
class _Resnet(nn.Module):
    """The ResNet Model"""

    def __init__(self, block, num_blocks, cfg, dataset):
        super().__init__()
        self.in_planes = 64
        self.track_running_stats = cfg.model.resnet.batchnorm_runstat

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(
            64, track_running_stats=self.track_running_stats)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(
            512 * block.expansion, dataset.num_classes())

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stri in strides:
            layers.append(block(
                self.in_planes, planes, stri, self.track_running_stats))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, batch):
        """Overrides Module.forward"""
        out = F.relu(self.batchnorm1(self.conv1(batch)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Resnet18(_Resnet):
    def __init__(self, cfg, dataset):
        super().__init__(_BasicBlock, [2, 2, 2, 2], cfg, dataset)


class Resnet34(_Resnet):
    def __init__(self, cfg, dataset):
        super().__init__(_BasicBlock, [3, 4, 6, 3], cfg, dataset)


class Resnet50(_Resnet):
    def __init__(self, cfg, dataset):
        super().__init__(_Bottleneck, [3, 4, 6, 3], cfg, dataset)


class Resnet101(_Resnet):
    def __init__(self, cfg, dataset):
        super().__init__(_Bottleneck, [3, 4, 23, 3], cfg, dataset)


class Resnet152(_Resnet):
    def __init__(self, cfg, dataset):
        super().__init__(_Bottleneck, [3, 8, 36, 3], cfg, dataset)

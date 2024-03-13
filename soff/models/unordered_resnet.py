
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import re
import logging
from munch import Munch
import torch
from torch import nn
import torch.nn.functional as F
from ..datasets.fl_split.base import _FLSplit
from ..utils.arg_parser import ArgParseOption, options
from .unordered_net import (
    _UnorderedNet, _MatchableLayer, _MatchableConv2d,
    _MatchableBatchNorm2d, _MatchableLinear)

log = logging.getLogger(__name__)


class _ShortcutNonExpanding(nn.Module):
    def __init__(self, num_neurons: int, root_module=None):
        super().__init__()
        self.num_neurons = num_neurons

        # dummy weight and bias, for matching/updating purpose
        self.register_parameter("weight", nn.Parameter(torch.Tensor()))
        self.register_parameter("bias", None)

        # as pytorch overwrites __setattr__ and __getattr__,
        # direct assignment will cause infinite recursion
        # (for access, we can directly use `self.root_module`,
        # since __getattribute__ is not overriden)
        object.__setattr__(self, "root_module", root_module)

    def forward(self, x):
        # server shorcut relies on mapping to work. Client side doesn't.
        if object.__getattribute__(self, "root_module").is_client:
            return x

        new_shape = list(x.shape)

        # We also need to preturb the input according to input order
        # This is just a monkey patch. We need a more systematic way
        # to solve this issue
        name_dict = {v: k for k, v in self.root_module.named_modules()}
        input_orders = self.root_module.calc_input_orders(name_dict[self])
        assert len(input_orders[0]) == self.orig_output_neurons

        new_shape[1] = self.orig_output_neurons
        restored_out = torch.zeros(new_shape).to(x.device)
        for order in input_orders:
            restored_out += torch.index_select(x, 1, order.to(x.device))
        restored_out /= len(input_orders)

        # TODO: this can be optimized (on the serverside).
        assert len(self.map_orders[0]) == restored_out.shape[1]
        assert len(self.map_orders) == len(self.map_weights)

        new_shape[1] = self.map_size
        out = torch.zeros(new_shape).to(x.device)
        for order, weight in zip(self.map_orders, self.map_weights):
            out.index_add_(1, order.to(x.device), restored_out * weight)
        return out


class _MatchableShorcut(_MatchableLayer, _ShortcutNonExpanding):
    def get_out_size(self):
        return self.num_neurons


class _UnorderedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, track_running_stats=True,
                 root_module=None):
        super(_UnorderedBasicBlock, self).__init__()

        self.conv1 = _MatchableConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.conv2 = _MatchableConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # For conv layers, the equivalent number of neurons is equivalent to
        # number out output planes (which is the same as the bias term)
        for conv in (self.conv1, self.conv2):
            conv.batchnorm = _MatchableBatchNorm2d(
                planes, track_running_stats=track_running_stats)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut1 = _MatchableConv2d(
                in_planes, self.expansion * planes, kernel_size=1,
                stride=stride, bias=False)
            self.shortcut1.batchnorm = _MatchableBatchNorm2d(
                self.expansion * planes,
                track_running_stats=track_running_stats)
        else:
            self.shortcut1 = _MatchableShorcut(
                self.expansion * planes, root_module)
        self.shortcut2 = _MatchableShorcut(
            self.expansion * planes, root_module)

    def forward(self, x):
        out = x

        for conv in (self.conv1, self.conv2):
            out = conv(out)
            out = conv.batchnorm(out)
            if conv != self.conv2:
                out = F.relu(out)

        shortcut_out = self.shortcut1(x)
        if hasattr(self.shortcut1, "batchnorm"):
            shortcut_out = self.shortcut1.batchnorm(shortcut_out)
        shortcut_out = self.shortcut2(shortcut_out)

        out += shortcut_out
        out = F.relu(out)
        return out


class _UnorderedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, track_running_stats=True,
                 root_module=False):
        super(_UnorderedBottleneck, self).__init__()

        self.conv1 = _MatchableConv2d(
            in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = _MatchableConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.conv3 = _MatchableConv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False)

        for conv in (self.conv1, self.conv2, self.conv3):
            conv.batchnorm = _MatchableBatchNorm2d(
                conv.out_channels, track_running_stats=track_running_stats)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut1 = _MatchableConv2d(
                in_planes, self.expansion * planes, kernel_size=1,
                stride=stride, bias=False)
            self.shortcut1.batchnorm = _MatchableBatchNorm2d(
                self.expansion * planes,
                track_running_stats=track_running_stats)
        else:
            self.shortcut1 = _MatchableShorcut(
                self.expansion * planes, root_module)

        self.shortcut2 = _MatchableShorcut(
            self.expansion * planes, root_module)
        self.shortcut3 = _MatchableShorcut(
            self.expansion * planes, root_module)

    def forward(self, x):
        shortcut_out = self.shortcut.batchnorm(self.shortcut(x))
        out = x

        for conv in (self.conv1, self.conv2, self.conv3):
            out = conv(out)
            out = conv.batchnorm(out)
            if conv != self.conv3:
                out = F.relu(out)

        shortcut_out = self.shortcut1.shortcut1(x)
        if hasattr(self.shortcut1, "batchnorm"):
            shortcut_out = self.shortcut1.batchnorm(shortcut_out)
        shortcut_out = self.shortcut2(shortcut_out)
        shortcut_out = self.shortcut3(shortcut_out)

        out += shortcut_out
        out = F.relu(out)
        return out


@options(
    "Unordered Resnet Model Configs",
    ArgParseOption(
        'ures.tr', 'unordered-resnet.track-runstat',
        action='store_true',
        help="Track running stats of batchnorm layer."))
class _UnorderedResnet(_UnorderedNet):
    def __init__(self, block, num_blocks, cfg: Munch, dataset: _FLSplit):
        self.in_planes = 64
        self.is_client = hasattr(cfg, 'client_id')
        super().__init__(
            block, num_blocks, dataset.num_classes(),
            cfg.model.unordered_resnet.track_runstat)

    def init_submodules(
            self, block, num_blocks, num_classes, track_running_stats):

        self.conv1 = _MatchableConv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], stride=1,
            track_running_stats=track_running_stats, root_module=self)
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2,
            track_running_stats=track_running_stats, root_module=self)
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2,
            track_running_stats=track_running_stats, root_module=self)
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2,
            track_running_stats=track_running_stats, root_module=self)
        self.linear = _MatchableLinear(512 * block.expansion, num_classes)

        self.conv1.batchnorm = _MatchableBatchNorm2d(
            64, track_running_stats=track_running_stats)

    def init_matchable_layers(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and \
                    'shortcut' not in name:
                # If the matching layer had a parallel shortcut,
                # preturbing the layer and the shortcut simultaneously
                modules = dict(self.named_modules())
                parent_module = '.'.join(name.split('.')[:-1])
                if isinstance(modules[parent_module],
                              (_UnorderedBasicBlock, _UnorderedBottleneck)):

                    # conv layer, parallel shortcut, and conv's batchnorm
                    parallel_shortcut = re.sub('conv', 'shortcut', name)
                    self.matchable_layers.append(
                        [(name, module),
                         (parallel_shortcut, modules[parallel_shortcut]),
                         (name+'.batchnorm', modules[name + '.batchnorm'])])

                    # shortcut's batchnorm, if exists
                    sc_bnorm = re.sub('conv', 'shortcut', name) + '.batchnorm'
                    if sc_bnorm in modules:
                        self.matchable_layers[-1] += \
                            [(sc_bnorm, modules[sc_bnorm])]
                else:
                    self.matchable_layers.append([(name, module)])
                    if name+'.batchnorm' in modules:
                        self.matchable_layers[-1] += [
                            (name+'.batchnorm', modules[name + '.batchnorm'])]

    def _make_layer(self, block, planes, num_blocks, stride,
                    track_running_stats, root_module):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride,
                      track_running_stats, root_module))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.conv1.batchnorm(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class UnorderedResnet18(_UnorderedResnet):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__(
            _UnorderedBasicBlock, [2, 2, 2, 2], cfg, dataset)


class UnorderedResnet34(_UnorderedResnet):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__(
            _UnorderedBasicBlock, [3, 4, 6, 3], cfg, dataset)


class UnorderedResnet50(_UnorderedResnet):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__(
            _UnorderedBottleneck, [3, 4, 6, 3], cfg, dataset)


class UnorderedResnet101(_UnorderedResnet):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__(
            _UnorderedBottleneck, [3, 4, 23, 3], cfg, dataset)


class UnorderedResnet152(_UnorderedResnet):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__(
            _UnorderedBottleneck, [3, 8, 36, 3], cfg, dataset)

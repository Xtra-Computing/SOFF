"""InceptionNet network for the FedGMA algorithm"""

import re
import logging
from typing import List, Tuple
from itertools import accumulate
from collections import namedtuple
import torch
import torch.nn.functional as F
from torch import nn
from munch import Munch
from ..datasets.fl_split.base import _FLSplit
from ..utils.arg_parser import ArgParseOption, options
from .unordered_net import (
    _UnorderedNet, _MatchableConv2d, _MatchableBatchNorm2d, _MatchableLinear)

_InceptionOuputs = namedtuple("_InceptionOuputs", ["logits", "aux_logits"])

log = logging.getLogger(__name__)


class _BasicConv2d(nn.Module):
    def __init__(
            self, in_channels, out_channels, track_running_stats, /, **kwargs):
        super().__init__()
        self.conv = _MatchableConv2d(
            in_channels, out_channels, bias=False, **kwargs)
        self.bn = _MatchableBatchNorm2d(
            out_channels, track_running_stats=track_running_stats, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class _ShortcutRestore(nn.Module):
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

        name_dict = {v: k for k, v in self.root_module.named_modules()}
        input_orders = self.root_module.calc_input_orders(name_dict[self])
        assert len(input_orders[0]) == self.num_neurons

        new_shape[1] = self.num_neurons
        restored_out = torch.zeros(new_shape).to(x.device)
        for order in input_orders:
            restored_out += torch.index_select(x, 1, order.to(x.device))
        restored_out /= len(input_orders)

        return restored_out


def _calc_map_size(*layers, restore_branchpool_layer=None):
    map_sizes = [layer.map_size for layer in layers]
    if restore_branchpool_layer is not None:
        return sum(map_sizes) + restore_branchpool_layer.num_neurons
    else:
        return sum(map_sizes)


def _calc_map_orders(*layers, restore_branchpool_layer=None):
    map_sizes = [layer.map_size for layer in layers]
    map_sizes = list(accumulate([0] + map_sizes))

    additional_orders = []
    if any([len(layer.map_orders) == 1 for layer in layers]):
        if restore_branchpool_layer is not None:
            additional_orders = [
                torch.arange(restore_branchpool_layer.num_neurons)
                .repeat(1, 1) + map_sizes[-1]]
        return list(torch.hstack([
            torch.stack(layer.map_orders[:1]) + map_sizes[i]
            for i, layer in enumerate(layers)] + additional_orders))
    else:
        if restore_branchpool_layer is not None:
            additional_orders = [
                torch.arange(restore_branchpool_layer.num_neurons)
                .repeat(len(layers[0].map_orders), 1) + map_sizes[-1]]
        return list(torch.hstack([
            torch.stack(layer.map_orders) + map_sizes[i]
            for i, layer in enumerate(layers)] + additional_orders))


class _InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features, track_running_stats):
        super(_InceptionA, self).__init__()
        self.branch1x1_1 = _BasicConv2d(
            in_channels, 64, track_running_stats, kernel_size=1)

        self.branch5x5_1 = _BasicConv2d(
            in_channels, 48, track_running_stats, kernel_size=1)
        self.branch5x5_2 = _BasicConv2d(
            48, 64, track_running_stats, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = _BasicConv2d(
            in_channels, 64, track_running_stats, kernel_size=1)
        self.branch3x3dbl_2 = _BasicConv2d(
            64, 96, track_running_stats, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = _BasicConv2d(
            96, 96, track_running_stats, kernel_size=3, padding=1)

        self.branch_pool = _BasicConv2d(
            in_channels, pool_features, track_running_stats, kernel_size=1)

        # Used by main module to help calculate next layers:
        self._next_layers = {
            'branch1x1_1': None,
            'branch5x5_1': 'branch5x5_2',
            'branch5x5_2': None,
            'branch3x3dbl_1': 'branch3x3dbl_2',
            'branch3x3dbl_2': 'branch3x3dbl_3',
            'branch3x3dbl_3': None,
            'branch_pool': None,
        }

    def forward(self, x):
        branch1x1 = self.branch1x1_1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

    def calc_output_size(self):
        return _calc_map_size(
            self.branch1x1_1.conv,
            self.branch5x5_2.conv,
            self.branch3x3dbl_3.conv,
            self.branch_pool.conv,
            restore_branchpool_layer=None)

    def calc_output_orders(self):
        return _calc_map_orders(
            self.branch1x1_1.conv,
            self.branch5x5_2.conv,
            self.branch3x3dbl_3.conv,
            self.branch_pool.conv,
            restore_branchpool_layer=None)


class _InceptionB(nn.Module):
    def __init__(self, in_channels, root_module, track_running_stats):
        super(_InceptionB, self).__init__()
        self.branch3x3_1 = _BasicConv2d(
            in_channels, 384, track_running_stats, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = _BasicConv2d(
            in_channels, 64, track_running_stats, kernel_size=1)
        self.branch3x3dbl_2 = _BasicConv2d(
            64, 96, track_running_stats, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = _BasicConv2d(
            96, 96, track_running_stats, kernel_size=3, stride=2)

        # NOTE: There's no actual branch_pool layer.
        self.sc = _ShortcutRestore(in_channels, root_module)

        self._next_layers = {
            'branch3x3_1': None,
            'branch3x3dbl_1': 'branch3x3dbl_2',
            'branch3x3dbl_2': 'branch3x3dbl_3',
            'branch3x3dbl_3': None,
        }

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.sc(x)
        branch_pool = F.max_pool2d(branch_pool, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

    def calc_output_size(self):
        return _calc_map_size(
            self.branch3x3_1.conv,
            self.branch3x3dbl_3.conv,
            restore_branchpool_layer=self.sc)

    def calc_output_orders(self):
        return _calc_map_orders(
            self.branch3x3_1.conv,
            self.branch3x3dbl_3.conv,
            restore_branchpool_layer=self.sc)


class _InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7, track_running_stats):
        super(_InceptionC, self).__init__()
        self.branch1x1_1 = _BasicConv2d(
            in_channels, 192, track_running_stats, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = _BasicConv2d(
            in_channels, c7, track_running_stats, kernel_size=1)
        self.branch7x7_2 = _BasicConv2d(
            c7, c7, track_running_stats, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = _BasicConv2d(
            c7, 192, track_running_stats, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = _BasicConv2d(
            in_channels, c7, track_running_stats, kernel_size=1)
        self.branch7x7dbl_2 = _BasicConv2d(
            c7, c7, track_running_stats, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = _BasicConv2d(
            c7, c7, track_running_stats, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = _BasicConv2d(
            c7, c7, track_running_stats, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = _BasicConv2d(
            c7, 192, track_running_stats, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = _BasicConv2d(
            in_channels, 192, track_running_stats, kernel_size=1)

        # Used by main module to help calculate next layers:
        self._next_layers = {
            'branch1x1_1': None,
            'branch7x7_1': 'branch7x7_2',
            'branch7x7_2': 'branch7x7_3',
            'branch7x7_3': None,
            'branch7x7dbl_1': 'branch7x7dbl_2',
            'branch7x7dbl_2': 'branch7x7dbl_3',
            'branch7x7dbl_3': 'branch7x7dbl_4',
            'branch7x7dbl_4': 'branch7x7dbl_5',
            'branch7x7dbl_5': None,
            'branch_pool': None,
        }

    def forward(self, x):
        branch1x1 = self.branch1x1_1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)

    def calc_output_size(self):
        return _calc_map_size(
            self.branch1x1_1.conv,
            self.branch7x7_3.conv,
            self.branch7x7dbl_5.conv,
            self.branch_pool.conv,
            restore_branchpool_layer=None)

    def calc_output_orders(self):
        return _calc_map_orders(
            self.branch1x1_1.conv,
            self.branch7x7_3.conv,
            self.branch7x7dbl_5.conv,
            self.branch_pool.conv,
            restore_branchpool_layer=None)


class _InceptionD(nn.Module):
    def __init__(self, in_channels, root_module, track_running_stats):
        super(_InceptionD, self).__init__()
        self.branch3x3_1 = _BasicConv2d(
            in_channels, 192, track_running_stats, kernel_size=1)
        self.branch3x3_2 = _BasicConv2d(
            192, 320, track_running_stats, kernel_size=3, stride=2)

        self.branch7x7x3_1 = _BasicConv2d(
            in_channels, 192, track_running_stats, kernel_size=1)
        self.branch7x7x3_2 = _BasicConv2d(
            192, 192, track_running_stats, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = _BasicConv2d(
            192, 192, track_running_stats, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = _BasicConv2d(
            192, 192, track_running_stats, kernel_size=3, stride=2)

        # NOTE: There's no actual branch_pool layer.
        self.sc = _ShortcutRestore(in_channels, root_module)

        self._next_layers = {
            'branch3x3_1': 'branch3x3_2',
            'branch3x3_2': None,
            'branch7x7x3_1': 'branch7x7x3_2',
            'branch7x7x3_2': 'branch7x7x3_3',
            'branch7x7x3_3': 'branch7x7x3_4',
            'branch7x7x3_4': None,
        }

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.sc(x)
        branch_pool = F.max_pool2d(branch_pool, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)

    def calc_output_size(self):
        return _calc_map_size(
            self.branch3x3_2.conv,
            self.branch7x7x3_4.conv,
            restore_branchpool_layer=self.sc)

    def calc_output_orders(self):
        return _calc_map_orders(
            self.branch3x3_2.conv,
            self.branch7x7x3_4.conv,
            restore_branchpool_layer=self.sc)


class _InceptionE(nn.Module):
    def __init__(self, in_channels, track_running_stats):
        super(_InceptionE, self).__init__()
        self.branch1x1_1 = _BasicConv2d(
            in_channels, 320, track_running_stats, kernel_size=1)

        self.branch3x3_1 = _BasicConv2d(
            in_channels, 384, track_running_stats, kernel_size=1)
        self.branch3x3_2a = _BasicConv2d(
            384, 384, track_running_stats, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = _BasicConv2d(
            384, 384, track_running_stats, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = _BasicConv2d(
            in_channels, 448, track_running_stats, kernel_size=1)
        self.branch3x3dbl_2 = _BasicConv2d(
            448, 384, track_running_stats, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = _BasicConv2d(
            384, 384, track_running_stats, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = _BasicConv2d(
            384, 384, track_running_stats, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = _BasicConv2d(
            in_channels, 192, track_running_stats, kernel_size=1)

        self._next_layers = {
            'branch1x1_1': None,
            'branch3x3_1': ['branch3x3_2a', 'branch3x3_2b'],
            'branch3x3_2a': None,
            'branch3x3_2b': None,
            'branch3x3dbl_1': 'branch3x3dbl_2',
            'branch3x3dbl_2': ['branch3x3dbl_3a', 'branch3x3dbl_3b'],
            'branch3x3dbl_3a': None,
            'branch3x3dbl_3b': None,
            'branch_pool': None,
        }

    def forward(self, x):
        branch1x1 = self.branch1x1_1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

    def calc_output_size(self):
        return _calc_map_size(
            self.branch1x1_1.conv,
            self.branch3x3_2a.conv,
            self.branch3x3_2b.conv,
            self.branch3x3dbl_3a.conv,
            self.branch3x3dbl_3b.conv,
            self.branch_pool.conv,
            restore_branchpool_layer=None)

    def calc_output_orders(self):
        return _calc_map_orders(
            self.branch1x1_1.conv,
            self.branch3x3_2a.conv,
            self.branch3x3_2b.conv,
            self.branch3x3dbl_3a.conv,
            self.branch3x3dbl_3b.conv,
            self.branch_pool.conv,
            restore_branchpool_layer=None)


class _InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, track_running_stats):
        super().__init__()
        self.conv_1 = _BasicConv2d(
            in_channels, 128, track_running_stats, kernel_size=1)
        self.conv_2 = _BasicConv2d(
            128, 768, track_running_stats, kernel_size=5)

        self.conv_2.stddev = 0.01
        self.fc = _MatchableLinear(768, num_classes)
        self.fc.stddev = 0.001

        self._next_layers = {
            'conv_1': 'conv_2',
            'conv_2': 'fc',
            'fc': None,
        }

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv_1(x)
        # N x 128 x 5 x 5
        x = self.conv_2(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x

    def calc_output_size(self):
        return self.fc.map_size

    def calc_output_order(self):
        return self.fc.map_orders


@options(
    "Unordered InceptionNet Model Configs",
    ArgParseOption(
        'uinc.al', 'unordered-inceptionnet.aux-logits',
        action='store_true',
        help="Use AUX logits."),
    ArgParseOption(
        'uinc.ti', 'unordered-inceptionnet.transform-input',
        action='store_true',
        help="Transform the inputs."),

    ArgParseOption(
        'uinc.tr', 'unordered-inceptionnet.track-runstat',
        action='store_true',
        help="Track running stats of batchnorm layer."))
class UnorderedInception3(_UnorderedNet):
    # CIFAR10: aux_logits True->False
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        """
        Important: In contrast to the other models the inception_v3 expects
        tensors with a size of N x 3 x 299 x 299. images should be resized
        accordingly.
        """
        self.is_client = hasattr(cfg, 'client_id')
        super().__init__(
            dataset.num_classes(),
            cfg.model.unordered_inceptionnet.aux_logits,
            cfg.model.unordered_inceptionnet.transform_input,
            cfg.model.unordered_inceptionnet.track_runstat)

    def init_submodules(
            self, num_classes, aux_logits, transform_input, track_running_stats):
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        # CIFAR10: stride 2->1, padding 0 -> 1
        self.Conv2d_1a_3x3 = _BasicConv2d(
            3, 192, track_running_stats, kernel_size=3, stride=1, padding=1)
        self.Mixed_5b = _InceptionA(192, 32, track_running_stats)
        self.Mixed_5c = _InceptionA(256, 64, track_running_stats)
        self.Mixed_5d = _InceptionA(288, 64, track_running_stats)
        self.Mixed_6a = _InceptionB(288, self, track_running_stats)
        self.Mixed_6b = _InceptionC(768, 128, track_running_stats)
        self.Mixed_6c = _InceptionC(768, 160, track_running_stats)
        self.Mixed_6d = _InceptionC(768, 160, track_running_stats)
        self.Mixed_6e = _InceptionC(768, 192, track_running_stats)
        if aux_logits:
            self.AuxLogits = _InceptionAux(
                768, num_classes, track_running_stats)
        self.Mixed_7a = _InceptionD(768, self, track_running_stats)
        self.Mixed_7b = _InceptionE(1280, track_running_stats)
        self.Mixed_7c = _InceptionE(2048, track_running_stats)
        self.fc = _MatchableLinear(2048, num_classes)

    def init_matchable_layers(self):
        for name, module in self.named_modules():
            if isinstance(module, _MatchableConv2d) and "conv" in name:
                self.matchable_layers.append([(name, module)])
                bn_name = re.sub('conv', 'bn', name)
                if bn_name in dict(self.named_modules()):
                    self.matchable_layers[-1].append((
                        bn_name, dict(self.named_modules())[bn_name]))
            elif isinstance(module, _MatchableLinear):
                self.matchable_layers.append([(name, module)])

    def skip_training(self, num_frozen_groups):
        layer_name = self.matchable_layers[num_frozen_groups][0][0]
        if ("Conv2d_1a_3x3.conv" == layer_name) or ("fc" == layer_name) or \
                layer_name in {'Mixed_5b.branch1x1_1.conv',
                               'Mixed_5c.branch1x1_1.conv',
                               'Mixed_5d.branch1x1_1.conv',
                               'Mixed_6a.branch3x3_1.conv',
                               'Mixed_6b.branch1x1_1.conv',
                               'Mixed_6c.branch1x1_1.conv',
                               'Mixed_6d.branch1x1_1.conv',
                               'Mixed_6e.branch1x1_1.conv',
                               'AuxLogits.conv_1.conv',
                               'self.Mixed_7a.branch3x3_1.conv',
                               'self.Mixed_7b.branch1x1_1.conv',
                               'self.Mixed_7c.branch1x1_1.conv'}:
            return False
        return True

    def calc_input_size(self, layer_name: str) -> int:
        if layer_name == 'Conv2d_1a_3x3.conv':
            # first layer
            return self.Conv2d_1a_3x3.conv.in_channels
        elif layer_name == 'fc':
            # last layer
            return self.Mixed_7c.calc_output_size()
        elif layer_name.endswith('.sc'):
            child_names = [n for n, _ in self.named_children()]
            child_modules = [l for _, l in self.named_children()]
            parent_name = '.'.join(layer_name.split('.')[:-1])
            parent_idx = child_names.index(parent_name)
            parent_module = child_modules[parent_idx]
            return child_modules[parent_idx - 1].calc_output_size()
        else:
            # modules' first layers' size is dependent on the previous layer
            child_names = [n for n, _ in self.named_children()]
            child_modules = [l for _, l in self.named_children()]
            parent_name = '.'.join(layer_name.split('.')[:-2])
            parent_idx = child_names.index(parent_name)
            parent_module = child_modules[parent_idx]

            if layer_name.endswith('_1.conv') or \
                    layer_name.endswith('branch_pool.conv'):
                if parent_idx == 1:
                    return child_modules[0].conv.map_size
                else:
                    return child_modules[parent_idx - 1].calc_output_size()
            else:
                layer_sname = layer_name.split('.')[-2]
                for name, value in parent_module._next_layers.items():
                    if (isinstance(value, str) and value == layer_sname) or (
                            isinstance(value, list) and layer_sname in value):
                        return dict(parent_module.named_children())[
                            name].conv.map_size
                raise RuntimeError("Cannot find previous layer")
        raise RuntimeError("Unknown layer")

    def calc_input_orders(self, layer_name: str) -> List[List[int]]:
        # if is the first in the parent module, we need search cross-submodule

        if layer_name == 'Conv2d_1a_3x3.conv':
            # first layer
            return [list(range(self.Conv2d_1a_3x3.conv.in_channels))]
        elif layer_name == 'fc':
            # last layer
            return self.Mixed_7c.calc_output_orders()
        elif layer_name.endswith('.sc'):
            child_names = [n for n, _ in self.named_children()]
            child_modules = [l for _, l in self.named_children()]
            parent_name = '.'.join(layer_name.split('.')[:-1])
            parent_idx = child_names.index(parent_name)
            parent_module = child_modules[parent_idx]
            return child_modules[parent_idx - 1].calc_output_orders()
        else:
            # modules' first layers' order is dependent on the previous layer
            child_names = [n for n, _ in self.named_children()]
            child_modules = [l for _, l in self.named_children()]
            parent_name = '.'.join(layer_name.split('.')[:-2])
            parent_idx = child_names.index(parent_name)
            parent_module = child_modules[parent_idx]

            if layer_name.endswith('_1.conv') or \
                    layer_name.endswith('branch_pool.conv'):
                if parent_idx == 1:
                    return child_modules[0].conv.map_orders
                else:
                    return child_modules[parent_idx - 1].calc_output_orders()
            else:
                layer_sname = layer_name.split('.')[-2]
                for name, value in parent_module._next_layers.items():
                    if (isinstance(value, str) and value == layer_sname) or (
                            isinstance(value, list) and layer_sname in value):
                        return dict(parent_module.named_children())[
                            name].conv.map_orders
                raise RuntimeError("Cannot find previous layer")
        raise RuntimeError("Unknown layer")

    def calc_next_layers(self, layer_name: str) -> List[Tuple[str, nn.Module]]:
        # If last layer
        if layer_name == 'fc':
            return []

        layer_sname = layer_name.split('.')[-2]

        # if first layer
        if layer_sname == 'Conv2d_1a_3x3':
            return [(f"Mixed_5b.{name}", module) for name, module in
                    filter(lambda n: "_1.conv" in n[0]
                           or "branch_pool.conv" in n[0],
                           self.Mixed_5b.named_modules())]
        else:
            # child name of `self`
            child_names = [n for n, _ in self.named_children()]
            # parent name of `layer_name`
            print(layer_name)
            parent_name = '.'.join(layer_name.split('.')[: -2])
            print(parent_name)
            parent_idx = child_names.index(parent_name)
            parent_module = dict(self.named_children())[parent_name]

        # find next modules
        if isinstance(parent_module._next_layers[layer_sname], str):
            # within the same parent module
            name = re.sub(layer_sname, parent_module._next_layers[layer_sname],
                          layer_name)
            return [(name, dict(self.named_modules())[name])]
        if isinstance(parent_module._next_layers[layer_sname], list):
            # within the same parent module, but is a list
            new_snames = parent_module._next_layers[layer_sname]
            names = [re.sub(layer_sname, n, layer_name) for n in new_snames]
            return [(n, dict(self.named_modules())[n]) for n in names]
        if parent_module._next_layers[layer_sname] is None:
            # if last layer
            next_parent_name = child_names[parent_idx + 1]
            if next_parent_name == "fc":
                return [("fc", dict(self.named_modules())["fc"])]
            else:
                next_parent = dict(self.named_children())[next_parent_name]
                return [(f"{next_parent_name}.{name}", module)
                        for name, module in filter(
                            lambda n: "_1.conv" in n[0]
                            or "branch_pool.conv" in n[0],
                            next_parent.named_modules())]

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + \
                (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + \
                (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + \
                (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)

        # CIFAR10
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)                        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)                        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)                        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)                        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)                        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)                        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)                        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)                        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)                 # N x 768 x 17 x 17
        x = self.Mixed_7a(x)                        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)                        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)                        # N x 2048 x 8 x 8

        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)    # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)                   # N x 2048
        x = self.fc(x)                              # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return _InceptionOuputs(x, aux)
        return x

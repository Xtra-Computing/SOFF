"""VGG network for the FedGMA algorithm"""

import re
import logging
from munch import Munch
import torch
from torch import nn
from ..datasets.fl_split.base import _FLSplit
from ..utils.arg_parser import ArgParseOption, options, require
from .unordered_net import (
    _UnorderedNet, _MatchableConv2d, _MatchableBatchNorm2d, _MatchableLinear)

log = logging.getLogger(__name__)


class _UnorderedClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(True)

        setattr(self, "linear1", _MatchableLinear(4096, 512))
        setattr(self, "linear2", _MatchableLinear(512, 512))
        setattr(self, "linear3", _MatchableLinear(512, num_classes))

    def forward(self, batch):
        batch = self.linear1(batch)
        batch = self.relu(batch)
        batch = self.dropout(batch)

        batch = self.linear2(batch)
        batch = self.relu(batch)
        batch = self.dropout(batch)

        batch = self.linear3(batch)
        return batch


class _UnorderedFeatures(nn.Module):
    def __init__(self, layers_cfg, batch_norm):
        super().__init__()

        self.layer_names = []
        in_channels = 3
        num_maxpool = 0
        num_dropout = 0
        num_conv = 0

        def _add_layer(layer_name, layer):
            setattr(self, layer_name, layer)
            self.layer_names.append(layer_name)

        for val in layers_cfg:
            if val == 'M':
                _add_layer(
                    f"maxpool{num_maxpool}",
                    nn.MaxPool2d(kernel_size=2, stride=2))
                num_maxpool += 1
            elif isinstance(val, str) and val.startswith('D'):
                _add_layer(
                    f"dropout{num_maxpool}",
                    nn.Dropout(int(val.removeprefix('D')) / 100.))
                num_dropout += 1
            else:
                _add_layer(
                    f"conv{num_conv}",
                    _MatchableConv2d(in_channels, val, kernel_size=3, padding=1))
                if batch_norm:
                    _add_layer(f"bn{num_conv}", _MatchableBatchNorm2d(val))
                _add_layer(f"relu{num_conv}", nn.ReLU(inplace=True))
                in_channels = val
                num_conv += 1

    def forward(self, batch):
        for layer_name in self.layer_names:
            batch = getattr(self, layer_name)(batch)
        return batch


@options(
    "Unordered VGG Model Configs",
    ArgParseOption(
        'uvgg.bn', 'unordered-vgg.batch-norm',
        action='store_true',
        help="Use batchnorm layer."),
    ArgParseOption(
        'uvgg.tr', 'unordered-vgg.track-runstat',
        action='store_true',
        help="Track running stats of batchnorm layer."))
@require('model.unordered_vgg.batch_norm', 'model.unordered_vgg.track_runstat')
class _UnorderedVGG(_UnorderedNet):
    def __init__(self, layers_cfg, cfg: Munch, dataset: _FLSplit):
        super().__init__(
            layers_cfg, cfg.model.unordered_vgg.batch_norm,
            dataset.num_classes())

    def init_submodules(self, layers_cfg, batch_norm, num_classes):
        self.features = _UnorderedFeatures(layers_cfg, batch_norm)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = _UnorderedClassifier(num_classes)

    def init_matchable_layers(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.matchable_layers.append([(name, module)])
                modules = dict(self.named_modules())
                if 'conv' in name and re.sub('conv', 'bn', name) in modules:
                    self.matchable_layers[-1] += [(
                        re.sub('conv', 'bn', name),
                        modules[re.sub('conv', 'bn', name)])]

    def forward(self, batch):
        batch = self.features(batch)
        # x = self.avgpool(x)
        batch = torch.flatten(batch, 1)
        batch = self.classifier(batch)
        return batch


class UnorderedVGG9(_UnorderedVGG):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__([
            32, 64, 'M', 128, 128, 'M', 'D5', 256, 256, 'M', 'D10'
        ], cfg, dataset)


class UnorderedVGG11(_UnorderedVGG):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__([
            64, 'M', 128, 'M', 256, 256, 'M',
            512, 512, 'M', 512, 512, 'M'
        ], cfg, dataset)


class UnorderedVGG13(_UnorderedVGG):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__([
            64, 64, 'M', 128, 128, 'M', 256, 256, 'M',
            512, 512, 'M', 512, 512, 'M'
        ], cfg, dataset)


class UnorderedVGG16(_UnorderedVGG):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__([
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
            512, 512, 512, 'M', 512, 512, 512, 'M'
        ], cfg, dataset)


class UnorderedVGG19(_UnorderedVGG):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__([
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
            512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'
        ], cfg, dataset)

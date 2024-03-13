"""VGG models (FedMA adaption)"""
import logging
from typing import Set
import torch
from torch import nn
from munch import Munch
from ..datasets.fl_split.base import _FLSplit
from ..utils.arg_parser import ArgParseOption, options

log = logging.getLogger(__name__)


class _FedmaClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(_FedmaClassifier, self).__init__()

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(True)

        # setattr(self, "linear1", nn.Linear(512, 512))
        setattr(self, "linear1", nn.Linear(4096, 512))
        setattr(self, "linear2", nn.Linear(512, 512))
        setattr(self, "linear3", nn.Linear(512, num_classes))

        # setattr(self, "linear1", nn.Linear(512 * 7 * 7, 4096))
        # setattr(self, "linear2", nn.Linear(4096, 4096))
        # setattr(self, "linear3", nn.Linear(4096, num_classes))

        self.matchable_layers = ["linear1", "linear2", "linear3"]

    def update_layer(self, layer_name: str, new_layer_weights: torch.Tensor,
                     new_layer_bias: torch.Tensor):
        assert hasattr(self, layer_name)
        assert len(new_layer_weights.shape) == 2
        assert len(new_layer_bias.shape) == 1

        input_size = new_layer_weights.shape[1]
        output_size = new_layer_weights.shape[0]

        # copy data
        new_layer = nn.Linear(input_size, output_size)
        new_layer.to(new_layer_weights.device)
        with torch.no_grad():
            new_layer.weight.copy_(new_layer_weights)
            new_layer.bias.copy_(new_layer_bias)

        delattr(self, layer_name)
        setattr(self, layer_name, new_layer)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear3(x)
        return x


class _FedmaFeatures(nn.Module):
    def __init__(self, features):
        super(_FedmaFeatures, self).__init__()

        # register features
        self.layer_names = [feature[0] for feature in features]
        self.matchable_layers = []
        for layer_name, layer in features:
            if layer_name.startswith("conv"):
                # register layer and order buffer
                self.matchable_layers.append(layer_name)
            setattr(self, layer_name, layer)

    def forward(self, x):
        for layer_name in self.layer_names:
            x = getattr(self, layer_name)(x)
        return x

    def update_layer(self, layer_name: str,
                     new_layer_weights: torch.Tensor,
                     new_layer_bias: torch.Tensor):
        assert hasattr(self, layer_name)
        assert len(new_layer_weights.shape) == 4
        assert len(new_layer_bias.shape) == 1

        input_size = new_layer_weights.shape[1]
        output_size = new_layer_weights.shape[0]

        new_layer = nn.Conv2d(
            input_size, output_size, kernel_size=3, padding=1)
        new_layer.to(new_layer_weights.device)

        with torch.no_grad():
            new_layer.weight.copy_(new_layer_weights)
            new_layer.bias.copy_(new_layer_bias)

        delattr(self, layer_name)
        setattr(self, layer_name, new_layer)


@options(
    "VGG Model Configs",
    ArgParseOption(
        'mavgg.bn', 'fedma-vgg.batch-norm', action='store_true',
        help="Use batchnorm layers."))
class _FedmaVGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super().__init__()
        self.features = _FedmaFeatures(features)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = _FedmaClassifier(num_classes)
        self.matchable_layers = \
            ["features." + x for x in self.features.matchable_layers] + \
            ["classifier." + x for x in self.classifier.matchable_layers]

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.parameter_names = set(k for k, _ in self.named_parameters())

    def update_layer(self, layer_name: str, new_layer_weights: torch.Tensor,
                     new_layer_bias: torch.Tensor, update_next_layer=True):

        assert layer_name in self.matchable_layers
        assert new_layer_bias.shape[0] == new_layer_weights.shape[0]

        idx = self.matchable_layers.index(layer_name)
        new_num_neurons = new_layer_bias.shape[0]

        # fix current layer
        if layer_name.startswith("classifier."):
            layer_name = layer_name.removeprefix("classifier.")
            old_num_neurons = getattr(
                self.classifier, layer_name).bias.shape[0]
            self.classifier.update_layer(
                layer_name, new_layer_weights, new_layer_bias)
        elif layer_name.startswith("features."):
            layer_name = layer_name.removeprefix("features.")
            old_num_neurons = getattr(self.features, layer_name).bias.shape[0]
            self.features.update_layer(
                layer_name, new_layer_weights, new_layer_bias)

        log.debug(
            "idx: %s, old: %s, new: %s",
            idx, old_num_neurons, new_num_neurons)
        # fix next layer, if not last layer
        if (idx < len(self.matchable_layers) - 1 and
                old_num_neurons != new_num_neurons):

            next_layer_name = self.matchable_layers[idx + 1]
            log.debug("Fixing layer {}".format(next_layer_name))

            # TODO: use named_parameters here
            if next_layer_name.startswith("classifier"):
                prefix = "classifier."
                module = self.classifier
            elif next_layer_name.startswith("features"):
                prefix = "features."
                module = self.features

            next_layer_name = next_layer_name.removeprefix(prefix)

            # pad (or shrink) weights
            next_layer_weights = getattr(module, next_layer_name).weight

            # the input of next layer could be mulitple times of previous
            # layer's output size, instead of just equal
            assert next_layer_weights.shape[1] % old_num_neurons == 0
            group_size = next_layer_weights.shape[1] // old_num_neurons

            new_next_layer_shape = list(next_layer_weights.shape)
            new_next_layer_shape[1] = new_num_neurons * group_size
            new_next_layer_weights = torch.zeros(
                new_next_layer_shape).to(next_layer_weights.device)

            num_neurons = min(new_num_neurons, old_num_neurons)
            new_next_layer_weights[:, :num_neurons] = \
                next_layer_weights[:, :num_neurons]

            # use unchanged bias
            next_layer_bias = getattr(module, next_layer_name).bias

            module.update_layer(
                next_layer_name, new_next_layer_weights, next_layer_bias)

    def freeze_layers(self, layer_names: Set[str]):
        freeze_params = set()
        for layer_name in layer_names:
            assert (layer_name + ".weight") in self.parameter_names
            freeze_params.add(layer_name + ".weight")
            freeze_params.add(layer_name + ".bias")

        for name, value in self.named_parameters():
            if name in freeze_params:
                value.requires_grad_(False)

    def unfreeze_layers(self, layer_names: Set[str]):
        unfreeze_params = set()
        for layer_name in layer_names:
            assert (layer_name + ".weight") in self.parameter_names
            unfreeze_params.add(layer_name + ".weight")
            unfreeze_params.add(layer_name + ".bias")

        for name, value in self.named_parameters():
            if name in unfreeze_params:
                value.requires_grad_(True)

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(layers_config, batch_norm=False):
    layers = []
    in_channels = 3

    num_maxpool = 0
    num_dropout = 0
    num_conv = 0
    for val in layers_config:
        if val == 'M':
            maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            layers += [(f"maxpool{num_maxpool}", maxpool)]
            num_maxpool += 1
        elif isinstance(val, str) and val.startswith('D'):
            dropout = nn.Dropout(int(val.removeprefix('D')) / 100.)
            layers += [(f"dropout{num_dropout}", dropout)]
            num_dropout += 1
        else:
            assert isinstance(val, int)
            conv2d = nn.Conv2d(in_channels, val, kernel_size=3, padding=1)
            if batch_norm:
                layers += [(f"conv{num_conv}", conv2d),
                           (f"bn{num_conv}", nn.BatchNorm2d(val)),
                           (f"relu{num_conv}", nn.ReLU(inplace=True))]
            else:
                layers += [(f"conv{num_conv}", conv2d),
                           (f"relu{num_conv}", nn.ReLU(inplace=True))]
            in_channels = val
            num_conv += 1

    return layers


layers_config = {
    '9': [32, 64, 'M',
          128, 128, 'M', 'D5',
          256, 256, 'M', 'D10'],
    'A': [64, 'M',
          128, 'M',
          256, 256, 'M',
          512, 512, 'M',
          512, 512, 'M'],
    'B': [64, 64, 'M',
          128, 128, 'M',
          256, 256, 'M',
          512, 512, 'M',
          512, 512, 'M'],
    'D': [64, 64, 'M',
          128, 128, 'M',
          256, 256, 256, 'M',
          512, 512, 512, 'M',
          512, 512, 512, 'M'],
    'E': [64, 64, 'M',
          128, 128, 'M',
          256, 256, 256, 256, 'M',
          512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


class FedmaVGG9(_FedmaVGG):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__(
            make_layers(layers_config['9'], cfg.model.fedma_vgg.batch_norm),
            dataset.num_classes())


class FedmaVGG11(_FedmaVGG):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__(
            make_layers(layers_config['A'], cfg.model.fedma_vgg.batch_norm),
            dataset.num_classes())


class FedmaVGG13(_FedmaVGG):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__(
            make_layers(layers_config['B'], cfg.model.fedma_vgg.batch_norm),
            dataset.num_classes())


class FedmaVGG16(_FedmaVGG):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__(
            make_layers(layers_config['D'], cfg.model.fedma_vgg.batch_norm),
            dataset.num_classes())


class FedmaVGG19(_FedmaVGG):
    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__(
            make_layers(layers_config['E'], cfg.model.fedma_vgg.batch_norm),
            dataset.num_classes())

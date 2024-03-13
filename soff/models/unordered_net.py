"""Base network for the FedGMA algorithm"""
import logging
from typing import List, Tuple
import torch
from torch import nn
from ..utils.training import is_batchnorm

log = logging.getLogger(__name__)


class _MatchableLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.out_size = self.get_out_size()

        # Initializations for client sside
        self.map_orders = [torch.arange(self.out_size, dtype=torch.int32)]
        self.map_weights = [1.]
        self.map_size = self.out_size
        self.orig_output_neurons = self.out_size

    def get_out_size(self):
        raise NotImplementedError


class _MatchableConv1d(_MatchableLayer, nn.Conv1d):
    def get_out_size(self):
        return self.out_channels


class _MatchableConv2d(_MatchableLayer, nn.Conv2d):
    def get_out_size(self):
        return self.out_channels


class _MatchableConv3d(_MatchableLayer, nn.Conv3d):
    def get_out_size(self):
        return self.out_channels


class _MatchableBatchNorm1d(_MatchableLayer, nn.BatchNorm1d):
    def get_out_size(self):
        return self.num_features


class _MatchableBatchNorm2d(_MatchableLayer, nn.BatchNorm2d):
    def get_out_size(self):
        return self.num_features


class _MatchableBatchNorm3d(_MatchableLayer, nn.BatchNorm3d):
    def get_out_size(self):
        return self.num_features


class _MatchableLinear(_MatchableLayer, nn.Linear):
    def get_out_size(self):
        return self.out_features


class _DummyLayer(_MatchableLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    @property
    def weight(self):
        return torch.Tensor()

    @property
    def bias(self):
        return None

    @weight.setter
    def weight(self, value):
        pass

    @bias.setter
    def bias(self, value):
        pass

    def get_out_size(self):
        return 0


class _UnorderedNet(nn.Module):
    """
    Restrictions: When subclassing this class, one cannot use nn.Sequential
    as members. i.e. The name of every submodule must have a proper name.
    """

    def __init__(self, *argc, **argv):
        super().__init__()

        self.init_submodules(*argc, **argv)
        self.matchable_layers = []
        self.init_matchable_layers()

        # Convenience member for fast accessing matchable_layers
        self.matchable_layer_idx = {}
        for group_idx, group in enumerate(self.matchable_layers):
            for layer_idx, layer in enumerate(group):
                self.matchable_layer_idx[layer[0]] = (group_idx, layer_idx)

        # every layer in matchable_layers should be a MatchableLayer
        for group in self.matchable_layers:
            for name, module in group:
                assert isinstance(module, _MatchableLayer)

    def init_submodules(self, *argc, **argv):
        """ Put what normally would be in __init__() here.  """
        raise NotImplementedError

    def init_matchable_layers(self):
        """
        # Register matchable_layers. matchable_layers has the form of:
        # [[(name, module), ...], ...]
        # Each list of [(name, module), ...] are matched simultaneously
        """
        raise NotImplementedError

    def skip_training(self, num_frozen_groups):
        """Override to implement a custom training scheme"""
        return False

    def update_layer(self, layer_name: str, new_layer_weights: torch.Tensor,
                     new_layer_bias: torch.Tensor, update_next_layer=True):
        assert isinstance(new_layer_weights, torch.Tensor)
        assert isinstance(new_layer_bias, torch.Tensor)

        # for simple shortcuts, no need to update weights.
        # Orders are updated in `update_mapping`
        if new_layer_weights.numel() == 0 and new_layer_bias.numel() == 0:
            return

        assert (layer_name in self.matchable_layer_idx.keys())
        assert (new_layer_bias.numel() == 0 or
                new_layer_bias.shape[0] == new_layer_weights.shape[0])

        # obtain group_idx and layer_idx
        group_idx, layer_idx = self.matchable_layer_idx[layer_name]
        new_num_neurons = new_layer_weights.shape[0]

        current_module = dict(self.named_modules())[
            '.'.join(layer_name.split('.')[:-1])]
        current_layer_name = layer_name.split('.')[-1]

        current_layer = dict(self.named_modules())[layer_name]
        old_num_neurons = current_layer.weight.shape[0]

        assert len(new_layer_weights.shape) == len(current_layer.weight.shape)
        assert new_layer_bias.numel() == 0 or len(new_layer_bias.shape) == 1

        # replace the layer and its info in `matchable_layers`
        new_layer = self._update_a_layer(
            current_module, current_layer_name, current_layer,
            new_layer_weights, new_layer_bias)
        self.matchable_layers[group_idx][layer_idx] = (layer_name, new_layer)

        log.debug("layer: {}, old: {}, new: {}".format(
            layer_name, old_num_neurons, new_num_neurons))

        if update_next_layer:
            pass

    def update_batchnorm_layer(
            self, name, weight, bias, mean, variance, num_batches):
        self.update_layer(name, weight, bias, False)

        # update mean and variance
        current_layer = dict(self.named_modules())[name]
        if current_layer.track_running_stats:
            current_layer.running_mean.set_(
                mean.to(current_layer.running_mean.device))
            current_layer.running_var.set_(
                variance.to(current_layer.running_var.device))

            # update num_batches on server
            if not self.is_client:
                assert isinstance(num_batches, int)
                current_layer.num_batches_tracked = \
                    torch.Tensor([num_batches]).to(
                        current_layer.num_batches_tracked.device)

    def update_mapping(
            self, layer_name: str, new_mapping_size: int,
            new_mappings: List[List[int]], new_mapping_weights: List[int]):
        """
        new_mapping_size: number of neurons after mapping
        new_mappings: (shape: num_clients * num_neurons_before_mapping)
        new_mapping_weights: (shape: num_clients), the weight of each client
        """
        if layer_name == self.matchable_layers[-1][0]:
            return

        # FIXME: here: for shortcut, also need to update the input order here.
        # We cannot rely on "next layer fixing", as that's optional
        cur_layer = dict(self.named_modules())[layer_name]
        assert isinstance(cur_layer, _MatchableLayer)

        cur_layer.map_size = new_mapping_size
        cur_layer.map_orders = \
            [torch.LongTensor(order) for order in new_mappings]
        cur_layer.map_weights = new_mapping_weights

    def calc_input_size(self, layer_name: str) -> int:
        """both server and client can use this"""
        group_idx, layer_idx = self.matchable_layer_idx[layer_name]

        # the first layer is a conv layer
        if group_idx == 0:
            return self.matchable_layers[group_idx][layer_idx][1].in_channels
        else:
            return self.matchable_layers[group_idx-1][0][1].map_size

    def calc_input_orders(self, layer_name: str) -> List[List[int]]:
        """for server only"""
        group_idx, layer_idx = self.matchable_layer_idx[layer_name]
        if group_idx == 0:
            return [list(range(
                self.matchable_layers[group_idx][layer_idx][1].in_channels))]
        else:
            return self.matchable_layers[group_idx-1][0][1].map_orders

    def calc_input_order(self, layer_name: str) -> List[int]:
        """for client only"""
        return self.calc_input_orders(layer_name)[0]

    def calc_next_layers(self, layer_name: str) -> List[Tuple[str, nn.Module]]:
        group_idx, layer_idx = self.matchable_layer_idx[layer_name]
        if group_idx == len(self.matchable_layers) - 1:
            return []
        else:
            return list(filter(lambda l: not is_batchnorm(l[1]),
                               self.matchable_layers[group_idx + 1]))

    @staticmethod
    def _update_a_layer(module, layer_name, old_layer, new_weights, new_bias):
        """
        Update the layer given by `layer_name` in `module`
        Return the updated layer
        """
        if isinstance(old_layer, _MatchableConv2d):
            new_layer = _MatchableConv2d(
                new_weights.shape[1], new_weights.shape[0],
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=(old_layer.bias is not None)).to(
                old_layer.weight.device)
            # FIXME: this is for u_resnet only
            if hasattr(old_layer, "batchnorm"):
                new_layer.batchnorm = old_layer.batchnorm
            new_layer.orig_output_neurons = old_layer.orig_output_neurons
        elif isinstance(old_layer, _MatchableLinear):
            new_layer = _MatchableLinear(
                new_weights.shape[1], new_weights.shape[0],
                bias=(old_layer.bias is not None)).to(
                old_layer.weight.device)
        elif is_batchnorm(old_layer):
            new_layer = old_layer.__class__(
                new_weights.shape[0],
                track_running_stats=old_layer.track_running_stats).to(
                old_layer.weight.device)
        elif isinstance(old_layer, _DummyLayer):
            new_layer = _DummyLayer()
        else:
            raise Exception("unrecognized layer type")

        new_layer.map_orders = old_layer.map_orders
        new_layer.map_weights = old_layer.map_weights
        new_layer.map_size = old_layer.map_size

        with torch.no_grad():
            new_layer.weight.copy_(new_weights)
            if new_bias.numel() > 0:
                new_layer.bias.copy_(new_bias)
            delattr(module, layer_name)
            setattr(module, layer_name, new_layer)

        return new_layer

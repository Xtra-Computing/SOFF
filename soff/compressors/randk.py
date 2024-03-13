"""Random compressor"""
from abc import abstractmethod
import math
import random
import torch
import numpy as np
from munch import Munch
from torch.nn.utils.convert_parameters import (
    parameters_to_vector, vector_to_parameters)
from .base import (
    _TensorSubsample, _TensorRandom,
    _ModelCompressor, _PerModelCompressor, _PerLayerCompressor)
from .compress_utils import (
    Offset, pack_tensor, unpack_tensor,
    pack_tensor_shape, unpack_tensor_shape)
from ..utils.arg_parser import ArgParseOption, options


class TensorRandK(_TensorSubsample, _TensorRandom):
    def __init__(self, ratio: float, seed=random.Random()):
        super().__init__(ratio, seed)

    def compress(self, tensor: torch.Tensor) -> bytearray:
        data = bytearray()

        # generate random mask
        k = math.ceil(self.ratio * tensor.numel())
        torch.manual_seed(self.seed)
        mask = torch.randperm(tensor.numel()).to(tensor.device) < k
        idx = mask.int().sort(descending=True).indices

        # pack original tensor shape
        data.extend(pack_tensor_shape(tensor))
        # mask and pack tensor
        data.extend(pack_tensor(tensor[idx[:k]].detach().cpu()))

        return data

    def decompress(self, data, offset=None):
        offset = offset or Offset()

        # unpack original tensor shape
        shape = unpack_tensor_shape(data, offset)
        # unpack compressed tensor
        tensor_data = unpack_tensor(data, offset)

        # restore random mask
        k = math.ceil(self.ratio * np.prod(shape))
        torch.manual_seed(self.seed)
        mask = torch.randperm(int(np.prod(shape))) < k
        idx = mask.int().sort(descending=True).indices

        # restore tensor
        tensor = torch.zeros(int(np.prod(shape)))
        tensor.index_copy_(0, idx[:k], tensor_data)
        tensor = tensor.reshape(shape)

        return tensor

    def zero_with_mask(self, tensor):
        torch.manual_seed(self.seed)

        # restore random mask
        k = math.ceil(self.ratio * tensor.numel())
        torch.manual_seed(self.seed)
        mask = torch.randperm(tensor.numel()).to(tensor.device) < k
        idx = mask.int().sort(descending=True).indices

        tensor.view(-1).index_fill_(0, idx[:k], 0)


class RandK(_ModelCompressor):
    @abstractmethod
    def set_seed(self, new_seed):
        """Set the seed of the current compressor"""
        raise NotImplementedError


@options(
    "RandK-Per-Model Compressor Configs",
    ArgParseOption(
        'rnkm.r', 'randk-per-model.ratio',
        type=float, default=0.01, metavar='RATIO',
        help='Compression ratio'),
    ArgParseOption(
        'rnkm.s', 'randk-per-model.seed', type=int,
        help="Random seed"))
class RandKPerModel(RandK, _PerModelCompressor):
    def __init__(self, cfg: Munch):
        super().__init__(compressor=TensorRandK(
            cfg.compression.randk_per_model.ratio,
            cfg.compression.randk_per_model.seed))

    def zero_with_mask(self, net: torch.nn.Module):
        """Zero net with this compressor's rand-k mask"""
        assert isinstance(self.compressor, TensorRandK)
        net_as_vec = parameters_to_vector(net.parameters())
        self.compressor.zero_with_mask(net_as_vec)
        vector_to_parameters(net_as_vec, net.parameters())

    def set_seed(self, seed):
        rand_gen = random.Random(seed)
        rand_gen.random()
        self.compressor.seed = rand_gen.random()


@options(
    "RandK-Per-Layer Compressor Configs",
    ArgParseOption(
        'rndkl.r', 'randk-per-layer.ratios',
        type=float, nargs='+', default=[0.01], metavar='RATIO',
        help='Compression ratio(s), one per layer'),
    ArgParseOption(
        'rndkl.s', 'randk-per-layer.seed', type=int,
        help="Random seed"))
class RandKPerLayer(RandK, _PerLayerCompressor):
    def __init__(self, cfg: Munch):
        super().__init__(compressors=[
            TensorRandK(r, cfg.compression.randk_per_layer.seed)
            for r in cfg.compression.randk_per_layer.ratios])

    def zero_with_mask(self, net: torch.nn.Module):
        for i, param in enumerate(net.parameters()):
            compressor = self.compressors[i]
            assert isinstance(compressor, TensorRandK)
            compressor.zero_with_mask(param)

    def set_seed(self, seed):
        rand_gen = random.Random(seed)
        rand_gen.random()
        for compressor in self.compressors:
            compressor.seed = rand_gen.random()

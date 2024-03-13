"""Topk-related compressors"""
import math
from types import GeneratorType
from typing import List, Tuple, Sequence
import torch
import numpy as np
from munch import Munch
from torch import Tensor
from torch.nn.utils.convert_parameters import (
    parameters_to_vector, vector_to_parameters)
from .base import _TensorSubsample, _PerModelCompressor, _PerLayerCompressor
from .compress_utils import (
    Offset, pack_tensor, unpack_tensor, pack_tensor_shape, unpack_tensor_shape)
from ..utils.arg_parser import ArgParseOption, options, require


class TensorTopK(_TensorSubsample):
    """TopK compression for a tensor"""

    def compressed_data_and_idx(self, tensor) -> Tuple[Tensor, Tensor]:
        """Get compressed (data, idx) tensor pair"""
        return self.topk(
            tensor.view(-1), math.ceil(self.ratio * tensor.numel()))

    def compress(self, tensor):
        assert 0 <= self.ratio <= 1
        data = bytearray()

        # pack tensor shape
        data.extend(pack_tensor_shape(tensor))
        # get top-k, at least 1 element needs to be packed
        sparse_tensor, indices = self.compressed_data_and_idx(tensor)
        # pack sparsified tensor
        data.extend(pack_tensor(indices.detach().cpu()))
        data.extend(pack_tensor(sparse_tensor.cpu()))
        return data

    def decompress_data_and_idx(
            self, shape: Sequence[int], data: Tensor, idx: Tensor):
        tensor = torch.zeros(np.prod(shape)).to(data.device)
        tensor.index_copy_(0, idx, data)
        return tensor.reshape(list(shape))

    def decompress(self, data, offset=None):
        offset = offset or Offset()

        # unpack tensor shape
        shape = unpack_tensor_shape(data, offset)
        # unpack sparse indices and value
        tensor_indices = unpack_tensor(data, offset)
        # TODO: add options to DEFLATE here
        tensor_data = unpack_tensor(data, offset)

        tensor = torch.zeros(int(np.prod(shape)))
        tensor.index_copy_(0, tensor_indices, tensor_data)
        tensor = tensor.reshape(list(shape))

        return tensor

    def zero_with_mask(self, tensor):
        """zero out the top-k elements in the tensor"""
        assert 0 <= self.ratio <= 1
        _, mask = self.topk(
            tensor.view(-1), math.ceil(self.ratio * tensor.numel()))
        tensor.view(-1).index_fill_(0, mask, 0)

    # See: https://github.com/pytorch/pytorch/issues/22812
    @staticmethod
    def topk(tensor: Tensor, k: int):
        """Top-K subsampler"""
        # assert tensor.is_cuda

        # sort by absolute value
        idx = tensor.abs().sort(descending=True).indices
        return tensor[idx[:k]].clone().detach(), idx[:k]


@options(
    "TopK-Per-Model Compressor Configs",
    ArgParseOption(
        'tpkm.r', 'topk-per-model.ratio',
        type=float, default=0.01, metavar='RATIO',
        help="Compression ratio"))
@require('compression.topk_per_model.ratio')
class TopKPerModel(_PerModelCompressor):
    """Compress the entire model with topk compressor"""

    # def __init__(self, ratio: float = 0.01):
    def __init__(self, cfg: Munch):
        assert 0. < cfg.compression.topk_per_model.ratio <= 1.
        super().__init__(
            compressor=TensorTopK(cfg.compression.topk_per_model.ratio))

    def zero_with_mask(self, net: torch.nn.Module):
        """Zero net with this compressor's top-k mask"""
        assert isinstance(self.compressor, TensorTopK)
        net_as_vec = parameters_to_vector(net.parameters())
        self.compressor.zero_with_mask(net_as_vec)
        vector_to_parameters(net_as_vec, net.parameters())


@options(
    "TopK-Per-Layer Compressor Configs",
    ArgParseOption(
        'tpkl.r', 'topk-per-layer.ratios',
        type=float, nargs='+', default=[0.01], metavar='RATIO',
        help="Compression ratio(s), one per layer"))
@require('compression.topk_per_layer.ratios')
class TopKPerLayer(_PerLayerCompressor):
    """Compress the layer model with topk compressor"""

    def __init__(self, cfg: Munch):
        assert all(
            0. < ratio <= 1. for ratio in
            cfg.compression.topk_per_layer.ratios)
        super().__init__(compressors=[
            TensorTopK(ratio) for ratio in
            cfg.compression.topk_per_layer.ratios])

    def compressed_data_and_idx(self, params: List[Tensor]) \
            -> List[Tuple[Tensor, Tensor]]:
        """Get a list of compressed (data, idx) tensor pairs"""
        assert not isinstance(params, GeneratorType)
        assert len(self.compressors) == len(params)
        result = []
        for compressor, param in zip(self.compressors, params):
            assert isinstance(compressor, TensorTopK)
            result.append(compressor.compressed_data_and_idx(param))
        return result

    def decompress_data_and_idx(
            self, shapes: List[List[int]],
            datas: List[Tuple[Tensor, Tensor]]) -> List[Tensor]:
        """datas: a list of (data, idx) pairs"""
        assert len(self.compressors) == len(shapes) == len(datas)
        result = []
        for compressor, shape, data in zip(self.compressors, shapes, datas):
            assert isinstance(compressor, TensorTopK)
            result.append(compressor.decompress_data_and_idx(shape, *data))
        return result

    def zero_with_mask(self, net: torch.nn.Module):
        """Zero net with this compressor's top-k mask for each layer"""
        for i, param in enumerate(net.parameters()):
            compressor = self.compressors[i]
            assert isinstance(compressor, TensorTopK)
            compressor.zero_with_mask(param)

import math
import torch
import numpy as np
from typing import Union, List
from oarf.compressors.base import (
    _TensorSubsample, _PerModelCompressor,
    _PerLayerCompressor, _SubsampleCompressor)
from oarf.compressors.compress_utils import (
    Offset, pack_tensor, unpack_tensor,
    pack_tensor_shape, unpack_tensor_shape)


class TensorTopK(_TensorSubsample):
    def __init__(self, ratio: float):
        super().__init__(ratio)

    def compress(self, tensor: torch.Tensor) -> bytearray:
        assert (0 <= self.ratio <= 1)
        data = bytearray()

        # pack tensor shape
        data.extend(pack_tensor_shape(tensor))
        # get top-k, at least 1 element needs to be packed
        sparse_tensor, indices = TensorTopK.topk(
            tensor.view(-1), math.ceil(self.ratio * tensor.numel()))
        # pack sparsified tensor
        data.extend(pack_tensor(indices.detach().cpu()))
        data.extend(pack_tensor(sparse_tensor.cpu()))
        return data

    def decompress(self, data: bytearray,
                   offset: Offset = None) -> torch.Tensor:
        offset = offset or Offset()

        # unpack tensor shape
        shape = unpack_tensor_shape(data, offset)
        # unpack sparse indices and value
        tensor_indices = unpack_tensor(data, offset)
        tensor_data = unpack_tensor(data, offset)

        tensor = torch.zeros(int(np.prod(shape)))
        tensor.index_copy_(0, tensor_indices, tensor_data)
        tensor = tensor.reshape(list(shape))

        return tensor

    def zero_with_mask(self, tensor: torch.Tensor) -> None:
        """zero out the top-k elements in the tensor"""
        assert (0 <= self.ratio <= 1)
        _, mask = self.topk(
            tensor.view(-1), math.ceil(self.ratio * tensor.numel()))
        tensor.view(-1).index_fill_(0, mask, 0)

    # See: https://github.com/pytorch/pytorch/issues/22812
    @staticmethod
    def topk(tensor: torch.Tensor, k: int):
        assert tensor.is_cuda
        # sort by absolute value
        idx = tensor.abs().sort(descending=True).indices
        return tensor[idx[:k]].clone().detach(), idx[:k]


class TopKPerModel(_PerModelCompressor, _SubsampleCompressor):
    def __init__(self, ratio: float = 0.01):
        super().__init__(compressors=TensorTopK(ratio))


class TopKPerLayer(_PerLayerCompressor, _SubsampleCompressor):
    def __init__(self, ratios: Union[float, List[float]] = 0.01):
        if isinstance(ratios, float):
            super().__init__(compressors=TensorTopK(ratios))
        elif isinstance(ratios, list):
            super().__init__(compressors=[TensorTopK(r) for r in ratios])

    def zero_with_mask(self, net: torch.nn.Module):
        for i, param in enumerate(net.parameters()):
            _, mask = self.compressors[i].zero_with_mask(param)

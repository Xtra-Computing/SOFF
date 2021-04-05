import math
import torch
import random
import numpy as np
from typing import Union, List
from oarf.compressors.base import (
    _TensorSubsample, _TensorRandom,
    _PerModelCompressor, _PerLayerCompressor,
    _SubsampleCompressor, _RandomCompressor)
from oarf.compressors.compress_utils import (
    Offset, pack_tensor, unpack_tensor,
    pack_tensor_shape, unpack_tensor_shape)


class TensorRandK(_TensorSubsample, _TensorRandom):
    def __init__(self, ratio: float, seed=random.Random()):
        super().__init__(ratio, seed)

    def compress(self, tensor: torch.Tensor) -> bytearray:
        data = bytearray()

        # generate random mask
        k = math.ceil(self.ratio * tensor.numel())
        torch.manual_seed(self.seed)
        mask = torch.randperm(tensor.numel()).cuda() < k
        idx = mask.int().sort(descending=True).indices

        # pack original tensor shape
        data.extend(pack_tensor_shape(tensor))
        # mask and pack tensor
        data.extend(pack_tensor(tensor[idx[:k]].detach().cpu()))

        return data

    def decompress(self, data: bytearray,
                   offset: Offset = None) -> torch.Tensor:
        offset = offset or Offset()

        # unpack original tensor shape
        shape = unpack_tensor_shape(data, offset)
        # unpack compressed tensor
        tensor_data = unpack_tensor(data, offset)

        # restore random mask
        k = math.ceil(self.ratio * np.prod(shape))
        torch.manual_seed(self.seed)
        mask = torch.randperm(int(np.prod(shape))).cuda() < k
        idx = mask.int().sort(descending=True).indices

        # restore tensor
        tensor = torch.zeros(int(np.prod(shape)))
        tensor.index_copy_(0, idx[:k].cpu(), tensor_data)
        tensor = tensor.reshape(shape)

        return tensor

    def zero_with_mask(self, tensor: torch.nn.Module) -> None:
        torch.manual_seed(self.seed)

        # restore random mask
        k = math.ceil(self.ratio * tensor.numel())
        torch.manual_seed(self.seed)
        mask = torch.randperm(tensor.numel()).cuda() < k
        idx = mask.int().sort(descending=True).indices

        tensor.view(-1).index_fill_(0, idx[:k], 0)


class RandKPerModel(_PerModelCompressor, _SubsampleCompressor,
                    _RandomCompressor):
    def __init__(self, ratio: float = 0.01, seed=random.seed()):
        super().__init__(compressors=TensorRandK(ratio, None))
        self.set_seed(seed)


class RandKPerLayer(_PerLayerCompressor, _SubsampleCompressor,
                    _RandomCompressor):
    def __init__(self, ratios: Union[float, List[float]] = 0.01,
                 seed=random.seed()):
        if isinstance(ratios, float):
            super().__init__(compressors=TensorRandK(ratios, None))
        elif isinstance(ratios, list):
            super().__init__(compressors=[
                TensorRandK(r, None) for r in ratios])
        self.set_seed(seed)

    def zero_with_mask(self, net: torch.nn.Module):
        for param in net.parameters():
            self.compressors.zero_with_mask(param)

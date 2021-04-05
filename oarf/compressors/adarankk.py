import torch
import random
from typing import Union, List
from oarf.compressors.base import (
    _TensorRandom, _TensorLowRank,
    _PerLayerCompressor, _RandomCompressor, _LowRankCompressor)
from oarf.compressors.compress_utils import (
    Offset, pack_tensor, unpack_tensor,
    pack_tensor_shape, unpack_tensor_shape, orthogonalize)


class TensorAdaRankK(_TensorRandom, _TensorLowRank):
    def compress(self, tensor: torch.Tensor) -> bytearray:
        raise NotImplementedError
        # TODO

    def decompress(self, data: bytearray,
                   offset: Offset = None) -> torch.Tensor:
        offset = offset or Offset()
        raise NotImplementedError
        # TODO


class AdaRankKPerLayer(
        _PerLayerCompressor,  _RandomCompressor, _LowRankCompressor):
    def __init__(self, max_ranks: Union[int, List[int]] = 7,
                 seed=random.random()):
        raise NotImplementedError
        # TODO

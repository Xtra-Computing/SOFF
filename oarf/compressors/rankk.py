import torch
import random
from typing import Union, List
from oarf.compressors.base import (
    _TensorRandom, _TensorLowRank,
    _PerLayerCompressor, _RandomCompressor, _LowRankCompressor)
from oarf.compressors.compress_utils import (
    Offset, pack_tensor, unpack_tensor,
    pack_tensor_shape, unpack_tensor_shape, orthogonalize)


class TensorRankK(_TensorRandom, _TensorLowRank):
    def __init__(self, rank: int, seed=random.Random()):
        super().__init__(seed=seed, rank=rank)

    def compress(self, tensor: torch.Tensor) -> bytearray:
        torch.manual_seed(self.seed)
        data = bytearray()
        # pack the shape of the tensor
        data.extend(pack_tensor_shape(tensor))
        # pack 0 and 1-dim tensor as-is
        if tensor.ndim <= 1:
            data.extend(pack_tensor(tensor.detach().cpu()))
        else:
            # Random Gaussian sample matrix
            matrix = tensor.view(tensor.shape[0], -1)
            n, m = matrix.shape[0], matrix.shape[1]
            rank = min(n, m, self.rank)

            # low-rank decomposition via matmul
            with torch.no_grad():
                q = torch.randn(m, rank).cuda()
                p = torch.matmul(matrix, q)
                orthogonalize(p)
                # TODO: need to reduce p here for multi party??
                # but 1 party not working
                q = torch.matmul(matrix.t(), p)

            # pack tensors
            data.extend(pack_tensor(p.cpu()))
            data.extend(pack_tensor(q.cpu()))
        return data

    def decompress(self, data: bytearray,
                   offset: Offset = None) -> torch.Tensor:
        offset = offset or Offset()
        torch.manual_seed(self.seed)
        shape = unpack_tensor_shape(data, offset)
        if len(shape) <= 1:
            return unpack_tensor(data, offset)
        else:
            p = unpack_tensor(data, offset)
            q = unpack_tensor(data, offset)
            return torch.matmul(p, q.t()).view(shape)


class RankKPerLayer(
        _PerLayerCompressor, _RandomCompressor, _LowRankCompressor):
    def __init__(self, ranks: Union[int, List[int]] = 3, seed=random.random()):
        if isinstance(ranks, int):
            super().__init__(compressors=TensorRankK(ranks, None))
        elif isinstance(ranks, list):
            super().__init__(compressors=[TensorRankK(r, None) for r in ranks])
        self.set_seed(seed)

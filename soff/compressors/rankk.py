"""SVD-based compressor"""
import random
import torch
from munch import Munch
from .base import (
    _TensorRandom, _TensorLowRank, _PerLayerCompressor)
from .compress_utils import (
    Offset, pack_tensor, unpack_tensor,
    pack_tensor_shape, unpack_tensor_shape, orthogonalize)
from ..utils.arg_parser import ArgParseOption, options


class TensorRankK(_TensorRandom, _TensorLowRank):
    def __init__(self, rank: int, seed=random.Random()):
        super().__init__(seed=seed, rank=rank)

    def compress(self, tensor):
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
                q = torch.randn(m, rank).to(tensor.device)
                p = torch.matmul(matrix, q)
                orthogonalize(p)
                # TODO: need to reduce p here for multi party??
                # but 1 party not working
                q = torch.matmul(matrix.t(), p)

            # pack tensors
            data.extend(pack_tensor(p.cpu()))
            data.extend(pack_tensor(q.cpu()))
        return data

    def decompress(self, data, offset=None):
        offset = offset or Offset()
        torch.manual_seed(self.seed)
        shape = unpack_tensor_shape(data, offset)
        if len(shape) <= 1:
            return unpack_tensor(data, offset)
        else:
            p = unpack_tensor(data, offset)
            q = unpack_tensor(data, offset)
            return torch.matmul(p, q.t()).view(shape)


@options(
    "RandK-Per-Layer Compressor Configs",
    ArgParseOption(
        'rnkk.r', 'rankk-per-layer.ranks',
        type=int, nargs='+', metavar='RANK',
        help="Compressor rank"),
    ArgParseOption(
        'rnkk.s', 'rankk-per-layer.seed', type=int,
        help="Compressor seed"))
class RankKPerLayer(_PerLayerCompressor):
    def __init__(self, cfg: Munch):
        super().__init__(compressors=[
            TensorRankK(r, cfg.rankk_perlayer.seed)
            for r in cfg.rankk_per_layer.ranks])

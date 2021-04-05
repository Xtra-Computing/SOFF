import torch
from typing import Union, List
from oarf.compressors.base import (
    _TensorLowRank, _PerLayerCompressor, _LowRankCompressor)
from oarf.compressors.compress_utils import (
    Offset, pack_tensor, unpack_tensor, pack_float32, unpack_float32,
    pack_tensor_shape, unpack_tensor_shape)


class TensorSVD(_TensorLowRank):
    def __init__(self, rank: int):
        super().__init__(rank=rank)

    def compress(self, tensor: torch.Tensor) -> bytearray:
        data = bytearray()
        # pack the shape of the tensor
        data.extend(pack_tensor_shape(tensor))
        # pack 0 and 1-dim tensor as-is
        if tensor.ndim <= 1:
            data.extend(pack_tensor(tensor.detach().cpu()))
        else:
            # Random Gaussian sample matrix
            mean = torch.mean(tensor)

            # zero the mean
            matrix = tensor.view(tensor.shape[0], -1) - mean

            n, m = matrix.shape[0], matrix.shape[1]
            rank = min(n, m, self.rank)

            # low-rank decomposition via matmul
            with torch.no_grad():
                U, S, V = torch.svd(matrix)

            # pack mean and diag
            data.extend(pack_float32(mean))
            data.extend(pack_tensor(S[:rank].cpu()))

            # pack tensors
            data.extend(pack_tensor(U[:, :rank].cpu()))
            data.extend(pack_tensor(V[:, :rank].cpu()))
        return data

    def decompress(self, data: bytearray,
                   offset: Offset = None) -> torch.Tensor:
        offset = offset or Offset()

        shape = unpack_tensor_shape(data, offset)
        if len(shape) <= 1:
            return unpack_tensor(data, offset)
        else:
            mean = unpack_float32(data, offset)
            S = unpack_tensor(data, offset)
            U = unpack_tensor(data, offset)
            V = unpack_tensor(data, offset)
            return (torch.mm(
                torch.mm(U, torch.diag(S)), V.t()) + mean).view(shape)


class SVDPerLayer(_PerLayerCompressor, _LowRankCompressor):
    def __init__(self, ranks: Union[int, List[int]] = 3):
        if isinstance(ranks, int):
            super().__init__(compressors=TensorSVD(ranks))
        elif isinstance(ranks, list):
            super().__init__(compressors=[TensorSVD(r) for r in ranks])

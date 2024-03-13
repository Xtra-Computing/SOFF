"""Sparsification-based compressor"""

import torch
from .base import _TensorCompressor, _PerModelCompressor
from .compress_utils import (
    Offset, pack_tensor, unpack_tensor,
    pack_tensor_shape, unpack_tensor_shape)


class TensorSparse(_TensorCompressor):
    """Tensor sparsifier"""
    def compress(self, tensor):
        data = bytearray()

        # pack original tensor shape
        pack_tensor_shape(tensor)

        # sparsify tensor
        sparse_tensor = tensor.to_sparse()
        data.extend(pack_tensor(sparse_tensor.indices().detach().cpu()))
        data.extend(pack_tensor(sparse_tensor.values().detach().cpu()))

        return data

    def decompress(self, data, offset=None):
        offset = offset or Offset()

        # unpack original tensor shape
        shape = unpack_tensor_shape(data, offset)

        # unpack indices and data
        indices = unpack_tensor(data, offset)
        values = unpack_tensor(data, offset)

        # re-construct tensor
        tensor = torch.zeros(shape)
        tensor.index_copy_(0, indices, values)

        return tensor


class SparseCompress(_PerModelCompressor):
    """Sparsification-based tensor compressor"""
    def __init__(self):
        super().__init__(compressor=TensorSparse())

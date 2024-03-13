"""No-op compressor"""

import torch
from .base import _TensorCompressor, _PerModelCompressor
from .compress_utils import Offset, pack_tensor, unpack_tensor


class TensorNoCompress(_TensorCompressor):
    """no any compress, just transform the model to data"""

    def compress(self, tensor: torch.Tensor):
        return pack_tensor(tensor.clone().detach().cpu())

    def decompress(self, data, offset=None):
        offset = offset or Offset()
        return unpack_tensor(data, offset)


class NoCompress(_PerModelCompressor):
    """No-op compressor"""

    def __init__(self):
        super().__init__(compressor=TensorNoCompress())

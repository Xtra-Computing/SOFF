import torch
from oarf.compressors.base import _TensorCompressor, _PerModelCompressor
from oarf.compressors.compress_utils import Offset, pack_tensor, unpack_tensor


class TensorNoCompress(_TensorCompressor):
    """no any compress, just transform the model to data"""

    def compress(self, tensor: torch.Tensor) -> bytearray:
        return pack_tensor(tensor.clone().detach().cpu())

    def decompress(self, data: bytearray,
                   offset: Offset = None) -> torch.Tensor:
        offset = offset or Offset()
        return unpack_tensor(data, offset)


class NoCompress(_PerModelCompressor):
    def __init__(self):
        super().__init__(compressors=TensorNoCompress())

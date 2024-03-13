"""Base class of tensor compressors and model compressors"""
import copy
from types import GeneratorType
from typing import Optional, List
from abc import ABC, abstractmethod
import torch
from torch.nn.utils.convert_parameters import (
    parameters_to_vector, vector_to_parameters)
from .compress_utils import (
    pack_int, unpack_int, Offset, pack_tensor_shape,
    unpack_tensor_shape, check_single_or_plural, unpack_fixed_width_data)


# Tensor (layer) compressors ##################################################
class _TensorCompressor(ABC):
    """
    The compressor compresses a network by default
    If need to compress gradient, store the gradients into model parameters
    """

    @abstractmethod
    def compress(self, tensor: torch.Tensor) -> bytearray:
        """Compress the tensor. Return the compressed bytearray"""
        raise NotImplementedError("Usage of abstract method.")

    @abstractmethod
    def decompress(self, data: bytearray,
                   offset: Optional[Offset] = None) -> torch.Tensor:
        """Decompress the bytearray. Return the decomprssed tensor"""
        offset = offset or Offset()
        raise NotImplementedError("Usage of abstract method.")

    @staticmethod
    def zero_with_reference(tensor: torch.Tensor, reference: torch.Tensor):
        """
        zero out all elements in `tensor` where the value in the corresponding
        position in `reference` is 0
        Args:
            reference: reference gradients represented as a list of tensor
        """
        # if no `with torch.no_grad()` --> memory leaking
        with torch.no_grad():
            mask = (reference == 0.0).float().to(tensor.device)
            tensor.mul_(mask)


class _TensorSubsample(_TensorCompressor):
    """ attr `ratio`: compression ratio """

    def __init__(self, ratio: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0 <= ratio <= 1
        self.ratio = ratio

    # @abstractmethod
    # def zero_with_mask(self, tensor: torch.Tensor) -> None:
    #     """Zero `tensor` with computed topk mask"""
    #     raise NotImplementedError("Please use a concrete subclass")


class _TensorRandom(_TensorCompressor):
    """ attr `seed`: must be the same when compress and decompress """

    def __init__(self, seed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed


class _TensorLowRank(_TensorCompressor):
    """ attr `rank`: (de)compression rank"""

    def __init__(self, rank: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert rank >= 1
        self.rank = rank


# Model compressors ###########################################################
class _ModelCompressor(ABC):
    @abstractmethod
    def compress(self, params: List[torch.Tensor]) -> bytearray:
        """Compress model parameters/gradients"""
        raise NotImplementedError("Pleas use a concrete subclass")

    @abstractmethod
    def decompress(self, data: bytearray) -> List[torch.Tensor]:
        """Decompress parameters/gradients"""
        raise NotImplementedError("Pleas use a concrete subclass")

    @staticmethod
    def zero_with_reference(net: torch.nn.Module, reference: List[torch.Tensor]):
        """
        zero out all elements in `net` that has 0 value in the corresponding
        position in `reference`
        Args:
            reference: reference gradients represented as a list of tensor
        """
        # if no `with torch.no_grad()` --> memory leaking
        with torch.no_grad():
            for param, ref in zip(net.parameters(), reference):
                _TensorCompressor.zero_with_reference(param, ref)

    @staticmethod
    def check_compressors(compressors, compressor_type):
        """Check whether all `compressors` have type `compressor_type`"""
        check_single_or_plural(
            compressors,
            lambda x: isinstance(x, compressor_type),
            lambda v: True)


class _PerModelCompressor(_ModelCompressor):
    """Treat the whole model as one single vector and compress it"""

    def __init__(self, compressor: _TensorCompressor):
        super().__init__()
        assert isinstance(compressor, _TensorCompressor)
        self.compressor = compressor

    type_map = {
        torch.uint8: b'u',
        torch.int8: b's',
        torch.int16: b'S',
        torch.int32: b'i',
        torch.int64: b'I',
        torch.float16: b'd',
        torch.float32: b'f',
        torch.float64: b'F',
        torch.complex64: b'c',
        torch.complex128: b'C'
    }
    type_reverse_map = {v: k for k, v in type_map.items()}

    def compress(self, params: List[torch.Tensor]) -> bytearray:
        assert not isinstance(params, GeneratorType), \
            "Don't pass a generator (e.g. module.parameters()). It causes error"

        data = bytearray()

        # pack number of tensors
        data.extend(pack_int(len(list(params))))

        # deal with empty list. (parameters_to_vector rejects empty list)
        if len(list(params)) == 0:
            return data

        # convert every parameters to a big vector and compress it
        big_vec = parameters_to_vector(params)

        # pack the dim and shape for each tensor
        for param in params:
            data.extend(self.type_map[param.dtype])
            data.extend(pack_tensor_shape(param))

        # pack compressed data
        data.extend(self.compressor.compress(big_vec))

        return data

    def decompress(self, data: bytearray) -> List[torch.Tensor]:
        result: List[torch.Tensor] = []
        result_types = []
        offset = Offset(0)

        # get number of tensors
        num_tensors = unpack_int(data, offset)

        # deal with empty list
        if num_tensors == 0:
            return result

        # re-construct the tensor list
        for i in range(num_tensors):
            dtype = bytes(unpack_fixed_width_data(data, offset, 1))
            result_types.append(self.type_reverse_map[dtype])
            shape = unpack_tensor_shape(data, offset)
            result.append(torch.zeros(shape))

        # restore compressed data
        compressed_data = self.compressor.decompress(data, offset)
        vector_to_parameters(compressed_data, result)

        # restore types
        for i, res in enumerate(result):
            result[i] = res.to(dtype=result_types[i])
        # for i in range(len(result)):
        #     result[i] = result[i].to(dtype=result_types[i])

        return result

    # def zero_with_mask(self, net: torch.nn.Module):
    #     assert hasattr(self.compressor, 'zero_with_mask')
    #     net_as_vec = parameters_to_vector(net.parameters())
    #     self.compressor.zero_with_mask(net_as_vec)
    #     vector_to_parameters(net_as_vec, net.parameters())


class _PerLayerCompressor(_ModelCompressor):
    def __init__(self, compressors: List[_TensorCompressor]):
        super().__init__()
        assert all(isinstance(c, _TensorCompressor) for c in compressors)
        self.compressors = compressors

    def compress(self, params: List[torch.Tensor]) -> bytearray:
        assert not isinstance(params, GeneratorType), \
            "Don't pass a generator (e.g. module.parameters()). It causes error"

        # choose compressor or compressor list
        assert len(self.compressors) in (1, len(list(params)))
        compressors = (
            [copy.deepcopy(self.compressors[0])
             for _ in range(len(list(params)))]
            if len(self.compressors) == 1 else self.compressors)

        # encode number of layers
        data = bytearray()
        data.extend(pack_int(len(list(params))))

        # encode layer data
        for compressor, param in zip(compressors, params):
            data.extend(compressor.compress(param))

        return data

    def decompress(self, data: bytearray) -> List[torch.Tensor]:
        result = []

        offset = Offset(0)
        num_layers = unpack_int(data, offset)

        assert len(self.compressors) in (1, num_layers)
        compressors = (
            [copy.deepcopy(self.compressors[0]) for _ in range(num_layers)]
            if len(self.compressors) == 1 else self.compressors)

        for i in range(num_layers):
            result.append(compressors[i].decompress(data, offset))

        return result

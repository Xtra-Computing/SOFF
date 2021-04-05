import torch
import random
from typing import Union, List
from oarf.compressors.compress_utils import (
    pack_int, unpack_int, Offset, pack_tensor_shape,
    unpack_tensor_shape, check_single_or_plural, unpack_fixed_width_data)


# Tensor (layer) compressors ##################################################
class _TensorCompressor:
    """
    The compressor compresses a network by default
    If need to compress gradient, store the gradients into model parameters
    """

    def compress(self, tensor: torch.Tensor) -> bytearray:
        raise NotImplementedError("Usage of abstract method.")

    def decompress(self, data: bytearray,
                   offset: Offset = None) -> torch.Tensor:
        offset = offset or Offset()
        raise NotImplementedError("Usage of abstract method.")

    # zero out all elements in `tensor` that has 0 value in the corresponding
    # position in `reference`
    # @param reference: reference gradients represented as a list of tensor
    @staticmethod
    def zero_with_reference(tensor: torch.Tensor, reference: torch.Tensor):
        # if no `with torch.no_grad()` --> memory leaking
        with torch.no_grad():
            mask = (reference == 0.0).float().cuda()
            tensor.mul_(mask)


class _TensorSubsample(_TensorCompressor):
    """ attr `ratio`: compression ratio """

    def __init__(self, ratio: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (0 <= ratio <= 1)
        self.ratio = ratio

    def zero_with_mask(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError("Please use a concrete subclass")


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
class _ModelCompressor:
    def __init__(self, compressors: Union[_TensorCompressor, List[
            _TensorCompressor]], *args, **kwargs):
        """
        compressors: can be a single tensor compressor or a list of them
        """
        self.check_compressors(compressors, _TensorCompressor)
        self.compressors = compressors

    def compress(self, params: List[torch.Tensor]) -> bytearray:
        raise NotImplementedError("Pleas use a concrete subclass")

    def decompress(self, data: bytearray) -> [torch.Tensor]:
        raise NotImplementedError("Pleas use a concrete subclass")

    # zero out all elements in `net` that has 0 value in the corresponding
    # position in `reference`
    # @param reference: reference gradients represented as a list of tensor
    @staticmethod
    def zero_with_reference(net: torch.nn.Module, reference: [torch.Tensor]):
        # if no `with torch.no_grad()` --> memory leaking
        with torch.no_grad():
            for param, ref in zip(net.parameters(), reference):
                _TensorCompressor.zero_with_reference(param, ref)

    @staticmethod
    def check_compressors(compressors, compressor_type):
        check_single_or_plural(compressors,
                               lambda x: isinstance(x, compressor_type),
                               lambda v: True)


class _PerModelCompressor(_ModelCompressor):
    """Treat the whole model as one single vector and compress it"""

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compress(self, params: List[torch.Tensor]) -> bytearray:
        data = bytearray()

        # convert every parameters to a big vector and compress it
        big_vec = torch.nn.utils.parameters_to_vector(params)

        # pack number of tensors
        data.extend(pack_int(len(list(params))))
        # pack the dim and shape for each tensor
        for param in params:
            data.extend(self.type_map[param.dtype])
            data.extend(pack_tensor_shape(param))
        # pack compressed data
        data.extend(self.compressors.compress(big_vec))

        return data

    def decompress(self, data: bytearray) -> [torch.Tensor]:
        result = []
        result_types = []
        offset = Offset(0)

        # get number of tensors
        num_tensors = unpack_int(data, offset)

        # re-construct the tensor list
        for i in range(num_tensors):
            dtype = bytes(unpack_fixed_width_data(data, offset, 1))
            result_types.append(self.type_reverse_map[dtype])
            shape = unpack_tensor_shape(data, offset)
            result.append(torch.zeros(shape))

        # restore compressed data
        compressed_data = self.compressors.decompress(data, offset)
        torch.nn.utils.vector_to_parameters(compressed_data, result)

        # restore types
        for i in range(len(result)):
            result[i] = result[i].to(dtype=result_types[i])

        return result

    def zero_with_mask(self, net: torch.nn.Module):
        assert hasattr(self.compressors, 'zero_with_mask')
        net_as_vec = torch.nn.utils.parameters_to_vector(net.parameters())
        self.compressors.zero_with_mask(net_as_vec)
        torch.nn.utils.vector_to_parameters(net_as_vec, net.parameters())


class _PerLayerCompressor(_ModelCompressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compress(self, params: List[torch.Tensor]) -> bytearray:
        # choose compressor or compressor list
        if isinstance(self.compressors, _TensorCompressor):
            compressors = [self.compressors] * len(list(params))
        elif isinstance(self.compressors, list):
            assert (len(self.compressors) == len(list(params)))
            compressors = self.compressors

        # encode number of layers
        data = bytearray()
        data.extend(pack_int(len(list(params))))

        # encode layer data
        for compressor, param in zip(compressors, params):
            data.extend(compressor.compress(param))

        return data

    def decompress(self, data: bytearray) -> [torch.Tensor]:
        result = []

        offset = Offset(0)
        num_layers = unpack_int(data, offset)

        if isinstance(self.compressors, _TensorCompressor):
            compressors = [self.compressors] * num_layers
        elif isinstance(self.compressors, list):
            assert (len(self.compressors) == num_layers)
            compressors = self.compressors

        for i in range(num_layers):
            result.append(compressors[i].decompress(data, offset))

        return result


class _SubsampleCompressor(_ModelCompressor):
    """ Provides ratio checking and update support """

    def __init__(self, compressors, *args, **kwargs):
        """
        ratios: can be a single float (apply to all layer) or a list (whose
               length is the same as the same as number of layers to compress).
        """
        self.check_compressors(compressors, _TensorSubsample)
        super().__init__(compressors=compressors, *args, **kwargs)

    def set_ratios(self, ratios):
        self.check_ratios(ratios)
        if isinstance(self.compressors, _TensorSubsample):
            if isinstance(ratios, float):
                self.compressors.ratio = ratios
            else:
                raise ValueError("ratio should be a single value")
        elif isinstance(self.compressors, list):
            if isinstance(ratios, float):
                for compressor in self.compressors:
                    compressor.ratio = ratios
            elif isinstance(ratios, list):
                assert (len(self.compressors) == ratios)
                for compressor, ratio in zip(self.compressors, ratios):
                    compressor.ratio = ratio
        else:
            raise ValueError("must use (a list of) subsample compressor")

    @staticmethod
    def check_ratios(ratios):
        check_single_or_plural(ratios, lambda x: isinstance(x, float),
                               lambda r: 0 <= r <= 1)


class _LowRankCompressor(_ModelCompressor):
    def __init__(self, compressors, *args, **kwargs):
        """
        ranks: can be a single float (apply to all layer) or a list (whose
               length is the same as the same as number of layers to compress).
        """
        self.check_compressors(compressors, _TensorLowRank)

        super().__init__(compressors=compressors, *args, **kwargs)

    def set_ranks(self, ranks):
        self.check_ranks(ranks)
        if isinstance(self.compressors, _TensorLowRank):
            if isinstance(ranks, int):
                self.compressors.rank = ranks
            else:
                raise ValueError("ratio should be a single value")
        elif isinstance(self.compressors, list):
            if isinstance(ranks, float):
                for compressor in self.compressors:
                    compressor.rank = ranks
            elif isinstance(ranks, list):
                assert (len(self.compressors) == ranks)
                for compressor, rank in zip(self.compressors, ranks):
                    compressor.rank = rank
        else:
            raise ValueError("must use (a list of) lowrank compressor")

    @staticmethod
    def check_ranks(ranks):
        check_single_or_plural(ranks, lambda x: isinstance(x, int),
                               lambda r: r >= 1)


class _RandomCompressor(_ModelCompressor):
    """ Provides seed checking and update support """

    def __init__(self, compressors, *args, **kwargs):
        self.check_compressors(compressors, _TensorRandom)
        super().__init__(compressors=compressors, *args, **kwargs)

    def set_seed(self, seed):
        rand_gen = random.Random(seed)
        rand_gen.random()

        if isinstance(self.compressors, _TensorRandom):
            self.compressors.seed = rand_gen.random()
        elif isinstance(self.compressors, list):
            for compressor in self.compressors:
                compressor.seed = rand_gen.random()
        else:
            raise ValueError("must use (a list of) random compressor")


class HybridComporessor(_PerLayerCompressor):
    def __init__(self, comprssors, compressor_args):
        pass

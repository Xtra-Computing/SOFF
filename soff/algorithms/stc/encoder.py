"""Encoder for quantization results"""
import logging
import importlib
from enum import IntEnum
from types import ModuleType
from abc import ABC, abstractmethod
import torch
import numpy as np
from munch import Munch
from torch import Tensor
from .quantizer import QuantizeResult
from ...utils.arg_parser import BaseConfParser
from ...compressors.compress_utils import (
    Offset, pack_int, unpack_int, pack_tensor, unpack_tensor,
    pack_raw_data, unpack_raw_data)

log = logging.getLogger(__name__)


class Const(IntEnum):
    """Constants"""
    BINARY_BIT = 1
    NIBBLE_BIT = 4
    BYTE_BIT = 8
    HALF_WORD_BIT = 16
    WORD_BIT = 32
    FLOAT_BIT = 32
    DOUBLE_WORD_BIT = 64


class EntropyCoder(ABC):
    """Interface for compressing and decompressing a given tensor."""

    @abstractmethod
    def encode(self, result: QuantizeResult) -> bytearray:
        """Compresses a quantization result"""

    @abstractmethod
    def decode(self, data: bytearray) -> QuantizeResult:
        """Decompress the data and returns the quatization result"""

    @staticmethod
    def _entropy(histogram) -> float:
        return float(sum(prob and -prob * np.log2(prob) for prob in histogram))


class PlainCoder(EntropyCoder):
    """A plain coder does nothing but calculate the number of bits"""

    def __init__(self, cfg: Munch):
        self.total_codewords = cfg.stc.quantizer.quantization_level
        if cfg.stc.quantizer.quantization_level == 2:
            # sign bits are overlapped with level bits
            self.ratio_ = 0
        elif cfg.stc.quantizer.quantization_level == 0:
            # this is kept for FedAvg testing only
            self.ratio_ = Const.FLOAT_BIT
        else:
            self.ratio_ = np.ceil(np.log2(self.total_codewords))

    def encode(self, result):
        tensor = result.quantized_tensor
        total_symbols = torch.prod(torch.tensor(tensor.shape)).item()

        log.debug(
            "original bits: %s, Compressed bits: %s",
            total_symbols * Const.FLOAT_BIT,
            self._compressed_bits(total_symbols))

        return pack_tensor(result.norms) + pack_tensor(tensor)

    def decode(self, data):
        """Plain decoder directly returns the code."""
        offset = Offset()
        norms = unpack_tensor(data, offset)
        tensor = unpack_tensor(data, offset)
        return QuantizeResult(norms=norms, quantized_tensor=tensor)

    def _compressed_bits(self, total_symbols):
        magnitude_bits = 1 * Const.FLOAT_BIT
        if self.ratio_ == Const.FLOAT_BIT:
            return total_symbols * self.ratio_
        return total_symbols * self.ratio_ + magnitude_bits


class IdealCoder(EntropyCoder):
    """
    An ideal coder uses the histogram as an estimation of entropy
    and assume the entropy lower bound can be achieved. Nothing is
    actually done but the number of estimated bits will be returned
    """

    def __init__(self, cfg: Munch):
        # self.total_codewords = cfg.stc.quantizer.quantization_level
        self.total_codewords = int(2 ** np.ceil(
            np.log2(cfg.stc.quantizer.quantization_level)))

    def entropy(self, tensor: Tensor):
        histogram = torch.histc(
            tensor, bins=self.total_codewords,
            min=0, max=self.total_codewords-1)
        total_symbols = torch.sum(histogram).item()

        # log.debug(
        #     "Original bits: %s, Compressed bits: %s",
        #     total_symbols * Const.FLOAT_BIT,
        #     self._compressed_bits(total_symbols, entropy))

        return self._entropy(
            (histogram.to(torch.float)/total_symbols).detach().cpu().numpy())

    def encode(self, result):
        """
        Simulate an ideal entropy coding to a quantized tensor
        without actually coding the tensor array.
        """
        return pack_tensor(result.norms) + pack_tensor(result.quantized_tensor)

    def decode(self, data):
        """Simulate an ideal entropy decoding without actually decoding."""
        offset = Offset()
        norms = unpack_tensor(data, offset)
        tensor = unpack_tensor(data, offset)
        return QuantizeResult(norms=norms, quantized_tensor=tensor)

    def _compressed_bits(self, total_symbols, ratio):
        magnitude_bits = 1 * Const.FLOAT_BIT
        return total_symbols*ratio + magnitude_bits


class ArithmeticCoder(EntropyCoder):
    """https://github.com/fab-jul/torchac"""
    torchac: ModuleType

    def __init__(self, cfg: Munch):
        # self.symbs = cfg.stc.quantizer.quantization_level
        self.symbs = int(2 ** np.ceil(
            np.log2(cfg.stc.quantizer.quantization_level)))

        # Torchac requires JIT compilation, thus avoid loading when not needed
        if not hasattr(ArithmeticCoder, 'torchac'):
            ArithmeticCoder.torchac = importlib.import_module('torchac')

    def encode(self, result):
        # assert all(result.quantized_tensor >= 0) # slow
        assert len(result.quantized_tensor.shape) == 1

        tensor = result.quantized_tensor
        seq_len = int(tensor.shape[0])
        histc = torch.histc(tensor, bins=self.symbs, min=0, max=self.symbs-1)
        cdf = self._histc_to_cdf(histc.cpu(), seq_len)
        encoded = self.torchac.encode_float_cdf(
            cdf.repeat(seq_len, 1), tensor.short().cpu())

        return (
            pack_int(seq_len) + pack_tensor(result.norms.cpu()) +
            pack_tensor(histc.cpu()) + pack_raw_data(encoded))

    def decode(self, data):
        offset = Offset()
        seq_len = unpack_int(data, offset)
        norms = unpack_tensor(data, offset)
        histc = unpack_tensor(data, offset)
        encoded = bytes(unpack_raw_data(data, offset))

        cdf = self._histc_to_cdf(histc, seq_len)
        tensor = self.torchac.decode_float_cdf(cdf.repeat(seq_len, 1), encoded)
        return QuantizeResult(norms=norms, quantized_tensor=tensor)

    @staticmethod
    def _histc_to_cdf(histc: Tensor, num_symbs: int):
        return torch.cumsum(
            torch.cat((torch.FloatTensor([0.]), histc), dim=0), dim=0
        ) / float(num_symbs)

encoders = {
    "plain":        PlainCoder,
    "entropy":      IdealCoder,
    "arithmetic":   ArithmeticCoder,
}

class STCEncoderConfParser(BaseConfParser):
    """Config parser for encoders"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = self.add_argument_group(
            "STC Encoder Configs (S,S->C)")
        args.add_argument(
            '-stc.ec.n', '--stc.encoder.name',
            default='arithmetic', choices=encoders.keys(),
            help="Type of encoder to use")


def create_encoder(cfg: Munch) -> EntropyCoder:
    return encoders[cfg.stc.encoder.name](cfg)

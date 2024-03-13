"""Tensor quantizer"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type
import torch
from munch import Munch
from torch import Tensor
from ...utils.arg_parser import BaseConfParser


@dataclass
class QuantizeResult:
    norms: Tensor
    quantized_tensor: Tensor


class Quantizer(ABC):
    """Interface for quantizing and dequantizing a given tensor."""

    def __init__(self, *_, **__) -> None:
        pass

    @abstractmethod
    def quantize(self, tensor: Tensor) -> QuantizeResult:
        """Compresses a tensor with the given compression context,
        and then returns it with the context needed to decompress it."""

    @abstractmethod
    def dequantize(self, result: QuantizeResult) -> Tensor:
        """Decompress the tensor with the given decompression context."""


class NoneQuantizer(Quantizer):
    """No-op quantizer"""

    def quantize(self, tensor):
        return QuantizeResult(norms=Tensor(), quantized_tensor=tensor)

    def dequantize(self, result):
        return result.quantized_tensor


class StcQuantizer(Quantizer):
    """STC quantizer"""

    def __init__(self, cfg: Munch):
        # Verify the quantization level so that encoder can correctly en/decode
        assert cfg.qsgd.quantizer.quantization_level == 3

    def quantize(self, tensor):
        norm = tensor.abs().mean()
        return QuantizeResult(
            norms=norm, quantized_tensor=(tensor.sign() + torch.tensor(1.)))

    def dequantize(self, result):
        """Decode the signs to float format """
        quantized_arr = (result.quantized_tensor - torch.tensor(1.))
        norm = result.norms
        return norm * quantized_arr


class QsgdPosQuantizer(Quantizer):
    def __init__(self, cfg):
        self.quantlevel = cfg.qsgd.quantizer.quantization_level
        self.quantbound = (cfg.qsgd.quantizer.quantization_level - 1)/2
        self._cut_neg = (self.quantlevel % 2 == 0)

    def quantize(self, tensor):
        norm = tensor.norm()
        abs_arr = tensor.abs()

        level_float = abs_arr / norm * self.quantbound
        lower_level = level_float.floor()
        rand_variable = torch.empty_like(tensor).uniform_()
        is_upper_level = rand_variable < (level_float - lower_level)
        new_level = (lower_level + is_upper_level)

        sign = tensor.sign()
        quantized_arr = sign * torch.round(new_level).to(torch.int)

        if self._cut_neg:
            quantized_arr = torch.where(
                quantized_arr == -torch.ceil(torch.tensor(self.quantbound)),
                quantized_arr+1, quantized_arr)

        quantized_arr = torch.where(
            quantized_arr > 0, 2*quantized_arr-1, -2*quantized_arr)

        return QuantizeResult(norms=norm, quantized_tensor=quantized_arr)

    def dequantize(self, result):
        quant_arr = result.quantized_tensor
        dequant_arr = torch.where(
            quant_arr % 2 == 0, -0.5*quant_arr, 0.5*(quant_arr+1))
        dequant_arr = dequant_arr/self.quantbound
        dequant_arr = result.norms * dequant_arr
        return dequant_arr


class QsgdinfPosQuantizer(Quantizer):
    def __init__(self, cfg):
        self.quantlevel = cfg.qsgd.quantizer.quantization_level
        self.quantbound = (cfg.qsgd.quantizer.quantization_level - 1)/2
        self._cut_neg = (self.quantlevel % 2 == 0)

    def quantize(self, tensor: Tensor):
        norm = torch.max(tensor.abs())
        abs_arr = tensor.abs()

        level_float = abs_arr / norm * self.quantbound
        lower_level = level_float.floor()
        rand_variable = torch.empty_like(tensor).uniform_()
        is_upper_level = rand_variable < (level_float - lower_level)
        new_level = (lower_level + is_upper_level)

        sign = tensor.sign()
        quantized_arr = sign * torch.round(new_level).to(torch.int)

        if self._cut_neg:
            quantized_arr = torch.where(
                quantized_arr == -torch.ceil(torch.tensor(self.quantbound)),
                quantized_arr+1, quantized_arr)

        quantized_arr = torch.where(
            quantized_arr > 0, 2*quantized_arr-1, -2*quantized_arr)
        return QuantizeResult(norms=norm, quantized_tensor=quantized_arr)

    def dequantize(self, result):
        quant_arr = result.quantized_tensor
        dequant_arr = torch.where(
            quant_arr % 2 == 0, -0.5*quant_arr, 0.5*(quant_arr+1))
        dequant_arr = dequant_arr/self.quantbound
        dequant_arr = result.norms * dequant_arr
        return dequant_arr


quantizers: Dict[str, Type[Quantizer]] = {
    "none":         NoneQuantizer,
    "stc":          StcQuantizer,
    "qsgdpos":      QsgdPosQuantizer,
    "qsgdinf":      QsgdinfPosQuantizer,
}


class QSGDQuantizerConfParser(BaseConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = self.add_argument_group(
            "QSGD Quantizer Configs (S,S->C)")
        args.add_argument(
            '-qsgd.qt.n', '--qsgd.quantizer.name',
            default='qsgdinf', choices=quantizers.keys(),
            help="Type of quantizer to use")
        args.add_argument(
            '-qsgd.qt.ql', '--qsgd.quantizer.quantization-level',
            type=int, default=3,
            help="Quantization level for the quantizer")


def create_quantizer(cfg: Munch) -> Quantizer:
    return quantizers[cfg.qsgd.quantizer.name](cfg)

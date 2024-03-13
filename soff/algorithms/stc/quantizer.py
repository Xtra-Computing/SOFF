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
        assert cfg.stc.quantizer.quantization_level == 3

    def quantize(self, tensor):
        norm = tensor.abs().mean()
        return QuantizeResult(
            norms=norm, quantized_tensor=(tensor.sign() + torch.tensor(1.)))

    def dequantize(self, result):
        """Decode the signs to float format """
        quantized_arr = (result.quantized_tensor - torch.tensor(1.))
        norm = result.norms
        return norm * quantized_arr


class StcSignedQuantizer(Quantizer):
    """Signed STC quantizer"""

    def __init__(self, cfg: Munch):
        # Verify the quantization level so that encoder can correctly en/decode
        assert cfg.stc.quantizer.quantization_level == 3

    def quantize(self, tensor):
        norm_pos = (
            tensor[tensor > 0.].abs().mean()
            if tensor[tensor > 0].numel() > 0 else torch.tensor(0.))
        norm_neg = (
            tensor[tensor < 0.].abs().mean()
            if tensor[tensor < 0].numel() > 0 else torch.tensor(0.))
        return QuantizeResult(
            norms=torch.FloatTensor([norm_pos, norm_neg]),
            quantized_tensor=(tensor.sign() + torch.tensor(1.)))

    def dequantize(self, result):
        assert result.norms.numel() == 2
        qarr = result.quantized_tensor - torch.tensor(1.)
        norm_pos, norm_neg = result.norms
        return (norm_pos * qarr.where(qarr > 0, torch.zeros_like(qarr)) +
                norm_neg * qarr.where(qarr < 0, torch.zeros_like(qarr)))

class ExtendedStcSignedQuantizer(Quantizer):
    def __init__(self, cfg: Munch):
        self.quant_level = cfg.stc.quantizer.quantization_level
        self.quantbound = (cfg.stc.quantizer.quantization_level - 1) / 2
        assert self.quant_level % 2 != 0

    def quantize(self, tensor: Tensor):
        assert tensor.ndim == 1

        means = []
        sorted_tensor, sorted_idx = tensor.sort()
        sorted_pos_idxs = (sorted_tensor > 0).nonzero(as_tuple=True)[0]
        sorted_neg_idxs = (sorted_tensor < 0).nonzero(as_tuple=True)[0]

        quant_arr = torch.ones_like(tensor) * self.quantbound
        qbound = int(self.quantbound)
        for i in range(qbound):
            segment_idxs = sorted_idx[sorted_neg_idxs[
                sorted_neg_idxs.numel() * i // qbound:
                sorted_neg_idxs.numel() * (i+1) // qbound]]
            quant_arr.index_fill_(0, segment_idxs, i)
            means.append(tensor[segment_idxs].mean().item())
        for i in range(int(self.quantbound)):
            segment_idxs = sorted_idx[sorted_pos_idxs[
                sorted_pos_idxs.numel() * i//qbound:
                sorted_pos_idxs.numel() * (i+1)//qbound]]
            quant_arr.index_fill_(0, segment_idxs, i + self.quantbound + 1)
            means.append(tensor[segment_idxs].mean().item())

        return QuantizeResult(
            norms=torch.Tensor(means),
            quantized_tensor=quant_arr)

    def dequantize(self, result):
        means = result.norms
        quant_arr = result.quantized_tensor

        qbound = int(self.quantbound)
        dequant_arr = torch.zeros_like(quant_arr, dtype=torch.float)
        for i in range(qbound):
            dequant_arr[quant_arr == i] = means[i]
        for i in range(qbound):
            dequant_arr[quant_arr == i + qbound + 1] = means[i + qbound]
        return dequant_arr


quantizers: Dict[str, Type[Quantizer]] = {
    "none":         NoneQuantizer,
    "stc":          StcQuantizer,
    "stcs":         StcSignedQuantizer,
    "estcs":        ExtendedStcSignedQuantizer,
}


class STCQuantizerConfParser(BaseConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = self.add_argument_group(
            "STC Quantizer Configs (S,S->C)")
        args.add_argument(
            '-stc.qt.n', '--stc.quantizer.name',
            default='stcs', choices=quantizers.keys(),
            help="Type of quantizer to use")
        args.add_argument(
            '-stc.qt.ql', '--stc.quantizer.quantization-level',
            type=int, default=3,
            help="Quantization level for the quantizer")


def create_quantizer(cfg: Munch) -> Quantizer:
    return quantizers[cfg.stc.quantizer.name](cfg)

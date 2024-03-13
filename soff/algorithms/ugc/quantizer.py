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
        assert cfg.ugc.quantizer.quantization_level == 3

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
        assert cfg.ugc.quantizer.quantization_level == 3

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
        self.quant_level = cfg.ugc.quantizer.quantization_level
        self.quantbound = (cfg.ugc.quantizer.quantization_level - 1) / 2
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
        dequant_arr = torch.zeros_like(quant_arr)
        for i in range(qbound):
            dequant_arr[quant_arr == i] = means[i]
        for i in range(qbound):
            dequant_arr[quant_arr == i + qbound + 1] = means[i + qbound]
        return dequant_arr


class UniformPosQuantizer(Quantizer):
    def __init__(self, cfg: Munch):
        self.quant_level = cfg.ugc.quantizer.quantization_level
        self.quantbound = (cfg.ugc.quantizer.quantization_level - 1)/2
        self.mid_tread = (self.quant_level % 2 != 0)

    def quantize(self, tensor: Tensor):
        max_val = torch.max(tensor.abs())
        quant_step = max_val/self.quantbound
        quantized_arr = torch.floor(
            tensor/quant_step + (0.5 if self.mid_tread else 0))

        quantized_arr = torch.where(
            quantized_arr > 0, 2*quantized_arr-1, -2*quantized_arr)

        return QuantizeResult(norms=max_val, quantized_tensor=quantized_arr)

    def dequantize(self, result):
        quant_arr = result.quantized_tensor
        dequant_arr = torch.where(
            quant_arr % 2 == 0, -0.5*quant_arr, 0.5*(quant_arr+1))
        dequant_arr = dequant_arr/self.quantbound
        dequant_arr = result.norms * dequant_arr
        return dequant_arr


class UniformPosSignedQuantizer(Quantizer):
    def __init__(self, cfg: Munch):
        self.quant_level = cfg.ugc.quantizer.quantization_level
        self.quantbound = (cfg.ugc.quantizer.quantization_level - 1) / 2
        assert self.quant_level % 2 != 0

    def quantize(self, tensor: Tensor):
        pos_max, pos_min = (
            torch.max(tensor[tensor > 0.].abs()),
            torch.min(tensor[tensor > 0.].abs())
        ) if tensor[tensor > 0].numel() > 0 else (
            torch.tensor(0.), torch.tensor(0.))
        neg_max, neg_min = (
            torch.max(tensor[tensor < 0.].abs()),
            torch.min(tensor[tensor < 0.].abs())
        ) if tensor[tensor < 0].numel() > 0 else (
            torch.tensor(0.), torch.tensor(0.))
        eps = (pos_max + neg_max) / 1.e6

        pos_quant_step = (eps + pos_max - pos_min) / self.quantbound
        neg_quant_step = (eps + neg_max - neg_min) / self.quantbound

        pos_quantized_arr = torch.where(
            tensor > 0.,
            torch.floor((tensor - pos_min + pos_quant_step) / pos_quant_step),
            torch.zeros_like(tensor))
        neg_quantized_arr = torch.where(
            tensor < 0.,
            torch.ceil((tensor + neg_min - neg_quant_step)/neg_quant_step),
            torch.zeros_like(tensor))

        quantized_arr = (pos_quantized_arr +
                         neg_quantized_arr) + self.quantbound

        return QuantizeResult(
            norms=torch.Tensor([pos_max, pos_min, neg_max, neg_min]),
            quantized_tensor=quantized_arr)

    def dequantize(self, result):
        quant_arr = result.quantized_tensor
        dequant_arr = quant_arr - self.quantbound
        pos_max, pos_min, neg_max, neg_min = result.norms
        pos_quant_step = (pos_max - pos_min) / self.quantbound
        neg_quant_step = (neg_max - neg_min) / self.quantbound
        dequant_arr = torch.where(
            dequant_arr > 0,
            (quant_arr - 0.5) * pos_quant_step + pos_min,
            torch.zeros_like(dequant_arr)
        ) + torch.where(
            dequant_arr < 0,
            (quant_arr + 0.5) * neg_quant_step - neg_min,
            torch.zeros_like(dequant_arr)
        )

        return dequant_arr


class NormPosQuantizer(Quantizer):
    def __init__(self, cfg):
        self.quant_level = cfg.ugc.quantizer.quantization_level
        self.quantbound = (cfg.ugc.quantizer.quantization_level - 1)/2
        self._cut_neg = (self.quant_level % 2 == 0)
        self.scale = 90

    def quantize(self, tensor):
        norm = self.scale * torch.max(torch.abs(tensor))
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


class QsgdPosQuantizer(Quantizer):
    def __init__(self, cfg):
        self.quantlevel = cfg.ugc.quantizer.quantization_level
        self.quantbound = (cfg.ugc.quantizer.quantization_level - 1)/2
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

        quantized_set = QuantizeResult(
            norms=norm, quantized_tensor=quantized_arr)

        return quantized_set

    def dequantize(self, quantized_set):
        quant_arr = quantized_set.quantized_tensor
        dequant_arr = torch.where(
            quant_arr % 2 == 0, -0.5*quant_arr, 0.5*(quant_arr+1))
        dequant_arr = dequant_arr/self.quantbound
        dequant_arr = quantized_set.norms * dequant_arr
        return dequant_arr


class QsgdinfPosQuantizer(Quantizer):
    def __init__(self, cfg):
        self.quantlevel = cfg.ugc.quantizer.quantization_level
        self.quantbound = (cfg.ugc.quantizer.quantization_level - 1)/2
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
    "qsgdpos":      QsgdPosQuantizer,
    "qsgdinf":      QsgdinfPosQuantizer,
    "uniform":      UniformPosQuantizer,
    "norm":         NormPosQuantizer,
    "stc":          StcQuantizer,
    "stcs":         StcSignedQuantizer,
    "estcs":        ExtendedStcSignedQuantizer,
}


class UGCQuantizerConfParser(BaseConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = self.add_argument_group(
            "UGC Quantizer Configs (S,S->C)")
        args.add_argument(
            '-ugc.qt.n', '--ugc.quantizer.name',
            default='stcs', choices=quantizers.keys(),
            help="Type of quantizer to use")
        args.add_argument(
            '-ugc.qt.ql', '--ugc.quantizer.quantization-level',
            type=int, default=3,
            help="Quantization level for the quantizer")


def create_quantizer(cfg: Munch) -> Quantizer:
    return quantizers[cfg.ugc.quantizer.name](cfg)

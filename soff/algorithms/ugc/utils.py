import math
from typing import Callable
from typing import List
import numpy as np
from torch import nn
from munch import Munch
from ...compressors.topk import TopKPerLayer
from ...utils.arg_parser import Conf

# Map {model name -> (first x params, last x params)}
ugc_non_compress_params = {
    'resnet':  (1, 2),
    'resnet18': (1, 2),
    'resnet34': (1, 2),
    'resnet50': (1, 2),
    'resnet101': (1, 2),
    'resnet152': (1, 2),
    'resnet20': (1, 2),
    'resnet32': (1, 2),
    'resnet44': (1, 2),
    'resnet56': (1, 2),
    'resnet110': (1, 2),
    'resnet120': (1, 2),
    'vgg11': (2, 2),
    'vgg13': (2, 2),
    'vgg16': (2, 2),
    'vgg19': (2, 2),
    # TEST
    'language-modeling-lstm': (0, 0),
    'femnist-cnn': (0, 0),
}


def create_ugc_compressor(
        net: nn.Module, ratio: float, cfg: Munch) -> TopKPerLayer:

    if cfg.model.name not in ugc_non_compress_params:
        raise RuntimeError(f"Model {cfg.model.name} not supported.")

    num_params = range(len(list(net.parameters())))
    grad_topk_ratios = [ratio for _ in num_params]
    for i in range(ugc_non_compress_params[cfg.model.name][0]):
        grad_topk_ratios[i] = 1.
    for i in range(ugc_non_compress_params[cfg.model.name][1]):
        grad_topk_ratios[-1-i] = 1.

    return TopKPerLayer(Conf({
        'compression.topk_per_layer.ratios': grad_topk_ratios}))


def ugc_ae_param_idxs(net: nn.Module) -> List[int]:
    """Filter network parameter indices for ugc's autoencoder training"""
    param_names = list(name for name, _ in net.named_parameters())
    return [
        i for i, name in enumerate(param_names) if 'weight' in name
        and 'batchnorm' not in name
        and 'shortcut' not in name
        and i not in {0, len(param_names) - 1, len(param_names) - 2}
    ]


def fit_sparse_ratio_exp(
        x1: int, x2: int, y1: float, y2: float) -> Callable[[int], float]:
    assert x1 < x2 and y1 > y2
    b = math.pow(y1/y2, 1/(x2-x1))
    a = y1 / (b ** (-x1))

    def func(x):
        return a * (b ** (-x))
    return func


def fit_sparse_ratio_lin(
        x1: int, x2: int, y1: float, y2: float) -> Callable[[int], float]:
    assert x1 < x2 and y1 > y2
    a = (y1-y2)/(x1-x2)
    b = a * x1 - y1
    def func(x):
        return a*x + b
    return func

def fit_sparse_ratio_con(y:float) -> Callable[[int], float]:
    def func(_):
        return y
    return func


def golomb_idx_size(cfg: Munch, net: nn.Module, sparse_rate: float) -> float:
    # for sparsification ratio >5, sending full graident is more efficient
    if sparse_rate >= 0.5:
        return 0

    layer_num_elems = [param.numel() for param in net.parameters()]
    start = ugc_non_compress_params[cfg.model.name][0]
    end = ugc_non_compress_params[cfg.model.name][0]
    num = sum(layer_num_elems[start:-end])

    M = round(-1 / np.log2(1 - sparse_rate))
    b = round(np.log2(M))
    estimated_compression_ratio = sparse_rate / 2 * \
        (b + 1 / (1 - np.power(1 - sparse_rate / 2, 2**b)))
    compressed_size = num * 2 * estimated_compression_ratio / \
        32 + 1  # The unit is float, not bit
    return compressed_size

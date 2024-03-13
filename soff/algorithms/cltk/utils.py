from torch import nn
from munch import Munch
from ...utils.arg_parser import Conf
from ...compressors.topk import TopKPerLayer


def create_cltk_compressor(
        net: nn.Module, ratio: float, cfg: Munch) -> TopKPerLayer:

    # Map {model name -> (first x params, last x params)}
    cltk_non_compress_params = {
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
    }
    if cfg.model.name not in cltk_non_compress_params:
        raise RuntimeError(f"Model {cfg.model.name} not supported.")

    num_params = range(len(list(net.parameters())))
    grad_topk_ratios = [ratio for _ in num_params]
    for i in range(cltk_non_compress_params[cfg.model.name][0]):
        grad_topk_ratios[i] = 1.
    for i in range(cltk_non_compress_params[cfg.model.name][1]):
        grad_topk_ratios[-1-i] = 1.

    return TopKPerLayer(Conf({
        'compression.topk_per_layer.ratios': grad_topk_ratios}))

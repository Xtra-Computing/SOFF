"""Compressor configs and creator"""
import copy
from itertools import chain
from typing import Dict, Type
from munch import Munch
from .base import _ModelCompressor
from ..utils.arg_parser import BaseConfParser, r_getattr
from ..utils.module import (
    camel_to_snake, load_all_direct_classes, load_all_submodules)


compressor_name_map: Dict[str, Type[_ModelCompressor]] = {
    camel_to_snake(name): cls
    for module in load_all_submodules()
    for name, cls in load_all_direct_classes(module).items()
    if issubclass(cls, _ModelCompressor) and not name.startswith('_')
}


class CompressionConfParser(BaseConfParser):
    """Parse arguments for gradient/model compression"""

    def __init__(self, *args, stag='', ltag='', **kwargs):
        super().__init__(*args, **kwargs)

        # Client-server shared Compression-related args
        compress_args = self.add_argument_group(
            "Compression-related Configs (S,S->C)")

        compress_args.add_argument(
            f'-cp{stag}.c', f'--compression{ltag}.compressor',
            default='no_compress', type=str, metavar='NAME',
            choices=list(compressor_name_map.keys()),
            help="Compressor for compressing model."
            f"Available compressors {list(compressor_name_map.keys())}")

        for _, cls in chain.from_iterable(
                load_all_direct_classes(module).values()
                for module in load_all_submodules()):
            if not hasattr(cls, 'argparse_options'):
                continue
            arg_parser = compress_args if cls == _ModelCompressor else self
            cls.add_options_to(
                arg_parser, pfxs=('cp', 'compression'), tags=(stag, ltag))
            for option in cls.argparse_options():
                self.register_cfg_dep(
                    f'compress{ltag}.{option.flags[1]}',
                    lambda cfg, cls=cls: any(
                        issubclass(compressor_name_map[name], cls)
                        for name in r_getattr(
                            cfg, f"compression{ltag.replace('/','.')}.compressor")))


def create_compressor(cfg: Munch, tag: str = '') -> _ModelCompressor:
    cfg = copy.deepcopy(cfg)
    cfg.compression = cfg.compression[tag] if tag else cfg.compression
    return compressor_name_map[cfg.compression.compressor](cfg)

"""
Compose Raw to produce FLSplits for federated learning
"""
import copy
from itertools import chain
from typing import List, Optional
from munch import Munch
from .base import _FLSplit
from ..raw_dataset.base import _RawDataset
from ...utils.arg_parser import FLConfParser, r_getattr
from ...utils.module import (
    camel_to_snake, load_all_direct_classes, load_all_submodules)

data_splitting_name_to_class = {
    camel_to_snake(name[:-len('Split')] if name.endswith('Split') else name).replace('_', '-'): cls
    for module in load_all_submodules()
    for name, cls in load_all_direct_classes(module).items()
    if not name.startswith('_') and issubclass(cls, _FLSplit)
}


class FLSplitConfParser(FLConfParser):
    """Parse config for FL datasets"""

    def __init__(self, *args, stag='', ltag='', **kwargs):
        super().__init__(*args, **kwargs)

        fl_data_args = self.add_argument_group(
            "FL Data Split Configs (S,C)")
        fl_data_args.add_argument(
            f'-dt{stag}.fs.m', f'--data{ltag}.fl-split.method',
            default='iid', type=str,
            metavar='METHOD', choices=data_splitting_name_to_class.keys(),
            help="Method to split/compse the datasets. If multiple datasets "
            "are selected, then those datasets are first combined into a large "
            "dataset before splitting. Exceptions are the 'realistic' and "
            "'realistic-subsample' method, which is only applicable to "
            "non-synthetic datset, and if it is selected, the number of node "
            "that uses the datasets must match the number of datasets. "
            f"Available methods: {list(data_splitting_name_to_class.keys())}")

        for cls in chain.from_iterable(
                load_all_direct_classes(module).values()
                for module in load_all_submodules()):
            if not hasattr(cls, 'argparse_options'):
                continue

            arg_parser = fl_data_args if cls == _FLSplit else self
            cls.add_options_to(
                arg_parser, pfxs=('dt', 'data'),
                tags=(stag, ltag), ifxs=('fs', 'fl-split'))

            for option in cls.argparse_options():
                self.register_cfg_dep(
                    f'data{ltag}.fl_split.{option.flags[1]}',
                    lambda cfg, cls=cls: issubclass(
                        data_splitting_name_to_class[r_getattr(
                            cfg, f"data{ltag.replace('/','.')}.fl_split.method"
                        )], cls))


def create_fl_split(
        cfg: Munch, datasets: List[_RawDataset],
        mode, split_id: Optional[int] = None, tag: str = '') -> _FLSplit:
    """
    Create a FL dataset from existing datasets
    Args:
        cfg: The config object
        datasets: A list of Raws
        mode: 'train', 'test' or 'eval'
        split_id: int, ranging from 0 to (data.fl_split.num - 1) or None
    """
    cfg = copy.deepcopy(cfg)
    cfg.data = cfg.data[tag] if tag else cfg.data
    return data_splitting_name_to_class[cfg.data.fl_split.method](
        cfg, datasets=datasets, mode=mode, split_id=split_id)

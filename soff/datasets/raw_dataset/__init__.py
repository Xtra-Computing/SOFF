"""Interface file to raw datasets"""
import copy
from typing import Optional
from itertools import chain
from munch import Munch
from .base import _RawDataset, _CachedDataset
from ...utils.arg_parser import FLConfParser, r_getattr
from ...utils.module import (
    camel_to_snake, load_all_direct_classes, load_all_submodules)


dataset_name_to_class = {
    camel_to_snake(name).replace('_', '-'): cls
    for module in load_all_submodules()
    for name, cls in load_all_direct_classes(module).items()
    if not name.startswith('_') and issubclass(cls, _RawDataset)
}


class RawDataConfParser(FLConfParser):
    """Parse data/dataset-related arguments, including dataset-specific args"""

    def __init__(self, *args, stag='', ltag='', **kwargs):
        super().__init__(*args, **kwargs)
        data_args = self.add_argument_group("Dataset Configs")

        data_args.add_argument(
            f'-dt{stag}.r.d', f'--data{ltag}.raw.datasets',
            default=['cifar10'], nargs='+',
            metavar='NAME', choices=dataset_name_to_class.keys(),
            help="A list of datasets. "
            f"Available datasets: {list(dataset_name_to_class.keys())}")

        for cls in chain.from_iterable(
                load_all_direct_classes(module).values()
                for module in load_all_submodules()):
            if not hasattr(cls, 'argparse_options'):
                continue

            arg_parser = data_args if cls == _CachedDataset else self
            cls.add_options_to(
                arg_parser, pfxs=('dt', 'data'),
                tags=(stag, ltag), ifxs=('r', 'raw'))

            for option in cls.argparse_options():
                self.register_cfg_dep(
                    f'data{ltag}.raw.{option.flags[1]}',
                    lambda cfg, cls=cls: any(
                        issubclass(dataset_name_to_class[ds_name], cls)
                        for ds_name in r_getattr(
                            cfg, f"data{ltag.replace('/','.')}.raw.datasets")))


def create_raw_dataset(
        cfg: Munch, name: str, mode,
        split_id: Optional[int] = None, tag: str = '') -> _RawDataset:
    """
    Create a FL dataset from args
    Args:
        cfg: The config object
        name: Name of the dataset
        mode: 'train', 'test' or 'eval'
        split_id: int, ranging from 0 to (num_splits - 1), or None
    """
    # Remove tags
    cfg = copy.deepcopy(cfg)
    cfg.data = cfg.data[tag] if tag else cfg.data
    return dataset_name_to_class[name](cfg, mode=mode, split_id=split_id)

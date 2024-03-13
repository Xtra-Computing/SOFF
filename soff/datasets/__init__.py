"""Interface file to datset creation"""

from typing import List, Optional
from munch import Munch
from .fl_split import FLSplitConfParser, create_fl_split
from .fl_split.base import _FLSplit
from .raw_dataset import RawDataConfParser, create_raw_dataset


class DataConfParser(FLSplitConfParser, RawDataConfParser):
    """Parse data/dataset-related arguments, including dataset-specific args"""


def create_dataset(
        cfg: Munch, datasets: List[str],
        mode, split_id: Optional[int] = None, tag: str = '') -> _FLSplit:
    """
    Create a FL dataset from args
    Args:
        cfg: The config object
        datasets: A list of dataset names such as "img:cifar10"
        mode: 'train', 'test' or 'eval'
        split_id: int, ranging from 0 to (num_splits - 1), or None
    """
    # `mode` and `split_id` are per-dataset information,
    # thus we need to pass them separately from `cfg`
    return create_fl_split(cfg, datasets=[
        create_raw_dataset(cfg, name, mode=mode, split_id=split_id, tag=tag)
        for name in datasets
    ], mode=mode, split_id=split_id, tag=tag)

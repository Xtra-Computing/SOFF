"""
FL Splits for vertical FL
For vertical FL, datasets are specified on a per-client basis,
thus not much index manipulation is required.
"""

import random
from itertools import repeat
from typing import List, final
from .base import _VerticalFLSplit
from ..raw_dataset.base import _RawDataset
from ...utils.arg_parser import ArgParseOption, options


@final
class VerticalSplit(_VerticalFLSplit):
    """Base class of all realistic datasets"""

    def __init__(self, cfg, datasets: List[_RawDataset], mode, split_id):
        """split_id is ignored for vertical dataset"""
        super().__init__(cfg, datasets, mode, 0)
        assert split_id is None
        self.preproces_and_cache()

    def _load_desc(self):
        if self.mode == 'train':
            return list(zip(self.datasets[0].load_train_descs(), repeat(0)))
        method = f"load_{self.mode}_descs"
        return list(zip(getattr(self.datasets[0], method)(), repeat(0)))

    def _preprocess_train_descs(self, descs):
        return descs


@final
@options(
    "VFLSubsampleSplit Options",
    ArgParseOption(
        'v.sr', 'vfl.sample-ratio',
        default=0.1, type=float, metavar='RATIO',
        help="Data sample ratio, valid for iid-subsample datasets"))
class VerticalSubsampleSplit(_VerticalFLSplit):
    """Unifromly subsample from each client's data."""

    def __init__(self, cfg, *args, **kwargs):
        assert 0. < cfg.data.fl_split.vfl.sample_ratio <= 1.
        super().__init__(cfg, *args, **kwargs)
        self.data_sample_ratio = cfg.data.fl_split.vfl.sample_ratio
        self.preproces_and_cache()

    def metadata(self):
        return {
            **super().metadata(),
            'data_sample_ratio': self.data_sample_ratio
        }

    def _load_desc(self):
        if self.mode == 'train':
            return list(zip(self.datasets[0].load_train_descs(), repeat(0)))
        method = f"load_{self.mode}_descs"
        return list(zip(getattr(self.datasets[0], method)(), repeat(0)))

    def _preprocess_train_descs(self, descs):
        number_samples = int(len(descs) * self.data_sample_ratio)
        random.Random(self.seed).shuffle(descs)
        return descs[:number_samples + 1]

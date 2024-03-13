"""Realistic splitting"""
import random
from itertools import chain, repeat
from typing import List, final
from ..raw_dataset.base import _RawDataset
from .base import _HorizontalFLSplit


class _BaseRealisticSplit(_HorizontalFLSplit):
    """Base class of all realistic datasets"""

    def __init__(self, cfg, datasets: List[_RawDataset], mode, split_id):
        """If mode is not 'train', num_splits and split_id are ignored"""

        assert mode != 'train' or cfg.data.fl_split.number == len(datasets)
        assert split_id is None or 0 <= split_id < len(datasets)

        super().__init__(cfg, datasets, mode, split_id)

        self.split_id = split_id

    def _load_desc(self):
        if self.mode == 'train':
            assert self.split_id is not None
            return list(zip(
                self.datasets[self.split_id].load_train_descs(),
                repeat(self.split_id)))

        method = f"load_{self.mode}_descs"
        return list(chain(*[
            zip(getattr(dataset, method)(), repeat(idx))
            for idx, dataset in enumerate(self.datasets)]))


@final
class RealisticSplit(_BaseRealisticSplit):
    """Multiple datasets are used as-is. No combination and re-splitting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preproces_and_cache()

    def _preprocess_train_descs(self, descs):
        return descs


@final
class RealisticSubsampleSplit(_BaseRealisticSplit):
    """Unifromly subsample from each client's data."""

    def __init__(self, cfg, *args, **kwargs):
        assert 0. < cfg.data.fl_split.hfl.sample_ratio <= 1.
        super().__init__(cfg, *args, **kwargs)
        self.data_sample_ratio = cfg.data.fl_split.hfl.sample_ratio
        self.preproces_and_cache()

    def metadata(self):
        return {
            **super().metadata(),
            'data_sample_ratio': self.data_sample_ratio
        }

    def _preprocess_train_descs(self, descs):
        number_samples = int(len(descs) * self.data_sample_ratio)
        random.Random(self.seed).shuffle(descs)
        return descs[:number_samples + 1]


@final
class RealisticIidSubsampleSplit(_BaseRealisticSplit):
    """
    Unifromly subsample from all data, and applies the subsample result to
    all clients, so that they have (roughly) the same label distribution
    """

    def __init__(self, cfg, *args, **kwargs):
        assert 0. < cfg.data.fl_split.hfl.sample_ratio <= 1.
        super().__init__(cfg, *args, **kwargs)
        self.data_sample_ratio = cfg.data.fl_split.hfl.sample_ratio
        self.preproces_and_cache()

    def metadata(self):
        return {
            **super().metadata(),
            'data_sample_ratio': self.data_sample_ratio
        }

    def _preprocess_train_descs(self, descs):
        # Unifromly sample across all datas
        data_idxes = [
            (ds_idx, sample_idx)
            for ds_idx in range(len(self.datasets))
            for sample_idx in range(len(
                self.datasets[ds_idx].load_train_descs()))]
        data_idxes = random.Random(self.seed).sample(
            data_idxes, int(self.data_sample_ratio * len(data_idxes)))

        # Filter out data for this client
        sample_idxes = set(
            sample_idx for ds_idx, sample_idx in data_idxes
            if ds_idx == self.split_id)

        return [descs[i] for i in sample_idxes]

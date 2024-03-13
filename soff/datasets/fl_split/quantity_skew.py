"""Quantity-skewed splitting"""
import logging
import random
from typing import List, final
from abc import ABC, abstractmethod
import numpy as np
from munch import Munch
from scipy.stats import powerlaw
from .base import _SyntheticSplit

log = logging.getLogger(__name__)


class _QuantitySkewSplit(_SyntheticSplit, ABC):
    """Datasets that only skews in the quantity dimension"""

    def metadata(self):
        return {**super().metadata(), 'pdf': self.pdf()}

    def _preprocess_train_descs(self, descs):
        assert self.spl_id is not None, \
            "Training dataset preprocessing depends on spl_id"

        pdf = self.pdf()
        assert (0 < sum(pdf) < (1. + 1e-5))
        if sum(pdf) < (1. - 1e-5):
            log.warning(
                "%s: sum(pdf) < 1, dataset not fully utilized",
                self.__class__.__name__)
        cdf = list(np.cumsum(pdf))

        # group sample indices by label
        class_idx_map = {cls: [] for cls in self.classes}
        idxs = list(range(len(descs)))
        random.Random(self.seed).shuffle(idxs)
        for idx in idxs:
            sample, ds_idx = descs[idx]
            class_idx_map[self.datasets[ds_idx].get_label(sample)].append(idx)

        # i.i.d splitting, and fetch the result for this client only
        split_idx_map = []
        for cls in self.classes:
            begin = 0 if self.spl_id == 0 else \
                round(len(class_idx_map[cls]) * cdf[self.spl_id - 1])
            end = round(len(class_idx_map[cls]) * cdf[self.spl_id])
            split_idx_map.extend(class_idx_map[cls][begin:end])

        random.Random(self.seed).shuffle(split_idx_map)
        return [descs[i] for i in split_idx_map]

    @abstractmethod
    def pdf(self) -> List[float]:
        """Should return the ratio of data samples w.r.t clients.
        Impelementation must be deterministic with given self.seed"""
        raise NotImplementedError


@final
class IidSplit(_QuantitySkewSplit):
    """All clients evenly sample without duplication from the dataset"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preproces_and_cache()

    def pdf(self):
        return [1. / self.num_splits] * self.num_splits


@final
class IidSubsampleSplit(_QuantitySkewSplit):
    """All clients evenly sample without duplication from the dataset"""

    def __init__(self, cfg: Munch, *args, **kwargs):
        """if mode is eval or test, num_splits and split_id are ignored"""
        # Process the dataset if needs processing
        assert 0. < cfg.data.fl_split.hfl.sample_ratio <= 1.
        super().__init__(cfg, *args, **kwargs)
        self.data_sample_ratio = cfg.data.fl_split.hfl.sample_ratio
        self.preproces_and_cache()

    def pdf(self):
        return [
            (1. * self.data_sample_ratio) / self.num_splits
        ] * self.num_splits


@final
class DirichletQuantitySkewSplit(_QuantitySkewSplit):
    """The clients' dataset sizes follow a Dirichlet distribution"""

    def __init__(self, cfg, *args, **kwargs):
        """alpha: concentration level"""
        super().__init__(*args, **kwargs)
        self.alpha = cfg.data.fl_split.hfl.alpha
        self.preproces_and_cache()

    def metadata(self):
        return {**super().metadata(), 'alpha': self.alpha}

    def pdf(self):
        np.random.seed(self.seed)
        return np.random.dirichlet([self.alpha] * self.num_splits).tolist()


@final
class PowerLawQuantitySkewSplit(_QuantitySkewSplit):
    """The clients' dataset sizes follow a power-law distribution"""

    def __init__(self, cfg, *args, **kwargs):
        """alpha: power distribution: alpha * x ^ {alpha - 1}, x: client id"""
        super().__init__(cfg, *args, **kwargs)
        self.alpha = cfg.data.fl_split.hfl.alpha
        self.preproces_and_cache()

    def metadata(self):
        return {**super().metadata(), 'alpha': self.alpha}

    def pdf(self):
        dist = powerlaw.pdf(
            np.arange(1, 1+self.num_splits) / float(self.num_splits),
            self.alpha)
        dist /= sum(dist)
        return dist.tolist()

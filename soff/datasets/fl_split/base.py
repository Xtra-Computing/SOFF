"""Base class of fl datasets"""
import pickle
import logging
import hashlib
import pathlib
from itertools import repeat, chain
from abc import ABC, abstractmethod
from typing import List, Sequence, Any, Tuple, Dict, Optional, Set
import numpy as np
from munch import Munch
from torch.utils.data import Dataset
from ..raw_dataset.base import _RawDataset
from ..utils import save_dict, metadata_updated, save_obj, load_obj
from ...utils.arg_parser import ArgParseOption, options, require

log = logging.getLogger(__name__)


@require('federation.seed', 'data.fl_split.num')
@options(
    ArgParseOption(
        'n', 'num', default=4, type=int, required=True,
        help="Number of data splits."))
class _FLSplit(Dataset, ABC):
    """
    Base class of all FL datasets.
    FL Datasets are composed from multiple _RawDatasets. Subclasses define how
    they're composed.

    Args:
        datasets: a list of _RawDatasets to combine
        mode: train | test | eval
        split_id: 0~(num_splits-1) or `None` if this dataset is for server
    """

    def __init__(self, cfg: Munch, datasets: Sequence[_RawDataset],
                 mode, split_id: Optional[int]):
        self.root = pathlib.Path("data/fl")

        self.mode = mode
        self.seed = cfg.federation.seed
        self.num_splits = cfg.data.fl_split.num
        self.spl_id = split_id
        self.datasets = datasets
        """The composed datapoint descriptor list"""

        assert split_id is None or 0 <= split_id < self.num_splits

        self.cache_dir: pathlib.Path = pathlib.Path("")
        self.meta: Dict[str, Any] = {}
        self.meta_cache_path: pathlib.Path = pathlib.Path("")

        self.classes: Set[Any] = set()
        """Number of classes. override computed if specified by raw datasets"""
        self.desc: List[Tuple[Any, int]] = []
        """A list of (datapoint descriptor, dataset idx) tuples"""

    def preproces_and_cache(self):
        """
        Load descs from underlying raw datasets, preproces the descs and cache
        them along with metadatas.
        """
        cname = self.__class__.__name__
        raw_descs = self._load_desc()

        self.classes = set(getattr(self.datasets[0], 'classes')) \
            if (all(hasattr(d, 'classes') for d in self.datasets) and
                all(getattr(d, 'classes') ==
                    getattr(self.datasets[0], 'classes')
                    for d in self.datasets)) \
            else set(self.datasets[idx].get_label(smpl)
                     for smpl, idx in raw_descs)

        self.meta = self.metadata()

        # cache_dir dpends on meta
        self.cache_dir = self.root.joinpath(f"{self._get_cache_dir()}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir.joinpath(
            f"{self._get_cache_file_stem()}.pkl")
        self.meta_cache_path = self.cache_dir.joinpath(
            f"{self._get_cache_file_stem()}_meta.pkl")

        log.info("%s: seed: %s, cache path: %s", cname, self.seed, cache_path)

        if self._re_preprocess():
            log.info("%s: Cache invalid. re-processing", cname)
            self.desc = self._preprocess_train_descs(raw_descs) \
                if self.mode == 'train' else raw_descs
            save_obj(self.desc, cache_path)
        else:
            self.desc = load_obj(cache_path)

        if metadata_updated(self.meta, self.meta_cache_path):
            save_dict(self.meta, self.meta_cache_path)

    def metadata(self) -> Dict[str, Any]:
        """
        Return a dict of metadatas to be cached.
        This method can be overriden if subclass has additional metadatas. e.g.
        `return {**super().metadata(), 'key': <value>, ...}`
        """
        return {
            'seed': self.seed,
            'mode': self.mode,
            'num_splits': self.num_splits,
            'split_id': self.spl_id,
            'datasets': [ds.__class__.__name__ for ds in self.datasets]
        }

    def _get_cache_dir(self) -> str:
        """
        Name of the cache dir is a hash that depends on:

            * The class name of FLSplit subclass
            * The metadatas used by the FLSplit subclass (excluding `mode`
              and `split_id`, so that relevant datas are saved in one place)
            * The name and number of underlying datasets

        Each hash dir contains a session, i.e. the train/test datasets for
        each clients under the same dataset composition config.
        """
        name_hash = hashlib.md5()
        name_hash.update(self.__class__.__name__.encode())
        name_hash.update(';'.join(
            f"{ds.__module__.split('.')[-1]}.{ds.__class__.__name__}"
            for ds in self.datasets).encode())
        name_hash.update(pickle.dumps(
            {k: v for k, v in self.metadata().items()
                if k not in {'mode', 'split_id'}}))
        return f"{name_hash.hexdigest()}"

    def _get_cache_file_stem(self):
        return (f"data_{self.mode}_{self.spl_id}"
                if self.spl_id is not None else f"data_{self.mode}")

    def _re_preprocess(self) -> bool:
        return any(d.dataset_updated() for d in self.datasets) or \
            metadata_updated(self.meta, self.meta_cache_path)

    @abstractmethod
    def _preprocess_train_descs(self, descs: List[Tuple[Any, int]]) \
            -> List[Tuple[Any, int]]:
        """Subclass should override this to implement preprocessing"""
        raise NotImplementedError

    @abstractmethod
    def _load_desc(self) \
            -> List[Tuple[Any, int]]:
        """
        Child class should implement this method.
        Should return: [(sample, index of `datasets`) ... ]
        """
        raise NotImplementedError

    def num_classes(self):
        """Get number of classes (unique labels)"""
        assert len(self.classes) > 0
        return len(self.classes)

    def histogram(self):
        """Get the labels' histogram"""
        hist: Dict[Any, int] = dict(zip(*np.unique(
            [self.datasets[ds_idx].get_label(sample)
             for sample, ds_idx in self.desc],
            return_counts=True)))
        return [(int(hist[cls]) if cls in hist else 0) for cls in self.classes]

    def __getitem__(self, idx):
        """return: feature, label"""
        sample, ds_idx = self.desc[idx]
        return self.datasets[ds_idx].get_data(sample), \
            self.datasets[ds_idx].get_label(sample)

    def __len__(self):
        return len(self.desc)

    def __str__(self):
        counter = {}
        for cls in self.classes:
            counter[cls] = 0
        for desc, ds_idx in self.desc:
            counter[self.datasets[ds_idx].get_label(desc)] += 1
        return f"Client id: {self.spl_id}\n" \
            f"  # entries: {len(self.desc)}\n" \
            f"  label dist: {str(counter)}"


@options(
    "Horizontal FL Split Configs (S,C)",
    ArgParseOption(
        'h.sr', 'hfl.sample-ratio',
        default=0.2, type=float, metavar='RATIO',
        help="Data sample ratio, valid for iid-subsample datasets"),
    ArgParseOption(
        'h.al', 'hfl.alpha',
        default=0.5, type=float, metavar='ALPHA',
        help="Alpha value for skew partitioning"))
class _HorizontalFLSplit(_FLSplit):
    def __init__(
            self, cfg: Munch, datasets: Sequence[_RawDataset],
            mode, split_id: Optional[int]):
        super().__init__(cfg, datasets, mode, split_id)
        assert split_id is None or 0 <= split_id < cfg.data.fl_split.num


class _VerticalFLSplit(_FLSplit):
    def __init__(
            self, cfg: Munch, datasets: Sequence[_RawDataset],
            mode, split_id: Optional[int]):
        super().__init__(cfg, datasets, mode, split_id)

class _SyntheticSplit(_HorizontalFLSplit):
    def _load_desc(self):
        method = f"load_{self.mode}_descs"
        return list(chain(*[
            zip(getattr(dataset, method)(), repeat(idx))
            for idx, dataset in enumerate(self.datasets)]))

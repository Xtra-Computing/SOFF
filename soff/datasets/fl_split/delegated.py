import logging
from itertools import repeat
from typing import final, Sequence, Optional
from munch import Munch

from ..raw_dataset.base import _RawDataset
from .base import _HorizontalFLSplit

log = logging.getLogger(__name__)


@final
class DelegatedSplit(_HorizontalFLSplit):
    """
    Delegate the dataset splitting process to the single underlying dataset.
    Similar to _RealisticSplit but requires only one underlying raw dataset.
    Used when post-processing after splitting a single dataset is needed.
    Allows more flexible data splitting

    When implementing a raw dataset class intended to be used with this class,
    the raw dataset class should process num_splits and split_id and return
    different descriptor lists for each party when load_*_descs is called.
    """

    def __init__(
            self, cfg: Munch, datasets: Sequence[_RawDataset],
            mode, split_id: Optional[int]):

        assert len(datasets) == 1, \
            "DeletagedSplit no more than one underlying raw dataset"

        super().__init__(cfg, datasets, mode, split_id)
        self.split_id = split_id
        self.preproces_and_cache()

    def preproces_and_cache(self):
        # We do not need cache mechanism in delegated split since everything
        # is delegated to underlying classes.
        self.desc = self._load_desc()
        self.classes = set(getattr(self.datasets[0], 'classes')) \
            if (all(hasattr(d, 'classes') for d in self.datasets) and
                all(getattr(d, 'classes') ==
                    getattr(self.datasets[0], 'classes')
                    for d in self.datasets)) \
            else set(self.datasets[idx].get_label(smpl)
                     for smpl, idx in self.desc)

    def _load_desc(self):
        method = f"load_{self.mode}_descs"
        return list(zip(
            getattr(self.datasets[0], method)(), repeat(0)))

    def _preprocess_train_descs(self, descs):
        return descs

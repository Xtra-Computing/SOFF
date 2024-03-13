"""Label-skewed splitting"""
import random
from abc import ABC, abstractmethod
from typing import List, Any, Dict, final
import numpy as np
from .base import _SyntheticSplit


class _LabelSkewSplit(_SyntheticSplit, ABC):
    """Splits skews in both quantity and distribution dimension"""

    def metadata(self):
        return {**super().metadata(), 'pdfs': self.pdfs()}

    def _preprocess_train_descs(self, descs):
        assert self.spl_id is not None, \
            "Training dataset preprocessing depends on spl_id"

        pdfs = self.pdfs()
        cdfs = {key: list(np.cumsum(val)) for key, val in pdfs.items()}

        # group sample indices by label
        class_idx_map = {}
        idxs = list(range(len(descs)))
        random.Random(self.seed).shuffle(idxs)
        for cls in self.classes:
            class_idx_map[cls] = []
        for idx in idxs:
            sample, ds_idx = descs[idx]
            class_idx_map[self.datasets[ds_idx].get_label(sample)].append(idx)

        # i.i.d splitting, and fetch the result for this client only
        split_idx_map = []
        for cls in self.classes:
            begin = 0 if self.spl_id == 0 else \
                round(len(class_idx_map[cls]) *
                      cdfs[cls][self.spl_id - 1])
            end = round(len(class_idx_map[cls]) * cdfs[cls][self.spl_id])
            split_idx_map.extend(class_idx_map[cls][begin:end])

        random.Random(self.seed).shuffle(split_idx_map)
        return [descs[i] for i in split_idx_map]

    @abstractmethod
    def pdfs(self) -> Dict[Any, List[float]]:
        """returns dict, where the key is class (labels),
        and the value is pdf for each parties"""
        raise NotImplementedError


@final
class DirichletLabelSkewSplit(_LabelSkewSplit):
    """The clients' dataset labels follow Dirichlet distributions"""

    def __init__(self, cfg, *args, **kwargs):
        """alpha: concentration level"""
        super().__init__(cfg, *args, **kwargs)
        self.alpha = cfg.data.fl_split.hfl.alpha
        self.preproces_and_cache()

    def metadata(self):
        return {**super().metadata(), 'alpha': self.alpha}

    def pdfs(self):
        np.random.seed(self.seed)
        return {
            cls: np.random.dirichlet([self.alpha] * self.num_splits).tolist()
            for cls in range(len(self.classes))
        }

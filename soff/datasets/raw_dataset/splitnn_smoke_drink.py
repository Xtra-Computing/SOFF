"""
https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset/
"""
import copy
from itertools import cycle
import torch
import numpy as np
import pandas as pd
from .base import _RawDataset
from ..utils import file_exists


class _SplitnnSmokingDrinkingDataset(_RawDataset):
    _root = 'data/src/splitnn_smoke_drink'

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.cache_path = f"{self.root}/{self.__class__.__name__}.npy"
        self.preprocess_and_cache_meta()
        self.dataset: np.ndarray = np.load(self.cache_path)

    def load_sample_descriptors(self):
        return list(range(len(self.dataset)))

    def re_preprocess(self):
        return not file_exists(self.meta_cache_path)


class SplitnnSmokingDrinkingLabels(_SplitnnSmokingDrinkingDataset):
    def preprocess(self):
        dframe = pd.read_csv(f"{self.root}/smoking_driking_dataset_Ver01.csv")
        dframe = dframe['DRK_YN'] == 'Y'
        dataset = dframe.to_numpy().astype(np.float32)
        np.save(self.cache_path, dataset)

    def get_data(self, _):
        # Initiator dataset don't require features
        return torch.Tensor()

    def get_label(self, desc):
        return self.dataset[desc]


class _SplitnnSmokingDrinkingFeatures(_SplitnnSmokingDrinkingDataset):
    def get_data(self, desc):
        # drop id when getting features
        return torch.Tensor(self.dataset[desc])

    def get_label(self, _):
        # Collaborator dataset doesn't require labels
        return torch.Tensor()


class SplitnnSmokingDrinkingFeatures1(_SplitnnSmokingDrinkingFeatures):
    def preprocess(self):
        dframe = pd.read_csv(f"{self.root}/smoking_driking_dataset_Ver01.csv")
        dframe = dframe.iloc[:, :7]
        dframe.replace({'sex': {'Male': 0., 'Female': 1.}}, inplace=True)
        dataset = dframe.to_numpy()
        np.save(self.cache_path, dataset)


class SplitnnSmokingDrinkingFeatures2(_SplitnnSmokingDrinkingFeatures):
    def preprocess(self):
        dframe = pd.read_csv(f"{self.root}/smoking_driking_dataset_Ver01.csv")
        dframe = dframe.iloc[:, 7:].drop('DRK_YN', axis=1)
        dataset = dframe.to_numpy()
        np.save(self.cache_path, dataset)

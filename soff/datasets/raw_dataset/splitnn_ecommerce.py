"""
https://github.com/Elliezza/Time_Series_FE
"""
from itertools import cycle
import torch
import numpy as np
import pandas as pd
from .base import _RawDataset
from ..utils import metadata_updated
from ...utils.arg_parser import ArgParseOption, options


@options(
    "SplitNN-ECommerce Dataset Options",
    ArgParseOption(
        'ecmm.os', 'e-commerce.original-split', action='store_true',
        help="Use original data split instead of re-splitting the data"),
    ArgParseOption(
        'ecmm.st', 'e-commerce.setting',
        type=str, default='B', choices=['A', 'B'],
        help="Vertical split setting")
)
class _SplitnnEcommerceDataset(_RawDataset):
    _root = 'data/src/splitnn_ecommerce'

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.setting = cfg.data.raw.e_commerce.setting
        self.original_split = cfg.data.raw.e_commerce.original_split
        self.cache_path = f"{self.root}/{self.__class__.__name__}_{self.setting}.npy"
        if self.original_split:
            self.load_train_descs = lambda: (
                self.load_sample_descriptors()[:4992911])
            self.load_eval_descs = lambda: []
            self.load_test_descs = lambda: (
                self.load_sample_descriptors()[4992911:])
        self.preprocess_and_cache_meta()
        self.dataset: np.ndarray = np.load(self.cache_path)

    def metadata(self):
        return {
            **super().metadata(),
            'original_split': self.original_split,
            'setting': self.setting
        }

    def load_sample_descriptors(self):
        return list(range(len(self.dataset)))

    def re_preprocess(self):
        return metadata_updated(
            self.meta, self.meta_cache_path, ['original_split', 'setting'])


class SplitnnEcommerceLabels(_SplitnnEcommerceDataset):
    def preprocess(self):
        dframe1 = pd.read_csv(
            f"{self.root}/setting_{self.setting}/train_initiator.csv")
        dframe2 = pd.read_csv(
            f"{self.root}/setting_{self.setting}/test_initiator.csv")
        dframe = pd.concat((dframe1, dframe2), ignore_index=True)['y']
        dataset = dframe.to_numpy().astype(np.float32)
        np.save(self.cache_path, dataset)

    def get_data(self, _):
        # Initiator dataset don't require features
        return torch.Tensor()

    def get_label(self, desc):
        return self.dataset[desc]


class _SplitnnEcommerceFeatures(_SplitnnEcommerceDataset):
    def get_data(self, desc):
        # drop id when getting features
        return torch.Tensor(self.dataset[desc])

    def get_label(self, _):
        # Collaborator dataset doesn't require labels
        return torch.Tensor()


class SplitnnEcommerceFeatures1(_SplitnnEcommerceFeatures):
    def preprocess(self):
        dframe1 = pd.read_csv(
            f"{self.root}/setting_{self.setting}/train_initiator.csv")
        dframe2 = pd.read_csv(
            f"{self.root}/setting_{self.setting}/test_initiator.csv")
        dframe = pd.concat((dframe1, dframe2), ignore_index=True)
        dframe = dframe.drop(['id', 'y'], axis=1)
        # Normalization
        dframe = dframe.astype(np.float32).apply(
            lambda x: (x-x.mean()) / x.std(), axis=0).fillna(0.)
        dataset = dframe.to_numpy()
        np.save(self.cache_path, dataset)


class SplitnnEcommerceFeatures2(_SplitnnEcommerceFeatures):
    def preprocess(self):
        dframe1 = pd.read_csv(
            f"{self.root}/setting_{self.setting}/train_collaborator.csv")
        dframe2 = pd.read_csv(
            f"{self.root}/setting_{self.setting}/test_collaborator.csv")
        dframe = pd.concat((dframe1, dframe2), ignore_index=True).drop(
            ['id'], axis=1)
        # Normalization
        dframe = dframe.astype(np.float32).apply(
            lambda x: (x-x.mean()) / x.std(), axis=0).fillna(0.)
        dataset = dframe.to_numpy()
        np.save(self.cache_path, dataset)

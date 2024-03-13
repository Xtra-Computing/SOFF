"""Wind Turbine data (binary classification)"""
from typing import Dict, List, Any
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from .base import _RawDataset
from ..utils import metadata_updated


class Turbine(_RawDataset):
    """This is a delegated dataset"""

    _root = 'data/src/turbine'

    def __init__(self, cfg, mode, split_id, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.cache_stem = f"{self.root}/{self.__class__.__name__}"
        self.preprocess_and_cache_meta()
        if mode == 'train':
            assert 0 <= split_id <= 32
            self.dataset = np.load(self.cache_stem + f'_{split_id:02d}.npy')
        elif mode == 'eval':
            self.dataset = np.load(self.cache_stem + '_eval.npy')
        elif mode == 'test':
            self.dataset = np.load(self.cache_stem + '_test.npy')
        else:
            raise RuntimeError(f"Unknown mode {mode}")

    def metadata(self) -> Dict[str, Any]:
        return {'seed': self.seed, 'train_eval_test': self.train_eval_test}

    def re_preprocess(self) -> bool:
        return metadata_updated(self.meta, self.meta_cache_path)

    def preprocess(self) -> None:
        actions = pd.read_parquet(self.root.as_posix() + '/action')
        feature = pd.read_parquet(self.root.as_posix() + '/flattenRequest')

        # Process special columns
        actions = actions['actionValue'].astype('float32')
        feature = feature.drop('reqId', axis=1)

        # Drop highly relevant columns
        feature = feature.drop([
            'state_new_2_3min',
            'state_new_4_3min',
            'state_new_2_5min',
            'state_new_4_5min',
            'state_new_2_3min_5min',
            'state_new_2_5min_10min',
            'state_new_2_10min_30min'
        ], axis=1)

        feature['eventTime'] = (
            feature['eventTime'] - feature['eventTime'].min()
        ).dt.total_seconds().astype('float32')
        feature['date'] = pd.to_datetime(feature['date'])
        feature['date'] = (
            feature['date'] - feature['date'].min()
        ).dt.total_seconds().astype('float32')

        # Convert all 'object' columns in feature to 'float32' columns
        for col in tqdm(feature.columns):
            if feature[col].dtype == 'object' and col != 'col_2':
                feature[col] = feature[col].replace('', 'NaN')
                feature[col] = feature[col].astype('float32')

        # Fill 0s to NaNs
        feature = feature.fillna(0)

        # Normalize all feature columns except col_2
        for col in tqdm(feature.columns):
            if col != 'col_2':
                feature[col] = (
                    feature[col] - feature[col].mean()
                ) / feature[col].std()

        # Fill 0s to NaNs. Some features have std of 0
        feature = feature.fillna(0)

        # Join two dataframes
        dataset = pd.concat([actions, feature], axis=1)

        # Split by col_2, sort by eventTime
        grouped = dataset.groupby('col_2')
        datasets = [
            grouped.get_group(x).sort_values(
                by='eventTime').drop('col_2', axis=1).to_numpy()
            for x in grouped.groups]

        for dataset in datasets:
            random.Random(self.seed).shuffle(dataset)

        train_, eval_, _ = self.train_eval_test
        for i, dataset in enumerate(datasets):
            np.save(self.cache_stem + f'_{i:02d}.npy', dataset[
                :int(len(dataset) * (train_ / sum(self.train_eval_test)))])
        np.save(self.cache_stem + '_eval.npy', np.concatenate([dataset[
            int(len(dataset) * (train_ / sum(self.train_eval_test))):
            int(len(dataset) * ((train_ + eval_) / sum(self.train_eval_test)))
        ] for dataset in datasets]))
        np.save(self.cache_stem + '_test.npy', np.concatenate([dataset[
            int(len(dataset) * ((train_ + eval_) / sum(self.train_eval_test))):
        ] for dataset in datasets]))

    def load_sample_descriptors(self) -> List[Any]:
        return list(range(len(self.dataset)))

    def get_data(self, desc) -> Any:
        return torch.Tensor(self.dataset[desc][1:])

    def get_label(self, desc) -> int:
        return self.dataset[desc][0]

    def load_train_descs(self) -> List[Any]:
        return super().load_train_descs()

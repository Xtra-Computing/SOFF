"""Agriculture Bank of China (binary classification)"""
import pickle
from typing import List, Any
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from .base import _RawDataset
from ..utils import file_exists
from ...utils.arg_parser import ArgParseOption, options


@options(
    "ABChina Dataset Configs",
    ArgParseOption(
        'abc.ysl', 'abchina.y-seq-len', default=543, type=int, metavar='LEN',
        help="Sequence lenth for feature y (bill detail)"),
    ArgParseOption(
        'abc.zsl', 'abchina.z-seq-len', default=10515, type=int, metavar='LEN',
        help="Sequence lenth for feature z (transaction detail)"))
class Abchina(_RawDataset):
    _root = 'data/src/abchina'

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.cache_path = f"{self.root}/{self.__class__.__name__}.pkl"
        self.preprocess_and_cache_meta()
        with open(self.cache_path, 'rb') as file:
            self.dataset = pickle.load(file)
        self.y_avg = np.average(np.concatenate(
            [d[1] for d in self.dataset]), axis=0)
        self.z_avg = np.average(np.concatenate(
            [d[2] for d in self.dataset]), axis=0)
        self.y_len = min(
            max(len(d[1]) for d in self.dataset),
            cfg.data.raw.abchina.z_seq_len)
        self.z_len = min(
            max(len(d[2]) for d in self.dataset),
            cfg.data.raw.abchina.z_seq_len)

    def re_preprocess(self) -> bool:
        return not file_exists(self.meta_cache_path)

    def preprocess(self) -> None:
        root = self.root.as_posix()

        # 44476 × 3 table -----------------------------------------------------
        actions = pd.read_parquet(f'{root}/action')
        feature = pd.read_parquet(f'{root}/flattenRequest')
        user_info = pd.read_parquet(f'{root}/bo_user')
        # concact actions and features
        feature = pd.concat([
            actions.drop(['reqId', 'eventTime'], axis=1),
            feature.drop(['reqId'], axis=1)
        ], axis=1)
        feature = feature.sort_values('new_user_id', ignore_index=True)
        user_info = user_info.sort_values('new_user_id', ignore_index=True)
        # concat feature and user_info
        feature = pd.concat([
            feature.drop(['new_user_id', 'ingestionTime'], axis=1),
            user_info.drop('ingestionTime', axis=1)
        ], axis=1)

        # Convert everything to float, and normalize
        feature['time1'] = pd.to_datetime(feature['time1'])
        feature['time1'] = (
            feature['time1'] - feature['time1'].min()
        ).dt.total_seconds().astype('float32')
        feature['eventTime'] = (
            feature['eventTime'] - feature['eventTime'].min()
        ).dt.total_seconds().astype('float32')

        for col in feature.columns:
            if col == 'new_user_id':
                continue
            if feature[col].dtype != 'float32':
                feature[col] = feature[col].astype('float32')
            if col == 'actionValue':
                continue
            feature[col] = (
                feature[col] - feature[col].mean()) / feature[col].std()

        # Aux tables ----------------------------------------------------------
        # 950122 'repay_status' is the label? linked by 'new_user_id'
        bill_detail = pd.read_parquet(f'{root}/bo_bill_detail')
        # 4453258 transaction details? linked by 'new_user_id'
        txn_detail = pd.read_parquet(f'{root}/bo_detail')
        # 16960884 browse history? linked by 'new_user_id'. Ignore this for now
        browse_history = pd.read_parquet(f'{root}/bo_browse_history')

        # Make all column in bill_detail floats
        bill_detail = bill_detail.drop(['ingestionTime', 'bill_ts'], axis=1)
        for col in bill_detail.columns:
            if col == 'new_user_id':
                continue
            bill_detail[col] = bill_detail[col].astype('float32')
            bill_detail[col] = (
                bill_detail[col] - bill_detail[col].mean()
            ) / bill_detail[col].std()

        # Make all column in txn_detail floats
        txn_detail = txn_detail.drop(['ingestionTime'], axis=1)
        for col in txn_detail.columns:
            if col == 'new_user_id':
                continue
            txn_detail[col] = txn_detail[col].astype('float32')
            txn_detail[col] = (
                txn_detail[col] - txn_detail[col].mean()
            ) / txn_detail[col].std()

        # Group and link by new_user_id ---------------------------------------
        dataset = []
        bill_detail_avg = bill_detail.drop(
            'new_user_id', axis=1).mean().to_numpy()
        txn_detail_avg = txn_detail.drop(
            'new_user_id', axis=1).mean().to_numpy()

        bill_detail_groups = bill_detail.groupby('new_user_id')
        txn_detail_groups = txn_detail.groupby('new_user_id')
        for feat, user_id in tqdm(zip(
                feature.drop('new_user_id', axis=1).to_numpy(),
                feature['new_user_id'].to_numpy())):
            try:
                bill = bill_detail_groups.get_group(
                    user_id).drop('new_user_id', axis=1).to_numpy()
            except KeyError:
                bill = np.empty((0, len(bill_detail_avg)), dtype='float32')

            try:
                txn = txn_detail_groups.get_group(
                    user_id).drop('new_user_id', axis=1).to_numpy()
            except KeyError:
                txn = np.empty((0, len(txn_detail_avg)), dtype='float32')

            dataset.append((feat, bill, txn))

        with open(self.cache_path, 'wb') as f:
            pickle.dump(dataset, f)

    def load_sample_descriptors(self) -> List[Any]:
        return list(range(len(self.dataset)))

    def get_label(self, desc) -> int:
        return self.dataset[desc][0][0]

    def get_data(self, desc) -> Any:
        # Pre-pad with avg value
        return (
            self.dataset[desc][0][1:],
            np.concatenate([
                np.repeat(
                    np.array([self.y_avg]),
                    max(0, self.y_len - len(self.dataset[desc][1])), axis=0),
                self.dataset[desc][1][:self.y_len]]),
            np.concatenate([
                np.repeat(
                    np.array([self.z_avg]),
                    max(0, self.z_len - len(self.dataset[desc][2])), axis=0),
                self.dataset[desc][2][:self.z_len]]))

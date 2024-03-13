"""PTB dataset"""

import os
import pickle
import hashlib
import logging
from random import Random
from itertools import chain, repeat
from typing import Any, Dict, List
import torch
from torch import Tensor
from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from .base import _RawDataset
from ..utils import load_obj, metadata_updated, save_obj
from ...utils.arg_parser import ArgParseOption, options, require


log = logging.getLogger(__name__)


@require(
    'training.batch_size',
    'data.raw.penn_treebank.batch_size',
    'data.fl_split.num')
@options(
    ArgParseOption(
        'ptb.bs', 'penn-treebank.batch-size', type=int, default=128,
        help="Batch size for language modelling. Not to be confused with "
        "training.batch_size, which is used as seq_length in language modelling."))
class PennTreebank(_RawDataset):
    """
    Wrapper of torchtexts' PTB dataset
    get_data and get_label returns tensors of shape [batch_size]
    i.e. batch_first = False (stacked data will have shape [seq_len, batch_size])

    Embeddings are not performed.

    Intended to be used with DeletagedSplit split method. Uses i.i.d split.
    """

    _root = "data/src/lang_model.ptb"

    def __init__(self, cfg, mode, split_id, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.train_batch_size = cfg.training.batch_size
        self.batch_size = cfg.data.raw.penn_treebank.batch_size
        self.num_splits = cfg.data.fl_split.num
        self.split_id = split_id

        cache_root = self.root.joinpath(f"{self._hashed_meta()}")
        os.makedirs(cache_root, exist_ok=True)
        self.cache_path = cache_root.joinpath(f"{self.__class__.__name__}.pkl")
        self.meta_cache_path = cache_root.joinpath(f"{self.__class__.__name__}_meta.pkl")
        self.preprocess_and_cache_meta()

        cache = load_obj(self.cache_path)
        self.data = {
            'train': (
                cache['data']['train'][split_id]
                if split_id is not None else None),
            'val': cache['data']['val'],
            'test': cache['data']['test']
        }
        self.vocab_size = cache['vocab_size']

    def metadata(self) -> Dict[str, Any]:
        return {**super().metadata(), **{
            'seq_length': self.train_batch_size,
            'batch_size': self.batch_size,
            'num_splits': self.num_splits,
        }}

    def _hashed_meta(self):
        name_hash = hashlib.md5()
        name_hash.update(pickle.dumps(
            {k: v for k, v in self.metadata().items()
                if k not in {'mode', 'split_id'}}))
        return f"{name_hash.hexdigest()}"

    def re_preprocess(self) -> bool:
        return metadata_updated(self.meta, self.meta_cache_path)

    def preprocess(self) -> None:
        data = list(chain.from_iterable(datasets.PennTreebank(
            root=self.root, split=('train', 'valid', 'test'))))
        Random(self.seed).shuffle(data)

        # Language modelling dataset cannot be directly constructed and split
        # in the first dimension. For each different split we need construct
        # different datasets, thus we split it here.
        train, val, test = self.train_eval_test
        total = sum(self.train_eval_test)
        fold_size = len(data) // sum(self.train_eval_test)

        def load_folds(start: int, num: int):
            return data[start * fold_size: (start + num) * fold_size] \
                if start + num <= total \
                else data[:((start + num) % total) * fold_size] + \
                data[start * fold_size:]

        def load_split(train_data, split_id: int):
            split_size = len(train_data) // self.num_splits
            return train_data[
                split_id * split_size: (split_id + 1) * split_size]

        train_datas, val_data, test_data = (
            [load_split(load_folds(self.fold, train), id)
             for id in range(self.num_splits)],
            load_folds(self.fold + train, val),
            load_folds(self.fold + train + val, test))

        tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(
            map(tokenizer, data), specials=['<unk>', '<eos>'])
        vocab.set_default_index(vocab['<unk>'])

        def tokenize(raw_text_iter) -> Tensor:
            """Converts raw text into a flat Tensor."""
            data = [torch.tensor(
                vocab(tokenizer(item) + ['<eos>']), dtype=torch.long
            ) for item in raw_text_iter]
            return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

        def batchify(data: Tensor) -> Tensor:
            num_batches = data.size(0) // self.batch_size
            data = data[:num_batches * self.batch_size]
            data = data.view(self.batch_size, num_batches).t().contiguous()
            return data

        train_datas, val_data, test_data = (
            [batchify(tokenize(train_data)) for train_data in train_datas],
            batchify(tokenize(val_data)),
            batchify(tokenize(test_data)))

        # data shape: [seq_len (training.batch_size), batch_size]

        save_obj({
            'data': {
                'train': train_datas,
                'val': val_data,
                'test': test_data
            }, 'vocab_size': len(vocab),
        }, self.cache_path)

    def load_sample_descriptors(self):
        raise RuntimeError("Use load_{train|eval|test}_descs instead")

    def _aligned_len(self, mode):
        return (
            (len(self.data[mode]) - 1) // self.train_batch_size
        ) * self.train_batch_size

    def load_train_descs(self) -> List[Any]:
        return list(zip(repeat('train'), range(self._aligned_len('train'))))

    def load_eval_descs(self) -> List[Any]:
        return list(zip(repeat('val'), range(self._aligned_len('val'))))

    def load_test_descs(self) -> List[Any]:
        return list(zip(repeat('test'), range(self._aligned_len('test'))))

    def get_data(self, desc):
        ds_idx, dt_idx = desc[0], desc[1]
        return self.data[ds_idx][dt_idx]

    def get_label(self, desc):
        ds_idx, dt_idx = desc[0], desc[1]
        return self.data[ds_idx][dt_idx + 1]

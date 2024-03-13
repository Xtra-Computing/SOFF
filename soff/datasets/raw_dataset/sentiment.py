"""Sentiment analysis datasets"""

import csv
import random
import logging
import multiprocessing
from typing import Tuple
from string import punctuation
from dataclasses import dataclass
import torch
import numpy as np
from tqdm import tqdm
from ...utils.arg_parser import ArgParseOption, options, require
from ..utils import (
    metadata_updated, save_obj, load_obj, file_exists, multiprocess_num_jobs)
from .base import _CachedDataset, _RawDataset, _OnlineDataset, _Source

log = logging.getLogger(__name__)


class GloVe(_CachedDataset, _OnlineDataset):
    """Base class of GloVe embeddings"""
    _root = 'data/src/glove'
    # Subclasses must define these attributes
    src_file: str = ''
    size: int = 0
    dim: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_path = f"{self.root}/{self.__class__.__name__}.pkl"
        self.download_and_extrat()
        self.preprocess_and_cache_meta()

    def metadata(self):
        # Immutable dataset does not need to record seed/fold/split
        return {'src_file': self.src_file, 'size': self.size, 'dim': self.dim}

    def skip_download_extract(self):
        return not self.dataset_updated()

    def re_preprocess(self):
        return not file_exists(self.meta_cache_path)

    def preprocess(self):
        src_path = self.root.joinpath(self.src_file).as_posix()
        embedding_dict = {}
        with open(src_path, 'r', encoding="utf-8") as file:
            with multiprocessing.Pool(multiprocess_num_jobs()) as pool:
                embedding_dict = dict(tqdm(pool.imap(
                    ProcessGlove(self.dim), file, chunksize=10000),
                    total=self.size))
        avg_vec = np.mean(list(embedding_dict.values()), axis=0).tolist()
        save_obj({'dict': embedding_dict, 'unk': avg_vec}, self.cache_path)

    def load_embedding_dict(self) -> Tuple[dict, list]:
        """returns: dict: the embedding_dict, list: the embeddings for <UNK>"""
        log.info("Loading GloVe embeddings...")
        data = load_obj(self.cache_path)
        return data['dict'], data['unk']


@dataclass
class ProcessGlove:
    """Glove processing functor"""
    dim: int

    def __call__(self, line):
        values = line.split()
        return (values[0],
                np.asarray(values[-self.dim:], np.float32).tolist())


class GloVe840B300D(GloVe):
    """GloVe with 840 billion words and embedding dim of 300"""
    src_file = 'glove.840B.300d.txt'
    size = 2196017
    dim = 300

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, srcs=_Source(
            url='http://nlp.stanford.edu/data/glove.840B.300d.zip',
            file='glove.840B.300d.zip',
            md5='2ffafcc9f9ae46fc8c95f32372976137',
            compress_type='zip'))


class Glove6B50D(GloVe):
    """GloVe with 6 billion words and embedding dim of 50"""
    src_file = 'glove.6B.50d.txt'
    size = 400000
    dim = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, srcs=_Source(
            url='http://nlp.stanford.edu/data/glove.6B.zip',
            file='glove.6B.zip',
            md5='056ea991adb4740ac6bf1b6d9b50408b',
            compress_type='zip'))


glove_name_to_class = {
    'glove840b300d': GloVe840B300D,
    'glove6b50d': Glove6B50D,
}



@require(
    'data.raw.sentiment.seq_length'
    'data.raw.sentiment.glove'
)
@options(
    "Sentiment Dataset Configs",
    ArgParseOption(
        'sent.sl', 'sentiment.seq-length',
        default=200, type=int, metavar='LENGTH',
        help="Maximum sequence length for the sentment datasets"),
    ArgParseOption(
        'sent.gl', 'sentiment.glove',
        default='glove840b300d', choices=glove_name_to_class.keys(),
        help="Glove embedding dict to use")
)
class _SentimentDataset(_RawDataset, _OnlineDataset):
    """Base class of all sentiment analysis datasets (with binary labels)"""

    def __init__(self, cfg, *args, **kwargs):
        # glove needs to be loaded before the rest can be initialzed
        super().__init__(cfg, *args, **kwargs)

        self.seq_length = cfg.data.raw.sentiment.seq_length
        self.glove = glove_name_to_class[cfg.data.raw.sentiment.glove](cfg)
        self.dim = self.glove.dim
        self.cache_path = f"{self.root}/{self.__class__.__name__}.pkl"

        self.download_and_extrat()
        self.preprocess_and_cache_meta()

        # free unused memory after initialization (glove is quite large)
        del self.glove

        # load dataset into memory
        self.dataset = load_obj(self.cache_path)

    def metadata(self):
        return {
            **super().metadata(),
            'seq_length': self.seq_length,
            'glove': self.glove.__class__.__name__
        }

    def re_preprocess(self):
        # I.e. If only seed/fold updated, there's no need to re-preprocess.
        return metadata_updated(
            self.meta, self.meta_cache_path, ['seq_length', 'glove'])

    def preprocess(self):
        raise NotImplementedError

    def embedding_dim(self) -> int:
        """The embedding dimension (depnds on glove)"""
        return self.dim

    def get_data(self, desc):
        return torch.Tensor(self.dataset[desc][0])

    def get_label(self, desc):
        return float(self.dataset[desc][1])

    def skip_download_extract(self):
        return not self.re_preprocess()

    def load_sample_descriptors(self):
        return list(range(len(self.dataset)))


class IMDB(_SentimentDataset):
    """The IMDB dataset (https://ai.stanford.edu/~amaas/data/sentiment/)"""
    _root = 'data/src/sentiment.imdb'

    def __init__(self, *args, **kwargs):
        self.size = 25000
        super().__init__(
            *args, **kwargs,
            srcs=_Source(
                url='https://ai.stanford.edu/~amaas/'
                'data/sentiment/aclImdb_v1.tar.gz',
                file='aclImdb_v1.tgz',
                md5='7c2ac02c03563afcf9b574c7e56c153a',
                compress_type='tgz'))

    def preprocess(self):
        dataset = []
        embedding_dict, unk = self.glove.load_embedding_dict()

        def parse_entry(path, label):
            with open(path, 'r', encoding="utf-8") as file:
                data = []
                i = 0
                for word in file.read().lower().split():
                    if word in embedding_dict and word not in punctuation:
                        data.append(embedding_dict[word])
                        i += 1
                    if i >= self.seq_length:
                        break
                while i < self.seq_length:
                    i += 1
                    data.append(unk)
                dataset.append((data, label))

        for path in tqdm(
                self.root.glob('aclImdb/*/pos/*.txt'), total=self.size):
            parse_entry(path, 1)
        for path in tqdm(
                self.root.glob('aclImdb/*/neg/*.txt'), total=self.size):
            parse_entry(path, 0)

        save_obj(dataset, self.cache_path)


class SST2(_SentimentDataset, _OnlineDataset):
    """The SST2 dataset (https://nlp.stanford.edu/sentiment/)"""
    _root = 'data/src/sentiment.sst2'

    def __init__(self, *args, **kwargs):
        self.size = 67349
        super().__init__(
            *args, **kwargs,
            srcs=_Source(
                url='https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
                file='SST-2.zip',
                md5='9f81648d4199384278b86e315dac217c',
                compress_type='zip'))

    def preprocess(self):
        dataset = []
        embedding_dict, unk = self.glove.load_embedding_dict()

        def parse_entry(row):
            data = []
            i = 0
            for word in row[0].lower().split():
                if word in embedding_dict and word not in punctuation:
                    data.append(embedding_dict[word])
                    i += 1
                if i >= self.seq_length:
                    break
            while i < self.seq_length:
                i += 1
                data.append(unk)
            dataset.append((data, int(row[1])))

        with open(self.root.joinpath('SST-2/train.tsv'),
                  'r', encoding='utf-8') as file:
            rows = csv.reader(file, delimiter='\t')
            next(rows)  # skip header
            for row in tqdm(rows, total=self.size):
                parse_entry(row)

        save_obj(dataset, self.cache_path)


class Amazon(_SentimentDataset, _OnlineDataset):
    """
    The amazon movie review dataset
    (https://snap.stanford.edu/data/web-Movies.html)
    """
    _root = 'data/src/sentiment.amazon'

    def __init__(
            self, *args, sample_ratio=0.01, sample_method='equal', **kwargs):
        self.size = 7911684
        self.sample_ratio = sample_ratio
        self.sample_method = sample_method
        self.sample_seed = 0xdeadbeef
        super().__init__(
            *args, **kwargs,
            srcs=_Source(
                url='https://snap.stanford.edu/data/movies.txt.gz',
                file='movies.txt.gz',
                md5='76a57d8213f0163d0da9f6118b839bb6',
                compress_type='gz'))

    def metadata(self):
        return {
            **super().metadata(),
            'sample_ratio': self.sample_ratio,
            'sample_method': self.sample_method,
            'sample_seed': self.sample_seed
        }

    def preprocess(self):
        dataset = []
        embedding_dict, unk = self.glove.load_embedding_dict()

        num_samples = round(self.size * self.sample_ratio)
        assert 0 < num_samples <= self.size

        def parse_label(file):
            label = None
            line = file.readline()
            while line and (not line.isspace()):
                if line.startswith('review/score:'):
                    label = (1 if float(line.split(':')[1].strip()) >= 2.5
                             else 0)
                line = file.readline()
            file.readline()
            return label

        def parse_entry(file):
            data = []
            label = None

            line = file.readline()
            while line and (not line.isspace()):
                if line.startswith('review/score:'):
                    label = (1 if float(line.split(':')[1].strip()) >= 2.5
                             else 0)
                elif line.startswith('review/text:'):
                    i = 0
                    for word in line[12:].strip().lower().split():
                        if word in embedding_dict and word not in punctuation:
                            data.append(embedding_dict[word])
                            i += 1
                        if i >= self.seq_length:
                            break
                    while i < self.seq_length:
                        i += 1
                        data.append(unk)
                line = file.readline()
            # consume the empty line
            file.readline()

            if label is not None:
                dataset.append((data, label))

        def skip_entry(file):
            line = file.readline()
            while line and (not line.isspace()):
                line = file.readline()
            # consume the empty line
            file.readline()

        with open(self.root.joinpath('movies.txt').as_posix(), 'r',
                  encoding='iso-8859-1') as file:
            if self.sample_method == 'natrual':
                samples = set(random.Random(self.sample_seed)
                              .sample(range(self.size), num_samples))
                for i in tqdm(range(self.size), total=self.size):
                    if i in samples:
                        parse_entry(file)
                    else:
                        skip_entry(file)
            elif self.sample_method == 'equal':
                # first get labels of all entries
                labels = []
                for i in tqdm(range(self.size), total=self.size):
                    labels.append(parse_label(file))

                # get indices whose labels are half negative, half positive
                pos = 0
                neg = 0
                shuffled_idx = list(range(self.size))
                random.Random(self.sample_seed).shuffle(shuffled_idx)
                samples = set()
                for idx in shuffled_idx:
                    if pos == num_samples // 2 and neg == num_samples // 2:
                        break
                    if pos < num_samples // 2 and labels[idx] == 1:
                        pos += 1
                        samples.add(idx)
                    elif neg < num_samples // 2 and labels[idx] == 0:
                        neg += 1
                        samples.add(idx)

                # parse subsamples
                file.seek(0, 0)
                for i in tqdm(range(self.size), total=self.size):
                    if i in samples:
                        parse_entry(file)
                    else:
                        skip_entry(file)
            else:
                raise NotImplementedError("Unknown sample method")

        save_obj(dataset, self.cache_path)

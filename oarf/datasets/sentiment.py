import csv
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from oarf.datasets.fl_dataset import RawDataset
from oarf.datasets.base import OnlineDataset, Source
from string import punctuation

log = logging.getLogger(__name__)


class sst2:
    root = 'data/src/sentiment.sst'
    url = 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip'
    src = 'SST-2.zip'
    src_md5 = '9f81648d4199384278b86e315dac217c'
    size = 67349

    file_base = 'sst2'
    shuffle_seed = 0xdeadbeef
    file_md5 = '2a3a0a43f594a08050b6934e06ae058f'


class amazon:
    root = 'data/src/sentiment.amazon'
    url = ''


class SentimentDataset(RawDataset):
    def __init__(self, seq_length, glove, *args, **kwargs):
        self.seq_length = seq_length
        self.glove = GloVe(glove)
        # glove needs to be loaded before the rest can be initialzed
        super().__init__(*args, **kwargs)

    def embedding_dim(self) -> int:
        return self.glove.dim

    def get_data(self, sample):
        return torch.Tensor(sample[0])

    def get_label(self, sample):
        return float(sample[1])


class GloVe(OnlineDataset):
    def __init__(self, glove='glove_840B_300D'):
        if glove == 'glove_840B_300D':
            self.src_file = 'glove.840B.300d.txt'
            self.size = 2196017
            self.dim = 300
            super().__init__(
                root='data/src/glove',
                srcs=Source(
                    url='http://nlp.stanford.edu/data/glove.840B.300d.zip',
                    file='glove.840B.300d.zip',
                    md5='2ffafcc9f9ae46fc8c95f32372976137',
                    compress_type='zip'),
                cache_file='glove.840B.300d.pkl',
                cache_md5='d07e31f7dcfe0d95a91fe3cdb9753b52')
        elif glove == 'glove_6B_50D':
            self.src_file = 'glove.6B.50d.txt'
            self.size = 400000
            self.dim = 50
            super().__init__(
                root='data/src/glove',
                srcs=Source(
                    url='http://nlp.stanford.edu/data/glove.6B.zip',
                    file='glove.6B.zip',
                    md5='056ea991adb4740ac6bf1b6d9b50408b',
                    compress_type='zip'),
                cache_file='glove.6B.50d.pkl',
                cache_md5='615ed67ff2b2815fc9840541a60a112c')

    def preprocess(self):
        src_path = self.root.joinpath(self.src_file).as_posix()
        embedding_dict = {}
        sum_embeddings = np.zeros(self.dim)
        num_words = 0
        with open(src_path, 'r', encoding="utf-8") as f:
            for line in tqdm(f, total=self.size):
                values = line.split()
                embedding_dict[values[0]] = np.asarray(
                    values[-self.dim:], np.float32).tolist()
                sum_embeddings += embedding_dict[values[0]]
                num_words += 1
        average_vec = (sum_embeddings / num_words).tolist()
        return {'dict': embedding_dict, 'unk': average_vec}

    def load_embedding_dict(self) -> (dict, list):
        """returns: dict: the embedding_dict, list: the embeddings for <UNK>"""
        log.info("Loading GloVe embeddings...")
        data = self.load_cache()
        return data['dict'], data['unk']


class IMDB(SentimentDataset, OnlineDataset):
    def __init__(self, seq_length=200, glove='glove_840B_300D', *_, **__):
        self.size = 25000
        self.cache_base = 'imdb'
        self.shuffle_seed = 0xdeadbeef

        super().__init__(
            root='data/src/sentiment.imdb',
            srcs=Source(
                url='https://ai.stanford.edu/~amaas/'
                'data/sentiment/aclImdb_v1.tar.gz',
                file='aclImdb_v1.tgz',
                md5='7c2ac02c03563afcf9b574c7e56c153a',
                compress_type='tgz'),
            cache_file=self.cache_base + '_{}_{}_{}.pkl'.format(
                self.shuffle_seed, seq_length, glove),
            seq_length=seq_length, glove=glove)

    def preprocess(self):
        dataset = []
        embedding_dict, UNK = self.glove.load_embedding_dict()

        def parse_entry(path, label):
            with open(path, 'r', encoding="utf-8") as f:
                data = []
                i = 0
                for w in f.read().lower().split():
                    if w in embedding_dict and w not in punctuation:
                        data.append(embedding_dict[w])
                        i += 1
                    if i >= self.seq_length:
                        break
                while i < self.seq_length:
                    i += 1
                    data.append(UNK)
                dataset.append((data, label))

        for p in tqdm(self.root.glob('aclImdb/*/pos/*.txt'), total=self.size):
            parse_entry(p, 1)
        for p in tqdm(self.root.glob('aclImdb/*/neg/*.txt'), total=self.size):
            parse_entry(p, 0)

        random.Random(self.shuffle_seed).shuffle(dataset)
        return {'train': dataset[:len(dataset) * 5 // 6],
                'test': dataset[len(dataset) * 5 // 6: len(dataset) * 11//12],
                'val': dataset[len(dataset) * 11//12:]}

    def load_train_dataset(self, _):
        return self.load_cache()['train']

    def load_eval_dataset(self, seed):
        return self.load_cache()['test']

    def load_test_dataset(self, seed):
        return self.load_cache()['val']


class SST2(SentimentDataset, OnlineDataset):
    def __init__(self, seq_length=200, glove='glove_840B_300D', *_, **__):
        self.size = 67349
        self.cache_base = 'sst2'
        self.shuffle_seed = 0xdeadbeef

        super().__init__(
            root='data/src/sentiment.sst2',
            srcs=Source(
                url='https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
                file='SST-2.zip',
                md5='9f81648d4199384278b86e315dac217c',
                compress_type='zip'),
            cache_file=self.cache_base + '_{}_{}_{}.pkl'.format(
                self.shuffle_seed, seq_length, glove),
            seq_length=seq_length, glove=glove)

    def preprocess(self):
        dataset = []
        embedding_dict, UNK = self.glove.load_embedding_dict()

        def parse_entry(row):
            data = []
            i = 0
            for w in row[0].lower().split():
                if w in embedding_dict and w not in punctuation:
                    data.append(embedding_dict[w])
                    i += 1
                if i >= self.seq_length:
                    break
            while i < self.seq_length:
                i += 1
                data.append(UNK)
            dataset.append((data, int(row[1])))

        with open(self.root.joinpath('SST-2/train.tsv'), 'r') as f:
            rows = csv.reader(f, delimiter='\t')
            next(rows)  # skip header
            for row in tqdm(rows, total=sst2.size):
                parse_entry(row)

        random.Random(sst2.shuffle_seed).shuffle(dataset)
        return {'train': dataset[:len(dataset) * 5 // 6],
                'test': dataset[len(dataset) * 5 // 6: len(dataset) * 11//12],
                'val': dataset[len(dataset) * 11//12:]}

    def load_train_dataset(self, _):
        return self.load_cache()['train']

    def load_eval_dataset(self, seed):
        return self.load_cache()['test']

    def load_test_dataset(self, seed):
        return self.load_cache()['val']


class Amazon(SentimentDataset, OnlineDataset):
    def __init__(self, seq_length: int = 200, glove='glove_840B_300D',
                 sample_ratio=0.01, sample_method='equal', *_, **__):
        self.cache_base = 'amazon'
        self.shuffle_seed = 0xdeadbeef
        self.sample_ratio = sample_ratio
        self.sample_method = sample_method
        self.size = 7911684
        super().__init__(
            root='data/src/sentiment.amazon',
            srcs=Source(
                url='https://snap.stanford.edu/data/movies.txt.gz',
                file='movies.txt.gz',
                md5='76a57d8213f0163d0da9f6118b839bb6',
                compress_type='gz'),
            cache_file=self.cache_base + '_{}_{}_{}_{}_{}.pkl'.format(
                self.shuffle_seed, seq_length, glove,
                self.sample_method, self.sample_ratio),
            seq_length=seq_length, glove=glove)

    def preprocess(self):
        dataset = []
        embedding_dict, UNK = self.glove.load_embedding_dict()

        num_samples = round(self.size * self.sample_ratio)
        assert 0 < num_samples <= self.size

        def parse_label(f):
            label = None
            line = f.readline()
            while line and (not line.isspace()):
                if line.startswith('review/score:'):
                    label = (1 if float(line.split(':')[1].strip()) >= 2.5
                             else 0)
                line = f.readline()
            f.readline()
            return label

        def parse_entry(f):
            data = []
            label = None

            line = f.readline()
            while line and (not line.isspace()):
                if line.startswith('review/score:'):
                    label = (1 if float(line.split(':')[1].strip()) >= 2.5
                             else 0)
                elif line.startswith('review/text:'):
                    i = 0
                    for w in line[12:].strip().lower().split():
                        if w in embedding_dict and w not in punctuation:
                            data.append(embedding_dict[w])
                            i += 1
                        if i >= self.seq_length:
                            break
                    while i < self.seq_length:
                        i += 1
                        data.append(UNK)
                line = f.readline()
            # consume the empty line
            f.readline()

            if label is not None:
                dataset.append((data, label))

        def skip_entry(f):
            line = f.readline()
            while line and (not line.isspace()):
                line = f.readline()
            # consume the empty line
            f.readline()

        with open(self.root.joinpath('movies.txt').as_posix(), 'r',
                  encoding='iso-8859-1') as f:
            if self.sample_method == 'natrual':
                samples = set(random.Random(self.shuffle_seed)
                              .sample(range(self.size), num_samples))
                for i in tqdm(range(self.size), total=self.size):
                    if i in samples:
                        parse_entry(f)
                    else:
                        skip_entry(f)
            elif self.sample_method == 'equal':
                # first get labels of all entries
                labels = []
                for i in tqdm(range(self.size), total=self.size):
                    labels.append(parse_label(f))

                # get indices whose labels are half negative, half positive
                pos = 0
                neg = 0
                shuffled_idx = list(range(self.size))
                random.Random(self.shuffle_seed).shuffle(shuffled_idx)
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
                f.seek(0, 0)
                for i in tqdm(range(self.size), total=self.size):
                    if i in samples:
                        parse_entry(f)
                    else:
                        skip_entry(f)
            else:
                raise NotImplementedError("Unknown sample method")

        random.Random(sst2.shuffle_seed).shuffle(dataset)
        return {'train': dataset[:len(dataset) * 5 // 6],
                'test': dataset[len(dataset) * 5 // 6: len(dataset) * 11//12],
                'val': dataset[len(dataset) * 11//12:]}

    def load_train_dataset(self, _):
        return self.load_cache()['train']

    def load_eval_dataset(self, seed):
        return self.load_cache()['test']

    def load_test_dataset(self, seed):
        return self.load_cache()['val']

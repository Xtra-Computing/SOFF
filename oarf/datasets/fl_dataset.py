import os
import random
import pickle
import logging
import numpy as np
from os import path
from typing import List
from itertools import repeat
from scipy.stats import powerlaw
from torch.utils.data import Dataset, ConcatDataset
from oarf.datasets.base import RawDataset

log = logging.getLogger(__name__)


class FLDataset(Dataset):
    def __init__(self, datasets: List[RawDataset],
                 mode, num_clients, client_id, seed=0xdeadbeef, *_, **__):

        save_path = self.get_save_path(datasets, mode, num_clients, client_id)
        log.debug("Data save path: {}".format(save_path))
        self.datasets = datasets

        if self.needs_cache(save_path, seed):
            self.seed = seed
            self.dataset = self.load_dataset(datasets, mode, seed)
            self.num_clients = num_clients
            self.id = client_id

            # override classes if specified by underlying datasets
            if all([hasattr(d, 'classes') for d in datasets]) and \
                    all([d.classes == datasets[0].classes for d in datasets]):
                self.classes = set(datasets[0].classes)
            else:
                self.classes = set([datasets[idx].get_label(sample)
                                    for sample, idx in self.dataset])

            # we don't split test and eval dataset, C/S use the same one
            if mode == 'train':
                self.preprocess_train_dataset()

            self.cache_dataset(save_path)

    # TODO: this is not unique enough
    def get_save_path(self, datasets, mode, num_clients, client_id):
        save_dir = 'data/{}/{}'.format(
            '_'.join([d.__module__.split('.')[-1] + '_' + d.__class__.__name__
                      for d in datasets]),
            self.__class__.__name__)

        file_stem = self.gen_save_file_stem(mode, num_clients, client_id)
        save_path = '{}/{}.pkl'.format(save_dir, file_stem)
        return save_path

    def gen_save_file_stem(self, mode, num_clients, client_id):
        if mode == 'train':
            if(client_id < 0 or client_id > num_clients - 1 or
                    type(client_id) is not int):
                raise Exception("Illegal client id {}".format(client_id))
            return 'data_{}_{}_{}'.format(mode, num_clients, client_id)
        else:
            return 'data_{}'.format(mode)

    def needs_cache(self, save_path, seed):
        # If save path is given, try to load data
        if path.exists(save_path):
            log.info("{}: Loading saved data..."
                     .format(self.__class__.__name__))
            with open(save_path, 'rb') as f:
                self.__dict__.update(pickle.load(f))
            # if seed is changed, we need to re-process the dataset
            if self.seed != seed:
                return True
            # TODO: check if underlying dataset is the same
            return False
        return True

    def cache_dataset(self, save_path):
        os.makedirs(path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_dataset(self, datasets, mode, seed):
        """
        Child class should implement this method.
        Should return: [(sample, index of `datasets`) ... ]
        """
        raise NotImplementedError

    def preprocess_train_dataset(self):
        pass

    def num_classes(self):
        return len(self.classes)

    def histogram(self):
        hist = dict(zip(*np.unique(
            [self.datasets[ds_idx].get_label(sample)
             for sample, ds_idx in self.dataset],
            return_counts=True)))
        return [(int(hist[cls]) if cls in hist else 0) for cls in self.classes]

    def __getitem__(self, idx):
        """return: feature, label"""
        sample, ds_idx = self.dataset[idx]
        return self.datasets[ds_idx].get_data(sample), \
            self.datasets[ds_idx].get_label(sample)

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        counter = {}
        for cls in self.classes:
            counter[cls] = 0
        for _, label in self.dataset:
            counter[label] += 1
        return "Client id: {}\n  # entries: {}\n  label dist: {}".format(
            self.id, len(self.dataset), str(counter))


class _SyntheticDataset(FLDataset):
    def __init__(self, datasets: List[RawDataset],
                 mode, num_clients, client_id, seed=0xdeadbeef, *_, **__):
        """if mode is eval or test, num_clients and client_id are ignored"""
        # Process the dataset if needs processing
        self.mode = mode
        super().__init__(datasets, mode, num_clients, client_id, seed)

    def needs_cache(self, save_path, seed):
        return super().needs_cache(save_path, seed) or (
            self.mode == 'train' and self.require_preprocess())

    def load_dataset(self, datasets, mode, seed):
        if mode == 'train':
            return ConcatDataset([
                list(zip(dataset.load_train_dataset(seed), repeat(idx)))
                for idx, dataset in enumerate(datasets)])
        elif mode == 'eval':
            return ConcatDataset([list(zip(
                dataset.load_eval_dataset(seed), repeat(idx)))
                for idx, dataset in enumerate(datasets)])
        elif mode == 'test':
            return ConcatDataset([
                list(zip(dataset.load_test_dataset(seed), repeat(idx)))
                for idx, dataset in enumerate(datasets)])
        else:
            raise Exception("Mode {} unrecognized".format(mode))

    def require_preprocess(self):
        """Implemented by intermediate classes. End-user should ignore this"""
        raise NotImplementedError

    def preprocess_train_dataset(self):
        """Implemented by intermediate classes. End-user should ignore this"""
        raise NotImplementedError


class _QuantitySkewDataset(_SyntheticDataset):
    """datasets that only skews in quantity instead of distributions"""

    def require_preprocess(self):
        """if cdf is different than the old cdf, then preprocess is needed"""
        pdf = self.pdf()
        cdf = list(np.cumsum(pdf))
        return (not hasattr(self, 'cdf')) or (self.cdf != cdf)

    def preprocess_train_dataset(self):
        pdf = self.pdf()
        assert(abs(sum(pdf) - 1.) < 1e-5)
        self.cdf = list(np.cumsum(pdf))

        # group sample indices by label
        class_idx_map = {}
        idxs = list(range(len(self.dataset)))
        random.Random(self.seed).shuffle(idxs)
        for cls in self.classes:
            class_idx_map[cls] = []
        for idx in idxs:
            sample, ds_idx = self.dataset[idx]
            class_idx_map[self.datasets[ds_idx].get_label(sample)].append(idx)

        # i.i.d splitting, and fetch the result for this client only
        client_idx_map = []
        for cls in self.classes:
            begin = 0 if self.id == 0 else \
                round(len(class_idx_map[cls]) * self.cdf[self.id - 1])
            end = round(len(class_idx_map[cls]) * self.cdf[self.id])
            client_idx_map.extend(class_idx_map[cls][begin:end])

        random.Random(self.seed).shuffle(client_idx_map)
        self.dataset = [self.dataset[i] for i in client_idx_map]

    def pdf(self):
        """Impelementation must be deterministic with given self.seed"""
        raise NotImplementedError


class IidDataset(_QuantitySkewDataset):
    def pdf(self):
        return [1. / self.num_clients] * self.num_clients


class DirichletQuantitySkewDataset(_QuantitySkewDataset):
    def __init__(self, alpha: float = 0.5, *args, **kwargs):
        """alpha: concentration level"""
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def pdf(self):
        np.random.seed(self.seed)
        return np.random.dirichlet([self.alpha] * self.num_clients)

    def gen_save_file_stem(self, mode, num_clients, client_id):
        stem = super().gen_save_file_stem(mode, num_clients, client_id)
        return stem + '_{}'.format(self.alpha) if mode == 'train' else stem


class PowerLawQuantitySkewDataset(_QuantitySkewDataset):
    def __init__(self, alpha: float = 0.5, *args, **kwargs):
        """alpha: power distribution: alpha * x ^ {alpha - 1}, x: client id"""
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def pdf(self):
        dist = powerlaw.pdf(np.arange(1, 1+self.num_clients)
                            / float(self.num_clients), self.alpha)
        dist /= sum(dist)
        return dist

    def gen_save_file_stem(self, mode, num_clients, client_id):
        stem = super().gen_save_file_stem(mode, num_clients, client_id)
        return stem + '_{}'.format(self.alpha) if mode == 'train' else stem


class _LabelSkewDataset(_SyntheticDataset):
    def require_preprocess(self):
        if not hasattr(self, 'cdfs'):
            return True
        else:
            pdfs = self.pdfs()
            for key, cdf in self.cdfs.items():
                if cdf != list(np.cumsum(pdfs[key])):
                    return True

        return False

    def preprocess_train_dataset(self):
        pdfs = self.pdfs()
        self.cdfs = {key: list(np.cumsum(val)) for key, val in pdfs.items()}

        # group sample indices by label
        class_idx_map = {}
        idxs = list(range(len(self.dataset)))
        random.Random(self.seed).shuffle(idxs)
        for cls in self.classes:
            class_idx_map[cls] = []
        for idx in idxs:
            sample, ds_idx = self.dataset[idx]
            class_idx_map[self.datasets[ds_idx].get_label(sample)].append(idx)

        # i.i.d splitting, and fetch the result for this client only
        client_idx_map = []
        for cls in self.classes:
            begin = 0 if self.id == 0 else \
                round(len(class_idx_map[cls]) * self.cdfs[cls][self.id - 1])
            end = round(len(class_idx_map[cls]) * self.cdfs[cls][self.id])
            client_idx_map.extend(class_idx_map[cls][begin:end])

        random.Random(self.seed).shuffle(client_idx_map)
        self.dataset = [self.dataset[i] for i in client_idx_map]

    def pdfs(self):
        """returns dict, where the key is class (labels),
        and the value is pdf for each parties"""
        raise NotImplementedError


class DirichletLabelSkewDataset(_LabelSkewDataset):
    def __init__(self, alpha: float = 0.5, *args, **kwargs):
        """alpha: concentration level"""
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def pdfs(self):
        np.random.seed(self.seed)
        return {cls: np.random.dirichlet([self.alpha] * self.num_clients)
                for cls in range(len(self.classes))}

    def gen_save_file_stem(self, mode, num_clients, client_id):
        stem = super().gen_save_file_stem(mode, num_clients, client_id)
        return stem + '_{}'.format(self.alpha) if mode == 'train' else stem


class RealisticDataset(FLDataset):
    def __init__(self, datasets: List[RawDataset],
                 mode, num_clients, client_id, seed=0xdeadbeef, *_, **__):
        """if mode is eval or test, num_clients and client_id are ignored"""

        assert num_clients == len(datasets) and \
            ((client_id is None) or (0 <= client_id < len(datasets))), \
            "num_clients must equal to len(datasets) " \
            "when using realistic splitting"

        self.client_id = client_id
        super().__init__(datasets, mode, num_clients, client_id, seed)

    def load_dataset(self, datasets, mode, seed):
        if mode == 'train':
            return list(zip(datasets[self.client_id].load_train_dataset(seed),
                            repeat(self.client_id)))
        elif mode == 'eval':
            return ConcatDataset([
                list(zip(dataset.load_eval_dataset(seed), repeat(idx)))
                for idx, dataset in enumerate(datasets)])
        elif mode == 'test':
            return ConcatDataset([
                list(zip(dataset.load_test_dataset(seed), repeat(idx)))
                for idx, dataset in enumerate(datasets)])
        else:
            raise Exception("Mode {} unrecognized".format(mode))

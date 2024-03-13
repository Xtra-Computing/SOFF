"""CIFAR10/100 and MNIST dataset"""

import logging
import pathlib
import pickle
from typing import Callable
from abc import ABC, abstractmethod
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
from .base import _RawDataset
from ..utils import save_obj, load_obj, file_exists
from ...utils.arg_parser import ArgParseOption, options, require

log = logging.getLogger(__name__)


@require("data.raw.cifar_mnist.rotation_degree")
@options(
    "CIFAR/MNIST Dataset Config",
    ArgParseOption(
        'cm.rd', 'cifar-mnist.rotation-degree',
        default=5.0, type=float, metavar="DEGREE",
        help="Random rotation degree ranges, for data augmentation")
)
class _AugmentedDataset(_RawDataset, ABC):
    def __init__(self, cfg, mode, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs, mode=mode)
        self.rotation_degree = cfg.data.raw.cifar_mnist.rotation_degree
        self.transform = self._load_transform(mode, self.rotation_degree)

    def metadata(self):
        return {**super().metadata(), 'rotation_degree': self.rotation_degree}

    def re_preprocess(self):
        return not file_exists(self.meta_cache_path)

    @abstractmethod
    def _load_dataset(self, root) -> ConcatDataset:
        pass

    @abstractmethod
    def _load_transform(self, mode, rotation_degree) -> Callable:
        pass


class _EagerDataset(_AugmentedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = self._load_dataset(self.root)
        self.preprocess_and_cache_meta()

    def load_sample_descriptors(self):
        # Simply return a list of indices
        return list(range(len(self.dataset)))

    def get_data(self, desc):
        return self.transform(self.dataset[desc][0])

    def get_label(self, desc):
        return self.dataset[desc][1]


class _LazyDataset(_AugmentedDataset):
    """Load data lazily to save memory"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_path = f"{self._root}/{self.__class__.__name__}.pkl"
        self.data_dir = pathlib.Path(f"{self._root}/img")
        self.preprocess_and_cache_meta()

    def preprocess(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        dataset = self._load_dataset(self.root)
        file_list = []
        idx = 0
        for data, label in dataset:
            filename = f"{idx}_{label}"
            with open(self.data_dir.joinpath(filename), 'wb') as file:
                pickle.dump(data, file)
            file_list.append(filename)
            idx += 1
        save_obj(file_list, self.cache_path)

    def load_sample_descriptors(self):
        return load_obj(self.cache_path)

    def get_data(self, desc):
        with open(self.data_dir.joinpath(desc), 'rb') as file:
            img = pickle.load(file)
            return self.transform(img)

    def get_label(self, desc):
        return int(desc.split('_')[1])


def _load_cifar_transform(mode, rotation_degree):
    return transforms.Compose([
        transforms.RandomRotation(rotation_degree),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])
    ]) if mode == 'train' else \
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010])
        ])


def _load_cifar10_dataset(root):
    return ConcatDataset([
        datasets.CIFAR10(root=root, train=True, download=True),
        datasets.CIFAR10(root=root, train=False, download=True)])


def _load_cifar100_dataset(root):
    return ConcatDataset([
        datasets.CIFAR100(root=root, train=True, download=True),
        datasets.CIFAR100(root=root, train=False, download=True)])


def _load_mnist_transform(mode, rotation_degree):
    return transforms.Compose([
        transforms.RandomRotation(rotation_degree),
        transforms.RandomCrop(28, padding=4),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]) if mode == 'train' else \
        transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])


def _load_mnist_dataset(root):
    return ConcatDataset([
        datasets.MNIST(root=root, train=True, download=True),
        datasets.MNIST(root=root, train=False, download=True)])


class CIFAR10(_EagerDataset):
    """CIFAR10 dataset"""
    _root = 'data/src/cifar10'

    def _load_transform(self, mode, rotation_degree):
        return _load_cifar_transform(mode, rotation_degree)

    def _load_dataset(self, root):
        return _load_cifar10_dataset(root)


class CIFAR100(_EagerDataset):
    """CIFAR100 dataset"""
    _root = 'data/src/cifar100'

    def _load_transform(self, mode, rotation_degree):
        return _load_cifar_transform(mode, rotation_degree)

    def _load_dataset(self, root):
        return _load_cifar100_dataset(root)


class MNIST(_EagerDataset):
    """MNIST dataset"""
    _root = 'data/src/mnist'

    def _load_transform(self, mode, rotation_degree):
        return _load_mnist_transform(mode, rotation_degree)

    def _load_dataset(self, root):
        return _load_mnist_dataset(root)


class LazyCIFAR10(_LazyDataset):
    """CIFAR10 dataset, lazy version"""
    _root = 'data/src/lazy_cifar10'

    def _load_transform(self, mode, rotation_degree):
        return _load_cifar_transform(mode, rotation_degree)

    def _load_dataset(self, root):
        return _load_cifar10_dataset(root)


class LazyCIFAR100(_LazyDataset):
    """CIFAR100 dataset, lazy version"""
    _root = 'data/src/lazy_cifar100'

    def _load_transform(self, mode, rotation_degree):
        return _load_cifar_transform(mode, rotation_degree)

    def _load_dataset(self, root):
        return _load_cifar100_dataset(root)


class LazyMNIST(_LazyDataset):
    """MNIST dataset, lazy version"""
    _root = 'data/src/lazy_mnist'

    def _load_transform(self, mode, rotation_degree):
        return _load_mnist_transform(mode, rotation_degree)

    def _load_dataset(self, root):
        return _load_mnist_dataset(root)

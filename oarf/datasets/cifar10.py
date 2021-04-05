import random
import logging
from torchvision import datasets, transforms
from oarf.datasets.fl_dataset import RawDataset

log = logging.getLogger(__name__)


class CIFAR10(RawDataset):
    def __init__(self, mode, rotation_degree=5, *_, **__):
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomRotation(rotation_degree),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010])
            ])

    def load_train_dataset(self, _):
        return datasets.CIFAR10(
            root='data/src/cifar10', train=True, download=True)

    def load_eval_dataset(self, seed):
        data = list(datasets.CIFAR10(
            root='data/src/cifar10', train=False, download=True))
        random.Random(seed).shuffle(data)
        return data[:len(data) // 2]

    def load_test_dataset(self, seed):
        data = list(datasets.CIFAR10(
            root='data/src/cifar10', train=False, download=True))
        random.Random(seed).shuffle(data)
        return data[len(data) // 2:]

    def get_data(self, sample):
        return self.transform(sample[0])

    def get_label(self, sample):
        return sample[1]

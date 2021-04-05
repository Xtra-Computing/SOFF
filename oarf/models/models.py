import re
from torch import nn
from oarf.datasets.fl_dataset import FLDataset
from oarf.models.vgg import VGG16
from oarf.models.lstm import LSTM
from oarf.models.resnet import (
    ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)

models_name_map = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'lstm': LSTM,
    'vgg16': VGG16,
}


def init_model(name, batchnorm_runstat, dataset: FLDataset, **kwargs):
    """
    batchnorm_runstat: for models that uses batchnorm layer
    dataset: for reading infos from dataset, e.g. lstm embedding dim
    """
    name = name.lower()
    if name == 'lstm':
        return models_name_map[name](
            output_size=dataset.num_classes(),
            embedding_dim=dataset.datasets[0].embedding_dim(),
            **kwargs)
    elif re.match(r'^resnet.*', name):
        return models_name_map[name](
            num_classes=dataset.num_classes(), **kwargs)
    elif re.match(r'^vgg*', name):
        return models_name_map[name](
            num_classes=dataset.num_classes(), **kwargs)
    else:
        raise Exception("Unrecognized dataset.")


loss_criterion_name_map = {
    'resnet18': nn.CrossEntropyLoss,
    'resnet34': nn.CrossEntropyLoss,
    'resnet50': nn.CrossEntropyLoss,
    'resnet101': nn.CrossEntropyLoss,
    'resnet152': nn.CrossEntropyLoss,
    'lstm': nn.BCELoss,
    'vgg16': nn.CrossEntropyLoss,
}


def init_loss_criterion(name):
    return loss_criterion_name_map[name.lower()]()

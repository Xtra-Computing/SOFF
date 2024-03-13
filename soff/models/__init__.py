"""FL models"""
import re
import copy
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
from torch import nn, Tensor
from munch import Munch
from torch.utils.data import DataLoader
from .base import _ModelTrainer
from ..utils.metrics import _MetricCreator
from ..datasets.fl_split.base import _FLSplit
from ..utils.arg_parser import BaseConfParser, HWConfParser, r_getattr
from ..utils.logging import DataLogger
from ..utils.module import (
    camel_to_snake, load_all_direct_classes, load_all_submodules)

# Models ######################################################################

models_name_map: Dict[str, Type] = {
    camel_to_snake(name).replace('_', '-'): cls
    for module in load_all_submodules()
    for name, cls in load_all_direct_classes(module).items()
    if not name.startswith('_') and issubclass(cls, nn.Module)
}


class ModelConfParser(BaseConfParser):
    """Parse arguments for various machine learning models"""

    def __init__(self, *args, stag='', ltag='', **kwargs):
        super().__init__(*args, **kwargs)

        model_args = self.add_argument_group(
            "Model Initialization Configs (S,S->C)")

        # Configure the model itself
        model_args.add_argument(
            f'-md{stag}.n', f'--model{ltag}.name',
            default='resnet18', type=str, metavar='NAME',
            choices=models_name_map.keys(),
            help="Machine learning model to use. "
            f"Available models: {list(models_name_map.keys())}")

        for cls in chain.from_iterable(
                load_all_direct_classes(module).values()
                for module in load_all_submodules()):
            if not hasattr(cls, 'argparse_options'):
                continue
            cls.add_options_to(self, pfxs=('md', 'model'), tags=(stag, ltag))

            for option in cls.argparse_options():
                self.register_cfg_dep(
                    f'model{ltag}.{option.flags[1]}',
                    lambda cfg, cls=cls: issubclass(models_name_map[r_getattr(
                        cfg, f"model{ltag.replace('/', '.')}.name")], cls))

        # Configure the loss function
        model_args.add_argument(
            f'-md{stag}.l.n', f'--model{ltag}.loss.name',
            default='ce', type=str, metavar='NAME',
            choices=loss_criterion_name_map.keys(),
            help="Loss function to use. "
            f"Available functions: {list(loss_criterion_name_map.keys())}")

        # Programmatically add arguments from torch.optim
        def shorten(key):
            return re.sub(r'(?<!^)[aeiou_]', '', key)

        for name, (_, params) in loss_criterion_name_map.items():
            for key, _type, default, _help in params:
                model_args.add_argument(
                    f"-md{stag}.l.{shorten(name)}.{shorten(key)}",
                    f"--model{ltag}.loss.{name}.{key}", type=_type,
                    default=default, metavar=f"{key}".upper(),
                    help=_help)

            self.register_cfg_dep(
                f"model{ltag}.loss.{name}",
                lambda cfg, name=name: r_getattr(
                    cfg, f"model{ltag.replace('/','.')}.loss.name"
                ) == name)


def create_model(cfg: Munch, dataset: _FLSplit, tag: str = ''):
    """
    dataset: for reading dataset-dependent infos,
        e.g. number of classes, lstm embedding dim, etc.
    """
    cfg = copy.deepcopy(cfg)
    cfg.data = cfg.model[tag] if tag else cfg.model
    return models_name_map[cfg.model.name.lower()](cfg, dataset)


# Model Trainers ##############################################################

model_trainer_name_map = {
    camel_to_snake(name[:-len('Trainer')] if name.endswith('Trainer') else name).replace('_', '-'): cls
    for module in load_all_submodules()
    for name, cls in load_all_direct_classes(module).items()
    if not name.startswith('_') and issubclass(cls, _ModelTrainer)
}


class ModelTrainerConfParser(HWConfParser):
    def __init__(self, *args, stag='', ltag='', **kwargs):
        super().__init__(*args, **kwargs)

        model_trainer_args = self.add_argument_group("Model Trainer Configs")
        model_trainer_args.add_argument(
            f'-tr.mt{stag}.n', f'--training.model-trainer{ltag}.name',
            type=str, default='per-epoch', metavar='NAME',
            choices=model_trainer_name_map.keys(),
            help="Name of the model trainer. "
            f"Available: {list(model_trainer_name_map.keys())}")


def create_model_trainer(
        cfg: Munch, data_loader: DataLoader,
        loss_fn: Callable, metrics: List[_MetricCreator],
        datalogger: Optional[DataLogger] = None, tag: str = ''):
    """Create a model trainer"""

    cfg = copy.deepcopy(cfg)
    cfg.training.model_trainer = (
        cfg.training.model_trainer[tag] if tag else cfg.training.model_trainer)
    return model_trainer_name_map[cfg.training.model_trainer.name](
        cfg, data_loader, loss_fn, metrics, datalogger)


# Loss criterions #############################################################


# fmt: off
loss_criterion_name_map: Dict[str, Tuple[Callable, Iterable[Tuple[str, Type, Any, str]]]] = {
    'l1': (nn.L1Loss, (
        ('reduction', str, 'mean', "Specifies the reduction to apply to the output."),
    )),
    'mse': (nn.MSELoss, (
        ('reduction', str, 'mean', "Specifies the reduction to apply to the output."),
    )),
    'ce': (nn.CrossEntropyLoss, (
        ('weight', Tensor, None, "A manual rescaling weight given to each class. If given, has to be a Tensor of size C"),
        ('ignore_index', int, -100, "Specifies a target value that is ignored and does not contribute to the input gradient."),
        ('reduction', str, 'mean', "Specifies the reduction to apply to the output."),
        ('label_smoothing', float, 0.0, "Specifies the amount of smoothing when computing the loss.")
    )),
    'bce': (nn.BCELoss, (
        ('weight', Tensor, None, "A manual rescaling weight given to each class. If given, has to be a Tensor of size C"),
        ('reduction', str, 'mean', "Specifies the reduction to apply to the output."),
    )),
    'bcewl': (nn.BCEWithLogitsLoss, (
        ('weight', Tensor, None, "A manual rescaling weight given to each class. If given, has to be a Tensor of size C"),
        ('reduction', str, 'mean', "Specifies the reduction to apply to the output."),
        ('pos_weight', Tensor, None, "A weight of positive examples to be broadcasted with target."),
    )),
}
# fmt: on


def create_loss_criterion(cfg: Munch, tag: str = '', **kwargs):
    """Crate loss calculator for backward propagation"""
    if tag:
        cfg.model.loss = cfg.model[tag].loss
    return loss_criterion_name_map[cfg.model.loss.name][0](**{
        **cfg.model.loss[cfg.model.loss.name],
        **kwargs
    })

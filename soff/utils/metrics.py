"""A list of metrics for training"""
import re
import copy
import argparse
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type
from torcheval import metrics
from munch import Munch
from .arg_parser import TrainingConfParser, r_getattr


class MetricsConfParser(TrainingConfParser):
    """Parser for optimizer config"""

    def __init__(self, *args, stag='', ltag='', **kwargs):
        super().__init__(*args, **kwargs)

        optimizer_args = self.add_argument_group(
            "Metrics Configs (S,S->C)")

        optimizer_args.add_argument(
            f'-tr.m{stag}.ns', f'--training.metric{ltag}.names',
            default=['accuracy'], type=str, nargs='+', metavar='NAME',
            choices=metrics_name_map.keys(),
            help="Metrics for both server and clients. Available metrics: "
            f"{list(metrics_name_map.keys())}")

        # Programmatically add arguments from torch.optim
        def shorten(key):
            return re.sub(r'(?<!^)[aeiou_]', '', key)

        for name, (_, params) in metrics_name_map.items():
            for key, _type, default, _help in params:
                optimizer_args.add_argument(
                    f"-tr.m{stag}.{shorten(name)}.{shorten(key)}",
                    f"--training.metric{ltag}.{name}.{key}", type=_type,
                    default=default, metavar=f"{key}".upper(),
                    help=_help)

            self.register_cfg_dep(
                f"training.metric{ltag}.{name}",
                lambda cfg, name=name: name in r_getattr(
                    cfg, f"training.metric{ltag.replace('/','.')}.names"))


# fmt: off
metrics_name_map: Dict[str, Tuple[Callable, Iterable[Tuple[str, Type, Any, str]]]] = {
    'baccuracy': (metrics.BinaryAccuracy, (
        ('threshold', float, 0.5, "Binary accuarcy threshold"),
    )),
    'bauroc': (metrics.BinaryAUROC, (
        ('num_tasks', int, 1, "Number of tasks"),
        ('use_fbgemm', bool, False, "Use FBGEMM")
    )),
    'accuracy': (metrics.MulticlassAccuracy, (
        ('average', str, 'micro', "Average policy"),
        ('num_classes', int, None, "Number of classes. Required for 'macro' and None average methods."),
        ('k', int, 1, "Number of top probabilities to be considered. K should be an integer greater than or equal to 1."),
    )),
    'auroc': (metrics.MulticlassAUROC, (
        ('num_classes', int, argparse.SUPPRESS, "Number of classes. Required for 'macro' and None average methods."),
        ('average', str, 'macro', "Average policy"),
    )),
    'mse': (metrics.MeanSquaredError, (
        ('multioutput', str, 'uniform_average', "uniform_average | raw_values."),
    )),
    'f1': (metrics.MulticlassF1Score, (
        ('num_classes', int, None, "Number of classes."),
        ('average', str, 'micro', "Average policy.")
    )),
    'ppl': (metrics.Perplexity, (
        ('ignore_index', int, None, "if specified, the target class with ‘ignore_index’ will be ignored when calculating perplexity."),
    )),
    # To add metrics, ensure that the added classs have `update` and `compute`
    # methods. The `update` methods should take (pred, label) as args, while
    # the `compute` methods take no additional args other than `self`
}
# fmt: on


class _MetricCreator:
    def __init__(self, name, **kwargs):
        assert name in metrics_name_map
        assert hasattr(metrics_name_map[name][0], 'update')
        assert hasattr(metrics_name_map[name][0], 'compute')

        self.name = name
        self.metric_class = metrics_name_map[name][0]
        self.kwargs = kwargs

    def __call__(self):
        return self.metric_class(**self.kwargs)


def create_metrics(cfg: Munch, tag: str = '', **kwargs) -> List[_MetricCreator]:
    """
    Create list of metrics creators. If `kwargs` are provided,
    they will override corresponding options specified in `cfg`.

    Creators instead metroc objects are returned to avoid re-using the same
    metric object. Re-using the same metric object may lead to false evaluation.
    """

    cfg = copy.deepcopy(cfg)
    if tag:
        cfg.training.metric = cfg.training.metric[tag]
    return [_MetricCreator(name, **{**cfg.training.metric[name], **kwargs})
            for name in cfg.training.metric.names]

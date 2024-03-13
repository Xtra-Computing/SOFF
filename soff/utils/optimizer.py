"""Create optimizers"""
import re
import copy
from typing import Any, Dict, Callable, Iterable, Tuple, Type
import torch
from munch import Munch
from .arg_parser import TrainingConfParser, r_getattr


class OptimizerConfParser(TrainingConfParser):
    """Parser for optimizer config"""

    def __init__(self, *args, stag='', ltag='', **kwargs):
        super().__init__(*args, **kwargs)

        optimizer_args = self.add_argument_group(
            "Optimizer Configs (S,S->C)")

        optimizer_args.add_argument(
            f'-tr.o{stag}.n', f'--training.optimizer{ltag}.name',
            default='sgd', type=str, metavar='NAME',
            choices=optimizer_name_map.keys(),
            help="Optimizer for local training. \033[1mNOTE:\033[0m "
            "Detailed help messages for per-optimizer options are suppressed "
            "to avoid redundancy. The supported args are the same as pytorch's "
            "respective optimizer's initialization arguments (except for lr, "
            "which is initialized from --training.learning-rate). You can use "
            "--training.optimizer.<optimizer-name>.<arg-name> <arg-value> to "
            "specify them. E.g. you can specify "
            "--training.optimizer.sgd.momentum 0.9 to set a momentum of 0.9 "
            "for the sgd optimizer. or specify -tr.o.adam.bts 0.9 0.99 "
            "to specify a beta of (0.9, 0.99) for the adam optimizer")

        # Programmatically add arguments from torch.optim
        def shorten(key):
            return re.sub(r'(?<!^)[aeiou_]', '', key)

        for name, (_, params) in optimizer_name_map.items():
            for key, _type, default, _help in params:
                # Fix option types
                optimizer_args.add_argument(
                    f"-tr.o{stag}.{shorten(name)}.{shorten(key)}",
                    f"--training.optimizer{ltag}.{name}.{key}", type=_type,
                    default=default, metavar=f"{key}".upper(),
                    help=_help)

            self.register_cfg_dep(
                f"training.optimizer{ltag}.{name}",
                lambda cfg, optim=name: r_getattr(
                    cfg, f"training.optimizer{ltag.replace('/', '.')}.name"
                ) == optim)


# fmt: off
optimizer_name_map: Dict[str, Tuple[Callable, Iterable[Tuple[str, Type, Any, str]]]] = {
    'sgd': (torch.optim.SGD, (
        ('momentum', float, 0, "momentum factor"),
        ('weight_decay', float, 0, "weight decay (L2 penalty)"),
        ('dampening', float, 0, "dampening for momentum"),
        ('nesterov', bool, False, "enables Nesterov momentum"),
        ('maximize', bool, False, "maximize the params based on the objective, instead of minimizing"),
        ('foreach', bool, None, "whether foreach implementation of optimizer is used."),
        ('differentiable', bool, False, "whether autograd should occur through the optimizer step in training. "),
    )),
    'adam': (torch.optim.Adam, (
        ('betas', Tuple[float, float], (0.9, 0.999), "coefficients used for computing running averages of gradient and its square"),
        ('eps', float, 1e-8, "term added to the denominator to improve numerical stability"),
        ('weight_decay', float, 0, "weight decay (L2 penalty)"),
        ('amsgrad', bool, False, "whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond"),
        ('foreach', bool, None, "whether foreach implementation of optimizer is used."),
        ('maximize', bool, False, "maximize the params based on the objective, instead of minimizing"),
        ('capturable', bool, False, "whether this instance is safe to capture in a CUDA graph."),
        ('differentiable', bool, False, "whether autograd should occur through the optimizer step in training."),
        ('fused', bool, None, "whether the fused implementation (CUDA only) is used."),
    )),
    'asgd': (torch.optim.ASGD, (
        ('lambd', float, 1e-4, "decay term"),
        ('alpha', float, 0.75, "power for eta update"),
        ('t0', float, 1e6, "point at which to start averaging"),
        ('weight_decay', float, 0, "weight decay (L2 penalty)"),
        ('foreach', bool, None, "whether foreach implementation of optimizer is used."),
        ('maximize', bool, False, "maximize the params based on the objective, instead of minimizing"),
        ('differentiable', bool, False, "whether autograd should occur through the optimizer step in training."),
    )),
    'adamw': (torch.optim.AdamW, (
        ('betas', Tuple[float, float], (0.9, 0.999), "coefficients used for computing running averages of gradient and its square"),
        ('eps', float, 1e-8, "term added to the denominator to improve numerical stability"),
        ('weight_decay', float, 1e-2, "weight decay coefficient"),
        ('amsgrad', bool, False, "whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond"),
        ('maximize', bool, False, "maximize the params based on the objective, instead of minimizing"),
        ('foreach', bool, None, "whether foreach implementation of optimizer is used."),
        ('capturable', bool, False, "whether this instance is safe to capture in a CUDA graph."),
        ('differentiable', bool, False, "whether autograd should occur through the optimizer step in training."),
        ('fused', bool, None, "whether the fused implementation (CUDA only) is used."),
    )),
    'lbfgs': (torch.optim.LBFGS, (
        ('max_iter', int, 20, "maximal number of iterations per optimization step"),
        ('max_eval', int, 25, "maximal number of function evaluations per optimization step"),
        ('tolerance_grad', float, 1e-5, "termination tolerance on first order optimality"),
        ('tolerance_change', float, 1e-9, "termination tolerance on function value/parameter changes"),
        ('history_size', int, 100, "update history size"),
        ('line_search_fn', str, None, "either ‘strong_wolfe’ or None"),
    )),
    'nadam': (torch.optim.NAdam, (
        ('betas', Tuple[float, float], (0.9, 0.999), "coefficients used for computing running averages of gradient and its square"),
        ('eps', float, 1e-8, "term added to the denominator to improve numerical stability"),
        ('weight_decay', float, 0, "weight decay (L2 penalty)"),
        ('momentum_decay', float, 4e-3, "momentum momentum_decay"),
        ('foreach', bool, None, "whether foreach implementation of optimizer is used."),
        ('differentiable', bool, False, "whether autograd should occur through the optimizer step in training."),
    )),
    'radam': (torch.optim.RAdam,(
        ('betas', Tuple[float, float], (0.9, 0.999), "coefficients used for computing running averages of gradient and its square"),
        ('eps', float, 1e-8, "term added to the denominator to improve numerical stability"),
        ('weight_decay', float, 0, "weight decay (L2 penalty)"),
        ('momentum_decay', float, 4e-3, "momentum momentum_decay"),
        ('foreach', bool, None, "whether foreach implementation of optimizer is used."),
        ('differentiable', bool, False, "whether autograd should occur through the optimizer step in training."),
    )),
    'rprop': (torch.optim.Rprop,(
        ('etas', Tuple[float, float], (0.5, 1.2), "pair of (etaminus, etaplus), that are multiplicative increase and decrease factors"),
        ('step_sizes', Tuple[float, float], (1e-6, 50), "a pair of minimal and maximal allowed step sizes"),
        ('foreach', bool, None, "whether foreach implementation of optimizer is used."),
        ('maximize', bool, False, "maximize the params based on the objective, instead of minimizing"),
        ('differentiable', bool, False, "whether autograd should occur through the optimizer step in training."),
    )),
    'adamax': (torch.optim.Adamax,(
        ('betas', Tuple[float, float], (0.9, 0.999), "coefficients used for computing running averages of gradient and its square"),
        ('eps', float, 1e-8, "term added to the denominator to improve numerical stability"),
        ('weight_decay', float, 0, "weight decay (L2 penalty)"),
        ('foreach', bool, None, "whether foreach implementation of optimizer is used."),
        ('maximize', bool, False, "maximize the params based on the objective, instead of minimizing"),
        ('differentiable', bool, False, "whether autograd should occur through the optimizer step in training."),
    )),
    'adagrad': (torch.optim.Adagrad,(
        ('lr_decay', float, 0, "learning rate decay"),
        ('weight_decay', float, 0, "weight decay (L2 penalty)"),
        ('eps', float, 1e-8, "term added to the denominator to improve numerical stability"),
        ('foreach', bool, None, "whether foreach implementation of optimizer is used."),
        ('maximize', bool, False, "maximize the params based on the objective, instead of minimizing"),
        ('differentiable', bool, False, "whether autograd should occur through the optimizer step in training."),
    )),
    'rmsprop': (torch.optim.RMSprop,(
        ('momentum', float,0, "momentum factor"),
        ('alpha', float, 0.99, "smoothing constant"),
        ('eps', float, 1e-8, "term added to the denominator to improve numerical stability"),
        ('centered', bool, False, "if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance"),
        ('weight_decay', float, 0, "weight decay (L2 penalty)"),
        ('foreach', bool, None, "whether foreach implementation of optimizer is used."),
        ('maximize', bool, False, "maximize the params based on the objective, instead of minimizing"),
        ('differentiable', bool, False, "whether autograd should occur through the optimizer step in training."),
    )),
    'adadelta': (torch.optim.Adadelta,(
        ('rho', float, 0.9, "coefficient used for computing a running average of squared gradients"),
        ('eps', float, 1e-6, "term added to the denominator to improve numerical stability"),
        ('weight_decay', float, 0, "weight decay (L2 penalty)"),
        ('foreach', bool, None, "whether foreach implementation of optimizer is used."),
        ('maximize', bool, False, "maximize the params based on the objective, instead of minimizing"),
        ('differentiable', bool, False, "whether autograd should occur through the optimizer step in training."),
    )),
    'sparseadam': (torch.optim.SparseAdam,(
        ('betas', Tuple[float, float], (0.9, 0.999), "coefficients used for computing running averages of gradient and its square"),
        ('eps', float, 1e-8, "term added to the denominator to improve numerical stability"),
        ('maximize', bool, False, "maximize the params based on the objective, instead of minimizing"),
    )),
}
# fmt: on


def create_optimizer(cfg: Munch, params, tag: str = '', **kwargs):
    """
    Create an optimizer. parameters to optimize must be explicitly provied.
    For other arguments, if `kwargs` are provided, they will override
    corresponding options specified in `cfg`.

    By default, Learning rate is read from `cfg.training.learning_rate`, and
    other options are read from `cfg.training.optimizer.<name>`
    """
    cfg = copy.deepcopy(cfg)
    if tag:
        cfg.training.optimizer = cfg.training.optimizer[tag]
    return optimizer_name_map[cfg.training.optimizer.name][0](params, **{
        'lr': cfg.training.learning_rate,
        **cfg.training.optimizer[cfg.training.optimizer.name],
        **kwargs})

"""Creating learning rate schedulers"""
import re
import copy
import argparse
from typing import Any, Dict, Callable, Iterable, Tuple, Type
from munch import Munch
from torch.optim import lr_scheduler, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from ..utils.arg_parser import TrainingConfParser, r_getattr


class LRSchedulerConfParser(TrainingConfParser):
    """Parser for optimizer config"""

    def __init__(self, *args, stag='', ltag='', **kwargs):
        super().__init__(*args, **kwargs)

        scheduler_args = self.add_argument_group(
            "Learning Rate Scheduler Configs (S,S->C)")

        scheduler_args.add_argument(
            f'-tr.s{stag}.n', f'--training.scheduler{ltag}.name',
            default='reduceonplateau', type=str, metavar='NAME',
            choices=scheduler_name_map.keys(),
            help="Optimizer for local training. \033[1mNOTE:\033[0m "
            "Detailed help messages for per-scheduler options are suppressed "
            "to avoid redundancy. The supported args are the same as pytorch's "
            "respective scheduler's initialization arguments. You can use "
            "--training.scheduler.<optimizer-name>.<arg-name> <arg-value> to "
            "specify them. E.g. you can specify "
            "--training.scheduler.linear.total_iters 1000 to set iter of 1000 "
            "for the linear scheduler. or specify -tr.s.mltstp.gmm 0.2 "
            "to specify a gamma of 0.2 for the multistep scheduler.")

        # These are shceudlar args
        scheduler_args.add_argument(
            f'-tr.s{stag}.wd', f'--training.scheduler{ltag}.warmup-duration',
            default=0, type=int, metavar='N',
            help="Number of scheduler steps for warmup. If greater than 0, "
            "`create_scheduler` will wrap the scheduler specified by "
            "--training.scheduler.name into a linear warmup scheduler.")

        # Programmatically add arguments from torch.optim.lr_scheduler
        def shorten(key):
            return re.sub(r'(?<!^)[aeiou_]', '', key)

        for name, (_, params) in scheduler_name_map.items():
            for key, _type, default, _help in params:
                scheduler_args.add_argument(
                    f"-tr.s{stag}.{shorten(name)}.{shorten(key)}",
                    f"--training.scheduler{ltag}.{name}.{key}",
                    type=_type[0] if isinstance(_type, list) else _type,
                    **dict((('nargs', '+'),) if isinstance(_type, list) else ()),
                    default=default, metavar=f"{key}".upper(),
                    help=_help)

            self.register_cfg_dep(
                f"training.scheduler{ltag}.{name}",
                lambda cfg, sched=name: r_getattr(
                    cfg, f"training.scheduler{ltag.replace('/', '.')}.name"
                ) == sched)


# fmt: off
scheduler_name_map: Dict[str, Tuple[Callable, Iterable[Tuple[str, Type, Any, str]]]] = {
    'step': (lr_scheduler.StepLR, (
        ('step_size', int, argparse.SUPPRESS, "Period of learning rate decay."),
        ('gamma', float, 0.1, "Multiplicative factor of learning rate decay."),
        ('last_epoch', int, -1, "The index of last epoch."),
        ('verbose', bool, False, "If True, prints a message to stdout for each update."),
    )),
    'cyclic': (lr_scheduler.CyclicLR, (
        ('base_lr', [float], argparse.SUPPRESS, "Initial learning rate which is the lower boundary in the cycle for each parameter group."),
        ('max_lr', [float], argparse.SUPPRESS, "Upper learning rate boundaries in the cycle for each parameter group."),
        ('step_size_up', int, 2000, "Number of training iterations in the increasing half of a cycle."),
        ('step_size_down', int, None, "Number of training iterations in the decreasing half of a cycle."),
        ('mode', str, 'triangular', "One of {triangular, triangular2, exp_range}. Values correspond to policies detailed above."),
        ('gamma', float, 1.0, "Constant in ‘exp_range’ scaling function: gamma**(cycle iterations)"),
        ('scale_fn', Callable, None, "Custom scaling policy defined by a single argument lambda function, where 0 <= scale_fn(x) <= 1 for all x >= 0."),
        ('scale_mode', str, 'cycle', "{‘cycle’, ‘iterations’}. Defines whether scale_fn is evaluated on cycle number or cycle iterations."),
        ('cycle_momentum', bool, True, "If True, momentum is cycled inversely to learning rate between ‘base_momentum’ and ‘max_momentum’."),
        ('base_momentum', [float], 0.8, "Lower momentum boundaries in the cycle for each parameter group. Note that momentum is cycled inversely to learning rate."),
        ('max_momentum', [float], 0.9, "Upper momentum boundaries in the cycle for each parameter group."),
        ('last_epoch', int, -1, "The index of the last batch. This parameter is used when resuming a training job."),
                ('verbose', bool, False, "If True, prints a message to stdout for each update."),
    )),
    'linear': (lr_scheduler.LinearLR, (
        ('start_factor', float, 1./3, "The number we multiply learning rate in the first epoch."),
        ('end_factor', float, 1.0, "The number we multiply learning rate at the end of linear changing process."),
        ('total_iters', int, 5, "The number of iterations that multiplicative factor reaches to 1."),
        ('last_epoch', int, -1, "The index of the last epoch."),
        ('verbose', bool, False, "If True, prints a message to stdout for each update."),
    )),
    'constant': (lr_scheduler.ConstantLR, (
        ('factor', float, 1./3, "The number we multiply learning rate until the milestone."),
        ('total_iters', int, 5, "The number of steps that the scheduler decays the learning rate."),
        ('last_epoch', int, -1, "The index of the last epoch."),
        ('verbose', bool, False, "If True, prints a message to stdout for each update."),
    )),
    'onecycle': (lr_scheduler.OneCycleLR, (
        ('max_lr', [float], argparse.SUPPRESS, "Upper learning rate boundaries in the cycle for each parameter group."),
        ('total_steps', int, None, "The total number of steps in the cycle."),
        ('epochs', int, None, "The number of epochs to train for."),
        ('steps_per_epoch', int, None, "The number of steps per epoch to train for."),
        ('pct_start', float, 0.3, "The percentage of the cycle (in number of steps) spent increasing the learning rate."),
        ('anneal_strategy', str, 'cos', "{‘cos’, ‘linear’} Specifies the annealing strategy."),
        ('cycle_momentum', bool, True, "If True, momentum is cycled inversely to learning rate between ‘base_momentum’ and ‘max_momentum’."),
        ('base_momentum', [float], 0.85, "Lower momentum boundaries in the cycle for each parameter group. "),
        ('max_momentum', [float], 0.95, "Upper momentum boundaries in the cycle for each parameter group."),
        ('div_factor', float, 25, "Determines the initial learning rate via initial_lr = max_lr/div_factor"),
        ('final_div_factor', float, 1e4, "Determines the minimum learning rate via min_lr = initial_lr/final_div_factor"),
        ('three_phase', bool, False, "If True, use a third phase of the schedule to annihilate the learning rate according to ‘final_div_factor’)"),
        ('last_epoch', int, -1, "The index of the last batch."),
        ('verbose', bool, False, "If True, prints a message to stdout for each update."),
    )),
    'multistep': (lr_scheduler.MultiStepLR, (
        ('milestones', [int], argparse.SUPPRESS, "List of epoch indices. Must be increasing"),
        ('gamma', float, 0.1, "Multiplicative factor of learning rate decay."),
        ('last_epoch', int, -1, "The index of last epoch."),
        ('verbose', bool, False, "If True, prints a message to stdout for each update."),
    )),
    'exponential': (lr_scheduler.ExponentialLR, (
        ('gamma', float, argparse.SUPPRESS, "Multiplicative factor of learning rate decay."),
        ('last_epoch', int, -1, "The index of last epoch."),
        ('verbose', bool, False, "If True, prints a message to stdout for each update."),
    )),
    'multiplicative': (lr_scheduler.MultiplicativeLR, (
        ('lr_lambda', Tuple[Callable,list], argparse.SUPPRESS, "A function which computes a multiplicative factor given an integer parameter epoch."),
        ('last_epoch', int, -1, "The index of last epoch."),
        ('verbose', bool, False, "If True, prints a message to stdout for each update."),
    )),
    'cosineannealing': (lr_scheduler.CosineAnnealingLR, (
        ('T_max', int, argparse.SUPPRESS, "Maximum number of iterations."),
        ('eta_min', float, 0, "Minimum learning rate."),
        ('last_epoch', int, -1, "The index of last epoch."),
        ('verbose', bool, False, "If True, prints a message to stdout for each update."),
    )),
    'reduceonplateau': (lr_scheduler.ReduceLROnPlateau, (
        ('mode', str, 'min', "One of min, max."),
        ('factor', float, 0.1, "Factor by which the learning rate will be reduced. new_lr = lr * factor."),
        ('patience', int, 10, "Number of epochs with no improvement after which learning rate will be reduced. "),
        ('threshold', float, 1e-4, "Threshold for measuring the new optimum, to only focus on significant changes."),
        ('threshold_mode', str, 'rel', "One of rel, abs."),
        ('cooldown', int, 0, "Number of epochs to wait before resuming normal operation after lr has been reduced."),
        ('min_lr', [float], 0, "A scalar or a list of scalars."),
        ('eps', float, 1e-8, "Minimal decay applied to lr."),
        ('verbose', bool, False, "If True, prints a message to stdout for each update."),
    )),

    # The following schedulers are not included (they are composite schedulers)
    # 'chained': lr_scheduler.ChainedScheduler,
    # 'sequential': lr_scheduler.SequentialLR,
    # 'lambda': lr_scheduler.LambdaLR,
}
# fmt: on


def create_scheduler(cfg: Munch, optimizer, tag: str = '', **kwargs):
    """
    Create a scheduler. optimizer to schedule must be explicitly provided.
    For other arguments, if `kwargs` are provided, they will override
    corresponding arguments specified in `cfg`.

    By default, options are read from `cfg.training.scheduler.<name>`
    """

    cfg = copy.deepcopy(cfg)
    if tag:
        cfg.training.scheduler = cfg.training.scheduler[tag]
    scheduler = scheduler_name_map[cfg.training.scheduler.name][0](optimizer, **{
        **cfg.training.scheduler[cfg.training.scheduler.name],
        **kwargs
    })

    warmup_duration = cfg.training.scheduler.warmup_duration
    assert warmup_duration >= 0, "Warmup duration must >= 0"

    return (GradualWarmupScheduler(
        optimizer, warmup_duration=cfg.training.scheduler.warmup_duration,
        after_scheduler=scheduler)
        if cfg.training.scheduler.warmup_duration > 0
        else scheduler)

    # Use SequentialLR also works but produces a deprecation warning:
    # return lr_scheduler.SequentialLR(
    #     optimizer, schedulers=[
    #         lr_scheduler.LinearLR(
    #             optimizer, start_factor=1./warmup_duration,
    #             end_factor=1., total_iters=warmup_duration),
    #         scheduler
    #     ], milestones=[warmup_duration]
    # ) if cfg.training.scheduler.warmup_duration > 0 else scheduler


class GradualWarmupScheduler(_LRScheduler):
    """
    Gradually warm-up(increasing) learning rate in optimizer.
    adapted from https://github.com/ildoonet/pytorch-gradual-warmup-lr
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_duration: duration for the learning rate to finish warmup
        after_scheduler: this scheduler is used after warmup finishs.
    """

    def __init__(self, optimizer: Optimizer, warmup_duration: int,
                 after_scheduler: _LRScheduler):
        self.warmup_duration = warmup_duration
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        """Overrides super().get_lr()"""
        if self.last_epoch > self.warmup_duration:
            if not self.finished:
                self.after_scheduler.base_lrs = copy.deepcopy(self.base_lrs)
                self.finished = True
            return self.after_scheduler.get_last_lr()

        return [base_lr * (float(self.last_epoch) / self.warmup_duration)
                for base_lr in self.base_lrs]

    def step_reduce_lr_on_plateau(self, metrics):
        """
        ReduceLROnPlateau is called at the end of epoch, whereas others are
        called at beginning
        """
        epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.warmup_duration:
            warmup_lr = [
                base_lr * (float(self.last_epoch) / self.warmup_duration)
                for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            self.after_scheduler.step(metrics)

    def step(self, epoch=None, metrics=None):
        if isinstance(self.after_scheduler, lr_scheduler.ReduceLROnPlateau):
            self.step_reduce_lr_on_plateau(metrics)
        else:
            if self.finished:
                self.after_scheduler.step(
                    epoch - self.warmup_duration if epoch else None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                super().step(epoch)

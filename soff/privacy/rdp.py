"""RDP utilities"""
import os
import sys
import math
import logging
from typing import Optional
import torch
import dp_accounting
from torch import nn
from munch import Munch
from scipy.optimize import bisect
from ..utils.arg_parser import BaseConfParser

log = logging.getLogger(__name__)


class DPConfParser(BaseConfParser):
    """Parse configs for the differential privacy module"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        c_dp_args = self.add_argument_group(
            "DP-related Configs (S,S->C)")
        c_dp_args.add_argument(
            '-dp.dt', '--differential-privacy.dp-type',
            default=None, choices=[None, 'rdp'],
            help="Type of differential privacy to use")

        c_dp_args.add_argument(
            '-dp.r.ep', '--differential-privacy.rdp.epsilon',
            default=2.0, type=float, metavar='ε',
            help="ε value for rdp")
        c_dp_args.add_argument(
            '-dp.r.de', '--differential-privacy.rdp.delta',
            default=0, type=float, metavar='δ',
            help="δ value for rdp")
        c_dp_args.add_argument(
            '-dp.r.cl', '--differential-privacy.rdp.clip',
            default=5.0, type=float, metavar='CLIP',
            help="Gradient clipping value for DP")
        self.register_cfg_dep(
            '--differential-privacy.rdp',
            lambda cfg: cfg.differential_privacy.dp_type == 'rdp')


def compute_dp_sgd_example_privacy(
    num_epochs: float,
    noise_multiplier: float,
    example_delta: float,
    used_microbatching: bool = True,
    client_fraction: float = 1.,
) -> float:
    """From: tensorflow_privacy
    Computes add-or-remove-one-example DP epsilon.

    This privacy guarantee uses add-or-remove-one-example adjacency, and protects
    release of all model checkpoints in addition to the final model.

    Returns:
      The epsilon value.
    """
    if num_epochs <= 0:
        raise ValueError(f'num_epochs must be positive. Found {num_epochs}.')
    if noise_multiplier < 0:
        raise ValueError(
            f'noise_multiplier must be non-negative. Found {noise_multiplier}.'
        )
    if not 0 <= example_delta <= 1:
        raise ValueError(
            f'delta must be between 0 and 1. Found {example_delta}.')

    if used_microbatching:
        noise_multiplier /= 2

    event = dp_accounting.GaussianDpEvent(noise_multiplier)
    event = dp_accounting.PoissonSampledDpEvent(client_fraction, event=event)
    count = int(math.ceil(num_epochs))
    event = dp_accounting.SelfComposedDpEvent(event=event, count=count)

    return (
        dp_accounting.rdp.RdpAccountant()
        .compose(event).get_epsilon(example_delta))


def compute_gaussian_sigma(cfg: Munch, data_size: int) -> float:
    delta = (
        min(1e-5, 1. / data_size)
        if cfg.differential_privacy.rdp.delta == 0
        else cfg.differential_privacy.rdp.delta)
    epsilon = cfg.differential_privacy.rdp.epsilon
    epochs = cfg.training.epochs
    client_fraction = cfg.fedavg.client_fraction if 'fedavg' in cfg else 1.

    # poisson_subsampling_probability = batch_size / number_of_examples,
    """Compute the level of noise to add when using rdp"""
    def compute_dp_sgd_wrapper(sigma):
        return compute_dp_sgd_example_privacy(
            num_epochs=epochs, noise_multiplier=sigma,
            example_delta=delta, used_microbatching=True,
            client_fraction=client_fraction) - epsilon

    # turn off output
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # calculate sigma
    sigma = bisect(compute_dp_sgd_wrapper, 1e-6, 1e6)

    # calculte actual privacy budget
    actual_epsilon = compute_dp_sgd_example_privacy(
        num_epochs=epochs, noise_multiplier=sigma,
        example_delta=delta, used_microbatching=True,
        client_fraction=client_fraction)
    log.info("Actual (ε,δ) is: (%s, %s), σ = %s", actual_epsilon, delta, sigma)

    # turn on output
    sys.stdout.close()
    sys.stdout = old_stdout
    return sigma


def clip_tensor(tensor: torch.Tensor, clip_bound):
    nn.utils.clip_grad_norm_(tensor, clip_bound)


def clip_gradients(net: nn.Module, clip_bound):
    for param in net.parameters():
        clip_tensor(param, clip_bound)


def add_gaussian_noise(tensor: torch.Tensor, batch_size, sigma, clip_bound):
    """add noise to a list tensors"""
    noise_to_add = torch.zeros(
        tensor.shape, requires_grad=False).to(tensor.device)
    noise_to_add.normal_(0., std=clip_bound * sigma)
    noise = noise_to_add / float(batch_size)
    with torch.no_grad():
        tensor.add_(noise)

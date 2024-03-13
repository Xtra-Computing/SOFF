"""Training utilities"""
import os
import re
import sys
import random
import logging
import threading
from itertools import chain
from typing import Callable, List, Tuple
import torch
import numpy as np
from munch import Munch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from .metrics import _MetricCreator

log = logging.getLogger(__name__)


def seed_everything(seed: int):
    """Set seed for everything"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(int(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_determinism(cfg: Munch):
    """Initialize determinism for deterministic training"""
    # set deterministic behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed_everything(cfg.federation.seed)

    if re.match('^vgg.*$', cfg.model.name.lower()) is not None or \
            re.match('^u_.*$', cfg.model.name.lower()) is not None:
        log.warning("VGG/U_* requires non-deterministic implementation, "
                    "disabling pytorch determinism...")
        return

    torch.use_deterministic_algorithms(True)
    cudnn.deterministic = True
    cudnn.benchmark = False


def init_excepthook():
    """
    Handle all uncaught exceptions and make exceptions pass from subthread to
    the main thread.
    """
    # handle all uncaught exceptions
    def uncaught_exception_handler(type_, value, tback):
        logging.error("Uncaught exception.", exc_info=(type_, value, tback))

    sys.excepthook = uncaught_exception_handler

    # Workaround for sys.excepthook thread bug
    # http://spyced.blogspot.com/2007/06/workaround-for-sysexcepthook-bug.html
    # Call once from __main__ before creating any threads.
    init_old = threading.Thread.__init__

    def init(self, *args, **kwargs):
        init_old(self, *args, **kwargs)
        run_old = self.run

        def run_with_except_hook(*args, **kw):
            try:
                run_old(*args, **kw)
            except (KeyboardInterrupt, SystemExit):
                sys.excepthook(*sys.exc_info())
                raise
            except Exception:
                sys.excepthook(*sys.exc_info())
        self.run = run_with_except_hook
    threading.Thread.__init__ = init


def all_params(net: torch.nn.Module):
    """returns a iterator of all parameters and buffers"""
    return list(chain(net.parameters(), net.buffers()))


def init_buffer(buf: torch.nn.Module, device: torch.device):
    """Send tensor to device and initialize it to zero"""
    buf.to(device)
    for param in buf.parameters():
        param.requires_grad_(False)
    for param in all_params(buf):
        param.zero_()


def do_svd_analysis(
        grad: torch.nn.Module, tfboard_writer: SummaryWriter, tag_suffix: str,
        epoch: int):
    """
    grad: gradient stored in network's parameters
    """
    for name, param in grad.named_parameters():
        # filter out all batchnorm layers
        if 'batchnorm' not in name:
            # Do SVD with unbiased gradients
            diag = torch.svd(
                param.view(param.shape[0], -1).transpose(0, 1) -
                torch.mean(param), compute_uv=False)[1]

            log.debug(
                "SVD for layer %s = %s", str(name).replace('.', '_'),
                '[' + ', '.join(map(str, diag.tolist())) + ']')

            # need to multiply the diag to get the variance


def is_batchnorm(layer: torch.nn.Module) -> bool:
    """Decide if `layer` is a batchnorm layer"""
    return isinstance(layer, (
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))

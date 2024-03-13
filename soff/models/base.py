"""Model trainers"""
import logging
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple
from munch import Munch
import torch
from torch.utils.data import DataLoader
from ..utils.arg_parser import r_hasattr, require
from ..utils.logging import DataLogger
from ..utils.metrics import _MetricCreator
from ..privacy.rdp import (
    compute_gaussian_sigma, add_gaussian_noise, clip_gradients)

log = logging.getLogger(__name__)


@require('hardware.gpus')
class _ModelTrainer(ABC):
    """The base model trainer class"""

    def __init__(
            self, cfg: Munch, data_loader: DataLoader,
            loss_fn: Callable, metrics: List[_MetricCreator],
            datalogger: Optional[DataLogger] = None):

        # Initialize basic variable for trianing
        self.data_loader = data_loader
        self.iter = iter(data_loader)

        self.losses: List[float] = []
        self.loss_fn = loss_fn
        # For now we only train on a single GPU
        self.device = (
            torch.device('cuda', cfg.hardware.gpus[0])
            if len(cfg.hardware.gpus) > 0 else torch.device('cpu'))

        self.metrics = metrics
        self.running_metrics: List[Callable] = []
        self._reset_running_metrics()

        # Intiailize differential privacy
        self.use_dp = (
            'differential_privacy' in cfg and
            cfg.differential_privacy.dp_type is not None)
        self.noise_level = 0.
        self.clip = None
        if self.use_dp and cfg.differential_privacy.dp_type == 'rdp':
            self.noise_level = compute_gaussian_sigma(cfg, len(data_loader))
            self.clip = cfg.differential_privacy.rdp.clip

        # Logging
        self.datalogger = datalogger

        # Debug
        self.debug_skip_training = (
            r_hasattr(cfg, 'debug.skip_training') and cfg.debug.skip_training)

    def _reset_running_metrics(self):
        self.losses = []
        self.running_metrics = [met().to(self.device) for met in self.metrics]

    @abstractmethod
    def train_model(self, net: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, iters: int):
        """Subclass should implement this to define how the model is trained"""
        raise NotImplementedError("please use a subclass of ModelTrainer")

    def _calc_grad(self, net, datas, labels):
        predictions = net(datas)
        loss = self.loss_fn(predictions, labels)
        loss.backward()
        return predictions, loss

    def _train_one_step(self, net, datas, labels, optimizer):
        optimizer.zero_grad()
        res = self._calc_grad(net, datas, labels)

        if self.use_dp:
            # clip and add dp noise
            if self.clip is not None:
                clip_gradients(net, self.clip)

            log.debug("Adding gaussian noise")
            for param in net.parameters():
                add_gaussian_noise(
                    param.grad,
                    self.data_loader.batch_size,
                    self.noise_level, self.clip)

        optimizer.step()
        return res

    def _log_training(self, loss, predictions, labels, iters):
        self.losses.append(loss)
        for met in self.running_metrics:
            met.update(predictions, labels)

        if iters % 50 == 0:
            train_loss = float(torch.mean(torch.Tensor(self.losses)))

            # some metrics, like auroc, are not deterministic
            is_deterministic = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(False)
            results = [
                met.compute().cpu().item() for met in self.running_metrics]
            torch.use_deterministic_algorithms(is_deterministic)

            log.info("  iter %s (%s/%s)", iters,
                     iters % len(self.data_loader), len(self.data_loader))

            log.info("    Train loss: %s", train_loss)
            for met, res in zip(self.metrics, results):
                log.info("    Train %s: %s", met.name, res)
            if self.datalogger is not None:
                self.datalogger.add_scalar("Train:loss", train_loss, iters)
                for met, res in zip(self.metrics, results):
                    self.datalogger.add_scalar(f"Train:{met.name}", res, iters)

            self._reset_running_metrics()

    @staticmethod
    def evaluate_model(
        net: torch.nn.Module, data_loader: DataLoader, loss_criterion: Callable,
        additional_metrics: List[_MetricCreator], device: torch.device) \
            -> Tuple[float, List[float]]:
        """
        return: (loss, [results for additional_criteria])
        """
        model_is_training = net.training
        net.eval()

        is_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        with torch.no_grad():
            losses = []

            metric_objs = [met().to(device) for met in additional_metrics]

            for datas, labels in data_loader:
                datas, labels = datas.to(device), labels.to(device)
                predictions = net(datas)

                losses.append(loss_criterion(predictions, labels))
                for met in metric_objs:
                    # additional_metrics[i].append(met(predictions, labels))
                    met.update(predictions, labels)

            loss = float(torch.mean(torch.Tensor(losses)))
            additional_results = [
                met.compute().cpu().item() for met in metric_objs]
        torch.use_deterministic_algorithms(is_deterministic)

        # restore model state
        if model_is_training:
            net.train()

        # return global model loss
        return loss, additional_results


class PerIterTrainer(_ModelTrainer):
    """Train model iter-by-iter. Each iter consumes one minibatch."""

    def train_model(self, net, optimizer, iters):

        datas, labels = next(self.iter, (None, None))
        if datas is not None and labels is not None:
            datas, labels = datas.to(self.device), labels.to(self.device)

        if datas is None:
            self.iter = iter(self.data_loader)
            datas, labels = next(self.iter)
            datas, labels = datas.to(self.device), labels.to(self.device)
        iters += 1
        predictions, loss = self._train_one_step(net, datas, labels, optimizer)
        self._log_training(loss, predictions, labels, iters)

        return iters


class PerEpochTrainer(_ModelTrainer):
    """Train model epoch-by-epoch. Each epoch consumes the full dataset once"""

    def train_model(self, net, optimizer,  iters):
        for datas, labels in self.data_loader:
            datas, labels = datas.to(self.device), labels.to(self.device)
            iters += 1

            predictions, loss = self._train_one_step(
                net, datas, labels, optimizer)
            self._log_training(loss, predictions, labels, iters)

            if self.debug_skip_training:
                log.warning("--debug.skip-training enabled, skipping ...")
                break

        return iters

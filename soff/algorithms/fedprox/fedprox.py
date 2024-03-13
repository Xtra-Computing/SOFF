"""
The FedProx Algorithm:

[1] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith,
“Federated optimization in heterogeneous networks,” in Proceedings of
machine learning and systems, 2020, vol. 2, pp. 429–450. [Online]. Available:
https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf
"""

import torch
from torch import nn
from .. import fedavg
from ...models.base import _ModelTrainer, PerEpochTrainer, PerIterTrainer
from ...models.lstm import PerEpochLSTMTrainer, PerIterLSTMTrainer


class ConfParser(fedavg.Server.conf_parser()):
    """Simple FedProx config parser"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        fedprox_args = self.add_argument_group(
            "FedProx-related Arguments (S->C)")
        fedprox_args.add_argument(
            '-prox.mu', '--fedprox.mu', default=0.1, type=float, metavar='μ',
            help="FedProx parameter μ")


class Server(fedavg.Server):
    @classmethod
    def conf_parser(cls):
        return ConfParser


class Client(fedavg.Client):
    """Simple FedProx algorithm client """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.trainer = FedProxPerEpochTrainer(
            cfg, self.train_loader, self.train_criterion,
            self.additional_metrics, self.datalogger)

    def train_one_round(self):
        return self.trainer.train_model_fedprox(
            self.net_global, self.net, self.optimizer, self.iters)


class FedProxTrainerAdapter(_ModelTrainer):
    """
    An adapter to ModelTrainer, hooks model updating & gradient calculation
    process to accumulate gradient.
    """

    def __init__(self, cfg, *args, **kwargs):
        """
        mu: μ value of fedprox
        """
        super().__init__(cfg, *args, **kwargs)
        self._mu = cfg.fedprox.mu
        self._global_model: nn.Module

    def train_model_fedprox(self, global_model: nn.Module, *args, **kwargs):
        """Wrapper of super.train_model, storing the global model in advance"""
        self._global_model = global_model
        return super().train_model(*args, **kwargs)

    def _calc_grad(self, net, datas, labels):
        """add hooks to incorporate gradient correction"""
        predictions = net(datas)
        loss = self.loss_fn(predictions, labels)
        # add some salt
        for param, g_param in \
                zip(net.parameters(), self._global_model.parameters()):
            loss += (self._mu / 2 * torch.nn.MSELoss()(param, g_param) ** 2)
        loss.backward()
        return predictions, loss


class FedProxLSTMTrainerAdapter(FedProxTrainerAdapter):
    def _calc_grad(self, net, datas, labels):
        self.hidden = net.detach_hidden(self.hidden)
        predictions, self.hidden = net(datas, self.hidden)
        loss = self.loss_fn(
            predictions.reshape(datas.shape[0] * datas.shape[1], -1),
            labels.reshape(-1))
        for param, g_param in \
                zip(net.parameters(), self._global_model.parameters()):
            loss += (self._mu / 2 * torch.nn.MSELoss()(param, g_param) ** 2)
        # predictions, labels)
        loss.backward()
        return predictions, loss


class FedProxPerEpochTrainer(FedProxTrainerAdapter, PerEpochTrainer):
    """Per-epoch variant of FedProx's model trainer"""


class FedProxPerIterTrainer(FedProxTrainerAdapter, PerIterTrainer):
    """Per-iter variant of FedProx's model trainer"""


class FedProxPerEpochLSTMTrainer(FedProxLSTMTrainerAdapter, PerEpochLSTMTrainer):
    """Per-epoch variant of FedProx's LSTM trainer"""

class FedProxPerIterLSTMTrainer(FedProxLSTMTrainerAdapter, PerIterLSTMTrainer):
    """Per-iter variant of FedProx's LSTM trainer"""

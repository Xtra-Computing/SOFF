"""Model for the Abchina Dataset"""

import torch
import logging
from torch import nn
from ..datasets.fl_split import _FLSplit
from ..datasets.raw_dataset.abchina import Abchina
from ..utils.arg_parser import ArgParseOption, options
from .base import _ModelTrainer, PerEpochTrainer, PerIterTrainer

log = logging.getLogger(__name__)


@options(
    "Abchina Hybrid CNN/LSTM Model Configs",
    ArgParseOption(
        'hlstm.ls', 'hlstm.linear-size', default=300,
        type=int, metavar='NUM',
        help="Linear layer hidden dimension."),
    ArgParseOption(
        'hlstm.n', 'hlstm.num-layers', default=2, type=int, metavar='NUM',
        help="Number of lstm layers."),
    ArgParseOption(
        'hlstm.hd1', 'hlstm.hidden-dim-1', default=300,
        type=int, metavar='NUM',
        help="LSTM1 hidden dimension."),
    ArgParseOption(
        'hlstm.hd2', 'hlstm.hidden-dim-2', default=300,
        type=int, metavar='NUM',
        help="LSTM2 hidden dimension."),
    ArgParseOption(
        'hlstm.dp', 'hlstm.drop-prob', default=0.2, type=float, metavar='PROB',
        help="Dropout probability."))
class HybridCnnLstm(nn.Module):
    def __init__(self, cfg, dataset: _FLSplit) -> None:
        super().__init__()
        assert all(isinstance(data, Abchina) for data in dataset.datasets)
        (x, _, _), _ = dataset[0]
        y_avg = dataset.datasets[0].y_avg
        z_avg = dataset.datasets[0].z_avg

        self.num_layers = cfg.model.hlstm.num_layers
        self.linear_dim = cfg.model.hlstm.linear_size
        self.y_hidden_dim = cfg.model.hlstm.hidden_dim_1
        self.z_hidden_dim = cfg.model.hlstm.hidden_dim_2

        self.cnn = nn.Sequential(
            nn.Linear(len(x), self.linear_dim),
            nn.ReLU(),
            nn.Linear(self.linear_dim, self.linear_dim),
        )

        # self.lstm1 = nn.LSTM(
        #     len(y_avg), self.y_hidden_dim, self.num_layers,
        #     dropout=cfg.model.hlstm.drop_prob,
        #     batch_first=True)
        # self.lstm2 = nn.LSTM(
        #     len(z_avg), self.z_hidden_dim, self.num_layers,
        #     dropout=cfg.model.hlstm.drop_prob,
        #     batch_first=True)
        self.lstm1 = nn.Sequential(
            nn.Linear(len(y_avg), self.y_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.y_hidden_dim, self.y_hidden_dim),
            nn.ReLU(),
        )
        self.lstm2 = nn.Sequential(
            nn.Linear(len(z_avg), self.z_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.z_hidden_dim, self.z_hidden_dim),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                self.linear_dim + self.y_hidden_dim + self.z_hidden_dim,
                self.linear_dim),
            nn.ReLU(),
            nn.Linear(self.linear_dim, 1)
        )

    def forward(self, x, y, y_h, z, z_h):
        # NOTE: injection
        # y = torch.zeros_like(y)

        x = self.cnn(x)
        # y, (yh, yc) = self.lstm1(y, (
        #     y_h[0][:, :len(y)].contiguous(), y_h[1][:, :len(y)].contiguous()))
        # z, (zh, zc) = self.lstm2(z, (
        #     z_h[0][:, :len(z)].contiguous(), z_h[1][:, :len(z)].contiguous()))
        y = self.lstm1(y)
        z = self.lstm2(z)

        y = torch.mean(y, dim=1)
        z = torch.mean(z, dim=1)

        x = nn.functional.normalize(x)
        y = nn.functional.normalize(y)
        z = nn.functional.normalize(z)

        c = torch.cat((x, y, z), dim=1)
        c = torch.sigmoid(self.fc(c).view(-1))

        # return (
        #     c,
        #     (torch.cat((yh, y_h[0][:, len(y):]), dim=1),
        #      torch.cat((yc, y_h[1][:, len(y):]), dim=1)),
        #     (torch.cat((zh, z_h[0][:, len(z):]), dim=1),
        #      torch.cat((zc, z_h[1][:, len(z):]), dim=1)))
        return c, y_h, z_h

    def init_hidden(self, batch_size, device: torch.device):
        y_hidden = torch.zeros(
            self.num_layers, batch_size, self.y_hidden_dim).to(device)
        y_cell = torch.zeros(
            self.num_layers, batch_size, self.y_hidden_dim).to(device)
        z_hidden = torch.zeros(
            self.num_layers, batch_size, self.z_hidden_dim).to(device)
        z_cell = torch.zeros(
            self.num_layers, batch_size, self.z_hidden_dim).to(device)
        return (y_hidden, y_cell), (z_hidden, z_cell)

    @staticmethod
    def detach_hidden(yh, zh):
        return (
            (yh[0].detach().contiguous(), yh[1].detach().contiguous()),
            (zh[0].detach().contiguous(), zh[1].detach().contiguous()))


class _HybridLSTMTrainer(_ModelTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.y_hidden = None
        self.z_hidden = None

    def _calc_grad(self, net, datas, labels):
        self.y_hidden, self.z_hidden = net.detach_hidden(
            self.y_hidden, self.z_hidden)
        x, y, z = datas
        predictions, self.y_hidden, self.z_hidden = net(
            x, y, self.y_hidden, z, self.z_hidden)

        loss = self.loss_fn(
            predictions.reshape(-1), labels.reshape(-1))
        # predictions, labels)
        loss.backward()
        return predictions, loss

    @staticmethod
    def evaluate_model(
            net: HybridCnnLstm,
            data_loader, loss_criterion, additional_metrics, device):

        model_is_training = net.training
        net.eval()
        with torch.no_grad():
            losses = []

            metric_objs = [met().to(device) for met in additional_metrics]
            y_hidden = None
            z_hidden = None

            for (x, y, z), labels in data_loader:
                x, y, z = x.to(device), y.to(device), z.to(device)
                labels = labels.to(device)
                y_hidden, z_hidden = (
                    (y_hidden, z_hidden) if y_hidden is not None else
                    net.init_hidden(len(x), device))
                y_hidden, z_hidden = net.detach_hidden(y_hidden, z_hidden)
                predictions, y_hidden, z_hidden = net(
                    x, y, y_hidden, z, z_hidden)

                losses.append(loss_criterion(
                    predictions.reshape(-1), labels.reshape(-1)))

                for met in metric_objs:
                    met.update(predictions, labels)

            loss = float(torch.mean(torch.Tensor(losses)))

            # bauroc are not deterministic
            is_deterministic = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(False)
            additional_results = [
                met.compute().cpu().item() for met in metric_objs]
            torch.use_deterministic_algorithms(is_deterministic)

        # restore model state
        if model_is_training:
            net.train()

        # return global model loss
        return loss, additional_results


class HybridLSTMPerIterTrainer(PerIterTrainer, _HybridLSTMTrainer):
    def train_model(self, net, optimizer, iters):
        datas, labels = next(self.iter, (None, None))
        if datas is not None and labels is not None:
            datas, labels = tuple(
                d.to(self.device) for d in datas), labels.to(self.device)
            self.y_hidden, self.z_hidden = (
                (self.y_hidden, self.z_hidden)
                if self.y_hidden is not None else
                net.init_hidden(len(datas[0]), self.device))

        if datas is None:
            self.iter = iter(self.data_loader)
            datas, labels = next(self.iter)
            datas, labels = tuple(
                d.to(self.device) for d in datas), labels.to(self.device)
            labels = labels.to(self.device)
            self.y_hidden, self.z_hidden = net.init_hidden(
                len(datas[0]), self.device)
        iters += 1

        predictions, loss = self._train_one_step(
            net, datas, labels.reshape(-1), optimizer)
        self._log_training(loss, predictions, labels, iters)

        return iters


class HybridLSTMPerEpochTrainer(PerEpochTrainer, _HybridLSTMTrainer):
    def train_model(self, net, optimizer,  iters):
        assert hasattr(net, 'init_hidden')
        self.hidden = None
        for datas, labels in self.data_loader:
            datas, labels = tuple(
                d.to(self.device) for d in datas), labels.to(self.device)
            self.y_hidden, self.z_hidden = (
                (self.y_hidden, self.z_hidden)
                if self.y_hidden is not None else
                net.init_hidden(len(datas[0]), self.device))
            iters += 1

            predictions, loss = self._train_one_step(
                net, datas, labels.reshape(-1), optimizer)
            self._log_training(loss, predictions, labels, iters)
        return iters

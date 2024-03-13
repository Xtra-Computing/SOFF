"""Simple LSTM model"""
import math
import logging
import torch
from torch import nn
from munch import Munch
from ..datasets.fl_split.base import _FLSplit
from ..utils.arg_parser import ArgParseOption, options
from ..privacy.rdp import add_gaussian_noise
from .base import _ModelTrainer, PerEpochTrainer, PerIterTrainer

log = logging.getLogger(__name__)


@options(
    "LSTM Model Configs",
    ArgParseOption(
        'lstm.n', 'lstm.num-layers', default=2, type=int, metavar='NUM',
        help="Number of lstm layers."),
    ArgParseOption(
        'lstm.hd', 'lstm.hidden-dim', default=200, type=int, metavar='DIM',
        help="Hidden dimension size."),
    ArgParseOption(
        'lstm.dp', 'lstm.drop-prob', default=0.5, type=float, metavar='PROB',
        help="Dropout probability."))
class LSTM(nn.Module):
    """The LSTM model"""

    def __init__(self, cfg, dataset):
        super().__init__()

        self.num_layers = cfg.model.lstm.num_layers
        self.hidden_dim = cfg.model.lstm.hidden_dim

        # self.embedding = nn.Embedding(vocabSize, embedding_dim)
        self.lstm = nn.LSTM(
            dataset.datasets[0].embedding_dim(), self.hidden_dim,
            self.num_layers, dropout=cfg.model.lstm.drop_prob,
            batch_first=True)
        self.dropout = nn.Dropout(0.3)

        self.linear = nn.Linear(self.hidden_dim, dataset.num_classes())
        self.sig = nn.Sigmoid()

    def forward(self, batch):
        """Overrides Module.forward"""
        batch_size = batch.size(0)

        # embeds = self.embedding(batch)
        # lstmOut, hidden = self.lstm(embeds, hidden)
        lstm_out, _ = self.lstm(batch, (
            torch.randn(self.num_layers, batch_size, self.hidden_dim)
            .to(batch.device),
            torch.randn(self.num_layers, batch_size, self.hidden_dim)
            .to(batch.device)))
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.linear(out)

        sigmoid_out = self.sig(out)
        sigmoid_out = sigmoid_out.view(batch_size, -1)
        sigmoid_out = sigmoid_out[:, -1]

        return sigmoid_out.double()


@options(
    "LSTM Model Configs",
    ArgParseOption(
        'lm-lstm.n', 'lang-model-lstm.num-layers',
        default=2, type=int, metavar='NUM',
        help="Number of lstm layers."),
    ArgParseOption(
        'lm-lstm.hd', 'lang-model-lstm.hidden-dim',
        default=200, type=int, metavar='DIM',
        help="Hidden dimension size."),
    ArgParseOption(
        'lm-lstm.ed', 'lang-model-lstm.embedding-dim',
        default=200, type=int, metavar='DIM',
        help="Embedding dimension size."),
    ArgParseOption(
        'lm-lstm.dp', 'lang-model-lstm.drop-prob',
        default=0.5, type=float, metavar='PROB',
        help="Dropout probability."),
    ArgParseOption(
        'lm-lstm.dp2', 'lang-model-lstm.drop-prob-2',
        default=0.3, type=float, metavar='PROB',
        help="Dropout probability."),
    ArgParseOption(
        'lm-lstm.tw', 'lang-model-lstm.tie-weights', action="store_true",
        help="Embedding dimension size."),
    ArgParseOption(
        'lm-lstm.cl', 'lang-model-lstm.clip',
        default=0.25, type=float, metavar='CLIP',
        help="Clip grard norm."))
class LanguageModelingLSTM(nn.Module):
    """
    LSTM for langauge modelling.
    The used _RawDataset subclass must have the same '.vocab_size' attribute

    Input dataset should have shape [seq_len, bath_size] (i.e. batch_first=False)
    """

    def __init__(self, cfg: Munch, dataset: _FLSplit):
        super().__init__()

        vocab_size = dataset.datasets[0].vocab_size
        assert all(ds.vocab_size == vocab_size for ds in dataset.datasets)

        self.num_layers = cfg.model.lang_model_lstm.num_layers
        self.hidden_dim = cfg.model.lang_model_lstm.hidden_dim
        self.embedding_dim = cfg.model.lang_model_lstm.embedding_dim
        self.dropout_prob = cfg.model.lang_model_lstm.drop_prob
        self.dropout_prob_2 = cfg.model.lang_model_lstm.drop_prob_2

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            self.embedding_dim, self.hidden_dim, num_layers=self.num_layers,
            dropout=self.dropout_prob, batch_first=False)
        self.dropout = nn.Dropout(self.dropout_prob_2)
        self.fc = nn.Linear(self.hidden_dim, vocab_size)

        if cfg.model.lang_model_lstm.tie_weights:
            assert self.embedding_dim == self.hidden_dim, 'cannot tie, check dims'
            self.embedding.weight = self.fc.weight
        self.init_weights()

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hidden_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(
                self.embedding_dim, self.hidden_dim
            ).uniform_(-init_range_other, init_range_other)
            self.lstm.all_weights[i][1] = torch.FloatTensor(
                self.hidden_dim, self.hidden_dim
            ).uniform_(-init_range_other, init_range_other)

    def init_hidden(self, batch_size, device: torch.device):
        hidden = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell

    @staticmethod
    def detach_hidden(hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden


class _LSTMTrainer(_ModelTrainer):
    """
    Requires the trained model:
        1. to have an `init_hidden` function accepting `batch_size, device`
        1. to have an `detach_hidden` function accepting `hidden`
        2. to have an `forward` function accepting `src, hidden`
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.hidden = None
        self.clip = cfg.model.lang_model_lstm.clip

    def _calc_grad(self, net, datas, labels):
        self.hidden = net.detach_hidden(self.hidden)
        predictions, self.hidden = net(datas, self.hidden)
        loss = self.loss_fn(
            predictions.reshape(datas.shape[0] * datas.shape[1], -1),
            labels.reshape(-1))
        # predictions, labels)
        loss.backward()
        return predictions, loss

    def _log_training(self, loss, predictions, labels, iters):
        self.losses.append(loss)
        for met in self.running_metrics:
            met.update(predictions.swapdims(0, 1), labels.swapdims(0, 1))

        if iters % 50 == 0:
            train_loss = float(torch.mean(torch.Tensor(self.losses)))
            results = [
                met.compute().cpu().item() for met in self.running_metrics]

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
            net: LanguageModelingLSTM,
            data_loader, loss_criterion, additional_metrics, device):

        model_is_training = net.training
        net.eval()
        with torch.no_grad():
            losses = []

            metric_objs = [met().to(device) for met in additional_metrics]
            hidden = None

            for datas, labels in data_loader:
                datas, labels = datas.to(device), labels.to(device)
                hidden = (
                    hidden if hidden is not None else
                    net.init_hidden(datas.shape[1], device))
                predictions, hidden = net(datas, hidden)

                losses.append(loss_criterion(
                    predictions.reshape(datas.shape[0] * datas.shape[1], -1),
                    labels.reshape(-1)))

                for met in metric_objs:
                    # Torchmetrics assumes batch_first, so we swap batch here.
                    met.update(
                        predictions.swapdims(0, 1), labels.swapdims(0, 1))

            loss = float(torch.mean(torch.Tensor(losses)))
            additional_results = [
                met.compute().cpu().item() for met in metric_objs]

        # restore model state
        if model_is_training:
            net.train()

        # return global model loss
        return loss, additional_results


class PerIterLSTMTrainer(PerIterTrainer, _LSTMTrainer):
    def train_model(self, net: LanguageModelingLSTM, optimizer, iters):
        datas, labels = next(self.iter, (None, None))
        if datas is not None and labels is not None:
            datas, labels = datas.to(self.device), labels.to(self.device)
            self.hidden = (
                self.hidden if self.hidden is not None else
                net.init_hidden(datas.shape[1], self.device))

        if datas is None:
            self.iter = iter(self.data_loader)
            datas, labels = next(self.iter)
            datas, labels = datas.to(self.device), labels.to(self.device)
            self.hidden = net.init_hidden(datas.shape[1], self.device)
        iters += 1

        predictions, loss = self._train_one_step(
            net, datas, labels.reshape(-1), optimizer)
        self._log_training(loss, predictions, labels, iters)

        return iters


class PerEpochLSTMTrainer(PerEpochTrainer, _LSTMTrainer):
    def train_model(self, net, optimizer,  iters):
        assert hasattr(net, 'init_hidden')
        self.hidden = None
        for datas, labels in self.data_loader:
            datas, labels = datas.to(self.device), labels.to(self.device)
            self.hidden = (
                self.hidden if self.hidden is not None else
                net.init_hidden(datas.shape[1], self.device))
            iters += 1

            predictions, loss = self._train_one_step(
                net, datas, labels.reshape(-1), optimizer)
            self._log_training(loss, predictions, labels, iters)

        return iters

# From https://github.com/KAI-YUE/Predictive-Coding-FL
from abc import ABC, abstractmethod
import copy
from typing import Dict, List, Tuple, Type
from munch import Munch
import torch
from torch import optim, Tensor
from ...utils.arg_parser import BaseConfParser


class Buffer(ABC):
    """Buffer to keep history global weight"""
    @abstractmethod
    def push(self, delta_weight: Tensor):
        """Push the delta weight into the buffer"""

    @abstractmethod
    def init_buffer_(self, weight_template: Tensor):
        """Initialize the buffer"""


class DeltaBuffer(Buffer):
    """DeltaBuffer to keep history global weight"""

    def __init__(self, delta_k):
        self._buffer = []
        self.buffer_full_ = False
        self._buffer_len = delta_k
        self.top_ = -1

    def init_buffer_(self, weight_template):
        for _ in range(self._buffer_len):
            self._buffer.append(torch.zeros_like(weight_template))

    def push(self, delta_weight):
        self.top_ += 1
        if self.top_ == self._buffer_len:
            self.top_ = 0
            self.buffer_full_ = True

        self._buffer[self.top_] = delta_weight.clone()

    def delta(self):
        if self.buffer_full_:
            delta_ = self._buffer[0].clone()
            for i in range(1, self._buffer_len):
                delta_ += self._buffer[i]
            delta_ *= 1/self._buffer_len
        elif self.top_ >= 0:
            # delta_ = self._buffer[0].clone()
            # for i in range(1, self.top_+1):
            #     delta_ += self._buffer[i]
            # delta_ *= 1/(self.top_ + 1)
            delta_ = torch.zeros_like(self._buffer[0])
        else:
            delta_ = self._buffer[0].clone()

        return delta_


class OuArBuffer(Buffer):
    def __init__(self):
        self._buffer = []
        self.buffer_full_ = False
        self._buffer_len = 2        # for the ou process, we set buffer length to 2
        self.top_ = 0
        self.shape = None

    def init_buffer_(self, weight_template):
        for _ in range(self._buffer_len):
            self._buffer.append(weight_template.clone().flatten().view(-1, 1))
        self.shape = weight_template.shape

    def push(self, weight):
        if self.top_ == -1:
            self.top = 0
        elif self.top_ == 0:
            self.buffer_full_ = True
            self.top = 1
        elif self.top == 1:
            self._buffer[0] = self._buffer[1].clone()

        self._buffer[self.top_] = weight.clone().flatten().view(-1, 1)

    def output(self):
        if self.buffer_full_:
            prev_weight = self._buffer[0]
            beta = torch.inverse(
                prev_weight.T @ prev_weight) @ prev_weight.T @ self._buffer[1]
            out = beta*self._buffer[1].clone()
        else:
            out = self._buffer[1].clone()

        out = out.view(self.shape)

        return out


class ArDeltaBuffer(Buffer):
    """ArDeltaBuffer predicts delta with dynamic AR coefficients"""

    def __init__(self, order):
        self._buffer = []
        self.buffer_full_ = False
        self.order = order
        self._buffer_len = order + 1
        self.top_ = -1
        self.shape = None

    def init_buffer_(self, weight_template):
        for _ in range(self._buffer_len):
            self._buffer.append(torch.zeros_like(weight_template))
        self.shape = weight_template.shape

    def push(self, delta_weight):
        self.top_ += 1
        if self.top_ == self._buffer_len:
            self.top_ = 0
            self.buffer_full_ = True

        self._buffer[self.top_] = delta_weight.clone().flatten().view(-1, 1)

    def delta(self):
        if self.buffer_full_:
            buffer_copy = copy.deepcopy(self._buffer)
            top_delta = buffer_copy.pop(self.top_)
            aug_delta = torch.cat(buffer_copy, dim=1)
            betas = torch.inverse(
                aug_delta.T @ aug_delta) @ aug_delta.T @ top_delta

            delta_ = betas[self.top_-1] * top_delta
            for i in range(1, self.order):
                delta_ += self._buffer[self.top_-i] * betas[self.top_-i-1]
        else:
            delta_ = self._buffer[-1].clone()

        delta_ = delta_.view(self.shape)

        return delta_


class AdaMBuffer(Buffer):
    def __init__(self, betas):
        self.exp_avg = None
        self.exp_avg_sq = None
        self.betas = betas
        self.step = 0
        # self.eps = 1.e-4
        self.eps = 1.e-6
        self.delta_ = None

    def init_buffer_(self, weight_template):
        self.exp_avg = torch.zeros_like(weight_template)
        self.exp_avg_sq = torch.zeros_like(weight_template)

    def push(self, delta_weight):
        self.step += 1

        self.exp_avg = (
            self.betas[0] * self.exp_avg +
            (1-self.betas[0]) * delta_weight)
        self.exp_avg_sq = (
            self.betas[1] * self.exp_avg_sq +
            (1-self.betas[1]) * delta_weight ** 2)

        # bias_correction1 = 1 - self.betas[0] ** self.step
        # bias_correction2 = 1 - self.betas[1] ** self.step

        # denom = ((self.exp_avg_sq).sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
        # self.delta_ = (1./bias_correction1) * (self.exp_avg/denom)

        denom = (self.exp_avg_sq).sqrt().add_(self.eps)
        self.delta_ = (self.exp_avg / denom)

    def delta(self):
        return self.delta_


class MaskedAdaMBuffer(AdaMBuffer):
    def push(self, delta_weight):
        self.step += 1
        self.exp_avg = ((
            torch.ones_like(self.exp_avg) -
            (1 - self.betas[0]) * (delta_weight != 0).float()
        ) * self.exp_avg + (1 - self.betas[0]) * delta_weight)
        self.exp_avg_sq = ((
            torch.ones_like(self.exp_avg) -
            (1 - self.betas[1]) * (delta_weight != 0).float()
        ) * self.exp_avg_sq + (1 - self.betas[1]) * delta_weight ** 2)

        denom = (self.exp_avg_sq).sqrt().add_(self.eps)
        self.delta_ = (self.exp_avg / denom)


class ExpAvgBuffer(Buffer):
    def __init__(self, beta):
        self.exp_avg = None
        self.beta = beta
        self.delta_ = None

    def init_buffer_(self, weight_template):
        self.exp_avg = torch.zeros_like(weight_template)

    def push(self, delta_weight):
        self.exp_avg = self.beta*self.exp_avg + (1-self.beta)*delta_weight
        self.delta_ = self.exp_avg

    def delta(self):
        return self.delta_

# Adaptive OU Buffer


class OuAdaptiveBuffer(Buffer):
    def __init__(self, step_size: Tuple[float, float], scaler: float):
        self._buffer = []
        self.buffer_full_ = False
        self.order = 1
        self._buffer_len = 2
        self.top_ = 0

        # learnable coefficients
        self.coeffs = []
        self.bias = []
        self.scaler = scaler
        self.step_size = step_size

    def init_buffer_(self, weight_template):
        template = weight_template.flatten()
        for k in range(self._buffer_len):
            self._buffer.append(weight_template.detach().clone())

        self.shape = weight_template.shape

        # initialize learnable  coefficients.params and the corresponding optimizer
        self.coeffs.append(torch.ones_like(
            weight_template, requires_grad=True))  # coefficients
        self.bias.append(torch.randn_like(
            weight_template, requires_grad=True))  # bias
        self.coeffs[0].data *= self.scaler
        self.bias[0].data *= 0.

        self.optimizer = optim.AdamW([
            {"params": self.coeffs_params(), "lr": self.step_size[0]},
            {"params": self.bias_params(), "lr": self.step_size[1]}],
            weight_decay=1.e-3)

        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        # self.optimizer = optim.SGD(self.learnable_params(), lr=self.step_size)

    def coeffs_params(self):
        for param in self.coeffs:
            yield param

    def bias_params(self):
        for param in self.bias:
            yield param

    def push(self, weight):
        if self.top_ == 0:
            self.buffer_full_ = True
            self.top_ = 1
        elif self.top_ == 1:
            self._buffer[0] = self._buffer[1].clone()

        self._buffer[self.top_] = weight.detach().clone()

    def output(self):
        out = self.bias[0] + self.coeffs[0] * self._buffer[1]
        self.predict_register = out.clone()
        return out.data

    def optimizer_step(self):
        if not self.buffer_full_:
            pass
        else:
            self.optimizer.zero_grad()
            mse = torch.norm(self.predict_register - self._buffer[1])**2
            # mse = torch.norm(self.predict_register - torch.zeros_like(self.predict_register))**2
            # print("-- mse {:.3e} --".format(mse))
            mse.backward()
            # loss.backward()

            self.optimizer.step()
            # self.lr_scheduler.step()


class Predictor(ABC):
    """Interface for predicting the weight tensor"""

    def __init__(self, *_, **__) -> None:
        pass

    @abstractmethod
    def predict(self, weights: List[Tensor]) -> List[Tensor]:
        """Predict the weight in the next step."""

    @abstractmethod
    def update_buffer(self, weights: List[Tensor]) -> None:
        """Update the underlying buffer"""


class PrevFramePredictor(Predictor):
    def init_weight_buffers(self, _):
        pass

    def predict(self, weights):
        """
        :math: \hat{w}^{(t+1)} = w^{(t)} + \frac{1}{K} \sum_{k=1}^K \delta^{(t-k)}
        Use the history weight delta to predict the next step.
        """
        return copy.deepcopy(weights)

    def update_buffer(self, _):
        pass


class OuAdaptivePredictor(Predictor):
    def __init__(self, step_size, scaler, weights):
        self._weight_buffer = []
        self.step_size = step_size
        self.scaler = scaler

        for weight in weights:
            weight_buffer = OuAdaptiveBuffer(
                step_size=self.step_size, scaler=self.scaler)
            weight_buffer.init_buffer_(weight)
            self._weight_buffer.append(weight_buffer)

    def predict(self, weights):
        """:math: \hat{w}^{(t+1)} = w^{(t)} + \frac{1}{K} \sum_{k=1}^K \delta^{(t-k)}
        Use the history weight delta to predict the next step.
        """
        assert len(weights) == len(self._weight_buffer)
        return [buf.output() for buf in self._weight_buffer]

    def update_buffer(self, weights):
        assert len(weights) == len(self._weight_buffer)
        for buf, weight in zip(self._weight_buffer, weights):
            buf.push(weight)
            buf.optimizer_step()


class OuArPredictor(Predictor):
    def __init__(self, weights):
        self._weight_buffer = []
        for weight in weights:
            weight_buffer = OuArBuffer()
            weight_buffer.init_buffer_(weight)
            self._weight_buffer.append(weight_buffer)

    def predict(self, weights):
        """:math: \hat{w}^{(t+1)} = w^{(t)} + \frac{1}{K} \sum_{k=1}^K \delta^{(t-k)}
        Use the history weight delta to predict the next step.
        """
        assert len(weights) == len(self._weight_buffer)
        return [buf.output() for buf in self._weight_buffer]

    def update_buffer(self, weights):
        assert len(weights) == len(self._weight_buffer)
        for buf, weight in zip(self._weight_buffer, weights):
            buf.push(weight)


class AdaMPredictor(Predictor):
    def __init__(self, betas: Tuple[float, float], step_size: float, weights):
        self._delta_buffers = []
        self.betas = betas
        self.step_size = step_size

        for weight in weights:
            delta_buffer = AdaMBuffer(self.betas)
            delta_buffer.init_buffer_(weight)
            self._delta_buffers.append(delta_buffer)

    def predict(self, weights):
        """Use the history weight delta to predict the next step."""
        assert len(weights) == len(self._delta_buffers)
        return (
            copy.deepcopy(weights) if self._delta_buffers[0].step == 0 else [
                weight - self.step_size * buf.delta()
                for weight, buf in zip(weights, self._delta_buffers)
            ])

    def update_buffer(self, deltas):
        assert len(deltas) == len(self._delta_buffers)
        for buf, delta in zip(self._delta_buffers, deltas):
            buf.push(delta)


class MaskedAdaMPreditor(AdaMPredictor):
    def __init__(self, betas: Tuple[float, float], step_size: float, weights):
        self._delta_buffers = []
        self.betas = betas
        self.step_size = step_size

        for weight in weights:
            delta_buffer = MaskedAdaMBuffer(self.betas)
            delta_buffer.init_buffer_(weight)
            self._delta_buffers.append(delta_buffer)


class ExpAvgPredictor(Predictor):
    def __init__(self, beta, weights):
        self._delta_buffers = []
        for weight in weights:
            delta_buffer = ExpAvgBuffer(beta)
            delta_buffer.init_buffer_(weight)
            self._delta_buffers.append(delta_buffer)

    def predict(self, weights):
        """Use the history weight delta to predict the next step."""
        assert len(weights) == len(self._delta_buffers)
        return (
            copy.deepcopy(weights) if self._delta_buffers[0].step == 0 else [
                weight - buf.delta()
                for weight, buf in zip(weights, self._delta_buffers)
            ])

    def update_buffer(self, deltas):
        assert len(deltas) == len(self._delta_buffers)
        for buf, delta in zip(self._delta_buffers, deltas):
            buf.push(delta)


class DeltaStepPredictor(Predictor):
    def __init__(self, order, weights):
        self._delta_buffers = []
        self.order = order
        """Initialize delta buffer layer wise."""
        for weight in weights:
            delta_buffer = DeltaBuffer(self.order)
            delta_buffer.init_buffer_(weight)
            self._delta_buffers.append(delta_buffer)

    def predict(self, weights):
        """Use the history weight delta to predict the next step."""
        assert len(weights) == len(self._delta_buffers)
        return [
            weight - buf.delta()
            for weight, buf in zip(weights, self._delta_buffers)
        ]

    def update_buffer(self, deltas):
        assert len(deltas) == len(self._delta_buffers)
        for buf, delta in zip(self._delta_buffers, deltas):
            buf.push(delta)


class MaskedAccGradBuffer(Buffer):
    def __init__(self, momentum: float):
        self._momentum = momentum
        self._buffer: torch.Tensor

    def init_buffer_(self, weight_template):
        self._buffer = torch.ones_like(weight_template) * 1e-4

    def push(self, delta_weight: Tensor):
        self._buffer = (self._buffer * (
            torch.ones_like(delta_weight) -
            (1 - self._momentum) * (delta_weight != 0).float()
        ) + (1 - self._momentum) * delta_weight)

    def delta(self):
        return self._buffer


class MaskedAccGradPredictor(Predictor):
    def __init__(self, momentum: float, weights) -> None:
        self._delta_buffers = []
        for weight in weights:
            delta_buffer = MaskedAccGradBuffer(momentum)
            delta_buffer.init_buffer_(weight)
            self._delta_buffers.append(delta_buffer)

    def predict(self, weights):
        assert len(weights) == len(self._delta_buffers)
        return [
            weight - buf.delta()
            for weight, buf in zip(weights, self._delta_buffers)
        ]

    def update_buffer(self, deltas):
        assert len(deltas) == len(self._delta_buffers)
        for buf, delta in zip(self._delta_buffers, deltas):
            buf.push(delta)


predictors: Dict[str, Type[Predictor]] = {
    "previous": PrevFramePredictor,
    "ou":       OuAdaptivePredictor,
    "delta":    DeltaStepPredictor,
    "adam":     AdaMPredictor,
    "madam":    MaskedAdaMPreditor,
    "expavg":   ExpAvgPredictor,
    "mag":      MaskedAccGradPredictor,
}


class UGCPredictorConfParser(BaseConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = self.add_argument_group(
            "UGC Predictor Configs (S,S->C)")
        args.add_argument(
            '-ugc.pd.n', '--ugc.predictor.name',
            default='adam', choices=predictors.keys(),
            help="Type of preditor to use")

        args.add_argument(
            '-ugc.pd.ou.ss', '--ugc.predictor.ou.step-size',
            type=float, default=(1.e-3, 1.e-4), nargs=2,
            help="Step size for the OuAdaptive predictor")
        args.add_argument(
            '-ugc.pd.ou.sc', '--ugc.predictor.ou.scaler',
            type=float, default=0.99,
            help="Step size for the adam predictor")
        self.register_cfg_dep(
            '--ugc.predictor.ou',
            lambda cfg: cfg.ugc.predictor.name == 'ou')

        args.add_argument(
            '-ugc.pd.delta.o', '--ugc.predictor.delta.order',
            type=float, default=3,
            help="Order for the DeltaStep predictor")
        self.register_cfg_dep(
            '--ugc.predictor.delta',
            lambda cfg: cfg.ugc.predictor.name == 'delta')

        args.add_argument(
            '-ugc.pd.adam.b', '--ugc.predictor.adam.betas',
            type=float, default=(0.8, 0.9), nargs=2,
            help="Betas for the adam predictor")
        args.add_argument(
            '-ugc.pd.adam.ss', '--ugc.predictor.adam.step-size',
            type=float, default=1.e-3,
            help="Step size for the adam predictor")
        self.register_cfg_dep(
            '--ugc.predictor.adam',
            lambda cfg: cfg.ugc.predictor.name == 'adam')

        args.add_argument(
            '-ugc.pd.madam.b', '--ugc.predictor.madam.betas',
            type=float, default=(0.9, 0.99), nargs=2,
            help="Betas for the adam predictor")
        args.add_argument(
            '-ugc.pd.madam.ss', '--ugc.predictor.madam.step-size',
            type=float, default=1.,
            help="Step size for the adam predictor")
        self.register_cfg_dep(
            '--ugc.predictor.madam',
            lambda cfg: cfg.ugc.predictor.name == 'madam')

        args.add_argument(
            '-ugc.pd.mag.m', '--ugc.predictor.mag.momentum',
            type=float, default=0.9,
            help="Momentum for the MaskedAccGrad predictor")
        self.register_cfg_dep(
            '--ugc.predictor.mag',
            lambda cfg: cfg.ugc.predictor.name == 'mag')


def create_predictor(cfg: Munch, init_weights: List[Tensor]) -> Predictor:
    """Create a predictor from config and """
    name = cfg.ugc.predictor.name
    return predictors[name](
        **(cfg.ugc.predictor[name] if name in cfg.ugc.predictor else {}),
        weights=init_weights)

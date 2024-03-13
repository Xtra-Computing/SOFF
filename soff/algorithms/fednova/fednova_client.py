"""FedNova algorithm client and corresponding config parser"""

import torch
from .. import fedavg
from ...compressors.none import NoCompress
from ...compressors.compress_utils import pack_float32
from ...communications.protocol import MessageType
from ...utils.training import init_buffer
from ...models import create_model
from ...models.base import _ModelTrainer, PerIterTrainer, PerEpochTrainer


class ClientConfParser(fedavg.Client.conf_parser()):
    """Parse configs for the FedNova algorithm (Per-Client)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fednova_args = self.add_argument_group(
            "FedNova-related Arguments (C)")

        fednova_args.add_argument(
            '-nova.sw', '--fednova.step-weights', default=1, metavar='WEIGHT',
            help="When given a number, local steps has equal weights ( "
            "equivalent to fedavg). When given a list, each local step has "
            "weight corresponding to the value in the list, and the size of "
            "the list must match the number of local steps. "
            "Note that when used together with SGD momentum, the momentum "
            "factor will be combined into this argument before being sent to "
            "the server for aggregation.")


class Client(fedavg.Client):
    """FedNova algorithm client"""
    @classmethod
    def conf_parser(cls):
        return ClientConfParser

    def __init__(self, cfg):
        super().__init__(cfg)

        self.momentum = (
            self.optimizer.param_groups[0]['momentum']
            if isinstance(self.optimizer, torch.optim.SGD) else 0)
        self.average_every = cfg.fedavg.average_every

        # assign setp_weights (\vec{a}_i) for fednova
        if isinstance(cfg.fednova.step_weights, (int, float)):
            self.step_weights = [cfg.fednova.step_weights] * self.average_every
        elif isinstance(cfg.fednova.step_weights, list):
            assert len(cfg.fednova.step_weights) == self.average_every
            for weight in cfg.fednova.step_weights:
                assert isinstance(weight, (int, float))
            self.step_weights = cfg.fednova.step_weights

        self._step_weights_norm = 0.
        """Normalized step weights"""

        # prepare gradient correction buffer for fednova.
        # similar to err feedback, but for correcting errs in the fednova proto
        if cfg.fednova.gradient_correction:
            self.gradient_correction = create_model(cfg, self.train_dataset)
            self.gradient_correction.to(self.devices[0])
            init_buffer(self.gradient_correction, self.devices[0])
        else:
            self.gradient_correction = None

        self.trainer = FedNovaPerEpochTrainer(
            cfg, self.train_loader, self.train_criterion,
            self.additional_metrics, self.datalogger)

    def unload_resources(self):
        if self.gradient_correction is not None:
            self.gradient_correction.cpu()
        super().unload_resources()

    def load_resources(self):
        super().load_resources()
        if self.gradient_correction is not None:
            self.gradient_correction.to(self.devices[0])

    def train_one_round(self):
        return self.trainer.train_model_fednova(
            self.gradient_correction, self.net, self.optimizer, self.iters)

    def update_global_params(self):
        super().update_global_params()

        # if gradient correction is enabled, server will also send the weighted
        # sum of normalized gradient
        if self.gradient_correction:
            msg_type, data = self.dispatcher.recv_msg()
            self.handle_bye(msg_type)
            assert msg_type == MessageType.WEIGHTED_SUM_GRADIENT

            global_grad = NoCompress().decompress(data)
            for g_c, grad, g_g in zip(
                    self.gradient_correction.parameters(),
                    self.gradient.parameters(), global_grad):
                g_c.set_(g_g.to(self.devices[0]) - grad)

    def calc_gradient(self):
        super().calc_gradient()

        # correctly average all the step weights, including the momentum
        self._step_weights_norm = 0.
        if self.momentum > 0:
            self._step_weights_norm = sum(
                abs(a) * self.momentum *
                (1 - (self.momentum ** (self.average_every - i))) /
                (1 - self.momentum) for i, a in enumerate(self.step_weights))
        else:
            self._step_weights_norm = sum(abs(a) for a in self.step_weights)

        self.log.debug("ǁaᵢǁ = %s", self._step_weights_norm)

        # convert gradient to normalized gradient
        for grad in self.gradient.parameters():
            grad.div_(self._step_weights_norm)

    def aggregate(self, *args, **kwargs):
        super().aggregate(*args, **kwargs)
        self.dispatcher.send_msg(
            MessageType.STEP_WEIGHTS, pack_float32(self._step_weights_norm))


class FedNovaTrainerAdapter(_ModelTrainer):
    """
    An adapter to ModelTrainer, hooks model updating & gradient calculation
    process to accumulate gradient.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_correction = None
        self.net = None

    def train_model_fednova(self, gradient_correction, net, *args, **kwargs):
        """add hooks to the train_model method of the adapted class"""
        self.gradient_correction = gradient_correction
        self.net = net
        return super().train_model(net, *args, **kwargs)

    def _calc_grad(self, *argc, **argv):
        """add hooks to incorporate gradient correction"""
        res = super()._calc_grad(*argc, **argv)
        if self.gradient_correction is not None:
            assert self.net is not None
            with torch.no_grad():
                for param, corr in zip(
                        self.net.parameters(),
                        self.gradient_correction.parameters()):
                    param.grad.add_(corr)
        return res


class FedNovaPerEpochTrainer(FedNovaTrainerAdapter, PerEpochTrainer):
    """Per-epoch variant of FedNova's model trainer"""


class FedNovaPerIterTrainer(FedNovaTrainerAdapter, PerIterTrainer):
    """Per-iter variant of FedNova's model trainer"""

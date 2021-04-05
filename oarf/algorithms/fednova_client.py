import torch
from oarf.compressors.compressors import NoCompress
from oarf.compressors.compress_utils import pack_float32
from oarf.algorithms.fedavg import FedAvgClient
from oarf.communications.protocol import MessageType
from oarf.models.models import init_model
from oarf.utils.training import (
    ModelTrainer, init_buffer, PerIterModelTrainer, PerEpochModelTrainer)


class FedNovaClient(FedAvgClient):
    def init_training(
            self, batchnorm_runstat, momentum, average_every, step_weights,
            gradient_correction, model, *args, **kwargs):
        super().init_training(
            batchnorm_runstat=batchnorm_runstat, momentum=momentum,
            average_every=average_every, step_weights=step_weights,
            model=model, *args, **kwargs)

        self.momentum = momentum
        self.average_every = average_every
        self.gradient_correction = gradient_correction

        # assign setp_weights (\vec{a}_i) for fednova
        if isinstance(step_weights, (int, float)):
            self.step_weights = [step_weights] * average_every
        elif isinstance(step_weights, list):
            assert len(step_weights) == average_every
            for weight in step_weights:
                assert isinstance(weight, (int, float))
            self.step_weights = step_weights

        # prepare gradient correction buffer for fednova.
        # similar to err feedback, but for correcting errs in the fednova proto
        if gradient_correction:
            self.gradient_correction = init_model(
                model, batchnorm_runstat=batchnorm_runstat,
                dataset=self.train_dataset)
            init_buffer(self.gradient_correction)
        else:
            self.gradient_correction = None

    def init_trainer(self, average_policy, id, delta, clip):
        if average_policy == 'iter':
            self.trainer = FedNovaPerIterTrainer(
                self.train_loader, (self.dp_type is not None),
                self.dp_noise_level, clip,
                self.tfboard_writer, "Client {}".format(id))
        elif average_policy == 'epoch':
            self.trainer = FedNovaPerEpochTrainer(
                self.train_loader, (self.dp_type is not None),
                self.dp_noise_level, clip,
                self.tfboard_writer, "Client {}".format(id))
        else:
            raise ValueError("Unkown average policy")

    def train_one_round(self):
        return self.trainer.train_model(
            self.gradient_correction, self.net, self.optimizer,
            self.train_criterion, self.additional_criteria, self.iters)

    def update_global_params(self, server_momentum_masking):
        super().update_global_params(server_momentum_masking)

        # if gradient correction is enabled, server will also send the weighted
        # sum of normalized gradient
        if self.gradient_correction:
            msg_type, data = self.dispatcher.recv_msg()
            self.handle_bye(msg_type)
            assert msg_type == MessageType.WEIGHTED_SUM_GRADIENT

            global_grad = NoCompress().decompress(data)
            for gc, grad, gg in zip(
                    self.gradient_correction, self.gradient, global_grad):
                gc.set_(gg - grad)

    def calc_gradient(self):
        super().calc_gradient()

        # correctly average all the step weights, including the momentum
        self.step_weights_norm = 0.
        if self.momentum > 0:
            self.step_weights_norm = sum([
                abs(a) * self.momentum *
                (1 - (self.momentum ** (self.average_every - i))) /
                (1 - self.momentum) for i, a in enumerate(self.step_weights)])
        else:
            self.step_weights_norm = sum([abs(a) for a in self.step_weights])

        self.log.debug("ǁaᵢǁ = {}".format(self.step_weights_norm))
        # convert gradient to normalized gradient
        for grad in self.gradient.parameters():
            grad.div_(self.step_weights_norm)

    def aggregate(self, *args, **kwargs):
        super().aggregate(*args, **kwargs)
        self.dispatcher.send_msg(
            MessageType.STEP_WEIGHTS, pack_float32(self.step_weights_norm))
        # TODO: step_weights_norm should be multiplied to gradient if secure aggregation is used.


class FedNovaTrainerAdapter:
    """
    An adapter to ModelTrainer, hooks model updating & gradient calculation
    process to accumulate gradient.
    """

    def train_model(self, gradient_correction=None, *args, **kwargs):
        """add hooks to the train_model method of the adapted class"""
        self.gradient_correction = gradient_correction
        return super().train_model(*args, **kwargs)

    def calc_grad(self, *argc, **argv):
        """add hooks to incorporate gradient correction"""
        res = super().calc_grad(*argc, **argv)
        if self.gradient_correction is not None:
            with torch.no_grad():
                for param, corr in zip(self.net, self.gradient_correction):
                    param.grad.add_(corr)
        return res


class FedNovaPerEpochTrainer(FedNovaTrainerAdapter, PerEpochModelTrainer):
    pass


class FedNovaPerIterTrainer(FedNovaTrainerAdapter, PerIterModelTrainer):
    pass

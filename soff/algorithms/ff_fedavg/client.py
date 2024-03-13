"""Full-feature FedAvg algorithm client"""
import torch
from .. import fedavg
from ..base.ff_base_client import FFBaseClient
from ...communications.protocol import MessageType
from ...compressors.none import NoCompress
from ...utils.training import all_params
from ...models import create_model_trainer
from ...models.base import PerEpochTrainer, PerIterTrainer


class Client(fedavg.Client, FFBaseClient):
    """Full feature client for the FedAvg algorithm"""

    def __init__(self, cfg):
        super().__init__(cfg)

        # Note the iterations number when the model is last tested
        self._last_test_iters = 0
        self.optimizer_global = torch.optim.SGD(
            self.net_global.parameters(), lr=1.0)

        self.trainer = create_model_trainer(
            cfg, self.train_loader, self.train_criterion,
            self.additional_metrics, self.datalogger)

    def update_global_params(self):
        msg_type, data = self.dispatcher.recv_msg()
        self.handle_bye(msg_type)
        if msg_type == MessageType.GRADIENT:
            # update global net using global grad
            gradients = NoCompress().decompress(data)
            self.net_global.train()

            # update parameters and buffers
            with torch.no_grad():
                for param, grad in zip(self.net_global.parameters(),
                                       gradients[:self.num_params]):
                    param.grad = grad.to(param.device)
                for buf, grad in zip(self.net_global.buffers(),
                                     gradients[self.num_params:]):
                    buf.copy_(buf - grad.to(buf.device))
            self.optimizer_global.step()
            self.net_global.eval()
        elif msg_type == MessageType.MODEL:
            params = NoCompress().decompress(data)
            with torch.no_grad():
                for g_param, param in zip(all_params(self.net_global), params):
                    g_param.copy_(param.to(self.devices[0]))
        else:
            raise Exception("Client can only handle gradient or model.")

    def aggregate(self):
        # sent gradient info to aggregator
        data = NoCompress().compress(
            self.secure_exchange_splits(all_params(self.gradient))
            if self.secure_aggregation else
            all_params(self.gradient))

        # send gradient to server
        self.log.info("Sending gradient (%s bytes)", len(data))
        self.dispatcher.send_msg(MessageType.GRADIENT, data)

    def update_epoch_number(self) -> None:
        if isinstance(self.trainer, PerEpochTrainer):
            super().update_epoch_number()
            return
        assert isinstance(self.trainer, PerIterTrainer)
        self.epochs = self.iters // len(self.train_loader)

    def test_model(self):
        if isinstance(self.trainer, PerEpochTrainer):
            super().test_model()
            return

        assert isinstance(self.trainer, PerIterTrainer)
        if ((self.iters - self._last_test_iters)
                / self.cfg.fedavg.average_every
                // self.cfg.federation.test_every > 0):
            self._last_test_iters = self.iters
            self._test_model()

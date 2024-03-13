"""Full-feature FedNova algorithm"""
from .. import ff_fedavg
from .. import fednova
from ..base.ff_base_server import FFBaseServerConfParser
from ...models.base import PerEpochTrainer, PerIterTrainer


class ServerConfParser(
        fednova.ServerConfParser, FFBaseServerConfParser):
    """Server config parser for the fednova algorithm"""


class Server(fednova.Server, ff_fedavg.Server):
    """Full-feature FedNova algorithm server"""
    @classmethod
    def conf_parser(cls):
        return ServerConfParser

class ClientConfParser(fednova.ClientConfParser):
    """Client config parser for the fednova algorithm"""


class Client(fednova.Client, ff_fedavg.Client):
    """Full-feature fednova client"""

    def __init__(self, cfg):
        super().__init__(cfg)

        if isinstance(self.trainer, PerEpochTrainer):
            self.trainer = fednova.PerEpochTrainer(
                cfg, self.train_loader, self.train_criterion,
                self.additional_metrics, self.datalogger)
        if isinstance(self.trainer, PerIterTrainer):
            self.trainer = fednova.PerIterTrainer(
                cfg, self.train_loader, self.train_criterion,
                self.additional_metrics, self.datalogger)
        else:
            raise RuntimeError("Model trainer not initailized")

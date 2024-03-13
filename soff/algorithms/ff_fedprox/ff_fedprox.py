from .. import ff_fedavg
from .. import fedprox
from ..fedprox.fedprox import FedProxPerEpochTrainer, FedProxPerIterTrainer


class ConfParser(fedprox.ConfParser, ff_fedavg.ConfParser):
    """Full-feature FedProx config parser"""


class Server(ff_fedavg.Server):
    @classmethod
    def conf_parser(cls):
        return ConfParser


class Client(fedprox.Client, ff_fedavg.Client):
    """Full feature FedProx algorithm's client """

    def __init__(self, cfg):
        super().__init__(cfg)

        if cfg.training.model_trainer.name == 'per-epoch':
            self.trainer = FedProxPerEpochTrainer(
                cfg, self.train_loader, self.train_criterion,
                self.additional_metrics, self.datalogger)
        if cfg.training.model_trainer.name == 'per-iter':
            self.trainer = FedProxPerIterTrainer(
                cfg, self.train_loader, self.train_criterion,
                self.additional_metrics, self.datalogger)
        else:
            raise RuntimeError("Unknown model trainer")

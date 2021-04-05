# should be a special case of fednova, see fednova paper
import torch
from oarf.algorithms.fedavg import FedAvgServer, FedAvgClient
from oarf.utils.training import PerEpochModelTrainer, PerIterModelTrainer

FedProxServer = FedAvgServer


class FedProxClient(FedAvgClient):
    def init_training(self, mu, *args, **kwargs):
        self.mu = mu
        super().init_training(*args, **kwargs)

    def init_trainer(self, average_policy, id, delta, clip):
        if average_policy == 'iter':
            self.trainer = FedProxPerIterTrainer(
                self.mu,
                self.train_loader, (self.dp_type is not None),
                self.dp_noise_level, clip,
                self.tfboard_writer, "Client {}".format(id))
        elif average_policy == 'epoch':
            self.trainer = FedProxPerEpochTrainer(
                self.mu,
                self.train_loader, (self.dp_type is not None),
                self.dp_noise_level, clip,
                self.tfboard_writer, "Client {}".format(id))
        else:
            raise ValueError("Unkown average policy")

    def train_one_round(self):
        return self.trainer.train_model(
            self.net_global, self.net, self.optimizer,
            self.train_criterion, self.additional_criteria, self.iters)


class FedProxTrainerAdapter:
    """
    An adapter to ModelTrainer, hooks model updating & gradient calculation
    process to accumulate gradient.
    """

    def __init__(self, mu, *args, **kwargs):
        """
        mu: μ value of fedprox
        """
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.global_model = None

    def train_model(self, global_model=None, *args, **kwargs):
        self.global_model = global_model
        return super().train_model(*args, **kwargs)

    def calc_grad(self, net, datas, labels, train_criterion):
        """add hooks to incorporate gradient correction"""
        predictions = net(datas)
        loss = train_criterion(predictions, labels)
        # add some salt
        for param, g_param in \
                zip(net.parameters(), self.global_model.parameters()):
            loss += (self.mu / 2 * torch.norm(param - g_param) ** 2)
        loss.backward()
        return predictions, loss


class FedProxPerEpochTrainer(FedProxTrainerAdapter, PerEpochModelTrainer):
    pass


class FedProxPerIterTrainer(FedProxTrainerAdapter, PerIterModelTrainer):
    pass


__all__ = ['FedProxServer', 'FedProxClient']

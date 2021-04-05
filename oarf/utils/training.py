import os
import re
import inspect
import torch
import random
import logging
import numpy as np
from itertools import chain
from typing import Callable, List
from oarf.utils.module import Module
from oarf.privacy.rdp import add_gaussian_noise, clip_gradients
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(int(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Deterministic(Module):
    """Any class that needs deterministic behavior can inherit this class"""

    def __init__(self, model, seed, **kwargs):
        super().__init__(**kwargs, seed=seed, model=model)
        # set deterministic behavior
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        seed_everything(seed)

        if re.match('^vgg.*', model.lower()) is not None:
            log.warning("VGG requires non-deterministic implementation, "
                        "disabling pytorch determinism...")
            return

        torch.set_deterministic(True)
        torch.backends.cudnn.deterministic = True


optimizer_name_map = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'RMSprop': torch.optim.RMSprop,
}


def init_optimizer(name: str, *args, **kwargs):
    valid_args = set(
        inspect.signature(optimizer_name_map[name]).parameters.keys())
    return optimizer_name_map[name](
        *args, **{k: v for k, v in kwargs.items() if k in valid_args})


class ModelTrainer:
    def __init__(self, data_loader: torch.utils.data.DataLoader,
                 use_dp=False, noise_level=0, clip=None,
                 tfboard_writer: SummaryWriter = None, tag_suffix: str = ""):

        self.data_loader = data_loader
        self.iter = iter(data_loader)

        self.tfboard_writer = tfboard_writer
        self.tag_suffix = tag_suffix

        self.losses = []
        self.additional_results = []

        self.use_dp = use_dp
        self.noise_level = noise_level
        self.clip = clip

    def train_model(
            self, net: torch.nn.Module, optimizer: torch.optim.Optimizer,
            train_criterion, additional_criteria: List[Callable], iters: int,
            *_, **__):
        raise NotImplementedError("please use a subclass of ModelTrainer")

    def calc_grad(self, net, datas, labels, train_criterion):
        predictions = net(datas)
        loss = train_criterion(predictions, labels)
        loss.backward()
        return predictions, loss

    def train_one_step(
            self, net, datas, labels, train_criterion, optimizer, iters):

        optimizer.zero_grad()
        res = self.calc_grad(net, datas, labels, train_criterion)

        if self.use_dp:
            # clip and add dp noise
            if self.clip is not None:
                clip_gradients(net, self.clip)

            log.debug("Adding gaussian noise")
            for param in net.parameters():
                add_gaussian_noise(
                    param.grad,
                    self.data_loader.batch_size,
                    self.noise_level, self.clip)

        optimizer.step()
        return res

    def log_training(self, iters, loss, additional_criteria: List[Callable],
                     additional_results: List[float]):
        self.losses.append(loss.data)
        for i, r in enumerate(additional_results):
            if i >= len(self.additional_results):
                self.additional_results.append([r])
            else:
                self.additional_results[i].append(r)

        if iters % 50 == 0:
            train_loss = float(torch.mean(torch.Tensor(self.losses)))
            additional_results = [float(torch.mean(torch.Tensor(r)))
                                  for r in self.additional_results]

            log.info(
                "  iter {} ({}/{})".format(
                    iters, iters % len(self.data_loader),
                    len(self.data_loader)))
            log.info("    Train loss: {}".format(train_loss))
            for c, r in zip(additional_criteria, additional_results):
                log.info("    Train {}: {}".format(c.__name__, r))

            if self.tfboard_writer is not None:
                self.tfboard_writer.add_scalar(
                    "Train:loss/" + self.tag_suffix, train_loss, iters)
                for c, r in zip(additional_criteria, additional_results):
                    self.tfboard_writer.add_scalar(
                        "Train:{}/".format(c.__name__) + self.tag_suffix,
                        r, iters)

            self.losses = []
            for i in range(len(self.additional_results)):
                self.additional_results[i] = []


class PerIterModelTrainer(ModelTrainer):
    def train_model(
            self, net: torch.nn.Module, optimizer: torch.optim.Optimizer,
            train_criterion, additional_criteria: List[Callable], iters: int,
            *_, **__):

        datas, labels = next(self.iter, (None, None))
        datas, labels = datas.cuda(), labels.cuda()
        if datas is None:
            self.iter = iter(self.data_loader)
            datas, labels = next(self.iter)

        iters += 1

        predictions, loss = self.train_one_step(
            net, datas, labels, train_criterion, optimizer, iters)

        additional_results = [
            c(predictions, labels) for c in additional_criteria]
        self.log_training(iters, loss, additional_criteria, additional_results)

        return iters


class PerEpochModelTrainer(ModelTrainer):
    def train_model(
            self, net: torch.nn.Module, optimizer: torch.optim.Optimizer,
            train_criterion, additional_criteria: List[Callable], iters: int,
            *_, **__):

        for datas, labels in self.data_loader:
            datas, labels = datas.cuda(), labels.cuda()
            iters += 1

            predictions, loss = self.train_one_step(
                net, datas, labels, train_criterion, optimizer, iters)

            additional_results = [
                c(predictions, labels) for c in additional_criteria]
            self.log_training(
                iters, loss, additional_criteria, additional_results)

            if "QUICK_DEBUG" in os.environ:
                log.warning("QUICK_DEBUG enabled, skipping training...")
                break

        return iters


def evaluate_model(
        prefix: str,
        model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
        loss_criterion: Callable, additional_criteria: List[Callable],
        tfboard_writer: SummaryWriter = None, tag_suffix: str = "",
        epoch: int = None) -> (float, List[float]):
    """
    return: (loss, [results for additional_criteria])
    """
    model_is_training = model.training
    model.eval()
    with torch.no_grad():
        losses = []
        additional_metrics = [[] for _ in additional_criteria]
        for datas, labels in data_loader:
            datas, labels = datas.cuda(), labels.cuda()
            predictions = model(datas)

            losses.append(loss_criterion(predictions, labels))
            for i, c in enumerate(additional_criteria):
                additional_metrics[i].append(c(predictions, labels))

        loss = float(torch.mean(torch.Tensor(losses)))
        log.info("  {} loss: {}".format(prefix, loss))

        additional_results = []
        for c, m in zip(additional_criteria, additional_metrics):
            additional_results.append(float(torch.mean(torch.Tensor(m))))
            log.info("  {} {}: {}".format(
                prefix, c.__name__, additional_results[-1]))

        if tfboard_writer is not None:
            log.debug("Writing to tensorboard")
            tfboard_writer.add_scalar(
                "{}:loss/".format(prefix) + tag_suffix, loss, epoch)
            for c, r in zip(additional_criteria, additional_results):
                tfboard_writer.add_scalar(
                    "{}:{}/".format(prefix, c.__name__) + tag_suffix, r, epoch)

    # restore model state
    if model_is_training:
        model.train()

    # return global model loss
    return loss, additional_results


def all_params(net: torch.nn.Module):
    return list(chain(net.parameters(), net.buffers()))


def init_buffer(buf: torch.nn.Module):
    buf.cuda()
    for param in buf.parameters():
        param.requires_grad_(False)
    for param in all_params(buf):
        param.zero_()


def do_svd_analysis(
        grad: torch.nn.Module, tfboard_writer: SummaryWriter, tag_suffix: str,
        epoch: int):
    """
    grad: gradient stored in network's parameters
    """
    for name, param in grad.named_parameters():
        # filter out all batchnorm layers
        if 'batchnorm' not in name:
            # Do SVD with unbiased gradients
            diag = torch.svd(
                param.view(param.shape[0], -1).transpose(0, 1) -
                torch.mean(param), compute_uv=False)[1]

            log.debug(
                "SVD for layer {} = {}".format(
                    str(name).replace('.', '_'),
                    '[' + ', '.join(map(str, diag.tolist())) + ']'))

            # need to multiply the diag to get the variance

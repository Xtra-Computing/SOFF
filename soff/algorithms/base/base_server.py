"""Base class of all FL servers"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict
from munch import Munch
import torch
from torch.utils.data import DataLoader
from .base import Node
from .base_server_scheduler import (
    StaticBaseServerScheduler, StaticBaseServerSchedulerConfParser)
from ...utils.scheduler import LRSchedulerConfParser
from ...utils.training import init_determinism
from ...utils.logging import LogConfParser, DataLogger
from ...utils.optimizer import OptimizerConfParser
from ...datasets import DataConfParser, create_dataset, _FLSplit
from ...datasets.raw_dataset import dataset_name_to_class
from ...models import ModelConfParser
from ...utils.metrics import MetricsConfParser
from ...utils.arg_parser import FLConfParser, HWConfParser, TrainingConfParser
from ...communications.dispatcher import ServerDispatcherConfParser


class BaseServerConfParser(
        LogConfParser, HWConfParser, ModelConfParser, DataConfParser,
        MetricsConfParser, OptimizerConfParser, LRSchedulerConfParser,
        TrainingConfParser, ServerDispatcherConfParser, FLConfParser):
    """Parse base configs that will used by the server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_args = self.add_argument_group("Server-Only Configs (S)")
        self.server_args.add_argument(
            '-sv.nc', '--server.num-cache',
            default=5, type=int, metavar='N',
            help="# of received client models that exist at the same time\n"
            " in memory, 2 * nc should be less than total host memory size")

    @staticmethod
    def filter_server_cfg(cfg: dict) -> Munch:
        """Remove all arguments under the namesapce `server.`"""
        # Filter out server-specific configs
        return Munch.fromDict({
            k: v for (k, v) in cfg.items()
            if k not in {'server', 'log', 'hardware', 'dispatcher'}})


class BaseServer(Node, ABC):
    """Base server for federated learning"""
    @classmethod
    def conf_parser(cls):
        return BaseServerConfParser

    def __init__(self, cfg, scheduler):
        super().__init__()

        # Store config
        self.cfg = cfg

        # Initialize CUDA context on the first listed GPU
        if len(cfg.hardware.gpus) > 0:
            torch.cuda.device(cfg.hardware.gpus[0])

        # Note down devices to use
        self.devices = ([
            torch.device('cuda', gpu_idx) for gpu_idx in cfg.hardware.gpus]
            if len(cfg.hardware.gpus) > 0 else [torch.device('cpu')])

        # Initialize logging and tensorboard writer
        self.log = logging.getLogger(self.__class__.__name__)

        # Initialize infrastructures
        init_determinism(cfg)

        self.datalogger = DataLogger(cfg, 'Server')
        self.log.info("\n%s", json.dumps(cfg, indent=2))

        # Initialize server scheduler
        self.scheduler = scheduler and scheduler(
            cfg, self.conf_parser().filter_server_cfg(cfg))

    @abstractmethod
    def start_training(self, cfg):
        """Subclass should override this method to define the training alg."""
        self.scheduler.register_cleanup_hook(self.cleanup_hook)
        self.scheduler.start()

    def cleanup(self):
        self.scheduler.cleanup()

    def cleanup_hook(self):
        """Cleanup function. Called at dispatcher shutdown"""
        self.datalogger.close()


class DatasetInitializedServer(BaseServer):
    """Specific interface of BaseServer with initialized eval/test datasets"""

    def __init__(self, cfg, scheduler):
        super().__init__(cfg, scheduler)
        self.eval_dataset: _FLSplit
        self.test_datasets: Dict[str, _FLSplit]
        self.eval_loader: DataLoader
        self.test_loaders: Dict[str, DataLoader]
        self._init_dataset(cfg)

    @abstractmethod
    def _init_dataset(self, cfg: Munch):
        raise NotImplementedError("_init_dataset not implemented")


class ClientServerBaseServerConfParser(
        BaseServerConfParser, StaticBaseServerSchedulerConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cs_args = self.add_argument_group(
            "Client/Server Architecture Configs (S,S->C)")

        cs_args.add_argument(
            '-cs.n', '--client-server.num-clients',
            type=int, metavar='N',
            help="Total number of clients involved in training")

        cs_args.add_argument(
            '-cs.td', '--client-server.test-datasets',
            nargs='+', metavar="DATASET",
            choices=dataset_name_to_class.keys(),
            help="Seleted test datasets must have the same prefix, "
            "Combined test dataset can be separted with ',' (without space), "
            "e.g. 'amazon,imdb'.\n"
            "Will test on all datasets if multiple datasets are specified.\n"
            "Will use the test set tied to the train set if set to None.")


class ClientServerBaseServer(DatasetInitializedServer):
    """Base server for federated learning following client-server paradigm"""

    @classmethod
    def conf_parser(cls):
        return ClientServerBaseServerConfParser

    def __init__(self, cfg, scheduler=StaticBaseServerScheduler):
        super().__init__(cfg, scheduler)
        assert cfg.client_server.num_clients == cfg.data.fl_split.num, (
            "Number of clients must match number of data splits")

    def _init_dataset(self, cfg: Munch):
        """Initialize test and eval dataset for the server"""
        self.eval_dataset = create_dataset(
            cfg, datasets=cfg.data.raw.datasets, mode='eval', split_id=None)

        test_datasets = (
            cfg.client_server.test_datasets
            if cfg.client_server.test_datasets
            else [','.join(cfg.data.raw.datasets)])

        self.test_datasets = {
            tdss: create_dataset(
                cfg, datasets=tdss.split(','), mode='test', split_id=None)
            for tdss in test_datasets
        }

        eval_g = torch.Generator()
        eval_g.manual_seed(cfg.federation.seed)
        self.eval_loader = DataLoader(
            self.eval_dataset, cfg.training.batch_size,
            shuffle=False, generator=eval_g)

        test_gs = {k: torch.Generator() for k in self.test_datasets.keys()}
        for generator in test_gs.values():
            generator.manual_seed(cfg.federation.seed)

        self.test_loaders = {
            k: DataLoader(
                v, cfg.training.batch_size,
                shuffle=False, generator=test_gs[k])
            for k, v in self.test_datasets.items()
        }


class HierarchicalBaseServerConfParser(
        BaseServerConfParser, StaticBaseServerSchedulerConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hc_args = self.add_argument_group(
            "Hierarchical Architecture Configs (S,S->T/C,T->C)")

        hc_args.add_argument(
            '-hc.td', '--hierarchical.test-datasets',
            nargs='+', metavar="DATASET",
            choices=dataset_name_to_class.keys(),
            help="Seleted test datasets must have the same prefix, "
            "Combined test dataset can be separted with ',' (without space), "
            "e.g. 'amazon,imdb'.\n"
            "Will test on all datasets if multiple datasets are specified.\n"
            "Will use the test set tied to the train set if set to None.")


class HierarchicalBaseServer(DatasetInitializedServer):
    @classmethod
    def conf_parser(cls):
        return HierarchicalBaseServerConfParser

    def __init__(self, cfg, scheduler=StaticBaseServerScheduler):
        super().__init__(cfg, scheduler)

    def _init_dataset(self, cfg: Munch):
        """Initialize test and eval dataset for the server"""
        self.eval_dataset = create_dataset(
            cfg, datasets=cfg.data.raw.datasets, mode='eval', split_id=None)

        test_datasets = (
            cfg.hierarchical.test_datasets
            if cfg.hierarchical.test_datasets
            else [','.join(cfg.data.raw.datasets)])

        self.test_datasets = {
            tdss: create_dataset(
                cfg, datasets=tdss.split(','), mode='test', split_id=None)
            for tdss in test_datasets
        }

        eval_g = torch.Generator()
        eval_g.manual_seed(cfg.federation.seed)
        self.eval_loader = DataLoader(
            self.eval_dataset, cfg.training.batch_size,
            shuffle=False, generator=eval_g)

        test_gs = {k: torch.Generator() for k in self.test_datasets.keys()}
        for generator in test_gs.values():
            generator.manual_seed(cfg.federation.seed)

        self.test_loaders = {
            k: DataLoader(
                v, cfg.training.batch_size,
                shuffle=False, generator=test_gs[k])
            for k, v in self.test_datasets.items()
        }

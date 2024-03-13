"""Base class of all FL clients for the C/S architecture"""
import argparse
import copy
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Sequence, Callable
import torch
from munch import Munch
from torch.utils.data import DataLoader
from .base import Node
from ...communications.protocol import MessageType, ForwardedConfig
from ...utils.arg_parser import BaseConfParser, HWConfParser
from ...utils.logging import DataLogger, LogConfParser
from ...utils.training import init_determinism
from ...datasets import create_dataset, _FLSplit
from ...communications.dispatcher import (
    ClientDispatcherConfParser, ClientDispatcher)


class BaseClientConfParser(
        LogConfParser, HWConfParser, ClientDispatcherConfParser):
    """Parse base arguments that will used by clients only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        client_args = self.add_argument_group(
            "Client-Only Configs (C)")

        client_args.add_argument(
            '-dbg.st', '--debug.skip-training',
            default=argparse.SUPPRESS, action='store_true',
            help="Skip the training process (for debug)")


class ResourceManagedClient(Node, ABC):
    """
    All client classes to be managed by MultiClientSimulator need to inherit
    and implement the interface of this class and use `yield_gpu_and_wait`
    to claim GPU resources before using them.

    When Used independently without MultiClient, `yield_resource_and_wait` will
    do nothing to avoid unnecessary load to/offload from GPUs.
    """

    def __init__(self, cfg) -> None:

        self.acquire_resource_fn: Callable[[Sequence[int]], None]
        """The function to acquire resource"""
        self.release_resource_fn: Callable[[Sequence[int]], None]
        """The function to release resource"""
        self.__resource_loaded: bool = False
        """Load resource once for non-multithreading launcher"""

        # Initialize CUDA context on the first listed GPU
        if 'hardware' in cfg and len(cfg.hardware.gpus) > 0:
            torch.cuda.device(cfg.hardware.gpus[0])

        # Note down devices to use
        self.devices = ([
            torch.device('cuda', gpu_idx) for gpu_idx in cfg.hardware.gpus]
            if 'hardware' in cfg and len(cfg.hardware.gpus) > 0
            else [torch.device('cpu')])

        # Create unique stream for synchronization, required by multi-cli
        if self.devices[0].type != 'cpu':
            self.stream = torch.cuda.Stream(self.devices[0])

    def unload_resources(self):
        """
        Implement this function to unload resources from GPU memory.
        Call super().unload_resources() at last.
        """
        # Avoid initialize cuda context on GPU 0
        # see https://github.com/pytorch/pytorch/issues/25752
        with torch.cuda.device(self.devices[0]):
            self.stream.synchronize()
            torch.cuda.empty_cache()

    def load_resources(self):
        """
        Implement this function to load resource to GPU memory
        call super().load_resources() at first.
        """

    def acquire_and_load_resources(self, gpu_indices: Sequence[int]):
        """Acquire gpus, then load local resources to gpus"""
        if self.is_launched_by_multithread_launcher():
            self.acquire_resource_fn(gpu_indices)
            self.load_resources()
        elif not self.__resource_loaded:
            # If not launched w/ multithreading, Load resource once and for all
            self.load_resources()
            self.__resource_loaded = True

    def unload_resources_and_release(self, gpu_indices: Sequence[int]):
        """unloaed resources from gpus, then release gpus"""
        if self.is_launched_by_multithread_launcher():
            self.unload_resources()
            self.release_resource_fn(gpu_indices)

    def is_launched_by_multithread_launcher(self) -> bool:
        return (
            hasattr(self, 'acquire_resource_fn') and
            hasattr(self, 'release_resource_fn'))


class BaseClient(ResourceManagedClient, ABC):
    """Base client for federated learning"""
    @classmethod
    def conf_parser(cls):
        return BaseClientConfParser

    def __init__(self, cfg: Munch):
        super().__init__(cfg)

        # Initialize logger
        self.log = logging.getLogger(self.__class__.__name__)

        # Client is blocking. Just use a raw dispatcher. No scheduler needed.
        self.dispatcher = ClientDispatcher(cfg)
        self.dispatcher.start()

        # Send handshake (register) message
        self.dispatcher.send_msg(MessageType.HANDSHAKE, bytearray(b'Hello!'))

        # Receive config, initialize client_id
        msg_type, data = self.dispatcher.recv_msg()
        assert msg_type == MessageType.TRAINING_CONFIG
        msg = ForwardedConfig().decode(data)

        # Note down assigned client id
        self.client_id = msg.data.client_id

        # Use merged config to initialize everything else. Modify cfg in-place.
        # Make directly specified args having higher priority
        client_cfg = copy.deepcopy(cfg)
        cfg.clear()

        cfg.update(BaseConfParser.merge(msg.data, client_cfg))

        # Store merged config
        self.cfg = cfg

        # Initialize infrastructures
        init_determinism(cfg)
        self.log.info('\n%s', json.dumps(cfg, indent=2))

        self.train_dataset: _FLSplit
        self.train_loader: DataLoader
        self.test_datasets: Dict[str, _FLSplit]
        self.test_loaders: Dict[str, DataLoader]
        self._init_dataset(cfg)

        # Initialize logging
        self.datalogger = DataLogger(cfg, f"Client{msg.data.client_id}")

    @abstractmethod
    def _init_dataset(self, cfg):
        raise NotImplementedError

    @abstractmethod
    def start_training(self, cfg):
        """Subclass should override this method to define the trianing alg."""
        raise NotImplementedError

    def cleanup(self):
        """Cleanup function. Called at client exit"""
        self.datalogger.close()


class ClientServerBaseClient(BaseClient):
    def _init_dataset(self, cfg):
        # we leave eval to server, only using train/test here
        self.train_dataset = create_dataset(
            cfg, datasets=cfg.data.raw.datasets, mode='train',
            split_id=cfg.client_id)

        test_datasets = cfg.client_server.test_datasets \
            if cfg.client_server.test_datasets \
            else [','.join(cfg.data.raw.datasets)]

        self.test_datasets = {
            tdss: create_dataset(
                cfg, datasets=tdss.split(','), mode='test',
                split_id=None)
            for tdss in test_datasets
        }

        train_g = torch.Generator()
        train_g.manual_seed(cfg.federation.seed)
        self.train_loader = DataLoader(
            self.train_dataset, cfg.training.batch_size,
            shuffle=False, generator=train_g)

        test_gs = {k: torch.Generator() for k in self.test_datasets.keys()}
        for generator in test_gs.values():
            generator.manual_seed(cfg.federation.seed)

        self.test_loaders = {
            k: DataLoader(
                v, cfg.training.batch_size,
                shuffle=False, generator=test_gs[k])
            for k, v in self.test_datasets.items()
        }


class HierarchicalBaseClient(BaseClient):
    def _init_dataset(self, cfg):
        # we leave eval to server, only using train/test here
        self.train_dataset = create_dataset(
            cfg, datasets=cfg.data.raw.datasets, mode='train',
            split_id=cfg.client_id)

        test_datasets = cfg.hierarchical.test_datasets \
            if cfg.hierarchical.test_datasets \
            else [','.join(cfg.data.raw.datasets)]

        self.test_datasets = {
            tdss: create_dataset(
                cfg, datasets=tdss.split(','), mode='test',
                split_id=None)
            for tdss in test_datasets
        }

        train_g = torch.Generator()
        train_g.manual_seed(cfg.federation.seed)
        self.train_loader = DataLoader(
            self.train_dataset, cfg.training.batch_size,
            shuffle=False, generator=train_g)

        test_gs = {k: torch.Generator() for k in self.test_datasets.keys()}
        for generator in test_gs.values():
            generator.manual_seed(cfg.federation.seed)

        self.test_loaders = {
            k: DataLoader(
                v, cfg.training.batch_size,
                shuffle=False, generator=test_gs[k])
            for k, v in self.test_datasets.items()
        }

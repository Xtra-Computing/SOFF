import copy
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from munch import Munch
from torch.utils.data import DataLoader

from .base import Node
from .base_server_scheduler import (
    BaseServerScheduler, StaticBaseServerScheduler,
    StaticBaseServerSchedulerConfParser)
from ...models import ModelConfParser
from ...utils.metrics import MetricsConfParser
from ...utils.optimizer import OptimizerConfParser
from ...utils.scheduler import LRSchedulerConfParser
from ...datasets import DataConfParser, create_dataset
from ...datasets.raw_dataset import dataset_name_to_class
from ...utils.logging import LogConfParser, DataLogger
from ...utils.training import init_determinism
from ...utils.arg_parser import BaseConfParser, FLConfParser, Tagged, TrainingConfParser
from ...utils.arg_parser import HWConfParser
from ...communications.protocol import ForwardedConfig, MessageType, MetaData
from ...communications.dispatcher import (
    ClientDispatcherConfParserAdapter, DispatcherConfParserAdapter,
    ServerDispatcherConfParser, ClientDispatcherConfParser,
    ClientDispatcher, ServerDispatcherConfParserAdapater)


class BaseTransceiverConfParser(BaseConfParser):
    @staticmethod
    def filter_transceiver_cfg(cfg: dict) -> Munch:
        """remove all arguments under the namespace transceiver"""
        return Munch.fromDict({
            k: v for (k, v) in cfg.items()
            if k not in {'transceiver', 'log', 'hardware', 'dispatcher'}})


class HierarchicalBaseTransceiverConfParser(
        LogConfParser, HWConfParser,
        Tagged[ServerDispatcherConfParser, ('s', 'server')],
        Tagged[ClientDispatcherConfParser, ('c', 'client')],
        BaseTransceiverConfParser, StaticBaseServerSchedulerConfParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        transceiver_args = self.add_argument_group(
            "Transceiver-Only Configs (T)")

        transceiver_args.add_argument(
            '-tc.nc', '--transceiver.num-cache',
            default=5, type=int, metavar='N',
            help="# of received client models that exist at the same time\n"
            " in memory, 2 * nc should be less than total host memory size")

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
            "Will not initialize the test set it not set.")


class HierarchicalBaseTransceiver(Node, ABC):
    """Base transceiver for FL (act as both a server and a client)."""
    @classmethod
    def conf_parser(cls):
        return HierarchicalBaseTransceiverConfParser

    def __init__(self, cfg: Munch, scheduler=StaticBaseServerScheduler) -> None:
        super().__init__()

        # Initialize CUDA context on the first listed GPU
        if len(cfg.hardware.gpus) > 0:
            torch.cuda.device(cfg.hardware.gpus[0])

        # Note down devices to use
        self.devices = ([
            torch.device('cuda', gpu_idx) for gpu_idx in cfg.hardware.gpus]
            if len(cfg.hardware.gpus) > 0 else [torch.device('cpu')])

        # Initialize logging and tensorboard writer
        self.log = logging.getLogger(self.__class__.__name__)

        # Client dispatcher
        cli_dispatcher_cfg = copy.deepcopy(cfg)
        cli_dispatcher_cfg["dispatcher"] = cli_dispatcher_cfg.dispatcher.client
        self.client_dispatcher = ClientDispatcher(cli_dispatcher_cfg)
        self.client_dispatcher.start()

        # Send handshake (register) message
        self.client_dispatcher.send_msg(
            MessageType.HANDSHAKE, bytearray(b'Hello!'))

        # Receive config, initialize id
        msg_type, data = self.client_dispatcher.recv_msg()
        assert msg_type == MessageType.TRAINING_CONFIG
        msg = ForwardedConfig().decode(data)

        self.transceiver_id = msg.data.client_id
        transceiver_cfg = copy.deepcopy(cfg)

        cfg.clear()
        cfg.update(BaseConfParser.merge(msg.data, transceiver_cfg))
        self.cfg = cfg

        # Initialize infrastructures
        self.log.info("\n%s", json.dumps(cfg, indent=2))
        self.datalogger = DataLogger(cfg, f'Transceiver{self.transceiver_id}')
        init_determinism(cfg)

        # Server scheduler
        srv_scheduler_cfg = copy.deepcopy(cfg)
        srv_scheduler_cfg["dispatcher"] = srv_scheduler_cfg.dispatcher.server
        self.server_scheduler = scheduler(
            srv_scheduler_cfg,
            BaseTransceiverConfParser.filter_transceiver_cfg(cfg))

        self.__init_dataset(cfg)

    @abstractmethod
    def start_training(self, cfg):
        """Subclass should override this method to define the training alg."""
        self.server_scheduler.register_cleanup_hook(self.cleanup_hook)
        self.server_scheduler.start()

    def cleanup(self):
        self.server_scheduler.cleanup()

    def cleanup_hook(self):
        """Cleanup function. Called at dispatcher shutdown"""
        self.datalogger.close()

    def __init_dataset(self, cfg):
        """Initialize test and eval dataset for the transceiver"""
        self.eval_dataset = create_dataset(
            cfg, datasets=cfg.data.raw.datasets, mode='eval', split_id=None)

        eval_g = torch.Generator()
        eval_g.manual_seed(cfg.federation.seed)
        self.eval_loader = DataLoader(
            self.eval_dataset, cfg.training.batch_size,
            shuffle=False, generator=eval_g)

        test_datasets = (
            cfg.hierarchical.test_datasets
            if cfg.hierarchical.test_datasets
            else [','.join(cfg.data.raw.datasets)])

        self.test_datasets = {
            tdss: create_dataset(
                cfg, datasets=tdss.split(','), mode='test', split_id=None)
            for tdss in test_datasets
        }

        test_gs = {k: torch.Generator() for k in self.test_datasets.keys()}
        for generator in test_gs.values():
            generator.manual_seed(cfg.federation.seed)

        self.test_loaders = {
            k: DataLoader(
                v, cfg.training.batch_size,
                shuffle=False, generator=test_gs[k])
            for k, v in self.test_datasets.items()
        }


class DecentralizedBaseTransceiverConfParser(
        LogConfParser, HWConfParser, ModelConfParser, DataConfParser,
        MetricsConfParser, OptimizerConfParser, LRSchedulerConfParser,
        TrainingConfParser, FLConfParser,
        DispatcherConfParserAdapter,
        ServerDispatcherConfParserAdapater,
        ClientDispatcherConfParserAdapter,
        BaseTransceiverConfParser, StaticBaseServerSchedulerConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add server dispatcher configs
        self.add_dispatcher_args('/s', '/server')

        dc_args = self.add_argument_group(
            "Decentralized Transceiver-Only Configs (T)")

        dc_args.add_argument(
            '-tc.nc', '--transceiver.num-cache',
            default=5, type=int, metavar='N',
            help="# of received client models that exist at the same time\n"
            " in memory, 2 * nc should be less than total host memory size")

        dc_args.add_argument(
            '-tc.id', '--transceiver.id', type=int, required=True,
            help="For decentralized algorithm, the id is self-assigned.")

        # For client, there can be multiple dispatchers
        dc_args.add_argument(
            '-dc/c.s', '--dispatcher/client.socket-type',
            default='unix', help='<unix|tcp>', metavar='TYPE')
        dc_args.add_argument(
            '-dc/c.as', '--dispatcher/client.addresses',
            default=['0::/tmp/soff/fed-comm.sock'], type=str, nargs='+', metavar='ADDR',
            help="List of socket addresses (unix socket only for local "
            "simulation, for cross-machine support, use tcp socket). Number "
            "of addresses must equal to the nuber of nodes it connects to. "
            "Addresses must be specified in the format for <id>::<addr>")

        self.add_server_dispatcher_args('/s', '/server')
        self.add_client_dispatcher_args('/c', '/client')

        dc_args.add_argument(
            '-dc.td', '--decentralized.test-datasets',
            nargs='+', metavar="DATASET",
            choices=dataset_name_to_class.keys(),
            help="Seleted test datasets must have the same prefix, "
            "Combined test dataset can be separted with ',' (without space), "
            "e.g. 'amazon,imdb'.\n"
            "Will test on all datasets if multiple datasets are specified.\n"
            "Will not initialize the test set it not set.")


class DecentralizedBaseServerScheduler(BaseServerScheduler):
    def __init__(
            self, cfg: Munch, client_cfg: Optional[Munch] = None,
            datalogger: Optional[DataLogger] = None) -> None:
        super().__init__(cfg, client_cfg, datalogger)
        self.self_claimed_id_to_alloc_id_map: Dict[int, int] = {}
        self.alloc_id_to_self_claimed_id_map: Dict[int, int] = {}
        self.dispatcher.register_msg_event(
            MessageType.METADATA, self.process_metadata)

    def process_metadata(self, socket, data):
        claimed_id = MetaData().decode(data).data.id
        alloced_id = self.clients_socket_id_map[socket]
        self.self_claimed_id_to_alloc_id_map[claimed_id] = alloced_id
        self.alloc_id_to_self_claimed_id_map[alloced_id] = claimed_id


class DecentralizedBaseTransceiver(Node, ABC):
    @classmethod
    def conf_parser(cls):
        return DecentralizedBaseTransceiverConfParser

    def __init__(self, cfg: Munch, scheduler=DecentralizedBaseServerScheduler) -> None:
        super().__init__()

        # Initialize CUDA context on the first listed GPU
        if len(cfg.hardware.gpus) > 0:
            torch.cuda.device(cfg.hardware.gpus[0])

        # Note down devices to use
        self.devices = ([
            torch.device('cuda', gpu_idx) for gpu_idx in cfg.hardware.gpus]
            if len(cfg.hardware.gpus) > 0 else [torch.device('cpu')])

        # Initialize logging and tensorboard writer
        self.log = logging.getLogger(self.__class__.__name__)
        self.cfg = cfg

        # For decentralized algorthm, we need to start server scheduler first
        srv_scheduler_cfg = copy.deepcopy(cfg)
        srv_scheduler_cfg["dispatcher"] = srv_scheduler_cfg.dispatcher.server
        self.server_scheduler = scheduler(
            srv_scheduler_cfg,
            BaseTransceiverConfParser.filter_transceiver_cfg(cfg))
        self.server_scheduler.register_cleanup_hook(self.cleanup_hook)
        self.server_scheduler.start()

        # Intialize multiple client dispatchers
        self.client_dispatchers: Dict[int, ClientDispatcher] = {}
        for _addr in cfg.dispatcher.client.addresses:
            assert '::' in _addr, "Address format must be <id>::<addr>"
            cfg_ = copy.deepcopy(cfg)
            cli_id, addr = _addr.split('::')
            cfg_["dispatcher"] = cfg_.dispatcher.client
            cfg_.dispatcher["address"] = addr
            cfg_.dispatcher["socket_type"] = cfg.dispatcher.client.socket_type
            self.client_dispatchers[int(cli_id)] = ClientDispatcher(cfg_)
        # Id is self-assigned
        self.transceiver_id = cfg.transceiver.id
        # Note down this node's allocated id assigned by others
        self.assigned_transceiver_ids = dict()

        # Start dispatcher (cid for "self-claimed id")
        for scid, dispatcher in self.client_dispatchers.items():
            dispatcher.start()
            dispatcher.send_msg(MessageType.HANDSHAKE, bytearray(b'Hello!'))
            dispatcher.send_msg(
                MessageType.METADATA,
                MetaData().set_data({'id': self.transceiver_id}).encode())
            msg_type, data = dispatcher.recv_msg()
            assert msg_type == MessageType.TRAINING_CONFIG
            msg = ForwardedConfig().decode(data)
            self.assigned_transceiver_ids[scid] = msg.data.client_id

        # Initialize infrastructures
        self.log.info("\n%s", json.dumps(cfg, indent=2))
        self.datalogger = DataLogger(cfg, 'Node')
        init_determinism(cfg)

        self.__init_dataset(cfg)

    @abstractmethod
    def start_training(self, cfg):
        """Subclass should override this method to define the training alg."""

    def cleanup(self):
        self.server_scheduler.cleanup()

    def cleanup_hook(self):
        """Cleanup function. Called at dispatcher shutdown"""
        self.datalogger.close()

    def __init_dataset(self, cfg):
        """Initialize test and eval dataset for the transceiver"""
        self.train_dataset = create_dataset(
            cfg, datasets=cfg.data.raw.datasets, mode='train',
            split_id=cfg.transceiver.id)

        train_g = torch.Generator()
        train_g.manual_seed(cfg.federation.seed)
        self.train_loader = DataLoader(
            self.train_dataset, cfg.training.batch_size,
            shuffle=False, generator=train_g)

        self.eval_dataset = create_dataset(
            cfg, datasets=cfg.data.raw.datasets, mode='eval', split_id=None)

        eval_g = torch.Generator()
        eval_g.manual_seed(cfg.federation.seed)
        self.eval_loader = DataLoader(
            self.eval_dataset, cfg.training.batch_size,
            shuffle=False, generator=eval_g)

        test_datasets = (
            cfg.decentralized.test_datasets
            if cfg.decentralized.test_datasets
            else [','.join(cfg.data.raw.datasets)])

        self.test_datasets = {
            tdss: create_dataset(
                cfg, datasets=tdss.split(','), mode='test', split_id=None)
            for tdss in test_datasets
        }

        test_gs = {k: torch.Generator() for k in self.test_datasets.keys()}
        for generator in test_gs.values():
            generator.manual_seed(cfg.federation.seed)

        self.test_loaders = {
            k: DataLoader(
                v, cfg.training.batch_size,
                shuffle=False, generator=test_gs[k])
            for k, v in self.test_datasets.items()
        }

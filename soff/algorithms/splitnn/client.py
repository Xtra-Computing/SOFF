"""SplitNN (Vertical FL) algorithm client"""
import sys
import time
from munch import Munch
import torch
from .utils import SplitnnDataInitializer
from ..base.base_client import (
    BaseClient, BaseClientConfParser, ClientServerBaseClient)
from ...communications.protocol import ClientInfo, MessageType, SyncInfos
from ...compressors.none import NoCompress
from ...datasets import DataConfParser
from ...models import ModelConfParser, create_model
from ...utils.training import seed_everything
from ...utils.optimizer import create_optimizer


class ConfParser(ModelConfParser, DataConfParser, BaseClientConfParser):
    pass


class SplitnnClientCommunication(BaseClient):
    def send_client_infos(self):
        """Send client metadata to the aggregator"""
        self.dispatcher.send_msg(
            MessageType.CLIENT_INFOS, ClientInfo().set_data({
                'data_len': len(self.train_dataset),
            }).encode())

    def receive_sync_info(self):
        """receive info in the synchronization message"""
        msg_type, data = self.dispatcher.recv_msg()
        self.handle_bye(msg_type)
        assert msg_type == MessageType.SYNC_INFOS
        sync_info = SyncInfos().decode(data)
        return sync_info

    def receive_grad(self):
        msg_type, data = self.dispatcher.recv_msg()
        self.handle_bye(msg_type)
        assert msg_type == MessageType.GRADIENT
        grad = NoCompress().decompress(data)
        return grad

    def handle_bye(self, msg_type):
        """Client exit if server exists"""
        if msg_type in {MessageType.BYE, MessageType._BYE}:
            self.log.info("Server gone. Stopping ...")
            sys.exit(0)


class Client(
        SplitnnDataInitializer,
        SplitnnClientCommunication,
        ClientServerBaseClient):
    @classmethod
    def conf_parser(cls):
        return ConfParser

    def __init__(self, cfg: Munch):
        super().__init__(cfg)

        # the network to train on
        seed_everything(cfg.federation.seed)
        self.net = create_model(cfg, self.train_dataset)
        self.net.to(self.devices[0])

        self.optimizer = create_optimizer(cfg, self.net.parameters())

        self.global_lr = cfg.training.learning_rate
        self.global_random_seed: int = 0

        self.stage = 'train'  # train | eval | test
        self.representation: torch.Tensor
        self.epochs = 0

    def start_training(self, cfg):
        self.send_client_infos()

        self.optimizer.zero_grad()
        self.optimizer.step()
        while True:
            time_comm_round_start = time.time()
            self.init_comm_round()

            self.forward()

            # Receive parameters from server and update global/local models
            self.update_sync_info()
            self.update_lr()

            if self.stage == 'train':
                self.backward()

            # All testings are perofrmed on the server
            time_comm_round_end = time.time()
            self.datalogger.add_scalar(
                "Time:CommRound", time_comm_round_end - time_comm_round_start,
                self.epochs)
            self.datalogger.flush()

    def init_comm_round(self):
        """Initialize comm. round after receiving synchronization data"""
        self.log.debug(
            "Client %s - Local Epoch: %s ==========",
            self.client_id, self.epochs + 1)
        self.log.debug(
            "Learning rate: %s, seed: %s",
            self.optimizer.param_groups[0]['lr'], self.global_random_seed)

    def update_sync_info(self):
        """Update info in the synchronization message"""
        sync_info = self.receive_sync_info()
        self._update_sync_info(sync_info)
        seed_everything(self.global_random_seed)

    def _update_sync_info(self, sync_info: SyncInfos) -> None:
        self.global_lr = sync_info.data.lr
        self.global_random_seed = sync_info.data.seed

    def update_lr(self):
        """Update learning rate"""
        for group in self.optimizer.param_groups:
            group['lr'] = self.global_lr

    def forward(self):
        """Train first half and sent gradient info to aggregator"""
        self.net.train()
        self.optimizer.zero_grad()
        input_data = None

        def forward_train():
            nonlocal input_data
            self.net.train()
            try:
                input_data, _ = next(self.train_iter)
            except StopIteration:
                # If reaches the end, enters the eval stage
                self.train_iter = iter(self.train_loader)
                self.stage = 'eval'

        def forward_eval():
            nonlocal input_data
            self.net.eval()
            try:
                input_data, _ = next(self.eval_iter)
            except StopIteration:
                self.eval_iter = iter(self.eval_loader)
                self.stage = 'test'

        def forward_test():
            nonlocal input_data
            self.net.eval()
            try:
                input_data, _ = next(self.test_iter)
            except StopIteration:
                self.test_iter = iter(self.test_loader)
                self.stage = 'train'
                self.epochs += 1
                self.log.info(
                    "Client %s - Local Epoch: %s, Learning rate: %s, seed: %s",
                    self.client_id, self.epochs + 1,
                    self.optimizer.param_groups[0]['lr'],
                    self.global_random_seed)

        if self.stage == 'train':
            forward_train()
        if self.stage == 'eval':
            forward_eval()
        if self.stage == 'test':
            forward_test()
            if self.stage == 'train':
                forward_train()
        self.representation = self.net(input_data.to(self.devices[0]))

        data = NoCompress().compress([self.representation])
        self.log.debug("Sending repr (%s bytes)", len(data))
        send_start = time.time()
        self.dispatcher.send_msg(MessageType.MODEL, data)
        send_end = time.time()
        self.datalogger.add_scalar(
            'Time:SendData', send_end - send_start, self.epochs)

    def backward(self):
        """Update the copy of global network"""
        grad = self.receive_grad()
        self.representation.backward(gradient=grad[0].to(self.devices[0]))
        self.optimizer.step()

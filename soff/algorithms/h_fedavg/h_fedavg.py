"""Edge aggregator for the hierarchical fedavg algroithm"""
import os
import time
import random
import threading
from typing import Dict
import torch
from munch import Munch
from eventfd import EventFD
from .. import fedavg
from ..fedavg.server import FedAvgConfParserAdapter, FedAvgServerSchedulerAdapter, FedAvgServerAdapter
from ..fedavg.client import FedAvgClientAdapter
from ..base.base_server import HierarchicalBaseServer, HierarchicalBaseServerConfParser
from ..base.base_client import HierarchicalBaseClient
from ..base.base_transceiver import HierarchicalBaseTransceiver
from ...models import create_loss_criterion, create_model
from ...models.base import PerEpochTrainer
from ...compressors.none import NoCompress
from ...utils.metrics import create_metrics
from ...utils.tensor_buffer import TensorBuffer
from ...utils.training import all_params, init_buffer, seed_everything
from ...communications.protocol import (
    ClientInfo, MessageType, Protocol, SyncInfos)


class HierarchicalFedAvgScheduler(FedAvgServerSchedulerAdapter):
    def __init__(self, cfg: Munch, *args, **kwargs) -> None:
        super(FedAvgServerSchedulerAdapter, self).__init__(
            cfg, *args, **kwargs)

        self.cynical = cfg.fedavg.cynical
        self.num_clients_each_round = self.num_clients

        self.clients_dataset_length: Dict[int, int] = {}
        """Record each clients' dataset lengths"""

        self.sum_data_lengths = 0
        self.clients_dataset_lock = threading.Lock()
        self.event_training_started = threading.Event()
        """This event is fired once all clients' info are received"""

        self.dispatcher.register_msg_event(
            MessageType.CLIENT_INFOS, self.process_client_info)


class ConfParser(FedAvgConfParserAdapter, HierarchicalBaseServerConfParser):
    pass


class Server(FedAvgServerAdapter, HierarchicalBaseServer):
    def __init__(self, cfg, scheduler=HierarchicalFedAvgScheduler):
        super().__init__(cfg, scheduler)

    @classmethod
    def conf_parser(cls):
        return ConfParser


class Client(FedAvgClientAdapter, HierarchicalBaseClient):
    pass


class Transceiver(HierarchicalBaseTransceiver):
    """Hierarchical FedAvg Edge Aggregator (Transceiver)"""

    def __init__(self, cfg, scheduler=HierarchicalFedAvgScheduler):
        super().__init__(cfg, scheduler)

        # Initialize networks #################################################
        # server network, ensure same global model initialization in C/S
        seed_everything(cfg.federation.seed)

        self.net = create_model(cfg, self.eval_dataset)
        self.net.to(self.devices[0])
        self.num_params = len(list(self.net.parameters()))

        # Gradient of `net`, stored in the parameters of a model.
        # Must be zeroed out so that the first epoch is correct.
        self.grads = create_model(cfg, self.eval_dataset)
        init_buffer(self.grads, self.devices[0])

        self.trainer_class = PerEpochTrainer
        self.train_criterion = create_loss_criterion(cfg)
        self.train_criterion.to(self.devices[0])
        self.additional_metrics = create_metrics(cfg)

        self.global_lr = cfg.training.learning_rate

        # Initialize training #################################################
        # a list of client networks to receive
        self.client_nets_buffer = TensorBuffer(cfg.transceiver.num_cache)
        """Total number of samples of all clients"""

        # Then, start then training process
        self.global_model_loss = 10.0
        self.global_random_seed: int

        self.comm_round = 0
        self.old_epoch = 0
        self.iters = 0
        self.epoch = 0
        """equivalent epoch number, calculated by
        (number_samples_processed / total_number_samples)"""
        self.selected = False

        self.clients_comm_rounds: Dict[int, int] = {}
        """Communication rounds for each client"""

        self.selected_client_ids = set()
        """Ids of selected clients in each communication round"""

        self.sem_client_event = threading.Semaphore(0)
        """Semaphore that signals an client event has occurred.
        E.g. a client's data is received and parsed, or clients are gone."""

        self.event_aggregation_done = EventFD()
        """signal the global model is aggregated and ready to be send"""

        self.event_broadcast_done = threading.Event()
        """signals the gradients broadcast is done, and training can proceed"""

        # Record hyperparameters and metrics
        self.datalogger.register_hparams(cfg, [*[
            f"Test({name}):loss"
            for name in self.test_loaders.keys()
        ], *[
            f"Test({name}):{metric.name}"
            for name in self.test_loaders.keys()
            for metric in self.additional_metrics]
        ])

    def start_training(self, cfg) -> None:
        assert isinstance(
            self.server_scheduler, fedavg.server.FedAvgServerSchedulerAdapter)

        super().start_training(cfg)
        self.register_event_handlers()

        # First, wait for all clients to connect
        num_clients = cfg.server_scheduler.num_endpoints
        self.log.info("Waiting for %s clients to connect...", num_clients)
        self.server_scheduler.event_all_clients_connected.wait()

        # Initialize each client's communication rounds
        self.clients_comm_rounds = {
            id: 0 for id in self.server_scheduler.clients_id_socket_map.keys()}

        self.log.info("Waiting for %s clients' metadatas...", num_clients)
        self.server_scheduler.event_training_started.wait()

        # Wait for init broadcast to complete before starting other tasks
        self.server_scheduler.dispatcher.insert_task_barrier()

        # Forward client infos to the server
        self.send_client_infos()

        # Start the main training loop
        while True:
            self.log.info("Waiting for server's instruction ...")

            # Receive sync information (also singal to start) from upstream
            self.update_sync_info()

            # Shortcut if this client is not selected in this round
            if not self.selected:
                self.log.debug("Not updating this round")
                continue

            time_comm_round_start = time.time()
            self.init_comm_round()

            # Receive parameters from server and update global/local models
            self.update_global_params()

            # Broacast the model and wait for broadcast to finish.
            # Before that, avoid changing gradient or selected_client_ids
            self.signal_broadcast_model()
            self.event_broadcast_done.wait()
            self.event_broadcast_done.clear()

            # Aggregate the model and update the global model
            self.aggregate()
            self.update_equivalent_epoch_number()

            # Test the model
            self.test_model()
            self.comm_round += 1

            time_comm_round_end = time.time()
            self.datalogger.add_scalar(
                "Time:CommRound", time_comm_round_end - time_comm_round_start,
                self.epoch)

    def register_event_handlers(self) -> None:
        """Register handlers to handle client events/messages"""
        self.server_scheduler.dispatcher.register_msg_event(
            MessageType.GRADIENT, self.schedule_process_gradient)
        self.server_scheduler.dispatcher.register_fd_event(
            self.event_aggregation_done, self.schedule_broadcast_model)

    def schedule_process_gradient(self, socket, data) -> None:
        """Process gradient sent by clients"""
        def process_gradient(data):
            # skip processing if client already disconnected
            with self.server_scheduler.clients_info_lock:
                if socket in self.server_scheduler.clients_socket_id_map:
                    # put data into slot
                    buf = self.client_nets_buffer.get_slot_for_receive()
                    buf.set_data(NoCompress().decompress(data))
                    buf.client_id = self.server_scheduler.clients_socket_id_map[socket]
                    buf.release_as_ready()
                    self.log.debug("  Gradient of client %s ✔", buf.client_id)
                    self.sem_client_event.release()
        self.server_scheduler.dispatcher.schedule_task(process_gradient, data)

    def schedule_broadcast_model(self, _) -> None:
        """Schedule a task to broadcast sync info and model data"""
        self.log.info("  Broadcasting model and related infos...")
        self.schedule_broadcast_sync_info()
        self.schedule_broadcast_model_data()
        self.server_scheduler.dispatcher.schedule_task(
            self.event_broadcast_done.set)

    def schedule_broadcast_sync_info(self) -> None:
        """Schedule a task to broadcast sync info"""
        sync_info = self.gen_sync_info()

        # Bcast sync info
        self.log.info("  Broadcasting sync infos")
        for cli_id, socket in self.server_scheduler.clients_id_socket_map.items():
            sync_info.data['selected'] = (cli_id in self.selected_client_ids)
            self.server_scheduler.dispatcher.schedule_task(
                Protocol.send_data, socket,
                MessageType.SYNC_INFOS, sync_info.encode(),
                self.datalogger, self.epoch)

        # must wait for syncinfo to finish, before sending gradient
        self.server_scheduler.dispatcher.insert_task_barrier()

    def gen_sync_info(self) -> SyncInfos:
        """Generate syncronization info"""
        return SyncInfos().set_data({
            'lr': self.global_lr,
            'seed': self.global_random_seed})

    def schedule_broadcast_model_data(self) -> None:
        """Schedule a task to broadcast model data"""

        data = NoCompress().compress(all_params(self.net))
        msg_type = MessageType.MODEL

        self.log.info(
            "  Broadcasting model to %s (%s bytes × %s clients)",
            self.selected_client_ids, len(data), len(self.selected_client_ids))

        # broadcasting model data
        for client_id in self.selected_client_ids:
            socket = self.server_scheduler.clients_id_socket_map[client_id]
            self.server_scheduler.dispatcher.schedule_task(
                Protocol.send_data, socket, msg_type, data,
                self.datalogger, self.epoch)

        # fire event once everything is sent
        self.server_scheduler.dispatcher.insert_task_barrier()

    def send_client_infos(self):
        """Send client metadata to the aggregator"""
        assert isinstance(
            self.server_scheduler, fedavg.server.FedAvgServerSchedulerAdapter)
        self.client_dispatcher.send_msg(
            MessageType.CLIENT_INFOS, ClientInfo().set_data({
                'data_len': self.server_scheduler.sum_data_lengths,
            }).encode())

    def update_sync_info(self):
        """Update info in the synchronization message"""
        msg_type, data = self.client_dispatcher.recv_msg()
        self.handle_bye(msg_type)
        assert msg_type == MessageType.SYNC_INFOS, (
            f"msg_type({msg_type}) is not `MessageType.SYNC_INFOS`")

        sync_info = SyncInfos().decode(data)
        self._update_sync_info(sync_info)
        seed_everything(self.global_random_seed)    # sync global seed

    def update_global_params(self):
        """Update the copy of global network"""
        msg_type, data = self.client_dispatcher.recv_msg()
        self.handle_bye(msg_type)
        params = NoCompress().decompress(data)
        with torch.no_grad():
            for g_param, param in zip(all_params(self.net), params):
                g_param.copy_(param.to(g_param.device))

    def _update_sync_info(self, sync_info: SyncInfos) -> None:
        self.global_lr = sync_info.data.lr
        self.global_random_seed = sync_info.data.seed
        self.selected = sync_info.data.selected

    def handle_bye(self, msg_type):
        """Signal all downstreams to exit if upstream exists"""
        if msg_type in {MessageType.BYE, MessageType._BYE}:
            self.log.info("Upstream gone. Stopping ...")
            self.server_scheduler.dispatcher.schedule_broadcast(
                MessageType.BYE, data=b'Bye!',
                datalogger=self.datalogger, epochs=self.epoch)
            self.server_scheduler.dispatcher.insert_task_barrier()
            self.server_scheduler.dispatcher.schedule_task(self._exit)

    def init_comm_round(self) -> None:
        """Initialization actions at the start of every communication round"""
        self.log.info('=' * 30)
        self.log.info("Communication Round %s", self.comm_round + 1)
        seed_everything(self.global_random_seed)

        # Select a faction of clients randomly, we use unifrom sample
        # without replacement, then average by datasize as weight.
        # We use a scoped random generator to avoid polluting the global seed.
        self.selected_client_ids = set(
            random.Random(self.global_random_seed).sample(
                self.server_scheduler.clients_id_socket_map.keys(),
                min(self.server_scheduler.num_clients_each_round,
                    len(self.server_scheduler.clients_id_socket_map.keys()))))

    def signal_broadcast_model(self) -> None:
        """Signal the scheduler to broadcast the model from main thread"""
        # send network
        self.log.info("  Broadcasting network parameters...")
        self.event_aggregation_done.set()

    def aggregate(self) -> None:
        """Aggregate client models"""

        # zero out all params in grads for client aggregation
        for param in all_params(self.grads):
            param.zero_()

        self.log.info("  Waiting for clients input...")
        selected_data_length = sum(
            self.server_scheduler.clients_dataset_length[id]
            for id in self.selected_client_ids)
        clients_aggregated = {id: False for id in self.selected_client_ids}
        for _ in range(len(self.selected_client_ids)):
            # handle client events
            self.sem_client_event.acquire()
            buf = self.client_nets_buffer.get_slot_for_aggregate()

            # report if a client already aggregated by appeared again
            if clients_aggregated[buf.client_id]:
                raise Exception(
                    f"Client {buf.client_id} is not obeying the "
                    f"{self.__class__.__name__} protocol")
            clients_aggregated[buf.client_id] = True

            # aggregation
            assert buf.client_id is not None
            self.clients_comm_rounds[buf.client_id] += 1
            client_weight = (
                self.server_scheduler.clients_dataset_length[buf.client_id] /
                selected_data_length)
            self.log.info(
                "  Aggregating client %s (weight %s)",
                buf.client_id, client_weight)

            # update parameter and buffers gradients
            for grad, client_grad in zip(all_params(self.grads), buf.data):
                # update parameter grad
                grad.set_(
                    ((grad if (grad is not None)
                      else torch.zeros_like(client_grad)) +
                     client_grad.to(self.devices[0]) * client_weight)
                    .type(grad.dtype).clone())

            # release the net to make a vacancy for following receive
            buf.release_as_idle()

        data = NoCompress().compress(all_params(self.grads))
        self.log.info("Sending gradient (%s bytes)", len(data))
        send_start = time.time()
        self.client_dispatcher.send_msg(MessageType.GRADIENT, data)
        send_end = time.time()
        self.datalogger.add_scalar(
            'Time:SendData', send_end - send_start, self.epoch)

        mem_stat = torch.cuda.memory_stats(self.devices[0])
        self.datalogger.add_scalar(
            "CUDA Mem Curr", mem_stat['allocated_bytes.all.current'])
        self.datalogger.add_scalar(
            "CUDA Mem Peak", mem_stat['allocated_bytes.all.peak'])

    def update_equivalent_epoch_number(self) -> None:
        """Calculate and update the equivalent epoch number """
        self.old_epoch = self.epoch
        sum_samples = sum(
            self.clients_comm_rounds[id]
            * self.cfg.fedavg.average_every
            * self.server_scheduler.clients_dataset_length[id]
            for id in self.server_scheduler.clients_id_socket_map.keys())
        self.epoch = sum_samples // self.server_scheduler.sum_data_lengths

    def test_model(self) -> None:
        """Test the global model"""
        if self.comm_round % self.cfg.federation.test_every != 0:
            return

        for name, loader in self.test_loaders.items():
            test_loss, test_results = self.trainer_class.evaluate_model(
                self.net, loader, self.train_criterion,
                self.additional_metrics, self.devices[0])
            self._log_evaluation_result(
                f"Test({name})", test_loss, test_results)

    def _log_evaluation_result(self, pfx, loss, results) -> None:
        self.log.info("  %s loss: %s", pfx, loss)
        for met, res in zip(self.additional_metrics, results):
            self.log.info("  %s %s: %s", pfx, met.name, res)

        self.log.debug("Writing to tensorboard")
        self.datalogger.add_scalar(
            f"{pfx}:loss", loss, self.epoch)
        for met, res in zip(self.additional_metrics, results):
            self.datalogger.add_scalar(
                f"{pfx}:{met.name}", res, self.epoch)

    def _exit(self):
        self.server_scheduler.cleanup()
        os._exit(0)

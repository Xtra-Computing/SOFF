"Asynchronous fedaveraged averaging algroithm"

import time
import threading
from typing import Dict
from munch import Munch
import torch

from ..base.base_server import ClientServerBaseServer
from ..base.base_client import ClientServerBaseClient
from ..base.base_server_scheduler import DynamicBaseServerScheduler
from ..fedavg.server import FedAvgServerAdapter
from ..fedavg.client import FedAvgClientAdapter
from ...utils.tensor_buffer import QueuedTensorBuffer
from ...utils.training import all_params, seed_everything
from ...communications.protocol import ClientInfo, MessageType, Protocol
from ...compressors.none import NoCompress


class AsyncFedavgServerScheduler(DynamicBaseServerScheduler):
    """Scheduler for the FedAvg algorithm"""

    def __init__(self, cfg: Munch, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

        self.clients_dataset_length: Dict[int, int] = {}
        """Record each clients' dataset lengths"""

        self.sum_data_lengths = 0
        self.clients_dataset_lock = threading.Lock()
        self.event_training_started = threading.Event()
        """This event is fired once all clients' info are received"""

        self.dispatcher.register_msg_event(
            MessageType.CLIENT_INFOS, self.process_client_info)

    def process_bye(self, socket, _):
        super().process_bye(socket, _)
        self.log.info(
            "Client %s (fd=%s) is gone",
            self.clients_socket_id_map[socket], socket.fileno())

    def process_client_info(self, socket, data):
        """Process metadata sent by clients"""
        client_info = ClientInfo().decode(data).data
        client_id = self.clients_socket_id_map[socket]
        self.update_client_info(client_id, client_info)

    def update_client_info(self, client_id: int, client_info: Munch):
        with self.clients_dataset_lock:
            self.clients_dataset_length[client_id] = client_info.data_len
            self.log.info(
                "  Client %s dataset length: %s",
                client_id, client_info.data_len)
            self.sum_data_lengths = sum(self.clients_dataset_length.values())


class AsyncFedavgServerAdapter(FedAvgServerAdapter):
    def __init__(self, cfg, scheduler=AsyncFedavgServerScheduler):
        super().__init__(cfg, scheduler)
        self.global_model_lock = threading.Lock()
        self.sem_global_model_update = threading.Semaphore(value=0)
        self.client_nets_buffer = QueuedTensorBuffer(cfg.server.num_cache)

    def register_event_handlers(self) -> None:
        self.scheduler.dispatcher.register_msg_event(
            MessageType.GRADIENT, self.schedule_process_gradient)

    def schedule_process_gradient(self, socket, data) -> None:
        """Process gradient sent by clients"""

        def process_gradient(data):
            # skip processing if client already disconnected
            with self.scheduler.clients_info_lock:
                if socket in self.scheduler.clients_socket_id_map:
                    # put data into slot
                    buf = self.client_nets_buffer.get_slot_for_receive()
                    buf.set_data(NoCompress().decompress(data))
                    buf.client_id = self.scheduler.clients_socket_id_map[socket]
                    buf.release_as_ready()
                    self.log.debug("  Gradient of client %s ✔", buf.client_id)
                    self.sem_client_event.release()
        self.scheduler.dispatcher.schedule_task(process_gradient, data)

        self.scheduler.dispatcher.insert_task_barrier()

        # Send updated model to client
        def send_sync_info():
            self.log.info("  Sending sync info to fd=%s", socket.fileno())
            sync_info = self.gen_sync_info()
            try:
                Protocol.send_data(
                    socket, MessageType.SYNC_INFOS, sync_info.encode(),
                    self.datalogger, self.epoch)
            except InterruptedError:
                self.log.warning("Send to fd=%s failed.", socket.fileno())

        self.scheduler.dispatcher.schedule_task(send_sync_info)

        # must wait for syncinfo to finish, before sending gradient
        self.scheduler.dispatcher.insert_task_barrier()

        def send_model_data():
            self.sem_global_model_update.acquire()
            with self.global_model_lock:
                data = NoCompress().compress(all_params(self.net))
                self.log.info(
                    "  Sending model to fd=%s (%s bytes)",
                    socket.fileno(), len(data))
            try:
                Protocol.send_data(
                    socket, MessageType.MODEL, data, self.datalogger, self.epoch)
            except InterruptedError:
                self.log.warning("Send to fd=%s failed.", socket.fileno())

        # broadcasting model data
        self.scheduler.dispatcher.schedule_task(send_model_data)

        # Wait for everything to finish
        self.scheduler.dispatcher.insert_task_barrier()

    def gen_sync_info(self):
        sync_info = super().gen_sync_info()
        sync_info.data['selected'] = True
        return sync_info

    def start_training(self, cfg) -> None:
        super(FedAvgServerAdapter, self).start_training(cfg)
        self.register_event_handlers()

        # initialize each client's communication rounds
        self.clients_comm_rounds = {
            id: 0 for id in self.scheduler.clients_id_socket_map.keys()}

        total_comm_rounds = self.calc_total_comm_rounds()
        self.log.info("Total communication rounds: %s", total_comm_rounds)
        for comm_round in range(total_comm_rounds):
            if self.early_stop():
                break

            time_comm_round_start = time.time()
            self.init_comm_round(comm_round)

            # Broacast the model and wait for broadcast to finish.
            # Before that, avoid changing gradient or selected_client_ids
            # self.signal_broadcast_model()
            # self.event_broadcast_done.wait()
            # self.event_broadcast_done.clear()

            # Aggregate the model and update the global model
            self.aggregate()
            self.update_global_model()
            self.update_equivalent_epoch_number()

            # Evaluate the model and adjust learning rate
            self.eval_model()
            self.adjust_lr()

            # Test the model
            self.test_model(comm_round)

            self.update_random_seed()

            time_comm_round_end = time.time()
            self.datalogger.add_scalar(
                "Time:CommRound", time_comm_round_end - time_comm_round_start,
                self.epoch)

        self.schedule_broadcast_bye()
        self.event_broadcast_done.wait()
        self.log.info("Training finished. Stopping...")

    def init_comm_round(self, comm_round) -> None:
        self.log.info('=' * 30)
        self.log.info("Communication Round %s", comm_round + 1)
        seed_everything(self.global_random_seed)

    def aggregate(self) -> None:
        # zero out all params in grads for client aggregation
        for param in all_params(self.grads):
            param.zero_()

        self.log.info("  Waiting for clients input...")

        # handle client events
        self.sem_client_event.acquire()
        buf = self.client_nets_buffer.get_slot_for_aggregate()

        # aggregation
        assert buf.client_id is not None
        self.clients_comm_rounds.setdefault(buf.client_id, 0)
        self.clients_comm_rounds[buf.client_id] += 1
        client_weight = min(
            self.scheduler.clients_dataset_length[buf.client_id] /
            max(self.scheduler.sum_data_lengths, 1), 1.0)
        self.log.info(
            "  Aggregating client %s (weight %s)", buf.client_id, client_weight)

        # update parameter and buffers gradients
        for grad, client_grad in zip(all_params(self.grads), buf.data):
            # update parameter grad
            grad.set_(
                ((grad if (grad is not None)
                  else torch.zeros_like(client_grad)) +
                 client_grad.to(self.devices[0]) * client_weight)
                .type(grad.dtype).clone())

        buf.release_as_idle()

    def update_global_model(self) -> None:
        with self.global_model_lock:
            super().update_global_model()
        # release the net to make a vacancy for following receive
        self.log.warning("Global model updated %s",
                         self.sem_global_model_update._value)
        self.sem_global_model_update.release()

    def update_equivalent_epoch_number(self) -> None:
        """Calculate and update the equivalent epoch number """
        self.old_epoch = self.epoch
        sum_samples = sum(
            (self.clients_comm_rounds[id]
             * self.cfg.fedavg.average_every
             * self.scheduler.clients_dataset_length[id]
             if id in self.clients_comm_rounds else 0)
            for id in self.scheduler.clients_id_socket_map.keys())
        self.epoch = sum_samples // self.scheduler.sum_data_lengths


class AsyncFedavgClientAdapter(FedAvgClientAdapter):
    def send_client_infos(self):
        super().send_client_infos()
        # Send an empty gradient to initialize communication
        # In async-fedavg setting, clients initializes each communication round.
        data = NoCompress().compress(list(
            torch.zeros_like(t) for t in all_params(self.gradient)))
        self.log.info("Sending gradient (%s bytes)", len(data))
        self.dispatcher.send_msg(MessageType.GRADIENT, data)


class Server(AsyncFedavgServerAdapter, ClientServerBaseServer):
    pass


class Client(AsyncFedavgClientAdapter, ClientServerBaseClient):
    pass

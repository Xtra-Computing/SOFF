"""Full-feature FedAvg algorithm server"""
import os
import time
import random
import threading
from typing import Dict
from munch import Munch
import torch
from eventfd import EventFD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..base.base_server_scheduler import StaticBaseServerScheduler
from ..base.base_server import (
    ClientServerBaseServer, ClientServerBaseServerConfParser,
    DatasetInitializedServer)
from ...compressors.none import NoCompress
from ...utils.tensor_buffer import TensorBuffer
from ...utils.scheduler import GradualWarmupScheduler, create_scheduler
from ...utils.optimizer import create_optimizer
from ...models import create_loss_criterion, create_model
from ...utils.metrics import create_metrics
from ...communications.protocol import (
    SyncInfos, MessageType, Protocol, ClientInfo)
from ...utils.training import all_params, seed_everything, init_buffer
from ...utils.arg_parser import BaseConfParser
from ...models.base import PerEpochTrainer


class FedAvgServerSchedulerAdapter(StaticBaseServerScheduler):
    """Scheduler for the FedAvg algorithm"""

    def __init__(self, cfg: Munch, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.cynical = True
        self.num_clients_each_round = self.num_clients
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
        if self.event_all_clients_connected.is_set() and self.cynical:
            self.log.info(
                "Client %s (fd=%s) is gone, shutting down",
                self.clients_socket_id_map[socket], socket.fileno())
            self.cleanup()
            os._exit(1)

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

            if len(self.clients_dataset_length.keys()) == self.num_clients:
                self.sum_data_lengths = sum(
                    self.clients_dataset_length.values())
                self.event_training_started.set()


class FedAvgServerScheduler(FedAvgServerSchedulerAdapter):
    def __init__(self, cfg: Munch, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        assert self.num_clients == cfg.client_server.num_clients, (
            "The Number of scheduler endpoints must match the number of clients.")
        self.cynical = cfg.fedavg.cynical
        self.num_clients_each_round = max(round(
            self.num_clients * cfg.fedavg.client_fraction), 1)


class FedAvgConfParserAdapter(BaseConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fedavg_args = self.add_argument_group("FedAvg Configs (S,S->C)")

        fedavg_args.add_argument(
            '-fedavg.cf', '--fedavg.client-fraction',
            default=1.0, type=float, metavar='FRACTION',
            help="Fraction of clients selected each communication round.")
        fedavg_args.add_argument(
            '-fedavg.a', '--fedavg.average-every',
            default=1, type=int, metavar='N',
            help="Average interval (Number of local epochs/iters)")
        fedavg_args.add_argument(
            '-fedavg.eslr', '--fedavg.early-stop-learning-rate',
            default=-1, type=float, metavar='LR',
            help="Stop when learning rate is below this value")
        fedavg_args.add_argument(
            '-fedavg.amo', '--fedavg.advanced-memory-offload',
            action='store_true',
            help="Use advanced memory offloading (reconstruct optimizer "
            "every communication round. Costs more time but saves memory "
            "when using multithreading launchers).")
        fedavg_args.add_argument(
            '-fedavg.cy', '--fedavg.cynical', action='store_true',
            help='Causes the server to exit once any client is disconnected.')


class ConfParser(FedAvgConfParserAdapter, ClientServerBaseServerConfParser):
    pass


class FedAvgServerAdapter(DatasetInitializedServer):
    """Full feature server for the FedAvg algorithm"""
    @classmethod
    def conf_parser(cls):
        return ConfParser

    def __init__(self, cfg, scheduler=FedAvgServerScheduler):
        super().__init__(cfg, scheduler)

        # Initialize server network ###########################################
        # server network, ensure same global model initialization in C/S
        seed_everything(cfg.federation.seed)

        self.net = create_model(cfg, self.eval_dataset)
        self.net.to(self.devices[0])
        self.num_params = len(list(self.net.parameters()))

        # Subtracting gradient from network with a simple SGD stepper
        self.stepper = torch.optim.SGD(self.net.parameters(), lr=1.0)

        # Gradient of `net`, stored in the parameters of a model.
        # Must be zeroed out so that the first epoch is correct.
        self.grads = create_model(cfg, self.eval_dataset)
        init_buffer(self.grads, self.devices[0])

        self.trainer_class = PerEpochTrainer
        self.train_criterion = create_loss_criterion(cfg)
        self.train_criterion.to(self.devices[0])
        self.additional_metrics = create_metrics(cfg)

        # Initialize training #################################################
        # a list of client networks to receive
        self.client_nets_buffer = TensorBuffer(cfg.server.num_cache)
        """Total number of samples of all clients"""

        # Then, start then training process
        self.global_model_loss = 10.0
        self.global_random_seed = random.randint(0, 2 ** 31)

        # Dummy optimizer and scheduler for global lr scheduling
        self.global_optimizer = create_optimizer(cfg, [torch.Tensor()])
        self.lr_scheduler = create_scheduler(cfg, self.global_optimizer)

        self.old_epoch = 0
        self.epoch = 0
        """equivalent epoch number, calculated by
        (number_samples_processed / total_number_samples)"""

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

    def register_event_handlers(self) -> None:
        """Register handlers to handle client events/messages"""
        self.scheduler.dispatcher.register_msg_event(
            MessageType.GRADIENT, self.schedule_process_gradient)
        self.scheduler.dispatcher.register_fd_event(
            self.event_aggregation_done, self.schedule_broadcast_model)

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

    def schedule_broadcast_model(self, _) -> None:
        """Schedule a task to broadcast sync info and model data"""
        self.log.info("  Broadcasting model and related infos...")
        self.schedule_broadcast_sync_info()
        self.schedule_broadcast_model_data()
        self.scheduler.dispatcher.schedule_task(self.event_broadcast_done.set)

    def schedule_broadcast_sync_info(self) -> None:
        """Schedule a task to broadcast sync info"""
        sync_info = self.gen_sync_info()

        # Bcast sync info
        self.log.info("  Broadcasting sync infos")
        for cli_id, socket in self.scheduler.clients_id_socket_map.items():
            sync_info.data['selected'] = (cli_id in self.selected_client_ids)
            self.scheduler.dispatcher.schedule_task(
                Protocol.send_data, socket,
                MessageType.SYNC_INFOS, sync_info.encode(),
                self.datalogger, self.epoch)

        # must wait for syncinfo to finish, before sending gradient
        self.scheduler.dispatcher.insert_task_barrier()

    def gen_sync_info(self) -> SyncInfos:
        """Generate syncronization info"""
        return SyncInfos().set_data({
            'lr': self.global_optimizer.param_groups[0]['lr'],
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
            socket = self.scheduler.clients_id_socket_map[client_id]
            self.scheduler.dispatcher.schedule_task(
                Protocol.send_data, socket, msg_type, data,
                self.datalogger, self.epoch)

        # fire event once everything is sent
        self.scheduler.dispatcher.insert_task_barrier()

    def schedule_broadcast_bye(self) -> None:
        """Schedule a task to broadcast shutdown signals"""
        self.log.info("  Broadcasting shutdown signal")
        self.scheduler.dispatcher.schedule_broadcast(
            MessageType.BYE, data=b'Bye!',
            datalogger=self.datalogger, epochs=self.epoch)
        self.scheduler.dispatcher.insert_task_barrier()
        self.scheduler.dispatcher.schedule_task(self.event_broadcast_done.set)

    def start_training(self, cfg) -> None:
        super().start_training(cfg)
        self.register_event_handlers()

        # First, wait for all clients to connect
        num_clients = self.scheduler.num_clients
        self.log.info("Waiting for %s clients to connect...", num_clients)
        self.scheduler.event_all_clients_connected.wait()

        # initialize each client's communication rounds
        self.clients_comm_rounds = {
            id: 0 for id in self.scheduler.clients_id_socket_map.keys()}

        self.log.info("Waiting for %s clients' metadatas...", num_clients)
        self.scheduler.event_training_started.wait()

        # Wait for init broadcast to complete before starting other tasks
        self.scheduler.dispatcher.insert_task_barrier()

        # Start the main training loop
        total_comm_rounds = self.calc_total_comm_rounds()
        self.log.info("Total communication rounds: %s", total_comm_rounds)
        for comm_round in range(total_comm_rounds):
            if self.early_stop():
                break

            time_comm_round_start = time.time()
            self.init_comm_round(comm_round)

            # Broacast the model and wait for broadcast to finish.
            # Before that, avoid changing gradient or selected_client_ids
            self.signal_broadcast_model()
            self.event_broadcast_done.wait()
            self.event_broadcast_done.clear()

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

    def calc_total_comm_rounds(self) -> int:
        """Compute total communication rounds to execute"""
        return self.cfg.training.epochs

    def early_stop(self):
        """Override this for custom early-stopping criteria"""
        if (isinstance(self.lr_scheduler, ReduceLROnPlateau) and
                any(grp['lr'] < self.cfg.fedavg.early_stop_learning_rate
                    for grp in self.global_optimizer.param_groups)):
            return True
        return False

    def init_comm_round(self, comm_round) -> None:
        """Initialization actions at the start of every communication round"""
        self.log.info('=' * 30)
        self.log.info("Communication Round %s", comm_round + 1)
        seed_everything(self.global_random_seed)

        # Select a faction of clients randomly, we use unifrom sample
        # without replacement, then average by datasize as weight.
        # We use a scoped random generator to avoid polluting the global seed.
        self.selected_client_ids = set(
            random.Random(self.global_random_seed).sample(
                self.scheduler.clients_id_socket_map.keys(),
                min(self.scheduler.num_clients_each_round,
                    len(self.scheduler.clients_id_socket_map.keys()))))

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
            self.scheduler.clients_dataset_length[id]
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
                self.scheduler.clients_dataset_length[buf.client_id] /
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

        mem_stat = torch.cuda.memory_stats(self.devices[0])
        self.datalogger.add_scalar(
            "CUDA Mem Curr", mem_stat['allocated_bytes.all.current'])
        self.datalogger.add_scalar(
            "CUDA Mem Peak", mem_stat['allocated_bytes.all.peak'])

    def update_global_model(self) -> None:
        """Update the global model"""
        grads = all_params(self.grads)

        # Update global model
        self.net.train()
        # Update parameter and buffers. use copy here to avoid referencing
        for param, grad in zip(self.net.parameters(), grads[:self.num_params]):
            param.grad = grad.clone()
        for param, grad in zip(self.net.buffers(), grads[self.num_params:]):
            param.copy_(
                (param if param is not None else torch.zeros_like(grad)) - grad)
        self.stepper.step()
        self.net.eval()

    def update_equivalent_epoch_number(self) -> None:
        """Calculate and update the equivalent epoch number """
        self.old_epoch = self.epoch
        sum_samples = sum(
            self.clients_comm_rounds[id]
            * self.cfg.fedavg.average_every
            * self.scheduler.clients_dataset_length[id]
            for id in self.scheduler.clients_id_socket_map.keys())
        self.epoch = sum_samples // self.scheduler.sum_data_lengths

    def eval_model(self):
        """Evaluate the global model"""
        if self.epoch > self.old_epoch:
            self.global_model_loss, eval_results = (
                self.trainer_class.evaluate_model(
                    self.net, self.eval_loader, self.train_criterion,
                    self.additional_metrics, self.devices[0]))
            self._log_evaluation_result(
                "Eval", self.global_model_loss, eval_results)

    def adjust_lr(self) -> None:
        """Adjust learning rate (only when equivalent epoch changes)"""
        if self.epoch > self.old_epoch:
            if isinstance(self.lr_scheduler, (
                    ReduceLROnPlateau, GradualWarmupScheduler)):
                self.lr_scheduler.step(metrics=self.global_model_loss)
            else:
                self.lr_scheduler.step()

            self.log.info(
                "Equivalent epoch: %s, lr: %s", self.epoch,
                [grp['lr'] for grp in self.global_optimizer.param_groups])

    def test_model(self, comm_round) -> None:
        """Test the global model"""
        if comm_round % self.cfg.federation.test_every != 0:
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

    def update_random_seed(self) -> None:
        """Update global random seed (to be broadcasted to clients)"""
        self.global_random_seed = random.randint(0, 2 ** 31)
        self.log.info("  Updated random seed: %s",  self.global_random_seed)


class Server(FedAvgServerAdapter, ClientServerBaseServer):
    pass

"""SplitNN (Vertical FL) algorithm server"""
import sys
import time
import random
import threading
from collections import OrderedDict
from typing import Dict
from tqdm import tqdm
from munch import Munch
from eventfd import EventFD
from numpy import cumsum
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .utils import SplitnnDataInitializer
from ..base.base_server import (
    ClientServerBaseServer, ClientServerBaseServerConfParser)
from ..base.base_server_scheduler import StaticBaseServerScheduler
from ...models import create_loss_criterion, create_model
from ...utils.metrics import create_metrics
from ...communications.protocol import ClientInfo, MessageType, Protocol, SyncInfos
from ...compressors.none import NoCompress
from ...utils.training import seed_everything
from ...utils.optimizer import create_optimizer
from ...utils.scheduler import GradualWarmupScheduler, create_scheduler
from ...utils.tensor_buffer import TensorBuffer


class SplitNNServerScheduler(StaticBaseServerScheduler):
    def __init__(self, cfg: Munch, datalogger=None) -> None:
        super().__init__(cfg, datalogger)

        self.clients_dataset_lock = threading.Lock()
        self.clients_dataset_length: Dict[int, int] = {}

        self.event_training_started = threading.Event()
        """This event is fired once all clients' info are received"""

        self.dispatcher.register_msg_event(
            MessageType.CLIENT_INFOS, self.process_client_info)

        assert self.num_clients == cfg.client_server.num_clients

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
                self.event_training_started.set()


class ConfParser(ClientServerBaseServerConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        splitnn_args = self.add_argument_group("SplitNN Configs (S,S->C)")
        splitnn_args.add_argument(
            '-splitnn.sd', '--splitnn.server-dataset',
            default=(), type=str, metavar='DATASEt',
            help="Dataset with label")

        splitnn_args.add_argument(
            '-splitnn.eslr', '--splitnn.early-stop-learning-rate',
            default=-1, type=float, metavar='LR',
            help="Stop when learning rate is below this value")


class Server(SplitnnDataInitializer, ClientServerBaseServer):
    def __init__(self, cfg, scheduler=SplitNNServerScheduler):
        super().__init__(cfg, scheduler)

        # Initialize server network ###########################################
        # server network, ensure same global model initialization in C/S
        seed_everything(cfg.federation.seed)

        self.net = create_model(cfg, self.train_dataset)
        self.net.to(self.devices[0])

        self.train_criterion = create_loss_criterion(cfg).to(self.devices[0])
        self.additional_metrics = create_metrics(cfg)
        self.optimizer = create_optimizer(cfg, self.net.parameters())
        self.lr_scheduler = create_scheduler(cfg, self.optimizer)

        self.concated_repr: torch.Tensor
        """concatenated representation, for retrieving gradient"""
        self.concated_repr_grad: torch.Tensor
        self.repr_sizes = []
        """representation sizes, for splitting gradient"""

        # A list of losses in this epoch
        self.losses = []
        self.metrics = [met().to(self.devices[0])
                        for met in self.additional_metrics]
        self.stage = 'train'  # train | eval | test

        # Initialize training #################################################
        self.old_epoch = 0
        self.epoch = 0
        """equivalent epoch number, calculated by
        (number_samples_processed / total_number_samples)"""

        self.client_repr_buffer = TensorBuffer(cfg.server.num_cache)
        """Receive representations of last layers from each client"""

        self.global_random_seed = random.randint(0, 2 ** 31)
        """Random seed for training"""

        # optimizer and scheduler for global lr scheduling
        self.sem_client_event = threading.Semaphore(0)
        """Semaphore that signals an client event has occurred.
        E.g. a client's data is received and parsed, or clients are gone."""

        self.event_aggregation_done = EventFD()
        """signal the global model is aggregated and ready to be send"""

        self.event_broadcast_done = threading.Event()
        """signals the gradients broadcast is done, and training can proceed"""

    def register_event_handlers(self) -> None:
        """Register handlers to handle client events/messages"""
        self.scheduler.dispatcher.register_msg_event(
            MessageType.MODEL, self.schedule_process_representation)
        self.scheduler.dispatcher.register_fd_event(
            self.event_aggregation_done, self.schedule_broadcast_gradients)

    def start_training(self, cfg):
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

        assert all(
            l == len(self.train_dataset) for l in
            self.scheduler.clients_dataset_length.values()), (
            "Datasets must be pre-matched and have the same length, "
            f"got {len(self.train_dataset)}, "
            f"{self.scheduler.clients_dataset_length.values()}")

        # Wait for init broadcast to complete before starting other tasks
        self.scheduler.dispatcher.insert_task_barrier()

        # Start the main training loop
        total_comm_rounds = self.calc_total_comm_rounds()
        self.log.info("Total communication rounds: %s", total_comm_rounds)

        # Progress bar
        for comm_round in tqdm(range(total_comm_rounds), file=sys.stdout):
            if self.early_stop():
                break

            time_comm_round_start = time.time()
            self.init_comm_round(comm_round)

            # Aggregate the model and update the global model
            # Training, validating and testing are all performed in
            #   this function, and is done through state change.
            self.aggregate()

            # The last loop doesn't sync (since clients init the loop)
            if comm_round == total_comm_rounds - 1:
                break

            # Broacast the model and wait for broadcast to finish.
            # Before that, avoid changing network or
            self.signal_broadcast_grads()
            self.event_broadcast_done.wait()
            self.event_broadcast_done.clear()

            time_comm_round_end = time.time()
            self.datalogger.add_scalar(
                "Time:CommRound", time_comm_round_end - time_comm_round_start,
                self.epoch)

        self.schedule_broadcast_bye()
        self.event_broadcast_done.wait()
        self.log.info("Training finished. Stopping...")

    def calc_total_comm_rounds(self) -> int:
        """Compute total communication rounds to execute"""
        return self.cfg.training.epochs * (
            (sum(((len(ds) - 1) // self.cfg.training.batch_size) + 1 for ds in [
                self.train_dataset, self.eval_dataset, self.test_dataset])))

    def early_stop(self):
        """Override this for custom early-stopping criteria"""
        if (isinstance(self.lr_scheduler, ReduceLROnPlateau) and
                any(grp['lr'] < self.cfg.fedavg.early_stop_learning_rate
                    for grp in self.optimizer.param_groups)):
            return True
        return False

    def init_comm_round(self, comm_round) -> None:
        """Initialization actions at the start of every communication round"""
        self.log.debug('=' * 30)
        self.log.debug("Communication Round %s", comm_round + 1)
        seed_everything(self.global_random_seed)

    def signal_broadcast_grads(self) -> None:
        """Signal the scheduler to broadcast the model from main thread"""
        # send network
        self.log.debug("  Broadcasting network parameters...")
        self.event_aggregation_done.set()

    def aggregate(self) -> None:
        """Aggregate client models"""

        # zero out all params in grads for client aggregation
        self.log.debug("  Waiting for clients input...")
        clients_aggregated: OrderedDict[int, torch.Tensor] = OrderedDict(
            (id, torch.Tensor()) for id in
            sorted(self.scheduler.clients_id_socket_map.keys()))

        for _ in range(len(clients_aggregated)):
            # handle client events
            self.sem_client_event.acquire()
            buf = self.client_repr_buffer.get_slot_for_aggregate()

            # report if a client already aggregated by appeared again
            if clients_aggregated[buf.client_id].numel():
                raise Exception(
                    f"Client {buf.client_id} is not obeying the "
                    f"{self.__class__.__name__} protocol")

            # aggregation
            assert buf.client_id is not None
            self.clients_comm_rounds[buf.client_id] += 1
            self.log.debug("  Aggregating client %s", buf.client_id)
            clients_aggregated[buf.client_id] = buf.data[0].clone()

            # release the net to make a vacancy for following receive
            buf.release_as_idle()

        # Continue train/val/test
        self.optimizer.zero_grad()

        # Concat on dim1 (dim0 is batch dimension)
        self.repr_sizes = [
            repr.shape[1] for repr in clients_aggregated.values()]
        self.concated_repr = torch.concat(
            list(clients_aggregated.values()), dim=1).to(self.devices[0])
        self.concated_repr.requires_grad_()

        # and continue forwarding
        output = self.net(self.concated_repr)

        def _update_metrics(loss, output, labels):
            self.losses.append(loss.detach())
            for met in self.metrics:
                met.update(torch.sigmoid(output.detach()), labels.detach())

        def _compute_metrics():
            torch.use_deterministic_algorithms(False)  # BAUROC requires this
            result = [met.compute().cpu().item() for met in self.metrics]
            torch.use_deterministic_algorithms(True)
            return result

        def _reset_metrics():
            self.losses = []
            self.metrics = [
                met().to(self.devices[0])
                for met in self.additional_metrics]

        def aggregate_train():
            self.net.train()
            try:
                _, labels = next(self.train_iter)
                loss = self.train_criterion(output, labels.to(self.devices[0]))
                loss.backward()
                # Store result for sending back to client
                assert self.concated_repr.grad is not None
                self.concated_repr_grad = self.concated_repr.grad.detach().cpu()
                _update_metrics(loss, output, labels)
            except StopIteration:
                # If reaches the end, enters the eval stage
                loss = torch.mean(torch.Tensor(self.losses)).item()
                results = _compute_metrics()
                self._log_evaluation_result("Train", loss, results)
                # Reset stuff
                _reset_metrics()
                self.train_iter = iter(self.train_loader)
                self.stage = 'eval'

        def aggregate_eval():
            self.net.eval()
            try:
                _, labels = next(self.eval_iter)
                loss = self.train_criterion(output, labels.to(self.devices[0]))
                _update_metrics(loss, output, labels)
            except StopIteration:
                # If reaches the end, enters the test stage
                self.adjust_lr()
                if len(self.losses):
                    loss = torch.mean(torch.Tensor(self.losses)).item()
                    results = _compute_metrics()
                    self._log_evaluation_result("Eval", loss, results)
                # Reset stuff
                _reset_metrics()
                self.eval_iter = iter(self.eval_loader)
                self.stage = 'test'

        def aggregate_test():
            self.net.eval()
            try:
                _, labels = next(self.test_iter)
                loss = self.train_criterion(output, labels.to(self.devices[0]))
                _update_metrics(loss, output, labels)
            except StopIteration:
                # If reaches the end, enters the train stage
                self.update_epoch_number()
                if len(self.losses):
                    loss = torch.mean(torch.Tensor(self.losses)).item()
                    results = _compute_metrics()
                    self._log_evaluation_result("Test", loss, results)
                # Reset stuff
                _reset_metrics()
                self.test_iter = iter(self.test_loader)
                self.stage = 'train'

        if self.stage == 'train':
            aggregate_train()
        if self.stage == 'eval':
            aggregate_eval()
        if self.stage == 'test':
            aggregate_test()
            if self.stage == 'train':
                aggregate_train()

        self.optimizer.step()

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

    def update_epoch_number(self) -> None:
        """Update epoch number during training"""
        self.old_epoch = self.epoch
        self.epoch += 1

    def adjust_lr(self) -> None:
        """Adjust learning rate (only when equivalent epoch changes)"""
        if isinstance(self.lr_scheduler, (
                ReduceLROnPlateau, GradualWarmupScheduler)):
            self.lr_scheduler.step(
                metrics=torch.mean(torch.Tensor(self.losses)).item())
        else:
            self.lr_scheduler.step()

        self.log.info(
            "Equivalent epoch: %s, lr: %s", self.epoch,
            [grp['lr'] for grp in self.optimizer.param_groups])

    def schedule_process_representation(self, socket, data):
        """Process representations sent by clients"""
        def process_representation(data):
            # skip processing if client already disconnected
            with self.scheduler.clients_info_lock:
                if socket in self.scheduler.clients_socket_id_map:
                    # put data into slot
                    buf = self.client_repr_buffer.get_slot_for_receive()
                    buf.set_data(NoCompress().decompress(data))
                    buf.client_id = self.scheduler.clients_socket_id_map[socket]
                    buf.release_as_ready()
                    self.log.debug("  Gradient of client %s ✔", buf.client_id)
                    self.sem_client_event.release()
        self.scheduler.dispatcher.schedule_task(process_representation, data)

    def schedule_broadcast_gradients(self, _) -> None:
        """Schedule a task to broadcast sync info and gradient data"""
        self.log.debug("  Broadcasting model and related infos...")
        self.schedule_broadcast_sync_info()
        # Only need backward when training
        if self.stage == 'train':
            self.schedule_broadcast_grads_data()
        self.scheduler.dispatcher.schedule_task(self.event_broadcast_done.set)

    def schedule_broadcast_sync_info(self) -> None:
        """Schedule a task to broadcast sync info"""
        sync_info = self.gen_sync_info()

        # Bcast sync info
        self.log.debug("  Broadcasting sync infos")
        for socket in self.scheduler.clients_id_socket_map.values():
            sync_info.data['selected'] = True
            self.scheduler.dispatcher.schedule_task(
                Protocol.send_data, socket,
                MessageType.SYNC_INFOS, sync_info.encode(),
                self.datalogger, self.epoch)

        # must wait for syncinfo to finish, before sending gradient
        self.scheduler.dispatcher.insert_task_barrier()

    def gen_sync_info(self) -> SyncInfos:
        """Generate syncronization info"""
        return SyncInfos().set_data({
            'lr': self.optimizer.param_groups[0]['lr'],
            'seed': self.global_random_seed})

    def schedule_broadcast_grads_data(self) -> None:
        """Schedule a task to broadcast model data"""
        # broadcasting model data
        shapes = cumsum([0] + self.repr_sizes)
        for i, client_id in enumerate(
                sorted(self.scheduler.clients_id_socket_map.keys())):
            data = NoCompress().compress([
                self.concated_repr_grad[:, shapes[i]:shapes[i+1]].contiguous()])
            msg_type = MessageType.GRADIENT

            self.log.debug(
                "  Broadcasting grad to %s (%s bytes)", client_id, len(data))

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

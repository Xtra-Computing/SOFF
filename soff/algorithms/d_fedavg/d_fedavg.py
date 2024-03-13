"""Transceiver synchronous decentralized fedavg algroithm. """
import os
import ast
import copy
import time
import random
import threading
from typing import Dict
from textwrap import dedent
import torch
from eventfd import EventFD

from ...utils.optimizer import create_optimizer
from .. import fedavg
from ..fedavg.server import FedAvgConfParserAdapter, FedAvgServerSchedulerAdapter
from ..base.base_transceiver import (
    DecentralizedBaseServerScheduler, DecentralizedBaseTransceiver,
    DecentralizedBaseTransceiverConfParser)
from ...models import (
    create_loss_criterion, create_model,
    create_model_trainer, ModelTrainerConfParser)
from ...models.base import PerEpochTrainer
from ...compressors.none import NoCompress
from ...utils.metrics import create_metrics
from ...utils.tensor_buffer import TensorBuffer
from ...utils.training import all_params, init_buffer, seed_everything
from ...utils.arg_parser import Conf
from ...communications.protocol import (
    ClientInfo, MessageType, Protocol, SyncInfos)


class ConfParser(
        FedAvgConfParserAdapter,
        DecentralizedBaseTransceiverConfParser,
        ModelTrainerConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        d_fedavg_args = self.add_argument_group(
            "Decentralized FedAvg Configs")

        d_fedavg_args.add_argument(
            '-davg.ep', '--d-fedavg.execution-plan', required=True, type=str,
            help=dedent("""
            We assume undirected graph topo. Define a buffer (1 model size) a basic operation:
              ((a,b,c), (d,e)):
                  1. clear buffer,
                  2. wait for grad/models from a b c
                  3. accumulate them in buffer (weighted)
                  4. send buffer content to (d, e)
            The content that remains in the buffer after the last operation is defined as the final result.
            The execution plan is a list of such basic operations.

            Each client only need to use a sequence of this ops to construct
            the communication protocol. They don't need to know network topo.

            For 4-clients full-connection all-redeuce:
                cli 1: ((1), (2, 3, 4)), ((1, 2, 3, 4), ())
                cli 2: ((2), (1, 3, 4)), ((1, 2, 3, 4), ())
                cli 3: ((3), (1, 2, 4)), ((1, 2, 3, 4), ())
                cli 4: ((4), (1, 2, 3)), ((1, 2, 3, 4), ())

            For 4-clients ring all-reduce:
                cli 1: ((1), (2)), ((4, 1), (2)), ((4, 1), (2)), ((4, 1), ())
                cli 2: ((2), (3)), ((1, 2), (3)), ((1, 2), (3)), ((1, 2), ())
                cli 3: ((3), (4)), ((2, 3), (4)), ((2, 3), (4)), ((2, 3), ())
                cli 4: ((4), (1)), ((3, 4), (1)), ((3, 4), (1)), ((3, 4), ())

            For 3-clients tree all-reduce (2 as root):
                cli 1: ((1), (2)), ((2), ())
                cli 2: ((1, 2, 3), (1, 3))
                cli 3: ((3), (2)), ((2), ())
            """))

        d_fedavg_args.add_argument(
            '-davg.a', '--d-fedavg.average-every',
            default=1, type=int, metavar='N',
            help="Average interval (Number of local epochs/iters)")
        d_fedavg_args.add_argument(
            '-davg.amo', '--d-fedavg.advanced-memory-offload',
            action='store_true',
            help="Use advanced memory offloading (reconstruct optimizer "
            "every communication round. Costs more time but saves memory "
            "when using multithreading launchers).")


class DecentralizedFedAvgScheduler(
        FedAvgServerSchedulerAdapter, DecentralizedBaseServerScheduler):
    def __init__(self, cfg, *args, **kwargs) -> None:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy['fedavg'] = Conf({'cynical': True})
        super().__init__(cfg_copy, *args, **kwargs)


class Transceiver(DecentralizedBaseTransceiver):
    """Hierarchical FedAvg Edge Aggregator (Transceiver)"""
    @classmethod
    def conf_parser(cls):
        return ConfParser

    def __init__(self, cfg, scheduler=DecentralizedFedAvgScheduler):
        super().__init__(cfg, scheduler)

        # Initialize networks #################################################
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

        # Gradient accumulation buffer
        self.acc_buffer = create_model(cfg, self.eval_dataset)
        init_buffer(self.acc_buffer, self.devices[0])

        self.train_criterion = create_loss_criterion(cfg)
        self.train_criterion.to(self.devices[0])
        self.additional_metrics = create_metrics(cfg)

        self.global_lr = cfg.training.learning_rate

        # Initialize execution plan ###########################################
        self.execution_plan = ast.literal_eval(cfg.d_fedavg.execution_plan)
        assert all((all(
            isinstance(r, int) for r in recvs) and
            isinstance(s, int) for s in sends)
            for recvs, sends in self.execution_plan)
        # The last send list must be empty
        assert self.execution_plan[-1][1] == (
        ), f"last execution plan element '{self.execution_plan[-1][1]}' must be ()"

        # Initialize training #################################################
        # a list of client networks to receive
        self.client_nets_buffer = TensorBuffer(cfg.transceiver.num_cache)
        """Total number of samples of all clients"""

        # Then, start then training process
        self.train_data_length = len(self.train_dataset)
        self.global_model_loss = 10.0
        self.global_random_seed = cfg.federation.seed

        self.comm_round = 0
        self.old_epoch = 0
        self.iters = 0
        self.epoch = 0
        """equivalent epoch number, calculated by
        (number_samples_processed / total_number_samples)"""
        self.selected = False

        self.trainer = create_model_trainer(
            cfg, self.train_loader, self.train_criterion,
            self.additional_metrics, self.datalogger)
        # Currently only support per epoch training
        assert isinstance(self.trainer, PerEpochTrainer)
        self.trainer_class = self.trainer.__class__
        self.optimizer = create_optimizer(cfg, self.net.parameters())
        self.train_criterion = create_loss_criterion(cfg)
        self.additional_metrics = create_metrics(cfg)

        self.clients_comm_rounds: Dict[int, int] = {}
        """Communication rounds for each client"""

        self.recv_client_ids = set()
        """Ids of planned clients to recv from in this communication round"""
        self.send_server_ids = set()
        """Ids of planned servers to send to in this communication round"""
        self.next_recv_client_ids = set()
        """Ids of planned clients to recv from in next communication round"""

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
            id: 0 for id in
            self.server_scheduler.self_claimed_id_to_alloc_id_map.keys()}

        # Send client infos to the server
        self.send_client_infos()

        # Wait for init broadcast to complete before starting other tasks
        self.log.info("Waiting for %s clients' metadatas...", num_clients)
        self.server_scheduler.event_training_started.wait()
        self.server_scheduler.dispatcher.insert_task_barrier()

        # Start the main training loop
        while self.epoch < cfg.training.epochs:
            self.log.info("Waiting for server's instruction ...")

            # Receive sync information (also singal to start) from upstream
            time_comm_round_start = time.time()
            self.init_comm_round()

            # Aggregate the model and update the global model
            self.train_model()

            for (recv_from, send_to), (next_recv_from, _) in zip(
                    self.execution_plan, self.execution_plan[1:] + (((), ()),)):

                self.recv_client_ids = set(recv_from)
                self.send_server_ids = set(send_to)
                self.next_recv_client_ids = set(next_recv_from)

                # Signal ready state by sending sync info to all connected clients
                self.schedule_send_sync_info()

                # Wait for server ready before starting
                self.update_sync_info()
                self.event_broadcast_done.wait()
                self.event_broadcast_done.clear()

                self.aggregate()

            self.update_global_model()

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

    def schedule_process_gradient(self, socket, data) -> None:
        """Process gradient sent by clients"""
        def process_gradient(data):
            # skip processing if client already disconnected
            with self.server_scheduler.clients_info_lock:
                if socket in self.server_scheduler.clients_socket_id_map:
                    # Put data into slot
                    buf = self.client_nets_buffer.get_slot_for_receive()
                    buf.set_data(NoCompress().decompress(data))
                    # Translate allocated id to self-claimed id
                    buf.client_id = (
                        self.server_scheduler.alloc_id_to_self_claimed_id_map[
                            self.server_scheduler.clients_socket_id_map[socket]])
                    buf.release_as_ready()
                    self.log.debug("  Gradient of client %s ✔", buf.client_id)
                    self.sem_client_event.release()
        self.server_scheduler.dispatcher.schedule_task(process_gradient, data)

    def schedule_send_sync_info(self) -> None:
        """Schedule a task to send sync info to receiving clients this round"""
        sync_info = self.gen_sync_info()
        # Bcast sync info
        dst_nodes = set()
        for id, socket in self.server_scheduler.clients_id_socket_map.items():
            claimed_id = self.server_scheduler.alloc_id_to_self_claimed_id_map[id]
            if claimed_id in self.next_recv_client_ids:
                dst_nodes.add(claimed_id)
                sync_info.data['selected'] = True
                self.server_scheduler.dispatcher.schedule_task(
                    Protocol.send_data, socket,
                    MessageType.SYNC_INFOS, sync_info.encode(),
                    self.datalogger, self.epoch)
        self.log.info("  Broadcasting sync infos to %s", dst_nodes)
        # must wait for syncinfo to finish, before sending gradient
        self.server_scheduler.dispatcher.insert_task_barrier()
        self.server_scheduler.dispatcher.schedule_task(
            self.event_broadcast_done.set)

    def gen_sync_info(self) -> SyncInfos:
        """Generate syncronization info"""
        return SyncInfos().set_data({
            'lr': self.global_lr,
            'seed': self.global_random_seed})

    def send_client_infos(self):
        """Send client metadata to the aggregator"""
        assert isinstance(
            self.server_scheduler, fedavg.server.FedAvgServerSchedulerAdapter)
        for dispatcher in self.client_dispatchers.values():
            dispatcher.send_msg(
                MessageType.CLIENT_INFOS, ClientInfo().set_data({
                    'data_len': self.train_data_length}).encode())

    def update_sync_info(self):
        """Receive synchronization message"""
        for server_id in self.send_server_ids:
            msg_type, _ = self.client_dispatchers[server_id].recv_msg()
            self.handle_bye(msg_type)
            assert msg_type == MessageType.SYNC_INFOS

    def update_global_model(self) -> None:
        """Update the global model"""
        for grad, acc in zip(all_params(self.grads), all_params(self.acc_buffer)):
            grad.copy_(acc)
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

        # self-update global random seed
        self.global_random_seed = int(random.random() * (2**32))

        # zero out all params in grads for client aggregation
        for param in all_params(self.grads):
            param.zero_()

    def signal_broadcast_model(self) -> None:
        """Signal the scheduler to broadcast the model from main thread"""
        # send network
        self.log.info("  Broadcasting network parameters...")
        self.event_aggregation_done.set()

    def aggregate(self) -> None:
        """Aggregate client models"""

        self.log.info("  Waiting for clients input...")
        # selected_data_length = sum(
        #     self.server_scheduler.clients_dataset_length[
        #         self.server_scheduler.self_claimed_id_to_alloc_id_map[id]]
        #     if id != self.transceiver_id else self.train_data_length
        #     for id in self.recv_client_ids)J
        # TODO: properly propagate the length of datasets and compute weights there
        total_data_length = sum(
            self.server_scheduler.clients_dataset_length.values())
        clients_aggregated = {id: False for id in self.recv_client_ids}

        # Zero out all params in grads for client aggregation if not adding self in this step
        for buf, param in zip(all_params(self.acc_buffer), all_params(self.grads)):
            buf.copy_(
                param.to(buf.device) * 0.25
                # self.train_data_length / total_data_length
                if self.transceiver_id in self.recv_client_ids
                else torch.zeros_like(buf))

        for _ in range(len(self.recv_client_ids) - (
                1 if self.transceiver_id in self.recv_client_ids else 0)):

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
            # client_weight = (
            #     self.server_scheduler.clients_dataset_length[
            #         self.server_scheduler.self_claimed_id_to_alloc_id_map[
            #             buf.client_id]] / total_data_length)
            client_weight = 0.25
            self.log.info(
                "  Aggregating client %s (weight %s)",
                buf.client_id, client_weight)

            # aggregate gradients into the temporary buffer self.acc_buffer
            for acc, client_grad in zip(all_params(self.acc_buffer), buf.data):
                # update parameter grad
                acc.set_(
                    ((acc if (acc is not None)
                      else torch.zeros_like(client_grad)) +
                     client_grad.to(self.devices[0]) * client_weight)
                    .type(acc.dtype).clone())

            # release the net to make a vacancy for following receive
            buf.release_as_idle()

        data = NoCompress().compress(all_params(self.acc_buffer))
        self.log.info("Sending gradient (%s bytes)", len(data))
        send_start = time.time()
        for cli_id, dispatcher in self.client_dispatchers.items():
            if cli_id in self.send_server_ids:
                dispatcher.send_msg(MessageType.GRADIENT, data)
        send_end = time.time()
        self.datalogger.add_scalar(
            'Time:SendData', send_end - send_start, self.epoch)

        mem_stat = torch.cuda.memory_stats(self.devices[0])
        self.datalogger.add_scalar(
            "CUDA Mem Curr", mem_stat['allocated_bytes.all.current'])
        self.datalogger.add_scalar(
            "CUDA Mem Peak", mem_stat['allocated_bytes.all.peak'])

    def train_model(self):
        """Train the model"""
        time_training_start = time.time()
        self.net.train()
        for _ in range(self.cfg.fedavg.average_every):
            self.update_epoch_number()
            self.iters = self.train_one_round()
        time_training_end = time.time()
        self.log.info("Training finished")
        self.datalogger.add_scalar(
            "Time:Training", time_training_end - time_training_start,
            self.epoch)

    def train_one_round(self):
        """Train the model for one epoch/iter"""
        return self.trainer.train_model(self.net, self.optimizer, self.iters)

    def update_epoch_number(self) -> None:
        """Update epoch number during training"""
        self.epoch += 1

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

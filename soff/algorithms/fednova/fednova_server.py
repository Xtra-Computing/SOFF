"""FedNova algorithm server and corresponding config parser"""
import threading
import torch
from ...models import create_model
from ...utils.training import init_buffer
from .. import fedavg
from ...compressors.none import NoCompress
from ...compressors.compress_utils import Offset, unpack_float32
from ...communications.protocol import MessageType, Protocol


class ServerConfParser(fedavg.ConfParser):
    """simple config parser for the fednova algorithm"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        fednova_args = self.add_argument_group(
            "FedNova-related Arguments (S,S->C)")
        fednova_args.add_argument(
            '-nova.gc', '--fednova.gradient-correction', action='store_true',
            help="Enable gradient correction")


class Server(fedavg.Server):
    """Simple FedNova server"""
    @classmethod
    def conf_parser(cls):
        return ServerConfParser

    def __init__(self, cfg):
        super().__init__(cfg)

        self.gradient_correction = cfg.fednova.gradient_correction
        self.sem_clients_step_weights = threading.Semaphore(0)
        self.clients_step_weights = {}

        # fednova requires sending weight sum of normalized gradients
        self.weight_sum_gradient = create_model(cfg, self.eval_dataset)
        self.weight_sum_gradient.to(self.devices[0])
        init_buffer(self.weight_sum_gradient, self.devices[0])

    def register_event_handlers(self):
        super().register_event_handlers()
        self.scheduler.dispatcher.register_msg_event(
            MessageType.STEP_WEIGHTS, self.process_step_weights)

    def process_step_weights(self, socket, data):
        """Process step weights sent from client"""
        client_id = self.scheduler.clients_socket_id_map[socket]
        self.clients_step_weights[client_id] = unpack_float32(data, Offset())
        self.sem_clients_step_weights.release()

    def schedule_broadcast_model(self, _):
        self.log.info("  Broadcasting model and related infos...")

        self.schedule_broadcast_sync_info()
        self.schedule_broadcast_model_data()
        # fednova also broadcasts weight sum of normalized graients,
        # if gradient_orrection is required
        self.schedule_broadcast_gradient_correction()
        self.scheduler.dispatcher.schedule_task(self.event_broadcast_done.set)

    def schedule_broadcast_gradient_correction(self):
        """Broadcased the averagea and normalized gradient correction"""
        if self.gradient_correction:
            self.scheduler.dispatcher.insert_task_barrier()

            data = NoCompress().compress(list(
                self.weight_sum_gradient.parameters()))
            self.log.info(
                "  Broadcasting global grad to %s (%s bytes × %s clients)",
                self.selected_client_ids,
                len(data), len(self.selected_client_ids))

            for client_id in self.selected_client_ids:
                socket = self.scheduler.clients_id_socket_map[client_id]
                self.scheduler.dispatcher.schedule_task(
                    Protocol.send_data, socket,
                    MessageType.WEIGHTED_SUM_GRADIENT, data,
                    self.datalogger, self.epoch)

        # fire event once everything is sent
        self.scheduler.dispatcher.insert_task_barrier()

    def aggregate(self):
        super().aggregate()

        # copy intermediate result, for broadcasting to client
        self.weight_sum_gradient.load_state_dict(self.grads.state_dict())

        # wait for all clients to send their normalized step weights
        for _ in range(len(self.selected_client_ids)):
            self.sem_clients_step_weights.acquire()

        # coeff = \tau_{eff}
        #       = sum(p_i||a_i^(t)||_1) / (sum_{i\in S_t} p_i)^2
        # see algorithm 1 of the fednova paper
        coeff = 0
        total_client_weight = 0
        selected_data_length = sum(
            self.scheduler.clients_dataset_length[cli_id]
            for cli_id in self.selected_client_ids)

        for cli_id in self.selected_client_ids:
            client_weight = (
                self.scheduler.clients_dataset_length[cli_id]
                / selected_data_length)
            total_client_weight += client_weight
            coeff += client_weight * self.clients_step_weights[cli_id]
        coeff /= (total_client_weight ** 2)

        # multiple to grad, ignoring learning rate, since it's set by
        # the learning rate of optimizer
        with torch.no_grad():
            for grad in self.grads.parameters():
                grad.mul_(coeff)

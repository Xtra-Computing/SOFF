import threading
from oarf.models.models import init_model
from oarf.utils.training import init_buffer
from oarf.algorithms.fedavg import FedAvgServer
from oarf.compressors.compressors import NoCompress
from oarf.compressors.compress_utils import Offset, unpack_float32
from oarf.communications.protocol import MessageType, Protocol


class FedNovaServer(FedAvgServer):
    def __init__(self, num_clients, client_fraction,
                 broadcast_type, gradient_correction, **kwargs):
        super().__init__(
            num_clients=num_clients, client_fraction=client_fraction,
            broadcast_type=broadcast_type,
            gradient_correction=gradient_correction, **kwargs)

        self.gradient_correction = gradient_correction
        self.sem_clients_step_weights = threading.Semaphore(0)
        self.clients_step_weights = {}

        self.dispatcher.register_msg_event(
            MessageType.STEP_WEIGHTS, self.process_step_weights)

    def process_step_weights(self, socket, data):
        client_id = self.clients_socket_id_map[socket]
        self.clients_step_weights[client_id] = unpack_float32(data, Offset())
        self.sem_clients_step_weights.release()

    def schedule_broadcast_model(
            self, _, /, broadcast_type):
        self.log.info("  Broadcasting model and related infos...")

        self.schedule_broadcast_sync_info()
        self.schedule_broadcast_gradient(broadcast_type)
        # fednova also broadcasts weight sum of normalized graients,
        # if gradient_orrection is required
        self.schedule_broadcast_gradient_correction()
        self.dispatcher.schedule_task(self.event_broadcast_done.set)

    def schedule_broadcast_gradient_correction(self):
        if self.gradient_correction:
            self.dispatcher.insert_task_barrier()
            for client_id in self.selected_client_ids:
                socket = self.clients_id_socket_map[client_id]
                self.dispatcher.schedule_task(
                    Protocol.send_data, socket,
                    MessageType.WEIGHTED_SUM_GRADIENT,
                    NoCompress().compress(self.weight_sum_gradient))

        # fire event once everything is sent
        self.dispatcher.insert_task_barrier()

    def init_training(self, model, batchnorm_runstat, *args, **kwargs):
        super().init_training(
            model=model, batchnorm_runstat=batchnorm_runstat, *args, **kwargs)

        # fednova requires sending weight sum of normalized gradients
        self.weight_sum_gradient = init_model(
            model, batchnorm_runstat=batchnorm_runstat,
            dataset=self.eval_dataset)
        self.weight_sum_gradient.cuda()
        init_buffer(self.weight_sum_gradient)

    def aggregate(self, num_clients):
        super().aggregate(num_clients)

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
        selected_data_length = sum([self.clients_dataset_length[id]
                                    for id in self.selected_client_ids])
        for id in self.selected_client_ids:
            client_weight = self.clients_dataset_length[id] / \
                selected_data_length
            total_client_weight += client_weight
            coeff += client_weight * self.clients_step_weights[id]
        coeff /= (total_client_weight ** 2)

        # multiple to grad, ignoring learning rate, since it's set by
        # the learning rate of optimizer
        for grad in self.grads.parameters():
            grad.mul_(coeff)

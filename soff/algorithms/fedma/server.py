"""FedMA algorithm server and corresponding argument parser"""
import math
from typing import Dict, List
import torch
import numpy as np
from lapsolver import solve_dense
from .. import ff_fedavg
from ...utils.tensor_buffer import TensorBuffer
from ...compressors.none import NoCompress
from ...communications.protocol import (
    FedMASyncInfos, MessageType, Protocol, SyncInfos)


class ConfParser(ff_fedavg.ConfParser):
    """Argument parser for Fed(G)MA server"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        fedma_args = self.add_argument_group(
            "FedMA-related Arguments (S,S->C)")
        fedma_args.add_argument(
            '-ma.ga', '--fedma.gamma', default=7.0, type=float, metavar='γ',
            help="FedMA parameter γ")
        fedma_args.add_argument(
            '-ma.s0', '--fedma.sigma0', default=1.0, type=float, metavar='σ₀',
            help="FedMA parameter σ₀")
        fedma_args.add_argument(
            '-ma.si', '--fedma.sigma', default=1.0, type=float, metavar='σ',
            help="FedMA parameter σ")
        fedma_args.add_argument(
            '-ma.mi', '--fedma.match-iters',
            default=5, type=int, metavar='ITERS',
            help="FedMA matching iterations. The Larger this value is, "
            "the more likely that the global model size is smaller")


class Server(ff_fedavg.Server):
    """The FedMA algorithm server"""

    def __init__(self, cfg):
        super().__init__(cfg)

        assert cfg.fedavg.client_fraction == 1, \
            "FedMA only supports full participation"

        # Initialize server network is compatible with fedma
        assert hasattr(self.net, "matchable_layers"), \
            "FedMA requires using FedMA-adapted model"

        # Currently frozen layers
        self.num_frozen_layers = 0

        # assignments cache for the previous layer
        self.prev_assignment = {}

        # assignments and matched weights/bias for this layer
        self.assignment = {}
        self.client_weights = {}

        # this variable is [weight, bias], where [weight] is always
        # 2 dimensional (flattened after the 1st dim)
        self.global_weights_bias: List[torch.Tensor] = []

        # current layer's correct shape after matching, used to restore
        # the shape of self.global_weights_bias
        self.weights_shape: List[int] = []

        # Initialize training ##################################################
        # As we need every participant's layer details in the iteration step
        # of matching (to reduce the size), we need enough amount of cache
        # instead of an arbitrary num_cache
        self.clients_layers_buffer = TensorBuffer(
            self.scheduler.num_clients_each_round)

    def register_event_handlers(self) -> None:
        self.scheduler.dispatcher.register_msg_event(
            MessageType.LAYER_WEIGHTS, self.schedule_process_layer_weights)
        self.scheduler.dispatcher.register_fd_event(
            self.event_aggregation_done, self.schedule_broadcast_layer)

    def schedule_process_layer_weights(self, socket, data):
        def process_weights(data):
            # skip processing if client already disconnected
            with self.scheduler.clients_info_lock:
                if socket in self.scheduler.clients_socket_id_map:
                    # put data into slot
                    buf = self.clients_layers_buffer.get_slot_for_receive()
                    buf.set_data(NoCompress().decompress(data))
                    buf.set_id(self.scheduler.clients_socket_id_map[socket])
                    buf.release_as_ready()

                    self.log.info("  Layer data of client %s ✔", buf.client_id)
                    self.sem_client_event.release()
        self.scheduler.dispatcher.schedule_task(process_weights, data)

    def schedule_broadcast_layer(self, _):
        """Schdule broadcast layer data"""
        self.log.info("  Broadcasting layer and related infos...")
        self.schedule_broadcast_sync_info()
        self.schedule_broadcast_layer_data()
        self.scheduler.dispatcher.schedule_task(self.event_broadcast_done.set)

    def gen_sync_info(self) -> SyncInfos:
        return FedMASyncInfos().set_data({
            'lr': self.global_optimizer.param_groups[0]['lr'],
            'seed': self.global_random_seed,
            'frozen_layers': self.num_frozen_layers})

    def schedule_broadcast_layer_data(self):
        # Get weight and bias to send
        layer_to_send = self.net.matchable_layers[self.num_frozen_layers-1][0][0]
        weight_name = layer_to_send + ".weight"
        bias_name = layer_to_send + ".bias"

        weight = dict(self.net.named_parameters())[weight_name]
        bias = dict(self.net.named_parameters())[bias_name]

        # broadcasting gradient
        msg_type = MessageType.LAYER_WEIGHTS
        num_bytes = 0
        for client_id in self.selected_client_ids:
            # restore client weights/bias from server weights/bias
            if len(self.assignment) > 0:
                # not only do we need to restore according to this layer's
                # order, but also the previous layer's order (if exists)
                if len(self.prev_assignment) == 0:
                    restored_weight = weight.cpu()
                elif weight.shape[1] == len(self.prev_assignment[client_id]):
                    restored_weight = weight.cpu().index_select(
                        1, torch.tensor(self.prev_assignment[client_id]))
                else:
                    prev_assign_len = len(self.prev_assignment[client_id])
                    assert weight.shape[1] % prev_assign_len == 0
                    orig_shape = list(weight.shape).copy()
                    group_size = weight.shape[1] // prev_assign_len
                    restored_weight = weight.cpu(
                    ).reshape(
                        [weight.shape[0], prev_assign_len, group_size] +
                        list(weight.shape)[2:]
                    ).index_select(
                        1, torch.tensor(self.prev_assignment[client_id])
                    ).reshape(orig_shape)

                restored_weight = restored_weight.index_select(
                    0, torch.tensor(
                        self.assignment[client_id], dtype=torch.int64))
                restored_bias = bias.cpu().index_select(
                    0, torch.tensor(
                        self.assignment[client_id], dtype=torch.int64))
            else:
                # before the first matching, we transport the last
                # layer, but the layer order does not need to be restored
                restored_weight = weight
                restored_bias = bias

            # compress weights & bias
            socket = self.scheduler.clients_id_socket_map[client_id]
            data = NoCompress().compress([restored_weight, restored_bias])
            num_bytes += len(data)

            # broadcast layer data
            self.scheduler.dispatcher.schedule_task(
                Protocol.send_data, socket, msg_type, data,
                self.datalogger, self.epoch)

        self.log.info(
            "  Broadcasted layer to %s (total %s bytes)",
            self.selected_client_ids, num_bytes)

        # fire event once everything is sent
        self.scheduler.dispatcher.insert_task_barrier()

    # def early_stop(self):
    #     return (self.global_optimizer.param_groups[0]['lr']
    #             < self.cfg.training.learning_rate * 0.99)

    def init_comm_round(self, comm_round) -> None:
        super().init_comm_round(comm_round)
        self.log.info(
            "Current Layer: %s",
            self.net.matchable_layers[self.num_frozen_layers])

    def aggregate(self) -> None:
        self.log.info("  Waiting for clients input...")
        selected_data_length = sum(
            self.scheduler.clients_dataset_length[id]
            for id in self.selected_client_ids)
        clients_gathered = {
            cid: False for cid in self.selected_client_ids}
        clients_layers: Dict[int, List] = {
            cid: [] for cid in self.selected_client_ids}

        # for the first layer, prev_assignment should be set to empty
        self.prev_assignment = self.assignment \
            if self.num_frozen_layers > 0 else {}

        # clear assignment for this round matching
        # assignment[cli][i] = j means assign the i-th neuron of cli
        # to the j-th global neuron
        self.assignment = {id: [] for id in self.selected_client_ids}

        # Gather data from all clients and store into clients_layers
        for i in range(len(self.selected_client_ids)):
            # handle client events
            self.sem_client_event.acquire()

            # note that idx is the index of `client_nets_buffer` array, not
            # the client id
            buf_layer = self.clients_layers_buffer.get_slot_for_aggregate()

            # update communication rounds and client aggregation weights
            assert buf_layer.client_id is not None
            self.clients_comm_rounds[buf_layer.client_id] += 1
            self.client_weights[buf_layer.client_id] = (
                self.scheduler.clients_dataset_length[buf_layer.client_id] /
                selected_data_length)
            self.log.info(
                "  Gathering client %s (weight %s)", buf_layer.client_id,
                self.client_weights[buf_layer.client_id])

            # report if a client already aggregated by appeared again
            if clients_gathered[buf_layer.client_id]:
                raise Exception(
                    f"Client {buf_layer.client_id} is not obeying the "
                    "{self.__class__.__name__} protocol")

            clients_gathered[buf_layer.client_id] = True

            # Fix the 2nd dim order according to the previous layer's order
            # (Wn <- Wn Π{n-1})
            # NOTE: here, the self.assignment is still last round's assignment
            if self.num_frozen_layers > 0:
                prev_layer_size = dict(self.net.named_parameters())[
                    self.net.matchable_layers[self.num_frozen_layers-1] +
                    ".weight"].shape[0]

                patched_weights = patch_weight(
                    buf_layer.data[0], prev_layer_size,
                    self.prev_assignment[buf_layer.client_id])
            else:
                patched_weights = buf_layer.data[0]
            patched_layer = [patched_weights, buf_layer.data[1]]

            # dump everyting into the `clients_layers` dict
            assert buf_layer.client_id in clients_layers
            clients_layers[buf_layer.client_id] = patched_layer

            # release the net to make a vacancy for following receive
            buf_layer.release_as_idle()

        # For the last layer, we simply aggregate
        if self.num_frozen_layers == len(self.net.matchable_layers) - 1:
            averaged_weight = torch.mean(torch.stack([
                data[0] for data in clients_layers.values()]), 0)
            averaged_bias = torch.mean(torch.stack([
                data[1] for data in clients_layers.values()]), 0)

            self.global_weights_bias = [averaged_weight, averaged_bias]
            self.weights_shape = list(averaged_weight.shape)

            # set necessary stuffs for next round
            self.log.info("Full model matching finished. Resetting...")
            self.num_frozen_layers = 0
            self.assignment = {id: list(range(len(averaged_bias)))
                               for id in self.selected_client_ids}
            return

        # For the rest layers, we first prepare global parameters before
        # initial matching.
        sigma = self.cfg.fedma.sigma
        sigma0 = self.cfg.fedma.sigma0
        gamma = self.cfg.fedma.gamma

        sigma_bias = sigma
        sigma0_bias = sigma0
        mu0 = 0.
        mu0_bias = 0.1

        clients_order = sorted(self.selected_client_ids,
                               key=lambda id: len(clients_layers[id][0]),
                               reverse=True)
        neuron_param_shape = clients_layers[clients_order[0]][0].shape
        self.weights_shape = list(neuron_param_shape)

        # NOTE: these arrays has the shape of [num_params + 1] instead of
        # [1 + num_params], (corrected from original impl)
        sigma_inv_prior = torch.tensor(
            [1 / sigma0] * np.prod(neuron_param_shape[1:]) + [1 / sigma0_bias])
        mean_prior = torch.tensor(
            [mu0] * np.prod(neuron_param_shape[1:]) + [mu0_bias])
        sigma_inv = torch.tensor(
            [1 / sigma] * np.prod(neuron_param_shape[1:]) + [1 / sigma_bias])
        prior_mean_norm = mean_prior * sigma_inv_prior

        # here we initialize from existing global weights, instead of prior
        biggest_client_weight = clients_layers[clients_order[0]][0]
        biggest_client_bias = clients_layers[clients_order[0]][1]
        global_weights = prior_mean_norm + torch.hstack((
            biggest_client_weight.reshape(biggest_client_weight.shape[0], -1),
            biggest_client_bias.reshape(-1, 1)))
        self.assignment[clients_order[0]] = list(range(
            biggest_client_weight.shape[0]))

        # global_sigmas is a matrix which has the shape of
        # [num_global_neurons, 1 + num_params]
        global_sigmas = torch.outer(
            torch.ones(global_weights.shape[0]), sigma_inv)

        popularity_counts = [1] * global_weights.shape[0]

        def prepare_client_params(layer_data):
            # prepare parameters, reshape weights to 2dim
            client_weights_bias = torch.hstack((
                layer_data[0].reshape(layer_data[0].shape[0], -1),
                layer_data[1].reshape(-1, 1)
            ))

            # NOTE: This calculation has problem in the original impl
            # client_weights_bias has the shape of [num_neurons, num_params+1]
            # while sigma_inv has the shape of [1+num_params]. Although it
            # works when sigma_bias = sigma, it doesn't work when it's not.
            client_weights_norm = client_weights_bias * sigma_inv
            return client_weights_norm

        # FedMA initialization matching step (exclude the first client)
        for client_id in clients_order[1:]:
            layer_data = clients_layers[client_id]
            client_weights_norm = prepare_client_params(layer_data)

            # -- before match_layer
            # after match_layer ---

            # Initialization (initial matching)
            global_weights, global_sigmas, popularity_counts, assign_j = matching_update(
                client_weights_norm, global_weights.cpu().detach(),
                sigma_inv, global_sigmas, prior_mean_norm,
                sigma_inv_prior, popularity_counts, gamma,
                self.scheduler.num_clients_each_round)
            self.assignment[client_id] = assign_j

        # FedMA iteration step, try to reduce the number of global neurons
        for _ in range(self.cfg.fedma.match_iters):
            random_order = np.random.permutation(
                list(self.selected_client_ids))
            for client_id in random_order:
                # global neuron indices to delete
                to_delete = []

                client_weights_norm = prepare_client_params(
                    clients_layers[client_id])

                # delete inactive global neurons to shrink its size
                for l_i, i in sorted(enumerate(self.assignment[client_id]),
                                     key=lambda x: -x[1]):
                    popularity_counts[i] -= 1
                    if popularity_counts[i] == 0:
                        del popularity_counts[i]
                        to_delete.append(i)
                        for clean_id in self.selected_client_ids:
                            for idx, l_idx in enumerate(
                                    self.assignment[clean_id]):
                                if i < l_idx and clean_id != client_id:
                                    self.assignment[clean_id][idx] -= 1
                                elif i == l_idx and clean_id != client_id:
                                    self.log.warning("weird unmatching")
                    else:
                        global_weights[i] -= client_weights_norm[l_i]
                        global_sigmas[i] -= sigma_inv

                global_weights = torch.tensor(np.delete(
                    global_weights.numpy(), to_delete, axis=0))
                global_sigmas = torch.tensor(np.delete(
                    global_sigmas.numpy(), to_delete, axis=0))

                # rematch and update popularity_counts
                global_weights, global_sigmas, popularity_counts, assign_j = matching_update(
                    client_weights_norm, global_weights.cpu().detach(),
                    sigma_inv, global_sigmas, prior_mean_norm,
                    sigma_inv_prior, popularity_counts, gamma,
                    self.scheduler.num_clients_each_round)
                self.assignment[client_id] = assign_j

        global_weights /= global_sigmas
        num_neurons = global_weights.shape[0]
        num_params = global_weights.shape[1] - 1
        self.global_weights_bias = [global_weights[:, :num_params],
                                    global_weights[:, num_params]]

        # after matching, we pad the permutation matrix, so it becomes a
        # squar matrix. (from Lj->L to L->L, where Lj is # of local neurons,
        # and L is # of global neurons)
        self.weights_shape[0] = num_neurons
        for client_id in self.selected_client_ids:
            if num_neurons > len(self.assignment[client_id]):
                self.assignment[client_id] += \
                    np.delete(np.arange(num_neurons),
                              self.assignment[client_id]).tolist()
            else:
                assert num_neurons == len(self.assignment[client_id])
                # Already square matrix, do not need to tweak.
                # Plus, global number of neurons cannot be smaller than local,
                # so the only possibility is that they're equal

        # Freeze current layer.
        self.num_frozen_layers += 1

    def update_global_model(self):
        matched_layer = self.net.matchable_layers[self.num_frozen_layers-1]

        # update the layer just matched
        self.net.update_layer(
            matched_layer,
            self.global_weights_bias[0].reshape(
                self.weights_shape).to(self.devices[0]),
            self.global_weights_bias[1].to(self.devices[0]))


def patch_weight(weight: torch.Tensor, prev_layer_size: int,
                 assignment: List[int]):
    """This should be peformed on CPU"""
    shape = list(weight.shape)

    # this can happen when the previous layer is a average pooling layer
    # which is not involved in matching but changes the size of input
    if len(assignment) != shape[1]:
        assert shape[1] % len(assignment) == 0, "second dim of prev layer"\
            " must be integer times of prev layer's output shape"
        grouped_old_shape = [
            shape[0], len(assignment), shape[1] // len(assignment)] + shape[2:]

        grouped_new_shape = grouped_old_shape.copy()
        grouped_new_shape[1] = prev_layer_size
        new_shape = [shape[0], prev_layer_size *
                     (shape[1] // len(assignment))] + shape[2:]

        patched_weights = torch.zeros(grouped_new_shape)
        patched_weights.index_copy_(
            1, torch.tensor(assignment), weight.reshape(grouped_old_shape))
        patched_weights = patched_weights.reshape(new_shape)
    else:
        new_shape = shape.copy()
        new_shape[1] = prev_layer_size
        patched_weights = torch.zeros(new_shape)
        patched_weights.index_copy_(1, torch.tensor(assignment), weight)

    return patched_weights


def matching_update(
        weights, global_weights, sigma_inv, global_sigmas, prior_mean_norm,
        prior_inv_sigma, popularity_counts, gamma, num_clients):

    # number of global neurons
    L = global_weights.shape[0]

    # compute full cost of each pair of global and local neurons
    full_cost = compute_cost(
        global_weights.type(torch.float32), weights.type(torch.float32),
        global_sigmas.type(torch.float32), sigma_inv.type(torch.float32),
        prior_mean_norm.type(torch.float32),
        prior_inv_sigma.type(torch.float32),
        popularity_counts, gamma, num_clients)

    # solve the linear assignment problem using lapsolver
    # row represents local neurons, col represents global neurons
    row_ind, col_ind = solve_dense(-full_cost.numpy())
    assignment = []

    new_L = L
    for local_idx, global_idx in zip(row_ind, col_ind):
        if global_idx < L:
            # existing newron
            popularity_counts[global_idx] += 1
            assignment.append(int(global_idx))
            global_weights[global_idx] += weights[local_idx]
            global_sigmas[global_idx] += sigma_inv
        else:
            # new neuron
            popularity_counts += [1]
            assignment.append(int(new_L))
            new_L += 1
            global_weights = torch.vstack(
                (global_weights, prior_mean_norm + weights[local_idx]))
            global_sigmas = torch.vstack((global_sigmas, sigma_inv))

    return global_weights, global_sigmas, popularity_counts, assignment


def compute_cost(
        global_weights, weights, global_sigmas, sigma_inv, prior_mean_norm,
        prior_inv_sigma, popularity_counts, gamma,
        num_clients):
    """
    compute the cost of assigning each elements in global_weights to weights_j.

    The returning matrix consists of two parts, the left part of the matches
    local layer into exiting global model, while the right parts matches
    the local layer to empty neurons (thus expanding the size of the
    global model)
    """

    Lj = weights.shape[0]
    counts = torch.minimum(
        torch.tensor(popularity_counts, dtype=torch.float32),
        torch.tensor(10))

    sij_p_gs = sigma_inv + global_sigmas
    red_term = (global_weights ** 2 / global_sigmas).sum(axis=1)

    # Nrnparametric cost (matching between global/local neurons)
    param_cost = torch.stack([row_param_cost_simplified(
        global_weights, weights[layer], sij_p_gs, red_term)
        for layer in range(Lj)])

    param_cost += torch.log(counts / (num_clients - counts))

    # Nonparametric cost (limiting the size of global neurons)
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))
    nonparam_cost = torch.outer(((
        (weights + prior_mean_norm) ** 2 /
        (prior_inv_sigma + sigma_inv)).sum(axis=1) -
        (prior_mean_norm ** 2 / prior_inv_sigma).sum()),
        torch.ones(max_added, dtype=torch.float32))
    cost_pois = 2 * torch.log(torch.arange(1, max_added + 1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2 * math.log(gamma / num_clients)

    full_cost = torch.hstack((param_cost, nonparam_cost)).type(torch.float32)
    return full_cost


def row_param_cost_simplified(global_weights, weights_j_l, sij_p_gs, red_term):
    """
    sij_p_gs: sigma_inv_j + global_sigma
    red_term: sum of the normalized square of global weights
    """
    match_norms = ((weights_j_l + global_weights) ** 2
                   / sij_p_gs).sum(axis=1) - red_term
    return match_norms

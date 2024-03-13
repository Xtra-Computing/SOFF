"""FedGMA algorithm server and corresponding argument parser"""

import math
from typing import List, Tuple
import torch
import numpy as np
from lapsolver import solve_dense
from .. import fedma
from ...utils.training import is_batchnorm
from ...compressors.none import NoCompress
from ...communications.protocol import MessageType, Protocol
from ...compressors.compress_utils import (
    pack_raw_data_list, unpack_raw_data_list, Offset)


class ConfParser(fedma.ConfParser):
    """Argument parser for FedGMA server"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        fedgma_args = self.add_argument_group(
            "FedGMA-related Arguments (S,S->C)")
        fedgma_args.add_argument(
            '-gma.ct', '--fedgma.cost-type',
            default='cos', type=str, choices=['cos', 'prob'],
            help="FedMA cost type. cos: Cosine cost, prob: Probablistic cost")


class Server(fedma.Server):
    """FedGMA algorithm server"""
    @classmethod
    def conf_parser(cls):
        return ConfParser

    def __init__(self, cfg):
        super().__init__(cfg)

        if cfg.fedavg.client_fraction != 1:
            raise Exception("FedGMA only supports full participation")

        # Register cost type
        self.cost_type = cfg.fedgma.cost_type

        self.client_ratios = {}

        # Types of some variables are redefined ------------------------------

        # this variable is [weight_1, bias_1, ...], where [weight_i] is always
        # 2 dimensional (flattened after the 1st dim)
        self.global_weights_bias: List[torch.Tensor] = []

        # [(weight_1, bias_1, mean_1, var_1, num_batches_1), ...]
        self.global_bnorm_params: List[Tuple] = []

        # current matching layers' correct shapes after matching, for restoring
        # the shape of self.global_weights_bias
        self.weights_shape: List[List[int]] = []

    def schedule_process_layer_weights(self, socket, data):
        def process_weights(data):
            # skip processing if client already disconnected
            with self.scheduler.clients_info_lock:
                if socket in self.scheduler.clients_socket_id_map:
                    # put data into slot
                    buf = self.clients_layers_buffer.get_slot_for_receive()

                    params, bnorms, orders = \
                        unpack_raw_data_list(data, Offset())

                    # [weight_1, bias_1, order_1, weight_2, ...]
                    param_list = NoCompress().decompress(params)
                    # [weight, bias, optional[mean, var, num_batch], ...]
                    bnorm_list = NoCompress().decompress(bnorms)
                    # [size, order]
                    order_list = NoCompress().decompress(orders)

                    assert (len(param_list) == len(order_list))

                    buf.set_data((param_list, bnorm_list, order_list))

                    # release the slot
                    buf.set_id(self.scheduler.clients_socket_id_map[socket])
                    buf.release_as_ready()

                    # inform main thread
                    self.log.info("  Layer data of client %s ✔", buf.client_id)
                    self.sem_client_event.release()
        self.scheduler.dispatcher.schedule_task(process_weights, data)

    def schedule_broadcast_layer_data(self):
        # get weight and bias to send
        layers_to_send = \
            self.net.matchable_layers[self.num_frozen_layers-1]

        # broadcasting gradient
        msg_type = MessageType.LAYER_WEIGHTS
        num_bytes = 0
        map_size = -1

        for client_id in self.selected_client_ids:
            param_list = []
            bnorm_list = []
            for i, layer in enumerate(filter(
                    lambda l: not is_batchnorm(l[1]), layers_to_send)):
                weight = layer[1].weight
                bias = layer[1].bias
                map_size = layer[1].map_size

                # Restore client weights/bias from server weights/bias
                # The orders of the last layer and batchnorm layers do not
                # need to be restored.
                if len(self.assignment) > 0:
                    # Not only do we need to restore the output order,
                    # but also the input order
                    s, o = self.prev_assignment[client_id][i]
                    restored_weight = unpatch_weight(weight.cpu(), s, o)

                    # Restore output order (if not empty)
                    if restored_weight.numel() > 0:
                        restored_weight = restored_weight.index_select(
                            0, torch.LongTensor(self.assignment[client_id]))
                    if bias is not None and bias.numel() > 0:
                        restored_bias = bias.cpu().index_select(
                            0, torch.LongTensor(self.assignment[client_id]))
                    else:
                        restored_bias = torch.Tensor()
                else:
                    restored_weight = weight
                    restored_bias = bias if bias is not None \
                        else torch.Tensor()
                param_list += [restored_weight, restored_bias]

            for i, (_, layer) in enumerate(filter(
                    lambda l: is_batchnorm(l[1]), layers_to_send)):
                weight = layer.weight
                bias = layer.bias \
                    if layer.bias is not None else torch.Tensor()

                # Restore output order
                restored_weight = weight.cpu().index_select(
                    0, torch.LongTensor(self.assignment[client_id]))
                restored_bias = bias.cpu().index_select(
                    0, torch.LongTensor(self.assignment[client_id])) \
                    if bias is not None else torch.Tensor()

                if layer.track_running_stats:
                    restored_mean = layer.running_mean.cpu().index_select(
                        0, torch.LongTensor(self.assignment[client_id]))
                    restored_var = layer.running_var.cpu().index_select(
                        0, torch.LongTensor(self.assignment[client_id]))
                    num_batches = layer.num_batches_tracked.cpu()
                else:
                    restored_mean = torch.Tensor().to()
                    restored_var = torch.Tensor()
                    num_batches = torch.LongTensor()

                bnorm_list += [
                    restored_weight, restored_bias,
                    restored_mean, restored_var, num_batches]

            if len(self.assignment) > 0:
                assignment = self.assignment[client_id]
            else:
                # initialize the assignment of the 0-th matching
                assignment = list(range(len(
                    self.net.matchable_layers[-1][0][1].weight)))

            # different than the clinet->server message, server only
            # produces one assignment, thus only need to pass assignment
            # to each client once (instead of each layer once).
            order_list = [torch.LongTensor([map_size]),
                          torch.LongTensor(assignment)]

            # compress weights & bias
            socket = self.scheduler.clients_id_socket_map[client_id]
            data = pack_raw_data_list([
                NoCompress().compress(param_list),
                NoCompress().compress(bnorm_list),
                NoCompress().compress(order_list)])
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

    def init_comm_round(self, comm_round) -> None:
        super(fedma.Server, self).init_comm_round(comm_round)
        self.log.info(
            "Current Layers: %s",
            [l[0] for l in self.net.matchable_layers[self.num_frozen_layers]])

    def aggregate(self):
        """
        updated values (results):
        self.weights_shape
        self.global_weights_bias
        self.num_frozen_layers
        self.assignment
        """

        self.log.info("  Waiting for clients input...")

        # clear assignment for this round matching
        # assignment[cli][i] = j means assign the i-th neuron of cli
        # to the j-th global neuron
        self.assignment = {id: [] for id in self.selected_client_ids}

        # Harvest each client's data from communicator
        clients_layers, clients_bnorms = self.gather_clients_data()

        # For the last layer, we simply aggregate.
        # (here we assume only one last-layer, i.e. no multi-task learning)
        if self.num_frozen_layers == len(self.net.matchable_layers) - 1:
            averaged_weight = torch.mean(torch.stack([
                data[0][0] for data in clients_layers.values()]), 0)
            averaged_bias = torch.mean(torch.stack([
                data[0][1] for data in clients_layers.values()]), 0) \
                if self.net.matchable_layers[-1][0][1].bias is not None \
                else None

            self.global_weights_bias = [averaged_weight, averaged_bias]
            self.weights_shape = [list(averaged_weight.shape)]

            # set necessary stuffs for next round
            self.log.info("Full model matching finished. Resetting...")
            self.num_frozen_layers = 0
            self.assignment = {id: list(range(averaged_weight.shape[0]))
                               for id in self.selected_client_ids}
            return

        # For the rest layers, we first prepare global parameters before
        # initial matching.
        sigma = self.cfg.fedma.sigma
        sigma0 = self.cfg.fedma.sigma0
        gamma = self.cfg.fedma.gamma
        mu0 = 0.

        # As each client has the same number of neurons, there's no need to
        # find the biggest one anymore.
        clients_order = sorted(self.selected_client_ids)

        def reshape_client_params(layer_data):
            # prepare parameters, reshape weights to 2dim,and stack all layers
            # required to match horizontally.
            reshaped_layers = []
            for weight, bias in layer_data:
                reshaped_layers.append(weight.reshape(weight.shape[0], -1)
                                       if weight.numel() > 0 else weight)
                reshaped_layers.append(bias.reshape(-1, 1) if bias is not None
                                       else torch.Tensor())
            client_weights_bias = torch.hstack(reshaped_layers)
            return client_weights_bias

        # Initailize from biggest client
        biggest_clients_params = reshape_client_params(
            clients_layers[clients_order[0]])

        # Record metadatas for each layer, for later weight restoration usage
        # > The shape of each layer weight. No need to record bias.
        params_shape = [wb[0].shape for wb in clients_layers[clients_order[0]]]
        self.weights_shape = [list(shape) for shape in params_shape]

        # > The size of param of each layer's weight and bias,
        # after reshaped to 2dim
        reshaped_param_sizes = []
        for weight, bias in clients_layers[clients_order[0]]:
            reshaped_param_sizes.append(
                int(np.prod(weight.shape[1:])) if weight.ndim > 1 else 0)
            reshaped_param_sizes.append(1 if bias is not None else 0)

        sigma_inv_prior = torch.tensor(
            [1 / sigma0] * biggest_clients_params.shape[1])
        mean_prior = torch.tensor(
            [mu0] * biggest_clients_params.shape[1])
        sigma_inv = torch.tensor(
            [1 / sigma] * biggest_clients_params.shape[1])
        prior_mean_norm = mean_prior * sigma_inv_prior

        biggest_clients_params *= sigma_inv
        global_weights = prior_mean_norm + biggest_clients_params

        self.assignment[clients_order[0]] = list(range(
            biggest_clients_params.shape[0]))

        # `global_sigmas` is a matrix which has the shape of
        # [num_global_neurons, num_params]
        global_sigmas = torch.outer(torch.ones(global_weights.shape[0]),
                                    sigma_inv)  # +sigma_inv_prior)

        popularity_counts = [1] * global_weights.shape[0]

        def prepare_client_params(layer_data):
            client_weights_bias = reshape_client_params(layer_data)
            client_weights_norm = client_weights_bias * sigma_inv
            return client_weights_norm

        # FedMA initialization matching step (exclude the first client)
        for client_id in clients_order[1:]:
            layer_data = clients_layers[client_id]
            client_weights_norm = prepare_client_params(layer_data)

            # Initialization (initial matching)
            global_weights, global_sigmas, popularity_counts, assign_j = \
                matching_update(
                    client_weights_norm, global_weights.cpu().detach(),
                    sigma_inv, global_sigmas, prior_mean_norm,
                    sigma_inv_prior, popularity_counts, gamma,
                    self.scheduler.num_clients_each_round, self.cost_type)
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
                global_weights, global_sigmas, popularity_counts, assign_j = \
                    matching_update(
                        client_weights_norm, global_weights.cpu().detach(),
                        sigma_inv, global_sigmas, prior_mean_norm,
                        sigma_inv_prior, popularity_counts, gamma,
                        self.scheduler.num_clients_each_round, self.cost_type)
                self.assignment[client_id] = assign_j

        global_weights /= global_sigmas
        num_neurons = global_weights.shape[0]

        self.log.debug("Num reordered neurons: %s", {
            cli_id: int(torch.sum((
                torch.arange(len(assign)) != torch.LongTensor(assign)).int()))
            for cli_id, assign in self.assignment.items()})

        # Store each layer's matched weight into self.global_weights_bias
        self.global_weights_bias = []
        cur_size = 0
        for size in reshaped_param_sizes:
            self.global_weights_bias.append(
                global_weights[:, cur_size:cur_size + size]
                if size != 0 else torch.Tensor())
            cur_size += size

        # Update the matched layers' shape
        for shape in filter(lambda s: s != [0], self.weights_shape):
            shape[0] = num_neurons

        # Match and update batchnorm layers here
        self.global_bnorm_params = []
        # clients are weighted by their respective number of datas
        client_weights = [self.scheduler.clients_dataset_length[id]
                          for id in self.selected_client_ids]
        for i in range(len(clients_bnorms[clients_order[0]])):
            # param_idx:: 0:weight, 1:bias, 2:mean, 3:var, 4:num_batches
            bnorm_params = [[clients_bnorms[cid][i][param_idx]
                             for cid in self.selected_client_ids]
                            for param_idx in range(5)]
            bnorm_assignments = [self.assignment[cid]
                                 for cid in self.selected_client_ids]
            matched_bnorm_params = match_batchnorm_layers(
                *bnorm_params, bnorm_assignments, client_weights, num_neurons)

            # store the matched params
            self.global_bnorm_params += [matched_bnorm_params]

        # Freeze current layer.
        self.num_frozen_layers += 1

    def gather_clients_data(self):
        """
        return: clients_layers, clients_bnorms
        clients_layers format:
            {id: [(weight, bias), ...], ...}.
        clients_bnorms format:
            {id: [(weight, bias, [mean, var, size]), ...], ...}
        """
        self.prev_assignment = {id: [] for id in self.selected_client_ids}

        clients_gathered = {id: False for id in self.selected_client_ids}
        selected_data_length = sum(
            self.scheduler.clients_dataset_length[id]
            for id in self.selected_client_ids)

        # Stores all the weights and bias to be mached
        clients_layers = {id: [] for id in self.selected_client_ids}
        clients_bnorms = {id: [] for id in self.selected_client_ids}

        # Gather data from all clients and store into clients_layers
        for i in range(len(self.selected_client_ids)):
            # handle client events
            self.sem_client_event.acquire()

            buf_layer = self.clients_layers_buffer.get_slot_for_aggregate()

            # update communication rounds and client aggregation weights
            assert buf_layer.client_id is not None
            self.clients_comm_rounds[buf_layer.client_id] += 1
            self.client_ratios[buf_layer.client_id] = (
                self.scheduler.clients_dataset_length[buf_layer.client_id] /
                selected_data_length)
            self.log.info(
                "  Gathering client %s (weight %s)", buf_layer.client_id,
                self.client_ratios[buf_layer.client_id])

            # report if a client already aggregated by appeared again
            if clients_gathered[buf_layer.client_id]:
                raise Exception(
                    f"Client {buf_layer.client_id,} is not obeying the "
                    "{self.__class__.__name__} protocol")
            clients_gathered[buf_layer.client_id] = True

            # Fix the 2nd dim order according to the previous layer's order
            # (Wn <- Wn Π{n-1})
            assert len(buf_layer.data) == 3
            param_list, bnorm_list, order_list = buf_layer.data

            patched_layer = []
            for i in range(len(param_list) // 2):
                client_weight = param_list[i*2]
                client_bias = (param_list[i*2+1]
                               if param_list[i*2+1].numel() > 0 else None)

                input_size = int(order_list[i*2])
                input_order = order_list[i*2+1].tolist()

                # store input assignment for restoration after matching
                self.prev_assignment[buf_layer.client_id].append(
                    (input_size, input_order))

                # Patch weight and store the result (except input layer)
                if self.num_frozen_layers > 0:
                    patched_weights = patch_weight(
                        client_weight, input_size, input_order)
                else:
                    patched_weights = client_weight
                patched_layer += [(patched_weights, client_bias)]

            # dump everything into the `clients_layers` and `clients_bnorms`
            assert buf_layer.client_id in clients_layers
            clients_layers[buf_layer.client_id] = patched_layer
            for i in range(len(bnorm_list) // 5):
                clients_bnorms[buf_layer.client_id] += \
                    [tuple(bnorm_list[i*5:i*5+5])]

            # release the net to make a vacancy for following receive
            buf_layer.release_as_idle()

        return clients_layers, clients_bnorms

    def update_global_model(self):
        num_neurons = self.global_weights_bias[0].shape[0]

        # Get some metadata about next layers
        next_layers = {}
        for name, _ in self.net.matchable_layers[self.num_frozen_layers-1]:
            for next_name, next_layer in self.net.calc_next_layers(name):
                next_layers[next_name] = next_layer
        next_layer_old_input_size = {}
        for name, layer in next_layers.items():
            with torch.no_grad():
                next_layer_old_input_size[name] = \
                    self.net.calc_input_size(name)

        # update regular layers
        for i, (name, _) in enumerate(filter(
                lambda l: not is_batchnorm(l[1]),
                self.net.matchable_layers[self.num_frozen_layers-1])):

            weight = None
            if self.global_weights_bias[i * 2] is not None:
                weight = self.global_weights_bias[i * 2].reshape(
                    self.weights_shape[i]).to(self.devices[0])

            bias = None
            if self.global_weights_bias[i * 2 + 1] is not None:
                bias = (
                    self.global_weights_bias[i * 2 + 1].reshape(-1)
                    .to(self.devices[0]))

            # update the layer just matched
            self.net.update_layer(name, weight, bias)

            # update mapping if not last layer and not shortcut layer
            self.net.update_mapping(
                name, num_neurons, [v for _, v in self.assignment.items()],
                [1. / len(self.assignment)] * len(self.assignment))

        # update batchnorm layers
        for i, (name, _) in enumerate(filter(
                lambda l: is_batchnorm(l[1]),
                self.net.matchable_layers[self.num_frozen_layers-1])):

            self.net.update_batchnorm_layer(
                name, *self.global_bnorm_params[i])

            self.net.update_mapping(
                name, num_neurons, list(self.assignment.values()),
                [1. / len(self.assignment)] * len(self.assignment))

        # Update next layers (this update is temporary and will be
        # overwritten in the next round. It's only for proper net eval).
        # This is required if the global model's weight is set as the
        # initial matching weight the next round, otherwise it's optional.
        for name, layer in next_layers.items():
            with torch.no_grad():
                patched_weight = pad_or_shrink_weight(
                    layer.weight, next_layer_old_input_size[name],
                    self.net.calc_input_size(name))
            self.net.update_layer(name, patched_weight, (
                layer.bias if layer.bias is not None else torch.Tensor()),
                False)

    def eval_model(self):
        if self.num_frozen_layers == 0:
            super().eval_model()

    def adjust_lr(self):
        if self.num_frozen_layers == 0:
            super().adjust_lr()

    def test_model(self, comm_round):
        if self.num_frozen_layers == 0:
            super().test_model(comm_round)

    def _log_evaluation_result(self, pfx, loss, results):
        self.log.info("  Test loss: %s", loss)
        for met, res in zip(self.additional_metrics, results):
            self.log.info("  Test %s: %s", met.name, res)

        self.log.debug("Writing to tensorboard")
        self.datalogger.add_scalar(
            f"{pfx}:loss", loss, self.epoch)
        for met, res in zip(self.additional_metrics, results):
            self.datalogger.add_scalar(
                f"{pfx}:{met.name}", res, self.epoch)


def pad_or_shrink_weight(
        weight: torch.Tensor, old_input_size: int, target_input_size: int):
    """
    Force the input size of `weight` to be `target_input_size`. Pad zeros
    if necessary.
    """

    if weight.ndim < 2 or target_input_size == weight.shape[1]:
        return weight

    # Consider input grouping. Otherwise VGG won't work
    group_size = 1
    if old_input_size != weight.shape[1]:
        assert weight.shape[1] % old_input_size == 0
        group_size = int(weight.shape[1] / old_input_size)

    new_shape = list(weight.shape)
    new_shape[1] = target_input_size * group_size
    new_weight = weight.detach().clone().resize_(new_shape)
    if target_input_size * group_size > weight.shape[1]:
        new_weight[:, target_input_size * group_size:] = 0
    return new_weight


def patch_weight(weight: torch.Tensor, input_size: int, assignment: List[int]):
    """
    This should be peformed on CPU
    input_size: input size after matching
    """
    # no need to patch 1-dim and 0-dim  weight
    if weight.ndim < 2:
        return weight

    shape = list(weight.shape)

    # this can happen when the previous layer is a average pooling layer
    # which is not involved in matching but changes the size of input
    if len(assignment) != shape[1]:
        assert shape[1] % len(assignment) == 0, "second dim of prev layer"\
            " must be integer times of prev layer's output shape"
        grouped_old_shape = [
            shape[0], len(assignment), shape[1] // len(assignment)] + shape[2:]

        grouped_new_shape = grouped_old_shape.copy()
        grouped_new_shape[1] = input_size
        new_shape = [shape[0], input_size *
                     (shape[1] // len(assignment))] + shape[2:]

        patched_weights = torch.zeros(grouped_new_shape).to(weight.device)
        patched_weights.index_copy_(
            1, torch.tensor(assignment).to(weight.device),
            weight.reshape(grouped_old_shape))
        patched_weights = patched_weights.reshape(new_shape)
    else:
        new_shape = shape.copy()
        new_shape[1] = input_size
        patched_weights = torch.zeros(new_shape).to(weight.device)
        patched_weights.index_copy_(
            1, torch.tensor(assignment).to(weight.device), weight)

    return patched_weights


def unpatch_weight(
        weight: torch.Tensor, mapped_size: int, input_assignment: List[int]):
    """
    Restore the input order of weight before sending them to clients.
    """
    # no need to restore 1-dim and 0-dim  weight
    if weight.ndim < 2:
        return weight

    if weight.shape[1] == mapped_size:
        restored_weight = weight.cpu().index_select(
            1, torch.tensor(input_assignment))
    else:
        # here weight has mapped_size groups, since the mapped_size is
        # essentially previous layer's # of output neurons
        assert weight.shape[1] % mapped_size == 0
        group_size = weight.shape[1] // mapped_size
        orig_shape = list(weight.shape)
        orig_shape[1] = group_size * len(input_assignment)

        restored_weight = weight.cpu().reshape(
            [weight.shape[0], mapped_size, group_size] + list(weight.shape)[2:]
        ).index_select(1, torch.tensor(input_assignment)).reshape(orig_shape)
    return restored_weight


def matching_update(
        client_weights_norm, global_weights, sigma_inv, global_sigmas,
        prior_mean_norm, sigma_inv_prior, popularity_counts, gamma,
        num_clients, cost_type):

    # number of global neurons
    num_global_neurons = global_weights.shape[0]

    if cost_type == 'cos':
        compute_cost = compute_cost_cos
    elif cost_type == 'prob':
        compute_cost = compute_cost_prob
    else:
        raise NotImplementedError

    # compute full cost of each pair of global and local neurons
    full_cost = compute_cost(
        global_weights.type(
            torch.float32), client_weights_norm.type(torch.float32),
        global_sigmas.type(torch.float32), sigma_inv.type(torch.float32),
        prior_mean_norm.type(torch.float32),
        sigma_inv_prior.type(torch.float32),
        popularity_counts, gamma, num_clients)

    # solve the linear assignment problem using lapsolver
    # row represents local neurons, col represents global neurons
    row_ind, col_ind = solve_dense(-full_cost.numpy())
    assignment = []

    new_l = num_global_neurons
    for local_idx, global_idx in zip(row_ind, col_ind):
        if global_idx < num_global_neurons:
            # existing newron
            popularity_counts[global_idx] += 1
            assignment.append(int(global_idx))
            global_weights[global_idx] += client_weights_norm[local_idx]
            global_sigmas[global_idx] += sigma_inv
        else:
            # new neuron
            popularity_counts += [1]
            assignment.append(int(new_l))
            new_l += 1
            global_weights = torch.vstack(
                (global_weights, prior_mean_norm + client_weights_norm[local_idx]))
            global_sigmas = torch.vstack((global_sigmas, sigma_inv))

    return global_weights, global_sigmas, popularity_counts, assignment


def compute_cost_cos(global_weights, client_weights_norm, *_, **__):
    l_j = client_weights_norm.shape[0]
    max_add = 0.8 * l_j
    param_cost = torch.stack([
        torch.nn.CosineSimilarity()(
            client_weights_norm[layer].reshape(1, -1), global_weights)
        for layer in range(l_j)])
    nonparam_cost = torch.outer(
        # local:
        torch.zeros(param_cost.shape[0]),
        # global:
        torch.arange(0, max_add))

    full_cost = torch.hstack((param_cost, nonparam_cost)).type(torch.float32)
    return full_cost


def compute_cost_prob(
        global_weights, client_weights_norm, global_sigmas, sigma_inv,
        prior_mean_norm, sigma_inv_prior, popularity_counts, gamma,
        num_clients):
    """
    compute the cost of assigning each elements in global_weights to weights_j.

    The returning matrix consists of two parts, the left part of the matches
    local layer into exiting global model, while the right parts matches
    the local layer to empty neurons (thus expanding the size of the
    global model)
    """

    l_j = client_weights_norm.shape[0]
    counts = torch.minimum(
        torch.tensor(popularity_counts, dtype=torch.float32),
        torch.tensor(10))

    # parametric cost (matching between global/local neurons)
    param_cost = torch.stack([
        (((client_weights_norm[layer] + global_weights) ** 2
            / (sigma_inv + global_sigmas)).sum(axis=1)
            - (global_weights ** 2 / global_sigmas).sum(axis=1))
        for layer in range(l_j)])

    param_cost += torch.log(counts / (num_clients - counts))

    # Nonparametric cost (limiting the size of global neurons)
    num_neurons = global_weights.shape[0]
    # first argument to max() is the upper limit of added neurons
    max_added = min(l_j, max(int(num_neurons), 1))
    nonparam_cost = torch.outer(((
        (client_weights_norm + prior_mean_norm) ** 2 /
        (sigma_inv_prior + sigma_inv)).sum(axis=1) -
        (prior_mean_norm ** 2 / sigma_inv_prior).sum()),
        torch.ones(max_added, dtype=torch.float32))
    cost_pois = 2 * torch.log(torch.arange(1, max_added + 1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2 * math.log(gamma / num_clients)

    full_cost = torch.hstack((param_cost, nonparam_cost)).type(torch.float32)
    return full_cost


def match_batchnorm_layers(
        weights: List[torch.Tensor], biases: List[torch.Tensor],
        means: List[torch.Tensor], variances: List[torch.Tensor],
        num_batches: List[torch.Tensor], assignments: List[int],
        client_weights: List[int], map_size: int):
    """
    return: weight, bias, mean, variance, num_batch
    """
    assert len(assignments) == len(weights)
    assert len(assignments) == len(biases)
    assert len(assignments) == len(means)
    assert len(assignments) == len(variances)
    assert len(assignments) == len(num_batches)
    assert len(assignments) == len(client_weights)

    weight = torch.zeros(map_size)
    bias = torch.zeros(map_size)
    mean = torch.zeros(map_size)
    variance = torch.zeros(map_size)

    if len(means[0]) > 0:
        num_batches = [int(n) for n in num_batches]
        num_batch = sum(num_batches)

        popularity_counts = torch.zeros_like(mean)
        for m, v, n, a in zip(means, variances, num_batches, assignments):
            matched_mean = torch.zeros_like(mean)
            matched_mean.index_copy_(0, torch.LongTensor(a), m)
            mean += (matched_mean * n)

            matched_variance = torch.zeros_like(variance)
            matched_variance.index_copy_(0, torch.LongTensor(a), v)
            variance += (matched_variance * n)

            matched_p_counts = torch.zeros_like(mean)
            matched_p_counts.index_copy_(
                0, torch.LongTensor(a), torch.ones_like(m) * n)
            popularity_counts += matched_p_counts
        mean /= popularity_counts
        variance /= popularity_counts
    else:
        # proportional to each client's amount of data
        num_batches = client_weights
        num_batch = sum(num_batches)

    popularity_counts = torch.zeros_like(weight)
    for w, b, n, a in zip(weights, biases, num_batches, assignments):
        matched_weight = torch.zeros_like(weight)
        matched_weight.index_copy_(0, torch.LongTensor(a), w)
        weight += matched_weight * n

        matched_bias = torch.zeros_like(bias)
        matched_bias.index_copy_(0, torch.LongTensor(a), b)
        bias += matched_bias * n

        matched_p_counts = torch.zeros_like(mean)
        matched_p_counts.index_copy_(
            0, torch.LongTensor(a), torch.ones_like(w) * n)
        popularity_counts += matched_p_counts

    weight /= popularity_counts
    bias /= popularity_counts

    return weight, bias, mean, variance, num_batch

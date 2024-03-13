import torch
from .. import fedgma
from ...utils.training import is_batchnorm


class Server(fedgma.Server):
    """
    This class is not a proper per-layer fedavg impl, its just used
    as a baseline for FedGMA.
    """

    def aggregate(self, *_, **__):
        # Harvest each client's data from communicator
        clients_layers, clients_bnorms = self.gather_clients_data()

        # Update weights shape
        params_shape = [wb[0].shape for wb in clients_layers[0]]
        self.weights_shape = [list(shape) for shape in params_shape]

        # Update global weights/bias and assignment
        self.global_weights_bias = []
        self.assignment = None
        for i, (_, layer) in enumerate(filter(
                lambda l: not is_batchnorm(l[1]),
                self.net.matchable_layers[self.num_frozen_layers])):

            averaged_weight = torch.mean(torch.stack([
                data[i][0] for data in clients_layers.values()]), 0)
            self.global_weights_bias.append(averaged_weight)

            if layer.bias is not None:
                averaged_bias = torch.mean(torch.stack([
                    data[i][1] for data in clients_layers.values()]), 0)
            else:
                averaged_bias = torch.Tensor()
            self.global_weights_bias.append(averaged_bias)

            if averaged_weight.shape[0] > 0 and self.assignment is None:
                self.assignment = {
                    id: list(range(averaged_weight.shape[0]))
                    for id in self.selected_client_ids}

        # Update batchnorm
        self.global_bnorm_params = []

        for i, (_, layer) in enumerate(filter(
                lambda l: is_batchnorm(l[1]),
                self.net.matchable_layers[self.num_frozen_layers])):

            # param_idx:: 0:weight, 1:bias, 2:mean, 3:var, 4:num_batches
            bnorm_params = [[clients_bnorms[cid][i][param_idx]
                             for cid in self.selected_client_ids]
                            for param_idx in range(5)]
            weights = bnorm_params[0]
            biases = bnorm_params[1]

            weight = torch.mean(torch.stack(weights), 0)
            bias = torch.mean(torch.stack(biases), 0)

            mean = layer.running_mean
            var = layer.running_var
            num = layer.num_batches_tracked

            # store the matched params
            self.global_bnorm_params += [(weight, bias, mean, var, num)]

        # Update other metadatas
        self.num_frozen_layers = \
            (self.num_frozen_layers + 1) % len(self.net.matchable_layers)


class Client(fedgma.Client):
    pass

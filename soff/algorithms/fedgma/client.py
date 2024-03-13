"""FedGMA algorithm client"""
import torch
from .. import fedma
from ...models.unordered_net import _UnorderedNet
from ...utils.optimizer import create_optimizer
from ...compressors.none import NoCompress
from ...utils.training import is_batchnorm
from ...communications.protocol import MessageType
from ...compressors.compress_utils import (
    pack_raw_data_list, unpack_raw_data_list, Offset)


class Client(fedma.Client):
    """The FedGMA algorithm client"""

    def __init__(self, cfg):
        super().__init__(cfg)

        # check whether this is a model adapted for fedma
        assert isinstance(self.net, _UnorderedNet), \
            "FedGMA requires using FedGMA-adapted model"

    def _log_evaluation_result(self, pfx, loss, results):
        self.log.info("  Test loss: %s", loss)
        for met, res in zip(self.additional_metrics, results):
            self.log.info("  Test %s: %s", met.name, res)

        self.log.debug("Writing to tensorboard")
        self.datalogger.add_scalar(f"{pfx}:loss", loss, self.epochs)
        for met, res in zip(self.additional_metrics, results):
            self.datalogger.add_scalar(f"{pfx}:{met.name}", res, self.epochs)

    def update_local_params(self):
        msg_type, data = self.dispatcher.recv_msg()

        self.handle_bye(msg_type)
        assert msg_type == MessageType.LAYER_WEIGHTS

        param_list, bnorm_list, order_list = \
            unpack_raw_data_list(data, Offset())
        param_list = NoCompress().decompress(param_list)
        bnorm_list = NoCompress().decompress(bnorm_list)
        order_list = NoCompress().decompress(order_list)
        assert len(order_list) == 2

        # update this layer, and pad next layer with 0's
        layers_to_update = self.net.matchable_layers[self.num_frozen_layers - 1]

        with torch.no_grad():
            for i, (name, _) in enumerate(filter(
                    lambda l: not is_batchnorm(l[1]), layers_to_update)):
                self.net.update_layer(
                    name, param_list[2*i].to(self.devices[0]),
                    param_list[2*i+1].to(self.devices[0]))

                self.net.update_mapping(
                    name, int(order_list[0]), [list(order_list[1])], [1.])

            for i, (name, layer) in enumerate(filter(
                    lambda l: is_batchnorm(l[1]), layers_to_update)):
                if layer.track_running_stats:
                    # keep runstat the same
                    self.net.update_batchnorm_layer(
                        name, bnorm_list[0], bnorm_list[1],
                        layer.running_mean, layer.running_var,
                        int(layer.num_batches_tracked))
                else:
                    self.net.update_batchnorm_layer(
                        name, *bnorm_list[i*5:i*5+5])

                self.net.update_mapping(
                    name, int(order_list[0]), [list(order_list[1])], [1.])

        # important. after we updated layers, optimizer no longer holds the
        # corresponding parameters of that layer, so we need to change the
        # optimizer accordingly

        # Re-construct the optimizer. LR is unimportant, will be updated later.
        del self.optimizer
        self.optimizer = create_optimizer(
            self.cfg, self.net.parameters(), lr=99999)

        # Danger: directly changing optimizer.param_groups causes memory leak
        # self.optimizer.param_groups = []
        # self.optimizer.add_param_group(
        #     {'params': list(self.net.parameters())})

        # Learnig rate update must be placed after optimizer update
        super(fedma.Client, self).update_lr()

        # freeze corresponding layers
        for i, group in enumerate(self.net.matchable_layers):
            for name, layer in group:
                require_grad = (i >= self.num_frozen_layers)
                if layer.weight is not None:
                    layer.weight.requires_grad_(require_grad)
                if layer.bias is not None:
                    layer.bias.requires_grad_(require_grad)

    def aggregate(self):
        layers_to_send = self.net.matchable_layers[self.num_frozen_layers]

        # pack layer weights, bias and orders
        param_list = []
        bnorm_list = []
        order_list = []

        for name, layer in layers_to_send:
            # First, compress and pack weights and bias (if available)
            if not is_batchnorm(layer):
                param_list += [
                    layer.weight,
                    (layer.bias if layer.bias is not None
                     else torch.Tensor().to(layer.weight.device))]

                # Then, pack order without compression
                # (batchnorm layers do not need input orders)
                input_size = self.net.calc_input_size(name)
                input_order = self.net.calc_input_order(name)
                order_list += [torch.LongTensor([input_size]),
                               torch.LongTensor(input_order)]
            else:
                bnorm_list += [
                    layer.weight, layer.bias,
                    torch.Tensor().to(layer.weight.device),
                    torch.Tensor().to(layer.weight.device),
                    torch.LongTensor().to(layer.weight.device)]

        # Last, pack everything into a single package
        data = pack_raw_data_list([
            NoCompress().compress(param_list),
            NoCompress().compress(bnorm_list),
            NoCompress().compress(order_list)])

        # send layer weights to server
        self.log.info("Sending layer weights (%s bytes)", len(data))
        self.dispatcher.send_msg(MessageType.LAYER_WEIGHTS, data)

    def init_comm_round(self) -> None:
        super(fedma.Client, self).init_comm_round()
        self.log.info(
            "Current Layers: %s",
            [l[0] for l in self.net.matchable_layers[self.num_frozen_layers]])

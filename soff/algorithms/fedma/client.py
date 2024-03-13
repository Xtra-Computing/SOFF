"""FedMA algorithm client"""
import torch
from .. import ff_fedavg, fedavg
from ...utils.optimizer import create_optimizer
from ...compressors.none import NoCompress
from ...utils.training import seed_everything
from ...communications.protocol import FedMASyncInfos, MessageType

class Client(ff_fedavg.Client):
    """The FedMA algorithm client"""

    def __init__(self, cfg):
        super().__init__(cfg)

        # check whether this is a model adapted for fedma
        assert hasattr(self.net, "matchable_layers"), \
            "FedMA requires using FedMA-adapted model"

        # remove unused attributes
        del self.gradient
        del self.net_global

        # FedMA: a number currently frozen layers
        self.num_frozen_layers = 0

    def unload_resources(self):
        self.net.cpu()
        super(fedavg.Client, self).unload_resources()

    def load_resources(self):
        super(fedavg.Client, self).load_resources()
        self.net.to(self.devices[0])

    def update_sync_info(self):
        msg_type, data = self.dispatcher.recv_msg()
        self.handle_bye(msg_type)
        assert msg_type == MessageType.SYNC_INFOS

        sync_info = FedMASyncInfos().decode(data)
        self.global_lr = sync_info.data.lr
        self.global_random_seed = sync_info.data.seed
        self.selected = sync_info.data.selected
        self.num_frozen_layers = sync_info.data.frozen_layers
        seed_everything(self.global_random_seed)    # sync global seed

    def update_lr(self):
        # Learning rate updated is placed the end of `update_local_params`,
        # after the optimizer is being replaced
        pass

    def update_global_params(self):
        # There's no global params to updated in FedMA
        pass

    def update_local_params(self):
        msg_type, data = self.dispatcher.recv_msg()

        self.handle_bye(msg_type)
        assert msg_type == MessageType.LAYER_WEIGHTS

        params = NoCompress().decompress(data)
        params = [params[0].to(self.devices[0]), params[1].to(self.devices[0])]

        # update this layer, and pad next layer with 0's
        layer_to_update = self.net.matchable_layers[
            self.num_frozen_layers - 1]

        with torch.no_grad():
            self.net.update_layer(
                layer_to_update, params[0], params[1])

        # Important. after we updated layers, optimizer no longer holds the
        # corresponding parameters of that layer, so we need to change the
        # optimizer accordingly

        # Re-construct the optimizer. LR is unimportant, will be updated later.
        del self.optimizer
        self.optimizer = create_optimizer(
            self.cfg, self.net.parameters(), lr=99999)

        # Learnig rate update must be placed after optimizer update
        super().update_lr()

        # freeze corresponding layers
        self.net.freeze_layers(set(
            self.net.matchable_layers[:self.num_frozen_layers]))
        self.net.unfreeze_layers(set(
            self.net.matchable_layers[self.num_frozen_layers:]))


    def calc_gradient(self):
        # There's no gradient to calcualte in FedMA
        pass

    def aggregate(self):
        # update error for error feedback in the next rond
        layer_to_send = self.net.matchable_layers[self.num_frozen_layers]

        data = NoCompress().compress([
            dict(self.net.named_parameters())[layer_to_send + ".weight"],
            dict(self.net.named_parameters())[layer_to_send + ".bias"]])
        self.log.info("Sending layer weights (%s bytes)", len(data))
        self.dispatcher.send_msg(MessageType.LAYER_WEIGHTS, data)

    def init_comm_round(self):
        super().init_comm_round()
        self.log.info(
            "Current Layer: %s",
            self.net.matchable_layers[self.num_frozen_layers])

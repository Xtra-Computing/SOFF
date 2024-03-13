"""Learned-gradient compression client."""
import torch
from .. import ff_fedavg
from .utils import LGCAutoEncoder, create_lgc_compressors
from ...communications.protocol import MessageType, SyncInfos
from ...compressors.compress_utils import pack_raw_data_list, pack_tensor
from ...compressors.none import NoCompress
from ...models import create_model
from ...utils.training import init_buffer, seed_everything


class Client(ff_fedavg.Client):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.is_master_node = False
        """This is client the master node"""

        self.current_stage = 1
        """Current stage for the 3-stage training (1, 2, 3)"""

        self.grad_topk, self.inno_topk = create_lgc_compressors(cfg, self.net)
        self.encoder = LGCAutoEncoder.create_encoder().to(self.devices[0])
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        # Accumulated gradient for error feedback (gradient correction)
        self.grad_acc = create_model(cfg, dataset=self.train_dataset)
        init_buffer(self.grad_acc, self.devices[0])

    def init_comm_round(self):
        super().init_comm_round()
        self.log.info("Stage %s", self.current_stage)

    def update_sync_info(self):
        msg_type, data = self.dispatcher.recv_msg()
        self.handle_bye(msg_type)
        assert msg_type == MessageType.SYNC_INFOS

        sync_info = SyncInfos().decode(data)
        self.global_lr = sync_info.data.lr
        self.global_random_seed = sync_info.data.seed
        self.selected = sync_info.data.selected
        self.current_stage = sync_info.data.stage

        if 'is_master' in sync_info.data and sync_info.data['is_master']:
            self.log.info("Setting up autoencoder for master node...")

            msg_type, ae_data = self.dispatcher.recv_msg()
            self.handle_bye(msg_type)
            assert msg_type == MessageType.MODEL

            self.is_master_node = True
            for local_param, param in zip(
                    self.encoder.parameters(),
                    NoCompress().decompress(ae_data)):
                local_param.copy_(param.to(self.devices[0]))

        seed_everything(self.global_random_seed)

    def calc_gradient(self):
        super().calc_gradient()
        self.amortize_gradient()

    def amortize_gradient(self):
        for grad, g_acc in zip(
                self.gradient.parameters(), self.grad_acc.parameters()):
            grad.add_(g_acc)

    def aggregate(self):
        data = None
        if self.current_stage == 1:
            # Send full gradient in the first stage
            super().aggregate()
            return

        if self.current_stage == 2:
            # sent compressed gradient info to aggregator
            # data = self.grad_topk.compress(list(self.gradient.parameters()))

            data = pack_raw_data_list([
                pack_tensor(t.cpu()) for pair in
                self.grad_topk.compressed_data_and_idx(
                    list(self.gradient.parameters()))
                for t in pair])
        else:
            assert self.current_stage == 3
            inno_grad = self.inno_topk.compress(
                list(self.gradient.parameters()))
            if self.is_master_node:
                compressed_data_idx = self.grad_topk.compressed_data_and_idx(
                    list(self.gradient.parameters()))
                # For master node, send encoded intermediate representation
                enc_grad = pack_tensor(self.encoder(
                    torch.cat([
                        d[0] for d in compressed_data_idx
                    ]).reshape((1, 1, -1)).to(self.devices[0])).cpu())
                packed_idx = [
                    pack_tensor(d[1].cpu()) for d in compressed_data_idx]
                data = pack_raw_data_list([enc_grad, inno_grad, *packed_idx])
            else:
                # For other nodes, send innovation gradient
                data = inno_grad

        self.log.info("Sending gradient (%s bytes)", len(data))
        self.dispatcher.send_msg(MessageType.GRADIENT, data)

        # Update gradient accmumulation
        self.grad_topk.zero_with_mask(self.gradient)
        for g_acc, grad in zip(
                self.grad_acc.parameters(), self.gradient.parameters()):
            # g_acc.add_(grad) # This seems wrong according to DGC...
            g_acc.copy_(grad)

"""Learned-gradient compression client."""
import itertools
import torch
from .. import ff_fedavg
from .utils import create_cltk_compressor
from ...communications.protocol import MessageType, SyncInfos
from ...compressors.compress_utils import pack_raw_data_list, unpack_raw_data_list
from ...compressors.topk import TopKPerLayer
from ...compressors.none import pack_tensor, unpack_tensor
from ...models import create_model
from ...security.rsa import Offset
from ...utils.training import init_buffer


class Client(ff_fedavg.Client):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.grad_topk: TopKPerLayer
        self.is_master = False

        # Accumulated gradient for error feedback (gradient correction)
        self.grad_acc = create_model(cfg, dataset=self.train_dataset)
        init_buffer(self.grad_acc, self.devices[0])

    def _update_sync_info(self, sync_info: SyncInfos):
        super()._update_sync_info(sync_info)
        self.is_master = sync_info.data.is_master
        self.grad_topk = create_cltk_compressor(
            self.net, sync_info.data.topk_ratio, self.cfg)

    def calc_gradient(self):
        super().calc_gradient()
        self.amortize_gradient()

    def amortize_gradient(self):
        for grad, g_acc in zip(
                self.gradient.parameters(), self.grad_acc.parameters()):
            grad.add_(g_acc)

    def aggregate(self):
        def send_data(data):
            self.log.info("Sending gradient (%s bytes)", len(data))
            self.dispatcher.send_msg(MessageType.GRADIENT, data)

        # sent compressed gradient info to aggregator
        if self.is_master:
            data = pack_raw_data_list(list(itertools.chain(*[
                [pack_tensor(value.cpu()), pack_tensor(idx.cpu())]
                for value, idx in self.grad_topk.compressed_data_and_idx(
                    list(self.gradient.parameters()))
            ])))
            send_data(data)

            # Update gradient accumulation
            self.grad_topk.zero_with_mask(self.gradient)
        else:
            # Recv the indices mask from master client
            msg_type, data = self.dispatcher.recv_msg()
            self.handle_bye(msg_type)

            assert msg_type == MessageType.INDICES
            indices = [
                unpack_tensor(d, Offset()) for d in
                unpack_raw_data_list(data, Offset())]

            # Select data according to master index
            data = pack_tensor(torch.cat([
                torch.index_select(
                    param.clone().detach().view(-1), 0, idx.to(param.device))
                for param, idx in zip(self.gradient.parameters(), indices)
            ]).cpu())
            send_data(data)

            # Update gradient accumulation
            for param, idx in zip(self.gradient.parameters(), indices):
                param.view(-1).index_fill_(0, idx.to(param.device), 0)

        for g_acc, grad in zip(
                self.grad_acc.parameters(), self.gradient.parameters()):
            g_acc.copy_(grad)

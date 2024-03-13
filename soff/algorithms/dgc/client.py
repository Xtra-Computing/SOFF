"""Learned-gradient compression client."""
import gzip
import numpy as np
from torch import nn
from .. import ff_fedavg
from ...communications.protocol import MessageType, SyncInfos
from ...compressors.topk import TopKPerModel
from ...models import create_model
from ...utils.arg_parser import Conf
from ...utils.training import all_params, init_buffer


class Client(ff_fedavg.Client):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.grad_topk = TopKPerModel(Conf({
            'compression.topk_per_model.ratio': cfg.dgc.gradient_init_ratio}))

        # Accumulated gradient for error feedback (gradient correction)
        self.grad_acc = create_model(cfg, dataset=self.train_dataset)
        init_buffer(self.grad_acc, self.devices[0])

    def _update_sync_info(self, sync_info: SyncInfos):
        super()._update_sync_info(sync_info)
        self.grad_topk = TopKPerModel(Conf({
            'compression.topk_per_model.ratio':  sync_info.data.topk_ratio}))

    def calc_gradient(self):
        super().calc_gradient()
        self.amortize_gradient()

    def amortize_gradient(self):
        for grad, g_acc in zip(
                self.gradient.parameters(), self.grad_acc.parameters()):
            grad.add_(g_acc)

    def aggregate(self):
        # sent compressed gradient info to aggregator
        data = self.grad_topk.compress(all_params(self.gradient))

        self.datalogger.add_scalar(
            "Number idx", sum([
                param.numel() for param in self.net.parameters()]
            ) * self.grad_topk.compressor.ratio, self.epochs)
        self.datalogger.add_scalar(
            "Golomb Idx", self.__golomb_idx_size(
                self.net, self.grad_topk.compressor.ratio), self.epochs)

        self.datalogger.add_scalar("Data Size", len(data), self.epochs)
        data = gzip.compress(data)  # , preset=9)
        self.datalogger.add_scalar("LZMA Data Size", len(data), self.epochs)

        self.log.info("Sending gradient (%s bytes)", len(data))
        self.dispatcher.send_msg(MessageType.GRADIENT, data)

        # Update gradient accmumulation
        self.grad_topk.zero_with_mask(self.gradient)
        for g_acc, grad in zip(
                self.grad_acc.parameters(), self.gradient.parameters()):
            g_acc.copy_(grad)


    @staticmethod
    def __golomb_idx_size(net: nn.Module, sparse_rate: float):
        # for sparsification ratio >5, sending full graident is more efficient
        if sparse_rate >= 0.5:
            return 0

        layer_num_elems = [param.numel() for param in net.parameters()]
        num = sum(layer_num_elems)

        M = round(-1 / np.log2(1 - sparse_rate))
        b = round(np.log2(M))
        estimated_compression_ratio = sparse_rate / 2 * \
            (b + 1 / (1 - np.power(1 - sparse_rate / 2, 2**b)))
        compressed_size = num * 2 * estimated_compression_ratio / \
            32 + 1  # The unit is float, not bit
        return compressed_size

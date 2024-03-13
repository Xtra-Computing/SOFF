"""Hybrid-gradient compression client."""
import gzip
import itertools
from typing import List
import torch
from torch import LongTensor, Tensor
from .. import ff_fedavg
from ...communications.protocol import MessageType, SyncInfos
from ...compressors.compress_utils import pack_raw_data_list, unpack_raw_data_list
from ...compressors.topk import TopKPerLayer
from ...compressors.none import pack_tensor, unpack_tensor
from ...models import create_model
from ...security.rsa import Offset
from ...utils.optimizer import create_optimizer
from ...utils.training import init_buffer
from .utils import create_stc_compressor, golomb_idx_size
from .quantizer import QuantizeResult, create_quantizer
from .encoder import IdealCoder, create_encoder


class Client(ff_fedavg.Client):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Encoder uses torch.histc which doesn't support determinism
        torch.use_deterministic_algorithms(False)

        self.topk_ratio = -1
        self.grad_topk: TopKPerLayer
        self.is_master = False
        self.master_indices: List[Tensor] = [
            LongTensor() for _ in self.net.parameters()]

        self.quantizer = create_quantizer(cfg)
        self.encoder = create_encoder(cfg)

        self.ideal_coder = IdealCoder(cfg)
        """For measuring entropy only"""

        # Accumulated gradient for error feedback (gradient correction)
        self.grad_acc = create_model(cfg, dataset=self.train_dataset)
        init_buffer(self.grad_acc, torch.device('cpu'))

    def load_resources(self):
        super().load_resources()
        self.grad_acc = self.grad_acc.to(self.devices[0])

    def unload_resources(self):
        self.grad_acc = self.grad_acc.cpu()
        super().unload_resources()

    def _update_sync_info(self, sync_info: SyncInfos):
        super()._update_sync_info(sync_info)
        self.is_master = sync_info.data.is_master

        if sync_info.data.clear_ef_buffer:
            self.log.warning("Clearing error feedback buffer ...")
            # clear gradient accumulation
            for g_acc in self.grad_acc.parameters():
                g_acc.zero_()
            # clear momentum buffer
            self.optimizer = create_optimizer(self.cfg, self.net.parameters())
            assert len(self.optimizer.state) == 0

        if self.topk_ratio != sync_info.data.topk_ratio:
            self.grad_topk = create_stc_compressor(
                self.net, sync_info.data.topk_ratio, self.cfg)
            self.topk_ratio = sync_info.data.topk_ratio

    def update_global_params(self):
        msg_type, data = self.dispatcher.recv_msg()
        self.handle_bye(msg_type)
        assert msg_type == MessageType.GRADIENT

        # update global net using global grad
        data = gzip.decompress(data)

        raw_list = unpack_raw_data_list(data, Offset())
        quant_grad = [
            self.encoder.decode(d)
            for d in raw_list[::2]]
        indices = [
            unpack_tensor(d, Offset())
            for d in raw_list[1::2]]

        self.net_global.train()
        with torch.no_grad():
            restored_grad = [
                torch.zeros(p.numel(), device=p.device).index_copy_(
                    0, idx.to(p.device),
                    self.quantizer.dequantize(q).to(p.device)
                ).reshape(p.shape) for p, q, idx in
                zip(self.net.parameters(), quant_grad, indices)]

            for grad, param in zip(restored_grad, self.net_global.parameters()):
                param.grad = grad.clone()

        self.optimizer_global.step()
        self.net_global.eval()

    def calc_gradient(self):
        super().calc_gradient()
        self.amortize_gradient()

    def amortize_gradient(self):
        for grad, g_acc in zip(
                self.gradient.parameters(), self.grad_acc.parameters()):
            grad.add_(g_acc)

    def aggregate(self):
        def mask_and_quant(tensors: List[Tensor], masks) -> List[QuantizeResult]:
            return [self.quantizer.quantize(
                tensor.view(-1).index_select(0, mask.to(tensor.device))
            ) for tensor, mask in zip(tensors, masks)]

        def send_data(data):
            self.log.info("Sending gradient (%s bytes)", len(data))
            self.dispatcher.send_msg(MessageType.GRADIENT, data)

        # Update gradient accumulation
        topk_idxs = [
            idx.to(self.devices[0]) for _, idx in
            self.grad_topk.compressed_data_and_idx(
                list(self.gradient.parameters()))]

        self.datalogger.add_scalar(
            "Number idx", sum([idx.numel() for idx in topk_idxs]), self.epochs)
        self.datalogger.add_scalar(
            "Golomb Idx", golomb_idx_size(self.cfg, self.net, self.topk_ratio), self.epochs)

        quant_grad = mask_and_quant(
            list(self.gradient.parameters()), topk_idxs)

        self.datalogger.add_scalar("Grad Entropy", self.ideal_coder.entropy(
            torch.cat([res.quantized_tensor for res in quant_grad], dim=0)
        ), self.epochs)

        data = pack_raw_data_list(list(itertools.chain.from_iterable([
            self.encoder.encode(q_r), pack_tensor(idx.cpu())
        ] for q_r, idx in zip(quant_grad, topk_idxs))))

        self.datalogger.add_scalar("Data Size", len(data), self.epochs)
        data = gzip.compress(data)  # , preset=9)
        self.datalogger.add_scalar("LZMA Data Size", len(data), self.epochs)
        send_data(data)

        # Update gradient accumulation (count both topk and quantization err)
        for grad, idx, quant in zip(
                self.gradient.parameters(), topk_idxs, quant_grad):
            grad.view(-1).index_copy_(0, idx, (
                grad.view(-1).index_select(0, idx) -
                self.quantizer.dequantize(quant)))

        for g_acc, grad in zip(
                self.grad_acc.parameters(), self.gradient.parameters()):
            g_acc.copy_(grad)

"""Hybrid-gradient compression client."""
import lzma
import gzip
import itertools
from typing import List
import torch
import numpy as np
from torch import LongTensor, Tensor, optim
from .. import ff_fedavg
from ...communications.protocol import MessageType, SyncInfos
from ...compressors.compress_utils import pack_raw_data_list, unpack_raw_data_list
from ...compressors.topk import TopKPerLayer
from ...compressors.none import pack_tensor, unpack_tensor
from ...models import create_model
from ...models.lstm import PerEpochLSTMTrainer, PerIterLSTMTrainer
from ...security.rsa import Offset
from ...models.base import PerEpochTrainer, PerIterTrainer
from ...utils.optimizer import create_optimizer
from ...utils.training import init_buffer
from .utils import create_ugc_compressor, golomb_idx_size
from .quantizer import QuantizeResult, create_quantizer
from .encoder import IdealCoder, create_encoder
from .predictor import (
    AdaMPredictor, DeltaStepPredictor, MaskedAccGradPredictor,
    MaskedAdaMPreditor, create_predictor)
from ..fedprox.fedprox import (
    FedProxPerEpochLSTMTrainer, FedProxPerEpochTrainer,
    FedProxPerIterLSTMTrainer, FedProxPerIterTrainer)


class Client(ff_fedavg.Client):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Encoder uses torch.histc which doesn't support determinism
        torch.use_deterministic_algorithms(False)

        self.topk_ratio = -1
        self.grad_topk: TopKPerLayer
        self.is_master = False
        self.current_stage = 1
        self.global_epochs = 0
        self.master_indices: List[Tensor] = [
            LongTensor() for _ in self.net.parameters()]

        self.predictor = create_predictor(cfg, list(
            param.to(self.devices[0]) for param in self.net.parameters()))
        self.quantizer = create_quantizer(cfg)
        self.encoder = create_encoder(cfg)

        self.ideal_coder = IdealCoder(cfg)
        """For measuring entropy only"""

        # Accumulated gradient for error feedback (gradient correction)
        self.grad_acc = create_model(cfg, dataset=self.train_dataset)
        init_buffer(self.grad_acc, torch.device('cpu'))

        self.gmm_grad = create_model(cfg, dataset=self.train_dataset)
        init_buffer(self.gmm_grad, torch.device('cpu'))

        self._model_saved = False
        if len(cfg.ugc.continue_from) > 0:
            self._load_state(
                f"{cfg.ugc.continue_from}_cli_{self.client_id}.torch")

        # Use FedProx Trainer
        if isinstance(self.trainer, PerIterLSTMTrainer):
            self.trainer = FedProxPerIterLSTMTrainer(
                cfg, self.train_loader, self.train_criterion,
                self.additional_metrics, self.datalogger)
        elif isinstance(self.trainer, PerEpochLSTMTrainer):
            self.trainer = FedProxPerEpochLSTMTrainer(
                cfg, self.train_loader, self.train_criterion,
                self.additional_metrics, self.datalogger)
        elif isinstance(self.trainer, PerEpochTrainer):
            self.trainer = FedProxPerEpochTrainer(
                cfg, self.train_loader, self.train_criterion,
                self.additional_metrics, self.datalogger)
        elif isinstance(self.trainer, PerIterTrainer):
            self.trainer = FedProxPerIterTrainer(
                cfg, self.train_loader, self.train_criterion,
                self.additional_metrics, self.datalogger)
        else:
            raise RuntimeError("Model trainer not initailized")

    def load_resources(self):
        super().load_resources()
        self.grad_acc = self.grad_acc.to(self.devices[0])
        self.gmm_grad = self.gmm_grad.to(self.devices[0])

    def unload_resources(self):
        self.grad_acc = self.grad_acc.cpu()
        self.gmm_grad = self.gmm_grad.cpu()
        super().unload_resources()

    def train_one_round(self):
        return self.trainer.train_model_fedprox(
            self.net_global, self.net, self.optimizer, self.iters)

    def _update_sync_info(self, sync_info: SyncInfos):
        super()._update_sync_info(sync_info)
        self.is_master = sync_info.data.is_master
        self.current_stage = sync_info.data.current_stage
        self.global_epochs = sync_info.data.global_epochs

        if sync_info.data.clear_ef_buffer:
            self.log.warning("Clearing error feedback buffer ...")
            # clear gradient accumulation
            for g_acc in self.grad_acc.parameters():
                g_acc.zero_()
            # clear momentum buffer
            self.optimizer = create_optimizer(self.cfg, self.net.parameters())
            assert len(self.optimizer.state) == 0

        if self.topk_ratio != sync_info.data.topk_ratio:
            self.grad_topk = create_ugc_compressor(
                self.net, sync_info.data.topk_ratio, self.cfg)
            self.topk_ratio = sync_info.data.topk_ratio

        if not self._model_saved and sync_info.data.save_model:
            self._save_state(
                f"{self.cfg.ugc.save_stem}_cli_{self.client_id}.torch")
            self._model_saved = True

    def init_comm_round(self):
        super().init_comm_round()
        self.log.info("Stage %s", self.current_stage)

        mwes = self.cfg.ugc.momentum_warmup_epochs
        for i, epoch in enumerate(mwes[::-1]):
            if (self.global_epochs >= epoch and
                    isinstance(self.optimizer, optim.SGD)):
                mmntm = self.cfg.ugc.target_momentum * (1 - i / len(mwes))
                for param_group in self.optimizer.param_groups:
                    if param_group['momentum'] != mmntm:
                        self.log.warning("Setting local momentum to %s", mmntm)
                        param_group['momentum'] = mmntm
                break

    def update_global_params(self):
        msg_type, data = self.dispatcher.recv_msg()
        self.handle_bye(msg_type)
        assert msg_type == MessageType.GRADIENT

        # update global net using global grad
        data = gzip.decompress(data)
        flat_grads = unpack_tensor(data, Offset())
        starts = np.cumsum([0] + [len(idx) for idx in self.master_indices])

        # update parameters and buffers
        self.net_global.train()
        # Accumulate to global momentum
        with torch.no_grad():
            for i, (gmmg, param) in enumerate(zip(
                    self.gmm_grad.parameters(), self.net_global.parameters())):
                grad = torch.zeros_like(gmmg)
                flat_grads = flat_grads.to(grad.device)
                grad.view(-1).index_copy_(
                    0, self.master_indices[i].to(grad.device),
                    flat_grads[starts[i]: starts[i+1]])
                gmmg.copy_(gmmg * self.cfg.ugc.global_momentum + grad)
                param.grad = gmmg.clone()
        self.optimizer_global.step()
        self.net_global.eval()

        if self.iters == 0:
            return

        with torch.no_grad():
            self.predictor.update_buffer(
                list(
                    param.grad.to(self.devices[0])
                    for param in self.net_global.parameters())
                if isinstance(self.predictor, (
                    AdaMPredictor, MaskedAdaMPreditor,
                    DeltaStepPredictor, MaskedAccGradPredictor))
                else list(
                    param.to(self.devices[0])
                    for param in self.net_global.parameters()))

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

        pred_weights = self.predictor.predict(
            list(self.net_global.parameters()))
        pred_grads = [
            param - pred for param, pred in
            zip(self.net_global.parameters(), pred_weights)]

        self.unload_resources_and_release(self.cfg.hardware.gpus)

        # Get indieces
        if self.is_master:
            # Update gradient accumulation
            idxs = [
                idx.to(self.devices[0]) for _, idx in
                self.grad_topk.compressed_data_and_idx(
                    list(self.gradient.parameters()))]
        else:
            # Recv the indices mask from master client
            msg_type, data = self.dispatcher.recv_msg()
            data = gzip.decompress(data)
            self.handle_bye(msg_type)

            assert msg_type == MessageType.INDICES
            idxs = [
                unpack_tensor(d, Offset()).to(self.devices[0]) for d in
                unpack_raw_data_list(data, Offset())]

        self.datalogger.add_scalar(
            "Number idx", sum([idx.numel() for idx in idxs]), self.epochs)
        self.datalogger.add_scalar(
            "Golomb Idx", golomb_idx_size(self.cfg, self.net, self.topk_ratio), self.epochs)

        self.acquire_and_load_resources(self.cfg.hardware.gpus)

        # For next round's global model update
        self.master_indices = idxs

        quant_pred = mask_and_quant(pred_grads, idxs)
        quant_grad = mask_and_quant(list(self.gradient.parameters()), idxs)

        quant_resi = [QuantizeResult(
            norms=qg.norms, quantized_tensor=(
                qp.quantized_tensor.int().bitwise_xor(
                    qg.quantized_tensor.int()).float()))
            for qg, qp in zip(quant_grad, quant_pred)]

        self.datalogger.add_scalar("Mask Pred Acc",  float(sum(
            torch.sum((r.quantized_tensor == torch.tensor(0)).float())
            for r in quant_resi
        ) / sum(r.quantized_tensor.numel() for r in quant_resi)), self.epochs)

        self.datalogger.add_scalar("Grad Entropy", self.ideal_coder.entropy(
            torch.cat([res.quantized_tensor for res in quant_grad], dim=0)
        ), self.epochs)
        self.datalogger.add_scalar("Resi Entropy", self.ideal_coder.entropy(
            torch.cat([res.quantized_tensor for res in quant_resi], dim=0)
        ), self.epochs)

        if self.current_stage == 1:
            quant_grad = list(self.gradient.parameters())
            if self.is_master:
                data = pack_raw_data_list(list(itertools.chain(*[[
                    pack_tensor(grad.view(-1).index_select(0, idx).cpu()),
                    pack_tensor(idx.cpu())
                ] for grad, idx in zip(self.gradient.parameters(), idxs)])))
            else:
                data = pack_tensor(torch.cat([
                    grad.detach().view(-1).index_select(0, idx)
                    for grad, idx in zip(self.gradient.parameters(), idxs)
                ]).cpu())
        else:
            # Send compressed gradient info to aggregator
            if self.is_master:
                data = pack_raw_data_list(list(itertools.chain(*[[
                    self.encoder.encode(q_r), pack_tensor(idx.cpu())
                ] for q_r, idx in zip(quant_resi, idxs)])))
            else:
                # Select data according to master index
                data = pack_raw_data_list([
                    self.encoder.encode(q_r) for q_r in quant_resi])

        self.datalogger.add_scalar("Data Size", len(data), self.epochs)
        data = gzip.compress(data)  # , preset=9)
        self.datalogger.add_scalar("LZMA Data Size", len(data), self.epochs)
        send_data(data)

        if self.current_stage == 1:
            for grad, idx in zip(self.gradient.parameters(), idxs):
                grad.view(-1).index_fill_(0, idx, 0)
        else:
            # Update gradient accumulation (count both topk and quantization err)
            for grad, idx, quant in zip(
                    self.gradient.parameters(), idxs, quant_grad):
                grad.view(-1).index_copy_(0, idx, (
                    grad.view(-1).index_select(0, idx) -
                    self.quantizer.dequantize(quant)))

        for g_acc, grad in zip(
                self.grad_acc.parameters(), self.gradient.parameters()):
            g_acc.copy_(grad)

    def _save_state(self, file):
        self.log.warning("Saving models to %s", file)
        torch.save({
            'net': self.net.state_dict(),
            'net_global': self.net_global.state_dict(),
            'grad_acc': self.grad_acc.state_dict()
        }, file)

    def _load_state(self, file):
        self.log.warning("Loading from %s", file)
        data = torch.load(file, map_location=self.devices[0])
        self.net.load_state_dict(data['net'])
        self.net_global.load_state_dict(data['net_global'])
        self.grad_acc.load_state_dict(data['grad_acc'])

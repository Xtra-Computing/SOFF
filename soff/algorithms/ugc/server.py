"""Hybrid-gradient compression server."""
from collections import deque
from typing import Deque, List, Set
import os
import lzma
import gzip
import pathlib
import torch
import numpy as np
from eventfd import EventFD
from torch import LongTensor, Tensor
from torch.optim.lr_scheduler import MultiStepLR
from .. import fedprox
from .. import ff_fedavg
from ...communications.protocol import MessageType, Protocol
from ...compressors.topk import TopKPerLayer
from ...models import create_model
from ...security.rsa import Offset
from ...utils.training import all_params, init_buffer
from ...compressors.compress_utils import (
    pack_tensor, unpack_tensor, pack_raw_data_list, unpack_raw_data_list)
from .utils import (
    create_ugc_compressor, fit_sparse_ratio_exp,
    fit_sparse_ratio_lin, fit_sparse_ratio_con)
from .encoder import UGCEncoderConfParser, create_encoder
from .predictor import (
    AdaMPredictor, DeltaStepPredictor, MaskedAccGradPredictor,
    MaskedAdaMPreditor, UGCPredictorConfParser, create_predictor)
from .quantizer import (
    QuantizeResult, UGCQuantizerConfParser, create_quantizer)


class ConfParser(
        UGCPredictorConfParser,
        UGCQuantizerConfParser,
        UGCEncoderConfParser,
        ff_fedavg.ConfParser,
        fedprox.ConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        server_args = self.add_argument_group(
            "UGC Configs (S,S->C)")

        server_args.add_argument(
            '-ugc.gir', '--ugc.gradient-init-ratio', type=float, default=0.25,
            help="Initial compression ratio")
        server_args.add_argument(
            '-ugc.wr', '--ugc.warmup-rounds', type=int, default=300,
            help="Communication rounds to warmup compression ratio")
        server_args.add_argument(
            '-ugc.wt', '--ugc.warmup-type', default='exponential',
            choices=('exponential', 'linear', 'constant'),
            help="Compression ratio warmup types.")

        server_args.add_argument(
            '-ugc.ce', '--ugc.correction-epochs', type=int, default=20,
            help="Epochs to correct accumulated sparsity erros")
        server_args.add_argument(
            '-ugc.cr', '--ugc.correction-ratio', type=float, default=0.01,
            help="Topk ratio to correct accumulated sparsity erros")

        server_args.add_argument(
            '-ugc.mwe', '--ugc.momentum-warmup-epochs', type=int,
            default=(60, 100, 140), nargs=3,
            help="[TEST] momentum warmup to boost accuracy")
        server_args.add_argument(
            '-ugc.tm', '--ugc.target-momentum', type=float,
            default=0.9,
            help="[TEST] target momentum for momentum warmup")

        server_args.add_argument(
            '-ugc.sr', '--ugc.stage-rounds',
            default=[300, 300], type=int, nargs=2,
            help="Number of comm rounds for first 2 stages in the three-stage "
            "training. The last stage depends on the total epochs/comm rounds")
        server_args.add_argument(
            '-ugc.gr', '--ugc.gradient-topk-ratio',
            type=float, default=0.001,
            help="Subsample ratio of of the master node.")
        server_args.add_argument(
            '-ugc.sp', '--ugc.share-predictor', action='store_false',
            help="Share predictor across clients")
        server_args.add_argument(
            '-ugc.gm', '--ugc.global-momentum', type=float, default=0.0,
            help="Global momentum")

        server_args.add_argument(
            '-ugc.sa', '--ugc.save-at', default=-1, type=int,
            help="Save model at epoch")
        server_args.add_argument(
            '-ugc.ss', '--ugc.save-stem', default="saves/model", type=str,
            help="Checkpoint file stem")
        server_args.add_argument(
            '-ugc.cf', '--ugc.continue-from', default="", type=str,
            help="Save model at epoch")


class Server(ff_fedavg.Server):
    """Server for the UGC"""
    @classmethod
    def conf_parser(cls):
        return ConfParser

    def __init__(self, cfg):
        super().__init__(cfg)

        # Encoder uses torch.histc which doesn't support determinism
        torch.use_deterministic_algorithms(False)

        assert cfg.fffedavg.broadcast_type == 'gradient'
        # assert cfg.ugc.warmup_rounds <= cfg.ugc.stage_rounds[0]
        assert self.scheduler.num_clients_each_round == cfg.client_server.num_clients

        self.master_client_id: int
        self.topk_ratio = cfg.ugc.gradient_init_ratio
        self.grad_topk: TopKPerLayer
        self.topk_ratio_warmup_func = (
            fit_sparse_ratio_exp(
                1, cfg.ugc.warmup_rounds,
                cfg.ugc.gradient_init_ratio, cfg.ugc.gradient_topk_ratio)
            if cfg.ugc.warmup_type == 'exponential' else
            fit_sparse_ratio_lin(
                1, cfg.ugc.warmup_rounds,
                cfg.ugc.gradient_init_ratio, cfg.ugc.gradient_topk_ratio)
            if cfg.ugc.warmup_type == 'linear' else
            fit_sparse_ratio_con(
                cfg.ugc.gradient_init_ratio))

        self.event_master_grad_received = EventFD()
        """Signal the master gradient is received and indices is to be send"""
        self.master_indices: List[Tensor] = [
            LongTensor().to(self.devices[0]) for _ in self.grads.parameters()]

        self.current_stage = 1
        """Current stage for the 3-stage training (1, 2, 3)"""

        self.predictor = create_predictor(cfg, list(self.net.parameters()))
        self.quantizer = create_quantizer(cfg)
        self.encoder = create_encoder(cfg)

        self.comm_round = 0
        self.clear_ef_buffer = False
        """Singal the client to clear error feedback buffer"""

        self.gmm_grad = create_model(cfg, dataset=self.eval_dataset)
        init_buffer(self.gmm_grad, self.devices[0])

        # Checkpointing infrastructures
        self._model_saved = False
        if len(cfg.ugc.continue_from) > 0:
            self._load_state(f"{cfg.ugc.continue_from}_server.torch")

        # TEST
        self.master_quant_grad: List[Tensor]
        self.last_indices: Deque[List[Set]] = deque(maxlen=4)

    def register_event_handlers(self):
        super().register_event_handlers()
        self.scheduler.dispatcher.register_fd_event(
            self.event_master_grad_received,
            self.schedule_broadcast_master_indices)

    def schedule_process_gradient(self, socket, data) -> None:
        """Process gradient sent by clients"""
        def process_gradient(data):
            # skip processing if client already disconnected
            with self.scheduler.clients_info_lock:
                if socket in self.scheduler.clients_socket_id_map:
                    # put data into slot
                    buf = self.client_nets_buffer.get_slot_for_receive()
                    buf.set_data(data)
                    buf.client_id = self.scheduler.clients_socket_id_map[socket]
                    buf.release_as_ready()
                    self.log.info("  Gradient of client %s ✔", buf.client_id)
                    self.sem_client_event.release()
        self.scheduler.dispatcher.schedule_task(process_gradient, data)

    def schedule_broadcast_master_indices(self, _) -> None:
        self.event_master_grad_received.clear()
        # Send master indices to all clients other than the master
        data = pack_raw_data_list(
            [pack_tensor(idx.cpu()) for idx in self.master_indices])
        data = gzip.compress(data)
        self.log.info(
            "Broadcasting idxs (%s B × %s cli)",
            len(data), len(self.selected_client_ids)-1)

        for cli_id in self.selected_client_ids:
            if cli_id == self.master_client_id:
                continue
            socket = self.scheduler.clients_id_socket_map[cli_id]
            self.scheduler.dispatcher.schedule_task(
                Protocol.send_data, socket, MessageType.INDICES,
                data, self.datalogger, self.epoch)
        self.scheduler.dispatcher.insert_task_barrier()

    def init_comm_round(self, comm_round) -> None:
        super().init_comm_round(comm_round)
        self.comm_round = comm_round

        if not self._model_saved and (self.epoch == self.cfg.ugc.save_at):
            self._save_state(f"{self.cfg.ugc.save_stem}_server.torch")
            self._model_saved = True

        # Select the master client
        self.master_client_id = sorted(list(self.selected_client_ids))[
            comm_round % self.scheduler.num_clients_each_round]
        self.log.info("Master client: %s", self.master_client_id)

        def update_warmup_ratio(rounds):
            self.topk_ratio = (max(
                self.cfg.ugc.gradient_topk_ratio,
                self.topk_ratio_warmup_func(rounds + 1))
                if rounds < self.cfg.ugc.warmup_rounds else
                self.cfg.ugc.gradient_topk_ratio)
            self.log.info("Setting top-k ratio to %s", self.topk_ratio)
            self.grad_topk = create_ugc_compressor(
                self.net, self.topk_ratio, self.cfg)

        if comm_round <= self.cfg.ugc.warmup_rounds:
            # Warmup
            update_warmup_ratio(comm_round)
        elif isinstance(self.lr_scheduler, MultiStepLR):
            # Error correction via ratio change
            if any(ms - self.cfg.ugc.correction_epochs <= self.epoch
                   < ms + self.cfg.ugc.correction_epochs
                   for ms in self.lr_scheduler.milestones.keys()):
                topk_ratio = min(self.cfg.ugc.correction_ratio, 1.)
                if self.topk_ratio != topk_ratio:
                    self.topk_ratio = topk_ratio
                    self.log.info("Setting top-k ratio to %s", self.topk_ratio)
                    self.grad_topk = create_ugc_compressor(
                        self.net, self.topk_ratio, self.cfg)
            elif self.topk_ratio != self.cfg.ugc.gradient_topk_ratio:
                # This branch is a sigle-comm-round pulse
                self.clear_ef_buffer = True
                self.topk_ratio = self.cfg.ugc.gradient_topk_ratio
                self.log.info("Setting top-k ratio to %s", self.topk_ratio)
                self.grad_topk = create_ugc_compressor(
                    self.net, self.topk_ratio, self.cfg)
            else:
                self.clear_ef_buffer = False

        if comm_round >= sum(self.cfg.ugc.stage_rounds):
            if self.current_stage == 3:
                return
            self.log.info("Entering stage 3")
            self.current_stage = 3
        elif comm_round >= self.cfg.ugc.stage_rounds[0]:
            if self.current_stage == 2:
                return
            self.log.info("Entering stage 2")
            self.current_stage = 2

    def gen_sync_info(self):
        """Generate syncronization info"""
        sync_info = super().gen_sync_info()
        sync_info.data['topk_ratio'] = self.topk_ratio
        sync_info.data['current_stage'] = self.current_stage
        sync_info.data['clear_ef_buffer'] = self.clear_ef_buffer
        sync_info.data['global_epochs'] = self.epoch

        # For debug
        sync_info.data['save_model'] = self._model_saved
        return sync_info

    def schedule_broadcast_sync_info(self) -> None:
        # Send trained encoder to the master node
        sync_info = self.gen_sync_info()
        for cli_id, socket in self.scheduler.clients_id_socket_map.items():
            # Select a master client each round
            sync_info.data['is_master'] = cli_id == self.master_client_id
            sync_info.data['selected'] = cli_id in self.selected_client_ids
            self.scheduler.dispatcher.schedule_task(
                Protocol.send_data, socket,
                MessageType.SYNC_INFOS, sync_info.encode(),
                self.datalogger, self.epoch)
        # must wait for syncinfo to finish, before sending gradient
        self.scheduler.dispatcher.insert_task_barrier()

    def schedule_broadcast_model_data(self) -> None:
        # Send sparsified and compressed gradient
        data = pack_tensor(torch.cat([
            grad.detach().view(-1).index_select(0, idx) for grad, idx in
            zip(self.grads.parameters(), self.master_indices)
        ], dim=0).cpu())
        data = gzip.compress(data)
        msg_type = MessageType.GRADIENT

        self.log.info(
            "  Broadcasting grad to %s (%s bytes × %s clients)",
            self.selected_client_ids, len(data), len(self.selected_client_ids))

        # broadcasting gradient
        for client_id in self.selected_client_ids:
            socket = self.scheduler.clients_id_socket_map[client_id]
            self.scheduler.dispatcher.schedule_task(
                Protocol.send_data, socket, msg_type, data,
                self.datalogger, self.epoch)

        # fire event once everything is sent
        self.scheduler.dispatcher.insert_task_barrier()

    def aggregate(self) -> None:
        """Aggregate client models"""

        # zero out all params in grads for client aggregation
        for param in all_params(self.grads):
            param.zero_()
        self.log.info("  Waiting for clients input...")
        selected_data_length = sum(
            self.scheduler.clients_dataset_length[id]
            for id in self.selected_client_ids)

        def mask_and_quant(tensors: List[Tensor], masks) -> List[QuantizeResult]:
            return [self.quantizer.quantize(
                tensor.view(-1).index_select(0, mask.to(tensor.device))
            ) for tensor, mask in zip(tensors, masks)]

        pred_weights = self.predictor.predict(list(self.net.parameters()))
        pred_grads = [
            param - pred for param, pred in
            zip(self.net.parameters(), pred_weights)]

        def aggregate_stage_1(i, buf, data, client_weight):
            if i == 0:
                assert buf.client_id == self.master_client_id
                raw_list = unpack_raw_data_list(data, Offset())
                values = [
                    unpack_tensor(d, Offset()).to(self.devices[0])
                    for d in raw_list[::2]]
                indices = [
                    unpack_tensor(d, Offset()).to(self.devices[0])
                    for d in raw_list[1::2]]
                params = self.grad_topk.decompress_data_and_idx(
                    list(param.shape for param in self.net.parameters()),
                    list(zip(values, indices)))

                # Broadcast the master's indices
                assert len(indices) == len(list(self.net.parameters()))
                self.master_indices = indices
                self.event_master_grad_received.set()
            else:
                flattened_params = unpack_tensor(
                    data, Offset()).to(self.devices[0])
                starts = np.cumsum(
                    [0] + [idx.numel() for idx in self.master_indices])
                params = [
                    torch.zeros(np.prod(param.shape)).to(self.devices[0])
                    .index_copy_(
                        0, idx, flattened_params[starts[i]:starts[i+1]]
                    ).reshape(param.shape) for i, (param, idx) in
                    enumerate(zip(self.net.parameters(), self.master_indices))]

            for grad, client_grad in zip(all_params(self.grads), params):
                # update parameter grad
                grad.set_(
                    ((grad if (grad is not None)
                      else torch.zeros_like(client_grad)) +
                     client_grad.to(self.devices[0]) * client_weight)
                    .type(grad.dtype).clone())

        def aggregate_stage_2(i, buf, data, client_weight):
            # First received gradient must be master client's gradient
            if i == 0:
                assert buf.client_id == self.master_client_id
                raw_list = unpack_raw_data_list(data, Offset())
                quant_resi = [self.encoder.decode(d) for d in raw_list[::2]]
                indices = [
                    unpack_tensor(d, Offset()).to(self.devices[0])
                    for d in raw_list[1::2]]

                # Broadcast the master's indices
                assert len(indices) == len(list(self.net.parameters()))
                self.master_indices = indices
                self.event_master_grad_received.set()
            else:
                quant_resi = [
                    self.encoder.decode(d) for d in
                    unpack_raw_data_list(data, Offset())]

            for q_r in quant_resi:
                q_r.quantized_tensor = q_r.quantized_tensor.to(self.devices[0])

            # Restore quantized gradient from residules
            quant_pred = mask_and_quant(pred_grads, self.master_indices)
            quant_grad = [QuantizeResult(
                norms=qr.norms, quantized_tensor=(
                    qr.quantized_tensor.int().bitwise_xor(
                        qp.quantized_tensor.int()).float()))
                for qr, qp in zip(quant_resi, quant_pred)]

            # TEST: master grad and other grad overlap
            # if i == 0:
            #     self.master_quant_grad = [
            #         q.quantized_tensor.int() for q in quant_grad]

            #     for j, old_idx in enumerate(self.last_indices):
            #         self.log.warning(
            #             "%s C-T IDX Sim: %s", j, sum(
            #                 len(set(idx.tolist()).intersection(oidx))
            #                 for idx, oidx in
            #                 zip(self.master_indices[1:-2], old_idx[1:-2])
            #             ) / sum(len(oidx) for oidx in old_idx[1:-2]))
            #     self.last_indices.append([
            #         set(idx.tolist()) for idx in self.master_indices])
            # else:
            #     self.log.warning("%s C-C QG Sim: %s", i, float(sum(
            #         sum((q.quantized_tensor.int() == mq).float())
            #         for q, mq in zip(quant_grad, self.master_quant_grad))
            #     ) / sum(mq.numel() for mq in self.master_quant_grad))

            with torch.no_grad():
                restored_grad = [
                    torch.zeros(p.numel(), device=p.device).index_copy_(
                        0, idx, self.quantizer.dequantize(q)
                    ).reshape(p.shape) for p, q, idx in
                    zip(self.net.parameters(), quant_grad, self.master_indices)]

            # if self.current_stage in {1, 2}:
            # update parameter and buffers gradients
            for grad, rq_grad in zip(all_params(self.grads), restored_grad):
                # update parameter grad
                grad.set_(
                    ((grad if (grad is not None)
                      else torch.zeros_like(rq_grad)) +
                     rq_grad.to(self.devices[0]) * client_weight)
                    .type(grad.dtype).clone())

        for i in range(len(self.selected_client_ids)):
            self.sem_client_event.acquire()
            buf = self.client_nets_buffer.get_slot_for_aggregate()
            data = bytearray(gzip.decompress(buf.data))

            # Aggregation
            client_weight = (
                self.scheduler.clients_dataset_length[buf.client_id] /
                selected_data_length)
            self.log.info(
                "  Aggregating client %s (weight %s)",
                buf.client_id, client_weight)

            if self.current_stage == 1:
                aggregate_stage_1(i, buf, data, client_weight)
            else:
                aggregate_stage_2(i, buf, data, client_weight)

            self.clients_comm_rounds[buf.client_id] += 1
            # release the net to make a vacancy for following receive
            buf.release_as_idle()

        # Accumulate to global momentum
        with torch.no_grad():
            for gmmg, grad in zip(
                    self.gmm_grad.parameters(), self.grads.parameters()):
                gmmg.copy_(gmmg * self.cfg.ugc.global_momentum + grad)

    def update_global_model(self) -> None:
        # update global model
        self.net.train()
        for prm, gmg in zip(self.net.parameters(), self.gmm_grad.parameters()):
            prm.grad = gmg.clone()
        self.stepper.step()
        self.net.eval()

        # Update predictor
        with torch.no_grad():
            self.predictor.update_buffer(
                list(self.grads.parameters())
                if isinstance(self.predictor, (
                    AdaMPredictor, MaskedAdaMPreditor,
                    DeltaStepPredictor, MaskedAccGradPredictor))
                else list(self.net.parameters()))

    def _save_state(self, file):
        self.log.warning("Saving models to %s", file)
        if not os.path.exists(pathlib.Path(file).parent):
            os.makedirs(pathlib.Path(file).parent)
        torch.save({
            'net': self.net.state_dict(),
        }, file)

    def _load_state(self, file):
        self.log.warning("Loading from %s", file)
        data = torch.load(file, map_location=self.devices[0])
        self.net.load_state_dict(data['net'])

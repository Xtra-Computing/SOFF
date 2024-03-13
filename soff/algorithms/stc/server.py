"""Hybrid-gradient compression server."""
from typing import List
import gzip
import itertools
import torch
from torch import LongTensor, Tensor
from .. import fedprox
from .. import ff_fedavg
from ...communications.protocol import MessageType, Protocol
from ...security.rsa import Offset
from ...utils.training import all_params
from ...compressors.compress_utils import (
    pack_tensor, unpack_tensor, pack_raw_data_list, unpack_raw_data_list)
from .utils import (
    create_stc_compressor, fit_sparse_ratio_con, fit_sparse_ratio_exp)
from .encoder import STCEncoderConfParser, create_encoder
from .quantizer import (
    QuantizeResult, STCQuantizerConfParser, create_quantizer)


class ConfParser(
        STCQuantizerConfParser,
        STCEncoderConfParser,
        ff_fedavg.ConfParser,
        fedprox.ConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        server_args = self.add_argument_group(
            "STC Configs (S,S->C)")

        server_args.add_argument(
            '-stc.wt', '--stc.warmup-type', default='exponential',
            choices=('exponential', 'constant'),
            help="Compression ratio warmup types.")
        server_args.add_argument(
            '-stc.gir', '--stc.gradient-init-ratio', type=float, default=0.25,
            help="Initial compression ratio")
        server_args.add_argument(
            '-stc.wr', '--stc.warmup-rounds', type=int, default=300,
            help="Communication rounds to warmup compression ratio")
        server_args.add_argument(
            '-stc.gr', '--stc.gradient-topk-ratio',
            type=float, default=0.001,
            help="Subsample ratio of of the master node.")


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
        assert self.scheduler.num_clients_each_round == cfg.client_server.num_clients

        self.master_client_id: int
        self.topk_ratio = cfg.stc.gradient_init_ratio
        self.grad_topk = create_stc_compressor(
            self.net, self.topk_ratio, self.cfg)
        self.topk_ratio_warmup_func = fit_sparse_ratio_exp(
            1, cfg.stc.warmup_rounds,
            cfg.stc.gradient_init_ratio, cfg.stc.gradient_topk_ratio)
        self.topk_ratio_warmup_func = (
            fit_sparse_ratio_exp(
                1, cfg.stc.warmup_rounds,
                cfg.stc.gradient_init_ratio, cfg.stc.gradient_topk_ratio)
            if cfg.stc.warmup_type == 'exponential' else
            fit_sparse_ratio_con(
                cfg.stc.gradient_init_ratio))

        self.quantizer = create_quantizer(cfg)
        self.encoder = create_encoder(cfg)

        self.topk_idxs = [
            idx.to(self.devices[0]) for _, idx in
            self.grad_topk.compressed_data_and_idx(
                list(self.grads.parameters()))]

        self.quant_grad = self._mask_and_quant(
            list(self.grads.parameters()), self.topk_idxs)

        self.comm_round = 0
        self.clear_ef_buffer = False
        """Singal the client to clear error feedback buffer"""

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

    def init_comm_round(self, comm_round) -> None:
        super().init_comm_round(comm_round)
        self.comm_round = comm_round

        # Select the master client
        self.master_client_id = sorted(list(self.selected_client_ids))[
            comm_round % self.scheduler.num_clients_each_round]
        self.log.info("Master client: %s", self.master_client_id)

        # Warmup
        if comm_round <= self.cfg.stc.warmup_rounds:
            self.topk_ratio = (max(
                self.cfg.stc.gradient_topk_ratio,
                self.topk_ratio_warmup_func(comm_round + 1))
                if comm_round < self.cfg.stc.warmup_rounds else
                self.cfg.stc.gradient_topk_ratio)
            self.log.info("Setting top-k ratio to %s", self.topk_ratio)
            self.grad_topk = create_stc_compressor(
                self.net, self.topk_ratio, self.cfg)
        else:
            self.topk_ratio = self.cfg.stc.gradient_topk_ratio
            self.log.info("Setting top-k ratio to %s", self.topk_ratio)
            self.grad_topk = create_stc_compressor(
                self.net, self.topk_ratio, self.cfg)

    def gen_sync_info(self):
        """Generate syncronization info"""
        sync_info = super().gen_sync_info()
        sync_info.data['topk_ratio'] = self.topk_ratio
        sync_info.data['clear_ef_buffer'] = self.clear_ef_buffer
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
        data = pack_raw_data_list(list(itertools.chain.from_iterable([
            self.encoder.encode(q_r), pack_tensor(idx.cpu())
        ] for q_r, idx in zip(self.quant_grad, self.topk_idxs))))

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

            raw_list = unpack_raw_data_list(data, Offset())
            quant_grad = [
                self.encoder.decode(d)
                for d in raw_list[::2]]
            indices = [
                unpack_tensor(d, Offset()).to(self.devices[0])
                for d in raw_list[1::2]]

            # Broadcast the master's indices
            assert len(indices) == len(list(self.net.parameters()))

            for q_g in quant_grad:
                q_g.quantized_tensor = q_g.quantized_tensor.to(self.devices[0])

            with torch.no_grad():
                restored_grad = [
                    torch.zeros(p.numel(), device=p.device).index_copy_(
                        0, idx, self.quantizer.dequantize(q)
                    ).reshape(p.shape) for p, q, idx in
                    zip(self.net.parameters(), quant_grad, indices)]

            # update parameter and buffers gradients
            for grad, rq_grad in zip(all_params(self.grads), restored_grad):
                # update parameter grad
                grad.set_(
                    ((grad if (grad is not None)
                      else torch.zeros_like(rq_grad)) +
                     rq_grad.to(self.devices[0]) * client_weight)
                    .type(grad.dtype).clone())

            self.clients_comm_rounds[buf.client_id] += 1
            # release the net to make a vacancy for following receive
            buf.release_as_idle()

    def _mask_and_quant(
            self, tensors: List[Tensor], masks) -> List[QuantizeResult]:
        return [self.quantizer.quantize(
            tensor.view(-1).index_select(0, mask.to(tensor.device))
        ) for tensor, mask in zip(tensors, masks)]

    def update_global_model(self) -> None:
        # Send sparsified and compressed gradient

        # Update gradient accumulation
        self.topk_idxs = [
            idx.to(self.devices[0]) for _, idx in
            self.grad_topk.compressed_data_and_idx(
                list(self.grads.parameters()))]

        self.quant_grad = self._mask_and_quant(
            list(self.grads.parameters()), self.topk_idxs)

        # update global model
        self.net.train()
        # for prm, grd in zip(self.net.parameters(), self.grads.parameters()):
        #     prm.grad = grd.clone()
        for prm, qnt, idx in zip(
                self.net.parameters(), self.quant_grad, self.topk_idxs):
            prm.grad = torch.zeros(prm.numel(), device=prm.device).index_copy_(
                0, idx.to(prm.device),
                self.quantizer.dequantize(qnt).to(prm.device)
            ).reshape(prm.shape)

        self.stepper.step()
        self.net.eval()

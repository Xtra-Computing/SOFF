"""
Learned gradient compression [1] server.

[1] L. Abrahamyan, Y. Chen, G. Bekoulis, and N. Deligiannis, “Learned
Gradient Compression for Distributed Deep Learning,” IEEE Trans. Neural
Netw. Learning Syst., pp. 1–1, 2021, doi: 10.1109/TNNLS.2021.3084806.
"""
from typing import Any, List
import torch
import numpy as np
from .. import ff_fedavg
from ..base.ff_base_server import FFBaseServerConfParser
from ...compressors.none import NoCompress
from ...utils.tensor_buffer import TensorBuffer
from ...utils.training import all_params
from ...communications.protocol import SyncInfos, MessageType, Protocol
from ...compressors.compress_utils import (
    Offset, unpack_raw_data_list, unpack_tensor)
from .utils import LGCAutoEncoder, create_lgc_compressors


class ConfParser(FFBaseServerConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        server_args = self.add_argument_group(
            "LGC Configs (S,S->C)")

        server_args.add_argument(
            '-lgc.mc', '--lgc.master-client',
            default=0, type=int,
            help="Id of master client (to send the encoded gradient)")
        server_args.add_argument(
            '-lgc.sr', '--lgc.stage-rounds',
            default=[200, 300], type=int, nargs=2,
            help="Number of comm rounds for first 2 stages in the three-stage "
            "training. The last stage depends on the total epochs/comm rounds")

        server_args.add_argument(
            '-lgc.gr', '--lgc.gradient-topk-ratio',
            default=1e-3, type=float,
            help="Subsample ratio of of the master node.")

        server_args.add_argument(
            '-lgc.ir', '--lgc.innovation-topk-ratio',
            default=1e-5, type=float,
            help="Innovation gradient subsample ratio.")

        server_args.add_argument(
            '-lgc.aelr', '--lgc.autoencoder-learning-rate',
            default=0.001, type=float,
            help="Learning rate to train the autoencoder.")
        server_args.add_argument(
            '-lgc.aete', '--lgc.autoencoder-training-epochs',
            default=1, type=int,
            help="Eopchs to train the autoencoder in each comm round.")
        server_args.add_argument(
            '-lgc.ael1', '--lgc.autoencoder-lambda-1',
            default=0.5, type=float,
            help="Reconstruction loss weight of the autoencoder")
        server_args.add_argument(
            '-lgc.ael2', '--lgc.autoencoder-lambda-2',
            default=0.5, type=float,
            help="Similarity loss weight of the autoencoder")


class Server(ff_fedavg.Server):
    """Server for the LGC"""
    @classmethod
    def conf_parser(cls):
        return ConfParser

    def __init__(self, cfg):
        super().__init__(cfg)

        assert cfg.lgc.innovation_topk_ratio < cfg.lgc.gradient_topk_ratio, \
            "Innovation topk ratio must be smaller than gradient topk ratio"
        assert 0 <= cfg.lgc.master_client < cfg.client_server.num_clients

        self.last_stage = 1
        """Last stage for the 3-stage training (1, 2, 3)"""
        self.current_stage = 1
        """Current stage for the 3-stage training (1, 2, 3)"""

        del self.client_nets_buffer
        self.cli_buffers: List[Any]
        """Use fixed buffer to cache client gradients since LGC requires
        differentiation between master and non-master clients"""

        self.grad_topk, self.inno_topk = create_lgc_compressors(cfg, self.net)

        self.autoencoder = LGCAutoEncoder(
            cfg.client_server.num_clients)
        self.autoencoder.to(self.devices[0])
        self.ae_optimizer = torch.optim.SGD(
            self.autoencoder.parameters(),
            lr=cfg.lgc.autoencoder_learning_rate)

        self.innovation_grads = TensorBuffer(cfg.server.num_cache)
        # NOTE: supports per-iter aggregation as in the oringinal paper

    def schedule_process_gradient(self, socket, data) -> None:
        def process_gradient(data):
            with self.scheduler.clients_info_lock:
                if socket not in self.scheduler.clients_socket_id_map:
                    return
                client_id = self.scheduler.clients_socket_id_map[socket]

            assert self.cli_buffers[client_id] is None, \
                f"Client {client_id} is not obeying the " \
                f"{self.__class__.__name__} protocol"

            self.cli_buffers[client_id] = data
            self.sem_client_event.release()
            self.log.info("  Gradient of client %s ✔", client_id)

        self.scheduler.dispatcher.schedule_task(process_gradient, data)

    def schedule_broadcast_sync_info(self) -> None:
        if not (self.last_stage == 2 and self.current_stage == 3):
            super().schedule_broadcast_sync_info()
            return

        # Send trained encoder to the master node
        sync_info = self.gen_sync_info()
        for cli_id, socket in self.scheduler.clients_id_socket_map.items():
            sync_info.data['is_master'] = cli_id == self.cfg.lgc.master_client
            sync_info.data['selected'] = cli_id in self.selected_client_ids
            self.scheduler.dispatcher.schedule_task(
                Protocol.send_data, socket,
                MessageType.SYNC_INFOS, sync_info.encode(),
                self.datalogger, self.epoch)

            if cli_id == self.cfg.lgc.master_client:
                self.scheduler.dispatcher.insert_task_barrier()
                self.scheduler.dispatcher.schedule_task(
                    Protocol.send_data, socket, MessageType.MODEL,
                    NoCompress().compress(all_params(self.autoencoder.encoder)),
                    self.datalogger, self.epoch)

        # must wait for syncinfo to finish, before sending gradient
        self.scheduler.dispatcher.insert_task_barrier()

    def gen_sync_info(self) -> SyncInfos:
        return SyncInfos().set_data({
            'lr': self.global_optimizer.param_groups[0]['lr'],
            'seed': self.global_random_seed,
            'stage': self.current_stage})

    def init_comm_round(self, comm_round) -> None:
        super().init_comm_round(comm_round)
        self.cli_buffers = [
            None for _ in range(self.cfg.client_server.num_clients)]

        self.last_stage = self.current_stage
        if comm_round >= sum(self.cfg.lgc.stage_rounds):
            if self.current_stage == 3:
                return
            self.log.info("Entering stage 3")
            self.current_stage = 3
            self.autoencoder.to(self.devices[0])
            self.autoencoder.eval()
            for param in self.autoencoder.parameters():
                param.requires_grad_(False)
        elif comm_round >= self.cfg.lgc.stage_rounds[0]:
            if self.current_stage == 2:
                return
            self.log.info("Entering stage 2")
            self.current_stage = 2

    def aggregate(self) -> None:
        """Aggregate client models"""

        # Wait for _all_ client's input to be gathered
        self.log.info("  Waiting for clients input...")
        selected_data_length = sum(
            self.scheduler.clients_dataset_length[id]
            for id in self.selected_client_ids)
        for _ in range(len(self.selected_client_ids)):
            # handle client events
            self.sem_client_event.acquire()

        # zero out all params in grads for client aggregation
        for param in all_params(self.grads):
            param.zero_()

        # encoded gradient is shared in the 3rd stage
        enc_grad, inno_grad = torch.Tensor(), []
        unpacked_idx = []
        if self.current_stage == 3:
            master_data = self.cli_buffers[self.cfg.lgc.master_client]
            assert master_data is not None
            data_list = unpack_raw_data_list(master_data, Offset())
            # Extract shared data from the master client
            enc_grad = unpack_tensor(data_list[0], Offset())
            inno_grad = self.inno_topk.decompress(data_list[1])
            unpacked_idx = [
                unpack_tensor(d, Offset()).to(self.devices[0])
                for d in data_list[2:]]

        for client_id in self.selected_client_ids:
            # aggregation
            client_weight = (
                self.scheduler.clients_dataset_length[client_id] /
                selected_data_length)
            self.log.info(
                "  Aggregating client %s (weight %s)",
                client_id, client_weight)

            self.clients_comm_rounds[client_id] += 1

            raw_data = self.cli_buffers[client_id]
            assert raw_data is not None
            if self.current_stage == 1:
                data = NoCompress().decompress(raw_data)
            elif self.current_stage == 2:
                # data = self.grad_topk.decompress(raw_data)
                raw_list = unpack_raw_data_list(raw_data, Offset())
                unpacked_data = [
                    (unpack_tensor(d, Offset()), unpack_tensor(i, Offset()))
                    for d, i in zip(raw_list[::2], raw_list[1::2])]
                data = self.grad_topk.decompress_data_and_idx(
                    list(param.shape for param in self.net.parameters()),
                    unpacked_data)

                # Store restored params to cli_buffers for autoencoder training
                self.cli_buffers[client_id] = data
            elif self.current_stage == 3:
                if client_id != self.cfg.lgc.master_client:
                    inno_grad = self.inno_topk.decompress(raw_data)
                flat_data = self.autoencoder.decode(
                    client_id, enc_grad.to(self.devices[0]), torch.cat([
                        g[0] for g in
                        self.grad_topk.compressed_data_and_idx(inno_grad)
                    ]).reshape((1, 1, -1)).to(self.devices[0])).reshape(-1)

                # Re-split concatenated and flattened data to each layer
                start_idxs = np.cumsum(
                    [0] + [idx.numel() for idx in unpacked_idx])
                data = [
                    flat_data[start_idxs[i]:start_idxs[i+1]] for i in
                    range(len(unpacked_idx))]

                # Restore from each layer's data
                data = self.grad_topk.decompress_data_and_idx(
                    list(param.shape for param in self.net.parameters()),
                    list(zip(data, unpacked_idx)))

                # num_elems = 0
                # for grad in self.grads.parameters():
                #     data.append(
                #         flat_data[num_elems: num_elems + grad.numel()]
                #         .reshape(grad.shape))
                #     num_elems += grad.numel()
            else:
                raise RuntimeError("Unknown stage")

            # update parameter and buffers gradients
            for grad, client_grad in zip(self.grads.parameters(), data):
                # update parameter grad
                grad.set_(
                    ((grad if (grad is not None)
                      else torch.zeros_like(client_grad)) +
                     client_grad.to(self.devices[0]) * client_weight)
                    .type(grad.dtype).clone())

        self.train_autoencoder()

    def train_autoencoder(self):
        # Autoencoder is trained only on stage 2
        if self.current_stage != 2:
            return

        self.log.info("Start training autoencoder")
        self.autoencoder.train()
        for _ in range(self.cfg.lgc.autoencoder_training_epochs):
            self.autoencoder.zero_grad()
            topk_grads = []
            inno_grads = []
            for cli_id in self.selected_client_ids:
                buffer = self.cli_buffers[cli_id]
                assert buffer is not None

                topk_grads.append(torch.cat([
                    g[0] for g in
                    self.grad_topk.compressed_data_and_idx(buffer)
                ]).reshape((1, 1, -1)).to(self.devices[0]))

                inno_grads.append(torch.cat([
                    g[0] for g in
                    self.grad_topk.compressed_data_and_idx(
                        self.inno_topk.decompress(
                            self.inno_topk.compress(buffer)))
                ]).reshape((1, 1, -1)).to(self.devices[0]))

            encoded, decoded = self.autoencoder(topk_grads, inno_grads)
            rec_loss, sim_loss = self.autoencoder.compute_losses(
                topk_grads, encoded, decoded)

            loss = (
                self.cfg.lgc.autoencoder_lambda_1 * rec_loss +
                self.cfg.lgc.autoencoder_lambda_2 * sim_loss)

            self.log.info(
                "Autoencoder rec loss: %s, sim loss: %s, loss: %s",
                float(rec_loss), float(sim_loss), float(loss))
            loss.backward()
            self.ae_optimizer.step()
        self.autoencoder.eval()

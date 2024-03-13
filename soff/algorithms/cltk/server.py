"""
Learned gradient compression [1] server.

[1] L. Abrahamyan, Y. Chen, G. Bekoulis, and N. Deligiannis, “Learned
Gradient Compression for Distributed Deep Learning,” IEEE Trans. Neural
Netw. Learning Syst., pp. 1–1, 2021, doi: 10.1109/TNNLS.2021.3084806.
"""
from typing import List
import torch
import numpy as np
from eventfd import EventFD
from .utils import create_cltk_compressor
from ..base.ff_base_server import MessageType, Protocol
from .. import ff_fedavg
from ...compressors.compress_utils import pack_raw_data_list, unpack_raw_data_list
from ...compressors.topk import TopKPerLayer
from ...compressors.none import pack_tensor, unpack_tensor
from ...security.rsa import Offset
from ...utils.training import all_params


class ConfParser(ff_fedavg.ConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        server_args = self.add_argument_group(
            "CLT-k Configs (S,S->C)")
        server_args.add_argument(
            '-cltk.wr', '--cltk.warmup-rounds', type=int, default=500,
            help="Communication rounds to warmup")
        server_args.add_argument(
            '-cltk.gr', '--cltk.gradient-topk-ratio',
            type=float, default=0.001,
            help="Subsample ratio of of the master node.")


class Server(ff_fedavg.Server):
    """Server for the LGC"""
    @classmethod
    def conf_parser(cls):
        return ConfParser

    def __init__(self, cfg):
        super().__init__(cfg)

        self.master_client_id: int

        self.topk_ratio = 0.25
        self.grad_topk: TopKPerLayer

        self.event_master_grad_received = EventFD()
        """signal the master gradient is received and indices is to be send"""
        self.master_indices: List[torch.Tensor]

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
            [pack_tensor(idx) for idx in self.master_indices])
        self.log.info(
            "Broadcasting idxs (%s b × %s cli)",
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

        # Select the master client
        self.master_client_id = sorted(list(self.selected_client_ids))[
            comm_round % self.scheduler.num_clients_each_round]
        self.log.info("Master client: %s", self.master_client_id)

        if comm_round < self.cfg.cltk.warmup_rounds:
            self.topk_ratio = max(
                self.cfg.cltk.gradient_topk_ratio,
                0.25 ** (comm_round * 4 / self.cfg.cltk.warmup_rounds + 1))
            self.log.info("Setting top-k ratio to %s", self.topk_ratio)
            self.grad_topk = create_cltk_compressor(
                self.net, self.topk_ratio, self.cfg)

    def gen_sync_info(self):
        """Generate syncronization info"""
        sync_info = super().gen_sync_info()
        sync_info.data['topk_ratio'] = self.topk_ratio
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

            # First received gradient must be master client's gradient
            if i == 0:
                assert buf.client_id == self.master_client_id
                raw_list = unpack_raw_data_list(buf.data, Offset())
                values = [unpack_tensor(d, Offset()) for d in raw_list[::2]]
                indices = [unpack_tensor(d, Offset()) for d in raw_list[1::2]]
                params = self.grad_topk.decompress_data_and_idx(
                    list(param.shape for param in self.net.parameters()),
                    list(zip(values, indices)))

                # Broadcast the master's indices
                assert len(indices) == len(list(self.net.parameters()))
                self.master_indices = indices
                self.event_master_grad_received.set()
            else:
                flattened_params = unpack_tensor(buf.data, Offset())
                starts = np.cumsum(
                    [0] + [idx.numel() for idx in self.master_indices])
                params = [
                    torch.zeros(np.prod(param.shape)).index_copy_(
                        0, idx, flattened_params[starts[i]:starts[i+1]]
                    ).reshape(param.shape) for i, (param, idx) in enumerate(
                        zip(self.net.parameters(), self.master_indices))]

            # aggregation
            client_weight = (
                self.scheduler.clients_dataset_length[buf.client_id] /
                selected_data_length)
            self.log.info(
                "  Aggregating client %s (weight %s)",
                buf.client_id, client_weight)

            self.clients_comm_rounds[buf.client_id] += 1

            # update parameter and buffers gradients
            for grad, client_grad in zip(all_params(self.grads), params):
                # update parameter grad
                grad.set_(
                    ((grad if (grad is not None)
                      else torch.zeros_like(client_grad)) +
                     client_grad.to(self.devices[0]) * client_weight)
                    .type(grad.dtype).clone())

            # release the net to make a vacancy for following receive
            buf.release_as_idle()

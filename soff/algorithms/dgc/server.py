"""
Learned gradient compression [1] server.

[1] L. Abrahamyan, Y. Chen, G. Bekoulis, and N. Deligiannis, “Learned
Gradient Compression for Distributed Deep Learning,” IEEE Trans. Neural
Netw. Learning Syst., pp. 1–1, 2021, doi: 10.1109/TNNLS.2021.3084806.
"""
import gzip
import math
from typing import Callable
import torch
from .. import ff_fedavg
from ...compressors.topk import TopKPerModel
from ...utils.arg_parser import Conf
from ...utils.training import all_params


class ConfParser(ff_fedavg.ConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        server_args = self.add_argument_group(
            "DGC Configs (S,S->C)")

        server_args.add_argument(
            '-dgc.wt', '--dgc.warmup-type', default='exponential',
            choices=('exponential', 'linear', 'constant'),
            help="Compression ratio warmup types.")
        server_args.add_argument(
            '-dgc.gir', '--dgc.gradient-init-ratio', type=float, default=0.25,
            help="Initial ratio of the gradient compression."
        )
        server_args.add_argument(
            '-dgc.wr', '--dgc.warmup-rounds', type=int, default=100,
            help="Communication rounds to warmup")
        server_args.add_argument(
            '-dgc.gr', '--dgc.gradient-topk-ratio',
            type=float, default=1e-3,
            help="Subsample ratio of of the gradient compression.")


class Server(ff_fedavg.Server):
    """Server for the LGC"""
    @classmethod
    def conf_parser(cls):
        return ConfParser

    def __init__(self, cfg):
        super().__init__(cfg)
        self.topk_ratio = cfg.dgc.gradient_init_ratio
        self.grad_topk = TopKPerModel(Conf({
            'compression.topk_per_model.ratio': self.topk_ratio}))

        self.topk_ratio_warmup_func = (self.__fit_sparse_ratio_exp(
            1, cfg.dgc.warmup_rounds,
            cfg.dgc.gradient_init_ratio, cfg.dgc.gradient_topk_ratio)
            if cfg.dgc.warmup_type == 'exponential' else
            self.__fit_sparse_ratio_con(
                cfg.dgc.gradient_init_ratio))

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
        if comm_round < self.cfg.dgc.warmup_rounds:
            self.topk_ratio = max(
                self.cfg.dgc.gradient_topk_ratio,
                self.topk_ratio_warmup_func(comm_round + 1))
            self.log.info("Setting top-k ratio to %s", self.topk_ratio)
            self.grad_topk = TopKPerModel(Conf({
                'compression.topk_per_model.ratio': self.topk_ratio}))
        elif self.topk_ratio != self.cfg.dgc.gradient_topk_ratio:
            self.topk_ratio = self.cfg.dgc.gradient_topk_ratio
            self.log.info("Setting top-k ratio to %s", self.topk_ratio)
            self.grad_topk = TopKPerModel(Conf({
                'compression.topk_per_model.ratio': self.topk_ratio}))

    def gen_sync_info(self):
        """Generate syncronization info"""
        sync_info = super().gen_sync_info()
        sync_info.data['topk_ratio'] = self.topk_ratio
        return sync_info

    def aggregate(self) -> None:
        """Aggregate client models"""

        # zero out all params in grads for client aggregation
        for param in all_params(self.grads):
            param.zero_()

        self.log.info("  Waiting for clients input...")
        selected_data_length = sum(
            self.scheduler.clients_dataset_length[id]
            for id in self.selected_client_ids)

        for _ in range(len(self.selected_client_ids)):
            self.sem_client_event.acquire()
            buf = self.client_nets_buffer.get_slot_for_aggregate()
            data = bytearray(gzip.decompress(buf.data))

            # aggregation
            client_weight = (
                self.scheduler.clients_dataset_length[buf.client_id] /
                selected_data_length)
            self.log.info(
                "  Aggregating client %s (weight %s)",
                buf.client_id, client_weight)

            self.clients_comm_rounds[buf.client_id] += 1

            # update parameter and buffers gradients
            for grad, client_grad in zip(
                    all_params(self.grads),
                    self.grad_topk.decompress(data)):
                # update parameter grad
                grad.set_(
                    ((grad if (grad is not None)
                      else torch.zeros_like(client_grad)) +
                     client_grad.to(self.devices[0]) * client_weight)
                    .type(grad.dtype).clone())

            # release the net to make a vacancy for following receive
            buf.release_as_idle()

    @staticmethod
    def __fit_sparse_ratio_exp(
            x1: int, x2: int, y1: float, y2: float) -> Callable[[int], float]:
        assert x1 < x2 and y1 > y2
        b = math.pow(y1/y2, 1/(x2-x1))
        a = y1 / (b ** (-x1))

        def func(x):
            return a * (b ** (-x))
        return func

    @staticmethod
    def __fit_sparse_ratio_con(y: float) -> Callable[[int], float]:
        def func(_):
            return y
        return func

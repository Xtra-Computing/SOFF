import gzip
import torch
from .. import ff_fedavg
from ...compressors.compress_utils import Offset, unpack_raw_data_list
from ...utils.training import all_params
from .encoder import QSGDEncoderConfParser, create_encoder
from .quantizer import QSGDQuantizerConfParser, create_quantizer


class ConfParser(
        QSGDEncoderConfParser,
        QSGDQuantizerConfParser,
        ff_fedavg.ConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        server_args = self.add_argument_group(
            "QSGD Configs (S,S->C)")
        server_args.add_argument(
            '-qsgd.ef', '--qsgd.error-feedback', action='store_true',
            help="Enable error feedback buffer")
        server_args.add_argument(
            '-qsgd.gzip', '--qsgd.gzip', action='store_true',
            help="Enable gzip compression (slow)")


class Server(ff_fedavg.Server):
    """Server for the QSGD"""
    @classmethod
    def conf_parser(cls):
        return ConfParser

    def __init__(self, cfg):
        super().__init__(cfg)
        self.quantizer = create_quantizer(cfg)
        self.encoder = create_encoder(cfg)

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
            data = (
                bytearray(gzip.decompress(buf.data))
                if self.cfg.qsgd.gzip else buf.data)

            # aggregation
            client_weight = (
                self.scheduler.clients_dataset_length[buf.client_id] /
                selected_data_length)
            self.log.info(
                "  Aggregating client %s (weight %s)",
                buf.client_id, client_weight)
            self.clients_comm_rounds[buf.client_id] += 1

            raw_list = unpack_raw_data_list(data, Offset())
            restored_grad = [
                self.quantizer.dequantize(self.encoder.decode(data))
                for data in raw_list]

            # update parameter and buffers gradients
            for grad, client_grad in zip(all_params(self.grads), restored_grad):
                # update parameter grad
                grad.set_(
                    ((grad if (grad is not None)
                      else torch.zeros_like(client_grad)) +
                     client_grad.view(grad.shape).to(self.devices[0]) * client_weight)
                    .type(grad.dtype).clone())

            # release the net to make a vacancy for following receive
            buf.release_as_idle()

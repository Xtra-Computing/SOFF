"""Learned-gradient compression client."""
import gzip
from .. import ff_fedavg
from .encoder import create_encoder
from .quantizer import create_quantizer
from ...communications.protocol import MessageType
from ...compressors.compress_utils import pack_raw_data_list
from ...models import create_model
from ...utils.training import all_params, init_buffer


class Client(ff_fedavg.Client):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.quantizer = create_quantizer(cfg)
        self.encoder = create_encoder(cfg)

        # Accumulated gradient for error feedback (gradient correction)
        self.grad_acc = create_model(cfg, dataset=self.train_dataset)
        init_buffer(self.grad_acc, self.devices[0])

    def calc_gradient(self):
        super().calc_gradient()
        if self.cfg.qsgd.error_feedback:
            self.amortize_gradient()

    def amortize_gradient(self):
        for grad, g_acc in zip(
                self.gradient.parameters(), self.grad_acc.parameters()):
            grad.add_(g_acc)

    def aggregate(self):
        # sent compressed gradient info to aggregator
        self.log.info("Start Quant")
        quant_grads = [
            self.quantizer.quantize(grad.view(-1))
            for grad in all_params(self.gradient)]
        self.log.info("Start Encode and Pack")
        data = pack_raw_data_list([
            self.encoder.encode(quant) for quant in quant_grads])

        self.log.info("Start Compress")
        self.datalogger.add_scalar("Data Size", len(data), self.epochs)
        if self.cfg.qsgd.gzip:
            data = gzip.compress(data)
            self.datalogger.add_scalar(
                "LZMA Data Size", len(data), self.epochs)

        self.log.info("Sending gradient (%s bytes)", len(data))
        self.dispatcher.send_msg(MessageType.GRADIENT, data)

        # Update gradient accmumulation
        for g_acc, grad, q_g in zip(
                self.grad_acc.parameters(),
                self.gradient.parameters(), quant_grads):
            g_acc.copy_(grad - self.quantizer.dequantize(
                q_g).view(grad.shape).to(grad.device))

    def load_resources(self):
        super().load_resources()
        self.grad_acc = self.grad_acc.to(self.devices[0])

    def unload_resources(self):
        self.grad_acc = self.grad_acc.cpu()
        super().unload_resources()

import random
import logging
from typing import List, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ...compressors.topk import TopKPerLayer
from ...utils.arg_parser import Conf

log = logging.getLogger(__name__)


def create_lgc_compressors(cfg, net: nn.Module):
    # Map {model name -> (first x params, last x params)}
    lgc_non_compress_params = {
        'resnet':  (1, 2),
        'resnet18': (1, 2),
        'resnet34': (1, 2),
        'resnet50': (1, 2),
        'resnet101': (1, 2),
        'resnet152': (1, 2),
        'resnet20': (1, 2),
        'resnet32': (1, 2),
        'resnet44': (1, 2),
        'resnet56': (1, 2),
        'resnet110': (1, 2),
        'resnet120': (1, 2),
        'vgg11': (2, 2),
        'vgg13': (2, 2),
        'vgg16': (2, 2),
        'vgg19': (2, 2),
    }
    if cfg.model.name not in lgc_non_compress_params:
        raise RuntimeError(f"Model {cfg.model.name} not supported.")

    num_params = range(len(list(net.parameters())))
    grad_topk_ratios = [cfg.lgc.gradient_topk_ratio for _ in num_params]
    inno_topk_ratios = [cfg.lgc.innovation_topk_ratio for _ in num_params]
    for ratio in [grad_topk_ratios, inno_topk_ratios]:
        for i in range(lgc_non_compress_params[cfg.model.name][0]):
            ratio[i] = 1.
        for i in range(lgc_non_compress_params[cfg.model.name][1]):
            ratio[-1-i] = 1.

    return (
        TopKPerLayer(Conf({
            'compression.topk_per_layer.ratios': grad_topk_ratios})),
        TopKPerLayer(Conf({
            'compression.topk_per_layer.ratios': inno_topk_ratios})))


class LGCAutoEncoder(nn.Module):
    """Autoencoder for the LGC algorithm"""

    def __init__(self, num_clients) -> None:
        super().__init__()
        self.encoder = self.create_encoder()
        self.decoders = nn.ModuleList([
            self.create_decoder()
            for _ in range(num_clients)])
        self.outputs = nn.ModuleList([
            nn.Conv1d(33, 1, kernel_size=1, stride=1)
            for _ in range(num_clients)])

    def encode(self, topk_grad):
        """Encode with the encoder"""
        return self.encoder(topk_grad)

    def decode(self, decoder_idx, encoded, inno_grad):
        """Encode with the decoder"""
        decoded = self.decoders[decoder_idx](encoded)
        decoded = torch.cat((
            F.interpolate(decoded, size=inno_grad.shape[-1]), inno_grad), dim=1)
        decoded = self.outputs[decoder_idx](decoded)
        return decoded

    def forward(
            self, topk_grads: List[Tensor],
            inno_grads: List[Tensor]) -> \
            Tuple[List[Tensor], List[Tensor]]:
        """Forward pass of the autoencoder"""
        # input shape: [batch-size, 1 (channels), num-params]
        encoded = [self.encode(topk_grad) for topk_grad in topk_grads]
        # encoded = [nn.LeakyReLU()(enc) for enc in encoded]

        # randomly select one encoded to combine with inno_grads
        enc = random.choice(encoded)
        decoded = [
            self.decode(i, enc, inno_grad)
            for i, inno_grad in enumerate(inno_grads)]
        return encoded, decoded

    @staticmethod
    def create_encoder() -> nn.Sequential:
        """Create the encoder part for the autoencoder"""
        return nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(256, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(64, 4, kernel_size=1, stride=1),
            nn.LeakyReLU(),
        )

    @staticmethod
    def create_decoder() -> nn.Sequential:
        """Create the decoder part for the autoencoder"""
        return nn.Sequential(
            nn.ConvTranspose1d(4, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 128, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(),
        )

    @staticmethod
    def compute_losses(
            original: List[Tensor],
            encoded: List[Tensor],
            decoded: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Return (reconstruction loss, similarity loss)"""

        assert len(original) == len(encoded) == len(decoded)

        # Reconstruction loss
        rec_losses = []
        for orig, dec in zip(original, decoded):
            rec_losses.append(nn.MSELoss()(orig, dec))

        # Similarity loss
        sim_losses = []
        for i, enc_i in enumerate(encoded):
            for j, enc_j in enumerate(encoded):
                if i != j:
                    sim_losses.append(nn.MSELoss()(enc_i, enc_j))

        return (sum(rec_losses, start=torch.tensor(0.)),
                sum(sim_losses, start=torch.tensor(0.)))

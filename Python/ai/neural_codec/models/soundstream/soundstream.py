import torch.nn as nn

from .encoder import SoundStreamEncoder
from .decoder import SoundStreamDecoder
from .rvq import ResidualVectorQuantizer

class SoundStream(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        num_quantizers=8,
        codebook_size=1024,
        base_channels=32
    ):
        super().__init__()

        self.encoder = SoundStreamEncoder(base_channels, latent_dim)
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers, codebook_size, latent_dim
        )
        self.decoder = SoundStreamDecoder(base_channels, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        z = z.permute(0, 2, 1)  # (B, T, D)
        z_q, codes, vq_loss = self.quantizer(z)
        z_q = z_q.permute(0, 2, 1)
        x_hat = self.decoder(z_q)
        return x_hat, codes, vq_loss

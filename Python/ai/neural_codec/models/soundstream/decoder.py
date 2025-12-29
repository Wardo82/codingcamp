import torch
import torch.nn as nn

from .residuals import ResidualUnit1D

class DecoderBlock(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            channels, channels // 2,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2
        )
        self.residuals = nn.Sequential(
            ResidualUnit1D(channels // 2, dilation=1),
            ResidualUnit1D(channels // 2, dilation=3),
            ResidualUnit1D(channels // 2, dilation=9),
        )

    def forward(self, x):
        x = self.upsample(x)
        return self.residuals(x)

class SoundStreamDecoder(nn.Module):
    def __init__(self, base_channels=32, latent_dim=128):
        super().__init__()
        self.initial = nn.Conv1d(latent_dim, base_channels * 16, kernel_size=7, padding=3)

        self.blocks = nn.Sequential(
            DecoderBlock(base_channels * 16, stride=8),
            DecoderBlock(base_channels * 8, stride=5),
            DecoderBlock(base_channels * 4, stride=4),
            DecoderBlock(base_channels * 2, stride=2),
        )

        self.final = nn.Conv1d(base_channels, 1, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        return torch.tanh(self.final(x))

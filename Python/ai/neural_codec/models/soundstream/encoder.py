import torch.nn as nn

from .residuals import ResidualUnit1D

class EncoderBlock(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.residuals = nn.Sequential(
            ResidualUnit1D(channels // 2, dilation=1),
            ResidualUnit1D(channels // 2, dilation=3),
            ResidualUnit1D(channels // 2, dilation=9),
        )
        self.downsample = nn.Conv1d(
            channels // 2, channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2
        )

    def forward(self, x):
        x = self.residuals(x)
        return self.downsample(x)

class SoundStreamEncoder(nn.Module):
    def __init__(self, base_channels=32, latent_dim=128):
        super().__init__()
        self.initial = nn.Conv1d(1, base_channels, kernel_size=7, padding=3)

        self.blocks = nn.Sequential(
            EncoderBlock(base_channels * 2, stride=2),
            EncoderBlock(base_channels * 4, stride=4),
            EncoderBlock(base_channels * 8, stride=5),
            EncoderBlock(base_channels * 16, stride=8),
        )

        self.final = nn.Conv1d(
            base_channels * 16, latent_dim, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        return self.final(x)

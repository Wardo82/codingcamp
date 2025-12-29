import torch.nn as nn

from .residuals import ResidualUnit2D

class STFTDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit2D(1, base_channels, stride=(1, 2)),
            ResidualUnit2D(base_channels, base_channels * 2, stride=(2, 2)),
            ResidualUnit2D(base_channels * 2, base_channels * 4, stride=(1, 2)),
            ResidualUnit2D(base_channels * 4, base_channels * 8, stride=(2, 2)),
        )
        self.final = nn.Conv2d(base_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        x = self.layers(x)
        return self.final(x)

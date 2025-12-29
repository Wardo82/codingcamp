import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualUnit1D(nn.Module):

    def __init__(self, channels, dilation):
        """
        ResidualUnit1D (used in encoder/decoder)
        """
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=7,
            dilation=dilation,
            padding=(7 // 2) * dilation
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x + residual

class ResidualUnit2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        ResidualUnit2D (used in STFT discriminator)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )

        self.skip = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            stride=stride
        )

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x + residual

import torchaudio
import torch
import torch.nn as nn

def waveform_loss(x_hat, x):
    """
    Time-domain reconstruction loss (L1)

    Why L1:
    - Preserves transients
    - Less smoothing than L2
    - Standard for audio codecs
    """
    return torch.mean(torch.abs(x_hat - x))

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft_configs = [
            (1024, 256),
            (2048, 512),
            (512, 128),
        ]

    def stft(self, x, fft_size, hop_size):
        return torch.stft(
            x.squeeze(1),
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=fft_size,
            window=torch.hann_window(fft_size).to(x.device),
            return_complex=True,
        )

    def forward(self, x_hat, x):
        loss = 0.0
        for fft_size, hop_size in self.stft_configs:
            X = self.stft(x, fft_size, hop_size)
            X_hat = self.stft(x_hat, fft_size, hop_size)

            mag = torch.abs(X)
            mag_hat = torch.abs(X_hat)

            loss += torch.mean(torch.abs(mag - mag_hat))
            loss += torch.mean(torch.abs(torch.log(mag + 1e-7) - torch.log(mag_hat + 1e-7)))

        return loss

def discriminator_loss(d_real, d_fake):
    """
    Adversarial losses (hinge GAN, SoundStream-style)
    """
    loss_real = torch.mean(F.relu(1.0 - d_real))
    loss_fake = torch.mean(F.relu(1.0 + d_fake))
    return loss_real + loss_fake

def generator_adversarial_loss(d_fake):
    return -torch.mean(d_fake)

def feature_matching_loss(real_features, fake_features):
    loss = 0.0
    for rf, ff in zip(real_features, fake_features):
        loss += torch.mean(torch.abs(rf - ff))
    return loss

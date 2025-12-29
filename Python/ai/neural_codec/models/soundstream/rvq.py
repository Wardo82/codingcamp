import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_quantizers: int,
        codebook_size: int,
        dim: int,
        commitment_weight: float = 0.25,
    ):
        super().__init__()

        self.commitment_weight = commitment_weight

        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, dim)
            for _ in range(num_quantizers)
        ])

        for cb in self.codebooks:
            nn.init.uniform_(cb.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        """

        The VQ loss teaches the model how to discretize information without
        destroying gradients — without it, a neural codec cannot train.
        
        z: (B, T, D) or (B, D)
        """
        residual = z
        quantized = torch.zeros_like(z)

        all_indices = []
        vq_loss = 0.0

        for codebook in self.codebooks:
            # Compute distances
            distances = (
                residual.pow(2).sum(dim=-1, keepdim=True)
                - 2 * residual @ codebook.weight.t()
                + codebook.weight.pow(2).sum(dim=1)
            )

            indices = distances.argmin(dim=-1)
            q = codebook(indices)

            # --- losses ---
            codebook_loss = F.mse_loss(q, residual.detach())
            commit_loss = F.mse_loss(residual, q.detach())

            vq_loss = vq_loss + codebook_loss + self.commitment_weight * commit_loss

            # --- straight-through estimator ---
            q_st = residual + (q - residual).detach()

            quantized = quantized + q_st
            residual = residual - q_st

            all_indices.append(indices)

        return quantized, all_indices, vq_loss


if __name__ == '__main__':
    num_quantizers=8
    codebook_size=1024
    latent_dim=128
    quantizer = ResidualVectorQuantizer(
        num_quantizers, codebook_size, latent_dim
    )
    z = torch.randn(2, 100, 128)
    z_q, codes, vq_loss = quantizer(z)

    assert z_q.shape == z.shape
    assert vq_loss.requires_grad

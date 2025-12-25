import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, latent_dim=8, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, z):
        return self.fc(z)

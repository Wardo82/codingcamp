import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import models

filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

data = []
for filename in filenames:
    with gzip.open(filename, "rb") as f:
        if "labels" in filename:
            # Labels: 1D array, skip 8-byte header
            data.append(np.frombuffer(f.read(), dtype=np.uint8, offset=8))
        else:
            # Images: skip 16-byte header, reshape to (N, 784)
            data.append(
                np.frombuffer(f.read(), dtype=np.uint8, offset=16)
                .reshape(-1, 28 * 28)
            )

# Split into training and testing sets
X_train, y_train, X_test, y_test = data

# Normalize pixel values
X_train = X_train.astype(np.float32) / 255.0
# Ensure labels are integers
y_train = y_train.astype(np.int64)

# Convert the training data to PyTorch tensors
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

# Create the vae+classifier model and optimizer
class VAEWithClassifier(nn.Module):
    def __init__(self, latent_dim=8, num_classes=10):
        super().__init__()
        self.vae = models.VAE()
        self.classifier = models.Classifier(latent_dim, num_classes)

    def forward(self, x):
        encoded, decoded, mu, logvar = self.vae(x)
        logits = self.classifier(mu)
        return encoded, decoded, mu, logvar, logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAEWithClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5
)

# Define a loss function that combines binary cross-entropy and Kullback-Leibler divergence
def loss_function(recon_x, x, mu, logvar, logits, labels, beta=1.0):
    # Reconstruction: preserves pixel information
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    # KL: regularizes latent distribution
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Classification: forces class separation in latent space
    CE = F.cross_entropy(logits, labels, reduction="sum")
    return BCE + beta * KLD + CE

# Create a DataLoader to handle batching of the training data
batch_size = 128
num_epochs = 20
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Training loop
loss_history = []
for epoch in range(num_epochs):
    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        encoded, recon, mu, logvar, logits = model(x)
        # Compute the loss and perform backpropagation
        loss = loss_function(recon, x, mu, logvar, logits, y, beta=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the running loss
        total_loss += loss.item()

    # Print the epoch loss
    epoch_loss = total_loss / len(train_loader.dataset)
    #scheduler.step()
    loss_history.append(epoch_loss)
    print(
        "Epoch {}/{}: loss={:.4f}".format(epoch + 1, num_epochs, epoch_loss)
    )

# Save latent space explitcitly
test_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(X_test.astype(np.float32) / 255.0),
    torch.from_numpy(y_test.astype(np.int64))
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)
output_path = 'output/vae_classifier'
os.makedirs(output_path, exist_ok=True)
model.eval()
Z = []
Y = []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        encoded, recon, mu, logvar, logits = model(x)
        Z.append(mu.cpu())
        Y.append(y)

Z = torch.cat(Z).numpy()
Y = torch.cat(Y).numpy()
np.save(f"{output_path}/latent_train.npy", Z)
np.save(f"{output_path}/labels_train.npy", Y)

# model.save()

# Plot loss_history
plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("ELBO (BCE + KL + CE)")
plt.title("Training Loss")
plt.grid(True)
plt.show()

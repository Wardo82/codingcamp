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

# Create the autoencoder model and optimizer
model = models.VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5
)

# Define a loss function that combines binary cross-entropy and Kullback-Leibler divergence
def loss_function(recon_x, x, mu, logvar):
    # Compute the binary cross-entropy loss between the reconstructed output and the input data
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    # Compute the Kullback-Leibler divergence between the learned latent variable distribution and a standard Gaussian distribution
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Combine the two losses by adding them together and return the result
    return BCE + KLD

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create a DataLoader to handle batching of the training data
batch_size = 128
num_epochs = 20
train_loader = torch.utils.data.DataLoader(
    X_train, batch_size=batch_size, shuffle=True
)

# Training loop
loss_history = []
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        # Get a batch of training data and move it to the device
        data = data.to(device)

        # Forward pass
        encoded, decoded, mu, log_var = model(data)

        # Compute the loss and perform backpropagation
        loss = loss_function(decoded, data, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the running loss
        total_loss += loss.item() * data.size(0)

    # Print the epoch loss
    epoch_loss = total_loss / len(train_loader.dataset)
    #scheduler.step()
    loss_history.append(epoch_loss)
    print(
        "Epoch {}/{}: loss={:.4f}".format(epoch + 1, num_epochs, epoch_loss)
    )

# Save latent space explitcitly
output_path = 'output/vae'
os.makedirs(output_path, exist_ok=True)
model.eval()
with torch.no_grad():
    latent_train = []
    for batch in train_loader:
        batch = batch.to(device)
        encoded, decoded, mu, log_var = model(data)
        latent_train.append(mu.cpu())
latent_train = torch.cat(latent_train).numpy()
np.save(f"{output_path}/latent_train.npy", latent_train)
np.save(f"{output_path}/labels_train.npy", y_train)

model.save()

# Plot loss_history
plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()

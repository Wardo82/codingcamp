import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Load the latent representations and labels
latent_train = np.load("output/vae/latent_train.npy")
y_train = np.load("output/vae/labels_train.npy")

# Use t-SNE to reduce to 2D for visualization (if latent dim > 2)
if latent_train.shape[1] > 2:
    print(f"Reducing {latent_train.shape[1]}D latent space to 2D using t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0)
    # Use subset for faster computation
    n_samples = min(10000, len(latent_train))
    latent_2d = tsne.fit_transform(latent_train[:n_samples])
    labels_plot = y_train[:n_samples].astype(int)
else:
    # If already 2D, use directly
    latent_2d = latent_train
    labels_plot = y_train.astype(int)

# Create the plot with colored points by digit class
plt.figure(figsize=(8, 6))

# Use tab10 colormap for 10 distinct colors (one per digit)
colors = plt.cm.tab10(np.arange(10))

# Plot each digit class with its own color
for digit in range(10):
    mask = labels_plot == digit
    plt.scatter(
        latent_2d[mask, 0],
        latent_2d[mask, 1],
        c=[colors[digit]],
        label=str(digit),
        s=5,
        alpha=0.7
    )

plt.title("Encoded training data")
plt.xlabel("Latent dim 1")
plt.ylabel("Latent dim 2")
plt.legend(title="Digit", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

import gzip
import matplotlib.pyplot as plt
import numpy as np

def show_images(images, labels):
    """
    Display a set of images and their labels using matplotlib.
    """
    pixels = images.reshape(-1, 28, 28)

    fig, axs = plt.subplots(
        ncols=len(images), nrows=1, figsize=(10, 3)
    )

    # Handle case where only one image is shown
    if len(images) == 1:
        axs = [axs]

    for i in range(len(images)):
        axs[i].imshow(pixels[i], cmap="gray")
        axs[i].set_title(f"Label: {labels[i]}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel(f"Index: {i}")

    fig.subplots_adjust(hspace=0.5)
    plt.show()


# ---- SCRIPT ENTRY POINT ----
if __name__ == "__main__":
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

    # Example: show first 5 training images
    num_images = 5

    images_to_show = X_train[:num_images]
    labels_to_show = y_train[:num_images]

    show_images(images_to_show, labels_to_show)

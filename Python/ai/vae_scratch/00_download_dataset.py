import gzip
import numpy as np
import urllib.request

# Download the files
url = "https://raw.githubusercontent.com/fgnt/mnist/master/"
filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

data = []

for filename in filenames:
    print("Downloading", filename)
    urllib.request.urlretrieve(url + filename, filename)

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
X_test = X_test.astype(np.float32) / 255.0

# Ensure labels are integers
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

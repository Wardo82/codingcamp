import gzip
import numpy as np
import urllib.request

# Download the files
url = "https://openslr.trmal.net/resources/12/"
filenames = ['train-clean-100.tar.gz',
             'test-clean.tar.gz']

for filename in filenames:
    print("Downloading", filename)
    urllib.request.urlretrieve(url + filename, f"data/LibriSpeech/{filename}")

# SoundStream: End-to-End Neural Audio Coding

SoundStream is a fully end-to-end neural audio codec that can compress speech, music, and general audio at bitrates as low as 3 kbps—outperforming traditional codecs at much higher bitrates.

## Key Innovations

- End-to-End Training: The entire pipeline—encoder, quantizer, and decoder—is trained jointly, optimizing for both reconstruction accuracy and perceptual quality via adversarial losses.

- Residual Vector Quantization (RVQ): Instead of a single quantization step, SoundStream uses a multi-stage (residual) vector quantizer. This allows it to represent audio more efficiently and enables bitrate scalability.

- Bitrate Scalability: Thanks to a novel "quantizer dropout" during training, a single SoundStream model can operate at different bitrates (3–18 kbps) with minimal quality loss.

- Low Latency & Real-Time: The model is fully convolutional and causal, making it suitable for low-latency, real-time applications—even on a smartphone CPU.

- Joint Compression and Enhancement: SoundStream can simultaneously compress and enhance audio (e.g., denoise speech) with no extra latency.


## Architectural Design

The system uses a fully convolutional U-Net structure with strided convolutions for downsampling and transposed convolutions for upsampling. A residual vector quantizer (RVQ) between encoder and decoder discretizes the latent space while maintaining reconstruction fidelity. ***Crucially, SoundStream introduced structured dropout during training, enabling a single model to operate across multiple bitrates (3-18 kbps) without quality degradation.***

## Residual vector quantizer (RVQ)

Allows to store and transmit information efficiently. It has it roots in Vector Quantization, used to make models smaller (from float32 to int8 for example) and more "efficient", with some trade-offs.

In the Soundstream papaer, the Quantizer takes 75 vectors of audio embeddings with 128 elements where, during traninig, it learns to quantize it efficiently and pass it over to the decoder for reconstruction.

### Codebook quantization

Learn a set of possible vectors that cover the entire embedding space. The goal is the to assing each embedding to the closest codebook and store its index. This is not very precise as it introduces many errors during quantization.

Residual vector quantization consist of iteratively running this codebook assignment on the vector embeddings.

[TODO] Add picture and algorithm of RVQ
[TODO] Add unit tests

The number of iterations of the RVQ determines the bitrate. 
As an example, the model inputs 75 frames per second to the RVQ. The RVQ makes two iterations for an output of 2 indices. All in all, the RVQ has 1024 codebook entries, that means, it tries to learn a representation of the embeddings with 1024 fixed vectors that can be represented with 10 bits per entry. 
As a result, the model runs at 75 frames * 2 iterations * 10 bits = 1.5kbps

### Learning the codebook vectors

Learning the codebook can be done by a trained encoder and using K-Means to cluster the embeddings. The centroid of the cluster determines the point where the "minimum distance" will be computed for assigning a new embedding to a codebook.

[TODO] Study again the loss of this quantizer

- Codebook update: Exponential moving average with decay 0.99
- Commitment loss: Incentivices the encoder to encode into the codebooks of the RVQ. (Wouldn't this be counter productive? par de locos going in the wrong direction?)
- [TODO not sure how this worked] Random restarts: Helps have codewords that are actually used by the encoding process of the samples. Sometimes a codebook exists where no sample is mapped to. 


## Training Methodology

SoundStream combines adversarial training with multi-resolution spectral losses:

- A GAN discriminator distinguishes real/fake audio samples, forcing the decoder to generate perceptually convincing outputs.

- Multi-scale spectrogram losses ensure accurate frequency domain reconstruction.

- Feature matching losses align intermediate layer activations between original and reconstructed audio.

## Results

The results are impressive:

- At 3 kbps, SoundStream outperforms Opus at 12 kbps and approaches the quality of EVS at 9.6 kbps.

- It works for speech, music, and general audio—not just speech.

- Subjective tests (MUSHRA) show that listeners prefer SoundStream's output at low bitrates over traditional codecs.

## References

[1] SoundStream: An End-to-End Neural Audio Codec - Paper: https://arxiv.org/abs/2107.03312 | Video: https://www.youtube.com/watch?v=V4jj-yhiclk&ab_channel=RISEResearchInstitutesofSweden
[2]  Residual Vector Quantization for Audio and Speech Embeddings https://www.youtube.com/watch?v=Xt9S74BHsvc

## TODOs

- Add multi-scale STFT discriminators
- Add feature matching loss
- Add commitment + codebook losses
- Add bitrate control via number of quantizers
- Replace BCE/L2 with multi-resolution spectral loss

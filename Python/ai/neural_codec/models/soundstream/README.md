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


## TODOs

- Add multi-scale STFT discriminators
- Add feature matching loss
- Add commitment + codebook losses
- Add bitrate control via number of quantizers
- Replace BCE/L2 with multi-resolution spectral loss

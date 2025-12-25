## What a neural codec actually is (demystified)

A neural codec is **not** just an autoencoder.

A neural codec learns:

[
x ;\xrightarrow{\text{encoder}}; y ;\xrightarrow{\text{quantizer}}; \hat{y}
;\xrightarrow{\text{decoder}}; \hat{x}
]

With an explicit **rate–distortion tradeoff**:

[
\mathcal{L} = \text{Distortion} + \lambda \cdot \text{Rate}
]

Where:

* **Distortion** ≈ reconstruction loss
* **Rate** ≈ number of bits needed to encode latent variables

This is where your VAE knowledge directly applies.

---

## Direct mapping: VAE → Neural Codec

| What you learned    | Neural codec equivalent              |
| ------------------- | ------------------------------------ |
| Latent space z      | Compressed bitstream                 |
| KL divergence       | Rate (entropy)                       |
| Decoder likelihood  | Distortion                           |
| μ, σ                | Probability model for entropy coding |
| Smooth latent space | Compressible latent space            |

In fact:

> **VAEs are already probabilistic codecs without quantization**

---

## Next concrete steps (ordered, practical)

### Step 1 — Deterministic autoencoder + quantization

Replace VAE sampling with **quantization**:

```python
z_q = torch.round(z / Δ) * Δ
```

Key ideas:

* Quantization introduces information loss
* Forces robustness
* Makes latent variables discrete → encodable

Study:

* Uniform quantization
* Straight-through estimator (STE)

---

### Step 2 — Learnable entropy model

Replace KL with explicit entropy modeling:

[
\text{Rate} \approx -\log p(z_q)
]

Learn:

* Factorized Gaussian entropy models
* Hyperprior models (Ballé et al.)

This directly reuses:

* μ, σ modeling
* KL intuition
* Probabilistic decoding

---

### Step 3 — Rate–distortion optimization

Train with:

```python
loss = distortion + λ * rate
```

Where:

* λ controls compression strength
* Larger λ → fewer bits, worse quality

This is *exactly* what you already did with β-VAE.

---

### Step 4 — Domain jump (images → audio → video)

MNIST was geometry learning.
Neural codecs care about **structure**.

Next domains:

1. Natural images (CIFAR → ImageNet crops)
2. Audio waveforms (speech codecs)
3. Video (motion + residuals)

What transfers:

* Encoder/decoder design
* Latent regularization
* Visualization of embeddings

What changes:

* Convolutions
* Temporal structure
* Perceptual losses

---

## 5. Models you should study next (in this order)

### Must-read papers

1. **Ballé et al., 2018** — Learned Image Compression
2. **Minnen et al., 2018** — Joint autoregressive + hierarchical priors
3. **VQ-VAE** — Discrete latents
4. **SoundStream / EnCodec** — Neural audio codecs

---

## 6. What mental shift to make now

Stop thinking in terms of:

> “Latent space visualization”

Start thinking:

> “Information allocation under constraints”

Neural codecs are about:

* Where to spend bits
* What information is perceptually important
* What can be thrown away

Your classifier-regularized VAE already did this — just for *semantics* instead of *bitrate*.

---

## 7. Suggested next project (very concrete)

**Build a tiny image codec**:

1. Conv encoder → latent tensor
2. Quantize latents
3. Gaussian entropy model
4. Conv decoder
5. Plot:

   * Bits-per-pixel vs PSNR
   * Reconstructions at different λ

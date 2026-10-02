use candle_core::{Result, Tensor};
use candle_nn::{Linear, linear, ops};
use candle_nn::{Module, VarBuilder};

use crate::ssm::embeddings::Embedding;
use crate::ssm::sequenceblock::SequenceBlock;

pub enum Encoder {
    Dense(Linear),    // 1 -> d_model, for raw pixel/scalar input
    Embed(Embedding), // vocab -> d_model, for token input
}

impl Encoder {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Encoder::Dense(l) => l.forward(x),
            Encoder::Embed(e) => e.forward(x),
        }
    }
}

pub struct StackedModel {
    encoder: Encoder,
    decoder: Linear,
    layers: Vec<SequenceBlock>,
    classification: bool,
    embedding: bool,
    decode: bool,
}

impl StackedModel {
    pub fn new(
        vb: VarBuilder,
        d_output: usize,
        d_model: usize,
        n_layers: usize,
        n: usize,
        l_max: usize,
        dropout_p: f32,
        prenorm: bool,
        glu: bool,
        embedding: bool,
        classification: bool,
        decode: bool,
    ) -> Result<Self> {
        let encoder = if embedding {
            Encoder::Embed(Embedding::new(d_output, d_model, vb.pp("encoder"))?)
        } else {
            Encoder::Dense(linear(1, d_model, vb.pp("encoder"))?)
        };
        let decoder = linear(d_model, d_output, vb.pp("decoder"))?;

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(SequenceBlock::new(
                vb.pp(format!("layer{i}")),
                d_model,
                n,
                l_max,
                dropout_p,
                prenorm,
                glu,
                decode,
            )?);
        }
        Ok(Self {
            encoder,
            decoder,
            layers,
            classification,
            embedding,
            decode,
        })
    }

    pub fn reset_cache(&self) -> Result<()> {
        for l in &self.layers {
            l.reset_cache()?;
        }
        Ok(())
    }

    /// x: (L, 1). A single, unbatched sequence (see batching note below).
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let mut x = x.clone();

        if !self.classification {
            if !self.embedding {
                x = x.affine(1.0 / 255.0, 0.0)?; // normalize pixels
            }
            if !self.decode {
                // pad(x[:-1], [(1,0),(0,0)]): shift right, prepend zero
                let l = x.dim(0)?;
                let truncated = x.narrow(0, 0, l - 1)?;
                let mut pad_dims = truncated.dims().to_vec();
                pad_dims[0] = 1;
                let zero = Tensor::zeros(pad_dims, x.dtype(), x.device())?;
                x = Tensor::cat(&[&zero, &truncated], 0)?;
            }
        }

        let mut h = self.encoder.forward(&x)?; // (L, d_model)
        for layer in &self.layers {
            h = layer.forward(&h, train)?;
        }
        if self.classification {
            h = h.mean(0)?.unsqueeze(0)?; // (1, d_model)
        }
        let logits = self.decoder.forward(&h)?; // (L_or_1, d_output)
        ops::log_softmax(&logits, candle_core::D::Minus1)
    }
}

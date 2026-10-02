use candle_core::{DType, Result, Tensor};
use candle_nn::{Embedding as CandleEmbedding, Module, VarBuilder, embedding};

pub struct Embedding {
    inner: CandleEmbedding,
}

impl Embedding {
    pub fn new(num_embeddings: usize, features: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            inner: embedding(num_embeddings, features, vb)?,
        })
    }

    /// x: (L, 1), token ids (as floats or ints), with 0 meaning "padding".
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ids = x.squeeze(1)?.to_dtype(DType::U32)?; // x[..., 0], cast for lookup
        let y = self.inner.forward(&ids)?; // (L, features)
        let mask = x.gt(0.0)?.to_dtype(y.dtype())?; // (L, 1), 1.0 or 0.0
        y.broadcast_mul(&mask)
    }
}

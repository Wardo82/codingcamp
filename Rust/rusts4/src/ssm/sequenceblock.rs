use candle_core::{Result, Tensor};
use candle_nn::{Dropout, LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, ops};

use crate::ssm::layerstack::SsmLayerStack;

pub struct SequenceBlock {
    seq: SsmLayerStack,
    norm: LayerNorm,
    out: Linear,
    out2: Option<Linear>,
    drop: Dropout,
    prenorm: bool,
    glu: bool,
}

impl SequenceBlock {
    pub fn new(
        vb: VarBuilder,
        d_model: usize,
        n: usize,
        l_max: usize,
        dropout_p: f32,
        prenorm: bool,
        glu: bool,
        decode: bool,
    ) -> Result<Self> {
        let seq = SsmLayerStack::new(vb.pp("seq"), n, l_max, d_model, decode)?;
        let norm = layer_norm(d_model, 1e-5, vb.pp("norm"))?;
        let out = linear(d_model, d_model, vb.pp("out"))?;
        let out2 = if glu {
            Some(linear(d_model, d_model, vb.pp("out2"))?)
        } else {
            None
        };
        let drop = Dropout::new(dropout_p);
        Ok(Self {
            seq,
            norm,
            out,
            out2,
            drop,
            prenorm,
            glu,
        })
    }

    pub fn reset_cache(&self) -> Result<()> {
        self.seq.reset_cache()
    }

    /// x: (L, d_model) -> (L, d_model)
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let skip = x.clone();

        let mut h = if self.prenorm {
            self.norm.forward(x)?
        } else {
            x.clone()
        };
        h = self.seq.forward(&h)?;
        h = self.drop.forward(&h.gelu_erf()?, train)?;

        h = if self.glu {
            let a = self.out.forward(&h)?;
            let b = self.out2.as_ref().unwrap().forward(&h)?;
            (a * ops::sigmoid(&b)?)?
        } else {
            self.out.forward(&h)?
        };

        let mut y = (skip + self.drop.forward(&h, train)?)?;
        if !self.prenorm {
            y = self.norm.forward(&y)?;
        }
        Ok(y)
    }
}

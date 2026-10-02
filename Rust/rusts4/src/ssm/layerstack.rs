use crate::ssm::layer::SsmLayer; // from the previous step
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// H independent SsmLayer channels — the Candle equivalent of cloneLayer(SSMLayer).
pub struct SsmLayerStack {
    channels: Vec<SsmLayer>,
}

impl SsmLayerStack {
    pub fn new(vb: VarBuilder, n: usize, l_max: usize, h: usize, decode: bool) -> Result<Self> {
        let mut channels = Vec::with_capacity(h);
        for i in 0..h {
            // vb.pp(...) namespaces params per channel, so each gets its own
            // A/B/C/D/log_step and its own random init — this is the
            // split_rngs={"params": True} behavior.
            channels.push(SsmLayer::new(vb.pp(format!("ch{i}")), n, l_max, decode)?);
        }
        Ok(Self { channels })
    }

    pub fn reset_cache(&self) -> Result<()> {
        for c in &self.channels {
            c.reset_cache()?;
        }
        Ok(())
    }

    /// x: (L, H) -> (L, H)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.channels.len();
        let mut outs = Vec::with_capacity(h);
        for i in 0..h {
            let u_i = x.narrow(1, i, 1)?.squeeze(1)?; // (L,)
            outs.push(self.channels[i].forward(&u_i)?.unsqueeze(1)?); // (L,1)
        }
        Tensor::cat(&outs, 1) // (L, H)
    }
}

use crate::conv::{causal_convolution, k_conv};
use crate::linalg::inv;
use crate::scan::scan_ssm;
use crate::ssm::init::{lecun_normal, log_step_init};
use crate::ssm::ssm::{DiscreteSsm, discretize_t};
use candle_core::{Device, Result, Tensor};
use candle_nn::{Init, VarBuilder};
use std::cell::RefCell;

pub struct SsmLayer {
    a: Tensor,        // (N, N)
    b: Tensor,        // (N, 1)
    c: Tensor,        // (1, N)
    d: Tensor,        // (1,)
    log_step: Tensor, // (1,)
    l_max: usize,
    decode: bool,
    cache: RefCell<Tensor>, // (N, 1), RNN-mode state
}

impl SsmLayer {
    pub fn new(vb: VarBuilder, n: usize, l_max: usize, decode: bool) -> Result<Self> {
        let a = vb.get_with_hints((n, n), "A", lecun_normal(n))?;
        let b = vb.get_with_hints((n, 1), "B", lecun_normal(1))?;
        let c = vb.get_with_hints((1, n), "C", lecun_normal(n))?;
        let d = vb.get_with_hints(1, "D", Init::Const(1.0))?;
        let log_step = vb.get_with_hints(1, "log_step", log_step_init(0.001, 0.1))?;
        let cache = RefCell::new(Tensor::zeros((n, 1), a.dtype(), a.device())?);
        Ok(Self {
            a,
            b,
            c,
            d,
            log_step,
            l_max,
            decode,
            cache,
        })
    }

    /// Zero the RNN-mode cache, e.g. at the start of a new sequence.
    pub fn reset_cache(&self) -> Result<()> {
        let n = self.a.dim(0)?;
        *self.cache.borrow_mut() = Tensor::zeros((n, 1), self.a.dtype(), self.a.device())?;
        Ok(())
    }

    /// u: (L,). CNN mode requires L == l_max, matching the tutorial's fixed kernel length.
    pub fn forward(&self, u: &Tensor) -> Result<Tensor> {
        let step = self.log_step.exp()?; // Δ = exp(log_step)
        let (ab, bb, cb) = discretize_t(&self.a, &self.b, &self.c, &step)?;
        let du = self.d.broadcast_mul(u)?;

        if !self.decode {
            // CNN mode: one shot, all L outputs from one convolution
            let l = u.dim(0)?;
            if l != self.l_max {
                candle_core::bail!("SsmLayer: expected u of length {}, got {}", self.l_max, l);
            }
            let k = k_conv(&ab, &bb, &cb, l)?;
            let y = causal_convolution(u, &k)?;
            Ok((y + du)?)
        } else {
            // RNN mode: one step, state carried in `cache`
            let d = DiscreteSsm {
                a: ab,
                b: bb,
                c: cb,
            };
            let x0 = self.cache.borrow().clone();
            let (x_k, y) = scan_ssm(&d, &u.unsqueeze(1)?, &x0)?;
            *self.cache.borrow_mut() = x_k;
            Ok((y.flatten_all()? + du)?)
        }
    }
}

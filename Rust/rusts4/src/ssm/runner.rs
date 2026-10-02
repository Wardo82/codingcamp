use crate::scan::scan_ssm;
use crate::ssm::ssm::{Ssm, discretize};
use candle_core::{Result, Tensor};

pub fn run_ssm(a: &Tensor, b: &Tensor, c: &Tensor, u: &Tensor) -> Result<Tensor> {
    let l = u.dim(0)?; // sequence length, u: (L,)
    let n = a.dim(0)?; // state size

    let ssm = Ssm {
        a: a.clone(),
        b: b.clone(),
        c: c.clone(),
    };
    let d_ssm = discretize(&ssm, 1.0 / l as f64)?;

    // Run recurrence: u needs shape (L, 1), the state x0 needs shape (N, 1)
    let x0 = Tensor::zeros((n, 1), a.dtype(), a.device())?;
    let (_, ys) = scan_ssm(&d_ssm, &u.unsqueeze(1)?, &x0)?;
    Ok(ys) // (L, 1)
}

#[test]
fn run_ssm_output_shape() -> Result<()> {
    use crate::ssm::ssm::random_ssm;
    use rand::{SeedableRng, rngs::StdRng};

    let dev = candle_core::Device::Cpu;
    let s = random_ssm(&mut StdRng::seed_from_u64(1), 4, &dev)?;
    let u = Tensor::arange(0f32, 8., &dev)?;
    let y = run_ssm(&s.a, &s.b, &s.c, &u)?;
    assert_eq!(y.dims(), &[8, 1]);
    Ok(())
}

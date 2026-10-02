use crate::complex::Cx;
use crate::linalg::inv;
use candle_core::DType;
use candle_core::{Device, Result, Tensor};
use rand::{RngExt, rngs::StdRng};

/// Continuous-time SSM:  x'(t) = A x(t) + B u(t),  y(t) = C x(t)
pub struct Ssm {
    pub a: Tensor, // (N, N)
    pub b: Tensor, // (N, 1)
    pub c: Tensor, // (1, N)
}

/// Uniform [0, 1) matrix, like jax.random.uniform
fn uniform(rng: &mut StdRng, rows: usize, cols: usize, dev: &Device) -> Result<Tensor> {
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.random::<f32>()).collect();
    Tensor::from_vec(data, (rows, cols), dev)
}

pub fn random_ssm(rng: &mut StdRng, n: usize, dev: &Device) -> Result<Ssm> {
    Ok(Ssm {
        a: uniform(rng, n, n, dev)?,
        b: uniform(rng, n, 1, dev)?,
        c: uniform(rng, 1, n, dev)?,
    })
}

/// Discrete-time SSM:  x_k = Ab x_{k-1} + Bb u_k,  y_k = C x_k
pub struct DiscreteSsm {
    pub a: Tensor, // (N, N)
    pub b: Tensor, // (N, 1)
    pub c: Tensor, // (1, N)
}

/// Bilinear (Tustin) discretization, as in the tutorial.
pub fn discretize(s: &Ssm, step: f64) -> Result<DiscreteSsm> {
    let n = s.a.dim(0)?;
    let eye = Tensor::eye(n, s.a.dtype(), s.a.device())?;
    let half_a = s.a.affine(step / 2.0, 0.0)?; // Δ/2 · A

    let bl = inv(&(&eye - &half_a)?)?; // (I - Δ/2 A)^-1
    let ab = bl.matmul(&(&eye + &half_a)?)?;
    let bb = bl.matmul(&s.b)?.affine(step, 0.0)?; // (BL · Δ) @ B
    Ok(DiscreteSsm {
        a: ab,
        b: bb,
        c: s.c.clone(),
    })
}

/// Same as `discretize`, but `step` is a (1,) tensor so gradients flow
/// back to whatever produced it (e.g. log_step.exp()).
pub fn discretize_t(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    step: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let n = a.dim(0)?;
    let eye = Tensor::eye(n, a.dtype(), a.device())?;
    let half_step = step.affine(0.5, 0.0)?; // Δ/2, still a tensor
    let half_a = a.broadcast_mul(&half_step)?; // Δ/2 · A

    let bl = inv(&(&eye - &half_a)?)?;
    let ab = bl.matmul(&(&eye + &half_a)?)?;
    let bb = bl.matmul(b)?.broadcast_mul(step)?;
    Ok((ab, bb, c.clone()))
}

/// Diagonal continuous SSM: Λ, B, C are all (N,) complex.
pub struct DiagSsm {
    pub lambda: Cx,
    pub b: Cx,
    pub c: Cx,
}
pub struct DiagDiscrete {
    pub a: Cx,
    pub b: Cx,
    pub c: Cx,
}

#[derive(Clone, Copy, Debug)]
pub enum Method {
    Bilinear,
    Zoh,
}

pub fn discretize_diag(s: &DiagSsm, step: f64, method: Method) -> Result<DiagDiscrete> {
    let (a, b) = match method {
        Method::Bilinear => {
            let half = s.lambda.scale(step / 2.0)?;
            let denom = half.neg()?.add_real(1.0)?; // 1 - Δ/2 Λ
            let numer = half.add_real(1.0)?; // 1 + Δ/2 Λ
            (numer.div(&denom)?, s.b.scale(step)?.div(&denom)?)
        }
        Method::Zoh => {
            let a = s.lambda.scale(step)?.exp()?; // exp(ΔΛ)
            let b = a.add_real(-1.0)?.div(&s.lambda)?.mul(&s.b)?;
            (a, b)
        }
    };
    Ok(DiagDiscrete {
        a,
        b,
        c: s.c.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn random_ssm_has_expected_shapes() -> Result<()> {
        // equivalent of jax.random.PRNGKey(1)
        let mut rng = StdRng::seed_from_u64(1);
        let dev = Device::Cpu;

        let s = random_ssm(&mut rng, 4, &dev)?;
        assert_eq!(s.a.dims(), &[4, 4]);
        assert_eq!(s.b.dims(), &[4, 1]);
        assert_eq!(s.c.dims(), &[1, 4]);
        Ok(())
    }

    #[test]
    fn random_ssm_values_are_in_unit_interval() -> Result<()> {
        let mut rng = StdRng::seed_from_u64(1);
        let s = random_ssm(&mut rng, 4, &Device::Cpu)?;

        for t in [&s.a, &s.b, &s.c] {
            let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
            assert!(v.iter().all(|x| (0.0..1.0).contains(x)));
        }
        Ok(())
    }

    #[test]
    fn random_ssm_is_reproducible_for_same_seed() -> Result<()> {
        let dev = Device::Cpu;
        let s1 = random_ssm(&mut StdRng::seed_from_u64(1), 4, &dev)?;
        let s2 = random_ssm(&mut StdRng::seed_from_u64(1), 4, &dev)?;

        let a1: Vec<f32> = s1.a.flatten_all()?.to_vec1()?;
        let a2: Vec<f32> = s2.a.flatten_all()?.to_vec1()?;
        assert_eq!(a1, a2);
        Ok(())
    }

    #[test]
    fn random_ssm_differs_for_different_seeds() -> Result<()> {
        let dev = Device::Cpu;
        let s1 = random_ssm(&mut StdRng::seed_from_u64(1), 4, &dev)?;
        let s2 = random_ssm(&mut StdRng::seed_from_u64(2), 4, &dev)?;

        let a1: Vec<f32> = s1.a.flatten_all()?.to_vec1()?;
        let a2: Vec<f32> = s2.a.flatten_all()?.to_vec1()?;
        assert_ne!(a1, a2);
        Ok(())
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
    }

    #[test]
    fn dense_discretize_scalar_by_hand() -> Result<()> {
        // A=-1, B=1, C=1, Δ=1: BL = 1/1.5, Ab = (2/3)(0.5) = 1/3, Bb = 2/3
        let dev = Device::Cpu;
        let s = Ssm {
            a: Tensor::new(&[[-1f32]], &dev)?,
            b: Tensor::new(&[[1f32]], &dev)?,
            c: Tensor::new(&[[1f32]], &dev)?,
        };
        let d = discretize(&s, 1.0)?;
        assert!((d.a.get(0)?.get(0)?.to_scalar::<f32>()? - 1.0 / 3.0).abs() < 1e-6);
        assert!((d.b.get(0)?.get(0)?.to_scalar::<f32>()? - 2.0 / 3.0).abs() < 1e-6);
        Ok(())
    }

    fn real_diag(lam: &[f32], b: &[f32], dev: &Device) -> Result<DiagSsm> {
        let n = lam.len();
        let z = Tensor::zeros(n, DType::F32, dev)?;
        Ok(DiagSsm {
            lambda: Cx::new(Tensor::new(lam, dev)?, z.clone())?,
            b: Cx::new(Tensor::new(b, dev)?, z.clone())?,
            c: Cx::new(Tensor::ones(n, DType::F32, dev)?, z)?,
        })
    }

    #[test]
    fn zoh_real_scalar_by_hand() -> Result<()> {
        // Λ=-1, B=1, Δ=0.5: Ab = e^-0.5, Bb = 1 - e^-0.5
        let d = discretize_diag(&real_diag(&[-1.0], &[1.0], &Device::Cpu)?, 0.5, Method::Zoh)?;
        let a = d.a.re.to_vec1::<f32>()?[0];
        let b = d.b.re.to_vec1::<f32>()?[0];
        assert!((a - (-0.5f32).exp()).abs() < 1e-6);
        assert!((b - (1.0 - (-0.5f32).exp())).abs() < 1e-6);
        assert!(d.a.im.to_vec1::<f32>()?[0].abs() < 1e-7);
        Ok(())
    }

    #[test]
    fn diagonal_bilinear_matches_dense_bilinear() -> Result<()> {
        let dev = Device::Cpu;
        let dense = Ssm {
            a: Tensor::new(&[[-1f32, 0.], [0., -2.]], &dev)?,
            b: Tensor::new(&[[1f32], [3.]], &dev)?,
            c: Tensor::new(&[[1f32, 1.]], &dev)?,
        };
        let dd = discretize(&dense, 0.1)?;
        let dg = discretize_diag(
            &real_diag(&[-1., -2.], &[1., 3.], &dev)?,
            0.1,
            Method::Bilinear,
        )?;

        // dense Ab is diagonal here; compare its diagonal and Bb column
        let ab_diag = Tensor::new(
            &[
                dd.a.get(0)?.get(0)?.to_scalar::<f32>()?,
                dd.a.get(1)?.get(1)?.to_scalar::<f32>()?,
            ],
            &dev,
        )?;
        assert!(max_abs_diff(&ab_diag, &dg.a.re)? < 1e-6);
        assert!(max_abs_diff(&dd.b.flatten_all()?, &dg.b.re)? < 1e-6);
        Ok(())
    }
}

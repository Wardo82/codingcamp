use crate::complex::Cx;
use crate::ssm::ssm::{DiagDiscrete, DiscreteSsm};
use candle_core::{Result, Tensor};

/// Rust version of jax.lax.scan: thread a carry through `xs`, collect the outputs.
pub fn scan<C, X, Y>(
    mut f: impl FnMut(C, X) -> Result<(C, Y)>,
    init: C,
    xs: impl IntoIterator<Item = X>,
) -> Result<(C, Vec<Y>)> {
    let mut carry = init;
    let mut ys = Vec::new();
    for x in xs {
        let (c, y) = f(carry, x)?;
        carry = c;
        ys.push(y);
    }
    Ok((carry, ys))
}

/// Dense RNN mode, as in the tutorial.
/// u: (L, 1), x0: (N, 1)  ->  (final state (N, 1), ys (L, 1))
///
/// The state is a column (N, 1) instead of the tutorial's (N,), because
/// Candle's matmul needs 2-D operands.
pub fn scan_ssm(d: &DiscreteSsm, u: &Tensor, x0: &Tensor) -> Result<(Tensor, Tensor)> {
    let l = u.dim(0)?;
    let us = (0..l)
        .map(|k| u.narrow(0, k, 1))
        .collect::<Result<Vec<_>>>()?; // each (1,1)

    let step = |x: Tensor, u_k: Tensor| -> Result<(Tensor, Tensor)> {
        let x_k = (d.a.matmul(&x)? + d.b.matmul(&u_k)?)?; // Ab @ x + Bb @ u_k  -> (N,1)
        let y_k = d.c.matmul(&x_k)?; // Cb @ x_k          -> (1,1)
        Ok((x_k, y_k))
    };

    let (x_last, ys) = scan(step, x0.clone(), us)?;
    Ok((x_last, Tensor::cat(&ys, 0)?)) // (L, 1)
}

/// Diagonal (S4D) RNN mode. Everything is elementwise, so there is no matmul.
/// u: (L,), x0: (N,) complex  ->  (final state, ys (L,))
///
/// `conj_pairs`: S4D stores only half of each conjugate pair of eigenvalues,
/// so the output is 2*Re(C x). Pass `false` to get plain Re(C x)
/// (useful for checking against the dense path with a real diagonal).
pub fn scan_diag(d: &DiagDiscrete, u: &Tensor, x0: &Cx, conj_pairs: bool) -> Result<(Cx, Tensor)> {
    let l = u.dim(0)?;
    let us = (0..l).map(|k| u.get(k)).collect::<Result<Vec<_>>>()?; // scalars

    let step = |x: Cx, u_k: Tensor| -> Result<(Cx, Tensor)> {
        let ax = d.a.mul(&x)?;
        let re = (&ax.re + d.b.re.broadcast_mul(&u_k)?)?; // Ab x + Bb u_k
        let im = (&ax.im + d.b.im.broadcast_mul(&u_k)?)?;
        let mut y = (d.c.re.mul(&re)? - d.c.im.mul(&im)?)?.sum_all()?; // Re(C x)
        if conj_pairs {
            y = y.affine(2.0, 0.0)?;
        }
        Ok((Cx::new(re, im)?, y))
    };

    let (x_last, ys) = scan(step, x0.clone(), us)?;
    Ok((x_last, Tensor::stack(&ys, 0)?)) // (L,)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssm::ssm::{DiagSsm, Method, Ssm, discretize, discretize_diag, random_ssm};
    use candle_core::{DType, Device};
    use rand::{SeedableRng, rngs::StdRng};

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
    }

    #[test]
    fn scalar_recurrence_by_hand() -> Result<()> {
        // Ab=0.5, Bb=1, Cb=1, u=[1,1,1]: x = 1, 1.5, 1.75
        let dev = Device::Cpu;
        let d = DiscreteSsm {
            a: Tensor::new(&[[0.5f32]], &dev)?,
            b: Tensor::new(&[[1f32]], &dev)?,
            c: Tensor::new(&[[1f32]], &dev)?,
        };
        let u = Tensor::new(&[[1f32], [1.], [1.]], &dev)?;
        let x0 = Tensor::zeros((1, 1), DType::F32, &dev)?;
        let (x_last, ys) = scan_ssm(&d, &u, &x0)?;
        assert_eq!(ys.dims(), &[3, 1]);
        assert_eq!(ys.flatten_all()?.to_vec1::<f32>()?, vec![1.0, 1.5, 1.75]);
        assert_eq!(x_last.flatten_all()?.to_vec1::<f32>()?, vec![1.75]);
        Ok(())
    }

    #[test]
    fn carrying_state_across_chunks_equals_one_run() -> Result<()> {
        // This is the property that makes streaming inference on a device work.
        let dev = Device::Cpu;
        let s = random_ssm(&mut StdRng::seed_from_u64(1), 3, &dev)?;
        let d = discretize(&s, 0.1)?;
        let u = Tensor::arange(0f32, 6., &dev)?.reshape((6, 1))?;
        let x0 = Tensor::zeros((3, 1), DType::F32, &dev)?;

        let (x_full, y_full) = scan_ssm(&d, &u, &x0)?;

        let (x_mid, y1) = scan_ssm(&d, &u.narrow(0, 0, 3)?, &x0)?;
        let (x_end, y2) = scan_ssm(&d, &u.narrow(0, 3, 3)?, &x_mid)?;
        let y_chunked = Tensor::cat(&[&y1, &y2], 0)?;

        assert!(max_abs_diff(&y_full, &y_chunked)? < 1e-4);
        assert!(max_abs_diff(&x_full, &x_end)? < 1e-4);
        Ok(())
    }

    #[test]
    fn diagonal_scan_matches_dense_scan() -> Result<()> {
        let dev = Device::Cpu;
        let z = Tensor::zeros(2, DType::F32, &dev)?;
        let (lam, b) = ([-1f32, -2.], [1f32, 3.]);

        let dense = Ssm {
            a: Tensor::new(&[[-1f32, 0.], [0., -2.]], &dev)?,
            b: Tensor::new(&[[1f32], [3.]], &dev)?,
            c: Tensor::new(&[[1f32, 1.]], &dev)?,
        };
        let diag = DiagSsm {
            lambda: Cx::new(Tensor::new(&lam, &dev)?, z.clone())?,
            b: Cx::new(Tensor::new(&b, &dev)?, z.clone())?,
            c: Cx::new(Tensor::ones(2, DType::F32, &dev)?, z.clone())?,
        };

        let u = Tensor::new(&[0.3f32, -1.0, 0.5, 2.0, 0.0], &dev)?;
        let dd = discretize(&dense, 0.1)?;
        let dg = discretize_diag(&diag, 0.1, Method::Bilinear)?;

        let (_, y_dense) = scan_ssm(
            &dd,
            &u.unsqueeze(1)?,
            &Tensor::zeros((2, 1), DType::F32, &dev)?,
        )?;
        let x0 = Cx::new(z.clone(), z)?;
        let (_, y_diag) = scan_diag(&dg, &u, &x0, false)?;

        assert!(max_abs_diff(&y_dense.flatten_all()?, &y_diag)? < 1e-5);
        Ok(())
    }
}

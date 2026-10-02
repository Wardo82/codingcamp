use candle_core::{DType, Result, Tensor, bail};

/// Inverse of an n x n row-major matrix via Gauss-Jordan with partial pivoting.
pub fn inv_vec(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let w = 2 * n;
    // Augmented matrix [A | I], row-major, n rows x 2n cols
    let mut m = vec![0.0f64; n * w];
    for i in 0..n {
        for j in 0..n {
            m[i * w + j] = a[i * n + j];
        }
        m[i * w + n + i] = 1.0;
    }

    for col in 0..n {
        // 1. Partial pivoting: largest |entry| at or below the diagonal
        let mut p = col;
        for r in col + 1..n {
            if m[r * w + col].abs() > m[p * w + col].abs() {
                p = r;
            }
        }
        if m[p * w + col].abs() < 1e-12 {
            bail!("inv: matrix is singular (or nearly so) at column {col}");
        }
        if p != col {
            for j in 0..w {
                m.swap(p * w + j, col * w + j);
            }
        }

        // 2. Normalize the pivot row so the pivot becomes 1
        let piv = m[col * w + col];
        for j in 0..w {
            m[col * w + j] /= piv;
        }

        // 3. Eliminate this column from every other row
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * w + col];
            if f != 0.0 {
                for j in 0..w {
                    m[r * w + j] -= f * m[col * w + j];
                }
            }
        }
    }

    // Right half is the inverse
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = m[i * w + n + j];
        }
    }
    Ok(out)
}

/// Tensor wrapper: (n, n) -> (n, n), same device as the input.
pub fn inv(a: &Tensor) -> Result<Tensor> {
    let (n, m) = a.dims2()?;
    if n != m {
        bail!("inv: expected a square matrix, got {n}x{m}");
    }
    let data: Vec<f64> = a
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f64::from)
        .collect();
    let out: Vec<f32> = inv_vec(&data, n)?.into_iter().map(|x| x as f32).collect();
    Tensor::from_vec(out, (n, n), a.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
    }

    #[test]
    fn inverse_of_known_2x2() -> Result<()> {
        let dev = Device::Cpu;
        let a = Tensor::new(&[[4f32, 7.], [2., 6.]], &dev)?;
        let expected = Tensor::new(&[[0.6f32, -0.7], [-0.2, 0.4]], &dev)?;
        assert!(max_abs_diff(&inv(&a)?, &expected)? < 1e-5);
        Ok(())
    }

    #[test]
    fn pivoting_handles_zero_on_diagonal() -> Result<()> {
        let dev = Device::Cpu;
        let a = Tensor::new(&[[0f32, 1.], [1., 0.]], &dev)?;
        assert!(max_abs_diff(&inv(&a)?, &a)? < 1e-6); // its own inverse
        Ok(())
    }

    #[test]
    fn a_times_inv_a_is_identity() -> Result<()> {
        use crate::ssm::ssm::random_ssm;
        use rand::{SeedableRng, rngs::StdRng};
        let dev = Device::Cpu;
        let a = random_ssm(&mut StdRng::seed_from_u64(1), 6, &dev)?.a;
        let i = Tensor::eye(6, DType::F32, &dev)?;
        assert!(max_abs_diff(&a.matmul(&inv(&a)?)?, &i)? < 1e-3);
        Ok(())
    }

    #[test]
    fn singular_matrix_is_an_error() {
        let a = Tensor::new(&[[1f32, 2.], [2., 4.]], &Device::Cpu).unwrap();
        assert!(inv(&a).is_err());
    }
}

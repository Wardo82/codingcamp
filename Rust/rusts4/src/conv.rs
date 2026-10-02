use candle_core::{Result, Tensor};

/// K = (C Bb, C Ab Bb, C Ab^2 Bb, ..., C Ab^(L-1) Bb), shape (L,)
pub fn k_conv(ab: &Tensor, bb: &Tensor, cb: &Tensor, l: usize) -> Result<Tensor> {
    let n = ab.dim(0)?;
    let mut power = Tensor::eye(n, ab.dtype(), ab.device())?; // Ab^0 = I
    let mut ks = Vec::with_capacity(l);
    for _ in 0..l {
        let k_i = cb.matmul(&power)?.matmul(bb)?; // (1,N)@(N,N)@(N,1) -> (1,1)
        ks.push(k_i.reshape(1)?);
        power = ab.matmul(&power)?; // Ab^(i+1)
    }
    Tensor::cat(&ks, 0) // (L,)
}

/// Causal convolution, equivalent to convolve(u, K, mode="full")[:len(u)].
/// u and K must have the same length L.
pub fn causal_convolution(u: &Tensor, k: &Tensor) -> Result<Tensor> {
    let l = u.dim(0)?;
    if k.dim(0)? != l {
        candle_core::bail!("causal_convolution: u and K must have equal length");
    }
    let uv: Vec<f32> = u.to_vec1()?;
    let kv: Vec<f32> = k.to_vec1()?;

    let mut y = vec![0f32; l];
    for n in 0..l {
        let mut s = 0f32;
        for i in 0..=n {
            s += uv[n - i] * kv[i];
        }
        y[n] = s;
    }
    Tensor::from_vec(y, l, u.device())
}

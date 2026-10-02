use candle_core::{DType, Device, Result, Tensor};

#[test]
fn conv_mode_matches_rnn_mode() -> Result<()> {
    use rand::{SeedableRng, rngs::StdRng};
    use rusts4::conv::*;
    use rusts4::ssm::runner::run_ssm;
    use rusts4::ssm::ssm::{discretize, random_ssm};

    let dev = Device::Cpu;
    let l = 16;
    let u = Tensor::rand(0f32, 1f32, l, &dev)?;
    let s = random_ssm(&mut StdRng::seed_from_u64(1), 4, &dev)?;
    // Approximation of y(t)
    // RNN mode
    let ys = run_ssm(&s.a, &s.b, &s.c, &u)?;

    // Convolution mode
    let d = discretize(&s, 1.0 / l as f64)?;
    let k = k_conv(&d.a, &d.b, &d.c, l)?;
    let y_conv = causal_convolution(&u, &k)?;

    let diff = (ys.flatten_all()? - &y_conv)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-3, "RNN and conv modes disagree: max diff {diff}");
    Ok(())
}

///
/// This test checks something different and more basic: does data actually flow correctly through the full pipeline
/// — encoder → N stacked SSM blocks → decoder → log-softmax — without a shape mismatch, a NaN, or a numerical blow-up?
///
/// It is not a check that the model would learn anything useful.
/// `A` is still randomly initialized (we haven't built HiPPO yet), so there's no claim here that this SSM is a good one — only that the architecture is wired together correctly.
///

#[cfg(test)]
mod stacked_model_tests {
    use rusts4::ssm::stackedmodel::StackedModel;

    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn stacked_model_runs_end_to_end() -> Result<()> {
        // Device. We stay on Device::Cpu — no code here depends on CUDA/Metal, and CPU is what the test runners (and ultimately the phone) will actually use.
        let dev = Device::Cpu;

        // VarMap is created empty; it will be populated the first time each
        // parameter is requested below, inside StackedModel::new.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        // Hyperparameters:
        // Stating each one explicitly rather than letting defaults hide a decision, since the point of this test is to understand what's actually being configured.
        let n = 4; // SSM state size N: dimension of the hidden state x_k inside each channel's recurrence
        let d_model = 8; // width of the residual stream between blocks (H in "H independent SSM channels")
        let d_output = 10; // number of output classes, e.g. digits 0-9
        let n_layers = 2; // how many SequenceBlocks are stacked
        let l_max = 16; // fixed sequence length the convolution kernel K is built for
        let dropout_p = 0.1;
        let prenorm = true;
        let glu = true;
        let embedding = false; // false => encoder is nn.Dense(1 -> d_model), input is raw scalars (like pixel intensities)
        let classification = false; // false => output is per-timestep logits, not a single pooled prediction
        let decode = false; // false => CNN/convolution mode, not the streaming RNN mode

        let model = StackedModel::new(
            vb,
            d_output,
            d_model,
            n_layers,
            n,
            l_max,
            dropout_p,
            prenorm,
            glu,
            embedding,
            classification,
            decode,
        )?;

        // Raw "pixel-like" values in [0, 255), matching what the model's
        // `x / 255.0` normalization branch expects as input.
        let x = Tensor::rand(0f32, 255f32, (l_max, 1), &dev)?;

        // train=false: dropout is disabled, so the only randomness left is
        // the model's parameters themselves (fixed once, at construction).
        let out = model.forward(&x, false)?;

        // Shape check:
        // classification=false and decode=false => the decoder is applied to
        // every timestep's hidden vector, not just a pooled summary. So the
        // output must have one row of logits per input timestep.
        assert_eq!(out.dims(), &[l_max, d_output]);

        // Numerical sanity:
        // It's a log-probability check: exp(log_softmax(row)) must sum to 1
        // for each row, by definition of softmax. This doesn't test the SSM
        // math at all; it only confirms the decoder's output was finite and
        // correctly normalized, i.e. nothing upstream produced NaN/Inf that
        // silently corrupted the softmax.
        let row_sums = out.exp()?.sum(1)?.to_vec1::<f32>()?;
        for (i, &s) in row_sums.iter().enumerate() {
            assert!((s - 1.0).abs() < 1e-4, "row {i} sums to {s}, expected ~1.0");
        }

        // Explicit finiteness check
        let vals = out.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "forward pass produced a non-finite value (NaN or Inf)"
        );

        Ok(())
    }
}

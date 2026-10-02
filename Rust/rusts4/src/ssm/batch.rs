pub struct BatchStackedModel {
    model: StackedModel,
}

impl BatchStackedModel {
    pub fn new(model: StackedModel) -> Self {
        Self { model }
    }

    /// xs: B sequences, each (L, 1). Params are shared (one `self.model`);
    /// dropout is independent per call; cache is isolated by resetting it
    /// before each sequence.
    pub fn forward_batch(&self, xs: &[Tensor], train: bool) -> Result<Tensor> {
        let mut ys = Vec::with_capacity(xs.len());
        for x in xs {
            if self.model.decode {
                self.model.reset_cache()?;
            }
            ys.push(self.model.forward(x, train)?.unsqueeze(0)?);
        }
        Tensor::cat(&ys, 0) // (B, L_or_1, d_output)
    }
}

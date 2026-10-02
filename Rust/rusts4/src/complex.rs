use candle_core::{Result, Tensor};

#[derive(Clone)]
pub struct Cx {
    pub re: Tensor,
    pub im: Tensor,
}

impl Cx {
    pub fn new(re: Tensor, im: Tensor) -> Result<Self> {
        if re.dims() != im.dims() {
            candle_core::bail!(
                "Cx: re/im shape mismatch {:?} vs {:?}",
                re.dims(),
                im.dims()
            );
        }
        Ok(Self { re, im })
    }
    pub fn scale(&self, s: f64) -> Result<Cx> {
        Cx::new(self.re.affine(s, 0.0)?, self.im.affine(s, 0.0)?)
    }
    pub fn neg(&self) -> Result<Cx> {
        Cx::new(self.re.neg()?, self.im.neg()?)
    }
    /// z + r for a real scalar r
    pub fn add_real(&self, r: f64) -> Result<Cx> {
        Cx::new(self.re.affine(1.0, r)?, self.im.clone())
    }
    pub fn mul(&self, o: &Cx) -> Result<Cx> {
        Cx::new(
            (self.re.broadcast_mul(&o.re)? - self.im.broadcast_mul(&o.im)?)?,
            (self.re.broadcast_mul(&o.im)? + self.im.broadcast_mul(&o.re)?)?,
        )
    }
    /// (a+ib)/(c+id) = ((ac+bd) + i(bc-ad)) / (c²+d²)
    pub fn div(&self, o: &Cx) -> Result<Cx> {
        let d = (o.re.sqr()? + o.im.sqr()?)?;
        let re = (self.re.broadcast_mul(&o.re)? + self.im.broadcast_mul(&o.im)?)?;
        let im = (self.im.broadcast_mul(&o.re)? - self.re.broadcast_mul(&o.im)?)?;
        Cx::new(re.broadcast_div(&d)?, im.broadcast_div(&d)?)
    }
    /// exp(a+ib) = e^a (cos b + i sin b)
    pub fn exp(&self) -> Result<Cx> {
        let m = self.re.exp()?;
        Cx::new(m.mul(&self.im.cos()?)?, m.mul(&self.im.sin()?)?)
    }
}

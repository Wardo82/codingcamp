use candle_nn::Init;

pub fn log_step_init(dt_min: f64, dt_max: f64) -> Init {
    Init::Uniform {
        lo: dt_min.ln(),
        up: dt_max.ln(),
    }
}

pub fn lecun_normal(fan_in: usize) -> Init {
    Init::Randn {
        mean: 0.0,
        stdev: (1.0 / fan_in as f64).sqrt(),
    }
}

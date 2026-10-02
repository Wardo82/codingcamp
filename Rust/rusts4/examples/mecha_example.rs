use candle_core::{Device, Tensor};
use plotters::prelude::*;
use rusts4::ssm::runner::run_ssm;

/// Spring (k), damper (b), mass (m) as a continuous-time SSM.
fn example_mass(
    k: f32,
    b: f32,
    m: f32,
    dev: &Device,
) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    let a = Tensor::new(&[[0f32, 1.0], [-k / m, -b / m]], dev)?;
    let bm = Tensor::new(&[[0f32], [1.0 / m]], dev)?;
    let c = Tensor::new(&[[1f32, 0.0]], dev)?;
    Ok((a, bm, c))
}

/// Intermittent push: sin(10 t), kept only where it is above 0.5.
fn example_force(t: f32) -> f32 {
    let x = (10.0 * t).sin();
    if x > 0.5 { x } else { 0.0 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let (a, b, c) = example_mass(40.0, 0.0, 1.0, &dev)?;

    // L samples of u(t) on [0, 1)
    let l = 100usize;
    let step = 1.0 / l as f32;
    let ts: Vec<f32> = (0..l).map(|k| k as f32 * step).collect();
    let us: Vec<f32> = ts.iter().map(|&t| example_force(t)).collect();
    let u = Tensor::from_vec(us.clone(), l, &dev)?;
    println!("wrote mecha_example.svg");

    // Approximation of y(t)
    let ys: Vec<f32> = run_ssm(&a, &b, &c, &u)?.flatten_all()?.to_vec1()?;

    // Plot force and position
    let root = SVGBackend::new("mecha_example.svg", (900, 500)).into_drawing_area();
    root.fill(&WHITE)?;
    let (y_min, y_max) = ys
        .iter()
        .chain(us.iter())
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let pad = 0.1 * (y_max - y_min).max(1e-6);

    let mut chart = ChartBuilder::on(&root)
        .caption("Spring-mass-damper via SSM (RNN mode)", ("sans-serif", 22))
        .margin(10)
        .x_label_area_size(35)
        .y_label_area_size(50)
        .build_cartesian_2d(0f32..1f32, (y_min - pad)..(y_max + pad))?;
    chart.configure_mesh().x_desc("t").draw()?;

    chart
        .draw_series(LineSeries::new(
            ts.iter().copied().zip(us.iter().copied()),
            &RED,
        ))?
        .label("force u(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
    chart
        .draw_series(LineSeries::new(
            ts.iter().copied().zip(ys.iter().copied()),
            &BLUE,
        ))?
        .label("position y(t)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

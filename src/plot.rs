use plotters::prelude::*;

use crate::constants::FloatPrecision;

pub fn plot(title: &str, path: &str, xs: &Vec<FloatPrecision>, ys: &Vec<FloatPrecision>, res: (u32, u32), xdims: (f64, f64), ydims: (f64, f64)) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, res).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(xdims.0..xdims.1, ydims.0..ydims.1)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            (0..xs.len()).map(|i| (xs[i], ys[i])),
            &RED,
        ))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

pub fn plot2(title: &str, path: &str, xs0: &Vec<FloatPrecision>, ys0: &Vec<FloatPrecision>, xs: &Vec<FloatPrecision>, ys: &Vec<FloatPrecision>, res: (u32, u32), xdims: (f64, f64), ydims: (f64, f64)) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, res).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(xdims.0..xdims.1, ydims.0..ydims.1)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            (0..xs0.len()).map(|i| (xs0[i], ys0[i])),
            &RED,
        ))?;
    chart
        .draw_series(LineSeries::new(
            (0..xs.len()).map(|i| (xs[i], ys[i])),
            &BLUE,
        ))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

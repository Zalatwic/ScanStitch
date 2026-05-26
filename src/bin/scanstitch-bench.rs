use clap::Parser;
use ndarray::{s, Array3};
use scanstitch::{base_detect, border, colorspace, density, ica, stitch, tonemap};
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "scanstitch-bench",
    about = "Run lightweight local ScanStitch phase benchmarks"
)]
struct BenchCli {
    /// Synthetic image width.
    #[arg(long, default_value_t = 640)]
    width: usize,

    /// Synthetic image height.
    #[arg(long, default_value_t = 360)]
    height: usize,

    /// Benchmark iterations per phase.
    #[arg(long, default_value_t = 3)]
    iterations: usize,

    /// Include FastICA in the synthetic benchmark.
    #[arg(long, default_value_t = false)]
    include_ica: bool,

    /// Benchmark the local LOGAN stitch path if the source TIFFs exist.
    #[arg(long, default_value_t = false)]
    real_logan: bool,

    /// Override LOGAN component 1 path.
    #[arg(long, default_value = "LOGAN043.tif")]
    component1: PathBuf,

    /// Override LOGAN component 2 path.
    #[arg(long, default_value = "LOGAN044.tif")]
    component2: PathBuf,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    width: usize,
    height: usize,
    iterations: usize,
    phases: Vec<BenchPhase>,
}

#[derive(Debug, Serialize)]
struct BenchPhase {
    name: String,
    iterations: usize,
    total_ms: u128,
    avg_ms: f64,
    notes: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = BenchCli::parse();
    let iterations = cli.iterations.max(1);
    let source = synthetic_negative(cli.height, cli.width);
    let base = base_detect::detect_film_base(&source).base_color;
    let density_result = density::phase3_invert_with_diagnostics(&source, &base, 14);
    let transmittance = density::density_image_to_normalized_transmittance(
        &density_result.positive_density,
        density_result.diagnostics.shared_robust_d_max,
    );
    let prophoto = colorspace::map_to_prophoto_d50(&transmittance);
    let tone_params = tonemap::fit_tone_params(&prophoto);
    let (comp1, comp2) = synthetic_split_pair(cli.height, cli.width.max(360), 96);

    let mut phases = Vec::new();
    phases.push(bench_phase("border_removal", iterations, || {
        let _ = border::remove_borders_with_diagnostics(&source, 2);
    }));
    phases.push(bench_phase("base_detect", iterations, || {
        let _ = base_detect::detect_film_base(&source);
    }));
    phases.push(bench_phase("stitch_translation", iterations, || {
        let _ = stitch::stitch_components(&comp1, &comp2, &stitch::StitchConfig::default());
    }));
    phases.push(bench_phase("density_inversion", iterations, || {
        let _ = density::phase3_invert_with_diagnostics(&source, &base, 14);
    }));
    if cli.include_ica {
        phases.push(bench_phase("fastica", iterations, || {
            let _ = ica::run_fastica(&density_result.positive_density, 30, 1e-4);
        }));
    }
    phases.push(bench_phase("colorspace_mapping", iterations, || {
        let _ = colorspace::map_to_prophoto_d50_with_diagnostics(&transmittance);
    }));
    phases.push(bench_phase("tone_mapping", iterations, || {
        let _ = tonemap::apply_tonemap_with_params_and_diagnostics(&prophoto, &tone_params);
    }));
    phases.push(bench_phase("grain_diagnostics", iterations, || {
        let rendered = tonemap::apply_tonemap_with_params(&prophoto, &tone_params);
        let _ = tonemap::render_grain_diagnostics(&rendered);
    }));

    if cli.real_logan {
        phases.push(bench_real_logan(&cli)?);
    }

    let report = BenchReport {
        width: cli.width,
        height: cli.height,
        iterations,
        phases,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn bench_phase(name: &str, iterations: usize, mut f: impl FnMut()) -> BenchPhase {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let total_ms = start.elapsed().as_millis();
    BenchPhase {
        name: name.to_string(),
        iterations,
        total_ms,
        avg_ms: total_ms as f64 / iterations as f64,
        notes: Vec::new(),
    }
}

fn bench_real_logan(cli: &BenchCli) -> Result<BenchPhase, Box<dyn std::error::Error>> {
    if !cli.component1.exists() || !cli.component2.exists() {
        return Ok(BenchPhase {
            name: "real_logan_stitch".to_string(),
            iterations: 0,
            total_ms: 0,
            avg_ms: 0.0,
            notes: vec!["LOGAN component files were not found; skipped".to_string()],
        });
    }

    let start = Instant::now();
    let loaded1 = scanstitch::tiff_io::load_tiff_u16(&cli.component1, 14)?;
    let loaded2 = scanstitch::tiff_io::load_tiff_u16(&cli.component2, 14)?;
    let border1 = border::remove_borders_with_diagnostics(&loaded1.image, 2);
    let border2 = border::remove_borders_with_diagnostics(&loaded2.image, 2);
    let result = stitch::stitch_components(
        &border1.cropped,
        &border2.cropped,
        &stitch::StitchConfig::default(),
    );
    let total_ms = start.elapsed().as_millis();
    Ok(BenchPhase {
        name: "real_logan_stitch".to_string(),
        iterations: 1,
        total_ms,
        avg_ms: total_ms as f64,
        notes: vec![format!(
            "decision={} confidence={:.3}",
            result
                .report
                .metrics
                .get("decision")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(""),
            result.report.confidence
        )],
    })
}

fn synthetic_negative(height: usize, width: usize) -> Array3<u16> {
    let mut arr = Array3::<u16>::zeros((height, width, 3));
    let rebate = (width / 12).clamp(8, 48);
    for y in 0..height {
        for x in 0..width {
            let is_rebate = x < rebate || x >= width.saturating_sub(rebate);
            let texture = ((x * 37 + y * 53 + (x * y) % 97) % 1200) as u16;
            let rgb = if is_rebate {
                [12000, 7200, 3200]
            } else {
                [
                    5200u16.saturating_add(texture),
                    4200u16.saturating_add(texture / 2),
                    3600u16.saturating_add(texture / 3),
                ]
            };
            for c in 0..3 {
                arr[[y, x, c]] = rgb[c];
            }
        }
    }
    arr
}

fn synthetic_split_pair(
    height: usize,
    total_width: usize,
    overlap: usize,
) -> (Array3<u16>, Array3<u16>) {
    let full = synthetic_negative(height, total_width);
    let split_point = total_width / 2 + overlap / 2;
    let comp1 = full.slice(s![.., 0..split_point, ..]).to_owned();
    let comp2 = full
        .slice(s![.., (split_point - overlap)..total_width, ..])
        .to_owned();
    (comp1, comp2)
}

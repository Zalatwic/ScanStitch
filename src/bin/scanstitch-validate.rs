use clap::Parser;
use scanstitch::cli::Cli as PipelineCli;
use scanstitch::report::PipelineReport;
use scanstitch::validation::{summarize_report, summary_to_markdown};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "scanstitch-validate",
    about = "Run or summarize real-image validation fixtures"
)]
struct ValidationCli {
    /// Fixture name to record in the summary. `logan` maps to LOGAN043/LOGAN044 by default.
    #[arg(long, default_value = "logan")]
    fixture: String,

    /// First component TIFF. Required unless --report is used or --fixture logan files exist.
    #[arg(long)]
    component1: Option<PathBuf>,

    /// Second component TIFF. Required unless --report is used or --fixture logan files exist.
    #[arg(long)]
    component2: Option<PathBuf>,

    /// Summarize an existing report instead of running the pipeline.
    #[arg(long)]
    report: Option<PathBuf>,

    /// Output directory for pipeline renders and validation summaries.
    #[arg(long, default_value = "output/validation/logan")]
    output_dir: PathBuf,

    /// Path for the compact JSON summary.
    #[arg(long)]
    summary_json: Option<PathBuf>,

    /// Path for the compact Markdown summary.
    #[arg(long)]
    summary_md: Option<PathBuf>,

    /// Enable pipeline debug artifacts for local inspection.
    #[arg(long, default_value_t = false)]
    debug: bool,

    /// Force stitching even if classification is uncertain.
    #[arg(long, default_value_t = false)]
    force_stitch: bool,

    /// Force no-stitch mode.
    #[arg(long, default_value_t = false)]
    force_no_stitch: bool,

    /// Transform mode passed to the pipeline: auto / translation / affine / homography.
    #[arg(long, default_value = "auto")]
    transform: String,

    /// Maximum iterations for ICA convergence.
    #[arg(long, default_value_t = 100)]
    ica_max_iter: usize,

    /// ICA convergence threshold.
    #[arg(long, default_value_t = 1e-5)]
    ica_tol: f64,

    /// Input bit depth for the pipeline.
    #[arg(long, default_value_t = 14)]
    bit_depth: u8,

    /// Request the OpenCV backend if this binary was built with it.
    #[arg(long, default_value_t = false)]
    use_opencv: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = ValidationCli::parse();

    let report = if let Some(report_path) = &cli.report {
        let contents = std::fs::read_to_string(report_path)?;
        serde_json::from_str::<PipelineReport>(&contents)?
    } else {
        let (component1, component2) = resolve_components(&cli)?;
        let pipeline_cli = PipelineCli {
            component1,
            component2,
            output_dir: cli.output_dir.clone(),
            debug: cli.debug,
            force_stitch: cli.force_stitch,
            force_no_stitch: cli.force_no_stitch,
            transform: cli.transform.clone(),
            ica_max_iter: cli.ica_max_iter,
            ica_tol: cli.ica_tol,
            bit_depth: cli.bit_depth,
            use_opencv: cli.use_opencv,
        };
        let report = scanstitch::pipeline::run(&pipeline_cli)?;
        report.save(&pipeline_cli.output_dir.join("report.json"))?;
        report
    };

    let summary = summarize_report(&cli.fixture, &report);
    let json_path = cli
        .summary_json
        .clone()
        .unwrap_or_else(|| cli.output_dir.join("summary.json"));
    let md_path = cli
        .summary_md
        .clone()
        .unwrap_or_else(|| cli.output_dir.join("summary.md"));

    write_text(
        &json_path,
        &serde_json::to_string_pretty(&summary).expect("summary should serialize"),
    )?;
    write_text(&md_path, &summary_to_markdown(&summary))?;

    println!("summary_json={}", json_path.display());
    println!("summary_md={}", md_path.display());
    println!(
        "decision={} confidence={:.3} mapping_strategy={} exposure_scale={:.3}",
        summary.stitch.decision.as_deref().unwrap_or(""),
        summary.stitch.confidence.unwrap_or(0.0),
        summary.colorspace.mapping_strategy.as_deref().unwrap_or(""),
        summary.colorspace.exposure_scale.unwrap_or(0.0)
    );

    Ok(())
}

fn resolve_components(
    cli: &ValidationCli,
) -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
    if let (Some(component1), Some(component2)) = (&cli.component1, &cli.component2) {
        return Ok((component1.clone(), component2.clone()));
    }

    if cli.component1.is_some() || cli.component2.is_some() {
        return Err(
            "both --component1 and --component2 are required when overriding fixture paths".into(),
        );
    }

    let pair = match cli.fixture.as_str() {
        "logan" => Some((PathBuf::from("LOGAN043.tif"), PathBuf::from("LOGAN044.tif"))),
        _ => None,
    };

    let Some((component1, component2)) = pair else {
        return Err(format!(
            "unknown fixture `{}`; pass --component1 and --component2 or use --report",
            cli.fixture
        )
        .into());
    };

    if !component1.exists() || !component2.exists() {
        return Err(format!(
            "fixture `{}` expects {} and {}; pass --component1/--component2 or summarize an existing --report",
            cli.fixture,
            component1.display(),
            component2.display()
        )
        .into());
    }

    Ok((component1, component2))
}

fn write_text(path: &PathBuf, contents: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, contents)?;
    Ok(())
}

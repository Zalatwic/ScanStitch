use clap::Parser;
use std::path::PathBuf;

/// 35mm color negative film scan reconstruction pipeline.
#[derive(Parser, Debug, Clone)]
#[command(name = "scanstitch", version, about)]
pub struct Cli {
    /// Path to the first component TIFF scan.
    pub component1: PathBuf,

    /// Path to the second component TIFF scan.
    pub component2: PathBuf,

    /// Output directory for results.
    #[arg(short = 'o', long, default_value = "output")]
    pub output_dir: PathBuf,

    /// Enable debug output (intermediate images, verbose logging).
    #[arg(long, default_value_t = false)]
    pub debug: bool,

    /// Force stitching even if overlap detection suggests single-frame.
    #[arg(long, default_value_t = false)]
    pub force_stitch: bool,

    /// Force no-stitch mode (treat inputs as independent frames).
    #[arg(long, default_value_t = false)]
    pub force_no_stitch: bool,

    /// Transform mode: "auto", "translation", "affine", or "homography".
    #[arg(long, default_value = "auto")]
    pub transform: String,

    /// Maximum iterations for ICA convergence.
    #[arg(long, default_value_t = 100)]
    pub ica_max_iter: usize,

    /// Tolerance for ICA convergence.
    #[arg(long, default_value_t = 1e-5)]
    pub ica_tol: f64,

    /// Output bit depth (14 or 16).
    #[arg(long, default_value_t = 14)]
    pub bit_depth: u8,

    /// Use OpenCV backend for feature matching (requires use-opencv feature).
    #[arg(long, default_value_t = false)]
    pub use_opencv: bool,
}

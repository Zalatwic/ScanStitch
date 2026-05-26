use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use crate::colorspace::ColorMode;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum RenderInputMode {
    /// Choose between ICA-separated and direct-density render inputs automatically.
    Auto,
    /// Use ICA-separated density channels for the render input.
    Ica,
    /// Use the film-base-corrected direct density image for the render input.
    DirectDensity,
}

impl RenderInputMode {
    pub fn as_str(self) -> &'static str {
        match self {
            RenderInputMode::Auto => "auto",
            RenderInputMode::Ica => "ica",
            RenderInputMode::DirectDensity => "direct-density",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum InputMode {
    /// Default negative-film scan workflow with density inversion.
    Negative,
    /// Already-positive RGB scan workflow for slides or software-inverted negatives.
    #[value(alias = "slide", alias = "positive-slide")]
    Positive,
}

impl InputMode {
    pub fn as_str(self) -> &'static str {
        match self {
            InputMode::Negative => "negative",
            InputMode::Positive => "positive",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum RenderIntent {
    /// Deterministic finished render with clean tone placement, bounded vibrance, and chroma noise cleanup.
    ModernClean,
    /// Neutral finished render with conservative saturation enrichment and normal cleanup.
    NaturalNeutral,
    /// Conservative render that preserves film texture and avoids final creative cleanup passes.
    FilmFaithful,
}

impl RenderIntent {
    pub fn as_str(self) -> &'static str {
        match self {
            RenderIntent::ModernClean => "modern-clean",
            RenderIntent::NaturalNeutral => "natural-neutral",
            RenderIntent::FilmFaithful => "film-faithful",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum QualityMode {
    /// Full deterministic pipeline with scene-referred master and review proof artifacts.
    Perfect,
    /// Normal deterministic pipeline with production diagnostics and optional debug artifacts.
    Balanced,
    /// Prefer lower-latency output while preserving deterministic behavior.
    Fast,
}

impl QualityMode {
    pub fn as_str(self) -> &'static str {
        match self {
            QualityMode::Perfect => "perfect",
            QualityMode::Balanced => "balanced",
            QualityMode::Fast => "fast",
        }
    }
}

/// 35mm color negative film scan reconstruction pipeline.
#[derive(Parser, Debug, Clone)]
#[command(version, about)]
pub struct Cli {
    /// Path to the first component TIFF scan.
    pub component1: PathBuf,

    /// Path to the second component TIFF scan.
    pub component2: PathBuf,

    /// Output directory for results.
    #[arg(short = 'o', long, default_value = "output")]
    pub output_dir: PathBuf,

    /// Optional scanner/film calibration profile JSON.
    #[arg(long)]
    pub calibration_profile: Option<PathBuf>,

    /// Optional local calibration library directory with scanner, roll, and film-hint JSON records.
    #[arg(long)]
    pub calibration_library: Option<PathBuf>,

    /// Scanner/settings profile ID to select from the calibration library.
    #[arg(long)]
    pub scanner_profile: Option<String>,

    /// Roll profile ID to select from the calibration library.
    #[arg(long)]
    pub roll_profile: Option<String>,

    /// Film stock label used to rank calibration evidence and reject mismatched roll profiles.
    #[arg(long)]
    pub film_stock: Option<String>,

    /// Override density film-base RGB as comma-separated scanner sample values.
    #[arg(long, value_name = "R,G,B")]
    pub base_color: Option<String>,

    /// Internal provenance label for non-manual base-color overrides.
    #[arg(skip)]
    pub base_color_source: Option<String>,

    /// Internal confidence for non-manual base-color overrides.
    #[arg(skip)]
    pub base_color_confidence: Option<f64>,

    /// Internal diagnostic reason for non-manual base-color overrides.
    #[arg(skip)]
    pub base_color_reason: Option<String>,

    /// Color mapping mode: auto, calibrated, image-derived, or neutral.
    #[arg(long, value_enum, default_value = "auto")]
    pub color_mode: ColorMode,

    /// Render input selection: auto, ica, or direct-density.
    #[arg(long, value_enum, default_value = "auto")]
    pub render_input: RenderInputMode,

    /// Input scan mode: negative film or already-positive RGB.
    #[arg(long, value_enum, default_value = "negative")]
    pub input_mode: InputMode,

    /// Finished-render intent.
    #[arg(long, value_enum, default_value = "modern-clean")]
    pub render_intent: RenderIntent,

    /// Quality/throughput mode. Perfect writes archival master and review proof artifacts by default.
    #[arg(long, value_enum, default_value = "perfect")]
    pub quality_mode: QualityMode,

    /// Write the scene-referred 32-bit float ProPhoto master. Enabled automatically in perfect mode.
    #[arg(long, default_value_t = false)]
    pub write_master: bool,

    /// Apply saved guided-review render decisions from a sidecar JSON.
    #[arg(long)]
    pub review_sidecar: Option<PathBuf>,

    /// Write guided-review decisions to a sidecar JSON after rendering/saving.
    #[arg(long)]
    pub write_review_sidecar: Option<PathBuf>,

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

impl Cli {
    pub fn validate(&self) -> Result<(), String> {
        if self.input_mode == InputMode::Positive && self.render_input == RenderInputMode::Ica {
            return Err("--input-mode positive cannot be used with --render-input ica because ICA requires density-inverted negative-film data".to_string());
        }
        Ok(())
    }

    pub fn should_write_master(&self) -> bool {
        self.write_master || self.quality_mode == QualityMode::Perfect
    }

    pub fn should_write_review_proof(&self) -> bool {
        self.quality_mode == QualityMode::Perfect
    }
}

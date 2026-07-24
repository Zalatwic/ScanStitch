use clap::{Args, Parser, ValueEnum};
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
    /// Deterministic finished render with clean tone placement and bounded vibrance.
    ModernClean,
    /// Neutral finished render with conservative saturation enrichment.
    NaturalNeutral,
    /// Conservative render that avoids final creative enhancement passes.
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum GrainReductionMode {
    /// Preserve the rendered film texture without a grain-reduction pass.
    Off,
    /// Apply edge- and detail-aware film-grain reduction.
    On,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum TechnicalWhiteBalanceMode {
    /// Estimate a scene illuminant only when neutral evidence is spatially and tonally robust.
    Auto,
    /// Use the explicitly supplied technical temperature and tint as the source illuminant.
    Manual,
    /// Preserve the color reconstruction result without a separate illuminant adaptation.
    Off,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum DeskewMode {
    /// Correct a single scan only when independent opposing or cross-axis border evidence agrees.
    Auto,
    /// Correct the explicitly declared signed source-skew angle.
    Manual,
    /// Preserve decoded geometry without absolute deskew.
    Off,
}

/// Explicit orthogonal correction applied after EXIF/DNG orientation has been materialized.
/// This is intentionally relative to metadata rather than a replacement for it, so valid source
/// orientation is always honored first.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum OrientationCorrection {
    /// Do not apply an additional semantic-orientation correction.
    None,
    FlipHorizontal,
    #[value(name = "rotate-180")]
    Rotate180,
    FlipVertical,
    #[value(name = "rotate-90-clockwise-then-flip-horizontal")]
    Rotate90ClockwiseThenFlipHorizontal,
    #[value(name = "rotate-90-clockwise")]
    Rotate90Clockwise,
    #[value(name = "rotate-270-clockwise-then-flip-horizontal")]
    Rotate270ClockwiseThenFlipHorizontal,
    #[value(name = "rotate-270-clockwise")]
    Rotate270Clockwise,
}

impl OrientationCorrection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::FlipHorizontal => "flip-horizontal",
            Self::Rotate180 => "rotate-180",
            Self::FlipVertical => "flip-vertical",
            Self::Rotate90ClockwiseThenFlipHorizontal => "rotate-90-clockwise-then-flip-horizontal",
            Self::Rotate90Clockwise => "rotate-90-clockwise",
            Self::Rotate270ClockwiseThenFlipHorizontal => {
                "rotate-270-clockwise-then-flip-horizontal"
            }
            Self::Rotate270Clockwise => "rotate-270-clockwise",
        }
    }

    /// EXIF orientation transform equivalent to this metadata-relative correction.
    pub fn exif_tag(self) -> u16 {
        match self {
            Self::None => 1,
            Self::FlipHorizontal => 2,
            Self::Rotate180 => 3,
            Self::FlipVertical => 4,
            Self::Rotate90ClockwiseThenFlipHorizontal => 5,
            Self::Rotate90Clockwise => 6,
            Self::Rotate270ClockwiseThenFlipHorizontal => 7,
            Self::Rotate270Clockwise => 8,
        }
    }
}

impl DeskewMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Manual => "manual",
            Self::Off => "off",
        }
    }
}

/// Absolute single-scan geometry correction controls.
#[derive(Args, Copy, Clone, Debug, PartialEq)]
pub struct GeometryArgs {
    /// Orthogonal semantic-orientation correction, applied after source orientation metadata.
    #[arg(long, value_enum, default_value = "none")]
    pub orientation_correction: OrientationCorrection,

    /// Absolute deskew policy. Auto is evidence-gated and currently applies to single inputs.
    #[arg(long, value_enum, default_value = "auto")]
    pub deskew: DeskewMode,

    /// Signed clockwise source skew in degrees for manual deskew; correction uses the opposite sign.
    #[arg(
        long,
        default_value_t = 0.0,
        value_name = "-3..3",
        allow_hyphen_values = true
    )]
    pub deskew_angle_degrees: f64,
}

impl Default for GeometryArgs {
    fn default() -> Self {
        Self {
            orientation_correction: OrientationCorrection::None,
            deskew: DeskewMode::Auto,
            deskew_angle_degrees: 0.0,
        }
    }
}

impl GeometryArgs {
    pub fn validate(self) -> Result<(), String> {
        if !self.deskew_angle_degrees.is_finite()
            || !(-3.0..=3.0).contains(&self.deskew_angle_degrees)
        {
            return Err("--deskew-angle-degrees must be finite and between -3 and 3".to_string());
        }
        Ok(())
    }
}

impl TechnicalWhiteBalanceMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Manual => "manual",
            Self::Off => "off",
        }
    }
}

/// Technical illuminant reconstruction and independent creative temperature/tint controls.
#[derive(Args, Copy, Clone, Debug, PartialEq)]
pub struct WhiteBalanceArgs {
    /// Technical white-balance policy applied in scene-referred linear ProPhoto RGB.
    #[arg(long, value_enum, default_value = "auto")]
    pub technical_white_balance: TechnicalWhiteBalanceMode,

    /// Source-illuminant CCT for manual technical white balance.
    #[arg(long, default_value_t = 5003.0, value_name = "2000..25000")]
    pub technical_temperature_kelvin: f64,

    /// Manual technical green-to-magenta illuminant offset; positive values remove green.
    #[arg(
        long,
        default_value_t = 0.0,
        value_name = "-1..1",
        allow_hyphen_values = true
    )]
    pub technical_tint: f64,

    /// Independent creative cool-to-warm adjustment; zero leaves the technical master unchanged.
    #[arg(
        long,
        default_value_t = 0.0,
        value_name = "-1..1",
        allow_hyphen_values = true
    )]
    pub creative_temperature: f64,

    /// Independent creative green-to-magenta adjustment; zero is neutral.
    #[arg(
        long,
        default_value_t = 0.0,
        value_name = "-1..1",
        allow_hyphen_values = true
    )]
    pub creative_tint: f64,
}

impl Default for WhiteBalanceArgs {
    fn default() -> Self {
        Self {
            technical_white_balance: TechnicalWhiteBalanceMode::Auto,
            technical_temperature_kelvin: 5003.0,
            technical_tint: 0.0,
            creative_temperature: 0.0,
            creative_tint: 0.0,
        }
    }
}

impl WhiteBalanceArgs {
    pub fn validate(self) -> Result<(), String> {
        if !self.technical_temperature_kelvin.is_finite()
            || !(2000.0..=25_000.0).contains(&self.technical_temperature_kelvin)
        {
            return Err(
                "--technical-temperature-kelvin must be finite and between 2000 and 25000"
                    .to_string(),
            );
        }
        if !self.technical_tint.is_finite() || !(-1.0..=1.0).contains(&self.technical_tint) {
            return Err("--technical-tint must be finite and between -1 and 1".to_string());
        }
        if !self.creative_temperature.is_finite()
            || !(-1.0..=1.0).contains(&self.creative_temperature)
        {
            return Err("--creative-temperature must be finite and between -1 and 1".to_string());
        }
        if !self.creative_tint.is_finite() || !(-1.0..=1.0).contains(&self.creative_tint) {
            return Err("--creative-tint must be finite and between -1 and 1".to_string());
        }
        Ok(())
    }
}

impl GrainReductionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
        }
    }

    pub fn enabled(self) -> bool {
        self == Self::On
    }
}

/// Independent film-grain reduction controls shared by batch, validation, and review UIs.
#[derive(Args, Copy, Clone, Debug, PartialEq)]
pub struct GrainReductionArgs {
    /// Optional film-grain reduction. Off by default so render intent never silently smooths film.
    #[arg(long, value_enum, default_value = "off")]
    pub grain_reduction: GrainReductionMode,

    /// Grain-reduction strength from 0 (no effect) through 1 (maximum bounded effect).
    #[arg(long, default_value_t = 0.5, value_name = "0..1")]
    pub grain_strength: f64,

    /// Grain spatial scale multiplier; 1 targets the native default grain radius.
    #[arg(long, default_value_t = 1.0, value_name = "0.5..4")]
    pub grain_scale: f64,
}

impl Default for GrainReductionArgs {
    fn default() -> Self {
        Self {
            grain_reduction: GrainReductionMode::Off,
            grain_strength: 0.5,
            grain_scale: 1.0,
        }
    }
}

/// 35mm color negative film scan reconstruction pipeline.
#[derive(Parser, Debug, Clone)]
#[command(version, about)]
pub struct Cli {
    /// One or more input film scans. Overlapping components are ordered and stitched
    /// automatically; a single input is processed directly without a dummy companion.
    #[arg(value_name = "INPUT", num_args = 1.., required = true)]
    pub inputs: Vec<PathBuf>,

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

    #[command(flatten)]
    pub white_balance: WhiteBalanceArgs,

    #[command(flatten)]
    pub geometry: GeometryArgs,

    #[command(flatten)]
    pub grain: GrainReductionArgs,

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

    /// Preserve artifacts, then require reviewable evidence and independently intact deliverables.
    #[arg(long, default_value_t = false)]
    pub require_reviewable: bool,
}

impl Cli {
    pub fn validate(&self) -> Result<(), String> {
        if self.inputs.is_empty() {
            return Err("at least one input scan is required".to_string());
        }
        if self.input_mode == InputMode::Positive && self.render_input == RenderInputMode::Ica {
            return Err("--input-mode positive cannot be used with --render-input ica because ICA requires density-inverted negative-film data".to_string());
        }
        if !self.grain.grain_strength.is_finite()
            || !(0.0..=1.0).contains(&self.grain.grain_strength)
        {
            return Err("--grain-strength must be finite and between 0 and 1".to_string());
        }
        if !self.grain.grain_scale.is_finite() || !(0.5..=4.0).contains(&self.grain.grain_scale) {
            return Err("--grain-scale must be finite and between 0.5 and 4".to_string());
        }
        self.geometry.validate()?;
        self.white_balance.validate()?;
        Ok(())
    }

    pub fn should_write_master(&self) -> bool {
        self.write_master || self.quality_mode == QualityMode::Perfect
    }

    pub fn should_write_review_proof(&self) -> bool {
        self.quality_mode == QualityMode::Perfect
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_single_input_without_dummy_component() {
        let cli = Cli::try_parse_from(["scanstitch", "single.tiff", "-o", "result"])
            .expect("single positional input");
        assert_eq!(cli.inputs, vec![PathBuf::from("single.tiff")]);
        assert_eq!(cli.output_dir, PathBuf::from("result"));
        assert!(!cli.require_reviewable);
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn parses_fail_closed_final_render_review_gate() {
        let cli = Cli::try_parse_from(["scanstitch", "single.tiff", "--require-reviewable"])
            .expect("reviewability gate");

        assert!(cli.require_reviewable);
    }

    #[test]
    fn preserves_all_positional_inputs_in_caller_order() {
        let cli = Cli::try_parse_from([
            "scanstitch",
            "right.tiff",
            "left.tiff",
            "middle.tiff",
            "--quality-mode",
            "fast",
        ])
        .expect("three positional inputs");
        assert_eq!(
            cli.inputs,
            vec![
                PathBuf::from("right.tiff"),
                PathBuf::from("left.tiff"),
                PathBuf::from("middle.tiff")
            ]
        );
    }

    #[test]
    fn programmatic_empty_input_list_is_rejected() {
        let mut cli = Cli::try_parse_from(["scanstitch", "single.tiff"]).unwrap();
        cli.inputs.clear();
        assert_eq!(
            cli.validate().unwrap_err(),
            "at least one input scan is required"
        );
    }

    #[test]
    fn grain_reduction_is_independent_and_disabled_by_default() {
        let default = Cli::try_parse_from(["scanstitch", "single.tiff"]).unwrap();
        assert_eq!(default.grain, GrainReductionArgs::default());

        let explicit = Cli::try_parse_from([
            "scanstitch",
            "single.tiff",
            "--render-intent",
            "film-faithful",
            "--grain-reduction",
            "on",
            "--grain-strength",
            "0.7",
            "--grain-scale",
            "1.5",
        ])
        .unwrap();
        assert_eq!(explicit.render_intent, RenderIntent::FilmFaithful);
        assert_eq!(explicit.grain.grain_reduction, GrainReductionMode::On);
        assert_eq!(explicit.grain.grain_strength, 0.7);
        assert_eq!(explicit.grain.grain_scale, 1.5);
        assert!(explicit.validate().is_ok());
    }

    #[test]
    fn invalid_grain_controls_are_rejected() {
        let mut cli = Cli::try_parse_from(["scanstitch", "single.tiff"]).unwrap();
        cli.grain.grain_strength = 1.01;
        assert!(cli.validate().unwrap_err().contains("grain-strength"));
        cli.grain.grain_strength = 0.5;
        cli.grain.grain_scale = 0.49;
        assert!(cli.validate().unwrap_err().contains("grain-scale"));
    }

    #[test]
    fn technical_and_creative_white_balance_controls_are_independent() {
        let default = Cli::try_parse_from(["scanstitch", "single.tiff"]).unwrap();
        assert_eq!(default.white_balance, WhiteBalanceArgs::default());

        let explicit = Cli::try_parse_from([
            "scanstitch",
            "single.tiff",
            "--technical-white-balance",
            "manual",
            "--technical-temperature-kelvin",
            "6500",
            "--technical-tint",
            "0.15",
            "--creative-temperature",
            "0.35",
            "--creative-tint",
            "-0.2",
        ])
        .unwrap();
        assert_eq!(
            explicit.white_balance.technical_white_balance,
            TechnicalWhiteBalanceMode::Manual
        );
        assert_eq!(explicit.white_balance.technical_temperature_kelvin, 6500.0);
        assert_eq!(explicit.white_balance.technical_tint, 0.15);
        assert_eq!(explicit.white_balance.creative_temperature, 0.35);
        assert_eq!(explicit.white_balance.creative_tint, -0.2);
        assert!(explicit.validate().is_ok());
    }

    #[test]
    fn invalid_white_balance_controls_are_rejected() {
        let mut cli = Cli::try_parse_from(["scanstitch", "single.tiff"]).unwrap();
        cli.white_balance.technical_temperature_kelvin = 1900.0;
        assert!(cli
            .validate()
            .unwrap_err()
            .contains("technical-temperature-kelvin"));
        cli.white_balance.technical_temperature_kelvin = 5003.0;
        cli.white_balance.creative_tint = 1.01;
        assert!(cli.validate().unwrap_err().contains("creative-tint"));
    }

    #[test]
    fn deskew_controls_are_bounded_and_signed() {
        let default = Cli::try_parse_from(["scanstitch", "single.tiff"]).unwrap();
        assert_eq!(default.geometry, GeometryArgs::default());

        let manual = Cli::try_parse_from([
            "scanstitch",
            "single.tiff",
            "--deskew",
            "manual",
            "--deskew-angle-degrees",
            "-1.25",
        ])
        .unwrap();
        assert_eq!(manual.geometry.deskew, DeskewMode::Manual);
        assert_eq!(manual.geometry.deskew_angle_degrees, -1.25);
        assert!(manual.validate().is_ok());

        let mut invalid = default;
        invalid.geometry.deskew_angle_degrees = 3.01;
        assert!(invalid
            .validate()
            .unwrap_err()
            .contains("deskew-angle-degrees"));
    }

    #[test]
    fn orientation_correction_is_explicit_and_metadata_relative() {
        let default = Cli::try_parse_from(["scanstitch", "single.tiff"]).unwrap();
        assert_eq!(
            default.geometry.orientation_correction,
            OrientationCorrection::None
        );

        let corrected = Cli::try_parse_from([
            "scanstitch",
            "upside-down.tiff",
            "--orientation-correction",
            "rotate-180",
        ])
        .unwrap();
        assert_eq!(
            corrected.geometry.orientation_correction,
            OrientationCorrection::Rotate180
        );
        assert_eq!(corrected.geometry.orientation_correction.exif_tag(), 3);
    }
}

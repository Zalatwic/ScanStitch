use clap::Parser;
use image::{Rgb, RgbImage};
use ndarray::Array3;
use scanstitch::cli::{
    Cli as PipelineCli, DeskewMode, GeometryArgs, GrainReductionArgs, GrainReductionMode,
    InputMode, OrientationCorrection, QualityMode, RenderInputMode, RenderIntent, WhiteBalanceArgs,
};
use scanstitch::colorspace::ColorMode;
use scanstitch::report::PipelineReport;
use scanstitch::tonemap::{GRAIN_DETAIL_MIN_PROBE_COUNT, GRAIN_DETAIL_P10_RETENTION_MIN};
use scanstitch::validation::{
    compare_render_summaries, compare_summary_baseline, delivery_artifact_integrity_issues,
    final_delivery_evidence_is_reviewable, run_synthetic_color_suite, summarize_report_with_source,
    summary_to_markdown, synthetic_color_suite_to_markdown, tracked_baseline_from_summary,
    BorderCropComponentValidationSummary, ColorCandidateAcceptanceSummary,
    InputOrientationComponentValidationSummary, TrackedValidationBaseline, ValidationSummary,
};
use scanstitch::{atomic_file, base_detect, border, positive_input, render_review, tiff_io};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const ROLL_BASE_PREPASS_MAX_DIMENSION: usize = 1600;
const ROLL_CONTACT_SHEET_THUMB_WIDTH: u32 = 180;
const ROLL_CONTACT_SHEET_THUMB_HEIGHT: u32 = 120;
const ROLL_CONTACT_SHEET_COLUMNS: usize = 6;
const ROLL_CONTACT_SHEET_GUTTER: u32 = 8;
const ORIENTATION_REVIEW_PREVIEW_MAX_WIDTH: u32 = 1200;
const ORIENTATION_REVIEW_PREVIEW_MAX_HEIGHT: u32 = 1200;
const STITCH_NORMALIZATION_MIN_DETAIL_SCALE_COUNT: usize = 2;
const STITCH_NORMALIZATION_MAX_DETAIL_ENERGY_RATIO: f64 = 2.0;
const STITCH_NORMALIZATION_MAX_GRADIENT_RATIO: f64 = 1.05;
const STITCH_NORMALIZATION_MAX_OFFSET_RATIO: f64 = 0.05;
const GEOMETRY_PREPARATION_MIN_DESKEW_RETAINED_AREA_RATIO: f64 = 0.90;
const GEOMETRY_PREPARATION_MIN_BORDER_RETAINED_AREA_RATIO: f64 = 0.50;
const GEOMETRY_PREPARATION_MIN_REMOVED_EDGES_PER_COMPONENT: usize = 4;
const GEOMETRY_ACCURACY_MAX_DESKEW_TOLERANCE_DEGREES: f64 = 0.05;
const GEOMETRY_ACCURACY_MAX_CROP_TOLERANCE_PX: usize = 4;
const NEGATIVE_RECONSTRUCTION_MIN_BASE_CONFIDENCE: f64 = 0.30;
const NEGATIVE_RECONSTRUCTION_MIN_MEASURED_CONFIDENCE: f64 = 0.75;
const NEGATIVE_RECONSTRUCTION_MAX_HELD_OUT_DELTA_E00_RMS: f64 = 6.0;
const NEGATIVE_RECONSTRUCTION_MAX_HELD_OUT_DELTA_E00_MAX: f64 = 15.0;
const NEGATIVE_RECONSTRUCTION_MIN_BASELINE_IMPROVEMENT: f64 = 0.25;
const NEGATIVE_RECONSTRUCTION_MAX_DENSITY_NOISE_GAIN: f64 = 4.0;
const NEGATIVE_RECONSTRUCTION_MAX_EXTRAPOLATION_RATIO: f64 = 0.01;

#[derive(Parser, Debug)]
#[command(
    name = "scanstitch-validate",
    about = "Run or summarize real-image validation fixtures"
)]
struct ValidationCli {
    /// Fixture name to record in the summary. `logan` maps to LOGAN043/LOGAN044 by default.
    #[arg(long, default_value = "logan")]
    fixture: String,

    /// JSON fixture registry with component paths and optional default output directories.
    #[arg(long)]
    fixture_registry: Option<PathBuf>,

    /// List available fixtures and exit.
    #[arg(long, default_value_t = false)]
    list_fixtures: bool,

    /// Run deterministic in-memory synthetic color decision cases and exit.
    #[arg(long, default_value_t = false)]
    synthetic_color_suite: bool,

    /// Audit fixture registry readiness without running the pipeline.
    #[arg(long, default_value_t = false)]
    fixture_coverage: bool,

    /// Decode the selected fixture and write upright-review previews plus a non-approved annotation draft.
    #[arg(long, value_name = "DIR")]
    write_orientation_review: Option<PathBuf>,

    /// Write a hash-bound, explicitly unapproved human review draft for --report artifacts.
    #[arg(long, value_name = "DIR")]
    write_render_review: Option<PathBuf>,

    /// Bind the exact matched grain-off perfect-render report used to review a grain-on result.
    #[arg(long, value_name = "REPORT")]
    render_review_grain_control_report: Option<PathBuf>,

    /// Compute fixture SHA-256 values during fixture coverage even when the registry has no expected hashes.
    #[arg(long, default_value_t = false)]
    compute_fixture_hashes: bool,

    /// Write a registry snapshot with available fixture hashes filled in during fixture coverage.
    #[arg(long)]
    write_fixture_hash_registry: Option<PathBuf>,

    /// Run every registered fixture as a real-image validation suite.
    #[arg(long, default_value_t = false)]
    fixture_suite: bool,

    /// Limit --fixture-suite rendering to one or more fixture names.
    #[arg(long = "fixture-suite-fixture", value_delimiter = ',')]
    fixture_suite_fixtures: Vec<String>,

    /// Directory containing a scanner roll to inspect or validate frame-by-frame.
    #[arg(long)]
    roll_dir: Option<PathBuf>,

    /// Inspect every supported scan file in --roll-dir without rendering.
    #[arg(long, default_value_t = false)]
    roll_inventory: bool,

    /// Write a fixture-registry scaffold from usable --roll-inventory frames.
    #[arg(long)]
    write_roll_fixture_registry: Option<PathBuf>,

    /// Run every readable scan in --roll-dir as an independent no-stitch validation frame.
    #[arg(long, default_value_t = false)]
    roll_suite: bool,

    /// Limit --roll-suite rendering to one or more frame names, stems, or slugs.
    #[arg(long = "roll-suite-frame", value_delimiter = ',')]
    roll_suite_frames: Vec<String>,

    /// Scene tag(s) to stamp onto entries generated by --write-roll-fixture-registry.
    #[arg(long = "roll-fixture-scene-tag", value_delimiter = ',')]
    roll_fixture_scene_tags: Vec<String>,

    /// Exposure tag(s) to stamp onto entries generated by --write-roll-fixture-registry.
    #[arg(long = "roll-fixture-exposure-tag", value_delimiter = ',')]
    roll_fixture_exposure_tags: Vec<String>,

    /// JSON metadata sidecar for per-frame entries generated by --write-roll-fixture-registry.
    #[arg(long)]
    roll_fixture_metadata: Option<PathBuf>,

    /// Write a per-frame metadata sidecar template from usable --roll-inventory frames.
    #[arg(long)]
    write_roll_fixture_metadata_template: Option<PathBuf>,

    /// Write a low-resolution PNG contact sheet from usable --roll-inventory frames.
    #[arg(long)]
    write_roll_contact_sheet: Option<PathBuf>,

    /// Write a JSON tile index for --write-roll-contact-sheet.
    #[arg(long)]
    write_roll_contact_sheet_index: Option<PathBuf>,

    /// Internal worker mode used to isolate one roll-suite render in a child process.
    #[arg(long, hide = true, default_value_t = false)]
    roll_suite_child: bool,

    /// First component TIFF. May be used alone for a direct single-scan run.
    #[arg(long)]
    component1: Option<PathBuf>,

    /// Optional second component TIFF. Requires --component1.
    #[arg(long)]
    component2: Option<PathBuf>,

    /// Summarize an existing report instead of running the pipeline.
    #[arg(long)]
    report: Option<PathBuf>,

    /// Compare render diagnostics against another report without mutating it.
    #[arg(long)]
    compare_report: Option<PathBuf>,

    /// Compare against a tracked compact validation summary baseline.
    #[arg(long)]
    compare_summary: Option<PathBuf>,

    /// Compare a roll-suite run against a previous roll-suite JSON summary.
    #[arg(long)]
    compare_roll_suite: Option<PathBuf>,

    /// Write a tracked compact validation summary baseline from the current summary.
    #[arg(long)]
    write_summary_baseline: Option<PathBuf>,

    /// Write missing tracked compact baselines declared by --fixture-suite registry entries.
    #[arg(long, default_value_t = false)]
    write_fixture_suite_baselines: bool,

    /// Replace existing tracked compact baselines when --write-fixture-suite-baselines is used.
    #[arg(long, default_value_t = false)]
    overwrite_fixture_suite_baselines: bool,

    /// Exit with an error when --compare-report finds review-required differences.
    #[arg(long, default_value_t = false)]
    strict: bool,

    /// Preserve reports and summaries, then require reviewable evidence and independently
    /// verified primary and requested auxiliary artifacts for every rendered output.
    #[arg(long, default_value_t = false)]
    require_reviewable: bool,

    /// Exit nonzero for selected comparison issue groups or exact issue names.
    #[arg(long, value_delimiter = ',')]
    fail_on: Vec<String>,

    /// Suppress human-readable terminal output.
    #[arg(long, default_value_t = false)]
    quiet: bool,

    /// Print the JSON summary to stdout after writing files.
    #[arg(long, default_value_t = false)]
    print_json_summary: bool,

    /// Output directory for pipeline renders and validation summaries.
    #[arg(long, default_value = "output/validation/logan")]
    output_dir: PathBuf,

    /// Optional scanner/film calibration profile JSON passed to the pipeline.
    #[arg(long)]
    calibration_profile: Option<PathBuf>,

    /// Optional local calibration library directory passed to the pipeline.
    #[arg(long)]
    calibration_library: Option<PathBuf>,

    /// Scanner/settings profile ID to select from the calibration library.
    #[arg(long)]
    scanner_profile: Option<String>,

    /// Roll profile ID to select from the calibration library.
    #[arg(long)]
    roll_profile: Option<String>,

    /// Film stock label used to rank calibration evidence and reject mismatched roll profiles.
    #[arg(long)]
    film_stock: Option<String>,

    /// Override density film-base RGB as comma-separated scanner sample values.
    #[arg(long, value_name = "R,G,B")]
    base_color: Option<String>,

    /// Internal provenance label for non-manual base-color overrides.
    #[arg(long, hide = true)]
    base_color_source: Option<String>,

    /// Internal confidence for non-manual base-color overrides.
    #[arg(long, hide = true)]
    base_color_confidence: Option<f64>,

    /// Internal diagnostic reason for non-manual base-color overrides.
    #[arg(long, hide = true)]
    base_color_reason: Option<String>,

    /// Color mapping mode passed to the pipeline: auto / calibrated / image-derived / neutral.
    #[arg(long, value_enum, default_value = "auto")]
    color_mode: ColorMode,

    /// Render input selection passed to the pipeline: auto / ica / direct-density.
    #[arg(long, value_enum, default_value = "auto")]
    render_input: RenderInputMode,

    /// Input scan mode passed to the pipeline: negative or already-positive RGB.
    #[arg(long, value_enum, default_value = "negative")]
    input_mode: InputMode,

    /// Finished-render intent passed to the pipeline.
    #[arg(long, value_enum, default_value = "modern-clean")]
    render_intent: RenderIntent,

    /// Quality/throughput mode passed to the pipeline.
    ///
    /// Validation defaults to balanced so strict fixture gates do not write large
    /// perfect-mode review artifacts unless explicitly requested.
    #[arg(long, value_enum, default_value = "balanced")]
    quality_mode: QualityMode,

    #[command(flatten)]
    white_balance: WhiteBalanceArgs,

    #[command(flatten)]
    geometry: GeometryArgs,

    #[command(flatten)]
    grain: GrainReductionArgs,

    /// Write the scene-referred master even outside perfect mode.
    #[arg(long, default_value_t = false)]
    write_master: bool,

    /// Apply saved guided-review decisions from a sidecar JSON.
    #[arg(long)]
    review_sidecar: Option<PathBuf>,

    /// Write guided-review decisions from validation pipeline saves.
    #[arg(long)]
    write_review_sidecar: Option<PathBuf>,

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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FixtureRegistry {
    fixtures: BTreeMap<String, FixtureEntry>,
    #[serde(default)]
    coverage_requirements: FixtureCoverageRequirements,
}

#[derive(Debug, Clone)]
struct LoadedFixtureRegistry {
    fixtures: BTreeMap<String, FixtureEntry>,
    coverage_requirements: FixtureCoverageRequirements,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FixtureCoverageRequirements {
    #[serde(default)]
    min_fixtures: Option<usize>,
    #[serde(default)]
    min_component_pairs: Option<usize>,
    #[serde(default)]
    min_component_sha256_pairs: Option<usize>,
    #[serde(default)]
    min_n_component_fixtures: Option<usize>,
    #[serde(default)]
    min_component_sha256_sets: Option<usize>,
    #[serde(default)]
    min_readable_tiff_pairs: Option<usize>,
    #[serde(default)]
    min_tiff_layout_consistent_pairs: Option<usize>,
    #[serde(default)]
    min_tiff_dimension_matched_pairs: Option<usize>,
    #[serde(default)]
    min_tiff_bits_per_sample: Option<u8>,
    #[serde(default)]
    min_summary_baselines: Option<usize>,
    #[serde(default)]
    min_summary_baseline_sha256_fixtures: Option<usize>,
    #[serde(default)]
    min_calibrated_fixtures: Option<usize>,
    #[serde(default)]
    min_calibration_sha256_fixtures: Option<usize>,
    #[serde(default)]
    min_uncalibrated_fixtures: Option<usize>,
    #[serde(default)]
    min_unique_scanner_profiles: Option<usize>,
    #[serde(default)]
    min_unique_roll_profiles: Option<usize>,
    #[serde(default)]
    min_unique_film_stocks: Option<usize>,
    #[serde(default)]
    min_scene_tags: Option<usize>,
    #[serde(default)]
    min_exposure_tags: Option<usize>,
    #[serde(default)]
    min_calibration_cases: Option<usize>,
    #[serde(default)]
    min_film_stock_calibration_pairs: Option<usize>,
    #[serde(default)]
    min_scene_exposure_pairs: Option<usize>,
    #[serde(default)]
    min_reference_fixtures: Option<usize>,
    #[serde(default)]
    min_reference_evidence_types: Option<usize>,
    #[serde(default)]
    min_reference_patch_fixtures: Option<usize>,
    #[serde(default)]
    min_reference_patch_count: Option<usize>,
    #[serde(default)]
    min_approved_render_review_fixtures: Option<usize>,
    #[serde(default)]
    min_debug_artifact_expectation_fixtures: Option<usize>,
    #[serde(default)]
    min_render_dynamic_range_contract_fixtures: Option<usize>,
    #[serde(default)]
    min_stitch_normalization_contract_fixtures: Option<usize>,
    #[serde(default)]
    min_geometry_preparation_contract_fixtures: Option<usize>,
    #[serde(default)]
    min_geometry_accuracy_contract_fixtures: Option<usize>,
    #[serde(default)]
    min_orientation_accuracy_contract_fixtures: Option<usize>,
    #[serde(default)]
    min_negative_reconstruction_contract_fixtures: Option<usize>,
    #[serde(default)]
    min_grain_reduction_enabled_fixtures: Option<usize>,
    #[serde(default)]
    min_grain_detail_contract_fixtures: Option<usize>,
    #[serde(default)]
    min_grain_reduction_effect_contract_fixtures: Option<usize>,
    #[serde(default)]
    required_film_stocks: Vec<String>,
    #[serde(default)]
    required_scene_tags: Vec<String>,
    #[serde(default)]
    required_exposure_tags: Vec<String>,
    #[serde(default)]
    required_calibration_cases: Vec<String>,
    #[serde(default)]
    required_scanner_profiles: Vec<String>,
    #[serde(default)]
    required_roll_profiles: Vec<String>,
    #[serde(default)]
    required_reference_evidence: Vec<String>,
    #[serde(default)]
    required_film_stock_calibration_pairs: Vec<String>,
    #[serde(default)]
    required_scene_exposure_pairs: Vec<String>,
    #[serde(default)]
    required_debug_artifact_kinds: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AdditionalFixtureComponent {
    path: PathBuf,
    #[serde(default)]
    sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FixtureEntry {
    component1: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    component2: Option<PathBuf>,
    #[serde(default)]
    component1_sha256: Option<String>,
    #[serde(default)]
    component2_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    additional_components: Vec<AdditionalFixtureComponent>,
    #[serde(default)]
    output_dir: Option<PathBuf>,
    #[serde(default)]
    input_mode: Option<String>,
    #[serde(default)]
    bit_depth: Option<u8>,
    #[serde(default)]
    grain_reduction: Option<String>,
    #[serde(default)]
    grain_strength: Option<f64>,
    #[serde(default)]
    grain_scale: Option<f64>,
    #[serde(default)]
    deskew: Option<String>,
    #[serde(default)]
    orientation_correction: Option<String>,
    #[serde(default)]
    deskew_angle_degrees: Option<f64>,
    #[serde(default)]
    force_stitch: bool,
    #[serde(default)]
    force_no_stitch: bool,
    #[serde(default)]
    calibration_profile: Option<PathBuf>,
    #[serde(default)]
    calibration_profile_sha256: Option<String>,
    #[serde(default)]
    calibration_library: Option<PathBuf>,
    #[serde(default)]
    calibration_library_sha256: Option<String>,
    #[serde(default)]
    scanner_profile: Option<String>,
    #[serde(default)]
    roll_profile: Option<String>,
    #[serde(default)]
    film_stock: Option<String>,
    #[serde(default)]
    scene_tags: Vec<String>,
    #[serde(default)]
    exposure_tags: Vec<String>,
    #[serde(default)]
    reference_evidence: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    render_review: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    render_review_sha256: Option<String>,
    #[serde(default)]
    calibration_case: Option<String>,
    #[serde(default)]
    expectations: FixtureExpectations,
    #[serde(default)]
    summary_baseline: Option<PathBuf>,
    #[serde(default)]
    summary_baseline_sha256: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

impl FixtureEntry {
    fn component_count(&self) -> usize {
        1 + usize::from(self.component2.is_some()) + self.additional_components.len()
    }

    fn pipeline_inputs(&self) -> Vec<PathBuf> {
        let mut inputs = Vec::with_capacity(self.component_count());
        inputs.push(self.component1.clone());
        inputs.extend(self.component2.iter().cloned());
        inputs.extend(
            self.additional_components
                .iter()
                .map(|component| component.path.clone()),
        );
        inputs
    }

    fn component_hashes(&self) -> Vec<Option<&str>> {
        let mut hashes = Vec::with_capacity(self.component_count());
        hashes.push(self.component1_sha256.as_deref());
        if self.component2.is_some() {
            hashes.push(self.component2_sha256.as_deref());
        }
        hashes.extend(
            self.additional_components
                .iter()
                .map(|component| component.sha256.as_deref()),
        );
        hashes
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RollFixtureMetadata {
    #[serde(default)]
    coverage_requirements: FixtureCoverageRequirements,
    #[serde(default)]
    frames: BTreeMap<String, RollFixtureMetadataEntry>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RollFixtureMetadataEntry {
    #[serde(default)]
    calibration_profile: Option<PathBuf>,
    #[serde(default)]
    calibration_profile_sha256: Option<String>,
    #[serde(default)]
    calibration_library: Option<PathBuf>,
    #[serde(default)]
    calibration_library_sha256: Option<String>,
    #[serde(default)]
    scanner_profile: Option<String>,
    #[serde(default)]
    roll_profile: Option<String>,
    #[serde(default)]
    film_stock: Option<String>,
    #[serde(default)]
    scene_tags: Option<Vec<String>>,
    #[serde(default)]
    exposure_tags: Option<Vec<String>>,
    #[serde(default)]
    reference_evidence: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    render_review: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    render_review_sha256: Option<String>,
    #[serde(default)]
    calibration_case: Option<String>,
    #[serde(default)]
    expectations: FixtureExpectations,
    #[serde(default)]
    summary_baseline: Option<PathBuf>,
    #[serde(default)]
    summary_baseline_sha256: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BorderCropComponentExpectation {
    component_index: usize,
    top_removed: usize,
    bottom_removed: usize,
    left_removed: usize,
    right_removed: usize,
    tolerance_px: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct OrientationComponentExpectation {
    component_index: usize,
    upright_approved: bool,
    decoded_pixel_sha256: String,
    tag_present: bool,
    tag_value: Option<u16>,
    transform: String,
    applied: bool,
    source_width: usize,
    source_height: usize,
    output_width: usize,
    output_height: usize,
}

#[derive(Debug, Serialize)]
struct OrientationReviewManifest {
    schema_version: u32,
    fixture: String,
    review_status: &'static str,
    instructions: &'static str,
    input_mode: String,
    working_bit_depth: u8,
    orientation_correction: String,
    preview_transform: &'static str,
    previews: Vec<OrientationReviewPreview>,
    orientation_components_expected: Vec<OrientationComponentExpectation>,
}

#[derive(Debug, Serialize)]
struct OrientationReviewPreview {
    component_index: usize,
    source_path: String,
    preview_path: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FixtureExpectations {
    #[serde(default)]
    deskew_status: Option<String>,
    #[serde(default)]
    deskew_applied: Option<bool>,
    #[serde(default)]
    deskew_review_required: Option<bool>,
    #[serde(default)]
    deskew_retained_area_ratio_min: Option<f64>,
    #[serde(default)]
    deskew_all_components_applied: Option<bool>,
    #[serde(default)]
    deskew_minimum_component_retained_area_ratio_min: Option<f64>,
    #[serde(default)]
    border_crop_all_components_cropped: Option<bool>,
    #[serde(default)]
    border_crop_minimum_removed_edge_count_per_component_min: Option<usize>,
    #[serde(default)]
    border_crop_retained_area_ratio_min: Option<f64>,
    #[serde(default)]
    border_crop_retained_area_ratio_max: Option<f64>,
    #[serde(default)]
    border_crop_rejected: Option<bool>,
    #[serde(default)]
    deskew_correction_degrees_expected: Option<f64>,
    #[serde(default)]
    deskew_correction_tolerance_degrees: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    border_crop_components_expected: Vec<BorderCropComponentExpectation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    orientation_components_expected: Vec<OrientationComponentExpectation>,
    #[serde(default)]
    stitch_decision: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    inferred_component_order: Vec<usize>,
    #[serde(default)]
    technical_white_balance_status: Option<String>,
    #[serde(default)]
    technical_white_balance_applied: Option<bool>,
    #[serde(default)]
    technical_white_balance_review_required: Option<bool>,
    #[serde(default)]
    creative_temperature: Option<f64>,
    #[serde(default)]
    creative_tint: Option<f64>,
    #[serde(default)]
    seam_exposure_model: Option<String>,
    #[serde(default)]
    seam_exposure_held_out_validation_passed: Option<bool>,
    #[serde(default)]
    seam_exposure_held_out_improvement_over_gain_min: Option<f64>,
    #[serde(default)]
    seam_exposure_offset_normalized_abs_max: Option<f64>,
    #[serde(default)]
    seam_exposure_spatial_slope_abs_min: Option<f64>,
    #[serde(default)]
    seam_exposure_spatial_slope_abs_max: Option<f64>,
    #[serde(default)]
    seam_exposure_spatial_slope_agreement_ratio_min: Option<f64>,
    #[serde(default)]
    seam_exposure_held_out_spatial_improvement_over_best_constant_min: Option<f64>,
    #[serde(default)]
    seam_exposure_spatial_offset_slope_normalized_abs_min: Option<f64>,
    #[serde(default)]
    seam_exposure_spatial_offset_slope_normalized_abs_max: Option<f64>,
    #[serde(default)]
    seam_exposure_spatial_offset_endpoint_normalized_abs_max: Option<f64>,
    #[serde(default)]
    seam_exposure_spatial_affine_slope_agreement_ratio_min: Option<f64>,
    #[serde(default)]
    seam_exposure_spatial_affine_center_offset_delta_normalized_max: Option<f64>,
    #[serde(default)]
    seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler_min: Option<f64>,
    #[serde(default)]
    seam_exposure_spatial_2d_gain_accepted: Option<bool>,
    #[serde(default)]
    seam_exposure_spatial_2d_gain_offset_accepted: Option<bool>,
    #[serde(default)]
    seam_exposure_spatial_quadratic_gain_accepted: Option<bool>,
    #[serde(default)]
    seam_exposure_spatial_quadratic_gain_offset_accepted: Option<bool>,
    #[serde(default)]
    seam_exposure_spatial_2d_distinct_columns_min: Option<usize>,
    #[serde(default)]
    seam_exposure_spatial_2d_horizontal_slope_agreement_ratio_min: Option<f64>,
    #[serde(default)]
    seam_exposure_held_out_spatial_2d_improvement_over_best_simpler_min: Option<f64>,
    #[serde(default)]
    seam_blend_required: bool,
    #[serde(default)]
    seam_blend_mode: Option<String>,
    #[serde(default)]
    seam_blend_review_required: Option<bool>,
    #[serde(default)]
    seam_detail_review_required: Option<bool>,
    #[serde(default)]
    seam_detail_supported_scale_count_min: Option<usize>,
    #[serde(default)]
    seam_detail_max_symmetric_energy_ratio_max: Option<f64>,
    #[serde(default)]
    seam_gradient_ratio_max: Option<f64>,
    #[serde(default)]
    seam_overlap_p95_abs_difference_max: Option<f64>,
    #[serde(default)]
    base_estimate_source: Option<String>,
    #[serde(default)]
    base_confidence_min: Option<f64>,
    #[serde(default)]
    density_inversion_skipped: Option<bool>,
    #[serde(default)]
    negative_response_model: Option<String>,
    #[serde(default)]
    negative_response_source: Option<String>,
    #[serde(default)]
    negative_response_accepted: Option<bool>,
    #[serde(default)]
    negative_response_review_required: Option<bool>,
    #[serde(default)]
    negative_response_crosstalk_model: Option<String>,
    #[serde(default)]
    negative_response_characteristic_curve_model: Option<String>,
    #[serde(default)]
    negative_response_measured_model_id: Option<String>,
    #[serde(default)]
    negative_response_measured_confidence_min: Option<f64>,
    #[serde(default)]
    negative_response_held_out_delta_e00_rms_max: Option<f64>,
    #[serde(default)]
    negative_response_held_out_max_delta_e00_max: Option<f64>,
    #[serde(default)]
    negative_response_held_out_improvement_over_unit_slope_min: Option<f64>,
    #[serde(default)]
    negative_response_density_noise_gain_max: Option<f64>,
    #[serde(default)]
    negative_response_curve_extrapolated_ratio_max: Option<f64>,
    #[serde(default)]
    negative_response_signed_headroom_preserved: Option<bool>,
    #[serde(default)]
    negative_response_curve_interpolation: Option<String>,
    #[serde(default)]
    output_color_space: Option<String>,
    #[serde(default)]
    render_input_source: Option<String>,
    #[serde(default)]
    render_input_reason_contains: Option<String>,
    #[serde(default)]
    mapping_strategy: Option<String>,
    #[serde(default)]
    selected_mapping_reason_contains: Option<String>,
    #[serde(default)]
    selected_candidate: Option<String>,
    #[serde(default)]
    selected_candidate_rank: Option<usize>,
    #[serde(default)]
    calibration_acceptance_status: Option<String>,
    #[serde(default)]
    calibration_color_mapping_applied: Option<bool>,
    #[serde(default)]
    calibration_confidence_min: Option<f64>,
    #[serde(default)]
    calibration_matrix_condition_number_max: Option<f64>,
    #[serde(default)]
    calibration_rejection_details_required: Vec<String>,
    #[serde(default)]
    candidate_risk: Option<String>,
    #[serde(default)]
    tone_color_trust_state: Option<String>,
    #[serde(default)]
    neutral_safety_rescue_applied: Option<bool>,
    #[serde(default)]
    neutral_safety_rescue_preserved_ratio_gain_min: Option<f64>,
    #[serde(default)]
    neutral_safety_rescue_midtone_saturation_p95_reduction_min: Option<f64>,
    #[serde(default)]
    neutral_safety_rescue_reason_contains: Option<String>,
    #[serde(default)]
    highlight_chroma_compressed_ratio_min: Option<f64>,
    #[serde(default)]
    highlight_chroma_compressed_ratio_max: Option<f64>,
    #[serde(default)]
    highlight_neutral_chroma_compressed_ratio_max: Option<f64>,
    #[serde(default)]
    shadow_chroma_compressed_ratio_max: Option<f64>,
    #[serde(default)]
    grain_reduction_enabled: Option<bool>,
    #[serde(default)]
    grain_reduction_applied_ratio_min: Option<f64>,
    #[serde(default)]
    grain_reduction_structure_excluded_ratio_min: Option<f64>,
    #[serde(default)]
    grain_reduction_flat_luma_p95_reduction_ratio_min: Option<f64>,
    #[serde(default)]
    grain_reduction_flat_chroma_p95_reduction_ratio_min: Option<f64>,
    #[serde(default)]
    grain_detail_review_required: Option<bool>,
    #[serde(default)]
    grain_detail_decision_supported: Option<bool>,
    #[serde(default)]
    grain_detail_luminance_probe_count_min: Option<usize>,
    #[serde(default)]
    grain_detail_chroma_probe_count_min: Option<usize>,
    #[serde(default)]
    grain_detail_luminance_p10_retention_min: Option<f64>,
    #[serde(default)]
    grain_detail_chroma_p10_retention_min: Option<f64>,
    #[serde(default)]
    selected_quality_score_max: Option<f64>,
    #[serde(default)]
    technical_safety_score_max: Option<f64>,
    #[serde(default)]
    color_fidelity_score_max: Option<f64>,
    #[serde(default)]
    memory_color_penalty_max: Option<f64>,
    #[serde(default)]
    spatial_consistency_penalty_max: Option<f64>,
    #[serde(default)]
    selected_runner_up_quality_delta_min: Option<f64>,
    #[serde(default)]
    density_monotonicity_score_min: Option<f64>,
    #[serde(default)]
    hue_linearity_score_min: Option<f64>,
    #[serde(default)]
    saturation_preservation_median_ratio_min: Option<f64>,
    #[serde(default)]
    spatial_neutral_delta_p95_max: Option<f64>,
    #[serde(default)]
    post_scale_preserved_ratio_min: Option<f64>,
    #[serde(default)]
    render_luminance_range_p05_p95_min: Option<f64>,
    #[serde(default)]
    render_review_status: Option<String>,
    #[serde(default)]
    render_reviewable: Option<bool>,
    #[serde(default)]
    tone_output_confidence_status: Option<String>,
    #[serde(default)]
    tone_output_review_required: Option<bool>,
    #[serde(default)]
    tone_output_evidence_confidence_min: Option<f64>,
    #[serde(default)]
    render_to_mapped_luminance_range_ratio_min: Option<f64>,
    #[serde(default)]
    post_chroma_compression_clipped_high_ratio_max: Option<f64>,
    #[serde(default)]
    post_chroma_compression_clipped_low_ratio_max: Option<f64>,
    #[serde(default)]
    reference_patch_evaluation_required: bool,
    #[serde(default)]
    reference_patch_count_min: Option<usize>,
    #[serde(default)]
    reference_patch_hue_family_regression_count_max: Option<usize>,
    #[serde(default)]
    reference_patch_selected_regresses_image_derived: Option<bool>,
    #[serde(default)]
    reference_patch_delta_e2000_delta_vs_image_derived_max: Option<f64>,
    #[serde(default)]
    reference_patch_max_delta_vs_image_derived_max: Option<f64>,
    #[serde(default)]
    reference_patch_delta_e_max_delta_vs_image_derived_max: Option<f64>,
    #[serde(default)]
    reference_patch_delta_e2000_max_delta_vs_image_derived_max: Option<f64>,
    #[serde(default)]
    reference_patch_rms_delta_e_max: Option<f64>,
    #[serde(default)]
    reference_patch_rms_delta_e2000_max: Option<f64>,
    #[serde(default)]
    debug_artifacts_required: bool,
    #[serde(default)]
    debug_artifact_kinds_required: Vec<String>,
    #[serde(default)]
    candidate_acceptance_signatures_required: Vec<String>,
    #[serde(default)]
    selection_rejections_required: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FixtureCoverageSummary {
    status: String,
    fixture_count: usize,
    component_file_count: usize,
    max_component_count: usize,
    n_component_fixture_count: usize,
    n_component_validation_ready_fixture_count: usize,
    component_set_available_count: usize,
    component_sha256_declared_set_count: usize,
    component_sha256_computed_set_count: usize,
    component_sha256_set_count: usize,
    readable_tiff_set_count: usize,
    tiff_layout_consistent_set_count: usize,
    tiff_dimension_matched_set_count: usize,
    component_pair_available_count: usize,
    component_sha256_declared_pair_count: usize,
    component_sha256_computed_pair_count: usize,
    component_sha256_pair_count: usize,
    readable_tiff_pair_count: usize,
    tiff_layout_consistent_pair_count: usize,
    tiff_dimension_matched_pair_count: usize,
    validation_ready_fixture_count: usize,
    render_dynamic_range_contract_fixture_count: usize,
    stitch_normalization_contract_fixture_count: usize,
    geometry_preparation_contract_fixture_count: usize,
    geometry_accuracy_contract_fixture_count: usize,
    orientation_accuracy_contract_fixture_count: usize,
    negative_reconstruction_contract_fixture_count: usize,
    grain_reduction_enabled_fixture_count: usize,
    grain_detail_contract_fixture_count: usize,
    grain_reduction_effect_contract_fixture_count: usize,
    summary_baseline_declared_count: usize,
    summary_baseline_file_count: usize,
    summary_baseline_parseable_count: usize,
    summary_baseline_contract_complete_count: usize,
    summary_baseline_count: usize,
    summary_baseline_sha256_declared_count: usize,
    summary_baseline_sha256_computed_count: usize,
    summary_baseline_sha256_count: usize,
    calibration_evidence_declared_count: usize,
    calibration_evidence_count: usize,
    calibration_evidence_unusable_count: usize,
    calibration_sha256_declared_count: usize,
    calibration_sha256_computed_count: usize,
    calibration_sha256_count: usize,
    uncalibrated_fixture_count: usize,
    scanner_profile_count: usize,
    unique_scanner_profiles: Vec<String>,
    missing_required_scanner_profiles: Vec<String>,
    roll_profile_count: usize,
    unique_roll_profiles: Vec<String>,
    missing_required_roll_profiles: Vec<String>,
    film_stock_count: usize,
    unique_film_stocks: Vec<String>,
    missing_required_film_stocks: Vec<String>,
    scene_tag_count: usize,
    scene_tags: Vec<String>,
    missing_required_scene_tags: Vec<String>,
    exposure_tag_count: usize,
    exposure_tags: Vec<String>,
    missing_required_exposure_tags: Vec<String>,
    calibration_case_count: usize,
    calibration_cases: Vec<String>,
    missing_required_calibration_cases: Vec<String>,
    reference_fixture_count: usize,
    reference_evidence_type_count: usize,
    reference_evidence: Vec<String>,
    missing_required_reference_evidence: Vec<String>,
    reference_patch_fixture_count: usize,
    reference_patch_count: usize,
    approved_render_review_fixture_count: usize,
    debug_artifact_expectation_fixture_count: usize,
    debug_artifact_kinds_required: Vec<String>,
    missing_required_debug_artifact_kinds: Vec<String>,
    film_stock_calibration_pair_count: usize,
    film_stock_calibration_pairs: Vec<String>,
    missing_required_film_stock_calibration_pairs: Vec<String>,
    scene_exposure_pair_count: usize,
    scene_exposure_pairs: Vec<String>,
    missing_required_scene_exposure_pairs: Vec<String>,
    coverage_requirements: FixtureCoverageRequirements,
    fixtures: Vec<FixtureCoverageEntry>,
    action_items: Vec<String>,
    issues: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FixtureCoverageEntry {
    name: String,
    component_count: usize,
    component1: String,
    component1_exists: bool,
    component1_sha256: Option<FixtureSha256Probe>,
    component1_tiff: Option<FixtureTiffProbe>,
    component2: Option<String>,
    component2_exists: bool,
    component2_sha256: Option<FixtureSha256Probe>,
    component2_tiff: Option<FixtureTiffProbe>,
    tiff_pair: Option<FixtureTiffPairProbe>,
    additional_components: Vec<AdditionalFixtureComponentCoverage>,
    all_components_exist: bool,
    all_component_sha256_declared: bool,
    all_component_sha256_computed: bool,
    all_component_sha256_matched: bool,
    all_components_readable_tiff: bool,
    all_component_layouts_consistent: bool,
    all_component_dimensions_matched: bool,
    output_dir: Option<String>,
    input_mode: Option<String>,
    bit_depth: Option<u8>,
    grain_reduction: Option<String>,
    grain_strength: Option<f64>,
    grain_scale: Option<f64>,
    grain_reduction_enabled_declared: bool,
    grain_detail_contract_complete: bool,
    grain_detail_contract_missing_fields: Vec<String>,
    grain_reduction_effect_contract_complete: bool,
    grain_reduction_effect_contract_missing_fields: Vec<String>,
    render_dynamic_range_contract_declared: bool,
    render_dynamic_range_contract_complete: bool,
    render_dynamic_range_contract_missing_fields: Vec<String>,
    stitch_normalization_contract_declared: bool,
    stitch_normalization_contract_complete: bool,
    stitch_normalization_contract_missing_fields: Vec<String>,
    geometry_preparation_contract_declared: bool,
    geometry_preparation_contract_complete: bool,
    geometry_preparation_contract_missing_fields: Vec<String>,
    geometry_accuracy_contract_declared: bool,
    geometry_accuracy_contract_complete: bool,
    geometry_accuracy_contract_missing_fields: Vec<String>,
    orientation_accuracy_contract_declared: bool,
    orientation_accuracy_contract_complete: bool,
    orientation_accuracy_contract_missing_fields: Vec<String>,
    negative_reconstruction_contract_declared: bool,
    negative_reconstruction_contract_complete: bool,
    negative_reconstruction_contract_missing_fields: Vec<String>,
    deskew: Option<String>,
    deskew_angle_degrees: Option<f64>,
    force_stitch: bool,
    force_no_stitch: bool,
    summary_baseline: Option<String>,
    summary_baseline_exists: Option<bool>,
    summary_baseline_parse_status: Option<String>,
    summary_baseline_contract_status: Option<String>,
    summary_baseline_contract_missing_fields: Vec<String>,
    summary_baseline_valid: bool,
    summary_baseline_sha256: Option<FixtureSha256Probe>,
    render_review: Option<String>,
    render_review_exists: Option<bool>,
    render_review_sha256: Option<FixtureSha256Probe>,
    render_review_inspection: Option<render_review::RenderReviewInspection>,
    approved_render_review: bool,
    calibration_profile: Option<String>,
    calibration_profile_exists: Option<bool>,
    calibration_profile_parse_status: Option<String>,
    calibration_profile_sha256: Option<FixtureSha256Probe>,
    calibration_library: Option<String>,
    calibration_library_exists: Option<bool>,
    calibration_library_selection_status: Option<String>,
    calibration_library_sha256: Option<FixtureSha256Probe>,
    calibration_reference_patch_count: Option<usize>,
    scanner_profile: Option<String>,
    roll_profile: Option<String>,
    film_stock: Option<String>,
    scene_tags: Vec<String>,
    exposure_tags: Vec<String>,
    reference_evidence: Vec<String>,
    calibration_case: Option<String>,
    debug_artifacts_required: bool,
    debug_artifact_kinds_required: Vec<String>,
    calibration_evidence_declared: bool,
    calibration_evidence_usable: bool,
    validation_ready: bool,
    action_items: Vec<String>,
    repair_plan: Vec<FixtureRepairPlanItem>,
    issues: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AdditionalFixtureComponentCoverage {
    component_number: usize,
    path: String,
    exists: bool,
    sha256: Option<FixtureSha256Probe>,
    tiff: Option<FixtureTiffProbe>,
    comparison_to_component1: Option<FixtureTiffPairProbe>,
}

#[derive(Debug, Clone, Serialize)]
struct FixtureRepairPlanItem {
    action: String,
    paths: Vec<String>,
    details: String,
}

#[derive(Debug, Clone, Serialize)]
struct FixtureSha256Probe {
    status: String,
    expected_sha256: Option<String>,
    actual_sha256: Option<String>,
    file_size_bytes: Option<u64>,
    file_count: Option<usize>,
    matched: bool,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct FixtureTiffProbe {
    status: String,
    readable: bool,
    width: Option<usize>,
    height: Option<usize>,
    color_type: Option<String>,
    source_bits_per_sample: Option<u8>,
    source_channel_count: Option<usize>,
    source_has_alpha: Option<bool>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct FixtureTiffPairProbe {
    dimensions_match: Option<bool>,
    width_delta: Option<usize>,
    height_delta: Option<usize>,
    dimensions_compatible: bool,
    color_type_match: Option<bool>,
    bits_per_sample_match: Option<bool>,
    channel_count_match: Option<bool>,
    alpha_flag_match: Option<bool>,
    layout_consistent: bool,
    dimension_matched: bool,
}

#[derive(Debug, Serialize)]
struct FixtureSuiteSummary {
    status: String,
    fixture_count: usize,
    passed_count: usize,
    review_required_count: usize,
    failed_count: usize,
    coverage: FixtureCoverageSummary,
    fixtures: Vec<FixtureSuiteEntry>,
    issues: Vec<String>,
}

#[derive(Debug, Default, Serialize)]
struct GeometryPreparationFixtureSuiteEvidence {
    all_deskew_components_applied: Option<bool>,
    expected_all_deskew_components_applied: Option<bool>,
    minimum_deskew_retained_area_ratio: Option<f64>,
    expected_minimum_deskew_retained_area_ratio: Option<f64>,
    all_border_crop_components_cropped: Option<bool>,
    expected_all_border_crop_components_cropped: Option<bool>,
    minimum_removed_edge_count_per_component: Option<usize>,
    expected_minimum_removed_edge_count_per_component: Option<usize>,
    minimum_border_crop_retained_area_ratio: Option<f64>,
    expected_minimum_border_crop_retained_area_ratio: Option<f64>,
    maximum_border_crop_retained_area_ratio: Option<f64>,
    expected_maximum_border_crop_retained_area_ratio: Option<f64>,
    border_crop_rejected: Option<bool>,
    expected_border_crop_rejected: Option<bool>,
    deskew_correction_degrees: Option<f64>,
    expected_deskew_correction_degrees: Option<f64>,
    expected_deskew_correction_tolerance_degrees: Option<f64>,
    border_crop_components: Vec<BorderCropComponentValidationSummary>,
    expected_border_crop_components: Vec<BorderCropComponentExpectation>,
}

#[derive(Debug, Default, Serialize)]
struct OrientationFixtureSuiteEvidence {
    components: Vec<InputOrientationComponentValidationSummary>,
    expected_components: Vec<OrientationComponentExpectation>,
}

#[derive(Debug, Default, Serialize)]
struct NegativeReconstructionFixtureSuiteEvidence {
    base_confidence: Option<f64>,
    expected_base_confidence_min: Option<f64>,
    density_inversion_skipped: Option<bool>,
    expected_density_inversion_skipped: Option<bool>,
    response_model: Option<String>,
    expected_response_model: Option<String>,
    response_source: Option<String>,
    expected_response_source: Option<String>,
    response_accepted: Option<bool>,
    expected_response_accepted: Option<bool>,
    response_review_required: Option<bool>,
    expected_response_review_required: Option<bool>,
    crosstalk_model: Option<String>,
    expected_crosstalk_model: Option<String>,
    characteristic_curve_model: Option<String>,
    expected_characteristic_curve_model: Option<String>,
    measured_model_id: Option<String>,
    expected_measured_model_id: Option<String>,
    measured_confidence: Option<f64>,
    expected_measured_confidence_min: Option<f64>,
    held_out_delta_e00_rms: Option<f64>,
    expected_held_out_delta_e00_rms_max: Option<f64>,
    held_out_delta_e00_max: Option<f64>,
    expected_held_out_delta_e00_max: Option<f64>,
    held_out_improvement_over_unit_slope: Option<f64>,
    expected_held_out_improvement_over_unit_slope_min: Option<f64>,
    maximum_density_noise_gain: Option<f64>,
    expected_maximum_density_noise_gain: Option<f64>,
    curve_extrapolated_any_ratio: Option<f64>,
    expected_curve_extrapolated_any_ratio_max: Option<f64>,
    signed_headroom_preserved: Option<bool>,
    expected_signed_headroom_preserved: Option<bool>,
    curve_interpolation: Option<String>,
    expected_curve_interpolation: Option<String>,
}

#[derive(Debug, Serialize)]
struct FixtureSuiteEntry {
    name: String,
    status: String,
    component_count: usize,
    coverage_validation_ready: bool,
    coverage_issues: Vec<String>,
    coverage_action_items: Vec<String>,
    output_dir: String,
    output_path: Option<String>,
    output_modified_at: Option<String>,
    output_width: Option<usize>,
    output_height: Option<usize>,
    output_color_space: Option<String>,
    expected_output_color_space: Option<String>,
    output_file_icc_profile_matches_report: Option<bool>,
    delivery_artifacts_intact: Option<bool>,
    delivery_artifact_issues: Vec<String>,
    stale_render_artifact_count: Option<usize>,
    report_path: Option<String>,
    summary_json_path: Option<String>,
    summary_md_path: Option<String>,
    summary_baseline_path: Option<String>,
    summary_baseline_status: Option<String>,
    summary_baseline_write_status: Option<String>,
    summary_baseline_written_path: Option<String>,
    deskew_status: Option<String>,
    expected_deskew_status: Option<String>,
    deskew_applied: Option<bool>,
    expected_deskew_applied: Option<bool>,
    deskew_review_required: Option<bool>,
    expected_deskew_review_required: Option<bool>,
    deskew_retained_area_ratio: Option<f64>,
    expected_deskew_retained_area_ratio_min: Option<f64>,
    geometry_preparation: GeometryPreparationFixtureSuiteEvidence,
    input_orientation: OrientationFixtureSuiteEvidence,
    stitch_decision: Option<String>,
    expected_stitch_decision: Option<String>,
    inferred_component_order: Vec<usize>,
    expected_inferred_component_order: Vec<usize>,
    technical_white_balance_status: Option<String>,
    expected_technical_white_balance_status: Option<String>,
    technical_white_balance_applied: Option<bool>,
    expected_technical_white_balance_applied: Option<bool>,
    technical_white_balance_review_required: Option<bool>,
    expected_technical_white_balance_review_required: Option<bool>,
    creative_temperature: Option<f64>,
    expected_creative_temperature: Option<f64>,
    creative_tint: Option<f64>,
    expected_creative_tint: Option<f64>,
    seam_exposure_model: Option<String>,
    expected_seam_exposure_model: Option<String>,
    seam_exposure_held_out_validation_passed: Option<bool>,
    expected_seam_exposure_held_out_validation_passed: Option<bool>,
    seam_exposure_held_out_improvement_over_gain: Option<f64>,
    expected_seam_exposure_held_out_improvement_over_gain_min: Option<f64>,
    seam_exposure_offset_normalized_abs_max: Option<f64>,
    expected_seam_exposure_offset_normalized_abs_max: Option<f64>,
    seam_exposure_spatial_slope_abs_max: Option<f64>,
    expected_seam_exposure_spatial_slope_abs_min: Option<f64>,
    expected_seam_exposure_spatial_slope_abs_max: Option<f64>,
    seam_exposure_spatial_slope_agreement_ratio: Option<f64>,
    expected_seam_exposure_spatial_slope_agreement_ratio_min: Option<f64>,
    seam_exposure_held_out_spatial_improvement_over_best_constant: Option<f64>,
    expected_seam_exposure_held_out_spatial_improvement_over_best_constant_min: Option<f64>,
    seam_exposure_spatial_offset_slope_normalized_abs_max: Option<f64>,
    expected_seam_exposure_spatial_offset_slope_normalized_abs_min: Option<f64>,
    expected_seam_exposure_spatial_offset_slope_normalized_abs_max: Option<f64>,
    seam_exposure_spatial_offset_endpoint_normalized_abs_max: Option<f64>,
    expected_seam_exposure_spatial_offset_endpoint_normalized_abs_max: Option<f64>,
    seam_exposure_spatial_affine_slope_agreement_ratio: Option<f64>,
    expected_seam_exposure_spatial_affine_slope_agreement_ratio_min: Option<f64>,
    seam_exposure_spatial_affine_center_offset_delta_normalized: Option<f64>,
    expected_seam_exposure_spatial_affine_center_offset_delta_normalized_max: Option<f64>,
    seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler: Option<f64>,
    expected_seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler_min:
        Option<f64>,
    seam_blend_mode: Option<String>,
    expected_seam_blend_required: bool,
    expected_seam_blend_mode: Option<String>,
    seam_blend_review_required: Option<bool>,
    expected_seam_blend_review_required: Option<bool>,
    seam_detail_review_required: Option<bool>,
    expected_seam_detail_review_required: Option<bool>,
    seam_detail_supported_scale_count: Option<usize>,
    expected_seam_detail_supported_scale_count_min: Option<usize>,
    seam_detail_max_symmetric_energy_ratio: Option<f64>,
    expected_seam_detail_max_symmetric_energy_ratio_max: Option<f64>,
    seam_gradient_ratio: Option<f64>,
    expected_seam_gradient_ratio_max: Option<f64>,
    seam_overlap_p95_abs_difference: Option<f64>,
    expected_seam_overlap_p95_abs_difference_max: Option<f64>,
    base_estimate_source: Option<String>,
    expected_base_estimate_source: Option<String>,
    negative_reconstruction: NegativeReconstructionFixtureSuiteEvidence,
    reference_evidence: Vec<String>,
    render_input_source: Option<String>,
    expected_render_input_source: Option<String>,
    render_input_reason: Option<String>,
    expected_render_input_reason_contains: Option<String>,
    selected_mapping_reason: Option<String>,
    expected_selected_mapping_reason_contains: Option<String>,
    expected_calibration_source: Option<String>,
    expected_calibration_scanner_profile: Option<String>,
    expected_calibration_roll_profile: Option<String>,
    expected_calibration_film_stock: Option<String>,
    calibration_status: Option<String>,
    calibration_source: Option<String>,
    calibration_scanner_profile_status: Option<String>,
    calibration_scanner_profile_id: Option<String>,
    calibration_roll_profile_status: Option<String>,
    calibration_roll_profile_id: Option<String>,
    calibration_requested_film_stock: Option<String>,
    calibration_acceptance_status: Option<String>,
    calibration_color_mapping_applied: Option<bool>,
    expected_calibration_color_mapping_applied: Option<bool>,
    calibration_confidence: Option<f64>,
    expected_calibration_confidence_min: Option<f64>,
    calibration_matrix_condition_number: Option<f64>,
    expected_calibration_matrix_condition_number_max: Option<f64>,
    calibration_rejection_details: Vec<String>,
    expected_calibration_rejection_details_required: Vec<String>,
    selected_candidate: Option<String>,
    expected_selected_candidate: Option<String>,
    selected_candidate_rank: Option<usize>,
    expected_selected_candidate_rank: Option<usize>,
    candidate_acceptance_signatures: Vec<String>,
    expected_candidate_acceptance_signatures_required: Vec<String>,
    selection_rejections: Vec<String>,
    expected_selection_rejections_required: Vec<String>,
    selected_quality_score: Option<f64>,
    expected_selected_quality_score_max: Option<f64>,
    selected_runner_up_quality_delta: Option<f64>,
    expected_selected_runner_up_quality_delta_min: Option<f64>,
    technical_safety_score: Option<f64>,
    expected_technical_safety_score_max: Option<f64>,
    color_fidelity_score: Option<f64>,
    expected_color_fidelity_score_max: Option<f64>,
    memory_color_penalty: Option<f64>,
    expected_memory_color_penalty_max: Option<f64>,
    spatial_consistency_penalty: Option<f64>,
    expected_spatial_consistency_penalty_max: Option<f64>,
    density_monotonicity_score: Option<f64>,
    expected_density_monotonicity_score_min: Option<f64>,
    hue_linearity_score: Option<f64>,
    expected_hue_linearity_score_min: Option<f64>,
    saturation_preservation_median_ratio: Option<f64>,
    expected_saturation_preservation_median_ratio_min: Option<f64>,
    spatial_neutral_delta_p95: Option<f64>,
    expected_spatial_neutral_delta_p95_max: Option<f64>,
    candidate_risk: Option<String>,
    expected_candidate_risk: Option<String>,
    tone_color_trust_state: Option<String>,
    expected_tone_color_trust_state: Option<String>,
    neutral_safety_rescue_applied: Option<bool>,
    expected_neutral_safety_rescue_applied: Option<bool>,
    neutral_safety_rescue_preserved_ratio_gain: Option<f64>,
    expected_neutral_safety_rescue_preserved_ratio_gain_min: Option<f64>,
    neutral_safety_rescue_midtone_saturation_p95_reduction: Option<f64>,
    expected_neutral_safety_rescue_midtone_saturation_p95_reduction_min: Option<f64>,
    neutral_safety_rescue_reason: Option<String>,
    expected_neutral_safety_rescue_reason_contains: Option<String>,
    highlight_chroma_compressed_ratio: Option<f64>,
    expected_highlight_chroma_compressed_ratio_min: Option<f64>,
    expected_highlight_chroma_compressed_ratio_max: Option<f64>,
    highlight_neutral_chroma_compressed_ratio: Option<f64>,
    expected_highlight_neutral_chroma_compressed_ratio_max: Option<f64>,
    shadow_chroma_compressed_ratio: Option<f64>,
    expected_shadow_chroma_compressed_ratio_max: Option<f64>,
    effective_grain_reduction: Option<String>,
    effective_grain_strength: Option<f64>,
    effective_grain_scale: Option<f64>,
    grain_reduction_enabled: Option<bool>,
    expected_grain_reduction_enabled: Option<bool>,
    grain_reduction_applied_ratio: Option<f64>,
    expected_grain_reduction_applied_ratio_min: Option<f64>,
    grain_reduction_structure_excluded_ratio: Option<f64>,
    expected_grain_reduction_structure_excluded_ratio_min: Option<f64>,
    grain_reduction_flat_luma_p95_reduction_ratio: Option<f64>,
    expected_grain_reduction_flat_luma_p95_reduction_ratio_min: Option<f64>,
    grain_reduction_flat_chroma_p95_reduction_ratio: Option<f64>,
    expected_grain_reduction_flat_chroma_p95_reduction_ratio_min: Option<f64>,
    grain_detail_review_required: Option<bool>,
    expected_grain_detail_review_required: Option<bool>,
    grain_detail_decision_supported: Option<bool>,
    expected_grain_detail_decision_supported: Option<bool>,
    grain_detail_luminance_probe_count: Option<usize>,
    expected_grain_detail_luminance_probe_count_min: Option<usize>,
    grain_detail_chroma_probe_count: Option<usize>,
    expected_grain_detail_chroma_probe_count_min: Option<usize>,
    grain_detail_luminance_p10_retention: Option<f64>,
    expected_grain_detail_luminance_p10_retention_min: Option<f64>,
    grain_detail_chroma_p10_retention: Option<f64>,
    expected_grain_detail_chroma_p10_retention_min: Option<f64>,
    mapping_strategy: Option<String>,
    expected_mapping_strategy: Option<String>,
    post_scale_preserved_ratio: Option<f64>,
    expected_post_scale_preserved_ratio_min: Option<f64>,
    render_luminance_range_p05_p95: Option<f64>,
    expected_render_luminance_range_p05_p95_min: Option<f64>,
    render_review_status: Option<String>,
    expected_render_review_status: Option<String>,
    render_reviewable: Option<bool>,
    expected_render_reviewable: Option<bool>,
    tone_output_confidence_status: Option<String>,
    expected_tone_output_confidence_status: Option<String>,
    tone_output_review_required: Option<bool>,
    expected_tone_output_review_required: Option<bool>,
    tone_output_evidence_confidence: Option<f64>,
    expected_tone_output_evidence_confidence_min: Option<f64>,
    render_to_mapped_luminance_range_ratio: Option<f64>,
    expected_render_to_mapped_luminance_range_ratio_min: Option<f64>,
    post_chroma_compression_clipped_high_ratio_max: Option<f64>,
    expected_post_chroma_compression_clipped_high_ratio_max: Option<f64>,
    post_chroma_compression_clipped_low_ratio_max: Option<f64>,
    expected_post_chroma_compression_clipped_low_ratio_max: Option<f64>,
    reference_patch_evaluation_present: Option<bool>,
    reference_patch_patch_count: Option<usize>,
    reference_patch_selected_rms_delta_e: Option<f64>,
    reference_patch_selected_rms_delta_e2000: Option<f64>,
    reference_patch_delta_e2000_delta_vs_image_derived: Option<f64>,
    reference_patch_max_delta_vs_image_derived: Option<f64>,
    reference_patch_delta_e_max_delta_vs_image_derived: Option<f64>,
    reference_patch_delta_e2000_max_delta_vs_image_derived: Option<f64>,
    reference_patch_selected_regresses_image_derived: Option<bool>,
    reference_patch_hue_family_regressions: Vec<String>,
    expected_reference_patch_evaluation_required: bool,
    expected_reference_patch_count_min: Option<usize>,
    expected_reference_patch_hue_family_regression_count_max: Option<usize>,
    expected_reference_patch_selected_regresses_image_derived: Option<bool>,
    expected_reference_patch_delta_e2000_delta_vs_image_derived_max: Option<f64>,
    expected_reference_patch_max_delta_vs_image_derived_max: Option<f64>,
    expected_reference_patch_delta_e_max_delta_vs_image_derived_max: Option<f64>,
    expected_reference_patch_delta_e2000_max_delta_vs_image_derived_max: Option<f64>,
    expected_reference_patch_rms_delta_e_max: Option<f64>,
    expected_reference_patch_rms_delta_e2000_max: Option<f64>,
    debug_artifact_count: Option<usize>,
    debug_artifact_invalid_count: Option<usize>,
    debug_artifact_kinds: Vec<String>,
    expected_debug_artifacts_required: bool,
    expected_debug_artifact_kinds_required: Vec<String>,
    issues: Vec<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct FixtureSuiteCoverageContext {
    validation_ready: bool,
    issues: Vec<String>,
    action_items: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RollInventorySummary {
    status: String,
    roll_dir: String,
    roll_name: String,
    frame_count: usize,
    usable_frame_count: usize,
    unreadable_frame_count: usize,
    sequence_gap_count: usize,
    sequence_gaps: Vec<RollSequenceGap>,
    frames: Vec<RollFrameInspection>,
    issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RollFrameInspection {
    name: String,
    stem: String,
    path: String,
    file_size_bytes: Option<u64>,
    status: String,
    usable: bool,
    width: Option<usize>,
    height: Option<usize>,
    color_type: Option<String>,
    source_bits_per_sample: Option<u8>,
    source_channel_count: Option<usize>,
    source_has_alpha: Option<bool>,
    sequence_prefix: Option<String>,
    sequence_number: Option<usize>,
    sequence_width: Option<usize>,
    positive_input_probe: Option<positive_input::PositiveInputInspection>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RollSequenceGap {
    prefix: String,
    width: usize,
    start: usize,
    end: usize,
    count: usize,
    missing: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RollSuiteSummary {
    status: String,
    roll_dir: String,
    roll_name: String,
    frame_count: usize,
    passed_count: usize,
    review_required_count: usize,
    failed_count: usize,
    output_dir: String,
    roll_base_color: Option<[f64; 3]>,
    roll_base_source: Option<String>,
    roll_base_confidence: Option<f64>,
    roll_base_frame_count: usize,
    roll_base_candidate_count: usize,
    roll_base_rejected_dark_candidate_count: usize,
    roll_base_high_transmittance_envelope: Option<[f64; 3]>,
    roll_base_reason: Option<String>,
    roll_base_clusters: Vec<RollBaseClusterSummary>,
    review: RollSuiteReviewSummary,
    quality: RollSuiteQualitySummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparison: Option<RollSuiteComparison>,
    inventory: RollInventorySummary,
    frames: Vec<RollSuiteEntry>,
    issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RollSuiteReviewSummary {
    frame_count: usize,
    render_reviewable_count: usize,
    render_not_reviewable_count: usize,
    render_reviewable_unknown_count: usize,
    tone_output_evaluated_count: usize,
    tone_output_review_required_count: usize,
    tone_output_unknown_count: usize,
    candidate_safe_count: usize,
    candidate_review_required_count: usize,
    candidate_unknown_count: usize,
    tone_color_trusted_count: usize,
    tone_color_review_required_count: usize,
    tone_color_unknown_count: usize,
    reference_patch_evaluation_present_count: usize,
    reference_patch_evaluation_missing_count: usize,
    reference_patch_evaluation_unknown_count: usize,
    render_review_status_counts: Vec<RollSuiteValueCount>,
    tone_output_confidence_status_counts: Vec<RollSuiteValueCount>,
    tone_output_review_reason_counts: Vec<RollSuiteValueCount>,
    candidate_risk_counts: Vec<RollSuiteValueCount>,
    tone_color_trust_state_counts: Vec<RollSuiteValueCount>,
    calibration_status_counts: Vec<RollSuiteValueCount>,
    issue_counts: Vec<RollSuiteValueCount>,
}

#[derive(Debug, Clone, Serialize)]
struct RollSuiteValueCount {
    value: String,
    count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RollSuiteComparison {
    baseline_path: String,
    status: String,
    issues: Vec<String>,
    frame_count_delta: isize,
    review_required_count_delta: isize,
    failed_count_delta: isize,
    baseline_only_frames: Vec<String>,
    current_only_frames: Vec<String>,
    quality: RollSuiteQualityComparison,
    frames: Vec<RollSuiteFrameComparison>,
}

#[derive(Debug, Clone, Serialize)]
struct RollSuiteQualityComparison {
    noise_reduction_enabled_count_delta: Option<isize>,
    grain_detail_evaluated_count_delta: Option<isize>,
    grain_detail_decision_supported_count_delta: Option<isize>,
    grain_detail_review_required_count_delta: Option<isize>,
    grain_detail_luminance_supported_count_delta: Option<isize>,
    grain_detail_chroma_supported_count_delta: Option<isize>,
    grain_detail_luminance_p10_retention_mean_delta: Option<f64>,
    grain_detail_luminance_p10_retention_min_delta: Option<f64>,
    grain_detail_chroma_p10_retention_mean_delta: Option<f64>,
    grain_detail_chroma_p10_retention_min_delta: Option<f64>,
    render_luminance_range_p05_p95_mean_delta: Option<f64>,
    render_luminance_range_p05_p95_min_delta: Option<f64>,
    render_luminance_range_p05_p95_max_delta: Option<f64>,
    midtone_luminance_p50_mean_delta: Option<f64>,
    midtone_luminance_p50_min_delta: Option<f64>,
    midtone_luminance_p50_max_delta: Option<f64>,
    midtone_luminance_p50_range_delta: Option<f64>,
    shadow_saturation_p95_mean_delta: Option<f64>,
    shadow_saturation_p95_max_delta: Option<f64>,
    midtone_neutral_saturation_p95_mean_delta: Option<f64>,
    midtone_neutral_saturation_p95_max_delta: Option<f64>,
    bright_neutral_saturation_p95_mean_delta: Option<f64>,
    bright_neutral_saturation_p95_max_delta: Option<f64>,
    midtone_neutral_rgb_balance_delta_mean_delta: Option<f64>,
    midtone_neutral_rgb_balance_delta_max_delta: Option<f64>,
    bright_neutral_rgb_balance_delta_mean_delta: Option<f64>,
    bright_neutral_rgb_balance_delta_max_delta: Option<f64>,
    high_frequency_chroma_residual_p95_mean_delta: Option<f64>,
    high_frequency_chroma_residual_p95_max_delta: Option<f64>,
    high_frequency_flat_chroma_residual_p95_mean_delta: Option<f64>,
    high_frequency_flat_chroma_residual_p95_max_delta: Option<f64>,
    high_frequency_chroma_to_luma_p95_ratio_mean_delta: Option<f64>,
    high_frequency_flat_chroma_to_luma_p95_ratio_mean_delta: Option<f64>,
    noise_reduction_saturation_limited_ratio_mean_delta: Option<f64>,
    noise_reduction_mean_abs_chroma_delta_mean_delta: Option<f64>,
    colorspace_post_scale_preserved_ratio_min_delta: Option<f64>,
    post_chroma_compression_clipped_high_max_delta: Option<f64>,
    post_chroma_compression_clipped_low_max_delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct RollSuiteFrameComparison {
    name: String,
    baseline_status: Option<String>,
    current_status: Option<String>,
    status_changed: bool,
    baseline_render_review_status: Option<String>,
    current_render_review_status: Option<String>,
    render_review_status_changed: bool,
    baseline_render_reviewable: Option<bool>,
    current_render_reviewable: Option<bool>,
    render_reviewable_changed: bool,
    baseline_tone_output_confidence_status: Option<String>,
    current_tone_output_confidence_status: Option<String>,
    tone_output_confidence_status_changed: bool,
    baseline_tone_output_review_required: Option<bool>,
    current_tone_output_review_required: Option<bool>,
    tone_output_review_required_changed: bool,
    tone_output_evidence_confidence_delta: Option<f64>,
    tone_output_render_to_mapped_luminance_range_ratio_delta: Option<f64>,
    tone_output_maximum_post_tone_high_clip_ratio_delta: Option<f64>,
    tone_output_maximum_post_tone_low_clip_ratio_delta: Option<f64>,
    baseline_candidate_risk: Option<String>,
    current_candidate_risk: Option<String>,
    candidate_risk_changed: bool,
    baseline_grain_detail_review_required: Option<bool>,
    current_grain_detail_review_required: Option<bool>,
    grain_detail_review_required_changed: bool,
    baseline_grain_detail_decision_supported: Option<bool>,
    current_grain_detail_decision_supported: Option<bool>,
    grain_detail_decision_supported_changed: bool,
    baseline_grain_detail_luminance_supported: Option<bool>,
    current_grain_detail_luminance_supported: Option<bool>,
    grain_detail_luminance_supported_changed: bool,
    baseline_grain_detail_chroma_supported: Option<bool>,
    current_grain_detail_chroma_supported: Option<bool>,
    grain_detail_chroma_supported_changed: bool,
    grain_detail_luminance_p10_retention_delta: Option<f64>,
    grain_detail_chroma_p10_retention_delta: Option<f64>,
    render_luminance_range_p05_p95_delta: Option<f64>,
    midtone_luminance_p50_delta: Option<f64>,
    shadow_saturation_p95_delta: Option<f64>,
    midtone_neutral_saturation_p95_delta: Option<f64>,
    bright_neutral_saturation_p95_delta: Option<f64>,
    midtone_neutral_rgb_balance_delta_delta: Option<f64>,
    bright_neutral_rgb_balance_delta_delta: Option<f64>,
    high_frequency_chroma_residual_p95_delta: Option<f64>,
    high_frequency_flat_chroma_residual_p95_delta: Option<f64>,
    colorspace_post_scale_preserved_ratio_delta: Option<f64>,
    post_chroma_compression_clipped_high_max_delta: Option<f64>,
    post_chroma_compression_clipped_low_max_delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct RollSuiteQualitySummary {
    frame_count: usize,
    high_frequency_frame_count: usize,
    high_frequency_luma_residual_p95_mean: Option<f64>,
    high_frequency_luma_residual_p95_max: Option<f64>,
    high_frequency_chroma_residual_p95_mean: Option<f64>,
    high_frequency_chroma_residual_p95_max: Option<f64>,
    high_frequency_chroma_to_luma_p95_ratio_mean: Option<f64>,
    high_frequency_chroma_to_luma_p95_ratio_max: Option<f64>,
    high_frequency_flat_frame_count: usize,
    high_frequency_flat_sample_ratio_mean: Option<f64>,
    high_frequency_flat_luma_residual_p95_mean: Option<f64>,
    high_frequency_flat_luma_residual_p95_max: Option<f64>,
    high_frequency_flat_chroma_residual_p95_mean: Option<f64>,
    high_frequency_flat_chroma_residual_p95_max: Option<f64>,
    high_frequency_flat_chroma_to_luma_p95_ratio_mean: Option<f64>,
    high_frequency_flat_chroma_to_luma_p95_ratio_max: Option<f64>,
    noise_reduction_enabled_count: usize,
    noise_reduction_applied_ratio_mean: Option<f64>,
    noise_reduction_applied_ratio_max: Option<f64>,
    noise_reduction_structure_excluded_ratio_mean: Option<f64>,
    noise_reduction_structure_excluded_ratio_min: Option<f64>,
    noise_reduction_texture_limited_ratio_mean: Option<f64>,
    noise_reduction_saturation_limited_ratio_mean: Option<f64>,
    noise_reduction_mean_abs_chroma_delta_mean: Option<f64>,
    noise_reduction_mean_abs_luma_delta_mean: Option<f64>,
    noise_reduction_max_abs_chroma_delta_max: Option<f64>,
    noise_reduction_max_abs_luma_delta_max: Option<f64>,
    grain_detail_evaluated_count: usize,
    grain_detail_decision_supported_count: usize,
    grain_detail_review_required_count: usize,
    grain_detail_luminance_supported_count: usize,
    grain_detail_chroma_supported_count: usize,
    grain_detail_luminance_probe_count_min: Option<usize>,
    grain_detail_chroma_probe_count_min: Option<usize>,
    grain_detail_luminance_median_retention_mean: Option<f64>,
    grain_detail_luminance_median_retention_min: Option<f64>,
    grain_detail_luminance_p10_retention_mean: Option<f64>,
    grain_detail_luminance_p10_retention_min: Option<f64>,
    grain_detail_chroma_median_retention_mean: Option<f64>,
    grain_detail_chroma_median_retention_min: Option<f64>,
    grain_detail_chroma_p10_retention_mean: Option<f64>,
    grain_detail_chroma_p10_retention_min: Option<f64>,
    colorspace_post_scale_preserved_ratio_mean: Option<f64>,
    colorspace_post_scale_preserved_ratio_min: Option<f64>,
    render_luminance_range_p05_p95_mean: Option<f64>,
    render_luminance_range_p05_p95_min: Option<f64>,
    render_luminance_range_p05_p95_max: Option<f64>,
    midtone_luminance_p50_mean: Option<f64>,
    midtone_luminance_p50_min: Option<f64>,
    midtone_luminance_p50_max: Option<f64>,
    midtone_luminance_p50_range: Option<f64>,
    shadow_saturation_p95_mean: Option<f64>,
    shadow_saturation_p95_max: Option<f64>,
    shadow_visible_saturation_p95_mean: Option<f64>,
    shadow_visible_saturation_p95_max: Option<f64>,
    midtone_neutral_saturation_p95_mean: Option<f64>,
    midtone_neutral_saturation_p95_max: Option<f64>,
    bright_neutral_saturation_p95_mean: Option<f64>,
    bright_neutral_saturation_p95_max: Option<f64>,
    shadow_rgb_balance_delta_mean: Option<f64>,
    shadow_rgb_balance_delta_max: Option<f64>,
    shadow_visible_rgb_balance_delta_mean: Option<f64>,
    shadow_visible_rgb_balance_delta_max: Option<f64>,
    midtone_rgb_balance_delta_mean: Option<f64>,
    midtone_rgb_balance_delta_max: Option<f64>,
    midtone_neutral_rgb_balance_delta_mean: Option<f64>,
    midtone_neutral_rgb_balance_delta_max: Option<f64>,
    bright_neutral_rgb_balance_delta_mean: Option<f64>,
    bright_neutral_rgb_balance_delta_max: Option<f64>,
    highlight_chroma_compressed_ratio_mean: Option<f64>,
    highlight_chroma_compressed_ratio_max: Option<f64>,
    shadow_chroma_compressed_ratio_mean: Option<f64>,
    shadow_chroma_compressed_ratio_max: Option<f64>,
    post_chroma_compression_clipped_high_max: Option<f64>,
    post_chroma_compression_clipped_low_max: Option<f64>,
}

#[derive(Debug, Clone)]
struct RollBaseEstimate {
    color: [f64; 3],
    source: String,
    confidence: f64,
    frame_count: usize,
    candidate_count: usize,
    rejected_dark_candidate_count: usize,
    high_transmittance_envelope: [f64; 3],
    reason: String,
    clusters: Vec<RollBaseClusterSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct RollBaseClusterSummary {
    color: [f64; 3],
    frame_count: usize,
    mean_luminance: f64,
    max_relative_luminance_spread: f64,
    source_counts: Vec<RollBaseSourceCount>,
}

#[derive(Debug, Clone, Serialize)]
struct RollBaseSourceCount {
    source: String,
    count: usize,
}

#[derive(Debug, Serialize)]
struct RollSuiteEntry {
    name: String,
    source_path: String,
    status: String,
    output_dir: String,
    source_width: Option<usize>,
    source_height: Option<usize>,
    source_color_type: Option<String>,
    source_bits_per_sample: Option<u8>,
    output_path: Option<String>,
    output_width: Option<usize>,
    output_height: Option<usize>,
    output_color_space: Option<String>,
    output_file_icc_profile_matches_report: Option<bool>,
    delivery_artifacts_intact: Option<bool>,
    delivery_artifact_issues: Vec<String>,
    stale_render_artifact_count: Option<usize>,
    report_path: Option<String>,
    summary_json_path: Option<String>,
    summary_md_path: Option<String>,
    stitch_decision: Option<String>,
    base_confidence: Option<f64>,
    raw_base_confidence: Option<f64>,
    base_estimate_source: Option<String>,
    input_base_confidence: Option<f64>,
    render_review_status: Option<String>,
    render_reviewable: Option<bool>,
    tone_output_evidence_evaluated: Option<bool>,
    tone_output_evidence_confidence: Option<f64>,
    tone_output_confidence_status: Option<String>,
    tone_output_review_required: Option<bool>,
    tone_output_review_reason: Option<String>,
    confidence_limited_by_tone_output_evidence: Option<bool>,
    input_luminance_range_p05_p95: Option<f64>,
    mapped_luminance_range_p05_p95: Option<f64>,
    render_to_mapped_luminance_range_ratio: Option<f64>,
    maximum_post_tone_high_clip_ratio: Option<f64>,
    maximum_post_tone_low_clip_ratio: Option<f64>,
    positive_input_likely_negative_like: Option<bool>,
    positive_input_accepted_high_warm_score: Option<bool>,
    positive_input_orange_mask_score: Option<f64>,
    positive_input_reason: Option<String>,
    base_color_override_applied: Option<bool>,
    density_confidence: Option<f64>,
    render_input_source: Option<String>,
    render_input_reason: Option<String>,
    mapping_strategy: Option<String>,
    selected_mapping_reason: Option<String>,
    selected_candidate: Option<String>,
    selected_candidate_rank: Option<usize>,
    selected_quality_score: Option<f64>,
    candidate_risk: Option<String>,
    tone_color_trust_state: Option<String>,
    colorspace_pre_scale_preserved_ratio: Option<f64>,
    colorspace_post_scale_preserved_ratio: Option<f64>,
    render_luminance_p05: Option<f64>,
    render_luminance_p50: Option<f64>,
    render_luminance_p95: Option<f64>,
    render_luminance_range_p05_p95: Option<f64>,
    midtone_luminance_p50: Option<f64>,
    shadow_saturation_p95: Option<f64>,
    shadow_visible_pixel_count: Option<usize>,
    shadow_visible_saturation_p95: Option<f64>,
    midtone_neutral_pixel_count: Option<usize>,
    midtone_neutral_saturation_p95: Option<f64>,
    bright_neutral_saturation_p95: Option<f64>,
    shadow_rgb_median: Option<Vec<f64>>,
    shadow_visible_rgb_median: Option<Vec<f64>>,
    midtone_rgb_median: Option<Vec<f64>>,
    midtone_neutral_rgb_median: Option<Vec<f64>>,
    bright_neutral_rgb_median: Option<Vec<f64>>,
    shadow_rgb_balance_delta: Option<f64>,
    shadow_visible_rgb_balance_delta: Option<f64>,
    midtone_rgb_balance_delta: Option<f64>,
    midtone_neutral_rgb_balance_delta: Option<f64>,
    bright_neutral_rgb_balance_delta: Option<f64>,
    highlight_chroma_compressed_ratio: Option<f64>,
    highlight_neutral_chroma_compressed_ratio: Option<f64>,
    shadow_chroma_compressed_ratio: Option<f64>,
    post_chroma_compression_clipped_high_max: Option<f64>,
    post_chroma_compression_clipped_low_max: Option<f64>,
    high_frequency_luma_residual_p95: Option<f64>,
    high_frequency_chroma_residual_p95: Option<f64>,
    high_frequency_chroma_to_luma_p95_ratio: Option<f64>,
    high_frequency_flat_sample_count: Option<usize>,
    high_frequency_flat_sample_ratio: Option<f64>,
    high_frequency_flat_luma_residual_p95: Option<f64>,
    high_frequency_flat_chroma_residual_p95: Option<f64>,
    high_frequency_flat_chroma_to_luma_p95_ratio: Option<f64>,
    noise_reduction_enabled: Option<bool>,
    noise_reduction_applied_ratio: Option<f64>,
    noise_reduction_structure_gate_start: Option<f64>,
    noise_reduction_structure_gate_end: Option<f64>,
    noise_reduction_structure_excluded_ratio: Option<f64>,
    noise_reduction_texture_limited_ratio: Option<f64>,
    noise_reduction_saturation_limited_ratio: Option<f64>,
    noise_reduction_mean_abs_chroma_delta: Option<f64>,
    noise_reduction_max_abs_chroma_delta: Option<f64>,
    noise_reduction_mean_abs_luma_delta: Option<f64>,
    noise_reduction_max_abs_luma_delta: Option<f64>,
    grain_detail_evaluated: Option<bool>,
    grain_detail_decision_supported: Option<bool>,
    grain_detail_review_required: Option<bool>,
    grain_detail_luminance_supported: Option<bool>,
    grain_detail_chroma_supported: Option<bool>,
    grain_detail_luminance_probe_count: Option<usize>,
    grain_detail_chroma_probe_count: Option<usize>,
    grain_detail_luminance_median_retention: Option<f64>,
    grain_detail_luminance_p10_retention: Option<f64>,
    grain_detail_chroma_median_retention: Option<f64>,
    grain_detail_chroma_p10_retention: Option<f64>,
    calibration_status: Option<String>,
    calibration_source: Option<String>,
    calibration_acceptance_status: Option<String>,
    calibration_color_mapping_applied: Option<bool>,
    calibration_confidence: Option<f64>,
    reference_patch_evaluation_present: Option<bool>,
    debug_artifact_count: Option<usize>,
    debug_artifact_invalid_count: Option<usize>,
    debug_artifact_kinds: Vec<String>,
    issues: Vec<String>,
    error: Option<String>,
}

fn run_roll_suite_child(cli: &ValidationCli) -> Result<(), Box<dyn std::error::Error>> {
    let component1 = cli
        .component1
        .clone()
        .ok_or("--roll-suite-child requires --component1")?;
    let component2 = cli
        .component2
        .clone()
        .ok_or("--roll-suite-child requires --component2")?;
    let inputs = if component1 == component2 && cli.force_no_stitch {
        vec![component1]
    } else {
        vec![component1, component2]
    };
    let pipeline_cli = PipelineCli {
        inputs,
        output_dir: cli.output_dir.clone(),
        calibration_profile: cli.calibration_profile.clone(),
        calibration_library: cli.calibration_library.clone(),
        scanner_profile: cli.scanner_profile.clone(),
        roll_profile: cli.roll_profile.clone(),
        film_stock: cli.film_stock.clone(),
        base_color: cli.base_color.clone(),
        base_color_source: cli.base_color_source.clone(),
        base_color_confidence: cli.base_color_confidence,
        base_color_reason: cli.base_color_reason.clone(),
        color_mode: cli.color_mode,
        render_input: cli.render_input,
        input_mode: cli.input_mode,
        render_intent: cli.render_intent,
        quality_mode: cli.quality_mode,
        white_balance: cli.white_balance,
        geometry: cli.geometry,
        grain: cli.grain,
        write_master: cli.write_master,
        review_sidecar: cli.review_sidecar.clone(),
        write_review_sidecar: cli.write_review_sidecar.clone(),
        debug: cli.debug,
        force_stitch: cli.force_stitch,
        force_no_stitch: cli.force_no_stitch,
        transform: cli.transform.clone(),
        ica_max_iter: cli.ica_max_iter,
        ica_tol: cli.ica_tol,
        bit_depth: cli.bit_depth,
        use_opencv: cli.use_opencv,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&pipeline_cli)?;
    let report_path = pipeline_cli.output_dir.join("report.json");
    report.save(&report_path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = ValidationCli::parse();
    validate_input_mode_options(cli.input_mode, cli.render_input)?;
    cli.geometry.validate()?;
    cli.white_balance.validate()?;
    if cli.roll_suite_child {
        return run_roll_suite_child(&cli);
    }

    if cli.require_reviewable
        && (cli.list_fixtures
            || cli.write_orientation_review.is_some()
            || cli.write_render_review.is_some()
            || cli.fixture_coverage
            || cli.synthetic_color_suite
            || (cli.roll_inventory && !cli.roll_suite))
    {
        return Err("--require-reviewable requires a direct render/report, --fixture-suite, or --roll-suite; inventory, coverage, listing, orientation-review, render-review-draft, and synthetic-decision modes do not produce final renders".into());
    }

    let registry = load_fixture_registry(cli.fixture_registry.as_ref())?;
    let fixtures = &registry.fixtures;

    if cli.write_orientation_review.is_some()
        && (cli.list_fixtures
            || cli.synthetic_color_suite
            || cli.fixture_coverage
            || cli.fixture_suite
            || cli.roll_inventory
            || cli.roll_suite
            || cli.report.is_some()
            || cli.write_render_review.is_some())
    {
        return Err("--write-orientation-review is a standalone decode/review mode and cannot be combined with listing, report, coverage, suite, or roll modes".into());
    }
    if cli.write_render_review.is_some() {
        if cli.report.is_none() {
            return Err("--write-render-review requires --report".into());
        }
        if cli.list_fixtures
            || cli.synthetic_color_suite
            || cli.fixture_coverage
            || cli.fixture_suite
            || cli.roll_inventory
            || cli.roll_suite
            || cli.write_orientation_review.is_some()
            || cli.compare_report.is_some()
            || cli.compare_summary.is_some()
            || cli.write_summary_baseline.is_some()
        {
            return Err("--write-render-review is a standalone report-review draft mode and cannot be combined with listing, comparison, baseline, coverage, suite, inventory, or orientation-review modes".into());
        }
    }
    if cli.render_review_grain_control_report.is_some() && cli.write_render_review.is_none() {
        return Err("--render-review-grain-control-report requires --write-render-review".into());
    }
    if cli.write_fixture_hash_registry.is_some() && !cli.fixture_coverage {
        return Err("--write-fixture-hash-registry requires --fixture-coverage".into());
    }
    if cli.write_fixture_suite_baselines && !cli.fixture_suite {
        return Err("--write-fixture-suite-baselines requires --fixture-suite".into());
    }
    if !cli.fixture_suite && !cli.fixture_suite_fixtures.is_empty() {
        return Err("--fixture-suite-fixture requires --fixture-suite".into());
    }
    if cli.overwrite_fixture_suite_baselines && !cli.write_fixture_suite_baselines {
        return Err(
            "--overwrite-fixture-suite-baselines requires --write-fixture-suite-baselines".into(),
        );
    }
    if cli.strict && cli.write_fixture_suite_baselines {
        return Err("--write-fixture-suite-baselines is a corpus-building mode; omit --strict and run a separate strict fixture-suite after accepting the baselines".into());
    }
    if !cli.roll_suite
        && !cli.roll_suite_frames.is_empty()
        && cli.write_roll_fixture_registry.is_none()
        && cli.write_roll_fixture_metadata_template.is_none()
        && cli.write_roll_contact_sheet.is_none()
    {
        return Err(
            "--roll-suite-frame requires --roll-suite, --write-roll-fixture-registry, --write-roll-fixture-metadata-template, or --write-roll-contact-sheet".into(),
        );
    }
    if cli.write_roll_fixture_registry.is_none()
        && cli.write_roll_fixture_metadata_template.is_none()
        && (!cli.roll_fixture_scene_tags.is_empty() || !cli.roll_fixture_exposure_tags.is_empty())
    {
        return Err("--roll-fixture-scene-tag and --roll-fixture-exposure-tag require --write-roll-fixture-registry or --write-roll-fixture-metadata-template".into());
    }
    if cli.roll_fixture_metadata.is_some()
        && cli.write_roll_fixture_registry.is_none()
        && cli.write_roll_fixture_metadata_template.is_none()
    {
        return Err("--roll-fixture-metadata requires --write-roll-fixture-registry or --write-roll-fixture-metadata-template".into());
    }
    if cli.write_roll_contact_sheet_index.is_some() && cli.write_roll_contact_sheet.is_none() {
        return Err("--write-roll-contact-sheet-index requires --write-roll-contact-sheet".into());
    }

    if cli.list_fixtures {
        for (name, fixture) in fixtures {
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                name,
                fixture.component1.display(),
                fixture
                    .component2
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_default(),
                fixture
                    .summary_baseline
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_default(),
                fixture_calibration_label(fixture),
                fixture.scanner_profile.as_deref().unwrap_or(""),
                fixture.roll_profile.as_deref().unwrap_or(""),
                fixture.film_stock.as_deref().unwrap_or(""),
                fixture.scene_tags.join(","),
                fixture.exposure_tags.join(","),
                fixture.reference_evidence.join(","),
                fixture.calibration_case.as_deref().unwrap_or(""),
                fixture.description.as_deref().unwrap_or("")
            );
        }
        return Ok(());
    }

    if let Some(output_dir) = &cli.write_orientation_review {
        let (manifest, json_path, md_path) =
            write_orientation_review_package(&cli, fixtures, output_dir)?;
        if cli.print_json_summary {
            println!(
                "{}",
                serde_json::to_string_pretty(&manifest)
                    .expect("orientation review manifest should serialize")
            );
        }
        if !cli.quiet {
            println!("orientation_review_json={}", json_path.display());
            println!("orientation_review_md={}", md_path.display());
            println!(
                "orientation_review_status={} components={} action=visually_approve_each_preview_then_copy_the_draft_and_set_upright_approved_true",
                manifest.review_status,
                manifest.orientation_components_expected.len()
            );
        }
        return Ok(());
    }

    if let Some(output_dir) = &cli.write_render_review {
        let report_path = cli
            .report
            .as_deref()
            .expect("--write-render-review validation requires --report");
        let draft = render_review::write_render_review_draft_with_grain_control(
            &cli.fixture,
            report_path,
            cli.render_review_grain_control_report.as_deref(),
            output_dir,
        )?;
        if cli.print_json_summary {
            println!(
                "{}",
                serde_json::to_string_pretty(&draft.manifest)
                    .expect("render review manifest should serialize")
            );
        }
        if !cli.quiet {
            println!("render_review_json={}", draft.json_path.display());
            println!("render_review_md={}", draft.markdown_path.display());
            println!(
                "render_review_status={} technical_delivery_reviewable={} artifacts={} grain_control_bound={} action=inspect_exact_artifacts_then_complete_all_applicable_human_decisions",
                draft.manifest.review_status,
                draft.manifest.technical_delivery_reviewable,
                draft.manifest.artifacts.len(),
                draft.manifest.grain_reduction_control.is_some()
            );
        }
        return Ok(());
    }

    if cli.roll_inventory || cli.roll_suite {
        validate_roll_inputs(&cli)?;
        if cli.roll_suite {
            let mut summary = run_roll_suite(&cli);
            if let Some(compare_path) = &cli.compare_roll_suite {
                let baseline_json = std::fs::read_to_string(compare_path)?;
                let baseline_summary = serde_json::from_str::<serde_json::Value>(&baseline_json)?;
                let current_summary =
                    serde_json::to_value(&summary).expect("roll suite should serialize to value");
                summary.comparison = Some(compare_roll_suites(
                    compare_path.to_string_lossy().to_string(),
                    &baseline_summary,
                    &current_summary,
                ));
            }
            let output_root = roll_mode_output_dir(&cli);
            let json_path = cli
                .summary_json
                .clone()
                .unwrap_or_else(|| output_root.join("roll-suite.json"));
            let md_path = cli
                .summary_md
                .clone()
                .unwrap_or_else(|| output_root.join("roll-suite.md"));
            let json_summary =
                serde_json::to_string_pretty(&summary).expect("roll suite should serialize");
            write_text(&json_path, &json_summary)?;
            write_text(&md_path, &roll_suite_to_markdown(&summary))?;
            if cli.print_json_summary {
                println!("{json_summary}");
            }
            if !cli.quiet {
                println!("roll_suite_json={}", json_path.display());
                println!("roll_suite_md={}", md_path.display());
                println!(
                    "roll_suite_status={} passed={} review_required={} failed={} issues={}",
                    summary.status,
                    summary.passed_count,
                    summary.review_required_count,
                    summary.failed_count,
                    if summary.issues.is_empty() {
                        "none".to_string()
                    } else {
                        summary.issues.join(",")
                    }
                );
                if let Some(comparison) = &summary.comparison {
                    println!(
                        "roll_suite_comparison_status={} issues={}",
                        comparison.status,
                        if comparison.issues.is_empty() {
                            "none".to_string()
                        } else {
                            comparison.issues.join(",")
                        }
                    );
                }
            }
            if cli.require_reviewable {
                let rejected = summary
                    .frames
                    .iter()
                    .filter(|frame| {
                        !final_render_is_reviewable(
                            frame.render_review_status.as_deref(),
                            frame.render_reviewable,
                        ) || frame.delivery_artifacts_intact != Some(true)
                    })
                    .map(|frame| frame.name.clone())
                    .collect::<Vec<_>>();
                if !rejected.is_empty() {
                    return Err(format!(
                        "--require-reviewable rejected roll-suite frame(s): {}; reports and roll summary artifacts were retained",
                        rejected.join(", ")
                    )
                    .into());
                }
            }
            let mut all_issues = summary.issues.clone();
            if let Some(comparison) = &summary.comparison {
                all_issues.extend(comparison.issues.iter().cloned());
            }
            let selected_failures = selected_failures(&all_issues, &cli.fail_on);
            if cli.strict && !summary.issues.is_empty() {
                return Err(format!(
                    "strict roll suite failed with issue(s): {}",
                    summary.issues.join(", ")
                )
                .into());
            }
            if let Some(comparison) = &summary.comparison {
                if cli.strict && !comparison.issues.is_empty() {
                    return Err(format!(
                        "strict roll suite comparison failed with issue(s): {}",
                        comparison.issues.join(", ")
                    )
                    .into());
                }
            }
            if !selected_failures.is_empty() {
                return Err(format!(
                    "--fail-on matched roll suite issue(s): {}",
                    selected_failures.join(", ")
                )
                .into());
            }
            return Ok(());
        }

        let summary = run_roll_inventory(&cli);
        let output_root = roll_mode_output_dir(&cli);
        let json_path = cli
            .summary_json
            .clone()
            .unwrap_or_else(|| output_root.join("roll-inventory.json"));
        let md_path = cli
            .summary_md
            .clone()
            .unwrap_or_else(|| output_root.join("roll-inventory.md"));
        let json_summary =
            serde_json::to_string_pretty(&summary).expect("roll inventory should serialize");
        write_text(&json_path, &json_summary)?;
        write_text(&md_path, &roll_inventory_to_markdown(&summary))?;
        if let Some(registry_path) = &cli.write_roll_fixture_registry {
            let registry = roll_fixture_registry_scaffold(&cli, &summary)?;
            let mut registry_json =
                serde_json::to_value(&registry).expect("roll fixture registry should serialize");
            strip_empty_registry_snapshot_values(&mut registry_json);
            let registry_contents = serde_json::to_string_pretty(&registry_json)
                .expect("roll fixture registry JSON should serialize");
            write_text(registry_path, &registry_contents)?;
        }
        if let Some(metadata_template_path) = &cli.write_roll_fixture_metadata_template {
            let metadata = roll_fixture_metadata_template(&cli, &summary)?;
            let mut metadata_json =
                serde_json::to_value(&metadata).expect("roll fixture metadata should serialize");
            strip_empty_registry_snapshot_values(&mut metadata_json);
            let metadata_contents = serde_json::to_string_pretty(&metadata_json)
                .expect("roll fixture metadata JSON should serialize");
            write_text(metadata_template_path, &metadata_contents)?;
        }
        if let Some(contact_sheet_path) = &cli.write_roll_contact_sheet {
            write_roll_contact_sheet(
                &cli,
                &summary,
                contact_sheet_path,
                cli.write_roll_contact_sheet_index.as_deref(),
            )?;
        }
        if cli.print_json_summary {
            println!("{json_summary}");
        }
        if !cli.quiet {
            println!("roll_inventory_json={}", json_path.display());
            println!("roll_inventory_md={}", md_path.display());
            if let Some(registry_path) = &cli.write_roll_fixture_registry {
                println!("roll_fixture_registry={}", registry_path.display());
            }
            if let Some(metadata_template_path) = &cli.write_roll_fixture_metadata_template {
                println!(
                    "roll_fixture_metadata_template={}",
                    metadata_template_path.display()
                );
            }
            if let Some(contact_sheet_path) = &cli.write_roll_contact_sheet {
                println!("roll_contact_sheet={}", contact_sheet_path.display());
            }
            if let Some(contact_sheet_index_path) = &cli.write_roll_contact_sheet_index {
                println!(
                    "roll_contact_sheet_index={}",
                    contact_sheet_index_path.display()
                );
            }
            println!(
                "roll_inventory_status={} frames={} usable={} issues={}",
                summary.status,
                summary.frame_count,
                summary.usable_frame_count,
                if summary.issues.is_empty() {
                    "none".to_string()
                } else {
                    summary.issues.join(",")
                }
            );
        }
        let selected_failures = selected_failures(&summary.issues, &cli.fail_on);
        if cli.strict && !summary.issues.is_empty() {
            return Err(format!(
                "strict roll inventory failed with issue(s): {}",
                summary.issues.join(", ")
            )
            .into());
        }
        if !selected_failures.is_empty() {
            return Err(format!(
                "--fail-on matched roll inventory issue(s): {}",
                selected_failures.join(", ")
            )
            .into());
        }
        return Ok(());
    }

    if cli.compare_roll_suite.is_some() {
        return Err("--compare-roll-suite requires --roll-suite".into());
    }

    if cli.fixture_coverage {
        validate_fixture_hash_registry_writer(&cli)?;
        let compute_fixture_hashes =
            cli.compute_fixture_hashes || cli.write_fixture_hash_registry.is_some();
        let summary = fixture_coverage_summary(
            fixtures,
            &registry.coverage_requirements,
            compute_fixture_hashes,
        );
        let json_path = cli
            .summary_json
            .clone()
            .unwrap_or_else(|| fixture_set_output_dir(&cli).join("fixture-coverage.json"));
        let md_path = cli
            .summary_md
            .clone()
            .unwrap_or_else(|| fixture_set_output_dir(&cli).join("fixture-coverage.md"));
        let json_summary =
            serde_json::to_string_pretty(&summary).expect("fixture coverage should serialize");
        write_text(&json_path, &json_summary)?;
        write_text(&md_path, &fixture_coverage_to_markdown(&summary))?;
        if let Some(hash_registry_path) = &cli.write_fixture_hash_registry {
            let snapshot =
                fixture_hash_registry_snapshot(fixtures, &registry.coverage_requirements, &summary);
            let mut snapshot_json = serde_json::to_value(&snapshot)
                .expect("fixture registry snapshot should serialize");
            strip_empty_registry_snapshot_values(&mut snapshot_json);
            let snapshot_contents = serde_json::to_string_pretty(&snapshot_json)
                .expect("fixture registry snapshot JSON should serialize");
            write_text(hash_registry_path, &snapshot_contents)?;
        }
        if cli.print_json_summary {
            println!("{json_summary}");
        }
        if !cli.quiet {
            println!("fixture_coverage_json={}", json_path.display());
            println!("fixture_coverage_md={}", md_path.display());
            if let Some(hash_registry_path) = &cli.write_fixture_hash_registry {
                println!("fixture_hash_registry={}", hash_registry_path.display());
            }
            println!(
                "fixture_coverage_status={} issues={}",
                summary.status,
                if summary.issues.is_empty() {
                    "none".to_string()
                } else {
                    summary.issues.join(",")
                }
            );
        }
        let selected_failures = selected_failures(&summary.issues, &cli.fail_on);
        if cli.strict && !summary.issues.is_empty() {
            return Err(format!(
                "strict fixture coverage failed with issue(s): {}",
                summary.issues.join(", ")
            )
            .into());
        }
        if !selected_failures.is_empty() {
            return Err(format!(
                "--fail-on matched fixture coverage issue(s): {}",
                selected_failures.join(", ")
            )
            .into());
        }
        return Ok(());
    }

    if cli.fixture_suite {
        validate_fixture_suite_inputs(&cli, fixtures)?;
        let summary = run_fixture_suite(&cli, fixtures, &registry.coverage_requirements);
        let json_path = cli
            .summary_json
            .clone()
            .unwrap_or_else(|| fixture_set_output_dir(&cli).join("fixture-suite.json"));
        let md_path = cli
            .summary_md
            .clone()
            .unwrap_or_else(|| fixture_set_output_dir(&cli).join("fixture-suite.md"));
        let json_summary =
            serde_json::to_string_pretty(&summary).expect("fixture suite should serialize");
        write_text(&json_path, &json_summary)?;
        write_text(&md_path, &fixture_suite_to_markdown(&summary))?;
        if cli.print_json_summary {
            println!("{json_summary}");
        }
        if !cli.quiet {
            println!("fixture_suite_json={}", json_path.display());
            println!("fixture_suite_md={}", md_path.display());
            println!(
                "fixture_suite_status={} passed={} review_required={} failed={} issues={}",
                summary.status,
                summary.passed_count,
                summary.review_required_count,
                summary.failed_count,
                if summary.issues.is_empty() {
                    "none".to_string()
                } else {
                    summary.issues.join(",")
                }
            );
        }
        if cli.require_reviewable {
            let rejected = summary
                .fixtures
                .iter()
                .filter(|fixture| {
                    !final_render_is_reviewable(
                        fixture.render_review_status.as_deref(),
                        fixture.render_reviewable,
                    ) || fixture.delivery_artifacts_intact != Some(true)
                })
                .map(|fixture| fixture.name.clone())
                .collect::<Vec<_>>();
            if !rejected.is_empty() {
                return Err(format!(
                    "--require-reviewable rejected fixture-suite output(s): {}; reports and fixture-suite summary artifacts were retained",
                    rejected.join(", ")
                )
                .into());
            }
        }
        let selected_failures = selected_failures(&summary.issues, &cli.fail_on);
        if cli.strict && !summary.issues.is_empty() {
            return Err(format!(
                "strict fixture suite failed with issue(s): {}",
                summary.issues.join(", ")
            )
            .into());
        }
        if !selected_failures.is_empty() {
            return Err(format!(
                "--fail-on matched fixture suite issue(s): {}",
                selected_failures.join(", ")
            )
            .into());
        }
        return Ok(());
    }

    if cli.synthetic_color_suite {
        let summary = run_synthetic_color_suite();
        let json_path = cli.summary_json.clone().unwrap_or_else(|| {
            effective_output_dir(&cli, fixtures).join("synthetic-color-suite.json")
        });
        let md_path = cli.summary_md.clone().unwrap_or_else(|| {
            effective_output_dir(&cli, fixtures).join("synthetic-color-suite.md")
        });
        let json_summary =
            serde_json::to_string_pretty(&summary).expect("synthetic suite should serialize");
        write_text(&json_path, &json_summary)?;
        write_text(&md_path, &synthetic_color_suite_to_markdown(&summary))?;
        if cli.print_json_summary {
            println!("{json_summary}");
        }
        if !cli.quiet {
            println!("synthetic_color_suite_json={}", json_path.display());
            println!("synthetic_color_suite_md={}", md_path.display());
            println!(
                "synthetic_color_suite_status={} issues={}",
                summary.status,
                if summary.issues.is_empty() {
                    "none".to_string()
                } else {
                    summary.issues.join(",")
                }
            );
        }
        let selected_failures = selected_failures(&summary.issues, &cli.fail_on);
        if cli.strict && !summary.issues.is_empty() {
            return Err(format!(
                "strict synthetic color suite failed with issue(s): {}",
                summary.issues.join(", ")
            )
            .into());
        }
        if !selected_failures.is_empty() {
            return Err(format!(
                "--fail-on matched synthetic color suite issue(s): {}",
                selected_failures.join(", ")
            )
            .into());
        }
        return Ok(());
    }

    let mut current_report_path = cli.report.clone();
    let report = if let Some(report_path) = &cli.report {
        let contents = std::fs::read_to_string(report_path)?;
        serde_json::from_str::<PipelineReport>(&contents)?
    } else {
        let inputs = resolve_inputs(&cli, fixtures)?;
        let output_dir = effective_output_dir(&cli, fixtures);
        let input_mode = effective_input_mode(&cli, fixtures)?;
        validate_input_mode_options(input_mode, cli.render_input)?;
        let bit_depth = effective_bit_depth(&cli, fixtures)?;
        let grain = effective_grain(&cli, fixtures)?;
        let geometry = effective_geometry(&cli, fixtures)?;
        let (force_stitch, force_no_stitch) = effective_force_settings(&cli, fixtures)?;
        let pipeline_cli = PipelineCli {
            inputs,
            output_dir,
            calibration_profile: effective_calibration_profile(&cli, fixtures),
            calibration_library: effective_calibration_library(&cli, fixtures),
            scanner_profile: effective_scanner_profile(&cli, fixtures),
            roll_profile: effective_roll_profile(&cli, fixtures),
            film_stock: effective_pipeline_film_stock(&cli, fixtures),
            base_color: cli.base_color.clone(),
            base_color_source: None,
            base_color_confidence: None,
            base_color_reason: None,
            color_mode: cli.color_mode,
            render_input: cli.render_input,
            input_mode,
            render_intent: cli.render_intent,
            quality_mode: cli.quality_mode,
            white_balance: cli.white_balance,
            geometry,
            grain,
            write_master: cli.write_master,
            review_sidecar: cli.review_sidecar.clone(),
            write_review_sidecar: cli.write_review_sidecar.clone(),
            debug: cli.debug,
            force_stitch,
            force_no_stitch,
            transform: cli.transform.clone(),
            ica_max_iter: cli.ica_max_iter,
            ica_tol: cli.ica_tol,
            bit_depth,
            use_opencv: cli.use_opencv,
            require_reviewable: false,
        };
        let report = scanstitch::pipeline::run(&pipeline_cli)?;
        let report_path = pipeline_cli.output_dir.join("report.json");
        report.save(&report_path)?;
        current_report_path = Some(report_path);
        report
    };

    let mut summary =
        summarize_report_with_source(&cli.fixture, &report, current_report_path.as_deref());
    if let Some(compare_report_path) = &cli.compare_report {
        let contents = std::fs::read_to_string(compare_report_path)?;
        let compare_report = serde_json::from_str::<PipelineReport>(&contents)?;
        let compare_summary = summarize_report_with_source(
            format!("{}-baseline", cli.fixture),
            &compare_report,
            Some(compare_report_path),
        );
        summary.comparison = Some(compare_render_summaries(
            compare_report_path.to_string_lossy().to_string(),
            &compare_summary.render,
            current_report_path
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            &summary.render,
        ));
    }
    let compare_summary_path = cli
        .compare_summary
        .clone()
        .or_else(|| fixture_summary_baseline(&cli, fixtures));
    if let Some(compare_summary_path) = &compare_summary_path {
        let contents = std::fs::read_to_string(compare_summary_path)?;
        let baseline = serde_json::from_str::<TrackedValidationBaseline>(&contents)?;
        let mut comparison = compare_summary_baseline(
            compare_summary_path.to_string_lossy().to_string(),
            &baseline,
            &summary,
        );
        comparison.issues.extend(
            summary_baseline_contract_issues(&baseline)
                .into_iter()
                .map(|field| format!("summary_baseline_incomplete:{field}")),
        );
        if !comparison.issues.is_empty() {
            comparison.status = "review_required".to_string();
        }
        summary.summary_baseline_comparison = Some(comparison);
    }
    let json_path = cli
        .summary_json
        .clone()
        .unwrap_or_else(|| effective_output_dir(&cli, fixtures).join("summary.json"));
    let md_path = cli
        .summary_md
        .clone()
        .unwrap_or_else(|| effective_output_dir(&cli, fixtures).join("summary.md"));

    let json_summary = serde_json::to_string_pretty(&summary).expect("summary should serialize");
    write_text(&json_path, &json_summary)?;
    write_text(&md_path, &summary_to_markdown(&summary))?;
    if let Some(baseline_path) = &cli.write_summary_baseline {
        let baseline = tracked_baseline_from_summary(&summary);
        let baseline_json =
            serde_json::to_string_pretty(&baseline).expect("summary baseline should serialize");
        write_text(baseline_path, &baseline_json)?;
    }

    if cli.print_json_summary {
        println!("{json_summary}");
    }
    if !cli.quiet {
        println!("summary_json={}", json_path.display());
        println!("summary_md={}", md_path.display());
        if let Some(baseline_path) = &cli.write_summary_baseline {
            println!("summary_baseline={}", baseline_path.display());
        }
        print_summary_table(&summary);
    }
    if !summary.diagnostic_consistency_issues.is_empty() {
        if !cli.quiet {
            println!(
                "diagnostic_consistency_issues={}",
                summary.diagnostic_consistency_issues.join(",")
            );
        }
        let selected_failures =
            selected_failures(&summary.diagnostic_consistency_issues, &cli.fail_on);
        if cli.strict {
            return Err(format!(
                "strict report consistency validation failed with issue(s): {}",
                summary.diagnostic_consistency_issues.join(", ")
            )
            .into());
        }
        if !selected_failures.is_empty() {
            return Err(format!(
                "--fail-on matched diagnostic consistency issue(s): {}",
                selected_failures.join(", ")
            )
            .into());
        }
    }
    if cli.require_reviewable && !final_delivery_evidence_is_reviewable(&summary.render) {
        let artifact_issues = delivery_artifact_integrity_issues(&summary.render);
        return Err(format!(
            "--require-reviewable rejected validation output: render_review_status={} render_reviewable={} delivery_artifact_issues={}; report and validation summary artifacts were retained",
            summary
                .render
                .render_review_status
                .as_deref()
                .unwrap_or("missing"),
            summary
                .render
                .render_reviewable
                .map(|value| value.to_string())
                .unwrap_or_else(|| "missing".to_string()),
            if artifact_issues.is_empty() {
                "none".to_string()
            } else {
                artifact_issues.join(",")
            }
        )
        .into());
    }
    if let Some(comparison) = &summary.comparison {
        if !cli.quiet {
            println!(
                "comparison_status={} issues={}",
                comparison.status,
                if comparison.issues.is_empty() {
                    "none".to_string()
                } else {
                    comparison.issues.join(",")
                }
            );
        }
        let selected_failures = selected_failures(&comparison.issues, &cli.fail_on);
        if cli.strict && !comparison.issues.is_empty() {
            return Err(format!(
                "strict comparison failed with issue(s): {}",
                comparison.issues.join(", ")
            )
            .into());
        }
        if !selected_failures.is_empty() {
            return Err(format!(
                "--fail-on matched comparison issue(s): {}",
                selected_failures.join(", ")
            )
            .into());
        }
    }
    if let Some(comparison) = &summary.summary_baseline_comparison {
        if !cli.quiet {
            println!(
                "summary_baseline_status={} issues={}",
                comparison.status,
                if comparison.issues.is_empty() {
                    "none".to_string()
                } else {
                    comparison.issues.join(",")
                }
            );
        }
        let selected_failures = selected_failures(&comparison.issues, &cli.fail_on);
        if cli.strict && !comparison.issues.is_empty() {
            return Err(format!(
                "strict summary baseline comparison failed with issue(s): {}",
                comparison.issues.join(", ")
            )
            .into());
        }
        if !selected_failures.is_empty() {
            return Err(format!(
                "--fail-on matched summary baseline issue(s): {}",
                selected_failures.join(", ")
            )
            .into());
        }
    }

    Ok(())
}

fn load_fixture_registry(
    path: Option<&PathBuf>,
) -> Result<LoadedFixtureRegistry, Box<dyn std::error::Error>> {
    let mut fixtures = BTreeMap::new();
    fixtures.insert(
        "logan".to_string(),
        FixtureEntry {
            component1: PathBuf::from("LOGAN043.tif"),
            component2: Some(PathBuf::from("LOGAN044.tif")),
            component1_sha256: Some(
                "62af658c26d41ab69ebc91d09a2a37d30f806123c1b4e507b464a334d836b329".to_string(),
            ),
            component2_sha256: Some(
                "f220edf8704d4d81374ad2befe22c5fdcaf44faafa39228d9f2bf8409bd2a1eb".to_string(),
            ),
            additional_components: Vec::new(),
            output_dir: Some(PathBuf::from("output/validation/logan")),
            input_mode: None,
            bit_depth: None,
            grain_reduction: None,
            grain_strength: None,
            grain_scale: None,
            deskew: None,
            orientation_correction: None,
            deskew_angle_degrees: None,
            force_stitch: false,
            force_no_stitch: false,
            calibration_profile: None,
            calibration_profile_sha256: None,
            calibration_library: None,
            calibration_library_sha256: None,
            scanner_profile: None,
            roll_profile: None,
            film_stock: Some("Kodak Gold 200".to_string()),
            scene_tags: vec!["outdoor".to_string(), "logan-real-scan".to_string()],
            exposure_tags: vec!["normal-exposure".to_string()],
            reference_evidence: Vec::new(),
            render_review: None,
            render_review_sha256: None,
            calibration_case: Some("uncalibrated-image-derived".to_string()),
            expectations: FixtureExpectations {
                stitch_decision: Some("accepted".to_string()),
                base_estimate_source: Some("component_consensus".to_string()),
                output_color_space: Some("linear_prophoto_rgb_d50".to_string()),
                render_input_source: Some("fastica_separated_transmittance".to_string()),
                mapping_strategy: Some("gamut_trusted_image_matrix_blend".to_string()),
                selected_candidate: Some("gamut_trusted_image_matrix_blend".to_string()),
                selected_candidate_rank: Some(1),
                calibration_acceptance_status: Some("not_applicable".to_string()),
                calibration_color_mapping_applied: Some(false),
                candidate_risk: Some("review_neutral_support".to_string()),
                tone_color_trust_state: Some("review_required".to_string()),
                render_reviewable: Some(false),
                ..FixtureExpectations::default()
            },
            summary_baseline: Some(PathBuf::from(
                "tests/fixtures/baselines/logan_summary_baseline.json",
            )),
            summary_baseline_sha256: Some(
                "0f9f8a68159bfb8e1308f8692d18401d338b09c4a0970ef9194d16ef404de717".to_string(),
            ),
            description: Some("Local LOGAN split-frame pair".to_string()),
        },
    );

    let mut coverage_requirements = FixtureCoverageRequirements::default();
    if let Some(path) = path {
        let contents = std::fs::read_to_string(path)?;
        let registry = serde_json::from_str::<FixtureRegistry>(&contents)?;
        let registry_issues = validate_fixture_registry(&registry);
        if !registry_issues.is_empty() {
            return Err(format!(
                "fixture registry {} invalid: {}",
                path.display(),
                registry_issues.join(", ")
            )
            .into());
        }
        coverage_requirements = registry.coverage_requirements;
        fixtures.extend(registry.fixtures);
    }

    Ok(LoadedFixtureRegistry {
        fixtures,
        coverage_requirements,
    })
}

fn validate_fixture_hash_registry_writer(
    cli: &ValidationCli,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(snapshot_path) = &cli.write_fixture_hash_registry else {
        return Ok(());
    };
    let Some(registry_path) = &cli.fixture_registry else {
        return Ok(());
    };

    let registry_path = std::fs::canonicalize(registry_path)?;
    if std::fs::canonicalize(snapshot_path)
        .is_ok_and(|snapshot_path| snapshot_path == registry_path)
    {
        return Err("--write-fixture-hash-registry must not overwrite --fixture-registry; choose a separate snapshot path".into());
    }

    Ok(())
}

fn fixture_hash_registry_snapshot(
    fixtures: &BTreeMap<String, FixtureEntry>,
    requirements: &FixtureCoverageRequirements,
    summary: &FixtureCoverageSummary,
) -> FixtureRegistry {
    let mut snapshot = FixtureRegistry {
        fixtures: fixtures.clone(),
        coverage_requirements: requirements.clone(),
    };
    let coverage_entries = summary
        .fixtures
        .iter()
        .map(|entry| (entry.name.as_str(), entry))
        .collect::<BTreeMap<_, _>>();

    for (name, fixture) in &mut snapshot.fixtures {
        let Some(entry) = coverage_entries.get(name.as_str()) else {
            continue;
        };
        fill_hash_from_probe(
            &mut fixture.component1_sha256,
            entry.component1_sha256.as_ref(),
        );
        fill_hash_from_probe(
            &mut fixture.component2_sha256,
            entry.component2_sha256.as_ref(),
        );
        for (component, probe) in fixture
            .additional_components
            .iter_mut()
            .zip(&entry.additional_components)
        {
            fill_hash_from_probe(&mut component.sha256, probe.sha256.as_ref());
        }
        fill_hash_from_probe(
            &mut fixture.summary_baseline_sha256,
            entry.summary_baseline_sha256.as_ref(),
        );
        fill_hash_from_probe(
            &mut fixture.render_review_sha256,
            entry.render_review_sha256.as_ref(),
        );
        fill_hash_from_probe(
            &mut fixture.calibration_profile_sha256,
            entry.calibration_profile_sha256.as_ref(),
        );
        fill_hash_from_probe(
            &mut fixture.calibration_library_sha256,
            entry.calibration_library_sha256.as_ref(),
        );
    }

    snapshot
}

fn fill_hash_from_probe(target: &mut Option<String>, probe: Option<&FixtureSha256Probe>) {
    if let Some(actual_sha256) = probe.and_then(|probe| probe.actual_sha256.as_ref()) {
        *target = Some(actual_sha256.clone());
    }
}

fn strip_empty_registry_snapshot_values(value: &mut serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => {
            let keys = map.keys().cloned().collect::<Vec<_>>();
            for key in keys {
                let remove_default_false = (key == "seam_blend_required"
                    || key == "reference_patch_evaluation_required"
                    || key == "debug_artifacts_required")
                    && map.get(&key) == Some(&serde_json::Value::Bool(false));
                let remove_child = !remove_default_false
                    && map
                        .get_mut(&key)
                        .is_some_and(strip_empty_registry_snapshot_values);
                if remove_default_false || remove_child {
                    map.remove(&key);
                }
            }
            map.is_empty()
        }
        serde_json::Value::Array(items) => {
            for index in (0..items.len()).rev() {
                if strip_empty_registry_snapshot_values(&mut items[index]) {
                    items.remove(index);
                }
            }
            items.is_empty()
        }
        serde_json::Value::Null => true,
        _ => false,
    }
}

fn effective_output_dir(cli: &ValidationCli, fixtures: &BTreeMap<String, FixtureEntry>) -> PathBuf {
    let default_output = PathBuf::from("output/validation/logan");
    if cli.output_dir == default_output {
        if let Some(output_dir) = fixtures
            .get(&cli.fixture)
            .and_then(|fixture| fixture.output_dir.clone())
        {
            return output_dir;
        }
    }
    cli.output_dir.clone()
}

fn fixture_set_output_dir(cli: &ValidationCli) -> PathBuf {
    let default_output = PathBuf::from("output/validation/logan");
    if cli.output_dir == default_output {
        PathBuf::from("output/validation")
    } else {
        cli.output_dir.clone()
    }
}

fn fixture_summary_baseline(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Option<PathBuf> {
    fixtures
        .get(&cli.fixture)
        .and_then(|fixture| fixture.summary_baseline.clone())
}

fn effective_calibration_profile(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Option<PathBuf> {
    cli.calibration_profile.clone().or_else(|| {
        fixtures
            .get(&cli.fixture)
            .and_then(|fixture| fixture.calibration_profile.clone())
    })
}

fn effective_calibration_library(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Option<PathBuf> {
    cli.calibration_library.clone().or_else(|| {
        fixtures
            .get(&cli.fixture)
            .and_then(|fixture| fixture.calibration_library.clone())
    })
}

fn effective_scanner_profile(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Option<String> {
    cli.scanner_profile.clone().or_else(|| {
        fixtures
            .get(&cli.fixture)
            .and_then(|fixture| fixture.scanner_profile.clone())
    })
}

fn effective_roll_profile(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Option<String> {
    cli.roll_profile.clone().or_else(|| {
        fixtures
            .get(&cli.fixture)
            .and_then(|fixture| fixture.roll_profile.clone())
    })
}

fn effective_pipeline_film_stock(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Option<String> {
    let fixture = fixtures.get(&cli.fixture);
    let calibration_library = effective_calibration_library(cli, fixtures);
    pipeline_film_stock(
        cli.film_stock.as_ref(),
        fixture,
        calibration_library.as_ref(),
    )
}

fn geometry_for_fixture(
    default: GeometryArgs,
    fixture: Option<&FixtureEntry>,
) -> Result<GeometryArgs, Box<dyn std::error::Error>> {
    let Some(fixture) = fixture else {
        return Ok(default);
    };
    let deskew = match fixture.deskew.as_deref() {
        Some(label) => parse_deskew_mode_label(label)
            .ok_or_else(|| format!("invalid fixture deskew mode `{label}`"))?,
        None => default.deskew,
    };
    let orientation_correction = match fixture.orientation_correction.as_deref() {
        Some(label) => parse_orientation_correction_label(label)
            .ok_or_else(|| format!("invalid fixture orientation_correction `{label}`"))?,
        None => default.orientation_correction,
    };
    let geometry = GeometryArgs {
        orientation_correction,
        deskew,
        deskew_angle_degrees: fixture
            .deskew_angle_degrees
            .unwrap_or(default.deskew_angle_degrees),
    };
    geometry.validate()?;
    Ok(geometry)
}

fn effective_geometry(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Result<GeometryArgs, Box<dyn std::error::Error>> {
    geometry_for_fixture(cli.geometry, fixtures.get(&cli.fixture))
}

fn effective_input_mode(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Result<InputMode, Box<dyn std::error::Error>> {
    match fixtures
        .get(&cli.fixture)
        .and_then(|fixture| fixture.input_mode.as_deref())
    {
        Some(label) => parse_input_mode_label(label)
            .ok_or_else(|| format!("invalid fixture input_mode `{label}`").into()),
        None => Ok(cli.input_mode),
    }
}

fn effective_bit_depth(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Result<u8, Box<dyn std::error::Error>> {
    let bit_depth = fixtures
        .get(&cli.fixture)
        .and_then(|fixture| fixture.bit_depth)
        .unwrap_or(cli.bit_depth);
    if matches!(bit_depth, 14 | 16) {
        Ok(bit_depth)
    } else {
        Err(format!("invalid fixture bit_depth `{bit_depth}`").into())
    }
}

fn grain_for_fixture(
    default: GrainReductionArgs,
    fixture: Option<&FixtureEntry>,
) -> Result<GrainReductionArgs, Box<dyn std::error::Error>> {
    let grain_reduction = match fixture.and_then(|fixture| fixture.grain_reduction.as_deref()) {
        Some(label) => parse_grain_reduction_mode_label(label)
            .ok_or_else(|| format!("invalid fixture grain_reduction `{label}`"))?,
        None => default.grain_reduction,
    };
    let grain = GrainReductionArgs {
        grain_reduction,
        grain_strength: fixture
            .and_then(|fixture| fixture.grain_strength)
            .unwrap_or(default.grain_strength),
        grain_scale: fixture
            .and_then(|fixture| fixture.grain_scale)
            .unwrap_or(default.grain_scale),
    };
    if let Err(error) = validate_grain_settings(grain) {
        return Err(error.into());
    }
    Ok(grain)
}

fn effective_grain(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Result<GrainReductionArgs, Box<dyn std::error::Error>> {
    grain_for_fixture(cli.grain, fixtures.get(&cli.fixture))
}

fn validate_grain_settings(grain: GrainReductionArgs) -> Result<(), String> {
    if !grain.grain_strength.is_finite() || !(0.0..=1.0).contains(&grain.grain_strength) {
        return Err("fixture grain_strength must be finite and between 0 and 1".to_string());
    }
    if !grain.grain_scale.is_finite() || !(0.5..=4.0).contains(&grain.grain_scale) {
        return Err("fixture grain_scale must be finite and between 0.5 and 4".to_string());
    }
    Ok(())
}

fn fixture_declares_grain_reduction_enabled(fixture: &FixtureEntry) -> bool {
    fixture
        .grain_reduction
        .as_deref()
        .and_then(parse_grain_reduction_mode_label)
        .is_some_and(GrainReductionMode::enabled)
}

fn fixture_grain_detail_contract_missing_fields(fixture: &FixtureEntry) -> Vec<String> {
    if !fixture_declares_grain_reduction_enabled(fixture) {
        return Vec::new();
    }

    let expectations = &fixture.expectations;
    let mut missing = Vec::new();
    if !fixture.grain_strength.is_some_and(|value| value > 0.0) {
        missing.push("grain_strength>0".to_string());
    }
    if fixture.grain_scale.is_none() {
        missing.push("grain_scale".to_string());
    }
    if expectations.grain_reduction_enabled != Some(true) {
        missing.push("expectations.grain_reduction_enabled=true".to_string());
    }
    if expectations.grain_detail_review_required != Some(false) {
        missing.push("expectations.grain_detail_review_required=false".to_string());
    }
    if expectations.grain_detail_decision_supported != Some(true) {
        missing.push("expectations.grain_detail_decision_supported=true".to_string());
    }
    if expectations
        .grain_detail_luminance_probe_count_min
        .is_none_or(|value| value < GRAIN_DETAIL_MIN_PROBE_COUNT)
    {
        missing.push(format!(
            "expectations.grain_detail_luminance_probe_count_min>={GRAIN_DETAIL_MIN_PROBE_COUNT}"
        ));
    }
    if expectations
        .grain_detail_chroma_probe_count_min
        .is_none_or(|value| value < GRAIN_DETAIL_MIN_PROBE_COUNT)
    {
        missing.push(format!(
            "expectations.grain_detail_chroma_probe_count_min>={GRAIN_DETAIL_MIN_PROBE_COUNT}"
        ));
    }
    if !expectations
        .grain_detail_luminance_p10_retention_min
        .is_some_and(|value| value >= GRAIN_DETAIL_P10_RETENTION_MIN)
    {
        missing.push(format!(
            "expectations.grain_detail_luminance_p10_retention_min>={GRAIN_DETAIL_P10_RETENTION_MIN:.2}"
        ));
    }
    if !expectations
        .grain_detail_chroma_p10_retention_min
        .is_some_and(|value| value >= GRAIN_DETAIL_P10_RETENTION_MIN)
    {
        missing.push(format!(
            "expectations.grain_detail_chroma_p10_retention_min>={GRAIN_DETAIL_P10_RETENTION_MIN:.2}"
        ));
    }
    missing
}

fn fixture_grain_reduction_effect_contract_missing_fields(fixture: &FixtureEntry) -> Vec<String> {
    if !fixture_declares_grain_reduction_enabled(fixture) {
        return Vec::new();
    }

    let expectations = &fixture.expectations;
    let mut missing = Vec::new();
    if !fixture_grain_detail_contract_missing_fields(fixture).is_empty() {
        missing.push("grain_detail_contract_complete".to_string());
    }
    if !expectations
        .grain_reduction_applied_ratio_min
        .is_some_and(|value| value > 0.0)
    {
        missing.push("expectations.grain_reduction_applied_ratio_min>0".to_string());
    }
    if !expectations
        .grain_reduction_structure_excluded_ratio_min
        .is_some_and(|value| value > 0.0)
    {
        missing.push("expectations.grain_reduction_structure_excluded_ratio_min>0".to_string());
    }
    if !expectations
        .grain_reduction_flat_luma_p95_reduction_ratio_min
        .is_some_and(|value| value > 0.0)
    {
        missing
            .push("expectations.grain_reduction_flat_luma_p95_reduction_ratio_min>0".to_string());
    }
    if !expectations
        .grain_reduction_flat_chroma_p95_reduction_ratio_min
        .is_some_and(|value| value > 0.0)
    {
        missing
            .push("expectations.grain_reduction_flat_chroma_p95_reduction_ratio_min>0".to_string());
    }
    missing
}

fn fixture_declares_render_dynamic_range_contract(fixture: &FixtureEntry) -> bool {
    let expectations = &fixture.expectations;
    expectations.post_scale_preserved_ratio_min.is_some()
        || expectations.render_luminance_range_p05_p95_min.is_some()
        || expectations.render_review_status.as_deref() == Some("reviewable")
        || expectations.render_reviewable == Some(true)
        || expectations.tone_output_confidence_status.is_some()
        || expectations.tone_output_review_required.is_some()
        || expectations.tone_output_evidence_confidence_min.is_some()
        || expectations
            .render_to_mapped_luminance_range_ratio_min
            .is_some()
        || expectations
            .post_chroma_compression_clipped_high_ratio_max
            .is_some()
        || expectations
            .post_chroma_compression_clipped_low_ratio_max
            .is_some()
}

fn fixture_render_dynamic_range_contract_missing_fields(fixture: &FixtureEntry) -> Vec<String> {
    let expectations = &fixture.expectations;
    let mut missing = Vec::new();
    if !expectations
        .post_scale_preserved_ratio_min
        .is_some_and(|value| value > 0.0)
    {
        missing.push("expectations.post_scale_preserved_ratio_min>0".to_string());
    }
    if !expectations
        .render_luminance_range_p05_p95_min
        .is_some_and(|value| value > 0.0)
    {
        missing.push("expectations.render_luminance_range_p05_p95_min>0".to_string());
    }
    if expectations.render_review_status.as_deref() != Some("reviewable") {
        missing.push("expectations.render_review_status=reviewable".to_string());
    }
    if expectations.render_reviewable != Some(true) {
        missing.push("expectations.render_reviewable=true".to_string());
    }
    if expectations.tone_output_confidence_status.as_deref()
        != Some("supported_render_tonal_distribution")
    {
        missing.push(
            "expectations.tone_output_confidence_status=supported_render_tonal_distribution"
                .to_string(),
        );
    }
    if expectations.tone_output_review_required != Some(false) {
        missing.push("expectations.tone_output_review_required=false".to_string());
    }
    if !expectations
        .tone_output_evidence_confidence_min
        .is_some_and(|value| value > 0.0)
    {
        missing.push("expectations.tone_output_evidence_confidence_min>0".to_string());
    }
    if !expectations
        .render_to_mapped_luminance_range_ratio_min
        .is_some_and(|value| value > 0.0)
    {
        missing.push("expectations.render_to_mapped_luminance_range_ratio_min>0".to_string());
    }
    if !expectations
        .post_chroma_compression_clipped_high_ratio_max
        .is_some_and(|value| value < 1.0)
    {
        missing.push("expectations.post_chroma_compression_clipped_high_ratio_max<1".to_string());
    }
    if !expectations
        .post_chroma_compression_clipped_low_ratio_max
        .is_some_and(|value| value < 1.0)
    {
        missing.push("expectations.post_chroma_compression_clipped_low_ratio_max<1".to_string());
    }
    missing
}

fn fixture_declares_stitch_normalization_contract(fixture: &FixtureEntry) -> bool {
    let expectations = &fixture.expectations;
    matches!(
        expectations.stitch_decision.as_deref(),
        Some("accepted" | "accepted_sequence")
    ) || expectations.seam_exposure_model.is_some()
        || expectations
            .seam_exposure_held_out_validation_passed
            .is_some()
        || expectations
            .seam_exposure_offset_normalized_abs_max
            .is_some()
        || expectations.seam_blend_required
        || expectations.seam_blend_mode.is_some()
        || expectations.seam_blend_review_required.is_some()
        || expectations.seam_detail_review_required.is_some()
        || expectations.seam_detail_supported_scale_count_min.is_some()
        || expectations
            .seam_detail_max_symmetric_energy_ratio_max
            .is_some()
        || expectations.seam_gradient_ratio_max.is_some()
        || expectations.seam_overlap_p95_abs_difference_max.is_some()
}

fn known_stitch_exposure_model(model: &str) -> bool {
    matches!(
        model,
        "sequence_mixed"
            | "identity"
            | "gain_only_scalar"
            | "gain_only_rgb"
            | "gain_offset_rgb"
            | "gain_spatial_y_rgb"
            | "gain_offset_spatial_y_rgb"
            | "gain_spatial_xy_rgb"
            | "gain_offset_spatial_xy_rgb"
            | "gain_spatial_quadratic_xy_rgb"
            | "gain_offset_spatial_quadratic_xy_rgb"
    )
}

fn fixture_stitch_normalization_contract_missing_fields(fixture: &FixtureEntry) -> Vec<String> {
    let expectations = &fixture.expectations;
    let mut missing = Vec::new();

    if fixture.force_no_stitch {
        missing.push("force_no_stitch=false".to_string());
    }
    if !matches!(
        expectations.stitch_decision.as_deref(),
        Some("accepted" | "accepted_sequence")
    ) {
        missing.push("expectations.stitch_decision=accepted|accepted_sequence".to_string());
    }
    let exposure_model = expectations.seam_exposure_model.as_deref();
    if !exposure_model.is_some_and(known_stitch_exposure_model) {
        missing.push("expectations.seam_exposure_model=known_model".to_string());
    }
    let expected_held_out_validation = exposure_model
        .filter(|model| known_stitch_exposure_model(model))
        .map(|model| model != "identity");
    if let Some(expected) = expected_held_out_validation {
        if expectations.seam_exposure_held_out_validation_passed != Some(expected) {
            missing.push(format!(
                "expectations.seam_exposure_held_out_validation_passed={expected}"
            ));
        }
    } else if expectations
        .seam_exposure_held_out_validation_passed
        .is_none()
    {
        missing
            .push("expectations.seam_exposure_held_out_validation_passed={true|false}".to_string());
    }
    if !expectations
        .seam_exposure_offset_normalized_abs_max
        .is_some_and(|value| (0.0..=STITCH_NORMALIZATION_MAX_OFFSET_RATIO).contains(&value))
    {
        missing.push(format!(
            "expectations.seam_exposure_offset_normalized_abs_max<={STITCH_NORMALIZATION_MAX_OFFSET_RATIO:.2}"
        ));
    }
    if !expectations.seam_blend_required {
        missing.push("expectations.seam_blend_required=true".to_string());
    }
    if expectations.seam_blend_mode.as_deref() != Some("seam_aware_multiband") {
        missing.push("expectations.seam_blend_mode=seam_aware_multiband".to_string());
    }
    if expectations.seam_blend_review_required != Some(false) {
        missing.push("expectations.seam_blend_review_required=false".to_string());
    }
    if expectations.seam_detail_review_required != Some(false) {
        missing.push("expectations.seam_detail_review_required=false".to_string());
    }
    if expectations
        .seam_detail_supported_scale_count_min
        .is_none_or(|value| value < STITCH_NORMALIZATION_MIN_DETAIL_SCALE_COUNT)
    {
        missing.push(format!(
            "expectations.seam_detail_supported_scale_count_min>={STITCH_NORMALIZATION_MIN_DETAIL_SCALE_COUNT}"
        ));
    }
    if !expectations
        .seam_detail_max_symmetric_energy_ratio_max
        .is_some_and(|value| (1.0..=STITCH_NORMALIZATION_MAX_DETAIL_ENERGY_RATIO).contains(&value))
    {
        missing.push(format!(
            "expectations.seam_detail_max_symmetric_energy_ratio_max<={STITCH_NORMALIZATION_MAX_DETAIL_ENERGY_RATIO:.2}"
        ));
    }
    if !expectations
        .seam_gradient_ratio_max
        .is_some_and(|value| (0.0..=STITCH_NORMALIZATION_MAX_GRADIENT_RATIO).contains(&value))
    {
        missing.push(format!(
            "expectations.seam_gradient_ratio_max<={STITCH_NORMALIZATION_MAX_GRADIENT_RATIO:.2}"
        ));
    }
    if !expectations
        .seam_overlap_p95_abs_difference_max
        .is_some_and(|value| (0.0..1.0).contains(&value))
    {
        missing.push("expectations.seam_overlap_p95_abs_difference_max<1".to_string());
    }
    missing
}

fn fixture_declares_geometry_preparation_contract(fixture: &FixtureEntry) -> bool {
    let expectations = &fixture.expectations;
    expectations.deskew_all_components_applied.is_some()
        || expectations
            .deskew_minimum_component_retained_area_ratio_min
            .is_some()
        || expectations.border_crop_all_components_cropped.is_some()
        || expectations
            .border_crop_minimum_removed_edge_count_per_component_min
            .is_some()
        || expectations.border_crop_retained_area_ratio_min.is_some()
        || expectations.border_crop_retained_area_ratio_max.is_some()
        || expectations.border_crop_rejected.is_some()
}

fn fixture_geometry_preparation_contract_missing_fields(fixture: &FixtureEntry) -> Vec<String> {
    let expectations = &fixture.expectations;
    let mut missing = Vec::new();
    if expectations.deskew_status.as_deref() != Some("applied") {
        missing.push("expectations.deskew_status=applied".to_string());
    }
    if expectations.deskew_applied != Some(true) {
        missing.push("expectations.deskew_applied=true".to_string());
    }
    if expectations.deskew_all_components_applied != Some(true) {
        missing.push("expectations.deskew_all_components_applied=true".to_string());
    }
    if expectations.deskew_review_required != Some(false) {
        missing.push("expectations.deskew_review_required=false".to_string());
    }
    if expectations
        .deskew_minimum_component_retained_area_ratio_min
        .is_none_or(|value| value < GEOMETRY_PREPARATION_MIN_DESKEW_RETAINED_AREA_RATIO)
    {
        missing.push(format!(
            "expectations.deskew_minimum_component_retained_area_ratio_min>={GEOMETRY_PREPARATION_MIN_DESKEW_RETAINED_AREA_RATIO:.2}"
        ));
    }
    if expectations.border_crop_all_components_cropped != Some(true) {
        missing.push("expectations.border_crop_all_components_cropped=true".to_string());
    }
    if expectations
        .border_crop_minimum_removed_edge_count_per_component_min
        .is_none_or(|value| value < GEOMETRY_PREPARATION_MIN_REMOVED_EDGES_PER_COMPONENT)
    {
        missing.push(format!(
            "expectations.border_crop_minimum_removed_edge_count_per_component_min>={GEOMETRY_PREPARATION_MIN_REMOVED_EDGES_PER_COMPONENT}"
        ));
    }
    if expectations
        .border_crop_retained_area_ratio_min
        .is_none_or(|value| value < GEOMETRY_PREPARATION_MIN_BORDER_RETAINED_AREA_RATIO)
    {
        missing.push(format!(
            "expectations.border_crop_retained_area_ratio_min>={GEOMETRY_PREPARATION_MIN_BORDER_RETAINED_AREA_RATIO:.2}"
        ));
    }
    if !expectations
        .border_crop_retained_area_ratio_max
        .is_some_and(|value| (0.0..1.0).contains(&value))
    {
        missing.push("expectations.border_crop_retained_area_ratio_max<1".to_string());
    }
    if expectations.border_crop_rejected != Some(false) {
        missing.push("expectations.border_crop_rejected=false".to_string());
    }
    missing
}

fn fixture_declares_geometry_accuracy_contract(fixture: &FixtureEntry) -> bool {
    let expectations = &fixture.expectations;
    expectations.deskew_correction_degrees_expected.is_some()
        || expectations.deskew_correction_tolerance_degrees.is_some()
        || !expectations.border_crop_components_expected.is_empty()
}

fn fixture_geometry_accuracy_contract_missing_fields(fixture: &FixtureEntry) -> Vec<String> {
    let expectations = &fixture.expectations;
    let mut missing = fixture_geometry_preparation_contract_missing_fields(fixture);
    if !expectations
        .deskew_correction_degrees_expected
        .is_some_and(|value| (-3.0..=3.0).contains(&value))
    {
        missing.push(
            "expectations.deskew_correction_degrees_expected=-3..3_factual_target".to_string(),
        );
    }
    if !expectations
        .deskew_correction_tolerance_degrees
        .is_some_and(|value| {
            (0.0..=GEOMETRY_ACCURACY_MAX_DESKEW_TOLERANCE_DEGREES).contains(&value)
        })
    {
        missing.push(format!(
            "expectations.deskew_correction_tolerance_degrees<={GEOMETRY_ACCURACY_MAX_DESKEW_TOLERANCE_DEGREES:.2}"
        ));
    }
    if expectations.border_crop_components_expected.len() != fixture.component_count() {
        missing.push(format!(
            "expectations.border_crop_components_expected=all_{}_components",
            fixture.component_count()
        ));
    }
    let mut component_indices = BTreeSet::new();
    for component in &expectations.border_crop_components_expected {
        if !(1..=fixture.component_count()).contains(&component.component_index) {
            missing.push(format!(
                "expectations.border_crop_components_expected.component_index={}_within_1..{}",
                component.component_index,
                fixture.component_count()
            ));
        } else if !component_indices.insert(component.component_index) {
            missing.push(format!(
                "expectations.border_crop_components_expected.component_index={}unique",
                component.component_index
            ));
        }
        if component.tolerance_px > GEOMETRY_ACCURACY_MAX_CROP_TOLERANCE_PX {
            missing.push(format!(
                "expectations.border_crop_components_expected.component_{}.tolerance_px<={GEOMETRY_ACCURACY_MAX_CROP_TOLERANCE_PX}",
                component.component_index
            ));
        }
    }
    missing.sort();
    missing.dedup();
    missing
}

fn expected_orientation_transform(
    tag_present: bool,
    tag_value: Option<u16>,
) -> Option<&'static str> {
    match (tag_present, tag_value) {
        (false, None) | (true, Some(1)) => Some("identity"),
        (true, Some(2)) => Some("flip_horizontal"),
        (true, Some(3)) => Some("rotate_180"),
        (true, Some(4)) => Some("flip_vertical"),
        (true, Some(5)) => Some("rotate_90_clockwise_then_flip_horizontal"),
        (true, Some(6)) => Some("rotate_90_clockwise"),
        (true, Some(7)) => Some("rotate_270_clockwise_then_flip_horizontal"),
        (true, Some(8)) => Some("rotate_270_clockwise"),
        _ => None,
    }
}

fn fixture_orientation_correction(fixture: &FixtureEntry) -> OrientationCorrection {
    fixture
        .orientation_correction
        .as_deref()
        .and_then(parse_orientation_correction_label)
        .unwrap_or(OrientationCorrection::None)
}

fn orientation_swaps_dimensions(tag_value: Option<u16>) -> bool {
    matches!(tag_value, Some(5..=8))
}

fn fixture_declares_orientation_accuracy_contract(fixture: &FixtureEntry) -> bool {
    !fixture
        .expectations
        .orientation_components_expected
        .is_empty()
}

fn fixture_orientation_accuracy_contract_missing_fields(fixture: &FixtureEntry) -> Vec<String> {
    let expectations = &fixture.expectations.orientation_components_expected;
    let mut missing = Vec::new();
    if expectations.len() != fixture.component_count() {
        missing.push(format!(
            "expectations.orientation_components_expected=all_{}_components",
            fixture.component_count()
        ));
    }
    let mut component_indices = BTreeSet::new();
    for component in expectations {
        let prefix = format!(
            "expectations.orientation_components_expected.component_{}",
            component.component_index
        );
        if !(1..=fixture.component_count()).contains(&component.component_index) {
            missing.push(format!(
                "expectations.orientation_components_expected.component_index={}_within_1..{}",
                component.component_index,
                fixture.component_count()
            ));
        } else if !component_indices.insert(component.component_index) {
            missing.push(format!(
                "expectations.orientation_components_expected.component_index={}_unique",
                component.component_index
            ));
        }
        if !component.upright_approved {
            missing.push(format!("{prefix}.upright_approved=true"));
        }
        if !is_valid_sha256_hex(&component.decoded_pixel_sha256) {
            missing.push(format!("{prefix}.decoded_pixel_sha256=64_hex"));
        }
        let metadata_transform =
            expected_orientation_transform(component.tag_present, component.tag_value);
        if metadata_transform.is_none() {
            missing.push(if component.tag_present {
                format!("{prefix}.tag_value=1..8_when_tag_present")
            } else {
                format!("{prefix}.tag_value=null_when_tag_absent")
            });
        }
        let correction = fixture_orientation_correction(fixture);
        let effective_tag = metadata_transform
            .is_some()
            .then(|| tiff_io::compose_orientation_tags(component.tag_value, correction.exif_tag()))
            .flatten();
        let expected_transform = effective_tag
            .map(|tag| tiff_io::orientation_transform_name(Some(tag)))
            .or_else(|| {
                (!component.tag_present && correction == OrientationCorrection::None)
                    .then_some("identity")
            });
        if expected_transform.is_some_and(|expected| component.transform != expected) {
            missing.push(format!(
                "{prefix}.transform={}",
                expected_transform.unwrap_or("known_transform")
            ));
        }
        let expected_applied = (component.tag_present && component.tag_value != Some(1))
            || correction != OrientationCorrection::None;
        if expected_transform.is_some() && component.applied != expected_applied {
            missing.push(format!("{prefix}.applied={expected_applied}"));
        }
        if component.source_width == 0 || component.source_height == 0 {
            missing.push(format!("{prefix}.source_dimensions>0"));
        } else {
            let (expected_width, expected_height) = if orientation_swaps_dimensions(effective_tag) {
                (component.source_height, component.source_width)
            } else {
                (component.source_width, component.source_height)
            };
            if component.output_width != expected_width
                || component.output_height != expected_height
            {
                missing.push(format!(
                    "{prefix}.output_dimensions={expected_width}x{expected_height}"
                ));
            }
        }
    }
    missing.sort();
    missing.dedup();
    missing
}

fn fixture_declares_negative_reconstruction_contract(fixture: &FixtureEntry) -> bool {
    let expectations = &fixture.expectations;
    expectations.density_inversion_skipped.is_some()
        || expectations.negative_response_model.is_some()
        || expectations.negative_response_source.is_some()
        || expectations.negative_response_accepted.is_some()
        || expectations.negative_response_review_required.is_some()
        || expectations.negative_response_measured_model_id.is_some()
        || expectations
            .negative_response_curve_extrapolated_ratio_max
            .is_some()
}

fn trusted_negative_base_source(source: &str) -> bool {
    matches!(
        source,
        "working_edges"
            | "horizontal_base_region"
            | "vertical_base_region"
            | "pre_crop_rebate_measurement"
            | "component_consensus"
            | "roll_consensus_base"
    )
}

fn fixture_negative_reconstruction_contract_missing_fields(fixture: &FixtureEntry) -> Vec<String> {
    let expectations = &fixture.expectations;
    let mut missing = Vec::new();
    if fixture.input_mode.as_deref() != Some("negative") {
        missing.push("input_mode=negative".to_string());
    }
    if !expectations
        .base_estimate_source
        .as_deref()
        .is_some_and(trusted_negative_base_source)
    {
        missing.push("expectations.base_estimate_source=measured_base_source".to_string());
    }
    if expectations
        .base_confidence_min
        .is_none_or(|value| value < NEGATIVE_RECONSTRUCTION_MIN_BASE_CONFIDENCE)
    {
        missing.push(format!(
            "expectations.base_confidence_min>={NEGATIVE_RECONSTRUCTION_MIN_BASE_CONFIDENCE:.2}"
        ));
    }
    if expectations.density_inversion_skipped != Some(false) {
        missing.push("expectations.density_inversion_skipped=false".to_string());
    }
    if expectations.negative_response_model.as_deref() != Some("measured_nonlinear_dye_separation")
    {
        missing.push(
            "expectations.negative_response_model=measured_nonlinear_dye_separation".to_string(),
        );
    }
    if expectations.negative_response_source.as_deref()
        != Some("measured_roll_target_with_held_out_validation")
    {
        missing.push(
            "expectations.negative_response_source=measured_roll_target_with_held_out_validation"
                .to_string(),
        );
    }
    if expectations.negative_response_accepted != Some(true) {
        missing.push("expectations.negative_response_accepted=true".to_string());
    }
    if expectations.negative_response_review_required != Some(false) {
        missing.push("expectations.negative_response_review_required=false".to_string());
    }
    if expectations.negative_response_crosstalk_model.as_deref()
        != Some("measured_3x3_scanner_density_to_film_layers")
    {
        missing.push(
            "expectations.negative_response_crosstalk_model=measured_3x3_scanner_density_to_film_layers"
                .to_string(),
        );
    }
    if expectations
        .negative_response_characteristic_curve_model
        .as_deref()
        != Some("measured_monotone_pchip_density_to_scene_log_exposure")
    {
        missing.push(
            "expectations.negative_response_characteristic_curve_model=measured_monotone_pchip_density_to_scene_log_exposure"
                .to_string(),
        );
    }
    if expectations
        .negative_response_measured_model_id
        .as_deref()
        .is_none_or(|value| value.trim().is_empty())
    {
        missing.push("expectations.negative_response_measured_model_id=nonempty".to_string());
    }
    if expectations
        .negative_response_measured_confidence_min
        .is_none_or(|value| value < NEGATIVE_RECONSTRUCTION_MIN_MEASURED_CONFIDENCE)
    {
        missing.push(format!(
            "expectations.negative_response_measured_confidence_min>={NEGATIVE_RECONSTRUCTION_MIN_MEASURED_CONFIDENCE:.2}"
        ));
    }
    if !expectations
        .negative_response_held_out_delta_e00_rms_max
        .is_some_and(|value| {
            (0.0..=NEGATIVE_RECONSTRUCTION_MAX_HELD_OUT_DELTA_E00_RMS).contains(&value)
        })
    {
        missing.push(format!(
            "expectations.negative_response_held_out_delta_e00_rms_max<={NEGATIVE_RECONSTRUCTION_MAX_HELD_OUT_DELTA_E00_RMS:.1}"
        ));
    }
    if !expectations
        .negative_response_held_out_max_delta_e00_max
        .is_some_and(|value| {
            (0.0..=NEGATIVE_RECONSTRUCTION_MAX_HELD_OUT_DELTA_E00_MAX).contains(&value)
        })
    {
        missing.push(format!(
            "expectations.negative_response_held_out_max_delta_e00_max<={NEGATIVE_RECONSTRUCTION_MAX_HELD_OUT_DELTA_E00_MAX:.1}"
        ));
    }
    if expectations
        .negative_response_held_out_improvement_over_unit_slope_min
        .is_none_or(|value| value < NEGATIVE_RECONSTRUCTION_MIN_BASELINE_IMPROVEMENT)
    {
        missing.push(format!(
            "expectations.negative_response_held_out_improvement_over_unit_slope_min>={NEGATIVE_RECONSTRUCTION_MIN_BASELINE_IMPROVEMENT:.2}"
        ));
    }
    if !expectations
        .negative_response_density_noise_gain_max
        .is_some_and(|value| {
            (0.0..=NEGATIVE_RECONSTRUCTION_MAX_DENSITY_NOISE_GAIN).contains(&value)
        })
    {
        missing.push(format!(
            "expectations.negative_response_density_noise_gain_max<={NEGATIVE_RECONSTRUCTION_MAX_DENSITY_NOISE_GAIN:.1}"
        ));
    }
    if !expectations
        .negative_response_curve_extrapolated_ratio_max
        .is_some_and(|value| {
            (0.0..=NEGATIVE_RECONSTRUCTION_MAX_EXTRAPOLATION_RATIO).contains(&value)
        })
    {
        missing.push(format!(
            "expectations.negative_response_curve_extrapolated_ratio_max<={NEGATIVE_RECONSTRUCTION_MAX_EXTRAPOLATION_RATIO:.2}"
        ));
    }
    if expectations.negative_response_signed_headroom_preserved != Some(true) {
        missing.push("expectations.negative_response_signed_headroom_preserved=true".to_string());
    }
    if expectations
        .negative_response_curve_interpolation
        .as_deref()
        != Some("monotone_piecewise_cubic_hermite_with_endpoint_tangent_extrapolation")
    {
        missing.push(
            "expectations.negative_response_curve_interpolation=monotone_piecewise_cubic_hermite_with_endpoint_tangent_extrapolation"
                .to_string(),
        );
    }
    if expectations.render_input_source.as_deref() != Some("direct_density_transmittance") {
        missing.push("expectations.render_input_source=direct_density_transmittance".to_string());
    }
    if expectations.calibration_acceptance_status.as_deref() != Some("accepted") {
        missing.push("expectations.calibration_acceptance_status=accepted".to_string());
    }
    if expectations.calibration_color_mapping_applied != Some(true) {
        missing.push("expectations.calibration_color_mapping_applied=true".to_string());
    }
    if expectations.candidate_risk.as_deref() != Some("safe") {
        missing.push("expectations.candidate_risk=safe".to_string());
    }
    if expectations.tone_color_trust_state.as_deref() != Some("trusted") {
        missing.push("expectations.tone_color_trust_state=trusted".to_string());
    }
    if expectations.output_color_space.as_deref() != Some("linear_prophoto_rgb_d50") {
        missing.push("expectations.output_color_space=linear_prophoto_rgb_d50".to_string());
    }
    missing
}

fn effective_force_settings(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Result<(bool, bool), Box<dyn std::error::Error>> {
    let fixture = fixtures.get(&cli.fixture);
    let force_stitch = cli.force_stitch || fixture.is_some_and(|fixture| fixture.force_stitch);
    let force_no_stitch =
        cli.force_no_stitch || fixture.is_some_and(|fixture| fixture.force_no_stitch);
    if force_stitch && force_no_stitch {
        Err("effective fixture settings request both force-stitch and force-no-stitch".into())
    } else {
        Ok((force_stitch, force_no_stitch))
    }
}

fn pipeline_film_stock(
    cli_film_stock: Option<&String>,
    fixture: Option<&FixtureEntry>,
    calibration_library: Option<&PathBuf>,
) -> Option<String> {
    cli_film_stock.cloned().or_else(|| {
        calibration_library
            .is_some()
            .then(|| fixture.and_then(|fixture| fixture.film_stock.clone()))
            .flatten()
    })
}

fn validate_fixture_registry(registry: &FixtureRegistry) -> Vec<String> {
    let mut issues = Vec::new();
    if registry.fixtures.is_empty() {
        issues.push("fixtures_empty".to_string());
    }

    validate_unique_non_empty_values(
        "coverage_requirements:required_film_stock",
        &registry.coverage_requirements.required_film_stocks,
        &mut issues,
    );
    validate_unique_non_empty_values(
        "coverage_requirements:required_scene_tag",
        &registry.coverage_requirements.required_scene_tags,
        &mut issues,
    );
    validate_unique_non_empty_values(
        "coverage_requirements:required_exposure_tag",
        &registry.coverage_requirements.required_exposure_tags,
        &mut issues,
    );
    validate_unique_non_empty_values(
        "coverage_requirements:required_calibration_case",
        &registry.coverage_requirements.required_calibration_cases,
        &mut issues,
    );
    validate_unique_non_empty_values(
        "coverage_requirements:required_scanner_profile",
        &registry.coverage_requirements.required_scanner_profiles,
        &mut issues,
    );
    validate_unique_non_empty_values(
        "coverage_requirements:required_roll_profile",
        &registry.coverage_requirements.required_roll_profiles,
        &mut issues,
    );
    validate_unique_non_empty_values(
        "coverage_requirements:required_reference_evidence",
        &registry.coverage_requirements.required_reference_evidence,
        &mut issues,
    );
    validate_unique_non_empty_values(
        "coverage_requirements:required_debug_artifact_kind",
        &registry.coverage_requirements.required_debug_artifact_kinds,
        &mut issues,
    );
    for kind in &registry.coverage_requirements.required_debug_artifact_kinds {
        if !kind.is_empty() && !is_known_debug_artifact_kind(kind) {
            issues.push(format!(
                "coverage_requirements:required_debug_artifact_kind_unknown:{kind}"
            ));
        }
    }
    if registry
        .coverage_requirements
        .min_tiff_bits_per_sample
        .is_some_and(|bits| bits > 16)
    {
        issues.push("coverage_requirements:min_tiff_bits_per_sample_out_of_range".to_string());
    }
    validate_required_pair_labels(
        "coverage_requirements:required_film_stock_calibration_pair",
        &registry
            .coverage_requirements
            .required_film_stock_calibration_pairs,
        &mut issues,
    );
    validate_required_pair_labels(
        "coverage_requirements:required_scene_exposure_pair",
        &registry.coverage_requirements.required_scene_exposure_pairs,
        &mut issues,
    );

    for (name, fixture) in &registry.fixtures {
        validate_fixture_entry(name, fixture, &mut issues);
    }

    issues
}

fn validate_fixture_entry(name: &str, fixture: &FixtureEntry, issues: &mut Vec<String>) {
    validate_required_path(name, "component1", &fixture.component1, issues);
    if let Some(component2) = &fixture.component2 {
        validate_required_path(name, "component2", component2, issues);
    }
    validate_optional_sha256(
        name,
        "component1_sha256",
        fixture.component1_sha256.as_deref(),
        issues,
    );
    validate_optional_sha256(
        name,
        "component2_sha256",
        fixture.component2_sha256.as_deref(),
        issues,
    );
    if fixture.component2_sha256.is_some() && fixture.component2.is_none() {
        issues.push(format!("{name}:component2_sha256_without_component2"));
    }
    if !fixture.additional_components.is_empty() && fixture.component2.is_none() {
        issues.push(format!("{name}:additional_components_without_component2"));
    }
    for (offset, component) in fixture.additional_components.iter().enumerate() {
        let component_number = offset + 3;
        validate_required_path(
            name,
            &format!("additional_components[{offset}].path"),
            &component.path,
            issues,
        );
        validate_optional_sha256(
            name,
            &format!("additional_components[{offset}].sha256"),
            component.sha256.as_deref(),
            issues,
        );
        if component.path == fixture.component1
            || fixture
                .component2
                .as_ref()
                .is_some_and(|component2| component.path == *component2)
        {
            issues.push(format!("{name}:component{component_number}_path_duplicate"));
        }
        if fixture.additional_components[..offset]
            .iter()
            .any(|previous| previous.path == component.path)
        {
            issues.push(format!("{name}:component{component_number}_path_duplicate"));
        }
    }
    validate_optional_path(name, "output_dir", fixture.output_dir.as_deref(), issues);
    validate_optional_input_mode_label(name, fixture.input_mode.as_deref(), issues);
    validate_optional_deskew_mode_label(name, fixture.deskew.as_deref(), issues);
    validate_optional_orientation_correction_label(
        name,
        fixture.orientation_correction.as_deref(),
        issues,
    );
    if fixture
        .bit_depth
        .is_some_and(|bit_depth| !matches!(bit_depth, 14 | 16))
    {
        issues.push(format!("{name}:bit_depth_invalid"));
    }
    let fixture_grain_mode = fixture
        .grain_reduction
        .as_deref()
        .and_then(parse_grain_reduction_mode_label);
    if let Some(label) = fixture.grain_reduction.as_deref() {
        if label.trim().is_empty() {
            issues.push(format!("{name}:grain_reduction_empty"));
        } else if fixture_grain_mode.is_none() {
            issues.push(format!("{name}:grain_reduction_invalid:{label}"));
        }
    }
    if fixture
        .grain_strength
        .is_some_and(|strength| !strength.is_finite() || !(0.0..=1.0).contains(&strength))
    {
        issues.push(format!("{name}:grain_strength_invalid"));
    }
    if fixture
        .grain_scale
        .is_some_and(|scale| !scale.is_finite() || !(0.5..=4.0).contains(&scale))
    {
        issues.push(format!("{name}:grain_scale_invalid"));
    }
    if let (Some(mode), Some(expected_enabled)) = (
        fixture_grain_mode,
        fixture.expectations.grain_reduction_enabled,
    ) {
        if mode.enabled() != expected_enabled {
            issues.push(format!(
                "{name}:grain_reduction_conflicts_with_enabled_expectation"
            ));
        }
    }
    if fixture
        .deskew_angle_degrees
        .is_some_and(|angle| !angle.is_finite() || !(-3.0..=3.0).contains(&angle))
    {
        issues.push(format!("{name}:deskew_angle_degrees_invalid"));
    }
    if fixture.deskew_angle_degrees.is_some() && fixture.deskew.as_deref() != Some("manual") {
        issues.push(format!(
            "{name}:deskew_angle_degrees_requires_manual_deskew"
        ));
    }
    if fixture.force_stitch && fixture.force_no_stitch {
        issues.push(format!("{name}:force_stitch_and_force_no_stitch"));
    }
    if fixture.force_stitch && fixture.component_count() < 2 {
        issues.push(format!("{name}:force_stitch_requires_multiple_components"));
    }
    validate_optional_path(
        name,
        "summary_baseline",
        fixture.summary_baseline.as_deref(),
        issues,
    );
    validate_optional_sha256(
        name,
        "summary_baseline_sha256",
        fixture.summary_baseline_sha256.as_deref(),
        issues,
    );
    validate_optional_path(
        name,
        "render_review",
        fixture.render_review.as_deref(),
        issues,
    );
    validate_optional_sha256(
        name,
        "render_review_sha256",
        fixture.render_review_sha256.as_deref(),
        issues,
    );
    validate_optional_path(
        name,
        "calibration_profile",
        fixture.calibration_profile.as_deref(),
        issues,
    );
    validate_optional_sha256(
        name,
        "calibration_profile_sha256",
        fixture.calibration_profile_sha256.as_deref(),
        issues,
    );
    validate_optional_path(
        name,
        "calibration_library",
        fixture.calibration_library.as_deref(),
        issues,
    );
    validate_optional_sha256(
        name,
        "calibration_library_sha256",
        fixture.calibration_library_sha256.as_deref(),
        issues,
    );
    validate_optional_label(
        name,
        "scanner_profile",
        fixture.scanner_profile.as_deref(),
        issues,
    );
    validate_optional_label(
        name,
        "roll_profile",
        fixture.roll_profile.as_deref(),
        issues,
    );
    validate_optional_label(name, "film_stock", fixture.film_stock.as_deref(), issues);
    validate_optional_label(
        name,
        "calibration_case",
        fixture.calibration_case.as_deref(),
        issues,
    );
    validate_fixture_expectations(
        name,
        &fixture.expectations,
        fixture.component_count(),
        issues,
    );
    validate_optional_label(name, "description", fixture.description.as_deref(), issues);
    validate_unique_non_empty_values(&format!("{name}:scene_tag"), &fixture.scene_tags, issues);
    validate_unique_non_empty_values(
        &format!("{name}:exposure_tag"),
        &fixture.exposure_tags,
        issues,
    );
    validate_unique_non_empty_values(
        &format!("{name}:reference_evidence"),
        &fixture.reference_evidence,
        issues,
    );

    if fixture.calibration_profile.is_some() && fixture.calibration_library.is_some() {
        issues.push(format!(
            "{name}:calibration_profile_and_calibration_library_both_declared"
        ));
    }
    if fixture.component2.is_some()
        && fixture.component1_sha256.is_some() != fixture.component2_sha256.is_some()
    {
        issues.push(format!("{name}:component_sha256_pair_incomplete"));
    }
    let declared_component_hashes = fixture
        .component_hashes()
        .iter()
        .filter(|hash| hash.is_some())
        .count();
    if declared_component_hashes > 0 && declared_component_hashes != fixture.component_count() {
        issues.push(format!("{name}:component_sha256_set_incomplete"));
    }
    if fixture.summary_baseline_sha256.is_some() && fixture.summary_baseline.is_none() {
        issues.push(format!(
            "{name}:summary_baseline_sha256_without_summary_baseline"
        ));
    }
    if fixture.render_review_sha256.is_some() && fixture.render_review.is_none() {
        issues.push(format!("{name}:render_review_sha256_without_render_review"));
    }
    if fixture.calibration_profile_sha256.is_some() && fixture.calibration_profile.is_none() {
        issues.push(format!(
            "{name}:calibration_profile_sha256_without_calibration_profile"
        ));
    }
    if fixture.calibration_library_sha256.is_some() && fixture.calibration_library.is_none() {
        issues.push(format!(
            "{name}:calibration_library_sha256_without_calibration_library"
        ));
    }
    if fixture.calibration_library.is_none()
        && (fixture.scanner_profile.is_some() || fixture.roll_profile.is_some())
    {
        issues.push(format!("{name}:profile_id_without_calibration_library"));
    }
    if fixture.roll_profile.is_some() && fixture.scanner_profile.is_none() {
        issues.push(format!("{name}:roll_profile_without_scanner_profile"));
    }

    match fixture.calibration_case.as_deref() {
        Some("scanner-roll-library") => {
            if fixture.calibration_library.is_none() {
                issues.push(format!(
                    "{name}:scanner_roll_library_case_without_calibration_library"
                ));
            }
            if fixture.scanner_profile.is_none() {
                issues.push(format!(
                    "{name}:scanner_roll_library_case_without_scanner_profile"
                ));
            }
        }
        Some("external-profile") => {
            if fixture.calibration_profile.is_none() {
                issues.push(format!(
                    "{name}:external_profile_case_without_calibration_profile"
                ));
            }
        }
        Some("uncalibrated-image-derived") => {
            if fixture.calibration_profile.is_some() {
                issues.push(format!(
                    "{name}:uncalibrated_case_declares_calibration_profile"
                ));
            }
            if fixture.calibration_library.is_some() {
                issues.push(format!(
                    "{name}:uncalibrated_case_declares_calibration_library"
                ));
            }
            if fixture.scanner_profile.is_some() {
                issues.push(format!("{name}:uncalibrated_case_declares_scanner_profile"));
            }
            if fixture.roll_profile.is_some() {
                issues.push(format!("{name}:uncalibrated_case_declares_roll_profile"));
            }
        }
        _ => {}
    }
}

fn validate_fixture_expectations(
    fixture_name: &str,
    expectations: &FixtureExpectations,
    component_count: usize,
    issues: &mut Vec<String>,
) {
    validate_optional_label(
        fixture_name,
        "expectations.deskew_status",
        expectations.deskew_status.as_deref(),
        issues,
    );
    if expectations
        .deskew_retained_area_ratio_min
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.deskew_retained_area_ratio_min_out_of_range"
        ));
    }
    for (field, value) in [
        (
            "deskew_minimum_component_retained_area_ratio_min",
            expectations.deskew_minimum_component_retained_area_ratio_min,
        ),
        (
            "border_crop_retained_area_ratio_min",
            expectations.border_crop_retained_area_ratio_min,
        ),
        (
            "border_crop_retained_area_ratio_max",
            expectations.border_crop_retained_area_ratio_max,
        ),
    ] {
        if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
            issues.push(format!("{fixture_name}:expectations.{field}_out_of_range"));
        }
    }
    if expectations
        .border_crop_retained_area_ratio_min
        .zip(expectations.border_crop_retained_area_ratio_max)
        .is_some_and(|(minimum, maximum)| minimum > maximum)
    {
        issues.push(format!(
            "{fixture_name}:expectations.border_crop_retained_area_ratio_bounds_inverted"
        ));
    }
    if expectations.border_crop_minimum_removed_edge_count_per_component_min == Some(0) {
        issues.push(format!(
            "{fixture_name}:expectations.border_crop_minimum_removed_edge_count_per_component_min_out_of_range"
        ));
    }
    if expectations
        .deskew_correction_degrees_expected
        .is_some_and(|value| !value.is_finite() || !(-3.0..=3.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.deskew_correction_degrees_expected_out_of_range"
        ));
    }
    if expectations
        .deskew_correction_tolerance_degrees
        .is_some_and(|value| !value.is_finite() || !(0.0..=3.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.deskew_correction_tolerance_degrees_out_of_range"
        ));
    }
    let mut crop_component_indices = BTreeSet::new();
    for component in &expectations.border_crop_components_expected {
        if !(1..=component_count).contains(&component.component_index) {
            issues.push(format!(
                "{fixture_name}:expectations.border_crop_components_expected_component_out_of_range:{}",
                component.component_index
            ));
        } else if !crop_component_indices.insert(component.component_index) {
            issues.push(format!(
                "{fixture_name}:expectations.border_crop_components_expected_component_duplicate:{}",
                component.component_index
            ));
        }
        if component.tolerance_px > 64 {
            issues.push(format!(
                "{fixture_name}:expectations.border_crop_components_expected_tolerance_out_of_range:{}",
                component.component_index
            ));
        }
    }
    let mut orientation_component_indices = BTreeSet::new();
    for component in &expectations.orientation_components_expected {
        if !(1..=component_count).contains(&component.component_index) {
            issues.push(format!(
                "{fixture_name}:expectations.orientation_components_expected_component_out_of_range:{}",
                component.component_index
            ));
        } else if !orientation_component_indices.insert(component.component_index) {
            issues.push(format!(
                "{fixture_name}:expectations.orientation_components_expected_component_duplicate:{}",
                component.component_index
            ));
        }
        if component.tag_present != component.tag_value.is_some()
            || component
                .tag_value
                .is_some_and(|value| !(1..=8).contains(&value))
        {
            issues.push(format!(
                "{fixture_name}:expectations.orientation_components_expected_tag_invalid:{}",
                component.component_index
            ));
        }
        if component.transform.trim().is_empty() {
            issues.push(format!(
                "{fixture_name}:expectations.orientation_components_expected_transform_empty:{}",
                component.component_index
            ));
        }
        if !is_valid_sha256_hex(&component.decoded_pixel_sha256) {
            issues.push(format!(
                "{fixture_name}:expectations.orientation_components_expected_decoded_pixel_sha256_invalid:{}",
                component.component_index
            ));
        }
        if component.source_width == 0
            || component.source_height == 0
            || component.output_width == 0
            || component.output_height == 0
        {
            issues.push(format!(
                "{fixture_name}:expectations.orientation_components_expected_dimensions_out_of_range:{}",
                component.component_index
            ));
        }
    }
    validate_optional_label(
        fixture_name,
        "expectations.stitch_decision",
        expectations.stitch_decision.as_deref(),
        issues,
    );
    validate_optional_label(
        fixture_name,
        "expectations.technical_white_balance_status",
        expectations.technical_white_balance_status.as_deref(),
        issues,
    );
    if !expectations.inferred_component_order.is_empty() {
        if expectations.inferred_component_order.len() != component_count {
            issues.push(format!(
                "{fixture_name}:expectations.inferred_component_order_length_mismatch"
            ));
        }
        let mut seen = BTreeSet::new();
        for &component_number in &expectations.inferred_component_order {
            if !(1..=component_count).contains(&component_number) {
                issues.push(format!(
                    "{fixture_name}:expectations.inferred_component_order_out_of_range:{component_number}"
                ));
            } else if !seen.insert(component_number) {
                issues.push(format!(
                    "{fixture_name}:expectations.inferred_component_order_duplicate:{component_number}"
                ));
            }
        }
    }
    if expectations
        .creative_temperature
        .is_some_and(|value| !value.is_finite() || !(-1.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.creative_temperature_out_of_range"
        ));
    }
    if expectations
        .creative_tint
        .is_some_and(|value| !value.is_finite() || !(-1.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.creative_tint_out_of_range"
        ));
    }
    validate_optional_label(
        fixture_name,
        "expectations.seam_exposure_model",
        expectations.seam_exposure_model.as_deref(),
        issues,
    );
    if expectations
        .seam_exposure_held_out_improvement_over_gain_min
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_held_out_improvement_over_gain_min_out_of_range"
        ));
    }
    if expectations
        .seam_exposure_offset_normalized_abs_max
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_offset_normalized_abs_max_out_of_range"
        ));
    }
    if expectations
        .seam_exposure_spatial_slope_abs_min
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_spatial_slope_abs_min_out_of_range"
        ));
    }
    if expectations
        .seam_exposure_spatial_slope_abs_max
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_spatial_slope_abs_max_out_of_range"
        ));
    }
    if expectations
        .seam_exposure_spatial_slope_abs_min
        .zip(expectations.seam_exposure_spatial_slope_abs_max)
        .is_some_and(|(minimum, maximum)| minimum > maximum)
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_spatial_slope_bounds_inverted"
        ));
    }
    if expectations
        .seam_exposure_spatial_slope_agreement_ratio_min
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_spatial_slope_agreement_ratio_min_out_of_range"
        ));
    }
    if expectations
        .seam_exposure_held_out_spatial_improvement_over_best_constant_min
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_held_out_spatial_improvement_over_best_constant_min_out_of_range"
        ));
    }
    if expectations
        .seam_exposure_spatial_offset_slope_normalized_abs_min
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_spatial_offset_slope_normalized_abs_min_out_of_range"
        ));
    }
    if expectations
        .seam_exposure_spatial_offset_slope_normalized_abs_max
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_spatial_offset_slope_normalized_abs_max_out_of_range"
        ));
    }
    if expectations
        .seam_exposure_spatial_offset_slope_normalized_abs_min
        .zip(expectations.seam_exposure_spatial_offset_slope_normalized_abs_max)
        .is_some_and(|(minimum, maximum)| minimum > maximum)
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_spatial_offset_slope_bounds_inverted"
        ));
    }
    if expectations
        .seam_exposure_spatial_offset_endpoint_normalized_abs_max
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_spatial_offset_endpoint_normalized_abs_max_out_of_range"
        ));
    }
    if expectations
        .seam_exposure_spatial_affine_slope_agreement_ratio_min
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_spatial_affine_slope_agreement_ratio_min_out_of_range"
        ));
    }
    if expectations
        .seam_exposure_spatial_affine_center_offset_delta_normalized_max
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_spatial_affine_center_offset_delta_normalized_max_out_of_range"
        ));
    }
    if expectations
        .seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler_min
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler_min_out_of_range"
        ));
    }
    if expectations
        .seam_exposure_spatial_2d_horizontal_slope_agreement_ratio_min
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_spatial_2d_horizontal_slope_agreement_ratio_min_out_of_range"
        ));
    }
    if expectations.seam_exposure_spatial_2d_distinct_columns_min == Some(0) {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_spatial_2d_distinct_columns_min_out_of_range"
        ));
    }
    if expectations
        .seam_exposure_held_out_spatial_2d_improvement_over_best_simpler_min
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_exposure_held_out_spatial_2d_improvement_over_best_simpler_min_out_of_range"
        ));
    }
    validate_optional_label(
        fixture_name,
        "expectations.seam_blend_mode",
        expectations.seam_blend_mode.as_deref(),
        issues,
    );
    validate_optional_label(
        fixture_name,
        "expectations.base_estimate_source",
        expectations.base_estimate_source.as_deref(),
        issues,
    );
    for (field, value) in [
        ("base_confidence_min", expectations.base_confidence_min),
        (
            "negative_response_measured_confidence_min",
            expectations.negative_response_measured_confidence_min,
        ),
        (
            "negative_response_curve_extrapolated_ratio_max",
            expectations.negative_response_curve_extrapolated_ratio_max,
        ),
    ] {
        if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
            issues.push(format!("{fixture_name}:expectations.{field}_out_of_range"));
        }
    }
    for (field, value) in [
        (
            "negative_response_held_out_delta_e00_rms_max",
            expectations.negative_response_held_out_delta_e00_rms_max,
        ),
        (
            "negative_response_held_out_max_delta_e00_max",
            expectations.negative_response_held_out_max_delta_e00_max,
        ),
        (
            "negative_response_held_out_improvement_over_unit_slope_min",
            expectations.negative_response_held_out_improvement_over_unit_slope_min,
        ),
        (
            "negative_response_density_noise_gain_max",
            expectations.negative_response_density_noise_gain_max,
        ),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            issues.push(format!("{fixture_name}:expectations.{field}_out_of_range"));
        }
    }
    for (field, value) in [
        (
            "expectations.negative_response_model",
            expectations.negative_response_model.as_deref(),
        ),
        (
            "expectations.negative_response_source",
            expectations.negative_response_source.as_deref(),
        ),
        (
            "expectations.negative_response_crosstalk_model",
            expectations.negative_response_crosstalk_model.as_deref(),
        ),
        (
            "expectations.negative_response_characteristic_curve_model",
            expectations
                .negative_response_characteristic_curve_model
                .as_deref(),
        ),
        (
            "expectations.negative_response_measured_model_id",
            expectations.negative_response_measured_model_id.as_deref(),
        ),
        (
            "expectations.negative_response_curve_interpolation",
            expectations
                .negative_response_curve_interpolation
                .as_deref(),
        ),
    ] {
        validate_optional_label(fixture_name, field, value, issues);
    }
    validate_optional_label(
        fixture_name,
        "expectations.output_color_space",
        expectations.output_color_space.as_deref(),
        issues,
    );
    validate_optional_label(
        fixture_name,
        "expectations.render_input_source",
        expectations.render_input_source.as_deref(),
        issues,
    );
    validate_optional_label(
        fixture_name,
        "expectations.render_input_reason_contains",
        expectations.render_input_reason_contains.as_deref(),
        issues,
    );
    validate_optional_label(
        fixture_name,
        "expectations.mapping_strategy",
        expectations.mapping_strategy.as_deref(),
        issues,
    );
    validate_optional_label(
        fixture_name,
        "expectations.selected_mapping_reason_contains",
        expectations.selected_mapping_reason_contains.as_deref(),
        issues,
    );
    validate_optional_label(
        fixture_name,
        "expectations.selected_candidate",
        expectations.selected_candidate.as_deref(),
        issues,
    );
    validate_optional_label(
        fixture_name,
        "expectations.calibration_acceptance_status",
        expectations.calibration_acceptance_status.as_deref(),
        issues,
    );
    validate_optional_label(
        fixture_name,
        "expectations.candidate_risk",
        expectations.candidate_risk.as_deref(),
        issues,
    );
    validate_optional_label(
        fixture_name,
        "expectations.tone_color_trust_state",
        expectations.tone_color_trust_state.as_deref(),
        issues,
    );
    validate_optional_label(
        fixture_name,
        "expectations.neutral_safety_rescue_reason_contains",
        expectations
            .neutral_safety_rescue_reason_contains
            .as_deref(),
        issues,
    );
    validate_optional_label(
        fixture_name,
        "expectations.render_review_status",
        expectations.render_review_status.as_deref(),
        issues,
    );
    validate_optional_label(
        fixture_name,
        "expectations.tone_output_confidence_status",
        expectations.tone_output_confidence_status.as_deref(),
        issues,
    );
    validate_unique_non_empty_values(
        &format!("{fixture_name}:expectations.debug_artifact_kinds_required"),
        &expectations.debug_artifact_kinds_required,
        issues,
    );
    validate_unique_non_empty_values(
        &format!("{fixture_name}:expectations.candidate_acceptance_signatures_required"),
        &expectations.candidate_acceptance_signatures_required,
        issues,
    );
    validate_unique_non_empty_values(
        &format!("{fixture_name}:expectations.calibration_rejection_details_required"),
        &expectations.calibration_rejection_details_required,
        issues,
    );
    validate_unique_non_empty_values(
        &format!("{fixture_name}:expectations.selection_rejections_required"),
        &expectations.selection_rejections_required,
        issues,
    );
    for kind in &expectations.debug_artifact_kinds_required {
        let kind = kind.trim();
        if !kind.is_empty() && !is_known_debug_artifact_kind(kind) {
            issues.push(format!(
                "{fixture_name}:expectations.debug_artifact_kinds_required_unknown:{kind}"
            ));
        }
    }
    if expectations
        .selected_quality_score_max
        .is_some_and(|value| value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.selected_quality_score_max_out_of_range"
        ));
    }
    if expectations
        .seam_detail_supported_scale_count_min
        .is_some_and(|value| value == 0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_detail_supported_scale_count_min_out_of_range"
        ));
    }
    if expectations
        .seam_detail_max_symmetric_energy_ratio_max
        .is_some_and(|value| !value.is_finite() || value < 1.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_detail_max_symmetric_energy_ratio_max_out_of_range"
        ));
    }
    if expectations
        .seam_gradient_ratio_max
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_gradient_ratio_max_out_of_range"
        ));
    }
    if expectations
        .seam_overlap_p95_abs_difference_max
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.seam_overlap_p95_abs_difference_max_out_of_range"
        ));
    }
    for (field, value) in [
        (
            "grain_reduction_applied_ratio_min",
            expectations.grain_reduction_applied_ratio_min,
        ),
        (
            "grain_reduction_structure_excluded_ratio_min",
            expectations.grain_reduction_structure_excluded_ratio_min,
        ),
        (
            "grain_reduction_flat_luma_p95_reduction_ratio_min",
            expectations.grain_reduction_flat_luma_p95_reduction_ratio_min,
        ),
        (
            "grain_reduction_flat_chroma_p95_reduction_ratio_min",
            expectations.grain_reduction_flat_chroma_p95_reduction_ratio_min,
        ),
    ] {
        if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
            issues.push(format!("{fixture_name}:expectations.{field}_out_of_range"));
        }
    }
    for (field, value) in [
        (
            "grain_detail_luminance_probe_count_min",
            expectations.grain_detail_luminance_probe_count_min,
        ),
        (
            "grain_detail_chroma_probe_count_min",
            expectations.grain_detail_chroma_probe_count_min,
        ),
    ] {
        if value.is_some_and(|value| value == 0) {
            issues.push(format!("{fixture_name}:expectations.{field}_out_of_range"));
        }
    }
    for (field, value) in [
        (
            "grain_detail_luminance_p10_retention_min",
            expectations.grain_detail_luminance_p10_retention_min,
        ),
        (
            "grain_detail_chroma_p10_retention_min",
            expectations.grain_detail_chroma_p10_retention_min,
        ),
    ] {
        if value.is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value)) {
            issues.push(format!("{fixture_name}:expectations.{field}_out_of_range"));
        }
    }
    if expectations
        .neutral_safety_rescue_preserved_ratio_gain_min
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.neutral_safety_rescue_preserved_ratio_gain_min_out_of_range"
        ));
    }
    if expectations
        .neutral_safety_rescue_midtone_saturation_p95_reduction_min
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.neutral_safety_rescue_midtone_saturation_p95_reduction_min_out_of_range"
        ));
    }
    if expectations
        .highlight_chroma_compressed_ratio_min
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.highlight_chroma_compressed_ratio_min_out_of_range"
        ));
    }
    if expectations
        .highlight_chroma_compressed_ratio_max
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.highlight_chroma_compressed_ratio_max_out_of_range"
        ));
    }
    if expectations
        .highlight_neutral_chroma_compressed_ratio_max
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.highlight_neutral_chroma_compressed_ratio_max_out_of_range"
        ));
    }
    if expectations
        .shadow_chroma_compressed_ratio_max
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.shadow_chroma_compressed_ratio_max_out_of_range"
        ));
    }
    if expectations
        .calibration_confidence_min
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.calibration_confidence_min_out_of_range"
        ));
    }
    if expectations
        .calibration_matrix_condition_number_max
        .is_some_and(|value| value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.calibration_matrix_condition_number_max_out_of_range"
        ));
    }
    if expectations
        .technical_safety_score_max
        .is_some_and(|value| value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.technical_safety_score_max_out_of_range"
        ));
    }
    if expectations
        .color_fidelity_score_max
        .is_some_and(|value| value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.color_fidelity_score_max_out_of_range"
        ));
    }
    if expectations
        .memory_color_penalty_max
        .is_some_and(|value| value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.memory_color_penalty_max_out_of_range"
        ));
    }
    if expectations
        .spatial_consistency_penalty_max
        .is_some_and(|value| value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.spatial_consistency_penalty_max_out_of_range"
        ));
    }
    if expectations
        .selected_runner_up_quality_delta_min
        .is_some_and(|value| value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.selected_runner_up_quality_delta_min_out_of_range"
        ));
    }
    if expectations
        .density_monotonicity_score_min
        .is_some_and(|value| value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.density_monotonicity_score_min_out_of_range"
        ));
    }
    if expectations
        .hue_linearity_score_min
        .is_some_and(|value| value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.hue_linearity_score_min_out_of_range"
        ));
    }
    if expectations
        .saturation_preservation_median_ratio_min
        .is_some_and(|value| value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.saturation_preservation_median_ratio_min_out_of_range"
        ));
    }
    if expectations
        .spatial_neutral_delta_p95_max
        .is_some_and(|value| value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.spatial_neutral_delta_p95_max_out_of_range"
        ));
    }
    if expectations
        .post_scale_preserved_ratio_min
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.post_scale_preserved_ratio_min_out_of_range"
        ));
    }
    if expectations
        .render_luminance_range_p05_p95_min
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.render_luminance_range_p05_p95_min_out_of_range"
        ));
    }
    if expectations
        .tone_output_evidence_confidence_min
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.tone_output_evidence_confidence_min_out_of_range"
        ));
    }
    if expectations
        .render_to_mapped_luminance_range_ratio_min
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.render_to_mapped_luminance_range_ratio_min_out_of_range"
        ));
    }
    if expectations
        .post_chroma_compression_clipped_high_ratio_max
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.post_chroma_compression_clipped_high_ratio_max_out_of_range"
        ));
    }
    if expectations
        .post_chroma_compression_clipped_low_ratio_max
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        issues.push(format!(
            "{fixture_name}:expectations.post_chroma_compression_clipped_low_ratio_max_out_of_range"
        ));
    }
    if expectations
        .reference_patch_rms_delta_e_max
        .is_some_and(|value| value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.reference_patch_rms_delta_e_max_out_of_range"
        ));
    }
    if expectations
        .reference_patch_rms_delta_e2000_max
        .is_some_and(|value| value < 0.0)
    {
        issues.push(format!(
            "{fixture_name}:expectations.reference_patch_rms_delta_e2000_max_out_of_range"
        ));
    }
}

fn is_known_debug_artifact_kind(kind: &str) -> bool {
    matches!(
        kind,
        "candidate_comparison" | "gamut_clipping_map" | "scene_referred_prophoto_float"
    )
}

fn validate_required_path(
    fixture_name: &str,
    field_name: &str,
    path: &Path,
    issues: &mut Vec<String>,
) {
    if path.as_os_str().is_empty() {
        issues.push(format!("{fixture_name}:{field_name}_empty"));
    }
}

fn validate_optional_path(
    fixture_name: &str,
    field_name: &str,
    path: Option<&Path>,
    issues: &mut Vec<String>,
) {
    if path.is_some_and(|path| path.as_os_str().is_empty()) {
        issues.push(format!("{fixture_name}:{field_name}_empty"));
    }
}

fn validate_optional_label(
    fixture_name: &str,
    field_name: &str,
    value: Option<&str>,
    issues: &mut Vec<String>,
) {
    if value.is_some_and(|value| value.trim().is_empty()) {
        issues.push(format!("{fixture_name}:{field_name}_empty"));
    }
}

fn validate_optional_input_mode_label(
    fixture_name: &str,
    value: Option<&str>,
    issues: &mut Vec<String>,
) {
    if let Some(value) = value {
        if value.trim().is_empty() {
            issues.push(format!("{fixture_name}:input_mode_empty"));
        } else if parse_input_mode_label(value).is_none() {
            issues.push(format!("{fixture_name}:input_mode_invalid:{value}"));
        }
    }
}

fn validate_optional_deskew_mode_label(
    fixture_name: &str,
    value: Option<&str>,
    issues: &mut Vec<String>,
) {
    if let Some(value) = value {
        if value.trim().is_empty() {
            issues.push(format!("{fixture_name}:deskew_empty"));
        } else if parse_deskew_mode_label(value).is_none() {
            issues.push(format!("{fixture_name}:deskew_invalid:{value}"));
        }
    }
}

fn validate_optional_orientation_correction_label(
    fixture_name: &str,
    value: Option<&str>,
    issues: &mut Vec<String>,
) {
    if let Some(value) = value {
        if value.trim().is_empty() {
            issues.push(format!("{fixture_name}:orientation_correction_empty"));
        } else if parse_orientation_correction_label(value).is_none() {
            issues.push(format!(
                "{fixture_name}:orientation_correction_invalid:{value}"
            ));
        }
    }
}

fn validate_optional_sha256(
    fixture_name: &str,
    field_name: &str,
    value: Option<&str>,
    issues: &mut Vec<String>,
) {
    if value.is_some_and(|value| !is_valid_sha256_hex(value)) {
        issues.push(format!("{fixture_name}:{field_name}_invalid"));
    }
}

fn is_valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_unique_non_empty_values(prefix: &str, values: &[String], issues: &mut Vec<String>) {
    let mut seen = BTreeSet::new();
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            issues.push(format!("{prefix}_empty"));
        } else if !seen.insert(trimmed) {
            issues.push(format!("{prefix}_duplicate:{trimmed}"));
        }
    }
}

fn validate_required_pair_labels(prefix: &str, values: &[String], issues: &mut Vec<String>) {
    validate_unique_non_empty_values(prefix, values, issues);
    for value in values {
        if !is_valid_pair_label(value) {
            issues.push(format!("{prefix}_invalid:{value}"));
        }
    }
}

fn is_valid_pair_label(value: &str) -> bool {
    let Some((left, right)) = value.split_once('|') else {
        return false;
    };
    !left.trim().is_empty() && !right.trim().is_empty() && !right.contains('|')
}

fn fixture_calibration_label(fixture: &FixtureEntry) -> String {
    if let Some(profile) = &fixture.calibration_profile {
        format!("profile={}", profile.display())
    } else if let Some(library) = &fixture.calibration_library {
        format!("library={}", library.display())
    } else {
        String::new()
    }
}

fn fixture_coverage_summary(
    fixtures: &BTreeMap<String, FixtureEntry>,
    requirements: &FixtureCoverageRequirements,
    compute_fixture_hashes: bool,
) -> FixtureCoverageSummary {
    let mut entries = Vec::new();
    let mut issues = Vec::new();

    for (name, fixture) in fixtures {
        let entry = fixture_coverage_entry(name, fixture, requirements, compute_fixture_hashes);
        issues.extend(entry.issues.iter().cloned());
        entries.push(entry);
    }

    let component_file_count = entries.iter().map(|entry| entry.component_count).sum();
    let max_component_count = entries
        .iter()
        .map(|entry| entry.component_count)
        .max()
        .unwrap_or(0);
    let n_component_fixture_count = entries
        .iter()
        .filter(|entry| entry.component_count > 2)
        .count();
    let n_component_validation_ready_fixture_count = entries
        .iter()
        .filter(|entry| entry.component_count > 2 && entry.validation_ready)
        .count();
    let component_set_available_count = entries
        .iter()
        .filter(|entry| entry.all_components_exist)
        .count();
    let component_sha256_declared_set_count = entries
        .iter()
        .filter(|entry| entry.all_component_sha256_declared)
        .count();
    let component_sha256_computed_set_count = entries
        .iter()
        .filter(|entry| entry.all_component_sha256_computed)
        .count();
    let component_sha256_set_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.all_component_sha256_matched)
        .count();
    let readable_tiff_set_count = entries
        .iter()
        .filter(|entry| entry.all_components_readable_tiff)
        .count();
    let tiff_layout_consistent_set_count = entries
        .iter()
        .filter(|entry| entry.all_component_layouts_consistent)
        .count();
    let tiff_dimension_matched_set_count = entries
        .iter()
        .filter(|entry| entry.all_component_dimensions_matched)
        .count();
    let component_pair_available_count = entries
        .iter()
        .filter(|entry| entry.component1_exists && entry.component2_exists)
        .count();
    let component_sha256_declared_pair_count = entries
        .iter()
        .filter(|entry| {
            entry
                .component1_sha256
                .as_ref()
                .is_some_and(|probe| probe.expected_sha256.is_some())
                && entry
                    .component2_sha256
                    .as_ref()
                    .is_some_and(|probe| probe.expected_sha256.is_some())
        })
        .count();
    let component_sha256_computed_pair_count = entries
        .iter()
        .filter(|entry| {
            entry
                .component1_sha256
                .as_ref()
                .is_some_and(|probe| probe.actual_sha256.is_some())
                && entry
                    .component2_sha256
                    .as_ref()
                    .is_some_and(|probe| probe.actual_sha256.is_some())
        })
        .count();
    let component_sha256_pair_count = entries
        .iter()
        .filter(|entry| {
            entry.validation_ready
                && entry
                    .component1_sha256
                    .as_ref()
                    .is_some_and(|probe| probe.matched)
                && entry
                    .component2_sha256
                    .as_ref()
                    .is_some_and(|probe| probe.matched)
        })
        .count();
    let readable_tiff_pair_count = entries
        .iter()
        .filter(|entry| {
            entry
                .component1_tiff
                .as_ref()
                .is_some_and(|probe| probe.readable)
                && entry
                    .component2_tiff
                    .as_ref()
                    .is_some_and(|probe| probe.readable)
        })
        .count();
    let tiff_layout_consistent_pair_count = entries
        .iter()
        .filter(|entry| {
            entry
                .tiff_pair
                .as_ref()
                .is_some_and(|pair| pair.layout_consistent)
        })
        .count();
    let tiff_dimension_matched_pair_count = entries
        .iter()
        .filter(|entry| {
            entry
                .tiff_pair
                .as_ref()
                .is_some_and(|pair| pair.dimension_matched)
        })
        .count();
    let validation_ready_fixture_count = entries
        .iter()
        .filter(|entry| entry.validation_ready)
        .count();
    let render_dynamic_range_contract_fixture_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.render_dynamic_range_contract_complete)
        .count();
    let stitch_normalization_contract_fixture_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.stitch_normalization_contract_complete)
        .count();
    let geometry_preparation_contract_fixture_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.geometry_preparation_contract_complete)
        .count();
    let geometry_accuracy_contract_fixture_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.geometry_accuracy_contract_complete)
        .count();
    let orientation_accuracy_contract_fixture_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.orientation_accuracy_contract_complete)
        .count();
    let negative_reconstruction_contract_fixture_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.negative_reconstruction_contract_complete)
        .count();
    let grain_reduction_enabled_fixture_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.grain_reduction_enabled_declared)
        .count();
    let grain_detail_contract_fixture_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.grain_detail_contract_complete)
        .count();
    let grain_reduction_effect_contract_fixture_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.grain_reduction_effect_contract_complete)
        .count();
    let summary_baseline_declared_count = entries
        .iter()
        .filter(|entry| entry.summary_baseline.is_some())
        .count();
    let summary_baseline_file_count = entries
        .iter()
        .filter(|entry| entry.summary_baseline_exists == Some(true))
        .count();
    let summary_baseline_parseable_count = entries
        .iter()
        .filter(|entry| entry.summary_baseline_parse_status.as_deref() == Some("valid"))
        .count();
    let summary_baseline_contract_complete_count = entries
        .iter()
        .filter(|entry| entry.summary_baseline_valid)
        .count();
    let summary_baseline_count = summary_baseline_contract_complete_count;
    let summary_baseline_sha256_declared_count = entries
        .iter()
        .filter(|entry| {
            entry
                .summary_baseline_sha256
                .as_ref()
                .is_some_and(|probe| probe.expected_sha256.is_some())
        })
        .count();
    let summary_baseline_sha256_computed_count = entries
        .iter()
        .filter(|entry| {
            entry
                .summary_baseline_sha256
                .as_ref()
                .is_some_and(|probe| probe.actual_sha256.is_some())
        })
        .count();
    let summary_baseline_sha256_count = entries
        .iter()
        .filter(|entry| {
            entry.validation_ready
                && entry
                    .summary_baseline_sha256
                    .as_ref()
                    .is_some_and(|probe| probe.matched)
        })
        .count();
    let calibration_evidence_declared_count = entries
        .iter()
        .filter(|entry| entry.calibration_evidence_declared)
        .count();
    let calibration_evidence_count = entries
        .iter()
        .filter(|entry| entry.calibration_evidence_usable)
        .count();
    let calibration_evidence_unusable_count =
        calibration_evidence_declared_count.saturating_sub(calibration_evidence_count);
    let calibration_sha256_declared_count = entries
        .iter()
        .filter(|entry| {
            fixture_calibration_sha256_probe(entry)
                .is_some_and(|probe| probe.expected_sha256.is_some())
        })
        .count();
    let calibration_sha256_computed_count = entries
        .iter()
        .filter(|entry| {
            fixture_calibration_sha256_probe(entry)
                .is_some_and(|probe| probe.actual_sha256.is_some())
        })
        .count();
    let calibration_sha256_count = entries
        .iter()
        .filter(|entry| {
            entry.validation_ready
                && fixture_calibration_sha256_probe(entry).is_some_and(|probe| probe.matched)
        })
        .count();
    let uncalibrated_fixture_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && !entry.calibration_evidence_declared)
        .count();
    let scanner_profile_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.scanner_profile.is_some())
        .count();
    let unique_scanner_profiles = entries
        .iter()
        .filter(|entry| entry.validation_ready)
        .filter_map(|entry| entry.scanner_profile.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let roll_profile_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.roll_profile.is_some())
        .count();
    let unique_roll_profiles = entries
        .iter()
        .filter(|entry| entry.validation_ready)
        .filter_map(|entry| entry.roll_profile.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let film_stock_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.film_stock.is_some())
        .count();
    let unique_film_stocks = entries
        .iter()
        .filter(|entry| entry.validation_ready)
        .filter_map(|entry| entry.film_stock.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let scene_tags = entries
        .iter()
        .filter(|entry| entry.validation_ready)
        .flat_map(|entry| entry.scene_tags.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let exposure_tags = entries
        .iter()
        .filter(|entry| entry.validation_ready)
        .flat_map(|entry| entry.exposure_tags.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let calibration_cases = entries
        .iter()
        .filter(|entry| entry.validation_ready)
        .filter_map(|entry| entry.calibration_case.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let reference_fixture_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && !entry.reference_evidence.is_empty())
        .count();
    let reference_evidence = entries
        .iter()
        .filter(|entry| entry.validation_ready)
        .flat_map(|entry| entry.reference_evidence.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let reference_patch_fixture_count = entries
        .iter()
        .filter(|entry| {
            entry.validation_ready && entry.calibration_reference_patch_count.unwrap_or(0) > 0
        })
        .count();
    let reference_patch_count = entries
        .iter()
        .filter(|entry| entry.validation_ready)
        .filter_map(|entry| entry.calibration_reference_patch_count)
        .sum::<usize>();
    let approved_render_review_fixture_count = entries
        .iter()
        .filter(|entry| entry.validation_ready && entry.approved_render_review)
        .count();
    let debug_artifact_expectation_fixture_count = entries
        .iter()
        .filter(|entry| {
            entry.validation_ready
                && (entry.debug_artifacts_required
                    || !entry.debug_artifact_kinds_required.is_empty())
        })
        .count();
    let debug_artifact_kinds_required = entries
        .iter()
        .filter(|entry| entry.validation_ready)
        .flat_map(|entry| entry.debug_artifact_kinds_required.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let missing_required_scanner_profiles = missing_required_values(
        &requirements.required_scanner_profiles,
        &unique_scanner_profiles,
    );
    let missing_required_roll_profiles =
        missing_required_values(&requirements.required_roll_profiles, &unique_roll_profiles);
    let missing_required_film_stocks =
        missing_required_values(&requirements.required_film_stocks, &unique_film_stocks);
    let missing_required_scene_tags =
        missing_required_values(&requirements.required_scene_tags, &scene_tags);
    let missing_required_exposure_tags =
        missing_required_values(&requirements.required_exposure_tags, &exposure_tags);
    let missing_required_calibration_cases =
        missing_required_values(&requirements.required_calibration_cases, &calibration_cases);
    let missing_required_reference_evidence = missing_required_values(
        &requirements.required_reference_evidence,
        &reference_evidence,
    );
    let film_stock_calibration_pairs = fixture_film_stock_calibration_pairs(&entries);
    let scene_exposure_pairs = fixture_scene_exposure_pairs(&entries);
    let missing_required_film_stock_calibration_pairs = missing_required_values(
        &requirements.required_film_stock_calibration_pairs,
        &film_stock_calibration_pairs,
    );
    let missing_required_scene_exposure_pairs = missing_required_values(
        &requirements.required_scene_exposure_pairs,
        &scene_exposure_pairs,
    );
    let missing_required_debug_artifact_kinds = missing_required_values(
        &requirements.required_debug_artifact_kinds,
        &debug_artifact_kinds_required,
    );
    let mut action_items = fixture_coverage_action_items(
        requirements,
        validation_ready_fixture_count,
        render_dynamic_range_contract_fixture_count,
        stitch_normalization_contract_fixture_count,
        geometry_preparation_contract_fixture_count,
        geometry_accuracy_contract_fixture_count,
        orientation_accuracy_contract_fixture_count,
        negative_reconstruction_contract_fixture_count,
        grain_reduction_enabled_fixture_count,
        grain_detail_contract_fixture_count,
        grain_reduction_effect_contract_fixture_count,
        n_component_validation_ready_fixture_count,
        component_sha256_set_count,
        component_pair_available_count,
        component_sha256_pair_count,
        readable_tiff_pair_count,
        tiff_layout_consistent_pair_count,
        tiff_dimension_matched_pair_count,
        summary_baseline_count,
        summary_baseline_sha256_count,
        calibration_evidence_count,
        calibration_sha256_count,
        uncalibrated_fixture_count,
        unique_scanner_profiles.len(),
        unique_roll_profiles.len(),
        unique_film_stocks.len(),
        scene_tags.len(),
        exposure_tags.len(),
        calibration_cases.len(),
        reference_fixture_count,
        reference_evidence.len(),
        reference_patch_fixture_count,
        reference_patch_count,
        approved_render_review_fixture_count,
        debug_artifact_expectation_fixture_count,
        film_stock_calibration_pairs.len(),
        scene_exposure_pairs.len(),
        &missing_required_film_stocks,
        &missing_required_scene_tags,
        &missing_required_exposure_tags,
        &missing_required_calibration_cases,
        &missing_required_scanner_profiles,
        &missing_required_roll_profiles,
        &missing_required_reference_evidence,
        &missing_required_debug_artifact_kinds,
        &missing_required_film_stock_calibration_pairs,
        &missing_required_scene_exposure_pairs,
    );
    push_fixture_coverage_entry_actions(&mut action_items, &entries);

    if calibration_evidence_count == 0 && uncalibrated_fixture_count == 0 {
        issues.push("fixture_coverage_no_calibrated_fixtures".to_string());
    }
    if validation_ready_fixture_count == 0 {
        issues.push("fixture_coverage_no_validation_ready_fixtures".to_string());
    }
    if film_stock_count == 0 {
        issues.push("fixture_coverage_no_film_stock_metadata".to_string());
    }
    if scene_tags.is_empty() {
        issues.push("fixture_coverage_no_scene_metadata".to_string());
    }
    if exposure_tags.is_empty() {
        issues.push("fixture_coverage_no_exposure_metadata".to_string());
    }
    if calibration_cases.is_empty() {
        issues.push("fixture_coverage_no_calibration_case_metadata".to_string());
    }
    apply_coverage_requirements(
        requirements,
        validation_ready_fixture_count,
        render_dynamic_range_contract_fixture_count,
        stitch_normalization_contract_fixture_count,
        geometry_preparation_contract_fixture_count,
        geometry_accuracy_contract_fixture_count,
        orientation_accuracy_contract_fixture_count,
        negative_reconstruction_contract_fixture_count,
        grain_reduction_enabled_fixture_count,
        grain_detail_contract_fixture_count,
        grain_reduction_effect_contract_fixture_count,
        n_component_validation_ready_fixture_count,
        component_sha256_set_count,
        component_pair_available_count,
        component_sha256_pair_count,
        readable_tiff_pair_count,
        tiff_layout_consistent_pair_count,
        tiff_dimension_matched_pair_count,
        summary_baseline_count,
        summary_baseline_sha256_count,
        calibration_evidence_count,
        calibration_sha256_count,
        uncalibrated_fixture_count,
        &unique_scanner_profiles,
        &unique_roll_profiles,
        &unique_film_stocks,
        &scene_tags,
        &exposure_tags,
        &calibration_cases,
        reference_fixture_count,
        &reference_evidence,
        reference_patch_fixture_count,
        reference_patch_count,
        approved_render_review_fixture_count,
        debug_artifact_expectation_fixture_count,
        &debug_artifact_kinds_required,
        &film_stock_calibration_pairs,
        &scene_exposure_pairs,
        &mut issues,
    );

    FixtureCoverageSummary {
        status: if issues.is_empty() {
            "passed".to_string()
        } else {
            "review_required".to_string()
        },
        fixture_count: entries.len(),
        component_file_count,
        max_component_count,
        n_component_fixture_count,
        n_component_validation_ready_fixture_count,
        component_set_available_count,
        component_sha256_declared_set_count,
        component_sha256_computed_set_count,
        component_sha256_set_count,
        readable_tiff_set_count,
        tiff_layout_consistent_set_count,
        tiff_dimension_matched_set_count,
        component_pair_available_count,
        component_sha256_declared_pair_count,
        component_sha256_computed_pair_count,
        component_sha256_pair_count,
        readable_tiff_pair_count,
        tiff_layout_consistent_pair_count,
        tiff_dimension_matched_pair_count,
        validation_ready_fixture_count,
        render_dynamic_range_contract_fixture_count,
        stitch_normalization_contract_fixture_count,
        geometry_preparation_contract_fixture_count,
        geometry_accuracy_contract_fixture_count,
        orientation_accuracy_contract_fixture_count,
        negative_reconstruction_contract_fixture_count,
        grain_reduction_enabled_fixture_count,
        grain_detail_contract_fixture_count,
        grain_reduction_effect_contract_fixture_count,
        summary_baseline_declared_count,
        summary_baseline_file_count,
        summary_baseline_parseable_count,
        summary_baseline_contract_complete_count,
        summary_baseline_count,
        summary_baseline_sha256_declared_count,
        summary_baseline_sha256_computed_count,
        summary_baseline_sha256_count,
        calibration_evidence_declared_count,
        calibration_evidence_count,
        calibration_evidence_unusable_count,
        calibration_sha256_declared_count,
        calibration_sha256_computed_count,
        calibration_sha256_count,
        uncalibrated_fixture_count,
        scanner_profile_count,
        unique_scanner_profiles,
        missing_required_scanner_profiles,
        roll_profile_count,
        unique_roll_profiles,
        missing_required_roll_profiles,
        film_stock_count,
        unique_film_stocks,
        missing_required_film_stocks,
        scene_tag_count: scene_tags.len(),
        scene_tags,
        missing_required_scene_tags,
        exposure_tag_count: exposure_tags.len(),
        exposure_tags,
        missing_required_exposure_tags,
        calibration_case_count: calibration_cases.len(),
        calibration_cases,
        missing_required_calibration_cases,
        reference_fixture_count,
        reference_evidence_type_count: reference_evidence.len(),
        reference_evidence,
        missing_required_reference_evidence,
        reference_patch_fixture_count,
        reference_patch_count,
        approved_render_review_fixture_count,
        debug_artifact_expectation_fixture_count,
        debug_artifact_kinds_required,
        missing_required_debug_artifact_kinds,
        film_stock_calibration_pair_count: film_stock_calibration_pairs.len(),
        film_stock_calibration_pairs,
        missing_required_film_stock_calibration_pairs,
        scene_exposure_pair_count: scene_exposure_pairs.len(),
        scene_exposure_pairs,
        missing_required_scene_exposure_pairs,
        coverage_requirements: requirements.clone(),
        fixtures: entries,
        action_items,
        issues,
    }
}

#[allow(clippy::too_many_arguments)]
fn fixture_coverage_action_items(
    requirements: &FixtureCoverageRequirements,
    validation_ready_fixture_count: usize,
    render_dynamic_range_contract_fixture_count: usize,
    stitch_normalization_contract_fixture_count: usize,
    geometry_preparation_contract_fixture_count: usize,
    geometry_accuracy_contract_fixture_count: usize,
    orientation_accuracy_contract_fixture_count: usize,
    negative_reconstruction_contract_fixture_count: usize,
    grain_reduction_enabled_fixture_count: usize,
    grain_detail_contract_fixture_count: usize,
    grain_reduction_effect_contract_fixture_count: usize,
    n_component_validation_ready_fixture_count: usize,
    component_sha256_set_count: usize,
    component_pair_available_count: usize,
    component_sha256_pair_count: usize,
    readable_tiff_pair_count: usize,
    tiff_layout_consistent_pair_count: usize,
    tiff_dimension_matched_pair_count: usize,
    summary_baseline_count: usize,
    summary_baseline_sha256_count: usize,
    calibration_evidence_count: usize,
    calibration_sha256_count: usize,
    uncalibrated_fixture_count: usize,
    unique_scanner_profile_count: usize,
    unique_roll_profile_count: usize,
    unique_film_stock_count: usize,
    scene_tag_count: usize,
    exposure_tag_count: usize,
    calibration_case_count: usize,
    reference_fixture_count: usize,
    reference_evidence_type_count: usize,
    reference_patch_fixture_count: usize,
    reference_patch_count: usize,
    approved_render_review_fixture_count: usize,
    debug_artifact_expectation_fixture_count: usize,
    film_stock_calibration_pair_count: usize,
    scene_exposure_pair_count: usize,
    missing_required_film_stocks: &[String],
    missing_required_scene_tags: &[String],
    missing_required_exposure_tags: &[String],
    missing_required_calibration_cases: &[String],
    missing_required_scanner_profiles: &[String],
    missing_required_roll_profiles: &[String],
    missing_required_reference_evidence: &[String],
    missing_required_debug_artifact_kinds: &[String],
    missing_required_film_stock_calibration_pairs: &[String],
    missing_required_scene_exposure_pairs: &[String],
) -> Vec<String> {
    let mut actions = Vec::new();
    push_fixture_coverage_min_action(
        &mut actions,
        "add_validation_ready_fixtures",
        requirements.min_fixtures,
        validation_ready_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_complete_render_dynamic_range_contracts_for_validation_ready_fixtures",
        requirements.min_render_dynamic_range_contract_fixtures,
        render_dynamic_range_contract_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_complete_stitch_normalization_contracts_for_validation_ready_fixtures",
        requirements.min_stitch_normalization_contract_fixtures,
        stitch_normalization_contract_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_complete_geometry_preparation_contracts_for_validation_ready_fixtures",
        requirements.min_geometry_preparation_contract_fixtures,
        geometry_preparation_contract_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_complete_geometry_accuracy_contracts_for_validation_ready_fixtures",
        requirements.min_geometry_accuracy_contract_fixtures,
        geometry_accuracy_contract_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_complete_orientation_accuracy_contracts_for_validation_ready_fixtures",
        requirements.min_orientation_accuracy_contract_fixtures,
        orientation_accuracy_contract_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_complete_negative_reconstruction_contracts_for_validation_ready_fixtures",
        requirements.min_negative_reconstruction_contract_fixtures,
        negative_reconstruction_contract_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_validation_ready_grain_reduction_enabled_fixtures",
        requirements.min_grain_reduction_enabled_fixtures,
        grain_reduction_enabled_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_complete_grain_detail_contracts_for_validation_ready_fixtures",
        requirements.min_grain_detail_contract_fixtures,
        grain_detail_contract_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_complete_grain_reduction_effect_contracts_for_validation_ready_fixtures",
        requirements.min_grain_reduction_effect_contract_fixtures,
        grain_reduction_effect_contract_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_validation_ready_n_component_fixtures",
        requirements.min_n_component_fixtures,
        n_component_validation_ready_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "pin_complete_component_sha256_sets_for_validation_ready_fixtures",
        requirements.min_component_sha256_sets,
        component_sha256_set_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_component_pairs",
        requirements.min_component_pairs,
        component_pair_available_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "pin_component_sha256_pairs_for_validation_ready_fixtures",
        requirements.min_component_sha256_pairs,
        component_sha256_pair_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_readable_tiff_pairs",
        requirements.min_readable_tiff_pairs,
        readable_tiff_pair_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_tiff_layout_consistent_pairs",
        requirements.min_tiff_layout_consistent_pairs,
        tiff_layout_consistent_pair_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_tiff_dimension_matched_pairs",
        requirements.min_tiff_dimension_matched_pairs,
        tiff_dimension_matched_pair_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_complete_summary_baselines",
        requirements.min_summary_baselines,
        summary_baseline_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "pin_summary_baseline_sha256_for_validation_ready_fixtures",
        requirements.min_summary_baseline_sha256_fixtures,
        summary_baseline_sha256_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_validation_ready_calibrated_fixtures",
        requirements.min_calibrated_fixtures,
        calibration_evidence_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "pin_calibration_sha256_for_validation_ready_fixtures",
        requirements.min_calibration_sha256_fixtures,
        calibration_sha256_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_validation_ready_uncalibrated_fixtures",
        requirements.min_uncalibrated_fixtures,
        uncalibrated_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_unique_scanner_profile_coverage",
        requirements.min_unique_scanner_profiles,
        unique_scanner_profile_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_unique_roll_profile_coverage",
        requirements.min_unique_roll_profiles,
        unique_roll_profile_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_unique_film_stock_coverage",
        requirements.min_unique_film_stocks,
        unique_film_stock_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_scene_tag_coverage",
        requirements.min_scene_tags,
        scene_tag_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_exposure_tag_coverage",
        requirements.min_exposure_tags,
        exposure_tag_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_calibration_case_coverage",
        requirements.min_calibration_cases,
        calibration_case_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_film_stock_calibration_pairs",
        requirements.min_film_stock_calibration_pairs,
        film_stock_calibration_pair_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_scene_exposure_pairs",
        requirements.min_scene_exposure_pairs,
        scene_exposure_pair_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_reference_fixtures",
        requirements.min_reference_fixtures,
        reference_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_reference_evidence_types",
        requirements.min_reference_evidence_types,
        reference_evidence_type_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_reference_patch_fixtures",
        requirements.min_reference_patch_fixtures,
        reference_patch_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_reference_patches",
        requirements.min_reference_patch_count,
        reference_patch_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_hash_bound_approved_render_reviews",
        requirements.min_approved_render_review_fixtures,
        approved_render_review_fixture_count,
    );
    push_fixture_coverage_min_action(
        &mut actions,
        "add_debug_artifact_expectation_fixtures",
        requirements.min_debug_artifact_expectation_fixtures,
        debug_artifact_expectation_fixture_count,
    );
    if validation_ready_fixture_count == 0 {
        actions.push("add_validation_ready_fixture".to_string());
    }
    push_fixture_coverage_actions(
        &mut actions,
        "add_validation_ready_fixture_for_film_stock",
        missing_required_film_stocks,
    );
    push_fixture_coverage_actions(
        &mut actions,
        "add_validation_ready_fixture_for_scene_tag",
        missing_required_scene_tags,
    );
    push_fixture_coverage_actions(
        &mut actions,
        "add_validation_ready_fixture_for_exposure_tag",
        missing_required_exposure_tags,
    );
    push_fixture_coverage_actions(
        &mut actions,
        "add_validation_ready_fixture_for_calibration_case",
        missing_required_calibration_cases,
    );
    push_fixture_coverage_actions(
        &mut actions,
        "add_validation_ready_fixture_for_scanner_profile",
        missing_required_scanner_profiles,
    );
    push_fixture_coverage_actions(
        &mut actions,
        "add_validation_ready_fixture_for_roll_profile",
        missing_required_roll_profiles,
    );
    push_fixture_coverage_actions(
        &mut actions,
        "add_validation_ready_fixture_with_reference_evidence",
        missing_required_reference_evidence,
    );
    push_fixture_coverage_actions(
        &mut actions,
        "declare_debug_artifact_expectation_for_validation_ready_fixture",
        missing_required_debug_artifact_kinds,
    );
    push_fixture_coverage_actions(
        &mut actions,
        "add_validation_ready_fixture_for_film_stock_calibration_pair",
        missing_required_film_stock_calibration_pairs,
    );
    push_fixture_coverage_actions(
        &mut actions,
        "add_validation_ready_fixture_for_scene_exposure_pair",
        missing_required_scene_exposure_pairs,
    );
    actions
}

fn push_fixture_coverage_actions(actions: &mut Vec<String>, prefix: &str, values: &[String]) {
    actions.extend(values.iter().map(|value| format!("{prefix}:{value}")));
}

fn push_fixture_coverage_min_action(
    actions: &mut Vec<String>,
    prefix: &str,
    minimum: Option<usize>,
    actual: usize,
) {
    if let Some(minimum) = minimum {
        if actual < minimum {
            actions.push(format!("{prefix}:{}", minimum - actual));
        }
    }
}

fn push_fixture_coverage_entry_actions(
    actions: &mut Vec<String>,
    entries: &[FixtureCoverageEntry],
) {
    for entry in entries {
        actions.extend(entry.action_items.iter().cloned());
    }
}

fn fixture_coverage_entry_actions(entry: &FixtureCoverageEntry) -> Vec<String> {
    let mut actions = Vec::new();
    let name = entry.name.as_str();
    if !entry.component1_exists {
        actions.push(format!("provide_component1_for_fixture:{name}"));
    } else if fixture_coverage_entry_has_issue_prefix(entry, "component1_tiff_") {
        actions.push(format!("repair_component1_tiff_for_fixture:{name}"));
    }
    if fixture_coverage_entry_has_issue(entry, "component1_tiff_bits_below_min") {
        actions.push(format!(
            "replace_component1_with_minimum_bit_depth_tiff_for_fixture:{name}"
        ));
    }
    if entry.component2.is_some() {
        if !entry.component2_exists {
            actions.push(format!("provide_component2_for_fixture:{name}"));
        } else if fixture_coverage_entry_has_issue_prefix(entry, "component2_tiff_") {
            actions.push(format!("repair_component2_tiff_for_fixture:{name}"));
        }
        if fixture_coverage_entry_has_issue(entry, "component2_tiff_bits_below_min") {
            actions.push(format!(
                "replace_component2_with_minimum_bit_depth_tiff_for_fixture:{name}"
            ));
        }
    }
    if fixture_coverage_entry_has_issue(entry, "tiff_pair_layout_mismatch") {
        actions.push(format!(
            "replace_component_pair_with_layout_matched_tiffs_for_fixture:{name}"
        ));
    }
    if fixture_coverage_entry_has_issue(entry, "tiff_pair_dimensions_mismatch") {
        actions.push(format!(
            "replace_component_pair_with_dimension_matched_tiffs_for_fixture:{name}"
        ));
    }
    if fixture_coverage_entry_has_issue_prefix(entry, "component1_sha256_") {
        actions.push(format!("update_component1_sha256_for_fixture:{name}"));
    }
    if fixture_coverage_entry_has_issue_prefix(entry, "component2_sha256_") {
        actions.push(format!("update_component2_sha256_for_fixture:{name}"));
    }
    for component in &entry.additional_components {
        let number = component.component_number;
        let issue_prefix = format!("component{number}_");
        if !component.exists {
            actions.push(format!(
                "provide_additional_component_for_fixture:{name}:{number}"
            ));
        } else if fixture_coverage_entry_has_issue_prefix(entry, &format!("{issue_prefix}tiff_")) {
            actions.push(format!(
                "repair_additional_component_tiff_for_fixture:{name}:{number}"
            ));
        }
        if fixture_coverage_entry_has_issue(entry, &format!("{issue_prefix}tiff_bits_below_min")) {
            actions.push(format!(
                "replace_additional_component_with_minimum_bit_depth_tiff_for_fixture:{name}:{number}"
            ));
        }
        if fixture_coverage_entry_has_issue_prefix(entry, &format!("{issue_prefix}sha256_")) {
            actions.push(format!(
                "update_additional_component_sha256_for_fixture:{name}:{number}"
            ));
        }
        if fixture_coverage_entry_has_issue(
            entry,
            &format!("component{number}_layout_mismatch_component1"),
        ) {
            actions.push(format!(
                "replace_additional_component_with_layout_matched_tiff_for_fixture:{name}:{number}"
            ));
        }
        if fixture_coverage_entry_has_issue(
            entry,
            &format!("component{number}_dimensions_mismatch_component1"),
        ) {
            actions.push(format!(
                "replace_additional_component_with_dimension_matched_tiff_for_fixture:{name}:{number}"
            ));
        }
    }

    match (
        entry.summary_baseline.as_ref(),
        entry.summary_baseline_exists,
        entry.summary_baseline_parse_status.as_deref(),
        entry.summary_baseline_contract_status.as_deref(),
    ) {
        (None, _, _, _) => actions.push(format!("declare_summary_baseline_for_fixture:{name}")),
        (Some(_), Some(false), _, _) => {
            actions.push(format!("provide_summary_baseline_for_fixture:{name}"))
        }
        (Some(_), Some(true), Some("invalid"), _) => {
            actions.push(format!("repair_summary_baseline_for_fixture:{name}"))
        }
        (Some(_), Some(true), _, Some("incomplete")) => {
            actions.push(format!("complete_summary_baseline_for_fixture:{name}"))
        }
        _ => {}
    }
    if fixture_coverage_entry_has_issue_prefix(entry, "summary_baseline_sha256_") {
        actions.push(format!("update_summary_baseline_sha256_for_fixture:{name}"));
    }

    if entry.render_review.is_some() {
        if entry.render_review_exists == Some(false) {
            actions.push(format!("provide_render_review_for_fixture:{name}"));
        } else if !entry.approved_render_review {
            actions.push(format!("complete_render_review_for_fixture:{name}"));
        }
        if fixture_coverage_entry_has_issue_prefix(entry, "render_review_sha256_")
            || fixture_coverage_entry_has_issue(entry, "render_review_sha256_not_declared")
        {
            actions.push(format!("update_render_review_sha256_for_fixture:{name}"));
        }
    }

    match (
        entry.calibration_profile.as_ref(),
        entry.calibration_profile_exists,
        entry.calibration_profile_parse_status.as_deref(),
    ) {
        (Some(_), Some(false), _) => {
            actions.push(format!("provide_calibration_profile_for_fixture:{name}"))
        }
        (Some(_), Some(true), Some("invalid")) => {
            actions.push(format!("repair_calibration_profile_for_fixture:{name}"))
        }
        _ => {}
    }
    if fixture_coverage_entry_has_issue_prefix(entry, "calibration_profile_sha256_") {
        actions.push(format!(
            "update_calibration_profile_sha256_for_fixture:{name}"
        ));
    }

    match (
        entry.calibration_library.as_ref(),
        entry.calibration_library_exists,
        entry.calibration_library_selection_status.as_deref(),
    ) {
        (Some(_), Some(false), _) => {
            actions.push(format!("provide_calibration_library_for_fixture:{name}"))
        }
        (Some(_), Some(true), Some(status)) if status != "applied" => {
            actions.push(format!("repair_calibration_library_for_fixture:{name}"))
        }
        _ => {}
    }
    if fixture_coverage_entry_has_issue_prefix(entry, "calibration_library_sha256_") {
        actions.push(format!(
            "update_calibration_library_sha256_for_fixture:{name}"
        ));
    }

    if entry.film_stock.is_none() {
        actions.push(format!("declare_film_stock_for_fixture:{name}"));
    }
    if entry.scene_tags.is_empty() {
        actions.push(format!("declare_scene_tags_for_fixture:{name}"));
    }
    if entry.exposure_tags.is_empty() {
        actions.push(format!("declare_exposure_tags_for_fixture:{name}"));
    }
    if entry.calibration_case.is_none() {
        actions.push(format!("declare_calibration_case_for_fixture:{name}"));
    }
    if fixture_coverage_entry_has_issue(
        entry,
        "reference_patch_evaluation_required_without_calibration_patches",
    ) {
        actions.push(format!("add_reference_patches_for_fixture:{name}"));
    }
    if entry.grain_reduction_enabled_declared && !entry.grain_detail_contract_complete {
        actions.push(format!("complete_grain_detail_contract_for_fixture:{name}"));
    }
    if entry.grain_reduction_enabled_declared && !entry.grain_reduction_effect_contract_complete {
        actions.push(format!(
            "complete_grain_reduction_effect_contract_for_fixture:{name}"
        ));
    }
    if entry.render_dynamic_range_contract_declared && !entry.render_dynamic_range_contract_complete
    {
        actions.push(format!(
            "complete_render_dynamic_range_contract_for_fixture:{name}"
        ));
    }
    if entry.stitch_normalization_contract_declared && !entry.stitch_normalization_contract_complete
    {
        actions.push(format!(
            "complete_stitch_normalization_contract_for_fixture:{name}"
        ));
    }
    if entry.geometry_preparation_contract_declared && !entry.geometry_preparation_contract_complete
    {
        actions.push(format!(
            "complete_geometry_preparation_contract_for_fixture:{name}"
        ));
    }
    if entry.geometry_accuracy_contract_declared && !entry.geometry_accuracy_contract_complete {
        actions.push(format!(
            "complete_geometry_accuracy_contract_for_fixture:{name}"
        ));
    }
    if entry.orientation_accuracy_contract_declared && !entry.orientation_accuracy_contract_complete
    {
        actions.push(format!(
            "complete_orientation_accuracy_contract_for_fixture:{name}"
        ));
    }
    if entry.negative_reconstruction_contract_declared
        && !entry.negative_reconstruction_contract_complete
    {
        actions.push(format!(
            "complete_negative_reconstruction_contract_for_fixture:{name}"
        ));
    }
    actions
}

fn fixture_coverage_entry_repair_plan(entry: &FixtureCoverageEntry) -> Vec<FixtureRepairPlanItem> {
    entry
        .action_items
        .iter()
        .map(|action| fixture_coverage_repair_plan_item(entry, action))
        .collect()
}

fn fixture_coverage_repair_plan_item(
    entry: &FixtureCoverageEntry,
    action: &str,
) -> FixtureRepairPlanItem {
    let mut paths = Vec::new();
    let additional_component = action
        .rsplit_once(':')
        .and_then(|(_, number)| number.parse::<usize>().ok())
        .and_then(|number| {
            entry
                .additional_components
                .iter()
                .find(|component| component.component_number == number)
        });
    let details = if action.starts_with("provide_component1_for_fixture:") {
        paths.push(entry.component1.clone());
        "Provide the first component TIFF declared by the fixture registry.".to_string()
    } else if action.starts_with("provide_component2_for_fixture:") {
        push_optional_path(&mut paths, entry.component2.as_ref());
        "Provide the second component TIFF declared by the fixture registry.".to_string()
    } else if action.starts_with("repair_component1_tiff_for_fixture:") {
        paths.push(entry.component1.clone());
        "Replace or repair component1 so it is a readable supported TIFF.".to_string()
    } else if action.starts_with("repair_component2_tiff_for_fixture:") {
        push_optional_path(&mut paths, entry.component2.as_ref());
        "Replace or repair component2 so it is a readable supported TIFF.".to_string()
    } else if action.starts_with("replace_component1_with_minimum_bit_depth_tiff_for_fixture:") {
        paths.push(entry.component1.clone());
        "Replace component1 with a TIFF that satisfies the registry minimum bit depth.".to_string()
    } else if action.starts_with("replace_component2_with_minimum_bit_depth_tiff_for_fixture:") {
        push_optional_path(&mut paths, entry.component2.as_ref());
        "Replace component2 with a TIFF that satisfies the registry minimum bit depth.".to_string()
    } else if action.starts_with("replace_component_pair_with_layout_matched_tiffs_for_fixture:") {
        paths.push(entry.component1.clone());
        push_optional_path(&mut paths, entry.component2.as_ref());
        "Replace the component pair with TIFFs whose color type, bit depth, channel count, and alpha layout match.".to_string()
    } else if action.starts_with("replace_component_pair_with_dimension_matched_tiffs_for_fixture:")
    {
        paths.push(entry.component1.clone());
        push_optional_path(&mut paths, entry.component2.as_ref());
        "Replace the component pair with TIFFs whose dimensions match.".to_string()
    } else if action.starts_with("update_component1_sha256_for_fixture:") {
        paths.push(entry.component1.clone());
        "Update component1_sha256 after intentionally replacing or accepting the local component file.".to_string()
    } else if action.starts_with("update_component2_sha256_for_fixture:") {
        push_optional_path(&mut paths, entry.component2.as_ref());
        "Update component2_sha256 after intentionally replacing or accepting the local component file.".to_string()
    } else if action.starts_with("provide_additional_component_for_fixture:") {
        if let Some(component) = additional_component {
            paths.push(component.path.clone());
        }
        "Provide the additional component TIFF declared by the fixture registry.".to_string()
    } else if action.starts_with("repair_additional_component_tiff_for_fixture:") {
        if let Some(component) = additional_component {
            paths.push(component.path.clone());
        }
        "Replace or repair the additional component so it is a readable supported TIFF.".to_string()
    } else if action
        .starts_with("replace_additional_component_with_minimum_bit_depth_tiff_for_fixture:")
    {
        if let Some(component) = additional_component {
            paths.push(component.path.clone());
        }
        "Replace the additional component with a TIFF that satisfies the registry minimum bit depth."
            .to_string()
    } else if action.starts_with("update_additional_component_sha256_for_fixture:") {
        if let Some(component) = additional_component {
            paths.push(component.path.clone());
        }
        "Update the additional component SHA-256 after intentionally replacing or accepting the file."
            .to_string()
    } else if action
        .starts_with("replace_additional_component_with_layout_matched_tiff_for_fixture:")
    {
        paths.push(entry.component1.clone());
        if let Some(component) = additional_component {
            paths.push(component.path.clone());
        }
        "Replace the additional component with a TIFF whose channel layout matches component1."
            .to_string()
    } else if action
        .starts_with("replace_additional_component_with_dimension_matched_tiff_for_fixture:")
    {
        paths.push(entry.component1.clone());
        if let Some(component) = additional_component {
            paths.push(component.path.clone());
        }
        "Replace the additional component with a TIFF whose dimensions are compatible with component1."
            .to_string()
    } else if action.starts_with("declare_summary_baseline_for_fixture:") {
        "Declare a summary_baseline path for this fixture.".to_string()
    } else if action.starts_with("provide_summary_baseline_for_fixture:") {
        push_optional_path(&mut paths, entry.summary_baseline.as_ref());
        "Provide the compact summary baseline declared by the fixture registry.".to_string()
    } else if action.starts_with("repair_summary_baseline_for_fixture:") {
        push_optional_path(&mut paths, entry.summary_baseline.as_ref());
        "Repair the compact summary baseline so it parses as the validation baseline schema."
            .to_string()
    } else if action.starts_with("complete_summary_baseline_for_fixture:") {
        push_optional_path(&mut paths, entry.summary_baseline.as_ref());
        "Regenerate or complete the compact summary baseline so all tracked contract fields are present.".to_string()
    } else if action.starts_with("update_summary_baseline_sha256_for_fixture:") {
        push_optional_path(&mut paths, entry.summary_baseline.as_ref());
        "Update summary_baseline_sha256 after intentionally replacing or accepting the baseline."
            .to_string()
    } else if action.starts_with("provide_render_review_for_fixture:") {
        push_optional_path(&mut paths, entry.render_review.as_ref());
        "Provide the hash-bound render-review manifest declared by the fixture registry."
            .to_string()
    } else if action.starts_with("complete_render_review_for_fixture:") {
        push_optional_path(&mut paths, entry.render_review.as_ref());
        "Inspect the exact report and delivery artifacts, then complete every applicable decision with specific notes, reviewer identity, review time, and overall notes. Technical review failures cannot be overridden by human approval.".to_string()
    } else if action.starts_with("update_render_review_sha256_for_fixture:") {
        push_optional_path(&mut paths, entry.render_review.as_ref());
        "Pin render_review_sha256 only after intentionally accepting the completed human-review manifest; any later edit must invalidate the fixture.".to_string()
    } else if action.starts_with("provide_calibration_profile_for_fixture:") {
        push_optional_path(&mut paths, entry.calibration_profile.as_ref());
        "Provide the external calibration profile declared by the fixture registry.".to_string()
    } else if action.starts_with("repair_calibration_profile_for_fixture:") {
        push_optional_path(&mut paths, entry.calibration_profile.as_ref());
        "Repair the external calibration profile so it loads and validates.".to_string()
    } else if action.starts_with("update_calibration_profile_sha256_for_fixture:") {
        push_optional_path(&mut paths, entry.calibration_profile.as_ref());
        "Update calibration_profile_sha256 after intentionally replacing or accepting the profile."
            .to_string()
    } else if action.starts_with("provide_calibration_library_for_fixture:") {
        push_optional_path(&mut paths, entry.calibration_library.as_ref());
        "Provide the calibration library declared by the fixture registry.".to_string()
    } else if action.starts_with("repair_calibration_library_for_fixture:") {
        push_optional_path(&mut paths, entry.calibration_library.as_ref());
        "Repair the calibration library so the requested scanner, roll, and film-stock evidence can be selected.".to_string()
    } else if action.starts_with("update_calibration_library_sha256_for_fixture:") {
        push_optional_path(&mut paths, entry.calibration_library.as_ref());
        "Update calibration_library_sha256 after intentionally replacing or accepting the library."
            .to_string()
    } else if action.starts_with("declare_film_stock_for_fixture:") {
        "Declare fixture film_stock metadata.".to_string()
    } else if action.starts_with("declare_scene_tags_for_fixture:") {
        "Declare fixture scene_tags metadata.".to_string()
    } else if action.starts_with("declare_exposure_tags_for_fixture:") {
        "Declare fixture exposure_tags metadata.".to_string()
    } else if action.starts_with("declare_calibration_case_for_fixture:") {
        "Declare fixture calibration_case metadata.".to_string()
    } else if action.starts_with("add_reference_patches_for_fixture:") {
        push_optional_path(&mut paths, entry.calibration_profile.as_ref());
        push_optional_path(&mut paths, entry.calibration_library.as_ref());
        "Add target/reference patches to the selected calibration evidence for reference-patch validation.".to_string()
    } else if action.starts_with("complete_grain_detail_contract_for_fixture:") {
        format!(
            "Pin nonzero grain strength, grain scale, enabled/no-review/support expectations, and both-channel >= {GRAIN_DETAIL_MIN_PROBE_COUNT} probe / >= {GRAIN_DETAIL_P10_RETENTION_MIN:.2} p10-retention floors. Missing: {}.",
            entry.grain_detail_contract_missing_fields.join(", ")
        )
    } else if action.starts_with("complete_grain_reduction_effect_contract_for_fixture:") {
        format!(
            "Complete the grain-detail contract and pin positive minimum applied-pixel, exact structure-excluded, and flat-area luminance/chroma p95 residual-reduction ratios. Missing: {}.",
            entry
                .grain_reduction_effect_contract_missing_fields
                .join(", ")
        )
    } else if action.starts_with("complete_render_dynamic_range_contract_for_fixture:") {
        format!(
            "Pin reviewable final delivery, supported/no-review tone-output evidence, positive tone confidence and render-to-mapped range retention, positive post-scale preservation and p05-p95 render-luminance span, plus fixture-approved maximum post-tone high/low channel clipping ratios below 1. Missing: {}.",
            entry
                .render_dynamic_range_contract_missing_fields
                .join(", ")
        )
    } else if action.starts_with("complete_stitch_normalization_contract_for_fixture:") {
        format!(
            "Pin an accepted stitch, a known exposure-normalization model with model-appropriate held-out validation and <= {STITCH_NORMALIZATION_MAX_OFFSET_RATIO:.2} normalized offset, seam-aware multiband blending with no review, >= {STITCH_NORMALIZATION_MIN_DETAIL_SCALE_COUNT} supported detail scales, <= {STITCH_NORMALIZATION_MAX_DETAIL_ENERGY_RATIO:.2} detail-energy imbalance, <= {STITCH_NORMALIZATION_MAX_GRADIENT_RATIO:.2} seam-gradient ratio, and < 1 overlap p95 difference. Missing: {}.",
            entry
                .stitch_normalization_contract_missing_fields
                .join(", ")
        )
    } else if action.starts_with("complete_geometry_preparation_contract_for_fixture:") {
        format!(
            "Pin all-component applied/no-review deskew with >= {GEOMETRY_PREPARATION_MIN_DESKEW_RETAINED_AREA_RATIO:.2} retained area, plus four-edge all-component scanner-border cropping, >= {GEOMETRY_PREPARATION_MIN_BORDER_RETAINED_AREA_RATIO:.2} retained border-crop area, a maximum below 1, and no rejected crop. Missing: {}.",
            entry
                .geometry_preparation_contract_missing_fields
                .join(", ")
        )
    } else if action.starts_with("complete_geometry_accuracy_contract_for_fixture:") {
        format!(
            "Pin the approved deskew correction within <= {GEOMETRY_ACCURACY_MAX_DESKEW_TOLERANCE_DEGREES:.2} degrees and every component's expected top/bottom/left/right crop within <= {GEOMETRY_ACCURACY_MAX_CROP_TOLERANCE_PX} pixels, in addition to the complete preparation contract. Missing: {}.",
            entry.geometry_accuracy_contract_missing_fields.join(", ")
        )
    } else if action.starts_with("complete_orientation_accuracy_contract_for_fixture:") {
        format!(
            "Pin every component's EXIF-orientation tag presence/value, named decoded transform, applied flag, original scanner dimensions, and oriented output dimensions. Missing or inconsistent: {}.",
            entry.orientation_accuracy_contract_missing_fields.join(", ")
        )
    } else if action.starts_with("complete_negative_reconstruction_contract_for_fixture:") {
        format!(
            "Pin a measured film-base source, >= {NEGATIVE_RECONSTRUCTION_MIN_BASE_CONFIDENCE:.2} base confidence, measured nonlinear 3x3 dye separation/PCHIP response with held-out DeltaE00 and <= {NEGATIVE_RECONSTRUCTION_MAX_DENSITY_NOISE_GAIN:.1} noise-gain gates, <= {NEGATIVE_RECONSTRUCTION_MAX_EXTRAPOLATION_RATIO:.2} curve extrapolation, signed headroom, direct-density rendering, accepted calibration, safe/trusted color, and linear ProPhoto output. Missing: {}.",
            entry
                .negative_reconstruction_contract_missing_fields
                .join(", ")
        )
    } else {
        "Resolve the fixture coverage action reported by the validation harness.".to_string()
    };

    FixtureRepairPlanItem {
        action: action.to_string(),
        paths,
        details,
    }
}

fn push_optional_path(paths: &mut Vec<String>, path: Option<&String>) {
    if let Some(path) = path {
        paths.push(path.clone());
    }
}

fn fixture_coverage_entry_has_issue_prefix(
    entry: &FixtureCoverageEntry,
    suffix_prefix: &str,
) -> bool {
    let issue_prefix = format!("{}:{suffix_prefix}", entry.name);
    entry
        .issues
        .iter()
        .any(|issue| issue.starts_with(&issue_prefix))
}

fn fixture_coverage_entry_has_issue(entry: &FixtureCoverageEntry, suffix: &str) -> bool {
    let issue = format!("{}:{suffix}", entry.name);
    entry.issues.iter().any(|entry_issue| entry_issue == &issue)
}

#[allow(clippy::too_many_arguments)]
fn apply_coverage_requirements(
    requirements: &FixtureCoverageRequirements,
    fixture_count: usize,
    render_dynamic_range_contract_fixture_count: usize,
    stitch_normalization_contract_fixture_count: usize,
    geometry_preparation_contract_fixture_count: usize,
    geometry_accuracy_contract_fixture_count: usize,
    orientation_accuracy_contract_fixture_count: usize,
    negative_reconstruction_contract_fixture_count: usize,
    grain_reduction_enabled_fixture_count: usize,
    grain_detail_contract_fixture_count: usize,
    grain_reduction_effect_contract_fixture_count: usize,
    n_component_fixture_count: usize,
    component_sha256_set_count: usize,
    component_pair_available_count: usize,
    component_sha256_pair_count: usize,
    readable_tiff_pair_count: usize,
    tiff_layout_consistent_pair_count: usize,
    tiff_dimension_matched_pair_count: usize,
    summary_baseline_count: usize,
    summary_baseline_sha256_count: usize,
    calibration_evidence_count: usize,
    calibration_sha256_count: usize,
    uncalibrated_fixture_count: usize,
    unique_scanner_profiles: &[String],
    unique_roll_profiles: &[String],
    unique_film_stocks: &[String],
    scene_tags: &[String],
    exposure_tags: &[String],
    calibration_cases: &[String],
    reference_fixture_count: usize,
    reference_evidence: &[String],
    reference_patch_fixture_count: usize,
    reference_patch_count: usize,
    approved_render_review_fixture_count: usize,
    debug_artifact_expectation_fixture_count: usize,
    debug_artifact_kinds_required: &[String],
    film_stock_calibration_pairs: &[String],
    scene_exposure_pairs: &[String],
    issues: &mut Vec<String>,
) {
    push_min_requirement_issue(
        requirements.min_fixtures,
        fixture_count,
        "fixture_coverage_min_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_render_dynamic_range_contract_fixtures,
        render_dynamic_range_contract_fixture_count,
        "fixture_coverage_min_render_dynamic_range_contract_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_stitch_normalization_contract_fixtures,
        stitch_normalization_contract_fixture_count,
        "fixture_coverage_min_stitch_normalization_contract_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_geometry_preparation_contract_fixtures,
        geometry_preparation_contract_fixture_count,
        "fixture_coverage_min_geometry_preparation_contract_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_geometry_accuracy_contract_fixtures,
        geometry_accuracy_contract_fixture_count,
        "fixture_coverage_min_geometry_accuracy_contract_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_orientation_accuracy_contract_fixtures,
        orientation_accuracy_contract_fixture_count,
        "fixture_coverage_min_orientation_accuracy_contract_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_negative_reconstruction_contract_fixtures,
        negative_reconstruction_contract_fixture_count,
        "fixture_coverage_min_negative_reconstruction_contract_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_grain_reduction_enabled_fixtures,
        grain_reduction_enabled_fixture_count,
        "fixture_coverage_min_grain_reduction_enabled_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_grain_detail_contract_fixtures,
        grain_detail_contract_fixture_count,
        "fixture_coverage_min_grain_detail_contract_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_grain_reduction_effect_contract_fixtures,
        grain_reduction_effect_contract_fixture_count,
        "fixture_coverage_min_grain_reduction_effect_contract_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_n_component_fixtures,
        n_component_fixture_count,
        "fixture_coverage_min_n_component_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_component_sha256_sets,
        component_sha256_set_count,
        "fixture_coverage_min_component_sha256_sets_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_component_pairs,
        component_pair_available_count,
        "fixture_coverage_min_component_pairs_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_component_sha256_pairs,
        component_sha256_pair_count,
        "fixture_coverage_min_component_sha256_pairs_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_readable_tiff_pairs,
        readable_tiff_pair_count,
        "fixture_coverage_min_readable_tiff_pairs_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_tiff_layout_consistent_pairs,
        tiff_layout_consistent_pair_count,
        "fixture_coverage_min_tiff_layout_consistent_pairs_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_tiff_dimension_matched_pairs,
        tiff_dimension_matched_pair_count,
        "fixture_coverage_min_tiff_dimension_matched_pairs_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_summary_baselines,
        summary_baseline_count,
        "fixture_coverage_min_summary_baselines_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_summary_baseline_sha256_fixtures,
        summary_baseline_sha256_count,
        "fixture_coverage_min_summary_baseline_sha256_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_calibrated_fixtures,
        calibration_evidence_count,
        "fixture_coverage_min_calibrated_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_calibration_sha256_fixtures,
        calibration_sha256_count,
        "fixture_coverage_min_calibration_sha256_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_uncalibrated_fixtures,
        uncalibrated_fixture_count,
        "fixture_coverage_min_uncalibrated_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_unique_scanner_profiles,
        unique_scanner_profiles.len(),
        "fixture_coverage_min_unique_scanner_profiles_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_unique_roll_profiles,
        unique_roll_profiles.len(),
        "fixture_coverage_min_unique_roll_profiles_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_unique_film_stocks,
        unique_film_stocks.len(),
        "fixture_coverage_min_unique_film_stocks_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_scene_tags,
        scene_tags.len(),
        "fixture_coverage_min_scene_tags_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_exposure_tags,
        exposure_tags.len(),
        "fixture_coverage_min_exposure_tags_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_calibration_cases,
        calibration_cases.len(),
        "fixture_coverage_min_calibration_cases_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_film_stock_calibration_pairs,
        film_stock_calibration_pairs.len(),
        "fixture_coverage_min_film_stock_calibration_pairs_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_scene_exposure_pairs,
        scene_exposure_pairs.len(),
        "fixture_coverage_min_scene_exposure_pairs_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_reference_fixtures,
        reference_fixture_count,
        "fixture_coverage_min_reference_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_reference_evidence_types,
        reference_evidence.len(),
        "fixture_coverage_min_reference_evidence_types_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_reference_patch_fixtures,
        reference_patch_fixture_count,
        "fixture_coverage_min_reference_patch_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_reference_patch_count,
        reference_patch_count,
        "fixture_coverage_min_reference_patch_count_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_approved_render_review_fixtures,
        approved_render_review_fixture_count,
        "fixture_coverage_min_approved_render_review_fixtures_not_met",
        issues,
    );
    push_min_requirement_issue(
        requirements.min_debug_artifact_expectation_fixtures,
        debug_artifact_expectation_fixture_count,
        "fixture_coverage_min_debug_artifact_expectation_fixtures_not_met",
        issues,
    );
    push_missing_required_values(
        &requirements.required_film_stocks,
        unique_film_stocks,
        "fixture_coverage_required_film_stock_missing",
        issues,
    );
    push_missing_required_values(
        &requirements.required_scene_tags,
        scene_tags,
        "fixture_coverage_required_scene_tag_missing",
        issues,
    );
    push_missing_required_values(
        &requirements.required_exposure_tags,
        exposure_tags,
        "fixture_coverage_required_exposure_tag_missing",
        issues,
    );
    push_missing_required_values(
        &requirements.required_calibration_cases,
        calibration_cases,
        "fixture_coverage_required_calibration_case_missing",
        issues,
    );
    push_missing_required_values(
        &requirements.required_scanner_profiles,
        unique_scanner_profiles,
        "fixture_coverage_required_scanner_profile_missing",
        issues,
    );
    push_missing_required_values(
        &requirements.required_roll_profiles,
        unique_roll_profiles,
        "fixture_coverage_required_roll_profile_missing",
        issues,
    );
    push_missing_required_values(
        &requirements.required_reference_evidence,
        reference_evidence,
        "fixture_coverage_required_reference_evidence_missing",
        issues,
    );
    push_missing_required_values(
        &requirements.required_film_stock_calibration_pairs,
        film_stock_calibration_pairs,
        "fixture_coverage_required_film_stock_calibration_pair_missing",
        issues,
    );
    push_missing_required_values(
        &requirements.required_scene_exposure_pairs,
        scene_exposure_pairs,
        "fixture_coverage_required_scene_exposure_pair_missing",
        issues,
    );
    push_missing_required_values(
        &requirements.required_debug_artifact_kinds,
        debug_artifact_kinds_required,
        "fixture_coverage_required_debug_artifact_kind_missing",
        issues,
    );
}

fn fixture_film_stock_calibration_pairs(entries: &[FixtureCoverageEntry]) -> Vec<String> {
    entries
        .iter()
        .filter(|entry| entry.validation_ready)
        .filter_map(|entry| {
            Some(coverage_pair_label(
                non_empty_metadata(entry.film_stock.as_deref())?,
                non_empty_metadata(entry.calibration_case.as_deref())?,
            ))
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn fixture_calibration_sha256_probe(entry: &FixtureCoverageEntry) -> Option<&FixtureSha256Probe> {
    entry
        .calibration_profile_sha256
        .as_ref()
        .or(entry.calibration_library_sha256.as_ref())
}

fn fixture_scene_exposure_pairs(entries: &[FixtureCoverageEntry]) -> Vec<String> {
    entries
        .iter()
        .filter(|entry| entry.validation_ready)
        .flat_map(|entry| {
            entry
                .scene_tags
                .iter()
                .filter_map(|scene| non_empty_metadata(Some(scene.as_str())))
                .flat_map(move |scene| {
                    entry
                        .exposure_tags
                        .iter()
                        .filter_map(|exposure| non_empty_metadata(Some(exposure.as_str())))
                        .map(move |exposure| coverage_pair_label(scene, exposure))
                })
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn non_empty_metadata(value: Option<&str>) -> Option<&str> {
    value.and_then(|value| {
        let value = value.trim();
        (!value.is_empty()).then_some(value)
    })
}

fn coverage_pair_label(left: &str, right: &str) -> String {
    format!("{left}|{right}")
}

fn push_min_requirement_issue(
    required: Option<usize>,
    actual: usize,
    issue: &str,
    issues: &mut Vec<String>,
) {
    if required.is_some_and(|minimum| actual < minimum) {
        issues.push(issue.to_string());
    }
}

fn push_missing_required_values(
    required: &[String],
    actual: &[String],
    issue_prefix: &str,
    issues: &mut Vec<String>,
) {
    for value in missing_required_values(required, actual) {
        issues.push(format!("{issue_prefix}:{value}"));
    }
}

fn missing_required_values(required: &[String], actual: &[String]) -> Vec<String> {
    let actual = actual.iter().collect::<BTreeSet<_>>();
    required
        .iter()
        .filter(|value| !actual.contains(*value))
        .cloned()
        .collect()
}

fn probe_fixture_component_sha256(
    path: &Path,
    expected_sha256: Option<&str>,
    compute_undeclared_hashes: bool,
) -> Option<FixtureSha256Probe> {
    let expected_sha256 = expected_sha256.map(str::to_ascii_lowercase);
    if expected_sha256.is_none() && !compute_undeclared_hashes {
        return None;
    }
    if expected_sha256
        .as_deref()
        .is_some_and(|expected| !is_valid_sha256_hex(expected))
    {
        return Some(FixtureSha256Probe {
            status: "invalid_expected".to_string(),
            expected_sha256,
            actual_sha256: None,
            file_size_bytes: None,
            file_count: None,
            matched: false,
            error: Some("expected SHA-256 is not 64 hexadecimal characters".to_string()),
        });
    }
    if !path.exists() {
        return Some(FixtureSha256Probe {
            status: "missing".to_string(),
            expected_sha256,
            actual_sha256: None,
            file_size_bytes: None,
            file_count: None,
            matched: false,
            error: None,
        });
    }

    match hash_file_sha256(path) {
        Ok((actual_sha256, file_size_bytes)) => {
            let matched = expected_sha256
                .as_deref()
                .is_some_and(|expected| actual_sha256 == expected);
            Some(FixtureSha256Probe {
                status: match (expected_sha256.is_some(), matched) {
                    (true, true) => "matched",
                    (true, false) => "mismatch",
                    (false, _) => "computed",
                }
                .to_string(),
                expected_sha256,
                actual_sha256: Some(actual_sha256),
                file_size_bytes: Some(file_size_bytes),
                file_count: Some(1),
                matched,
                error: None,
            })
        }
        Err(err) => Some(FixtureSha256Probe {
            status: "unreadable".to_string(),
            expected_sha256,
            actual_sha256: None,
            file_size_bytes: None,
            file_count: None,
            matched: false,
            error: Some(err.to_string()),
        }),
    }
}

fn hash_file_sha256(path: &Path) -> Result<(String, u64), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let file_size_bytes = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok((format!("{:x}", hasher.finalize()), file_size_bytes))
}

fn probe_fixture_calibration_library_sha256(
    path: &Path,
    expected_sha256: Option<&str>,
    compute_undeclared_hashes: bool,
) -> Option<FixtureSha256Probe> {
    let expected_sha256 = expected_sha256.map(str::to_ascii_lowercase);
    if expected_sha256.is_none() && !compute_undeclared_hashes {
        return None;
    }
    if expected_sha256
        .as_deref()
        .is_some_and(|expected| !is_valid_sha256_hex(expected))
    {
        return Some(FixtureSha256Probe {
            status: "invalid_expected".to_string(),
            expected_sha256,
            actual_sha256: None,
            file_size_bytes: None,
            file_count: None,
            matched: false,
            error: Some("expected SHA-256 is not 64 hexadecimal characters".to_string()),
        });
    }
    if !path.exists() {
        return Some(FixtureSha256Probe {
            status: "missing".to_string(),
            expected_sha256,
            actual_sha256: None,
            file_size_bytes: None,
            file_count: None,
            matched: false,
            error: None,
        });
    }

    match hash_calibration_library_sha256(path) {
        Ok((actual_sha256, file_size_bytes, file_count)) => {
            let matched = expected_sha256
                .as_deref()
                .is_some_and(|expected| actual_sha256 == expected);
            Some(FixtureSha256Probe {
                status: match (expected_sha256.is_some(), matched) {
                    (true, true) => "matched",
                    (true, false) => "mismatch",
                    (false, _) => "computed",
                }
                .to_string(),
                expected_sha256,
                actual_sha256: Some(actual_sha256),
                file_size_bytes: Some(file_size_bytes),
                file_count: Some(file_count),
                matched,
                error: None,
            })
        }
        Err(err) => Some(FixtureSha256Probe {
            status: "unreadable".to_string(),
            expected_sha256,
            actual_sha256: None,
            file_size_bytes: None,
            file_count: None,
            matched: false,
            error: Some(err.to_string()),
        }),
    }
}

fn hash_calibration_library_sha256(
    path: &Path,
) -> Result<(String, u64, usize), Box<dyn std::error::Error>> {
    if !path.is_dir() {
        return Err("calibration library is not a directory".into());
    }

    let mut stack = vec![path.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.is_dir() {
                stack.push(entry_path);
            } else if entry_path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
            {
                files.push(entry_path);
            }
        }
    }
    files.sort_by_key(|file| relative_path_label(path, file));

    let mut hasher = Sha256::new();
    let mut file_size_bytes = 0_u64;
    let file_count = files.len();
    for file in files {
        let relative_path = relative_path_label(path, &file);
        let (file_hash, size) = hash_file_sha256(&file)?;
        hasher.update(relative_path.as_bytes());
        hasher.update(b"\0");
        hasher.update(file_hash.as_bytes());
        hasher.update(b"\0");
        file_size_bytes = file_size_bytes.saturating_add(size);
    }

    Ok((
        format!("{:x}", hasher.finalize()),
        file_size_bytes,
        file_count,
    ))
}

fn relative_path_label(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|component| component.as_os_str().to_string_lossy().to_string())
        .collect::<Vec<_>>()
        .join("/")
}

fn validate_fixture_component_sha256_probe(
    fixture_name: &str,
    component: &str,
    probe: Option<&FixtureSha256Probe>,
    issues: &mut Vec<String>,
) {
    let Some(probe) = probe else {
        return;
    };
    match probe.status.as_str() {
        "matched" | "computed" | "missing" => {}
        "mismatch" => issues.push(format!("{fixture_name}:{component}_sha256_mismatch")),
        "invalid_expected" => issues.push(format!("{fixture_name}:{component}_sha256_invalid")),
        "unreadable" => issues.push(format!("{fixture_name}:{component}_sha256_read_failed")),
        _ => issues.push(format!("{fixture_name}:{component}_sha256_probe_failed")),
    }
}

fn probe_fixture_tiff(path: &Path) -> FixtureTiffProbe {
    match probe_fixture_tiff_inner(path) {
        Ok(probe) => probe,
        Err(err) => FixtureTiffProbe {
            status: "unreadable".to_string(),
            readable: false,
            width: None,
            height: None,
            color_type: None,
            source_bits_per_sample: None,
            source_channel_count: None,
            source_has_alpha: None,
            error: Some(err.to_string()),
        },
    }
}

fn probe_fixture_tiff_inner(path: &Path) -> Result<FixtureTiffProbe, Box<dyn std::error::Error>> {
    let inspection = scanstitch::tiff_io::inspect_scan_source(path)?;
    Ok(FixtureTiffProbe {
        status: "readable".to_string(),
        readable: true,
        width: Some(inspection.width),
        height: Some(inspection.height),
        color_type: Some(inspection.color_type),
        source_bits_per_sample: Some(inspection.source_bits_per_sample),
        source_channel_count: Some(inspection.source_channel_count),
        source_has_alpha: Some(inspection.source_has_alpha),
        error: None,
    })
}

fn validate_fixture_tiff_probe(
    fixture_name: &str,
    component: &str,
    probe: Option<&FixtureTiffProbe>,
    min_bits_per_sample: Option<u8>,
    issues: &mut Vec<String>,
) {
    let Some(probe) = probe else {
        return;
    };
    if !probe.readable {
        issues.push(format!("{fixture_name}:{component}_tiff_unreadable"));
        return;
    }
    if let Some(min_bits) = min_bits_per_sample {
        if probe
            .source_bits_per_sample
            .is_some_and(|actual| actual < min_bits)
        {
            issues.push(format!("{fixture_name}:{component}_tiff_bits_below_min"));
        }
    }
}

fn validate_fixture_tiff_pair_probe(
    fixture_name: &str,
    pair: Option<&FixtureTiffPairProbe>,
    issues: &mut Vec<String>,
) {
    let Some(pair) = pair else {
        return;
    };
    if !pair.dimension_matched {
        issues.push(format!("{fixture_name}:tiff_pair_dimensions_mismatch"));
    }

    let mut layout_mismatch = false;
    if pair.color_type_match == Some(false) {
        layout_mismatch = true;
        issues.push(format!("{fixture_name}:tiff_pair_color_type_mismatch"));
    }
    if pair.bits_per_sample_match == Some(false) {
        layout_mismatch = true;
        issues.push(format!("{fixture_name}:tiff_pair_bits_per_sample_mismatch"));
    }
    if pair.channel_count_match == Some(false) {
        layout_mismatch = true;
        issues.push(format!("{fixture_name}:tiff_pair_channel_count_mismatch"));
    }
    if pair.alpha_flag_match == Some(false) {
        layout_mismatch = true;
        issues.push(format!("{fixture_name}:tiff_pair_alpha_flag_mismatch"));
    }
    if layout_mismatch {
        issues.push(format!("{fixture_name}:tiff_pair_layout_mismatch"));
    }
}

fn fixture_tiff_pair_probe(
    component1: Option<&FixtureTiffProbe>,
    component2: Option<&FixtureTiffProbe>,
) -> Option<FixtureTiffPairProbe> {
    let (Some(component1), Some(component2)) = (component1, component2) else {
        return None;
    };
    if !component1.readable || !component2.readable {
        return Some(FixtureTiffPairProbe {
            dimensions_match: None,
            width_delta: None,
            height_delta: None,
            dimensions_compatible: false,
            color_type_match: None,
            bits_per_sample_match: None,
            channel_count_match: None,
            alpha_flag_match: None,
            layout_consistent: false,
            dimension_matched: false,
        });
    }
    let dimensions_match =
        component1.width == component2.width && component1.height == component2.height;
    let width_delta = option_abs_diff(component1.width, component2.width);
    let height_delta = option_abs_diff(component1.height, component2.height);
    let dimensions_compatible = dimensions_match
        || (width_delta == Some(0) && height_delta.is_some_and(|delta| delta <= 1));
    let color_type_match = component1.color_type == component2.color_type;
    let bits_per_sample_match =
        component1.source_bits_per_sample == component2.source_bits_per_sample;
    let channel_count_match = component1.source_channel_count == component2.source_channel_count;
    let alpha_flag_match = component1.source_has_alpha == component2.source_has_alpha;
    let layout_consistent =
        color_type_match && bits_per_sample_match && channel_count_match && alpha_flag_match;

    Some(FixtureTiffPairProbe {
        dimensions_match: Some(dimensions_match),
        width_delta,
        height_delta,
        dimensions_compatible,
        color_type_match: Some(color_type_match),
        bits_per_sample_match: Some(bits_per_sample_match),
        channel_count_match: Some(channel_count_match),
        alpha_flag_match: Some(alpha_flag_match),
        layout_consistent,
        dimension_matched: dimensions_compatible,
    })
}

fn option_abs_diff(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    Some(left?.abs_diff(right?))
}

fn optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "not set".to_string())
}

fn optional_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "not set".to_string())
}

fn optional_bool(value: Option<bool>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "not set".to_string())
}

fn optional_f64_vec(value: Option<&[f64]>) -> String {
    value
        .map(|values| {
            values
                .iter()
                .map(|value| format!("{value:.3}"))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_else(|| "not set".to_string())
}

fn optional_delta_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:+.6}"))
        .unwrap_or_default()
}

fn optional_delta_isize(value: Option<isize>) -> String {
    value.map(|value| format!("{value:+}")).unwrap_or_default()
}

fn fixture_coverage_entry(
    name: &str,
    fixture: &FixtureEntry,
    requirements: &FixtureCoverageRequirements,
    compute_fixture_hashes: bool,
) -> FixtureCoverageEntry {
    let mut issues = Vec::new();
    let component1_exists = fixture.component1.exists();
    let component2_exists = fixture
        .component2
        .as_ref()
        .is_some_and(|component2| component2.exists());
    if !component1_exists {
        issues.push(format!("{name}:component1_missing"));
    }
    if fixture.component2.is_some() && !component2_exists {
        issues.push(format!("{name}:component2_missing"));
    }
    let component1_sha256 = probe_fixture_component_sha256(
        &fixture.component1,
        fixture.component1_sha256.as_deref(),
        compute_fixture_hashes,
    );
    let component2_sha256 = fixture.component2.as_ref().and_then(|component2| {
        probe_fixture_component_sha256(
            component2,
            fixture.component2_sha256.as_deref(),
            compute_fixture_hashes,
        )
    });
    validate_fixture_component_sha256_probe(
        name,
        "component1",
        component1_sha256.as_ref(),
        &mut issues,
    );
    validate_fixture_component_sha256_probe(
        name,
        "component2",
        component2_sha256.as_ref(),
        &mut issues,
    );
    let component1_tiff = component1_exists.then(|| probe_fixture_tiff(&fixture.component1));
    let component2_tiff = fixture
        .component2
        .as_ref()
        .and_then(|component2| component2_exists.then(|| probe_fixture_tiff(component2)));
    let tiff_pair = fixture_tiff_pair_probe(component1_tiff.as_ref(), component2_tiff.as_ref());
    validate_fixture_tiff_probe(
        name,
        "component1",
        component1_tiff.as_ref(),
        requirements.min_tiff_bits_per_sample,
        &mut issues,
    );
    validate_fixture_tiff_probe(
        name,
        "component2",
        component2_tiff.as_ref(),
        requirements.min_tiff_bits_per_sample,
        &mut issues,
    );
    validate_fixture_tiff_pair_probe(name, tiff_pair.as_ref(), &mut issues);
    let mut additional_components = Vec::with_capacity(fixture.additional_components.len());
    for (offset, component) in fixture.additional_components.iter().enumerate() {
        let component_number = offset + 3;
        let component_label = format!("component{component_number}");
        let exists = component.path.exists();
        if !exists {
            issues.push(format!("{name}:{component_label}_missing"));
        }
        let sha256 = probe_fixture_component_sha256(
            &component.path,
            component.sha256.as_deref(),
            compute_fixture_hashes,
        );
        validate_fixture_component_sha256_probe(
            name,
            &component_label,
            sha256.as_ref(),
            &mut issues,
        );
        let tiff = exists.then(|| probe_fixture_tiff(&component.path));
        validate_fixture_tiff_probe(
            name,
            &component_label,
            tiff.as_ref(),
            requirements.min_tiff_bits_per_sample,
            &mut issues,
        );
        let comparison_to_component1 =
            fixture_tiff_pair_probe(component1_tiff.as_ref(), tiff.as_ref());
        if let Some(comparison) = &comparison_to_component1 {
            if !comparison.dimension_matched {
                issues.push(format!(
                    "{name}:{component_label}_dimensions_mismatch_component1"
                ));
            }
            if !comparison.layout_consistent {
                issues.push(format!(
                    "{name}:{component_label}_layout_mismatch_component1"
                ));
            }
        }
        additional_components.push(AdditionalFixtureComponentCoverage {
            component_number,
            path: component.path.display().to_string(),
            exists,
            sha256,
            tiff,
            comparison_to_component1,
        });
    }
    let component2_exists_or_absent = fixture.component2.is_none() || component2_exists;
    let all_components_exist = component1_exists
        && component2_exists_or_absent
        && additional_components
            .iter()
            .all(|component| component.exists);
    let all_component_sha256_declared = component1_sha256
        .as_ref()
        .is_some_and(|probe| probe.expected_sha256.is_some())
        && (fixture.component2.is_none()
            || component2_sha256
                .as_ref()
                .is_some_and(|probe| probe.expected_sha256.is_some()))
        && additional_components.iter().all(|component| {
            component
                .sha256
                .as_ref()
                .is_some_and(|probe| probe.expected_sha256.is_some())
        });
    let all_component_sha256_computed = component1_sha256
        .as_ref()
        .is_some_and(|probe| probe.actual_sha256.is_some())
        && (fixture.component2.is_none()
            || component2_sha256
                .as_ref()
                .is_some_and(|probe| probe.actual_sha256.is_some()))
        && additional_components.iter().all(|component| {
            component
                .sha256
                .as_ref()
                .is_some_and(|probe| probe.actual_sha256.is_some())
        });
    let all_component_sha256_matched = component1_sha256
        .as_ref()
        .is_some_and(|probe| probe.matched)
        && (fixture.component2.is_none()
            || component2_sha256
                .as_ref()
                .is_some_and(|probe| probe.matched))
        && additional_components
            .iter()
            .all(|component| component.sha256.as_ref().is_some_and(|probe| probe.matched));
    let all_components_readable_tiff = component1_tiff.as_ref().is_some_and(|probe| probe.readable)
        && (fixture.component2.is_none()
            || component2_tiff.as_ref().is_some_and(|probe| probe.readable))
        && additional_components
            .iter()
            .all(|component| component.tiff.as_ref().is_some_and(|probe| probe.readable));
    let all_component_layouts_consistent = all_components_readable_tiff
        && (fixture.component2.is_none()
            || tiff_pair
                .as_ref()
                .is_some_and(|comparison| comparison.layout_consistent))
        && additional_components.iter().all(|component| {
            component
                .comparison_to_component1
                .as_ref()
                .is_some_and(|comparison| comparison.layout_consistent)
        });
    let all_component_dimensions_matched = all_components_readable_tiff
        && (fixture.component2.is_none()
            || tiff_pair
                .as_ref()
                .is_some_and(|comparison| comparison.dimension_matched))
        && additional_components.iter().all(|component| {
            component
                .comparison_to_component1
                .as_ref()
                .is_some_and(|comparison| comparison.dimension_matched)
        });
    if fixture.scene_tags.is_empty() {
        issues.push(format!("{name}:scene_tags_not_declared"));
    }
    if fixture.exposure_tags.is_empty() {
        issues.push(format!("{name}:exposure_tags_not_declared"));
    }
    if fixture.calibration_case.is_none() {
        issues.push(format!("{name}:calibration_case_not_declared"));
    }

    let (
        summary_baseline_exists,
        summary_baseline_parse_status,
        summary_baseline_contract_status,
        summary_baseline_contract_missing_fields,
    ) = match &fixture.summary_baseline {
        Some(path) => {
            if path.exists() {
                match std::fs::read_to_string(path).ok().and_then(|contents| {
                    serde_json::from_str::<TrackedValidationBaseline>(&contents).ok()
                }) {
                    Some(baseline) => {
                        let missing_fields = summary_baseline_contract_issues(&baseline)
                            .into_iter()
                            .map(str::to_string)
                            .collect::<Vec<_>>();
                        if missing_fields.is_empty() {
                            (
                                Some(true),
                                Some("valid".to_string()),
                                Some("complete".to_string()),
                                missing_fields,
                            )
                        } else {
                            issues.extend(missing_fields.iter().map(|field| {
                                format!("{name}:summary_baseline_incomplete:{field}")
                            }));
                            (
                                Some(true),
                                Some("valid".to_string()),
                                Some("incomplete".to_string()),
                                missing_fields,
                            )
                        }
                    }
                    None => {
                        issues.push(format!("{name}:summary_baseline_invalid"));
                        (Some(true), Some("invalid".to_string()), None, Vec::new())
                    }
                }
            } else {
                issues.push(format!("{name}:summary_baseline_missing"));
                (Some(false), Some("missing".to_string()), None, Vec::new())
            }
        }
        None => {
            issues.push(format!("{name}:summary_baseline_not_declared"));
            (None, None, None, Vec::new())
        }
    };
    let summary_baseline_valid = summary_baseline_parse_status.as_deref() == Some("valid")
        && summary_baseline_contract_status.as_deref() == Some("complete");

    let summary_baseline_sha256 = fixture.summary_baseline.as_ref().and_then(|path| {
        probe_fixture_component_sha256(
            path,
            fixture.summary_baseline_sha256.as_deref(),
            compute_fixture_hashes,
        )
    });
    validate_fixture_component_sha256_probe(
        name,
        "summary_baseline",
        summary_baseline_sha256.as_ref(),
        &mut issues,
    );

    let render_review_exists = fixture.render_review.as_ref().map(|path| path.exists());
    if render_review_exists == Some(false) {
        issues.push(format!("{name}:render_review_missing"));
    }
    let render_review_sha256 = fixture.render_review.as_ref().and_then(|path| {
        probe_fixture_component_sha256(
            path,
            fixture.render_review_sha256.as_deref(),
            compute_fixture_hashes,
        )
    });
    validate_fixture_component_sha256_probe(
        name,
        "render_review",
        render_review_sha256.as_ref(),
        &mut issues,
    );
    if fixture.render_review.is_some() && fixture.render_review_sha256.is_none() {
        issues.push(format!("{name}:render_review_sha256_not_declared"));
    }
    let render_review_inspection = fixture
        .render_review
        .as_ref()
        .filter(|path| path.exists())
        .map(|path| {
            render_review::inspect_render_review_manifest_for_inputs(
                path,
                name,
                &fixture.pipeline_inputs(),
            )
        });
    if let Some(inspection) = &render_review_inspection {
        issues.extend(
            inspection
                .issues
                .iter()
                .map(|issue| format!("{name}:render_review:{issue}")),
        );
    }
    let approved_render_review = render_review_inspection
        .as_ref()
        .is_some_and(|inspection| inspection.approved)
        && render_review_sha256
            .as_ref()
            .is_some_and(|probe| probe.matched);

    let calibration_profile_sha256 = fixture.calibration_profile.as_ref().and_then(|path| {
        probe_fixture_component_sha256(
            path,
            fixture.calibration_profile_sha256.as_deref(),
            compute_fixture_hashes,
        )
    });
    let calibration_library_sha256 = fixture.calibration_library.as_ref().and_then(|path| {
        probe_fixture_calibration_library_sha256(
            path,
            fixture.calibration_library_sha256.as_deref(),
            compute_fixture_hashes,
        )
    });
    validate_fixture_component_sha256_probe(
        name,
        "calibration_profile",
        calibration_profile_sha256.as_ref(),
        &mut issues,
    );
    validate_fixture_component_sha256_probe(
        name,
        "calibration_library",
        calibration_library_sha256.as_ref(),
        &mut issues,
    );

    let (
        calibration_profile_exists,
        calibration_profile_parse_status,
        calibration_profile_reference_patch_count,
    ) = fixture
        .calibration_profile
        .as_ref()
        .map_or((None, None, None), |path| {
            let exists = path.exists();
            if !exists {
                issues.push(format!("{name}:calibration_profile_missing"));
                (Some(false), Some("missing".to_string()), None)
            } else {
                let profile = scanstitch::color_calibration::load_profile(path);
                if let Some(profile) = profile.profile {
                    (
                        Some(true),
                        Some("valid".to_string()),
                        Some(profile.target_patches.len()),
                    )
                } else {
                    issues.push(format!("{name}:calibration_profile_invalid"));
                    (Some(true), Some("invalid".to_string()), None)
                }
            }
        });
    let (
        calibration_library_exists,
        calibration_library_selection_status,
        calibration_library_reference_patch_count,
    ) = fixture
        .calibration_library
        .as_ref()
        .map_or((None, None, None), |path| {
            let exists = path.exists();
            if !exists {
                issues.push(format!("{name}:calibration_library_missing"));
                return (Some(false), Some("missing".to_string()), None);
            }
            if !path.is_dir() {
                issues.push(format!("{name}:calibration_library_not_directory"));
                return (Some(true), Some("not_directory".to_string()), None);
            }

            let calibration = scanstitch::color_calibration::load_calibration(
                None,
                Some(path.as_path()),
                fixture.scanner_profile.as_deref(),
                fixture.roll_profile.as_deref(),
                fixture.film_stock.as_deref(),
                None,
            );
            let status = calibration.diagnostics.status;
            let reference_patch_count = calibration
                .profile
                .as_ref()
                .map(|profile| profile.target_patches.len());
            if calibration.profile.is_none() {
                issues.push(format!(
                    "{name}:calibration_library_selection_unusable:{status}"
                ));
            }
            (Some(true), Some(status), reference_patch_count)
        });
    if fixture.calibration_library.is_none()
        && (fixture.scanner_profile.is_some() || fixture.roll_profile.is_some())
    {
        issues.push(format!("{name}:profile_id_without_calibration_library"));
    }

    let calibration_evidence_declared =
        fixture.calibration_profile.is_some() || fixture.calibration_library.is_some();
    let calibration_evidence_usable = calibration_profile_parse_status.as_deref() == Some("valid")
        || calibration_library_selection_status.as_deref() == Some("applied");
    let calibration_reference_patch_count = calibration_profile_reference_patch_count
        .or(calibration_library_reference_patch_count)
        .filter(|count| *count > 0);
    if fixture.expectations.reference_patch_evaluation_required
        && calibration_reference_patch_count.unwrap_or(0) == 0
    {
        issues.push(format!(
            "{name}:reference_patch_evaluation_required_without_calibration_patches"
        ));
    }
    let validation_ready = all_components_exist && summary_baseline_valid && issues.is_empty();
    let grain_reduction_enabled_declared = fixture_declares_grain_reduction_enabled(fixture);
    let grain_detail_contract_missing_fields =
        fixture_grain_detail_contract_missing_fields(fixture);
    let grain_detail_contract_complete =
        grain_reduction_enabled_declared && grain_detail_contract_missing_fields.is_empty();
    let grain_reduction_effect_contract_missing_fields =
        fixture_grain_reduction_effect_contract_missing_fields(fixture);
    let grain_reduction_effect_contract_complete = grain_reduction_enabled_declared
        && grain_reduction_effect_contract_missing_fields.is_empty();
    let render_dynamic_range_contract_declared =
        fixture_declares_render_dynamic_range_contract(fixture);
    let render_dynamic_range_contract_missing_fields =
        fixture_render_dynamic_range_contract_missing_fields(fixture);
    let render_dynamic_range_contract_complete = render_dynamic_range_contract_declared
        && render_dynamic_range_contract_missing_fields.is_empty();
    let stitch_normalization_contract_declared =
        fixture_declares_stitch_normalization_contract(fixture);
    let stitch_normalization_contract_missing_fields =
        fixture_stitch_normalization_contract_missing_fields(fixture);
    let stitch_normalization_contract_complete = stitch_normalization_contract_declared
        && stitch_normalization_contract_missing_fields.is_empty();
    let geometry_preparation_contract_declared =
        fixture_declares_geometry_preparation_contract(fixture);
    let geometry_preparation_contract_missing_fields =
        fixture_geometry_preparation_contract_missing_fields(fixture);
    let geometry_preparation_contract_complete = geometry_preparation_contract_declared
        && geometry_preparation_contract_missing_fields.is_empty();
    let geometry_accuracy_contract_declared = fixture_declares_geometry_accuracy_contract(fixture);
    let geometry_accuracy_contract_missing_fields =
        fixture_geometry_accuracy_contract_missing_fields(fixture);
    let geometry_accuracy_contract_complete =
        geometry_accuracy_contract_declared && geometry_accuracy_contract_missing_fields.is_empty();
    let orientation_accuracy_contract_declared =
        fixture_declares_orientation_accuracy_contract(fixture);
    let orientation_accuracy_contract_missing_fields =
        fixture_orientation_accuracy_contract_missing_fields(fixture);
    let orientation_accuracy_contract_complete = orientation_accuracy_contract_declared
        && orientation_accuracy_contract_missing_fields.is_empty();
    let negative_reconstruction_contract_declared =
        fixture_declares_negative_reconstruction_contract(fixture);
    let negative_reconstruction_contract_missing_fields =
        fixture_negative_reconstruction_contract_missing_fields(fixture);
    let negative_reconstruction_contract_complete = negative_reconstruction_contract_declared
        && negative_reconstruction_contract_missing_fields.is_empty();

    let mut entry = FixtureCoverageEntry {
        name: name.to_string(),
        component_count: fixture.component_count(),
        component1: fixture.component1.display().to_string(),
        component1_exists,
        component1_sha256,
        component1_tiff,
        component2: fixture
            .component2
            .as_ref()
            .map(|path| path.display().to_string()),
        component2_exists,
        component2_sha256,
        component2_tiff,
        tiff_pair,
        additional_components,
        all_components_exist,
        all_component_sha256_declared,
        all_component_sha256_computed,
        all_component_sha256_matched,
        all_components_readable_tiff,
        all_component_layouts_consistent,
        all_component_dimensions_matched,
        output_dir: fixture
            .output_dir
            .as_ref()
            .map(|path| path.display().to_string()),
        input_mode: fixture.input_mode.clone(),
        bit_depth: fixture.bit_depth,
        grain_reduction: fixture.grain_reduction.clone(),
        grain_strength: fixture.grain_strength,
        grain_scale: fixture.grain_scale,
        grain_reduction_enabled_declared,
        grain_detail_contract_complete,
        grain_detail_contract_missing_fields,
        grain_reduction_effect_contract_complete,
        grain_reduction_effect_contract_missing_fields,
        render_dynamic_range_contract_declared,
        render_dynamic_range_contract_complete,
        render_dynamic_range_contract_missing_fields,
        stitch_normalization_contract_declared,
        stitch_normalization_contract_complete,
        stitch_normalization_contract_missing_fields,
        geometry_preparation_contract_declared,
        geometry_preparation_contract_complete,
        geometry_preparation_contract_missing_fields,
        geometry_accuracy_contract_declared,
        geometry_accuracy_contract_complete,
        geometry_accuracy_contract_missing_fields,
        orientation_accuracy_contract_declared,
        orientation_accuracy_contract_complete,
        orientation_accuracy_contract_missing_fields,
        negative_reconstruction_contract_declared,
        negative_reconstruction_contract_complete,
        negative_reconstruction_contract_missing_fields,
        deskew: fixture.deskew.clone(),
        deskew_angle_degrees: fixture.deskew_angle_degrees,
        force_stitch: fixture.force_stitch,
        force_no_stitch: fixture.force_no_stitch,
        summary_baseline: fixture
            .summary_baseline
            .as_ref()
            .map(|path| path.display().to_string()),
        summary_baseline_exists,
        summary_baseline_parse_status,
        summary_baseline_contract_status,
        summary_baseline_contract_missing_fields,
        summary_baseline_valid,
        summary_baseline_sha256,
        render_review: fixture
            .render_review
            .as_ref()
            .map(|path| path.display().to_string()),
        render_review_exists,
        render_review_sha256,
        render_review_inspection,
        approved_render_review,
        calibration_profile: fixture
            .calibration_profile
            .as_ref()
            .map(|path| path.display().to_string()),
        calibration_profile_exists,
        calibration_profile_parse_status,
        calibration_profile_sha256,
        calibration_library: fixture
            .calibration_library
            .as_ref()
            .map(|path| path.display().to_string()),
        calibration_library_exists,
        calibration_library_selection_status,
        calibration_library_sha256,
        calibration_reference_patch_count,
        scanner_profile: fixture.scanner_profile.clone(),
        roll_profile: fixture.roll_profile.clone(),
        film_stock: fixture.film_stock.clone(),
        scene_tags: fixture.scene_tags.clone(),
        exposure_tags: fixture.exposure_tags.clone(),
        reference_evidence: fixture.reference_evidence.clone(),
        calibration_case: fixture.calibration_case.clone(),
        debug_artifacts_required: fixture.expectations.debug_artifacts_required,
        debug_artifact_kinds_required: fixture.expectations.debug_artifact_kinds_required.clone(),
        calibration_evidence_declared,
        calibration_evidence_usable,
        validation_ready,
        action_items: Vec::new(),
        repair_plan: Vec::new(),
        issues,
    };
    entry.action_items = fixture_coverage_entry_actions(&entry);
    entry.repair_plan = fixture_coverage_entry_repair_plan(&entry);
    entry
}

fn fixture_coverage_to_markdown(summary: &FixtureCoverageSummary) -> String {
    let mut out = String::new();
    out.push_str("# Fixture Coverage\n\n");
    out.push_str(&format!("- status: `{}`\n", summary.status));
    out.push_str(&format!("- fixtures: `{}`\n", summary.fixture_count));
    out.push_str(&format!(
        "- component files declared: `{}` (maximum per fixture: `{}`)\n",
        summary.component_file_count, summary.max_component_count
    ));
    out.push_str(&format!(
        "- N-component fixtures declared / validation-ready: `{}` / `{}`\n",
        summary.n_component_fixture_count, summary.n_component_validation_ready_fixture_count
    ));
    out.push_str(&format!(
        "- complete component sets available / SHA-256 declared / computed / matched: `{}` / `{}` / `{}` / `{}`\n",
        summary.component_set_available_count,
        summary.component_sha256_declared_set_count,
        summary.component_sha256_computed_set_count,
        summary.component_sha256_set_count
    ));
    out.push_str(&format!(
        "- complete TIFF sets readable / layout-consistent / dimension-matched: `{}` / `{}` / `{}`\n",
        summary.readable_tiff_set_count,
        summary.tiff_layout_consistent_set_count,
        summary.tiff_dimension_matched_set_count
    ));
    out.push_str(&format!(
        "- component pairs available: `{}`\n",
        summary.component_pair_available_count
    ));
    out.push_str(&format!(
        "- component SHA-256 pairs declared: `{}`\n",
        summary.component_sha256_declared_pair_count
    ));
    out.push_str(&format!(
        "- component SHA-256 pairs computed: `{}`\n",
        summary.component_sha256_computed_pair_count
    ));
    out.push_str(&format!(
        "- validation-ready component SHA-256 pairs matched: `{}`\n",
        summary.component_sha256_pair_count
    ));
    out.push_str(&format!(
        "- readable TIFF pairs: `{}`\n",
        summary.readable_tiff_pair_count
    ));
    out.push_str(&format!(
        "- TIFF layout-consistent pairs: `{}`\n",
        summary.tiff_layout_consistent_pair_count
    ));
    out.push_str(&format!(
        "- TIFF dimension-matched pairs: `{}`\n",
        summary.tiff_dimension_matched_pair_count
    ));
    out.push_str(&format!(
        "- validation-ready fixtures: `{}`\n",
        summary.validation_ready_fixture_count
    ));
    out.push_str(&format!(
        "- validation-ready complete render-dynamic-range contracts: `{}`\n",
        summary.render_dynamic_range_contract_fixture_count
    ));
    out.push_str(&format!(
        "- validation-ready complete stitch-normalization contracts: `{}`\n",
        summary.stitch_normalization_contract_fixture_count
    ));
    out.push_str(&format!(
        "- validation-ready complete geometry-preparation contracts: `{}`\n",
        summary.geometry_preparation_contract_fixture_count
    ));
    out.push_str(&format!(
        "- validation-ready complete geometry-accuracy contracts: `{}`\n",
        summary.geometry_accuracy_contract_fixture_count
    ));
    out.push_str(&format!(
        "- validation-ready complete orientation-accuracy contracts: `{}`\n",
        summary.orientation_accuracy_contract_fixture_count
    ));
    out.push_str(&format!(
        "- validation-ready complete negative-reconstruction contracts: `{}`\n",
        summary.negative_reconstruction_contract_fixture_count
    ));
    out.push_str(&format!(
        "- validation-ready grain-reduction-enabled fixtures: `{}`\n",
        summary.grain_reduction_enabled_fixture_count
    ));
    out.push_str(&format!(
        "- validation-ready complete grain-detail contracts: `{}`\n",
        summary.grain_detail_contract_fixture_count
    ));
    out.push_str(&format!(
        "- validation-ready complete grain-reduction-effect contracts: `{}`\n",
        summary.grain_reduction_effect_contract_fixture_count
    ));
    out.push_str(&format!(
        "- summary baselines parseable: `{}`\n",
        summary.summary_baseline_parseable_count
    ));
    out.push_str(&format!(
        "- summary baselines contract-complete: `{}`\n",
        summary.summary_baseline_contract_complete_count
    ));
    out.push_str(&format!(
        "- summary baselines declared: `{}`\n",
        summary.summary_baseline_declared_count
    ));
    out.push_str(&format!(
        "- summary baseline SHA-256 fixtures declared: `{}`\n",
        summary.summary_baseline_sha256_declared_count
    ));
    out.push_str(&format!(
        "- summary baseline SHA-256 fixtures computed: `{}`\n",
        summary.summary_baseline_sha256_computed_count
    ));
    out.push_str(&format!(
        "- validation-ready summary baseline SHA-256 fixtures matched: `{}`\n",
        summary.summary_baseline_sha256_count
    ));
    out.push_str(&format!(
        "- calibration evidence usable: `{}`\n",
        summary.calibration_evidence_count
    ));
    out.push_str(&format!(
        "- calibration evidence declared: `{}`\n",
        summary.calibration_evidence_declared_count
    ));
    out.push_str(&format!(
        "- calibration SHA-256 fixtures declared: `{}`\n",
        summary.calibration_sha256_declared_count
    ));
    out.push_str(&format!(
        "- calibration SHA-256 fixtures computed: `{}`\n",
        summary.calibration_sha256_computed_count
    ));
    out.push_str(&format!(
        "- validation-ready calibration SHA-256 fixtures matched: `{}`\n",
        summary.calibration_sha256_count
    ));
    out.push_str(&format!(
        "- uncalibrated fixtures: `{}`\n",
        summary.uncalibrated_fixture_count
    ));
    out.push_str(&format!(
        "- scanner profiles: `{}`\n",
        summary.scanner_profile_count
    ));
    out.push_str(&format!(
        "- unique scanner profiles: `{}`\n",
        summary.unique_scanner_profiles.join(", ")
    ));
    out.push_str(&format!(
        "- missing required scanner profiles: `{}`\n",
        summary.missing_required_scanner_profiles.join(", ")
    ));
    out.push_str(&format!(
        "- roll profiles: `{}`\n",
        summary.roll_profile_count
    ));
    out.push_str(&format!(
        "- unique roll profiles: `{}`\n",
        summary.unique_roll_profiles.join(", ")
    ));
    out.push_str(&format!(
        "- missing required roll profiles: `{}`\n",
        summary.missing_required_roll_profiles.join(", ")
    ));
    out.push_str(&format!(
        "- film stocks: `{}`\n\n",
        summary.film_stock_count
    ));
    out.push_str(&format!(
        "- unique film stocks: `{}`\n",
        summary.unique_film_stocks.join(", ")
    ));
    out.push_str(&format!(
        "- missing required film stocks: `{}`\n",
        summary.missing_required_film_stocks.join(", ")
    ));
    out.push_str(&format!(
        "- scene tags: `{}`\n",
        summary.scene_tags.join(", ")
    ));
    out.push_str(&format!(
        "- missing required scene tags: `{}`\n",
        summary.missing_required_scene_tags.join(", ")
    ));
    out.push_str(&format!(
        "- exposure tags: `{}`\n",
        summary.exposure_tags.join(", ")
    ));
    out.push_str(&format!(
        "- missing required exposure tags: `{}`\n",
        summary.missing_required_exposure_tags.join(", ")
    ));
    out.push_str(&format!(
        "- calibration cases: `{}`\n\n",
        summary.calibration_cases.join(", ")
    ));
    out.push_str(&format!(
        "- missing required calibration cases: `{}`\n\n",
        summary.missing_required_calibration_cases.join(", ")
    ));
    out.push_str(&format!(
        "- reference fixtures: `{}`\n",
        summary.reference_fixture_count
    ));
    out.push_str(&format!(
        "- reference evidence types: `{}`\n",
        summary.reference_evidence_type_count
    ));
    out.push_str(&format!(
        "- reference evidence: `{}`\n",
        summary.reference_evidence.join(", ")
    ));
    out.push_str(&format!(
        "- missing required reference evidence: `{}`\n\n",
        summary.missing_required_reference_evidence.join(", ")
    ));
    out.push_str(&format!(
        "- reference-patch fixtures: `{}`\n",
        summary.reference_patch_fixture_count
    ));
    out.push_str(&format!(
        "- reference patches: `{}`\n\n",
        summary.reference_patch_count
    ));
    out.push_str(&format!(
        "- validation-ready hash-bound approved render reviews: `{}`\n\n",
        summary.approved_render_review_fixture_count
    ));
    out.push_str(&format!(
        "- validation-ready debug-artifact expectation fixtures: `{}`\n",
        summary.debug_artifact_expectation_fixture_count
    ));
    out.push_str(&format!(
        "- required debug artifact kinds declared by validation-ready fixtures: `{}`\n",
        summary.debug_artifact_kinds_required.join(", ")
    ));
    out.push_str(&format!(
        "- missing required debug artifact kinds: `{}`\n\n",
        summary.missing_required_debug_artifact_kinds.join(", ")
    ));
    out.push_str(&format!(
        "- film stock + calibration pairs: `{}`\n",
        summary.film_stock_calibration_pairs.join(", ")
    ));
    out.push_str(&format!(
        "- missing required film stock + calibration pairs: `{}`\n",
        summary
            .missing_required_film_stock_calibration_pairs
            .join(", ")
    ));
    out.push_str(&format!(
        "- scene + exposure pairs: `{}`\n\n",
        summary.scene_exposure_pairs.join(", ")
    ));
    out.push_str(&format!(
        "- missing required scene + exposure pairs: `{}`\n\n",
        summary.missing_required_scene_exposure_pairs.join(", ")
    ));
    if !summary.action_items.is_empty() {
        out.push_str("## Action Items\n\n");
        for action in &summary.action_items {
            out.push_str(&format!("- `{action}`\n"));
        }
        out.push('\n');
    }
    out.push_str("## Requirements\n\n");
    out.push_str(&format!(
        "- min fixtures: `{}`\n",
        optional_usize(summary.coverage_requirements.min_fixtures)
    ));
    out.push_str(&format!(
        "- min complete render-dynamic-range-contract fixtures: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_render_dynamic_range_contract_fixtures
        )
    ));
    out.push_str(&format!(
        "- min complete stitch-normalization-contract fixtures: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_stitch_normalization_contract_fixtures
        )
    ));
    out.push_str(&format!(
        "- min complete geometry-preparation-contract fixtures: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_geometry_preparation_contract_fixtures
        )
    ));
    out.push_str(&format!(
        "- min complete geometry-accuracy-contract fixtures: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_geometry_accuracy_contract_fixtures
        )
    ));
    out.push_str(&format!(
        "- min complete orientation-accuracy-contract fixtures: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_orientation_accuracy_contract_fixtures
        )
    ));
    out.push_str(&format!(
        "- min complete negative-reconstruction-contract fixtures: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_negative_reconstruction_contract_fixtures
        )
    ));
    out.push_str(&format!(
        "- min grain-reduction-enabled fixtures: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_grain_reduction_enabled_fixtures
        )
    ));
    out.push_str(&format!(
        "- min complete grain-detail-contract fixtures: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_grain_detail_contract_fixtures
        )
    ));
    out.push_str(&format!(
        "- min complete grain-reduction-effect-contract fixtures: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_grain_reduction_effect_contract_fixtures
        )
    ));
    out.push_str(&format!(
        "- min component pairs: `{}`\n",
        optional_usize(summary.coverage_requirements.min_component_pairs)
    ));
    out.push_str(&format!(
        "- min component SHA-256 pairs: `{}`\n",
        optional_usize(summary.coverage_requirements.min_component_sha256_pairs)
    ));
    out.push_str(&format!(
        "- min validation-ready N-component fixtures: `{}`\n",
        optional_usize(summary.coverage_requirements.min_n_component_fixtures)
    ));
    out.push_str(&format!(
        "- min complete component SHA-256 sets: `{}`\n",
        optional_usize(summary.coverage_requirements.min_component_sha256_sets)
    ));
    out.push_str(&format!(
        "- min readable TIFF pairs: `{}`\n",
        optional_usize(summary.coverage_requirements.min_readable_tiff_pairs)
    ));
    out.push_str(&format!(
        "- min TIFF layout-consistent pairs: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_tiff_layout_consistent_pairs
        )
    ));
    out.push_str(&format!(
        "- min TIFF dimension-matched pairs: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_tiff_dimension_matched_pairs
        )
    ));
    out.push_str(&format!(
        "- min TIFF bits per sample: `{}`\n",
        summary
            .coverage_requirements
            .min_tiff_bits_per_sample
            .map(|value| value.to_string())
            .unwrap_or_else(|| "not set".to_string())
    ));
    out.push_str(&format!(
        "- min summary baselines: `{}`\n",
        optional_usize(summary.coverage_requirements.min_summary_baselines)
    ));
    out.push_str(&format!(
        "- min summary baseline SHA-256 fixtures: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_summary_baseline_sha256_fixtures
        )
    ));
    out.push_str(&format!(
        "- min calibrated fixtures: `{}`\n",
        optional_usize(summary.coverage_requirements.min_calibrated_fixtures)
    ));
    out.push_str(&format!(
        "- min calibration SHA-256 fixtures: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_calibration_sha256_fixtures
        )
    ));
    out.push_str(&format!(
        "- min uncalibrated fixtures: `{}`\n",
        optional_usize(summary.coverage_requirements.min_uncalibrated_fixtures)
    ));
    out.push_str(&format!(
        "- min unique scanner profiles: `{}`\n",
        optional_usize(summary.coverage_requirements.min_unique_scanner_profiles)
    ));
    out.push_str(&format!(
        "- min unique roll profiles: `{}`\n",
        optional_usize(summary.coverage_requirements.min_unique_roll_profiles)
    ));
    out.push_str(&format!(
        "- min unique film stocks: `{}`\n",
        optional_usize(summary.coverage_requirements.min_unique_film_stocks)
    ));
    out.push_str(&format!(
        "- min film stock + calibration pairs: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_film_stock_calibration_pairs
        )
    ));
    out.push_str(&format!(
        "- min scene + exposure pairs: `{}`\n",
        optional_usize(summary.coverage_requirements.min_scene_exposure_pairs)
    ));
    out.push_str(&format!(
        "- min reference fixtures: `{}`\n",
        optional_usize(summary.coverage_requirements.min_reference_fixtures)
    ));
    out.push_str(&format!(
        "- min reference evidence types: `{}`\n",
        optional_usize(summary.coverage_requirements.min_reference_evidence_types)
    ));
    out.push_str(&format!(
        "- min reference-patch fixtures: `{}`\n",
        optional_usize(summary.coverage_requirements.min_reference_patch_fixtures)
    ));
    out.push_str(&format!(
        "- min reference patches: `{}`\n",
        optional_usize(summary.coverage_requirements.min_reference_patch_count)
    ));
    out.push_str(&format!(
        "- min hash-bound approved render-review fixtures: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_approved_render_review_fixtures
        )
    ));
    out.push_str(&format!(
        "- min debug-artifact expectation fixtures: `{}`\n",
        optional_usize(
            summary
                .coverage_requirements
                .min_debug_artifact_expectation_fixtures
        )
    ));
    out.push_str(&format!(
        "- required scene tags: `{}`\n",
        summary.coverage_requirements.required_scene_tags.join(", ")
    ));
    out.push_str(&format!(
        "- required exposure tags: `{}`\n",
        summary
            .coverage_requirements
            .required_exposure_tags
            .join(", ")
    ));
    out.push_str(&format!(
        "- required calibration cases: `{}`\n\n",
        summary
            .coverage_requirements
            .required_calibration_cases
            .join(", ")
    ));
    out.push_str(&format!(
        "- required scanner profiles: `{}`\n",
        summary
            .coverage_requirements
            .required_scanner_profiles
            .join(", ")
    ));
    out.push_str(&format!(
        "- required roll profiles: `{}`\n\n",
        summary
            .coverage_requirements
            .required_roll_profiles
            .join(", ")
    ));
    out.push_str(&format!(
        "- required reference evidence: `{}`\n\n",
        summary
            .coverage_requirements
            .required_reference_evidence
            .join(", ")
    ));
    out.push_str(&format!(
        "- required film stock + calibration pairs: `{}`\n",
        summary
            .coverage_requirements
            .required_film_stock_calibration_pairs
            .join(", ")
    ));
    out.push_str(&format!(
        "- required scene + exposure pairs: `{}`\n\n",
        summary
            .coverage_requirements
            .required_scene_exposure_pairs
            .join(", ")
    ));
    out.push_str(&format!(
        "- required debug artifact kinds: `{}`\n\n",
        summary
            .coverage_requirements
            .required_debug_artifact_kinds
            .join(", ")
    ));
    out.push_str("| Fixture | Ready | Dynamic range | Stitch normalization | Geometry preparation / orientation | Negative reconstruction | Grain | Components | SHA-256 | TIFF | Baseline | Baseline SHA-256 | Render review | Calibration | Calibration SHA-256 | Reference patches | Film stock | Scene tags | Exposure tags | Reference evidence | Calibration case | Debug expectations | Actions | Issues |\n");
    out.push_str("|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|\n");
    for fixture in &summary.fixtures {
        let components = if fixture.all_components_exist {
            "ok"
        } else {
            "missing"
        };
        let sha256 = fixture_sha256_pair_label(fixture);
        let tiff = fixture_tiff_pair_label(fixture);
        let baseline = fixture
            .summary_baseline_parse_status
            .as_deref()
            .unwrap_or("none");
        let baseline = fixture
            .summary_baseline_contract_status
            .as_deref()
            .map(|contract| format!("{baseline}/{contract}"))
            .unwrap_or_else(|| baseline.to_string());
        let baseline_sha256 = fixture_sha256_probe_label(fixture.summary_baseline_sha256.as_ref());
        let render_review = fixture_coverage_render_review_label(fixture);
        let calibration = if let Some(profile) = &fixture.calibration_profile {
            let status = fixture
                .calibration_profile_parse_status
                .as_deref()
                .unwrap_or("unknown");
            format!("profile: {profile} ({status})")
        } else if let Some(library) = &fixture.calibration_library {
            let status = fixture
                .calibration_library_selection_status
                .as_deref()
                .unwrap_or("unknown");
            format!("library: {library} ({status})")
        } else {
            "none".to_string()
        };
        let calibration_sha256 = fixture_calibration_sha256_label(fixture);
        let reference_patches = fixture
            .calibration_reference_patch_count
            .map(|count| count.to_string())
            .unwrap_or_default();
        let film_stock = fixture.film_stock.as_deref().unwrap_or("");
        let scene_tags = fixture.scene_tags.join(", ");
        let exposure_tags = fixture.exposure_tags.join(", ");
        let reference_evidence = fixture.reference_evidence.join(", ");
        let calibration_case = fixture.calibration_case.as_deref().unwrap_or("");
        let dynamic_range = fixture_coverage_dynamic_range_label(fixture);
        let stitch_normalization = fixture_coverage_stitch_normalization_label(fixture);
        let geometry_preparation = fixture_coverage_geometry_preparation_label(fixture);
        let negative_reconstruction = fixture_coverage_negative_reconstruction_label(fixture);
        let grain = fixture_coverage_grain_label(fixture);
        let debug_expectations = fixture_coverage_debug_expectation_label(fixture);
        let actions = if fixture.action_items.is_empty() {
            "none".to_string()
        } else {
            fixture.action_items.join(", ")
        };
        let issues = if fixture.issues.is_empty() {
            "none".to_string()
        } else {
            fixture.issues.join(", ")
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            fixture.name,
            if fixture.validation_ready {
                "yes"
            } else {
                "no"
            },
            dynamic_range,
            stitch_normalization,
            geometry_preparation,
            negative_reconstruction,
            grain,
            components,
            sha256,
            tiff,
            baseline,
            baseline_sha256,
            render_review,
            calibration,
            calibration_sha256,
            reference_patches,
            film_stock,
            scene_tags,
            exposure_tags,
            reference_evidence,
            calibration_case,
            debug_expectations,
            actions,
            issues
        ));
    }
    if !summary.issues.is_empty() {
        out.push_str("\n## Issues\n\n");
        for issue in &summary.issues {
            out.push_str(&format!("- `{issue}`\n"));
        }
    }
    out
}

fn fixture_coverage_render_review_label(fixture: &FixtureCoverageEntry) -> String {
    let Some(path) = fixture.render_review.as_deref() else {
        return String::new();
    };
    let status = fixture
        .render_review_inspection
        .as_ref()
        .map(|inspection| inspection.status.as_str())
        .unwrap_or("missing");
    let hash = fixture_sha256_probe_label(fixture.render_review_sha256.as_ref());
    format!("{path} ({status}; SHA-256 {hash})")
}

fn fixture_sha256_pair_label(fixture: &FixtureCoverageEntry) -> String {
    if fixture.component2.is_none() {
        return fixture_sha256_probe_label(fixture.component1_sha256.as_ref());
    }
    match (
        fixture.component1_sha256.as_ref(),
        fixture.component2_sha256.as_ref(),
    ) {
        (None, None) => "not declared".to_string(),
        (component1, component2) => format!(
            "{}/{}",
            fixture_sha256_probe_label(component1),
            fixture_sha256_probe_label(component2)
        ),
    }
}

fn fixture_sha256_probe_label(probe: Option<&FixtureSha256Probe>) -> String {
    probe
        .map(|probe| probe.status.clone())
        .unwrap_or_else(|| "not declared".to_string())
}

fn fixture_calibration_sha256_label(fixture: &FixtureCoverageEntry) -> String {
    if fixture.calibration_profile.is_some() {
        return fixture_sha256_probe_label(fixture.calibration_profile_sha256.as_ref());
    }
    if fixture.calibration_library.is_some() {
        return fixture_sha256_probe_label(fixture.calibration_library_sha256.as_ref());
    }
    String::new()
}

fn fixture_coverage_debug_expectation_label(fixture: &FixtureCoverageEntry) -> String {
    if !fixture.debug_artifacts_required && fixture.debug_artifact_kinds_required.is_empty() {
        return String::new();
    }
    let mut parts = Vec::new();
    if fixture.debug_artifacts_required {
        parts.push("debug required".to_string());
    }
    if !fixture.debug_artifact_kinds_required.is_empty() {
        parts.push(format!(
            "kinds {}",
            fixture.debug_artifact_kinds_required.join(", ")
        ));
    }
    parts.join("; ")
}

fn fixture_coverage_dynamic_range_label(fixture: &FixtureCoverageEntry) -> String {
    if !fixture.render_dynamic_range_contract_declared {
        return String::new();
    }
    if fixture.render_dynamic_range_contract_complete {
        "contract=complete".to_string()
    } else {
        format!(
            "contract=incomplete ({})",
            fixture
                .render_dynamic_range_contract_missing_fields
                .join(", ")
        )
    }
}

fn fixture_coverage_stitch_normalization_label(fixture: &FixtureCoverageEntry) -> String {
    if !fixture.stitch_normalization_contract_declared {
        return String::new();
    }
    if fixture.stitch_normalization_contract_complete {
        "contract=complete".to_string()
    } else {
        format!(
            "contract=incomplete ({})",
            fixture
                .stitch_normalization_contract_missing_fields
                .join(", ")
        )
    }
}

fn fixture_coverage_geometry_preparation_label(fixture: &FixtureCoverageEntry) -> String {
    let mut parts = Vec::new();
    if fixture.geometry_preparation_contract_declared {
        if fixture.geometry_preparation_contract_complete {
            parts.push("preparation=complete".to_string());
        } else {
            parts.push(format!(
                "preparation=incomplete ({})",
                fixture
                    .geometry_preparation_contract_missing_fields
                    .join(", ")
            ));
        }
    }
    if fixture.geometry_accuracy_contract_declared {
        if fixture.geometry_accuracy_contract_complete {
            parts.push("accuracy=complete".to_string());
        } else {
            parts.push(format!(
                "accuracy=incomplete ({})",
                fixture.geometry_accuracy_contract_missing_fields.join(", ")
            ));
        }
    }
    if fixture.orientation_accuracy_contract_declared {
        if fixture.orientation_accuracy_contract_complete {
            parts.push("orientation=complete".to_string());
        } else {
            parts.push(format!(
                "orientation=incomplete ({})",
                fixture
                    .orientation_accuracy_contract_missing_fields
                    .join(", ")
            ));
        }
    }
    parts.join("; ")
}

fn fixture_coverage_negative_reconstruction_label(fixture: &FixtureCoverageEntry) -> String {
    if !fixture.negative_reconstruction_contract_declared {
        return String::new();
    }
    if fixture.negative_reconstruction_contract_complete {
        "contract=complete".to_string()
    } else {
        format!(
            "contract=incomplete ({})",
            fixture
                .negative_reconstruction_contract_missing_fields
                .join(", ")
        )
    }
}

fn fixture_coverage_grain_label(fixture: &FixtureCoverageEntry) -> String {
    let Some(mode) = fixture.grain_reduction.as_deref() else {
        return String::new();
    };
    let mut parts = vec![mode.to_string()];
    if let Some(strength) = fixture.grain_strength {
        parts.push(format!("strength={strength:.3}"));
    }
    if let Some(scale) = fixture.grain_scale {
        parts.push(format!("scale={scale:.3}"));
    }
    if fixture.grain_reduction_enabled_declared {
        if fixture.grain_detail_contract_complete {
            parts.push("detail-contract=complete".to_string());
        } else {
            parts.push(format!(
                "detail-contract=incomplete ({})",
                fixture.grain_detail_contract_missing_fields.join(", ")
            ));
        }
        if fixture.grain_reduction_effect_contract_complete {
            parts.push("effect-contract=complete".to_string());
        } else {
            parts.push(format!(
                "effect-contract=incomplete ({})",
                fixture
                    .grain_reduction_effect_contract_missing_fields
                    .join(", ")
            ));
        }
    }
    parts.join("; ")
}

fn fixture_tiff_pair_label(fixture: &FixtureCoverageEntry) -> String {
    if fixture.component2.is_none() {
        return fixture_tiff_probe_label(fixture.component1_tiff.as_ref());
    }
    let labels = format!(
        "{}/{}",
        fixture_tiff_probe_label(fixture.component1_tiff.as_ref()),
        fixture_tiff_probe_label(fixture.component2_tiff.as_ref())
    );
    if let Some(pair) = &fixture.tiff_pair {
        format!(
            "{labels}; layout {}; dimensions {}",
            if pair.layout_consistent {
                "matched"
            } else {
                "mismatch"
            },
            if pair.dimension_matched {
                if pair.dimensions_match == Some(true) {
                    "matched"
                } else {
                    "stitch-compatible"
                }
            } else {
                "mismatch"
            }
        )
    } else {
        labels
    }
}

fn fixture_tiff_probe_label(probe: Option<&FixtureTiffProbe>) -> String {
    let Some(probe) = probe else {
        return "missing".to_string();
    };
    if !probe.readable {
        return probe.status.clone();
    }
    match (
        probe.color_type.as_deref(),
        probe.width,
        probe.height,
        probe.source_bits_per_sample,
    ) {
        (Some(color), Some(width), Some(height), Some(bits)) => {
            format!("{color} {width}x{height} {bits}bpc")
        }
        _ => probe.status.clone(),
    }
}

fn validate_fixture_suite_inputs(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Result<(), Box<dyn std::error::Error>> {
    if cli.component1.is_some()
        || cli.component2.is_some()
        || cli.report.is_some()
        || cli.compare_report.is_some()
        || cli.compare_summary.is_some()
        || cli.write_summary_baseline.is_some()
    {
        return Err("--fixture-suite runs registry entries only; omit --component1/--component2, --report, --compare-report, --compare-summary, and --write-summary-baseline".into());
    }
    if cli.force_stitch && cli.force_no_stitch {
        return Err("--fixture-suite cannot combine --force-stitch and --force-no-stitch".into());
    }
    if cli
        .fixture_suite_fixtures
        .iter()
        .any(|fixture| fixture.trim().is_empty())
    {
        return Err("--fixture-suite-fixture cannot be empty".into());
    }
    let missing_fixtures = cli
        .fixture_suite_fixtures
        .iter()
        .filter(|fixture| !fixtures.contains_key(*fixture))
        .cloned()
        .collect::<Vec<_>>();
    if !missing_fixtures.is_empty() {
        return Err(format!(
            "--fixture-suite-fixture did not match registry fixture(s): {}",
            missing_fixtures.join(", ")
        )
        .into());
    }
    Ok(())
}

fn validate_input_mode_options(
    input_mode: InputMode,
    render_input: RenderInputMode,
) -> Result<(), Box<dyn std::error::Error>> {
    if input_mode == InputMode::Positive && render_input == RenderInputMode::Ica {
        return Err("--input-mode positive cannot be used with --render-input ica because ICA requires density-inverted negative-film data".into());
    }
    Ok(())
}

fn parse_input_mode_label(value: &str) -> Option<InputMode> {
    match value {
        "negative" => Some(InputMode::Negative),
        "positive" => Some(InputMode::Positive),
        _ => None,
    }
}

fn parse_grain_reduction_mode_label(value: &str) -> Option<GrainReductionMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" => Some(GrainReductionMode::Off),
        "on" => Some(GrainReductionMode::On),
        _ => None,
    }
}

fn parse_deskew_mode_label(value: &str) -> Option<DeskewMode> {
    match value {
        "auto" => Some(DeskewMode::Auto),
        "manual" => Some(DeskewMode::Manual),
        "off" => Some(DeskewMode::Off),
        _ => None,
    }
}

fn parse_orientation_correction_label(value: &str) -> Option<OrientationCorrection> {
    match value {
        "none" => Some(OrientationCorrection::None),
        "flip-horizontal" => Some(OrientationCorrection::FlipHorizontal),
        "rotate-180" => Some(OrientationCorrection::Rotate180),
        "flip-vertical" => Some(OrientationCorrection::FlipVertical),
        "rotate-90-clockwise-then-flip-horizontal" => {
            Some(OrientationCorrection::Rotate90ClockwiseThenFlipHorizontal)
        }
        "rotate-90-clockwise" => Some(OrientationCorrection::Rotate90Clockwise),
        "rotate-270-clockwise-then-flip-horizontal" => {
            Some(OrientationCorrection::Rotate270ClockwiseThenFlipHorizontal)
        }
        "rotate-270-clockwise" => Some(OrientationCorrection::Rotate270Clockwise),
        _ => None,
    }
}

fn validate_roll_inputs(cli: &ValidationCli) -> Result<(), Box<dyn std::error::Error>> {
    let Some(roll_dir) = &cli.roll_dir else {
        return Err("--roll-inventory and --roll-suite require --roll-dir".into());
    };
    if !roll_dir.is_dir() {
        return Err(format!("--roll-dir {} is not a directory", roll_dir.display()).into());
    }
    if cli.component1.is_some()
        || cli.component2.is_some()
        || cli.report.is_some()
        || cli.compare_report.is_some()
        || cli.compare_summary.is_some()
        || cli.write_summary_baseline.is_some()
    {
        return Err("--roll-inventory/--roll-suite use --roll-dir; omit --component1/--component2, --report, --compare-report, --compare-summary, and --write-summary-baseline".into());
    }
    if cli.roll_suite && cli.force_stitch {
        return Err("--roll-suite validates independent frames; omit --force-stitch".into());
    }
    if cli.roll_inventory && cli.compare_roll_suite.is_some() {
        return Err("--compare-roll-suite requires --roll-suite, not --roll-inventory".into());
    }
    if !cli.roll_suite
        && !cli.roll_suite_frames.is_empty()
        && cli.write_roll_fixture_registry.is_none()
        && cli.write_roll_fixture_metadata_template.is_none()
        && cli.write_roll_contact_sheet.is_none()
    {
        return Err(
            "--roll-suite-frame requires --roll-suite, --write-roll-fixture-registry, --write-roll-fixture-metadata-template, or --write-roll-contact-sheet".into(),
        );
    }
    if cli.write_roll_fixture_registry.is_some() && !cli.roll_inventory {
        return Err("--write-roll-fixture-registry requires --roll-inventory".into());
    }
    if cli.write_roll_fixture_registry.is_some() && cli.roll_suite {
        return Err("--write-roll-fixture-registry cannot be combined with --roll-suite".into());
    }
    if cli.write_roll_fixture_metadata_template.is_some() && !cli.roll_inventory {
        return Err("--write-roll-fixture-metadata-template requires --roll-inventory".into());
    }
    if cli.write_roll_fixture_metadata_template.is_some() && cli.roll_suite {
        return Err(
            "--write-roll-fixture-metadata-template cannot be combined with --roll-suite".into(),
        );
    }
    if cli.write_roll_contact_sheet.is_some() && !cli.roll_inventory {
        return Err("--write-roll-contact-sheet requires --roll-inventory".into());
    }
    if cli.write_roll_contact_sheet.is_some() && cli.roll_suite {
        return Err("--write-roll-contact-sheet cannot be combined with --roll-suite".into());
    }
    if cli.write_roll_contact_sheet_index.is_some() && cli.write_roll_contact_sheet.is_none() {
        return Err("--write-roll-contact-sheet-index requires --write-roll-contact-sheet".into());
    }
    if cli.write_roll_fixture_registry.is_some()
        && cli.calibration_profile.is_some()
        && cli.calibration_library.is_some()
    {
        return Err("--write-roll-fixture-registry cannot scaffold both --calibration-profile and --calibration-library; choose the evidence path for this registry".into());
    }
    if cli.write_roll_fixture_registry.is_some()
        && (cli.scanner_profile.is_some() || cli.roll_profile.is_some())
        && cli.calibration_library.is_none()
    {
        return Err("--write-roll-fixture-registry requires --calibration-library when --scanner-profile or --roll-profile is supplied".into());
    }
    if cli.write_roll_fixture_registry.is_some()
        && cli.roll_profile.is_some()
        && cli.scanner_profile.is_none()
    {
        return Err("--write-roll-fixture-registry requires --scanner-profile when --roll-profile is supplied".into());
    }
    if cli.write_roll_fixture_registry.is_some() {
        normalized_roll_fixture_labels("--roll-fixture-scene-tag", &cli.roll_fixture_scene_tags)?;
        normalized_roll_fixture_labels(
            "--roll-fixture-exposure-tag",
            &cli.roll_fixture_exposure_tags,
        )?;
    }
    Ok(())
}

fn run_roll_inventory(cli: &ValidationCli) -> RollInventorySummary {
    let roll_dir = cli
        .roll_dir
        .as_ref()
        .expect("roll inputs should require --roll-dir");
    let roll_name = roll_name(roll_dir);
    let mut issues = Vec::new();
    let mut paths = Vec::new();

    match std::fs::read_dir(roll_dir) {
        Ok(entries) => {
            for entry in entries {
                match entry {
                    Ok(entry) => {
                        let path = entry.path();
                        if path.is_file() && is_supported_roll_scan_path(&path) {
                            paths.push(path);
                        }
                    }
                    Err(err) => issues.push(format!("roll_inventory:read_dir_entry_failed:{err}")),
                }
            }
        }
        Err(err) => {
            issues.push(format!("roll_inventory:read_dir_failed:{err}"));
        }
    }
    paths.sort_by_key(|path| {
        path.file_name()
            .map(|name| name.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default()
    });

    let mut frames = paths
        .iter()
        .map(|path| inspect_roll_frame(path, cli.input_mode, cli.bit_depth))
        .collect::<Vec<_>>();
    let sequence_gaps = roll_sequence_gaps(&frames);
    for gap in &sequence_gaps {
        issues.push(format!(
            "roll_inventory:sequence_gap:{}{}-{}",
            gap.prefix,
            roll_padded_number(gap.start, gap.width),
            roll_padded_number(gap.end, gap.width)
        ));
    }
    if frames.is_empty() {
        issues.push("roll_inventory:no_supported_scan_files".to_string());
    }
    for frame in &frames {
        if !frame.usable {
            issues.push(format!(
                "roll_inventory:{}:unreadable",
                roll_frame_issue_label(frame)
            ));
        }
        if cli.input_mode == InputMode::Positive
            && frame
                .positive_input_probe
                .as_ref()
                .is_some_and(|probe| probe.likely_negative_like)
        {
            issues.push(format!(
                "roll_inventory:{}:positive_input_negative_like",
                roll_frame_issue_label(frame)
            ));
        }
    }

    let usable_frame_count = frames.iter().filter(|frame| frame.usable).count();
    let unreadable_frame_count = frames.len().saturating_sub(usable_frame_count);
    let status = if frames.is_empty() || unreadable_frame_count > 0 {
        "failed"
    } else if !issues.is_empty() {
        "review_required"
    } else {
        "passed"
    }
    .to_string();

    RollInventorySummary {
        status,
        roll_dir: roll_dir.display().to_string(),
        roll_name,
        frame_count: frames.len(),
        usable_frame_count,
        unreadable_frame_count,
        sequence_gap_count: sequence_gaps.len(),
        sequence_gaps,
        frames: {
            frames.shrink_to_fit();
            frames
        },
        issues,
    }
}

fn roll_fixture_registry_scaffold(
    cli: &ValidationCli,
    inventory: &RollInventorySummary,
) -> Result<FixtureRegistry, Box<dyn std::error::Error>> {
    let roll_slug = slug_label(&inventory.roll_name);
    let output_root = roll_mode_output_dir(cli);
    let metadata = load_roll_fixture_metadata(cli.roll_fixture_metadata.as_ref())?;
    let scene_tags =
        normalized_roll_fixture_labels("--roll-fixture-scene-tag", &cli.roll_fixture_scene_tags)?;
    let exposure_tags = normalized_roll_fixture_labels(
        "--roll-fixture-exposure-tag",
        &cli.roll_fixture_exposure_tags,
    )?;
    let mut fixtures = BTreeMap::new();
    let mut selection_issues = Vec::new();
    let mut matched_metadata_keys = BTreeSet::new();
    let selected_frames = roll_suite_render_frames(cli, &inventory.frames, &mut selection_issues);
    if !selection_issues.is_empty() {
        return Err(format!(
            "--write-roll-fixture-registry frame selection failed: {}",
            selection_issues.join(", ")
        )
        .into());
    }

    for frame in selected_frames.into_iter().filter(|frame| frame.usable) {
        let frame_slug = roll_frame_slug(frame);
        let name = format!("{roll_slug}-{frame_slug}");
        let component = PathBuf::from(&frame.path);
        let mut expectations = FixtureExpectations {
            stitch_decision: Some("skipped_single_input".to_string()),
            output_color_space: Some("linear_prophoto_rgb_d50".to_string()),
            ..FixtureExpectations::default()
        };
        if cli.input_mode == InputMode::Positive {
            expectations.render_input_source = Some("positive_scan_rgb".to_string());
            expectations.mapping_strategy = Some("positive_rgb_passthrough".to_string());
        }

        let mut fixture = FixtureEntry {
            component1: component,
            component2: None,
            component1_sha256: None,
            component2_sha256: None,
            additional_components: Vec::new(),
            output_dir: Some(output_root.join(&frame_slug)),
            input_mode: Some(cli.input_mode.as_str().to_string()),
            bit_depth: Some(cli.bit_depth),
            grain_reduction: Some(cli.grain.grain_reduction.as_str().to_string()),
            grain_strength: Some(cli.grain.grain_strength),
            grain_scale: Some(cli.grain.grain_scale),
            deskew: Some(cli.geometry.deskew.as_str().to_string()),
            orientation_correction: Some(cli.geometry.orientation_correction.as_str().to_string()),
            deskew_angle_degrees: (cli.geometry.deskew == DeskewMode::Manual)
                .then_some(cli.geometry.deskew_angle_degrees),
            force_stitch: false,
            force_no_stitch: true,
            calibration_profile: cli.calibration_profile.clone(),
            calibration_profile_sha256: None,
            calibration_library: cli.calibration_library.clone(),
            calibration_library_sha256: None,
            scanner_profile: cli.scanner_profile.clone(),
            roll_profile: cli.roll_profile.clone(),
            film_stock: cli.film_stock.clone(),
            scene_tags: scene_tags.clone(),
            exposure_tags: exposure_tags.clone(),
            reference_evidence: Vec::new(),
            render_review: None,
            render_review_sha256: None,
            calibration_case: roll_fixture_calibration_case(cli),
            expectations,
            summary_baseline: Some(
                PathBuf::from("local-fixtures")
                    .join("baselines")
                    .join(format!("{name}-summary-baseline.json")),
            ),
            summary_baseline_sha256: None,
            description: Some(format!(
                "Scaffolded from roll inventory frame {}",
                frame.name
            )),
        };
        if let Some((metadata_key, entry)) =
            roll_fixture_metadata_entry_for_frame(metadata.as_ref(), frame, &name)?
        {
            matched_metadata_keys.insert(metadata_key.to_string());
            apply_roll_fixture_metadata_entry(&mut fixture, entry);
        }

        fixtures.insert(name.clone(), fixture);
    }

    if fixtures.is_empty() {
        return Err("--write-roll-fixture-registry found no usable frames to scaffold".into());
    }

    if let Some(metadata) = &metadata {
        let unmatched_keys = metadata
            .frames
            .keys()
            .filter(|key| !matched_metadata_keys.contains(*key))
            .cloned()
            .collect::<Vec<_>>();
        if !unmatched_keys.is_empty() {
            return Err(format!(
                "--roll-fixture-metadata contains frame key(s) that did not match selected usable inventory frames: {}",
                unmatched_keys.join(", ")
            )
            .into());
        }
    }

    let registry = FixtureRegistry {
        fixtures,
        coverage_requirements: metadata
            .as_ref()
            .map(|metadata| metadata.coverage_requirements.clone())
            .unwrap_or_default(),
    };
    let issues = validate_fixture_registry(&registry);
    if !issues.is_empty() {
        return Err(format!(
            "--write-roll-fixture-registry produced invalid fixture registry: {}",
            issues.join(", ")
        )
        .into());
    }

    Ok(registry)
}

fn load_roll_fixture_metadata(
    path: Option<&PathBuf>,
) -> Result<Option<RollFixtureMetadata>, Box<dyn std::error::Error>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let file = File::open(path)?;
    let metadata = serde_json::from_reader::<_, RollFixtureMetadata>(BufReader::new(file))?;
    validate_roll_fixture_metadata(path, &metadata)?;
    Ok(Some(metadata))
}

fn validate_roll_fixture_metadata(
    path: &Path,
    metadata: &RollFixtureMetadata,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut issues = Vec::new();
    let registry = FixtureRegistry {
        fixtures: BTreeMap::new(),
        coverage_requirements: metadata.coverage_requirements.clone(),
    };
    issues.extend(
        validate_fixture_registry(&registry)
            .into_iter()
            .filter(|issue| issue != "fixtures_empty"),
    );
    for (key, entry) in &metadata.frames {
        if key.trim().is_empty() {
            issues.push("frame_key_empty".to_string());
        }
        validate_optional_path(
            key,
            "summary_baseline",
            entry.summary_baseline.as_deref(),
            &mut issues,
        );
        validate_optional_sha256(
            key,
            "summary_baseline_sha256",
            entry.summary_baseline_sha256.as_deref(),
            &mut issues,
        );
        validate_optional_path(
            key,
            "render_review",
            entry.render_review.as_deref(),
            &mut issues,
        );
        validate_optional_sha256(
            key,
            "render_review_sha256",
            entry.render_review_sha256.as_deref(),
            &mut issues,
        );
        validate_optional_path(
            key,
            "calibration_profile",
            entry.calibration_profile.as_deref(),
            &mut issues,
        );
        validate_optional_sha256(
            key,
            "calibration_profile_sha256",
            entry.calibration_profile_sha256.as_deref(),
            &mut issues,
        );
        validate_optional_path(
            key,
            "calibration_library",
            entry.calibration_library.as_deref(),
            &mut issues,
        );
        validate_optional_sha256(
            key,
            "calibration_library_sha256",
            entry.calibration_library_sha256.as_deref(),
            &mut issues,
        );
        validate_optional_label(
            key,
            "scanner_profile",
            entry.scanner_profile.as_deref(),
            &mut issues,
        );
        validate_optional_label(
            key,
            "roll_profile",
            entry.roll_profile.as_deref(),
            &mut issues,
        );
        validate_optional_label(key, "film_stock", entry.film_stock.as_deref(), &mut issues);
        validate_optional_label(
            key,
            "calibration_case",
            entry.calibration_case.as_deref(),
            &mut issues,
        );
        validate_optional_label(
            key,
            "description",
            entry.description.as_deref(),
            &mut issues,
        );
        if let Some(values) = &entry.scene_tags {
            validate_unique_non_empty_values(&format!("{key}:scene_tag"), values, &mut issues);
        }
        if let Some(values) = &entry.exposure_tags {
            validate_unique_non_empty_values(&format!("{key}:exposure_tag"), values, &mut issues);
        }
        if let Some(values) = &entry.reference_evidence {
            validate_unique_non_empty_values(
                &format!("{key}:reference_evidence"),
                values,
                &mut issues,
            );
        }
        validate_fixture_expectations(key, &entry.expectations, 2, &mut issues);
        if entry.calibration_profile.is_some() && entry.calibration_library.is_some() {
            issues.push(format!(
                "{key}:calibration_profile_and_calibration_library_both_declared"
            ));
        }
        if entry.summary_baseline_sha256.is_some() && entry.summary_baseline.is_none() {
            issues.push(format!(
                "{key}:summary_baseline_sha256_without_summary_baseline"
            ));
        }
        if entry.render_review_sha256.is_some() && entry.render_review.is_none() {
            issues.push(format!("{key}:render_review_sha256_without_render_review"));
        }
        if entry.calibration_profile_sha256.is_some() && entry.calibration_profile.is_none() {
            issues.push(format!(
                "{key}:calibration_profile_sha256_without_calibration_profile"
            ));
        }
        if entry.calibration_library_sha256.is_some() && entry.calibration_library.is_none() {
            issues.push(format!(
                "{key}:calibration_library_sha256_without_calibration_library"
            ));
        }
    }
    if !issues.is_empty() {
        return Err(format!(
            "{} contains invalid roll fixture metadata: {}",
            path.display(),
            issues.join(", ")
        )
        .into());
    }
    Ok(())
}

fn roll_fixture_metadata_template(
    cli: &ValidationCli,
    inventory: &RollInventorySummary,
) -> Result<RollFixtureMetadata, Box<dyn std::error::Error>> {
    let existing_metadata = load_roll_fixture_metadata(cli.roll_fixture_metadata.as_ref())?;
    let scene_tags =
        normalized_roll_fixture_labels("--roll-fixture-scene-tag", &cli.roll_fixture_scene_tags)?;
    let exposure_tags = normalized_roll_fixture_labels(
        "--roll-fixture-exposure-tag",
        &cli.roll_fixture_exposure_tags,
    )?;
    let mut selection_issues = Vec::new();
    let selected_frames = roll_suite_render_frames(cli, &inventory.frames, &mut selection_issues);
    if !selection_issues.is_empty() {
        return Err(format!(
            "--write-roll-fixture-metadata-template frame selection failed: {}",
            selection_issues.join(", ")
        )
        .into());
    }

    let mut frames = BTreeMap::new();
    let mut matched_metadata_keys = BTreeSet::new();
    let roll_slug = slug_label(&inventory.roll_name);
    for frame in selected_frames.into_iter().filter(|frame| frame.usable) {
        let frame_slug = roll_frame_slug(frame);
        let fixture_name = format!("{roll_slug}-{frame_slug}");
        if let Some((metadata_key, entry)) =
            roll_fixture_metadata_entry_for_frame(existing_metadata.as_ref(), frame, &fixture_name)?
        {
            matched_metadata_keys.insert(metadata_key.to_string());
            frames.insert(frame.stem.clone(), entry.clone());
            continue;
        }

        frames.insert(
            frame.stem.clone(),
            RollFixtureMetadataEntry {
                calibration_profile: cli.calibration_profile.clone(),
                calibration_profile_sha256: None,
                calibration_library: cli.calibration_library.clone(),
                calibration_library_sha256: None,
                scanner_profile: cli.scanner_profile.clone(),
                roll_profile: cli.roll_profile.clone(),
                film_stock: cli.film_stock.clone(),
                scene_tags: (!scene_tags.is_empty()).then_some(scene_tags.clone()),
                exposure_tags: (!exposure_tags.is_empty()).then_some(exposure_tags.clone()),
                reference_evidence: None,
                render_review: None,
                render_review_sha256: None,
                calibration_case: roll_fixture_calibration_case(cli),
                expectations: FixtureExpectations::default(),
                summary_baseline: None,
                summary_baseline_sha256: None,
                description: Some(format!(
                    "TODO: curate factual fixture metadata for {}",
                    frame.name
                )),
            },
        );
    }

    if frames.is_empty() {
        return Err("--write-roll-fixture-metadata-template found no usable frames".into());
    }

    if let Some(metadata) = &existing_metadata {
        let unmatched_keys = metadata
            .frames
            .keys()
            .filter(|key| !matched_metadata_keys.contains(*key))
            .cloned()
            .collect::<Vec<_>>();
        if !unmatched_keys.is_empty() {
            return Err(format!(
                "--roll-fixture-metadata contains frame key(s) that did not match selected usable inventory frames while refreshing template: {}",
                unmatched_keys.join(", ")
            )
            .into());
        }
    }

    Ok(RollFixtureMetadata {
        coverage_requirements: existing_metadata
            .as_ref()
            .map(|metadata| metadata.coverage_requirements.clone())
            .unwrap_or_default(),
        frames,
    })
}

fn write_roll_contact_sheet(
    cli: &ValidationCli,
    inventory: &RollInventorySummary,
    sheet_path: &Path,
    index_path: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut selection_issues = Vec::new();
    let selected_frames = roll_suite_render_frames(cli, &inventory.frames, &mut selection_issues);
    if !selection_issues.is_empty() {
        return Err(format!(
            "--write-roll-contact-sheet frame selection failed: {}",
            selection_issues.join(", ")
        )
        .into());
    }
    let frames = selected_frames
        .into_iter()
        .filter(|frame| frame.usable)
        .collect::<Vec<_>>();
    if frames.is_empty() {
        return Err("--write-roll-contact-sheet found no usable frames".into());
    }

    let columns = ROLL_CONTACT_SHEET_COLUMNS.min(frames.len()).max(1);
    let rows = frames.len().div_ceil(columns);
    let sheet_width = ROLL_CONTACT_SHEET_GUTTER
        + columns as u32 * (ROLL_CONTACT_SHEET_THUMB_WIDTH + ROLL_CONTACT_SHEET_GUTTER);
    let sheet_height = ROLL_CONTACT_SHEET_GUTTER
        + rows as u32 * (ROLL_CONTACT_SHEET_THUMB_HEIGHT + ROLL_CONTACT_SHEET_GUTTER);
    let mut sheet = RgbImage::from_pixel(sheet_width, sheet_height, Rgb([238, 238, 238]));
    let mut index_entries = Vec::new();

    for (idx, frame) in frames.iter().enumerate() {
        let row = idx / columns;
        let column = idx % columns;
        let tile_x = ROLL_CONTACT_SHEET_GUTTER
            + column as u32 * (ROLL_CONTACT_SHEET_THUMB_WIDTH + ROLL_CONTACT_SHEET_GUTTER);
        let tile_y = ROLL_CONTACT_SHEET_GUTTER
            + row as u32 * (ROLL_CONTACT_SHEET_THUMB_HEIGHT + ROLL_CONTACT_SHEET_GUTTER);
        let thumbnail = roll_frame_contact_thumbnail(frame, cli.input_mode, cli.bit_depth)?;
        let offset_x = tile_x + (ROLL_CONTACT_SHEET_THUMB_WIDTH - thumbnail.width()) / 2;
        let offset_y = tile_y + (ROLL_CONTACT_SHEET_THUMB_HEIGHT - thumbnail.height()) / 2;
        draw_tile_background(
            &mut sheet,
            tile_x,
            tile_y,
            ROLL_CONTACT_SHEET_THUMB_WIDTH,
            ROLL_CONTACT_SHEET_THUMB_HEIGHT,
        );
        blit_rgb_image(&thumbnail, &mut sheet, offset_x, offset_y);
        index_entries.push(serde_json::json!({
            "index": idx,
            "name": frame.name,
            "stem": frame.stem,
            "path": frame.path,
            "row": row,
            "column": column,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "tile_width": ROLL_CONTACT_SHEET_THUMB_WIDTH,
            "tile_height": ROLL_CONTACT_SHEET_THUMB_HEIGHT,
            "preview_transform": if cli.input_mode == InputMode::Negative {
                "per_channel_stretch_inverted_gamma"
            } else {
                "per_channel_stretch_gamma"
            },
            "source_width": frame.width,
            "source_height": frame.height,
            "source_color_type": frame.color_type,
            "source_bits_per_sample": frame.source_bits_per_sample,
        }));
    }

    write_rgb_image(sheet_path, &sheet)?;
    if let Some(index_path) = index_path {
        let index = serde_json::json!({
            "contact_sheet": sheet_path.display().to_string(),
            "roll_dir": inventory.roll_dir,
            "roll_name": inventory.roll_name,
            "input_mode": cli.input_mode.as_str(),
            "bit_depth": cli.bit_depth,
            "columns": columns,
            "rows": rows,
            "thumb_width": ROLL_CONTACT_SHEET_THUMB_WIDTH,
            "thumb_height": ROLL_CONTACT_SHEET_THUMB_HEIGHT,
            "gutter": ROLL_CONTACT_SHEET_GUTTER,
            "frames": index_entries,
        });
        write_text(
            index_path,
            &serde_json::to_string_pretty(&index).expect("contact sheet index should serialize"),
        )?;
    }
    Ok(())
}

fn write_orientation_review_package(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
    output_dir: &Path,
) -> Result<(OrientationReviewManifest, PathBuf, PathBuf), Box<dyn std::error::Error>> {
    let inputs = resolve_orientation_review_inputs(cli, fixtures)?;
    let input_mode = effective_input_mode(cli, fixtures)?;
    let bit_depth = effective_bit_depth(cli, fixtures)?;
    let geometry = effective_geometry(cli, fixtures)?;
    let preview_transform = if input_mode == InputMode::Negative {
        "per_channel_stretch_inverted_gamma_orientation_only"
    } else {
        "per_channel_stretch_gamma_orientation_only"
    };
    let mut previews = Vec::with_capacity(inputs.len());
    let mut expectations = Vec::with_capacity(inputs.len());

    for (index, path) in inputs.iter().enumerate() {
        let component_index = index + 1;
        let mut loaded = tiff_io::load_tiff_u16(path, bit_depth).map_err(|error| {
            format!(
                "failed to decode orientation-review component {component_index} {}: {error}",
                path.display()
            )
        })?;
        tiff_io::apply_orientation_correction_u16(
            &mut loaded,
            geometry.orientation_correction.exif_tag(),
            geometry.orientation_correction.as_str(),
            bit_depth,
        )?;
        let preview = decoded_orientation_preview(
            &loaded.image,
            input_mode,
            ORIENTATION_REVIEW_PREVIEW_MAX_WIDTH,
            ORIENTATION_REVIEW_PREVIEW_MAX_HEIGHT,
        )?;
        let preview_name = format!("component-{component_index:02}-orientation-preview.png");
        let preview_path = output_dir.join(&preview_name);
        write_rgb_image(&preview_path, &preview)?;
        let orientation = &loaded.diagnostics.orientation;
        let correction = &loaded.diagnostics.orientation_correction;
        previews.push(OrientationReviewPreview {
            component_index,
            source_path: path.to_string_lossy().to_string(),
            preview_path: preview_name,
        });
        expectations.push(OrientationComponentExpectation {
            component_index,
            upright_approved: false,
            decoded_pixel_sha256: loaded.diagnostics.decoded_pixel_sha256.clone(),
            tag_present: orientation.tag_value.is_some(),
            tag_value: orientation.tag_value,
            transform: correction.effective_transform.clone(),
            applied: orientation.applied || correction.applied,
            source_width: orientation.source_width,
            source_height: orientation.source_height,
            output_width: correction.output_width,
            output_height: correction.output_height,
        });
    }

    let manifest = OrientationReviewManifest {
        schema_version: 1,
        fixture: cli.fixture.clone(),
        review_status: "requires_human_approval",
        instructions: "Inspect every preview for semantic uprightness. Only after approval, copy orientation_components_expected into the fixture registry and change each upright_approved value to true. The preview stretch/inversion is for orientation review, not colour approval.",
        input_mode: input_mode.as_str().to_string(),
        working_bit_depth: bit_depth,
        orientation_correction: geometry.orientation_correction.as_str().to_string(),
        preview_transform,
        previews,
        orientation_components_expected: expectations,
    };
    let json_path = output_dir.join("orientation-review.json");
    let md_path = output_dir.join("orientation-review.md");
    write_text(
        &json_path,
        &serde_json::to_string_pretty(&manifest)
            .expect("orientation review manifest should serialize"),
    )?;
    write_text(&md_path, &orientation_review_to_markdown(&manifest))?;
    Ok((manifest, json_path, md_path))
}

fn orientation_review_to_markdown(manifest: &OrientationReviewManifest) -> String {
    let mut out = String::from("# Orientation Review Draft\n\n");
    out.push_str(
        "Status: `requires_human_approval`. This package never approves its own output.\n\n",
    );
    out.push_str(&format!("{}\n\n", manifest.instructions));
    out.push_str(&format!(
        "Fixture: `{}`. Input mode: `{}`. Working bit depth: `{}`. Metadata-relative orientation correction: `{}`. Preview transform: `{}`.\n\n",
        manifest.fixture,
        manifest.input_mode,
        manifest.working_bit_depth,
        manifest.orientation_correction,
        manifest.preview_transform
    ));
    out.push_str("| Component | Preview | Decoded pixel SHA-256 | Tag | Transform | Applied | Source | Output |\n");
    out.push_str("|-|-|-|-|-|-|-|-|\n");
    for (preview, expected) in manifest
        .previews
        .iter()
        .zip(&manifest.orientation_components_expected)
    {
        let tag = expected
            .tag_value
            .map(|value| value.to_string())
            .unwrap_or_else(|| "absent".to_string());
        out.push_str(&format!(
            "| {} | [{}]({}) | `{}` | {} | `{}` | {} | {}x{} | {}x{} |\n",
            expected.component_index,
            preview.preview_path,
            preview.preview_path,
            expected.decoded_pixel_sha256,
            tag,
            expected.transform,
            expected.applied,
            expected.source_width,
            expected.source_height,
            expected.output_width,
            expected.output_height
        ));
    }
    out.push_str("\nThe JSON draft deliberately leaves every `upright_approved` value `false`. A digest match proves only that strict validation saw the same decoded pixels; the reviewer supplies the semantic uprightness decision.\n");
    out
}

fn draw_tile_background(sheet: &mut RgbImage, x: u32, y: u32, width: u32, height: u32) {
    for yy in y..(y + height).min(sheet.height()) {
        for xx in x..(x + width).min(sheet.width()) {
            sheet.put_pixel(xx, yy, Rgb([250, 250, 250]));
        }
    }
}

fn blit_rgb_image(source: &RgbImage, target: &mut RgbImage, x: u32, y: u32) {
    for yy in 0..source.height() {
        for xx in 0..source.width() {
            let target_x = x + xx;
            let target_y = y + yy;
            if target_x < target.width() && target_y < target.height() {
                target.put_pixel(target_x, target_y, *source.get_pixel(xx, yy));
            }
        }
    }
}

fn roll_frame_contact_thumbnail(
    frame: &RollFrameInspection,
    input_mode: InputMode,
    bit_depth: u8,
) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let loaded = tiff_io::load_tiff_u16(Path::new(&frame.path), bit_depth)?;
    decoded_orientation_preview(
        &loaded.image,
        input_mode,
        ROLL_CONTACT_SHEET_THUMB_WIDTH,
        ROLL_CONTACT_SHEET_THUMB_HEIGHT,
    )
}

fn decoded_orientation_preview(
    image: &Array3<u16>,
    input_mode: InputMode,
    max_width: u32,
    max_height: u32,
) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let (height, width, channels) = image.dim();
    if width == 0 || height == 0 || channels < 3 || max_width == 0 || max_height == 0 {
        return Err(
            "cannot build an orientation preview from empty or unsupported dimensions".into(),
        );
    }
    let (mins, maxes) = sampled_channel_min_max(image);
    let scale = (max_width as f64 / width as f64)
        .min(max_height as f64 / height as f64)
        .max(1.0 / width.max(height) as f64);
    let thumb_width = ((width as f64 * scale).round() as u32).max(1);
    let thumb_height = ((height as f64 * scale).round() as u32).max(1);
    let mut thumbnail = RgbImage::new(thumb_width, thumb_height);

    for y in 0..thumb_height {
        let source_y = ((y as usize * height) / thumb_height as usize).min(height - 1);
        for x in 0..thumb_width {
            let source_x = ((x as usize * width) / thumb_width as usize).min(width - 1);
            let mut rgb = [0u8; 3];
            for channel in 0..3 {
                let min = mins[channel] as f64;
                let max = maxes[channel] as f64;
                let sample = image[[source_y, source_x, channel]] as f64;
                let mut normalized = if max > min {
                    ((sample - min) / (max - min)).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                if input_mode == InputMode::Negative {
                    normalized = 1.0 - normalized;
                }
                rgb[channel] = (normalized.powf(1.0 / 2.2) * 255.0).round() as u8;
            }
            thumbnail.put_pixel(x, y, Rgb(rgb));
        }
    }
    Ok(thumbnail)
}

fn sampled_channel_min_max(image: &Array3<u16>) -> ([u16; 3], [u16; 3]) {
    let (height, width, _) = image.dim();
    let mut mins = [u16::MAX; 3];
    let mut maxes = [0u16; 3];
    let target_samples = 200_000usize;
    let pixel_count = height.saturating_mul(width).max(1);
    let step = (pixel_count as f64 / target_samples as f64)
        .sqrt()
        .ceil()
        .max(1.0) as usize;
    for y in (0..height).step_by(step) {
        for x in (0..width).step_by(step) {
            for channel in 0..3 {
                let sample = image[[y, x, channel]];
                mins[channel] = mins[channel].min(sample);
                maxes[channel] = maxes[channel].max(sample);
            }
        }
    }
    (mins, maxes)
}

fn roll_fixture_metadata_entry_for_frame<'a>(
    metadata: Option<&'a RollFixtureMetadata>,
    frame: &RollFrameInspection,
    fixture_name: &str,
) -> Result<Option<(&'a str, &'a RollFixtureMetadataEntry)>, Box<dyn std::error::Error>> {
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    let frame_slug = roll_frame_slug(frame);
    let candidates = [&frame.name, &frame.stem, frame_slug.as_str(), fixture_name];
    let mut matches = Vec::new();
    for candidate in candidates {
        if let Some((key, entry)) = metadata.frames.get_key_value(candidate) {
            if !matches
                .iter()
                .any(|(existing_key, _): &(&String, &RollFixtureMetadataEntry)| {
                    *existing_key == key
                })
            {
                matches.push((key, entry));
            }
        }
    }
    match matches.len() {
        0 => Ok(None),
        1 => {
            let (key, entry) = matches.into_iter().next().expect("one metadata match");
            Ok(Some((key.as_str(), entry)))
        }
        _ => Err(format!(
            "--roll-fixture-metadata has multiple entries matching frame {}: {}",
            frame.name,
            matches
                .iter()
                .map(|(key, _)| key.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
        .into()),
    }
}

fn apply_roll_fixture_metadata_entry(fixture: &mut FixtureEntry, entry: &RollFixtureMetadataEntry) {
    if let Some(value) = &entry.calibration_profile {
        fixture.calibration_profile = Some(value.clone());
    }
    if let Some(value) = &entry.calibration_profile_sha256 {
        fixture.calibration_profile_sha256 = Some(value.clone());
    }
    if let Some(value) = &entry.calibration_library {
        fixture.calibration_library = Some(value.clone());
    }
    if let Some(value) = &entry.calibration_library_sha256 {
        fixture.calibration_library_sha256 = Some(value.clone());
    }
    if let Some(value) = &entry.scanner_profile {
        fixture.scanner_profile = Some(value.clone());
    }
    if let Some(value) = &entry.roll_profile {
        fixture.roll_profile = Some(value.clone());
    }
    if let Some(value) = &entry.film_stock {
        fixture.film_stock = Some(value.clone());
    }
    if let Some(values) = &entry.scene_tags {
        fixture.scene_tags = values.clone();
    }
    if let Some(values) = &entry.exposure_tags {
        fixture.exposure_tags = values.clone();
    }
    if let Some(values) = &entry.reference_evidence {
        fixture.reference_evidence = values.clone();
    }
    if let Some(value) = &entry.render_review {
        fixture.render_review = Some(value.clone());
    }
    if let Some(value) = &entry.render_review_sha256 {
        fixture.render_review_sha256 = Some(value.clone());
    }
    if let Some(value) = &entry.calibration_case {
        fixture.calibration_case = Some(value.clone());
    }
    merge_fixture_expectations(&mut fixture.expectations, entry.expectations.clone());
    if let Some(value) = &entry.summary_baseline {
        fixture.summary_baseline = Some(value.clone());
    }
    if let Some(value) = &entry.summary_baseline_sha256 {
        fixture.summary_baseline_sha256 = Some(value.clone());
    }
    if let Some(value) = &entry.description {
        fixture.description = Some(value.clone());
    }
}

fn merge_fixture_expectations(base: &mut FixtureExpectations, overlay: FixtureExpectations) {
    macro_rules! merge_option {
        ($field:ident) => {
            if let Some(value) = overlay.$field {
                base.$field = Some(value);
            }
        };
    }
    macro_rules! merge_vec {
        ($field:ident) => {
            if !overlay.$field.is_empty() {
                base.$field = overlay.$field;
            }
        };
    }

    merge_option!(deskew_all_components_applied);
    merge_option!(deskew_minimum_component_retained_area_ratio_min);
    merge_option!(border_crop_all_components_cropped);
    merge_option!(border_crop_minimum_removed_edge_count_per_component_min);
    merge_option!(border_crop_retained_area_ratio_min);
    merge_option!(border_crop_retained_area_ratio_max);
    merge_option!(border_crop_rejected);
    merge_option!(deskew_correction_degrees_expected);
    merge_option!(deskew_correction_tolerance_degrees);
    merge_vec!(border_crop_components_expected);
    merge_vec!(orientation_components_expected);
    merge_option!(stitch_decision);
    merge_option!(seam_exposure_model);
    merge_option!(seam_exposure_held_out_validation_passed);
    merge_option!(seam_exposure_held_out_improvement_over_gain_min);
    merge_option!(seam_exposure_offset_normalized_abs_max);
    merge_option!(seam_exposure_spatial_slope_abs_min);
    merge_option!(seam_exposure_spatial_slope_abs_max);
    merge_option!(seam_exposure_spatial_slope_agreement_ratio_min);
    merge_option!(seam_exposure_held_out_spatial_improvement_over_best_constant_min);
    merge_option!(seam_exposure_spatial_offset_slope_normalized_abs_min);
    merge_option!(seam_exposure_spatial_offset_slope_normalized_abs_max);
    merge_option!(seam_exposure_spatial_offset_endpoint_normalized_abs_max);
    merge_option!(seam_exposure_spatial_affine_slope_agreement_ratio_min);
    merge_option!(seam_exposure_spatial_affine_center_offset_delta_normalized_max);
    merge_option!(seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler_min);
    merge_option!(seam_exposure_spatial_2d_gain_accepted);
    merge_option!(seam_exposure_spatial_2d_gain_offset_accepted);
    merge_option!(seam_exposure_spatial_quadratic_gain_accepted);
    merge_option!(seam_exposure_spatial_quadratic_gain_offset_accepted);
    merge_option!(seam_exposure_spatial_2d_distinct_columns_min);
    merge_option!(seam_exposure_spatial_2d_horizontal_slope_agreement_ratio_min);
    merge_option!(seam_exposure_held_out_spatial_2d_improvement_over_best_simpler_min);
    base.seam_blend_required |= overlay.seam_blend_required;
    merge_option!(seam_blend_mode);
    merge_option!(seam_blend_review_required);
    merge_option!(seam_detail_review_required);
    merge_option!(seam_detail_supported_scale_count_min);
    merge_option!(seam_detail_max_symmetric_energy_ratio_max);
    merge_option!(seam_gradient_ratio_max);
    merge_option!(seam_overlap_p95_abs_difference_max);
    merge_option!(base_estimate_source);
    merge_option!(base_confidence_min);
    merge_option!(density_inversion_skipped);
    merge_option!(negative_response_model);
    merge_option!(negative_response_source);
    merge_option!(negative_response_accepted);
    merge_option!(negative_response_review_required);
    merge_option!(negative_response_crosstalk_model);
    merge_option!(negative_response_characteristic_curve_model);
    merge_option!(negative_response_measured_model_id);
    merge_option!(negative_response_measured_confidence_min);
    merge_option!(negative_response_held_out_delta_e00_rms_max);
    merge_option!(negative_response_held_out_max_delta_e00_max);
    merge_option!(negative_response_held_out_improvement_over_unit_slope_min);
    merge_option!(negative_response_density_noise_gain_max);
    merge_option!(negative_response_curve_extrapolated_ratio_max);
    merge_option!(negative_response_signed_headroom_preserved);
    merge_option!(negative_response_curve_interpolation);
    merge_option!(output_color_space);
    merge_option!(render_input_source);
    merge_option!(render_input_reason_contains);
    merge_option!(mapping_strategy);
    merge_option!(selected_mapping_reason_contains);
    merge_option!(selected_candidate);
    merge_option!(selected_candidate_rank);
    merge_option!(calibration_acceptance_status);
    merge_option!(calibration_color_mapping_applied);
    merge_option!(calibration_confidence_min);
    merge_option!(calibration_matrix_condition_number_max);
    merge_vec!(calibration_rejection_details_required);
    merge_option!(candidate_risk);
    merge_option!(tone_color_trust_state);
    merge_option!(neutral_safety_rescue_applied);
    merge_option!(neutral_safety_rescue_preserved_ratio_gain_min);
    merge_option!(neutral_safety_rescue_midtone_saturation_p95_reduction_min);
    merge_option!(neutral_safety_rescue_reason_contains);
    merge_option!(highlight_chroma_compressed_ratio_min);
    merge_option!(highlight_chroma_compressed_ratio_max);
    merge_option!(highlight_neutral_chroma_compressed_ratio_max);
    merge_option!(shadow_chroma_compressed_ratio_max);
    merge_option!(grain_reduction_enabled);
    merge_option!(grain_reduction_applied_ratio_min);
    merge_option!(grain_reduction_structure_excluded_ratio_min);
    merge_option!(grain_reduction_flat_luma_p95_reduction_ratio_min);
    merge_option!(grain_reduction_flat_chroma_p95_reduction_ratio_min);
    merge_option!(grain_detail_review_required);
    merge_option!(grain_detail_decision_supported);
    merge_option!(grain_detail_luminance_probe_count_min);
    merge_option!(grain_detail_chroma_probe_count_min);
    merge_option!(grain_detail_luminance_p10_retention_min);
    merge_option!(grain_detail_chroma_p10_retention_min);
    merge_option!(selected_quality_score_max);
    merge_option!(technical_safety_score_max);
    merge_option!(color_fidelity_score_max);
    merge_option!(memory_color_penalty_max);
    merge_option!(spatial_consistency_penalty_max);
    merge_option!(selected_runner_up_quality_delta_min);
    merge_option!(density_monotonicity_score_min);
    merge_option!(hue_linearity_score_min);
    merge_option!(saturation_preservation_median_ratio_min);
    merge_option!(spatial_neutral_delta_p95_max);
    merge_option!(post_scale_preserved_ratio_min);
    merge_option!(render_luminance_range_p05_p95_min);
    merge_option!(render_review_status);
    merge_option!(render_reviewable);
    merge_option!(tone_output_confidence_status);
    merge_option!(tone_output_review_required);
    merge_option!(tone_output_evidence_confidence_min);
    merge_option!(render_to_mapped_luminance_range_ratio_min);
    merge_option!(post_chroma_compression_clipped_high_ratio_max);
    merge_option!(post_chroma_compression_clipped_low_ratio_max);
    base.reference_patch_evaluation_required |= overlay.reference_patch_evaluation_required;
    merge_option!(reference_patch_count_min);
    merge_option!(reference_patch_hue_family_regression_count_max);
    merge_option!(reference_patch_selected_regresses_image_derived);
    merge_option!(reference_patch_delta_e2000_delta_vs_image_derived_max);
    merge_option!(reference_patch_max_delta_vs_image_derived_max);
    merge_option!(reference_patch_delta_e_max_delta_vs_image_derived_max);
    merge_option!(reference_patch_delta_e2000_max_delta_vs_image_derived_max);
    merge_option!(reference_patch_rms_delta_e_max);
    merge_option!(reference_patch_rms_delta_e2000_max);
    base.debug_artifacts_required |= overlay.debug_artifacts_required;
    merge_vec!(debug_artifact_kinds_required);
    merge_vec!(candidate_acceptance_signatures_required);
    merge_vec!(selection_rejections_required);
}

fn normalized_roll_fixture_labels(
    option_name: &str,
    values: &[String],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut normalized = Vec::new();
    let mut seen = BTreeSet::new();
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(format!("{option_name} cannot contain empty labels").into());
        }
        if !seen.insert(trimmed.to_string()) {
            return Err(format!("{option_name} contains duplicate label: {trimmed}").into());
        }
        normalized.push(trimmed.to_string());
    }
    Ok(normalized)
}

fn roll_fixture_calibration_case(cli: &ValidationCli) -> Option<String> {
    if cli.calibration_profile.is_some() {
        Some("external-profile".to_string())
    } else if cli.calibration_library.is_some() && cli.scanner_profile.is_some() {
        Some("scanner-roll-library".to_string())
    } else if cli.calibration_library.is_some() {
        None
    } else {
        Some("uncalibrated-image-derived".to_string())
    }
}

fn run_roll_suite(cli: &ValidationCli) -> RollSuiteSummary {
    let inventory = run_roll_inventory(cli);
    let output_root = roll_mode_output_dir(cli);
    let mut issues = inventory.issues.clone();
    let render_frames = roll_suite_render_frames(cli, &inventory.frames, &mut issues);
    let roll_base_estimate = if cli.input_mode == InputMode::Negative && cli.base_color.is_none() {
        derive_roll_base_estimate(cli, &inventory.frames)
    } else {
        None
    };
    let base_color_arg = if cli.input_mode == InputMode::Negative {
        cli.base_color.clone().or_else(|| {
            roll_base_estimate
                .as_ref()
                .map(|estimate| format_base_color(estimate.color))
        })
    } else {
        None
    };
    let no_selected_frames = render_frames.is_empty();
    let no_selected_usable_frames = render_frames.iter().all(|frame| !frame.usable);
    let mut frames = render_frames
        .iter()
        .map(|frame| {
            run_roll_suite_entry(
                cli,
                frame,
                &output_root,
                base_color_arg.as_ref(),
                roll_base_estimate.as_ref(),
            )
        })
        .collect::<Vec<_>>();
    if no_selected_frames {
        issues.push("roll_suite:no_selected_frames".to_string());
    }
    if no_selected_usable_frames {
        issues.push("roll_suite:no_usable_frames".to_string());
    }
    issues.extend(
        frames
            .iter()
            .flat_map(|frame| frame.issues.iter().cloned())
            .collect::<Vec<_>>(),
    );

    let passed_count = frames
        .iter()
        .filter(|frame| frame.status == "passed")
        .count();
    let review_required_count = frames
        .iter()
        .filter(|frame| frame.status == "review_required")
        .count();
    let failed_count = frames
        .iter()
        .filter(|frame| frame.status == "failed")
        .count();
    let status = if failed_count > 0 || no_selected_frames || no_selected_usable_frames {
        "failed"
    } else if review_required_count > 0 || !issues.is_empty() {
        "review_required"
    } else {
        "passed"
    }
    .to_string();

    frames.shrink_to_fit();
    let review = summarize_roll_suite_review(&frames, &issues);
    let quality = summarize_roll_suite_quality(&frames);
    RollSuiteSummary {
        status,
        roll_dir: inventory.roll_dir.clone(),
        roll_name: inventory.roll_name.clone(),
        frame_count: frames.len(),
        passed_count,
        review_required_count,
        failed_count,
        output_dir: output_root.display().to_string(),
        roll_base_color: roll_base_estimate.as_ref().map(|estimate| estimate.color),
        roll_base_source: roll_base_estimate
            .as_ref()
            .map(|estimate| estimate.source.clone())
            .or_else(|| {
                (cli.input_mode == InputMode::Negative && cli.base_color.is_some())
                    .then(|| "manual_cli_base_color".to_string())
            }),
        roll_base_confidence: roll_base_estimate
            .as_ref()
            .map(|estimate| estimate.confidence),
        roll_base_frame_count: roll_base_estimate
            .as_ref()
            .map(|estimate| estimate.frame_count)
            .unwrap_or(0),
        roll_base_candidate_count: roll_base_estimate
            .as_ref()
            .map(|estimate| estimate.candidate_count)
            .unwrap_or(0),
        roll_base_rejected_dark_candidate_count: roll_base_estimate
            .as_ref()
            .map(|estimate| estimate.rejected_dark_candidate_count)
            .unwrap_or(0),
        roll_base_high_transmittance_envelope: roll_base_estimate
            .as_ref()
            .map(|estimate| estimate.high_transmittance_envelope),
        roll_base_reason: roll_base_estimate
            .as_ref()
            .map(|estimate| estimate.reason.clone()),
        roll_base_clusters: roll_base_estimate
            .as_ref()
            .map(|estimate| estimate.clusters.clone())
            .unwrap_or_default(),
        review,
        quality,
        comparison: None,
        inventory,
        frames,
        issues,
    }
}

fn roll_suite_render_frames<'a>(
    cli: &ValidationCli,
    frames: &'a [RollFrameInspection],
    issues: &mut Vec<String>,
) -> Vec<&'a RollFrameInspection> {
    if cli.roll_suite_frames.is_empty() {
        return frames.iter().collect();
    }

    let mut selected_indices = BTreeSet::<usize>::new();
    for selector in &cli.roll_suite_frames {
        let selector = selector.trim();
        if selector.is_empty() {
            continue;
        }
        let mut matched = false;
        for (idx, frame) in frames.iter().enumerate() {
            if roll_suite_frame_matches(frame, selector) {
                selected_indices.insert(idx);
                matched = true;
            }
        }
        if !matched {
            issues.push(format!(
                "roll_suite:frame_selector_not_found:{}",
                roll_suite_selector_token(selector)
            ));
        }
    }

    selected_indices
        .into_iter()
        .filter_map(|idx| frames.get(idx))
        .collect()
}

fn roll_suite_frame_matches(frame: &RollFrameInspection, selector: &str) -> bool {
    let selector = roll_suite_selector_token(selector);
    let candidates = [
        frame.name.clone(),
        frame.stem.clone(),
        roll_frame_slug(frame),
        roll_frame_issue_label(frame),
    ];
    candidates
        .iter()
        .any(|candidate| roll_suite_selector_token(candidate) == selector)
}

fn roll_suite_selector_token(selector: &str) -> String {
    let path = Path::new(selector);
    let label = path
        .file_stem()
        .or_else(|| path.file_name())
        .map(|value| value.to_string_lossy())
        .unwrap_or_else(|| selector.into());
    let mut token = String::new();
    let mut last_was_dash = false;
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() {
            token.push(ch.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash {
            token.push('-');
            last_was_dash = true;
        }
    }
    token.trim_matches('-').to_string()
}

fn summarize_roll_suite_review(
    frames: &[RollSuiteEntry],
    issues: &[String],
) -> RollSuiteReviewSummary {
    RollSuiteReviewSummary {
        frame_count: frames.len(),
        render_reviewable_count: frames
            .iter()
            .filter(|frame| frame.render_reviewable == Some(true))
            .count(),
        render_not_reviewable_count: frames
            .iter()
            .filter(|frame| frame.render_reviewable == Some(false))
            .count(),
        render_reviewable_unknown_count: frames
            .iter()
            .filter(|frame| frame.render_reviewable.is_none())
            .count(),
        tone_output_evaluated_count: frames
            .iter()
            .filter(|frame| frame.tone_output_evidence_evaluated == Some(true))
            .count(),
        tone_output_review_required_count: frames
            .iter()
            .filter(|frame| frame.tone_output_review_required == Some(true))
            .count(),
        tone_output_unknown_count: frames
            .iter()
            .filter(|frame| frame.tone_output_review_required.is_none())
            .count(),
        candidate_safe_count: frames
            .iter()
            .filter(|frame| frame.candidate_risk.as_deref() == Some("safe"))
            .count(),
        candidate_review_required_count: frames
            .iter()
            .filter(|frame| {
                frame
                    .candidate_risk
                    .as_deref()
                    .is_some_and(|risk| risk != "safe")
            })
            .count(),
        candidate_unknown_count: frames
            .iter()
            .filter(|frame| frame.candidate_risk.as_deref().is_none_or(str::is_empty))
            .count(),
        tone_color_trusted_count: frames
            .iter()
            .filter(|frame| frame.tone_color_trust_state.as_deref() == Some("trusted"))
            .count(),
        tone_color_review_required_count: frames
            .iter()
            .filter(|frame| {
                frame
                    .tone_color_trust_state
                    .as_deref()
                    .is_some_and(|state| state != "trusted")
            })
            .count(),
        tone_color_unknown_count: frames
            .iter()
            .filter(|frame| {
                frame
                    .tone_color_trust_state
                    .as_deref()
                    .is_none_or(str::is_empty)
            })
            .count(),
        reference_patch_evaluation_present_count: frames
            .iter()
            .filter(|frame| frame.reference_patch_evaluation_present == Some(true))
            .count(),
        reference_patch_evaluation_missing_count: frames
            .iter()
            .filter(|frame| frame.reference_patch_evaluation_present == Some(false))
            .count(),
        reference_patch_evaluation_unknown_count: frames
            .iter()
            .filter(|frame| frame.reference_patch_evaluation_present.is_none())
            .count(),
        render_review_status_counts: roll_suite_value_counts(
            frames
                .iter()
                .map(|frame| frame.render_review_status.as_deref()),
        ),
        tone_output_confidence_status_counts: roll_suite_value_counts(
            frames
                .iter()
                .map(|frame| frame.tone_output_confidence_status.as_deref()),
        ),
        tone_output_review_reason_counts: roll_suite_value_counts(
            frames
                .iter()
                .map(|frame| frame.tone_output_review_reason.as_deref()),
        ),
        candidate_risk_counts: roll_suite_value_counts(
            frames.iter().map(|frame| frame.candidate_risk.as_deref()),
        ),
        tone_color_trust_state_counts: roll_suite_value_counts(
            frames
                .iter()
                .map(|frame| frame.tone_color_trust_state.as_deref()),
        ),
        calibration_status_counts: roll_suite_value_counts(
            frames
                .iter()
                .map(|frame| frame.calibration_status.as_deref()),
        ),
        issue_counts: roll_suite_value_counts(
            issues
                .iter()
                .map(|issue| Some(roll_suite_issue_kind(issue.as_str()))),
        ),
    }
}

fn roll_suite_value_counts<'a>(
    values: impl Iterator<Item = Option<&'a str>>,
) -> Vec<RollSuiteValueCount> {
    let mut counts = BTreeMap::<String, usize>::new();
    for value in values {
        let value = value
            .filter(|value| !value.is_empty())
            .unwrap_or("unknown")
            .to_string();
        *counts.entry(value).or_insert(0) += 1;
    }
    let mut counts = counts
        .into_iter()
        .map(|(value, count)| RollSuiteValueCount { value, count })
        .collect::<Vec<_>>();
    counts.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));
    counts
}

fn roll_suite_issue_kind(issue: &str) -> &str {
    let mut parts = issue.splitn(3, ':');
    match (parts.next(), parts.next(), parts.next()) {
        (Some("roll_suite"), Some(_frame), Some(kind)) if !kind.is_empty() => kind,
        _ => issue,
    }
}

fn summarize_roll_suite_quality(frames: &[RollSuiteEntry]) -> RollSuiteQualitySummary {
    let midtone_luminance_p50_min =
        min_optional(frames.iter().map(|frame| frame.midtone_luminance_p50));
    let midtone_luminance_p50_max =
        max_optional(frames.iter().map(|frame| frame.midtone_luminance_p50));
    let midtone_luminance_p50_range = match (midtone_luminance_p50_min, midtone_luminance_p50_max) {
        (Some(min), Some(max)) => Some(max - min),
        _ => None,
    };
    let evaluated_grain_frames = frames
        .iter()
        .filter(|frame| frame.grain_detail_evaluated == Some(true))
        .collect::<Vec<_>>();
    let supported_luminance_grain_frames = evaluated_grain_frames
        .iter()
        .copied()
        .filter(|frame| frame.grain_detail_luminance_supported == Some(true))
        .collect::<Vec<_>>();
    let supported_chroma_grain_frames = evaluated_grain_frames
        .iter()
        .copied()
        .filter(|frame| frame.grain_detail_chroma_supported == Some(true))
        .collect::<Vec<_>>();

    RollSuiteQualitySummary {
        frame_count: frames.len(),
        high_frequency_frame_count: frames
            .iter()
            .filter(|frame| {
                frame.high_frequency_luma_residual_p95.is_some()
                    || frame.high_frequency_chroma_residual_p95.is_some()
                    || frame.high_frequency_chroma_to_luma_p95_ratio.is_some()
            })
            .count(),
        high_frequency_luma_residual_p95_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.high_frequency_luma_residual_p95),
        ),
        high_frequency_luma_residual_p95_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.high_frequency_luma_residual_p95),
        ),
        high_frequency_chroma_residual_p95_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.high_frequency_chroma_residual_p95),
        ),
        high_frequency_chroma_residual_p95_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.high_frequency_chroma_residual_p95),
        ),
        high_frequency_chroma_to_luma_p95_ratio_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.high_frequency_chroma_to_luma_p95_ratio),
        ),
        high_frequency_chroma_to_luma_p95_ratio_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.high_frequency_chroma_to_luma_p95_ratio),
        ),
        high_frequency_flat_frame_count: frames
            .iter()
            .filter(|frame| frame.high_frequency_flat_sample_count.unwrap_or(0) > 0)
            .count(),
        high_frequency_flat_sample_ratio_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.high_frequency_flat_sample_ratio),
        ),
        high_frequency_flat_luma_residual_p95_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.high_frequency_flat_luma_residual_p95),
        ),
        high_frequency_flat_luma_residual_p95_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.high_frequency_flat_luma_residual_p95),
        ),
        high_frequency_flat_chroma_residual_p95_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.high_frequency_flat_chroma_residual_p95),
        ),
        high_frequency_flat_chroma_residual_p95_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.high_frequency_flat_chroma_residual_p95),
        ),
        high_frequency_flat_chroma_to_luma_p95_ratio_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.high_frequency_flat_chroma_to_luma_p95_ratio),
        ),
        high_frequency_flat_chroma_to_luma_p95_ratio_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.high_frequency_flat_chroma_to_luma_p95_ratio),
        ),
        noise_reduction_enabled_count: frames
            .iter()
            .filter(|frame| frame.noise_reduction_enabled == Some(true))
            .count(),
        noise_reduction_applied_ratio_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.noise_reduction_applied_ratio),
        ),
        noise_reduction_applied_ratio_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.noise_reduction_applied_ratio),
        ),
        noise_reduction_structure_excluded_ratio_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.noise_reduction_structure_excluded_ratio),
        ),
        noise_reduction_structure_excluded_ratio_min: min_optional(
            frames
                .iter()
                .map(|frame| frame.noise_reduction_structure_excluded_ratio),
        ),
        noise_reduction_texture_limited_ratio_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.noise_reduction_texture_limited_ratio),
        ),
        noise_reduction_saturation_limited_ratio_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.noise_reduction_saturation_limited_ratio),
        ),
        noise_reduction_mean_abs_chroma_delta_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.noise_reduction_mean_abs_chroma_delta),
        ),
        noise_reduction_mean_abs_luma_delta_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.noise_reduction_mean_abs_luma_delta),
        ),
        noise_reduction_max_abs_chroma_delta_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.noise_reduction_max_abs_chroma_delta),
        ),
        noise_reduction_max_abs_luma_delta_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.noise_reduction_max_abs_luma_delta),
        ),
        grain_detail_evaluated_count: evaluated_grain_frames.len(),
        grain_detail_decision_supported_count: evaluated_grain_frames
            .iter()
            .filter(|frame| frame.grain_detail_decision_supported == Some(true))
            .count(),
        grain_detail_review_required_count: evaluated_grain_frames
            .iter()
            .filter(|frame| frame.grain_detail_review_required == Some(true))
            .count(),
        grain_detail_luminance_supported_count: supported_luminance_grain_frames.len(),
        grain_detail_chroma_supported_count: supported_chroma_grain_frames.len(),
        grain_detail_luminance_probe_count_min: min_optional_usize(
            evaluated_grain_frames
                .iter()
                .map(|frame| frame.grain_detail_luminance_probe_count),
        ),
        grain_detail_chroma_probe_count_min: min_optional_usize(
            evaluated_grain_frames
                .iter()
                .map(|frame| frame.grain_detail_chroma_probe_count),
        ),
        grain_detail_luminance_median_retention_mean: mean_optional(
            supported_luminance_grain_frames
                .iter()
                .map(|frame| frame.grain_detail_luminance_median_retention),
        ),
        grain_detail_luminance_median_retention_min: min_optional(
            supported_luminance_grain_frames
                .iter()
                .map(|frame| frame.grain_detail_luminance_median_retention),
        ),
        grain_detail_luminance_p10_retention_mean: mean_optional(
            supported_luminance_grain_frames
                .iter()
                .map(|frame| frame.grain_detail_luminance_p10_retention),
        ),
        grain_detail_luminance_p10_retention_min: min_optional(
            supported_luminance_grain_frames
                .iter()
                .map(|frame| frame.grain_detail_luminance_p10_retention),
        ),
        grain_detail_chroma_median_retention_mean: mean_optional(
            supported_chroma_grain_frames
                .iter()
                .map(|frame| frame.grain_detail_chroma_median_retention),
        ),
        grain_detail_chroma_median_retention_min: min_optional(
            supported_chroma_grain_frames
                .iter()
                .map(|frame| frame.grain_detail_chroma_median_retention),
        ),
        grain_detail_chroma_p10_retention_mean: mean_optional(
            supported_chroma_grain_frames
                .iter()
                .map(|frame| frame.grain_detail_chroma_p10_retention),
        ),
        grain_detail_chroma_p10_retention_min: min_optional(
            supported_chroma_grain_frames
                .iter()
                .map(|frame| frame.grain_detail_chroma_p10_retention),
        ),
        colorspace_post_scale_preserved_ratio_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.colorspace_post_scale_preserved_ratio),
        ),
        colorspace_post_scale_preserved_ratio_min: min_optional(
            frames
                .iter()
                .map(|frame| frame.colorspace_post_scale_preserved_ratio),
        ),
        render_luminance_range_p05_p95_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.render_luminance_range_p05_p95),
        ),
        render_luminance_range_p05_p95_min: min_optional(
            frames
                .iter()
                .map(|frame| frame.render_luminance_range_p05_p95),
        ),
        render_luminance_range_p05_p95_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.render_luminance_range_p05_p95),
        ),
        midtone_luminance_p50_mean: mean_optional(
            frames.iter().map(|frame| frame.midtone_luminance_p50),
        ),
        midtone_luminance_p50_min,
        midtone_luminance_p50_max,
        midtone_luminance_p50_range,
        shadow_saturation_p95_mean: mean_optional(
            frames.iter().map(|frame| frame.shadow_saturation_p95),
        ),
        shadow_saturation_p95_max: max_optional(
            frames.iter().map(|frame| frame.shadow_saturation_p95),
        ),
        shadow_visible_saturation_p95_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.shadow_visible_saturation_p95),
        ),
        shadow_visible_saturation_p95_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.shadow_visible_saturation_p95),
        ),
        midtone_neutral_saturation_p95_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.midtone_neutral_saturation_p95),
        ),
        midtone_neutral_saturation_p95_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.midtone_neutral_saturation_p95),
        ),
        bright_neutral_saturation_p95_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.bright_neutral_saturation_p95),
        ),
        bright_neutral_saturation_p95_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.bright_neutral_saturation_p95),
        ),
        shadow_rgb_balance_delta_mean: mean_optional(
            frames.iter().map(|frame| frame.shadow_rgb_balance_delta),
        ),
        shadow_rgb_balance_delta_max: max_optional(
            frames.iter().map(|frame| frame.shadow_rgb_balance_delta),
        ),
        shadow_visible_rgb_balance_delta_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.shadow_visible_rgb_balance_delta),
        ),
        shadow_visible_rgb_balance_delta_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.shadow_visible_rgb_balance_delta),
        ),
        midtone_rgb_balance_delta_mean: mean_optional(
            frames.iter().map(|frame| frame.midtone_rgb_balance_delta),
        ),
        midtone_rgb_balance_delta_max: max_optional(
            frames.iter().map(|frame| frame.midtone_rgb_balance_delta),
        ),
        midtone_neutral_rgb_balance_delta_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.midtone_neutral_rgb_balance_delta),
        ),
        midtone_neutral_rgb_balance_delta_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.midtone_neutral_rgb_balance_delta),
        ),
        bright_neutral_rgb_balance_delta_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.bright_neutral_rgb_balance_delta),
        ),
        bright_neutral_rgb_balance_delta_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.bright_neutral_rgb_balance_delta),
        ),
        highlight_chroma_compressed_ratio_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.highlight_chroma_compressed_ratio),
        ),
        highlight_chroma_compressed_ratio_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.highlight_chroma_compressed_ratio),
        ),
        shadow_chroma_compressed_ratio_mean: mean_optional(
            frames
                .iter()
                .map(|frame| frame.shadow_chroma_compressed_ratio),
        ),
        shadow_chroma_compressed_ratio_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.shadow_chroma_compressed_ratio),
        ),
        post_chroma_compression_clipped_high_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.post_chroma_compression_clipped_high_max),
        ),
        post_chroma_compression_clipped_low_max: max_optional(
            frames
                .iter()
                .map(|frame| frame.post_chroma_compression_clipped_low_max),
        ),
    }
}

fn mean_optional(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.flatten().filter(|value| value.is_finite()) {
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f64)
}

fn max_optional(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    values
        .flatten()
        .filter(|value| value.is_finite())
        .max_by(|a, b| a.total_cmp(b))
}

fn min_optional(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    values
        .flatten()
        .filter(|value| value.is_finite())
        .min_by(|a, b| a.total_cmp(b))
}

fn min_optional_usize(values: impl Iterator<Item = Option<usize>>) -> Option<usize> {
    values.flatten().min()
}

fn compare_roll_suites(
    baseline_path: String,
    baseline: &serde_json::Value,
    current: &serde_json::Value,
) -> RollSuiteComparison {
    const MIDTONE_MIN_DROP_THRESHOLD: f64 = -0.02;
    const MIDTONE_RANGE_INCREASE_THRESHOLD: f64 = 0.02;
    const CHROMA_MEAN_INCREASE_THRESHOLD: f64 = 0.005;
    const CHROMA_MAX_INCREASE_THRESHOLD: f64 = 0.010;
    const FLAT_CHROMA_MEAN_INCREASE_THRESHOLD: f64 = 0.003;
    const BALANCE_MAX_INCREASE_THRESHOLD: f64 = 0.020;
    const PRESERVED_MIN_DROP_THRESHOLD: f64 = -0.005;
    const CLIP_HIGH_INCREASE_THRESHOLD: f64 = 0.001;
    const RENDER_LUMINANCE_RANGE_DROP_THRESHOLD: f64 = -0.03;
    const FRAME_MIDTONE_DROP_THRESHOLD: f64 = -0.03;
    const FRAME_RENDER_LUMINANCE_RANGE_DROP_THRESHOLD: f64 = -0.05;
    const FRAME_CHROMA_INCREASE_THRESHOLD: f64 = 0.02;
    const FRAME_FLAT_CHROMA_INCREASE_THRESHOLD: f64 = 0.01;
    const FRAME_BALANCE_INCREASE_THRESHOLD: f64 = 0.04;
    const FRAME_PRESERVED_DROP_THRESHOLD: f64 = -0.01;
    const GRAIN_DETAIL_RETENTION_DROP_THRESHOLD: f64 = -0.03;
    const FRAME_GRAIN_DETAIL_RETENTION_DROP_THRESHOLD: f64 = -0.05;
    const FRAME_TONE_OUTPUT_CONFIDENCE_DROP_THRESHOLD: f64 = -0.001;
    const FRAME_TONE_OUTPUT_RANGE_RETENTION_DROP_THRESHOLD: f64 = -0.05;

    let baseline_frames = roll_suite_frames_by_name(baseline);
    let current_frames = roll_suite_frames_by_name(current);
    let baseline_names = baseline_frames.keys().cloned().collect::<BTreeSet<_>>();
    let current_names = current_frames.keys().cloned().collect::<BTreeSet<_>>();
    let baseline_only_frames = baseline_names
        .difference(&current_names)
        .cloned()
        .collect::<Vec<_>>();
    let current_only_frames = current_names
        .difference(&baseline_names)
        .cloned()
        .collect::<Vec<_>>();

    let quality = RollSuiteQualityComparison {
        noise_reduction_enabled_count_delta: roll_suite_quality_isize_delta(
            baseline,
            current,
            "noise_reduction_enabled_count",
        ),
        grain_detail_evaluated_count_delta: roll_suite_quality_isize_delta(
            baseline,
            current,
            "grain_detail_evaluated_count",
        ),
        grain_detail_decision_supported_count_delta: roll_suite_quality_isize_delta(
            baseline,
            current,
            "grain_detail_decision_supported_count",
        ),
        grain_detail_review_required_count_delta: roll_suite_quality_isize_delta(
            baseline,
            current,
            "grain_detail_review_required_count",
        ),
        grain_detail_luminance_supported_count_delta: roll_suite_quality_isize_delta(
            baseline,
            current,
            "grain_detail_luminance_supported_count",
        ),
        grain_detail_chroma_supported_count_delta: roll_suite_quality_isize_delta(
            baseline,
            current,
            "grain_detail_chroma_supported_count",
        ),
        grain_detail_luminance_p10_retention_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "grain_detail_luminance_p10_retention_mean",
        ),
        grain_detail_luminance_p10_retention_min_delta: roll_suite_quality_delta(
            baseline,
            current,
            "grain_detail_luminance_p10_retention_min",
        ),
        grain_detail_chroma_p10_retention_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "grain_detail_chroma_p10_retention_mean",
        ),
        grain_detail_chroma_p10_retention_min_delta: roll_suite_quality_delta(
            baseline,
            current,
            "grain_detail_chroma_p10_retention_min",
        ),
        render_luminance_range_p05_p95_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "render_luminance_range_p05_p95_mean",
        ),
        render_luminance_range_p05_p95_min_delta: roll_suite_quality_delta(
            baseline,
            current,
            "render_luminance_range_p05_p95_min",
        ),
        render_luminance_range_p05_p95_max_delta: roll_suite_quality_delta(
            baseline,
            current,
            "render_luminance_range_p05_p95_max",
        ),
        midtone_luminance_p50_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "midtone_luminance_p50_mean",
        ),
        midtone_luminance_p50_min_delta: roll_suite_quality_delta(
            baseline,
            current,
            "midtone_luminance_p50_min",
        ),
        midtone_luminance_p50_max_delta: roll_suite_quality_delta(
            baseline,
            current,
            "midtone_luminance_p50_max",
        ),
        midtone_luminance_p50_range_delta: roll_suite_quality_delta(
            baseline,
            current,
            "midtone_luminance_p50_range",
        ),
        shadow_saturation_p95_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "shadow_saturation_p95_mean",
        ),
        shadow_saturation_p95_max_delta: roll_suite_quality_delta(
            baseline,
            current,
            "shadow_saturation_p95_max",
        ),
        midtone_neutral_saturation_p95_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "midtone_neutral_saturation_p95_mean",
        ),
        midtone_neutral_saturation_p95_max_delta: roll_suite_quality_delta(
            baseline,
            current,
            "midtone_neutral_saturation_p95_max",
        ),
        bright_neutral_saturation_p95_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "bright_neutral_saturation_p95_mean",
        ),
        bright_neutral_saturation_p95_max_delta: roll_suite_quality_delta(
            baseline,
            current,
            "bright_neutral_saturation_p95_max",
        ),
        midtone_neutral_rgb_balance_delta_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "midtone_neutral_rgb_balance_delta_mean",
        ),
        midtone_neutral_rgb_balance_delta_max_delta: roll_suite_quality_delta(
            baseline,
            current,
            "midtone_neutral_rgb_balance_delta_max",
        ),
        bright_neutral_rgb_balance_delta_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "bright_neutral_rgb_balance_delta_mean",
        ),
        bright_neutral_rgb_balance_delta_max_delta: roll_suite_quality_delta(
            baseline,
            current,
            "bright_neutral_rgb_balance_delta_max",
        ),
        high_frequency_chroma_residual_p95_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "high_frequency_chroma_residual_p95_mean",
        ),
        high_frequency_chroma_residual_p95_max_delta: roll_suite_quality_delta(
            baseline,
            current,
            "high_frequency_chroma_residual_p95_max",
        ),
        high_frequency_flat_chroma_residual_p95_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "high_frequency_flat_chroma_residual_p95_mean",
        ),
        high_frequency_flat_chroma_residual_p95_max_delta: roll_suite_quality_delta(
            baseline,
            current,
            "high_frequency_flat_chroma_residual_p95_max",
        ),
        high_frequency_chroma_to_luma_p95_ratio_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "high_frequency_chroma_to_luma_p95_ratio_mean",
        ),
        high_frequency_flat_chroma_to_luma_p95_ratio_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "high_frequency_flat_chroma_to_luma_p95_ratio_mean",
        ),
        noise_reduction_saturation_limited_ratio_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "noise_reduction_saturation_limited_ratio_mean",
        ),
        noise_reduction_mean_abs_chroma_delta_mean_delta: roll_suite_quality_delta(
            baseline,
            current,
            "noise_reduction_mean_abs_chroma_delta_mean",
        ),
        colorspace_post_scale_preserved_ratio_min_delta: roll_suite_quality_delta(
            baseline,
            current,
            "colorspace_post_scale_preserved_ratio_min",
        ),
        post_chroma_compression_clipped_high_max_delta: roll_suite_quality_delta(
            baseline,
            current,
            "post_chroma_compression_clipped_high_max",
        ),
        post_chroma_compression_clipped_low_max_delta: roll_suite_quality_delta(
            baseline,
            current,
            "post_chroma_compression_clipped_low_max",
        ),
    };

    let mut issues = Vec::new();
    let frame_count_delta =
        roll_suite_isize_delta(baseline, current, "frame_count").unwrap_or_default();
    let review_required_count_delta =
        roll_suite_isize_delta(baseline, current, "review_required_count").unwrap_or_default();
    let failed_count_delta =
        roll_suite_isize_delta(baseline, current, "failed_count").unwrap_or_default();
    let frame_set_changed = frame_count_delta != 0
        || !baseline_only_frames.is_empty()
        || !current_only_frames.is_empty();
    if frame_set_changed {
        issues.push("roll_suite_compare:frame_set_changed".to_string());
    }
    if failed_count_delta > 0 {
        issues.push("roll_suite_compare:failed_count_increased".to_string());
    }
    if review_required_count_delta > 0 {
        issues.push("roll_suite_compare:review_required_count_increased".to_string());
    }
    if !frame_set_changed {
        if quality
            .noise_reduction_enabled_count_delta
            .is_some_and(|delta| delta != 0)
        {
            issues.push("roll_suite_compare:noise_reduction_enabled_count_changed".to_string());
        }
        if quality
            .grain_detail_evaluated_count_delta
            .is_some_and(|delta| delta != 0)
        {
            issues.push("roll_suite_compare:grain_detail_evaluated_count_changed".to_string());
        }
        if quality
            .grain_detail_review_required_count_delta
            .is_some_and(|delta| delta > 0)
        {
            issues.push(
                "roll_suite_compare:grain_detail_review_required_count_increased".to_string(),
            );
        }
        if quality
            .grain_detail_decision_supported_count_delta
            .is_some_and(|delta| delta < 0)
        {
            issues.push(
                "roll_suite_compare:grain_detail_decision_supported_count_dropped".to_string(),
            );
        }
        if quality
            .grain_detail_luminance_supported_count_delta
            .is_some_and(|delta| delta < 0)
        {
            issues.push(
                "roll_suite_compare:grain_detail_luminance_supported_count_dropped".to_string(),
            );
        }
        if quality
            .grain_detail_chroma_supported_count_delta
            .is_some_and(|delta| delta < 0)
        {
            issues
                .push("roll_suite_compare:grain_detail_chroma_supported_count_dropped".to_string());
        }
        if option_lt(
            quality.grain_detail_luminance_p10_retention_mean_delta,
            GRAIN_DETAIL_RETENTION_DROP_THRESHOLD,
        ) {
            issues.push("roll_suite_compare:grain_detail_luminance_p10_mean_dropped".to_string());
        }
        if option_lt(
            quality.grain_detail_luminance_p10_retention_min_delta,
            GRAIN_DETAIL_RETENTION_DROP_THRESHOLD,
        ) {
            issues.push("roll_suite_compare:grain_detail_luminance_p10_min_dropped".to_string());
        }
        if option_lt(
            quality.grain_detail_chroma_p10_retention_mean_delta,
            GRAIN_DETAIL_RETENTION_DROP_THRESHOLD,
        ) {
            issues.push("roll_suite_compare:grain_detail_chroma_p10_mean_dropped".to_string());
        }
        if option_lt(
            quality.grain_detail_chroma_p10_retention_min_delta,
            GRAIN_DETAIL_RETENTION_DROP_THRESHOLD,
        ) {
            issues.push("roll_suite_compare:grain_detail_chroma_p10_min_dropped".to_string());
        }
        if option_lt(
            quality.midtone_luminance_p50_min_delta,
            MIDTONE_MIN_DROP_THRESHOLD,
        ) {
            issues.push("roll_suite_compare:midtone_luminance_p50_min_dropped".to_string());
        }
        if option_gt(
            quality.midtone_luminance_p50_range_delta,
            MIDTONE_RANGE_INCREASE_THRESHOLD,
        ) {
            issues.push("roll_suite_compare:midtone_luminance_p50_range_increased".to_string());
        }
        if option_lt(
            quality.render_luminance_range_p05_p95_mean_delta,
            RENDER_LUMINANCE_RANGE_DROP_THRESHOLD,
        ) {
            issues.push("roll_suite_compare:render_luminance_range_mean_dropped".to_string());
        }
        if option_lt(
            quality.render_luminance_range_p05_p95_min_delta,
            RENDER_LUMINANCE_RANGE_DROP_THRESHOLD,
        ) {
            issues.push("roll_suite_compare:render_luminance_range_min_dropped".to_string());
        }
        if option_gt(
            quality.high_frequency_chroma_residual_p95_mean_delta,
            CHROMA_MEAN_INCREASE_THRESHOLD,
        ) {
            issues.push("roll_suite_compare:chroma_residual_p95_mean_increased".to_string());
        }
        if option_gt(
            quality.high_frequency_chroma_residual_p95_max_delta,
            CHROMA_MAX_INCREASE_THRESHOLD,
        ) {
            issues.push("roll_suite_compare:chroma_residual_p95_max_increased".to_string());
        }
        if option_gt(
            quality.high_frequency_flat_chroma_residual_p95_mean_delta,
            FLAT_CHROMA_MEAN_INCREASE_THRESHOLD,
        ) {
            issues.push("roll_suite_compare:flat_chroma_residual_p95_mean_increased".to_string());
        }
        if option_gt(
            quality.midtone_neutral_rgb_balance_delta_max_delta,
            BALANCE_MAX_INCREASE_THRESHOLD,
        ) {
            issues
                .push("roll_suite_compare:midtone_neutral_balance_delta_max_increased".to_string());
        }
        if option_gt(
            quality.bright_neutral_rgb_balance_delta_max_delta,
            BALANCE_MAX_INCREASE_THRESHOLD,
        ) {
            issues
                .push("roll_suite_compare:bright_neutral_balance_delta_max_increased".to_string());
        }
        if option_lt(
            quality.colorspace_post_scale_preserved_ratio_min_delta,
            PRESERVED_MIN_DROP_THRESHOLD,
        ) {
            issues.push("roll_suite_compare:preserved_gamut_min_dropped".to_string());
        }
        if option_gt(
            quality.post_chroma_compression_clipped_high_max_delta,
            CLIP_HIGH_INCREASE_THRESHOLD,
        ) {
            issues.push("roll_suite_compare:post_tone_high_clipping_increased".to_string());
        }
    }

    let mut frame_comparisons = Vec::new();
    for name in baseline_names.union(&current_names) {
        let baseline_frame = baseline_frames.get(name);
        let current_frame = current_frames.get(name);
        let baseline_status = baseline_frame.and_then(|frame| roll_suite_string(frame, "status"));
        let current_status = current_frame.and_then(|frame| roll_suite_string(frame, "status"));
        let baseline_render_review_status =
            baseline_frame.and_then(|frame| roll_suite_string(frame, "render_review_status"));
        let current_render_review_status =
            current_frame.and_then(|frame| roll_suite_string(frame, "render_review_status"));
        let baseline_render_reviewable =
            baseline_frame.and_then(|frame| roll_suite_bool(frame, "render_reviewable"));
        let current_render_reviewable =
            current_frame.and_then(|frame| roll_suite_bool(frame, "render_reviewable"));
        let baseline_tone_output_confidence_status = baseline_frame
            .and_then(|frame| roll_suite_string(frame, "tone_output_confidence_status"));
        let current_tone_output_confidence_status = current_frame
            .and_then(|frame| roll_suite_string(frame, "tone_output_confidence_status"));
        let baseline_tone_output_review_required =
            baseline_frame.and_then(|frame| roll_suite_bool(frame, "tone_output_review_required"));
        let current_tone_output_review_required =
            current_frame.and_then(|frame| roll_suite_bool(frame, "tone_output_review_required"));
        let baseline_candidate_risk =
            baseline_frame.and_then(|frame| roll_suite_string(frame, "candidate_risk"));
        let current_candidate_risk =
            current_frame.and_then(|frame| roll_suite_string(frame, "candidate_risk"));
        let baseline_grain_detail_review_required =
            baseline_frame.and_then(|frame| roll_suite_bool(frame, "grain_detail_review_required"));
        let current_grain_detail_review_required =
            current_frame.and_then(|frame| roll_suite_bool(frame, "grain_detail_review_required"));
        let baseline_grain_detail_decision_supported = baseline_frame
            .and_then(|frame| roll_suite_bool(frame, "grain_detail_decision_supported"));
        let current_grain_detail_decision_supported = current_frame
            .and_then(|frame| roll_suite_bool(frame, "grain_detail_decision_supported"));
        let baseline_grain_detail_luminance_supported = baseline_frame
            .and_then(|frame| roll_suite_bool(frame, "grain_detail_luminance_supported"));
        let current_grain_detail_luminance_supported = current_frame
            .and_then(|frame| roll_suite_bool(frame, "grain_detail_luminance_supported"));
        let baseline_grain_detail_chroma_supported = baseline_frame
            .and_then(|frame| roll_suite_bool(frame, "grain_detail_chroma_supported"));
        let current_grain_detail_chroma_supported =
            current_frame.and_then(|frame| roll_suite_bool(frame, "grain_detail_chroma_supported"));
        let comparison = RollSuiteFrameComparison {
            name: name.clone(),
            status_changed: baseline_status != current_status,
            baseline_status,
            current_status,
            render_review_status_changed: asserted_option_string_changed(
                baseline_render_review_status.as_ref(),
                current_render_review_status.as_ref(),
            ),
            baseline_render_review_status,
            current_render_review_status,
            render_reviewable_changed: asserted_option_bool_changed(
                baseline_render_reviewable,
                current_render_reviewable,
            ),
            baseline_render_reviewable,
            current_render_reviewable,
            tone_output_confidence_status_changed: asserted_option_string_changed(
                baseline_tone_output_confidence_status.as_ref(),
                current_tone_output_confidence_status.as_ref(),
            ),
            baseline_tone_output_confidence_status,
            current_tone_output_confidence_status,
            tone_output_review_required_changed: asserted_option_bool_changed(
                baseline_tone_output_review_required,
                current_tone_output_review_required,
            ),
            baseline_tone_output_review_required,
            current_tone_output_review_required,
            tone_output_evidence_confidence_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "tone_output_evidence_confidence",
            ),
            tone_output_render_to_mapped_luminance_range_ratio_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "render_to_mapped_luminance_range_ratio",
            ),
            tone_output_maximum_post_tone_high_clip_ratio_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "maximum_post_tone_high_clip_ratio",
            ),
            tone_output_maximum_post_tone_low_clip_ratio_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "maximum_post_tone_low_clip_ratio",
            ),
            candidate_risk_changed: baseline_candidate_risk != current_candidate_risk,
            baseline_candidate_risk,
            current_candidate_risk,
            grain_detail_review_required_changed: option_bool_changed(
                baseline_grain_detail_review_required,
                current_grain_detail_review_required,
            ),
            baseline_grain_detail_review_required,
            current_grain_detail_review_required,
            grain_detail_decision_supported_changed: option_bool_changed(
                baseline_grain_detail_decision_supported,
                current_grain_detail_decision_supported,
            ),
            baseline_grain_detail_decision_supported,
            current_grain_detail_decision_supported,
            grain_detail_luminance_supported_changed: option_bool_changed(
                baseline_grain_detail_luminance_supported,
                current_grain_detail_luminance_supported,
            ),
            baseline_grain_detail_luminance_supported,
            current_grain_detail_luminance_supported,
            grain_detail_chroma_supported_changed: option_bool_changed(
                baseline_grain_detail_chroma_supported,
                current_grain_detail_chroma_supported,
            ),
            baseline_grain_detail_chroma_supported,
            current_grain_detail_chroma_supported,
            grain_detail_luminance_p10_retention_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "grain_detail_luminance_p10_retention",
            ),
            grain_detail_chroma_p10_retention_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "grain_detail_chroma_p10_retention",
            ),
            render_luminance_range_p05_p95_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "render_luminance_range_p05_p95",
            ),
            midtone_luminance_p50_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "midtone_luminance_p50",
            ),
            shadow_saturation_p95_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "shadow_saturation_p95",
            ),
            midtone_neutral_saturation_p95_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "midtone_neutral_saturation_p95",
            ),
            bright_neutral_saturation_p95_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "bright_neutral_saturation_p95",
            ),
            midtone_neutral_rgb_balance_delta_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "midtone_neutral_rgb_balance_delta",
            ),
            bright_neutral_rgb_balance_delta_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "bright_neutral_rgb_balance_delta",
            ),
            high_frequency_chroma_residual_p95_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "high_frequency_chroma_residual_p95",
            ),
            high_frequency_flat_chroma_residual_p95_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "high_frequency_flat_chroma_residual_p95",
            ),
            colorspace_post_scale_preserved_ratio_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "colorspace_post_scale_preserved_ratio",
            ),
            post_chroma_compression_clipped_high_max_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "post_chroma_compression_clipped_high_max",
            ),
            post_chroma_compression_clipped_low_max_delta: roll_suite_frame_delta(
                baseline_frame.copied(),
                current_frame.copied(),
                "post_chroma_compression_clipped_low_max",
            ),
        };

        if comparison.current_status.as_deref() == Some("failed")
            && comparison.baseline_status.as_deref() != Some("failed")
        {
            issues.push(format!(
                "roll_suite_compare:{}:status_regressed_to_failed",
                roll_frame_issue_token(name)
            ));
        }
        let both_frames_present = baseline_frame.is_some() && current_frame.is_some();
        if both_frames_present
            && comparison.baseline_render_reviewable == Some(true)
            && comparison.current_render_reviewable != Some(true)
        {
            issues.push(format!(
                "roll_suite_compare:{}:render_reviewability_lost",
                roll_frame_issue_token(name)
            ));
        }
        if both_frames_present
            && comparison.baseline_tone_output_confidence_status.as_deref()
                == Some("supported_render_tonal_distribution")
            && comparison.current_tone_output_confidence_status.as_deref()
                != Some("supported_render_tonal_distribution")
        {
            issues.push(format!(
                "roll_suite_compare:{}:tone_output_status_regressed",
                roll_frame_issue_token(name)
            ));
        }
        if both_frames_present
            && comparison.baseline_tone_output_review_required == Some(false)
            && comparison.current_tone_output_review_required == Some(true)
        {
            issues.push(format!(
                "roll_suite_compare:{}:tone_output_review_newly_required",
                roll_frame_issue_token(name)
            ));
        }
        let baseline_has_tone_output_evidence =
            comparison.baseline_tone_output_confidence_status.is_some()
                || comparison.baseline_tone_output_review_required.is_some();
        let current_has_tone_output_evidence =
            comparison.current_tone_output_confidence_status.is_some()
                || comparison.current_tone_output_review_required.is_some();
        if both_frames_present
            && baseline_has_tone_output_evidence
            && !current_has_tone_output_evidence
        {
            issues.push(format!(
                "roll_suite_compare:{}:tone_output_evidence_missing",
                roll_frame_issue_token(name)
            ));
        }
        if option_lt(
            comparison.tone_output_evidence_confidence_delta,
            FRAME_TONE_OUTPUT_CONFIDENCE_DROP_THRESHOLD,
        ) {
            issues.push(format!(
                "roll_suite_compare:{}:tone_output_evidence_confidence_dropped",
                roll_frame_issue_token(name)
            ));
        }
        if option_lt(
            comparison.tone_output_render_to_mapped_luminance_range_ratio_delta,
            FRAME_TONE_OUTPUT_RANGE_RETENTION_DROP_THRESHOLD,
        ) {
            issues.push(format!(
                "roll_suite_compare:{}:tone_output_range_retention_dropped",
                roll_frame_issue_token(name)
            ));
        }
        if option_gt(
            comparison.tone_output_maximum_post_tone_high_clip_ratio_delta,
            CLIP_HIGH_INCREASE_THRESHOLD,
        ) {
            issues.push(format!(
                "roll_suite_compare:{}:tone_output_high_clipping_increased",
                roll_frame_issue_token(name)
            ));
        }
        if option_gt(
            comparison.tone_output_maximum_post_tone_low_clip_ratio_delta,
            CLIP_HIGH_INCREASE_THRESHOLD,
        ) {
            issues.push(format!(
                "roll_suite_compare:{}:tone_output_low_clipping_increased",
                roll_frame_issue_token(name)
            ));
        }
        if baseline_frame.is_some()
            && current_frame.is_some()
            && roll_suite_risk_rank(comparison.current_candidate_risk.as_deref())
                > roll_suite_risk_rank(comparison.baseline_candidate_risk.as_deref())
        {
            issues.push(format!(
                "roll_suite_compare:{}:candidate_risk_regressed",
                roll_frame_issue_token(name)
            ));
        }
        if comparison.baseline_grain_detail_review_required == Some(false)
            && comparison.current_grain_detail_review_required == Some(true)
        {
            issues.push(format!(
                "roll_suite_compare:{}:grain_detail_review_newly_required",
                roll_frame_issue_token(name)
            ));
        }
        if comparison.baseline_grain_detail_decision_supported == Some(true)
            && comparison.current_grain_detail_decision_supported == Some(false)
        {
            issues.push(format!(
                "roll_suite_compare:{}:grain_detail_decision_support_lost",
                roll_frame_issue_token(name)
            ));
        }
        if comparison.baseline_grain_detail_luminance_supported == Some(true)
            && comparison.current_grain_detail_luminance_supported == Some(false)
        {
            issues.push(format!(
                "roll_suite_compare:{}:grain_detail_luminance_support_lost",
                roll_frame_issue_token(name)
            ));
        }
        if comparison.baseline_grain_detail_chroma_supported == Some(true)
            && comparison.current_grain_detail_chroma_supported == Some(false)
        {
            issues.push(format!(
                "roll_suite_compare:{}:grain_detail_chroma_support_lost",
                roll_frame_issue_token(name)
            ));
        }
        if option_lt(
            comparison.grain_detail_luminance_p10_retention_delta,
            FRAME_GRAIN_DETAIL_RETENTION_DROP_THRESHOLD,
        ) {
            issues.push(format!(
                "roll_suite_compare:{}:grain_detail_luminance_p10_dropped",
                roll_frame_issue_token(name)
            ));
        }
        if option_lt(
            comparison.grain_detail_chroma_p10_retention_delta,
            FRAME_GRAIN_DETAIL_RETENTION_DROP_THRESHOLD,
        ) {
            issues.push(format!(
                "roll_suite_compare:{}:grain_detail_chroma_p10_dropped",
                roll_frame_issue_token(name)
            ));
        }
        if option_lt(
            comparison.midtone_luminance_p50_delta,
            FRAME_MIDTONE_DROP_THRESHOLD,
        ) {
            issues.push(format!(
                "roll_suite_compare:{}:midtone_luminance_p50_dropped",
                roll_frame_issue_token(name)
            ));
        }
        if option_lt(
            comparison.render_luminance_range_p05_p95_delta,
            FRAME_RENDER_LUMINANCE_RANGE_DROP_THRESHOLD,
        ) {
            issues.push(format!(
                "roll_suite_compare:{}:render_luminance_range_dropped",
                roll_frame_issue_token(name)
            ));
        }
        if option_gt(
            comparison.high_frequency_chroma_residual_p95_delta,
            FRAME_CHROMA_INCREASE_THRESHOLD,
        ) {
            issues.push(format!(
                "roll_suite_compare:{}:chroma_residual_p95_increased",
                roll_frame_issue_token(name)
            ));
        }
        if option_gt(
            comparison.high_frequency_flat_chroma_residual_p95_delta,
            FRAME_FLAT_CHROMA_INCREASE_THRESHOLD,
        ) {
            issues.push(format!(
                "roll_suite_compare:{}:flat_chroma_residual_p95_increased",
                roll_frame_issue_token(name)
            ));
        }
        if option_gt(
            comparison.midtone_neutral_rgb_balance_delta_delta,
            FRAME_BALANCE_INCREASE_THRESHOLD,
        ) || option_gt(
            comparison.bright_neutral_rgb_balance_delta_delta,
            FRAME_BALANCE_INCREASE_THRESHOLD,
        ) {
            issues.push(format!(
                "roll_suite_compare:{}:neutral_balance_delta_increased",
                roll_frame_issue_token(name)
            ));
        }
        if option_lt(
            comparison.colorspace_post_scale_preserved_ratio_delta,
            FRAME_PRESERVED_DROP_THRESHOLD,
        ) {
            issues.push(format!(
                "roll_suite_compare:{}:preserved_gamut_dropped",
                roll_frame_issue_token(name)
            ));
        }
        if option_gt(
            comparison.post_chroma_compression_clipped_high_max_delta,
            CLIP_HIGH_INCREASE_THRESHOLD,
        ) {
            issues.push(format!(
                "roll_suite_compare:{}:post_tone_high_clipping_increased",
                roll_frame_issue_token(name)
            ));
        }

        frame_comparisons.push(comparison);
    }

    RollSuiteComparison {
        baseline_path,
        status: if issues.is_empty() {
            "comparable".to_string()
        } else {
            "review_required".to_string()
        },
        issues,
        frame_count_delta,
        review_required_count_delta,
        failed_count_delta,
        baseline_only_frames,
        current_only_frames,
        quality,
        frames: frame_comparisons,
    }
}

fn roll_suite_frames_by_name(summary: &serde_json::Value) -> BTreeMap<String, &serde_json::Value> {
    summary
        .get("frames")
        .and_then(|frames| frames.as_array())
        .into_iter()
        .flatten()
        .filter_map(|frame| Some((roll_suite_string(frame, "name")?, frame)))
        .collect()
}

fn roll_suite_quality_delta(
    baseline: &serde_json::Value,
    current: &serde_json::Value,
    field: &str,
) -> Option<f64> {
    f64_delta(
        roll_suite_quality_value(baseline, field),
        roll_suite_quality_value(current, field),
    )
}

fn roll_suite_quality_isize_delta(
    baseline: &serde_json::Value,
    current: &serde_json::Value,
    field: &str,
) -> Option<isize> {
    Some(
        current.get("quality")?.get(field)?.as_i64()? as isize
            - baseline.get("quality")?.get(field)?.as_i64()? as isize,
    )
}

fn roll_suite_quality_value(summary: &serde_json::Value, field: &str) -> Option<f64> {
    if let Some(value) = summary
        .get("quality")
        .and_then(|quality| quality.get(field))
        .and_then(|value| value.as_f64())
    {
        return Some(value);
    }

    match field {
        "midtone_luminance_p50_max" => {
            roll_suite_frame_metric_extreme(summary, "midtone_luminance_p50", f64::max)
        }
        "midtone_luminance_p50_range" => {
            let min = roll_suite_frame_metric_extreme(summary, "midtone_luminance_p50", f64::min)?;
            let max = roll_suite_frame_metric_extreme(summary, "midtone_luminance_p50", f64::max)?;
            Some(max - min)
        }
        _ => None,
    }
}

fn roll_suite_frame_metric_extreme(
    summary: &serde_json::Value,
    field: &str,
    combine: fn(f64, f64) -> f64,
) -> Option<f64> {
    let mut result = None::<f64>;
    for value in summary
        .get("frames")
        .and_then(|frames| frames.as_array())
        .into_iter()
        .flatten()
        .filter_map(|frame| frame.get(field).and_then(|value| value.as_f64()))
        .filter(|value| value.is_finite())
    {
        result = Some(match result {
            Some(current) => combine(current, value),
            None => value,
        });
    }
    result
}

fn roll_suite_frame_delta(
    baseline: Option<&serde_json::Value>,
    current: Option<&serde_json::Value>,
    field: &str,
) -> Option<f64> {
    f64_delta(
        baseline?.get(field)?.as_f64(),
        current?.get(field)?.as_f64(),
    )
}

fn roll_suite_isize_delta(
    baseline: &serde_json::Value,
    current: &serde_json::Value,
    field: &str,
) -> Option<isize> {
    Some(current.get(field)?.as_i64()? as isize - baseline.get(field)?.as_i64()? as isize)
}

fn roll_suite_string(value: &serde_json::Value, field: &str) -> Option<String> {
    value.get(field)?.as_str().map(ToString::to_string)
}

fn roll_suite_bool(value: &serde_json::Value, field: &str) -> Option<bool> {
    value.get(field)?.as_bool()
}

fn option_bool_changed(baseline: Option<bool>, current: Option<bool>) -> bool {
    matches!((baseline, current), (Some(baseline), Some(current)) if baseline != current)
}

fn asserted_option_bool_changed(baseline: Option<bool>, current: Option<bool>) -> bool {
    baseline.is_some() && baseline != current
}

fn asserted_option_string_changed(baseline: Option<&String>, current: Option<&String>) -> bool {
    baseline.is_some() && baseline != current
}

fn f64_delta(baseline: Option<f64>, current: Option<f64>) -> Option<f64> {
    Some(current? - baseline?)
}

fn option_gt(value: Option<f64>, threshold: f64) -> bool {
    value.is_some_and(|value| value > threshold)
}

fn option_lt(value: Option<f64>, threshold: f64) -> bool {
    value.is_some_and(|value| value < threshold)
}

fn roll_suite_risk_rank(risk: Option<&str>) -> u8 {
    match risk {
        Some("safe") => 0,
        Some(risk) if risk.starts_with("review_") => 1,
        Some(_) => 2,
        None => 2,
    }
}

fn roll_frame_issue_token(name: &str) -> String {
    name.trim_end_matches(".tif")
        .trim_end_matches(".tiff")
        .to_ascii_lowercase()
        .replace('_', "-")
}

fn max_f64_slice(values: Option<&[f64]>) -> Option<f64> {
    values.and_then(|values| {
        values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .max_by(|a, b| a.total_cmp(b))
    })
}

fn phase_f64_vec_metric(
    report: &PipelineReport,
    phase_name: &str,
    metric: &str,
) -> Option<Vec<f64>> {
    let phase = report
        .phases
        .iter()
        .find(|phase| phase.name == phase_name)?;
    let values = phase.metrics.get(metric)?.as_array()?;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(value.as_f64()?);
    }
    Some(out)
}

fn phase_usize_metric(report: &PipelineReport, phase_name: &str, metric: &str) -> Option<usize> {
    let value = report
        .phases
        .iter()
        .find(|phase| phase.name == phase_name)?
        .metrics
        .get(metric)?
        .as_u64()?;
    usize::try_from(value).ok()
}

fn rgb_balance_delta(values: &[f64]) -> Option<f64> {
    if values.len() < 3 {
        return None;
    }
    let rgb = [values[0], values[1], values[2]];
    if !rgb.iter().all(|value| value.is_finite()) {
        return None;
    }
    let max_value = rgb
        .iter()
        .copied()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);
    let min_value = rgb
        .iter()
        .copied()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);
    Some((max_value - min_value).max(0.0))
}

fn tone_rgb_balance_delta(report: &PipelineReport, metric: &str) -> Option<f64> {
    let values = phase_f64_vec_metric(report, "tone_mapping", metric)?;
    rgb_balance_delta(&values)
}

fn nth_f64_slice(values: Option<&[f64]>, index: usize) -> Option<f64> {
    values
        .and_then(|values| values.get(index))
        .copied()
        .filter(|value| value.is_finite())
}

fn derive_roll_base_estimate(
    cli: &ValidationCli,
    frames: &[RollFrameInspection],
) -> Option<RollBaseEstimate> {
    let mut candidates = Vec::<base_detect::RollBaseCandidate>::new();
    for frame in frames.iter().filter(|frame| frame.usable) {
        let path = PathBuf::from(&frame.path);
        let Ok(loaded) = tiff_io::load_tiff_u16(&path, cli.bit_depth) else {
            continue;
        };
        let prepass_image = roll_base_prepass_image(&loaded.image);
        let cropped = border::remove_borders_with_diagnostics(&prepass_image, 2).cropped;
        let detection = base_detect::detect_film_base(&cropped);
        candidates.push(base_detect::RollBaseCandidate::from_detection(
            frame.stem.clone(),
            &detection,
        ));
    }

    let consensus = base_detect::roll_consensus_base(&candidates)?;

    Some(RollBaseEstimate {
        color: consensus.color,
        source: consensus.source.to_string(),
        confidence: consensus.confidence,
        frame_count: consensus.selected_cluster_frame_count,
        candidate_count: consensus.candidate_count,
        rejected_dark_candidate_count: consensus.rejected_dark_candidate_count,
        high_transmittance_envelope: consensus.high_transmittance_envelope,
        reason: consensus.reason,
        clusters: consensus
            .clusters
            .into_iter()
            .map(roll_base_cluster_summary)
            .collect(),
    })
}

fn roll_base_prepass_image(image: &Array3<u16>) -> Array3<u16> {
    let (height, width, channels) = image.dim();
    let max_dimension = height.max(width);
    let stride = max_dimension.div_ceil(ROLL_BASE_PREPASS_MAX_DIMENSION);
    if stride <= 1 {
        return image.clone();
    }

    let out_height = height.div_ceil(stride);
    let out_width = width.div_ceil(stride);
    let mut sampled = Array3::<u16>::zeros((out_height, out_width, channels));
    for out_y in 0..out_height {
        let y = (out_y * stride).min(height.saturating_sub(1));
        for out_x in 0..out_width {
            let x = (out_x * stride).min(width.saturating_sub(1));
            for channel in 0..channels {
                sampled[[out_y, out_x, channel]] = image[[y, x, channel]];
            }
        }
    }
    sampled
}

fn roll_base_cluster_summary(cluster: base_detect::RollBaseCluster) -> RollBaseClusterSummary {
    RollBaseClusterSummary {
        color: cluster.color,
        frame_count: cluster.frame_count,
        mean_luminance: cluster.mean_luminance,
        max_relative_luminance_spread: cluster.max_relative_luminance_spread,
        source_counts: cluster
            .source_counts
            .into_iter()
            .map(|(source, count)| RollBaseSourceCount { source, count })
            .collect(),
    }
}

fn format_base_color(color: [f64; 3]) -> String {
    format!("{:.6},{:.6},{:.6}", color[0], color[1], color[2])
}

fn roll_suite_render_input(
    input_mode: InputMode,
    render_input: RenderInputMode,
) -> RenderInputMode {
    match (input_mode, render_input) {
        (InputMode::Negative, RenderInputMode::Auto) => RenderInputMode::DirectDensity,
        (_, explicit) => explicit,
    }
}

fn run_roll_suite_pipeline_child(
    pipeline_cli: &PipelineCli,
) -> Result<PipelineReport, Box<dyn std::error::Error>> {
    let report_path = pipeline_cli.output_dir.join("report.json");
    if report_path.exists() {
        std::fs::remove_file(&report_path)?;
    }

    let component1 = pipeline_cli
        .inputs
        .first()
        .ok_or("roll suite pipeline requires at least one input")?;
    let component2 = pipeline_cli.inputs.get(1).unwrap_or(component1);
    let mut command = Command::new(std::env::current_exe()?);
    command
        .arg("--roll-suite-child")
        .arg("--component1")
        .arg(component1)
        .arg("--component2")
        .arg(component2)
        .arg("--output-dir")
        .arg(&pipeline_cli.output_dir)
        .arg("--color-mode")
        .arg(pipeline_cli.color_mode.as_str())
        .arg("--render-input")
        .arg(pipeline_cli.render_input.as_str())
        .arg("--input-mode")
        .arg(pipeline_cli.input_mode.as_str())
        .arg("--render-intent")
        .arg(pipeline_cli.render_intent.as_str())
        .arg("--quality-mode")
        .arg(pipeline_cli.quality_mode.as_str())
        .arg("--technical-white-balance")
        .arg(pipeline_cli.white_balance.technical_white_balance.as_str())
        .arg("--technical-temperature-kelvin")
        .arg(
            pipeline_cli
                .white_balance
                .technical_temperature_kelvin
                .to_string(),
        )
        .arg("--technical-tint")
        .arg(pipeline_cli.white_balance.technical_tint.to_string())
        .arg("--creative-temperature")
        .arg(pipeline_cli.white_balance.creative_temperature.to_string())
        .arg("--creative-tint")
        .arg(pipeline_cli.white_balance.creative_tint.to_string())
        .arg("--deskew")
        .arg(pipeline_cli.geometry.deskew.as_str())
        .arg("--deskew-angle-degrees")
        .arg(pipeline_cli.geometry.deskew_angle_degrees.to_string())
        .arg("--grain-reduction")
        .arg(pipeline_cli.grain.grain_reduction.as_str())
        .arg("--grain-strength")
        .arg(pipeline_cli.grain.grain_strength.to_string())
        .arg("--grain-scale")
        .arg(pipeline_cli.grain.grain_scale.to_string())
        .arg("--transform")
        .arg(&pipeline_cli.transform)
        .arg("--ica-max-iter")
        .arg(pipeline_cli.ica_max_iter.to_string())
        .arg("--ica-tol")
        .arg(pipeline_cli.ica_tol.to_string())
        .arg("--bit-depth")
        .arg(pipeline_cli.bit_depth.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    if let Some(path) = &pipeline_cli.calibration_profile {
        command.arg("--calibration-profile").arg(path);
    }
    if let Some(path) = &pipeline_cli.calibration_library {
        command.arg("--calibration-library").arg(path);
    }
    if let Some(profile) = &pipeline_cli.scanner_profile {
        command.arg("--scanner-profile").arg(profile);
    }
    if let Some(profile) = &pipeline_cli.roll_profile {
        command.arg("--roll-profile").arg(profile);
    }
    if let Some(film_stock) = &pipeline_cli.film_stock {
        command.arg("--film-stock").arg(film_stock);
    }
    if let Some(base_color) = &pipeline_cli.base_color {
        command.arg("--base-color").arg(base_color);
    }
    if let Some(source) = &pipeline_cli.base_color_source {
        command.arg("--base-color-source").arg(source);
    }
    if let Some(confidence) = pipeline_cli.base_color_confidence {
        command
            .arg("--base-color-confidence")
            .arg(format!("{confidence:.17}"));
    }
    if let Some(reason) = &pipeline_cli.base_color_reason {
        command.arg("--base-color-reason").arg(reason);
    }
    if pipeline_cli.write_master {
        command.arg("--write-master");
    }
    if let Some(path) = &pipeline_cli.review_sidecar {
        command.arg("--review-sidecar").arg(path);
    }
    if let Some(path) = &pipeline_cli.write_review_sidecar {
        command.arg("--write-review-sidecar").arg(path);
    }
    if pipeline_cli.debug {
        command.arg("--debug");
    }
    if pipeline_cli.force_stitch {
        command.arg("--force-stitch");
    }
    if pipeline_cli.force_no_stitch {
        command.arg("--force-no-stitch");
    }
    if pipeline_cli.use_opencv {
        command.arg("--use-opencv");
    }

    let output = command.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr.trim();
        let detail = if detail.is_empty() {
            format!("exit status {}", output.status)
        } else {
            format!("exit status {}; stderr: {detail}", output.status)
        };
        return Err(format!("roll-suite child pipeline failed: {detail}").into());
    }

    let contents = std::fs::read_to_string(&report_path)?;
    Ok(serde_json::from_str::<PipelineReport>(&contents)?)
}

fn run_roll_suite_entry(
    cli: &ValidationCli,
    frame: &RollFrameInspection,
    output_root: &Path,
    base_color: Option<&String>,
    roll_base_estimate: Option<&RollBaseEstimate>,
) -> RollSuiteEntry {
    let name = roll_frame_issue_label(frame);
    let output_dir = output_root.join(roll_frame_slug(frame));
    let mut entry = RollSuiteEntry {
        name: frame.name.clone(),
        source_path: frame.path.clone(),
        status: "passed".to_string(),
        output_dir: output_dir.display().to_string(),
        source_width: frame.width,
        source_height: frame.height,
        source_color_type: frame.color_type.clone(),
        source_bits_per_sample: frame.source_bits_per_sample,
        output_path: None,
        output_width: None,
        output_height: None,
        output_color_space: None,
        output_file_icc_profile_matches_report: None,
        delivery_artifacts_intact: None,
        delivery_artifact_issues: Vec::new(),
        stale_render_artifact_count: None,
        report_path: None,
        summary_json_path: None,
        summary_md_path: None,
        stitch_decision: None,
        base_confidence: None,
        raw_base_confidence: None,
        base_estimate_source: None,
        input_base_confidence: None,
        render_review_status: None,
        render_reviewable: None,
        tone_output_evidence_evaluated: None,
        tone_output_evidence_confidence: None,
        tone_output_confidence_status: None,
        tone_output_review_required: None,
        tone_output_review_reason: None,
        confidence_limited_by_tone_output_evidence: None,
        input_luminance_range_p05_p95: None,
        mapped_luminance_range_p05_p95: None,
        render_to_mapped_luminance_range_ratio: None,
        maximum_post_tone_high_clip_ratio: None,
        maximum_post_tone_low_clip_ratio: None,
        positive_input_likely_negative_like: None,
        positive_input_accepted_high_warm_score: None,
        positive_input_orange_mask_score: None,
        positive_input_reason: None,
        base_color_override_applied: None,
        density_confidence: None,
        render_input_source: None,
        render_input_reason: None,
        mapping_strategy: None,
        selected_mapping_reason: None,
        selected_candidate: None,
        selected_candidate_rank: None,
        selected_quality_score: None,
        candidate_risk: None,
        tone_color_trust_state: None,
        colorspace_pre_scale_preserved_ratio: None,
        colorspace_post_scale_preserved_ratio: None,
        render_luminance_p05: None,
        render_luminance_p50: None,
        render_luminance_p95: None,
        render_luminance_range_p05_p95: None,
        midtone_luminance_p50: None,
        shadow_saturation_p95: None,
        shadow_visible_pixel_count: None,
        shadow_visible_saturation_p95: None,
        midtone_neutral_pixel_count: None,
        midtone_neutral_saturation_p95: None,
        bright_neutral_saturation_p95: None,
        shadow_rgb_median: None,
        shadow_visible_rgb_median: None,
        midtone_rgb_median: None,
        midtone_neutral_rgb_median: None,
        bright_neutral_rgb_median: None,
        shadow_rgb_balance_delta: None,
        shadow_visible_rgb_balance_delta: None,
        midtone_rgb_balance_delta: None,
        midtone_neutral_rgb_balance_delta: None,
        bright_neutral_rgb_balance_delta: None,
        highlight_chroma_compressed_ratio: None,
        highlight_neutral_chroma_compressed_ratio: None,
        shadow_chroma_compressed_ratio: None,
        post_chroma_compression_clipped_high_max: None,
        post_chroma_compression_clipped_low_max: None,
        high_frequency_luma_residual_p95: None,
        high_frequency_chroma_residual_p95: None,
        high_frequency_chroma_to_luma_p95_ratio: None,
        high_frequency_flat_sample_count: None,
        high_frequency_flat_sample_ratio: None,
        high_frequency_flat_luma_residual_p95: None,
        high_frequency_flat_chroma_residual_p95: None,
        high_frequency_flat_chroma_to_luma_p95_ratio: None,
        noise_reduction_enabled: None,
        noise_reduction_applied_ratio: None,
        noise_reduction_structure_gate_start: None,
        noise_reduction_structure_gate_end: None,
        noise_reduction_structure_excluded_ratio: None,
        noise_reduction_texture_limited_ratio: None,
        noise_reduction_saturation_limited_ratio: None,
        noise_reduction_mean_abs_chroma_delta: None,
        noise_reduction_max_abs_chroma_delta: None,
        noise_reduction_mean_abs_luma_delta: None,
        noise_reduction_max_abs_luma_delta: None,
        grain_detail_evaluated: None,
        grain_detail_decision_supported: None,
        grain_detail_review_required: None,
        grain_detail_luminance_supported: None,
        grain_detail_chroma_supported: None,
        grain_detail_luminance_probe_count: None,
        grain_detail_chroma_probe_count: None,
        grain_detail_luminance_median_retention: None,
        grain_detail_luminance_p10_retention: None,
        grain_detail_chroma_median_retention: None,
        grain_detail_chroma_p10_retention: None,
        calibration_status: None,
        calibration_source: None,
        calibration_acceptance_status: None,
        calibration_color_mapping_applied: None,
        calibration_confidence: None,
        reference_patch_evaluation_present: None,
        debug_artifact_count: None,
        debug_artifact_invalid_count: None,
        debug_artifact_kinds: Vec::new(),
        issues: Vec::new(),
        error: None,
    };

    if !frame.usable {
        entry.status = "failed".to_string();
        entry
            .issues
            .push(format!("roll_suite:{name}:unusable_frame"));
        entry.error = frame.error.clone();
        return entry;
    }

    let source_path = PathBuf::from(&frame.path);
    let pipeline_cli = PipelineCli {
        inputs: vec![source_path],
        output_dir: output_dir.clone(),
        calibration_profile: cli.calibration_profile.clone(),
        calibration_library: cli.calibration_library.clone(),
        scanner_profile: cli.scanner_profile.clone(),
        roll_profile: cli.roll_profile.clone(),
        film_stock: pipeline_film_stock(
            cli.film_stock.as_ref(),
            None,
            cli.calibration_library.as_ref(),
        ),
        base_color: base_color.cloned(),
        base_color_source: roll_base_estimate.map(|estimate| estimate.source.clone()),
        base_color_confidence: roll_base_estimate.map(|estimate| estimate.confidence),
        base_color_reason: roll_base_estimate.map(|estimate| estimate.reason.clone()),
        color_mode: cli.color_mode,
        render_input: roll_suite_render_input(cli.input_mode, cli.render_input),
        input_mode: cli.input_mode,
        render_intent: cli.render_intent,
        quality_mode: cli.quality_mode,
        white_balance: cli.white_balance,
        geometry: cli.geometry,
        grain: cli.grain,
        write_master: cli.write_master,
        review_sidecar: cli.review_sidecar.clone(),
        write_review_sidecar: cli.write_review_sidecar.clone(),
        debug: cli.debug,
        force_stitch: false,
        force_no_stitch: true,
        transform: cli.transform.clone(),
        ica_max_iter: cli.ica_max_iter,
        ica_tol: cli.ica_tol,
        bit_depth: cli.bit_depth,
        use_opencv: cli.use_opencv,
        require_reviewable: false,
    };

    let report = match run_roll_suite_pipeline_child(&pipeline_cli) {
        Ok(report) => report,
        Err(err) => {
            entry.status = "failed".to_string();
            entry
                .issues
                .push(format!("roll_suite:{name}:pipeline_failed"));
            entry.error = Some(err.to_string());
            return entry;
        }
    };

    let report_path = pipeline_cli.output_dir.join("report.json");
    if let Err(err) = report.save(&report_path) {
        entry.status = "failed".to_string();
        entry
            .issues
            .push(format!("roll_suite:{name}:report_write_failed"));
        entry.error = Some(err.to_string());
        return entry;
    }
    entry.report_path = Some(report_path.display().to_string());

    let summary = summarize_report_with_source(frame.stem.clone(), &report, Some(&report_path));
    entry.issues.extend(
        summary
            .diagnostic_consistency_issues
            .iter()
            .map(|issue| format!("roll_suite:{name}:diagnostic_consistency:{issue}")),
    );
    entry.output_path = summary.render.output_path.clone();
    entry.output_width = summary.render.output_width;
    entry.output_height = summary.render.output_height;
    entry.output_color_space = summary.render.output_color_space.clone();
    entry.output_file_icc_profile_matches_report =
        summary.render.output_file_icc_profile_matches_report;
    entry.delivery_artifact_issues = delivery_artifact_integrity_issues(&summary.render)
        .into_iter()
        .map(str::to_string)
        .collect();
    entry.delivery_artifacts_intact = Some(entry.delivery_artifact_issues.is_empty());
    entry.stale_render_artifact_count = summary.render.stale_render_artifact_count;
    entry.stitch_decision = summary.stitch.decision.clone();
    entry.base_confidence = summary.base_density.base_confidence;
    entry.raw_base_confidence = summary.base_density.raw_base_confidence;
    entry.base_estimate_source = summary.base_density.base_estimate_source.clone();
    entry.input_base_confidence = summary.render.input_base_confidence;
    entry.render_review_status = summary.render.render_review_status.clone();
    entry.render_reviewable = summary.render.render_reviewable;
    entry.tone_output_evidence_evaluated = summary.tone.tone_output_evidence_evaluated;
    entry.tone_output_evidence_confidence = summary.tone.tone_output_evidence_confidence;
    entry.tone_output_confidence_status = summary.tone.tone_output_confidence_status.clone();
    entry.tone_output_review_required = summary.tone.tone_output_review_required;
    entry.tone_output_review_reason = summary.tone.tone_output_review_reason.clone();
    entry.confidence_limited_by_tone_output_evidence =
        summary.tone.confidence_limited_by_tone_output_evidence;
    entry.input_luminance_range_p05_p95 = summary.tone.input_luminance_range_p05_p95;
    entry.mapped_luminance_range_p05_p95 = summary.tone.mapped_luminance_range_p05_p95;
    entry.render_to_mapped_luminance_range_ratio =
        summary.tone.render_to_mapped_luminance_range_ratio;
    entry.maximum_post_tone_high_clip_ratio = summary.tone.maximum_post_tone_high_clip_ratio;
    entry.maximum_post_tone_low_clip_ratio = summary.tone.maximum_post_tone_low_clip_ratio;
    entry.positive_input_likely_negative_like = summary.render.positive_input_likely_negative_like;
    entry.positive_input_accepted_high_warm_score =
        summary.render.positive_input_accepted_high_warm_score;
    entry.positive_input_orange_mask_score = summary.render.positive_input_orange_mask_score;
    entry.positive_input_reason = summary.render.positive_input_reason.clone();
    entry.base_color_override_applied = summary
        .base_density
        .base_estimate_source
        .as_deref()
        .map(|source| matches!(source, "manual_base_color_override" | "roll_consensus_base"));
    entry.density_confidence = summary.base_density.density_confidence;
    entry.render_input_source = summary.colorspace.render_input_source.clone();
    entry.render_input_reason = summary.colorspace.render_input_reason.clone();
    entry.mapping_strategy = summary.colorspace.mapping_strategy.clone();
    entry.selected_mapping_reason = summary.colorspace.selected_mapping_reason.clone();
    entry.selected_candidate = summary.colorspace.selected_candidate.clone();
    entry.selected_candidate_rank = summary.colorspace.selected_candidate_rank;
    entry.selected_quality_score = summary.colorspace.selected_quality_score;
    entry.candidate_risk = summary.colorspace.candidate_risk.clone();
    entry.tone_color_trust_state = summary.colorspace.tone_color_trust_state.clone();
    entry.colorspace_pre_scale_preserved_ratio = summary.colorspace.pre_scale_preserved_ratio;
    entry.colorspace_post_scale_preserved_ratio = summary.colorspace.post_scale_preserved_ratio;
    entry.render_luminance_p05 =
        nth_f64_slice(summary.tone.render_luminance_percentiles.as_deref(), 0);
    entry.render_luminance_p50 =
        nth_f64_slice(summary.tone.render_luminance_percentiles.as_deref(), 1);
    entry.render_luminance_p95 =
        nth_f64_slice(summary.tone.render_luminance_percentiles.as_deref(), 2);
    entry.render_luminance_range_p05_p95 = summary
        .tone
        .render_luminance_range_p05_p95
        .or_else(|| Some(entry.render_luminance_p95? - entry.render_luminance_p05?));
    entry.midtone_luminance_p50 =
        nth_f64_slice(summary.tone.midtone_luminance_percentiles.as_deref(), 1);
    entry.shadow_saturation_p95 = summary.tone.shadow_saturation_p95;
    entry.shadow_rgb_median = summary.tone.shadow_rgb_median.clone();
    entry.shadow_visible_pixel_count =
        phase_usize_metric(&report, "tone_mapping", "shadow_visible_pixel_count");
    if entry.shadow_visible_pixel_count.unwrap_or(0) > 0 {
        entry.shadow_visible_saturation_p95 = summary.tone.shadow_visible_saturation_p95;
        entry.shadow_visible_rgb_median = summary.tone.shadow_visible_rgb_median.clone();
        entry.shadow_visible_rgb_balance_delta =
            tone_rgb_balance_delta(&report, "shadow_visible_rgb_median");
    }
    entry.midtone_neutral_pixel_count =
        phase_usize_metric(&report, "tone_mapping", "midtone_neutral_pixel_count");
    if entry.midtone_neutral_pixel_count.unwrap_or(0) > 0 {
        entry.midtone_neutral_saturation_p95 = summary.tone.midtone_neutral_saturation_p95;
        entry.midtone_neutral_rgb_median = summary.tone.midtone_neutral_rgb_median.clone();
        entry.midtone_neutral_rgb_balance_delta =
            tone_rgb_balance_delta(&report, "midtone_neutral_rgb_median");
    }
    entry.bright_neutral_saturation_p95 = summary.tone.bright_neutral_saturation_p95;
    entry.midtone_rgb_median = summary.tone.midtone_rgb_median.clone();
    entry.bright_neutral_rgb_median = summary.tone.bright_neutral_rgb_median.clone();
    entry.shadow_rgb_balance_delta = tone_rgb_balance_delta(&report, "shadow_rgb_median");
    entry.midtone_rgb_balance_delta = tone_rgb_balance_delta(&report, "midtone_rgb_median");
    entry.bright_neutral_rgb_balance_delta =
        tone_rgb_balance_delta(&report, "bright_neutral_rgb_median");
    entry.highlight_chroma_compressed_ratio = summary.tone.highlight_chroma_compressed_ratio;
    entry.highlight_neutral_chroma_compressed_ratio =
        summary.tone.highlight_neutral_chroma_compressed_ratio;
    entry.shadow_chroma_compressed_ratio = summary.tone.shadow_chroma_compressed_ratio;
    entry.post_chroma_compression_clipped_high_max = max_f64_slice(
        summary
            .tone
            .post_chroma_compression_clipped_high_ratio
            .as_deref(),
    );
    entry.post_chroma_compression_clipped_low_max = max_f64_slice(
        summary
            .tone
            .post_chroma_compression_clipped_low_ratio
            .as_deref(),
    );
    if let Some(grain) = &summary.tone.high_frequency_grain {
        entry.high_frequency_luma_residual_p95 = grain.luma_residual_p95;
        entry.high_frequency_chroma_residual_p95 = grain.chroma_residual_p95;
        entry.high_frequency_chroma_to_luma_p95_ratio = grain.chroma_to_luma_p95_ratio;
        entry.high_frequency_flat_sample_count = grain.flat_sample_count;
        entry.high_frequency_flat_sample_ratio = grain.flat_sample_ratio;
        entry.high_frequency_flat_luma_residual_p95 = grain.flat_luma_residual_p95;
        entry.high_frequency_flat_chroma_residual_p95 = grain.flat_chroma_residual_p95;
        entry.high_frequency_flat_chroma_to_luma_p95_ratio = grain.flat_chroma_to_luma_p95_ratio;
    }
    entry.noise_reduction_enabled = summary.tone.noise_reduction_enabled;
    entry.noise_reduction_applied_ratio = summary.tone.noise_reduction_applied_ratio;
    entry.noise_reduction_structure_gate_start = summary.tone.noise_reduction_structure_gate_start;
    entry.noise_reduction_structure_gate_end = summary.tone.noise_reduction_structure_gate_end;
    entry.noise_reduction_structure_excluded_ratio =
        summary.tone.noise_reduction_structure_excluded_ratio;
    entry.noise_reduction_texture_limited_ratio =
        summary.tone.noise_reduction_texture_limited_ratio;
    entry.noise_reduction_saturation_limited_ratio =
        summary.tone.noise_reduction_saturation_limited_ratio;
    entry.noise_reduction_mean_abs_chroma_delta =
        summary.tone.noise_reduction_mean_abs_chroma_delta;
    entry.noise_reduction_max_abs_chroma_delta = summary.tone.noise_reduction_max_abs_chroma_delta;
    entry.noise_reduction_mean_abs_luma_delta = summary.tone.noise_reduction_mean_abs_luma_delta;
    entry.noise_reduction_max_abs_luma_delta = summary.tone.noise_reduction_max_abs_luma_delta;
    if let Some(detail) = &summary.tone.grain_detail_retention {
        entry.grain_detail_evaluated = detail.evaluated;
        entry.grain_detail_decision_supported = detail.decision_supported;
        entry.grain_detail_review_required = detail.review_required;
        entry.grain_detail_luminance_supported = detail.luminance_decision_supported;
        entry.grain_detail_chroma_supported = detail.chroma_decision_supported;
        entry.grain_detail_luminance_probe_count = detail.luminance_probe_count;
        entry.grain_detail_chroma_probe_count = detail.chroma_probe_count;
        entry.grain_detail_luminance_median_retention = detail.luminance_median_retention;
        entry.grain_detail_luminance_p10_retention = detail.luminance_p10_retention;
        entry.grain_detail_chroma_median_retention = detail.chroma_median_retention;
        entry.grain_detail_chroma_p10_retention = detail.chroma_p10_retention;
    }
    entry.calibration_status = summary.colorspace.calibration_status.clone();
    entry.calibration_source = summary.colorspace.calibration_source.clone();
    entry.calibration_acceptance_status = summary
        .colorspace
        .calibration_acceptance
        .as_ref()
        .and_then(|acceptance| acceptance.status.clone());
    entry.calibration_color_mapping_applied = summary
        .colorspace
        .calibration_color_mapping_application
        .as_ref()
        .and_then(|application| application.applied);
    entry.calibration_confidence = summary.colorspace.calibration_confidence;
    entry.reference_patch_evaluation_present =
        Some(summary.colorspace.reference_patch_evaluation.is_some());
    let debug_artifact_issues = fixture_suite_debug_artifact_issues(&summary);
    entry.debug_artifact_count = Some(summary.colorspace.debug_artifacts.len());
    entry.debug_artifact_invalid_count = Some(debug_artifact_issues.len());
    entry.debug_artifact_kinds = summary
        .colorspace
        .debug_artifacts
        .iter()
        .map(|artifact| artifact.kind.clone())
        .collect();

    validate_roll_suite_summary(
        &name,
        cli.input_mode,
        positive_roll_suite_requires_calibration(cli),
        &summary,
        &debug_artifact_issues,
        &mut entry,
    );

    let summary_json_path = output_dir.join("summary.json");
    let summary_md_path = output_dir.join("summary.md");
    let json_summary = match serde_json::to_string_pretty(&summary) {
        Ok(json) => json,
        Err(err) => {
            entry.status = "failed".to_string();
            entry
                .issues
                .push(format!("roll_suite:{name}:summary_serialize_failed"));
            entry.error = Some(err.to_string());
            return entry;
        }
    };
    if let Err(err) = write_text(&summary_json_path, &json_summary) {
        entry.status = "failed".to_string();
        entry
            .issues
            .push(format!("roll_suite:{name}:summary_json_write_failed"));
        entry.error = Some(err.to_string());
        return entry;
    }
    if let Err(err) = write_text(&summary_md_path, &summary_to_markdown(&summary)) {
        entry.status = "failed".to_string();
        entry
            .issues
            .push(format!("roll_suite:{name}:summary_md_write_failed"));
        entry.error = Some(err.to_string());
        return entry;
    }
    entry.summary_json_path = Some(summary_json_path.display().to_string());
    entry.summary_md_path = Some(summary_md_path.display().to_string());

    if cli.debug {
        if let Err(err) = prune_roll_suite_debug_artifacts(&output_dir) {
            entry
                .issues
                .push(format!("roll_suite:{name}:debug_artifact_prune_failed"));
            entry.error = Some(format!("failed to prune roll-suite debug artifacts: {err}"));
        }
    }

    if entry.status != "failed" && !entry.issues.is_empty() {
        entry.status = "review_required".to_string();
    }
    entry
}

fn prune_roll_suite_debug_artifacts(output_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    const KEEP_TIFFS: &[&str] = &[
        "output.tiff",
        "phase3_positive_passthrough.tiff",
        "phase46_color_candidate_comparison.tiff",
        "phase46_gamut_clipping_map.tiff",
        "phase46_scene_referred_prophoto_float.tiff",
    ];

    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
            continue;
        };
        if !extension.eq_ignore_ascii_case("tif") && !extension.eq_ignore_ascii_case("tiff") {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if KEEP_TIFFS
            .iter()
            .any(|keep| file_name.eq_ignore_ascii_case(keep))
        {
            continue;
        }
        std::fs::remove_file(path)?;
    }
    Ok(())
}

fn validate_roll_suite_summary(
    name: &str,
    input_mode: InputMode,
    positive_calibration_required: bool,
    summary: &ValidationSummary,
    debug_artifact_issues: &[String],
    entry: &mut RollSuiteEntry,
) {
    for issue in delivery_artifact_integrity_issues(&summary.render) {
        entry.status = "failed".to_string();
        entry.issues.push(format!("roll_suite:{name}:{issue}"));
    }
    if !debug_artifact_issues.is_empty() {
        entry.status = "failed".to_string();
        entry.issues.extend(
            debug_artifact_issues
                .iter()
                .map(|issue| format!("roll_suite:{name}:{issue}")),
        );
    }
    append_roll_suite_render_review_issues(
        name,
        summary.render.render_reviewable,
        summary.tone.tone_output_review_required,
        &mut entry.issues,
    );

    if input_mode == InputMode::Positive
        && summary.render.positive_input_likely_negative_like == Some(true)
    {
        entry
            .issues
            .push(format!("roll_suite:{name}:positive_input_negative_like"));
    }

    if input_mode == InputMode::Negative {
        if summary
            .base_density
            .base_confidence
            .map(|confidence| confidence < 0.5)
            .unwrap_or(true)
        {
            entry
                .issues
                .push(format!("roll_suite:{name}:base_confidence_review_required"));
        }
        if summary
            .base_density
            .base_estimate_source
            .as_deref()
            .is_some_and(|source| source == "high_transmittance_fallback")
        {
            entry
                .issues
                .push(format!("roll_suite:{name}:base_estimate_fallback"));
        }
    }
    if input_mode == InputMode::Negative || positive_calibration_required {
        if summary.colorspace.calibration_status.as_deref() != Some("applied") {
            entry
                .issues
                .push(format!("roll_suite:{name}:calibration_not_applied"));
        }
        if summary.colorspace.reference_patch_evaluation.is_none() {
            entry.issues.push(format!(
                "roll_suite:{name}:reference_patch_evaluation_missing"
            ));
        }
    }
    if summary.colorspace.candidate_risk.as_deref() != Some("safe") {
        entry
            .issues
            .push(format!("roll_suite:{name}:candidate_risk_review_required"));
    }
    if summary.colorspace.tone_color_trust_state.as_deref() != Some("trusted") {
        entry.issues.push(format!(
            "roll_suite:{name}:tone_color_trust_review_required"
        ));
    }
    if summary
        .colorspace
        .mapping_strategy
        .as_deref()
        .is_some_and(|strategy| strategy.to_ascii_lowercase().contains("fallback"))
    {
        entry
            .issues
            .push(format!("roll_suite:{name}:mapping_strategy_fallback"));
    }
    if summary
        .colorspace
        .selected_candidate
        .as_deref()
        .is_some_and(|candidate| candidate.to_ascii_lowercase().contains("fallback"))
    {
        entry
            .issues
            .push(format!("roll_suite:{name}:selected_candidate_fallback"));
    }
}

fn append_roll_suite_render_review_issues(
    name: &str,
    render_reviewable: Option<bool>,
    tone_output_review_required: Option<bool>,
    issues: &mut Vec<String>,
) {
    match render_reviewable {
        Some(true) => {}
        Some(false) => issues.push(format!("roll_suite:{name}:render_review_not_supported")),
        None => issues.push(format!("roll_suite:{name}:render_review_evidence_missing")),
    }
    match tone_output_review_required {
        Some(false) => {}
        Some(true) => issues.push(format!("roll_suite:{name}:tone_output_review_required")),
        None => issues.push(format!("roll_suite:{name}:tone_output_evidence_missing")),
    }
}

fn positive_roll_suite_requires_calibration(cli: &ValidationCli) -> bool {
    cli.input_mode == InputMode::Positive
        && (cli.calibration_profile.is_some()
            || cli.calibration_library.is_some()
            || cli.scanner_profile.is_some()
            || cli.roll_profile.is_some()
            || cli.film_stock.is_some())
}

fn inspect_roll_frame(path: &Path, input_mode: InputMode, bit_depth: u8) -> RollFrameInspection {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string());
    let stem = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| name.clone());
    let file_size_bytes = std::fs::metadata(path).ok().map(|metadata| metadata.len());
    let sequence = roll_sequence_parts(&stem);
    let probe = probe_fixture_tiff(path);
    let positive_input_probe = (input_mode == InputMode::Positive && probe.readable)
        .then(|| inspect_positive_roll_input(path, bit_depth))
        .and_then(Result::ok);
    RollFrameInspection {
        name,
        stem,
        path: path.display().to_string(),
        file_size_bytes,
        status: probe.status,
        usable: probe.readable,
        width: probe.width,
        height: probe.height,
        color_type: probe.color_type,
        source_bits_per_sample: probe.source_bits_per_sample,
        source_channel_count: probe.source_channel_count,
        source_has_alpha: probe.source_has_alpha,
        sequence_prefix: sequence.as_ref().map(|(prefix, _, _)| prefix.clone()),
        sequence_number: sequence.as_ref().map(|(_, number, _)| *number),
        sequence_width: sequence.as_ref().map(|(_, _, width)| *width),
        positive_input_probe,
        error: probe.error,
    }
}

fn inspect_positive_roll_input(
    path: &Path,
    bit_depth: u8,
) -> Result<positive_input::PositiveInputInspection, Box<dyn std::error::Error>> {
    let loaded = tiff_io::load_tiff_u16(path, bit_depth)?;
    Ok(positive_input::inspect_u16_image(&loaded.image, bit_depth))
}

fn roll_sequence_parts(stem: &str) -> Option<(String, usize, usize)> {
    let end = stem.len();
    let start = stem
        .char_indices()
        .rev()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .map(|(idx, _)| idx)
        .last()?;
    let digits = &stem[start..end];
    if digits.is_empty() {
        return None;
    }
    let number = digits.parse::<usize>().ok()?;
    Some((stem[..start].to_string(), number, digits.len()))
}

fn roll_sequence_gaps(frames: &[RollFrameInspection]) -> Vec<RollSequenceGap> {
    let mut groups: BTreeMap<(String, usize), Vec<usize>> = BTreeMap::new();
    for frame in frames {
        if let (Some(prefix), Some(number), Some(width)) = (
            frame.sequence_prefix.as_ref(),
            frame.sequence_number,
            frame.sequence_width,
        ) {
            groups
                .entry((prefix.clone(), width))
                .or_default()
                .push(number);
        }
    }

    let mut gaps = Vec::new();
    for ((prefix, width), mut numbers) in groups {
        numbers.sort_unstable();
        numbers.dedup();
        for pair in numbers.windows(2) {
            let previous = pair[0];
            let next = pair[1];
            if next > previous + 1 {
                let start = previous + 1;
                let end = next - 1;
                let missing = (start..=end)
                    .map(|number| format!("{prefix}{}", roll_padded_number(number, width)))
                    .collect::<Vec<_>>();
                gaps.push(RollSequenceGap {
                    prefix: prefix.clone(),
                    width,
                    start,
                    end,
                    count: end - start + 1,
                    missing,
                });
            }
        }
    }
    gaps
}

fn roll_padded_number(number: usize, width: usize) -> String {
    format!("{number:0width$}")
}

fn is_supported_roll_scan_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "dng" | "tif" | "tiff"
            )
        })
}

fn roll_name(roll_dir: &Path) -> String {
    roll_dir
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "roll".to_string())
}

fn roll_mode_output_dir(cli: &ValidationCli) -> PathBuf {
    let default_output_dir = PathBuf::from("output/validation/logan");
    if cli.output_dir == default_output_dir {
        let roll_dir = cli
            .roll_dir
            .as_ref()
            .expect("roll inputs should require --roll-dir");
        PathBuf::from("output/validation").join(roll_name(roll_dir))
    } else {
        cli.output_dir.clone()
    }
}

fn roll_frame_slug(frame: &RollFrameInspection) -> String {
    slug_label(&frame.stem)
}

fn slug_label(label: &str) -> String {
    let mut slug = String::new();
    let mut last_was_dash = false;
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash {
            slug.push('-');
            last_was_dash = true;
        }
    }
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        "frame".to_string()
    } else {
        slug
    }
}

fn roll_frame_issue_label(frame: &RollFrameInspection) -> String {
    roll_frame_slug(frame)
}

fn roll_inventory_to_markdown(summary: &RollInventorySummary) -> String {
    let mut md = String::new();
    md.push_str(&format!("# Roll Inventory: {}\n\n", summary.roll_name));
    md.push_str(&format!("- Status: `{}`\n", summary.status));
    md.push_str(&format!("- Roll dir: `{}`\n", summary.roll_dir));
    md.push_str(&format!(
        "- Frames: {} total, {} usable, {} unreadable\n",
        summary.frame_count, summary.usable_frame_count, summary.unreadable_frame_count
    ));
    md.push_str(&format!(
        "- Sequence gaps: {}\n\n",
        summary.sequence_gap_count
    ));
    if !summary.sequence_gaps.is_empty() {
        md.push_str("## Sequence Gaps\n\n");
        md.push_str("| Prefix | Missing | Count |\n");
        md.push_str("| --- | --- | ---: |\n");
        for gap in &summary.sequence_gaps {
            md.push_str(&format!(
                "| `{}` | `{}` | {} |\n",
                markdown_cell(&gap.prefix),
                markdown_cell(&gap.missing.join(", ")),
                gap.count
            ));
        }
        md.push('\n');
    }
    if !summary.issues.is_empty() {
        md.push_str("## Issues\n\n");
        for issue in &summary.issues {
            md.push_str(&format!("- `{}`\n", markdown_cell(issue)));
        }
        md.push('\n');
    }
    md.push_str("## Frames\n\n");
    let show_positive_probe = summary
        .frames
        .iter()
        .any(|frame| frame.positive_input_probe.is_some());
    if show_positive_probe {
        md.push_str("| Frame | Status | Source | Dimensions | Bits | Size | Positive Input |\n");
        md.push_str("| --- | --- | --- | --- | ---: | ---: | --- |\n");
    } else {
        md.push_str("| Frame | Status | Source | Dimensions | Bits | Size |\n");
        md.push_str("| --- | --- | --- | --- | ---: | ---: |\n");
    }
    for frame in &summary.frames {
        let base_columns = format!(
            "| `{}` | `{}` | `{}` | `{}` | {} | {} |",
            markdown_cell(&frame.name),
            frame.status,
            markdown_cell(frame.color_type.as_deref().unwrap_or("unknown")),
            roll_dimensions_label(frame.width, frame.height),
            frame
                .source_bits_per_sample
                .map(|bits| bits.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            frame
                .file_size_bytes
                .map(|size| size.to_string())
                .unwrap_or_else(|| "n/a".to_string())
        );
        if show_positive_probe {
            md.push_str(&format!(
                "{} `{}` |\n",
                base_columns,
                markdown_cell(&positive_roll_probe_label(
                    frame.positive_input_probe.as_ref()
                ))
            ));
        } else {
            md.push_str(&format!("{base_columns}\n"));
        }
    }
    md
}

fn positive_roll_probe_label(probe: Option<&positive_input::PositiveInputInspection>) -> String {
    match probe {
        Some(probe) if probe.likely_negative_like => {
            format!("negative-like score {:.3}", probe.orange_mask_score)
        }
        Some(probe) if probe.accepted_high_warm_score() => {
            format!("ok warm score {:.3}", probe.orange_mask_score)
        }
        Some(probe) => format!("ok score {:.3}", probe.orange_mask_score),
        None => "n/a".to_string(),
    }
}

fn roll_suite_to_markdown(summary: &RollSuiteSummary) -> String {
    let mut md = String::new();
    md.push_str(&format!("# Roll Suite: {}\n\n", summary.roll_name));
    md.push_str(&format!("- Status: `{}`\n", summary.status));
    md.push_str(&format!("- Roll dir: `{}`\n", summary.roll_dir));
    md.push_str(&format!("- Output dir: `{}`\n", summary.output_dir));
    md.push_str(&format!(
        "- Frames: {} total, {} passed, {} review required, {} failed\n",
        summary.frame_count,
        summary.passed_count,
        summary.review_required_count,
        summary.failed_count
    ));
    md.push_str(&format!(
        "- Inventory status: `{}` with {} sequence gap(s)\n\n",
        summary.inventory.status, summary.inventory.sequence_gap_count
    ));
    if let Some(base_color) = summary.roll_base_color {
        md.push_str(&format!(
            "- Roll base color: `{:.1},{:.1},{:.1}` from `{}` using {} agreeing frame(s), confidence `{}`\n",
            base_color[0],
            base_color[1],
            base_color[2],
            summary.roll_base_source.as_deref().unwrap_or("unknown"),
            summary.roll_base_frame_count,
            optional_f64(summary.roll_base_confidence)
        ));
        md.push_str(&format!(
            "- Roll base candidates: `{}` total, `{}` rejected as dark against envelope\n",
            summary.roll_base_candidate_count, summary.roll_base_rejected_dark_candidate_count
        ));
        if let Some(envelope) = summary.roll_base_high_transmittance_envelope {
            md.push_str(&format!(
                "- Roll high-transmittance envelope: `{:.1},{:.1},{:.1}`\n",
                envelope[0], envelope[1], envelope[2]
            ));
        }
        if let Some(reason) = &summary.roll_base_reason {
            md.push_str(&format!(
                "- Roll base reason: `{}`\n",
                markdown_cell(reason)
            ));
        }
        md.push('\n');
    }
    if !summary.roll_base_clusters.is_empty() {
        md.push_str("## Roll Base Clusters\n\n");
        md.push_str("| Rank | Frames | Luminance | Spread | Color | Sources |\n");
        md.push_str("| ---: | ---: | ---: | ---: | --- | --- |\n");
        for (idx, cluster) in summary.roll_base_clusters.iter().enumerate() {
            let sources = cluster
                .source_counts
                .iter()
                .map(|source| format!("{}:{}", source.source, source.count))
                .collect::<Vec<_>>()
                .join(", ");
            md.push_str(&format!(
                "| {} | {} | {:.1} | {:.3} | `{:.1},{:.1},{:.1}` | `{}` |\n",
                idx + 1,
                cluster.frame_count,
                cluster.mean_luminance,
                cluster.max_relative_luminance_spread,
                cluster.color[0],
                cluster.color[1],
                cluster.color[2],
                markdown_cell(&sources)
            ));
        }
        md.push('\n');
    }
    md.push_str("## Roll Review Audit\n\n");
    md.push_str(&format!(
        "- Final render reviewable/not reviewable/unknown: `{}` / `{}` / `{}`\n",
        summary.review.render_reviewable_count,
        summary.review.render_not_reviewable_count,
        summary.review.render_reviewable_unknown_count
    ));
    md.push_str(&format!(
        "- Final render review status counts: `{}`\n",
        markdown_cell(&roll_suite_counts_cell(
            &summary.review.render_review_status_counts
        ))
    ));
    md.push_str(&format!(
        "- Tone output evaluated/review required/unknown: `{}` / `{}` / `{}`\n",
        summary.review.tone_output_evaluated_count,
        summary.review.tone_output_review_required_count,
        summary.review.tone_output_unknown_count
    ));
    md.push_str(&format!(
        "- Tone output status counts: `{}`\n",
        markdown_cell(&roll_suite_counts_cell(
            &summary.review.tone_output_confidence_status_counts
        ))
    ));
    md.push_str(&format!(
        "- Tone output review reason counts: `{}`\n",
        markdown_cell(&roll_suite_counts_cell(
            &summary.review.tone_output_review_reason_counts
        ))
    ));
    md.push_str(&format!(
        "- Candidate risk safe/review/unknown: `{}` / `{}` / `{}`\n",
        summary.review.candidate_safe_count,
        summary.review.candidate_review_required_count,
        summary.review.candidate_unknown_count
    ));
    md.push_str(&format!(
        "- Candidate risk counts: `{}`\n",
        markdown_cell(&roll_suite_counts_cell(
            &summary.review.candidate_risk_counts
        ))
    ));
    md.push_str(&format!(
        "- Tone color trust trusted/review/unknown: `{}` / `{}` / `{}`\n",
        summary.review.tone_color_trusted_count,
        summary.review.tone_color_review_required_count,
        summary.review.tone_color_unknown_count
    ));
    md.push_str(&format!(
        "- Tone color trust counts: `{}`\n",
        markdown_cell(&roll_suite_counts_cell(
            &summary.review.tone_color_trust_state_counts
        ))
    ));
    md.push_str(&format!(
        "- Calibration status counts: `{}`\n",
        markdown_cell(&roll_suite_counts_cell(
            &summary.review.calibration_status_counts
        ))
    ));
    md.push_str(&format!(
        "- Reference patch evaluation present/missing/unknown: `{}` / `{}` / `{}`\n",
        summary.review.reference_patch_evaluation_present_count,
        summary.review.reference_patch_evaluation_missing_count,
        summary.review.reference_patch_evaluation_unknown_count
    ));
    md.push_str(&format!(
        "- Review issue counts: `{}`\n\n",
        markdown_cell(&roll_suite_counts_cell(&summary.review.issue_counts))
    ));
    md.push_str("## Roll Quality Diagnostics\n\n");
    md.push_str(&format!(
        "- High-frequency residual frames: `{}` of `{}`\n",
        summary.quality.high_frequency_frame_count, summary.quality.frame_count
    ));
    md.push_str(&format!(
        "- Luma residual p95 mean/max: `{}` / `{}`\n",
        optional_f64(summary.quality.high_frequency_luma_residual_p95_mean),
        optional_f64(summary.quality.high_frequency_luma_residual_p95_max)
    ));
    md.push_str(&format!(
        "- Chroma residual p95 mean/max: `{}` / `{}`\n",
        optional_f64(summary.quality.high_frequency_chroma_residual_p95_mean),
        optional_f64(summary.quality.high_frequency_chroma_residual_p95_max)
    ));
    md.push_str(&format!(
        "- Chroma:luma residual p95 ratio mean/max: `{}` / `{}`\n",
        optional_f64(summary.quality.high_frequency_chroma_to_luma_p95_ratio_mean),
        optional_f64(summary.quality.high_frequency_chroma_to_luma_p95_ratio_max)
    ));
    md.push_str(&format!(
        "- Flat residual frames: `{}` of `{}`; sample ratio mean: `{}`\n",
        summary.quality.high_frequency_flat_frame_count,
        summary.quality.frame_count,
        optional_f64(summary.quality.high_frequency_flat_sample_ratio_mean)
    ));
    md.push_str(&format!(
        "- Flat luma/chroma residual p95 mean/max: `{}` / `{}`; `{}` / `{}`\n",
        optional_f64(summary.quality.high_frequency_flat_luma_residual_p95_mean),
        optional_f64(summary.quality.high_frequency_flat_luma_residual_p95_max),
        optional_f64(summary.quality.high_frequency_flat_chroma_residual_p95_mean),
        optional_f64(summary.quality.high_frequency_flat_chroma_residual_p95_max)
    ));
    md.push_str(&format!(
        "- Flat chroma:luma residual p95 ratio mean/max: `{}` / `{}`\n",
        optional_f64(
            summary
                .quality
                .high_frequency_flat_chroma_to_luma_p95_ratio_mean
        ),
        optional_f64(
            summary
                .quality
                .high_frequency_flat_chroma_to_luma_p95_ratio_max
        )
    ));
    md.push_str(&format!(
        "- Denoise enabled frames: `{}`; applied ratio mean/max: `{}` / `{}`\n",
        summary.quality.noise_reduction_enabled_count,
        optional_f64(summary.quality.noise_reduction_applied_ratio_mean),
        optional_f64(summary.quality.noise_reduction_applied_ratio_max)
    ));
    md.push_str(&format!(
        "- Denoise exact structure-excluded ratio mean/min: `{}` / `{}`\n",
        optional_f64(
            summary
                .quality
                .noise_reduction_structure_excluded_ratio_mean
        ),
        optional_f64(summary.quality.noise_reduction_structure_excluded_ratio_min)
    ));
    md.push_str(&format!(
        "- Denoise texture/saturation limited mean: `{}` / `{}`; mean chroma/luma delta: `{}` / `{}`\n",
        optional_f64(summary.quality.noise_reduction_texture_limited_ratio_mean),
        optional_f64(summary.quality.noise_reduction_saturation_limited_ratio_mean),
        optional_f64(summary.quality.noise_reduction_mean_abs_chroma_delta_mean),
        optional_f64(summary.quality.noise_reduction_mean_abs_luma_delta_mean)
    ));
    md.push_str(&format!(
        "- Grain detail evaluated/supported/review frames: `{}` / `{}` / `{}`; luminance/chroma supported: `{}` / `{}`\n",
        summary.quality.grain_detail_evaluated_count,
        summary.quality.grain_detail_decision_supported_count,
        summary.quality.grain_detail_review_required_count,
        summary.quality.grain_detail_luminance_supported_count,
        summary.quality.grain_detail_chroma_supported_count
    ));
    md.push_str(&format!(
        "- Grain detail luminance probes min, median retention mean/min, p10 retention mean/min: `{}` / `{}` / `{}` / `{}` / `{}`\n",
        optional_usize(summary.quality.grain_detail_luminance_probe_count_min),
        optional_f64(
            summary
                .quality
                .grain_detail_luminance_median_retention_mean
        ),
        optional_f64(summary.quality.grain_detail_luminance_median_retention_min),
        optional_f64(summary.quality.grain_detail_luminance_p10_retention_mean),
        optional_f64(summary.quality.grain_detail_luminance_p10_retention_min)
    ));
    md.push_str(&format!(
        "- Grain detail chroma probes min, median retention mean/min, p10 retention mean/min: `{}` / `{}` / `{}` / `{}` / `{}`\n",
        optional_usize(summary.quality.grain_detail_chroma_probe_count_min),
        optional_f64(summary.quality.grain_detail_chroma_median_retention_mean),
        optional_f64(summary.quality.grain_detail_chroma_median_retention_min),
        optional_f64(summary.quality.grain_detail_chroma_p10_retention_mean),
        optional_f64(summary.quality.grain_detail_chroma_p10_retention_min)
    ));
    md.push_str(&format!(
        "- Post-scale preserved ratio mean/min: `{}` / `{}`\n",
        optional_f64(summary.quality.colorspace_post_scale_preserved_ratio_mean),
        optional_f64(summary.quality.colorspace_post_scale_preserved_ratio_min)
    ));
    md.push_str(&format!(
        "- Render luminance p05-p95 range mean/min/max: `{}` / `{}` / `{}`\n",
        optional_f64(summary.quality.render_luminance_range_p05_p95_mean),
        optional_f64(summary.quality.render_luminance_range_p05_p95_min),
        optional_f64(summary.quality.render_luminance_range_p05_p95_max)
    ));
    md.push_str(&format!(
        "- Midtone luminance p50 mean/min/max/range: `{}` / `{}` / `{}` / `{}`\n",
        optional_f64(summary.quality.midtone_luminance_p50_mean),
        optional_f64(summary.quality.midtone_luminance_p50_min),
        optional_f64(summary.quality.midtone_luminance_p50_max),
        optional_f64(summary.quality.midtone_luminance_p50_range)
    ));
    md.push_str(&format!(
        "- Shadow saturation p95 mean/max: `{}` / `{}`; midtone-neutral saturation p95 mean/max: `{}` / `{}`; bright-neutral saturation p95 mean/max: `{}` / `{}`\n",
        optional_f64(summary.quality.shadow_saturation_p95_mean),
        optional_f64(summary.quality.shadow_saturation_p95_max),
        optional_f64(summary.quality.midtone_neutral_saturation_p95_mean),
        optional_f64(summary.quality.midtone_neutral_saturation_p95_max),
        optional_f64(summary.quality.bright_neutral_saturation_p95_mean),
        optional_f64(summary.quality.bright_neutral_saturation_p95_max)
    ));
    md.push_str(&format!(
        "- RGB balance delta mean/max: shadow `{}` / `{}`, midtone `{}` / `{}`, midtone-neutral `{}` / `{}`, bright-neutral `{}` / `{}`\n",
        optional_f64(summary.quality.shadow_rgb_balance_delta_mean),
        optional_f64(summary.quality.shadow_rgb_balance_delta_max),
        optional_f64(summary.quality.midtone_rgb_balance_delta_mean),
        optional_f64(summary.quality.midtone_rgb_balance_delta_max),
        optional_f64(summary.quality.midtone_neutral_rgb_balance_delta_mean),
        optional_f64(summary.quality.midtone_neutral_rgb_balance_delta_max),
        optional_f64(summary.quality.bright_neutral_rgb_balance_delta_mean),
        optional_f64(summary.quality.bright_neutral_rgb_balance_delta_max)
    ));
    md.push_str(&format!(
        "- Highlight chroma compression mean/max: `{}` / `{}`; shadow chroma compression mean/max: `{}` / `{}`\n",
        optional_f64(summary.quality.highlight_chroma_compressed_ratio_mean),
        optional_f64(summary.quality.highlight_chroma_compressed_ratio_max),
        optional_f64(summary.quality.shadow_chroma_compressed_ratio_mean),
        optional_f64(summary.quality.shadow_chroma_compressed_ratio_max)
    ));
    md.push_str(&format!(
        "- Post-tone clipped high/low max: `{}` / `{}`\n\n",
        optional_f64(summary.quality.post_chroma_compression_clipped_high_max),
        optional_f64(summary.quality.post_chroma_compression_clipped_low_max)
    ));
    if let Some(comparison) = &summary.comparison {
        md.push_str("## Roll Suite Comparison\n\n");
        md.push_str(&format!(
            "- Baseline: `{}`\n",
            markdown_cell(&comparison.baseline_path)
        ));
        md.push_str(&format!("- Status: `{}`\n", comparison.status));
        md.push_str(&format!(
            "- Frame/review/failed count deltas: `{:+}` / `{:+}` / `{:+}`\n\n",
            comparison.frame_count_delta,
            comparison.review_required_count_delta,
            comparison.failed_count_delta
        ));
        md.push_str(&format!(
            "- Denoise enabled/evaluated/detail-supported/detail-review/luminance-supported/chroma-supported count deltas: `{}` / `{}` / `{}` / `{}` / `{}` / `{}`\n\n",
            optional_delta_isize(comparison.quality.noise_reduction_enabled_count_delta),
            optional_delta_isize(comparison.quality.grain_detail_evaluated_count_delta),
            optional_delta_isize(
                comparison
                    .quality
                    .grain_detail_decision_supported_count_delta
            ),
            optional_delta_isize(
                comparison
                    .quality
                    .grain_detail_review_required_count_delta
            ),
            optional_delta_isize(
                comparison
                    .quality
                    .grain_detail_luminance_supported_count_delta
            ),
            optional_delta_isize(
                comparison
                    .quality
                    .grain_detail_chroma_supported_count_delta
            )
        ));
        if !comparison.issues.is_empty() {
            md.push_str("### Comparison Issues\n\n");
            for issue in &comparison.issues {
                md.push_str(&format!("- `{}`\n", markdown_cell(issue)));
            }
            md.push('\n');
        }
        if !comparison.baseline_only_frames.is_empty() || !comparison.current_only_frames.is_empty()
        {
            md.push_str(&format!(
                "- Baseline-only frames: `{}`\n",
                markdown_cell(&comparison.baseline_only_frames.join(", "))
            ));
            md.push_str(&format!(
                "- Current-only frames: `{}`\n\n",
                markdown_cell(&comparison.current_only_frames.join(", "))
            ));
        }
        md.push_str("| Metric | Delta |\n");
        md.push_str("| --- | ---: |\n");
        for (label, delta) in [
            (
                "grain luma p10 retention mean",
                comparison
                    .quality
                    .grain_detail_luminance_p10_retention_mean_delta,
            ),
            (
                "grain luma p10 retention min",
                comparison
                    .quality
                    .grain_detail_luminance_p10_retention_min_delta,
            ),
            (
                "grain chroma p10 retention mean",
                comparison
                    .quality
                    .grain_detail_chroma_p10_retention_mean_delta,
            ),
            (
                "grain chroma p10 retention min",
                comparison
                    .quality
                    .grain_detail_chroma_p10_retention_min_delta,
            ),
            (
                "render luminance range mean",
                comparison.quality.render_luminance_range_p05_p95_mean_delta,
            ),
            (
                "render luminance range min",
                comparison.quality.render_luminance_range_p05_p95_min_delta,
            ),
            (
                "render luminance range max",
                comparison.quality.render_luminance_range_p05_p95_max_delta,
            ),
            (
                "midtone p50 mean",
                comparison.quality.midtone_luminance_p50_mean_delta,
            ),
            (
                "midtone p50 min",
                comparison.quality.midtone_luminance_p50_min_delta,
            ),
            (
                "midtone p50 max",
                comparison.quality.midtone_luminance_p50_max_delta,
            ),
            (
                "midtone p50 range",
                comparison.quality.midtone_luminance_p50_range_delta,
            ),
            (
                "shadow saturation p95 mean",
                comparison.quality.shadow_saturation_p95_mean_delta,
            ),
            (
                "shadow saturation p95 max",
                comparison.quality.shadow_saturation_p95_max_delta,
            ),
            (
                "midtone-neutral saturation p95 mean",
                comparison.quality.midtone_neutral_saturation_p95_mean_delta,
            ),
            (
                "bright-neutral saturation p95 mean",
                comparison.quality.bright_neutral_saturation_p95_mean_delta,
            ),
            (
                "midtone-neutral RGB delta max",
                comparison
                    .quality
                    .midtone_neutral_rgb_balance_delta_max_delta,
            ),
            (
                "bright-neutral RGB delta max",
                comparison
                    .quality
                    .bright_neutral_rgb_balance_delta_max_delta,
            ),
            (
                "chroma residual p95 mean",
                comparison
                    .quality
                    .high_frequency_chroma_residual_p95_mean_delta,
            ),
            (
                "chroma residual p95 max",
                comparison
                    .quality
                    .high_frequency_chroma_residual_p95_max_delta,
            ),
            (
                "flat chroma residual p95 mean",
                comparison
                    .quality
                    .high_frequency_flat_chroma_residual_p95_mean_delta,
            ),
            (
                "flat chroma residual p95 max",
                comparison
                    .quality
                    .high_frequency_flat_chroma_residual_p95_max_delta,
            ),
            (
                "preserved gamut min",
                comparison
                    .quality
                    .colorspace_post_scale_preserved_ratio_min_delta,
            ),
            (
                "post-tone high clipping max",
                comparison
                    .quality
                    .post_chroma_compression_clipped_high_max_delta,
            ),
        ] {
            md.push_str(&format!("| {} | {} |\n", label, optional_delta_f64(delta)));
        }
        md.push_str("\n| Frame | Status | Risk | Grain Review | Grain Support (all/L/C) | Grain L p10 d | Grain C p10 d | Render Range d | Midtone p50 d | Shadow Sat d | Mid Neutral Sat d | Bright Neutral Sat d | Chroma p95 d | Flat Chroma p95 d | Preserve d | Clip High d |\n");
        md.push_str(
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
        );
        for frame in &comparison.frames {
            let status = if frame.status_changed {
                format!(
                    "{} -> {}",
                    frame.baseline_status.as_deref().unwrap_or(""),
                    frame.current_status.as_deref().unwrap_or("")
                )
            } else {
                frame.current_status.clone().unwrap_or_default()
            };
            let risk = if frame.candidate_risk_changed {
                format!(
                    "{} -> {}",
                    frame.baseline_candidate_risk.as_deref().unwrap_or(""),
                    frame.current_candidate_risk.as_deref().unwrap_or("")
                )
            } else {
                frame.current_candidate_risk.clone().unwrap_or_default()
            };
            let grain_review = roll_suite_bool_transition(
                frame.baseline_grain_detail_review_required,
                frame.current_grain_detail_review_required,
            );
            let grain_support = [
                roll_suite_bool_transition(
                    frame.baseline_grain_detail_decision_supported,
                    frame.current_grain_detail_decision_supported,
                ),
                roll_suite_bool_transition(
                    frame.baseline_grain_detail_luminance_supported,
                    frame.current_grain_detail_luminance_supported,
                ),
                roll_suite_bool_transition(
                    frame.baseline_grain_detail_chroma_supported,
                    frame.current_grain_detail_chroma_supported,
                ),
            ]
            .join("/");
            md.push_str(&format!(
                "| `{}` | `{}` | `{}` | `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                markdown_cell(&frame.name),
                markdown_cell(&status),
                markdown_cell(&risk),
                markdown_cell(&grain_review),
                markdown_cell(&grain_support),
                optional_delta_f64(frame.grain_detail_luminance_p10_retention_delta),
                optional_delta_f64(frame.grain_detail_chroma_p10_retention_delta),
                optional_delta_f64(frame.render_luminance_range_p05_p95_delta),
                optional_delta_f64(frame.midtone_luminance_p50_delta),
                optional_delta_f64(frame.shadow_saturation_p95_delta),
                optional_delta_f64(frame.midtone_neutral_saturation_p95_delta),
                optional_delta_f64(frame.bright_neutral_saturation_p95_delta),
                optional_delta_f64(frame.high_frequency_chroma_residual_p95_delta),
                optional_delta_f64(frame.high_frequency_flat_chroma_residual_p95_delta),
                optional_delta_f64(frame.colorspace_post_scale_preserved_ratio_delta),
                optional_delta_f64(frame.post_chroma_compression_clipped_high_max_delta)
            ));
        }
        md.push_str("\n### Frame Tone Output Comparison\n\n");
        md.push_str("| Frame | Final Review | Reviewable | Tone Status | Tone Review | Confidence d | Render:Mapped d | Clip High d | Clip Low d |\n");
        md.push_str("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |\n");
        for frame in &comparison.frames {
            let render_status = roll_suite_string_transition(
                frame.baseline_render_review_status.as_deref(),
                frame.current_render_review_status.as_deref(),
                frame.render_review_status_changed,
            );
            let tone_status = roll_suite_string_transition(
                frame.baseline_tone_output_confidence_status.as_deref(),
                frame.current_tone_output_confidence_status.as_deref(),
                frame.tone_output_confidence_status_changed,
            );
            md.push_str(&format!(
                "| `{}` | `{}` | `{}` | `{}` | `{}` | {} | {} | {} | {} |\n",
                markdown_cell(&frame.name),
                markdown_cell(&render_status),
                markdown_cell(&roll_suite_bool_transition(
                    frame.baseline_render_reviewable,
                    frame.current_render_reviewable,
                )),
                markdown_cell(&tone_status),
                markdown_cell(&roll_suite_bool_transition(
                    frame.baseline_tone_output_review_required,
                    frame.current_tone_output_review_required,
                )),
                optional_delta_f64(frame.tone_output_evidence_confidence_delta),
                optional_delta_f64(frame.tone_output_render_to_mapped_luminance_range_ratio_delta),
                optional_delta_f64(frame.tone_output_maximum_post_tone_high_clip_ratio_delta),
                optional_delta_f64(frame.tone_output_maximum_post_tone_low_clip_ratio_delta)
            ));
        }
        md.push('\n');
    }
    if !summary.issues.is_empty() {
        md.push_str("## Issues\n\n");
        for issue in &summary.issues {
            md.push_str(&format!("- `{}`\n", markdown_cell(issue)));
        }
        md.push('\n');
    }
    md.push_str("## Frames\n\n");
    md.push_str("| Frame | Status | Review | Base | Mapping | Candidate | Risk | Trust | Calibration | Issues |\n");
    md.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |\n");
    for frame in &summary.frames {
        md.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} |\n",
            markdown_cell(&frame.name),
            frame.status,
            markdown_cell(frame.render_review_status.as_deref().unwrap_or("unknown")),
            markdown_cell(frame.base_estimate_source.as_deref().unwrap_or("unknown")),
            markdown_cell(frame.mapping_strategy.as_deref().unwrap_or("unknown")),
            markdown_cell(frame.selected_candidate.as_deref().unwrap_or("unknown")),
            markdown_cell(frame.candidate_risk.as_deref().unwrap_or("unknown")),
            markdown_cell(frame.tone_color_trust_state.as_deref().unwrap_or("unknown")),
            markdown_cell(frame.calibration_status.as_deref().unwrap_or("unknown")),
            frame.issues.len()
        ));
    }
    md.push_str("\n## Frame Tone Output Evidence\n\n");
    md.push_str("| Frame | Final Review | Reviewable | Evaluated | Confidence | Tone Status | Tone Review | Reason | Input Range | Mapped Range | Render Range | Render:Mapped | Clip High | Clip Low | Limited |\n");
    md.push_str("| --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n");
    for frame in &summary.frames {
        md.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | {} | `{}` | `{}` | `{}` | {} | {} | {} | {} | {} | {} | `{}` |\n",
            markdown_cell(&frame.name),
            markdown_cell(frame.render_review_status.as_deref().unwrap_or("unknown")),
            optional_bool(frame.render_reviewable),
            optional_bool(frame.tone_output_evidence_evaluated),
            optional_f64(frame.tone_output_evidence_confidence),
            markdown_cell(
                frame
                    .tone_output_confidence_status
                    .as_deref()
                    .unwrap_or("unknown")
            ),
            optional_bool(frame.tone_output_review_required),
            markdown_cell(
                frame
                    .tone_output_review_reason
                    .as_deref()
                    .unwrap_or("unknown")
            ),
            optional_f64(frame.input_luminance_range_p05_p95),
            optional_f64(frame.mapped_luminance_range_p05_p95),
            optional_f64(frame.render_luminance_range_p05_p95),
            optional_f64(frame.render_to_mapped_luminance_range_ratio),
            optional_f64(frame.maximum_post_tone_high_clip_ratio),
            optional_f64(frame.maximum_post_tone_low_clip_ratio),
            optional_bool(frame.confidence_limited_by_tone_output_evidence)
        ));
    }
    md.push_str("\n## Frame Quality Diagnostics\n\n");
    md.push_str("| Frame | Render p05 | Render p50 | Render p95 | Render Range | Midtone p50 | Shadow Sat p95 | Mid Neutral Count | Mid Neutral Sat p95 | Bright Neutral Sat p95 | Shadow RGB | Midtone RGB | Mid Neutral RGB | Bright Neutral RGB | Shadow RGB Delta | Midtone RGB Delta | Mid Neutral RGB Delta | Bright RGB Delta | Luma p95 | Chroma p95 | Chroma:Luma | Flat % | Flat Luma p95 | Flat Chroma p95 | Flat C:L | Denoise | Texture Limit | Saturation Limit | Mean C Delta | Mean L Delta | Preserve | Highlight Comp | Shadow Comp | Clip High | Clip Low |\n");
    md.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for frame in &summary.frames {
        md.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            markdown_cell(&frame.name),
            optional_f64(frame.render_luminance_p05),
            optional_f64(frame.render_luminance_p50),
            optional_f64(frame.render_luminance_p95),
            optional_f64(frame.render_luminance_range_p05_p95),
            optional_f64(frame.midtone_luminance_p50),
            optional_f64(frame.shadow_saturation_p95),
            frame
                .midtone_neutral_pixel_count
                .map(|count| count.to_string())
                .unwrap_or_default(),
            optional_f64(frame.midtone_neutral_saturation_p95),
            optional_f64(frame.bright_neutral_saturation_p95),
            optional_f64_vec(frame.shadow_rgb_median.as_deref()),
            optional_f64_vec(frame.midtone_rgb_median.as_deref()),
            optional_f64_vec(frame.midtone_neutral_rgb_median.as_deref()),
            optional_f64_vec(frame.bright_neutral_rgb_median.as_deref()),
            optional_f64(frame.shadow_rgb_balance_delta),
            optional_f64(frame.midtone_rgb_balance_delta),
            optional_f64(frame.midtone_neutral_rgb_balance_delta),
            optional_f64(frame.bright_neutral_rgb_balance_delta),
            optional_f64(frame.high_frequency_luma_residual_p95),
            optional_f64(frame.high_frequency_chroma_residual_p95),
            optional_f64(frame.high_frequency_chroma_to_luma_p95_ratio),
            optional_f64(frame.high_frequency_flat_sample_ratio),
            optional_f64(frame.high_frequency_flat_luma_residual_p95),
            optional_f64(frame.high_frequency_flat_chroma_residual_p95),
            optional_f64(frame.high_frequency_flat_chroma_to_luma_p95_ratio),
            optional_f64(frame.noise_reduction_applied_ratio),
            optional_f64(frame.noise_reduction_texture_limited_ratio),
            optional_f64(frame.noise_reduction_saturation_limited_ratio),
            optional_f64(frame.noise_reduction_mean_abs_chroma_delta),
            optional_f64(frame.noise_reduction_mean_abs_luma_delta),
            optional_f64(frame.colorspace_post_scale_preserved_ratio),
            optional_f64(frame.highlight_chroma_compressed_ratio),
            optional_f64(frame.shadow_chroma_compressed_ratio),
            optional_f64(frame.post_chroma_compression_clipped_high_max),
            optional_f64(frame.post_chroma_compression_clipped_low_max)
        ));
    }
    md.push_str("\n## Frame Grain Detail Diagnostics\n\n");
    md.push_str("| Frame | Enabled | Evaluated | Supported | Review | L Probes | L Support | L Median | L p10 | C Probes | C Support | C Median | C p10 |\n");
    md.push_str(
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |\n",
    );
    for frame in &summary.frames {
        md.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | {} | `{}` | {} | {} | {} | `{}` | {} | {} |\n",
            markdown_cell(&frame.name),
            optional_bool(frame.noise_reduction_enabled),
            optional_bool(frame.grain_detail_evaluated),
            optional_bool(frame.grain_detail_decision_supported),
            optional_bool(frame.grain_detail_review_required),
            optional_usize(frame.grain_detail_luminance_probe_count),
            optional_bool(frame.grain_detail_luminance_supported),
            optional_f64(frame.grain_detail_luminance_median_retention),
            optional_f64(frame.grain_detail_luminance_p10_retention),
            optional_usize(frame.grain_detail_chroma_probe_count),
            optional_bool(frame.grain_detail_chroma_supported),
            optional_f64(frame.grain_detail_chroma_median_retention),
            optional_f64(frame.grain_detail_chroma_p10_retention)
        ));
    }
    md
}

fn roll_dimensions_label(width: Option<usize>, height: Option<usize>) -> String {
    match (width, height) {
        (Some(width), Some(height)) => format!("{width}x{height}"),
        _ => "n/a".to_string(),
    }
}

fn roll_suite_bool_transition(baseline: Option<bool>, current: Option<bool>) -> String {
    match (baseline, current) {
        (Some(baseline), Some(current)) if baseline != current => {
            format!("{baseline} -> {current}")
        }
        (_, Some(current)) => current.to_string(),
        (Some(baseline), None) => format!("{baseline} -> n/a"),
        (None, None) => String::new(),
    }
}

fn roll_suite_string_transition(
    baseline: Option<&str>,
    current: Option<&str>,
    changed: bool,
) -> String {
    if changed {
        format!(
            "{} -> {}",
            baseline.unwrap_or("n/a"),
            current.unwrap_or("n/a")
        )
    } else {
        current.or(baseline).unwrap_or_default().to_string()
    }
}

fn roll_suite_counts_cell(counts: &[RollSuiteValueCount]) -> String {
    if counts.is_empty() {
        return "none".to_string();
    }
    counts
        .iter()
        .map(|entry| format!("{}:{}", entry.value, entry.count))
        .collect::<Vec<_>>()
        .join(", ")
}

fn markdown_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn run_fixture_suite(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
    requirements: &FixtureCoverageRequirements,
) -> FixtureSuiteSummary {
    let coverage = fixture_coverage_summary(fixtures, requirements, cli.compute_fixture_hashes);
    let coverage_by_fixture = coverage
        .fixtures
        .iter()
        .map(|entry| {
            (
                entry.name.clone(),
                FixtureSuiteCoverageContext {
                    validation_ready: entry.validation_ready,
                    issues: entry.issues.clone(),
                    action_items: entry.action_items.clone(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let selected_fixture_names = cli
        .fixture_suite_fixtures
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let fixtures = fixtures
        .iter()
        .filter(|(name, _)| {
            selected_fixture_names.is_empty() || selected_fixture_names.contains(*name)
        })
        .map(|(name, fixture)| {
            run_fixture_suite_entry(
                cli,
                name,
                fixture,
                coverage_by_fixture.get(name).cloned().unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();
    let mut issues = coverage.issues.clone();
    issues.extend(
        fixtures
            .iter()
            .flat_map(|fixture| fixture.issues.iter().cloned())
            .collect::<Vec<_>>(),
    );
    let passed_count = fixtures
        .iter()
        .filter(|fixture| fixture.status == "passed")
        .count();
    let review_required_count = fixtures
        .iter()
        .filter(|fixture| fixture.status == "review_required")
        .count();
    let failed_count = fixtures
        .iter()
        .filter(|fixture| fixture.status == "failed")
        .count();

    FixtureSuiteSummary {
        status: if failed_count > 0 {
            "failed".to_string()
        } else if review_required_count > 0 || !coverage.issues.is_empty() {
            "review_required".to_string()
        } else {
            "passed".to_string()
        },
        fixture_count: fixtures.len(),
        passed_count,
        review_required_count,
        failed_count,
        coverage,
        fixtures,
        issues,
    }
}

fn run_fixture_suite_entry(
    cli: &ValidationCli,
    name: &str,
    fixture: &FixtureEntry,
    coverage: FixtureSuiteCoverageContext,
) -> FixtureSuiteEntry {
    let output_dir = fixture_suite_output_dir(cli, name, fixture);
    let mut entry = FixtureSuiteEntry {
        name: name.to_string(),
        status: "passed".to_string(),
        component_count: fixture.component_count(),
        coverage_validation_ready: coverage.validation_ready,
        coverage_issues: coverage.issues,
        coverage_action_items: coverage.action_items,
        output_dir: output_dir.display().to_string(),
        output_path: None,
        output_modified_at: None,
        output_width: None,
        output_height: None,
        output_color_space: None,
        expected_output_color_space: None,
        output_file_icc_profile_matches_report: None,
        delivery_artifacts_intact: None,
        delivery_artifact_issues: Vec::new(),
        stale_render_artifact_count: None,
        report_path: None,
        summary_json_path: None,
        summary_md_path: None,
        summary_baseline_path: fixture
            .summary_baseline
            .as_ref()
            .map(|path| path.display().to_string()),
        summary_baseline_status: None,
        summary_baseline_write_status: None,
        summary_baseline_written_path: None,
        deskew_status: None,
        expected_deskew_status: None,
        deskew_applied: None,
        expected_deskew_applied: None,
        deskew_review_required: None,
        expected_deskew_review_required: None,
        deskew_retained_area_ratio: None,
        expected_deskew_retained_area_ratio_min: None,
        geometry_preparation: GeometryPreparationFixtureSuiteEvidence {
            expected_all_deskew_components_applied: fixture
                .expectations
                .deskew_all_components_applied,
            expected_minimum_deskew_retained_area_ratio: fixture
                .expectations
                .deskew_minimum_component_retained_area_ratio_min,
            expected_all_border_crop_components_cropped: fixture
                .expectations
                .border_crop_all_components_cropped,
            expected_minimum_removed_edge_count_per_component: fixture
                .expectations
                .border_crop_minimum_removed_edge_count_per_component_min,
            expected_minimum_border_crop_retained_area_ratio: fixture
                .expectations
                .border_crop_retained_area_ratio_min,
            expected_maximum_border_crop_retained_area_ratio: fixture
                .expectations
                .border_crop_retained_area_ratio_max,
            expected_border_crop_rejected: fixture.expectations.border_crop_rejected,
            expected_deskew_correction_degrees: fixture
                .expectations
                .deskew_correction_degrees_expected,
            expected_deskew_correction_tolerance_degrees: fixture
                .expectations
                .deskew_correction_tolerance_degrees,
            expected_border_crop_components: fixture
                .expectations
                .border_crop_components_expected
                .clone(),
            ..GeometryPreparationFixtureSuiteEvidence::default()
        },
        input_orientation: OrientationFixtureSuiteEvidence {
            expected_components: fixture.expectations.orientation_components_expected.clone(),
            ..OrientationFixtureSuiteEvidence::default()
        },
        stitch_decision: None,
        expected_stitch_decision: None,
        inferred_component_order: Vec::new(),
        expected_inferred_component_order: Vec::new(),
        technical_white_balance_status: None,
        expected_technical_white_balance_status: None,
        technical_white_balance_applied: None,
        expected_technical_white_balance_applied: None,
        technical_white_balance_review_required: None,
        expected_technical_white_balance_review_required: None,
        creative_temperature: None,
        expected_creative_temperature: None,
        creative_tint: None,
        expected_creative_tint: None,
        seam_exposure_model: None,
        expected_seam_exposure_model: None,
        seam_exposure_held_out_validation_passed: None,
        expected_seam_exposure_held_out_validation_passed: None,
        seam_exposure_held_out_improvement_over_gain: None,
        expected_seam_exposure_held_out_improvement_over_gain_min: None,
        seam_exposure_offset_normalized_abs_max: None,
        expected_seam_exposure_offset_normalized_abs_max: None,
        seam_exposure_spatial_slope_abs_max: None,
        expected_seam_exposure_spatial_slope_abs_min: None,
        expected_seam_exposure_spatial_slope_abs_max: None,
        seam_exposure_spatial_slope_agreement_ratio: None,
        expected_seam_exposure_spatial_slope_agreement_ratio_min: None,
        seam_exposure_held_out_spatial_improvement_over_best_constant: None,
        expected_seam_exposure_held_out_spatial_improvement_over_best_constant_min: None,
        seam_exposure_spatial_offset_slope_normalized_abs_max: None,
        expected_seam_exposure_spatial_offset_slope_normalized_abs_min: None,
        expected_seam_exposure_spatial_offset_slope_normalized_abs_max: None,
        seam_exposure_spatial_offset_endpoint_normalized_abs_max: None,
        expected_seam_exposure_spatial_offset_endpoint_normalized_abs_max: None,
        seam_exposure_spatial_affine_slope_agreement_ratio: None,
        expected_seam_exposure_spatial_affine_slope_agreement_ratio_min: None,
        seam_exposure_spatial_affine_center_offset_delta_normalized: None,
        expected_seam_exposure_spatial_affine_center_offset_delta_normalized_max: None,
        seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler: None,
        expected_seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler_min: None,
        seam_blend_mode: None,
        expected_seam_blend_required: false,
        expected_seam_blend_mode: None,
        seam_blend_review_required: None,
        expected_seam_blend_review_required: None,
        seam_detail_review_required: None,
        expected_seam_detail_review_required: None,
        seam_detail_supported_scale_count: None,
        expected_seam_detail_supported_scale_count_min: None,
        seam_detail_max_symmetric_energy_ratio: None,
        expected_seam_detail_max_symmetric_energy_ratio_max: None,
        seam_gradient_ratio: None,
        expected_seam_gradient_ratio_max: None,
        seam_overlap_p95_abs_difference: None,
        expected_seam_overlap_p95_abs_difference_max: None,
        base_estimate_source: None,
        expected_base_estimate_source: None,
        negative_reconstruction: NegativeReconstructionFixtureSuiteEvidence {
            expected_base_confidence_min: fixture.expectations.base_confidence_min,
            expected_density_inversion_skipped: fixture.expectations.density_inversion_skipped,
            expected_response_model: fixture.expectations.negative_response_model.clone(),
            expected_response_source: fixture.expectations.negative_response_source.clone(),
            expected_response_accepted: fixture.expectations.negative_response_accepted,
            expected_response_review_required: fixture
                .expectations
                .negative_response_review_required,
            expected_crosstalk_model: fixture
                .expectations
                .negative_response_crosstalk_model
                .clone(),
            expected_characteristic_curve_model: fixture
                .expectations
                .negative_response_characteristic_curve_model
                .clone(),
            expected_measured_model_id: fixture
                .expectations
                .negative_response_measured_model_id
                .clone(),
            expected_measured_confidence_min: fixture
                .expectations
                .negative_response_measured_confidence_min,
            expected_held_out_delta_e00_rms_max: fixture
                .expectations
                .negative_response_held_out_delta_e00_rms_max,
            expected_held_out_delta_e00_max: fixture
                .expectations
                .negative_response_held_out_max_delta_e00_max,
            expected_held_out_improvement_over_unit_slope_min: fixture
                .expectations
                .negative_response_held_out_improvement_over_unit_slope_min,
            expected_maximum_density_noise_gain: fixture
                .expectations
                .negative_response_density_noise_gain_max,
            expected_curve_extrapolated_any_ratio_max: fixture
                .expectations
                .negative_response_curve_extrapolated_ratio_max,
            expected_signed_headroom_preserved: fixture
                .expectations
                .negative_response_signed_headroom_preserved,
            expected_curve_interpolation: fixture
                .expectations
                .negative_response_curve_interpolation
                .clone(),
            ..NegativeReconstructionFixtureSuiteEvidence::default()
        },
        reference_evidence: fixture.reference_evidence.clone(),
        render_input_source: None,
        expected_render_input_source: None,
        render_input_reason: None,
        expected_render_input_reason_contains: None,
        selected_mapping_reason: None,
        expected_selected_mapping_reason_contains: None,
        expected_calibration_source: None,
        expected_calibration_scanner_profile: None,
        expected_calibration_roll_profile: None,
        expected_calibration_film_stock: None,
        calibration_status: None,
        calibration_source: None,
        calibration_scanner_profile_status: None,
        calibration_scanner_profile_id: None,
        calibration_roll_profile_status: None,
        calibration_roll_profile_id: None,
        calibration_requested_film_stock: None,
        calibration_acceptance_status: None,
        calibration_color_mapping_applied: None,
        expected_calibration_color_mapping_applied: None,
        calibration_confidence: None,
        expected_calibration_confidence_min: None,
        calibration_matrix_condition_number: None,
        expected_calibration_matrix_condition_number_max: None,
        calibration_rejection_details: Vec::new(),
        expected_calibration_rejection_details_required: Vec::new(),
        selected_candidate: None,
        expected_selected_candidate: None,
        selected_candidate_rank: None,
        expected_selected_candidate_rank: None,
        candidate_acceptance_signatures: Vec::new(),
        expected_candidate_acceptance_signatures_required: Vec::new(),
        selection_rejections: Vec::new(),
        expected_selection_rejections_required: Vec::new(),
        selected_quality_score: None,
        expected_selected_quality_score_max: None,
        selected_runner_up_quality_delta: None,
        expected_selected_runner_up_quality_delta_min: None,
        technical_safety_score: None,
        expected_technical_safety_score_max: None,
        color_fidelity_score: None,
        expected_color_fidelity_score_max: None,
        memory_color_penalty: None,
        expected_memory_color_penalty_max: None,
        spatial_consistency_penalty: None,
        expected_spatial_consistency_penalty_max: None,
        density_monotonicity_score: None,
        expected_density_monotonicity_score_min: None,
        hue_linearity_score: None,
        expected_hue_linearity_score_min: None,
        saturation_preservation_median_ratio: None,
        expected_saturation_preservation_median_ratio_min: None,
        spatial_neutral_delta_p95: None,
        expected_spatial_neutral_delta_p95_max: None,
        candidate_risk: None,
        expected_candidate_risk: None,
        tone_color_trust_state: None,
        expected_tone_color_trust_state: None,
        neutral_safety_rescue_applied: None,
        expected_neutral_safety_rescue_applied: None,
        neutral_safety_rescue_preserved_ratio_gain: None,
        expected_neutral_safety_rescue_preserved_ratio_gain_min: None,
        neutral_safety_rescue_midtone_saturation_p95_reduction: None,
        expected_neutral_safety_rescue_midtone_saturation_p95_reduction_min: None,
        neutral_safety_rescue_reason: None,
        expected_neutral_safety_rescue_reason_contains: None,
        highlight_chroma_compressed_ratio: None,
        expected_highlight_chroma_compressed_ratio_min: None,
        expected_highlight_chroma_compressed_ratio_max: None,
        highlight_neutral_chroma_compressed_ratio: None,
        expected_highlight_neutral_chroma_compressed_ratio_max: None,
        shadow_chroma_compressed_ratio: None,
        expected_shadow_chroma_compressed_ratio_max: None,
        effective_grain_reduction: None,
        effective_grain_strength: None,
        effective_grain_scale: None,
        grain_reduction_enabled: None,
        expected_grain_reduction_enabled: None,
        grain_reduction_applied_ratio: None,
        expected_grain_reduction_applied_ratio_min: None,
        grain_reduction_structure_excluded_ratio: None,
        expected_grain_reduction_structure_excluded_ratio_min: None,
        grain_reduction_flat_luma_p95_reduction_ratio: None,
        expected_grain_reduction_flat_luma_p95_reduction_ratio_min: None,
        grain_reduction_flat_chroma_p95_reduction_ratio: None,
        expected_grain_reduction_flat_chroma_p95_reduction_ratio_min: None,
        grain_detail_review_required: None,
        expected_grain_detail_review_required: None,
        grain_detail_decision_supported: None,
        expected_grain_detail_decision_supported: None,
        grain_detail_luminance_probe_count: None,
        expected_grain_detail_luminance_probe_count_min: None,
        grain_detail_chroma_probe_count: None,
        expected_grain_detail_chroma_probe_count_min: None,
        grain_detail_luminance_p10_retention: None,
        expected_grain_detail_luminance_p10_retention_min: None,
        grain_detail_chroma_p10_retention: None,
        expected_grain_detail_chroma_p10_retention_min: None,
        mapping_strategy: None,
        expected_mapping_strategy: None,
        post_scale_preserved_ratio: None,
        expected_post_scale_preserved_ratio_min: None,
        render_luminance_range_p05_p95: None,
        expected_render_luminance_range_p05_p95_min: None,
        render_review_status: None,
        expected_render_review_status: None,
        render_reviewable: None,
        expected_render_reviewable: None,
        tone_output_confidence_status: None,
        expected_tone_output_confidence_status: None,
        tone_output_review_required: None,
        expected_tone_output_review_required: None,
        tone_output_evidence_confidence: None,
        expected_tone_output_evidence_confidence_min: None,
        render_to_mapped_luminance_range_ratio: None,
        expected_render_to_mapped_luminance_range_ratio_min: None,
        post_chroma_compression_clipped_high_ratio_max: None,
        expected_post_chroma_compression_clipped_high_ratio_max: None,
        post_chroma_compression_clipped_low_ratio_max: None,
        expected_post_chroma_compression_clipped_low_ratio_max: None,
        reference_patch_evaluation_present: None,
        reference_patch_patch_count: None,
        reference_patch_selected_rms_delta_e: None,
        reference_patch_selected_rms_delta_e2000: None,
        reference_patch_delta_e2000_delta_vs_image_derived: None,
        reference_patch_max_delta_vs_image_derived: None,
        reference_patch_delta_e_max_delta_vs_image_derived: None,
        reference_patch_delta_e2000_max_delta_vs_image_derived: None,
        reference_patch_selected_regresses_image_derived: None,
        reference_patch_hue_family_regressions: Vec::new(),
        expected_reference_patch_evaluation_required: false,
        expected_reference_patch_count_min: None,
        expected_reference_patch_hue_family_regression_count_max: None,
        expected_reference_patch_selected_regresses_image_derived: None,
        expected_reference_patch_delta_e2000_delta_vs_image_derived_max: None,
        expected_reference_patch_max_delta_vs_image_derived_max: None,
        expected_reference_patch_delta_e_max_delta_vs_image_derived_max: None,
        expected_reference_patch_delta_e2000_max_delta_vs_image_derived_max: None,
        expected_reference_patch_rms_delta_e_max: None,
        expected_reference_patch_rms_delta_e2000_max: None,
        debug_artifact_count: None,
        debug_artifact_invalid_count: None,
        debug_artifact_kinds: Vec::new(),
        expected_debug_artifacts_required: false,
        expected_debug_artifact_kinds_required: Vec::new(),
        issues: Vec::new(),
        error: None,
    };

    let inputs = fixture.pipeline_inputs();
    let missing_components = inputs
        .iter()
        .enumerate()
        .filter(|(_, path)| !path.exists())
        .map(|(index, _)| index + 1)
        .collect::<Vec<_>>();
    for component_number in &missing_components {
        entry.issues.push(format!(
            "fixture_suite:{name}:component{component_number}_missing"
        ));
    }
    if !missing_components.is_empty() {
        entry.status = "failed".to_string();
        entry.error = Some(format!(
            "component file(s) missing: {}",
            missing_components
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
        return entry;
    }
    if !entry.coverage_validation_ready {
        entry.issues.push(format!(
            "fixture_suite:{name}:coverage_not_validation_ready"
        ));
    }

    let calibration_profile = cli
        .calibration_profile
        .clone()
        .or_else(|| fixture.calibration_profile.clone());
    let calibration_library = cli
        .calibration_library
        .clone()
        .or_else(|| fixture.calibration_library.clone());
    let film_stock = pipeline_film_stock(
        cli.film_stock.as_ref(),
        Some(fixture),
        calibration_library.as_ref(),
    );
    let input_mode = if let Some(input_mode) = fixture.input_mode.as_deref() {
        match parse_input_mode_label(input_mode) {
            Some(input_mode) => input_mode,
            None => {
                entry.status = "failed".to_string();
                entry
                    .issues
                    .push(format!("fixture_suite:{name}:input_mode_invalid"));
                entry.error = Some(format!("invalid fixture input_mode `{input_mode}`"));
                return entry;
            }
        }
    } else {
        cli.input_mode
    };
    let bit_depth = fixture.bit_depth.unwrap_or(cli.bit_depth);
    if !matches!(bit_depth, 14 | 16) {
        entry.status = "failed".to_string();
        entry
            .issues
            .push(format!("fixture_suite:{name}:bit_depth_invalid"));
        entry.error = Some(format!("invalid fixture bit_depth `{bit_depth}`"));
        return entry;
    }
    let geometry = match geometry_for_fixture(cli.geometry, Some(fixture)) {
        Ok(geometry) => geometry,
        Err(error) => {
            entry.status = "failed".to_string();
            entry
                .issues
                .push(format!("fixture_suite:{name}:invalid_geometry_settings"));
            entry.error = Some(error.to_string());
            return entry;
        }
    };
    let grain = match grain_for_fixture(cli.grain, Some(fixture)) {
        Ok(grain) => grain,
        Err(error) => {
            entry.status = "failed".to_string();
            entry
                .issues
                .push(format!("fixture_suite:{name}:invalid_grain_settings"));
            entry.error = Some(error.to_string());
            return entry;
        }
    };
    entry.effective_grain_reduction = Some(grain.grain_reduction.as_str().to_string());
    entry.effective_grain_strength = Some(grain.grain_strength);
    entry.effective_grain_scale = Some(grain.grain_scale);
    let force_stitch = cli.force_stitch || fixture.force_stitch;
    let force_no_stitch = cli.force_no_stitch || fixture.force_no_stitch;
    if force_stitch && force_no_stitch {
        entry.status = "failed".to_string();
        entry.issues.push(format!(
            "fixture_suite:{name}:force_stitch_and_force_no_stitch"
        ));
        entry.error = Some(
            "fixture effective settings request both force-stitch and force-no-stitch".to_string(),
        );
        return entry;
    }
    if let Err(err) = validate_input_mode_options(input_mode, cli.render_input) {
        entry.status = "failed".to_string();
        entry.issues.push(format!(
            "fixture_suite:{name}:input_mode_render_input_invalid"
        ));
        entry.error = Some(err.to_string());
        return entry;
    }

    let pipeline_cli = PipelineCli {
        inputs,
        output_dir: output_dir.clone(),
        calibration_profile,
        calibration_library,
        scanner_profile: cli
            .scanner_profile
            .clone()
            .or_else(|| fixture.scanner_profile.clone()),
        roll_profile: cli
            .roll_profile
            .clone()
            .or_else(|| fixture.roll_profile.clone()),
        film_stock,
        base_color: cli.base_color.clone(),
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: cli.color_mode,
        render_input: cli.render_input,
        input_mode,
        render_intent: cli.render_intent,
        quality_mode: cli.quality_mode,
        white_balance: cli.white_balance,
        geometry,
        grain,
        write_master: cli.write_master,
        review_sidecar: cli.review_sidecar.clone(),
        write_review_sidecar: cli.write_review_sidecar.clone(),
        debug: cli.debug,
        force_stitch,
        force_no_stitch,
        transform: cli.transform.clone(),
        ica_max_iter: cli.ica_max_iter,
        ica_tol: cli.ica_tol,
        bit_depth,
        use_opencv: cli.use_opencv,
        require_reviewable: false,
    };
    entry.expected_calibration_source =
        expected_calibration_source(&pipeline_cli).map(str::to_string);
    entry.expected_calibration_scanner_profile = pipeline_cli.scanner_profile.clone();
    entry.expected_calibration_roll_profile = pipeline_cli.roll_profile.clone();
    entry.expected_calibration_film_stock = pipeline_cli.film_stock.clone();
    entry.expected_deskew_status = fixture.expectations.deskew_status.clone();
    entry.expected_deskew_applied = fixture.expectations.deskew_applied;
    entry.expected_deskew_review_required = fixture.expectations.deskew_review_required;
    entry.expected_deskew_retained_area_ratio_min =
        fixture.expectations.deskew_retained_area_ratio_min;
    entry.expected_stitch_decision = fixture.expectations.stitch_decision.clone();
    entry.expected_inferred_component_order = fixture.expectations.inferred_component_order.clone();
    entry.expected_technical_white_balance_status =
        fixture.expectations.technical_white_balance_status.clone();
    entry.expected_technical_white_balance_applied =
        fixture.expectations.technical_white_balance_applied;
    entry.expected_technical_white_balance_review_required =
        fixture.expectations.technical_white_balance_review_required;
    entry.expected_creative_temperature = fixture.expectations.creative_temperature;
    entry.expected_creative_tint = fixture.expectations.creative_tint;
    entry.expected_seam_exposure_model = fixture.expectations.seam_exposure_model.clone();
    entry.expected_seam_exposure_held_out_validation_passed = fixture
        .expectations
        .seam_exposure_held_out_validation_passed;
    entry.expected_seam_exposure_held_out_improvement_over_gain_min = fixture
        .expectations
        .seam_exposure_held_out_improvement_over_gain_min;
    entry.expected_seam_exposure_offset_normalized_abs_max =
        fixture.expectations.seam_exposure_offset_normalized_abs_max;
    entry.expected_seam_exposure_spatial_slope_abs_min =
        fixture.expectations.seam_exposure_spatial_slope_abs_min;
    entry.expected_seam_exposure_spatial_slope_abs_max =
        fixture.expectations.seam_exposure_spatial_slope_abs_max;
    entry.expected_seam_exposure_spatial_slope_agreement_ratio_min = fixture
        .expectations
        .seam_exposure_spatial_slope_agreement_ratio_min;
    entry.expected_seam_exposure_held_out_spatial_improvement_over_best_constant_min = fixture
        .expectations
        .seam_exposure_held_out_spatial_improvement_over_best_constant_min;
    entry.expected_seam_exposure_spatial_offset_slope_normalized_abs_min = fixture
        .expectations
        .seam_exposure_spatial_offset_slope_normalized_abs_min;
    entry.expected_seam_exposure_spatial_offset_slope_normalized_abs_max = fixture
        .expectations
        .seam_exposure_spatial_offset_slope_normalized_abs_max;
    entry.expected_seam_exposure_spatial_offset_endpoint_normalized_abs_max = fixture
        .expectations
        .seam_exposure_spatial_offset_endpoint_normalized_abs_max;
    entry.expected_seam_exposure_spatial_affine_slope_agreement_ratio_min = fixture
        .expectations
        .seam_exposure_spatial_affine_slope_agreement_ratio_min;
    entry.expected_seam_exposure_spatial_affine_center_offset_delta_normalized_max = fixture
        .expectations
        .seam_exposure_spatial_affine_center_offset_delta_normalized_max;
    entry.expected_seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler_min =
        fixture
            .expectations
            .seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler_min;
    entry.expected_seam_blend_required = fixture.expectations.seam_blend_required;
    entry.expected_seam_blend_mode = fixture.expectations.seam_blend_mode.clone();
    entry.expected_seam_blend_review_required = fixture.expectations.seam_blend_review_required;
    entry.expected_seam_detail_review_required = fixture.expectations.seam_detail_review_required;
    entry.expected_seam_detail_supported_scale_count_min =
        fixture.expectations.seam_detail_supported_scale_count_min;
    entry.expected_seam_detail_max_symmetric_energy_ratio_max = fixture
        .expectations
        .seam_detail_max_symmetric_energy_ratio_max;
    entry.expected_seam_gradient_ratio_max = fixture.expectations.seam_gradient_ratio_max;
    entry.expected_seam_overlap_p95_abs_difference_max =
        fixture.expectations.seam_overlap_p95_abs_difference_max;
    entry.expected_base_estimate_source = fixture.expectations.base_estimate_source.clone();
    entry.expected_output_color_space = fixture.expectations.output_color_space.clone();
    entry.expected_render_input_source = fixture.expectations.render_input_source.clone();
    entry.expected_render_input_reason_contains =
        fixture.expectations.render_input_reason_contains.clone();
    entry.expected_selected_mapping_reason_contains = fixture
        .expectations
        .selected_mapping_reason_contains
        .clone();
    entry.expected_selected_candidate = fixture.expectations.selected_candidate.clone();
    entry.expected_selected_candidate_rank = fixture.expectations.selected_candidate_rank;
    entry.expected_calibration_color_mapping_applied =
        fixture.expectations.calibration_color_mapping_applied;
    entry.expected_candidate_acceptance_signatures_required = fixture
        .expectations
        .candidate_acceptance_signatures_required
        .clone();
    entry.expected_selection_rejections_required =
        fixture.expectations.selection_rejections_required.clone();
    entry.expected_calibration_rejection_details_required = fixture
        .expectations
        .calibration_rejection_details_required
        .clone();
    entry.expected_calibration_confidence_min = fixture.expectations.calibration_confidence_min;
    entry.expected_calibration_matrix_condition_number_max =
        fixture.expectations.calibration_matrix_condition_number_max;
    entry.expected_selected_quality_score_max = fixture.expectations.selected_quality_score_max;
    entry.expected_selected_runner_up_quality_delta_min =
        fixture.expectations.selected_runner_up_quality_delta_min;
    entry.expected_technical_safety_score_max = fixture.expectations.technical_safety_score_max;
    entry.expected_color_fidelity_score_max = fixture.expectations.color_fidelity_score_max;
    entry.expected_memory_color_penalty_max = fixture.expectations.memory_color_penalty_max;
    entry.expected_spatial_consistency_penalty_max =
        fixture.expectations.spatial_consistency_penalty_max;
    entry.expected_density_monotonicity_score_min =
        fixture.expectations.density_monotonicity_score_min;
    entry.expected_hue_linearity_score_min = fixture.expectations.hue_linearity_score_min;
    entry.expected_saturation_preservation_median_ratio_min = fixture
        .expectations
        .saturation_preservation_median_ratio_min;
    entry.expected_spatial_neutral_delta_p95_max =
        fixture.expectations.spatial_neutral_delta_p95_max;
    entry.expected_candidate_risk = fixture.expectations.candidate_risk.clone();
    entry.expected_tone_color_trust_state = fixture.expectations.tone_color_trust_state.clone();
    entry.expected_neutral_safety_rescue_applied =
        fixture.expectations.neutral_safety_rescue_applied;
    entry.expected_neutral_safety_rescue_preserved_ratio_gain_min = fixture
        .expectations
        .neutral_safety_rescue_preserved_ratio_gain_min;
    entry.expected_neutral_safety_rescue_midtone_saturation_p95_reduction_min = fixture
        .expectations
        .neutral_safety_rescue_midtone_saturation_p95_reduction_min;
    entry.expected_neutral_safety_rescue_reason_contains = fixture
        .expectations
        .neutral_safety_rescue_reason_contains
        .clone();
    entry.expected_highlight_chroma_compressed_ratio_min =
        fixture.expectations.highlight_chroma_compressed_ratio_min;
    entry.expected_highlight_chroma_compressed_ratio_max =
        fixture.expectations.highlight_chroma_compressed_ratio_max;
    entry.expected_highlight_neutral_chroma_compressed_ratio_max = fixture
        .expectations
        .highlight_neutral_chroma_compressed_ratio_max;
    entry.expected_shadow_chroma_compressed_ratio_max =
        fixture.expectations.shadow_chroma_compressed_ratio_max;
    entry.expected_grain_reduction_enabled = fixture.expectations.grain_reduction_enabled;
    entry.expected_grain_reduction_applied_ratio_min =
        fixture.expectations.grain_reduction_applied_ratio_min;
    entry.expected_grain_reduction_structure_excluded_ratio_min = fixture
        .expectations
        .grain_reduction_structure_excluded_ratio_min;
    entry.expected_grain_reduction_flat_luma_p95_reduction_ratio_min = fixture
        .expectations
        .grain_reduction_flat_luma_p95_reduction_ratio_min;
    entry.expected_grain_reduction_flat_chroma_p95_reduction_ratio_min = fixture
        .expectations
        .grain_reduction_flat_chroma_p95_reduction_ratio_min;
    entry.expected_grain_detail_review_required = fixture.expectations.grain_detail_review_required;
    entry.expected_grain_detail_decision_supported =
        fixture.expectations.grain_detail_decision_supported;
    entry.expected_grain_detail_luminance_probe_count_min =
        fixture.expectations.grain_detail_luminance_probe_count_min;
    entry.expected_grain_detail_chroma_probe_count_min =
        fixture.expectations.grain_detail_chroma_probe_count_min;
    entry.expected_grain_detail_luminance_p10_retention_min = fixture
        .expectations
        .grain_detail_luminance_p10_retention_min;
    entry.expected_grain_detail_chroma_p10_retention_min =
        fixture.expectations.grain_detail_chroma_p10_retention_min;
    entry.expected_mapping_strategy = fixture.expectations.mapping_strategy.clone();
    entry.expected_post_scale_preserved_ratio_min =
        fixture.expectations.post_scale_preserved_ratio_min;
    entry.expected_render_luminance_range_p05_p95_min =
        fixture.expectations.render_luminance_range_p05_p95_min;
    entry.expected_render_review_status = fixture.expectations.render_review_status.clone();
    entry.expected_render_reviewable = fixture.expectations.render_reviewable;
    entry.expected_tone_output_confidence_status =
        fixture.expectations.tone_output_confidence_status.clone();
    entry.expected_tone_output_review_required = fixture.expectations.tone_output_review_required;
    entry.expected_tone_output_evidence_confidence_min =
        fixture.expectations.tone_output_evidence_confidence_min;
    entry.expected_render_to_mapped_luminance_range_ratio_min = fixture
        .expectations
        .render_to_mapped_luminance_range_ratio_min;
    entry.expected_post_chroma_compression_clipped_high_ratio_max = fixture
        .expectations
        .post_chroma_compression_clipped_high_ratio_max;
    entry.expected_post_chroma_compression_clipped_low_ratio_max = fixture
        .expectations
        .post_chroma_compression_clipped_low_ratio_max;
    entry.expected_reference_patch_evaluation_required =
        fixture.expectations.reference_patch_evaluation_required;
    entry.expected_reference_patch_count_min = fixture.expectations.reference_patch_count_min;
    entry.expected_reference_patch_hue_family_regression_count_max = fixture
        .expectations
        .reference_patch_hue_family_regression_count_max;
    entry.expected_reference_patch_selected_regresses_image_derived = fixture
        .expectations
        .reference_patch_selected_regresses_image_derived;
    entry.expected_reference_patch_delta_e2000_delta_vs_image_derived_max = fixture
        .expectations
        .reference_patch_delta_e2000_delta_vs_image_derived_max;
    entry.expected_reference_patch_max_delta_vs_image_derived_max = fixture
        .expectations
        .reference_patch_max_delta_vs_image_derived_max;
    entry.expected_reference_patch_delta_e_max_delta_vs_image_derived_max = fixture
        .expectations
        .reference_patch_delta_e_max_delta_vs_image_derived_max;
    entry.expected_reference_patch_delta_e2000_max_delta_vs_image_derived_max = fixture
        .expectations
        .reference_patch_delta_e2000_max_delta_vs_image_derived_max;
    entry.expected_reference_patch_rms_delta_e_max =
        fixture.expectations.reference_patch_rms_delta_e_max;
    entry.expected_reference_patch_rms_delta_e2000_max =
        fixture.expectations.reference_patch_rms_delta_e2000_max;
    entry.expected_debug_artifacts_required = fixture.expectations.debug_artifacts_required;
    entry.expected_debug_artifact_kinds_required =
        fixture.expectations.debug_artifact_kinds_required.clone();

    let report = match scanstitch::pipeline::run(&pipeline_cli) {
        Ok(report) => report,
        Err(err) => {
            entry.status = "failed".to_string();
            entry
                .issues
                .push(format!("fixture_suite:{name}:pipeline_failed"));
            entry.error = Some(err.to_string());
            return entry;
        }
    };

    let report_path = pipeline_cli.output_dir.join("report.json");
    if let Err(err) = report.save(&report_path) {
        entry.status = "failed".to_string();
        entry
            .issues
            .push(format!("fixture_suite:{name}:report_write_failed"));
        entry.error = Some(err.to_string());
        return entry;
    }
    entry.report_path = Some(report_path.display().to_string());

    let mut summary = summarize_report_with_source(name.to_string(), &report, Some(&report_path));
    entry.issues.extend(
        summary
            .diagnostic_consistency_issues
            .iter()
            .map(|issue| format!("fixture_suite:{name}:diagnostic_consistency:{issue}")),
    );
    write_fixture_suite_summary_baseline(cli, name, fixture, &summary, &mut entry);
    if entry.status != "failed" {
        apply_fixture_suite_summary_baseline(name, fixture, &mut summary, &mut entry);
    }

    entry.output_path = summary.render.output_path.clone();
    entry.output_modified_at = summary.render.output_modified_at.clone();
    entry.output_width = summary.render.output_width;
    entry.output_height = summary.render.output_height;
    entry.output_color_space = summary.render.output_color_space.clone();
    entry.output_file_icc_profile_matches_report =
        summary.render.output_file_icc_profile_matches_report;
    entry.delivery_artifact_issues = delivery_artifact_integrity_issues(&summary.render)
        .into_iter()
        .map(str::to_string)
        .collect();
    entry.delivery_artifacts_intact = Some(entry.delivery_artifact_issues.is_empty());
    entry.stale_render_artifact_count = summary.render.stale_render_artifact_count;
    entry.deskew_status = summary.deskew.status.clone();
    entry.deskew_applied = summary.deskew.applied;
    entry.deskew_review_required = summary.deskew.review_required;
    entry.deskew_retained_area_ratio = summary.deskew.retained_area_ratio;
    entry.geometry_preparation.all_deskew_components_applied =
        summary.deskew.all_components_applied;
    entry
        .geometry_preparation
        .minimum_deskew_retained_area_ratio = summary.deskew.minimum_component_retained_area_ratio;
    entry
        .geometry_preparation
        .all_border_crop_components_cropped = summary.border_crop.all_components_cropped;
    entry
        .geometry_preparation
        .minimum_removed_edge_count_per_component =
        summary.border_crop.minimum_removed_edge_count_per_component;
    entry
        .geometry_preparation
        .minimum_border_crop_retained_area_ratio = summary.border_crop.minimum_retained_area_ratio;
    entry
        .geometry_preparation
        .maximum_border_crop_retained_area_ratio = summary.border_crop.maximum_retained_area_ratio;
    entry.geometry_preparation.border_crop_rejected =
        Some(summary.border_crop.rejected_crop_warning_count > 0);
    entry.geometry_preparation.deskew_correction_degrees = summary.deskew.correction_degrees;
    entry.geometry_preparation.border_crop_components = summary.border_crop.components.clone();
    entry.input_orientation.components = summary.input_orientation.components.clone();
    entry.stitch_decision = summary.stitch.decision.clone();
    entry.inferred_component_order = summary.stitch.inferred_order.clone();
    entry.technical_white_balance_status = summary.white_balance.technical_status.clone();
    entry.technical_white_balance_applied = summary.white_balance.technical_applied;
    entry.technical_white_balance_review_required = summary.white_balance.technical_review_required;
    entry.creative_temperature = summary.white_balance.creative_temperature;
    entry.creative_tint = summary.white_balance.creative_tint;
    entry.seam_exposure_model = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| correction.model.clone());
    entry.seam_exposure_held_out_validation_passed = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| correction.held_out_validation_passed);
    entry.seam_exposure_held_out_improvement_over_gain = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| correction.held_out_improvement_over_gain);
    entry.seam_exposure_offset_normalized_abs_max = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| correction.offset_rgb_normalized.as_ref())
        .map(|offsets| offsets.iter().map(|value| value.abs()).fold(0.0, f64::max));
    entry.seam_exposure_spatial_slope_abs_max = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| correction.spatial_gain_log_slope_y_rgb.as_ref())
        .map(|slopes| slopes.iter().map(|value| value.abs()).fold(0.0, f64::max));
    entry.seam_exposure_spatial_slope_agreement_ratio = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| correction.spatial_slope_agreement_ratio);
    entry.seam_exposure_held_out_spatial_improvement_over_best_constant = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| correction.held_out_spatial_improvement_over_best_constant);
    entry.seam_exposure_spatial_offset_slope_normalized_abs_max = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| correction.spatial_offset_slope_y_rgb_normalized.as_ref())
        .map(|slopes| slopes.iter().map(|value| value.abs()).fold(0.0, f64::max));
    entry.seam_exposure_spatial_offset_endpoint_normalized_abs_max = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| {
            correction
                .spatial_offset_top_rgb_normalized
                .as_ref()
                .zip(correction.spatial_offset_bottom_rgb_normalized.as_ref())
        })
        .map(|(top, bottom)| {
            top.iter()
                .chain(bottom.iter())
                .map(|value| value.abs())
                .fold(0.0, f64::max)
        });
    entry.seam_exposure_spatial_affine_slope_agreement_ratio = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| correction.spatial_affine_slope_agreement_ratio);
    entry.seam_exposure_spatial_affine_center_offset_delta_normalized = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| correction.spatial_affine_center_offset_delta_normalized);
    entry.seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| {
            correction.held_out_spatial_gain_offset_improvement_over_best_simpler
        });
    entry.seam_blend_mode = summary
        .stitch
        .seam_blend
        .as_ref()
        .and_then(|blend| blend.mode.clone());
    entry.seam_blend_review_required = summary
        .stitch
        .seam_blend
        .as_ref()
        .and_then(|blend| blend.review_required);
    entry.seam_detail_review_required = summary
        .stitch
        .seam_blend
        .as_ref()
        .and_then(|blend| blend.detail_consistency.as_ref())
        .and_then(|detail| detail.review_required);
    entry.seam_detail_supported_scale_count = summary
        .stitch
        .seam_blend
        .as_ref()
        .and_then(|blend| blend.detail_consistency.as_ref())
        .and_then(|detail| detail.minimum_supported_scale_count);
    entry.seam_detail_max_symmetric_energy_ratio = summary
        .stitch
        .seam_blend
        .as_ref()
        .and_then(|blend| blend.detail_consistency.as_ref())
        .and_then(|detail| detail.maximum_symmetric_energy_ratio);
    entry.seam_gradient_ratio = summary
        .stitch
        .seam_blend
        .as_ref()
        .and_then(|blend| blend.output_to_source_seam_gradient_ratio);
    entry.seam_overlap_p95_abs_difference = summary
        .stitch
        .seam_blend
        .as_ref()
        .and_then(|blend| blend.overlap_p95_abs_difference);
    entry.base_estimate_source = summary.base_density.base_estimate_source.clone();
    entry.negative_reconstruction.base_confidence = summary.base_density.base_confidence;
    entry.negative_reconstruction.density_inversion_skipped =
        summary.negative_reconstruction.density_inversion_skipped;
    entry.negative_reconstruction.response_model =
        summary.negative_reconstruction.response_model.clone();
    entry.negative_reconstruction.response_source =
        summary.negative_reconstruction.response_source.clone();
    entry.negative_reconstruction.response_accepted =
        summary.negative_reconstruction.response_accepted;
    entry.negative_reconstruction.response_review_required = summary
        .negative_reconstruction
        .reconstruction_review_required;
    entry.negative_reconstruction.crosstalk_model =
        summary.negative_reconstruction.crosstalk_model.clone();
    entry.negative_reconstruction.characteristic_curve_model = summary
        .negative_reconstruction
        .characteristic_curve_model
        .clone();
    entry.negative_reconstruction.measured_model_id =
        summary.negative_reconstruction.measured_model_id.clone();
    entry.negative_reconstruction.measured_confidence =
        summary.negative_reconstruction.measured_confidence;
    entry.negative_reconstruction.held_out_delta_e00_rms =
        summary.negative_reconstruction.held_out_delta_e00_rms;
    entry.negative_reconstruction.held_out_delta_e00_max =
        summary.negative_reconstruction.held_out_delta_e00_max;
    entry
        .negative_reconstruction
        .held_out_improvement_over_unit_slope = summary
        .negative_reconstruction
        .held_out_improvement_over_unit_slope;
    entry.negative_reconstruction.maximum_density_noise_gain =
        summary.negative_reconstruction.maximum_density_noise_gain;
    entry.negative_reconstruction.curve_extrapolated_any_ratio =
        summary.negative_reconstruction.curve_extrapolated_any_ratio;
    entry.negative_reconstruction.signed_headroom_preserved =
        summary.negative_reconstruction.signed_headroom_preserved;
    entry.negative_reconstruction.curve_interpolation =
        summary.negative_reconstruction.curve_interpolation.clone();
    entry.render_input_source = summary.colorspace.render_input_source.clone();
    entry.render_input_reason = summary.colorspace.render_input_reason.clone();
    entry.selected_mapping_reason = summary.colorspace.selected_mapping_reason.clone();
    entry.calibration_status = summary.colorspace.calibration_status.clone();
    entry.calibration_source = summary.colorspace.calibration_source.clone();
    entry.calibration_scanner_profile_status = summary
        .colorspace
        .calibration_scanner_profile_status
        .clone();
    entry.calibration_scanner_profile_id =
        summary.colorspace.calibration_scanner_profile_id.clone();
    entry.calibration_roll_profile_status =
        summary.colorspace.calibration_roll_profile_status.clone();
    entry.calibration_roll_profile_id = summary.colorspace.calibration_roll_profile_id.clone();
    entry.calibration_requested_film_stock =
        summary.colorspace.calibration_requested_film_stock.clone();
    entry.calibration_acceptance_status = summary
        .colorspace
        .calibration_acceptance
        .as_ref()
        .and_then(|acceptance| acceptance.status.clone());
    entry.calibration_color_mapping_applied = summary
        .colorspace
        .calibration_color_mapping_application
        .as_ref()
        .and_then(|application| application.applied);
    entry.calibration_confidence = summary.colorspace.calibration_confidence;
    entry.calibration_matrix_condition_number =
        summary.colorspace.calibration_matrix_condition_number;
    entry.calibration_rejection_details = summary.colorspace.calibration_rejection_details.clone();
    entry.selected_candidate = summary.colorspace.selected_candidate.clone();
    entry.selected_candidate_rank = summary.colorspace.selected_candidate_rank;
    entry.candidate_acceptance_signatures =
        fixture_suite_candidate_acceptance_signatures(&summary.colorspace.candidate_acceptance);
    entry.selection_rejections = summary.colorspace.selection_rejections.clone();
    entry.selected_quality_score = summary.colorspace.selected_quality_score;
    entry.selected_runner_up_quality_delta = summary.colorspace.selected_runner_up_quality_delta;
    entry.technical_safety_score = summary.colorspace.technical_safety_score;
    entry.color_fidelity_score = summary.colorspace.color_fidelity_score;
    entry.memory_color_penalty = summary.render.colorspace_memory_color_penalty;
    entry.spatial_consistency_penalty = summary.render.colorspace_spatial_consistency_penalty;
    entry.density_monotonicity_score = summary.render.colorspace_density_monotonicity_score;
    entry.hue_linearity_score = summary.render.colorspace_hue_linearity_score;
    entry.saturation_preservation_median_ratio = summary
        .render
        .colorspace_saturation_preservation_median_ratio;
    entry.spatial_neutral_delta_p95 = summary.render.colorspace_spatial_neutral_delta_p95;
    entry.candidate_risk = summary.colorspace.candidate_risk.clone();
    entry.tone_color_trust_state = summary.colorspace.tone_color_trust_state.clone();
    let neutral_safety_rescue = summary.colorspace.neutral_safety_rescue.as_ref();
    entry.neutral_safety_rescue_applied = neutral_safety_rescue.and_then(|rescue| rescue.applied);
    entry.neutral_safety_rescue_preserved_ratio_gain =
        neutral_safety_rescue.and_then(|rescue| rescue.preserved_ratio_gain);
    entry.neutral_safety_rescue_midtone_saturation_p95_reduction =
        neutral_safety_rescue.and_then(|rescue| rescue.midtone_saturation_p95_reduction);
    entry.neutral_safety_rescue_reason =
        neutral_safety_rescue.and_then(|rescue| rescue.reason.clone());
    entry.highlight_chroma_compressed_ratio = summary.tone.highlight_chroma_compressed_ratio;
    entry.highlight_neutral_chroma_compressed_ratio =
        summary.tone.highlight_neutral_chroma_compressed_ratio;
    entry.shadow_chroma_compressed_ratio = summary.tone.shadow_chroma_compressed_ratio;
    entry.grain_reduction_enabled = summary.tone.noise_reduction_enabled;
    entry.grain_reduction_applied_ratio = summary.tone.noise_reduction_applied_ratio;
    entry.grain_reduction_structure_excluded_ratio =
        summary.tone.noise_reduction_structure_excluded_ratio;
    entry.grain_reduction_flat_luma_p95_reduction_ratio =
        summary.tone.noise_reduction_flat_luma_p95_reduction_ratio;
    entry.grain_reduction_flat_chroma_p95_reduction_ratio =
        summary.tone.noise_reduction_flat_chroma_p95_reduction_ratio;
    let grain_detail = summary.tone.grain_detail_retention.as_ref();
    entry.grain_detail_review_required = grain_detail.and_then(|detail| detail.review_required);
    entry.grain_detail_decision_supported =
        grain_detail.and_then(|detail| detail.decision_supported);
    entry.grain_detail_luminance_probe_count =
        grain_detail.and_then(|detail| detail.luminance_probe_count);
    entry.grain_detail_chroma_probe_count =
        grain_detail.and_then(|detail| detail.chroma_probe_count);
    entry.grain_detail_luminance_p10_retention =
        grain_detail.and_then(|detail| detail.luminance_p10_retention);
    entry.grain_detail_chroma_p10_retention =
        grain_detail.and_then(|detail| detail.chroma_p10_retention);
    entry.mapping_strategy = summary.colorspace.mapping_strategy.clone();
    entry.post_scale_preserved_ratio = summary.colorspace.post_scale_preserved_ratio;
    entry.render_luminance_range_p05_p95 = summary.tone.render_luminance_range_p05_p95;
    entry.render_review_status = summary.render.render_review_status.clone();
    entry.render_reviewable = summary.render.render_reviewable;
    entry.tone_output_confidence_status = summary.tone.tone_output_confidence_status.clone();
    entry.tone_output_review_required = summary.tone.tone_output_review_required;
    entry.tone_output_evidence_confidence = summary.tone.tone_output_evidence_confidence;
    entry.render_to_mapped_luminance_range_ratio =
        summary.tone.render_to_mapped_luminance_range_ratio;
    entry.post_chroma_compression_clipped_high_ratio_max = max_f64_slice(
        summary
            .tone
            .post_chroma_compression_clipped_high_ratio
            .as_deref(),
    );
    entry.post_chroma_compression_clipped_low_ratio_max = max_f64_slice(
        summary
            .tone
            .post_chroma_compression_clipped_low_ratio
            .as_deref(),
    );
    entry.reference_patch_evaluation_present =
        Some(summary.colorspace.reference_patch_evaluation.is_some());
    entry.reference_patch_patch_count = summary
        .colorspace
        .reference_patch_evaluation
        .as_ref()
        .and_then(|evaluation| evaluation.patch_count);
    entry.reference_patch_selected_rms_delta_e = summary
        .colorspace
        .reference_patch_evaluation
        .as_ref()
        .and_then(|evaluation| evaluation.selected_rms_delta_e);
    entry.reference_patch_selected_rms_delta_e2000 = summary
        .colorspace
        .reference_patch_evaluation
        .as_ref()
        .and_then(|evaluation| evaluation.selected_rms_delta_e2000);
    entry.reference_patch_delta_e2000_delta_vs_image_derived = summary
        .colorspace
        .reference_patch_evaluation
        .as_ref()
        .and_then(|evaluation| evaluation.delta_e2000_rms_delta_vs_image_derived);
    entry.reference_patch_max_delta_vs_image_derived = summary
        .colorspace
        .reference_patch_evaluation
        .as_ref()
        .and_then(|evaluation| evaluation.max_error_delta_vs_image_derived);
    entry.reference_patch_delta_e_max_delta_vs_image_derived = summary
        .colorspace
        .reference_patch_evaluation
        .as_ref()
        .and_then(|evaluation| evaluation.delta_e_max_delta_vs_image_derived);
    entry.reference_patch_delta_e2000_max_delta_vs_image_derived = summary
        .colorspace
        .reference_patch_evaluation
        .as_ref()
        .and_then(|evaluation| evaluation.delta_e2000_max_delta_vs_image_derived);
    entry.reference_patch_selected_regresses_image_derived = summary
        .colorspace
        .reference_patch_evaluation
        .as_ref()
        .and_then(|evaluation| evaluation.selected_regresses_image_derived);
    entry.reference_patch_hue_family_regressions = summary
        .colorspace
        .reference_patch_evaluation
        .as_ref()
        .map(|evaluation| evaluation.hue_family_regressions.clone())
        .unwrap_or_default();
    let debug_artifact_issues = fixture_suite_debug_artifact_issues(&summary);
    entry.debug_artifact_count = Some(summary.colorspace.debug_artifacts.len());
    entry.debug_artifact_invalid_count = Some(debug_artifact_issues.len());
    entry.debug_artifact_kinds = summary
        .colorspace
        .debug_artifacts
        .iter()
        .map(|artifact| artifact.kind.clone())
        .collect();
    validate_fixture_suite_output_guardrails(
        name,
        &fixture.expectations,
        cli.require_reviewable,
        &summary,
        &mut entry,
    );
    validate_fixture_suite_calibration_expectations(name, &pipeline_cli, &summary, &mut entry);
    validate_fixture_suite_color_expectations(
        name,
        &fixture.expectations,
        &summary,
        &debug_artifact_issues,
        &mut entry,
    );

    let summary_json_path = output_dir.join("summary.json");
    let summary_md_path = output_dir.join("summary.md");
    let json_summary = match serde_json::to_string_pretty(&summary) {
        Ok(json) => json,
        Err(err) => {
            entry.status = "failed".to_string();
            entry
                .issues
                .push(format!("fixture_suite:{name}:summary_serialize_failed"));
            entry.error = Some(err.to_string());
            return entry;
        }
    };
    if let Err(err) = write_text(&summary_json_path, &json_summary) {
        entry.status = "failed".to_string();
        entry
            .issues
            .push(format!("fixture_suite:{name}:summary_json_write_failed"));
        entry.error = Some(err.to_string());
        return entry;
    }
    if let Err(err) = write_text(&summary_md_path, &summary_to_markdown(&summary)) {
        entry.status = "failed".to_string();
        entry
            .issues
            .push(format!("fixture_suite:{name}:summary_md_write_failed"));
        entry.error = Some(err.to_string());
        return entry;
    }
    entry.summary_json_path = Some(summary_json_path.display().to_string());
    entry.summary_md_path = Some(summary_md_path.display().to_string());

    if entry.status != "failed" && !entry.issues.is_empty() {
        entry.status = "review_required".to_string();
    }
    entry
}

fn validate_fixture_suite_color_expectations(
    name: &str,
    expectations: &FixtureExpectations,
    summary: &ValidationSummary,
    debug_artifact_issues: &[String],
    entry: &mut FixtureSuiteEntry,
) {
    push_fixture_suite_expected_string_issue(
        name,
        "deskew_status",
        expectations.deskew_status.as_deref(),
        summary.deskew.status.as_deref(),
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "deskew_applied",
        expectations.deskew_applied,
        summary.deskew.applied,
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "deskew_review_required",
        expectations.deskew_review_required,
        summary.deskew.review_required,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "deskew_retained_area_ratio",
        expectations.deskew_retained_area_ratio_min,
        summary.deskew.retained_area_ratio,
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "deskew_all_components_applied",
        expectations.deskew_all_components_applied,
        summary.deskew.all_components_applied,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "deskew_minimum_component_retained_area_ratio",
        expectations.deskew_minimum_component_retained_area_ratio_min,
        summary.deskew.minimum_component_retained_area_ratio,
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "border_crop_all_components_cropped",
        expectations.border_crop_all_components_cropped,
        summary.border_crop.all_components_cropped,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "border_crop_minimum_removed_edge_count_per_component",
        expectations
            .border_crop_minimum_removed_edge_count_per_component_min
            .map(|value| value as f64),
        summary
            .border_crop
            .minimum_removed_edge_count_per_component
            .map(|value| value as f64),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "border_crop_retained_area_ratio",
        expectations.border_crop_retained_area_ratio_min,
        summary.border_crop.minimum_retained_area_ratio,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "border_crop_retained_area_ratio",
        expectations.border_crop_retained_area_ratio_max,
        summary.border_crop.maximum_retained_area_ratio,
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "border_crop_rejected",
        expectations.border_crop_rejected,
        Some(summary.border_crop.rejected_crop_warning_count > 0),
        entry,
    );
    validate_fixture_suite_geometry_accuracy_expectations(name, expectations, summary, entry);
    validate_fixture_suite_orientation_accuracy_expectations(name, expectations, summary, entry);
    push_fixture_suite_expected_string_issue(
        name,
        "stitch_decision",
        expectations.stitch_decision.as_deref(),
        summary.stitch.decision.as_deref(),
        entry,
    );
    if !expectations.inferred_component_order.is_empty()
        && expectations.inferred_component_order != summary.stitch.inferred_order
    {
        entry.issues.push(format!(
            "fixture_suite:{name}:inferred_component_order_mismatch:expected={:?}:actual={:?}",
            expectations.inferred_component_order, summary.stitch.inferred_order
        ));
    }
    push_fixture_suite_expected_string_issue(
        name,
        "technical_white_balance_status",
        expectations.technical_white_balance_status.as_deref(),
        summary.white_balance.technical_status.as_deref(),
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "technical_white_balance_applied",
        expectations.technical_white_balance_applied,
        summary.white_balance.technical_applied,
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "technical_white_balance_review_required",
        expectations.technical_white_balance_review_required,
        summary.white_balance.technical_review_required,
        entry,
    );
    push_fixture_suite_expected_f64_issue(
        name,
        "creative_temperature",
        expectations.creative_temperature,
        summary.white_balance.creative_temperature,
        entry,
    );
    push_fixture_suite_expected_f64_issue(
        name,
        "creative_tint",
        expectations.creative_tint,
        summary.white_balance.creative_tint,
        entry,
    );
    let seam_exposure = summary.stitch.seam_exposure_correction.as_ref();
    push_fixture_suite_expected_string_issue(
        name,
        "seam_exposure_model",
        expectations.seam_exposure_model.as_deref(),
        seam_exposure.and_then(|correction| correction.model.as_deref()),
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "seam_exposure_held_out_validation_passed",
        expectations.seam_exposure_held_out_validation_passed,
        seam_exposure.and_then(|correction| correction.held_out_validation_passed),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "seam_exposure_held_out_improvement_over_gain",
        expectations.seam_exposure_held_out_improvement_over_gain_min,
        seam_exposure.and_then(|correction| correction.held_out_improvement_over_gain),
        entry,
    );
    let offset_normalized_abs_max = seam_exposure
        .and_then(|correction| correction.offset_rgb_normalized.as_ref())
        .map(|offsets| offsets.iter().map(|value| value.abs()).fold(0.0, f64::max));
    push_fixture_suite_expected_max_issue(
        name,
        "seam_exposure_offset_normalized_abs_max",
        expectations.seam_exposure_offset_normalized_abs_max,
        offset_normalized_abs_max,
        entry,
    );
    let spatial_slope_abs_max = seam_exposure.and_then(|correction| {
        let x = correction
            .spatial_gain_log_slope_x_rgb
            .as_ref()
            .map(|slopes| slopes.iter().map(|value| value.abs()).fold(0.0, f64::max));
        let y = correction
            .spatial_gain_log_slope_y_rgb
            .as_ref()
            .map(|slopes| slopes.iter().map(|value| value.abs()).fold(0.0, f64::max));
        x.zip(y).map(|(x, y)| x.max(y)).or(x).or(y)
    });
    push_fixture_suite_expected_min_issue(
        name,
        "seam_exposure_spatial_slope_abs_max",
        expectations.seam_exposure_spatial_slope_abs_min,
        spatial_slope_abs_max,
        entry,
    );
    let seam_blend = summary.stitch.seam_blend.as_ref();
    let seam_detail = seam_blend.and_then(|blend| blend.detail_consistency.as_ref());
    push_fixture_suite_expected_bool_issue(
        name,
        "seam_blend_review_required",
        expectations.seam_blend_review_required,
        seam_blend.and_then(|blend| blend.review_required),
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "seam_detail_review_required",
        expectations.seam_detail_review_required,
        seam_detail.and_then(|detail| detail.review_required),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "seam_detail_supported_scale_count",
        expectations
            .seam_detail_supported_scale_count_min
            .map(|value| value as f64),
        seam_detail
            .and_then(|detail| detail.minimum_supported_scale_count)
            .map(|value| value as f64),
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "seam_detail_max_symmetric_energy_ratio",
        expectations.seam_detail_max_symmetric_energy_ratio_max,
        seam_detail.and_then(|detail| detail.maximum_symmetric_energy_ratio),
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "seam_exposure_spatial_slope_abs_max",
        expectations.seam_exposure_spatial_slope_abs_max,
        spatial_slope_abs_max,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "seam_exposure_spatial_slope_agreement_ratio",
        expectations.seam_exposure_spatial_slope_agreement_ratio_min,
        seam_exposure.and_then(|correction| correction.spatial_slope_agreement_ratio),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "seam_exposure_held_out_spatial_improvement_over_best_constant",
        expectations.seam_exposure_held_out_spatial_improvement_over_best_constant_min,
        seam_exposure
            .and_then(|correction| correction.held_out_spatial_improvement_over_best_constant),
        entry,
    );
    let spatial_offset_slope_abs_max = seam_exposure.and_then(|correction| {
        let x = correction
            .spatial_offset_slope_x_rgb_normalized
            .as_ref()
            .map(|slopes| slopes.iter().map(|value| value.abs()).fold(0.0, f64::max));
        let y = correction
            .spatial_offset_slope_y_rgb_normalized
            .as_ref()
            .map(|slopes| slopes.iter().map(|value| value.abs()).fold(0.0, f64::max));
        x.zip(y).map(|(x, y)| x.max(y)).or(x).or(y)
    });
    push_fixture_suite_expected_min_issue(
        name,
        "seam_exposure_spatial_offset_slope_normalized_abs_max",
        expectations.seam_exposure_spatial_offset_slope_normalized_abs_min,
        spatial_offset_slope_abs_max,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "seam_exposure_spatial_offset_slope_normalized_abs_max",
        expectations.seam_exposure_spatial_offset_slope_normalized_abs_max,
        spatial_offset_slope_abs_max,
        entry,
    );
    let spatial_offset_endpoint_abs_max = seam_exposure
        .and_then(|correction| {
            correction
                .spatial_offset_top_rgb_normalized
                .as_ref()
                .zip(correction.spatial_offset_bottom_rgb_normalized.as_ref())
        })
        .map(|(top, bottom)| {
            top.iter()
                .chain(bottom.iter())
                .map(|value| value.abs())
                .fold(0.0, f64::max)
        });
    push_fixture_suite_expected_max_issue(
        name,
        "seam_exposure_spatial_offset_endpoint_normalized_abs_max",
        expectations.seam_exposure_spatial_offset_endpoint_normalized_abs_max,
        spatial_offset_endpoint_abs_max,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "seam_exposure_spatial_affine_slope_agreement_ratio",
        expectations.seam_exposure_spatial_affine_slope_agreement_ratio_min,
        seam_exposure.and_then(|correction| correction.spatial_affine_slope_agreement_ratio),
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "seam_exposure_spatial_affine_center_offset_delta_normalized",
        expectations.seam_exposure_spatial_affine_center_offset_delta_normalized_max,
        seam_exposure
            .and_then(|correction| correction.spatial_affine_center_offset_delta_normalized),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler",
        expectations.seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler_min,
        seam_exposure.and_then(|correction| {
            correction.held_out_spatial_gain_offset_improvement_over_best_simpler
        }),
        entry,
    );
    let spatial_2d = seam_exposure.and_then(|correction| correction.spatial_2d_validation.as_ref());
    push_fixture_suite_expected_bool_issue(
        name,
        "seam_exposure_spatial_2d_gain_accepted",
        expectations.seam_exposure_spatial_2d_gain_accepted,
        spatial_2d.and_then(|validation| validation.gain_accepted),
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "seam_exposure_spatial_2d_gain_offset_accepted",
        expectations.seam_exposure_spatial_2d_gain_offset_accepted,
        spatial_2d.and_then(|validation| validation.gain_offset_accepted),
        entry,
    );
    let spatial_quadratic = spatial_2d.and_then(|validation| validation.quadratic.as_ref());
    push_fixture_suite_expected_bool_issue(
        name,
        "seam_exposure_spatial_quadratic_gain_accepted",
        expectations.seam_exposure_spatial_quadratic_gain_accepted,
        spatial_quadratic.and_then(|validation| validation.gain_accepted),
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "seam_exposure_spatial_quadratic_gain_offset_accepted",
        expectations.seam_exposure_spatial_quadratic_gain_offset_accepted,
        spatial_quadratic.and_then(|validation| validation.gain_offset_accepted),
        entry,
    );
    if let Some(expected) = expectations.seam_exposure_spatial_2d_distinct_columns_min {
        match spatial_2d.and_then(|validation| {
            validation
                .distinct_training_columns
                .zip(validation.distinct_held_out_columns)
                .map(|(training, held_out)| training.min(held_out))
        }) {
            Some(actual) if actual >= expected => {}
            Some(actual) => entry.issues.push(format!(
                "fixture_suite:{name}:seam_exposure_spatial_2d_distinct_columns_below_minimum:expected_min={expected}:actual={actual}"
            )),
            None => entry.issues.push(format!(
                "fixture_suite:{name}:seam_exposure_spatial_2d_distinct_columns_missing"
            )),
        }
    }
    let selected_model = seam_exposure.and_then(|correction| correction.model.as_deref());
    let selected_2d_gain_offset = matches!(
        selected_model,
        Some("gain_offset_spatial_xy_rgb" | "gain_offset_spatial_quadratic_xy_rgb")
    );
    let selected_quadratic = matches!(
        selected_model,
        Some("gain_spatial_quadratic_xy_rgb" | "gain_offset_spatial_quadratic_xy_rgb")
    );
    let selected_2d_horizontal_agreement = spatial_2d.and_then(|validation| {
        if selected_quadratic {
            validation.quadratic.as_ref().and_then(|quadratic| {
                if selected_2d_gain_offset {
                    quadratic.gain_offset_curvature_coefficient_agreement_ratio
                } else {
                    quadratic.gain_curvature_coefficient_agreement_ratio
                }
            })
        } else if selected_2d_gain_offset {
            validation.gain_offset_horizontal_slope_agreement_ratio
        } else {
            validation.gain_horizontal_slope_agreement_ratio
        }
    });
    push_fixture_suite_expected_min_issue(
        name,
        "seam_exposure_spatial_2d_horizontal_slope_agreement_ratio",
        expectations.seam_exposure_spatial_2d_horizontal_slope_agreement_ratio_min,
        selected_2d_horizontal_agreement,
        entry,
    );
    let selected_2d_improvement = spatial_2d.and_then(|validation| {
        if selected_quadratic {
            validation.quadratic.as_ref().and_then(|quadratic| {
                if selected_2d_gain_offset {
                    quadratic.gain_offset_improvement_over_best_simpler
                } else {
                    quadratic.gain_improvement_over_best_simpler
                }
            })
        } else if selected_2d_gain_offset {
            validation.gain_offset_improvement_over_best_simpler
        } else {
            validation.gain_improvement_over_best_simpler
        }
    });
    push_fixture_suite_expected_min_issue(
        name,
        "seam_exposure_held_out_spatial_2d_improvement_over_best_simpler",
        expectations.seam_exposure_held_out_spatial_2d_improvement_over_best_simpler_min,
        selected_2d_improvement,
        entry,
    );
    if expectations.seam_blend_required
        && !summary
            .stitch
            .seam_blend
            .as_ref()
            .is_some_and(|blend| blend.applied == Some(true) && blend.applied_merge_count > 0)
    {
        entry.issues.push(format!(
            "fixture_suite:{name}:seam_blend_missing_or_not_applied"
        ));
    }
    push_fixture_suite_expected_string_issue(
        name,
        "seam_blend_mode",
        expectations.seam_blend_mode.as_deref(),
        summary
            .stitch
            .seam_blend
            .as_ref()
            .and_then(|blend| blend.mode.as_deref()),
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "seam_gradient_ratio",
        expectations.seam_gradient_ratio_max,
        summary
            .stitch
            .seam_blend
            .as_ref()
            .and_then(|blend| blend.output_to_source_seam_gradient_ratio),
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "seam_overlap_p95_abs_difference",
        expectations.seam_overlap_p95_abs_difference_max,
        summary
            .stitch
            .seam_blend
            .as_ref()
            .and_then(|blend| blend.overlap_p95_abs_difference),
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "base_estimate_source",
        expectations.base_estimate_source.as_deref(),
        summary.base_density.base_estimate_source.as_deref(),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "base_confidence",
        expectations.base_confidence_min,
        summary.base_density.base_confidence,
        entry,
    );
    let negative = &summary.negative_reconstruction;
    push_fixture_suite_expected_bool_issue(
        name,
        "density_inversion_skipped",
        expectations.density_inversion_skipped,
        negative.density_inversion_skipped,
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "negative_response_model",
        expectations.negative_response_model.as_deref(),
        negative.response_model.as_deref(),
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "negative_response_source",
        expectations.negative_response_source.as_deref(),
        negative.response_source.as_deref(),
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "negative_response_accepted",
        expectations.negative_response_accepted,
        negative.response_accepted,
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "negative_response_review_required",
        expectations.negative_response_review_required,
        negative.reconstruction_review_required,
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "negative_response_crosstalk_model",
        expectations.negative_response_crosstalk_model.as_deref(),
        negative.crosstalk_model.as_deref(),
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "negative_response_characteristic_curve_model",
        expectations
            .negative_response_characteristic_curve_model
            .as_deref(),
        negative.characteristic_curve_model.as_deref(),
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "negative_response_measured_model_id",
        expectations.negative_response_measured_model_id.as_deref(),
        negative.measured_model_id.as_deref(),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "negative_response_measured_confidence",
        expectations.negative_response_measured_confidence_min,
        negative.measured_confidence,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "negative_response_held_out_delta_e00_rms",
        expectations.negative_response_held_out_delta_e00_rms_max,
        negative.held_out_delta_e00_rms,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "negative_response_held_out_max_delta_e00",
        expectations.negative_response_held_out_max_delta_e00_max,
        negative.held_out_delta_e00_max,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "negative_response_held_out_improvement_over_unit_slope",
        expectations.negative_response_held_out_improvement_over_unit_slope_min,
        negative.held_out_improvement_over_unit_slope,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "negative_response_density_noise_gain",
        expectations.negative_response_density_noise_gain_max,
        negative.maximum_density_noise_gain,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "negative_response_curve_extrapolated_ratio",
        expectations.negative_response_curve_extrapolated_ratio_max,
        negative.curve_extrapolated_any_ratio,
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "negative_response_signed_headroom_preserved",
        expectations.negative_response_signed_headroom_preserved,
        negative.signed_headroom_preserved,
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "negative_response_curve_interpolation",
        expectations
            .negative_response_curve_interpolation
            .as_deref(),
        negative.curve_interpolation.as_deref(),
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "output_color_space",
        expectations.output_color_space.as_deref(),
        summary.render.output_color_space.as_deref(),
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "render_input_source",
        expectations.render_input_source.as_deref(),
        summary.colorspace.render_input_source.as_deref(),
        entry,
    );
    push_fixture_suite_expected_contains_issue(
        name,
        "render_input_reason",
        expectations.render_input_reason_contains.as_deref(),
        summary.colorspace.render_input_reason.as_deref(),
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "mapping_strategy",
        expectations.mapping_strategy.as_deref(),
        summary.colorspace.mapping_strategy.as_deref(),
        entry,
    );
    push_fixture_suite_expected_contains_issue(
        name,
        "selected_mapping_reason",
        expectations.selected_mapping_reason_contains.as_deref(),
        summary.colorspace.selected_mapping_reason.as_deref(),
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "selected_candidate",
        expectations.selected_candidate.as_deref(),
        summary.colorspace.selected_candidate.as_deref(),
        entry,
    );
    if expectations.selected_candidate_rank.is_some()
        && expectations.selected_candidate_rank != summary.colorspace.selected_candidate_rank
    {
        entry.issues.push(format!(
            "fixture_suite:{name}:selected_candidate_rank_mismatch"
        ));
    }
    let candidate_acceptance_signatures =
        fixture_suite_candidate_acceptance_signatures(&summary.colorspace.candidate_acceptance);
    for required_signature in &expectations.candidate_acceptance_signatures_required {
        if !candidate_acceptance_signatures.contains(required_signature) {
            entry.issues.push(format!(
                "fixture_suite:{name}:candidate_acceptance_signature_missing:{required_signature}"
            ));
        }
    }
    push_fixture_suite_expected_string_issue(
        name,
        "calibration_acceptance_status",
        expectations.calibration_acceptance_status.as_deref(),
        summary
            .colorspace
            .calibration_acceptance
            .as_ref()
            .and_then(|acceptance| acceptance.status.as_deref()),
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "calibration_color_mapping_applied",
        expectations.calibration_color_mapping_applied,
        summary
            .colorspace
            .calibration_color_mapping_application
            .as_ref()
            .and_then(|application| application.applied),
        entry,
    );
    for required_detail in &expectations.calibration_rejection_details_required {
        if !summary
            .colorspace
            .calibration_rejection_details
            .contains(required_detail)
        {
            entry.issues.push(format!(
                "fixture_suite:{name}:calibration_rejection_detail_missing:{required_detail}"
            ));
        }
    }
    for required_rejection in &expectations.selection_rejections_required {
        if !summary
            .colorspace
            .selection_rejections
            .contains(required_rejection)
        {
            entry.issues.push(format!(
                "fixture_suite:{name}:selection_rejection_missing:{required_rejection}"
            ));
        }
    }
    push_fixture_suite_expected_min_issue(
        name,
        "calibration_confidence",
        expectations.calibration_confidence_min,
        summary.colorspace.calibration_confidence,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "calibration_matrix_condition_number",
        expectations.calibration_matrix_condition_number_max,
        summary.colorspace.calibration_matrix_condition_number,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "selected_quality_score",
        expectations.selected_quality_score_max,
        summary.colorspace.selected_quality_score,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "technical_safety_score",
        expectations.technical_safety_score_max,
        summary.colorspace.technical_safety_score,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "color_fidelity_score",
        expectations.color_fidelity_score_max,
        summary.colorspace.color_fidelity_score,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "memory_color_penalty",
        expectations.memory_color_penalty_max,
        summary.render.colorspace_memory_color_penalty,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "spatial_consistency_penalty",
        expectations.spatial_consistency_penalty_max,
        summary.render.colorspace_spatial_consistency_penalty,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "selected_runner_up_quality_delta",
        expectations.selected_runner_up_quality_delta_min,
        summary.colorspace.selected_runner_up_quality_delta,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "density_monotonicity_score",
        expectations.density_monotonicity_score_min,
        summary.render.colorspace_density_monotonicity_score,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "hue_linearity_score",
        expectations.hue_linearity_score_min,
        summary.render.colorspace_hue_linearity_score,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "saturation_preservation_median_ratio",
        expectations.saturation_preservation_median_ratio_min,
        summary
            .render
            .colorspace_saturation_preservation_median_ratio,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "spatial_neutral_delta_p95",
        expectations.spatial_neutral_delta_p95_max,
        summary.render.colorspace_spatial_neutral_delta_p95,
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "candidate_risk",
        expectations.candidate_risk.as_deref(),
        summary.colorspace.candidate_risk.as_deref(),
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "tone_color_trust_state",
        expectations.tone_color_trust_state.as_deref(),
        summary.colorspace.tone_color_trust_state.as_deref(),
        entry,
    );
    let neutral_safety_rescue = summary.colorspace.neutral_safety_rescue.as_ref();
    push_fixture_suite_expected_bool_issue(
        name,
        "neutral_safety_rescue_applied",
        expectations.neutral_safety_rescue_applied,
        neutral_safety_rescue.and_then(|rescue| rescue.applied),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "neutral_safety_rescue_preserved_ratio_gain",
        expectations.neutral_safety_rescue_preserved_ratio_gain_min,
        neutral_safety_rescue.and_then(|rescue| rescue.preserved_ratio_gain),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "neutral_safety_rescue_midtone_saturation_p95_reduction",
        expectations.neutral_safety_rescue_midtone_saturation_p95_reduction_min,
        neutral_safety_rescue.and_then(|rescue| rescue.midtone_saturation_p95_reduction),
        entry,
    );
    push_fixture_suite_expected_contains_issue(
        name,
        "neutral_safety_rescue_reason",
        expectations
            .neutral_safety_rescue_reason_contains
            .as_deref(),
        neutral_safety_rescue.and_then(|rescue| rescue.reason.as_deref()),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "highlight_chroma_compressed_ratio",
        expectations.highlight_chroma_compressed_ratio_min,
        summary.tone.highlight_chroma_compressed_ratio,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "highlight_chroma_compressed_ratio",
        expectations.highlight_chroma_compressed_ratio_max,
        summary.tone.highlight_chroma_compressed_ratio,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "highlight_neutral_chroma_compressed_ratio",
        expectations.highlight_neutral_chroma_compressed_ratio_max,
        summary.tone.highlight_neutral_chroma_compressed_ratio,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "shadow_chroma_compressed_ratio",
        expectations.shadow_chroma_compressed_ratio_max,
        summary.tone.shadow_chroma_compressed_ratio,
        entry,
    );
    let grain_detail = summary.tone.grain_detail_retention.as_ref();
    push_fixture_suite_expected_bool_issue(
        name,
        "grain_reduction_enabled",
        expectations.grain_reduction_enabled,
        summary.tone.noise_reduction_enabled,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "grain_reduction_applied_ratio",
        expectations.grain_reduction_applied_ratio_min,
        summary.tone.noise_reduction_applied_ratio,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "grain_reduction_structure_excluded_ratio",
        expectations.grain_reduction_structure_excluded_ratio_min,
        summary.tone.noise_reduction_structure_excluded_ratio,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "grain_reduction_flat_luma_p95_reduction_ratio",
        expectations.grain_reduction_flat_luma_p95_reduction_ratio_min,
        summary.tone.noise_reduction_flat_luma_p95_reduction_ratio,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "grain_reduction_flat_chroma_p95_reduction_ratio",
        expectations.grain_reduction_flat_chroma_p95_reduction_ratio_min,
        summary.tone.noise_reduction_flat_chroma_p95_reduction_ratio,
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "grain_detail_review_required",
        expectations.grain_detail_review_required,
        grain_detail.and_then(|detail| detail.review_required),
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "grain_detail_decision_supported",
        expectations.grain_detail_decision_supported,
        grain_detail.and_then(|detail| detail.decision_supported),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "grain_detail_luminance_probe_count",
        expectations
            .grain_detail_luminance_probe_count_min
            .map(|value| value as f64),
        grain_detail
            .and_then(|detail| detail.luminance_probe_count)
            .map(|value| value as f64),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "grain_detail_chroma_probe_count",
        expectations
            .grain_detail_chroma_probe_count_min
            .map(|value| value as f64),
        grain_detail
            .and_then(|detail| detail.chroma_probe_count)
            .map(|value| value as f64),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "grain_detail_luminance_p10_retention",
        expectations.grain_detail_luminance_p10_retention_min,
        grain_detail.and_then(|detail| detail.luminance_p10_retention),
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "grain_detail_chroma_p10_retention",
        expectations.grain_detail_chroma_p10_retention_min,
        grain_detail.and_then(|detail| detail.chroma_p10_retention),
        entry,
    );
    if let Some(minimum) = expectations.post_scale_preserved_ratio_min {
        if !summary
            .colorspace
            .post_scale_preserved_ratio
            .is_some_and(|actual| actual >= minimum)
        {
            entry.issues.push(format!(
                "fixture_suite:{name}:post_scale_preserved_ratio_below_expected"
            ));
        }
    }
    push_fixture_suite_expected_min_issue(
        name,
        "render_luminance_range_p05_p95",
        expectations.render_luminance_range_p05_p95_min,
        summary.tone.render_luminance_range_p05_p95,
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "render_review_status",
        expectations.render_review_status.as_deref(),
        summary.render.render_review_status.as_deref(),
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "render_reviewable",
        expectations.render_reviewable,
        summary.render.render_reviewable,
        entry,
    );
    push_fixture_suite_expected_string_issue(
        name,
        "tone_output_confidence_status",
        expectations.tone_output_confidence_status.as_deref(),
        summary.tone.tone_output_confidence_status.as_deref(),
        entry,
    );
    push_fixture_suite_expected_bool_issue(
        name,
        "tone_output_review_required",
        expectations.tone_output_review_required,
        summary.tone.tone_output_review_required,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "tone_output_evidence_confidence",
        expectations.tone_output_evidence_confidence_min,
        summary.tone.tone_output_evidence_confidence,
        entry,
    );
    push_fixture_suite_expected_min_issue(
        name,
        "render_to_mapped_luminance_range_ratio",
        expectations.render_to_mapped_luminance_range_ratio_min,
        summary.tone.render_to_mapped_luminance_range_ratio,
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "post_chroma_compression_clipped_high_ratio_max",
        expectations.post_chroma_compression_clipped_high_ratio_max,
        max_f64_slice(
            summary
                .tone
                .post_chroma_compression_clipped_high_ratio
                .as_deref(),
        ),
        entry,
    );
    push_fixture_suite_expected_max_issue(
        name,
        "post_chroma_compression_clipped_low_ratio_max",
        expectations.post_chroma_compression_clipped_low_ratio_max,
        max_f64_slice(
            summary
                .tone
                .post_chroma_compression_clipped_low_ratio
                .as_deref(),
        ),
        entry,
    );
    if expectations.reference_patch_evaluation_required
        && summary.colorspace.reference_patch_evaluation.is_none()
    {
        entry.issues.push(format!(
            "fixture_suite:{name}:reference_patch_evaluation_missing"
        ));
    }
    if let Some(minimum) = expectations.reference_patch_count_min {
        let actual = summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.patch_count)
            .unwrap_or(0);
        if actual < minimum {
            entry.issues.push(format!(
                "fixture_suite:{name}:reference_patch_count_below_expected"
            ));
        }
    }
    if let Some(maximum) = expectations.reference_patch_hue_family_regression_count_max {
        let actual = summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .map(|evaluation| evaluation.hue_family_regressions.len());
        if actual.is_none_or(|actual| actual > maximum) {
            entry.issues.push(format!(
                "fixture_suite:{name}:reference_patch_hue_family_regression_count_above_expected"
            ));
        }
    }
    if let Some(expected) = expectations.reference_patch_selected_regresses_image_derived {
        let actual = summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.selected_regresses_image_derived);
        if actual != Some(expected) {
            entry.issues.push(format!(
                "fixture_suite:{name}:reference_patch_selected_regression_mismatch"
            ));
        }
    }
    if let Some(maximum) = expectations.reference_patch_delta_e2000_delta_vs_image_derived_max {
        let actual = summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.delta_e2000_rms_delta_vs_image_derived);
        if actual.is_none_or(|actual| actual > maximum) {
            entry.issues.push(format!(
                "fixture_suite:{name}:reference_patch_delta_e2000_delta_vs_image_derived_above_expected"
            ));
        }
    }
    if let Some(maximum) = expectations.reference_patch_max_delta_vs_image_derived_max {
        let actual = summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.max_error_delta_vs_image_derived);
        if actual.is_none_or(|actual| actual > maximum) {
            entry.issues.push(format!(
                "fixture_suite:{name}:reference_patch_max_delta_vs_image_derived_above_expected"
            ));
        }
    }
    if let Some(maximum) = expectations.reference_patch_delta_e_max_delta_vs_image_derived_max {
        let actual = summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.delta_e_max_delta_vs_image_derived);
        if actual.is_none_or(|actual| actual > maximum) {
            entry.issues.push(format!(
                "fixture_suite:{name}:reference_patch_delta_e_max_delta_vs_image_derived_above_expected"
            ));
        }
    }
    if let Some(maximum) = expectations.reference_patch_delta_e2000_max_delta_vs_image_derived_max {
        let actual = summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.delta_e2000_max_delta_vs_image_derived);
        if actual.is_none_or(|actual| actual > maximum) {
            entry.issues.push(format!(
                "fixture_suite:{name}:reference_patch_delta_e2000_max_delta_vs_image_derived_above_expected"
            ));
        }
    }
    if let Some(maximum) = expectations.reference_patch_rms_delta_e_max {
        let actual = summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.selected_rms_delta_e);
        if !actual.is_some_and(|actual| actual <= maximum) {
            entry.issues.push(format!(
                "fixture_suite:{name}:reference_patch_rms_delta_e_above_expected"
            ));
        }
    }
    if let Some(maximum) = expectations.reference_patch_rms_delta_e2000_max {
        let actual = summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.selected_rms_delta_e2000);
        if !actual.is_some_and(|actual| actual <= maximum) {
            entry.issues.push(format!(
                "fixture_suite:{name}:reference_patch_rms_delta_e2000_above_expected"
            ));
        }
    }
    if expectations.debug_artifacts_required {
        if summary.colorspace.debug_artifacts.is_empty() {
            entry
                .issues
                .push(format!("fixture_suite:{name}:debug_artifacts_missing"));
        }
        entry.issues.extend(
            debug_artifact_issues
                .iter()
                .map(|issue| format!("fixture_suite:{name}:{issue}")),
        );
    }
    for required_kind in &expectations.debug_artifact_kinds_required {
        let required_kind = required_kind.trim();
        if required_kind.is_empty() {
            continue;
        }
        let Some(artifact) = summary
            .colorspace
            .debug_artifacts
            .iter()
            .find(|artifact| artifact.kind == required_kind)
        else {
            entry.issues.push(format!(
                "fixture_suite:{name}:debug_artifact_kind_missing:{required_kind}"
            ));
            continue;
        };
        if !expectations.debug_artifacts_required && artifact.status != "fresh" {
            entry.issues.push(format!(
                "fixture_suite:{name}:colorspace_debug_artifact_invalid:{}:{}",
                artifact.kind, artifact.status
            ));
        }
    }
}

fn fixture_suite_debug_artifact_issues(summary: &ValidationSummary) -> Vec<String> {
    summary
        .colorspace
        .debug_artifacts
        .iter()
        .filter(|artifact| artifact.status != "fresh")
        .map(|artifact| {
            format!(
                "colorspace_debug_artifact_invalid:{}:{}",
                artifact.kind, artifact.status
            )
        })
        .collect()
}

fn fixture_suite_candidate_acceptance_signatures(
    candidates: &[ColorCandidateAcceptanceSummary],
) -> Vec<String> {
    candidates
        .iter()
        .map(|candidate| {
            format!(
                "{}|kind={}|strategy={}|status={}|rank={}|selected={}|eligible={}|rejected={}",
                candidate.candidate.as_deref().unwrap_or("unknown"),
                candidate.candidate_kind.as_deref().unwrap_or("unknown"),
                candidate.mapping_strategy.as_deref().unwrap_or("unknown"),
                candidate.status.as_deref().unwrap_or("unknown"),
                candidate
                    .rank
                    .map(|rank| rank.to_string())
                    .unwrap_or_else(|| "n/a".to_string()),
                candidate
                    .selected
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "n/a".to_string()),
                candidate
                    .eligible_in_color_mode
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "n/a".to_string()),
                candidate
                    .rejected
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "n/a".to_string()),
            )
        })
        .collect()
}

fn validate_fixture_suite_geometry_accuracy_expectations(
    name: &str,
    expectations: &FixtureExpectations,
    summary: &ValidationSummary,
    entry: &mut FixtureSuiteEntry,
) {
    if let Some(expected) = expectations.deskew_correction_degrees_expected {
        match (
            expectations.deskew_correction_tolerance_degrees,
            summary.deskew.correction_degrees,
        ) {
            (Some(tolerance), Some(actual)) if (actual - expected).abs() > tolerance => {
                entry.issues.push(format!(
                    "fixture_suite:{name}:deskew_correction_degrees_outside_tolerance:expected={expected:.6}:tolerance={tolerance:.6}:actual={actual:.6}"
                ));
            }
            (Some(_), None) => entry.issues.push(format!(
                "fixture_suite:{name}:deskew_correction_degrees_missing"
            )),
            (None, _) => entry.issues.push(format!(
                "fixture_suite:{name}:deskew_correction_tolerance_degrees_missing"
            )),
            _ => {}
        }
    }

    for expected in &expectations.border_crop_components_expected {
        let Some(actual) = summary
            .border_crop
            .components
            .iter()
            .find(|component| component.index == Some(expected.component_index))
        else {
            entry.issues.push(format!(
                "fixture_suite:{name}:border_crop_component{}_missing",
                expected.component_index
            ));
            continue;
        };
        for (edge, expected_value, actual_value) in [
            ("top", expected.top_removed, actual.top_removed),
            ("bottom", expected.bottom_removed, actual.bottom_removed),
            ("left", expected.left_removed, actual.left_removed),
            ("right", expected.right_removed, actual.right_removed),
        ] {
            match actual_value {
                Some(actual_value)
                    if actual_value.abs_diff(expected_value) > expected.tolerance_px =>
                {
                    entry.issues.push(format!(
                        "fixture_suite:{name}:border_crop_component{}_{edge}_outside_tolerance:expected={expected_value}:tolerance={}:actual={actual_value}",
                        expected.component_index, expected.tolerance_px
                    ));
                }
                None => entry.issues.push(format!(
                    "fixture_suite:{name}:border_crop_component{}_{edge}_missing",
                    expected.component_index
                )),
                _ => {}
            }
        }
    }
}

fn validate_fixture_suite_orientation_accuracy_expectations(
    name: &str,
    expectations: &FixtureExpectations,
    summary: &ValidationSummary,
    entry: &mut FixtureSuiteEntry,
) {
    for expected in &expectations.orientation_components_expected {
        let Some(actual) = summary
            .input_orientation
            .components
            .iter()
            .find(|component| component.index == Some(expected.component_index))
        else {
            entry.issues.push(format!(
                "fixture_suite:{name}:orientation_component{}_missing",
                expected.component_index
            ));
            continue;
        };
        if actual.tag_value.is_some() != expected.tag_present {
            entry.issues.push(format!(
                "fixture_suite:{name}:orientation_component{}_tag_present_mismatch:expected={}:actual={}",
                expected.component_index,
                expected.tag_present,
                actual.tag_value.is_some()
            ));
        }
        if actual.tag_value != expected.tag_value {
            entry.issues.push(format!(
                "fixture_suite:{name}:orientation_component{}_tag_value_mismatch:expected={:?}:actual={:?}",
                expected.component_index, expected.tag_value, actual.tag_value
            ));
        }
        if actual.transform.as_deref() != Some(expected.transform.as_str()) {
            entry.issues.push(format!(
                "fixture_suite:{name}:orientation_component{}_transform_mismatch:expected={}:actual={}",
                expected.component_index,
                expected.transform,
                actual.transform.as_deref().unwrap_or("missing")
            ));
        }
        if actual.decoded_pixel_sha256.as_deref() != Some(expected.decoded_pixel_sha256.as_str()) {
            entry.issues.push(format!(
                "fixture_suite:{name}:orientation_component{}_decoded_pixel_sha256_mismatch:expected={}:actual={}",
                expected.component_index,
                expected.decoded_pixel_sha256,
                actual.decoded_pixel_sha256.as_deref().unwrap_or("missing")
            ));
        }
        for (field, expected_value, actual_value) in [
            (
                "applied",
                expected.applied.to_string(),
                actual.applied.map(|value| value.to_string()),
            ),
            (
                "source_width",
                expected.source_width.to_string(),
                actual.source_width.map(|value| value.to_string()),
            ),
            (
                "source_height",
                expected.source_height.to_string(),
                actual.source_height.map(|value| value.to_string()),
            ),
            (
                "output_width",
                expected.output_width.to_string(),
                actual.output_width.map(|value| value.to_string()),
            ),
            (
                "output_height",
                expected.output_height.to_string(),
                actual.output_height.map(|value| value.to_string()),
            ),
        ] {
            if actual_value.as_deref() != Some(expected_value.as_str()) {
                entry.issues.push(format!(
                    "fixture_suite:{name}:orientation_component{}_{field}_mismatch:expected={expected_value}:actual={}",
                    expected.component_index,
                    actual_value.as_deref().unwrap_or("missing")
                ));
            }
        }
    }
}

fn push_fixture_suite_expected_string_issue(
    name: &str,
    field: &str,
    expected: Option<&str>,
    actual: Option<&str>,
    entry: &mut FixtureSuiteEntry,
) {
    if expected.is_some() && expected != actual {
        entry
            .issues
            .push(format!("fixture_suite:{name}:{field}_mismatch"));
    }
}

fn push_fixture_suite_expected_bool_issue(
    name: &str,
    field: &str,
    expected: Option<bool>,
    actual: Option<bool>,
    entry: &mut FixtureSuiteEntry,
) {
    if expected.is_some() && expected != actual {
        entry
            .issues
            .push(format!("fixture_suite:{name}:{field}_mismatch"));
    }
}

fn push_fixture_suite_expected_f64_issue(
    name: &str,
    field: &str,
    expected: Option<f64>,
    actual: Option<f64>,
    entry: &mut FixtureSuiteEntry,
) {
    if let Some(expected) = expected {
        if !actual.is_some_and(|actual| (actual - expected).abs() <= 1e-9) {
            entry
                .issues
                .push(format!("fixture_suite:{name}:{field}_mismatch"));
        }
    }
}

fn push_fixture_suite_expected_contains_issue(
    name: &str,
    field: &str,
    expected: Option<&str>,
    actual: Option<&str>,
    entry: &mut FixtureSuiteEntry,
) {
    if let Some(expected) = expected {
        if !actual.is_some_and(|actual| actual.contains(expected)) {
            entry.issues.push(format!(
                "fixture_suite:{name}:{field}_missing_expected_text"
            ));
        }
    }
}

fn push_fixture_suite_expected_max_issue(
    name: &str,
    field: &str,
    maximum: Option<f64>,
    actual: Option<f64>,
    entry: &mut FixtureSuiteEntry,
) {
    if let Some(maximum) = maximum {
        if !actual.is_some_and(|actual| actual <= maximum) {
            entry
                .issues
                .push(format!("fixture_suite:{name}:{field}_above_expected"));
        }
    }
}

fn push_fixture_suite_expected_min_issue(
    name: &str,
    field: &str,
    minimum: Option<f64>,
    actual: Option<f64>,
    entry: &mut FixtureSuiteEntry,
) {
    if let Some(minimum) = minimum {
        if !actual.is_some_and(|actual| actual >= minimum) {
            entry
                .issues
                .push(format!("fixture_suite:{name}:{field}_below_expected"));
        }
    }
}

fn validate_fixture_suite_output_guardrails(
    name: &str,
    expectations: &FixtureExpectations,
    require_reviewable: bool,
    summary: &ValidationSummary,
    entry: &mut FixtureSuiteEntry,
) {
    for issue in delivery_artifact_integrity_issues(&summary.render) {
        entry.issues.push(format!("fixture_suite:{name}:{issue}"));
    }
    append_fixture_suite_render_review_issues(
        name,
        expectations.render_reviewable,
        require_reviewable,
        summary.render.render_review_status.as_deref(),
        summary.render.render_reviewable,
        &mut entry.issues,
    );
}

fn final_render_is_reviewable(
    render_review_status: Option<&str>,
    render_reviewable: Option<bool>,
) -> bool {
    render_review_status == Some("reviewable") && render_reviewable == Some(true)
}

fn final_render_review_evidence_is_consistent(
    render_review_status: Option<&str>,
    render_reviewable: Option<bool>,
) -> bool {
    match (render_review_status, render_reviewable) {
        (Some("reviewable"), Some(true)) => true,
        (Some(status), Some(false)) if status != "reviewable" => true,
        _ => false,
    }
}

fn append_fixture_suite_render_review_issues(
    name: &str,
    expected_render_reviewable: Option<bool>,
    require_reviewable: bool,
    render_review_status: Option<&str>,
    render_reviewable: Option<bool>,
    issues: &mut Vec<String>,
) {
    if !final_render_review_evidence_is_consistent(render_review_status, render_reviewable) {
        issues.push(format!(
            "fixture_suite:{name}:render_reviewability_evidence_missing_or_inconsistent"
        ));
        return;
    }
    if final_render_is_reviewable(render_review_status, render_reviewable) {
        return;
    }
    if require_reviewable {
        issues.push(format!(
            "fixture_suite:{name}:require_reviewable_not_satisfied"
        ));
    } else if expected_render_reviewable != Some(false) {
        issues.push(format!(
            "fixture_suite:{name}:non_reviewable_render_not_explicitly_expected"
        ));
    }
}

fn validate_fixture_suite_calibration_expectations(
    name: &str,
    pipeline_cli: &PipelineCli,
    summary: &ValidationSummary,
    entry: &mut FixtureSuiteEntry,
) {
    let Some(expected_source) = expected_calibration_source(pipeline_cli) else {
        return;
    };

    if summary.colorspace.calibration_status.as_deref() != Some("applied") {
        entry.issues.push(format!(
            "fixture_suite:{name}:declared_calibration_not_applied"
        ));
        return;
    }
    if summary.colorspace.calibration_source.as_deref() != Some(expected_source) {
        entry
            .issues
            .push(format!("fixture_suite:{name}:calibration_source_mismatch"));
    }
    if let Some(expected_scanner) = pipeline_cli.scanner_profile.as_deref() {
        if summary.colorspace.calibration_scanner_profile_id.as_deref() != Some(expected_scanner) {
            entry.issues.push(format!(
                "fixture_suite:{name}:calibration_scanner_profile_mismatch"
            ));
        }
    }
    if let Some(expected_roll) = pipeline_cli.roll_profile.as_deref() {
        if summary.colorspace.calibration_roll_profile_id.as_deref() != Some(expected_roll) {
            entry.issues.push(format!(
                "fixture_suite:{name}:calibration_roll_profile_mismatch"
            ));
        }
    }
    if pipeline_cli.calibration_library.is_some() {
        if let Some(expected_stock) = pipeline_cli.film_stock.as_deref() {
            if summary
                .colorspace
                .calibration_requested_film_stock
                .as_deref()
                != Some(expected_stock)
            {
                entry.issues.push(format!(
                    "fixture_suite:{name}:calibration_film_stock_mismatch"
                ));
            }
        }
    }
}

fn expected_calibration_source(pipeline_cli: &PipelineCli) -> Option<&'static str> {
    if pipeline_cli.calibration_library.is_some()
        || pipeline_cli.scanner_profile.is_some()
        || pipeline_cli.roll_profile.is_some()
        || pipeline_cli.film_stock.is_some()
    {
        Some("calibration_library")
    } else if pipeline_cli.calibration_profile.is_some() {
        Some("external_calibration_profile")
    } else {
        None
    }
}

fn write_fixture_suite_summary_baseline(
    cli: &ValidationCli,
    name: &str,
    fixture: &FixtureEntry,
    summary: &ValidationSummary,
    entry: &mut FixtureSuiteEntry,
) {
    if !cli.write_fixture_suite_baselines {
        return;
    }

    let Some(baseline_path) = &fixture.summary_baseline else {
        entry.summary_baseline_write_status = Some("path_missing".to_string());
        entry.issues.push(format!(
            "fixture_suite:{name}:summary_baseline_write_path_missing"
        ));
        return;
    };

    let baseline_exists = baseline_path.exists();
    if baseline_exists && !cli.overwrite_fixture_suite_baselines {
        entry.summary_baseline_write_status = Some("skipped_exists".to_string());
        return;
    }

    let baseline = tracked_baseline_from_summary(summary);
    let baseline_json =
        serde_json::to_string_pretty(&baseline).expect("summary baseline should serialize");
    if let Err(err) = write_text(baseline_path, &baseline_json) {
        entry.summary_baseline_write_status = Some("failed".to_string());
        entry.issues.push(format!(
            "fixture_suite:{name}:summary_baseline_write_failed"
        ));
        entry.status = "failed".to_string();
        entry.error = Some(err.to_string());
        return;
    }

    entry.summary_baseline_write_status = Some(if baseline_exists {
        "overwritten".to_string()
    } else {
        "written".to_string()
    });
    entry.summary_baseline_written_path = Some(baseline_path.display().to_string());
}

fn apply_fixture_suite_summary_baseline(
    name: &str,
    fixture: &FixtureEntry,
    summary: &mut ValidationSummary,
    entry: &mut FixtureSuiteEntry,
) {
    let Some(compare_summary_path) = &fixture.summary_baseline else {
        entry.issues.push(format!(
            "fixture_suite:{name}:summary_baseline_not_declared"
        ));
        return;
    };

    let contents = match std::fs::read_to_string(compare_summary_path) {
        Ok(contents) => contents,
        Err(err) => {
            entry
                .issues
                .push(format!("fixture_suite:{name}:summary_baseline_read_failed"));
            entry.status = "failed".to_string();
            entry.error = Some(err.to_string());
            return;
        }
    };
    let baseline = match serde_json::from_str::<TrackedValidationBaseline>(&contents) {
        Ok(baseline) => baseline,
        Err(err) => {
            entry.issues.push(format!(
                "fixture_suite:{name}:summary_baseline_parse_failed"
            ));
            entry.status = "failed".to_string();
            entry.error = Some(err.to_string());
            return;
        }
    };
    entry.issues.extend(
        summary_baseline_contract_issues(&baseline)
            .into_iter()
            .map(|field| format!("fixture_suite:{name}:summary_baseline_incomplete:{field}")),
    );
    let comparison = compare_summary_baseline(
        compare_summary_path.to_string_lossy().to_string(),
        &baseline,
        summary,
    );
    entry.summary_baseline_status = Some(comparison.status.clone());
    entry.issues.extend(
        comparison
            .issues
            .iter()
            .map(|issue| format!("fixture_suite:{name}:{issue}")),
    );
    summary.summary_baseline_comparison = Some(comparison);
}

fn summary_baseline_contract_issues(baseline: &TrackedValidationBaseline) -> Vec<&'static str> {
    let mut issues = Vec::new();

    if baseline.stitch.decision.is_none() {
        issues.push("stitch.decision");
    }
    if baseline.render.output_width.is_none() {
        issues.push("render.output_width");
    }
    if baseline.render.output_height.is_none() {
        issues.push("render.output_height");
    }
    if baseline.render.output_color_space.is_none() {
        issues.push("render.output_color_space");
    }
    if baseline
        .render
        .output_file_icc_profile_matches_report
        .is_none()
    {
        issues.push("render.output_file_icc_profile_matches_report");
    }
    if baseline.render.base_estimate_source.is_none() {
        issues.push("render.base_estimate_source");
    }
    if baseline.render.render_input_source.is_none() {
        issues.push("render.render_input_source");
    }
    if baseline.render.colorspace_mapping_strategy.is_none() {
        issues.push("render.colorspace_mapping_strategy");
    }
    if baseline.colorspace.calibration_status.is_none() {
        issues.push("colorspace.calibration_status");
    }
    if baseline.colorspace.selected_candidate.is_none() {
        issues.push("colorspace.selected_candidate");
    }
    if baseline.colorspace.calibration_acceptance_status.is_none() {
        issues.push("colorspace.calibration_acceptance_status");
    }
    if baseline
        .colorspace
        .calibration_color_mapping_applied
        .is_none()
    {
        issues.push("colorspace.calibration_color_mapping_applied");
    }
    if baseline.colorspace.candidate_risk.is_none() {
        issues.push("colorspace.candidate_risk");
    }
    if baseline.colorspace.tone_color_trust_state.is_none() {
        issues.push("colorspace.tone_color_trust_state");
    }
    if baseline.colorspace.selected_quality_score.is_none() {
        issues.push("colorspace.selected_quality_score");
    }
    if baseline.colorspace.technical_safety_score.is_none() {
        issues.push("colorspace.technical_safety_score");
    }
    if baseline.colorspace.color_fidelity_score.is_none() {
        issues.push("colorspace.color_fidelity_score");
    }
    if baseline.colorspace.post_scale_preserved_ratio.is_none() {
        issues.push("colorspace.post_scale_preserved_ratio");
    }
    if baseline.colorspace.neutral_estimate_score.is_none() {
        issues.push("colorspace.neutral_estimate_score");
    }
    if baseline.colorspace.dominant_anchor_accepted.is_none() {
        issues.push("colorspace.dominant_anchor_accepted");
    }
    if baseline.colorspace.channel_anchor_min_count.is_none() {
        issues.push("colorspace.channel_anchor_min_count");
    }
    if baseline.colorspace.weak_anchor_fallback_used.is_none() {
        issues.push("colorspace.weak_anchor_fallback_used");
    }
    if baseline.colorspace.gamut_fallback_used.is_none() {
        issues.push("colorspace.gamut_fallback_used");
    }
    if baseline.colorspace.neutral_trim_applied.is_none() {
        issues.push("colorspace.neutral_trim_applied");
    }
    if baseline
        .colorspace
        .candidate_acceptance_signatures
        .is_empty()
    {
        issues.push("colorspace.candidate_acceptance_signatures");
    }
    if baseline.tone.highlight_chroma_compressed_ratio.is_none() {
        issues.push("tone.highlight_chroma_compressed_ratio");
    }

    issues
}

fn fixture_suite_output_dir(cli: &ValidationCli, name: &str, fixture: &FixtureEntry) -> PathBuf {
    let default_output = PathBuf::from("output/validation/logan");
    if cli.output_dir == default_output {
        fixture
            .output_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("output/validation").join(name))
    } else {
        cli.output_dir.join(name)
    }
}

fn fixture_suite_to_markdown(summary: &FixtureSuiteSummary) -> String {
    let mut out = String::new();
    out.push_str("# Fixture Suite\n\n");
    out.push_str(&format!("- status: `{}`\n", summary.status));
    out.push_str(&format!("- fixtures: `{}`\n", summary.fixture_count));
    out.push_str(&format!("- passed: `{}`\n", summary.passed_count));
    out.push_str(&format!(
        "- review required: `{}`\n",
        summary.review_required_count
    ));
    out.push_str(&format!("- failed: `{}`\n", summary.failed_count));
    out.push_str(&format!(
        "- coverage status: `{}`\n",
        summary.coverage.status
    ));
    out.push_str(&format!(
        "- coverage issues: `{}`\n\n",
        if summary.coverage.issues.is_empty() {
            "none".to_string()
        } else {
            summary.coverage.issues.join(", ")
        }
    ));
    if !summary.coverage.action_items.is_empty() {
        out.push_str("## Coverage Action Items\n\n");
        for action in &summary.coverage.action_items {
            out.push_str(&format!("- `{action}`\n"));
        }
        out.push('\n');
    }
    out.push_str("| Fixture | Status | Coverage | Coverage actions | Baseline | Stitch | Base | Geometry preparation / orientation | Negative reconstruction | Output space | Reference evidence | Render input | Calibration | Candidate | Quality scores | Risk | Tone trust | Tone protection | Dynamic range | Reference patch fit | Debug artifacts | ICC | Stale artifacts | Issues |\n");
    out.push_str("|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|\n");
    for fixture in &summary.fixtures {
        let coverage = if fixture.coverage_validation_ready {
            "ready".to_string()
        } else if fixture.coverage_issues.is_empty() {
            "blocked".to_string()
        } else {
            format!("blocked: {}", fixture.coverage_issues.join(", "))
        };
        let coverage_actions = fixture.coverage_action_items.join(", ");
        let baseline = match (
            fixture.summary_baseline_status.as_deref(),
            fixture.summary_baseline_write_status.as_deref(),
        ) {
            (Some(status), Some(write_status)) => format!("{status}; write {write_status}"),
            (Some(status), None) => status.to_string(),
            (None, Some(write_status)) => format!("write {write_status}"),
            (None, None) => "none".to_string(),
        };
        let stitch = expected_actual_label(
            fixture.expected_stitch_decision.as_deref(),
            fixture.stitch_decision.as_deref(),
        );
        let base = expected_actual_label(
            fixture.expected_base_estimate_source.as_deref(),
            fixture.base_estimate_source.as_deref(),
        );
        let geometry_preparation = fixture_suite_geometry_preparation_label(fixture);
        let negative_reconstruction = fixture_suite_negative_reconstruction_label(fixture);
        let output_space = expected_actual_label(
            fixture.expected_output_color_space.as_deref(),
            fixture.output_color_space.as_deref(),
        );
        let reference_evidence = fixture.reference_evidence.join(", ");
        let render_input = expected_actual_label(
            fixture.expected_render_input_source.as_deref(),
            fixture.render_input_source.as_deref(),
        );
        let render_input_reason = fixture_suite_contains_label(
            "reason",
            fixture.expected_render_input_reason_contains.as_deref(),
            fixture.render_input_reason.as_deref(),
        );
        let render_input = [render_input, render_input_reason]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("; ");
        let calibration = match (&fixture.calibration_status, &fixture.calibration_source) {
            (Some(status), Some(source)) => format!("{status} / {source}"),
            (Some(status), None) => status.clone(),
            _ => String::new(),
        };
        let calibration = if let Some(expected) = &fixture.expected_calibration_source {
            format!("expected {expected}; actual {calibration}")
        } else {
            calibration
        };
        let scanner_profile = match (
            &fixture.expected_calibration_scanner_profile,
            &fixture.calibration_scanner_profile_id,
        ) {
            (Some(expected), Some(actual)) => format!("scanner {actual} (expected {expected})"),
            (Some(expected), None) => format!("scanner missing (expected {expected})"),
            (None, Some(actual)) => format!("scanner {actual}"),
            (None, None) => String::new(),
        };
        let roll_profile = match (
            &fixture.expected_calibration_roll_profile,
            &fixture.calibration_roll_profile_id,
        ) {
            (Some(expected), Some(actual)) => format!("roll {actual} (expected {expected})"),
            (Some(expected), None) => format!("roll missing (expected {expected})"),
            (None, Some(actual)) => format!("roll {actual}"),
            (None, None) => String::new(),
        };
        let calibration_rejections = fixture_suite_required_count_label(
            "rejection details",
            &fixture.expected_calibration_rejection_details_required,
            &fixture.calibration_rejection_details,
        );
        let calibration_diagnostics = fixture_suite_calibration_diagnostic_label(fixture);
        let calibration = [
            calibration,
            scanner_profile,
            roll_profile,
            calibration_diagnostics,
            calibration_rejections,
        ]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("; ");
        let candidate = fixture_suite_candidate_label(fixture);
        let quality_scores = fixture_suite_quality_score_label(fixture);
        let risk = expected_actual_label(
            fixture.expected_candidate_risk.as_deref(),
            fixture.candidate_risk.as_deref(),
        );
        let tone_trust = expected_actual_label(
            fixture.expected_tone_color_trust_state.as_deref(),
            fixture.tone_color_trust_state.as_deref(),
        );
        let tone_protection = fixture_suite_tone_protection_label(fixture);
        let dynamic_range = fixture_suite_dynamic_range_label(fixture);
        let reference_patch_fit = fixture_suite_reference_patch_label(fixture);
        let debug_artifacts = fixture_suite_debug_artifact_label(fixture);
        let icc = fixture
            .output_file_icc_profile_matches_report
            .map(|value| value.to_string())
            .unwrap_or_default();
        let stale = fixture
            .stale_render_artifact_count
            .map(|value| value.to_string())
            .unwrap_or_default();
        let issues = if fixture.issues.is_empty() {
            "none".to_string()
        } else {
            fixture.issues.join(", ")
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            fixture.name,
            fixture.status,
            coverage,
            coverage_actions,
            baseline,
            stitch,
            base,
            geometry_preparation,
            negative_reconstruction,
            output_space,
            reference_evidence,
            render_input,
            calibration,
            candidate,
            quality_scores,
            risk,
            tone_trust,
            tone_protection,
            dynamic_range,
            reference_patch_fit,
            debug_artifacts,
            icc,
            stale,
            issues
        ));
    }
    if !summary.issues.is_empty() {
        out.push_str("\n## Issues\n\n");
        for issue in &summary.issues {
            out.push_str(&format!("- `{issue}`\n"));
        }
    }
    out
}

fn fixture_suite_geometry_preparation_label(fixture: &FixtureSuiteEntry) -> String {
    let evidence = &fixture.geometry_preparation;
    let mut parts = Vec::new();
    push_fixture_suite_expected_value_label(
        &mut parts,
        "deskew all components",
        evidence.expected_all_deskew_components_applied,
        evidence.all_deskew_components_applied,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "deskew retained area",
        evidence.expected_minimum_deskew_retained_area_ratio,
        evidence.minimum_deskew_retained_area_ratio,
    );
    match (
        evidence.expected_deskew_correction_degrees,
        evidence.expected_deskew_correction_tolerance_degrees,
        evidence.deskew_correction_degrees,
    ) {
        (Some(expected), Some(tolerance), Some(actual)) => parts.push(format!(
            "deskew correction expected {expected:.6} +/- {tolerance:.6}; actual {actual:.6}"
        )),
        (Some(expected), Some(tolerance), None) => parts.push(format!(
            "deskew correction expected {expected:.6} +/- {tolerance:.6}; actual missing"
        )),
        (Some(expected), None, actual) => parts.push(format!(
            "deskew correction expected {expected:.6}; tolerance missing; actual {}",
            actual
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "missing".to_string())
        )),
        (None, _, Some(actual)) => parts.push(format!("deskew correction {actual:.6}")),
        _ => {}
    }
    push_fixture_suite_expected_value_label(
        &mut parts,
        "crop all components",
        evidence.expected_all_border_crop_components_cropped,
        evidence.all_border_crop_components_cropped,
    );
    push_fixture_suite_expected_value_label(
        &mut parts,
        "removed edges minimum",
        evidence.expected_minimum_removed_edge_count_per_component,
        evidence.minimum_removed_edge_count_per_component,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "crop retained area",
        evidence.expected_minimum_border_crop_retained_area_ratio,
        evidence.minimum_border_crop_retained_area_ratio,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "crop retained area",
        evidence.expected_maximum_border_crop_retained_area_ratio,
        evidence.maximum_border_crop_retained_area_ratio,
    );
    push_fixture_suite_expected_value_label(
        &mut parts,
        "crop rejected",
        evidence.expected_border_crop_rejected,
        evidence.border_crop_rejected,
    );
    for expected in &evidence.expected_border_crop_components {
        let actual = evidence
            .border_crop_components
            .iter()
            .find(|component| component.index == Some(expected.component_index));
        let actual_label = actual
            .map(|component| {
                format!(
                    "{}/{}/{}/{}",
                    optional_usize(component.top_removed),
                    optional_usize(component.bottom_removed),
                    optional_usize(component.left_removed),
                    optional_usize(component.right_removed)
                )
            })
            .unwrap_or_else(|| "missing".to_string());
        parts.push(format!(
            "crop component {} top/bottom/left/right expected {}/{}/{}/{} +/- {}; actual {actual_label}",
            expected.component_index,
            expected.top_removed,
            expected.bottom_removed,
            expected.left_removed,
            expected.right_removed,
            expected.tolerance_px
        ));
    }
    for expected in &fixture.input_orientation.expected_components {
        let actual = fixture
            .input_orientation
            .components
            .iter()
            .find(|component| component.index == Some(expected.component_index));
        let expected_tag = expected
            .tag_value
            .map(|value| value.to_string())
            .unwrap_or_else(|| "absent".to_string());
        let actual_tag = actual
            .and_then(|component| component.tag_value)
            .map(|value| value.to_string())
            .unwrap_or_else(|| "absent/missing".to_string());
        let actual_transform = actual
            .and_then(|component| component.transform.as_deref())
            .unwrap_or("missing");
        let actual_applied = actual
            .and_then(|component| component.applied)
            .map(|value| value.to_string())
            .unwrap_or_else(|| "missing".to_string());
        let actual_source = actual
            .and_then(|component| {
                component
                    .source_width
                    .zip(component.source_height)
                    .map(|(width, height)| format!("{width}x{height}"))
            })
            .unwrap_or_else(|| "missing".to_string());
        let actual_output = actual
            .and_then(|component| {
                component
                    .output_width
                    .zip(component.output_height)
                    .map(|(width, height)| format!("{width}x{height}"))
            })
            .unwrap_or_else(|| "missing".to_string());
        let actual_decoded_pixel_sha256 = actual
            .and_then(|component| component.decoded_pixel_sha256.as_deref())
            .unwrap_or("missing");
        parts.push(format!(
            "orientation component {} upright approved {}; decoded pixel SHA-256 expected {}; actual {actual_decoded_pixel_sha256}; tag expected {expected_tag}; actual {actual_tag}; transform expected {}; actual {actual_transform}; applied expected {}; actual {actual_applied}; source expected {}x{}; actual {actual_source}; output expected {}x{}; actual {actual_output}",
            expected.component_index,
            expected.upright_approved,
            expected.decoded_pixel_sha256,
            expected.transform,
            expected.applied,
            expected.source_width,
            expected.source_height,
            expected.output_width,
            expected.output_height
        ));
    }
    parts.join("; ")
}

fn fixture_suite_negative_reconstruction_label(fixture: &FixtureSuiteEntry) -> String {
    let evidence = &fixture.negative_reconstruction;
    let mut parts = Vec::new();
    push_fixture_suite_min_label(
        &mut parts,
        "base confidence",
        evidence.expected_base_confidence_min,
        evidence.base_confidence,
    );
    push_fixture_suite_expected_value_label(
        &mut parts,
        "inversion skipped",
        evidence.expected_density_inversion_skipped,
        evidence.density_inversion_skipped,
    );
    for (label, expected, actual) in [
        (
            "response model",
            evidence.expected_response_model.as_deref(),
            evidence.response_model.as_deref(),
        ),
        (
            "response source",
            evidence.expected_response_source.as_deref(),
            evidence.response_source.as_deref(),
        ),
        (
            "crosstalk",
            evidence.expected_crosstalk_model.as_deref(),
            evidence.crosstalk_model.as_deref(),
        ),
        (
            "characteristic curve",
            evidence.expected_characteristic_curve_model.as_deref(),
            evidence.characteristic_curve_model.as_deref(),
        ),
        (
            "measured model",
            evidence.expected_measured_model_id.as_deref(),
            evidence.measured_model_id.as_deref(),
        ),
        (
            "curve interpolation",
            evidence.expected_curve_interpolation.as_deref(),
            evidence.curve_interpolation.as_deref(),
        ),
    ] {
        let value = expected_actual_label(expected, actual);
        if !value.is_empty() {
            parts.push(format!("{label} {value}"));
        }
    }
    push_fixture_suite_expected_value_label(
        &mut parts,
        "response accepted",
        evidence.expected_response_accepted,
        evidence.response_accepted,
    );
    push_fixture_suite_expected_value_label(
        &mut parts,
        "response review",
        evidence.expected_response_review_required,
        evidence.response_review_required,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "measured confidence",
        evidence.expected_measured_confidence_min,
        evidence.measured_confidence,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "held-out dE00 RMS",
        evidence.expected_held_out_delta_e00_rms_max,
        evidence.held_out_delta_e00_rms,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "held-out dE00 maximum",
        evidence.expected_held_out_delta_e00_max,
        evidence.held_out_delta_e00_max,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "unit-slope improvement",
        evidence.expected_held_out_improvement_over_unit_slope_min,
        evidence.held_out_improvement_over_unit_slope,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "density noise gain",
        evidence.expected_maximum_density_noise_gain,
        evidence.maximum_density_noise_gain,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "curve extrapolation",
        evidence.expected_curve_extrapolated_any_ratio_max,
        evidence.curve_extrapolated_any_ratio,
    );
    push_fixture_suite_expected_value_label(
        &mut parts,
        "signed headroom",
        evidence.expected_signed_headroom_preserved,
        evidence.signed_headroom_preserved,
    );
    parts.join("; ")
}

fn fixture_suite_quality_score_label(fixture: &FixtureSuiteEntry) -> String {
    let mut parts = Vec::new();
    push_fixture_suite_max_label(
        &mut parts,
        "selected",
        fixture.expected_selected_quality_score_max,
        fixture.selected_quality_score,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "safety",
        fixture.expected_technical_safety_score_max,
        fixture.technical_safety_score,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "fidelity",
        fixture.expected_color_fidelity_score_max,
        fixture.color_fidelity_score,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "memory",
        fixture.expected_memory_color_penalty_max,
        fixture.memory_color_penalty,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "spatial consistency",
        fixture.expected_spatial_consistency_penalty_max,
        fixture.spatial_consistency_penalty,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "runner-up",
        fixture.expected_selected_runner_up_quality_delta_min,
        fixture.selected_runner_up_quality_delta,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "density",
        fixture.expected_density_monotonicity_score_min,
        fixture.density_monotonicity_score,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "hue",
        fixture.expected_hue_linearity_score_min,
        fixture.hue_linearity_score,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "saturation",
        fixture.expected_saturation_preservation_median_ratio_min,
        fixture.saturation_preservation_median_ratio,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "spatial neutral",
        fixture.expected_spatial_neutral_delta_p95_max,
        fixture.spatial_neutral_delta_p95,
    );
    parts.join("; ")
}

fn fixture_suite_tone_protection_label(fixture: &FixtureSuiteEntry) -> String {
    let mut parts = Vec::new();
    if let Some(mode) = fixture.effective_grain_reduction.as_deref() {
        parts.push(format!(
            "grain config {mode}, strength {}, scale {}",
            fixture
                .effective_grain_strength
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "n/a".to_string()),
            fixture
                .effective_grain_scale
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "n/a".to_string())
        ));
    }
    push_fixture_suite_min_label(
        &mut parts,
        "highlight chroma",
        fixture.expected_highlight_chroma_compressed_ratio_min,
        fixture.highlight_chroma_compressed_ratio,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "highlight chroma",
        fixture.expected_highlight_chroma_compressed_ratio_max,
        fixture.highlight_chroma_compressed_ratio,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "neutral highlight",
        fixture.expected_highlight_neutral_chroma_compressed_ratio_max,
        fixture.highlight_neutral_chroma_compressed_ratio,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "shadow chroma",
        fixture.expected_shadow_chroma_compressed_ratio_max,
        fixture.shadow_chroma_compressed_ratio,
    );
    if let Some(expected) = fixture.expected_grain_reduction_enabled {
        parts.push(format!(
            "grain enabled expected {expected}; actual {}",
            fixture
                .grain_reduction_enabled
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string())
        ));
    }
    push_fixture_suite_min_label(
        &mut parts,
        "grain applied ratio",
        fixture.expected_grain_reduction_applied_ratio_min,
        fixture.grain_reduction_applied_ratio,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "grain exact structure-excluded ratio",
        fixture.expected_grain_reduction_structure_excluded_ratio_min,
        fixture.grain_reduction_structure_excluded_ratio,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "grain flat-luma reduction",
        fixture.expected_grain_reduction_flat_luma_p95_reduction_ratio_min,
        fixture.grain_reduction_flat_luma_p95_reduction_ratio,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "grain flat-chroma reduction",
        fixture.expected_grain_reduction_flat_chroma_p95_reduction_ratio_min,
        fixture.grain_reduction_flat_chroma_p95_reduction_ratio,
    );
    if let Some(expected) = fixture.expected_grain_detail_review_required {
        parts.push(format!(
            "grain review expected {expected}; actual {}",
            fixture
                .grain_detail_review_required
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string())
        ));
    }
    if let Some(expected) = fixture.expected_grain_detail_decision_supported {
        parts.push(format!(
            "grain detail supported expected {expected}; actual {}",
            fixture
                .grain_detail_decision_supported
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string())
        ));
    }
    push_fixture_suite_min_label(
        &mut parts,
        "grain luma probes",
        fixture
            .expected_grain_detail_luminance_probe_count_min
            .map(|value| value as f64),
        fixture
            .grain_detail_luminance_probe_count
            .map(|value| value as f64),
    );
    push_fixture_suite_min_label(
        &mut parts,
        "grain chroma probes",
        fixture
            .expected_grain_detail_chroma_probe_count_min
            .map(|value| value as f64),
        fixture
            .grain_detail_chroma_probe_count
            .map(|value| value as f64),
    );
    push_fixture_suite_min_label(
        &mut parts,
        "grain luma p10",
        fixture.expected_grain_detail_luminance_p10_retention_min,
        fixture.grain_detail_luminance_p10_retention,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "grain chroma p10",
        fixture.expected_grain_detail_chroma_p10_retention_min,
        fixture.grain_detail_chroma_p10_retention,
    );
    parts.join("; ")
}

fn fixture_suite_dynamic_range_label(fixture: &FixtureSuiteEntry) -> String {
    let mut parts = Vec::new();
    push_fixture_suite_min_label(
        &mut parts,
        "preserved",
        fixture.expected_post_scale_preserved_ratio_min,
        fixture.post_scale_preserved_ratio,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "luma p05-p95",
        fixture.expected_render_luminance_range_p05_p95_min,
        fixture.render_luminance_range_p05_p95,
    );
    if let Some(expected) = fixture.expected_render_review_status.as_deref() {
        parts.push(format!(
            "render status expected {expected}; actual {}",
            fixture.render_review_status.as_deref().unwrap_or("n/a")
        ));
    }
    if let Some(expected) = fixture.expected_render_reviewable {
        parts.push(format!(
            "reviewable expected {expected}; actual {}",
            fixture
                .render_reviewable
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string())
        ));
    }
    if let Some(expected) = fixture.expected_tone_output_confidence_status.as_deref() {
        parts.push(format!(
            "tone status expected {expected}; actual {}",
            fixture
                .tone_output_confidence_status
                .as_deref()
                .unwrap_or("n/a")
        ));
    }
    if let Some(expected) = fixture.expected_tone_output_review_required {
        parts.push(format!(
            "tone review expected {expected}; actual {}",
            fixture
                .tone_output_review_required
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string())
        ));
    }
    push_fixture_suite_min_label(
        &mut parts,
        "tone evidence",
        fixture.expected_tone_output_evidence_confidence_min,
        fixture.tone_output_evidence_confidence,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "render:mapped",
        fixture.expected_render_to_mapped_luminance_range_ratio_min,
        fixture.render_to_mapped_luminance_range_ratio,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "high clip",
        fixture.expected_post_chroma_compression_clipped_high_ratio_max,
        fixture.post_chroma_compression_clipped_high_ratio_max,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "low clip",
        fixture.expected_post_chroma_compression_clipped_low_ratio_max,
        fixture.post_chroma_compression_clipped_low_ratio_max,
    );
    parts.join("; ")
}

fn fixture_suite_candidate_label(fixture: &FixtureSuiteEntry) -> String {
    let candidate = match (&fixture.selected_candidate, fixture.selected_quality_score) {
        (Some(candidate), Some(score)) => format!("{candidate} ({score:.6})"),
        (Some(candidate), None) => candidate.clone(),
        _ => String::new(),
    };
    let candidate = if let Some(expected) = &fixture.expected_selected_candidate {
        format!("expected {expected}; actual {candidate}")
    } else {
        candidate
    };
    let mut parts = Vec::new();
    if !candidate.is_empty() {
        parts.push(candidate);
    }
    match (
        fixture.expected_selected_candidate_rank,
        fixture.selected_candidate_rank,
    ) {
        (Some(expected), Some(actual)) => {
            parts.push(format!("rank expected {expected}; actual {actual}"));
        }
        (Some(expected), None) => {
            parts.push(format!("rank expected {expected}; actual missing"));
        }
        (None, Some(actual)) => {
            parts.push(format!("rank {actual}"));
        }
        (None, None) => {}
    }
    if !fixture
        .expected_candidate_acceptance_signatures_required
        .is_empty()
    {
        parts.push(format!(
            "required signatures {}",
            fixture
                .expected_candidate_acceptance_signatures_required
                .len()
        ));
    }
    if !fixture.candidate_acceptance_signatures.is_empty() {
        parts.push(format!(
            "acceptance signatures {}",
            fixture.candidate_acceptance_signatures.len()
        ));
    }
    let mapping_reason = fixture_suite_contains_label(
        "mapping reason",
        fixture.expected_selected_mapping_reason_contains.as_deref(),
        fixture.selected_mapping_reason.as_deref(),
    );
    if !mapping_reason.is_empty() {
        parts.push(mapping_reason);
    }
    let selection_rejections = fixture_suite_required_count_label(
        "selection rejections",
        &fixture.expected_selection_rejections_required,
        &fixture.selection_rejections,
    );
    if !selection_rejections.is_empty() {
        parts.push(selection_rejections);
    }
    match (
        fixture.expected_neutral_safety_rescue_applied,
        fixture.neutral_safety_rescue_applied,
    ) {
        (Some(expected), Some(actual)) => parts.push(format!(
            "neutral rescue expected {expected}; actual {actual}"
        )),
        (Some(expected), None) => parts.push(format!(
            "neutral rescue expected {expected}; actual missing"
        )),
        (None, Some(actual)) => parts.push(format!("neutral rescue {actual}")),
        (None, None) => {}
    }
    push_fixture_suite_min_label(
        &mut parts,
        "neutral rescue preserved gain",
        fixture.expected_neutral_safety_rescue_preserved_ratio_gain_min,
        fixture.neutral_safety_rescue_preserved_ratio_gain,
    );
    push_fixture_suite_min_label(
        &mut parts,
        "neutral rescue saturation reduction",
        fixture.expected_neutral_safety_rescue_midtone_saturation_p95_reduction_min,
        fixture.neutral_safety_rescue_midtone_saturation_p95_reduction,
    );
    let neutral_rescue_reason = fixture_suite_contains_label(
        "neutral rescue reason",
        fixture
            .expected_neutral_safety_rescue_reason_contains
            .as_deref(),
        fixture.neutral_safety_rescue_reason.as_deref(),
    );
    if !neutral_rescue_reason.is_empty() {
        parts.push(neutral_rescue_reason);
    }
    parts.join("; ")
}

fn fixture_suite_contains_label(
    label: &str,
    expected: Option<&str>,
    actual: Option<&str>,
) -> String {
    match (expected, actual) {
        (Some(expected), Some(actual)) => {
            format!("{label} contains {expected}; actual {actual}")
        }
        (Some(expected), None) => {
            format!("{label} contains {expected}; actual missing")
        }
        (None, Some(actual)) => format!("{label} {actual}"),
        (None, None) => String::new(),
    }
}

fn fixture_suite_calibration_diagnostic_label(fixture: &FixtureSuiteEntry) -> String {
    let mut parts = Vec::new();
    push_fixture_suite_min_label(
        &mut parts,
        "confidence",
        fixture.expected_calibration_confidence_min,
        fixture.calibration_confidence,
    );
    push_fixture_suite_max_label(
        &mut parts,
        "condition",
        fixture.expected_calibration_matrix_condition_number_max,
        fixture.calibration_matrix_condition_number,
    );
    parts.join("; ")
}

fn fixture_suite_required_count_label(
    label: &str,
    expected: &[String],
    actual: &[String],
) -> String {
    if !expected.is_empty() {
        return format!(
            "required {label} {}; actual {}",
            expected.len(),
            actual.len()
        );
    }
    if !actual.is_empty() {
        return format!("{label} {}", actual.len());
    }
    String::new()
}

fn push_fixture_suite_expected_value_label<T: std::fmt::Display + Copy>(
    parts: &mut Vec<String>,
    label: &str,
    expected: Option<T>,
    actual: Option<T>,
) {
    match (expected, actual) {
        (Some(expected), Some(actual)) => {
            parts.push(format!("{label} expected {expected}; actual {actual}"));
        }
        (Some(expected), None) => {
            parts.push(format!("{label} expected {expected}; actual missing"));
        }
        (None, Some(actual)) => parts.push(format!("{label} {actual}")),
        (None, None) => {}
    }
}

fn push_fixture_suite_max_label(
    parts: &mut Vec<String>,
    label: &str,
    maximum: Option<f64>,
    actual: Option<f64>,
) {
    match (maximum, actual) {
        (Some(maximum), Some(actual)) => {
            parts.push(format!("{label} max {maximum:.6}; actual {actual:.6}"));
        }
        (Some(maximum), None) => {
            parts.push(format!("{label} max {maximum:.6}; actual missing"));
        }
        (None, Some(actual)) => parts.push(format!("{label} {actual:.6}")),
        (None, None) => {}
    }
}

fn push_fixture_suite_min_label(
    parts: &mut Vec<String>,
    label: &str,
    minimum: Option<f64>,
    actual: Option<f64>,
) {
    match (minimum, actual) {
        (Some(minimum), Some(actual)) => {
            parts.push(format!("{label} min {minimum:.6}; actual {actual:.6}"));
        }
        (Some(minimum), None) => {
            parts.push(format!("{label} min {minimum:.6}; actual missing"));
        }
        (None, Some(actual)) => parts.push(format!("{label} {actual:.6}")),
        (None, None) => {}
    }
}

fn fixture_suite_debug_artifact_label(fixture: &FixtureSuiteEntry) -> String {
    let count = fixture.debug_artifact_count.unwrap_or(0);
    let invalid_count = fixture.debug_artifact_invalid_count.unwrap_or(0);
    let mut parts = Vec::new();
    if fixture.expected_debug_artifacts_required {
        if count == 0 {
            parts.push("required; actual missing".to_string());
        } else {
            parts.push(format!("required; actual {count}; invalid {invalid_count}"));
        }
    } else if count > 0 {
        parts.push(format!("{count}; invalid {invalid_count}"));
    }
    if !fixture.expected_debug_artifact_kinds_required.is_empty() {
        parts.push(format!(
            "required kinds {}",
            fixture.expected_debug_artifact_kinds_required.join(", ")
        ));
        if fixture.debug_artifact_kinds.is_empty() {
            parts.push("actual kinds missing".to_string());
        }
    }
    if !fixture.debug_artifact_kinds.is_empty() {
        parts.push(format!(
            "actual kinds {}",
            fixture.debug_artifact_kinds.join(", ")
        ));
    }
    parts.join("; ")
}

fn fixture_suite_reference_patch_label(fixture: &FixtureSuiteEntry) -> String {
    let mut actual_parts = Vec::new();
    if let Some(count) = fixture.reference_patch_patch_count {
        actual_parts.push(format!("{count} patches"));
    }
    if let Some(delta_e) = fixture.reference_patch_selected_rms_delta_e {
        actual_parts.push(format!("rms dE {delta_e:.6}"));
    }
    if let Some(delta_e2000) = fixture.reference_patch_selected_rms_delta_e2000 {
        actual_parts.push(format!("rms dE2000 {delta_e2000:.6}"));
    }
    if let Some(delta_e2000_delta) = fixture.reference_patch_delta_e2000_delta_vs_image_derived {
        actual_parts.push(format!("rms dE2000 vs image {delta_e2000_delta:.6}"));
    }
    if let Some(max_delta) = fixture.reference_patch_max_delta_vs_image_derived {
        actual_parts.push(format!("max XYZ vs image {max_delta:.6}"));
    }
    if let Some(max_delta) = fixture.reference_patch_delta_e_max_delta_vs_image_derived {
        actual_parts.push(format!("max dE vs image {max_delta:.6}"));
    }
    if let Some(max_delta) = fixture.reference_patch_delta_e2000_max_delta_vs_image_derived {
        actual_parts.push(format!("max dE2000 vs image {max_delta:.6}"));
    }
    if let Some(regresses) = fixture.reference_patch_selected_regresses_image_derived {
        actual_parts.push(format!("selected regresses {regresses}"));
    }
    if !fixture.reference_patch_hue_family_regressions.is_empty() {
        actual_parts.push(format!(
            "hue regressions {}",
            fixture.reference_patch_hue_family_regressions.join(", ")
        ));
    }
    let actual =
        if actual_parts.is_empty() && fixture.reference_patch_evaluation_present == Some(false) {
            "missing".to_string()
        } else {
            actual_parts.join("; ")
        };
    let actual = if fixture.expected_reference_patch_evaluation_required {
        if actual.is_empty() {
            "required; actual missing".to_string()
        } else {
            format!("required; actual {actual}")
        }
    } else {
        actual
    };
    let mut parts = Vec::new();
    if let Some(minimum) = fixture.expected_reference_patch_count_min {
        parts.push(format!("min patches {minimum}"));
    }
    if let Some(maximum) = fixture.expected_reference_patch_hue_family_regression_count_max {
        parts.push(format!("max hue regressions {maximum}"));
    }
    if let Some(expected) = fixture.expected_reference_patch_selected_regresses_image_derived {
        parts.push(format!("selected regresses {expected}"));
    }
    if let Some(maximum) = fixture.expected_reference_patch_delta_e2000_delta_vs_image_derived_max {
        parts.push(format!("max rms dE2000 vs image {maximum:.6}"));
    }
    if let Some(maximum) = fixture.expected_reference_patch_max_delta_vs_image_derived_max {
        parts.push(format!("max XYZ vs image {maximum:.6}"));
    }
    if let Some(maximum) = fixture.expected_reference_patch_delta_e_max_delta_vs_image_derived_max {
        parts.push(format!("max dE vs image {maximum:.6}"));
    }
    if let Some(maximum) =
        fixture.expected_reference_patch_delta_e2000_max_delta_vs_image_derived_max
    {
        parts.push(format!("max dE2000 vs image {maximum:.6}"));
    }
    if let Some(maximum) = fixture.expected_reference_patch_rms_delta_e_max {
        parts.push(format!("max rms dE {maximum:.6}"));
    }
    if let Some(maximum) = fixture.expected_reference_patch_rms_delta_e2000_max {
        parts.push(format!("max rms dE2000 {maximum:.6}"));
    }
    if actual.is_empty() && !parts.is_empty() {
        parts.push("actual missing".to_string());
    } else if !actual.is_empty() {
        parts.push(actual);
    }
    parts.join("; ")
}

fn expected_actual_label(expected: Option<&str>, actual: Option<&str>) -> String {
    match (expected, actual) {
        (Some(expected), Some(actual)) => format!("expected {expected}; actual {actual}"),
        (Some(expected), None) => format!("expected {expected}; actual missing"),
        (None, Some(actual)) => actual.to_string(),
        (None, None) => String::new(),
    }
}

fn resolve_inputs(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    match (&cli.component1, &cli.component2) {
        (Some(component1), Some(component2)) => {
            return Ok(vec![component1.clone(), component2.clone()]);
        }
        (Some(component1), None) => return Ok(vec![component1.clone()]),
        (None, Some(_)) => return Err("--component2 requires --component1".into()),
        (None, None) => {}
    }

    let Some(fixture) = fixtures.get(&cli.fixture) else {
        return Err(format!(
            "unknown fixture `{}`; pass --component1 with an optional --component2, use --report, or inspect --list-fixtures",
            cli.fixture
        )
        .into());
    };
    let inputs = fixture.pipeline_inputs();
    let missing = inputs
        .iter()
        .enumerate()
        .filter(|(_, path)| !path.exists())
        .map(|(index, path)| format!("component{}={}", index + 1, path.display()))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "fixture `{}` is missing {}; use a direct --component1 with an optional --component2 override, or summarize an existing --report",
            cli.fixture,
            missing.join(", ")
        )
        .into());
    }

    Ok(inputs)
}

fn resolve_orientation_review_inputs(
    cli: &ValidationCli,
    fixtures: &BTreeMap<String, FixtureEntry>,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    resolve_inputs(cli, fixtures)
}

fn print_summary_table(summary: &scanstitch::validation::ValidationSummary) {
    println!("{:<12} {:<34} Value", "Area", "Field");
    println!("{:-<12} {:-<34} {:-<1}", "", "", "");
    print_row(
        "report",
        "generated_at",
        summary.report.generated_at.as_deref().unwrap_or(""),
    );
    print_row(
        "render",
        "output_path",
        summary.render.output_path.as_deref().unwrap_or(""),
    );
    print_row(
        "render",
        "dimensions",
        &format!(
            "{}x{}",
            summary.render.output_width.unwrap_or(0),
            summary.render.output_height.unwrap_or(0)
        ),
    );
    print_row(
        "render",
        "review_status",
        summary.render.render_review_status.as_deref().unwrap_or(""),
    );
    print_row(
        "render",
        "reviewable",
        &summary
            .render
            .render_reviewable
            .map(|value| value.to_string())
            .unwrap_or_default(),
    );
    print_row(
        "stitch",
        "decision",
        summary.stitch.decision.as_deref().unwrap_or(""),
    );
    print_row(
        "colorspace",
        "calibration_record_status",
        summary
            .colorspace
            .calibration_status
            .as_deref()
            .unwrap_or(""),
    );
    let calibration_color_mapping = summary
        .colorspace
        .calibration_color_mapping_application
        .as_ref();
    print_row(
        "colorspace",
        "calibration_color_mapping_status",
        calibration_color_mapping
            .and_then(|application| application.selection_status.as_deref())
            .unwrap_or(""),
    );
    print_row(
        "colorspace",
        "calibration_color_mapping_applied",
        &calibration_color_mapping
            .and_then(|application| application.applied)
            .map(|value| value.to_string())
            .unwrap_or_default(),
    );
    print_row(
        "colorspace",
        "calibration_color_mapping_selected_candidate",
        calibration_color_mapping
            .and_then(|application| application.selected_candidate.as_deref())
            .unwrap_or(""),
    );
    print_row(
        "colorspace",
        "calibration_color_mapping_preferred_candidate",
        calibration_color_mapping
            .and_then(|application| application.preferred_candidate.as_deref())
            .unwrap_or(""),
    );
    print_row(
        "colorspace",
        "calibration_color_mapping_consistency_issues",
        &if summary.diagnostic_consistency_issues.is_empty() {
            "none".to_string()
        } else {
            summary.diagnostic_consistency_issues.join(",")
        },
    );
    print_row(
        "colorspace",
        "mapping_strategy",
        summary.colorspace.mapping_strategy.as_deref().unwrap_or(""),
    );
    print_row(
        "colorspace",
        "selected_candidate",
        summary
            .colorspace
            .selected_candidate
            .as_deref()
            .unwrap_or(""),
    );
    print_row(
        "colorspace",
        "selected_quality_score",
        &summary
            .colorspace
            .selected_quality_score
            .map(|value| format!("{value:.6}"))
            .unwrap_or_default(),
    );
    print_row(
        "colorspace",
        "calibration_acceptance",
        summary
            .colorspace
            .calibration_acceptance
            .as_ref()
            .and_then(|acceptance| acceptance.status.as_deref())
            .unwrap_or(""),
    );
    print_row(
        "colorspace",
        "neutral_estimate_quality",
        &summary
            .colorspace
            .neutral_estimate_quality
            .as_ref()
            .and_then(|quality| quality.score)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_default(),
    );
    print_row(
        "colorspace",
        "post_scale_preserved_ratio",
        &summary
            .colorspace
            .post_scale_preserved_ratio
            .map(|value| format!("{value:.6}"))
            .unwrap_or_default(),
    );
    print_row(
        "colorspace",
        "neutral_trim_applied",
        &summary
            .colorspace
            .neutral_trim_applied
            .map(|value| value.to_string())
            .unwrap_or_default(),
    );
    print_row(
        "colorspace",
        "tone_color_trust_state",
        summary
            .colorspace
            .tone_color_trust_state
            .as_deref()
            .unwrap_or(""),
    );
    print_row(
        "tone",
        "color_trust_state",
        summary.tone.color_trust_state.as_deref().unwrap_or(""),
    );
    print_row(
        "tone",
        "tone_output_confidence_status",
        summary
            .tone
            .tone_output_confidence_status
            .as_deref()
            .unwrap_or(""),
    );
    print_row(
        "tone",
        "tone_output_review_required",
        &summary
            .tone
            .tone_output_review_required
            .map(|value| value.to_string())
            .unwrap_or_default(),
    );
    print_row(
        "tone",
        "tone_output_evidence_confidence",
        &summary
            .tone
            .tone_output_evidence_confidence
            .map(|value| format!("{value:.6}"))
            .unwrap_or_default(),
    );
    print_row(
        "tone",
        "render_to_mapped_range_ratio",
        &summary
            .tone
            .render_to_mapped_luminance_range_ratio
            .map(|value| format!("{value:.6}"))
            .unwrap_or_default(),
    );
    print_row(
        "tone",
        "luma_residual_p95",
        &summary
            .tone
            .high_frequency_grain
            .as_ref()
            .and_then(|grain| grain.luma_residual_p95)
            .map(|value| format!("{value:.6}"))
            .unwrap_or_default(),
    );
}

fn print_row(area: &str, field: &str, value: &str) {
    println!("{area:<12} {field:<34} {value}");
}

fn selected_failures(issues: &[String], selectors: &[String]) -> Vec<String> {
    if selectors.is_empty() {
        return Vec::new();
    }
    issues
        .iter()
        .filter(|issue| {
            selectors
                .iter()
                .any(|selector| issue_matches(issue, selector))
        })
        .cloned()
        .collect()
}

fn issue_matches(issue: &str, selector: &str) -> bool {
    let selector = selector.trim().to_ascii_lowercase();
    match selector.as_str() {
        "any" => true,
        "stale-output" => issue.contains("stale_render_artifacts"),
        "dimensions" => issue.contains("output_dimensions"),
        "stitch-change" => issue.contains("stitch_decision"),
        "base-source" => issue.contains("base_estimate_source"),
        "render-input-change" => issue.contains("render_input_source"),
        "colorspace-strategy" => issue.contains("colorspace_mapping_strategy"),
        "colorspace-quality" | "quality" => issue.contains("colorspace_quality"),
        "calibration" => issue.contains("calibration_"),
        "gamut" => issue.contains("gamut") || issue.contains("preservation"),
        "grain" => {
            issue.contains("residual_p95")
                || issue.contains("grain_detail_")
                || issue.contains("noise_reduction_")
        }
        "tone" => issue.contains("chroma_compression") || issue.contains("tone_output_"),
        "debug-artifact" | "debug-artifacts" => issue.contains("debug_artifact"),
        "summary-baseline" => issue.contains("summary_baseline_"),
        "fixture-coverage" => {
            issue.starts_with("fixture_coverage_")
                || (issue.contains(':') && !issue.starts_with("fixture_suite:"))
        }
        "fixture-suite" => issue.starts_with("fixture_suite:"),
        "roll-inventory" => issue.starts_with("roll_inventory:"),
        "roll-suite" => issue.starts_with("roll_suite:"),
        exact => issue == exact,
    }
}

fn write_text(path: &Path, contents: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    atomic_file::write_bytes(path, contents.as_bytes())?;
    Ok(())
}

fn write_rgb_image(path: &Path, image: &RgbImage) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let format = image::ImageFormat::from_path(path)?;
    let mut staged = atomic_file::AtomicFile::new(path)?;
    image.write_to(staged.file_mut(), format)?;
    staged.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        append_fixture_suite_render_review_issues, append_roll_suite_render_review_issues,
        compare_roll_suites, final_render_is_reviewable, issue_matches,
    };

    #[test]
    fn fixture_suite_render_review_guard_requires_explicit_diagnostic_intent() {
        assert!(final_render_is_reviewable(Some("reviewable"), Some(true)));
        assert!(!final_render_is_reviewable(
            Some("review_required_color"),
            Some(false)
        ));

        let mut issues = Vec::new();
        append_fixture_suite_render_review_issues(
            "healthy",
            None,
            false,
            Some("reviewable"),
            Some(true),
            &mut issues,
        );
        assert!(issues.is_empty());

        append_fixture_suite_render_review_issues(
            "undeclared-diagnostic",
            None,
            false,
            Some("review_required_color"),
            Some(false),
            &mut issues,
        );
        assert_eq!(
            issues,
            ["fixture_suite:undeclared-diagnostic:non_reviewable_render_not_explicitly_expected"]
        );

        issues.clear();
        append_fixture_suite_render_review_issues(
            "declared-diagnostic",
            Some(false),
            false,
            Some("review_required_color"),
            Some(false),
            &mut issues,
        );
        assert!(issues.is_empty());

        append_fixture_suite_render_review_issues(
            "production-gate",
            Some(false),
            true,
            Some("review_required_color"),
            Some(false),
            &mut issues,
        );
        assert_eq!(
            issues,
            ["fixture_suite:production-gate:require_reviewable_not_satisfied"]
        );

        issues.clear();
        append_fixture_suite_render_review_issues(
            "inconsistent",
            None,
            false,
            Some("reviewable"),
            Some(false),
            &mut issues,
        );
        assert_eq!(
            issues,
            ["fixture_suite:inconsistent:render_reviewability_evidence_missing_or_inconsistent"]
        );
    }

    #[test]
    fn grain_and_tone_failure_selectors_include_evidence_regressions() {
        assert!(issue_matches(
            "grain_detail_chroma_retention_regressed",
            "grain"
        ));
        assert!(issue_matches("luma_residual_p95_ratio_changed", "grain"));
        assert!(issue_matches(
            "roll_suite_compare:noise_reduction_enabled_count_changed",
            "grain"
        ));
        assert!(!issue_matches("post_scale_preservation_regressed", "grain"));
        assert!(issue_matches(
            "tone_output_evidence_confidence_regressed",
            "tone"
        ));
        assert!(issue_matches("tone_output_high_clipping_regressed", "tone"));
        assert!(issue_matches(
            "roll_suite:frame-a:tone_output_review_required",
            "tone"
        ));
        assert!(issue_matches(
            "roll_suite_compare:frame-a:tone_output_status_regressed",
            "tone"
        ));

        let mut issues = Vec::new();
        append_roll_suite_render_review_issues("healthy", Some(true), Some(false), &mut issues);
        assert!(issues.is_empty());
        append_roll_suite_render_review_issues("blocked", Some(false), Some(true), &mut issues);
        assert_eq!(
            issues,
            [
                "roll_suite:blocked:render_review_not_supported",
                "roll_suite:blocked:tone_output_review_required",
            ]
        );
        issues.clear();
        append_roll_suite_render_review_issues("missing", None, None, &mut issues);
        assert_eq!(
            issues,
            [
                "roll_suite:missing:render_review_evidence_missing",
                "roll_suite:missing:tone_output_evidence_missing",
            ]
        );
    }

    #[test]
    fn roll_suite_comparison_flags_grain_and_tone_regressions_without_penalizing_legacy_absence() {
        let baseline = serde_json::json!({
            "frame_count": 2,
            "review_required_count": 0,
            "failed_count": 0,
            "quality": {
                "noise_reduction_enabled_count": 2,
                "grain_detail_evaluated_count": 2,
                "grain_detail_decision_supported_count": 2,
                "grain_detail_review_required_count": 0,
                "grain_detail_luminance_supported_count": 2,
                "grain_detail_chroma_supported_count": 2,
                "grain_detail_luminance_p10_retention_mean": 0.96,
                "grain_detail_luminance_p10_retention_min": 0.94,
                "grain_detail_chroma_p10_retention_mean": 0.95,
                "grain_detail_chroma_p10_retention_min": 0.93
            },
            "frames": [
                {
                    "name": "frame-a.tif",
                    "status": "passed",
                    "render_review_status": "reviewable",
                    "render_reviewable": true,
                    "tone_output_confidence_status": "supported_render_tonal_distribution",
                    "tone_output_review_required": false,
                    "tone_output_evidence_confidence": 1.0,
                    "render_to_mapped_luminance_range_ratio": 1.0,
                    "maximum_post_tone_high_clip_ratio": 0.001,
                    "maximum_post_tone_low_clip_ratio": 0.001,
                    "candidate_risk": "safe",
                    "grain_detail_review_required": false,
                    "grain_detail_decision_supported": true,
                    "grain_detail_luminance_supported": true,
                    "grain_detail_chroma_supported": true,
                    "grain_detail_luminance_p10_retention": 0.96,
                    "grain_detail_chroma_p10_retention": 0.95
                },
                {
                    "name": "frame-b.tif",
                    "status": "passed",
                    "candidate_risk": "safe",
                    "grain_detail_review_required": false,
                    "grain_detail_decision_supported": true,
                    "grain_detail_luminance_supported": true,
                    "grain_detail_chroma_supported": true,
                    "grain_detail_luminance_p10_retention": 0.94,
                    "grain_detail_chroma_p10_retention": 0.93
                }
            ]
        });
        let current = serde_json::json!({
            "frame_count": 2,
            "review_required_count": 1,
            "failed_count": 0,
            "quality": {
                "noise_reduction_enabled_count": 2,
                "grain_detail_evaluated_count": 2,
                "grain_detail_decision_supported_count": 1,
                "grain_detail_review_required_count": 1,
                "grain_detail_luminance_supported_count": 1,
                "grain_detail_chroma_supported_count": 1,
                "grain_detail_luminance_p10_retention_mean": 0.70,
                "grain_detail_luminance_p10_retention_min": 0.68,
                "grain_detail_chroma_p10_retention_mean": 0.72,
                "grain_detail_chroma_p10_retention_min": 0.69
            },
            "frames": [
                {
                    "name": "frame-a.tif",
                    "status": "review_required",
                    "render_review_status": "review_required_tone_output",
                    "render_reviewable": false,
                    "tone_output_confidence_status": "review_required_collapsed_render_luminance_range",
                    "tone_output_review_required": true,
                    "tone_output_evidence_confidence": 0.0,
                    "render_to_mapped_luminance_range_ratio": 0.02,
                    "maximum_post_tone_high_clip_ratio": 0.70,
                    "maximum_post_tone_low_clip_ratio": 0.60,
                    "candidate_risk": "safe",
                    "grain_detail_review_required": true,
                    "grain_detail_decision_supported": true,
                    "grain_detail_luminance_supported": true,
                    "grain_detail_chroma_supported": true,
                    "grain_detail_luminance_p10_retention": 0.70,
                    "grain_detail_chroma_p10_retention": 0.72
                },
                {
                    "name": "frame-b.tif",
                    "status": "passed",
                    "candidate_risk": "safe",
                    "grain_detail_review_required": false,
                    "grain_detail_decision_supported": false,
                    "grain_detail_luminance_supported": false,
                    "grain_detail_chroma_supported": false
                }
            ]
        });

        let comparison = compare_roll_suites("baseline.json".to_string(), &baseline, &current);
        assert_eq!(comparison.status, "review_required");
        for expected in [
            "roll_suite_compare:grain_detail_review_required_count_increased",
            "roll_suite_compare:grain_detail_decision_supported_count_dropped",
            "roll_suite_compare:grain_detail_luminance_supported_count_dropped",
            "roll_suite_compare:grain_detail_chroma_supported_count_dropped",
            "roll_suite_compare:grain_detail_luminance_p10_mean_dropped",
            "roll_suite_compare:grain_detail_chroma_p10_min_dropped",
            "roll_suite_compare:frame-a:render_reviewability_lost",
            "roll_suite_compare:frame-a:tone_output_status_regressed",
            "roll_suite_compare:frame-a:tone_output_review_newly_required",
            "roll_suite_compare:frame-a:tone_output_evidence_confidence_dropped",
            "roll_suite_compare:frame-a:tone_output_range_retention_dropped",
            "roll_suite_compare:frame-a:tone_output_high_clipping_increased",
            "roll_suite_compare:frame-a:tone_output_low_clipping_increased",
            "roll_suite_compare:frame-a:grain_detail_review_newly_required",
            "roll_suite_compare:frame-a:grain_detail_luminance_p10_dropped",
            "roll_suite_compare:frame-a:grain_detail_chroma_p10_dropped",
            "roll_suite_compare:frame-b:grain_detail_decision_support_lost",
            "roll_suite_compare:frame-b:grain_detail_luminance_support_lost",
            "roll_suite_compare:frame-b:grain_detail_chroma_support_lost",
        ] {
            assert!(
                comparison.issues.iter().any(|issue| issue == expected),
                "missing {expected}: {:?}",
                comparison.issues
            );
        }
        let frame_a = comparison
            .frames
            .iter()
            .find(|frame| frame.name == "frame-a.tif")
            .expect("frame-a comparison");
        assert!(frame_a.render_review_status_changed);
        assert!(frame_a.render_reviewable_changed);
        assert!(frame_a.tone_output_confidence_status_changed);
        assert!(frame_a.tone_output_review_required_changed);
        assert_eq!(frame_a.tone_output_evidence_confidence_delta, Some(-1.0));
        assert_eq!(
            frame_a.tone_output_render_to_mapped_luminance_range_ratio_delta,
            Some(-0.98)
        );

        let legacy = serde_json::json!({
            "frame_count": 2,
            "review_required_count": 1,
            "failed_count": 0,
            "quality": {},
            "frames": [
                {"name": "frame-a.tif", "status": "review_required", "candidate_risk": "safe"},
                {"name": "frame-b.tif", "status": "passed", "candidate_risk": "safe"}
            ]
        });
        let legacy_comparison = compare_roll_suites("legacy.json".to_string(), &legacy, &current);
        assert_eq!(legacy_comparison.status, "comparable");
        assert!(legacy_comparison.issues.is_empty());
    }
}

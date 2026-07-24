use crate::base_detect;
use crate::constants::{MAX_14BIT, MAX_16BIT};
use crate::cv_adapter::opencv_match;
use crate::report::PhaseReport;
use crate::tiff_io;
use nalgebra::{Matrix3, Matrix4, SMatrix, SVector, Vector3, Vector4};
use ndarray::{s, Array2, Array3, ArrayView2};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

const SIGNATURE_SCREEN_LIMIT: usize = 18;
const SIGNATURE_SCREEN_NEIGHBORHOOD: usize = 2;
const EXPANDED_SIGNATURE_SCREEN_LIMIT: usize = 4;
const EXPANDED_SIGNATURE_SCREEN_NEIGHBORHOOD: usize = 1;
const CANDIDATE_OVERLAP_DIVERSITY_PX: usize = 8;
const MAX_TRANSLATION_CANDIDATES_TO_VALIDATE: usize = 6;
const MAX_REPORTED_TRANSLATION_CANDIDATES: usize = 5;
const MAX_REPORTED_EVALUATED_CANDIDATES: usize = 6;
const HOMOGRAPHY_MIN_TRAINING_INLIERS: usize = 12;
const HOMOGRAPHY_MAX_HELD_OUT_MEDIAN_ERROR_PX: f64 = 2.5;
const HOMOGRAPHY_MAX_HELD_OUT_P95_ERROR_PX: f64 = 2.75;
const MIN_PLAUSIBLE_OVERLAP_FRACTION: f64 = 0.03;
const MAX_PLAUSIBLE_OVERLAP_FRACTION: f64 = 0.85;
const NARROW_SEAM_RESCUE_MAX_OVERLAP_FRACTION: f64 = 0.12;
const MIN_INFORMATIVE_LOCAL_WINDOWS: usize = 8;
const MIN_INFORMATIVE_WINDOW_RATIO: f64 = 0.25;
const INFORMATIVE_MODEL_INLIER_RESIDUAL_PX: f64 = 2.0;
const MAX_INFORMATIVE_MODEL_P95_RESIDUAL_PX: f64 = 3.0;
const MIN_NARROW_SEAM_SIGNATURE_SCORE: f64 = 0.25;
const MIN_NARROW_SEAM_SEAM_SUPPORT_SCORE: f64 = 0.65;
const MIN_NARROW_SEAM_ROW_SUPPORT_RATIO: f64 = 0.60;
const MIN_NARROW_SEAM_COLUMN_SUPPORT_RATIO: f64 = 0.60;
const NARROW_SEAM_OBJECTIVE_MARGIN: f64 = 0.15;
const SEAM_EXPOSURE_MIN_VALID_SAMPLES: usize = 512;
const SEAM_EXPOSURE_MIN_VALID_SAMPLE_RATIO: f64 = 0.02;
const SEAM_EXPOSURE_MIN_WINDOWS: usize = 4;
const SEAM_EXPOSURE_MIN_SCORE: f64 = 0.035;
const SEAM_EXPOSURE_MIN_IMPROVEMENT: f64 = 0.015;
const SEAM_EXPOSURE_MIN_GAIN: f64 = 0.80;
const SEAM_EXPOSURE_MAX_GAIN: f64 = 1.25;
const SEAM_EXPOSURE_SCALAR_RGB_SPREAD: f64 = 1.04;
const SEAM_EXPOSURE_MAX_RGB_SPREAD: f64 = 1.30;
const SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO: f64 = 0.65;
const SEAM_EXPOSURE_PER_CHANNEL_MARGIN: f64 = 0.006;
const SEAM_EXPOSURE_MAX_CLIP_INCREASE: f64 = 0.0015;
const SEAM_EXPOSURE_MIN_HELD_OUT_WINDOWS: usize = 2;
const SEAM_EXPOSURE_MIN_HELD_OUT_SAMPLES: usize = 256;
const SEAM_EXPOSURE_MIN_AFFINE_WINDOWS_PER_SPLIT: usize = 3;
const SEAM_EXPOSURE_MAX_OFFSET_NORMALIZED: f64 = 0.05;
const SEAM_EXPOSURE_MIN_OFFSET_NORMALIZED: f64 = 0.002;
const SEAM_EXPOSURE_MIN_AFFINE_OVER_GAIN_IMPROVEMENT: f64 = 0.006;
const SEAM_EXPOSURE_MIN_AFFINE_OVER_GAIN_FRACTION: f64 = 0.12;
const SEAM_EXPOSURE_MIN_SPATIAL_ROWS_PER_SPLIT: usize = 3;
const SEAM_EXPOSURE_MIN_SPATIAL_ENDPOINT_RATIO: f64 = 1.03;
const SEAM_EXPOSURE_MIN_SPATIAL_OVER_CONSTANT_IMPROVEMENT: f64 = 0.004;
const SEAM_EXPOSURE_MIN_SPATIAL_OVER_CONSTANT_FRACTION: f64 = 0.10;
const SEAM_EXPOSURE_MAX_SPATIAL_SLOPE_DELTA: f64 = 0.035;
const SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO: f64 = 2.0 / 3.0;
const SEAM_EXPOSURE_MIN_SPATIAL_AFFINE_ROWS_PER_SPLIT: usize = 4;
const SEAM_EXPOSURE_MIN_SPATIAL_OFFSET_ENDPOINT_DELTA_NORMALIZED: f64 = 0.003;
const SEAM_EXPOSURE_MAX_SPATIAL_OFFSET_SLOPE_DELTA_NORMALIZED: f64 = 0.008;
const SEAM_EXPOSURE_MAX_SPATIAL_OFFSET_CENTER_DELTA_NORMALIZED: f64 = 0.012;
const SEAM_EXPOSURE_MIN_SPATIAL_AFFINE_OVER_SIMPLER_IMPROVEMENT: f64 = 0.0025;
const SEAM_EXPOSURE_MIN_SPATIAL_AFFINE_OVER_SIMPLER_FRACTION: f64 = 0.20;
const SEAM_EXPOSURE_MIN_SPATIAL_2D_ROWS_PER_SPLIT: usize = 4;
const SEAM_EXPOSURE_MIN_SPATIAL_2D_COLUMNS_PER_SPLIT: usize = 3;
const SEAM_EXPOSURE_MAX_SPATIAL_CENTER_GAIN_LOG_DELTA: f64 = 0.035;
const SEAM_EXPOSURE_MIN_QUADRATIC_WINDOWS_PER_SPLIT: usize = 18;
const SEAM_EXPOSURE_MIN_QUADRATIC_ROWS_PER_SPLIT: usize = 6;
const SEAM_EXPOSURE_MIN_QUADRATIC_COLUMNS_PER_SPLIT: usize = 5;
const SEAM_EXPOSURE_QUADRATIC_REGULARIZATION: f64 = 1e-3;
const SEAM_EXPOSURE_MAX_QUADRATIC_DESIGN_CONDITION: f64 = 10_000.0;
const SEAM_EXPOSURE_MIN_QUADRATIC_GAIN_SIGNAL: f64 = 0.018;
const SEAM_EXPOSURE_MIN_QUADRATIC_OFFSET_SIGNAL_NORMALIZED: f64 = 0.0025;
const SEAM_EXPOSURE_MAX_QUADRATIC_COEFFICIENT_DELTA: f64 = 0.05;
const SEAM_EXPOSURE_MAX_QUADRATIC_GAIN_FIELD_DELTA: f64 = 0.055;
const SEAM_EXPOSURE_MAX_QUADRATIC_OFFSET_FIELD_DELTA_NORMALIZED: f64 = 0.012;
const SEAM_EXPOSURE_MIN_QUADRATIC_OVER_SIMPLER_IMPROVEMENT: f64 = 0.0025;
const SEAM_EXPOSURE_MIN_QUADRATIC_OVER_SIMPLER_FRACTION: f64 = 0.20;
const SEAM_EXPOSURE_QUADRATIC_EVALUATION_GRID_SIZE: usize = 9;
const SEAM_DETAIL_GRID_ROWS: usize = 6;
const SEAM_DETAIL_GRID_COLUMNS: usize = 6;
const SEAM_DETAIL_RADII_PX: [usize; 3] = [1, 2, 4];
const SEAM_DETAIL_MIN_PIXELS_PER_WINDOW: usize = 24;
const SEAM_DETAIL_MIN_WINDOWS_PER_SPLIT: usize = 4;
const SEAM_DETAIL_MIN_ENERGY: f64 = 0.0015;
const SEAM_DETAIL_ENERGY_REGULARIZATION: f64 = 0.0005;
const SEAM_DETAIL_REVIEW_RATIO: f64 = 2.0;
const SEAM_DETAIL_MIN_DIRECTION_CONSISTENCY: f64 = 0.75;
const SEAM_DETAIL_MAX_CROSS_SPLIT_RATIO: f64 = 1.35;
const SEAM_DETAIL_MIN_REPEATED_SCALES: usize = 2;
const SEAM_BLEND_MAX_GRADIENT_AMPLIFICATION: f64 = 1.05;
const SPATIAL_FIELD_CORNERS: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
const AFFINE_MAX_ROTATION_DEG: f64 = 3.0;
const AFFINE_COARSE_ROTATION_STEP_DEG: f64 = 0.10;
const AFFINE_FINE_ROTATION_STEP_DEG: f64 = 0.02;
const AFFINE_ROTATION_BOUNDARY_MARGIN_DEG: f64 = AFFINE_FINE_ROTATION_STEP_DEG * 0.51;
const AFFINE_MAX_ALIGNMENT_SAMPLES: usize = 36_000;
const AFFINE_MIN_ALIGNMENT_SAMPLES: usize = 768;
const AFFINE_MIN_HELD_OUT_NCC_IMPROVEMENT: f64 = 0.006;
const AFFINE_MIN_HELD_OUT_ERROR_REDUCTION: f64 = 0.08;
const AFFINE_AUTO_LOCAL_SLOPE_TRIGGER_PX: f64 = 0.70;
const AFFINE_MIN_SIGNIFICANT_ROTATION_DEG: f64 = 0.035;
const AFFINE_MIN_SIGNIFICANT_LINEAR_DEFORMATION: f64 = 0.0010;
const AFFINE_MIN_SIGNIFICANT_TRANSLATION_REFINEMENT_PX: f64 = 0.30;
const HOMOGRAPHY_MIN_SPATIAL_SPLIT_IMPROVEMENT: f64 = 0.003;
const HOMOGRAPHY_MIN_MEAN_SPATIAL_IMPROVEMENT: f64 = 0.006;
const HOMOGRAPHY_MIN_SPATIAL_SPLIT_ERROR_REDUCTION: f64 = 0.04;
const HOMOGRAPHY_MIN_MEAN_SPATIAL_ERROR_REDUCTION: f64 = 0.08;
const HOMOGRAPHY_MIN_SIGNIFICANT_DEVIATION_PX: f64 = 0.40;

/// Runtime transform preference for stitching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformMode {
    Auto,
    Translation,
    Affine,
    Homography,
}

impl TransformMode {
    pub fn from_cli(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "translation" => Ok(Self::Translation),
            "affine" => Ok(Self::Affine),
            "homography" => Ok(Self::Homography),
            other => Err(format!(
                "invalid --transform value '{}'; expected auto, translation, affine, or homography",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Translation => "translation",
            Self::Affine => "affine",
            Self::Homography => "homography",
        }
    }
}

/// Configuration for the stitching algorithm.
#[derive(Debug, Clone)]
pub struct StitchConfig {
    pub max_overlap: usize,
    pub min_ncc_score: f64,
    pub min_overlap: usize,
    pub blend_width: usize,
    /// Expected vertical placement of component 2 relative to component 1. The pipeline derives
    /// this from independently removed top scanner borders so cropping cannot move a valid seam
    /// outside the mechanical-drift search window.
    pub expected_y_offset: i32,
    /// Maximum vertical drift in pixels around `expected_y_offset`.
    pub max_y_offset: i32,
    pub transform_mode: TransformMode,
    pub use_opencv: bool,
    pub input_bit_depth: u8,
    pub target_width: Option<usize>,
    pub target_height: Option<usize>,
    pub min_objective_score: f64,
    pub min_local_inlier_ratio: f64,
    pub max_local_median_residual: f64,
    pub max_local_p95_residual: f64,
    pub min_validation_windows: usize,
    pub debug_dir: Option<PathBuf>,
}

impl Default for StitchConfig {
    fn default() -> Self {
        Self {
            max_overlap: 300,
            min_ncc_score: 0.6,
            min_overlap: 20,
            blend_width: 50,
            expected_y_offset: 0,
            max_y_offset: 15,
            transform_mode: TransformMode::Auto,
            use_opencv: false,
            input_bit_depth: 14,
            target_width: None,
            target_height: None,
            min_objective_score: 0.60,
            min_local_inlier_ratio: 0.55,
            max_local_median_residual: 1.25,
            max_local_p95_residual: 3.0,
            min_validation_windows: 4,
            debug_dir: None,
        }
    }
}

/// Result of a stitch operation.
pub struct StitchResult {
    pub result: Option<Array3<u16>>,
    pub x_offset: i32,
    pub y_offset: i32,
    pub ncc_score: f64,
    pub order: StitchOrder,
    pub report: PhaseReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct SequenceOrderingEdge {
    /// One-based component index in the caller's input sequence.
    pub from_component: usize,
    /// One-based component index in the caller's input sequence.
    pub to_component: usize,
    pub accepted: bool,
    pub objective_score: f64,
    pub global_ncc_score: f64,
    pub overlap_width_px: Option<usize>,
    pub expected_vertical_offset_px: i32,
    pub vertical_offset_px: Option<i32>,
    pub rejection_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SequenceOrderingDiagnostics {
    pub method: &'static str,
    /// One-based component indices, ordered from the inferred left edge to right edge.
    pub inferred_order: Vec<usize>,
    pub accepted_adjacent_edges: usize,
    pub required_adjacent_edges: usize,
    pub total_objective_score: f64,
    pub all_adjacencies_validated: bool,
    pub pairwise_edges: Vec<SequenceOrderingEdge>,
}

pub struct StitchSequenceResult {
    pub result: Option<Array3<u16>>,
    /// Zero-based component indices in inferred left-to-right order.
    pub order: Vec<usize>,
    pub report: PhaseReport,
}

#[derive(Debug, Clone)]
struct SequenceOrderState {
    accepted_edges: usize,
    objective_score: f64,
    path: Vec<usize>,
}

/// Detected stitch ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StitchOrder {
    LeftRight,
    RightLeft,
    NoStitch,
}

impl StitchOrder {
    fn label(self) -> &'static str {
        match self {
            Self::LeftRight => "[1|2]",
            Self::RightLeft => "[2|1]",
            Self::NoStitch => "[skip]",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CropInfo {
    x_start: usize,
    y_start: usize,
    width: usize,
    height: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct HomographyFeatureValidationMetrics {
    pub accepted: bool,
    pub reason: String,
    pub partition_method: String,
    pub match_count: usize,
    pub training_match_count: usize,
    pub held_out_match_count: usize,
    pub training_spatial_cell_count: usize,
    pub held_out_spatial_cell_count: usize,
    pub training_inlier_count: usize,
    pub training_inlier_ratio: f64,
    pub training_median_error_px: f64,
    pub training_p95_error_px: f64,
    pub held_out_inlier_count: usize,
    pub held_out_inlier_ratio: f64,
    pub held_out_median_error_px: f64,
    pub held_out_p95_error_px: f64,
    pub reverse_validation_inlier_count: usize,
    pub reverse_validation_inlier_ratio: f64,
    pub reverse_validation_median_error_px: f64,
    pub reverse_validation_p95_error_px: f64,
    pub cross_fit_max_disagreement_px: f64,
}

impl From<&opencv_match::MatchResult> for HomographyFeatureValidationMetrics {
    fn from(result: &opencv_match::MatchResult) -> Self {
        Self {
            accepted: result.disjoint_validation_passed,
            reason: result.validation_reason.clone(),
            partition_method: result.partition_method.to_string(),
            match_count: result.match_count,
            training_match_count: result.training_match_count,
            held_out_match_count: result.held_out_match_count,
            training_spatial_cell_count: result.training_spatial_cell_count,
            held_out_spatial_cell_count: result.held_out_spatial_cell_count,
            training_inlier_count: result.inliers,
            training_inlier_ratio: result.training_inlier_ratio,
            training_median_error_px: result.training_median_error,
            training_p95_error_px: result.training_p95_error,
            held_out_inlier_count: result.held_out_inlier_count,
            held_out_inlier_ratio: result.held_out_inlier_ratio,
            held_out_median_error_px: result.median_error,
            held_out_p95_error_px: result.p95_error,
            reverse_validation_inlier_count: result.reverse_validation_inlier_count,
            reverse_validation_inlier_ratio: result.reverse_validation_inlier_ratio,
            reverse_validation_median_error_px: result.reverse_validation_median_error,
            reverse_validation_p95_error_px: result.reverse_validation_p95_error,
            cross_fit_max_disagreement_px: result.cross_fit_max_disagreement_px,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StitchValidationMetrics {
    pub overlap_width_px: usize,
    pub comparison_height_px: usize,
    pub global_ncc_score: f64,
    pub signature_score: f64,
    pub overlap_support_score: f64,
    pub vertical_offset_plausibility_score: f64,
    pub plausibility_score: f64,
    pub evidence_score: f64,
    pub prior_weight: f64,
    pub local_consistency_score: f64,
    pub local_windows_evaluated: usize,
    pub local_inlier_count: usize,
    pub local_inlier_ratio: f64,
    pub median_local_ncc: f64,
    pub informative_ncc_threshold: f64,
    pub informative_window_count: usize,
    pub informative_window_ratio: f64,
    pub informative_inlier_count: usize,
    pub informative_inlier_ratio: f64,
    pub row_support_ratio: f64,
    pub column_support_ratio: f64,
    pub seam_support_score: f64,
    pub median_residual_px: f64,
    pub p95_residual_px: f64,
    pub median_model_residual_px: f64,
    pub p95_model_residual_px: f64,
    pub mean_dx_px: f64,
    pub mean_dy_px: f64,
    /// Weighted local displacement plane `[intercept, x_slope, y_slope]` in overlap-normalized
    /// coordinates. Non-zero slopes are direct evidence that translation alone is insufficient.
    pub local_dx_plane_px: [f64; 3],
    pub local_dy_plane_px: [f64; 3],
    pub overlap_fraction_left: f64,
    pub overlap_fraction_right: f64,
    pub plausibility_ok: bool,
    pub plausibility_reason: Option<String>,
    pub homography_inliers: Option<usize>,
    pub homography_median_error: Option<f64>,
    pub homography_p95_error: Option<f64>,
    pub homography_feature_validation: Option<HomographyFeatureValidationMetrics>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StitchSearchCandidate {
    pub rank: usize,
    pub search_band: String,
    pub overlap_width: usize,
    pub x_offset: usize,
    pub vertical_offset: i32,
    pub global_ncc_score: f64,
    pub signature_score: f64,
    pub overlap_fraction_left: f64,
    pub overlap_fraction_right: f64,
    pub overlap_support_score: f64,
    pub vertical_offset_plausibility_score: f64,
    pub plausibility_score: f64,
    pub correspondence_score: f64,
    pub prior_weight: f64,
    pub search_score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StitchEvaluatedCandidate {
    pub validation_rank: usize,
    pub search_rank: Option<usize>,
    pub search_band: String,
    pub overlap_width: usize,
    pub x_offset: usize,
    pub vertical_offset: i32,
    pub objective_score: f64,
    pub accepted: bool,
    pub acceptance_reason: Option<String>,
    pub rejection_reason: Option<String>,
    pub global_ncc_score: f64,
    pub signature_score: f64,
    pub overlap_support_score: f64,
    pub vertical_offset_plausibility_score: f64,
    pub plausibility_score: f64,
    pub evidence_score: f64,
    pub prior_weight: f64,
    pub local_consistency_score: f64,
    pub informative_window_count: usize,
    pub informative_window_ratio: f64,
    pub informative_inlier_ratio: f64,
    pub row_support_ratio: f64,
    pub column_support_ratio: f64,
    pub seam_support_score: f64,
    pub local_inlier_ratio: f64,
    pub median_local_ncc: f64,
    pub median_residual_px: f64,
    pub p95_residual_px: f64,
    pub p95_model_residual_px: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StitchSearchDiagnostics {
    pub screening_strategy: String,
    pub screened_overlap_count: usize,
    pub expanded_screened_overlap_count: usize,
    pub evaluated_overlap_count: usize,
    pub exhaustive_fallback_used: bool,
    pub expanded_overlap_search_used: bool,
    pub primary_max_overlap_considered: usize,
    pub max_overlap_considered: usize,
    pub selected_candidate_rank: Option<usize>,
    pub selected_candidate_search_score: Option<f64>,
    pub search_score_gap_to_runner_up: Option<f64>,
    pub selection_reason: Option<String>,
    pub top_candidates: Vec<StitchSearchCandidate>,
    pub evaluated_candidates: Vec<StitchEvaluatedCandidate>,
}

#[derive(Debug, Clone, Serialize)]
struct SpatialPhotometric2dDiagnostics {
    coordinate_system: &'static str,
    corner_order: [&'static str; 4],
    training_window_count: usize,
    held_out_window_count: usize,
    distinct_training_rows: usize,
    distinct_held_out_rows: usize,
    distinct_training_columns: usize,
    distinct_held_out_columns: usize,
    gain_consistent_window_ratio: f64,
    gain_slope_agreement_ratio: f64,
    gain_horizontal_slope_agreement_ratio: f64,
    gain_center_log_delta: f64,
    gain_offset_consistent_window_ratio: f64,
    gain_offset_slope_agreement_ratio: f64,
    gain_offset_horizontal_slope_agreement_ratio: f64,
    gain_offset_center_gain_log_delta: f64,
    gain_offset_center_offset_delta_normalized: f64,
    estimated_gain_center_rgb: [f64; 3],
    estimated_gain_log_slope_x_rgb: [f64; 3],
    estimated_gain_log_slope_y_rgb: [f64; 3],
    estimated_gain_corners_rgb: [[f64; 3]; 4],
    estimated_gain_offset_center_gain_rgb: [f64; 3],
    estimated_gain_offset_center_offset_rgb: [f64; 3],
    estimated_gain_offset_log_slope_x_rgb: [f64; 3],
    estimated_gain_offset_log_slope_y_rgb: [f64; 3],
    estimated_gain_offset_slope_x_rgb: [f64; 3],
    estimated_gain_offset_slope_y_rgb: [f64; 3],
    estimated_gain_offset_gain_corners_rgb: [[f64; 3]; 4],
    estimated_gain_offset_offset_corners_rgb: [[f64; 3]; 4],
    held_out_gain_seam_score: f64,
    held_out_gain_offset_seam_score: f64,
    gain_best_simpler_model: String,
    gain_improvement_over_best_simpler: f64,
    gain_offset_best_simpler_model: String,
    gain_offset_improvement_over_best_simpler: f64,
    gain_accepted: bool,
    gain_offset_accepted: bool,
    gain_rejection_reason: String,
    gain_offset_rejection_reason: String,
    quadratic: SpatialPhotometricQuadraticDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
struct SpatialPhotometricQuadraticDiagnostics {
    basis: [&'static str; 6],
    evaluation_grid_size: usize,
    regularization_lambda: f64,
    minimum_windows_per_split: usize,
    minimum_rows_per_split: usize,
    minimum_columns_per_split: usize,
    training_window_count: usize,
    held_out_window_count: usize,
    distinct_training_rows: usize,
    distinct_held_out_rows: usize,
    distinct_training_columns: usize,
    distinct_held_out_columns: usize,
    gain_design_condition_number: f64,
    held_out_gain_design_condition_number: f64,
    gain_consistent_window_ratio: f64,
    gain_curvature_coefficient_agreement_ratio: f64,
    gain_max_validation_field_log_delta: f64,
    gain_curvature_signal: f64,
    estimated_gain_log_quadratic_xx_rgb: [f64; 3],
    estimated_gain_log_quadratic_xy_rgb: [f64; 3],
    estimated_gain_log_quadratic_yy_rgb: [f64; 3],
    gain_grid_min_rgb: [f64; 3],
    gain_grid_max_rgb: [f64; 3],
    gain_offset_design_condition_number: f64,
    held_out_gain_offset_design_condition_number: f64,
    gain_offset_consistent_window_ratio: f64,
    gain_offset_curvature_coefficient_agreement_ratio: f64,
    gain_offset_max_validation_gain_field_log_delta: f64,
    gain_offset_max_validation_offset_field_delta_normalized: f64,
    gain_offset_curvature_signal: f64,
    estimated_gain_offset_log_quadratic_xx_rgb: [f64; 3],
    estimated_gain_offset_log_quadratic_xy_rgb: [f64; 3],
    estimated_gain_offset_log_quadratic_yy_rgb: [f64; 3],
    estimated_gain_offset_quadratic_xx_rgb: [f64; 3],
    estimated_gain_offset_quadratic_xy_rgb: [f64; 3],
    estimated_gain_offset_quadratic_yy_rgb: [f64; 3],
    gain_offset_grid_gain_min_rgb: [f64; 3],
    gain_offset_grid_gain_max_rgb: [f64; 3],
    gain_offset_grid_offset_abs_max_normalized: f64,
    held_out_gain_seam_score: f64,
    held_out_gain_offset_seam_score: f64,
    gain_best_simpler_model: String,
    gain_improvement_over_best_simpler: f64,
    gain_offset_best_simpler_model: String,
    gain_offset_improvement_over_best_simpler: f64,
    gain_accepted: bool,
    gain_offset_accepted: bool,
    gain_rejection_reason: String,
    gain_offset_rejection_reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct SeamExposureCorrectionDiagnostics {
    mode: String,
    model: String,
    applied: bool,
    reason: String,
    sample_count: usize,
    valid_sample_ratio: f64,
    gain_rgb: [f64; 3],
    gain_luma: f64,
    spatial_gain_log_slope_x_rgb: [f64; 3],
    spatial_gain_log_slope_x_luma: f64,
    spatial_gain_log_slope_y_rgb: [f64; 3],
    spatial_gain_log_slope_y_luma: f64,
    spatial_gain_log_quadratic_xx_rgb: [f64; 3],
    spatial_gain_log_quadratic_xx_luma: f64,
    spatial_gain_log_quadratic_xy_rgb: [f64; 3],
    spatial_gain_log_quadratic_xy_luma: f64,
    spatial_gain_log_quadratic_yy_rgb: [f64; 3],
    spatial_gain_log_quadratic_yy_luma: f64,
    spatial_gain_top_rgb: [f64; 3],
    spatial_gain_bottom_rgb: [f64; 3],
    spatial_gain_corners_rgb: [[f64; 3]; 4],
    spatial_offset_slope_x_rgb: [f64; 3],
    spatial_offset_slope_x_luma: f64,
    spatial_offset_slope_x_rgb_normalized: [f64; 3],
    spatial_offset_slope_y_rgb: [f64; 3],
    spatial_offset_slope_y_luma: f64,
    spatial_offset_slope_y_rgb_normalized: [f64; 3],
    spatial_offset_quadratic_xx_rgb: [f64; 3],
    spatial_offset_quadratic_xx_luma: f64,
    spatial_offset_quadratic_xx_rgb_normalized: [f64; 3],
    spatial_offset_quadratic_xy_rgb: [f64; 3],
    spatial_offset_quadratic_xy_luma: f64,
    spatial_offset_quadratic_xy_rgb_normalized: [f64; 3],
    spatial_offset_quadratic_yy_rgb: [f64; 3],
    spatial_offset_quadratic_yy_luma: f64,
    spatial_offset_quadratic_yy_rgb_normalized: [f64; 3],
    spatial_offset_top_rgb: [f64; 3],
    spatial_offset_bottom_rgb: [f64; 3],
    spatial_offset_top_rgb_normalized: [f64; 3],
    spatial_offset_bottom_rgb_normalized: [f64; 3],
    spatial_offset_corners_rgb: [[f64; 3]; 4],
    spatial_offset_corners_rgb_normalized: [[f64; 3]; 4],
    offset_rgb: [f64; 3],
    offset_luma: f64,
    offset_rgb_normalized: [f64; 3],
    delta_luma_before: f64,
    delta_luma_after: f64,
    delta_rgb_before: [f64; 3],
    delta_rgb_after: [f64; 3],
    seam_score_before: f64,
    seam_score_after: f64,
    clipped_high_before: [f64; 3],
    clipped_high_after: [f64; 3],
    clipped_low_before: [f64; 3],
    clipped_low_after: [f64; 3],
    estimated_gain_rgb: [f64; 3],
    estimated_gain_luma: f64,
    estimated_offset_rgb: [f64; 3],
    estimated_offset_luma: f64,
    estimated_spatial_gain_log_slope_y_rgb: [f64; 3],
    estimated_spatial_gain_log_slope_y_luma: f64,
    estimated_spatial_gain_top_rgb: [f64; 3],
    estimated_spatial_gain_bottom_rgb: [f64; 3],
    estimated_spatial_affine_center_gain_rgb: [f64; 3],
    estimated_spatial_affine_center_offset_rgb: [f64; 3],
    estimated_spatial_affine_gain_log_slope_y_rgb: [f64; 3],
    estimated_spatial_affine_gain_log_slope_y_luma: f64,
    estimated_spatial_affine_offset_slope_y_rgb: [f64; 3],
    estimated_spatial_affine_offset_slope_y_luma: f64,
    estimated_spatial_affine_gain_top_rgb: [f64; 3],
    estimated_spatial_affine_gain_bottom_rgb: [f64; 3],
    estimated_spatial_affine_offset_top_rgb: [f64; 3],
    estimated_spatial_affine_offset_bottom_rgb: [f64; 3],
    candidate_gain_rgb: [f64; 3],
    candidate_gain_luma: f64,
    candidate_offset_rgb: [f64; 3],
    candidate_offset_luma: f64,
    candidate_delta_luma_after: f64,
    candidate_delta_rgb_after: [f64; 3],
    candidate_seam_score_after: f64,
    candidate_clipped_high_after: [f64; 3],
    candidate_clipped_low_after: [f64; 3],
    window_count: usize,
    consistent_window_ratio: f64,
    per_channel_gain_used: bool,
    training_window_count: usize,
    held_out_window_count: usize,
    training_sample_count: usize,
    held_out_sample_count: usize,
    gain_offset_training_window_count: usize,
    gain_offset_held_out_window_count: usize,
    gain_offset_consistent_window_ratio: f64,
    spatial_training_window_count: usize,
    spatial_held_out_window_count: usize,
    spatial_distinct_training_rows: usize,
    spatial_distinct_held_out_rows: usize,
    spatial_consistent_window_ratio: f64,
    spatial_slope_agreement_ratio: f64,
    spatial_affine_training_window_count: usize,
    spatial_affine_held_out_window_count: usize,
    spatial_affine_distinct_training_rows: usize,
    spatial_affine_distinct_held_out_rows: usize,
    spatial_affine_consistent_window_ratio: f64,
    spatial_affine_slope_agreement_ratio: f64,
    spatial_affine_center_offset_delta_normalized: f64,
    held_out_identity_seam_score: f64,
    held_out_gain_seam_score: f64,
    held_out_gain_offset_seam_score: f64,
    held_out_spatial_gain_seam_score: f64,
    held_out_spatial_gain_offset_seam_score: f64,
    held_out_selected_seam_score: f64,
    held_out_improvement_over_identity: f64,
    held_out_improvement_over_gain: f64,
    held_out_spatial_improvement_over_best_constant: f64,
    held_out_spatial_gain_offset_improvement_over_best_simpler: f64,
    held_out_validation_passed: bool,
    gain_offset_rejection_reason: String,
    spatial_rejection_reason: String,
    spatial_gain_offset_rejection_reason: String,
    spatial_2d_validation: SpatialPhotometric2dDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
struct SeamBlendDiagnostics {
    mode: String,
    applied: bool,
    reason: String,
    overlap_width_px: usize,
    overlap_height_px: usize,
    transition_width_px: usize,
    pyramid_levels: usize,
    pyramid_radii_px: Vec<usize>,
    processing_roi_x: [usize; 2],
    seam_path_min_x: usize,
    seam_path_max_x: usize,
    seam_path_mean_x: f64,
    seam_path_mean_normalized_cost: f64,
    seam_path_p95_normalized_cost: f64,
    overlap_mean_abs_difference: f64,
    overlap_p95_abs_difference: f64,
    output_seam_gradient_p95: f64,
    source_seam_gradient_p95: f64,
    output_to_source_seam_gradient_ratio: f64,
    detail_consistency: SeamDetailConsistencyDiagnostics,
    review_required: bool,
    review_reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct SeamDetailConsistencyDiagnostics {
    method: &'static str,
    evaluated: bool,
    reason: String,
    grid_rows: usize,
    grid_columns: usize,
    minimum_energy: f64,
    energy_regularization: f64,
    review_ratio_threshold: f64,
    minimum_direction_consistency: f64,
    maximum_cross_split_ratio: f64,
    minimum_repeated_scale_count: usize,
    supported_scale_count: usize,
    decision_supported: bool,
    imbalanced_scale_count: usize,
    maximum_symmetric_energy_ratio: f64,
    review_required: bool,
    review_reason: String,
    scales: Vec<SeamDetailScaleDiagnostics>,
}

#[derive(Debug, Clone, Serialize)]
struct SeamDetailScaleDiagnostics {
    radius_px: usize,
    training_window_count: usize,
    held_out_window_count: usize,
    training_left_energy_median: f64,
    training_right_energy_median: f64,
    held_out_left_energy_median: f64,
    held_out_right_energy_median: f64,
    training_right_to_left_ratio: f64,
    held_out_right_to_left_ratio: f64,
    training_symmetric_energy_ratio: f64,
    held_out_symmetric_energy_ratio: f64,
    training_direction_consistent_window_ratio: f64,
    held_out_direction_consistent_window_ratio: f64,
    cross_split_log_ratio_delta: f64,
    cross_split_direction_agrees: bool,
    evidence_supported: bool,
    imbalanced: bool,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct NativeAffineDiagnostics {
    attempted: bool,
    accepted: bool,
    reason: String,
    model: &'static str,
    trigger: String,
    training_baseline_ncc: f64,
    training_candidate_ncc: f64,
    held_out_baseline_ncc: f64,
    held_out_candidate_ncc: f64,
    held_out_ncc_improvement: f64,
    held_out_error_reduction: f64,
    training_sample_count: usize,
    held_out_sample_count: usize,
    local_correspondence_count: usize,
    local_inlier_count: usize,
    rotation_deg: f64,
    rotation_search_limit_deg: f64,
    rotation_search_boundary_hit: bool,
    scale_x: f64,
    scale_y: f64,
    shear_cosine: f64,
    translation_refinement_px: [f64; 2],
    transform_right_to_left: [[f64; 3]; 3],
    interpolation: &'static str,
}

#[derive(Debug, Clone)]
struct NativeAffineEstimate {
    diagnostics: NativeAffineDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
struct HomographySpatialValidationDiagnostics {
    accepted: bool,
    reason: String,
    method: &'static str,
    baseline_ncc: [f64; 2],
    candidate_ncc: [f64; 2],
    ncc_improvement: [f64; 2],
    mean_ncc_improvement: f64,
    registration_error_reduction: [f64; 2],
    mean_registration_error_reduction: f64,
    sample_count: [usize; 2],
    maximum_deviation_from_translation_px: f64,
    transform_right_to_left: [[f64; 3]; 3],
    model_selection_limit: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct AffineCorrespondence {
    source: [f64; 2],
    destination: [f64; 2],
    score: f64,
}

#[derive(Debug, Clone, Copy)]
struct Rectangle {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

#[derive(Debug)]
struct ProjectivePairComposition {
    image: Array3<u16>,
    right_to_canvas: [[f64; 3]; 3],
    left_origin: [usize; 2],
    canvas_width: usize,
    canvas_height: usize,
    valid_union_pixel_count: usize,
    canvas_void_pixel_count: usize,
    overlap_pixel_count: usize,
    overlap_outside_multiband_pixel_count: usize,
    multiband_overlap_rectangle: Rectangle,
    seam_exposure_correction: SeamExposureCorrectionDiagnostics,
    seam_blend: SeamBlendDiagnostics,
}

#[derive(Debug, Clone, Copy)]
struct ProjectedCanvasGeometry {
    left_origin: [usize; 2],
    canvas_width: usize,
    canvas_height: usize,
    right_to_canvas: [[f64; 3]; 3],
}

#[derive(Debug, Clone, Copy)]
struct SeamSample {
    left: [f64; 3],
    right: [f64; 3],
    x_normalized: f64,
    y_normalized: f64,
}

#[derive(Debug, Clone, Copy)]
struct SeamWindowEstimate {
    gain_rgb: [f64; 3],
    gain_luma: f64,
    affine_gain_rgb: [f64; 3],
    offset_rgb: [f64; 3],
    affine_reliable: bool,
}

#[derive(Debug, Clone)]
struct SeamWindow {
    row: usize,
    col: usize,
    x_normalized: f64,
    y_normalized: f64,
    samples: Vec<SeamSample>,
    estimate: SeamWindowEstimate,
}

#[derive(Debug, Clone, Copy)]
struct SeamMeasurements {
    delta_luma: f64,
    delta_rgb: [f64; 3],
    seam_score: f64,
}

#[derive(Debug, Clone, Copy)]
struct SpatialPhotometricField {
    center_gain: [f64; 3],
    center_offset: [f64; 3],
    log_gain_slope_x: [f64; 3],
    log_gain_slope_y: [f64; 3],
    log_gain_quadratic_xx: [f64; 3],
    log_gain_quadratic_xy: [f64; 3],
    log_gain_quadratic_yy: [f64; 3],
    offset_slope_x: [f64; 3],
    offset_slope_y: [f64; 3],
    offset_quadratic_xx: [f64; 3],
    offset_quadratic_xy: [f64; 3],
    offset_quadratic_yy: [f64; 3],
}

fn applied_photometric_field(
    diagnostics: &SeamExposureCorrectionDiagnostics,
) -> SpatialPhotometricField {
    SpatialPhotometricField {
        center_gain: diagnostics.gain_rgb,
        center_offset: diagnostics.offset_rgb,
        log_gain_slope_x: diagnostics.spatial_gain_log_slope_x_rgb,
        log_gain_slope_y: diagnostics.spatial_gain_log_slope_y_rgb,
        log_gain_quadratic_xx: diagnostics.spatial_gain_log_quadratic_xx_rgb,
        log_gain_quadratic_xy: diagnostics.spatial_gain_log_quadratic_xy_rgb,
        log_gain_quadratic_yy: diagnostics.spatial_gain_log_quadratic_yy_rgb,
        offset_slope_x: diagnostics.spatial_offset_slope_x_rgb,
        offset_slope_y: diagnostics.spatial_offset_slope_y_rgb,
        offset_quadratic_xx: diagnostics.spatial_offset_quadratic_xx_rgb,
        offset_quadratic_xy: diagnostics.spatial_offset_quadratic_xy_rgb,
        offset_quadratic_yy: diagnostics.spatial_offset_quadratic_yy_rgb,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StitchHypothesis {
    pub ordering: String,
    pub x_offset: Option<usize>,
    pub overlap_width: Option<usize>,
    pub vertical_offset: Option<i32>,
    pub transform_model: String,
    pub objective_score: f64,
    pub accepted: bool,
    pub acceptance_reason: Option<String>,
    pub rejection_reason: Option<String>,
    pub search: StitchSearchDiagnostics,
    pub validation: StitchValidationMetrics,
}

/// Convert an RGB u16 image to grayscale f64 using luminance weights.
fn to_grayscale(img: &Array3<u16>) -> Array2<f64> {
    let (h, w, _) = img.dim();
    let mut gray = Array2::<f64>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let r = img[[y, x, 0]] as f64;
            let g = img[[y, x, 1]] as f64;
            let b = img[[y, x, 2]] as f64;
            gray[[y, x]] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }
    }
    gray
}

/// Compute normalized cross-correlation between two 2D arrays of the same shape.
fn ncc(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
    let n = a.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean_a = a.sum() / n;
    let mean_b = b.sum() / n;

    let mut sum_ab = 0.0;
    let mut sum_aa = 0.0;
    let mut sum_bb = 0.0;

    for (va, vb) in a.iter().zip(b.iter()) {
        let da = va - mean_a;
        let db = vb - mean_b;
        sum_ab += da * db;
        sum_aa += da * da;
        sum_bb += db * db;
    }

    let denom = (sum_aa * sum_bb).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }
    sum_ab / denom
}

/// Compute overlap score using exposure-invariant NCC.
fn overlap_score(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
    ncc(a, b).max(0.0)
}

fn overlap_score_at(
    gray_l: &Array2<f64>,
    gray_r: &Array2<f64>,
    h: usize,
    w_l: usize,
    ovl: usize,
    dy: i32,
) -> Option<f64> {
    let abs_dy = dy.unsigned_abs() as usize;
    let cmp_h = h.saturating_sub(abs_dy);
    if cmp_h < 10 {
        return None;
    }

    let (l_y_start, r_y_start) = if dy >= 0 {
        (dy as usize, 0usize)
    } else {
        (0usize, abs_dy)
    };

    let strip_l = gray_l.slice(s![l_y_start..(l_y_start + cmp_h), (w_l - ovl)..w_l]);
    let strip_r = gray_r.slice(s![r_y_start..(r_y_start + cmp_h), 0..ovl]);
    Some(overlap_score(strip_l.view(), strip_r.view()))
}

fn ncc1d(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let mean_a = a.iter().take(n).sum::<f64>() / n as f64;
    let mean_b = b.iter().take(n).sum::<f64>() / n as f64;
    let mut sum_ab = 0.0;
    let mut sum_aa = 0.0;
    let mut sum_bb = 0.0;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        sum_ab += da * db;
        sum_aa += da * da;
        sum_bb += db * db;
    }

    let denom = (sum_aa * sum_bb).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        sum_ab / denom
    }
}

fn column_feature_series(gray: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    let (h, w) = gray.dim();
    let mut mean_series = vec![0.0f64; w];
    let mut grad_series = vec![0.0f64; w];

    for x in 0..w {
        let mut sum = 0.0f64;
        let mut grad = 0.0f64;
        for y in 0..h {
            let v = gray[[y, x]];
            sum += v;
            if y > 0 {
                grad += (v - gray[[y - 1, x]]).abs();
            }
        }
        mean_series[x] = sum / h.max(1) as f64;
        grad_series[x] = grad / h.max(2) as f64;
    }

    (mean_series, grad_series)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TranslationSearchResult {
    search_band: SearchBand,
    x_offset: usize,
    overlap: usize,
    y_offset: i32,
    global_ncc: f64,
    signature_score: f64,
    overlap_support_score: f64,
    vertical_offset_plausibility_score: f64,
    plausibility_score: f64,
    correspondence_score: f64,
    prior_weight: f64,
    search_score: f64,
}

#[derive(Debug, Clone)]
struct TranslationSearchOutcome {
    candidates: Vec<TranslationSearchResult>,
    screened_overlap_count: usize,
    expanded_screened_overlap_count: usize,
    evaluated_overlap_count: usize,
    exhaustive_fallback_used: bool,
    expanded_overlap_search_used: bool,
    primary_max_overlap_considered: usize,
    max_overlap_considered: usize,
}

#[derive(Debug, Clone)]
struct EvaluatedTranslationCandidate {
    search: TranslationSearchResult,
    validation: StitchValidationMetrics,
    objective_score: f64,
    acceptance_reason: Option<String>,
    rejection_reason: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct LocalWindowMatch {
    row_idx: usize,
    col_idx: usize,
    x_norm: f64,
    y_norm: f64,
    score: f64,
    dx: i32,
    dy: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchBand {
    PrimaryScreened,
    PrimaryExhaustive,
    ExpandedSignature,
}

impl SearchBand {
    fn as_str(self) -> &'static str {
        match self {
            Self::PrimaryScreened => "primary_signature_screen",
            Self::PrimaryExhaustive => "primary_exhaustive",
            Self::ExpandedSignature => "expanded_signature_guided",
        }
    }
}

fn smoothstep01(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn fit_weighted_local_offset_plane(
    matches: &[LocalWindowMatch],
    value: fn(&LocalWindowMatch) -> f64,
) -> Option<Vector3<f64>> {
    if matches.is_empty() {
        return None;
    }

    if matches.len() < 3 {
        let total_weight = matches.iter().map(|m| m.score.max(1e-3)).sum::<f64>();
        let mean = matches
            .iter()
            .map(|m| value(m) * m.score.max(1e-3))
            .sum::<f64>()
            / total_weight.max(1e-6);
        return Some(Vector3::new(mean, 0.0, 0.0));
    }

    let mut ata = Matrix3::<f64>::zeros();
    let mut atb = Vector3::<f64>::zeros();
    for m in matches {
        let weight = m.score.max(1e-3);
        let basis = Vector3::new(1.0, m.x_norm, m.y_norm);
        ata += basis * basis.transpose() * weight;
        atb += basis * (value(m) * weight);
    }

    (ata + Matrix3::identity() * 1e-6)
        .try_inverse()
        .map(|inv| inv * atb)
}

fn evaluate_local_offset_plane(model: &Vector3<f64>, m: &LocalWindowMatch) -> f64 {
    model[0] + model[1] * m.x_norm + model[2] * m.y_norm
}

fn overlap_support_score(overlap_fraction: f64) -> f64 {
    let rise = smoothstep01((overlap_fraction - 0.015) / 0.055);
    let fall = 1.0 - smoothstep01((overlap_fraction - 0.80) / 0.15);
    (rise * fall).clamp(0.0, 1.0)
}

fn vertical_offset_plausibility_score(y_offset: i32, max_y_offset: i32) -> f64 {
    let limit = max_y_offset.unsigned_abs() as f64;
    if limit <= 0.5 {
        return 1.0;
    }
    let offset_fraction = y_offset.unsigned_abs() as f64 / limit;
    (1.0 - 0.85 * smoothstep01((offset_fraction - 0.35) / 0.65)).clamp(0.15, 1.0)
}

fn translation_plausibility_score(
    overlap_support_score: f64,
    vertical_offset_plausibility_score: f64,
) -> f64 {
    (0.65 * overlap_support_score + 0.35 * vertical_offset_plausibility_score).clamp(0.0, 1.0)
}

fn weighted_geometric_mean(components: &[(f64, f64)]) -> f64 {
    let mut weighted_log_sum = 0.0;
    let mut total_weight = 0.0;
    for &(value, weight) in components {
        if weight <= 0.0 {
            continue;
        }
        weighted_log_sum += weight * value.clamp(1e-6, 1.0).ln();
        total_weight += weight;
    }
    if total_weight <= 0.0 {
        0.0
    } else {
        (weighted_log_sum / total_weight).exp().clamp(0.0, 1.0)
    }
}

fn overlap_prior_weight(overlap_support_score: f64) -> f64 {
    let support = overlap_support_score.clamp(0.0, 1.0);
    if support < 0.25 {
        // Keep aggressively penalizing tiny edge-only overlaps.
        (0.18 + 1.52 * support).clamp(0.18, 0.56)
    } else {
        // Once the overlap is in a plausible seam band, treat size as a weak prior.
        (0.82 + 0.18 * support.sqrt()).clamp(0.82, 1.0)
    }
}

fn vertical_prior_weight(vertical_offset_plausibility_score: f64) -> f64 {
    (0.55 + 0.45 * vertical_offset_plausibility_score.clamp(0.0, 1.0)).clamp(0.55, 1.0)
}

fn translation_prior_weight(
    overlap_support_score: f64,
    vertical_offset_plausibility_score: f64,
) -> f64 {
    (overlap_prior_weight(overlap_support_score)
        * vertical_prior_weight(vertical_offset_plausibility_score))
    .clamp(0.0, 1.0)
}

fn translation_search_correspondence_score(global_ncc: f64, signature_score: f64) -> f64 {
    let arithmetic = (global_ncc * 0.80 + signature_score * 0.20).clamp(0.0, 1.0);
    let geometric = weighted_geometric_mean(&[(global_ncc, 0.80), (signature_score, 0.20)]);
    (0.80 * arithmetic + 0.20 * geometric).clamp(0.0, 1.0)
}

fn translation_search_score(
    correspondence_score: f64,
    overlap_support_score: f64,
    vertical_offset_plausibility_score: f64,
) -> f64 {
    (correspondence_score
        * translation_prior_weight(overlap_support_score, vertical_offset_plausibility_score))
    .clamp(0.0, 1.0)
}

fn translation_evidence_score(
    global_ncc_score: f64,
    signature_score: f64,
    local_consistency_score: f64,
) -> f64 {
    let arithmetic =
        (global_ncc_score * 0.60 + signature_score * 0.10 + local_consistency_score * 0.30)
            .clamp(0.0, 1.0);
    let geometric = weighted_geometric_mean(&[
        (global_ncc_score, 0.60),
        (signature_score, 0.10),
        (local_consistency_score, 0.30),
    ]);
    (0.75 * arithmetic + 0.25 * geometric).clamp(0.0, 1.0)
}

fn translation_objective_score(
    evidence_score: f64,
    overlap_support_score: f64,
    vertical_offset_plausibility_score: f64,
) -> f64 {
    (evidence_score
        * translation_prior_weight(overlap_support_score, vertical_offset_plausibility_score))
    .clamp(0.0, 1.0)
}

fn compare_translation_candidates(
    candidate: &TranslationSearchResult,
    current: &TranslationSearchResult,
) -> std::cmp::Ordering {
    current
        .search_score
        .partial_cmp(&candidate.search_score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| {
            current
                .global_ncc
                .partial_cmp(&candidate.global_ncc)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            current
                .plausibility_score
                .partial_cmp(&candidate.plausibility_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            current
                .signature_score
                .partial_cmp(&candidate.signature_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            candidate
                .y_offset
                .unsigned_abs()
                .cmp(&current.y_offset.unsigned_abs())
        })
        .then_with(|| candidate.overlap.cmp(&current.overlap))
}

fn compare_raw_translation_candidates(
    candidate: &TranslationSearchResult,
    current: &TranslationSearchResult,
) -> std::cmp::Ordering {
    current
        .global_ncc
        .partial_cmp(&candidate.global_ncc)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| {
            current
                .signature_score
                .partial_cmp(&candidate.signature_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| candidate.overlap.cmp(&current.overlap))
        .then_with(|| {
            candidate
                .y_offset
                .unsigned_abs()
                .cmp(&current.y_offset.unsigned_abs())
        })
}

fn signature_overlap_score_from_series(
    mean_l: &[f64],
    grad_l: &[f64],
    mean_r: &[f64],
    grad_r: &[f64],
    overlap: usize,
) -> f64 {
    let w_l = mean_l.len();
    let w_r = mean_r.len();
    if overlap == 0 || overlap > w_l || overlap > w_r {
        return 0.0;
    }

    let score_mean = ncc1d(&mean_l[w_l - overlap..w_l], &mean_r[0..overlap]).max(0.0);
    let score_grad = ncc1d(&grad_l[w_l - overlap..w_l], &grad_r[0..overlap]).max(0.0);
    0.65 * score_mean + 0.35 * score_grad
}

fn signature_overlap_scores(
    mean_l: &[f64],
    grad_l: &[f64],
    mean_r: &[f64],
    grad_r: &[f64],
    max_overlap: usize,
) -> Vec<f64> {
    let mut scores = vec![0.0f64; max_overlap.saturating_add(1)];
    for (overlap, slot) in scores.iter_mut().enumerate().skip(10) {
        *slot = signature_overlap_score_from_series(mean_l, grad_l, mean_r, grad_r, overlap);
    }
    scores
}

fn screened_overlaps_from_scores(
    signature_scores: &[f64],
    min_overlap: usize,
    max_overlap: usize,
    limit: usize,
    neighborhood: usize,
    pin_extrema: bool,
) -> Vec<usize> {
    if max_overlap < min_overlap || max_overlap >= signature_scores.len() {
        return Vec::new();
    }

    let mut signature_ranks: Vec<(usize, f64)> = (min_overlap..=max_overlap)
        .map(|overlap| (overlap, signature_scores[overlap]))
        .collect();
    signature_ranks.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut overlaps = BTreeSet::<usize>::new();
    for (overlap, _) in signature_ranks.iter().take(limit) {
        let start = overlap.saturating_sub(neighborhood).max(min_overlap);
        let end = (*overlap + neighborhood).min(max_overlap);
        for candidate in start..=end {
            overlaps.insert(candidate);
        }
    }

    if pin_extrema {
        // Keep the search honest when signatures are flat by pinning the extrema.
        overlaps.insert(min_overlap);
        overlaps.insert(max_overlap);
    }

    overlaps.into_iter().collect()
}

fn expanded_overlap_search_limit(primary_max_overlap: usize, w_l: usize, w_r: usize) -> usize {
    let width_limit = w_l.min(w_r);
    let plausible_limit = ((width_limit as f64) * MAX_PLAUSIBLE_OVERLAP_FRACTION).floor() as usize;
    if plausible_limit <= primary_max_overlap {
        return primary_max_overlap.min(width_limit);
    }
    let target = ((width_limit as f64) * 0.72).round() as usize;
    target
        .max(primary_max_overlap)
        .min(plausible_limit)
        .min(width_limit)
}

fn select_distinct_translation_candidates(
    candidates: &[TranslationSearchResult],
    limit: usize,
) -> Vec<TranslationSearchResult> {
    let mut distinct = Vec::<TranslationSearchResult>::new();
    for candidate in candidates {
        let too_close = distinct.iter().any(|existing| {
            existing.overlap.abs_diff(candidate.overlap) <= CANDIDATE_OVERLAP_DIVERSITY_PX
                && (existing.y_offset - candidate.y_offset).unsigned_abs() <= 2
        });
        if too_close {
            continue;
        }
        distinct.push(*candidate);
        if distinct.len() >= limit {
            break;
        }
    }
    distinct
}

fn search_translation_candidates_in_overlaps(
    gray_l: &Array2<f64>,
    gray_r: &Array2<f64>,
    signature_scores: &[f64],
    expected_y_offset: i32,
    max_y_offset: i32,
    overlaps: &[usize],
    search_band: SearchBand,
) -> Vec<TranslationSearchResult> {
    let h = gray_l.dim().0.min(gray_r.dim().0);
    let w_l = gray_l.dim().1;
    let w_r = gray_r.dim().1;
    let mut candidates: Vec<TranslationSearchResult> = overlaps
        .par_iter()
        .filter_map(|&overlap| {
            let signature_score = signature_scores.get(overlap).copied().unwrap_or(0.0);
            let mut best_for_overlap: Option<TranslationSearchResult> = None;

            let drift = max_y_offset.unsigned_abs().min(i32::MAX as u32) as i32;
            let min_y_offset = expected_y_offset.saturating_sub(drift);
            let max_y_offset = expected_y_offset.saturating_add(drift);
            for dy in min_y_offset..=max_y_offset {
                let Some(score) = overlap_score_at(gray_l, gray_r, h, w_l, overlap, dy) else {
                    continue;
                };
                let candidate = TranslationSearchResult {
                    search_band,
                    x_offset: w_l - overlap,
                    overlap,
                    y_offset: dy,
                    global_ncc: score,
                    signature_score,
                    overlap_support_score: overlap_support_score(
                        (overlap as f64 / w_l.max(1) as f64)
                            .max(overlap as f64 / w_r.max(1) as f64),
                    ),
                    vertical_offset_plausibility_score: vertical_offset_plausibility_score(
                        dy.saturating_sub(expected_y_offset),
                        drift,
                    ),
                    plausibility_score: 0.0,
                    correspondence_score: 0.0,
                    prior_weight: 0.0,
                    search_score: 0.0,
                };
                let mut candidate = candidate;
                candidate.plausibility_score = translation_plausibility_score(
                    candidate.overlap_support_score,
                    candidate.vertical_offset_plausibility_score,
                );
                candidate.correspondence_score = translation_search_correspondence_score(
                    candidate.global_ncc,
                    candidate.signature_score,
                );
                candidate.prior_weight = translation_prior_weight(
                    candidate.overlap_support_score,
                    candidate.vertical_offset_plausibility_score,
                );
                candidate.search_score = translation_search_score(
                    candidate.correspondence_score,
                    candidate.overlap_support_score,
                    candidate.vertical_offset_plausibility_score,
                );
                if best_for_overlap
                    .as_ref()
                    .map(|current| compare_translation_candidates(&candidate, current).is_lt())
                    .unwrap_or(true)
                {
                    best_for_overlap = Some(candidate);
                }
            }

            best_for_overlap
        })
        .collect();

    for candidate in &mut candidates {
        candidate.correspondence_score = translation_search_correspondence_score(
            candidate.global_ncc,
            candidate.signature_score,
        );
        candidate.prior_weight = translation_prior_weight(
            candidate.overlap_support_score,
            candidate.vertical_offset_plausibility_score,
        );
        candidate.search_score = translation_search_score(
            candidate.correspondence_score,
            candidate.overlap_support_score,
            candidate.vertical_offset_plausibility_score,
        );
    }
    candidates.sort_by(compare_translation_candidates);
    candidates
}

fn screened_search_is_sufficient(candidates: &[TranslationSearchResult]) -> bool {
    candidates.iter().take(3).any(|candidate| {
        candidate.global_ncc >= 0.55
            && candidate.plausibility_score >= 0.45
            && candidate.search_score >= 0.30
    })
}

fn should_expand_overlap_search(
    primary_candidates: &[TranslationSearchResult],
    primary_max_overlap: usize,
    expanded_max_overlap: usize,
) -> bool {
    if expanded_max_overlap <= primary_max_overlap {
        return false;
    }

    if !screened_search_is_sufficient(primary_candidates) {
        return true;
    }

    let near_primary_cap = primary_candidates
        .first()
        .map(|candidate| primary_max_overlap.saturating_sub(candidate.overlap) <= 24)
        .unwrap_or(false);
    let search_gap_is_tight = primary_candidates.len() >= 2
        && (primary_candidates[0].search_score - primary_candidates[1].search_score).abs() < 0.015;

    near_primary_cap || search_gap_is_tight
}

fn search_translation_hypotheses(
    gray_l: &Array2<f64>,
    gray_r: &Array2<f64>,
    max_overlap: usize,
    expected_y_offset: i32,
    max_y_offset: i32,
) -> Option<TranslationSearchOutcome> {
    let (h_l, w_l) = gray_l.dim();
    let (h_r, w_r) = gray_r.dim();
    let h = h_l.min(h_r);
    if h < 10 || w_l == 0 || w_r == 0 {
        return None;
    }

    let primary_max_ovl = max_overlap.min(w_l).min(w_r);
    if primary_max_ovl < 10 {
        return None;
    }

    let expanded_max_ovl = expanded_overlap_search_limit(primary_max_ovl, w_l, w_r);
    let (mean_l, grad_l) = column_feature_series(gray_l);
    let (mean_r, grad_r) = column_feature_series(gray_r);
    let signature_scores =
        signature_overlap_scores(&mean_l, &grad_l, &mean_r, &grad_r, expanded_max_ovl);
    let screened = screened_overlaps_from_scores(
        &signature_scores,
        10,
        primary_max_ovl,
        SIGNATURE_SCREEN_LIMIT,
        SIGNATURE_SCREEN_NEIGHBORHOOD,
        true,
    );
    let screened_candidates = search_translation_candidates_in_overlaps(
        gray_l,
        gray_r,
        &signature_scores,
        expected_y_offset,
        max_y_offset,
        &screened,
        SearchBand::PrimaryScreened,
    );

    let exhaustive_fallback_used = !screened_search_is_sufficient(&screened_candidates);
    let primary_candidates = if exhaustive_fallback_used {
        let exhaustive: Vec<usize> = (10..=primary_max_ovl).collect();
        search_translation_candidates_in_overlaps(
            gray_l,
            gray_r,
            &signature_scores,
            expected_y_offset,
            max_y_offset,
            &exhaustive,
            SearchBand::PrimaryExhaustive,
        )
    } else {
        screened_candidates
    };
    let expanded_overlap_search_used =
        should_expand_overlap_search(&primary_candidates, primary_max_ovl, expanded_max_ovl);
    let (expanded_screened_overlap_count, mut expanded_candidates) = if expanded_overlap_search_used
    {
        let expanded_overlaps = screened_overlaps_from_scores(
            &signature_scores,
            primary_max_ovl.saturating_add(1),
            expanded_max_ovl,
            EXPANDED_SIGNATURE_SCREEN_LIMIT,
            EXPANDED_SIGNATURE_SCREEN_NEIGHBORHOOD,
            false,
        );
        let count = expanded_overlaps.len();
        let candidates = search_translation_candidates_in_overlaps(
            gray_l,
            gray_r,
            &signature_scores,
            expected_y_offset,
            max_y_offset,
            &expanded_overlaps,
            SearchBand::ExpandedSignature,
        );
        (count, candidates)
    } else {
        (0usize, Vec::new())
    };

    let mut final_candidates = primary_candidates;
    final_candidates.append(&mut expanded_candidates);
    final_candidates.sort_by(compare_translation_candidates);
    let distinct_candidates = select_distinct_translation_candidates(
        &final_candidates,
        MAX_TRANSLATION_CANDIDATES_TO_VALIDATE.max(8),
    );
    if distinct_candidates.is_empty() {
        return None;
    }

    Some(TranslationSearchOutcome {
        candidates: distinct_candidates,
        screened_overlap_count: screened.len(),
        expanded_screened_overlap_count,
        evaluated_overlap_count: if exhaustive_fallback_used {
            primary_max_ovl.saturating_sub(9)
        } else {
            screened.len()
        } + expanded_screened_overlap_count,
        exhaustive_fallback_used,
        expanded_overlap_search_used,
        primary_max_overlap_considered: primary_max_ovl,
        max_overlap_considered: if expanded_overlap_search_used {
            expanded_max_ovl
        } else {
            primary_max_ovl
        },
    })
}

fn percentile_from_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn overlap_plausibility_reason(
    overlap_width: usize,
    overlap_fraction_left: f64,
    overlap_fraction_right: f64,
) -> Option<String> {
    let overlap_fraction = overlap_fraction_left.max(overlap_fraction_right);
    if overlap_fraction < MIN_PLAUSIBLE_OVERLAP_FRACTION {
        Some(format!(
            "high NCC on a {}px overlap is likely an edge-only false positive; overlap fraction {:.3}/{:.3} is below the minimum plausible band {:.3}",
            overlap_width,
            overlap_fraction_left,
            overlap_fraction_right,
            MIN_PLAUSIBLE_OVERLAP_FRACTION
        ))
    } else if overlap_fraction > MAX_PLAUSIBLE_OVERLAP_FRACTION {
        Some(format!(
            "overlap fraction {:.3}/{:.3} is too large to be a plausible split-frame seam (max {:.3})",
            overlap_fraction_left,
            overlap_fraction_right,
            MAX_PLAUSIBLE_OVERLAP_FRACTION
        ))
    } else {
        None
    }
}

fn default_validation_metrics() -> StitchValidationMetrics {
    StitchValidationMetrics {
        overlap_width_px: 0,
        comparison_height_px: 0,
        global_ncc_score: 0.0,
        signature_score: 0.0,
        overlap_support_score: 0.0,
        vertical_offset_plausibility_score: 0.0,
        plausibility_score: 0.0,
        evidence_score: 0.0,
        prior_weight: 0.0,
        local_consistency_score: 0.0,
        local_windows_evaluated: 0,
        local_inlier_count: 0,
        local_inlier_ratio: 0.0,
        median_local_ncc: 0.0,
        informative_ncc_threshold: 0.0,
        informative_window_count: 0,
        informative_window_ratio: 0.0,
        informative_inlier_count: 0,
        informative_inlier_ratio: 0.0,
        row_support_ratio: 0.0,
        column_support_ratio: 0.0,
        seam_support_score: 0.0,
        median_residual_px: 0.0,
        p95_residual_px: 0.0,
        median_model_residual_px: 0.0,
        p95_model_residual_px: 0.0,
        mean_dx_px: 0.0,
        mean_dy_px: 0.0,
        local_dx_plane_px: [0.0; 3],
        local_dy_plane_px: [0.0; 3],
        overlap_fraction_left: 0.0,
        overlap_fraction_right: 0.0,
        plausibility_ok: false,
        plausibility_reason: Some("no overlap candidate".to_string()),
        homography_inliers: None,
        homography_median_error: None,
        homography_p95_error: None,
        homography_feature_validation: None,
    }
}

fn validate_translation_hypothesis(
    gray_l: &Array2<f64>,
    gray_r: &Array2<f64>,
    search: TranslationSearchResult,
    expected_y_offset: i32,
    config: &StitchConfig,
) -> StitchValidationMetrics {
    let (h_l, w_l) = gray_l.dim();
    let (h_r, w_r) = gray_r.dim();
    let abs_dy = search.y_offset.unsigned_abs() as usize;
    let cmp_h = h_l.min(h_r).saturating_sub(abs_dy);
    if cmp_h < 16 || search.overlap < 12 {
        let mut metrics = default_validation_metrics();
        metrics.overlap_width_px = search.overlap;
        metrics.comparison_height_px = cmp_h;
        metrics.global_ncc_score = search.global_ncc;
        metrics.signature_score = search.signature_score;
        metrics.overlap_support_score = search.overlap_support_score;
        metrics.vertical_offset_plausibility_score = search.vertical_offset_plausibility_score;
        metrics.plausibility_score = search.plausibility_score;
        metrics.prior_weight = search.prior_weight;
        metrics.informative_ncc_threshold = (search.global_ncc * 0.70).clamp(0.35, 0.8);
        metrics.plausibility_reason = Some("overlap too small for local validation".to_string());
        return metrics;
    }

    let (l_y_start, r_y_start) = if search.y_offset >= 0 {
        (search.y_offset as usize, 0usize)
    } else {
        (0usize, abs_dy)
    };
    let left_strip = gray_l.slice(s![
        l_y_start..(l_y_start + cmp_h),
        (w_l - search.overlap)..w_l
    ]);
    let right_strip = gray_r.slice(s![r_y_start..(r_y_start + cmp_h), 0..search.overlap]);

    let grid_rows = (cmp_h / 72).clamp(2, 6);
    let grid_cols = (search.overlap / 36).clamp(2, 6);
    let window_h = (cmp_h / grid_rows).max(12).min(cmp_h);
    let window_w = (search.overlap / grid_cols).max(10).min(search.overlap);
    let search_radius = 2i32;
    let inlier_ncc_threshold = (search.global_ncc * 0.70).clamp(0.35, 0.8);
    let mut local_matches = Vec::<LocalWindowMatch>::new();

    let mut local_ncc = Vec::<f64>::new();
    let mut residuals = Vec::<f64>::new();
    let mut dx_values = Vec::<f64>::new();
    let mut dy_values = Vec::<f64>::new();
    let mut inliers = 0usize;

    for row_idx in 0..grid_rows {
        let y0 = if grid_rows == 1 || cmp_h == window_h {
            0
        } else {
            row_idx * (cmp_h - window_h) / (grid_rows - 1)
        };
        for col_idx in 0..grid_cols {
            let x0 = if grid_cols == 1 || search.overlap == window_w {
                0
            } else {
                col_idx * (search.overlap - window_w) / (grid_cols - 1)
            };

            let patch_l = left_strip.slice(s![y0..(y0 + window_h), x0..(x0 + window_w)]);
            let mut best_score = f64::NEG_INFINITY;
            let mut best_dx = 0i32;
            let mut best_dy = 0i32;

            for local_dy in -search_radius..=search_radius {
                let ry = y0 as i32 + local_dy;
                if ry < 0 || ry + window_h as i32 > cmp_h as i32 {
                    continue;
                }

                for local_dx in -search_radius..=search_radius {
                    let rx = x0 as i32 + local_dx;
                    if rx < 0 || rx + window_w as i32 > search.overlap as i32 {
                        continue;
                    }

                    let patch_r = right_strip.slice(s![
                        ry as usize..(ry as usize + window_h),
                        rx as usize..(rx as usize + window_w)
                    ]);
                    let score = overlap_score(patch_l.view(), patch_r.view());
                    if score > best_score {
                        best_score = score;
                        best_dx = local_dx;
                        best_dy = local_dy;
                    }
                }
            }

            if best_score.is_finite() {
                let residual = ((best_dx * best_dx + best_dy * best_dy) as f64).sqrt();
                local_matches.push(LocalWindowMatch {
                    row_idx,
                    col_idx,
                    x_norm: if search.overlap == window_w {
                        0.0
                    } else {
                        x0 as f64 / (search.overlap - window_w) as f64
                    },
                    y_norm: if cmp_h == window_h {
                        0.0
                    } else {
                        y0 as f64 / (cmp_h - window_h) as f64
                    },
                    score: best_score,
                    dx: best_dx,
                    dy: best_dy,
                });
                local_ncc.push(best_score);
                residuals.push(residual);
                dx_values.push(best_dx as f64);
                dy_values.push(best_dy as f64);

                if best_score >= inlier_ncc_threshold && residual <= 1.5 {
                    inliers += 1;
                }
            }
        }
    }

    let mut sorted_ncc = local_ncc.clone();
    sorted_ncc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut sorted_residuals = residuals.clone();
    sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean_dx = if dx_values.is_empty() {
        0.0
    } else {
        dx_values.iter().sum::<f64>() / dx_values.len() as f64
    };
    let mean_dy = if dy_values.is_empty() {
        0.0
    } else {
        dy_values.iter().sum::<f64>() / dy_values.len() as f64
    };
    let local_inlier_ratio = if local_ncc.is_empty() {
        0.0
    } else {
        inliers as f64 / local_ncc.len() as f64
    };
    let median_local_ncc = percentile_from_sorted(&sorted_ncc, 0.5);
    let median_residual_px = percentile_from_sorted(&sorted_residuals, 0.5);
    let p95_residual_px = percentile_from_sorted(&sorted_residuals, 0.95);
    let informative_matches: Vec<LocalWindowMatch> = local_matches
        .iter()
        .copied()
        .filter(|m| m.score >= inlier_ncc_threshold)
        .collect();
    let dx_plane = fit_weighted_local_offset_plane(&informative_matches, |m| m.dx as f64)
        .unwrap_or_else(|| Vector3::new(0.0, 0.0, 0.0));
    let dy_plane = fit_weighted_local_offset_plane(&informative_matches, |m| m.dy as f64)
        .unwrap_or_else(|| Vector3::new(0.0, 0.0, 0.0));
    let informative_model_residuals: Vec<f64> = informative_matches
        .iter()
        .map(|m| {
            let dx_residual = evaluate_local_offset_plane(&dx_plane, m) - m.dx as f64;
            let dy_residual = evaluate_local_offset_plane(&dy_plane, m) - m.dy as f64;
            (dx_residual * dx_residual + dy_residual * dy_residual).sqrt()
        })
        .collect();
    let informative_window_count = informative_matches.len();
    let informative_window_ratio = if local_matches.is_empty() {
        0.0
    } else {
        informative_window_count as f64 / local_matches.len() as f64
    };
    let informative_inlier_count = informative_model_residuals
        .iter()
        .filter(|&&residual| residual <= INFORMATIVE_MODEL_INLIER_RESIDUAL_PX)
        .count();
    let informative_inlier_ratio = if informative_window_count == 0 {
        0.0
    } else {
        informative_inlier_count as f64 / informative_window_count as f64
    };
    let mut row_supported = vec![false; grid_rows];
    let mut column_supported = vec![false; grid_cols];
    for (m, &residual) in informative_matches
        .iter()
        .zip(informative_model_residuals.iter())
    {
        if residual <= INFORMATIVE_MODEL_INLIER_RESIDUAL_PX {
            row_supported[m.row_idx] = true;
            column_supported[m.col_idx] = true;
        }
    }
    let row_support_ratio = if grid_rows == 0 {
        0.0
    } else {
        row_supported.iter().filter(|&&supported| supported).count() as f64 / grid_rows as f64
    };
    let column_support_ratio = if grid_cols == 0 {
        0.0
    } else {
        column_supported
            .iter()
            .filter(|&&supported| supported)
            .count() as f64
            / grid_cols as f64
    };
    let mut sorted_informative_ncc = informative_matches
        .iter()
        .map(|m| m.score)
        .collect::<Vec<_>>();
    sorted_informative_ncc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut sorted_model_residuals = informative_model_residuals.clone();
    sorted_model_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let informative_median_local_ncc = percentile_from_sorted(&sorted_informative_ncc, 0.5);
    let median_model_residual_px = percentile_from_sorted(&sorted_model_residuals, 0.5);
    let p95_model_residual_px = percentile_from_sorted(&sorted_model_residuals, 0.95);
    let informative_window_factor =
        smoothstep01((informative_window_ratio - 0.10) / (MIN_INFORMATIVE_WINDOW_RATIO - 0.10));
    let coverage_score = (row_support_ratio * column_support_ratio)
        .sqrt()
        .clamp(0.0, 1.0);
    let informative_residual_score = if informative_window_count == 0 {
        0.0
    } else {
        (1.0 - (p95_model_residual_px / 4.0)).clamp(0.0, 1.0)
    };
    let seam_support_score = if informative_window_count == 0 {
        0.0
    } else {
        (0.25 * informative_window_factor
            + 0.25 * informative_inlier_ratio
            + 0.20 * coverage_score
            + 0.15 * informative_median_local_ncc
            + 0.15 * informative_residual_score)
            .clamp(0.0, 1.0)
    };
    let legacy_local_consistency_score = if local_ncc.is_empty() {
        0.0
    } else {
        let residual_score = (1.0 - (p95_residual_px / 4.0)).clamp(0.0, 1.0);
        (0.45 * local_inlier_ratio + 0.35 * median_local_ncc + 0.20 * residual_score)
            .clamp(0.0, 1.0)
    };
    let local_consistency_score =
        (0.40 * legacy_local_consistency_score + 0.60 * seam_support_score).clamp(0.0, 1.0);
    let overlap_fraction_left = search.overlap as f64 / w_l.max(1) as f64;
    let overlap_fraction_right = search.overlap as f64 / w_r.max(1) as f64;
    let vertical_drift = search.y_offset.saturating_sub(expected_y_offset);
    let vertical_offset_plausibility_score =
        vertical_offset_plausibility_score(vertical_drift, config.max_y_offset);
    let plausibility_score = translation_plausibility_score(
        search.overlap_support_score,
        vertical_offset_plausibility_score,
    );
    let plausibility_reason = if vertical_drift.unsigned_abs() > config.max_y_offset.unsigned_abs()
    {
        Some(format!(
            "vertical offset {} drifted {}px from expected {}px, beyond the {}px search radius",
            search.y_offset, vertical_drift, expected_y_offset, config.max_y_offset
        ))
    } else {
        overlap_plausibility_reason(
            search.overlap,
            overlap_fraction_left,
            overlap_fraction_right,
        )
    };

    let evidence_score = translation_evidence_score(
        search.global_ncc,
        search.signature_score,
        local_consistency_score,
    );
    let prior_weight = translation_prior_weight(
        search.overlap_support_score,
        vertical_offset_plausibility_score,
    );

    StitchValidationMetrics {
        overlap_width_px: search.overlap,
        comparison_height_px: cmp_h,
        global_ncc_score: search.global_ncc,
        signature_score: search.signature_score,
        overlap_support_score: search.overlap_support_score,
        vertical_offset_plausibility_score,
        plausibility_score,
        evidence_score,
        prior_weight,
        local_consistency_score,
        local_windows_evaluated: local_ncc.len(),
        local_inlier_count: inliers,
        local_inlier_ratio,
        median_local_ncc,
        informative_ncc_threshold: inlier_ncc_threshold,
        informative_window_count,
        informative_window_ratio,
        informative_inlier_count,
        informative_inlier_ratio,
        row_support_ratio,
        column_support_ratio,
        seam_support_score,
        median_residual_px,
        p95_residual_px,
        median_model_residual_px,
        p95_model_residual_px,
        mean_dx_px: mean_dx,
        mean_dy_px: mean_dy,
        local_dx_plane_px: [dx_plane[0], dx_plane[1], dx_plane[2]],
        local_dy_plane_px: [dy_plane[0], dy_plane[1], dy_plane[2]],
        overlap_fraction_left,
        overlap_fraction_right,
        plausibility_ok: plausibility_reason.is_none(),
        plausibility_reason,
        homography_inliers: None,
        homography_median_error: None,
        homography_p95_error: None,
        homography_feature_validation: None,
    }
}

fn objective_score(validation: &StitchValidationMetrics) -> f64 {
    translation_objective_score(
        validation.evidence_score,
        validation.overlap_support_score,
        validation.vertical_offset_plausibility_score,
    )
}

fn narrow_seam_global_ncc_floor(config: &StitchConfig) -> f64 {
    (config.min_ncc_score - 0.10).max(0.50)
}

fn narrow_seam_objective_floor(config: &StitchConfig) -> f64 {
    (config.min_objective_score - NARROW_SEAM_OBJECTIVE_MARGIN).max(0.40)
}

fn narrow_overlap_seam_acceptance_reason(
    validation: &StitchValidationMetrics,
    config: &StitchConfig,
) -> Option<String> {
    let overlap_fraction = validation
        .overlap_fraction_left
        .max(validation.overlap_fraction_right);
    let objective = objective_score(validation);
    let min_informative_windows = MIN_INFORMATIVE_LOCAL_WINDOWS.max(config.min_validation_windows);

    if !validation.plausibility_ok
        || !(MIN_PLAUSIBLE_OVERLAP_FRACTION..=NARROW_SEAM_RESCUE_MAX_OVERLAP_FRACTION)
            .contains(&overlap_fraction)
        || validation.global_ncc_score < narrow_seam_global_ncc_floor(config)
        || validation.signature_score < MIN_NARROW_SEAM_SIGNATURE_SCORE
        || validation.informative_window_count < min_informative_windows
        || validation.informative_window_ratio < MIN_INFORMATIVE_WINDOW_RATIO
        || validation.informative_inlier_ratio < config.min_local_inlier_ratio
        || validation.row_support_ratio < MIN_NARROW_SEAM_ROW_SUPPORT_RATIO
        || validation.column_support_ratio < MIN_NARROW_SEAM_COLUMN_SUPPORT_RATIO
        || validation.seam_support_score < MIN_NARROW_SEAM_SEAM_SUPPORT_SCORE
        || validation.p95_model_residual_px > MAX_INFORMATIVE_MODEL_P95_RESIDUAL_PX
        || objective < narrow_seam_objective_floor(config)
    {
        return None;
    }

    Some(format!(
        "accepted via narrow-overlap seam support: NCC {:.3} is below the default {:.3}, but {} informative windows ({:.1}% of local probes) support a coherent seam across {:.0}% of row bands and {:.0}% of column bands; seam-support score {:.3}, signature {:.3}, objective {:.3}",
        validation.global_ncc_score,
        config.min_ncc_score,
        validation.informative_window_count,
        validation.informative_window_ratio * 100.0,
        validation.row_support_ratio * 100.0,
        validation.column_support_ratio * 100.0,
        validation.seam_support_score,
        validation.signature_score,
        objective
    ))
}

fn translation_rejection_reason(
    validation: &StitchValidationMetrics,
    config: &StitchConfig,
    acceptance_reason: Option<&str>,
) -> Option<String> {
    if acceptance_reason.is_some() {
        return None;
    }
    if validation.global_ncc_score < config.min_ncc_score {
        return Some(format!(
            "global NCC {:.3} below threshold {:.3}",
            validation.global_ncc_score, config.min_ncc_score
        ));
    }
    if validation.local_windows_evaluated < config.min_validation_windows {
        return Some(format!(
            "only {} local windows validated; need at least {}",
            validation.local_windows_evaluated, config.min_validation_windows
        ));
    }
    if validation.plausibility_ok
        && validation.overlap_support_score >= 0.55
        && validation.global_ncc_score >= (config.min_ncc_score + 0.18).min(0.85)
        && validation.signature_score >= 0.70
        && validation.local_consistency_score >= 0.45
        && validation.vertical_offset_plausibility_score >= 0.55
    {
        return None;
    }
    if validation.overlap_support_score < 0.25 {
        return Some(format!(
            "high NCC on a {}px overlap is not sufficient support; overlap-support score {:.3} is too weak",
            validation.overlap_width_px, validation.overlap_support_score
        ));
    }
    if validation.local_inlier_ratio < config.min_local_inlier_ratio {
        return Some(format!(
            "local inlier ratio {:.3} below threshold {:.3}",
            validation.local_inlier_ratio, config.min_local_inlier_ratio
        ));
    }
    if validation.median_residual_px > config.max_local_median_residual {
        return Some(format!(
            "median local residual {:.2}px exceeded {:.2}px",
            validation.median_residual_px, config.max_local_median_residual
        ));
    }
    if validation.p95_residual_px > config.max_local_p95_residual {
        return Some(format!(
            "p95 local residual {:.2}px exceeded {:.2}px",
            validation.p95_residual_px, config.max_local_p95_residual
        ));
    }
    if !validation.plausibility_ok {
        return Some(
            validation
                .plausibility_reason
                .clone()
                .unwrap_or_else(|| "translation plausibility check failed".to_string()),
        );
    }
    let objective = objective_score(validation);
    if objective < config.min_objective_score {
        return Some(format!(
            "objective score {:.3} below threshold {:.3}",
            objective, config.min_objective_score
        ));
    }
    None
}

fn compare_evaluated_translation_candidates(
    candidate: &EvaluatedTranslationCandidate,
    current: &EvaluatedTranslationCandidate,
) -> std::cmp::Ordering {
    (current.rejection_reason.is_none() as u8)
        .cmp(&(candidate.rejection_reason.is_none() as u8))
        .then_with(|| {
            current
                .objective_score
                .partial_cmp(&candidate.objective_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| compare_translation_candidates(&candidate.search, &current.search))
}

fn default_search_diagnostics() -> StitchSearchDiagnostics {
    StitchSearchDiagnostics {
        screening_strategy: "no_overlap_candidate".to_string(),
        screened_overlap_count: 0,
        expanded_screened_overlap_count: 0,
        evaluated_overlap_count: 0,
        exhaustive_fallback_used: false,
        expanded_overlap_search_used: false,
        primary_max_overlap_considered: 0,
        max_overlap_considered: 0,
        selected_candidate_rank: None,
        selected_candidate_search_score: None,
        search_score_gap_to_runner_up: None,
        selection_reason: None,
        top_candidates: Vec::new(),
        evaluated_candidates: Vec::new(),
    }
}

fn build_selection_reason(
    outcome: &TranslationSearchOutcome,
    evaluated: &[EvaluatedTranslationCandidate],
    selected_candidate_rank: Option<usize>,
    selected: &EvaluatedTranslationCandidate,
) -> Option<String> {
    let gap_note = if outcome.candidates.len() >= 2 {
        let gap = (outcome.candidates[0].search_score - outcome.candidates[1].search_score).abs();
        if gap < 0.01 {
            format!(" Search-score gap to the runner-up was only {:.3}.", gap)
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    if outcome.expanded_overlap_search_used
        && matches!(selected.search.search_band, SearchBand::ExpandedSignature)
    {
        let best_primary = evaluated.iter().find(|candidate| {
            !matches!(candidate.search.search_band, SearchBand::ExpandedSignature)
        });
        if let Some(primary) = best_primary {
            return Some(format!(
                "Expanded signature-guided search widened the overlap sweep from {}px to {}px and surfaced the selected {}px candidate. It outranked the best <= {}px candidate on objective {:.3} vs {:.3}, correspondence evidence {:.3} vs {:.3}, prior weight {:.3} vs {:.3}, local consistency {:.3} vs {:.3}, and global NCC {:.3} vs {:.3}.{}",
                outcome.primary_max_overlap_considered,
                outcome.max_overlap_considered,
                selected.search.overlap,
                outcome.primary_max_overlap_considered,
                selected.objective_score,
                primary.objective_score,
                selected.validation.evidence_score,
                primary.validation.evidence_score,
                selected.validation.prior_weight,
                primary.validation.prior_weight,
                selected.validation.local_consistency_score,
                primary.validation.local_consistency_score,
                selected.validation.global_ncc_score,
                primary.validation.global_ncc_score,
                gap_note
            ));
        }
        return Some(format!(
            "Expanded signature-guided search widened the overlap sweep from {}px to {}px and surfaced the selected {}px candidate.{}",
            outcome.primary_max_overlap_considered,
            outcome.max_overlap_considered,
            selected.search.overlap,
            gap_note
        ));
    }

    if let Some(rank) = selected_candidate_rank {
        if rank > 0 {
            let search_leader = outcome.candidates.first()?;
            let search_leader_eval = evaluated
                .iter()
                .find(|candidate| candidate.search == *search_leader);
            if let Some(leader) = search_leader_eval {
                return Some(format!(
                    "Validation promoted search rank {} over search rank 1: objective {:.3} vs {:.3}, correspondence evidence {:.3} vs {:.3}, prior weight {:.3} vs {:.3}, local consistency {:.3} vs {:.3}, and global NCC {:.3} vs {:.3}.{}",
                    rank + 1,
                    selected.objective_score,
                    leader.objective_score,
                    selected.validation.evidence_score,
                    leader.validation.evidence_score,
                    selected.validation.prior_weight,
                    leader.validation.prior_weight,
                    selected.validation.local_consistency_score,
                    leader.validation.local_consistency_score,
                    selected.validation.global_ncc_score,
                    leader.validation.global_ncc_score,
                    gap_note
                ));
            }
            return Some(format!(
                "Validation promoted search rank {} over search rank 1.{}",
                rank + 1,
                gap_note
            ));
        }
    }

    Some(format!(
        "Search rank 1 remained the best-scoring candidate after validation: objective {:.3}, correspondence evidence {:.3}, prior weight {:.3}, local consistency {:.3}, global NCC {:.3}.{}",
        selected.objective_score,
        selected.validation.evidence_score,
        selected.validation.prior_weight,
        selected.validation.local_consistency_score,
        selected.validation.global_ncc_score,
        gap_note
    ))
}

fn build_search_diagnostics(
    outcome: &TranslationSearchOutcome,
    w_l: usize,
    w_r: usize,
    evaluated: &[EvaluatedTranslationCandidate],
    selected: Option<&EvaluatedTranslationCandidate>,
) -> StitchSearchDiagnostics {
    let selected_candidate_rank = selected.and_then(|needle| {
        outcome
            .candidates
            .iter()
            .position(|candidate| candidate == &needle.search)
    });
    let search_score_gap_to_runner_up = if outcome.candidates.len() >= 2 {
        Some((outcome.candidates[0].search_score - outcome.candidates[1].search_score).abs())
    } else {
        None
    };
    let selection_reason = selected.and_then(|candidate| {
        build_selection_reason(outcome, evaluated, selected_candidate_rank, candidate)
    });

    StitchSearchDiagnostics {
        screening_strategy: if outcome.exhaustive_fallback_used
            && outcome.expanded_overlap_search_used
        {
            "signature_screen_then_primary_exhaustive_plus_expanded_signature".to_string()
        } else if outcome.exhaustive_fallback_used {
            "signature_screen_then_primary_exhaustive".to_string()
        } else if outcome.expanded_overlap_search_used {
            "signature_screen_plus_expanded_signature".to_string()
        } else {
            "signature_screen_only".to_string()
        },
        screened_overlap_count: outcome.screened_overlap_count,
        expanded_screened_overlap_count: outcome.expanded_screened_overlap_count,
        evaluated_overlap_count: outcome.evaluated_overlap_count,
        exhaustive_fallback_used: outcome.exhaustive_fallback_used,
        expanded_overlap_search_used: outcome.expanded_overlap_search_used,
        primary_max_overlap_considered: outcome.primary_max_overlap_considered,
        max_overlap_considered: outcome.max_overlap_considered,
        selected_candidate_rank: selected_candidate_rank.map(|rank| rank + 1),
        selected_candidate_search_score: selected.map(|candidate| candidate.search.search_score),
        search_score_gap_to_runner_up,
        selection_reason,
        top_candidates: outcome
            .candidates
            .iter()
            .take(MAX_REPORTED_TRANSLATION_CANDIDATES)
            .enumerate()
            .map(|(rank, candidate)| StitchSearchCandidate {
                rank: rank + 1,
                search_band: candidate.search_band.as_str().to_string(),
                overlap_width: candidate.overlap,
                x_offset: candidate.x_offset,
                vertical_offset: candidate.y_offset,
                global_ncc_score: candidate.global_ncc,
                signature_score: candidate.signature_score,
                overlap_fraction_left: candidate.overlap as f64 / w_l.max(1) as f64,
                overlap_fraction_right: candidate.overlap as f64 / w_r.max(1) as f64,
                overlap_support_score: candidate.overlap_support_score,
                vertical_offset_plausibility_score: candidate.vertical_offset_plausibility_score,
                plausibility_score: candidate.plausibility_score,
                correspondence_score: candidate.correspondence_score,
                prior_weight: candidate.prior_weight,
                search_score: candidate.search_score,
            })
            .collect(),
        evaluated_candidates: evaluated
            .iter()
            .take(MAX_REPORTED_EVALUATED_CANDIDATES)
            .enumerate()
            .map(|(validation_rank, candidate)| StitchEvaluatedCandidate {
                validation_rank: validation_rank + 1,
                search_rank: outcome
                    .candidates
                    .iter()
                    .position(|search_candidate| search_candidate == &candidate.search)
                    .map(|rank| rank + 1),
                search_band: candidate.search.search_band.as_str().to_string(),
                overlap_width: candidate.search.overlap,
                x_offset: candidate.search.x_offset,
                vertical_offset: candidate.search.y_offset,
                objective_score: candidate.objective_score,
                accepted: candidate.rejection_reason.is_none(),
                acceptance_reason: candidate.acceptance_reason.clone(),
                rejection_reason: candidate.rejection_reason.clone(),
                global_ncc_score: candidate.validation.global_ncc_score,
                signature_score: candidate.validation.signature_score,
                overlap_support_score: candidate.validation.overlap_support_score,
                vertical_offset_plausibility_score: candidate
                    .validation
                    .vertical_offset_plausibility_score,
                plausibility_score: candidate.validation.plausibility_score,
                evidence_score: candidate.validation.evidence_score,
                prior_weight: candidate.validation.prior_weight,
                local_consistency_score: candidate.validation.local_consistency_score,
                informative_window_count: candidate.validation.informative_window_count,
                informative_window_ratio: candidate.validation.informative_window_ratio,
                informative_inlier_ratio: candidate.validation.informative_inlier_ratio,
                row_support_ratio: candidate.validation.row_support_ratio,
                column_support_ratio: candidate.validation.column_support_ratio,
                seam_support_score: candidate.validation.seam_support_score,
                local_inlier_ratio: candidate.validation.local_inlier_ratio,
                median_local_ncc: candidate.validation.median_local_ncc,
                median_residual_px: candidate.validation.median_residual_px,
                p95_residual_px: candidate.validation.p95_residual_px,
                p95_model_residual_px: candidate.validation.p95_model_residual_px,
            })
            .collect(),
    }
}

fn evaluate_translation_hypothesis(
    left: &Array3<u16>,
    right: &Array3<u16>,
    order: StitchOrder,
    config: &StitchConfig,
) -> StitchHypothesis {
    let (w_l, w_r) = (left.dim().1, right.dim().1);
    let gray_l = to_grayscale(left);
    let gray_r = to_grayscale(right);
    let expected_y_offset = if matches!(order, StitchOrder::LeftRight) {
        config.expected_y_offset
    } else {
        config.expected_y_offset.saturating_neg()
    };
    let Some(search_outcome) = search_translation_hypotheses(
        &gray_l,
        &gray_r,
        config.max_overlap,
        expected_y_offset,
        config.max_y_offset,
    ) else {
        return StitchHypothesis {
            ordering: order.label().to_string(),
            x_offset: None,
            overlap_width: None,
            vertical_offset: None,
            transform_model: "translation".to_string(),
            objective_score: 0.0,
            accepted: false,
            acceptance_reason: None,
            rejection_reason: Some("no overlap found".to_string()),
            search: default_search_diagnostics(),
            validation: default_validation_metrics(),
        };
    };

    let mut evaluated: Vec<EvaluatedTranslationCandidate> = search_outcome
        .candidates
        .iter()
        .take(MAX_TRANSLATION_CANDIDATES_TO_VALIDATE)
        .map(|search| {
            let validation = validate_translation_hypothesis(
                &gray_l,
                &gray_r,
                *search,
                expected_y_offset,
                config,
            );
            let objective_score = objective_score(&validation);
            let acceptance_reason = narrow_overlap_seam_acceptance_reason(&validation, config);
            let rejection_reason = if search.overlap < config.min_overlap {
                Some(format!(
                    "overlap {} below minimum {}",
                    search.overlap, config.min_overlap
                ))
            } else {
                translation_rejection_reason(&validation, config, acceptance_reason.as_deref())
            };

            EvaluatedTranslationCandidate {
                search: *search,
                validation,
                objective_score,
                acceptance_reason,
                rejection_reason,
            }
        })
        .collect();
    evaluated.sort_by(compare_evaluated_translation_candidates);

    let best = evaluated
        .first()
        .expect("search outcome should contain at least one candidate");
    let search_diagnostics =
        build_search_diagnostics(&search_outcome, w_l, w_r, &evaluated, Some(best));

    StitchHypothesis {
        ordering: order.label().to_string(),
        x_offset: Some(best.search.x_offset),
        overlap_width: Some(best.search.overlap),
        vertical_offset: Some(best.search.y_offset),
        transform_model: "translation".to_string(),
        objective_score: best.objective_score,
        accepted: best.rejection_reason.is_none(),
        acceptance_reason: best.acceptance_reason.clone(),
        rejection_reason: best.rejection_reason.clone(),
        search: search_diagnostics,
        validation: best.validation.clone(),
    }
}

fn acceptance_thresholds_json(config: &StitchConfig) -> serde_json::Value {
    serde_json::json!({
        "min_ncc_score": config.min_ncc_score,
        "min_overlap": config.min_overlap,
        "min_plausible_overlap_fraction": MIN_PLAUSIBLE_OVERLAP_FRACTION,
        "max_plausible_overlap_fraction": MAX_PLAUSIBLE_OVERLAP_FRACTION,
        "min_objective_score": config.min_objective_score,
        "min_local_inlier_ratio": config.min_local_inlier_ratio,
        "max_local_median_residual": config.max_local_median_residual,
        "max_local_p95_residual": config.max_local_p95_residual,
        "min_validation_windows": config.min_validation_windows,
        "expected_y_offset": config.expected_y_offset,
        "max_y_offset": config.max_y_offset,
        "narrow_overlap_rescue": {
            "max_overlap_fraction": NARROW_SEAM_RESCUE_MAX_OVERLAP_FRACTION,
            "min_global_ncc": narrow_seam_global_ncc_floor(config),
            "min_signature_score": MIN_NARROW_SEAM_SIGNATURE_SCORE,
            "min_informative_windows": MIN_INFORMATIVE_LOCAL_WINDOWS.max(config.min_validation_windows),
            "min_informative_window_ratio": MIN_INFORMATIVE_WINDOW_RATIO,
            "min_informative_inlier_ratio": config.min_local_inlier_ratio,
            "min_row_support_ratio": MIN_NARROW_SEAM_ROW_SUPPORT_RATIO,
            "min_column_support_ratio": MIN_NARROW_SEAM_COLUMN_SUPPORT_RATIO,
            "min_seam_support_score": MIN_NARROW_SEAM_SEAM_SUPPORT_SCORE,
            "max_p95_model_residual_px": MAX_INFORMATIVE_MODEL_P95_RESIDUAL_PX,
            "min_objective_score": narrow_seam_objective_floor(config),
        },
    })
}

/// Find the best overlap between the right edge of `left` and the left edge of
/// `right` using normalized cross-correlation.
///
/// Searches a 2D window: horizontal overlaps from 10..=max_overlap and vertical offsets
/// from -max_y_offset..=+max_y_offset to account for mechanical stepper drift.
///
/// Returns `Some((x_offset, y_offset, score))` where `x_offset` is the column in `left`
/// coordinates where `right` starts, `y_offset` is the vertical shift applied to `right`
/// (positive = right image shifted down), or `None` if no good match is found.
pub fn find_overlap_ncc(
    left: &Array3<u16>,
    right: &Array3<u16>,
    max_overlap: usize,
    max_y_offset: i32,
) -> Option<(usize, i32, f64)> {
    let (h_l, w_l, _) = left.dim();
    let (h_r, w_r, _) = right.dim();
    let h = h_l.min(h_r);
    if h < 10 || w_l == 0 || w_r == 0 {
        return None;
    }

    let gray_l = to_grayscale(left);
    let gray_r = to_grayscale(right);
    let (mean_l, grad_l) = column_feature_series(&gray_l);
    let (mean_r, grad_r) = column_feature_series(&gray_r);
    let max_ovl = max_overlap.min(w_l).min(w_r);
    if max_ovl < 10 {
        return None;
    }

    let signature_scores = signature_overlap_scores(&mean_l, &grad_l, &mean_r, &grad_r, max_ovl);
    let exhaustive: Vec<usize> = (10..=max_ovl).collect();
    let best = search_translation_candidates_in_overlaps(
        &gray_l,
        &gray_r,
        &signature_scores,
        0,
        max_y_offset,
        &exhaustive,
        SearchBand::PrimaryExhaustive,
    )
    .into_iter()
    .min_by(compare_raw_translation_candidates);

    best.and_then(|search| {
        if search.global_ncc > 0.3 {
            Some((search.x_offset, search.y_offset, search.global_ncc))
        } else {
            None
        }
    })
}

/// Score a left-right hypothesis. Returns the NCC score or 0.0 if no overlap found.
pub fn score_hypothesis(
    left: &Array3<u16>,
    right: &Array3<u16>,
    max_overlap: usize,
    max_y_offset: i32,
) -> f64 {
    match find_overlap_ncc(left, right, max_overlap, max_y_offset) {
        Some((_, _, score)) => score,
        None => 0.0,
    }
}

/// Evaluate a directional `left -> right` adjacency with the same robust translation evidence
/// used by the production stitch selector. This is intentionally richer than raw NCC so sequence
/// ordering cannot be dominated by a tiny or repetitive false overlap.
pub fn evaluate_directional_overlap(
    left: &Array3<u16>,
    right: &Array3<u16>,
    config: &StitchConfig,
) -> StitchHypothesis {
    evaluate_translation_hypothesis(left, right, StitchOrder::LeftRight, config)
}

fn sequence_state_is_better(
    candidate: &SequenceOrderState,
    current: Option<&SequenceOrderState>,
) -> bool {
    let Some(current) = current else {
        return true;
    };
    if candidate.accepted_edges != current.accepted_edges {
        return candidate.accepted_edges > current.accepted_edges;
    }
    if (candidate.objective_score - current.objective_score).abs() > 1e-12 {
        return candidate.objective_score > current.objective_score;
    }
    candidate.path < current.path
}

fn score_sequence_path(
    path: &[usize],
    hypotheses: &[Vec<Option<StitchHypothesis>>],
) -> SequenceOrderState {
    let mut accepted_edges = 0usize;
    let mut objective_score = 0.0;
    for pair in path.windows(2) {
        if let Some(hypothesis) = hypotheses[pair[0]][pair[1]].as_ref() {
            accepted_edges += usize::from(hypothesis.accepted);
            objective_score += hypothesis.objective_score;
        }
    }
    SequenceOrderState {
        accepted_edges,
        objective_score,
        path: path.to_vec(),
    }
}

fn exact_sequence_order(hypotheses: &[Vec<Option<StitchHypothesis>>]) -> SequenceOrderState {
    let component_count = hypotheses.len();
    let state_count = 1usize << component_count;
    let mut states = vec![vec![None::<SequenceOrderState>; component_count]; state_count];
    for start in 0..component_count {
        states[1usize << start][start] = Some(SequenceOrderState {
            accepted_edges: 0,
            objective_score: 0.0,
            path: vec![start],
        });
    }

    for mask in 1usize..state_count {
        for end in 0..component_count {
            let Some(current) = states[mask][end].clone() else {
                continue;
            };
            for next in 0..component_count {
                if mask & (1usize << next) != 0 {
                    continue;
                }
                let Some(hypothesis) = hypotheses[end][next].as_ref() else {
                    continue;
                };
                let next_mask = mask | (1usize << next);
                let mut candidate = current.clone();
                candidate.accepted_edges += usize::from(hypothesis.accepted);
                candidate.objective_score += hypothesis.objective_score;
                candidate.path.push(next);
                if sequence_state_is_better(&candidate, states[next_mask][next].as_ref()) {
                    states[next_mask][next] = Some(candidate);
                }
            }
        }
    }

    let full_mask = state_count - 1;
    states[full_mask]
        .iter()
        .flatten()
        .cloned()
        .reduce(|current, candidate| {
            if sequence_state_is_better(&candidate, Some(&current)) {
                candidate
            } else {
                current
            }
        })
        .unwrap_or_else(|| SequenceOrderState {
            accepted_edges: 0,
            objective_score: 0.0,
            path: (0..component_count).collect(),
        })
}

fn greedy_sequence_order(hypotheses: &[Vec<Option<StitchHypothesis>>]) -> SequenceOrderState {
    let component_count = hypotheses.len();
    let mut best: Option<SequenceOrderState> = None;
    for start in 0..component_count {
        let mut path = vec![start];
        let mut unused = (0..component_count)
            .filter(|index| *index != start)
            .collect::<BTreeSet<_>>();
        while let Some(&end) = path.last() {
            let Some(next) = unused.iter().copied().max_by(|left, right| {
                let left_hypothesis = hypotheses[end][*left]
                    .as_ref()
                    .expect("non-diagonal sequence hypothesis");
                let right_hypothesis = hypotheses[end][*right]
                    .as_ref()
                    .expect("non-diagonal sequence hypothesis");
                (left_hypothesis.accepted as u8)
                    .cmp(&(right_hypothesis.accepted as u8))
                    .then_with(|| {
                        left_hypothesis
                            .objective_score
                            .partial_cmp(&right_hypothesis.objective_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| right.cmp(left))
            }) else {
                break;
            };
            path.push(next);
            unused.remove(&next);
        }
        let candidate = score_sequence_path(&path, hypotheses);
        if sequence_state_is_better(&candidate, best.as_ref()) {
            best = Some(candidate);
        }
    }
    best.unwrap_or_else(|| SequenceOrderState {
        accepted_edges: 0,
        objective_score: 0.0,
        path: (0..component_count).collect(),
    })
}

/// Infer a left-to-right scan order from a complete directed overlap graph. Typical film scan
/// counts use an exact Hamiltonian-path search; unusually large sets use a deterministic greedy
/// search to avoid exponential memory growth.
pub fn infer_component_sequence_order(
    components: &[&Array3<u16>],
    config: &StitchConfig,
) -> (Vec<usize>, SequenceOrderingDiagnostics) {
    let vertical_crop_origins = vec![0i32; components.len()];
    infer_component_sequence_order_with_vertical_origins(components, &vertical_crop_origins, config)
}

/// Origin-aware sequence ordering used after each input has been cropped independently. Each
/// origin is the number of source rows removed from that component's top edge.
pub fn infer_component_sequence_order_with_vertical_origins(
    components: &[&Array3<u16>],
    vertical_crop_origins: &[i32],
    config: &StitchConfig,
) -> (Vec<usize>, SequenceOrderingDiagnostics) {
    let component_count = components.len();
    let vertical_crop_origins = if vertical_crop_origins.len() == component_count {
        vertical_crop_origins.to_vec()
    } else {
        vec![0i32; component_count]
    };
    if component_count <= 1 {
        let order = (0..component_count).collect::<Vec<_>>();
        return (
            order.clone(),
            SequenceOrderingDiagnostics {
                method: "single_component",
                inferred_order: order.iter().map(|index| index + 1).collect(),
                accepted_adjacent_edges: 0,
                required_adjacent_edges: 0,
                total_objective_score: 0.0,
                all_adjacencies_validated: true,
                pairwise_edges: Vec::new(),
            },
        );
    }

    let mut ordering_config = config.clone();
    ordering_config.debug_dir = None;
    ordering_config.target_width = None;
    ordering_config.target_height = None;
    let pairs = (0..component_count)
        .flat_map(|from| {
            (0..component_count)
                .filter(move |to| *to != from)
                .map(move |to| (from, to))
        })
        .collect::<Vec<_>>();
    let evaluations = pairs
        .par_iter()
        .map(|&(from, to)| {
            let mut pair_config = ordering_config.clone();
            pair_config.expected_y_offset =
                vertical_crop_origins[to].saturating_sub(vertical_crop_origins[from]);
            (
                from,
                to,
                evaluate_directional_overlap(components[from], components[to], &pair_config),
            )
        })
        .collect::<Vec<_>>();
    let mut hypotheses = vec![vec![None::<StitchHypothesis>; component_count]; component_count];
    for (from, to, hypothesis) in evaluations {
        hypotheses[from][to] = Some(hypothesis);
    }

    const EXACT_SEQUENCE_ORDER_LIMIT: usize = 12;
    let (method, selected) = if component_count <= EXACT_SEQUENCE_ORDER_LIMIT {
        (
            "exact_validated_overlap_hamiltonian_path",
            exact_sequence_order(&hypotheses),
        )
    } else {
        (
            "deterministic_validated_overlap_greedy_path",
            greedy_sequence_order(&hypotheses),
        )
    };
    let required_adjacent_edges = component_count.saturating_sub(1);
    let mut pairwise_edges = Vec::with_capacity(component_count * (component_count - 1));
    for (from, row) in hypotheses.iter().enumerate() {
        for (to, hypothesis) in row.iter().enumerate() {
            let Some(hypothesis) = hypothesis else {
                continue;
            };
            pairwise_edges.push(SequenceOrderingEdge {
                from_component: from + 1,
                to_component: to + 1,
                accepted: hypothesis.accepted,
                objective_score: hypothesis.objective_score,
                global_ncc_score: hypothesis.validation.global_ncc_score,
                overlap_width_px: hypothesis.overlap_width,
                expected_vertical_offset_px: vertical_crop_origins[to]
                    .saturating_sub(vertical_crop_origins[from]),
                vertical_offset_px: hypothesis.vertical_offset,
                rejection_reason: hypothesis.rejection_reason.clone(),
            });
        }
    }
    let diagnostics = SequenceOrderingDiagnostics {
        method,
        inferred_order: selected.path.iter().map(|index| index + 1).collect(),
        accepted_adjacent_edges: selected.accepted_edges,
        required_adjacent_edges,
        total_objective_score: selected.objective_score,
        all_adjacencies_validated: selected.accepted_edges == required_adjacent_edges,
        pairwise_edges,
    };
    (selected.path, diagnostics)
}

/// Stitch one or more components into a single union. For sequences, ordering is inferred from a
/// validated directed overlap graph and every incremental merge must pass the normal production
/// stitch gates. Target cropping is disabled so accepted geometry is never silently discarded.
pub fn stitch_component_sequence(
    components: &[&Array3<u16>],
    config: &StitchConfig,
) -> StitchSequenceResult {
    let vertical_crop_origins = vec![0i32; components.len()];
    stitch_component_sequence_with_vertical_origins(components, &vertical_crop_origins, config)
}

/// Stitch a sequence while preserving the coordinate shift introduced by independent top-edge
/// border crops. This keeps every pair's narrow mechanical-drift search centered on the geometry
/// implied by its source crop rather than on an incorrect zero offset.
pub fn stitch_component_sequence_with_vertical_origins(
    components: &[&Array3<u16>],
    vertical_crop_origins: &[i32],
    config: &StitchConfig,
) -> StitchSequenceResult {
    if components.is_empty() {
        return StitchSequenceResult {
            result: None,
            order: Vec::new(),
            report: PhaseReport::fail("stitch", "no input components were supplied"),
        };
    }
    if components.len() == 1 {
        return StitchSequenceResult {
            result: Some(components[0].clone()),
            order: vec![0],
            report: PhaseReport::ok(
                "stitch",
                0.0,
                serde_json::json!({
                    "decision": "skipped_single_input",
                    "stitch_evidence_evaluated": false,
                    "confidence_basis": "not_applicable_single_input",
                    "user_override": false,
                    "input_count": 1,
                    "inferred_order": [1],
                    "preserved_full_valid_union": true,
                }),
            ),
        };
    }

    let vertical_crop_origins = if vertical_crop_origins.len() == components.len() {
        vertical_crop_origins.to_vec()
    } else {
        vec![0i32; components.len()]
    };
    let (order, ordering) = infer_component_sequence_order_with_vertical_origins(
        components,
        &vertical_crop_origins,
        config,
    );
    let mut pair_reports = Vec::<serde_json::Value>::new();
    let mut warnings = Vec::<String>::new();
    let mut seam_quality_review_reasons = Vec::<String>::new();
    let mut confidence = 1.0f64;
    let mut merged_component_ids = vec![order[0] + 1];
    let mut mosaic = components[order[0]].clone();
    let mut mosaic_vertical_origin = vertical_crop_origins[order[0]];
    if !ordering.all_adjacencies_validated {
        warnings.push(format!(
            "sequence ordering graph validated only {} of {} selected adjacencies; incremental stitching must independently validate every merge",
            ordering.accepted_adjacent_edges, ordering.required_adjacent_edges
        ));
    }

    for (step, &next_index) in order.iter().enumerate().skip(1) {
        let mut pair_config = config.clone();
        pair_config.target_width = None;
        pair_config.target_height = None;
        pair_config.expected_y_offset =
            vertical_crop_origins[next_index].saturating_sub(mosaic_vertical_origin);
        if let Some(debug_root) = config.debug_dir.as_ref() {
            let pair_dir = debug_root.join(format!(
                "stitch_sequence_pair_{:02}_component_{:02}",
                step,
                next_index + 1
            ));
            match std::fs::create_dir_all(&pair_dir) {
                Ok(()) => pair_config.debug_dir = Some(pair_dir),
                Err(error) => {
                    pair_config.debug_dir = None;
                    warnings.push(format!(
                        "could not create sequence stitch debug directory for step {}: {}",
                        step, error
                    ));
                }
            }
        }
        let pair = stitch_components(&mosaic, components[next_index], &pair_config);
        confidence = confidence.min(pair.report.confidence);
        warnings.extend(
            pair.report
                .warnings
                .iter()
                .map(|warning| format!("sequence stitch step {step}: {warning}")),
        );
        let pair_success = pair.result.is_some();
        if pair
            .report
            .metrics
            .get("seam_blend")
            .and_then(|blend| blend.get("review_required"))
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            let reason = pair
                .report
                .metrics
                .get("seam_blend")
                .and_then(|blend| blend.get("review_reason"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("seam detail or gradient evidence requires review");
            seam_quality_review_reasons.push(format!(
                "sequence stitch step {step}, component {}: {reason}",
                next_index + 1
            ));
        }
        pair_reports.push(serde_json::json!({
            "step": step,
            "mosaic_components_before": merged_component_ids.clone(),
            "next_component": next_index + 1,
            "mosaic_vertical_origin_before": mosaic_vertical_origin,
            "next_component_vertical_origin": vertical_crop_origins[next_index],
            "expected_y_offset": pair_config.expected_y_offset,
            "accepted": pair_success,
            "x_offset": pair.x_offset,
            "y_offset": pair.y_offset,
            "ncc_score": pair.ncc_score,
            "detected_order": pair.order.label(),
            "pair_report": pair.report,
        }));
        let Some(stitched) = pair.result else {
            let message = format!(
                "sequence stitch step {step} rejected while adding component {}; no partial mosaic was accepted as complete",
                next_index + 1
            );
            return StitchSequenceResult {
                result: None,
                order,
                report: PhaseReport {
                    name: "stitch".to_string(),
                    success: false,
                    confidence,
                    duration_ms: 0,
                    metrics: serde_json::json!({
                        "decision": "rejected_sequence",
                        "input_count": components.len(),
                        "ordering": ordering,
                        "accepted_merge_count": step - 1,
                        "required_merge_count": components.len() - 1,
                        "pair_merges": pair_reports,
                        "seam_quality_review_required": !seam_quality_review_reasons.is_empty(),
                        "seam_quality_review_reasons": seam_quality_review_reasons,
                        "preserved_full_valid_union": false,
                        "target_crop_disabled": true,
                    }),
                    warnings,
                    errors: vec![message],
                },
            };
        };
        mosaic_vertical_origin = match pair.order {
            StitchOrder::LeftRight => mosaic_vertical_origin.saturating_add(pair.y_offset.min(0)),
            StitchOrder::RightLeft => {
                vertical_crop_origins[next_index].saturating_add(pair.y_offset.min(0))
            }
            StitchOrder::NoStitch => mosaic_vertical_origin,
        };
        mosaic = stitched;
        merged_component_ids.push(next_index + 1);
    }

    let output_shape = [mosaic.shape()[0], mosaic.shape()[1]];
    StitchSequenceResult {
        result: Some(mosaic),
        order,
        report: PhaseReport {
            name: "stitch".to_string(),
            success: true,
            confidence,
            duration_ms: 0,
            metrics: serde_json::json!({
                "decision": "accepted_sequence",
                "input_count": components.len(),
                "ordering": ordering,
                "vertical_crop_origins": vertical_crop_origins,
                "output_vertical_origin": mosaic_vertical_origin,
                "accepted_merge_count": components.len() - 1,
                "required_merge_count": components.len() - 1,
                "pair_merges": pair_reports,
                "seam_quality_review_required": !seam_quality_review_reasons.is_empty(),
                "seam_quality_review_reasons": seam_quality_review_reasons,
                "output_shape": output_shape,
                "preserved_full_valid_union": true,
                "target_crop_disabled": true,
            }),
            warnings,
            errors: Vec::new(),
        },
    }
}

/// Stitch two image components together, automatically detecting the correct order.
///
/// Accounts for vertical mechanical drift by searching a 2D window and placing the
/// right component at the detected y_offset on an expanded canvas.
pub fn stitch_components(
    comp1: &Array3<u16>,
    comp2: &Array3<u16>,
    config: &StitchConfig,
) -> StitchResult {
    let stitch_start = Instant::now();
    let opencv_available = opencv_match::is_available();
    let mut report_warnings = Vec::<String>::new();
    if config.use_opencv && !opencv_available {
        report_warnings.push(
            "OpenCV was requested at runtime, but this binary was built without the `use-opencv` feature."
                .to_string(),
        );
    }
    if !config.use_opencv && matches!(config.transform_mode, TransformMode::Homography) {
        report_warnings.push(
            "The requested homography mode needs the OpenCV backend at runtime; falling back to translation stitching."
                .to_string(),
        );
    }
    let translation_start = Instant::now();
    let mut hypotheses = vec![
        evaluate_translation_hypothesis(comp1, comp2, StitchOrder::LeftRight, config),
        evaluate_translation_hypothesis(comp2, comp1, StitchOrder::RightLeft, config),
    ];
    let translation_evaluation_duration_ms = elapsed_ms_u64(translation_start);
    if let Some(debug_dir) = &config.debug_dir {
        save_hypothesis_debug_artifacts(comp1, comp2, &hypotheses, debug_dir);
    }

    let best_index = {
        let rank0 = (
            hypotheses[0].accepted as u8,
            hypotheses[0].objective_score,
            hypotheses[0]
                .search
                .selected_candidate_search_score
                .unwrap_or(0.0),
            hypotheses[0].validation.global_ncc_score,
            hypotheses[0]
                .vertical_offset
                .map(|dy| -(dy.unsigned_abs() as i32))
                .unwrap_or(i32::MIN),
        );
        let rank1 = (
            hypotheses[1].accepted as u8,
            hypotheses[1].objective_score,
            hypotheses[1]
                .search
                .selected_candidate_search_score
                .unwrap_or(0.0),
            hypotheses[1].validation.global_ncc_score,
            hypotheses[1]
                .vertical_offset
                .map(|dy| -(dy.unsigned_abs() as i32))
                .unwrap_or(i32::MIN),
        );
        if rank1 > rank0 {
            1
        } else {
            0
        }
    };
    let other_index = 1 - best_index;
    let order_gap =
        (hypotheses[best_index].objective_score - hypotheses[other_index].objective_score).abs();

    let best_order = if best_index == 0 {
        StitchOrder::LeftRight
    } else {
        StitchOrder::RightLeft
    };
    let (left, right) = if matches!(best_order, StitchOrder::LeftRight) {
        (comp1, comp2)
    } else {
        (comp2, comp1)
    };

    let best_x_offset = hypotheses[best_index].x_offset.unwrap_or(0);
    let best_y_offset = hypotheses[best_index].vertical_offset.unwrap_or(0);
    let best_overlap = hypotheses[best_index].overlap_width.unwrap_or(0);
    let best_ncc = hypotheses[best_index].validation.global_ncc_score;
    let translation_was_accepted = hypotheses[best_index].accepted;
    let mut native_affine_diagnostics = None::<NativeAffineDiagnostics>;
    let mut native_affine_duration_ms = None::<u64>;
    if best_overlap > 0 {
        if let Some(trigger) = native_affine_trigger(&hypotheses[best_index], config.transform_mode)
        {
            let affine_start = Instant::now();
            let mut estimate = estimate_native_affine(
                left,
                right,
                best_x_offset,
                best_y_offset,
                best_overlap,
                trigger,
            );
            native_affine_duration_ms = Some(elapsed_ms_u64(affine_start));
            if estimate.diagnostics.accepted {
                hypotheses[best_index].accepted = true;
                hypotheses[best_index].rejection_reason = None;
                hypotheses[best_index].acceptance_reason =
                    Some(estimate.diagnostics.reason.clone());
                hypotheses[best_index].transform_model = "native_affine".to_string();
                match stitch_native_affine(
                    left,
                    right,
                    best_x_offset,
                    best_y_offset,
                    best_overlap,
                    best_order,
                    config,
                    &hypotheses,
                    best_index,
                    estimate.diagnostics.clone(),
                    report_warnings.clone(),
                ) {
                    Ok(mut result) => {
                        attach_stitch_runtime(
                            &mut result.report,
                            stitch_start,
                            translation_evaluation_duration_ms,
                            None,
                        );
                        attach_native_affine_diagnostics(
                            &mut result.report,
                            Some(&estimate.diagnostics),
                            native_affine_duration_ms,
                        );
                        return result;
                    }
                    Err(reason) => {
                        estimate.diagnostics.accepted = false;
                        estimate.diagnostics.reason = format!(
                            "affine registration passed, but full-union composition failed: {reason}"
                        );
                        hypotheses[best_index].accepted = translation_was_accepted;
                        hypotheses[best_index].transform_model = "translation".to_string();
                        if !translation_was_accepted {
                            hypotheses[best_index].rejection_reason = Some(reason.clone());
                        }
                        report_warnings.push(format!(
                            "Native affine composition failed ({reason}); falling back to the next justified model."
                        ));
                    }
                }
            } else if matches!(config.transform_mode, TransformMode::Affine) {
                report_warnings.push(format!(
                    "Requested affine model was rejected: {}; falling back to translation only if its independent validation passed.",
                    estimate.diagnostics.reason
                ));
            }
            native_affine_diagnostics = Some(estimate.diagnostics);
        }
    }
    let attempt_homography = config.use_opencv
        && opencv_available
        && (matches!(config.transform_mode, TransformMode::Homography)
            || !hypotheses[best_index].accepted);

    let homography_start = if attempt_homography && best_overlap > 0 {
        Some(Instant::now())
    } else {
        None
    };

    if let Some(homography_start) = homography_start {
        let (h_l, w_l, _) = left.dim();
        let (h_r, w_r, _) = right.dim();
        let strip_l = left.slice(s![.., (w_l - best_overlap).., ..]).to_owned();
        let strip_r = right.slice(s![.., ..best_overlap, ..]).to_owned();
        let homography =
            opencv_match::find_homography_robust(&strip_l, &strip_r, config.input_bit_depth);

        if let Some(ref mr) = homography {
            let plausibility =
                assess_transform_plausibility(mr.transform, best_overlap, h_l.min(h_r), w_r, h_r);
            let feature_quality = assess_homography_feature_quality(mr);
            hypotheses[best_index].validation.homography_inliers = Some(mr.inliers);
            hypotheses[best_index].validation.homography_median_error = Some(mr.median_error);
            hypotheses[best_index].validation.homography_p95_error = Some(mr.p95_error);
            hypotheses[best_index]
                .validation
                .homography_feature_validation = Some(HomographyFeatureValidationMetrics::from(mr));

            if feature_quality.is_ok() && plausibility.is_ok() {
                let h_inv = match invert_3x3(mr.transform) {
                    Some(inv) => inv,
                    None => {
                        report_warnings.push(
                            "Homography matrix was singular; falling back to translation stitching."
                                .to_string(),
                        );
                        if !translation_was_accepted {
                            let reason = hypotheses[best_index]
                                .rejection_reason
                                .clone()
                                .unwrap_or_else(|| {
                                    "best stitch hypothesis rejected after scoring".to_string()
                                });
                            let mut report = stitch_failure_report(
                                config,
                                opencv_available,
                                &hypotheses,
                                best_index,
                                reason,
                                report_warnings,
                            );
                            attach_stitch_runtime(
                                &mut report,
                                stitch_start,
                                translation_evaluation_duration_ms,
                                Some(elapsed_ms_u64(homography_start)),
                            );
                            attach_native_affine_diagnostics(
                                &mut report,
                                native_affine_diagnostics.as_ref(),
                                native_affine_duration_ms,
                            );
                            return StitchResult {
                                result: None,
                                x_offset: best_x_offset as i32,
                                y_offset: best_y_offset,
                                ncc_score: best_ncc,
                                order: StitchOrder::NoStitch,
                                report,
                            };
                        }
                        let mut result = stitch_ncc_fallback(
                            left,
                            right,
                            best_x_offset,
                            best_y_offset,
                            best_overlap,
                            best_order,
                            config,
                            &hypotheses,
                            best_index,
                            report_warnings,
                        );
                        attach_stitch_runtime(
                            &mut result.report,
                            stitch_start,
                            translation_evaluation_duration_ms,
                            Some(elapsed_ms_u64(homography_start)),
                        );
                        attach_native_affine_diagnostics(
                            &mut result.report,
                            native_affine_diagnostics.as_ref(),
                            native_affine_duration_ms,
                        );
                        return result;
                    }
                };

                let t_l_inv: [[f64; 3]; 3] = [
                    [1.0, 0.0, best_x_offset as f64],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ];
                let right_to_left = mat_mul_3x3(t_l_inv, h_inv);
                let homography_spatial_validation = validate_homography_spatial_evidence(
                    left,
                    right,
                    best_x_offset,
                    best_y_offset,
                    best_overlap,
                    right_to_left,
                );
                if !homography_spatial_validation.accepted {
                    report_warnings.push(format!(
                        "Homography failed spatial model validation: {}; falling back to translation stitching.",
                        homography_spatial_validation.reason
                    ));
                    if !translation_was_accepted {
                        let reason = hypotheses[best_index]
                            .rejection_reason
                            .clone()
                            .unwrap_or_else(|| {
                                "best stitch hypothesis rejected after scoring".to_string()
                            });
                        let mut report = stitch_failure_report(
                            config,
                            opencv_available,
                            &hypotheses,
                            best_index,
                            reason,
                            report_warnings,
                        );
                        attach_stitch_runtime(
                            &mut report,
                            stitch_start,
                            translation_evaluation_duration_ms,
                            Some(elapsed_ms_u64(homography_start)),
                        );
                        attach_native_affine_diagnostics(
                            &mut report,
                            native_affine_diagnostics.as_ref(),
                            native_affine_duration_ms,
                        );
                        attach_homography_spatial_validation(
                            &mut report,
                            &homography_spatial_validation,
                        );
                        return StitchResult {
                            result: None,
                            x_offset: best_x_offset as i32,
                            y_offset: best_y_offset,
                            ncc_score: best_ncc,
                            order: StitchOrder::NoStitch,
                            report,
                        };
                    }
                    let mut result = stitch_ncc_fallback(
                        left,
                        right,
                        best_x_offset,
                        best_y_offset,
                        best_overlap,
                        best_order,
                        config,
                        &hypotheses,
                        best_index,
                        report_warnings,
                    );
                    attach_stitch_runtime(
                        &mut result.report,
                        stitch_start,
                        translation_evaluation_duration_ms,
                        Some(elapsed_ms_u64(homography_start)),
                    );
                    attach_native_affine_diagnostics(
                        &mut result.report,
                        native_affine_diagnostics.as_ref(),
                        native_affine_duration_ms,
                    );
                    attach_homography_spatial_validation(
                        &mut result.report,
                        &homography_spatial_validation,
                    );
                    return result;
                }
                let composition = match compose_projective_pair(left, right, right_to_left, config)
                {
                    Ok(composition) => composition,
                    Err(composition_error) => {
                        report_warnings.push(format!(
                            "Homography compositor rejected the projective canvas: {composition_error}; falling back to translation stitching."
                        ));
                        if !translation_was_accepted {
                            let reason = hypotheses[best_index]
                                .rejection_reason
                                .clone()
                                .unwrap_or_else(|| {
                                    "best stitch hypothesis rejected after scoring".to_string()
                                });
                            let mut report = stitch_failure_report(
                                config,
                                opencv_available,
                                &hypotheses,
                                best_index,
                                reason,
                                report_warnings,
                            );
                            attach_stitch_runtime(
                                &mut report,
                                stitch_start,
                                translation_evaluation_duration_ms,
                                Some(elapsed_ms_u64(homography_start)),
                            );
                            attach_native_affine_diagnostics(
                                &mut report,
                                native_affine_diagnostics.as_ref(),
                                native_affine_duration_ms,
                            );
                            attach_homography_spatial_validation(
                                &mut report,
                                &homography_spatial_validation,
                            );
                            return StitchResult {
                                result: None,
                                x_offset: best_x_offset as i32,
                                y_offset: best_y_offset,
                                ncc_score: best_ncc,
                                order: StitchOrder::NoStitch,
                                report,
                            };
                        }
                        let mut result = stitch_ncc_fallback(
                            left,
                            right,
                            best_x_offset,
                            best_y_offset,
                            best_overlap,
                            best_order,
                            config,
                            &hypotheses,
                            best_index,
                            report_warnings,
                        );
                        attach_stitch_runtime(
                            &mut result.report,
                            stitch_start,
                            translation_evaluation_duration_ms,
                            Some(elapsed_ms_u64(homography_start)),
                        );
                        attach_native_affine_diagnostics(
                            &mut result.report,
                            native_affine_diagnostics.as_ref(),
                            native_affine_duration_ms,
                        );
                        attach_homography_spatial_validation(
                            &mut result.report,
                            &homography_spatial_validation,
                        );
                        return result;
                    }
                };

                hypotheses[best_index].accepted = true;
                hypotheses[best_index].rejection_reason = None;
                hypotheses[best_index].transform_model =
                    if matches!(config.transform_mode, TransformMode::Affine) {
                        "homography_fallback_for_affine_request".to_string()
                    } else {
                        "homography".to_string()
                    };

                let ProjectivePairComposition {
                    image,
                    right_to_canvas,
                    left_origin,
                    canvas_width,
                    canvas_height,
                    valid_union_pixel_count,
                    canvas_void_pixel_count,
                    overlap_pixel_count,
                    overlap_outside_multiband_pixel_count,
                    multiband_overlap_rectangle,
                    seam_exposure_correction,
                    seam_blend,
                } = composition;
                let (stitched, crop_info) = apply_target_crop(&image, config);
                let preserved_full_valid_union = crop_info.is_none();
                if let Some(debug_dir) = &config.debug_dir {
                    save_final_seam_overlay(
                        left,
                        right,
                        best_overlap,
                        best_y_offset,
                        best_order,
                        debug_dir,
                    );
                }

                let mut report = PhaseReport::ok(
                    "stitch",
                    hypotheses[best_index].objective_score,
                    serde_json::json!({
                        "decision": "accepted",
                        "method": "homography",
                        "requested_transform": config.transform_mode.as_str(),
                        "transform_model_used": hypotheses[best_index].transform_model,
                        "opencv_requested": config.use_opencv,
                        "opencv_available": opencv_available,
                        "chosen_hypothesis": hypotheses[best_index].ordering,
                        "hypotheses": hypotheses,
                        "acceptance_thresholds": acceptance_thresholds_json(config),
                        "x_offset": best_x_offset,
                        "y_offset": best_y_offset,
                        "overlap": best_overlap,
                        "objective_score": hypotheses[best_index].objective_score,
                        "order_gap": order_gap,
                        "total_width": canvas_width,
                        "canvas_height": canvas_height,
                        "valid_union_pixel_count": valid_union_pixel_count,
                        "canvas_void_pixel_count": canvas_void_pixel_count,
                        "overlap_pixel_count": overlap_pixel_count,
                        "overlap_outside_multiband_pixel_count": overlap_outside_multiband_pixel_count,
                        "preserved_full_valid_union": preserved_full_valid_union,
                        "target_crop_disabled": config.target_width.is_none() && config.target_height.is_none(),
                        "compositor": "validity_masked_seam_aware_multiband",
                        "interpolation": "bicubic_catmull_rom_single_resample",
                        "exposure_gain": seam_exposure_correction.gain_rgb,
                        "exposure_offset": seam_exposure_correction.offset_rgb,
                        "seam_exposure_correction": seam_exposure_correction,
                        "seam_blend": seam_blend,
                        "homography_matrix": mr.transform,
                        "homography_matrix_left_overlap_to_right_overlap": mr.transform,
                        "homography_feature_validation": HomographyFeatureValidationMetrics::from(mr),
                        "projective_matrix_right_to_left": right_to_left,
                        "projective_matrix_right_to_canvas": right_to_canvas,
                        "homography_spatial_validation": homography_spatial_validation,
                        "left_canvas_origin": left_origin,
                        "multiband_overlap_rectangle": {
                            "x": multiband_overlap_rectangle.x,
                            "y": multiband_overlap_rectangle.y,
                            "width": multiband_overlap_rectangle.width,
                            "height": multiband_overlap_rectangle.height,
                        },
                        "crop": crop_metrics_json(crop_info),
                    }),
                );
                report.warnings.extend(report_warnings);
                attach_stitch_runtime(
                    &mut report,
                    stitch_start,
                    translation_evaluation_duration_ms,
                    Some(elapsed_ms_u64(homography_start)),
                );
                attach_native_affine_diagnostics(
                    &mut report,
                    native_affine_diagnostics.as_ref(),
                    native_affine_duration_ms,
                );

                return StitchResult {
                    result: Some(stitched),
                    x_offset: best_x_offset as i32,
                    y_offset: best_y_offset,
                    ncc_score: best_ncc,
                    order: best_order,
                    report,
                };
            }

            if let Err(reason) = feature_quality {
                report_warnings.push(format!(
                    "Homography failed disjoint feature validation ({reason}); falling back to translation stitching."
                ));
            }
            if let Err(reason) = plausibility {
                report_warnings.push(format!(
                    "Homography failed plausibility checks ({}); falling back to translation stitching.",
                    reason
                ));
            }
        } else {
            report_warnings.push(
                "OpenCV matching did not produce a homography; falling back to translation stitching."
                    .to_string(),
            );
        }
    }

    if !hypotheses[best_index].accepted {
        let reason = hypotheses[best_index]
            .rejection_reason
            .clone()
            .unwrap_or_else(|| "best stitch hypothesis rejected after scoring".to_string());
        let mut report = stitch_failure_report(
            config,
            opencv_available,
            &hypotheses,
            best_index,
            reason,
            report_warnings,
        );
        attach_stitch_runtime(
            &mut report,
            stitch_start,
            translation_evaluation_duration_ms,
            homography_start.map(elapsed_ms_u64),
        );
        attach_native_affine_diagnostics(
            &mut report,
            native_affine_diagnostics.as_ref(),
            native_affine_duration_ms,
        );
        return StitchResult {
            result: None,
            x_offset: best_x_offset as i32,
            y_offset: best_y_offset,
            ncc_score: best_ncc,
            order: StitchOrder::NoStitch,
            report,
        };
    }

    let mut result = stitch_ncc_fallback(
        left,
        right,
        best_x_offset,
        best_y_offset,
        best_overlap,
        best_order,
        config,
        &hypotheses,
        best_index,
        report_warnings,
    );
    attach_stitch_runtime(
        &mut result.report,
        stitch_start,
        translation_evaluation_duration_ms,
        homography_start.map(elapsed_ms_u64),
    );
    attach_native_affine_diagnostics(
        &mut result.report,
        native_affine_diagnostics.as_ref(),
        native_affine_duration_ms,
    );
    result
}

fn assess_homography_feature_quality(result: &opencv_match::MatchResult) -> Result<(), String> {
    if !result.disjoint_validation_passed {
        return Err(result.validation_reason.clone());
    }
    if result.inliers < HOMOGRAPHY_MIN_TRAINING_INLIERS {
        return Err(format!(
            "training partition retained {} inliers; need {}",
            result.inliers, HOMOGRAPHY_MIN_TRAINING_INLIERS
        ));
    }
    if !result.median_error.is_finite()
        || !result.p95_error.is_finite()
        || result.median_error > HOMOGRAPHY_MAX_HELD_OUT_MEDIAN_ERROR_PX
        || result.p95_error > HOMOGRAPHY_MAX_HELD_OUT_P95_ERROR_PX
    {
        return Err(format!(
            "held-out residuals were too large (median {:.2}px, p95 {:.2}px; limits {:.2}px/{:.2}px)",
            result.median_error,
            result.p95_error,
            HOMOGRAPHY_MAX_HELD_OUT_MEDIAN_ERROR_PX,
            HOMOGRAPHY_MAX_HELD_OUT_P95_ERROR_PX
        ));
    }
    Ok(())
}

fn assess_transform_plausibility(
    m: [[f64; 3]; 3],
    overlap: usize,
    overlap_h: usize,
    full_w: usize,
    full_h: usize,
) -> Result<(), String> {
    let col0 = [m[0][0], m[1][0]];
    let col1 = [m[0][1], m[1][1]];
    let scale_x = (col0[0] * col0[0] + col0[1] * col0[1]).sqrt();
    let scale_y = (col1[0] * col1[0] + col1[1] * col1[1]).sqrt();
    let rotation_deg = col0[1].atan2(col0[0]).to_degrees().abs();
    let shear = if scale_x > 1e-6 && scale_y > 1e-6 {
        (col0[0] * col1[0] + col0[1] * col1[1]).abs() / (scale_x * scale_y)
    } else {
        1.0
    };
    let perspective = (m[2][0].abs() * overlap.max(full_w) as f64)
        .max(m[2][1].abs() * overlap_h.max(full_h) as f64);

    if !(0.95..=1.05).contains(&scale_x) || !(0.95..=1.05).contains(&scale_y) {
        return Err(format!(
            "scale out of range ({:.4}, {:.4})",
            scale_x, scale_y
        ));
    }
    if rotation_deg > 6.0 {
        return Err(format!("rotation too large ({:.2} deg)", rotation_deg));
    }
    if shear > 0.08 {
        return Err(format!("shear too large ({:.4})", shear));
    }
    if perspective > 0.02 {
        return Err(format!("perspective too large ({:.5})", perspective));
    }
    Ok(())
}

fn elapsed_ms_u64(start: Instant) -> u64 {
    start.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

fn attach_stitch_runtime(
    report: &mut PhaseReport,
    total_start: Instant,
    translation_evaluation_duration_ms: u64,
    homography_duration_ms: Option<u64>,
) {
    let seam_review_reason = report
        .metrics
        .get("seam_blend")
        .filter(|blend| {
            blend
                .get("review_required")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
        })
        .and_then(|blend| blend.get("review_reason"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    if let serde_json::Value::Object(metrics) = &mut report.metrics {
        metrics.insert(
            "runtime".to_string(),
            serde_json::json!({
                "total_ms": elapsed_ms_u64(total_start),
                "translation_evaluation_ms": translation_evaluation_duration_ms,
                "homography_attempted": homography_duration_ms.is_some(),
                "homography_ms": homography_duration_ms,
            }),
        );
    }
    if let Some(reason) = seam_review_reason {
        report.confidence = report.confidence.min(0.35);
        let warning = format!("accepted stitch requires seam-quality review: {reason}");
        if !report.warnings.contains(&warning) {
            report.warnings.push(warning);
        }
    }
}

fn attach_native_affine_diagnostics(
    report: &mut PhaseReport,
    diagnostics: Option<&NativeAffineDiagnostics>,
    duration_ms: Option<u64>,
) {
    let serde_json::Value::Object(metrics) = &mut report.metrics else {
        return;
    };
    metrics.insert(
        "native_affine".to_string(),
        diagnostics.map_or(serde_json::Value::Null, |diagnostics| {
            serde_json::to_value(diagnostics).expect("native affine diagnostics should serialize")
        }),
    );
    if let Some(serde_json::Value::Object(runtime)) = metrics.get_mut("runtime") {
        runtime.insert(
            "native_affine_attempted".to_string(),
            serde_json::Value::Bool(diagnostics.is_some()),
        );
        runtime.insert(
            "native_affine_ms".to_string(),
            duration_ms.map_or(serde_json::Value::Null, serde_json::Value::from),
        );
    }
}

fn attach_homography_spatial_validation(
    report: &mut PhaseReport,
    diagnostics: &HomographySpatialValidationDiagnostics,
) {
    let serde_json::Value::Object(metrics) = &mut report.metrics else {
        return;
    };
    metrics.insert(
        "homography_spatial_validation".to_string(),
        serde_json::to_value(diagnostics).expect("homography spatial diagnostics should serialize"),
    );
}

fn crop_metrics_json(crop_info: Option<CropInfo>) -> serde_json::Value {
    match crop_info {
        Some(crop) => serde_json::json!({
            "x_start": crop.x_start,
            "y_start": crop.y_start,
            "width": crop.width,
            "height": crop.height,
        }),
        None => serde_json::Value::Null,
    }
}

fn aligned_overlap_strips(
    left: &Array3<u16>,
    right: &Array3<u16>,
    overlap: usize,
    y_offset: i32,
) -> Option<(Array3<u16>, Array3<u16>)> {
    let (h_l, w_l, _) = left.dim();
    let (h_r, w_r, _) = right.dim();
    if overlap == 0 || overlap > w_l || overlap > w_r {
        return None;
    }

    let abs_dy = y_offset.unsigned_abs() as usize;
    let cmp_h = h_l.min(h_r).saturating_sub(abs_dy);
    if cmp_h == 0 {
        return None;
    }

    let (l_y_start, r_y_start) = if y_offset >= 0 {
        (y_offset as usize, 0usize)
    } else {
        (0usize, abs_dy)
    };

    Some((
        left.slice(s![l_y_start..(l_y_start + cmp_h), (w_l - overlap)..w_l, ..])
            .to_owned(),
        right
            .slice(s![r_y_start..(r_y_start + cmp_h), 0..overlap, ..])
            .to_owned(),
    ))
}

fn seam_overlay_from_strips(left_strip: &Array3<u16>, right_strip: &Array3<u16>) -> Array3<u16> {
    let (h, w, _) = left_strip.dim();
    let mut overlay = Array3::<u16>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            let l_luma = (left_strip[[y, x, 0]] as f64 * 0.2126
                + left_strip[[y, x, 1]] as f64 * 0.7152
                + left_strip[[y, x, 2]] as f64 * 0.0722)
                .round()
                .clamp(0.0, u16::MAX as f64) as u16;
            let r_luma = (right_strip[[y, x, 0]] as f64 * 0.2126
                + right_strip[[y, x, 1]] as f64 * 0.7152
                + right_strip[[y, x, 2]] as f64 * 0.0722)
                .round()
                .clamp(0.0, u16::MAX as f64) as u16;
            overlay[[y, x, 0]] = l_luma;
            overlay[[y, x, 1]] = r_luma;
            overlay[[y, x, 2]] = (((l_luma as u32 + r_luma as u32) / 2) as u16).max(1);
        }
    }
    overlay
}

fn save_debug_tiff(path: PathBuf, img: &Array3<u16>) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(err) = tiff_io::save_tiff_u16(img, &path) {
        log::warn!(
            "failed to save stitch debug artifact {}: {}",
            path.display(),
            err
        );
    }
}

fn save_hypothesis_debug_artifacts(
    comp1: &Array3<u16>,
    comp2: &Array3<u16>,
    hypotheses: &[StitchHypothesis],
    debug_dir: &Path,
) {
    for hypothesis in hypotheses {
        let (left, right, suffix) = match hypothesis.ordering.as_str() {
            "[1|2]" => (comp1, comp2, "12"),
            "[2|1]" => (comp2, comp1, "21"),
            _ => continue,
        };
        let (Some(overlap), Some(y_offset)) =
            (hypothesis.overlap_width, hypothesis.vertical_offset)
        else {
            continue;
        };

        if let Some((left_strip, right_strip)) =
            aligned_overlap_strips(left, right, overlap, y_offset)
        {
            let overlay = seam_overlay_from_strips(&left_strip, &right_strip);
            save_debug_tiff(
                debug_dir.join(format!("stitch_overlap_{}_left.tiff", suffix)),
                &left_strip,
            );
            save_debug_tiff(
                debug_dir.join(format!("stitch_overlap_{}_right.tiff", suffix)),
                &right_strip,
            );
            save_debug_tiff(
                debug_dir.join(format!("stitch_overlap_{}_overlay.tiff", suffix)),
                &overlay,
            );
        }
    }
}

fn save_final_seam_overlay(
    left: &Array3<u16>,
    right: &Array3<u16>,
    overlap: usize,
    y_offset: i32,
    _order: StitchOrder,
    debug_dir: &Path,
) {
    if let Some((left_strip, right_strip)) = aligned_overlap_strips(left, right, overlap, y_offset)
    {
        let overlay = seam_overlay_from_strips(&left_strip, &right_strip);
        save_debug_tiff(debug_dir.join("stitch_seam_overlay.tiff"), &overlay);
    }
}

fn stitch_failure_report(
    config: &StitchConfig,
    opencv_available: bool,
    hypotheses: &[StitchHypothesis],
    best_index: usize,
    message: String,
    warnings: Vec<String>,
) -> PhaseReport {
    let mut report = PhaseReport {
        name: "stitch".to_string(),
        success: false,
        confidence: hypotheses
            .get(best_index)
            .map(|hyp| hyp.objective_score)
            .unwrap_or(0.0),
        duration_ms: 0,
        metrics: serde_json::json!({
            "decision": "rejected",
            "requested_transform": config.transform_mode.as_str(),
            "opencv_requested": config.use_opencv,
            "opencv_available": opencv_available,
            "chosen_hypothesis": hypotheses.get(best_index).map(|hyp| hyp.ordering.clone()),
            "transform_model_used": hypotheses.get(best_index).map(|hyp| hyp.transform_model.clone()),
            "rejection_reason": message.clone(),
            "acceptance_thresholds": acceptance_thresholds_json(config),
            "hypotheses": hypotheses,
            "preserved_full_valid_union": false,
        }),
        warnings,
        errors: vec![message],
    };
    if !opencv_available && config.use_opencv {
        report.warnings.push(
            "OpenCV was requested at runtime, but this binary was built without the `use-opencv` feature."
                .to_string(),
        );
    }
    report
}

fn apply_target_crop(img: &Array3<u16>, config: &StitchConfig) -> (Array3<u16>, Option<CropInfo>) {
    let (h, w, _) = img.dim();
    let target_w = config.target_width.unwrap_or(w).min(w);
    let target_h = config.target_height.unwrap_or(h).min(h);
    if target_w == w && target_h == h {
        return (img.clone(), None);
    }

    let y_start = choose_vertical_crop(img, target_h);
    let x_start = choose_horizontal_crop(img, target_w);
    let cropped = img
        .slice(s![
            y_start..(y_start + target_h),
            x_start..(x_start + target_w),
            ..
        ])
        .to_owned();

    (
        cropped,
        Some(CropInfo {
            x_start,
            y_start,
            width: target_w,
            height: target_h,
        }),
    )
}

fn choose_vertical_crop(img: &Array3<u16>, target_h: usize) -> usize {
    let (h, w, _) = img.dim();
    if target_h >= h {
        return 0;
    }

    let mut top = 0usize;
    while top < h {
        let row_nonzero =
            (0..w).any(|x| img[[top, x, 0]] != 0 || img[[top, x, 1]] != 0 || img[[top, x, 2]] != 0);
        if row_nonzero {
            break;
        }
        top += 1;
    }

    let mut bottom = h.saturating_sub(1);
    while bottom > top {
        let row_nonzero = (0..w).any(|x| {
            img[[bottom, x, 0]] != 0 || img[[bottom, x, 1]] != 0 || img[[bottom, x, 2]] != 0
        });
        if row_nonzero {
            break;
        }
        bottom = bottom.saturating_sub(1);
    }

    let center = ((top + bottom) / 2).min(h.saturating_sub(1));
    center.saturating_sub(target_h / 2).min(h - target_h)
}

fn choose_horizontal_crop(img: &Array3<u16>, target_w: usize) -> usize {
    let (h, w, _) = img.dim();
    if target_w >= w {
        return 0;
    }

    let scan = base_detect::detect_vertical_base_regions(img);
    let mut base_mask = scan.column_mask;
    if base_mask.len() != w {
        base_mask = vec![0.0; w];
    }

    let mut fill = vec![0.0f64; w];
    let mut content_weight = vec![0.0f64; w];
    for x in 0..w {
        let mut filled = 0usize;
        let mut grad = 0.0f64;
        for y in 0..h {
            let nonzero = img[[y, x, 0]] != 0 || img[[y, x, 1]] != 0 || img[[y, x, 2]] != 0;
            if nonzero {
                filled += 1;
            }
            if x > 0 {
                let curr = img[[y, x, 0]] as f64 * 0.2126
                    + img[[y, x, 1]] as f64 * 0.7152
                    + img[[y, x, 2]] as f64 * 0.0722;
                let prev = img[[y, x - 1, 0]] as f64 * 0.2126
                    + img[[y, x - 1, 1]] as f64 * 0.7152
                    + img[[y, x - 1, 2]] as f64 * 0.0722;
                grad += (curr - prev).abs();
            }
        }
        fill[x] = filled as f64 / h.max(1) as f64;
        base_mask[x] *= fill[x];
        content_weight[x] = (1.0 - base_mask[x]).max(0.0) * (grad / h.max(1) as f64 + 1.0);
    }

    let total_weight: f64 = content_weight.iter().sum();
    let content_center = if total_weight > 1e-6 {
        content_weight
            .iter()
            .enumerate()
            .map(|(x, wgt)| x as f64 * *wgt)
            .sum::<f64>()
            / total_weight
    } else {
        w as f64 / 2.0
    };

    let base_prefix = prefix_sum(&base_mask);
    let fill_prefix = prefix_sum(&fill);
    let edge_band = (target_w / 18).max(16).min(target_w / 2);

    let mut best_start = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for start in 0..=(w - target_w) {
        let end = start + target_w;
        let left_base = range_average(&base_prefix, start, (start + edge_band).min(end));
        let right_base = range_average(&base_prefix, end.saturating_sub(edge_band), end);
        let fill_score = range_average(&fill_prefix, start, end);
        let center_penalty = (((start + target_w / 2) as f64 - content_center).abs()
            / target_w.max(1) as f64)
            .min(1.0);
        let score =
            left_base * 0.45 + right_base * 0.45 + fill_score * 0.20 - center_penalty * 0.25;
        if score > best_score {
            best_score = score;
            best_start = start;
        }
    }

    best_start
}

fn prefix_sum(values: &[f64]) -> Vec<f64> {
    let mut prefix = Vec::with_capacity(values.len() + 1);
    prefix.push(0.0);
    let mut sum = 0.0f64;
    for value in values {
        sum += *value;
        prefix.push(sum);
    }
    prefix
}

fn range_average(prefix: &[f64], start: usize, end: usize) -> f64 {
    if end <= start {
        return 0.0;
    }
    (prefix[end] - prefix[start]) / (end - start) as f64
}

fn stitch_sample_max(config: &StitchConfig) -> f64 {
    match config.input_bit_depth {
        14 => MAX_14BIT,
        _ => MAX_16BIT,
    }
}

fn rgb_luma(rgb: &[f64; 3]) -> f64 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Return the same nearest-rank percentile as a full sort without ordering unrelated elements.
fn nearest_rank_percentile(mut values: Vec<f64>, percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    let (_, selected, _) = values.select_nth_unstable_by(index, |left, right| {
        left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
    });
    *selected
}

fn median_ratio(values: &[f64]) -> f64 {
    let mut bounded = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && (0.5..=2.0).contains(value))
        .collect::<Vec<_>>();
    if bounded.is_empty() {
        return 1.0;
    }
    bounded.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_from_sorted(&bounded, 0.5)
}

fn max_gain_spread(gain: &[f64; 3]) -> f64 {
    let min_gain = gain.iter().copied().fold(f64::INFINITY, f64::min);
    let max_gain = gain.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if min_gain <= 0.0 || !min_gain.is_finite() || !max_gain.is_finite() {
        f64::INFINITY
    } else {
        max_gain / min_gain
    }
}

fn gain_in_bounds(gain: &[f64; 3]) -> bool {
    gain.iter()
        .all(|value| (SEAM_EXPOSURE_MIN_GAIN..=SEAM_EXPOSURE_MAX_GAIN).contains(value))
}

fn clipping_delta(
    before_high: &[f64; 3],
    after_high: &[f64; 3],
    before_low: &[f64; 3],
    after_low: &[f64; 3],
) -> f64 {
    (0..3)
        .map(|c| {
            (after_high[c] - before_high[c]).max(0.0) + (after_low[c] - before_low[c]).max(0.0)
        })
        .sum()
}

fn seam_pixel_is_usable(left: &[f64; 3], right: &[f64; 3], max_value: f64) -> bool {
    let low_floor = (max_value * 0.015).max(4.0);
    let high_floor = max_value * 0.985;
    let left_luma = rgb_luma(left);
    let right_luma = rgb_luma(right);
    if left_luma <= low_floor || right_luma <= low_floor {
        return false;
    }
    for c in 0..3 {
        if left[c] <= 1.0 || right[c] <= 1.0 || left[c] >= high_floor || right[c] >= high_floor {
            return false;
        }
    }
    true
}

fn seam_sample_ratios(sample: &SeamSample) -> ([f64; 3], f64) {
    let rgb = std::array::from_fn(|c| {
        if sample.right[c] <= 1.0 {
            1.0
        } else {
            sample.left[c] / sample.right[c]
        }
    });
    let left_luma = rgb_luma(&sample.left);
    let right_luma = rgb_luma(&sample.right);
    let luma = if right_luma <= 1.0 {
        1.0
    } else {
        left_luma / right_luma
    };
    (rgb, luma)
}

fn seam_measurements(
    samples: &[SeamSample],
    gain: &[f64; 3],
    offset: &[f64; 3],
    max_value: f64,
) -> SeamMeasurements {
    seam_measurements_spatial(samples, gain, offset, &[0.0; 3], max_value)
}

fn spatial_gain_at_normalized_y(
    center_gain: &[f64; 3],
    log_slope_y: &[f64; 3],
    y_normalized: f64,
) -> [f64; 3] {
    spatial_gain_at_normalized_xy(center_gain, &[0.0; 3], log_slope_y, 0.0, y_normalized)
}

fn spatial_gain_at_normalized_xy(
    center_gain: &[f64; 3],
    log_slope_x: &[f64; 3],
    log_slope_y: &[f64; 3],
    x_normalized: f64,
    y_normalized: f64,
) -> [f64; 3] {
    spatial_gain_at_normalized_quadratic_xy(
        center_gain,
        log_slope_x,
        log_slope_y,
        &[0.0; 3],
        &[0.0; 3],
        &[0.0; 3],
        x_normalized,
        y_normalized,
    )
}

#[allow(clippy::too_many_arguments)]
fn spatial_gain_at_normalized_quadratic_xy(
    center_gain: &[f64; 3],
    log_slope_x: &[f64; 3],
    log_slope_y: &[f64; 3],
    log_quadratic_xx: &[f64; 3],
    log_quadratic_xy: &[f64; 3],
    log_quadratic_yy: &[f64; 3],
    x_normalized: f64,
    y_normalized: f64,
) -> [f64; 3] {
    let x = x_normalized.clamp(-1.0, 1.0);
    let y = y_normalized.clamp(-1.0, 1.0);
    std::array::from_fn(|channel| {
        center_gain[channel]
            * (log_slope_x[channel] * x
                + log_slope_y[channel] * y
                + log_quadratic_xx[channel] * x * x
                + log_quadratic_xy[channel] * x * y
                + log_quadratic_yy[channel] * y * y)
                .exp()
    })
}

fn spatial_photometric_at_normalized_xy(
    field: &SpatialPhotometricField,
    x_normalized: f64,
    y_normalized: f64,
) -> ([f64; 3], [f64; 3]) {
    let x = x_normalized.clamp(-1.0, 1.0);
    let y = y_normalized.clamp(-1.0, 1.0);
    let gain = spatial_gain_at_normalized_quadratic_xy(
        &field.center_gain,
        &field.log_gain_slope_x,
        &field.log_gain_slope_y,
        &field.log_gain_quadratic_xx,
        &field.log_gain_quadratic_xy,
        &field.log_gain_quadratic_yy,
        x,
        y,
    );
    let offset = std::array::from_fn(|channel| {
        field.center_offset[channel]
            + field.offset_slope_x[channel] * x
            + field.offset_slope_y[channel] * y
            + field.offset_quadratic_xx[channel] * x * x
            + field.offset_quadratic_xy[channel] * x * y
            + field.offset_quadratic_yy[channel] * y * y
    });
    (gain, offset)
}

fn spatial_field_has_horizontal_variation(field: &SpatialPhotometricField) -> bool {
    [
        &field.log_gain_slope_x,
        &field.log_gain_quadratic_xx,
        &field.log_gain_quadratic_xy,
        &field.offset_slope_x,
        &field.offset_quadratic_xx,
        &field.offset_quadratic_xy,
    ]
    .into_iter()
    .any(|coefficients| coefficients.iter().any(|coefficient| *coefficient != 0.0))
}

fn seam_measurements_spatial(
    samples: &[SeamSample],
    center_gain: &[f64; 3],
    center_offset: &[f64; 3],
    log_slope_y: &[f64; 3],
    max_value: f64,
) -> SeamMeasurements {
    seam_measurements_spatial_affine(
        samples,
        center_gain,
        center_offset,
        log_slope_y,
        &[0.0; 3],
        max_value,
    )
}

fn seam_measurements_spatial_affine(
    samples: &[SeamSample],
    center_gain: &[f64; 3],
    center_offset: &[f64; 3],
    log_slope_y: &[f64; 3],
    offset_slope_y: &[f64; 3],
    max_value: f64,
) -> SeamMeasurements {
    seam_measurements_spatial_affine_2d(
        samples,
        center_gain,
        center_offset,
        &[0.0; 3],
        log_slope_y,
        &[0.0; 3],
        offset_slope_y,
        max_value,
    )
}

#[allow(clippy::too_many_arguments)]
fn seam_measurements_spatial_affine_2d(
    samples: &[SeamSample],
    center_gain: &[f64; 3],
    center_offset: &[f64; 3],
    log_slope_x: &[f64; 3],
    log_slope_y: &[f64; 3],
    offset_slope_x: &[f64; 3],
    offset_slope_y: &[f64; 3],
    max_value: f64,
) -> SeamMeasurements {
    let field = SpatialPhotometricField {
        center_gain: *center_gain,
        center_offset: *center_offset,
        log_gain_slope_x: *log_slope_x,
        log_gain_slope_y: *log_slope_y,
        log_gain_quadratic_xx: [0.0; 3],
        log_gain_quadratic_xy: [0.0; 3],
        log_gain_quadratic_yy: [0.0; 3],
        offset_slope_x: *offset_slope_x,
        offset_slope_y: *offset_slope_y,
        offset_quadratic_xx: [0.0; 3],
        offset_quadratic_xy: [0.0; 3],
        offset_quadratic_yy: [0.0; 3],
    };
    seam_measurements_photometric_field(samples, &field, max_value)
}

fn seam_measurements_photometric_field(
    samples: &[SeamSample],
    field: &SpatialPhotometricField,
    max_value: f64,
) -> SeamMeasurements {
    if samples.is_empty() {
        return SeamMeasurements {
            delta_luma: 0.0,
            delta_rgb: [0.0; 3],
            seam_score: 0.0,
        };
    }

    let mut luma_delta = Vec::<f64>::with_capacity(samples.len());
    let mut luma_abs_delta = Vec::<f64>::with_capacity(samples.len());
    let mut rgb_delta = [Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()];
    let mut rgb_abs_delta = [Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()];
    let horizontal_field_is_constant = !spatial_field_has_horizontal_variation(field);
    let mut cached_y = f64::NAN;
    let mut cached_gain = [1.0; 3];
    let mut cached_offset = [0.0; 3];

    for sample in samples {
        let (gain, offset) = if horizontal_field_is_constant {
            if sample.y_normalized != cached_y {
                cached_y = sample.y_normalized;
                (cached_gain, cached_offset) =
                    spatial_photometric_at_normalized_xy(field, 0.0, sample.y_normalized);
            }
            (cached_gain, cached_offset)
        } else {
            spatial_photometric_at_normalized_xy(field, sample.x_normalized, sample.y_normalized)
        };
        let adjusted =
            std::array::from_fn(|c| (sample.right[c] * gain[c] + offset[c]).clamp(0.0, max_value));
        let left_luma = rgb_luma(&sample.left).max(1.0);
        let adjusted_luma = rgb_luma(&adjusted).max(1.0);
        let luma_log = (adjusted_luma / left_luma).ln();
        luma_delta.push(luma_log);
        luma_abs_delta.push(luma_log.abs());
        for c in 0..3 {
            let channel_log = (adjusted[c].max(1.0) / sample.left[c].max(1.0)).ln();
            rgb_delta[c].push(channel_log);
            rgb_abs_delta[c].push(channel_log.abs());
        }
    }

    let delta_luma = nearest_rank_percentile(luma_delta, 0.5);
    let luma_score = nearest_rank_percentile(luma_abs_delta, 0.5);
    let mut delta_rgb = [0.0; 3];
    let mut rgb_score = 0.0;
    for c in 0..3 {
        delta_rgb[c] = nearest_rank_percentile(std::mem::take(&mut rgb_delta[c]), 0.5);
        rgb_score += nearest_rank_percentile(std::mem::take(&mut rgb_abs_delta[c]), 0.5);
    }
    rgb_score /= 3.0;

    SeamMeasurements {
        delta_luma,
        delta_rgb,
        seam_score: (0.55 * luma_score + 0.45 * rgb_score).max(0.0),
    }
}

fn strip_clipping_ratios(
    strip: &Array3<u16>,
    gain: &[f64; 3],
    offset: &[f64; 3],
    max_value: f64,
) -> ([f64; 3], [f64; 3]) {
    strip_clipping_ratios_spatial(strip, gain, offset, &[0.0; 3], max_value)
}

fn strip_clipping_ratios_spatial(
    strip: &Array3<u16>,
    center_gain: &[f64; 3],
    offset: &[f64; 3],
    log_slope_y: &[f64; 3],
    max_value: f64,
) -> ([f64; 3], [f64; 3]) {
    strip_clipping_ratios_spatial_affine(
        strip,
        center_gain,
        offset,
        log_slope_y,
        &[0.0; 3],
        max_value,
    )
}

fn strip_clipping_ratios_spatial_affine(
    strip: &Array3<u16>,
    center_gain: &[f64; 3],
    center_offset: &[f64; 3],
    log_slope_y: &[f64; 3],
    offset_slope_y: &[f64; 3],
    max_value: f64,
) -> ([f64; 3], [f64; 3]) {
    strip_clipping_ratios_spatial_affine_2d(
        strip,
        center_gain,
        center_offset,
        &[0.0; 3],
        log_slope_y,
        &[0.0; 3],
        offset_slope_y,
        max_value,
    )
}

#[allow(clippy::too_many_arguments)]
fn strip_clipping_ratios_spatial_affine_2d(
    strip: &Array3<u16>,
    center_gain: &[f64; 3],
    center_offset: &[f64; 3],
    log_slope_x: &[f64; 3],
    log_slope_y: &[f64; 3],
    offset_slope_x: &[f64; 3],
    offset_slope_y: &[f64; 3],
    max_value: f64,
) -> ([f64; 3], [f64; 3]) {
    let field = SpatialPhotometricField {
        center_gain: *center_gain,
        center_offset: *center_offset,
        log_gain_slope_x: *log_slope_x,
        log_gain_slope_y: *log_slope_y,
        log_gain_quadratic_xx: [0.0; 3],
        log_gain_quadratic_xy: [0.0; 3],
        log_gain_quadratic_yy: [0.0; 3],
        offset_slope_x: *offset_slope_x,
        offset_slope_y: *offset_slope_y,
        offset_quadratic_xx: [0.0; 3],
        offset_quadratic_xy: [0.0; 3],
        offset_quadratic_yy: [0.0; 3],
    };
    strip_clipping_ratios_photometric_field(strip, &field, max_value)
}

fn strip_clipping_ratios_photometric_field(
    strip: &Array3<u16>,
    field: &SpatialPhotometricField,
    max_value: f64,
) -> ([f64; 3], [f64; 3]) {
    let (h, w, _) = strip.dim();
    let n = (h * w).max(1) as f64;
    let mut clipped_high = [0usize; 3];
    let mut clipped_low = [0usize; 3];
    let horizontal_field_is_constant = !spatial_field_has_horizontal_variation(field);
    for y in 0..h {
        let y_normalized = if h <= 1 {
            0.0
        } else {
            (2.0 * y as f64 / (h - 1) as f64 - 1.0).clamp(-1.0, 1.0)
        };
        let row_field = horizontal_field_is_constant
            .then(|| spatial_photometric_at_normalized_xy(field, 0.0, y_normalized));
        for x in 0..w {
            let (gain, offset) = match row_field {
                Some(field_value) => field_value,
                None => {
                    let x_normalized = if w <= 1 {
                        0.0
                    } else {
                        (2.0 * x as f64 / (w - 1) as f64 - 1.0).clamp(-1.0, 1.0)
                    };
                    spatial_photometric_at_normalized_xy(field, x_normalized, y_normalized)
                }
            };
            for c in 0..3 {
                let adjusted = strip[[y, x, c]] as f64 * gain[c] + offset[c];
                if adjusted >= max_value {
                    clipped_high[c] += 1;
                }
                if adjusted <= 0.0 {
                    clipped_low[c] += 1;
                }
            }
        }
    }
    (
        std::array::from_fn(|c| clipped_high[c] as f64 / n),
        std::array::from_fn(|c| clipped_low[c] as f64 / n),
    )
}

fn window_texture_ok(samples: &[SeamSample], max_value: f64) -> bool {
    if samples.len() < 16 {
        return false;
    }
    let left_luma: Vec<f64> = samples
        .iter()
        .map(|sample| rgb_luma(&sample.left))
        .collect();
    let right_luma: Vec<f64> = samples
        .iter()
        .map(|sample| rgb_luma(&sample.right))
        .collect();
    let left_range =
        nearest_rank_percentile(left_luma.clone(), 0.90) - nearest_rank_percentile(left_luma, 0.10);
    let right_range = nearest_rank_percentile(right_luma.clone(), 0.90)
        - nearest_rank_percentile(right_luma, 0.10);
    left_range.max(right_range) >= max_value * 0.003
}

fn least_squares_affine(pairs: &[(f64, f64)]) -> (f64, f64) {
    if pairs.len() < 2 {
        return (1.0, 0.0);
    }
    let count = pairs.len() as f64;
    let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / count;
    let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / count;
    let variance_x = pairs
        .iter()
        .map(|(x, _)| {
            let centered = x - mean_x;
            centered * centered
        })
        .sum::<f64>();
    if variance_x <= f64::EPSILON {
        return (1.0, mean_y - mean_x);
    }
    let covariance = pairs
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let gain = covariance / variance_x;
    (gain, mean_y - gain * mean_x)
}

fn robust_affine_pairs(pairs: &[(f64, f64)], max_value: f64) -> (f64, f64) {
    if pairs.len() < 16 {
        return (1.0, 0.0);
    }
    let stride = pairs.len().div_ceil(65_536).max(1);
    let mut working = pairs.iter().step_by(stride).copied().collect::<Vec<_>>();
    if working.len() < 16 {
        return (1.0, 0.0);
    }

    let mut estimate = least_squares_affine(&working);
    for _ in 0..3 {
        if !estimate.0.is_finite() || !estimate.1.is_finite() {
            return (1.0, 0.0);
        }
        let residuals = working
            .iter()
            .map(|(x, y)| y - (estimate.0 * x + estimate.1))
            .collect::<Vec<_>>();
        let center = nearest_rank_percentile(residuals.clone(), 0.5);
        let mad = nearest_rank_percentile(
            residuals
                .iter()
                .map(|residual| (residual - center).abs())
                .collect(),
            0.5,
        );
        let threshold = (3.5 * 1.4826 * mad).max(max_value * 0.0015);
        let filtered = working
            .iter()
            .copied()
            .filter(|(x, y)| (y - (estimate.0 * x + estimate.1) - center).abs() <= threshold)
            .collect::<Vec<_>>();
        if filtered.len() < 16 || filtered.len() * 2 < working.len() {
            break;
        }
        working = filtered;
        estimate = least_squares_affine(&working);
    }
    estimate
}

fn robust_affine_for_channel(
    samples: &[SeamSample],
    channel: Option<usize>,
    max_value: f64,
) -> (f64, f64) {
    let pairs = samples
        .iter()
        .map(|sample| match channel {
            Some(channel) => (sample.right[channel], sample.left[channel]),
            None => (rgb_luma(&sample.right), rgb_luma(&sample.left)),
        })
        .collect::<Vec<_>>();
    robust_affine_pairs(&pairs, max_value)
}

fn affine_window_has_signal(samples: &[SeamSample], max_value: f64) -> bool {
    if samples.len() < 32 {
        return false;
    }
    (0..3).all(|channel| {
        let values = samples
            .iter()
            .map(|sample| sample.right[channel])
            .collect::<Vec<_>>();
        nearest_rank_percentile(values.clone(), 0.90) - nearest_rank_percentile(values, 0.10)
            >= max_value * 0.015
    })
}

fn estimate_window_photometric(samples: &[SeamSample], max_value: f64) -> SeamWindowEstimate {
    let mut ratios_rgb = [Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()];
    let mut ratios_luma = Vec::<f64>::new();
    for sample in samples {
        let (rgb, luma) = seam_sample_ratios(sample);
        for c in 0..3 {
            ratios_rgb[c].push(rgb[c]);
        }
        ratios_luma.push(luma);
    }

    let affine_reliable = affine_window_has_signal(samples, max_value);
    let affine_rgb: [(f64, f64); 3] = std::array::from_fn(|channel| {
        if affine_reliable {
            robust_affine_for_channel(samples, Some(channel), max_value)
        } else {
            (1.0, 0.0)
        }
    });
    SeamWindowEstimate {
        gain_rgb: std::array::from_fn(|c| median_ratio(&ratios_rgb[c])),
        gain_luma: median_ratio(&ratios_luma),
        affine_gain_rgb: std::array::from_fn(|c| affine_rgb[c].0),
        offset_rgb: std::array::from_fn(|c| affine_rgb[c].1),
        affine_reliable,
    }
}

fn collect_seam_windows(
    strip_l: &Array3<u16>,
    strip_r: &Array3<u16>,
    max_value: f64,
) -> (Vec<SeamWindow>, usize) {
    let (h_l, w_l, _) = strip_l.dim();
    let (h_r, w_r, _) = strip_r.dim();
    let h = h_l.min(h_r);
    let w = w_l.min(w_r);
    if h == 0 || w == 0 {
        return (Vec::new(), 0);
    }

    let grid_rows = (h / 96).clamp(2, 8).min(h);
    let grid_cols = (w / 32).clamp(2, 8).min(w);
    let mut windows = Vec::<SeamWindow>::new();

    for row in 0..grid_rows {
        let y0 = row * h / grid_rows;
        let y1 = ((row + 1) * h / grid_rows).max(y0 + 1).min(h);
        for col in 0..grid_cols {
            let x0 = col * w / grid_cols;
            let x1 = ((col + 1) * w / grid_cols).max(x0 + 1).min(w);
            let window_x_normalized = if w <= 1 {
                0.0
            } else {
                (2.0 * ((x0 + x1 - 1) as f64 * 0.5) / (w - 1) as f64 - 1.0).clamp(-1.0, 1.0)
            };
            let window_y_normalized = if h <= 1 {
                0.0
            } else {
                (2.0 * ((y0 + y1 - 1) as f64 * 0.5) / (h - 1) as f64 - 1.0).clamp(-1.0, 1.0)
            };
            let mut window_samples = Vec::<SeamSample>::new();
            for y in y0..y1 {
                let y_normalized = if h <= 1 {
                    0.0
                } else {
                    (2.0 * y as f64 / (h - 1) as f64 - 1.0).clamp(-1.0, 1.0)
                };
                for x in x0..x1 {
                    let x_normalized = if w <= 1 {
                        0.0
                    } else {
                        (2.0 * x as f64 / (w - 1) as f64 - 1.0).clamp(-1.0, 1.0)
                    };
                    let left = [
                        strip_l[[y, x, 0]] as f64,
                        strip_l[[y, x, 1]] as f64,
                        strip_l[[y, x, 2]] as f64,
                    ];
                    let right = [
                        strip_r[[y, x, 0]] as f64,
                        strip_r[[y, x, 1]] as f64,
                        strip_r[[y, x, 2]] as f64,
                    ];
                    if seam_pixel_is_usable(&left, &right, max_value) {
                        window_samples.push(SeamSample {
                            left,
                            right,
                            x_normalized,
                            y_normalized,
                        });
                    }
                }
            }

            if window_texture_ok(&window_samples, max_value) {
                let estimate = estimate_window_photometric(&window_samples, max_value);
                windows.push(SeamWindow {
                    row,
                    col,
                    x_normalized: window_x_normalized,
                    y_normalized: window_y_normalized,
                    samples: window_samples,
                    estimate,
                });
            }
        }
    }

    let total_pixels = h * w;
    (windows, total_pixels)
}

fn seam_window_split(windows: &[SeamWindow], affine_only: bool) -> (Vec<usize>, Vec<usize>) {
    let eligible = windows
        .iter()
        .enumerate()
        .filter_map(|(index, window)| {
            (!affine_only || window.estimate.affine_reliable).then_some(index)
        })
        .collect::<Vec<_>>();
    let mut training = eligible
        .iter()
        .copied()
        .filter(|index| (windows[*index].row + windows[*index].col).is_multiple_of(2))
        .collect::<Vec<_>>();
    let mut held_out = eligible
        .iter()
        .copied()
        .filter(|index| (windows[*index].row + windows[*index].col) % 2 == 1)
        .collect::<Vec<_>>();
    if training.is_empty() || held_out.is_empty() {
        training.clear();
        held_out.clear();
        for (position, index) in eligible.into_iter().enumerate() {
            if position.is_multiple_of(2) {
                training.push(index);
            } else {
                held_out.push(index);
            }
        }
    }
    (training, held_out)
}

fn seam_samples_for_windows(windows: &[SeamWindow], indices: &[usize]) -> Vec<SeamSample> {
    indices
        .iter()
        .flat_map(|index| windows[*index].samples.iter().copied())
        .collect()
}

fn seam_estimates_for_windows(
    windows: &[SeamWindow],
    indices: &[usize],
) -> Vec<SeamWindowEstimate> {
    indices
        .iter()
        .map(|index| windows[*index].estimate)
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct SpatialGainFit {
    center_gain_rgb: [f64; 3],
    center_gain_luma: f64,
    log_slope_y_rgb: [f64; 3],
    log_slope_y_luma: f64,
}

#[derive(Debug, Clone, Copy)]
struct SpatialAffineFit {
    center_gain_rgb: [f64; 3],
    center_gain_luma: f64,
    log_gain_slope_y_rgb: [f64; 3],
    log_gain_slope_y_luma: f64,
    center_offset_rgb: [f64; 3],
    center_offset_luma: f64,
    offset_slope_y_rgb: [f64; 3],
    offset_slope_y_luma: f64,
}

#[derive(Debug, Clone, Copy)]
struct SpatialGain2dFit {
    center_gain_rgb: [f64; 3],
    center_gain_luma: f64,
    log_slope_x_rgb: [f64; 3],
    log_slope_x_luma: f64,
    log_slope_y_rgb: [f64; 3],
    log_slope_y_luma: f64,
}

#[derive(Debug, Clone, Copy)]
struct SpatialAffine2dFit {
    center_gain_rgb: [f64; 3],
    center_gain_luma: f64,
    log_gain_slope_x_rgb: [f64; 3],
    log_gain_slope_x_luma: f64,
    log_gain_slope_y_rgb: [f64; 3],
    log_gain_slope_y_luma: f64,
    center_offset_rgb: [f64; 3],
    center_offset_luma: f64,
    offset_slope_x_rgb: [f64; 3],
    offset_slope_x_luma: f64,
    offset_slope_y_rgb: [f64; 3],
    offset_slope_y_luma: f64,
}

/// A bounded quadratic surface in the fixed basis `[1, x, y, x^2, x*y, y^2]`.
/// Gain coefficients are represented in log space; offsets remain in sample units.
#[derive(Debug, Clone, Copy)]
struct SpatialQuadraticGain2dFit {
    log_gain_rgb: [[f64; 6]; 3],
    log_gain_luma: [f64; 6],
    design_condition_number: f64,
}

#[derive(Debug, Clone, Copy)]
struct SpatialQuadraticAffine2dFit {
    log_gain_rgb: [[f64; 6]; 3],
    log_gain_luma: [f64; 6],
    offset_rgb: [[f64; 6]; 3],
    offset_luma: [f64; 6],
    design_condition_number: f64,
}

fn robust_linear_line(points: &[(f64, f64)]) -> (f64, f64) {
    let points = points
        .iter()
        .copied()
        .filter(|(position, value)| position.is_finite() && value.is_finite())
        .collect::<Vec<_>>();
    if points.is_empty() {
        return (0.0, 0.0);
    }

    // A Theil-Sen line gives every reliable overlap window equal influence and resists a
    // textured subject region masquerading as scanner shading.
    let mut slopes = Vec::<f64>::new();
    for first in 0..points.len() {
        for second in (first + 1)..points.len() {
            let delta_position = points[second].0 - points[first].0;
            if delta_position.abs() > 1e-6 {
                slopes.push((points[second].1 - points[first].1) / delta_position);
            }
        }
    }
    let slope = if slopes.is_empty() {
        0.0
    } else {
        nearest_rank_percentile(slopes, 0.5)
    };
    let intercepts = points
        .iter()
        .map(|(position, value)| value - slope * position)
        .collect::<Vec<_>>();
    (nearest_rank_percentile(intercepts, 0.5), slope)
}

fn robust_log_gain_line(points: &[(f64, f64)]) -> (f64, f64) {
    let points = points
        .iter()
        .copied()
        .filter(|(position, gain)| {
            position.is_finite() && gain.is_finite() && (0.5..=2.0).contains(gain)
        })
        .map(|(position, gain)| (position, gain.ln()))
        .collect::<Vec<_>>();
    robust_linear_line(&points)
}

fn fit_spatial_gain(windows: &[SeamWindow], indices: &[usize]) -> SpatialGainFit {
    let rgb_lines: [(f64, f64); 3] = std::array::from_fn(|channel| {
        robust_log_gain_line(
            &indices
                .iter()
                .map(|index| {
                    (
                        windows[*index].y_normalized,
                        windows[*index].estimate.gain_rgb[channel],
                    )
                })
                .collect::<Vec<_>>(),
        )
    });
    let luma_line = robust_log_gain_line(
        &indices
            .iter()
            .map(|index| {
                (
                    windows[*index].y_normalized,
                    windows[*index].estimate.gain_luma,
                )
            })
            .collect::<Vec<_>>(),
    );
    SpatialGainFit {
        center_gain_rgb: std::array::from_fn(|channel| rgb_lines[channel].0.exp()),
        center_gain_luma: luma_line.0.exp(),
        log_slope_y_rgb: std::array::from_fn(|channel| rgb_lines[channel].1),
        log_slope_y_luma: luma_line.1,
    }
}

fn robust_plane(points: &[(f64, f64, f64)]) -> [f64; 3] {
    let points = points
        .iter()
        .copied()
        .filter(|(x, y, value)| x.is_finite() && y.is_finite() && value.is_finite())
        .collect::<Vec<_>>();
    if points.len() < 3 {
        return [
            nearest_rank_percentile(points.iter().map(|point| point.2).collect(), 0.5),
            0.0,
            0.0,
        ];
    }
    let mut parameters = Vector3::new(
        nearest_rank_percentile(points.iter().map(|point| point.2).collect(), 0.5),
        0.0,
        0.0,
    );
    for _ in 0..10 {
        let absolute_residuals = points
            .iter()
            .map(|(x, y, value)| (value - parameters.dot(&Vector3::new(1.0, *x, *y))).abs())
            .collect::<Vec<_>>();
        let robust_scale = nearest_rank_percentile(absolute_residuals, 0.5)
            .mul_add(2.5, 0.0)
            .max(5e-5);
        let mut normal = Matrix3::<f64>::identity() * 1e-8;
        let mut right_hand_side = Vector3::<f64>::zeros();
        for (x, y, value) in &points {
            let basis = Vector3::new(1.0, *x, *y);
            let residual = value - parameters.dot(&basis);
            let weight = if residual.abs() <= robust_scale {
                1.0
            } else {
                robust_scale / residual.abs().max(1e-12)
            };
            normal += basis * basis.transpose() * weight;
            right_hand_side += basis * (*value * weight);
        }
        let Some(next) = normal.lu().solve(&right_hand_side) else {
            break;
        };
        if (next - parameters)
            .iter()
            .all(|difference| difference.abs() < 1e-7)
        {
            parameters = next;
            break;
        }
        parameters = next;
    }
    [
        parameters[0],
        parameters[1].clamp(-0.35, 0.35),
        parameters[2].clamp(-0.35, 0.35),
    ]
}

fn robust_log_gain_plane(points: &[(f64, f64, f64)]) -> [f64; 3] {
    let log_points = points
        .iter()
        .copied()
        .filter(|(_, _, gain)| gain.is_finite() && (0.5..=2.0).contains(gain))
        .map(|(x, y, gain)| (x, y, gain.ln()))
        .collect::<Vec<_>>();
    robust_plane(&log_points)
}

fn quadratic_surface_basis(x: f64, y: f64) -> SVector<f64, 6> {
    SVector::<f64, 6>::from_row_slice(&[1.0, x, y, x * x, x * y, y * y])
}

fn quadratic_design_condition_number(points: &[(f64, f64, f64)]) -> f64 {
    if points.len() < 6 {
        return 1.0e12;
    }
    let mut normal = SMatrix::<f64, 6, 6>::zeros();
    for (x, y, _) in points {
        let basis = quadratic_surface_basis(*x, *y);
        normal += basis * basis.transpose();
    }
    let singular_values = normal.svd(false, false).singular_values;
    let maximum = singular_values.iter().copied().fold(0.0f64, f64::max);
    let minimum = singular_values
        .iter()
        .copied()
        .filter(|value| *value > 1e-12)
        .fold(f64::INFINITY, f64::min);
    if !maximum.is_finite() || !minimum.is_finite() || minimum <= 0.0 {
        1.0e12
    } else {
        (maximum / minimum).sqrt()
    }
}

fn robust_quadratic_surface(points: &[(f64, f64, f64)]) -> ([f64; 6], f64) {
    let points = points
        .iter()
        .copied()
        .filter(|(x, y, value)| x.is_finite() && y.is_finite() && value.is_finite())
        .collect::<Vec<_>>();
    let condition_number = quadratic_design_condition_number(&points);
    if points.len() < 6 {
        let plane = robust_plane(&points);
        return (
            [plane[0], plane[1], plane[2], 0.0, 0.0, 0.0],
            condition_number,
        );
    }

    let plane = robust_plane(&points);
    let mut parameters =
        SVector::<f64, 6>::from_row_slice(&[plane[0], plane[1], plane[2], 0.0, 0.0, 0.0]);
    for _ in 0..12 {
        let absolute_residuals = points
            .iter()
            .map(|(x, y, value)| (value - parameters.dot(&quadratic_surface_basis(*x, *y))).abs())
            .collect::<Vec<_>>();
        let robust_scale = nearest_rank_percentile(absolute_residuals, 0.5)
            .mul_add(2.5, 0.0)
            .max(5e-5);
        let mut normal = SMatrix::<f64, 6, 6>::identity() * 1e-8;
        for index in 3..6 {
            normal[(index, index)] += SEAM_EXPOSURE_QUADRATIC_REGULARIZATION;
        }
        let mut right_hand_side = SVector::<f64, 6>::zeros();
        for (x, y, value) in &points {
            let basis = quadratic_surface_basis(*x, *y);
            let residual = value - parameters.dot(&basis);
            let weight = if residual.abs() <= robust_scale {
                1.0
            } else {
                robust_scale / residual.abs().max(1e-12)
            };
            normal += basis * basis.transpose() * weight;
            right_hand_side += basis * (*value * weight);
        }
        let Some(next) = normal.lu().solve(&right_hand_side) else {
            break;
        };
        let difference = next - parameters;
        parameters = next;
        parameters[1] = parameters[1].clamp(-0.35, 0.35);
        parameters[2] = parameters[2].clamp(-0.35, 0.35);
        for coefficient in parameters.iter_mut().skip(3) {
            *coefficient = coefficient.clamp(-0.20, 0.20);
        }
        if difference.iter().all(|value| value.abs() < 1e-7) {
            break;
        }
    }
    (parameters.into(), condition_number)
}

fn robust_log_gain_quadratic_surface(points: &[(f64, f64, f64)]) -> ([f64; 6], f64) {
    let log_points = points
        .iter()
        .copied()
        .filter(|(_, _, gain)| gain.is_finite() && (0.5..=2.0).contains(gain))
        .map(|(x, y, gain)| (x, y, gain.ln()))
        .collect::<Vec<_>>();
    robust_quadratic_surface(&log_points)
}

fn fit_spatial_quadratic_gain_2d(
    windows: &[SeamWindow],
    indices: &[usize],
) -> SpatialQuadraticGain2dFit {
    let rgb: [([f64; 6], f64); 3] = std::array::from_fn(|channel| {
        robust_log_gain_quadratic_surface(
            &indices
                .iter()
                .map(|index| {
                    let window = &windows[*index];
                    (
                        window.x_normalized,
                        window.y_normalized,
                        window.estimate.gain_rgb[channel],
                    )
                })
                .collect::<Vec<_>>(),
        )
    });
    let luma = robust_log_gain_quadratic_surface(
        &indices
            .iter()
            .map(|index| {
                let window = &windows[*index];
                (
                    window.x_normalized,
                    window.y_normalized,
                    window.estimate.gain_luma,
                )
            })
            .collect::<Vec<_>>(),
    );
    SpatialQuadraticGain2dFit {
        log_gain_rgb: std::array::from_fn(|channel| rgb[channel].0),
        log_gain_luma: luma.0,
        design_condition_number: rgb
            .iter()
            .map(|fit| fit.1)
            .chain(std::iter::once(luma.1))
            .fold(0.0f64, f64::max),
    }
}

fn spatial_quadratic_gain_to_field(fit: &SpatialQuadraticGain2dFit) -> SpatialPhotometricField {
    SpatialPhotometricField {
        center_gain: std::array::from_fn(|channel| fit.log_gain_rgb[channel][0].exp()),
        center_offset: [0.0; 3],
        log_gain_slope_x: std::array::from_fn(|channel| fit.log_gain_rgb[channel][1]),
        log_gain_slope_y: std::array::from_fn(|channel| fit.log_gain_rgb[channel][2]),
        log_gain_quadratic_xx: std::array::from_fn(|channel| fit.log_gain_rgb[channel][3]),
        log_gain_quadratic_xy: std::array::from_fn(|channel| fit.log_gain_rgb[channel][4]),
        log_gain_quadratic_yy: std::array::from_fn(|channel| fit.log_gain_rgb[channel][5]),
        offset_slope_x: [0.0; 3],
        offset_slope_y: [0.0; 3],
        offset_quadratic_xx: [0.0; 3],
        offset_quadratic_xy: [0.0; 3],
        offset_quadratic_yy: [0.0; 3],
    }
}

fn fit_spatial_gain_2d(windows: &[SeamWindow], indices: &[usize]) -> SpatialGain2dFit {
    let rgb_planes: [[f64; 3]; 3] = std::array::from_fn(|channel| {
        robust_log_gain_plane(
            &indices
                .iter()
                .map(|index| {
                    let window = &windows[*index];
                    (
                        window.x_normalized,
                        window.y_normalized,
                        window.estimate.gain_rgb[channel],
                    )
                })
                .collect::<Vec<_>>(),
        )
    });
    let luma_plane = robust_log_gain_plane(
        &indices
            .iter()
            .map(|index| {
                let window = &windows[*index];
                (
                    window.x_normalized,
                    window.y_normalized,
                    window.estimate.gain_luma,
                )
            })
            .collect::<Vec<_>>(),
    );
    SpatialGain2dFit {
        center_gain_rgb: std::array::from_fn(|channel| rgb_planes[channel][0].exp()),
        center_gain_luma: luma_plane[0].exp(),
        log_slope_x_rgb: std::array::from_fn(|channel| rgb_planes[channel][1]),
        log_slope_x_luma: luma_plane[1],
        log_slope_y_rgb: std::array::from_fn(|channel| rgb_planes[channel][2]),
        log_slope_y_luma: luma_plane[2],
    }
}

fn spatial_gain_2d_at(fit: &SpatialGain2dFit, x_normalized: f64, y_normalized: f64) -> [f64; 3] {
    spatial_gain_at_normalized_xy(
        &fit.center_gain_rgb,
        &fit.log_slope_x_rgb,
        &fit.log_slope_y_rgb,
        x_normalized,
        y_normalized,
    )
}

fn spatial_gain_2d_corners(fit: &SpatialGain2dFit) -> [[f64; 3]; 4] {
    SPATIAL_FIELD_CORNERS.map(|[x, y]| spatial_gain_2d_at(fit, x, y))
}

fn robust_spatial_affine_channel(
    samples: &[SeamSample],
    channel: Option<usize>,
    initial_center_gain: f64,
    initial_log_gain_slope_y: f64,
    max_value: f64,
) -> [f64; 4] {
    if samples.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let stride = samples.len().div_ceil(20_000).max(1);
    let sample_values = |sample: &SeamSample| {
        let left = channel
            .map(|index| sample.left[index])
            .unwrap_or_else(|| rgb_luma(&sample.left))
            / max_value;
        let right = channel
            .map(|index| sample.right[index])
            .unwrap_or_else(|| rgb_luma(&sample.right))
            / max_value;
        (left, right, sample.y_normalized)
    };
    let mut parameters = Vector4::new(
        initial_center_gain.clamp(0.5, 2.0).ln(),
        initial_log_gain_slope_y.clamp(-0.35, 0.35),
        0.0,
        0.0,
    );

    for _ in 0..10 {
        let mut absolute_residuals = samples
            .iter()
            .step_by(stride)
            .map(|sample| {
                let (left, right, y) = sample_values(sample);
                let gain = (parameters[0] + parameters[1] * y).exp();
                let predicted = right * gain + parameters[2] + parameters[3] * y;
                (left - predicted).abs()
            })
            .collect::<Vec<_>>();
        let robust_scale = nearest_rank_percentile(std::mem::take(&mut absolute_residuals), 0.5)
            .mul_add(2.5, 0.0)
            .max(5e-5);
        let mut normal = Matrix4::<f64>::identity() * 1e-8;
        let mut right_hand_side = Vector4::<f64>::zeros();
        for sample in samples.iter().step_by(stride) {
            let (left, right, y) = sample_values(sample);
            let gain = (parameters[0] + parameters[1] * y).exp();
            let predicted = right * gain + parameters[2] + parameters[3] * y;
            let residual = left - predicted;
            let weight = if residual.abs() <= robust_scale {
                1.0
            } else {
                robust_scale / residual.abs().max(1e-12)
            };
            let jacobian = Vector4::new(right * gain, right * gain * y, 1.0, y);
            normal += jacobian * jacobian.transpose() * weight;
            right_hand_side += jacobian * (residual * weight);
        }
        let Some(mut update) = normal.lu().solve(&right_hand_side) else {
            break;
        };
        update[0] = update[0].clamp(-0.06, 0.06);
        update[1] = update[1].clamp(-0.06, 0.06);
        update[2] = update[2].clamp(-0.03, 0.03);
        update[3] = update[3].clamp(-0.03, 0.03);
        parameters += update;
        parameters[0] = parameters[0].clamp(0.5f64.ln(), 2.0f64.ln());
        parameters[1] = parameters[1].clamp(-0.35, 0.35);
        parameters[2] = parameters[2].clamp(-0.10, 0.10);
        parameters[3] = parameters[3].clamp(-0.10, 0.10);
        if update.iter().all(|value| value.abs() < 1e-7) {
            break;
        }
    }

    [
        parameters[0],
        parameters[1],
        parameters[2] * max_value,
        parameters[3] * max_value,
    ]
}

fn fit_spatial_affine(
    samples: &[SeamSample],
    gain_seed: &SpatialGainFit,
    max_value: f64,
) -> SpatialAffineFit {
    let rgb: [[f64; 4]; 3] = std::array::from_fn(|channel| {
        robust_spatial_affine_channel(
            samples,
            Some(channel),
            gain_seed.center_gain_rgb[channel],
            gain_seed.log_slope_y_rgb[channel],
            max_value,
        )
    });
    let luma = robust_spatial_affine_channel(
        samples,
        None,
        gain_seed.center_gain_luma,
        gain_seed.log_slope_y_luma,
        max_value,
    );
    SpatialAffineFit {
        center_gain_rgb: std::array::from_fn(|channel| rgb[channel][0].exp()),
        center_gain_luma: luma[0].exp(),
        log_gain_slope_y_rgb: std::array::from_fn(|channel| rgb[channel][1]),
        log_gain_slope_y_luma: luma[1],
        center_offset_rgb: std::array::from_fn(|channel| rgb[channel][2]),
        center_offset_luma: luma[2],
        offset_slope_y_rgb: std::array::from_fn(|channel| rgb[channel][3]),
        offset_slope_y_luma: luma[3],
    }
}

#[allow(clippy::too_many_arguments)]
fn robust_spatial_affine_2d_channel(
    samples: &[SeamSample],
    channel: Option<usize>,
    initial_center_gain: f64,
    initial_log_gain_slope_x: f64,
    initial_log_gain_slope_y: f64,
    initial_center_offset: f64,
    initial_offset_slope_y: f64,
    max_value: f64,
) -> [f64; 6] {
    if samples.is_empty() {
        return [0.0; 6];
    }
    let stride = samples.len().div_ceil(12_000).max(1);
    let sample_values = |sample: &SeamSample| {
        let left = channel
            .map(|index| sample.left[index])
            .unwrap_or_else(|| rgb_luma(&sample.left))
            / max_value;
        let right = channel
            .map(|index| sample.right[index])
            .unwrap_or_else(|| rgb_luma(&sample.right))
            / max_value;
        (left, right, sample.x_normalized, sample.y_normalized)
    };
    let mut parameters = SVector::<f64, 6>::from_row_slice(&[
        initial_center_gain.clamp(0.5, 2.0).ln(),
        initial_log_gain_slope_x.clamp(-0.35, 0.35),
        initial_log_gain_slope_y.clamp(-0.35, 0.35),
        (initial_center_offset / max_value).clamp(-0.10, 0.10),
        0.0,
        (initial_offset_slope_y / max_value).clamp(-0.10, 0.10),
    ]);

    for _ in 0..8 {
        let absolute_residuals = samples
            .iter()
            .step_by(stride)
            .map(|sample| {
                let (left, right, x, y) = sample_values(sample);
                let gain = (parameters[0] + parameters[1] * x + parameters[2] * y).exp();
                let predicted =
                    right * gain + parameters[3] + parameters[4] * x + parameters[5] * y;
                (left - predicted).abs()
            })
            .collect::<Vec<_>>();
        let robust_scale = nearest_rank_percentile(absolute_residuals, 0.5)
            .mul_add(2.5, 0.0)
            .max(5e-5);
        let mut normal = SMatrix::<f64, 6, 6>::identity() * 1e-8;
        let mut right_hand_side = SVector::<f64, 6>::zeros();
        for sample in samples.iter().step_by(stride) {
            let (left, right, x, y) = sample_values(sample);
            let gain = (parameters[0] + parameters[1] * x + parameters[2] * y).exp();
            let predicted = right * gain + parameters[3] + parameters[4] * x + parameters[5] * y;
            let residual = left - predicted;
            let weight = if residual.abs() <= robust_scale {
                1.0
            } else {
                robust_scale / residual.abs().max(1e-12)
            };
            let jacobian = SVector::<f64, 6>::from_row_slice(&[
                right * gain,
                right * gain * x,
                right * gain * y,
                1.0,
                x,
                y,
            ]);
            normal += jacobian * jacobian.transpose() * weight;
            right_hand_side += jacobian * (residual * weight);
        }
        let Some(mut update) = normal.lu().solve(&right_hand_side) else {
            break;
        };
        for parameter in update.iter_mut().take(3) {
            *parameter = parameter.clamp(-0.06, 0.06);
        }
        for parameter in update.iter_mut().skip(3) {
            *parameter = parameter.clamp(-0.03, 0.03);
        }
        parameters += update;
        parameters[0] = parameters[0].clamp(0.5f64.ln(), 2.0f64.ln());
        parameters[1] = parameters[1].clamp(-0.35, 0.35);
        parameters[2] = parameters[2].clamp(-0.35, 0.35);
        parameters[3] = parameters[3].clamp(-0.10, 0.10);
        parameters[4] = parameters[4].clamp(-0.10, 0.10);
        parameters[5] = parameters[5].clamp(-0.10, 0.10);
        if update.iter().all(|value| value.abs() < 1e-7) {
            break;
        }
    }

    [
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[3] * max_value,
        parameters[4] * max_value,
        parameters[5] * max_value,
    ]
}

fn fit_spatial_affine_2d(
    samples: &[SeamSample],
    gain_seed: &SpatialGain2dFit,
    affine_seed: &SpatialAffineFit,
    max_value: f64,
) -> SpatialAffine2dFit {
    let rgb: [[f64; 6]; 3] = std::array::from_fn(|channel| {
        robust_spatial_affine_2d_channel(
            samples,
            Some(channel),
            gain_seed.center_gain_rgb[channel],
            gain_seed.log_slope_x_rgb[channel],
            gain_seed.log_slope_y_rgb[channel],
            affine_seed.center_offset_rgb[channel],
            affine_seed.offset_slope_y_rgb[channel],
            max_value,
        )
    });
    let luma = robust_spatial_affine_2d_channel(
        samples,
        None,
        gain_seed.center_gain_luma,
        gain_seed.log_slope_x_luma,
        gain_seed.log_slope_y_luma,
        affine_seed.center_offset_luma,
        affine_seed.offset_slope_y_luma,
        max_value,
    );
    SpatialAffine2dFit {
        center_gain_rgb: std::array::from_fn(|channel| rgb[channel][0].exp()),
        center_gain_luma: luma[0].exp(),
        log_gain_slope_x_rgb: std::array::from_fn(|channel| rgb[channel][1]),
        log_gain_slope_x_luma: luma[1],
        log_gain_slope_y_rgb: std::array::from_fn(|channel| rgb[channel][2]),
        log_gain_slope_y_luma: luma[2],
        center_offset_rgb: std::array::from_fn(|channel| rgb[channel][3]),
        center_offset_luma: luma[3],
        offset_slope_x_rgb: std::array::from_fn(|channel| rgb[channel][4]),
        offset_slope_x_luma: luma[4],
        offset_slope_y_rgb: std::array::from_fn(|channel| rgb[channel][5]),
        offset_slope_y_luma: luma[5],
    }
}

fn quadratic_affine_condition_number(normal: SMatrix<f64, 12, 12>) -> f64 {
    let singular_values = normal.svd(false, false).singular_values;
    let maximum = singular_values.iter().copied().fold(0.0f64, f64::max);
    let minimum = singular_values
        .iter()
        .copied()
        .filter(|value| *value > 1e-12)
        .fold(f64::INFINITY, f64::min);
    if !maximum.is_finite() || !minimum.is_finite() || minimum <= 0.0 {
        1.0e12
    } else {
        (maximum / minimum).sqrt()
    }
}

fn robust_spatial_quadratic_affine_2d_channel(
    samples: &[SeamSample],
    channel: Option<usize>,
    initial_log_gain: &[f64; 6],
    initial_offset: &[f64; 3],
    max_value: f64,
) -> ([f64; 12], f64) {
    if samples.is_empty() {
        return ([0.0; 12], 1.0e12);
    }
    // A spatially uniform deterministic subsample keeps the 12-parameter robust solve bounded;
    // held-out whole windows, not training sample volume, decide whether the model is accepted.
    let stride = samples.len().div_ceil(2_500).max(1);
    let sample_values = |sample: &SeamSample| {
        let left = channel
            .map(|index| sample.left[index])
            .unwrap_or_else(|| rgb_luma(&sample.left))
            / max_value;
        let right = channel
            .map(|index| sample.right[index])
            .unwrap_or_else(|| rgb_luma(&sample.right))
            / max_value;
        (left, right, sample.x_normalized, sample.y_normalized)
    };
    let mut parameters = SVector::<f64, 12>::zeros();
    for index in 0..6 {
        parameters[index] = initial_log_gain[index];
    }
    for index in 0..3 {
        parameters[index + 6] = (initial_offset[index] / max_value).clamp(-0.10, 0.10);
    }

    for _ in 0..6 {
        let absolute_residuals = samples
            .iter()
            .step_by(stride)
            .map(|sample| {
                let (left, right, x, y) = sample_values(sample);
                let basis = quadratic_surface_basis(x, y);
                let gain = parameters.fixed_rows::<6>(0).dot(&basis).exp();
                let offset = parameters.fixed_rows::<6>(6).dot(&basis);
                (left - (right * gain + offset)).abs()
            })
            .collect::<Vec<_>>();
        let robust_scale = nearest_rank_percentile(absolute_residuals, 0.5)
            .mul_add(2.5, 0.0)
            .max(5e-5);
        let mut normal = SMatrix::<f64, 12, 12>::identity() * 1e-8;
        let mut right_hand_side = SVector::<f64, 12>::zeros();
        for index in [3usize, 4, 5, 9, 10, 11] {
            normal[(index, index)] += SEAM_EXPOSURE_QUADRATIC_REGULARIZATION;
            right_hand_side[index] -= SEAM_EXPOSURE_QUADRATIC_REGULARIZATION * parameters[index];
        }
        for sample in samples.iter().step_by(stride) {
            let (left, right, x, y) = sample_values(sample);
            let basis = quadratic_surface_basis(x, y);
            let gain = parameters.fixed_rows::<6>(0).dot(&basis).exp();
            let offset = parameters.fixed_rows::<6>(6).dot(&basis);
            let residual = left - (right * gain + offset);
            let weight = if residual.abs() <= robust_scale {
                1.0
            } else {
                robust_scale / residual.abs().max(1e-12)
            };
            let mut jacobian = SVector::<f64, 12>::zeros();
            for index in 0..6 {
                jacobian[index] = right * gain * basis[index];
                jacobian[index + 6] = basis[index];
            }
            normal += jacobian * jacobian.transpose() * weight;
            right_hand_side += jacobian * (residual * weight);
        }
        let Some(mut update) = normal.lu().solve(&right_hand_side) else {
            break;
        };
        for coefficient in update.iter_mut().take(6) {
            *coefficient = coefficient.clamp(-0.05, 0.05);
        }
        for coefficient in update.iter_mut().skip(6) {
            *coefficient = coefficient.clamp(-0.025, 0.025);
        }
        parameters += update;
        parameters[0] = parameters[0].clamp(0.5f64.ln(), 2.0f64.ln());
        parameters[1] = parameters[1].clamp(-0.35, 0.35);
        parameters[2] = parameters[2].clamp(-0.35, 0.35);
        for coefficient in parameters.iter_mut().take(6).skip(3) {
            *coefficient = coefficient.clamp(-0.20, 0.20);
        }
        for coefficient in parameters.iter_mut().skip(6) {
            *coefficient = coefficient.clamp(-0.10, 0.10);
        }
        if update.iter().all(|value| value.abs() < 1e-7) {
            break;
        }
    }

    let mut design_normal = SMatrix::<f64, 12, 12>::zeros();
    for sample in samples.iter().step_by(stride) {
        let (_, right, x, y) = sample_values(sample);
        let basis = quadratic_surface_basis(x, y);
        let gain = parameters.fixed_rows::<6>(0).dot(&basis).exp();
        let mut jacobian = SVector::<f64, 12>::zeros();
        for index in 0..6 {
            jacobian[index] = right * gain * basis[index];
            jacobian[index + 6] = basis[index];
        }
        design_normal += jacobian * jacobian.transpose();
    }
    let condition_number = quadratic_affine_condition_number(design_normal);
    let output = std::array::from_fn(|index| {
        if index < 6 {
            parameters[index]
        } else {
            parameters[index] * max_value
        }
    });
    (output, condition_number)
}

fn fit_spatial_quadratic_affine_2d(
    samples: &[SeamSample],
    gain_seed: &SpatialQuadraticGain2dFit,
    affine_seed: &SpatialAffine2dFit,
    max_value: f64,
) -> SpatialQuadraticAffine2dFit {
    let rgb: [([f64; 12], f64); 3] = std::array::from_fn(|channel| {
        robust_spatial_quadratic_affine_2d_channel(
            samples,
            Some(channel),
            &gain_seed.log_gain_rgb[channel],
            &[
                affine_seed.center_offset_rgb[channel],
                affine_seed.offset_slope_x_rgb[channel],
                affine_seed.offset_slope_y_rgb[channel],
            ],
            max_value,
        )
    });
    let luma = robust_spatial_quadratic_affine_2d_channel(
        samples,
        None,
        &gain_seed.log_gain_luma,
        &[
            affine_seed.center_offset_luma,
            affine_seed.offset_slope_x_luma,
            affine_seed.offset_slope_y_luma,
        ],
        max_value,
    );
    SpatialQuadraticAffine2dFit {
        log_gain_rgb: std::array::from_fn(|channel| {
            std::array::from_fn(|index| rgb[channel].0[index])
        }),
        log_gain_luma: std::array::from_fn(|index| luma.0[index]),
        offset_rgb: std::array::from_fn(|channel| {
            std::array::from_fn(|index| rgb[channel].0[index + 6])
        }),
        offset_luma: std::array::from_fn(|index| luma.0[index + 6]),
        design_condition_number: rgb
            .iter()
            .map(|fit| fit.1)
            .chain(std::iter::once(luma.1))
            .fold(0.0f64, f64::max),
    }
}

fn spatial_quadratic_affine_to_field(fit: &SpatialQuadraticAffine2dFit) -> SpatialPhotometricField {
    SpatialPhotometricField {
        center_gain: std::array::from_fn(|channel| fit.log_gain_rgb[channel][0].exp()),
        center_offset: std::array::from_fn(|channel| fit.offset_rgb[channel][0]),
        log_gain_slope_x: std::array::from_fn(|channel| fit.log_gain_rgb[channel][1]),
        log_gain_slope_y: std::array::from_fn(|channel| fit.log_gain_rgb[channel][2]),
        log_gain_quadratic_xx: std::array::from_fn(|channel| fit.log_gain_rgb[channel][3]),
        log_gain_quadratic_xy: std::array::from_fn(|channel| fit.log_gain_rgb[channel][4]),
        log_gain_quadratic_yy: std::array::from_fn(|channel| fit.log_gain_rgb[channel][5]),
        offset_slope_x: std::array::from_fn(|channel| fit.offset_rgb[channel][1]),
        offset_slope_y: std::array::from_fn(|channel| fit.offset_rgb[channel][2]),
        offset_quadratic_xx: std::array::from_fn(|channel| fit.offset_rgb[channel][3]),
        offset_quadratic_xy: std::array::from_fn(|channel| fit.offset_rgb[channel][4]),
        offset_quadratic_yy: std::array::from_fn(|channel| fit.offset_rgb[channel][5]),
    }
}

fn spatial_quadratic_affine_from_planar(fit: &SpatialAffine2dFit) -> SpatialQuadraticAffine2dFit {
    SpatialQuadraticAffine2dFit {
        log_gain_rgb: std::array::from_fn(|channel| {
            [
                fit.center_gain_rgb[channel].max(1e-9).ln(),
                fit.log_gain_slope_x_rgb[channel],
                fit.log_gain_slope_y_rgb[channel],
                0.0,
                0.0,
                0.0,
            ]
        }),
        log_gain_luma: [
            fit.center_gain_luma.max(1e-9).ln(),
            fit.log_gain_slope_x_luma,
            fit.log_gain_slope_y_luma,
            0.0,
            0.0,
            0.0,
        ],
        offset_rgb: std::array::from_fn(|channel| {
            [
                fit.center_offset_rgb[channel],
                fit.offset_slope_x_rgb[channel],
                fit.offset_slope_y_rgb[channel],
                0.0,
                0.0,
                0.0,
            ]
        }),
        offset_luma: [
            fit.center_offset_luma,
            fit.offset_slope_x_luma,
            fit.offset_slope_y_luma,
            0.0,
            0.0,
            0.0,
        ],
        design_condition_number: 0.0,
    }
}

fn spatial_affine_2d_gain_at(
    fit: &SpatialAffine2dFit,
    x_normalized: f64,
    y_normalized: f64,
) -> [f64; 3] {
    spatial_gain_at_normalized_xy(
        &fit.center_gain_rgb,
        &fit.log_gain_slope_x_rgb,
        &fit.log_gain_slope_y_rgb,
        x_normalized,
        y_normalized,
    )
}

fn spatial_affine_2d_offset_at(
    fit: &SpatialAffine2dFit,
    x_normalized: f64,
    y_normalized: f64,
) -> [f64; 3] {
    std::array::from_fn(|channel| {
        fit.center_offset_rgb[channel]
            + fit.offset_slope_x_rgb[channel] * x_normalized
            + fit.offset_slope_y_rgb[channel] * y_normalized
    })
}

fn spatial_affine_2d_gain_corners(fit: &SpatialAffine2dFit) -> [[f64; 3]; 4] {
    SPATIAL_FIELD_CORNERS.map(|[x, y]| spatial_affine_2d_gain_at(fit, x, y))
}

fn spatial_affine_2d_offset_corners(fit: &SpatialAffine2dFit) -> [[f64; 3]; 4] {
    SPATIAL_FIELD_CORNERS.map(|[x, y]| spatial_affine_2d_offset_at(fit, x, y))
}

fn spatial_affine_gain_at_y(fit: &SpatialAffineFit, y_normalized: f64) -> [f64; 3] {
    spatial_gain_at_normalized_y(
        &fit.center_gain_rgb,
        &fit.log_gain_slope_y_rgb,
        y_normalized,
    )
}

fn spatial_affine_offset_at_y(fit: &SpatialAffineFit, y_normalized: f64) -> [f64; 3] {
    std::array::from_fn(|channel| {
        fit.center_offset_rgb[channel] + fit.offset_slope_y_rgb[channel] * y_normalized
    })
}

fn distinct_window_rows(windows: &[SeamWindow], indices: &[usize]) -> usize {
    indices
        .iter()
        .map(|index| windows[*index].row)
        .collect::<BTreeSet<_>>()
        .len()
}

fn distinct_window_columns(windows: &[SeamWindow], indices: &[usize]) -> usize {
    indices
        .iter()
        .map(|index| windows[*index].col)
        .collect::<BTreeSet<_>>()
        .len()
}

fn spatial_gain_2d_window_consistency(
    windows: &[SeamWindow],
    indices: &[usize],
    fit: &SpatialGain2dFit,
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let consistent = indices
        .iter()
        .flat_map(|index| {
            let window = &windows[*index];
            (0..3).map(move |channel| {
                let predicted = fit.center_gain_rgb[channel].max(1e-6).ln()
                    + fit.log_slope_x_rgb[channel] * window.x_normalized
                    + fit.log_slope_y_rgb[channel] * window.y_normalized;
                (window.estimate.gain_rgb[channel].max(1e-6).ln() - predicted).abs() <= 0.07
            })
        })
        .filter(|consistent| *consistent)
        .count();
    consistent as f64 / (indices.len() * 3) as f64
}

fn spatial_gain_2d_slope_agreement_ratios(
    training: &SpatialGain2dFit,
    held_out: &SpatialGain2dFit,
) -> (f64, f64) {
    let minimum_slope = SEAM_EXPOSURE_MIN_SPATIAL_ENDPOINT_RATIO.ln() * 0.5;
    let mut relevant = 0usize;
    let mut agreeing = 0usize;
    let mut horizontal_relevant = 0usize;
    let mut horizontal_agreeing = 0usize;
    for channel in 0..3 {
        for (axis, training_slope, held_out_slope) in [
            (
                0,
                training.log_slope_x_rgb[channel],
                held_out.log_slope_x_rgb[channel],
            ),
            (
                1,
                training.log_slope_y_rgb[channel],
                held_out.log_slope_y_rgb[channel],
            ),
        ] {
            if training_slope.abs().max(held_out_slope.abs()) < minimum_slope {
                continue;
            }
            relevant += 1;
            if axis == 0 {
                horizontal_relevant += 1;
            }
            let agrees = training_slope * held_out_slope > 0.0
                && (training_slope - held_out_slope).abs() <= SEAM_EXPOSURE_MAX_SPATIAL_SLOPE_DELTA;
            if agrees {
                agreeing += 1;
                if axis == 0 {
                    horizontal_agreeing += 1;
                }
            }
        }
    }
    (
        if relevant == 0 {
            0.0
        } else {
            agreeing as f64 / relevant as f64
        },
        if horizontal_relevant == 0 {
            0.0
        } else {
            horizontal_agreeing as f64 / horizontal_relevant as f64
        },
    )
}

fn spatial_center_gain_log_delta(training: &[f64; 3], held_out: &[f64; 3]) -> f64 {
    (0..3)
        .map(|channel| (training[channel].max(1e-6).ln() - held_out[channel].max(1e-6).ln()).abs())
        .fold(0.0f64, f64::max)
}

fn spatial_affine_2d_window_consistency(
    windows: &[SeamWindow],
    indices: &[usize],
    fit: &SpatialAffine2dFit,
    max_value: f64,
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let consistent = indices
        .iter()
        .filter(|index| {
            let window = &windows[**index];
            let before = seam_measurements(&window.samples, &[1.0; 3], &[0.0; 3], max_value);
            let after = seam_measurements_spatial_affine_2d(
                &window.samples,
                &fit.center_gain_rgb,
                &fit.center_offset_rgb,
                &fit.log_gain_slope_x_rgb,
                &fit.log_gain_slope_y_rgb,
                &fit.offset_slope_x_rgb,
                &fit.offset_slope_y_rgb,
                max_value,
            );
            let required_improvement = if before.seam_score >= 0.02 {
                0.002f64.max(before.seam_score * 0.10)
            } else {
                0.0
            };
            after.seam_score <= before.seam_score - required_improvement + 1e-9
        })
        .count();
    consistent as f64 / indices.len() as f64
}

fn spatial_affine_2d_slope_agreement_ratios(
    training: &SpatialAffine2dFit,
    held_out: &SpatialAffine2dFit,
    max_value: f64,
) -> (f64, f64) {
    let minimum_gain_slope = SEAM_EXPOSURE_MIN_SPATIAL_ENDPOINT_RATIO.ln() * 0.5;
    let minimum_offset_slope =
        max_value * SEAM_EXPOSURE_MIN_SPATIAL_OFFSET_ENDPOINT_DELTA_NORMALIZED * 0.5;
    let mut relevant = 0usize;
    let mut agreeing = 0usize;
    let mut horizontal_relevant = 0usize;
    let mut horizontal_agreeing = 0usize;
    for channel in 0..3 {
        for (axis, training_slope, held_out_slope) in [
            (
                0,
                training.log_gain_slope_x_rgb[channel],
                held_out.log_gain_slope_x_rgb[channel],
            ),
            (
                1,
                training.log_gain_slope_y_rgb[channel],
                held_out.log_gain_slope_y_rgb[channel],
            ),
        ] {
            if training_slope.abs().max(held_out_slope.abs()) < minimum_gain_slope {
                continue;
            }
            relevant += 1;
            if axis == 0 {
                horizontal_relevant += 1;
            }
            let agrees = training_slope * held_out_slope > 0.0
                && (training_slope - held_out_slope).abs() <= SEAM_EXPOSURE_MAX_SPATIAL_SLOPE_DELTA;
            if agrees {
                agreeing += 1;
                if axis == 0 {
                    horizontal_agreeing += 1;
                }
            }
        }
        for (axis, training_slope, held_out_slope) in [
            (
                0,
                training.offset_slope_x_rgb[channel],
                held_out.offset_slope_x_rgb[channel],
            ),
            (
                1,
                training.offset_slope_y_rgb[channel],
                held_out.offset_slope_y_rgb[channel],
            ),
        ] {
            if training_slope.abs().max(held_out_slope.abs()) < minimum_offset_slope {
                continue;
            }
            relevant += 1;
            if axis == 0 {
                horizontal_relevant += 1;
            }
            let agrees = training_slope * held_out_slope > 0.0
                && (training_slope - held_out_slope).abs() / max_value
                    <= SEAM_EXPOSURE_MAX_SPATIAL_OFFSET_SLOPE_DELTA_NORMALIZED;
            if agrees {
                agreeing += 1;
                if axis == 0 {
                    horizontal_agreeing += 1;
                }
            }
        }
    }
    (
        if relevant == 0 {
            0.0
        } else {
            agreeing as f64 / relevant as f64
        },
        if horizontal_relevant == 0 {
            0.0
        } else {
            horizontal_agreeing as f64 / horizontal_relevant as f64
        },
    )
}

fn spatial_affine_2d_center_offset_delta_normalized(
    training: &SpatialAffine2dFit,
    held_out: &SpatialAffine2dFit,
    max_value: f64,
) -> f64 {
    (0..3)
        .map(|channel| {
            (training.center_offset_rgb[channel] - held_out.center_offset_rgb[channel]).abs()
                / max_value
        })
        .fold(0.0f64, f64::max)
}

fn spatial_window_consistency(
    windows: &[SeamWindow],
    indices: &[usize],
    fit: &SpatialGainFit,
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let consistent = indices
        .iter()
        .flat_map(|index| {
            let window = &windows[*index];
            (0..3).map(move |channel| {
                let predicted = fit.center_gain_rgb[channel].max(1e-6).ln()
                    + fit.log_slope_y_rgb[channel] * window.y_normalized;
                (window.estimate.gain_rgb[channel].max(1e-6).ln() - predicted).abs() <= 0.07
            })
        })
        .filter(|consistent| *consistent)
        .count();
    consistent as f64 / (indices.len() * 3) as f64
}

fn spatial_slope_agreement_ratio(training: &SpatialGainFit, held_out: &SpatialGainFit) -> f64 {
    let minimum_slope = SEAM_EXPOSURE_MIN_SPATIAL_ENDPOINT_RATIO.ln() * 0.5;
    let mut relevant = 0usize;
    let mut agreeing = 0usize;
    for channel in 0..3 {
        let training_slope = training.log_slope_y_rgb[channel];
        let held_out_slope = held_out.log_slope_y_rgb[channel];
        if training_slope.abs().max(held_out_slope.abs()) < minimum_slope {
            continue;
        }
        relevant += 1;
        if training_slope * held_out_slope > 0.0
            && (training_slope - held_out_slope).abs() <= SEAM_EXPOSURE_MAX_SPATIAL_SLOPE_DELTA
        {
            agreeing += 1;
        }
    }
    if relevant == 0 {
        0.0
    } else {
        agreeing as f64 / relevant as f64
    }
}

fn spatial_endpoint_ratio(log_slope_y_rgb: &[f64; 3]) -> f64 {
    log_slope_y_rgb
        .iter()
        .map(|slope| (2.0 * slope.abs()).exp())
        .fold(1.0f64, f64::max)
}

fn spatial_affine_window_consistency(
    windows: &[SeamWindow],
    indices: &[usize],
    fit: &SpatialAffineFit,
    max_value: f64,
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let consistent = indices
        .iter()
        .filter(|index| {
            let window = &windows[**index];
            let before = seam_measurements(&window.samples, &[1.0; 3], &[0.0; 3], max_value);
            let after = seam_measurements_spatial_affine(
                &window.samples,
                &fit.center_gain_rgb,
                &fit.center_offset_rgb,
                &fit.log_gain_slope_y_rgb,
                &fit.offset_slope_y_rgb,
                max_value,
            );
            let required_improvement = if before.seam_score >= 0.02 {
                0.002f64.max(before.seam_score * 0.10)
            } else {
                0.0
            };
            after.seam_score <= before.seam_score - required_improvement + 1e-9
        })
        .count();
    consistent as f64 / indices.len() as f64
}

fn spatial_affine_slope_agreement_ratio(
    training: &SpatialAffineFit,
    held_out: &SpatialAffineFit,
    max_value: f64,
) -> f64 {
    let minimum_gain_slope = SEAM_EXPOSURE_MIN_SPATIAL_ENDPOINT_RATIO.ln() * 0.5;
    let minimum_offset_slope =
        max_value * SEAM_EXPOSURE_MIN_SPATIAL_OFFSET_ENDPOINT_DELTA_NORMALIZED * 0.5;
    let mut relevant = 0usize;
    let mut agreeing = 0usize;
    for channel in 0..3 {
        let training_gain = training.log_gain_slope_y_rgb[channel];
        let held_out_gain = held_out.log_gain_slope_y_rgb[channel];
        if training_gain.abs().max(held_out_gain.abs()) >= minimum_gain_slope {
            relevant += 1;
            if training_gain * held_out_gain > 0.0
                && (training_gain - held_out_gain).abs() <= SEAM_EXPOSURE_MAX_SPATIAL_SLOPE_DELTA
            {
                agreeing += 1;
            }
        }

        let training_offset = training.offset_slope_y_rgb[channel];
        let held_out_offset = held_out.offset_slope_y_rgb[channel];
        if training_offset.abs().max(held_out_offset.abs()) >= minimum_offset_slope {
            relevant += 1;
            if training_offset * held_out_offset > 0.0
                && (training_offset - held_out_offset).abs() / max_value
                    <= SEAM_EXPOSURE_MAX_SPATIAL_OFFSET_SLOPE_DELTA_NORMALIZED
            {
                agreeing += 1;
            }
        }
    }
    if relevant == 0 {
        0.0
    } else {
        agreeing as f64 / relevant as f64
    }
}

fn spatial_affine_center_offset_delta_normalized(
    training: &SpatialAffineFit,
    held_out: &SpatialAffineFit,
    max_value: f64,
) -> f64 {
    (0..3)
        .map(|channel| {
            (training.center_offset_rgb[channel] - held_out.center_offset_rgb[channel]).abs()
                / max_value
        })
        .fold(0.0f64, f64::max)
}

fn maximum_normalized_magnitude(values: &[f64; 3], max_value: f64) -> f64 {
    values
        .iter()
        .map(|value| value.abs() / max_value)
        .fold(0.0f64, f64::max)
}

fn maximum_normalized_endpoint_delta(top: &[f64; 3], bottom: &[f64; 3], max_value: f64) -> f64 {
    (0..3)
        .map(|channel| (top[channel] - bottom[channel]).abs() / max_value)
        .fold(0.0f64, f64::max)
}

#[derive(Debug, Clone, Copy)]
struct SpatialFieldGridBounds {
    gain_min_rgb: [f64; 3],
    gain_max_rgb: [f64; 3],
    maximum_gain_rgb_spread: f64,
    offset_abs_max_normalized: f64,
}

fn for_each_spatial_evaluation_grid_point(mut evaluate: impl FnMut(f64, f64)) {
    let denominator = (SEAM_EXPOSURE_QUADRATIC_EVALUATION_GRID_SIZE - 1) as f64;
    for row in 0..SEAM_EXPOSURE_QUADRATIC_EVALUATION_GRID_SIZE {
        let y = -1.0 + 2.0 * row as f64 / denominator;
        for column in 0..SEAM_EXPOSURE_QUADRATIC_EVALUATION_GRID_SIZE {
            let x = -1.0 + 2.0 * column as f64 / denominator;
            evaluate(x, y);
        }
    }
}

fn for_each_quadratic_surface_extremum(coefficients: [f64; 6], mut evaluate: impl FnMut(f64, f64)) {
    for [x, y] in SPATIAL_FIELD_CORNERS {
        evaluate(x, y);
    }

    let [_, slope_x, slope_y, quadratic_xx, quadratic_xy, quadratic_yy] = coefficients;
    let determinant = 4.0 * quadratic_xx * quadratic_yy - quadratic_xy * quadratic_xy;
    if determinant.abs() > 1e-12 {
        let x = (-2.0 * quadratic_yy * slope_x + quadratic_xy * slope_y) / determinant;
        let y = (quadratic_xy * slope_x - 2.0 * quadratic_xx * slope_y) / determinant;
        if (-1.0..=1.0).contains(&x) && (-1.0..=1.0).contains(&y) {
            evaluate(x, y);
        }
    }

    if quadratic_yy.abs() > 1e-12 {
        for x in [-1.0, 1.0] {
            let y = -(slope_y + quadratic_xy * x) / (2.0 * quadratic_yy);
            if (-1.0..=1.0).contains(&y) {
                evaluate(x, y);
            }
        }
    }
    if quadratic_xx.abs() > 1e-12 {
        for y in [-1.0, 1.0] {
            let x = -(slope_x + quadratic_xy * y) / (2.0 * quadratic_xx);
            if (-1.0..=1.0).contains(&x) {
                evaluate(x, y);
            }
        }
    }
}

fn quadratic_surface_value(coefficients: &[f64; 6], x: f64, y: f64) -> f64 {
    coefficients[0]
        + coefficients[1] * x
        + coefficients[2] * y
        + coefficients[3] * x * x
        + coefficients[4] * x * y
        + coefficients[5] * y * y
}

fn update_spatial_field_bounds(
    bounds: &mut SpatialFieldGridBounds,
    field: &SpatialPhotometricField,
    max_value: f64,
    x: f64,
    y: f64,
) {
    let (gain, offset) = spatial_photometric_at_normalized_xy(field, x, y);
    for channel in 0..3 {
        bounds.gain_min_rgb[channel] = bounds.gain_min_rgb[channel].min(gain[channel]);
        bounds.gain_max_rgb[channel] = bounds.gain_max_rgb[channel].max(gain[channel]);
        bounds.offset_abs_max_normalized = bounds
            .offset_abs_max_normalized
            .max(offset[channel].abs() / max_value);
    }
    bounds.maximum_gain_rgb_spread = bounds.maximum_gain_rgb_spread.max(max_gain_spread(&gain));
}

fn spatial_field_grid_bounds(
    field: &SpatialPhotometricField,
    max_value: f64,
) -> SpatialFieldGridBounds {
    let mut bounds = SpatialFieldGridBounds {
        gain_min_rgb: [f64::INFINITY; 3],
        gain_max_rgb: [f64::NEG_INFINITY; 3],
        maximum_gain_rgb_spread: 1.0,
        offset_abs_max_normalized: 0.0,
    };
    for_each_spatial_evaluation_grid_point(|x, y| {
        update_spatial_field_bounds(&mut bounds, field, max_value, x, y);
    });

    // Keep the grid for auditable field coverage, then add every exact stationary point and edge
    // vertex so a quadratic cannot cross a hard physical bound between reported grid nodes.
    for channel in 0..3 {
        let gain_coefficients = [
            field.center_gain[channel].max(1e-12).ln(),
            field.log_gain_slope_x[channel],
            field.log_gain_slope_y[channel],
            field.log_gain_quadratic_xx[channel],
            field.log_gain_quadratic_xy[channel],
            field.log_gain_quadratic_yy[channel],
        ];
        for_each_quadratic_surface_extremum(gain_coefficients, |x, y| {
            update_spatial_field_bounds(&mut bounds, field, max_value, x, y);
        });
        let offset_coefficients = [
            field.center_offset[channel],
            field.offset_slope_x[channel],
            field.offset_slope_y[channel],
            field.offset_quadratic_xx[channel],
            field.offset_quadratic_xy[channel],
            field.offset_quadratic_yy[channel],
        ];
        for_each_quadratic_surface_extremum(offset_coefficients, |x, y| {
            update_spatial_field_bounds(&mut bounds, field, max_value, x, y);
        });
    }
    for first in 0..3 {
        for second in (first + 1)..3 {
            let spread_coefficients = [
                (field.center_gain[first] / field.center_gain[second].max(1e-12))
                    .max(1e-12)
                    .ln(),
                field.log_gain_slope_x[first] - field.log_gain_slope_x[second],
                field.log_gain_slope_y[first] - field.log_gain_slope_y[second],
                field.log_gain_quadratic_xx[first] - field.log_gain_quadratic_xx[second],
                field.log_gain_quadratic_xy[first] - field.log_gain_quadratic_xy[second],
                field.log_gain_quadratic_yy[first] - field.log_gain_quadratic_yy[second],
            ];
            for_each_quadratic_surface_extremum(spread_coefficients, |x, y| {
                update_spatial_field_bounds(&mut bounds, field, max_value, x, y);
            });
        }
    }
    bounds
}

fn spatial_field_gain_curvature_signal(field: &SpatialPhotometricField) -> f64 {
    let mut signal = 0.0f64;
    for channel in 0..3 {
        let coefficients = [
            0.0,
            0.0,
            0.0,
            field.log_gain_quadratic_xx[channel],
            field.log_gain_quadratic_xy[channel],
            field.log_gain_quadratic_yy[channel],
        ];
        for_each_quadratic_surface_extremum(coefficients, |x, y| {
            signal = signal.max(quadratic_surface_value(&coefficients, x, y).abs());
        });
    }
    signal
}

fn spatial_field_offset_curvature_signal_normalized(
    field: &SpatialPhotometricField,
    max_value: f64,
) -> f64 {
    let mut signal = 0.0f64;
    for channel in 0..3 {
        let coefficients = [
            0.0,
            0.0,
            0.0,
            field.offset_quadratic_xx[channel],
            field.offset_quadratic_xy[channel],
            field.offset_quadratic_yy[channel],
        ];
        for_each_quadratic_surface_extremum(coefficients, |x, y| {
            signal = signal.max(quadratic_surface_value(&coefficients, x, y).abs() / max_value);
        });
    }
    signal
}

#[allow(clippy::too_many_arguments)]
fn update_spatial_field_validation_deltas(
    training: &SpatialPhotometricField,
    held_out: &SpatialPhotometricField,
    max_value: f64,
    x: f64,
    y: f64,
    gain_log_delta: &mut f64,
    offset_delta: &mut f64,
) {
    let (training_gain, training_offset) = spatial_photometric_at_normalized_xy(training, x, y);
    let (held_out_gain, held_out_offset) = spatial_photometric_at_normalized_xy(held_out, x, y);
    for channel in 0..3 {
        *gain_log_delta = gain_log_delta.max(
            (training_gain[channel].max(1e-9).ln() - held_out_gain[channel].max(1e-9).ln()).abs(),
        );
        *offset_delta = offset_delta
            .max((training_offset[channel] - held_out_offset[channel]).abs() / max_value);
    }
}

fn spatial_field_validation_deltas(
    training: &SpatialPhotometricField,
    held_out: &SpatialPhotometricField,
    max_value: f64,
) -> (f64, f64) {
    let mut gain_log_delta = 0.0f64;
    let mut offset_delta = 0.0f64;
    for_each_spatial_evaluation_grid_point(|x, y| {
        update_spatial_field_validation_deltas(
            training,
            held_out,
            max_value,
            x,
            y,
            &mut gain_log_delta,
            &mut offset_delta,
        );
    });
    for channel in 0..3 {
        let gain_difference = [
            (training.center_gain[channel] / held_out.center_gain[channel].max(1e-12))
                .max(1e-12)
                .ln(),
            training.log_gain_slope_x[channel] - held_out.log_gain_slope_x[channel],
            training.log_gain_slope_y[channel] - held_out.log_gain_slope_y[channel],
            training.log_gain_quadratic_xx[channel] - held_out.log_gain_quadratic_xx[channel],
            training.log_gain_quadratic_xy[channel] - held_out.log_gain_quadratic_xy[channel],
            training.log_gain_quadratic_yy[channel] - held_out.log_gain_quadratic_yy[channel],
        ];
        for_each_quadratic_surface_extremum(gain_difference, |x, y| {
            update_spatial_field_validation_deltas(
                training,
                held_out,
                max_value,
                x,
                y,
                &mut gain_log_delta,
                &mut offset_delta,
            );
        });
        let offset_difference = [
            training.center_offset[channel] - held_out.center_offset[channel],
            training.offset_slope_x[channel] - held_out.offset_slope_x[channel],
            training.offset_slope_y[channel] - held_out.offset_slope_y[channel],
            training.offset_quadratic_xx[channel] - held_out.offset_quadratic_xx[channel],
            training.offset_quadratic_xy[channel] - held_out.offset_quadratic_xy[channel],
            training.offset_quadratic_yy[channel] - held_out.offset_quadratic_yy[channel],
        ];
        for_each_quadratic_surface_extremum(offset_difference, |x, y| {
            update_spatial_field_validation_deltas(
                training,
                held_out,
                max_value,
                x,
                y,
                &mut gain_log_delta,
                &mut offset_delta,
            );
        });
    }
    (gain_log_delta, offset_delta)
}

fn quadratic_curvature_coefficient_agreement_ratio(
    training: &SpatialPhotometricField,
    held_out: &SpatialPhotometricField,
    max_value: f64,
    include_offset: bool,
) -> f64 {
    let minimum_gain_coefficient = SEAM_EXPOSURE_MIN_QUADRATIC_GAIN_SIGNAL / 3.0;
    let minimum_offset_coefficient =
        max_value * SEAM_EXPOSURE_MIN_QUADRATIC_OFFSET_SIGNAL_NORMALIZED / 3.0;
    let mut relevant = 0usize;
    let mut agreeing = 0usize;
    for channel in 0..3 {
        for (training_coefficient, held_out_coefficient) in [
            (
                training.log_gain_quadratic_xx[channel],
                held_out.log_gain_quadratic_xx[channel],
            ),
            (
                training.log_gain_quadratic_xy[channel],
                held_out.log_gain_quadratic_xy[channel],
            ),
            (
                training.log_gain_quadratic_yy[channel],
                held_out.log_gain_quadratic_yy[channel],
            ),
        ] {
            if training_coefficient.abs().max(held_out_coefficient.abs()) < minimum_gain_coefficient
            {
                continue;
            }
            relevant += 1;
            if training_coefficient * held_out_coefficient > 0.0
                && (training_coefficient - held_out_coefficient).abs()
                    <= SEAM_EXPOSURE_MAX_QUADRATIC_COEFFICIENT_DELTA
            {
                agreeing += 1;
            }
        }
        if include_offset {
            for (training_coefficient, held_out_coefficient) in [
                (
                    training.offset_quadratic_xx[channel],
                    held_out.offset_quadratic_xx[channel],
                ),
                (
                    training.offset_quadratic_xy[channel],
                    held_out.offset_quadratic_xy[channel],
                ),
                (
                    training.offset_quadratic_yy[channel],
                    held_out.offset_quadratic_yy[channel],
                ),
            ] {
                if training_coefficient.abs().max(held_out_coefficient.abs())
                    < minimum_offset_coefficient
                {
                    continue;
                }
                relevant += 1;
                if training_coefficient * held_out_coefficient > 0.0
                    && (training_coefficient - held_out_coefficient).abs() / max_value
                        <= SEAM_EXPOSURE_MAX_QUADRATIC_OFFSET_FIELD_DELTA_NORMALIZED
                {
                    agreeing += 1;
                }
            }
        }
    }
    if relevant == 0 {
        0.0
    } else {
        agreeing as f64 / relevant as f64
    }
}

fn spatial_quadratic_gain_window_consistency(
    windows: &[SeamWindow],
    indices: &[usize],
    field: &SpatialPhotometricField,
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let consistent = indices
        .iter()
        .flat_map(|index| {
            let window = &windows[*index];
            let (predicted_gain, _) = spatial_photometric_at_normalized_xy(
                field,
                window.x_normalized,
                window.y_normalized,
            );
            (0..3).map(move |channel| {
                (window.estimate.gain_rgb[channel].max(1e-6).ln()
                    - predicted_gain[channel].max(1e-6).ln())
                .abs()
                    <= 0.07
            })
        })
        .filter(|consistent| *consistent)
        .count();
    consistent as f64 / (indices.len() * 3) as f64
}

fn spatial_quadratic_affine_window_consistency(
    windows: &[SeamWindow],
    indices: &[usize],
    field: &SpatialPhotometricField,
    max_value: f64,
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let consistent = indices
        .iter()
        .filter(|index| {
            let window = &windows[**index];
            let before = seam_measurements(&window.samples, &[1.0; 3], &[0.0; 3], max_value);
            let after = seam_measurements_photometric_field(&window.samples, field, max_value);
            let required_improvement = if before.seam_score >= 0.02 {
                0.002f64.max(before.seam_score * 0.10)
            } else {
                0.0
            };
            after.seam_score <= before.seam_score - required_improvement + 1e-9
        })
        .count();
    consistent as f64 / indices.len() as f64
}

fn filter_outlier_seam_samples(
    samples: Vec<SeamSample>,
    estimated_gain_luma: f64,
    estimated_gain_rgb: &[f64; 3],
) -> Vec<SeamSample> {
    samples
        .into_iter()
        .filter(|sample| {
            let (rgb, luma) = seam_sample_ratios(sample);
            if (luma / estimated_gain_luma.max(1e-6)).ln().abs() > 0.22 {
                return false;
            }
            for c in 0..3 {
                if (rgb[c] / estimated_gain_rgb[c].max(1e-6)).ln().abs() > 0.28 {
                    return false;
                }
            }
            true
        })
        .collect()
}

fn window_consistency_ratio(windows: &[SeamWindowEstimate], gain_luma: f64) -> f64 {
    if windows.is_empty() {
        return 0.0;
    }
    let target_log = gain_luma.max(1e-6).ln();
    let target_direction = if target_log.abs() < SEAM_EXPOSURE_MIN_SCORE {
        0.0
    } else {
        target_log.signum()
    };
    let consistent = windows
        .iter()
        .filter(|window| {
            let window_log = window.gain_luma.max(1e-6).ln();
            let magnitude_ok = (window_log - target_log).abs() <= 0.06;
            let direction_ok = target_direction == 0.0 || window_log.signum() == target_direction;
            magnitude_ok && direction_ok
        })
        .count();
    consistent as f64 / windows.len() as f64
}

fn per_channel_window_consistency(windows: &[SeamWindowEstimate], gain_rgb: &[f64; 3]) -> f64 {
    if windows.is_empty() {
        return 0.0;
    }
    let mut channel_scores = [0.0f64; 3];
    for c in 0..3 {
        let target_log = gain_rgb[c].max(1e-6).ln();
        let target_direction = if target_log.abs() < SEAM_EXPOSURE_MIN_SCORE {
            0.0
        } else {
            target_log.signum()
        };
        let consistent = windows
            .iter()
            .filter(|window| {
                let window_log = window.gain_rgb[c].max(1e-6).ln();
                let magnitude_ok = (window_log - target_log).abs() <= 0.07;
                let direction_ok =
                    target_direction == 0.0 || window_log.signum() == target_direction;
                magnitude_ok && direction_ok
            })
            .count();
        channel_scores[c] = consistent as f64 / windows.len() as f64;
    }
    channel_scores.iter().sum::<f64>() / 3.0
}

fn gain_offset_window_consistency(
    windows: &[SeamWindowEstimate],
    gain_rgb: &[f64; 3],
    offset_rgb: &[f64; 3],
    max_value: f64,
) -> f64 {
    if windows.is_empty() {
        return 0.0;
    }
    let consistent = windows
        .iter()
        .flat_map(|window| {
            (0..3).map(move |channel| {
                (window.affine_gain_rgb[channel] - gain_rgb[channel]).abs() <= 0.06
                    && (window.offset_rgb[channel] - offset_rgb[channel]).abs() <= max_value * 0.012
            })
        })
        .filter(|value| *value)
        .count();
    consistent as f64 / (windows.len() * 3) as f64
}

fn seam_exposure_correction(
    strip_l: &Array3<u16>,
    strip_r: &Array3<u16>,
    config: &StitchConfig,
) -> SeamExposureCorrectionDiagnostics {
    let max_value = stitch_sample_max(config);
    let zero_offset = [0.0; 3];
    let identity_gain = [1.0; 3];
    let (windows, total_pixels) = collect_seam_windows(strip_l, strip_r, max_value);
    let all_indices = (0..windows.len()).collect::<Vec<_>>();
    let raw_samples = seam_samples_for_windows(&windows, &all_indices);
    let all_estimates = seam_estimates_for_windows(&windows, &all_indices);
    let (training_indices, held_out_indices) = seam_window_split(&windows, false);
    let training_estimates = seam_estimates_for_windows(&windows, &training_indices);
    let raw_training_samples = seam_samples_for_windows(&windows, &training_indices);
    let held_out_samples = seam_samples_for_windows(&windows, &held_out_indices);

    let median_window_gain_rgb = |estimates: &[SeamWindowEstimate]| {
        std::array::from_fn(|channel| {
            median_ratio(
                &estimates
                    .iter()
                    .map(|window| window.gain_rgb[channel])
                    .collect::<Vec<_>>(),
            )
        })
    };
    let median_window_gain_luma = |estimates: &[SeamWindowEstimate]| {
        median_ratio(
            &estimates
                .iter()
                .map(|window| window.gain_luma)
                .collect::<Vec<_>>(),
        )
    };
    let estimated_gain_rgb = median_window_gain_rgb(&all_estimates);
    let estimated_gain_luma = median_window_gain_luma(&all_estimates);
    let training_gain_rgb = median_window_gain_rgb(&training_estimates);
    let training_gain_luma = median_window_gain_luma(&training_estimates);

    let filtered_samples =
        filter_outlier_seam_samples(raw_samples, training_gain_luma, &training_gain_rgb);
    let filtered_training_samples =
        filter_outlier_seam_samples(raw_training_samples, training_gain_luma, &training_gain_rgb);
    let valid_sample_ratio = if total_pixels == 0 {
        0.0
    } else {
        filtered_samples.len() as f64 / total_pixels as f64
    };
    let sample_count = filtered_samples.len();
    let measurements_before =
        seam_measurements(&filtered_samples, &identity_gain, &zero_offset, max_value);
    let (clipped_high_before, clipped_low_before) =
        strip_clipping_ratios(strip_r, &identity_gain, &zero_offset, max_value);

    let consistent_window_ratio = window_consistency_ratio(&all_estimates, training_gain_luma);
    let channel_consistency = per_channel_window_consistency(&all_estimates, &training_gain_rgb);
    let rgb_spread = max_gain_spread(&training_gain_rgb);
    let scalar_gain = [training_gain_luma; 3];
    let scalar_training_measurements = seam_measurements(
        &filtered_training_samples,
        &scalar_gain,
        &zero_offset,
        max_value,
    );
    let channel_training_measurements = seam_measurements(
        &filtered_training_samples,
        &training_gain_rgb,
        &zero_offset,
        max_value,
    );
    let use_per_channel = rgb_spread > SEAM_EXPOSURE_SCALAR_RGB_SPREAD
        && rgb_spread <= SEAM_EXPOSURE_MAX_RGB_SPREAD
        && channel_consistency >= SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO
        && channel_training_measurements.seam_score + SEAM_EXPOSURE_PER_CHANNEL_MARGIN
            < scalar_training_measurements.seam_score;
    let gain_candidate_rgb = if use_per_channel {
        training_gain_rgb
    } else {
        scalar_gain
    };
    let gain_candidate_luma = if use_per_channel {
        training_gain_luma
    } else {
        gain_candidate_rgb[0]
    };
    let gain_measurements = seam_measurements(
        &filtered_samples,
        &gain_candidate_rgb,
        &zero_offset,
        max_value,
    );
    let held_out_identity =
        seam_measurements(&held_out_samples, &identity_gain, &zero_offset, max_value);
    let held_out_gain = seam_measurements(
        &held_out_samples,
        &gain_candidate_rgb,
        &zero_offset,
        max_value,
    );
    let (gain_clipped_high, gain_clipped_low) =
        strip_clipping_ratios(strip_r, &gain_candidate_rgb, &zero_offset, max_value);

    let (affine_training_indices, affine_held_out_indices) = seam_window_split(&windows, true);
    let affine_training_samples = seam_samples_for_windows(&windows, &affine_training_indices);
    let affine_held_out_samples = seam_samples_for_windows(&windows, &affine_held_out_indices);
    let affine_held_out_estimates = seam_estimates_for_windows(&windows, &affine_held_out_indices);
    let affine_rgb: [(f64, f64); 3] = std::array::from_fn(|channel| {
        robust_affine_for_channel(&affine_training_samples, Some(channel), max_value)
    });
    let affine_luma = robust_affine_for_channel(&affine_training_samples, None, max_value);
    let affine_gain_rgb = std::array::from_fn(|channel| affine_rgb[channel].0);
    let affine_offset_rgb = std::array::from_fn(|channel| affine_rgb[channel].1);
    let affine_gain_luma = affine_luma.0;
    let affine_offset_luma = affine_luma.1;
    let affine_consistency = gain_offset_window_consistency(
        &affine_held_out_estimates,
        &affine_gain_rgb,
        &affine_offset_rgb,
        max_value,
    );
    let affine_measurements = seam_measurements(
        &filtered_samples,
        &affine_gain_rgb,
        &affine_offset_rgb,
        max_value,
    );
    let affine_held_out_identity = seam_measurements(
        &affine_held_out_samples,
        &identity_gain,
        &zero_offset,
        max_value,
    );
    let affine_held_out_gain = seam_measurements(
        &affine_held_out_samples,
        &gain_candidate_rgb,
        &zero_offset,
        max_value,
    );
    let affine_held_out = seam_measurements(
        &affine_held_out_samples,
        &affine_gain_rgb,
        &affine_offset_rgb,
        max_value,
    );
    let affine_general_held_out = seam_measurements(
        &held_out_samples,
        &affine_gain_rgb,
        &affine_offset_rgb,
        max_value,
    );
    let (affine_clipped_high, affine_clipped_low) =
        strip_clipping_ratios(strip_r, &affine_gain_rgb, &affine_offset_rgb, max_value);

    let spatial_fit = fit_spatial_gain(&windows, &training_indices);
    let spatial_held_out_fit = fit_spatial_gain(&windows, &held_out_indices);
    let spatial_training_rows = distinct_window_rows(&windows, &training_indices);
    let spatial_held_out_rows = distinct_window_rows(&windows, &held_out_indices);
    let spatial_consistency = spatial_window_consistency(&windows, &held_out_indices, &spatial_fit);
    let spatial_slope_agreement =
        spatial_slope_agreement_ratio(&spatial_fit, &spatial_held_out_fit);
    let spatial_top_gain = spatial_gain_at_normalized_y(
        &spatial_fit.center_gain_rgb,
        &spatial_fit.log_slope_y_rgb,
        -1.0,
    );
    let spatial_bottom_gain = spatial_gain_at_normalized_y(
        &spatial_fit.center_gain_rgb,
        &spatial_fit.log_slope_y_rgb,
        1.0,
    );
    let spatial_measurements = seam_measurements_spatial(
        &filtered_samples,
        &spatial_fit.center_gain_rgb,
        &zero_offset,
        &spatial_fit.log_slope_y_rgb,
        max_value,
    );
    let spatial_held_out = seam_measurements_spatial(
        &held_out_samples,
        &spatial_fit.center_gain_rgb,
        &zero_offset,
        &spatial_fit.log_slope_y_rgb,
        max_value,
    );
    let (spatial_clipped_high, spatial_clipped_low) = strip_clipping_ratios_spatial(
        strip_r,
        &spatial_fit.center_gain_rgb,
        &zero_offset,
        &spatial_fit.log_slope_y_rgb,
        max_value,
    );

    let spatial_affine_gain_seed = fit_spatial_gain(&windows, &affine_training_indices);
    let spatial_affine_held_out_gain_seed = fit_spatial_gain(&windows, &affine_held_out_indices);
    let spatial_affine_fit = fit_spatial_affine(
        &affine_training_samples,
        &spatial_affine_gain_seed,
        max_value,
    );
    let spatial_affine_held_out_fit = fit_spatial_affine(
        &affine_held_out_samples,
        &spatial_affine_held_out_gain_seed,
        max_value,
    );
    let spatial_affine_training_rows = distinct_window_rows(&windows, &affine_training_indices);
    let spatial_affine_held_out_rows = distinct_window_rows(&windows, &affine_held_out_indices);
    let spatial_affine_consistency = spatial_affine_window_consistency(
        &windows,
        &affine_held_out_indices,
        &spatial_affine_fit,
        max_value,
    );
    let spatial_affine_slope_agreement = spatial_affine_slope_agreement_ratio(
        &spatial_affine_fit,
        &spatial_affine_held_out_fit,
        max_value,
    );
    let spatial_affine_center_offset_delta = spatial_affine_center_offset_delta_normalized(
        &spatial_affine_fit,
        &spatial_affine_held_out_fit,
        max_value,
    );
    let spatial_affine_top_gain = spatial_affine_gain_at_y(&spatial_affine_fit, -1.0);
    let spatial_affine_bottom_gain = spatial_affine_gain_at_y(&spatial_affine_fit, 1.0);
    let spatial_affine_top_offset = spatial_affine_offset_at_y(&spatial_affine_fit, -1.0);
    let spatial_affine_bottom_offset = spatial_affine_offset_at_y(&spatial_affine_fit, 1.0);
    let spatial_affine_measurements = seam_measurements_spatial_affine(
        &filtered_samples,
        &spatial_affine_fit.center_gain_rgb,
        &spatial_affine_fit.center_offset_rgb,
        &spatial_affine_fit.log_gain_slope_y_rgb,
        &spatial_affine_fit.offset_slope_y_rgb,
        max_value,
    );
    let spatial_affine_held_out = seam_measurements_spatial_affine(
        &held_out_samples,
        &spatial_affine_fit.center_gain_rgb,
        &spatial_affine_fit.center_offset_rgb,
        &spatial_affine_fit.log_gain_slope_y_rgb,
        &spatial_affine_fit.offset_slope_y_rgb,
        max_value,
    );
    let (spatial_affine_clipped_high, spatial_affine_clipped_low) =
        strip_clipping_ratios_spatial_affine(
            strip_r,
            &spatial_affine_fit.center_gain_rgb,
            &spatial_affine_fit.center_offset_rgb,
            &spatial_affine_fit.log_gain_slope_y_rgb,
            &spatial_affine_fit.offset_slope_y_rgb,
            max_value,
        );

    let spatial_2d_gain_fit = fit_spatial_gain_2d(&windows, &training_indices);
    let spatial_2d_gain_held_out_fit = fit_spatial_gain_2d(&windows, &held_out_indices);
    let spatial_2d_training_rows = distinct_window_rows(&windows, &training_indices);
    let spatial_2d_held_out_rows = distinct_window_rows(&windows, &held_out_indices);
    let spatial_2d_training_columns = distinct_window_columns(&windows, &training_indices);
    let spatial_2d_held_out_columns = distinct_window_columns(&windows, &held_out_indices);
    let spatial_2d_gain_consistency =
        spatial_gain_2d_window_consistency(&windows, &held_out_indices, &spatial_2d_gain_fit);
    let (spatial_2d_gain_slope_agreement, spatial_2d_gain_horizontal_slope_agreement) =
        spatial_gain_2d_slope_agreement_ratios(&spatial_2d_gain_fit, &spatial_2d_gain_held_out_fit);
    let spatial_2d_gain_center_delta = spatial_center_gain_log_delta(
        &spatial_2d_gain_fit.center_gain_rgb,
        &spatial_2d_gain_held_out_fit.center_gain_rgb,
    );
    let spatial_2d_gain_corners = spatial_gain_2d_corners(&spatial_2d_gain_fit);
    let spatial_2d_gain_measurements = seam_measurements_spatial_affine_2d(
        &filtered_samples,
        &spatial_2d_gain_fit.center_gain_rgb,
        &zero_offset,
        &spatial_2d_gain_fit.log_slope_x_rgb,
        &spatial_2d_gain_fit.log_slope_y_rgb,
        &zero_offset,
        &zero_offset,
        max_value,
    );
    let spatial_2d_gain_held_out = seam_measurements_spatial_affine_2d(
        &held_out_samples,
        &spatial_2d_gain_fit.center_gain_rgb,
        &zero_offset,
        &spatial_2d_gain_fit.log_slope_x_rgb,
        &spatial_2d_gain_fit.log_slope_y_rgb,
        &zero_offset,
        &zero_offset,
        max_value,
    );
    let (spatial_2d_gain_clipped_high, spatial_2d_gain_clipped_low) =
        strip_clipping_ratios_spatial_affine_2d(
            strip_r,
            &spatial_2d_gain_fit.center_gain_rgb,
            &zero_offset,
            &spatial_2d_gain_fit.log_slope_x_rgb,
            &spatial_2d_gain_fit.log_slope_y_rgb,
            &zero_offset,
            &zero_offset,
            max_value,
        );

    let spatial_2d_affine_gain_seed = fit_spatial_gain_2d(&windows, &affine_training_indices);
    let spatial_2d_affine_held_out_gain_seed =
        fit_spatial_gain_2d(&windows, &affine_held_out_indices);
    let spatial_2d_affine_fit = fit_spatial_affine_2d(
        &affine_training_samples,
        &spatial_2d_affine_gain_seed,
        &spatial_affine_fit,
        max_value,
    );
    let spatial_2d_affine_held_out_fit = fit_spatial_affine_2d(
        &affine_held_out_samples,
        &spatial_2d_affine_held_out_gain_seed,
        &spatial_affine_held_out_fit,
        max_value,
    );
    let spatial_2d_affine_training_rows = distinct_window_rows(&windows, &affine_training_indices);
    let spatial_2d_affine_held_out_rows = distinct_window_rows(&windows, &affine_held_out_indices);
    let spatial_2d_affine_training_columns =
        distinct_window_columns(&windows, &affine_training_indices);
    let spatial_2d_affine_held_out_columns =
        distinct_window_columns(&windows, &affine_held_out_indices);
    let spatial_2d_affine_consistency = spatial_affine_2d_window_consistency(
        &windows,
        &affine_held_out_indices,
        &spatial_2d_affine_fit,
        max_value,
    );
    let (spatial_2d_affine_slope_agreement, spatial_2d_affine_horizontal_slope_agreement) =
        spatial_affine_2d_slope_agreement_ratios(
            &spatial_2d_affine_fit,
            &spatial_2d_affine_held_out_fit,
            max_value,
        );
    let spatial_2d_affine_center_gain_delta = spatial_center_gain_log_delta(
        &spatial_2d_affine_fit.center_gain_rgb,
        &spatial_2d_affine_held_out_fit.center_gain_rgb,
    );
    let spatial_2d_affine_center_offset_delta = spatial_affine_2d_center_offset_delta_normalized(
        &spatial_2d_affine_fit,
        &spatial_2d_affine_held_out_fit,
        max_value,
    );
    let spatial_2d_affine_gain_corners = spatial_affine_2d_gain_corners(&spatial_2d_affine_fit);
    let spatial_2d_affine_offset_corners = spatial_affine_2d_offset_corners(&spatial_2d_affine_fit);
    let spatial_2d_affine_measurements = seam_measurements_spatial_affine_2d(
        &filtered_samples,
        &spatial_2d_affine_fit.center_gain_rgb,
        &spatial_2d_affine_fit.center_offset_rgb,
        &spatial_2d_affine_fit.log_gain_slope_x_rgb,
        &spatial_2d_affine_fit.log_gain_slope_y_rgb,
        &spatial_2d_affine_fit.offset_slope_x_rgb,
        &spatial_2d_affine_fit.offset_slope_y_rgb,
        max_value,
    );
    let spatial_2d_affine_held_out = seam_measurements_spatial_affine_2d(
        &held_out_samples,
        &spatial_2d_affine_fit.center_gain_rgb,
        &spatial_2d_affine_fit.center_offset_rgb,
        &spatial_2d_affine_fit.log_gain_slope_x_rgb,
        &spatial_2d_affine_fit.log_gain_slope_y_rgb,
        &spatial_2d_affine_fit.offset_slope_x_rgb,
        &spatial_2d_affine_fit.offset_slope_y_rgb,
        max_value,
    );
    let (spatial_2d_affine_clipped_high, spatial_2d_affine_clipped_low) =
        strip_clipping_ratios_spatial_affine_2d(
            strip_r,
            &spatial_2d_affine_fit.center_gain_rgb,
            &spatial_2d_affine_fit.center_offset_rgb,
            &spatial_2d_affine_fit.log_gain_slope_x_rgb,
            &spatial_2d_affine_fit.log_gain_slope_y_rgb,
            &spatial_2d_affine_fit.offset_slope_x_rgb,
            &spatial_2d_affine_fit.offset_slope_y_rgb,
            max_value,
        );

    let spatial_quadratic_gain_fit = fit_spatial_quadratic_gain_2d(&windows, &training_indices);
    let spatial_quadratic_gain_held_out_fit =
        fit_spatial_quadratic_gain_2d(&windows, &held_out_indices);
    let spatial_quadratic_gain_field = spatial_quadratic_gain_to_field(&spatial_quadratic_gain_fit);
    let spatial_quadratic_gain_held_out_field =
        spatial_quadratic_gain_to_field(&spatial_quadratic_gain_held_out_fit);
    let spatial_quadratic_gain_consistency = spatial_quadratic_gain_window_consistency(
        &windows,
        &held_out_indices,
        &spatial_quadratic_gain_field,
    );
    let spatial_quadratic_gain_coefficient_agreement =
        quadratic_curvature_coefficient_agreement_ratio(
            &spatial_quadratic_gain_field,
            &spatial_quadratic_gain_held_out_field,
            max_value,
            false,
        );
    let (spatial_quadratic_gain_validation_delta, _) = spatial_field_validation_deltas(
        &spatial_quadratic_gain_field,
        &spatial_quadratic_gain_held_out_field,
        max_value,
    );
    let spatial_quadratic_gain_signal =
        spatial_field_gain_curvature_signal(&spatial_quadratic_gain_field);
    let spatial_quadratic_gain_bounds =
        spatial_field_grid_bounds(&spatial_quadratic_gain_field, max_value);
    let spatial_quadratic_gain_measurements = seam_measurements_photometric_field(
        &filtered_samples,
        &spatial_quadratic_gain_field,
        max_value,
    );
    let spatial_quadratic_gain_held_out = seam_measurements_photometric_field(
        &held_out_samples,
        &spatial_quadratic_gain_field,
        max_value,
    );
    let (spatial_quadratic_gain_clipped_high, spatial_quadratic_gain_clipped_low) =
        strip_clipping_ratios_photometric_field(strip_r, &spatial_quadratic_gain_field, max_value);

    // A gain-only quadratic residual below the absolute complexity margin cannot possibly
    // justify another six offset coefficients. Skip the expensive coupled solve in that case.
    let spatial_quadratic_affine_fit_performed = affine_training_indices.len()
        >= SEAM_EXPOSURE_MIN_QUADRATIC_WINDOWS_PER_SPLIT
        && affine_held_out_indices.len() >= SEAM_EXPOSURE_MIN_QUADRATIC_WINDOWS_PER_SPLIT
        && spatial_2d_affine_training_rows >= SEAM_EXPOSURE_MIN_QUADRATIC_ROWS_PER_SPLIT
        && spatial_2d_affine_held_out_rows >= SEAM_EXPOSURE_MIN_QUADRATIC_ROWS_PER_SPLIT
        && spatial_2d_affine_training_columns >= SEAM_EXPOSURE_MIN_QUADRATIC_COLUMNS_PER_SPLIT
        && spatial_2d_affine_held_out_columns >= SEAM_EXPOSURE_MIN_QUADRATIC_COLUMNS_PER_SPLIT
        && spatial_quadratic_gain_held_out.seam_score
            > SEAM_EXPOSURE_MIN_QUADRATIC_OVER_SIMPLER_IMPROVEMENT;
    let (spatial_quadratic_affine_fit, spatial_quadratic_affine_held_out_fit) =
        if spatial_quadratic_affine_fit_performed {
            let gain_seed = fit_spatial_quadratic_gain_2d(&windows, &affine_training_indices);
            let held_out_gain_seed =
                fit_spatial_quadratic_gain_2d(&windows, &affine_held_out_indices);
            (
                fit_spatial_quadratic_affine_2d(
                    &affine_training_samples,
                    &gain_seed,
                    &spatial_2d_affine_fit,
                    max_value,
                ),
                fit_spatial_quadratic_affine_2d(
                    &affine_held_out_samples,
                    &held_out_gain_seed,
                    &spatial_2d_affine_held_out_fit,
                    max_value,
                ),
            )
        } else {
            (
                spatial_quadratic_affine_from_planar(&spatial_2d_affine_fit),
                spatial_quadratic_affine_from_planar(&spatial_2d_affine_held_out_fit),
            )
        };
    let spatial_quadratic_affine_field =
        spatial_quadratic_affine_to_field(&spatial_quadratic_affine_fit);
    let spatial_quadratic_affine_held_out_field =
        spatial_quadratic_affine_to_field(&spatial_quadratic_affine_held_out_fit);
    let spatial_quadratic_affine_consistency = spatial_quadratic_affine_window_consistency(
        &windows,
        &affine_held_out_indices,
        &spatial_quadratic_affine_field,
        max_value,
    );
    let spatial_quadratic_affine_coefficient_agreement =
        quadratic_curvature_coefficient_agreement_ratio(
            &spatial_quadratic_affine_field,
            &spatial_quadratic_affine_held_out_field,
            max_value,
            true,
        );
    let (
        spatial_quadratic_affine_gain_validation_delta,
        spatial_quadratic_affine_offset_validation_delta,
    ) = spatial_field_validation_deltas(
        &spatial_quadratic_affine_field,
        &spatial_quadratic_affine_held_out_field,
        max_value,
    );
    let spatial_quadratic_affine_gain_signal =
        spatial_field_gain_curvature_signal(&spatial_quadratic_affine_field);
    let spatial_quadratic_affine_offset_signal = spatial_field_offset_curvature_signal_normalized(
        &spatial_quadratic_affine_field,
        max_value,
    );
    let spatial_quadratic_affine_signal =
        spatial_quadratic_affine_gain_signal.max(spatial_quadratic_affine_offset_signal);
    let spatial_quadratic_affine_bounds =
        spatial_field_grid_bounds(&spatial_quadratic_affine_field, max_value);
    let spatial_quadratic_affine_measurements = seam_measurements_photometric_field(
        &filtered_samples,
        &spatial_quadratic_affine_field,
        max_value,
    );
    let spatial_quadratic_affine_held_out = seam_measurements_photometric_field(
        &held_out_samples,
        &spatial_quadratic_affine_field,
        max_value,
    );
    let (spatial_quadratic_affine_clipped_high, spatial_quadratic_affine_clipped_low) =
        strip_clipping_ratios_photometric_field(
            strip_r,
            &spatial_quadratic_affine_field,
            max_value,
        );

    let common_rejection = if windows.len() < SEAM_EXPOSURE_MIN_WINDOWS {
        Some(format!(
            "insufficient reliable overlap windows: {} found, need at least {}",
            windows.len(),
            SEAM_EXPOSURE_MIN_WINDOWS
        ))
    } else if training_indices.len() < SEAM_EXPOSURE_MIN_HELD_OUT_WINDOWS
        || held_out_indices.len() < SEAM_EXPOSURE_MIN_HELD_OUT_WINDOWS
    {
        Some(format!(
            "insufficient spatially separate overlap windows for training/held-out validation: {}/{}",
            training_indices.len(),
            held_out_indices.len()
        ))
    } else if sample_count < SEAM_EXPOSURE_MIN_VALID_SAMPLES
        || valid_sample_ratio < SEAM_EXPOSURE_MIN_VALID_SAMPLE_RATIO
    {
        Some(format!(
            "insufficient reliable overlap samples: {} samples ({:.1}%), need at least {} samples and {:.1}%",
            sample_count,
            valid_sample_ratio * 100.0,
            SEAM_EXPOSURE_MIN_VALID_SAMPLES,
            SEAM_EXPOSURE_MIN_VALID_SAMPLE_RATIO * 100.0
        ))
    } else if held_out_samples.len() < SEAM_EXPOSURE_MIN_HELD_OUT_SAMPLES {
        Some(format!(
            "insufficient held-out overlap samples: {} found, need at least {}",
            held_out_samples.len(),
            SEAM_EXPOSURE_MIN_HELD_OUT_SAMPLES
        ))
    } else {
        None
    };

    let gain_rejection = if let Some(reason) = common_rejection.clone() {
        Some(reason)
    } else if held_out_identity.seam_score < SEAM_EXPOSURE_MIN_SCORE {
        Some(format!(
            "held-out seam exposure mismatch below threshold: score {:.3} < {:.3}",
            held_out_identity.seam_score, SEAM_EXPOSURE_MIN_SCORE
        ))
    } else if !gain_in_bounds(&gain_candidate_rgb) {
        Some(format!(
            "candidate gain out of bounds: rgb={:.3}/{:.3}/{:.3}, allowed {:.2}..{:.2}",
            gain_candidate_rgb[0],
            gain_candidate_rgb[1],
            gain_candidate_rgb[2],
            SEAM_EXPOSURE_MIN_GAIN,
            SEAM_EXPOSURE_MAX_GAIN
        ))
    } else if rgb_spread > SEAM_EXPOSURE_MAX_RGB_SPREAD
        && scalar_training_measurements.seam_score >= measurements_before.seam_score
    {
        Some(format!(
            "estimated RGB gains diverged suspiciously: spread {:.3} exceeds {:.3}",
            rgb_spread, SEAM_EXPOSURE_MAX_RGB_SPREAD
        ))
    } else if consistent_window_ratio < SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO {
        Some(format!(
            "overlap windows do not agree on exposure correction: {:.1}% consistent, need {:.1}%",
            consistent_window_ratio * 100.0,
            SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO * 100.0
        ))
    } else {
        let improvement = held_out_identity.seam_score - held_out_gain.seam_score;
        let required_improvement =
            SEAM_EXPOSURE_MIN_IMPROVEMENT.max(held_out_identity.seam_score * 0.25);
        if improvement < required_improvement {
            Some(format!(
                "gain-only correction did not improve held-out seam enough: improvement {:.3}, need {:.3}",
                improvement, required_improvement
            ))
        } else {
            let clip_increase = clipping_delta(
                &clipped_high_before,
                &gain_clipped_high,
                &clipped_low_before,
                &gain_clipped_low,
            );
            (clip_increase > SEAM_EXPOSURE_MAX_CLIP_INCREASE).then(|| {
                format!(
                    "gain-only correction would increase clipping by {:.4}, limit {:.4}",
                    clip_increase, SEAM_EXPOSURE_MAX_CLIP_INCREASE
                )
            })
        }
    };

    let affine_offset_max = affine_offset_rgb
        .iter()
        .map(|offset| offset.abs() / max_value)
        .fold(0.0f64, f64::max);
    let affine_identity_improvement =
        affine_held_out_identity.seam_score - affine_held_out.seam_score;
    let affine_gain_improvement = affine_held_out_gain.seam_score - affine_held_out.seam_score;
    let affine_required_identity_improvement =
        SEAM_EXPOSURE_MIN_IMPROVEMENT.max(affine_held_out_identity.seam_score * 0.25);
    let affine_required_gain_improvement = SEAM_EXPOSURE_MIN_AFFINE_OVER_GAIN_IMPROVEMENT
        .max(affine_held_out_gain.seam_score * SEAM_EXPOSURE_MIN_AFFINE_OVER_GAIN_FRACTION);
    let affine_clip_increase = clipping_delta(
        &clipped_high_before,
        &affine_clipped_high,
        &clipped_low_before,
        &affine_clipped_low,
    );
    let affine_rejection = if let Some(reason) = common_rejection.clone() {
        Some(reason)
    } else if affine_training_indices.len() < SEAM_EXPOSURE_MIN_AFFINE_WINDOWS_PER_SPLIT
        || affine_held_out_indices.len() < SEAM_EXPOSURE_MIN_AFFINE_WINDOWS_PER_SPLIT
    {
        Some(format!(
            "gain+offset model needs at least {} signal-rich windows in each split; found {}/{}",
            SEAM_EXPOSURE_MIN_AFFINE_WINDOWS_PER_SPLIT,
            affine_training_indices.len(),
            affine_held_out_indices.len()
        ))
    } else if affine_training_samples.len() < SEAM_EXPOSURE_MIN_HELD_OUT_SAMPLES
        || affine_held_out_samples.len() < SEAM_EXPOSURE_MIN_HELD_OUT_SAMPLES
    {
        Some(format!(
            "gain+offset model has insufficient training/held-out samples: {}/{}",
            affine_training_samples.len(),
            affine_held_out_samples.len()
        ))
    } else if affine_held_out_identity.seam_score < SEAM_EXPOSURE_MIN_SCORE {
        Some(format!(
            "gain+offset held-out mismatch below threshold: {:.3} < {:.3}",
            affine_held_out_identity.seam_score, SEAM_EXPOSURE_MIN_SCORE
        ))
    } else if !gain_in_bounds(&affine_gain_rgb) {
        Some(format!(
            "gain+offset candidate gain out of bounds: rgb={:.3}/{:.3}/{:.3}",
            affine_gain_rgb[0], affine_gain_rgb[1], affine_gain_rgb[2]
        ))
    } else if affine_offset_max > SEAM_EXPOSURE_MAX_OFFSET_NORMALIZED {
        Some(format!(
            "gain+offset candidate offset exceeds {:.1}% of sample range: {:.2}%",
            SEAM_EXPOSURE_MAX_OFFSET_NORMALIZED * 100.0,
            affine_offset_max * 100.0
        ))
    } else if affine_offset_max < SEAM_EXPOSURE_MIN_OFFSET_NORMALIZED {
        Some(format!(
            "gain+offset candidate is unnecessary: largest offset {:.2}% < {:.2}% of sample range",
            affine_offset_max * 100.0,
            SEAM_EXPOSURE_MIN_OFFSET_NORMALIZED * 100.0
        ))
    } else if affine_consistency < SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO {
        Some(format!(
            "held-out windows do not agree on gain+offset correction: {:.1}% consistent, need {:.1}%",
            affine_consistency * 100.0,
            SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO * 100.0
        ))
    } else if affine_identity_improvement < affine_required_identity_improvement {
        Some(format!(
            "gain+offset correction did not improve held-out identity enough: {:.3}, need {:.3}",
            affine_identity_improvement, affine_required_identity_improvement
        ))
    } else if affine_gain_improvement < affine_required_gain_improvement {
        Some(format!(
            "gain+offset correction did not beat gain-only on held-out windows: {:.3}, need {:.3}",
            affine_gain_improvement, affine_required_gain_improvement
        ))
    } else if affine_clip_increase > SEAM_EXPOSURE_MAX_CLIP_INCREASE {
        Some(format!(
            "gain+offset correction would increase clipping by {:.4}, limit {:.4}",
            affine_clip_increase, SEAM_EXPOSURE_MAX_CLIP_INCREASE
        ))
    } else {
        None
    };

    let mut best_constant_model = "identity";
    let mut best_constant_held_out_score = held_out_identity.seam_score;
    if gain_rejection.is_none() && held_out_gain.seam_score < best_constant_held_out_score {
        best_constant_model = "gain-only";
        best_constant_held_out_score = held_out_gain.seam_score;
    }
    if affine_rejection.is_none()
        && affine_general_held_out.seam_score < best_constant_held_out_score
    {
        best_constant_model = "gain+offset";
        best_constant_held_out_score = affine_general_held_out.seam_score;
    }
    let spatial_identity_improvement = held_out_identity.seam_score - spatial_held_out.seam_score;
    let spatial_best_constant_improvement =
        best_constant_held_out_score - spatial_held_out.seam_score;
    let spatial_required_identity_improvement =
        SEAM_EXPOSURE_MIN_IMPROVEMENT.max(held_out_identity.seam_score * 0.25);
    let spatial_required_constant_improvement = SEAM_EXPOSURE_MIN_SPATIAL_OVER_CONSTANT_IMPROVEMENT
        .max(best_constant_held_out_score * SEAM_EXPOSURE_MIN_SPATIAL_OVER_CONSTANT_FRACTION);
    let spatial_clip_increase = clipping_delta(
        &clipped_high_before,
        &spatial_clipped_high,
        &clipped_low_before,
        &spatial_clipped_low,
    );
    let spatial_signal_ratio = spatial_endpoint_ratio(&spatial_fit.log_slope_y_rgb);
    let spatial_rejection = if let Some(reason) = common_rejection.clone() {
        Some(reason)
    } else if spatial_training_rows < SEAM_EXPOSURE_MIN_SPATIAL_ROWS_PER_SPLIT
        || spatial_held_out_rows < SEAM_EXPOSURE_MIN_SPATIAL_ROWS_PER_SPLIT
    {
        Some(format!(
            "vertical gain field needs at least {} distinct rows in each split; found {}/{}",
            SEAM_EXPOSURE_MIN_SPATIAL_ROWS_PER_SPLIT, spatial_training_rows, spatial_held_out_rows
        ))
    } else if held_out_identity.seam_score < SEAM_EXPOSURE_MIN_SCORE {
        Some(format!(
            "vertical gain field held-out mismatch below threshold: {:.3} < {:.3}",
            held_out_identity.seam_score, SEAM_EXPOSURE_MIN_SCORE
        ))
    } else if spatial_signal_ratio < SEAM_EXPOSURE_MIN_SPATIAL_ENDPOINT_RATIO {
        Some(format!(
            "vertical gain field is unnecessary: endpoint ratio {:.3} < {:.3}",
            spatial_signal_ratio, SEAM_EXPOSURE_MIN_SPATIAL_ENDPOINT_RATIO
        ))
    } else if !gain_in_bounds(&spatial_top_gain) || !gain_in_bounds(&spatial_bottom_gain) {
        Some(format!(
            "vertical gain field endpoints are out of bounds: top {:.3}/{:.3}/{:.3}, bottom {:.3}/{:.3}/{:.3}",
            spatial_top_gain[0],
            spatial_top_gain[1],
            spatial_top_gain[2],
            spatial_bottom_gain[0],
            spatial_bottom_gain[1],
            spatial_bottom_gain[2]
        ))
    } else if max_gain_spread(&spatial_top_gain) > SEAM_EXPOSURE_MAX_RGB_SPREAD
        || max_gain_spread(&spatial_bottom_gain) > SEAM_EXPOSURE_MAX_RGB_SPREAD
    {
        Some(format!(
            "vertical gain field has suspicious endpoint RGB spread: {:.3}/{:.3}, limit {:.3}",
            max_gain_spread(&spatial_top_gain),
            max_gain_spread(&spatial_bottom_gain),
            SEAM_EXPOSURE_MAX_RGB_SPREAD
        ))
    } else if spatial_consistency < SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO {
        Some(format!(
            "held-out windows do not agree on vertical gain field: {:.1}% consistent, need {:.1}%",
            spatial_consistency * 100.0,
            SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO * 100.0
        ))
    } else if spatial_slope_agreement < SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO {
        Some(format!(
            "training and held-out vertical gain slopes disagree: {:.1}% agreement, need {:.1}%",
            spatial_slope_agreement * 100.0,
            SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO * 100.0
        ))
    } else if spatial_identity_improvement < spatial_required_identity_improvement {
        Some(format!(
            "vertical gain field did not improve held-out identity enough: {:.3}, need {:.3}",
            spatial_identity_improvement, spatial_required_identity_improvement
        ))
    } else if spatial_best_constant_improvement < spatial_required_constant_improvement {
        Some(format!(
            "vertical gain field did not beat the accepted {} model on held-out windows: {:.3}, need {:.3}",
            best_constant_model,
            spatial_best_constant_improvement,
            spatial_required_constant_improvement
        ))
    } else if spatial_clip_increase > SEAM_EXPOSURE_MAX_CLIP_INCREASE {
        Some(format!(
            "vertical gain field would increase clipping by {:.4}, limit {:.4}",
            spatial_clip_increase, SEAM_EXPOSURE_MAX_CLIP_INCREASE
        ))
    } else {
        None
    };

    let mut best_simpler_model = "identity";
    let mut best_simpler_held_out_score = held_out_identity.seam_score;
    if gain_rejection.is_none() && held_out_gain.seam_score < best_simpler_held_out_score {
        best_simpler_model = "gain-only";
        best_simpler_held_out_score = held_out_gain.seam_score;
    }
    if affine_rejection.is_none()
        && affine_general_held_out.seam_score < best_simpler_held_out_score
    {
        best_simpler_model = "gain+offset";
        best_simpler_held_out_score = affine_general_held_out.seam_score;
    }
    if spatial_rejection.is_none() && spatial_held_out.seam_score < best_simpler_held_out_score {
        best_simpler_model = "vertical gain";
        best_simpler_held_out_score = spatial_held_out.seam_score;
    }
    let spatial_affine_identity_improvement =
        held_out_identity.seam_score - spatial_affine_held_out.seam_score;
    let spatial_affine_simpler_improvement =
        best_simpler_held_out_score - spatial_affine_held_out.seam_score;
    let spatial_affine_required_identity_improvement =
        SEAM_EXPOSURE_MIN_IMPROVEMENT.max(held_out_identity.seam_score * 0.25);
    let spatial_affine_required_simpler_improvement =
        SEAM_EXPOSURE_MIN_SPATIAL_AFFINE_OVER_SIMPLER_IMPROVEMENT.max(
            best_simpler_held_out_score * SEAM_EXPOSURE_MIN_SPATIAL_AFFINE_OVER_SIMPLER_FRACTION,
        );
    let spatial_affine_gain_signal =
        spatial_endpoint_ratio(&spatial_affine_fit.log_gain_slope_y_rgb);
    let spatial_affine_offset_max =
        maximum_normalized_magnitude(&spatial_affine_top_offset, max_value).max(
            maximum_normalized_magnitude(&spatial_affine_bottom_offset, max_value),
        );
    let spatial_affine_offset_delta = maximum_normalized_endpoint_delta(
        &spatial_affine_top_offset,
        &spatial_affine_bottom_offset,
        max_value,
    );
    let spatial_affine_clip_increase = clipping_delta(
        &clipped_high_before,
        &spatial_affine_clipped_high,
        &clipped_low_before,
        &spatial_affine_clipped_low,
    );
    let spatial_affine_rejection = if let Some(reason) = common_rejection.clone() {
        Some(reason)
    } else if affine_training_indices.len() < SEAM_EXPOSURE_MIN_AFFINE_WINDOWS_PER_SPLIT
        || affine_held_out_indices.len() < SEAM_EXPOSURE_MIN_AFFINE_WINDOWS_PER_SPLIT
    {
        Some(format!(
            "vertical gain+offset field needs at least {} signal-rich windows in each split; found {}/{}",
            SEAM_EXPOSURE_MIN_AFFINE_WINDOWS_PER_SPLIT,
            affine_training_indices.len(),
            affine_held_out_indices.len()
        ))
    } else if spatial_affine_training_rows < SEAM_EXPOSURE_MIN_SPATIAL_AFFINE_ROWS_PER_SPLIT
        || spatial_affine_held_out_rows < SEAM_EXPOSURE_MIN_SPATIAL_AFFINE_ROWS_PER_SPLIT
    {
        Some(format!(
            "vertical gain+offset field needs at least {} distinct signal-rich rows in each split; found {}/{}",
            SEAM_EXPOSURE_MIN_SPATIAL_AFFINE_ROWS_PER_SPLIT,
            spatial_affine_training_rows,
            spatial_affine_held_out_rows
        ))
    } else if held_out_identity.seam_score < SEAM_EXPOSURE_MIN_SCORE {
        Some(format!(
            "vertical gain+offset field held-out mismatch below threshold: {:.3} < {:.3}",
            held_out_identity.seam_score, SEAM_EXPOSURE_MIN_SCORE
        ))
    } else if spatial_affine_gain_signal < SEAM_EXPOSURE_MIN_SPATIAL_ENDPOINT_RATIO
        && spatial_affine_offset_delta < SEAM_EXPOSURE_MIN_SPATIAL_OFFSET_ENDPOINT_DELTA_NORMALIZED
    {
        Some(format!(
            "vertical gain+offset field is unnecessary: gain endpoint ratio {:.3} and offset endpoint delta {:.2}% are below {:.3}/{:.2}%",
            spatial_affine_gain_signal,
            spatial_affine_offset_delta * 100.0,
            SEAM_EXPOSURE_MIN_SPATIAL_ENDPOINT_RATIO,
            SEAM_EXPOSURE_MIN_SPATIAL_OFFSET_ENDPOINT_DELTA_NORMALIZED * 100.0
        ))
    } else if !gain_in_bounds(&spatial_affine_top_gain)
        || !gain_in_bounds(&spatial_affine_bottom_gain)
    {
        Some(format!(
            "vertical gain+offset field gain endpoints are out of bounds: top {:.3}/{:.3}/{:.3}, bottom {:.3}/{:.3}/{:.3}",
            spatial_affine_top_gain[0],
            spatial_affine_top_gain[1],
            spatial_affine_top_gain[2],
            spatial_affine_bottom_gain[0],
            spatial_affine_bottom_gain[1],
            spatial_affine_bottom_gain[2]
        ))
    } else if spatial_affine_offset_max > SEAM_EXPOSURE_MAX_OFFSET_NORMALIZED {
        Some(format!(
            "vertical gain+offset field offset endpoints exceed {:.1}% of sample range: {:.2}%",
            SEAM_EXPOSURE_MAX_OFFSET_NORMALIZED * 100.0,
            spatial_affine_offset_max * 100.0
        ))
    } else if spatial_affine_offset_max < SEAM_EXPOSURE_MIN_OFFSET_NORMALIZED {
        Some(format!(
            "vertical gain+offset field additive term is unnecessary: {:.2}% < {:.2}% of sample range",
            spatial_affine_offset_max * 100.0,
            SEAM_EXPOSURE_MIN_OFFSET_NORMALIZED * 100.0
        ))
    } else if max_gain_spread(&spatial_affine_top_gain) > SEAM_EXPOSURE_MAX_RGB_SPREAD
        || max_gain_spread(&spatial_affine_bottom_gain) > SEAM_EXPOSURE_MAX_RGB_SPREAD
    {
        Some(format!(
            "vertical gain+offset field has suspicious endpoint RGB spread: {:.3}/{:.3}, limit {:.3}",
            max_gain_spread(&spatial_affine_top_gain),
            max_gain_spread(&spatial_affine_bottom_gain),
            SEAM_EXPOSURE_MAX_RGB_SPREAD
        ))
    } else if spatial_affine_consistency < SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO {
        Some(format!(
            "held-out windows do not agree on vertical gain+offset field: {:.1}% consistent, need {:.1}%",
            spatial_affine_consistency * 100.0,
            SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO * 100.0
        ))
    } else if spatial_affine_slope_agreement < SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO {
        Some(format!(
            "training and held-out vertical gain+offset slopes disagree: {:.1}% agreement, need {:.1}%",
            spatial_affine_slope_agreement * 100.0,
            SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO * 100.0
        ))
    } else if spatial_affine_center_offset_delta
        > SEAM_EXPOSURE_MAX_SPATIAL_OFFSET_CENTER_DELTA_NORMALIZED
    {
        Some(format!(
            "training and held-out vertical field center offsets differ by {:.2}% of sample range, limit {:.2}%",
            spatial_affine_center_offset_delta * 100.0,
            SEAM_EXPOSURE_MAX_SPATIAL_OFFSET_CENTER_DELTA_NORMALIZED * 100.0
        ))
    } else if spatial_affine_identity_improvement < spatial_affine_required_identity_improvement {
        Some(format!(
            "vertical gain+offset field did not improve held-out identity enough: {:.3}, need {:.3}",
            spatial_affine_identity_improvement,
            spatial_affine_required_identity_improvement
        ))
    } else if spatial_affine_simpler_improvement < spatial_affine_required_simpler_improvement {
        Some(format!(
            "vertical gain+offset field did not beat the accepted {} model on held-out windows: {:.3}, need {:.3}",
            best_simpler_model,
            spatial_affine_simpler_improvement,
            spatial_affine_required_simpler_improvement
        ))
    } else if spatial_affine_clip_increase > SEAM_EXPOSURE_MAX_CLIP_INCREASE {
        Some(format!(
            "vertical gain+offset field would increase clipping by {:.4}, limit {:.4}",
            spatial_affine_clip_increase, SEAM_EXPOSURE_MAX_CLIP_INCREASE
        ))
    } else {
        None
    };

    let mut spatial_2d_gain_best_simpler_model = best_simpler_model;
    let mut spatial_2d_gain_best_simpler_score = best_simpler_held_out_score;
    if spatial_affine_rejection.is_none()
        && spatial_affine_held_out.seam_score < spatial_2d_gain_best_simpler_score
    {
        spatial_2d_gain_best_simpler_model = "vertical gain+offset";
        spatial_2d_gain_best_simpler_score = spatial_affine_held_out.seam_score;
    }
    let spatial_2d_gain_identity_improvement =
        held_out_identity.seam_score - spatial_2d_gain_held_out.seam_score;
    let spatial_2d_gain_simpler_improvement =
        spatial_2d_gain_best_simpler_score - spatial_2d_gain_held_out.seam_score;
    let spatial_2d_gain_required_identity_improvement =
        SEAM_EXPOSURE_MIN_IMPROVEMENT.max(held_out_identity.seam_score * 0.25);
    let spatial_2d_gain_required_simpler_improvement =
        SEAM_EXPOSURE_MIN_SPATIAL_AFFINE_OVER_SIMPLER_IMPROVEMENT.max(
            spatial_2d_gain_best_simpler_score
                * SEAM_EXPOSURE_MIN_SPATIAL_AFFINE_OVER_SIMPLER_FRACTION,
        );
    let spatial_2d_gain_horizontal_signal =
        spatial_endpoint_ratio(&spatial_2d_gain_fit.log_slope_x_rgb);
    let spatial_2d_gain_corners_in_bounds = spatial_2d_gain_corners.iter().all(gain_in_bounds);
    let spatial_2d_gain_max_corner_spread = spatial_2d_gain_corners
        .iter()
        .map(max_gain_spread)
        .fold(1.0f64, f64::max);
    let spatial_2d_gain_clip_increase = clipping_delta(
        &clipped_high_before,
        &spatial_2d_gain_clipped_high,
        &clipped_low_before,
        &spatial_2d_gain_clipped_low,
    );
    let spatial_2d_gain_rejection = if let Some(reason) = common_rejection.clone() {
        Some(reason)
    } else if spatial_2d_training_rows < SEAM_EXPOSURE_MIN_SPATIAL_2D_ROWS_PER_SPLIT
        || spatial_2d_held_out_rows < SEAM_EXPOSURE_MIN_SPATIAL_2D_ROWS_PER_SPLIT
        || spatial_2d_training_columns < SEAM_EXPOSURE_MIN_SPATIAL_2D_COLUMNS_PER_SPLIT
        || spatial_2d_held_out_columns < SEAM_EXPOSURE_MIN_SPATIAL_2D_COLUMNS_PER_SPLIT
    {
        Some(format!(
            "2D gain field needs at least {} rows and {} columns in each split; found rows {}/{}, columns {}/{}",
            SEAM_EXPOSURE_MIN_SPATIAL_2D_ROWS_PER_SPLIT,
            SEAM_EXPOSURE_MIN_SPATIAL_2D_COLUMNS_PER_SPLIT,
            spatial_2d_training_rows,
            spatial_2d_held_out_rows,
            spatial_2d_training_columns,
            spatial_2d_held_out_columns
        ))
    } else if held_out_identity.seam_score < SEAM_EXPOSURE_MIN_SCORE {
        Some(format!(
            "2D gain field held-out mismatch below threshold: {:.3} < {:.3}",
            held_out_identity.seam_score, SEAM_EXPOSURE_MIN_SCORE
        ))
    } else if spatial_2d_gain_horizontal_signal < SEAM_EXPOSURE_MIN_SPATIAL_ENDPOINT_RATIO {
        Some(format!(
            "2D gain field is unnecessary: horizontal endpoint ratio {:.3} < {:.3}",
            spatial_2d_gain_horizontal_signal, SEAM_EXPOSURE_MIN_SPATIAL_ENDPOINT_RATIO
        ))
    } else if !spatial_2d_gain_corners_in_bounds {
        Some(format!(
            "2D gain field has an out-of-bounds corner; allowed {:.2}..{:.2}",
            SEAM_EXPOSURE_MIN_GAIN, SEAM_EXPOSURE_MAX_GAIN
        ))
    } else if spatial_2d_gain_max_corner_spread > SEAM_EXPOSURE_MAX_RGB_SPREAD {
        Some(format!(
            "2D gain field corner RGB spread {:.3} exceeds {:.3}",
            spatial_2d_gain_max_corner_spread, SEAM_EXPOSURE_MAX_RGB_SPREAD
        ))
    } else if spatial_2d_gain_consistency < SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO {
        Some(format!(
            "held-out windows do not agree on 2D gain field: {:.1}% consistent, need {:.1}%",
            spatial_2d_gain_consistency * 100.0,
            SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO * 100.0
        ))
    } else if spatial_2d_gain_slope_agreement < SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO
        || spatial_2d_gain_horizontal_slope_agreement
            < SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO
    {
        Some(format!(
            "training and held-out 2D gain slopes disagree: overall {:.1}%, horizontal {:.1}%, need {:.1}%",
            spatial_2d_gain_slope_agreement * 100.0,
            spatial_2d_gain_horizontal_slope_agreement * 100.0,
            SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO * 100.0
        ))
    } else if spatial_2d_gain_center_delta > SEAM_EXPOSURE_MAX_SPATIAL_CENTER_GAIN_LOG_DELTA {
        Some(format!(
            "training and held-out 2D gain centers differ by {:.4} log units, limit {:.4}",
            spatial_2d_gain_center_delta, SEAM_EXPOSURE_MAX_SPATIAL_CENTER_GAIN_LOG_DELTA
        ))
    } else if spatial_2d_gain_identity_improvement < spatial_2d_gain_required_identity_improvement {
        Some(format!(
            "2D gain field did not improve held-out identity enough: {:.3}, need {:.3}",
            spatial_2d_gain_identity_improvement, spatial_2d_gain_required_identity_improvement
        ))
    } else if spatial_2d_gain_simpler_improvement < spatial_2d_gain_required_simpler_improvement {
        Some(format!(
            "2D gain field did not beat accepted {} on held-out windows: {:.3}, need {:.3}",
            spatial_2d_gain_best_simpler_model,
            spatial_2d_gain_simpler_improvement,
            spatial_2d_gain_required_simpler_improvement
        ))
    } else if spatial_2d_gain_clip_increase > SEAM_EXPOSURE_MAX_CLIP_INCREASE {
        Some(format!(
            "2D gain field would increase clipping by {:.4}, limit {:.4}",
            spatial_2d_gain_clip_increase, SEAM_EXPOSURE_MAX_CLIP_INCREASE
        ))
    } else {
        None
    };

    let mut spatial_2d_affine_best_simpler_model = spatial_2d_gain_best_simpler_model;
    let mut spatial_2d_affine_best_simpler_score = spatial_2d_gain_best_simpler_score;
    if spatial_2d_gain_rejection.is_none()
        && spatial_2d_gain_held_out.seam_score < spatial_2d_affine_best_simpler_score
    {
        spatial_2d_affine_best_simpler_model = "2D gain";
        spatial_2d_affine_best_simpler_score = spatial_2d_gain_held_out.seam_score;
    }
    let spatial_2d_affine_identity_improvement =
        held_out_identity.seam_score - spatial_2d_affine_held_out.seam_score;
    let spatial_2d_affine_simpler_improvement =
        spatial_2d_affine_best_simpler_score - spatial_2d_affine_held_out.seam_score;
    let spatial_2d_affine_required_identity_improvement =
        SEAM_EXPOSURE_MIN_IMPROVEMENT.max(held_out_identity.seam_score * 0.25);
    let spatial_2d_affine_required_simpler_improvement =
        SEAM_EXPOSURE_MIN_SPATIAL_AFFINE_OVER_SIMPLER_IMPROVEMENT.max(
            spatial_2d_affine_best_simpler_score
                * SEAM_EXPOSURE_MIN_SPATIAL_AFFINE_OVER_SIMPLER_FRACTION,
        );
    let spatial_2d_affine_horizontal_gain_signal =
        spatial_endpoint_ratio(&spatial_2d_affine_fit.log_gain_slope_x_rgb);
    let spatial_2d_affine_horizontal_offset_delta = spatial_2d_affine_fit
        .offset_slope_x_rgb
        .iter()
        .map(|slope| 2.0 * slope.abs() / max_value)
        .fold(0.0f64, f64::max);
    let spatial_2d_affine_offset_max = spatial_2d_affine_offset_corners
        .iter()
        .map(|corner| maximum_normalized_magnitude(corner, max_value))
        .fold(0.0f64, f64::max);
    let spatial_2d_affine_corners_in_bounds =
        spatial_2d_affine_gain_corners.iter().all(gain_in_bounds);
    let spatial_2d_affine_max_corner_spread = spatial_2d_affine_gain_corners
        .iter()
        .map(max_gain_spread)
        .fold(1.0f64, f64::max);
    let spatial_2d_affine_clip_increase = clipping_delta(
        &clipped_high_before,
        &spatial_2d_affine_clipped_high,
        &clipped_low_before,
        &spatial_2d_affine_clipped_low,
    );
    let spatial_2d_affine_rejection = if let Some(reason) = common_rejection.clone() {
        Some(reason)
    } else if affine_training_indices.len() < SEAM_EXPOSURE_MIN_AFFINE_WINDOWS_PER_SPLIT
        || affine_held_out_indices.len() < SEAM_EXPOSURE_MIN_AFFINE_WINDOWS_PER_SPLIT
    {
        Some(format!(
            "2D gain+offset field needs at least {} signal-rich windows in each split; found {}/{}",
            SEAM_EXPOSURE_MIN_AFFINE_WINDOWS_PER_SPLIT,
            affine_training_indices.len(),
            affine_held_out_indices.len()
        ))
    } else if spatial_2d_affine_training_rows < SEAM_EXPOSURE_MIN_SPATIAL_2D_ROWS_PER_SPLIT
        || spatial_2d_affine_held_out_rows < SEAM_EXPOSURE_MIN_SPATIAL_2D_ROWS_PER_SPLIT
        || spatial_2d_affine_training_columns < SEAM_EXPOSURE_MIN_SPATIAL_2D_COLUMNS_PER_SPLIT
        || spatial_2d_affine_held_out_columns < SEAM_EXPOSURE_MIN_SPATIAL_2D_COLUMNS_PER_SPLIT
    {
        Some(format!(
            "2D gain+offset field needs at least {} rows and {} columns in each signal-rich split; found rows {}/{}, columns {}/{}",
            SEAM_EXPOSURE_MIN_SPATIAL_2D_ROWS_PER_SPLIT,
            SEAM_EXPOSURE_MIN_SPATIAL_2D_COLUMNS_PER_SPLIT,
            spatial_2d_affine_training_rows,
            spatial_2d_affine_held_out_rows,
            spatial_2d_affine_training_columns,
            spatial_2d_affine_held_out_columns
        ))
    } else if held_out_identity.seam_score < SEAM_EXPOSURE_MIN_SCORE {
        Some(format!(
            "2D gain+offset field held-out mismatch below threshold: {:.3} < {:.3}",
            held_out_identity.seam_score, SEAM_EXPOSURE_MIN_SCORE
        ))
    } else if spatial_2d_affine_horizontal_gain_signal < SEAM_EXPOSURE_MIN_SPATIAL_ENDPOINT_RATIO
        && spatial_2d_affine_horizontal_offset_delta
            < SEAM_EXPOSURE_MIN_SPATIAL_OFFSET_ENDPOINT_DELTA_NORMALIZED
    {
        Some(format!(
            "2D gain+offset field is unnecessary horizontally: gain endpoint ratio {:.3}, offset delta {:.2}%",
            spatial_2d_affine_horizontal_gain_signal,
            spatial_2d_affine_horizontal_offset_delta * 100.0
        ))
    } else if !spatial_2d_affine_corners_in_bounds {
        Some(format!(
            "2D gain+offset field has an out-of-bounds gain corner; allowed {:.2}..{:.2}",
            SEAM_EXPOSURE_MIN_GAIN, SEAM_EXPOSURE_MAX_GAIN
        ))
    } else if spatial_2d_affine_max_corner_spread > SEAM_EXPOSURE_MAX_RGB_SPREAD {
        Some(format!(
            "2D gain+offset field corner RGB spread {:.3} exceeds {:.3}",
            spatial_2d_affine_max_corner_spread, SEAM_EXPOSURE_MAX_RGB_SPREAD
        ))
    } else if spatial_2d_affine_offset_max > SEAM_EXPOSURE_MAX_OFFSET_NORMALIZED {
        Some(format!(
            "2D gain+offset field offset corners exceed {:.1}% of sample range: {:.2}%",
            SEAM_EXPOSURE_MAX_OFFSET_NORMALIZED * 100.0,
            spatial_2d_affine_offset_max * 100.0
        ))
    } else if spatial_2d_affine_offset_max < SEAM_EXPOSURE_MIN_OFFSET_NORMALIZED {
        Some(format!(
            "2D gain+offset additive term is unnecessary: {:.2}% < {:.2}% of sample range",
            spatial_2d_affine_offset_max * 100.0,
            SEAM_EXPOSURE_MIN_OFFSET_NORMALIZED * 100.0
        ))
    } else if spatial_2d_affine_consistency < SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO {
        Some(format!(
            "held-out windows do not agree on 2D gain+offset field: {:.1}% consistent, need {:.1}%",
            spatial_2d_affine_consistency * 100.0,
            SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO * 100.0
        ))
    } else if spatial_2d_affine_slope_agreement < SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO
        || spatial_2d_affine_horizontal_slope_agreement
            < SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO
    {
        Some(format!(
            "training and held-out 2D gain+offset slopes disagree: overall {:.1}%, horizontal {:.1}%, need {:.1}%",
            spatial_2d_affine_slope_agreement * 100.0,
            spatial_2d_affine_horizontal_slope_agreement * 100.0,
            SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO * 100.0
        ))
    } else if spatial_2d_affine_center_gain_delta > SEAM_EXPOSURE_MAX_SPATIAL_CENTER_GAIN_LOG_DELTA
    {
        Some(format!(
            "training and held-out 2D gain+offset centers differ by {:.4} log units, limit {:.4}",
            spatial_2d_affine_center_gain_delta, SEAM_EXPOSURE_MAX_SPATIAL_CENTER_GAIN_LOG_DELTA
        ))
    } else if spatial_2d_affine_center_offset_delta
        > SEAM_EXPOSURE_MAX_SPATIAL_OFFSET_CENTER_DELTA_NORMALIZED
    {
        Some(format!(
            "training and held-out 2D center offsets differ by {:.2}% of sample range, limit {:.2}%",
            spatial_2d_affine_center_offset_delta * 100.0,
            SEAM_EXPOSURE_MAX_SPATIAL_OFFSET_CENTER_DELTA_NORMALIZED * 100.0
        ))
    } else if spatial_2d_affine_identity_improvement
        < spatial_2d_affine_required_identity_improvement
    {
        Some(format!(
            "2D gain+offset field did not improve held-out identity enough: {:.3}, need {:.3}",
            spatial_2d_affine_identity_improvement, spatial_2d_affine_required_identity_improvement
        ))
    } else if spatial_2d_affine_simpler_improvement < spatial_2d_affine_required_simpler_improvement
    {
        Some(format!(
            "2D gain+offset field did not beat accepted {} on held-out windows: {:.3}, need {:.3}",
            spatial_2d_affine_best_simpler_model,
            spatial_2d_affine_simpler_improvement,
            spatial_2d_affine_required_simpler_improvement
        ))
    } else if spatial_2d_affine_clip_increase > SEAM_EXPOSURE_MAX_CLIP_INCREASE {
        Some(format!(
            "2D gain+offset field would increase clipping by {:.4}, limit {:.4}",
            spatial_2d_affine_clip_increase, SEAM_EXPOSURE_MAX_CLIP_INCREASE
        ))
    } else {
        None
    };

    let mut spatial_quadratic_gain_best_simpler_model = spatial_2d_affine_best_simpler_model;
    let mut spatial_quadratic_gain_best_simpler_score = spatial_2d_affine_best_simpler_score;
    if spatial_2d_affine_rejection.is_none()
        && spatial_2d_affine_held_out.seam_score < spatial_quadratic_gain_best_simpler_score
    {
        spatial_quadratic_gain_best_simpler_model = "2D gain+offset";
        spatial_quadratic_gain_best_simpler_score = spatial_2d_affine_held_out.seam_score;
    }
    let spatial_quadratic_gain_identity_improvement =
        held_out_identity.seam_score - spatial_quadratic_gain_held_out.seam_score;
    let spatial_quadratic_gain_simpler_improvement =
        spatial_quadratic_gain_best_simpler_score - spatial_quadratic_gain_held_out.seam_score;
    let spatial_quadratic_gain_required_identity_improvement =
        SEAM_EXPOSURE_MIN_IMPROVEMENT.max(held_out_identity.seam_score * 0.25);
    let spatial_quadratic_gain_required_simpler_improvement =
        SEAM_EXPOSURE_MIN_QUADRATIC_OVER_SIMPLER_IMPROVEMENT.max(
            spatial_quadratic_gain_best_simpler_score
                * SEAM_EXPOSURE_MIN_QUADRATIC_OVER_SIMPLER_FRACTION,
        );
    let spatial_quadratic_gain_grid_in_bounds = (0..3).all(|channel| {
        spatial_quadratic_gain_bounds.gain_min_rgb[channel] >= SEAM_EXPOSURE_MIN_GAIN
            && spatial_quadratic_gain_bounds.gain_max_rgb[channel] <= SEAM_EXPOSURE_MAX_GAIN
    });
    let spatial_quadratic_gain_clip_increase = clipping_delta(
        &clipped_high_before,
        &spatial_quadratic_gain_clipped_high,
        &clipped_low_before,
        &spatial_quadratic_gain_clipped_low,
    );
    let spatial_quadratic_gain_rejection = if let Some(reason) = common_rejection.clone() {
        Some(reason)
    } else if training_indices.len() < SEAM_EXPOSURE_MIN_QUADRATIC_WINDOWS_PER_SPLIT
        || held_out_indices.len() < SEAM_EXPOSURE_MIN_QUADRATIC_WINDOWS_PER_SPLIT
    {
        Some(format!(
            "quadratic 2D gain field needs at least {} windows in each split; found {}/{}",
            SEAM_EXPOSURE_MIN_QUADRATIC_WINDOWS_PER_SPLIT,
            training_indices.len(),
            held_out_indices.len()
        ))
    } else if spatial_2d_training_rows < SEAM_EXPOSURE_MIN_QUADRATIC_ROWS_PER_SPLIT
        || spatial_2d_held_out_rows < SEAM_EXPOSURE_MIN_QUADRATIC_ROWS_PER_SPLIT
        || spatial_2d_training_columns < SEAM_EXPOSURE_MIN_QUADRATIC_COLUMNS_PER_SPLIT
        || spatial_2d_held_out_columns < SEAM_EXPOSURE_MIN_QUADRATIC_COLUMNS_PER_SPLIT
    {
        Some(format!(
            "quadratic 2D gain field needs at least {} rows and {} columns in each split; found rows {}/{}, columns {}/{}",
            SEAM_EXPOSURE_MIN_QUADRATIC_ROWS_PER_SPLIT,
            SEAM_EXPOSURE_MIN_QUADRATIC_COLUMNS_PER_SPLIT,
            spatial_2d_training_rows,
            spatial_2d_held_out_rows,
            spatial_2d_training_columns,
            spatial_2d_held_out_columns
        ))
    } else if spatial_quadratic_gain_fit.design_condition_number
        > SEAM_EXPOSURE_MAX_QUADRATIC_DESIGN_CONDITION
        || spatial_quadratic_gain_held_out_fit.design_condition_number
            > SEAM_EXPOSURE_MAX_QUADRATIC_DESIGN_CONDITION
    {
        Some(format!(
            "quadratic 2D gain design is ill-conditioned: training {:.1}, held-out {:.1}, limit {:.1}",
            spatial_quadratic_gain_fit.design_condition_number,
            spatial_quadratic_gain_held_out_fit.design_condition_number,
            SEAM_EXPOSURE_MAX_QUADRATIC_DESIGN_CONDITION
        ))
    } else if held_out_identity.seam_score < SEAM_EXPOSURE_MIN_SCORE {
        Some(format!(
            "quadratic 2D gain field held-out mismatch below threshold: {:.3} < {:.3}",
            held_out_identity.seam_score, SEAM_EXPOSURE_MIN_SCORE
        ))
    } else if spatial_quadratic_gain_signal < SEAM_EXPOSURE_MIN_QUADRATIC_GAIN_SIGNAL {
        Some(format!(
            "quadratic 2D gain curvature is unnecessary: {:.4} log units < {:.4}",
            spatial_quadratic_gain_signal, SEAM_EXPOSURE_MIN_QUADRATIC_GAIN_SIGNAL
        ))
    } else if !spatial_quadratic_gain_grid_in_bounds {
        Some(format!(
            "quadratic 2D gain field leaves the allowed {:.2}..{:.2} range on the {}x{} support grid",
            SEAM_EXPOSURE_MIN_GAIN,
            SEAM_EXPOSURE_MAX_GAIN,
            SEAM_EXPOSURE_QUADRATIC_EVALUATION_GRID_SIZE,
            SEAM_EXPOSURE_QUADRATIC_EVALUATION_GRID_SIZE
        ))
    } else if spatial_quadratic_gain_bounds.maximum_gain_rgb_spread > SEAM_EXPOSURE_MAX_RGB_SPREAD {
        Some(format!(
            "quadratic 2D gain field RGB spread {:.3} exceeds {:.3} on the support grid",
            spatial_quadratic_gain_bounds.maximum_gain_rgb_spread, SEAM_EXPOSURE_MAX_RGB_SPREAD
        ))
    } else if spatial_quadratic_gain_consistency < SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO {
        Some(format!(
            "held-out windows do not agree on quadratic 2D gain field: {:.1}% consistent, need {:.1}%",
            spatial_quadratic_gain_consistency * 100.0,
            SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO * 100.0
        ))
    } else if spatial_quadratic_gain_coefficient_agreement
        < SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO
    {
        Some(format!(
            "training and held-out quadratic gain curvature coefficients disagree: {:.1}% agreement, need {:.1}%",
            spatial_quadratic_gain_coefficient_agreement * 100.0,
            SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO * 100.0
        ))
    } else if spatial_quadratic_gain_validation_delta > SEAM_EXPOSURE_MAX_QUADRATIC_GAIN_FIELD_DELTA
    {
        Some(format!(
            "training and held-out quadratic gain fields differ by {:.4} log units on the support grid, limit {:.4}",
            spatial_quadratic_gain_validation_delta,
            SEAM_EXPOSURE_MAX_QUADRATIC_GAIN_FIELD_DELTA
        ))
    } else if spatial_quadratic_gain_identity_improvement
        < spatial_quadratic_gain_required_identity_improvement
    {
        Some(format!(
            "quadratic 2D gain field did not improve held-out identity enough: {:.3}, need {:.3}",
            spatial_quadratic_gain_identity_improvement,
            spatial_quadratic_gain_required_identity_improvement
        ))
    } else if spatial_quadratic_gain_simpler_improvement
        < spatial_quadratic_gain_required_simpler_improvement
    {
        Some(format!(
            "quadratic 2D gain field did not beat accepted {} on held-out windows: {:.3}, need {:.3}",
            spatial_quadratic_gain_best_simpler_model,
            spatial_quadratic_gain_simpler_improvement,
            spatial_quadratic_gain_required_simpler_improvement
        ))
    } else if spatial_quadratic_gain_clip_increase > SEAM_EXPOSURE_MAX_CLIP_INCREASE {
        Some(format!(
            "quadratic 2D gain field would increase clipping by {:.4}, limit {:.4}",
            spatial_quadratic_gain_clip_increase, SEAM_EXPOSURE_MAX_CLIP_INCREASE
        ))
    } else {
        None
    };

    let mut spatial_quadratic_affine_best_simpler_model = spatial_quadratic_gain_best_simpler_model;
    let mut spatial_quadratic_affine_best_simpler_score = spatial_quadratic_gain_best_simpler_score;
    if spatial_quadratic_gain_rejection.is_none()
        && spatial_quadratic_gain_held_out.seam_score < spatial_quadratic_affine_best_simpler_score
    {
        spatial_quadratic_affine_best_simpler_model = "quadratic 2D gain";
        spatial_quadratic_affine_best_simpler_score = spatial_quadratic_gain_held_out.seam_score;
    }
    let spatial_quadratic_affine_identity_improvement =
        held_out_identity.seam_score - spatial_quadratic_affine_held_out.seam_score;
    let spatial_quadratic_affine_simpler_improvement =
        spatial_quadratic_affine_best_simpler_score - spatial_quadratic_affine_held_out.seam_score;
    let spatial_quadratic_affine_required_identity_improvement =
        SEAM_EXPOSURE_MIN_IMPROVEMENT.max(held_out_identity.seam_score * 0.25);
    let spatial_quadratic_affine_required_simpler_improvement =
        SEAM_EXPOSURE_MIN_QUADRATIC_OVER_SIMPLER_IMPROVEMENT.max(
            spatial_quadratic_affine_best_simpler_score
                * SEAM_EXPOSURE_MIN_QUADRATIC_OVER_SIMPLER_FRACTION,
        );
    let spatial_quadratic_affine_grid_in_bounds = (0..3).all(|channel| {
        spatial_quadratic_affine_bounds.gain_min_rgb[channel] >= SEAM_EXPOSURE_MIN_GAIN
            && spatial_quadratic_affine_bounds.gain_max_rgb[channel] <= SEAM_EXPOSURE_MAX_GAIN
    });
    let spatial_quadratic_affine_clip_increase = clipping_delta(
        &clipped_high_before,
        &spatial_quadratic_affine_clipped_high,
        &clipped_low_before,
        &spatial_quadratic_affine_clipped_low,
    );
    let spatial_quadratic_affine_rejection = if let Some(reason) = common_rejection.clone() {
        Some(reason)
    } else if affine_training_indices.len() < SEAM_EXPOSURE_MIN_QUADRATIC_WINDOWS_PER_SPLIT
        || affine_held_out_indices.len() < SEAM_EXPOSURE_MIN_QUADRATIC_WINDOWS_PER_SPLIT
    {
        Some(format!(
            "quadratic 2D gain+offset field needs at least {} signal-rich windows in each split; found {}/{}",
            SEAM_EXPOSURE_MIN_QUADRATIC_WINDOWS_PER_SPLIT,
            affine_training_indices.len(),
            affine_held_out_indices.len()
        ))
    } else if spatial_2d_affine_training_rows < SEAM_EXPOSURE_MIN_QUADRATIC_ROWS_PER_SPLIT
        || spatial_2d_affine_held_out_rows < SEAM_EXPOSURE_MIN_QUADRATIC_ROWS_PER_SPLIT
        || spatial_2d_affine_training_columns < SEAM_EXPOSURE_MIN_QUADRATIC_COLUMNS_PER_SPLIT
        || spatial_2d_affine_held_out_columns < SEAM_EXPOSURE_MIN_QUADRATIC_COLUMNS_PER_SPLIT
    {
        Some(format!(
            "quadratic 2D gain+offset field needs at least {} rows and {} columns in each signal-rich split; found rows {}/{}, columns {}/{}",
            SEAM_EXPOSURE_MIN_QUADRATIC_ROWS_PER_SPLIT,
            SEAM_EXPOSURE_MIN_QUADRATIC_COLUMNS_PER_SPLIT,
            spatial_2d_affine_training_rows,
            spatial_2d_affine_held_out_rows,
            spatial_2d_affine_training_columns,
            spatial_2d_affine_held_out_columns
        ))
    } else if !spatial_quadratic_affine_fit_performed {
        Some(format!(
            "quadratic 2D gain+offset solve was unnecessary: quadratic gain residual {:.4} cannot clear the {:.4} absolute complexity margin",
            spatial_quadratic_gain_held_out.seam_score,
            SEAM_EXPOSURE_MIN_QUADRATIC_OVER_SIMPLER_IMPROVEMENT
        ))
    } else if spatial_quadratic_affine_fit.design_condition_number
        > SEAM_EXPOSURE_MAX_QUADRATIC_DESIGN_CONDITION
        || spatial_quadratic_affine_held_out_fit.design_condition_number
            > SEAM_EXPOSURE_MAX_QUADRATIC_DESIGN_CONDITION
    {
        Some(format!(
            "quadratic 2D gain+offset design is ill-conditioned: training {:.1}, held-out {:.1}, limit {:.1}",
            spatial_quadratic_affine_fit.design_condition_number,
            spatial_quadratic_affine_held_out_fit.design_condition_number,
            SEAM_EXPOSURE_MAX_QUADRATIC_DESIGN_CONDITION
        ))
    } else if held_out_identity.seam_score < SEAM_EXPOSURE_MIN_SCORE {
        Some(format!(
            "quadratic 2D gain+offset held-out mismatch below threshold: {:.3} < {:.3}",
            held_out_identity.seam_score, SEAM_EXPOSURE_MIN_SCORE
        ))
    } else if spatial_quadratic_affine_gain_signal < SEAM_EXPOSURE_MIN_QUADRATIC_GAIN_SIGNAL
        && spatial_quadratic_affine_offset_signal
            < SEAM_EXPOSURE_MIN_QUADRATIC_OFFSET_SIGNAL_NORMALIZED
    {
        Some(format!(
            "quadratic 2D gain+offset curvature is unnecessary: gain {:.4} log units, offset {:.2}% of sample range",
            spatial_quadratic_affine_gain_signal,
            spatial_quadratic_affine_offset_signal * 100.0
        ))
    } else if !spatial_quadratic_affine_grid_in_bounds {
        Some(format!(
            "quadratic 2D gain+offset field leaves the allowed gain range {:.2}..{:.2} on the support grid",
            SEAM_EXPOSURE_MIN_GAIN, SEAM_EXPOSURE_MAX_GAIN
        ))
    } else if spatial_quadratic_affine_bounds.maximum_gain_rgb_spread > SEAM_EXPOSURE_MAX_RGB_SPREAD
    {
        Some(format!(
            "quadratic 2D gain+offset RGB spread {:.3} exceeds {:.3} on the support grid",
            spatial_quadratic_affine_bounds.maximum_gain_rgb_spread, SEAM_EXPOSURE_MAX_RGB_SPREAD
        ))
    } else if spatial_quadratic_affine_bounds.offset_abs_max_normalized
        > SEAM_EXPOSURE_MAX_OFFSET_NORMALIZED
    {
        Some(format!(
            "quadratic 2D gain+offset additive field exceeds {:.1}% of sample range: {:.2}%",
            SEAM_EXPOSURE_MAX_OFFSET_NORMALIZED * 100.0,
            spatial_quadratic_affine_bounds.offset_abs_max_normalized * 100.0
        ))
    } else if spatial_quadratic_affine_bounds.offset_abs_max_normalized
        < SEAM_EXPOSURE_MIN_OFFSET_NORMALIZED
    {
        Some(format!(
            "quadratic 2D gain+offset additive term is unnecessary: {:.2}% < {:.2}% of sample range",
            spatial_quadratic_affine_bounds.offset_abs_max_normalized * 100.0,
            SEAM_EXPOSURE_MIN_OFFSET_NORMALIZED * 100.0
        ))
    } else if spatial_quadratic_affine_consistency < SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO {
        Some(format!(
            "held-out windows do not agree on quadratic 2D gain+offset field: {:.1}% consistent, need {:.1}%",
            spatial_quadratic_affine_consistency * 100.0,
            SEAM_EXPOSURE_MIN_CONSISTENT_WINDOW_RATIO * 100.0
        ))
    } else if spatial_quadratic_affine_coefficient_agreement
        < SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO
    {
        Some(format!(
            "training and held-out quadratic gain+offset curvature coefficients disagree: {:.1}% agreement, need {:.1}%",
            spatial_quadratic_affine_coefficient_agreement * 100.0,
            SEAM_EXPOSURE_MIN_SPATIAL_SLOPE_AGREEMENT_RATIO * 100.0
        ))
    } else if spatial_quadratic_affine_gain_validation_delta
        > SEAM_EXPOSURE_MAX_QUADRATIC_GAIN_FIELD_DELTA
    {
        Some(format!(
            "training and held-out quadratic gain+offset gain fields differ by {:.4} log units, limit {:.4}",
            spatial_quadratic_affine_gain_validation_delta,
            SEAM_EXPOSURE_MAX_QUADRATIC_GAIN_FIELD_DELTA
        ))
    } else if spatial_quadratic_affine_offset_validation_delta
        > SEAM_EXPOSURE_MAX_QUADRATIC_OFFSET_FIELD_DELTA_NORMALIZED
    {
        Some(format!(
            "training and held-out quadratic offset fields differ by {:.2}% of sample range, limit {:.2}%",
            spatial_quadratic_affine_offset_validation_delta * 100.0,
            SEAM_EXPOSURE_MAX_QUADRATIC_OFFSET_FIELD_DELTA_NORMALIZED * 100.0
        ))
    } else if spatial_quadratic_affine_identity_improvement
        < spatial_quadratic_affine_required_identity_improvement
    {
        Some(format!(
            "quadratic 2D gain+offset field did not improve held-out identity enough: {:.3}, need {:.3}",
            spatial_quadratic_affine_identity_improvement,
            spatial_quadratic_affine_required_identity_improvement
        ))
    } else if spatial_quadratic_affine_simpler_improvement
        < spatial_quadratic_affine_required_simpler_improvement
    {
        Some(format!(
            "quadratic 2D gain+offset field did not beat accepted {} on held-out windows: {:.3}, need {:.3}",
            spatial_quadratic_affine_best_simpler_model,
            spatial_quadratic_affine_simpler_improvement,
            spatial_quadratic_affine_required_simpler_improvement
        ))
    } else if spatial_quadratic_affine_clip_increase > SEAM_EXPOSURE_MAX_CLIP_INCREASE {
        Some(format!(
            "quadratic 2D gain+offset field would increase clipping by {:.4}, limit {:.4}",
            spatial_quadratic_affine_clip_increase, SEAM_EXPOSURE_MAX_CLIP_INCREASE
        ))
    } else {
        None
    };

    let prefer_affine_candidate = affine_held_out.seam_score < held_out_gain.seam_score
        && !affine_held_out_samples.is_empty();
    let (candidate_gain_rgb, candidate_gain_luma, candidate_offset_rgb, candidate_offset_luma) =
        if prefer_affine_candidate {
            (
                affine_gain_rgb,
                affine_gain_luma,
                affine_offset_rgb,
                affine_offset_luma,
            )
        } else {
            (gain_candidate_rgb, gain_candidate_luma, zero_offset, 0.0)
        };
    let candidate_measurements = if prefer_affine_candidate {
        affine_measurements
    } else {
        gain_measurements
    };
    let (candidate_clipped_high_after, candidate_clipped_low_after) = if prefer_affine_candidate {
        (affine_clipped_high, affine_clipped_low)
    } else {
        (gain_clipped_high, gain_clipped_low)
    };

    let mut diagnostics = SeamExposureCorrectionDiagnostics {
        mode: "auto".to_string(),
        model: "identity".to_string(),
        applied: false,
        reason: gain_rejection.clone().unwrap_or_else(|| {
            "gain-only candidate passed; checking gain+offset candidate".to_string()
        }),
        sample_count,
        valid_sample_ratio,
        gain_rgb: identity_gain,
        gain_luma: 1.0,
        spatial_gain_log_slope_x_rgb: zero_offset,
        spatial_gain_log_slope_x_luma: 0.0,
        spatial_gain_log_slope_y_rgb: zero_offset,
        spatial_gain_log_slope_y_luma: 0.0,
        spatial_gain_log_quadratic_xx_rgb: zero_offset,
        spatial_gain_log_quadratic_xx_luma: 0.0,
        spatial_gain_log_quadratic_xy_rgb: zero_offset,
        spatial_gain_log_quadratic_xy_luma: 0.0,
        spatial_gain_log_quadratic_yy_rgb: zero_offset,
        spatial_gain_log_quadratic_yy_luma: 0.0,
        spatial_gain_top_rgb: identity_gain,
        spatial_gain_bottom_rgb: identity_gain,
        spatial_gain_corners_rgb: [identity_gain; 4],
        spatial_offset_slope_x_rgb: zero_offset,
        spatial_offset_slope_x_luma: 0.0,
        spatial_offset_slope_x_rgb_normalized: zero_offset,
        spatial_offset_slope_y_rgb: zero_offset,
        spatial_offset_slope_y_luma: 0.0,
        spatial_offset_slope_y_rgb_normalized: zero_offset,
        spatial_offset_quadratic_xx_rgb: zero_offset,
        spatial_offset_quadratic_xx_luma: 0.0,
        spatial_offset_quadratic_xx_rgb_normalized: zero_offset,
        spatial_offset_quadratic_xy_rgb: zero_offset,
        spatial_offset_quadratic_xy_luma: 0.0,
        spatial_offset_quadratic_xy_rgb_normalized: zero_offset,
        spatial_offset_quadratic_yy_rgb: zero_offset,
        spatial_offset_quadratic_yy_luma: 0.0,
        spatial_offset_quadratic_yy_rgb_normalized: zero_offset,
        spatial_offset_top_rgb: zero_offset,
        spatial_offset_bottom_rgb: zero_offset,
        spatial_offset_top_rgb_normalized: zero_offset,
        spatial_offset_bottom_rgb_normalized: zero_offset,
        spatial_offset_corners_rgb: [zero_offset; 4],
        spatial_offset_corners_rgb_normalized: [zero_offset; 4],
        offset_rgb: zero_offset,
        offset_luma: 0.0,
        offset_rgb_normalized: zero_offset,
        delta_luma_before: measurements_before.delta_luma,
        delta_luma_after: measurements_before.delta_luma,
        delta_rgb_before: measurements_before.delta_rgb,
        delta_rgb_after: measurements_before.delta_rgb,
        seam_score_before: measurements_before.seam_score,
        seam_score_after: measurements_before.seam_score,
        clipped_high_before,
        clipped_high_after: clipped_high_before,
        clipped_low_before,
        clipped_low_after: clipped_low_before,
        estimated_gain_rgb,
        estimated_gain_luma,
        estimated_offset_rgb: affine_offset_rgb,
        estimated_offset_luma: affine_offset_luma,
        estimated_spatial_gain_log_slope_y_rgb: spatial_fit.log_slope_y_rgb,
        estimated_spatial_gain_log_slope_y_luma: spatial_fit.log_slope_y_luma,
        estimated_spatial_gain_top_rgb: spatial_top_gain,
        estimated_spatial_gain_bottom_rgb: spatial_bottom_gain,
        estimated_spatial_affine_center_gain_rgb: spatial_affine_fit.center_gain_rgb,
        estimated_spatial_affine_center_offset_rgb: spatial_affine_fit.center_offset_rgb,
        estimated_spatial_affine_gain_log_slope_y_rgb: spatial_affine_fit.log_gain_slope_y_rgb,
        estimated_spatial_affine_gain_log_slope_y_luma: spatial_affine_fit.log_gain_slope_y_luma,
        estimated_spatial_affine_offset_slope_y_rgb: spatial_affine_fit.offset_slope_y_rgb,
        estimated_spatial_affine_offset_slope_y_luma: spatial_affine_fit.offset_slope_y_luma,
        estimated_spatial_affine_gain_top_rgb: spatial_affine_top_gain,
        estimated_spatial_affine_gain_bottom_rgb: spatial_affine_bottom_gain,
        estimated_spatial_affine_offset_top_rgb: spatial_affine_top_offset,
        estimated_spatial_affine_offset_bottom_rgb: spatial_affine_bottom_offset,
        candidate_gain_rgb,
        candidate_gain_luma,
        candidate_offset_rgb,
        candidate_offset_luma,
        candidate_delta_luma_after: candidate_measurements.delta_luma,
        candidate_delta_rgb_after: candidate_measurements.delta_rgb,
        candidate_seam_score_after: candidate_measurements.seam_score,
        candidate_clipped_high_after,
        candidate_clipped_low_after,
        window_count: windows.len(),
        consistent_window_ratio,
        per_channel_gain_used: use_per_channel,
        training_window_count: training_indices.len(),
        held_out_window_count: held_out_indices.len(),
        training_sample_count: filtered_training_samples.len(),
        held_out_sample_count: held_out_samples.len(),
        gain_offset_training_window_count: affine_training_indices.len(),
        gain_offset_held_out_window_count: affine_held_out_indices.len(),
        gain_offset_consistent_window_ratio: affine_consistency,
        spatial_training_window_count: training_indices.len(),
        spatial_held_out_window_count: held_out_indices.len(),
        spatial_distinct_training_rows: spatial_training_rows,
        spatial_distinct_held_out_rows: spatial_held_out_rows,
        spatial_consistent_window_ratio: spatial_consistency,
        spatial_slope_agreement_ratio: spatial_slope_agreement,
        spatial_affine_training_window_count: affine_training_indices.len(),
        spatial_affine_held_out_window_count: affine_held_out_indices.len(),
        spatial_affine_distinct_training_rows: spatial_affine_training_rows,
        spatial_affine_distinct_held_out_rows: spatial_affine_held_out_rows,
        spatial_affine_consistent_window_ratio: spatial_affine_consistency,
        spatial_affine_slope_agreement_ratio: spatial_affine_slope_agreement,
        spatial_affine_center_offset_delta_normalized: spatial_affine_center_offset_delta,
        held_out_identity_seam_score: held_out_identity.seam_score,
        held_out_gain_seam_score: held_out_gain.seam_score,
        held_out_gain_offset_seam_score: affine_held_out.seam_score,
        held_out_spatial_gain_seam_score: spatial_held_out.seam_score,
        held_out_spatial_gain_offset_seam_score: spatial_affine_held_out.seam_score,
        held_out_selected_seam_score: held_out_identity.seam_score,
        held_out_improvement_over_identity: 0.0,
        held_out_improvement_over_gain: affine_gain_improvement,
        held_out_spatial_improvement_over_best_constant: spatial_best_constant_improvement,
        held_out_spatial_gain_offset_improvement_over_best_simpler:
            spatial_affine_simpler_improvement,
        held_out_validation_passed: false,
        gain_offset_rejection_reason: affine_rejection
            .clone()
            .unwrap_or_else(|| "accepted".to_string()),
        spatial_rejection_reason: spatial_rejection
            .clone()
            .unwrap_or_else(|| "accepted".to_string()),
        spatial_gain_offset_rejection_reason: spatial_affine_rejection
            .clone()
            .unwrap_or_else(|| "accepted".to_string()),
        spatial_2d_validation: SpatialPhotometric2dDiagnostics {
            coordinate_system: "normalized_overlap_xy_clamped_outside_support",
            corner_order: ["top_left", "top_right", "bottom_left", "bottom_right"],
            training_window_count: training_indices.len(),
            held_out_window_count: held_out_indices.len(),
            distinct_training_rows: spatial_2d_training_rows,
            distinct_held_out_rows: spatial_2d_held_out_rows,
            distinct_training_columns: spatial_2d_training_columns,
            distinct_held_out_columns: spatial_2d_held_out_columns,
            gain_consistent_window_ratio: spatial_2d_gain_consistency,
            gain_slope_agreement_ratio: spatial_2d_gain_slope_agreement,
            gain_horizontal_slope_agreement_ratio: spatial_2d_gain_horizontal_slope_agreement,
            gain_center_log_delta: spatial_2d_gain_center_delta,
            gain_offset_consistent_window_ratio: spatial_2d_affine_consistency,
            gain_offset_slope_agreement_ratio: spatial_2d_affine_slope_agreement,
            gain_offset_horizontal_slope_agreement_ratio:
                spatial_2d_affine_horizontal_slope_agreement,
            gain_offset_center_gain_log_delta: spatial_2d_affine_center_gain_delta,
            gain_offset_center_offset_delta_normalized: spatial_2d_affine_center_offset_delta,
            estimated_gain_center_rgb: spatial_2d_gain_fit.center_gain_rgb,
            estimated_gain_log_slope_x_rgb: spatial_2d_gain_fit.log_slope_x_rgb,
            estimated_gain_log_slope_y_rgb: spatial_2d_gain_fit.log_slope_y_rgb,
            estimated_gain_corners_rgb: spatial_2d_gain_corners,
            estimated_gain_offset_center_gain_rgb: spatial_2d_affine_fit.center_gain_rgb,
            estimated_gain_offset_center_offset_rgb: spatial_2d_affine_fit.center_offset_rgb,
            estimated_gain_offset_log_slope_x_rgb: spatial_2d_affine_fit.log_gain_slope_x_rgb,
            estimated_gain_offset_log_slope_y_rgb: spatial_2d_affine_fit.log_gain_slope_y_rgb,
            estimated_gain_offset_slope_x_rgb: spatial_2d_affine_fit.offset_slope_x_rgb,
            estimated_gain_offset_slope_y_rgb: spatial_2d_affine_fit.offset_slope_y_rgb,
            estimated_gain_offset_gain_corners_rgb: spatial_2d_affine_gain_corners,
            estimated_gain_offset_offset_corners_rgb: spatial_2d_affine_offset_corners,
            held_out_gain_seam_score: spatial_2d_gain_held_out.seam_score,
            held_out_gain_offset_seam_score: spatial_2d_affine_held_out.seam_score,
            gain_best_simpler_model: spatial_2d_gain_best_simpler_model.to_string(),
            gain_improvement_over_best_simpler: spatial_2d_gain_simpler_improvement,
            gain_offset_best_simpler_model: spatial_2d_affine_best_simpler_model.to_string(),
            gain_offset_improvement_over_best_simpler: spatial_2d_affine_simpler_improvement,
            gain_accepted: spatial_2d_gain_rejection.is_none(),
            gain_offset_accepted: spatial_2d_affine_rejection.is_none(),
            gain_rejection_reason: spatial_2d_gain_rejection
                .clone()
                .unwrap_or_else(|| "accepted".to_string()),
            gain_offset_rejection_reason: spatial_2d_affine_rejection
                .clone()
                .unwrap_or_else(|| "accepted".to_string()),
            quadratic: SpatialPhotometricQuadraticDiagnostics {
                basis: ["1", "x", "y", "x_squared", "x_y", "y_squared"],
                evaluation_grid_size: SEAM_EXPOSURE_QUADRATIC_EVALUATION_GRID_SIZE,
                regularization_lambda: SEAM_EXPOSURE_QUADRATIC_REGULARIZATION,
                minimum_windows_per_split: SEAM_EXPOSURE_MIN_QUADRATIC_WINDOWS_PER_SPLIT,
                minimum_rows_per_split: SEAM_EXPOSURE_MIN_QUADRATIC_ROWS_PER_SPLIT,
                minimum_columns_per_split: SEAM_EXPOSURE_MIN_QUADRATIC_COLUMNS_PER_SPLIT,
                training_window_count: training_indices.len(),
                held_out_window_count: held_out_indices.len(),
                distinct_training_rows: spatial_2d_training_rows,
                distinct_held_out_rows: spatial_2d_held_out_rows,
                distinct_training_columns: spatial_2d_training_columns,
                distinct_held_out_columns: spatial_2d_held_out_columns,
                gain_design_condition_number: spatial_quadratic_gain_fit.design_condition_number,
                held_out_gain_design_condition_number: spatial_quadratic_gain_held_out_fit
                    .design_condition_number,
                gain_consistent_window_ratio: spatial_quadratic_gain_consistency,
                gain_curvature_coefficient_agreement_ratio:
                    spatial_quadratic_gain_coefficient_agreement,
                gain_max_validation_field_log_delta: spatial_quadratic_gain_validation_delta,
                gain_curvature_signal: spatial_quadratic_gain_signal,
                estimated_gain_log_quadratic_xx_rgb: spatial_quadratic_gain_field
                    .log_gain_quadratic_xx,
                estimated_gain_log_quadratic_xy_rgb: spatial_quadratic_gain_field
                    .log_gain_quadratic_xy,
                estimated_gain_log_quadratic_yy_rgb: spatial_quadratic_gain_field
                    .log_gain_quadratic_yy,
                gain_grid_min_rgb: spatial_quadratic_gain_bounds.gain_min_rgb,
                gain_grid_max_rgb: spatial_quadratic_gain_bounds.gain_max_rgb,
                gain_offset_design_condition_number: spatial_quadratic_affine_fit
                    .design_condition_number,
                held_out_gain_offset_design_condition_number: spatial_quadratic_affine_held_out_fit
                    .design_condition_number,
                gain_offset_consistent_window_ratio: spatial_quadratic_affine_consistency,
                gain_offset_curvature_coefficient_agreement_ratio:
                    spatial_quadratic_affine_coefficient_agreement,
                gain_offset_max_validation_gain_field_log_delta:
                    spatial_quadratic_affine_gain_validation_delta,
                gain_offset_max_validation_offset_field_delta_normalized:
                    spatial_quadratic_affine_offset_validation_delta,
                gain_offset_curvature_signal: spatial_quadratic_affine_signal,
                estimated_gain_offset_log_quadratic_xx_rgb: spatial_quadratic_affine_field
                    .log_gain_quadratic_xx,
                estimated_gain_offset_log_quadratic_xy_rgb: spatial_quadratic_affine_field
                    .log_gain_quadratic_xy,
                estimated_gain_offset_log_quadratic_yy_rgb: spatial_quadratic_affine_field
                    .log_gain_quadratic_yy,
                estimated_gain_offset_quadratic_xx_rgb: spatial_quadratic_affine_field
                    .offset_quadratic_xx,
                estimated_gain_offset_quadratic_xy_rgb: spatial_quadratic_affine_field
                    .offset_quadratic_xy,
                estimated_gain_offset_quadratic_yy_rgb: spatial_quadratic_affine_field
                    .offset_quadratic_yy,
                gain_offset_grid_gain_min_rgb: spatial_quadratic_affine_bounds.gain_min_rgb,
                gain_offset_grid_gain_max_rgb: spatial_quadratic_affine_bounds.gain_max_rgb,
                gain_offset_grid_offset_abs_max_normalized: spatial_quadratic_affine_bounds
                    .offset_abs_max_normalized,
                held_out_gain_seam_score: spatial_quadratic_gain_held_out.seam_score,
                held_out_gain_offset_seam_score: spatial_quadratic_affine_held_out.seam_score,
                gain_best_simpler_model: spatial_quadratic_gain_best_simpler_model.to_string(),
                gain_improvement_over_best_simpler: spatial_quadratic_gain_simpler_improvement,
                gain_offset_best_simpler_model: spatial_quadratic_affine_best_simpler_model
                    .to_string(),
                gain_offset_improvement_over_best_simpler:
                    spatial_quadratic_affine_simpler_improvement,
                gain_accepted: spatial_quadratic_gain_rejection.is_none(),
                gain_offset_accepted: spatial_quadratic_affine_rejection.is_none(),
                gain_rejection_reason: spatial_quadratic_gain_rejection
                    .clone()
                    .unwrap_or_else(|| "accepted".to_string()),
                gain_offset_rejection_reason: spatial_quadratic_affine_rejection
                    .clone()
                    .unwrap_or_else(|| "accepted".to_string()),
            },
        },
    };

    if spatial_quadratic_affine_rejection.is_none() {
        let gain_corners = SPATIAL_FIELD_CORNERS.map(|[x, y]| {
            spatial_photometric_at_normalized_xy(&spatial_quadratic_affine_field, x, y).0
        });
        let offset_corners = SPATIAL_FIELD_CORNERS.map(|[x, y]| {
            spatial_photometric_at_normalized_xy(&spatial_quadratic_affine_field, x, y).1
        });
        let (center_top_gain, center_top_offset) =
            spatial_photometric_at_normalized_xy(&spatial_quadratic_affine_field, 0.0, -1.0);
        let (center_bottom_gain, center_bottom_offset) =
            spatial_photometric_at_normalized_xy(&spatial_quadratic_affine_field, 0.0, 1.0);
        diagnostics.model = "gain_offset_spatial_quadratic_xy_rgb".to_string();
        diagnostics.applied = true;
        diagnostics.reason = format!(
            "applied bounded overlap-derived quadratic 2D RGB gain+offset field after disjoint validation; score {:.3} -> {:.3}, beating accepted {} by {:.3}",
            held_out_identity.seam_score,
            spatial_quadratic_affine_held_out.seam_score,
            spatial_quadratic_affine_best_simpler_model,
            spatial_quadratic_affine_simpler_improvement
        );
        diagnostics.gain_rgb = spatial_quadratic_affine_field.center_gain;
        diagnostics.gain_luma = spatial_quadratic_affine_fit.log_gain_luma[0].exp();
        diagnostics.offset_rgb = spatial_quadratic_affine_field.center_offset;
        diagnostics.offset_luma = spatial_quadratic_affine_fit.offset_luma[0];
        diagnostics.offset_rgb_normalized = std::array::from_fn(|channel| {
            spatial_quadratic_affine_field.center_offset[channel] / max_value
        });
        diagnostics.spatial_gain_log_slope_x_rgb = spatial_quadratic_affine_field.log_gain_slope_x;
        diagnostics.spatial_gain_log_slope_x_luma = spatial_quadratic_affine_fit.log_gain_luma[1];
        diagnostics.spatial_gain_log_slope_y_rgb = spatial_quadratic_affine_field.log_gain_slope_y;
        diagnostics.spatial_gain_log_slope_y_luma = spatial_quadratic_affine_fit.log_gain_luma[2];
        diagnostics.spatial_gain_log_quadratic_xx_rgb =
            spatial_quadratic_affine_field.log_gain_quadratic_xx;
        diagnostics.spatial_gain_log_quadratic_xx_luma =
            spatial_quadratic_affine_fit.log_gain_luma[3];
        diagnostics.spatial_gain_log_quadratic_xy_rgb =
            spatial_quadratic_affine_field.log_gain_quadratic_xy;
        diagnostics.spatial_gain_log_quadratic_xy_luma =
            spatial_quadratic_affine_fit.log_gain_luma[4];
        diagnostics.spatial_gain_log_quadratic_yy_rgb =
            spatial_quadratic_affine_field.log_gain_quadratic_yy;
        diagnostics.spatial_gain_log_quadratic_yy_luma =
            spatial_quadratic_affine_fit.log_gain_luma[5];
        diagnostics.spatial_gain_top_rgb = center_top_gain;
        diagnostics.spatial_gain_bottom_rgb = center_bottom_gain;
        diagnostics.spatial_gain_corners_rgb = gain_corners;
        diagnostics.spatial_offset_slope_x_rgb = spatial_quadratic_affine_field.offset_slope_x;
        diagnostics.spatial_offset_slope_x_luma = spatial_quadratic_affine_fit.offset_luma[1];
        diagnostics.spatial_offset_slope_x_rgb_normalized = std::array::from_fn(|channel| {
            spatial_quadratic_affine_field.offset_slope_x[channel] / max_value
        });
        diagnostics.spatial_offset_slope_y_rgb = spatial_quadratic_affine_field.offset_slope_y;
        diagnostics.spatial_offset_slope_y_luma = spatial_quadratic_affine_fit.offset_luma[2];
        diagnostics.spatial_offset_slope_y_rgb_normalized = std::array::from_fn(|channel| {
            spatial_quadratic_affine_field.offset_slope_y[channel] / max_value
        });
        diagnostics.spatial_offset_quadratic_xx_rgb =
            spatial_quadratic_affine_field.offset_quadratic_xx;
        diagnostics.spatial_offset_quadratic_xx_luma = spatial_quadratic_affine_fit.offset_luma[3];
        diagnostics.spatial_offset_quadratic_xx_rgb_normalized = std::array::from_fn(|channel| {
            spatial_quadratic_affine_field.offset_quadratic_xx[channel] / max_value
        });
        diagnostics.spatial_offset_quadratic_xy_rgb =
            spatial_quadratic_affine_field.offset_quadratic_xy;
        diagnostics.spatial_offset_quadratic_xy_luma = spatial_quadratic_affine_fit.offset_luma[4];
        diagnostics.spatial_offset_quadratic_xy_rgb_normalized = std::array::from_fn(|channel| {
            spatial_quadratic_affine_field.offset_quadratic_xy[channel] / max_value
        });
        diagnostics.spatial_offset_quadratic_yy_rgb =
            spatial_quadratic_affine_field.offset_quadratic_yy;
        diagnostics.spatial_offset_quadratic_yy_luma = spatial_quadratic_affine_fit.offset_luma[5];
        diagnostics.spatial_offset_quadratic_yy_rgb_normalized = std::array::from_fn(|channel| {
            spatial_quadratic_affine_field.offset_quadratic_yy[channel] / max_value
        });
        diagnostics.spatial_offset_top_rgb = center_top_offset;
        diagnostics.spatial_offset_bottom_rgb = center_bottom_offset;
        diagnostics.spatial_offset_top_rgb_normalized =
            std::array::from_fn(|channel| center_top_offset[channel] / max_value);
        diagnostics.spatial_offset_bottom_rgb_normalized =
            std::array::from_fn(|channel| center_bottom_offset[channel] / max_value);
        diagnostics.spatial_offset_corners_rgb = offset_corners;
        diagnostics.spatial_offset_corners_rgb_normalized =
            offset_corners.map(|corner| std::array::from_fn(|channel| corner[channel] / max_value));
        diagnostics.delta_luma_after = spatial_quadratic_affine_measurements.delta_luma;
        diagnostics.delta_rgb_after = spatial_quadratic_affine_measurements.delta_rgb;
        diagnostics.seam_score_after = spatial_quadratic_affine_measurements.seam_score;
        diagnostics.clipped_high_after = spatial_quadratic_affine_clipped_high;
        diagnostics.clipped_low_after = spatial_quadratic_affine_clipped_low;
        diagnostics.candidate_gain_rgb = spatial_quadratic_affine_field.center_gain;
        diagnostics.candidate_gain_luma = spatial_quadratic_affine_fit.log_gain_luma[0].exp();
        diagnostics.candidate_offset_rgb = spatial_quadratic_affine_field.center_offset;
        diagnostics.candidate_offset_luma = spatial_quadratic_affine_fit.offset_luma[0];
        diagnostics.candidate_delta_luma_after = spatial_quadratic_affine_measurements.delta_luma;
        diagnostics.candidate_delta_rgb_after = spatial_quadratic_affine_measurements.delta_rgb;
        diagnostics.candidate_seam_score_after = spatial_quadratic_affine_measurements.seam_score;
        diagnostics.candidate_clipped_high_after = spatial_quadratic_affine_clipped_high;
        diagnostics.candidate_clipped_low_after = spatial_quadratic_affine_clipped_low;
        diagnostics.per_channel_gain_used = true;
        diagnostics.held_out_selected_seam_score = spatial_quadratic_affine_held_out.seam_score;
        diagnostics.held_out_improvement_over_identity =
            spatial_quadratic_affine_identity_improvement;
        diagnostics.held_out_improvement_over_gain =
            held_out_gain.seam_score - spatial_quadratic_affine_held_out.seam_score;
        diagnostics.held_out_validation_passed = true;
        return diagnostics;
    }

    if spatial_quadratic_gain_rejection.is_none() {
        let gain_corners = SPATIAL_FIELD_CORNERS.map(|[x, y]| {
            spatial_photometric_at_normalized_xy(&spatial_quadratic_gain_field, x, y).0
        });
        diagnostics.model = "gain_spatial_quadratic_xy_rgb".to_string();
        diagnostics.applied = true;
        diagnostics.reason = format!(
            "applied bounded overlap-derived quadratic 2D RGB gain field after disjoint validation; score {:.3} -> {:.3}, beating accepted {} by {:.3}",
            held_out_identity.seam_score,
            spatial_quadratic_gain_held_out.seam_score,
            spatial_quadratic_gain_best_simpler_model,
            spatial_quadratic_gain_simpler_improvement
        );
        diagnostics.gain_rgb = spatial_quadratic_gain_field.center_gain;
        diagnostics.gain_luma = spatial_quadratic_gain_fit.log_gain_luma[0].exp();
        diagnostics.spatial_gain_log_slope_x_rgb = spatial_quadratic_gain_field.log_gain_slope_x;
        diagnostics.spatial_gain_log_slope_x_luma = spatial_quadratic_gain_fit.log_gain_luma[1];
        diagnostics.spatial_gain_log_slope_y_rgb = spatial_quadratic_gain_field.log_gain_slope_y;
        diagnostics.spatial_gain_log_slope_y_luma = spatial_quadratic_gain_fit.log_gain_luma[2];
        diagnostics.spatial_gain_log_quadratic_xx_rgb =
            spatial_quadratic_gain_field.log_gain_quadratic_xx;
        diagnostics.spatial_gain_log_quadratic_xx_luma =
            spatial_quadratic_gain_fit.log_gain_luma[3];
        diagnostics.spatial_gain_log_quadratic_xy_rgb =
            spatial_quadratic_gain_field.log_gain_quadratic_xy;
        diagnostics.spatial_gain_log_quadratic_xy_luma =
            spatial_quadratic_gain_fit.log_gain_luma[4];
        diagnostics.spatial_gain_log_quadratic_yy_rgb =
            spatial_quadratic_gain_field.log_gain_quadratic_yy;
        diagnostics.spatial_gain_log_quadratic_yy_luma =
            spatial_quadratic_gain_fit.log_gain_luma[5];
        diagnostics.spatial_gain_top_rgb =
            spatial_photometric_at_normalized_xy(&spatial_quadratic_gain_field, 0.0, -1.0).0;
        diagnostics.spatial_gain_bottom_rgb =
            spatial_photometric_at_normalized_xy(&spatial_quadratic_gain_field, 0.0, 1.0).0;
        diagnostics.spatial_gain_corners_rgb = gain_corners;
        diagnostics.delta_luma_after = spatial_quadratic_gain_measurements.delta_luma;
        diagnostics.delta_rgb_after = spatial_quadratic_gain_measurements.delta_rgb;
        diagnostics.seam_score_after = spatial_quadratic_gain_measurements.seam_score;
        diagnostics.clipped_high_after = spatial_quadratic_gain_clipped_high;
        diagnostics.clipped_low_after = spatial_quadratic_gain_clipped_low;
        diagnostics.candidate_gain_rgb = spatial_quadratic_gain_field.center_gain;
        diagnostics.candidate_gain_luma = spatial_quadratic_gain_fit.log_gain_luma[0].exp();
        diagnostics.candidate_offset_rgb = zero_offset;
        diagnostics.candidate_offset_luma = 0.0;
        diagnostics.candidate_delta_luma_after = spatial_quadratic_gain_measurements.delta_luma;
        diagnostics.candidate_delta_rgb_after = spatial_quadratic_gain_measurements.delta_rgb;
        diagnostics.candidate_seam_score_after = spatial_quadratic_gain_measurements.seam_score;
        diagnostics.candidate_clipped_high_after = spatial_quadratic_gain_clipped_high;
        diagnostics.candidate_clipped_low_after = spatial_quadratic_gain_clipped_low;
        diagnostics.per_channel_gain_used = true;
        diagnostics.held_out_selected_seam_score = spatial_quadratic_gain_held_out.seam_score;
        diagnostics.held_out_improvement_over_identity =
            spatial_quadratic_gain_identity_improvement;
        diagnostics.held_out_improvement_over_gain =
            held_out_gain.seam_score - spatial_quadratic_gain_held_out.seam_score;
        diagnostics.held_out_validation_passed = true;
        return diagnostics;
    }

    if spatial_2d_affine_rejection.is_none() {
        let center_top_gain = spatial_affine_2d_gain_at(&spatial_2d_affine_fit, 0.0, -1.0);
        let center_bottom_gain = spatial_affine_2d_gain_at(&spatial_2d_affine_fit, 0.0, 1.0);
        let center_top_offset = spatial_affine_2d_offset_at(&spatial_2d_affine_fit, 0.0, -1.0);
        let center_bottom_offset = spatial_affine_2d_offset_at(&spatial_2d_affine_fit, 0.0, 1.0);
        diagnostics.model = "gain_offset_spatial_xy_rgb".to_string();
        diagnostics.applied = true;
        diagnostics.reason = format!(
            "applied bounded overlap-derived 2D RGB gain+offset field after disjoint validation; score {:.3} -> {:.3}, beating accepted {} by {:.3}",
            held_out_identity.seam_score,
            spatial_2d_affine_held_out.seam_score,
            spatial_2d_affine_best_simpler_model,
            spatial_2d_affine_simpler_improvement
        );
        diagnostics.gain_rgb = spatial_2d_affine_fit.center_gain_rgb;
        diagnostics.gain_luma = spatial_2d_affine_fit.center_gain_luma;
        diagnostics.offset_rgb = spatial_2d_affine_fit.center_offset_rgb;
        diagnostics.offset_luma = spatial_2d_affine_fit.center_offset_luma;
        diagnostics.offset_rgb_normalized = std::array::from_fn(|channel| {
            spatial_2d_affine_fit.center_offset_rgb[channel] / max_value
        });
        diagnostics.spatial_gain_log_slope_x_rgb = spatial_2d_affine_fit.log_gain_slope_x_rgb;
        diagnostics.spatial_gain_log_slope_x_luma = spatial_2d_affine_fit.log_gain_slope_x_luma;
        diagnostics.spatial_gain_log_slope_y_rgb = spatial_2d_affine_fit.log_gain_slope_y_rgb;
        diagnostics.spatial_gain_log_slope_y_luma = spatial_2d_affine_fit.log_gain_slope_y_luma;
        diagnostics.spatial_gain_top_rgb = center_top_gain;
        diagnostics.spatial_gain_bottom_rgb = center_bottom_gain;
        diagnostics.spatial_gain_corners_rgb = spatial_2d_affine_gain_corners;
        diagnostics.spatial_offset_slope_x_rgb = spatial_2d_affine_fit.offset_slope_x_rgb;
        diagnostics.spatial_offset_slope_x_luma = spatial_2d_affine_fit.offset_slope_x_luma;
        diagnostics.spatial_offset_slope_x_rgb_normalized = std::array::from_fn(|channel| {
            spatial_2d_affine_fit.offset_slope_x_rgb[channel] / max_value
        });
        diagnostics.spatial_offset_slope_y_rgb = spatial_2d_affine_fit.offset_slope_y_rgb;
        diagnostics.spatial_offset_slope_y_luma = spatial_2d_affine_fit.offset_slope_y_luma;
        diagnostics.spatial_offset_slope_y_rgb_normalized = std::array::from_fn(|channel| {
            spatial_2d_affine_fit.offset_slope_y_rgb[channel] / max_value
        });
        diagnostics.spatial_offset_top_rgb = center_top_offset;
        diagnostics.spatial_offset_bottom_rgb = center_bottom_offset;
        diagnostics.spatial_offset_top_rgb_normalized =
            std::array::from_fn(|channel| center_top_offset[channel] / max_value);
        diagnostics.spatial_offset_bottom_rgb_normalized =
            std::array::from_fn(|channel| center_bottom_offset[channel] / max_value);
        diagnostics.spatial_offset_corners_rgb = spatial_2d_affine_offset_corners;
        diagnostics.spatial_offset_corners_rgb_normalized = spatial_2d_affine_offset_corners
            .map(|corner| std::array::from_fn(|channel| corner[channel] / max_value));
        diagnostics.delta_luma_after = spatial_2d_affine_measurements.delta_luma;
        diagnostics.delta_rgb_after = spatial_2d_affine_measurements.delta_rgb;
        diagnostics.seam_score_after = spatial_2d_affine_measurements.seam_score;
        diagnostics.clipped_high_after = spatial_2d_affine_clipped_high;
        diagnostics.clipped_low_after = spatial_2d_affine_clipped_low;
        diagnostics.candidate_gain_rgb = spatial_2d_affine_fit.center_gain_rgb;
        diagnostics.candidate_gain_luma = spatial_2d_affine_fit.center_gain_luma;
        diagnostics.candidate_offset_rgb = spatial_2d_affine_fit.center_offset_rgb;
        diagnostics.candidate_offset_luma = spatial_2d_affine_fit.center_offset_luma;
        diagnostics.candidate_delta_luma_after = spatial_2d_affine_measurements.delta_luma;
        diagnostics.candidate_delta_rgb_after = spatial_2d_affine_measurements.delta_rgb;
        diagnostics.candidate_seam_score_after = spatial_2d_affine_measurements.seam_score;
        diagnostics.candidate_clipped_high_after = spatial_2d_affine_clipped_high;
        diagnostics.candidate_clipped_low_after = spatial_2d_affine_clipped_low;
        diagnostics.per_channel_gain_used = true;
        diagnostics.held_out_selected_seam_score = spatial_2d_affine_held_out.seam_score;
        diagnostics.held_out_improvement_over_identity = spatial_2d_affine_identity_improvement;
        diagnostics.held_out_improvement_over_gain =
            held_out_gain.seam_score - spatial_2d_affine_held_out.seam_score;
        diagnostics.held_out_validation_passed = true;
        return diagnostics;
    }

    if spatial_2d_gain_rejection.is_none() {
        diagnostics.model = "gain_spatial_xy_rgb".to_string();
        diagnostics.applied = true;
        diagnostics.reason = format!(
            "applied bounded overlap-derived 2D RGB gain field after disjoint validation; score {:.3} -> {:.3}, beating accepted {} by {:.3}",
            held_out_identity.seam_score,
            spatial_2d_gain_held_out.seam_score,
            spatial_2d_gain_best_simpler_model,
            spatial_2d_gain_simpler_improvement
        );
        diagnostics.gain_rgb = spatial_2d_gain_fit.center_gain_rgb;
        diagnostics.gain_luma = spatial_2d_gain_fit.center_gain_luma;
        diagnostics.spatial_gain_log_slope_x_rgb = spatial_2d_gain_fit.log_slope_x_rgb;
        diagnostics.spatial_gain_log_slope_x_luma = spatial_2d_gain_fit.log_slope_x_luma;
        diagnostics.spatial_gain_log_slope_y_rgb = spatial_2d_gain_fit.log_slope_y_rgb;
        diagnostics.spatial_gain_log_slope_y_luma = spatial_2d_gain_fit.log_slope_y_luma;
        diagnostics.spatial_gain_top_rgb = spatial_gain_2d_at(&spatial_2d_gain_fit, 0.0, -1.0);
        diagnostics.spatial_gain_bottom_rgb = spatial_gain_2d_at(&spatial_2d_gain_fit, 0.0, 1.0);
        diagnostics.spatial_gain_corners_rgb = spatial_2d_gain_corners;
        diagnostics.delta_luma_after = spatial_2d_gain_measurements.delta_luma;
        diagnostics.delta_rgb_after = spatial_2d_gain_measurements.delta_rgb;
        diagnostics.seam_score_after = spatial_2d_gain_measurements.seam_score;
        diagnostics.clipped_high_after = spatial_2d_gain_clipped_high;
        diagnostics.clipped_low_after = spatial_2d_gain_clipped_low;
        diagnostics.candidate_gain_rgb = spatial_2d_gain_fit.center_gain_rgb;
        diagnostics.candidate_gain_luma = spatial_2d_gain_fit.center_gain_luma;
        diagnostics.candidate_offset_rgb = zero_offset;
        diagnostics.candidate_offset_luma = 0.0;
        diagnostics.candidate_delta_luma_after = spatial_2d_gain_measurements.delta_luma;
        diagnostics.candidate_delta_rgb_after = spatial_2d_gain_measurements.delta_rgb;
        diagnostics.candidate_seam_score_after = spatial_2d_gain_measurements.seam_score;
        diagnostics.candidate_clipped_high_after = spatial_2d_gain_clipped_high;
        diagnostics.candidate_clipped_low_after = spatial_2d_gain_clipped_low;
        diagnostics.per_channel_gain_used = true;
        diagnostics.held_out_selected_seam_score = spatial_2d_gain_held_out.seam_score;
        diagnostics.held_out_improvement_over_identity = spatial_2d_gain_identity_improvement;
        diagnostics.held_out_improvement_over_gain =
            held_out_gain.seam_score - spatial_2d_gain_held_out.seam_score;
        diagnostics.held_out_validation_passed = true;
        return diagnostics;
    }

    if spatial_affine_rejection.is_none() {
        diagnostics.model = "gain_offset_spatial_y_rgb".to_string();
        diagnostics.applied = true;
        diagnostics.reason = format!(
            "applied bounded overlap-derived vertical RGB gain+offset field after held-out validation; score {:.3} -> {:.3}, beating accepted {} by {:.3}",
            held_out_identity.seam_score,
            spatial_affine_held_out.seam_score,
            best_simpler_model,
            spatial_affine_simpler_improvement
        );
        diagnostics.gain_rgb = spatial_affine_fit.center_gain_rgb;
        diagnostics.gain_luma = spatial_affine_fit.center_gain_luma;
        diagnostics.offset_rgb = spatial_affine_fit.center_offset_rgb;
        diagnostics.offset_luma = spatial_affine_fit.center_offset_luma;
        diagnostics.offset_rgb_normalized = std::array::from_fn(|channel| {
            spatial_affine_fit.center_offset_rgb[channel] / max_value
        });
        diagnostics.spatial_gain_log_slope_y_rgb = spatial_affine_fit.log_gain_slope_y_rgb;
        diagnostics.spatial_gain_log_slope_y_luma = spatial_affine_fit.log_gain_slope_y_luma;
        diagnostics.spatial_gain_top_rgb = spatial_affine_top_gain;
        diagnostics.spatial_gain_bottom_rgb = spatial_affine_bottom_gain;
        diagnostics.spatial_gain_corners_rgb = [
            spatial_affine_top_gain,
            spatial_affine_top_gain,
            spatial_affine_bottom_gain,
            spatial_affine_bottom_gain,
        ];
        diagnostics.spatial_offset_slope_y_rgb = spatial_affine_fit.offset_slope_y_rgb;
        diagnostics.spatial_offset_slope_y_luma = spatial_affine_fit.offset_slope_y_luma;
        diagnostics.spatial_offset_slope_y_rgb_normalized = std::array::from_fn(|channel| {
            spatial_affine_fit.offset_slope_y_rgb[channel] / max_value
        });
        diagnostics.spatial_offset_top_rgb = spatial_affine_top_offset;
        diagnostics.spatial_offset_bottom_rgb = spatial_affine_bottom_offset;
        diagnostics.spatial_offset_top_rgb_normalized =
            std::array::from_fn(|channel| spatial_affine_top_offset[channel] / max_value);
        diagnostics.spatial_offset_bottom_rgb_normalized =
            std::array::from_fn(|channel| spatial_affine_bottom_offset[channel] / max_value);
        diagnostics.spatial_offset_corners_rgb = [
            spatial_affine_top_offset,
            spatial_affine_top_offset,
            spatial_affine_bottom_offset,
            spatial_affine_bottom_offset,
        ];
        diagnostics.spatial_offset_corners_rgb_normalized = [
            diagnostics.spatial_offset_top_rgb_normalized,
            diagnostics.spatial_offset_top_rgb_normalized,
            diagnostics.spatial_offset_bottom_rgb_normalized,
            diagnostics.spatial_offset_bottom_rgb_normalized,
        ];
        diagnostics.delta_luma_after = spatial_affine_measurements.delta_luma;
        diagnostics.delta_rgb_after = spatial_affine_measurements.delta_rgb;
        diagnostics.seam_score_after = spatial_affine_measurements.seam_score;
        diagnostics.clipped_high_after = spatial_affine_clipped_high;
        diagnostics.clipped_low_after = spatial_affine_clipped_low;
        diagnostics.candidate_gain_rgb = spatial_affine_fit.center_gain_rgb;
        diagnostics.candidate_gain_luma = spatial_affine_fit.center_gain_luma;
        diagnostics.candidate_offset_rgb = spatial_affine_fit.center_offset_rgb;
        diagnostics.candidate_offset_luma = spatial_affine_fit.center_offset_luma;
        diagnostics.candidate_delta_luma_after = spatial_affine_measurements.delta_luma;
        diagnostics.candidate_delta_rgb_after = spatial_affine_measurements.delta_rgb;
        diagnostics.candidate_seam_score_after = spatial_affine_measurements.seam_score;
        diagnostics.candidate_clipped_high_after = spatial_affine_clipped_high;
        diagnostics.candidate_clipped_low_after = spatial_affine_clipped_low;
        diagnostics.per_channel_gain_used = true;
        diagnostics.held_out_selected_seam_score = spatial_affine_held_out.seam_score;
        diagnostics.held_out_improvement_over_identity = spatial_affine_identity_improvement;
        diagnostics.held_out_improvement_over_gain =
            held_out_gain.seam_score - spatial_affine_held_out.seam_score;
        diagnostics.held_out_validation_passed = true;
        return diagnostics;
    }

    if spatial_rejection.is_none() {
        diagnostics.model = "gain_spatial_y_rgb".to_string();
        diagnostics.applied = true;
        diagnostics.reason = format!(
            "applied bounded overlap-derived vertical RGB gain field after held-out validation; score {:.3} -> {:.3}, beating accepted {} by {:.3}",
            held_out_identity.seam_score,
            spatial_held_out.seam_score,
            best_constant_model,
            spatial_best_constant_improvement
        );
        diagnostics.gain_rgb = spatial_fit.center_gain_rgb;
        diagnostics.gain_luma = spatial_fit.center_gain_luma;
        diagnostics.spatial_gain_log_slope_y_rgb = spatial_fit.log_slope_y_rgb;
        diagnostics.spatial_gain_log_slope_y_luma = spatial_fit.log_slope_y_luma;
        diagnostics.spatial_gain_top_rgb = spatial_top_gain;
        diagnostics.spatial_gain_bottom_rgb = spatial_bottom_gain;
        diagnostics.spatial_gain_corners_rgb = [
            spatial_top_gain,
            spatial_top_gain,
            spatial_bottom_gain,
            spatial_bottom_gain,
        ];
        diagnostics.delta_luma_after = spatial_measurements.delta_luma;
        diagnostics.delta_rgb_after = spatial_measurements.delta_rgb;
        diagnostics.seam_score_after = spatial_measurements.seam_score;
        diagnostics.clipped_high_after = spatial_clipped_high;
        diagnostics.clipped_low_after = spatial_clipped_low;
        diagnostics.candidate_gain_rgb = spatial_fit.center_gain_rgb;
        diagnostics.candidate_gain_luma = spatial_fit.center_gain_luma;
        diagnostics.candidate_offset_rgb = zero_offset;
        diagnostics.candidate_offset_luma = 0.0;
        diagnostics.candidate_delta_luma_after = spatial_measurements.delta_luma;
        diagnostics.candidate_delta_rgb_after = spatial_measurements.delta_rgb;
        diagnostics.candidate_seam_score_after = spatial_measurements.seam_score;
        diagnostics.candidate_clipped_high_after = spatial_clipped_high;
        diagnostics.candidate_clipped_low_after = spatial_clipped_low;
        diagnostics.per_channel_gain_used = true;
        diagnostics.held_out_selected_seam_score = spatial_held_out.seam_score;
        diagnostics.held_out_improvement_over_identity = spatial_identity_improvement;
        diagnostics.held_out_improvement_over_gain =
            held_out_gain.seam_score - spatial_held_out.seam_score;
        diagnostics.held_out_validation_passed = true;
        return diagnostics;
    }

    if affine_rejection.is_none() {
        diagnostics.model = "gain_offset_rgb".to_string();
        diagnostics.applied = true;
        diagnostics.reason = format!(
            "applied bounded overlap-derived gain+offset model after held-out validation; score {:.3} -> {:.3}, beating gain-only by {:.3}",
            affine_held_out_identity.seam_score,
            affine_held_out.seam_score,
            affine_gain_improvement
        );
        diagnostics.gain_rgb = affine_gain_rgb;
        diagnostics.gain_luma = affine_gain_luma;
        diagnostics.spatial_gain_top_rgb = affine_gain_rgb;
        diagnostics.spatial_gain_bottom_rgb = affine_gain_rgb;
        diagnostics.spatial_gain_corners_rgb = [affine_gain_rgb; 4];
        diagnostics.offset_rgb = affine_offset_rgb;
        diagnostics.offset_luma = affine_offset_luma;
        diagnostics.offset_rgb_normalized =
            std::array::from_fn(|channel| affine_offset_rgb[channel] / max_value);
        diagnostics.spatial_offset_top_rgb = affine_offset_rgb;
        diagnostics.spatial_offset_bottom_rgb = affine_offset_rgb;
        diagnostics.spatial_offset_top_rgb_normalized = diagnostics.offset_rgb_normalized;
        diagnostics.spatial_offset_bottom_rgb_normalized = diagnostics.offset_rgb_normalized;
        diagnostics.spatial_offset_corners_rgb = [affine_offset_rgb; 4];
        diagnostics.spatial_offset_corners_rgb_normalized = [diagnostics.offset_rgb_normalized; 4];
        diagnostics.delta_luma_after = affine_measurements.delta_luma;
        diagnostics.delta_rgb_after = affine_measurements.delta_rgb;
        diagnostics.seam_score_after = affine_measurements.seam_score;
        diagnostics.clipped_high_after = affine_clipped_high;
        diagnostics.clipped_low_after = affine_clipped_low;
        diagnostics.candidate_gain_rgb = affine_gain_rgb;
        diagnostics.candidate_gain_luma = affine_gain_luma;
        diagnostics.candidate_offset_rgb = affine_offset_rgb;
        diagnostics.candidate_offset_luma = affine_offset_luma;
        diagnostics.candidate_delta_luma_after = affine_measurements.delta_luma;
        diagnostics.candidate_delta_rgb_after = affine_measurements.delta_rgb;
        diagnostics.candidate_seam_score_after = affine_measurements.seam_score;
        diagnostics.candidate_clipped_high_after = affine_clipped_high;
        diagnostics.candidate_clipped_low_after = affine_clipped_low;
        diagnostics.per_channel_gain_used = true;
        diagnostics.held_out_selected_seam_score = affine_held_out.seam_score;
        diagnostics.held_out_improvement_over_identity = affine_identity_improvement;
        diagnostics.held_out_validation_passed = true;
        return diagnostics;
    }

    if gain_rejection.is_none() {
        diagnostics.model = if use_per_channel {
            "gain_only_rgb".to_string()
        } else {
            "gain_only_scalar".to_string()
        };
        diagnostics.applied = true;
        diagnostics.reason = format!(
            "applied {} overlap gain after held-out validation; score {:.3} -> {:.3}",
            if use_per_channel {
                "stable per-channel"
            } else {
                "luma-only"
            },
            held_out_identity.seam_score,
            held_out_gain.seam_score
        );
        diagnostics.gain_rgb = gain_candidate_rgb;
        diagnostics.gain_luma = gain_candidate_luma;
        diagnostics.spatial_gain_top_rgb = gain_candidate_rgb;
        diagnostics.spatial_gain_bottom_rgb = gain_candidate_rgb;
        diagnostics.spatial_gain_corners_rgb = [gain_candidate_rgb; 4];
        diagnostics.delta_luma_after = gain_measurements.delta_luma;
        diagnostics.delta_rgb_after = gain_measurements.delta_rgb;
        diagnostics.seam_score_after = gain_measurements.seam_score;
        diagnostics.clipped_high_after = gain_clipped_high;
        diagnostics.clipped_low_after = gain_clipped_low;
        diagnostics.candidate_gain_rgb = gain_candidate_rgb;
        diagnostics.candidate_gain_luma = gain_candidate_luma;
        diagnostics.candidate_offset_rgb = zero_offset;
        diagnostics.candidate_offset_luma = 0.0;
        diagnostics.candidate_delta_luma_after = gain_measurements.delta_luma;
        diagnostics.candidate_delta_rgb_after = gain_measurements.delta_rgb;
        diagnostics.candidate_seam_score_after = gain_measurements.seam_score;
        diagnostics.candidate_clipped_high_after = gain_clipped_high;
        diagnostics.candidate_clipped_low_after = gain_clipped_low;
        diagnostics.held_out_selected_seam_score = held_out_gain.seam_score;
        diagnostics.held_out_improvement_over_identity =
            held_out_identity.seam_score - held_out_gain.seam_score;
        diagnostics.held_out_validation_passed = true;
    }

    diagnostics
}

/// Apply a per-channel affine photometric correction, returning a new owned copy.
///
/// Non-zero spatial slopes are referenced to the measured overlap rectangle. Coordinates outside
/// that support clamp to its nearest endpoint instead of extrapolating an unmeasured field.
#[allow(clippy::too_many_arguments)]
fn apply_photometric_correction(
    img: &Array3<u16>,
    field: &SpatialPhotometricField,
    reference_x_origin: usize,
    reference_width: usize,
    reference_y_origin: usize,
    reference_height: usize,
    max_value: f64,
) -> Array3<u16> {
    let mut out = img.clone();
    let (h, w, _) = out.dim();
    let horizontal_field_is_constant = !spatial_field_has_horizontal_variation(field);
    for y in 0..h {
        let y_normalized = if reference_height <= 1 {
            0.0
        } else {
            (2.0 * (y as f64 - reference_y_origin as f64) / (reference_height - 1) as f64 - 1.0)
                .clamp(-1.0, 1.0)
        };
        let row_field = horizontal_field_is_constant
            .then(|| spatial_photometric_at_normalized_xy(field, 0.0, y_normalized));
        for x in 0..w {
            let (gain, offset) = match row_field {
                Some(field_value) => field_value,
                None => {
                    let x_normalized = if reference_width <= 1 {
                        0.0
                    } else {
                        (2.0 * (x as f64 - reference_x_origin as f64)
                            / (reference_width - 1) as f64
                            - 1.0)
                            .clamp(-1.0, 1.0)
                    };
                    spatial_photometric_at_normalized_xy(field, x_normalized, y_normalized)
                }
            };
            for c in 0..3 {
                let val = out[[y, x, c]] as f64 * gain[c] + offset[c];
                out[[y, x, c]] = val.round().clamp(0.0, max_value) as u16;
            }
        }
    }
    out
}

fn overlap_luma(img: &Array3<u16>, y: usize, x: usize) -> f64 {
    0.2126 * img[[y, x, 0]] as f64 + 0.7152 * img[[y, x, 1]] as f64 + 0.0722 * img[[y, x, 2]] as f64
}

fn seam_cost_map(left: &Array3<u16>, right: &Array3<u16>, max_value: f64) -> Array2<f64> {
    let (h, w, _) = left.dim();
    let mut costs = Array2::<f64>::zeros((h, w));
    let scale = max_value.max(1.0);
    for y in 0..h {
        for x in 0..w {
            let color_difference = (0..3)
                .map(|channel| {
                    (left[[y, x, channel]] as f64 - right[[y, x, channel]] as f64).abs() / scale
                })
                .sum::<f64>()
                / 3.0;
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(w.saturating_sub(1));
            let left_gradient = (overlap_luma(left, y, x1) - overlap_luma(left, y, x0)).abs()
                / (scale * (x1.saturating_sub(x0)).max(1) as f64);
            let right_gradient = (overlap_luma(right, y, x1) - overlap_luma(right, y, x0)).abs()
                / (scale * (x1.saturating_sub(x0)).max(1) as f64);
            let edge_energy = left_gradient.max(right_gradient);
            let gradient_disagreement = (left_gradient - right_gradient).abs();
            let edge_distance = x.min(w.saturating_sub(1).saturating_sub(x));
            let edge_margin = (w / 12).clamp(2, 16).min(w.saturating_sub(1) / 2);
            let boundary_penalty = if edge_margin == 0 || edge_distance >= edge_margin {
                0.0
            } else {
                0.12 * (1.0 - edge_distance as f64 / edge_margin as f64)
            };
            let centered = if w <= 1 {
                0.0
            } else {
                ((x as f64 / (w - 1) as f64) - 0.5).abs() * 2.0
            };
            costs[[y, x]] = (color_difference
                + 0.15 * gradient_disagreement
                + 0.08 * edge_energy
                + 0.015 * centered
                + boundary_penalty)
                .clamp(0.0, 1.0);
        }
    }
    costs
}

fn minimum_cost_vertical_seam(costs: &Array2<f64>) -> Vec<usize> {
    let (h, w) = costs.dim();
    if h == 0 || w == 0 {
        return Vec::new();
    }
    let mut previous = (0..w).map(|x| costs[[0, x]]).collect::<Vec<_>>();
    let mut current = vec![0.0f64; w];
    let mut predecessor_delta = vec![0i8; h * w];
    for y in 1..h {
        for x in 0..w {
            let mut best_cost = f64::INFINITY;
            let mut best_delta = 0i8;
            for delta in -2isize..=2 {
                let predecessor = x as isize + delta;
                if predecessor < 0 || predecessor >= w as isize {
                    continue;
                }
                let transition_penalty = delta.unsigned_abs() as f64 * 0.006;
                let candidate = previous[predecessor as usize] + transition_penalty;
                if candidate < best_cost {
                    best_cost = candidate;
                    best_delta = delta as i8;
                }
            }
            current[x] = costs[[y, x]] + best_cost;
            predecessor_delta[y * w + x] = best_delta;
        }
        std::mem::swap(&mut previous, &mut current);
    }

    let mut x = previous
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(w / 2);
    let mut seam = vec![x; h];
    for y in (1..h).rev() {
        let delta = predecessor_delta[y * w + x] as isize;
        x = (x as isize + delta).clamp(0, w.saturating_sub(1) as isize) as usize;
        seam[y - 1] = x;
    }
    seam
}

fn box_blur_array2(input: &Array2<f64>, radius: usize) -> Array2<f64> {
    if radius == 0 || input.is_empty() {
        return input.clone();
    }
    let (h, w) = input.dim();
    let mut horizontal = Array2::<f64>::zeros((h, w));
    for y in 0..h {
        let mut sum = 0.0;
        let mut count = 0usize;
        for x in 0..w {
            let add = x + radius;
            if add < w {
                sum += input[[y, add]];
                count += 1;
            }
            if x == 0 {
                for initial in 0..radius.min(w) {
                    sum += input[[y, initial]];
                    count += 1;
                }
            } else {
                let remove = x.saturating_sub(radius + 1);
                if x > radius {
                    sum -= input[[y, remove]];
                    count = count.saturating_sub(1);
                }
            }
            horizontal[[y, x]] = sum / count.max(1) as f64;
        }
    }

    let mut output = Array2::<f64>::zeros((h, w));
    for x in 0..w {
        let mut sum = 0.0;
        let mut count = 0usize;
        for y in 0..h {
            let add = y + radius;
            if add < h {
                sum += horizontal[[add, x]];
                count += 1;
            }
            if y == 0 {
                for initial in 0..radius.min(h) {
                    sum += horizontal[[initial, x]];
                    count += 1;
                }
            } else {
                let remove = y.saturating_sub(radius + 1);
                if y > radius {
                    sum -= horizontal[[remove, x]];
                    count = count.saturating_sub(1);
                }
            }
            output[[y, x]] = sum / count.max(1) as f64;
        }
    }
    output
}

fn box_blur_array3(input: &Array3<f64>, radius: usize) -> Array3<f64> {
    if radius == 0 || input.is_empty() {
        return input.clone();
    }
    let (h, w, channels) = input.dim();
    let mut horizontal = Array3::<f64>::zeros((h, w, channels));
    for y in 0..h {
        for channel in 0..channels {
            let mut sum = 0.0;
            let mut count = 0usize;
            for x in 0..w {
                let add = x + radius;
                if add < w {
                    sum += input[[y, add, channel]];
                    count += 1;
                }
                if x == 0 {
                    for initial in 0..radius.min(w) {
                        sum += input[[y, initial, channel]];
                        count += 1;
                    }
                } else {
                    let remove = x.saturating_sub(radius + 1);
                    if x > radius {
                        sum -= input[[y, remove, channel]];
                        count = count.saturating_sub(1);
                    }
                }
                horizontal[[y, x, channel]] = sum / count.max(1) as f64;
            }
        }
    }

    let mut output = Array3::<f64>::zeros((h, w, channels));
    for x in 0..w {
        for channel in 0..channels {
            let mut sum = 0.0;
            let mut count = 0usize;
            for y in 0..h {
                let add = y + radius;
                if add < h {
                    sum += horizontal[[add, x, channel]];
                    count += 1;
                }
                if y == 0 {
                    for initial in 0..radius.min(h) {
                        sum += horizontal[[initial, x, channel]];
                        count += 1;
                    }
                } else {
                    let remove = y.saturating_sub(radius + 1);
                    if y > radius {
                        sum -= horizontal[[remove, x, channel]];
                        count = count.saturating_sub(1);
                    }
                }
                output[[y, x, channel]] = sum / count.max(1) as f64;
            }
        }
    }
    output
}

fn seam_blend_radii(transition_width: usize) -> Vec<usize> {
    if transition_width < 2 {
        return Vec::new();
    }
    let mut radii = Vec::new();
    let mut radius = 1usize;
    while radii.len() < 5 && radius <= (transition_width / 2).max(1) {
        radii.push(radius);
        radius *= 2;
    }
    radii
}

fn multiband_blend_roi(
    left: &Array3<f64>,
    right: &Array3<f64>,
    mask: &Array2<f64>,
    radii: &[usize],
) -> Array3<f64> {
    let (h, w, channels) = left.dim();
    let mut current_left = left.clone();
    let mut current_right = right.clone();
    let mut current_mask = mask.clone();
    let mut output = Array3::<f64>::zeros((h, w, channels));

    for &radius in radii {
        let blurred_left = box_blur_array3(&current_left, radius);
        let blurred_right = box_blur_array3(&current_right, radius);
        for y in 0..h {
            for x in 0..w {
                let alpha = current_mask[[y, x]].clamp(0.0, 1.0);
                for channel in 0..channels {
                    let left_band = current_left[[y, x, channel]] - blurred_left[[y, x, channel]];
                    let right_band =
                        current_right[[y, x, channel]] - blurred_right[[y, x, channel]];
                    output[[y, x, channel]] += left_band * (1.0 - alpha) + right_band * alpha;
                }
            }
        }
        current_left = blurred_left;
        current_right = blurred_right;
        current_mask = box_blur_array2(&current_mask, radius);
    }

    for y in 0..h {
        for x in 0..w {
            let alpha = current_mask[[y, x]].clamp(0.0, 1.0);
            for channel in 0..channels {
                output[[y, x, channel]] += current_left[[y, x, channel]] * (1.0 - alpha)
                    + current_right[[y, x, channel]] * alpha;
            }
        }
    }
    output
}

#[derive(Debug, Clone, Copy, Default)]
struct SeamDetailPartitionStats {
    window_count: usize,
    left_energy_median: f64,
    right_energy_median: f64,
    median_log_ratio: f64,
    right_to_left_ratio: f64,
    symmetric_ratio: f64,
    direction_consistency: f64,
}

fn seam_detail_partition_stats(windows: &[[f64; 3]]) -> SeamDetailPartitionStats {
    if windows.is_empty() {
        return SeamDetailPartitionStats {
            right_to_left_ratio: 1.0,
            symmetric_ratio: 1.0,
            ..SeamDetailPartitionStats::default()
        };
    }
    let left_energy_median =
        nearest_rank_percentile(windows.iter().map(|window| window[0]).collect(), 0.5);
    let right_energy_median =
        nearest_rank_percentile(windows.iter().map(|window| window[1]).collect(), 0.5);
    let median_log_ratio =
        nearest_rank_percentile(windows.iter().map(|window| window[2]).collect(), 0.5)
            .clamp(-10.0, 10.0);
    let direction_consistency = windows
        .iter()
        .filter(|window| {
            if median_log_ratio >= 0.0 {
                window[2] >= 0.0
            } else {
                window[2] <= 0.0
            }
        })
        .count() as f64
        / windows.len() as f64;
    let right_to_left_ratio = median_log_ratio.exp().clamp(0.001, 1_000.0);
    let symmetric_ratio = median_log_ratio.abs().exp().clamp(1.0, 1_000.0);
    SeamDetailPartitionStats {
        window_count: windows.len(),
        left_energy_median,
        right_energy_median,
        median_log_ratio,
        right_to_left_ratio,
        symmetric_ratio,
        direction_consistency,
    }
}

fn unsupported_seam_detail_scale(
    radius_px: usize,
    reason: impl Into<String>,
) -> SeamDetailScaleDiagnostics {
    SeamDetailScaleDiagnostics {
        radius_px,
        training_window_count: 0,
        held_out_window_count: 0,
        training_left_energy_median: 0.0,
        training_right_energy_median: 0.0,
        held_out_left_energy_median: 0.0,
        held_out_right_energy_median: 0.0,
        training_right_to_left_ratio: 1.0,
        held_out_right_to_left_ratio: 1.0,
        training_symmetric_energy_ratio: 1.0,
        held_out_symmetric_energy_ratio: 1.0,
        training_direction_consistent_window_ratio: 0.0,
        held_out_direction_consistent_window_ratio: 0.0,
        cross_split_log_ratio_delta: 0.0,
        cross_split_direction_agrees: false,
        evidence_supported: false,
        imbalanced: false,
        reason: reason.into(),
    }
}

fn seam_detail_consistency(
    left: &Array3<u16>,
    right: &Array3<u16>,
    max_value: f64,
) -> SeamDetailConsistencyDiagnostics {
    let (height, width, channels) = left.dim();
    let mut scales = Vec::with_capacity(SEAM_DETAIL_RADII_PX.len());
    if left.dim() != right.dim() || height == 0 || width == 0 || channels < 3 {
        for radius_px in SEAM_DETAIL_RADII_PX {
            scales.push(unsupported_seam_detail_scale(
                radius_px,
                "aligned overlap dimensions are unavailable",
            ));
        }
        return SeamDetailConsistencyDiagnostics {
            method: "multiscale_luma_highpass_energy_disjoint_checkerboard_windows",
            evaluated: false,
            reason: "aligned overlap strips were empty or dimensionally inconsistent".to_string(),
            grid_rows: SEAM_DETAIL_GRID_ROWS,
            grid_columns: SEAM_DETAIL_GRID_COLUMNS,
            minimum_energy: SEAM_DETAIL_MIN_ENERGY,
            energy_regularization: SEAM_DETAIL_ENERGY_REGULARIZATION,
            review_ratio_threshold: SEAM_DETAIL_REVIEW_RATIO,
            minimum_direction_consistency: SEAM_DETAIL_MIN_DIRECTION_CONSISTENCY,
            maximum_cross_split_ratio: SEAM_DETAIL_MAX_CROSS_SPLIT_RATIO,
            minimum_repeated_scale_count: SEAM_DETAIL_MIN_REPEATED_SCALES,
            supported_scale_count: 0,
            decision_supported: false,
            imbalanced_scale_count: 0,
            maximum_symmetric_energy_ratio: 1.0,
            review_required: true,
            review_reason:
                "detail continuity could not be evaluated because the aligned overlap was invalid"
                    .to_string(),
            scales,
        };
    }

    let scale = max_value.max(1.0);
    let mut left_luma = Array2::<f64>::zeros((height, width));
    let mut right_luma = Array2::<f64>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            left_luma[[y, x]] = overlap_luma(left, y, x) / scale;
            right_luma[[y, x]] = overlap_luma(right, y, x) / scale;
        }
    }

    let mut supported_scale_count = 0usize;
    let mut imbalanced_scale_count = 0usize;
    let mut maximum_symmetric_energy_ratio = 1.0f64;
    for radius_px in SEAM_DETAIL_RADII_PX {
        let margin = radius_px.saturating_add(1);
        if height <= margin.saturating_mul(2) || width <= margin.saturating_mul(2) {
            scales.push(unsupported_seam_detail_scale(
                radius_px,
                "overlap is too small for this spatial-frequency radius",
            ));
            continue;
        }
        let left_blurred = box_blur_array2(&left_luma, radius_px);
        let right_blurred = box_blur_array2(&right_luma, radius_px);
        let mut training_windows = Vec::<[f64; 3]>::new();
        let mut held_out_windows = Vec::<[f64; 3]>::new();
        for row in 0..SEAM_DETAIL_GRID_ROWS {
            let y_start = (row * height / SEAM_DETAIL_GRID_ROWS).max(margin);
            let y_end =
                ((row + 1) * height / SEAM_DETAIL_GRID_ROWS).min(height.saturating_sub(margin));
            for column in 0..SEAM_DETAIL_GRID_COLUMNS {
                let x_start = (column * width / SEAM_DETAIL_GRID_COLUMNS).max(margin);
                let x_end = ((column + 1) * width / SEAM_DETAIL_GRID_COLUMNS)
                    .min(width.saturating_sub(margin));
                if y_end <= y_start || x_end <= x_start {
                    continue;
                }
                let pixel_count = (y_end - y_start).saturating_mul(x_end - x_start);
                if pixel_count < SEAM_DETAIL_MIN_PIXELS_PER_WINDOW {
                    continue;
                }
                let mut left_squared_energy = 0.0f64;
                let mut right_squared_energy = 0.0f64;
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        let left_residual = left_luma[[y, x]] - left_blurred[[y, x]];
                        let right_residual = right_luma[[y, x]] - right_blurred[[y, x]];
                        left_squared_energy += left_residual * left_residual;
                        right_squared_energy += right_residual * right_residual;
                    }
                }
                let left_energy = (left_squared_energy / pixel_count as f64).sqrt();
                let right_energy = (right_squared_energy / pixel_count as f64).sqrt();
                if left_energy.max(right_energy) < SEAM_DETAIL_MIN_ENERGY {
                    continue;
                }
                let log_ratio = ((right_energy + SEAM_DETAIL_ENERGY_REGULARIZATION)
                    / (left_energy + SEAM_DETAIL_ENERGY_REGULARIZATION))
                    .ln()
                    .clamp(-10.0, 10.0);
                let window = [left_energy, right_energy, log_ratio];
                if (row + column) % 2 == 0 {
                    training_windows.push(window);
                } else {
                    held_out_windows.push(window);
                }
            }
        }

        let training = seam_detail_partition_stats(&training_windows);
        let held_out = seam_detail_partition_stats(&held_out_windows);
        let evidence_supported = training.window_count >= SEAM_DETAIL_MIN_WINDOWS_PER_SPLIT
            && held_out.window_count >= SEAM_DETAIL_MIN_WINDOWS_PER_SPLIT;
        let cross_split_direction_agrees =
            training.median_log_ratio * held_out.median_log_ratio > 0.0;
        let cross_split_log_ratio_delta =
            (training.median_log_ratio - held_out.median_log_ratio).abs();
        let cross_split_ratio_agrees =
            cross_split_log_ratio_delta <= SEAM_DETAIL_MAX_CROSS_SPLIT_RATIO.ln();
        let imbalanced = evidence_supported
            && training.symmetric_ratio >= SEAM_DETAIL_REVIEW_RATIO
            && held_out.symmetric_ratio >= SEAM_DETAIL_REVIEW_RATIO
            && training.direction_consistency >= SEAM_DETAIL_MIN_DIRECTION_CONSISTENCY
            && held_out.direction_consistency >= SEAM_DETAIL_MIN_DIRECTION_CONSISTENCY
            && cross_split_direction_agrees
            && cross_split_ratio_agrees;
        if evidence_supported {
            supported_scale_count += 1;
            maximum_symmetric_energy_ratio = maximum_symmetric_energy_ratio
                .max(training.symmetric_ratio)
                .max(held_out.symmetric_ratio);
        }
        if imbalanced {
            imbalanced_scale_count += 1;
        }
        let reason = if !evidence_supported {
            format!(
                "needs at least {SEAM_DETAIL_MIN_WINDOWS_PER_SPLIT} signal-rich windows in each checkerboard split; found {} training and {} held-out",
                training.window_count, held_out.window_count
            )
        } else if training.symmetric_ratio < SEAM_DETAIL_REVIEW_RATIO
            || held_out.symmetric_ratio < SEAM_DETAIL_REVIEW_RATIO
        {
            "both disjoint splits do not exceed the review ratio".to_string()
        } else if training.direction_consistency < SEAM_DETAIL_MIN_DIRECTION_CONSISTENCY
            || held_out.direction_consistency < SEAM_DETAIL_MIN_DIRECTION_CONSISTENCY
        {
            "the detail-energy direction is not spatially consistent within both splits".to_string()
        } else if !cross_split_direction_agrees {
            "training and held-out splits disagree on which component has more detail energy"
                .to_string()
        } else if !cross_split_ratio_agrees {
            "training and held-out detail-energy ratios disagree beyond the bounded cross-split ratio"
                .to_string()
        } else {
            "both disjoint splits reproduce a spatially consistent detail-energy imbalance"
                .to_string()
        };
        scales.push(SeamDetailScaleDiagnostics {
            radius_px,
            training_window_count: training.window_count,
            held_out_window_count: held_out.window_count,
            training_left_energy_median: training.left_energy_median,
            training_right_energy_median: training.right_energy_median,
            held_out_left_energy_median: held_out.left_energy_median,
            held_out_right_energy_median: held_out.right_energy_median,
            training_right_to_left_ratio: training.right_to_left_ratio,
            held_out_right_to_left_ratio: held_out.right_to_left_ratio,
            training_symmetric_energy_ratio: training.symmetric_ratio,
            held_out_symmetric_energy_ratio: held_out.symmetric_ratio,
            training_direction_consistent_window_ratio: training.direction_consistency,
            held_out_direction_consistent_window_ratio: held_out.direction_consistency,
            cross_split_log_ratio_delta,
            cross_split_direction_agrees,
            evidence_supported,
            imbalanced,
            reason,
        });
    }

    let evaluated = supported_scale_count > 0;
    let decision_supported = supported_scale_count >= SEAM_DETAIL_MIN_REPEATED_SCALES;
    let repeated_imbalance = imbalanced_scale_count >= SEAM_DETAIL_MIN_REPEATED_SCALES;
    let review_required = !decision_supported || repeated_imbalance;
    let reason = if !decision_supported {
        "insufficient signal-rich checkerboard support for a detail-continuity decision".to_string()
    } else if repeated_imbalance {
        format!(
            "{imbalanced_scale_count} of {supported_scale_count} supported frequency scales independently reproduce a cross-scan detail-energy imbalance"
        )
    } else {
        format!(
            "only {imbalanced_scale_count} of {supported_scale_count} supported frequency scales reproduce a qualifying imbalance"
        )
    };
    let review_reason = if !decision_supported {
        format!(
            "only {supported_scale_count} spatial-frequency scale(s) had disjoint signal support; at least {SEAM_DETAIL_MIN_REPEATED_SCALES} are required to validate seam detail continuity"
        )
    } else if repeated_imbalance {
        format!(
            "overlap detail/focus/noise energy differs by at least {:.2}x across both disjoint spatial partitions on {imbalanced_scale_count} repeated frequency scales; the stitch needs visual geometry review",
            SEAM_DETAIL_REVIEW_RATIO
        )
    } else if evaluated {
        "multi-scale disjoint overlap evidence does not show a repeated severe detail-energy transition"
            .to_string()
    } else {
        "detail evidence was insufficient, but no severe mismatch was inferred".to_string()
    };
    SeamDetailConsistencyDiagnostics {
        method: "multiscale_luma_highpass_energy_disjoint_checkerboard_windows",
        evaluated,
        reason,
        grid_rows: SEAM_DETAIL_GRID_ROWS,
        grid_columns: SEAM_DETAIL_GRID_COLUMNS,
        minimum_energy: SEAM_DETAIL_MIN_ENERGY,
        energy_regularization: SEAM_DETAIL_ENERGY_REGULARIZATION,
        review_ratio_threshold: SEAM_DETAIL_REVIEW_RATIO,
        minimum_direction_consistency: SEAM_DETAIL_MIN_DIRECTION_CONSISTENCY,
        maximum_cross_split_ratio: SEAM_DETAIL_MAX_CROSS_SPLIT_RATIO,
        minimum_repeated_scale_count: SEAM_DETAIL_MIN_REPEATED_SCALES,
        supported_scale_count,
        decision_supported,
        imbalanced_scale_count,
        maximum_symmetric_energy_ratio,
        review_required,
        review_reason,
        scales,
    }
}

fn seam_gradient_percentiles(
    left: &Array3<u16>,
    right: &Array3<u16>,
    output: &Array3<u16>,
    seam: &[usize],
    max_value: f64,
) -> (f64, f64) {
    let (h, w, _) = output.dim();
    let mut output_gradients = Vec::with_capacity(h * 3);
    let mut source_gradients = Vec::with_capacity(h * 3);
    for y in 0..h.min(seam.len()) {
        if w < 3 {
            continue;
        }
        let x = seam[y].clamp(1, w - 2);
        for channel in 0..3 {
            output_gradients.push(
                (output[[y, x + 1, channel]] as f64 - output[[y, x - 1, channel]] as f64).abs()
                    / max_value.max(1.0),
            );
            let left_gradient =
                (left[[y, x + 1, channel]] as f64 - left[[y, x - 1, channel]] as f64).abs()
                    / max_value.max(1.0);
            let right_gradient =
                (right[[y, x + 1, channel]] as f64 - right[[y, x - 1, channel]] as f64).abs()
                    / max_value.max(1.0);
            source_gradients.push(left_gradient.max(right_gradient));
        }
    }
    (
        nearest_rank_percentile(output_gradients, 0.95),
        nearest_rank_percentile(source_gradients, 0.95),
    )
}

fn seam_aware_multiband_blend(
    left: &Array3<u16>,
    right: &Array3<u16>,
    config: &StitchConfig,
) -> (Array3<u16>, SeamBlendDiagnostics) {
    let (h, w, channels) = left.dim();
    let max_value = stitch_sample_max(config);
    let detail_consistency = seam_detail_consistency(left, right, max_value);
    if left.dim() != right.dim() || h == 0 || w == 0 || channels < 3 {
        let review_reason =
            "seam blending and detail-continuity validation could not run on the aligned overlap"
                .to_string();
        return (
            left.clone(),
            SeamBlendDiagnostics {
                mode: "seam_aware_multiband".to_string(),
                applied: false,
                reason: "aligned overlap strips were empty or dimensionally inconsistent"
                    .to_string(),
                overlap_width_px: w,
                overlap_height_px: h,
                transition_width_px: 0,
                pyramid_levels: 0,
                pyramid_radii_px: Vec::new(),
                processing_roi_x: [0, w],
                seam_path_min_x: 0,
                seam_path_max_x: 0,
                seam_path_mean_x: 0.0,
                seam_path_mean_normalized_cost: 0.0,
                seam_path_p95_normalized_cost: 0.0,
                overlap_mean_abs_difference: 0.0,
                overlap_p95_abs_difference: 0.0,
                output_seam_gradient_p95: 0.0,
                source_seam_gradient_p95: 0.0,
                output_to_source_seam_gradient_ratio: 0.0,
                detail_consistency,
                review_required: true,
                review_reason,
            },
        );
    }

    let costs = seam_cost_map(left, right, max_value);
    let seam = minimum_cost_vertical_seam(&costs);
    let seam_min = seam.iter().copied().min().unwrap_or(w / 2);
    let seam_max = seam.iter().copied().max().unwrap_or(w / 2);
    let seam_mean = seam.iter().sum::<usize>() as f64 / seam.len().max(1) as f64;
    let seam_costs = seam
        .iter()
        .enumerate()
        .map(|(y, &x)| costs[[y, x]])
        .collect::<Vec<_>>();
    let seam_cost_mean = seam_costs.iter().sum::<f64>() / seam_costs.len().max(1) as f64;
    let seam_cost_p95 = nearest_rank_percentile(seam_costs, 0.95);

    let mut overlap_differences = Vec::with_capacity(h * w * 3);
    for y in 0..h {
        for x in 0..w {
            for channel in 0..3 {
                overlap_differences.push(
                    (left[[y, x, channel]] as f64 - right[[y, x, channel]] as f64).abs()
                        / max_value.max(1.0),
                );
            }
        }
    }
    let overlap_mean_abs_difference =
        overlap_differences.iter().sum::<f64>() / overlap_differences.len().max(1) as f64;
    let overlap_p95_abs_difference = nearest_rank_percentile(overlap_differences, 0.95);

    let transition_width = config.blend_width.min(w.saturating_sub(1));
    let radii = seam_blend_radii(transition_width);
    let largest_radius = radii.last().copied().unwrap_or(0);
    let padding = transition_width.saturating_add(largest_radius.saturating_mul(2));
    let roi_start = seam_min.saturating_sub(padding);
    let roi_end = (seam_max.saturating_add(padding).saturating_add(1)).min(w);
    let roi_width = roi_end.saturating_sub(roi_start);

    let mut output = Array3::<u16>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            let source = if x <= seam[y] { left } else { right };
            for channel in 0..3 {
                output[[y, x, channel]] = source[[y, x, channel]];
            }
        }
    }

    if roi_width > 0 {
        let mut left_roi = Array3::<f64>::zeros((h, roi_width, 3));
        let mut right_roi = Array3::<f64>::zeros((h, roi_width, 3));
        let mut mask = Array2::<f64>::zeros((h, roi_width));
        let half_width = transition_width as f64 / 2.0;
        for y in 0..h {
            for local_x in 0..roi_width {
                let x = roi_start + local_x;
                for channel in 0..3 {
                    left_roi[[y, local_x, channel]] = left[[y, x, channel]] as f64;
                    right_roi[[y, local_x, channel]] = right[[y, x, channel]] as f64;
                }
                let alpha = if transition_width == 0 {
                    if x > seam[y] {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    let start = seam[y] as f64 - half_width;
                    smoothstep01((x as f64 - start) / transition_width as f64)
                };
                mask[[y, local_x]] = alpha;
            }
        }
        let blended = multiband_blend_roi(&left_roi, &right_roi, &mask, &radii);
        for y in 0..h {
            for local_x in 0..roi_width {
                let x = roi_start + local_x;
                for channel in 0..3 {
                    output[[y, x, channel]] =
                        blended[[y, local_x, channel]].round().clamp(0.0, max_value) as u16;
                }
            }
        }
    }

    let (output_seam_gradient_p95, source_seam_gradient_p95) =
        seam_gradient_percentiles(left, right, &output, &seam, max_value);
    let gradient_ratio = if source_seam_gradient_p95 > 1e-9 {
        output_seam_gradient_p95 / source_seam_gradient_p95
    } else if output_seam_gradient_p95 <= 1e-9 {
        0.0
    } else {
        1.0e12
    };
    let gradient_review_required = gradient_ratio > SEAM_BLEND_MAX_GRADIENT_AMPLIFICATION;
    let review_required = detail_consistency.review_required || gradient_review_required;
    let review_reason = match (
        detail_consistency.review_required,
        gradient_review_required,
    ) {
        (true, true) => format!(
            "{}; output seam-gradient p95 is {:.3}x the corresponding source gradient, above the {:.3}x review limit",
            detail_consistency.review_reason,
            gradient_ratio,
            SEAM_BLEND_MAX_GRADIENT_AMPLIFICATION
        ),
        (true, false) => detail_consistency.review_reason.clone(),
        (false, true) => format!(
            "output seam-gradient p95 is {:.3}x the corresponding source gradient, above the {:.3}x review limit",
            gradient_ratio, SEAM_BLEND_MAX_GRADIENT_AMPLIFICATION
        ),
        (false, false) =>
            "seam-gradient continuity and multi-scale detail consistency passed their conservative review gates"
                .to_string(),
    };
    (
        output,
        SeamBlendDiagnostics {
            mode: "seam_aware_multiband".to_string(),
            applied: true,
            reason: format!(
                "selected a minimum-cost low-detail seam and blended {} spatial-frequency band{} inside a bounded overlap ROI",
                radii.len(),
                if radii.len() == 1 { "" } else { "s" }
            ),
            overlap_width_px: w,
            overlap_height_px: h,
            transition_width_px: transition_width,
            pyramid_levels: radii.len(),
            pyramid_radii_px: radii,
            processing_roi_x: [roi_start, roi_end],
            seam_path_min_x: seam_min,
            seam_path_max_x: seam_max,
            seam_path_mean_x: seam_mean,
            seam_path_mean_normalized_cost: seam_cost_mean,
            seam_path_p95_normalized_cost: seam_cost_p95,
            overlap_mean_abs_difference,
            overlap_p95_abs_difference,
            output_seam_gradient_p95,
            source_seam_gradient_p95,
            output_to_source_seam_gradient_ratio: gradient_ratio,
            detail_consistency,
            review_required,
            review_reason,
        },
    )
}

/// Invert a 3x3 matrix. Returns `None` if the determinant is near zero.
fn invert_3x3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

/// Multiply two 3x3 matrices: result = a * b.
fn mat_mul_3x3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Apply a 3x3 homography to a 2D point, returning the projected (x, y).
fn transform_point(m: [[f64; 3]; 3], x: f64, y: f64) -> (f64, f64) {
    let w = m[2][0] * x + m[2][1] * y + m[2][2];
    if w.abs() < 1e-12 {
        return (f64::MAX, f64::MAX);
    }
    let px = (m[0][0] * x + m[0][1] * y + m[0][2]) / w;
    let py = (m[1][0] * x + m[1][1] * y + m[1][2]) / w;
    (px, py)
}

fn bilinear_gray_sample(image: &Array2<f64>, x: f64, y: f64) -> Option<f64> {
    let (height, width) = image.dim();
    if height == 0
        || width == 0
        || !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x > width.saturating_sub(1) as f64
        || y > height.saturating_sub(1) as f64
    {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let top = image[[y0, x0]] * (1.0 - fx) + image[[y0, x1]] * fx;
    let bottom = image[[y1, x0]] * (1.0 - fx) + image[[y1, x1]] * fx;
    Some(top * (1.0 - fy) + bottom * fy)
}

fn sampled_transform_ncc(
    gray_left: &Array2<f64>,
    gray_right: &Array2<f64>,
    right_to_left: [[f64; 3]; 3],
    overlap_x_start: usize,
    split: usize,
) -> (f64, usize) {
    let Some(left_to_right) = invert_3x3(right_to_left) else {
        return (0.0, 0);
    };
    let (height, width) = gray_left.dim();
    if overlap_x_start >= width || height == 0 {
        return (0.0, 0);
    }
    let area = height.saturating_mul(width - overlap_x_start).max(1);
    let stride = ((area as f64 / (AFFINE_MAX_ALIGNMENT_SAMPLES * 2).max(1) as f64)
        .sqrt()
        .ceil() as usize)
        .max(1);

    let mut count = 0usize;
    let mut sum_left = 0.0;
    let mut sum_right = 0.0;
    let mut sum_left_sq = 0.0;
    let mut sum_right_sq = 0.0;
    let mut sum_product = 0.0;
    for y in (0..height).step_by(stride) {
        for x in (overlap_x_start..width).step_by(stride) {
            if ((x / stride) + (y / stride)) & 1 != split & 1 {
                continue;
            }
            let (source_x, source_y) = transform_point(left_to_right, x as f64, y as f64);
            let Some(right_value) = bilinear_gray_sample(gray_right, source_x, source_y) else {
                continue;
            };
            let left_value = gray_left[[y, x]];
            count += 1;
            sum_left += left_value;
            sum_right += right_value;
            sum_left_sq += left_value * left_value;
            sum_right_sq += right_value * right_value;
            sum_product += left_value * right_value;
        }
    }
    if count < 2 {
        return (0.0, count);
    }
    let n = count as f64;
    let covariance = sum_product - sum_left * sum_right / n;
    let left_variance = (sum_left_sq - sum_left * sum_left / n).max(0.0);
    let right_variance = (sum_right_sq - sum_right * sum_right / n).max(0.0);
    let denominator = (left_variance * right_variance).sqrt();
    if denominator <= 1e-12 {
        (0.0, count)
    } else {
        ((covariance / denominator).clamp(-1.0, 1.0), count)
    }
}

fn validate_homography_spatial_evidence(
    left: &Array3<u16>,
    right: &Array3<u16>,
    x_offset: usize,
    y_offset: i32,
    overlap: usize,
    candidate_right_to_left: [[f64; 3]; 3],
) -> HomographySpatialValidationDiagnostics {
    let gray_left = to_grayscale(left);
    let gray_right = to_grayscale(right);
    let baseline = similarity_right_to_left_transform(
        overlap,
        right.dim().0,
        x_offset,
        y_offset,
        0.0,
        [0.0, 0.0],
    );
    let mut baseline_ncc = [0.0; 2];
    let mut candidate_ncc = [0.0; 2];
    let mut sample_count = [0usize; 2];
    for split in 0..2 {
        let (baseline_score, baseline_samples) =
            sampled_transform_ncc(&gray_left, &gray_right, baseline, x_offset, split);
        let (candidate_score, candidate_samples) = sampled_transform_ncc(
            &gray_left,
            &gray_right,
            candidate_right_to_left,
            x_offset,
            split,
        );
        baseline_ncc[split] = baseline_score;
        candidate_ncc[split] = candidate_score;
        sample_count[split] = baseline_samples.min(candidate_samples);
    }
    let ncc_improvement = std::array::from_fn(|split| candidate_ncc[split] - baseline_ncc[split]);
    let mean_ncc_improvement = ncc_improvement.iter().sum::<f64>() * 0.5;
    let registration_error_reduction = std::array::from_fn(|split| {
        if baseline_ncc[split] < 1.0 - 1e-9 {
            ncc_improvement[split] / (1.0 - baseline_ncc[split]).max(1e-9)
        } else {
            0.0
        }
    });
    let mean_registration_error_reduction = registration_error_reduction.iter().sum::<f64>() * 0.5;
    let (right_height, right_width, _) = right.dim();
    let probes = [
        [0.0, 0.0],
        [right_width as f64, 0.0],
        [right_width as f64, right_height as f64],
        [0.0, right_height as f64],
        [right_width as f64 * 0.5, right_height as f64 * 0.5],
    ];
    let maximum_deviation_from_translation_px = probes
        .iter()
        .map(|probe| {
            let baseline_point = transform_point(baseline, probe[0], probe[1]);
            let candidate_point = transform_point(candidate_right_to_left, probe[0], probe[1]);
            ((candidate_point.0 - baseline_point.0).powi(2)
                + (candidate_point.1 - baseline_point.1).powi(2))
            .sqrt()
        })
        .fold(0.0f64, f64::max);
    let enough_samples = sample_count
        .iter()
        .all(|samples| *samples >= AFFINE_MIN_ALIGNMENT_SAMPLES);
    let both_splits_improve = ncc_improvement
        .iter()
        .all(|improvement| *improvement >= HOMOGRAPHY_MIN_SPATIAL_SPLIT_IMPROVEMENT);
    let both_splits_reduce_error = registration_error_reduction
        .iter()
        .all(|reduction| *reduction >= HOMOGRAPHY_MIN_SPATIAL_SPLIT_ERROR_REDUCTION);
    let significant =
        maximum_deviation_from_translation_px >= HOMOGRAPHY_MIN_SIGNIFICANT_DEVIATION_PX;
    let accepted = enough_samples
        && significant
        && both_splits_improve
        && mean_ncc_improvement >= HOMOGRAPHY_MIN_MEAN_SPATIAL_IMPROVEMENT
        && both_splits_reduce_error
        && mean_registration_error_reduction >= HOMOGRAPHY_MIN_MEAN_SPATIAL_ERROR_REDUCTION;
    let reason = if !enough_samples {
        format!(
            "insufficient spatial validation samples: {}/{}, need {} in each checkerboard split",
            sample_count[0], sample_count[1], AFFINE_MIN_ALIGNMENT_SAMPLES
        )
    } else if !significant {
        format!(
            "homography was indistinguishable from translation: maximum deviation {:.3}px < {:.3}px",
            maximum_deviation_from_translation_px,
            HOMOGRAPHY_MIN_SIGNIFICANT_DEVIATION_PX
        )
    } else if !both_splits_improve {
        format!(
            "homography did not improve both spatial splits by {:.3}: improvements {:.4}/{:.4}",
            HOMOGRAPHY_MIN_SPATIAL_SPLIT_IMPROVEMENT, ncc_improvement[0], ncc_improvement[1]
        )
    } else if mean_ncc_improvement < HOMOGRAPHY_MIN_MEAN_SPATIAL_IMPROVEMENT {
        format!(
            "homography mean spatial improvement {:.4} < {:.4}",
            mean_ncc_improvement, HOMOGRAPHY_MIN_MEAN_SPATIAL_IMPROVEMENT
        )
    } else if !both_splits_reduce_error {
        format!(
            "homography did not reduce registration error by {:.1}% in both splits: {:.1}%/{:.1}%",
            HOMOGRAPHY_MIN_SPATIAL_SPLIT_ERROR_REDUCTION * 100.0,
            registration_error_reduction[0] * 100.0,
            registration_error_reduction[1] * 100.0
        )
    } else if mean_registration_error_reduction < HOMOGRAPHY_MIN_MEAN_SPATIAL_ERROR_REDUCTION {
        format!(
            "homography mean registration-error reduction {:.1}% < {:.1}%",
            mean_registration_error_reduction * 100.0,
            HOMOGRAPHY_MIN_MEAN_SPATIAL_ERROR_REDUCTION * 100.0
        )
    } else {
        format!(
            "homography improved independent spatial checkerboard NCC by {:.4} with {:.1}% mean registration-error reduction",
            mean_ncc_improvement,
            mean_registration_error_reduction * 100.0
        )
    };

    HomographySpatialValidationDiagnostics {
        accepted,
        reason,
        method: "two_spatial_checkerboard_ncc_splits_against_translation",
        baseline_ncc,
        candidate_ncc,
        ncc_improvement,
        mean_ncc_improvement,
        registration_error_reduction,
        mean_registration_error_reduction,
        sample_count,
        maximum_deviation_from_translation_px,
        transform_right_to_left: candidate_right_to_left,
        model_selection_limit: "candidate features are fit only on whole-cell training regions; reverse feature cross-fit and image-domain checkerboards provide separate held-out evidence",
    }
}

fn similarity_right_to_left_transform(
    overlap: usize,
    right_height: usize,
    x_offset: usize,
    y_offset: i32,
    angle_deg: f64,
    translation_refinement: [f64; 2],
) -> [[f64; 3]; 3] {
    let radians = angle_deg.to_radians();
    let cosine = radians.cos();
    let sine = radians.sin();
    let source_anchor = [
        overlap as f64 * 0.5,
        right_height.saturating_sub(1) as f64 * 0.5,
    ];
    let destination_anchor = [
        x_offset as f64 + source_anchor[0] + translation_refinement[0],
        y_offset as f64 + source_anchor[1] + translation_refinement[1],
    ];
    [
        [
            cosine,
            -sine,
            destination_anchor[0] - cosine * source_anchor[0] + sine * source_anchor[1],
        ],
        [
            sine,
            cosine,
            destination_anchor[1] - sine * source_anchor[0] - cosine * source_anchor[1],
        ],
        [0.0, 0.0, 1.0],
    ]
}

fn patch_transform_ncc(
    gray_left: &Array2<f64>,
    gray_right: &Array2<f64>,
    left_to_right: [[f64; 3]; 3],
    destination: [f64; 2],
    source_offset: [f64; 2],
) -> f64 {
    const PATCH_RADIUS: i32 = 18;
    const PATCH_STRIDE: usize = 2;
    let (left_height, left_width) = gray_left.dim();
    let mut left_values = Vec::with_capacity(400);
    let mut right_values = Vec::with_capacity(400);
    for dy in (-PATCH_RADIUS..=PATCH_RADIUS).step_by(PATCH_STRIDE) {
        for dx in (-PATCH_RADIUS..=PATCH_RADIUS).step_by(PATCH_STRIDE) {
            let left_x = destination[0] + dx as f64;
            let left_y = destination[1] + dy as f64;
            if left_x < 0.0
                || left_y < 0.0
                || left_x > left_width.saturating_sub(1) as f64
                || left_y > left_height.saturating_sub(1) as f64
            {
                continue;
            }
            let (source_x, source_y) = transform_point(left_to_right, left_x, left_y);
            let Some(right_value) = bilinear_gray_sample(
                gray_right,
                source_x + source_offset[0],
                source_y + source_offset[1],
            ) else {
                continue;
            };
            let Some(left_value) = bilinear_gray_sample(gray_left, left_x, left_y) else {
                continue;
            };
            left_values.push(left_value);
            right_values.push(right_value);
        }
    }
    if left_values.len() < 64 {
        return 0.0;
    }
    let count = left_values.len() as f64;
    let left_mean = left_values.iter().sum::<f64>() / count;
    let right_mean = right_values.iter().sum::<f64>() / count;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (left, right) in left_values.iter().zip(right_values.iter()) {
        let left_delta = left - left_mean;
        let right_delta = right - right_mean;
        covariance += left_delta * right_delta;
        left_variance += left_delta * left_delta;
        right_variance += right_delta * right_delta;
    }
    let denominator = (left_variance * right_variance).sqrt();
    if denominator <= 1e-12 {
        0.0
    } else {
        (covariance / denominator).clamp(-1.0, 1.0)
    }
}

fn collect_affine_correspondences(
    gray_left: &Array2<f64>,
    gray_right: &Array2<f64>,
    seed_right_to_left: [[f64; 3]; 3],
    x_offset: usize,
    overlap: usize,
) -> Vec<AffineCorrespondence> {
    let Some(left_to_right) = invert_3x3(seed_right_to_left) else {
        return Vec::new();
    };
    let (height, width) = gray_left.dim();
    let overlap_end = (x_offset + overlap).min(width);
    if overlap_end <= x_offset + 48 || height < 64 {
        return Vec::new();
    }
    let rows = 7usize;
    let columns = 5usize;
    let margin = 22.0;
    let mut correspondences = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        for column in 0..columns {
            if (row + column) & 1 == 1 {
                continue;
            }
            let destination = [
                x_offset as f64
                    + margin
                    + column as f64 * ((overlap_end - x_offset) as f64 - 2.0 * margin)
                        / (columns - 1) as f64,
                margin + row as f64 * (height as f64 - 1.0 - 2.0 * margin) / (rows - 1) as f64,
            ];
            let (predicted_source_x, predicted_source_y) =
                transform_point(left_to_right, destination[0], destination[1]);
            let mut best_score = f64::NEG_INFINITY;
            let mut best_offset = [0.0, 0.0];
            for dy in -3..=3 {
                for dx in -3..=3 {
                    let offset = [dx as f64, dy as f64];
                    let score = patch_transform_ncc(
                        gray_left,
                        gray_right,
                        left_to_right,
                        destination,
                        offset,
                    );
                    if score > best_score {
                        best_score = score;
                        best_offset = offset;
                    }
                }
            }
            if best_score >= 0.35 {
                correspondences.push(AffineCorrespondence {
                    source: [
                        predicted_source_x + best_offset[0],
                        predicted_source_y + best_offset[1],
                    ],
                    destination,
                    score: best_score,
                });
            }
        }
    }
    correspondences
}

fn fit_affine_correction(
    seed: [[f64; 3]; 3],
    correspondences: &[AffineCorrespondence],
) -> Option<([[f64; 3]; 3], usize)> {
    if correspondences.len() < 6 {
        return None;
    }
    let center_x = correspondences
        .iter()
        .map(|point| point.source[0])
        .sum::<f64>()
        / correspondences.len() as f64;
    let center_y = correspondences
        .iter()
        .map(|point| point.source[1])
        .sum::<f64>()
        / correspondences.len() as f64;
    let scale_x = correspondences
        .iter()
        .map(|point| (point.source[0] - center_x).abs())
        .fold(0.0, f64::max)
        .max(1.0);
    let scale_y = correspondences
        .iter()
        .map(|point| (point.source[1] - center_y).abs())
        .fold(0.0, f64::max)
        .max(1.0);

    let fit = |points: &[AffineCorrespondence]| -> Option<[[f64; 3]; 3]> {
        let mut normal = Matrix3::<f64>::zeros();
        let mut target_x = Vector3::<f64>::zeros();
        let mut target_y = Vector3::<f64>::zeros();
        for point in points {
            let weight = point.score.max(1e-3).powi(2);
            let basis = Vector3::new(
                1.0,
                (point.source[0] - center_x) / scale_x,
                (point.source[1] - center_y) / scale_y,
            );
            let (seed_x, seed_y) = transform_point(seed, point.source[0], point.source[1]);
            normal += basis * basis.transpose() * weight;
            target_x += basis * ((point.destination[0] - seed_x) * weight);
            target_y += basis * ((point.destination[1] - seed_y) * weight);
        }
        let inverse = (normal + Matrix3::identity() * 1e-8).try_inverse()?;
        let correction_x = inverse * target_x;
        let correction_y = inverse * target_y;
        let mut transform = seed;
        transform[0][0] += correction_x[1] / scale_x;
        transform[0][1] += correction_x[2] / scale_y;
        transform[0][2] += correction_x[0]
            - correction_x[1] * center_x / scale_x
            - correction_x[2] * center_y / scale_y;
        transform[1][0] += correction_y[1] / scale_x;
        transform[1][1] += correction_y[2] / scale_y;
        transform[1][2] += correction_y[0]
            - correction_y[1] * center_x / scale_x
            - correction_y[2] * center_y / scale_y;
        Some(transform)
    };

    let initial = fit(correspondences)?;
    let mut residuals = correspondences
        .iter()
        .map(|point| {
            let (x, y) = transform_point(initial, point.source[0], point.source[1]);
            ((x - point.destination[0]).powi(2) + (y - point.destination[1]).powi(2)).sqrt()
        })
        .collect::<Vec<_>>();
    residuals.sort_by(f64::total_cmp);
    let median = percentile_from_sorted(&residuals, 0.5);
    let threshold = (median * 2.5).clamp(1.0, 3.0);
    let inliers = correspondences
        .iter()
        .copied()
        .filter(|point| {
            let (x, y) = transform_point(initial, point.source[0], point.source[1]);
            ((x - point.destination[0]).powi(2) + (y - point.destination[1]).powi(2)).sqrt()
                <= threshold
        })
        .collect::<Vec<_>>();
    if inliers.len() < 6 {
        return None;
    }
    fit(&inliers).map(|transform| (transform, inliers.len()))
}

fn affine_linear_metrics(transform: [[f64; 3]; 3]) -> (f64, f64, f64, f64) {
    let first = [transform[0][0], transform[1][0]];
    let second = [transform[0][1], transform[1][1]];
    let scale_x = first[0].hypot(first[1]);
    let scale_y = second[0].hypot(second[1]);
    let rotation = first[1].atan2(first[0]).to_degrees();
    let shear = if scale_x > 1e-12 && scale_y > 1e-12 {
        (first[0] * second[0] + first[1] * second[1]) / (scale_x * scale_y)
    } else {
        1.0
    };
    (rotation, scale_x, scale_y, shear)
}

fn native_affine_trigger(
    hypothesis: &StitchHypothesis,
    transform_mode: TransformMode,
) -> Option<String> {
    if matches!(transform_mode, TransformMode::Affine) {
        return Some("explicit affine transform request".to_string());
    }
    if !matches!(transform_mode, TransformMode::Auto) {
        return None;
    }
    if !hypothesis.accepted {
        return Some(
            "translation hypothesis was rejected and affine may rescue coherent skew".to_string(),
        );
    }
    let slopes = [
        hypothesis.validation.local_dx_plane_px[1].abs(),
        hypothesis.validation.local_dx_plane_px[2].abs(),
        hypothesis.validation.local_dy_plane_px[1].abs(),
        hypothesis.validation.local_dy_plane_px[2].abs(),
    ];
    let maximum_slope = slopes.into_iter().fold(0.0, f64::max);
    if maximum_slope >= AFFINE_AUTO_LOCAL_SLOPE_TRIGGER_PX {
        Some(format!(
            "local overlap displacement changed by {:.2}px across the validation grid",
            maximum_slope
        ))
    } else {
        None
    }
}

fn estimate_native_affine(
    left: &Array3<u16>,
    right: &Array3<u16>,
    x_offset: usize,
    y_offset: i32,
    overlap: usize,
    trigger: String,
) -> NativeAffineEstimate {
    let gray_left = to_grayscale(left);
    let gray_right = to_grayscale(right);
    let baseline = similarity_right_to_left_transform(
        overlap,
        right.dim().0,
        x_offset,
        y_offset,
        0.0,
        [0.0, 0.0],
    );
    let (training_baseline_ncc, training_sample_count) =
        sampled_transform_ncc(&gray_left, &gray_right, baseline, x_offset, 0);
    let (held_out_baseline_ncc, held_out_sample_count) =
        sampled_transform_ncc(&gray_left, &gray_right, baseline, x_offset, 1);

    let mut best_transform = baseline;
    let mut best_training_ncc = training_baseline_ncc;
    let mut best_angle = 0.0;
    let mut best_refinement = [0.0, 0.0];
    for angle_step in -(AFFINE_MAX_ROTATION_DEG / AFFINE_COARSE_ROTATION_STEP_DEG) as i32
        ..=(AFFINE_MAX_ROTATION_DEG / AFFINE_COARSE_ROTATION_STEP_DEG) as i32
    {
        let angle = angle_step as f64 * AFFINE_COARSE_ROTATION_STEP_DEG;
        for vertical in -3..=3 {
            let transform = similarity_right_to_left_transform(
                overlap,
                right.dim().0,
                x_offset,
                y_offset,
                angle,
                [0.0, vertical as f64],
            );
            let (score, samples) =
                sampled_transform_ncc(&gray_left, &gray_right, transform, x_offset, 0);
            if samples >= AFFINE_MIN_ALIGNMENT_SAMPLES && score > best_training_ncc {
                best_training_ncc = score;
                best_transform = transform;
                best_angle = angle;
                best_refinement = [0.0, vertical as f64];
            }
        }
    }

    let coarse_angle = best_angle;
    let coarse_refinement = best_refinement;
    for angle_step in -5..=5 {
        let angle = coarse_angle + angle_step as f64 * AFFINE_FINE_ROTATION_STEP_DEG;
        if angle.abs() > AFFINE_MAX_ROTATION_DEG {
            continue;
        }
        for vertical_step in -2..=2 {
            for horizontal_step in -2..=2 {
                let refinement = [
                    coarse_refinement[0] + horizontal_step as f64 * 0.25,
                    coarse_refinement[1] + vertical_step as f64 * 0.25,
                ];
                let transform = similarity_right_to_left_transform(
                    overlap,
                    right.dim().0,
                    x_offset,
                    y_offset,
                    angle,
                    refinement,
                );
                let (score, samples) =
                    sampled_transform_ncc(&gray_left, &gray_right, transform, x_offset, 0);
                if samples >= AFFINE_MIN_ALIGNMENT_SAMPLES && score > best_training_ncc {
                    best_training_ncc = score;
                    best_transform = transform;
                }
            }
        }
    }

    let correspondences =
        collect_affine_correspondences(&gray_left, &gray_right, best_transform, x_offset, overlap);
    let mut local_inlier_count = 0usize;
    if let Some((fitted, inliers)) = fit_affine_correction(best_transform, &correspondences) {
        if assess_transform_plausibility(
            fitted,
            overlap,
            left.dim().0.min(right.dim().0),
            right.dim().1,
            right.dim().0,
        )
        .is_ok()
        {
            let (score, samples) =
                sampled_transform_ncc(&gray_left, &gray_right, fitted, x_offset, 0);
            if samples >= AFFINE_MIN_ALIGNMENT_SAMPLES && score > best_training_ncc + 0.0005 {
                best_training_ncc = score;
                best_transform = fitted;
                local_inlier_count = inliers;
            }
        }
    }

    let (held_out_candidate_ncc, candidate_held_out_samples) =
        sampled_transform_ncc(&gray_left, &gray_right, best_transform, x_offset, 1);
    let held_out_sample_count = held_out_sample_count.min(candidate_held_out_samples);
    let improvement = held_out_candidate_ncc - held_out_baseline_ncc;
    let error_reduction = if held_out_baseline_ncc < 1.0 - 1e-9 {
        improvement / (1.0 - held_out_baseline_ncc).max(1e-9)
    } else {
        0.0
    };
    let (rotation, scale_x, scale_y, shear) = affine_linear_metrics(best_transform);
    let rotation_search_boundary_hit =
        rotation.abs() >= AFFINE_MAX_ROTATION_DEG - AFFINE_ROTATION_BOUNDARY_MARGIN_DEG;
    let baseline_anchor = [overlap as f64 * 0.5, right.dim().0 as f64 * 0.5];
    let (baseline_anchor_x, baseline_anchor_y) =
        transform_point(baseline, baseline_anchor[0], baseline_anchor[1]);
    let (candidate_anchor_x, candidate_anchor_y) =
        transform_point(best_transform, baseline_anchor[0], baseline_anchor[1]);
    let translation_refinement = [
        candidate_anchor_x - baseline_anchor_x,
        candidate_anchor_y - baseline_anchor_y,
    ];
    let significant_deformation = rotation.abs() >= AFFINE_MIN_SIGNIFICANT_ROTATION_DEG
        || (scale_x - 1.0).abs() >= AFFINE_MIN_SIGNIFICANT_LINEAR_DEFORMATION
        || (scale_y - 1.0).abs() >= AFFINE_MIN_SIGNIFICANT_LINEAR_DEFORMATION
        || shear.abs() >= AFFINE_MIN_SIGNIFICANT_LINEAR_DEFORMATION
        || translation_refinement[0].abs() >= AFFINE_MIN_SIGNIFICANT_TRANSLATION_REFINEMENT_PX
        || translation_refinement[1].abs() >= AFFINE_MIN_SIGNIFICANT_TRANSLATION_REFINEMENT_PX;
    let plausibility = assess_transform_plausibility(
        best_transform,
        overlap,
        left.dim().0.min(right.dim().0),
        right.dim().1,
        right.dim().0,
    );
    let enough_samples = training_sample_count >= AFFINE_MIN_ALIGNMENT_SAMPLES
        && held_out_sample_count >= AFFINE_MIN_ALIGNMENT_SAMPLES;
    let accepted = enough_samples
        && plausibility.is_ok()
        && !rotation_search_boundary_hit
        && significant_deformation
        && improvement >= AFFINE_MIN_HELD_OUT_NCC_IMPROVEMENT
        && error_reduction >= AFFINE_MIN_HELD_OUT_ERROR_REDUCTION;
    let reason = if !enough_samples {
        format!(
            "insufficient disjoint overlap samples: training {}, held-out {}, need {} each",
            training_sample_count, held_out_sample_count, AFFINE_MIN_ALIGNMENT_SAMPLES
        )
    } else if let Err(reason) = plausibility {
        format!("candidate affine transform was implausible: {reason}")
    } else if rotation_search_boundary_hit {
        format!(
            "candidate rotation {:.3} degrees hit the bounded search limit of +/-{:.3} degrees; refusing an unconverged affine optimum",
            rotation, AFFINE_MAX_ROTATION_DEG
        )
    } else if !significant_deformation {
        "candidate was indistinguishable from translation at the configured geometric precision"
            .to_string()
    } else if improvement < AFFINE_MIN_HELD_OUT_NCC_IMPROVEMENT {
        format!(
            "held-out NCC improvement {:.4} was below the {:.4} requirement",
            improvement, AFFINE_MIN_HELD_OUT_NCC_IMPROVEMENT
        )
    } else if error_reduction < AFFINE_MIN_HELD_OUT_ERROR_REDUCTION {
        format!(
            "held-out registration-error reduction {:.1}% was below the {:.1}% requirement",
            error_reduction * 100.0,
            AFFINE_MIN_HELD_OUT_ERROR_REDUCTION * 100.0
        )
    } else {
        format!(
            "affine improved disjoint held-out NCC {:.4} -> {:.4} ({:.1}% registration-error reduction)",
            held_out_baseline_ncc,
            held_out_candidate_ncc,
            error_reduction * 100.0
        )
    };

    NativeAffineEstimate {
        diagnostics: NativeAffineDiagnostics {
            attempted: true,
            accepted,
            reason,
            model: "native_six_parameter_affine_with_similarity_seed",
            trigger,
            training_baseline_ncc,
            training_candidate_ncc: best_training_ncc,
            held_out_baseline_ncc,
            held_out_candidate_ncc,
            held_out_ncc_improvement: improvement,
            held_out_error_reduction: error_reduction,
            training_sample_count,
            held_out_sample_count,
            local_correspondence_count: correspondences.len(),
            local_inlier_count,
            rotation_deg: rotation,
            rotation_search_limit_deg: AFFINE_MAX_ROTATION_DEG,
            rotation_search_boundary_hit,
            scale_x,
            scale_y,
            shear_cosine: shear,
            translation_refinement_px: translation_refinement,
            transform_right_to_left: best_transform,
            interpolation: "bicubic_catmull_rom_single_resample",
        },
    }
}

fn cubic_catmull_rom_weight(distance: f64) -> f64 {
    let distance = distance.abs();
    if distance <= 1.0 {
        1.5 * distance.powi(3) - 2.5 * distance.powi(2) + 1.0
    } else if distance < 2.0 {
        -0.5 * distance.powi(3) + 2.5 * distance.powi(2) - 4.0 * distance + 2.0
    } else {
        0.0
    }
}

fn warp_projective_bicubic(
    source: &Array3<u16>,
    source_to_canvas: [[f64; 3]; 3],
    canvas_width: usize,
    canvas_height: usize,
    max_value: f64,
) -> Option<(Array3<u16>, Array2<u8>)> {
    let canvas_to_source = invert_3x3(source_to_canvas)?;
    let (source_height, source_width, _) = source.dim();
    let mut pixels = vec![0u16; canvas_height.saturating_mul(canvas_width).saturating_mul(3)];
    let mut validity = vec![0u8; canvas_height.saturating_mul(canvas_width)];
    pixels
        .par_chunks_mut(canvas_width * 3)
        .zip(validity.par_chunks_mut(canvas_width))
        .enumerate()
        .for_each(|(canvas_y, (row, validity_row))| {
            for canvas_x in 0..canvas_width {
                let (source_x, source_y) = transform_point(
                    canvas_to_source,
                    canvas_x as f64 + 0.5,
                    canvas_y as f64 + 0.5,
                );
                let source_x = source_x - 0.5;
                let source_y = source_y - 0.5;
                if source_x < 0.0
                    || source_y < 0.0
                    || source_x > source_width.saturating_sub(1) as f64
                    || source_y > source_height.saturating_sub(1) as f64
                {
                    continue;
                }
                validity_row[canvas_x] = 1;
                let base_x = source_x.floor() as isize;
                let base_y = source_y.floor() as isize;
                let mut accumulated = [0.0f64; 3];
                let mut total_weight = 0.0;
                for offset_y in -1isize..=2 {
                    let sample_y = (base_y + offset_y)
                        .clamp(0, source_height.saturating_sub(1) as isize)
                        as usize;
                    let weight_y = cubic_catmull_rom_weight(source_y - (base_y + offset_y) as f64);
                    for offset_x in -1isize..=2 {
                        let sample_x = (base_x + offset_x)
                            .clamp(0, source_width.saturating_sub(1) as isize)
                            as usize;
                        let weight_x =
                            cubic_catmull_rom_weight(source_x - (base_x + offset_x) as f64);
                        let weight = weight_x * weight_y;
                        total_weight += weight;
                        for channel in 0..3 {
                            accumulated[channel] +=
                                source[[sample_y, sample_x, channel]] as f64 * weight;
                        }
                    }
                }
                for channel in 0..3 {
                    row[canvas_x * 3 + channel] = (accumulated[channel] / total_weight.max(1e-12))
                        .round()
                        .clamp(0.0, max_value)
                        as u16;
                }
            }
        });
    Some((
        Array3::from_shape_vec((canvas_height, canvas_width, 3), pixels).ok()?,
        Array2::from_shape_vec((canvas_height, canvas_width), validity).ok()?,
    ))
}

fn largest_valid_overlap_rectangle(
    right_validity: &Array2<u8>,
    left_origin: [usize; 2],
    left_width: usize,
    left_height: usize,
) -> Option<Rectangle> {
    let (canvas_height, canvas_width) = right_validity.dim();
    let mut heights = vec![0usize; canvas_width];
    let mut best: Option<Rectangle> = None;
    for y in 0..canvas_height {
        for (x, height) in heights.iter_mut().enumerate() {
            let in_left = x >= left_origin[0]
                && x < left_origin[0] + left_width
                && y >= left_origin[1]
                && y < left_origin[1] + left_height;
            if in_left && right_validity[[y, x]] != 0 {
                *height += 1;
            } else {
                *height = 0;
            }
        }
        let mut stack = Vec::<usize>::new();
        for x in 0..=canvas_width {
            let current_height = if x == canvas_width { 0 } else { heights[x] };
            while stack
                .last()
                .is_some_and(|&index| heights[index] > current_height)
            {
                let index = stack.pop().expect("non-empty histogram stack");
                let height = heights[index];
                let start = stack.last().map_or(0, |previous| previous + 1);
                let width = x - start;
                let candidate = Rectangle {
                    x: start,
                    y: y + 1 - height,
                    width,
                    height,
                };
                if best.is_none_or(|best| {
                    candidate.width * candidate.height > best.width * best.height
                }) {
                    best = Some(candidate);
                }
            }
            stack.push(x);
        }
    }
    best.filter(|rectangle| rectangle.width >= 12 && rectangle.height >= 16)
}

fn projected_pair_canvas_geometry(
    left_width: usize,
    left_height: usize,
    right_width: usize,
    right_height: usize,
    right_to_left: [[f64; 3]; 3],
) -> Result<ProjectedCanvasGeometry, String> {
    if left_width == 0 || left_height == 0 || right_width == 0 || right_height == 0 {
        return Err("projective composition received an empty component".to_string());
    }
    let right_corners = [
        [0.0, 0.0],
        [right_width as f64, 0.0],
        [right_width as f64, right_height as f64],
        [0.0, right_height as f64],
    ];
    let homogeneous_w = right_corners.map(|corner| {
        right_to_left[2][0] * corner[0] + right_to_left[2][1] * corner[1] + right_to_left[2][2]
    });
    if homogeneous_w
        .iter()
        .any(|value| !value.is_finite() || value.abs() < 1e-6)
        || homogeneous_w
            .windows(2)
            .any(|values| values[0].signum() != values[1].signum())
    {
        return Err(
            "projective transform crosses or approaches its horizon inside the image".to_string(),
        );
    }

    let mut min_x = 0.0f64;
    let mut min_y = 0.0f64;
    let mut max_x = left_width as f64;
    let mut max_y = left_height as f64;
    for corner in right_corners {
        let (x, y) = transform_point(right_to_left, corner[0], corner[1]);
        if !x.is_finite() || !y.is_finite() || x.abs() > 1.0e9 || y.abs() > 1.0e9 {
            return Err("projective transform produced a non-finite canvas corner".to_string());
        }
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let min_x_floor = min_x.floor();
    let min_y_floor = min_y.floor();
    let max_x_ceil = max_x.ceil();
    let max_y_ceil = max_y.ceil();
    let canvas_width = (max_x_ceil - min_x_floor).max(1.0) as usize;
    let canvas_height = (max_y_ceil - min_y_floor).max(1.0) as usize;
    let maximum_width = left_width.saturating_add(right_width).saturating_mul(3);
    let maximum_height = left_height.saturating_add(right_height).saturating_mul(3);
    let source_area = left_width
        .saturating_mul(left_height)
        .saturating_add(right_width.saturating_mul(right_height));
    let maximum_area = source_area.saturating_mul(8);
    if canvas_width > maximum_width
        || canvas_height > maximum_height
        || canvas_width.saturating_mul(canvas_height) > maximum_area
    {
        return Err(format!(
            "projective canvas expansion was implausible: {}x{} from {}x{} and {}x{}",
            canvas_width, canvas_height, left_width, left_height, right_width, right_height
        ));
    }

    let left_origin = [(-min_x_floor) as usize, (-min_y_floor) as usize];
    let canvas_shift = [
        [1.0, 0.0, left_origin[0] as f64],
        [0.0, 1.0, left_origin[1] as f64],
        [0.0, 0.0, 1.0],
    ];
    let right_to_canvas = mat_mul_3x3(canvas_shift, right_to_left);
    Ok(ProjectedCanvasGeometry {
        left_origin,
        canvas_width,
        canvas_height,
        right_to_canvas,
    })
}

fn compose_projective_pair(
    left: &Array3<u16>,
    right: &Array3<u16>,
    right_to_left: [[f64; 3]; 3],
    config: &StitchConfig,
) -> Result<ProjectivePairComposition, String> {
    let (left_height, left_width, _) = left.dim();
    let (right_height, right_width, _) = right.dim();
    let ProjectedCanvasGeometry {
        left_origin,
        canvas_width,
        canvas_height,
        right_to_canvas,
    } = projected_pair_canvas_geometry(
        left_width,
        left_height,
        right_width,
        right_height,
        right_to_left,
    )?;
    let (warped_right, right_validity) = warp_projective_bicubic(
        right,
        right_to_canvas,
        canvas_width,
        canvas_height,
        stitch_sample_max(config),
    )
    .ok_or_else(|| "projective warp matrix was singular".to_string())?;
    let blend_rectangle =
        largest_valid_overlap_rectangle(&right_validity, left_origin, left_width, left_height)
            .ok_or_else(|| {
                "projective warp did not retain a rectangular overlap large enough to blend"
                    .to_string()
            })?;

    let mut left_overlap = Array3::<u16>::zeros((blend_rectangle.height, blend_rectangle.width, 3));
    let mut right_overlap =
        Array3::<u16>::zeros((blend_rectangle.height, blend_rectangle.width, 3));
    for y in 0..blend_rectangle.height {
        for x in 0..blend_rectangle.width {
            let canvas_x = blend_rectangle.x + x;
            let canvas_y = blend_rectangle.y + y;
            let left_x = canvas_x - left_origin[0];
            let left_y = canvas_y - left_origin[1];
            for channel in 0..3 {
                left_overlap[[y, x, channel]] = left[[left_y, left_x, channel]];
                right_overlap[[y, x, channel]] = warped_right[[canvas_y, canvas_x, channel]];
            }
        }
    }
    let seam_exposure_correction = seam_exposure_correction(&left_overlap, &right_overlap, config);
    let photometric_field = applied_photometric_field(&seam_exposure_correction);
    let compensated_right = apply_photometric_correction(
        &warped_right,
        &photometric_field,
        blend_rectangle.x,
        blend_rectangle.width,
        blend_rectangle.y,
        blend_rectangle.height,
        stitch_sample_max(config),
    );
    for y in 0..blend_rectangle.height {
        for x in 0..blend_rectangle.width {
            let canvas_x = blend_rectangle.x + x;
            let canvas_y = blend_rectangle.y + y;
            for channel in 0..3 {
                right_overlap[[y, x, channel]] = compensated_right[[canvas_y, canvas_x, channel]];
            }
        }
    }
    let (blended_overlap, seam_blend) =
        seam_aware_multiband_blend(&left_overlap, &right_overlap, config);

    let mut row_overlap_bounds = vec![None::<(usize, usize)>; canvas_height];
    let mut valid_union_pixel_count = 0usize;
    let mut overlap_pixel_count = 0usize;
    for y in 0..canvas_height {
        let mut row_min = usize::MAX;
        let mut row_max = 0usize;
        for x in 0..canvas_width {
            let in_left = x >= left_origin[0]
                && x < left_origin[0] + left_width
                && y >= left_origin[1]
                && y < left_origin[1] + left_height;
            let in_right = right_validity[[y, x]] != 0;
            if in_left || in_right {
                valid_union_pixel_count += 1;
            }
            if in_left && in_right {
                overlap_pixel_count += 1;
                row_min = row_min.min(x);
                row_max = row_max.max(x);
            }
        }
        if row_min != usize::MAX {
            row_overlap_bounds[y] = Some((row_min, row_max));
        }
    }

    let max_value = stitch_sample_max(config);
    let mut image = Array3::<u16>::zeros((canvas_height, canvas_width, 3));
    for y in 0..canvas_height {
        for x in 0..canvas_width {
            let in_left = x >= left_origin[0]
                && x < left_origin[0] + left_width
                && y >= left_origin[1]
                && y < left_origin[1] + left_height;
            let in_right = right_validity[[y, x]] != 0;
            if !in_left && !in_right {
                continue;
            }
            if in_left
                && in_right
                && x >= blend_rectangle.x
                && x < blend_rectangle.x + blend_rectangle.width
                && y >= blend_rectangle.y
                && y < blend_rectangle.y + blend_rectangle.height
            {
                let blend_x = x - blend_rectangle.x;
                let blend_y = y - blend_rectangle.y;
                for channel in 0..3 {
                    image[[y, x, channel]] = blended_overlap[[blend_y, blend_x, channel]];
                }
                continue;
            }
            let left_coordinates = if in_left {
                Some((y - left_origin[1], x - left_origin[0]))
            } else {
                None
            };
            match (left_coordinates, in_right) {
                (Some((left_y, left_x)), true) => {
                    let (row_min, row_max) = row_overlap_bounds[y].unwrap_or((x, x));
                    let midpoint = (row_min + row_max) as f64 * 0.5;
                    let transition = config.blend_width.max(1) as f64;
                    let alpha =
                        smoothstep01((x as f64 - (midpoint - transition * 0.5)) / transition);
                    for channel in 0..3 {
                        image[[y, x, channel]] =
                            (left[[left_y, left_x, channel]] as f64 * (1.0 - alpha)
                                + compensated_right[[y, x, channel]] as f64 * alpha)
                                .round()
                                .clamp(0.0, max_value) as u16;
                    }
                }
                (Some((left_y, left_x)), false) => {
                    for channel in 0..3 {
                        image[[y, x, channel]] = left[[left_y, left_x, channel]];
                    }
                }
                (None, true) => {
                    for channel in 0..3 {
                        image[[y, x, channel]] = compensated_right[[y, x, channel]];
                    }
                }
                (None, false) => {}
            }
        }
    }

    let canvas_void_pixel_count = canvas_width
        .saturating_mul(canvas_height)
        .saturating_sub(valid_union_pixel_count);
    let overlap_outside_multiband_pixel_count = overlap_pixel_count
        .saturating_sub(blend_rectangle.width.saturating_mul(blend_rectangle.height));
    Ok(ProjectivePairComposition {
        image,
        right_to_canvas,
        left_origin,
        canvas_width,
        canvas_height,
        valid_union_pixel_count,
        canvas_void_pixel_count,
        overlap_pixel_count,
        overlap_outside_multiband_pixel_count,
        multiband_overlap_rectangle: blend_rectangle,
        seam_exposure_correction,
        seam_blend,
    })
}

#[allow(clippy::too_many_arguments)]
fn stitch_native_affine(
    left: &Array3<u16>,
    right: &Array3<u16>,
    x_offset: usize,
    y_offset: i32,
    overlap: usize,
    order: StitchOrder,
    config: &StitchConfig,
    hypotheses: &[StitchHypothesis],
    best_index: usize,
    diagnostics: NativeAffineDiagnostics,
    warnings: Vec<String>,
) -> Result<StitchResult, String> {
    let ProjectivePairComposition {
        image,
        right_to_canvas,
        left_origin,
        canvas_width,
        canvas_height,
        valid_union_pixel_count,
        canvas_void_pixel_count,
        overlap_pixel_count,
        overlap_outside_multiband_pixel_count,
        multiband_overlap_rectangle,
        seam_exposure_correction,
        seam_blend,
    } = compose_projective_pair(left, right, diagnostics.transform_right_to_left, config)?;
    let (stitched, crop_info) = apply_target_crop(&image, config);
    let preserved_full_valid_union = crop_info.is_none();
    let mut report = PhaseReport::ok(
        "stitch",
        hypotheses[best_index].objective_score,
        serde_json::json!({
            "decision": "accepted",
            "method": "native_affine",
            "requested_transform": config.transform_mode.as_str(),
            "transform_model_used": "native_affine",
            "opencv_requested": config.use_opencv,
            "opencv_available": opencv_match::is_available(),
            "chosen_hypothesis": hypotheses[best_index].ordering,
            "hypotheses": hypotheses,
            "acceptance_thresholds": acceptance_thresholds_json(config),
            "x_offset": x_offset,
            "y_offset": y_offset,
            "overlap": overlap,
            "objective_score": hypotheses[best_index].objective_score,
            "ncc_score": hypotheses[best_index].validation.global_ncc_score,
            "total_width": canvas_width,
            "canvas_height": canvas_height,
            "valid_union_pixel_count": valid_union_pixel_count,
            "canvas_void_pixel_count": canvas_void_pixel_count,
            "overlap_pixel_count": overlap_pixel_count,
            "overlap_outside_multiband_pixel_count": overlap_outside_multiband_pixel_count,
            "preserved_full_valid_union": preserved_full_valid_union,
            "target_crop_disabled": config.target_width.is_none() && config.target_height.is_none(),
            "order": order.label(),
            "affine_validation": diagnostics,
            "affine_matrix_right_to_canvas": right_to_canvas,
            "left_canvas_origin": left_origin,
            "multiband_overlap_rectangle": {
                "x": multiband_overlap_rectangle.x,
                "y": multiband_overlap_rectangle.y,
                "width": multiband_overlap_rectangle.width,
                "height": multiband_overlap_rectangle.height,
            },
            "exposure_gain": seam_exposure_correction.gain_rgb,
            "exposure_offset": seam_exposure_correction.offset_rgb,
            "seam_exposure_correction": seam_exposure_correction,
            "seam_blend": seam_blend,
            "crop": crop_metrics_json(crop_info),
        }),
    );
    report.warnings.extend(warnings);
    Ok(StitchResult {
        result: Some(stitched),
        x_offset: x_offset as i32,
        y_offset,
        ncc_score: hypotheses[best_index].validation.global_ncc_score,
        order,
        report,
    })
}

/// NCC-only translation stitch.
#[allow(clippy::too_many_arguments)]
fn stitch_ncc_fallback(
    left: &Array3<u16>,
    right: &Array3<u16>,
    x_offset: usize,
    y_offset: i32,
    overlap: usize,
    order: StitchOrder,
    config: &StitchConfig,
    hypotheses: &[StitchHypothesis],
    best_index: usize,
    warnings: Vec<String>,
) -> StitchResult {
    let (h_l, w_l, _) = left.dim();
    let (h_r, w_r, _) = right.dim();

    let abs_dy = y_offset.unsigned_abs() as usize;
    let l_y0 = if y_offset >= 0 { 0 } else { abs_dy };
    let r_y0 = if y_offset >= 0 { abs_dy } else { 0 };
    let canvas_h = (l_y0 + h_l).max(r_y0 + h_r);
    let total_w = x_offset + w_r;

    let mut stitched = Array3::<u16>::zeros((canvas_h, total_w, 3));
    stitched
        .slice_mut(s![l_y0..(l_y0 + h_l), 0..x_offset, ..])
        .assign(&left.slice(s![0..h_l, 0..x_offset, ..]));

    let blend_top = l_y0.max(r_y0);
    let blend_bot = (l_y0 + h_l).min(r_y0 + h_r);

    let seam_exposure_correction = if let Some((left_strip, right_strip)) =
        aligned_overlap_strips(left, right, overlap, y_offset)
    {
        seam_exposure_correction(&left_strip, &right_strip, config)
    } else {
        let empty = Array3::<u16>::zeros((0, 0, 3));
        seam_exposure_correction(&empty, &empty, config)
    };

    let photometric_field = applied_photometric_field(&seam_exposure_correction);
    let right_compensated = apply_photometric_correction(
        right,
        &photometric_field,
        0,
        overlap,
        if y_offset >= 0 { 0 } else { abs_dy },
        blend_bot.saturating_sub(blend_top),
        stitch_sample_max(config),
    );
    let (seam_blended_overlap, seam_blend_diagnostics) = if let Some((left_strip, right_strip)) =
        aligned_overlap_strips(left, right, overlap, y_offset)
    {
        let right_strip = apply_photometric_correction(
            &right_strip,
            &photometric_field,
            0,
            right_strip.dim().1,
            0,
            right_strip.dim().0,
            stitch_sample_max(config),
        );
        let (blended, diagnostics) = seam_aware_multiband_blend(&left_strip, &right_strip, config);
        (Some(blended), diagnostics)
    } else {
        let empty = Array3::<u16>::zeros((0, 0, 3));
        let (_, diagnostics) = seam_aware_multiband_blend(&empty, &empty, config);
        (None, diagnostics)
    };

    if blend_bot > blend_top {
        for canvas_y in blend_top..blend_bot {
            let ly = canvas_y - l_y0;
            let ry = canvas_y - r_y0;
            let blend_y = canvas_y - blend_top;
            for x in 0..overlap {
                let abs_x = x_offset + x;
                for c in 0..3 {
                    stitched[[canvas_y, abs_x, c]] = seam_blended_overlap
                        .as_ref()
                        .filter(|blended| blend_y < blended.dim().0 && x < blended.dim().1)
                        .map(|blended| blended[[blend_y, x, c]])
                        .unwrap_or_else(|| {
                            if x < overlap / 2 {
                                left[[ly, abs_x, c]]
                            } else {
                                right_compensated[[ry, x, c]]
                            }
                        });
                }
            }
        }

        for canvas_y in l_y0..(l_y0 + h_l) {
            if canvas_y >= blend_top && canvas_y < blend_bot {
                continue;
            }
            let ly = canvas_y - l_y0;
            for x in 0..overlap {
                let abs_x = x_offset + x;
                for c in 0..3 {
                    stitched[[canvas_y, abs_x, c]] = left[[ly, abs_x, c]];
                }
            }
        }

        for canvas_y in r_y0..(r_y0 + h_r) {
            if canvas_y >= blend_top && canvas_y < blend_bot {
                continue;
            }
            let ry = canvas_y - r_y0;
            for x in 0..overlap {
                let abs_x = x_offset + x;
                for c in 0..3 {
                    stitched[[canvas_y, abs_x, c]] = right_compensated[[ry, x, c]];
                }
            }
        }
    }

    if w_r > overlap {
        stitched
            .slice_mut(s![r_y0..(r_y0 + h_r), w_l..total_w, ..])
            .assign(&right_compensated.slice(s![0..h_r, overlap..w_r, ..]));
    }

    let (stitched, crop_info) = apply_target_crop(&stitched, config);
    let preserved_full_valid_union = crop_info.is_none();
    if let Some(debug_dir) = &config.debug_dir {
        save_final_seam_overlay(left, right, overlap, y_offset, order, debug_dir);
    }
    let order_gap = if hypotheses.len() >= 2 {
        (hypotheses[0].objective_score - hypotheses[1].objective_score).abs()
    } else {
        0.0
    };

    let mut report = PhaseReport::ok(
        "stitch",
        hypotheses[best_index].objective_score,
        serde_json::json!({
            "decision": "accepted",
            "method": "ncc_translation",
            "requested_transform": config.transform_mode.as_str(),
            "transform_model_used": hypotheses[best_index].transform_model,
            "opencv_requested": config.use_opencv,
            "opencv_available": opencv_match::is_available(),
            "chosen_hypothesis": hypotheses[best_index].ordering,
            "acceptance_thresholds": acceptance_thresholds_json(config),
            "hypotheses": hypotheses,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "overlap": overlap,
            "objective_score": hypotheses[best_index].objective_score,
            "ncc_score": hypotheses[best_index].validation.global_ncc_score,
            "order_gap": order_gap,
            "total_width": total_w,
            "canvas_height": canvas_h,
            "preserved_full_valid_union": preserved_full_valid_union,
            "target_crop_disabled": config.target_width.is_none() && config.target_height.is_none(),
            "order": order.label(),
            "exposure_gain": seam_exposure_correction.gain_rgb,
            "exposure_offset": seam_exposure_correction.offset_rgb,
            "seam_exposure_correction": seam_exposure_correction.clone(),
            "seam_blend": seam_blend_diagnostics,
            "crop": crop_metrics_json(crop_info),
        }),
    );
    report.warnings.extend(warnings);

    StitchResult {
        result: Some(stitched),
        x_offset: x_offset as i32,
        y_offset,
        ncc_score: hypotheses[best_index].validation.global_ncc_score,
        order,
        report,
    }
}

#[cfg(test)]
mod tests {
    use super::gain_in_bounds;
    use super::{
        applied_photometric_field, apply_photometric_correction, assess_homography_feature_quality,
        box_blur_array2, box_blur_array3, compose_projective_pair, default_validation_metrics,
        for_each_spatial_evaluation_grid_point, minimum_cost_vertical_seam,
        nearest_rank_percentile, objective_score, overlap_plausibility_reason,
        overlap_support_score, projected_pair_canvas_geometry, seam_aware_multiband_blend,
        seam_detail_consistency, seam_exposure_correction, spatial_field_grid_bounds,
        spatial_photometric_at_normalized_xy, translation_evidence_score,
        translation_plausibility_score, translation_prior_weight,
        translation_search_correspondence_score, translation_search_score,
        vertical_offset_plausibility_score, SpatialPhotometricField, StitchConfig,
        StitchValidationMetrics, TransformMode, SEAM_DETAIL_GRID_COLUMNS, SEAM_DETAIL_GRID_ROWS,
        SEAM_EXPOSURE_MAX_GAIN, SPATIAL_FIELD_CORNERS,
    };
    use crate::cv_adapter::opencv_match::MatchResult;
    use ndarray::{s, Array2, Array3};

    #[test]
    fn selected_percentile_matches_full_sort_nearest_rank() {
        let values = (0..10_003)
            .map(|index| ((index * 7919) % 997) as f64 / 37.0)
            .collect::<Vec<_>>();
        let mut sorted = values.clone();
        sorted.sort_by(|left, right| left.partial_cmp(right).unwrap());

        for percentile in [0.0, 0.10, 0.50, 0.90, 0.95, 1.0] {
            let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
            assert_eq!(
                nearest_rank_percentile(values.clone(), percentile),
                sorted[index]
            );
        }
    }

    fn textured_seam_strips(h: usize, w: usize) -> (Array3<u16>, Array3<u16>) {
        let mut left = Array3::<u16>::zeros((h, w, 3));
        let mut right = Array3::<u16>::zeros((h, w, 3));
        for y in 0..h {
            for x in 0..w {
                let texture = ((x * 131 + y * 79 + (x * y) % 37) % 1400) as f64;
                let base = [
                    4200.0 + x as f64 * 14.0 + y as f64 * 5.0 + texture,
                    5000.0 + x as f64 * 7.0 + y as f64 * 11.0 + texture * 0.8,
                    4600.0 + x as f64 * 10.0 + y as f64 * 8.0 + texture * 0.6,
                ];
                for c in 0..3 {
                    left[[y, x, c]] = base[c].round() as u16;
                    right[[y, x, c]] = base[c].round() as u16;
                }
            }
        }
        (left, right)
    }

    fn blur_u16_image(image: &Array3<u16>, radius: usize) -> Array3<u16> {
        let floating = image.mapv(|value| value as f64);
        box_blur_array3(&floating, radius).mapv(|value| value.round().clamp(0.0, 16383.0) as u16)
    }

    fn rotated_textured_split_pair(
        height: usize,
        panorama_width: usize,
        component_width: usize,
        overlap: usize,
        angle_deg: f64,
    ) -> (Array3<u16>, Array3<u16>) {
        let padding = 48usize;
        let panorama_height = height + padding * 2;
        let mut panorama = Array3::<u16>::zeros((panorama_height, panorama_width, 3));
        for y in 0..panorama_height {
            for x in 0..panorama_width {
                let wave = ((x as f64 * 0.071).sin() * 1250.0
                    + (y as f64 * 0.113).cos() * 900.0
                    + ((x + y) as f64 * 0.037).sin() * 620.0)
                    .round();
                let texture = ((x * 131 + y * 79 + (x * y) % 193) % 2100) as f64;
                let values = [
                    5400.0 + x as f64 * 4.0 + y as f64 * 2.0 + wave + texture,
                    6100.0 + x as f64 * 2.0 + y as f64 * 5.0 + wave * 0.72 + texture * 0.8,
                    4800.0 + x as f64 * 3.0 + y as f64 * 3.0 + wave * 0.45 + texture * 0.6,
                ];
                for channel in 0..3 {
                    panorama[[y, x, channel]] = values[channel].clamp(0.0, 16383.0) as u16;
                }
            }
        }
        let x_offset = component_width - overlap;
        let left = panorama
            .slice(s![padding..padding + height, 0..component_width, ..])
            .to_owned();
        let radians = angle_deg.to_radians();
        let cosine = radians.cos();
        let sine = radians.sin();
        let center_x = component_width.saturating_sub(1) as f64 * 0.5;
        let center_y = height.saturating_sub(1) as f64 * 0.5;
        let mut right = Array3::<u16>::zeros((height, component_width, 3));
        for y in 0..height {
            for x in 0..component_width {
                let relative_x = x as f64 - center_x;
                let relative_y = y as f64 - center_y;
                let panorama_x =
                    x_offset as f64 + center_x + cosine * relative_x - sine * relative_y;
                let panorama_y =
                    padding as f64 + center_y + sine * relative_x + cosine * relative_y;
                let x0 = panorama_x.floor() as usize;
                let y0 = panorama_y.floor() as usize;
                let x1 = (x0 + 1).min(panorama_width - 1);
                let y1 = (y0 + 1).min(panorama_height - 1);
                let fx = panorama_x - x0 as f64;
                let fy = panorama_y - y0 as f64;
                for channel in 0..3 {
                    let top = panorama[[y0, x0, channel]] as f64 * (1.0 - fx)
                        + panorama[[y0, x1, channel]] as f64 * fx;
                    let bottom = panorama[[y1, x0, channel]] as f64 * (1.0 - fx)
                        + panorama[[y1, x1, channel]] as f64 * fx;
                    right[[y, x, channel]] = (top * (1.0 - fy) + bottom * fy).round() as u16;
                }
            }
        }
        (left, right)
    }

    fn projective_textured_split_pair(
        height: usize,
        panorama_width: usize,
        component_width: usize,
        overlap: usize,
        right_to_left: [[f64; 3]; 3],
    ) -> (Array3<u16>, Array3<u16>) {
        let padding = 48usize;
        let panorama_height = height + padding * 2;
        let mut panorama = Array3::<u16>::zeros((panorama_height, panorama_width, 3));
        for y in 0..panorama_height {
            for x in 0..panorama_width {
                let wave = ((x as f64 * 0.067).sin() * 1350.0
                    + (y as f64 * 0.109).cos() * 950.0
                    + ((x * 3 + y * 5) as f64 * 0.031).sin() * 700.0)
                    .round();
                let texture = ((x * 137 + y * 83 + (x * y) % 211) % 2300) as f64;
                let values = [
                    5200.0 + x as f64 * 4.0 + y as f64 * 2.0 + wave + texture,
                    6000.0 + x as f64 * 2.0 + y as f64 * 5.0 + wave * 0.73 + texture * 0.82,
                    4700.0 + x as f64 * 3.0 + y as f64 * 3.0 + wave * 0.47 + texture * 0.63,
                ];
                for channel in 0..3 {
                    panorama[[y, x, channel]] = values[channel].clamp(0.0, 16383.0) as u16;
                }
            }
        }

        let expected_x_offset = component_width - overlap;
        assert!((right_to_left[0][2] - expected_x_offset as f64).abs() < 1e-9);
        let left = panorama
            .slice(s![padding..padding + height, 0..component_width, ..])
            .to_owned();
        let mut right = Array3::<u16>::zeros((height, component_width, 3));
        for y in 0..height {
            for x in 0..component_width {
                let (panorama_x, left_y) =
                    super::transform_point(right_to_left, x as f64, y as f64);
                let panorama_y = padding as f64 + left_y;
                assert!(
                    panorama_x >= 0.0
                        && panorama_y >= 0.0
                        && panorama_x <= panorama_width.saturating_sub(1) as f64
                        && panorama_y <= panorama_height.saturating_sub(1) as f64,
                    "projective fixture sampled outside its panorama at ({panorama_x}, {panorama_y})"
                );
                let x0 = panorama_x.floor() as usize;
                let y0 = panorama_y.floor() as usize;
                let x1 = (x0 + 1).min(panorama_width - 1);
                let y1 = (y0 + 1).min(panorama_height - 1);
                let fx = panorama_x - x0 as f64;
                let fy = panorama_y - y0 as f64;
                for channel in 0..3 {
                    let top = panorama[[y0, x0, channel]] as f64 * (1.0 - fx)
                        + panorama[[y0, x1, channel]] as f64 * fx;
                    let bottom = panorama[[y1, x0, channel]] as f64 * (1.0 - fx)
                        + panorama[[y1, x1, channel]] as f64 * fx;
                    right[[y, x, channel]] = (top * (1.0 - fy) + bottom * fy).round() as u16;
                }
            }
        }
        (left, right)
    }

    #[test]
    fn native_affine_uses_held_out_improvement_for_rotated_split_scan() {
        let (left, right) = rotated_textured_split_pair(260, 960, 520, 180, 0.72);
        let config = StitchConfig {
            transform_mode: TransformMode::Auto,
            max_overlap: 240,
            ..StitchConfig::default()
        };

        let result = super::stitch_components(&left, &right, &config);

        assert!(result.result.is_some(), "{}", result.report.metrics);
        assert_eq!(result.report.metrics["method"], "native_affine");
        assert_eq!(
            result.report.metrics["transform_model_used"],
            "native_affine"
        );
        let diagnostics = &result.report.metrics["affine_validation"];
        assert_eq!(diagnostics["accepted"], true, "{diagnostics}");
        assert!(
            diagnostics["held_out_ncc_improvement"]
                .as_f64()
                .is_some_and(|improvement| improvement >= 0.006),
            "{diagnostics}"
        );
        assert!(
            diagnostics["held_out_error_reduction"]
                .as_f64()
                .is_some_and(|reduction| reduction >= 0.08),
            "{diagnostics}"
        );
        assert!(
            (diagnostics["rotation_deg"].as_f64().unwrap() - 0.72).abs() < 0.15,
            "{diagnostics}"
        );
        assert_eq!(
            diagnostics["rotation_search_boundary_hit"], false,
            "{diagnostics}"
        );
        assert_eq!(result.report.metrics["preserved_full_valid_union"], true);
        assert_eq!(
            result.report.metrics["seam_blend"]["mode"],
            "seam_aware_multiband"
        );
    }

    #[test]
    fn homography_selection_requires_disjoint_feature_validation() {
        let mut result = MatchResult {
            match_count: 48,
            training_match_count: 24,
            held_out_match_count: 24,
            training_spatial_cell_count: 8,
            held_out_spatial_cell_count: 8,
            inliers: 22,
            training_inlier_ratio: 22.0 / 24.0,
            training_median_error: 0.35,
            training_p95_error: 0.8,
            median_error: 0.42,
            p95_error: 1.1,
            held_out_inlier_count: 21,
            held_out_inlier_ratio: 21.0 / 24.0,
            reverse_validation_inlier_count: 22,
            reverse_validation_inlier_ratio: 22.0 / 24.0,
            reverse_validation_median_error: 0.45,
            reverse_validation_p95_error: 1.2,
            cross_fit_max_disagreement_px: 0.6,
            disjoint_validation_passed: true,
            validation_reason: "spatially disjoint cross-fit passed".to_string(),
            partition_method: "4x4_whole_cell_checkerboard_cross_fit",
            transform: [[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        assert!(assess_homography_feature_quality(&result).is_ok());

        result.disjoint_validation_passed = false;
        result.validation_reason =
            "training-fit homography did not generalize to held-out cells".to_string();
        let rejection = assess_homography_feature_quality(&result).unwrap_err();

        assert!(rejection.contains("did not generalize"));
    }

    #[test]
    fn homography_spatial_validation_requires_repeatable_projective_improvement() {
        let component_width = 560usize;
        let overlap = 180usize;
        let x_offset = component_width - overlap;
        let candidate = [
            [1.0, 0.0015, x_offset as f64],
            [0.0008, 1.0, -1.0],
            [0.000_022, -0.000_007, 1.0],
        ];
        let (left, right) =
            projective_textured_split_pair(280, 1020, component_width, overlap, candidate);

        let diagnostics = super::validate_homography_spatial_evidence(
            &left, &right, x_offset, 0, overlap, candidate,
        );

        assert!(diagnostics.accepted, "{diagnostics:?}");
        assert!(
            diagnostics
                .sample_count
                .iter()
                .all(|samples| *samples >= 768),
            "{diagnostics:?}"
        );
        assert!(
            diagnostics
                .ncc_improvement
                .iter()
                .all(|improvement| *improvement >= 0.003),
            "{diagnostics:?}"
        );
        assert!(diagnostics.mean_ncc_improvement >= 0.006, "{diagnostics:?}");
        assert!(
            diagnostics.mean_registration_error_reduction >= 0.08,
            "{diagnostics:?}"
        );
    }

    #[test]
    fn homography_spatial_validation_rejects_translation_equivalent_complexity() {
        let component_width = 560usize;
        let overlap = 180usize;
        let x_offset = component_width - overlap;
        let translation = [
            [1.0, 0.0, x_offset as f64],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let (left, right) =
            projective_textured_split_pair(280, 1020, component_width, overlap, translation);

        let diagnostics = super::validate_homography_spatial_evidence(
            &left,
            &right,
            x_offset,
            0,
            overlap,
            translation,
        );

        assert!(!diagnostics.accepted, "{diagnostics:?}");
        assert!(diagnostics.reason.contains("indistinguishable"));
        assert!(
            diagnostics.maximum_deviation_from_translation_px < 1e-9,
            "{diagnostics:?}"
        );
    }

    #[test]
    fn projective_compositor_tracks_valid_black_pixels_with_an_explicit_mask() {
        let left = Array3::<u16>::zeros((96, 160, 3));
        let right = Array3::<u16>::zeros((96, 160, 3));
        let right_to_left = [[1.0, 0.0, 100.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let composition =
            compose_projective_pair(&left, &right, right_to_left, &StitchConfig::default())
                .expect("valid translated projective composition");

        assert_eq!(composition.image.dim(), (96, 260, 3));
        assert_eq!(composition.valid_union_pixel_count, 96 * 260);
        assert_eq!(composition.canvas_void_pixel_count, 0);
        assert_eq!(composition.overlap_pixel_count, 96 * 60);
        assert_eq!(composition.multiband_overlap_rectangle.width, 60);
        assert_eq!(composition.multiband_overlap_rectangle.height, 96);
        assert!(composition.image.iter().all(|value| *value == 0));
        assert!(composition.seam_blend.applied);
        assert_eq!(composition.seam_blend.mode, "seam_aware_multiband");
    }

    #[test]
    fn projective_compositor_preserves_union_and_multiband_blend_under_perspective() {
        let (left, right) = textured_seam_strips(160, 220);
        let right_to_left = [
            [1.0, 0.002, 140.0],
            [0.001, 1.0, -2.0],
            [0.000_015, -0.000_008, 1.0],
        ];

        let composition =
            compose_projective_pair(&left, &right, right_to_left, &StitchConfig::default())
                .expect("bounded projective composition");

        assert!(composition.right_to_canvas[2][0].abs() > 0.0);
        assert!(composition.valid_union_pixel_count > left.dim().0 * left.dim().1);
        assert!(composition.overlap_pixel_count > 1_000);
        assert!(composition.multiband_overlap_rectangle.width >= 12);
        assert!(composition.multiband_overlap_rectangle.height >= 16);
        assert!(composition.seam_blend.applied);
        assert!(
            composition.overlap_outside_multiband_pixel_count < composition.overlap_pixel_count
        );
    }

    #[test]
    fn projective_canvas_rejects_horizon_crossing_and_implausible_expansion() {
        let horizon = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -0.02, 1.0]];
        let horizon_error = projected_pair_canvas_geometry(160, 100, 160, 100, horizon)
            .expect_err("horizon crossing must be rejected");
        assert!(horizon_error.contains("horizon"), "{horizon_error}");

        let expansion = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 1.0]];
        let expansion_error = projected_pair_canvas_geometry(160, 100, 160, 100, expansion)
            .expect_err("implausible canvas expansion must be rejected");
        assert!(expansion_error.contains("expansion"), "{expansion_error}");
    }

    fn scale_strip(strip: &mut Array3<u16>, gain: [f64; 3]) {
        let (h, w, _) = strip.dim();
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    strip[[y, x, c]] = (strip[[y, x, c]] as f64 * gain[c])
                        .round()
                        .clamp(0.0, 16383.0) as u16;
                }
            }
        }
    }

    fn offset_strip(strip: &mut Array3<u16>, offset: [f64; 3]) {
        let (h, w, _) = strip.dim();
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    strip[[y, x, c]] = (strip[[y, x, c]] as f64 + offset[c])
                        .round()
                        .clamp(0.0, 16383.0) as u16;
                }
            }
        }
    }

    fn sample_validation_metrics(
        global_ncc_score: f64,
        signature_score: f64,
        overlap_support_score: f64,
        vertical_offset_plausibility_score: f64,
        local_consistency_score: f64,
    ) -> StitchValidationMetrics {
        let mut metrics = default_validation_metrics();
        metrics.global_ncc_score = global_ncc_score;
        metrics.signature_score = signature_score;
        metrics.overlap_support_score = overlap_support_score;
        metrics.vertical_offset_plausibility_score = vertical_offset_plausibility_score;
        metrics.plausibility_score = translation_plausibility_score(
            overlap_support_score,
            vertical_offset_plausibility_score,
        );
        metrics.local_consistency_score = local_consistency_score;
        metrics.evidence_score =
            translation_evidence_score(global_ncc_score, signature_score, local_consistency_score);
        metrics.prior_weight =
            translation_prior_weight(overlap_support_score, vertical_offset_plausibility_score);
        metrics.local_windows_evaluated = 36;
        metrics.plausibility_ok = true;
        metrics.plausibility_reason = None;
        metrics
    }

    #[test]
    fn seam_exposure_skips_low_overlap_samples() {
        let (left, mut right) = textured_seam_strips(24, 12);
        scale_strip(&mut right, [0.88, 0.88, 0.88]);

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(!correction.applied);
        assert_eq!(correction.gain_rgb, [1.0, 1.0, 1.0]);
        assert!(
            correction.reason.contains("insufficient reliable overlap")
                || correction
                    .reason
                    .contains("insufficient reliable overlap samples"),
            "{}",
            correction.reason
        );
    }

    #[test]
    fn seam_exposure_skips_film_base_like_overlap() {
        let left = Array3::<u16>::from_elem((96, 48, 3), 9000);
        let right = Array3::<u16>::from_elem((96, 48, 3), 7800);

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(!correction.applied);
        assert_eq!(correction.gain_rgb, [1.0, 1.0, 1.0]);
        assert!(correction
            .reason
            .contains("insufficient reliable overlap windows"));
    }

    #[test]
    fn seam_exposure_rejects_correction_that_increases_clipping() {
        let (left, mut right) = textured_seam_strips(120, 64);
        scale_strip(&mut right, [0.86, 0.86, 0.86]);
        let (h, w, _) = right.dim();
        for y in 0..h {
            for x in 0..w {
                if (x + y) % 7 == 0 {
                    for c in 0..3 {
                        right[[y, x, c]] = 16000;
                    }
                }
            }
        }

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(!correction.applied);
        assert_eq!(correction.gain_rgb, [1.0, 1.0, 1.0]);
        assert!(
            correction.reason.contains("increase clipping"),
            "{}",
            correction.reason
        );
        let before: f64 = correction.clipped_high_before.iter().sum();
        let candidate_after: f64 = correction.candidate_clipped_high_after.iter().sum();
        assert!(candidate_after > before);
    }

    #[test]
    fn seam_exposure_accepts_additive_flare_only_after_held_out_improvement() {
        let (left, mut right) = textured_seam_strips(192, 128);
        offset_strip(&mut right, [780.0, 760.0, 800.0]);

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(correction.applied, "{}", correction.reason);
        assert_eq!(correction.model, "gain_offset_rgb", "{}", correction.reason);
        assert!(correction.held_out_validation_passed);
        assert!(
            correction.held_out_gain_offset_seam_score
                < correction.held_out_gain_seam_score - 0.006,
            "{correction:?}"
        );
        assert!(correction.gain_offset_training_window_count >= 3);
        assert!(correction.gain_offset_held_out_window_count >= 3);
        for (actual, expected) in correction.offset_rgb.iter().zip([-780.0, -760.0, -800.0]) {
            assert!((actual - expected).abs() < 8.0, "{correction:?}");
        }
    }

    #[test]
    fn seam_exposure_does_not_invent_offset_for_multiplicative_mismatch() {
        let (left, mut right) = textured_seam_strips(192, 128);
        scale_strip(&mut right, [0.88, 0.88, 0.88]);

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(correction.applied, "{}", correction.reason);
        assert_eq!(correction.model, "gain_only_scalar", "{correction:?}");
        assert_eq!(correction.offset_rgb, [0.0; 3]);
        assert!(correction
            .gain_offset_rejection_reason
            .contains("unnecessary"));
    }

    #[test]
    fn seam_exposure_rejects_spatially_inconsistent_additive_model() {
        let (left, mut right) = textured_seam_strips(384, 128);
        let (h, w, _) = right.dim();
        for y in 0..h {
            for x in 0..w {
                let offset = if ((y / 96) + (x / 32)) % 2 == 0 {
                    720.0
                } else {
                    180.0
                };
                for c in 0..3 {
                    right[[y, x, c]] = (right[[y, x, c]] as f64 + offset)
                        .round()
                        .clamp(0.0, 16383.0) as u16;
                }
            }
        }

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(
            correction.model != "gain_offset_rgb"
                && correction.model != "gain_offset_spatial_y_rgb"
                && correction.model != "gain_offset_spatial_xy_rgb"
                && correction.model != "gain_offset_spatial_quadratic_xy_rgb",
            "{correction:?}"
        );
        assert!(
            correction.gain_offset_consistent_window_ratio < 0.65,
            "{correction:?}"
        );
        assert!(!correction.held_out_validation_passed, "{correction:?}");
    }

    #[test]
    fn seam_exposure_accepts_held_out_validated_vertical_gain_field() {
        let (left, mut right) = textured_seam_strips(384, 128);
        let source_slopes = [0.14, 0.11, 0.08];
        let (height, width, _) = right.dim();
        for y in 0..height {
            let y_normalized = 2.0 * y as f64 / (height - 1) as f64 - 1.0;
            for x in 0..width {
                for channel in 0..3 {
                    let shade = (source_slopes[channel] * y_normalized).exp();
                    right[[y, x, channel]] = (right[[y, x, channel]] as f64 * shade)
                        .round()
                        .clamp(0.0, 16383.0) as u16;
                }
            }
        }

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(correction.applied, "{correction:?}");
        assert_eq!(correction.model, "gain_spatial_y_rgb", "{correction:?}");
        assert!(correction.held_out_validation_passed, "{correction:?}");
        assert!(correction.spatial_distinct_training_rows >= 3);
        assert!(correction.spatial_distinct_held_out_rows >= 3);
        assert!(correction.spatial_slope_agreement_ratio >= 2.0 / 3.0);
        assert!(
            correction.held_out_spatial_gain_seam_score
                < correction.held_out_gain_seam_score - 0.004,
            "{correction:?}"
        );
        for (actual, source) in correction
            .spatial_gain_log_slope_y_rgb
            .iter()
            .zip(source_slopes)
        {
            assert!((actual + source).abs() < 0.012, "{correction:?}");
        }

        let field = applied_photometric_field(&correction);
        let corrected = apply_photometric_correction(&right, &field, 0, width, 0, height, 16383.0);
        let mean_absolute_error = corrected
            .iter()
            .zip(left.iter())
            .map(|(actual, expected)| (*actual as f64 - *expected as f64).abs())
            .sum::<f64>()
            / corrected.len() as f64;
        assert!(mean_absolute_error < 3.0, "{correction:?}");
    }

    #[test]
    fn seam_exposure_accepts_held_out_validated_vertical_gain_offset_field() {
        let (left, mut right) = textured_seam_strips(768, 128);
        let expected_center_gain = [0.98, 1.02, 1.00];
        let expected_gain_slopes = [-0.12, -0.09, -0.07];
        let expected_center_offset = [-500.0, -450.0, -550.0];
        let expected_offset_slopes = [160.0, 120.0, 90.0];
        let (height, width, _) = right.dim();
        for y in 0..height {
            let y_normalized = 2.0 * y as f64 / (height - 1) as f64 - 1.0;
            for x in 0..width {
                for channel in 0..3 {
                    let gain = expected_center_gain[channel]
                        * (expected_gain_slopes[channel] * y_normalized).exp();
                    let offset = expected_center_offset[channel]
                        + expected_offset_slopes[channel] * y_normalized;
                    right[[y, x, channel]] = ((left[[y, x, channel]] as f64 - offset) / gain)
                        .round()
                        .clamp(0.0, 16383.0) as u16;
                }
            }
        }

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(correction.applied, "{correction:?}");
        assert_eq!(
            correction.model, "gain_offset_spatial_y_rgb",
            "{correction:?}"
        );
        assert!(correction.held_out_validation_passed, "{correction:?}");
        assert!(correction.spatial_affine_distinct_training_rows >= 4);
        assert!(correction.spatial_affine_distinct_held_out_rows >= 4);
        assert!(correction.spatial_affine_slope_agreement_ratio >= 2.0 / 3.0);
        assert!(
            correction.held_out_spatial_gain_offset_seam_score
                < correction.held_out_spatial_gain_seam_score - 0.0025,
            "{correction:?}"
        );
        for channel in 0..3 {
            assert!(
                (correction.spatial_gain_log_slope_y_rgb[channel] - expected_gain_slopes[channel])
                    .abs()
                    < 0.035,
                "{correction:?}"
            );
            assert!(
                (correction.offset_rgb[channel] - expected_center_offset[channel]).abs() < 180.0,
                "{correction:?}"
            );
            assert!(
                (correction.spatial_offset_slope_y_rgb[channel] - expected_offset_slopes[channel])
                    .abs()
                    < 120.0,
                "{correction:?}"
            );
        }

        let field = applied_photometric_field(&correction);
        let corrected = apply_photometric_correction(&right, &field, 0, width, 0, height, 16383.0);
        let mean_absolute_error = corrected
            .iter()
            .zip(left.iter())
            .map(|(actual, expected)| (*actual as f64 - *expected as f64).abs())
            .sum::<f64>()
            / corrected.len() as f64;
        assert!(mean_absolute_error < 35.0, "{correction:?}");
    }

    #[test]
    fn seam_exposure_accepts_held_out_validated_2d_gain_field() {
        let (left, mut right) = textured_seam_strips(768, 192);
        let expected_center_gain = [1.00, 0.99, 1.01];
        let expected_x_slopes = [-0.12, -0.10, -0.08];
        let expected_y_slopes = [-0.04, -0.03, -0.02];
        let (height, width, _) = right.dim();
        for y in 0..height {
            let y_normalized = 2.0 * y as f64 / (height - 1) as f64 - 1.0;
            for x in 0..width {
                let x_normalized = 2.0 * x as f64 / (width - 1) as f64 - 1.0;
                for channel in 0..3 {
                    let gain = expected_center_gain[channel]
                        * (expected_x_slopes[channel] * x_normalized
                            + expected_y_slopes[channel] * y_normalized)
                            .exp();
                    right[[y, x, channel]] = (left[[y, x, channel]] as f64 / gain)
                        .round()
                        .clamp(0.0, 16383.0) as u16;
                }
            }
        }

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(correction.applied, "{correction:?}");
        assert_eq!(correction.model, "gain_spatial_xy_rgb", "{correction:?}");
        assert!(correction.held_out_validation_passed, "{correction:?}");
        assert!(correction.spatial_2d_validation.gain_accepted);
        assert!(correction.spatial_2d_validation.distinct_training_columns >= 3);
        assert!(correction.spatial_2d_validation.distinct_held_out_columns >= 3);
        assert!(
            correction
                .spatial_2d_validation
                .gain_horizontal_slope_agreement_ratio
                >= 2.0 / 3.0
        );
        for channel in 0..3 {
            assert!(
                (correction.spatial_gain_log_slope_x_rgb[channel] - expected_x_slopes[channel])
                    .abs()
                    < 0.025,
                "{correction:?}"
            );
            assert!(
                (correction.spatial_gain_log_slope_y_rgb[channel] - expected_y_slopes[channel])
                    .abs()
                    < 0.025,
                "{correction:?}"
            );
        }

        let field = applied_photometric_field(&correction);
        let corrected = apply_photometric_correction(&right, &field, 0, width, 0, height, 16383.0);
        let mean_absolute_error = corrected
            .iter()
            .zip(left.iter())
            .map(|(actual, expected)| (*actual as f64 - *expected as f64).abs())
            .sum::<f64>()
            / corrected.len() as f64;
        assert!(mean_absolute_error < 5.0, "{correction:?}");
    }

    #[test]
    fn seam_exposure_accepts_held_out_validated_2d_gain_offset_field() {
        let (left, mut right) = textured_seam_strips(768, 192);
        let expected_center_gain = [0.98, 1.01, 1.00];
        let expected_gain_x_slopes = [-0.08, -0.06, -0.05];
        let expected_gain_y_slopes = [-0.05, -0.04, -0.03];
        let expected_center_offset = [-460.0, -500.0, -420.0];
        let expected_offset_x_slopes = [150.0, 120.0, 90.0];
        let expected_offset_y_slopes = [100.0, 80.0, 70.0];
        let (height, width, _) = right.dim();
        for y in 0..height {
            let y_normalized = 2.0 * y as f64 / (height - 1) as f64 - 1.0;
            for x in 0..width {
                let x_normalized = 2.0 * x as f64 / (width - 1) as f64 - 1.0;
                for channel in 0..3 {
                    let gain = expected_center_gain[channel]
                        * (expected_gain_x_slopes[channel] * x_normalized
                            + expected_gain_y_slopes[channel] * y_normalized)
                            .exp();
                    let offset = expected_center_offset[channel]
                        + expected_offset_x_slopes[channel] * x_normalized
                        + expected_offset_y_slopes[channel] * y_normalized;
                    right[[y, x, channel]] = ((left[[y, x, channel]] as f64 - offset) / gain)
                        .round()
                        .clamp(0.0, 16383.0) as u16;
                }
            }
        }

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(correction.applied, "{correction:?}");
        assert_eq!(
            correction.model, "gain_offset_spatial_xy_rgb",
            "{correction:?}"
        );
        assert!(correction.held_out_validation_passed, "{correction:?}");
        assert!(correction.spatial_2d_validation.gain_offset_accepted);
        assert!(
            correction
                .spatial_2d_validation
                .gain_offset_horizontal_slope_agreement_ratio
                >= 2.0 / 3.0
        );
        for channel in 0..3 {
            assert!(
                (correction.spatial_gain_log_slope_x_rgb[channel]
                    - expected_gain_x_slopes[channel])
                    .abs()
                    < 0.035,
                "{correction:?}"
            );
            assert!(
                (correction.spatial_offset_slope_x_rgb[channel]
                    - expected_offset_x_slopes[channel])
                    .abs()
                    < 130.0,
                "{correction:?}"
            );
        }

        let field = applied_photometric_field(&correction);
        let corrected = apply_photometric_correction(&right, &field, 0, width, 0, height, 16383.0);
        let mean_absolute_error = corrected
            .iter()
            .zip(left.iter())
            .map(|(actual, expected)| (*actual as f64 - *expected as f64).abs())
            .sum::<f64>()
            / corrected.len() as f64;
        assert!(mean_absolute_error < 45.0, "{correction:?}");
    }

    #[test]
    fn seam_exposure_accepts_held_out_validated_quadratic_2d_gain_field() {
        let (left, mut right) = textured_seam_strips(768, 192);
        let expected_center_gain = [1.00, 0.99, 1.01];
        let expected_x_slopes = [-0.025, -0.020, -0.015];
        let expected_y_slopes = [-0.020, -0.015, -0.010];
        let expected_xx = [-0.090, -0.080, -0.070];
        let expected_xy = [0.030, 0.026, 0.022];
        let expected_yy = [-0.060, -0.052, -0.045];
        let (height, width, _) = right.dim();
        for y in 0..height {
            let y_normalized = 2.0 * y as f64 / (height - 1) as f64 - 1.0;
            for x in 0..width {
                let x_normalized = 2.0 * x as f64 / (width - 1) as f64 - 1.0;
                for channel in 0..3 {
                    let log_gain = expected_x_slopes[channel] * x_normalized
                        + expected_y_slopes[channel] * y_normalized
                        + expected_xx[channel] * x_normalized * x_normalized
                        + expected_xy[channel] * x_normalized * y_normalized
                        + expected_yy[channel] * y_normalized * y_normalized;
                    let gain = expected_center_gain[channel] * log_gain.exp();
                    right[[y, x, channel]] = (left[[y, x, channel]] as f64 / gain)
                        .round()
                        .clamp(0.0, 16383.0) as u16;
                }
            }
        }

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(correction.applied, "{correction:?}");
        assert_eq!(
            correction.model, "gain_spatial_quadratic_xy_rgb",
            "{correction:?}"
        );
        assert!(correction.held_out_validation_passed, "{correction:?}");
        assert!(correction.spatial_2d_validation.quadratic.gain_accepted);
        assert!(
            correction
                .spatial_2d_validation
                .quadratic
                .gain_curvature_coefficient_agreement_ratio
                >= 2.0 / 3.0,
            "{correction:?}"
        );
        for channel in 0..3 {
            assert!(
                (correction.spatial_gain_log_quadratic_xx_rgb[channel] - expected_xx[channel])
                    .abs()
                    < 0.025,
                "{correction:?}"
            );
            assert!(
                (correction.spatial_gain_log_quadratic_yy_rgb[channel] - expected_yy[channel])
                    .abs()
                    < 0.025,
                "{correction:?}"
            );
        }

        let field = applied_photometric_field(&correction);
        let corrected = apply_photometric_correction(&right, &field, 0, width, 0, height, 16383.0);
        let mean_absolute_error = corrected
            .iter()
            .zip(left.iter())
            .map(|(actual, expected)| (*actual as f64 - *expected as f64).abs())
            .sum::<f64>()
            / corrected.len() as f64;
        assert!(mean_absolute_error < 12.0, "{correction:?}");
    }

    #[test]
    fn seam_exposure_accepts_held_out_validated_quadratic_2d_gain_offset_field() {
        let (left, mut right) = textured_seam_strips(768, 192);
        let center_gain = [0.99, 1.01, 1.00];
        let gain_xx = [0.090, 0.080, 0.075];
        let gain_xy = [0.025, 0.021, 0.017];
        let gain_yy = [0.065, 0.060, 0.055];
        let center_offset = [0.0, 0.0, 0.0];
        let offset_xx = [600.0, 550.0, 500.0];
        let offset_xy = [80.0, 70.0, 60.0];
        let offset_yy = [-600.0, -550.0, -500.0];
        let (height, width, _) = right.dim();
        for y in 0..height {
            let y_normalized = 2.0 * y as f64 / (height - 1) as f64 - 1.0;
            for x in 0..width {
                let x_normalized = 2.0 * x as f64 / (width - 1) as f64 - 1.0;
                for channel in 0..3 {
                    let log_gain = -0.02 * x_normalized - 0.015 * y_normalized
                        + gain_xx[channel] * x_normalized * x_normalized
                        + gain_xy[channel] * x_normalized * y_normalized
                        + gain_yy[channel] * y_normalized * y_normalized;
                    let gain = center_gain[channel] * log_gain.exp();
                    let offset = center_offset[channel]
                        + offset_xx[channel] * x_normalized * x_normalized
                        + offset_xy[channel] * x_normalized * y_normalized
                        + offset_yy[channel] * y_normalized * y_normalized;
                    right[[y, x, channel]] = ((left[[y, x, channel]] as f64 - offset) / gain)
                        .round()
                        .clamp(0.0, 16383.0) as u16;
                }
            }
        }

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(correction.applied, "{correction:?}");
        assert_eq!(
            correction.model, "gain_offset_spatial_quadratic_xy_rgb",
            "{correction:?}"
        );
        assert!(
            correction
                .spatial_2d_validation
                .quadratic
                .gain_offset_accepted,
            "{correction:?}"
        );
        for channel in 0..3 {
            assert!(
                (correction.spatial_gain_log_quadratic_xx_rgb[channel] - gain_xx[channel]).abs()
                    < 0.04,
                "{correction:?}"
            );
            assert!(
                (correction.spatial_offset_quadratic_xx_rgb[channel] - offset_xx[channel]).abs()
                    < 170.0,
                "{correction:?}"
            );
        }

        let field = applied_photometric_field(&correction);
        let corrected = apply_photometric_correction(&right, &field, 0, width, 0, height, 16383.0);
        let mean_absolute_error = corrected
            .iter()
            .zip(left.iter())
            .map(|(actual, expected)| (*actual as f64 - *expected as f64).abs())
            .sum::<f64>()
            / corrected.len() as f64;
        assert!(mean_absolute_error < 60.0, "{correction:?}");
    }

    #[test]
    fn quadratic_field_bounds_sample_interior_extrema() {
        let field = SpatialPhotometricField {
            center_gain: [1.219; 3],
            center_offset: [0.0; 3],
            log_gain_slope_x: [0.11; 3],
            log_gain_slope_y: [0.0; 3],
            log_gain_quadratic_xx: [-0.12; 3],
            log_gain_quadratic_xy: [0.0; 3],
            log_gain_quadratic_yy: [0.0; 3],
            offset_slope_x: [0.0; 3],
            offset_slope_y: [0.0; 3],
            offset_quadratic_xx: [0.0; 3],
            offset_quadratic_xy: [0.0; 3],
            offset_quadratic_yy: [0.0; 3],
        };
        let corners = SPATIAL_FIELD_CORNERS
            .map(|[x, y]| spatial_photometric_at_normalized_xy(&field, x, y).0);
        assert!(corners.iter().all(gain_in_bounds));
        let mut grid_max = 0.0f64;
        for_each_spatial_evaluation_grid_point(|x, y| {
            grid_max = grid_max.max(spatial_photometric_at_normalized_xy(&field, x, y).0[0]);
        });
        assert!(grid_max <= SEAM_EXPOSURE_MAX_GAIN);
        let bounds = spatial_field_grid_bounds(&field, 16383.0);
        assert!(bounds.gain_max_rgb[0] > SEAM_EXPOSURE_MAX_GAIN);
    }

    #[test]
    fn seam_exposure_rejects_2d_field_confined_to_training_windows() {
        let (left, mut right) = textured_seam_strips(768, 192);
        let (height, width, _) = right.dim();
        for y in 0..height {
            let row = y * 8 / height;
            let y_normalized = 2.0 * y as f64 / (height - 1) as f64 - 1.0;
            for x in 0..width {
                let col = x * 6 / width;
                if !(row + col).is_multiple_of(2) {
                    continue;
                }
                let x_normalized = 2.0 * x as f64 / (width - 1) as f64 - 1.0;
                for channel in 0..3 {
                    let gain = (-0.11 * x_normalized - 0.04 * y_normalized).exp();
                    right[[y, x, channel]] = (left[[y, x, channel]] as f64 / gain)
                        .round()
                        .clamp(0.0, 16383.0) as u16;
                }
            }
        }

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(
            correction.model != "gain_spatial_xy_rgb"
                && correction.model != "gain_offset_spatial_xy_rgb"
                && correction.model != "gain_spatial_quadratic_xy_rgb"
                && correction.model != "gain_offset_spatial_quadratic_xy_rgb",
            "{correction:?}"
        );
        assert!(!correction.spatial_2d_validation.gain_accepted);
        assert!(!correction.spatial_2d_validation.gain_offset_accepted);
        assert!(!correction.spatial_2d_validation.quadratic.gain_accepted);
        assert!(
            !correction
                .spatial_2d_validation
                .quadratic
                .gain_offset_accepted
        );
        assert!(
            correction
                .spatial_2d_validation
                .gain_rejection_reason
                .contains("held-out")
                || correction
                    .spatial_2d_validation
                    .gain_rejection_reason
                    .contains("disagree"),
            "{correction:?}"
        );
    }

    #[test]
    fn seam_exposure_rejects_quadratic_field_confined_to_training_windows() {
        let (left, mut right) = textured_seam_strips(768, 192);
        let (height, width, _) = right.dim();
        for y in 0..height {
            let row = y * 8 / height;
            let y_normalized = 2.0 * y as f64 / (height - 1) as f64 - 1.0;
            for x in 0..width {
                let col = x * 6 / width;
                if !(row + col).is_multiple_of(2) {
                    continue;
                }
                let x_normalized = 2.0 * x as f64 / (width - 1) as f64 - 1.0;
                let gain = (-0.10 * x_normalized * x_normalized
                    + 0.035 * x_normalized * y_normalized
                    - 0.065 * y_normalized * y_normalized)
                    .exp();
                for channel in 0..3 {
                    right[[y, x, channel]] = (left[[y, x, channel]] as f64 / gain)
                        .round()
                        .clamp(0.0, 16383.0) as u16;
                }
            }
        }

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());
        let quadratic = &correction.spatial_2d_validation.quadratic;

        assert_ne!(
            correction.model, "gain_spatial_quadratic_xy_rgb",
            "{correction:?}"
        );
        assert_ne!(
            correction.model, "gain_offset_spatial_quadratic_xy_rgb",
            "{correction:?}"
        );
        assert!(!quadratic.gain_accepted, "{correction:?}");
        assert!(!quadratic.gain_offset_accepted, "{correction:?}");
        assert!(
            quadratic.gain_rejection_reason.contains("held-out")
                || quadratic.gain_rejection_reason.contains("disagree"),
            "{correction:?}"
        );
    }

    #[test]
    fn seam_exposure_rejects_vertical_gain_offset_field_confined_to_training_windows() {
        let (left, mut right) = textured_seam_strips(768, 128);
        let center_gain = [0.98, 1.02, 1.00];
        let gain_slopes = [-0.12, -0.09, -0.07];
        let center_offset = [-500.0, -450.0, -550.0];
        let offset_slopes = [160.0, 120.0, 90.0];
        let (height, width, _) = right.dim();
        for y in 0..height {
            let row = y * 8 / height;
            let y_normalized = 2.0 * y as f64 / (height - 1) as f64 - 1.0;
            for x in 0..width {
                let col = x * 4 / width;
                if !(row + col).is_multiple_of(2) {
                    continue;
                }
                for channel in 0..3 {
                    let gain = center_gain[channel] * (gain_slopes[channel] * y_normalized).exp();
                    let offset = center_offset[channel] + offset_slopes[channel] * y_normalized;
                    right[[y, x, channel]] = ((left[[y, x, channel]] as f64 - offset) / gain)
                        .round()
                        .clamp(0.0, 16383.0) as u16;
                }
            }
        }

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert_ne!(
            correction.model, "gain_offset_spatial_y_rgb",
            "{correction:?}"
        );
        assert_ne!(
            correction.model, "gain_offset_spatial_xy_rgb",
            "{correction:?}"
        );
        assert_ne!(
            correction.model, "gain_offset_spatial_quadratic_xy_rgb",
            "{correction:?}"
        );
        assert!(
            correction
                .spatial_gain_offset_rejection_reason
                .contains("held-out")
                || correction
                    .spatial_gain_offset_rejection_reason
                    .contains("disagree"),
            "{correction:?}"
        );
    }

    #[test]
    fn seam_exposure_models_horizontal_shading_as_a_2d_field() {
        let (left, mut right) = textured_seam_strips(384, 128);
        let (height, width, _) = right.dim();
        for y in 0..height {
            for x in 0..width {
                let x_normalized = 2.0 * x as f64 / (width - 1) as f64 - 1.0;
                let shade = (0.14 * x_normalized).exp();
                for channel in 0..3 {
                    right[[y, x, channel]] = (right[[y, x, channel]] as f64 * shade)
                        .round()
                        .clamp(0.0, 16383.0) as u16;
                }
            }
        }

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert_eq!(correction.model, "gain_spatial_xy_rgb", "{correction:?}");
        assert!(
            correction.spatial_gain_log_slope_x_rgb[0] < -0.10
                && correction.spatial_gain_log_slope_y_rgb[0].abs() < 0.02,
            "{correction:?}"
        );
    }

    #[test]
    fn photometric_field_clamps_outside_measured_horizontal_support() {
        let image = Array3::<u16>::from_elem((4, 128, 3), 1000);
        let field = SpatialPhotometricField {
            center_gain: [1.0; 3],
            center_offset: [0.0; 3],
            log_gain_slope_x: [0.10; 3],
            log_gain_slope_y: [0.0; 3],
            log_gain_quadratic_xx: [0.0; 3],
            log_gain_quadratic_xy: [0.0; 3],
            log_gain_quadratic_yy: [0.0; 3],
            offset_slope_x: [0.0; 3],
            offset_slope_y: [0.0; 3],
            offset_quadratic_xx: [0.0; 3],
            offset_quadratic_xy: [0.0; 3],
            offset_quadratic_yy: [0.0; 3],
        };

        let corrected = apply_photometric_correction(&image, &field, 0, 64, 0, 4, 16383.0);

        assert!(corrected[[0, 0, 0]] < corrected[[0, 63, 0]]);
        assert_eq!(corrected[[0, 63, 0]], corrected[[0, 127, 0]]);
    }

    #[test]
    fn seam_exposure_rejects_additive_model_that_creates_clipping() {
        let (mut left, right) = textured_seam_strips(192, 128);
        offset_strip(&mut left, [800.0; 3]);
        let (h, w, _) = left.dim();
        let mut right = right;
        for y in 0..h {
            for x in 0..w {
                if (x + y) % 11 == 0 {
                    for c in 0..3 {
                        right[[y, x, c]] = 16300;
                    }
                }
            }
        }

        let correction = seam_exposure_correction(&left, &right, &StitchConfig::default());

        assert!(
            correction.model != "gain_offset_rgb"
                && correction.model != "gain_offset_spatial_y_rgb"
                && correction.model != "gain_offset_spatial_xy_rgb"
                && correction.model != "gain_offset_spatial_quadratic_xy_rgb",
            "{correction:?}"
        );
        assert!(
            correction
                .gain_offset_rejection_reason
                .contains("increase clipping"),
            "{correction:?}"
        );
    }

    #[test]
    fn minimum_cost_seam_tracks_low_cost_corridor() {
        let mut costs = Array2::<f64>::from_elem((48, 19), 0.65);
        for y in 0..48 {
            let corridor = 4 + (y / 12).min(3);
            costs[[y, corridor]] = 0.01;
            if corridor + 1 < 19 {
                costs[[y, corridor + 1]] = 0.03;
            }
        }

        let seam = minimum_cost_vertical_seam(&costs);

        assert_eq!(seam.len(), 48);
        assert!(seam.iter().enumerate().all(|(y, &x)| {
            let corridor = 4 + (y / 12).min(3);
            x.abs_diff(corridor) <= 1
        }));
        assert!(seam.windows(2).all(|pair| pair[0].abs_diff(pair[1]) <= 2));
    }

    #[test]
    fn multiscale_blur_preserves_constant_fields() {
        let plane = Array2::<f64>::from_elem((17, 23), 0.37);
        let image = Array3::<f64>::from_elem((17, 23, 3), 8123.5);
        let blurred_plane = box_blur_array2(&plane, 4);
        let blurred_image = box_blur_array3(&image, 4);

        assert!(blurred_plane
            .iter()
            .all(|value| (*value - 0.37).abs() < 1e-12));
        assert!(blurred_image
            .iter()
            .all(|value| (*value - 8123.5).abs() < 1e-9));
    }

    #[test]
    fn seam_aware_multiband_blend_is_identity_for_matched_overlap() {
        let (left, right) = textured_seam_strips(96, 72);
        let (blended, diagnostics) =
            seam_aware_multiband_blend(&left, &right, &StitchConfig::default());

        assert!(diagnostics.applied);
        assert_eq!(diagnostics.mode, "seam_aware_multiband");
        assert!(diagnostics.pyramid_levels >= 1);
        assert_eq!(diagnostics.overlap_mean_abs_difference, 0.0);
        assert_eq!(diagnostics.overlap_p95_abs_difference, 0.0);
        assert_eq!(blended, left);
        assert!(diagnostics.output_to_source_seam_gradient_ratio <= 1.0 + 1e-9);
        assert!(diagnostics.detail_consistency.evaluated);
        assert!(!diagnostics.detail_consistency.review_required);
        assert!(!diagnostics.review_required, "{diagnostics:?}");
    }

    #[test]
    fn seam_detail_consistency_flags_repeated_cross_scan_blur() {
        let (left, _) = textured_seam_strips(240, 180);
        let right = blur_u16_image(&left, 4);

        let diagnostics = seam_detail_consistency(&left, &right, 16383.0);

        assert!(diagnostics.evaluated, "{diagnostics:?}");
        assert!(diagnostics.review_required, "{diagnostics:?}");
        assert!(diagnostics.imbalanced_scale_count >= 2, "{diagnostics:?}");
        assert!(
            diagnostics.maximum_symmetric_energy_ratio >= 2.0,
            "{diagnostics:?}"
        );
        assert!(
            diagnostics
                .scales
                .iter()
                .filter(|scale| scale.imbalanced)
                .all(|scale| {
                    scale.training_window_count >= 4
                        && scale.held_out_window_count >= 4
                        && scale.cross_split_direction_agrees
                }),
            "{diagnostics:?}"
        );
    }

    #[test]
    fn seam_detail_consistency_rejects_checkerboard_only_blur_evidence() {
        let (left, _) = textured_seam_strips(240, 180);
        let globally_blurred = blur_u16_image(&left, 4);
        let mut right = left.clone();
        let (height, width, _) = right.dim();
        for row in 0..SEAM_DETAIL_GRID_ROWS {
            let y_start = row * height / SEAM_DETAIL_GRID_ROWS;
            let y_end = (row + 1) * height / SEAM_DETAIL_GRID_ROWS;
            for column in 0..SEAM_DETAIL_GRID_COLUMNS {
                if (row + column) % 2 != 0 {
                    continue;
                }
                let x_start = column * width / SEAM_DETAIL_GRID_COLUMNS;
                let x_end = (column + 1) * width / SEAM_DETAIL_GRID_COLUMNS;
                right
                    .slice_mut(s![y_start..y_end, x_start..x_end, ..])
                    .assign(&globally_blurred.slice(s![y_start..y_end, x_start..x_end, ..]));
            }
        }

        let diagnostics = seam_detail_consistency(&left, &right, 16383.0);

        assert!(diagnostics.evaluated, "{diagnostics:?}");
        assert!(!diagnostics.review_required, "{diagnostics:?}");
        assert!(
            diagnostics.scales.iter().all(|scale| !scale.imbalanced),
            "{diagnostics:?}"
        );
    }

    #[test]
    fn seam_detail_consistency_requires_repeated_signal_support() {
        let left = Array3::<u16>::from_elem((180, 120, 3), 8_000);
        let right = left.clone();

        let diagnostics = seam_detail_consistency(&left, &right, 16383.0);

        assert!(!diagnostics.evaluated, "{diagnostics:?}");
        assert!(!diagnostics.decision_supported, "{diagnostics:?}");
        assert_eq!(diagnostics.supported_scale_count, 0);
        assert!(diagnostics.review_required, "{diagnostics:?}");
        assert!(diagnostics.review_reason.contains("at least 2"));
    }

    #[test]
    fn tiny_overlap_reason_mentions_edge_only_false_positive() {
        let reason = overlap_plausibility_reason(77, 0.0129, 0.0129)
            .expect("tiny overlap should be implausible");
        assert!(reason.contains("edge-only false positive"));
        assert!(reason.contains("77px"));
    }

    #[test]
    fn overlap_support_score_penalizes_tiny_overlaps() {
        assert!(overlap_support_score(0.013) < 0.25);
        assert!(overlap_support_score(0.12) > 0.9);
    }

    #[test]
    fn vertical_offset_plausibility_penalizes_search_limit() {
        assert!(vertical_offset_plausibility_score(2, 15) > 0.95);
        assert!(vertical_offset_plausibility_score(15, 15) < 0.20);
    }

    #[test]
    fn prior_weight_penalizes_vertical_limit_even_for_large_overlap() {
        let expanded_prior =
            translation_prior_weight(1.0, vertical_offset_plausibility_score(15, 15));
        let compact_prior = translation_prior_weight(
            0.4440053690809217,
            vertical_offset_plausibility_score(2, 15),
        );

        assert!(
            compact_prior > expanded_prior,
            "a candidate on the vertical search limit should not retain the strongest prior just because its overlap is larger"
        );
    }

    #[test]
    fn search_score_prefers_stronger_correspondence_over_saturated_overlap_prior() {
        let expanded_candidate = translation_search_score(
            translation_search_correspondence_score(0.21954560297343648, 0.5261482372999441),
            1.0,
            vertical_offset_plausibility_score(-15, 15),
        );
        let smaller_candidate = translation_search_score(
            translation_search_correspondence_score(0.46234207388873, 0.48565055075364444),
            0.4440053690809217,
            vertical_offset_plausibility_score(2, 15),
        );

        assert!(
            smaller_candidate > expanded_candidate,
            "smaller candidate should win on search score when its correspondence evidence is much stronger"
        );
    }

    #[test]
    fn objective_score_prefers_stronger_correspondence_over_larger_overlap_prior() {
        let expanded_candidate =
            sample_validation_metrics(0.21954560297343648, 0.5261482372999441, 1.0, 0.15, 0.092);
        let smaller_candidate = sample_validation_metrics(
            0.46234207388873,
            0.48565055075364444,
            0.4440053690809217,
            1.0,
            0.199,
        );

        assert!(
            expanded_candidate.plausibility_score > smaller_candidate.plausibility_score,
            "the larger overlap should still retain the stronger plausibility prior"
        );
        assert!(
            objective_score(&smaller_candidate) > objective_score(&expanded_candidate),
            "stronger correspondence evidence should outweigh a saturated overlap prior"
        );
    }
}

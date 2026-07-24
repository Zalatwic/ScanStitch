use crate::constants::{BRADFORD_LMS_TO_XYZ, BRADFORD_XYZ_TO_LMS, D50_WHITE};
use crate::density::{self, MeasuredNegativeResponseCalibration};
use crate::scanner_linearization::{self, ScannerLinearizationCalibration};
use crate::tiff_io::DngMetadata;
use nalgebra::{DMatrix, Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};

pub const CALIBRATION_PROFILE_SCHEMA_VERSION: u32 = 2;
pub const CALIBRATION_LIBRARY_SCHEMA_VERSION: u32 = 2;
pub const MIN_SUPPORTED_CALIBRATION_SCHEMA_VERSION: u32 = 1;

pub const MIN_CALIBRATION_CONFIDENCE: f64 = 0.50;
const MIN_AUTO_MATCH_CONFIDENCE: f64 = 0.90;
pub const MAX_MATRIX_CONDITION_NUMBER: f64 = 1_000.0;
pub const MIN_TARGET_PATCHES: usize = 4;
pub const MIN_TARGET_TRAINING_PATCHES: usize = 12;
pub const MIN_TARGET_HELD_OUT_PATCHES: usize = 12;
const MAX_REPORTED_CANDIDATES: usize = 5;
pub const ROOT_POLYNOMIAL_BASIS: &str = "homogeneous_root_polynomial_rgb";
pub const ROOT_POLYNOMIAL_OUTPUT_SPACE: &str = "reference_xyz_d50";
pub const RESIDUAL_LUT_3D_MODEL_TYPE: &str = "smooth_residual_lut_3d";
pub const RESIDUAL_LUT_3D_INTERPOLATION: &str = "tetrahedral";
const ROOT_POLYNOMIAL_MAX_CONDITION_NUMBER: f64 = 100_000_000.0;
const ROOT_POLYNOMIAL_MAX_COEFFICIENT_ABS: f64 = 50.0;
const ROOT_POLYNOMIAL_MAX_DELTA_E00_RMS: f64 = 6.0;
const ROOT_POLYNOMIAL_MAX_DELTA_E00_MAX: f64 = 15.0;
const ROOT_POLYNOMIAL_MIN_DELTA_E00_RMS_IMPROVEMENT: f64 = 0.25;
const ROOT_POLYNOMIAL_MIN_DELTA_E00_RMS_IMPROVEMENT_FRACTION: f64 = 0.05;
const ROOT_POLYNOMIAL_MAX_DELTA_E00_MAX_REGRESSION: f64 = 1.0;
const ROOT_POLYNOMIAL_MAX_XYZ_RMS_REGRESSION_FRACTION: f64 = 0.02;
const ROOT_POLYNOMIAL_MAX_HUE_RMS_REGRESSION: f64 = 0.50;
const ROOT_POLYNOMIAL_HIGHER_DEGREE_MIN_ABSOLUTE_GAIN: f64 = 0.15;
const ROOT_POLYNOMIAL_HIGHER_DEGREE_MIN_RELATIVE_GAIN: f64 = 0.03;
const RESIDUAL_LUT_3D_MAX_CONDITION_NUMBER: f64 = 10_000_000_000.0;
const RESIDUAL_LUT_3D_MAX_NODE_ABS: f64 = 0.25;
const RESIDUAL_LUT_3D_MAX_NODE_RMS: f64 = 0.10;
const RESIDUAL_LUT_3D_MAX_ROUGHNESS_RMS: f64 = 0.10;
const RESIDUAL_LUT_3D_MIN_OCCUPIED_CELL_FRACTION: f64 = 0.35;
const RESIDUAL_LUT_3D_MIN_HELD_OUT_INSIDE_FRACTION: f64 = 0.98;
const RESIDUAL_LUT_3D_MAX_DELTA_E00_RMS: f64 = 4.0;
const RESIDUAL_LUT_3D_MAX_DELTA_E00_MAX: f64 = 10.0;
const RESIDUAL_LUT_3D_MIN_DELTA_E00_RMS_IMPROVEMENT: f64 = 0.20;
const RESIDUAL_LUT_3D_MIN_DELTA_E00_RMS_IMPROVEMENT_FRACTION: f64 = 0.05;
const RESIDUAL_LUT_3D_MAX_DELTA_E00_MAX_REGRESSION: f64 = 0.50;
const RESIDUAL_LUT_3D_MAX_XYZ_RMS_REGRESSION_FRACTION: f64 = 0.01;
const RESIDUAL_LUT_3D_MAX_HUE_RMS_REGRESSION: f64 = 0.35;
const RESIDUAL_LUT_3D_HIGHER_GRID_MIN_ABSOLUTE_GAIN: f64 = 0.15;
const RESIDUAL_LUT_3D_HIGHER_GRID_MIN_RELATIVE_GAIN: f64 = 0.03;
const RESIDUAL_LUT_3D_HULL_EXPANSION: f64 = 1.10;

#[derive(Debug, Clone)]
pub struct CalibrationLoadResult {
    pub profile: Option<CalibrationProfile>,
    pub diagnostics: CalibrationDiagnostics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationApplicationMode {
    DirectProfile,
    ScannerConstrainedImageAdaptation,
}

#[derive(Debug, Clone)]
pub struct CalibrationProfile {
    pub schema_version: u32,
    pub profile_id: Option<String>,
    pub source_space: serde_json::Value,
    pub scanner: serde_json::Value,
    pub film: serde_json::Value,
    pub target: serde_json::Value,
    pub reference: serde_json::Value,
    pub gamut_limits: Option<GamutLimits>,
    pub whitepoint: [f64; 3],
    pub work_to_xyz: [[f64; 3]; 3],
    pub confidence: f64,
    pub matrix_condition_number: f64,
    pub fit: Option<TargetFitDiagnostics>,
    pub target_patches: Vec<TargetPatch>,
    pub application_mode: CalibrationApplicationMode,
    pub scanner_profile_id: Option<String>,
    pub roll_profile_id: Option<String>,
    pub roll_correction_applied: bool,
    pub scanner_prior_work_to_xyz: Option<[[f64; 3]; 3]>,
    pub scanner_prior_confidence: Option<f64>,
    pub scanner_prior_fit: Option<TargetFitDiagnostics>,
    pub negative_response: Option<MeasuredNegativeResponseCalibration>,
    pub scanner_linearization: Option<ScannerLinearizationCalibration>,
    pub color_model: Option<RootPolynomialColorModel>,
    pub lut_3d_model: Option<ResidualLut3dColorModel>,
    /// Optional XYZ-domain roll correction applied after `color_model`.
    pub color_model_post_xyz: Option<[[f64; 3]; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TargetFitDiagnostics {
    pub method: String,
    pub patch_count: usize,
    pub target_residual_rms: f64,
    pub target_residual_max: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation: Option<TargetFitValidationDiagnostics>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_hue_residuals: Vec<TargetHueResidual>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub worst_patches: Vec<TargetPatchFitResidual>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TargetFitValidationDiagnostics {
    pub evaluation_set: String,
    pub training_patch_count: usize,
    pub held_out_patch_count: usize,
    pub training_residual_rms: f64,
    pub training_residual_max: f64,
    pub identity_baseline_residual_rms: f64,
    pub identity_baseline_residual_max: f64,
    pub held_out_identity_improvement_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TargetHueResidual {
    pub hue_family: String,
    pub patch_count: usize,
    pub residual_rms: f64,
    pub residual_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TargetPatchFitResidual {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch_id: Option<String>,
    pub hue_family: String,
    pub source_rgb: [f64; 3],
    pub reference_xyz: [f64; 3],
    pub fitted_xyz: [f64; 3],
    pub residual_xyz: [f64; 3],
    pub residual_error: f64,
    pub max_channel_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TargetPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch_id: Option<String>,
    /// Optional normalized scanner-frame coordinate `[x, y]` in `[-1, 1]`.
    /// Required when a spatial scanner shading model precedes the colour matrix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scanner_xy: Option<[f64; 2]>,
    pub source_rgb: [f64; 3],
    pub reference_xyz: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFitResult {
    pub matrix: [[f64; 3]; 3],
    pub whitepoint: [f64; 3],
    pub confidence: f64,
    pub matrix_condition_number: f64,
    pub fit: TargetFitDiagnostics,
}

/// A homogeneous root-polynomial scanner-RGB to reference-XYZ transform.
///
/// Every basis term has effective degree one, so exposure scaling remains
/// linear while cross-channel nonlinearities can model scanner/spectral
/// interactions. Only models carrying independently recomputable held-out
/// evidence are accepted by the profile loader.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RootPolynomialColorModel {
    pub model_id: String,
    pub basis: String,
    pub degree: u8,
    pub output_space: String,
    /// One XYZ coefficient triplet per basis term, in documented basis order.
    pub coefficients: Vec<[f64; 3]>,
    /// Exact training measurements retained so fitting, regularization, model
    /// complexity, conditioning, and scene-support claims can be recomputed.
    pub training_patches: Vec<TargetPatch>,
    /// Convex hull of measured training RGB chromaticities, stored as `[r/sum, g/sum]`.
    pub training_chromaticity_hull: Vec<[f64; 2]>,
    pub validation: RootPolynomialValidation,
    pub fit: TargetFitDiagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RootPolynomialValidation {
    pub training_patch_count: usize,
    pub held_out_patch_count: usize,
    pub cross_validation_folds: usize,
    pub regularization_lambda: f64,
    pub design_condition_number: f64,
    pub coefficient_max_abs: f64,
    pub training_delta_e00_rms: f64,
    pub training_delta_e00_max: f64,
    pub held_out_delta_e00_rms: f64,
    pub held_out_delta_e00_max: f64,
    pub matrix_held_out_delta_e00_rms: f64,
    pub matrix_held_out_delta_e00_max: f64,
    pub held_out_delta_e00_rms_improvement: f64,
    pub held_out_delta_e00_rms_improvement_fraction: f64,
    pub held_out_xyz_rms_improvement_fraction: f64,
    pub maximum_hue_family_delta_e00_rms_regression: f64,
    pub selected_over_matrix: bool,
}

/// Audit record for the scanner transform used to place roll-target samples
/// into D50 XYZ before fitting an XYZ-domain roll correction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScannerColorModelApplicationDiagnostics {
    pub transform: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    pub output_domain: String,
    pub numerical_zero_tolerance: f64,
    pub numerical_zero_clamp_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RootPolynomialCandidateDiagnostics {
    pub degree: u8,
    pub term_count: usize,
    pub status: String,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation: Option<RootPolynomialValidation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RootPolynomialSelectionDiagnostics {
    pub status: String,
    pub selection_rule: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_degree: Option<u8>,
    pub candidates: Vec<RootPolynomialCandidateDiagnostics>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootPolynomialFitSelection {
    pub selected_model: Option<RootPolynomialColorModel>,
    pub diagnostics: RootPolynomialSelectionDiagnostics,
}

/// A smooth 3D residual LUT applied after a simpler scanner matrix or
/// root-polynomial baseline. Boundary nodes are identically zero so the model
/// returns continuously to the baseline at its measured RGB-domain limits.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ResidualLut3dColorModel {
    pub model_id: String,
    pub model_type: String,
    pub interpolation: String,
    pub grid_size: u8,
    pub output_space: String,
    pub baseline_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_model_id: Option<String>,
    pub input_min: [f64; 3],
    pub input_max: [f64; 3],
    /// Residual XYZ nodes in R-major/G-middle/B-minor order.
    pub residual_nodes_xyz: Vec<[f64; 3]>,
    pub training_patches: Vec<TargetPatch>,
    pub training_chromaticity_hull: Vec<[f64; 2]>,
    pub validation: ResidualLut3dValidation,
    pub fit: TargetFitDiagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ResidualLut3dValidation {
    pub training_patch_count: usize,
    pub held_out_patch_count: usize,
    pub cross_validation_folds: usize,
    pub regularization_lambda: f64,
    pub regularized_condition_number: f64,
    pub occupied_cell_fraction: f64,
    pub held_out_inside_domain_fraction: f64,
    pub residual_node_max_abs: f64,
    pub residual_node_rms: f64,
    pub residual_edge_roughness_rms: f64,
    pub training_delta_e00_rms: f64,
    pub training_delta_e00_max: f64,
    pub held_out_delta_e00_rms: f64,
    pub held_out_delta_e00_max: f64,
    pub baseline_held_out_delta_e00_rms: f64,
    pub baseline_held_out_delta_e00_max: f64,
    pub held_out_delta_e00_rms_improvement: f64,
    pub held_out_delta_e00_rms_improvement_fraction: f64,
    pub held_out_xyz_rms_improvement_fraction: f64,
    pub maximum_hue_family_delta_e00_rms_regression: f64,
    pub selected_over_baseline: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ResidualLut3dCandidateDiagnostics {
    pub grid_size: u8,
    pub node_count: usize,
    pub fitted_interior_node_count: usize,
    pub status: String,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation: Option<ResidualLut3dValidation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ResidualLut3dSelectionDiagnostics {
    pub status: String,
    pub baseline_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_model_id: Option<String>,
    pub selection_rule: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_grid_size: Option<u8>,
    pub candidates: Vec<ResidualLut3dCandidateDiagnostics>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidualLut3dFitSelection {
    pub selected_model: Option<ResidualLut3dColorModel>,
    pub diagnostics: ResidualLut3dSelectionDiagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationDiagnostics {
    pub status: String,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_mapping_application: Option<CalibrationColorMappingApplicationDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_profile: Option<ExternalProfileDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library: Option<CalibrationLibraryDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scanner_profile: Option<ScannerProfileDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roll_profile: Option<RollProfileDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_film_stock: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub film_stock: Option<FilmStockDiagnostics>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nearest_roll_candidates: Vec<RollCandidateDiagnostics>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nearest_film_candidates: Vec<FilmHintCandidateDiagnostics>,
    pub profile_schema_version: Option<u32>,
    pub confidence: Option<f64>,
    pub reason: String,
    pub matrix_condition_number: Option<f64>,
    pub whitepoint: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_upgrade_available: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_upgrade_reason: Option<String>,
    #[serde(default)]
    pub rejection_details: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationColorMappingApplicationDiagnostics {
    pub evaluated: bool,
    pub applied: bool,
    pub selection_status: String,
    pub selected_candidate: String,
    pub preferred_candidate: Option<String>,
    pub reason: String,
    pub definition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExternalProfileDiagnostics {
    pub path: Option<String>,
    pub profile_id: Option<String>,
    pub source_space: Option<serde_json::Value>,
    pub scanner: Option<serde_json::Value>,
    pub film: Option<serde_json::Value>,
    pub target: Option<serde_json::Value>,
    pub reference: Option<serde_json::Value>,
    pub gamut_limits: Option<GamutLimits>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit: Option<TargetFitDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_patch_signal_domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_patch_linearization_model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_sampling: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_model: Option<RootPolynomialColorModel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lut_3d_model: Option<ResidualLut3dColorModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationLibraryDiagnostics {
    pub path: String,
    pub scanner_profiles_loaded: usize,
    pub roll_profiles_loaded: usize,
    pub film_hints_loaded: usize,
    #[serde(default)]
    pub invalid_entries: Vec<LibraryEntryRejection>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LibraryEntryRejection {
    pub path: String,
    pub record_type: Option<String>,
    pub profile_id: Option<String>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScannerProfileDiagnostics {
    pub status: String,
    pub selection: String,
    pub path: Option<String>,
    pub schema_version: Option<u32>,
    pub profile_id: Option<String>,
    pub scanner: Option<serde_json::Value>,
    pub settings: Option<serde_json::Value>,
    pub confidence: Option<f64>,
    pub matrix_condition_number: Option<f64>,
    pub whitepoint: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_curves: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scanner_linearization: Option<ScannerLinearizationCalibration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flare_black_white_diagnostics: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polynomial_fit: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lut_3d: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lut_3d_fit: Option<ResidualLut3dSelectionDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_e00_summary: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit: Option<TargetFitDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_patch_signal_domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_patch_linearization_model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_sampling: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_model: Option<RootPolynomialColorModel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lut_3d_model: Option<ResidualLut3dColorModel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scanner_settings_fingerprint: Option<String>,
    pub reason: String,
    #[serde(default)]
    pub rejection_details: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RollProfileDiagnostics {
    pub status: String,
    pub selection: String,
    pub path: Option<String>,
    pub schema_version: Option<u32>,
    pub profile_id: Option<String>,
    pub scanner_profile_id: Option<String>,
    pub film: Option<serde_json::Value>,
    pub development: Option<serde_json::Value>,
    pub metadata: Option<serde_json::Value>,
    pub base_color: Option<[f64; 3]>,
    pub observed_base_color: Option<[f64; 3]>,
    pub base_color_delta: Option<f64>,
    pub confidence: Option<f64>,
    pub correction_domain: Option<String>,
    pub correction_matrix_condition_number: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_transform_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hue_family_residuals: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_e00_summary: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit: Option<TargetFitDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_patch_signal_domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_patch_linearization_model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_sampling: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scanner_color_model_application: Option<ScannerColorModelApplicationDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scanner_settings_fingerprint: Option<String>,
    pub correction_applied: bool,
    pub lut_present: bool,
    pub reason: String,
    #[serde(default)]
    pub rejection_details: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RollCandidateDiagnostics {
    pub profile_id: Option<String>,
    pub path: Option<String>,
    pub scanner_profile_id: Option<String>,
    pub film: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub film_stock_match: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub film_match_reason: Option<String>,
    pub confidence: f64,
    pub base_color: Option<[f64; 3]>,
    pub base_color_delta: Option<f64>,
    pub correction_available: bool,
    pub advisory_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FilmStockDiagnostics {
    pub status: String,
    pub selection: String,
    pub requested_stock: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_hint: Option<FilmHintCandidateDiagnostics>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub matched_roll_profiles: Vec<String>,
    pub reason: String,
    #[serde(default)]
    pub rejection_details: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FilmHintCandidateDiagnostics {
    pub film_id: Option<String>,
    pub path: Option<String>,
    pub schema_version: Option<u32>,
    pub stock: Option<String>,
    pub aliases: Vec<String>,
    pub similarity_tags: Vec<String>,
    pub metadata: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub film_stock_match: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub film_match_reason: Option<String>,
    pub advisory_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GamutLimits {
    #[serde(default)]
    pub min: Option<[f64; 3]>,
    #[serde(default)]
    pub max: Option<[f64; 3]>,
}

#[derive(Debug, Deserialize)]
struct RawCalibrationProfile {
    #[serde(default)]
    schema_version: Option<u32>,
    #[serde(default)]
    profile_id: Option<String>,
    #[serde(default)]
    source_space: Option<serde_json::Value>,
    #[serde(default)]
    scanner: Option<serde_json::Value>,
    #[serde(default)]
    film: Option<serde_json::Value>,
    #[serde(default)]
    target: Option<serde_json::Value>,
    #[serde(default)]
    reference: Option<serde_json::Value>,
    #[serde(default)]
    gamut_limits: Option<GamutLimits>,
    #[serde(default)]
    whitepoint: Option<[f64; 3]>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    work_to_xyz: Option<[[f64; 3]; 3]>,
    #[serde(default)]
    fit_matrix: Option<[[f64; 3]; 3]>,
    #[serde(default)]
    fit: Option<TargetFitDiagnostics>,
    #[serde(default)]
    patches: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    target_patch_signal_domain: Option<String>,
    #[serde(default)]
    target_patch_linearization_model_id: Option<String>,
    #[serde(default)]
    target_sampling: Option<serde_json::Value>,
    #[serde(default)]
    negative_response: Option<MeasuredNegativeResponseCalibration>,
    #[serde(default)]
    scanner_linearization: Option<ScannerLinearizationCalibration>,
    #[serde(default)]
    color_model: Option<RootPolynomialColorModel>,
    #[serde(default)]
    lut_3d_model: Option<ResidualLut3dColorModel>,
}

#[derive(Debug, Deserialize)]
struct RawScannerProfile {
    #[serde(default)]
    schema_version: Option<u32>,
    #[serde(default)]
    record_type: Option<String>,
    #[serde(default)]
    profile_id: Option<String>,
    #[serde(default)]
    source_space: Option<serde_json::Value>,
    #[serde(default)]
    scanner: Option<serde_json::Value>,
    #[serde(default)]
    settings: Option<serde_json::Value>,
    #[serde(default)]
    response_curves: Option<serde_json::Value>,
    #[serde(default)]
    scanner_linearization: Option<ScannerLinearizationCalibration>,
    #[serde(default)]
    color_model: Option<RootPolynomialColorModel>,
    #[serde(default)]
    lut_3d_model: Option<ResidualLut3dColorModel>,
    #[serde(default)]
    flare_black_white_diagnostics: Option<serde_json::Value>,
    #[serde(default)]
    target: Option<serde_json::Value>,
    #[serde(default)]
    reference: Option<serde_json::Value>,
    #[serde(default)]
    whitepoint: Option<[f64; 3]>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    scanner_rgb_to_xyz: Option<[[f64; 3]; 3]>,
    #[serde(default)]
    work_to_xyz: Option<[[f64; 3]; 3]>,
    #[serde(default)]
    fit_matrix: Option<[[f64; 3]; 3]>,
    #[serde(default)]
    fit: Option<TargetFitDiagnostics>,
    #[serde(default)]
    polynomial_fit: Option<serde_json::Value>,
    #[serde(default)]
    lut_3d: Option<serde_json::Value>,
    #[serde(default)]
    lut_3d_fit: Option<ResidualLut3dSelectionDiagnostics>,
    #[serde(default)]
    delta_e00_summary: Option<serde_json::Value>,
    #[serde(default)]
    patches: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    target_patch_signal_domain: Option<String>,
    #[serde(default)]
    target_patch_linearization_model_id: Option<String>,
    #[serde(default)]
    target_sampling: Option<serde_json::Value>,
    #[serde(default)]
    scanner_settings_fingerprint: Option<String>,
    #[serde(default)]
    auto_match: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RawRollProfile {
    #[serde(default)]
    schema_version: Option<u32>,
    #[serde(default)]
    record_type: Option<String>,
    #[serde(default)]
    profile_id: Option<String>,
    #[serde(default)]
    scanner_profile_id: Option<String>,
    #[serde(default)]
    film: Option<serde_json::Value>,
    #[serde(default)]
    development: Option<serde_json::Value>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    #[serde(default)]
    target: Option<serde_json::Value>,
    #[serde(default)]
    reference: Option<serde_json::Value>,
    #[serde(default)]
    base_color: Option<[f64; 3]>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    correction_matrix: Option<[[f64; 3]; 3]>,
    #[serde(default)]
    correction_domain: Option<String>,
    #[serde(default)]
    fit: Option<TargetFitDiagnostics>,
    #[serde(default)]
    hue_family_residuals: Option<serde_json::Value>,
    #[serde(default)]
    delta_e00_summary: Option<serde_json::Value>,
    #[serde(default)]
    patches: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    target_patch_signal_domain: Option<String>,
    #[serde(default)]
    target_patch_linearization_model_id: Option<String>,
    #[serde(default)]
    target_sampling: Option<serde_json::Value>,
    #[serde(default)]
    scanner_color_model_application: Option<ScannerColorModelApplicationDiagnostics>,
    #[serde(default)]
    scanner_settings_fingerprint: Option<String>,
    #[serde(default)]
    lut: Option<serde_json::Value>,
    #[serde(default)]
    negative_response: Option<MeasuredNegativeResponseCalibration>,
}

#[derive(Debug, Deserialize)]
struct RawFilmHint {
    #[serde(default)]
    schema_version: Option<u32>,
    #[serde(default)]
    record_type: Option<String>,
    #[serde(default)]
    film_id: Option<String>,
    #[serde(default)]
    profile_id: Option<String>,
    #[serde(default)]
    stock: Option<String>,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    similarity_tags: Vec<String>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
struct ScannerProfile {
    path: Option<PathBuf>,
    schema_version: u32,
    profile_id: Option<String>,
    source_space: serde_json::Value,
    scanner: serde_json::Value,
    settings: Option<serde_json::Value>,
    response_curves: Option<serde_json::Value>,
    scanner_linearization: Option<ScannerLinearizationCalibration>,
    color_model: Option<RootPolynomialColorModel>,
    lut_3d_model: Option<ResidualLut3dColorModel>,
    flare_black_white_diagnostics: Option<serde_json::Value>,
    target: serde_json::Value,
    reference: serde_json::Value,
    whitepoint: [f64; 3],
    scanner_rgb_to_xyz: [[f64; 3]; 3],
    confidence: f64,
    matrix_condition_number: f64,
    fit: Option<TargetFitDiagnostics>,
    polynomial_fit: Option<serde_json::Value>,
    lut_3d: Option<serde_json::Value>,
    lut_3d_fit: Option<ResidualLut3dSelectionDiagnostics>,
    delta_e00_summary: Option<serde_json::Value>,
    target_patches: Vec<TargetPatch>,
    target_patch_signal_domain: Option<String>,
    target_patch_linearization_model_id: Option<String>,
    target_sampling: Option<serde_json::Value>,
    scanner_settings_fingerprint: Option<String>,
    auto_match: bool,
}

#[derive(Debug, Clone)]
struct RollProfile {
    path: Option<PathBuf>,
    schema_version: u32,
    profile_id: Option<String>,
    scanner_profile_id: Option<String>,
    film: serde_json::Value,
    development: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
    target: Option<serde_json::Value>,
    reference: Option<serde_json::Value>,
    base_color: Option<[f64; 3]>,
    confidence: f64,
    correction_matrix: Option<[[f64; 3]; 3]>,
    correction_domain: Option<String>,
    correction_matrix_condition_number: Option<f64>,
    fit: Option<TargetFitDiagnostics>,
    hue_family_residuals: Option<serde_json::Value>,
    delta_e00_summary: Option<serde_json::Value>,
    target_patches: Vec<TargetPatch>,
    target_patch_signal_domain: Option<String>,
    target_patch_linearization_model_id: Option<String>,
    target_sampling: Option<serde_json::Value>,
    scanner_color_model_application: Option<ScannerColorModelApplicationDiagnostics>,
    scanner_settings_fingerprint: Option<String>,
    lut: Option<serde_json::Value>,
    negative_response: Option<MeasuredNegativeResponseCalibration>,
}

#[derive(Debug, Clone)]
struct FilmHint {
    path: Option<PathBuf>,
    schema_version: u32,
    film_id: Option<String>,
    stock: Option<String>,
    aliases: Vec<String>,
    similarity_tags: Vec<String>,
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Default)]
struct LoadedLibrary {
    scanners: Vec<ScannerProfile>,
    rolls: Vec<RollProfile>,
    film_hints: Vec<FilmHint>,
    invalid_entries: Vec<LibraryEntryRejection>,
}

enum ValidatedLibraryRecord {
    Scanner(Box<ScannerProfile>),
    Roll(Box<RollProfile>),
    FilmHint(Box<FilmHint>),
}

pub fn not_configured() -> CalibrationLoadResult {
    CalibrationLoadResult {
        profile: None,
        diagnostics: CalibrationDiagnostics {
            status: "not_configured".to_string(),
            source: "image_derived_neutral_and_dominant_anchors".to_string(),
            color_mapping_application: None,
            external_profile: None,
            library: None,
            scanner_profile: None,
            roll_profile: None,
            requested_film_stock: None,
            film_stock: None,
            nearest_roll_candidates: Vec::new(),
            nearest_film_candidates: Vec::new(),
            profile_schema_version: None,
            confidence: None,
            reason: "no external calibration profile provided".to_string(),
            matrix_condition_number: None,
            whitepoint: None,
            calibration_upgrade_available: None,
            calibration_upgrade_reason: None,
            rejection_details: Vec::new(),
        },
    }
}

fn dng_standard_illuminant_xyz(code: u64) -> Option<([f64; 3], &'static str)> {
    // EXIF LightSource values used by DNG CalibrationIlluminant tags. Values are
    // normalized to Y=1 using the standard CIE chromaticities.
    match code {
        3 | 17 => Some(([1.098_50, 1.0, 0.355_85], "CalibrationIlluminant1 A")),
        18 => Some(([0.990_72, 1.0, 0.852_23], "CalibrationIlluminant1 B")),
        19 => Some(([0.980_74, 1.0, 1.182_32], "CalibrationIlluminant1 C")),
        20 => Some(([0.956_82, 1.0, 0.921_49], "CalibrationIlluminant1 D55")),
        21 => Some(([0.950_47, 1.0, 1.088_83], "CalibrationIlluminant1 D65")),
        22 => Some(([0.949_72, 1.0, 1.226_38], "CalibrationIlluminant1 D75")),
        23 => Some((D50_WHITE, "CalibrationIlluminant1 D50")),
        _ => None,
    }
}

fn dng_source_white(
    metadata: &DngMetadata,
    camera_to_xyz: &Matrix3<f64>,
    rejection_details: &mut Vec<String>,
) -> Option<([f64; 3], &'static str)> {
    if let Some(neutral) = metadata.as_shot_neutral.as_deref() {
        if neutral.len() != 3
            || neutral
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            rejection_details.push(
                "DNG AsShotNeutral must contain exactly three finite positive camera coordinates"
                    .to_string(),
            );
            return None;
        }
        let xyz = camera_to_xyz * Vector3::new(neutral[0], neutral[1], neutral[2]);
        if xyz.iter().any(|value| !value.is_finite() || *value <= 0.0) || xyz[1] <= 1e-12 {
            rejection_details.push(
                "DNG ColorMatrix1 inverse mapped AsShotNeutral to an invalid CIE XYZ white"
                    .to_string(),
            );
            return None;
        }
        return Some(([xyz[0] / xyz[1], 1.0, xyz[2] / xyz[1]], "AsShotNeutral"));
    }

    if let Some(code) = metadata.calibration_illuminant1.filter(|code| *code != 0) {
        return match dng_standard_illuminant_xyz(code) {
            Some(illuminant) => Some(illuminant),
            None => {
                rejection_details.push(format!(
                    "DNG CalibrationIlluminant1 value {code} is not a supported standard illuminant; refusing to guess its whitepoint"
                ));
                None
            }
        };
    }

    // Some scanner-produced DNG files omit both AsShotNeutral and
    // CalibrationIlluminant1 even though ColorMatrix1 is normalized so equal
    // reference-camera coordinates describe its calibration white. That is not
    // guaranteed by the DNG specification, so accept it only as an explicitly
    // inferred, low-confidence advisory prior and only when it maps to a
    // plausible white chromaticity.
    let xyz = camera_to_xyz * Vector3::new(1.0, 1.0, 1.0);
    let sum = xyz.iter().sum::<f64>();
    if xyz.iter().any(|value| !value.is_finite() || *value <= 0.0)
        || !sum.is_finite()
        || sum <= 1e-12
        || xyz[1] <= 1e-12
    {
        rejection_details.push(
            "DNG ColorMatrix1 with missing white metadata did not map equal camera coordinates to a finite positive CIE XYZ white"
                .to_string(),
        );
        return None;
    }
    let chromaticity_x = xyz[0] / sum;
    let chromaticity_y = xyz[1] / sum;
    if !(0.25..=0.45).contains(&chromaticity_x) || !(0.25..=0.45).contains(&chromaticity_y) {
        rejection_details.push(format!(
            "DNG ColorMatrix1 with missing white metadata implied implausible equal-camera chromaticity x={chromaticity_x:.4}, y={chromaticity_y:.4}"
        ));
        return None;
    }
    Some((
        [xyz[0] / xyz[1], 1.0, xyz[2] / xyz[1]],
        "inferred equal-camera neutral",
    ))
}

fn dng_bradford_to_d50(
    source_white: &[f64; 3],
    rejection_details: &mut Vec<String>,
) -> Option<Matrix3<f64>> {
    let xyz_to_lms = rows_to_matrix3(&BRADFORD_XYZ_TO_LMS);
    let lms_to_xyz = rows_to_matrix3(&BRADFORD_LMS_TO_XYZ);
    let source_lms = xyz_to_lms * Vector3::new(source_white[0], source_white[1], source_white[2]);
    let d50_lms = xyz_to_lms * Vector3::new(D50_WHITE[0], D50_WHITE[1], D50_WHITE[2]);
    if source_lms
        .iter()
        .chain(d50_lms.iter())
        .any(|value| !value.is_finite() || *value <= 1e-12)
    {
        rejection_details.push(
            "DNG source white cannot form a finite positive Bradford chromatic adaptation"
                .to_string(),
        );
        return None;
    }
    let scale = Matrix3::from_diagonal(&Vector3::new(
        d50_lms[0] / source_lms[0],
        d50_lms[1] / source_lms[1],
        d50_lms[2] / source_lms[2],
    ));
    Some(lms_to_xyz * scale * xyz_to_lms)
}

pub fn dng_color_matrix1_advisory_prior(metadata: &DngMetadata) -> Option<CalibrationLoadResult> {
    let stored_xyz_to_camera = metadata.color_matrix1?;
    let mut rejection_details = Vec::new();
    validate_matrix_values(
        "DNG ColorMatrix1",
        &stored_xyz_to_camera,
        &mut rejection_details,
    );
    let xyz_to_camera = rows_to_matrix3(&stored_xyz_to_camera);
    let stored_matrix_condition_number = validate_matrix_condition(
        "DNG ColorMatrix1 XYZ-to-camera",
        &xyz_to_camera,
        &mut rejection_details,
    );
    let camera_to_xyz = xyz_to_camera.try_inverse();
    if camera_to_xyz.is_none()
        && !rejection_details
            .iter()
            .any(|detail| detail.contains("singular"))
    {
        rejection_details
            .push("DNG ColorMatrix1 XYZ-to-camera matrix could not be inverted".to_string());
    }
    let (source_white, source_white_basis) = camera_to_xyz
        .as_ref()
        .and_then(|matrix| dng_source_white(metadata, matrix, &mut rejection_details))
        .unwrap_or((D50_WHITE, "unavailable"));
    let work_to_xyz_matrix = camera_to_xyz.as_ref().and_then(|matrix| {
        dng_bradford_to_d50(&source_white, &mut rejection_details).map(|cat| cat * matrix)
    });
    let work_to_xyz = work_to_xyz_matrix
        .as_ref()
        .map(matrix3_to_rows)
        .unwrap_or([[0.0; 3]; 3]);
    if work_to_xyz_matrix.is_some() {
        validate_matrix_values(
            "derived DNG camera-to-XYZ D50",
            &work_to_xyz,
            &mut rejection_details,
        );
    }
    let matrix_condition_number = work_to_xyz_matrix
        .as_ref()
        .map(|matrix| {
            validate_matrix_condition(
                "derived DNG camera-to-XYZ D50",
                matrix,
                &mut rejection_details,
            )
        })
        .unwrap_or(stored_matrix_condition_number);
    let confidence = match source_white_basis {
        "AsShotNeutral" => 0.64,
        "inferred equal-camera neutral" => 0.50,
        _ => 0.58,
    };
    let profile_id = "dng-color-matrix1-advisory-prior".to_string();
    let scanner_json = serde_json::json!({
        "source": "dng_metadata",
        "make": metadata.make,
        "model": metadata.model,
        "unique_camera_model": metadata.unique_camera_model,
        "software": metadata.software,
        "calibration_illuminant1": metadata.calibration_illuminant1,
        "calibration_illuminant2": metadata.calibration_illuminant2,
    });
    let scanner_profile = ScannerProfileDiagnostics {
        status: if rejection_details.is_empty() {
            "applied".to_string()
        } else {
            "rejected".to_string()
        },
        selection: "dng_metadata_advisory".to_string(),
        path: None,
        schema_version: None,
        profile_id: Some(profile_id.clone()),
        scanner: Some(scanner_json.clone()),
        settings: Some(serde_json::json!({
            "source": "dng_metadata",
            "software": metadata.software,
        })),
        confidence: Some(confidence),
        matrix_condition_number: matrix_condition_number
            .is_finite()
            .then_some(matrix_condition_number),
        whitepoint: Some(D50_WHITE),
        response_curves: None,
        scanner_linearization: None,
        flare_black_white_diagnostics: None,
        transform_type: Some("matrix".to_string()),
        polynomial_fit: None,
        lut_3d: None,
        lut_3d_fit: None,
        delta_e00_summary: None,
        fit: None,
        target_patch_signal_domain: None,
        target_patch_linearization_model_id: None,
        target_sampling: None,
        color_model: None,
        lut_3d_model: None,
        scanner_settings_fingerprint: None,
        reason: if rejection_details.is_empty() {
            format!(
                "DNG ColorMatrix1 was inverted from XYZ-to-camera, adapted from {source_white_basis} to D50 with linear Bradford, and selected as a weak scanner prior"
            )
        } else {
            "DNG ColorMatrix1 was present but failed advisory-prior validation".to_string()
        },
        rejection_details: rejection_details.clone(),
    };

    if !rejection_details.is_empty() {
        return Some(CalibrationLoadResult {
            profile: None,
            diagnostics: CalibrationDiagnostics {
                status: "rejected".to_string(),
                source: "dng_color_matrix1_advisory_prior".to_string(),
                color_mapping_application: None,
                external_profile: None,
                library: None,
                scanner_profile: Some(scanner_profile),
                roll_profile: None,
                requested_film_stock: None,
                film_stock: None,
                nearest_roll_candidates: Vec::new(),
                nearest_film_candidates: Vec::new(),
                profile_schema_version: None,
                confidence: None,
                reason: "DNG ColorMatrix1 advisory prior was rejected; using image-derived mapping"
                    .to_string(),
                matrix_condition_number: matrix_condition_number
                    .is_finite()
                    .then_some(matrix_condition_number),
                whitepoint: Some(D50_WHITE),
                calibration_upgrade_available: None,
                calibration_upgrade_reason: None,
                rejection_details,
            },
        });
    }

    let profile = CalibrationProfile {
        schema_version: CALIBRATION_PROFILE_SCHEMA_VERSION,
        profile_id: Some(profile_id.clone()),
        source_space: serde_json::json!({
            "type": "dng_linear_raw_scanner_rgb",
            "matrix_tag": "ColorMatrix1",
            "stored_matrix_direction": "xyz_to_reference_camera",
            "derived_matrix_direction": "camera_to_xyz_d50",
            "source_white": source_white,
            "source_white_basis": source_white_basis,
            "chromatic_adaptation": "linear_bradford_to_d50",
            "application": "scanner_constrained_image_adaptation",
            "advisory": true,
        }),
        scanner: scanner_json,
        film: serde_json::json!({
            "source": "unknown_roll",
            "advisory": true,
        }),
        target: serde_json::json!({
            "source": "dng_metadata",
            "matrix_tag": "ColorMatrix1",
        }),
        reference: serde_json::json!({
            "source": "dng_metadata",
            "whitepoint": "D50",
            "stored_matrix_condition_number": stored_matrix_condition_number,
            "note": "ColorMatrix1 follows the DNG XYZ-to-camera convention; its inverse and D50 chromatic adaptation are used only as a scanner prior, while scene anchors still adapt the roll.",
        }),
        gamut_limits: None,
        whitepoint: D50_WHITE,
        work_to_xyz,
        confidence,
        matrix_condition_number,
        fit: None,
        target_patches: Vec::new(),
        application_mode: CalibrationApplicationMode::ScannerConstrainedImageAdaptation,
        scanner_profile_id: Some(profile_id),
        roll_profile_id: None,
        roll_correction_applied: false,
        scanner_prior_work_to_xyz: None,
        scanner_prior_confidence: None,
        scanner_prior_fit: None,
        negative_response: None,
        scanner_linearization: None,
        color_model: None,
        lut_3d_model: None,
        color_model_post_xyz: None,
    };

    Some(CalibrationLoadResult {
        diagnostics: CalibrationDiagnostics {
            status: "applied".to_string(),
            source: "dng_color_matrix1_advisory_prior".to_string(),
            color_mapping_application: None,
            external_profile: None,
            library: None,
            scanner_profile: Some(scanner_profile),
            roll_profile: None,
            requested_film_stock: None,
            film_stock: None,
            nearest_roll_candidates: Vec::new(),
            nearest_film_candidates: Vec::new(),
            profile_schema_version: Some(profile.schema_version),
            confidence: Some(profile.confidence),
            reason: format!(
                "DNG ColorMatrix1 was interpreted according to the DNG XYZ-to-camera convention, inverted, and Bradford-adapted from {source_white_basis} to D50 as a weak scanner prior; automatic candidate scoring still compares it against image-derived mapping"
            ),
            matrix_condition_number: Some(profile.matrix_condition_number),
            whitepoint: Some(profile.whitepoint),
            calibration_upgrade_available: None,
            calibration_upgrade_reason: None,
            rejection_details: Vec::new(),
        },
        profile: Some(profile),
    })
}

pub fn load_profile(path: &Path) -> CalibrationLoadResult {
    match std::fs::read_to_string(path) {
        Ok(contents) => parse_profile_json(&contents, Some(path)),
        Err(err) => rejected(
            Some(path),
            None,
            format!("failed to read calibration profile: {err}"),
            vec![err.to_string()],
        ),
    }
}

pub fn load_calibration(
    compatibility_profile_path: Option<&Path>,
    library_dir: Option<&Path>,
    scanner_profile_id: Option<&str>,
    roll_profile_id: Option<&str>,
    requested_film_stock: Option<&str>,
    observed_base_color: Option<[f64; 3]>,
) -> CalibrationLoadResult {
    if library_dir.is_some()
        || scanner_profile_id.is_some()
        || roll_profile_id.is_some()
        || requested_film_stock.is_some()
    {
        let library_result = load_library_selection(
            library_dir,
            scanner_profile_id,
            roll_profile_id,
            requested_film_stock,
            observed_base_color,
        );
        if library_result.profile.is_none()
            && library_result.diagnostics.status != "rejected"
            && scanner_profile_id.is_none()
            && roll_profile_id.is_none()
        {
            if let Some(path) = compatibility_profile_path {
                let mut compatibility_result = load_profile(path);
                compatibility_result.diagnostics.library = library_result.diagnostics.library;
                compatibility_result.diagnostics.scanner_profile =
                    library_result.diagnostics.scanner_profile;
                compatibility_result.diagnostics.roll_profile =
                    library_result.diagnostics.roll_profile;
                compatibility_result.diagnostics.requested_film_stock =
                    library_result.diagnostics.requested_film_stock;
                compatibility_result.diagnostics.film_stock = library_result.diagnostics.film_stock;
                compatibility_result.diagnostics.nearest_roll_candidates =
                    library_result.diagnostics.nearest_roll_candidates;
                compatibility_result.diagnostics.nearest_film_candidates =
                    library_result.diagnostics.nearest_film_candidates;
                return compatibility_result;
            }
        }
        return library_result;
    }

    compatibility_profile_path
        .map(load_profile)
        .unwrap_or_else(not_configured)
}

pub fn parse_profile_json(contents: &str, path: Option<&Path>) -> CalibrationLoadResult {
    let raw: RawCalibrationProfile = match serde_json::from_str(contents) {
        Ok(raw) => raw,
        Err(err) => {
            return rejected(
                path,
                None,
                format!("failed to parse calibration profile JSON: {err}"),
                vec![err.to_string()],
            )
        }
    };

    validate_raw_profile(raw, path)
}

pub fn lab_d50_to_xyz(lab: [f64; 3]) -> [f64; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    [
        D50_WHITE[0] * lab_inverse_f(fx),
        D50_WHITE[1] * lab_inverse_f(fy),
        D50_WHITE[2] * lab_inverse_f(fz),
    ]
}

/// Decode the explicit, disjoint patch sets required to fit a production
/// scanner or roll color matrix. A single `patches` list is intentionally not
/// accepted here because it cannot support an independent residual claim.
pub fn disjoint_target_patch_sets_from_measurement(
    value: &serde_json::Value,
) -> Result<(Vec<TargetPatch>, Vec<TargetPatch>), String> {
    let training = value.get("training_patches");
    let held_out = value.get("held_out_patches");
    match (training, held_out) {
        (Some(training), Some(held_out)) => Ok((
            target_patches_from_named_array(training, "training_patches")?,
            target_patches_from_named_array(held_out, "held_out_patches")?,
        )),
        (Some(_), None) => {
            Err("target matrix fitting requires held_out_patches[] in addition to training_patches[]"
                .to_string())
        }
        (None, Some(_)) => {
            Err("target matrix fitting requires training_patches[] in addition to held_out_patches[]"
                .to_string())
        }
        (None, None) if value.get("patches").is_some() => Err(
            "target matrix fitting no longer accepts a single patches[] set because in-sample residuals are not independent; provide disjoint training_patches[] and held_out_patches[] with unique sample IDs"
                .to_string(),
        ),
        (None, None) => Err(
            "target matrix fitting requires disjoint training_patches[] and held_out_patches[]"
                .to_string(),
        ),
    }
}

fn target_patches_from_named_array(
    value: &serde_json::Value,
    field: &str,
) -> Result<Vec<TargetPatch>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{field} must be a JSON array"))?
        .iter()
        .enumerate()
        .map(|(idx, patch)| target_patch_from_named_object(field, idx, patch))
        .collect()
}

pub fn target_patches_from_measurement(
    value: &serde_json::Value,
) -> Result<Vec<TargetPatch>, String> {
    if let Some(patches) = value.get("patches") {
        let patches = patches
            .as_array()
            .ok_or_else(|| "patches must be a JSON array".to_string())?;
        return patches
            .iter()
            .enumerate()
            .map(|(idx, patch)| target_patch_from_object(idx, patch))
            .collect();
    }

    let Some(source_values) = value
        .get("patch_rgb")
        .or_else(|| value.get("measured_rgb"))
        .or_else(|| value.get("source_rgb"))
    else {
        return Err(
            "target measurements require patches[] or patch_rgb/measured_rgb arrays".to_string(),
        );
    };
    let source_values = source_values
        .as_array()
        .ok_or_else(|| "patch RGB measurements must be a JSON array".to_string())?;

    let reference_xyz = value
        .get("reference_xyz")
        .or_else(|| value.get("target_xyz"))
        .or_else(|| value.get("xyz"));
    let reference_lab = value
        .get("reference_lab")
        .or_else(|| value.get("target_lab"))
        .or_else(|| value.get("lab"));

    let references = reference_xyz
        .or(reference_lab)
        .ok_or_else(|| "target measurements require reference_xyz or reference_lab".to_string())?
        .as_array()
        .ok_or_else(|| "reference patch values must be a JSON array".to_string())?;
    if source_values.len() != references.len() {
        return Err(format!(
            "patch RGB count {} does not match reference count {}",
            source_values.len(),
            references.len()
        ));
    }

    source_values
        .iter()
        .zip(references.iter())
        .enumerate()
        .map(|(idx, (source, reference))| {
            let source_rgb = triplet_from_value(source, &format!("patch_rgb[{idx}]"))?;
            let reference_xyz = if reference_xyz.is_some() {
                triplet_from_value(reference, &format!("reference_xyz[{idx}]"))?
            } else {
                lab_d50_to_xyz(triplet_from_value(
                    reference,
                    &format!("reference_lab[{idx}]"),
                )?)
            };
            Ok(TargetPatch {
                patch_id: None,
                scanner_xy: None,
                source_rgb,
                reference_xyz,
            })
        })
        .collect()
}

fn optional_target_patches_from_values(
    patches: Option<&[serde_json::Value]>,
    field: &str,
    rejection_details: &mut Vec<String>,
) -> Vec<TargetPatch> {
    let Some(patches) = patches else {
        return Vec::new();
    };
    if patches.len() < MIN_TARGET_PATCHES {
        rejection_details.push(format!(
            "{field} contains {} patch(es), below required minimum {MIN_TARGET_PATCHES}",
            patches.len()
        ));
        return Vec::new();
    }

    let mut out = Vec::with_capacity(patches.len());
    for (idx, patch) in patches.iter().enumerate() {
        match target_patch_from_object(idx, patch).and_then(|patch| {
            validate_target_patch(idx, &patch)?;
            Ok(patch)
        }) {
            Ok(patch) => out.push(patch),
            Err(err) => rejection_details.push(format!("{field}.{err}")),
        }
    }
    out
}

pub fn scanner_settings_fingerprint(settings: Option<&serde_json::Value>) -> Option<String> {
    settings.map(|settings| {
        let canonical = canonical_json(settings);
        format!("fnv1a64:{:016x}", fnv1a64(canonical.as_bytes()))
    })
}

fn lab_f(value: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if value > DELTA * DELTA * DELTA {
        value.cbrt()
    } else {
        value / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

fn xyz_d50_to_lab(xyz: [f64; 3]) -> [f64; 3] {
    let fx = lab_f(xyz[0] / D50_WHITE[0].max(1e-9));
    let fy = lab_f(xyz[1] / D50_WHITE[1].max(1e-9));
    let fz = lab_f(xyz[2] / D50_WHITE[2].max(1e-9));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn target_hue_family(reference_xyz: [f64; 3]) -> &'static str {
    let lab = xyz_d50_to_lab(reference_xyz);
    let chroma = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
    if chroma < 6.0 {
        return "neutral";
    }
    let mut hue = lab[2].atan2(lab[1]).to_degrees();
    if hue < 0.0 {
        hue += 360.0;
    }
    match hue {
        h if !(20.0..345.0).contains(&h) => "red",
        h if h < 55.0 => "orange",
        h if h < 90.0 => "yellow",
        h if h < 165.0 => "green",
        h if h < 225.0 => "cyan",
        h if h < 285.0 => "blue",
        _ => "magenta",
    }
}

fn vector3_to_array(value: Vector3<f64>) -> [f64; 3] {
    [value[0], value[1], value[2]]
}

pub fn root_polynomial_term_count(degree: u8) -> Option<usize> {
    match degree {
        2 => Some(6),
        3 => Some(13),
        _ => None,
    }
}

fn signed_nth_root(value: f64, degree: i32) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value.signum() * value.abs().powf(1.0 / f64::from(degree))
    }
}

/// Evaluate the documented homogeneous root-polynomial basis. The first six
/// terms are `R, G, B, sqrt(RG), sqrt(RB), sqrt(GB)`. Degree three appends
/// `cbrt(R^2G), cbrt(R^2B), cbrt(G^2R), cbrt(G^2B), cbrt(B^2R),
/// cbrt(B^2G), cbrt(RGB)`.
pub fn root_polynomial_basis_values(degree: u8, rgb: [f64; 3]) -> Result<Vec<f64>, String> {
    let expected = root_polynomial_term_count(degree)
        .ok_or_else(|| format!("unsupported root-polynomial degree {degree}; expected 2 or 3"))?;
    if rgb.iter().any(|value| !value.is_finite()) {
        return Err("root-polynomial input must contain finite RGB values".to_string());
    }
    let [r, g, b] = rgb;
    let mut basis = vec![
        r,
        g,
        b,
        signed_nth_root(r * g, 2),
        signed_nth_root(r * b, 2),
        signed_nth_root(g * b, 2),
    ];
    if degree == 3 {
        basis.extend([
            signed_nth_root(r * r * g, 3),
            signed_nth_root(r * r * b, 3),
            signed_nth_root(g * g * r, 3),
            signed_nth_root(g * g * b, 3),
            signed_nth_root(b * b * r, 3),
            signed_nth_root(b * b * g, 3),
            signed_nth_root(r * g * b, 3),
        ]);
    }
    debug_assert_eq!(basis.len(), expected);
    Ok(basis)
}

pub fn evaluate_root_polynomial_xyz(model: &RootPolynomialColorModel, rgb: [f64; 3]) -> [f64; 3] {
    let Ok(basis) = root_polynomial_basis_values(model.degree, rgb) else {
        return [f64::NAN; 3];
    };
    let mut xyz = [0.0f64; 3];
    for (term, coefficient) in basis.iter().zip(&model.coefficients) {
        for channel in 0..3 {
            xyz[channel] += term * coefficient[channel];
        }
    }
    xyz
}

impl CalibrationProfile {
    /// Evaluate the selected calibrated source-domain model into the profile's
    /// target XYZ domain. The matrix remains the exact fallback whenever no
    /// held-out-qualified nonlinear model is present.
    pub fn calibrated_source_to_xyz(&self, rgb: [f64; 3]) -> [f64; 3] {
        if let Some(model) = self.lut_3d_model.as_ref() {
            let source_matrix = self
                .scanner_prior_work_to_xyz
                .as_ref()
                .unwrap_or(&self.work_to_xyz);
            let root_baseline = (model.baseline_kind == "root_polynomial")
                .then_some(self.color_model.as_ref())
                .flatten();
            let xyz = evaluate_residual_lut_3d_xyz(model, source_matrix, root_baseline, rgb);
            if let Some(post_rows) = self.color_model_post_xyz.as_ref() {
                let post = rows_to_matrix3(post_rows);
                return vector3_to_array(post * Vector3::new(xyz[0], xyz[1], xyz[2]));
            }
            xyz
        } else if let Some(model) = self.color_model.as_ref() {
            let xyz = evaluate_root_polynomial_xyz(model, rgb);
            if let Some(post_rows) = self.color_model_post_xyz.as_ref() {
                let post = rows_to_matrix3(post_rows);
                return vector3_to_array(post * Vector3::new(xyz[0], xyz[1], xyz[2]));
            }
            xyz
        } else {
            let matrix = rows_to_matrix3(&self.work_to_xyz);
            vector3_to_array(matrix * Vector3::new(rgb[0], rgb[1], rgb[2]))
        }
    }
}

fn root_polynomial_chromaticity(rgb: [f64; 3]) -> Option<[f64; 2]> {
    if rgb.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return None;
    }
    let sum = rgb.iter().sum::<f64>();
    (sum > 1e-9).then_some([rgb[0] / sum, rgb[1] / sum])
}

fn hull_cross(origin: [f64; 2], left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])
}

fn root_polynomial_chromaticity_hull(patches: &[TargetPatch]) -> Vec<[f64; 2]> {
    let mut points = patches
        .iter()
        .filter_map(|patch| root_polynomial_chromaticity(patch.source_rgb))
        .collect::<Vec<_>>();
    points.sort_by(|left, right| {
        left[0]
            .total_cmp(&right[0])
            .then_with(|| left[1].total_cmp(&right[1]))
    });
    points.dedup_by(|left, right| left[0] == right[0] && left[1] == right[1]);
    if points.len() <= 2 {
        return points;
    }

    let mut lower = Vec::<[f64; 2]>::new();
    for point in &points {
        while lower.len() >= 2
            && hull_cross(lower[lower.len() - 2], lower[lower.len() - 1], *point) <= 1e-12
        {
            lower.pop();
        }
        lower.push(*point);
    }
    let mut upper = Vec::<[f64; 2]>::new();
    for point in points.iter().rev() {
        while upper.len() >= 2
            && hull_cross(upper[upper.len() - 2], upper[upper.len() - 1], *point) <= 1e-12
        {
            upper.pop();
        }
        upper.push(*point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn root_polynomial_minimum_counts(degree: u8) -> Option<(usize, usize)> {
    match degree {
        2 => Some((18, 18)),
        3 => Some((39, 24)),
        _ => None,
    }
}

fn root_polynomial_design_matrix(
    patches: &[TargetPatch],
    degree: u8,
) -> Result<DMatrix<f64>, String> {
    let term_count = root_polynomial_term_count(degree)
        .ok_or_else(|| format!("unsupported root-polynomial degree {degree}"))?;
    let mut values = Vec::with_capacity(patches.len() * term_count);
    for patch in patches {
        values.extend(root_polynomial_basis_values(degree, patch.source_rgb)?);
    }
    Ok(DMatrix::from_row_slice(patches.len(), term_count, &values))
}

fn target_xyz_matrix(patches: &[TargetPatch]) -> DMatrix<f64> {
    DMatrix::from_fn(patches.len(), 3, |row, channel| {
        patches[row].reference_xyz[channel]
    })
}

fn solve_root_polynomial_coefficients(
    patches: &[TargetPatch],
    degree: u8,
    regularization_lambda: f64,
    weights: Option<&[f64]>,
) -> Result<Vec<[f64; 3]>, String> {
    let mut design = root_polynomial_design_matrix(patches, degree)?;
    let mut target = target_xyz_matrix(patches);
    if let Some(weights) = weights {
        if weights.len() != patches.len() {
            return Err("root-polynomial fit weight count does not match patch count".to_string());
        }
        for (row, weight) in weights.iter().enumerate().take(patches.len()) {
            let scale = weight.clamp(0.0, 1.0).sqrt();
            design.row_mut(row).scale_mut(scale);
            target.row_mut(row).scale_mut(scale);
        }
    }
    let transpose = design.transpose();
    let mut normal = &transpose * &design;
    let trace_scale = (normal.trace() / normal.nrows().max(1) as f64).max(1e-12);
    normal +=
        DMatrix::identity(normal.nrows(), normal.ncols()) * (regularization_lambda * trace_scale);
    let right = transpose * target;
    let coefficients = normal.lu().solve(&right).ok_or_else(|| {
        "root-polynomial regularized least-squares solve was singular".to_string()
    })?;
    let out = (0..coefficients.nrows())
        .map(|term| {
            [
                coefficients[(term, 0)],
                coefficients[(term, 1)],
                coefficients[(term, 2)],
            ]
        })
        .collect::<Vec<_>>();
    if out.iter().flatten().any(|value| !value.is_finite()) {
        return Err("root-polynomial fit produced non-finite coefficients".to_string());
    }
    Ok(out)
}

fn root_polynomial_design_condition_number(
    patches: &[TargetPatch],
    degree: u8,
) -> Result<f64, String> {
    let design = root_polynomial_design_matrix(patches, degree)?;
    let singular = design.svd(false, false).singular_values;
    let maximum = singular.iter().copied().fold(0.0f64, f64::max);
    let minimum = singular.iter().copied().fold(f64::INFINITY, f64::min);
    if !minimum.is_finite() || minimum <= 1e-12 || maximum <= 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(maximum / minimum)
    }
}

fn evaluate_coefficients_xyz(coefficients: &[[f64; 3]], degree: u8, rgb: [f64; 3]) -> [f64; 3] {
    let Ok(basis) = root_polynomial_basis_values(degree, rgb) else {
        return [f64::NAN; 3];
    };
    let mut xyz = [0.0f64; 3];
    for (term, coefficient) in basis.iter().zip(coefficients) {
        for channel in 0..3 {
            xyz[channel] += term * coefficient[channel];
        }
    }
    xyz
}

#[derive(Debug, Clone)]
struct RootPolynomialEvaluation {
    fit: TargetFitDiagnostics,
    delta_e00_rms: f64,
    delta_e00_max: f64,
    per_hue_delta_e00_rms: std::collections::BTreeMap<String, f64>,
}

fn evaluate_target_mapping<F>(
    patches: &[TargetPatch],
    method: &str,
    mut map: F,
) -> RootPolynomialEvaluation
where
    F: FnMut([f64; 3]) -> [f64; 3],
{
    let fit = evaluate_target_transform(patches, method, &mut map);
    let mut delta_e00_sum_sq = 0.0f64;
    let mut delta_e00_max = 0.0f64;
    let mut hue = std::collections::BTreeMap::<String, (usize, f64)>::new();
    for patch in patches {
        let mapped = map(patch.source_rgb);
        let delta_e00 = crate::colorspace::lab_delta_e2000(
            crate::colorspace::xyz_d50_to_lab(mapped),
            crate::colorspace::xyz_d50_to_lab(patch.reference_xyz),
        );
        delta_e00_sum_sq += delta_e00 * delta_e00;
        delta_e00_max = delta_e00_max.max(delta_e00);
        let entry = hue
            .entry(target_hue_family(patch.reference_xyz).to_string())
            .or_default();
        entry.0 += 1;
        entry.1 += delta_e00 * delta_e00;
    }
    let count = patches.len().max(1) as f64;
    RootPolynomialEvaluation {
        fit,
        delta_e00_rms: (delta_e00_sum_sq / count).sqrt(),
        delta_e00_max,
        per_hue_delta_e00_rms: hue
            .into_iter()
            .map(|(family, (count, sum_sq))| (family, (sum_sq / count.max(1) as f64).sqrt()))
            .collect(),
    }
}

fn root_polynomial_cross_validation_lambda(
    patches: &[TargetPatch],
    degree: u8,
) -> Result<(f64, usize), String> {
    const LAMBDAS: [f64; 7] = [1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3];
    let term_count = root_polynomial_term_count(degree).expect("degree checked");
    let folds = if patches.len() >= term_count * 4 {
        5
    } else {
        3
    };
    let mut ordered = (0..patches.len()).collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        target_hue_family(patches[*left].reference_xyz)
            .cmp(target_hue_family(patches[*right].reference_xyz))
            .then_with(|| {
                patches[*left].reference_xyz[1].total_cmp(&patches[*right].reference_xyz[1])
            })
            .then_with(|| patches[*left].patch_id.cmp(&patches[*right].patch_id))
    });

    let mut best = None::<(f64, f64)>;
    for lambda in LAMBDAS {
        let mut sum_sq = 0.0f64;
        let mut count = 0usize;
        let mut valid = true;
        for fold in 0..folds {
            let mut fit_patches = Vec::new();
            let mut validation_patches = Vec::new();
            for (position, patch_index) in ordered.iter().enumerate() {
                if position % folds == fold {
                    validation_patches.push(patches[*patch_index].clone());
                } else {
                    fit_patches.push(patches[*patch_index].clone());
                }
            }
            if fit_patches.len() <= term_count || validation_patches.is_empty() {
                valid = false;
                break;
            }
            let coefficients =
                solve_root_polynomial_coefficients(&fit_patches, degree, lambda, None)?;
            for patch in &validation_patches {
                let predicted = evaluate_coefficients_xyz(&coefficients, degree, patch.source_rgb);
                for (channel, predicted_value) in predicted.iter().enumerate() {
                    let residual = *predicted_value - patch.reference_xyz[channel];
                    sum_sq += residual * residual;
                    count += 1;
                }
            }
        }
        if valid && count > 0 {
            let rms = (sum_sq / count as f64).sqrt();
            if best.as_ref().is_none_or(|(best_rms, best_lambda)| {
                rms < *best_rms - 1e-12
                    || ((rms - *best_rms).abs() <= 1e-12 && lambda > *best_lambda)
            }) {
                best = Some((rms, lambda));
            }
        }
    }
    best.map(|(_, lambda)| (lambda, folds)).ok_or_else(|| {
        "root-polynomial training-only cross-validation could not produce a fit".to_string()
    })
}

fn median_f64(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}

fn robust_root_polynomial_coefficients(
    patches: &[TargetPatch],
    degree: u8,
    lambda: f64,
) -> Result<Vec<[f64; 3]>, String> {
    let mut weights = vec![1.0f64; patches.len()];
    let mut coefficients = solve_root_polynomial_coefficients(patches, degree, lambda, None)?;
    for _ in 0..5 {
        let residuals = patches
            .iter()
            .map(|patch| {
                let predicted = evaluate_coefficients_xyz(&coefficients, degree, patch.source_rgb);
                (0..3)
                    .map(|channel| {
                        let residual = predicted[channel] - patch.reference_xyz[channel];
                        residual * residual
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .collect::<Vec<_>>();
        let mut residual_copy = residuals.clone();
        let median = median_f64(&mut residual_copy);
        let mut deviations = residuals
            .iter()
            .map(|residual| (residual - median).abs())
            .collect::<Vec<_>>();
        let sigma = (1.4826 * median_f64(&mut deviations)).max(1e-6);
        let huber = 1.5 * sigma;
        for (weight, residual) in weights.iter_mut().zip(residuals) {
            *weight = if residual <= huber {
                1.0
            } else {
                (huber / residual.max(1e-12)).clamp(0.05, 1.0)
            };
        }
        coefficients = solve_root_polynomial_coefficients(patches, degree, lambda, Some(&weights))?;
    }
    Ok(coefficients)
}

fn maximum_hue_rms_regression(
    candidate: &RootPolynomialEvaluation,
    baseline: &RootPolynomialEvaluation,
) -> f64 {
    candidate
        .per_hue_delta_e00_rms
        .iter()
        .filter_map(|(family, candidate_rms)| {
            baseline
                .per_hue_delta_e00_rms
                .get(family)
                .map(|baseline_rms| candidate_rms - baseline_rms)
        })
        .fold(0.0f64, f64::max)
}

fn root_polynomial_acceptance_reason(validation: &RootPolynomialValidation) -> (bool, String) {
    let xyz_regression_limit = -ROOT_POLYNOMIAL_MAX_XYZ_RMS_REGRESSION_FRACTION;
    let accepted = validation.design_condition_number.is_finite()
        && validation.design_condition_number <= ROOT_POLYNOMIAL_MAX_CONDITION_NUMBER
        && validation.coefficient_max_abs <= ROOT_POLYNOMIAL_MAX_COEFFICIENT_ABS
        && validation.held_out_delta_e00_rms <= ROOT_POLYNOMIAL_MAX_DELTA_E00_RMS
        && validation.held_out_delta_e00_max <= ROOT_POLYNOMIAL_MAX_DELTA_E00_MAX
        && validation.held_out_delta_e00_rms_improvement
            >= ROOT_POLYNOMIAL_MIN_DELTA_E00_RMS_IMPROVEMENT
        && validation.held_out_delta_e00_rms_improvement_fraction
            >= ROOT_POLYNOMIAL_MIN_DELTA_E00_RMS_IMPROVEMENT_FRACTION
        && validation.held_out_delta_e00_max
            <= validation.matrix_held_out_delta_e00_max
                + ROOT_POLYNOMIAL_MAX_DELTA_E00_MAX_REGRESSION
        && validation.held_out_xyz_rms_improvement_fraction >= xyz_regression_limit
        && validation.maximum_hue_family_delta_e00_rms_regression
            <= ROOT_POLYNOMIAL_MAX_HUE_RMS_REGRESSION;
    let reason = if accepted {
        format!(
            "held-out DeltaE00 RMS improved by {:.3} ({:.1}%) without exceeding XYZ, maximum-error, hue-family, conditioning, or coefficient limits",
            validation.held_out_delta_e00_rms_improvement,
            validation.held_out_delta_e00_rms_improvement_fraction * 100.0
        )
    } else {
        let mut reasons = Vec::new();
        if !validation.design_condition_number.is_finite()
            || validation.design_condition_number > ROOT_POLYNOMIAL_MAX_CONDITION_NUMBER
        {
            reasons.push(format!(
                "design condition {:.3} exceeds {:.3}",
                validation.design_condition_number, ROOT_POLYNOMIAL_MAX_CONDITION_NUMBER
            ));
        }
        if validation.coefficient_max_abs > ROOT_POLYNOMIAL_MAX_COEFFICIENT_ABS {
            reasons.push(format!(
                "coefficient magnitude {:.3} exceeds {:.3}",
                validation.coefficient_max_abs, ROOT_POLYNOMIAL_MAX_COEFFICIENT_ABS
            ));
        }
        if validation.held_out_delta_e00_rms > ROOT_POLYNOMIAL_MAX_DELTA_E00_RMS
            || validation.held_out_delta_e00_max > ROOT_POLYNOMIAL_MAX_DELTA_E00_MAX
        {
            reasons.push(format!(
                "held-out DeltaE00 RMS/max {:.3}/{:.3} exceeds {:.3}/{:.3}",
                validation.held_out_delta_e00_rms,
                validation.held_out_delta_e00_max,
                ROOT_POLYNOMIAL_MAX_DELTA_E00_RMS,
                ROOT_POLYNOMIAL_MAX_DELTA_E00_MAX
            ));
        }
        if validation.held_out_delta_e00_rms_improvement
            < ROOT_POLYNOMIAL_MIN_DELTA_E00_RMS_IMPROVEMENT
            || validation.held_out_delta_e00_rms_improvement_fraction
                < ROOT_POLYNOMIAL_MIN_DELTA_E00_RMS_IMPROVEMENT_FRACTION
        {
            reasons.push(format!(
                "held-out DeltaE00 RMS gain {:.3} ({:.1}%) is below {:.3} and {:.1}%",
                validation.held_out_delta_e00_rms_improvement,
                validation.held_out_delta_e00_rms_improvement_fraction * 100.0,
                ROOT_POLYNOMIAL_MIN_DELTA_E00_RMS_IMPROVEMENT,
                ROOT_POLYNOMIAL_MIN_DELTA_E00_RMS_IMPROVEMENT_FRACTION * 100.0
            ));
        }
        if validation.held_out_delta_e00_max
            > validation.matrix_held_out_delta_e00_max
                + ROOT_POLYNOMIAL_MAX_DELTA_E00_MAX_REGRESSION
        {
            reasons.push("held-out maximum DeltaE00 regressed beyond tolerance".to_string());
        }
        if validation.held_out_xyz_rms_improvement_fraction < xyz_regression_limit {
            reasons.push("held-out XYZ RMS regressed beyond tolerance".to_string());
        }
        if validation.maximum_hue_family_delta_e00_rms_regression
            > ROOT_POLYNOMIAL_MAX_HUE_RMS_REGRESSION
        {
            reasons.push("a held-out hue family regressed beyond tolerance".to_string());
        }
        reasons.join("; ")
    };
    (accepted, reason)
}

fn fit_root_polynomial_candidate(
    model_id: String,
    degree: u8,
    training_patches: &[TargetPatch],
    held_out_patches: &[TargetPatch],
    matrix: &Matrix3<f64>,
) -> Result<RootPolynomialColorModel, String> {
    let (lambda, folds) = root_polynomial_cross_validation_lambda(training_patches, degree)?;
    let coefficients = robust_root_polynomial_coefficients(training_patches, degree, lambda)?;
    let training = evaluate_target_mapping(training_patches, "root_polynomial_training", |rgb| {
        evaluate_coefficients_xyz(&coefficients, degree, rgb)
    });
    let mut held_out = evaluate_target_mapping(
        held_out_patches,
        &format!("root_polynomial_degree_{degree}_held_out"),
        |rgb| evaluate_coefficients_xyz(&coefficients, degree, rgb),
    );
    let matrix_held_out = evaluate_target_mapping(
        held_out_patches,
        "least_squares_matrix_held_out_baseline",
        |rgb| vector3_to_array(matrix * Vector3::new(rgb[0], rgb[1], rgb[2])),
    );
    let identity = evaluate_target_matrix(
        &Matrix3::identity(),
        held_out_patches,
        "identity_rgb_to_xyz_baseline",
    );
    let matrix_rms = matrix_held_out.fit.target_residual_rms;
    let delta_e_improvement = matrix_held_out.delta_e00_rms - held_out.delta_e00_rms;
    let delta_e_improvement_fraction = if matrix_held_out.delta_e00_rms > 1e-12 {
        delta_e_improvement / matrix_held_out.delta_e00_rms
    } else {
        0.0
    };
    let xyz_improvement_fraction = if matrix_rms > 1e-12 {
        (matrix_rms - held_out.fit.target_residual_rms) / matrix_rms
    } else {
        0.0
    };
    held_out.fit.validation = Some(TargetFitValidationDiagnostics {
        evaluation_set: "held_out".to_string(),
        training_patch_count: training_patches.len(),
        held_out_patch_count: held_out_patches.len(),
        training_residual_rms: training.fit.target_residual_rms,
        training_residual_max: training.fit.target_residual_max,
        identity_baseline_residual_rms: identity.target_residual_rms,
        identity_baseline_residual_max: identity.target_residual_max,
        held_out_identity_improvement_fraction: if identity.target_residual_rms > 1e-12 {
            (identity.target_residual_rms - held_out.fit.target_residual_rms)
                / identity.target_residual_rms
        } else {
            0.0
        },
    });
    let coefficient_max_abs = coefficients
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0f64, f64::max);
    let validation = RootPolynomialValidation {
        training_patch_count: training_patches.len(),
        held_out_patch_count: held_out_patches.len(),
        cross_validation_folds: folds,
        regularization_lambda: lambda,
        design_condition_number: root_polynomial_design_condition_number(training_patches, degree)?,
        coefficient_max_abs,
        training_delta_e00_rms: training.delta_e00_rms,
        training_delta_e00_max: training.delta_e00_max,
        held_out_delta_e00_rms: held_out.delta_e00_rms,
        held_out_delta_e00_max: held_out.delta_e00_max,
        matrix_held_out_delta_e00_rms: matrix_held_out.delta_e00_rms,
        matrix_held_out_delta_e00_max: matrix_held_out.delta_e00_max,
        held_out_delta_e00_rms_improvement: delta_e_improvement,
        held_out_delta_e00_rms_improvement_fraction: delta_e_improvement_fraction,
        held_out_xyz_rms_improvement_fraction: xyz_improvement_fraction,
        maximum_hue_family_delta_e00_rms_regression: maximum_hue_rms_regression(
            &held_out,
            &matrix_held_out,
        ),
        selected_over_matrix: false,
    };
    Ok(RootPolynomialColorModel {
        model_id,
        basis: ROOT_POLYNOMIAL_BASIS.to_string(),
        degree,
        output_space: ROOT_POLYNOMIAL_OUTPUT_SPACE.to_string(),
        coefficients,
        training_patches: training_patches.to_vec(),
        training_chromaticity_hull: root_polynomial_chromaticity_hull(training_patches),
        validation,
        fit: held_out.fit,
    })
}

/// Fit degree-two and, when sufficiently constrained, degree-three
/// root-polynomial candidates on training patches only. The independently held
/// out set decides whether either candidate is retained and whether added
/// complexity beats the simpler accepted model.
pub fn fit_root_polynomial_color_model_from_disjoint_patches(
    model_id_prefix: &str,
    training_patches: &[TargetPatch],
    held_out_patches: &[TargetPatch],
    matrix_rows: &[[f64; 3]; 3],
) -> Result<RootPolynomialFitSelection, String> {
    validate_disjoint_target_patch_sets(training_patches, held_out_patches)?;
    let matrix = rows_to_matrix3(matrix_rows);
    let mut diagnostics = Vec::<RootPolynomialCandidateDiagnostics>::new();
    let mut accepted = Vec::<(usize, RootPolynomialColorModel)>::new();
    for degree in [2u8, 3u8] {
        let term_count = root_polynomial_term_count(degree).expect("supported degree");
        let (minimum_training, minimum_held_out) =
            root_polynomial_minimum_counts(degree).expect("supported degree");
        if training_patches.len() < minimum_training || held_out_patches.len() < minimum_held_out {
            diagnostics.push(RootPolynomialCandidateDiagnostics {
                degree,
                term_count,
                status: "not_evaluated_insufficient_samples".to_string(),
                reason: format!(
                    "degree {degree} requires at least {minimum_training} training and {minimum_held_out} held-out patches; received {} and {}",
                    training_patches.len(),
                    held_out_patches.len()
                ),
                model_id: None,
                validation: None,
            });
            continue;
        }
        let model_id = format!("{model_id_prefix}-root-polynomial-d{degree}");
        match fit_root_polynomial_candidate(
            model_id.clone(),
            degree,
            training_patches,
            held_out_patches,
            &matrix,
        ) {
            Ok(mut model) => {
                let hull_valid = model.training_chromaticity_hull.len() >= 3;
                let (mut qualifies, mut reason) =
                    root_polynomial_acceptance_reason(&model.validation);
                if !hull_valid {
                    qualifies = false;
                    reason =
                        "training RGB measurements do not span a non-degenerate chromaticity hull"
                            .to_string();
                }
                model.validation.selected_over_matrix = qualifies;
                let index = diagnostics.len();
                diagnostics.push(RootPolynomialCandidateDiagnostics {
                    degree,
                    term_count,
                    status: if qualifies {
                        "accepted_candidate".to_string()
                    } else {
                        "rejected_held_out".to_string()
                    },
                    reason,
                    model_id: Some(model_id),
                    validation: Some(model.validation.clone()),
                });
                if qualifies {
                    accepted.push((index, model));
                }
            }
            Err(error) => diagnostics.push(RootPolynomialCandidateDiagnostics {
                degree,
                term_count,
                status: "rejected_fit".to_string(),
                reason: error,
                model_id: Some(model_id),
                validation: None,
            }),
        }
    }

    let mut selected = accepted.first().cloned();
    for candidate in accepted.iter().skip(1) {
        let Some((_, current)) = selected.as_ref() else {
            selected = Some(candidate.clone());
            continue;
        };
        let absolute_gain = current.validation.held_out_delta_e00_rms
            - candidate.1.validation.held_out_delta_e00_rms;
        let relative_gain = if current.validation.held_out_delta_e00_rms > 1e-12 {
            absolute_gain / current.validation.held_out_delta_e00_rms
        } else {
            0.0
        };
        let maximum_ok = candidate.1.validation.held_out_delta_e00_max
            <= current.validation.held_out_delta_e00_max + 0.50;
        if absolute_gain >= ROOT_POLYNOMIAL_HIGHER_DEGREE_MIN_ABSOLUTE_GAIN
            && relative_gain >= ROOT_POLYNOMIAL_HIGHER_DEGREE_MIN_RELATIVE_GAIN
            && maximum_ok
        {
            selected = Some(candidate.clone());
        }
    }
    let selected_index = selected.as_ref().map(|(index, _)| *index);
    for (index, candidate) in diagnostics.iter_mut().enumerate() {
        if candidate.status == "accepted_candidate" {
            if Some(index) == selected_index {
                candidate.status = "selected".to_string();
                candidate.reason = format!(
                    "{}; selected as the most complex model with a material held-out gain over every simpler accepted model",
                    candidate.reason
                );
            } else {
                candidate.status = "accepted_not_selected".to_string();
                candidate.reason = format!(
                    "{}; retained as evidence but not selected because added complexity did not materially beat the selected simpler model",
                    candidate.reason
                );
            }
        }
    }
    let selected_model = selected.map(|(_, model)| model);
    Ok(RootPolynomialFitSelection {
        diagnostics: RootPolynomialSelectionDiagnostics {
            status: if selected_model.is_some() {
                "selected_higher_order_model".to_string()
            } else {
                "matrix_retained".to_string()
            },
            selection_rule: "fit and regularization selection use training data only; held-out DeltaE00, XYZ residual, maximum error, hue-family regression, conditioning, and coefficient bounds gate each model; a higher degree must materially beat the simpler accepted model"
                .to_string(),
            selected_model_id: selected_model.as_ref().map(|model| model.model_id.clone()),
            selected_degree: selected_model.as_ref().map(|model| model.degree),
            candidates: diagnostics,
        },
        selected_model,
    })
}

fn residual_lut_3d_minimum_counts(grid_size: u8) -> Option<(usize, usize)> {
    match grid_size {
        5 => Some((108, 48)),
        7 => Some((500, 96)),
        _ => None,
    }
}

fn residual_lut_3d_node_count(grid_size: u8) -> usize {
    usize::from(grid_size).pow(3)
}

fn residual_lut_3d_interior_node_count(grid_size: u8) -> usize {
    usize::from(grid_size.saturating_sub(2)).pow(3)
}

fn residual_lut_3d_node_index(grid_size: usize, r: usize, g: usize, b: usize) -> usize {
    (r * grid_size + g) * grid_size + b
}

fn residual_lut_3d_interior_index(grid_size: usize, r: usize, g: usize, b: usize) -> Option<usize> {
    if r == 0 || g == 0 || b == 0 || r + 1 >= grid_size || g + 1 >= grid_size || b + 1 >= grid_size
    {
        return None;
    }
    let interior = grid_size - 2;
    Some(((r - 1) * interior + (g - 1)) * interior + (b - 1))
}

fn residual_lut_3d_domain(patches: &[TargetPatch]) -> Result<([f64; 3], [f64; 3]), String> {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for patch in patches {
        for channel in 0..3 {
            minimum[channel] = minimum[channel].min(patch.source_rgb[channel]);
            maximum[channel] = maximum[channel].max(patch.source_rgb[channel]);
        }
    }
    for channel in 0..3 {
        let span = maximum[channel] - minimum[channel];
        if !minimum[channel].is_finite()
            || !maximum[channel].is_finite()
            || minimum[channel] < 0.0
            || span < 0.25
        {
            return Err(format!(
                "residual 3D LUT training channel {channel} must span at least 0.25 finite non-negative source units; observed [{:.6}, {:.6}]",
                minimum[channel], maximum[channel]
            ));
        }
    }
    Ok((minimum, maximum))
}

fn residual_lut_3d_normalized_coordinates(
    input_min: [f64; 3],
    input_max: [f64; 3],
    rgb: [f64; 3],
) -> Option<[f64; 3]> {
    if rgb.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut normalized = [0.0; 3];
    for channel in 0..3 {
        let span = input_max[channel] - input_min[channel];
        if !span.is_finite()
            || span <= 1e-12
            || rgb[channel] < input_min[channel] - 1e-12
            || rgb[channel] > input_max[channel] + 1e-12
        {
            return None;
        }
        normalized[channel] = ((rgb[channel] - input_min[channel]) / span).clamp(0.0, 1.0);
    }
    Some(normalized)
}

fn residual_lut_3d_tetrahedral_weights(
    grid_size: usize,
    normalized: [f64; 3],
) -> [(usize, f64); 4] {
    let scale = (grid_size - 1) as f64;
    let scaled = normalized.map(|value| value * scale);
    let base = scaled.map(|value| (value.floor() as usize).min(grid_size - 2));
    let fraction: [f64; 3] = std::array::from_fn(|channel| scaled[channel] - base[channel] as f64);
    let [fr, fg, fb] = fraction;
    let vertices = if fr >= fg {
        if fg >= fb {
            [
                ([0, 0, 0], 1.0 - fr),
                ([1, 0, 0], fr - fg),
                ([1, 1, 0], fg - fb),
                ([1, 1, 1], fb),
            ]
        } else if fr >= fb {
            [
                ([0, 0, 0], 1.0 - fr),
                ([1, 0, 0], fr - fb),
                ([1, 0, 1], fb - fg),
                ([1, 1, 1], fg),
            ]
        } else {
            [
                ([0, 0, 0], 1.0 - fb),
                ([0, 0, 1], fb - fr),
                ([1, 0, 1], fr - fg),
                ([1, 1, 1], fg),
            ]
        }
    } else if fr >= fb {
        [
            ([0, 0, 0], 1.0 - fg),
            ([0, 1, 0], fg - fr),
            ([1, 1, 0], fr - fb),
            ([1, 1, 1], fb),
        ]
    } else if fg >= fb {
        [
            ([0, 0, 0], 1.0 - fg),
            ([0, 1, 0], fg - fb),
            ([0, 1, 1], fb - fr),
            ([1, 1, 1], fr),
        ]
    } else {
        [
            ([0, 0, 0], 1.0 - fb),
            ([0, 0, 1], fb - fg),
            ([0, 1, 1], fg - fr),
            ([1, 1, 1], fr),
        ]
    };
    vertices.map(|(offset, weight)| {
        (
            residual_lut_3d_node_index(
                grid_size,
                base[0] + offset[0],
                base[1] + offset[1],
                base[2] + offset[2],
            ),
            weight,
        )
    })
}

fn point_inside_scaled_chromaticity_hull(point: [f64; 2], hull: &[[f64; 2]], scale: f64) -> bool {
    if hull.len() < 3 {
        return false;
    }
    let centroid = [
        hull.iter().map(|vertex| vertex[0]).sum::<f64>() / hull.len() as f64,
        hull.iter().map(|vertex| vertex[1]).sum::<f64>() / hull.len() as f64,
    ];
    (0..hull.len()).all(|index| {
        let expanded = |vertex: [f64; 2]| {
            [
                centroid[0] + (vertex[0] - centroid[0]) * scale,
                centroid[1] + (vertex[1] - centroid[1]) * scale,
            ]
        };
        let left = expanded(hull[index]);
        let right = expanded(hull[(index + 1) % hull.len()]);
        hull_cross(left, right, point) >= -1e-10
    })
}

fn residual_lut_3d_hull_weight(rgb: [f64; 3], hull: &[[f64; 2]]) -> f64 {
    let Some(point) = root_polynomial_chromaticity(rgb) else {
        return 0.0;
    };
    if point_inside_scaled_chromaticity_hull(point, hull, 1.0) {
        return 1.0;
    }
    if !point_inside_scaled_chromaticity_hull(point, hull, RESIDUAL_LUT_3D_HULL_EXPANSION) {
        return 0.0;
    }
    let mut outside_scale = 1.0;
    let mut inside_scale = RESIDUAL_LUT_3D_HULL_EXPANSION;
    for _ in 0..24 {
        let middle = 0.5 * (outside_scale + inside_scale);
        if point_inside_scaled_chromaticity_hull(point, hull, middle) {
            inside_scale = middle;
        } else {
            outside_scale = middle;
        }
    }
    ((RESIDUAL_LUT_3D_HULL_EXPANSION - inside_scale) / (RESIDUAL_LUT_3D_HULL_EXPANSION - 1.0))
        .clamp(0.0, 1.0)
}

fn interpolate_residual_lut_3d(
    grid_size: u8,
    input_min: [f64; 3],
    input_max: [f64; 3],
    residual_nodes_xyz: &[[f64; 3]],
    rgb: [f64; 3],
) -> Option<[f64; 3]> {
    let grid_size_usize = usize::from(grid_size);
    if grid_size_usize < 2 || residual_nodes_xyz.len() != residual_lut_3d_node_count(grid_size) {
        return None;
    }
    let normalized = residual_lut_3d_normalized_coordinates(input_min, input_max, rgb)?;
    let weights = residual_lut_3d_tetrahedral_weights(grid_size_usize, normalized);
    let mut residual = [0.0; 3];
    for (node, weight) in weights {
        for channel in 0..3 {
            residual[channel] += residual_nodes_xyz[node][channel] * weight;
        }
    }
    Some(residual)
}

fn residual_lut_3d_baseline_xyz(
    matrix: &Matrix3<f64>,
    root_model: Option<&RootPolynomialColorModel>,
    rgb: [f64; 3],
) -> [f64; 3] {
    root_model.map_or_else(
        || vector3_to_array(matrix * Vector3::new(rgb[0], rgb[1], rgb[2])),
        |model| evaluate_root_polynomial_xyz(model, rgb),
    )
}

#[derive(Clone, Copy)]
struct ResidualLut3dLattice<'a> {
    grid_size: u8,
    input_min: [f64; 3],
    input_max: [f64; 3],
    nodes_xyz: &'a [[f64; 3]],
    chromaticity_hull: &'a [[f64; 2]],
}

fn evaluate_residual_lut_3d_nodes_xyz(
    lattice: ResidualLut3dLattice<'_>,
    matrix: &Matrix3<f64>,
    root_model: Option<&RootPolynomialColorModel>,
    rgb: [f64; 3],
) -> [f64; 3] {
    let baseline = residual_lut_3d_baseline_xyz(matrix, root_model, rgb);
    let Some(residual) = interpolate_residual_lut_3d(
        lattice.grid_size,
        lattice.input_min,
        lattice.input_max,
        lattice.nodes_xyz,
        rgb,
    ) else {
        return baseline;
    };
    let weight = residual_lut_3d_hull_weight(rgb, lattice.chromaticity_hull);
    std::array::from_fn(|channel| baseline[channel] + residual[channel] * weight)
}

/// Evaluate an evidence-qualified residual LUT with its declared matrix or
/// root-polynomial baseline. Pixels outside measured support return the
/// baseline instead of extrapolating the residual lattice.
pub fn evaluate_residual_lut_3d_xyz(
    model: &ResidualLut3dColorModel,
    matrix_rows: &[[f64; 3]; 3],
    root_model: Option<&RootPolynomialColorModel>,
    rgb: [f64; 3],
) -> [f64; 3] {
    let baseline_matches = match model.baseline_kind.as_str() {
        "matrix" => model.baseline_model_id.is_none() && root_model.is_none(),
        "root_polynomial" => root_model
            .is_some_and(|root| model.baseline_model_id.as_deref() == Some(root.model_id.as_str())),
        _ => false,
    };
    if !baseline_matches {
        return [f64::NAN; 3];
    }
    evaluate_residual_lut_3d_nodes_xyz(
        ResidualLut3dLattice {
            grid_size: model.grid_size,
            input_min: model.input_min,
            input_max: model.input_max,
            nodes_xyz: &model.residual_nodes_xyz,
            chromaticity_hull: &model.training_chromaticity_hull,
        },
        &rows_to_matrix3(matrix_rows),
        root_model,
        rgb,
    )
}

/// True only when a source sample receives the full residual LUT rather than a
/// tapered or baseline-only fallback.
pub fn residual_lut_3d_has_full_support(model: &ResidualLut3dColorModel, rgb: [f64; 3]) -> bool {
    residual_lut_3d_normalized_coordinates(model.input_min, model.input_max, rgb).is_some()
        && residual_lut_3d_hull_weight(rgb, &model.training_chromaticity_hull) >= 1.0 - 1e-9
}

fn residual_lut_3d_design_matrix(
    patches: &[TargetPatch],
    grid_size: u8,
    input_min: [f64; 3],
    input_max: [f64; 3],
) -> Result<DMatrix<f64>, String> {
    let grid = usize::from(grid_size);
    let unknowns = residual_lut_3d_interior_node_count(grid_size);
    let mut design = DMatrix::<f64>::zeros(patches.len(), unknowns);
    for (row, patch) in patches.iter().enumerate() {
        let normalized =
            residual_lut_3d_normalized_coordinates(input_min, input_max, patch.source_rgb)
                .ok_or_else(|| {
                    format!(
                        "residual 3D LUT patch {:?} is outside the training input domain",
                        patch.patch_id
                    )
                })?;
        for (full_node, weight) in residual_lut_3d_tetrahedral_weights(grid, normalized) {
            let r = full_node / (grid * grid);
            let remainder = full_node % (grid * grid);
            let g = remainder / grid;
            let b = remainder % grid;
            if let Some(interior) = residual_lut_3d_interior_index(grid, r, g, b) {
                design[(row, interior)] += weight;
            }
        }
    }
    Ok(design)
}

fn residual_lut_3d_smoothness_penalty(grid_size: u8) -> DMatrix<f64> {
    let grid = usize::from(grid_size);
    let unknowns = residual_lut_3d_interior_node_count(grid_size);
    let mut penalty = DMatrix::<f64>::zeros(unknowns, unknowns);
    for r in 1..grid - 1 {
        for g in 1..grid - 1 {
            for b in 1..grid - 1 {
                let current = residual_lut_3d_interior_index(grid, r, g, b)
                    .expect("interior node has an index");
                for (axis, coordinate) in [(0usize, r), (1, g), (2, b)] {
                    if coordinate == 1 {
                        penalty[(current, current)] += 1.0;
                    }
                    let mut neighbor = [r, g, b];
                    neighbor[axis] += 1;
                    if neighbor[axis] + 1 == grid {
                        penalty[(current, current)] += 1.0;
                    } else {
                        let adjacent = residual_lut_3d_interior_index(
                            grid,
                            neighbor[0],
                            neighbor[1],
                            neighbor[2],
                        )
                        .expect("positive interior neighbor has an index");
                        penalty[(current, current)] += 1.0;
                        penalty[(adjacent, adjacent)] += 1.0;
                        penalty[(current, adjacent)] -= 1.0;
                        penalty[(adjacent, current)] -= 1.0;
                    }
                }
            }
        }
    }
    penalty
}

fn dynamic_condition_number(matrix: &DMatrix<f64>) -> f64 {
    let singular = matrix.clone().svd(false, false).singular_values;
    let maximum = singular.iter().copied().fold(0.0f64, f64::max);
    let minimum = singular.iter().copied().fold(f64::INFINITY, f64::min);
    if !minimum.is_finite() || minimum <= 1e-15 || maximum <= 0.0 {
        f64::INFINITY
    } else {
        maximum / minimum
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_residual_lut_3d_nodes(
    patches: &[TargetPatch],
    grid_size: u8,
    input_min: [f64; 3],
    input_max: [f64; 3],
    matrix: &Matrix3<f64>,
    root_model: Option<&RootPolynomialColorModel>,
    regularization_lambda: f64,
    weights: Option<&[f64]>,
) -> Result<(Vec<[f64; 3]>, f64), String> {
    let mut design = residual_lut_3d_design_matrix(patches, grid_size, input_min, input_max)?;
    let mut target = DMatrix::from_fn(patches.len(), 3, |row, channel| {
        let baseline = residual_lut_3d_baseline_xyz(matrix, root_model, patches[row].source_rgb);
        patches[row].reference_xyz[channel] - baseline[channel]
    });
    if let Some(weights) = weights {
        if weights.len() != patches.len() {
            return Err("residual 3D LUT fit weight count does not match patch count".to_string());
        }
        for (row, weight) in weights.iter().enumerate() {
            let scale = weight.clamp(0.0, 1.0).sqrt();
            design.row_mut(row).scale_mut(scale);
            target.row_mut(row).scale_mut(scale);
        }
    }
    let transpose = design.transpose();
    let mut normal = &transpose * &design;
    let data_trace = normal.trace().max(1e-12);
    let penalty = residual_lut_3d_smoothness_penalty(grid_size);
    let penalty_scale = data_trace / penalty.trace().max(1e-12);
    normal += penalty * (regularization_lambda * penalty_scale);
    let ridge_scale = data_trace / normal.nrows().max(1) as f64;
    normal += DMatrix::identity(normal.nrows(), normal.ncols())
        * (regularization_lambda * ridge_scale * 1e-3);
    let condition = dynamic_condition_number(&normal);
    let right = transpose * target;
    let coefficients = normal.lu().solve(&right).ok_or_else(|| {
        "residual 3D LUT regularized least-squares solve was singular".to_string()
    })?;
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err("residual 3D LUT fit produced non-finite nodes".to_string());
    }
    let grid = usize::from(grid_size);
    let mut nodes = vec![[0.0; 3]; residual_lut_3d_node_count(grid_size)];
    for r in 1..grid - 1 {
        for g in 1..grid - 1 {
            for b in 1..grid - 1 {
                let interior = residual_lut_3d_interior_index(grid, r, g, b)
                    .expect("interior node has an index");
                nodes[residual_lut_3d_node_index(grid, r, g, b)] = [
                    coefficients[(interior, 0)],
                    coefficients[(interior, 1)],
                    coefficients[(interior, 2)],
                ];
            }
        }
    }
    Ok((nodes, condition))
}

#[allow(clippy::too_many_arguments)]
fn residual_lut_3d_cross_validation_lambda(
    patches: &[TargetPatch],
    grid_size: u8,
    input_min: [f64; 3],
    input_max: [f64; 3],
    hull: &[[f64; 2]],
    matrix: &Matrix3<f64>,
    root_model: Option<&RootPolynomialColorModel>,
) -> Result<(f64, usize), String> {
    const LAMBDAS: [f64; 6] = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0];
    let unknowns = residual_lut_3d_interior_node_count(grid_size);
    let folds = if patches.len() >= unknowns * 7 { 5 } else { 3 };
    let mut ordered = (0..patches.len()).collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        target_hue_family(patches[*left].reference_xyz)
            .cmp(target_hue_family(patches[*right].reference_xyz))
            .then_with(|| {
                patches[*left].reference_xyz[1].total_cmp(&patches[*right].reference_xyz[1])
            })
            .then_with(|| patches[*left].patch_id.cmp(&patches[*right].patch_id))
    });
    let mut best = None::<(f64, f64)>;
    for lambda in LAMBDAS {
        let mut sum_sq = 0.0;
        let mut count = 0usize;
        for fold in 0..folds {
            let mut fit_patches = Vec::new();
            let mut validation_patches = Vec::new();
            for (position, patch_index) in ordered.iter().enumerate() {
                if position % folds == fold {
                    validation_patches.push(patches[*patch_index].clone());
                } else {
                    fit_patches.push(patches[*patch_index].clone());
                }
            }
            let (nodes, _) = solve_residual_lut_3d_nodes(
                &fit_patches,
                grid_size,
                input_min,
                input_max,
                matrix,
                root_model,
                lambda,
                None,
            )?;
            let lattice = ResidualLut3dLattice {
                grid_size,
                input_min,
                input_max,
                nodes_xyz: &nodes,
                chromaticity_hull: hull,
            };
            for patch in &validation_patches {
                let predicted = evaluate_residual_lut_3d_nodes_xyz(
                    lattice,
                    matrix,
                    root_model,
                    patch.source_rgb,
                );
                for (channel, predicted_value) in predicted.iter().enumerate() {
                    let residual = *predicted_value - patch.reference_xyz[channel];
                    sum_sq += residual * residual;
                    count += 1;
                }
            }
        }
        if count > 0 {
            let rms = (sum_sq / count as f64).sqrt();
            if best.as_ref().is_none_or(|(best_rms, best_lambda)| {
                rms < *best_rms - 1e-12
                    || ((rms - *best_rms).abs() <= 1e-12 && lambda > *best_lambda)
            }) {
                best = Some((rms, lambda));
            }
        }
    }
    best.map(|(_, lambda)| (lambda, folds)).ok_or_else(|| {
        "residual 3D LUT training-only cross-validation could not produce a fit".to_string()
    })
}

#[allow(clippy::too_many_arguments)]
fn robust_residual_lut_3d_nodes(
    patches: &[TargetPatch],
    grid_size: u8,
    input_min: [f64; 3],
    input_max: [f64; 3],
    hull: &[[f64; 2]],
    matrix: &Matrix3<f64>,
    root_model: Option<&RootPolynomialColorModel>,
    lambda: f64,
) -> Result<(Vec<[f64; 3]>, f64), String> {
    let mut weights = vec![1.0; patches.len()];
    let (mut nodes, mut condition) = solve_residual_lut_3d_nodes(
        patches, grid_size, input_min, input_max, matrix, root_model, lambda, None,
    )?;
    for _ in 0..5 {
        let lattice = ResidualLut3dLattice {
            grid_size,
            input_min,
            input_max,
            nodes_xyz: &nodes,
            chromaticity_hull: hull,
        };
        let residuals = patches
            .iter()
            .map(|patch| {
                let predicted = evaluate_residual_lut_3d_nodes_xyz(
                    lattice,
                    matrix,
                    root_model,
                    patch.source_rgb,
                );
                (0..3)
                    .map(|channel| {
                        let residual = predicted[channel] - patch.reference_xyz[channel];
                        residual * residual
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .collect::<Vec<_>>();
        let mut residual_copy = residuals.clone();
        let median = median_f64(&mut residual_copy);
        let mut deviations = residuals
            .iter()
            .map(|residual| (residual - median).abs())
            .collect::<Vec<_>>();
        let sigma = (1.4826 * median_f64(&mut deviations)).max(1e-6);
        let huber = 1.5 * sigma;
        for (weight, residual) in weights.iter_mut().zip(residuals) {
            *weight = if residual <= huber {
                1.0
            } else {
                (huber / residual.max(1e-12)).clamp(0.05, 1.0)
            };
        }
        (nodes, condition) = solve_residual_lut_3d_nodes(
            patches,
            grid_size,
            input_min,
            input_max,
            matrix,
            root_model,
            lambda,
            Some(&weights),
        )?;
    }
    Ok((nodes, condition))
}

fn residual_lut_3d_node_metrics(grid_size: u8, nodes: &[[f64; 3]]) -> (f64, f64, f64) {
    let maximum = nodes
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0f64, f64::max);
    let node_rms = (nodes
        .iter()
        .flatten()
        .map(|value| value * value)
        .sum::<f64>()
        / (nodes.len() * 3).max(1) as f64)
        .sqrt();
    let grid = usize::from(grid_size);
    let mut edge_sum_sq = 0.0;
    let mut edge_count = 0usize;
    for r in 0..grid {
        for g in 0..grid {
            for b in 0..grid {
                let current = nodes[residual_lut_3d_node_index(grid, r, g, b)];
                for (axis, coordinate) in [(0usize, r), (1, g), (2, b)] {
                    if coordinate + 1 >= grid {
                        continue;
                    }
                    let mut adjacent_coordinate = [r, g, b];
                    adjacent_coordinate[axis] += 1;
                    let adjacent = nodes[residual_lut_3d_node_index(
                        grid,
                        adjacent_coordinate[0],
                        adjacent_coordinate[1],
                        adjacent_coordinate[2],
                    )];
                    for channel in 0..3 {
                        let difference = adjacent[channel] - current[channel];
                        edge_sum_sq += difference * difference;
                        edge_count += 1;
                    }
                }
            }
        }
    }
    let roughness = (edge_sum_sq / edge_count.max(1) as f64).sqrt();
    (maximum, node_rms, roughness)
}

fn residual_lut_3d_occupied_cell_fraction(
    patches: &[TargetPatch],
    grid_size: u8,
    input_min: [f64; 3],
    input_max: [f64; 3],
) -> f64 {
    let grid = usize::from(grid_size);
    let mut occupied = std::collections::BTreeSet::<usize>::new();
    for patch in patches {
        let Some(normalized) =
            residual_lut_3d_normalized_coordinates(input_min, input_max, patch.source_rgb)
        else {
            continue;
        };
        let scale = (grid - 1) as f64;
        let cell = normalized.map(|value| (value * scale).floor() as usize);
        let cell = cell.map(|value| value.min(grid - 2));
        occupied.insert((cell[0] * (grid - 1) + cell[1]) * (grid - 1) + cell[2]);
    }
    occupied.len() as f64 / (grid - 1).pow(3).max(1) as f64
}

fn residual_lut_3d_inside_fraction(
    patches: &[TargetPatch],
    input_min: [f64; 3],
    input_max: [f64; 3],
    hull: &[[f64; 2]],
) -> f64 {
    let inside = patches
        .iter()
        .filter(|patch| {
            residual_lut_3d_normalized_coordinates(input_min, input_max, patch.source_rgb).is_some()
                && residual_lut_3d_hull_weight(patch.source_rgb, hull) >= 1.0 - 1e-9
        })
        .count();
    inside as f64 / patches.len().max(1) as f64
}

fn residual_lut_3d_acceptance_reason(validation: &ResidualLut3dValidation) -> (bool, String) {
    let accepted = validation.regularized_condition_number.is_finite()
        && validation.regularized_condition_number <= RESIDUAL_LUT_3D_MAX_CONDITION_NUMBER
        && validation.occupied_cell_fraction >= RESIDUAL_LUT_3D_MIN_OCCUPIED_CELL_FRACTION
        && validation.held_out_inside_domain_fraction
            >= RESIDUAL_LUT_3D_MIN_HELD_OUT_INSIDE_FRACTION
        && validation.residual_node_max_abs <= RESIDUAL_LUT_3D_MAX_NODE_ABS
        && validation.residual_node_rms <= RESIDUAL_LUT_3D_MAX_NODE_RMS
        && validation.residual_edge_roughness_rms <= RESIDUAL_LUT_3D_MAX_ROUGHNESS_RMS
        && validation.held_out_delta_e00_rms <= RESIDUAL_LUT_3D_MAX_DELTA_E00_RMS
        && validation.held_out_delta_e00_max <= RESIDUAL_LUT_3D_MAX_DELTA_E00_MAX
        && validation.held_out_delta_e00_rms_improvement
            >= RESIDUAL_LUT_3D_MIN_DELTA_E00_RMS_IMPROVEMENT
        && validation.held_out_delta_e00_rms_improvement_fraction
            >= RESIDUAL_LUT_3D_MIN_DELTA_E00_RMS_IMPROVEMENT_FRACTION
        && validation.held_out_delta_e00_max
            <= validation.baseline_held_out_delta_e00_max
                + RESIDUAL_LUT_3D_MAX_DELTA_E00_MAX_REGRESSION
        && validation.held_out_xyz_rms_improvement_fraction
            >= -RESIDUAL_LUT_3D_MAX_XYZ_RMS_REGRESSION_FRACTION
        && validation.maximum_hue_family_delta_e00_rms_regression
            <= RESIDUAL_LUT_3D_MAX_HUE_RMS_REGRESSION;
    if accepted {
        return (
            true,
            format!(
                "held-out DeltaE00 RMS improved by {:.3} ({:.1}%) over the declared baseline with bounded support, nodes, roughness, conditioning, XYZ error, maximum error, and hue regressions",
                validation.held_out_delta_e00_rms_improvement,
                validation.held_out_delta_e00_rms_improvement_fraction * 100.0
            ),
        );
    }
    let mut reasons = Vec::new();
    if !validation.regularized_condition_number.is_finite()
        || validation.regularized_condition_number > RESIDUAL_LUT_3D_MAX_CONDITION_NUMBER
    {
        reasons.push("regularized system condition exceeds limit".to_string());
    }
    if validation.occupied_cell_fraction < RESIDUAL_LUT_3D_MIN_OCCUPIED_CELL_FRACTION {
        reasons.push(format!(
            "occupied training-cell fraction {:.1}% is below {:.1}%",
            validation.occupied_cell_fraction * 100.0,
            RESIDUAL_LUT_3D_MIN_OCCUPIED_CELL_FRACTION * 100.0
        ));
    }
    if validation.held_out_inside_domain_fraction < RESIDUAL_LUT_3D_MIN_HELD_OUT_INSIDE_FRACTION {
        reasons.push("held-out samples do not remain inside measured LUT support".to_string());
    }
    if validation.residual_node_max_abs > RESIDUAL_LUT_3D_MAX_NODE_ABS
        || validation.residual_node_rms > RESIDUAL_LUT_3D_MAX_NODE_RMS
        || validation.residual_edge_roughness_rms > RESIDUAL_LUT_3D_MAX_ROUGHNESS_RMS
    {
        reasons.push("residual node magnitude or lattice roughness exceeds limit".to_string());
    }
    if validation.held_out_delta_e00_rms > RESIDUAL_LUT_3D_MAX_DELTA_E00_RMS
        || validation.held_out_delta_e00_max > RESIDUAL_LUT_3D_MAX_DELTA_E00_MAX
    {
        reasons.push("held-out DeltaE00 RMS or maximum exceeds limit".to_string());
    }
    if validation.held_out_delta_e00_rms_improvement < RESIDUAL_LUT_3D_MIN_DELTA_E00_RMS_IMPROVEMENT
        || validation.held_out_delta_e00_rms_improvement_fraction
            < RESIDUAL_LUT_3D_MIN_DELTA_E00_RMS_IMPROVEMENT_FRACTION
    {
        reasons.push("held-out DeltaE00 gain over baseline is not material".to_string());
    }
    if validation.held_out_delta_e00_max
        > validation.baseline_held_out_delta_e00_max + RESIDUAL_LUT_3D_MAX_DELTA_E00_MAX_REGRESSION
    {
        reasons.push("held-out maximum DeltaE00 regressed beyond tolerance".to_string());
    }
    if validation.held_out_xyz_rms_improvement_fraction
        < -RESIDUAL_LUT_3D_MAX_XYZ_RMS_REGRESSION_FRACTION
    {
        reasons.push("held-out XYZ RMS regressed beyond tolerance".to_string());
    }
    if validation.maximum_hue_family_delta_e00_rms_regression
        > RESIDUAL_LUT_3D_MAX_HUE_RMS_REGRESSION
    {
        reasons.push("a held-out hue family regressed beyond tolerance".to_string());
    }
    (false, reasons.join("; "))
}

#[allow(clippy::too_many_arguments)]
fn fit_residual_lut_3d_candidate(
    model_id: String,
    grid_size: u8,
    training_patches: &[TargetPatch],
    held_out_patches: &[TargetPatch],
    matrix: &Matrix3<f64>,
    root_model: Option<&RootPolynomialColorModel>,
) -> Result<ResidualLut3dColorModel, String> {
    let (input_min, input_max) = residual_lut_3d_domain(training_patches)?;
    let hull = root_polynomial_chromaticity_hull(training_patches);
    if hull.len() < 3 {
        return Err("residual 3D LUT training RGB does not span a chromaticity hull".to_string());
    }
    let (lambda, folds) = residual_lut_3d_cross_validation_lambda(
        training_patches,
        grid_size,
        input_min,
        input_max,
        &hull,
        matrix,
        root_model,
    )?;
    let (nodes, condition) = robust_residual_lut_3d_nodes(
        training_patches,
        grid_size,
        input_min,
        input_max,
        &hull,
        matrix,
        root_model,
        lambda,
    )?;
    let lattice = ResidualLut3dLattice {
        grid_size,
        input_min,
        input_max,
        nodes_xyz: &nodes,
        chromaticity_hull: &hull,
    };
    let map = |rgb| evaluate_residual_lut_3d_nodes_xyz(lattice, matrix, root_model, rgb);
    let training = evaluate_target_mapping(training_patches, "residual_lut_3d_training", map);
    let mut held_out = evaluate_target_mapping(
        held_out_patches,
        &format!("residual_lut_3d_grid_{grid_size}_held_out"),
        map,
    );
    let baseline = evaluate_target_mapping(
        held_out_patches,
        "residual_lut_3d_declared_baseline_held_out",
        |rgb| residual_lut_3d_baseline_xyz(matrix, root_model, rgb),
    );
    let identity = evaluate_target_matrix(
        &Matrix3::identity(),
        held_out_patches,
        "identity_rgb_to_xyz_baseline",
    );
    held_out.fit.validation = Some(TargetFitValidationDiagnostics {
        evaluation_set: "held_out".to_string(),
        training_patch_count: training_patches.len(),
        held_out_patch_count: held_out_patches.len(),
        training_residual_rms: training.fit.target_residual_rms,
        training_residual_max: training.fit.target_residual_max,
        identity_baseline_residual_rms: identity.target_residual_rms,
        identity_baseline_residual_max: identity.target_residual_max,
        held_out_identity_improvement_fraction: if identity.target_residual_rms > 1e-12 {
            (identity.target_residual_rms - held_out.fit.target_residual_rms)
                / identity.target_residual_rms
        } else {
            0.0
        },
    });
    let delta_e_improvement = baseline.delta_e00_rms - held_out.delta_e00_rms;
    let delta_e_improvement_fraction = if baseline.delta_e00_rms > 1e-12 {
        delta_e_improvement / baseline.delta_e00_rms
    } else {
        0.0
    };
    let xyz_improvement_fraction = if baseline.fit.target_residual_rms > 1e-12 {
        (baseline.fit.target_residual_rms - held_out.fit.target_residual_rms)
            / baseline.fit.target_residual_rms
    } else {
        0.0
    };
    let (node_max, node_rms, roughness) = residual_lut_3d_node_metrics(grid_size, &nodes);
    let validation = ResidualLut3dValidation {
        training_patch_count: training_patches.len(),
        held_out_patch_count: held_out_patches.len(),
        cross_validation_folds: folds,
        regularization_lambda: lambda,
        regularized_condition_number: condition,
        occupied_cell_fraction: residual_lut_3d_occupied_cell_fraction(
            training_patches,
            grid_size,
            input_min,
            input_max,
        ),
        held_out_inside_domain_fraction: residual_lut_3d_inside_fraction(
            held_out_patches,
            input_min,
            input_max,
            &hull,
        ),
        residual_node_max_abs: node_max,
        residual_node_rms: node_rms,
        residual_edge_roughness_rms: roughness,
        training_delta_e00_rms: training.delta_e00_rms,
        training_delta_e00_max: training.delta_e00_max,
        held_out_delta_e00_rms: held_out.delta_e00_rms,
        held_out_delta_e00_max: held_out.delta_e00_max,
        baseline_held_out_delta_e00_rms: baseline.delta_e00_rms,
        baseline_held_out_delta_e00_max: baseline.delta_e00_max,
        held_out_delta_e00_rms_improvement: delta_e_improvement,
        held_out_delta_e00_rms_improvement_fraction: delta_e_improvement_fraction,
        held_out_xyz_rms_improvement_fraction: xyz_improvement_fraction,
        maximum_hue_family_delta_e00_rms_regression: maximum_hue_rms_regression(
            &held_out, &baseline,
        ),
        selected_over_baseline: false,
    };
    Ok(ResidualLut3dColorModel {
        model_id,
        model_type: RESIDUAL_LUT_3D_MODEL_TYPE.to_string(),
        interpolation: RESIDUAL_LUT_3D_INTERPOLATION.to_string(),
        grid_size,
        output_space: ROOT_POLYNOMIAL_OUTPUT_SPACE.to_string(),
        baseline_kind: if root_model.is_some() {
            "root_polynomial"
        } else {
            "matrix"
        }
        .to_string(),
        baseline_model_id: root_model.map(|model| model.model_id.clone()),
        input_min,
        input_max,
        residual_nodes_xyz: nodes,
        training_patches: training_patches.to_vec(),
        training_chromaticity_hull: hull,
        validation,
        fit: held_out.fit,
    })
}

/// Fit smooth 5^3 and, with substantially more evidence, 7^3 tetrahedral
/// residual LUT candidates. The LUT is selected only when it materially beats
/// the already-qualified root-polynomial or matrix baseline on disjoint data.
pub fn fit_residual_lut_3d_color_model_from_disjoint_patches(
    model_id_prefix: &str,
    training_patches: &[TargetPatch],
    held_out_patches: &[TargetPatch],
    matrix_rows: &[[f64; 3]; 3],
    root_model: Option<&RootPolynomialColorModel>,
) -> Result<ResidualLut3dFitSelection, String> {
    validate_disjoint_target_patch_sets(training_patches, held_out_patches)?;
    let matrix = rows_to_matrix3(matrix_rows);
    let baseline_kind = if root_model.is_some() {
        "root_polynomial"
    } else {
        "matrix"
    };
    let baseline_model_id = root_model.map(|model| model.model_id.clone());
    let mut diagnostics = Vec::<ResidualLut3dCandidateDiagnostics>::new();
    let mut accepted = Vec::<(usize, ResidualLut3dColorModel)>::new();
    for grid_size in [5u8, 7u8] {
        let (minimum_training, minimum_held_out) =
            residual_lut_3d_minimum_counts(grid_size).expect("supported grid size");
        let node_count = residual_lut_3d_node_count(grid_size);
        let fitted_interior_node_count = residual_lut_3d_interior_node_count(grid_size);
        if training_patches.len() < minimum_training || held_out_patches.len() < minimum_held_out {
            diagnostics.push(ResidualLut3dCandidateDiagnostics {
                grid_size,
                node_count,
                fitted_interior_node_count,
                status: "not_evaluated_insufficient_samples".to_string(),
                reason: format!(
                    "grid {grid_size} requires at least {minimum_training} training and {minimum_held_out} held-out patches; received {} and {}",
                    training_patches.len(),
                    held_out_patches.len()
                ),
                model_id: None,
                validation: None,
            });
            continue;
        }
        let model_id = format!("{model_id_prefix}-residual-lut3d-g{grid_size}");
        match fit_residual_lut_3d_candidate(
            model_id.clone(),
            grid_size,
            training_patches,
            held_out_patches,
            &matrix,
            root_model,
        ) {
            Ok(mut model) => {
                let (qualifies, reason) = residual_lut_3d_acceptance_reason(&model.validation);
                model.validation.selected_over_baseline = qualifies;
                let index = diagnostics.len();
                diagnostics.push(ResidualLut3dCandidateDiagnostics {
                    grid_size,
                    node_count,
                    fitted_interior_node_count,
                    status: if qualifies {
                        "accepted_candidate"
                    } else {
                        "rejected_held_out"
                    }
                    .to_string(),
                    reason,
                    model_id: Some(model_id),
                    validation: Some(model.validation.clone()),
                });
                if qualifies {
                    accepted.push((index, model));
                }
            }
            Err(error) => diagnostics.push(ResidualLut3dCandidateDiagnostics {
                grid_size,
                node_count,
                fitted_interior_node_count,
                status: "rejected_fit".to_string(),
                reason: error,
                model_id: Some(model_id),
                validation: None,
            }),
        }
    }
    let mut selected = accepted.first().cloned();
    for candidate in accepted.iter().skip(1) {
        let Some((_, current)) = selected.as_ref() else {
            selected = Some(candidate.clone());
            continue;
        };
        let gain = current.validation.held_out_delta_e00_rms
            - candidate.1.validation.held_out_delta_e00_rms;
        let gain_fraction = if current.validation.held_out_delta_e00_rms > 1e-12 {
            gain / current.validation.held_out_delta_e00_rms
        } else {
            0.0
        };
        if gain >= RESIDUAL_LUT_3D_HIGHER_GRID_MIN_ABSOLUTE_GAIN
            && gain_fraction >= RESIDUAL_LUT_3D_HIGHER_GRID_MIN_RELATIVE_GAIN
            && candidate.1.validation.held_out_delta_e00_max
                <= current.validation.held_out_delta_e00_max + 0.50
        {
            selected = Some(candidate.clone());
        }
    }
    let selected_index = selected.as_ref().map(|(index, _)| *index);
    for (index, candidate) in diagnostics.iter_mut().enumerate() {
        if candidate.status == "accepted_candidate" {
            if Some(index) == selected_index {
                candidate.status = "selected".to_string();
                candidate.reason = format!(
                    "{}; selected as the most complex residual lattice with a material held-out gain over every simpler accepted baseline",
                    candidate.reason
                );
            } else {
                candidate.status = "accepted_not_selected".to_string();
                candidate.reason = format!(
                    "{}; accepted but added lattice density did not materially beat the selected simpler model",
                    candidate.reason
                );
            }
        }
    }
    let selected_model = selected.map(|(_, model)| model);
    Ok(ResidualLut3dFitSelection {
        diagnostics: ResidualLut3dSelectionDiagnostics {
            status: if selected_model.is_some() {
                "selected_residual_lut_3d"
            } else {
                "baseline_retained"
            }
            .to_string(),
            baseline_kind: baseline_kind.to_string(),
            baseline_model_id,
            selection_rule: "lattice domain, regularization, and robust fit use training evidence; disjoint held-out DeltaE00, XYZ residual, maximum error, hue-family regression, occupancy, support, node magnitude, roughness, and conditioning must beat the already-qualified baseline; a denser grid must materially beat the simpler accepted lattice"
                .to_string(),
            selected_model_id: selected_model.as_ref().map(|model| model.model_id.clone()),
            selected_grid_size: selected_model.as_ref().map(|model| model.grid_size),
            candidates: diagnostics,
        },
        selected_model,
    })
}

fn metric_matches(recorded: f64, recomputed: f64) -> bool {
    let tolerance = 1e-9 + 1e-8 * recorded.abs().max(recomputed.abs());
    recorded.is_finite() && recomputed.is_finite() && (recorded - recomputed).abs() <= tolerance
}

fn validate_chromaticity_hull(hull: &[[f64; 2]], rejection_details: &mut Vec<String>) {
    if hull.len() < 3 {
        rejection_details.push(
            "color_model.training_chromaticity_hull must contain at least three vertices"
                .to_string(),
        );
        return;
    }
    for (index, point) in hull.iter().enumerate() {
        if point
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            || point[0] + point[1] > 1.0 + 1e-9
        {
            rejection_details.push(format!(
                "color_model.training_chromaticity_hull[{index}] must be a finite RGB chromaticity inside the unit simplex"
            ));
        }
    }
    for index in 0..hull.len() {
        let cross = hull_cross(
            hull[index],
            hull[(index + 1) % hull.len()],
            hull[(index + 2) % hull.len()],
        );
        if !cross.is_finite() || cross <= 1e-12 {
            rejection_details.push(
                "color_model.training_chromaticity_hull must be a strictly convex counter-clockwise polygon"
                    .to_string(),
            );
            break;
        }
    }
}

/// Validate a stored nonlinear model against its retained training evidence,
/// the independently held-out patches, and the profile's matrix fallback.
/// Fitting, regularization, complexity selection, support hull, residuals, and
/// acceptance deltas are recomputed rather than trusting serialized claims.
pub fn validate_root_polynomial_color_model(
    model: &RootPolynomialColorModel,
    matrix_rows: &[[f64; 3]; 3],
    held_out_patches: &[TargetPatch],
) -> Vec<String> {
    let mut rejection_details = Vec::new();
    if model.model_id.trim().is_empty() {
        rejection_details.push("color_model.model_id must not be empty".to_string());
    }
    if model.basis != ROOT_POLYNOMIAL_BASIS {
        rejection_details.push(format!(
            "color_model.basis must be `{ROOT_POLYNOMIAL_BASIS}`"
        ));
    }
    if model.output_space != ROOT_POLYNOMIAL_OUTPUT_SPACE {
        rejection_details.push(format!(
            "color_model.output_space must be `{ROOT_POLYNOMIAL_OUTPUT_SPACE}`"
        ));
    }
    let Some(term_count) = root_polynomial_term_count(model.degree) else {
        rejection_details.push("color_model.degree must be 2 or 3".to_string());
        return rejection_details;
    };
    if model.coefficients.len() != term_count {
        rejection_details.push(format!(
            "color_model.coefficients contains {} terms; degree {} requires {term_count}",
            model.coefficients.len(),
            model.degree
        ));
    }
    if model
        .coefficients
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        rejection_details.push("color_model.coefficients must be finite".to_string());
    }
    let recomputed_coefficient_max = model
        .coefficients
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0f64, f64::max);
    if recomputed_coefficient_max > ROOT_POLYNOMIAL_MAX_COEFFICIENT_ABS {
        rejection_details.push(format!(
            "color_model coefficient magnitude {recomputed_coefficient_max:.3} exceeds {ROOT_POLYNOMIAL_MAX_COEFFICIENT_ABS:.3}"
        ));
    }
    if !metric_matches(
        model.validation.coefficient_max_abs,
        recomputed_coefficient_max,
    ) {
        rejection_details.push(
            "color_model.validation.coefficient_max_abs is inconsistent with coefficients"
                .to_string(),
        );
    }
    if let Err(error) =
        validate_disjoint_target_patch_sets(&model.training_patches, held_out_patches)
    {
        rejection_details.push(format!(
            "color_model retained training/held-out evidence is invalid: {error}"
        ));
    }
    if model.validation.training_patch_count != model.training_patches.len() {
        rejection_details.push(format!(
            "color_model.validation.training_patch_count {} must equal retained training_patches count {}",
            model.validation.training_patch_count,
            model.training_patches.len()
        ));
    }
    let recomputed_hull = root_polynomial_chromaticity_hull(&model.training_patches);
    let hull_matches = recomputed_hull.len() == model.training_chromaticity_hull.len()
        && recomputed_hull
            .iter()
            .zip(&model.training_chromaticity_hull)
            .all(|(recomputed, recorded)| {
                metric_matches(recorded[0], recomputed[0])
                    && metric_matches(recorded[1], recomputed[1])
            });
    if !hull_matches {
        rejection_details.push(
            "color_model.training_chromaticity_hull is inconsistent with retained training_patches"
                .to_string(),
        );
    }
    validate_chromaticity_hull(&model.training_chromaticity_hull, &mut rejection_details);
    validate_optional_fit(
        "color_model.fit",
        Some(&model.fit),
        Some(CALIBRATION_PROFILE_SCHEMA_VERSION),
        &mut rejection_details,
    );
    let Some((minimum_training, minimum_held_out)) = root_polynomial_minimum_counts(model.degree)
    else {
        return rejection_details;
    };
    if model.validation.training_patch_count < minimum_training {
        rejection_details.push(format!(
            "color_model.validation.training_patch_count {} is below degree-{} minimum {minimum_training}",
            model.validation.training_patch_count, model.degree
        ));
    }
    if model.validation.held_out_patch_count < minimum_held_out {
        rejection_details.push(format!(
            "color_model.validation.held_out_patch_count {} is below degree-{} minimum {minimum_held_out}",
            model.validation.held_out_patch_count, model.degree
        ));
    }
    if model.validation.held_out_patch_count != held_out_patches.len()
        || model.fit.patch_count != held_out_patches.len()
    {
        rejection_details.push(format!(
            "color_model held-out counts must equal retained patches count {}",
            held_out_patches.len()
        ));
    }
    if !(3..=5).contains(&model.validation.cross_validation_folds) {
        rejection_details
            .push("color_model.validation.cross_validation_folds must be in [3, 5]".to_string());
    }
    if !model.validation.regularization_lambda.is_finite()
        || !(1e-10..=1e-3).contains(&model.validation.regularization_lambda)
    {
        rejection_details.push(
            "color_model.validation.regularization_lambda must be finite and in [1e-10, 1e-3]"
                .to_string(),
        );
    }
    if !model.validation.design_condition_number.is_finite()
        || model.validation.design_condition_number > ROOT_POLYNOMIAL_MAX_CONDITION_NUMBER
    {
        rejection_details.push(format!(
            "color_model.validation.design_condition_number must be finite and at most {ROOT_POLYNOMIAL_MAX_CONDITION_NUMBER:.3}"
        ));
    }
    for (field, value) in [
        (
            "training_delta_e00_rms",
            model.validation.training_delta_e00_rms,
        ),
        (
            "training_delta_e00_max",
            model.validation.training_delta_e00_max,
        ),
        (
            "held_out_delta_e00_rms",
            model.validation.held_out_delta_e00_rms,
        ),
        (
            "held_out_delta_e00_max",
            model.validation.held_out_delta_e00_max,
        ),
        (
            "matrix_held_out_delta_e00_rms",
            model.validation.matrix_held_out_delta_e00_rms,
        ),
        (
            "matrix_held_out_delta_e00_max",
            model.validation.matrix_held_out_delta_e00_max,
        ),
        (
            "maximum_hue_family_delta_e00_rms_regression",
            model.validation.maximum_hue_family_delta_e00_rms_regression,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            rejection_details.push(format!(
                "color_model.validation.{field} must be finite and non-negative"
            ));
        }
    }
    if !model.validation.selected_over_matrix {
        rejection_details.push(
            "color_model.validation.selected_over_matrix must be true for an applied model"
                .to_string(),
        );
    }
    if !rejection_details.is_empty()
        || held_out_patches.is_empty()
        || model.coefficients.len() != term_count
    {
        return rejection_details;
    }

    let matrix = rows_to_matrix3(matrix_rows);
    match fit_root_polynomial_color_model_from_disjoint_patches(
        "load-time-revalidation",
        &model.training_patches,
        held_out_patches,
        matrix_rows,
    ) {
        Ok(selection) => match selection.selected_model {
            Some(recomputed) if recomputed.degree == model.degree => {
                for (term, (recorded, expected)) in model
                    .coefficients
                    .iter()
                    .zip(&recomputed.coefficients)
                    .enumerate()
                {
                    for channel in 0..3 {
                        if !metric_matches(recorded[channel], expected[channel]) {
                            rejection_details.push(format!(
                                "color_model.coefficients[{term}][{channel}] is inconsistent with the retained training patches and deterministic robust fit"
                            ));
                        }
                    }
                }
                if model.validation.cross_validation_folds
                    != recomputed.validation.cross_validation_folds
                {
                    rejection_details.push(
                        "color_model.validation.cross_validation_folds is inconsistent with retained training patches"
                            .to_string(),
                    );
                }
                for (field, recorded, expected) in [
                    (
                        "regularization_lambda",
                        model.validation.regularization_lambda,
                        recomputed.validation.regularization_lambda,
                    ),
                    (
                        "design_condition_number",
                        model.validation.design_condition_number,
                        recomputed.validation.design_condition_number,
                    ),
                    (
                        "training_delta_e00_rms",
                        model.validation.training_delta_e00_rms,
                        recomputed.validation.training_delta_e00_rms,
                    ),
                    (
                        "training_delta_e00_max",
                        model.validation.training_delta_e00_max,
                        recomputed.validation.training_delta_e00_max,
                    ),
                ] {
                    if !metric_matches(recorded, expected) {
                        rejection_details.push(format!(
                            "color_model.validation.{field} is inconsistent with retained training patches and deterministic refit"
                        ));
                    }
                }
            }
            Some(recomputed) => rejection_details.push(format!(
                "color_model degree {} is inconsistent with deterministic held-out complexity selection, which selected degree {}",
                model.degree, recomputed.degree
            )),
            None => rejection_details.push(
                "color_model is inconsistent with deterministic held-out complexity selection, which retained the matrix fallback"
                    .to_string(),
            ),
        },
        Err(error) => rejection_details.push(format!(
            "color_model could not be deterministically refit from retained evidence: {error}"
        )),
    }
    let candidate = evaluate_target_mapping(
        held_out_patches,
        &format!("root_polynomial_degree_{}_held_out", model.degree),
        |rgb| evaluate_root_polynomial_xyz(model, rgb),
    );
    let baseline = evaluate_target_mapping(
        held_out_patches,
        "least_squares_matrix_held_out_baseline",
        |rgb| vector3_to_array(matrix * Vector3::new(rgb[0], rgb[1], rgb[2])),
    );
    let expected_delta_e_improvement = baseline.delta_e00_rms - candidate.delta_e00_rms;
    let expected_delta_e_fraction = if baseline.delta_e00_rms > 1e-12 {
        expected_delta_e_improvement / baseline.delta_e00_rms
    } else {
        0.0
    };
    let expected_xyz_fraction = if baseline.fit.target_residual_rms > 1e-12 {
        (baseline.fit.target_residual_rms - candidate.fit.target_residual_rms)
            / baseline.fit.target_residual_rms
    } else {
        0.0
    };
    let expected_hue_regression = maximum_hue_rms_regression(&candidate, &baseline);
    for (field, recorded, recomputed) in [
        (
            "held_out_delta_e00_rms",
            model.validation.held_out_delta_e00_rms,
            candidate.delta_e00_rms,
        ),
        (
            "held_out_delta_e00_max",
            model.validation.held_out_delta_e00_max,
            candidate.delta_e00_max,
        ),
        (
            "matrix_held_out_delta_e00_rms",
            model.validation.matrix_held_out_delta_e00_rms,
            baseline.delta_e00_rms,
        ),
        (
            "matrix_held_out_delta_e00_max",
            model.validation.matrix_held_out_delta_e00_max,
            baseline.delta_e00_max,
        ),
        (
            "held_out_delta_e00_rms_improvement",
            model.validation.held_out_delta_e00_rms_improvement,
            expected_delta_e_improvement,
        ),
        (
            "held_out_delta_e00_rms_improvement_fraction",
            model.validation.held_out_delta_e00_rms_improvement_fraction,
            expected_delta_e_fraction,
        ),
        (
            "held_out_xyz_rms_improvement_fraction",
            model.validation.held_out_xyz_rms_improvement_fraction,
            expected_xyz_fraction,
        ),
        (
            "maximum_hue_family_delta_e00_rms_regression",
            model.validation.maximum_hue_family_delta_e00_rms_regression,
            expected_hue_regression,
        ),
        (
            "fit.target_residual_rms",
            model.fit.target_residual_rms,
            candidate.fit.target_residual_rms,
        ),
        (
            "fit.target_residual_max",
            model.fit.target_residual_max,
            candidate.fit.target_residual_max,
        ),
    ] {
        if !metric_matches(recorded, recomputed) {
            rejection_details.push(format!(
                "color_model.validation.{field} is inconsistent with retained held-out patches and coefficients"
            ));
        }
    }
    let mut recomputed_validation = model.validation.clone();
    recomputed_validation.coefficient_max_abs = recomputed_coefficient_max;
    recomputed_validation.held_out_delta_e00_rms = candidate.delta_e00_rms;
    recomputed_validation.held_out_delta_e00_max = candidate.delta_e00_max;
    recomputed_validation.matrix_held_out_delta_e00_rms = baseline.delta_e00_rms;
    recomputed_validation.matrix_held_out_delta_e00_max = baseline.delta_e00_max;
    recomputed_validation.held_out_delta_e00_rms_improvement = expected_delta_e_improvement;
    recomputed_validation.held_out_delta_e00_rms_improvement_fraction = expected_delta_e_fraction;
    recomputed_validation.held_out_xyz_rms_improvement_fraction = expected_xyz_fraction;
    recomputed_validation.maximum_hue_family_delta_e00_rms_regression = expected_hue_regression;
    let (accepted, reason) = root_polynomial_acceptance_reason(&recomputed_validation);
    if !accepted {
        rejection_details.push(format!(
            "color_model no longer passes recomputed held-out acceptance: {reason}"
        ));
    }
    rejection_details
}

/// Validate and deterministically refit a stored residual 3D LUT from its
/// retained training evidence, independently held-out patches, and declared
/// matrix/root-polynomial baseline.
pub fn validate_residual_lut_3d_color_model(
    model: &ResidualLut3dColorModel,
    matrix_rows: &[[f64; 3]; 3],
    root_model: Option<&RootPolynomialColorModel>,
    held_out_patches: &[TargetPatch],
) -> Vec<String> {
    let mut rejection_details = Vec::new();
    if model.model_id.trim().is_empty() {
        rejection_details.push("lut_3d_model.model_id must not be empty".to_string());
    }
    if model.model_type != RESIDUAL_LUT_3D_MODEL_TYPE {
        rejection_details.push(format!(
            "lut_3d_model.model_type must be `{RESIDUAL_LUT_3D_MODEL_TYPE}`"
        ));
    }
    if model.interpolation != RESIDUAL_LUT_3D_INTERPOLATION {
        rejection_details.push(format!(
            "lut_3d_model.interpolation must be `{RESIDUAL_LUT_3D_INTERPOLATION}`"
        ));
    }
    if model.output_space != ROOT_POLYNOMIAL_OUTPUT_SPACE {
        rejection_details.push(format!(
            "lut_3d_model.output_space must be `{ROOT_POLYNOMIAL_OUTPUT_SPACE}`"
        ));
    }
    let Some((minimum_training, minimum_held_out)) =
        residual_lut_3d_minimum_counts(model.grid_size)
    else {
        rejection_details.push("lut_3d_model.grid_size must be 5 or 7".to_string());
        return rejection_details;
    };
    let baseline_matches = match model.baseline_kind.as_str() {
        "matrix" => {
            if model.baseline_model_id.is_some() {
                rejection_details.push(
                    "matrix-based lut_3d_model must not declare baseline_model_id".to_string(),
                );
            }
            if root_model.is_some() {
                rejection_details.push(
                    "matrix-based lut_3d_model is inconsistent with the retained selected root-polynomial baseline"
                        .to_string(),
                );
            }
            root_model.is_none() && model.baseline_model_id.is_none()
        }
        "root_polynomial" => match (model.baseline_model_id.as_deref(), root_model) {
            (Some(expected), Some(root)) if expected == root.model_id => true,
            (Some(expected), Some(root)) => {
                rejection_details.push(format!(
                    "lut_3d_model baseline_model_id `{expected}` does not match root-polynomial model `{}`",
                    root.model_id
                ));
                false
            }
            (None, _) => {
                rejection_details
                    .push("root-polynomial lut_3d_model requires baseline_model_id".to_string());
                false
            }
            (Some(_), None) => {
                rejection_details.push(
                    "root-polynomial lut_3d_model requires the declared color_model baseline"
                        .to_string(),
                );
                false
            }
        },
        baseline => {
            rejection_details.push(format!(
                "unsupported lut_3d_model.baseline_kind `{baseline}`"
            ));
            false
        }
    };
    let expected_node_count = residual_lut_3d_node_count(model.grid_size);
    if model.residual_nodes_xyz.len() != expected_node_count {
        rejection_details.push(format!(
            "lut_3d_model.residual_nodes_xyz contains {} nodes; grid {} requires {expected_node_count}",
            model.residual_nodes_xyz.len(), model.grid_size
        ));
    }
    if model
        .residual_nodes_xyz
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        rejection_details.push("lut_3d_model.residual_nodes_xyz must be finite".to_string());
    }
    if model.residual_nodes_xyz.len() == expected_node_count {
        let grid = usize::from(model.grid_size);
        for r in 0..grid {
            for g in 0..grid {
                for b in 0..grid {
                    if r == 0 || g == 0 || b == 0 || r + 1 == grid || g + 1 == grid || b + 1 == grid
                    {
                        let node =
                            model.residual_nodes_xyz[residual_lut_3d_node_index(grid, r, g, b)];
                        if node.iter().any(|value| value.abs() > 1e-12) {
                            rejection_details.push(
                                "lut_3d_model boundary residual nodes must be zero for continuous baseline fallback"
                                    .to_string(),
                            );
                            break;
                        }
                    }
                }
            }
        }
    }
    if let Err(error) =
        validate_disjoint_target_patch_sets(&model.training_patches, held_out_patches)
    {
        rejection_details.push(format!(
            "lut_3d_model retained training/held-out evidence is invalid: {error}"
        ));
    }
    if model.training_patches.len() < minimum_training {
        rejection_details.push(format!(
            "lut_3d_model training count {} is below grid-{} minimum {minimum_training}",
            model.training_patches.len(),
            model.grid_size
        ));
    }
    if held_out_patches.len() < minimum_held_out {
        rejection_details.push(format!(
            "lut_3d_model held-out count {} is below grid-{} minimum {minimum_held_out}",
            held_out_patches.len(),
            model.grid_size
        ));
    }
    if model.validation.training_patch_count != model.training_patches.len()
        || model.validation.held_out_patch_count != held_out_patches.len()
        || model.fit.patch_count != held_out_patches.len()
    {
        rejection_details.push(
            "lut_3d_model validation/fit counts must match retained training and held-out patches"
                .to_string(),
        );
    }
    match residual_lut_3d_domain(&model.training_patches) {
        Ok((minimum, maximum)) => {
            for channel in 0..3 {
                if !metric_matches(model.input_min[channel], minimum[channel])
                    || !metric_matches(model.input_max[channel], maximum[channel])
                {
                    rejection_details.push(
                        "lut_3d_model input domain is inconsistent with retained training patches"
                            .to_string(),
                    );
                    break;
                }
            }
        }
        Err(error) => rejection_details.push(format!(
            "lut_3d_model retained training domain is invalid: {error}"
        )),
    }
    let recomputed_hull = root_polynomial_chromaticity_hull(&model.training_patches);
    if recomputed_hull.len() != model.training_chromaticity_hull.len()
        || !recomputed_hull
            .iter()
            .zip(&model.training_chromaticity_hull)
            .all(|(expected, recorded)| {
                metric_matches(expected[0], recorded[0]) && metric_matches(expected[1], recorded[1])
            })
    {
        rejection_details.push(
            "lut_3d_model.training_chromaticity_hull is inconsistent with retained training patches"
                .to_string(),
        );
    }
    validate_chromaticity_hull(&model.training_chromaticity_hull, &mut rejection_details);
    validate_optional_fit(
        "lut_3d_model.fit",
        Some(&model.fit),
        Some(CALIBRATION_PROFILE_SCHEMA_VERSION),
        &mut rejection_details,
    );
    if !(3..=5).contains(&model.validation.cross_validation_folds) {
        rejection_details
            .push("lut_3d_model.validation.cross_validation_folds must be in [3, 5]".to_string());
    }
    if !model.validation.regularization_lambda.is_finite()
        || !(1e-4..=10.0).contains(&model.validation.regularization_lambda)
    {
        rejection_details.push(
            "lut_3d_model.validation.regularization_lambda must be finite and in [1e-4, 10]"
                .to_string(),
        );
    }
    for (field, value) in [
        (
            "regularized_condition_number",
            model.validation.regularized_condition_number,
        ),
        (
            "occupied_cell_fraction",
            model.validation.occupied_cell_fraction,
        ),
        (
            "held_out_inside_domain_fraction",
            model.validation.held_out_inside_domain_fraction,
        ),
        (
            "residual_node_max_abs",
            model.validation.residual_node_max_abs,
        ),
        ("residual_node_rms", model.validation.residual_node_rms),
        (
            "residual_edge_roughness_rms",
            model.validation.residual_edge_roughness_rms,
        ),
        (
            "training_delta_e00_rms",
            model.validation.training_delta_e00_rms,
        ),
        (
            "training_delta_e00_max",
            model.validation.training_delta_e00_max,
        ),
        (
            "held_out_delta_e00_rms",
            model.validation.held_out_delta_e00_rms,
        ),
        (
            "held_out_delta_e00_max",
            model.validation.held_out_delta_e00_max,
        ),
        (
            "baseline_held_out_delta_e00_rms",
            model.validation.baseline_held_out_delta_e00_rms,
        ),
        (
            "baseline_held_out_delta_e00_max",
            model.validation.baseline_held_out_delta_e00_max,
        ),
        (
            "maximum_hue_family_delta_e00_rms_regression",
            model.validation.maximum_hue_family_delta_e00_rms_regression,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            rejection_details.push(format!(
                "lut_3d_model.validation.{field} must be finite and non-negative"
            ));
        }
    }
    for (field, value) in [
        (
            "occupied_cell_fraction",
            model.validation.occupied_cell_fraction,
        ),
        (
            "held_out_inside_domain_fraction",
            model.validation.held_out_inside_domain_fraction,
        ),
    ] {
        if value > 1.0 {
            rejection_details.push(format!("lut_3d_model.validation.{field} must be at most 1"));
        }
    }
    if !model.validation.selected_over_baseline {
        rejection_details.push(
            "lut_3d_model.validation.selected_over_baseline must be true for an applied model"
                .to_string(),
        );
    }
    if !baseline_matches
        || !rejection_details.is_empty()
        || model.residual_nodes_xyz.len() != expected_node_count
    {
        return rejection_details;
    }

    match fit_residual_lut_3d_color_model_from_disjoint_patches(
        "load-time-revalidation",
        &model.training_patches,
        held_out_patches,
        matrix_rows,
        root_model,
    ) {
        Ok(selection) => match selection.selected_model {
            Some(recomputed) if recomputed.grid_size == model.grid_size => {
                for (node, (recorded, expected)) in model
                    .residual_nodes_xyz
                    .iter()
                    .zip(&recomputed.residual_nodes_xyz)
                    .enumerate()
                {
                    for channel in 0..3 {
                        if !metric_matches(recorded[channel], expected[channel]) {
                            rejection_details.push(format!(
                                "lut_3d_model.residual_nodes_xyz[{node}][{channel}] is inconsistent with retained training patches and deterministic robust fit"
                            ));
                        }
                    }
                }
                if model.validation.cross_validation_folds
                    != recomputed.validation.cross_validation_folds
                {
                    rejection_details.push(
                        "lut_3d_model.validation.cross_validation_folds is inconsistent with deterministic refit"
                            .to_string(),
                    );
                }
                for (field, recorded, expected) in [
                    (
                        "regularization_lambda",
                        model.validation.regularization_lambda,
                        recomputed.validation.regularization_lambda,
                    ),
                    (
                        "regularized_condition_number",
                        model.validation.regularized_condition_number,
                        recomputed.validation.regularized_condition_number,
                    ),
                    (
                        "occupied_cell_fraction",
                        model.validation.occupied_cell_fraction,
                        recomputed.validation.occupied_cell_fraction,
                    ),
                    (
                        "held_out_inside_domain_fraction",
                        model.validation.held_out_inside_domain_fraction,
                        recomputed.validation.held_out_inside_domain_fraction,
                    ),
                    (
                        "residual_node_max_abs",
                        model.validation.residual_node_max_abs,
                        recomputed.validation.residual_node_max_abs,
                    ),
                    (
                        "residual_node_rms",
                        model.validation.residual_node_rms,
                        recomputed.validation.residual_node_rms,
                    ),
                    (
                        "residual_edge_roughness_rms",
                        model.validation.residual_edge_roughness_rms,
                        recomputed.validation.residual_edge_roughness_rms,
                    ),
                    (
                        "training_delta_e00_rms",
                        model.validation.training_delta_e00_rms,
                        recomputed.validation.training_delta_e00_rms,
                    ),
                    (
                        "training_delta_e00_max",
                        model.validation.training_delta_e00_max,
                        recomputed.validation.training_delta_e00_max,
                    ),
                    (
                        "held_out_delta_e00_rms",
                        model.validation.held_out_delta_e00_rms,
                        recomputed.validation.held_out_delta_e00_rms,
                    ),
                    (
                        "held_out_delta_e00_max",
                        model.validation.held_out_delta_e00_max,
                        recomputed.validation.held_out_delta_e00_max,
                    ),
                    (
                        "baseline_held_out_delta_e00_rms",
                        model.validation.baseline_held_out_delta_e00_rms,
                        recomputed.validation.baseline_held_out_delta_e00_rms,
                    ),
                    (
                        "baseline_held_out_delta_e00_max",
                        model.validation.baseline_held_out_delta_e00_max,
                        recomputed.validation.baseline_held_out_delta_e00_max,
                    ),
                    (
                        "held_out_delta_e00_rms_improvement",
                        model.validation.held_out_delta_e00_rms_improvement,
                        recomputed.validation.held_out_delta_e00_rms_improvement,
                    ),
                    (
                        "held_out_delta_e00_rms_improvement_fraction",
                        model
                            .validation
                            .held_out_delta_e00_rms_improvement_fraction,
                        recomputed
                            .validation
                            .held_out_delta_e00_rms_improvement_fraction,
                    ),
                    (
                        "held_out_xyz_rms_improvement_fraction",
                        model.validation.held_out_xyz_rms_improvement_fraction,
                        recomputed.validation.held_out_xyz_rms_improvement_fraction,
                    ),
                    (
                        "maximum_hue_family_delta_e00_rms_regression",
                        model
                            .validation
                            .maximum_hue_family_delta_e00_rms_regression,
                        recomputed
                            .validation
                            .maximum_hue_family_delta_e00_rms_regression,
                    ),
                    (
                        "fit.target_residual_rms",
                        model.fit.target_residual_rms,
                        recomputed.fit.target_residual_rms,
                    ),
                    (
                        "fit.target_residual_max",
                        model.fit.target_residual_max,
                        recomputed.fit.target_residual_max,
                    ),
                ] {
                    if !metric_matches(recorded, expected) {
                        rejection_details.push(format!(
                            "lut_3d_model.validation.{field} is inconsistent with retained evidence and deterministic refit"
                        ));
                    }
                }
            }
            Some(recomputed) => rejection_details.push(format!(
                "lut_3d_model grid {} is inconsistent with deterministic complexity selection, which selected grid {}",
                model.grid_size, recomputed.grid_size
            )),
            None => rejection_details.push(
                "lut_3d_model is inconsistent with deterministic held-out complexity selection, which retained its simpler baseline"
                    .to_string(),
            ),
        },
        Err(error) => rejection_details.push(format!(
            "lut_3d_model could not be deterministically refit from retained evidence: {error}"
        )),
    }
    let (accepted, reason) = residual_lut_3d_acceptance_reason(&model.validation);
    if !accepted {
        rejection_details.push(format!(
            "lut_3d_model no longer passes recomputed held-out acceptance: {reason}"
        ));
    }
    rejection_details
}

fn fit_rgb_to_xyz_in_sample(
    patches: &[TargetPatch],
    method: &str,
    whitepoint_override: Option<[f64; 3]>,
    confidence_limit: Option<f64>,
) -> Result<MatrixFitResult, String> {
    if patches.len() < MIN_TARGET_PATCHES {
        return Err(format!(
            "target fit requires at least {MIN_TARGET_PATCHES} patches, got {}",
            patches.len()
        ));
    }

    let mut normal = Matrix3::zeros();
    let mut cross = Matrix3::zeros();
    for (idx, patch) in patches.iter().enumerate() {
        validate_target_patch(idx, patch)?;
        let source = Vector3::new(
            patch.source_rgb[0],
            patch.source_rgb[1],
            patch.source_rgb[2],
        );
        let target = Vector3::new(
            patch.reference_xyz[0],
            patch.reference_xyz[1],
            patch.reference_xyz[2],
        );
        normal += source * source.transpose();
        cross += target * source.transpose();
    }

    let normal_condition = matrix_condition_number(&normal);
    if !normal_condition.is_finite() {
        return Err("target patch RGB basis is singular".to_string());
    }
    if normal_condition > MAX_MATRIX_CONDITION_NUMBER {
        return Err(format!(
            "target patch RGB basis condition number {normal_condition:.3} exceeds maximum {MAX_MATRIX_CONDITION_NUMBER:.3}"
        ));
    }
    let inverse = normal
        .try_inverse()
        .ok_or_else(|| "target patch RGB basis is singular".to_string())?;
    let matrix = cross * inverse;

    let mut matrix_rejections = Vec::new();
    let matrix_condition_number =
        validate_matrix_condition("fitted target", &matrix, &mut matrix_rejections);
    if !matrix_rejections.is_empty() {
        return Err(matrix_rejections.join("; "));
    }

    let fit = evaluate_target_matrix(&matrix, patches, method);
    let confidence = target_fit_confidence(&fit, confidence_limit, "fitted target")?;

    let whitepoint = match whitepoint_override {
        Some(whitepoint) => {
            let mut rejections = Vec::new();
            validate_whitepoint(&whitepoint, &mut rejections);
            if !rejections.is_empty() {
                return Err(rejections.join("; "));
            }
            whitepoint
        }
        None => estimate_whitepoint_from_patches(patches)?,
    };

    Ok(MatrixFitResult {
        matrix: matrix3_to_rows(&matrix),
        whitepoint,
        confidence,
        matrix_condition_number,
        fit,
    })
}

/// Fit on one target sample set and derive every residual/confidence claim
/// from a separate held-out set. Patch IDs are treated as sample IDs and must
/// be present, unique, and disjoint across both sets.
pub fn fit_rgb_to_xyz_from_disjoint_patches(
    training_patches: &[TargetPatch],
    held_out_patches: &[TargetPatch],
    method: &str,
    whitepoint_override: Option<[f64; 3]>,
    confidence_limit: Option<f64>,
) -> Result<MatrixFitResult, String> {
    validate_disjoint_target_patch_sets(training_patches, held_out_patches)?;

    let training_result =
        fit_rgb_to_xyz_in_sample(training_patches, method, whitepoint_override, None)?;
    let matrix = rows_to_matrix3(&training_result.matrix);
    let training_fit = training_result.fit;
    let mut held_out_fit = evaluate_target_matrix(&matrix, held_out_patches, method);
    let identity_fit = evaluate_target_matrix(
        &Matrix3::identity(),
        held_out_patches,
        "identity_rgb_to_xyz_baseline",
    );
    let held_out_identity_improvement_fraction = if identity_fit.target_residual_rms > 1e-12 {
        (identity_fit.target_residual_rms - held_out_fit.target_residual_rms)
            / identity_fit.target_residual_rms
    } else {
        0.0
    };
    held_out_fit.validation = Some(TargetFitValidationDiagnostics {
        evaluation_set: "held_out".to_string(),
        training_patch_count: training_patches.len(),
        held_out_patch_count: held_out_patches.len(),
        training_residual_rms: training_fit.target_residual_rms,
        training_residual_max: training_fit.target_residual_max,
        identity_baseline_residual_rms: identity_fit.target_residual_rms,
        identity_baseline_residual_max: identity_fit.target_residual_max,
        held_out_identity_improvement_fraction,
    });
    let confidence = target_fit_confidence(&held_out_fit, confidence_limit, "held-out target")?;

    Ok(MatrixFitResult {
        matrix: training_result.matrix,
        whitepoint: training_result.whitepoint,
        confidence,
        matrix_condition_number: training_result.matrix_condition_number,
        fit: held_out_fit,
    })
}

pub fn validate_disjoint_target_patch_sets(
    training_patches: &[TargetPatch],
    held_out_patches: &[TargetPatch],
) -> Result<(), String> {
    if training_patches.len() < MIN_TARGET_TRAINING_PATCHES {
        return Err(format!(
            "target matrix fit requires at least {MIN_TARGET_TRAINING_PATCHES} training patches, got {}",
            training_patches.len()
        ));
    }
    if held_out_patches.len() < MIN_TARGET_HELD_OUT_PATCHES {
        return Err(format!(
            "target matrix fit requires at least {MIN_TARGET_HELD_OUT_PATCHES} held-out patches, got {}",
            held_out_patches.len()
        ));
    }

    let mut sample_ids = std::collections::HashSet::<String>::new();
    let mut sample_values = std::collections::HashSet::<[u64; 8]>::new();
    for (set_name, patches) in [
        ("training_patches", training_patches),
        ("held_out_patches", held_out_patches),
    ] {
        for (idx, patch) in patches.iter().enumerate() {
            validate_target_patch(idx, patch).map_err(|err| format!("{set_name}.{err}"))?;
            let sample_id = patch
                .patch_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| {
                    format!(
                        "{set_name}[{idx}] requires a non-empty id/patch_id so training and held-out samples can be proven disjoint"
                    )
                })?;
            let normalized_id = sample_id.to_ascii_lowercase();
            if !sample_ids.insert(normalized_id) {
                return Err(format!(
                    "target matrix training and held-out sample IDs must be globally unique; duplicate `{sample_id}`"
                ));
            }
            let value_signature = std::array::from_fn(|channel| {
                let value = match channel {
                    0..=2 => Some(patch.source_rgb[channel]),
                    3..=5 => Some(patch.reference_xyz[channel - 3]),
                    6..=7 => patch.scanner_xy.map(|xy| xy[channel - 6]),
                    _ => unreachable!(),
                };
                match value {
                    Some(0.0) => 0,
                    Some(value) => value.to_bits(),
                    None => u64::MAX,
                }
            });
            if !sample_values.insert(value_signature) {
                return Err(format!(
                    "target matrix training and held-out measurements must be disjoint; {set_name}[{idx}] exactly duplicates an earlier RGB/reference sample"
                ));
            }
        }
    }
    Ok(())
}

fn evaluate_target_matrix(
    matrix: &Matrix3<f64>,
    patches: &[TargetPatch],
    method: &str,
) -> TargetFitDiagnostics {
    evaluate_target_transform(patches, method, |source_rgb| {
        let source = Vector3::new(source_rgb[0], source_rgb[1], source_rgb[2]);
        vector3_to_array(matrix * source)
    })
}

fn evaluate_target_transform<F>(
    patches: &[TargetPatch],
    method: &str,
    mut transform: F,
) -> TargetFitDiagnostics
where
    F: FnMut([f64; 3]) -> [f64; 3],
{
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut patch_residuals = Vec::with_capacity(patches.len());
    let mut hue_residuals = std::collections::BTreeMap::<String, (usize, f64, f64)>::new();
    for patch in patches {
        let target = Vector3::new(
            patch.reference_xyz[0],
            patch.reference_xyz[1],
            patch.reference_xyz[2],
        );
        let fitted_array = transform(patch.source_rgb);
        let fitted = Vector3::new(fitted_array[0], fitted_array[1], fitted_array[2]);
        let residual = fitted - target;
        let mut patch_max_abs = 0.0f64;
        for channel in 0..3 {
            sum_sq += residual[channel] * residual[channel];
            max_abs = max_abs.max(residual[channel].abs());
            patch_max_abs = patch_max_abs.max(residual[channel].abs());
        }
        let residual_error = residual.norm();
        let hue_family = target_hue_family(patch.reference_xyz).to_string();
        let hue_entry = hue_residuals
            .entry(hue_family.clone())
            .or_insert((0, 0.0, 0.0));
        hue_entry.0 += 1;
        hue_entry.1 += residual_error * residual_error;
        hue_entry.2 = hue_entry.2.max(residual_error);
        patch_residuals.push(TargetPatchFitResidual {
            patch_id: patch.patch_id.clone(),
            hue_family,
            source_rgb: patch.source_rgb,
            reference_xyz: patch.reference_xyz,
            fitted_xyz: vector3_to_array(fitted),
            residual_xyz: vector3_to_array(residual),
            residual_error,
            max_channel_error: patch_max_abs,
        });
    }
    let target_residual_rms = (sum_sq / (patches.len() as f64 * 3.0)).sqrt();
    let mut per_hue_residuals = hue_residuals
        .into_iter()
        .map(
            |(hue_family, (patch_count, sum_sq, residual_max))| TargetHueResidual {
                hue_family,
                patch_count,
                residual_rms: (sum_sq / patch_count.max(1) as f64).sqrt(),
                residual_max,
            },
        )
        .collect::<Vec<_>>();
    per_hue_residuals.sort_by(|left, right| {
        right
            .residual_max
            .partial_cmp(&left.residual_max)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                right
                    .residual_rms
                    .partial_cmp(&left.residual_rms)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.hue_family.cmp(&right.hue_family))
    });
    let mut worst_patches = patch_residuals;
    worst_patches.sort_by(|left, right| {
        right
            .residual_error
            .partial_cmp(&left.residual_error)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                right
                    .max_channel_error
                    .partial_cmp(&left.max_channel_error)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.patch_id.cmp(&right.patch_id))
    });
    worst_patches.truncate(5);
    TargetFitDiagnostics {
        method: method.to_string(),
        patch_count: patches.len(),
        target_residual_rms,
        target_residual_max: max_abs,
        validation: None,
        per_hue_residuals,
        worst_patches,
    }
}

fn target_fit_confidence(
    fit: &TargetFitDiagnostics,
    confidence_limit: Option<f64>,
    label: &str,
) -> Result<f64, String> {
    let residual_confidence =
        (-8.0 * fit.target_residual_rms - 2.0 * fit.target_residual_max).exp();
    let confidence = confidence_limit
        .unwrap_or(1.0)
        .min(residual_confidence)
        .clamp(0.0, 1.0);
    if confidence < MIN_CALIBRATION_CONFIDENCE {
        return Err(format!(
            "{label} confidence {confidence:.3} is below required minimum {MIN_CALIBRATION_CONFIDENCE:.3} (rms residual {:.6}, max residual {:.6})",
            fit.target_residual_rms, fit.target_residual_max
        ));
    }
    Ok(confidence)
}

fn load_library_selection(
    library_dir: Option<&Path>,
    requested_scanner_id: Option<&str>,
    requested_roll_id: Option<&str>,
    requested_film_stock: Option<&str>,
    observed_base_color: Option<[f64; 3]>,
) -> CalibrationLoadResult {
    let Some(library_dir) = library_dir else {
        return library_rejected(
            None,
            None,
            None,
            requested_film_stock.map(str::to_string),
            requested_film_stock.map(|stock| FilmStockDiagnostics {
                status: "rejected".to_string(),
                selection: "explicit".to_string(),
                requested_stock: stock.to_string(),
                matched_hint: None,
                matched_roll_profiles: Vec::new(),
                reason:
                    "a calibration library directory is required when selecting film stock evidence"
                        .to_string(),
                rejection_details: vec!["missing calibration library directory".to_string()],
            }),
            Vec::new(),
            "a calibration library directory is required when selecting scanner, roll, or film stock profiles"
                .to_string(),
            vec!["missing calibration library directory".to_string()],
        );
    };

    let library = load_library_dir(library_dir);
    let mut diagnostics = CalibrationDiagnostics {
        status: "not_configured".to_string(),
        source: "image_derived_neutral_and_dominant_anchors".to_string(),
        color_mapping_application: None,
        external_profile: None,
        library: Some(CalibrationLibraryDiagnostics {
            path: library_dir.to_string_lossy().to_string(),
            scanner_profiles_loaded: library.scanners.len(),
            roll_profiles_loaded: library.rolls.len(),
            film_hints_loaded: library.film_hints.len(),
            invalid_entries: library.invalid_entries.clone(),
        }),
        scanner_profile: None,
        roll_profile: None,
        requested_film_stock: requested_film_stock.map(str::to_string),
        film_stock: film_stock_diagnostics(
            &library.film_hints,
            &library.rolls,
            requested_film_stock,
        ),
        nearest_roll_candidates: nearest_roll_candidates(
            &library.rolls,
            observed_base_color,
            requested_film_stock,
        ),
        nearest_film_candidates: film_hint_candidates(&library.film_hints, requested_film_stock),
        profile_schema_version: None,
        confidence: None,
        reason: "no scanner profile was selected or confidently auto-matched".to_string(),
        matrix_condition_number: None,
        whitepoint: None,
        calibration_upgrade_available: None,
        calibration_upgrade_reason: None,
        rejection_details: Vec::new(),
    };

    let (selected_scanner, scanner_diag, scanner_rejected) =
        select_scanner_profile(&library.scanners, requested_scanner_id);
    diagnostics.scanner_profile = scanner_diag;

    let (selected_roll, roll_diag, roll_rejected) = select_roll_profile(
        &library.rolls,
        requested_roll_id,
        selected_scanner.as_ref(),
        requested_film_stock,
        observed_base_color,
    );
    diagnostics.roll_profile = roll_diag;

    if scanner_rejected || roll_rejected {
        diagnostics.status = "rejected".to_string();
        diagnostics.reason =
            "selected calibration library profile could not be applied; using image-derived mapping"
                .to_string();
        diagnostics.rejection_details = selected_rejection_details(&diagnostics);
        return CalibrationLoadResult {
            profile: None,
            diagnostics,
        };
    }

    let Some(scanner) = selected_scanner else {
        return CalibrationLoadResult {
            profile: None,
            diagnostics,
        };
    };

    let profile = build_library_profile(&scanner, selected_roll.as_ref(), observed_base_color);
    diagnostics.status = "applied".to_string();
    diagnostics.source = "calibration_library".to_string();
    diagnostics.profile_schema_version = Some(profile.schema_version);
    diagnostics.confidence = Some(profile.confidence);
    diagnostics.matrix_condition_number = Some(profile.matrix_condition_number);
    diagnostics.whitepoint = Some(profile.whitepoint);
    if profile.schema_version < CALIBRATION_PROFILE_SCHEMA_VERSION {
        diagnostics.calibration_upgrade_available = Some(true);
        diagnostics.calibration_upgrade_reason = Some(format!(
            "selected calibration library profile uses schema_version {}; schema_version {} can record scanner-domain provenance, validate and apply typed root-polynomial colour, and retain review-grade DeltaE00 summaries",
            profile.schema_version, CALIBRATION_PROFILE_SCHEMA_VERSION
        ));
    }
    diagnostics.reason = match (
        profile.application_mode,
        profile.roll_profile_id.as_ref(),
        profile.roll_correction_applied,
    ) {
        (CalibrationApplicationMode::DirectProfile, Some(_), true) => {
            "scanner profile and explicit roll correction selected from calibration library"
                .to_string()
        }
        (CalibrationApplicationMode::ScannerConstrainedImageAdaptation, Some(_), false) => {
            "scanner profile selected and base-only roll metadata recorded; using scanner-constrained image-derived adaptation"
                .to_string()
        }
        _ => {
            "scanner profile selected; using scanner-constrained image-derived adaptation because no roll correction was selected"
                .to_string()
        }
    };

    CalibrationLoadResult {
        profile: Some(profile),
        diagnostics,
    }
}

#[allow(clippy::too_many_arguments)]
fn library_rejected(
    library: Option<CalibrationLibraryDiagnostics>,
    scanner_profile: Option<ScannerProfileDiagnostics>,
    roll_profile: Option<RollProfileDiagnostics>,
    requested_film_stock: Option<String>,
    film_stock: Option<FilmStockDiagnostics>,
    candidates: Vec<RollCandidateDiagnostics>,
    reason: String,
    rejection_details: Vec<String>,
) -> CalibrationLoadResult {
    CalibrationLoadResult {
        profile: None,
        diagnostics: CalibrationDiagnostics {
            status: "rejected".to_string(),
            source: "image_derived_neutral_and_dominant_anchors".to_string(),
            color_mapping_application: None,
            external_profile: None,
            library,
            scanner_profile,
            roll_profile,
            requested_film_stock,
            film_stock,
            nearest_roll_candidates: candidates,
            nearest_film_candidates: Vec::new(),
            profile_schema_version: None,
            confidence: None,
            reason,
            matrix_condition_number: None,
            whitepoint: None,
            calibration_upgrade_available: None,
            calibration_upgrade_reason: None,
            rejection_details,
        },
    }
}

fn validate_raw_profile(raw: RawCalibrationProfile, path: Option<&Path>) -> CalibrationLoadResult {
    let mut rejection_details = Vec::new();

    let schema_version = validate_schema_version(raw.schema_version, &mut rejection_details);

    for (field, value) in [
        ("source_space", raw.source_space.as_ref()),
        ("scanner", raw.scanner.as_ref()),
        ("film", raw.film.as_ref()),
        ("target", raw.target.as_ref()),
        ("reference", raw.reference.as_ref()),
    ] {
        validate_required_object(field, value, &mut rejection_details);
    }
    validate_target_sampling_evidence(
        "target_sampling",
        raw.target_sampling.as_ref(),
        &mut rejection_details,
    );

    let confidence = validate_confidence(raw.confidence, &mut rejection_details);
    validate_optional_fit(
        "fit",
        raw.fit.as_ref(),
        schema_version,
        &mut rejection_details,
    );
    let signal_domain_fit = raw
        .fit
        .as_ref()
        .or_else(|| raw.color_model.as_ref().map(|model| &model.fit))
        .or_else(|| raw.lut_3d_model.as_ref().map(|model| &model.fit));
    validate_target_patch_signal_domain(
        schema_version,
        signal_domain_fit,
        raw.scanner_linearization.as_ref(),
        raw.target_patch_signal_domain.as_deref(),
        raw.target_patch_linearization_model_id.as_deref(),
        &mut rejection_details,
    );
    let target_patches = optional_target_patches_from_values(
        raw.patches.as_deref(),
        "patches",
        &mut rejection_details,
    );
    if let Some(response) = raw.negative_response.as_ref() {
        if let Err(errors) = density::validate_measured_negative_response(response) {
            rejection_details.extend(errors);
        }
    }
    if let Some(linearization) = raw.scanner_linearization.as_ref() {
        if let Err(errors) = scanner_linearization::validate_scanner_linearization(linearization) {
            rejection_details.extend(errors);
        }
    }

    let whitepoint = match raw.whitepoint {
        Some(whitepoint) => {
            validate_whitepoint(&whitepoint, &mut rejection_details);
            Some(whitepoint)
        }
        None => {
            rejection_details.push("missing whitepoint".to_string());
            None
        }
    };

    let matrix = match raw.work_to_xyz.or(raw.fit_matrix) {
        Some(matrix) => {
            validate_matrix_values("work_to_xyz", &matrix, &mut rejection_details);
            Some(matrix)
        }
        None => {
            rejection_details.push("missing work_to_xyz or fit_matrix".to_string());
            None
        }
    };
    let matrix_condition_number = matrix
        .as_ref()
        .map(rows_to_matrix3)
        .map(|matrix| validate_matrix_condition("work_to_xyz", &matrix, &mut rejection_details));
    if raw.color_model.is_some() && schema_version.is_some_and(|version| version < 2) {
        rejection_details.push("color_model requires schema_version 2 or newer".to_string());
    }
    if raw.lut_3d_model.is_some() && schema_version.is_some_and(|version| version < 2) {
        rejection_details.push("lut_3d_model requires schema_version 2 or newer".to_string());
    }
    if let (Some(model), Some(matrix)) = (raw.color_model.as_ref(), matrix.as_ref()) {
        rejection_details.extend(validate_root_polynomial_color_model(
            model,
            matrix,
            &target_patches,
        ));
    }
    if let (Some(model), Some(matrix)) = (raw.lut_3d_model.as_ref(), matrix.as_ref()) {
        rejection_details.extend(validate_residual_lut_3d_color_model(
            model,
            matrix,
            raw.color_model.as_ref(),
            &target_patches,
        ));
    }

    if !rejection_details.is_empty() {
        return rejected(
            path,
            Some(&raw),
            format!(
                "calibration profile rejected: {}",
                rejection_details
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "validation failed".to_string())
            ),
            rejection_details,
        );
    }

    let profile = CalibrationProfile {
        schema_version: schema_version.expect("schema version validated"),
        profile_id: raw.profile_id.clone(),
        source_space: raw.source_space.clone().expect("source_space validated"),
        scanner: raw.scanner.clone().expect("scanner validated"),
        film: raw.film.clone().expect("film validated"),
        target: raw.target.clone().expect("target validated"),
        reference: raw.reference.clone().expect("reference validated"),
        gamut_limits: raw.gamut_limits.clone(),
        whitepoint: whitepoint.expect("whitepoint validated"),
        work_to_xyz: matrix.expect("matrix validated"),
        confidence: confidence.expect("confidence validated"),
        matrix_condition_number: matrix_condition_number.expect("condition number validated"),
        fit: raw.fit.clone(),
        target_patches,
        application_mode: CalibrationApplicationMode::DirectProfile,
        scanner_profile_id: None,
        roll_profile_id: None,
        roll_correction_applied: false,
        scanner_prior_work_to_xyz: None,
        scanner_prior_confidence: None,
        scanner_prior_fit: None,
        negative_response: raw.negative_response.clone(),
        scanner_linearization: raw.scanner_linearization.clone(),
        color_model: raw.color_model.clone(),
        lut_3d_model: raw.lut_3d_model.clone(),
        color_model_post_xyz: None,
    };
    CalibrationLoadResult {
        diagnostics: CalibrationDiagnostics {
            status: "applied".to_string(),
            source: "external_calibration_profile".to_string(),
            color_mapping_application: None,
            external_profile: Some(external_profile_diagnostics(path, Some(&raw))),
            library: None,
            scanner_profile: None,
            roll_profile: None,
            requested_film_stock: None,
            film_stock: None,
            nearest_roll_candidates: Vec::new(),
            nearest_film_candidates: Vec::new(),
            profile_schema_version: Some(profile.schema_version),
            confidence: Some(profile.confidence),
            reason: "valid external calibration profile selected over image-derived estimate"
                .to_string(),
            matrix_condition_number: Some(profile.matrix_condition_number),
            whitepoint: Some(profile.whitepoint),
            calibration_upgrade_available: (profile.schema_version
                < CALIBRATION_PROFILE_SCHEMA_VERSION)
                .then_some(true),
            calibration_upgrade_reason: (profile.schema_version
                < CALIBRATION_PROFILE_SCHEMA_VERSION)
                .then(|| {
                    format!(
                        "external profile uses schema_version {}; schema_version {} can record scanner-domain provenance and validate and apply typed root-polynomial colour",
                        profile.schema_version, CALIBRATION_PROFILE_SCHEMA_VERSION
                    )
                }),
            rejection_details: Vec::new(),
        },
        profile: Some(profile),
    }
}

fn rejected(
    path: Option<&Path>,
    raw: Option<&RawCalibrationProfile>,
    reason: String,
    rejection_details: Vec<String>,
) -> CalibrationLoadResult {
    CalibrationLoadResult {
        profile: None,
        diagnostics: CalibrationDiagnostics {
            status: "rejected".to_string(),
            source: "image_derived_neutral_and_dominant_anchors".to_string(),
            color_mapping_application: None,
            external_profile: Some(external_profile_diagnostics(path, raw)),
            library: None,
            scanner_profile: None,
            roll_profile: None,
            requested_film_stock: None,
            film_stock: None,
            nearest_roll_candidates: Vec::new(),
            nearest_film_candidates: Vec::new(),
            profile_schema_version: raw.and_then(|raw| raw.schema_version),
            confidence: raw.and_then(|raw| raw.confidence),
            reason,
            matrix_condition_number: raw
                .and_then(|raw| raw.work_to_xyz.or(raw.fit_matrix))
                .map(|matrix| matrix_condition_number(&rows_to_matrix3(&matrix))),
            whitepoint: raw.and_then(|raw| raw.whitepoint),
            calibration_upgrade_available: raw.and_then(|raw| {
                raw.schema_version
                    .is_some_and(|version| version < CALIBRATION_PROFILE_SCHEMA_VERSION)
                    .then_some(true)
            }),
            calibration_upgrade_reason: raw.and_then(|raw| {
                raw.schema_version.and_then(|version| {
                    (version < CALIBRATION_PROFILE_SCHEMA_VERSION).then(|| {
                        format!(
                            "profile uses schema_version {version}; schema_version {CALIBRATION_PROFILE_SCHEMA_VERSION} can record complete perfect-mode calibration evidence"
                        )
                    })
                })
            }),
            rejection_details,
        },
    }
}

fn external_profile_diagnostics(
    path: Option<&Path>,
    raw: Option<&RawCalibrationProfile>,
) -> ExternalProfileDiagnostics {
    ExternalProfileDiagnostics {
        path: path.map(|path| path.to_string_lossy().to_string()),
        profile_id: raw.and_then(|raw| raw.profile_id.clone()),
        source_space: raw.and_then(|raw| raw.source_space.clone()),
        scanner: raw.and_then(|raw| raw.scanner.clone()),
        film: raw.and_then(|raw| raw.film.clone()),
        target: raw.and_then(|raw| raw.target.clone()),
        reference: raw.and_then(|raw| raw.reference.clone()),
        gamut_limits: raw.and_then(|raw| raw.gamut_limits.clone()),
        fit: raw.and_then(|raw| raw.fit.clone()),
        target_patch_signal_domain: raw.and_then(|raw| raw.target_patch_signal_domain.clone()),
        target_patch_linearization_model_id: raw
            .and_then(|raw| raw.target_patch_linearization_model_id.clone()),
        target_sampling: raw.and_then(|raw| raw.target_sampling.clone()),
        color_model: raw.and_then(|raw| raw.color_model.clone()),
        lut_3d_model: raw.and_then(|raw| raw.lut_3d_model.clone()),
    }
}

fn load_library_dir(library_dir: &Path) -> LoadedLibrary {
    let mut library = LoadedLibrary::default();
    let mut stack = vec![library_dir.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            library.invalid_entries.push(LibraryEntryRejection {
                path: dir.to_string_lossy().to_string(),
                record_type: None,
                profile_id: None,
                reasons: vec!["failed to read calibration library directory".to_string()],
            });
            continue;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| !extension.eq_ignore_ascii_case("json"))
                .unwrap_or(true)
            {
                continue;
            }
            load_library_file(&path, &mut library);
        }
    }

    library
}

fn load_library_file(path: &Path, library: &mut LoadedLibrary) {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(err) => {
            library.invalid_entries.push(LibraryEntryRejection {
                path: path.to_string_lossy().to_string(),
                record_type: None,
                profile_id: None,
                reasons: vec![format!("failed to read JSON: {err}")],
            });
            return;
        }
    };
    let value: serde_json::Value = match serde_json::from_str(&contents) {
        Ok(value) => value,
        Err(err) => {
            library.invalid_entries.push(LibraryEntryRejection {
                path: path.to_string_lossy().to_string(),
                record_type: None,
                profile_id: None,
                reasons: vec![format!("failed to parse JSON: {err}")],
            });
            return;
        }
    };

    match validate_library_record(&value, Some(path)) {
        Ok(ValidatedLibraryRecord::Scanner(profile)) => library.scanners.push(*profile),
        Ok(ValidatedLibraryRecord::Roll(profile)) => library.rolls.push(*profile),
        Ok(ValidatedLibraryRecord::FilmHint(hint)) => library.film_hints.push(*hint),
        Err(rejection) => library.invalid_entries.push(rejection),
    }
}

pub fn validate_library_record_value(
    value: &serde_json::Value,
    path: Option<&Path>,
) -> Result<(), LibraryEntryRejection> {
    validate_library_record(value, path).map(|_| ())
}

fn validate_library_record(
    value: &serde_json::Value,
    path: Option<&Path>,
) -> Result<ValidatedLibraryRecord, LibraryEntryRejection> {
    let inference_path = path.unwrap_or_else(|| Path::new(""));
    let Some(record_type) = infer_record_type(value, inference_path) else {
        return Err(LibraryEntryRejection {
            path: path_label(path),
            record_type: value
                .get("record_type")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            profile_id: profile_id_from_value(value),
            reasons: vec!["unable to infer calibration library record_type".to_string()],
        });
    };

    match record_type {
        "scanner_profile" => serde_json::from_value::<RawScannerProfile>(value.clone())
            .map_err(|err| LibraryEntryRejection {
                path: path_label(path),
                record_type: Some(record_type.to_string()),
                profile_id: profile_id_from_value(value),
                reasons: vec![format!("failed to decode scanner profile: {err}")],
            })
            .and_then(|raw| validate_scanner_profile(raw, path))
            .map(|profile| ValidatedLibraryRecord::Scanner(Box::new(profile))),
        "roll_profile" => serde_json::from_value::<RawRollProfile>(value.clone())
            .map_err(|err| LibraryEntryRejection {
                path: path_label(path),
                record_type: Some(record_type.to_string()),
                profile_id: profile_id_from_value(value),
                reasons: vec![format!("failed to decode roll profile: {err}")],
            })
            .and_then(|raw| validate_roll_profile(raw, path))
            .map(|profile| ValidatedLibraryRecord::Roll(Box::new(profile))),
        "film_hint" => serde_json::from_value::<RawFilmHint>(value.clone())
            .map_err(|err| LibraryEntryRejection {
                path: path_label(path),
                record_type: Some(record_type.to_string()),
                profile_id: profile_id_from_value(value),
                reasons: vec![format!("failed to decode film hint: {err}")],
            })
            .and_then(|raw| validate_film_hint(raw, path))
            .map(|hint| ValidatedLibraryRecord::FilmHint(Box::new(hint))),
        _ => Err(LibraryEntryRejection {
            path: path_label(path),
            record_type: Some(record_type.to_string()),
            profile_id: profile_id_from_value(value),
            reasons: vec![format!("unsupported calibration record_type {record_type}")],
        }),
    }
}

fn infer_record_type(value: &serde_json::Value, path: &Path) -> Option<&'static str> {
    if let Some(record_type) = value.get("record_type").and_then(serde_json::Value::as_str) {
        return match record_type {
            "scanner" | "scanner_profile" => Some("scanner_profile"),
            "roll" | "roll_profile" => Some("roll_profile"),
            "film" | "film_hint" => Some("film_hint"),
            _ => None,
        };
    }

    let path_label = path.to_string_lossy().to_ascii_lowercase();
    if path_label.contains("scanner") {
        return Some("scanner_profile");
    }
    if path_label.contains("roll") {
        return Some("roll_profile");
    }
    if path_label.contains("film") || path_label.contains("hint") {
        return Some("film_hint");
    }
    if value.get("scanner_rgb_to_xyz").is_some() {
        return Some("scanner_profile");
    }
    if value.get("base_color").is_some() || value.get("correction_matrix").is_some() {
        return Some("roll_profile");
    }
    if value.get("similarity_tags").is_some() || value.get("stock").is_some() {
        return Some("film_hint");
    }
    None
}

fn profile_id_from_value(value: &serde_json::Value) -> Option<String> {
    value
        .get("profile_id")
        .or_else(|| value.get("film_id"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn validate_scanner_profile(
    raw: RawScannerProfile,
    path: Option<&Path>,
) -> Result<ScannerProfile, LibraryEntryRejection> {
    let mut rejection_details = Vec::new();
    let schema_version = validate_schema_version(raw.schema_version, &mut rejection_details);
    validate_required_object("scanner", raw.scanner.as_ref(), &mut rejection_details);
    validate_required_object("target", raw.target.as_ref(), &mut rejection_details);
    validate_required_object("reference", raw.reference.as_ref(), &mut rejection_details);
    if let Some(settings) = raw.settings.as_ref() {
        validate_optional_object("settings", Some(settings), &mut rejection_details);
    }
    validate_target_sampling_evidence(
        "target_sampling",
        raw.target_sampling.as_ref(),
        &mut rejection_details,
    );
    let computed_settings_fingerprint = scanner_settings_fingerprint(raw.settings.as_ref());
    if let (Some(expected), Some(computed)) = (
        raw.scanner_settings_fingerprint.as_ref(),
        computed_settings_fingerprint.as_ref(),
    ) {
        if expected != computed {
            rejection_details.push(format!(
                "scanner_settings_fingerprint `{expected}` does not match settings fingerprint `{computed}`"
            ));
        }
    }
    validate_optional_fit(
        "fit",
        raw.fit.as_ref(),
        schema_version,
        &mut rejection_details,
    );
    let signal_domain_fit = raw
        .fit
        .as_ref()
        .or_else(|| raw.color_model.as_ref().map(|model| &model.fit))
        .or_else(|| raw.lut_3d_model.as_ref().map(|model| &model.fit));
    validate_target_patch_signal_domain(
        schema_version,
        signal_domain_fit,
        raw.scanner_linearization.as_ref(),
        raw.target_patch_signal_domain.as_deref(),
        raw.target_patch_linearization_model_id.as_deref(),
        &mut rejection_details,
    );
    if let Some(linearization) = raw.scanner_linearization.as_ref() {
        if let Err(errors) = scanner_linearization::validate_scanner_linearization(linearization) {
            rejection_details.extend(errors);
        }
    }
    let target_patches = optional_target_patches_from_values(
        raw.patches.as_deref(),
        "patches",
        &mut rejection_details,
    );

    let source_space = raw.source_space.unwrap_or_else(|| {
        serde_json::json!({
            "name": "linear scanner RGB",
            "encoding": "linear"
        })
    });
    validate_required_object("source_space", Some(&source_space), &mut rejection_details);
    let confidence = validate_confidence(raw.confidence, &mut rejection_details);

    let whitepoint = match raw.whitepoint {
        Some(whitepoint) => {
            validate_whitepoint(&whitepoint, &mut rejection_details);
            Some(whitepoint)
        }
        None => {
            rejection_details.push("missing whitepoint".to_string());
            None
        }
    };

    let matrix = match raw
        .scanner_rgb_to_xyz
        .or(raw.work_to_xyz)
        .or(raw.fit_matrix)
    {
        Some(matrix) => {
            validate_matrix_values("scanner_rgb_to_xyz", &matrix, &mut rejection_details);
            Some(matrix)
        }
        None => {
            rejection_details
                .push("missing scanner_rgb_to_xyz, work_to_xyz, or fit_matrix".to_string());
            None
        }
    };
    let matrix_condition_number = matrix.as_ref().map(rows_to_matrix3).map(|matrix| {
        validate_matrix_condition("scanner_rgb_to_xyz", &matrix, &mut rejection_details)
    });
    if raw.color_model.is_some() && schema_version.is_some_and(|version| version < 2) {
        rejection_details.push("color_model requires schema_version 2 or newer".to_string());
    }
    if raw.lut_3d_model.is_some() && schema_version.is_some_and(|version| version < 2) {
        rejection_details.push("lut_3d_model requires schema_version 2 or newer".to_string());
    }
    if let (Some(model), Some(matrix)) = (raw.color_model.as_ref(), matrix.as_ref()) {
        rejection_details.extend(validate_root_polynomial_color_model(
            model,
            matrix,
            &target_patches,
        ));
    }
    if let (Some(model), Some(matrix)) = (raw.lut_3d_model.as_ref(), matrix.as_ref()) {
        rejection_details.extend(validate_residual_lut_3d_color_model(
            model,
            matrix,
            raw.color_model.as_ref(),
            &target_patches,
        ));
    }

    if !rejection_details.is_empty() {
        return Err(LibraryEntryRejection {
            path: path_label(path),
            record_type: raw
                .record_type
                .or_else(|| Some("scanner_profile".to_string())),
            profile_id: raw.profile_id,
            reasons: rejection_details,
        });
    }

    Ok(ScannerProfile {
        path: path.map(Path::to_path_buf),
        schema_version: schema_version.expect("schema version validated"),
        profile_id: raw.profile_id,
        source_space,
        scanner: raw.scanner.expect("scanner validated"),
        settings: raw.settings,
        response_curves: raw.response_curves,
        scanner_linearization: raw.scanner_linearization,
        color_model: raw.color_model,
        lut_3d_model: raw.lut_3d_model,
        flare_black_white_diagnostics: raw.flare_black_white_diagnostics,
        target: raw.target.expect("target validated"),
        reference: raw.reference.expect("reference validated"),
        whitepoint: whitepoint.expect("whitepoint validated"),
        scanner_rgb_to_xyz: matrix.expect("matrix validated"),
        confidence: confidence.expect("confidence validated"),
        matrix_condition_number: matrix_condition_number.expect("condition number validated"),
        fit: raw.fit,
        polynomial_fit: raw.polynomial_fit,
        lut_3d: raw.lut_3d,
        lut_3d_fit: raw.lut_3d_fit,
        delta_e00_summary: raw.delta_e00_summary,
        target_patches,
        target_patch_signal_domain: raw.target_patch_signal_domain,
        target_patch_linearization_model_id: raw.target_patch_linearization_model_id,
        target_sampling: raw.target_sampling,
        scanner_settings_fingerprint: computed_settings_fingerprint
            .or(raw.scanner_settings_fingerprint),
        auto_match: raw.auto_match.unwrap_or(false),
    })
}

fn validate_roll_profile(
    raw: RawRollProfile,
    path: Option<&Path>,
) -> Result<RollProfile, LibraryEntryRejection> {
    let mut rejection_details = Vec::new();
    let schema_version = validate_schema_version(raw.schema_version, &mut rejection_details);
    validate_required_object("film", raw.film.as_ref(), &mut rejection_details);
    if let Some(development) = raw.development.as_ref() {
        validate_optional_object("development", Some(development), &mut rejection_details);
    }
    if let Some(metadata) = raw.metadata.as_ref() {
        validate_optional_object("metadata", Some(metadata), &mut rejection_details);
    }
    validate_target_sampling_evidence(
        "target_sampling",
        raw.target_sampling.as_ref(),
        &mut rejection_details,
    );
    validate_scanner_color_model_application(
        raw.scanner_color_model_application.as_ref(),
        raw.correction_matrix.is_some(),
        &mut rejection_details,
    );
    validate_optional_fit(
        "fit",
        raw.fit.as_ref(),
        schema_version,
        &mut rejection_details,
    );
    validate_declared_target_patch_signal_domain(
        schema_version,
        raw.fit.as_ref(),
        raw.target_patch_signal_domain.as_deref(),
        raw.target_patch_linearization_model_id.as_deref(),
        &mut rejection_details,
    );
    let target_patches = optional_target_patches_from_values(
        raw.patches.as_deref(),
        "patches",
        &mut rejection_details,
    );
    if let Some(response) = raw.negative_response.as_ref() {
        if let Err(errors) = density::validate_measured_negative_response(response) {
            rejection_details.extend(errors);
        }
    }
    if let Some(target) = raw.target.as_ref() {
        validate_optional_object("target", Some(target), &mut rejection_details);
    }
    if let Some(reference) = raw.reference.as_ref() {
        validate_optional_object("reference", Some(reference), &mut rejection_details);
    }

    let confidence = validate_confidence(raw.confidence, &mut rejection_details);

    if let Some(base_color) = raw.base_color.as_ref() {
        validate_base_color(base_color, &mut rejection_details);
    }

    let correction_matrix_condition_number = raw.correction_matrix.as_ref().map(|matrix| {
        validate_matrix_values("correction_matrix", matrix, &mut rejection_details);
        let matrix = rows_to_matrix3(matrix);
        validate_matrix_condition("correction_matrix", &matrix, &mut rejection_details)
    });

    if let Some(domain) = raw.correction_domain.as_deref() {
        if domain != "xyz_post_scanner" {
            rejection_details.push(format!(
                "unsupported correction_domain {domain}; expected xyz_post_scanner"
            ));
        }
    }

    if !rejection_details.is_empty() {
        return Err(LibraryEntryRejection {
            path: path_label(path),
            record_type: raw.record_type.or_else(|| Some("roll_profile".to_string())),
            profile_id: raw.profile_id,
            reasons: rejection_details,
        });
    }

    Ok(RollProfile {
        path: path.map(Path::to_path_buf),
        schema_version: schema_version.expect("schema version validated"),
        profile_id: raw.profile_id,
        scanner_profile_id: raw.scanner_profile_id,
        film: raw.film.expect("film validated"),
        development: raw.development,
        metadata: raw.metadata,
        target: raw.target,
        reference: raw.reference,
        base_color: raw.base_color,
        confidence: confidence.expect("confidence validated"),
        correction_matrix: raw.correction_matrix,
        correction_domain: raw.correction_domain.or_else(|| {
            raw.correction_matrix
                .map(|_| "xyz_post_scanner".to_string())
        }),
        correction_matrix_condition_number,
        fit: raw.fit,
        hue_family_residuals: raw.hue_family_residuals,
        delta_e00_summary: raw.delta_e00_summary,
        target_patches,
        target_patch_signal_domain: raw.target_patch_signal_domain,
        target_patch_linearization_model_id: raw.target_patch_linearization_model_id,
        target_sampling: raw.target_sampling,
        scanner_color_model_application: raw.scanner_color_model_application,
        scanner_settings_fingerprint: raw.scanner_settings_fingerprint,
        lut: raw.lut,
        negative_response: raw.negative_response,
    })
}

fn validate_film_hint(
    raw: RawFilmHint,
    path: Option<&Path>,
) -> Result<FilmHint, LibraryEntryRejection> {
    let mut rejection_details = Vec::new();
    let schema_version = validate_schema_version(raw.schema_version, &mut rejection_details);
    if raw.film_id.is_none() && raw.profile_id.is_none() && raw.stock.is_none() {
        rejection_details.push("film hint requires film_id, profile_id, or stock".to_string());
    }
    if let Some(metadata) = raw.metadata.as_ref() {
        validate_optional_object("metadata", Some(metadata), &mut rejection_details);
    }

    if !rejection_details.is_empty() {
        return Err(LibraryEntryRejection {
            path: path_label(path),
            record_type: raw.record_type.or_else(|| Some("film_hint".to_string())),
            profile_id: raw.film_id.or(raw.profile_id),
            reasons: rejection_details,
        });
    }

    Ok(FilmHint {
        path: path.map(Path::to_path_buf),
        schema_version: schema_version.expect("schema version validated"),
        film_id: raw.film_id.or(raw.profile_id),
        stock: raw.stock,
        aliases: raw.aliases,
        similarity_tags: raw.similarity_tags,
        metadata: raw.metadata,
    })
}

fn normalized_film_label(label: &str) -> String {
    label
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn film_value_labels(film: &serde_json::Value) -> Vec<String> {
    let mut labels = Vec::new();
    if let Some(object) = film.as_object() {
        for key in [
            "stock",
            "film_stock",
            "film_id",
            "profile_id",
            "id",
            "name",
            "label",
        ] {
            if let Some(value) = object.get(key).and_then(|value| value.as_str()) {
                labels.push(value.to_string());
            }
        }
        if let (Some(maker), Some(stock)) = (
            object
                .get("manufacturer")
                .or_else(|| object.get("make"))
                .or_else(|| object.get("brand"))
                .and_then(|value| value.as_str()),
            object
                .get("stock")
                .or_else(|| object.get("film_stock"))
                .and_then(|value| value.as_str()),
        ) {
            labels.push(format!("{maker} {stock}"));
        }
    } else if let Some(label) = film.as_str() {
        labels.push(label.to_string());
    }
    labels
}

fn film_hint_labels(hint: &FilmHint) -> Vec<String> {
    let mut labels = Vec::new();
    if let Some(film_id) = &hint.film_id {
        labels.push(film_id.clone());
    }
    if let Some(stock) = &hint.stock {
        labels.push(stock.clone());
    }
    labels.extend(hint.aliases.clone());
    labels
}

fn labels_match_requested(labels: &[String], requested: &str) -> bool {
    let requested = normalized_film_label(requested);
    !requested.is_empty()
        && labels
            .iter()
            .map(|label| normalized_film_label(label))
            .any(|label| label == requested)
}

fn roll_film_match(
    roll: &RollProfile,
    requested_film_stock: Option<&str>,
) -> (Option<bool>, Option<String>) {
    let Some(requested) = requested_film_stock else {
        return (None, None);
    };
    let labels = film_value_labels(&roll.film);
    let matched = labels_match_requested(&labels, requested);
    let reason = if matched {
        format!("roll film metadata matched requested film stock `{requested}`")
    } else {
        format!(
            "roll film metadata did not match requested film stock `{requested}`; labels={}",
            if labels.is_empty() {
                "<none>".to_string()
            } else {
                labels.join(", ")
            }
        )
    };
    (Some(matched), Some(reason))
}

fn film_hint_match(
    hint: &FilmHint,
    requested_film_stock: Option<&str>,
) -> (Option<bool>, Option<String>) {
    let Some(requested) = requested_film_stock else {
        return (None, None);
    };
    let labels = film_hint_labels(hint);
    let matched = labels_match_requested(&labels, requested);
    let reason = if matched {
        format!("film hint matched requested film stock `{requested}`")
    } else {
        format!(
            "film hint did not match requested film stock `{requested}`; labels={}",
            if labels.is_empty() {
                "<none>".to_string()
            } else {
                labels.join(", ")
            }
        )
    };
    (Some(matched), Some(reason))
}

fn film_hint_candidate(
    hint: &FilmHint,
    requested_film_stock: Option<&str>,
) -> FilmHintCandidateDiagnostics {
    let (film_stock_match, film_match_reason) = film_hint_match(hint, requested_film_stock);
    FilmHintCandidateDiagnostics {
        film_id: hint.film_id.clone(),
        path: hint
            .path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        schema_version: Some(hint.schema_version),
        stock: hint.stock.clone(),
        aliases: hint.aliases.clone(),
        similarity_tags: hint.similarity_tags.clone(),
        metadata: hint.metadata.clone(),
        film_stock_match,
        film_match_reason,
        advisory_reason: "film hints are metadata for matching and audit only; they do not apply a correction without a compatible roll profile"
            .to_string(),
    }
}

fn film_stock_diagnostics(
    hints: &[FilmHint],
    rolls: &[RollProfile],
    requested_film_stock: Option<&str>,
) -> Option<FilmStockDiagnostics> {
    let requested = requested_film_stock?;
    let matching_hints = hints
        .iter()
        .filter(|hint| film_hint_match(hint, Some(requested)).0 == Some(true))
        .collect::<Vec<_>>();
    let matched_roll_profiles = rolls
        .iter()
        .filter(|roll| roll_film_match(roll, Some(requested)).0 == Some(true))
        .filter_map(|roll| roll.profile_id.clone())
        .collect::<Vec<_>>();

    if matching_hints.len() == 1 {
        return Some(FilmStockDiagnostics {
            status: "matched".to_string(),
            selection: "explicit_film_stock".to_string(),
            requested_stock: requested.to_string(),
            matched_hint: Some(film_hint_candidate(matching_hints[0], Some(requested))),
            matched_roll_profiles,
            reason: "requested film stock matched one film hint; evidence is recorded and explicit roll selections are checked against it"
                .to_string(),
            rejection_details: Vec::new(),
        });
    }

    if matching_hints.len() > 1 {
        return Some(FilmStockDiagnostics {
            status: "ambiguous".to_string(),
            selection: "explicit_film_stock".to_string(),
            requested_stock: requested.to_string(),
            matched_hint: None,
            matched_roll_profiles,
            reason: format!(
                "requested film stock matched {} film hints; select a roll profile explicitly before applying roll correction",
                matching_hints.len()
            ),
            rejection_details: vec![
                "film stock matched multiple film hints in the calibration library".to_string(),
            ],
        });
    }

    if !matched_roll_profiles.is_empty() {
        return Some(FilmStockDiagnostics {
            status: "matched_roll_metadata".to_string(),
            selection: "explicit_film_stock".to_string(),
            requested_stock: requested.to_string(),
            matched_hint: None,
            matched_roll_profiles,
            reason: "requested film stock matched roll metadata but no film hint record"
                .to_string(),
            rejection_details: Vec::new(),
        });
    }

    Some(FilmStockDiagnostics {
        status: "unmatched".to_string(),
        selection: "explicit_film_stock".to_string(),
        requested_stock: requested.to_string(),
        matched_hint: None,
        matched_roll_profiles,
        reason:
            "requested film stock was not found in calibration library film hints or roll metadata"
                .to_string(),
        rejection_details: vec![format!(
            "requested film stock `{requested}` was not found in calibration library film evidence"
        )],
    })
}

fn select_scanner_profile(
    scanners: &[ScannerProfile],
    requested_id: Option<&str>,
) -> (
    Option<ScannerProfile>,
    Option<ScannerProfileDiagnostics>,
    bool,
) {
    if let Some(requested_id) = requested_id {
        if let Some(scanner) = scanners
            .iter()
            .find(|scanner| scanner.profile_id.as_deref() == Some(requested_id))
        {
            return (
                Some(scanner.clone()),
                Some(scanner_diagnostics(
                    scanner,
                    "applied",
                    "explicit",
                    "explicit scanner profile selected",
                    Vec::new(),
                )),
                false,
            );
        }
        return (
            None,
            Some(ScannerProfileDiagnostics {
                status: "rejected".to_string(),
                selection: "explicit".to_string(),
                path: None,
                schema_version: None,
                profile_id: Some(requested_id.to_string()),
                scanner: None,
                settings: None,
                confidence: None,
                matrix_condition_number: None,
                whitepoint: None,
                response_curves: None,
                scanner_linearization: None,
                flare_black_white_diagnostics: None,
                transform_type: None,
                polynomial_fit: None,
                lut_3d: None,
                lut_3d_fit: None,
                delta_e00_summary: None,
                fit: None,
                target_patch_signal_domain: None,
                target_patch_linearization_model_id: None,
                target_sampling: None,
                color_model: None,
                lut_3d_model: None,
                scanner_settings_fingerprint: None,
                reason: format!("requested scanner profile `{requested_id}` was not found"),
                rejection_details: vec![format!(
                    "requested scanner profile `{requested_id}` was not found"
                )],
            }),
            true,
        );
    }

    let auto_candidates = scanners
        .iter()
        .filter(|scanner| scanner.auto_match && scanner.confidence >= MIN_AUTO_MATCH_CONFIDENCE)
        .collect::<Vec<_>>();
    if auto_candidates.len() == 1 {
        let scanner = auto_candidates[0];
        return (
            Some(scanner.clone()),
            Some(scanner_diagnostics(
                scanner,
                "applied",
                "auto_matched",
                "single high-confidence scanner profile opted into auto-match",
                Vec::new(),
            )),
            false,
        );
    }

    if let Some(best) = scanners.iter().max_by(compare_scanner_confidence) {
        let reason = if auto_candidates.len() > 1 {
            "multiple high-confidence scanner profiles opted into auto-match; select one explicitly"
                .to_string()
        } else if best.auto_match {
            format!(
                "best scanner auto-match confidence {:.3} is below required {:.3}; advisory only",
                best.confidence, MIN_AUTO_MATCH_CONFIDENCE
            )
        } else {
            "scanner profiles are available but none opted into auto-match; select one explicitly"
                .to_string()
        };
        return (
            None,
            Some(scanner_diagnostics(
                best,
                "advisory",
                "auto_match_advisory",
                &reason,
                Vec::new(),
            )),
            false,
        );
    }

    (None, None, false)
}

fn select_roll_profile(
    rolls: &[RollProfile],
    requested_id: Option<&str>,
    scanner: Option<&ScannerProfile>,
    requested_film_stock: Option<&str>,
    observed_base_color: Option<[f64; 3]>,
) -> (Option<RollProfile>, Option<RollProfileDiagnostics>, bool) {
    let Some(requested_id) = requested_id else {
        return (None, None, false);
    };

    let Some(roll) = rolls
        .iter()
        .find(|roll| roll.profile_id.as_deref() == Some(requested_id))
    else {
        return (
            None,
            Some(RollProfileDiagnostics {
                status: "rejected".to_string(),
                selection: "explicit".to_string(),
                path: None,
                schema_version: None,
                profile_id: Some(requested_id.to_string()),
                scanner_profile_id: None,
                film: None,
                development: None,
                metadata: None,
                base_color: None,
                observed_base_color,
                base_color_delta: None,
                confidence: None,
                correction_domain: None,
                correction_matrix_condition_number: None,
                correction_transform_type: None,
                hue_family_residuals: None,
                delta_e00_summary: None,
                fit: None,
                target_patch_signal_domain: None,
                target_patch_linearization_model_id: None,
                target_sampling: None,
                scanner_color_model_application: None,
                scanner_settings_fingerprint: None,
                correction_applied: false,
                lut_present: false,
                reason: format!("requested roll profile `{requested_id}` was not found"),
                rejection_details: vec![format!(
                    "requested roll profile `{requested_id}` was not found"
                )],
            }),
            true,
        );
    };

    let mut rejection_details = Vec::new();
    if let (Some(false), Some(reason)) = roll_film_match(roll, requested_film_stock) {
        rejection_details.push(reason);
    }
    let requires_exact_scanner =
        roll.correction_matrix.is_some() || roll.negative_response.is_some();
    if requires_exact_scanner && scanner.is_none() {
        rejection_details.push(
            "roll correction or negative-response reconstruction requires an applied scanner profile"
                .to_string(),
        );
    }
    if requires_exact_scanner && roll.scanner_profile_id.is_none() {
        rejection_details.push(
            "roll correction or negative-response reconstruction requires scanner_profile_id matching the selected scanner profile".to_string(),
        );
    }
    if requires_exact_scanner && scanner.is_some_and(|scanner| scanner.profile_id.is_none()) {
        rejection_details.push(
            "roll correction or negative-response reconstruction requires the selected scanner profile to have a profile_id".to_string(),
        );
    }
    if let (Some(expected), Some(scanner)) = (&roll.scanner_profile_id, scanner) {
        if scanner.profile_id.as_deref() != Some(expected.as_str()) {
            rejection_details.push(format!(
                "roll profile expects scanner_profile_id `{expected}` but selected scanner is `{}`",
                scanner.profile_id.as_deref().unwrap_or("<unnamed>")
            ));
        }
    }
    if let Some(scanner) = scanner {
        let (expected_transform, expected_model_id) =
            if let Some(model) = scanner.lut_3d_model.as_ref() {
                (
                    format!("residual_lut_3d_grid_{}", model.grid_size),
                    Some(model.model_id.as_str()),
                )
            } else if let Some(model) = scanner.color_model.as_ref() {
                (
                    format!("root_polynomial_degree_{}", model.degree),
                    Some(model.model_id.as_str()),
                )
            } else {
                ("matrix".to_string(), None)
            };
        match roll.scanner_color_model_application.as_ref() {
            Some(application) => {
                if application.transform != expected_transform {
                    rejection_details.push(format!(
                        "roll scanner_color_model_application transform `{}` does not match selected scanner transform `{expected_transform}`",
                        application.transform
                    ));
                }
                if application.model_id.as_deref() != expected_model_id {
                    rejection_details.push(format!(
                        "roll scanner_color_model_application model_id `{}` does not match selected scanner color model `{}`",
                        application.model_id.as_deref().unwrap_or("<none>"),
                        expected_model_id.unwrap_or("<none>")
                    ));
                }
            }
            None if roll.correction_matrix.is_some()
                && (scanner.color_model.is_some() || scanner.lut_3d_model.is_some()) =>
            {
                rejection_details.push(
                    "roll correction fitted for a nonlinear scanner profile requires scanner_color_model_application evidence"
                        .to_string(),
                );
            }
            None => {}
        }
    }
    if roll.schema_version >= 2 && roll.fit.is_some() {
        if let Some(scanner) = scanner {
            let (expected_domain, expected_model_id) = scanner
                .scanner_linearization
                .as_ref()
                .map_or(("normalized_scanner_signal", None), |linearization| {
                    (
                        "scanner_linearized_transmittance",
                        Some(linearization.model_id.as_str()),
                    )
                });
            if roll.target_patch_signal_domain.as_deref() != Some(expected_domain) {
                rejection_details.push(format!(
                    "roll target_patch_signal_domain `{}` does not match selected scanner runtime domain `{expected_domain}`",
                    roll.target_patch_signal_domain.as_deref().unwrap_or("<missing>")
                ));
            }
            if roll.target_patch_linearization_model_id.as_deref() != expected_model_id {
                rejection_details.push(format!(
                    "roll target_patch_linearization_model_id `{}` does not match selected scanner linearization model `{}`",
                    roll.target_patch_linearization_model_id
                        .as_deref()
                        .unwrap_or("<none>"),
                    expected_model_id.unwrap_or("<none>")
                ));
            }
        }
    }
    match (
        roll.scanner_settings_fingerprint.as_ref(),
        scanner.and_then(|scanner| scanner.scanner_settings_fingerprint.as_ref()),
        requires_exact_scanner,
    ) {
        (Some(expected), Some(actual), _) if actual != expected => rejection_details.push(format!(
            "roll profile scanner_settings_fingerprint `{expected}` does not match selected scanner fingerprint `{actual}`"
        )),
        (Some(_), None, _) => rejection_details.push(
            "roll profile declares scanner_settings_fingerprint but selected scanner has no settings fingerprint"
                .to_string(),
        ),
        (None, Some(_), true) => rejection_details.push(
            "roll correction or negative-response reconstruction requires scanner_settings_fingerprint matching the selected scanner settings"
                .to_string(),
        ),
        (None, None, true) => rejection_details.push(
            "roll correction or negative-response reconstruction requires scanner_settings_fingerprint on both the roll and selected scanner profile"
                .to_string(),
        ),
        _ => {}
    }

    if !rejection_details.is_empty() {
        return (
            None,
            Some(roll_diagnostics(
                roll,
                observed_base_color,
                "rejected",
                "explicit",
                "explicit roll profile could not be applied with the selected scanner",
                false,
                rejection_details,
            )),
            true,
        );
    }

    let correction_applied = roll.correction_matrix.is_some() && scanner.is_some();
    let reason = if correction_applied && roll.negative_response.is_some() {
        "explicit roll color correction and held-out-validated negative-response reconstruction selected"
            .to_string()
    } else if correction_applied {
        "explicit roll correction selected".to_string()
    } else if roll.negative_response.is_some() {
        "explicit held-out-validated negative-response reconstruction selected; scanner color prior remains image-adapted"
            .to_string()
    } else if roll.lut.is_some() {
        "explicit roll LUT recorded but v1 rendering only applies matrix corrections; metadata recorded without LUT correction"
            .to_string()
    } else {
        "explicit base-only roll profile selected; metadata recorded without matrix correction"
            .to_string()
    };

    (
        Some(roll.clone()),
        Some(roll_diagnostics(
            roll,
            observed_base_color,
            "applied",
            "explicit",
            &reason,
            correction_applied,
            Vec::new(),
        )),
        false,
    )
}

fn build_library_profile(
    scanner: &ScannerProfile,
    roll: Option<&RollProfile>,
    _observed_base_color: Option<[f64; 3]>,
) -> CalibrationProfile {
    let scanner_matrix = rows_to_matrix3(&scanner.scanner_rgb_to_xyz);
    let roll_correction = roll.and_then(|roll| roll.correction_matrix.as_ref());
    let (work_to_xyz_matrix, whitepoint, roll_correction_applied) =
        if let Some(correction_rows) = roll_correction {
            let correction = rows_to_matrix3(correction_rows);
            let combined = correction * scanner_matrix;
            let corrected_white = correction
                * Vector3::new(
                    scanner.whitepoint[0],
                    scanner.whitepoint[1],
                    scanner.whitepoint[2],
                );
            let corrected_whitepoint = [
                corrected_white[0].max(1e-6),
                corrected_white[1].max(1e-6),
                corrected_white[2].max(1e-6),
            ];
            (combined, corrected_whitepoint, true)
        } else {
            (scanner_matrix, scanner.whitepoint, false)
        };
    let work_to_xyz = matrix3_to_rows(&work_to_xyz_matrix);
    let confidence = roll
        .map(|roll| scanner.confidence.min(roll.confidence))
        .unwrap_or(scanner.confidence);
    let fit = roll
        .and_then(|roll| roll.fit.clone())
        .or_else(|| scanner.fit.clone());
    let target_patches = roll
        .filter(|_| roll_correction_applied)
        .map(|roll| roll.target_patches.clone())
        .filter(|patches| !patches.is_empty())
        .unwrap_or_else(|| scanner.target_patches.clone());
    let matrix_condition_number = matrix_condition_number(&work_to_xyz_matrix);
    let roll_profile_id = roll.and_then(|roll| roll.profile_id.clone());
    let application_mode = if roll_correction_applied {
        CalibrationApplicationMode::DirectProfile
    } else {
        CalibrationApplicationMode::ScannerConstrainedImageAdaptation
    };
    let schema_version = roll
        .map(|roll| scanner.schema_version.min(roll.schema_version))
        .unwrap_or(scanner.schema_version);

    CalibrationProfile {
        schema_version,
        profile_id: Some(match (&scanner.profile_id, &roll_profile_id) {
            (Some(scanner_id), Some(roll_id)) => format!("{scanner_id}+{roll_id}"),
            (Some(scanner_id), None) => scanner_id.clone(),
            (None, Some(roll_id)) => roll_id.clone(),
            (None, None) => "calibration-library-profile".to_string(),
        }),
        source_space: scanner.source_space.clone(),
        scanner: serde_json::json!({
            "profile_id": scanner.profile_id,
            "scanner": scanner.scanner,
            "settings": scanner.settings,
            "settings_fingerprint": scanner.scanner_settings_fingerprint,
            "response_curves": scanner.response_curves,
            "scanner_linearization": scanner.scanner_linearization,
            "flare_black_white_diagnostics": scanner.flare_black_white_diagnostics,
            "polynomial_fit_recorded": scanner.polynomial_fit.is_some(),
            "lut_3d_recorded": scanner.lut_3d.is_some(),
            "validated_lut_3d_model": scanner.lut_3d_model,
            "delta_e00_summary": scanner.delta_e00_summary,
            "fit": scanner.fit,
        }),
        film: roll
            .map(|roll| roll.film.clone())
            .unwrap_or_else(|| serde_json::json!({})),
        target: serde_json::json!({
            "scanner_target": scanner.target,
            "roll_target": roll.and_then(|roll| roll.target.clone()),
        }),
        reference: serde_json::json!({
            "scanner_reference": scanner.reference,
            "roll_reference": roll.and_then(|roll| roll.reference.clone()),
            "scanner_delta_e00_summary": scanner.delta_e00_summary,
            "roll_delta_e00_summary": roll.and_then(|roll| roll.delta_e00_summary.clone()),
        }),
        gamut_limits: None,
        whitepoint,
        work_to_xyz,
        confidence,
        matrix_condition_number,
        fit,
        target_patches,
        application_mode,
        scanner_profile_id: scanner.profile_id.clone(),
        roll_profile_id,
        roll_correction_applied,
        scanner_prior_work_to_xyz: roll_correction_applied.then_some(scanner.scanner_rgb_to_xyz),
        scanner_prior_confidence: roll_correction_applied.then_some(scanner.confidence),
        scanner_prior_fit: if roll_correction_applied {
            scanner.fit.clone()
        } else {
            None
        },
        negative_response: roll.and_then(|roll| roll.negative_response.clone()),
        scanner_linearization: scanner.scanner_linearization.clone(),
        color_model: scanner.color_model.clone(),
        lut_3d_model: scanner.lut_3d_model.clone(),
        color_model_post_xyz: roll_correction_applied
            .then(|| *roll_correction.expect("roll correction is present when applied")),
    }
}

fn scanner_diagnostics(
    scanner: &ScannerProfile,
    status: &str,
    selection: &str,
    reason: &str,
    rejection_details: Vec<String>,
) -> ScannerProfileDiagnostics {
    ScannerProfileDiagnostics {
        status: status.to_string(),
        selection: selection.to_string(),
        path: scanner
            .path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        schema_version: Some(scanner.schema_version),
        profile_id: scanner.profile_id.clone(),
        scanner: Some(scanner.scanner.clone()),
        settings: scanner.settings.clone(),
        confidence: Some(scanner.confidence),
        matrix_condition_number: Some(scanner.matrix_condition_number),
        whitepoint: Some(scanner.whitepoint),
        response_curves: scanner.response_curves.clone(),
        scanner_linearization: scanner.scanner_linearization.clone(),
        flare_black_white_diagnostics: scanner.flare_black_white_diagnostics.clone(),
        transform_type: Some(
            if let Some(model) = scanner.lut_3d_model.as_ref() {
                match (scanner.color_model.is_some(), model.grid_size) {
                    (true, 7) => "matrix_with_validated_root_polynomial_and_7x7x7_residual_lut",
                    (true, _) => "matrix_with_validated_root_polynomial_and_5x5x5_residual_lut",
                    (false, 7) => "matrix_with_validated_7x7x7_residual_lut",
                    (false, _) => "matrix_with_validated_5x5x5_residual_lut",
                }
            } else if let Some(model) = scanner.color_model.as_ref() {
                if model.degree == 3 {
                    "matrix_with_validated_cubic_root_polynomial"
                } else {
                    "matrix_with_validated_quadratic_root_polynomial"
                }
            } else if scanner.lut_3d.is_some() {
                "matrix_with_recorded_3d_lut_candidate"
            } else if scanner.polynomial_fit.is_some() {
                "matrix_with_recorded_polynomial_candidate"
            } else {
                "matrix"
            }
            .to_string(),
        ),
        polynomial_fit: scanner.polynomial_fit.clone(),
        lut_3d: scanner.lut_3d.clone(),
        lut_3d_fit: scanner.lut_3d_fit.clone(),
        delta_e00_summary: scanner.delta_e00_summary.clone(),
        fit: scanner.fit.clone(),
        target_patch_signal_domain: scanner.target_patch_signal_domain.clone(),
        target_patch_linearization_model_id: scanner.target_patch_linearization_model_id.clone(),
        target_sampling: scanner.target_sampling.clone(),
        color_model: scanner.color_model.clone(),
        lut_3d_model: scanner.lut_3d_model.clone(),
        scanner_settings_fingerprint: scanner.scanner_settings_fingerprint.clone(),
        reason: reason.to_string(),
        rejection_details,
    }
}

fn roll_diagnostics(
    roll: &RollProfile,
    observed_base_color: Option<[f64; 3]>,
    status: &str,
    selection: &str,
    reason: &str,
    correction_applied: bool,
    rejection_details: Vec<String>,
) -> RollProfileDiagnostics {
    RollProfileDiagnostics {
        status: status.to_string(),
        selection: selection.to_string(),
        path: roll
            .path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        schema_version: Some(roll.schema_version),
        profile_id: roll.profile_id.clone(),
        scanner_profile_id: roll.scanner_profile_id.clone(),
        film: Some(roll.film.clone()),
        development: roll.development.clone(),
        metadata: roll.metadata.clone(),
        base_color: roll.base_color,
        observed_base_color,
        base_color_delta: roll.base_color.and_then(|base_color| {
            observed_base_color.and_then(|observed| base_color_delta(base_color, observed))
        }),
        confidence: Some(roll.confidence),
        correction_domain: roll.correction_domain.clone(),
        correction_matrix_condition_number: roll.correction_matrix_condition_number,
        correction_transform_type: Some(
            if roll.lut.is_some() && roll.correction_matrix.is_some() {
                "matrix_with_recorded_lut_candidate"
            } else if roll.lut.is_some() {
                "recorded_lut_not_applied"
            } else if roll.correction_matrix.is_some() {
                "matrix"
            } else {
                "metadata_only"
            }
            .to_string(),
        ),
        hue_family_residuals: roll.hue_family_residuals.clone(),
        delta_e00_summary: roll.delta_e00_summary.clone(),
        fit: roll.fit.clone(),
        target_patch_signal_domain: roll.target_patch_signal_domain.clone(),
        target_patch_linearization_model_id: roll.target_patch_linearization_model_id.clone(),
        target_sampling: roll.target_sampling.clone(),
        scanner_color_model_application: roll.scanner_color_model_application.clone(),
        scanner_settings_fingerprint: roll.scanner_settings_fingerprint.clone(),
        correction_applied,
        lut_present: roll.lut.is_some(),
        reason: reason.to_string(),
        rejection_details,
    }
}

fn nearest_roll_candidates(
    rolls: &[RollProfile],
    observed_base_color: Option<[f64; 3]>,
    requested_film_stock: Option<&str>,
) -> Vec<RollCandidateDiagnostics> {
    let mut candidates = rolls
        .iter()
        .map(|roll| {
            let (film_stock_match, film_match_reason) = roll_film_match(roll, requested_film_stock);
            RollCandidateDiagnostics {
                profile_id: roll.profile_id.clone(),
                path: roll
                    .path
                    .as_ref()
                    .map(|path| path.to_string_lossy().to_string()),
                scanner_profile_id: roll.scanner_profile_id.clone(),
                film: Some(roll.film.clone()),
                film_stock_match,
                film_match_reason,
                confidence: roll.confidence,
                base_color: roll.base_color,
                base_color_delta: roll.base_color.and_then(|base_color| {
                    observed_base_color.and_then(|observed| base_color_delta(base_color, observed))
                }),
                correction_available: roll.correction_matrix.is_some(),
                advisory_reason:
                    "roll profiles are advisory unless selected explicitly with --roll-profile"
                        .to_string(),
            }
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|a, b| match (a.film_stock_match, b.film_stock_match) {
        (Some(true), Some(false)) => Ordering::Less,
        (Some(false), Some(true)) => Ordering::Greater,
        _ => match (a.base_color_delta, b.base_color_delta) {
            (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => b
                .confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(Ordering::Equal),
        },
    });
    candidates.truncate(MAX_REPORTED_CANDIDATES);
    candidates
}

fn film_hint_candidates(
    hints: &[FilmHint],
    requested_film_stock: Option<&str>,
) -> Vec<FilmHintCandidateDiagnostics> {
    let mut candidates = hints
        .iter()
        .map(|hint| film_hint_candidate(hint, requested_film_stock))
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| match (a.film_stock_match, b.film_stock_match) {
        (Some(true), Some(false)) => Ordering::Less,
        (Some(false), Some(true)) => Ordering::Greater,
        _ => a
            .stock
            .cmp(&b.stock)
            .then_with(|| a.film_id.cmp(&b.film_id)),
    });
    candidates.truncate(MAX_REPORTED_CANDIDATES);
    candidates
}

fn selected_rejection_details(diagnostics: &CalibrationDiagnostics) -> Vec<String> {
    let mut details = Vec::new();
    if let Some(scanner) = diagnostics.scanner_profile.as_ref() {
        details.extend(scanner.rejection_details.clone());
    }
    if let Some(roll) = diagnostics.roll_profile.as_ref() {
        details.extend(roll.rejection_details.clone());
    }
    details
}

fn lab_inverse_f(value: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if value > DELTA {
        value * value * value
    } else {
        3.0 * DELTA * DELTA * (value - 4.0 / 29.0)
    }
}

fn target_patch_from_object(idx: usize, patch: &serde_json::Value) -> Result<TargetPatch, String> {
    target_patch_from_named_object("patches", idx, patch)
}

fn target_patch_from_named_object(
    field: &str,
    idx: usize,
    patch: &serde_json::Value,
) -> Result<TargetPatch, String> {
    let object = patch
        .as_object()
        .ok_or_else(|| format!("{field}[{idx}] must be a JSON object"))?;
    let patch_id = object
        .get("id")
        .or_else(|| object.get("patch_id"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let scanner_xy = object
        .get("scanner_xy")
        .map(|value| pair_from_value(value, &format!("{field}[{idx}].scanner_xy")))
        .transpose()?;
    let source_rgb = first_triplet(
        object,
        &["rgb", "scanner_rgb", "source_rgb", "measured_rgb"],
        &format!("{field}[{idx}] source RGB"),
    )?;
    let reference_xyz = if let Ok(xyz) = first_triplet(
        object,
        &["xyz", "reference_xyz", "target_xyz"],
        &format!("{field}[{idx}] reference XYZ"),
    ) {
        xyz
    } else {
        let lab = first_triplet(
            object,
            &["lab", "reference_lab", "target_lab"],
            &format!("{field}[{idx}] reference Lab"),
        )?;
        lab_d50_to_xyz(lab)
    };

    Ok(TargetPatch {
        patch_id,
        scanner_xy,
        source_rgb,
        reference_xyz,
    })
}

fn first_triplet(
    object: &serde_json::Map<String, serde_json::Value>,
    fields: &[&str],
    label: &str,
) -> Result<[f64; 3], String> {
    for field in fields {
        if let Some(value) = object.get(*field) {
            return triplet_from_value(value, field);
        }
    }
    Err(format!("{label} requires one of: {}", fields.join(", ")))
}

fn triplet_from_value(value: &serde_json::Value, label: &str) -> Result<[f64; 3], String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array of three numbers"))?;
    if values.len() != 3 {
        return Err(format!("{label} must contain exactly three numbers"));
    }
    let mut out = [0.0f64; 3];
    for (idx, value) in values.iter().enumerate() {
        out[idx] = value
            .as_f64()
            .ok_or_else(|| format!("{label}[{idx}] must be a finite number"))?;
        if !out[idx].is_finite() {
            return Err(format!("{label}[{idx}] must be finite"));
        }
    }
    Ok(out)
}

fn pair_from_value(value: &serde_json::Value, label: &str) -> Result<[f64; 2], String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array of two numbers"))?;
    if values.len() != 2 {
        return Err(format!("{label} must contain exactly two numbers"));
    }
    let mut out = [0.0f64; 2];
    for (idx, value) in values.iter().enumerate() {
        out[idx] = value
            .as_f64()
            .ok_or_else(|| format!("{label}[{idx}] must be a finite number"))?;
        if !out[idx].is_finite() {
            return Err(format!("{label}[{idx}] must be finite"));
        }
    }
    Ok(out)
}

fn validate_target_patch(idx: usize, patch: &TargetPatch) -> Result<(), String> {
    if patch.scanner_xy.is_some_and(|coordinates| {
        coordinates
            .iter()
            .any(|value| !value.is_finite() || !(-1.0..=1.0).contains(value))
    }) {
        return Err(format!(
            "patch {idx} scanner_xy values must be finite normalized scanner-frame coordinates in [-1,1]"
        ));
    }
    if patch
        .source_rgb
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(format!(
            "patch {idx} source RGB values must be finite and non-negative"
        ));
    }
    if patch.source_rgb.iter().all(|value| *value <= 1e-12) {
        return Err(format!("patch {idx} source RGB must not be all zero"));
    }
    if patch
        .reference_xyz
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(format!(
            "patch {idx} reference XYZ values must be finite and non-negative"
        ));
    }
    Ok(())
}

fn estimate_whitepoint_from_patches(patches: &[TargetPatch]) -> Result<[f64; 3], String> {
    let reference = patches
        .iter()
        .map(|patch| patch.reference_xyz)
        .max_by(|a, b| a[1].partial_cmp(&b[1]).unwrap_or(Ordering::Equal))
        .ok_or_else(|| "cannot estimate whitepoint without target patches".to_string())?;
    if reference[1] <= 1e-9 {
        return Err("cannot estimate whitepoint from zero-luminance target patches".to_string());
    }
    let whitepoint = [
        reference[0] / reference[1],
        1.0,
        reference[2] / reference[1],
    ];
    let mut rejections = Vec::new();
    validate_whitepoint(&whitepoint, &mut rejections);
    if !rejections.is_empty() {
        return Err(format!(
            "estimated whitepoint from brightest target patch was invalid: {}",
            rejections.join("; ")
        ));
    }
    Ok(whitepoint)
}

fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(value) => value.to_string(),
        serde_json::Value::Number(value) => value.to_string(),
        serde_json::Value::String(value) => serde_json::to_string(value).unwrap_or_default(),
        serde_json::Value::Array(values) => {
            let values = values.iter().map(canonical_json).collect::<Vec<_>>();
            format!("[{}]", values.join(","))
        }
        serde_json::Value::Object(values) => {
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_by(|(a, _), (b, _)| a.cmp(b));
            let entries = entries
                .into_iter()
                .map(|(key, value)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).unwrap_or_default(),
                        canonical_json(value)
                    )
                })
                .collect::<Vec<_>>();
            format!("{{{}}}", entries.join(","))
        }
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn validate_schema_version(
    schema_version: Option<u32>,
    rejection_details: &mut Vec<String>,
) -> Option<u32> {
    match schema_version {
        Some(version)
            if (MIN_SUPPORTED_CALIBRATION_SCHEMA_VERSION..=CALIBRATION_PROFILE_SCHEMA_VERSION)
                .contains(&version) =>
        {
            Some(version)
        }
        Some(version) => {
            rejection_details.push(format!(
                "unsupported schema_version {version}; expected {MIN_SUPPORTED_CALIBRATION_SCHEMA_VERSION}..={CALIBRATION_PROFILE_SCHEMA_VERSION}"
            ));
            Some(version)
        }
        None => {
            rejection_details.push("missing schema_version".to_string());
            None
        }
    }
}

fn validate_required_object(
    field: &str,
    value: Option<&serde_json::Value>,
    rejection_details: &mut Vec<String>,
) {
    match value {
        Some(value) if value.is_object() => {}
        Some(_) => rejection_details.push(format!("{field} metadata must be a JSON object")),
        None => rejection_details.push(format!("missing {field} metadata")),
    }
}

fn validate_optional_object(
    field: &str,
    value: Option<&serde_json::Value>,
    rejection_details: &mut Vec<String>,
) {
    if let Some(value) = value {
        if !value.is_object() {
            rejection_details.push(format!("{field} metadata must be a JSON object"));
        }
    }
}

fn validate_scanner_color_model_application(
    application: Option<&ScannerColorModelApplicationDiagnostics>,
    correction_matrix_present: bool,
    rejection_details: &mut Vec<String>,
) {
    let Some(application) = application else {
        return;
    };
    if !correction_matrix_present {
        rejection_details.push(
            "scanner_color_model_application requires an XYZ-domain correction_matrix".to_string(),
        );
    }
    let requires_model_id = match application.transform.as_str() {
        "matrix" => false,
        "root_polynomial_degree_2"
        | "root_polynomial_degree_3"
        | "residual_lut_3d_grid_5"
        | "residual_lut_3d_grid_7" => true,
        transform => {
            rejection_details.push(format!(
                "unsupported scanner_color_model_application.transform `{transform}`"
            ));
            false
        }
    };
    match (requires_model_id, application.model_id.as_deref()) {
        (false, Some(_)) if application.transform == "matrix" => rejection_details
            .push("matrix scanner_color_model_application must not declare model_id".to_string()),
        (true, None) => rejection_details
            .push("nonlinear scanner_color_model_application requires model_id".to_string()),
        (_, Some(model_id)) if model_id.trim().is_empty() => rejection_details
            .push("scanner_color_model_application.model_id must not be empty".to_string()),
        _ => {}
    }
    if application.output_domain != ROOT_POLYNOMIAL_OUTPUT_SPACE {
        rejection_details.push(format!(
            "scanner_color_model_application.output_domain `{}` is unsupported; expected {ROOT_POLYNOMIAL_OUTPUT_SPACE}",
            application.output_domain
        ));
    }
    if !application.numerical_zero_tolerance.is_finite()
        || !(0.0..=1e-9).contains(&application.numerical_zero_tolerance)
    {
        rejection_details.push(
            "scanner_color_model_application.numerical_zero_tolerance must be finite and in [0, 1e-9]"
                .to_string(),
        );
    }
}

fn validate_target_sampling_evidence(
    field: &str,
    value: Option<&serde_json::Value>,
    rejection_details: &mut Vec<String>,
) {
    let Some(value) = value else {
        return;
    };
    let Some(object) = value.as_object() else {
        rejection_details.push(format!("{field} must be a JSON object"));
        return;
    };
    if object.get("status").and_then(serde_json::Value::as_str) != Some("accepted") {
        rejection_details.push(format!(
            "{field}.status must be accepted; rejected acquisition evidence cannot support a calibration"
        ));
    }
    if target_sampling_usize(object, "schema_version", field, rejection_details) != Some(1) {
        rejection_details.push(format!("{field}.schema_version must equal 1"));
    }
    let reference_patch_count =
        target_sampling_usize(object, "reference_patch_count", field, rejection_details);
    if reference_patch_count.is_some_and(|count| count < MIN_TARGET_PATCHES) {
        rejection_details.push(format!(
            "{field}.reference_patch_count is below required minimum {MIN_TARGET_PATCHES}"
        ));
    }
    let reference_patch_sha256 = object
        .get("reference_patch_sha256")
        .and_then(serde_json::Value::as_str);
    if reference_patch_sha256.is_none_or(|hash| !valid_sha256(hash)) {
        rejection_details.push(format!(
            "{field}.reference_patch_sha256 must be 64 hexadecimal characters"
        ));
    }
    match object
        .get("reference_patches")
        .cloned()
        .map(serde_json::from_value::<Vec<crate::calibration_target::TargetReferencePatch>>)
    {
        Some(Ok(reference_patches)) => {
            if reference_patch_count != Some(reference_patches.len()) {
                rejection_details.push(format!(
                    "{field}.reference_patches length must equal reference_patch_count"
                ));
            }
            let mut ids = std::collections::HashSet::<String>::new();
            let mut positions = std::collections::HashSet::<(usize, usize)>::new();
            for (index, patch) in reference_patches.iter().enumerate() {
                if patch.id.trim().is_empty() || !ids.insert(patch.id.trim().to_ascii_lowercase()) {
                    rejection_details.push(format!(
                        "{field}.reference_patches[{index}].id must be non-empty and unique"
                    ));
                }
                if !positions.insert((patch.row, patch.column)) {
                    rejection_details.push(format!(
                        "{field}.reference_patches[{index}] duplicates a grid position"
                    ));
                }
                let values = match (patch.reference_xyz, patch.reference_lab) {
                    (Some(values), None) => Some(("reference_xyz", values)),
                    (None, Some(values)) => Some(("reference_lab", values)),
                    _ => None,
                };
                let Some((value_field, values)) = values else {
                    rejection_details.push(format!(
                        "{field}.reference_patches[{index}] must declare exactly one of reference_xyz/reference_lab"
                    ));
                    continue;
                };
                if values.iter().any(|value| !value.is_finite())
                    || (value_field == "reference_xyz" && values.iter().any(|value| *value < 0.0))
                {
                    rejection_details.push(format!(
                        "{field}.reference_patches[{index}].{value_field} is invalid"
                    ));
                }
            }
            let actual = crate::calibration_target::sha256_reference_patches(&reference_patches);
            if reference_patch_sha256.is_none_or(|expected| !expected.eq_ignore_ascii_case(&actual))
            {
                rejection_details.push(format!(
                    "{field}.reference_patch_sha256 does not match reference_patches"
                ));
            }
        }
        Some(Err(_)) => rejection_details.push(format!(
            "{field}.reference_patches must contain valid reference patch objects"
        )),
        None => rejection_details.push(format!("{field}.reference_patches is required")),
    }
    let reference_review = object
        .get("reference_review")
        .and_then(serde_json::Value::as_object);
    if reference_review
        .and_then(|review| review.get("status"))
        .and_then(serde_json::Value::as_str)
        != Some("approved")
    {
        rejection_details.push(format!("{field}.reference_review.status must be approved"));
    }
    if reference_review
        .and_then(|review| review.get("approved"))
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        rejection_details.push(format!("{field}.reference_review.approved must be true"));
    }
    if reference_review
        .and_then(|review| review.get("reference_patch_sha256_matched"))
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        rejection_details.push(format!(
            "{field}.reference_review.reference_patch_sha256_matched must be true"
        ));
    }
    match reference_review
        .and_then(|review| review.get("expected_reference_patch_sha256"))
        .and_then(serde_json::Value::as_str)
    {
        Some(expected)
            if valid_sha256(expected)
                && reference_patch_sha256
                    .is_some_and(|actual| expected.eq_ignore_ascii_case(actual)) => {}
        Some(expected) if !valid_sha256(expected) => rejection_details.push(format!(
            "{field}.reference_review.expected_reference_patch_sha256 must be 64 hexadecimal characters"
        )),
        Some(_) => rejection_details.push(format!(
            "{field}.reference_review.expected_reference_patch_sha256 must equal reference_patch_sha256"
        )),
        None => rejection_details.push(format!(
            "{field}.reference_review.expected_reference_patch_sha256 is required"
        )),
    }
    let training_patch_count =
        target_sampling_usize(object, "training_patch_count", field, rejection_details);
    if training_patch_count.is_some_and(|count| count < MIN_TARGET_TRAINING_PATCHES) {
        rejection_details.push(format!(
            "{field}.training_patch_count is below required minimum {MIN_TARGET_TRAINING_PATCHES}"
        ));
    }
    let held_out_patch_count =
        target_sampling_usize(object, "held_out_patch_count", field, rejection_details);
    if held_out_patch_count.is_some_and(|count| count < MIN_TARGET_HELD_OUT_PATCHES) {
        rejection_details.push(format!(
            "{field}.held_out_patch_count is below required minimum {MIN_TARGET_HELD_OUT_PATCHES}"
        ));
    }
    if target_sampling_usize(object, "rejected_patch_count", field, rejection_details) != Some(0) {
        rejection_details.push(format!(
            "{field}.rejected_patch_count must be zero for retained sampling evidence"
        ));
    }
    let signal_bit_depth =
        target_sampling_usize(object, "scanner_signal_bit_depth", field, rejection_details);
    if signal_bit_depth.is_some_and(|bits| !(8..=16).contains(&bits)) {
        rejection_details.push(format!(
            "{field}.scanner_signal_bit_depth must be in [8,16]"
        ));
    }
    let signal_code_max =
        target_sampling_usize(object, "scanner_signal_code_max", field, rejection_details);
    if let (Some(bits), Some(actual_max)) = (signal_bit_depth, signal_code_max) {
        let expected_max = if bits == 16 {
            u16::MAX as usize
        } else {
            (1usize << bits) - 1
        };
        if actual_max != expected_max {
            rejection_details.push(format!(
                "{field}.scanner_signal_code_max {actual_max} does not match {bits}-bit maximum {expected_max}"
            ));
        }
    }

    let Some(captures) = object.get("captures").and_then(serde_json::Value::as_array) else {
        rejection_details.push(format!("{field}.captures must be an array"));
        return;
    };
    if target_sampling_usize(
        object,
        "corner_review_approved_capture_count",
        field,
        rejection_details,
    ) != Some(captures.len())
    {
        rejection_details.push(format!(
            "{field}.corner_review_approved_capture_count must equal the capture count"
        ));
    }
    if target_sampling_usize(
        object,
        "corner_review_required_capture_count",
        field,
        rejection_details,
    ) != Some(0)
    {
        rejection_details.push(format!(
            "{field}.corner_review_required_capture_count must be zero"
        ));
    }
    if captures.len() < 2 {
        rejection_details.push(format!(
            "{field}.captures must contain independent training and held-out captures"
        ));
    }
    let mut capture_ids = std::collections::HashSet::<String>::new();
    let mut file_hashes = std::collections::HashSet::<String>::new();
    let mut pixel_hashes = std::collections::HashSet::<String>::new();
    let mut training_sum = 0usize;
    let mut held_out_sum = 0usize;
    let mut training_capture_count = 0usize;
    let mut held_out_capture_count = 0usize;
    for (index, capture) in captures.iter().enumerate() {
        let capture_field = format!("{field}.captures[{index}]");
        let Some(capture) = capture.as_object() else {
            rejection_details.push(format!("{capture_field} must be an object"));
            continue;
        };
        match capture
            .get("capture_id")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
        {
            Some(id) if capture_ids.insert(id.to_ascii_lowercase()) => {}
            Some(id) => rejection_details.push(format!(
                "{field}.captures contains duplicate capture_id {id}"
            )),
            None => rejection_details.push(format!(
                "{capture_field}.capture_id must be a non-empty string"
            )),
        }
        for (hash_field, hashes) in [
            ("image_sha256", &mut file_hashes),
            ("decoded_pixel_sha256", &mut pixel_hashes),
        ] {
            match capture
                .get(hash_field)
                .and_then(serde_json::Value::as_str)
            {
                Some(hash) if valid_sha256(hash) && hashes.insert(hash.to_ascii_lowercase()) => {}
                Some(hash) if !valid_sha256(hash) => rejection_details.push(format!(
                    "{capture_field}.{hash_field} must be 64 lowercase or uppercase hexadecimal characters"
                )),
                Some(_) => rejection_details.push(format!(
                    "{field}.captures contains duplicate {hash_field}"
                )),
                None => rejection_details.push(format!(
                    "{capture_field}.{hash_field} is required"
                )),
            }
        }
        let decoded_pixel_sha256 = capture
            .get("decoded_pixel_sha256")
            .and_then(serde_json::Value::as_str);
        let chart_corners_sha256 = capture
            .get("chart_corners_sha256")
            .and_then(serde_json::Value::as_str);
        if chart_corners_sha256.is_none_or(|hash| !valid_sha256(hash)) {
            rejection_details.push(format!(
                "{capture_field}.chart_corners_sha256 must be 64 hexadecimal characters"
            ));
        }
        let overlay_pixel_sha256 = capture
            .get("overlay_pixel_sha256")
            .and_then(serde_json::Value::as_str);
        if overlay_pixel_sha256.is_none_or(|hash| !valid_sha256(hash)) {
            rejection_details.push(format!(
                "{capture_field}.overlay_pixel_sha256 must be 64 hexadecimal characters"
            ));
        }
        match capture
            .get("chart_corners")
            .cloned()
            .map(serde_json::from_value::<
                crate::calibration_target::TargetChartCorners,
            >)
        {
            Some(Ok(corners)) => {
                let actual = crate::calibration_target::sha256_chart_corners(corners);
                if chart_corners_sha256
                    .is_none_or(|expected| !expected.eq_ignore_ascii_case(&actual))
                {
                    rejection_details.push(format!(
                        "{capture_field}.chart_corners_sha256 does not match chart_corners"
                    ));
                }
            }
            Some(Err(_)) => rejection_details.push(format!(
                "{capture_field}.chart_corners must contain finite top_left/top_right/bottom_right/bottom_left coordinate pairs"
            )),
            None => rejection_details.push(format!(
                "{capture_field}.chart_corners is required"
            )),
        }
        let Some(corner_review) = capture
            .get("corner_review")
            .and_then(serde_json::Value::as_object)
        else {
            rejection_details.push(format!("{capture_field}.corner_review must be an object"));
            continue;
        };
        if corner_review
            .get("status")
            .and_then(serde_json::Value::as_str)
            != Some("approved")
        {
            rejection_details.push(format!(
                "{capture_field}.corner_review.status must be approved"
            ));
        }
        if corner_review
            .get("approved")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            rejection_details.push(format!(
                "{capture_field}.corner_review.approved must be true"
            ));
        }
        if corner_review
            .get("decoded_pixel_sha256_matched")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            rejection_details.push(format!(
                "{capture_field}.corner_review.decoded_pixel_sha256_matched must be true"
            ));
        }
        if corner_review
            .get("chart_corners_sha256_matched")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            rejection_details.push(format!(
                "{capture_field}.corner_review.chart_corners_sha256_matched must be true"
            ));
        }
        if corner_review
            .get("overlay_pixel_sha256_matched")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            rejection_details.push(format!(
                "{capture_field}.corner_review.overlay_pixel_sha256_matched must be true"
            ));
        }
        match corner_review
            .get("expected_decoded_pixel_sha256")
            .and_then(serde_json::Value::as_str)
        {
            Some(expected)
                if valid_sha256(expected)
                    && decoded_pixel_sha256
                        .is_some_and(|actual| expected.eq_ignore_ascii_case(actual)) => {}
            Some(expected) if !valid_sha256(expected) => rejection_details.push(format!(
                "{capture_field}.corner_review.expected_decoded_pixel_sha256 must be 64 hexadecimal characters"
            )),
            Some(_) => rejection_details.push(format!(
                "{capture_field}.corner_review.expected_decoded_pixel_sha256 must equal decoded_pixel_sha256"
            )),
            None => rejection_details.push(format!(
                "{capture_field}.corner_review.expected_decoded_pixel_sha256 is required"
            )),
        }
        match corner_review
            .get("expected_chart_corners_sha256")
            .and_then(serde_json::Value::as_str)
        {
            Some(expected)
                if valid_sha256(expected)
                    && chart_corners_sha256
                        .is_some_and(|actual| expected.eq_ignore_ascii_case(actual)) => {}
            Some(expected) if !valid_sha256(expected) => rejection_details.push(format!(
                "{capture_field}.corner_review.expected_chart_corners_sha256 must be 64 hexadecimal characters"
            )),
            Some(_) => rejection_details.push(format!(
                "{capture_field}.corner_review.expected_chart_corners_sha256 must equal chart_corners_sha256"
            )),
            None => rejection_details.push(format!(
                "{capture_field}.corner_review.expected_chart_corners_sha256 is required"
            )),
        }
        match corner_review
            .get("expected_overlay_pixel_sha256")
            .and_then(serde_json::Value::as_str)
        {
            Some(expected)
                if valid_sha256(expected)
                    && overlay_pixel_sha256
                        .is_some_and(|actual| expected.eq_ignore_ascii_case(actual)) => {}
            Some(expected) if !valid_sha256(expected) => rejection_details.push(format!(
                "{capture_field}.corner_review.expected_overlay_pixel_sha256 must be 64 hexadecimal characters"
            )),
            Some(_) => rejection_details.push(format!(
                "{capture_field}.corner_review.expected_overlay_pixel_sha256 must equal overlay_pixel_sha256"
            )),
            None => rejection_details.push(format!(
                "{capture_field}.corner_review.expected_overlay_pixel_sha256 is required"
            )),
        }
        let accepted = target_sampling_usize(
            capture,
            "accepted_patch_count",
            &capture_field,
            rejection_details,
        );
        let rejected = target_sampling_usize(
            capture,
            "rejected_patch_count",
            &capture_field,
            rejection_details,
        );
        if rejected != Some(0) {
            rejection_details.push(format!("{capture_field}.rejected_patch_count must be zero"));
        }
        if let (Some(accepted), Some(reference_count)) = (accepted, reference_patch_count) {
            if accepted != reference_count {
                rejection_details.push(format!(
                    "{capture_field}.accepted_patch_count {accepted} must equal reference_patch_count {reference_count}"
                ));
            }
        }
        let patch_count = capture
            .get("patches")
            .and_then(serde_json::Value::as_array)
            .map(|patches| {
                for (patch_index, patch) in patches.iter().enumerate() {
                    if patch.get("status").and_then(serde_json::Value::as_str) != Some("accepted") {
                        rejection_details.push(format!(
                            "{capture_field}.patches[{patch_index}].status must be accepted"
                        ));
                    }
                }
                patches.len()
            });
        if patch_count.is_none() {
            rejection_details.push(format!("{capture_field}.patches must be an array"));
        } else if patch_count != accepted {
            rejection_details.push(format!(
                "{capture_field}.patches length must equal accepted_patch_count"
            ));
        }
        match capture.get("role").and_then(serde_json::Value::as_str) {
            Some("training") => {
                training_capture_count += 1;
                training_sum = training_sum.saturating_add(accepted.unwrap_or(0));
            }
            Some("held_out") => {
                held_out_capture_count += 1;
                held_out_sum = held_out_sum.saturating_add(accepted.unwrap_or(0));
            }
            _ => {
                rejection_details.push(format!("{capture_field}.role must be training or held_out"))
            }
        }
    }
    if training_capture_count == 0 || held_out_capture_count == 0 {
        rejection_details.push(format!(
            "{field}.captures must include both training and held_out roles"
        ));
    }
    if training_patch_count != Some(training_sum) {
        rejection_details.push(format!(
            "{field}.training_patch_count does not equal accepted training capture patches"
        ));
    }
    if held_out_patch_count != Some(held_out_sum) {
        rejection_details.push(format!(
            "{field}.held_out_patch_count does not equal accepted held-out capture patches"
        ));
    }
}

fn target_sampling_usize(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    field: &str,
    rejection_details: &mut Vec<String>,
) -> Option<usize> {
    let value = object
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    if value.is_none() {
        rejection_details.push(format!("{field}.{key} must be a non-negative integer"));
    }
    value
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_declared_target_patch_signal_domain(
    schema_version: Option<u32>,
    fit: Option<&TargetFitDiagnostics>,
    domain: Option<&str>,
    linearization_model_id: Option<&str>,
    rejection_details: &mut Vec<String>,
) {
    if schema_version.is_none_or(|version| version < 2) || fit.is_none() {
        return;
    }
    match domain {
        Some("normalized_scanner_signal") => {
            if linearization_model_id.is_some() {
                rejection_details.push(
                    "target_patch_linearization_model_id must be absent when target_patch_signal_domain is normalized_scanner_signal"
                        .to_string(),
                );
            }
        }
        Some("scanner_linearized_transmittance") => {
            if linearization_model_id.is_none_or(|model_id| model_id.trim().is_empty()) {
                rejection_details.push(
                    "target_patch_linearization_model_id is required when target_patch_signal_domain is scanner_linearized_transmittance"
                        .to_string(),
                );
            }
        }
        Some(other) => rejection_details.push(format!(
            "unsupported target_patch_signal_domain `{other}`; expected normalized_scanner_signal or scanner_linearized_transmittance"
        )),
        None => rejection_details.push(
            "target_patch_signal_domain is required for schema v2 fitted records so matrix inputs match the runtime scanner domain"
                .to_string(),
        ),
    }
}

fn validate_target_patch_signal_domain(
    schema_version: Option<u32>,
    fit: Option<&TargetFitDiagnostics>,
    linearization: Option<&ScannerLinearizationCalibration>,
    domain: Option<&str>,
    linearization_model_id: Option<&str>,
    rejection_details: &mut Vec<String>,
) {
    validate_declared_target_patch_signal_domain(
        schema_version,
        fit,
        domain,
        linearization_model_id,
        rejection_details,
    );
    if schema_version.is_none_or(|version| version < 2) || fit.is_none() {
        return;
    }
    match linearization {
        Some(linearization) => {
            if domain != Some("scanner_linearized_transmittance") {
                rejection_details.push(
                    "a fitted matrix paired with scanner_linearization must use target_patch_signal_domain `scanner_linearized_transmittance`"
                        .to_string(),
                );
            }
            if linearization_model_id != Some(linearization.model_id.as_str()) {
                rejection_details.push(format!(
                    "target_patch_linearization_model_id must match scanner_linearization.model_id `{}`",
                    linearization.model_id
                ));
            }
        }
        None => {
            if domain != Some("normalized_scanner_signal") {
                rejection_details.push(
                    "a fitted matrix without scanner_linearization must use target_patch_signal_domain `normalized_scanner_signal`"
                        .to_string(),
                );
            }
        }
    }
}

fn validate_optional_fit(
    field: &str,
    value: Option<&TargetFitDiagnostics>,
    schema_version: Option<u32>,
    rejection_details: &mut Vec<String>,
) {
    let Some(value) = value else {
        return;
    };
    if value.method.trim().is_empty() {
        rejection_details.push(format!("{field}.method must not be empty"));
    }
    if value.patch_count < MIN_TARGET_PATCHES {
        rejection_details.push(format!(
            "{field}.patch_count {} is below required minimum {MIN_TARGET_PATCHES}",
            value.patch_count
        ));
    }
    if !value.target_residual_rms.is_finite() || value.target_residual_rms < 0.0 {
        rejection_details.push(format!(
            "{field}.target_residual_rms must be finite and non-negative"
        ));
    }
    if !value.target_residual_max.is_finite() || value.target_residual_max < 0.0 {
        rejection_details.push(format!(
            "{field}.target_residual_max must be finite and non-negative"
        ));
    }
    if schema_version.is_some_and(|version| version >= 2) && value.validation.is_none() {
        rejection_details.push(format!(
            "{field}.validation is required for schema v2 fitted records; in-sample residuals cannot establish calibration confidence"
        ));
    }
    if let Some(validation) = value.validation.as_ref() {
        if validation.evaluation_set != "held_out" {
            rejection_details.push(format!(
                "{field}.validation.evaluation_set must be `held_out`"
            ));
        }
        if validation.training_patch_count < MIN_TARGET_TRAINING_PATCHES {
            rejection_details.push(format!(
                "{field}.validation.training_patch_count {} is below required minimum {MIN_TARGET_TRAINING_PATCHES}",
                validation.training_patch_count
            ));
        }
        if validation.held_out_patch_count < MIN_TARGET_HELD_OUT_PATCHES {
            rejection_details.push(format!(
                "{field}.validation.held_out_patch_count {} is below required minimum {MIN_TARGET_HELD_OUT_PATCHES}",
                validation.held_out_patch_count
            ));
        }
        if validation.held_out_patch_count != value.patch_count {
            rejection_details.push(format!(
                "{field}.patch_count {} must equal validation.held_out_patch_count {}",
                value.patch_count, validation.held_out_patch_count
            ));
        }
        for (metric, metric_value) in [
            ("training_residual_rms", validation.training_residual_rms),
            ("training_residual_max", validation.training_residual_max),
            (
                "identity_baseline_residual_rms",
                validation.identity_baseline_residual_rms,
            ),
            (
                "identity_baseline_residual_max",
                validation.identity_baseline_residual_max,
            ),
        ] {
            if !metric_value.is_finite() || metric_value < 0.0 {
                rejection_details.push(format!(
                    "{field}.validation.{metric} must be finite and non-negative"
                ));
            }
        }
        if !validation
            .held_out_identity_improvement_fraction
            .is_finite()
            || validation.held_out_identity_improvement_fraction > 1.0
        {
            rejection_details.push(format!(
                "{field}.validation.held_out_identity_improvement_fraction must be finite and at most 1"
            ));
        }
        let expected_improvement = if validation.identity_baseline_residual_rms > 1e-12 {
            (validation.identity_baseline_residual_rms - value.target_residual_rms)
                / validation.identity_baseline_residual_rms
        } else {
            0.0
        };
        if expected_improvement.is_finite()
            && (validation.held_out_identity_improvement_fraction - expected_improvement).abs()
                > 1e-9
        {
            rejection_details.push(format!(
                "{field}.validation.held_out_identity_improvement_fraction is inconsistent with the held-out and identity RMS residuals"
            ));
        }
    }
    for (idx, residual) in value.per_hue_residuals.iter().enumerate() {
        if residual.hue_family.trim().is_empty() {
            rejection_details.push(format!(
                "{field}.per_hue_residuals[{idx}].hue_family must not be empty"
            ));
        }
        if residual.patch_count == 0 {
            rejection_details.push(format!(
                "{field}.per_hue_residuals[{idx}].patch_count must be positive"
            ));
        }
        if !residual.residual_rms.is_finite() || residual.residual_rms < 0.0 {
            rejection_details.push(format!(
                "{field}.per_hue_residuals[{idx}].residual_rms must be finite and non-negative"
            ));
        }
        if !residual.residual_max.is_finite() || residual.residual_max < 0.0 {
            rejection_details.push(format!(
                "{field}.per_hue_residuals[{idx}].residual_max must be finite and non-negative"
            ));
        }
    }
    for (idx, residual) in value.worst_patches.iter().enumerate() {
        if residual.hue_family.trim().is_empty() {
            rejection_details.push(format!(
                "{field}.worst_patches[{idx}].hue_family must not be empty"
            ));
        }
        if !residual.residual_error.is_finite() || residual.residual_error < 0.0 {
            rejection_details.push(format!(
                "{field}.worst_patches[{idx}].residual_error must be finite and non-negative"
            ));
        }
        if !residual.max_channel_error.is_finite() || residual.max_channel_error < 0.0 {
            rejection_details.push(format!(
                "{field}.worst_patches[{idx}].max_channel_error must be finite and non-negative"
            ));
        }
        if residual
            .source_rgb
            .iter()
            .chain(residual.reference_xyz.iter())
            .chain(residual.fitted_xyz.iter())
            .chain(residual.residual_xyz.iter())
            .any(|channel| !channel.is_finite())
        {
            rejection_details.push(format!(
                "{field}.worst_patches[{idx}] contains a non-finite RGB/XYZ residual value"
            ));
        }
    }
}

fn validate_confidence(
    confidence: Option<f64>,
    rejection_details: &mut Vec<String>,
) -> Option<f64> {
    match confidence {
        Some(value) if value.is_finite() && (0.0..=1.0).contains(&value) => {
            if value < MIN_CALIBRATION_CONFIDENCE {
                rejection_details.push(format!(
                    "confidence {value:.3} is below required minimum {MIN_CALIBRATION_CONFIDENCE:.3}"
                ));
            }
            Some(value)
        }
        Some(value) => {
            rejection_details.push(format!(
                "confidence must be finite and in [0, 1], got {value}"
            ));
            Some(value)
        }
        None => {
            rejection_details.push("missing confidence".to_string());
            None
        }
    }
}

fn validate_whitepoint(whitepoint: &[f64; 3], rejection_details: &mut Vec<String>) {
    if whitepoint
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        rejection_details.push("whitepoint values must be finite and positive".to_string());
        return;
    }
    if !(0.5..=1.5).contains(&whitepoint[1]) {
        rejection_details.push(format!(
            "whitepoint Y must be near normalized luminance 1.0, got {:.4}",
            whitepoint[1]
        ));
    }
    let sum = whitepoint.iter().sum::<f64>();
    if sum <= 1e-9 {
        rejection_details.push("whitepoint sum must be positive".to_string());
        return;
    }
    let x = whitepoint[0] / sum;
    let y = whitepoint[1] / sum;
    if !(0.20..=0.45).contains(&x) || !(0.20..=0.45).contains(&y) {
        rejection_details.push(format!(
            "whitepoint chromaticity is outside the expected scanner/illuminant range (x={x:.4}, y={y:.4})"
        ));
    }
}

fn validate_base_color(base_color: &[f64; 3], rejection_details: &mut Vec<String>) {
    if base_color
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        rejection_details.push("base_color values must be finite and positive".to_string());
    }
}

fn validate_matrix_values(name: &str, matrix: &[[f64; 3]; 3], rejection_details: &mut Vec<String>) {
    for (row_idx, row) in matrix.iter().enumerate() {
        for (col_idx, value) in row.iter().enumerate() {
            if !value.is_finite() {
                rejection_details.push(format!("{name}[{row_idx}][{col_idx}] must be finite"));
            }
        }
    }
}

fn validate_matrix_condition(
    name: &str,
    matrix: &Matrix3<f64>,
    rejection_details: &mut Vec<String>,
) -> f64 {
    let condition = matrix_condition_number(matrix);
    if !condition.is_finite() {
        rejection_details.push(format!("{name} matrix is singular"));
    } else if condition > MAX_MATRIX_CONDITION_NUMBER {
        rejection_details.push(format!(
            "{name} matrix condition number {condition:.3} exceeds maximum {MAX_MATRIX_CONDITION_NUMBER:.3}"
        ));
    }
    condition
}

fn rows_to_matrix3(rows: &[[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::new(
        rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0],
        rows[2][1], rows[2][2],
    )
}

fn matrix3_to_rows(matrix: &Matrix3<f64>) -> [[f64; 3]; 3] {
    [
        [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
        [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
        [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
    ]
}

pub fn matrix_condition_number_rows(rows: &[[f64; 3]; 3]) -> f64 {
    matrix_condition_number(&rows_to_matrix3(rows))
}

fn matrix_condition_number(matrix: &Matrix3<f64>) -> f64 {
    let svd = matrix.svd(false, false);
    let mut max_sv = 0.0f64;
    let mut min_sv = f64::INFINITY;
    for value in svd.singular_values.iter() {
        max_sv = max_sv.max(*value);
        min_sv = min_sv.min(*value);
    }
    if min_sv <= 1e-12 {
        f64::INFINITY
    } else {
        max_sv / min_sv
    }
}

fn compare_scanner_confidence(a: &&ScannerProfile, b: &&ScannerProfile) -> Ordering {
    a.confidence
        .partial_cmp(&b.confidence)
        .unwrap_or(Ordering::Equal)
}

fn base_color_delta(profile_base: [f64; 3], observed_base: [f64; 3]) -> Option<f64> {
    let profile = normalized_chroma(profile_base)?;
    let observed = normalized_chroma(observed_base)?;
    Some(
        ((profile[0] - observed[0]).powi(2)
            + (profile[1] - observed[1]).powi(2)
            + (profile[2] - observed[2]).powi(2))
        .sqrt(),
    )
}

fn normalized_chroma(rgb: [f64; 3]) -> Option<[f64; 3]> {
    if rgb.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return None;
    }
    let sum = rgb.iter().sum::<f64>();
    if sum <= 1e-9 {
        return None;
    }
    Some([rgb[0] / sum, rgb[1] / sum, rgb[2] / sum])
}

fn path_label(path: Option<&Path>) -> String {
    path.map(|path| path.to_string_lossy().to_string())
        .unwrap_or_else(|| "<memory>".to_string())
}

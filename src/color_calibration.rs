use crate::constants::D50_WHITE;
use crate::tiff_io::DngMetadata;
use nalgebra::{Matrix3, Vector3};
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
const MAX_REPORTED_CANDIDATES: usize = 5;

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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TargetFitDiagnostics {
    pub method: String,
    pub patch_count: usize,
    pub target_residual_rms: f64,
    pub target_residual_max: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_hue_residuals: Vec<TargetHueResidual>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub worst_patches: Vec<TargetPatchFitResidual>,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationDiagnostics {
    pub status: String,
    pub source: String,
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
    pub flare_black_white_diagnostics: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polynomial_fit: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lut_3d: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_e00_summary: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit: Option<TargetFitDiagnostics>,
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
    delta_e00_summary: Option<serde_json::Value>,
    #[serde(default)]
    patches: Option<Vec<serde_json::Value>>,
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
    scanner_settings_fingerprint: Option<String>,
    #[serde(default)]
    lut: Option<serde_json::Value>,
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
    delta_e00_summary: Option<serde_json::Value>,
    target_patches: Vec<TargetPatch>,
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
    scanner_settings_fingerprint: Option<String>,
    lut: Option<serde_json::Value>,
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
    Scanner(ScannerProfile),
    Roll(RollProfile),
    FilmHint(FilmHint),
}

pub fn not_configured() -> CalibrationLoadResult {
    CalibrationLoadResult {
        profile: None,
        diagnostics: CalibrationDiagnostics {
            status: "not_configured".to_string(),
            source: "image_derived_neutral_and_dominant_anchors".to_string(),
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

pub fn dng_color_matrix1_advisory_prior(metadata: &DngMetadata) -> Option<CalibrationLoadResult> {
    let work_to_xyz = metadata.color_matrix1?;
    let mut rejection_details = Vec::new();
    validate_matrix_values("DNG ColorMatrix1", &work_to_xyz, &mut rejection_details);
    let matrix = rows_to_matrix3(&work_to_xyz);
    let matrix_condition_number =
        validate_matrix_condition("DNG ColorMatrix1", &matrix, &mut rejection_details);
    let confidence = 0.62;
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
        flare_black_white_diagnostics: None,
        transform_type: Some("matrix".to_string()),
        polynomial_fit: None,
        lut_3d: None,
        delta_e00_summary: None,
        fit: None,
        scanner_settings_fingerprint: None,
        reason: if rejection_details.is_empty() {
            "DNG ColorMatrix1 selected as a weak scanner prior for automatic image adaptation"
                .to_string()
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
            "note": "ColorMatrix1 is used only as a scanner prior; scene anchors still adapt the roll.",
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
    };

    Some(CalibrationLoadResult {
        diagnostics: CalibrationDiagnostics {
            status: "applied".to_string(),
            source: "dng_color_matrix1_advisory_prior".to_string(),
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
            reason: "DNG ColorMatrix1 is being used as a weak scanner prior; automatic candidate scoring still compares it against image-derived mapping".to_string(),
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

pub fn fit_rgb_to_xyz_from_patches(
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

    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut patch_residuals = Vec::with_capacity(patches.len());
    let mut hue_residuals = std::collections::BTreeMap::<String, (usize, f64, f64)>::new();
    for patch in patches {
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
        let fitted = matrix * source;
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
    let fit = TargetFitDiagnostics {
        method: method.to_string(),
        patch_count: patches.len(),
        target_residual_rms,
        target_residual_max: max_abs,
        per_hue_residuals,
        worst_patches,
    };
    let residual_confidence = (-8.0 * target_residual_rms - 2.0 * max_abs).exp();
    let confidence = confidence_limit
        .unwrap_or(1.0)
        .min(residual_confidence)
        .clamp(0.0, 1.0);
    if confidence < MIN_CALIBRATION_CONFIDENCE {
        return Err(format!(
            "fitted target confidence {confidence:.3} is below required minimum {MIN_CALIBRATION_CONFIDENCE:.3} (rms residual {target_residual_rms:.6}, max residual {max_abs:.6})"
        ));
    }

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
            "selected calibration library profile uses schema_version {}; schema_version {} can record response curves, scanner diagnostics, higher-order transforms, and review-grade DeltaE00 summaries",
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

    let confidence = validate_confidence(raw.confidence, &mut rejection_details);
    validate_optional_fit("fit", raw.fit.as_ref(), &mut rejection_details);
    let target_patches = optional_target_patches_from_values(
        raw.patches.as_deref(),
        "patches",
        &mut rejection_details,
    );

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
    };
    CalibrationLoadResult {
        diagnostics: CalibrationDiagnostics {
            status: "applied".to_string(),
            source: "external_calibration_profile".to_string(),
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
                        "external profile uses schema_version {}; schema_version {} can record scanner response diagnostics and higher-order transform candidates",
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
        Ok(ValidatedLibraryRecord::Scanner(profile)) => library.scanners.push(profile),
        Ok(ValidatedLibraryRecord::Roll(profile)) => library.rolls.push(profile),
        Ok(ValidatedLibraryRecord::FilmHint(hint)) => library.film_hints.push(hint),
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
            .map(ValidatedLibraryRecord::Scanner),
        "roll_profile" => serde_json::from_value::<RawRollProfile>(value.clone())
            .map_err(|err| LibraryEntryRejection {
                path: path_label(path),
                record_type: Some(record_type.to_string()),
                profile_id: profile_id_from_value(value),
                reasons: vec![format!("failed to decode roll profile: {err}")],
            })
            .and_then(|raw| validate_roll_profile(raw, path))
            .map(ValidatedLibraryRecord::Roll),
        "film_hint" => serde_json::from_value::<RawFilmHint>(value.clone())
            .map_err(|err| LibraryEntryRejection {
                path: path_label(path),
                record_type: Some(record_type.to_string()),
                profile_id: profile_id_from_value(value),
                reasons: vec![format!("failed to decode film hint: {err}")],
            })
            .and_then(|raw| validate_film_hint(raw, path))
            .map(ValidatedLibraryRecord::FilmHint),
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
    validate_optional_fit("fit", raw.fit.as_ref(), &mut rejection_details);
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
        delta_e00_summary: raw.delta_e00_summary,
        target_patches,
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
    validate_optional_fit("fit", raw.fit.as_ref(), &mut rejection_details);
    let target_patches = optional_target_patches_from_values(
        raw.patches.as_deref(),
        "patches",
        &mut rejection_details,
    );
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
        scanner_settings_fingerprint: raw.scanner_settings_fingerprint,
        lut: raw.lut,
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
                flare_black_white_diagnostics: None,
                transform_type: None,
                polynomial_fit: None,
                lut_3d: None,
                delta_e00_summary: None,
                fit: None,
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
    if roll.correction_matrix.is_some() && scanner.is_none() {
        rejection_details.push("roll correction requires an applied scanner profile".to_string());
    }
    if roll.correction_matrix.is_some() && roll.scanner_profile_id.is_none() {
        rejection_details.push(
            "roll correction requires scanner_profile_id matching the selected scanner profile"
                .to_string(),
        );
    }
    if roll.correction_matrix.is_some()
        && scanner.is_some_and(|scanner| scanner.profile_id.is_none())
    {
        rejection_details.push(
            "roll correction requires the selected scanner profile to have a profile_id"
                .to_string(),
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
    match (
        roll.scanner_settings_fingerprint.as_ref(),
        scanner.and_then(|scanner| scanner.scanner_settings_fingerprint.as_ref()),
        roll.correction_matrix.is_some(),
    ) {
        (Some(expected), Some(actual), _) if actual != expected => rejection_details.push(format!(
            "roll profile scanner_settings_fingerprint `{expected}` does not match selected scanner fingerprint `{actual}`"
        )),
        (Some(_), None, _) => rejection_details.push(
            "roll profile declares scanner_settings_fingerprint but selected scanner has no settings fingerprint"
                .to_string(),
        ),
        (None, Some(_), true) => rejection_details.push(
            "roll correction requires scanner_settings_fingerprint matching the selected scanner settings"
                .to_string(),
        ),
        (None, None, true) => rejection_details.push(
            "roll correction requires scanner_settings_fingerprint on both the roll and selected scanner profile"
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
    let reason = if correction_applied {
        "explicit roll correction selected".to_string()
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
            "flare_black_white_diagnostics": scanner.flare_black_white_diagnostics,
            "polynomial_fit_recorded": scanner.polynomial_fit.is_some(),
            "lut_3d_recorded": scanner.lut_3d.is_some(),
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
        flare_black_white_diagnostics: scanner.flare_black_white_diagnostics.clone(),
        transform_type: Some(
            if scanner.lut_3d.is_some() {
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
        delta_e00_summary: scanner.delta_e00_summary.clone(),
        fit: scanner.fit.clone(),
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
    let object = patch
        .as_object()
        .ok_or_else(|| format!("patches[{idx}] must be a JSON object"))?;
    let patch_id = object
        .get("id")
        .or_else(|| object.get("patch_id"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let source_rgb = first_triplet(
        object,
        &["rgb", "scanner_rgb", "source_rgb", "measured_rgb"],
        &format!("patches[{idx}] source RGB"),
    )?;
    let reference_xyz = if let Ok(xyz) = first_triplet(
        object,
        &["xyz", "reference_xyz", "target_xyz"],
        &format!("patches[{idx}] reference XYZ"),
    ) {
        xyz
    } else {
        let lab = first_triplet(
            object,
            &["lab", "reference_lab", "target_lab"],
            &format!("patches[{idx}] reference Lab"),
        )?;
        lab_d50_to_xyz(lab)
    };

    Ok(TargetPatch {
        patch_id,
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

fn validate_target_patch(idx: usize, patch: &TargetPatch) -> Result<(), String> {
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

fn validate_optional_fit(
    field: &str,
    value: Option<&TargetFitDiagnostics>,
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

use crate::report::{
    format_system_time_utc, hash_file_sha256, system_time_unix_ms, PhaseReport, PipelineReport,
};
use crate::tiff_io;
use crate::{color_calibration, colorspace, tonemap};
use nalgebra::{Matrix3, Vector3};
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationSummary {
    pub fixture: String,
    pub report: ReportIdentitySummary,
    pub render: RenderDiagnosticSummary,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostic_consistency_issues: Vec<String>,
    #[serde(default)]
    pub input_orientation: InputOrientationValidationSummary,
    #[serde(default)]
    pub deskew: DeskewValidationSummary,
    #[serde(default)]
    pub border_crop: BorderCropValidationSummary,
    pub stitch: StitchValidationSummary,
    pub base_density: BaseDensityValidationSummary,
    #[serde(default)]
    pub negative_reconstruction: NegativeReconstructionValidationSummary,
    pub colorspace: ColorspaceValidationSummary,
    #[serde(default)]
    pub white_balance: WhiteBalanceValidationSummary,
    pub tone: ToneValidationSummary,
    pub warnings: Vec<PhaseWarnings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_baseline_comparison: Option<SummaryBaselineComparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comparison: Option<RenderComparisonSummary>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct InputOrientationValidationSummary {
    pub input_count: Option<usize>,
    pub component_count: usize,
    pub all_components_reported: Option<bool>,
    pub components: Vec<InputOrientationComponentValidationSummary>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct InputOrientationComponentValidationSummary {
    pub index: Option<usize>,
    pub decoded_pixel_sha256: Option<String>,
    pub source_orientation_materialized_decoded_pixel_sha256: Option<String>,
    pub tag_value: Option<u16>,
    pub metadata_transform: Option<String>,
    pub metadata_applied: Option<bool>,
    pub orientation_correction_requested: Option<String>,
    pub orientation_correction_transform: Option<String>,
    pub orientation_correction_applied: Option<bool>,
    pub effective_tag_value: Option<u16>,
    pub transform: Option<String>,
    pub applied: Option<bool>,
    pub source_width: Option<usize>,
    pub source_height: Option<usize>,
    pub output_width: Option<usize>,
    pub output_height: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DeskewValidationSummary {
    pub requested_mode: Option<String>,
    pub status: Option<String>,
    pub applied: Option<bool>,
    pub applied_component_count: Option<usize>,
    pub input_count: Option<usize>,
    pub component_count: usize,
    pub all_components_reported: Option<bool>,
    pub all_components_applied: Option<bool>,
    pub minimum_component_retained_area_ratio: Option<f64>,
    pub detected_source_skew_degrees: Option<f64>,
    pub correction_degrees: Option<f64>,
    pub confidence: Option<f64>,
    pub review_required: Option<bool>,
    pub review_reason: Option<String>,
    pub retained_area_ratio: Option<f64>,
    pub proposed_retained_area_ratio: Option<f64>,
    pub supporting_side_count: Option<usize>,
    pub horizontal_side_count: Option<usize>,
    pub vertical_side_count: Option<usize>,
    pub side_angle_spread_degrees: Option<f64>,
    pub interpolation: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct BorderCropValidationSummary {
    pub input_count: Option<usize>,
    pub component_count: usize,
    pub all_components_reported: Option<bool>,
    pub cropped_component_count: usize,
    pub all_components_cropped: Option<bool>,
    pub total_removed_edge_count: usize,
    pub minimum_removed_edge_count_per_component: Option<usize>,
    pub minimum_retained_area_ratio: Option<f64>,
    pub maximum_retained_area_ratio: Option<f64>,
    pub rejected_crop_warning_count: usize,
    pub components: Vec<BorderCropComponentValidationSummary>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct BorderCropComponentValidationSummary {
    pub index: Option<usize>,
    pub top_removed: Option<usize>,
    pub bottom_removed: Option<usize>,
    pub left_removed: Option<usize>,
    pub right_removed: Option<usize>,
    pub removed_edge_count: Option<usize>,
    pub dead_zone_detected: Option<bool>,
    pub input_width: Option<usize>,
    pub input_height: Option<usize>,
    pub output_width: Option<usize>,
    pub output_height: Option<usize>,
    pub retained_area_ratio: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct WhiteBalanceValidationSummary {
    pub technical_requested_mode: Option<String>,
    pub technical_status: Option<String>,
    pub technical_source: Option<String>,
    pub technical_reason: Option<String>,
    pub technical_applied: Option<bool>,
    pub technical_confidence: Option<f64>,
    pub technical_review_required: Option<bool>,
    pub technical_sample_count: Option<usize>,
    pub technical_occupied_spatial_bin_count: Option<usize>,
    pub technical_populated_luminance_band_count: Option<usize>,
    pub technical_estimated_source_cct_kelvin: Option<f64>,
    pub technical_pre_neutral_log_chroma: Option<f64>,
    pub technical_post_neutral_log_chroma: Option<f64>,
    pub creative_applied: Option<bool>,
    pub creative_temperature: Option<f64>,
    pub creative_tint: Option<f64>,
    pub creative_target_temperature_kelvin: Option<f64>,
    pub creative_separated_from_technical_master: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReportIdentitySummary {
    pub source_report_path: Option<String>,
    pub generated_at: Option<String>,
    pub generated_at_unix_ms: Option<u64>,
    pub package_version: Option<String>,
    pub binary_name: Option<String>,
    #[serde(default)]
    pub binary_path: Option<String>,
    #[serde(default)]
    pub binary_sha256: Option<String>,
    #[serde(default)]
    pub binary_file_size_bytes: Option<u64>,
    #[serde(default)]
    pub binary_identity_status: Option<String>,
    #[serde(default)]
    pub binary_identity_error: Option<String>,
    pub working_directory: Option<String>,
    pub cli_args: Option<Vec<String>>,
    pub output_path: Option<String>,
    pub report_schema_version: Option<u32>,
    pub pipeline_schema_version: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RenderDiagnosticSummary {
    pub source_report_path: Option<String>,
    pub source_report_generated_at: Option<String>,
    #[serde(default)]
    pub artifact_sha256_binding_required: bool,
    pub output_path: Option<String>,
    pub output_modified_at: Option<String>,
    pub output_file_size_bytes: Option<u64>,
    #[serde(default)]
    pub output_sha256: Option<String>,
    #[serde(default)]
    pub output_file_sha256: Option<String>,
    #[serde(default)]
    pub output_file_sha256_matches_report: Option<bool>,
    pub output_color_space: Option<String>,
    pub output_icc_profile_embedded: Option<bool>,
    pub output_icc_profile_description: Option<String>,
    pub output_file_icc_profile_status: Option<String>,
    pub output_file_icc_profile_embedded: Option<bool>,
    pub output_file_icc_profile_valid: Option<bool>,
    pub output_file_icc_profile_description: Option<String>,
    pub output_file_icc_profile_matches_report: Option<bool>,
    #[serde(default)]
    pub output_file_width: Option<usize>,
    #[serde(default)]
    pub output_file_height: Option<usize>,
    #[serde(default)]
    pub output_file_color_type: Option<String>,
    #[serde(default)]
    pub output_file_sample_format: Option<Vec<u16>>,
    #[serde(default)]
    pub output_file_dimensions_match_report: Option<bool>,
    #[serde(default)]
    pub output_file_storage_matches_report: Option<bool>,
    pub output_width: Option<usize>,
    pub output_height: Option<usize>,
    #[serde(default)]
    pub render_intent: Option<String>,
    #[serde(default)]
    pub quality_mode: Option<String>,
    #[serde(default)]
    pub master_scene_referred_requested: Option<bool>,
    #[serde(default)]
    pub master_scene_referred_path: Option<String>,
    #[serde(default)]
    pub master_scene_referred_sha256: Option<String>,
    #[serde(default)]
    pub master_scene_referred_file_sha256: Option<String>,
    #[serde(default)]
    pub master_scene_referred_file_sha256_matches_report: Option<bool>,
    #[serde(default)]
    pub master_scene_referred_file_status: Option<String>,
    #[serde(default)]
    pub master_scene_referred_file_width: Option<usize>,
    #[serde(default)]
    pub master_scene_referred_file_height: Option<usize>,
    #[serde(default)]
    pub master_scene_referred_file_color_type: Option<String>,
    #[serde(default)]
    pub master_scene_referred_file_sample_format: Option<Vec<u16>>,
    #[serde(default)]
    pub master_scene_referred_file_icc_profile_valid: Option<bool>,
    #[serde(default)]
    pub master_scene_referred_file_icc_profile_description: Option<String>,
    #[serde(default)]
    pub master_scene_referred_file_matches_report: Option<bool>,
    #[serde(default)]
    pub review_srgb_requested: Option<bool>,
    #[serde(default)]
    pub review_srgb_path: Option<String>,
    #[serde(default)]
    pub review_srgb_sha256: Option<String>,
    #[serde(default)]
    pub review_srgb_file_sha256: Option<String>,
    #[serde(default)]
    pub review_srgb_file_sha256_matches_report: Option<bool>,
    #[serde(default)]
    pub review_srgb_file_status: Option<String>,
    #[serde(default)]
    pub review_srgb_file_width: Option<usize>,
    #[serde(default)]
    pub review_srgb_file_height: Option<usize>,
    #[serde(default)]
    pub review_srgb_file_color_type: Option<String>,
    #[serde(default)]
    pub review_srgb_file_icc_profile_valid: Option<bool>,
    #[serde(default)]
    pub review_srgb_file_icc_profile_description: Option<String>,
    #[serde(default)]
    pub review_srgb_file_icc_profile_matches_standard_srgb: Option<bool>,
    #[serde(default)]
    pub review_srgb_file_matches_report: Option<bool>,
    #[serde(default)]
    pub review_srgb_gamut_mapping_space: Option<String>,
    #[serde(default)]
    pub review_srgb_gamut_mapped_ratio: Option<f64>,
    #[serde(default)]
    pub review_srgb_gamut_mapping_mean_chroma_scale: Option<f64>,
    #[serde(default)]
    pub review_srgb_gamut_mapping_min_chroma_scale: Option<f64>,
    #[serde(default)]
    pub review_srgb_post_map_out_of_gamut_pixel_count: Option<usize>,
    #[serde(default)]
    pub review_srgb_gamut_mapping_supported: Option<bool>,
    #[serde(default)]
    pub review_sidecar_sha256: Option<String>,
    #[serde(default)]
    pub input_base_confidence: Option<f64>,
    #[serde(default)]
    pub render_review_status: Option<String>,
    #[serde(default)]
    pub render_reviewable: Option<bool>,
    #[serde(default)]
    pub render_review_reason: Option<String>,
    #[serde(default)]
    pub tone_output_review_required: Option<bool>,
    #[serde(default)]
    pub tone_output_review_reason: Option<String>,
    #[serde(default)]
    pub tone_output_confidence_status: Option<String>,
    #[serde(default)]
    pub tone_output_evidence_confidence: Option<f64>,
    #[serde(default)]
    pub tone_output_render_luminance_range_p05_p95: Option<f64>,
    #[serde(default)]
    pub tone_output_render_to_mapped_luminance_range_ratio: Option<f64>,
    #[serde(default)]
    pub tone_output_maximum_post_tone_high_clip_ratio: Option<f64>,
    #[serde(default)]
    pub tone_output_maximum_post_tone_low_clip_ratio: Option<f64>,
    #[serde(default)]
    pub positive_input_likely_negative_like: Option<bool>,
    #[serde(default)]
    pub positive_input_accepted_high_warm_score: Option<bool>,
    #[serde(default)]
    pub positive_input_orange_mask_score: Option<f64>,
    #[serde(default)]
    pub positive_input_reason: Option<String>,
    pub overwrote_existing_output: Option<bool>,
    pub stale_render_artifact_count: Option<usize>,
    pub stale_render_artifacts: Vec<StaleRenderArtifactSummary>,
    pub stitch_decision: Option<String>,
    pub stitch_crop: Option<CropSummary>,
    pub base_estimate_source: Option<String>,
    pub render_input_source: Option<String>,
    pub colorspace_mapping_strategy: Option<String>,
    pub calibration_status: Option<String>,
    pub calibration_source: Option<String>,
    pub colorspace_candidate_risk: Option<String>,
    pub colorspace_tone_color_trust_state: Option<String>,
    pub colorspace_selected_quality_score: Option<f64>,
    pub colorspace_density_monotonicity_score: Option<f64>,
    #[serde(default)]
    pub colorspace_hue_linearity_score: Option<f64>,
    pub colorspace_saturation_preservation_median_ratio: Option<f64>,
    pub colorspace_spatial_neutral_delta_p95: Option<f64>,
    #[serde(default)]
    pub colorspace_memory_color_penalty: Option<f64>,
    #[serde(default)]
    pub colorspace_spatial_consistency_penalty: Option<f64>,
    pub colorspace_reference_patch_rms_delta_e: Option<f64>,
    #[serde(default)]
    pub colorspace_reference_patch_rms_delta_e2000: Option<f64>,
    pub colorspace_pre_scale_preserved_ratio: Option<f64>,
    pub colorspace_post_scale_preserved_ratio: Option<f64>,
    pub highlight_chroma_compressed_ratio: Option<f64>,
    pub highlight_neutral_chroma_compressed_ratio: Option<f64>,
    pub shadow_chroma_compressed_ratio: Option<f64>,
    pub high_frequency_grain: Option<HighFrequencyGrainSummary>,
    #[serde(default)]
    pub noise_reduction_enabled: Option<bool>,
    #[serde(default)]
    pub noise_reduction_requested_enabled: Option<bool>,
    #[serde(default)]
    pub noise_reduction_requested_strength: Option<f64>,
    #[serde(default)]
    pub noise_reduction_requested_scale: Option<f64>,
    #[serde(default)]
    pub noise_reduction_applied_ratio: Option<f64>,
    #[serde(default)]
    pub noise_reduction_structure_gate_start: Option<f64>,
    #[serde(default)]
    pub noise_reduction_structure_gate_end: Option<f64>,
    #[serde(default)]
    pub noise_reduction_structure_excluded_ratio: Option<f64>,
    #[serde(default)]
    pub noise_reduction_mean_abs_chroma_delta: Option<f64>,
    #[serde(default)]
    pub noise_reduction_mean_abs_luma_delta: Option<f64>,
    #[serde(default)]
    pub noise_reduction_flat_luma_p95_reduction_ratio: Option<f64>,
    #[serde(default)]
    pub noise_reduction_flat_chroma_p95_reduction_ratio: Option<f64>,
}

/// Returns fail-closed delivery-artifact problems for a compact render record.
///
/// Final image reviewability is distinct from successful processing: the referenced output must
/// exist strongly enough to be independently inspected, have nonzero dimensions, carry the ICC
/// profile reported by the pipeline, and have an explicit zero stale-artifact count.
pub fn delivery_artifact_integrity_issues(render: &RenderDiagnosticSummary) -> Vec<&'static str> {
    let mut issues = Vec::new();
    if render
        .output_path
        .as_deref()
        .is_none_or(|path| path.trim().is_empty())
    {
        issues.push("output_path_missing");
    }
    if render.output_width.is_none_or(|width| width == 0)
        || render.output_height.is_none_or(|height| height == 0)
    {
        issues.push("output_dimensions_missing_or_invalid");
    }
    match render.output_file_dimensions_match_report {
        Some(true) => {}
        Some(false) => issues.push("output_dimensions_mismatch"),
        None => issues.push("output_dimension_evidence_missing"),
    }
    match render.output_file_storage_matches_report {
        Some(true) => {}
        Some(false) => issues.push("output_storage_mismatch"),
        None => issues.push("output_storage_evidence_missing"),
    }
    match render.output_file_icc_profile_matches_report {
        Some(true) => {}
        Some(false) => issues.push("output_icc_profile_mismatch"),
        None => issues.push("output_icc_profile_evidence_missing"),
    }
    if render.artifact_sha256_binding_required {
        append_artifact_sha256_issues(
            &mut issues,
            "output",
            render.output_sha256.as_deref(),
            render.output_file_sha256.as_deref(),
            render.output_file_sha256_matches_report,
        );
    }
    append_promised_artifact_issues(
        &mut issues,
        "master_scene_referred",
        render.master_scene_referred_requested,
        render.master_scene_referred_path.as_deref(),
        render.master_scene_referred_file_matches_report,
    );
    append_promised_artifact_issues(
        &mut issues,
        "review_srgb",
        render.review_srgb_requested,
        render.review_srgb_path.as_deref(),
        render.review_srgb_file_matches_report,
    );
    if render.artifact_sha256_binding_required {
        append_promised_artifact_sha256_issues(
            &mut issues,
            "master_scene_referred",
            render.master_scene_referred_requested,
            render.master_scene_referred_sha256.as_deref(),
            render.master_scene_referred_file_sha256.as_deref(),
            render.master_scene_referred_file_sha256_matches_report,
        );
        append_promised_artifact_sha256_issues(
            &mut issues,
            "review_srgb",
            render.review_srgb_requested,
            render.review_srgb_sha256.as_deref(),
            render.review_srgb_file_sha256.as_deref(),
            render.review_srgb_file_sha256_matches_report,
        );
    }
    match render.review_srgb_requested {
        Some(true) => match render.review_srgb_gamut_mapping_supported {
            Some(true) => {}
            Some(false) => issues.push("review_srgb_gamut_mapping_not_supported"),
            None => issues.push("review_srgb_gamut_mapping_evidence_missing"),
        },
        Some(false) => {
            if render.review_srgb_gamut_mapping_supported.is_some() {
                issues.push("review_srgb_gamut_mapping_request_evidence_inconsistent");
            }
        }
        None => {}
    }
    match render.stale_render_artifact_count {
        Some(0) => {}
        Some(_) => issues.push("stale_render_artifacts"),
        None => issues.push("stale_render_artifact_evidence_missing"),
    }
    issues
}

fn append_artifact_sha256_issues(
    issues: &mut Vec<&'static str>,
    artifact: &'static str,
    declared_sha256: Option<&str>,
    actual_sha256: Option<&str>,
    matches_report: Option<bool>,
) {
    let declaration_valid = declared_sha256.is_some_and(is_valid_sha256_hex);
    match artifact {
        "output" => {
            if !declaration_valid {
                issues.push("output_sha256_declaration_missing_or_invalid");
            } else if actual_sha256.is_none() {
                issues.push("output_sha256_evidence_missing");
            } else if matches_report != Some(true) {
                issues.push("output_sha256_mismatch");
            }
        }
        "master_scene_referred" => {
            if !declaration_valid {
                issues.push("master_scene_referred_sha256_declaration_missing_or_invalid");
            } else if actual_sha256.is_none() {
                issues.push("master_scene_referred_sha256_evidence_missing");
            } else if matches_report != Some(true) {
                issues.push("master_scene_referred_sha256_mismatch");
            }
        }
        "review_srgb" => {
            if !declaration_valid {
                issues.push("review_srgb_sha256_declaration_missing_or_invalid");
            } else if actual_sha256.is_none() {
                issues.push("review_srgb_sha256_evidence_missing");
            } else if matches_report != Some(true) {
                issues.push("review_srgb_sha256_mismatch");
            }
        }
        _ => unreachable!("unsupported SHA-256 delivery artifact"),
    }
}

fn append_promised_artifact_sha256_issues(
    issues: &mut Vec<&'static str>,
    artifact: &'static str,
    requested: Option<bool>,
    declared_sha256: Option<&str>,
    actual_sha256: Option<&str>,
    matches_report: Option<bool>,
) {
    match requested {
        Some(true) => append_artifact_sha256_issues(
            issues,
            artifact,
            declared_sha256,
            actual_sha256,
            matches_report,
        ),
        Some(false) => {
            if declared_sha256.is_some() || actual_sha256.is_some() || matches_report.is_some() {
                match artifact {
                    "master_scene_referred" => {
                        issues.push("master_scene_referred_sha256_request_evidence_inconsistent")
                    }
                    "review_srgb" => {
                        issues.push("review_srgb_sha256_request_evidence_inconsistent")
                    }
                    _ => unreachable!("unsupported promised SHA-256 delivery artifact"),
                }
            }
        }
        None => {}
    }
}

fn is_valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn append_promised_artifact_issues(
    issues: &mut Vec<&'static str>,
    artifact: &'static str,
    requested: Option<bool>,
    path: Option<&str>,
    matches_report: Option<bool>,
) {
    let path_present = path.is_some_and(|path| !path.trim().is_empty());
    match (artifact, requested) {
        ("master_scene_referred", Some(true)) => {
            if !path_present {
                issues.push("master_scene_referred_path_missing");
            }
            match matches_report {
                Some(true) => {}
                Some(false) => issues.push("master_scene_referred_artifact_mismatch"),
                None => issues.push("master_scene_referred_artifact_evidence_missing"),
            }
        }
        ("master_scene_referred", Some(false)) => {
            if path_present || matches_report.is_some() {
                issues.push("master_scene_referred_request_evidence_inconsistent");
            }
        }
        ("master_scene_referred", None) => {
            issues.push("master_scene_referred_request_evidence_missing");
        }
        ("review_srgb", Some(true)) => {
            if !path_present {
                issues.push("review_srgb_path_missing");
            }
            match matches_report {
                Some(true) => {}
                Some(false) => issues.push("review_srgb_artifact_mismatch"),
                None => issues.push("review_srgb_artifact_evidence_missing"),
            }
        }
        ("review_srgb", Some(false)) => {
            if path_present || matches_report.is_some() {
                issues.push("review_srgb_request_evidence_inconsistent");
            }
        }
        ("review_srgb", None) => issues.push("review_srgb_request_evidence_missing"),
        _ => unreachable!("unsupported delivery artifact"),
    }
}

pub fn final_delivery_evidence_is_reviewable(render: &RenderDiagnosticSummary) -> bool {
    render.render_review_status.as_deref() == Some("reviewable")
        && render.render_reviewable == Some(true)
        && delivery_artifact_integrity_issues(render).is_empty()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StaleRenderArtifactSummary {
    pub path: Option<String>,
    pub modified_at: Option<String>,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CropSummary {
    pub x_start: Option<usize>,
    pub y_start: Option<usize>,
    pub width: Option<usize>,
    pub height: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HighFrequencyGrainSummary {
    pub sample_count: Option<usize>,
    pub sample_stride: Option<usize>,
    #[serde(default)]
    pub flat_luma_structure_max: Option<f64>,
    #[serde(default)]
    pub flat_sample_count: Option<usize>,
    #[serde(default)]
    pub flat_sample_ratio: Option<f64>,
    pub luma_residual_median: Option<f64>,
    pub luma_residual_p95: Option<f64>,
    pub chroma_residual_median: Option<f64>,
    pub chroma_residual_p95: Option<f64>,
    pub chroma_to_luma_p95_ratio: Option<f64>,
    #[serde(default)]
    pub flat_luma_residual_p95: Option<f64>,
    #[serde(default)]
    pub flat_chroma_residual_p95: Option<f64>,
    #[serde(default)]
    pub flat_chroma_to_luma_p95_ratio: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RenderComparisonSummary {
    pub status: String,
    pub issues: Vec<String>,
    pub baseline_report_path: String,
    pub current_report_path: Option<String>,
    pub baseline: RenderDiagnosticSummary,
    pub current: RenderDiagnosticSummary,
    pub output_dimensions_match: Option<bool>,
    pub stitch_decision_changed: bool,
    pub base_estimate_source_changed: bool,
    pub render_input_source_changed: bool,
    pub colorspace_mapping_strategy_changed: bool,
    pub calibration_status_changed: bool,
    pub calibration_source_changed: bool,
    pub colorspace_candidate_risk_changed: bool,
    pub colorspace_tone_color_trust_state_changed: bool,
    #[serde(default)]
    pub render_review_status_changed: bool,
    #[serde(default)]
    pub render_reviewable_changed: bool,
    #[serde(default)]
    pub tone_output_confidence_status_changed: bool,
    #[serde(default)]
    pub tone_output_review_required_changed: bool,
    #[serde(default)]
    pub tone_output_evidence_confidence_delta: Option<f64>,
    #[serde(default)]
    pub tone_output_render_luminance_range_p05_p95_delta: Option<f64>,
    #[serde(default)]
    pub tone_output_render_to_mapped_luminance_range_ratio_delta: Option<f64>,
    #[serde(default)]
    pub tone_output_maximum_post_tone_high_clip_ratio_delta: Option<f64>,
    #[serde(default)]
    pub tone_output_maximum_post_tone_low_clip_ratio_delta: Option<f64>,
    pub colorspace_selected_quality_score_delta: Option<f64>,
    pub colorspace_density_monotonicity_score_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_hue_linearity_score_delta: Option<f64>,
    pub colorspace_saturation_preservation_median_ratio_delta: Option<f64>,
    pub colorspace_spatial_neutral_delta_p95_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_memory_color_penalty_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_spatial_consistency_penalty_delta: Option<f64>,
    pub colorspace_reference_patch_rms_delta_e_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_reference_patch_rms_delta_e2000_delta: Option<f64>,
    pub colorspace_post_scale_preserved_ratio_delta: Option<f64>,
    pub highlight_chroma_compressed_ratio_delta: Option<f64>,
    pub highlight_neutral_chroma_compressed_ratio_delta: Option<f64>,
    pub shadow_chroma_compressed_ratio_delta: Option<f64>,
    pub luma_residual_p95_ratio: Option<f64>,
    pub chroma_residual_p95_ratio: Option<f64>,
    pub chroma_to_luma_p95_ratio_delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct TrackedValidationBaseline {
    #[serde(default)]
    pub fixture: Option<String>,
    #[serde(default)]
    pub stitch: TrackedStitchBaseline,
    #[serde(default)]
    pub render: TrackedRenderBaseline,
    #[serde(default)]
    pub colorspace: TrackedColorspaceBaseline,
    #[serde(default)]
    pub tolerances: ValidationBaselineTolerances,
    #[serde(default)]
    pub tone: TrackedToneBaseline,
    #[serde(default)]
    pub grain: TrackedGrainBaseline,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct TrackedStitchBaseline {
    #[serde(default)]
    pub decision: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub chosen_hypothesis: Option<String>,
    #[serde(default)]
    pub seam_exposure_correction_applied: Option<bool>,
    #[serde(default)]
    pub seam_exposure_model: Option<String>,
    #[serde(default)]
    pub seam_exposure_spatial_2d_gain_accepted: Option<bool>,
    #[serde(default)]
    pub seam_exposure_spatial_2d_gain_offset_accepted: Option<bool>,
    #[serde(default)]
    pub seam_exposure_spatial_quadratic_gain_accepted: Option<bool>,
    #[serde(default)]
    pub seam_exposure_spatial_quadratic_gain_offset_accepted: Option<bool>,
    #[serde(default)]
    pub seam_blend_mode: Option<String>,
    #[serde(default)]
    pub seam_blend_applied: Option<bool>,
    #[serde(default)]
    pub seam_blend_review_required: Option<bool>,
    #[serde(default)]
    pub seam_detail_review_required: Option<bool>,
    #[serde(default)]
    pub seam_detail_max_symmetric_energy_ratio: Option<f64>,
    #[serde(default)]
    pub seam_gradient_ratio: Option<f64>,
    #[serde(default)]
    pub seam_overlap_p95_abs_difference: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct TrackedRenderBaseline {
    #[serde(default)]
    pub output_width: Option<usize>,
    #[serde(default)]
    pub output_height: Option<usize>,
    #[serde(default)]
    pub output_color_space: Option<String>,
    #[serde(default)]
    pub output_file_icc_profile_matches_report: Option<bool>,
    #[serde(default)]
    pub base_estimate_source: Option<String>,
    #[serde(default)]
    pub raw_base_proxy_confidence: Option<f64>,
    #[serde(default)]
    pub raw_base_support_fraction: Option<f64>,
    #[serde(default)]
    pub render_input_source: Option<String>,
    #[serde(default)]
    pub colorspace_mapping_strategy: Option<String>,
    #[serde(default)]
    pub render_review_status: Option<String>,
    #[serde(default)]
    pub render_reviewable: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct TrackedColorspaceBaseline {
    #[serde(default)]
    pub calibration_status: Option<String>,
    #[serde(default)]
    pub calibration_source: Option<String>,
    #[serde(default)]
    pub calibration_scanner_profile_status: Option<String>,
    #[serde(default)]
    pub calibration_scanner_profile_id: Option<String>,
    #[serde(default)]
    pub calibration_roll_profile_status: Option<String>,
    #[serde(default)]
    pub calibration_roll_profile_id: Option<String>,
    #[serde(default)]
    pub calibration_confidence: Option<f64>,
    #[serde(default)]
    pub calibration_matrix_condition_number: Option<f64>,
    #[serde(default)]
    pub calibration_requested_film_stock: Option<String>,
    #[serde(default)]
    pub calibration_film_stock_status: Option<String>,
    #[serde(default)]
    pub calibration_film_stock_matched_roll_profiles: Vec<String>,
    #[serde(default)]
    pub calibration_rejection_details: Vec<String>,
    #[serde(default)]
    pub selected_candidate: Option<String>,
    #[serde(default)]
    pub selected_candidate_rank: Option<usize>,
    #[serde(default)]
    pub calibration_acceptance_status: Option<String>,
    #[serde(default)]
    pub calibration_color_mapping_applied: Option<bool>,
    #[serde(default)]
    pub calibration_acceptance_preferred_candidate: Option<String>,
    #[serde(default)]
    pub calibration_acceptance_beats_image_derived: Option<bool>,
    #[serde(default)]
    pub candidate_risk: Option<String>,
    #[serde(default)]
    pub tone_color_trust_state: Option<String>,
    #[serde(default)]
    pub selected_quality_score: Option<f64>,
    #[serde(default)]
    pub technical_safety_score: Option<f64>,
    #[serde(default)]
    pub color_fidelity_score: Option<f64>,
    #[serde(default)]
    pub selected_runner_up_quality_delta: Option<f64>,
    #[serde(default)]
    pub density_monotonicity_score: Option<f64>,
    #[serde(default)]
    pub hue_linearity_score: Option<f64>,
    #[serde(default)]
    pub saturation_preservation_median_ratio: Option<f64>,
    #[serde(default)]
    pub spatial_neutral_delta_p95: Option<f64>,
    #[serde(default)]
    pub memory_color_penalty: Option<f64>,
    #[serde(default)]
    pub spatial_consistency_penalty: Option<f64>,
    #[serde(default)]
    pub post_scale_preserved_ratio: Option<f64>,
    #[serde(default)]
    pub candidate_acceptance_signatures: Vec<String>,
    #[serde(default)]
    pub reference_patch_selected_rms_delta_e: Option<f64>,
    #[serde(default)]
    pub reference_patch_selected_rms_delta_e2000: Option<f64>,
    #[serde(default)]
    pub reference_patch_max_error_delta_vs_image_derived: Option<f64>,
    #[serde(default)]
    pub reference_patch_delta_e_rms_delta_vs_image_derived: Option<f64>,
    #[serde(default)]
    pub reference_patch_delta_e_max_delta_vs_image_derived: Option<f64>,
    #[serde(default)]
    pub reference_patch_delta_e2000_rms_delta_vs_image_derived: Option<f64>,
    #[serde(default)]
    pub reference_patch_delta_e2000_max_delta_vs_image_derived: Option<f64>,
    #[serde(default)]
    pub reference_patch_selected_regresses_image_derived: Option<bool>,
    #[serde(default)]
    pub reference_patch_worst_hue_families: Vec<String>,
    #[serde(default)]
    pub reference_patch_hue_family_regressions: Vec<String>,
    #[serde(default)]
    pub reference_patch_regressed_candidates: Vec<String>,
    #[serde(default)]
    pub neutral_estimate_score: Option<f64>,
    #[serde(default)]
    pub neutral_estimate_accepted: Option<bool>,
    #[serde(default)]
    pub neutral_estimate_populated_band_count: Option<usize>,
    #[serde(default)]
    pub neutral_estimate_dominant_band_fraction: Option<f64>,
    #[serde(default)]
    pub dominant_anchor_score: Option<f64>,
    #[serde(default)]
    pub dominant_anchor_accepted: Option<bool>,
    #[serde(default)]
    pub dominant_anchor_unstable_channel_count: Option<usize>,
    #[serde(default)]
    pub dominant_anchor_channel_unstable: Vec<bool>,
    #[serde(default)]
    pub channel_anchor_min_count: Option<usize>,
    #[serde(default)]
    pub channel_anchor_low_support: Vec<bool>,
    #[serde(default)]
    pub weak_anchor_fallback_used: Option<bool>,
    #[serde(default)]
    pub gamut_fallback_used: Option<bool>,
    #[serde(default)]
    pub neutral_trim_applied: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ValidationBaselineTolerances {
    #[serde(default = "default_confidence_abs_tolerance")]
    pub confidence_abs: f64,
    #[serde(default = "default_base_support_fraction_abs_tolerance")]
    pub base_support_fraction_abs: f64,
    #[serde(default = "default_tone_ratio_abs_tolerance")]
    pub tone_ratio_abs: f64,
    #[serde(default = "default_grain_ratio_abs_tolerance")]
    pub grain_ratio_abs: f64,
    #[serde(default = "default_colorspace_quality_abs_tolerance")]
    pub colorspace_quality_abs: f64,
    #[serde(default = "default_colorspace_preserved_abs_tolerance")]
    pub colorspace_preserved_abs: f64,
    #[serde(default = "default_colorspace_score_abs_tolerance")]
    pub colorspace_score_abs: f64,
    #[serde(default = "default_color_ratio_abs_tolerance")]
    pub color_ratio_abs: f64,
    #[serde(default = "default_spatial_neutral_abs_tolerance")]
    pub spatial_neutral_abs: f64,
    #[serde(default = "default_reference_xyz_abs_tolerance")]
    pub reference_xyz_abs: f64,
    #[serde(default = "default_reference_delta_e_abs_tolerance")]
    pub reference_delta_e_abs: f64,
}

impl Default for ValidationBaselineTolerances {
    fn default() -> Self {
        Self {
            confidence_abs: default_confidence_abs_tolerance(),
            base_support_fraction_abs: default_base_support_fraction_abs_tolerance(),
            tone_ratio_abs: default_tone_ratio_abs_tolerance(),
            grain_ratio_abs: default_grain_ratio_abs_tolerance(),
            colorspace_quality_abs: default_colorspace_quality_abs_tolerance(),
            colorspace_preserved_abs: default_colorspace_preserved_abs_tolerance(),
            colorspace_score_abs: default_colorspace_score_abs_tolerance(),
            color_ratio_abs: default_color_ratio_abs_tolerance(),
            spatial_neutral_abs: default_spatial_neutral_abs_tolerance(),
            reference_xyz_abs: default_reference_xyz_abs_tolerance(),
            reference_delta_e_abs: default_reference_delta_e_abs_tolerance(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct TrackedToneBaseline {
    #[serde(default)]
    pub highlight_chroma_compressed_ratio: Option<f64>,
    #[serde(default)]
    pub highlight_neutral_chroma_compressed_ratio: Option<f64>,
    #[serde(default)]
    pub shadow_chroma_compressed_ratio: Option<f64>,
    #[serde(default)]
    pub tone_output_confidence_status: Option<String>,
    #[serde(default)]
    pub tone_output_review_required: Option<bool>,
    #[serde(default)]
    pub tone_output_evidence_confidence: Option<f64>,
    #[serde(default)]
    pub render_luminance_range_p05_p95: Option<f64>,
    #[serde(default)]
    pub render_to_mapped_luminance_range_ratio: Option<f64>,
    #[serde(default)]
    pub maximum_post_tone_high_clip_ratio: Option<f64>,
    #[serde(default)]
    pub maximum_post_tone_low_clip_ratio: Option<f64>,
    #[serde(default)]
    pub adaptive_vibrance_skin_memory_protection_enabled: Option<bool>,
    #[serde(default)]
    pub adaptive_vibrance_skin_memory_protection_space: Option<String>,
    #[serde(default)]
    pub adaptive_vibrance_skin_memory_protected_ratio: Option<f64>,
    #[serde(default)]
    pub adaptive_vibrance_skin_memory_mean_protection: Option<f64>,
    #[serde(default)]
    pub adaptive_vibrance_preferred_memory_color_guard_enabled: Option<bool>,
    #[serde(default)]
    pub adaptive_vibrance_preferred_memory_color_guard_space: Option<String>,
    #[serde(default)]
    pub adaptive_vibrance_preferred_memory_color_guard_reference: Option<String>,
    #[serde(default)]
    pub adaptive_vibrance_preferred_memory_color_matched_ratio: Option<f64>,
    #[serde(default)]
    pub adaptive_vibrance_preferred_memory_color_limited_ratio: Option<f64>,
    #[serde(default)]
    pub adaptive_vibrance_preferred_memory_color_mean_scale_reduction: Option<f64>,
    #[serde(default)]
    pub preferred_skin_rendering_enabled: Option<bool>,
    #[serde(default)]
    pub preferred_skin_rendering_space: Option<String>,
    #[serde(default)]
    pub preferred_skin_rendering_preference_reference: Option<String>,
    #[serde(default)]
    pub preferred_skin_rendering_adjusted_ratio: Option<f64>,
    #[serde(default)]
    pub preferred_skin_rendering_mean_delta_e_ab: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct TrackedGrainBaseline {
    #[serde(default)]
    pub luma_residual_p95: Option<f64>,
    #[serde(default)]
    pub chroma_residual_p95: Option<f64>,
    #[serde(default)]
    pub chroma_to_luma_p95_ratio: Option<f64>,
    #[serde(default)]
    pub detail_review_required: Option<bool>,
    #[serde(default)]
    pub detail_decision_supported: Option<bool>,
    #[serde(default)]
    pub luminance_p10_retention: Option<f64>,
    #[serde(default)]
    pub chroma_p10_retention: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SummaryBaselineComparison {
    pub status: String,
    pub issues: Vec<String>,
    pub baseline_summary_path: String,
    pub tolerances: ValidationBaselineTolerances,
    pub fixture_matches: Option<bool>,
    pub output_dimensions_match: Option<bool>,
    pub output_color_space_changed: bool,
    pub output_file_icc_profile_matches_report: Option<bool>,
    pub stitch_decision_changed: bool,
    pub stitch_confidence_delta: Option<f64>,
    pub chosen_hypothesis_changed: bool,
    pub seam_exposure_correction_changed: bool,
    #[serde(default)]
    pub seam_exposure_model_changed: bool,
    #[serde(default)]
    pub seam_exposure_spatial_2d_gain_acceptance_changed: bool,
    #[serde(default)]
    pub seam_exposure_spatial_2d_gain_offset_acceptance_changed: bool,
    #[serde(default)]
    pub seam_exposure_spatial_quadratic_gain_acceptance_changed: bool,
    #[serde(default)]
    pub seam_exposure_spatial_quadratic_gain_offset_acceptance_changed: bool,
    #[serde(default)]
    pub seam_blend_mode_changed: bool,
    #[serde(default)]
    pub seam_blend_applied_changed: bool,
    #[serde(default)]
    pub seam_blend_review_required_changed: bool,
    #[serde(default)]
    pub seam_detail_review_required_changed: bool,
    #[serde(default)]
    pub seam_detail_max_symmetric_energy_ratio_delta: Option<f64>,
    #[serde(default)]
    pub seam_gradient_ratio_delta: Option<f64>,
    #[serde(default)]
    pub seam_overlap_p95_abs_difference_delta: Option<f64>,
    pub base_estimate_source_changed: bool,
    #[serde(default)]
    pub raw_base_proxy_confidence_delta: Option<f64>,
    #[serde(default)]
    pub raw_base_support_fraction_delta: Option<f64>,
    pub render_input_source_changed: bool,
    pub colorspace_mapping_strategy_changed: bool,
    pub calibration_status_changed: bool,
    pub calibration_source_changed: bool,
    #[serde(default)]
    pub calibration_scanner_profile_status_changed: bool,
    #[serde(default)]
    pub calibration_scanner_profile_id_changed: bool,
    #[serde(default)]
    pub calibration_roll_profile_status_changed: bool,
    #[serde(default)]
    pub calibration_roll_profile_id_changed: bool,
    #[serde(default)]
    pub calibration_confidence_delta: Option<f64>,
    #[serde(default)]
    pub calibration_matrix_condition_number_delta: Option<f64>,
    #[serde(default)]
    pub calibration_requested_film_stock_changed: bool,
    #[serde(default)]
    pub calibration_film_stock_status_changed: bool,
    #[serde(default)]
    pub calibration_film_stock_matched_roll_profiles_changed: bool,
    #[serde(default)]
    pub calibration_rejection_details_changed: bool,
    pub selected_candidate_changed: bool,
    #[serde(default)]
    pub selected_candidate_rank_changed: bool,
    #[serde(default)]
    pub colorspace_candidate_acceptance_changed: bool,
    pub calibration_acceptance_status_changed: bool,
    #[serde(default)]
    pub calibration_color_mapping_applied_changed: bool,
    pub calibration_acceptance_preferred_candidate_changed: bool,
    pub calibration_acceptance_beats_image_derived_changed: bool,
    pub colorspace_candidate_risk_changed: bool,
    pub colorspace_tone_color_trust_state_changed: bool,
    pub colorspace_selected_quality_score_delta: Option<f64>,
    pub colorspace_technical_safety_score_delta: Option<f64>,
    pub colorspace_color_fidelity_score_delta: Option<f64>,
    pub colorspace_selected_runner_up_quality_delta_delta: Option<f64>,
    pub colorspace_density_monotonicity_score_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_hue_linearity_score_delta: Option<f64>,
    pub colorspace_saturation_preservation_median_ratio_delta: Option<f64>,
    pub colorspace_spatial_neutral_delta_p95_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_memory_color_penalty_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_spatial_consistency_penalty_delta: Option<f64>,
    pub colorspace_post_scale_preserved_ratio_delta: Option<f64>,
    pub colorspace_reference_patch_selected_rms_delta_e_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_reference_patch_selected_rms_delta_e2000_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_reference_patch_max_error_delta_vs_image_derived_delta: Option<f64>,
    pub colorspace_reference_patch_delta_e_rms_delta_vs_image_derived_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_reference_patch_delta_e_max_delta_vs_image_derived_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_reference_patch_delta_e2000_rms_delta_vs_image_derived_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_reference_patch_delta_e2000_max_delta_vs_image_derived_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_reference_patch_selected_regresses_image_derived_changed: bool,
    #[serde(default)]
    pub colorspace_reference_patch_worst_hue_families_changed: bool,
    #[serde(default)]
    pub colorspace_reference_patch_hue_family_regressions_changed: bool,
    #[serde(default)]
    pub colorspace_reference_patch_regressed_candidates_changed: bool,
    pub colorspace_neutral_estimate_score_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_neutral_estimate_accepted_changed: bool,
    #[serde(default)]
    pub colorspace_neutral_estimate_populated_band_count_changed: bool,
    pub colorspace_neutral_estimate_dominant_band_fraction_delta: Option<f64>,
    pub colorspace_dominant_anchor_score_delta: Option<f64>,
    #[serde(default)]
    pub colorspace_dominant_anchor_accepted_changed: bool,
    #[serde(default)]
    pub colorspace_dominant_anchor_unstable_channel_count_changed: bool,
    #[serde(default)]
    pub colorspace_dominant_anchor_channel_unstable_changed: bool,
    #[serde(default)]
    pub colorspace_channel_anchor_min_count_changed: bool,
    #[serde(default)]
    pub colorspace_channel_anchor_low_support_changed: bool,
    #[serde(default)]
    pub colorspace_weak_anchor_fallback_changed: bool,
    #[serde(default)]
    pub colorspace_gamut_fallback_changed: bool,
    #[serde(default)]
    pub colorspace_neutral_trim_applied_changed: bool,
    #[serde(default)]
    pub colorspace_debug_artifact_invalid_count: usize,
    #[serde(default)]
    pub colorspace_debug_artifact_issues: Vec<String>,
    #[serde(default)]
    pub render_review_status_changed: bool,
    #[serde(default)]
    pub render_reviewable_changed: bool,
    #[serde(default)]
    pub tone_output_confidence_status_changed: bool,
    #[serde(default)]
    pub tone_output_review_required_changed: bool,
    #[serde(default)]
    pub tone_output_evidence_confidence_delta: Option<f64>,
    #[serde(default)]
    pub tone_output_render_luminance_range_p05_p95_delta: Option<f64>,
    #[serde(default)]
    pub tone_output_render_to_mapped_luminance_range_ratio_delta: Option<f64>,
    #[serde(default)]
    pub tone_output_maximum_post_tone_high_clip_ratio_delta: Option<f64>,
    #[serde(default)]
    pub tone_output_maximum_post_tone_low_clip_ratio_delta: Option<f64>,
    pub highlight_chroma_compressed_ratio_delta: Option<f64>,
    pub highlight_neutral_chroma_compressed_ratio_delta: Option<f64>,
    pub shadow_chroma_compressed_ratio_delta: Option<f64>,
    #[serde(default)]
    pub adaptive_vibrance_skin_memory_protection_enabled_changed: bool,
    #[serde(default)]
    pub adaptive_vibrance_skin_memory_protection_space_changed: bool,
    #[serde(default)]
    pub adaptive_vibrance_skin_memory_protected_ratio_delta: Option<f64>,
    #[serde(default)]
    pub adaptive_vibrance_skin_memory_mean_protection_delta: Option<f64>,
    #[serde(default)]
    pub adaptive_vibrance_preferred_memory_color_guard_enabled_changed: bool,
    #[serde(default)]
    pub adaptive_vibrance_preferred_memory_color_guard_space_changed: bool,
    #[serde(default)]
    pub adaptive_vibrance_preferred_memory_color_guard_reference_changed: bool,
    #[serde(default)]
    pub adaptive_vibrance_preferred_memory_color_matched_ratio_delta: Option<f64>,
    #[serde(default)]
    pub adaptive_vibrance_preferred_memory_color_limited_ratio_delta: Option<f64>,
    #[serde(default)]
    pub adaptive_vibrance_preferred_memory_color_mean_scale_reduction_delta: Option<f64>,
    #[serde(default)]
    pub preferred_skin_rendering_enabled_changed: bool,
    #[serde(default)]
    pub preferred_skin_rendering_space_changed: bool,
    #[serde(default)]
    pub preferred_skin_rendering_preference_reference_changed: bool,
    #[serde(default)]
    pub preferred_skin_rendering_adjusted_ratio_delta: Option<f64>,
    #[serde(default)]
    pub preferred_skin_rendering_mean_delta_e_ab_delta: Option<f64>,
    pub luma_residual_p95_ratio: Option<f64>,
    pub chroma_residual_p95_ratio: Option<f64>,
    pub chroma_to_luma_p95_ratio_delta: Option<f64>,
    #[serde(default)]
    pub grain_detail_review_required_changed: bool,
    #[serde(default)]
    pub grain_detail_decision_supported_changed: bool,
    #[serde(default)]
    pub grain_detail_luminance_p10_retention_delta: Option<f64>,
    #[serde(default)]
    pub grain_detail_chroma_p10_retention_delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StitchValidationSummary {
    pub decision: Option<String>,
    #[serde(default)]
    pub inferred_order: Vec<usize>,
    pub confidence: Option<f64>,
    pub chosen_hypothesis: Option<String>,
    pub rejection_reason: Option<String>,
    pub search_selection_reason: Option<String>,
    pub evaluated_candidate_count: Option<usize>,
    pub max_overlap_considered: Option<usize>,
    pub top_candidates: Vec<StitchCandidateSummary>,
    pub evidence_score: Option<f64>,
    pub prior_weight: Option<f64>,
    pub overlap_support_score: Option<f64>,
    pub vertical_offset_plausibility_score: Option<f64>,
    pub local_consistency_score: Option<f64>,
    pub plausibility_score: Option<f64>,
    #[serde(default)]
    pub homography_feature_validation: Option<HomographyFeatureValidationSummary>,
    #[serde(default)]
    pub homography_spatial_validation: Option<HomographySpatialValidationSummary>,
    pub seam_exposure_correction: Option<SeamExposureCorrectionSummary>,
    #[serde(default)]
    pub seam_blend: Option<SeamBlendValidationSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StitchCandidateSummary {
    pub rank: Option<usize>,
    pub overlap_width: Option<usize>,
    pub vertical_offset: Option<i64>,
    pub correspondence_score: Option<f64>,
    pub prior_weight: Option<f64>,
    pub search_score: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct HomographyFeatureValidationSummary {
    pub accepted: Option<bool>,
    pub validation_count: usize,
    pub accepted_count: usize,
    pub partition_method: Option<String>,
    pub minimum_training_match_count: Option<usize>,
    pub minimum_held_out_match_count: Option<usize>,
    pub minimum_training_spatial_cell_count: Option<usize>,
    pub minimum_held_out_spatial_cell_count: Option<usize>,
    pub minimum_training_inlier_ratio: Option<f64>,
    pub minimum_held_out_inlier_ratio: Option<f64>,
    pub minimum_reverse_validation_inlier_ratio: Option<f64>,
    pub maximum_held_out_p95_error_px: Option<f64>,
    pub maximum_reverse_validation_p95_error_px: Option<f64>,
    pub maximum_cross_fit_disagreement_px: Option<f64>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct HomographySpatialValidationSummary {
    pub accepted: Option<bool>,
    pub validation_count: usize,
    pub accepted_count: usize,
    pub method: Option<String>,
    pub minimum_split_ncc_improvement: Option<f64>,
    pub minimum_mean_ncc_improvement: Option<f64>,
    pub minimum_split_registration_error_reduction: Option<f64>,
    pub minimum_mean_registration_error_reduction: Option<f64>,
    pub minimum_split_sample_count: Option<usize>,
    pub minimum_model_deviation_from_translation_px: Option<f64>,
    pub model_selection_limit: Option<String>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SeamExposureCorrectionSummary {
    pub mode: Option<String>,
    pub model: Option<String>,
    pub applied: Option<bool>,
    pub reason: Option<String>,
    pub sample_count: Option<usize>,
    pub valid_sample_ratio: Option<f64>,
    pub gain_rgb: Option<Vec<f64>>,
    pub gain_luma: Option<f64>,
    pub spatial_gain_log_slope_x_rgb: Option<Vec<f64>>,
    pub spatial_gain_log_slope_x_luma: Option<f64>,
    pub spatial_gain_log_slope_y_rgb: Option<Vec<f64>>,
    pub spatial_gain_log_slope_y_luma: Option<f64>,
    pub spatial_gain_log_quadratic_xx_rgb: Option<Vec<f64>>,
    pub spatial_gain_log_quadratic_xx_luma: Option<f64>,
    pub spatial_gain_log_quadratic_xy_rgb: Option<Vec<f64>>,
    pub spatial_gain_log_quadratic_xy_luma: Option<f64>,
    pub spatial_gain_log_quadratic_yy_rgb: Option<Vec<f64>>,
    pub spatial_gain_log_quadratic_yy_luma: Option<f64>,
    pub spatial_gain_top_rgb: Option<Vec<f64>>,
    pub spatial_gain_bottom_rgb: Option<Vec<f64>>,
    pub spatial_offset_slope_x_rgb: Option<Vec<f64>>,
    pub spatial_offset_slope_x_luma: Option<f64>,
    pub spatial_offset_slope_x_rgb_normalized: Option<Vec<f64>>,
    pub spatial_offset_slope_y_rgb: Option<Vec<f64>>,
    pub spatial_offset_slope_y_luma: Option<f64>,
    pub spatial_offset_slope_y_rgb_normalized: Option<Vec<f64>>,
    pub spatial_offset_quadratic_xx_rgb: Option<Vec<f64>>,
    pub spatial_offset_quadratic_xx_luma: Option<f64>,
    pub spatial_offset_quadratic_xx_rgb_normalized: Option<Vec<f64>>,
    pub spatial_offset_quadratic_xy_rgb: Option<Vec<f64>>,
    pub spatial_offset_quadratic_xy_luma: Option<f64>,
    pub spatial_offset_quadratic_xy_rgb_normalized: Option<Vec<f64>>,
    pub spatial_offset_quadratic_yy_rgb: Option<Vec<f64>>,
    pub spatial_offset_quadratic_yy_luma: Option<f64>,
    pub spatial_offset_quadratic_yy_rgb_normalized: Option<Vec<f64>>,
    pub spatial_offset_top_rgb: Option<Vec<f64>>,
    pub spatial_offset_bottom_rgb: Option<Vec<f64>>,
    pub spatial_offset_top_rgb_normalized: Option<Vec<f64>>,
    pub spatial_offset_bottom_rgb_normalized: Option<Vec<f64>>,
    pub offset_rgb: Option<Vec<f64>>,
    pub offset_luma: Option<f64>,
    pub offset_rgb_normalized: Option<Vec<f64>>,
    pub seam_score_before: Option<f64>,
    pub seam_score_after: Option<f64>,
    pub clipped_high_before: Option<Vec<f64>>,
    pub clipped_high_after: Option<Vec<f64>>,
    pub clipped_low_before: Option<Vec<f64>>,
    pub clipped_low_after: Option<Vec<f64>>,
    pub training_window_count: Option<usize>,
    pub held_out_window_count: Option<usize>,
    pub training_sample_count: Option<usize>,
    pub held_out_sample_count: Option<usize>,
    pub gain_offset_training_window_count: Option<usize>,
    pub gain_offset_held_out_window_count: Option<usize>,
    pub gain_offset_consistent_window_ratio: Option<f64>,
    pub spatial_training_window_count: Option<usize>,
    pub spatial_held_out_window_count: Option<usize>,
    pub spatial_distinct_training_rows: Option<usize>,
    pub spatial_distinct_held_out_rows: Option<usize>,
    pub spatial_consistent_window_ratio: Option<f64>,
    pub spatial_slope_agreement_ratio: Option<f64>,
    pub spatial_affine_training_window_count: Option<usize>,
    pub spatial_affine_held_out_window_count: Option<usize>,
    pub spatial_affine_distinct_training_rows: Option<usize>,
    pub spatial_affine_distinct_held_out_rows: Option<usize>,
    pub spatial_affine_consistent_window_ratio: Option<f64>,
    pub spatial_affine_slope_agreement_ratio: Option<f64>,
    pub spatial_affine_center_offset_delta_normalized: Option<f64>,
    pub held_out_identity_seam_score: Option<f64>,
    pub held_out_gain_seam_score: Option<f64>,
    pub held_out_gain_offset_seam_score: Option<f64>,
    pub held_out_spatial_gain_seam_score: Option<f64>,
    pub held_out_spatial_gain_offset_seam_score: Option<f64>,
    pub held_out_selected_seam_score: Option<f64>,
    pub held_out_improvement_over_identity: Option<f64>,
    pub held_out_improvement_over_gain: Option<f64>,
    pub held_out_spatial_improvement_over_best_constant: Option<f64>,
    pub held_out_spatial_gain_offset_improvement_over_best_simpler: Option<f64>,
    pub held_out_validation_passed: Option<bool>,
    pub gain_offset_rejection_reason: Option<String>,
    pub spatial_rejection_reason: Option<String>,
    pub spatial_gain_offset_rejection_reason: Option<String>,
    pub spatial_2d_validation: Option<SpatialPhotometric2dSummary>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SpatialPhotometric2dSummary {
    pub coordinate_system: Option<String>,
    pub training_window_count: Option<usize>,
    pub held_out_window_count: Option<usize>,
    pub distinct_training_rows: Option<usize>,
    pub distinct_held_out_rows: Option<usize>,
    pub distinct_training_columns: Option<usize>,
    pub distinct_held_out_columns: Option<usize>,
    pub gain_consistent_window_ratio: Option<f64>,
    pub gain_slope_agreement_ratio: Option<f64>,
    pub gain_horizontal_slope_agreement_ratio: Option<f64>,
    pub gain_center_log_delta: Option<f64>,
    pub gain_offset_consistent_window_ratio: Option<f64>,
    pub gain_offset_slope_agreement_ratio: Option<f64>,
    pub gain_offset_horizontal_slope_agreement_ratio: Option<f64>,
    pub gain_offset_center_gain_log_delta: Option<f64>,
    pub gain_offset_center_offset_delta_normalized: Option<f64>,
    pub held_out_gain_seam_score: Option<f64>,
    pub held_out_gain_offset_seam_score: Option<f64>,
    pub gain_best_simpler_model: Option<String>,
    pub gain_improvement_over_best_simpler: Option<f64>,
    pub gain_offset_best_simpler_model: Option<String>,
    pub gain_offset_improvement_over_best_simpler: Option<f64>,
    pub gain_accepted: Option<bool>,
    pub gain_offset_accepted: Option<bool>,
    pub gain_rejection_reason: Option<String>,
    pub gain_offset_rejection_reason: Option<String>,
    pub quadratic: Option<SpatialPhotometricQuadraticSummary>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SpatialPhotometricQuadraticSummary {
    pub basis: Option<Vec<String>>,
    pub evaluation_grid_size: Option<usize>,
    pub regularization_lambda: Option<f64>,
    pub minimum_windows_per_split: Option<usize>,
    pub minimum_rows_per_split: Option<usize>,
    pub minimum_columns_per_split: Option<usize>,
    pub training_window_count: Option<usize>,
    pub held_out_window_count: Option<usize>,
    pub distinct_training_rows: Option<usize>,
    pub distinct_held_out_rows: Option<usize>,
    pub distinct_training_columns: Option<usize>,
    pub distinct_held_out_columns: Option<usize>,
    pub gain_design_condition_number: Option<f64>,
    pub held_out_gain_design_condition_number: Option<f64>,
    pub gain_consistent_window_ratio: Option<f64>,
    pub gain_curvature_coefficient_agreement_ratio: Option<f64>,
    pub gain_max_validation_field_log_delta: Option<f64>,
    pub gain_curvature_signal: Option<f64>,
    pub estimated_gain_log_quadratic_xx_rgb: Option<Vec<f64>>,
    pub estimated_gain_log_quadratic_xy_rgb: Option<Vec<f64>>,
    pub estimated_gain_log_quadratic_yy_rgb: Option<Vec<f64>>,
    pub gain_grid_min_rgb: Option<Vec<f64>>,
    pub gain_grid_max_rgb: Option<Vec<f64>>,
    pub gain_offset_design_condition_number: Option<f64>,
    pub held_out_gain_offset_design_condition_number: Option<f64>,
    pub gain_offset_consistent_window_ratio: Option<f64>,
    pub gain_offset_curvature_coefficient_agreement_ratio: Option<f64>,
    pub gain_offset_max_validation_gain_field_log_delta: Option<f64>,
    pub gain_offset_max_validation_offset_field_delta_normalized: Option<f64>,
    pub gain_offset_curvature_signal: Option<f64>,
    pub estimated_gain_offset_log_quadratic_xx_rgb: Option<Vec<f64>>,
    pub estimated_gain_offset_log_quadratic_xy_rgb: Option<Vec<f64>>,
    pub estimated_gain_offset_log_quadratic_yy_rgb: Option<Vec<f64>>,
    pub estimated_gain_offset_quadratic_xx_rgb: Option<Vec<f64>>,
    pub estimated_gain_offset_quadratic_xy_rgb: Option<Vec<f64>>,
    pub estimated_gain_offset_quadratic_yy_rgb: Option<Vec<f64>>,
    pub gain_offset_grid_gain_min_rgb: Option<Vec<f64>>,
    pub gain_offset_grid_gain_max_rgb: Option<Vec<f64>>,
    pub gain_offset_grid_offset_abs_max_normalized: Option<f64>,
    pub held_out_gain_seam_score: Option<f64>,
    pub held_out_gain_offset_seam_score: Option<f64>,
    pub gain_best_simpler_model: Option<String>,
    pub gain_improvement_over_best_simpler: Option<f64>,
    pub gain_offset_best_simpler_model: Option<String>,
    pub gain_offset_improvement_over_best_simpler: Option<f64>,
    pub gain_accepted: Option<bool>,
    pub gain_offset_accepted: Option<bool>,
    pub gain_rejection_reason: Option<String>,
    pub gain_offset_rejection_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SeamBlendValidationSummary {
    pub mode: Option<String>,
    pub applied: Option<bool>,
    pub reason: Option<String>,
    pub review_required: Option<bool>,
    pub review_required_merge_count: usize,
    pub review_reasons: Vec<String>,
    pub merge_count: usize,
    pub applied_merge_count: usize,
    pub overlap_width_px: Option<usize>,
    pub overlap_height_px: Option<usize>,
    pub transition_width_px: Option<usize>,
    pub pyramid_levels: Option<usize>,
    pub seam_path_mean_normalized_cost: Option<f64>,
    pub seam_path_p95_normalized_cost: Option<f64>,
    pub overlap_mean_abs_difference: Option<f64>,
    pub overlap_p95_abs_difference: Option<f64>,
    pub output_seam_gradient_p95: Option<f64>,
    pub source_seam_gradient_p95: Option<f64>,
    pub output_to_source_seam_gradient_ratio: Option<f64>,
    #[serde(default)]
    pub detail_consistency: Option<SeamDetailConsistencyValidationSummary>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SeamDetailConsistencyValidationSummary {
    pub method: Option<String>,
    pub evaluated: Option<bool>,
    pub merge_count: usize,
    pub evaluated_merge_count: usize,
    pub decision_supported: Option<bool>,
    pub decision_supported_merge_count: usize,
    pub review_required: Option<bool>,
    pub review_required_merge_count: usize,
    pub minimum_supported_scale_count: Option<usize>,
    pub maximum_imbalanced_scale_count: Option<usize>,
    pub maximum_symmetric_energy_ratio: Option<f64>,
    pub review_ratio_threshold: Option<f64>,
    pub minimum_direction_consistency: Option<f64>,
    pub maximum_cross_split_ratio: Option<f64>,
    pub minimum_repeated_scale_count: Option<usize>,
    pub reasons: Vec<String>,
    pub review_reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaseDensityValidationSummary {
    pub base_confidence: Option<f64>,
    pub raw_base_confidence: Option<f64>,
    #[serde(default)]
    pub raw_base_proxy_confidence: Option<f64>,
    #[serde(default)]
    pub raw_base_support_fraction: Option<f64>,
    pub base_estimate_source: Option<String>,
    pub density_confidence: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct NegativeReconstructionValidationSummary {
    pub input_mode: Option<String>,
    pub density_inversion_skipped: Option<bool>,
    pub density_confidence: Option<f64>,
    pub response_model: Option<String>,
    pub response_source: Option<String>,
    pub response_accepted: Option<bool>,
    pub response_model_review_required: Option<bool>,
    pub crosstalk_model: Option<String>,
    pub characteristic_curve_model: Option<String>,
    pub measured_model_id: Option<String>,
    pub measured_confidence: Option<f64>,
    pub held_out_delta_e00_rms: Option<f64>,
    pub held_out_delta_e00_max: Option<f64>,
    pub unit_slope_delta_e00_rms: Option<f64>,
    pub held_out_improvement_over_unit_slope: Option<f64>,
    pub maximum_density_noise_gain: Option<f64>,
    pub reconstruction_review_required: Option<bool>,
    pub signed_headroom_preserved: Option<bool>,
    pub curve_extrapolated_any_ratio: Option<f64>,
    pub curve_interpolation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColorspaceValidationSummary {
    pub calibration_status: Option<String>,
    pub calibration_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_color_mapping_application: Option<CalibrationColorMappingApplicationSummary>,
    #[serde(default)]
    pub calibration_scanner_profile_status: Option<String>,
    #[serde(default)]
    pub calibration_scanner_profile_id: Option<String>,
    #[serde(default)]
    pub calibration_roll_profile_status: Option<String>,
    #[serde(default)]
    pub calibration_roll_profile_id: Option<String>,
    pub calibration_external_profile_path: Option<String>,
    pub calibration_profile_schema_version: Option<u32>,
    pub calibration_confidence: Option<f64>,
    pub calibration_reason: Option<String>,
    pub calibration_matrix_condition_number: Option<f64>,
    pub calibration_whitepoint: Option<Vec<f64>>,
    pub calibration_requested_film_stock: Option<String>,
    pub calibration_film_stock_status: Option<String>,
    pub calibration_film_stock_reason: Option<String>,
    #[serde(default)]
    pub calibration_film_stock_matched_roll_profiles: Vec<String>,
    pub calibration_rejection_details: Vec<String>,
    pub render_input_source: Option<String>,
    pub render_input_reason: Option<String>,
    pub density_candidate_evaluated: Option<bool>,
    pub exposure_scale: Option<f64>,
    pub mapping_strategy: Option<String>,
    pub selected_mapping_reason: Option<String>,
    pub selected_candidate: Option<String>,
    pub selected_candidate_rank: Option<usize>,
    pub selected_candidate_score: Option<f64>,
    pub selected_quality_score: Option<f64>,
    pub technical_safety_score: Option<f64>,
    pub color_fidelity_score: Option<f64>,
    pub candidate_risk: Option<String>,
    pub tone_color_trust_state: Option<String>,
    pub selected_runner_up_quality_delta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_quality_components: Option<ColorQualityComponentsSummary>,
    pub candidate_quality_scores: Vec<ColorCandidateQualitySummary>,
    pub candidate_acceptance: Vec<ColorCandidateAcceptanceSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neutral_estimate_quality: Option<NeutralEstimateQualitySummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neutral_sample_rejections: Option<NeutralSampleRejectionSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dominant_anchor_sample_rejections: Option<DominantAnchorSampleRejectionSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dominant_anchor_quality: Option<DominantAnchorQualitySummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_patch_evaluation: Option<ReferencePatchEvaluationSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neutral_trim_before_after: Option<NeutralTrimBeforeAfterSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_acceptance: Option<CalibrationAcceptanceSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neutral_safety_rescue: Option<NeutralSafetyRescueSummary>,
    pub selection_rejections: Vec<String>,
    pub regularization_lambda: Option<f64>,
    pub neutral_sample_bands: Option<Vec<usize>>,
    pub dominant_anchor_bands: Option<Vec<Vec<usize>>>,
    pub neutral_trim_scale: Option<Vec<f64>>,
    pub neutral_trim_applied: Option<bool>,
    pub channel_anchor_counts: Option<Vec<usize>>,
    pub channel_anchor_min_count: Option<usize>,
    pub channel_anchor_low_support: Option<Vec<bool>>,
    pub weak_anchor_fallback_used: Option<bool>,
    pub gamut_fallback_used: Option<bool>,
    pub color_candidate_comparison_artifact: Option<String>,
    pub gamut_clipping_map_artifact: Option<String>,
    #[serde(default)]
    pub scene_referred_prophoto_float_artifact: Option<String>,
    pub debug_artifacts: Vec<ColorDebugArtifactSummary>,
    pub gamut_clipping_map_encoding: Option<String>,
    pub gamut_clipping_map_preserved_ratio: Option<f64>,
    pub gamut_clipping_map_any_clipped_ratio: Option<f64>,
    pub image_matrix_pre_scale_clipped_low_ratio: Option<Vec<f64>>,
    pub image_matrix_pre_scale_clipped_high_ratio: Option<Vec<f64>>,
    pub image_matrix_exposure_scale: Option<f64>,
    pub image_matrix_pre_scale_preserved_ratio: Option<f64>,
    pub image_matrix_neutral_balance_delta: Option<Vec<f64>>,
    pub calibrated_profile_pre_scale_clipped_low_ratio: Option<Vec<f64>>,
    pub calibrated_profile_pre_scale_clipped_high_ratio: Option<Vec<f64>>,
    pub calibrated_profile_exposure_scale: Option<f64>,
    pub calibrated_profile_pre_scale_preserved_ratio: Option<f64>,
    pub calibrated_profile_neutral_balance_delta: Option<Vec<f64>>,
    pub pre_scale_preserved_ratio: Option<f64>,
    pub post_scale_preserved_ratio: Option<f64>,
    pub post_scale_clipped_high_ratio: Option<Vec<f64>>,
    pub post_scale_clipped_low_ratio: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CalibrationColorMappingApplicationSummary {
    pub evaluated: Option<bool>,
    pub applied: Option<bool>,
    pub selection_status: Option<String>,
    pub selected_candidate: Option<String>,
    pub preferred_candidate: Option<String>,
    pub reason: Option<String>,
    pub definition: Option<String>,
}

/// Checks that the explicit calibration color-mapping application record agrees with the
/// independently reported candidate-selection diagnostics. Reports created before this record
/// existed remain readable and are not treated as contradictory merely because it is absent.
pub fn calibration_color_mapping_consistency_issues(
    colorspace: &ColorspaceValidationSummary,
) -> Vec<String> {
    let Some(application) = colorspace.calibration_color_mapping_application.as_ref() else {
        return Vec::new();
    };
    let mut issues = Vec::new();
    let acceptance = colorspace.calibration_acceptance.as_ref();

    if application.evaluated.is_none() {
        issues.push("calibration_color_mapping_evaluated_missing".to_string());
    }
    if application.applied.is_none() {
        issues.push("calibration_color_mapping_applied_missing".to_string());
    }
    if application.selection_status.is_none() {
        issues.push("calibration_color_mapping_selection_status_missing".to_string());
    }
    if application.selected_candidate.is_none() {
        issues.push("calibration_color_mapping_selected_candidate_missing".to_string());
    }
    if application.reason.as_deref().is_none_or(str::is_empty) {
        issues.push("calibration_color_mapping_reason_missing".to_string());
    }
    if application.definition.as_deref().is_none_or(str::is_empty) {
        issues.push("calibration_color_mapping_definition_missing".to_string());
    }

    if let Some(evaluated) = application.evaluated {
        if evaluated != application.preferred_candidate.is_some() {
            issues.push(
                "calibration_color_mapping_evaluated_preferred_candidate_mismatch".to_string(),
            );
        }
    }
    if let (Some(applied), Some(status)) =
        (application.applied, application.selection_status.as_deref())
    {
        let status_means_applied = matches!(status, "accepted" | "forced");
        if applied != status_means_applied {
            issues.push("calibration_color_mapping_applied_status_mismatch".to_string());
        }
    }
    match (
        application.selection_status.as_deref(),
        acceptance.and_then(|value| value.status.as_deref()),
    ) {
        (Some(application_status), Some(acceptance_status))
            if application_status != acceptance_status =>
        {
            issues.push("calibration_color_mapping_acceptance_status_mismatch".to_string());
        }
        (_, None) => {
            issues.push("calibration_color_mapping_acceptance_status_missing".to_string());
        }
        _ => {}
    }
    if application.selected_candidate.as_deref() != colorspace.selected_candidate.as_deref() {
        issues.push("calibration_color_mapping_selected_candidate_mismatch".to_string());
    }
    if application.preferred_candidate.as_deref()
        != acceptance.and_then(|value| value.preferred_candidate.as_deref())
    {
        issues.push("calibration_color_mapping_preferred_candidate_mismatch".to_string());
    }

    issues
}

/// Checks the one-way preferred-memory-colour creative guard for internally contradictory
/// population and model diagnostics. Legacy reports without the optional guard remain readable.
pub fn adaptive_vibrance_preferred_memory_color_guard_consistency_issues(
    tone: &ToneValidationSummary,
) -> Vec<String> {
    let Some(guard) = tone.adaptive_vibrance_preferred_memory_color_guard.as_ref() else {
        return Vec::new();
    };
    let mut issues = Vec::new();
    let required_text = [
        ("method", guard.method.as_deref()),
        ("working_space", guard.working_space.as_deref()),
        ("reference", guard.reference.as_deref()),
        ("interpretation", guard.interpretation.as_deref()),
    ];
    if guard.enabled.is_none() {
        issues.push("preferred_memory_color_guard_enabled_missing".to_string());
    }
    for (field, value) in required_text {
        if value.is_none_or(str::is_empty) {
            issues.push(format!("preferred_memory_color_guard_{field}_missing"));
        }
    }
    match (
        guard.core_normalized_radius,
        guard.support_normalized_radius,
    ) {
        (Some(core), Some(support)) if core > 0.0 && support > core => {}
        (Some(_), Some(_)) => {
            issues.push("preferred_memory_color_guard_radius_order_invalid".to_string())
        }
        _ => issues.push("preferred_memory_color_guard_radius_missing".to_string()),
    }

    let ratio_fields = [
        ("evaluated", guard.evaluated_pixel_ratio),
        ("matched", guard.matched_pixel_ratio),
        ("limited", guard.limited_pixel_ratio),
    ];
    for (field, value) in ratio_fields {
        match value {
            Some(value) if (0.0..=1.0).contains(&value) => {}
            Some(_) => issues.push(format!(
                "preferred_memory_color_guard_{field}_pixel_ratio_invalid"
            )),
            None => issues.push(format!(
                "preferred_memory_color_guard_{field}_pixel_ratio_missing"
            )),
        }
    }
    if guard
        .matched_pixel_ratio
        .zip(guard.evaluated_pixel_ratio)
        .is_some_and(|(matched, evaluated)| matched > evaluated + 1e-12)
    {
        issues.push("preferred_memory_color_guard_matched_exceeds_evaluated".to_string());
    }
    if guard
        .limited_pixel_ratio
        .zip(guard.matched_pixel_ratio)
        .is_some_and(|(limited, matched)| limited > matched + 1e-12)
    {
        issues.push("preferred_memory_color_guard_limited_exceeds_matched".to_string());
    }
    if guard.enabled == Some(false)
        && [
            guard.evaluated_pixel_ratio,
            guard.matched_pixel_ratio,
            guard.limited_pixel_ratio,
            guard.mean_scale_reduction,
            guard.max_scale_reduction,
        ]
        .into_iter()
        .flatten()
        .any(|value| value.abs() > 1e-12)
    {
        issues.push("preferred_memory_color_guard_disabled_with_nonzero_effect".to_string());
    }
    match (guard.mean_scale_reduction, guard.max_scale_reduction) {
        (Some(mean), Some(maximum)) if mean >= 0.0 && maximum >= 0.0 && mean <= maximum + 1e-12 => {
        }
        (Some(_), Some(_)) => {
            issues.push("preferred_memory_color_guard_scale_reduction_invalid".to_string())
        }
        _ => issues.push("preferred_memory_color_guard_scale_reduction_missing".to_string()),
    }

    let expected_families = ["sky", "spring_grass", "autumn_grass"];
    let actual_families = guard
        .families
        .iter()
        .filter_map(|family| family.family.as_deref())
        .collect::<Vec<_>>();
    if actual_families != expected_families {
        issues.push("preferred_memory_color_guard_family_set_invalid".to_string());
    }
    let mut family_matched_sum = 0.0f64;
    let mut family_limited_sum = 0.0f64;
    for family in &guard.families {
        let name = family.family.as_deref().unwrap_or("unknown");
        if family
            .preferred_center_lab
            .as_ref()
            .is_none_or(|values| values.len() != 3)
            || family
                .preferred_center_lch
                .as_ref()
                .is_none_or(|values| values.len() != 3)
        {
            issues.push(format!(
                "preferred_memory_color_guard_family_{name}_center_invalid"
            ));
        }
        match (
            family.semi_major_axis_ab,
            family.semi_minor_axis_ab,
            family.axis_ratio,
        ) {
            (Some(major), Some(minor), Some(ratio))
                if major > 0.0
                    && minor > 0.0
                    && ratio > 0.0
                    && (minor - major / ratio).abs() <= 1e-9 => {}
            _ => issues.push(format!(
                "preferred_memory_color_guard_family_{name}_ellipse_invalid"
            )),
        }
        if family.ellipse_rotation_degrees.is_none() {
            issues.push(format!(
                "preferred_memory_color_guard_family_{name}_rotation_missing"
            ));
        }
        match (family.matched_pixel_ratio, family.limited_pixel_ratio) {
            (Some(matched), Some(limited))
                if (0.0..=1.0).contains(&matched)
                    && (0.0..=1.0).contains(&limited)
                    && limited <= matched + 1e-12 =>
            {
                family_matched_sum += matched;
                family_limited_sum += limited;
            }
            _ => issues.push(format!(
                "preferred_memory_color_guard_family_{name}_population_invalid"
            )),
        }
        match (family.mean_scale_reduction, family.max_scale_reduction) {
            (Some(mean), Some(maximum))
                if mean >= 0.0 && maximum >= 0.0 && mean <= maximum + 1e-12 => {}
            _ => issues.push(format!(
                "preferred_memory_color_guard_family_{name}_scale_reduction_invalid"
            )),
        }
    }
    if guard
        .matched_pixel_ratio
        .is_some_and(|overall| (family_matched_sum - overall).abs() > 1e-9)
    {
        issues.push("preferred_memory_color_guard_family_matched_sum_mismatch".to_string());
    }
    if guard
        .limited_pixel_ratio
        .is_some_and(|overall| (family_limited_sum - overall).abs() > 1e-9)
    {
        issues.push("preferred_memory_color_guard_family_limited_sum_mismatch".to_string());
    }

    issues
}

/// Checks the bounded preferred-skin display shoulder for coherent provenance, model geometry,
/// populations, and effect accounting. Legacy reports without the optional object remain readable.
pub fn preferred_skin_rendering_consistency_issues(tone: &ToneValidationSummary) -> Vec<String> {
    let Some(rendering) = tone.preferred_skin_rendering.as_ref() else {
        return Vec::new();
    };
    let mut issues = Vec::new();
    if rendering.enabled.is_none() {
        issues.push("preferred_skin_rendering_enabled_missing".to_string());
    }
    for (field, value) in [
        ("reason", rendering.reason.as_deref()),
        ("method", rendering.method.as_deref()),
        ("working_space", rendering.working_space.as_deref()),
        (
            "preference_reference",
            rendering.preference_reference.as_deref(),
        ),
        ("support_reference", rendering.support_reference.as_deref()),
        ("interpretation", rendering.interpretation.as_deref()),
    ] {
        if value.is_none_or(str::is_empty) {
            issues.push(format!("preferred_skin_rendering_{field}_missing"));
        }
    }

    let center_lab = rendering.preferred_center_lab.as_deref();
    let center_lch = rendering.preferred_center_lch.as_deref();
    if center_lab
        .is_none_or(|values| values.len() != 3 || values.iter().any(|value| !value.is_finite()))
        || center_lch
            .is_none_or(|values| values.len() != 3 || values.iter().any(|value| !value.is_finite()))
    {
        issues.push("preferred_skin_rendering_center_invalid".to_string());
    } else if let (Some(lab), Some(lch)) = (center_lab, center_lch) {
        let hue = lch[2].to_radians();
        if (lab[0] - lch[0]).abs() > 1e-9
            || (lab[1] - lch[1] * hue.cos()).abs() > 1e-9
            || (lab[2] - lch[1] * hue.sin()).abs() > 1e-9
        {
            issues.push("preferred_skin_rendering_center_lab_lch_mismatch".to_string());
        }
    }
    match (
        rendering.semi_major_axis_ab,
        rendering.semi_minor_axis_ab,
        rendering.axis_ratio,
    ) {
        (Some(major), Some(minor), Some(ratio))
            if major.is_finite()
                && minor.is_finite()
                && ratio.is_finite()
                && major > 0.0
                && minor > 0.0
                && ratio > 0.0
                && (minor - major / ratio).abs() <= 1e-9 => {}
        _ => issues.push("preferred_skin_rendering_ellipse_invalid".to_string()),
    }
    if rendering
        .ellipse_rotation_degrees
        .is_none_or(|value| !value.is_finite())
    {
        issues.push("preferred_skin_rendering_rotation_invalid".to_string());
    }
    match (
        rendering.core_normalized_radius,
        rendering.radial_excess_reduction,
        rendering.maximum_delta_e_ab,
        rendering.minimum_support_weight,
    ) {
        (Some(core), Some(reduction), Some(maximum), Some(support))
            if core.is_finite()
                && reduction.is_finite()
                && maximum.is_finite()
                && support.is_finite()
                && core > 0.0
                && reduction > 0.0
                && reduction <= 1.0
                && maximum > 0.0
                && (0.0..=1.0).contains(&support) => {}
        _ => issues.push("preferred_skin_rendering_policy_bounds_invalid".to_string()),
    }

    let ratio_fields = [
        ("evaluated", rendering.evaluated_pixel_ratio),
        ("matched", rendering.matched_pixel_ratio),
        (
            "outside_preferred_core",
            rendering.outside_preferred_core_ratio,
        ),
        ("adjusted", rendering.adjusted_pixel_ratio),
        ("gamut_limited", rendering.gamut_limited_pixel_ratio),
    ];
    for (field, value) in ratio_fields {
        match value {
            Some(value) if value.is_finite() && (0.0..=1.0).contains(&value) => {}
            Some(_) => issues.push(format!(
                "preferred_skin_rendering_{field}_pixel_ratio_invalid"
            )),
            None => issues.push(format!(
                "preferred_skin_rendering_{field}_pixel_ratio_missing"
            )),
        }
    }
    for (issue, smaller, larger) in [
        (
            "preferred_skin_rendering_matched_exceeds_evaluated",
            rendering.matched_pixel_ratio,
            rendering.evaluated_pixel_ratio,
        ),
        (
            "preferred_skin_rendering_outside_core_exceeds_matched",
            rendering.outside_preferred_core_ratio,
            rendering.matched_pixel_ratio,
        ),
        (
            "preferred_skin_rendering_adjusted_exceeds_outside_core",
            rendering.adjusted_pixel_ratio,
            rendering.outside_preferred_core_ratio,
        ),
        (
            "preferred_skin_rendering_gamut_limited_exceeds_adjusted",
            rendering.gamut_limited_pixel_ratio,
            rendering.adjusted_pixel_ratio,
        ),
    ] {
        if smaller
            .zip(larger)
            .is_some_and(|(smaller, larger)| smaller > larger + 1e-12)
        {
            issues.push(issue.to_string());
        }
    }

    let nonnegative_effect_pairs = [
        (
            "delta_e_ab",
            rendering.mean_delta_e_ab,
            rendering.max_delta_e_ab,
        ),
        (
            "abs_hue_shift_degrees",
            rendering.mean_abs_hue_shift_degrees,
            rendering.max_abs_hue_shift_degrees,
        ),
    ];
    for (field, mean, maximum) in nonnegative_effect_pairs {
        match (mean, maximum) {
            (Some(mean), Some(maximum))
                if mean.is_finite()
                    && maximum.is_finite()
                    && mean >= 0.0
                    && maximum >= 0.0
                    && mean <= maximum + 1e-12 => {}
            _ => issues.push(format!("preferred_skin_rendering_{field}_invalid")),
        }
    }
    if rendering
        .max_delta_e_ab
        .zip(rendering.maximum_delta_e_ab)
        .is_some_and(|(observed, configured)| observed > configured + 1e-9)
    {
        issues.push("preferred_skin_rendering_delta_e_cap_exceeded".to_string());
    }
    if rendering
        .max_abs_hue_shift_degrees
        .is_some_and(|value| value > 180.0 + 1e-9)
    {
        issues.push("preferred_skin_rendering_hue_shift_exceeds_180".to_string());
    }
    match (rendering.mean_chroma_delta, rendering.max_abs_chroma_delta) {
        (Some(mean), Some(maximum))
            if mean.is_finite()
                && maximum.is_finite()
                && maximum >= 0.0
                && mean.abs() <= maximum + 1e-12 => {}
        _ => issues.push("preferred_skin_rendering_chroma_effect_invalid".to_string()),
    }
    if rendering.enabled == Some(true)
        && rendering
            .mean_chroma_delta
            .is_some_and(|value| value > 1e-12)
    {
        issues.push("preferred_skin_rendering_chroma_increase_detected".to_string());
    }

    let effect_values = [
        rendering.mean_delta_e_ab,
        rendering.max_delta_e_ab,
        rendering.mean_abs_hue_shift_degrees,
        rendering.max_abs_hue_shift_degrees,
        rendering.mean_chroma_delta,
        rendering.max_abs_chroma_delta,
    ];
    if rendering.enabled == Some(false)
        && ratio_fields
            .into_iter()
            .filter_map(|(_, value)| value)
            .chain(effect_values.into_iter().flatten())
            .any(|value| value.abs() > 1e-12)
    {
        issues.push("preferred_skin_rendering_disabled_with_nonzero_effect".to_string());
    }
    if rendering.adjusted_pixel_ratio == Some(0.0)
        && effect_values
            .into_iter()
            .flatten()
            .any(|value| value.abs() > 1e-12)
    {
        issues.push("preferred_skin_rendering_zero_adjusted_with_nonzero_effect".to_string());
    }
    if rendering.enabled == Some(true) && tone.color_trust_state.as_deref() != Some("trusted") {
        issues.push("preferred_skin_rendering_enabled_without_trusted_color".to_string());
    }

    issues
}

/// Checks that optional grain-reduction evidence describes a selective, bounded operation.
/// Legacy reports without any grain fields remain readable, while a current enabled pass must
/// expose its exact structure-exclusion policy and coherent pre/post detail decision.
pub fn grain_reduction_consistency_issues(tone: &ToneValidationSummary) -> Vec<String> {
    let grain_fields_present = tone.noise_reduction_enabled.is_some()
        || tone.noise_reduction_requested_enabled.is_some()
        || tone.noise_reduction_applied_ratio.is_some()
        || tone.noise_reduction_structure_excluded_ratio.is_some()
        || tone.grain_detail_retention.is_some();
    if !grain_fields_present {
        return Vec::new();
    }

    let mut issues = Vec::new();
    let enabled = tone.noise_reduction_enabled;
    if enabled.is_none() {
        issues.push("grain_reduction_enabled_missing".to_string());
    }
    for (field, value) in [
        ("applied", tone.noise_reduction_applied_ratio),
        (
            "structure_excluded",
            tone.noise_reduction_structure_excluded_ratio,
        ),
        (
            "texture_limited",
            tone.noise_reduction_texture_limited_ratio,
        ),
        (
            "saturation_limited",
            tone.noise_reduction_saturation_limited_ratio,
        ),
    ] {
        match value {
            Some(value) if value.is_finite() && (0.0..=1.0).contains(&value) => {}
            Some(_) => issues.push(format!("grain_reduction_{field}_ratio_invalid")),
            None if enabled == Some(true) => {
                issues.push(format!("grain_reduction_{field}_ratio_missing"));
            }
            None => {}
        }
    }

    match (
        tone.noise_reduction_structure_gate_start,
        tone.noise_reduction_structure_gate_end,
    ) {
        (Some(start), Some(end))
            if start.is_finite() && end.is_finite() && start >= 0.0 && start < end => {}
        (None, None) if enabled != Some(true) => {}
        _ => issues.push("grain_reduction_structure_gate_invalid".to_string()),
    }
    if tone
        .noise_reduction_applied_ratio
        .zip(tone.noise_reduction_structure_excluded_ratio)
        .is_some_and(|(applied, excluded)| applied + excluded > 1.0 + 1e-12)
    {
        issues.push("grain_reduction_applied_and_excluded_overlap".to_string());
    }
    if tone
        .grain_detail_retention
        .as_ref()
        .and_then(|detail| detail.luminance_contrast_threshold)
        .zip(tone.noise_reduction_structure_gate_end)
        .is_some_and(|(detail_floor, gate_end)| (detail_floor - gate_end).abs() > 1e-12)
    {
        issues.push("grain_reduction_structure_gate_detail_floor_mismatch".to_string());
    }

    for (field, value) in [
        (
            "mean_abs_chroma_delta",
            tone.noise_reduction_mean_abs_chroma_delta,
        ),
        (
            "max_abs_chroma_delta",
            tone.noise_reduction_max_abs_chroma_delta,
        ),
        (
            "mean_abs_luma_delta",
            tone.noise_reduction_mean_abs_luma_delta,
        ),
        (
            "max_abs_luma_delta",
            tone.noise_reduction_max_abs_luma_delta,
        ),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            issues.push(format!("grain_reduction_{field}_invalid"));
        } else if enabled == Some(true) && value.is_none() {
            issues.push(format!("grain_reduction_{field}_missing"));
        }
    }
    if tone
        .noise_reduction_mean_abs_chroma_delta
        .zip(tone.noise_reduction_max_abs_chroma_delta)
        .is_some_and(|(mean, maximum)| mean > maximum + 1e-12)
    {
        issues.push("grain_reduction_chroma_delta_mean_exceeds_max".to_string());
    }
    if tone
        .noise_reduction_mean_abs_luma_delta
        .zip(tone.noise_reduction_max_abs_luma_delta)
        .is_some_and(|(mean, maximum)| mean > maximum + 1e-12)
    {
        issues.push("grain_reduction_luma_delta_mean_exceeds_max".to_string());
    }

    if enabled == Some(true) {
        if tone.noise_reduction_requested_enabled != Some(true) {
            issues.push("grain_reduction_enabled_without_request".to_string());
        }
        if tone
            .noise_reduction_requested_strength
            .is_none_or(|value| !value.is_finite() || value <= 0.0 || value > 1.0)
        {
            issues.push("grain_reduction_enabled_strength_invalid".to_string());
        }
        if tone
            .noise_reduction_requested_scale
            .is_none_or(|value| !value.is_finite() || !(0.5..=4.0).contains(&value))
        {
            issues.push("grain_reduction_enabled_scale_invalid".to_string());
        }
        if tone
            .noise_reduction_reason
            .as_deref()
            .is_none_or(str::is_empty)
        {
            issues.push("grain_reduction_reason_missing".to_string());
        }
        match tone.grain_detail_retention.as_ref() {
            Some(detail) if detail.evaluated == Some(true) => {}
            _ => issues.push("grain_reduction_detail_retention_not_evaluated".to_string()),
        }
    } else if enabled == Some(false) {
        let nonzero_effect = [
            tone.noise_reduction_applied_ratio,
            tone.noise_reduction_structure_excluded_ratio,
            tone.noise_reduction_mean_abs_chroma_delta,
            tone.noise_reduction_max_abs_chroma_delta,
            tone.noise_reduction_mean_abs_luma_delta,
            tone.noise_reduction_max_abs_luma_delta,
        ]
        .into_iter()
        .flatten()
        .any(|value| value.abs() > 1e-12);
        if nonzero_effect {
            issues.push("grain_reduction_disabled_with_nonzero_effect".to_string());
        }
    }

    issues
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct NeutralSafetyRescueSummary {
    pub evaluated: Option<bool>,
    pub applied: Option<bool>,
    pub matrix_candidate: Option<String>,
    pub matrix_candidate_kind: Option<String>,
    pub matrix_anchor_evidence_supported: Option<bool>,
    pub neutral_estimate_supported: Option<bool>,
    pub neutral_model_evidence_supported: Option<bool>,
    pub matrix_pre_scale_preserved_ratio: Option<f64>,
    pub neutral_pre_scale_preserved_ratio: Option<f64>,
    pub preserved_ratio_gain: Option<f64>,
    pub minimum_preserved_ratio: Option<f64>,
    pub minimum_preserved_ratio_gain: Option<f64>,
    pub matrix_midtone_saturation_p95: Option<f64>,
    pub neutral_midtone_saturation_p95: Option<f64>,
    pub midtone_saturation_p95_reduction: Option<f64>,
    pub maximum_midtone_saturation_p95: Option<f64>,
    pub minimum_midtone_saturation_p95_reduction: Option<f64>,
    pub matrix_memory_color_penalty: Option<f64>,
    pub neutral_memory_color_penalty: Option<f64>,
    pub matrix_spatial_consistency_penalty: Option<f64>,
    pub neutral_spatial_consistency_penalty: Option<f64>,
    pub neutral_saturation_preservation_sample_count: Option<usize>,
    pub neutral_saturation_preservation_p05_ratio: Option<f64>,
    pub neutral_saturation_preservation_median_ratio: Option<f64>,
    pub neutral_saturation_preservation_p95_ratio: Option<f64>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColorDebugArtifactSummary {
    pub kind: String,
    pub path: String,
    pub status: String,
    pub exists: bool,
    pub fresh_for_report: Option<bool>,
    pub file_size_bytes: Option<u64>,
    pub modified_at: Option<String>,
    pub modified_at_unix_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColorCandidateQualitySummary {
    pub candidate: Option<String>,
    pub rank: Option<usize>,
    pub selected: Option<bool>,
    pub quality_score: Option<f64>,
    pub technical_safety_score: Option<f64>,
    pub color_fidelity_score: Option<f64>,
    pub selected_quality_delta: Option<f64>,
    pub rejected: Option<bool>,
    pub rejection_reason: Option<String>,
    pub pre_scale_clipped_low_total: Option<f64>,
    pub pre_scale_clipped_low_max: Option<f64>,
    pub pre_scale_clipped_high_total: Option<f64>,
    pub pre_scale_clipped_high_max: Option<f64>,
    pub reference_patch_rms_error: Option<f64>,
    pub reference_patch_rms_delta_e: Option<f64>,
    #[serde(default)]
    pub reference_patch_rms_delta_e2000: Option<f64>,
    pub reference_patch_delta_vs_image_derived: Option<f64>,
    #[serde(default)]
    pub reference_patch_max_delta_vs_image_derived: Option<f64>,
    pub reference_patch_delta_e_delta_vs_image_derived: Option<f64>,
    #[serde(default)]
    pub reference_patch_delta_e_max_delta_vs_image_derived: Option<f64>,
    #[serde(default)]
    pub reference_patch_delta_e2000_delta_vs_image_derived: Option<f64>,
    #[serde(default)]
    pub reference_patch_delta_e2000_max_delta_vs_image_derived: Option<f64>,
    pub reference_patch_regresses_image_derived: Option<bool>,
    pub dominant_anchor_quality_score: Option<f64>,
    pub dominant_anchor_unstable_channels: Option<Vec<bool>>,
    pub density_monotonicity_score: Option<f64>,
    pub hue_linearity_score: Option<f64>,
    pub saturation_preservation_median_ratio: Option<f64>,
    pub spatial_neutral_delta_p95: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_components: Option<ColorQualityComponentsSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColorQualityComponentsSummary {
    pub low_gamut_clip_penalty: Option<f64>,
    pub high_gamut_clip_penalty: Option<f64>,
    pub preserved_gamut_penalty: Option<f64>,
    pub exposure_penalty: Option<f64>,
    pub neutral_balance_penalty: Option<f64>,
    pub neutral_estimate_penalty: Option<f64>,
    pub anchor_support_penalty: Option<f64>,
    pub anchor_stability_penalty: Option<f64>,
    pub condition_penalty: Option<f64>,
    pub calibration_confidence_penalty: Option<f64>,
    pub target_residual_penalty: Option<f64>,
    pub rendered_tone_penalty: Option<f64>,
    pub tone_chroma_cleanup_penalty: Option<f64>,
    pub density_monotonicity_penalty: Option<f64>,
    pub hue_linearity_penalty: Option<f64>,
    pub saturation_preservation_penalty: Option<f64>,
    pub memory_color_penalty: Option<f64>,
    pub spatial_consistency_penalty: Option<f64>,
    pub fallback_penalty: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColorCandidateAcceptanceSummary {
    pub candidate: Option<String>,
    pub candidate_kind: Option<String>,
    #[serde(default)]
    pub mapping_strategy: Option<String>,
    #[serde(default)]
    pub source_label: Option<String>,
    pub status: Option<String>,
    pub reason: Option<String>,
    pub rank: Option<usize>,
    pub selected: Option<bool>,
    #[serde(default)]
    pub eligible_in_color_mode: Option<bool>,
    pub quality_score: Option<f64>,
    pub selected_quality_delta: Option<f64>,
    pub rejected: Option<bool>,
    pub rejection_reason: Option<String>,
    pub beats_image_derived: Option<bool>,
    pub within_negative_gamut_limits: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeutralEstimateQualitySummary {
    pub score: Option<f64>,
    pub accepted: Option<bool>,
    pub broad_support: Option<bool>,
    pub reason: Option<String>,
    pub sample_count: Option<usize>,
    pub populated_band_count: Option<usize>,
    pub dominant_band_fraction: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeutralSampleRejectionSummary {
    pub total_pixels: Option<usize>,
    pub accepted_neutral_samples: Option<usize>,
    pub clipped: Option<usize>,
    pub border: Option<usize>,
    pub film_base_like_edge: Option<usize>,
    pub dust: Option<usize>,
    pub luma_out_of_range: Option<usize>,
    pub chroma_threshold: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DominantAnchorSampleRejectionSummary {
    pub total_pixels: Option<usize>,
    pub accepted_anchor_samples: Option<usize>,
    pub clipped: Option<usize>,
    pub border: Option<usize>,
    pub film_base_like_edge: Option<usize>,
    pub dust: Option<usize>,
    pub luma_out_of_range: Option<usize>,
    pub low_saturation: Option<usize>,
    pub weak_dominance: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DominantAnchorQualitySummary {
    pub score: Option<f64>,
    pub accepted: Option<bool>,
    pub reason: Option<String>,
    pub channel_populated_band_count: Option<Vec<usize>>,
    pub channel_dominant_band_fraction: Option<Vec<f64>>,
    pub channel_mean_dominance_margin: Option<Vec<f64>>,
    pub channel_stability_score: Option<Vec<f64>>,
    pub channel_unstable: Option<Vec<bool>>,
    pub unstable_channel_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReferencePatchEvaluationSummary {
    pub patch_count: Option<usize>,
    pub selected_candidate: Option<String>,
    pub selected_rms_error: Option<f64>,
    pub selected_rms_delta_e: Option<f64>,
    #[serde(default)]
    pub selected_rms_delta_e2000: Option<f64>,
    pub image_derived_rms_error: Option<f64>,
    pub image_derived_rms_delta_e: Option<f64>,
    #[serde(default)]
    pub image_derived_rms_delta_e2000: Option<f64>,
    pub rms_error_delta_vs_image_derived: Option<f64>,
    #[serde(default)]
    pub max_error_delta_vs_image_derived: Option<f64>,
    pub delta_e_rms_delta_vs_image_derived: Option<f64>,
    #[serde(default)]
    pub delta_e_max_delta_vs_image_derived: Option<f64>,
    #[serde(default)]
    pub delta_e2000_rms_delta_vs_image_derived: Option<f64>,
    #[serde(default)]
    pub delta_e2000_max_delta_vs_image_derived: Option<f64>,
    pub selected_regresses_image_derived: Option<bool>,
    pub worst_hue_families: Vec<String>,
    pub hue_family_regressions: Vec<String>,
    pub regressed_candidates: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeutralTrimBeforeAfterSummary {
    pub applied: Option<bool>,
    pub reason: Option<String>,
    pub before_neutral_delta_magnitude: Option<f64>,
    pub after_neutral_delta_magnitude: Option<f64>,
    pub before_neutral_band_delta_magnitude: Option<Vec<Option<f64>>>,
    pub after_neutral_band_delta_magnitude: Option<Vec<Option<f64>>>,
    pub neutral_delta_reduced: Option<bool>,
    pub neutral_band_delta_worsened: Option<bool>,
    pub low_clipping_increased: Option<bool>,
    pub high_clipping_increased: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationAcceptanceSummary {
    pub status: Option<String>,
    pub reason: Option<String>,
    pub preferred_candidate: Option<String>,
    pub preferred_candidate_quality_score: Option<f64>,
    pub image_derived_quality_score: Option<f64>,
    pub beats_image_derived: Option<bool>,
    pub within_negative_gamut_limits: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdaptiveVibranceSkinMemoryProtectionSummary {
    pub enabled: Option<bool>,
    pub method: Option<String>,
    pub working_space: Option<String>,
    pub reference: Option<String>,
    pub core_lightness: Option<Vec<f64>>,
    pub support_lightness: Option<Vec<f64>>,
    pub core_chroma: Option<Vec<f64>>,
    pub support_chroma: Option<Vec<f64>>,
    pub core_hue_degrees: Option<Vec<f64>>,
    pub support_hue_degrees: Option<Vec<f64>>,
    pub maximum_vibrance_reduction: Option<f64>,
    pub evaluated_pixel_ratio: Option<f64>,
    pub protected_pixel_ratio: Option<f64>,
    pub mean_protection_weight: Option<f64>,
    pub max_protection_weight: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdaptiveVibrancePreferredMemoryColorFamilySummary {
    pub family: Option<String>,
    pub preferred_center_lab: Option<Vec<f64>>,
    pub preferred_center_lch: Option<Vec<f64>>,
    pub semi_major_axis_ab: Option<f64>,
    pub semi_minor_axis_ab: Option<f64>,
    pub axis_ratio: Option<f64>,
    pub ellipse_rotation_degrees: Option<f64>,
    pub matched_pixel_ratio: Option<f64>,
    pub limited_pixel_ratio: Option<f64>,
    pub mean_scale_reduction: Option<f64>,
    pub max_scale_reduction: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdaptiveVibrancePreferredMemoryColorGuardSummary {
    pub enabled: Option<bool>,
    pub method: Option<String>,
    pub working_space: Option<String>,
    pub reference: Option<String>,
    pub interpretation: Option<String>,
    pub core_normalized_radius: Option<f64>,
    pub support_normalized_radius: Option<f64>,
    pub evaluated_pixel_ratio: Option<f64>,
    pub matched_pixel_ratio: Option<f64>,
    pub limited_pixel_ratio: Option<f64>,
    pub mean_scale_reduction: Option<f64>,
    pub max_scale_reduction: Option<f64>,
    #[serde(default)]
    pub families: Vec<AdaptiveVibrancePreferredMemoryColorFamilySummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreferredSkinRenderingSummary {
    pub enabled: Option<bool>,
    pub reason: Option<String>,
    pub method: Option<String>,
    pub working_space: Option<String>,
    pub preference_reference: Option<String>,
    pub support_reference: Option<String>,
    pub interpretation: Option<String>,
    pub preferred_center_lab: Option<Vec<f64>>,
    pub preferred_center_lch: Option<Vec<f64>>,
    pub semi_major_axis_ab: Option<f64>,
    pub semi_minor_axis_ab: Option<f64>,
    pub axis_ratio: Option<f64>,
    pub ellipse_rotation_degrees: Option<f64>,
    pub core_normalized_radius: Option<f64>,
    pub radial_excess_reduction: Option<f64>,
    pub maximum_delta_e_ab: Option<f64>,
    pub minimum_support_weight: Option<f64>,
    pub evaluated_pixel_ratio: Option<f64>,
    pub matched_pixel_ratio: Option<f64>,
    pub outside_preferred_core_ratio: Option<f64>,
    pub adjusted_pixel_ratio: Option<f64>,
    pub gamut_limited_pixel_ratio: Option<f64>,
    pub mean_delta_e_ab: Option<f64>,
    pub max_delta_e_ab: Option<f64>,
    pub mean_abs_hue_shift_degrees: Option<f64>,
    pub max_abs_hue_shift_degrees: Option<f64>,
    pub mean_chroma_delta: Option<f64>,
    pub max_abs_chroma_delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToneValidationSummary {
    #[serde(default)]
    pub tone_confidence_status: Option<String>,
    #[serde(default)]
    pub tone_output_evidence_evaluated: Option<bool>,
    #[serde(default)]
    pub tone_output_evidence_confidence: Option<f64>,
    #[serde(default)]
    pub tone_output_confidence_status: Option<String>,
    #[serde(default)]
    pub tone_output_review_required: Option<bool>,
    #[serde(default)]
    pub tone_output_review_reason: Option<String>,
    #[serde(default)]
    pub confidence_limited_by_tone_output_evidence: Option<bool>,
    #[serde(default)]
    pub input_luminance_range_p05_p95: Option<f64>,
    #[serde(default)]
    pub mapped_luminance_range_p05_p95: Option<f64>,
    #[serde(default)]
    pub render_to_mapped_luminance_range_ratio: Option<f64>,
    #[serde(default)]
    pub maximum_post_tone_high_clip_ratio: Option<f64>,
    #[serde(default)]
    pub maximum_post_tone_low_clip_ratio: Option<f64>,
    pub highlight_chroma_compressed_ratio: Option<f64>,
    pub highlight_neutral_chroma_compressed_ratio: Option<f64>,
    pub highlight_neutral_chroma_enabled: Option<bool>,
    pub shadow_chroma_compressed_ratio: Option<f64>,
    pub shadow_chroma_enabled: Option<bool>,
    pub color_protection_policy: Option<String>,
    pub color_trust_state: Option<String>,
    pub color_protection_reason: Option<String>,
    pub shadow_saturation_median: Option<f64>,
    pub shadow_saturation_p95: Option<f64>,
    #[serde(default)]
    pub shadow_rgb_median: Option<Vec<f64>>,
    #[serde(default)]
    pub shadow_visible_pixel_count: Option<usize>,
    #[serde(default)]
    pub shadow_visible_saturation_median: Option<f64>,
    #[serde(default)]
    pub shadow_visible_saturation_p95: Option<f64>,
    #[serde(default)]
    pub shadow_visible_rgb_median: Option<Vec<f64>>,
    pub midtone_saturation_median: Option<f64>,
    pub midtone_saturation_p95: Option<f64>,
    #[serde(default)]
    pub midtone_rgb_median: Option<Vec<f64>>,
    #[serde(default)]
    pub render_luminance_percentiles: Option<Vec<f64>>,
    #[serde(default)]
    pub render_luminance_range_p05_p95: Option<f64>,
    pub midtone_luminance_percentiles: Option<Vec<f64>>,
    #[serde(default)]
    pub midtone_neutral_pixel_count: Option<usize>,
    #[serde(default)]
    pub midtone_neutral_saturation_median: Option<f64>,
    #[serde(default)]
    pub midtone_neutral_saturation_p95: Option<f64>,
    #[serde(default)]
    pub midtone_neutral_rgb_median: Option<Vec<f64>>,
    #[serde(default)]
    pub midtone_saturated_saturation_median: Option<f64>,
    #[serde(default)]
    pub midtone_saturated_saturation_p95: Option<f64>,
    pub bright_neutral_saturation_median: Option<f64>,
    pub bright_neutral_saturation_p95: Option<f64>,
    #[serde(default)]
    pub bright_neutral_rgb_median: Option<Vec<f64>>,
    pub bright_saturated_saturation_median: Option<f64>,
    pub bright_saturated_saturation_p95: Option<f64>,
    pub post_chroma_compression_clipped_high_ratio: Option<Vec<f64>>,
    pub post_chroma_compression_clipped_low_ratio: Option<Vec<f64>>,
    #[serde(default)]
    pub perceptual_gamut_mapping_space: Option<String>,
    #[serde(default)]
    pub perceptual_gamut_mapped_ratio: Option<f64>,
    #[serde(default)]
    pub perceptual_gamut_mean_chroma_scale: Option<f64>,
    #[serde(default)]
    pub perceptual_gamut_min_chroma_scale: Option<f64>,
    #[serde(default)]
    pub adaptive_vibrance_skin_memory_protection:
        Option<AdaptiveVibranceSkinMemoryProtectionSummary>,
    #[serde(default)]
    pub adaptive_vibrance_preferred_memory_color_guard:
        Option<AdaptiveVibrancePreferredMemoryColorGuardSummary>,
    #[serde(default)]
    pub preferred_skin_rendering: Option<PreferredSkinRenderingSummary>,
    #[serde(default)]
    pub noise_reduction_enabled: Option<bool>,
    #[serde(default)]
    pub noise_reduction_requested_enabled: Option<bool>,
    #[serde(default)]
    pub noise_reduction_reason: Option<String>,
    #[serde(default)]
    pub noise_reduction_requested_strength: Option<f64>,
    #[serde(default)]
    pub noise_reduction_requested_scale: Option<f64>,
    #[serde(default)]
    pub noise_reduction_radius: Option<usize>,
    #[serde(default)]
    pub noise_reduction_chroma_amount: Option<f64>,
    #[serde(default)]
    pub noise_reduction_luma_amount: Option<f64>,
    #[serde(default)]
    pub noise_reduction_applied_ratio: Option<f64>,
    #[serde(default)]
    pub noise_reduction_structure_gate_start: Option<f64>,
    #[serde(default)]
    pub noise_reduction_structure_gate_end: Option<f64>,
    #[serde(default)]
    pub noise_reduction_structure_excluded_ratio: Option<f64>,
    #[serde(default)]
    pub noise_reduction_texture_limited_ratio: Option<f64>,
    #[serde(default)]
    pub noise_reduction_saturation_limited_ratio: Option<f64>,
    #[serde(default)]
    pub noise_reduction_mean_abs_chroma_delta: Option<f64>,
    #[serde(default)]
    pub noise_reduction_max_abs_chroma_delta: Option<f64>,
    #[serde(default)]
    pub noise_reduction_mean_abs_luma_delta: Option<f64>,
    #[serde(default)]
    pub noise_reduction_max_abs_luma_delta: Option<f64>,
    #[serde(default)]
    pub noise_reduction_flat_luma_p95_reduction_ratio: Option<f64>,
    #[serde(default)]
    pub noise_reduction_flat_chroma_p95_reduction_ratio: Option<f64>,
    #[serde(default)]
    pub grain_detail_retention: Option<GrainDetailRetentionValidationSummary>,
    pub high_frequency_grain: Option<HighFrequencyGrainSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrainDetailRetentionValidationSummary {
    pub method: Option<String>,
    pub evaluated: Option<bool>,
    pub decision_supported: Option<bool>,
    pub sample_stride: Option<usize>,
    pub probe_radius: Option<usize>,
    pub minimum_probe_count: Option<usize>,
    pub luminance_probe_count: Option<usize>,
    pub chroma_probe_count: Option<usize>,
    pub luminance_decision_supported: Option<bool>,
    pub chroma_decision_supported: Option<bool>,
    pub luminance_median_retention: Option<f64>,
    pub luminance_p10_retention: Option<f64>,
    pub chroma_median_retention: Option<f64>,
    pub chroma_p10_retention: Option<f64>,
    pub luminance_contrast_threshold: Option<f64>,
    pub chroma_contrast_threshold: Option<f64>,
    pub coherence_threshold: Option<f64>,
    pub median_retention_threshold: Option<f64>,
    pub p10_retention_threshold: Option<f64>,
    pub review_required: Option<bool>,
    pub reason: Option<String>,
    pub review_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhaseWarnings {
    pub phase: String,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SyntheticColorSuiteSummary {
    pub status: String,
    pub issues: Vec<String>,
    pub cases: Vec<SyntheticColorCaseSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SyntheticColorCaseSummary {
    pub name: String,
    pub status: String,
    pub expected_render_input_source: Option<String>,
    pub actual_render_input_source: Option<String>,
    pub expected_render_input_fallback_used: Option<bool>,
    pub actual_render_input_fallback_used: Option<bool>,
    pub expected_render_input_reason_contains: Option<String>,
    pub actual_render_input_reason: Option<String>,
    pub expected_selected_candidate: Option<String>,
    pub actual_selected_candidate: Option<String>,
    pub expected_mapping_strategy: Option<String>,
    pub actual_mapping_strategy: Option<String>,
    pub expected_gamut_fallback_used: Option<bool>,
    pub actual_gamut_fallback_used: Option<bool>,
    pub expected_image_matrix_pre_scale_low_clip_total_min: Option<f64>,
    pub actual_image_matrix_pre_scale_low_clip_total: Option<f64>,
    pub expected_selected_pre_scale_low_clip_total_max: Option<f64>,
    pub actual_selected_pre_scale_low_clip_total: Option<f64>,
    pub expected_post_scale_low_clip_total_max: Option<f64>,
    pub actual_post_scale_low_clip_total: Option<f64>,
    pub expected_calibration_acceptance_status: Option<String>,
    pub actual_calibration_acceptance_status: Option<String>,
    pub expected_calibration_beats_image_derived: Option<bool>,
    pub actual_calibration_beats_image_derived: Option<bool>,
    pub expected_calibrated_candidate_status: Option<String>,
    pub actual_calibrated_candidate_status: Option<String>,
    pub expected_preferred_calibration_candidate_status: Option<String>,
    pub actual_preferred_calibration_candidate_status: Option<String>,
    pub expected_candidate_risk: Option<String>,
    pub actual_candidate_risk: Option<String>,
    pub expected_tone_color_trust_state: Option<String>,
    pub actual_tone_color_trust_state: Option<String>,
    pub expected_tone_policy: Option<String>,
    pub actual_tone_policy: Option<String>,
    pub expected_tone_policy_color_trust_state: Option<String>,
    pub actual_tone_policy_color_trust_state: Option<String>,
    pub expected_tone_highlight_neutral_chroma_enabled: Option<bool>,
    pub actual_tone_highlight_neutral_chroma_enabled: Option<bool>,
    pub expected_tone_shadow_chroma_enabled: Option<bool>,
    pub actual_tone_shadow_chroma_enabled: Option<bool>,
    pub expected_tone_highlight_chroma_compressed_ratio_min: Option<f64>,
    pub actual_tone_highlight_chroma_compressed_ratio: Option<f64>,
    pub expected_tone_highlight_neutral_chroma_compressed_ratio_max: Option<f64>,
    pub actual_tone_highlight_neutral_chroma_compressed_ratio: Option<f64>,
    pub expected_tone_shadow_chroma_compressed_ratio_max: Option<f64>,
    pub actual_tone_shadow_chroma_compressed_ratio: Option<f64>,
    pub expected_neutral_estimate_accepted: Option<bool>,
    pub actual_neutral_estimate_accepted: Option<bool>,
    pub expected_dominant_anchor_accepted: Option<bool>,
    pub actual_dominant_anchor_accepted: Option<bool>,
    pub expected_dominant_anchor_channel_populated_band_count: Option<[usize; 3]>,
    pub actual_dominant_anchor_channel_populated_band_count: Option<[usize; 3]>,
    pub expected_dominant_anchor_channel_unstable: Option<[bool; 3]>,
    pub actual_dominant_anchor_channel_unstable: Option<[bool; 3]>,
    pub expected_channel_anchor_low_support: Option<[bool; 3]>,
    pub actual_channel_anchor_low_support: Option<[bool; 3]>,
    pub expected_neutral_rejection_minima: Option<SyntheticSampleRejectionMinimums>,
    pub actual_neutral_sample_rejections: Option<NeutralSampleRejectionSummary>,
    pub expected_dominant_anchor_rejection_minima: Option<SyntheticSampleRejectionMinimums>,
    pub actual_dominant_anchor_sample_rejections: Option<DominantAnchorSampleRejectionSummary>,
    pub expected_exposure_scale_min: Option<f64>,
    pub actual_exposure_scale: Option<f64>,
    pub expected_post_scale_high_clip_less_than_pre_scale: Option<bool>,
    pub actual_post_scale_high_clip_less_than_pre_scale: Option<bool>,
    pub actual_pre_scale_high_clip_total: Option<f64>,
    pub actual_post_scale_high_clip_total: Option<f64>,
    pub expected_post_scale_preserved_ratio_min: Option<f64>,
    pub actual_post_scale_preserved_ratio: Option<f64>,
    pub actual_density_monotonicity_score: Option<f64>,
    pub actual_hue_linearity_score: Option<f64>,
    pub actual_saturation_preservation_median_ratio: Option<f64>,
    pub actual_spatial_neutral_delta_p95: Option<f64>,
    pub actual_memory_color_penalty: Option<f64>,
    pub actual_spatial_consistency_penalty: Option<f64>,
    pub selected_quality_score: Option<f64>,
    pub image_derived_quality_score: Option<f64>,
    pub preferred_calibration_quality_score: Option<f64>,
    pub reference_patch_selected_regresses_image_derived: Option<bool>,
    pub expected_reference_patch_regressed_candidates: Option<Vec<String>>,
    pub actual_reference_patch_regressed_candidates: Vec<String>,
    pub expected_reference_patch_hue_family_regressions: Option<Vec<String>>,
    pub actual_reference_patch_hue_family_regressions: Vec<String>,
    pub actual_reference_patch_worst_hue_families: Vec<String>,
    pub expected_error_contains: Option<String>,
    pub actual_error: Option<String>,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SyntheticSampleRejectionMinimums {
    pub clipped: Option<usize>,
    pub border: Option<usize>,
    pub film_base_like_edge: Option<usize>,
    pub dust: Option<usize>,
}

#[derive(Default)]
struct SyntheticCaseExpectations<'a> {
    selected_candidate: Option<&'a str>,
    mapping_strategy: Option<&'a str>,
    gamut_fallback_used: Option<bool>,
    image_matrix_pre_scale_low_clip_total_min: Option<f64>,
    selected_pre_scale_low_clip_total_max: Option<f64>,
    post_scale_low_clip_total_max: Option<f64>,
    calibration_acceptance_status: Option<&'a str>,
    calibration_beats_image_derived: Option<bool>,
    calibrated_candidate_status: Option<&'a str>,
    preferred_calibration_candidate_status: Option<&'a str>,
    candidate_risk: Option<&'a str>,
    tone_color_trust_state: Option<&'a str>,
    neutral_estimate_accepted: Option<bool>,
    dominant_anchor_accepted: Option<bool>,
    dominant_anchor_channel_populated_band_count: Option<[usize; 3]>,
    dominant_anchor_channel_unstable: Option<[bool; 3]>,
    channel_anchor_low_support: Option<[bool; 3]>,
    neutral_rejection_minima: Option<SyntheticSampleRejectionMinimums>,
    dominant_anchor_rejection_minima: Option<SyntheticSampleRejectionMinimums>,
    exposure_scale_min: Option<f64>,
    post_scale_high_clip_less_than_pre_scale: Option<bool>,
    post_scale_preserved_ratio_min: Option<f64>,
    reference_patch_regresses_image_derived: Option<bool>,
    reference_patch_regressed_candidates: Option<Vec<&'a str>>,
    reference_patch_hue_family_regressions: Option<Vec<&'a str>>,
}

pub fn run_synthetic_color_suite() -> SyntheticColorSuiteSummary {
    let cases = vec![
        synthetic_case_calibrated_profile_beats_image_derived(),
        synthetic_case_root_polynomial_selected_inside_measured_support(),
        synthetic_case_root_polynomial_rejected_outside_measured_support(),
        synthetic_case_residual_lut_selected_inside_measured_support(),
        synthetic_case_residual_lut_rejected_outside_measured_rgb_volume(),
        synthetic_case_weak_calibration_rejected(),
        synthetic_case_calibration_neutral_regression_rejected(),
        synthetic_case_unsafe_calibration_rejected_auto(),
        synthetic_case_scanner_prior_beats_weak_anchor_image_candidate(),
        synthetic_case_unsafe_scanner_prior_rejected_auto(),
        synthetic_case_sparse_uncalibrated_anchor_review_required(),
        synthetic_case_dirty_edge_samples_rejected(),
        synthetic_case_biased_scene_anchor_review_required(),
        synthetic_case_high_key_exposure_normalization_preserves_headroom(),
        synthetic_case_destructive_gamut_fallback_preserves_output(),
        synthetic_case_direct_density_render_fallback_for_destructive_ica_gamut(),
        synthetic_case_direct_density_render_fallback_rejects_neutral_regression(),
        synthetic_case_direct_density_render_fallback_for_quality_win(),
        synthetic_case_direct_density_render_fallback_rejects_worse_risk(),
        synthetic_case_tone_policy_weak_neutral_keeps_bounded_neutral_cleanup(),
        synthetic_case_tone_policy_model_review_keeps_bounded_neutral_cleanup(),
        synthetic_case_tone_policy_color_review_disables_color_cleanup_preserves_gamut_repair(),
        synthetic_case_reference_patch_regression_rejected(),
        synthetic_case_forced_unsafe_calibration_fails(),
    ];
    let issues = cases
        .iter()
        .flat_map(|case| {
            case.issues
                .iter()
                .map(|issue| format!("{}:{issue}", case.name))
        })
        .collect::<Vec<_>>();
    let status = if issues.is_empty() {
        "passed"
    } else {
        "failed"
    };
    SyntheticColorSuiteSummary {
        status: status.to_string(),
        issues,
        cases,
    }
}

pub fn synthetic_color_suite_to_markdown(summary: &SyntheticColorSuiteSummary) -> String {
    let mut out = String::new();
    out.push_str("# Synthetic Color Validation Suite\n\n");
    out.push_str(&format!("status: `{}`\n\n", summary.status));
    out.push_str("| Case | Status | Render input | Selected | Mapping | Calibration | Risk | Tone trust | Tone policy | Anchor stability | Exposure | Gamut | Model | Reference | Rejected samples | Issues |\n");
    out.push_str("|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|\n");
    for case in &summary.cases {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            case.name,
            case.status,
            format_synthetic_render_input_evidence(case),
            case.actual_selected_candidate.as_deref().unwrap_or(""),
            case.actual_mapping_strategy.as_deref().unwrap_or(""),
            case.actual_calibration_acceptance_status
                .as_deref()
                .or(case.actual_error.as_deref())
                .unwrap_or(""),
            case.actual_candidate_risk.as_deref().unwrap_or(""),
            case.actual_tone_color_trust_state.as_deref().unwrap_or(""),
            format_synthetic_tone_policy_evidence(case),
            format_synthetic_anchor_stability(case),
            format_synthetic_exposure_evidence(case),
            format_synthetic_gamut_evidence(case),
            format_synthetic_model_evidence(case),
            format_synthetic_reference_evidence(case),
            format_synthetic_rejection_evidence(case),
            if case.issues.is_empty() {
                "none".to_string()
            } else {
                case.issues.join(", ")
            }
        ));
    }
    out
}

fn format_synthetic_anchor_stability(case: &SyntheticColorCaseSummary) -> String {
    match (
        case.actual_dominant_anchor_channel_populated_band_count,
        case.actual_dominant_anchor_channel_unstable,
    ) {
        (None, None) => String::new(),
        (bands, unstable) => format!(
            "bands {:?} unstable {:?}",
            bands.unwrap_or([0; 3]),
            unstable.unwrap_or([false; 3])
        ),
    }
}

fn format_synthetic_render_input_evidence(case: &SyntheticColorCaseSummary) -> String {
    match (
        case.actual_render_input_source.as_deref(),
        case.actual_render_input_fallback_used,
        case.actual_render_input_reason.as_deref(),
    ) {
        (None, None, None) => String::new(),
        (source, fallback, reason) => format!(
            "{} fallback {} {}",
            source.unwrap_or(""),
            fallback.map(|value| value.to_string()).unwrap_or_default(),
            reason.unwrap_or("")
        ),
    }
}

fn format_synthetic_tone_policy_evidence(case: &SyntheticColorCaseSummary) -> String {
    if case.actual_tone_policy.is_none()
        && case.actual_tone_policy_color_trust_state.is_none()
        && case.actual_tone_highlight_neutral_chroma_enabled.is_none()
        && case.actual_tone_shadow_chroma_enabled.is_none()
        && case.actual_tone_highlight_chroma_compressed_ratio.is_none()
        && case
            .actual_tone_highlight_neutral_chroma_compressed_ratio
            .is_none()
        && case.actual_tone_shadow_chroma_compressed_ratio.is_none()
    {
        return String::new();
    }
    format!(
        "{} trust {} hn_enabled {} shadow_enabled {} h_comp {} hn_comp {} sh_comp {}",
        case.actual_tone_policy.as_deref().unwrap_or(""),
        case.actual_tone_policy_color_trust_state
            .as_deref()
            .unwrap_or(""),
        case.actual_tone_highlight_neutral_chroma_enabled
            .map(|value| value.to_string())
            .unwrap_or_default(),
        case.actual_tone_shadow_chroma_enabled
            .map(|value| value.to_string())
            .unwrap_or_default(),
        case.actual_tone_highlight_chroma_compressed_ratio
            .map(|value| format!("{value:.3}"))
            .unwrap_or_default(),
        case.actual_tone_highlight_neutral_chroma_compressed_ratio
            .map(|value| format!("{value:.3}"))
            .unwrap_or_default(),
        case.actual_tone_shadow_chroma_compressed_ratio
            .map(|value| format!("{value:.3}"))
            .unwrap_or_default()
    )
}

fn format_synthetic_reference_evidence(case: &SyntheticColorCaseSummary) -> String {
    if case
        .reference_patch_selected_regresses_image_derived
        .is_none()
        && case.actual_reference_patch_regressed_candidates.is_empty()
        && case
            .actual_reference_patch_hue_family_regressions
            .is_empty()
        && case.actual_reference_patch_worst_hue_families.is_empty()
    {
        return String::new();
    }
    format!(
        "selected_regresses {} regressed {} hue_regressions {} worst {}",
        case.reference_patch_selected_regresses_image_derived
            .map(|value| value.to_string())
            .unwrap_or_default(),
        if case.actual_reference_patch_regressed_candidates.is_empty() {
            "none".to_string()
        } else {
            case.actual_reference_patch_regressed_candidates.join(",")
        },
        if case
            .actual_reference_patch_hue_family_regressions
            .is_empty()
        {
            "none".to_string()
        } else {
            case.actual_reference_patch_hue_family_regressions.join(",")
        },
        if case.actual_reference_patch_worst_hue_families.is_empty() {
            "none".to_string()
        } else {
            case.actual_reference_patch_worst_hue_families.join(",")
        }
    )
}

fn format_synthetic_model_evidence(case: &SyntheticColorCaseSummary) -> String {
    if case.actual_density_monotonicity_score.is_none()
        && case.actual_hue_linearity_score.is_none()
        && case.actual_saturation_preservation_median_ratio.is_none()
        && case.actual_spatial_neutral_delta_p95.is_none()
        && case.actual_memory_color_penalty.is_none()
        && case.actual_spatial_consistency_penalty.is_none()
    {
        return String::new();
    }
    format!(
        "density {} hue {} saturation {} spatial_delta_p95 {} memory_penalty {} spatial_penalty {}",
        case.actual_density_monotonicity_score
            .map(|value| format!("{value:.3}"))
            .unwrap_or_default(),
        case.actual_hue_linearity_score
            .map(|value| format!("{value:.3}"))
            .unwrap_or_default(),
        case.actual_saturation_preservation_median_ratio
            .map(|value| format!("{value:.3}"))
            .unwrap_or_default(),
        case.actual_spatial_neutral_delta_p95
            .map(|value| format!("{value:.3}"))
            .unwrap_or_default(),
        case.actual_memory_color_penalty
            .map(|value| format!("{value:.3}"))
            .unwrap_or_default(),
        case.actual_spatial_consistency_penalty
            .map(|value| format!("{value:.3}"))
            .unwrap_or_default()
    )
}

fn format_synthetic_gamut_evidence(case: &SyntheticColorCaseSummary) -> String {
    match (
        case.actual_gamut_fallback_used,
        case.actual_image_matrix_pre_scale_low_clip_total,
        case.actual_selected_pre_scale_low_clip_total,
        case.actual_post_scale_low_clip_total,
    ) {
        (None, None, None, None) => String::new(),
        (fallback, image_low, selected_low, post_low) => format!(
            "fallback {} image_low {} selected_low {} post_low {}",
            fallback
                .map(|value| value.to_string())
                .unwrap_or_else(|| "".to_string()),
            image_low
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "".to_string()),
            selected_low
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "".to_string()),
            post_low
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "".to_string())
        ),
    }
}

fn format_synthetic_exposure_evidence(case: &SyntheticColorCaseSummary) -> String {
    match (
        case.actual_exposure_scale,
        case.actual_pre_scale_high_clip_total,
        case.actual_post_scale_high_clip_total,
        case.actual_post_scale_preserved_ratio,
    ) {
        (None, None, None, None) => String::new(),
        (scale, pre_high, post_high, preserved) => format!(
            "scale {} high_clip {}->{} preserved {}",
            scale
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "".to_string()),
            pre_high
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "".to_string()),
            post_high
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "".to_string()),
            preserved
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "".to_string())
        ),
    }
}

fn format_synthetic_rejection_evidence(case: &SyntheticColorCaseSummary) -> String {
    let neutral = case
        .actual_neutral_sample_rejections
        .as_ref()
        .map(|rejections| {
            format!(
                "neutral c{} b{} f{} d{}",
                rejections.clipped.unwrap_or(0),
                rejections.border.unwrap_or(0),
                rejections.film_base_like_edge.unwrap_or(0),
                rejections.dust.unwrap_or(0)
            )
        })
        .unwrap_or_default();
    let anchors = case
        .actual_dominant_anchor_sample_rejections
        .as_ref()
        .map(|rejections| {
            format!(
                "anchors c{} b{} f{} d{}",
                rejections.clipped.unwrap_or(0),
                rejections.border.unwrap_or(0),
                rejections.film_base_like_edge.unwrap_or(0),
                rejections.dust.unwrap_or(0)
            )
        })
        .unwrap_or_default();
    if neutral.is_empty() && anchors.is_empty() {
        String::new()
    } else if neutral.is_empty() {
        anchors
    } else if anchors.is_empty() {
        neutral
    } else {
        format!("{neutral}; {anchors}")
    }
}

fn synthetic_case_calibrated_profile_beats_image_derived() -> SyntheticColorCaseSummary {
    let mut img = Array3::<f64>::zeros((24, 24, 3));
    for y in 0..24 {
        for x in 0..24 {
            let pixel = if x < 8 {
                [0.70, 0.20, 0.15]
            } else if x < 16 {
                [0.18, 0.68, 0.22]
            } else {
                [0.16, 0.24, 0.72]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }
    let profile = synthetic_calibration_profile();
    summarize_synthetic_result(
        "calibrated_profile_beats_image_derived",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            colorspace::ColorMode::Auto,
        ),
        SyntheticCaseExpectations {
            selected_candidate: Some("calibrated_direct_profile"),
            mapping_strategy: None,
            gamut_fallback_used: None,
            image_matrix_pre_scale_low_clip_total_min: None,
            selected_pre_scale_low_clip_total_max: None,
            post_scale_low_clip_total_max: None,
            calibration_acceptance_status: Some("accepted"),
            calibration_beats_image_derived: Some(true),
            calibrated_candidate_status: Some("selected"),
            preferred_calibration_candidate_status: Some("selected"),
            candidate_risk: None,
            tone_color_trust_state: None,
            neutral_estimate_accepted: None,
            dominant_anchor_accepted: None,
            dominant_anchor_channel_populated_band_count: None,
            dominant_anchor_channel_unstable: None,
            channel_anchor_low_support: None,
            neutral_rejection_minima: None,
            dominant_anchor_rejection_minima: None,
            exposure_scale_min: None,
            post_scale_high_clip_less_than_pre_scale: None,
            post_scale_preserved_ratio_min: None,
            reference_patch_regresses_image_derived: None,
            reference_patch_regressed_candidates: None,
            reference_patch_hue_family_regressions: None,
        },
    )
}

fn summarize_synthetic_nonlinear_support_result(
    name: &str,
    result: Result<colorspace::ColorspaceMappingResult, String>,
    nonlinear_candidate: &str,
    expected_support_status: &str,
    expected_candidate_rejected: bool,
    expected_selected_candidate: &str,
    expected_mapping_strategy: &str,
) -> SyntheticColorCaseSummary {
    let actual_support_status = result
        .as_ref()
        .ok()
        .and_then(|mapped| mapped.diagnostics.nonlinear_color_model.as_ref())
        .map(|diagnostics| diagnostics.support_status.clone());
    let actual_candidate_rejected = result.as_ref().ok().and_then(|mapped| {
        mapped
            .diagnostics
            .candidate_scores
            .iter()
            .find(|candidate| candidate.candidate == nonlinear_candidate)
            .map(|candidate| candidate.rejected)
    });
    let mut summary = summarize_synthetic_result(
        name,
        result,
        SyntheticCaseExpectations {
            selected_candidate: Some(expected_selected_candidate),
            mapping_strategy: Some(expected_mapping_strategy),
            ..SyntheticCaseExpectations::default()
        },
    );
    if actual_support_status.as_deref() != Some(expected_support_status) {
        summary.issues.push(format!(
            "nonlinear support status expected `{expected_support_status}` got `{actual_support_status:?}`"
        ));
    }
    if actual_candidate_rejected != Some(expected_candidate_rejected) {
        summary.issues.push(format!(
            "nonlinear candidate rejected expected `{expected_candidate_rejected}` got `{actual_candidate_rejected:?}`"
        ));
    }
    summary.status = if summary.issues.is_empty() {
        "passed"
    } else {
        "failed"
    }
    .to_string();
    summary
}

fn synthetic_case_root_polynomial_selected_inside_measured_support() -> SyntheticColorCaseSummary {
    let (profile, held_out) = synthetic_nonlinear_calibration_profile();
    let mut image = Array3::<f64>::zeros((16, 16, 3));
    for y in 0..16 {
        for x in 0..16 {
            let rgb = held_out[(y * 16 + x) % held_out.len()].source_rgb;
            for channel in 0..3 {
                image[[y, x, channel]] = rgb[channel];
            }
        }
    }
    summarize_synthetic_nonlinear_support_result(
        "root_polynomial_selected_inside_measured_support",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &image,
            Some(&profile),
            colorspace::ColorMode::Calibrated,
        ),
        "calibrated_root_polynomial",
        "accepted",
        false,
        "calibrated_root_polynomial",
        "calibrated_root_polynomial",
    )
}

fn synthetic_case_root_polynomial_rejected_outside_measured_support() -> SyntheticColorCaseSummary {
    let (profile, _) = synthetic_nonlinear_calibration_profile();
    let mut image = Array3::<f64>::zeros((16, 16, 3));
    for y in 0..16 {
        for x in 0..16 {
            image[[y, x, 0]] = 1.0;
            image[[y, x, 1]] = 0.001;
            image[[y, x, 2]] = 0.001;
        }
    }
    summarize_synthetic_nonlinear_support_result(
        "root_polynomial_rejected_outside_measured_support",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &image,
            Some(&profile),
            colorspace::ColorMode::Calibrated,
        ),
        "calibrated_root_polynomial",
        "rejected",
        true,
        "calibrated_direct_profile",
        "calibrated_profile",
    )
}

fn synthetic_case_residual_lut_selected_inside_measured_support() -> SyntheticColorCaseSummary {
    let (profile, held_out) = synthetic_residual_lut_calibration_profile();
    let model = profile
        .lut_3d_model
        .as_ref()
        .expect("synthetic residual LUT model");
    let supported = held_out
        .iter()
        .filter(|patch| {
            color_calibration::residual_lut_3d_has_full_support(model, patch.source_rgb)
        })
        .collect::<Vec<_>>();
    let mut image = Array3::<f64>::zeros((16, 16, 3));
    for y in 0..16 {
        for x in 0..16 {
            let rgb = supported[(y * 16 + x) % supported.len()].source_rgb;
            for channel in 0..3 {
                image[[y, x, channel]] = rgb[channel];
            }
        }
    }
    summarize_synthetic_nonlinear_support_result(
        "residual_lut_selected_inside_measured_support",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &image,
            Some(&profile),
            colorspace::ColorMode::Calibrated,
        ),
        "calibrated_residual_lut_3d",
        "accepted",
        false,
        "calibrated_residual_lut_3d",
        "calibrated_residual_lut_3d",
    )
}

fn synthetic_case_residual_lut_rejected_outside_measured_rgb_volume() -> SyntheticColorCaseSummary {
    let (profile, _) = synthetic_residual_lut_calibration_profile();
    let mut image = Array3::<f64>::zeros((16, 16, 3));
    image.fill(1.25);
    summarize_synthetic_nonlinear_support_result(
        "residual_lut_rejected_outside_measured_rgb_volume",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &image,
            Some(&profile),
            colorspace::ColorMode::Calibrated,
        ),
        "calibrated_residual_lut_3d",
        "rejected",
        true,
        "calibrated_direct_profile",
        "calibrated_profile",
    )
}

fn synthetic_case_weak_calibration_rejected() -> SyntheticColorCaseSummary {
    let mut img = Array3::<f64>::zeros((30, 30, 3));
    for y in 0..30 {
        let value = if y < 10 {
            0.18
        } else if y < 20 {
            0.50
        } else {
            0.82
        };
        for x in 0..30 {
            for c in 0..3 {
                img[[y, x, c]] = value;
            }
        }
    }
    let mut profile = synthetic_calibration_profile();
    profile.work_to_xyz = prophoto_matrix_rows();
    profile.whitepoint = crate::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;
    profile.confidence = 0.0;
    profile.fit = Some(color_calibration::TargetFitDiagnostics {
        method: "synthetic_bad_fit".to_string(),
        patch_count: 24,
        target_residual_rms: 0.50,
        target_residual_max: 1.20,
        validation: None,
        per_hue_residuals: Vec::new(),
        worst_patches: Vec::new(),
    });

    summarize_synthetic_result(
        "weak_calibration_rejected_quality",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            colorspace::ColorMode::Auto,
        ),
        SyntheticCaseExpectations {
            selected_candidate: Some("image_derived_matrix"),
            mapping_strategy: None,
            gamut_fallback_used: None,
            image_matrix_pre_scale_low_clip_total_min: None,
            selected_pre_scale_low_clip_total_max: None,
            post_scale_low_clip_total_max: None,
            calibration_acceptance_status: Some("rejected_quality"),
            calibration_beats_image_derived: Some(false),
            calibrated_candidate_status: Some("rejected_quality"),
            preferred_calibration_candidate_status: Some("rejected_quality"),
            candidate_risk: None,
            tone_color_trust_state: None,
            neutral_estimate_accepted: None,
            dominant_anchor_accepted: None,
            dominant_anchor_channel_populated_band_count: None,
            dominant_anchor_channel_unstable: None,
            channel_anchor_low_support: None,
            neutral_rejection_minima: None,
            dominant_anchor_rejection_minima: None,
            exposure_scale_min: None,
            post_scale_high_clip_less_than_pre_scale: None,
            post_scale_preserved_ratio_min: None,
            reference_patch_regresses_image_derived: None,
            reference_patch_regressed_candidates: None,
            reference_patch_hue_family_regressions: None,
        },
    )
}

fn synthetic_case_calibration_neutral_regression_rejected() -> SyntheticColorCaseSummary {
    let img = broad_neutral_band_image(30, 30);
    let prophoto_to_xyz = colorspace::prophoto_to_xyz_d50_matrix();
    let neutral_cast = Matrix3::new(1.12, 0.0, 0.0, 0.0, 0.88, 0.0, 0.0, 0.0, 1.0);
    let mut profile = synthetic_calibration_profile();
    profile.work_to_xyz = matrix3_to_rows(prophoto_to_xyz * neutral_cast);
    profile.whitepoint = crate::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;
    profile.confidence = 1.0;
    profile.fit = None;

    summarize_synthetic_result(
        "calibration_neutral_regression_rejected",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            colorspace::ColorMode::Auto,
        ),
        SyntheticCaseExpectations {
            selected_candidate: Some("image_derived_matrix"),
            mapping_strategy: Some("image_derived_matrix"),
            gamut_fallback_used: Some(false),
            image_matrix_pre_scale_low_clip_total_min: None,
            selected_pre_scale_low_clip_total_max: None,
            post_scale_low_clip_total_max: None,
            calibration_acceptance_status: Some("rejected_neutral"),
            calibration_beats_image_derived: Some(true),
            calibrated_candidate_status: Some("rejected_neutral"),
            preferred_calibration_candidate_status: Some("rejected_neutral"),
            candidate_risk: Some("review_neutral_support"),
            tone_color_trust_state: Some("review_required"),
            neutral_estimate_accepted: None,
            dominant_anchor_accepted: None,
            dominant_anchor_channel_populated_band_count: None,
            dominant_anchor_channel_unstable: None,
            channel_anchor_low_support: None,
            neutral_rejection_minima: None,
            dominant_anchor_rejection_minima: None,
            exposure_scale_min: None,
            post_scale_high_clip_less_than_pre_scale: None,
            post_scale_preserved_ratio_min: None,
            reference_patch_regresses_image_derived: None,
            reference_patch_regressed_candidates: None,
            reference_patch_hue_family_regressions: None,
        },
    )
}

fn synthetic_case_unsafe_calibration_rejected_auto() -> SyntheticColorCaseSummary {
    let img = sparse_strong_anchor_image();
    let mut profile = synthetic_calibration_profile();
    profile.work_to_xyz = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    profile.whitepoint = crate::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;

    summarize_synthetic_result(
        "unsafe_calibration_rejected_auto",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            colorspace::ColorMode::Auto,
        ),
        SyntheticCaseExpectations {
            selected_candidate: Some("gamut_trusted_image_matrix_blend"),
            mapping_strategy: Some("gamut_trusted_image_matrix_blend"),
            gamut_fallback_used: Some(false),
            image_matrix_pre_scale_low_clip_total_min: None,
            selected_pre_scale_low_clip_total_max: None,
            post_scale_low_clip_total_max: None,
            calibration_acceptance_status: Some("rejected_unsafe"),
            calibration_beats_image_derived: Some(false),
            calibrated_candidate_status: Some("rejected_safety"),
            preferred_calibration_candidate_status: Some("rejected_safety"),
            candidate_risk: None,
            tone_color_trust_state: None,
            neutral_estimate_accepted: None,
            dominant_anchor_accepted: None,
            dominant_anchor_channel_populated_band_count: None,
            dominant_anchor_channel_unstable: None,
            channel_anchor_low_support: None,
            neutral_rejection_minima: None,
            dominant_anchor_rejection_minima: None,
            exposure_scale_min: None,
            post_scale_high_clip_less_than_pre_scale: None,
            post_scale_preserved_ratio_min: None,
            reference_patch_regresses_image_derived: None,
            reference_patch_regressed_candidates: None,
            reference_patch_hue_family_regressions: None,
        },
    )
}

fn synthetic_case_scanner_prior_beats_weak_anchor_image_candidate() -> SyntheticColorCaseSummary {
    let img = weak_anchor_scanner_prior_image();
    let profile = synthetic_scanner_prior_profile();

    summarize_synthetic_result(
        "scanner_prior_beats_weak_anchor_image_candidate",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            colorspace::ColorMode::Auto,
        ),
        SyntheticCaseExpectations {
            selected_candidate: Some("scanner_prior_image_adaptation"),
            mapping_strategy: Some("scanner_constrained_image_derived_matrix"),
            gamut_fallback_used: Some(false),
            image_matrix_pre_scale_low_clip_total_min: None,
            selected_pre_scale_low_clip_total_max: None,
            post_scale_low_clip_total_max: None,
            calibration_acceptance_status: Some("accepted"),
            calibration_beats_image_derived: Some(true),
            calibrated_candidate_status: None,
            preferred_calibration_candidate_status: Some("selected"),
            candidate_risk: None,
            tone_color_trust_state: None,
            neutral_estimate_accepted: None,
            dominant_anchor_accepted: None,
            dominant_anchor_channel_populated_band_count: None,
            dominant_anchor_channel_unstable: None,
            channel_anchor_low_support: Some([false, true, false]),
            neutral_rejection_minima: None,
            dominant_anchor_rejection_minima: None,
            exposure_scale_min: None,
            post_scale_high_clip_less_than_pre_scale: None,
            post_scale_preserved_ratio_min: None,
            reference_patch_regresses_image_derived: None,
            reference_patch_regressed_candidates: None,
            reference_patch_hue_family_regressions: None,
        },
    )
}

fn synthetic_case_unsafe_scanner_prior_rejected_auto() -> SyntheticColorCaseSummary {
    let img = sparse_strong_anchor_image();
    let mut profile = synthetic_scanner_prior_profile();
    profile.work_to_xyz = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    profile.whitepoint = crate::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;

    summarize_synthetic_result(
        "unsafe_scanner_prior_rejected_auto",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            colorspace::ColorMode::Auto,
        ),
        SyntheticCaseExpectations {
            selected_candidate: Some("gamut_trusted_image_matrix_blend"),
            mapping_strategy: Some("gamut_trusted_image_matrix_blend"),
            gamut_fallback_used: Some(false),
            image_matrix_pre_scale_low_clip_total_min: None,
            selected_pre_scale_low_clip_total_max: None,
            post_scale_low_clip_total_max: None,
            calibration_acceptance_status: Some("rejected_unsafe"),
            calibration_beats_image_derived: Some(false),
            calibrated_candidate_status: None,
            preferred_calibration_candidate_status: Some("rejected_safety"),
            candidate_risk: None,
            tone_color_trust_state: None,
            neutral_estimate_accepted: None,
            dominant_anchor_accepted: None,
            dominant_anchor_channel_populated_band_count: None,
            dominant_anchor_channel_unstable: None,
            channel_anchor_low_support: None,
            neutral_rejection_minima: None,
            dominant_anchor_rejection_minima: None,
            exposure_scale_min: None,
            post_scale_high_clip_less_than_pre_scale: None,
            post_scale_preserved_ratio_min: None,
            reference_patch_regresses_image_derived: None,
            reference_patch_regressed_candidates: None,
            reference_patch_hue_family_regressions: None,
        },
    )
}

fn synthetic_case_sparse_uncalibrated_anchor_review_required() -> SyntheticColorCaseSummary {
    let img = sparse_weak_green_anchor_image();

    summarize_synthetic_result(
        "sparse_uncalibrated_anchor_review_required",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            None,
            colorspace::ColorMode::Auto,
        ),
        SyntheticCaseExpectations {
            selected_candidate: Some("image_derived_matrix"),
            mapping_strategy: None,
            gamut_fallback_used: None,
            image_matrix_pre_scale_low_clip_total_min: None,
            selected_pre_scale_low_clip_total_max: None,
            post_scale_low_clip_total_max: None,
            calibration_acceptance_status: Some("not_applicable"),
            calibration_beats_image_derived: None,
            calibrated_candidate_status: None,
            preferred_calibration_candidate_status: None,
            candidate_risk: Some("review_neutral_support"),
            tone_color_trust_state: Some("review_required"),
            neutral_estimate_accepted: Some(false),
            dominant_anchor_accepted: Some(false),
            dominant_anchor_channel_populated_band_count: None,
            dominant_anchor_channel_unstable: None,
            channel_anchor_low_support: Some([false, true, false]),
            neutral_rejection_minima: None,
            dominant_anchor_rejection_minima: None,
            exposure_scale_min: None,
            post_scale_high_clip_less_than_pre_scale: None,
            post_scale_preserved_ratio_min: None,
            reference_patch_regresses_image_derived: None,
            reference_patch_regressed_candidates: None,
            reference_patch_hue_family_regressions: None,
        },
    )
}

fn synthetic_case_dirty_edge_samples_rejected() -> SyntheticColorCaseSummary {
    let img = dirty_edge_rejection_image();

    summarize_synthetic_result(
        "dirty_edge_samples_rejected",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            None,
            colorspace::ColorMode::Auto,
        ),
        SyntheticCaseExpectations {
            selected_candidate: Some("gamut_trusted_image_matrix_blend"),
            mapping_strategy: None,
            gamut_fallback_used: None,
            image_matrix_pre_scale_low_clip_total_min: None,
            selected_pre_scale_low_clip_total_max: None,
            post_scale_low_clip_total_max: None,
            calibration_acceptance_status: Some("not_applicable"),
            calibration_beats_image_derived: None,
            calibrated_candidate_status: None,
            preferred_calibration_candidate_status: None,
            candidate_risk: None,
            tone_color_trust_state: None,
            neutral_estimate_accepted: Some(true),
            dominant_anchor_accepted: None,
            dominant_anchor_channel_populated_band_count: None,
            dominant_anchor_channel_unstable: None,
            channel_anchor_low_support: None,
            neutral_rejection_minima: Some(SyntheticSampleRejectionMinimums {
                clipped: Some(1),
                border: None,
                film_base_like_edge: Some(1),
                dust: Some(1),
            }),
            dominant_anchor_rejection_minima: Some(SyntheticSampleRejectionMinimums {
                clipped: Some(1),
                border: Some(1),
                film_base_like_edge: Some(1),
                dust: Some(1),
            }),
            exposure_scale_min: None,
            post_scale_high_clip_less_than_pre_scale: None,
            post_scale_preserved_ratio_min: None,
            reference_patch_regresses_image_derived: None,
            reference_patch_regressed_candidates: None,
            reference_patch_hue_family_regressions: None,
        },
    )
}

fn synthetic_case_biased_scene_anchor_review_required() -> SyntheticColorCaseSummary {
    let img = biased_single_band_anchor_image();

    summarize_synthetic_result(
        "biased_scene_anchor_review_required",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            None,
            colorspace::ColorMode::Auto,
        ),
        SyntheticCaseExpectations {
            selected_candidate: Some("neutral_balance_fallback"),
            mapping_strategy: Some("neutral_balance_evidence_rescue"),
            gamut_fallback_used: None,
            image_matrix_pre_scale_low_clip_total_min: None,
            selected_pre_scale_low_clip_total_max: None,
            post_scale_low_clip_total_max: None,
            calibration_acceptance_status: Some("not_applicable"),
            calibration_beats_image_derived: None,
            calibrated_candidate_status: None,
            preferred_calibration_candidate_status: None,
            candidate_risk: Some("fallback_only"),
            tone_color_trust_state: Some("review_required"),
            neutral_estimate_accepted: Some(true),
            dominant_anchor_accepted: Some(false),
            dominant_anchor_channel_populated_band_count: Some([1, 1, 1]),
            dominant_anchor_channel_unstable: Some([true, true, true]),
            channel_anchor_low_support: None,
            neutral_rejection_minima: None,
            dominant_anchor_rejection_minima: None,
            exposure_scale_min: None,
            post_scale_high_clip_less_than_pre_scale: None,
            post_scale_preserved_ratio_min: None,
            reference_patch_regresses_image_derived: None,
            reference_patch_regressed_candidates: None,
            reference_patch_hue_family_regressions: None,
        },
    )
}

fn synthetic_case_high_key_exposure_normalization_preserves_headroom() -> SyntheticColorCaseSummary
{
    let img = high_key_headroom_image();

    summarize_synthetic_result(
        "high_key_exposure_normalization_preserves_headroom",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            None,
            colorspace::ColorMode::Auto,
        ),
        SyntheticCaseExpectations {
            selected_candidate: Some("image_derived_matrix"),
            mapping_strategy: None,
            gamut_fallback_used: None,
            image_matrix_pre_scale_low_clip_total_min: None,
            selected_pre_scale_low_clip_total_max: None,
            post_scale_low_clip_total_max: None,
            calibration_acceptance_status: Some("not_applicable"),
            calibration_beats_image_derived: None,
            calibrated_candidate_status: None,
            preferred_calibration_candidate_status: None,
            candidate_risk: None,
            tone_color_trust_state: None,
            neutral_estimate_accepted: None,
            dominant_anchor_accepted: None,
            dominant_anchor_channel_populated_band_count: None,
            dominant_anchor_channel_unstable: None,
            channel_anchor_low_support: None,
            neutral_rejection_minima: None,
            dominant_anchor_rejection_minima: None,
            exposure_scale_min: Some(1.000001),
            post_scale_high_clip_less_than_pre_scale: Some(true),
            post_scale_preserved_ratio_min: Some(0.99),
            reference_patch_regresses_image_derived: None,
            reference_patch_regressed_candidates: None,
            reference_patch_hue_family_regressions: None,
        },
    )
}

fn synthetic_case_destructive_gamut_fallback_preserves_output() -> SyntheticColorCaseSummary {
    let img = destructive_gamut_fallback_image();

    summarize_synthetic_result(
        "destructive_gamut_fallback_preserves_output",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            None,
            colorspace::ColorMode::Auto,
        ),
        SyntheticCaseExpectations {
            selected_candidate: Some("gamut_safe_image_matrix_blend"),
            mapping_strategy: Some("gamut_safe_image_matrix_blend"),
            gamut_fallback_used: Some(false),
            image_matrix_pre_scale_low_clip_total_min: Some(0.18),
            selected_pre_scale_low_clip_total_max: Some(1e-9),
            post_scale_low_clip_total_max: Some(1e-9),
            calibration_acceptance_status: Some("not_applicable"),
            calibration_beats_image_derived: None,
            calibrated_candidate_status: None,
            preferred_calibration_candidate_status: None,
            candidate_risk: Some("review_neutral_support"),
            tone_color_trust_state: None,
            neutral_estimate_accepted: None,
            dominant_anchor_accepted: None,
            dominant_anchor_channel_populated_band_count: None,
            dominant_anchor_channel_unstable: None,
            channel_anchor_low_support: None,
            neutral_rejection_minima: None,
            dominant_anchor_rejection_minima: None,
            exposure_scale_min: None,
            post_scale_high_clip_less_than_pre_scale: None,
            post_scale_preserved_ratio_min: None,
            reference_patch_regresses_image_derived: None,
            reference_patch_regressed_candidates: None,
            reference_patch_hue_family_regressions: None,
        },
    )
}

fn synthetic_case_direct_density_render_fallback_for_destructive_ica_gamut(
) -> SyntheticColorCaseSummary {
    let ica = synthetic_render_input_diagnostics_with_low_clip([0.43, 0.16, 0.13], true);
    let direct = synthetic_render_input_diagnostics_with_low_clip([0.0, 0.0, 0.001], false);

    summarize_synthetic_render_input_result(
        "direct_density_render_fallback_for_destructive_ica_gamut",
        &ica,
        &direct,
        "direct_density_transmittance",
        true,
        Some("direct density transmittance reduced the image-matrix low clipping"),
    )
}

fn synthetic_case_direct_density_render_fallback_rejects_neutral_regression(
) -> SyntheticColorCaseSummary {
    let ica = synthetic_render_input_diagnostics_with_low_clip([0.43, 0.16, 0.13], true);
    let mut direct = synthetic_render_input_diagnostics_with_low_clip([0.0, 0.0, 0.001], false);
    direct.image_matrix_neutral_balance_delta = [0.04, -0.03, 0.02];

    summarize_synthetic_render_input_result(
        "direct_density_render_fallback_rejects_neutral_regression",
        &ica,
        &direct,
        "fastica_separated_transmittance",
        false,
        Some("ICA-separated density channels retained"),
    )
}

fn synthetic_case_direct_density_render_fallback_for_quality_win() -> SyntheticColorCaseSummary {
    let mut ica = synthetic_render_input_diagnostics_with_low_clip([0.01, 0.0, 0.0], false);
    ica.selected_quality_score = Some(1.20);
    ica.candidate_risk = "review_quality_score".to_string();
    let mut direct = synthetic_render_input_diagnostics_with_low_clip([0.005, 0.0, 0.0], false);
    direct.selected_quality_score = Some(0.70);
    direct.candidate_risk = "safe".to_string();

    summarize_synthetic_render_input_result(
        "direct_density_render_fallback_for_quality_win",
        &ica,
        &direct,
        "direct_density_transmittance",
        true,
        Some("materially stronger colorspace candidate"),
    )
}

fn synthetic_case_direct_density_render_fallback_rejects_worse_risk() -> SyntheticColorCaseSummary {
    let mut ica = synthetic_render_input_diagnostics_with_low_clip([0.01, 0.0, 0.0], false);
    ica.selected_quality_score = Some(1.20);
    ica.candidate_risk = "safe".to_string();
    let mut direct = synthetic_render_input_diagnostics_with_low_clip([0.005, 0.0, 0.0], false);
    direct.selected_quality_score = Some(0.70);
    direct.candidate_risk = "review_gamut".to_string();

    summarize_synthetic_render_input_result(
        "direct_density_render_fallback_rejects_worse_risk",
        &ica,
        &direct,
        "fastica_separated_transmittance",
        false,
        Some("ICA-separated density channels retained"),
    )
}

fn synthetic_case_tone_policy_weak_neutral_keeps_bounded_neutral_cleanup(
) -> SyntheticColorCaseSummary {
    let params = synthetic_linear_tone_params(2.0);
    let protection = tonemap::ToneColorProtection {
        policy: tonemap::ToneColorProtectionPolicy::WeakNeutralBoundedNeutralCleanup,
        highlight_neutral_chroma_enabled: true,
        midtone_neutral_chroma_enabled: true,
        shadow_chroma_enabled: true,
        reason: "synthetic weak neutral support".to_string(),
    };
    let mut img = Array3::<f64>::zeros((2, 1, 3));
    img[[0, 0, 0]] = 0.55;
    img[[0, 0, 1]] = 0.70;
    img[[0, 0, 2]] = 0.80;
    img[[1, 0, 0]] = 0.36;
    img[[1, 0, 1]] = 0.44;
    img[[1, 0, 2]] = 0.50;

    summarize_synthetic_tone_policy_result(
        "tone_policy_weak_neutral_keeps_bounded_neutral_cleanup",
        tonemap::apply_tonemap_with_params_and_color_protection_diagnostics(
            &img,
            &params,
            &protection,
        ),
        SyntheticTonePolicyExpectations {
            policy: "weak_neutral_bounded_neutral_cleanup",
            color_trust_state: "limited_weak_neutral",
            highlight_neutral_chroma_enabled: true,
            shadow_chroma_enabled: true,
            highlight_chroma_compressed_ratio_min: None,
            highlight_neutral_chroma_compressed_ratio_max: None,
            shadow_chroma_compressed_ratio_max: None,
        },
    )
}

fn synthetic_case_tone_policy_color_review_disables_color_cleanup_preserves_gamut_repair(
) -> SyntheticColorCaseSummary {
    let params = synthetic_linear_tone_params(5.0);
    let protection = tonemap::ToneColorProtection {
        policy: tonemap::ToneColorProtectionPolicy::DisabledColorCandidateReview,
        highlight_neutral_chroma_enabled: false,
        midtone_neutral_chroma_enabled: false,
        shadow_chroma_enabled: false,
        reason: "synthetic color candidate review".to_string(),
    };
    let mut img = Array3::<f64>::zeros((2, 1, 3));
    img[[0, 0, 0]] = 0.20;
    img[[0, 0, 1]] = 0.90;
    img[[0, 0, 2]] = 1.00;
    img[[1, 0, 0]] = 0.03;
    img[[1, 0, 1]] = 0.10;
    img[[1, 0, 2]] = 0.22;

    summarize_synthetic_tone_policy_result(
        "tone_policy_color_review_disables_color_cleanup_preserves_gamut_repair",
        tonemap::apply_tonemap_with_params_and_color_protection_diagnostics(
            &img,
            &params,
            &protection,
        ),
        SyntheticTonePolicyExpectations {
            policy: "disabled_color_candidate_review",
            color_trust_state: "review_required",
            highlight_neutral_chroma_enabled: false,
            shadow_chroma_enabled: false,
            highlight_chroma_compressed_ratio_min: Some(0.1),
            highlight_neutral_chroma_compressed_ratio_max: Some(0.0),
            shadow_chroma_compressed_ratio_max: Some(0.0),
        },
    )
}

fn synthetic_case_tone_policy_model_review_keeps_bounded_neutral_cleanup(
) -> SyntheticColorCaseSummary {
    let params = synthetic_linear_tone_params(5.0);
    let protection = tonemap::ToneColorProtection {
        policy: tonemap::ToneColorProtectionPolicy::ReviewBoundedNeutralAndShadowCleanup,
        highlight_neutral_chroma_enabled: true,
        midtone_neutral_chroma_enabled: true,
        shadow_chroma_enabled: true,
        reason: "synthetic model-plausibility review".to_string(),
    };
    let mut img = Array3::<f64>::zeros((4, 1, 3));
    img[[0, 0, 0]] = 0.20;
    img[[0, 0, 1]] = 0.90;
    img[[0, 0, 2]] = 1.00;
    img[[1, 0, 0]] = 0.34;
    img[[1, 0, 1]] = 0.42;
    img[[1, 0, 2]] = 0.50;
    img[[2, 0, 0]] = 0.03;
    img[[2, 0, 1]] = 0.10;
    img[[2, 0, 2]] = 0.22;
    // A bright, low-saturation pixel is required to exercise the bounded neutral-highlight
    // cleanup independently of the saturated-highlight gamut mapping above.
    img[[3, 0, 0]] = 0.80;
    img[[3, 0, 1]] = 0.83;
    img[[3, 0, 2]] = 0.86;

    summarize_synthetic_tone_policy_result(
        "tone_policy_model_review_keeps_bounded_neutral_cleanup",
        tonemap::apply_tonemap_with_params_and_color_protection_diagnostics(
            &img,
            &params,
            &protection,
        ),
        SyntheticTonePolicyExpectations {
            policy: "review_bounded_neutral_shadow_cleanup",
            color_trust_state: "review_required",
            highlight_neutral_chroma_enabled: true,
            shadow_chroma_enabled: true,
            highlight_chroma_compressed_ratio_min: Some(0.1),
            highlight_neutral_chroma_compressed_ratio_max: None,
            shadow_chroma_compressed_ratio_max: None,
        },
    )
}

fn synthetic_case_reference_patch_regression_rejected() -> SyntheticColorCaseSummary {
    let mut img = Array3::<f64>::zeros((36, 36, 3));
    for y in 0..36 {
        for x in 0..36 {
            let pixel = if x < 9 {
                [0.78, 0.12, 0.10]
            } else if x < 18 {
                [0.14, 0.72, 0.16]
            } else if x < 27 {
                [0.12, 0.18, 0.76]
            } else if y < 12 {
                [0.25, 0.25, 0.25]
            } else if y < 24 {
                [0.52, 0.52, 0.52]
            } else {
                [0.82, 0.82, 0.82]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }

    let image_only = colorspace::estimate_work_to_xyz(&img);
    let image_work_to_xyz = matrix3_from_rows(image_only.work_to_xyz);
    let image_matrix = colorspace::xyz_d50_to_prophoto_matrix()
        * colorspace::bradford_cat(&image_only.source_white)
        * image_work_to_xyz;
    let prophoto_to_xyz = colorspace::prophoto_to_xyz_d50_matrix();
    let uniform_underfit =
        prophoto_to_xyz * Matrix3::new(0.90, 0.0, 0.0, 0.0, 0.90, 0.0, 0.0, 0.0, 0.90);

    let mut profile = synthetic_calibration_profile();
    profile.work_to_xyz = matrix3_to_rows(uniform_underfit);
    profile.whitepoint = crate::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;
    profile.confidence = 1.0;
    profile.fit = None;
    profile.target_patches = reference_fit_patches_for_matrix(image_matrix);

    summarize_synthetic_result(
        "reference_patch_regression_rejected",
        colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            colorspace::ColorMode::Auto,
        ),
        SyntheticCaseExpectations {
            selected_candidate: None,
            mapping_strategy: None,
            gamut_fallback_used: None,
            image_matrix_pre_scale_low_clip_total_min: None,
            selected_pre_scale_low_clip_total_max: None,
            post_scale_low_clip_total_max: None,
            calibration_acceptance_status: Some("rejected_reference_fit"),
            calibration_beats_image_derived: None,
            calibrated_candidate_status: Some("rejected_reference_fit"),
            preferred_calibration_candidate_status: Some("rejected_reference_fit"),
            candidate_risk: Some("review_reference_fit"),
            tone_color_trust_state: Some("review_required"),
            neutral_estimate_accepted: None,
            dominant_anchor_accepted: None,
            dominant_anchor_channel_populated_band_count: None,
            dominant_anchor_channel_unstable: None,
            channel_anchor_low_support: None,
            neutral_rejection_minima: None,
            dominant_anchor_rejection_minima: None,
            exposure_scale_min: None,
            post_scale_high_clip_less_than_pre_scale: None,
            post_scale_preserved_ratio_min: None,
            reference_patch_regresses_image_derived: None,
            reference_patch_regressed_candidates: Some(vec![
                "gamut_safe_image_matrix_blend",
                "calibrated_direct_profile",
            ]),
            reference_patch_hue_family_regressions: None,
        },
    )
}

fn synthetic_case_forced_unsafe_calibration_fails() -> SyntheticColorCaseSummary {
    let img = sparse_strong_anchor_image();
    let mut profile = synthetic_calibration_profile();
    profile.work_to_xyz = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    profile.whitepoint = crate::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;

    let expected_error = "selected only unsafe colorspace candidate";
    match colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
        &img,
        Some(&profile),
        colorspace::ColorMode::Calibrated,
    ) {
        Ok(result) => {
            let mut case = summarize_synthetic_result(
                "forced_unsafe_calibration_fails",
                Ok(result),
                SyntheticCaseExpectations {
                    selected_candidate: None,
                    mapping_strategy: None,
                    gamut_fallback_used: None,
                    image_matrix_pre_scale_low_clip_total_min: None,
                    selected_pre_scale_low_clip_total_max: None,
                    post_scale_low_clip_total_max: None,
                    calibration_acceptance_status: None,
                    calibration_beats_image_derived: None,
                    calibrated_candidate_status: None,
                    preferred_calibration_candidate_status: None,
                    candidate_risk: None,
                    tone_color_trust_state: None,
                    neutral_estimate_accepted: None,
                    dominant_anchor_accepted: None,
                    dominant_anchor_channel_populated_band_count: None,
                    dominant_anchor_channel_unstable: None,
                    channel_anchor_low_support: None,
                    neutral_rejection_minima: None,
                    dominant_anchor_rejection_minima: None,
                    exposure_scale_min: None,
                    post_scale_high_clip_less_than_pre_scale: None,
                    post_scale_preserved_ratio_min: None,
                    reference_patch_regresses_image_derived: None,
                    reference_patch_regressed_candidates: None,
                    reference_patch_hue_family_regressions: None,
                },
            );
            case.expected_error_contains = Some(expected_error.to_string());
            case.issues
                .push("expected forced calibrated mode to fail".to_string());
            case.status = "failed".to_string();
            case
        }
        Err(err) => {
            let mut issues = Vec::new();
            if !err.contains(expected_error) {
                issues.push(format!("error did not contain `{expected_error}`"));
            }
            if !err.contains("negative channel values") {
                issues.push("error did not include negative-gamut safety reason".to_string());
            }
            SyntheticColorCaseSummary {
                name: "forced_unsafe_calibration_fails".to_string(),
                status: if issues.is_empty() {
                    "passed"
                } else {
                    "failed"
                }
                .to_string(),
                expected_render_input_source: None,
                actual_render_input_source: None,
                expected_render_input_fallback_used: None,
                actual_render_input_fallback_used: None,
                expected_render_input_reason_contains: None,
                actual_render_input_reason: None,
                expected_selected_candidate: None,
                actual_selected_candidate: None,
                expected_mapping_strategy: None,
                actual_mapping_strategy: None,
                expected_gamut_fallback_used: None,
                actual_gamut_fallback_used: None,
                expected_image_matrix_pre_scale_low_clip_total_min: None,
                actual_image_matrix_pre_scale_low_clip_total: None,
                expected_selected_pre_scale_low_clip_total_max: None,
                actual_selected_pre_scale_low_clip_total: None,
                expected_post_scale_low_clip_total_max: None,
                actual_post_scale_low_clip_total: None,
                expected_calibration_acceptance_status: None,
                actual_calibration_acceptance_status: None,
                expected_calibration_beats_image_derived: None,
                actual_calibration_beats_image_derived: None,
                expected_calibrated_candidate_status: None,
                actual_calibrated_candidate_status: None,
                expected_preferred_calibration_candidate_status: None,
                actual_preferred_calibration_candidate_status: None,
                expected_candidate_risk: None,
                actual_candidate_risk: None,
                expected_tone_color_trust_state: None,
                actual_tone_color_trust_state: None,
                expected_tone_policy: None,
                actual_tone_policy: None,
                expected_tone_policy_color_trust_state: None,
                actual_tone_policy_color_trust_state: None,
                expected_tone_highlight_neutral_chroma_enabled: None,
                actual_tone_highlight_neutral_chroma_enabled: None,
                expected_tone_shadow_chroma_enabled: None,
                actual_tone_shadow_chroma_enabled: None,
                expected_tone_highlight_chroma_compressed_ratio_min: None,
                actual_tone_highlight_chroma_compressed_ratio: None,
                expected_tone_highlight_neutral_chroma_compressed_ratio_max: None,
                actual_tone_highlight_neutral_chroma_compressed_ratio: None,
                expected_tone_shadow_chroma_compressed_ratio_max: None,
                actual_tone_shadow_chroma_compressed_ratio: None,
                expected_neutral_estimate_accepted: None,
                actual_neutral_estimate_accepted: None,
                expected_dominant_anchor_accepted: None,
                actual_dominant_anchor_accepted: None,
                expected_dominant_anchor_channel_populated_band_count: None,
                actual_dominant_anchor_channel_populated_band_count: None,
                expected_dominant_anchor_channel_unstable: None,
                actual_dominant_anchor_channel_unstable: None,
                expected_channel_anchor_low_support: None,
                actual_channel_anchor_low_support: None,
                expected_neutral_rejection_minima: None,
                actual_neutral_sample_rejections: None,
                expected_dominant_anchor_rejection_minima: None,
                actual_dominant_anchor_sample_rejections: None,
                expected_exposure_scale_min: None,
                actual_exposure_scale: None,
                expected_post_scale_high_clip_less_than_pre_scale: None,
                actual_post_scale_high_clip_less_than_pre_scale: None,
                actual_pre_scale_high_clip_total: None,
                actual_post_scale_high_clip_total: None,
                expected_post_scale_preserved_ratio_min: None,
                actual_post_scale_preserved_ratio: None,
                actual_density_monotonicity_score: None,
                actual_hue_linearity_score: None,
                actual_saturation_preservation_median_ratio: None,
                actual_spatial_neutral_delta_p95: None,
                actual_memory_color_penalty: None,
                actual_spatial_consistency_penalty: None,
                selected_quality_score: None,
                image_derived_quality_score: None,
                preferred_calibration_quality_score: None,
                reference_patch_selected_regresses_image_derived: None,
                expected_reference_patch_regressed_candidates: None,
                actual_reference_patch_regressed_candidates: Vec::new(),
                expected_reference_patch_hue_family_regressions: None,
                actual_reference_patch_hue_family_regressions: Vec::new(),
                actual_reference_patch_worst_hue_families: Vec::new(),
                expected_error_contains: Some(expected_error.to_string()),
                actual_error: Some(err),
                issues,
            }
        }
    }
}

fn neutral_rejection_summary_from_colorspace(
    rejections: &colorspace::NeutralSampleRejections,
) -> NeutralSampleRejectionSummary {
    NeutralSampleRejectionSummary {
        total_pixels: Some(rejections.total_pixels),
        accepted_neutral_samples: Some(rejections.accepted_neutral_samples),
        clipped: Some(rejections.clipped),
        border: Some(rejections.border),
        film_base_like_edge: Some(rejections.film_base_like_edge),
        dust: Some(rejections.dust),
        luma_out_of_range: Some(rejections.luma_out_of_range),
        chroma_threshold: Some(rejections.chroma_threshold),
    }
}

fn dominant_anchor_rejection_summary_from_colorspace(
    rejections: &colorspace::DominantAnchorSampleRejections,
) -> DominantAnchorSampleRejectionSummary {
    DominantAnchorSampleRejectionSummary {
        total_pixels: Some(rejections.total_pixels),
        accepted_anchor_samples: Some(rejections.accepted_anchor_samples),
        clipped: Some(rejections.clipped),
        border: Some(rejections.border),
        film_base_like_edge: Some(rejections.film_base_like_edge),
        dust: Some(rejections.dust),
        luma_out_of_range: Some(rejections.luma_out_of_range),
        low_saturation: Some(rejections.low_saturation),
        weak_dominance: Some(rejections.weak_dominance),
    }
}

fn check_rejection_minima(
    label: &str,
    minima: Option<&SyntheticSampleRejectionMinimums>,
    actual: &NeutralSampleRejectionSummary,
    issues: &mut Vec<String>,
) {
    let Some(minima) = minima else {
        return;
    };
    push_rejection_minimum_issue(label, "clipped", minima.clipped, actual.clipped, issues);
    push_rejection_minimum_issue(label, "border", minima.border, actual.border, issues);
    push_rejection_minimum_issue(
        label,
        "film_base_like_edge",
        minima.film_base_like_edge,
        actual.film_base_like_edge,
        issues,
    );
    push_rejection_minimum_issue(label, "dust", minima.dust, actual.dust, issues);
}

fn check_dominant_rejection_minima(
    minima: Option<&SyntheticSampleRejectionMinimums>,
    actual: &DominantAnchorSampleRejectionSummary,
    issues: &mut Vec<String>,
) {
    let Some(minima) = minima else {
        return;
    };
    push_rejection_minimum_issue(
        "dominant_anchor_sample_rejections",
        "clipped",
        minima.clipped,
        actual.clipped,
        issues,
    );
    push_rejection_minimum_issue(
        "dominant_anchor_sample_rejections",
        "border",
        minima.border,
        actual.border,
        issues,
    );
    push_rejection_minimum_issue(
        "dominant_anchor_sample_rejections",
        "film_base_like_edge",
        minima.film_base_like_edge,
        actual.film_base_like_edge,
        issues,
    );
    push_rejection_minimum_issue(
        "dominant_anchor_sample_rejections",
        "dust",
        minima.dust,
        actual.dust,
        issues,
    );
}

fn push_rejection_minimum_issue(
    label: &str,
    field: &str,
    minimum: Option<usize>,
    actual: Option<usize>,
    issues: &mut Vec<String>,
) {
    if let Some(minimum) = minimum {
        let actual = actual.unwrap_or(0);
        if actual < minimum {
            issues.push(format!(
                "{label}.{field} expected at least {minimum} got {actual}"
            ));
        }
    }
}

fn reference_patch_regressed_candidates(
    evaluation: Option<&colorspace::ReferencePatchEvaluation>,
) -> Vec<String> {
    evaluation
        .map(|evaluation| {
            evaluation
                .candidate_evaluations
                .iter()
                .filter(|candidate| candidate.regresses_image_derived == Some(true))
                .map(|candidate| candidate.candidate.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn reference_patch_hue_family_regressions(
    evaluation: Option<&colorspace::ReferencePatchEvaluation>,
) -> Vec<String> {
    evaluation
        .map(|evaluation| {
            evaluation
                .candidate_evaluations
                .iter()
                .filter(|candidate| candidate.regresses_image_derived == Some(true))
                .flat_map(|candidate| {
                    candidate
                        .hue_family_regressions
                        .iter()
                        .map(|regression| regression.hue_family.clone())
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        })
        .unwrap_or_default()
}

fn reference_patch_worst_hue_families(
    evaluation: Option<&colorspace::ReferencePatchEvaluation>,
) -> Vec<String> {
    evaluation
        .map(|evaluation| {
            evaluation
                .worst_hue_families
                .iter()
                .map(|family| family.hue_family.clone())
                .collect()
        })
        .unwrap_or_default()
}

fn expected_string_values(values: Option<&Vec<&str>>) -> Option<Vec<String>> {
    values.map(|values| values.iter().map(|value| (*value).to_string()).collect())
}

fn synthetic_render_input_diagnostics_with_low_clip(
    low_clip: [f64; 3],
    gamut_fallback_used: bool,
) -> colorspace::ColorspaceDiagnostics {
    let mut diagnostics = colorspace::estimate_work_to_xyz(&broad_neutral_band_image(16, 16));
    diagnostics.image_matrix_pre_scale_clipped_low_ratio = low_clip;
    diagnostics.gamut_fallback_used = gamut_fallback_used;
    diagnostics.gamut_fallback_reason =
        gamut_fallback_used.then(|| "synthetic destructive gamut fallback".to_string());
    diagnostics.mapping_strategy = if gamut_fallback_used {
        "neutral_balance_gamut_fallback"
    } else {
        "image_derived_matrix"
    };
    diagnostics.candidate_risk = if gamut_fallback_used {
        "fallback_only".to_string()
    } else {
        "safe".to_string()
    };
    diagnostics.selected_quality_score = Some(if gamut_fallback_used { 1.20 } else { 0.70 });
    diagnostics.post_scale_preserved_ratio = 1.0;
    diagnostics.image_matrix_neutral_balance_delta = [0.0; 3];
    diagnostics
}

fn summarize_synthetic_render_input_result(
    name: &str,
    ica: &colorspace::ColorspaceDiagnostics,
    direct: &colorspace::ColorspaceDiagnostics,
    expected_source: &str,
    expected_fallback_used: bool,
    expected_reason_contains: Option<&str>,
) -> SyntheticColorCaseSummary {
    let fallback_reason = colorspace::direct_density_render_fallback_reason(ica, direct);
    let actual_fallback_used = fallback_reason.is_some();
    let actual_source = if actual_fallback_used {
        "direct_density_transmittance"
    } else {
        "fastica_separated_transmittance"
    };
    let actual_reason = fallback_reason.unwrap_or_else(|| {
        "ICA-separated density channels retained; direct density candidate did not satisfy fallback criteria"
            .to_string()
    });
    let mut issues = Vec::new();
    if actual_source != expected_source {
        issues.push(format!(
            "render_input_source expected `{expected_source}` got `{actual_source}`"
        ));
    }
    if actual_fallback_used != expected_fallback_used {
        issues.push(format!(
            "render_input_fallback_used expected `{expected_fallback_used}` got `{actual_fallback_used}`"
        ));
    }
    if let Some(expected) = expected_reason_contains {
        if !actual_reason.contains(expected) {
            issues.push(format!(
                "render_input_reason expected to contain `{expected}` got `{actual_reason}`"
            ));
        }
    }

    SyntheticColorCaseSummary {
        name: name.to_string(),
        status: if issues.is_empty() {
            "passed".to_string()
        } else {
            "failed".to_string()
        },
        expected_render_input_source: Some(expected_source.to_string()),
        actual_render_input_source: Some(actual_source.to_string()),
        expected_render_input_fallback_used: Some(expected_fallback_used),
        actual_render_input_fallback_used: Some(actual_fallback_used),
        expected_render_input_reason_contains: expected_reason_contains.map(str::to_string),
        actual_render_input_reason: Some(actual_reason),
        issues,
        ..SyntheticColorCaseSummary::default()
    }
}

struct SyntheticTonePolicyExpectations {
    policy: &'static str,
    color_trust_state: &'static str,
    highlight_neutral_chroma_enabled: bool,
    shadow_chroma_enabled: bool,
    highlight_chroma_compressed_ratio_min: Option<f64>,
    highlight_neutral_chroma_compressed_ratio_max: Option<f64>,
    shadow_chroma_compressed_ratio_max: Option<f64>,
}

fn synthetic_linear_tone_params(slope: f64) -> tonemap::ToneCurveParams {
    tonemap::ToneCurveParams {
        domain: tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    }
}

fn summarize_synthetic_tone_policy_result(
    name: &str,
    result: tonemap::TonemapApplyResult,
    expectations: SyntheticTonePolicyExpectations,
) -> SyntheticColorCaseSummary {
    let diagnostics = result.diagnostics;
    let mut issues = Vec::new();
    if diagnostics.color_protection_policy != expectations.policy {
        issues.push(format!(
            "tone color policy expected `{}` got `{}`",
            expectations.policy, diagnostics.color_protection_policy
        ));
    }
    if diagnostics.color_trust_state != expectations.color_trust_state {
        issues.push(format!(
            "tone color trust state expected `{}` got `{}`",
            expectations.color_trust_state, diagnostics.color_trust_state
        ));
    }
    if diagnostics.highlight_neutral_chroma_enabled != expectations.highlight_neutral_chroma_enabled
    {
        issues.push(format!(
            "highlight_neutral_chroma_enabled expected `{}` got `{}`",
            expectations.highlight_neutral_chroma_enabled,
            diagnostics.highlight_neutral_chroma_enabled
        ));
    }
    if diagnostics.shadow_chroma_enabled != expectations.shadow_chroma_enabled {
        issues.push(format!(
            "shadow_chroma_enabled expected `{}` got `{}`",
            expectations.shadow_chroma_enabled, diagnostics.shadow_chroma_enabled
        ));
    }
    if let Some(minimum) = expectations.highlight_chroma_compressed_ratio_min {
        if diagnostics.highlight_chroma_compressed_ratio < minimum {
            issues.push(format!(
                "highlight_chroma_compressed_ratio expected at least {minimum:.6} got {:.6}",
                diagnostics.highlight_chroma_compressed_ratio
            ));
        }
    }
    if let Some(maximum) = expectations.highlight_neutral_chroma_compressed_ratio_max {
        if diagnostics.highlight_neutral_chroma_compressed_ratio > maximum {
            issues.push(format!(
                "highlight_neutral_chroma_compressed_ratio expected at most {maximum:.6} got {:.6}",
                diagnostics.highlight_neutral_chroma_compressed_ratio
            ));
        }
    }
    if let Some(maximum) = expectations.shadow_chroma_compressed_ratio_max {
        if diagnostics.shadow_chroma_compressed_ratio > maximum {
            issues.push(format!(
                "shadow_chroma_compressed_ratio expected at most {maximum:.6} got {:.6}",
                diagnostics.shadow_chroma_compressed_ratio
            ));
        }
    }

    SyntheticColorCaseSummary {
        name: name.to_string(),
        status: if issues.is_empty() {
            "passed".to_string()
        } else {
            "failed".to_string()
        },
        expected_tone_policy: Some(expectations.policy.to_string()),
        actual_tone_policy: Some(diagnostics.color_protection_policy.to_string()),
        expected_tone_policy_color_trust_state: Some(expectations.color_trust_state.to_string()),
        actual_tone_policy_color_trust_state: Some(diagnostics.color_trust_state.to_string()),
        expected_tone_highlight_neutral_chroma_enabled: Some(
            expectations.highlight_neutral_chroma_enabled,
        ),
        actual_tone_highlight_neutral_chroma_enabled: Some(
            diagnostics.highlight_neutral_chroma_enabled,
        ),
        expected_tone_shadow_chroma_enabled: Some(expectations.shadow_chroma_enabled),
        actual_tone_shadow_chroma_enabled: Some(diagnostics.shadow_chroma_enabled),
        expected_tone_highlight_chroma_compressed_ratio_min: expectations
            .highlight_chroma_compressed_ratio_min,
        actual_tone_highlight_chroma_compressed_ratio: Some(
            diagnostics.highlight_chroma_compressed_ratio,
        ),
        expected_tone_highlight_neutral_chroma_compressed_ratio_max: expectations
            .highlight_neutral_chroma_compressed_ratio_max,
        actual_tone_highlight_neutral_chroma_compressed_ratio: Some(
            diagnostics.highlight_neutral_chroma_compressed_ratio,
        ),
        expected_tone_shadow_chroma_compressed_ratio_max: expectations
            .shadow_chroma_compressed_ratio_max,
        actual_tone_shadow_chroma_compressed_ratio: Some(
            diagnostics.shadow_chroma_compressed_ratio,
        ),
        issues,
        ..SyntheticColorCaseSummary::default()
    }
}

fn summarize_synthetic_result(
    name: &str,
    result: Result<colorspace::ColorspaceMappingResult, String>,
    expectations: SyntheticCaseExpectations<'_>,
) -> SyntheticColorCaseSummary {
    let mut issues = Vec::new();
    match result {
        Ok(result) => {
            let diagnostics = result.diagnostics;
            let tone_color_trust_state = colorspace::tone_color_trust_state(&diagnostics);
            let neutral_sample_rejections =
                neutral_rejection_summary_from_colorspace(&diagnostics.neutral_sample_rejections);
            let dominant_anchor_sample_rejections =
                dominant_anchor_rejection_summary_from_colorspace(
                    &diagnostics.dominant_anchor_sample_rejections,
                );
            let pre_scale_high_clip_total = diagnostics.pre_scale_clipped_high_ratio.iter().sum();
            let post_scale_high_clip_total = diagnostics.post_scale_clipped_high_ratio.iter().sum();
            let image_matrix_pre_scale_low_clip_total = diagnostics
                .image_matrix_pre_scale_clipped_low_ratio
                .iter()
                .sum();
            let selected_pre_scale_low_clip_total =
                diagnostics.pre_scale_clipped_low_ratio.iter().sum();
            let post_scale_low_clip_total = diagnostics.post_scale_clipped_low_ratio.iter().sum();
            let post_scale_high_clip_less_than_pre_scale =
                post_scale_high_clip_total < pre_scale_high_clip_total;
            let reference_patch_evaluation = diagnostics.reference_patch_evaluation.as_ref();
            let actual_reference_patch_regressed_candidates =
                reference_patch_regressed_candidates(reference_patch_evaluation);
            let actual_reference_patch_hue_family_regressions =
                reference_patch_hue_family_regressions(reference_patch_evaluation);
            let actual_reference_patch_worst_hue_families =
                reference_patch_worst_hue_families(reference_patch_evaluation);
            let selected_candidate_score = diagnostics
                .candidate_scores
                .iter()
                .find(|candidate| candidate.selected)
                .or_else(|| {
                    diagnostics.candidate_scores.iter().find(|candidate| {
                        candidate.candidate == diagnostics.selected_candidate.as_str()
                    })
                });
            let selected_model_quality = selected_candidate_score
                .and_then(|candidate| candidate.color_model_quality.as_ref());
            let actual_density_monotonicity_score =
                selected_model_quality.map(|quality| quality.density_monotonicity_score);
            let actual_hue_linearity_score =
                selected_model_quality.map(|quality| quality.hue_linearity_score);
            let actual_saturation_preservation_median_ratio = selected_model_quality
                .and_then(|quality| quality.saturation_preservation_median_ratio);
            let actual_spatial_neutral_delta_p95 = selected_model_quality
                .and_then(|quality| quality.spatial_consistency.neutral_delta_p95);
            let actual_memory_color_penalty = selected_candidate_score
                .map(|candidate| candidate.quality_components.memory_color_penalty);
            let actual_spatial_consistency_penalty = selected_candidate_score
                .map(|candidate| candidate.quality_components.spatial_consistency_penalty);
            if selected_model_quality.is_none() {
                issues.push("selected candidate is missing color model diagnostics".to_string());
            }
            if let Some(expected) = expectations.selected_candidate {
                if diagnostics.selected_candidate != expected {
                    issues.push(format!(
                        "selected_candidate expected `{expected}` got `{}`",
                        diagnostics.selected_candidate
                    ));
                }
            }
            if let Some(expected) = expectations.mapping_strategy {
                if diagnostics.mapping_strategy != expected {
                    issues.push(format!(
                        "mapping_strategy expected `{expected}` got `{}`",
                        diagnostics.mapping_strategy
                    ));
                }
            }
            if let Some(expected) = expectations.gamut_fallback_used {
                if diagnostics.gamut_fallback_used != expected {
                    issues.push(format!(
                        "gamut_fallback_used expected `{expected}` got `{}`",
                        diagnostics.gamut_fallback_used
                    ));
                }
            }
            if let Some(minimum) = expectations.image_matrix_pre_scale_low_clip_total_min {
                if image_matrix_pre_scale_low_clip_total < minimum {
                    issues.push(format!(
                        "image_matrix_pre_scale_low_clip_total expected at least {minimum:.6} got {image_matrix_pre_scale_low_clip_total:.6}"
                    ));
                }
            }
            if let Some(maximum) = expectations.selected_pre_scale_low_clip_total_max {
                if selected_pre_scale_low_clip_total > maximum {
                    issues.push(format!(
                        "selected_pre_scale_low_clip_total expected at most {maximum:.6} got {selected_pre_scale_low_clip_total:.6}"
                    ));
                }
            }
            if let Some(maximum) = expectations.post_scale_low_clip_total_max {
                if post_scale_low_clip_total > maximum {
                    issues.push(format!(
                        "post_scale_low_clip_total expected at most {maximum:.6} got {post_scale_low_clip_total:.6}"
                    ));
                }
            }
            if let Some(expected) = expectations.calibration_acceptance_status {
                if diagnostics.calibration_acceptance.status != expected {
                    issues.push(format!(
                        "calibration_acceptance expected `{expected}` got `{}`",
                        diagnostics.calibration_acceptance.status
                    ));
                }
            }
            if let Some(expected) = expectations.calibration_beats_image_derived {
                if diagnostics.calibration_acceptance.beats_image_derived != Some(expected) {
                    issues.push(format!(
                        "beats_image_derived expected `{expected}` got `{:?}`",
                        diagnostics.calibration_acceptance.beats_image_derived
                    ));
                }
            }
            let calibrated_candidate_status = diagnostics
                .candidate_acceptance
                .iter()
                .find(|candidate| candidate.candidate == "calibrated_direct_profile")
                .map(|candidate| candidate.status.clone());
            let preferred_calibration_candidate_status = diagnostics
                .calibration_acceptance
                .preferred_candidate
                .as_deref()
                .and_then(|preferred| {
                    diagnostics
                        .candidate_acceptance
                        .iter()
                        .find(|candidate| candidate.candidate == preferred)
                        .map(|candidate| candidate.status.clone())
                });
            if let Some(expected) = expectations.calibrated_candidate_status {
                if calibrated_candidate_status.as_deref() != Some(expected) {
                    issues.push(format!(
                        "calibrated candidate status expected `{expected}` got `{:?}`",
                        calibrated_candidate_status
                    ));
                }
            }
            if let Some(expected) = expectations.preferred_calibration_candidate_status {
                if preferred_calibration_candidate_status.as_deref() != Some(expected) {
                    issues.push(format!(
                        "preferred calibration candidate status expected `{expected}` got `{:?}`",
                        preferred_calibration_candidate_status
                    ));
                }
            }
            if let Some(expected) = expectations.candidate_risk {
                if diagnostics.candidate_risk != expected {
                    issues.push(format!(
                        "candidate_risk expected `{expected}` got `{}`",
                        diagnostics.candidate_risk
                    ));
                }
            }
            if let Some(expected) = expectations.tone_color_trust_state {
                if tone_color_trust_state != expected {
                    issues.push(format!(
                        "tone_color_trust_state expected `{expected}` got `{}`",
                        tone_color_trust_state
                    ));
                }
            }
            if let Some(expected) = expectations.neutral_estimate_accepted {
                if diagnostics.neutral_estimate_quality.accepted != expected {
                    issues.push(format!(
                        "neutral_estimate_accepted expected `{expected}` got `{}`",
                        diagnostics.neutral_estimate_quality.accepted
                    ));
                }
            }
            if let Some(expected) = expectations.dominant_anchor_accepted {
                if diagnostics.dominant_anchor_quality.accepted != expected {
                    issues.push(format!(
                        "dominant_anchor_accepted expected `{expected}` got `{}`",
                        diagnostics.dominant_anchor_quality.accepted
                    ));
                }
            }
            if let Some(expected) = expectations.dominant_anchor_channel_populated_band_count {
                if diagnostics
                    .dominant_anchor_quality
                    .channel_populated_band_count
                    != expected
                {
                    issues.push(format!(
                        "dominant_anchor_channel_populated_band_count expected `{expected:?}` got `{:?}`",
                        diagnostics.dominant_anchor_quality.channel_populated_band_count
                    ));
                }
            }
            if let Some(expected) = expectations.dominant_anchor_channel_unstable {
                if diagnostics.dominant_anchor_quality.channel_unstable != expected {
                    issues.push(format!(
                        "dominant_anchor_channel_unstable expected `{expected:?}` got `{:?}`",
                        diagnostics.dominant_anchor_quality.channel_unstable
                    ));
                }
            }
            if let Some(expected) = expectations.channel_anchor_low_support {
                if diagnostics.channel_anchor_low_support != expected {
                    issues.push(format!(
                        "channel_anchor_low_support expected `{expected:?}` got `{:?}`",
                        diagnostics.channel_anchor_low_support
                    ));
                }
            }
            check_rejection_minima(
                "neutral_sample_rejections",
                expectations.neutral_rejection_minima.as_ref(),
                &neutral_sample_rejections,
                &mut issues,
            );
            check_dominant_rejection_minima(
                expectations.dominant_anchor_rejection_minima.as_ref(),
                &dominant_anchor_sample_rejections,
                &mut issues,
            );
            if let Some(minimum) = expectations.exposure_scale_min {
                if diagnostics.exposure_scale < minimum {
                    issues.push(format!(
                        "exposure_scale expected at least {minimum:.6} got {:.6}",
                        diagnostics.exposure_scale
                    ));
                }
            }
            if let Some(expected) = expectations.post_scale_high_clip_less_than_pre_scale {
                if post_scale_high_clip_less_than_pre_scale != expected {
                    issues.push(format!(
                        "post_scale_high_clip_less_than_pre_scale expected `{expected}` got `{post_scale_high_clip_less_than_pre_scale}` (pre={pre_scale_high_clip_total:.6}, post={post_scale_high_clip_total:.6})"
                    ));
                }
            }
            if let Some(minimum) = expectations.post_scale_preserved_ratio_min {
                if diagnostics.post_scale_preserved_ratio < minimum {
                    issues.push(format!(
                        "post_scale_preserved_ratio expected at least {minimum:.6} got {:.6}",
                        diagnostics.post_scale_preserved_ratio
                    ));
                }
            }
            if let Some(expected) = expectations.reference_patch_regresses_image_derived {
                let actual = diagnostics
                    .reference_patch_evaluation
                    .as_ref()
                    .map(|evaluation| evaluation.selected_regresses_image_derived);
                if actual != Some(expected) {
                    issues.push(format!(
                        "reference patch regression expected `{expected}` got `{:?}`",
                        actual
                    ));
                }
            }
            if let Some(expected) =
                expected_string_values(expectations.reference_patch_regressed_candidates.as_ref())
            {
                if actual_reference_patch_regressed_candidates != expected {
                    issues.push(format!(
                        "reference patch regressed candidates expected `{expected:?}` got `{:?}`",
                        actual_reference_patch_regressed_candidates
                    ));
                }
            }
            if let Some(expected) =
                expected_string_values(expectations.reference_patch_hue_family_regressions.as_ref())
            {
                if actual_reference_patch_hue_family_regressions != expected {
                    issues.push(format!(
                        "reference patch hue-family regressions expected `{expected:?}` got `{:?}`",
                        actual_reference_patch_hue_family_regressions
                    ));
                }
            }
            SyntheticColorCaseSummary {
                name: name.to_string(),
                status: if issues.is_empty() {
                    "passed"
                } else {
                    "failed"
                }
                .to_string(),
                expected_render_input_source: None,
                actual_render_input_source: None,
                expected_render_input_fallback_used: None,
                actual_render_input_fallback_used: None,
                expected_render_input_reason_contains: None,
                actual_render_input_reason: None,
                expected_selected_candidate: expectations.selected_candidate.map(str::to_string),
                actual_selected_candidate: Some(diagnostics.selected_candidate),
                expected_mapping_strategy: expectations.mapping_strategy.map(str::to_string),
                actual_mapping_strategy: Some(diagnostics.mapping_strategy.to_string()),
                expected_gamut_fallback_used: expectations.gamut_fallback_used,
                actual_gamut_fallback_used: Some(diagnostics.gamut_fallback_used),
                expected_image_matrix_pre_scale_low_clip_total_min: expectations
                    .image_matrix_pre_scale_low_clip_total_min,
                actual_image_matrix_pre_scale_low_clip_total: Some(
                    image_matrix_pre_scale_low_clip_total,
                ),
                expected_selected_pre_scale_low_clip_total_max: expectations
                    .selected_pre_scale_low_clip_total_max,
                actual_selected_pre_scale_low_clip_total: Some(selected_pre_scale_low_clip_total),
                expected_post_scale_low_clip_total_max: expectations.post_scale_low_clip_total_max,
                actual_post_scale_low_clip_total: Some(post_scale_low_clip_total),
                expected_calibration_acceptance_status: expectations
                    .calibration_acceptance_status
                    .map(str::to_string),
                actual_calibration_acceptance_status: Some(
                    diagnostics.calibration_acceptance.status,
                ),
                expected_calibration_beats_image_derived: expectations
                    .calibration_beats_image_derived,
                actual_calibration_beats_image_derived: diagnostics
                    .calibration_acceptance
                    .beats_image_derived,
                expected_calibrated_candidate_status: expectations
                    .calibrated_candidate_status
                    .map(str::to_string),
                actual_calibrated_candidate_status: calibrated_candidate_status,
                expected_preferred_calibration_candidate_status: expectations
                    .preferred_calibration_candidate_status
                    .map(str::to_string),
                actual_preferred_calibration_candidate_status:
                    preferred_calibration_candidate_status,
                expected_candidate_risk: expectations.candidate_risk.map(str::to_string),
                actual_candidate_risk: Some(diagnostics.candidate_risk.clone()),
                expected_tone_color_trust_state: expectations
                    .tone_color_trust_state
                    .map(str::to_string),
                actual_tone_color_trust_state: Some(tone_color_trust_state.to_string()),
                expected_tone_policy: None,
                actual_tone_policy: None,
                expected_tone_policy_color_trust_state: None,
                actual_tone_policy_color_trust_state: None,
                expected_tone_highlight_neutral_chroma_enabled: None,
                actual_tone_highlight_neutral_chroma_enabled: None,
                expected_tone_shadow_chroma_enabled: None,
                actual_tone_shadow_chroma_enabled: None,
                expected_tone_highlight_chroma_compressed_ratio_min: None,
                actual_tone_highlight_chroma_compressed_ratio: None,
                expected_tone_highlight_neutral_chroma_compressed_ratio_max: None,
                actual_tone_highlight_neutral_chroma_compressed_ratio: None,
                expected_tone_shadow_chroma_compressed_ratio_max: None,
                actual_tone_shadow_chroma_compressed_ratio: None,
                expected_neutral_estimate_accepted: expectations.neutral_estimate_accepted,
                actual_neutral_estimate_accepted: Some(
                    diagnostics.neutral_estimate_quality.accepted,
                ),
                expected_dominant_anchor_accepted: expectations.dominant_anchor_accepted,
                actual_dominant_anchor_accepted: Some(diagnostics.dominant_anchor_quality.accepted),
                expected_dominant_anchor_channel_populated_band_count: expectations
                    .dominant_anchor_channel_populated_band_count,
                actual_dominant_anchor_channel_populated_band_count: Some(
                    diagnostics
                        .dominant_anchor_quality
                        .channel_populated_band_count,
                ),
                expected_dominant_anchor_channel_unstable: expectations
                    .dominant_anchor_channel_unstable,
                actual_dominant_anchor_channel_unstable: Some(
                    diagnostics.dominant_anchor_quality.channel_unstable,
                ),
                expected_channel_anchor_low_support: expectations.channel_anchor_low_support,
                actual_channel_anchor_low_support: Some(diagnostics.channel_anchor_low_support),
                expected_neutral_rejection_minima: expectations.neutral_rejection_minima.clone(),
                actual_neutral_sample_rejections: Some(neutral_sample_rejections),
                expected_dominant_anchor_rejection_minima: expectations
                    .dominant_anchor_rejection_minima
                    .clone(),
                actual_dominant_anchor_sample_rejections: Some(dominant_anchor_sample_rejections),
                expected_exposure_scale_min: expectations.exposure_scale_min,
                actual_exposure_scale: Some(diagnostics.exposure_scale),
                expected_post_scale_high_clip_less_than_pre_scale: expectations
                    .post_scale_high_clip_less_than_pre_scale,
                actual_post_scale_high_clip_less_than_pre_scale: Some(
                    post_scale_high_clip_less_than_pre_scale,
                ),
                actual_pre_scale_high_clip_total: Some(pre_scale_high_clip_total),
                actual_post_scale_high_clip_total: Some(post_scale_high_clip_total),
                expected_post_scale_preserved_ratio_min: expectations
                    .post_scale_preserved_ratio_min,
                actual_post_scale_preserved_ratio: Some(diagnostics.post_scale_preserved_ratio),
                actual_density_monotonicity_score,
                actual_hue_linearity_score,
                actual_saturation_preservation_median_ratio,
                actual_spatial_neutral_delta_p95,
                actual_memory_color_penalty,
                actual_spatial_consistency_penalty,
                selected_quality_score: diagnostics.selected_quality_score,
                image_derived_quality_score: diagnostics
                    .candidate_scores
                    .iter()
                    .find(|candidate| candidate.candidate == "image_derived_matrix")
                    .map(|candidate| candidate.quality_score),
                preferred_calibration_quality_score: diagnostics
                    .calibration_acceptance
                    .preferred_candidate_quality_score,
                reference_patch_selected_regresses_image_derived: diagnostics
                    .reference_patch_evaluation
                    .map(|evaluation| evaluation.selected_regresses_image_derived),
                expected_reference_patch_regressed_candidates: expected_string_values(
                    expectations.reference_patch_regressed_candidates.as_ref(),
                ),
                actual_reference_patch_regressed_candidates,
                expected_reference_patch_hue_family_regressions: expected_string_values(
                    expectations.reference_patch_hue_family_regressions.as_ref(),
                ),
                actual_reference_patch_hue_family_regressions,
                actual_reference_patch_worst_hue_families,
                expected_error_contains: None,
                actual_error: None,
                issues,
            }
        }
        Err(err) => SyntheticColorCaseSummary {
            name: name.to_string(),
            status: "failed".to_string(),
            expected_render_input_source: None,
            actual_render_input_source: None,
            expected_render_input_fallback_used: None,
            actual_render_input_fallback_used: None,
            expected_render_input_reason_contains: None,
            actual_render_input_reason: None,
            expected_selected_candidate: expectations.selected_candidate.map(str::to_string),
            actual_selected_candidate: None,
            expected_mapping_strategy: expectations.mapping_strategy.map(str::to_string),
            actual_mapping_strategy: None,
            expected_gamut_fallback_used: expectations.gamut_fallback_used,
            actual_gamut_fallback_used: None,
            expected_image_matrix_pre_scale_low_clip_total_min: expectations
                .image_matrix_pre_scale_low_clip_total_min,
            actual_image_matrix_pre_scale_low_clip_total: None,
            expected_selected_pre_scale_low_clip_total_max: expectations
                .selected_pre_scale_low_clip_total_max,
            actual_selected_pre_scale_low_clip_total: None,
            expected_post_scale_low_clip_total_max: expectations.post_scale_low_clip_total_max,
            actual_post_scale_low_clip_total: None,
            expected_calibration_acceptance_status: expectations
                .calibration_acceptance_status
                .map(str::to_string),
            actual_calibration_acceptance_status: None,
            expected_calibration_beats_image_derived: expectations.calibration_beats_image_derived,
            actual_calibration_beats_image_derived: None,
            expected_calibrated_candidate_status: expectations
                .calibrated_candidate_status
                .map(str::to_string),
            actual_calibrated_candidate_status: None,
            expected_preferred_calibration_candidate_status: expectations
                .preferred_calibration_candidate_status
                .map(str::to_string),
            actual_preferred_calibration_candidate_status: None,
            expected_candidate_risk: expectations.candidate_risk.map(str::to_string),
            actual_candidate_risk: None,
            expected_tone_color_trust_state: expectations
                .tone_color_trust_state
                .map(str::to_string),
            actual_tone_color_trust_state: None,
            expected_tone_policy: None,
            actual_tone_policy: None,
            expected_tone_policy_color_trust_state: None,
            actual_tone_policy_color_trust_state: None,
            expected_tone_highlight_neutral_chroma_enabled: None,
            actual_tone_highlight_neutral_chroma_enabled: None,
            expected_tone_shadow_chroma_enabled: None,
            actual_tone_shadow_chroma_enabled: None,
            expected_tone_highlight_chroma_compressed_ratio_min: None,
            actual_tone_highlight_chroma_compressed_ratio: None,
            expected_tone_highlight_neutral_chroma_compressed_ratio_max: None,
            actual_tone_highlight_neutral_chroma_compressed_ratio: None,
            expected_tone_shadow_chroma_compressed_ratio_max: None,
            actual_tone_shadow_chroma_compressed_ratio: None,
            expected_neutral_estimate_accepted: expectations.neutral_estimate_accepted,
            actual_neutral_estimate_accepted: None,
            expected_dominant_anchor_accepted: expectations.dominant_anchor_accepted,
            actual_dominant_anchor_accepted: None,
            expected_dominant_anchor_channel_populated_band_count: expectations
                .dominant_anchor_channel_populated_band_count,
            actual_dominant_anchor_channel_populated_band_count: None,
            expected_dominant_anchor_channel_unstable: expectations
                .dominant_anchor_channel_unstable,
            actual_dominant_anchor_channel_unstable: None,
            expected_channel_anchor_low_support: expectations.channel_anchor_low_support,
            actual_channel_anchor_low_support: None,
            expected_neutral_rejection_minima: expectations.neutral_rejection_minima,
            actual_neutral_sample_rejections: None,
            expected_dominant_anchor_rejection_minima: expectations
                .dominant_anchor_rejection_minima,
            actual_dominant_anchor_sample_rejections: None,
            expected_exposure_scale_min: expectations.exposure_scale_min,
            actual_exposure_scale: None,
            expected_post_scale_high_clip_less_than_pre_scale: expectations
                .post_scale_high_clip_less_than_pre_scale,
            actual_post_scale_high_clip_less_than_pre_scale: None,
            actual_pre_scale_high_clip_total: None,
            actual_post_scale_high_clip_total: None,
            expected_post_scale_preserved_ratio_min: expectations.post_scale_preserved_ratio_min,
            actual_post_scale_preserved_ratio: None,
            actual_density_monotonicity_score: None,
            actual_hue_linearity_score: None,
            actual_saturation_preservation_median_ratio: None,
            actual_spatial_neutral_delta_p95: None,
            actual_memory_color_penalty: None,
            actual_spatial_consistency_penalty: None,
            selected_quality_score: None,
            image_derived_quality_score: None,
            preferred_calibration_quality_score: None,
            reference_patch_selected_regresses_image_derived: None,
            expected_reference_patch_regressed_candidates: expected_string_values(
                expectations.reference_patch_regressed_candidates.as_ref(),
            ),
            actual_reference_patch_regressed_candidates: Vec::new(),
            expected_reference_patch_hue_family_regressions: expected_string_values(
                expectations.reference_patch_hue_family_regressions.as_ref(),
            ),
            actual_reference_patch_hue_family_regressions: Vec::new(),
            actual_reference_patch_worst_hue_families: Vec::new(),
            expected_error_contains: None,
            actual_error: Some(err),
            issues: vec!["colorspace mapping returned an unexpected error".to_string()],
        },
    }
}

fn synthetic_calibration_profile() -> color_calibration::CalibrationProfile {
    let json = serde_json::json!({
        "schema_version": 1,
        "profile_id": "synthetic-prophoto-d50",
        "source_space": { "name": "synthetic linear scanner RGB", "encoding": "linear" },
        "scanner": { "make": "Synthetic", "model": "Validation Suite" },
        "film": { "stock": "Synthetic negative", "process": "C-41" },
        "target": { "type": "synthetic_color_checker", "illuminant": "D50" },
        "reference": { "dataset": "synthetic" },
        "whitepoint": [0.9642, 1.0, 0.8251],
        "work_to_xyz": [
            [0.7977, 0.1352, 0.0313],
            [0.2880, 0.7119, 0.0001],
            [0.0000, 0.0000, 0.8251]
        ],
        "confidence": 0.95
    })
    .to_string();
    color_calibration::parse_profile_json(&json, None)
        .profile
        .expect("synthetic calibration profile")
}

fn synthetic_nonlinear_target_patches(
    count: usize,
    prefix: &str,
    seed_offset: usize,
) -> Vec<color_calibration::TargetPatch> {
    let coefficients = [
        [0.65, 0.25, 0.03],
        [0.14, 0.69, 0.08],
        [0.05, 0.10, 0.76],
        [0.055, -0.030, 0.010],
        [-0.025, 0.012, 0.045],
        [0.012, 0.040, -0.018],
    ];
    (0..count)
        .map(|index| {
            let seed = index + seed_offset;
            let code =
                |multiplier: usize, add: usize| ((seed * multiplier + add) % 997) as f64 / 996.0;
            let source_rgb = [
                0.025 + 0.95 * code(173, 31),
                0.025 + 0.95 * code(379, 97),
                0.025 + 0.95 * code(613, 211),
            ];
            let basis = color_calibration::root_polynomial_basis_values(2, source_rgb)
                .expect("degree-two nonlinear validation basis");
            let reference_xyz = std::array::from_fn(|channel| {
                basis
                    .iter()
                    .zip(coefficients)
                    .map(|(term, coefficient)| term * coefficient[channel])
                    .sum()
            });
            color_calibration::TargetPatch {
                patch_id: Some(format!("{prefix}-{index:03}")),
                scanner_xy: None,
                source_rgb,
                reference_xyz,
            }
        })
        .collect()
}

fn synthetic_nonlinear_calibration_profile() -> (
    color_calibration::CalibrationProfile,
    Vec<color_calibration::TargetPatch>,
) {
    let training = synthetic_nonlinear_target_patches(48, "suite-train", 0);
    let held_out = synthetic_nonlinear_target_patches(36, "suite-held", 409);
    let matrix = color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training,
        &held_out,
        "suite_matrix_baseline",
        Some(crate::constants::D50_WHITE),
        None,
    )
    .expect("synthetic nonlinear matrix fit");
    let model = color_calibration::fit_root_polynomial_color_model_from_disjoint_patches(
        "suite-nonlinear",
        &training,
        &held_out,
        &matrix.matrix,
    )
    .expect("synthetic nonlinear model fit")
    .selected_model
    .expect("synthetic nonlinear model selection");
    let mut profile = synthetic_calibration_profile();
    profile.schema_version = color_calibration::CALIBRATION_PROFILE_SCHEMA_VERSION;
    profile.work_to_xyz = matrix.matrix;
    profile.whitepoint = crate::constants::D50_WHITE;
    profile.confidence = matrix.confidence;
    profile.matrix_condition_number = matrix.matrix_condition_number;
    profile.fit = Some(matrix.fit);
    profile.target_patches = held_out.clone();
    profile.application_mode = color_calibration::CalibrationApplicationMode::DirectProfile;
    profile.color_model = Some(model);
    profile.color_model_post_xyz = None;
    (profile, held_out)
}

fn synthetic_residual_lut_target_patches(
    count: usize,
    prefix: &str,
    seed_offset: usize,
    include_domain_corners: bool,
) -> Vec<color_calibration::TargetPatch> {
    let matrix = [[0.65, 0.14, 0.05], [0.25, 0.69, 0.10], [0.03, 0.08, 0.76]];
    (0..count)
        .map(|index| {
            let seed = index + seed_offset;
            let code =
                |multiplier: usize, add: usize| ((seed * multiplier + add) % 997) as f64 / 996.0;
            let source_rgb = if include_domain_corners && index < 8 {
                [
                    if index & 1 == 0 { 0.025 } else { 0.975 },
                    if index & 2 == 0 { 0.025 } else { 0.975 },
                    if index & 4 == 0 { 0.025 } else { 0.975 },
                ]
            } else {
                [
                    0.04 + 0.92 * code(173, 31),
                    0.04 + 0.92 * code(379, 97),
                    0.04 + 0.92 * code(613, 211),
                ]
            };
            let mut reference_xyz = std::array::from_fn(|row| {
                matrix[row][0] * source_rgb[0]
                    + matrix[row][1] * source_rgb[1]
                    + matrix[row][2] * source_rgb[2]
            });
            let normalized = source_rgb.map(|value| (value - 0.025) / 0.95);
            let shape = 64.0
                * normalized[0]
                * (1.0 - normalized[0])
                * normalized[1]
                * (1.0 - normalized[1])
                * normalized[2]
                * (1.0 - normalized[2]);
            let residual = [
                0.055 * shape * (0.70 + 0.30 * (2.0 * normalized[0] - 1.0)),
                -0.040 * shape * (0.75 + 0.25 * (2.0 * normalized[1] - 1.0)),
                0.050 * shape * (0.65 + 0.35 * (2.0 * normalized[2] - 1.0)),
            ];
            for channel in 0..3 {
                reference_xyz[channel] += residual[channel];
            }
            color_calibration::TargetPatch {
                patch_id: Some(format!("{prefix}-{index:03}")),
                scanner_xy: None,
                source_rgb,
                reference_xyz,
            }
        })
        .collect()
}

fn synthetic_residual_lut_calibration_profile() -> (
    color_calibration::CalibrationProfile,
    Vec<color_calibration::TargetPatch>,
) {
    let training = synthetic_residual_lut_target_patches(180, "suite-lut-train", 0, true);
    let held_out = synthetic_residual_lut_target_patches(80, "suite-lut-held", 431, false);
    let matrix = color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training,
        &held_out,
        "suite_lut_matrix_baseline",
        Some(crate::constants::D50_WHITE),
        None,
    )
    .expect("synthetic residual LUT matrix fit");
    let model = color_calibration::fit_residual_lut_3d_color_model_from_disjoint_patches(
        "suite-residual-lut",
        &training,
        &held_out,
        &matrix.matrix,
        None,
    )
    .expect("synthetic residual LUT fit")
    .selected_model
    .expect("synthetic residual LUT selection");
    let mut profile = synthetic_calibration_profile();
    profile.schema_version = color_calibration::CALIBRATION_PROFILE_SCHEMA_VERSION;
    profile.work_to_xyz = matrix.matrix;
    profile.whitepoint = crate::constants::D50_WHITE;
    profile.confidence = matrix.confidence;
    profile.matrix_condition_number = matrix.matrix_condition_number;
    profile.fit = Some(matrix.fit);
    profile.target_patches = held_out.clone();
    profile.application_mode = color_calibration::CalibrationApplicationMode::DirectProfile;
    profile.lut_3d_model = Some(model);
    profile.color_model_post_xyz = None;
    (profile, held_out)
}

fn synthetic_scanner_prior_profile() -> color_calibration::CalibrationProfile {
    let mut profile = synthetic_calibration_profile();
    profile.application_mode =
        color_calibration::CalibrationApplicationMode::ScannerConstrainedImageAdaptation;
    profile.work_to_xyz = prophoto_matrix_rows();
    profile.whitepoint = crate::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;
    profile
}

fn prophoto_matrix_rows() -> [[f64; 3]; 3] {
    let matrix = colorspace::prophoto_to_xyz_d50_matrix();
    matrix3_to_rows(matrix)
}

fn reference_fit_patches_for_matrix(
    matrix_to_prophoto: Matrix3<f64>,
) -> Vec<color_calibration::TargetPatch> {
    let prophoto_to_xyz = colorspace::prophoto_to_xyz_d50_matrix();
    [
        [0.78, 0.12, 0.10],
        [0.14, 0.72, 0.16],
        [0.12, 0.18, 0.76],
        [0.25, 0.25, 0.25],
        [0.52, 0.52, 0.52],
        [0.82, 0.82, 0.82],
    ]
    .into_iter()
    .enumerate()
    .map(|(idx, source_rgb)| {
        let xyz = prophoto_to_xyz
            * (matrix_to_prophoto * Vector3::new(source_rgb[0], source_rgb[1], source_rgb[2]));
        color_calibration::TargetPatch {
            patch_id: Some(format!("patch-{idx}")),
            scanner_xy: None,
            source_rgb,
            reference_xyz: [xyz[0], xyz[1], xyz[2]],
        }
    })
    .collect()
}

fn sparse_strong_anchor_image() -> Array3<f64> {
    let mut img = Array3::<f64>::zeros((40, 40, 3));
    for y in 0..40 {
        for x in 0..40 {
            let pixel = if x < 2 {
                [0.90, 0.22, 0.12]
            } else if x < 4 {
                [0.18, 0.82, 0.16]
            } else if x < 6 {
                [0.12, 0.20, 0.78]
            } else {
                [0.58, 0.56, 0.54]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }
    img
}

fn sparse_weak_green_anchor_image() -> Array3<f64> {
    let mut img = Array3::<f64>::zeros((20, 20, 3));
    for y in 0..20 {
        for x in 0..20 {
            let pixel = if x < 8 {
                [0.90, 0.22, 0.12]
            } else if x < 16 {
                [0.12, 0.20, 0.78]
            } else {
                [0.58, 0.56, 0.54]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }
    img
}

fn weak_anchor_scanner_prior_image() -> Array3<f64> {
    let mut img = Array3::<f64>::zeros((24, 24, 3));
    for y in 0..24 {
        for x in 0..24 {
            let pixel = if x < 3 {
                [0.90, 0.22, 0.12]
            } else if x < 6 {
                [0.12, 0.20, 0.78]
            } else {
                [0.58, 0.56, 0.54]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }
    img
}

fn broad_neutral_band_image(width: usize, height: usize) -> Array3<f64> {
    let mut img = Array3::<f64>::zeros((height, width, 3));
    for y in 0..height {
        let value = if y < height / 3 {
            0.18
        } else if y < 2 * height / 3 {
            0.50
        } else {
            0.82
        };
        for x in 0..width {
            for c in 0..3 {
                img[[y, x, c]] = value;
            }
        }
    }
    img
}

fn dirty_edge_rejection_image() -> Array3<f64> {
    let mut img = broad_neutral_band_image(30, 30);
    for y in 5..10 {
        for x in 5..10 {
            img[[y, x, 0]] = 0.32;
            img[[y, x, 1]] = 0.05;
            img[[y, x, 2]] = 0.05;
        }
    }
    for y in 10..15 {
        for x in 5..10 {
            img[[y, x, 0]] = 0.70;
            img[[y, x, 1]] = 0.18;
            img[[y, x, 2]] = 0.12;
        }
    }
    for y in 15..20 {
        for x in 5..10 {
            img[[y, x, 0]] = 0.95;
            img[[y, x, 1]] = 0.65;
            img[[y, x, 2]] = 0.55;
        }
    }

    img[[3, 3, 0]] = 0.9995;
    img[[3, 3, 1]] = 0.45;
    img[[3, 3, 2]] = 0.12;
    img[[4, 4, 0]] = 0.995;
    img[[4, 4, 1]] = 0.02;
    img[[4, 4, 2]] = 0.02;
    img[[5, 0, 0]] = 0.10;
    img[[5, 0, 1]] = 0.01;
    img[[5, 0, 2]] = 0.01;
    img[[29, 8, 0]] = 0.84;
    img[[29, 8, 1]] = 0.82;
    img[[29, 8, 2]] = 0.80;

    img
}

fn biased_single_band_anchor_image() -> Array3<f64> {
    let mut img = broad_neutral_band_image(45, 45);
    for y in 17..29 {
        for x in 4..17 {
            img[[y, x, 0]] = 0.55;
            img[[y, x, 1]] = 0.24;
            img[[y, x, 2]] = 0.20;
        }
        for x in 17..30 {
            img[[y, x, 0]] = 0.22;
            img[[y, x, 1]] = 0.55;
            img[[y, x, 2]] = 0.22;
        }
        for x in 30..43 {
            img[[y, x, 0]] = 0.20;
            img[[y, x, 1]] = 0.25;
            img[[y, x, 2]] = 0.56;
        }
    }
    img
}

fn high_key_headroom_image() -> Array3<f64> {
    let mut img = Array3::<f64>::zeros((40, 40, 3));
    for y in 0..40 {
        for x in 0..40 {
            let value = 0.80 + (x as f64 / 39.0) * 0.85 + (y as f64 / 39.0) * 0.15;
            let pixel = [value, value * 0.99, value * 1.01];
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }
    img
}

fn destructive_gamut_fallback_image() -> Array3<f64> {
    let mut img = Array3::<f64>::zeros((40, 40, 3));
    for y in 0..40 {
        for x in 0..40 {
            let pixel = if x < 10 {
                [0.73, 0.41, 0.34]
            } else if x < 20 {
                [0.34, 0.63, 0.32]
            } else if x < 30 {
                [0.35, 0.38, 0.62]
            } else {
                [0.44, 0.45, 0.49]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }
    img
}

fn matrix3_from_rows(rows: [[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::new(
        rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0],
        rows[2][1], rows[2][2],
    )
}

fn matrix3_to_rows(matrix: Matrix3<f64>) -> [[f64; 3]; 3] {
    [
        [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
        [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
        [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
    ]
}

pub fn summarize_report(fixture: impl Into<String>, report: &PipelineReport) -> ValidationSummary {
    summarize_report_with_source(fixture, report, None)
}

pub fn summarize_report_with_source(
    fixture: impl Into<String>,
    report: &PipelineReport,
    source_report_path: Option<&Path>,
) -> ValidationSummary {
    let load_phase = phase(report, "load");
    let deskew_phase = phase(report, "deskew");
    let border_phase = phase(report, "border_removal");
    let stitch_phase = phase(report, "stitch");
    let working_phase = phase(report, "working_image_select");
    let density_phase = phase(report, "density_inversion");
    let colorspace_phase = phase(report, "colorspace_mapping");
    let white_balance_phase = phase(report, "white_balance");
    let tone_phase = phase(report, "tone_mapping");
    let report_identity = summarize_report_identity(report, source_report_path);
    let input_orientation = summarize_input_orientation(load_phase);
    let deskew = summarize_deskew(deskew_phase);
    let border_crop = summarize_border_crop(border_phase);
    let stitch = summarize_stitch(stitch_phase);
    let base_density = BaseDensityValidationSummary {
        base_confidence: working_phase.map(|p| p.confidence),
        raw_base_confidence: working_phase.and_then(|p| f64_metric(p, "raw_base_confidence")),
        raw_base_proxy_confidence: working_phase
            .and_then(|p| f64_metric(p, "raw_base_proxy_confidence")),
        raw_base_support_fraction: working_phase
            .and_then(|p| f64_metric(p, "raw_base_support_fraction")),
        base_estimate_source: working_phase.and_then(|p| string_metric(p, "base_estimate_source")),
        density_confidence: density_phase.map(|p| p.confidence),
    };
    let negative_reconstruction =
        summarize_negative_reconstruction(density_phase, colorspace_phase);
    let colorspace = colorspace_phase
        .map(|phase| summarize_colorspace(phase, &report_identity))
        .unwrap_or_else(empty_colorspace_summary);
    let tone = tone_phase
        .map(summarize_tone)
        .unwrap_or_else(empty_tone_summary);
    let mut diagnostic_consistency_issues =
        calibration_color_mapping_consistency_issues(&colorspace);
    diagnostic_consistency_issues
        .extend(adaptive_vibrance_preferred_memory_color_guard_consistency_issues(&tone));
    diagnostic_consistency_issues.extend(preferred_skin_rendering_consistency_issues(&tone));
    diagnostic_consistency_issues.extend(grain_reduction_consistency_issues(&tone));
    let white_balance = summarize_white_balance(white_balance_phase, tone_phase);
    let render = summarize_render(
        report,
        &report_identity,
        &stitch,
        &base_density,
        &colorspace,
        &tone,
    );

    ValidationSummary {
        fixture: fixture.into(),
        report: report_identity,
        render,
        diagnostic_consistency_issues,
        input_orientation,
        deskew,
        border_crop,
        stitch,
        base_density,
        negative_reconstruction,
        colorspace,
        white_balance,
        tone,
        warnings: report
            .phases
            .iter()
            .filter(|phase| !phase.warnings.is_empty())
            .map(|phase| PhaseWarnings {
                phase: phase.name.clone(),
                warnings: phase.warnings.clone(),
            })
            .collect(),
        summary_baseline_comparison: None,
        comparison: None,
    }
}

pub fn compare_render_summaries(
    baseline_report_path: impl Into<String>,
    baseline: &RenderDiagnosticSummary,
    current_report_path: Option<String>,
    current: &RenderDiagnosticSummary,
) -> RenderComparisonSummary {
    let mut summary = RenderComparisonSummary {
        status: String::new(),
        issues: Vec::new(),
        baseline_report_path: baseline_report_path.into(),
        current_report_path,
        baseline: baseline.clone(),
        current: current.clone(),
        output_dimensions_match: match (
            baseline.output_width,
            baseline.output_height,
            current.output_width,
            current.output_height,
        ) {
            (Some(bw), Some(bh), Some(cw), Some(ch)) => Some(bw == cw && bh == ch),
            _ => None,
        },
        stitch_decision_changed: changed(&baseline.stitch_decision, &current.stitch_decision),
        base_estimate_source_changed: changed(
            &baseline.base_estimate_source,
            &current.base_estimate_source,
        ),
        render_input_source_changed: changed(
            &baseline.render_input_source,
            &current.render_input_source,
        ),
        colorspace_mapping_strategy_changed: changed(
            &baseline.colorspace_mapping_strategy,
            &current.colorspace_mapping_strategy,
        ),
        calibration_status_changed: changed(
            &baseline.calibration_status,
            &current.calibration_status,
        ),
        calibration_source_changed: changed(
            &baseline.calibration_source,
            &current.calibration_source,
        ),
        colorspace_candidate_risk_changed: changed(
            &baseline.colorspace_candidate_risk,
            &current.colorspace_candidate_risk,
        ),
        colorspace_tone_color_trust_state_changed: changed(
            &baseline.colorspace_tone_color_trust_state,
            &current.colorspace_tone_color_trust_state,
        ),
        render_review_status_changed: changed_expected(
            &baseline.render_review_status,
            &current.render_review_status,
        ),
        render_reviewable_changed: changed_expected(
            &baseline.render_reviewable,
            &current.render_reviewable,
        ),
        tone_output_confidence_status_changed: changed_expected(
            &baseline.tone_output_confidence_status,
            &current.tone_output_confidence_status,
        ),
        tone_output_review_required_changed: changed_expected(
            &baseline.tone_output_review_required,
            &current.tone_output_review_required,
        ),
        tone_output_evidence_confidence_delta: delta(
            baseline.tone_output_evidence_confidence,
            current.tone_output_evidence_confidence,
        ),
        tone_output_render_luminance_range_p05_p95_delta: delta(
            baseline.tone_output_render_luminance_range_p05_p95,
            current.tone_output_render_luminance_range_p05_p95,
        ),
        tone_output_render_to_mapped_luminance_range_ratio_delta: delta(
            baseline.tone_output_render_to_mapped_luminance_range_ratio,
            current.tone_output_render_to_mapped_luminance_range_ratio,
        ),
        tone_output_maximum_post_tone_high_clip_ratio_delta: delta(
            baseline.tone_output_maximum_post_tone_high_clip_ratio,
            current.tone_output_maximum_post_tone_high_clip_ratio,
        ),
        tone_output_maximum_post_tone_low_clip_ratio_delta: delta(
            baseline.tone_output_maximum_post_tone_low_clip_ratio,
            current.tone_output_maximum_post_tone_low_clip_ratio,
        ),
        colorspace_selected_quality_score_delta: delta(
            baseline.colorspace_selected_quality_score,
            current.colorspace_selected_quality_score,
        ),
        colorspace_density_monotonicity_score_delta: delta(
            baseline.colorspace_density_monotonicity_score,
            current.colorspace_density_monotonicity_score,
        ),
        colorspace_hue_linearity_score_delta: delta(
            baseline.colorspace_hue_linearity_score,
            current.colorspace_hue_linearity_score,
        ),
        colorspace_saturation_preservation_median_ratio_delta: delta(
            baseline.colorspace_saturation_preservation_median_ratio,
            current.colorspace_saturation_preservation_median_ratio,
        ),
        colorspace_spatial_neutral_delta_p95_delta: delta(
            baseline.colorspace_spatial_neutral_delta_p95,
            current.colorspace_spatial_neutral_delta_p95,
        ),
        colorspace_memory_color_penalty_delta: delta(
            baseline.colorspace_memory_color_penalty,
            current.colorspace_memory_color_penalty,
        ),
        colorspace_spatial_consistency_penalty_delta: delta(
            baseline.colorspace_spatial_consistency_penalty,
            current.colorspace_spatial_consistency_penalty,
        ),
        colorspace_reference_patch_rms_delta_e_delta: delta(
            baseline.colorspace_reference_patch_rms_delta_e,
            current.colorspace_reference_patch_rms_delta_e,
        ),
        colorspace_reference_patch_rms_delta_e2000_delta: delta(
            baseline.colorspace_reference_patch_rms_delta_e2000,
            current.colorspace_reference_patch_rms_delta_e2000,
        ),
        colorspace_post_scale_preserved_ratio_delta: delta(
            baseline.colorspace_post_scale_preserved_ratio,
            current.colorspace_post_scale_preserved_ratio,
        ),
        highlight_chroma_compressed_ratio_delta: delta(
            baseline.highlight_chroma_compressed_ratio,
            current.highlight_chroma_compressed_ratio,
        ),
        highlight_neutral_chroma_compressed_ratio_delta: delta(
            baseline.highlight_neutral_chroma_compressed_ratio,
            current.highlight_neutral_chroma_compressed_ratio,
        ),
        shadow_chroma_compressed_ratio_delta: delta(
            baseline.shadow_chroma_compressed_ratio,
            current.shadow_chroma_compressed_ratio,
        ),
        luma_residual_p95_ratio: ratio(
            baseline
                .high_frequency_grain
                .as_ref()
                .and_then(|grain| grain.luma_residual_p95),
            current
                .high_frequency_grain
                .as_ref()
                .and_then(|grain| grain.luma_residual_p95),
        ),
        chroma_residual_p95_ratio: ratio(
            baseline
                .high_frequency_grain
                .as_ref()
                .and_then(|grain| grain.chroma_residual_p95),
            current
                .high_frequency_grain
                .as_ref()
                .and_then(|grain| grain.chroma_residual_p95),
        ),
        chroma_to_luma_p95_ratio_delta: delta(
            baseline
                .high_frequency_grain
                .as_ref()
                .and_then(|grain| grain.chroma_to_luma_p95_ratio),
            current
                .high_frequency_grain
                .as_ref()
                .and_then(|grain| grain.chroma_to_luma_p95_ratio),
        ),
    };
    summary.issues = comparison_issues(&summary);
    summary.status = if summary.issues.is_empty() {
        "comparable".to_string()
    } else {
        "review_required".to_string()
    };
    summary
}

pub fn compare_summary_baseline(
    baseline_summary_path: impl Into<String>,
    baseline: &TrackedValidationBaseline,
    current: &ValidationSummary,
) -> SummaryBaselineComparison {
    let seam_applied = current
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| correction.applied);
    let seam_model = current
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| correction.model.clone());
    let seam_spatial_2d = current
        .stitch
        .seam_exposure_correction
        .as_ref()
        .and_then(|correction| correction.spatial_2d_validation.as_ref());
    let seam_spatial_quadratic =
        seam_spatial_2d.and_then(|validation| validation.quadratic.as_ref());
    let seam_blend = current.stitch.seam_blend.as_ref();
    let seam_detail = seam_blend.and_then(|blend| blend.detail_consistency.as_ref());
    let dimensions_expected =
        baseline.render.output_width.is_some() || baseline.render.output_height.is_some();
    let output_dimensions_match = if dimensions_expected {
        match (
            baseline.render.output_width,
            baseline.render.output_height,
            current.render.output_width,
            current.render.output_height,
        ) {
            (Some(bw), Some(bh), Some(cw), Some(ch)) => Some(bw == cw && bh == ch),
            _ => None,
        }
    } else {
        None
    };
    let current_grain = current.tone.high_frequency_grain.as_ref();
    let current_grain_detail = current.tone.grain_detail_retention.as_ref();
    let current_skin_memory = current
        .tone
        .adaptive_vibrance_skin_memory_protection
        .as_ref();
    let current_preferred_memory = current
        .tone
        .adaptive_vibrance_preferred_memory_color_guard
        .as_ref();
    let current_preferred_skin = current.tone.preferred_skin_rendering.as_ref();
    let reference_patch = current.colorspace.reference_patch_evaluation.as_ref();
    let neutral_quality = current.colorspace.neutral_estimate_quality.as_ref();
    let dominant_anchor_quality = current.colorspace.dominant_anchor_quality.as_ref();
    let selected_quality_components = current.colorspace.selected_quality_components.as_ref();
    let candidate_acceptance_signatures =
        candidate_acceptance_signatures(&current.colorspace.candidate_acceptance);
    let debug_artifact_issues = color_debug_artifact_issues(&current.colorspace.debug_artifacts);

    let mut comparison = SummaryBaselineComparison {
        status: String::new(),
        issues: Vec::new(),
        baseline_summary_path: baseline_summary_path.into(),
        tolerances: baseline.tolerances,
        fixture_matches: baseline
            .fixture
            .as_ref()
            .map(|fixture| fixture == &current.fixture),
        output_dimensions_match,
        output_color_space_changed: changed_expected(
            &baseline.render.output_color_space,
            &current.render.output_color_space,
        ),
        output_file_icc_profile_matches_report: match (
            baseline.render.output_file_icc_profile_matches_report,
            current.render.output_file_icc_profile_matches_report,
        ) {
            (Some(expected), Some(current)) => Some(expected == current),
            (Some(_), None) => Some(false),
            _ => current.render.output_file_icc_profile_matches_report,
        },
        stitch_decision_changed: changed_expected(
            &baseline.stitch.decision,
            &current.stitch.decision,
        ),
        stitch_confidence_delta: delta(baseline.stitch.confidence, current.stitch.confidence),
        chosen_hypothesis_changed: changed_expected(
            &baseline.stitch.chosen_hypothesis,
            &current.stitch.chosen_hypothesis,
        ),
        seam_exposure_correction_changed: changed_expected(
            &baseline.stitch.seam_exposure_correction_applied,
            &seam_applied,
        ),
        seam_exposure_model_changed: changed_expected(
            &baseline.stitch.seam_exposure_model,
            &seam_model,
        ),
        seam_exposure_spatial_2d_gain_acceptance_changed: changed_expected(
            &baseline.stitch.seam_exposure_spatial_2d_gain_accepted,
            &seam_spatial_2d.and_then(|validation| validation.gain_accepted),
        ),
        seam_exposure_spatial_2d_gain_offset_acceptance_changed: changed_expected(
            &baseline
                .stitch
                .seam_exposure_spatial_2d_gain_offset_accepted,
            &seam_spatial_2d.and_then(|validation| validation.gain_offset_accepted),
        ),
        seam_exposure_spatial_quadratic_gain_acceptance_changed: changed_expected(
            &baseline
                .stitch
                .seam_exposure_spatial_quadratic_gain_accepted,
            &seam_spatial_quadratic.and_then(|validation| validation.gain_accepted),
        ),
        seam_exposure_spatial_quadratic_gain_offset_acceptance_changed: changed_expected(
            &baseline
                .stitch
                .seam_exposure_spatial_quadratic_gain_offset_accepted,
            &seam_spatial_quadratic.and_then(|validation| validation.gain_offset_accepted),
        ),
        seam_blend_mode_changed: changed_expected(
            &baseline.stitch.seam_blend_mode,
            &seam_blend.and_then(|blend| blend.mode.clone()),
        ),
        seam_blend_applied_changed: changed_expected(
            &baseline.stitch.seam_blend_applied,
            &seam_blend.and_then(|blend| blend.applied),
        ),
        seam_blend_review_required_changed: changed_expected(
            &baseline.stitch.seam_blend_review_required,
            &seam_blend.and_then(|blend| blend.review_required),
        ),
        seam_detail_review_required_changed: changed_expected(
            &baseline.stitch.seam_detail_review_required,
            &seam_detail.and_then(|detail| detail.review_required),
        ),
        seam_detail_max_symmetric_energy_ratio_delta: delta(
            baseline.stitch.seam_detail_max_symmetric_energy_ratio,
            seam_detail.and_then(|detail| detail.maximum_symmetric_energy_ratio),
        ),
        seam_gradient_ratio_delta: delta(
            baseline.stitch.seam_gradient_ratio,
            seam_blend.and_then(|blend| blend.output_to_source_seam_gradient_ratio),
        ),
        seam_overlap_p95_abs_difference_delta: delta(
            baseline.stitch.seam_overlap_p95_abs_difference,
            seam_blend.and_then(|blend| blend.overlap_p95_abs_difference),
        ),
        base_estimate_source_changed: changed_expected(
            &baseline.render.base_estimate_source,
            &current.render.base_estimate_source,
        ),
        raw_base_proxy_confidence_delta: delta(
            baseline.render.raw_base_proxy_confidence,
            current.base_density.raw_base_proxy_confidence,
        ),
        raw_base_support_fraction_delta: delta(
            baseline.render.raw_base_support_fraction,
            current.base_density.raw_base_support_fraction,
        ),
        render_input_source_changed: changed_expected(
            &baseline.render.render_input_source,
            &current.render.render_input_source,
        ),
        colorspace_mapping_strategy_changed: changed_expected(
            &baseline.render.colorspace_mapping_strategy,
            &current.render.colorspace_mapping_strategy,
        ),
        calibration_status_changed: changed_expected(
            &baseline.colorspace.calibration_status,
            &current.colorspace.calibration_status,
        ),
        calibration_source_changed: changed_expected(
            &baseline.colorspace.calibration_source,
            &current.colorspace.calibration_source,
        ),
        calibration_scanner_profile_status_changed: changed_expected(
            &baseline.colorspace.calibration_scanner_profile_status,
            &current.colorspace.calibration_scanner_profile_status,
        ),
        calibration_scanner_profile_id_changed: changed_expected(
            &baseline.colorspace.calibration_scanner_profile_id,
            &current.colorspace.calibration_scanner_profile_id,
        ),
        calibration_roll_profile_status_changed: changed_expected(
            &baseline.colorspace.calibration_roll_profile_status,
            &current.colorspace.calibration_roll_profile_status,
        ),
        calibration_roll_profile_id_changed: changed_expected(
            &baseline.colorspace.calibration_roll_profile_id,
            &current.colorspace.calibration_roll_profile_id,
        ),
        calibration_confidence_delta: delta(
            baseline.colorspace.calibration_confidence,
            current.colorspace.calibration_confidence,
        ),
        calibration_matrix_condition_number_delta: delta(
            baseline.colorspace.calibration_matrix_condition_number,
            current.colorspace.calibration_matrix_condition_number,
        ),
        calibration_requested_film_stock_changed: changed_expected(
            &baseline.colorspace.calibration_requested_film_stock,
            &current.colorspace.calibration_requested_film_stock,
        ),
        calibration_film_stock_status_changed: changed_expected(
            &baseline.colorspace.calibration_film_stock_status,
            &current.colorspace.calibration_film_stock_status,
        ),
        calibration_film_stock_matched_roll_profiles_changed: !baseline
            .colorspace
            .calibration_film_stock_matched_roll_profiles
            .is_empty()
            && baseline
                .colorspace
                .calibration_film_stock_matched_roll_profiles
                != current
                    .colorspace
                    .calibration_film_stock_matched_roll_profiles,
        calibration_rejection_details_changed: !baseline
            .colorspace
            .calibration_rejection_details
            .is_empty()
            && baseline.colorspace.calibration_rejection_details
                != current.colorspace.calibration_rejection_details,
        selected_candidate_changed: changed_expected(
            &baseline.colorspace.selected_candidate,
            &current.colorspace.selected_candidate,
        ),
        selected_candidate_rank_changed: changed_expected(
            &baseline.colorspace.selected_candidate_rank,
            &current.colorspace.selected_candidate_rank,
        ),
        colorspace_candidate_acceptance_changed: !baseline
            .colorspace
            .candidate_acceptance_signatures
            .is_empty()
            && baseline.colorspace.candidate_acceptance_signatures
                != candidate_acceptance_signatures,
        calibration_acceptance_status_changed: changed_expected(
            &baseline.colorspace.calibration_acceptance_status,
            &current
                .colorspace
                .calibration_acceptance
                .as_ref()
                .and_then(|acceptance| acceptance.status.clone()),
        ),
        calibration_color_mapping_applied_changed: changed_expected(
            &baseline.colorspace.calibration_color_mapping_applied,
            &current
                .colorspace
                .calibration_color_mapping_application
                .as_ref()
                .and_then(|application| application.applied),
        ),
        calibration_acceptance_preferred_candidate_changed: changed_expected(
            &baseline
                .colorspace
                .calibration_acceptance_preferred_candidate,
            &current
                .colorspace
                .calibration_acceptance
                .as_ref()
                .and_then(|acceptance| acceptance.preferred_candidate.clone()),
        ),
        calibration_acceptance_beats_image_derived_changed: changed_expected(
            &baseline
                .colorspace
                .calibration_acceptance_beats_image_derived,
            &current
                .colorspace
                .calibration_acceptance
                .as_ref()
                .and_then(|acceptance| acceptance.beats_image_derived),
        ),
        colorspace_candidate_risk_changed: changed_expected(
            &baseline.colorspace.candidate_risk,
            &current.render.colorspace_candidate_risk,
        ),
        colorspace_tone_color_trust_state_changed: changed_expected(
            &baseline.colorspace.tone_color_trust_state,
            &current.render.colorspace_tone_color_trust_state,
        ),
        colorspace_selected_quality_score_delta: delta(
            baseline.colorspace.selected_quality_score,
            current.render.colorspace_selected_quality_score,
        ),
        colorspace_technical_safety_score_delta: delta(
            baseline.colorspace.technical_safety_score,
            current.colorspace.technical_safety_score,
        ),
        colorspace_color_fidelity_score_delta: delta(
            baseline.colorspace.color_fidelity_score,
            current.colorspace.color_fidelity_score,
        ),
        colorspace_selected_runner_up_quality_delta_delta: delta(
            baseline.colorspace.selected_runner_up_quality_delta,
            current.colorspace.selected_runner_up_quality_delta,
        ),
        colorspace_density_monotonicity_score_delta: delta(
            baseline.colorspace.density_monotonicity_score,
            current.render.colorspace_density_monotonicity_score,
        ),
        colorspace_hue_linearity_score_delta: delta(
            baseline.colorspace.hue_linearity_score,
            current.render.colorspace_hue_linearity_score,
        ),
        colorspace_saturation_preservation_median_ratio_delta: delta(
            baseline.colorspace.saturation_preservation_median_ratio,
            current
                .render
                .colorspace_saturation_preservation_median_ratio,
        ),
        colorspace_spatial_neutral_delta_p95_delta: delta(
            baseline.colorspace.spatial_neutral_delta_p95,
            current.render.colorspace_spatial_neutral_delta_p95,
        ),
        colorspace_memory_color_penalty_delta: delta(
            baseline.colorspace.memory_color_penalty,
            selected_quality_components.and_then(|components| components.memory_color_penalty),
        ),
        colorspace_spatial_consistency_penalty_delta: delta(
            baseline.colorspace.spatial_consistency_penalty,
            selected_quality_components
                .and_then(|components| components.spatial_consistency_penalty),
        ),
        colorspace_post_scale_preserved_ratio_delta: delta(
            baseline.colorspace.post_scale_preserved_ratio,
            current.render.colorspace_post_scale_preserved_ratio,
        ),
        colorspace_reference_patch_selected_rms_delta_e_delta: delta(
            baseline.colorspace.reference_patch_selected_rms_delta_e,
            reference_patch.and_then(|evaluation| evaluation.selected_rms_delta_e),
        ),
        colorspace_reference_patch_selected_rms_delta_e2000_delta: delta(
            baseline.colorspace.reference_patch_selected_rms_delta_e2000,
            reference_patch.and_then(|evaluation| evaluation.selected_rms_delta_e2000),
        ),
        colorspace_reference_patch_max_error_delta_vs_image_derived_delta: delta(
            baseline
                .colorspace
                .reference_patch_max_error_delta_vs_image_derived,
            reference_patch.and_then(|evaluation| evaluation.max_error_delta_vs_image_derived),
        ),
        colorspace_reference_patch_delta_e_rms_delta_vs_image_derived_delta: delta(
            baseline
                .colorspace
                .reference_patch_delta_e_rms_delta_vs_image_derived,
            reference_patch.and_then(|evaluation| evaluation.delta_e_rms_delta_vs_image_derived),
        ),
        colorspace_reference_patch_delta_e_max_delta_vs_image_derived_delta: delta(
            baseline
                .colorspace
                .reference_patch_delta_e_max_delta_vs_image_derived,
            reference_patch.and_then(|evaluation| evaluation.delta_e_max_delta_vs_image_derived),
        ),
        colorspace_reference_patch_delta_e2000_rms_delta_vs_image_derived_delta: delta(
            baseline
                .colorspace
                .reference_patch_delta_e2000_rms_delta_vs_image_derived,
            reference_patch
                .and_then(|evaluation| evaluation.delta_e2000_rms_delta_vs_image_derived),
        ),
        colorspace_reference_patch_delta_e2000_max_delta_vs_image_derived_delta: delta(
            baseline
                .colorspace
                .reference_patch_delta_e2000_max_delta_vs_image_derived,
            reference_patch
                .and_then(|evaluation| evaluation.delta_e2000_max_delta_vs_image_derived),
        ),
        colorspace_reference_patch_selected_regresses_image_derived_changed: changed_expected(
            &baseline
                .colorspace
                .reference_patch_selected_regresses_image_derived,
            &reference_patch.and_then(|evaluation| evaluation.selected_regresses_image_derived),
        ),
        colorspace_reference_patch_worst_hue_families_changed: !baseline
            .colorspace
            .reference_patch_worst_hue_families
            .is_empty()
            && baseline.colorspace.reference_patch_worst_hue_families
                != reference_patch
                    .map(|evaluation| evaluation.worst_hue_families.clone())
                    .unwrap_or_default(),
        colorspace_reference_patch_hue_family_regressions_changed: !baseline
            .colorspace
            .reference_patch_hue_family_regressions
            .is_empty()
            && baseline.colorspace.reference_patch_hue_family_regressions
                != reference_patch
                    .map(|evaluation| evaluation.hue_family_regressions.clone())
                    .unwrap_or_default(),
        colorspace_reference_patch_regressed_candidates_changed: !baseline
            .colorspace
            .reference_patch_regressed_candidates
            .is_empty()
            && baseline.colorspace.reference_patch_regressed_candidates
                != reference_patch
                    .map(|evaluation| evaluation.regressed_candidates.clone())
                    .unwrap_or_default(),
        colorspace_neutral_estimate_score_delta: delta(
            baseline.colorspace.neutral_estimate_score,
            neutral_quality.and_then(|quality| quality.score),
        ),
        colorspace_neutral_estimate_accepted_changed: changed_expected(
            &baseline.colorspace.neutral_estimate_accepted,
            &neutral_quality.and_then(|quality| quality.accepted),
        ),
        colorspace_neutral_estimate_populated_band_count_changed: changed_expected(
            &baseline.colorspace.neutral_estimate_populated_band_count,
            &neutral_quality.and_then(|quality| quality.populated_band_count),
        ),
        colorspace_neutral_estimate_dominant_band_fraction_delta: delta(
            baseline.colorspace.neutral_estimate_dominant_band_fraction,
            neutral_quality.and_then(|quality| quality.dominant_band_fraction),
        ),
        colorspace_dominant_anchor_score_delta: delta(
            baseline.colorspace.dominant_anchor_score,
            dominant_anchor_quality.and_then(|quality| quality.score),
        ),
        colorspace_dominant_anchor_accepted_changed: changed_expected(
            &baseline.colorspace.dominant_anchor_accepted,
            &dominant_anchor_quality.and_then(|quality| quality.accepted),
        ),
        colorspace_dominant_anchor_unstable_channel_count_changed: changed_expected(
            &baseline.colorspace.dominant_anchor_unstable_channel_count,
            &dominant_anchor_quality.and_then(|quality| quality.unstable_channel_count),
        ),
        colorspace_dominant_anchor_channel_unstable_changed: !baseline
            .colorspace
            .dominant_anchor_channel_unstable
            .is_empty()
            && baseline.colorspace.dominant_anchor_channel_unstable
                != dominant_anchor_quality
                    .and_then(|quality| quality.channel_unstable.clone())
                    .unwrap_or_default(),
        colorspace_channel_anchor_min_count_changed: changed_expected(
            &baseline.colorspace.channel_anchor_min_count,
            &current.colorspace.channel_anchor_min_count,
        ),
        colorspace_channel_anchor_low_support_changed: !baseline
            .colorspace
            .channel_anchor_low_support
            .is_empty()
            && baseline.colorspace.channel_anchor_low_support
                != current
                    .colorspace
                    .channel_anchor_low_support
                    .clone()
                    .unwrap_or_default(),
        colorspace_weak_anchor_fallback_changed: changed_expected(
            &baseline.colorspace.weak_anchor_fallback_used,
            &current.colorspace.weak_anchor_fallback_used,
        ),
        colorspace_gamut_fallback_changed: changed_expected(
            &baseline.colorspace.gamut_fallback_used,
            &current.colorspace.gamut_fallback_used,
        ),
        colorspace_neutral_trim_applied_changed: changed_expected(
            &baseline.colorspace.neutral_trim_applied,
            &current.colorspace.neutral_trim_applied,
        ),
        colorspace_debug_artifact_invalid_count: debug_artifact_issues.len(),
        colorspace_debug_artifact_issues: debug_artifact_issues,
        render_review_status_changed: changed_expected(
            &baseline.render.render_review_status,
            &current.render.render_review_status,
        ),
        render_reviewable_changed: changed_expected(
            &baseline.render.render_reviewable,
            &current.render.render_reviewable,
        ),
        tone_output_confidence_status_changed: changed_expected(
            &baseline.tone.tone_output_confidence_status,
            &current.tone.tone_output_confidence_status,
        ),
        tone_output_review_required_changed: changed_expected(
            &baseline.tone.tone_output_review_required,
            &current.tone.tone_output_review_required,
        ),
        tone_output_evidence_confidence_delta: delta(
            baseline.tone.tone_output_evidence_confidence,
            current.tone.tone_output_evidence_confidence,
        ),
        tone_output_render_luminance_range_p05_p95_delta: delta(
            baseline.tone.render_luminance_range_p05_p95,
            current.tone.render_luminance_range_p05_p95,
        ),
        tone_output_render_to_mapped_luminance_range_ratio_delta: delta(
            baseline.tone.render_to_mapped_luminance_range_ratio,
            current.tone.render_to_mapped_luminance_range_ratio,
        ),
        tone_output_maximum_post_tone_high_clip_ratio_delta: delta(
            baseline.tone.maximum_post_tone_high_clip_ratio,
            current.tone.maximum_post_tone_high_clip_ratio,
        ),
        tone_output_maximum_post_tone_low_clip_ratio_delta: delta(
            baseline.tone.maximum_post_tone_low_clip_ratio,
            current.tone.maximum_post_tone_low_clip_ratio,
        ),
        highlight_chroma_compressed_ratio_delta: delta(
            baseline.tone.highlight_chroma_compressed_ratio,
            current.tone.highlight_chroma_compressed_ratio,
        ),
        highlight_neutral_chroma_compressed_ratio_delta: delta(
            baseline.tone.highlight_neutral_chroma_compressed_ratio,
            current.tone.highlight_neutral_chroma_compressed_ratio,
        ),
        shadow_chroma_compressed_ratio_delta: delta(
            baseline.tone.shadow_chroma_compressed_ratio,
            current.tone.shadow_chroma_compressed_ratio,
        ),
        adaptive_vibrance_skin_memory_protection_enabled_changed: changed_expected(
            &baseline
                .tone
                .adaptive_vibrance_skin_memory_protection_enabled,
            &current_skin_memory.and_then(|protection| protection.enabled),
        ),
        adaptive_vibrance_skin_memory_protection_space_changed: changed_expected(
            &baseline.tone.adaptive_vibrance_skin_memory_protection_space,
            &current_skin_memory.and_then(|protection| protection.working_space.clone()),
        ),
        adaptive_vibrance_skin_memory_protected_ratio_delta: delta(
            baseline.tone.adaptive_vibrance_skin_memory_protected_ratio,
            current_skin_memory.and_then(|protection| protection.protected_pixel_ratio),
        ),
        adaptive_vibrance_skin_memory_mean_protection_delta: delta(
            baseline.tone.adaptive_vibrance_skin_memory_mean_protection,
            current_skin_memory.and_then(|protection| protection.mean_protection_weight),
        ),
        adaptive_vibrance_preferred_memory_color_guard_enabled_changed: changed_expected(
            &baseline
                .tone
                .adaptive_vibrance_preferred_memory_color_guard_enabled,
            &current_preferred_memory.and_then(|guard| guard.enabled),
        ),
        adaptive_vibrance_preferred_memory_color_guard_space_changed: changed_expected(
            &baseline
                .tone
                .adaptive_vibrance_preferred_memory_color_guard_space,
            &current_preferred_memory.and_then(|guard| guard.working_space.clone()),
        ),
        adaptive_vibrance_preferred_memory_color_guard_reference_changed: changed_expected(
            &baseline
                .tone
                .adaptive_vibrance_preferred_memory_color_guard_reference,
            &current_preferred_memory.and_then(|guard| guard.reference.clone()),
        ),
        adaptive_vibrance_preferred_memory_color_matched_ratio_delta: delta(
            baseline
                .tone
                .adaptive_vibrance_preferred_memory_color_matched_ratio,
            current_preferred_memory.and_then(|guard| guard.matched_pixel_ratio),
        ),
        adaptive_vibrance_preferred_memory_color_limited_ratio_delta: delta(
            baseline
                .tone
                .adaptive_vibrance_preferred_memory_color_limited_ratio,
            current_preferred_memory.and_then(|guard| guard.limited_pixel_ratio),
        ),
        adaptive_vibrance_preferred_memory_color_mean_scale_reduction_delta: delta(
            baseline
                .tone
                .adaptive_vibrance_preferred_memory_color_mean_scale_reduction,
            current_preferred_memory.and_then(|guard| guard.mean_scale_reduction),
        ),
        preferred_skin_rendering_enabled_changed: changed_expected(
            &baseline.tone.preferred_skin_rendering_enabled,
            &current_preferred_skin.and_then(|rendering| rendering.enabled),
        ),
        preferred_skin_rendering_space_changed: changed_expected(
            &baseline.tone.preferred_skin_rendering_space,
            &current_preferred_skin.and_then(|rendering| rendering.working_space.clone()),
        ),
        preferred_skin_rendering_preference_reference_changed: changed_expected(
            &baseline.tone.preferred_skin_rendering_preference_reference,
            &current_preferred_skin.and_then(|rendering| rendering.preference_reference.clone()),
        ),
        preferred_skin_rendering_adjusted_ratio_delta: delta(
            baseline.tone.preferred_skin_rendering_adjusted_ratio,
            current_preferred_skin.and_then(|rendering| rendering.adjusted_pixel_ratio),
        ),
        preferred_skin_rendering_mean_delta_e_ab_delta: delta(
            baseline.tone.preferred_skin_rendering_mean_delta_e_ab,
            current_preferred_skin.and_then(|rendering| rendering.mean_delta_e_ab),
        ),
        luma_residual_p95_ratio: ratio(
            baseline.grain.luma_residual_p95,
            current_grain.and_then(|grain| grain.luma_residual_p95),
        ),
        chroma_residual_p95_ratio: ratio(
            baseline.grain.chroma_residual_p95,
            current_grain.and_then(|grain| grain.chroma_residual_p95),
        ),
        chroma_to_luma_p95_ratio_delta: delta(
            baseline.grain.chroma_to_luma_p95_ratio,
            current_grain.and_then(|grain| grain.chroma_to_luma_p95_ratio),
        ),
        grain_detail_review_required_changed: changed_expected(
            &baseline.grain.detail_review_required,
            &current_grain_detail.and_then(|detail| detail.review_required),
        ),
        grain_detail_decision_supported_changed: changed_expected(
            &baseline.grain.detail_decision_supported,
            &current_grain_detail.and_then(|detail| detail.decision_supported),
        ),
        grain_detail_luminance_p10_retention_delta: delta(
            baseline.grain.luminance_p10_retention,
            current_grain_detail.and_then(|detail| detail.luminance_p10_retention),
        ),
        grain_detail_chroma_p10_retention_delta: delta(
            baseline.grain.chroma_p10_retention,
            current_grain_detail.and_then(|detail| detail.chroma_p10_retention),
        ),
    };
    comparison.issues = summary_baseline_issues(&comparison, baseline);
    comparison.status = if comparison.issues.is_empty() {
        "comparable".to_string()
    } else {
        "review_required".to_string()
    };
    comparison
}

pub fn tracked_baseline_from_summary(summary: &ValidationSummary) -> TrackedValidationBaseline {
    let seam_exposure = summary.stitch.seam_exposure_correction.as_ref();
    let seam_exposure_correction_applied = seam_exposure.and_then(|correction| correction.applied);
    let seam_exposure_model = seam_exposure.and_then(|correction| correction.model.clone());
    let seam_spatial_2d =
        seam_exposure.and_then(|correction| correction.spatial_2d_validation.as_ref());
    let seam_spatial_quadratic =
        seam_spatial_2d.and_then(|validation| validation.quadratic.as_ref());
    let seam_blend = summary.stitch.seam_blend.as_ref();
    let seam_detail = seam_blend.and_then(|blend| blend.detail_consistency.as_ref());
    let calibration_acceptance = summary.colorspace.calibration_acceptance.as_ref();
    let reference_patch = summary.colorspace.reference_patch_evaluation.as_ref();
    let neutral_quality = summary.colorspace.neutral_estimate_quality.as_ref();
    let dominant_anchor_quality = summary.colorspace.dominant_anchor_quality.as_ref();
    let grain = summary.tone.high_frequency_grain.as_ref();
    let grain_detail = summary.tone.grain_detail_retention.as_ref();
    let skin_memory = summary
        .tone
        .adaptive_vibrance_skin_memory_protection
        .as_ref();
    let preferred_memory = summary
        .tone
        .adaptive_vibrance_preferred_memory_color_guard
        .as_ref();
    let preferred_skin = summary.tone.preferred_skin_rendering.as_ref();

    TrackedValidationBaseline {
        fixture: Some(summary.fixture.clone()),
        stitch: TrackedStitchBaseline {
            decision: summary.stitch.decision.clone(),
            confidence: summary.stitch.confidence,
            chosen_hypothesis: summary.stitch.chosen_hypothesis.clone(),
            seam_exposure_correction_applied,
            seam_exposure_model,
            seam_exposure_spatial_2d_gain_accepted: seam_spatial_2d
                .and_then(|validation| validation.gain_accepted),
            seam_exposure_spatial_2d_gain_offset_accepted: seam_spatial_2d
                .and_then(|validation| validation.gain_offset_accepted),
            seam_exposure_spatial_quadratic_gain_accepted: seam_spatial_quadratic
                .and_then(|validation| validation.gain_accepted),
            seam_exposure_spatial_quadratic_gain_offset_accepted: seam_spatial_quadratic
                .and_then(|validation| validation.gain_offset_accepted),
            seam_blend_mode: seam_blend.and_then(|blend| blend.mode.clone()),
            seam_blend_applied: seam_blend.and_then(|blend| blend.applied),
            seam_blend_review_required: seam_blend.and_then(|blend| blend.review_required),
            seam_detail_review_required: seam_detail.and_then(|detail| detail.review_required),
            seam_detail_max_symmetric_energy_ratio: seam_detail
                .and_then(|detail| detail.maximum_symmetric_energy_ratio),
            seam_gradient_ratio: seam_blend
                .and_then(|blend| blend.output_to_source_seam_gradient_ratio),
            seam_overlap_p95_abs_difference: seam_blend
                .and_then(|blend| blend.overlap_p95_abs_difference),
        },
        render: TrackedRenderBaseline {
            output_width: summary.render.output_width,
            output_height: summary.render.output_height,
            output_color_space: summary.render.output_color_space.clone(),
            output_file_icc_profile_matches_report: summary
                .render
                .output_file_icc_profile_matches_report,
            base_estimate_source: summary.render.base_estimate_source.clone(),
            raw_base_proxy_confidence: summary.base_density.raw_base_proxy_confidence,
            raw_base_support_fraction: summary.base_density.raw_base_support_fraction,
            render_input_source: summary.render.render_input_source.clone(),
            colorspace_mapping_strategy: summary.render.colorspace_mapping_strategy.clone(),
            render_review_status: summary.render.render_review_status.clone(),
            render_reviewable: summary.render.render_reviewable,
        },
        colorspace: TrackedColorspaceBaseline {
            calibration_status: summary.colorspace.calibration_status.clone(),
            calibration_source: summary.colorspace.calibration_source.clone(),
            calibration_scanner_profile_status: summary
                .colorspace
                .calibration_scanner_profile_status
                .clone(),
            calibration_scanner_profile_id: summary
                .colorspace
                .calibration_scanner_profile_id
                .clone(),
            calibration_roll_profile_status: summary
                .colorspace
                .calibration_roll_profile_status
                .clone(),
            calibration_roll_profile_id: summary.colorspace.calibration_roll_profile_id.clone(),
            calibration_confidence: summary.colorspace.calibration_confidence,
            calibration_matrix_condition_number: summary
                .colorspace
                .calibration_matrix_condition_number,
            calibration_requested_film_stock: summary
                .colorspace
                .calibration_requested_film_stock
                .clone(),
            calibration_film_stock_status: summary.colorspace.calibration_film_stock_status.clone(),
            calibration_film_stock_matched_roll_profiles: summary
                .colorspace
                .calibration_film_stock_matched_roll_profiles
                .clone(),
            calibration_rejection_details: summary.colorspace.calibration_rejection_details.clone(),
            selected_candidate: summary.colorspace.selected_candidate.clone(),
            selected_candidate_rank: summary.colorspace.selected_candidate_rank,
            calibration_acceptance_status: calibration_acceptance
                .and_then(|acceptance| acceptance.status.clone()),
            calibration_color_mapping_applied: summary
                .colorspace
                .calibration_color_mapping_application
                .as_ref()
                .and_then(|application| application.applied),
            calibration_acceptance_preferred_candidate: calibration_acceptance
                .and_then(|acceptance| acceptance.preferred_candidate.clone()),
            calibration_acceptance_beats_image_derived: calibration_acceptance
                .and_then(|acceptance| acceptance.beats_image_derived),
            candidate_risk: summary.render.colorspace_candidate_risk.clone(),
            tone_color_trust_state: summary.render.colorspace_tone_color_trust_state.clone(),
            selected_quality_score: summary.render.colorspace_selected_quality_score,
            technical_safety_score: summary.colorspace.technical_safety_score,
            color_fidelity_score: summary.colorspace.color_fidelity_score,
            selected_runner_up_quality_delta: summary.colorspace.selected_runner_up_quality_delta,
            density_monotonicity_score: summary.render.colorspace_density_monotonicity_score,
            hue_linearity_score: summary.render.colorspace_hue_linearity_score,
            saturation_preservation_median_ratio: summary
                .render
                .colorspace_saturation_preservation_median_ratio,
            spatial_neutral_delta_p95: summary.render.colorspace_spatial_neutral_delta_p95,
            memory_color_penalty: summary
                .colorspace
                .selected_quality_components
                .as_ref()
                .and_then(|components| components.memory_color_penalty),
            spatial_consistency_penalty: summary
                .colorspace
                .selected_quality_components
                .as_ref()
                .and_then(|components| components.spatial_consistency_penalty),
            post_scale_preserved_ratio: summary.render.colorspace_post_scale_preserved_ratio,
            candidate_acceptance_signatures: candidate_acceptance_signatures(
                &summary.colorspace.candidate_acceptance,
            ),
            reference_patch_selected_rms_delta_e: reference_patch
                .and_then(|evaluation| evaluation.selected_rms_delta_e),
            reference_patch_selected_rms_delta_e2000: reference_patch
                .and_then(|evaluation| evaluation.selected_rms_delta_e2000),
            reference_patch_max_error_delta_vs_image_derived: reference_patch
                .and_then(|evaluation| evaluation.max_error_delta_vs_image_derived),
            reference_patch_delta_e_rms_delta_vs_image_derived: reference_patch
                .and_then(|evaluation| evaluation.delta_e_rms_delta_vs_image_derived),
            reference_patch_delta_e_max_delta_vs_image_derived: reference_patch
                .and_then(|evaluation| evaluation.delta_e_max_delta_vs_image_derived),
            reference_patch_delta_e2000_rms_delta_vs_image_derived: reference_patch
                .and_then(|evaluation| evaluation.delta_e2000_rms_delta_vs_image_derived),
            reference_patch_delta_e2000_max_delta_vs_image_derived: reference_patch
                .and_then(|evaluation| evaluation.delta_e2000_max_delta_vs_image_derived),
            reference_patch_selected_regresses_image_derived: reference_patch
                .and_then(|evaluation| evaluation.selected_regresses_image_derived),
            reference_patch_worst_hue_families: reference_patch
                .map(|evaluation| evaluation.worst_hue_families.clone())
                .unwrap_or_default(),
            reference_patch_hue_family_regressions: reference_patch
                .map(|evaluation| evaluation.hue_family_regressions.clone())
                .unwrap_or_default(),
            reference_patch_regressed_candidates: reference_patch
                .map(|evaluation| evaluation.regressed_candidates.clone())
                .unwrap_or_default(),
            neutral_estimate_score: neutral_quality.and_then(|quality| quality.score),
            neutral_estimate_accepted: neutral_quality.and_then(|quality| quality.accepted),
            neutral_estimate_populated_band_count: neutral_quality
                .and_then(|quality| quality.populated_band_count),
            neutral_estimate_dominant_band_fraction: neutral_quality
                .and_then(|quality| quality.dominant_band_fraction),
            dominant_anchor_score: dominant_anchor_quality.and_then(|quality| quality.score),
            dominant_anchor_accepted: dominant_anchor_quality.and_then(|quality| quality.accepted),
            dominant_anchor_unstable_channel_count: dominant_anchor_quality
                .and_then(|quality| quality.unstable_channel_count),
            dominant_anchor_channel_unstable: dominant_anchor_quality
                .and_then(|quality| quality.channel_unstable.clone())
                .unwrap_or_default(),
            channel_anchor_min_count: summary.colorspace.channel_anchor_min_count,
            channel_anchor_low_support: summary
                .colorspace
                .channel_anchor_low_support
                .clone()
                .unwrap_or_default(),
            weak_anchor_fallback_used: summary.colorspace.weak_anchor_fallback_used,
            gamut_fallback_used: summary.colorspace.gamut_fallback_used,
            neutral_trim_applied: summary.colorspace.neutral_trim_applied,
        },
        tolerances: ValidationBaselineTolerances::default(),
        tone: TrackedToneBaseline {
            highlight_chroma_compressed_ratio: summary.tone.highlight_chroma_compressed_ratio,
            highlight_neutral_chroma_compressed_ratio: summary
                .tone
                .highlight_neutral_chroma_compressed_ratio,
            shadow_chroma_compressed_ratio: summary.tone.shadow_chroma_compressed_ratio,
            tone_output_confidence_status: summary.tone.tone_output_confidence_status.clone(),
            tone_output_review_required: summary.tone.tone_output_review_required,
            tone_output_evidence_confidence: summary.tone.tone_output_evidence_confidence,
            render_luminance_range_p05_p95: summary.tone.render_luminance_range_p05_p95,
            render_to_mapped_luminance_range_ratio: summary
                .tone
                .render_to_mapped_luminance_range_ratio,
            maximum_post_tone_high_clip_ratio: summary.tone.maximum_post_tone_high_clip_ratio,
            maximum_post_tone_low_clip_ratio: summary.tone.maximum_post_tone_low_clip_ratio,
            adaptive_vibrance_skin_memory_protection_enabled: skin_memory
                .and_then(|protection| protection.enabled),
            adaptive_vibrance_skin_memory_protection_space: skin_memory
                .and_then(|protection| protection.working_space.clone()),
            adaptive_vibrance_skin_memory_protected_ratio: skin_memory
                .and_then(|protection| protection.protected_pixel_ratio),
            adaptive_vibrance_skin_memory_mean_protection: skin_memory
                .and_then(|protection| protection.mean_protection_weight),
            adaptive_vibrance_preferred_memory_color_guard_enabled: preferred_memory
                .and_then(|guard| guard.enabled),
            adaptive_vibrance_preferred_memory_color_guard_space: preferred_memory
                .and_then(|guard| guard.working_space.clone()),
            adaptive_vibrance_preferred_memory_color_guard_reference: preferred_memory
                .and_then(|guard| guard.reference.clone()),
            adaptive_vibrance_preferred_memory_color_matched_ratio: preferred_memory
                .and_then(|guard| guard.matched_pixel_ratio),
            adaptive_vibrance_preferred_memory_color_limited_ratio: preferred_memory
                .and_then(|guard| guard.limited_pixel_ratio),
            adaptive_vibrance_preferred_memory_color_mean_scale_reduction: preferred_memory
                .and_then(|guard| guard.mean_scale_reduction),
            preferred_skin_rendering_enabled: preferred_skin
                .and_then(|rendering| rendering.enabled),
            preferred_skin_rendering_space: preferred_skin
                .and_then(|rendering| rendering.working_space.clone()),
            preferred_skin_rendering_preference_reference: preferred_skin
                .and_then(|rendering| rendering.preference_reference.clone()),
            preferred_skin_rendering_adjusted_ratio: preferred_skin
                .and_then(|rendering| rendering.adjusted_pixel_ratio),
            preferred_skin_rendering_mean_delta_e_ab: preferred_skin
                .and_then(|rendering| rendering.mean_delta_e_ab),
        },
        grain: TrackedGrainBaseline {
            luma_residual_p95: grain.and_then(|grain| grain.luma_residual_p95),
            chroma_residual_p95: grain.and_then(|grain| grain.chroma_residual_p95),
            chroma_to_luma_p95_ratio: grain.and_then(|grain| grain.chroma_to_luma_p95_ratio),
            detail_review_required: grain_detail.and_then(|detail| detail.review_required),
            detail_decision_supported: grain_detail.and_then(|detail| detail.decision_supported),
            luminance_p10_retention: grain_detail.and_then(|detail| detail.luminance_p10_retention),
            chroma_p10_retention: grain_detail.and_then(|detail| detail.chroma_p10_retention),
        },
    }
}

fn candidate_acceptance_signatures(candidates: &[ColorCandidateAcceptanceSummary]) -> Vec<String> {
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

pub fn comparison_issues(comparison: &RenderComparisonSummary) -> Vec<String> {
    let mut issues = Vec::new();
    if comparison.output_dimensions_match == Some(false) {
        issues.push("output_dimensions_mismatch".to_string());
    }
    if comparison.output_dimensions_match.is_none() {
        issues.push("output_dimensions_unknown".to_string());
    }
    if comparison.stitch_decision_changed {
        issues.push("stitch_decision_changed".to_string());
    }
    if comparison.base_estimate_source_changed {
        issues.push("base_estimate_source_changed".to_string());
    }
    if comparison.render_input_source_changed {
        issues.push("render_input_source_changed".to_string());
    }
    if comparison.colorspace_mapping_strategy_changed {
        issues.push("colorspace_mapping_strategy_changed".to_string());
    }
    if comparison.calibration_status_changed {
        issues.push("calibration_status_changed".to_string());
    }
    if comparison.calibration_source_changed {
        issues.push("calibration_source_changed".to_string());
    }
    if comparison.colorspace_candidate_risk_changed {
        issues.push("colorspace_candidate_risk_changed".to_string());
    }
    if comparison.colorspace_tone_color_trust_state_changed {
        issues.push("colorspace_tone_color_trust_state_changed".to_string());
    }
    if comparison.render_review_status_changed {
        issues.push("render_review_status_changed".to_string());
    }
    if comparison.render_reviewable_changed {
        issues.push("render_reviewable_changed".to_string());
    }
    if comparison.tone_output_confidence_status_changed {
        issues.push("tone_output_confidence_status_changed".to_string());
    }
    if comparison.tone_output_review_required_changed {
        issues.push("tone_output_review_required_changed".to_string());
    }
    if comparison
        .tone_output_evidence_confidence_delta
        .is_some_and(|delta| delta < -1e-12)
    {
        issues.push("tone_output_evidence_confidence_regressed".to_string());
    }
    if comparison
        .tone_output_render_luminance_range_p05_p95_delta
        .is_some_and(|delta| delta < -0.02)
    {
        issues.push("tone_output_render_luminance_range_regressed".to_string());
    }
    if comparison
        .tone_output_render_to_mapped_luminance_range_ratio_delta
        .is_some_and(|delta| delta < -0.05)
    {
        issues.push("tone_output_range_retention_regressed".to_string());
    }
    if comparison
        .tone_output_maximum_post_tone_high_clip_ratio_delta
        .is_some_and(|delta| delta > 0.01)
    {
        issues.push("tone_output_high_clipping_regressed".to_string());
    }
    if comparison
        .tone_output_maximum_post_tone_low_clip_ratio_delta
        .is_some_and(|delta| delta > 0.01)
    {
        issues.push("tone_output_low_clipping_regressed".to_string());
    }
    if comparison
        .colorspace_selected_quality_score_delta
        .is_some_and(|delta| delta.abs() > 0.25)
    {
        issues.push("colorspace_quality_score_changed_materially".to_string());
    }
    if comparison
        .colorspace_density_monotonicity_score_delta
        .is_some_and(|delta| delta < -0.05)
    {
        issues.push("colorspace_density_monotonicity_regressed".to_string());
    }
    if comparison
        .colorspace_hue_linearity_score_delta
        .is_some_and(|delta| delta < -0.05)
    {
        issues.push("colorspace_hue_linearity_regressed".to_string());
    }
    if comparison
        .colorspace_saturation_preservation_median_ratio_delta
        .is_some_and(|delta| delta.abs() > 0.25)
        || comparison
            .current
            .colorspace_saturation_preservation_median_ratio
            .is_some_and(|ratio| !(0.45..=2.40).contains(&ratio))
    {
        issues.push("colorspace_saturation_preservation_changed_materially".to_string());
    }
    if comparison
        .colorspace_spatial_neutral_delta_p95_delta
        .is_some_and(|delta| delta > 0.03)
    {
        issues.push("colorspace_spatial_neutral_cast_regressed".to_string());
    }
    if comparison
        .colorspace_memory_color_penalty_delta
        .is_some_and(|delta| delta > 0.05)
    {
        issues.push("colorspace_memory_color_penalty_regressed".to_string());
    }
    if comparison
        .colorspace_spatial_consistency_penalty_delta
        .is_some_and(|delta| delta > 0.05)
    {
        issues.push("colorspace_spatial_consistency_penalty_regressed".to_string());
    }
    if comparison
        .colorspace_reference_patch_rms_delta_e_delta
        .is_some_and(|delta| delta > 1.0)
    {
        issues.push("colorspace_reference_delta_e_regressed".to_string());
    }
    if comparison
        .colorspace_reference_patch_rms_delta_e2000_delta
        .is_some_and(|delta| delta > 1.0)
    {
        issues.push("colorspace_reference_delta_e2000_regressed".to_string());
    }
    if comparison
        .colorspace_post_scale_preserved_ratio_delta
        .is_some_and(|delta| delta.abs() > 0.02)
    {
        issues.push("colorspace_gamut_preservation_changed_materially".to_string());
    }
    if comparison.current.stale_render_artifact_count.unwrap_or(0) > 0 {
        issues.push("current_output_has_stale_render_artifacts".to_string());
    }
    if comparison.current.output_file_icc_profile_matches_report == Some(false) {
        issues.push("current_output_icc_profile_mismatch".to_string());
    }
    if comparison.baseline.stale_render_artifact_count.unwrap_or(0) > 0 {
        issues.push("baseline_output_has_stale_render_artifacts".to_string());
    }
    if ratio_outside(comparison.luma_residual_p95_ratio, 0.90, 1.10) {
        issues.push("luma_residual_p95_changed_materially".to_string());
    }
    if ratio_outside(comparison.chroma_residual_p95_ratio, 0.90, 1.10) {
        issues.push("chroma_residual_p95_changed_materially".to_string());
    }
    if comparison
        .highlight_chroma_compressed_ratio_delta
        .is_some_and(|delta| delta.abs() > 0.01)
    {
        issues.push("highlight_chroma_compression_changed_materially".to_string());
    }
    if comparison
        .highlight_neutral_chroma_compressed_ratio_delta
        .is_some_and(|delta| delta.abs() > 0.01)
    {
        issues.push("highlight_neutral_chroma_compression_changed_materially".to_string());
    }
    if comparison
        .shadow_chroma_compressed_ratio_delta
        .is_some_and(|delta| delta.abs() > 0.01)
    {
        issues.push("shadow_chroma_compression_changed_materially".to_string());
    }
    issues
}

fn summary_baseline_issues(
    comparison: &SummaryBaselineComparison,
    baseline: &TrackedValidationBaseline,
) -> Vec<String> {
    let mut issues = Vec::new();
    let tolerances = comparison.tolerances;
    if comparison.fixture_matches == Some(false) {
        issues.push("summary_baseline_fixture_mismatch".to_string());
    }
    let dimensions_expected =
        baseline.render.output_width.is_some() || baseline.render.output_height.is_some();
    if dimensions_expected && comparison.output_dimensions_match == Some(false) {
        issues.push("output_dimensions_mismatch".to_string());
    }
    if dimensions_expected && comparison.output_dimensions_match.is_none() {
        issues.push("output_dimensions_unknown".to_string());
    }
    if comparison.output_color_space_changed {
        issues.push("output_color_space_changed".to_string());
    }
    if comparison.output_file_icc_profile_matches_report == Some(false) {
        issues.push("current_output_icc_profile_mismatch".to_string());
    }
    if comparison.stitch_decision_changed {
        issues.push("stitch_decision_changed".to_string());
    }
    if comparison
        .stitch_confidence_delta
        .is_some_and(|delta| delta.abs() > tolerances.confidence_abs)
    {
        issues.push("stitch_confidence_changed_materially".to_string());
    }
    if comparison.chosen_hypothesis_changed {
        issues.push("stitch_chosen_hypothesis_changed".to_string());
    }
    if comparison.seam_exposure_correction_changed {
        issues.push("seam_exposure_correction_changed".to_string());
    }
    if comparison.seam_exposure_model_changed {
        issues.push("seam_exposure_model_changed".to_string());
    }
    if comparison.seam_exposure_spatial_2d_gain_acceptance_changed {
        issues.push("seam_exposure_spatial_2d_gain_acceptance_changed".to_string());
    }
    if comparison.seam_exposure_spatial_2d_gain_offset_acceptance_changed {
        issues.push("seam_exposure_spatial_2d_gain_offset_acceptance_changed".to_string());
    }
    if comparison.seam_exposure_spatial_quadratic_gain_acceptance_changed {
        issues.push("seam_exposure_spatial_quadratic_gain_acceptance_changed".to_string());
    }
    if comparison.seam_exposure_spatial_quadratic_gain_offset_acceptance_changed {
        issues.push("seam_exposure_spatial_quadratic_gain_offset_acceptance_changed".to_string());
    }
    if comparison.seam_blend_mode_changed {
        issues.push("seam_blend_mode_changed".to_string());
    }
    if comparison.seam_blend_applied_changed {
        issues.push("seam_blend_applied_changed".to_string());
    }
    if comparison.seam_blend_review_required_changed {
        issues.push("seam_blend_review_required_changed".to_string());
    }
    if comparison.seam_detail_review_required_changed {
        issues.push("seam_detail_review_required_changed".to_string());
    }
    if comparison
        .seam_detail_max_symmetric_energy_ratio_delta
        .is_some_and(|delta| delta.abs() > tolerances.tone_ratio_abs)
    {
        issues.push("seam_detail_energy_ratio_changed_materially".to_string());
    }
    if comparison
        .seam_gradient_ratio_delta
        .is_some_and(|delta| delta.abs() > tolerances.tone_ratio_abs)
    {
        issues.push("seam_gradient_ratio_changed_materially".to_string());
    }
    if comparison
        .seam_overlap_p95_abs_difference_delta
        .is_some_and(|delta| delta.abs() > tolerances.tone_ratio_abs)
    {
        issues.push("seam_overlap_p95_abs_difference_changed_materially".to_string());
    }
    if comparison.base_estimate_source_changed {
        issues.push("base_estimate_source_changed".to_string());
    }
    if comparison
        .raw_base_proxy_confidence_delta
        .is_some_and(|delta| delta < -tolerances.confidence_abs)
    {
        issues.push("raw_base_proxy_confidence_regressed".to_string());
    }
    if comparison
        .raw_base_support_fraction_delta
        .is_some_and(|delta| {
            let support_tolerance = baseline
                .render
                .raw_base_support_fraction
                .map(|support| support * 0.25)
                .unwrap_or(0.0)
                .max(tolerances.base_support_fraction_abs);
            delta < -support_tolerance
        })
    {
        issues.push("raw_base_support_fraction_regressed".to_string());
    }
    if comparison.render_input_source_changed {
        issues.push("render_input_source_changed".to_string());
    }
    if comparison.colorspace_mapping_strategy_changed {
        issues.push("colorspace_mapping_strategy_changed".to_string());
    }
    if comparison.calibration_status_changed {
        issues.push("calibration_status_changed".to_string());
    }
    if comparison.calibration_source_changed {
        issues.push("calibration_source_changed".to_string());
    }
    if comparison.calibration_scanner_profile_status_changed {
        issues.push("calibration_scanner_profile_status_changed".to_string());
    }
    if comparison.calibration_scanner_profile_id_changed {
        issues.push("calibration_scanner_profile_id_changed".to_string());
    }
    if comparison.calibration_roll_profile_status_changed {
        issues.push("calibration_roll_profile_status_changed".to_string());
    }
    if comparison.calibration_roll_profile_id_changed {
        issues.push("calibration_roll_profile_id_changed".to_string());
    }
    if comparison
        .calibration_confidence_delta
        .is_some_and(|delta| delta < -tolerances.confidence_abs)
    {
        issues.push("calibration_confidence_regressed".to_string());
    }
    if comparison
        .calibration_matrix_condition_number_delta
        .is_some_and(|delta| delta > 1.0)
    {
        issues.push("calibration_matrix_condition_number_regressed".to_string());
    }
    if comparison.calibration_requested_film_stock_changed {
        issues.push("calibration_requested_film_stock_changed".to_string());
    }
    if comparison.calibration_film_stock_status_changed {
        issues.push("calibration_film_stock_status_changed".to_string());
    }
    if comparison.calibration_film_stock_matched_roll_profiles_changed {
        issues.push("calibration_film_stock_matched_roll_profiles_changed".to_string());
    }
    if comparison.calibration_rejection_details_changed {
        issues.push("calibration_rejection_details_changed".to_string());
    }
    if comparison.selected_candidate_changed {
        issues.push("colorspace_selected_candidate_changed".to_string());
    }
    if comparison.selected_candidate_rank_changed {
        issues.push("colorspace_selected_candidate_rank_changed".to_string());
    }
    if comparison.colorspace_candidate_acceptance_changed {
        issues.push("colorspace_candidate_acceptance_changed".to_string());
    }
    if comparison.calibration_acceptance_status_changed {
        issues.push("calibration_acceptance_status_changed".to_string());
    }
    if comparison.calibration_color_mapping_applied_changed {
        issues.push("calibration_color_mapping_applied_changed".to_string());
    }
    if comparison.calibration_acceptance_preferred_candidate_changed {
        issues.push("calibration_acceptance_preferred_candidate_changed".to_string());
    }
    if comparison.calibration_acceptance_beats_image_derived_changed {
        issues.push("calibration_acceptance_beats_image_derived_changed".to_string());
    }
    if comparison.colorspace_candidate_risk_changed {
        issues.push("colorspace_candidate_risk_changed".to_string());
    }
    if comparison.colorspace_tone_color_trust_state_changed {
        issues.push("colorspace_tone_color_trust_state_changed".to_string());
    }
    if comparison
        .colorspace_selected_quality_score_delta
        .is_some_and(|delta| delta.abs() > tolerances.colorspace_quality_abs)
    {
        issues.push("colorspace_quality_score_changed_materially".to_string());
    }
    if comparison
        .colorspace_technical_safety_score_delta
        .is_some_and(|delta| delta > tolerances.colorspace_quality_abs)
    {
        issues.push("colorspace_technical_safety_score_regressed".to_string());
    }
    if comparison
        .colorspace_color_fidelity_score_delta
        .is_some_and(|delta| delta > tolerances.colorspace_quality_abs)
    {
        issues.push("colorspace_color_fidelity_score_regressed".to_string());
    }
    if comparison
        .colorspace_selected_runner_up_quality_delta_delta
        .is_some_and(|delta| delta < -tolerances.colorspace_quality_abs)
    {
        issues.push("colorspace_runner_up_margin_shrank_materially".to_string());
    }
    if comparison
        .colorspace_density_monotonicity_score_delta
        .is_some_and(|delta| delta < -tolerances.colorspace_score_abs)
    {
        issues.push("colorspace_density_monotonicity_regressed".to_string());
    }
    if comparison
        .colorspace_hue_linearity_score_delta
        .is_some_and(|delta| delta < -tolerances.colorspace_score_abs)
    {
        issues.push("colorspace_hue_linearity_regressed".to_string());
    }
    if comparison
        .colorspace_saturation_preservation_median_ratio_delta
        .is_some_and(|delta| delta.abs() > tolerances.color_ratio_abs)
    {
        issues.push("colorspace_saturation_preservation_changed_materially".to_string());
    }
    if comparison
        .colorspace_spatial_neutral_delta_p95_delta
        .is_some_and(|delta| delta > tolerances.spatial_neutral_abs)
    {
        issues.push("colorspace_spatial_neutral_cast_regressed".to_string());
    }
    if comparison
        .colorspace_memory_color_penalty_delta
        .is_some_and(|delta| delta > tolerances.colorspace_score_abs)
    {
        issues.push("colorspace_memory_color_penalty_regressed".to_string());
    }
    if comparison
        .colorspace_spatial_consistency_penalty_delta
        .is_some_and(|delta| delta > tolerances.colorspace_score_abs)
    {
        issues.push("colorspace_spatial_consistency_penalty_regressed".to_string());
    }
    if comparison
        .colorspace_post_scale_preserved_ratio_delta
        .is_some_and(|delta| delta.abs() > tolerances.colorspace_preserved_abs)
    {
        issues.push("colorspace_gamut_preservation_changed_materially".to_string());
    }
    if comparison
        .colorspace_reference_patch_selected_rms_delta_e_delta
        .is_some_and(|delta| delta > tolerances.reference_delta_e_abs)
    {
        issues.push("colorspace_reference_delta_e_regressed".to_string());
    }
    if comparison
        .colorspace_reference_patch_selected_rms_delta_e2000_delta
        .is_some_and(|delta| delta > tolerances.reference_delta_e_abs)
    {
        issues.push("colorspace_reference_delta_e2000_regressed".to_string());
    }
    if comparison
        .colorspace_reference_patch_max_error_delta_vs_image_derived_delta
        .is_some_and(|delta| delta > tolerances.reference_xyz_abs)
    {
        issues.push("colorspace_reference_xyz_max_vs_image_derived_regressed".to_string());
    }
    if comparison
        .colorspace_reference_patch_delta_e_rms_delta_vs_image_derived_delta
        .is_some_and(|delta| delta > tolerances.reference_delta_e_abs)
    {
        issues.push("colorspace_reference_delta_e_vs_image_derived_regressed".to_string());
    }
    if comparison
        .colorspace_reference_patch_delta_e_max_delta_vs_image_derived_delta
        .is_some_and(|delta| delta > tolerances.reference_delta_e_abs)
    {
        issues.push("colorspace_reference_delta_e_max_vs_image_derived_regressed".to_string());
    }
    if comparison
        .colorspace_reference_patch_delta_e2000_rms_delta_vs_image_derived_delta
        .is_some_and(|delta| delta > tolerances.reference_delta_e_abs)
    {
        issues.push("colorspace_reference_delta_e2000_vs_image_derived_regressed".to_string());
    }
    if comparison
        .colorspace_reference_patch_delta_e2000_max_delta_vs_image_derived_delta
        .is_some_and(|delta| delta > tolerances.reference_delta_e_abs)
    {
        issues.push("colorspace_reference_delta_e2000_max_vs_image_derived_regressed".to_string());
    }
    if comparison.colorspace_reference_patch_selected_regresses_image_derived_changed {
        issues.push("colorspace_reference_selected_regression_changed".to_string());
    }
    if comparison.colorspace_reference_patch_worst_hue_families_changed {
        issues.push("colorspace_reference_worst_hue_families_changed".to_string());
    }
    if comparison.colorspace_reference_patch_hue_family_regressions_changed {
        issues.push("colorspace_reference_hue_family_regressions_changed".to_string());
    }
    if comparison.colorspace_reference_patch_regressed_candidates_changed {
        issues.push("colorspace_reference_regressed_candidates_changed".to_string());
    }
    if comparison
        .colorspace_neutral_estimate_score_delta
        .is_some_and(|delta| delta < -tolerances.colorspace_score_abs)
    {
        issues.push("colorspace_neutral_estimate_score_regressed".to_string());
    }
    if comparison.colorspace_neutral_estimate_accepted_changed {
        issues.push("colorspace_neutral_estimate_acceptance_changed".to_string());
    }
    if comparison.colorspace_neutral_estimate_populated_band_count_changed {
        issues.push("colorspace_neutral_estimate_band_support_changed".to_string());
    }
    if comparison
        .colorspace_neutral_estimate_dominant_band_fraction_delta
        .is_some_and(|delta| delta > tolerances.color_ratio_abs)
    {
        issues.push("colorspace_neutral_estimate_band_dominance_regressed".to_string());
    }
    if comparison
        .colorspace_dominant_anchor_score_delta
        .is_some_and(|delta| delta < -tolerances.colorspace_score_abs)
    {
        issues.push("colorspace_dominant_anchor_score_regressed".to_string());
    }
    if comparison.colorspace_dominant_anchor_accepted_changed {
        issues.push("colorspace_dominant_anchor_acceptance_changed".to_string());
    }
    if comparison.colorspace_dominant_anchor_unstable_channel_count_changed {
        issues.push("colorspace_dominant_anchor_unstable_channel_count_changed".to_string());
    }
    if comparison.colorspace_dominant_anchor_channel_unstable_changed {
        issues.push("colorspace_dominant_anchor_channel_unstable_changed".to_string());
    }
    if comparison.colorspace_channel_anchor_min_count_changed {
        issues.push("colorspace_channel_anchor_min_count_changed".to_string());
    }
    if comparison.colorspace_channel_anchor_low_support_changed {
        issues.push("colorspace_channel_anchor_low_support_changed".to_string());
    }
    if comparison.colorspace_weak_anchor_fallback_changed {
        issues.push("colorspace_weak_anchor_fallback_changed".to_string());
    }
    if comparison.colorspace_gamut_fallback_changed {
        issues.push("colorspace_gamut_fallback_changed".to_string());
    }
    if comparison.colorspace_neutral_trim_applied_changed {
        issues.push("colorspace_neutral_trim_applied_changed".to_string());
    }
    issues.extend(comparison.colorspace_debug_artifact_issues.iter().cloned());
    if comparison.render_review_status_changed {
        issues.push("render_review_status_changed".to_string());
    }
    if comparison.render_reviewable_changed {
        issues.push("render_reviewable_changed".to_string());
    }
    if comparison.tone_output_confidence_status_changed {
        issues.push("tone_output_confidence_status_changed".to_string());
    }
    if comparison.tone_output_review_required_changed {
        issues.push("tone_output_review_required_changed".to_string());
    }
    if comparison
        .tone_output_evidence_confidence_delta
        .is_some_and(|delta| delta < -tolerances.confidence_abs)
    {
        issues.push("tone_output_evidence_confidence_regressed".to_string());
    }
    if comparison
        .tone_output_render_luminance_range_p05_p95_delta
        .is_some_and(|delta| delta < -tolerances.tone_ratio_abs)
    {
        issues.push("tone_output_render_luminance_range_regressed".to_string());
    }
    if comparison
        .tone_output_render_to_mapped_luminance_range_ratio_delta
        .is_some_and(|delta| delta < -tolerances.tone_ratio_abs)
    {
        issues.push("tone_output_range_retention_regressed".to_string());
    }
    if comparison
        .tone_output_maximum_post_tone_high_clip_ratio_delta
        .is_some_and(|delta| delta > tolerances.tone_ratio_abs)
    {
        issues.push("tone_output_high_clipping_regressed".to_string());
    }
    if comparison
        .tone_output_maximum_post_tone_low_clip_ratio_delta
        .is_some_and(|delta| delta > tolerances.tone_ratio_abs)
    {
        issues.push("tone_output_low_clipping_regressed".to_string());
    }
    if comparison
        .highlight_chroma_compressed_ratio_delta
        .is_some_and(|delta| delta.abs() > tolerances.tone_ratio_abs)
    {
        issues.push("highlight_chroma_compression_changed_materially".to_string());
    }
    if comparison
        .highlight_neutral_chroma_compressed_ratio_delta
        .is_some_and(|delta| delta.abs() > tolerances.tone_ratio_abs)
    {
        issues.push("highlight_neutral_chroma_compression_changed_materially".to_string());
    }
    if comparison
        .shadow_chroma_compressed_ratio_delta
        .is_some_and(|delta| delta.abs() > tolerances.tone_ratio_abs)
    {
        issues.push("shadow_chroma_compression_changed_materially".to_string());
    }
    if comparison.adaptive_vibrance_skin_memory_protection_enabled_changed {
        issues.push("adaptive_vibrance_skin_memory_protection_enabled_changed".to_string());
    }
    if comparison.adaptive_vibrance_skin_memory_protection_space_changed {
        issues.push("adaptive_vibrance_skin_memory_protection_space_changed".to_string());
    }
    if comparison
        .adaptive_vibrance_skin_memory_protected_ratio_delta
        .is_some_and(|delta| delta.abs() > tolerances.tone_ratio_abs)
    {
        issues.push("adaptive_vibrance_skin_memory_protected_ratio_changed_materially".to_string());
    }
    if comparison
        .adaptive_vibrance_skin_memory_mean_protection_delta
        .is_some_and(|delta| delta.abs() > tolerances.tone_ratio_abs)
    {
        issues.push("adaptive_vibrance_skin_memory_mean_protection_changed_materially".to_string());
    }
    if comparison.adaptive_vibrance_preferred_memory_color_guard_enabled_changed {
        issues.push("adaptive_vibrance_preferred_memory_color_guard_enabled_changed".to_string());
    }
    if comparison.adaptive_vibrance_preferred_memory_color_guard_space_changed {
        issues.push("adaptive_vibrance_preferred_memory_color_guard_space_changed".to_string());
    }
    if comparison.adaptive_vibrance_preferred_memory_color_guard_reference_changed {
        issues.push("adaptive_vibrance_preferred_memory_color_guard_reference_changed".to_string());
    }
    if comparison
        .adaptive_vibrance_preferred_memory_color_matched_ratio_delta
        .is_some_and(|delta| delta.abs() > tolerances.tone_ratio_abs)
    {
        issues.push(
            "adaptive_vibrance_preferred_memory_color_matched_ratio_changed_materially".to_string(),
        );
    }
    if comparison
        .adaptive_vibrance_preferred_memory_color_limited_ratio_delta
        .is_some_and(|delta| delta.abs() > tolerances.tone_ratio_abs)
    {
        issues.push(
            "adaptive_vibrance_preferred_memory_color_limited_ratio_changed_materially".to_string(),
        );
    }
    if comparison
        .adaptive_vibrance_preferred_memory_color_mean_scale_reduction_delta
        .is_some_and(|delta| delta.abs() > tolerances.tone_ratio_abs)
    {
        issues.push(
            "adaptive_vibrance_preferred_memory_color_mean_scale_reduction_changed_materially"
                .to_string(),
        );
    }
    if comparison.preferred_skin_rendering_enabled_changed {
        issues.push("preferred_skin_rendering_enabled_changed".to_string());
    }
    if comparison.preferred_skin_rendering_space_changed {
        issues.push("preferred_skin_rendering_space_changed".to_string());
    }
    if comparison.preferred_skin_rendering_preference_reference_changed {
        issues.push("preferred_skin_rendering_preference_reference_changed".to_string());
    }
    if comparison
        .preferred_skin_rendering_adjusted_ratio_delta
        .is_some_and(|delta| delta.abs() > tolerances.tone_ratio_abs)
    {
        issues.push("preferred_skin_rendering_adjusted_ratio_changed_materially".to_string());
    }
    if comparison
        .preferred_skin_rendering_mean_delta_e_ab_delta
        .is_some_and(|delta| delta.abs() > tolerances.reference_delta_e_abs)
    {
        issues.push("preferred_skin_rendering_mean_delta_e_ab_changed_materially".to_string());
    }
    let grain_low = 1.0 - tolerances.grain_ratio_abs;
    let grain_high = 1.0 + tolerances.grain_ratio_abs;
    if ratio_outside(comparison.luma_residual_p95_ratio, grain_low, grain_high) {
        issues.push("luma_residual_p95_changed_materially".to_string());
    }
    if ratio_outside(comparison.chroma_residual_p95_ratio, grain_low, grain_high) {
        issues.push("chroma_residual_p95_changed_materially".to_string());
    }
    if comparison
        .chroma_to_luma_p95_ratio_delta
        .is_some_and(|delta| delta.abs() > tolerances.grain_ratio_abs)
    {
        issues.push("chroma_to_luma_p95_ratio_changed_materially".to_string());
    }
    if comparison.grain_detail_review_required_changed {
        issues.push("grain_detail_review_required_changed".to_string());
    }
    if comparison.grain_detail_decision_supported_changed {
        issues.push("grain_detail_decision_supported_changed".to_string());
    }
    if comparison
        .grain_detail_luminance_p10_retention_delta
        .is_some_and(|delta| delta < -tolerances.grain_ratio_abs)
    {
        issues.push("grain_detail_luminance_retention_regressed".to_string());
    }
    if comparison
        .grain_detail_chroma_p10_retention_delta
        .is_some_and(|delta| delta < -tolerances.grain_ratio_abs)
    {
        issues.push("grain_detail_chroma_retention_regressed".to_string());
    }
    issues
}

fn summarize_report_identity(
    report: &PipelineReport,
    source_report_path: Option<&Path>,
) -> ReportIdentitySummary {
    let metadata = report.metadata.as_ref();
    ReportIdentitySummary {
        source_report_path: source_report_path.map(path_to_string),
        generated_at: metadata.map(|metadata| metadata.generated_at.clone()),
        generated_at_unix_ms: metadata.map(|metadata| metadata.generated_at_unix_ms),
        package_version: metadata.map(|metadata| metadata.package_version.clone()),
        binary_name: metadata.map(|metadata| metadata.binary_name.clone()),
        binary_path: metadata.and_then(|metadata| metadata.binary_path.clone()),
        binary_sha256: metadata.and_then(|metadata| metadata.binary_sha256.clone()),
        binary_file_size_bytes: metadata.and_then(|metadata| metadata.binary_file_size_bytes),
        binary_identity_status: metadata
            .and_then(|metadata| metadata.binary_identity_status.clone()),
        binary_identity_error: metadata.and_then(|metadata| metadata.binary_identity_error.clone()),
        working_directory: metadata.map(|metadata| metadata.working_directory.clone()),
        cli_args: metadata.map(|metadata| metadata.cli_args.clone()),
        output_path: metadata.map(|metadata| metadata.output_path.clone()),
        report_schema_version: metadata.map(|metadata| metadata.report_schema_version),
        pipeline_schema_version: metadata.map(|metadata| metadata.pipeline_schema_version),
    }
}

fn summarize_render(
    report: &PipelineReport,
    identity: &ReportIdentitySummary,
    stitch: &StitchValidationSummary,
    base_density: &BaseDensityValidationSummary,
    colorspace: &ColorspaceValidationSummary,
    tone: &ToneValidationSummary,
) -> RenderDiagnosticSummary {
    let working_phase = phase(report, "working_image_select");
    let save_phase = phase(report, "save");
    let stitch_phase = phase(report, "stitch");
    let positive_input_inspection =
        working_phase.and_then(|phase| phase.metrics.get("positive_input_inspection"));
    let output_shape = save_phase.and_then(|phase| usize_vec_metric(phase, "output_shape"));
    let output_height = save_phase
        .and_then(|phase| usize_metric(phase, "output_height"))
        .or_else(|| {
            output_shape
                .as_ref()
                .and_then(|shape| shape.first().copied())
        });
    let output_width = save_phase
        .and_then(|phase| usize_metric(phase, "output_width"))
        .or_else(|| {
            output_shape
                .as_ref()
                .and_then(|shape| shape.get(1).copied())
        });
    let selected_color_candidate = colorspace
        .candidate_quality_scores
        .iter()
        .find(|candidate| candidate.selected == Some(true));
    let output_path = save_phase
        .and_then(|phase| string_metric(phase, "output_path"))
        .or_else(|| identity.output_path.clone());
    let artifact_sha256_binding_required = identity
        .report_schema_version
        .is_some_and(|version| version >= 4);
    let output_sha256 = save_phase.and_then(|phase| string_metric(phase, "output_sha256"));
    let output_file_sha256 = output_path.as_deref().and_then(|path| {
        (artifact_sha256_binding_required || output_sha256.is_some())
            .then(|| inspect_file_sha256(&resolve_report_relative_path(identity, path)))
            .flatten()
    });
    let output_file_sha256_matches_report = artifact_sha256_match(
        artifact_sha256_binding_required,
        output_sha256.as_deref(),
        output_file_sha256.as_deref(),
    );
    let output_icc_profile_embedded = save_phase
        .and_then(|phase| phase.metrics.get("output_icc_profile"))
        .and_then(|profile| profile.get("embedded"))
        .and_then(serde_json::Value::as_bool);
    let output_icc_profile_description = save_phase
        .and_then(|phase| phase.metrics.get("output_icc_profile"))
        .and_then(|profile| string_value(profile.get("description")));
    let output_file_icc = inspect_render_output_icc_profile(identity, output_path.as_deref());
    let output_file_icc_profile_matches_report = output_file_icc.as_ref().and_then(|inspection| {
        profile_inspection_matches_report(
            output_icc_profile_embedded,
            output_icc_profile_description.as_deref(),
            inspection,
        )
    });
    let output_file_dimensions_match_report = output_file_icc
        .as_ref()
        .map(|inspection| inspection.width == output_width && inspection.height == output_height);
    let output_encoding_claim_is_supported = save_phase
        .and_then(|phase| phase.metrics.get("output_encoding"))
        .is_some_and(|encoding| {
            string_value(encoding.get("container")).as_deref() == Some("tiff")
                && string_value(encoding.get("channels")).as_deref() == Some("RGB")
                && string_value(encoding.get("sample_format")).as_deref() == Some("UINT")
                && encoding
                    .get("bits_per_sample")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|values| {
                        values.len() == 3 && values.iter().all(|value| value.as_u64() == Some(16))
                    })
        });
    let output_file_storage_matches_report = output_file_icc.as_ref().map(|inspection| {
        output_encoding_claim_is_supported
            && inspection.color_type.as_deref() == Some("RGB(16)")
            && inspection.sample_format.as_deref().is_some_and(|formats| {
                !formats.is_empty() && formats.iter().all(|format| *format == 1)
            })
    });
    let quality_mode = save_phase.and_then(|phase| string_metric(phase, "quality_mode"));
    let master_scene_referred_path =
        save_phase.and_then(|phase| string_metric(phase, "master_scene_referred_path"));
    let master_scene_referred_sha256 =
        save_phase.and_then(|phase| string_metric(phase, "master_scene_referred_sha256"));
    let master_scene_referred_requested = save_phase
        .and_then(|phase| bool_metric(phase, "master_scene_referred_requested"))
        .or_else(|| {
            quality_mode
                .as_deref()
                .map(|quality| quality == "perfect" || master_scene_referred_path.is_some())
        });
    let master_scene_referred_file = master_scene_referred_path
        .as_deref()
        .map(|path| inspect_tiff_artifact(&resolve_report_relative_path(identity, path)));
    let master_scene_referred_file_sha256 =
        master_scene_referred_path.as_deref().and_then(|path| {
            (artifact_sha256_binding_required || master_scene_referred_sha256.is_some())
                .then(|| inspect_file_sha256(&resolve_report_relative_path(identity, path)))
                .flatten()
        });
    let master_scene_referred_file_sha256_matches_report = artifact_sha256_match(
        artifact_sha256_binding_required && master_scene_referred_requested == Some(true),
        master_scene_referred_sha256.as_deref(),
        master_scene_referred_file_sha256.as_deref(),
    );
    let master_report_claim_is_supported = save_phase.is_some_and(|phase| {
        string_metric(phase, "master_scene_referred_color_space").as_deref()
            == Some("scene_referred_linear_prophoto_rgb_d50")
            && phase
                .metrics
                .get("master_scene_referred_icc_profile")
                .and_then(|profile| profile.get("embedded"))
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            && phase
                .metrics
                .get("master_scene_referred_icc_profile")
                .and_then(|profile| string_value(profile.get("description")))
                .as_deref()
                == Some(tiff_io::PROPHOTO_LINEAR_ICC_DESCRIPTION)
            && phase
                .metrics
                .get("master_scene_referred_diagnostics")
                .and_then(|diagnostics| string_value(diagnostics.get("sample_format")))
                .as_deref()
                == Some("IEEEFP")
    });
    let master_scene_referred_file_matches_report = match (
        master_scene_referred_requested,
        master_scene_referred_path.as_deref(),
    ) {
        (Some(true), None) => Some(false),
        (_, Some(_)) => master_scene_referred_file.as_ref().map(|inspection| {
            let dimensions_match = output_file_icc.as_ref().is_some_and(|output| {
                inspection.width == output.width && inspection.height == output.height
            });
            master_report_claim_is_supported
                && dimensions_match
                && inspection.color_type.as_deref() == Some("RGB(32)")
                && inspection.sample_format.as_deref().is_some_and(|formats| {
                    !formats.is_empty() && formats.iter().all(|format| *format == 3)
                })
                && inspection.image_description.as_deref()
                    == Some(tiff_io::PROPHOTO_SCENE_REFERRED_FLOAT_DESCRIPTION)
                && inspection.embedded
                && inspection.valid
                && inspection.description.as_deref()
                    == Some(tiff_io::PROPHOTO_LINEAR_ICC_DESCRIPTION)
        }),
        _ => None,
    };
    let master_scene_referred_file_status =
        if master_scene_referred_requested == Some(true) && master_scene_referred_path.is_none() {
            Some("missing_path".to_string())
        } else {
            master_scene_referred_file.as_ref().map(|inspection| {
                if inspection.status == "inspection_failed" {
                    "inspection_failed"
                } else if master_scene_referred_file_matches_report == Some(true) {
                    "valid_scene_referred_master"
                } else {
                    "invalid_scene_referred_master"
                }
                .to_string()
            })
        };
    let review_srgb_path = save_phase.and_then(|phase| string_metric(phase, "review_srgb_path"));
    let review_srgb_sha256 =
        save_phase.and_then(|phase| string_metric(phase, "review_srgb_sha256"));
    let review_srgb_requested = save_phase
        .and_then(|phase| bool_metric(phase, "review_srgb_requested"))
        .or_else(|| quality_mode.as_deref().map(|quality| quality == "perfect"));
    let review_srgb_file = review_srgb_path
        .as_deref()
        .map(|path| inspect_review_srgb_artifact(&resolve_report_relative_path(identity, path)));
    let review_srgb_file_sha256 = review_srgb_path.as_deref().and_then(|path| {
        (artifact_sha256_binding_required || review_srgb_sha256.is_some())
            .then(|| inspect_file_sha256(&resolve_report_relative_path(identity, path)))
            .flatten()
    });
    let review_srgb_file_sha256_matches_report = artifact_sha256_match(
        artifact_sha256_binding_required && review_srgb_requested == Some(true),
        review_srgb_sha256.as_deref(),
        review_srgb_file_sha256.as_deref(),
    );
    let review_srgb_gamut_mapping = save_phase
        .and_then(|phase| phase.metrics.get("review_srgb_gamut_mapping"))
        .filter(|mapping| mapping.is_object());
    let review_srgb_gamut_mapping_space =
        review_srgb_gamut_mapping.and_then(|mapping| string_value(mapping.get("space")));
    let review_srgb_gamut_mapped_ratio =
        review_srgb_gamut_mapping.and_then(|mapping| f64_value(mapping.get("gamut_mapped_ratio")));
    let review_srgb_gamut_mapping_mean_chroma_scale = review_srgb_gamut_mapping
        .and_then(|mapping| f64_value(mapping.get("mapped_mean_chroma_scale")));
    let review_srgb_gamut_mapping_min_chroma_scale = review_srgb_gamut_mapping
        .and_then(|mapping| f64_value(mapping.get("mapped_min_chroma_scale")));
    let review_srgb_post_map_out_of_gamut_pixel_count = review_srgb_gamut_mapping
        .and_then(|mapping| usize_value(mapping.get("post_map_out_of_gamut_pixel_count")));
    let review_srgb_gamut_mapping_supported =
        match review_srgb_requested {
            Some(true) => Some(review_srgb_gamut_mapping.is_some_and(|mapping| {
                let width = usize_value(mapping.get("width"));
                let height = usize_value(mapping.get("height"));
                let pixel_count = usize_value(mapping.get("pixel_count"));
                let nonfinite = usize_value(mapping.get("nonfinite_input_pixel_count"));
                let mapped_count = usize_value(mapping.get("gamut_mapped_pixel_count"));
                let mapped_ratio = f64_value(mapping.get("gamut_mapped_ratio"));
                let mean_scale = f64_value(mapping.get("mapped_mean_chroma_scale"));
                let min_scale = f64_value(mapping.get("mapped_min_chroma_scale"));
                let dimensions_match = width
                    .zip(height)
                    .zip(
                        output_file_icc
                            .as_ref()
                            .and_then(|output| output.width.zip(output.height)),
                    )
                    .is_some_and(|((width, height), (output_width, output_height))| {
                        width == output_width && height == output_height
                    });
                let counts_are_coherent = width.zip(height).zip(pixel_count).is_some_and(
                    |((width, height), pixel_count)| width.checked_mul(height) == Some(pixel_count),
                ) && mapped_count
                    .zip(pixel_count)
                    .is_some_and(|(mapped, total)| mapped <= total)
                    && mapped_count.zip(pixel_count).zip(mapped_ratio).is_some_and(
                        |((mapped, total), ratio)| {
                            total > 0 && (ratio - mapped as f64 / total as f64).abs() <= 1e-12
                        },
                    );
                review_srgb_gamut_mapping_space.as_deref()
                    == Some(tiff_io::SRGB_REVIEW_GAMUT_MAPPING_SPACE)
                    && dimensions_match
                    && pixel_count.is_some_and(|count| count > 0)
                    && counts_are_coherent
                    && nonfinite == Some(0)
                    && mapped_ratio
                        .is_some_and(|ratio| ratio.is_finite() && (0.0..=1.0).contains(&ratio))
                    && mean_scale
                        .is_some_and(|scale| scale.is_finite() && (0.0..=1.0).contains(&scale))
                    && min_scale
                        .is_some_and(|scale| scale.is_finite() && (0.0..=1.0).contains(&scale))
                    && min_scale
                        .zip(mean_scale)
                        .is_some_and(|(min, mean)| min <= mean + 1e-12)
                    && review_srgb_post_map_out_of_gamut_pixel_count == Some(0)
            })),
            Some(false) if review_srgb_gamut_mapping.is_some() => Some(false),
            _ => None,
        };
    let review_report_claim_is_supported = save_phase.is_some_and(|phase| {
        string_metric(phase, "review_srgb_color_space").as_deref() == Some("srgb_display_png")
            && phase
                .metrics
                .get("review_srgb_encoding")
                .is_some_and(|encoding| {
                    string_value(encoding.get("container")).as_deref() == Some("png")
                        && string_value(encoding.get("channels")).as_deref() == Some("RGB")
                        && string_value(encoding.get("sample_format")).as_deref() == Some("UINT")
                        && encoding
                            .get("bits_per_sample")
                            .and_then(serde_json::Value::as_array)
                            .is_some_and(|values| {
                                values.len() == 3
                                    && values.iter().all(|value| value.as_u64() == Some(8))
                            })
                })
            && phase
                .metrics
                .get("review_srgb_icc_profile")
                .and_then(|profile| profile.get("embedded"))
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            && phase
                .metrics
                .get("review_srgb_icc_profile")
                .and_then(|profile| string_value(profile.get("description")))
                .as_deref()
                == Some(tiff_io::SRGB_ICC_DESCRIPTION)
    });
    let review_srgb_file_matches_report = match (review_srgb_requested, review_srgb_path.as_deref())
    {
        (Some(true), None) => Some(false),
        (_, Some(_)) => review_srgb_file.as_ref().map(|inspection| {
            let dimensions_match = output_file_icc.as_ref().is_some_and(|output| {
                inspection.width == output.width && inspection.height == output.height
            });
            review_report_claim_is_supported
                && dimensions_match
                && inspection.status == "valid_srgb_review_png"
                && inspection.color_type.as_deref() == Some("Rgb8")
                && inspection.icc_profile_valid
                && inspection.icc_profile_description.as_deref()
                    == Some(tiff_io::SRGB_ICC_DESCRIPTION)
                && inspection.icc_profile_matches_standard_srgb
        }),
        _ => None,
    };
    let review_srgb_file_status =
        if review_srgb_requested == Some(true) && review_srgb_path.is_none() {
            Some("missing_path".to_string())
        } else {
            review_srgb_file
                .as_ref()
                .map(|inspection| inspection.status.clone())
        };

    RenderDiagnosticSummary {
        source_report_path: identity.source_report_path.clone(),
        source_report_generated_at: identity.generated_at.clone(),
        artifact_sha256_binding_required,
        output_path,
        output_modified_at: save_phase.and_then(|phase| string_metric(phase, "output_modified_at")),
        output_file_size_bytes: save_phase
            .and_then(|phase| u64_metric(phase, "output_file_size_bytes")),
        output_sha256,
        output_file_sha256,
        output_file_sha256_matches_report,
        output_color_space: save_phase.and_then(|phase| string_metric(phase, "output_color_space")),
        output_icc_profile_embedded,
        output_icc_profile_description,
        output_file_icc_profile_status: output_file_icc
            .as_ref()
            .map(|inspection| inspection.status.clone()),
        output_file_icc_profile_embedded: output_file_icc
            .as_ref()
            .map(|inspection| inspection.embedded),
        output_file_icc_profile_valid: output_file_icc.as_ref().map(|inspection| inspection.valid),
        output_file_icc_profile_description: output_file_icc
            .as_ref()
            .and_then(|inspection| inspection.description.clone()),
        output_file_icc_profile_matches_report,
        output_file_width: output_file_icc
            .as_ref()
            .and_then(|inspection| inspection.width),
        output_file_height: output_file_icc
            .as_ref()
            .and_then(|inspection| inspection.height),
        output_file_color_type: output_file_icc
            .as_ref()
            .and_then(|inspection| inspection.color_type.clone()),
        output_file_sample_format: output_file_icc
            .as_ref()
            .and_then(|inspection| inspection.sample_format.clone()),
        output_file_dimensions_match_report,
        output_file_storage_matches_report,
        output_width,
        output_height,
        render_intent: save_phase.and_then(|phase| string_metric(phase, "render_intent")),
        quality_mode,
        master_scene_referred_requested,
        master_scene_referred_path,
        master_scene_referred_sha256,
        master_scene_referred_file_sha256,
        master_scene_referred_file_sha256_matches_report,
        master_scene_referred_file_status,
        master_scene_referred_file_width: master_scene_referred_file
            .as_ref()
            .and_then(|inspection| inspection.width),
        master_scene_referred_file_height: master_scene_referred_file
            .as_ref()
            .and_then(|inspection| inspection.height),
        master_scene_referred_file_color_type: master_scene_referred_file
            .as_ref()
            .and_then(|inspection| inspection.color_type.clone()),
        master_scene_referred_file_sample_format: master_scene_referred_file
            .as_ref()
            .and_then(|inspection| inspection.sample_format.clone()),
        master_scene_referred_file_icc_profile_valid: master_scene_referred_file
            .as_ref()
            .map(|inspection| inspection.valid),
        master_scene_referred_file_icc_profile_description: master_scene_referred_file
            .as_ref()
            .and_then(|inspection| inspection.description.clone()),
        master_scene_referred_file_matches_report,
        review_srgb_requested,
        review_srgb_path,
        review_srgb_sha256,
        review_srgb_file_sha256,
        review_srgb_file_sha256_matches_report,
        review_srgb_file_status,
        review_srgb_file_width: review_srgb_file
            .as_ref()
            .and_then(|inspection| inspection.width),
        review_srgb_file_height: review_srgb_file
            .as_ref()
            .and_then(|inspection| inspection.height),
        review_srgb_file_color_type: review_srgb_file
            .as_ref()
            .and_then(|inspection| inspection.color_type.clone()),
        review_srgb_file_icc_profile_valid: review_srgb_file
            .as_ref()
            .map(|inspection| inspection.icc_profile_valid),
        review_srgb_file_icc_profile_description: review_srgb_file
            .as_ref()
            .and_then(|inspection| inspection.icc_profile_description.clone()),
        review_srgb_file_icc_profile_matches_standard_srgb: review_srgb_file
            .as_ref()
            .map(|inspection| inspection.icc_profile_matches_standard_srgb),
        review_srgb_file_matches_report,
        review_srgb_gamut_mapping_space,
        review_srgb_gamut_mapped_ratio,
        review_srgb_gamut_mapping_mean_chroma_scale,
        review_srgb_gamut_mapping_min_chroma_scale,
        review_srgb_post_map_out_of_gamut_pixel_count,
        review_srgb_gamut_mapping_supported,
        review_sidecar_sha256: save_phase
            .and_then(|phase| string_metric(phase, "review_sidecar_sha256")),
        input_base_confidence: save_phase
            .and_then(|phase| f64_metric(phase, "input_base_confidence")),
        render_review_status: save_phase
            .and_then(|phase| string_metric(phase, "render_review_status")),
        render_reviewable: save_phase.and_then(|phase| bool_metric(phase, "render_reviewable")),
        render_review_reason: save_phase
            .and_then(|phase| string_metric(phase, "render_review_reason")),
        tone_output_review_required: save_phase
            .and_then(|phase| bool_metric(phase, "tone_output_review_required"))
            .or(tone.tone_output_review_required),
        tone_output_review_reason: save_phase
            .and_then(|phase| string_metric(phase, "tone_output_review_reason"))
            .or_else(|| tone.tone_output_review_reason.clone()),
        tone_output_confidence_status: save_phase
            .and_then(|phase| string_metric(phase, "tone_output_confidence_status"))
            .or_else(|| tone.tone_output_confidence_status.clone()),
        tone_output_evidence_confidence: save_phase
            .and_then(|phase| f64_metric(phase, "tone_output_evidence_confidence"))
            .or(tone.tone_output_evidence_confidence),
        tone_output_render_luminance_range_p05_p95: tone.render_luminance_range_p05_p95,
        tone_output_render_to_mapped_luminance_range_ratio: tone
            .render_to_mapped_luminance_range_ratio,
        tone_output_maximum_post_tone_high_clip_ratio: tone.maximum_post_tone_high_clip_ratio,
        tone_output_maximum_post_tone_low_clip_ratio: tone.maximum_post_tone_low_clip_ratio,
        positive_input_likely_negative_like: positive_input_inspection
            .and_then(|inspection| inspection.get("likely_negative_like"))
            .and_then(serde_json::Value::as_bool),
        positive_input_accepted_high_warm_score: positive_input_inspection
            .and_then(|inspection| inspection.get("accepted_high_warm_score"))
            .and_then(serde_json::Value::as_bool),
        positive_input_orange_mask_score: positive_input_inspection
            .and_then(|inspection| inspection.get("orange_mask_score"))
            .and_then(serde_json::Value::as_f64),
        positive_input_reason: positive_input_inspection
            .and_then(|inspection| string_value(inspection.get("reason"))),
        overwrote_existing_output: save_phase
            .and_then(|phase| bool_metric(phase, "overwrote_existing_output")),
        stale_render_artifact_count: save_phase
            .and_then(|phase| usize_metric(phase, "stale_render_artifact_count")),
        stale_render_artifacts: save_phase
            .and_then(|phase| phase.metrics.get("stale_render_artifacts"))
            .and_then(|value| value.as_array())
            .map(|artifacts| {
                artifacts
                    .iter()
                    .map(summarize_stale_render_artifact)
                    .collect()
            })
            .unwrap_or_default(),
        stitch_decision: stitch.decision.clone(),
        stitch_crop: stitch_phase
            .and_then(|phase| phase.metrics.get("crop"))
            .and_then(summarize_crop),
        base_estimate_source: base_density.base_estimate_source.clone(),
        render_input_source: colorspace.render_input_source.clone(),
        colorspace_mapping_strategy: colorspace.mapping_strategy.clone(),
        calibration_status: colorspace.calibration_status.clone(),
        calibration_source: colorspace.calibration_source.clone(),
        colorspace_candidate_risk: colorspace.candidate_risk.clone(),
        colorspace_tone_color_trust_state: colorspace.tone_color_trust_state.clone(),
        colorspace_selected_quality_score: colorspace.selected_quality_score,
        colorspace_density_monotonicity_score: selected_color_candidate
            .and_then(|candidate| candidate.density_monotonicity_score),
        colorspace_hue_linearity_score: selected_color_candidate
            .and_then(|candidate| candidate.hue_linearity_score),
        colorspace_saturation_preservation_median_ratio: selected_color_candidate
            .and_then(|candidate| candidate.saturation_preservation_median_ratio),
        colorspace_spatial_neutral_delta_p95: selected_color_candidate
            .and_then(|candidate| candidate.spatial_neutral_delta_p95),
        colorspace_memory_color_penalty: colorspace
            .selected_quality_components
            .as_ref()
            .and_then(|components| components.memory_color_penalty),
        colorspace_spatial_consistency_penalty: colorspace
            .selected_quality_components
            .as_ref()
            .and_then(|components| components.spatial_consistency_penalty),
        colorspace_reference_patch_rms_delta_e: colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.selected_rms_delta_e),
        colorspace_reference_patch_rms_delta_e2000: colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.selected_rms_delta_e2000),
        colorspace_pre_scale_preserved_ratio: colorspace.pre_scale_preserved_ratio,
        colorspace_post_scale_preserved_ratio: colorspace.post_scale_preserved_ratio,
        highlight_chroma_compressed_ratio: tone.highlight_chroma_compressed_ratio,
        highlight_neutral_chroma_compressed_ratio: tone.highlight_neutral_chroma_compressed_ratio,
        shadow_chroma_compressed_ratio: tone.shadow_chroma_compressed_ratio,
        high_frequency_grain: tone.high_frequency_grain.clone(),
        noise_reduction_enabled: tone.noise_reduction_enabled,
        noise_reduction_requested_enabled: tone.noise_reduction_requested_enabled,
        noise_reduction_requested_strength: tone.noise_reduction_requested_strength,
        noise_reduction_requested_scale: tone.noise_reduction_requested_scale,
        noise_reduction_applied_ratio: tone.noise_reduction_applied_ratio,
        noise_reduction_structure_gate_start: tone.noise_reduction_structure_gate_start,
        noise_reduction_structure_gate_end: tone.noise_reduction_structure_gate_end,
        noise_reduction_structure_excluded_ratio: tone.noise_reduction_structure_excluded_ratio,
        noise_reduction_mean_abs_chroma_delta: tone.noise_reduction_mean_abs_chroma_delta,
        noise_reduction_mean_abs_luma_delta: tone.noise_reduction_mean_abs_luma_delta,
        noise_reduction_flat_luma_p95_reduction_ratio: tone
            .noise_reduction_flat_luma_p95_reduction_ratio,
        noise_reduction_flat_chroma_p95_reduction_ratio: tone
            .noise_reduction_flat_chroma_p95_reduction_ratio,
    }
}

#[derive(Debug, Clone)]
struct RenderOutputIccInspectionSummary {
    status: String,
    width: Option<usize>,
    height: Option<usize>,
    color_type: Option<String>,
    sample_format: Option<Vec<u16>>,
    image_description: Option<String>,
    embedded: bool,
    valid: bool,
    description: Option<String>,
}

fn inspect_render_output_icc_profile(
    identity: &ReportIdentitySummary,
    output_path: Option<&str>,
) -> Option<RenderOutputIccInspectionSummary> {
    let path = render_output_path(identity, output_path)?;
    Some(inspect_tiff_artifact(&path))
}

fn inspect_tiff_artifact(path: &Path) -> RenderOutputIccInspectionSummary {
    let inspection = match tiff_io::inspect_tiff_icc_profile(path) {
        Ok(inspection) => inspection,
        Err(_) => {
            return RenderOutputIccInspectionSummary {
                status: "inspection_failed".to_string(),
                width: None,
                height: None,
                color_type: None,
                sample_format: None,
                image_description: None,
                embedded: false,
                valid: false,
                description: None,
            };
        }
    };
    let status = if inspection.icc_profile_valid {
        "valid_icc_profile"
    } else if inspection.icc_profile_embedded {
        "invalid_icc_profile"
    } else {
        "missing_icc_profile"
    };
    RenderOutputIccInspectionSummary {
        status: status.to_string(),
        width: usize::try_from(inspection.width).ok(),
        height: usize::try_from(inspection.height).ok(),
        color_type: Some(inspection.color_type),
        sample_format: inspection.sample_format,
        image_description: inspection.image_description,
        embedded: inspection.icc_profile_embedded,
        valid: inspection.icc_profile_valid,
        description: inspection.icc_profile_description,
    }
}

fn render_output_path(
    identity: &ReportIdentitySummary,
    output_path: Option<&str>,
) -> Option<PathBuf> {
    [identity.output_path.as_deref(), output_path]
        .into_iter()
        .flatten()
        .next()
        .map(|path| resolve_report_relative_path(identity, path))
}

fn profile_inspection_matches_report(
    report_embedded: Option<bool>,
    report_description: Option<&str>,
    inspection: &RenderOutputIccInspectionSummary,
) -> Option<bool> {
    let expected_embedded = report_embedded?;
    if inspection.embedded != expected_embedded {
        return Some(false);
    }
    if expected_embedded && !inspection.valid {
        return Some(false);
    }
    if let Some(expected_description) = report_description {
        return Some(inspection.description.as_deref() == Some(expected_description));
    }
    Some(true)
}

#[derive(Debug, Clone)]
struct ReviewSrgbInspectionSummary {
    status: String,
    width: Option<usize>,
    height: Option<usize>,
    color_type: Option<String>,
    icc_profile_valid: bool,
    icc_profile_description: Option<String>,
    icc_profile_matches_standard_srgb: bool,
}

fn inspect_review_srgb_artifact(path: &Path) -> ReviewSrgbInspectionSummary {
    let inspection = match tiff_io::inspect_srgb_png(path) {
        Ok(inspection) => inspection,
        Err(_) => {
            return ReviewSrgbInspectionSummary {
                status: "inspection_failed".to_string(),
                width: None,
                height: None,
                color_type: None,
                icc_profile_valid: false,
                icc_profile_description: None,
                icc_profile_matches_standard_srgb: false,
            };
        }
    };
    let valid = inspection.color_type == "Rgb8"
        && inspection.icc_profile_valid
        && inspection.icc_profile_matches_standard_srgb;
    ReviewSrgbInspectionSummary {
        status: if valid {
            "valid_srgb_review_png"
        } else {
            "invalid_srgb_review_png"
        }
        .to_string(),
        width: usize::try_from(inspection.width).ok(),
        height: usize::try_from(inspection.height).ok(),
        color_type: Some(inspection.color_type),
        icc_profile_valid: inspection.icc_profile_valid,
        icc_profile_description: inspection.icc_profile_description,
        icc_profile_matches_standard_srgb: inspection.icc_profile_matches_standard_srgb,
    }
}

fn summarize_stale_render_artifact(value: &serde_json::Value) -> StaleRenderArtifactSummary {
    StaleRenderArtifactSummary {
        path: string_value(value.get("path")),
        modified_at: string_value(value.get("modified_at")),
        size_bytes: u64_value(value.get("size_bytes")),
    }
}

fn summarize_crop(value: &serde_json::Value) -> Option<CropSummary> {
    if value.is_null() {
        return None;
    }
    Some(CropSummary {
        x_start: usize_value(value.get("x_start")),
        y_start: usize_value(value.get("y_start")),
        width: usize_value(value.get("width")),
        height: usize_value(value.get("height")),
    })
}

fn inspect_file_sha256(path: &Path) -> Option<String> {
    hash_file_sha256(path).ok().map(|(sha256, _)| sha256)
}

fn artifact_sha256_match(
    required: bool,
    declared_sha256: Option<&str>,
    actual_sha256: Option<&str>,
) -> Option<bool> {
    if !required && declared_sha256.is_none() {
        return None;
    }
    Some(
        declared_sha256.is_some_and(is_valid_sha256_hex)
            && declared_sha256
                .zip(actual_sha256)
                .is_some_and(|(declared, actual)| declared.eq_ignore_ascii_case(actual)),
    )
}

fn resolve_report_relative_path(identity: &ReportIdentitySummary, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        return candidate;
    }
    if let Some(working_directory) = identity
        .working_directory
        .as_deref()
        .filter(|directory| !directory.is_empty())
    {
        return PathBuf::from(working_directory).join(candidate);
    }
    if let Some(parent) = identity
        .source_report_path
        .as_deref()
        .map(Path::new)
        .and_then(Path::parent)
    {
        return parent.join(candidate);
    }
    candidate
}

fn inspect_color_debug_artifact(
    identity: &ReportIdentitySummary,
    kind: &str,
    path: &str,
) -> ColorDebugArtifactSummary {
    let resolved = resolve_report_relative_path(identity, path);
    let metadata = fs::metadata(&resolved).ok();
    let is_file = metadata.as_ref().is_some_and(|metadata| metadata.is_file());
    let modified_at = metadata
        .as_ref()
        .and_then(|metadata| metadata.modified().ok());
    let modified_at_unix_ms = modified_at.and_then(system_time_unix_ms);
    let fresh_for_report = match (modified_at_unix_ms, identity.generated_at_unix_ms) {
        (Some(modified), Some(generated)) => Some(modified >= generated),
        (None, Some(_)) if metadata.is_none() => Some(false),
        _ => None,
    };
    let status = if metadata.is_none() {
        "missing"
    } else if !is_file {
        "not_file"
    } else if fresh_for_report == Some(false) {
        "stale"
    } else if fresh_for_report == Some(true) {
        "fresh"
    } else {
        "exists_unknown_freshness"
    };

    ColorDebugArtifactSummary {
        kind: kind.to_string(),
        path: path.to_string(),
        status: status.to_string(),
        exists: is_file,
        fresh_for_report,
        file_size_bytes: metadata.as_ref().filter(|_| is_file).map(fs::Metadata::len),
        modified_at: modified_at.map(format_system_time_utc),
        modified_at_unix_ms,
    }
}

fn summarize_color_debug_artifacts(
    identity: &ReportIdentitySummary,
    phase: &PhaseReport,
) -> Vec<ColorDebugArtifactSummary> {
    [
        (
            "candidate_comparison",
            string_metric(phase, "color_candidate_comparison_artifact"),
        ),
        (
            "gamut_clipping_map",
            string_metric(phase, "gamut_clipping_map_artifact"),
        ),
        (
            "scene_referred_prophoto_float",
            string_metric(phase, "scene_referred_prophoto_float_artifact"),
        ),
    ]
    .into_iter()
    .filter_map(|(kind, path)| path.map(|path| inspect_color_debug_artifact(identity, kind, &path)))
    .collect()
}

fn color_debug_artifact_issues(artifacts: &[ColorDebugArtifactSummary]) -> Vec<String> {
    artifacts
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

pub fn summary_to_markdown(summary: &ValidationSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Validation Summary: {}\n\n", summary.fixture));
    out.push_str("| Area | Field | Value |\n");
    out.push_str("|-|-|-|\n");
    push_row(
        &mut out,
        "report",
        "source_report_path",
        &summary.report.source_report_path,
    );
    push_row(
        &mut out,
        "report",
        "generated_at",
        &summary.report.generated_at,
    );
    push_row(
        &mut out,
        "report",
        "binary_identity_status",
        &summary.report.binary_identity_status,
    );
    push_row(
        &mut out,
        "report",
        "binary_sha256",
        &summary.report.binary_sha256,
    );
    push_row(
        &mut out,
        "render",
        "output_path",
        &summary.render.output_path,
    );
    push_row(
        &mut out,
        "render",
        "output_modified_at",
        &summary.render.output_modified_at,
    );
    push_row(
        &mut out,
        "render",
        "output_dimensions",
        &format_dimensions(summary.render.output_width, summary.render.output_height),
    );
    push_row(
        &mut out,
        "render",
        "output_color_space",
        &summary.render.output_color_space,
    );
    push_row(
        &mut out,
        "render",
        "render_review_status",
        &summary.render.render_review_status,
    );
    push_row(
        &mut out,
        "render",
        "render_reviewable",
        &summary
            .render
            .render_reviewable
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "tone_output_review_required",
        &summary
            .render
            .tone_output_review_required
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "tone_output_confidence_status",
        &summary.render.tone_output_confidence_status,
    );
    push_row(
        &mut out,
        "render",
        "input_base_confidence",
        &format_f64(summary.render.input_base_confidence),
    );
    push_row(
        &mut out,
        "render",
        "positive_input_likely_negative_like",
        &summary
            .render
            .positive_input_likely_negative_like
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "positive_input_orange_mask_score",
        &format_f64(summary.render.positive_input_orange_mask_score),
    );
    push_row(
        &mut out,
        "render",
        "positive_input_accepted_high_warm_score",
        &summary
            .render
            .positive_input_accepted_high_warm_score
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "output_icc_profile_embedded",
        &summary
            .render
            .output_icc_profile_embedded
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "output_file_icc_profile_status",
        &summary.render.output_file_icc_profile_status,
    );
    push_row(
        &mut out,
        "render",
        "output_file_icc_profile_matches_report",
        &summary
            .render
            .output_file_icc_profile_matches_report
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "output_file_dimensions_match_report",
        &summary
            .render
            .output_file_dimensions_match_report
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "output_file_storage_matches_report",
        &summary
            .render
            .output_file_storage_matches_report
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "artifact_sha256_binding_required",
        &Some(summary.render.artifact_sha256_binding_required.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "output_file_sha256_matches_report",
        &summary
            .render
            .output_file_sha256_matches_report
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "render_intent",
        &summary.render.render_intent,
    );
    push_row(
        &mut out,
        "render",
        "quality_mode",
        &summary.render.quality_mode,
    );
    push_row(
        &mut out,
        "render",
        "master_scene_referred_requested",
        &summary
            .render
            .master_scene_referred_requested
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "master_scene_referred_path",
        &summary.render.master_scene_referred_path,
    );
    push_row(
        &mut out,
        "render",
        "master_scene_referred_file_status",
        &summary.render.master_scene_referred_file_status,
    );
    push_row(
        &mut out,
        "render",
        "master_scene_referred_file_matches_report",
        &summary
            .render
            .master_scene_referred_file_matches_report
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "master_scene_referred_file_sha256_matches_report",
        &summary
            .render
            .master_scene_referred_file_sha256_matches_report
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "review_srgb_requested",
        &summary
            .render
            .review_srgb_requested
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "review_srgb_path",
        &summary.render.review_srgb_path,
    );
    push_row(
        &mut out,
        "render",
        "review_srgb_file_status",
        &summary.render.review_srgb_file_status,
    );
    push_row(
        &mut out,
        "render",
        "review_srgb_file_matches_report",
        &summary
            .render
            .review_srgb_file_matches_report
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "review_srgb_file_sha256_matches_report",
        &summary
            .render
            .review_srgb_file_sha256_matches_report
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "review_srgb_gamut_mapping_space",
        &summary.render.review_srgb_gamut_mapping_space,
    );
    push_row(
        &mut out,
        "render",
        "review_srgb_gamut_mapped_ratio",
        &format_f64(summary.render.review_srgb_gamut_mapped_ratio),
    );
    push_row(
        &mut out,
        "render",
        "review_srgb_gamut_mapping_min_chroma_scale",
        &format_f64(summary.render.review_srgb_gamut_mapping_min_chroma_scale),
    );
    push_row(
        &mut out,
        "render",
        "review_srgb_post_map_out_of_gamut_pixel_count",
        &summary
            .render
            .review_srgb_post_map_out_of_gamut_pixel_count
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "render",
        "review_srgb_gamut_mapping_supported",
        &summary
            .render
            .review_srgb_gamut_mapping_supported
            .map(|value| value.to_string()),
    );
    let delivery_issues = delivery_artifact_integrity_issues(&summary.render);
    push_row(
        &mut out,
        "render",
        "delivery_artifact_integrity_issues",
        &Some(if delivery_issues.is_empty() {
            "none".to_string()
        } else {
            delivery_issues.join(",")
        }),
    );
    push_row(
        &mut out,
        "render",
        "stale_render_artifact_count",
        &summary
            .render
            .stale_render_artifact_count
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "input_orientation",
        "all_components_reported",
        &summary
            .input_orientation
            .all_components_reported
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "input_orientation",
        "components",
        &format_input_orientation_components(&summary.input_orientation.components),
    );
    push_row(&mut out, "deskew", "status", &summary.deskew.status);
    push_row(
        &mut out,
        "deskew",
        "applied",
        &summary.deskew.applied.map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "deskew",
        "detected_source_skew_degrees",
        &format_f64(summary.deskew.detected_source_skew_degrees),
    );
    push_row(
        &mut out,
        "deskew",
        "correction_degrees",
        &format_f64(summary.deskew.correction_degrees),
    );
    push_row(
        &mut out,
        "deskew",
        "retained_area_ratio",
        &format_f64(summary.deskew.retained_area_ratio),
    );
    push_row(
        &mut out,
        "deskew",
        "proposed_retained_area_ratio",
        &format_f64(summary.deskew.proposed_retained_area_ratio),
    );
    push_row(
        &mut out,
        "deskew",
        "review_required",
        &summary
            .deskew
            .review_required
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "deskew",
        "all_components_applied",
        &summary
            .deskew
            .all_components_applied
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "deskew",
        "minimum_component_retained_area_ratio",
        &format_f64(summary.deskew.minimum_component_retained_area_ratio),
    );
    push_row(
        &mut out,
        "border_crop",
        "all_components_cropped",
        &summary
            .border_crop
            .all_components_cropped
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "border_crop",
        "minimum_removed_edge_count_per_component",
        &summary
            .border_crop
            .minimum_removed_edge_count_per_component
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "border_crop",
        "minimum_retained_area_ratio",
        &format_f64(summary.border_crop.minimum_retained_area_ratio),
    );
    push_row(
        &mut out,
        "border_crop",
        "maximum_retained_area_ratio",
        &format_f64(summary.border_crop.maximum_retained_area_ratio),
    );
    push_row(&mut out, "stitch", "decision", &summary.stitch.decision);
    push_row(
        &mut out,
        "stitch",
        "confidence",
        &format_f64(summary.stitch.confidence),
    );
    push_row(
        &mut out,
        "stitch",
        "chosen_hypothesis",
        &summary.stitch.chosen_hypothesis,
    );
    push_row(
        &mut out,
        "stitch",
        "selection_reason",
        &summary.stitch.search_selection_reason,
    );
    let homography_spatial = summary.stitch.homography_spatial_validation.as_ref();
    let homography_feature = summary.stitch.homography_feature_validation.as_ref();
    push_row(
        &mut out,
        "stitch",
        "homography_feature_validation_accepted",
        &homography_feature
            .and_then(|validation| validation.accepted)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "stitch",
        "homography_feature_validations",
        &homography_feature.map(|validation| {
            format!(
                "{}/{}",
                validation.accepted_count, validation.validation_count
            )
        }),
    );
    push_row(
        &mut out,
        "stitch",
        "homography_minimum_held_out_feature_inlier_ratio",
        &format_f64(
            homography_feature.and_then(|validation| validation.minimum_held_out_inlier_ratio),
        ),
    );
    push_row(
        &mut out,
        "stitch",
        "homography_maximum_cross_fit_disagreement_px",
        &format_f64(
            homography_feature.and_then(|validation| validation.maximum_cross_fit_disagreement_px),
        ),
    );
    push_row(
        &mut out,
        "stitch",
        "homography_spatial_validation_accepted",
        &homography_spatial
            .and_then(|validation| validation.accepted)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "stitch",
        "homography_spatial_validations",
        &homography_spatial.map(|validation| {
            format!(
                "{}/{}",
                validation.accepted_count, validation.validation_count
            )
        }),
    );
    push_row(
        &mut out,
        "stitch",
        "homography_minimum_split_ncc_improvement",
        &format_f64(
            homography_spatial.and_then(|validation| validation.minimum_split_ncc_improvement),
        ),
    );
    push_row(
        &mut out,
        "stitch",
        "homography_minimum_mean_ncc_improvement",
        &format_f64(
            homography_spatial.and_then(|validation| validation.minimum_mean_ncc_improvement),
        ),
    );
    push_row(
        &mut out,
        "stitch",
        "homography_minimum_mean_registration_error_reduction",
        &format_f64(
            homography_spatial
                .and_then(|validation| validation.minimum_mean_registration_error_reduction),
        ),
    );
    push_row(
        &mut out,
        "white_balance",
        "technical_status",
        &summary.white_balance.technical_status,
    );
    push_row(
        &mut out,
        "white_balance",
        "technical_applied",
        &summary
            .white_balance
            .technical_applied
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "white_balance",
        "technical_source_cct_kelvin",
        &format_f64(summary.white_balance.technical_estimated_source_cct_kelvin),
    );
    push_row(
        &mut out,
        "white_balance",
        "creative_temperature",
        &format_f64(summary.white_balance.creative_temperature),
    );
    push_row(
        &mut out,
        "white_balance",
        "creative_tint",
        &format_f64(summary.white_balance.creative_tint),
    );
    let seam_exposure = summary.stitch.seam_exposure_correction.as_ref();
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_applied",
        &seam_exposure
            .and_then(|correction| correction.applied)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_model",
        &seam_exposure.and_then(|correction| correction.model.clone()),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_gain_rgb",
        &seam_exposure.and_then(|correction| format_f64_vec(&correction.gain_rgb)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_gain_log_slope_x_rgb",
        &seam_exposure
            .and_then(|correction| format_f64_vec(&correction.spatial_gain_log_slope_x_rgb)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_gain_log_slope_y_rgb",
        &seam_exposure
            .and_then(|correction| format_f64_vec(&correction.spatial_gain_log_slope_y_rgb)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_gain_top_rgb",
        &seam_exposure.and_then(|correction| format_f64_vec(&correction.spatial_gain_top_rgb)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_gain_bottom_rgb",
        &seam_exposure.and_then(|correction| format_f64_vec(&correction.spatial_gain_bottom_rgb)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_offset_slope_x_rgb",
        &seam_exposure
            .and_then(|correction| format_f64_vec(&correction.spatial_offset_slope_x_rgb)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_offset_slope_y_rgb",
        &seam_exposure
            .and_then(|correction| format_f64_vec(&correction.spatial_offset_slope_y_rgb)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_offset_top_rgb_normalized",
        &seam_exposure
            .and_then(|correction| format_f64_vec(&correction.spatial_offset_top_rgb_normalized)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_offset_bottom_rgb_normalized",
        &seam_exposure.and_then(|correction| {
            format_f64_vec(&correction.spatial_offset_bottom_rgb_normalized)
        }),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_offset_rgb",
        &seam_exposure.and_then(|correction| format_f64_vec(&correction.offset_rgb)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_score_before",
        &format_f64(seam_exposure.and_then(|correction| correction.seam_score_before)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_score_after",
        &format_f64(seam_exposure.and_then(|correction| correction.seam_score_after)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_held_out_validation_passed",
        &seam_exposure
            .and_then(|correction| correction.held_out_validation_passed)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_held_out_selected_score",
        &format_f64(seam_exposure.and_then(|correction| correction.held_out_selected_seam_score)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_held_out_spatial_gain_score",
        &format_f64(
            seam_exposure.and_then(|correction| correction.held_out_spatial_gain_seam_score),
        ),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_slope_agreement_ratio",
        &format_f64(seam_exposure.and_then(|correction| correction.spatial_slope_agreement_ratio)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_held_out_spatial_gain_offset_score",
        &format_f64(
            seam_exposure.and_then(|correction| correction.held_out_spatial_gain_offset_seam_score),
        ),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_affine_slope_agreement_ratio",
        &format_f64(
            seam_exposure.and_then(|correction| correction.spatial_affine_slope_agreement_ratio),
        ),
    );
    let spatial_2d = seam_exposure.and_then(|correction| correction.spatial_2d_validation.as_ref());
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_2d_gain_accepted",
        &spatial_2d
            .and_then(|validation| validation.gain_accepted)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_2d_gain_offset_accepted",
        &spatial_2d
            .and_then(|validation| validation.gain_offset_accepted)
            .map(|value| value.to_string()),
    );
    let spatial_quadratic = spatial_2d.and_then(|validation| validation.quadratic.as_ref());
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_quadratic_gain_accepted",
        &spatial_quadratic
            .and_then(|validation| validation.gain_accepted)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_quadratic_gain_offset_accepted",
        &spatial_quadratic
            .and_then(|validation| validation.gain_offset_accepted)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_2d_horizontal_slope_agreement_ratio",
        &format_f64(spatial_2d.and_then(|validation| {
            if validation.gain_offset_accepted == Some(true) {
                validation.gain_offset_horizontal_slope_agreement_ratio
            } else {
                validation.gain_horizontal_slope_agreement_ratio
            }
        })),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_exposure_spatial_2d_held_out_score",
        &format_f64(spatial_2d.and_then(|validation| {
            if validation.gain_offset_accepted == Some(true) {
                validation.held_out_gain_offset_seam_score
            } else {
                validation.held_out_gain_seam_score
            }
        })),
    );
    let seam_blend = summary.stitch.seam_blend.as_ref();
    push_row(
        &mut out,
        "stitch",
        "seam_blend_mode",
        &seam_blend.and_then(|blend| blend.mode.clone()),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_blend_applied_merges",
        &seam_blend.map(|blend| format!("{}/{}", blend.applied_merge_count, blend.merge_count)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_blend_review_required",
        &seam_blend
            .and_then(|blend| blend.review_required)
            .map(|value| value.to_string()),
    );
    let seam_detail = seam_blend.and_then(|blend| blend.detail_consistency.as_ref());
    push_row(
        &mut out,
        "stitch",
        "seam_detail_review_required",
        &seam_detail
            .and_then(|detail| detail.review_required)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_detail_supported_scales_min",
        &seam_detail
            .and_then(|detail| detail.minimum_supported_scale_count)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_detail_max_symmetric_energy_ratio",
        &format_f64(seam_detail.and_then(|detail| detail.maximum_symmetric_energy_ratio)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_overlap_p95_abs_difference",
        &format_f64(seam_blend.and_then(|blend| blend.overlap_p95_abs_difference)),
    );
    push_row(
        &mut out,
        "stitch",
        "seam_output_to_source_gradient_ratio",
        &format_f64(seam_blend.and_then(|blend| blend.output_to_source_seam_gradient_ratio)),
    );
    push_row(
        &mut out,
        "density",
        "base_confidence",
        &format_f64(summary.base_density.base_confidence),
    );
    push_row(
        &mut out,
        "density",
        "raw_base_proxy_confidence",
        &format_f64(summary.base_density.raw_base_proxy_confidence),
    );
    push_row(
        &mut out,
        "density",
        "raw_base_support_fraction",
        &format_f64(summary.base_density.raw_base_support_fraction),
    );
    push_row(
        &mut out,
        "density",
        "density_confidence",
        &format_f64(summary.base_density.density_confidence),
    );
    push_row(
        &mut out,
        "negative_reconstruction",
        "response_model",
        &summary.negative_reconstruction.response_model,
    );
    push_row(
        &mut out,
        "negative_reconstruction",
        "measured_model_id",
        &summary.negative_reconstruction.measured_model_id,
    );
    push_row(
        &mut out,
        "negative_reconstruction",
        "held_out_delta_e00_rms",
        &format_f64(summary.negative_reconstruction.held_out_delta_e00_rms),
    );
    push_row(
        &mut out,
        "negative_reconstruction",
        "held_out_improvement_over_unit_slope",
        &format_f64(
            summary
                .negative_reconstruction
                .held_out_improvement_over_unit_slope,
        ),
    );
    push_row(
        &mut out,
        "negative_reconstruction",
        "maximum_density_noise_gain",
        &format_f64(summary.negative_reconstruction.maximum_density_noise_gain),
    );
    push_row(
        &mut out,
        "negative_reconstruction",
        "curve_extrapolated_any_ratio",
        &format_f64(summary.negative_reconstruction.curve_extrapolated_any_ratio),
    );
    push_row(
        &mut out,
        "negative_reconstruction",
        "review_required",
        &summary
            .negative_reconstruction
            .reconstruction_review_required
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "colorspace",
        "render_input_source",
        &summary.colorspace.render_input_source,
    );
    push_row(
        &mut out,
        "colorspace",
        "calibration_record_status",
        &summary.colorspace.calibration_status,
    );
    push_row(
        &mut out,
        "colorspace",
        "calibration_source",
        &summary.colorspace.calibration_source,
    );
    let calibration_color_mapping = summary
        .colorspace
        .calibration_color_mapping_application
        .as_ref();
    push_row(
        &mut out,
        "colorspace",
        "calibration_color_mapping_evaluated",
        &calibration_color_mapping
            .and_then(|application| application.evaluated)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "colorspace",
        "calibration_color_mapping_applied",
        &calibration_color_mapping
            .and_then(|application| application.applied)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "colorspace",
        "calibration_color_mapping_status",
        &calibration_color_mapping.and_then(|application| application.selection_status.clone()),
    );
    push_row(
        &mut out,
        "colorspace",
        "calibration_color_mapping_selected_candidate",
        &calibration_color_mapping.and_then(|application| application.selected_candidate.clone()),
    );
    push_row(
        &mut out,
        "colorspace",
        "calibration_color_mapping_preferred_candidate",
        &calibration_color_mapping.and_then(|application| application.preferred_candidate.clone()),
    );
    push_row(
        &mut out,
        "colorspace",
        "calibration_color_mapping_reason",
        &calibration_color_mapping.and_then(|application| application.reason.clone()),
    );
    push_row(
        &mut out,
        "colorspace",
        "calibration_color_mapping_consistency_issues",
        &Some(if summary.diagnostic_consistency_issues.is_empty() {
            "none".to_string()
        } else {
            summary.diagnostic_consistency_issues.join(",")
        }),
    );
    push_row(
        &mut out,
        "colorspace",
        "calibration_scanner_profile",
        &summary.colorspace.calibration_scanner_profile_id,
    );
    push_row(
        &mut out,
        "colorspace",
        "calibration_roll_profile",
        &summary.colorspace.calibration_roll_profile_id,
    );
    push_row(
        &mut out,
        "colorspace",
        "calibration_profile",
        &summary.colorspace.calibration_external_profile_path,
    );
    push_row(
        &mut out,
        "colorspace",
        "calibration_film_stock_status",
        &summary.colorspace.calibration_film_stock_status,
    );
    push_row(
        &mut out,
        "colorspace",
        "mapping_strategy",
        &summary.colorspace.mapping_strategy,
    );
    push_row(
        &mut out,
        "colorspace",
        "selected_candidate",
        &summary.colorspace.selected_candidate,
    );
    push_row(
        &mut out,
        "colorspace",
        "selected_candidate_rank",
        &summary
            .colorspace
            .selected_candidate_rank
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "colorspace",
        "selected_candidate_score",
        &format_f64(summary.colorspace.selected_candidate_score),
    );
    push_row(
        &mut out,
        "colorspace",
        "selected_quality_score",
        &format_f64(summary.colorspace.selected_quality_score),
    );
    push_row(
        &mut out,
        "colorspace",
        "technical_safety_score",
        &format_f64(summary.colorspace.technical_safety_score),
    );
    push_row(
        &mut out,
        "colorspace",
        "color_fidelity_score",
        &format_f64(summary.colorspace.color_fidelity_score),
    );
    push_row(
        &mut out,
        "colorspace",
        "selected_quality_components",
        &format_quality_components(&summary.colorspace.selected_quality_components),
    );
    push_row(
        &mut out,
        "colorspace",
        "selected_density_monotonicity_score",
        &format_f64(summary.render.colorspace_density_monotonicity_score),
    );
    push_row(
        &mut out,
        "colorspace",
        "selected_hue_linearity_score",
        &format_f64(summary.render.colorspace_hue_linearity_score),
    );
    push_row(
        &mut out,
        "colorspace",
        "selected_saturation_preservation_median_ratio",
        &format_f64(
            summary
                .render
                .colorspace_saturation_preservation_median_ratio,
        ),
    );
    push_row(
        &mut out,
        "colorspace",
        "selected_spatial_neutral_delta_p95",
        &format_f64(summary.render.colorspace_spatial_neutral_delta_p95),
    );
    push_row(
        &mut out,
        "colorspace",
        "selected_memory_color_penalty",
        &format_f64(summary.render.colorspace_memory_color_penalty),
    );
    push_row(
        &mut out,
        "colorspace",
        "selected_spatial_consistency_penalty",
        &format_f64(summary.render.colorspace_spatial_consistency_penalty),
    );
    push_row(
        &mut out,
        "colorspace",
        "candidate_risk",
        &summary.colorspace.candidate_risk,
    );
    push_row(
        &mut out,
        "colorspace",
        "tone_color_trust_state",
        &summary.colorspace.tone_color_trust_state,
    );
    push_row(
        &mut out,
        "colorspace",
        "neutral_safety_rescue_applied",
        &summary
            .colorspace
            .neutral_safety_rescue
            .as_ref()
            .and_then(|rescue| rescue.applied)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "colorspace",
        "neutral_safety_rescue_preserved_ratio_gain",
        &format_f64(
            summary
                .colorspace
                .neutral_safety_rescue
                .as_ref()
                .and_then(|rescue| rescue.preserved_ratio_gain),
        ),
    );
    push_row(
        &mut out,
        "colorspace",
        "neutral_safety_rescue_midtone_saturation_p95_reduction",
        &format_f64(
            summary
                .colorspace
                .neutral_safety_rescue
                .as_ref()
                .and_then(|rescue| rescue.midtone_saturation_p95_reduction),
        ),
    );
    push_row(
        &mut out,
        "colorspace",
        "neutral_safety_rescue_reason",
        &summary
            .colorspace
            .neutral_safety_rescue
            .as_ref()
            .and_then(|rescue| rescue.reason.clone()),
    );
    push_row(
        &mut out,
        "colorspace",
        "selected_runner_up_quality_delta",
        &format_f64(summary.colorspace.selected_runner_up_quality_delta),
    );
    push_row(
        &mut out,
        "colorspace",
        "candidate_acceptance",
        &format_candidate_acceptance(&summary.colorspace.candidate_acceptance),
    );
    push_row(
        &mut out,
        "colorspace",
        "calibration_acceptance",
        &summary
            .colorspace
            .calibration_acceptance
            .as_ref()
            .and_then(|acceptance| acceptance.status.clone()),
    );
    push_row(
        &mut out,
        "colorspace",
        "neutral_estimate_quality",
        &format_f64(
            summary
                .colorspace
                .neutral_estimate_quality
                .as_ref()
                .and_then(|quality| quality.score),
        ),
    );
    push_row(
        &mut out,
        "colorspace",
        "neutral_sample_rejections",
        &summary
            .colorspace
            .neutral_sample_rejections
            .as_ref()
            .map(format_neutral_sample_rejections),
    );
    push_row(
        &mut out,
        "colorspace",
        "dominant_anchor_sample_rejections",
        &summary
            .colorspace
            .dominant_anchor_sample_rejections
            .as_ref()
            .map(format_dominant_anchor_sample_rejections),
    );
    push_row(
        &mut out,
        "colorspace",
        "dominant_anchor_quality",
        &summary
            .colorspace
            .dominant_anchor_quality
            .as_ref()
            .map(format_dominant_anchor_quality),
    );
    push_row(
        &mut out,
        "colorspace",
        "reference_patch_fit",
        &summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .map(format_reference_patch_evaluation),
    );
    push_row(
        &mut out,
        "colorspace",
        "exposure_scale",
        &format_f64(summary.colorspace.exposure_scale),
    );
    push_row(
        &mut out,
        "colorspace",
        "post_scale_preserved_ratio",
        &format_f64(summary.colorspace.post_scale_preserved_ratio),
    );
    push_row(
        &mut out,
        "colorspace",
        "color_candidate_comparison_artifact",
        &summary.colorspace.color_candidate_comparison_artifact,
    );
    push_row(
        &mut out,
        "colorspace",
        "gamut_clipping_map_artifact",
        &summary.colorspace.gamut_clipping_map_artifact,
    );
    push_row(
        &mut out,
        "colorspace",
        "scene_referred_prophoto_float_artifact",
        &summary.colorspace.scene_referred_prophoto_float_artifact,
    );
    push_row(
        &mut out,
        "colorspace",
        "debug_artifacts",
        &format_color_debug_artifacts(&summary.colorspace.debug_artifacts),
    );
    push_row(
        &mut out,
        "colorspace",
        "gamut_clipping_map_preserved_ratio",
        &format_f64(summary.colorspace.gamut_clipping_map_preserved_ratio),
    );
    push_row(
        &mut out,
        "colorspace",
        "calibrated_vs_image_exposure_delta",
        &delta(
            summary.colorspace.image_matrix_exposure_scale,
            summary.colorspace.calibrated_profile_exposure_scale,
        )
        .map(|value| format!("{value:.6}")),
    );
    push_row(
        &mut out,
        "colorspace",
        "channel_anchor_low_support",
        &format_bool_vec(&summary.colorspace.channel_anchor_low_support),
    );
    push_row(
        &mut out,
        "colorspace",
        "neutral_trim_applied",
        &summary
            .colorspace
            .neutral_trim_applied
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "colorspace",
        "neutral_trim_scale",
        &format_f64_vec(&summary.colorspace.neutral_trim_scale),
    );
    push_row(
        &mut out,
        "colorspace",
        "neutral_trim_delta_before",
        &format_f64(
            summary
                .colorspace
                .neutral_trim_before_after
                .as_ref()
                .and_then(|trim| trim.before_neutral_delta_magnitude),
        ),
    );
    push_row(
        &mut out,
        "colorspace",
        "neutral_trim_delta_after",
        &format_f64(
            summary
                .colorspace
                .neutral_trim_before_after
                .as_ref()
                .and_then(|trim| trim.after_neutral_delta_magnitude),
        ),
    );
    push_row(
        &mut out,
        "colorspace",
        "neutral_trim_band_delta_after",
        &summary
            .colorspace
            .neutral_trim_before_after
            .as_ref()
            .and_then(|trim| format_option_f64_vec(&trim.after_neutral_band_delta_magnitude)),
    );
    push_row(
        &mut out,
        "colorspace",
        "neutral_trim_band_worsened",
        &summary
            .colorspace
            .neutral_trim_before_after
            .as_ref()
            .and_then(|trim| trim.neutral_band_delta_worsened)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "tone",
        "tone_confidence_status",
        &summary.tone.tone_confidence_status,
    );
    push_row(
        &mut out,
        "tone",
        "tone_output_confidence_status",
        &summary.tone.tone_output_confidence_status,
    );
    push_row(
        &mut out,
        "tone",
        "tone_output_evidence_confidence",
        &format_f64(summary.tone.tone_output_evidence_confidence),
    );
    push_row(
        &mut out,
        "tone",
        "tone_output_review_required",
        &summary
            .tone
            .tone_output_review_required
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "tone",
        "input_luminance_range_p05_p95",
        &format_f64(summary.tone.input_luminance_range_p05_p95),
    );
    push_row(
        &mut out,
        "tone",
        "mapped_luminance_range_p05_p95",
        &format_f64(summary.tone.mapped_luminance_range_p05_p95),
    );
    push_row(
        &mut out,
        "tone",
        "render_to_mapped_luminance_range_ratio",
        &format_f64(summary.tone.render_to_mapped_luminance_range_ratio),
    );
    push_row(
        &mut out,
        "tone",
        "maximum_post_tone_high_clip_ratio",
        &format_f64(summary.tone.maximum_post_tone_high_clip_ratio),
    );
    push_row(
        &mut out,
        "tone",
        "maximum_post_tone_low_clip_ratio",
        &format_f64(summary.tone.maximum_post_tone_low_clip_ratio),
    );
    push_row(
        &mut out,
        "tone",
        "highlight_chroma_compressed_ratio",
        &format_f64(summary.tone.highlight_chroma_compressed_ratio),
    );
    push_row(
        &mut out,
        "tone",
        "highlight_neutral_chroma_compressed_ratio",
        &format_f64(summary.tone.highlight_neutral_chroma_compressed_ratio),
    );
    push_row(
        &mut out,
        "tone",
        "color_protection_policy",
        &summary.tone.color_protection_policy,
    );
    push_row(
        &mut out,
        "tone",
        "color_trust_state",
        &summary.tone.color_trust_state,
    );
    push_row(
        &mut out,
        "tone",
        "color_protection_reason",
        &summary.tone.color_protection_reason,
    );
    let skin_memory = summary
        .tone
        .adaptive_vibrance_skin_memory_protection
        .as_ref();
    push_row(
        &mut out,
        "tone",
        "adaptive_vibrance_skin_memory_protection_enabled",
        &skin_memory
            .and_then(|protection| protection.enabled)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "tone",
        "adaptive_vibrance_skin_memory_working_space",
        &skin_memory.and_then(|protection| protection.working_space.clone()),
    );
    push_row(
        &mut out,
        "tone",
        "adaptive_vibrance_skin_memory_protected_ratio",
        &format_f64(skin_memory.and_then(|protection| protection.protected_pixel_ratio)),
    );
    push_row(
        &mut out,
        "tone",
        "adaptive_vibrance_skin_memory_mean_protection",
        &format_f64(skin_memory.and_then(|protection| protection.mean_protection_weight)),
    );
    let preferred_memory = summary
        .tone
        .adaptive_vibrance_preferred_memory_color_guard
        .as_ref();
    push_row(
        &mut out,
        "tone",
        "adaptive_vibrance_preferred_memory_color_guard_enabled",
        &preferred_memory
            .and_then(|guard| guard.enabled)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "tone",
        "adaptive_vibrance_preferred_memory_color_working_space",
        &preferred_memory.and_then(|guard| guard.working_space.clone()),
    );
    push_row(
        &mut out,
        "tone",
        "adaptive_vibrance_preferred_memory_color_matched_ratio",
        &format_f64(preferred_memory.and_then(|guard| guard.matched_pixel_ratio)),
    );
    push_row(
        &mut out,
        "tone",
        "adaptive_vibrance_preferred_memory_color_limited_ratio",
        &format_f64(preferred_memory.and_then(|guard| guard.limited_pixel_ratio)),
    );
    push_row(
        &mut out,
        "tone",
        "adaptive_vibrance_preferred_memory_color_mean_scale_reduction",
        &format_f64(preferred_memory.and_then(|guard| guard.mean_scale_reduction)),
    );
    let preferred_skin = summary.tone.preferred_skin_rendering.as_ref();
    push_row(
        &mut out,
        "tone",
        "preferred_skin_rendering_enabled",
        &preferred_skin
            .and_then(|rendering| rendering.enabled)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "tone",
        "preferred_skin_rendering_working_space",
        &preferred_skin.and_then(|rendering| rendering.working_space.clone()),
    );
    push_row(
        &mut out,
        "tone",
        "preferred_skin_rendering_matched_ratio",
        &format_f64(preferred_skin.and_then(|rendering| rendering.matched_pixel_ratio)),
    );
    push_row(
        &mut out,
        "tone",
        "preferred_skin_rendering_adjusted_ratio",
        &format_f64(preferred_skin.and_then(|rendering| rendering.adjusted_pixel_ratio)),
    );
    push_row(
        &mut out,
        "tone",
        "preferred_skin_rendering_mean_delta_e_ab",
        &format_f64(preferred_skin.and_then(|rendering| rendering.mean_delta_e_ab)),
    );
    push_row(
        &mut out,
        "tone",
        "shadow_chroma_compressed_ratio",
        &format_f64(summary.tone.shadow_chroma_compressed_ratio),
    );
    push_row(
        &mut out,
        "tone",
        "shadow_saturation_p95",
        &format_f64(summary.tone.shadow_saturation_p95),
    );
    push_row(
        &mut out,
        "tone",
        "shadow_rgb_median",
        &format_f64_vec(&summary.tone.shadow_rgb_median),
    );
    push_row(
        &mut out,
        "tone",
        "shadow_visible_saturation_p95",
        &format_f64(summary.tone.shadow_visible_saturation_p95),
    );
    push_row(
        &mut out,
        "tone",
        "shadow_visible_pixel_count",
        &format_usize(summary.tone.shadow_visible_pixel_count),
    );
    push_row(
        &mut out,
        "tone",
        "shadow_visible_rgb_median",
        &format_f64_vec(&summary.tone.shadow_visible_rgb_median),
    );
    push_row(
        &mut out,
        "tone",
        "render_luminance_range_p05_p95",
        &format_f64(summary.tone.render_luminance_range_p05_p95),
    );
    push_row(
        &mut out,
        "tone",
        "midtone_saturation_p95",
        &format_f64(summary.tone.midtone_saturation_p95),
    );
    push_row(
        &mut out,
        "tone",
        "midtone_rgb_median",
        &format_f64_vec(&summary.tone.midtone_rgb_median),
    );
    push_row(
        &mut out,
        "tone",
        "midtone_neutral_saturation_p95",
        &format_f64(summary.tone.midtone_neutral_saturation_p95),
    );
    push_row(
        &mut out,
        "tone",
        "midtone_neutral_pixel_count",
        &format_usize(summary.tone.midtone_neutral_pixel_count),
    );
    push_row(
        &mut out,
        "tone",
        "midtone_neutral_rgb_median",
        &format_f64_vec(&summary.tone.midtone_neutral_rgb_median),
    );
    push_row(
        &mut out,
        "tone",
        "bright_neutral_saturation_p95",
        &format_f64(summary.tone.bright_neutral_saturation_p95),
    );
    push_row(
        &mut out,
        "tone",
        "bright_neutral_rgb_median",
        &format_f64_vec(&summary.tone.bright_neutral_rgb_median),
    );
    push_row(
        &mut out,
        "tone",
        "bright_saturated_saturation_p95",
        &format_f64(summary.tone.bright_saturated_saturation_p95),
    );
    push_row(
        &mut out,
        "tone",
        "noise_reduction_enabled",
        &summary
            .tone
            .noise_reduction_enabled
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "tone",
        "noise_reduction_applied_ratio",
        &format_f64(summary.tone.noise_reduction_applied_ratio),
    );
    push_row(
        &mut out,
        "tone",
        "noise_reduction_structure_gate_start",
        &format_f64(summary.tone.noise_reduction_structure_gate_start),
    );
    push_row(
        &mut out,
        "tone",
        "noise_reduction_structure_gate_end",
        &format_f64(summary.tone.noise_reduction_structure_gate_end),
    );
    push_row(
        &mut out,
        "tone",
        "noise_reduction_structure_excluded_ratio",
        &format_f64(summary.tone.noise_reduction_structure_excluded_ratio),
    );
    push_row(
        &mut out,
        "tone",
        "noise_reduction_saturation_limited_ratio",
        &format_f64(summary.tone.noise_reduction_saturation_limited_ratio),
    );
    let grain = summary.tone.high_frequency_grain.as_ref();
    push_row(
        &mut out,
        "tone",
        "high_frequency_luma_residual_p95",
        &format_f64(grain.and_then(|grain| grain.luma_residual_p95)),
    );
    push_row(
        &mut out,
        "tone",
        "high_frequency_chroma_residual_p95",
        &format_f64(grain.and_then(|grain| grain.chroma_residual_p95)),
    );
    push_row(
        &mut out,
        "tone",
        "high_frequency_flat_sample_ratio",
        &format_f64(grain.and_then(|grain| grain.flat_sample_ratio)),
    );
    push_row(
        &mut out,
        "tone",
        "high_frequency_flat_luma_residual_p95",
        &format_f64(grain.and_then(|grain| grain.flat_luma_residual_p95)),
    );
    push_row(
        &mut out,
        "tone",
        "high_frequency_flat_chroma_residual_p95",
        &format_f64(grain.and_then(|grain| grain.flat_chroma_residual_p95)),
    );
    let grain_detail = summary.tone.grain_detail_retention.as_ref();
    push_row(
        &mut out,
        "tone",
        "grain_detail_review_required",
        &grain_detail
            .and_then(|detail| detail.review_required)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "tone",
        "grain_detail_decision_supported",
        &grain_detail
            .and_then(|detail| detail.decision_supported)
            .map(|value| value.to_string()),
    );
    push_row(
        &mut out,
        "tone",
        "grain_detail_luminance_p10_retention",
        &format_f64(grain_detail.and_then(|detail| detail.luminance_p10_retention)),
    );
    push_row(
        &mut out,
        "tone",
        "grain_detail_chroma_p10_retention",
        &format_f64(grain_detail.and_then(|detail| detail.chroma_p10_retention)),
    );
    if let Some(comparison) = &summary.summary_baseline_comparison {
        push_row(
            &mut out,
            "summary_baseline",
            "status",
            &Some(comparison.status.clone()),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "issues",
            &Some(comparison.issues.join(", ")),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "baseline_summary_path",
            &Some(comparison.baseline_summary_path.clone()),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "output_dimensions_match",
            &comparison
                .output_dimensions_match
                .map(|value| value.to_string()),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "mapping_strategy_changed",
            &Some(comparison.colorspace_mapping_strategy_changed.to_string()),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "raw_base_proxy_confidence_delta",
            &format_f64(comparison.raw_base_proxy_confidence_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "raw_base_support_fraction_delta",
            &format_f64(comparison.raw_base_support_fraction_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "calibration_scanner_profile_status_changed",
            &Some(
                comparison
                    .calibration_scanner_profile_status_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "calibration_scanner_profile_id_changed",
            &Some(
                comparison
                    .calibration_scanner_profile_id_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "calibration_roll_profile_status_changed",
            &Some(
                comparison
                    .calibration_roll_profile_status_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "calibration_roll_profile_id_changed",
            &Some(comparison.calibration_roll_profile_id_changed.to_string()),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "calibration_confidence_delta",
            &format_f64(comparison.calibration_confidence_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "calibration_matrix_condition_number_delta",
            &format_f64(comparison.calibration_matrix_condition_number_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "calibration_requested_film_stock_changed",
            &Some(
                comparison
                    .calibration_requested_film_stock_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "calibration_film_stock_status_changed",
            &Some(comparison.calibration_film_stock_status_changed.to_string()),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "calibration_film_stock_matched_roll_profiles_changed",
            &Some(
                comparison
                    .calibration_film_stock_matched_roll_profiles_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "calibration_rejection_details_changed",
            &Some(comparison.calibration_rejection_details_changed.to_string()),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "candidate_acceptance_changed",
            &Some(
                comparison
                    .colorspace_candidate_acceptance_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "selected_quality_score_delta",
            &format_f64(comparison.colorspace_selected_quality_score_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "technical_safety_score_delta",
            &format_f64(comparison.colorspace_technical_safety_score_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "color_fidelity_score_delta",
            &format_f64(comparison.colorspace_color_fidelity_score_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "runner_up_quality_delta_delta",
            &format_f64(comparison.colorspace_selected_runner_up_quality_delta_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "reference_patch_selected_rms_delta_e_delta",
            &format_f64(comparison.colorspace_reference_patch_selected_rms_delta_e_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "reference_patch_selected_rms_delta_e2000_delta",
            &format_f64(comparison.colorspace_reference_patch_selected_rms_delta_e2000_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "reference_patch_xyz_max_vs_image_delta",
            &format_f64(
                comparison.colorspace_reference_patch_max_error_delta_vs_image_derived_delta,
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "reference_patch_delta_e_vs_image_delta",
            &format_f64(
                comparison.colorspace_reference_patch_delta_e_rms_delta_vs_image_derived_delta,
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "reference_patch_delta_e_max_vs_image_delta",
            &format_f64(
                comparison.colorspace_reference_patch_delta_e_max_delta_vs_image_derived_delta,
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "reference_patch_delta_e2000_vs_image_delta",
            &format_f64(
                comparison.colorspace_reference_patch_delta_e2000_rms_delta_vs_image_derived_delta,
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "reference_patch_delta_e2000_max_vs_image_delta",
            &format_f64(
                comparison.colorspace_reference_patch_delta_e2000_max_delta_vs_image_derived_delta,
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "reference_patch_regression_changed",
            &Some(
                comparison
                    .colorspace_reference_patch_selected_regresses_image_derived_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "reference_patch_hue_regressions_changed",
            &Some(
                comparison
                    .colorspace_reference_patch_hue_family_regressions_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "neutral_estimate_score_delta",
            &format_f64(comparison.colorspace_neutral_estimate_score_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "neutral_estimate_accepted_changed",
            &Some(
                comparison
                    .colorspace_neutral_estimate_accepted_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "dominant_anchor_score_delta",
            &format_f64(comparison.colorspace_dominant_anchor_score_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "dominant_anchor_accepted_changed",
            &Some(
                comparison
                    .colorspace_dominant_anchor_accepted_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "channel_anchor_low_support_changed",
            &Some(
                comparison
                    .colorspace_channel_anchor_low_support_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "weak_anchor_fallback_changed",
            &Some(
                comparison
                    .colorspace_weak_anchor_fallback_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "gamut_fallback_changed",
            &Some(comparison.colorspace_gamut_fallback_changed.to_string()),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "neutral_trim_applied_changed",
            &Some(
                comparison
                    .colorspace_neutral_trim_applied_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "debug_artifact_invalid_count",
            &Some(
                comparison
                    .colorspace_debug_artifact_invalid_count
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "debug_artifact_issues",
            &Some(comparison.colorspace_debug_artifact_issues.join(", ")),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "post_scale_preserved_ratio_delta",
            &format_f64(comparison.colorspace_post_scale_preserved_ratio_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "luma_residual_p95_ratio",
            &format_f64(comparison.luma_residual_p95_ratio),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "chroma_residual_p95_ratio",
            &format_f64(comparison.chroma_residual_p95_ratio),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "grain_detail_review_required_changed",
            &Some(comparison.grain_detail_review_required_changed.to_string()),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "grain_detail_decision_supported_changed",
            &Some(
                comparison
                    .grain_detail_decision_supported_changed
                    .to_string(),
            ),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "grain_detail_luminance_p10_retention_delta",
            &format_f64(comparison.grain_detail_luminance_p10_retention_delta),
        );
        push_row(
            &mut out,
            "summary_baseline",
            "grain_detail_chroma_p10_retention_delta",
            &format_f64(comparison.grain_detail_chroma_p10_retention_delta),
        );
    }
    if let Some(comparison) = &summary.comparison {
        push_row(
            &mut out,
            "comparison",
            "status",
            &Some(comparison.status.clone()),
        );
        push_row(
            &mut out,
            "comparison",
            "issues",
            &Some(comparison.issues.join(", ")),
        );
        push_row(
            &mut out,
            "comparison",
            "baseline_report_path",
            &Some(comparison.baseline_report_path.clone()),
        );
        push_row(
            &mut out,
            "comparison",
            "output_dimensions_match",
            &comparison
                .output_dimensions_match
                .map(|value| value.to_string()),
        );
        push_row(
            &mut out,
            "comparison",
            "stitch_decision_changed",
            &Some(comparison.stitch_decision_changed.to_string()),
        );
        push_row(
            &mut out,
            "comparison",
            "render_input_source_changed",
            &Some(comparison.render_input_source_changed.to_string()),
        );
        push_row(
            &mut out,
            "comparison",
            "calibration_status_changed",
            &Some(comparison.calibration_status_changed.to_string()),
        );
        push_row(
            &mut out,
            "comparison",
            "post_scale_preserved_ratio_delta",
            &format_f64(comparison.colorspace_post_scale_preserved_ratio_delta),
        );
        push_row(
            &mut out,
            "comparison",
            "selected_quality_score_delta",
            &format_f64(comparison.colorspace_selected_quality_score_delta),
        );
        push_row(
            &mut out,
            "comparison",
            "reference_patch_rms_delta_e2000_delta",
            &format_f64(comparison.colorspace_reference_patch_rms_delta_e2000_delta),
        );
        push_row(
            &mut out,
            "comparison",
            "luma_residual_p95_ratio",
            &format_f64(comparison.luma_residual_p95_ratio),
        );
        push_row(
            &mut out,
            "comparison",
            "chroma_residual_p95_ratio",
            &format_f64(comparison.chroma_residual_p95_ratio),
        );
    }

    if !summary.warnings.is_empty() {
        out.push_str("\n## Warnings\n\n");
        for phase in &summary.warnings {
            for warning in &phase.warnings {
                out.push_str(&format!("- `{}`: {}\n", phase.phase, warning));
            }
        }
    }

    out
}

fn summarize_input_orientation(phase: Option<&PhaseReport>) -> InputOrientationValidationSummary {
    let input_count = phase.and_then(|phase| usize_metric(phase, "input_count"));
    let components = phase
        .and_then(|phase| phase.metrics.get("components"))
        .and_then(serde_json::Value::as_array)
        .map(|components| {
            components
                .iter()
                .map(|component| {
                    let decode = component.get("decode");
                    let orientation = component
                        .get("decode")
                        .and_then(|decode| decode.get("orientation"));
                    let correction = decode.and_then(|decode| decode.get("orientation_correction"));
                    InputOrientationComponentValidationSummary {
                        index: usize_value(component.get("index")),
                        decoded_pixel_sha256: component
                            .get("decode")
                            .and_then(|decode| string_value(decode.get("decoded_pixel_sha256"))),
                        source_orientation_materialized_decoded_pixel_sha256: decode.and_then(
                            |decode| {
                                string_value(
                                    decode.get(
                                        "source_orientation_materialized_decoded_pixel_sha256",
                                    ),
                                )
                            },
                        ),
                        tag_value: orientation
                            .and_then(|orientation| usize_value(orientation.get("tag_value")))
                            .and_then(|value| u16::try_from(value).ok()),
                        metadata_transform: orientation
                            .and_then(|orientation| string_value(orientation.get("transform"))),
                        metadata_applied: orientation
                            .and_then(|orientation| orientation.get("applied"))
                            .and_then(serde_json::Value::as_bool),
                        orientation_correction_requested: correction
                            .and_then(|value| string_value(value.get("requested"))),
                        orientation_correction_transform: correction
                            .and_then(|value| string_value(value.get("transform"))),
                        orientation_correction_applied: correction
                            .and_then(|value| value.get("applied"))
                            .and_then(serde_json::Value::as_bool),
                        effective_tag_value: correction
                            .and_then(|value| usize_value(value.get("effective_tag_value")))
                            .and_then(|value| u16::try_from(value).ok()),
                        transform: correction
                            .and_then(|value| string_value(value.get("effective_transform")))
                            .or_else(|| {
                                orientation.and_then(|orientation| {
                                    string_value(orientation.get("transform"))
                                })
                            }),
                        applied: match (
                            orientation
                                .and_then(|orientation| orientation.get("applied"))
                                .and_then(serde_json::Value::as_bool),
                            correction
                                .and_then(|value| value.get("applied"))
                                .and_then(serde_json::Value::as_bool),
                        ) {
                            (Some(metadata), Some(correction)) => Some(metadata || correction),
                            (metadata, correction) => metadata.or(correction),
                        },
                        source_width: orientation
                            .and_then(|orientation| usize_value(orientation.get("source_width"))),
                        source_height: orientation
                            .and_then(|orientation| usize_value(orientation.get("source_height"))),
                        output_width: correction
                            .and_then(|value| usize_value(value.get("output_width")))
                            .or_else(|| {
                                orientation.and_then(|orientation| {
                                    usize_value(orientation.get("output_width"))
                                })
                            }),
                        output_height: correction
                            .and_then(|value| usize_value(value.get("output_height")))
                            .or_else(|| {
                                orientation.and_then(|orientation| {
                                    usize_value(orientation.get("output_height"))
                                })
                            }),
                    }
                })
                .collect::<Vec<InputOrientationComponentValidationSummary>>()
        })
        .unwrap_or_default();
    let component_count = components.len();
    InputOrientationValidationSummary {
        input_count,
        component_count,
        all_components_reported: input_count.map(|count| count == component_count),
        components,
    }
}

fn summarize_deskew(phase: Option<&PhaseReport>) -> DeskewValidationSummary {
    let input_count = phase.and_then(|phase| usize_metric(phase, "input_count"));
    let components = phase
        .and_then(|phase| phase.metrics.get("components"))
        .and_then(serde_json::Value::as_array);
    let component_count = components.map_or(0, Vec::len);
    let applied_component_count =
        phase.and_then(|phase| usize_metric(phase, "applied_component_count"));
    let minimum_component_retained_area_ratio = components.and_then(|components| {
        components
            .iter()
            .filter_map(|component| {
                component
                    .get("diagnostics")
                    .and_then(|diagnostics| f64_value(diagnostics.get("retained_area_ratio")))
            })
            .reduce(f64::min)
    });
    DeskewValidationSummary {
        requested_mode: phase.and_then(|phase| string_metric(phase, "requested_mode")),
        status: phase.and_then(|phase| string_metric(phase, "status")),
        applied: phase
            .and_then(|phase| phase.metrics.get("applied"))
            .and_then(serde_json::Value::as_bool),
        applied_component_count,
        input_count,
        component_count,
        all_components_reported: input_count.map(|count| count == component_count),
        all_components_applied: input_count
            .zip(applied_component_count)
            .map(|(input_count, applied_count)| input_count > 0 && applied_count == input_count),
        minimum_component_retained_area_ratio,
        detected_source_skew_degrees: phase
            .and_then(|phase| f64_metric(phase, "detected_source_skew_degrees")),
        correction_degrees: phase.and_then(|phase| f64_metric(phase, "correction_degrees")),
        confidence: phase.map(|phase| phase.confidence),
        review_required: phase
            .and_then(|phase| phase.metrics.get("review_required"))
            .and_then(serde_json::Value::as_bool),
        review_reason: phase.and_then(|phase| string_metric(phase, "review_reason")),
        retained_area_ratio: phase.and_then(|phase| f64_metric(phase, "retained_area_ratio")),
        proposed_retained_area_ratio: phase
            .and_then(|phase| f64_metric(phase, "proposed_retained_area_ratio")),
        supporting_side_count: phase.and_then(|phase| usize_metric(phase, "supporting_side_count")),
        horizontal_side_count: phase.and_then(|phase| usize_metric(phase, "horizontal_side_count")),
        vertical_side_count: phase.and_then(|phase| usize_metric(phase, "vertical_side_count")),
        side_angle_spread_degrees: phase
            .and_then(|phase| f64_metric(phase, "side_angle_spread_degrees")),
        interpolation: phase.and_then(|phase| string_metric(phase, "interpolation")),
        reason: phase.and_then(|phase| string_metric(phase, "reason")),
    }
}

fn summarize_border_crop(phase: Option<&PhaseReport>) -> BorderCropValidationSummary {
    let input_count = phase.and_then(|phase| usize_metric(phase, "input_count"));
    let components = phase
        .and_then(|phase| phase.metrics.get("components"))
        .and_then(serde_json::Value::as_array)
        .map(|components| {
            components
                .iter()
                .map(|component| {
                    let crop = component.get("crop");
                    let top_removed = crop.and_then(|crop| usize_value(crop.get("top_removed")));
                    let bottom_removed =
                        crop.and_then(|crop| usize_value(crop.get("bottom_removed")));
                    let left_removed = crop.and_then(|crop| usize_value(crop.get("left_removed")));
                    let right_removed =
                        crop.and_then(|crop| usize_value(crop.get("right_removed")));
                    let edges = [top_removed, bottom_removed, left_removed, right_removed];
                    let removed_edge_count = edges
                        .iter()
                        .copied()
                        .collect::<Option<Vec<_>>>()
                        .map(|edges| edges.into_iter().filter(|removed| *removed > 0).count());
                    let output_shape = component
                        .get("output_shape")
                        .and_then(serde_json::Value::as_array);
                    let output_height = output_shape
                        .and_then(|shape| shape.first())
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|value| usize::try_from(value).ok());
                    let output_width = output_shape
                        .and_then(|shape| shape.get(1))
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|value| usize::try_from(value).ok());
                    let input_height = output_height.zip(top_removed).zip(bottom_removed).and_then(
                        |((height, top), bottom)| height.checked_add(top)?.checked_add(bottom),
                    );
                    let input_width = output_width.zip(left_removed).zip(right_removed).and_then(
                        |((width, left), right)| width.checked_add(left)?.checked_add(right),
                    );
                    let retained_area_ratio = output_width
                        .zip(output_height)
                        .zip(input_width.zip(input_height))
                        .and_then(
                            |((output_width, output_height), (input_width, input_height))| {
                                let input_area = input_width.checked_mul(input_height)?;
                                let output_area = output_width.checked_mul(output_height)?;
                                (input_area > 0).then_some(output_area as f64 / input_area as f64)
                            },
                        );
                    BorderCropComponentValidationSummary {
                        index: usize_value(component.get("index")),
                        top_removed,
                        bottom_removed,
                        left_removed,
                        right_removed,
                        removed_edge_count,
                        dead_zone_detected: crop
                            .and_then(|crop| crop.get("dead_zone_detected"))
                            .and_then(serde_json::Value::as_bool),
                        input_width,
                        input_height,
                        output_width,
                        output_height,
                        retained_area_ratio,
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let component_count = components.len();
    let cropped_component_count = components
        .iter()
        .filter(|component| component.dead_zone_detected == Some(true))
        .count();
    let total_removed_edge_count = components
        .iter()
        .filter_map(|component| component.removed_edge_count)
        .sum();
    let minimum_removed_edge_count_per_component = components
        .iter()
        .filter_map(|component| component.removed_edge_count)
        .min();
    let minimum_retained_area_ratio = components
        .iter()
        .filter_map(|component| component.retained_area_ratio)
        .reduce(f64::min);
    let maximum_retained_area_ratio = components
        .iter()
        .filter_map(|component| component.retained_area_ratio)
        .reduce(f64::max);
    let rejected_crop_warning_count = phase.map_or(0, |phase| {
        phase
            .warnings
            .iter()
            .filter(|warning| warning.contains("crop rejected"))
            .count()
    });
    BorderCropValidationSummary {
        input_count,
        component_count,
        all_components_reported: input_count.map(|count| count == component_count),
        cropped_component_count,
        all_components_cropped: input_count
            .map(|count| count > 0 && count == component_count && cropped_component_count == count),
        total_removed_edge_count,
        minimum_removed_edge_count_per_component,
        minimum_retained_area_ratio,
        maximum_retained_area_ratio,
        rejected_crop_warning_count,
        components,
    }
}

fn summarize_negative_reconstruction(
    density_phase: Option<&PhaseReport>,
    colorspace_phase: Option<&PhaseReport>,
) -> NegativeReconstructionValidationSummary {
    let response =
        density_phase.and_then(|phase| phase.metrics.get("direct_density_response_model"));
    let reconstruction =
        colorspace_phase.and_then(|phase| phase.metrics.get("negative_response_reconstruction"));
    let held_out_delta_e00_rms =
        response.and_then(|response| f64_value(response.get("held_out_delta_e00_rms")));
    let unit_slope_delta_e00_rms =
        response.and_then(|response| f64_value(response.get("unit_slope_delta_e00_rms")));
    NegativeReconstructionValidationSummary {
        input_mode: density_phase.and_then(|phase| string_metric(phase, "input_mode")),
        density_inversion_skipped: density_phase
            .and_then(|phase| phase.metrics.get("skipped"))
            .and_then(serde_json::Value::as_bool),
        density_confidence: density_phase.map(|phase| phase.confidence),
        response_model: response.and_then(|response| string_value(response.get("model"))),
        response_source: response.and_then(|response| string_value(response.get("source"))),
        response_accepted: response
            .and_then(|response| response.get("accepted"))
            .and_then(serde_json::Value::as_bool),
        response_model_review_required: response
            .and_then(|response| response.get("review_required"))
            .and_then(serde_json::Value::as_bool),
        crosstalk_model: response
            .and_then(|response| string_value(response.get("crosstalk_model"))),
        characteristic_curve_model: response
            .and_then(|response| string_value(response.get("characteristic_curve_model"))),
        measured_model_id: response
            .and_then(|response| string_value(response.get("measured_model_id"))),
        measured_confidence: response
            .and_then(|response| f64_value(response.get("measured_confidence"))),
        held_out_delta_e00_rms,
        held_out_delta_e00_max: response
            .and_then(|response| f64_value(response.get("held_out_delta_e00_max"))),
        unit_slope_delta_e00_rms,
        held_out_improvement_over_unit_slope: unit_slope_delta_e00_rms
            .zip(held_out_delta_e00_rms)
            .map(|(baseline, measured)| baseline - measured),
        maximum_density_noise_gain: response
            .and_then(|response| f64_value(response.get("maximum_density_noise_gain"))),
        reconstruction_review_required: colorspace_phase
            .and_then(|phase| phase.metrics.get("negative_response_review_required"))
            .and_then(serde_json::Value::as_bool),
        signed_headroom_preserved: reconstruction
            .and_then(|reconstruction| reconstruction.get("signed_headroom_preserved"))
            .and_then(serde_json::Value::as_bool),
        curve_extrapolated_any_ratio: reconstruction.and_then(|reconstruction| {
            f64_value(reconstruction.get("curve_extrapolated_any_ratio"))
        }),
        curve_interpolation: reconstruction
            .and_then(|reconstruction| string_value(reconstruction.get("curve_interpolation"))),
    }
}

fn summarize_white_balance(
    phase: Option<&PhaseReport>,
    tone_phase: Option<&PhaseReport>,
) -> WhiteBalanceValidationSummary {
    let technical = phase.and_then(|phase| phase.metrics.get("technical"));
    let creative = tone_phase.and_then(|phase| phase.metrics.get("creative_white_balance"));
    WhiteBalanceValidationSummary {
        technical_requested_mode: technical
            .and_then(|value| string_value(value.get("requested_mode"))),
        technical_status: technical.and_then(|value| string_value(value.get("status"))),
        technical_source: technical.and_then(|value| string_value(value.get("source"))),
        technical_reason: technical.and_then(|value| string_value(value.get("reason"))),
        technical_applied: technical
            .and_then(|value| value.get("applied"))
            .and_then(serde_json::Value::as_bool),
        technical_confidence: technical.and_then(|value| f64_value(value.get("confidence"))),
        technical_review_required: technical
            .and_then(|value| value.get("review_required"))
            .and_then(serde_json::Value::as_bool),
        technical_sample_count: technical.and_then(|value| usize_value(value.get("sample_count"))),
        technical_occupied_spatial_bin_count: technical
            .and_then(|value| usize_value(value.get("occupied_spatial_bin_count"))),
        technical_populated_luminance_band_count: technical
            .and_then(|value| usize_value(value.get("populated_luminance_band_count"))),
        technical_estimated_source_cct_kelvin: technical
            .and_then(|value| f64_value(value.get("estimated_source_cct_kelvin"))),
        technical_pre_neutral_log_chroma: technical
            .and_then(|value| f64_value(value.get("pre_adaptation_neutral_log_chroma"))),
        technical_post_neutral_log_chroma: technical
            .and_then(|value| f64_value(value.get("post_adaptation_neutral_log_chroma"))),
        creative_applied: creative
            .and_then(|value| value.get("applied"))
            .and_then(serde_json::Value::as_bool),
        creative_temperature: creative.and_then(|value| f64_value(value.get("temperature"))),
        creative_tint: creative.and_then(|value| f64_value(value.get("tint"))),
        creative_target_temperature_kelvin: creative
            .and_then(|value| f64_value(value.get("target_temperature_kelvin"))),
        creative_separated_from_technical_master: creative
            .and_then(|value| value.get("separated_from_technical_master"))
            .and_then(serde_json::Value::as_bool),
    }
}

fn summarize_stitch(phase: Option<&PhaseReport>) -> StitchValidationSummary {
    let Some(phase) = phase else {
        return StitchValidationSummary {
            decision: None,
            inferred_order: Vec::new(),
            confidence: None,
            chosen_hypothesis: None,
            rejection_reason: None,
            search_selection_reason: None,
            evaluated_candidate_count: None,
            max_overlap_considered: None,
            top_candidates: Vec::new(),
            evidence_score: None,
            prior_weight: None,
            overlap_support_score: None,
            vertical_offset_plausibility_score: None,
            local_consistency_score: None,
            plausibility_score: None,
            homography_feature_validation: None,
            homography_spatial_validation: None,
            seam_exposure_correction: None,
            seam_blend: None,
        };
    };

    let selected = selected_hypothesis(phase);
    let selected_search = selected.and_then(|hyp| hyp.get("search"));
    let selected_validation = selected.and_then(|hyp| hyp.get("validation"));
    let top_candidates = selected_search
        .and_then(|search| search.get("top_candidates"))
        .and_then(|value| value.as_array())
        .map(|candidates| {
            candidates
                .iter()
                .take(3)
                .map(|candidate| StitchCandidateSummary {
                    rank: usize_value(candidate.get("rank")),
                    overlap_width: usize_value(candidate.get("overlap_width")),
                    vertical_offset: i64_value(candidate.get("vertical_offset")),
                    correspondence_score: f64_value(candidate.get("correspondence_score")),
                    prior_weight: f64_value(candidate.get("prior_weight")),
                    search_score: f64_value(candidate.get("search_score")),
                })
                .collect()
        })
        .unwrap_or_default();

    StitchValidationSummary {
        decision: string_metric(phase, "decision"),
        inferred_order: phase
            .metrics
            .get("ordering")
            .and_then(|ordering| ordering.get("inferred_order"))
            .or_else(|| phase.metrics.get("inferred_order"))
            .and_then(serde_json::Value::as_array)
            .map(|order| {
                order
                    .iter()
                    .filter_map(|value| value.as_u64())
                    .filter_map(|value| usize::try_from(value).ok())
                    .collect()
            })
            .unwrap_or_default(),
        confidence: Some(phase.confidence),
        chosen_hypothesis: string_metric(phase, "chosen_hypothesis"),
        rejection_reason: string_metric(phase, "rejection_reason"),
        search_selection_reason: selected_search
            .and_then(|search| string_value(search.get("selection_reason"))),
        evaluated_candidate_count: selected_search
            .and_then(|search| search.get("evaluated_candidates"))
            .and_then(|value| value.as_array())
            .map(Vec::len),
        max_overlap_considered: selected_search
            .and_then(|search| usize_value(search.get("max_overlap_considered"))),
        top_candidates,
        evidence_score: selected_validation.and_then(|v| f64_value(v.get("evidence_score"))),
        prior_weight: selected_validation.and_then(|v| f64_value(v.get("prior_weight"))),
        overlap_support_score: selected_validation
            .and_then(|v| f64_value(v.get("overlap_support_score"))),
        vertical_offset_plausibility_score: selected_validation
            .and_then(|v| f64_value(v.get("vertical_offset_plausibility_score"))),
        local_consistency_score: selected_validation
            .and_then(|v| f64_value(v.get("local_consistency_score"))),
        plausibility_score: selected_validation
            .and_then(|v| f64_value(v.get("plausibility_score"))),
        homography_feature_validation: summarize_homography_feature_validations(phase),
        homography_spatial_validation: summarize_homography_spatial_validations(phase),
        seam_exposure_correction: summarize_stitch_seam_exposure_corrections(phase),
        seam_blend: summarize_stitch_seam_blends(phase),
    }
}

fn append_homography_feature_validations<'a>(
    metrics: &'a serde_json::Value,
    values: &mut Vec<&'a serde_json::Value>,
) {
    if let Some(value) = metrics
        .get("homography_feature_validation")
        .filter(|value| value.is_object())
    {
        values.push(value);
        return;
    }
    if let Some(hypotheses) = metrics.get("hypotheses").and_then(|value| value.as_array()) {
        values.extend(hypotheses.iter().filter_map(|hypothesis| {
            hypothesis
                .get("validation")
                .and_then(|validation| validation.get("homography_feature_validation"))
                .filter(|value| value.is_object())
        }));
    }
}

fn summarize_homography_feature_validations(
    phase: &PhaseReport,
) -> Option<HomographyFeatureValidationSummary> {
    let mut values = Vec::<&serde_json::Value>::new();
    append_homography_feature_validations(&phase.metrics, &mut values);
    if let Some(merges) = phase
        .metrics
        .get("pair_merges")
        .and_then(|value| value.as_array())
    {
        for merge in merges {
            if let Some(metrics) = merge
                .get("pair_report")
                .and_then(|report| report.get("metrics"))
            {
                append_homography_feature_validations(metrics, &mut values);
            }
        }
    }
    if values.is_empty() {
        return None;
    }

    let acceptance = values
        .iter()
        .filter_map(|value| value.get("accepted").and_then(serde_json::Value::as_bool))
        .collect::<Vec<_>>();
    let accepted_count = acceptance.iter().filter(|accepted| **accepted).count();
    let common_string = |key: &str| {
        let strings = values
            .iter()
            .filter_map(|value| string_value(value.get(key)))
            .collect::<Vec<_>>();
        strings.first().cloned().map(|first| {
            if strings.iter().all(|value| value == &first) {
                first
            } else {
                "sequence_mixed".to_string()
            }
        })
    };
    let minimum_usize = |key: &str| {
        values
            .iter()
            .filter_map(|value| usize_value(value.get(key)))
            .min()
    };
    let minimum_f64 = |key: &str| {
        values
            .iter()
            .filter_map(|value| f64_value(value.get(key)))
            .filter(|value| value.is_finite())
            .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
    };
    let maximum_f64 = |key: &str| {
        values
            .iter()
            .filter_map(|value| f64_value(value.get(key)))
            .filter(|value| value.is_finite())
            .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
    };
    let mut reasons = values
        .iter()
        .filter_map(|value| string_value(value.get("reason")))
        .collect::<Vec<_>>();
    reasons.sort();
    reasons.dedup();

    Some(HomographyFeatureValidationSummary {
        accepted: (acceptance.len() == values.len()).then_some(accepted_count == values.len()),
        validation_count: values.len(),
        accepted_count,
        partition_method: common_string("partition_method"),
        minimum_training_match_count: minimum_usize("training_match_count"),
        minimum_held_out_match_count: minimum_usize("held_out_match_count"),
        minimum_training_spatial_cell_count: minimum_usize("training_spatial_cell_count"),
        minimum_held_out_spatial_cell_count: minimum_usize("held_out_spatial_cell_count"),
        minimum_training_inlier_ratio: minimum_f64("training_inlier_ratio"),
        minimum_held_out_inlier_ratio: minimum_f64("held_out_inlier_ratio"),
        minimum_reverse_validation_inlier_ratio: minimum_f64("reverse_validation_inlier_ratio"),
        maximum_held_out_p95_error_px: maximum_f64("held_out_p95_error_px"),
        maximum_reverse_validation_p95_error_px: maximum_f64("reverse_validation_p95_error_px"),
        maximum_cross_fit_disagreement_px: maximum_f64("cross_fit_max_disagreement_px"),
        reasons,
    })
}

fn summarize_homography_spatial_validations(
    phase: &PhaseReport,
) -> Option<HomographySpatialValidationSummary> {
    let mut values = Vec::<&serde_json::Value>::new();
    if let Some(value) = phase
        .metrics
        .get("homography_spatial_validation")
        .filter(|value| value.is_object())
    {
        values.push(value);
    }
    if let Some(merges) = phase
        .metrics
        .get("pair_merges")
        .and_then(|value| value.as_array())
    {
        for merge in merges {
            if let Some(value) = merge
                .get("pair_report")
                .and_then(|report| report.get("metrics"))
                .and_then(|metrics| metrics.get("homography_spatial_validation"))
                .filter(|value| value.is_object())
            {
                values.push(value);
            }
        }
    }
    if values.is_empty() {
        return None;
    }

    let common_string = |key: &str| {
        let strings = values
            .iter()
            .filter_map(|value| string_value(value.get(key)))
            .collect::<Vec<_>>();
        strings.first().cloned().map(|first| {
            if strings.iter().all(|value| value == &first) {
                first
            } else {
                "sequence_mixed".to_string()
            }
        })
    };
    let minimum_scalar_f64 = |key: &str| {
        values
            .iter()
            .filter_map(|value| f64_value(value.get(key)))
            .filter(|value| value.is_finite())
            .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
    };
    let minimum_array_f64 = |key: &str| {
        values
            .iter()
            .filter_map(|value| value.get(key).and_then(serde_json::Value::as_array))
            .flat_map(|array| array.iter())
            .filter_map(|value| value.as_f64())
            .filter(|value| value.is_finite())
            .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
    };
    let minimum_array_usize = |key: &str| {
        values
            .iter()
            .filter_map(|value| value.get(key).and_then(serde_json::Value::as_array))
            .flat_map(|array| array.iter())
            .filter_map(|value| value.as_u64())
            .filter_map(|value| usize::try_from(value).ok())
            .min()
    };
    let acceptance = values
        .iter()
        .filter_map(|value| value.get("accepted").and_then(serde_json::Value::as_bool))
        .collect::<Vec<_>>();
    let accepted_count = acceptance.iter().filter(|accepted| **accepted).count();
    let mut reasons = values
        .iter()
        .filter_map(|value| string_value(value.get("reason")))
        .collect::<Vec<_>>();
    reasons.sort();
    reasons.dedup();

    Some(HomographySpatialValidationSummary {
        accepted: (acceptance.len() == values.len()).then_some(accepted_count == values.len()),
        validation_count: values.len(),
        accepted_count,
        method: common_string("method"),
        minimum_split_ncc_improvement: minimum_array_f64("ncc_improvement"),
        minimum_mean_ncc_improvement: minimum_scalar_f64("mean_ncc_improvement"),
        minimum_split_registration_error_reduction: minimum_array_f64(
            "registration_error_reduction",
        ),
        minimum_mean_registration_error_reduction: minimum_scalar_f64(
            "mean_registration_error_reduction",
        ),
        minimum_split_sample_count: minimum_array_usize("sample_count"),
        minimum_model_deviation_from_translation_px: minimum_scalar_f64(
            "maximum_deviation_from_translation_px",
        ),
        model_selection_limit: common_string("model_selection_limit"),
        reasons,
    })
}

fn summarize_stitch_seam_detail_consistency(
    blends: &[&serde_json::Value],
) -> Option<SeamDetailConsistencyValidationSummary> {
    let values = blends
        .iter()
        .filter_map(|blend| blend.get("detail_consistency"))
        .filter(|value| value.is_object())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    let common_string = |key: &str| {
        let strings = values
            .iter()
            .filter_map(|value| string_value(value.get(key)))
            .collect::<Vec<_>>();
        strings.first().cloned().map(|first| {
            if strings.iter().all(|value| value == &first) {
                first
            } else {
                "sequence_mixed".to_string()
            }
        })
    };
    let min_usize = |key: &str| {
        values
            .iter()
            .filter_map(|value| usize_value(value.get(key)))
            .min()
    };
    let max_usize = |key: &str| {
        values
            .iter()
            .filter_map(|value| usize_value(value.get(key)))
            .max()
    };
    let max_f64 = |key: &str| {
        values
            .iter()
            .filter_map(|value| f64_value(value.get(key)))
            .filter(|value| value.is_finite())
            .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
    };
    let evaluated = values
        .iter()
        .filter_map(|value| value.get("evaluated").and_then(serde_json::Value::as_bool))
        .collect::<Vec<_>>();
    let evaluated_merge_count = evaluated.iter().filter(|value| **value).count();
    let decision_support = values
        .iter()
        .filter_map(|value| {
            value
                .get("decision_supported")
                .and_then(serde_json::Value::as_bool)
        })
        .collect::<Vec<_>>();
    let decision_supported_merge_count = decision_support.iter().filter(|value| **value).count();
    let review = values
        .iter()
        .filter_map(|value| {
            value
                .get("review_required")
                .and_then(serde_json::Value::as_bool)
        })
        .collect::<Vec<_>>();
    let review_required_merge_count = review.iter().filter(|value| **value).count();
    let mut reasons = values
        .iter()
        .filter_map(|value| string_value(value.get("reason")))
        .collect::<Vec<_>>();
    reasons.sort();
    reasons.dedup();
    let mut review_reasons = values
        .iter()
        .filter_map(|value| string_value(value.get("review_reason")))
        .collect::<Vec<_>>();
    review_reasons.sort();
    review_reasons.dedup();
    Some(SeamDetailConsistencyValidationSummary {
        method: common_string("method"),
        evaluated: (evaluated.len() == values.len())
            .then_some(evaluated_merge_count == values.len()),
        merge_count: values.len(),
        evaluated_merge_count,
        decision_supported: (decision_support.len() == values.len())
            .then_some(decision_supported_merge_count == values.len()),
        decision_supported_merge_count,
        review_required: (review.len() == values.len()).then_some(review_required_merge_count > 0),
        review_required_merge_count,
        minimum_supported_scale_count: min_usize("supported_scale_count"),
        maximum_imbalanced_scale_count: max_usize("imbalanced_scale_count"),
        maximum_symmetric_energy_ratio: max_f64("maximum_symmetric_energy_ratio"),
        review_ratio_threshold: max_f64("review_ratio_threshold"),
        minimum_direction_consistency: max_f64("minimum_direction_consistency"),
        maximum_cross_split_ratio: max_f64("maximum_cross_split_ratio"),
        minimum_repeated_scale_count: max_usize("minimum_repeated_scale_count"),
        reasons,
        review_reasons,
    })
}

fn summarize_stitch_seam_exposure_corrections(
    phase: &PhaseReport,
) -> Option<SeamExposureCorrectionSummary> {
    let mut values = Vec::<&serde_json::Value>::new();
    if let Some(value) = phase
        .metrics
        .get("seam_exposure_correction")
        .filter(|value| value.is_object())
    {
        values.push(value);
    }
    if let Some(merges) = phase
        .metrics
        .get("pair_merges")
        .and_then(|value| value.as_array())
    {
        for merge in merges {
            if let Some(value) = merge
                .get("pair_report")
                .and_then(|report| report.get("metrics"))
                .and_then(|metrics| metrics.get("seam_exposure_correction"))
                .filter(|value| value.is_object())
            {
                values.push(value);
            }
        }
    }
    let first = *values.first()?;
    if values.len() == 1 {
        return Some(summarize_seam_exposure_correction(first));
    }

    let common_string = |key: &str| {
        let strings = values
            .iter()
            .filter_map(|value| string_value(value.get(key)))
            .collect::<Vec<_>>();
        strings.first().cloned().map(|first| {
            if strings.len() == values.len() && strings.iter().all(|value| value == &first) {
                first
            } else {
                "sequence_mixed".to_string()
            }
        })
    };
    let min_f64 = |key: &str| {
        let measured = values
            .iter()
            .filter_map(|value| f64_value(value.get(key)))
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        (measured.len() == values.len()).then(|| {
            measured
                .into_iter()
                .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
                .expect("sequence exposure aggregate is non-empty")
        })
    };
    let max_f64 = |key: &str| {
        let measured = values
            .iter()
            .filter_map(|value| f64_value(value.get(key)))
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        (measured.len() == values.len()).then(|| {
            measured
                .into_iter()
                .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
                .expect("sequence exposure aggregate is non-empty")
        })
    };
    let max_abs_f64 = |key: &str| {
        let measured = values
            .iter()
            .filter_map(|value| f64_value(value.get(key)))
            .filter(|value| value.is_finite())
            .map(f64::abs)
            .collect::<Vec<_>>();
        (measured.len() == values.len()).then(|| {
            measured
                .into_iter()
                .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
                .expect("sequence exposure aggregate is non-empty")
        })
    };
    let min_usize = |key: &str| {
        let measured = values
            .iter()
            .filter_map(|value| usize_value(value.get(key)))
            .collect::<Vec<_>>();
        (measured.len() == values.len()).then(|| {
            measured
                .into_iter()
                .min()
                .expect("sequence exposure aggregate is non-empty")
        })
    };
    let sum_usize = |key: &str| {
        let counts = values
            .iter()
            .filter_map(|value| usize_value(value.get(key)))
            .collect::<Vec<_>>();
        (counts.len() == values.len()).then(|| counts.into_iter().sum())
    };
    let max_abs_vec = |key: &str| {
        let vectors = values
            .iter()
            .filter_map(|value| value.get(key).and_then(f64_vec_value))
            .collect::<Vec<_>>();
        if vectors.len() != values.len() {
            return None;
        }
        let width = vectors.first()?.len();
        if width == 0 || vectors.iter().any(|vector| vector.len() != width) {
            return None;
        }
        let mut maxima = vec![0.0_f64; width];
        for vector in vectors {
            for (index, value) in vector.into_iter().enumerate() {
                maxima[index] = maxima[index].max(value.abs());
            }
        }
        Some(maxima)
    };

    let models = values
        .iter()
        .filter_map(|value| value.get("model").and_then(serde_json::Value::as_str))
        .collect::<Vec<_>>();
    let held_out_flags = values
        .iter()
        .filter_map(|value| {
            value
                .get("held_out_validation_passed")
                .and_then(serde_json::Value::as_bool)
        })
        .collect::<Vec<_>>();
    let model_appropriate_held_out =
        (models.len() == values.len() && held_out_flags.len() == values.len()).then(|| {
            let any_non_identity = models.iter().any(|model| *model != "identity");
            if !any_non_identity {
                false
            } else {
                models
                    .iter()
                    .zip(held_out_flags.iter())
                    .all(|(model, passed)| *model == "identity" || *passed)
            }
        });
    let applied = values
        .iter()
        .filter_map(|value| value.get("applied").and_then(serde_json::Value::as_bool))
        .collect::<Vec<_>>();

    Some(SeamExposureCorrectionSummary {
        mode: common_string("mode"),
        model: common_string("model"),
        applied: (applied.len() == values.len()).then(|| applied.iter().any(|value| *value)),
        reason: Some(format!(
            "worst-case aggregate across {} accepted sequence merges",
            values.len()
        )),
        sample_count: sum_usize("sample_count"),
        valid_sample_ratio: min_f64("valid_sample_ratio"),
        spatial_gain_log_slope_x_rgb: max_abs_vec("spatial_gain_log_slope_x_rgb"),
        spatial_gain_log_slope_x_luma: max_abs_f64("spatial_gain_log_slope_x_luma"),
        spatial_gain_log_slope_y_rgb: max_abs_vec("spatial_gain_log_slope_y_rgb"),
        spatial_gain_log_slope_y_luma: max_abs_f64("spatial_gain_log_slope_y_luma"),
        spatial_offset_slope_x_rgb_normalized: max_abs_vec("spatial_offset_slope_x_rgb_normalized"),
        spatial_offset_slope_y_rgb_normalized: max_abs_vec("spatial_offset_slope_y_rgb_normalized"),
        spatial_offset_top_rgb_normalized: max_abs_vec("spatial_offset_top_rgb_normalized"),
        spatial_offset_bottom_rgb_normalized: max_abs_vec("spatial_offset_bottom_rgb_normalized"),
        offset_rgb: max_abs_vec("offset_rgb"),
        offset_rgb_normalized: max_abs_vec("offset_rgb_normalized"),
        seam_score_before: max_f64("seam_score_before"),
        seam_score_after: max_f64("seam_score_after"),
        clipped_high_before: max_abs_vec("clipped_high_before"),
        clipped_high_after: max_abs_vec("clipped_high_after"),
        clipped_low_before: max_abs_vec("clipped_low_before"),
        clipped_low_after: max_abs_vec("clipped_low_after"),
        training_window_count: sum_usize("training_window_count"),
        held_out_window_count: sum_usize("held_out_window_count"),
        training_sample_count: sum_usize("training_sample_count"),
        held_out_sample_count: sum_usize("held_out_sample_count"),
        gain_offset_training_window_count: sum_usize("gain_offset_training_window_count"),
        gain_offset_held_out_window_count: sum_usize("gain_offset_held_out_window_count"),
        gain_offset_consistent_window_ratio: min_f64("gain_offset_consistent_window_ratio"),
        spatial_training_window_count: min_usize("spatial_training_window_count"),
        spatial_held_out_window_count: min_usize("spatial_held_out_window_count"),
        spatial_distinct_training_rows: min_usize("spatial_distinct_training_rows"),
        spatial_distinct_held_out_rows: min_usize("spatial_distinct_held_out_rows"),
        spatial_consistent_window_ratio: min_f64("spatial_consistent_window_ratio"),
        spatial_slope_agreement_ratio: min_f64("spatial_slope_agreement_ratio"),
        spatial_affine_training_window_count: min_usize("spatial_affine_training_window_count"),
        spatial_affine_held_out_window_count: min_usize("spatial_affine_held_out_window_count"),
        spatial_affine_distinct_training_rows: min_usize("spatial_affine_distinct_training_rows"),
        spatial_affine_distinct_held_out_rows: min_usize("spatial_affine_distinct_held_out_rows"),
        spatial_affine_consistent_window_ratio: min_f64("spatial_affine_consistent_window_ratio"),
        spatial_affine_slope_agreement_ratio: min_f64("spatial_affine_slope_agreement_ratio"),
        spatial_affine_center_offset_delta_normalized: max_f64(
            "spatial_affine_center_offset_delta_normalized",
        ),
        held_out_identity_seam_score: max_f64("held_out_identity_seam_score"),
        held_out_gain_seam_score: max_f64("held_out_gain_seam_score"),
        held_out_gain_offset_seam_score: max_f64("held_out_gain_offset_seam_score"),
        held_out_spatial_gain_seam_score: max_f64("held_out_spatial_gain_seam_score"),
        held_out_spatial_gain_offset_seam_score: max_f64("held_out_spatial_gain_offset_seam_score"),
        held_out_selected_seam_score: max_f64("held_out_selected_seam_score"),
        held_out_improvement_over_identity: min_f64("held_out_improvement_over_identity"),
        held_out_improvement_over_gain: min_f64("held_out_improvement_over_gain"),
        held_out_spatial_improvement_over_best_constant: min_f64(
            "held_out_spatial_improvement_over_best_constant",
        ),
        held_out_spatial_gain_offset_improvement_over_best_simpler: min_f64(
            "held_out_spatial_gain_offset_improvement_over_best_simpler",
        ),
        held_out_validation_passed: model_appropriate_held_out,
        ..SeamExposureCorrectionSummary::default()
    })
}

fn summarize_stitch_seam_blends(phase: &PhaseReport) -> Option<SeamBlendValidationSummary> {
    let mut values = Vec::<&serde_json::Value>::new();
    if let Some(value) = phase
        .metrics
        .get("seam_blend")
        .filter(|value| value.is_object())
    {
        values.push(value);
    }
    if let Some(merges) = phase
        .metrics
        .get("pair_merges")
        .and_then(|value| value.as_array())
    {
        for merge in merges {
            if let Some(value) = merge
                .get("pair_report")
                .and_then(|report| report.get("metrics"))
                .and_then(|metrics| metrics.get("seam_blend"))
                .filter(|value| value.is_object())
            {
                values.push(value);
            }
        }
    }
    if values.is_empty() {
        return None;
    }

    let modes = values
        .iter()
        .filter_map(|value| string_value(value.get("mode")))
        .collect::<Vec<_>>();
    let mode = modes.first().cloned().map(|first| {
        if modes.iter().all(|value| value == &first) {
            first
        } else {
            "sequence_mixed".to_string()
        }
    });
    let applied_merge_count = values
        .iter()
        .filter(|value| value.get("applied").and_then(serde_json::Value::as_bool) == Some(true))
        .count();
    let review = values
        .iter()
        .filter_map(|value| {
            value
                .get("review_required")
                .and_then(serde_json::Value::as_bool)
        })
        .collect::<Vec<_>>();
    let review_required_merge_count = review.iter().filter(|value| **value).count();
    let mut review_reasons = values
        .iter()
        .filter_map(|value| string_value(value.get("review_reason")))
        .collect::<Vec<_>>();
    review_reasons.sort();
    review_reasons.dedup();
    let max_usize = |key: &str| {
        values
            .iter()
            .filter_map(|value| usize_value(value.get(key)))
            .max()
    };
    let max_f64 = |key: &str| {
        values
            .iter()
            .filter_map(|value| f64_value(value.get(key)))
            .filter(|value| value.is_finite())
            .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
    };
    let detail_consistency = summarize_stitch_seam_detail_consistency(&values);

    Some(SeamBlendValidationSummary {
        mode,
        applied: Some(applied_merge_count == values.len()),
        reason: (values.len() == 1)
            .then(|| string_value(values[0].get("reason")))
            .flatten(),
        review_required: (review.len() == values.len()).then_some(review_required_merge_count > 0),
        review_required_merge_count,
        review_reasons,
        merge_count: values.len(),
        applied_merge_count,
        overlap_width_px: max_usize("overlap_width_px"),
        overlap_height_px: max_usize("overlap_height_px"),
        transition_width_px: max_usize("transition_width_px"),
        pyramid_levels: max_usize("pyramid_levels"),
        seam_path_mean_normalized_cost: max_f64("seam_path_mean_normalized_cost"),
        seam_path_p95_normalized_cost: max_f64("seam_path_p95_normalized_cost"),
        overlap_mean_abs_difference: max_f64("overlap_mean_abs_difference"),
        overlap_p95_abs_difference: max_f64("overlap_p95_abs_difference"),
        output_seam_gradient_p95: max_f64("output_seam_gradient_p95"),
        source_seam_gradient_p95: max_f64("source_seam_gradient_p95"),
        output_to_source_seam_gradient_ratio: max_f64("output_to_source_seam_gradient_ratio"),
        detail_consistency,
    })
}

fn summarize_spatial_photometric_quadratic(
    value: &serde_json::Value,
) -> SpatialPhotometricQuadraticSummary {
    SpatialPhotometricQuadraticSummary {
        basis: value
            .get("basis")
            .and_then(serde_json::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(str::to_string))
                    .collect()
            }),
        evaluation_grid_size: usize_value(value.get("evaluation_grid_size")),
        regularization_lambda: f64_value(value.get("regularization_lambda")),
        minimum_windows_per_split: usize_value(value.get("minimum_windows_per_split")),
        minimum_rows_per_split: usize_value(value.get("minimum_rows_per_split")),
        minimum_columns_per_split: usize_value(value.get("minimum_columns_per_split")),
        training_window_count: usize_value(value.get("training_window_count")),
        held_out_window_count: usize_value(value.get("held_out_window_count")),
        distinct_training_rows: usize_value(value.get("distinct_training_rows")),
        distinct_held_out_rows: usize_value(value.get("distinct_held_out_rows")),
        distinct_training_columns: usize_value(value.get("distinct_training_columns")),
        distinct_held_out_columns: usize_value(value.get("distinct_held_out_columns")),
        gain_design_condition_number: f64_value(value.get("gain_design_condition_number")),
        held_out_gain_design_condition_number: f64_value(
            value.get("held_out_gain_design_condition_number"),
        ),
        gain_consistent_window_ratio: f64_value(value.get("gain_consistent_window_ratio")),
        gain_curvature_coefficient_agreement_ratio: f64_value(
            value.get("gain_curvature_coefficient_agreement_ratio"),
        ),
        gain_max_validation_field_log_delta: f64_value(
            value.get("gain_max_validation_field_log_delta"),
        ),
        gain_curvature_signal: f64_value(value.get("gain_curvature_signal")),
        estimated_gain_log_quadratic_xx_rgb: value
            .get("estimated_gain_log_quadratic_xx_rgb")
            .and_then(f64_vec_value),
        estimated_gain_log_quadratic_xy_rgb: value
            .get("estimated_gain_log_quadratic_xy_rgb")
            .and_then(f64_vec_value),
        estimated_gain_log_quadratic_yy_rgb: value
            .get("estimated_gain_log_quadratic_yy_rgb")
            .and_then(f64_vec_value),
        gain_grid_min_rgb: value.get("gain_grid_min_rgb").and_then(f64_vec_value),
        gain_grid_max_rgb: value.get("gain_grid_max_rgb").and_then(f64_vec_value),
        gain_offset_design_condition_number: f64_value(
            value.get("gain_offset_design_condition_number"),
        ),
        held_out_gain_offset_design_condition_number: f64_value(
            value.get("held_out_gain_offset_design_condition_number"),
        ),
        gain_offset_consistent_window_ratio: f64_value(
            value.get("gain_offset_consistent_window_ratio"),
        ),
        gain_offset_curvature_coefficient_agreement_ratio: f64_value(
            value.get("gain_offset_curvature_coefficient_agreement_ratio"),
        ),
        gain_offset_max_validation_gain_field_log_delta: f64_value(
            value.get("gain_offset_max_validation_gain_field_log_delta"),
        ),
        gain_offset_max_validation_offset_field_delta_normalized: f64_value(
            value.get("gain_offset_max_validation_offset_field_delta_normalized"),
        ),
        gain_offset_curvature_signal: f64_value(value.get("gain_offset_curvature_signal")),
        estimated_gain_offset_log_quadratic_xx_rgb: value
            .get("estimated_gain_offset_log_quadratic_xx_rgb")
            .and_then(f64_vec_value),
        estimated_gain_offset_log_quadratic_xy_rgb: value
            .get("estimated_gain_offset_log_quadratic_xy_rgb")
            .and_then(f64_vec_value),
        estimated_gain_offset_log_quadratic_yy_rgb: value
            .get("estimated_gain_offset_log_quadratic_yy_rgb")
            .and_then(f64_vec_value),
        estimated_gain_offset_quadratic_xx_rgb: value
            .get("estimated_gain_offset_quadratic_xx_rgb")
            .and_then(f64_vec_value),
        estimated_gain_offset_quadratic_xy_rgb: value
            .get("estimated_gain_offset_quadratic_xy_rgb")
            .and_then(f64_vec_value),
        estimated_gain_offset_quadratic_yy_rgb: value
            .get("estimated_gain_offset_quadratic_yy_rgb")
            .and_then(f64_vec_value),
        gain_offset_grid_gain_min_rgb: value
            .get("gain_offset_grid_gain_min_rgb")
            .and_then(f64_vec_value),
        gain_offset_grid_gain_max_rgb: value
            .get("gain_offset_grid_gain_max_rgb")
            .and_then(f64_vec_value),
        gain_offset_grid_offset_abs_max_normalized: f64_value(
            value.get("gain_offset_grid_offset_abs_max_normalized"),
        ),
        held_out_gain_seam_score: f64_value(value.get("held_out_gain_seam_score")),
        held_out_gain_offset_seam_score: f64_value(value.get("held_out_gain_offset_seam_score")),
        gain_best_simpler_model: string_value(value.get("gain_best_simpler_model")),
        gain_improvement_over_best_simpler: f64_value(
            value.get("gain_improvement_over_best_simpler"),
        ),
        gain_offset_best_simpler_model: string_value(value.get("gain_offset_best_simpler_model")),
        gain_offset_improvement_over_best_simpler: f64_value(
            value.get("gain_offset_improvement_over_best_simpler"),
        ),
        gain_accepted: value
            .get("gain_accepted")
            .and_then(serde_json::Value::as_bool),
        gain_offset_accepted: value
            .get("gain_offset_accepted")
            .and_then(serde_json::Value::as_bool),
        gain_rejection_reason: string_value(value.get("gain_rejection_reason")),
        gain_offset_rejection_reason: string_value(value.get("gain_offset_rejection_reason")),
    }
}

fn summarize_spatial_photometric_2d(value: &serde_json::Value) -> SpatialPhotometric2dSummary {
    SpatialPhotometric2dSummary {
        coordinate_system: string_value(value.get("coordinate_system")),
        training_window_count: usize_value(value.get("training_window_count")),
        held_out_window_count: usize_value(value.get("held_out_window_count")),
        distinct_training_rows: usize_value(value.get("distinct_training_rows")),
        distinct_held_out_rows: usize_value(value.get("distinct_held_out_rows")),
        distinct_training_columns: usize_value(value.get("distinct_training_columns")),
        distinct_held_out_columns: usize_value(value.get("distinct_held_out_columns")),
        gain_consistent_window_ratio: f64_value(value.get("gain_consistent_window_ratio")),
        gain_slope_agreement_ratio: f64_value(value.get("gain_slope_agreement_ratio")),
        gain_horizontal_slope_agreement_ratio: f64_value(
            value.get("gain_horizontal_slope_agreement_ratio"),
        ),
        gain_center_log_delta: f64_value(value.get("gain_center_log_delta")),
        gain_offset_consistent_window_ratio: f64_value(
            value.get("gain_offset_consistent_window_ratio"),
        ),
        gain_offset_slope_agreement_ratio: f64_value(
            value.get("gain_offset_slope_agreement_ratio"),
        ),
        gain_offset_horizontal_slope_agreement_ratio: f64_value(
            value.get("gain_offset_horizontal_slope_agreement_ratio"),
        ),
        gain_offset_center_gain_log_delta: f64_value(
            value.get("gain_offset_center_gain_log_delta"),
        ),
        gain_offset_center_offset_delta_normalized: f64_value(
            value.get("gain_offset_center_offset_delta_normalized"),
        ),
        held_out_gain_seam_score: f64_value(value.get("held_out_gain_seam_score")),
        held_out_gain_offset_seam_score: f64_value(value.get("held_out_gain_offset_seam_score")),
        gain_best_simpler_model: string_value(value.get("gain_best_simpler_model")),
        gain_improvement_over_best_simpler: f64_value(
            value.get("gain_improvement_over_best_simpler"),
        ),
        gain_offset_best_simpler_model: string_value(value.get("gain_offset_best_simpler_model")),
        gain_offset_improvement_over_best_simpler: f64_value(
            value.get("gain_offset_improvement_over_best_simpler"),
        ),
        gain_accepted: value
            .get("gain_accepted")
            .and_then(serde_json::Value::as_bool),
        gain_offset_accepted: value
            .get("gain_offset_accepted")
            .and_then(serde_json::Value::as_bool),
        gain_rejection_reason: string_value(value.get("gain_rejection_reason")),
        gain_offset_rejection_reason: string_value(value.get("gain_offset_rejection_reason")),
        quadratic: value
            .get("quadratic")
            .filter(|value| value.is_object())
            .map(summarize_spatial_photometric_quadratic),
    }
}

fn summarize_seam_exposure_correction(value: &serde_json::Value) -> SeamExposureCorrectionSummary {
    SeamExposureCorrectionSummary {
        mode: string_value(value.get("mode")),
        model: string_value(value.get("model")),
        applied: value.get("applied").and_then(serde_json::Value::as_bool),
        reason: string_value(value.get("reason")),
        sample_count: usize_value(value.get("sample_count")),
        valid_sample_ratio: f64_value(value.get("valid_sample_ratio")),
        gain_rgb: value.get("gain_rgb").and_then(f64_vec_value),
        gain_luma: f64_value(value.get("gain_luma")),
        spatial_gain_log_slope_x_rgb: value
            .get("spatial_gain_log_slope_x_rgb")
            .and_then(f64_vec_value),
        spatial_gain_log_slope_x_luma: f64_value(value.get("spatial_gain_log_slope_x_luma")),
        spatial_gain_log_slope_y_rgb: value
            .get("spatial_gain_log_slope_y_rgb")
            .and_then(f64_vec_value),
        spatial_gain_log_slope_y_luma: f64_value(value.get("spatial_gain_log_slope_y_luma")),
        spatial_gain_log_quadratic_xx_rgb: value
            .get("spatial_gain_log_quadratic_xx_rgb")
            .and_then(f64_vec_value),
        spatial_gain_log_quadratic_xx_luma: f64_value(
            value.get("spatial_gain_log_quadratic_xx_luma"),
        ),
        spatial_gain_log_quadratic_xy_rgb: value
            .get("spatial_gain_log_quadratic_xy_rgb")
            .and_then(f64_vec_value),
        spatial_gain_log_quadratic_xy_luma: f64_value(
            value.get("spatial_gain_log_quadratic_xy_luma"),
        ),
        spatial_gain_log_quadratic_yy_rgb: value
            .get("spatial_gain_log_quadratic_yy_rgb")
            .and_then(f64_vec_value),
        spatial_gain_log_quadratic_yy_luma: f64_value(
            value.get("spatial_gain_log_quadratic_yy_luma"),
        ),
        spatial_gain_top_rgb: value.get("spatial_gain_top_rgb").and_then(f64_vec_value),
        spatial_gain_bottom_rgb: value.get("spatial_gain_bottom_rgb").and_then(f64_vec_value),
        spatial_offset_slope_x_rgb: value
            .get("spatial_offset_slope_x_rgb")
            .and_then(f64_vec_value),
        spatial_offset_slope_x_luma: f64_value(value.get("spatial_offset_slope_x_luma")),
        spatial_offset_slope_x_rgb_normalized: value
            .get("spatial_offset_slope_x_rgb_normalized")
            .and_then(f64_vec_value),
        spatial_offset_slope_y_rgb: value
            .get("spatial_offset_slope_y_rgb")
            .and_then(f64_vec_value),
        spatial_offset_slope_y_luma: f64_value(value.get("spatial_offset_slope_y_luma")),
        spatial_offset_slope_y_rgb_normalized: value
            .get("spatial_offset_slope_y_rgb_normalized")
            .and_then(f64_vec_value),
        spatial_offset_quadratic_xx_rgb: value
            .get("spatial_offset_quadratic_xx_rgb")
            .and_then(f64_vec_value),
        spatial_offset_quadratic_xx_luma: f64_value(value.get("spatial_offset_quadratic_xx_luma")),
        spatial_offset_quadratic_xx_rgb_normalized: value
            .get("spatial_offset_quadratic_xx_rgb_normalized")
            .and_then(f64_vec_value),
        spatial_offset_quadratic_xy_rgb: value
            .get("spatial_offset_quadratic_xy_rgb")
            .and_then(f64_vec_value),
        spatial_offset_quadratic_xy_luma: f64_value(value.get("spatial_offset_quadratic_xy_luma")),
        spatial_offset_quadratic_xy_rgb_normalized: value
            .get("spatial_offset_quadratic_xy_rgb_normalized")
            .and_then(f64_vec_value),
        spatial_offset_quadratic_yy_rgb: value
            .get("spatial_offset_quadratic_yy_rgb")
            .and_then(f64_vec_value),
        spatial_offset_quadratic_yy_luma: f64_value(value.get("spatial_offset_quadratic_yy_luma")),
        spatial_offset_quadratic_yy_rgb_normalized: value
            .get("spatial_offset_quadratic_yy_rgb_normalized")
            .and_then(f64_vec_value),
        spatial_offset_top_rgb: value.get("spatial_offset_top_rgb").and_then(f64_vec_value),
        spatial_offset_bottom_rgb: value
            .get("spatial_offset_bottom_rgb")
            .and_then(f64_vec_value),
        spatial_offset_top_rgb_normalized: value
            .get("spatial_offset_top_rgb_normalized")
            .and_then(f64_vec_value),
        spatial_offset_bottom_rgb_normalized: value
            .get("spatial_offset_bottom_rgb_normalized")
            .and_then(f64_vec_value),
        offset_rgb: value.get("offset_rgb").and_then(f64_vec_value),
        offset_luma: f64_value(value.get("offset_luma")),
        offset_rgb_normalized: value.get("offset_rgb_normalized").and_then(f64_vec_value),
        seam_score_before: f64_value(value.get("seam_score_before")),
        seam_score_after: f64_value(value.get("seam_score_after")),
        clipped_high_before: value.get("clipped_high_before").and_then(f64_vec_value),
        clipped_high_after: value.get("clipped_high_after").and_then(f64_vec_value),
        clipped_low_before: value.get("clipped_low_before").and_then(f64_vec_value),
        clipped_low_after: value.get("clipped_low_after").and_then(f64_vec_value),
        training_window_count: usize_value(value.get("training_window_count")),
        held_out_window_count: usize_value(value.get("held_out_window_count")),
        training_sample_count: usize_value(value.get("training_sample_count")),
        held_out_sample_count: usize_value(value.get("held_out_sample_count")),
        gain_offset_training_window_count: usize_value(
            value.get("gain_offset_training_window_count"),
        ),
        gain_offset_held_out_window_count: usize_value(
            value.get("gain_offset_held_out_window_count"),
        ),
        gain_offset_consistent_window_ratio: f64_value(
            value.get("gain_offset_consistent_window_ratio"),
        ),
        spatial_training_window_count: usize_value(value.get("spatial_training_window_count")),
        spatial_held_out_window_count: usize_value(value.get("spatial_held_out_window_count")),
        spatial_distinct_training_rows: usize_value(value.get("spatial_distinct_training_rows")),
        spatial_distinct_held_out_rows: usize_value(value.get("spatial_distinct_held_out_rows")),
        spatial_consistent_window_ratio: f64_value(value.get("spatial_consistent_window_ratio")),
        spatial_slope_agreement_ratio: f64_value(value.get("spatial_slope_agreement_ratio")),
        spatial_affine_training_window_count: usize_value(
            value.get("spatial_affine_training_window_count"),
        ),
        spatial_affine_held_out_window_count: usize_value(
            value.get("spatial_affine_held_out_window_count"),
        ),
        spatial_affine_distinct_training_rows: usize_value(
            value.get("spatial_affine_distinct_training_rows"),
        ),
        spatial_affine_distinct_held_out_rows: usize_value(
            value.get("spatial_affine_distinct_held_out_rows"),
        ),
        spatial_affine_consistent_window_ratio: f64_value(
            value.get("spatial_affine_consistent_window_ratio"),
        ),
        spatial_affine_slope_agreement_ratio: f64_value(
            value.get("spatial_affine_slope_agreement_ratio"),
        ),
        spatial_affine_center_offset_delta_normalized: f64_value(
            value.get("spatial_affine_center_offset_delta_normalized"),
        ),
        held_out_identity_seam_score: f64_value(value.get("held_out_identity_seam_score")),
        held_out_gain_seam_score: f64_value(value.get("held_out_gain_seam_score")),
        held_out_gain_offset_seam_score: f64_value(value.get("held_out_gain_offset_seam_score")),
        held_out_spatial_gain_seam_score: f64_value(value.get("held_out_spatial_gain_seam_score")),
        held_out_spatial_gain_offset_seam_score: f64_value(
            value.get("held_out_spatial_gain_offset_seam_score"),
        ),
        held_out_selected_seam_score: f64_value(value.get("held_out_selected_seam_score")),
        held_out_improvement_over_identity: f64_value(
            value.get("held_out_improvement_over_identity"),
        ),
        held_out_improvement_over_gain: f64_value(value.get("held_out_improvement_over_gain")),
        held_out_spatial_improvement_over_best_constant: f64_value(
            value.get("held_out_spatial_improvement_over_best_constant"),
        ),
        held_out_spatial_gain_offset_improvement_over_best_simpler: f64_value(
            value.get("held_out_spatial_gain_offset_improvement_over_best_simpler"),
        ),
        held_out_validation_passed: value
            .get("held_out_validation_passed")
            .and_then(serde_json::Value::as_bool),
        gain_offset_rejection_reason: string_value(value.get("gain_offset_rejection_reason")),
        spatial_rejection_reason: string_value(value.get("spatial_rejection_reason")),
        spatial_gain_offset_rejection_reason: string_value(
            value.get("spatial_gain_offset_rejection_reason"),
        ),
        spatial_2d_validation: value
            .get("spatial_2d_validation")
            .filter(|value| value.is_object())
            .map(summarize_spatial_photometric_2d),
    }
}

fn summarize_high_frequency_grain(value: &serde_json::Value) -> HighFrequencyGrainSummary {
    HighFrequencyGrainSummary {
        sample_count: usize_value(value.get("sample_count")),
        sample_stride: usize_value(value.get("sample_stride")),
        flat_luma_structure_max: f64_value(value.get("flat_luma_structure_max")),
        flat_sample_count: usize_value(value.get("flat_sample_count")),
        flat_sample_ratio: f64_value(value.get("flat_sample_ratio")),
        luma_residual_median: f64_value(value.get("luma_residual_median")),
        luma_residual_p95: f64_value(value.get("luma_residual_p95")),
        chroma_residual_median: f64_value(value.get("chroma_residual_median")),
        chroma_residual_p95: f64_value(value.get("chroma_residual_p95")),
        chroma_to_luma_p95_ratio: f64_value(value.get("chroma_to_luma_p95_ratio")),
        flat_luma_residual_p95: f64_value(value.get("flat_luma_residual_p95")),
        flat_chroma_residual_p95: f64_value(value.get("flat_chroma_residual_p95")),
        flat_chroma_to_luma_p95_ratio: f64_value(value.get("flat_chroma_to_luma_p95_ratio")),
    }
}

fn summarize_colorspace(
    phase: &PhaseReport,
    identity: &ReportIdentitySummary,
) -> ColorspaceValidationSummary {
    let calibration = phase.metrics.get("calibration");
    let calibration_color_mapping_application = calibration
        .and_then(|value| value.get("color_mapping_application"))
        .filter(|value| value.is_object());
    let external_profile = calibration.and_then(|value| value.get("external_profile"));
    let scanner_profile = calibration.and_then(|value| value.get("scanner_profile"));
    let roll_profile = calibration.and_then(|value| value.get("roll_profile"));
    let film_stock = calibration.and_then(|value| value.get("film_stock"));
    let candidate_quality_scores = phase
        .metrics
        .get("candidate_quality_scores")
        .or_else(|| phase.metrics.get("candidate_scores"))
        .and_then(|value| value.as_array())
        .map(|scores| {
            scores
                .iter()
                .map(summarize_candidate_quality)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let selected_quality_components = candidate_quality_scores
        .iter()
        .find(|candidate| candidate.selected == Some(true))
        .and_then(|candidate| candidate.quality_components.clone())
        .or_else(|| {
            phase
                .metrics
                .get("selected_quality_components")
                .or_else(|| phase.metrics.get("quality_components"))
                .filter(|value| value.is_object())
                .map(summarize_quality_components)
        });
    ColorspaceValidationSummary {
        calibration_status: calibration.and_then(|value| string_value(value.get("status"))),
        calibration_source: calibration.and_then(|value| string_value(value.get("source"))),
        calibration_color_mapping_application: calibration_color_mapping_application
            .map(summarize_calibration_color_mapping_application),
        calibration_scanner_profile_status: scanner_profile
            .and_then(|value| string_value(value.get("status"))),
        calibration_scanner_profile_id: scanner_profile
            .and_then(|value| string_value(value.get("profile_id"))),
        calibration_roll_profile_status: roll_profile
            .and_then(|value| string_value(value.get("status"))),
        calibration_roll_profile_id: roll_profile
            .and_then(|value| string_value(value.get("profile_id"))),
        calibration_external_profile_path: external_profile.and_then(|value| {
            if value.is_string() {
                string_value(Some(value))
            } else {
                string_value(value.get("path"))
            }
        }),
        calibration_profile_schema_version: calibration
            .and_then(|value| u64_value(value.get("profile_schema_version")))
            .and_then(|value| u32::try_from(value).ok()),
        calibration_confidence: calibration.and_then(|value| f64_value(value.get("confidence"))),
        calibration_reason: calibration.and_then(|value| string_value(value.get("reason"))),
        calibration_matrix_condition_number: calibration
            .and_then(|value| f64_value(value.get("matrix_condition_number"))),
        calibration_whitepoint: calibration
            .and_then(|value| value.get("whitepoint"))
            .and_then(f64_vec_value),
        calibration_requested_film_stock: calibration
            .and_then(|value| string_value(value.get("requested_film_stock"))),
        calibration_film_stock_status: film_stock
            .and_then(|value| string_value(value.get("status"))),
        calibration_film_stock_reason: film_stock
            .and_then(|value| string_value(value.get("reason"))),
        calibration_film_stock_matched_roll_profiles: film_stock
            .and_then(|value| value.get("matched_roll_profiles"))
            .and_then(string_vec_value)
            .unwrap_or_default(),
        calibration_rejection_details: calibration
            .and_then(|value| value.get("rejection_details"))
            .and_then(string_vec_value)
            .unwrap_or_default(),
        render_input_source: string_metric(phase, "render_input_source"),
        render_input_reason: string_metric(phase, "render_input_reason"),
        density_candidate_evaluated: bool_metric(phase, "direct_density_candidate_evaluated"),
        exposure_scale: f64_metric(phase, "exposure_scale"),
        mapping_strategy: string_metric(phase, "mapping_strategy"),
        selected_mapping_reason: string_metric(phase, "selected_mapping_reason"),
        selected_candidate: string_metric(phase, "selected_candidate"),
        selected_candidate_rank: usize_metric(phase, "selected_candidate_rank"),
        selected_candidate_score: f64_metric(phase, "selected_candidate_score"),
        selected_quality_score: f64_metric(phase, "selected_quality_score")
            .or_else(|| f64_metric(phase, "selected_candidate_score")),
        technical_safety_score: f64_metric(phase, "technical_safety_score"),
        color_fidelity_score: f64_metric(phase, "color_fidelity_score"),
        candidate_risk: string_metric(phase, "candidate_risk"),
        tone_color_trust_state: string_metric(phase, "tone_color_trust_state").or_else(|| {
            phase
                .metrics
                .get("color_decision_summary")
                .and_then(|summary| string_value(summary.get("tone_color_trust_state")))
        }),
        selected_runner_up_quality_delta: f64_metric(phase, "selected_runner_up_quality_delta"),
        selected_quality_components,
        candidate_quality_scores,
        candidate_acceptance: phase
            .metrics
            .get("candidate_acceptance")
            .and_then(|value| value.as_array())
            .map(|diagnostics| {
                diagnostics
                    .iter()
                    .map(summarize_candidate_acceptance)
                    .collect()
            })
            .unwrap_or_default(),
        neutral_estimate_quality: phase
            .metrics
            .get("neutral_estimate_quality")
            .map(summarize_neutral_estimate_quality),
        neutral_sample_rejections: phase
            .metrics
            .get("neutral_sample_rejections")
            .map(summarize_neutral_sample_rejections),
        dominant_anchor_sample_rejections: phase
            .metrics
            .get("dominant_anchor_sample_rejections")
            .map(summarize_dominant_anchor_sample_rejections),
        dominant_anchor_quality: phase
            .metrics
            .get("dominant_anchor_quality")
            .map(summarize_dominant_anchor_quality),
        reference_patch_evaluation: phase
            .metrics
            .get("reference_patch_evaluation")
            .map(summarize_reference_patch_evaluation),
        neutral_trim_before_after: phase
            .metrics
            .get("neutral_trim_before_after")
            .map(summarize_neutral_trim_before_after),
        calibration_acceptance: phase
            .metrics
            .get("calibration_acceptance")
            .map(summarize_calibration_acceptance),
        neutral_safety_rescue: phase
            .metrics
            .get("neutral_safety_rescue")
            .map(summarize_neutral_safety_rescue),
        selection_rejections: string_vec_metric(phase, "selection_rejections").unwrap_or_default(),
        regularization_lambda: f64_metric(phase, "regularization_lambda"),
        neutral_sample_bands: usize_vec_metric(phase, "neutral_sample_bands"),
        dominant_anchor_bands: usize_vec_vec_metric(phase, "dominant_anchor_bands"),
        neutral_trim_scale: f64_vec_metric(phase, "neutral_trim_scale"),
        neutral_trim_applied: bool_metric(phase, "neutral_trim_applied"),
        channel_anchor_counts: usize_vec_metric(phase, "channel_anchor_counts"),
        channel_anchor_min_count: usize_metric(phase, "channel_anchor_min_count"),
        channel_anchor_low_support: bool_vec_metric(phase, "channel_anchor_low_support"),
        weak_anchor_fallback_used: bool_metric(phase, "weak_anchor_fallback_used"),
        gamut_fallback_used: bool_metric(phase, "gamut_fallback_used"),
        color_candidate_comparison_artifact: string_metric(
            phase,
            "color_candidate_comparison_artifact",
        ),
        gamut_clipping_map_artifact: string_metric(phase, "gamut_clipping_map_artifact"),
        scene_referred_prophoto_float_artifact: string_metric(
            phase,
            "scene_referred_prophoto_float_artifact",
        ),
        debug_artifacts: summarize_color_debug_artifacts(identity, phase),
        gamut_clipping_map_encoding: phase
            .metrics
            .get("gamut_clipping_map_diagnostics")
            .and_then(|value| string_value(value.get("encoding"))),
        gamut_clipping_map_preserved_ratio: phase
            .metrics
            .get("gamut_clipping_map_diagnostics")
            .and_then(|value| f64_value(value.get("preserved_ratio"))),
        gamut_clipping_map_any_clipped_ratio: phase
            .metrics
            .get("gamut_clipping_map_diagnostics")
            .and_then(|value| f64_value(value.get("any_clipped_ratio"))),
        image_matrix_pre_scale_clipped_low_ratio: f64_vec_metric(
            phase,
            "image_matrix_pre_scale_clipped_low_ratio",
        ),
        image_matrix_pre_scale_clipped_high_ratio: f64_vec_metric(
            phase,
            "image_matrix_pre_scale_clipped_high_ratio",
        ),
        image_matrix_exposure_scale: f64_metric(phase, "image_matrix_exposure_scale"),
        image_matrix_pre_scale_preserved_ratio: f64_metric(
            phase,
            "image_matrix_pre_scale_preserved_ratio",
        ),
        image_matrix_neutral_balance_delta: f64_vec_metric(
            phase,
            "image_matrix_neutral_balance_delta",
        ),
        calibrated_profile_pre_scale_clipped_low_ratio: f64_vec_metric(
            phase,
            "calibrated_profile_pre_scale_clipped_low_ratio",
        ),
        calibrated_profile_pre_scale_clipped_high_ratio: f64_vec_metric(
            phase,
            "calibrated_profile_pre_scale_clipped_high_ratio",
        ),
        calibrated_profile_exposure_scale: f64_metric(phase, "calibrated_profile_exposure_scale"),
        calibrated_profile_pre_scale_preserved_ratio: f64_metric(
            phase,
            "calibrated_profile_pre_scale_preserved_ratio",
        ),
        calibrated_profile_neutral_balance_delta: f64_vec_metric(
            phase,
            "calibrated_profile_neutral_balance_delta",
        ),
        pre_scale_preserved_ratio: f64_metric(phase, "pre_scale_preserved_ratio"),
        post_scale_preserved_ratio: f64_metric(phase, "post_scale_preserved_ratio"),
        post_scale_clipped_high_ratio: f64_vec_metric(phase, "post_scale_clipped_high_ratio"),
        post_scale_clipped_low_ratio: f64_vec_metric(phase, "post_scale_clipped_low_ratio"),
    }
}

fn summarize_candidate_quality(value: &serde_json::Value) -> ColorCandidateQualitySummary {
    ColorCandidateQualitySummary {
        candidate: string_value(value.get("candidate")),
        rank: usize_value(value.get("rank")),
        selected: value.get("selected").and_then(serde_json::Value::as_bool),
        quality_score: f64_value(value.get("quality_score"))
            .or_else(|| f64_value(value.get("score"))),
        technical_safety_score: f64_value(value.get("technical_safety_score")),
        color_fidelity_score: f64_value(value.get("color_fidelity_score")),
        selected_quality_delta: f64_value(value.get("selected_quality_delta")),
        rejected: value.get("rejected").and_then(serde_json::Value::as_bool),
        rejection_reason: string_value(value.get("rejection_reason")),
        pre_scale_clipped_low_total: f64_value(value.get("pre_scale_clipped_low_total")),
        pre_scale_clipped_low_max: f64_value(value.get("pre_scale_clipped_low_max")),
        pre_scale_clipped_high_total: f64_value(value.get("pre_scale_clipped_high_total")),
        pre_scale_clipped_high_max: f64_value(value.get("pre_scale_clipped_high_max")),
        reference_patch_rms_error: f64_value(value.get("reference_patch_rms_error")),
        reference_patch_rms_delta_e: f64_value(value.get("reference_patch_rms_delta_e")),
        reference_patch_rms_delta_e2000: f64_value(value.get("reference_patch_rms_delta_e2000")),
        reference_patch_delta_vs_image_derived: f64_value(
            value.get("reference_patch_delta_vs_image_derived"),
        ),
        reference_patch_max_delta_vs_image_derived: f64_value(
            value.get("reference_patch_max_delta_vs_image_derived"),
        ),
        reference_patch_delta_e_delta_vs_image_derived: f64_value(
            value.get("reference_patch_delta_e_delta_vs_image_derived"),
        ),
        reference_patch_delta_e_max_delta_vs_image_derived: f64_value(
            value.get("reference_patch_delta_e_max_delta_vs_image_derived"),
        ),
        reference_patch_delta_e2000_delta_vs_image_derived: f64_value(
            value.get("reference_patch_delta_e2000_delta_vs_image_derived"),
        ),
        reference_patch_delta_e2000_max_delta_vs_image_derived: f64_value(
            value.get("reference_patch_delta_e2000_max_delta_vs_image_derived"),
        ),
        reference_patch_regresses_image_derived: value
            .get("reference_patch_regresses_image_derived")
            .and_then(serde_json::Value::as_bool),
        dominant_anchor_quality_score: f64_value(value.get("dominant_anchor_quality_score")),
        dominant_anchor_unstable_channels: value
            .get("dominant_anchor_unstable_channels")
            .and_then(bool_vec_value),
        density_monotonicity_score: value
            .get("color_model_quality")
            .and_then(|model| f64_value(model.get("density_monotonicity_score"))),
        hue_linearity_score: value
            .get("color_model_quality")
            .and_then(|model| f64_value(model.get("hue_linearity_score"))),
        saturation_preservation_median_ratio: value
            .get("color_model_quality")
            .and_then(|model| f64_value(model.get("saturation_preservation_median_ratio"))),
        spatial_neutral_delta_p95: value
            .get("color_model_quality")
            .and_then(|model| model.get("spatial_consistency"))
            .and_then(|spatial| f64_value(spatial.get("neutral_delta_p95"))),
        quality_components: value
            .get("quality_components")
            .map(summarize_quality_components),
    }
}

fn summarize_quality_components(value: &serde_json::Value) -> ColorQualityComponentsSummary {
    ColorQualityComponentsSummary {
        low_gamut_clip_penalty: f64_value(value.get("low_gamut_clip_penalty")),
        high_gamut_clip_penalty: f64_value(value.get("high_gamut_clip_penalty")),
        preserved_gamut_penalty: f64_value(value.get("preserved_gamut_penalty")),
        exposure_penalty: f64_value(value.get("exposure_penalty")),
        neutral_balance_penalty: f64_value(value.get("neutral_balance_penalty")),
        neutral_estimate_penalty: f64_value(value.get("neutral_estimate_penalty")),
        anchor_support_penalty: f64_value(value.get("anchor_support_penalty")),
        anchor_stability_penalty: f64_value(value.get("anchor_stability_penalty")),
        condition_penalty: f64_value(value.get("condition_penalty")),
        calibration_confidence_penalty: f64_value(value.get("calibration_confidence_penalty")),
        target_residual_penalty: f64_value(value.get("target_residual_penalty")),
        rendered_tone_penalty: f64_value(value.get("rendered_tone_penalty")),
        tone_chroma_cleanup_penalty: f64_value(value.get("tone_chroma_cleanup_penalty")),
        density_monotonicity_penalty: f64_value(value.get("density_monotonicity_penalty")),
        hue_linearity_penalty: f64_value(value.get("hue_linearity_penalty")),
        saturation_preservation_penalty: f64_value(value.get("saturation_preservation_penalty")),
        memory_color_penalty: f64_value(value.get("memory_color_penalty")),
        spatial_consistency_penalty: f64_value(value.get("spatial_consistency_penalty")),
        fallback_penalty: f64_value(value.get("fallback_penalty")),
    }
}

fn summarize_candidate_acceptance(value: &serde_json::Value) -> ColorCandidateAcceptanceSummary {
    ColorCandidateAcceptanceSummary {
        candidate: string_value(value.get("candidate")),
        candidate_kind: string_value(value.get("candidate_kind")),
        mapping_strategy: string_value(value.get("mapping_strategy")),
        source_label: string_value(value.get("source_label")),
        status: string_value(value.get("status")),
        reason: string_value(value.get("reason")),
        rank: usize_value(value.get("rank")),
        selected: value.get("selected").and_then(serde_json::Value::as_bool),
        eligible_in_color_mode: value
            .get("eligible_in_color_mode")
            .and_then(serde_json::Value::as_bool),
        quality_score: f64_value(value.get("quality_score")),
        selected_quality_delta: f64_value(value.get("selected_quality_delta")),
        rejected: value.get("rejected").and_then(serde_json::Value::as_bool),
        rejection_reason: string_value(value.get("rejection_reason")),
        beats_image_derived: value
            .get("beats_image_derived")
            .and_then(serde_json::Value::as_bool),
        within_negative_gamut_limits: value
            .get("within_negative_gamut_limits")
            .and_then(serde_json::Value::as_bool),
    }
}

fn summarize_neutral_estimate_quality(value: &serde_json::Value) -> NeutralEstimateQualitySummary {
    NeutralEstimateQualitySummary {
        score: f64_value(value.get("score")),
        accepted: value.get("accepted").and_then(serde_json::Value::as_bool),
        broad_support: value
            .get("broad_support")
            .and_then(serde_json::Value::as_bool),
        reason: string_value(value.get("reason")),
        sample_count: usize_value(value.get("sample_count")),
        populated_band_count: usize_value(value.get("populated_band_count")),
        dominant_band_fraction: f64_value(value.get("dominant_band_fraction")),
    }
}

fn summarize_neutral_sample_rejections(value: &serde_json::Value) -> NeutralSampleRejectionSummary {
    NeutralSampleRejectionSummary {
        total_pixels: usize_value(value.get("total_pixels")),
        accepted_neutral_samples: usize_value(value.get("accepted_neutral_samples")),
        clipped: usize_value(value.get("clipped")),
        border: usize_value(value.get("border")),
        film_base_like_edge: usize_value(value.get("film_base_like_edge")),
        dust: usize_value(value.get("dust")),
        luma_out_of_range: usize_value(value.get("luma_out_of_range")),
        chroma_threshold: usize_value(value.get("chroma_threshold")),
    }
}

fn summarize_dominant_anchor_sample_rejections(
    value: &serde_json::Value,
) -> DominantAnchorSampleRejectionSummary {
    DominantAnchorSampleRejectionSummary {
        total_pixels: usize_value(value.get("total_pixels")),
        accepted_anchor_samples: usize_value(value.get("accepted_anchor_samples")),
        clipped: usize_value(value.get("clipped")),
        border: usize_value(value.get("border")),
        film_base_like_edge: usize_value(value.get("film_base_like_edge")),
        dust: usize_value(value.get("dust")),
        luma_out_of_range: usize_value(value.get("luma_out_of_range")),
        low_saturation: usize_value(value.get("low_saturation")),
        weak_dominance: usize_value(value.get("weak_dominance")),
    }
}

fn summarize_dominant_anchor_quality(value: &serde_json::Value) -> DominantAnchorQualitySummary {
    DominantAnchorQualitySummary {
        score: f64_value(value.get("score")),
        accepted: value.get("accepted").and_then(serde_json::Value::as_bool),
        reason: string_value(value.get("reason")),
        channel_populated_band_count: value
            .get("channel_populated_band_count")
            .and_then(usize_vec_value),
        channel_dominant_band_fraction: value
            .get("channel_dominant_band_fraction")
            .and_then(f64_vec_value),
        channel_mean_dominance_margin: value
            .get("channel_mean_dominance_margin")
            .and_then(f64_vec_value),
        channel_stability_score: value.get("channel_stability_score").and_then(f64_vec_value),
        channel_unstable: value.get("channel_unstable").and_then(bool_vec_value),
        unstable_channel_count: usize_value(value.get("unstable_channel_count")),
    }
}

fn summarize_reference_patch_evaluation(
    value: &serde_json::Value,
) -> ReferencePatchEvaluationSummary {
    let worst_hue_families = value
        .get("worst_hue_families")
        .and_then(|value| value.as_array())
        .map(|families| {
            families
                .iter()
                .filter_map(|family| string_value(family.get("hue_family")))
                .collect()
        })
        .unwrap_or_default();
    let hue_family_regressions = value
        .get("candidate_evaluations")
        .and_then(|value| value.as_array())
        .map(|candidates| {
            candidates
                .iter()
                .filter(|candidate| {
                    candidate
                        .get("regresses_image_derived")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false)
                })
                .flat_map(|candidate| {
                    candidate
                        .get("hue_family_regressions")
                        .and_then(|value| value.as_array())
                        .into_iter()
                        .flatten()
                        .filter_map(|family| string_value(family.get("hue_family")))
                })
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect()
        })
        .unwrap_or_default();
    let regressed_candidates = value
        .get("candidate_evaluations")
        .and_then(|value| value.as_array())
        .map(|candidates| {
            candidates
                .iter()
                .filter(|candidate| {
                    candidate
                        .get("regresses_image_derived")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false)
                })
                .filter_map(|candidate| string_value(candidate.get("candidate")))
                .collect()
        })
        .unwrap_or_default();
    ReferencePatchEvaluationSummary {
        patch_count: usize_value(value.get("patch_count")),
        selected_candidate: string_value(value.get("selected_candidate")),
        selected_rms_error: f64_value(value.get("selected_rms_error")),
        selected_rms_delta_e: f64_value(value.get("selected_rms_delta_e")),
        selected_rms_delta_e2000: f64_value(value.get("selected_rms_delta_e2000")),
        image_derived_rms_error: f64_value(value.get("image_derived_rms_error")),
        image_derived_rms_delta_e: f64_value(value.get("image_derived_rms_delta_e")),
        image_derived_rms_delta_e2000: f64_value(value.get("image_derived_rms_delta_e2000")),
        rms_error_delta_vs_image_derived: f64_value(value.get("rms_error_delta_vs_image_derived")),
        max_error_delta_vs_image_derived: f64_value(value.get("max_error_delta_vs_image_derived")),
        delta_e_rms_delta_vs_image_derived: f64_value(
            value.get("delta_e_rms_delta_vs_image_derived"),
        ),
        delta_e_max_delta_vs_image_derived: f64_value(
            value.get("delta_e_max_delta_vs_image_derived"),
        ),
        delta_e2000_rms_delta_vs_image_derived: f64_value(
            value.get("delta_e2000_rms_delta_vs_image_derived"),
        ),
        delta_e2000_max_delta_vs_image_derived: f64_value(
            value.get("delta_e2000_max_delta_vs_image_derived"),
        ),
        selected_regresses_image_derived: value
            .get("selected_regresses_image_derived")
            .and_then(serde_json::Value::as_bool),
        worst_hue_families,
        hue_family_regressions,
        regressed_candidates,
    }
}

fn summarize_neutral_trim_before_after(value: &serde_json::Value) -> NeutralTrimBeforeAfterSummary {
    let before = value.get("before");
    let after = value.get("after");
    NeutralTrimBeforeAfterSummary {
        applied: value.get("applied").and_then(serde_json::Value::as_bool),
        reason: string_value(value.get("reason")),
        before_neutral_delta_magnitude: before
            .and_then(|value| f64_value(value.get("neutral_delta_magnitude"))),
        after_neutral_delta_magnitude: after
            .and_then(|value| f64_value(value.get("neutral_delta_magnitude"))),
        before_neutral_band_delta_magnitude: before
            .and_then(|value| option_f64_vec_value(value.get("neutral_band_delta_magnitude"))),
        after_neutral_band_delta_magnitude: after
            .and_then(|value| option_f64_vec_value(value.get("neutral_band_delta_magnitude"))),
        neutral_delta_reduced: value
            .get("neutral_delta_reduced")
            .and_then(serde_json::Value::as_bool),
        neutral_band_delta_worsened: value
            .get("neutral_band_delta_worsened")
            .and_then(serde_json::Value::as_bool),
        low_clipping_increased: value
            .get("low_clipping_increased")
            .and_then(serde_json::Value::as_bool),
        high_clipping_increased: value
            .get("high_clipping_increased")
            .and_then(serde_json::Value::as_bool),
    }
}

fn summarize_calibration_acceptance(value: &serde_json::Value) -> CalibrationAcceptanceSummary {
    CalibrationAcceptanceSummary {
        status: string_value(value.get("status")),
        reason: string_value(value.get("reason")),
        preferred_candidate: string_value(value.get("preferred_candidate")),
        preferred_candidate_quality_score: f64_value(
            value.get("preferred_candidate_quality_score"),
        ),
        image_derived_quality_score: f64_value(value.get("image_derived_quality_score")),
        beats_image_derived: value
            .get("beats_image_derived")
            .and_then(serde_json::Value::as_bool),
        within_negative_gamut_limits: value
            .get("within_negative_gamut_limits")
            .and_then(serde_json::Value::as_bool),
    }
}

fn summarize_calibration_color_mapping_application(
    value: &serde_json::Value,
) -> CalibrationColorMappingApplicationSummary {
    CalibrationColorMappingApplicationSummary {
        evaluated: value.get("evaluated").and_then(serde_json::Value::as_bool),
        applied: value.get("applied").and_then(serde_json::Value::as_bool),
        selection_status: string_value(value.get("selection_status")),
        selected_candidate: string_value(value.get("selected_candidate")),
        preferred_candidate: string_value(value.get("preferred_candidate")),
        reason: string_value(value.get("reason")),
        definition: string_value(value.get("definition")),
    }
}

fn summarize_neutral_safety_rescue(value: &serde_json::Value) -> NeutralSafetyRescueSummary {
    NeutralSafetyRescueSummary {
        evaluated: value.get("evaluated").and_then(serde_json::Value::as_bool),
        applied: value.get("applied").and_then(serde_json::Value::as_bool),
        matrix_candidate: string_value(value.get("matrix_candidate")),
        matrix_candidate_kind: string_value(value.get("matrix_candidate_kind")),
        matrix_anchor_evidence_supported: value
            .get("matrix_anchor_evidence_supported")
            .and_then(serde_json::Value::as_bool),
        neutral_estimate_supported: value
            .get("neutral_estimate_supported")
            .and_then(serde_json::Value::as_bool),
        neutral_model_evidence_supported: value
            .get("neutral_model_evidence_supported")
            .and_then(serde_json::Value::as_bool),
        matrix_pre_scale_preserved_ratio: f64_value(value.get("matrix_pre_scale_preserved_ratio")),
        neutral_pre_scale_preserved_ratio: f64_value(
            value.get("neutral_pre_scale_preserved_ratio"),
        ),
        preserved_ratio_gain: f64_value(value.get("preserved_ratio_gain")),
        minimum_preserved_ratio: f64_value(value.get("minimum_preserved_ratio")),
        minimum_preserved_ratio_gain: f64_value(value.get("minimum_preserved_ratio_gain")),
        matrix_midtone_saturation_p95: f64_value(value.get("matrix_midtone_saturation_p95")),
        neutral_midtone_saturation_p95: f64_value(value.get("neutral_midtone_saturation_p95")),
        midtone_saturation_p95_reduction: f64_value(value.get("midtone_saturation_p95_reduction")),
        maximum_midtone_saturation_p95: f64_value(value.get("maximum_midtone_saturation_p95")),
        minimum_midtone_saturation_p95_reduction: f64_value(
            value.get("minimum_midtone_saturation_p95_reduction"),
        ),
        matrix_memory_color_penalty: f64_value(value.get("matrix_memory_color_penalty")),
        neutral_memory_color_penalty: f64_value(value.get("neutral_memory_color_penalty")),
        matrix_spatial_consistency_penalty: f64_value(
            value.get("matrix_spatial_consistency_penalty"),
        ),
        neutral_spatial_consistency_penalty: f64_value(
            value.get("neutral_spatial_consistency_penalty"),
        ),
        neutral_saturation_preservation_sample_count: usize_value(
            value.get("neutral_saturation_preservation_sample_count"),
        ),
        neutral_saturation_preservation_p05_ratio: f64_value(
            value.get("neutral_saturation_preservation_p05_ratio"),
        ),
        neutral_saturation_preservation_median_ratio: f64_value(
            value.get("neutral_saturation_preservation_median_ratio"),
        ),
        neutral_saturation_preservation_p95_ratio: f64_value(
            value.get("neutral_saturation_preservation_p95_ratio"),
        ),
        reason: string_value(value.get("reason")),
    }
}

fn summarize_grain_detail_retention(
    value: &serde_json::Value,
) -> GrainDetailRetentionValidationSummary {
    GrainDetailRetentionValidationSummary {
        method: string_value(value.get("method")),
        evaluated: value.get("evaluated").and_then(serde_json::Value::as_bool),
        decision_supported: value
            .get("decision_supported")
            .and_then(serde_json::Value::as_bool),
        sample_stride: usize_value(value.get("sample_stride")),
        probe_radius: usize_value(value.get("probe_radius")),
        minimum_probe_count: usize_value(value.get("minimum_probe_count")),
        luminance_probe_count: usize_value(value.get("luminance_probe_count")),
        chroma_probe_count: usize_value(value.get("chroma_probe_count")),
        luminance_decision_supported: value
            .get("luminance_decision_supported")
            .and_then(serde_json::Value::as_bool),
        chroma_decision_supported: value
            .get("chroma_decision_supported")
            .and_then(serde_json::Value::as_bool),
        luminance_median_retention: f64_value(value.get("luminance_median_retention")),
        luminance_p10_retention: f64_value(value.get("luminance_p10_retention")),
        chroma_median_retention: f64_value(value.get("chroma_median_retention")),
        chroma_p10_retention: f64_value(value.get("chroma_p10_retention")),
        luminance_contrast_threshold: f64_value(value.get("luminance_contrast_threshold")),
        chroma_contrast_threshold: f64_value(value.get("chroma_contrast_threshold")),
        coherence_threshold: f64_value(value.get("coherence_threshold")),
        median_retention_threshold: f64_value(value.get("median_retention_threshold")),
        p10_retention_threshold: f64_value(value.get("p10_retention_threshold")),
        review_required: value
            .get("review_required")
            .and_then(serde_json::Value::as_bool),
        reason: string_value(value.get("reason")),
        review_reason: string_value(value.get("review_reason")),
    }
}

fn summarize_adaptive_vibrance_skin_memory_protection(
    value: &serde_json::Value,
) -> AdaptiveVibranceSkinMemoryProtectionSummary {
    AdaptiveVibranceSkinMemoryProtectionSummary {
        enabled: value.get("enabled").and_then(serde_json::Value::as_bool),
        method: string_value(value.get("method")),
        working_space: string_value(value.get("working_space")),
        reference: string_value(value.get("reference")),
        core_lightness: value.get("core_lightness").and_then(f64_vec_value),
        support_lightness: value.get("support_lightness").and_then(f64_vec_value),
        core_chroma: value.get("core_chroma").and_then(f64_vec_value),
        support_chroma: value.get("support_chroma").and_then(f64_vec_value),
        core_hue_degrees: value.get("core_hue_degrees").and_then(f64_vec_value),
        support_hue_degrees: value.get("support_hue_degrees").and_then(f64_vec_value),
        maximum_vibrance_reduction: f64_value(value.get("maximum_vibrance_reduction")),
        evaluated_pixel_ratio: f64_value(value.get("evaluated_pixel_ratio")),
        protected_pixel_ratio: f64_value(value.get("protected_pixel_ratio")),
        mean_protection_weight: f64_value(value.get("mean_protection_weight")),
        max_protection_weight: f64_value(value.get("max_protection_weight")),
    }
}

fn summarize_adaptive_vibrance_preferred_memory_color_family(
    value: &serde_json::Value,
) -> AdaptiveVibrancePreferredMemoryColorFamilySummary {
    AdaptiveVibrancePreferredMemoryColorFamilySummary {
        family: string_value(value.get("family")),
        preferred_center_lab: value.get("preferred_center_lab").and_then(f64_vec_value),
        preferred_center_lch: value.get("preferred_center_lch").and_then(f64_vec_value),
        semi_major_axis_ab: f64_value(value.get("semi_major_axis_ab")),
        semi_minor_axis_ab: f64_value(value.get("semi_minor_axis_ab")),
        axis_ratio: f64_value(value.get("axis_ratio")),
        ellipse_rotation_degrees: f64_value(value.get("ellipse_rotation_degrees")),
        matched_pixel_ratio: f64_value(value.get("matched_pixel_ratio")),
        limited_pixel_ratio: f64_value(value.get("limited_pixel_ratio")),
        mean_scale_reduction: f64_value(value.get("mean_scale_reduction")),
        max_scale_reduction: f64_value(value.get("max_scale_reduction")),
    }
}

fn summarize_adaptive_vibrance_preferred_memory_color_guard(
    value: &serde_json::Value,
) -> AdaptiveVibrancePreferredMemoryColorGuardSummary {
    AdaptiveVibrancePreferredMemoryColorGuardSummary {
        enabled: value.get("enabled").and_then(serde_json::Value::as_bool),
        method: string_value(value.get("method")),
        working_space: string_value(value.get("working_space")),
        reference: string_value(value.get("reference")),
        interpretation: string_value(value.get("interpretation")),
        core_normalized_radius: f64_value(value.get("core_normalized_radius")),
        support_normalized_radius: f64_value(value.get("support_normalized_radius")),
        evaluated_pixel_ratio: f64_value(value.get("evaluated_pixel_ratio")),
        matched_pixel_ratio: f64_value(value.get("matched_pixel_ratio")),
        limited_pixel_ratio: f64_value(value.get("limited_pixel_ratio")),
        mean_scale_reduction: f64_value(value.get("mean_scale_reduction")),
        max_scale_reduction: f64_value(value.get("max_scale_reduction")),
        families: value
            .get("families")
            .and_then(serde_json::Value::as_array)
            .map(|families| {
                families
                    .iter()
                    .filter(|family| family.is_object())
                    .map(summarize_adaptive_vibrance_preferred_memory_color_family)
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn summarize_preferred_skin_rendering(value: &serde_json::Value) -> PreferredSkinRenderingSummary {
    PreferredSkinRenderingSummary {
        enabled: value.get("enabled").and_then(serde_json::Value::as_bool),
        reason: string_value(value.get("reason")),
        method: string_value(value.get("method")),
        working_space: string_value(value.get("working_space")),
        preference_reference: string_value(value.get("preference_reference")),
        support_reference: string_value(value.get("support_reference")),
        interpretation: string_value(value.get("interpretation")),
        preferred_center_lab: value.get("preferred_center_lab").and_then(f64_vec_value),
        preferred_center_lch: value.get("preferred_center_lch").and_then(f64_vec_value),
        semi_major_axis_ab: f64_value(value.get("semi_major_axis_ab")),
        semi_minor_axis_ab: f64_value(value.get("semi_minor_axis_ab")),
        axis_ratio: f64_value(value.get("axis_ratio")),
        ellipse_rotation_degrees: f64_value(value.get("ellipse_rotation_degrees")),
        core_normalized_radius: f64_value(value.get("core_normalized_radius")),
        radial_excess_reduction: f64_value(value.get("radial_excess_reduction")),
        maximum_delta_e_ab: f64_value(value.get("maximum_delta_e_ab")),
        minimum_support_weight: f64_value(value.get("minimum_support_weight")),
        evaluated_pixel_ratio: f64_value(value.get("evaluated_pixel_ratio")),
        matched_pixel_ratio: f64_value(value.get("matched_pixel_ratio")),
        outside_preferred_core_ratio: f64_value(value.get("outside_preferred_core_ratio")),
        adjusted_pixel_ratio: f64_value(value.get("adjusted_pixel_ratio")),
        gamut_limited_pixel_ratio: f64_value(value.get("gamut_limited_pixel_ratio")),
        mean_delta_e_ab: f64_value(value.get("mean_delta_e_ab")),
        max_delta_e_ab: f64_value(value.get("max_delta_e_ab")),
        mean_abs_hue_shift_degrees: f64_value(value.get("mean_abs_hue_shift_degrees")),
        max_abs_hue_shift_degrees: f64_value(value.get("max_abs_hue_shift_degrees")),
        mean_chroma_delta: f64_value(value.get("mean_chroma_delta")),
        max_abs_chroma_delta: f64_value(value.get("max_abs_chroma_delta")),
    }
}

fn summarize_tone(phase: &PhaseReport) -> ToneValidationSummary {
    ToneValidationSummary {
        tone_confidence_status: string_metric(phase, "tone_confidence_status"),
        tone_output_evidence_evaluated: bool_metric(phase, "tone_output_evidence_evaluated"),
        tone_output_evidence_confidence: f64_metric(phase, "tone_output_evidence_confidence"),
        tone_output_confidence_status: string_metric(phase, "tone_output_confidence_status"),
        tone_output_review_required: bool_metric(phase, "tone_output_review_required"),
        tone_output_review_reason: string_metric(phase, "tone_output_review_reason"),
        confidence_limited_by_tone_output_evidence: bool_metric(
            phase,
            "confidence_limited_by_tone_output_evidence",
        ),
        input_luminance_range_p05_p95: f64_metric(phase, "input_luminance_range_p05_p95"),
        mapped_luminance_range_p05_p95: f64_metric(phase, "mapped_luminance_range_p05_p95"),
        render_to_mapped_luminance_range_ratio: f64_metric(
            phase,
            "render_to_mapped_luminance_range_ratio",
        ),
        maximum_post_tone_high_clip_ratio: f64_metric(phase, "maximum_post_tone_high_clip_ratio"),
        maximum_post_tone_low_clip_ratio: f64_metric(phase, "maximum_post_tone_low_clip_ratio"),
        highlight_chroma_compressed_ratio: f64_metric(phase, "highlight_chroma_compressed_ratio"),
        highlight_neutral_chroma_compressed_ratio: f64_metric(
            phase,
            "highlight_neutral_chroma_compressed_ratio",
        ),
        highlight_neutral_chroma_enabled: bool_metric(phase, "highlight_neutral_chroma_enabled"),
        shadow_chroma_compressed_ratio: f64_metric(phase, "shadow_chroma_compressed_ratio"),
        shadow_chroma_enabled: bool_metric(phase, "shadow_chroma_enabled"),
        color_protection_policy: string_metric(phase, "color_protection_policy"),
        color_trust_state: string_metric(phase, "color_trust_state").or_else(|| {
            string_metric(phase, "color_protection_policy").map(|policy| match policy.as_str() {
                "enabled" => "trusted".to_string(),
                "neutral_highlight_disabled_weak_neutral" => "limited_weak_neutral".to_string(),
                "weak_neutral_bounded_highlight_cleanup" => "limited_weak_neutral".to_string(),
                "weak_neutral_bounded_neutral_cleanup" => "limited_weak_neutral".to_string(),
                "review_bounded_neutral_shadow_cleanup" => "review_required".to_string(),
                "review_shadow_cleanup_only" => "review_required".to_string(),
                "disabled_color_candidate_review" => "review_required".to_string(),
                _ => "unknown".to_string(),
            })
        }),
        color_protection_reason: string_metric(phase, "color_protection_reason"),
        shadow_saturation_median: f64_metric(phase, "shadow_saturation_median"),
        shadow_saturation_p95: f64_metric(phase, "shadow_saturation_p95"),
        shadow_rgb_median: f64_vec_metric(phase, "shadow_rgb_median"),
        shadow_visible_pixel_count: phase
            .metrics
            .get("shadow_visible_pixel_count")
            .and_then(serde_json::Value::as_u64)
            .and_then(|count| usize::try_from(count).ok()),
        shadow_visible_saturation_median: f64_metric(phase, "shadow_visible_saturation_median"),
        shadow_visible_saturation_p95: f64_metric(phase, "shadow_visible_saturation_p95"),
        shadow_visible_rgb_median: f64_vec_metric(phase, "shadow_visible_rgb_median"),
        midtone_saturation_median: f64_metric(phase, "midtone_saturation_median"),
        midtone_saturation_p95: f64_metric(phase, "midtone_saturation_p95"),
        midtone_rgb_median: f64_vec_metric(phase, "midtone_rgb_median"),
        render_luminance_percentiles: f64_vec_metric(phase, "render_luminance_percentiles"),
        render_luminance_range_p05_p95: f64_metric(phase, "render_luminance_range_p05_p95"),
        midtone_luminance_percentiles: f64_vec_metric(phase, "midtone_luminance_percentiles"),
        midtone_neutral_pixel_count: phase
            .metrics
            .get("midtone_neutral_pixel_count")
            .and_then(serde_json::Value::as_u64)
            .and_then(|count| usize::try_from(count).ok()),
        midtone_neutral_saturation_median: f64_metric(phase, "midtone_neutral_saturation_median"),
        midtone_neutral_saturation_p95: f64_metric(phase, "midtone_neutral_saturation_p95"),
        midtone_neutral_rgb_median: f64_vec_metric(phase, "midtone_neutral_rgb_median"),
        midtone_saturated_saturation_median: f64_metric(
            phase,
            "midtone_saturated_saturation_median",
        ),
        midtone_saturated_saturation_p95: f64_metric(phase, "midtone_saturated_saturation_p95"),
        bright_neutral_saturation_median: f64_metric(phase, "bright_neutral_saturation_median"),
        bright_neutral_saturation_p95: f64_metric(phase, "bright_neutral_saturation_p95"),
        bright_neutral_rgb_median: f64_vec_metric(phase, "bright_neutral_rgb_median"),
        bright_saturated_saturation_median: f64_metric(phase, "bright_saturated_saturation_median"),
        bright_saturated_saturation_p95: f64_metric(phase, "bright_saturated_saturation_p95"),
        post_chroma_compression_clipped_high_ratio: f64_vec_metric(
            phase,
            "post_chroma_compression_clipped_high_ratio",
        ),
        post_chroma_compression_clipped_low_ratio: f64_vec_metric(
            phase,
            "post_chroma_compression_clipped_low_ratio",
        ),
        perceptual_gamut_mapping_space: string_metric(phase, "perceptual_gamut_mapping_space"),
        perceptual_gamut_mapped_ratio: f64_metric(phase, "perceptual_gamut_mapped_ratio"),
        perceptual_gamut_mean_chroma_scale: f64_metric(phase, "perceptual_gamut_mean_chroma_scale"),
        perceptual_gamut_min_chroma_scale: f64_metric(phase, "perceptual_gamut_min_chroma_scale"),
        adaptive_vibrance_skin_memory_protection: phase
            .metrics
            .get("adaptive_vibrance_skin_memory_protection")
            .filter(|value| value.is_object())
            .map(summarize_adaptive_vibrance_skin_memory_protection),
        adaptive_vibrance_preferred_memory_color_guard: phase
            .metrics
            .get("adaptive_vibrance_preferred_memory_color_guard")
            .filter(|value| value.is_object())
            .map(summarize_adaptive_vibrance_preferred_memory_color_guard),
        preferred_skin_rendering: phase
            .metrics
            .get("preferred_skin_rendering")
            .filter(|value| value.is_object())
            .map(summarize_preferred_skin_rendering),
        noise_reduction_enabled: bool_metric(phase, "noise_reduction_enabled"),
        noise_reduction_requested_enabled: bool_metric(phase, "noise_reduction_requested_enabled"),
        noise_reduction_reason: string_metric(phase, "noise_reduction_reason"),
        noise_reduction_requested_strength: f64_metric(phase, "noise_reduction_requested_strength"),
        noise_reduction_requested_scale: f64_metric(phase, "noise_reduction_requested_scale"),
        noise_reduction_radius: usize_metric(phase, "noise_reduction_radius"),
        noise_reduction_chroma_amount: f64_metric(phase, "noise_reduction_chroma_amount"),
        noise_reduction_luma_amount: f64_metric(phase, "noise_reduction_luma_amount"),
        noise_reduction_applied_ratio: f64_metric(phase, "noise_reduction_applied_ratio"),
        noise_reduction_structure_gate_start: f64_metric(
            phase,
            "noise_reduction_structure_gate_start",
        ),
        noise_reduction_structure_gate_end: f64_metric(phase, "noise_reduction_structure_gate_end"),
        noise_reduction_structure_excluded_ratio: f64_metric(
            phase,
            "noise_reduction_structure_excluded_ratio",
        ),
        noise_reduction_texture_limited_ratio: f64_metric(
            phase,
            "noise_reduction_texture_limited_ratio",
        ),
        noise_reduction_saturation_limited_ratio: f64_metric(
            phase,
            "noise_reduction_saturation_limited_ratio",
        ),
        noise_reduction_mean_abs_chroma_delta: f64_metric(
            phase,
            "noise_reduction_mean_abs_chroma_delta",
        ),
        noise_reduction_max_abs_chroma_delta: f64_metric(
            phase,
            "noise_reduction_max_abs_chroma_delta",
        ),
        noise_reduction_mean_abs_luma_delta: f64_metric(
            phase,
            "noise_reduction_mean_abs_luma_delta",
        ),
        noise_reduction_max_abs_luma_delta: f64_metric(phase, "noise_reduction_max_abs_luma_delta"),
        noise_reduction_flat_luma_p95_reduction_ratio: f64_metric(
            phase,
            "noise_reduction_flat_luma_p95_reduction_ratio",
        ),
        noise_reduction_flat_chroma_p95_reduction_ratio: f64_metric(
            phase,
            "noise_reduction_flat_chroma_p95_reduction_ratio",
        ),
        grain_detail_retention: phase
            .metrics
            .get("grain_reduction")
            .and_then(|grain| grain.get("detail_retention"))
            .filter(|value| value.is_object())
            .map(summarize_grain_detail_retention),
        high_frequency_grain: phase
            .metrics
            .get("high_frequency_grain")
            .map(summarize_high_frequency_grain),
    }
}

fn empty_colorspace_summary() -> ColorspaceValidationSummary {
    ColorspaceValidationSummary {
        calibration_status: None,
        calibration_source: None,
        calibration_color_mapping_application: None,
        calibration_scanner_profile_status: None,
        calibration_scanner_profile_id: None,
        calibration_roll_profile_status: None,
        calibration_roll_profile_id: None,
        calibration_external_profile_path: None,
        calibration_profile_schema_version: None,
        calibration_confidence: None,
        calibration_reason: None,
        calibration_matrix_condition_number: None,
        calibration_whitepoint: None,
        calibration_requested_film_stock: None,
        calibration_film_stock_status: None,
        calibration_film_stock_reason: None,
        calibration_film_stock_matched_roll_profiles: Vec::new(),
        calibration_rejection_details: Vec::new(),
        render_input_source: None,
        render_input_reason: None,
        density_candidate_evaluated: None,
        exposure_scale: None,
        mapping_strategy: None,
        selected_mapping_reason: None,
        selected_candidate: None,
        selected_candidate_rank: None,
        selected_candidate_score: None,
        selected_quality_score: None,
        technical_safety_score: None,
        color_fidelity_score: None,
        candidate_risk: None,
        tone_color_trust_state: None,
        selected_runner_up_quality_delta: None,
        selected_quality_components: None,
        candidate_quality_scores: Vec::new(),
        candidate_acceptance: Vec::new(),
        neutral_estimate_quality: None,
        neutral_sample_rejections: None,
        dominant_anchor_sample_rejections: None,
        dominant_anchor_quality: None,
        reference_patch_evaluation: None,
        neutral_trim_before_after: None,
        calibration_acceptance: None,
        neutral_safety_rescue: None,
        selection_rejections: Vec::new(),
        regularization_lambda: None,
        neutral_sample_bands: None,
        dominant_anchor_bands: None,
        neutral_trim_scale: None,
        neutral_trim_applied: None,
        channel_anchor_counts: None,
        channel_anchor_min_count: None,
        channel_anchor_low_support: None,
        weak_anchor_fallback_used: None,
        gamut_fallback_used: None,
        color_candidate_comparison_artifact: None,
        gamut_clipping_map_artifact: None,
        scene_referred_prophoto_float_artifact: None,
        debug_artifacts: Vec::new(),
        gamut_clipping_map_encoding: None,
        gamut_clipping_map_preserved_ratio: None,
        gamut_clipping_map_any_clipped_ratio: None,
        image_matrix_pre_scale_clipped_low_ratio: None,
        image_matrix_pre_scale_clipped_high_ratio: None,
        image_matrix_exposure_scale: None,
        image_matrix_pre_scale_preserved_ratio: None,
        image_matrix_neutral_balance_delta: None,
        calibrated_profile_pre_scale_clipped_low_ratio: None,
        calibrated_profile_pre_scale_clipped_high_ratio: None,
        calibrated_profile_exposure_scale: None,
        calibrated_profile_pre_scale_preserved_ratio: None,
        calibrated_profile_neutral_balance_delta: None,
        pre_scale_preserved_ratio: None,
        post_scale_preserved_ratio: None,
        post_scale_clipped_high_ratio: None,
        post_scale_clipped_low_ratio: None,
    }
}

fn empty_tone_summary() -> ToneValidationSummary {
    ToneValidationSummary {
        tone_confidence_status: None,
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
        highlight_chroma_compressed_ratio: None,
        highlight_neutral_chroma_compressed_ratio: None,
        highlight_neutral_chroma_enabled: None,
        shadow_chroma_compressed_ratio: None,
        shadow_chroma_enabled: None,
        color_protection_policy: None,
        color_trust_state: None,
        color_protection_reason: None,
        shadow_saturation_median: None,
        shadow_saturation_p95: None,
        shadow_rgb_median: None,
        shadow_visible_pixel_count: None,
        shadow_visible_saturation_median: None,
        shadow_visible_saturation_p95: None,
        shadow_visible_rgb_median: None,
        midtone_saturation_median: None,
        midtone_saturation_p95: None,
        midtone_rgb_median: None,
        render_luminance_percentiles: None,
        render_luminance_range_p05_p95: None,
        midtone_luminance_percentiles: None,
        midtone_neutral_pixel_count: None,
        midtone_neutral_saturation_median: None,
        midtone_neutral_saturation_p95: None,
        midtone_neutral_rgb_median: None,
        midtone_saturated_saturation_median: None,
        midtone_saturated_saturation_p95: None,
        bright_neutral_saturation_median: None,
        bright_neutral_saturation_p95: None,
        bright_neutral_rgb_median: None,
        bright_saturated_saturation_median: None,
        bright_saturated_saturation_p95: None,
        post_chroma_compression_clipped_high_ratio: None,
        post_chroma_compression_clipped_low_ratio: None,
        perceptual_gamut_mapping_space: None,
        perceptual_gamut_mapped_ratio: None,
        perceptual_gamut_mean_chroma_scale: None,
        perceptual_gamut_min_chroma_scale: None,
        adaptive_vibrance_skin_memory_protection: None,
        adaptive_vibrance_preferred_memory_color_guard: None,
        preferred_skin_rendering: None,
        noise_reduction_enabled: None,
        noise_reduction_requested_enabled: None,
        noise_reduction_reason: None,
        noise_reduction_requested_strength: None,
        noise_reduction_requested_scale: None,
        noise_reduction_radius: None,
        noise_reduction_chroma_amount: None,
        noise_reduction_luma_amount: None,
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
        noise_reduction_flat_luma_p95_reduction_ratio: None,
        noise_reduction_flat_chroma_p95_reduction_ratio: None,
        grain_detail_retention: None,
        high_frequency_grain: None,
    }
}

fn selected_hypothesis(phase: &PhaseReport) -> Option<&serde_json::Value> {
    let hypotheses = phase.metrics.get("hypotheses")?.as_array()?;
    if let Some(chosen) = string_metric(phase, "chosen_hypothesis") {
        if let Some(hypothesis) = hypotheses
            .iter()
            .find(|hypothesis| string_value(hypothesis.get("ordering")).as_deref() == Some(&chosen))
        {
            return Some(hypothesis);
        }
    }
    hypotheses
        .iter()
        .find(|hypothesis| {
            hypothesis
                .get("accepted")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        })
        .or_else(|| hypotheses.first())
}

fn phase<'a>(report: &'a PipelineReport, name: &str) -> Option<&'a PhaseReport> {
    report.phases.iter().find(|phase| phase.name == name)
}

fn string_metric(phase: &PhaseReport, key: &str) -> Option<String> {
    string_value(phase.metrics.get(key))
}

fn f64_metric(phase: &PhaseReport, key: &str) -> Option<f64> {
    f64_value(phase.metrics.get(key))
}

fn usize_metric(phase: &PhaseReport, key: &str) -> Option<usize> {
    usize_value(phase.metrics.get(key))
}

fn bool_metric(phase: &PhaseReport, key: &str) -> Option<bool> {
    phase.metrics.get(key).and_then(serde_json::Value::as_bool)
}

fn u64_metric(phase: &PhaseReport, key: &str) -> Option<u64> {
    u64_value(phase.metrics.get(key))
}

fn f64_vec_metric(phase: &PhaseReport, key: &str) -> Option<Vec<f64>> {
    phase.metrics.get(key).and_then(f64_vec_value)
}

fn usize_vec_metric(phase: &PhaseReport, key: &str) -> Option<Vec<usize>> {
    phase.metrics.get(key).and_then(usize_vec_value)
}

fn usize_vec_vec_metric(phase: &PhaseReport, key: &str) -> Option<Vec<Vec<usize>>> {
    phase.metrics.get(key).and_then(|value| {
        value
            .as_array()
            .map(|rows| rows.iter().filter_map(usize_vec_value).collect())
    })
}

fn bool_vec_metric(phase: &PhaseReport, key: &str) -> Option<Vec<bool>> {
    phase.metrics.get(key).and_then(bool_vec_value)
}

fn string_vec_metric(phase: &PhaseReport, key: &str) -> Option<Vec<String>> {
    phase.metrics.get(key).and_then(string_vec_value)
}

fn string_value(value: Option<&serde_json::Value>) -> Option<String> {
    value
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string)
}

fn f64_value(value: Option<&serde_json::Value>) -> Option<f64> {
    value.and_then(serde_json::Value::as_f64)
}

fn i64_value(value: Option<&serde_json::Value>) -> Option<i64> {
    value.and_then(serde_json::Value::as_i64)
}

fn usize_value(value: Option<&serde_json::Value>) -> Option<usize> {
    value
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn u64_value(value: Option<&serde_json::Value>) -> Option<u64> {
    value.and_then(serde_json::Value::as_u64)
}

fn f64_vec_value(value: &serde_json::Value) -> Option<Vec<f64>> {
    value.as_array().map(|values| {
        values
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .collect()
    })
}

fn option_f64_vec_value(value: Option<&serde_json::Value>) -> Option<Vec<Option<f64>>> {
    value.and_then(|value| {
        value
            .as_array()
            .map(|values| values.iter().map(serde_json::Value::as_f64).collect())
    })
}

fn usize_vec_value(value: &serde_json::Value) -> Option<Vec<usize>> {
    value.as_array().map(|values| {
        values
            .iter()
            .filter_map(|value| value.as_u64().and_then(|value| usize::try_from(value).ok()))
            .collect()
    })
}

fn bool_vec_value(value: &serde_json::Value) -> Option<Vec<bool>> {
    value.as_array().map(|values| {
        values
            .iter()
            .filter_map(serde_json::Value::as_bool)
            .collect()
    })
}

fn string_vec_value(value: &serde_json::Value) -> Option<Vec<String>> {
    value.as_array().map(|values| {
        values
            .iter()
            .filter_map(serde_json::Value::as_str)
            .map(ToString::to_string)
            .collect()
    })
}

fn push_row(out: &mut String, area: &str, field: &str, value: &Option<String>) {
    out.push_str(&format!(
        "| {} | `{}` | {} |\n",
        area,
        field,
        value.as_deref().unwrap_or("")
    ));
}

fn format_f64(value: Option<f64>) -> Option<String> {
    value.map(|value| format!("{value:.6}"))
}

fn format_usize(value: Option<usize>) -> Option<String> {
    value.map(|value| value.to_string())
}

fn format_dimensions(width: Option<usize>, height: Option<usize>) -> Option<String> {
    match (width, height) {
        (Some(width), Some(height)) => Some(format!("{width}x{height}")),
        _ => None,
    }
}

fn format_bool_vec(value: &Option<Vec<bool>>) -> Option<String> {
    value.as_ref().map(|values| {
        values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    })
}

fn format_f64_vec(value: &Option<Vec<f64>>) -> Option<String> {
    value.as_ref().map(|values| {
        values
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(", ")
    })
}

fn format_input_orientation_components(
    components: &[InputOrientationComponentValidationSummary],
) -> Option<String> {
    (!components.is_empty()).then(|| {
        components
            .iter()
            .map(|component| {
                let index = component
                    .index
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "?".to_string());
                let tag = component
                    .tag_value
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "absent".to_string());
                let transform = component.transform.as_deref().unwrap_or("missing");
                let correction = component
                    .orientation_correction_requested
                    .as_deref()
                    .unwrap_or("missing");
                let applied = component
                    .applied
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "missing".to_string());
                let source = format_dimensions(component.source_width, component.source_height)
                    .unwrap_or_else(|| "missing".to_string());
                let output = format_dimensions(component.output_width, component.output_height)
                    .unwrap_or_else(|| "missing".to_string());
                let decoded_pixel_sha256 = component
                    .decoded_pixel_sha256
                    .as_deref()
                    .unwrap_or("missing");
                format!(
                    "component {index}: tag={tag} effective_transform={transform} correction={correction} applied={applied} source={source} output={output} decoded_pixel_sha256={decoded_pixel_sha256}"
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    })
}

fn format_color_debug_artifacts(artifacts: &[ColorDebugArtifactSummary]) -> Option<String> {
    if artifacts.is_empty() {
        return None;
    }
    Some(
        artifacts
            .iter()
            .map(|artifact| {
                let freshness = artifact
                    .fresh_for_report
                    .map(|fresh| if fresh { "fresh" } else { "not_fresh" })
                    .unwrap_or("freshness_unknown");
                format!(
                    "{}:{}:{}:{} bytes={}",
                    artifact.kind,
                    artifact.status,
                    freshness,
                    artifact.path,
                    artifact
                        .file_size_bytes
                        .map(|size| size.to_string())
                        .unwrap_or_else(|| "n/a".to_string())
                )
            })
            .collect::<Vec<_>>()
            .join("; "),
    )
}

fn format_option_f64_vec(value: &Option<Vec<Option<f64>>>) -> Option<String> {
    value.as_ref().map(|values| {
        values
            .iter()
            .map(|value| {
                value
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "n/a".to_string())
            })
            .collect::<Vec<_>>()
            .join(", ")
    })
}

fn format_quality_components(components: &Option<ColorQualityComponentsSummary>) -> Option<String> {
    let components = components.as_ref()?;
    let mut parts = Vec::new();
    let mut push_component = |name: &str, value: Option<f64>| {
        if let Some(value) = value {
            parts.push(format!("{name}={value:.6}"));
        }
    };
    push_component("low_gamut", components.low_gamut_clip_penalty);
    push_component("high_gamut", components.high_gamut_clip_penalty);
    push_component("preserved_gamut", components.preserved_gamut_penalty);
    push_component("exposure", components.exposure_penalty);
    push_component("neutral_balance", components.neutral_balance_penalty);
    push_component("neutral_support", components.neutral_estimate_penalty);
    push_component("anchor_support", components.anchor_support_penalty);
    push_component("anchor_stability", components.anchor_stability_penalty);
    push_component("condition", components.condition_penalty);
    push_component(
        "calibration_confidence",
        components.calibration_confidence_penalty,
    );
    push_component("target_residual", components.target_residual_penalty);
    push_component("rendered_tone", components.rendered_tone_penalty);
    push_component(
        "tone_chroma_cleanup",
        components.tone_chroma_cleanup_penalty,
    );
    push_component(
        "density_monotonicity",
        components.density_monotonicity_penalty,
    );
    push_component("hue_linearity", components.hue_linearity_penalty);
    push_component(
        "saturation_preservation",
        components.saturation_preservation_penalty,
    );
    push_component("memory_color", components.memory_color_penalty);
    push_component(
        "spatial_consistency",
        components.spatial_consistency_penalty,
    );
    push_component("fallback", components.fallback_penalty);

    (!parts.is_empty()).then(|| parts.join(", "))
}

fn format_candidate_acceptance(candidates: &[ColorCandidateAcceptanceSummary]) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }

    Some(
        candidates
            .iter()
            .map(|candidate| {
                let mut parts = vec![format!(
                    "{}={}",
                    candidate.candidate.as_deref().unwrap_or("unknown"),
                    candidate.status.as_deref().unwrap_or("unknown"),
                )];
                if let Some(kind) = &candidate.candidate_kind {
                    parts.push(format!("kind={kind}"));
                }
                if let Some(strategy) = &candidate.mapping_strategy {
                    parts.push(format!("strategy={strategy}"));
                }
                if let Some(rank) = candidate.rank {
                    parts.push(format!("rank={rank}"));
                }
                if let Some(selected) = candidate.selected {
                    parts.push(format!("selected={selected}"));
                }
                if let Some(eligible) = candidate.eligible_in_color_mode {
                    parts.push(format!("eligible={eligible}"));
                }
                if let Some(rejected) = candidate.rejected {
                    parts.push(format!("rejected={rejected}"));
                }
                if let Some(reason) = &candidate.reason {
                    parts.push(format!("reason={}", reason.replace('|', "/")));
                }
                parts.join(" ")
            })
            .collect::<Vec<_>>()
            .join(", "),
    )
}

fn format_neutral_sample_rejections(rejections: &NeutralSampleRejectionSummary) -> String {
    format!(
        "accepted={}/{} clipped={} border={} film_base={} dust={} luma={} chroma={}",
        rejections.accepted_neutral_samples.unwrap_or(0),
        rejections.total_pixels.unwrap_or(0),
        rejections.clipped.unwrap_or(0),
        rejections.border.unwrap_or(0),
        rejections.film_base_like_edge.unwrap_or(0),
        rejections.dust.unwrap_or(0),
        rejections.luma_out_of_range.unwrap_or(0),
        rejections.chroma_threshold.unwrap_or(0)
    )
}

fn format_dominant_anchor_sample_rejections(
    rejections: &DominantAnchorSampleRejectionSummary,
) -> String {
    format!(
        "accepted={}/{} clipped={} border={} film_base={} dust={} luma={} low_sat={} weak_dom={}",
        rejections.accepted_anchor_samples.unwrap_or(0),
        rejections.total_pixels.unwrap_or(0),
        rejections.clipped.unwrap_or(0),
        rejections.border.unwrap_or(0),
        rejections.film_base_like_edge.unwrap_or(0),
        rejections.dust.unwrap_or(0),
        rejections.luma_out_of_range.unwrap_or(0),
        rejections.low_saturation.unwrap_or(0),
        rejections.weak_dominance.unwrap_or(0)
    )
}

fn format_dominant_anchor_quality(quality: &DominantAnchorQualitySummary) -> String {
    format!(
        "score={} accepted={} unstable_channels={} reason={}",
        quality
            .score
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        quality
            .accepted
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        quality
            .unstable_channel_count
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        quality.reason.as_deref().unwrap_or("n/a")
    )
}

fn format_reference_patch_evaluation(evaluation: &ReferencePatchEvaluationSummary) -> String {
    let regressed = evaluation
        .selected_regresses_image_derived
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let worst = if evaluation.worst_hue_families.is_empty() {
        "none".to_string()
    } else {
        evaluation.worst_hue_families.join(", ")
    };
    let hue_regressions = if evaluation.hue_family_regressions.is_empty() {
        "none".to_string()
    } else {
        evaluation.hue_family_regressions.join(", ")
    };
    let regressed_candidates = if evaluation.regressed_candidates.is_empty() {
        "none".to_string()
    } else {
        evaluation.regressed_candidates.join(", ")
    };
    format!(
        "patches={} selected={} xyz_rms={} image_xyz_rms={} xyz_delta={} xyz_max_delta={} delta_e_rms={} image_delta_e_rms={} delta_e_delta={} delta_e_max_delta={} delta_e2000_rms={} image_delta_e2000_rms={} delta_e2000_delta={} delta_e2000_max_delta={} regressed={} worst={} hue_regressions={} regressed_candidates={}",
        evaluation.patch_count.unwrap_or(0),
        evaluation
            .selected_candidate
            .as_deref()
            .unwrap_or("unknown"),
        evaluation
            .selected_rms_error
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        evaluation
            .image_derived_rms_error
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        evaluation
            .rms_error_delta_vs_image_derived
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        evaluation
            .max_error_delta_vs_image_derived
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string()),
        evaluation
            .selected_rms_delta_e
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        evaluation
            .image_derived_rms_delta_e
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        evaluation
            .delta_e_rms_delta_vs_image_derived
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        evaluation
            .delta_e_max_delta_vs_image_derived
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        evaluation
            .selected_rms_delta_e2000
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        evaluation
            .image_derived_rms_delta_e2000
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        evaluation
            .delta_e2000_rms_delta_vs_image_derived
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        evaluation
            .delta_e2000_max_delta_vs_image_derived
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        regressed,
        worst,
        hue_regressions,
        regressed_candidates
    )
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn changed(left: &Option<String>, right: &Option<String>) -> bool {
    left != right && (left.is_some() || right.is_some())
}

fn changed_expected<T: PartialEq>(expected: &Option<T>, current: &Option<T>) -> bool {
    expected.is_some() && expected != current
}

fn delta(baseline: Option<f64>, current: Option<f64>) -> Option<f64> {
    Some(current? - baseline?)
}

fn ratio(baseline: Option<f64>, current: Option<f64>) -> Option<f64> {
    let baseline = baseline?;
    if baseline.abs() <= 1e-12 {
        return None;
    }
    Some(current? / baseline)
}

fn ratio_outside(value: Option<f64>, low: f64, high: f64) -> bool {
    value.is_some_and(|value| value < low || value > high)
}

fn default_confidence_abs_tolerance() -> f64 {
    0.02
}

fn default_base_support_fraction_abs_tolerance() -> f64 {
    0.00005
}

fn default_tone_ratio_abs_tolerance() -> f64 {
    0.02
}

fn default_grain_ratio_abs_tolerance() -> f64 {
    0.02
}

fn default_colorspace_quality_abs_tolerance() -> f64 {
    0.25
}

fn default_colorspace_preserved_abs_tolerance() -> f64 {
    0.02
}

fn default_colorspace_score_abs_tolerance() -> f64 {
    0.05
}

fn default_color_ratio_abs_tolerance() -> f64 {
    0.25
}

fn default_spatial_neutral_abs_tolerance() -> f64 {
    0.03
}

fn default_reference_xyz_abs_tolerance() -> f64 {
    0.005
}

fn default_reference_delta_e_abs_tolerance() -> f64 {
    1.0
}

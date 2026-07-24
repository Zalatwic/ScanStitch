use nalgebra::{Matrix3, Vector3};
use ndarray::parallel::prelude::*;
use ndarray::{Array3, Axis};
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::color_calibration::{
    CalibrationApplicationMode, CalibrationProfile, ResidualLut3dColorModel,
    RootPolynomialColorModel, TargetFitDiagnostics, TargetPatch,
};
use crate::constants::{
    BRADFORD_LMS_TO_XYZ, BRADFORD_XYZ_TO_LMS, D50_WHITE, PROPHOTO_TO_XYZ_D50, XYZ_D50_TO_PROPHOTO,
};
use crate::streaming;
use crate::tonemap;

const NEUTRAL_THRESHOLD: f64 = 0.15;
const MIN_NEUTRAL_FRACTION: f64 = 0.005;
const DOMINANT_RATIO_THRESHOLD: f64 = 1.18;
const MIN_DOMINANT_SATURATION: f64 = 0.12;
const HISTOGRAM_BINS: usize = 4096;
const HIGHLIGHT_HEADROOM_PERCENTILE: f64 = 0.995;
const MAX_LOW_GAMUT_CLIP_RATIO_PER_CHANNEL: f64 = 0.08;
const MAX_LOW_GAMUT_CLIP_RATIO_TOTAL: f64 = 0.18;
const DIRECT_RENDER_LOW_CLIP_RELATIVE_LIMIT: f64 = 0.50;
const DIRECT_RENDER_LOW_CLIP_ABSOLUTE_MARGIN: f64 = 0.05;
const DIRECT_RENDER_QUALITY_IMPROVEMENT_MARGIN: f64 = 0.20;
const DIRECT_RENDER_TRUST_IMPROVEMENT_QUALITY_TOLERANCE: f64 = 0.25;
const DIRECT_RENDER_PHYSICAL_PRIOR_QUALITY_TOLERANCE: f64 = 1.00;
const DIRECT_RENDER_PRESERVED_GAMUT_REGRESSION_TOLERANCE: f64 = 0.005;
const DIRECT_RENDER_TRUST_IMPROVEMENT_MIN_PRESERVED_GAMUT: f64 = 0.97;
const DIRECT_RENDER_LOW_CLIP_REGRESSION_TOLERANCE: f64 = 0.01;
const CATASTROPHIC_RENDER_MIN_PRESERVED_GAMUT: f64 = 0.90;
const CATASTROPHIC_RENDER_MAX_POST_SCALE_CLIP_TOTAL: f64 = 0.10;
const GAMUT_SAFE_BLEND_SEARCH_STEPS: usize = 8;
const GAMUT_SAFE_BLEND_MAX_NEUTRAL_MIX: f64 = 0.98;
const GAMUT_TRUSTED_BLEND_SEARCH_STEPS: usize = 10;
const GAMUT_TRUSTED_BLEND_MAX_LOW_RATIO_PER_CHANNEL: f64 = 0.01;
const GAMUT_TRUSTED_BLEND_MAX_LOW_RATIO_TOTAL: f64 = 0.02;
const GAMUT_STABILIZED_BLEND_MIN_TRUSTED_MIX: f64 = 0.85;
const GAMUT_STABILIZED_BLEND_MAX_EXPOSURE_SCALE: f64 = 2.25;
const GAMUT_STABILIZED_BLEND_EXTRA_FRACTION: f64 = 0.35;
const GAMUT_STABILIZED_BLEND_HIGH_MIX_EXTRA_FRACTION: f64 = 0.65;
const GAMUT_STABILIZED_BLEND_HIGH_MIX_MIN_TRUSTED_MIX: f64 = 0.91;
const GAMUT_STABILIZED_BLEND_MIN_RENDER_MIDTONE_P50: f64 = 0.30;
const GAMUT_STABILIZED_BLEND_MIN_RENDER_RANGE: f64 = 0.70;
const GAMUT_STABILIZED_BLEND_MIN_STEP: f64 = 0.005;
const MIN_CHANNEL_ANCHOR_SUPPORT: usize = 64;
const MIN_DOMINANT_ANCHOR_STABILITY_BANDS: usize = 2;
const MAX_DOMINANT_ANCHOR_BAND_FRACTION: f64 = 0.92;
const DOMINANT_ANCHOR_MARGIN_TARGET: f64 = 0.35;
const GAMUT_CLIP_EPSILON: f64 = 1e-6;
const NEUTRAL_LUMA_MIN: f64 = 0.03;
const NEUTRAL_LUMA_MAX: f64 = 0.97;
const ANCHOR_LUMA_MIN: f64 = 0.03;
const ANCHOR_LUMA_MAX: f64 = 0.98;
const MIN_NEUTRAL_TRIM_SAMPLES: usize = 64;
const MIN_NEUTRAL_TRIM_BANDS: usize = 2;
const MAX_NEUTRAL_DOMINANT_BAND_FRACTION: f64 = 0.92;
const MIN_NEUTRAL_TRIM_DELTA_IMPROVEMENT: f64 = 0.002;
const NEUTRAL_TRIM_BAND_WORSEN_TOLERANCE: f64 = 0.003;
const CLIPPING_INCREASE_EPSILON: f64 = 1e-6;
const NEUTRAL_TRIM_MIN: f64 = 0.85;
const NEUTRAL_TRIM_MAX: f64 = 1.18;
const CHANNEL_NAMES: [&str; 3] = ["red", "green", "blue"];
const NEUTRAL_BAND_NAMES: [&str; 3] = ["shadow", "midtone", "highlight"];
const COLOR_SCORE_ORDER: &str = "lower_is_better";
const CONDITION_SCORE_REFERENCE: f64 = 10.0;
const NON_FINITE_CONDITION_PENALTY: f64 = 5.0;
const CONDITION_SCORE_WEIGHT: f64 = 0.30;
const LOW_GAMUT_MAX_CHANNEL_PENALTY_WEIGHT: f64 = 12.0;
const LOW_GAMUT_TOTAL_PENALTY_WEIGHT: f64 = 8.0;
const HIGH_GAMUT_TOTAL_PENALTY_WEIGHT: f64 = 1.5;
const PRESERVED_GAMUT_PENALTY_WEIGHT: f64 = 5.0;
const EXPOSURE_PENALTY_WEIGHT: f64 = 0.4;
const NEUTRAL_BALANCE_PENALTY_WEIGHT: f64 = 0.8;
const NEUTRAL_ESTIMATE_PENALTY_WEIGHT: f64 = 0.35;
const ANCHOR_SUPPORT_PENALTY_PER_WEAK_CHANNEL: f64 = 0.55;
const ANCHOR_STABILITY_PENALTY_WEIGHT: f64 = 0.50;
const PROFILE_CONFIDENCE_PENALTY_WEIGHT: f64 = 0.80;
const IMAGE_DERIVED_CONFIDENCE_PENALTY: f64 = 0.18;
const NEUTRAL_FALLBACK_CONFIDENCE_PENALTY: f64 = 0.40;
// Neutral fallback preserves gamut but discards scene chroma, so auto should rank it
// behind any safe matrix candidate and reserve it for forced/failure cases.
const NEUTRAL_FALLBACK_FIDELITY_PENALTY: f64 = 25.0;
const CALIBRATION_DEFAULT_CONFIDENCE_PENALTY: f64 = 0.25;
const TARGET_RESIDUAL_RMS_PENALTY_WEIGHT: f64 = 10.0;
const TARGET_RESIDUAL_MAX_PENALTY_WEIGHT: f64 = 3.0;
const MATRIX_FALLBACK_PENALTY: f64 = 0.50;
pub const COLOR_CANDIDATE_REVIEW_QUALITY_SCORE: f64 = 3.0;
const REFERENCE_RMS_REGRESSION_TOLERANCE: f64 = 0.003;
const REFERENCE_MAX_REGRESSION_TOLERANCE: f64 = 0.010;
const NEUTRAL_REGRESSION_TOLERANCE: f64 = 0.006;
const NEUTRAL_BORDER_MARGIN_FRACTION: f64 = 0.035;
const NEUTRAL_BORDER_MARGIN_MAX: usize = 16;
const NEUTRAL_DUST_LUMA_LOW: f64 = 0.018;
const NEUTRAL_DUST_LUMA_HIGH: f64 = 0.992;
const NEUTRAL_DUST_SATURATION: f64 = 0.70;
const NEUTRAL_BORDER_DARK_LUMA: f64 = 0.040;
const NEUTRAL_BORDER_BRIGHT_LUMA: f64 = 0.960;
const NEUTRAL_FILM_BASE_EDGE_LUMA: f64 = 0.72;
const NEUTRAL_FILM_BASE_EDGE_SATURATION: f64 = 0.20;
const RENDER_TONE_MAX_SAMPLES: usize = 120_000;
const RENDER_TONE_SHADOW_SATURATION_TARGET: f64 = 0.70;
const RENDER_TONE_MIDTONE_SATURATION_TARGET: f64 = 0.85;
const RENDER_TONE_NEUTRAL_HIGHLIGHT_SATURATION_TARGET: f64 = 0.18;
const RENDER_TONE_SATURATED_HIGHLIGHT_LOSS_TOLERANCE: f64 = 0.08;
const RENDER_TONE_SHADOW_SATURATION_WEIGHT: f64 = 0.30;
const RENDER_TONE_MIDTONE_SATURATION_WEIGHT: f64 = 0.15;
const RENDER_TONE_NEUTRAL_HIGHLIGHT_WEIGHT: f64 = 1.20;
const RENDER_TONE_SATURATED_HIGHLIGHT_LOSS_WEIGHT: f64 = 0.80;
const TONE_CHROMA_CLEANUP_RATIO_WEIGHT: f64 = 0.25;
const TONE_CHROMA_CLEANUP_CLIP_WEIGHT: f64 = 6.0;
const TONE_QUALITY_REVIEW_PENALTY: f64 = 1.0;
const MODEL_QUALITY_REVIEW_PENALTY: f64 = 0.35;
const AUTO_NEUTRAL_RESCUE_MIN_PRESERVED_RATIO: f64 = 0.97;
const AUTO_NEUTRAL_RESCUE_MIN_PRESERVED_GAIN: f64 = 0.10;
const AUTO_NEUTRAL_RESCUE_MIN_MIDTONE_SATURATION_REDUCTION: f64 = 0.12;
const MODEL_DIAGNOSTIC_MAX_SAMPLES: usize = 120_000;
const DENSITY_MONOTONICITY_BINS: usize = 24;
const DENSITY_MONOTONICITY_MIN_BIN_SAMPLES: usize = 16;
const DENSITY_MONOTONICITY_TARGET_SCORE: f64 = 0.92;
const DENSITY_MONOTONICITY_TOLERANCE: f64 = 0.025;
const DENSITY_MONOTONICITY_PENALTY_WEIGHT: f64 = 0.50;
const HUE_LINEARITY_MIN_CHANNEL_SAMPLES: usize = 48;
const HUE_LINEARITY_TARGET_SCORE: f64 = 0.35;
const HUE_LINEARITY_PENALTY_WEIGHT: f64 = 0.20;
const SATURATION_PRESERVATION_MIN_SAMPLES: usize = 64;
const SATURATION_PRESERVATION_LOW_TARGET: f64 = 0.55;
const SATURATION_PRESERVATION_HIGH_TARGET: f64 = 2.20;
const SATURATION_PRESERVATION_LOW_P05_TARGET: f64 = 0.20;
const SATURATION_PRESERVATION_PENALTY_WEIGHT: f64 = 0.45;
const MEMORY_COLOR_MIN_FAMILY_SAMPLES: usize = 64;
const MEMORY_COLOR_IMPLAUSIBLE_TARGET: f64 = 0.35;
const MEMORY_COLOR_PENALTY_WEIGHT: f64 = 0.12;
const SPATIAL_CONSISTENCY_TILE_GRID: usize = 8;
const SPATIAL_CONSISTENCY_MIN_TILE_SAMPLES: usize = 12;
const SPATIAL_CONSISTENCY_MIN_TILES: usize = 4;
const SPATIAL_NEUTRAL_DELTA_P95_TARGET: f64 = 0.08;
const SPATIAL_CONSISTENCY_PENALTY_WEIGHT: f64 = 0.45;
const REFERENCE_DELTA_E_RMS_REGRESSION_TOLERANCE: f64 = 1.0;
const REFERENCE_DELTA_E_MAX_REGRESSION_TOLERANCE: f64 = 2.0;
const REFERENCE_DELTA_E2000_RMS_REGRESSION_TOLERANCE: f64 = 0.75;
const REFERENCE_DELTA_E2000_MAX_REGRESSION_TOLERANCE: f64 = 1.50;
const REFERENCE_DELTA_E_RMS_PENALTY_WEIGHT: f64 = 0.020;
const REFERENCE_DELTA_E_MAX_PENALTY_WEIGHT: f64 = 0.005;
const REFERENCE_DELTA_E2000_RMS_PENALTY_WEIGHT: f64 = 0.020;
const REFERENCE_DELTA_E2000_MAX_PENALTY_WEIGHT: f64 = 0.005;
const NONLINEAR_MODEL_MAX_SAMPLES: usize = 120_000;
const NONLINEAR_MODEL_HULL_EXPANSION: f64 = 1.10;
const NONLINEAR_MODEL_MAX_NEGATIVE_INPUT_RATIO: f64 = 0.02;
const NONLINEAR_MODEL_MAX_OUTSIDE_HULL_RATIO: f64 = 0.35;
const NONLINEAR_MODEL_MAX_OUTSIDE_INPUT_DOMAIN_RATIO: f64 = 0.20;
const NONLINEAR_MODEL_MIN_FULL_APPLICATION_RATIO: f64 = 0.65;
const NONLINEAR_MODEL_MIN_SUPPORT_SAMPLES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum ColorMode {
    Auto,
    Calibrated,
    ImageDerived,
    Neutral,
}

impl ColorMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ColorMode::Auto => "auto",
            ColorMode::Calibrated => "calibrated",
            ColorMode::ImageDerived => "image-derived",
            ColorMode::Neutral => "neutral",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ColorMappingQualityComponents {
    pub low_gamut_clip_penalty: f64,
    pub high_gamut_clip_penalty: f64,
    pub preserved_gamut_penalty: f64,
    pub exposure_penalty: f64,
    pub neutral_balance_penalty: f64,
    pub neutral_estimate_penalty: f64,
    pub anchor_support_penalty: f64,
    pub anchor_stability_penalty: f64,
    pub condition_penalty: f64,
    pub calibration_confidence_penalty: f64,
    pub target_residual_penalty: f64,
    pub rendered_tone_penalty: f64,
    pub tone_chroma_cleanup_penalty: f64,
    pub density_monotonicity_penalty: f64,
    pub hue_linearity_penalty: f64,
    pub saturation_preservation_penalty: f64,
    pub memory_color_penalty: f64,
    pub spatial_consistency_penalty: f64,
    pub fallback_penalty: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RenderedToneCandidateDiagnostics {
    pub sample_count: usize,
    pub sample_stride: usize,
    pub tone_fit_domain: &'static str,
    pub post_tone_render_luminance_percentiles: [f64; 3],
    pub post_tone_render_luminance_range: f64,
    pub post_tone_midtone_luminance_p50: f64,
    pub pre_tone_shadow_saturation_p95: f64,
    pub pre_tone_midtone_saturation_p95: f64,
    pub pre_tone_bright_neutral_saturation_p95: f64,
    pub pre_tone_bright_saturated_saturation_median: f64,
    pub shadow_saturation_median: f64,
    pub shadow_saturation_p95: f64,
    pub midtone_saturation_median: f64,
    pub midtone_saturation_p95: f64,
    pub bright_neutral_pixel_count: usize,
    pub bright_neutral_saturation_median: f64,
    pub bright_neutral_saturation_p95: f64,
    pub bright_saturated_pixel_count: usize,
    pub bright_saturated_saturation_median: f64,
    pub bright_saturated_saturation_p95: f64,
    pub saturated_highlight_saturation_loss: f64,
    pub tone_highlight_chroma_compressed_ratio: f64,
    pub tone_highlight_neutral_chroma_compressed_ratio: f64,
    pub tone_shadow_chroma_compressed_ratio: f64,
    pub tone_pre_chroma_clipped_high_total: f64,
    pub tone_post_chroma_clipped_high_total: f64,
    pub tone_post_chroma_clipped_low_total: f64,
    pub rendered_tone_penalty: f64,
    pub tone_chroma_cleanup_penalty: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryColorFamilyDiagnostics {
    pub sample_count: usize,
    pub median_l: Option<f64>,
    pub median_chroma: Option<f64>,
    pub median_hue_degrees: Option<f64>,
    pub implausible_ratio: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryColorCandidateDiagnostics {
    pub skin: MemoryColorFamilyDiagnostics,
    pub foliage: MemoryColorFamilyDiagnostics,
    pub sky: MemoryColorFamilyDiagnostics,
    pub penalty: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpatialConsistencyDiagnostics {
    pub neutral_sample_count: usize,
    pub populated_tile_count: usize,
    pub tile_grid: usize,
    pub neutral_delta_median: Option<f64>,
    pub neutral_delta_p95: Option<f64>,
    pub penalty: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColorModelCandidateDiagnostics {
    pub sample_count: usize,
    pub sample_stride: usize,
    pub density_monotonicity_score: f64,
    pub density_monotonicity_violation_ratio: f64,
    pub density_monotonicity_populated_bins: usize,
    pub hue_linearity_score: f64,
    pub hue_linearity_channel_samples: [usize; 3],
    pub hue_linearity_channel_resultant: [Option<f64>; 3],
    pub saturation_preservation_sample_count: usize,
    pub saturation_preservation_p05_ratio: Option<f64>,
    pub saturation_preservation_median_ratio: Option<f64>,
    pub saturation_preservation_p95_ratio: Option<f64>,
    pub memory_color: MemoryColorCandidateDiagnostics,
    pub spatial_consistency: SpatialConsistencyDiagnostics,
    pub density_monotonicity_penalty: f64,
    pub hue_linearity_penalty: f64,
    pub saturation_preservation_penalty: f64,
    pub memory_color_penalty: f64,
    pub spatial_consistency_penalty: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColorMappingCandidateScore {
    pub candidate: &'static str,
    pub mapping_strategy: &'static str,
    pub rank: Option<usize>,
    pub selected: bool,
    pub selected_quality_delta: Option<f64>,
    pub score: f64,
    pub quality_score: f64,
    pub technical_safety_score: f64,
    pub color_fidelity_score: f64,
    pub rejected: bool,
    pub rejection_reason: Option<String>,
    pub pre_scale_clipped_low_ratio: [f64; 3],
    pub pre_scale_clipped_low_max: f64,
    pub pre_scale_clipped_low_total: f64,
    pub pre_scale_clipped_high_ratio: [f64; 3],
    pub pre_scale_clipped_high_max: f64,
    pub pre_scale_clipped_high_total: f64,
    pub pre_scale_channel_min: [f64; 3],
    pub pre_scale_channel_max: [f64; 3],
    pub pre_scale_max_low_excursion: [f64; 3],
    pub pre_scale_max_high_excursion: [f64; 3],
    pub pre_scale_preserved_ratio: f64,
    pub exposure_scale: f64,
    pub neutral_balance_delta: [f64; 3],
    pub neutral_delta_magnitude: f64,
    pub neutral_estimate_quality_score: f64,
    pub channel_anchor_counts: [usize; 3],
    pub channel_anchor_low_support: [bool; 3],
    pub dominant_anchor_quality_score: f64,
    pub dominant_anchor_unstable_channels: [bool; 3],
    pub condition_number: f64,
    pub profile_confidence: Option<f64>,
    pub target_residual_rms: Option<f64>,
    pub target_residual_max: Option<f64>,
    pub reference_patch_rms_error: Option<f64>,
    pub reference_patch_max_error: Option<f64>,
    pub reference_patch_mean_delta_e: Option<f64>,
    pub reference_patch_rms_delta_e: Option<f64>,
    pub reference_patch_max_delta_e: Option<f64>,
    pub reference_patch_mean_delta_e2000: Option<f64>,
    pub reference_patch_rms_delta_e2000: Option<f64>,
    pub reference_patch_max_delta_e2000: Option<f64>,
    pub reference_patch_delta_vs_image_derived: Option<f64>,
    pub reference_patch_max_delta_vs_image_derived: Option<f64>,
    pub reference_patch_delta_e_delta_vs_image_derived: Option<f64>,
    pub reference_patch_delta_e_max_delta_vs_image_derived: Option<f64>,
    pub reference_patch_delta_e2000_delta_vs_image_derived: Option<f64>,
    pub reference_patch_delta_e2000_max_delta_vs_image_derived: Option<f64>,
    pub reference_patch_regresses_image_derived: Option<bool>,
    pub rendered_tone_quality: Option<RenderedToneCandidateDiagnostics>,
    pub color_model_quality: Option<ColorModelCandidateDiagnostics>,
    pub quality_components: ColorMappingQualityComponents,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColorMappingCandidateAcceptanceDiagnostics {
    pub candidate: String,
    pub candidate_kind: String,
    pub mapping_strategy: String,
    pub source_label: String,
    pub status: String,
    pub reason: String,
    pub rank: Option<usize>,
    pub selected: bool,
    pub eligible_in_color_mode: bool,
    pub quality_score: f64,
    pub selected_quality_delta: Option<f64>,
    pub rejected: bool,
    pub rejection_reason: Option<String>,
    pub beats_image_derived: Option<bool>,
    pub within_negative_gamut_limits: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NeutralEstimateQuality {
    pub score: f64,
    pub accepted: bool,
    pub broad_support: bool,
    pub reason: String,
    pub sample_count: usize,
    pub sample_fraction: f64,
    pub band_counts: [usize; 3],
    pub populated_band_count: usize,
    pub dominant_band_fraction: f64,
    pub minimum_samples: usize,
    pub minimum_bands: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct NeutralSampleRejections {
    pub total_pixels: usize,
    pub accepted_neutral_samples: usize,
    pub non_finite: usize,
    pub clipped: usize,
    pub border: usize,
    pub film_base_like_edge: usize,
    pub dust: usize,
    pub luma_out_of_range: usize,
    pub chroma_threshold: usize,
}

impl NeutralSampleRejections {
    fn empty(total_pixels: usize) -> Self {
        Self {
            total_pixels,
            accepted_neutral_samples: 0,
            non_finite: 0,
            clipped: 0,
            border: 0,
            film_base_like_edge: 0,
            dust: 0,
            luma_out_of_range: 0,
            chroma_threshold: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DominantAnchorSampleRejections {
    pub total_pixels: usize,
    pub accepted_anchor_samples: usize,
    pub non_finite: usize,
    pub clipped: usize,
    pub border: usize,
    pub film_base_like_edge: usize,
    pub dust: usize,
    pub luma_out_of_range: usize,
    pub low_saturation: usize,
    pub weak_dominance: usize,
}

impl DominantAnchorSampleRejections {
    fn empty(total_pixels: usize) -> Self {
        Self {
            total_pixels,
            accepted_anchor_samples: 0,
            non_finite: 0,
            clipped: 0,
            border: 0,
            film_base_like_edge: 0,
            dust: 0,
            luma_out_of_range: 0,
            low_saturation: 0,
            weak_dominance: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DominantAnchorQuality {
    pub score: f64,
    pub accepted: bool,
    pub reason: String,
    pub channel_populated_band_count: [usize; 3],
    pub channel_dominant_band_fraction: [f64; 3],
    pub channel_mean_dominance_margin: [f64; 3],
    pub channel_stability_score: [f64; 3],
    pub channel_unstable: [bool; 3],
    pub unstable_channel_count: usize,
    pub minimum_samples_per_channel: usize,
    pub minimum_bands: usize,
    pub max_dominant_band_fraction: f64,
    pub dominance_margin_target: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReferencePatchResidual {
    pub patch_id: Option<String>,
    pub hue_family: String,
    pub source_rgb: [f64; 3],
    pub reference_xyz: [f64; 3],
    pub selected_xyz: [f64; 3],
    pub image_derived_xyz: [f64; 3],
    pub selected_error: f64,
    pub image_derived_error: f64,
    pub selected_delta_e: f64,
    pub image_derived_delta_e: f64,
    pub selected_delta_e2000: f64,
    pub image_derived_delta_e2000: f64,
    pub selected_delta_vs_image_derived: f64,
    pub selected_delta_e_vs_image_derived: f64,
    pub selected_delta_e2000_vs_image_derived: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReferenceHueFamilyResidual {
    pub hue_family: String,
    pub patch_count: usize,
    pub rms_error: f64,
    pub max_error: f64,
    pub rms_delta_e: f64,
    pub max_delta_e: f64,
    pub rms_delta_e2000: f64,
    pub max_delta_e2000: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReferenceHueFamilyRegression {
    pub hue_family: String,
    pub patch_count: usize,
    pub candidate_rms_error: f64,
    pub image_derived_rms_error: f64,
    pub rms_delta_vs_image_derived: f64,
    pub candidate_max_error: f64,
    pub image_derived_max_error: f64,
    pub max_delta_vs_image_derived: f64,
    pub candidate_rms_delta_e: f64,
    pub image_derived_rms_delta_e: f64,
    pub delta_e_rms_delta_vs_image_derived: f64,
    pub candidate_max_delta_e: f64,
    pub image_derived_max_delta_e: f64,
    pub delta_e_max_delta_vs_image_derived: f64,
    pub candidate_rms_delta_e2000: f64,
    pub image_derived_rms_delta_e2000: f64,
    pub delta_e2000_rms_delta_vs_image_derived: f64,
    pub candidate_max_delta_e2000: f64,
    pub image_derived_max_delta_e2000: f64,
    pub delta_e2000_max_delta_vs_image_derived: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReferenceCandidatePatchEvaluation {
    pub candidate: String,
    pub patch_count: usize,
    pub rms_error: f64,
    pub max_error: f64,
    pub mean_error: f64,
    pub rms_delta_e: f64,
    pub max_delta_e: f64,
    pub mean_delta_e: f64,
    pub rms_delta_e2000: f64,
    pub max_delta_e2000: f64,
    pub mean_delta_e2000: f64,
    pub rms_delta_vs_image_derived: Option<f64>,
    pub max_delta_vs_image_derived: Option<f64>,
    pub delta_e_rms_delta_vs_image_derived: Option<f64>,
    pub delta_e_max_delta_vs_image_derived: Option<f64>,
    pub delta_e2000_rms_delta_vs_image_derived: Option<f64>,
    pub delta_e2000_max_delta_vs_image_derived: Option<f64>,
    pub regresses_image_derived: Option<bool>,
    pub worst_hue_families: Vec<ReferenceHueFamilyResidual>,
    pub hue_family_regressions: Vec<ReferenceHueFamilyRegression>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReferencePatchEvaluation {
    pub patch_count: usize,
    pub selected_candidate: String,
    pub image_derived_candidate: String,
    pub selected_rms_error: f64,
    pub selected_max_error: f64,
    pub image_derived_rms_error: f64,
    pub image_derived_max_error: f64,
    pub selected_rms_delta_e: f64,
    pub selected_max_delta_e: f64,
    pub image_derived_rms_delta_e: f64,
    pub image_derived_max_delta_e: f64,
    pub selected_rms_delta_e2000: f64,
    pub selected_max_delta_e2000: f64,
    pub image_derived_rms_delta_e2000: f64,
    pub image_derived_max_delta_e2000: f64,
    pub rms_error_delta_vs_image_derived: f64,
    pub max_error_delta_vs_image_derived: f64,
    pub delta_e_rms_delta_vs_image_derived: f64,
    pub delta_e_max_delta_vs_image_derived: f64,
    pub delta_e2000_rms_delta_vs_image_derived: f64,
    pub delta_e2000_max_delta_vs_image_derived: f64,
    pub selected_improves_image_derived: bool,
    pub selected_regresses_image_derived: bool,
    pub worst_hue_families: Vec<ReferenceHueFamilyResidual>,
    pub selected_hue_family_regressions: Vec<ReferenceHueFamilyRegression>,
    pub per_patch: Vec<ReferencePatchResidual>,
    pub candidate_evaluations: Vec<ReferenceCandidatePatchEvaluation>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NeutralTrimStats {
    pub neutral_balance_delta: [f64; 3],
    pub neutral_delta_magnitude: f64,
    pub neutral_band_names: [&'static str; 3],
    pub neutral_band_balance_delta: [Option<[f64; 3]>; 3],
    pub neutral_band_delta_magnitude: [Option<f64>; 3],
    pub pre_scale_clipped_low_ratio: [f64; 3],
    pub pre_scale_clipped_high_ratio: [f64; 3],
    pub pre_scale_preserved_ratio: f64,
    pub exposure_scale: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct NeutralTrimDiagnostics {
    pub applied: bool,
    pub scale: [f64; 3],
    pub reason: String,
    pub before: Option<NeutralTrimStats>,
    pub after: Option<NeutralTrimStats>,
    pub neutral_delta_reduced: bool,
    pub neutral_band_delta_worsened: bool,
    pub low_clipping_increased: bool,
    pub high_clipping_increased: bool,
    pub preserved_ratio_decreased: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationAcceptanceDiagnostics {
    pub status: String,
    pub reason: String,
    pub color_mode: &'static str,
    pub preferred_candidate: Option<String>,
    pub preferred_candidate_quality_score: Option<f64>,
    pub image_derived_quality_score: Option<f64>,
    pub beats_image_derived: Option<bool>,
    pub within_negative_gamut_limits: Option<bool>,
    pub forced_by_color_mode: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct NeutralSafetyRescueDiagnostics {
    pub evaluated: bool,
    pub applied: bool,
    pub matrix_candidate: Option<String>,
    pub matrix_candidate_kind: Option<String>,
    pub matrix_anchor_evidence_supported: Option<bool>,
    pub neutral_estimate_supported: bool,
    pub neutral_model_evidence_supported: bool,
    pub matrix_pre_scale_preserved_ratio: Option<f64>,
    pub neutral_pre_scale_preserved_ratio: f64,
    pub preserved_ratio_gain: Option<f64>,
    pub minimum_preserved_ratio: f64,
    pub minimum_preserved_ratio_gain: f64,
    pub matrix_midtone_saturation_p95: Option<f64>,
    pub neutral_midtone_saturation_p95: Option<f64>,
    pub midtone_saturation_p95_reduction: Option<f64>,
    pub maximum_midtone_saturation_p95: f64,
    pub minimum_midtone_saturation_p95_reduction: f64,
    pub matrix_memory_color_penalty: Option<f64>,
    pub neutral_memory_color_penalty: f64,
    pub matrix_spatial_consistency_penalty: Option<f64>,
    pub neutral_spatial_consistency_penalty: f64,
    pub neutral_saturation_preservation_sample_count: usize,
    pub neutral_saturation_preservation_p05_ratio: Option<f64>,
    pub neutral_saturation_preservation_median_ratio: Option<f64>,
    pub neutral_saturation_preservation_p95_ratio: Option<f64>,
    pub reason: String,
}

impl NeutralSafetyRescueDiagnostics {
    pub fn not_evaluated(reason: impl Into<String>) -> Self {
        Self {
            evaluated: false,
            applied: false,
            matrix_candidate: None,
            matrix_candidate_kind: None,
            matrix_anchor_evidence_supported: None,
            neutral_estimate_supported: false,
            neutral_model_evidence_supported: false,
            matrix_pre_scale_preserved_ratio: None,
            neutral_pre_scale_preserved_ratio: 0.0,
            preserved_ratio_gain: None,
            minimum_preserved_ratio: AUTO_NEUTRAL_RESCUE_MIN_PRESERVED_RATIO,
            minimum_preserved_ratio_gain: AUTO_NEUTRAL_RESCUE_MIN_PRESERVED_GAIN,
            matrix_midtone_saturation_p95: None,
            neutral_midtone_saturation_p95: None,
            midtone_saturation_p95_reduction: None,
            maximum_midtone_saturation_p95: RENDER_TONE_MIDTONE_SATURATION_TARGET,
            minimum_midtone_saturation_p95_reduction:
                AUTO_NEUTRAL_RESCUE_MIN_MIDTONE_SATURATION_REDUCTION,
            matrix_memory_color_penalty: None,
            neutral_memory_color_penalty: 0.0,
            matrix_spatial_consistency_penalty: None,
            neutral_spatial_consistency_penalty: 0.0,
            neutral_saturation_preservation_sample_count: 0,
            neutral_saturation_preservation_p05_ratio: None,
            neutral_saturation_preservation_median_ratio: None,
            neutral_saturation_preservation_p95_ratio: None,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct NonlinearColorModelRuntimeDiagnostics {
    pub model_id: String,
    pub model_type: &'static str,
    pub basis: String,
    pub degree: Option<u8>,
    pub grid_size: Option<u8>,
    pub baseline_kind: String,
    pub held_out_delta_e00_rms: f64,
    pub held_out_delta_e00_max: f64,
    pub matrix_held_out_delta_e00_rms: f64,
    pub baseline_held_out_delta_e00_rms: f64,
    pub sampled_pixel_count: usize,
    pub negative_input_ratio: f64,
    pub outside_training_chromaticity_hull_ratio: f64,
    pub outside_training_input_domain_ratio: f64,
    pub full_model_application_ratio: f64,
    pub support_status: String,
    pub support_reason: String,
}

#[derive(Debug, Clone)]
pub struct ColorspaceDiagnostics {
    pub source_white: [f64; 3],
    pub work_to_xyz: [[f64; 3]; 3],
    pub condition_number: f64,
    pub regularization_lambda: f64,
    pub neutral_pixel_count: usize,
    pub channel_anchor_counts: [usize; 3],
    pub channel_anchor_min_count: usize,
    pub channel_anchor_low_support_threshold: usize,
    pub channel_anchor_low_support: [bool; 3],
    pub dominant_anchor_rgb: [[f64; 3]; 3],
    pub dominant_anchor_sample_rejections: DominantAnchorSampleRejections,
    pub dominant_anchor_bands: [[usize; 3]; 3],
    pub dominant_anchor_quality: DominantAnchorQuality,
    pub neutral_sample_bands: [usize; 3],
    pub highlight_percentile: f64,
    pub pre_scale_channel_max: [f64; 3],
    pub pre_scale_channel_high_percentile: [f64; 3],
    pub pre_scale_clipped_high_ratio: [f64; 3],
    pub pre_scale_clipped_low_ratio: [f64; 3],
    pub post_scale_clipped_high_ratio: [f64; 3],
    pub post_scale_clipped_low_ratio: [f64; 3],
    pub exposure_scale: f64,
    pub fallback_used: bool,
    pub weak_anchor_fallback_used: bool,
    pub weak_anchor_fallback_reason: Option<String>,
    pub gamut_fallback_used: bool,
    pub gamut_fallback_reason: Option<String>,
    pub mapping_strategy: &'static str,
    pub selected_mapping_reason: String,
    pub candidate_scores: Vec<ColorMappingCandidateScore>,
    pub candidate_acceptance: Vec<ColorMappingCandidateAcceptanceDiagnostics>,
    pub selected_candidate: String,
    pub selected_candidate_rank: Option<usize>,
    pub selected_candidate_score: Option<f64>,
    pub selected_quality_score: Option<f64>,
    pub technical_safety_score: Option<f64>,
    pub color_fidelity_score: Option<f64>,
    pub selected_runner_up_quality_delta: Option<f64>,
    pub selection_rejections: Vec<String>,
    pub candidate_risk: String,
    pub neutral_sample_rejections: NeutralSampleRejections,
    pub reference_patch_evaluation: Option<ReferencePatchEvaluation>,
    pub neutral_estimate_quality: NeutralEstimateQuality,
    pub neutral_balance_scale: [f64; 3],
    pub neutral_trim_scale: [f64; 3],
    pub neutral_trim_applied: bool,
    pub neutral_trim_before_after: NeutralTrimDiagnostics,
    pub calibration_acceptance: CalibrationAcceptanceDiagnostics,
    pub neutral_safety_rescue: NeutralSafetyRescueDiagnostics,
    pub nonlinear_color_model: Option<NonlinearColorModelRuntimeDiagnostics>,
    pub pre_scale_preserved_ratio: f64,
    pub post_scale_preserved_ratio: f64,
    pub image_matrix_pre_scale_clipped_low_ratio: [f64; 3],
    pub image_matrix_pre_scale_clipped_high_ratio: [f64; 3],
    pub image_matrix_exposure_scale: f64,
    pub image_matrix_pre_scale_preserved_ratio: f64,
    pub image_matrix_neutral_balance_delta: [f64; 3],
    pub calibrated_profile_pre_scale_clipped_low_ratio: Option<[f64; 3]>,
    pub calibrated_profile_pre_scale_clipped_high_ratio: Option<[f64; 3]>,
    pub calibrated_profile_exposure_scale: Option<f64>,
    pub calibrated_profile_pre_scale_preserved_ratio: Option<f64>,
    pub calibrated_profile_neutral_balance_delta: Option<[f64; 3]>,
}

pub struct ColorspaceMappingResult {
    pub prophoto: Array3<f64>,
    pub diagnostics: ColorspaceDiagnostics,
    pub comparison_thumbnail: Option<Array3<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GamutClippingMapDiagnostics {
    pub encoding: &'static str,
    pub high_clipped_ratio: [f64; 3],
    pub low_clipped_ratio: [f64; 3],
    pub any_clipped_ratio: f64,
    pub preserved_ratio: f64,
    pub max_high_excursion: [f64; 3],
    pub max_low_excursion: [f64; 3],
    pub exposure_scale: f64,
}

fn channel_anchor_low_support(counts: [usize; 3]) -> [bool; 3] {
    std::array::from_fn(|c| counts[c] < MIN_CHANNEL_ANCHOR_SUPPORT)
}

fn channel_anchor_min_count(counts: [usize; 3]) -> usize {
    counts.iter().copied().min().unwrap_or(0)
}

pub fn weak_channel_anchor_warning(diagnostics: &ColorspaceDiagnostics) -> Option<String> {
    if !diagnostics
        .channel_anchor_low_support
        .iter()
        .any(|low_support| *low_support)
    {
        return None;
    }

    let weak_channels = diagnostics
        .channel_anchor_low_support
        .iter()
        .enumerate()
        .filter(|(_, low_support)| **low_support)
        .map(|(idx, _)| {
            format!(
                "{}={}",
                CHANNEL_NAMES[idx], diagnostics.channel_anchor_counts[idx]
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    Some(format!(
        "colorspace matrix has weak dominant-channel anchor support ({weak_channels}; threshold {}); matrix-derived color changes should be treated conservatively",
        diagnostics.channel_anchor_low_support_threshold
    ))
}

/// Helper to convert a `[[f64; 3]; 3]` constant into a `nalgebra::Matrix3<f64>`.
/// nalgebra stores columns, so we transpose the row-major constant.
fn const_to_matrix3(rows: &[[f64; 3]; 3]) -> Matrix3<f64> {
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

#[derive(Debug, Clone)]
struct NeutralEstimate {
    rgb: [f64; 3],
    band_rgb: [[f64; 3]; 3],
    count: usize,
    bands: [usize; 3],
    quality: NeutralEstimateQuality,
    rejections: NeutralSampleRejections,
}

impl NeutralTrimDiagnostics {
    fn skipped(reason: impl Into<String>) -> Self {
        Self {
            applied: false,
            scale: [1.0; 3],
            reason: reason.into(),
            before: None,
            after: None,
            neutral_delta_reduced: false,
            neutral_band_delta_worsened: false,
            low_clipping_increased: false,
            high_clipping_increased: false,
            preserved_ratio_decreased: false,
        }
    }
}

impl CalibrationAcceptanceDiagnostics {
    fn not_applicable(color_mode: ColorMode, reason: impl Into<String>) -> Self {
        Self {
            status: "not_applicable".to_string(),
            reason: reason.into(),
            color_mode: color_mode.as_str(),
            preferred_candidate: None,
            preferred_candidate_quality_score: None,
            image_derived_quality_score: None,
            beats_image_derived: None,
            within_negative_gamut_limits: None,
            forced_by_color_mode: color_mode != ColorMode::Auto,
        }
    }
}

/// ProPhoto RGB -> XYZ (D50) conversion matrix.
pub fn prophoto_to_xyz_d50_matrix() -> Matrix3<f64> {
    const_to_matrix3(&PROPHOTO_TO_XYZ_D50)
}

/// XYZ (D50) -> ProPhoto RGB conversion matrix.
pub fn xyz_d50_to_prophoto_matrix() -> Matrix3<f64> {
    const_to_matrix3(&XYZ_D50_TO_PROPHOTO)
}

/// Compute a Bradford chromatic adaptation transform from `source_white` to D50.
pub fn bradford_cat(source_white: &[f64; 3]) -> Matrix3<f64> {
    let m_to_lms = const_to_matrix3(&BRADFORD_XYZ_TO_LMS);
    let m_to_xyz = const_to_matrix3(&BRADFORD_LMS_TO_XYZ);

    let src = Vector3::new(source_white[0], source_white[1], source_white[2]);
    let d50 = Vector3::new(D50_WHITE[0], D50_WHITE[1], D50_WHITE[2]);

    let lms_src = m_to_lms * src;
    let lms_d50 = m_to_lms * d50;

    let diag = Matrix3::new(
        lms_d50[0] / lms_src[0].max(1e-6),
        0.0,
        0.0,
        0.0,
        lms_d50[1] / lms_src[1].max(1e-6),
        0.0,
        0.0,
        0.0,
        lms_d50[2] / lms_src[2].max(1e-6),
    );

    m_to_xyz * diag * m_to_lms
}

fn neutral_luminance_band(mu: f64) -> usize {
    if mu < 0.30 {
        0
    } else if mu < 0.70 {
        1
    } else {
        2
    }
}

fn neutral_border_margin(h: usize, w: usize) -> usize {
    let shortest = h.min(w);
    ((shortest as f64 * NEUTRAL_BORDER_MARGIN_FRACTION).round() as usize)
        .clamp(1, NEUTRAL_BORDER_MARGIN_MAX)
}

fn is_border_pixel(y: usize, x: usize, h: usize, w: usize, margin: usize) -> bool {
    y < margin || x < margin || y + margin >= h || x + margin >= w
}

fn estimate_neutral_stats_detailed(img: &Array3<f64>) -> NeutralEstimate {
    let (h, w, _) = img.dim();
    let total = h * w;
    let border_margin = neutral_border_margin(h, w);
    let mut rejections = NeutralSampleRejections::empty(total);
    let mut band_sums = [[0.0f64; 3]; 3];
    let mut band_counts = [0usize; 3];

    for y in 0..h {
        for x in 0..w {
            let r = img[[y, x, 0]];
            let g = img[[y, x, 1]];
            let b = img[[y, x, 2]];
            let pixel = [r, g, b];
            if pixel.iter().any(|value| !value.is_finite()) {
                rejections.non_finite += 1;
                continue;
            }
            if pixel.iter().any(|value| *value <= 1e-6 || *value >= 0.999) {
                rejections.clipped += 1;
                continue;
            }

            let mu = (r + g + b) / 3.0;
            let max_v = r.max(g).max(b);
            let min_v = r.min(g).min(b);
            let saturation = if max_v <= 1e-9 {
                0.0
            } else {
                (max_v - min_v) / max_v
            };
            let border_pixel = is_border_pixel(y, x, h, w, border_margin);
            if ((mu <= NEUTRAL_DUST_LUMA_LOW || mu >= NEUTRAL_DUST_LUMA_HIGH)
                || (max_v >= 0.990 && min_v <= 0.030))
                && saturation >= NEUTRAL_DUST_SATURATION
            {
                rejections.dust += 1;
                continue;
            }
            if border_pixel && (mu <= NEUTRAL_BORDER_DARK_LUMA || mu >= NEUTRAL_BORDER_BRIGHT_LUMA)
            {
                rejections.border += 1;
                continue;
            }
            if border_pixel
                && mu >= NEUTRAL_FILM_BASE_EDGE_LUMA
                && saturation <= NEUTRAL_FILM_BASE_EDGE_SATURATION
            {
                rejections.film_base_like_edge += 1;
                continue;
            }
            if !(NEUTRAL_LUMA_MIN..=NEUTRAL_LUMA_MAX).contains(&mu) {
                rejections.luma_out_of_range += 1;
                continue;
            }

            let max_dev = (r - mu).abs().max((g - mu).abs()).max((b - mu).abs());
            if max_dev / mu <= NEUTRAL_THRESHOLD {
                let band = neutral_luminance_band(mu);
                band_sums[band][0] += r;
                band_sums[band][1] += g;
                band_sums[band][2] += b;
                band_counts[band] += 1;
            } else {
                rejections.chroma_threshold += 1;
            }
        }
    }

    let count = band_counts.iter().sum::<usize>();
    rejections.accepted_neutral_samples = count;
    let sample_fraction = if total == 0 {
        0.0
    } else {
        count as f64 / total as f64
    };
    let populated_bands = band_counts.iter().filter(|count| **count > 0).count();
    let dominant_band_fraction = if count == 0 {
        0.0
    } else {
        band_counts.iter().copied().max().unwrap_or(0) as f64 / count as f64
    };
    let minimum_fraction_count = (total as f64 * MIN_NEUTRAL_FRACTION).ceil() as usize;
    let minimum_samples = MIN_NEUTRAL_TRIM_SAMPLES.max(minimum_fraction_count.max(1));
    let enough_samples = count >= minimum_samples;
    let broad_support = populated_bands >= MIN_NEUTRAL_TRIM_BANDS
        && dominant_band_fraction <= MAX_NEUTRAL_DOMINANT_BAND_FRACTION;
    let accepted = enough_samples && broad_support;
    let sample_score = if minimum_samples == 0 {
        0.0
    } else {
        (count as f64 / minimum_samples as f64).clamp(0.0, 1.0)
    };
    let band_score = (populated_bands as f64 / 3.0).clamp(0.0, 1.0);
    let balance_score = if count == 0 {
        0.0
    } else {
        ((1.0 - dominant_band_fraction) / (1.0 - 1.0 / 3.0)).clamp(0.0, 1.0)
    };
    let quality_score =
        (0.45 * sample_score + 0.35 * band_score + 0.20 * balance_score).clamp(0.0, 1.0);
    let reason = if !enough_samples {
        format!(
            "neutral estimate rejected: {} neutral pixels below required {}",
            count, minimum_samples
        )
    } else if populated_bands < MIN_NEUTRAL_TRIM_BANDS {
        format!(
            "neutral estimate weak: support appears in {} luminance band(s), require {}",
            populated_bands, MIN_NEUTRAL_TRIM_BANDS
        )
    } else if dominant_band_fraction > MAX_NEUTRAL_DOMINANT_BAND_FRACTION {
        format!(
            "neutral estimate weak: {:.1}% of neutral samples are concentrated in one luminance band",
            dominant_band_fraction * 100.0
        )
    } else {
        "neutral estimate accepted with broad luminance-band support".to_string()
    };
    let quality = NeutralEstimateQuality {
        score: quality_score,
        accepted,
        broad_support,
        reason,
        sample_count: count,
        sample_fraction,
        band_counts,
        populated_band_count: populated_bands,
        dominant_band_fraction,
        minimum_samples,
        minimum_bands: MIN_NEUTRAL_TRIM_BANDS,
    };
    let band_rgb = std::array::from_fn(|band| {
        if band_counts[band] == 0 {
            [1.0, 1.0, 1.0]
        } else {
            let n = band_counts[band] as f64;
            [
                band_sums[band][0] / n,
                band_sums[band][1] / n,
                band_sums[band][2] / n,
            ]
        }
    });

    if count == 0 || (count as f64) < (total as f64 * MIN_NEUTRAL_FRACTION) {
        return NeutralEstimate {
            rgb: [1.0, 1.0, 1.0],
            band_rgb,
            count,
            bands: band_counts,
            quality,
            rejections,
        };
    }

    let per_band_cap = (count as f64 / populated_bands.max(1) as f64).max(1.0);
    let mut weighted_sum = [0.0f64; 3];
    let mut total_weight = 0.0f64;
    for band in 0..3 {
        if band_counts[band] == 0 {
            continue;
        }
        let n = band_counts[band] as f64;
        let weight = n.min(per_band_cap);
        weighted_sum[0] += (band_sums[band][0] / n) * weight;
        weighted_sum[1] += (band_sums[band][1] / n) * weight;
        weighted_sum[2] += (band_sums[band][2] / n) * weight;
        total_weight += weight;
    }

    let total_weight = total_weight.max(1e-6);
    NeutralEstimate {
        rgb: [
            weighted_sum[0] / total_weight,
            weighted_sum[1] / total_weight,
            weighted_sum[2] / total_weight,
        ],
        band_rgb,
        count,
        bands: band_counts,
        quality,
        rejections,
    }
}

#[derive(Debug, Clone)]
struct DominantAnchorEstimate {
    rgb: [[f64; 3]; 3],
    counts: [usize; 3],
    bands: [[usize; 3]; 3],
    rejections: DominantAnchorSampleRejections,
    quality: DominantAnchorQuality,
}

fn dominant_anchor_quality(
    counts: [usize; 3],
    bands: [[usize; 3]; 3],
    dominance_margin_sums: [f64; 3],
) -> DominantAnchorQuality {
    let channel_populated_band_count =
        std::array::from_fn(|channel| bands[channel].iter().filter(|count| **count > 0).count());
    let channel_dominant_band_fraction = std::array::from_fn(|channel| {
        if counts[channel] == 0 {
            0.0
        } else {
            bands[channel].iter().copied().max().unwrap_or(0) as f64 / counts[channel] as f64
        }
    });
    let channel_mean_dominance_margin = std::array::from_fn(|channel| {
        if counts[channel] == 0 {
            0.0
        } else {
            dominance_margin_sums[channel] / counts[channel] as f64
        }
    });
    let channel_stability_score = std::array::from_fn(|channel| {
        let sample_score =
            (counts[channel] as f64 / MIN_CHANNEL_ANCHOR_SUPPORT as f64).clamp(0.0, 1.0);
        let band_score = (channel_populated_band_count[channel] as f64
            / MIN_DOMINANT_ANCHOR_STABILITY_BANDS as f64)
            .clamp(0.0, 1.0);
        let balance_score = if counts[channel] == 0 {
            0.0
        } else {
            ((1.0 - channel_dominant_band_fraction[channel]) / (1.0 - 1.0 / 3.0)).clamp(0.0, 1.0)
        };
        let margin_score = (channel_mean_dominance_margin[channel] / DOMINANT_ANCHOR_MARGIN_TARGET)
            .clamp(0.0, 1.0);
        (0.35 * sample_score + 0.25 * band_score + 0.20 * balance_score + 0.20 * margin_score)
            .clamp(0.0, 1.0)
    });
    let minimum_stable_margin = (DOMINANT_RATIO_THRESHOLD - 1.0 + 0.04).max(0.0);
    let channel_unstable = std::array::from_fn(|channel| {
        counts[channel] < MIN_CHANNEL_ANCHOR_SUPPORT
            || channel_populated_band_count[channel] < MIN_DOMINANT_ANCHOR_STABILITY_BANDS
            || channel_dominant_band_fraction[channel] > MAX_DOMINANT_ANCHOR_BAND_FRACTION
            || channel_mean_dominance_margin[channel] < minimum_stable_margin
    });
    let unstable_channel_count = channel_unstable
        .iter()
        .filter(|unstable| **unstable)
        .count();
    let score = channel_stability_score.iter().sum::<f64>() / 3.0;
    let accepted = unstable_channel_count == 0;
    let reason = if accepted {
        "dominant-channel anchors accepted with stable per-channel support".to_string()
    } else {
        let unstable = channel_unstable
            .iter()
            .enumerate()
            .filter(|(_, unstable)| **unstable)
            .map(|(channel, _)| {
                format!(
                    "{} count={} bands={} dominant_band={:.1}% mean_margin={:.3}",
                    CHANNEL_NAMES[channel],
                    counts[channel],
                    channel_populated_band_count[channel],
                    channel_dominant_band_fraction[channel] * 100.0,
                    channel_mean_dominance_margin[channel]
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        format!("dominant-channel anchors unstable: {unstable}")
    };

    DominantAnchorQuality {
        score,
        accepted,
        reason,
        channel_populated_band_count,
        channel_dominant_band_fraction,
        channel_mean_dominance_margin,
        channel_stability_score,
        channel_unstable,
        unstable_channel_count,
        minimum_samples_per_channel: MIN_CHANNEL_ANCHOR_SUPPORT,
        minimum_bands: MIN_DOMINANT_ANCHOR_STABILITY_BANDS,
        max_dominant_band_fraction: MAX_DOMINANT_ANCHOR_BAND_FRACTION,
        dominance_margin_target: DOMINANT_ANCHOR_MARGIN_TARGET,
    }
}

fn estimate_channel_anchors(img: &Array3<f64>) -> DominantAnchorEstimate {
    let (h, w, _) = img.dim();
    let total = h * w;
    let border_margin = neutral_border_margin(h, w);
    let mut rejections = DominantAnchorSampleRejections::empty(total);
    let mut band_sums = [[[0.0f64; 3]; 3]; 3];
    let mut band_counts = [[0usize; 3]; 3];
    let mut counts = [0usize; 3];
    let mut dominance_margin_sums = [0.0f64; 3];

    for y in 0..h {
        for x in 0..w {
            let pixel = [img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]]];
            if pixel.iter().any(|value| !value.is_finite()) {
                rejections.non_finite += 1;
                continue;
            }
            if pixel.iter().any(|value| *value <= 1e-6 || *value >= 0.999) {
                rejections.clipped += 1;
                continue;
            }

            let max_v = pixel[0].max(pixel[1]).max(pixel[2]);
            let min_v = pixel[0].min(pixel[1]).min(pixel[2]);
            let luma = (pixel[0] + pixel[1] + pixel[2]) / 3.0;
            let saturation = if max_v <= 1e-9 {
                0.0
            } else {
                (max_v - min_v) / max_v
            };
            let border_pixel = is_border_pixel(y, x, h, w, border_margin);
            if ((luma <= NEUTRAL_DUST_LUMA_LOW || luma >= NEUTRAL_DUST_LUMA_HIGH)
                || (max_v >= 0.990 && min_v <= 0.030))
                && saturation >= NEUTRAL_DUST_SATURATION
            {
                rejections.dust += 1;
                continue;
            }
            if border_pixel
                && (luma <= NEUTRAL_BORDER_DARK_LUMA || luma >= NEUTRAL_BORDER_BRIGHT_LUMA)
            {
                rejections.border += 1;
                continue;
            }
            if border_pixel
                && luma >= NEUTRAL_FILM_BASE_EDGE_LUMA
                && saturation <= NEUTRAL_FILM_BASE_EDGE_SATURATION
            {
                rejections.film_base_like_edge += 1;
                continue;
            }
            if max_v < 1e-6 || !(ANCHOR_LUMA_MIN..=ANCHOR_LUMA_MAX).contains(&luma) {
                rejections.luma_out_of_range += 1;
                continue;
            }

            if saturation < MIN_DOMINANT_SATURATION {
                rejections.low_saturation += 1;
                continue;
            }

            let mut order = [(0usize, pixel[0]), (1usize, pixel[1]), (2usize, pixel[2])];
            order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if order[0].1 < order[1].1 * DOMINANT_RATIO_THRESHOLD {
                rejections.weak_dominance += 1;
                continue;
            }

            let dominant = order[0].0;
            dominance_margin_sums[dominant] += order[0].1 / order[1].1.max(1e-9) - 1.0;
            let band = neutral_luminance_band(luma);
            for c in 0..3 {
                band_sums[dominant][band][c] += pixel[c];
            }
            band_counts[dominant][band] += 1;
            counts[dominant] += 1;
        }
    }

    let anchors = std::array::from_fn(|dominant| {
        if counts[dominant] == 0 {
            let mut unit = [0.0f64; 3];
            unit[dominant] = 1.0;
            unit
        } else {
            let populated_bands = band_counts[dominant]
                .iter()
                .filter(|count| **count > 0)
                .count();
            let per_band_cap = (counts[dominant] as f64 / populated_bands.max(1) as f64).max(1.0);
            let mut weighted_sum = [0.0f64; 3];
            let mut total_weight = 0.0f64;
            for band in 0..3 {
                let n = band_counts[dominant][band] as f64;
                if n == 0.0 {
                    continue;
                }
                let weight = n.min(per_band_cap);
                weighted_sum[0] += (band_sums[dominant][band][0] / n) * weight;
                weighted_sum[1] += (band_sums[dominant][band][1] / n) * weight;
                weighted_sum[2] += (band_sums[dominant][band][2] / n) * weight;
                total_weight += weight;
            }
            let total_weight = total_weight.max(1e-6);
            [
                weighted_sum[0] / total_weight,
                weighted_sum[1] / total_weight,
                weighted_sum[2] / total_weight,
            ]
        }
    });

    rejections.accepted_anchor_samples = counts.iter().sum();
    let quality = dominant_anchor_quality(counts, band_counts, dominance_margin_sums);
    DominantAnchorEstimate {
        rgb: anchors,
        counts,
        bands: band_counts,
        rejections,
        quality,
    }
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

fn histogram_bin(value: f64, max_value: f64) -> usize {
    if max_value <= 1e-12 {
        return 0;
    }
    let t = (value / max_value).clamp(0.0, 1.0);
    ((t * (HISTOGRAM_BINS - 1) as f64).round() as usize).min(HISTOGRAM_BINS - 1)
}

fn percentile_from_histogram(hist: &[u64], max_value: f64, percentile: f64) -> f64 {
    let total: u64 = hist.iter().sum();
    if total == 0 || max_value <= 1e-12 {
        return 0.0;
    }

    let target = ((total as f64 - 1.0) * percentile.clamp(0.0, 1.0)).round() as u64;
    let mut cumulative = 0u64;
    for (idx, count) in hist.iter().enumerate() {
        cumulative += *count;
        if cumulative > target {
            let t = idx as f64 / (HISTOGRAM_BINS - 1) as f64;
            return t * max_value;
        }
    }

    max_value
}

#[derive(Debug, Clone)]
struct MappingStats {
    pre_scale_channel_min: [f64; 3],
    pre_scale_channel_max: [f64; 3],
    pre_scale_channel_high_percentile: [f64; 3],
    pre_scale_clipped_high_ratio: [f64; 3],
    pre_scale_clipped_low_ratio: [f64; 3],
    pre_scale_max_low_excursion: [f64; 3],
    pre_scale_max_high_excursion: [f64; 3],
    exposure_scale: f64,
    pre_scale_preserved_ratio: f64,
    neutral_balance_delta: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateKind {
    CalibratedDirect,
    ScannerPrior,
    PositiveRgb,
    ImageDerived,
    NeutralFallback,
}

impl CandidateKind {
    fn as_str(self) -> &'static str {
        match self {
            CandidateKind::CalibratedDirect => "calibrated_direct",
            CandidateKind::ScannerPrior => "scanner_prior",
            CandidateKind::PositiveRgb => "positive_rgb",
            CandidateKind::ImageDerived => "image_derived",
            CandidateKind::NeutralFallback => "neutral_fallback",
        }
    }

    fn preference_rank(self) -> usize {
        match self {
            CandidateKind::CalibratedDirect => 0,
            CandidateKind::ScannerPrior => 1,
            CandidateKind::PositiveRgb => 2,
            CandidateKind::ImageDerived => 3,
            CandidateKind::NeutralFallback => 4,
        }
    }

    fn fallback_penalty(self) -> f64 {
        match self {
            CandidateKind::CalibratedDirect
            | CandidateKind::ScannerPrior
            | CandidateKind::PositiveRgb
            | CandidateKind::ImageDerived => 0.0,
            CandidateKind::NeutralFallback => NEUTRAL_FALLBACK_FIDELITY_PENALTY,
        }
    }

    fn is_matrix(self) -> bool {
        self != CandidateKind::NeutralFallback
    }
}

#[derive(Debug, Clone)]
struct ColorMappingCandidate {
    candidate: &'static str,
    mapping_strategy: &'static str,
    source_label: &'static str,
    kind: CandidateKind,
    matrix: Matrix3<f64>,
    nonlinear_model: Option<RuntimeNonlinearTransform>,
    stats: MappingStats,
    diagnostics: ColorspaceDiagnostics,
    neutral_balance_scale: [f64; 3],
    selected_mapping_reason: String,
    score: ColorMappingCandidateScore,
}

#[derive(Debug, Clone)]
enum RuntimeNonlinearTransform {
    RootPolynomial {
        model: Arc<RootPolynomialColorModel>,
        xyz_to_prophoto: Matrix3<f64>,
    },
    ResidualLut3d {
        model: Arc<ResidualLut3dColorModel>,
        baseline_matrix: [[f64; 3]; 3],
        root_baseline: Option<Arc<RootPolynomialColorModel>>,
        xyz_to_prophoto: Matrix3<f64>,
    },
}

impl RuntimeNonlinearTransform {
    fn map(&self, source: Vector3<f64>) -> Vector3<f64> {
        let rgb = [source[0], source[1], source[2]];
        match self {
            Self::RootPolynomial {
                model,
                xyz_to_prophoto,
            } => {
                let xyz = crate::color_calibration::evaluate_root_polynomial_xyz(model, rgb);
                xyz_to_prophoto * Vector3::new(xyz[0], xyz[1], xyz[2])
            }
            Self::ResidualLut3d {
                model,
                baseline_matrix,
                root_baseline,
                xyz_to_prophoto,
            } => {
                let xyz = crate::color_calibration::evaluate_residual_lut_3d_xyz(
                    model,
                    baseline_matrix,
                    root_baseline.as_deref(),
                    rgb,
                );
                xyz_to_prophoto * Vector3::new(xyz[0], xyz[1], xyz[2])
            }
        }
    }
}

fn map_candidate_pixel(candidate: &ColorMappingCandidate, source: Vector3<f64>) -> Vector3<f64> {
    candidate
        .nonlinear_model
        .as_ref()
        .map_or_else(|| candidate.matrix * source, |model| model.map(source))
}

fn point_inside_expanded_chromaticity_hull(point: [f64; 2], hull: &[[f64; 2]]) -> bool {
    if hull.len() < 3 {
        return false;
    }
    let centroid = [
        hull.iter().map(|vertex| vertex[0]).sum::<f64>() / hull.len() as f64,
        hull.iter().map(|vertex| vertex[1]).sum::<f64>() / hull.len() as f64,
    ];
    (0..hull.len()).all(|index| {
        let expand = |vertex: [f64; 2]| {
            [
                centroid[0] + (vertex[0] - centroid[0]) * NONLINEAR_MODEL_HULL_EXPANSION,
                centroid[1] + (vertex[1] - centroid[1]) * NONLINEAR_MODEL_HULL_EXPANSION,
            ]
        };
        let left = expand(hull[index]);
        let right = expand(hull[(index + 1) % hull.len()]);
        (right[0] - left[0]) * (point[1] - left[1]) - (right[1] - left[1]) * (point[0] - left[0])
            >= -1e-9
    })
}

fn nonlinear_model_runtime_diagnostics(
    img: &Array3<f64>,
    transform: &RuntimeNonlinearTransform,
) -> NonlinearColorModelRuntimeDiagnostics {
    let (hull, input_domain) = match transform {
        RuntimeNonlinearTransform::RootPolynomial { model, .. } => {
            (model.training_chromaticity_hull.as_slice(), None)
        }
        RuntimeNonlinearTransform::ResidualLut3d { model, .. } => (
            model.training_chromaticity_hull.as_slice(),
            Some((model.input_min, model.input_max)),
        ),
    };
    let (h, w, _) = img.dim();
    let total_pixels = h.saturating_mul(w);
    let stride = if total_pixels <= NONLINEAR_MODEL_MAX_SAMPLES {
        1
    } else {
        total_pixels.div_ceil(NONLINEAR_MODEL_MAX_SAMPLES)
    };
    let mut sampled = 0usize;
    let mut negative = 0usize;
    let mut outside = 0usize;
    let mut outside_input_domain = 0usize;
    let mut full_application = 0usize;
    for y in 0..h {
        for x in 0..w {
            let index = y * w + x;
            if index % stride != 0 {
                continue;
            }
            let rgb = [img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]]];
            if rgb.iter().any(|value| !value.is_finite()) {
                continue;
            }
            sampled += 1;
            if rgb.iter().any(|value| *value < -1e-6) {
                negative += 1;
                outside += 1;
                if input_domain.is_some() {
                    outside_input_domain += 1;
                }
                continue;
            }
            let nonnegative: [f64; 3] = std::array::from_fn(|channel| rgb[channel].max(0.0));
            let sum = nonnegative.iter().sum::<f64>();
            if sum <= 1e-9 {
                outside += 1;
                continue;
            }
            let chromaticity = [nonnegative[0] / sum, nonnegative[1] / sum];
            if !point_inside_expanded_chromaticity_hull(chromaticity, hull) {
                outside += 1;
            }
            match transform {
                RuntimeNonlinearTransform::RootPolynomial { .. } => full_application += 1,
                RuntimeNonlinearTransform::ResidualLut3d { model, .. } => {
                    let inside_input = input_domain.is_some_and(|(minimum, maximum)| {
                        (0..3).all(|channel| {
                            rgb[channel] >= minimum[channel] - 1e-12
                                && rgb[channel] <= maximum[channel] + 1e-12
                        })
                    });
                    if !inside_input {
                        outside_input_domain += 1;
                    }
                    if crate::color_calibration::residual_lut_3d_has_full_support(model, rgb) {
                        full_application += 1;
                    }
                }
            }
        }
    }
    let denominator = sampled.max(1) as f64;
    let negative_input_ratio = negative as f64 / denominator;
    let outside_training_chromaticity_hull_ratio = outside as f64 / denominator;
    let outside_training_input_domain_ratio = outside_input_domain as f64 / denominator;
    let full_model_application_ratio = full_application as f64 / denominator;
    let support_ok = sampled >= NONLINEAR_MODEL_MIN_SUPPORT_SAMPLES
        && negative_input_ratio <= NONLINEAR_MODEL_MAX_NEGATIVE_INPUT_RATIO
        && outside_training_chromaticity_hull_ratio <= NONLINEAR_MODEL_MAX_OUTSIDE_HULL_RATIO
        && outside_training_input_domain_ratio <= NONLINEAR_MODEL_MAX_OUTSIDE_INPUT_DOMAIN_RATIO
        && full_model_application_ratio >= NONLINEAR_MODEL_MIN_FULL_APPLICATION_RATIO;
    let support_reason = if sampled < NONLINEAR_MODEL_MIN_SUPPORT_SAMPLES {
        format!(
            "only {sampled} finite scene samples were available; at least {NONLINEAR_MODEL_MIN_SUPPORT_SAMPLES} are required"
        )
    } else if negative_input_ratio > NONLINEAR_MODEL_MAX_NEGATIVE_INPUT_RATIO {
        format!(
            "{:.1}% of sampled source pixels were negative, outside the non-negative target training domain (limit {:.1}%)",
            negative_input_ratio * 100.0,
            NONLINEAR_MODEL_MAX_NEGATIVE_INPUT_RATIO * 100.0
        )
    } else if outside_training_chromaticity_hull_ratio > NONLINEAR_MODEL_MAX_OUTSIDE_HULL_RATIO {
        format!(
            "{:.1}% of sampled source pixels were outside the expanded measured target chromaticity hull (limit {:.1}%)",
            outside_training_chromaticity_hull_ratio * 100.0,
            NONLINEAR_MODEL_MAX_OUTSIDE_HULL_RATIO * 100.0
        )
    } else if outside_training_input_domain_ratio > NONLINEAR_MODEL_MAX_OUTSIDE_INPUT_DOMAIN_RATIO {
        format!(
            "{:.1}% of sampled source pixels were outside the measured 3D LUT input domain (limit {:.1}%)",
            outside_training_input_domain_ratio * 100.0,
            NONLINEAR_MODEL_MAX_OUTSIDE_INPUT_DOMAIN_RATIO * 100.0
        )
    } else if full_model_application_ratio < NONLINEAR_MODEL_MIN_FULL_APPLICATION_RATIO {
        format!(
            "the full nonlinear model applied to {:.1}% of sampled source pixels, below the {:.1}% minimum",
            full_model_application_ratio * 100.0,
            NONLINEAR_MODEL_MIN_FULL_APPLICATION_RATIO * 100.0
        )
    } else {
        format!(
            "scene support is inside nonlinear-model limits: {:.1}% negative, {:.1}% outside the expanded measured chromaticity hull, {:.1}% outside the LUT input domain, and {:.1}% full-model application",
            negative_input_ratio * 100.0,
            outside_training_chromaticity_hull_ratio * 100.0,
            outside_training_input_domain_ratio * 100.0,
            full_model_application_ratio * 100.0
        )
    };
    let (
        model_id,
        model_type,
        basis,
        degree,
        grid_size,
        baseline_kind,
        held_out_delta_e00_rms,
        held_out_delta_e00_max,
        matrix_held_out_delta_e00_rms,
        baseline_held_out_delta_e00_rms,
    ) = match transform {
        RuntimeNonlinearTransform::RootPolynomial { model, .. } => (
            model.model_id.clone(),
            "root_polynomial",
            model.basis.clone(),
            Some(model.degree),
            None,
            "matrix".to_string(),
            model.validation.held_out_delta_e00_rms,
            model.validation.held_out_delta_e00_max,
            model.validation.matrix_held_out_delta_e00_rms,
            model.validation.matrix_held_out_delta_e00_rms,
        ),
        RuntimeNonlinearTransform::ResidualLut3d {
            model,
            root_baseline,
            ..
        } => (
            model.model_id.clone(),
            "residual_lut_3d",
            model.interpolation.clone(),
            None,
            Some(model.grid_size),
            model.baseline_kind.clone(),
            model.validation.held_out_delta_e00_rms,
            model.validation.held_out_delta_e00_max,
            root_baseline
                .as_ref()
                .map_or(model.validation.baseline_held_out_delta_e00_rms, |root| {
                    root.validation.matrix_held_out_delta_e00_rms
                }),
            model.validation.baseline_held_out_delta_e00_rms,
        ),
    };
    NonlinearColorModelRuntimeDiagnostics {
        model_id,
        model_type,
        basis,
        degree,
        grid_size,
        baseline_kind,
        held_out_delta_e00_rms,
        held_out_delta_e00_max,
        matrix_held_out_delta_e00_rms,
        baseline_held_out_delta_e00_rms,
        sampled_pixel_count: sampled,
        negative_input_ratio,
        outside_training_chromaticity_hull_ratio,
        outside_training_input_domain_ratio,
        full_model_application_ratio,
        support_status: if support_ok { "accepted" } else { "rejected" }.to_string(),
        support_reason,
    }
}

fn reject_candidate_for_nonlinear_support(
    score: &mut ColorMappingCandidateScore,
    diagnostics: &NonlinearColorModelRuntimeDiagnostics,
) {
    if diagnostics.support_status == "accepted" {
        return;
    }
    score.rejected = true;
    score.rejection_reason = Some(match score.rejection_reason.take() {
        Some(existing) => format!(
            "{existing}; nonlinear target-domain support rejected: {}",
            diagnostics.support_reason
        ),
        None => format!(
            "nonlinear target-domain support rejected: {}",
            diagnostics.support_reason
        ),
    });
}

fn neutral_balance_delta(
    neutral_rgb: [f64; 3],
    combined: &Matrix3<f64>,
    exposure_scale: f64,
) -> [f64; 3] {
    neutral_balance_delta_with(neutral_rgb, exposure_scale, |source| combined * source)
}

fn neutral_balance_delta_with<F>(neutral_rgb: [f64; 3], exposure_scale: f64, map: F) -> [f64; 3]
where
    F: Fn(Vector3<f64>) -> Vector3<f64>,
{
    let mapped = map(Vector3::new(neutral_rgb[0], neutral_rgb[1], neutral_rgb[2])) / exposure_scale;
    let mean = ((mapped[0] + mapped[1] + mapped[2]) / 3.0).max(1e-9);
    std::array::from_fn(|c| mapped[c] / mean - 1.0)
}

fn evaluate_mapping_stats(
    img: &Array3<f64>,
    combined: &Matrix3<f64>,
    neutral_rgb: [f64; 3],
) -> MappingStats {
    evaluate_mapping_stats_with(img, neutral_rgb, |source| combined * source)
}

fn evaluate_mapping_stats_with<F>(img: &Array3<f64>, neutral_rgb: [f64; 3], map: F) -> MappingStats
where
    F: Fn(Vector3<f64>) -> Vector3<f64>,
{
    let (h, w, _c) = img.dim();
    let total_pixels = (h * w).max(1) as f64;
    let mut pre_scale_channel_min = [f64::INFINITY; 3];
    let mut pre_scale_channel_max = [0.0f64; 3];
    let mut pre_scale_max_low_excursion = [0.0f64; 3];
    let mut pre_scale_max_high_excursion = [0.0f64; 3];
    let mut pre_scale_clipped_high = [0u64; 3];
    let mut pre_scale_clipped_low = [0u64; 3];
    let mut pre_scale_any_clipped = 0u64;

    for y in 0..h {
        for x in 0..w {
            let mapped = map(Vector3::new(img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]]));
            let mut any_clipped = false;
            for c in 0..3 {
                let value = mapped[c];
                if value.is_finite() {
                    pre_scale_channel_min[c] = pre_scale_channel_min[c].min(value);
                    pre_scale_channel_max[c] = pre_scale_channel_max[c].max(value.max(0.0));
                    if value > 1.0 + GAMUT_CLIP_EPSILON {
                        pre_scale_clipped_high[c] += 1;
                        pre_scale_max_high_excursion[c] =
                            pre_scale_max_high_excursion[c].max(value - 1.0);
                        any_clipped = true;
                    }
                    if value < -GAMUT_CLIP_EPSILON {
                        pre_scale_clipped_low[c] += 1;
                        pre_scale_max_low_excursion[c] = pre_scale_max_low_excursion[c].max(-value);
                        any_clipped = true;
                    }
                }
            }
            if any_clipped {
                pre_scale_any_clipped += 1;
            }
        }
    }

    let histogram_max: [f64; 3] = std::array::from_fn(|c| pre_scale_channel_max[c].max(1.0));
    let mut histograms = vec![vec![0u64; HISTOGRAM_BINS]; 3];
    for y in 0..h {
        for x in 0..w {
            let mapped = map(Vector3::new(img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]]));
            for c in 0..3 {
                let value = mapped[c].max(0.0);
                histograms[c][histogram_bin(value, histogram_max[c])] += 1;
            }
        }
    }

    let pre_scale_channel_high_percentile = std::array::from_fn(|c| {
        percentile_from_histogram(
            &histograms[c],
            histogram_max[c],
            HIGHLIGHT_HEADROOM_PERCENTILE,
        )
    });
    let exposure_scale = pre_scale_channel_high_percentile
        .iter()
        .copied()
        .fold(1.0f64, f64::max)
        .max(1.0);
    let neutral_balance_delta = neutral_balance_delta_with(neutral_rgb, exposure_scale, map);
    let pre_scale_channel_min = std::array::from_fn(|c| {
        if pre_scale_channel_min[c].is_finite() {
            pre_scale_channel_min[c]
        } else {
            0.0
        }
    });

    MappingStats {
        pre_scale_channel_min,
        pre_scale_channel_max,
        pre_scale_channel_high_percentile,
        pre_scale_clipped_high_ratio: std::array::from_fn(|c| {
            pre_scale_clipped_high[c] as f64 / total_pixels
        }),
        pre_scale_clipped_low_ratio: std::array::from_fn(|c| {
            pre_scale_clipped_low[c] as f64 / total_pixels
        }),
        pre_scale_max_low_excursion,
        pre_scale_max_high_excursion,
        exposure_scale,
        pre_scale_preserved_ratio: 1.0 - pre_scale_any_clipped as f64 / total_pixels,
        neutral_balance_delta,
    }
}

fn low_gamut_fallback_reason(stats: &MappingStats, candidate_source: &str) -> Option<String> {
    let max_channel_low = stats
        .pre_scale_clipped_low_ratio
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    let total_low = stats.pre_scale_clipped_low_ratio.iter().sum::<f64>();
    if max_channel_low > MAX_LOW_GAMUT_CLIP_RATIO_PER_CHANNEL
        || total_low > MAX_LOW_GAMUT_CLIP_RATIO_TOTAL
    {
        Some(format!(
            "{candidate_source} colorspace matrix would clamp too many negative channel values (max channel {:.1}%, total {:.1}%); using neutral-balance gamut fallback",
            max_channel_low * 100.0,
            total_low * 100.0
        ))
    } else {
        None
    }
}

fn within_low_gamut_limits(stats: &MappingStats) -> bool {
    let max_channel_low = stats
        .pre_scale_clipped_low_ratio
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    let total_low = stats.pre_scale_clipped_low_ratio.iter().sum::<f64>();
    max_channel_low <= MAX_LOW_GAMUT_CLIP_RATIO_PER_CHANNEL
        && total_low <= MAX_LOW_GAMUT_CLIP_RATIO_TOTAL
}

fn within_trusted_low_gamut_limits(stats: &MappingStats) -> bool {
    let max_channel_low = stats
        .pre_scale_clipped_low_ratio
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    let total_low = stats.pre_scale_clipped_low_ratio.iter().sum::<f64>();
    max_channel_low <= GAMUT_TRUSTED_BLEND_MAX_LOW_RATIO_PER_CHANNEL
        && total_low <= GAMUT_TRUSTED_BLEND_MAX_LOW_RATIO_TOTAL
}

fn blend_toward_neutral(
    image_matrix: &Matrix3<f64>,
    neutral_matrix: &Matrix3<f64>,
    neutral_mix: f64,
) -> Matrix3<f64> {
    image_matrix * (1.0 - neutral_mix) + neutral_matrix * neutral_mix
}

fn gamut_safe_image_blend(
    img: &Array3<f64>,
    image_matrix: &Matrix3<f64>,
    neutral_matrix: &Matrix3<f64>,
    neutral_rgb: [f64; 3],
    image_stats: &MappingStats,
) -> Option<(Matrix3<f64>, MappingStats, f64)> {
    if within_low_gamut_limits(image_stats) {
        return None;
    }

    let neutral_stats = evaluate_mapping_stats(img, neutral_matrix, neutral_rgb);
    if !within_low_gamut_limits(&neutral_stats) {
        return None;
    }

    let mut low = 0.0f64;
    let mut high = 1.0f64;
    let mut best_matrix = *neutral_matrix;
    let mut best_stats = neutral_stats;
    let mut best_mix = 1.0f64;

    for _ in 0..GAMUT_SAFE_BLEND_SEARCH_STEPS {
        let mix = (low + high) * 0.5;
        let matrix = blend_toward_neutral(image_matrix, neutral_matrix, mix);
        let stats = evaluate_mapping_stats(img, &matrix, neutral_rgb);
        if within_low_gamut_limits(&stats) {
            high = mix;
            best_matrix = matrix;
            best_stats = stats;
            best_mix = mix;
        } else {
            low = mix;
        }
    }

    (best_mix <= GAMUT_SAFE_BLEND_MAX_NEUTRAL_MIX).then_some((best_matrix, best_stats, best_mix))
}

fn gamut_trusted_image_blend(
    img: &Array3<f64>,
    image_matrix: &Matrix3<f64>,
    neutral_matrix: &Matrix3<f64>,
    neutral_rgb: [f64; 3],
    image_stats: &MappingStats,
) -> Option<(Matrix3<f64>, MappingStats, f64)> {
    if within_trusted_low_gamut_limits(image_stats) {
        return None;
    }

    let max_matrix = blend_toward_neutral(
        image_matrix,
        neutral_matrix,
        GAMUT_SAFE_BLEND_MAX_NEUTRAL_MIX,
    );
    let max_stats = evaluate_mapping_stats(img, &max_matrix, neutral_rgb);
    if !within_trusted_low_gamut_limits(&max_stats) {
        return None;
    }

    let mut low = 0.0f64;
    let mut high = GAMUT_SAFE_BLEND_MAX_NEUTRAL_MIX;
    let mut best_matrix = max_matrix;
    let mut best_stats = max_stats;
    let mut best_mix = GAMUT_SAFE_BLEND_MAX_NEUTRAL_MIX;

    for _ in 0..GAMUT_TRUSTED_BLEND_SEARCH_STEPS {
        let mix = (low + high) * 0.5;
        let matrix = blend_toward_neutral(image_matrix, neutral_matrix, mix);
        let stats = evaluate_mapping_stats(img, &matrix, neutral_rgb);
        if within_trusted_low_gamut_limits(&stats) {
            high = mix;
            best_matrix = matrix;
            best_stats = stats;
            best_mix = mix;
        } else {
            low = mix;
        }
    }

    Some((best_matrix, best_stats, best_mix))
}

fn gamut_stabilized_image_blend(
    img: &Array3<f64>,
    image_matrix: &Matrix3<f64>,
    neutral_matrix: &Matrix3<f64>,
    neutral_rgb: [f64; 3],
    trusted_mix: f64,
    trusted_stats: &MappingStats,
) -> Option<(Matrix3<f64>, MappingStats, f64)> {
    if trusted_mix < GAMUT_STABILIZED_BLEND_MIN_TRUSTED_MIX
        || trusted_stats.exposure_scale > GAMUT_STABILIZED_BLEND_MAX_EXPOSURE_SCALE
    {
        return None;
    }

    let extra_fraction = if trusted_mix >= GAMUT_STABILIZED_BLEND_HIGH_MIX_MIN_TRUSTED_MIX {
        GAMUT_STABILIZED_BLEND_HIGH_MIX_EXTRA_FRACTION
    } else {
        GAMUT_STABILIZED_BLEND_EXTRA_FRACTION
    };
    let stabilized_mix =
        trusted_mix + (GAMUT_SAFE_BLEND_MAX_NEUTRAL_MIX - trusted_mix) * extra_fraction;
    let stabilized = evaluate_stabilized_image_blend(
        img,
        image_matrix,
        neutral_matrix,
        neutral_rgb,
        trusted_mix,
        stabilized_mix,
    );
    if extra_fraction > GAMUT_STABILIZED_BLEND_EXTRA_FRACTION
        && stabilized.as_ref().is_some_and(|(matrix, stats, _)| {
            let tone = evaluate_candidate_rendered_tone(img, matrix, stats.exposure_scale);
            tone.post_tone_midtone_luminance_p50 < GAMUT_STABILIZED_BLEND_MIN_RENDER_MIDTONE_P50
                || tone.post_tone_render_luminance_range < GAMUT_STABILIZED_BLEND_MIN_RENDER_RANGE
        })
    {
        let fallback_mix = trusted_mix
            + (GAMUT_SAFE_BLEND_MAX_NEUTRAL_MIX - trusted_mix)
                * GAMUT_STABILIZED_BLEND_EXTRA_FRACTION;
        return evaluate_stabilized_image_blend(
            img,
            image_matrix,
            neutral_matrix,
            neutral_rgb,
            trusted_mix,
            fallback_mix,
        );
    }

    stabilized
}

fn evaluate_stabilized_image_blend(
    img: &Array3<f64>,
    image_matrix: &Matrix3<f64>,
    neutral_matrix: &Matrix3<f64>,
    neutral_rgb: [f64; 3],
    trusted_mix: f64,
    stabilized_mix: f64,
) -> Option<(Matrix3<f64>, MappingStats, f64)> {
    if stabilized_mix <= trusted_mix + GAMUT_STABILIZED_BLEND_MIN_STEP {
        return None;
    }

    let stabilized_matrix = blend_toward_neutral(image_matrix, neutral_matrix, stabilized_mix);
    let stabilized_stats = evaluate_mapping_stats(img, &stabilized_matrix, neutral_rgb);
    within_trusted_low_gamut_limits(&stabilized_stats).then_some((
        stabilized_matrix,
        stabilized_stats,
        stabilized_mix,
    ))
}

fn score_neutral_delta(delta: [f64; 3]) -> f64 {
    delta.iter().map(|value| value.abs()).sum::<f64>() / 3.0
}

fn total_ratio(ratios: [f64; 3]) -> f64 {
    ratios.iter().sum::<f64>()
}

fn excess(value: f64, target: f64) -> f64 {
    (value - target).max(0.0)
}

fn candidate_render_sample(
    img: &Array3<f64>,
    combined: &Matrix3<f64>,
    exposure_scale: f64,
) -> (Array3<f64>, usize) {
    candidate_render_sample_with(img, exposure_scale, |source| combined * source)
}

fn candidate_render_sample_with<F>(
    img: &Array3<f64>,
    exposure_scale: f64,
    map: F,
) -> (Array3<f64>, usize)
where
    F: Fn(Vector3<f64>) -> Vector3<f64>,
{
    let (h, w, _) = img.dim();
    let total_pixels = h.saturating_mul(w);
    let sample_stride = if total_pixels <= RENDER_TONE_MAX_SAMPLES {
        1
    } else {
        total_pixels.div_ceil(RENDER_TONE_MAX_SAMPLES)
    };
    let sample_count = if total_pixels == 0 {
        0
    } else {
        total_pixels.div_ceil(sample_stride)
    };
    let mut sample = Array3::<f64>::zeros((sample_count, 1, 3));
    if total_pixels == 0 {
        return (sample, sample_stride);
    }

    let mut out_idx = 0usize;
    let exposure_scale = exposure_scale.max(1e-9);
    for y in 0..h {
        for x in 0..w {
            let pixel_index = y * w + x;
            if pixel_index % sample_stride != 0 {
                continue;
            }
            let mapped =
                map(Vector3::new(img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]])) / exposure_scale;
            for c in 0..3 {
                sample[[out_idx, 0, c]] = if mapped[c].is_finite() {
                    mapped[c]
                } else {
                    0.0
                };
            }
            out_idx += 1;
        }
    }

    (sample, sample_stride)
}

fn evaluate_candidate_rendered_tone(
    img: &Array3<f64>,
    combined: &Matrix3<f64>,
    exposure_scale: f64,
) -> RenderedToneCandidateDiagnostics {
    let (sample, sample_stride) = candidate_render_sample(img, combined, exposure_scale);
    evaluate_rendered_tone_sample(sample, sample_stride)
}

fn evaluate_rendered_tone_sample(
    sample: Array3<f64>,
    sample_stride: usize,
) -> RenderedToneCandidateDiagnostics {
    let pre_quality = tonemap::render_quality_diagnostics(&sample);
    let tone_fit = tonemap::fit_tone_params_with_diagnostics(&sample);
    let tone_apply = tonemap::apply_tonemap_with_params_and_color_protection_diagnostics(
        &sample,
        &tone_fit.params,
        &tonemap::ToneColorProtection::default(),
    );
    let post_quality = tonemap::render_quality_diagnostics(&tone_apply.image);

    let saturated_highlight_saturation_loss = if pre_quality.bright_saturated.pixel_count > 0 {
        let post_saturated_median = if post_quality.bright_saturated.pixel_count > 0 {
            post_quality.bright_saturated.saturation_median
        } else {
            0.0
        };
        (pre_quality.bright_saturated.saturation_median - post_saturated_median).max(0.0)
    } else {
        0.0
    };
    let rendered_tone_penalty = excess(
        post_quality.shadow.saturation_p95,
        RENDER_TONE_SHADOW_SATURATION_TARGET,
    ) * RENDER_TONE_SHADOW_SATURATION_WEIGHT
        + excess(
            post_quality.midtone.saturation_p95,
            RENDER_TONE_MIDTONE_SATURATION_TARGET,
        ) * RENDER_TONE_MIDTONE_SATURATION_WEIGHT
        + excess(
            post_quality.bright_neutral.saturation_p95,
            RENDER_TONE_NEUTRAL_HIGHLIGHT_SATURATION_TARGET,
        ) * RENDER_TONE_NEUTRAL_HIGHLIGHT_WEIGHT
        + excess(
            saturated_highlight_saturation_loss,
            RENDER_TONE_SATURATED_HIGHLIGHT_LOSS_TOLERANCE,
        ) * RENDER_TONE_SATURATED_HIGHLIGHT_LOSS_WEIGHT;
    let tone_pre_chroma_clipped_high_total = total_ratio(
        tone_apply
            .diagnostics
            .pre_chroma_compression_clipped_high_ratio,
    );
    let tone_post_chroma_clipped_high_total = total_ratio(
        tone_apply
            .diagnostics
            .post_chroma_compression_clipped_high_ratio,
    );
    let tone_post_chroma_clipped_low_total = total_ratio(
        tone_apply
            .diagnostics
            .post_chroma_compression_clipped_low_ratio,
    );
    let tone_chroma_cleanup_penalty = (tone_apply
        .diagnostics
        .highlight_neutral_chroma_compressed_ratio
        + tone_apply.diagnostics.shadow_chroma_compressed_ratio)
        * TONE_CHROMA_CLEANUP_RATIO_WEIGHT
        + (tone_post_chroma_clipped_high_total + tone_post_chroma_clipped_low_total)
            * TONE_CHROMA_CLEANUP_CLIP_WEIGHT;

    RenderedToneCandidateDiagnostics {
        sample_count: sample.len_of(Axis(0)),
        sample_stride,
        tone_fit_domain: tone_fit.params.domain.as_str(),
        post_tone_render_luminance_percentiles: post_quality.overall.luminance_percentiles,
        post_tone_render_luminance_range: post_quality.overall.luminance_percentiles[2]
            - post_quality.overall.luminance_percentiles[0],
        post_tone_midtone_luminance_p50: post_quality.midtone.luminance_percentiles[1],
        pre_tone_shadow_saturation_p95: pre_quality.shadow.saturation_p95,
        pre_tone_midtone_saturation_p95: pre_quality.midtone.saturation_p95,
        pre_tone_bright_neutral_saturation_p95: pre_quality.bright_neutral.saturation_p95,
        pre_tone_bright_saturated_saturation_median: pre_quality.bright_saturated.saturation_median,
        shadow_saturation_median: post_quality.shadow.saturation_median,
        shadow_saturation_p95: post_quality.shadow.saturation_p95,
        midtone_saturation_median: post_quality.midtone.saturation_median,
        midtone_saturation_p95: post_quality.midtone.saturation_p95,
        bright_neutral_pixel_count: post_quality.bright_neutral.pixel_count,
        bright_neutral_saturation_median: post_quality.bright_neutral.saturation_median,
        bright_neutral_saturation_p95: post_quality.bright_neutral.saturation_p95,
        bright_saturated_pixel_count: post_quality.bright_saturated.pixel_count,
        bright_saturated_saturation_median: post_quality.bright_saturated.saturation_median,
        bright_saturated_saturation_p95: post_quality.bright_saturated.saturation_p95,
        saturated_highlight_saturation_loss,
        tone_highlight_chroma_compressed_ratio: tone_apply
            .diagnostics
            .highlight_chroma_compressed_ratio,
        tone_highlight_neutral_chroma_compressed_ratio: tone_apply
            .diagnostics
            .highlight_neutral_chroma_compressed_ratio,
        tone_shadow_chroma_compressed_ratio: tone_apply.diagnostics.shadow_chroma_compressed_ratio,
        tone_pre_chroma_clipped_high_total,
        tone_post_chroma_clipped_high_total,
        tone_post_chroma_clipped_low_total,
        rendered_tone_penalty,
        tone_chroma_cleanup_penalty,
    }
}

fn evaluate_nonlinear_candidate_rendered_tone(
    img: &Array3<f64>,
    model: &RuntimeNonlinearTransform,
    exposure_scale: f64,
) -> RenderedToneCandidateDiagnostics {
    let (sample, sample_stride) =
        candidate_render_sample_with(img, exposure_scale, |source| model.map(source));
    evaluate_rendered_tone_sample(sample, sample_stride)
}

fn percentile_from_sorted_values(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let idx = ((values.len() - 1) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    values.get(idx).copied()
}

fn rgb_mean_luminance(rgb: [f64; 3]) -> f64 {
    (rgb[0] + rgb[1] + rgb[2]) / 3.0
}

fn rgb_saturation(rgb: [f64; 3]) -> f64 {
    let max = rgb.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = rgb.iter().copied().fold(f64::INFINITY, f64::min);
    if !max.is_finite() || max <= 1e-9 {
        0.0
    } else {
        ((max - min) / max).clamp(0.0, f64::INFINITY)
    }
}

fn dominant_rgb_channel(rgb: [f64; 3]) -> Option<usize> {
    let mut order = [(0usize, rgb[0]), (1usize, rgb[1]), (2usize, rgb[2])];
    order.sort_by(|left, right| right.1.total_cmp(&left.1));
    (order[0].1 > order[1].1 * DOMINANT_RATIO_THRESHOLD).then_some(order[0].0)
}

fn lab_hue_chroma(lab: [f64; 3]) -> Option<(f64, f64)> {
    let chroma = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
    if !chroma.is_finite() || chroma < 1e-9 {
        return None;
    }
    let mut hue = lab[2].atan2(lab[1]).to_degrees();
    if hue < 0.0 {
        hue += 360.0;
    }
    Some((hue, chroma))
}

#[derive(Default)]
struct MemoryFamilyAccumulator {
    l_values: Vec<f64>,
    chroma_values: Vec<f64>,
    hue_values: Vec<f64>,
    implausible: usize,
}

impl MemoryFamilyAccumulator {
    fn push(&mut self, lab: [f64; 3], hue: f64, chroma: f64, plausible: bool) {
        self.l_values.push(lab[0]);
        self.chroma_values.push(chroma);
        self.hue_values.push(hue);
        if !plausible {
            self.implausible += 1;
        }
    }

    fn finish(mut self) -> MemoryColorFamilyDiagnostics {
        self.l_values.sort_by(|a, b| a.total_cmp(b));
        self.chroma_values.sort_by(|a, b| a.total_cmp(b));
        self.hue_values.sort_by(|a, b| a.total_cmp(b));
        let sample_count = self.l_values.len();
        let implausible_ratio =
            (sample_count > 0).then_some(self.implausible as f64 / sample_count as f64);
        MemoryColorFamilyDiagnostics {
            sample_count,
            median_l: percentile_from_sorted_values(&self.l_values, 0.50),
            median_chroma: percentile_from_sorted_values(&self.chroma_values, 0.50),
            median_hue_degrees: percentile_from_sorted_values(&self.hue_values, 0.50),
            implausible_ratio,
        }
    }
}

#[derive(Clone, Default)]
struct TileNeutralAccumulator {
    sum_delta: f64,
    count: usize,
}

fn memory_color_penalty(
    skin: &MemoryColorFamilyDiagnostics,
    foliage: &MemoryColorFamilyDiagnostics,
    sky: &MemoryColorFamilyDiagnostics,
) -> (f64, String) {
    let mut penalty = 0.0f64;
    let mut supported = 0usize;
    for (name, family) in [("skin", skin), ("foliage", foliage), ("sky", sky)] {
        if family.sample_count < MEMORY_COLOR_MIN_FAMILY_SAMPLES {
            continue;
        }
        supported += 1;
        let implausible_ratio = family.implausible_ratio.unwrap_or(0.0);
        penalty += excess(implausible_ratio, MEMORY_COLOR_IMPLAUSIBLE_TARGET)
            * MEMORY_COLOR_PENALTY_WEIGHT;
        if implausible_ratio > MEMORY_COLOR_IMPLAUSIBLE_TARGET {
            return (
                penalty,
                format!(
                    "{name} memory-colour proxy has {:.1}% broadly implausible Lab samples",
                    implausible_ratio * 100.0
                ),
            );
        }
    }

    if supported == 0 {
        (
            0.0,
            "memory-colour proxies were not sufficiently represented in the bounded candidate sample"
                .to_string(),
        )
    } else if penalty > 0.0 {
        (
            penalty,
            "one or more memory-colour proxy families exceeded the implausible-sample target"
                .to_string(),
        )
    } else {
        (
            0.0,
            "memory-colour proxy families stayed inside broad Lab plausibility bounds".to_string(),
        )
    }
}

fn evaluate_candidate_model_quality(
    img: &Array3<f64>,
    combined: &Matrix3<f64>,
    exposure_scale: f64,
) -> ColorModelCandidateDiagnostics {
    evaluate_candidate_model_quality_with(img, exposure_scale, |source| combined * source)
}

fn evaluate_candidate_model_quality_with<F>(
    img: &Array3<f64>,
    exposure_scale: f64,
    map: F,
) -> ColorModelCandidateDiagnostics
where
    F: Fn(Vector3<f64>) -> Vector3<f64>,
{
    let (h, w, _) = img.dim();
    let total_pixels = h.saturating_mul(w);
    let sample_stride = if total_pixels <= MODEL_DIAGNOSTIC_MAX_SAMPLES {
        1
    } else {
        total_pixels.div_ceil(MODEL_DIAGNOSTIC_MAX_SAMPLES)
    };
    let sample_count = if total_pixels == 0 {
        0
    } else {
        total_pixels.div_ceil(sample_stride)
    };
    let exposure_scale = exposure_scale.max(1e-9);
    let prophoto_to_xyz = prophoto_to_xyz_d50_matrix();

    let mut density_bin_output_sum = [0.0f64; DENSITY_MONOTONICITY_BINS];
    let mut density_bin_count = [0usize; DENSITY_MONOTONICITY_BINS];
    let mut hue_cos = [0.0f64; 3];
    let mut hue_sin = [0.0f64; 3];
    let mut hue_counts = [0usize; 3];
    let mut saturation_ratios = Vec::<f64>::new();
    let mut skin = MemoryFamilyAccumulator::default();
    let mut foliage = MemoryFamilyAccumulator::default();
    let mut sky = MemoryFamilyAccumulator::default();
    let tile_grid = SPATIAL_CONSISTENCY_TILE_GRID.max(1);
    let mut tiles = vec![TileNeutralAccumulator::default(); tile_grid * tile_grid];
    let mut spatial_neutral_sample_count = 0usize;

    if total_pixels > 0 {
        for y in 0..h {
            for x in 0..w {
                let pixel_index = y * w + x;
                if pixel_index % sample_stride != 0 {
                    continue;
                }
                let input = [img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]]];
                if !input.iter().all(|value| value.is_finite()) {
                    continue;
                }
                let mapped = map(Vector3::new(input[0], input[1], input[2])) / exposure_scale;
                let mapped_rgb = [mapped[0], mapped[1], mapped[2]];
                if !mapped_rgb.iter().all(|value| value.is_finite()) {
                    continue;
                }
                let mapped_clamped = std::array::from_fn(|c| mapped_rgb[c].clamp(0.0, 1.0));
                let input_luma = rgb_mean_luminance(input).clamp(0.0, 1.0);
                let output_xyz = prophoto_to_xyz
                    * Vector3::new(mapped_clamped[0], mapped_clamped[1], mapped_clamped[2]);
                let output_luma = output_xyz[1].max(0.0);
                let bin = ((input_luma * DENSITY_MONOTONICITY_BINS as f64).floor() as usize)
                    .min(DENSITY_MONOTONICITY_BINS - 1);
                density_bin_output_sum[bin] += output_luma;
                density_bin_count[bin] += 1;

                let input_saturation = rgb_saturation(input);
                let output_saturation = rgb_saturation(mapped_clamped);
                let lab = xyz_d50_to_lab(vector3_to_array(output_xyz));
                if let Some((hue, chroma)) = lab_hue_chroma(lab) {
                    if input_saturation >= MIN_DOMINANT_SATURATION
                        && input_luma > ANCHOR_LUMA_MIN
                        && input_luma < ANCHOR_LUMA_MAX
                    {
                        if let Some(channel) = dominant_rgb_channel(input) {
                            let radians = hue.to_radians();
                            hue_cos[channel] += radians.cos();
                            hue_sin[channel] += radians.sin();
                            hue_counts[channel] += 1;
                        }
                    }
                    if input_saturation >= MIN_DOMINANT_SATURATION
                        && input_luma > ANCHOR_LUMA_MIN
                        && input_luma < ANCHOR_LUMA_MAX
                    {
                        saturation_ratios.push(output_saturation / input_saturation.max(1e-9));
                    }

                    if chroma > 5.0 {
                        if (15.0..=75.0).contains(&hue) {
                            skin.push(
                                lab,
                                hue,
                                chroma,
                                (20.0..=90.0).contains(&lab[0]) && (8.0..=80.0).contains(&chroma),
                            );
                        } else if (90.0..=170.0).contains(&hue) {
                            foliage.push(
                                lab,
                                hue,
                                chroma,
                                (12.0..=85.0).contains(&lab[0]) && (8.0..=100.0).contains(&chroma),
                            );
                        } else if (210.0..=285.0).contains(&hue) {
                            sky.push(
                                lab,
                                hue,
                                chroma,
                                (35.0..=96.0).contains(&lab[0]) && (5.0..=60.0).contains(&chroma),
                            );
                        }
                    }
                }

                let input_neutral = input_saturation <= NEUTRAL_THRESHOLD
                    && input_luma > NEUTRAL_LUMA_MIN
                    && input_luma < NEUTRAL_LUMA_MAX;
                if input_neutral {
                    let mean = rgb_mean_luminance(mapped_clamped).max(1e-9);
                    let delta = mapped_clamped
                        .iter()
                        .map(|value| (value / mean - 1.0).abs())
                        .sum::<f64>()
                        / 3.0;
                    let tile_y = (y * tile_grid / h.max(1)).min(tile_grid - 1);
                    let tile_x = (x * tile_grid / w.max(1)).min(tile_grid - 1);
                    let tile = &mut tiles[tile_y * tile_grid + tile_x];
                    tile.sum_delta += delta;
                    tile.count += 1;
                    spatial_neutral_sample_count += 1;
                }
            }
        }
    }

    let mut density_bin_means = Vec::<f64>::new();
    for idx in 0..DENSITY_MONOTONICITY_BINS {
        if density_bin_count[idx] >= DENSITY_MONOTONICITY_MIN_BIN_SAMPLES {
            density_bin_means.push(density_bin_output_sum[idx] / density_bin_count[idx] as f64);
        }
    }
    let density_violations = density_bin_means
        .windows(2)
        .filter(|pair| pair[1] + DENSITY_MONOTONICITY_TOLERANCE < pair[0])
        .count();
    let density_comparisons = density_bin_means.len().saturating_sub(1);
    let density_violation_ratio = if density_comparisons == 0 {
        0.0
    } else {
        density_violations as f64 / density_comparisons as f64
    };
    let density_monotonicity_score = if density_comparisons == 0 {
        1.0
    } else {
        (1.0 - density_violation_ratio).clamp(0.0, 1.0)
    };
    let density_monotonicity_penalty = excess(
        DENSITY_MONOTONICITY_TARGET_SCORE,
        density_monotonicity_score,
    ) * DENSITY_MONOTONICITY_PENALTY_WEIGHT;

    let hue_linearity_channel_resultant = std::array::from_fn(|channel| {
        let count = hue_counts[channel];
        (count >= HUE_LINEARITY_MIN_CHANNEL_SAMPLES)
            .then(|| (hue_cos[channel].hypot(hue_sin[channel]) / count as f64).clamp(0.0, 1.0))
    });
    let supported_hue_channels = hue_linearity_channel_resultant
        .iter()
        .filter_map(|value| *value)
        .collect::<Vec<_>>();
    let hue_linearity_score = if supported_hue_channels.is_empty() {
        1.0
    } else {
        supported_hue_channels.iter().sum::<f64>() / supported_hue_channels.len() as f64
    };
    let hue_linearity_penalty = if supported_hue_channels.len() >= 2 {
        excess(HUE_LINEARITY_TARGET_SCORE, hue_linearity_score) * HUE_LINEARITY_PENALTY_WEIGHT
    } else {
        0.0
    };

    saturation_ratios.sort_by(|a, b| a.total_cmp(b));
    let saturation_preservation_p05_ratio = percentile_from_sorted_values(&saturation_ratios, 0.05);
    let saturation_preservation_median_ratio =
        percentile_from_sorted_values(&saturation_ratios, 0.50);
    let saturation_preservation_p95_ratio = percentile_from_sorted_values(&saturation_ratios, 0.95);
    let saturation_preservation_penalty =
        if saturation_ratios.len() >= SATURATION_PRESERVATION_MIN_SAMPLES {
            excess(
                SATURATION_PRESERVATION_LOW_TARGET,
                saturation_preservation_median_ratio.unwrap_or(1.0),
            ) * SATURATION_PRESERVATION_PENALTY_WEIGHT
                + excess(
                    SATURATION_PRESERVATION_LOW_P05_TARGET,
                    saturation_preservation_p05_ratio.unwrap_or(1.0),
                ) * SATURATION_PRESERVATION_PENALTY_WEIGHT
                + excess(
                    saturation_preservation_p95_ratio.unwrap_or(1.0),
                    SATURATION_PRESERVATION_HIGH_TARGET,
                ) * SATURATION_PRESERVATION_PENALTY_WEIGHT
        } else {
            0.0
        };

    let skin = skin.finish();
    let foliage = foliage.finish();
    let sky = sky.finish();
    let (memory_color_penalty, memory_color_reason) = memory_color_penalty(&skin, &foliage, &sky);

    let mut tile_deltas = tiles
        .iter()
        .filter_map(|tile| {
            (tile.count >= SPATIAL_CONSISTENCY_MIN_TILE_SAMPLES)
                .then_some(tile.sum_delta / tile.count as f64)
        })
        .collect::<Vec<_>>();
    tile_deltas.sort_by(|a, b| a.total_cmp(b));
    let populated_tile_count = tile_deltas.len();
    let neutral_delta_median = percentile_from_sorted_values(&tile_deltas, 0.50);
    let neutral_delta_p95 = percentile_from_sorted_values(&tile_deltas, 0.95);
    let spatial_consistency_penalty = if populated_tile_count >= SPATIAL_CONSISTENCY_MIN_TILES {
        excess(
            neutral_delta_p95.unwrap_or(0.0),
            SPATIAL_NEUTRAL_DELTA_P95_TARGET,
        ) * SPATIAL_CONSISTENCY_PENALTY_WEIGHT
    } else {
        0.0
    };
    let spatial_reason = if populated_tile_count < SPATIAL_CONSISTENCY_MIN_TILES {
        "not enough neutral-populated tiles for spatial cast consistency scoring".to_string()
    } else if spatial_consistency_penalty > 0.0 {
        format!(
            "neutral tile cast p95 {:.6} exceeded target {:.6}",
            neutral_delta_p95.unwrap_or(0.0),
            SPATIAL_NEUTRAL_DELTA_P95_TARGET
        )
    } else {
        "neutral tile casts were spatially consistent within target".to_string()
    };

    ColorModelCandidateDiagnostics {
        sample_count,
        sample_stride,
        density_monotonicity_score,
        density_monotonicity_violation_ratio: density_violation_ratio,
        density_monotonicity_populated_bins: density_bin_means.len(),
        hue_linearity_score,
        hue_linearity_channel_samples: hue_counts,
        hue_linearity_channel_resultant,
        saturation_preservation_sample_count: saturation_ratios.len(),
        saturation_preservation_p05_ratio,
        saturation_preservation_median_ratio,
        saturation_preservation_p95_ratio,
        memory_color: MemoryColorCandidateDiagnostics {
            skin,
            foliage,
            sky,
            penalty: memory_color_penalty,
            reason: memory_color_reason,
        },
        spatial_consistency: SpatialConsistencyDiagnostics {
            neutral_sample_count: spatial_neutral_sample_count,
            populated_tile_count,
            tile_grid,
            neutral_delta_median,
            neutral_delta_p95,
            penalty: spatial_consistency_penalty,
            reason: spatial_reason,
        },
        density_monotonicity_penalty,
        hue_linearity_penalty,
        saturation_preservation_penalty,
        memory_color_penalty,
        spatial_consistency_penalty,
    }
}

fn evaluate_nonlinear_candidate_model_quality(
    img: &Array3<f64>,
    model: &RuntimeNonlinearTransform,
    exposure_scale: f64,
) -> ColorModelCandidateDiagnostics {
    evaluate_candidate_model_quality_with(img, exposure_scale, |source| model.map(source))
}

fn condition_score(condition_number: f64) -> f64 {
    if !condition_number.is_finite() {
        NON_FINITE_CONDITION_PENALTY
    } else if condition_number <= CONDITION_SCORE_REFERENCE {
        0.0
    } else {
        (condition_number / CONDITION_SCORE_REFERENCE)
            .log10()
            .max(0.0)
            * CONDITION_SCORE_WEIGHT
    }
}

#[allow(clippy::too_many_arguments)]
fn score_color_candidate(
    candidate: &'static str,
    mapping_strategy: &'static str,
    kind: CandidateKind,
    stats: &MappingStats,
    diagnostics: &ColorspaceDiagnostics,
    anchor_low_support_for_score: [bool; 3],
    neutral_estimate_quality: &NeutralEstimateQuality,
    profile_confidence: Option<f64>,
    target_fit: Option<&TargetFitDiagnostics>,
    rendered_tone_quality: Option<&RenderedToneCandidateDiagnostics>,
    model_quality: Option<&ColorModelCandidateDiagnostics>,
) -> ColorMappingCandidateScore {
    let low_total = stats.pre_scale_clipped_low_ratio.iter().sum::<f64>();
    let low_max = stats
        .pre_scale_clipped_low_ratio
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    let high_total = stats.pre_scale_clipped_high_ratio.iter().sum::<f64>();
    let high_max = stats
        .pre_scale_clipped_high_ratio
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    let neutral_delta_magnitude = score_neutral_delta(stats.neutral_balance_delta);
    let weak_anchor_count = anchor_low_support_for_score
        .iter()
        .filter(|low_support| **low_support)
        .count() as f64;
    let dominant_anchor_quality_score = match kind {
        CandidateKind::ImageDerived | CandidateKind::ScannerPrior => {
            diagnostics.dominant_anchor_quality.score
        }
        CandidateKind::CalibratedDirect
        | CandidateKind::PositiveRgb
        | CandidateKind::NeutralFallback => 1.0,
    };
    let dominant_anchor_unstable_channels = match kind {
        CandidateKind::ImageDerived | CandidateKind::ScannerPrior => {
            diagnostics.dominant_anchor_quality.channel_unstable
        }
        CandidateKind::CalibratedDirect
        | CandidateKind::PositiveRgb
        | CandidateKind::NeutralFallback => [false; 3],
    };
    let target_residual_rms = target_fit.map(|fit| fit.target_residual_rms);
    let target_residual_max = target_fit.map(|fit| fit.target_residual_max);
    let rejection_reason = if kind.is_matrix() {
        low_gamut_fallback_reason(stats, diagnostics.mapping_strategy)
    } else {
        None
    };
    let rejected = rejection_reason.is_some();
    let components = ColorMappingQualityComponents {
        low_gamut_clip_penalty: low_max * LOW_GAMUT_MAX_CHANNEL_PENALTY_WEIGHT
            + low_total * LOW_GAMUT_TOTAL_PENALTY_WEIGHT,
        high_gamut_clip_penalty: high_total * HIGH_GAMUT_TOTAL_PENALTY_WEIGHT,
        preserved_gamut_penalty: (1.0 - stats.pre_scale_preserved_ratio).max(0.0)
            * PRESERVED_GAMUT_PENALTY_WEIGHT,
        exposure_penalty: (stats.exposure_scale - 1.0).max(0.0) * EXPOSURE_PENALTY_WEIGHT,
        neutral_balance_penalty: neutral_delta_magnitude * NEUTRAL_BALANCE_PENALTY_WEIGHT,
        neutral_estimate_penalty: (1.0 - neutral_estimate_quality.score).max(0.0)
            * NEUTRAL_ESTIMATE_PENALTY_WEIGHT,
        anchor_support_penalty: weak_anchor_count * ANCHOR_SUPPORT_PENALTY_PER_WEAK_CHANNEL,
        anchor_stability_penalty: (1.0 - dominant_anchor_quality_score).max(0.0)
            * ANCHOR_STABILITY_PENALTY_WEIGHT,
        condition_penalty: condition_score(diagnostics.condition_number),
        calibration_confidence_penalty: profile_confidence
            .map(|confidence| {
                (1.0 - confidence.clamp(0.0, 1.0)) * PROFILE_CONFIDENCE_PENALTY_WEIGHT
            })
            .unwrap_or_else(|| match kind {
                CandidateKind::ImageDerived => IMAGE_DERIVED_CONFIDENCE_PENALTY,
                CandidateKind::PositiveRgb => 0.0,
                CandidateKind::NeutralFallback => NEUTRAL_FALLBACK_CONFIDENCE_PENALTY,
                CandidateKind::CalibratedDirect | CandidateKind::ScannerPrior => {
                    CALIBRATION_DEFAULT_CONFIDENCE_PENALTY
                }
            }),
        target_residual_penalty: target_residual_rms.unwrap_or(0.0)
            * TARGET_RESIDUAL_RMS_PENALTY_WEIGHT
            + target_residual_max.unwrap_or(0.0) * TARGET_RESIDUAL_MAX_PENALTY_WEIGHT,
        rendered_tone_penalty: rendered_tone_quality
            .map(|diagnostics| diagnostics.rendered_tone_penalty)
            .unwrap_or(0.0),
        tone_chroma_cleanup_penalty: rendered_tone_quality
            .map(|diagnostics| diagnostics.tone_chroma_cleanup_penalty)
            .unwrap_or(0.0),
        density_monotonicity_penalty: model_quality
            .map(|diagnostics| diagnostics.density_monotonicity_penalty)
            .unwrap_or(0.0),
        hue_linearity_penalty: model_quality
            .map(|diagnostics| diagnostics.hue_linearity_penalty)
            .unwrap_or(0.0),
        saturation_preservation_penalty: model_quality
            .map(|diagnostics| diagnostics.saturation_preservation_penalty)
            .unwrap_or(0.0),
        memory_color_penalty: model_quality
            .map(|diagnostics| diagnostics.memory_color_penalty)
            .unwrap_or(0.0),
        spatial_consistency_penalty: model_quality
            .map(|diagnostics| diagnostics.spatial_consistency_penalty)
            .unwrap_or(0.0),
        fallback_penalty: if diagnostics.fallback_used {
            MATRIX_FALLBACK_PENALTY
        } else {
            0.0
        } + kind.fallback_penalty(),
    };
    let technical_safety_score = components.low_gamut_clip_penalty
        + components.high_gamut_clip_penalty
        + components.preserved_gamut_penalty
        + components.exposure_penalty
        + components.condition_penalty
        + components.fallback_penalty;
    let color_fidelity_score = components.neutral_balance_penalty
        + components.neutral_estimate_penalty
        + components.anchor_support_penalty
        + components.anchor_stability_penalty
        + components.calibration_confidence_penalty
        + components.target_residual_penalty
        + components.rendered_tone_penalty
        + components.tone_chroma_cleanup_penalty
        + components.density_monotonicity_penalty
        + components.hue_linearity_penalty
        + components.saturation_preservation_penalty
        + components.memory_color_penalty
        + components.spatial_consistency_penalty;
    let score = technical_safety_score + color_fidelity_score;

    ColorMappingCandidateScore {
        candidate,
        mapping_strategy,
        rank: None,
        selected: false,
        selected_quality_delta: None,
        score,
        quality_score: score,
        technical_safety_score,
        color_fidelity_score,
        rejected,
        rejection_reason,
        pre_scale_clipped_low_ratio: stats.pre_scale_clipped_low_ratio,
        pre_scale_clipped_low_max: low_max,
        pre_scale_clipped_low_total: low_total,
        pre_scale_clipped_high_ratio: stats.pre_scale_clipped_high_ratio,
        pre_scale_clipped_high_max: high_max,
        pre_scale_clipped_high_total: high_total,
        pre_scale_channel_min: stats.pre_scale_channel_min,
        pre_scale_channel_max: stats.pre_scale_channel_max,
        pre_scale_max_low_excursion: stats.pre_scale_max_low_excursion,
        pre_scale_max_high_excursion: stats.pre_scale_max_high_excursion,
        pre_scale_preserved_ratio: stats.pre_scale_preserved_ratio,
        exposure_scale: stats.exposure_scale,
        neutral_balance_delta: stats.neutral_balance_delta,
        neutral_delta_magnitude,
        neutral_estimate_quality_score: neutral_estimate_quality.score,
        channel_anchor_counts: diagnostics.channel_anchor_counts,
        channel_anchor_low_support: diagnostics.channel_anchor_low_support,
        dominant_anchor_quality_score,
        dominant_anchor_unstable_channels,
        condition_number: diagnostics.condition_number,
        profile_confidence,
        target_residual_rms,
        target_residual_max,
        reference_patch_rms_error: None,
        reference_patch_max_error: None,
        reference_patch_mean_delta_e: None,
        reference_patch_rms_delta_e: None,
        reference_patch_max_delta_e: None,
        reference_patch_mean_delta_e2000: None,
        reference_patch_rms_delta_e2000: None,
        reference_patch_max_delta_e2000: None,
        reference_patch_delta_vs_image_derived: None,
        reference_patch_max_delta_vs_image_derived: None,
        reference_patch_delta_e_delta_vs_image_derived: None,
        reference_patch_delta_e_max_delta_vs_image_derived: None,
        reference_patch_delta_e2000_delta_vs_image_derived: None,
        reference_patch_delta_e2000_max_delta_vs_image_derived: None,
        reference_patch_regresses_image_derived: None,
        rendered_tone_quality: rendered_tone_quality.cloned(),
        color_model_quality: model_quality.cloned(),
        quality_components: components,
    }
}

#[derive(Debug, Clone)]
struct PositiveRgbSupportDiagnostics {
    neutral_pixel_count: usize,
    neutral_sample_bands: [usize; 3],
    neutral_sample_rejections: NeutralSampleRejections,
    channel_anchor_counts: [usize; 3],
    channel_anchor_min_count: usize,
    dominant_anchor_rgb: [[f64; 3]; 3],
    dominant_anchor_sample_rejections: DominantAnchorSampleRejections,
    dominant_anchor_bands: [[usize; 3]; 3],
}

fn positive_rgb_support_diagnostics(img: &Array3<f64>) -> PositiveRgbSupportDiagnostics {
    let (h, w, _) = img.dim();
    let total_pixels = h.saturating_mul(w);
    let mut neutral_rejections = NeutralSampleRejections::empty(total_pixels);
    let mut anchor_rejections = DominantAnchorSampleRejections::empty(total_pixels);
    let mut neutral_bands = [0usize; 3];
    let mut anchor_counts = [0usize; 3];
    let mut anchor_bands = [[0usize; 3]; 3];
    let mut anchor_sums = [[0.0f64; 3]; 3];

    for y in 0..h {
        for x in 0..w {
            let rgb = [img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]]];
            let finite = rgb.iter().all(|value| value.is_finite());
            let clipped = finite
                && rgb
                    .iter()
                    .any(|value| *value < -GAMUT_CLIP_EPSILON || *value > 1.0 + GAMUT_CLIP_EPSILON);
            if !finite {
                neutral_rejections.non_finite += 1;
                anchor_rejections.non_finite += 1;
                continue;
            }
            if clipped {
                neutral_rejections.clipped += 1;
                anchor_rejections.clipped += 1;
                continue;
            }

            let luma = rgb_mean_luminance(rgb);
            let saturation = rgb_saturation(rgb);

            if !(NEUTRAL_LUMA_MIN..=NEUTRAL_LUMA_MAX).contains(&luma) {
                neutral_rejections.luma_out_of_range += 1;
            } else if saturation > NEUTRAL_THRESHOLD {
                neutral_rejections.chroma_threshold += 1;
            } else {
                neutral_rejections.accepted_neutral_samples += 1;
                neutral_bands[neutral_luminance_band(luma)] += 1;
            }

            if !(ANCHOR_LUMA_MIN..=ANCHOR_LUMA_MAX).contains(&luma) {
                anchor_rejections.luma_out_of_range += 1;
            } else if saturation < MIN_DOMINANT_SATURATION {
                anchor_rejections.low_saturation += 1;
            } else if let Some(channel) = dominant_rgb_channel(rgb) {
                anchor_rejections.accepted_anchor_samples += 1;
                anchor_counts[channel] += 1;
                anchor_bands[channel][neutral_luminance_band(luma)] += 1;
                for c in 0..3 {
                    anchor_sums[channel][c] += rgb[c];
                }
            } else {
                anchor_rejections.weak_dominance += 1;
            }
        }
    }

    let dominant_anchor_rgb = std::array::from_fn(|channel| {
        if anchor_counts[channel] == 0 {
            [0.0; 3]
        } else {
            std::array::from_fn(|c| anchor_sums[channel][c] / anchor_counts[channel] as f64)
        }
    });

    PositiveRgbSupportDiagnostics {
        neutral_pixel_count: neutral_rejections.accepted_neutral_samples,
        neutral_sample_bands: neutral_bands,
        neutral_sample_rejections: neutral_rejections,
        channel_anchor_counts: anchor_counts,
        channel_anchor_min_count: channel_anchor_min_count(anchor_counts),
        dominant_anchor_rgb,
        dominant_anchor_sample_rejections: anchor_rejections,
        dominant_anchor_bands: anchor_bands,
    }
}

fn positive_rgb_neutral_quality(support: &PositiveRgbSupportDiagnostics) -> NeutralEstimateQuality {
    let total = support.neutral_sample_rejections.total_pixels.max(1);
    let sample_count = support.neutral_pixel_count;
    let populated_band_count = support
        .neutral_sample_bands
        .iter()
        .filter(|count| **count > 0)
        .count();
    let dominant_band_fraction = support
        .neutral_sample_bands
        .iter()
        .copied()
        .max()
        .unwrap_or(0) as f64
        / sample_count.max(1) as f64;

    NeutralEstimateQuality {
        score: 1.0,
        accepted: true,
        broad_support: true,
        reason: "positive RGB passthrough does not derive colour from neutral anchors; neutral samples are diagnostic only".to_string(),
        sample_count,
        sample_fraction: sample_count as f64 / total as f64,
        band_counts: support.neutral_sample_bands,
        populated_band_count,
        dominant_band_fraction,
        minimum_samples: 0,
        minimum_bands: 0,
    }
}

fn positive_rgb_dominant_anchor_quality(
    support: &PositiveRgbSupportDiagnostics,
) -> DominantAnchorQuality {
    let channel_populated_band_count = std::array::from_fn(|channel| {
        support.dominant_anchor_bands[channel]
            .iter()
            .filter(|count| **count > 0)
            .count()
    });
    let channel_dominant_band_fraction = std::array::from_fn(|channel| {
        let count = support.channel_anchor_counts[channel].max(1) as f64;
        support.dominant_anchor_bands[channel]
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as f64
            / count
    });

    DominantAnchorQuality {
        score: 1.0,
        accepted: true,
        reason: "positive RGB passthrough preserves input channel ratios and does not derive colour from dominant anchors".to_string(),
        channel_populated_band_count,
        channel_dominant_band_fraction,
        channel_mean_dominance_margin: [1.0; 3],
        channel_stability_score: [1.0; 3],
        channel_unstable: [false; 3],
        unstable_channel_count: 0,
        minimum_samples_per_channel: 0,
        minimum_bands: 0,
        max_dominant_band_fraction: 1.0,
        dominance_margin_target: 0.0,
    }
}

fn positive_rgb_passthrough_image(
    img: &Array3<f64>,
    exposure_scale: f64,
) -> (Array3<f64>, [f64; 3], [f64; 3], f64) {
    let (h, w, _) = img.dim();
    let total_pixels = (h * w).max(1) as f64;
    let exposure_scale = exposure_scale.max(1e-9);
    let post_scale_clipped_high = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    let post_scale_clipped_low = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    let post_scale_any_clipped = AtomicU64::new(0);
    let mut out = Array3::<f64>::zeros((h, w, 3));

    out.axis_chunks_iter_mut(Axis(0), streaming::DEFAULT_TILE_ROWS)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut out_chunk)| {
            let row_start = chunk_idx * streaming::DEFAULT_TILE_ROWS;
            let chunk_h = out_chunk.dim().0;
            for local_y in 0..chunk_h {
                let y = row_start + local_y;
                for x in 0..w {
                    let mut any_clipped = false;
                    for c in 0..3 {
                        let value = img[[y, x, c]] / exposure_scale;
                        if value > 1.0 + GAMUT_CLIP_EPSILON {
                            post_scale_clipped_high[c].fetch_add(1, Ordering::Relaxed);
                            any_clipped = true;
                        }
                        if value < -GAMUT_CLIP_EPSILON {
                            post_scale_clipped_low[c].fetch_add(1, Ordering::Relaxed);
                            any_clipped = true;
                        }
                        out_chunk[[local_y, x, c]] = if value.is_finite() { value } else { 0.0 };
                    }
                    if any_clipped {
                        post_scale_any_clipped.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });

    (
        out,
        std::array::from_fn(|c| {
            post_scale_clipped_high[c].load(Ordering::Relaxed) as f64 / total_pixels
        }),
        std::array::from_fn(|c| {
            post_scale_clipped_low[c].load(Ordering::Relaxed) as f64 / total_pixels
        }),
        1.0 - post_scale_any_clipped.load(Ordering::Relaxed) as f64 / total_pixels,
    )
}

pub fn map_positive_scan_rgb_to_prophoto_d50_with_diagnostics(
    img: &Array3<f64>,
) -> ColorspaceMappingResult {
    map_positive_scan_rgb_with_profile_state(img, false)
}

pub fn map_profiled_positive_scan_rgb_to_prophoto_d50_with_diagnostics(
    img: &Array3<f64>,
) -> ColorspaceMappingResult {
    map_positive_scan_rgb_with_profile_state(img, true)
}

fn map_positive_scan_rgb_with_profile_state(
    img: &Array3<f64>,
    embedded_icc_applied: bool,
) -> ColorspaceMappingResult {
    let (mapping_strategy, selected_mapping_reason, selected_candidate, acceptance_reason) =
        if embedded_icc_applied {
            (
                "embedded_icc_to_linear_prophoto",
                "embedded ICC device-to-PCS transform and tone reproduction curves were applied before this identity ProPhoto working-space stage",
                "profiled_positive_scan_rgb",
                "positive RGB was transformed from its embedded ICC source space into linear ProPhoto RGB D50",
            )
        } else {
            (
                "positive_rgb_passthrough",
                "already-positive RGB values were preserved provisionally, but no source profile identified their primaries or transfer function; the working-buffer interpretation requires color review",
                "positive_scan_rgb_passthrough",
                "positive RGB values preserved without inventing an image-derived transform; source color space remains unverified",
            )
        };
    let identity = Matrix3::<f64>::identity();
    let stats = evaluate_mapping_stats(img, &identity, [0.5; 3]);
    let support = positive_rgb_support_diagnostics(img);
    let neutral_quality = positive_rgb_neutral_quality(&support);
    let dominant_anchor_quality = positive_rgb_dominant_anchor_quality(&support);
    let rendered_tone_quality =
        evaluate_candidate_rendered_tone(img, &identity, stats.exposure_scale);
    let model_quality = evaluate_candidate_model_quality(img, &identity, stats.exposure_scale);
    let (
        out,
        post_scale_clipped_high_ratio,
        post_scale_clipped_low_ratio,
        post_scale_preserved_ratio,
    ) = positive_rgb_passthrough_image(img, stats.exposure_scale);

    let mut diagnostics = ColorspaceDiagnostics {
        source_white: D50_WHITE,
        work_to_xyz: PROPHOTO_TO_XYZ_D50,
        condition_number: 1.0,
        regularization_lambda: 0.0,
        neutral_pixel_count: support.neutral_pixel_count,
        channel_anchor_counts: support.channel_anchor_counts,
        channel_anchor_min_count: support.channel_anchor_min_count,
        channel_anchor_low_support_threshold: 0,
        channel_anchor_low_support: [false; 3],
        dominant_anchor_rgb: support.dominant_anchor_rgb,
        dominant_anchor_sample_rejections: support.dominant_anchor_sample_rejections,
        dominant_anchor_bands: support.dominant_anchor_bands,
        dominant_anchor_quality,
        neutral_sample_bands: support.neutral_sample_bands,
        highlight_percentile: HIGHLIGHT_HEADROOM_PERCENTILE,
        pre_scale_channel_max: stats.pre_scale_channel_max,
        pre_scale_channel_high_percentile: stats.pre_scale_channel_high_percentile,
        pre_scale_clipped_high_ratio: stats.pre_scale_clipped_high_ratio,
        pre_scale_clipped_low_ratio: stats.pre_scale_clipped_low_ratio,
        post_scale_clipped_high_ratio,
        post_scale_clipped_low_ratio,
        exposure_scale: stats.exposure_scale,
        fallback_used: false,
        weak_anchor_fallback_used: false,
        weak_anchor_fallback_reason: None,
        gamut_fallback_used: false,
        gamut_fallback_reason: None,
        mapping_strategy,
        selected_mapping_reason: selected_mapping_reason.to_string(),
        candidate_scores: Vec::new(),
        candidate_acceptance: Vec::new(),
        selected_candidate: selected_candidate.to_string(),
        selected_candidate_rank: Some(1),
        selected_candidate_score: None,
        selected_quality_score: None,
        technical_safety_score: None,
        color_fidelity_score: None,
        selected_runner_up_quality_delta: None,
        selection_rejections: Vec::new(),
        candidate_risk: "safe".to_string(),
        neutral_sample_rejections: support.neutral_sample_rejections,
        reference_patch_evaluation: None,
        neutral_estimate_quality: neutral_quality,
        neutral_balance_scale: [1.0; 3],
        neutral_trim_scale: [1.0; 3],
        neutral_trim_applied: false,
        neutral_trim_before_after: NeutralTrimDiagnostics::skipped(
            "positive RGB passthrough does not apply image-derived neutral trim",
        ),
        calibration_acceptance: CalibrationAcceptanceDiagnostics::not_applicable(
            ColorMode::Auto,
            if embedded_icc_applied {
                "external scanner calibration is not required after the embedded ICC profile established the positive input source color space"
            } else {
                "positive RGB passthrough selected because no source calibration/profile was configured; colorimetric identity is unverified"
            },
        ),
        neutral_safety_rescue: NeutralSafetyRescueDiagnostics::not_evaluated(
            "neutral safety rescue is not applicable to the positive-input color path",
        ),
        nonlinear_color_model: None,
        pre_scale_preserved_ratio: stats.pre_scale_preserved_ratio,
        post_scale_preserved_ratio,
        image_matrix_pre_scale_clipped_low_ratio: stats.pre_scale_clipped_low_ratio,
        image_matrix_pre_scale_clipped_high_ratio: stats.pre_scale_clipped_high_ratio,
        image_matrix_exposure_scale: stats.exposure_scale,
        image_matrix_pre_scale_preserved_ratio: stats.pre_scale_preserved_ratio,
        image_matrix_neutral_balance_delta: [0.0; 3],
        calibrated_profile_pre_scale_clipped_low_ratio: None,
        calibrated_profile_pre_scale_clipped_high_ratio: None,
        calibrated_profile_exposure_scale: None,
        calibrated_profile_pre_scale_preserved_ratio: None,
        calibrated_profile_neutral_balance_delta: None,
    };

    let mut score = score_color_candidate(
        selected_candidate,
        mapping_strategy,
        CandidateKind::PositiveRgb,
        &stats,
        &diagnostics,
        [false; 3],
        &diagnostics.neutral_estimate_quality,
        Some(1.0),
        None,
        Some(&rendered_tone_quality),
        Some(&model_quality),
    );
    score.rank = Some(1);
    score.selected = true;
    score.selected_quality_delta = Some(0.0);
    diagnostics.selected_candidate_score = Some(score.score);
    diagnostics.selected_quality_score = Some(score.quality_score);
    diagnostics.technical_safety_score = Some(score.technical_safety_score);
    diagnostics.color_fidelity_score = Some(score.color_fidelity_score);
    diagnostics.candidate_scores = vec![score.clone()];
    diagnostics.candidate_acceptance = vec![ColorMappingCandidateAcceptanceDiagnostics {
        candidate: score.candidate.to_string(),
        candidate_kind: CandidateKind::PositiveRgb.as_str().to_string(),
        mapping_strategy: score.mapping_strategy.to_string(),
        source_label: if embedded_icc_applied {
            "embedded ICC profile transform"
        } else {
            "positive RGB passthrough"
        }
        .to_string(),
        status: "selected".to_string(),
        reason: acceptance_reason.to_string(),
        rank: Some(1),
        selected: true,
        eligible_in_color_mode: true,
        quality_score: score.quality_score,
        selected_quality_delta: Some(0.0),
        rejected: score.rejected,
        rejection_reason: score.rejection_reason.clone(),
        beats_image_derived: None,
        within_negative_gamut_limits: None,
    }];
    diagnostics.candidate_risk = candidate_risk_for_diagnostics(&diagnostics);

    ColorspaceMappingResult {
        prophoto: out,
        diagnostics,
        comparison_thumbnail: None,
    }
}

fn lab_f(value: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if value > DELTA * DELTA * DELTA {
        value.cbrt()
    } else {
        value / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

pub fn xyz_d50_to_lab(xyz: [f64; 3]) -> [f64; 3] {
    let fx = lab_f(xyz[0] / D50_WHITE[0].max(1e-9));
    let fy = lab_f(xyz[1] / D50_WHITE[1].max(1e-9));
    let fz = lab_f(xyz[2] / D50_WHITE[2].max(1e-9));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

pub fn lab_to_xyz_d50(lab: [f64; 3]) -> [f64; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    const DELTA: f64 = 6.0 / 29.0;
    let inverse_lab = |value: f64| {
        if value > DELTA {
            value.powi(3)
        } else {
            3.0 * DELTA * DELTA * (value - 4.0 / 29.0)
        }
    };
    [
        D50_WHITE[0] * inverse_lab(fx),
        D50_WHITE[1] * inverse_lab(fy),
        D50_WHITE[2] * inverse_lab(fz),
    ]
}

fn reference_hue_family(reference_xyz: [f64; 3]) -> &'static str {
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

fn patch_error(mapped_xyz: Vector3<f64>, reference_xyz: [f64; 3]) -> f64 {
    ((mapped_xyz[0] - reference_xyz[0]).powi(2)
        + (mapped_xyz[1] - reference_xyz[1]).powi(2)
        + (mapped_xyz[2] - reference_xyz[2]).powi(2))
    .sqrt()
}

fn patch_delta_e76(mapped_xyz: Vector3<f64>, reference_xyz: [f64; 3]) -> f64 {
    let mapped_lab = xyz_d50_to_lab(vector3_to_array(mapped_xyz));
    let reference_lab = xyz_d50_to_lab(reference_xyz);
    ((mapped_lab[0] - reference_lab[0]).powi(2)
        + (mapped_lab[1] - reference_lab[1]).powi(2)
        + (mapped_lab[2] - reference_lab[2]).powi(2))
    .sqrt()
}

fn patch_delta_e2000(mapped_xyz: Vector3<f64>, reference_xyz: [f64; 3]) -> f64 {
    let mapped_lab = xyz_d50_to_lab(vector3_to_array(mapped_xyz));
    let reference_lab = xyz_d50_to_lab(reference_xyz);
    lab_delta_e2000(mapped_lab, reference_lab)
}

pub fn lab_delta_e2000(left: [f64; 3], right: [f64; 3]) -> f64 {
    let (l1, a1, b1) = (left[0], left[1], left[2]);
    let (l2, a2, b2) = (right[0], right[1], right[2]);
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = 0.5 * (c1 + c2);
    let c_bar7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + 25.0f64.powi(7))).sqrt());
    let a1_prime = (1.0 + g) * a1;
    let a2_prime = (1.0 + g) * a2;
    let c1_prime = (a1_prime * a1_prime + b1 * b1).sqrt();
    let c2_prime = (a2_prime * a2_prime + b2 * b2).sqrt();
    let h1_prime = lab_hue_degrees(a1_prime, b1, c1_prime);
    let h2_prime = lab_hue_degrees(a2_prime, b2, c2_prime);

    let delta_l_prime = l2 - l1;
    let delta_c_prime = c2_prime - c1_prime;
    let delta_h_prime = if c1_prime * c2_prime <= f64::EPSILON {
        0.0
    } else {
        let mut delta = h2_prime - h1_prime;
        if delta > 180.0 {
            delta -= 360.0;
        } else if delta < -180.0 {
            delta += 360.0;
        }
        delta
    };
    let delta_h_term =
        2.0 * (c1_prime * c2_prime).sqrt() * (0.5 * delta_h_prime).to_radians().sin();

    let l_bar_prime = 0.5 * (l1 + l2);
    let c_bar_prime = 0.5 * (c1_prime + c2_prime);
    let h_bar_prime = if c1_prime * c2_prime <= f64::EPSILON {
        h1_prime + h2_prime
    } else if (h1_prime - h2_prime).abs() <= 180.0 {
        0.5 * (h1_prime + h2_prime)
    } else if h1_prime + h2_prime < 360.0 {
        0.5 * (h1_prime + h2_prime + 360.0)
    } else {
        0.5 * (h1_prime + h2_prime - 360.0)
    };

    let t = 1.0 - 0.17 * (h_bar_prime - 30.0).to_radians().cos()
        + 0.24 * (2.0 * h_bar_prime).to_radians().cos()
        + 0.32 * (3.0 * h_bar_prime + 6.0).to_radians().cos()
        - 0.20 * (4.0 * h_bar_prime - 63.0).to_radians().cos();
    let delta_theta = 30.0 * (-((h_bar_prime - 275.0) / 25.0).powi(2)).exp();
    let c_bar_prime7 = c_bar_prime.powi(7);
    let r_c = 2.0 * (c_bar_prime7 / (c_bar_prime7 + 25.0f64.powi(7))).sqrt();
    let l_centered = l_bar_prime - 50.0;
    let s_l = 1.0 + (0.015 * l_centered * l_centered) / (20.0 + l_centered * l_centered).sqrt();
    let s_c = 1.0 + 0.045 * c_bar_prime;
    let s_h = 1.0 + 0.015 * c_bar_prime * t;
    let r_t = -(2.0 * delta_theta).to_radians().sin() * r_c;

    let l_term = delta_l_prime / s_l;
    let c_term = delta_c_prime / s_c;
    let h_term = delta_h_term / s_h;
    (l_term * l_term + c_term * c_term + h_term * h_term + r_t * c_term * h_term)
        .max(0.0)
        .sqrt()
}

fn lab_hue_degrees(a: f64, b: f64, chroma: f64) -> f64 {
    if chroma <= f64::EPSILON {
        return 0.0;
    }
    let mut hue = b.atan2(a).to_degrees();
    if hue < 0.0 {
        hue += 360.0;
    }
    hue
}

#[derive(Debug, Default)]
struct ReferenceHueResidualAccum {
    patch_count: usize,
    error_sum_sq: f64,
    max_error: f64,
    delta_e_sum_sq: f64,
    max_delta_e: f64,
    delta_e2000_sum_sq: f64,
    max_delta_e2000: f64,
}

#[derive(Debug, Default)]
struct ReferenceHueRegressionAccum {
    patch_count: usize,
    candidate_error_sum_sq: f64,
    image_error_sum_sq: f64,
    candidate_max_error: f64,
    image_max_error: f64,
    candidate_delta_e_sum_sq: f64,
    image_delta_e_sum_sq: f64,
    candidate_max_delta_e: f64,
    image_max_delta_e: f64,
    candidate_delta_e2000_sum_sq: f64,
    image_delta_e2000_sum_sq: f64,
    candidate_max_delta_e2000: f64,
    image_max_delta_e2000: f64,
}

fn candidate_patch_evaluation(
    candidate: &ColorMappingCandidate,
    patches: &[TargetPatch],
    prophoto_to_xyz: &Matrix3<f64>,
) -> ReferenceCandidatePatchEvaluation {
    let mut sum_sq = 0.0f64;
    let mut sum = 0.0f64;
    let mut max_error = 0.0f64;
    let mut delta_e_sum_sq = 0.0f64;
    let mut delta_e_sum = 0.0f64;
    let mut max_delta_e = 0.0f64;
    let mut delta_e2000_sum_sq = 0.0f64;
    let mut delta_e2000_sum = 0.0f64;
    let mut max_delta_e2000 = 0.0f64;
    let mut hue_sums = std::collections::BTreeMap::<String, ReferenceHueResidualAccum>::new();
    for patch in patches {
        let source = Vector3::new(
            patch.source_rgb[0],
            patch.source_rgb[1],
            patch.source_rgb[2],
        );
        let mapped_xyz = prophoto_to_xyz * map_candidate_pixel(candidate, source);
        let error = patch_error(mapped_xyz, patch.reference_xyz);
        let delta_e = patch_delta_e76(mapped_xyz, patch.reference_xyz);
        let delta_e2000 = patch_delta_e2000(mapped_xyz, patch.reference_xyz);
        let hue_family = reference_hue_family(patch.reference_xyz).to_string();
        let entry = hue_sums.entry(hue_family).or_default();
        entry.patch_count += 1;
        entry.error_sum_sq += error * error;
        entry.max_error = entry.max_error.max(error);
        entry.delta_e_sum_sq += delta_e * delta_e;
        entry.max_delta_e = entry.max_delta_e.max(delta_e);
        entry.delta_e2000_sum_sq += delta_e2000 * delta_e2000;
        entry.max_delta_e2000 = entry.max_delta_e2000.max(delta_e2000);
        sum_sq += error * error;
        sum += error;
        max_error = max_error.max(error);
        delta_e_sum_sq += delta_e * delta_e;
        delta_e_sum += delta_e;
        max_delta_e = max_delta_e.max(delta_e);
        delta_e2000_sum_sq += delta_e2000 * delta_e2000;
        delta_e2000_sum += delta_e2000;
        max_delta_e2000 = max_delta_e2000.max(delta_e2000);
    }
    let patch_count = patches.len().max(1);
    let mut worst_hue_families = hue_sums
        .into_iter()
        .map(|(hue_family, accum)| {
            let patch_count = accum.patch_count.max(1);
            ReferenceHueFamilyResidual {
                hue_family,
                patch_count: accum.patch_count,
                rms_error: (accum.error_sum_sq / patch_count as f64).sqrt(),
                max_error: accum.max_error,
                rms_delta_e: (accum.delta_e_sum_sq / patch_count as f64).sqrt(),
                max_delta_e: accum.max_delta_e,
                rms_delta_e2000: (accum.delta_e2000_sum_sq / patch_count as f64).sqrt(),
                max_delta_e2000: accum.max_delta_e2000,
            }
        })
        .collect::<Vec<_>>();
    worst_hue_families.sort_by(|left, right| {
        right
            .max_error
            .partial_cmp(&left.max_error)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .rms_error
                    .partial_cmp(&left.rms_error)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left.hue_family.cmp(&right.hue_family))
    });
    worst_hue_families.truncate(3);

    ReferenceCandidatePatchEvaluation {
        candidate: candidate.candidate.to_string(),
        patch_count: patches.len(),
        rms_error: (sum_sq / patch_count as f64).sqrt(),
        max_error,
        mean_error: sum / patch_count as f64,
        rms_delta_e: (delta_e_sum_sq / patch_count as f64).sqrt(),
        max_delta_e,
        mean_delta_e: delta_e_sum / patch_count as f64,
        rms_delta_e2000: (delta_e2000_sum_sq / patch_count as f64).sqrt(),
        max_delta_e2000,
        mean_delta_e2000: delta_e2000_sum / patch_count as f64,
        rms_delta_vs_image_derived: None,
        max_delta_vs_image_derived: None,
        delta_e_rms_delta_vs_image_derived: None,
        delta_e_max_delta_vs_image_derived: None,
        delta_e2000_rms_delta_vs_image_derived: None,
        delta_e2000_max_delta_vs_image_derived: None,
        regresses_image_derived: None,
        worst_hue_families,
        hue_family_regressions: Vec::new(),
    }
}

fn candidate_hue_family_regressions(
    candidate: &ColorMappingCandidate,
    image_candidate: &ColorMappingCandidate,
    patches: &[TargetPatch],
    prophoto_to_xyz: &Matrix3<f64>,
) -> Vec<ReferenceHueFamilyRegression> {
    let mut hue_sums = std::collections::BTreeMap::<String, ReferenceHueRegressionAccum>::new();
    for patch in patches {
        let source = Vector3::new(
            patch.source_rgb[0],
            patch.source_rgb[1],
            patch.source_rgb[2],
        );
        let candidate_xyz = prophoto_to_xyz * map_candidate_pixel(candidate, source);
        let image_xyz = prophoto_to_xyz * map_candidate_pixel(image_candidate, source);
        let candidate_error = patch_error(candidate_xyz, patch.reference_xyz);
        let image_error = patch_error(image_xyz, patch.reference_xyz);
        let candidate_delta_e = patch_delta_e76(candidate_xyz, patch.reference_xyz);
        let image_delta_e = patch_delta_e76(image_xyz, patch.reference_xyz);
        let candidate_delta_e2000 = patch_delta_e2000(candidate_xyz, patch.reference_xyz);
        let image_delta_e2000 = patch_delta_e2000(image_xyz, patch.reference_xyz);
        let hue_family = reference_hue_family(patch.reference_xyz).to_string();
        let entry = hue_sums.entry(hue_family).or_default();
        entry.patch_count += 1;
        entry.candidate_error_sum_sq += candidate_error * candidate_error;
        entry.image_error_sum_sq += image_error * image_error;
        entry.candidate_max_error = entry.candidate_max_error.max(candidate_error);
        entry.image_max_error = entry.image_max_error.max(image_error);
        entry.candidate_delta_e_sum_sq += candidate_delta_e * candidate_delta_e;
        entry.image_delta_e_sum_sq += image_delta_e * image_delta_e;
        entry.candidate_max_delta_e = entry.candidate_max_delta_e.max(candidate_delta_e);
        entry.image_max_delta_e = entry.image_max_delta_e.max(image_delta_e);
        entry.candidate_delta_e2000_sum_sq += candidate_delta_e2000 * candidate_delta_e2000;
        entry.image_delta_e2000_sum_sq += image_delta_e2000 * image_delta_e2000;
        entry.candidate_max_delta_e2000 =
            entry.candidate_max_delta_e2000.max(candidate_delta_e2000);
        entry.image_max_delta_e2000 = entry.image_max_delta_e2000.max(image_delta_e2000);
    }

    let mut regressions = hue_sums
        .into_iter()
        .filter_map(|(hue_family, accum)| {
            let patch_count = accum.patch_count.max(1);
            let candidate_rms = (accum.candidate_error_sum_sq / patch_count as f64).sqrt();
            let image_rms = (accum.image_error_sum_sq / patch_count as f64).sqrt();
            let rms_delta = candidate_rms - image_rms;
            let max_delta = accum.candidate_max_error - accum.image_max_error;
            let candidate_rms_delta_e =
                (accum.candidate_delta_e_sum_sq / patch_count as f64).sqrt();
            let image_rms_delta_e = (accum.image_delta_e_sum_sq / patch_count as f64).sqrt();
            let delta_e_rms_delta = candidate_rms_delta_e - image_rms_delta_e;
            let delta_e_max_delta = accum.candidate_max_delta_e - accum.image_max_delta_e;
            let candidate_rms_delta_e2000 =
                (accum.candidate_delta_e2000_sum_sq / patch_count as f64).sqrt();
            let image_rms_delta_e2000 =
                (accum.image_delta_e2000_sum_sq / patch_count as f64).sqrt();
            let delta_e2000_rms_delta = candidate_rms_delta_e2000 - image_rms_delta_e2000;
            let delta_e2000_max_delta =
                accum.candidate_max_delta_e2000 - accum.image_max_delta_e2000;
            reference_patch_regresses_image_derived(
                rms_delta,
                max_delta,
                delta_e_rms_delta,
                delta_e_max_delta,
                delta_e2000_rms_delta,
                delta_e2000_max_delta,
            )
            .then_some(ReferenceHueFamilyRegression {
                hue_family,
                patch_count: accum.patch_count,
                candidate_rms_error: candidate_rms,
                image_derived_rms_error: image_rms,
                rms_delta_vs_image_derived: rms_delta,
                candidate_max_error: accum.candidate_max_error,
                image_derived_max_error: accum.image_max_error,
                max_delta_vs_image_derived: max_delta,
                candidate_rms_delta_e,
                image_derived_rms_delta_e: image_rms_delta_e,
                delta_e_rms_delta_vs_image_derived: delta_e_rms_delta,
                candidate_max_delta_e: accum.candidate_max_delta_e,
                image_derived_max_delta_e: accum.image_max_delta_e,
                delta_e_max_delta_vs_image_derived: delta_e_max_delta,
                candidate_rms_delta_e2000,
                image_derived_rms_delta_e2000: image_rms_delta_e2000,
                delta_e2000_rms_delta_vs_image_derived: delta_e2000_rms_delta,
                candidate_max_delta_e2000: accum.candidate_max_delta_e2000,
                image_derived_max_delta_e2000: accum.image_max_delta_e2000,
                delta_e2000_max_delta_vs_image_derived: delta_e2000_max_delta,
            })
        })
        .collect::<Vec<_>>();
    regressions.sort_by(|left, right| {
        right
            .max_delta_vs_image_derived
            .partial_cmp(&left.max_delta_vs_image_derived)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .delta_e_max_delta_vs_image_derived
                    .partial_cmp(&left.delta_e_max_delta_vs_image_derived)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                right
                    .rms_delta_vs_image_derived
                    .partial_cmp(&left.rms_delta_vs_image_derived)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                right
                    .delta_e_rms_delta_vs_image_derived
                    .partial_cmp(&left.delta_e_rms_delta_vs_image_derived)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left.hue_family.cmp(&right.hue_family))
    });
    regressions.truncate(3);
    regressions
}

fn reference_patch_regresses_image_derived(
    rms_delta: f64,
    max_delta: f64,
    delta_e_rms_delta: f64,
    delta_e_max_delta: f64,
    delta_e2000_rms_delta: f64,
    delta_e2000_max_delta: f64,
) -> bool {
    rms_delta > REFERENCE_RMS_REGRESSION_TOLERANCE
        || max_delta > REFERENCE_MAX_REGRESSION_TOLERANCE
        || delta_e_rms_delta > REFERENCE_DELTA_E_RMS_REGRESSION_TOLERANCE
        || delta_e_max_delta > REFERENCE_DELTA_E_MAX_REGRESSION_TOLERANCE
        || delta_e2000_rms_delta > REFERENCE_DELTA_E2000_RMS_REGRESSION_TOLERANCE
        || delta_e2000_max_delta > REFERENCE_DELTA_E2000_MAX_REGRESSION_TOLERANCE
}

fn reference_patch_improves_image_derived(
    rms_delta: f64,
    max_delta: f64,
    delta_e_rms_delta: f64,
    delta_e_max_delta: f64,
    delta_e2000_rms_delta: f64,
    delta_e2000_max_delta: f64,
) -> bool {
    rms_delta < -REFERENCE_RMS_REGRESSION_TOLERANCE
        || max_delta < -REFERENCE_MAX_REGRESSION_TOLERANCE
        || delta_e_rms_delta < -REFERENCE_DELTA_E_RMS_REGRESSION_TOLERANCE
        || delta_e_max_delta < -REFERENCE_DELTA_E_MAX_REGRESSION_TOLERANCE
        || delta_e2000_rms_delta < -REFERENCE_DELTA_E2000_RMS_REGRESSION_TOLERANCE
        || delta_e2000_max_delta < -REFERENCE_DELTA_E2000_MAX_REGRESSION_TOLERANCE
}

fn apply_reference_patch_evaluation(
    candidates: &mut [ColorMappingCandidate],
    patches: &[TargetPatch],
    selected_index: Option<usize>,
) -> Option<ReferencePatchEvaluation> {
    if patches.is_empty() {
        return None;
    }
    let image_index = candidates
        .iter()
        .position(|candidate| candidate.kind == CandidateKind::ImageDerived)?;
    let selected_index = selected_index.unwrap_or(image_index);
    let prophoto_to_xyz = prophoto_to_xyz_d50_matrix();
    let mut candidate_evaluations = candidates
        .iter()
        .filter(|candidate| candidate.kind.is_matrix())
        .map(|candidate| candidate_patch_evaluation(candidate, patches, &prophoto_to_xyz))
        .collect::<Vec<_>>();
    let image_eval = candidate_evaluations
        .iter()
        .find(|evaluation| evaluation.candidate == candidates[image_index].candidate)?
        .clone();
    let image_candidate = candidates[image_index].clone();

    for evaluation in candidate_evaluations.iter_mut() {
        let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.candidate == evaluation.candidate)
        else {
            continue;
        };
        let rms_delta = evaluation.rms_error - image_eval.rms_error;
        let max_delta = evaluation.max_error - image_eval.max_error;
        let delta_e_rms_delta = evaluation.rms_delta_e - image_eval.rms_delta_e;
        let delta_e_max_delta = evaluation.max_delta_e - image_eval.max_delta_e;
        let delta_e2000_rms_delta = evaluation.rms_delta_e2000 - image_eval.rms_delta_e2000;
        let delta_e2000_max_delta = evaluation.max_delta_e2000 - image_eval.max_delta_e2000;
        evaluation.rms_delta_vs_image_derived = Some(rms_delta);
        evaluation.max_delta_vs_image_derived = Some(max_delta);
        evaluation.delta_e_rms_delta_vs_image_derived = Some(delta_e_rms_delta);
        evaluation.delta_e_max_delta_vs_image_derived = Some(delta_e_max_delta);
        evaluation.delta_e2000_rms_delta_vs_image_derived = Some(delta_e2000_rms_delta);
        evaluation.delta_e2000_max_delta_vs_image_derived = Some(delta_e2000_max_delta);
        evaluation.regresses_image_derived = Some(reference_patch_regresses_image_derived(
            rms_delta,
            max_delta,
            delta_e_rms_delta,
            delta_e_max_delta,
            delta_e2000_rms_delta,
            delta_e2000_max_delta,
        ));
        evaluation.hue_family_regressions = candidate_hue_family_regressions(
            candidate,
            &image_candidate,
            patches,
            &prophoto_to_xyz,
        );
    }

    for candidate in candidates.iter_mut() {
        if let Some(evaluation) = candidate_evaluations
            .iter()
            .find(|evaluation| evaluation.candidate == candidate.candidate)
        {
            let rms_delta = evaluation.rms_error - image_eval.rms_error;
            let max_delta = evaluation.max_error - image_eval.max_error;
            let delta_e_rms_delta = evaluation.rms_delta_e - image_eval.rms_delta_e;
            let delta_e_max_delta = evaluation.max_delta_e - image_eval.max_delta_e;
            let delta_e2000_rms_delta = evaluation.rms_delta_e2000 - image_eval.rms_delta_e2000;
            let delta_e2000_max_delta = evaluation.max_delta_e2000 - image_eval.max_delta_e2000;
            let reference_residual_penalty = evaluation.rms_error
                * TARGET_RESIDUAL_RMS_PENALTY_WEIGHT
                + evaluation.max_error * TARGET_RESIDUAL_MAX_PENALTY_WEIGHT
                + evaluation.rms_delta_e * REFERENCE_DELTA_E_RMS_PENALTY_WEIGHT
                + evaluation.max_delta_e * REFERENCE_DELTA_E_MAX_PENALTY_WEIGHT
                + evaluation.rms_delta_e2000 * REFERENCE_DELTA_E2000_RMS_PENALTY_WEIGHT
                + evaluation.max_delta_e2000 * REFERENCE_DELTA_E2000_MAX_PENALTY_WEIGHT;
            let previous_residual_penalty =
                candidate.score.quality_components.target_residual_penalty;
            if reference_residual_penalty > previous_residual_penalty {
                let penalty_delta = reference_residual_penalty - previous_residual_penalty;
                candidate.score.quality_components.target_residual_penalty =
                    reference_residual_penalty;
                candidate.score.color_fidelity_score += penalty_delta;
                candidate.score.quality_score += penalty_delta;
                candidate.score.score += penalty_delta;
            }
            candidate.score.reference_patch_rms_error = Some(evaluation.rms_error);
            candidate.score.reference_patch_max_error = Some(evaluation.max_error);
            candidate.score.reference_patch_mean_delta_e = Some(evaluation.mean_delta_e);
            candidate.score.reference_patch_rms_delta_e = Some(evaluation.rms_delta_e);
            candidate.score.reference_patch_max_delta_e = Some(evaluation.max_delta_e);
            candidate.score.reference_patch_mean_delta_e2000 = Some(evaluation.mean_delta_e2000);
            candidate.score.reference_patch_rms_delta_e2000 = Some(evaluation.rms_delta_e2000);
            candidate.score.reference_patch_max_delta_e2000 = Some(evaluation.max_delta_e2000);
            candidate.score.reference_patch_delta_vs_image_derived = Some(rms_delta);
            candidate.score.reference_patch_max_delta_vs_image_derived = Some(max_delta);
            candidate
                .score
                .reference_patch_delta_e_delta_vs_image_derived = Some(delta_e_rms_delta);
            candidate
                .score
                .reference_patch_delta_e_max_delta_vs_image_derived = Some(delta_e_max_delta);
            candidate
                .score
                .reference_patch_delta_e2000_delta_vs_image_derived = Some(delta_e2000_rms_delta);
            candidate
                .score
                .reference_patch_delta_e2000_max_delta_vs_image_derived =
                Some(delta_e2000_max_delta);
            candidate.score.reference_patch_regresses_image_derived =
                Some(reference_patch_regresses_image_derived(
                    rms_delta,
                    max_delta,
                    delta_e_rms_delta,
                    delta_e_max_delta,
                    delta_e2000_rms_delta,
                    delta_e2000_max_delta,
                ));
        }
    }

    let selected_eval = candidate_evaluations
        .iter()
        .find(|evaluation| evaluation.candidate == candidates[selected_index].candidate)
        .cloned()
        .unwrap_or_else(|| image_eval.clone());
    let rms_delta = selected_eval.rms_error - image_eval.rms_error;
    let max_delta = selected_eval.max_error - image_eval.max_error;
    let delta_e_rms_delta = selected_eval.rms_delta_e - image_eval.rms_delta_e;
    let delta_e_max_delta = selected_eval.max_delta_e - image_eval.max_delta_e;
    let delta_e2000_rms_delta = selected_eval.rms_delta_e2000 - image_eval.rms_delta_e2000;
    let delta_e2000_max_delta = selected_eval.max_delta_e2000 - image_eval.max_delta_e2000;
    let selected_improves_image_derived = reference_patch_improves_image_derived(
        rms_delta,
        max_delta,
        delta_e_rms_delta,
        delta_e_max_delta,
        delta_e2000_rms_delta,
        delta_e2000_max_delta,
    );
    let selected_regresses_image_derived = reference_patch_regresses_image_derived(
        rms_delta,
        max_delta,
        delta_e_rms_delta,
        delta_e_max_delta,
        delta_e2000_rms_delta,
        delta_e2000_max_delta,
    );

    let selected = &candidates[selected_index];
    let image = &candidates[image_index];
    let per_patch = patches
        .iter()
        .map(|patch| {
            let source = Vector3::new(
                patch.source_rgb[0],
                patch.source_rgb[1],
                patch.source_rgb[2],
            );
            let selected_xyz = prophoto_to_xyz * (selected.matrix * source);
            let image_derived_xyz = prophoto_to_xyz * (image.matrix * source);
            let selected_error = patch_error(selected_xyz, patch.reference_xyz);
            let image_derived_error = patch_error(image_derived_xyz, patch.reference_xyz);
            let selected_delta_e = patch_delta_e76(selected_xyz, patch.reference_xyz);
            let image_derived_delta_e = patch_delta_e76(image_derived_xyz, patch.reference_xyz);
            let selected_delta_e2000 = patch_delta_e2000(selected_xyz, patch.reference_xyz);
            let image_derived_delta_e2000 =
                patch_delta_e2000(image_derived_xyz, patch.reference_xyz);
            ReferencePatchResidual {
                patch_id: patch.patch_id.clone(),
                hue_family: reference_hue_family(patch.reference_xyz).to_string(),
                source_rgb: patch.source_rgb,
                reference_xyz: patch.reference_xyz,
                selected_xyz: vector3_to_array(selected_xyz),
                image_derived_xyz: vector3_to_array(image_derived_xyz),
                selected_error,
                image_derived_error,
                selected_delta_e,
                image_derived_delta_e,
                selected_delta_e2000,
                image_derived_delta_e2000,
                selected_delta_vs_image_derived: selected_error - image_derived_error,
                selected_delta_e_vs_image_derived: selected_delta_e - image_derived_delta_e,
                selected_delta_e2000_vs_image_derived: selected_delta_e2000
                    - image_derived_delta_e2000,
            }
        })
        .collect::<Vec<_>>();

    Some(ReferencePatchEvaluation {
        patch_count: patches.len(),
        selected_candidate: selected.candidate.to_string(),
        image_derived_candidate: image.candidate.to_string(),
        selected_rms_error: selected_eval.rms_error,
        selected_max_error: selected_eval.max_error,
        image_derived_rms_error: image_eval.rms_error,
        image_derived_max_error: image_eval.max_error,
        selected_rms_delta_e: selected_eval.rms_delta_e,
        selected_max_delta_e: selected_eval.max_delta_e,
        image_derived_rms_delta_e: image_eval.rms_delta_e,
        image_derived_max_delta_e: image_eval.max_delta_e,
        selected_rms_delta_e2000: selected_eval.rms_delta_e2000,
        selected_max_delta_e2000: selected_eval.max_delta_e2000,
        image_derived_rms_delta_e2000: image_eval.rms_delta_e2000,
        image_derived_max_delta_e2000: image_eval.max_delta_e2000,
        rms_error_delta_vs_image_derived: rms_delta,
        max_error_delta_vs_image_derived: max_delta,
        delta_e_rms_delta_vs_image_derived: delta_e_rms_delta,
        delta_e_max_delta_vs_image_derived: delta_e_max_delta,
        delta_e2000_rms_delta_vs_image_derived: delta_e2000_rms_delta,
        delta_e2000_max_delta_vs_image_derived: delta_e2000_max_delta,
        selected_improves_image_derived,
        selected_regresses_image_derived,
        worst_hue_families: selected_eval.worst_hue_families.clone(),
        selected_hue_family_regressions: selected_eval.hue_family_regressions.clone(),
        per_patch,
        candidate_evaluations,
    })
}

pub fn image_matrix_low_clip_summary(diagnostics: &ColorspaceDiagnostics) -> (f64, f64) {
    let max_channel = diagnostics
        .image_matrix_pre_scale_clipped_low_ratio
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    let total = diagnostics
        .image_matrix_pre_scale_clipped_low_ratio
        .iter()
        .sum::<f64>();
    (max_channel, total)
}

pub fn has_destructive_gamut_fallback(diagnostics: &ColorspaceDiagnostics) -> bool {
    let (max_channel, total) = image_matrix_low_clip_summary(diagnostics);
    diagnostics.gamut_fallback_used
        && (max_channel > MAX_LOW_GAMUT_CLIP_RATIO_PER_CHANNEL
            || total > MAX_LOW_GAMUT_CLIP_RATIO_TOTAL)
}

/// Returns true only when the mapping that would actually be rendered has lost enough gamut
/// evidence that even a held-out-validated upstream negative response must not force selection.
/// Image-matrix *candidate* clipping is deliberately not used here: a safe selected fallback can
/// legitimately supersede a rejected image-derived matrix.
pub fn has_catastrophic_render_mapping(diagnostics: &ColorspaceDiagnostics) -> bool {
    let post_scale_clip_total = diagnostics
        .post_scale_clipped_low_ratio
        .iter()
        .chain(diagnostics.post_scale_clipped_high_ratio.iter())
        .sum::<f64>();
    !diagnostics.post_scale_preserved_ratio.is_finite()
        || diagnostics.post_scale_preserved_ratio < CATASTROPHIC_RENDER_MIN_PRESERVED_GAMUT
        || !post_scale_clip_total.is_finite()
        || post_scale_clip_total > CATASTROPHIC_RENDER_MAX_POST_SCALE_CLIP_TOTAL
}

fn candidate_risk_severity(risk: &str) -> usize {
    match risk {
        "safe" => 0,
        "review_neutral_support"
        | "review_anchor_support"
        | "review_quality_score"
        | "review_tone_quality" => 1,
        "review_reference_fit" | "review_gamut" | "review_model_plausibility" => 2,
        "fallback_only" => 3,
        _ if risk.starts_with("review_") => 2,
        _ => 3,
    }
}

fn tone_color_trust_severity(state: &str) -> usize {
    match state {
        "trusted" => 0,
        "limited_weak_neutral" => 1,
        "review_required" => 2,
        _ => 3,
    }
}

pub fn direct_density_render_fallback_reason(
    ica_diagnostics: &ColorspaceDiagnostics,
    direct_diagnostics: &ColorspaceDiagnostics,
) -> Option<String> {
    let (ica_max_low, ica_total_low) = image_matrix_low_clip_summary(ica_diagnostics);
    let (direct_max_low, direct_total_low) = image_matrix_low_clip_summary(direct_diagnostics);
    let ica_neutral_delta = score_neutral_delta(ica_diagnostics.image_matrix_neutral_balance_delta);
    let direct_neutral_delta =
        score_neutral_delta(direct_diagnostics.image_matrix_neutral_balance_delta);

    if has_destructive_gamut_fallback(ica_diagnostics) {
        let relative_limit = ica_total_low * DIRECT_RENDER_LOW_CLIP_RELATIVE_LIMIT;
        let absolute_limit = (ica_total_low - DIRECT_RENDER_LOW_CLIP_ABSOLUTE_MARGIN).max(0.0);
        let direct_is_materially_safer =
            direct_total_low <= relative_limit || direct_total_low <= absolute_limit;

        if !direct_is_materially_safer {
            return None;
        }

        if direct_neutral_delta > ica_neutral_delta + NEUTRAL_REGRESSION_TOLERANCE {
            return None;
        }

        return Some(format!(
            "ICA-separated transmittance produced destructive colorspace negative-gamut clipping (max channel {:.1}%, total {:.1}%); direct density transmittance reduced the image-matrix low clipping to max channel {:.1}%, total {:.1}% without regressing image-matrix neutral delta ({:.6} -> {:.6})",
            ica_max_low * 100.0,
            ica_total_low * 100.0,
            direct_max_low * 100.0,
            direct_total_low * 100.0,
            ica_neutral_delta,
            direct_neutral_delta
        ));
    }

    let ica_quality = ica_diagnostics.selected_quality_score?;
    let direct_quality = direct_diagnostics.selected_quality_score?;
    let ica_risk_severity = candidate_risk_severity(&ica_diagnostics.candidate_risk);
    let direct_risk_severity = candidate_risk_severity(&direct_diagnostics.candidate_risk);
    if direct_risk_severity > ica_risk_severity {
        return None;
    }
    let direct_risk_improves = direct_risk_severity < ica_risk_severity;
    let ica_tone_trust = tone_color_trust_state(ica_diagnostics);
    let direct_tone_trust = tone_color_trust_state(direct_diagnostics);
    let ica_tone_trust_severity = tone_color_trust_severity(ica_tone_trust);
    let direct_tone_trust_severity = tone_color_trust_severity(direct_tone_trust);
    if direct_tone_trust_severity > ica_tone_trust_severity {
        return None;
    }
    let direct_tone_trust_improves = direct_tone_trust_severity < ica_tone_trust_severity;
    let direct_review_state_improves = direct_risk_improves || direct_tone_trust_improves;
    let direct_quality_materially_stronger =
        direct_quality + DIRECT_RENDER_QUALITY_IMPROVEMENT_MARGIN < ica_quality;
    let direct_quality_close_enough_for_trust_improvement =
        direct_quality <= ica_quality + DIRECT_RENDER_TRUST_IMPROVEMENT_QUALITY_TOLERANCE;
    if direct_total_low > ica_total_low + DIRECT_RENDER_LOW_CLIP_REGRESSION_TOLERANCE {
        return None;
    }
    let preserved_gamut_regressed = direct_diagnostics.post_scale_preserved_ratio
        + DIRECT_RENDER_PRESERVED_GAMUT_REGRESSION_TOLERANCE
        < ica_diagnostics.post_scale_preserved_ratio;
    let preserved_gamut_still_high = direct_diagnostics.post_scale_preserved_ratio
        >= DIRECT_RENDER_TRUST_IMPROVEMENT_MIN_PRESERVED_GAMUT;
    if preserved_gamut_regressed && !(direct_review_state_improves && preserved_gamut_still_high) {
        return None;
    }

    let evidence_based_selection = direct_quality_materially_stronger
        || direct_review_state_improves && direct_quality_close_enough_for_trust_improvement;
    if evidence_based_selection {
        if direct_neutral_delta > ica_neutral_delta + NEUTRAL_REGRESSION_TOLERANCE
            && !direct_risk_improves
            && !direct_tone_trust_improves
        {
            return None;
        }

        return Some(format!(
            "direct density transmittance produced a materially stronger colorspace candidate than ICA-separated transmittance (quality score {:.6} -> {:.6}) without increasing render-input risk (`{}` -> `{}`), image-matrix low clipping (total {:.1}% -> {:.1}%), or color review state (`{}` -> `{}`)",
            ica_quality,
            direct_quality,
            ica_diagnostics.candidate_risk,
            direct_diagnostics.candidate_risk,
            ica_total_low * 100.0,
            direct_total_low * 100.0,
            ica_tone_trust,
            direct_tone_trust
        ));
    }

    let physical_prior_is_safe = direct_quality
        <= ica_quality + DIRECT_RENDER_PHYSICAL_PRIOR_QUALITY_TOLERANCE
        && direct_neutral_delta <= ica_neutral_delta + NEUTRAL_REGRESSION_TOLERANCE
        && !preserved_gamut_regressed
        && direct_diagnostics.post_scale_preserved_ratio
            >= DIRECT_RENDER_TRUST_IMPROVEMENT_MIN_PRESERVED_GAMUT;
    if !physical_prior_is_safe {
        return None;
    }

    Some(format!(
        "direct density transmittance was selected as the physically grounded negative-response reconstruction: it preserves scanner-density evidence while blind ICA is underconstrained, and remained within the conservative safety envelope (quality score {:.6} -> {:.6}, risk `{}` -> `{}`, image-matrix low clipping total {:.1}% -> {:.1}%, preserved gamut {:.1}% -> {:.1}%, color review state `{}` -> `{}`)",
        ica_quality,
        direct_quality,
        ica_diagnostics.candidate_risk,
        direct_diagnostics.candidate_risk,
        ica_total_low * 100.0,
        direct_total_low * 100.0,
        ica_diagnostics.post_scale_preserved_ratio * 100.0,
        direct_diagnostics.post_scale_preserved_ratio * 100.0,
        ica_tone_trust,
        direct_tone_trust
    ))
}

fn neutral_balance_mapping_from_neutral(neutral_rgb: [f64; 3]) -> (Matrix3<f64>, [f64; 3]) {
    let neutral_mean = ((neutral_rgb[0] + neutral_rgb[1] + neutral_rgb[2]) / 3.0).max(1e-6);
    let scale: [f64; 3] =
        std::array::from_fn(|c| (neutral_mean / neutral_rgb[c].max(1e-6)).clamp(0.25, 4.0));
    (
        Matrix3::new(scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, scale[2]),
        scale,
    )
}

fn diagonal_matrix(scale: [f64; 3]) -> Matrix3<f64> {
    Matrix3::new(scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, scale[2])
}

fn selected_prophoto_matrix_from_diagnostics(diagnostics: &ColorspaceDiagnostics) -> Matrix3<f64> {
    let base = if diagnostics.mapping_strategy.starts_with("neutral_balance") {
        diagonal_matrix(diagnostics.neutral_balance_scale)
    } else {
        xyz_d50_to_prophoto_matrix()
            * bradford_cat(&diagnostics.source_white)
            * const_to_matrix3(&diagnostics.work_to_xyz)
    };
    diagonal_matrix(diagnostics.neutral_trim_scale) * base
}

pub fn build_gamut_clipping_map(
    img: &Array3<f64>,
    diagnostics: &ColorspaceDiagnostics,
) -> (Array3<u16>, GamutClippingMapDiagnostics) {
    let (h, w, _c) = img.dim();
    let total_pixels = (h * w).max(1) as f64;
    let matrix = selected_prophoto_matrix_from_diagnostics(diagnostics);
    let exposure_scale = diagnostics.exposure_scale.max(1e-6);
    let mut out = Array3::<u16>::zeros((h.max(1), w.max(1), 3));
    let mut high_clipped = [0u64; 3];
    let mut low_clipped = [0u64; 3];
    let mut any_clipped = 0u64;
    let mut max_high_excursion = [0.0f64; 3];
    let mut max_low_excursion = [0.0f64; 3];

    for y in 0..h {
        for x in 0..w {
            let mapped = (matrix * Vector3::new(img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]]))
                / exposure_scale;
            let mut high = false;
            let mut low = false;
            let mut pixel_high_signal = 0.0f64;
            let mut pixel_low_signal = 0.0f64;
            let mut luminance = 0.0;
            for c in 0..3 {
                luminance += mapped[c].clamp(0.0, 1.0);
                if mapped[c] > 1.0 + GAMUT_CLIP_EPSILON {
                    let excursion = mapped[c] - 1.0;
                    high_clipped[c] += 1;
                    max_high_excursion[c] = max_high_excursion[c].max(excursion);
                    pixel_high_signal = pixel_high_signal.max(excursion);
                    high = true;
                }
                if mapped[c] < -GAMUT_CLIP_EPSILON {
                    let excursion = -mapped[c];
                    low_clipped[c] += 1;
                    max_low_excursion[c] = max_low_excursion[c].max(excursion);
                    pixel_low_signal = pixel_low_signal.max(excursion);
                    low = true;
                }
            }

            if high || low {
                any_clipped += 1;
            }

            let safe_luma = (luminance / 3.0).clamp(0.0, 1.0);
            let rgb = if high || low {
                let r = if high {
                    (0.45 + pixel_high_signal.min(0.55)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let b = if low {
                    (0.45 + pixel_low_signal.min(0.55)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let g = if high && low { 0.18 } else { 0.0 };
                [r, g, b]
            } else {
                [0.0, 0.12 + 0.68 * safe_luma, 0.0]
            };
            for c in 0..3 {
                out[[y, x, c]] = (rgb[c] * u16::MAX as f64).round() as u16;
            }
        }
    }

    (
        out,
        GamutClippingMapDiagnostics {
            encoding: "red=post-scale high clipping, blue=post-scale low clipping, green=in-gamut luminance; yellow/magenta indicates mixed high/low channel clipping",
            high_clipped_ratio: std::array::from_fn(|c| high_clipped[c] as f64 / total_pixels),
            low_clipped_ratio: std::array::from_fn(|c| low_clipped[c] as f64 / total_pixels),
            any_clipped_ratio: any_clipped as f64 / total_pixels,
            preserved_ratio: 1.0 - any_clipped as f64 / total_pixels,
            max_high_excursion,
            max_low_excursion,
            exposure_scale,
        },
    )
}

fn neutral_band_deltas(
    neutral_estimate: &NeutralEstimate,
    combined: &Matrix3<f64>,
    exposure_scale: f64,
) -> [Option<[f64; 3]>; 3] {
    std::array::from_fn(|band| {
        (neutral_estimate.bands[band] > 0).then(|| {
            neutral_balance_delta(
                neutral_estimate.band_rgb[band],
                combined,
                exposure_scale.max(1e-6),
            )
        })
    })
}

fn neutral_band_delta_magnitudes(
    neutral_band_balance_delta: [Option<[f64; 3]>; 3],
) -> [Option<f64>; 3] {
    std::array::from_fn(|band| neutral_band_balance_delta[band].map(score_neutral_delta))
}

fn neutral_trim_stats(
    stats: &MappingStats,
    neutral_estimate: &NeutralEstimate,
    combined: &Matrix3<f64>,
) -> NeutralTrimStats {
    let neutral_band_balance_delta =
        neutral_band_deltas(neutral_estimate, combined, stats.exposure_scale);
    NeutralTrimStats {
        neutral_balance_delta: stats.neutral_balance_delta,
        neutral_delta_magnitude: score_neutral_delta(stats.neutral_balance_delta),
        neutral_band_names: NEUTRAL_BAND_NAMES,
        neutral_band_balance_delta,
        neutral_band_delta_magnitude: neutral_band_delta_magnitudes(neutral_band_balance_delta),
        pre_scale_clipped_low_ratio: stats.pre_scale_clipped_low_ratio,
        pre_scale_clipped_high_ratio: stats.pre_scale_clipped_high_ratio,
        pre_scale_preserved_ratio: stats.pre_scale_preserved_ratio,
        exposure_scale: stats.exposure_scale,
    }
}

fn any_clip_ratio_increased(before: [f64; 3], after: [f64; 3]) -> bool {
    (0..3).any(|c| after[c] > before[c] + CLIPPING_INCREASE_EPSILON)
}

fn any_neutral_band_delta_worsened(before: [Option<f64>; 3], after: [Option<f64>; 3]) -> bool {
    (0..3).any(|band| match (before[band], after[band]) {
        (Some(before), Some(after)) => after > before + NEUTRAL_TRIM_BAND_WORSEN_TOLERANCE,
        _ => false,
    })
}

fn evaluate_neutral_trim(
    img: &Array3<f64>,
    neutral_estimate: &NeutralEstimate,
    combined: &Matrix3<f64>,
    before_stats: &MappingStats,
) -> (Matrix3<f64>, NeutralTrimDiagnostics) {
    let before = neutral_trim_stats(before_stats, neutral_estimate, combined);
    if !neutral_estimate.quality.accepted {
        let mut diagnostics = NeutralTrimDiagnostics::skipped(format!(
            "neutral trim skipped: {}",
            neutral_estimate.quality.reason
        ));
        diagnostics.before = Some(before);
        return (*combined, diagnostics);
    }

    let mapped = (combined
        * Vector3::new(
            neutral_estimate.rgb[0],
            neutral_estimate.rgb[1],
            neutral_estimate.rgb[2],
        ))
        / before_stats.exposure_scale.max(1e-6);
    if mapped
        .iter()
        .any(|value| !value.is_finite() || *value <= 1e-6)
    {
        let mut diagnostics =
            NeutralTrimDiagnostics::skipped("neutral trim skipped: mapped neutral was invalid");
        diagnostics.before = Some(before);
        return (*combined, diagnostics);
    }

    let mean = ((mapped[0] + mapped[1] + mapped[2]) / 3.0).max(1e-6);
    let scale: [f64; 3] = std::array::from_fn(|c| {
        (mean / mapped[c].max(1e-6)).clamp(NEUTRAL_TRIM_MIN, NEUTRAL_TRIM_MAX)
    });
    if !scale.iter().any(|value| (*value - 1.0).abs() > 0.005) {
        let mut diagnostics = NeutralTrimDiagnostics::skipped(
            "neutral trim skipped: measured neutral delta is already within trim tolerance",
        );
        diagnostics.before = Some(before);
        diagnostics.scale = [1.0; 3];
        return (*combined, diagnostics);
    }

    let trim = Matrix3::new(scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, scale[2]);
    let trimmed = trim * combined;
    let after_stats = evaluate_mapping_stats(img, &trimmed, neutral_estimate.rgb);
    let after = neutral_trim_stats(&after_stats, neutral_estimate, &trimmed);
    let neutral_delta_reduced = after.neutral_delta_magnitude + MIN_NEUTRAL_TRIM_DELTA_IMPROVEMENT
        < before.neutral_delta_magnitude;
    let neutral_band_delta_worsened = any_neutral_band_delta_worsened(
        before.neutral_band_delta_magnitude,
        after.neutral_band_delta_magnitude,
    );
    let low_clipping_increased = any_clip_ratio_increased(
        before.pre_scale_clipped_low_ratio,
        after.pre_scale_clipped_low_ratio,
    );
    let high_clipping_increased = any_clip_ratio_increased(
        before.pre_scale_clipped_high_ratio,
        after.pre_scale_clipped_high_ratio,
    );
    let preserved_ratio_decreased = after.pre_scale_preserved_ratio + CLIPPING_INCREASE_EPSILON
        < before.pre_scale_preserved_ratio;
    let applied = neutral_delta_reduced
        && !neutral_band_delta_worsened
        && !low_clipping_increased
        && !high_clipping_increased
        && !preserved_ratio_decreased;
    let reason = if applied {
        "neutral trim applied: broad neutral support reduced measured neutral delta without increasing clipping".to_string()
    } else if !neutral_delta_reduced {
        "neutral trim skipped: proposed trim did not materially reduce measured neutral delta"
            .to_string()
    } else if neutral_band_delta_worsened {
        "neutral trim skipped: proposed trim worsened at least one populated neutral luminance band"
            .to_string()
    } else if low_clipping_increased || high_clipping_increased || preserved_ratio_decreased {
        "neutral trim skipped: proposed trim increased clipping or reduced preserved gamut"
            .to_string()
    } else {
        "neutral trim skipped".to_string()
    };
    let diagnostics = NeutralTrimDiagnostics {
        applied,
        scale: if applied { scale } else { [1.0; 3] },
        reason,
        before: Some(before),
        after: Some(after),
        neutral_delta_reduced,
        neutral_band_delta_worsened,
        low_clipping_increased,
        high_clipping_increased,
        preserved_ratio_decreased,
    };

    if applied {
        (trimmed, diagnostics)
    } else {
        (*combined, diagnostics)
    }
}

#[derive(Debug, Clone)]
struct CandidateSelection {
    selected_index: usize,
    selection_rejections: Vec<String>,
    calibration_acceptance: CalibrationAcceptanceDiagnostics,
    neutral_safety_rescue: NeutralSafetyRescueDiagnostics,
}

fn neutral_safety_rescue_diagnostics(
    matrix_candidate: Option<&ColorMappingCandidate>,
    neutral_candidate: &ColorMappingCandidate,
    color_mode: ColorMode,
) -> NeutralSafetyRescueDiagnostics {
    let neutral_model = neutral_candidate.score.color_model_quality.as_ref();
    let neutral_tone = neutral_candidate.score.rendered_tone_quality.as_ref();
    let neutral_memory_supported = neutral_model.is_some_and(|model| {
        [
            &model.memory_color.skin,
            &model.memory_color.foliage,
            &model.memory_color.sky,
        ]
        .iter()
        .any(|family| family.sample_count >= MEMORY_COLOR_MIN_FAMILY_SAMPLES)
    });
    let neutral_model_evidence_supported = neutral_model.is_some_and(|model| {
        model.saturation_preservation_sample_count >= SATURATION_PRESERVATION_MIN_SAMPLES
            && model.spatial_consistency.populated_tile_count >= SPATIAL_CONSISTENCY_MIN_TILES
    }) && neutral_memory_supported;
    let matrix_tone = matrix_candidate.and_then(|candidate| {
        candidate
            .score
            .rendered_tone_quality
            .as_ref()
            .map(|tone| tone.midtone_saturation_p95)
    });
    let neutral_tone_p95 = neutral_tone.map(|tone| tone.midtone_saturation_p95);
    let matrix_preserved =
        matrix_candidate.map(|candidate| candidate.stats.pre_scale_preserved_ratio);
    let preserved_gain =
        matrix_preserved.map(|matrix| neutral_candidate.stats.pre_scale_preserved_ratio - matrix);
    let saturation_reduction = matrix_tone
        .zip(neutral_tone_p95)
        .map(|(matrix, neutral)| matrix - neutral);
    let matrix_anchor_supported = matrix_candidate.map(|candidate| {
        candidate.diagnostics.dominant_anchor_quality.accepted
            && !candidate
                .score
                .dominant_anchor_unstable_channels
                .iter()
                .any(|unstable| *unstable)
    });
    let matrix_memory_penalty =
        matrix_candidate.map(|candidate| candidate.score.quality_components.memory_color_penalty);
    let neutral_memory_penalty = neutral_candidate
        .score
        .quality_components
        .memory_color_penalty;
    let matrix_spatial_penalty = matrix_candidate.map(|candidate| {
        candidate
            .score
            .quality_components
            .spatial_consistency_penalty
    });
    let neutral_spatial_penalty = neutral_candidate
        .score
        .quality_components
        .spatial_consistency_penalty;
    let mut diagnostics = NeutralSafetyRescueDiagnostics {
        evaluated: false,
        applied: false,
        matrix_candidate: matrix_candidate.map(|candidate| candidate.candidate.to_string()),
        matrix_candidate_kind: matrix_candidate
            .map(|candidate| candidate.kind.as_str().to_string()),
        matrix_anchor_evidence_supported: matrix_anchor_supported,
        neutral_estimate_supported: neutral_candidate
            .diagnostics
            .neutral_estimate_quality
            .accepted,
        neutral_model_evidence_supported,
        matrix_pre_scale_preserved_ratio: matrix_preserved,
        neutral_pre_scale_preserved_ratio: neutral_candidate.stats.pre_scale_preserved_ratio,
        preserved_ratio_gain: preserved_gain,
        minimum_preserved_ratio: AUTO_NEUTRAL_RESCUE_MIN_PRESERVED_RATIO,
        minimum_preserved_ratio_gain: AUTO_NEUTRAL_RESCUE_MIN_PRESERVED_GAIN,
        matrix_midtone_saturation_p95: matrix_tone,
        neutral_midtone_saturation_p95: neutral_tone_p95,
        midtone_saturation_p95_reduction: saturation_reduction,
        maximum_midtone_saturation_p95: RENDER_TONE_MIDTONE_SATURATION_TARGET,
        minimum_midtone_saturation_p95_reduction:
            AUTO_NEUTRAL_RESCUE_MIN_MIDTONE_SATURATION_REDUCTION,
        matrix_memory_color_penalty: matrix_memory_penalty,
        neutral_memory_color_penalty: neutral_memory_penalty,
        matrix_spatial_consistency_penalty: matrix_spatial_penalty,
        neutral_spatial_consistency_penalty: neutral_spatial_penalty,
        neutral_saturation_preservation_sample_count: neutral_model
            .map(|model| model.saturation_preservation_sample_count)
            .unwrap_or(0),
        neutral_saturation_preservation_p05_ratio: neutral_model
            .and_then(|model| model.saturation_preservation_p05_ratio),
        neutral_saturation_preservation_median_ratio: neutral_model
            .and_then(|model| model.saturation_preservation_median_ratio),
        neutral_saturation_preservation_p95_ratio: neutral_model
            .and_then(|model| model.saturation_preservation_p95_ratio),
        reason: String::new(),
    };

    if color_mode != ColorMode::Auto {
        diagnostics.reason = format!(
            "neutral safety rescue is auto-only; --color-mode {} preserves the requested candidate class",
            color_mode.as_str()
        );
        return diagnostics;
    }
    let Some(matrix_candidate) = matrix_candidate else {
        diagnostics.reason =
            "neutral safety rescue was not needed because no matrix candidate survived the ordinary safety gate"
                .to_string();
        return diagnostics;
    };
    diagnostics.evaluated = true;

    let mut blockers = Vec::<String>::new();
    if matrix_candidate.kind != CandidateKind::ImageDerived {
        blockers.push(format!(
            "selected matrix kind `{}` is protected from an uncalibrated neutral rescue",
            matrix_candidate.kind.as_str()
        ));
    }
    if matrix_anchor_supported == Some(true) {
        blockers.push("matrix dominant-anchor evidence is supported and stable".to_string());
    }
    if !diagnostics.neutral_estimate_supported {
        blockers.push("neutral estimate is not supported".to_string());
    }
    if !neutral_model_evidence_supported {
        blockers.push(
            "neutral candidate lacks supported memory-colour, spatial-neutral, or chroma-retention samples"
                .to_string(),
        );
    }
    if neutral_candidate.stats.pre_scale_preserved_ratio < AUTO_NEUTRAL_RESCUE_MIN_PRESERVED_RATIO {
        blockers.push(format!(
            "neutral preserved-gamut ratio {:.6} is below {:.6}",
            neutral_candidate.stats.pre_scale_preserved_ratio,
            AUTO_NEUTRAL_RESCUE_MIN_PRESERVED_RATIO
        ));
    }
    if preserved_gain.is_none_or(|gain| gain < AUTO_NEUTRAL_RESCUE_MIN_PRESERVED_GAIN) {
        blockers.push(format!(
            "preserved-gamut gain {:.6} is below {:.6}",
            preserved_gain.unwrap_or(0.0),
            AUTO_NEUTRAL_RESCUE_MIN_PRESERVED_GAIN
        ));
    }
    if matrix_tone.is_none_or(|value| value <= RENDER_TONE_MIDTONE_SATURATION_TARGET) {
        blockers.push(format!(
            "matrix midtone saturation p95 {:.6} does not exceed {:.6}",
            matrix_tone.unwrap_or(0.0),
            RENDER_TONE_MIDTONE_SATURATION_TARGET
        ));
    }
    if neutral_tone_p95.is_none_or(|value| value > RENDER_TONE_MIDTONE_SATURATION_TARGET) {
        blockers.push(format!(
            "neutral midtone saturation p95 {:.6} exceeds {:.6}",
            neutral_tone_p95.unwrap_or(f64::INFINITY),
            RENDER_TONE_MIDTONE_SATURATION_TARGET
        ));
    }
    if saturation_reduction
        .is_none_or(|reduction| reduction < AUTO_NEUTRAL_RESCUE_MIN_MIDTONE_SATURATION_REDUCTION)
    {
        blockers.push(format!(
            "midtone saturation p95 reduction {:.6} is below {:.6}",
            saturation_reduction.unwrap_or(0.0),
            AUTO_NEUTRAL_RESCUE_MIN_MIDTONE_SATURATION_REDUCTION
        ));
    }

    let neutral_chroma_retained = neutral_model.is_some_and(|model| {
        model.saturation_preservation_sample_count >= SATURATION_PRESERVATION_MIN_SAMPLES
            && model
                .saturation_preservation_p05_ratio
                .is_some_and(|ratio| ratio >= SATURATION_PRESERVATION_LOW_P05_TARGET)
            && model
                .saturation_preservation_median_ratio
                .is_some_and(|ratio| ratio >= SATURATION_PRESERVATION_LOW_TARGET)
            && model
                .saturation_preservation_p95_ratio
                .is_some_and(|ratio| ratio <= SATURATION_PRESERVATION_HIGH_TARGET)
    });
    if !neutral_chroma_retained {
        blockers.push("neutral candidate did not retain supported scene chroma".to_string());
    }

    let memory_no_worse =
        matrix_memory_penalty.is_some_and(|matrix| neutral_memory_penalty <= matrix + 1e-9);
    let spatial_no_worse =
        matrix_spatial_penalty.is_some_and(|matrix| neutral_spatial_penalty <= matrix + 1e-9);
    let plausibility_improved = matrix_memory_penalty
        .is_some_and(|matrix| neutral_memory_penalty + 1e-9 < matrix)
        || matrix_spatial_penalty.is_some_and(|matrix| neutral_spatial_penalty + 1e-9 < matrix);
    if !memory_no_worse {
        blockers.push("neutral memory-colour plausibility regressed".to_string());
    }
    if !spatial_no_worse {
        blockers.push("neutral spatial-neutral consistency regressed".to_string());
    }
    if !plausibility_improved {
        blockers.push("neutral perceptual plausibility did not measurably improve".to_string());
    }

    diagnostics.applied = blockers.is_empty();
    diagnostics.reason = if diagnostics.applied {
        format!(
            "auto neutral safety rescue replaced unsupported image-derived candidate `{}`: preserved gamut {:.6} -> {:.6}, midtone saturation p95 {:.6} -> {:.6}, memory-colour penalty {:.6} -> {:.6}, spatial-neutral penalty {:.6} -> {:.6}; the neutral result remains fallback-only and review-required",
            matrix_candidate.candidate,
            matrix_candidate.stats.pre_scale_preserved_ratio,
            neutral_candidate.stats.pre_scale_preserved_ratio,
            matrix_tone.unwrap_or(0.0),
            neutral_tone_p95.unwrap_or(0.0),
            matrix_memory_penalty.unwrap_or(0.0),
            neutral_memory_penalty,
            matrix_spatial_penalty.unwrap_or(0.0),
            neutral_spatial_penalty,
        )
    } else {
        format!("neutral safety rescue not applied: {}", blockers.join("; "))
    };
    diagnostics
}

fn calibration_candidate_kind(kind: CandidateKind) -> bool {
    matches!(
        kind,
        CandidateKind::CalibratedDirect | CandidateKind::ScannerPrior
    )
}

fn color_mode_allows_candidate(kind: CandidateKind, color_mode: ColorMode) -> bool {
    match color_mode {
        ColorMode::Auto => true,
        ColorMode::Calibrated => calibration_candidate_kind(kind),
        ColorMode::ImageDerived => kind == CandidateKind::ImageDerived,
        ColorMode::Neutral => kind == CandidateKind::NeutralFallback,
    }
}

fn ranked_candidate_indices(candidates: &[ColorMappingCandidate]) -> Vec<usize> {
    let mut ranked = (0..candidates.len()).collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        let left = &candidates[*a];
        let right = &candidates[*b];
        left.score
            .quality_score
            .partial_cmp(&right.score.quality_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.kind
                    .preference_rank()
                    .cmp(&right.kind.preference_rank())
            })
            .then_with(|| left.candidate.cmp(right.candidate))
    });
    ranked
}

fn candidate_ranks(candidates: &[ColorMappingCandidate]) -> Vec<Option<usize>> {
    let mut ranks = vec![None; candidates.len()];
    for (rank, idx) in ranked_candidate_indices(candidates).into_iter().enumerate() {
        ranks[idx] = Some(rank + 1);
    }
    ranks
}

fn selected_runner_up_quality_delta(
    candidates: &[ColorMappingCandidate],
    selected_index: usize,
    color_mode: ColorMode,
) -> Option<f64> {
    let selected_kind = candidates[selected_index].kind;
    let image_candidate = candidates
        .iter()
        .find(|candidate| candidate.kind == CandidateKind::ImageDerived);
    let image_quality = image_candidate
        .map(|candidate| candidate.score.quality_score)
        .unwrap_or(f64::INFINITY);
    let image_rejected = image_candidate.is_some_and(|candidate| candidate.score.rejected);
    let mut eligible = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            if candidate.score.rejected {
                return false;
            }
            match color_mode {
                ColorMode::Auto => {
                    if selected_kind == CandidateKind::NeutralFallback {
                        candidate.kind == CandidateKind::NeutralFallback
                    } else {
                        candidate.kind.is_matrix()
                            && image_candidate.is_none_or(|image_candidate| {
                                auto_calibration_rejection(
                                    candidate,
                                    image_candidate,
                                    image_quality,
                                    image_rejected,
                                )
                                .is_none()
                            })
                    }
                }
                _ => color_mode_allows_candidate(candidate.kind, color_mode),
            }
        })
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();
    eligible.sort_by(|a, b| {
        candidates[*a]
            .score
            .quality_score
            .partial_cmp(&candidates[*b].score.quality_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                candidates[*a]
                    .kind
                    .preference_rank()
                    .cmp(&candidates[*b].kind.preference_rank())
            })
            .then_with(|| candidates[*a].candidate.cmp(candidates[*b].candidate))
    });

    eligible
        .into_iter()
        .find(|idx| *idx != selected_index)
        .map(|runner_up| {
            candidates[runner_up].score.quality_score
                - candidates[selected_index].score.quality_score
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoCalibrationRejectionKind {
    ReferenceFit,
    NeutralMetrics,
    Quality,
}

fn neutral_metric_regresses(
    candidate: &ColorMappingCandidate,
    image_candidate: &ColorMappingCandidate,
) -> bool {
    score_neutral_delta(candidate.stats.neutral_balance_delta)
        > score_neutral_delta(image_candidate.stats.neutral_balance_delta)
            + NEUTRAL_REGRESSION_TOLERANCE
}

fn auto_calibration_rejection(
    candidate: &ColorMappingCandidate,
    image_candidate: &ColorMappingCandidate,
    image_quality: f64,
    image_rejected: bool,
) -> Option<(AutoCalibrationRejectionKind, String)> {
    if !calibration_candidate_kind(candidate.kind) || candidate.score.rejected {
        return None;
    }
    if candidate
        .score
        .reference_patch_regresses_image_derived
        .unwrap_or(false)
    {
        return Some((
            AutoCalibrationRejectionKind::ReferenceFit,
            format!(
                "{} reference patch fit regressed versus image-derived mapping (XYZ rms delta {:.6}, XYZ max delta {:.6}, DeltaE rms delta {:.6}, DeltaE max delta {:.6}, CIEDE2000 rms delta {:.6}, CIEDE2000 max delta {:.6}); review_reference_fit",
                candidate.candidate,
                candidate
                    .score
                    .reference_patch_delta_vs_image_derived
                    .unwrap_or(0.0),
                candidate
                    .score
                    .reference_patch_max_delta_vs_image_derived
                    .unwrap_or(0.0),
                candidate
                    .score
                    .reference_patch_delta_e_delta_vs_image_derived
                    .unwrap_or(0.0),
                candidate
                    .score
                    .reference_patch_delta_e_max_delta_vs_image_derived
                    .unwrap_or(0.0),
                candidate
                    .score
                    .reference_patch_delta_e2000_delta_vs_image_derived
                    .unwrap_or(0.0),
                candidate
                    .score
                    .reference_patch_delta_e2000_max_delta_vs_image_derived
                    .unwrap_or(0.0)
            ),
        ));
    }
    if neutral_metric_regresses(candidate, image_candidate) {
        return Some((
            AutoCalibrationRejectionKind::NeutralMetrics,
            format!(
                "{} neutral balance delta {:.6} regressed versus image-derived delta {:.6}; review_neutral_support",
                candidate.candidate,
                score_neutral_delta(candidate.stats.neutral_balance_delta),
                score_neutral_delta(image_candidate.stats.neutral_balance_delta)
            ),
        ));
    }
    if !image_rejected && candidate.score.quality_score >= image_quality {
        return Some((
            AutoCalibrationRejectionKind::Quality,
            format!(
                "{} quality score {:.6} did not beat image-derived score {:.6}",
                candidate.candidate, candidate.score.quality_score, image_quality
            ),
        ));
    }
    None
}

fn select_color_candidate(
    candidates: &[ColorMappingCandidate],
    color_mode: ColorMode,
) -> Result<CandidateSelection, String> {
    let neutral_index = candidates
        .iter()
        .position(|candidate| candidate.kind == CandidateKind::NeutralFallback)
        .expect("neutral fallback candidate");
    let image_index = candidates
        .iter()
        .position(|candidate| candidate.kind == CandidateKind::ImageDerived)
        .expect("image-derived candidate");
    let image_quality = candidates[image_index].score.quality_score;
    let image_rejected = candidates[image_index].score.rejected;
    let image_candidate = &candidates[image_index];
    let calibration_candidates = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| calibration_candidate_kind(candidate.kind))
        .collect::<Vec<_>>();
    let preferred_calibration = calibration_candidates
        .iter()
        .min_by(|(_, left), (_, right)| {
            left.score
                .quality_score
                .partial_cmp(&right.score.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left.kind
                        .preference_rank()
                        .cmp(&right.kind.preference_rank())
                })
                .then_with(|| left.candidate.cmp(right.candidate))
        })
        .map(|(idx, candidate)| (*idx, *candidate));

    let mut eligible = match color_mode {
        ColorMode::Auto => {
            let matrix = candidates
                .iter()
                .enumerate()
                .filter_map(|(idx, candidate)| {
                    if !candidate.kind.is_matrix() || candidate.score.rejected {
                        return None;
                    }
                    if auto_calibration_rejection(
                        candidate,
                        image_candidate,
                        image_quality,
                        image_rejected,
                    )
                    .is_some()
                    {
                        return None;
                    }
                    Some(idx)
                })
                .collect::<Vec<_>>();
            if matrix.is_empty() {
                let calibration_acceptance = calibration_acceptance_for_selection(
                    candidates,
                    color_mode,
                    neutral_index,
                    preferred_calibration,
                    image_quality,
                    image_candidate,
                    image_rejected,
                );
                return Ok(CandidateSelection {
                    selected_index: neutral_index,
                    selection_rejections: selection_rejections_for_candidates(
                        candidates,
                        preferred_calibration,
                        image_quality,
                        image_candidate,
                        image_rejected,
                    ),
                    calibration_acceptance,
                    neutral_safety_rescue: neutral_safety_rescue_diagnostics(
                        None,
                        &candidates[neutral_index],
                        color_mode,
                    ),
                });
            }
            matrix
        }
        ColorMode::Calibrated => candidates
            .iter()
            .enumerate()
            .filter_map(|(idx, candidate)| {
                calibration_candidate_kind(candidate.kind).then_some(idx)
            })
            .collect(),
        ColorMode::ImageDerived => candidates
            .iter()
            .enumerate()
            .filter_map(|(idx, candidate)| {
                (candidate.kind == CandidateKind::ImageDerived).then_some(idx)
            })
            .collect(),
        ColorMode::Neutral => vec![neutral_index],
    };

    if eligible.is_empty() {
        return Err(format!(
            "--color-mode {} did not produce an eligible colorspace candidate",
            color_mode.as_str()
        ));
    }

    let rejected_reasons = eligible
        .iter()
        .filter_map(|idx| {
            let candidate = &candidates[*idx];
            candidate
                .score
                .rejection_reason
                .as_ref()
                .map(|reason| format!("{}: {}", candidate.candidate, reason))
        })
        .collect::<Vec<_>>();
    eligible.retain(|idx| !candidates[*idx].score.rejected);
    if eligible.is_empty() {
        return Err(format!(
            "--color-mode {} selected only unsafe colorspace candidate(s): {}",
            color_mode.as_str(),
            rejected_reasons.join("; ")
        ));
    }

    eligible.sort_by(|a, b| {
        let left = &candidates[*a];
        let right = &candidates[*b];
        left.score
            .score
            .partial_cmp(&right.score.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.kind
                    .preference_rank()
                    .cmp(&right.kind.preference_rank())
            })
            .then_with(|| left.candidate.cmp(right.candidate))
    });

    let matrix_selected_index = eligible[0];
    let matrix_candidate = candidates[matrix_selected_index]
        .kind
        .is_matrix()
        .then_some(&candidates[matrix_selected_index]);
    let neutral_safety_rescue =
        neutral_safety_rescue_diagnostics(matrix_candidate, &candidates[neutral_index], color_mode);
    let selected_index = if neutral_safety_rescue.applied {
        neutral_index
    } else {
        matrix_selected_index
    };
    let calibration_acceptance = calibration_acceptance_for_selection(
        candidates,
        color_mode,
        selected_index,
        preferred_calibration,
        image_quality,
        image_candidate,
        image_rejected,
    );
    let mut selection_rejections = selection_rejections_for_candidates(
        candidates,
        preferred_calibration,
        image_quality,
        image_candidate,
        image_rejected,
    );
    if neutral_safety_rescue.applied {
        selection_rejections.push(format!(
            "{} superseded by neutral safety rescue: {}",
            candidates[matrix_selected_index].candidate, neutral_safety_rescue.reason
        ));
    }
    Ok(CandidateSelection {
        selected_index,
        selection_rejections,
        calibration_acceptance,
        neutral_safety_rescue,
    })
}

fn selection_rejections_for_candidates(
    candidates: &[ColorMappingCandidate],
    preferred_calibration: Option<(usize, &ColorMappingCandidate)>,
    image_quality: f64,
    image_candidate: &ColorMappingCandidate,
    image_rejected: bool,
) -> Vec<String> {
    let mut rejections = candidates
        .iter()
        .filter_map(|candidate| {
            candidate
                .score
                .rejection_reason
                .as_ref()
                .map(|reason| format!("{} rejected: {}", candidate.candidate, reason))
        })
        .collect::<Vec<_>>();

    for candidate in candidates
        .iter()
        .filter(|candidate| calibration_candidate_kind(candidate.kind))
    {
        if let Some((_, reason)) =
            auto_calibration_rejection(candidate, image_candidate, image_quality, image_rejected)
        {
            rejections.push(format!("{} rejected: {}", candidate.candidate, reason));
        }
    }

    if preferred_calibration.is_none() {
        return rejections;
    }

    rejections
}

fn calibration_acceptance_for_selection(
    candidates: &[ColorMappingCandidate],
    color_mode: ColorMode,
    selected_index: usize,
    preferred_calibration: Option<(usize, &ColorMappingCandidate)>,
    image_quality: f64,
    image_candidate: &ColorMappingCandidate,
    image_rejected: bool,
) -> CalibrationAcceptanceDiagnostics {
    let Some((preferred_index, preferred)) = preferred_calibration else {
        return CalibrationAcceptanceDiagnostics::not_applicable(
            color_mode,
            "no calibration or scanner-prior candidate was constructed",
        );
    };
    let within_negative_gamut_limits = !preferred.score.rejected;
    let beats_image = preferred.score.quality_score < image_quality;
    let auto_rejection =
        auto_calibration_rejection(preferred, image_candidate, image_quality, image_rejected);
    let forced_by_color_mode = color_mode != ColorMode::Auto;
    let selected_calibration = selected_index == preferred_index
        || calibration_candidate_kind(candidates[selected_index].kind);
    let (status, reason) = if forced_by_color_mode && selected_calibration {
        (
            "forced",
            format!(
                "--color-mode {} selected {} regardless of automatic image-derived comparison",
                color_mode.as_str(),
                candidates[selected_index].source_label
            ),
        )
    } else if selected_calibration && beats_image && auto_rejection.is_none() {
        (
            "accepted",
            format!(
                "{} quality score {:.6} beat image-derived score {:.6} and stayed within negative-gamut safety limits",
                preferred.candidate, preferred.score.quality_score, image_quality
            ),
        )
    } else if selected_calibration && image_rejected && auto_rejection.is_none() {
        (
            "accepted",
            format!(
                "{} stayed within negative-gamut safety limits while image-derived mapping was rejected for safety",
                preferred.candidate
            ),
        )
    } else if !within_negative_gamut_limits {
        (
            "rejected_unsafe",
            preferred.score.rejection_reason.clone().unwrap_or_else(|| {
                "calibration candidate exceeded negative-gamut safety limits".to_string()
            }),
        )
    } else if let Some((AutoCalibrationRejectionKind::ReferenceFit, reason)) = &auto_rejection {
        ("rejected_reference_fit", reason.clone())
    } else if let Some((AutoCalibrationRejectionKind::NeutralMetrics, reason)) = &auto_rejection {
        ("rejected_neutral", reason.clone())
    } else if !beats_image {
        (
            "rejected_quality",
            auto_rejection
                .as_ref()
                .map(|(_, reason)| reason.clone())
                .unwrap_or_else(|| {
                    format!(
                        "{} quality score {:.6} did not beat image-derived score {:.6}",
                        preferred.candidate, preferred.score.quality_score, image_quality
                    )
                }),
        )
    } else {
        (
            "not_selected",
            "calibration candidate was not selected by the requested color mode".to_string(),
        )
    };

    CalibrationAcceptanceDiagnostics {
        status: status.to_string(),
        reason,
        color_mode: color_mode.as_str(),
        preferred_candidate: Some(preferred.candidate.to_string()),
        preferred_candidate_quality_score: Some(preferred.score.quality_score),
        image_derived_quality_score: Some(image_quality),
        beats_image_derived: Some(beats_image),
        within_negative_gamut_limits: Some(within_negative_gamut_limits),
        forced_by_color_mode,
    }
}

fn candidate_acceptance_for_selection(
    candidates: &[ColorMappingCandidate],
    color_mode: ColorMode,
    selected_index: usize,
    ranks: &[Option<usize>],
    neutral_safety_rescue: &NeutralSafetyRescueDiagnostics,
) -> Vec<ColorMappingCandidateAcceptanceDiagnostics> {
    let image_quality = candidates
        .iter()
        .find(|candidate| candidate.kind == CandidateKind::ImageDerived)
        .map(|candidate| candidate.score.quality_score)
        .unwrap_or(f64::INFINITY);
    let image_rejected = candidates
        .iter()
        .find(|candidate| candidate.kind == CandidateKind::ImageDerived)
        .is_some_and(|candidate| candidate.score.rejected);
    let image_candidate = candidates
        .iter()
        .find(|candidate| candidate.kind == CandidateKind::ImageDerived);
    let selected_quality = candidates[selected_index].score.quality_score;

    candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| {
            let selected = idx == selected_index;
            let eligible_in_color_mode = color_mode_allows_candidate(candidate.kind, color_mode);
            let selected_quality_delta = Some(candidate.score.quality_score - selected_quality);
            let beats_image_derived = calibration_candidate_kind(candidate.kind)
                .then_some(candidate.score.quality_score < image_quality);
            let within_negative_gamut_limits =
                candidate.kind.is_matrix().then_some(!candidate.score.rejected);
            let (status, reason) = if selected && neutral_safety_rescue.applied {
                (
                    "selected_evidence_rescue",
                    neutral_safety_rescue.reason.clone(),
                )
            } else if selected {
                (
                    "selected",
                    format!(
                        "{} selected by --color-mode {} with {} score {:.6}",
                        candidate.source_label,
                        color_mode.as_str(),
                        COLOR_SCORE_ORDER,
                        candidate.score.quality_score
                    ),
                )
            } else if !eligible_in_color_mode {
                (
                    "rejected_mode",
                    format!(
                        "{} candidate was outside --color-mode {}",
                        candidate.source_label,
                        color_mode.as_str()
                    ),
                )
            } else if let Some(reason) = &candidate.score.rejection_reason {
                ("rejected_safety", reason.clone())
            } else if color_mode == ColorMode::Auto
                && image_candidate
                    .and_then(|image_candidate| {
                        auto_calibration_rejection(
                            candidate,
                            image_candidate,
                            image_quality,
                            image_rejected,
                        )
                    })
                    .is_some()
            {
                let (kind, reason) = auto_calibration_rejection(
                    candidate,
                    image_candidate.expect("checked above"),
                    image_quality,
                    image_rejected,
                )
                .expect("checked above");
                let status = match kind {
                    AutoCalibrationRejectionKind::ReferenceFit => "rejected_reference_fit",
                    AutoCalibrationRejectionKind::NeutralMetrics => "rejected_neutral",
                    AutoCalibrationRejectionKind::Quality => "rejected_quality",
                };
                (status, reason)
            } else if neutral_safety_rescue.applied
                && neutral_safety_rescue.matrix_candidate.as_deref()
                    == Some(candidate.candidate)
            {
                (
                    "rejected_evidence_rescue",
                    neutral_safety_rescue.reason.clone(),
                )
            } else if candidate.kind == CandidateKind::NeutralFallback
                && color_mode == ColorMode::Auto
            {
                (
                    "available_fallback",
                    "neutral-balance fallback remained available but a safe matrix candidate was selected"
                        .to_string(),
                )
            } else {
                (
                    "accepted_runner_up",
                    format!(
                        "{} passed mode and safety checks but ranked behind selected {}",
                        candidate.source_label, candidates[selected_index].source_label
                    ),
                )
            };

            ColorMappingCandidateAcceptanceDiagnostics {
                candidate: candidate.candidate.to_string(),
                candidate_kind: candidate.kind.as_str().to_string(),
                mapping_strategy: candidate.mapping_strategy.to_string(),
                source_label: candidate.source_label.to_string(),
                status: status.to_string(),
                reason,
                rank: ranks.get(idx).copied().flatten(),
                selected,
                eligible_in_color_mode,
                quality_score: candidate.score.quality_score,
                selected_quality_delta,
                rejected: candidate.score.rejected,
                rejection_reason: candidate.score.rejection_reason.clone(),
                beats_image_derived,
                within_negative_gamut_limits,
            }
        })
        .collect()
}

fn legacy_image_candidate(
    candidates: &[ColorMappingCandidate],
    color_mode: ColorMode,
) -> &ColorMappingCandidate {
    if color_mode != ColorMode::ImageDerived {
        if let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.kind == CandidateKind::ScannerPrior)
        {
            return candidate;
        }
    }

    candidates
        .iter()
        .find(|candidate| candidate.kind == CandidateKind::ImageDerived)
        .expect("image-derived candidate")
}

fn candidate_risk_for_diagnostics(diagnostics: &ColorspaceDiagnostics) -> String {
    if diagnostics.mapping_strategy == "positive_rgb_passthrough" {
        return "review_unprofiled_input".to_string();
    }
    if diagnostics.mapping_strategy == "embedded_icc_to_linear_prophoto" {
        return "safe".to_string();
    }
    if diagnostics
        .reference_patch_evaluation
        .as_ref()
        .is_some_and(|evaluation| evaluation.selected_regresses_image_derived)
        || diagnostics.calibration_acceptance.status == "rejected_reference_fit"
    {
        return "review_reference_fit".to_string();
    }
    if diagnostics.calibration_acceptance.status == "rejected_neutral" {
        return "review_neutral_support".to_string();
    }
    if diagnostics.selected_candidate == "neutral_balance_fallback"
        || diagnostics.mapping_strategy == "neutral_balance_forced"
        || diagnostics.mapping_strategy == "neutral_balance_gamut_fallback"
    {
        return "fallback_only".to_string();
    }
    let low_max = diagnostics
        .pre_scale_clipped_low_ratio
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    let low_total = diagnostics.pre_scale_clipped_low_ratio.iter().sum::<f64>();
    let post_low_total = diagnostics.post_scale_clipped_low_ratio.iter().sum::<f64>();
    let post_high_total = diagnostics
        .post_scale_clipped_high_ratio
        .iter()
        .sum::<f64>();
    if diagnostics.gamut_fallback_used
        || low_max > MAX_LOW_GAMUT_CLIP_RATIO_PER_CHANNEL
        || low_total > MAX_LOW_GAMUT_CLIP_RATIO_TOTAL
        || post_low_total > 0.02
        || post_high_total > 0.02
    {
        return "review_gamut".to_string();
    }
    if !diagnostics.neutral_estimate_quality.accepted {
        return "review_neutral_support".to_string();
    }
    if diagnostics
        .candidate_scores
        .iter()
        .find(|candidate| candidate.selected)
        .is_some_and(|candidate| {
            candidate
                .dominant_anchor_unstable_channels
                .iter()
                .any(|unstable| *unstable)
        })
    {
        return "review_anchor_support".to_string();
    }
    if diagnostics
        .candidate_scores
        .iter()
        .find(|candidate| candidate.selected)
        .is_some_and(|candidate| {
            candidate.quality_components.rendered_tone_penalty
                + candidate.quality_components.tone_chroma_cleanup_penalty
                > TONE_QUALITY_REVIEW_PENALTY
        })
    {
        return "review_tone_quality".to_string();
    }
    if diagnostics
        .candidate_scores
        .iter()
        .find(|candidate| candidate.selected)
        .is_some_and(|candidate| {
            candidate.quality_components.density_monotonicity_penalty
                + candidate.quality_components.hue_linearity_penalty
                + candidate.quality_components.saturation_preservation_penalty
                + candidate.quality_components.memory_color_penalty
                + candidate.quality_components.spatial_consistency_penalty
                > MODEL_QUALITY_REVIEW_PENALTY
        })
    {
        return "review_model_plausibility".to_string();
    }
    if diagnostics
        .selected_quality_score
        .is_some_and(|score| score > COLOR_CANDIDATE_REVIEW_QUALITY_SCORE)
    {
        return "review_quality_score".to_string();
    }
    "safe".to_string()
}

pub fn tone_color_trust_state(diagnostics: &ColorspaceDiagnostics) -> &'static str {
    let selected_quality = diagnostics.selected_quality_score.unwrap_or(0.0);
    if diagnostics.candidate_risk.starts_with("review_")
        || diagnostics.candidate_risk == "fallback_only"
        || selected_quality > COLOR_CANDIDATE_REVIEW_QUALITY_SCORE
    {
        "review_required"
    } else if !diagnostics.neutral_estimate_quality.accepted {
        "limited_weak_neutral"
    } else {
        "trusted"
    }
}

fn thumbnail_dimensions(h: usize, w: usize) -> (usize, usize) {
    let max_side = 180usize;
    if h == 0 || w == 0 {
        return (1, 1);
    }
    if h >= w {
        let thumb_h = h.min(max_side).max(1);
        let thumb_w = ((w as f64 * thumb_h as f64 / h as f64).round() as usize).max(1);
        (thumb_h, thumb_w)
    } else {
        let thumb_w = w.min(max_side).max(1);
        let thumb_h = ((h as f64 * thumb_w as f64 / w as f64).round() as usize).max(1);
        (thumb_h, thumb_w)
    }
}

fn render_candidate_thumbnail_panel(
    img: &Array3<f64>,
    matrix: &Matrix3<f64>,
    exposure_scale: f64,
    thumb_h: usize,
    thumb_w: usize,
) -> Array3<f64> {
    render_candidate_thumbnail_panel_with(img, exposure_scale, thumb_h, thumb_w, |source| {
        matrix * source
    })
}

fn render_candidate_thumbnail_panel_with<F>(
    img: &Array3<f64>,
    exposure_scale: f64,
    thumb_h: usize,
    thumb_w: usize,
    map: F,
) -> Array3<f64>
where
    F: Fn(Vector3<f64>) -> Vector3<f64>,
{
    let (h, w, _) = img.dim();
    let mut panel = Array3::<f64>::zeros((thumb_h, thumb_w, 3));
    for ty in 0..thumb_h {
        let y = ((ty as f64 + 0.5) * h as f64 / thumb_h as f64)
            .floor()
            .min((h.saturating_sub(1)) as f64) as usize;
        for tx in 0..thumb_w {
            let x = ((tx as f64 + 0.5) * w as f64 / thumb_w as f64)
                .floor()
                .min((w.saturating_sub(1)) as f64) as usize;
            let pixel = Vector3::new(img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]]);
            let mapped = map(pixel) / exposure_scale.max(1e-6);
            for c in 0..3 {
                panel[[ty, tx, c]] = mapped[c].clamp(0.0, 1.0);
            }
        }
    }
    panel
}

fn render_candidate_thumbnail_candidate(
    img: &Array3<f64>,
    candidate: &ColorMappingCandidate,
    exposure_scale: f64,
    thumb_h: usize,
    thumb_w: usize,
) -> Array3<f64> {
    render_candidate_thumbnail_panel_with(img, exposure_scale, thumb_h, thumb_w, |source| {
        map_candidate_pixel(candidate, source)
    })
}

fn build_candidate_comparison_thumbnail(
    img: &Array3<f64>,
    candidates: &[ColorMappingCandidate],
    selected_index: usize,
    selected_matrix: &Matrix3<f64>,
    selected_exposure_scale: f64,
) -> Option<Array3<f64>> {
    let (h, w, _) = img.dim();
    if h == 0 || w == 0 {
        return None;
    }
    let image_index = candidates
        .iter()
        .position(|candidate| candidate.kind == CandidateKind::ImageDerived)?;
    let calibration_index = candidates
        .iter()
        .position(|candidate| calibration_candidate_kind(candidate.kind));
    let comparison_indices = [Some(selected_index), Some(image_index), calibration_index];
    if comparison_indices.iter().all(Option::is_none) {
        return None;
    }

    let (thumb_h, thumb_w) = thumbnail_dimensions(h, w);
    let separator = 2usize;
    let panel_count = comparison_indices.len();
    let total_w = thumb_w * panel_count + separator * (panel_count - 1);
    let mut out = Array3::<f64>::zeros((thumb_h, total_w, 3));
    for (panel_idx, candidate_index) in comparison_indices.into_iter().enumerate() {
        let x_offset = panel_idx * (thumb_w + separator);
        let panel = if panel_idx == 0 {
            let selected = &candidates[selected_index];
            if selected.nonlinear_model.is_some() {
                render_candidate_thumbnail_candidate(
                    img,
                    selected,
                    selected_exposure_scale,
                    thumb_h,
                    thumb_w,
                )
            } else {
                render_candidate_thumbnail_panel(
                    img,
                    selected_matrix,
                    selected_exposure_scale,
                    thumb_h,
                    thumb_w,
                )
            }
        } else if let Some(candidate_index) = candidate_index {
            let candidate = &candidates[candidate_index];
            render_candidate_thumbnail_candidate(
                img,
                candidate,
                candidate.stats.exposure_scale,
                thumb_h,
                thumb_w,
            )
        } else {
            Array3::<f64>::from_elem((thumb_h, thumb_w, 3), 0.08)
        };
        for y in 0..thumb_h {
            for x in 0..thumb_w {
                for c in 0..3 {
                    out[[y, x_offset + x, c]] = panel[[y, x, c]];
                }
            }
        }
        if panel_idx + 1 < panel_count {
            for y in 0..thumb_h {
                for sx in 0..separator {
                    for c in 0..3 {
                        out[[y, x_offset + thumb_w + sx, c]] = 0.5;
                    }
                }
            }
        }
    }
    Some(out)
}

fn estimate_work_to_xyz_with_prior_matrix(
    img: &Array3<f64>,
    prior: Matrix3<f64>,
    mapping_strategy: &'static str,
    selected_mapping_reason: String,
) -> ColorspaceDiagnostics {
    let neutral_estimate = estimate_neutral_stats_detailed(img);
    let neutral_rgb = neutral_estimate.rgb;
    let neutral_pixel_count = neutral_estimate.count;
    let anchor_estimate = estimate_channel_anchors(img);
    let anchor_rgb = anchor_estimate.rgb;
    let anchor_counts = anchor_estimate.counts;

    let anchor_matrix = Matrix3::from_columns(&[
        Vector3::new(anchor_rgb[0][0], anchor_rgb[0][1], anchor_rgb[0][2]),
        Vector3::new(anchor_rgb[1][0], anchor_rgb[1][1], anchor_rgb[1][2]),
        Vector3::new(anchor_rgb[2][0], anchor_rgb[2][1], anchor_rgb[2][2]),
    ]);

    let mean_neutral = ((neutral_rgb[0] + neutral_rgb[1] + neutral_rgb[2]) / 3.0).max(1e-6);
    let anchor_low_support = channel_anchor_low_support(anchor_counts);
    let regularization_lambda = if anchor_low_support.iter().any(|low_support| *low_support) {
        0.20
    } else {
        let cond = matrix_condition_number(&anchor_matrix);
        if cond.is_finite() && cond > 30.0 {
            0.15
        } else {
            0.05
        }
    };

    let regularized_anchors =
        anchor_matrix * (1.0 - regularization_lambda) + Matrix3::identity() * regularization_lambda;
    let inverse = regularized_anchors.try_inverse();

    let mut fallback_used = false;
    let work_to_xyz = if let Some(inv) = inverse {
        let target_columns = Matrix3::from_columns(&[
            prior.column(0) * (anchor_rgb[0][0] / mean_neutral).clamp(0.25, 4.0),
            prior.column(1) * (anchor_rgb[1][1] / mean_neutral).clamp(0.25, 4.0),
            prior.column(2) * (anchor_rgb[2][2] / mean_neutral).clamp(0.25, 4.0),
        ]);
        target_columns * inv
    } else {
        fallback_used = true;
        prior
    };

    let source_white_vec = work_to_xyz
        * Vector3::new(
            neutral_rgb[0].max(1e-6),
            neutral_rgb[1].max(1e-6),
            neutral_rgb[2].max(1e-6),
        );
    let source_white = [
        source_white_vec[0].max(1e-6),
        source_white_vec[1].max(1e-6),
        source_white_vec[2].max(1e-6),
    ];

    ColorspaceDiagnostics {
        source_white,
        work_to_xyz: matrix3_to_rows(&work_to_xyz),
        condition_number: matrix_condition_number(&regularized_anchors),
        regularization_lambda,
        neutral_pixel_count,
        channel_anchor_counts: anchor_counts,
        channel_anchor_min_count: channel_anchor_min_count(anchor_counts),
        channel_anchor_low_support_threshold: MIN_CHANNEL_ANCHOR_SUPPORT,
        channel_anchor_low_support: anchor_low_support,
        dominant_anchor_rgb: anchor_rgb,
        dominant_anchor_sample_rejections: anchor_estimate.rejections,
        dominant_anchor_bands: anchor_estimate.bands,
        dominant_anchor_quality: anchor_estimate.quality,
        neutral_sample_bands: neutral_estimate.bands,
        highlight_percentile: HIGHLIGHT_HEADROOM_PERCENTILE,
        pre_scale_channel_max: [0.0; 3],
        pre_scale_channel_high_percentile: [0.0; 3],
        pre_scale_clipped_high_ratio: [0.0; 3],
        pre_scale_clipped_low_ratio: [0.0; 3],
        post_scale_clipped_high_ratio: [0.0; 3],
        post_scale_clipped_low_ratio: [0.0; 3],
        exposure_scale: 1.0,
        fallback_used,
        weak_anchor_fallback_used: false,
        weak_anchor_fallback_reason: None,
        gamut_fallback_used: false,
        gamut_fallback_reason: None,
        mapping_strategy,
        selected_mapping_reason,
        candidate_scores: Vec::new(),
        candidate_acceptance: Vec::new(),
        selected_candidate: mapping_strategy.to_string(),
        selected_candidate_rank: None,
        selected_candidate_score: None,
        selected_quality_score: None,
        technical_safety_score: None,
        color_fidelity_score: None,
        selected_runner_up_quality_delta: None,
        selection_rejections: Vec::new(),
        candidate_risk: "safe".to_string(),
        neutral_sample_rejections: neutral_estimate.rejections.clone(),
        reference_patch_evaluation: None,
        neutral_estimate_quality: neutral_estimate.quality.clone(),
        neutral_balance_scale: [1.0; 3],
        neutral_trim_scale: [1.0; 3],
        neutral_trim_applied: false,
        neutral_trim_before_after: NeutralTrimDiagnostics::skipped(
            "neutral trim not evaluated before candidate selection",
        ),
        calibration_acceptance: CalibrationAcceptanceDiagnostics::not_applicable(
            ColorMode::Auto,
            "calibration selection not evaluated before candidate selection",
        ),
        neutral_safety_rescue: NeutralSafetyRescueDiagnostics::not_evaluated(
            "neutral safety rescue not evaluated before candidate selection",
        ),
        nonlinear_color_model: None,
        pre_scale_preserved_ratio: 1.0,
        post_scale_preserved_ratio: 1.0,
        image_matrix_pre_scale_clipped_low_ratio: [0.0; 3],
        image_matrix_pre_scale_clipped_high_ratio: [0.0; 3],
        image_matrix_exposure_scale: 1.0,
        image_matrix_pre_scale_preserved_ratio: 1.0,
        image_matrix_neutral_balance_delta: [0.0; 3],
        calibrated_profile_pre_scale_clipped_low_ratio: None,
        calibrated_profile_pre_scale_clipped_high_ratio: None,
        calibrated_profile_exposure_scale: None,
        calibrated_profile_pre_scale_preserved_ratio: None,
        calibrated_profile_neutral_balance_delta: None,
    }
}

pub fn estimate_work_to_xyz(img: &Array3<f64>) -> ColorspaceDiagnostics {
    estimate_work_to_xyz_with_prior_matrix(
        img,
        prophoto_to_xyz_d50_matrix(),
        "image_derived_matrix",
        "image-derived neutral and dominant anchors selected as the colorspace candidate"
            .to_string(),
    )
}

pub fn estimate_work_to_xyz_with_prior(
    img: &Array3<f64>,
    prior_work_to_xyz: &[[f64; 3]; 3],
) -> ColorspaceDiagnostics {
    estimate_work_to_xyz_with_prior_matrix(
        img,
        const_to_matrix3(prior_work_to_xyz),
        "scanner_constrained_image_derived_matrix",
        "scanner profile selected as the stable prior; image-derived neutral and dominant anchors adapted the roll"
            .to_string(),
    )
}

/// Map an image from work RGB to ProPhoto RGB (D50), preferring an external
/// calibration profile when one has already been validated.
pub fn map_to_prophoto_d50_with_calibration_diagnostics(
    img: &Array3<f64>,
    calibration_profile: Option<&CalibrationProfile>,
) -> ColorspaceMappingResult {
    map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
        img,
        calibration_profile,
        ColorMode::Auto,
    )
    .expect("automatic colorspace mapping should always have a neutral fallback")
}

pub fn map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
    img: &Array3<f64>,
    calibration_profile: Option<&CalibrationProfile>,
    color_mode: ColorMode,
) -> Result<ColorspaceMappingResult, String> {
    let (h, w, _c) = img.dim();

    let direct_profile = calibration_profile
        .filter(|profile| profile.application_mode == CalibrationApplicationMode::DirectProfile);
    let scanner_prior_profile = calibration_profile.filter(|profile| {
        profile.application_mode == CalibrationApplicationMode::ScannerConstrainedImageAdaptation
    });
    let scanner_prior_candidate = scanner_prior_profile
        .map(|profile| {
            (
                &profile.work_to_xyz,
                profile.confidence,
                profile.fit.as_ref(),
                "scanner profile selected as the stable prior; image-derived neutral and dominant anchors adapted the roll",
            )
        })
        .or_else(|| {
            direct_profile.and_then(|profile| {
                profile.scanner_prior_work_to_xyz.as_ref().map(|work_to_xyz| {
                    (
                        work_to_xyz,
                        profile
                            .scanner_prior_confidence
                            .unwrap_or(profile.confidence),
                        profile.scanner_prior_fit.as_ref(),
                        "scanner profile retained as an automatic prior candidate alongside the direct calibrated roll profile",
                    )
                })
            })
        });
    if color_mode == ColorMode::Calibrated
        && direct_profile.is_none()
        && scanner_prior_profile.is_none()
    {
        return Err(
            "--color-mode calibrated requires an applied calibration profile or calibration library candidate"
                .to_string(),
        );
    }

    let neutral_estimate = estimate_neutral_stats_detailed(img);
    let neutral_rgb = neutral_estimate.rgb;
    let xyz_to_pro = xyz_d50_to_prophoto_matrix();

    let image_diagnostics = estimate_work_to_xyz(img);
    let image_work_to_xyz = const_to_matrix3(&image_diagnostics.work_to_xyz);
    let image_cat = bradford_cat(&image_diagnostics.source_white);
    let image_matrix = xyz_to_pro * image_cat * image_work_to_xyz;
    let image_stats = evaluate_mapping_stats(img, &image_matrix, neutral_rgb);
    let image_rendered_tone =
        evaluate_candidate_rendered_tone(img, &image_matrix, image_stats.exposure_scale);
    let image_model_quality =
        evaluate_candidate_model_quality(img, &image_matrix, image_stats.exposure_scale);
    let (neutral_matrix, neutral_balance_scale) =
        neutral_balance_mapping_from_neutral(neutral_estimate.rgb);

    let mut candidates = Vec::<ColorMappingCandidate>::new();
    candidates.push(ColorMappingCandidate {
        candidate: "image_derived_matrix",
        mapping_strategy: "image_derived_matrix",
        source_label: "image-derived",
        kind: CandidateKind::ImageDerived,
        matrix: image_matrix,
        nonlinear_model: None,
        stats: image_stats.clone(),
        diagnostics: image_diagnostics.clone(),
        neutral_balance_scale: [1.0; 3],
        selected_mapping_reason:
            "image-derived neutral and dominant anchors selected as the colorspace candidate"
                .to_string(),
        score: score_color_candidate(
            "image_derived_matrix",
            "image_derived_matrix",
            CandidateKind::ImageDerived,
            &image_stats,
            &image_diagnostics,
            image_diagnostics.channel_anchor_low_support,
            &neutral_estimate.quality,
            None,
            None,
            Some(&image_rendered_tone),
            Some(&image_model_quality),
        ),
    });

    let safe_blend_neutral_mix = if let Some((blend_matrix, blend_stats, neutral_mix)) =
        gamut_safe_image_blend(
            img,
            &image_matrix,
            &neutral_matrix,
            neutral_rgb,
            &image_stats,
        ) {
        let blend_rendered_tone =
            evaluate_candidate_rendered_tone(img, &blend_matrix, blend_stats.exposure_scale);
        let blend_model_quality =
            evaluate_candidate_model_quality(img, &blend_matrix, blend_stats.exposure_scale);
        let mut blend_diagnostics = image_diagnostics.clone();
        blend_diagnostics.fallback_used = true;
        blend_diagnostics.mapping_strategy = "gamut_safe_image_matrix_blend";
        blend_diagnostics.selected_mapping_reason = format!(
            "image-derived matrix was blended {:.1}% toward neutral balance to stay inside negative-gamut safety limits while preserving scene chroma",
            neutral_mix * 100.0
        );
        candidates.push(ColorMappingCandidate {
            candidate: "gamut_safe_image_matrix_blend",
            mapping_strategy: "gamut_safe_image_matrix_blend",
            source_label: "gamut-safe image-derived blend",
            kind: CandidateKind::ImageDerived,
            matrix: blend_matrix,
            nonlinear_model: None,
            stats: blend_stats.clone(),
            diagnostics: blend_diagnostics.clone(),
            neutral_balance_scale: [1.0; 3],
            selected_mapping_reason: blend_diagnostics.selected_mapping_reason.clone(),
            score: score_color_candidate(
                "gamut_safe_image_matrix_blend",
                "gamut_safe_image_matrix_blend",
                CandidateKind::ImageDerived,
                &blend_stats,
                &blend_diagnostics,
                image_diagnostics.channel_anchor_low_support,
                &neutral_estimate.quality,
                None,
                None,
                Some(&blend_rendered_tone),
                Some(&blend_model_quality),
            ),
        });
        Some(neutral_mix)
    } else {
        None
    };

    if let Some((blend_matrix, blend_stats, neutral_mix)) = gamut_trusted_image_blend(
        img,
        &image_matrix,
        &neutral_matrix,
        neutral_rgb,
        &image_stats,
    ) {
        let materially_stricter = safe_blend_neutral_mix
            .map(|safe_mix| neutral_mix > safe_mix + 0.005)
            .unwrap_or(true);
        if !materially_stricter {
            // Avoid adding two indistinguishable image-derived blend candidates.
        } else {
            let blend_rendered_tone =
                evaluate_candidate_rendered_tone(img, &blend_matrix, blend_stats.exposure_scale);
            let blend_model_quality =
                evaluate_candidate_model_quality(img, &blend_matrix, blend_stats.exposure_scale);
            let mut blend_diagnostics = image_diagnostics.clone();
            blend_diagnostics.fallback_used = true;
            blend_diagnostics.mapping_strategy = "gamut_trusted_image_matrix_blend";
            blend_diagnostics.selected_mapping_reason = format!(
            "image-derived matrix was blended {:.1}% toward neutral balance to keep low-gamut excursions under the display review threshold while preserving remaining scene chroma",
            neutral_mix * 100.0
        );
            candidates.push(ColorMappingCandidate {
                candidate: "gamut_trusted_image_matrix_blend",
                mapping_strategy: "gamut_trusted_image_matrix_blend",
                source_label: "gamut-trusted image-derived blend",
                kind: CandidateKind::ImageDerived,
                matrix: blend_matrix,
                nonlinear_model: None,
                stats: blend_stats.clone(),
                diagnostics: blend_diagnostics.clone(),
                neutral_balance_scale: [1.0; 3],
                selected_mapping_reason: blend_diagnostics.selected_mapping_reason.clone(),
                score: score_color_candidate(
                    "gamut_trusted_image_matrix_blend",
                    "gamut_trusted_image_matrix_blend",
                    CandidateKind::ImageDerived,
                    &blend_stats,
                    &blend_diagnostics,
                    image_diagnostics.channel_anchor_low_support,
                    &neutral_estimate.quality,
                    None,
                    None,
                    Some(&blend_rendered_tone),
                    Some(&blend_model_quality),
                ),
            });

            if let Some((stable_matrix, stable_stats, stable_mix)) = neutral_estimate
                .quality
                .accepted
                .then(|| {
                    gamut_stabilized_image_blend(
                        img,
                        &image_matrix,
                        &neutral_matrix,
                        neutral_rgb,
                        neutral_mix,
                        &blend_stats,
                    )
                })
                .flatten()
            {
                let stable_rendered_tone = evaluate_candidate_rendered_tone(
                    img,
                    &stable_matrix,
                    stable_stats.exposure_scale,
                );
                let stable_model_quality = evaluate_candidate_model_quality(
                    img,
                    &stable_matrix,
                    stable_stats.exposure_scale,
                );
                let mut stable_diagnostics = image_diagnostics.clone();
                stable_diagnostics.fallback_used = true;
                stable_diagnostics.mapping_strategy = "gamut_stabilized_image_matrix_blend";
                stable_diagnostics.selected_mapping_reason = format!(
                    "image-derived matrix was blended {:.1}% toward neutral balance to stabilize high-mix low-exposure color while preserving bounded scene chroma",
                    stable_mix * 100.0
                );
                candidates.push(ColorMappingCandidate {
                    candidate: "gamut_stabilized_image_matrix_blend",
                    mapping_strategy: "gamut_stabilized_image_matrix_blend",
                    source_label: "gamut-stabilized image-derived blend",
                    kind: CandidateKind::ImageDerived,
                    matrix: stable_matrix,
                    nonlinear_model: None,
                    stats: stable_stats.clone(),
                    diagnostics: stable_diagnostics.clone(),
                    neutral_balance_scale: [1.0; 3],
                    selected_mapping_reason: stable_diagnostics.selected_mapping_reason.clone(),
                    score: score_color_candidate(
                        "gamut_stabilized_image_matrix_blend",
                        "gamut_stabilized_image_matrix_blend",
                        CandidateKind::ImageDerived,
                        &stable_stats,
                        &stable_diagnostics,
                        image_diagnostics.channel_anchor_low_support,
                        &neutral_estimate.quality,
                        None,
                        None,
                        Some(&stable_rendered_tone),
                        Some(&stable_model_quality),
                    ),
                });
            }
        }
    }

    if let Some((prior_work_to_xyz, confidence, fit, selected_mapping_reason)) =
        scanner_prior_candidate
    {
        let scanner_diagnostics = estimate_work_to_xyz_with_prior(img, prior_work_to_xyz);
        let scanner_work_to_xyz = const_to_matrix3(&scanner_diagnostics.work_to_xyz);
        let scanner_cat = bradford_cat(&scanner_diagnostics.source_white);
        let scanner_matrix = xyz_to_pro * scanner_cat * scanner_work_to_xyz;
        let scanner_stats = evaluate_mapping_stats(img, &scanner_matrix, neutral_rgb);
        let scanner_rendered_tone =
            evaluate_candidate_rendered_tone(img, &scanner_matrix, scanner_stats.exposure_scale);
        let scanner_model_quality =
            evaluate_candidate_model_quality(img, &scanner_matrix, scanner_stats.exposure_scale);
        candidates.push(ColorMappingCandidate {
            candidate: "scanner_prior_image_adaptation",
            mapping_strategy: "scanner_constrained_image_derived_matrix",
            source_label: "scanner-constrained image-derived",
            kind: CandidateKind::ScannerPrior,
            matrix: scanner_matrix,
            nonlinear_model: None,
            stats: scanner_stats.clone(),
            diagnostics: scanner_diagnostics.clone(),
            neutral_balance_scale: [1.0; 3],
            selected_mapping_reason: selected_mapping_reason.to_string(),
            score: score_color_candidate(
                "scanner_prior_image_adaptation",
                "scanner_constrained_image_derived_matrix",
                CandidateKind::ScannerPrior,
                &scanner_stats,
                &scanner_diagnostics,
                scanner_diagnostics.channel_anchor_low_support,
                &neutral_estimate.quality,
                Some(confidence),
                fit,
                Some(&scanner_rendered_tone),
                Some(&scanner_model_quality),
            ),
        });
    }

    if let Some(profile) = direct_profile {
        let work_to_xyz = const_to_matrix3(&profile.work_to_xyz);
        let cat = bradford_cat(&profile.whitepoint);
        let calibrated_matrix = xyz_to_pro * cat * work_to_xyz;
        let calibrated_stats = evaluate_mapping_stats(img, &calibrated_matrix, neutral_rgb);
        let calibrated_rendered_tone = evaluate_candidate_rendered_tone(
            img,
            &calibrated_matrix,
            calibrated_stats.exposure_scale,
        );
        let calibrated_model_quality = evaluate_candidate_model_quality(
            img,
            &calibrated_matrix,
            calibrated_stats.exposure_scale,
        );
        let mut calibrated_diagnostics = image_diagnostics.clone();
        calibrated_diagnostics.source_white = profile.whitepoint;
        calibrated_diagnostics.work_to_xyz = profile.work_to_xyz;
        calibrated_diagnostics.condition_number = profile.matrix_condition_number;
        calibrated_diagnostics.regularization_lambda = 0.0;
        calibrated_diagnostics.fallback_used = false;
        calibrated_diagnostics.mapping_strategy = "calibrated_profile";
        calibrated_diagnostics.selected_mapping_reason = if profile.roll_correction_applied {
            "scanner profile and roll correction selected over image-derived estimate".to_string()
        } else {
            "valid external calibration profile selected over image-derived estimate".to_string()
        };
        candidates.push(ColorMappingCandidate {
            candidate: "calibrated_direct_profile",
            mapping_strategy: "calibrated_profile",
            source_label: "calibrated profile",
            kind: CandidateKind::CalibratedDirect,
            matrix: calibrated_matrix,
            nonlinear_model: None,
            stats: calibrated_stats.clone(),
            diagnostics: calibrated_diagnostics.clone(),
            neutral_balance_scale: [1.0; 3],
            selected_mapping_reason: calibrated_diagnostics.selected_mapping_reason.clone(),
            score: score_color_candidate(
                "calibrated_direct_profile",
                "calibrated_profile",
                CandidateKind::CalibratedDirect,
                &calibrated_stats,
                &calibrated_diagnostics,
                [false; 3],
                &neutral_estimate.quality,
                Some(profile.confidence),
                profile.fit.as_ref(),
                Some(&calibrated_rendered_tone),
                Some(&calibrated_model_quality),
            ),
        });

        if let Some(model) = profile.color_model.as_ref() {
            let post_xyz = profile
                .color_model_post_xyz
                .as_ref()
                .map(const_to_matrix3)
                .unwrap_or_else(Matrix3::identity);
            let runtime_model = RuntimeNonlinearTransform::RootPolynomial {
                model: Arc::new(model.clone()),
                xyz_to_prophoto: xyz_to_pro * cat * post_xyz,
            };
            let nonlinear_stats =
                evaluate_mapping_stats_with(img, neutral_rgb, |source| runtime_model.map(source));
            let nonlinear_rendered_tone = evaluate_nonlinear_candidate_rendered_tone(
                img,
                &runtime_model,
                nonlinear_stats.exposure_scale,
            );
            let nonlinear_model_quality = evaluate_nonlinear_candidate_model_quality(
                img,
                &runtime_model,
                nonlinear_stats.exposure_scale,
            );
            let support = nonlinear_model_runtime_diagnostics(img, &runtime_model);
            let mut nonlinear_diagnostics = calibrated_diagnostics.clone();
            nonlinear_diagnostics.condition_number = model.validation.design_condition_number;
            nonlinear_diagnostics.regularization_lambda = model.validation.regularization_lambda;
            nonlinear_diagnostics.mapping_strategy = "calibrated_root_polynomial";
            nonlinear_diagnostics.selected_mapping_reason = format!(
                "held-out-qualified degree-{} root-polynomial model `{}` was evaluated against its matrix fallback with scene-domain, gamut, tone, and perceptual diagnostics",
                model.degree, model.model_id
            );
            nonlinear_diagnostics.nonlinear_color_model = Some(support.clone());
            let mut nonlinear_score = score_color_candidate(
                "calibrated_root_polynomial",
                "calibrated_root_polynomial",
                CandidateKind::CalibratedDirect,
                &nonlinear_stats,
                &nonlinear_diagnostics,
                [false; 3],
                &neutral_estimate.quality,
                Some(profile.confidence),
                Some(&model.fit),
                Some(&nonlinear_rendered_tone),
                Some(&nonlinear_model_quality),
            );
            reject_candidate_for_nonlinear_support(&mut nonlinear_score, &support);
            candidates.push(ColorMappingCandidate {
                candidate: "calibrated_root_polynomial",
                mapping_strategy: "calibrated_root_polynomial",
                source_label: "held-out calibrated root-polynomial profile",
                kind: CandidateKind::CalibratedDirect,
                matrix: calibrated_matrix,
                nonlinear_model: Some(runtime_model),
                stats: nonlinear_stats,
                diagnostics: nonlinear_diagnostics.clone(),
                neutral_balance_scale: [1.0; 3],
                selected_mapping_reason: nonlinear_diagnostics.selected_mapping_reason.clone(),
                score: nonlinear_score,
            });
        }

        if let Some(model) = profile.lut_3d_model.as_ref() {
            let post_xyz = profile
                .color_model_post_xyz
                .as_ref()
                .map(const_to_matrix3)
                .unwrap_or_else(Matrix3::identity);
            let baseline_matrix = *profile
                .scanner_prior_work_to_xyz
                .as_ref()
                .unwrap_or(&profile.work_to_xyz);
            let root_baseline = (model.baseline_kind == "root_polynomial")
                .then(|| profile.color_model.clone().map(Arc::new))
                .flatten();
            let runtime_model = RuntimeNonlinearTransform::ResidualLut3d {
                model: Arc::new(model.clone()),
                baseline_matrix,
                root_baseline,
                xyz_to_prophoto: xyz_to_pro * cat * post_xyz,
            };
            let nonlinear_stats =
                evaluate_mapping_stats_with(img, neutral_rgb, |source| runtime_model.map(source));
            let nonlinear_rendered_tone = evaluate_nonlinear_candidate_rendered_tone(
                img,
                &runtime_model,
                nonlinear_stats.exposure_scale,
            );
            let nonlinear_model_quality = evaluate_nonlinear_candidate_model_quality(
                img,
                &runtime_model,
                nonlinear_stats.exposure_scale,
            );
            let support = nonlinear_model_runtime_diagnostics(img, &runtime_model);
            let mut nonlinear_diagnostics = calibrated_diagnostics.clone();
            nonlinear_diagnostics.condition_number = model.validation.regularized_condition_number;
            nonlinear_diagnostics.regularization_lambda = model.validation.regularization_lambda;
            nonlinear_diagnostics.mapping_strategy = "calibrated_residual_lut_3d";
            nonlinear_diagnostics.selected_mapping_reason = format!(
                "held-out-qualified {}x{}x{} smooth residual LUT `{}` over its {} baseline was evaluated against the simpler calibrated candidates with scene-domain, gamut, tone, and perceptual diagnostics",
                model.grid_size,
                model.grid_size,
                model.grid_size,
                model.model_id,
                model.baseline_kind
            );
            nonlinear_diagnostics.nonlinear_color_model = Some(support.clone());
            let mut nonlinear_score = score_color_candidate(
                "calibrated_residual_lut_3d",
                "calibrated_residual_lut_3d",
                CandidateKind::CalibratedDirect,
                &nonlinear_stats,
                &nonlinear_diagnostics,
                [false; 3],
                &neutral_estimate.quality,
                Some(profile.confidence),
                Some(&model.fit),
                Some(&nonlinear_rendered_tone),
                Some(&nonlinear_model_quality),
            );
            reject_candidate_for_nonlinear_support(&mut nonlinear_score, &support);
            candidates.push(ColorMappingCandidate {
                candidate: "calibrated_residual_lut_3d",
                mapping_strategy: "calibrated_residual_lut_3d",
                source_label: "held-out calibrated smooth residual 3D LUT profile",
                kind: CandidateKind::CalibratedDirect,
                matrix: calibrated_matrix,
                nonlinear_model: Some(runtime_model),
                stats: nonlinear_stats,
                diagnostics: nonlinear_diagnostics.clone(),
                neutral_balance_scale: [1.0; 3],
                selected_mapping_reason: nonlinear_diagnostics.selected_mapping_reason.clone(),
                score: nonlinear_score,
            });
        }
    }

    let neutral_stats = evaluate_mapping_stats(img, &neutral_matrix, neutral_rgb);
    let neutral_rendered_tone =
        evaluate_candidate_rendered_tone(img, &neutral_matrix, neutral_stats.exposure_scale);
    let neutral_model_quality =
        evaluate_candidate_model_quality(img, &neutral_matrix, neutral_stats.exposure_scale);
    let mut neutral_diagnostics = image_diagnostics.clone();
    neutral_diagnostics.condition_number = 1.0;
    neutral_diagnostics.regularization_lambda = 0.0;
    neutral_diagnostics.fallback_used = false;
    neutral_diagnostics.mapping_strategy = "neutral_balance_fallback";
    neutral_diagnostics.selected_mapping_reason =
        "neutral balance selected as the explicit colorspace fallback".to_string();
    candidates.push(ColorMappingCandidate {
        candidate: "neutral_balance_fallback",
        mapping_strategy: "neutral_balance_fallback",
        source_label: "neutral-balance fallback",
        kind: CandidateKind::NeutralFallback,
        matrix: neutral_matrix,
        nonlinear_model: None,
        stats: neutral_stats.clone(),
        diagnostics: neutral_diagnostics.clone(),
        neutral_balance_scale,
        selected_mapping_reason: "neutral balance selected as the explicit colorspace fallback"
            .to_string(),
        score: score_color_candidate(
            "neutral_balance_fallback",
            "neutral_balance_fallback",
            CandidateKind::NeutralFallback,
            &neutral_stats,
            &neutral_diagnostics,
            [false; 3],
            &neutral_estimate.quality,
            None,
            None,
            Some(&neutral_rendered_tone),
            Some(&neutral_model_quality),
        ),
    });

    if let Some(profile) = calibration_profile {
        if !profile.target_patches.is_empty() {
            let _ =
                apply_reference_patch_evaluation(&mut candidates, &profile.target_patches, None);
        }
    }

    let selection = select_color_candidate(&candidates, color_mode)?;
    let selected_index = selection.selected_index;
    let reference_patch_evaluation = calibration_profile.and_then(|profile| {
        (!profile.target_patches.is_empty()).then(|| {
            apply_reference_patch_evaluation(
                &mut candidates,
                &profile.target_patches,
                Some(selected_index),
            )
        })?
    });
    let selected_candidate = candidates[selected_index].clone();
    let matrix_candidate_count = candidates
        .iter()
        .filter(|candidate| candidate.kind.is_matrix())
        .count();
    let rejected_matrix_count = candidates
        .iter()
        .filter(|candidate| candidate.kind.is_matrix() && candidate.score.rejected)
        .count();
    let selected_neutral_due_to_gamut = color_mode == ColorMode::Auto
        && selected_candidate.kind == CandidateKind::NeutralFallback
        && matrix_candidate_count > 0
        && matrix_candidate_count == rejected_matrix_count;

    let (combined, neutral_trim_diagnostics) = if selected_candidate.nonlinear_model.is_some() {
        (
            selected_candidate.matrix,
            NeutralTrimDiagnostics::skipped(
                "neutral trim skipped: the selected nonlinear calibration is preserved exactly as held-out validated; creative/technical white balance remains a separate reported stage",
            ),
        )
    } else if selected_candidate.kind.is_matrix() {
        evaluate_neutral_trim(
            img,
            &neutral_estimate,
            &selected_candidate.matrix,
            &selected_candidate.stats,
        )
    } else {
        let mut diagnostics = NeutralTrimDiagnostics::skipped(
            "neutral trim skipped: selected candidate is already the neutral-balance fallback",
        );
        diagnostics.before = Some(neutral_trim_stats(
            &selected_candidate.stats,
            &neutral_estimate,
            &selected_candidate.matrix,
        ));
        (selected_candidate.matrix, diagnostics)
    };
    let neutral_trim_scale = neutral_trim_diagnostics.scale;
    let neutral_trim_applied = neutral_trim_diagnostics.applied;
    let chosen_stats = if neutral_trim_applied {
        evaluate_mapping_stats(img, &combined, neutral_rgb)
    } else {
        selected_candidate.stats.clone()
    };
    let exposure_scale = chosen_stats.exposure_scale;
    let comparison_thumbnail = build_candidate_comparison_thumbnail(
        img,
        &candidates,
        selected_index,
        &combined,
        exposure_scale,
    );

    let mut out = Array3::<f64>::zeros((h, w, 3));
    let post_scale_clipped_high = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    let post_scale_clipped_low = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    let post_scale_any_clipped = AtomicU64::new(0);
    out.axis_chunks_iter_mut(Axis(0), streaming::DEFAULT_TILE_ROWS)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut out_chunk)| {
            let row_start = chunk_idx * streaming::DEFAULT_TILE_ROWS;
            let chunk_h = out_chunk.dim().0;
            for local_y in 0..chunk_h {
                let y = row_start + local_y;
                for x in 0..w {
                    let pixel = Vector3::new(img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]]);
                    let mapped = selected_candidate
                        .nonlinear_model
                        .as_ref()
                        .map_or_else(|| combined * pixel, |model| model.map(pixel))
                        / exposure_scale;
                    let mut any_clipped = false;
                    for c in 0..3 {
                        if mapped[c] > 1.0 + GAMUT_CLIP_EPSILON {
                            post_scale_clipped_high[c].fetch_add(1, Ordering::Relaxed);
                            any_clipped = true;
                        }
                        if mapped[c] < -GAMUT_CLIP_EPSILON {
                            post_scale_clipped_low[c].fetch_add(1, Ordering::Relaxed);
                            any_clipped = true;
                        }
                        out_chunk[[local_y, x, c]] = if mapped[c].is_finite() {
                            mapped[c]
                        } else {
                            0.0
                        };
                    }
                    if any_clipped {
                        post_scale_any_clipped.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });

    let legacy_image_candidate = legacy_image_candidate(&candidates, color_mode);
    let calibrated_candidate = candidates
        .iter()
        .find(|candidate| candidate.kind == CandidateKind::CalibratedDirect);
    let ranks = candidate_ranks(&candidates);
    let selected_runner_up_quality_delta =
        selected_runner_up_quality_delta(&candidates, selected_index, color_mode);
    let candidate_acceptance = candidate_acceptance_for_selection(
        &candidates,
        color_mode,
        selected_index,
        &ranks,
        &selection.neutral_safety_rescue,
    );
    let candidate_scores = candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| {
            let mut score = candidate.score.clone();
            score.rank = ranks[idx];
            score.selected = idx == selected_index;
            score.selected_quality_delta =
                Some(score.quality_score - selected_candidate.score.quality_score);
            score
        })
        .collect::<Vec<_>>();
    let mut diagnostics = selected_candidate.diagnostics.clone();
    diagnostics.nonlinear_color_model = selected_candidate
        .diagnostics
        .nonlinear_color_model
        .clone()
        .or_else(|| {
            candidates
                .iter()
                .rev()
                .find_map(|candidate| candidate.diagnostics.nonlinear_color_model.clone())
        });
    let total_pixels = (h * w).max(1) as f64;
    diagnostics.pre_scale_channel_max = chosen_stats.pre_scale_channel_max;
    diagnostics.pre_scale_channel_high_percentile = chosen_stats.pre_scale_channel_high_percentile;
    diagnostics.pre_scale_clipped_high_ratio = chosen_stats.pre_scale_clipped_high_ratio;
    diagnostics.pre_scale_clipped_low_ratio = chosen_stats.pre_scale_clipped_low_ratio;
    diagnostics.pre_scale_preserved_ratio = chosen_stats.pre_scale_preserved_ratio;
    diagnostics.post_scale_clipped_high_ratio = std::array::from_fn(|c| {
        post_scale_clipped_high[c].load(Ordering::Relaxed) as f64 / total_pixels
    });
    diagnostics.post_scale_clipped_low_ratio = std::array::from_fn(|c| {
        post_scale_clipped_low[c].load(Ordering::Relaxed) as f64 / total_pixels
    });
    diagnostics.post_scale_preserved_ratio =
        1.0 - post_scale_any_clipped.load(Ordering::Relaxed) as f64 / total_pixels;
    diagnostics.exposure_scale = exposure_scale;
    diagnostics.weak_anchor_fallback_used = false;
    diagnostics.weak_anchor_fallback_reason = None;
    diagnostics.gamut_fallback_used = selected_neutral_due_to_gamut;
    diagnostics.gamut_fallback_reason = if selected_neutral_due_to_gamut {
        Some(
            "all matrix colorspace candidates exceeded negative-gamut clipping safety limits; using neutral-balance fallback"
                .to_string(),
        )
    } else {
        None
    };
    diagnostics.mapping_strategy = if selected_neutral_due_to_gamut {
        "neutral_balance_gamut_fallback"
    } else if selection.neutral_safety_rescue.applied {
        "neutral_balance_evidence_rescue"
    } else if color_mode == ColorMode::Neutral {
        "neutral_balance_forced"
    } else {
        selected_candidate.mapping_strategy
    };
    diagnostics.selected_mapping_reason = if selected_neutral_due_to_gamut {
        diagnostics
            .gamut_fallback_reason
            .clone()
            .unwrap_or_default()
    } else if selection.neutral_safety_rescue.applied {
        selection.neutral_safety_rescue.reason.clone()
    } else if color_mode != ColorMode::Auto {
        format!(
            "--color-mode {} selected {}",
            color_mode.as_str(),
            selected_candidate.source_label
        )
    } else {
        selected_candidate.selected_mapping_reason.clone()
    };
    diagnostics.candidate_scores = candidate_scores;
    diagnostics.candidate_acceptance = candidate_acceptance;
    diagnostics.selected_candidate = selected_candidate.candidate.to_string();
    diagnostics.selected_candidate_rank = ranks[selected_index];
    diagnostics.selected_candidate_score = Some(selected_candidate.score.score);
    diagnostics.selected_quality_score = Some(selected_candidate.score.quality_score);
    diagnostics.technical_safety_score = Some(selected_candidate.score.technical_safety_score);
    diagnostics.color_fidelity_score = Some(selected_candidate.score.color_fidelity_score);
    diagnostics.selected_runner_up_quality_delta = selected_runner_up_quality_delta;
    diagnostics.neutral_safety_rescue = selection.neutral_safety_rescue.clone();
    diagnostics.selection_rejections = selection.selection_rejections;
    diagnostics.neutral_sample_rejections = neutral_estimate.rejections.clone();
    diagnostics.reference_patch_evaluation = reference_patch_evaluation;
    diagnostics.neutral_estimate_quality = neutral_estimate.quality.clone();
    diagnostics.neutral_balance_scale = selected_candidate.neutral_balance_scale;
    diagnostics.neutral_trim_scale = neutral_trim_scale;
    diagnostics.neutral_trim_applied = neutral_trim_applied;
    diagnostics.neutral_trim_before_after = neutral_trim_diagnostics;
    diagnostics.calibration_acceptance = selection.calibration_acceptance;
    diagnostics.image_matrix_pre_scale_clipped_low_ratio =
        legacy_image_candidate.stats.pre_scale_clipped_low_ratio;
    diagnostics.image_matrix_pre_scale_clipped_high_ratio =
        legacy_image_candidate.stats.pre_scale_clipped_high_ratio;
    diagnostics.image_matrix_exposure_scale = legacy_image_candidate.stats.exposure_scale;
    diagnostics.image_matrix_pre_scale_preserved_ratio =
        legacy_image_candidate.stats.pre_scale_preserved_ratio;
    diagnostics.image_matrix_neutral_balance_delta =
        legacy_image_candidate.stats.neutral_balance_delta;
    diagnostics.calibrated_profile_pre_scale_clipped_low_ratio =
        calibrated_candidate.map(|candidate| candidate.stats.pre_scale_clipped_low_ratio);
    diagnostics.calibrated_profile_pre_scale_clipped_high_ratio =
        calibrated_candidate.map(|candidate| candidate.stats.pre_scale_clipped_high_ratio);
    diagnostics.calibrated_profile_exposure_scale =
        calibrated_candidate.map(|candidate| candidate.stats.exposure_scale);
    diagnostics.calibrated_profile_pre_scale_preserved_ratio =
        calibrated_candidate.map(|candidate| candidate.stats.pre_scale_preserved_ratio);
    diagnostics.calibrated_profile_neutral_balance_delta =
        calibrated_candidate.map(|candidate| candidate.stats.neutral_balance_delta);
    diagnostics.candidate_risk = candidate_risk_for_diagnostics(&diagnostics);

    Ok(ColorspaceMappingResult {
        prophoto: out,
        diagnostics,
        comparison_thumbnail,
    })
}

/// Map an image from work RGB to ProPhoto RGB (D50) via:
/// 1. Estimate an explicit work->XYZ matrix from neutral and dominant pixels
/// 2. Estimate the source white in XYZ from the neutral set
/// 3. Bradford-adapt that white to D50
pub fn map_to_prophoto_d50_with_diagnostics(img: &Array3<f64>) -> ColorspaceMappingResult {
    map_to_prophoto_d50_with_calibration_diagnostics(img, None)
}

pub fn map_to_prophoto_d50(img: &Array3<f64>) -> Array3<f64> {
    map_to_prophoto_d50_with_diagnostics(img)
        .prophoto
        .mapv(|value| value.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scoring_stats() -> MappingStats {
        MappingStats {
            pre_scale_channel_min: [0.0; 3],
            pre_scale_channel_max: [0.95; 3],
            pre_scale_channel_high_percentile: [0.95; 3],
            pre_scale_clipped_high_ratio: [0.0; 3],
            pre_scale_clipped_low_ratio: [0.0; 3],
            pre_scale_max_low_excursion: [0.0; 3],
            pre_scale_max_high_excursion: [0.0; 3],
            exposure_scale: 1.0,
            pre_scale_preserved_ratio: 1.0,
            neutral_balance_delta: [0.0; 3],
        }
    }

    fn scoring_diagnostics(anchor_low_support: [bool; 3]) -> ColorspaceDiagnostics {
        let img = Array3::<f64>::from_elem((8, 8, 3), 0.5);
        let mut diagnostics = estimate_work_to_xyz(&img);
        diagnostics.condition_number = 1.0;
        diagnostics.fallback_used = false;
        diagnostics.channel_anchor_counts = [128, 128, 128];
        diagnostics.channel_anchor_low_support = anchor_low_support;
        diagnostics
    }

    fn neutral_quality(score: f64) -> NeutralEstimateQuality {
        NeutralEstimateQuality {
            score,
            accepted: score >= 1.0,
            broad_support: score >= 1.0,
            reason: "synthetic neutral quality".to_string(),
            sample_count: 256,
            sample_fraction: 0.5,
            band_counts: [80, 96, 80],
            populated_band_count: 3,
            dominant_band_fraction: 0.375,
            minimum_samples: MIN_NEUTRAL_TRIM_SAMPLES,
            minimum_bands: MIN_NEUTRAL_TRIM_BANDS,
        }
    }

    fn score_for(
        stats: &MappingStats,
        diagnostics: &ColorspaceDiagnostics,
        neutral: &NeutralEstimateQuality,
        target_fit: Option<&TargetFitDiagnostics>,
    ) -> ColorMappingCandidateScore {
        score_color_candidate(
            "calibrated_direct_profile",
            "calibrated_profile",
            CandidateKind::CalibratedDirect,
            stats,
            diagnostics,
            diagnostics.channel_anchor_low_support,
            neutral,
            Some(1.0),
            target_fit,
            None,
            None,
        )
    }

    #[test]
    fn trusted_gamut_blend_requires_more_neutral_mix_than_minimum_safe_blend() {
        let mut img = Array3::<f64>::zeros((12, 1000, 3));
        for y in 0..12 {
            for x in 0..1000 {
                img[[y, x, 0]] = (x as f64 + 0.5) / 1000.0;
                img[[y, x, 1]] = 1.0;
                img[[y, x, 2]] = 0.5;
            }
        }

        let destructive = Matrix3::new(1.0, -0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let neutral = Matrix3::identity();
        let neutral_rgb = [0.5, 1.0, 0.5];
        let image_stats = evaluate_mapping_stats(&img, &destructive, neutral_rgb);
        assert!(!within_low_gamut_limits(&image_stats));

        let (_safe_matrix, safe_stats, safe_mix) =
            gamut_safe_image_blend(&img, &destructive, &neutral, neutral_rgb, &image_stats)
                .expect("minimum safe blend");
        let (_trusted_matrix, trusted_stats, trusted_mix) =
            gamut_trusted_image_blend(&img, &destructive, &neutral, neutral_rgb, &image_stats)
                .expect("trusted display blend");

        assert!(
            trusted_mix > safe_mix + 0.10,
            "trusted mix {trusted_mix} should be materially stricter than safe mix {safe_mix}"
        );
        assert!(within_low_gamut_limits(&safe_stats));
        assert!(
            !within_trusted_low_gamut_limits(&safe_stats),
            "minimum safe blend should still exceed display-trust low-gamut limits"
        );
        assert!(
            within_trusted_low_gamut_limits(&trusted_stats),
            "trusted blend should satisfy the stricter display low-gamut limits"
        );
    }

    #[test]
    fn trusted_gamut_blend_runs_for_broad_safe_but_display_untrusted_candidate() {
        let mut img = Array3::<f64>::zeros((12, 100, 3));
        for y in 0..12 {
            for x in 0..100 {
                img[[y, x, 0]] = (x as f64 + 0.5) / 100.0;
                img[[y, x, 1]] = 1.0;
                img[[y, x, 2]] = 0.5;
            }
        }

        let mildly_negative = Matrix3::new(1.0, -0.03, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let neutral = Matrix3::identity();
        let neutral_rgb = [0.5, 1.0, 0.5];
        let image_stats = evaluate_mapping_stats(&img, &mildly_negative, neutral_rgb);

        assert!(
            within_low_gamut_limits(&image_stats),
            "fixture should be inside broad negative-gamut safety limits"
        );
        assert!(
            !within_trusted_low_gamut_limits(&image_stats),
            "fixture should still exceed display-trust low-gamut limits"
        );

        let (_trusted_matrix, trusted_stats, neutral_mix) =
            gamut_trusted_image_blend(&img, &mildly_negative, &neutral, neutral_rgb, &image_stats)
                .expect("trusted blend should be available for broad-safe review candidates");

        assert!(neutral_mix > 0.0);
        assert!(
            within_trusted_low_gamut_limits(&trusted_stats),
            "trusted blend should reduce clipping below display-trust limits"
        );
    }

    #[test]
    fn stabilized_gamut_blend_requires_high_mix_low_exposure_candidate() {
        let mut img = Array3::<f64>::zeros((12, 1000, 3));
        for y in 0..12 {
            for x in 0..1000 {
                img[[y, x, 0]] = (x as f64 + 0.5) / 1000.0;
                img[[y, x, 1]] = 1.0;
                img[[y, x, 2]] = 0.5;
            }
        }

        let destructive = Matrix3::new(1.0, -0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let neutral = Matrix3::identity();
        let neutral_rgb = [0.5, 1.0, 0.5];
        let image_stats = evaluate_mapping_stats(&img, &destructive, neutral_rgb);
        let (_trusted_matrix, trusted_stats, trusted_mix) =
            gamut_trusted_image_blend(&img, &destructive, &neutral, neutral_rgb, &image_stats)
                .expect("trusted blend should be available");

        assert!(
            trusted_mix >= GAMUT_STABILIZED_BLEND_MIN_TRUSTED_MIX,
            "fixture should require a high neutral mix, got {trusted_mix}"
        );
        assert!(
            trusted_stats.exposure_scale <= GAMUT_STABILIZED_BLEND_MAX_EXPOSURE_SCALE,
            "fixture should stay inside the low-exposure stabilization gate"
        );

        let (_stable_matrix, stable_stats, stable_mix) = gamut_stabilized_image_blend(
            &img,
            &destructive,
            &neutral,
            neutral_rgb,
            trusted_mix,
            &trusted_stats,
        )
        .expect("stabilized blend should be available for high-mix low-exposure case");

        assert!(stable_mix > trusted_mix + GAMUT_STABILIZED_BLEND_MIN_STEP);
        assert!(stable_mix <= GAMUT_SAFE_BLEND_MAX_NEUTRAL_MIX);
        assert!(within_trusted_low_gamut_limits(&stable_stats));
        assert!(
            total_ratio(stable_stats.pre_scale_clipped_low_ratio)
                < total_ratio(trusted_stats.pre_scale_clipped_low_ratio),
            "stabilized blend should reduce remaining low-gamut excursions"
        );

        let mut high_exposure_stats = trusted_stats.clone();
        high_exposure_stats.exposure_scale = GAMUT_STABILIZED_BLEND_MAX_EXPOSURE_SCALE + 0.01;
        assert!(gamut_stabilized_image_blend(
            &img,
            &destructive,
            &neutral,
            neutral_rgb,
            trusted_mix,
            &high_exposure_stats,
        )
        .is_none());
    }

    #[test]
    fn ciede2000_matches_reference_lab_pairs() {
        let cases = [
            ([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485], 2.0425),
            ([50.0, 3.1571, -77.2803], [50.0, 0.0, -82.7485], 2.8615),
            ([50.0, 2.8361, -74.0200], [50.0, 0.0, -82.7485], 3.4412),
            ([50.0, -1.3802, -84.2814], [50.0, 0.0, -82.7485], 1.0000),
        ];

        for (left, right, expected) in cases {
            let actual = lab_delta_e2000(left, right);
            assert!(
                (actual - expected).abs() < 0.0001,
                "DeltaE2000 mismatch: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn ciede2000_reference_regression_is_a_decision_signal() {
        assert!(!reference_patch_regresses_image_derived(
            0.0,
            0.0,
            0.0,
            0.0,
            REFERENCE_DELTA_E2000_RMS_REGRESSION_TOLERANCE,
            REFERENCE_DELTA_E2000_MAX_REGRESSION_TOLERANCE,
        ));
        assert!(reference_patch_regresses_image_derived(
            0.0,
            0.0,
            0.0,
            0.0,
            REFERENCE_DELTA_E2000_RMS_REGRESSION_TOLERANCE + 0.01,
            0.0,
        ));
        assert!(reference_patch_regresses_image_derived(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            REFERENCE_DELTA_E2000_MAX_REGRESSION_TOLERANCE + 0.01,
        ));
        assert!(reference_patch_improves_image_derived(
            0.0,
            0.0,
            0.0,
            0.0,
            -(REFERENCE_DELTA_E2000_RMS_REGRESSION_TOLERANCE + 0.01),
            0.0,
        ));
    }

    #[test]
    fn color_candidate_score_penalties_are_monotonic() {
        let diagnostics = scoring_diagnostics([false; 3]);
        let neutral = neutral_quality(1.0);
        let base_stats = scoring_stats();
        let base = score_for(&base_stats, &diagnostics, &neutral, None);

        let mut worse_low_gamut = base_stats.clone();
        worse_low_gamut.pre_scale_clipped_low_ratio = [0.02, 0.0, 0.0];
        worse_low_gamut.pre_scale_preserved_ratio = 0.98;
        let worse_low_gamut_score = score_for(&worse_low_gamut, &diagnostics, &neutral, None);
        assert!(worse_low_gamut_score.quality_score > base.quality_score);
        assert!(worse_low_gamut_score.technical_safety_score > base.technical_safety_score);
        assert_eq!(
            worse_low_gamut_score.color_fidelity_score,
            base.color_fidelity_score
        );

        let mut worse_high_gamut = base_stats.clone();
        worse_high_gamut.pre_scale_clipped_high_ratio = [0.0, 0.03, 0.0];
        worse_high_gamut.pre_scale_preserved_ratio = 0.97;
        let worse_high_gamut_score = score_for(&worse_high_gamut, &diagnostics, &neutral, None);
        assert!(worse_high_gamut_score.quality_score > base.quality_score);
        assert!(worse_high_gamut_score.technical_safety_score > base.technical_safety_score);

        let mut worse_exposure = base_stats.clone();
        worse_exposure.exposure_scale = 1.25;
        let worse_exposure_score = score_for(&worse_exposure, &diagnostics, &neutral, None);
        assert!(worse_exposure_score.quality_score > base.quality_score);
        assert!(worse_exposure_score.technical_safety_score > base.technical_safety_score);

        let mut worse_neutral = base_stats.clone();
        worse_neutral.neutral_balance_delta = [0.08, -0.05, 0.02];
        let worse_neutral_score = score_for(&worse_neutral, &diagnostics, &neutral, None);
        assert!(worse_neutral_score.quality_score > base.quality_score);
        assert!(worse_neutral_score.color_fidelity_score > base.color_fidelity_score);
        assert_eq!(
            worse_neutral_score.technical_safety_score,
            base.technical_safety_score
        );

        let weak_neutral = neutral_quality(0.25);
        let weak_neutral_score = score_for(&base_stats, &diagnostics, &weak_neutral, None);
        assert!(weak_neutral_score.quality_score > base.quality_score);
        assert!(weak_neutral_score.color_fidelity_score > base.color_fidelity_score);

        let weak_anchor_diagnostics = scoring_diagnostics([false, true, false]);
        let weak_anchor_score = score_for(&base_stats, &weak_anchor_diagnostics, &neutral, None);
        assert!(weak_anchor_score.quality_score > base.quality_score);
        assert!(weak_anchor_score.color_fidelity_score > base.color_fidelity_score);

        let fit = TargetFitDiagnostics {
            method: "synthetic".to_string(),
            patch_count: 24,
            target_residual_rms: 0.05,
            target_residual_max: 0.10,
            validation: None,
            per_hue_residuals: Vec::new(),
            worst_patches: Vec::new(),
        };
        let worse_fit_score = score_for(&base_stats, &diagnostics, &neutral, Some(&fit));
        assert!(worse_fit_score.quality_score > base.quality_score);
        assert!(worse_fit_score.color_fidelity_score > base.color_fidelity_score);
    }

    #[test]
    fn high_quality_score_marks_candidate_for_review() {
        let mut diagnostics = scoring_diagnostics([false; 3]);
        diagnostics.neutral_estimate_quality = neutral_quality(1.0);
        diagnostics.selected_quality_score = Some(COLOR_CANDIDATE_REVIEW_QUALITY_SCORE + 0.1);
        diagnostics.candidate_risk = candidate_risk_for_diagnostics(&diagnostics);

        assert_eq!(diagnostics.candidate_risk, "review_quality_score");
        assert_eq!(tone_color_trust_state(&diagnostics), "review_required");
    }
}

use ndarray::parallel::prelude::*;
use ndarray::{Array3, Axis};
use std::sync::atomic::{AtomicU64, Ordering};

const PROPHOTO_LUMA: [f64; 3] = [0.2880, 0.7119, 0.0001];
const NUM_BINS: usize = 1000;
const PERCEPTUAL_LUMINANCE_GAIN: f64 = 15.0;
const LINEAR_FIT_MEDIAN_THRESHOLD: f64 = 0.25;
const LINEAR_FIT_SLOPE_SCALE: f64 = 1.6;
const PERCEPTUAL_FIT_SLOPE_SCALE: f64 = 2.5;
const LINEAR_FIT_MIN_MIDPOINT: f64 = 0.10;
const LINEAR_FIT_MAX_MIDPOINT: f64 = 0.90;
const PERCEPTUAL_FIT_MIN_MIDPOINT: f64 = 0.005;
const PERCEPTUAL_FIT_MAX_MIDPOINT: f64 = 0.90;
const PERCEPTUAL_FIT_MAX_SLOPE: f64 = 10.0;
const NEGATIVE_LOW_RANGE_LINEAR_MAX_P95: f64 = 0.12;
const NEGATIVE_LOW_RANGE_LINEAR_DYNAMIC_RANGE: f64 = 0.045;
const NEGATIVE_LOW_RANGE_TARGET_LINEAR_MEDIAN: f64 = 0.35;
const NEGATIVE_LOW_RANGE_MIN_MAPPED_P95: f64 = 0.52;
const NEGATIVE_LOW_RANGE_MIN_MAPPED_P95_GAIN: f64 = 0.03;
const NEGATIVE_LOW_RANGE_MAX_SLOPE: f64 = 22.0;
const NEGATIVE_SHADOW_TARGET_LINEAR_MEDIAN: f64 = 0.32;
const NEGATIVE_SHADOW_TARGET_MAX_MAPPED_P95: f64 = 0.95;
const NEGATIVE_SHADOW_TARGET_LINEAR_MEDIAN_MAX: f64 = 0.22;
const NEGATIVE_SHADOW_MAX_SLOPE: f64 = 6.1;
const POSITIVE_SCAN_AUTO_EXPOSURE_HIGHLIGHT_TARGET: f64 = 0.90;
const POSITIVE_SCAN_AUTO_EXPOSURE_MIDTONE_CEILING: f64 = 0.64;
const POSITIVE_SCAN_AUTO_EXPOSURE_MIN_EV: f64 = -0.45;
const POSITIVE_SCAN_LINEAR_FIT_SLOPE_SCALE: f64 = 1.05;
const POSITIVE_SCAN_LINEAR_FIT_MIN_SLOPE: f64 = 1.35;
const POSITIVE_SCAN_LINEAR_FIT_MAX_SLOPE: f64 = 3.80;
const POSITIVE_SCAN_LINEAR_FIT_MAX_MIDPOINT: f64 = 0.999;
const POSITIVE_SCAN_TARGET_MEDIAN_INPUT_WEIGHT: f64 = 0.80;
const POSITIVE_SCAN_TARGET_MEDIAN_OFFSET: f64 = 0.045;
const POSITIVE_SCAN_TARGET_MEDIAN_MIN: f64 = 0.14;
const POSITIVE_SCAN_TARGET_MEDIAN_MAX: f64 = 0.64;
const POSITIVE_SCAN_TOE_LIFT: f64 = 0.003;
const POSITIVE_SCAN_SHOULDER_MAX: f64 = 0.988;
const HIGHLIGHT_NEUTRAL_CHROMA_START_LUMINANCE: f64 = 0.72;
const HIGHLIGHT_NEUTRAL_CHROMA_FULL_LUMINANCE: f64 = 0.92;
const HIGHLIGHT_NEUTRAL_CHROMA_MIN_SCALE: f64 = 0.40;
const HIGHLIGHT_NEUTRAL_CHROMA_MAX_SATURATION: f64 = 0.35;
const MIDTONE_NEUTRAL_CHROMA_START_LUMINANCE: f64 = 0.24;
const MIDTONE_NEUTRAL_CHROMA_FULL_LOW_LUMINANCE: f64 = 0.32;
const MIDTONE_NEUTRAL_CHROMA_FULL_HIGH_LUMINANCE: f64 = 0.62;
const MIDTONE_NEUTRAL_CHROMA_END_LUMINANCE: f64 = 0.76;
const MIDTONE_NEUTRAL_CHROMA_MIN_SCALE: f64 = 0.52;
const MIDTONE_NEUTRAL_CHROMA_MAX_SATURATION: f64 = 0.35;
const SHADOW_CHROMA_START_LUMINANCE: f64 = 0.24;
const SHADOW_CHROMA_FULL_LUMINANCE: f64 = 0.08;
const SHADOW_CHROMA_MIN_SCALE: f64 = 0.25;
const SHADOW_SATURATION_GUARD_DEEP_MAX: f64 = 0.54;
const SHADOW_SATURATION_GUARD_UPPER_MAX: f64 = 0.74;
const RENDER_QUALITY_MAX_SAMPLES: usize = 1_000_000;
const RENDER_SHADOW_BAND_FRACTION: f64 = 0.10;
const RENDER_SHADOW_VISIBLE_MIN_LUMINANCE: f64 = 0.03;
const RENDER_MIDTONE_BAND_LOW_FRACTION: f64 = 0.40;
const RENDER_MIDTONE_BAND_HIGH_FRACTION: f64 = 0.60;
const RENDER_BRIGHT_BAND_FRACTION: f64 = 0.10;
const RENDER_MIDTONE_NEUTRAL_MAX_SATURATION: f64 = 0.35;
const RENDER_BRIGHT_NEUTRAL_MAX_SATURATION: f64 = HIGHLIGHT_NEUTRAL_CHROMA_MAX_SATURATION;
const RENDER_GRAIN_MAX_SAMPLES: usize = 250_000;
const RENDER_GRAIN_FLAT_STRUCTURE_RADIUS: usize = 2;
const RENDER_GRAIN_FLAT_LUMA_STRUCTURE_MAX: f64 = 0.045;
const EXTENDED_HIGHLIGHT_KNEE: f64 = 0.92;
const EXTENDED_HIGHLIGHT_ROLLOFF: f64 = 0.60;
const LOCAL_LUMINANCE_DETAIL_RADIUS: usize = 7;
const LOCAL_LUMINANCE_DETAIL_AMOUNT: f64 = 0.18;
const LOCAL_LUMINANCE_DETAIL_MAX_EV: f64 = 0.14;
const LOCAL_LUMINANCE_DETAIL_SHADOW_START: f64 = 0.04;
const LOCAL_LUMINANCE_DETAIL_SHADOW_FULL: f64 = 0.18;
const LOCAL_LUMINANCE_DETAIL_HIGHLIGHT_START: f64 = 0.94;
const LOCAL_LUMINANCE_DETAIL_HIGHLIGHT_END: f64 = 1.0;
const LOCAL_LUMINANCE_DETAIL_POSITIVE_HEADROOM_START_EV: f64 = 0.015;
const LOCAL_LUMINANCE_DETAIL_POSITIVE_HEADROOM_FULL_EV: f64 = 0.12;
const LOCAL_LUMINANCE_DETAIL_STRUCTURE_START: f64 = 0.020;
const LOCAL_LUMINANCE_DETAIL_STRUCTURE_FULL: f64 = 0.075;
const ADAPTIVE_VIBRANCE_MIN_DIMENSION: usize = 15;
const ADAPTIVE_VIBRANCE_AMOUNT: f64 = 0.10;
const ADAPTIVE_VIBRANCE_MAX_SCALE: f64 = 1.12;
const ADAPTIVE_VIBRANCE_SHADOW_START: f64 = 0.08;
const ADAPTIVE_VIBRANCE_SHADOW_FULL: f64 = 0.22;
const ADAPTIVE_VIBRANCE_HIGHLIGHT_START: f64 = 0.86;
const ADAPTIVE_VIBRANCE_HIGHLIGHT_END: f64 = 0.98;
const ADAPTIVE_VIBRANCE_NEUTRAL_START: f64 = 0.035;
const ADAPTIVE_VIBRANCE_NEUTRAL_FULL: f64 = 0.12;
const ADAPTIVE_VIBRANCE_SATURATION_START: f64 = 0.42;
const ADAPTIVE_VIBRANCE_SATURATION_END: f64 = 0.82;
const ADAPTIVE_VIBRANCE_TEXTURE_RADIUS: usize = 2;
const ADAPTIVE_VIBRANCE_TEXTURE_START: f64 = 0.035;
const ADAPTIVE_VIBRANCE_TEXTURE_END: f64 = 0.12;
const RENDER_NOISE_REDUCTION_RADIUS: usize = 2;
const RENDER_NOISE_REDUCTION_CHROMA_RADIUS: usize = 2;
const RENDER_NOISE_REDUCTION_CHROMA_AMOUNT: f64 = 0.96;
const RENDER_NOISE_REDUCTION_LUMA_AMOUNT: f64 = 0.30;
const RENDER_NOISE_REDUCTION_TEXTURE_START: f64 = 0.018;
const RENDER_NOISE_REDUCTION_TEXTURE_END: f64 = 0.180;
const RENDER_NOISE_REDUCTION_CHROMA_TEXTURE_RESIDUAL_WEIGHT: f64 = 0.24;
const RENDER_NOISE_REDUCTION_LUMA_TEXTURE_RESIDUAL_WEIGHT: f64 = 0.10;
const RENDER_NOISE_REDUCTION_LUMA_TEXTURE_FLOOR: f64 = 0.50;
const RENDER_NOISE_REDUCTION_FRAME_CHROMA_GRAIN_START: f64 = 0.075;
const RENDER_NOISE_REDUCTION_FRAME_CHROMA_GRAIN_FULL: f64 = 0.125;
const RENDER_NOISE_REDUCTION_CHROMA_DAMP_AMOUNT: f64 = 0.85;
const RENDER_NOISE_REDUCTION_NEUTRAL_TEXTURE_FLOOR: f64 = 0.85;
const RENDER_NOISE_REDUCTION_NEUTRAL_FLOOR_HIGHLIGHT_START: f64 = 0.62;
const RENDER_NOISE_REDUCTION_NEUTRAL_FLOOR_HIGHLIGHT_END: f64 = 0.82;
const RENDER_NOISE_REDUCTION_SATURATION_GATE_FLOOR: f64 = 1.0;
const RENDER_NOISE_REDUCTION_SATURATION_START: f64 = 0.24;
const RENDER_NOISE_REDUCTION_SATURATION_END: f64 = 0.72;
const RENDER_NOISE_REDUCTION_SHADOW_SATURATION_RELAXATION: f64 = 0.90;
const RENDER_NOISE_REDUCTION_SHADOW_START: f64 = 0.08;
const RENDER_NOISE_REDUCTION_SHADOW_END: f64 = 0.42;
const SCENE_REFERRED_DETAIL_FUSION_RADIUS: usize = 5;
const SCENE_REFERRED_DETAIL_FUSION_AMOUNT: f64 = 0.42;
const SCENE_REFERRED_DETAIL_FUSION_MAX_EV: f64 = 0.16;
const SCENE_REFERRED_DETAIL_FUSION_SHADOW_START: f64 = 0.015;
const SCENE_REFERRED_DETAIL_FUSION_SHADOW_FULL: f64 = 0.08;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderStyle {
    ModernClean,
    NaturalNeutral,
    FilmFaithful,
}

impl RenderStyle {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ModernClean => "modern-clean",
            Self::NaturalNeutral => "natural-neutral",
            Self::FilmFaithful => "film-faithful",
        }
    }

    fn processing_config(self) -> RenderProcessingConfig {
        match self {
            Self::ModernClean => RenderProcessingConfig {
                local_luminance_detail: true,
                adaptive_vibrance: true,
                noise_reduction: true,
            },
            Self::NaturalNeutral => RenderProcessingConfig {
                local_luminance_detail: true,
                adaptive_vibrance: false,
                noise_reduction: true,
            },
            Self::FilmFaithful => RenderProcessingConfig {
                local_luminance_detail: false,
                adaptive_vibrance: false,
                noise_reduction: false,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RenderProcessingConfig {
    local_luminance_detail: bool,
    adaptive_vibrance: bool,
    noise_reduction: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneFitDomain {
    LinearLuminance,
    Log2CompressedLuminance,
}

impl ToneFitDomain {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LinearLuminance => "linear_luminance",
            Self::Log2CompressedLuminance => "log2_compressed_luminance",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToneCurveDiagnostics {
    pub fit_domain: &'static str,
    pub perceptual_luminance_gain: f64,
    pub input_linear_percentiles: [f64; 3],
    pub input_perceptual_percentiles: [f64; 3],
    pub mapped_linear_percentiles: [f64; 3],
    pub mapped_perceptual_percentiles: [f64; 3],
}

pub struct ToneFitResult {
    pub params: ToneCurveParams,
    pub diagnostics: ToneCurveDiagnostics,
}

#[derive(Debug, Clone)]
pub struct TonemapApplyDiagnostics {
    pub highlight_chroma_compressed_ratio: f64,
    pub highlight_neutral_chroma_compressed_ratio: f64,
    pub highlight_neutral_chroma_enabled: bool,
    pub highlight_neutral_chroma_start_luminance: f64,
    pub highlight_neutral_chroma_full_luminance: f64,
    pub highlight_neutral_chroma_min_scale: f64,
    pub highlight_neutral_chroma_max_saturation: f64,
    pub midtone_neutral_chroma_compressed_ratio: f64,
    pub midtone_neutral_chroma_enabled: bool,
    pub midtone_neutral_chroma_start_luminance: f64,
    pub midtone_neutral_chroma_end_luminance: f64,
    pub midtone_neutral_chroma_min_scale: f64,
    pub midtone_neutral_chroma_max_saturation: f64,
    pub shadow_chroma_compressed_ratio: f64,
    pub shadow_chroma_enabled: bool,
    pub shadow_chroma_start_luminance: f64,
    pub shadow_chroma_full_luminance: f64,
    pub shadow_chroma_min_scale: f64,
    pub color_protection_policy: &'static str,
    pub color_trust_state: &'static str,
    pub color_protection_reason: String,
    pub pre_chroma_compression_clipped_high_ratio: [f64; 3],
    pub post_chroma_compression_clipped_high_ratio: [f64; 3],
    pub post_chroma_compression_clipped_low_ratio: [f64; 3],
    pub local_luminance_detail_enabled: bool,
    pub local_luminance_detail_radius: usize,
    pub local_luminance_detail_amount: f64,
    pub local_luminance_detail_max_ev: f64,
    pub local_luminance_detail_applied_ratio: f64,
    pub local_luminance_detail_mean_abs_ev: f64,
    pub local_luminance_detail_max_abs_ev: f64,
    pub local_luminance_detail_headroom_limited_ratio: f64,
    pub local_luminance_detail_clip_limited_ratio: f64,
    pub adaptive_vibrance_enabled: bool,
    pub adaptive_vibrance_reason: String,
    pub adaptive_vibrance_amount: f64,
    pub adaptive_vibrance_max_scale: f64,
    pub adaptive_vibrance_applied_ratio: f64,
    pub adaptive_vibrance_mean_scale: f64,
    pub adaptive_vibrance_max_applied_scale: f64,
    pub adaptive_vibrance_texture_limited_ratio: f64,
    pub adaptive_vibrance_gamut_limited_ratio: f64,
    pub noise_reduction_enabled: bool,
    pub noise_reduction_reason: String,
    pub noise_reduction_radius: usize,
    pub noise_reduction_chroma_amount: f64,
    pub noise_reduction_luma_amount: f64,
    pub noise_reduction_applied_ratio: f64,
    pub noise_reduction_texture_limited_ratio: f64,
    pub noise_reduction_saturation_limited_ratio: f64,
    pub noise_reduction_mean_abs_chroma_delta: f64,
    pub noise_reduction_max_abs_chroma_delta: f64,
    pub noise_reduction_mean_abs_luma_delta: f64,
    pub noise_reduction_max_abs_luma_delta: f64,
}

pub struct TonemapApplyResult {
    pub image: Array3<f64>,
    pub diagnostics: TonemapApplyDiagnostics,
}

#[derive(Debug, Clone)]
pub struct SceneReferredDetailFusionDiagnostics {
    pub enabled: bool,
    pub reason: String,
    pub radius: usize,
    pub amount: f64,
    pub max_ev: f64,
    pub applied_ratio: f64,
    pub mean_abs_ev: f64,
    pub max_abs_ev: f64,
    pub skipped_negative_ratio: f64,
    pub skipped_nonfinite_ratio: f64,
}

impl SceneReferredDetailFusionDiagnostics {
    pub fn disabled(reason: impl Into<String>) -> Self {
        Self {
            enabled: false,
            reason: reason.into(),
            radius: SCENE_REFERRED_DETAIL_FUSION_RADIUS,
            amount: SCENE_REFERRED_DETAIL_FUSION_AMOUNT,
            max_ev: SCENE_REFERRED_DETAIL_FUSION_MAX_EV,
            applied_ratio: 0.0,
            mean_abs_ev: 0.0,
            max_abs_ev: 0.0,
            skipped_negative_ratio: 0.0,
            skipped_nonfinite_ratio: 0.0,
        }
    }
}

pub struct SceneReferredDetailFusionResult {
    pub image: Array3<f64>,
    pub diagnostics: SceneReferredDetailFusionDiagnostics,
}

#[derive(Debug, Clone, Copy)]
struct LocalLuminanceDetailDiagnostics {
    enabled: bool,
    radius: usize,
    amount: f64,
    max_ev: f64,
    applied_ratio: f64,
    mean_abs_ev: f64,
    max_abs_ev: f64,
    headroom_limited_ratio: f64,
    clip_limited_ratio: f64,
}

impl LocalLuminanceDetailDiagnostics {
    fn disabled() -> Self {
        Self {
            enabled: false,
            radius: LOCAL_LUMINANCE_DETAIL_RADIUS,
            amount: LOCAL_LUMINANCE_DETAIL_AMOUNT,
            max_ev: LOCAL_LUMINANCE_DETAIL_MAX_EV,
            applied_ratio: 0.0,
            mean_abs_ev: 0.0,
            max_abs_ev: 0.0,
            headroom_limited_ratio: 0.0,
            clip_limited_ratio: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
struct AdaptiveVibranceDiagnostics {
    enabled: bool,
    reason: String,
    amount: f64,
    max_scale: f64,
    applied_ratio: f64,
    mean_scale: f64,
    max_applied_scale: f64,
    texture_limited_ratio: f64,
    gamut_limited_ratio: f64,
}

impl AdaptiveVibranceDiagnostics {
    fn disabled(reason: impl Into<String>) -> Self {
        Self {
            enabled: false,
            reason: reason.into(),
            amount: ADAPTIVE_VIBRANCE_AMOUNT,
            max_scale: ADAPTIVE_VIBRANCE_MAX_SCALE,
            applied_ratio: 0.0,
            mean_scale: 1.0,
            max_applied_scale: 1.0,
            texture_limited_ratio: 0.0,
            gamut_limited_ratio: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
struct RenderNoiseReductionDiagnostics {
    enabled: bool,
    reason: String,
    radius: usize,
    chroma_amount: f64,
    luma_amount: f64,
    applied_ratio: f64,
    texture_limited_ratio: f64,
    saturation_limited_ratio: f64,
    mean_abs_chroma_delta: f64,
    max_abs_chroma_delta: f64,
    mean_abs_luma_delta: f64,
    max_abs_luma_delta: f64,
}

impl RenderNoiseReductionDiagnostics {
    fn disabled(reason: impl Into<String>) -> Self {
        Self {
            enabled: false,
            reason: reason.into(),
            radius: RENDER_NOISE_REDUCTION_RADIUS,
            chroma_amount: RENDER_NOISE_REDUCTION_CHROMA_AMOUNT,
            luma_amount: RENDER_NOISE_REDUCTION_LUMA_AMOUNT,
            applied_ratio: 0.0,
            texture_limited_ratio: 0.0,
            saturation_limited_ratio: 0.0,
            mean_abs_chroma_delta: 0.0,
            max_abs_chroma_delta: 0.0,
            mean_abs_luma_delta: 0.0,
            max_abs_luma_delta: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneColorProtectionPolicy {
    Enabled,
    NeutralHighlightDisabledWeakNeutral,
    WeakNeutralBoundedHighlightCleanup,
    WeakNeutralBoundedNeutralCleanup,
    ReviewBoundedNeutralAndShadowCleanup,
    DisabledColorCandidateReview,
}

impl ToneColorProtectionPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Enabled => "enabled",
            Self::NeutralHighlightDisabledWeakNeutral => "neutral_highlight_disabled_weak_neutral",
            Self::WeakNeutralBoundedHighlightCleanup => "weak_neutral_bounded_highlight_cleanup",
            Self::WeakNeutralBoundedNeutralCleanup => "weak_neutral_bounded_neutral_cleanup",
            Self::ReviewBoundedNeutralAndShadowCleanup => "review_bounded_neutral_shadow_cleanup",
            Self::DisabledColorCandidateReview => "disabled_color_candidate_review",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToneColorProtection {
    pub policy: ToneColorProtectionPolicy,
    pub highlight_neutral_chroma_enabled: bool,
    pub midtone_neutral_chroma_enabled: bool,
    pub shadow_chroma_enabled: bool,
    pub reason: String,
}

impl Default for ToneColorProtection {
    fn default() -> Self {
        Self {
            policy: ToneColorProtectionPolicy::Enabled,
            highlight_neutral_chroma_enabled: true,
            midtone_neutral_chroma_enabled: true,
            shadow_chroma_enabled: true,
            reason: "default tone color protection enabled".to_string(),
        }
    }
}

impl ToneColorProtection {
    pub fn color_trust_state(&self) -> &'static str {
        match self.policy {
            ToneColorProtectionPolicy::Enabled => "trusted",
            ToneColorProtectionPolicy::NeutralHighlightDisabledWeakNeutral
            | ToneColorProtectionPolicy::WeakNeutralBoundedHighlightCleanup
            | ToneColorProtectionPolicy::WeakNeutralBoundedNeutralCleanup => "limited_weak_neutral",
            ToneColorProtectionPolicy::ReviewBoundedNeutralAndShadowCleanup => "review_required",
            ToneColorProtectionPolicy::DisabledColorCandidateReview => "review_required",
        }
    }

    pub fn from_colorspace_diagnostics(
        diagnostics: &crate::colorspace::ColorspaceDiagnostics,
    ) -> Self {
        let selected_quality = diagnostics.selected_quality_score.unwrap_or(0.0);
        if diagnostics.candidate_risk == "review_neutral_support"
            || (!diagnostics.candidate_risk.starts_with("review_")
                && !diagnostics.neutral_estimate_quality.accepted)
        {
            return Self {
                policy: ToneColorProtectionPolicy::WeakNeutralBoundedNeutralCleanup,
                highlight_neutral_chroma_enabled: true,
                midtone_neutral_chroma_enabled: true,
                shadow_chroma_enabled: true,
                reason: format!(
                    "bounded neutral highlight, neutral midtone, and shadow chroma cleanup remain enabled because {}; color remains limited_weak_neutral because neutral support is not broad enough for full trust",
                    diagnostics.neutral_estimate_quality.reason
                ),
            };
        }

        if diagnostics.candidate_risk == "review_anchor_support" {
            return Self {
                policy: ToneColorProtectionPolicy::ReviewBoundedNeutralAndShadowCleanup,
                highlight_neutral_chroma_enabled: true,
                midtone_neutral_chroma_enabled: true,
                shadow_chroma_enabled: true,
                reason: "bounded neutral midtone/highlight and shadow chroma cleanup remain enabled for anchor-support review frames to reduce snow/highlight casts and low-tone colour speckle without marking colour trusted".to_string(),
            };
        }

        if diagnostics.candidate_risk == "review_model_plausibility" {
            return Self {
                policy: ToneColorProtectionPolicy::ReviewBoundedNeutralAndShadowCleanup,
                highlight_neutral_chroma_enabled: true,
                midtone_neutral_chroma_enabled: true,
                shadow_chroma_enabled: true,
                reason: "bounded neutral midtone/highlight and shadow chroma cleanup remain enabled for model-plausibility review frames because neutral and anchor evidence are accepted, reducing neutral casts and low-tone colour speckle without marking colour trusted".to_string(),
            };
        }

        if diagnostics.candidate_risk.starts_with("review_")
            || selected_quality > crate::colorspace::COLOR_CANDIDATE_REVIEW_QUALITY_SCORE
        {
            return Self {
                policy: ToneColorProtectionPolicy::DisabledColorCandidateReview,
                highlight_neutral_chroma_enabled: false,
                midtone_neutral_chroma_enabled: false,
                shadow_chroma_enabled: false,
                reason: if diagnostics.candidate_risk.starts_with("review_") {
                    format!(
                        "disabled because selected colorspace candidate risk `{}` requires color review",
                        diagnostics.candidate_risk
                    )
                } else {
                    format!(
                    "disabled because selected colorspace quality score {:.3} requires color review",
                    selected_quality
                    )
                },
            };
        }

        Self {
            policy: ToneColorProtectionPolicy::Enabled,
            highlight_neutral_chroma_enabled: true,
            midtone_neutral_chroma_enabled: true,
            shadow_chroma_enabled: true,
            reason: format!(
                "enabled from colorspace candidate `{}` with quality score {:.3}",
                diagnostics.selected_candidate, selected_quality
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderBandDiagnostics {
    pub pixel_count: usize,
    pub luminance_percentiles: [f64; 3],
    pub saturation_median: f64,
    pub saturation_p95: f64,
    pub rgb_median: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct RenderQualityDiagnostics {
    pub sample_count: usize,
    pub sample_stride: usize,
    pub shadow_luminance_max: f64,
    pub shadow_visible_luminance_min: f64,
    pub midtone_luminance_range: [f64; 2],
    pub midtone_neutral_max_saturation: f64,
    pub bright_neutral_luminance_min: f64,
    pub bright_neutral_max_saturation: f64,
    pub overall: RenderBandDiagnostics,
    pub shadow: RenderBandDiagnostics,
    pub shadow_visible: RenderBandDiagnostics,
    pub midtone: RenderBandDiagnostics,
    pub midtone_neutral: RenderBandDiagnostics,
    pub midtone_saturated: RenderBandDiagnostics,
    pub bright_neutral: RenderBandDiagnostics,
    pub bright_saturated: RenderBandDiagnostics,
}

#[derive(Debug, Clone)]
pub struct RenderGrainDiagnostics {
    pub sample_count: usize,
    pub sample_stride: usize,
    pub flat_luma_structure_max: f64,
    pub flat_sample_count: usize,
    pub flat_sample_ratio: f64,
    pub luma_residual_median: f64,
    pub luma_residual_p95: f64,
    pub chroma_residual_median: f64,
    pub chroma_residual_p95: f64,
    pub chroma_to_luma_p95_ratio: f64,
    pub flat_luma_residual_p95: f64,
    pub flat_chroma_residual_p95: f64,
    pub flat_chroma_to_luma_p95_ratio: f64,
}

#[derive(Debug, Clone, Copy)]
struct RenderQualityPixel {
    luminance: f64,
    saturation: f64,
    rgb: [f64; 3],
}

/// Parameters for the Naka-Rushton / modified Michaelis-Menten tone curve.
#[derive(Debug, Clone, Copy)]
pub struct ToneCurveParams {
    /// Luminance domain the curve is fit and applied in.
    pub domain: ToneFitDomain,
    /// Input value that maps to ~0.5 output (sigma in the formula).
    pub midpoint: f64,
    /// Contrast / steepness (exponent n in the formula).
    pub slope: f64,
    /// Minimum output value (toe).
    pub toe_lift: f64,
    /// Maximum output value (shoulder).
    pub shoulder_max: f64,
}

impl Default for ToneCurveParams {
    fn default() -> Self {
        Self {
            domain: ToneFitDomain::Log2CompressedLuminance,
            midpoint: 0.5,
            slope: 5.0,
            toe_lift: 0.005,
            shoulder_max: 0.995,
        }
    }
}

/// Apply the Naka-Rushton tone curve to a single value.
///
/// Formula: x^n / (x^n + sigma^n), rescaled to [toe_lift, shoulder_max].
pub fn apply_tone_curve(x: f64, params: &ToneCurveParams) -> f64 {
    let x = x.max(0.0);
    let curve_input = x.min(EXTENDED_HIGHLIGHT_KNEE);
    let n = params.slope;
    let sigma = params.midpoint;

    let xn = curve_input.powf(n);
    let sn = sigma.powf(n);
    let raw = (xn / (xn + sn)) * (1.0 + sn);
    let base = params.toe_lift + raw * (params.shoulder_max - params.toe_lift);
    if x <= EXTENDED_HIGHLIGHT_KNEE {
        return base;
    }

    let highlight = 1.0 - (-(x - EXTENDED_HIGHLIGHT_KNEE) / EXTENDED_HIGHLIGHT_ROLLOFF).exp();
    base + (params.shoulder_max - base) * highlight
}

fn linear_luminance(r: f64, g: f64, b: f64) -> f64 {
    (PROPHOTO_LUMA[0] * r + PROPHOTO_LUMA[1] * g + PROPHOTO_LUMA[2] * b).max(0.0)
}

fn encode_perceptual_luminance(lum: f64) -> f64 {
    let gain = PERCEPTUAL_LUMINANCE_GAIN;
    ((1.0 + gain * lum.max(0.0)).log2()) / (1.0 + gain).log2()
}

fn decode_perceptual_luminance(encoded: f64) -> f64 {
    let gain = PERCEPTUAL_LUMINANCE_GAIN;
    (2.0f64.powf(encoded.clamp(0.0, 1.0) * (1.0 + gain).log2()) - 1.0) / gain
}

fn compress_highlight_chroma(
    rgb: [f64; 3],
    mapped_lum: f64,
    gamut_ceiling: f64,
) -> ([f64; 3], bool) {
    if rgb.iter().all(|v| *v >= 0.0 && *v <= 1.0) {
        return (rgb, false);
    }

    let neutral = mapped_lum.clamp(0.0, 1.0);
    let upper = gamut_ceiling.clamp(neutral, 1.0);
    let mut chroma_scale = 1.0f64;
    for value in rgb {
        let delta = value - neutral;
        if value > 1.0 && delta > 1e-12 {
            chroma_scale = chroma_scale.min((upper - neutral) / delta);
        } else if value < 0.0 && delta < -1e-12 {
            chroma_scale = chroma_scale.min((0.0 - neutral) / delta);
        }
    }

    if !chroma_scale.is_finite() {
        chroma_scale = 0.0;
    }
    let chroma_scale = chroma_scale.clamp(0.0, 1.0);
    let compressed = std::array::from_fn(|c| neutral + (rgb[c] - neutral) * chroma_scale);
    (compressed, chroma_scale < 1.0 - 1e-12)
}

fn smoothstep01(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn rgb_saturation(rgb: [f64; 3]) -> f64 {
    let max_v = rgb.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_v <= 1e-12 {
        return 0.0;
    }
    let min_v = rgb.iter().copied().fold(f64::INFINITY, f64::min);
    ((max_v - min_v) / max_v).clamp(0.0, 1.0)
}

fn percentile_from_sorted_values(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn render_band_diagnostics(band: &[RenderQualityPixel]) -> RenderBandDiagnostics {
    if band.is_empty() {
        return RenderBandDiagnostics {
            pixel_count: 0,
            luminance_percentiles: [0.0; 3],
            saturation_median: 0.0,
            saturation_p95: 0.0,
            rgb_median: [0.0; 3],
        };
    }

    let mut luminance = Vec::with_capacity(band.len());
    let mut saturation = Vec::with_capacity(band.len());
    let mut rgb = [
        Vec::with_capacity(band.len()),
        Vec::with_capacity(band.len()),
        Vec::with_capacity(band.len()),
    ];
    for pixel in band {
        luminance.push(pixel.luminance);
        saturation.push(pixel.saturation);
        for (channel, value) in rgb.iter_mut().zip(pixel.rgb) {
            channel.push(value);
        }
    }
    luminance.sort_by(|a, b| a.total_cmp(b));
    saturation.sort_by(|a, b| a.total_cmp(b));
    for channel in &mut rgb {
        channel.sort_by(|a, b| a.total_cmp(b));
    }

    RenderBandDiagnostics {
        pixel_count: band.len(),
        luminance_percentiles: [
            percentile_from_sorted_values(&luminance, 0.05),
            percentile_from_sorted_values(&luminance, 0.50),
            percentile_from_sorted_values(&luminance, 0.95),
        ],
        saturation_median: percentile_from_sorted_values(&saturation, 0.50),
        saturation_p95: percentile_from_sorted_values(&saturation, 0.95),
        rgb_median: std::array::from_fn(|c| percentile_from_sorted_values(&rgb[c], 0.50)),
    }
}

/// Measure rendered-output luminance and saturation bands for quality review.
///
/// Bands are percentile-based so exposure shifts do not move the target sample
/// populations: shadows are the darkest 10%, midtones are the 40-60% luminance
/// band, and bright candidates are the brightest 10%. Bright candidates are
/// then split by saturation so neutral color-cast diagnostics do not pressure
/// intentionally saturated highlights.
pub fn render_quality_diagnostics(img: &Array3<f64>) -> RenderQualityDiagnostics {
    let shape = img.shape();
    let h = shape[0];
    let w = shape[1];
    let total_pixels = h.saturating_mul(w);
    let sample_stride = if total_pixels <= RENDER_QUALITY_MAX_SAMPLES {
        1
    } else {
        total_pixels.div_ceil(RENDER_QUALITY_MAX_SAMPLES)
    };

    let mut pixels = Vec::<RenderQualityPixel>::with_capacity(
        total_pixels
            .saturating_add(sample_stride.saturating_sub(1))
            .checked_div(sample_stride.max(1))
            .unwrap_or(0),
    );
    for y in 0..h {
        for x in 0..w {
            let pixel_index = y * w + x;
            if !pixel_index.is_multiple_of(sample_stride) {
                continue;
            }
            let rgb = [
                img[[y, x, 0]].clamp(0.0, 1.0),
                img[[y, x, 1]].clamp(0.0, 1.0),
                img[[y, x, 2]].clamp(0.0, 1.0),
            ];
            pixels.push(RenderQualityPixel {
                luminance: linear_luminance(rgb[0], rgb[1], rgb[2]),
                saturation: rgb_saturation(rgb),
                rgb,
            });
        }
    }
    pixels.sort_by(|a, b| a.luminance.total_cmp(&b.luminance));

    if pixels.is_empty() {
        let empty = render_band_diagnostics(&[]);
        return RenderQualityDiagnostics {
            sample_count: 0,
            sample_stride,
            shadow_luminance_max: 0.0,
            shadow_visible_luminance_min: RENDER_SHADOW_VISIBLE_MIN_LUMINANCE,
            midtone_luminance_range: [0.0, 0.0],
            midtone_neutral_max_saturation: RENDER_MIDTONE_NEUTRAL_MAX_SATURATION,
            bright_neutral_luminance_min: 0.0,
            bright_neutral_max_saturation: RENDER_BRIGHT_NEUTRAL_MAX_SATURATION,
            overall: empty.clone(),
            shadow: empty.clone(),
            shadow_visible: empty.clone(),
            midtone: empty.clone(),
            midtone_neutral: empty.clone(),
            midtone_saturated: empty.clone(),
            bright_neutral: empty.clone(),
            bright_saturated: empty,
        };
    }

    let n = pixels.len();
    let shadow_end = ((n as f64 * RENDER_SHADOW_BAND_FRACTION).ceil() as usize).clamp(1, n);
    let midtone_start = ((n as f64 * RENDER_MIDTONE_BAND_LOW_FRACTION).floor() as usize).min(n - 1);
    let midtone_end = ((n as f64 * RENDER_MIDTONE_BAND_HIGH_FRACTION).ceil() as usize)
        .clamp(midtone_start + 1, n);
    let bright_start =
        ((n as f64 * (1.0 - RENDER_BRIGHT_BAND_FRACTION)).floor() as usize).min(n - 1);
    let shadow_visible = pixels[..shadow_end]
        .iter()
        .copied()
        .filter(|pixel| pixel.luminance >= RENDER_SHADOW_VISIBLE_MIN_LUMINANCE)
        .collect::<Vec<_>>();
    let mut midtone_neutral = Vec::<RenderQualityPixel>::new();
    let mut midtone_saturated = Vec::<RenderQualityPixel>::new();
    for pixel in &pixels[midtone_start..midtone_end] {
        if pixel.saturation <= RENDER_MIDTONE_NEUTRAL_MAX_SATURATION {
            midtone_neutral.push(*pixel);
        } else {
            midtone_saturated.push(*pixel);
        }
    }
    let mut bright_neutral = Vec::<RenderQualityPixel>::new();
    let mut bright_saturated = Vec::<RenderQualityPixel>::new();
    for pixel in &pixels[bright_start..] {
        if pixel.saturation <= RENDER_BRIGHT_NEUTRAL_MAX_SATURATION {
            bright_neutral.push(*pixel);
        } else {
            bright_saturated.push(*pixel);
        }
    }

    RenderQualityDiagnostics {
        sample_count: n,
        sample_stride,
        shadow_luminance_max: pixels[shadow_end - 1].luminance,
        shadow_visible_luminance_min: RENDER_SHADOW_VISIBLE_MIN_LUMINANCE,
        midtone_luminance_range: [
            pixels[midtone_start].luminance,
            pixels[midtone_end - 1].luminance,
        ],
        midtone_neutral_max_saturation: RENDER_MIDTONE_NEUTRAL_MAX_SATURATION,
        bright_neutral_luminance_min: pixels[bright_start].luminance,
        bright_neutral_max_saturation: RENDER_BRIGHT_NEUTRAL_MAX_SATURATION,
        overall: render_band_diagnostics(&pixels),
        shadow: render_band_diagnostics(&pixels[..shadow_end]),
        shadow_visible: render_band_diagnostics(&shadow_visible),
        midtone: render_band_diagnostics(&pixels[midtone_start..midtone_end]),
        midtone_neutral: render_band_diagnostics(&midtone_neutral),
        midtone_saturated: render_band_diagnostics(&midtone_saturated),
        bright_neutral: render_band_diagnostics(&bright_neutral),
        bright_saturated: render_band_diagnostics(&bright_saturated),
    }
}

fn pixel_luminance(img: &Array3<f64>, y: usize, x: usize) -> f64 {
    linear_luminance(
        img[[y, x, 0]].clamp(0.0, 1.0),
        img[[y, x, 1]].clamp(0.0, 1.0),
        img[[y, x, 2]].clamp(0.0, 1.0),
    )
}

fn local_luminance_structure(img: &Array3<f64>, y: usize, x: usize) -> f64 {
    let mut left = 0.0;
    let mut right = 0.0;
    for yy in (y - 1)..=(y + 1) {
        for xx in (x - 2)..x {
            left += pixel_luminance(img, yy, xx);
        }
        for xx in (x + 1)..=(x + 2) {
            right += pixel_luminance(img, yy, xx);
        }
    }

    let mut up = 0.0;
    let mut down = 0.0;
    for yy in (y - 2)..y {
        for xx in (x - 1)..=(x + 1) {
            up += pixel_luminance(img, yy, xx);
        }
    }
    for yy in (y + 1)..=(y + 2) {
        for xx in (x - 1)..=(x + 1) {
            down += pixel_luminance(img, yy, xx);
        }
    }

    let horizontal = ((right - left) / 6.0f64).abs();
    let vertical = ((down - up) / 6.0f64).abs();
    horizontal.max(vertical)
}

fn smoothed_luminance_structure(
    luminance: &[f32],
    height: usize,
    width: usize,
    y: usize,
    x: usize,
    radius: usize,
) -> Option<f64> {
    if radius == 0 || y < radius || y + radius >= height || x < radius || x + radius >= width {
        return None;
    }
    let idx = y * width + x;
    let horizontal = (luminance[idx + radius] - luminance[idx - radius]).abs() as f64;
    let vertical =
        (luminance[(y + radius) * width + x] - luminance[(y - radius) * width + x]).abs() as f64;
    Some(horizontal.max(vertical))
}

/// Measure sampled high-frequency residuals in the final rendered image.
///
/// This is diagnostic-only: it compares each sampled pixel to a 3x3 local mean,
/// reports the residual split into luminance and chroma components, and repeats
/// the p95 measurements on low-structure samples so scene edges/textures do not
/// masquerade as flat-field grain.
pub fn render_grain_diagnostics(img: &Array3<f64>) -> RenderGrainDiagnostics {
    let shape = img.shape();
    let h = shape[0];
    let w = shape[1];
    if h < 3 || w < 3 {
        return RenderGrainDiagnostics {
            sample_count: 0,
            sample_stride: 1,
            flat_luma_structure_max: RENDER_GRAIN_FLAT_LUMA_STRUCTURE_MAX,
            flat_sample_count: 0,
            flat_sample_ratio: 0.0,
            luma_residual_median: 0.0,
            luma_residual_p95: 0.0,
            chroma_residual_median: 0.0,
            chroma_residual_p95: 0.0,
            chroma_to_luma_p95_ratio: 0.0,
            flat_luma_residual_p95: 0.0,
            flat_chroma_residual_p95: 0.0,
            flat_chroma_to_luma_p95_ratio: 0.0,
        };
    }

    let inner_pixels = (h - 2).saturating_mul(w - 2);
    let sample_stride = if inner_pixels <= RENDER_GRAIN_MAX_SAMPLES {
        1
    } else {
        inner_pixels.div_ceil(RENDER_GRAIN_MAX_SAMPLES)
    };
    let mut luma_residuals = Vec::<f64>::with_capacity(inner_pixels / sample_stride.max(1) + 1);
    let mut chroma_residuals = Vec::<f64>::with_capacity(luma_residuals.capacity());
    let mut flat_luma_residuals = Vec::<f64>::with_capacity(luma_residuals.capacity());
    let mut flat_chroma_residuals = Vec::<f64>::with_capacity(luma_residuals.capacity());

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let pixel_index = (y - 1) * (w - 2) + (x - 1);
            if !pixel_index.is_multiple_of(sample_stride) {
                continue;
            }

            let center = [
                img[[y, x, 0]].clamp(0.0, 1.0),
                img[[y, x, 1]].clamp(0.0, 1.0),
                img[[y, x, 2]].clamp(0.0, 1.0),
            ];
            let mut mean = [0.0f64; 3];
            for yy in (y - 1)..=(y + 1) {
                for xx in (x - 1)..=(x + 1) {
                    for c in 0..3 {
                        mean[c] += img[[yy, xx, c]].clamp(0.0, 1.0);
                    }
                }
            }
            for channel in &mut mean {
                *channel /= 9.0;
            }

            let residual: [f64; 3] = std::array::from_fn(|c| center[c] - mean[c]);
            let luma_signed = PROPHOTO_LUMA[0] * residual[0]
                + PROPHOTO_LUMA[1] * residual[1]
                + PROPHOTO_LUMA[2] * residual[2];
            let chroma_squared = residual
                .iter()
                .map(|channel| {
                    let chroma_component = *channel - luma_signed;
                    chroma_component * chroma_component
                })
                .sum::<f64>()
                / 3.0;
            let luma_residual = luma_signed.abs();
            let chroma_residual = chroma_squared.sqrt();
            luma_residuals.push(luma_residual);
            chroma_residuals.push(chroma_residual);

            let flat_radius = RENDER_GRAIN_FLAT_STRUCTURE_RADIUS;
            if y >= flat_radius && y + flat_radius < h && x >= flat_radius && x + flat_radius < w {
                let structure = local_luminance_structure(img, y, x);
                if structure <= RENDER_GRAIN_FLAT_LUMA_STRUCTURE_MAX {
                    flat_luma_residuals.push(luma_residual);
                    flat_chroma_residuals.push(chroma_residual);
                }
            }
        }
    }

    luma_residuals.sort_by(|a, b| a.total_cmp(b));
    chroma_residuals.sort_by(|a, b| a.total_cmp(b));
    flat_luma_residuals.sort_by(|a, b| a.total_cmp(b));
    flat_chroma_residuals.sort_by(|a, b| a.total_cmp(b));
    let luma_residual_median = percentile_from_sorted_values(&luma_residuals, 0.50);
    let luma_residual_p95 = percentile_from_sorted_values(&luma_residuals, 0.95);
    let chroma_residual_median = percentile_from_sorted_values(&chroma_residuals, 0.50);
    let chroma_residual_p95 = percentile_from_sorted_values(&chroma_residuals, 0.95);
    let chroma_to_luma_p95_ratio = if luma_residual_p95 > 1e-12 {
        chroma_residual_p95 / luma_residual_p95
    } else {
        0.0
    };
    let flat_sample_count = flat_luma_residuals.len();
    let flat_sample_ratio = if luma_residuals.is_empty() {
        0.0
    } else {
        flat_sample_count as f64 / luma_residuals.len() as f64
    };
    let flat_luma_residual_p95 = percentile_from_sorted_values(&flat_luma_residuals, 0.95);
    let flat_chroma_residual_p95 = percentile_from_sorted_values(&flat_chroma_residuals, 0.95);
    let flat_chroma_to_luma_p95_ratio = if flat_luma_residual_p95 > 1e-12 {
        flat_chroma_residual_p95 / flat_luma_residual_p95
    } else {
        0.0
    };

    RenderGrainDiagnostics {
        sample_count: luma_residuals.len(),
        sample_stride,
        flat_luma_structure_max: RENDER_GRAIN_FLAT_LUMA_STRUCTURE_MAX,
        flat_sample_count,
        flat_sample_ratio,
        luma_residual_median,
        luma_residual_p95,
        chroma_residual_median,
        chroma_residual_p95,
        chroma_to_luma_p95_ratio,
        flat_luma_residual_p95,
        flat_chroma_residual_p95,
        flat_chroma_to_luma_p95_ratio,
    }
}

fn smoothstep_range(edge0: f64, edge1: f64, value: f64) -> f64 {
    if (edge1 - edge0).abs() <= f64::EPSILON {
        return if value >= edge1 { 1.0 } else { 0.0 };
    }
    smoothstep01((value - edge0) / (edge1 - edge0))
}

fn box_blur_luminance(luminance: &[f32], height: usize, width: usize, radius: usize) -> Vec<f32> {
    if height == 0 || width == 0 || radius == 0 {
        return luminance.to_vec();
    }

    let mut horizontal = vec![0.0f32; luminance.len()];
    let mut prefix = vec![0.0f64; width + 1];
    for y in 0..height {
        prefix[0] = 0.0;
        for x in 0..width {
            prefix[x + 1] = prefix[x] + luminance[y * width + x] as f64;
        }
        for x in 0..width {
            let left = x.saturating_sub(radius);
            let right = (x + radius + 1).min(width);
            horizontal[y * width + x] =
                ((prefix[right] - prefix[left]) / (right - left) as f64) as f32;
        }
    }

    let mut blurred = vec![0.0f32; luminance.len()];
    prefix.resize(height + 1, 0.0);
    for x in 0..width {
        prefix[0] = 0.0;
        for y in 0..height {
            prefix[y + 1] = prefix[y] + horizontal[y * width + x] as f64;
        }
        for y in 0..height {
            let top = y.saturating_sub(radius);
            let bottom = (y + radius + 1).min(height);
            blurred[y * width + x] =
                ((prefix[bottom] - prefix[top]) / (bottom - top) as f64) as f32;
        }
    }

    blurred
}

fn scene_referred_luminance_for_detail(img: &Array3<f64>, y: usize, x: usize) -> Option<f64> {
    let r = img[[y, x, 0]];
    let g = img[[y, x, 1]];
    let b = img[[y, x, 2]];
    if !(r.is_finite() && g.is_finite() && b.is_finite()) {
        return None;
    }
    Some(linear_luminance(r.max(0.0), g.max(0.0), b.max(0.0)).max(0.0))
}

pub fn fuse_scene_referred_luminance_detail(
    base: &Array3<f64>,
    guide: &Array3<f64>,
) -> SceneReferredDetailFusionResult {
    let (height, width, channels) = base.dim();
    if base.dim() != guide.dim() {
        return SceneReferredDetailFusionResult {
            image: base.clone(),
            diagnostics: SceneReferredDetailFusionDiagnostics::disabled(
                "guide dimensions do not match the selected scene-referred render",
            ),
        };
    }
    if channels < 3 {
        return SceneReferredDetailFusionResult {
            image: base.clone(),
            diagnostics: SceneReferredDetailFusionDiagnostics::disabled(
                "scene-referred detail fusion requires at least three channels",
            ),
        };
    }
    if height < SCENE_REFERRED_DETAIL_FUSION_RADIUS * 2 + 1
        || width < SCENE_REFERRED_DETAIL_FUSION_RADIUS * 2 + 1
    {
        return SceneReferredDetailFusionResult {
            image: base.clone(),
            diagnostics: SceneReferredDetailFusionDiagnostics::disabled(
                "image is smaller than the scene-referred detail fusion window",
            ),
        };
    }

    let pixel_count = height * width;
    let mut base_luminance = vec![0.0f32; pixel_count];
    let mut guide_luminance = vec![0.0f32; pixel_count];
    let mut nonfinite = vec![false; pixel_count];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            match (
                scene_referred_luminance_for_detail(base, y, x),
                scene_referred_luminance_for_detail(guide, y, x),
            ) {
                (Some(base_lum), Some(guide_lum)) => {
                    base_luminance[idx] = base_lum as f32;
                    guide_luminance[idx] = guide_lum as f32;
                }
                _ => nonfinite[idx] = true,
            }
        }
    }

    let base_blur = box_blur_luminance(
        &base_luminance,
        height,
        width,
        SCENE_REFERRED_DETAIL_FUSION_RADIUS,
    );
    let guide_blur = box_blur_luminance(
        &guide_luminance,
        height,
        width,
        SCENE_REFERRED_DETAIL_FUSION_RADIUS,
    );
    let mut out = Array3::<f64>::zeros((height, width, channels));
    let mut applied = 0usize;
    let mut skipped_negative = 0usize;
    let mut skipped_nonfinite = 0usize;
    let mut sum_abs_ev = 0.0f64;
    let mut max_abs_ev = 0.0f64;
    let eps = 1e-5;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let mut scale = 1.0;
            if nonfinite[idx] {
                skipped_nonfinite += 1;
            } else if (0..3).any(|c| base[[y, x, c]] < -1e-6) {
                skipped_negative += 1;
            } else {
                let base_lum = base_luminance[idx] as f64;
                let base_local = base_blur[idx] as f64;
                let guide_lum = guide_luminance[idx] as f64;
                let guide_local = guide_blur[idx] as f64;
                let base_detail_ev = ((base_lum + eps) / (base_local + eps)).log2();
                let guide_detail_ev = ((guide_lum + eps) / (guide_local + eps)).log2();
                let shadow_gate = smoothstep_range(
                    SCENE_REFERRED_DETAIL_FUSION_SHADOW_START,
                    SCENE_REFERRED_DETAIL_FUSION_SHADOW_FULL,
                    base_lum,
                );
                let applied_ev = ((guide_detail_ev - base_detail_ev)
                    * SCENE_REFERRED_DETAIL_FUSION_AMOUNT
                    * shadow_gate)
                    .clamp(
                        -SCENE_REFERRED_DETAIL_FUSION_MAX_EV,
                        SCENE_REFERRED_DETAIL_FUSION_MAX_EV,
                    );
                scale = 2.0f64.powf(applied_ev);
                if applied_ev.abs() > 0.002 {
                    applied += 1;
                    let abs_ev = applied_ev.abs();
                    sum_abs_ev += abs_ev;
                    max_abs_ev = max_abs_ev.max(abs_ev);
                }
            }
            for c in 0..channels {
                out[[y, x, c]] = base[[y, x, c]] * scale;
            }
        }
    }

    let denom = pixel_count.max(1) as f64;
    SceneReferredDetailFusionResult {
        image: out,
        diagnostics: SceneReferredDetailFusionDiagnostics {
            enabled: true,
            reason: "transferred bounded scene-referred luminance detail from direct-density guide"
                .to_string(),
            radius: SCENE_REFERRED_DETAIL_FUSION_RADIUS,
            amount: SCENE_REFERRED_DETAIL_FUSION_AMOUNT,
            max_ev: SCENE_REFERRED_DETAIL_FUSION_MAX_EV,
            applied_ratio: applied as f64 / denom,
            mean_abs_ev: if applied == 0 {
                0.0
            } else {
                sum_abs_ev / applied as f64
            },
            max_abs_ev,
            skipped_negative_ratio: skipped_negative as f64 / denom,
            skipped_nonfinite_ratio: skipped_nonfinite as f64 / denom,
        },
    }
}

fn max_chroma_scale_inside_gamut(rgb: [f64; 3], luminance: f64) -> f64 {
    let mut limit = f64::INFINITY;
    for value in rgb {
        let delta = value - luminance;
        if delta > 1e-12 {
            limit = limit.min((1.0 - luminance) / delta);
        } else if delta < -1e-12 {
            limit = limit.min((0.0 - luminance) / delta);
        }
    }
    if limit.is_finite() {
        limit.max(1.0)
    } else {
        ADAPTIVE_VIBRANCE_MAX_SCALE
    }
}

fn apply_adaptive_vibrance(
    img: &Array3<f64>,
    color_protection: &ToneColorProtection,
) -> (Array3<f64>, AdaptiveVibranceDiagnostics) {
    let (height, width, channels) = img.dim();
    if color_protection.color_trust_state() != "trusted" {
        return (
            img.clone(),
            AdaptiveVibranceDiagnostics::disabled(format!(
                "disabled because tone color trust state is {}",
                color_protection.color_trust_state()
            )),
        );
    }
    if height < ADAPTIVE_VIBRANCE_MIN_DIMENSION || width < ADAPTIVE_VIBRANCE_MIN_DIMENSION {
        return (
            img.clone(),
            AdaptiveVibranceDiagnostics::disabled(
                "image is smaller than the adaptive vibrance minimum dimension",
            ),
        );
    }
    if channels < 3 {
        return (
            img.clone(),
            AdaptiveVibranceDiagnostics::disabled(
                "adaptive vibrance requires at least three channels",
            ),
        );
    }

    let mut out = Array3::<f64>::zeros((height, width, channels));
    let mut applied = 0usize;
    let mut texture_limited = 0usize;
    let mut gamut_limited = 0usize;
    let mut sum_scale = 0.0f64;
    let mut max_applied_scale = 1.0f64;
    let pixel_count = height * width;
    let mut luminance = vec![0.0f32; pixel_count];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            luminance[idx] = linear_luminance(
                img[[y, x, 0]].clamp(0.0, 1.0),
                img[[y, x, 1]].clamp(0.0, 1.0),
                img[[y, x, 2]].clamp(0.0, 1.0),
            )
            .clamp(0.0, 1.0) as f32;
        }
    }
    let blurred_luminance =
        box_blur_luminance(&luminance, height, width, ADAPTIVE_VIBRANCE_TEXTURE_RADIUS);

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let rgb = [
                img[[y, x, 0]].clamp(0.0, 1.0),
                img[[y, x, 1]].clamp(0.0, 1.0),
                img[[y, x, 2]].clamp(0.0, 1.0),
            ];
            let luminance = luminance[idx] as f64;
            let saturation = rgb_saturation(rgb);
            let texture = (luminance - blurred_luminance[idx] as f64).abs();
            let texture_gate = 1.0
                - smoothstep_range(
                    ADAPTIVE_VIBRANCE_TEXTURE_START,
                    ADAPTIVE_VIBRANCE_TEXTURE_END,
                    texture,
                );
            let neutral_gate = smoothstep_range(
                ADAPTIVE_VIBRANCE_NEUTRAL_START,
                ADAPTIVE_VIBRANCE_NEUTRAL_FULL,
                saturation,
            );
            let saturation_gate = 1.0
                - smoothstep_range(
                    ADAPTIVE_VIBRANCE_SATURATION_START,
                    ADAPTIVE_VIBRANCE_SATURATION_END,
                    saturation,
                );
            let luminance_gate = smoothstep_range(
                ADAPTIVE_VIBRANCE_SHADOW_START,
                ADAPTIVE_VIBRANCE_SHADOW_FULL,
                luminance,
            ) * (1.0
                - smoothstep_range(
                    ADAPTIVE_VIBRANCE_HIGHLIGHT_START,
                    ADAPTIVE_VIBRANCE_HIGHLIGHT_END,
                    luminance,
                ));
            let requested_scale = (1.0
                + ADAPTIVE_VIBRANCE_AMOUNT
                    * neutral_gate
                    * saturation_gate
                    * luminance_gate
                    * texture_gate)
                .min(ADAPTIVE_VIBRANCE_MAX_SCALE);
            let gamut_scale = max_chroma_scale_inside_gamut(rgb, luminance);
            let scale = requested_scale.min(gamut_scale);
            if texture_gate < 0.5 {
                texture_limited += 1;
            }
            if requested_scale > gamut_scale + 1e-12 {
                gamut_limited += 1;
            }
            if scale > 1.002 {
                applied += 1;
                sum_scale += scale;
                max_applied_scale = max_applied_scale.max(scale);
            }

            for c in 0..channels {
                if c < 3 {
                    out[[y, x, c]] = (luminance + (rgb[c] - luminance) * scale).clamp(0.0, 1.0);
                } else {
                    out[[y, x, c]] = img[[y, x, c]];
                }
            }
        }
    }

    let denom = height.saturating_mul(width).max(1) as f64;
    (
        out,
        AdaptiveVibranceDiagnostics {
            enabled: true,
            reason: "trusted color path received bounded hue-preserving adaptive vibrance"
                .to_string(),
            amount: ADAPTIVE_VIBRANCE_AMOUNT,
            max_scale: ADAPTIVE_VIBRANCE_MAX_SCALE,
            applied_ratio: applied as f64 / denom,
            mean_scale: if applied == 0 {
                1.0
            } else {
                sum_scale / applied as f64
            },
            max_applied_scale,
            texture_limited_ratio: texture_limited as f64 / denom,
            gamut_limited_ratio: gamut_limited as f64 / denom,
        },
    )
}

fn apply_render_noise_reduction(
    img: &Array3<f64>,
) -> (Array3<f64>, RenderNoiseReductionDiagnostics) {
    let (height, width, channels) = img.dim();
    if channels < 3 {
        return (
            img.clone(),
            RenderNoiseReductionDiagnostics::disabled(
                "render noise reduction requires at least three channels",
            ),
        );
    }
    let required_radius = RENDER_NOISE_REDUCTION_RADIUS.max(RENDER_NOISE_REDUCTION_CHROMA_RADIUS);
    if height < required_radius * 2 + 1 || width < required_radius * 2 + 1 {
        return (
            img.clone(),
            RenderNoiseReductionDiagnostics::disabled(
                "image is smaller than the render noise reduction window",
            ),
        );
    }

    let pixel_count = height * width;
    let mut luminance = vec![0.0f32; pixel_count];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            luminance[idx] = linear_luminance(
                img[[y, x, 0]].clamp(0.0, 1.0),
                img[[y, x, 1]].clamp(0.0, 1.0),
                img[[y, x, 2]].clamp(0.0, 1.0),
            ) as f32;
        }
    }

    let pre_noise = render_grain_diagnostics(img);
    let frame_chroma_grain = pre_noise
        .flat_chroma_residual_p95
        .max(pre_noise.chroma_residual_p95 * 0.65);
    let frame_chroma_damping = smoothstep_range(
        RENDER_NOISE_REDUCTION_FRAME_CHROMA_GRAIN_START,
        RENDER_NOISE_REDUCTION_FRAME_CHROMA_GRAIN_FULL,
        frame_chroma_grain,
    );

    let blurred_luminance =
        box_blur_luminance(&luminance, height, width, RENDER_NOISE_REDUCTION_RADIUS);
    let mut adjusted_luminance = vec![0.0f32; pixel_count];
    let mut chroma_strength = vec![0.0f32; pixel_count];
    let mut chroma_damping = vec![0.0f32; pixel_count];
    let mut applied = 0usize;
    let mut texture_limited = 0usize;
    let mut saturation_limited = 0usize;
    let mut luma_delta_sum = 0.0f64;
    let mut luma_delta_max = 0.0f64;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let lum = luminance[idx] as f64;
            let local_lum = blurred_luminance[idx] as f64;
            let rgb = [
                img[[y, x, 0]].clamp(0.0, 1.0),
                img[[y, x, 1]].clamp(0.0, 1.0),
                img[[y, x, 2]].clamp(0.0, 1.0),
            ];
            let saturation = rgb_saturation(rgb);
            let detail_residual = (lum - local_lum).abs();
            let structure = smoothed_luminance_structure(
                &blurred_luminance,
                height,
                width,
                y,
                x,
                RENDER_NOISE_REDUCTION_RADIUS,
            )
            .unwrap_or(detail_residual);
            let chroma_texture = detail_residual.min(
                structure + detail_residual * RENDER_NOISE_REDUCTION_CHROMA_TEXTURE_RESIDUAL_WEIGHT,
            );
            let luma_noise_texture =
                structure + detail_residual * RENDER_NOISE_REDUCTION_LUMA_TEXTURE_RESIDUAL_WEIGHT;
            let chroma_texture_gate = 1.0
                - smoothstep_range(
                    RENDER_NOISE_REDUCTION_TEXTURE_START,
                    RENDER_NOISE_REDUCTION_TEXTURE_END,
                    chroma_texture,
                );
            let luma_texture_gate = 1.0
                - smoothstep_range(
                    RENDER_NOISE_REDUCTION_TEXTURE_START,
                    RENDER_NOISE_REDUCTION_TEXTURE_END,
                    detail_residual.max(structure),
                );
            let luma_noise_gate = 1.0
                - smoothstep_range(
                    RENDER_NOISE_REDUCTION_TEXTURE_START,
                    RENDER_NOISE_REDUCTION_TEXTURE_END,
                    luma_noise_texture,
                );
            let saturation_gate = 1.0
                - smoothstep_range(
                    RENDER_NOISE_REDUCTION_SATURATION_START,
                    RENDER_NOISE_REDUCTION_SATURATION_END,
                    saturation,
                );
            let shadow_gate = 1.0
                - smoothstep_range(
                    RENDER_NOISE_REDUCTION_SHADOW_START,
                    RENDER_NOISE_REDUCTION_SHADOW_END,
                    lum,
                );
            let effective_saturation_gate = saturation_gate
                + (1.0 - saturation_gate)
                    * RENDER_NOISE_REDUCTION_SHADOW_SATURATION_RELAXATION
                    * shadow_gate;
            let neutral_floor_luma_gate = 1.0
                - smoothstep_range(
                    RENDER_NOISE_REDUCTION_NEUTRAL_FLOOR_HIGHLIGHT_START,
                    RENDER_NOISE_REDUCTION_NEUTRAL_FLOOR_HIGHLIGHT_END,
                    lum,
                );
            let chroma_texture_for_chroma = chroma_texture_gate.max(
                RENDER_NOISE_REDUCTION_NEUTRAL_TEXTURE_FLOOR
                    * effective_saturation_gate
                    * neutral_floor_luma_gate,
            );
            let chroma_gate = chroma_texture_for_chroma
                * (RENDER_NOISE_REDUCTION_SATURATION_GATE_FLOOR
                    + (1.0 - RENDER_NOISE_REDUCTION_SATURATION_GATE_FLOOR)
                        * effective_saturation_gate)
                * (0.70 + 0.30 * shadow_gate);
            let luma_texture_for_luma =
                luma_texture_gate.max(RENDER_NOISE_REDUCTION_LUMA_TEXTURE_FLOOR * luma_noise_gate);
            let luma_gate = luma_texture_for_luma * (0.35 + 0.65 * shadow_gate);
            let chroma = RENDER_NOISE_REDUCTION_CHROMA_AMOUNT * chroma_gate;
            let luma = RENDER_NOISE_REDUCTION_LUMA_AMOUNT * luma_gate;
            let damp = frame_chroma_damping
                * RENDER_NOISE_REDUCTION_CHROMA_DAMP_AMOUNT
                * chroma_texture_for_chroma
                * (0.70 + 0.30 * shadow_gate);
            let new_lum = lum * (1.0 - luma) + local_lum * luma;
            let luma_delta = (new_lum - lum).abs();

            if chroma > 0.02 || luma > 0.01 {
                applied += 1;
            }
            if chroma_texture_gate < 0.5 || luma_texture_gate < 0.5 {
                texture_limited += 1;
            }
            if effective_saturation_gate < 0.5 {
                saturation_limited += 1;
            }
            luma_delta_sum += luma_delta;
            luma_delta_max = luma_delta_max.max(luma_delta);
            adjusted_luminance[idx] = new_lum as f32;
            chroma_strength[idx] = chroma as f32;
            chroma_damping[idx] = damp as f32;
        }
    }

    let mut out = Array3::<f64>::zeros((height, width, channels));
    let mut chroma_delta_sum = 0.0f64;
    let mut chroma_delta_max = 0.0f64;
    let mut residual = vec![0.0f32; pixel_count];
    for channel in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                residual[idx] =
                    (img[[y, x, channel]].clamp(0.0, 1.0) - luminance[idx] as f64) as f32;
            }
        }
        let blurred_residual = box_blur_luminance(
            &residual,
            height,
            width,
            RENDER_NOISE_REDUCTION_CHROMA_RADIUS,
        );
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let strength = chroma_strength[idx] as f64;
                let base_residual = residual[idx] as f64;
                let smoothed_residual =
                    base_residual * (1.0 - strength) + blurred_residual[idx] as f64 * strength;
                let damped_residual =
                    smoothed_residual * (1.0 - chroma_damping[idx] as f64).clamp(0.0, 1.0);
                let chroma_delta = (damped_residual - base_residual).abs();
                chroma_delta_sum += chroma_delta;
                chroma_delta_max = chroma_delta_max.max(chroma_delta);
                out[[y, x, channel]] =
                    (adjusted_luminance[idx] as f64 + damped_residual).clamp(0.0, 1.0);
            }
        }
    }
    for channel in 3..channels {
        for y in 0..height {
            for x in 0..width {
                out[[y, x, channel]] = img[[y, x, channel]];
            }
        }
    }

    let denom = pixel_count.max(1) as f64;
    let chroma_denom = (pixel_count.max(1) * 3) as f64;
    (
        out,
        RenderNoiseReductionDiagnostics {
            enabled: true,
            reason: "edge-aware final render denoise smoothed chroma residuals and mild shadow luminance noise while preserving local luminance structure".to_string(),
            radius: required_radius,
            chroma_amount: RENDER_NOISE_REDUCTION_CHROMA_AMOUNT,
            luma_amount: RENDER_NOISE_REDUCTION_LUMA_AMOUNT,
            applied_ratio: applied as f64 / denom,
            texture_limited_ratio: texture_limited as f64 / denom,
            saturation_limited_ratio: saturation_limited as f64 / denom,
            mean_abs_chroma_delta: chroma_delta_sum / chroma_denom,
            max_abs_chroma_delta: chroma_delta_max,
            mean_abs_luma_delta: luma_delta_sum / denom,
            max_abs_luma_delta: luma_delta_max,
        },
    )
}

fn apply_local_luminance_detail(
    img: &Array3<f64>,
) -> (Array3<f64>, LocalLuminanceDetailDiagnostics) {
    let (height, width, channels) = img.dim();
    let base_diagnostics = LocalLuminanceDetailDiagnostics {
        enabled: false,
        radius: LOCAL_LUMINANCE_DETAIL_RADIUS,
        amount: LOCAL_LUMINANCE_DETAIL_AMOUNT,
        max_ev: LOCAL_LUMINANCE_DETAIL_MAX_EV,
        applied_ratio: 0.0,
        mean_abs_ev: 0.0,
        max_abs_ev: 0.0,
        headroom_limited_ratio: 0.0,
        clip_limited_ratio: 0.0,
    };
    if height < LOCAL_LUMINANCE_DETAIL_RADIUS * 2 + 1
        || width < LOCAL_LUMINANCE_DETAIL_RADIUS * 2 + 1
    {
        return (img.clone(), base_diagnostics);
    }

    let pixel_count = height * width;
    let mut luminance = vec![0.0f32; pixel_count];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            luminance[idx] = linear_luminance(
                img[[y, x, 0]].clamp(0.0, 1.0),
                img[[y, x, 1]].clamp(0.0, 1.0),
                img[[y, x, 2]].clamp(0.0, 1.0),
            ) as f32;
        }
    }

    let blurred = box_blur_luminance(&luminance, height, width, LOCAL_LUMINANCE_DETAIL_RADIUS);
    let mut out = Array3::<f64>::zeros((height, width, channels));
    let mut applied = 0usize;
    let mut headroom_limited = 0usize;
    let mut clip_limited = 0usize;
    let mut sum_abs_ev = 0.0f64;
    let mut max_abs_ev = 0.0f64;
    let eps = 1e-5;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let lum = luminance[idx] as f64;
            let local_mean = blurred[idx] as f64;
            let shadow_gate = smoothstep_range(
                LOCAL_LUMINANCE_DETAIL_SHADOW_START,
                LOCAL_LUMINANCE_DETAIL_SHADOW_FULL,
                lum,
            );
            let highlight_gate = 1.0
                - smoothstep_range(
                    LOCAL_LUMINANCE_DETAIL_HIGHLIGHT_START,
                    LOCAL_LUMINANCE_DETAIL_HIGHLIGHT_END,
                    lum,
                );
            let structure = smoothed_luminance_structure(
                &blurred,
                height,
                width,
                y,
                x,
                LOCAL_LUMINANCE_DETAIL_RADIUS,
            )
            .unwrap_or(0.0);
            let structure_gate = smoothstep_range(
                LOCAL_LUMINANCE_DETAIL_STRUCTURE_START,
                LOCAL_LUMINANCE_DETAIL_STRUCTURE_FULL,
                structure,
            );
            let strength =
                LOCAL_LUMINANCE_DETAIL_AMOUNT * shadow_gate * highlight_gate * structure_gate;
            let detail_ev = ((lum + eps) / (local_mean + eps)).log2();
            let mut requested_ev = (detail_ev * strength).clamp(
                -LOCAL_LUMINANCE_DETAIL_MAX_EV,
                LOCAL_LUMINANCE_DETAIL_MAX_EV,
            );
            let max_channel = (0..channels.min(3))
                .map(|c| img[[y, x, c]].clamp(0.0, 1.0))
                .fold(0.0f64, f64::max);
            if requested_ev > 0.0 && max_channel > 1e-9 {
                let available_ev = (1.0 / max_channel).log2().max(0.0);
                let headroom_gate = smoothstep_range(
                    LOCAL_LUMINANCE_DETAIL_POSITIVE_HEADROOM_START_EV,
                    LOCAL_LUMINANCE_DETAIL_POSITIVE_HEADROOM_FULL_EV,
                    available_ev,
                );
                let headroom_limited_ev = requested_ev * headroom_gate;
                let adjusted_ev = headroom_limited_ev.min(available_ev);
                if adjusted_ev + 1e-12 < requested_ev {
                    headroom_limited += 1;
                }
                requested_ev = adjusted_ev;
            }
            let mut scale = 2.0f64.powf(requested_ev);
            if scale > 1.0 && max_channel > 1e-9 {
                let capped = scale.min(1.0 / max_channel);
                if capped + 1e-12 < scale {
                    clip_limited += 1;
                }
                scale = capped;
            }
            let applied_ev = scale.log2();
            if applied_ev.abs() > 0.002 {
                applied += 1;
                let abs_ev = applied_ev.abs();
                sum_abs_ev += abs_ev;
                max_abs_ev = max_abs_ev.max(abs_ev);
            }
            for c in 0..channels {
                out[[y, x, c]] = (img[[y, x, c]].clamp(0.0, 1.0) * scale).clamp(0.0, 1.0);
            }
        }
    }

    let denom = pixel_count.max(1) as f64;
    (
        out,
        LocalLuminanceDetailDiagnostics {
            enabled: true,
            radius: LOCAL_LUMINANCE_DETAIL_RADIUS,
            amount: LOCAL_LUMINANCE_DETAIL_AMOUNT,
            max_ev: LOCAL_LUMINANCE_DETAIL_MAX_EV,
            applied_ratio: applied as f64 / denom,
            mean_abs_ev: if applied == 0 {
                0.0
            } else {
                sum_abs_ev / applied as f64
            },
            max_abs_ev,
            headroom_limited_ratio: headroom_limited as f64 / denom,
            clip_limited_ratio: clip_limited as f64 / denom,
        },
    )
}

fn compress_highlight_neutral_chroma(rgb: [f64; 3], mapped_lum: f64) -> ([f64; 3], bool) {
    if mapped_lum < HIGHLIGHT_NEUTRAL_CHROMA_START_LUMINANCE {
        return (rgb, false);
    }

    if rgb_saturation(rgb) > HIGHLIGHT_NEUTRAL_CHROMA_MAX_SATURATION {
        return (rgb, false);
    }

    let span = (HIGHLIGHT_NEUTRAL_CHROMA_FULL_LUMINANCE - HIGHLIGHT_NEUTRAL_CHROMA_START_LUMINANCE)
        .max(1e-12);
    let t = (mapped_lum - HIGHLIGHT_NEUTRAL_CHROMA_START_LUMINANCE) / span;
    let strength = smoothstep01(t);
    if strength <= 1e-12 {
        return (rgb, false);
    }

    let chroma_scale = 1.0 - strength * (1.0 - HIGHLIGHT_NEUTRAL_CHROMA_MIN_SCALE);
    let neutral = mapped_lum.clamp(0.0, 1.0);
    let compressed = std::array::from_fn(|c| neutral + (rgb[c] - neutral) * chroma_scale);
    (compressed, chroma_scale < 1.0 - 1e-12)
}

fn apply_highlight_neutral_chroma_cleanup(img: &Array3<f64>) -> (Array3<f64>, usize) {
    let (_, width, channels) = img.dim();
    if channels < 3 {
        return (img.clone(), 0);
    }

    let mut out = img.clone();
    let compressed_pixels = AtomicU64::new(0);
    out.axis_chunks_iter_mut(Axis(0), 512)
        .into_par_iter()
        .for_each(|mut chunk| {
            let chunk_h = chunk.dim().0;
            for y in 0..chunk_h {
                for x in 0..width {
                    let rgb = [
                        chunk[[y, x, 0]].clamp(0.0, 1.0),
                        chunk[[y, x, 1]].clamp(0.0, 1.0),
                        chunk[[y, x, 2]].clamp(0.0, 1.0),
                    ];
                    let luminance = linear_luminance(rgb[0], rgb[1], rgb[2]);
                    let (limited, changed) = compress_highlight_neutral_chroma(rgb, luminance);
                    if changed {
                        compressed_pixels.fetch_add(1, Ordering::Relaxed);
                        for ch in 0..3 {
                            chunk[[y, x, ch]] = limited[ch].clamp(0.0, 1.0);
                        }
                    }
                }
            }
        });

    (out, compressed_pixels.load(Ordering::Relaxed) as usize)
}

fn compress_shadow_chroma(rgb: [f64; 3], mapped_lum: f64) -> ([f64; 3], bool) {
    if mapped_lum >= SHADOW_CHROMA_START_LUMINANCE {
        return (rgb, false);
    }

    let span = (SHADOW_CHROMA_START_LUMINANCE - SHADOW_CHROMA_FULL_LUMINANCE).max(1e-12);
    let t = (mapped_lum - SHADOW_CHROMA_FULL_LUMINANCE) / span;
    let strength = (1.0 - smoothstep01(t)).sqrt();
    if strength <= 1e-12 {
        return (rgb, false);
    }

    let chroma_scale = 1.0 - strength * (1.0 - SHADOW_CHROMA_MIN_SCALE);
    let neutral = mapped_lum.clamp(0.0, 1.0);
    let compressed = std::array::from_fn(|c| neutral + (rgb[c] - neutral) * chroma_scale);
    (compressed, chroma_scale < 1.0 - 1e-12)
}

fn limit_midtone_neutral_chroma(rgb: [f64; 3], luminance: f64) -> ([f64; 3], bool) {
    if !(MIDTONE_NEUTRAL_CHROMA_START_LUMINANCE..MIDTONE_NEUTRAL_CHROMA_END_LUMINANCE)
        .contains(&luminance)
    {
        return (rgb, false);
    }

    if rgb_saturation(rgb) > MIDTONE_NEUTRAL_CHROMA_MAX_SATURATION {
        return (rgb, false);
    }

    let lower_gate = smoothstep_range(
        MIDTONE_NEUTRAL_CHROMA_START_LUMINANCE,
        MIDTONE_NEUTRAL_CHROMA_FULL_LOW_LUMINANCE,
        luminance,
    );
    let upper_gate = 1.0
        - smoothstep_range(
            MIDTONE_NEUTRAL_CHROMA_FULL_HIGH_LUMINANCE,
            MIDTONE_NEUTRAL_CHROMA_END_LUMINANCE,
            luminance,
        );
    let strength = lower_gate * upper_gate;
    if strength <= 1e-12 {
        return (rgb, false);
    }

    let neutral = luminance.clamp(0.0, 1.0);
    let chroma_scale = 1.0 - strength * (1.0 - MIDTONE_NEUTRAL_CHROMA_MIN_SCALE);
    let limited = std::array::from_fn(|c| neutral + (rgb[c] - neutral) * chroma_scale);
    (limited, chroma_scale < 1.0 - 1e-12)
}

fn apply_midtone_neutral_chroma_cleanup(img: &Array3<f64>) -> (Array3<f64>, usize) {
    let (_, width, channels) = img.dim();
    if channels < 3 {
        return (img.clone(), 0);
    }

    let mut out = img.clone();
    let compressed_pixels = AtomicU64::new(0);
    out.axis_chunks_iter_mut(Axis(0), 512)
        .into_par_iter()
        .for_each(|mut chunk| {
            let chunk_h = chunk.dim().0;
            for y in 0..chunk_h {
                for x in 0..width {
                    let rgb = [
                        chunk[[y, x, 0]].clamp(0.0, 1.0),
                        chunk[[y, x, 1]].clamp(0.0, 1.0),
                        chunk[[y, x, 2]].clamp(0.0, 1.0),
                    ];
                    let luminance = linear_luminance(rgb[0], rgb[1], rgb[2]);
                    let (limited, changed) = limit_midtone_neutral_chroma(rgb, luminance);
                    if changed {
                        compressed_pixels.fetch_add(1, Ordering::Relaxed);
                        for ch in 0..3 {
                            chunk[[y, x, ch]] = limited[ch].clamp(0.0, 1.0);
                        }
                    }
                }
            }
        });

    (out, compressed_pixels.load(Ordering::Relaxed) as usize)
}

fn shadow_saturation_guard_max(luminance: f64) -> f64 {
    let span = (SHADOW_CHROMA_START_LUMINANCE - SHADOW_CHROMA_FULL_LUMINANCE).max(1e-12);
    let t = ((luminance - SHADOW_CHROMA_FULL_LUMINANCE) / span).clamp(0.0, 1.0);
    SHADOW_SATURATION_GUARD_DEEP_MAX
        + (SHADOW_SATURATION_GUARD_UPPER_MAX - SHADOW_SATURATION_GUARD_DEEP_MAX) * smoothstep01(t)
}

fn limit_shadow_saturation(rgb: [f64; 3], luminance: f64) -> ([f64; 3], bool) {
    if luminance >= SHADOW_CHROMA_START_LUMINANCE {
        return (rgb, false);
    }

    let max_saturation = shadow_saturation_guard_max(luminance);
    if rgb_saturation(rgb) <= max_saturation {
        return (rgb, false);
    }

    let neutral = luminance.clamp(0.0, 1.0);
    let mut low = 0.0f64;
    let mut high = 1.0f64;
    for _ in 0..24 {
        let mid = (low + high) * 0.5;
        let candidate = std::array::from_fn(|c| neutral + (rgb[c] - neutral) * mid);
        if rgb_saturation(candidate) <= max_saturation {
            low = mid;
        } else {
            high = mid;
        }
    }

    let limited = std::array::from_fn(|c| neutral + (rgb[c] - neutral) * low);
    (limited, low < 1.0 - 1e-12)
}

fn apply_shadow_saturation_guard(img: &Array3<f64>) -> (Array3<f64>, usize) {
    let (_, width, channels) = img.dim();
    if channels < 3 {
        return (img.clone(), 0);
    }

    let mut out = img.clone();
    let limited_pixels = AtomicU64::new(0);
    out.axis_chunks_iter_mut(Axis(0), 512)
        .into_par_iter()
        .for_each(|mut chunk| {
            let chunk_h = chunk.dim().0;
            for y in 0..chunk_h {
                for x in 0..width {
                    let rgb = [
                        chunk[[y, x, 0]].clamp(0.0, 1.0),
                        chunk[[y, x, 1]].clamp(0.0, 1.0),
                        chunk[[y, x, 2]].clamp(0.0, 1.0),
                    ];
                    let luminance = linear_luminance(rgb[0], rgb[1], rgb[2]);
                    let (limited, changed) = limit_shadow_saturation(rgb, luminance);
                    if changed {
                        limited_pixels.fetch_add(1, Ordering::Relaxed);
                        for ch in 0..3 {
                            chunk[[y, x, ch]] = limited[ch].clamp(0.0, 1.0);
                        }
                    }
                }
            }
        });

    (out, limited_pixels.load(Ordering::Relaxed) as usize)
}

fn histogram_bin(value: f64, range_max: f64) -> usize {
    let normalized = if range_max > 1e-12 {
        value.max(0.0) / range_max
    } else {
        0.0
    };
    ((normalized * NUM_BINS as f64) as usize).min(NUM_BINS - 1)
}

fn histogram_percentiles(histogram: &[u64], total_pixels: u64, range_max: f64) -> [f64; 3] {
    let targets = [0.05, 0.50, 0.95];
    let mut values = [0.0f64; 3];
    let mut found = [false; 3];
    let mut cumulative = 0u64;
    let range_max = range_max.max(1e-12);

    for (i, &count) in histogram.iter().enumerate() {
        cumulative += count;
        let value = ((i as f64 + 0.5) / NUM_BINS as f64) * range_max;
        for (idx, percentile) in targets.iter().enumerate() {
            if !found[idx] && cumulative >= (total_pixels as f64 * percentile) as u64 {
                values[idx] = value;
                found[idx] = true;
            }
        }
    }

    values
}

fn tone_input_percentiles(img: &Array3<f64>) -> (u64, [f64; 3], [f64; 3]) {
    let shape = img.shape();
    let h = shape[0];
    let w = shape[1];

    let mut total_pixels = 0u64;
    let mut linear_histogram_max = 1.0f64;
    for y in 0..h {
        for x in 0..w {
            let linear_lum = linear_luminance(img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]]);
            if linear_lum.is_finite() {
                linear_histogram_max = linear_histogram_max.max(linear_lum);
            }
            total_pixels += 1;
        }
    }
    let perceptual_histogram_max = encode_perceptual_luminance(linear_histogram_max).max(1.0);

    let mut linear_histogram = vec![0u64; NUM_BINS];
    let mut perceptual_histogram = vec![0u64; NUM_BINS];

    for y in 0..h {
        for x in 0..w {
            let r = img[[y, x, 0]];
            let g = img[[y, x, 1]];
            let b = img[[y, x, 2]];
            let linear_lum = linear_luminance(r, g, b);
            let perceptual_lum = encode_perceptual_luminance(linear_lum);
            let linear_bin = histogram_bin(linear_lum, linear_histogram_max);
            let perceptual_bin = histogram_bin(perceptual_lum, perceptual_histogram_max);
            linear_histogram[linear_bin] += 1;
            perceptual_histogram[perceptual_bin] += 1;
        }
    }

    if total_pixels == 0 {
        return (0, [0.0; 3], [0.0; 3]);
    }

    (
        total_pixels,
        histogram_percentiles(&linear_histogram, total_pixels, linear_histogram_max),
        histogram_percentiles(
            &perceptual_histogram,
            total_pixels,
            perceptual_histogram_max,
        ),
    )
}

fn solve_midpoint_for_target(
    input_value: f64,
    target_value: f64,
    slope: f64,
    toe_lift: f64,
    shoulder_max: f64,
) -> f64 {
    solve_midpoint_for_target_in_bounds(
        input_value,
        target_value,
        slope,
        toe_lift,
        shoulder_max,
        POSITIVE_SCAN_TARGET_MEDIAN_MIN,
        POSITIVE_SCAN_LINEAR_FIT_MAX_MIDPOINT,
    )
}

fn solve_midpoint_for_target_in_bounds(
    input_value: f64,
    target_value: f64,
    slope: f64,
    toe_lift: f64,
    shoulder_max: f64,
    min_midpoint: f64,
    max_midpoint: f64,
) -> f64 {
    let finite_input = if input_value.is_nan() {
        1e-6
    } else {
        input_value
    };
    let curve_input = finite_input.clamp(1e-6, EXTENDED_HIGHLIGHT_KNEE);
    let z =
        ((target_value - toe_lift) / (shoulder_max - toe_lift).max(1e-12)).clamp(1e-6, 1.0 - 1e-6);
    let input_power = curve_input.powf(slope);
    let denominator = z - input_power;
    if denominator <= 1e-12 {
        return (curve_input * 1.35).clamp(min_midpoint, max_midpoint);
    }

    let midpoint_power = input_power * (1.0 - z) / denominator;
    midpoint_power
        .max(1e-300)
        .powf(1.0 / slope)
        .clamp(min_midpoint, max_midpoint)
}

fn is_negative_low_range(input_linear_percentiles: [f64; 3]) -> bool {
    let p05 = input_linear_percentiles[0];
    let p95 = input_linear_percentiles[2];
    let dynamic_range = p95 - p05;
    p95 <= NEGATIVE_LOW_RANGE_LINEAR_MAX_P95
        && dynamic_range <= NEGATIVE_LOW_RANGE_LINEAR_DYNAMIC_RANGE
}

fn negative_log_domain_target_linear_median(input_linear_percentiles: [f64; 3]) -> Option<f64> {
    let p50 = input_linear_percentiles[1];
    if is_negative_low_range(input_linear_percentiles) {
        Some(NEGATIVE_LOW_RANGE_TARGET_LINEAR_MEDIAN)
    } else if p50 <= NEGATIVE_SHADOW_TARGET_LINEAR_MEDIAN_MAX {
        Some(NEGATIVE_SHADOW_TARGET_LINEAR_MEDIAN)
    } else {
        None
    }
}

fn negative_mapped_linear_p95_for_target(
    input_perceptual_percentiles: [f64; 3],
    target_linear_median: f64,
    slope: f64,
    toe_lift: f64,
    shoulder_max: f64,
) -> f64 {
    let midpoint = solve_midpoint_for_target_in_bounds(
        input_perceptual_percentiles[1],
        encode_perceptual_luminance(target_linear_median),
        slope,
        toe_lift,
        shoulder_max,
        PERCEPTUAL_FIT_MIN_MIDPOINT,
        PERCEPTUAL_FIT_MAX_MIDPOINT,
    );
    decode_perceptual_luminance(apply_tone_curve(
        input_perceptual_percentiles[2],
        &ToneCurveParams {
            domain: ToneFitDomain::Log2CompressedLuminance,
            midpoint,
            slope,
            toe_lift,
            shoulder_max,
        },
    ))
}

fn negative_low_range_contrast_slope(
    input_perceptual_percentiles: [f64; 3],
    target_linear_median: f64,
    base_slope: f64,
    toe_lift: f64,
    shoulder_max: f64,
) -> f64 {
    let base_mapped_p95 = negative_mapped_linear_p95_for_target(
        input_perceptual_percentiles,
        target_linear_median,
        base_slope,
        toe_lift,
        shoulder_max,
    );
    if base_mapped_p95 >= NEGATIVE_LOW_RANGE_MIN_MAPPED_P95 {
        return base_slope;
    }

    let high_slope = NEGATIVE_LOW_RANGE_MAX_SLOPE.max(base_slope);
    let high_mapped_p95 = negative_mapped_linear_p95_for_target(
        input_perceptual_percentiles,
        target_linear_median,
        high_slope,
        toe_lift,
        shoulder_max,
    );
    if high_mapped_p95 < base_mapped_p95 + NEGATIVE_LOW_RANGE_MIN_MAPPED_P95_GAIN {
        return base_slope;
    }
    if high_mapped_p95 < NEGATIVE_LOW_RANGE_MIN_MAPPED_P95 {
        return high_slope;
    }

    let mut low = base_slope;
    let mut high = high_slope;
    for _ in 0..24 {
        let candidate = (low + high) * 0.5;
        if negative_mapped_linear_p95_for_target(
            input_perceptual_percentiles,
            target_linear_median,
            candidate,
            toe_lift,
            shoulder_max,
        ) >= NEGATIVE_LOW_RANGE_MIN_MAPPED_P95
        {
            high = candidate;
        } else {
            low = candidate;
        }
    }
    high
}

fn solve_negative_log_midpoint_for_target(
    input_perceptual_percentiles: [f64; 3],
    target_linear_median: f64,
    slope: f64,
    toe_lift: f64,
    shoulder_max: f64,
) -> f64 {
    let target_perceptual_median = encode_perceptual_luminance(target_linear_median);
    let midpoint = solve_midpoint_for_target_in_bounds(
        input_perceptual_percentiles[1],
        target_perceptual_median,
        slope,
        toe_lift,
        shoulder_max,
        PERCEPTUAL_FIT_MIN_MIDPOINT,
        PERCEPTUAL_FIT_MAX_MIDPOINT,
    );
    let p95_test_params = ToneCurveParams {
        domain: ToneFitDomain::Log2CompressedLuminance,
        midpoint,
        slope,
        toe_lift,
        shoulder_max,
    };
    let mapped_p95 = decode_perceptual_luminance(apply_tone_curve(
        input_perceptual_percentiles[2],
        &p95_test_params,
    ));
    if mapped_p95 <= NEGATIVE_SHADOW_TARGET_MAX_MAPPED_P95 {
        return midpoint;
    }

    let current_midpoint = input_perceptual_percentiles[1]
        .clamp(PERCEPTUAL_FIT_MIN_MIDPOINT, PERCEPTUAL_FIT_MAX_MIDPOINT);
    let mut low_target = decode_perceptual_luminance(apply_tone_curve(
        input_perceptual_percentiles[1],
        &ToneCurveParams {
            domain: ToneFitDomain::Log2CompressedLuminance,
            midpoint: current_midpoint,
            slope,
            toe_lift,
            shoulder_max,
        },
    ));
    let mut high_target = target_linear_median;
    let mut best_midpoint = current_midpoint;
    for _ in 0..24 {
        let candidate_target = (low_target + high_target) * 0.5;
        let candidate_midpoint = solve_midpoint_for_target_in_bounds(
            input_perceptual_percentiles[1],
            encode_perceptual_luminance(candidate_target),
            slope,
            toe_lift,
            shoulder_max,
            PERCEPTUAL_FIT_MIN_MIDPOINT,
            PERCEPTUAL_FIT_MAX_MIDPOINT,
        );
        let candidate_params = ToneCurveParams {
            domain: ToneFitDomain::Log2CompressedLuminance,
            midpoint: candidate_midpoint,
            slope,
            toe_lift,
            shoulder_max,
        };
        let candidate_p95 = decode_perceptual_luminance(apply_tone_curve(
            input_perceptual_percentiles[2],
            &candidate_params,
        ));
        if candidate_p95 <= NEGATIVE_SHADOW_TARGET_MAX_MAPPED_P95 {
            best_midpoint = candidate_midpoint;
            low_target = candidate_target;
        } else {
            high_target = candidate_target;
        }
    }
    best_midpoint
}

fn positive_scan_target_median(input_median: f64) -> f64 {
    (input_median * POSITIVE_SCAN_TARGET_MEDIAN_INPUT_WEIGHT + POSITIVE_SCAN_TARGET_MEDIAN_OFFSET)
        .clamp(
            POSITIVE_SCAN_TARGET_MEDIAN_MIN,
            POSITIVE_SCAN_TARGET_MEDIAN_MAX,
        )
}

pub fn positive_scan_auto_exposure_ev(img: &Array3<f64>) -> f64 {
    let (_, input_linear_percentiles, _) = tone_input_percentiles(img);
    let p50 = input_linear_percentiles[1].max(1e-9);
    let p95 = input_linear_percentiles[2].max(1e-9);
    let mut exposure_scale = 1.0f64;
    if p95 > POSITIVE_SCAN_AUTO_EXPOSURE_HIGHLIGHT_TARGET {
        exposure_scale = exposure_scale.min(POSITIVE_SCAN_AUTO_EXPOSURE_HIGHLIGHT_TARGET / p95);
    }
    if p50 > POSITIVE_SCAN_AUTO_EXPOSURE_MIDTONE_CEILING {
        exposure_scale = exposure_scale.min(POSITIVE_SCAN_AUTO_EXPOSURE_MIDTONE_CEILING / p50);
    }

    exposure_scale = exposure_scale.clamp(2.0f64.powf(POSITIVE_SCAN_AUTO_EXPOSURE_MIN_EV), 1.0);
    exposure_scale.log2().min(0.0)
}

/// Fit tone curve parameters from an image by analyzing a perceptually compressed
/// luminance histogram derived from linear ProPhoto RGB.
pub fn fit_tone_params_with_diagnostics(img: &Array3<f64>) -> ToneFitResult {
    let (total_pixels, input_linear_percentiles, input_perceptual_percentiles) =
        tone_input_percentiles(img);

    if total_pixels == 0 {
        return ToneFitResult {
            params: ToneCurveParams::default(),
            diagnostics: ToneCurveDiagnostics {
                fit_domain: ToneFitDomain::Log2CompressedLuminance.as_str(),
                perceptual_luminance_gain: PERCEPTUAL_LUMINANCE_GAIN,
                input_linear_percentiles: [0.0; 3],
                input_perceptual_percentiles: [0.0; 3],
                mapped_linear_percentiles: [0.0; 3],
                mapped_perceptual_percentiles: [0.0; 3],
            },
        };
    }
    let domain = if input_linear_percentiles[1] >= LINEAR_FIT_MEDIAN_THRESHOLD {
        ToneFitDomain::LinearLuminance
    } else {
        ToneFitDomain::Log2CompressedLuminance
    };

    let (midpoint, slope) = match domain {
        ToneFitDomain::LinearLuminance => {
            let dynamic_range = input_linear_percentiles[2] - input_linear_percentiles[0];
            let slope = if dynamic_range > 0.0 {
                (LINEAR_FIT_SLOPE_SCALE / dynamic_range).clamp(2.0, 6.0)
            } else {
                5.0
            };
            (
                input_linear_percentiles[1].clamp(LINEAR_FIT_MIN_MIDPOINT, LINEAR_FIT_MAX_MIDPOINT),
                slope,
            )
        }
        ToneFitDomain::Log2CompressedLuminance => {
            let dynamic_range = input_perceptual_percentiles[2] - input_perceptual_percentiles[0];
            let base_slope = if dynamic_range > 0.0 {
                (PERCEPTUAL_FIT_SLOPE_SCALE / dynamic_range).clamp(2.0, PERCEPTUAL_FIT_MAX_SLOPE)
            } else {
                5.0
            };
            let target_median = negative_log_domain_target_linear_median(input_linear_percentiles);
            let slope = if let Some(target_median) = target_median {
                if is_negative_low_range(input_linear_percentiles) {
                    negative_low_range_contrast_slope(
                        input_perceptual_percentiles,
                        target_median,
                        base_slope,
                        0.005,
                        0.995,
                    )
                } else {
                    base_slope.min(NEGATIVE_SHADOW_MAX_SLOPE)
                }
            } else {
                base_slope
            };
            let midpoint = if let Some(target_median) = target_median {
                solve_negative_log_midpoint_for_target(
                    input_perceptual_percentiles,
                    target_median,
                    slope,
                    0.005,
                    0.995,
                )
            } else {
                input_perceptual_percentiles[1]
                    .clamp(PERCEPTUAL_FIT_MIN_MIDPOINT, PERCEPTUAL_FIT_MAX_MIDPOINT)
            };
            (midpoint, slope)
        }
    };

    let params = ToneCurveParams {
        domain,
        midpoint,
        slope,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let (mapped_linear_percentiles, mapped_perceptual_percentiles) = match domain {
        ToneFitDomain::LinearLuminance => {
            let linear =
                std::array::from_fn(|idx| apply_tone_curve(input_linear_percentiles[idx], &params));
            let perceptual = std::array::from_fn(|idx| encode_perceptual_luminance(linear[idx]));
            (linear, perceptual)
        }
        ToneFitDomain::Log2CompressedLuminance => {
            let perceptual = std::array::from_fn(|idx| {
                apply_tone_curve(input_perceptual_percentiles[idx], &params)
            });
            let linear = std::array::from_fn(|idx| decode_perceptual_luminance(perceptual[idx]));
            (linear, perceptual)
        }
    };

    ToneFitResult {
        params,
        diagnostics: ToneCurveDiagnostics {
            fit_domain: domain.as_str(),
            perceptual_luminance_gain: PERCEPTUAL_LUMINANCE_GAIN,
            input_linear_percentiles,
            input_perceptual_percentiles,
            mapped_linear_percentiles,
            mapped_perceptual_percentiles,
        },
    }
}

/// Fit a conservative display tone curve for already-positive RGB scans.
///
/// This keeps positive scans out of the negative-film median-normalization path:
/// the fitted curve preserves the scan's luminance ordering, aims midtones below
/// the generic negative workflow's 0.5-ish placement, and uses a slightly lower
/// shoulder so near-white scan values retain display headroom.
pub fn fit_positive_scan_tone_params_with_diagnostics(img: &Array3<f64>) -> ToneFitResult {
    let (total_pixels, input_linear_percentiles, input_perceptual_percentiles) =
        tone_input_percentiles(img);

    if total_pixels == 0 {
        let params = ToneCurveParams {
            domain: ToneFitDomain::LinearLuminance,
            midpoint: 0.5,
            slope: POSITIVE_SCAN_LINEAR_FIT_MIN_SLOPE,
            toe_lift: POSITIVE_SCAN_TOE_LIFT,
            shoulder_max: POSITIVE_SCAN_SHOULDER_MAX,
        };
        return ToneFitResult {
            params,
            diagnostics: ToneCurveDiagnostics {
                fit_domain: ToneFitDomain::LinearLuminance.as_str(),
                perceptual_luminance_gain: PERCEPTUAL_LUMINANCE_GAIN,
                input_linear_percentiles: [0.0; 3],
                input_perceptual_percentiles: [0.0; 3],
                mapped_linear_percentiles: [0.0; 3],
                mapped_perceptual_percentiles: [0.0; 3],
            },
        };
    }

    let dynamic_range = (input_linear_percentiles[2] - input_linear_percentiles[0]).max(1e-6);
    let slope = (POSITIVE_SCAN_LINEAR_FIT_SLOPE_SCALE / dynamic_range).clamp(
        POSITIVE_SCAN_LINEAR_FIT_MIN_SLOPE,
        POSITIVE_SCAN_LINEAR_FIT_MAX_SLOPE,
    );
    let target_median = positive_scan_target_median(input_linear_percentiles[1]);
    let midpoint = solve_midpoint_for_target(
        input_linear_percentiles[1],
        target_median,
        slope,
        POSITIVE_SCAN_TOE_LIFT,
        POSITIVE_SCAN_SHOULDER_MAX,
    );

    let params = ToneCurveParams {
        domain: ToneFitDomain::LinearLuminance,
        midpoint,
        slope,
        toe_lift: POSITIVE_SCAN_TOE_LIFT,
        shoulder_max: POSITIVE_SCAN_SHOULDER_MAX,
    };
    let mapped_linear_percentiles =
        std::array::from_fn(|idx| apply_tone_curve(input_linear_percentiles[idx], &params));
    let mapped_perceptual_percentiles =
        std::array::from_fn(|idx| encode_perceptual_luminance(mapped_linear_percentiles[idx]));

    ToneFitResult {
        params,
        diagnostics: ToneCurveDiagnostics {
            fit_domain: ToneFitDomain::LinearLuminance.as_str(),
            perceptual_luminance_gain: PERCEPTUAL_LUMINANCE_GAIN,
            input_linear_percentiles,
            input_perceptual_percentiles,
            mapped_linear_percentiles,
            mapped_perceptual_percentiles,
        },
    }
}

pub fn fit_tone_params(img: &Array3<f64>) -> ToneCurveParams {
    fit_tone_params_with_diagnostics(img).params
}

/// Apply tone mapping to an image using automatically fitted parameters.
pub fn apply_tonemap(img: &Array3<f64>) -> Array3<f64> {
    let fit = fit_tone_params_with_diagnostics(img);
    apply_tonemap_with_params(img, &fit.params)
}

/// Apply tone mapping to an image with explicit parameters.
pub fn apply_tonemap_with_params(img: &Array3<f64>, params: &ToneCurveParams) -> Array3<f64> {
    apply_tonemap_with_params_and_diagnostics(img, params).image
}

pub fn apply_tonemap_with_params_and_color_protection(
    img: &Array3<f64>,
    params: &ToneCurveParams,
    color_protection: &ToneColorProtection,
) -> Array3<f64> {
    apply_tonemap_with_params_and_color_protection_diagnostics(img, params, color_protection).image
}

pub fn apply_tonemap_with_params_and_diagnostics(
    img: &Array3<f64>,
    params: &ToneCurveParams,
) -> TonemapApplyResult {
    apply_tonemap_with_params_and_color_protection_diagnostics(
        img,
        params,
        &ToneColorProtection::default(),
    )
}

pub fn apply_tonemap_with_params_and_color_protection_diagnostics(
    img: &Array3<f64>,
    params: &ToneCurveParams,
    color_protection: &ToneColorProtection,
) -> TonemapApplyResult {
    apply_tonemap_with_params_color_protection_and_style_diagnostics(
        img,
        params,
        color_protection,
        RenderStyle::ModernClean,
    )
}

pub fn apply_tonemap_with_params_color_protection_and_style_diagnostics(
    img: &Array3<f64>,
    params: &ToneCurveParams,
    color_protection: &ToneColorProtection,
    render_style: RenderStyle,
) -> TonemapApplyResult {
    let shape = img.shape();
    let h = shape[0];
    let w = shape[1];
    let c = shape[2];
    let processing_config = render_style.processing_config();

    let mut result = Array3::<f64>::zeros((h, w, c));
    let compressed_pixels = AtomicU64::new(0);
    let highlight_neutral_compressed_pixels = AtomicU64::new(0);
    let shadow_compressed_pixels = AtomicU64::new(0);
    let pre_high_clipped = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    let post_high_clipped = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    let post_low_clipped = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

    result
        .axis_chunks_iter_mut(Axis(0), 512)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut out_chunk)| {
            let row_start = chunk_idx * 512;
            let chunk_h = out_chunk.dim().0;
            for local_y in 0..chunk_h {
                let y = row_start + local_y;
                for x in 0..w {
                    let r = img[[y, x, 0]];
                    let g = img[[y, x, 1]];
                    let b = img[[y, x, 2]];
                    let linear_lum = linear_luminance(r, g, b);
                    let mapped_linear = match params.domain {
                        ToneFitDomain::LinearLuminance => apply_tone_curve(linear_lum, params),
                        ToneFitDomain::Log2CompressedLuminance => {
                            let perceptual_lum = encode_perceptual_luminance(linear_lum);
                            let mapped_perceptual = apply_tone_curve(perceptual_lum, params);
                            decode_perceptual_luminance(mapped_perceptual)
                        }
                    };
                    let scale = if linear_lum <= 1e-12 {
                        0.0
                    } else {
                        mapped_linear / linear_lum
                    };
                    let scaled = [r * scale, g * scale, b * scale];
                    for ch in 0..3 {
                        if scaled[ch] > 1.0 {
                            pre_high_clipped[ch].fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    let (mapped, compressed) =
                        compress_highlight_chroma(scaled, mapped_linear, params.shoulder_max);
                    if compressed {
                        compressed_pixels.fetch_add(1, Ordering::Relaxed);
                    }
                    let (mapped, highlight_neutral_compressed) =
                        if color_protection.highlight_neutral_chroma_enabled {
                            compress_highlight_neutral_chroma(mapped, mapped_linear)
                        } else {
                            (mapped, false)
                        };
                    if highlight_neutral_compressed {
                        highlight_neutral_compressed_pixels.fetch_add(1, Ordering::Relaxed);
                    }
                    let (mapped, shadow_compressed) = if color_protection.shadow_chroma_enabled {
                        compress_shadow_chroma(mapped, mapped_linear)
                    } else {
                        (mapped, false)
                    };
                    if shadow_compressed {
                        shadow_compressed_pixels.fetch_add(1, Ordering::Relaxed);
                    }

                    for ch in 0..3 {
                        if mapped[ch] > 1.0 {
                            post_high_clipped[ch].fetch_add(1, Ordering::Relaxed);
                        }
                        if mapped[ch] < 0.0 {
                            post_low_clipped[ch].fetch_add(1, Ordering::Relaxed);
                        }
                        out_chunk[[local_y, x, ch]] = mapped[ch].clamp(0.0, 1.0);
                    }
                }
            }
        });

    let total_pixels = (h * w).max(1) as f64;
    let (result, local_detail) = if processing_config.local_luminance_detail {
        apply_local_luminance_detail(&result)
    } else {
        (result, LocalLuminanceDetailDiagnostics::disabled())
    };
    let (result, adaptive_vibrance) = if processing_config.adaptive_vibrance {
        apply_adaptive_vibrance(&result, color_protection)
    } else {
        (
            result,
            AdaptiveVibranceDiagnostics::disabled(format!(
                "disabled by {} render intent",
                render_style.as_str()
            )),
        )
    };
    let (result, post_highlight_neutral_compressed_pixels) =
        if color_protection.highlight_neutral_chroma_enabled {
            apply_highlight_neutral_chroma_cleanup(&result)
        } else {
            (result, 0)
        };
    let (result, noise_reduction) = if processing_config.noise_reduction {
        apply_render_noise_reduction(&result)
    } else {
        (
            result,
            RenderNoiseReductionDiagnostics::disabled(format!(
                "disabled by {} render intent",
                render_style.as_str()
            )),
        )
    };
    let (result, shadow_saturation_guard_pixels) = if color_protection.shadow_chroma_enabled {
        apply_shadow_saturation_guard(&result)
    } else {
        (result, 0)
    };
    let (result, midtone_neutral_compressed_pixels) =
        if color_protection.midtone_neutral_chroma_enabled {
            apply_midtone_neutral_chroma_cleanup(&result)
        } else {
            (result, 0)
        };
    let midtone_neutral_chroma_compressed_ratio =
        midtone_neutral_compressed_pixels as f64 / total_pixels;
    let shadow_chroma_compressed_ratio = (shadow_compressed_pixels.load(Ordering::Relaxed) as f64
        / total_pixels)
        .max(shadow_saturation_guard_pixels as f64 / total_pixels);
    TonemapApplyResult {
        image: result,
        diagnostics: TonemapApplyDiagnostics {
            highlight_chroma_compressed_ratio: compressed_pixels.load(Ordering::Relaxed) as f64
                / total_pixels,
            highlight_neutral_chroma_compressed_ratio: (highlight_neutral_compressed_pixels
                .load(Ordering::Relaxed)
                as f64
                / total_pixels)
                .max(post_highlight_neutral_compressed_pixels as f64 / total_pixels),
            highlight_neutral_chroma_enabled: color_protection.highlight_neutral_chroma_enabled,
            highlight_neutral_chroma_start_luminance: HIGHLIGHT_NEUTRAL_CHROMA_START_LUMINANCE,
            highlight_neutral_chroma_full_luminance: HIGHLIGHT_NEUTRAL_CHROMA_FULL_LUMINANCE,
            highlight_neutral_chroma_min_scale: HIGHLIGHT_NEUTRAL_CHROMA_MIN_SCALE,
            highlight_neutral_chroma_max_saturation: HIGHLIGHT_NEUTRAL_CHROMA_MAX_SATURATION,
            midtone_neutral_chroma_compressed_ratio,
            midtone_neutral_chroma_enabled: color_protection.midtone_neutral_chroma_enabled,
            midtone_neutral_chroma_start_luminance: MIDTONE_NEUTRAL_CHROMA_START_LUMINANCE,
            midtone_neutral_chroma_end_luminance: MIDTONE_NEUTRAL_CHROMA_END_LUMINANCE,
            midtone_neutral_chroma_min_scale: MIDTONE_NEUTRAL_CHROMA_MIN_SCALE,
            midtone_neutral_chroma_max_saturation: MIDTONE_NEUTRAL_CHROMA_MAX_SATURATION,
            shadow_chroma_compressed_ratio,
            shadow_chroma_enabled: color_protection.shadow_chroma_enabled,
            shadow_chroma_start_luminance: SHADOW_CHROMA_START_LUMINANCE,
            shadow_chroma_full_luminance: SHADOW_CHROMA_FULL_LUMINANCE,
            shadow_chroma_min_scale: SHADOW_CHROMA_MIN_SCALE,
            color_protection_policy: color_protection.policy.as_str(),
            color_trust_state: color_protection.color_trust_state(),
            color_protection_reason: color_protection.reason.clone(),
            pre_chroma_compression_clipped_high_ratio: std::array::from_fn(|ch| {
                pre_high_clipped[ch].load(Ordering::Relaxed) as f64 / total_pixels
            }),
            post_chroma_compression_clipped_high_ratio: std::array::from_fn(|ch| {
                post_high_clipped[ch].load(Ordering::Relaxed) as f64 / total_pixels
            }),
            post_chroma_compression_clipped_low_ratio: std::array::from_fn(|ch| {
                post_low_clipped[ch].load(Ordering::Relaxed) as f64 / total_pixels
            }),
            local_luminance_detail_enabled: local_detail.enabled,
            local_luminance_detail_radius: local_detail.radius,
            local_luminance_detail_amount: local_detail.amount,
            local_luminance_detail_max_ev: local_detail.max_ev,
            local_luminance_detail_applied_ratio: local_detail.applied_ratio,
            local_luminance_detail_mean_abs_ev: local_detail.mean_abs_ev,
            local_luminance_detail_max_abs_ev: local_detail.max_abs_ev,
            local_luminance_detail_headroom_limited_ratio: local_detail.headroom_limited_ratio,
            local_luminance_detail_clip_limited_ratio: local_detail.clip_limited_ratio,
            adaptive_vibrance_enabled: adaptive_vibrance.enabled,
            adaptive_vibrance_reason: adaptive_vibrance.reason,
            adaptive_vibrance_amount: adaptive_vibrance.amount,
            adaptive_vibrance_max_scale: adaptive_vibrance.max_scale,
            adaptive_vibrance_applied_ratio: adaptive_vibrance.applied_ratio,
            adaptive_vibrance_mean_scale: adaptive_vibrance.mean_scale,
            adaptive_vibrance_max_applied_scale: adaptive_vibrance.max_applied_scale,
            adaptive_vibrance_texture_limited_ratio: adaptive_vibrance.texture_limited_ratio,
            adaptive_vibrance_gamut_limited_ratio: adaptive_vibrance.gamut_limited_ratio,
            noise_reduction_enabled: noise_reduction.enabled,
            noise_reduction_reason: noise_reduction.reason,
            noise_reduction_radius: noise_reduction.radius,
            noise_reduction_chroma_amount: noise_reduction.chroma_amount,
            noise_reduction_luma_amount: noise_reduction.luma_amount,
            noise_reduction_applied_ratio: noise_reduction.applied_ratio,
            noise_reduction_texture_limited_ratio: noise_reduction.texture_limited_ratio,
            noise_reduction_saturation_limited_ratio: noise_reduction.saturation_limited_ratio,
            noise_reduction_mean_abs_chroma_delta: noise_reduction.mean_abs_chroma_delta,
            noise_reduction_max_abs_chroma_delta: noise_reduction.max_abs_chroma_delta,
            noise_reduction_mean_abs_luma_delta: noise_reduction.mean_abs_luma_delta,
            noise_reduction_max_abs_luma_delta: noise_reduction.max_abs_luma_delta,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_midpoint_for_target_treats_nan_input_as_low_curve_bound() {
        let nan_midpoint = solve_midpoint_for_target(
            f64::NAN,
            0.35,
            2.0,
            POSITIVE_SCAN_TOE_LIFT,
            POSITIVE_SCAN_SHOULDER_MAX,
        );
        let low_bound_midpoint = solve_midpoint_for_target(
            1e-6,
            0.35,
            2.0,
            POSITIVE_SCAN_TOE_LIFT,
            POSITIVE_SCAN_SHOULDER_MAX,
        );

        assert!(nan_midpoint.is_finite());
        assert!((nan_midpoint - low_bound_midpoint).abs() < 1e-12);
    }
}

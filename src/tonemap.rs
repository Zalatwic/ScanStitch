use nalgebra::Vector3;
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
const NEGATIVE_SHADOW_TARGET_LINEAR_MEDIAN: f64 = 0.335;
const NEGATIVE_SHADOW_MIDRANGE_TARGET_LINEAR_MEDIAN: f64 = 0.32;
const NEGATIVE_SHADOW_TARGET_MAX_MAPPED_P95: f64 = 0.95;
const NEGATIVE_SHADOW_TARGET_LINEAR_MEDIAN_MAX: f64 = 0.22;
const NEGATIVE_SHADOW_LOW_RANGE_MAX_P95: f64 = 0.25;
const NEGATIVE_SHADOW_BROAD_HEADROOM_MIN_P95: f64 = 0.40;
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
const HIGHLIGHT_NEUTRAL_CHROMA_START_LUMINANCE: f64 = 0.63;
const HIGHLIGHT_NEUTRAL_CHROMA_FULL_LUMINANCE: f64 = 0.83;
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
// Van Song et al. measured 2,770 women across four continents and reported a global CIELAB
// skin-colour range of L*=27.12..73.71, C*=9.01..30.43, h=24.63..79.64 degrees
// (https://doi.org/10.1002/col.70012). The wider support below deliberately feathers that
// measured core after conversion into this renderer's D50 working condition. This is a
// memory-colour protection region, not semantic skin detection: false positives only receive less
// creative vibrance and never alter the technical scene-referred master.
const ADAPTIVE_VIBRANCE_SKIN_CORE_LIGHTNESS: [f64; 2] = [27.12, 73.71];
const ADAPTIVE_VIBRANCE_SKIN_SUPPORT_LIGHTNESS: [f64; 2] = [18.0, 88.0];
const ADAPTIVE_VIBRANCE_SKIN_CORE_CHROMA: [f64; 2] = [9.01, 30.43];
const ADAPTIVE_VIBRANCE_SKIN_SUPPORT_CHROMA: [f64; 2] = [5.0, 45.0];
const ADAPTIVE_VIBRANCE_SKIN_CORE_HUE_DEGREES: [f64; 2] = [24.63, 79.64];
const ADAPTIVE_VIBRANCE_SKIN_SUPPORT_HUE_DEGREES: [f64; 2] = [15.0, 90.0];
const ADAPTIVE_VIBRANCE_SKIN_MAXIMUM_REDUCTION: f64 = 0.90;
// Ji, Tian, and Luo fitted 50%-acceptability CIELAB a*b* ellipses for preferred sky,
// spring-grass, and autumn-grass reproduction in a psychophysical mobile-display experiment
// (https://doi.org/10.2352/issn.2169-2629.2021.29.170, Table 2). We use those published
// image-quality centers and ellipse shapes only as a one-way creative-vibrance overshoot guard.
// A supported pixel may keep a boost that moves its a*b* projection toward the nearest preferred
// center, but the guard can reduce a boost that would move farther away. It never pulls a pixel
// toward a center, changes the technical master, supplies semantic object detection, or establishes
// calibration truth. Radius 1 is the published 50% ellipse; radius 2 is an explicit feathered
// engineering support region whose false positives can only receive less optional vibrance.
const ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_CORE_RADIUS: f64 = 1.0;
const ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_SUPPORT_RADIUS: f64 = 2.0;
const ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_REFERENCE: &str =
    "https://doi.org/10.2352/issn.2169-2629.2021.29.170";
// The same study reports an aggregate preferred-skin image-quality centre and 50%-acceptability
// a*b* ellipse. Modern-clean rendering leaves that ellipse untouched and applies only a bounded
// one-way shoulder to high-chroma outlying pixels which are also supported by the broader
// four-continent measured skin-colour region above. The adjustment is an appearance proxy, not
// face detection or calibration: it never raises chroma, preserves CIELAB lightness, reduces only
// 35% of ellipse-radius excess, is capped at 3 DeltaE_ab, and is disabled unless upstream colour
// is trusted.
const PREFERRED_SKIN_RENDERING_CORE_RADIUS: f64 = 1.0;
const PREFERRED_SKIN_RENDERING_RADIAL_EXCESS_REDUCTION: f64 = 0.35;
const PREFERRED_SKIN_RENDERING_MAX_DELTA_E_AB: f64 = 3.0;
const PREFERRED_SKIN_RENDERING_MIN_SUPPORT_WEIGHT: f64 = 0.05;
const PREFERRED_SKIN_RENDERING_MIN_DELTA_E_AB: f64 = 0.001;

#[derive(Debug, Clone, Copy)]
struct PreferredMemoryColorModel {
    family: &'static str,
    preferred_center_lch: [f64; 3],
    semi_major_axis_ab: f64,
    axis_ratio: f64,
    ellipse_rotation_degrees: f64,
}

impl PreferredMemoryColorModel {
    const fn semi_minor_axis_ab(self) -> f64 {
        self.semi_major_axis_ab / self.axis_ratio
    }

    fn preferred_center_lab(self) -> [f64; 3] {
        let hue = self.preferred_center_lch[2].to_radians();
        [
            self.preferred_center_lch[0],
            self.preferred_center_lch[1] * hue.cos(),
            self.preferred_center_lch[1] * hue.sin(),
        ]
    }
}

const ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_MODELS: [PreferredMemoryColorModel; 3] = [
    PreferredMemoryColorModel {
        family: "sky",
        preferred_center_lch: [58.3, 43.7, 277.8],
        semi_major_axis_ab: 15.02,
        axis_ratio: 1.85,
        ellipse_rotation_degrees: 120.00,
    },
    PreferredMemoryColorModel {
        family: "spring_grass",
        preferred_center_lch: [37.7, 61.7, 124.0],
        semi_major_axis_ab: 32.71,
        axis_ratio: 2.75,
        ellipse_rotation_degrees: 125.77,
    },
    PreferredMemoryColorModel {
        family: "autumn_grass",
        preferred_center_lch: [45.3, 44.2, 88.3],
        semi_major_axis_ab: 16.08,
        axis_ratio: 1.25,
        ellipse_rotation_degrees: 96.77,
    },
];
const PREFERRED_SKIN_RENDERING_MODEL: PreferredMemoryColorModel = PreferredMemoryColorModel {
    family: "skin",
    preferred_center_lch: [61.3, 25.6, 40.3],
    semi_major_axis_ab: 19.41,
    axis_ratio: 2.78,
    ellipse_rotation_degrees: 71.57,
};
const RENDER_NOISE_REDUCTION_RADIUS: usize = 2;
const RENDER_NOISE_REDUCTION_CHROMA_RADIUS: usize = 2;
const RENDER_NOISE_REDUCTION_CHROMA_AMOUNT: f64 = 0.96;
const RENDER_NOISE_REDUCTION_LUMA_AMOUNT: f64 = 0.30;
const RENDER_NOISE_REDUCTION_TEXTURE_START: f64 = 0.018;
const RENDER_NOISE_REDUCTION_TEXTURE_END: f64 = 0.180;
// A pixel at or above the same contrast floor used by the independent detail-retention
// decision must be left exactly unchanged. The feather below that floor avoids a hard halo while
// still making the optional pass selective on real photographs rather than a nearly universal
// low-pass blend.
const RENDER_NOISE_REDUCTION_STRUCTURE_GATE_START: f64 = RENDER_NOISE_REDUCTION_TEXTURE_START;
const RENDER_NOISE_REDUCTION_STRUCTURE_GATE_END: f64 = GRAIN_DETAIL_LUMA_CONTRAST_MIN;
const RENDER_NOISE_REDUCTION_CHROMA_TEXTURE_RESIDUAL_WEIGHT: f64 = 0.24;
const RENDER_NOISE_REDUCTION_LUMA_TEXTURE_RESIDUAL_WEIGHT: f64 = 0.10;
const RENDER_NOISE_REDUCTION_LUMA_TEXTURE_FLOOR: f64 = 0.50;
const RENDER_NOISE_REDUCTION_FRAME_CHROMA_GRAIN_START: f64 = 0.075;
const RENDER_NOISE_REDUCTION_FRAME_CHROMA_GRAIN_FULL: f64 = 0.125;
const RENDER_NOISE_REDUCTION_CHROMA_DAMP_AMOUNT: f64 = 0.85;
const RENDER_NOISE_REDUCTION_NEUTRAL_TEXTURE_FLOOR: f64 = 0.85;
const RENDER_NOISE_REDUCTION_NEUTRAL_FLOOR_HIGHLIGHT_START: f64 = 0.62;
const RENDER_NOISE_REDUCTION_NEUTRAL_FLOOR_HIGHLIGHT_END: f64 = 0.82;
const RENDER_NOISE_REDUCTION_SATURATION_GATE_FLOOR: f64 = 0.35;
const RENDER_NOISE_REDUCTION_SATURATION_START: f64 = 0.24;
const RENDER_NOISE_REDUCTION_SATURATION_END: f64 = 0.72;
const RENDER_NOISE_REDUCTION_SHADOW_SATURATION_RELAXATION: f64 = 0.90;
const RENDER_NOISE_REDUCTION_SHADOW_START: f64 = 0.08;
const RENDER_NOISE_REDUCTION_SHADOW_END: f64 = 0.42;
const GRAIN_DETAIL_MAX_PROBES: usize = 200_000;
pub const GRAIN_DETAIL_MIN_PROBE_COUNT: usize = 64;
const GRAIN_DETAIL_LUMA_CONTRAST_MIN: f64 = 0.035;
const GRAIN_DETAIL_CHROMA_CONTRAST_MIN: f64 = 0.050;
const GRAIN_DETAIL_COHERENCE_MIN: f64 = 0.75;
const GRAIN_DETAIL_SCALE_AGREEMENT_MIN: f64 = 0.35;
const GRAIN_DETAIL_MEDIAN_RETENTION_MIN: f64 = 0.90;
pub const GRAIN_DETAIL_P10_RETENTION_MIN: f64 = 0.70;
const GRAIN_DETAIL_RETENTION_RATIO_MAX: f64 = 2.0;
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
                preferred_skin_rendering: true,
            },
            Self::NaturalNeutral => RenderProcessingConfig {
                local_luminance_detail: true,
                adaptive_vibrance: false,
                preferred_skin_rendering: false,
            },
            Self::FilmFaithful => RenderProcessingConfig {
                local_luminance_detail: false,
                adaptive_vibrance: false,
                preferred_skin_rendering: false,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RenderProcessingConfig {
    local_luminance_detail: bool,
    adaptive_vibrance: bool,
    preferred_skin_rendering: bool,
}

/// Film-grain reduction is deliberately independent of creative render style.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrainReductionSettings {
    pub enabled: bool,
    /// Normalized user strength in the inclusive range 0..=1.
    pub strength: f64,
    /// Spatial radius multiplier in the inclusive range 0.5..=4.
    pub scale: f64,
}

impl GrainReductionSettings {
    pub fn normalized(self) -> Self {
        Self {
            enabled: self.enabled,
            strength: if self.strength.is_finite() {
                self.strength.clamp(0.0, 1.0)
            } else {
                0.0
            },
            scale: if self.scale.is_finite() {
                self.scale.clamp(0.5, 4.0)
            } else {
                1.0
            },
        }
    }

    fn effective_radius(self) -> usize {
        ((RENDER_NOISE_REDUCTION_RADIUS as f64 * self.normalized().scale).round() as usize)
            .clamp(1, 8)
    }
}

impl Default for GrainReductionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            strength: 0.5,
            scale: 1.0,
        }
    }
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
pub struct AdaptiveVibranceSkinMemoryProtectionDiagnostics {
    pub enabled: bool,
    pub method: &'static str,
    pub working_space: &'static str,
    pub reference: &'static str,
    pub core_lightness: [f64; 2],
    pub support_lightness: [f64; 2],
    pub core_chroma: [f64; 2],
    pub support_chroma: [f64; 2],
    pub core_hue_degrees: [f64; 2],
    pub support_hue_degrees: [f64; 2],
    pub maximum_vibrance_reduction: f64,
    pub evaluated_pixel_ratio: f64,
    pub protected_pixel_ratio: f64,
    pub mean_protection_weight: f64,
    pub max_protection_weight: f64,
}

impl AdaptiveVibranceSkinMemoryProtectionDiagnostics {
    fn empty(enabled: bool) -> Self {
        Self {
            enabled,
            method: "feathered_cielab_lightness_chroma_hue_region_limits_only_creative_vibrance",
            working_space: "CIELAB_D50_from_linear_ProPhoto_RGB_D50",
            reference: "https://doi.org/10.1002/col.70012",
            core_lightness: ADAPTIVE_VIBRANCE_SKIN_CORE_LIGHTNESS,
            support_lightness: ADAPTIVE_VIBRANCE_SKIN_SUPPORT_LIGHTNESS,
            core_chroma: ADAPTIVE_VIBRANCE_SKIN_CORE_CHROMA,
            support_chroma: ADAPTIVE_VIBRANCE_SKIN_SUPPORT_CHROMA,
            core_hue_degrees: ADAPTIVE_VIBRANCE_SKIN_CORE_HUE_DEGREES,
            support_hue_degrees: ADAPTIVE_VIBRANCE_SKIN_SUPPORT_HUE_DEGREES,
            maximum_vibrance_reduction: ADAPTIVE_VIBRANCE_SKIN_MAXIMUM_REDUCTION,
            evaluated_pixel_ratio: 0.0,
            protected_pixel_ratio: 0.0,
            mean_protection_weight: 0.0,
            max_protection_weight: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreferredSkinRenderingDiagnostics {
    pub enabled: bool,
    pub reason: String,
    pub method: &'static str,
    pub working_space: &'static str,
    pub preference_reference: &'static str,
    pub support_reference: &'static str,
    pub interpretation: &'static str,
    pub preferred_center_lab: [f64; 3],
    pub preferred_center_lch: [f64; 3],
    pub semi_major_axis_ab: f64,
    pub semi_minor_axis_ab: f64,
    pub axis_ratio: f64,
    pub ellipse_rotation_degrees: f64,
    pub core_normalized_radius: f64,
    pub radial_excess_reduction: f64,
    pub maximum_delta_e_ab: f64,
    pub minimum_support_weight: f64,
    pub evaluated_pixel_ratio: f64,
    pub matched_pixel_ratio: f64,
    pub outside_preferred_core_ratio: f64,
    pub adjusted_pixel_ratio: f64,
    pub gamut_limited_pixel_ratio: f64,
    pub mean_delta_e_ab: f64,
    pub max_delta_e_ab: f64,
    pub mean_abs_hue_shift_degrees: f64,
    pub max_abs_hue_shift_degrees: f64,
    pub mean_chroma_delta: f64,
    pub max_abs_chroma_delta: f64,
}

impl PreferredSkinRenderingDiagnostics {
    fn empty(enabled: bool, reason: impl Into<String>) -> Self {
        Self {
            enabled,
            reason: reason.into(),
            method: "published_skin_preference_ellipse_with_bounded_one_way_excess_chroma_shoulder",
            working_space: "CIELAB_D50_ab_projection_from_linear_ProPhoto_RGB_D50",
            preference_reference: ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_REFERENCE,
            support_reference: "https://doi.org/10.1002/col.70012",
            interpretation: "display-only aggregate appearance proxy; not face or skin-group detection, calibration evidence, or a claim that every matched pixel is skin",
            preferred_center_lab: PREFERRED_SKIN_RENDERING_MODEL.preferred_center_lab(),
            preferred_center_lch: PREFERRED_SKIN_RENDERING_MODEL.preferred_center_lch,
            semi_major_axis_ab: PREFERRED_SKIN_RENDERING_MODEL.semi_major_axis_ab,
            semi_minor_axis_ab: PREFERRED_SKIN_RENDERING_MODEL.semi_minor_axis_ab(),
            axis_ratio: PREFERRED_SKIN_RENDERING_MODEL.axis_ratio,
            ellipse_rotation_degrees: PREFERRED_SKIN_RENDERING_MODEL.ellipse_rotation_degrees,
            core_normalized_radius: PREFERRED_SKIN_RENDERING_CORE_RADIUS,
            radial_excess_reduction: PREFERRED_SKIN_RENDERING_RADIAL_EXCESS_REDUCTION,
            maximum_delta_e_ab: PREFERRED_SKIN_RENDERING_MAX_DELTA_E_AB,
            minimum_support_weight: PREFERRED_SKIN_RENDERING_MIN_SUPPORT_WEIGHT,
            evaluated_pixel_ratio: 0.0,
            matched_pixel_ratio: 0.0,
            outside_preferred_core_ratio: 0.0,
            adjusted_pixel_ratio: 0.0,
            gamut_limited_pixel_ratio: 0.0,
            mean_delta_e_ab: 0.0,
            max_delta_e_ab: 0.0,
            mean_abs_hue_shift_degrees: 0.0,
            max_abs_hue_shift_degrees: 0.0,
            mean_chroma_delta: 0.0,
            max_abs_chroma_delta: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveVibrancePreferredMemoryColorFamilyDiagnostics {
    pub family: &'static str,
    pub preferred_center_lab: [f64; 3],
    pub preferred_center_lch: [f64; 3],
    pub semi_major_axis_ab: f64,
    pub semi_minor_axis_ab: f64,
    pub axis_ratio: f64,
    pub ellipse_rotation_degrees: f64,
    pub matched_pixel_ratio: f64,
    pub limited_pixel_ratio: f64,
    pub mean_scale_reduction: f64,
    pub max_scale_reduction: f64,
}

impl AdaptiveVibrancePreferredMemoryColorFamilyDiagnostics {
    fn empty(model: PreferredMemoryColorModel) -> Self {
        Self {
            family: model.family,
            preferred_center_lab: model.preferred_center_lab(),
            preferred_center_lch: model.preferred_center_lch,
            semi_major_axis_ab: model.semi_major_axis_ab,
            semi_minor_axis_ab: model.semi_minor_axis_ab(),
            axis_ratio: model.axis_ratio,
            ellipse_rotation_degrees: model.ellipse_rotation_degrees,
            matched_pixel_ratio: 0.0,
            limited_pixel_ratio: 0.0,
            mean_scale_reduction: 0.0,
            max_scale_reduction: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveVibrancePreferredMemoryColorGuardDiagnostics {
    pub enabled: bool,
    pub method: &'static str,
    pub working_space: &'static str,
    pub reference: &'static str,
    pub interpretation: &'static str,
    pub core_normalized_radius: f64,
    pub support_normalized_radius: f64,
    pub evaluated_pixel_ratio: f64,
    pub matched_pixel_ratio: f64,
    pub limited_pixel_ratio: f64,
    pub mean_scale_reduction: f64,
    pub max_scale_reduction: f64,
    pub families: Vec<AdaptiveVibrancePreferredMemoryColorFamilyDiagnostics>,
}

impl AdaptiveVibrancePreferredMemoryColorGuardDiagnostics {
    fn empty(enabled: bool) -> Self {
        Self {
            enabled,
            method: "published_preference_ellipse_support_then_one_way_ab_path_projection_limits_only_creative_vibrance",
            working_space: "CIELAB_D50_ab_projection_from_linear_ProPhoto_RGB_D50",
            reference: ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_REFERENCE,
            interpretation: "appearance-relative proxy only; not semantic detection, calibration evidence, or an instruction to pull pixels toward a preferred center",
            core_normalized_radius: ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_CORE_RADIUS,
            support_normalized_radius: ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_SUPPORT_RADIUS,
            evaluated_pixel_ratio: 0.0,
            matched_pixel_ratio: 0.0,
            limited_pixel_ratio: 0.0,
            mean_scale_reduction: 0.0,
            max_scale_reduction: 0.0,
            families: ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_MODELS
                .iter()
                .copied()
                .map(AdaptiveVibrancePreferredMemoryColorFamilyDiagnostics::empty)
                .collect(),
        }
    }
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
    pub perceptual_gamut_mapping_space: &'static str,
    pub perceptual_gamut_mapped_ratio: f64,
    pub perceptual_gamut_mean_chroma_scale: f64,
    pub perceptual_gamut_min_chroma_scale: f64,
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
    pub adaptive_vibrance_skin_memory_protection: AdaptiveVibranceSkinMemoryProtectionDiagnostics,
    pub adaptive_vibrance_preferred_memory_color_guard:
        AdaptiveVibrancePreferredMemoryColorGuardDiagnostics,
    pub preferred_skin_rendering: PreferredSkinRenderingDiagnostics,
    pub noise_reduction_enabled: bool,
    pub noise_reduction_requested_enabled: bool,
    pub noise_reduction_reason: String,
    pub noise_reduction_requested_strength: f64,
    pub noise_reduction_requested_scale: f64,
    pub noise_reduction_radius: usize,
    pub noise_reduction_chroma_amount: f64,
    pub noise_reduction_luma_amount: f64,
    pub noise_reduction_applied_ratio: f64,
    pub noise_reduction_structure_gate_start: f64,
    pub noise_reduction_structure_gate_end: f64,
    pub noise_reduction_structure_excluded_ratio: f64,
    pub noise_reduction_texture_limited_ratio: f64,
    pub noise_reduction_saturation_limited_ratio: f64,
    pub noise_reduction_mean_abs_chroma_delta: f64,
    pub noise_reduction_max_abs_chroma_delta: f64,
    pub noise_reduction_mean_abs_luma_delta: f64,
    pub noise_reduction_max_abs_luma_delta: f64,
    pub noise_reduction_pre_grain: RenderGrainDiagnostics,
    pub noise_reduction_post_grain: RenderGrainDiagnostics,
    pub noise_reduction_flat_luma_p95_reduction_ratio: f64,
    pub noise_reduction_flat_chroma_p95_reduction_ratio: f64,
    pub noise_reduction_detail_retention: GrainDetailRetentionDiagnostics,
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
    skin_memory_protection: AdaptiveVibranceSkinMemoryProtectionDiagnostics,
    preferred_memory_color_guard: AdaptiveVibrancePreferredMemoryColorGuardDiagnostics,
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
            skin_memory_protection: AdaptiveVibranceSkinMemoryProtectionDiagnostics::empty(false),
            preferred_memory_color_guard:
                AdaptiveVibrancePreferredMemoryColorGuardDiagnostics::empty(false),
        }
    }
}

#[derive(Debug, Clone)]
struct RenderNoiseReductionDiagnostics {
    enabled: bool,
    requested_enabled: bool,
    reason: String,
    requested_strength: f64,
    requested_scale: f64,
    radius: usize,
    chroma_amount: f64,
    luma_amount: f64,
    applied_ratio: f64,
    structure_gate_start: f64,
    structure_gate_end: f64,
    structure_excluded_ratio: f64,
    texture_limited_ratio: f64,
    saturation_limited_ratio: f64,
    mean_abs_chroma_delta: f64,
    max_abs_chroma_delta: f64,
    mean_abs_luma_delta: f64,
    max_abs_luma_delta: f64,
    detail_retention: GrainDetailRetentionDiagnostics,
}

impl RenderNoiseReductionDiagnostics {
    fn disabled(settings: GrainReductionSettings, reason: impl Into<String>) -> Self {
        let settings = settings.normalized();
        Self {
            enabled: false,
            requested_enabled: settings.enabled,
            reason: reason.into(),
            requested_strength: settings.strength,
            requested_scale: settings.scale,
            radius: settings.effective_radius(),
            chroma_amount: RENDER_NOISE_REDUCTION_CHROMA_AMOUNT * settings.strength,
            luma_amount: RENDER_NOISE_REDUCTION_LUMA_AMOUNT * settings.strength,
            applied_ratio: 0.0,
            structure_gate_start: RENDER_NOISE_REDUCTION_STRUCTURE_GATE_START,
            structure_gate_end: RENDER_NOISE_REDUCTION_STRUCTURE_GATE_END,
            structure_excluded_ratio: 0.0,
            texture_limited_ratio: 0.0,
            saturation_limited_ratio: 0.0,
            mean_abs_chroma_delta: 0.0,
            max_abs_chroma_delta: 0.0,
            mean_abs_luma_delta: 0.0,
            max_abs_luma_delta: 0.0,
            detail_retention: GrainDetailRetentionDiagnostics::not_applied(),
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
        if diagnostics.candidate_risk == "fallback_only" {
            return Self {
                policy: ToneColorProtectionPolicy::DisabledColorCandidateReview,
                highlight_neutral_chroma_enabled: false,
                midtone_neutral_chroma_enabled: false,
                shadow_chroma_enabled: false,
                reason: format!(
                    "disabled because selected colorspace candidate `{}` is fallback-only and cannot establish trusted colour",
                    diagnostics.selected_candidate
                ),
            };
        }
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

#[derive(Debug, Clone)]
pub struct GrainDetailRetentionDiagnostics {
    pub method: &'static str,
    pub evaluated: bool,
    pub decision_supported: bool,
    pub sample_stride: usize,
    pub probe_radius: usize,
    pub minimum_probe_count: usize,
    pub luminance_probe_count: usize,
    pub chroma_probe_count: usize,
    pub luminance_decision_supported: bool,
    pub chroma_decision_supported: bool,
    pub luminance_median_retention: f64,
    pub luminance_p10_retention: f64,
    pub chroma_median_retention: f64,
    pub chroma_p10_retention: f64,
    pub luminance_contrast_threshold: f64,
    pub chroma_contrast_threshold: f64,
    pub coherence_threshold: f64,
    pub median_retention_threshold: f64,
    pub p10_retention_threshold: f64,
    pub review_required: bool,
    pub reason: String,
    pub review_reason: Option<String>,
}

impl GrainDetailRetentionDiagnostics {
    fn not_applied() -> Self {
        Self {
            method: "coherent_multiscale_opponent_edge_retention_v1",
            evaluated: false,
            decision_supported: true,
            sample_stride: 1,
            probe_radius: 0,
            minimum_probe_count: GRAIN_DETAIL_MIN_PROBE_COUNT,
            luminance_probe_count: 0,
            chroma_probe_count: 0,
            luminance_decision_supported: false,
            chroma_decision_supported: false,
            luminance_median_retention: 1.0,
            luminance_p10_retention: 1.0,
            chroma_median_retention: 1.0,
            chroma_p10_retention: 1.0,
            luminance_contrast_threshold: GRAIN_DETAIL_LUMA_CONTRAST_MIN,
            chroma_contrast_threshold: GRAIN_DETAIL_CHROMA_CONTRAST_MIN,
            coherence_threshold: GRAIN_DETAIL_COHERENCE_MIN,
            median_retention_threshold: GRAIN_DETAIL_MEDIAN_RETENTION_MIN,
            p10_retention_threshold: GRAIN_DETAIL_P10_RETENTION_MIN,
            review_required: false,
            reason:
                "grain reduction did not run, so no pre/post detail-retention decision was needed"
                    .to_string(),
            review_reason: None,
        }
    }
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
) -> ([f64; 3], bool, f64) {
    let ceiling = gamut_ceiling.clamp(mapped_lum.clamp(0.0, 1.0), 1.0);
    if rgb.iter().any(|value| !value.is_finite()) {
        let neutral = mapped_lum.clamp(0.0, ceiling);
        return ([neutral; 3], true, 0.0);
    }
    if rgb
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0 && *value <= ceiling)
    {
        return (rgb, false, 1.0);
    }

    let prophoto_to_xyz = crate::colorspace::prophoto_to_xyz_d50_matrix();
    let xyz_to_prophoto = crate::colorspace::xyz_d50_to_prophoto_matrix();
    let xyz = prophoto_to_xyz * Vector3::new(rgb[0], rgb[1], rgb[2]);
    let lab = crate::colorspace::xyz_d50_to_lab([xyz[0], xyz[1], xyz[2]]);
    let chroma = lab[1].hypot(lab[2]);
    if lab.iter().all(|value| value.is_finite()) && chroma > 1e-9 {
        let hue_cos = lab[1] / chroma;
        let hue_sin = lab[2] / chroma;
        let candidate_for_scale = |scale: f64| {
            let candidate_lab = [
                lab[0].clamp(0.0, 100.0),
                chroma * scale * hue_cos,
                chroma * scale * hue_sin,
            ];
            let candidate_xyz = Vector3::from(crate::colorspace::lab_to_xyz_d50(candidate_lab));
            let candidate = xyz_to_prophoto * candidate_xyz;
            [candidate[0], candidate[1], candidate[2]]
        };

        let in_gamut = |candidate: &[f64; 3]| {
            candidate
                .iter()
                .all(|value| value.is_finite() && *value >= -1e-10 && *value <= ceiling + 1e-10)
        };
        let neutral = candidate_for_scale(0.0);
        if in_gamut(&neutral) {
            let mut low = 0.0f64;
            let mut high = 1.0f64;
            for _ in 0..22 {
                let mid = (low + high) * 0.5;
                if in_gamut(&candidate_for_scale(mid)) {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            let chroma_scale = (low * 0.999_999).clamp(0.0, 1.0);
            let compressed = candidate_for_scale(chroma_scale);
            return (compressed, chroma_scale < 1.0 - 1e-7, chroma_scale);
        }
    }

    // Degenerate/non-finite Lab values retain a bounded luminance and fall back to the same
    // neutral-axis geometry rather than channel-wise clipping, so hue is never independently
    // truncated in one channel.
    let neutral = mapped_lum.clamp(0.0, ceiling);
    let mut chroma_scale = 1.0f64;
    for value in rgb {
        let delta = value - neutral;
        if value > ceiling && delta > 1e-12 {
            chroma_scale = chroma_scale.min((ceiling - neutral) / delta);
        } else if value < 0.0 && delta < -1e-12 {
            chroma_scale = chroma_scale.min((0.0 - neutral) / delta);
        }
    }
    let chroma_scale = chroma_scale.clamp(0.0, 1.0);
    let compressed = std::array::from_fn(|c| neutral + (rgb[c] - neutral) * chroma_scale);
    (compressed, chroma_scale < 1.0 - 1e-12, chroma_scale)
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

fn feathered_range_weight(value: f64, support: [f64; 2], core: [f64; 2]) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    smoothstep_range(support[0], core[0], value)
        * (1.0 - smoothstep_range(core[1], support[1], value))
}

fn prophoto_rgb_to_lab(
    rgb: [f64; 3],
    prophoto_to_xyz: &nalgebra::Matrix3<f64>,
) -> Option<[f64; 3]> {
    if rgb.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let xyz = prophoto_to_xyz * Vector3::new(rgb[0], rgb[1], rgb[2]);
    let lab = crate::colorspace::xyz_d50_to_lab([xyz[0], xyz[1], xyz[2]]);
    lab.iter().all(|value| value.is_finite()).then_some(lab)
}

fn skin_memory_protection_weight_from_lab(lab: [f64; 3]) -> f64 {
    if lab.iter().any(|value| !value.is_finite()) {
        return 0.0;
    }
    let chroma = lab[1].hypot(lab[2]);
    let hue_degrees = lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0);
    feathered_range_weight(
        lab[0],
        ADAPTIVE_VIBRANCE_SKIN_SUPPORT_LIGHTNESS,
        ADAPTIVE_VIBRANCE_SKIN_CORE_LIGHTNESS,
    ) * feathered_range_weight(
        chroma,
        ADAPTIVE_VIBRANCE_SKIN_SUPPORT_CHROMA,
        ADAPTIVE_VIBRANCE_SKIN_CORE_CHROMA,
    ) * feathered_range_weight(
        hue_degrees,
        ADAPTIVE_VIBRANCE_SKIN_SUPPORT_HUE_DEGREES,
        ADAPTIVE_VIBRANCE_SKIN_CORE_HUE_DEGREES,
    )
}

#[cfg(test)]
fn skin_memory_protection_weight(rgb: [f64; 3], prophoto_to_xyz: &nalgebra::Matrix3<f64>) -> f64 {
    prophoto_rgb_to_lab(rgb, prophoto_to_xyz)
        .map(skin_memory_protection_weight_from_lab)
        .unwrap_or(0.0)
}

fn preferred_memory_color_normalized_radius(
    lab: [f64; 3],
    model: PreferredMemoryColorModel,
) -> f64 {
    if lab.iter().any(|value| !value.is_finite()) {
        return f64::INFINITY;
    }
    let center = model.preferred_center_lab();
    let delta_a = lab[1] - center[1];
    let delta_b = lab[2] - center[2];
    let theta = model.ellipse_rotation_degrees.to_radians();
    let along_major = theta.cos() * delta_a + theta.sin() * delta_b;
    let along_minor = -theta.sin() * delta_a + theta.cos() * delta_b;
    ((along_major / model.semi_major_axis_ab).powi(2)
        + (along_minor / model.semi_minor_axis_ab()).powi(2))
    .sqrt()
}

fn preferred_memory_color_support_weight(normalized_radius: f64) -> f64 {
    if !normalized_radius.is_finite()
        || normalized_radius >= ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_SUPPORT_RADIUS
    {
        return 0.0;
    }
    1.0 - smoothstep_range(
        ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_CORE_RADIUS,
        ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_SUPPORT_RADIUS,
        normalized_radius,
    )
}

#[derive(Debug, Clone, Copy)]
struct PreferredMemoryColorGuardDecision {
    scale: f64,
    family_index: Option<usize>,
    support_weight: f64,
    scale_reduction: f64,
}

fn preferred_memory_color_match(
    before_lab: [f64; 3],
) -> Option<(usize, PreferredMemoryColorModel, f64)> {
    ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_MODELS
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, model)| {
            let radius = preferred_memory_color_normalized_radius(before_lab, model);
            (radius < ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_SUPPORT_RADIUS)
                .then_some((index, model, radius))
        })
        .min_by(|left, right| left.2.total_cmp(&right.2))
}

fn preferred_memory_color_guard_scale_with_match(
    before_lab: [f64; 3],
    tentative_lab: [f64; 3],
    requested_scale: f64,
    matched: Option<(usize, PreferredMemoryColorModel, f64)>,
) -> PreferredMemoryColorGuardDecision {
    let Some((family_index, model, radius)) = matched else {
        return PreferredMemoryColorGuardDecision {
            scale: requested_scale,
            family_index: None,
            support_weight: 0.0,
            scale_reduction: 0.0,
        };
    };

    let support_weight = preferred_memory_color_support_weight(radius);
    let path_a = tentative_lab[1] - before_lab[1];
    let path_b = tentative_lab[2] - before_lab[2];
    let path_norm_sq = path_a * path_a + path_b * path_b;
    if path_norm_sq <= 1e-18 || requested_scale <= 1.0 + 1e-12 {
        return PreferredMemoryColorGuardDecision {
            scale: requested_scale,
            family_index: Some(family_index),
            support_weight,
            scale_reduction: 0.0,
        };
    }

    let center = model.preferred_center_lab();
    let target_a = center[1] - before_lab[1];
    let target_b = center[2] - before_lab[2];
    let closest_path_fraction =
        ((target_a * path_a + target_b * path_b) / path_norm_sq).clamp(0.0, 1.0);
    let closest_scale = 1.0 + (requested_scale - 1.0) * closest_path_fraction;
    let guarded_scale =
        requested_scale - support_weight * (requested_scale - closest_scale).max(0.0);
    let guarded_scale = guarded_scale.clamp(1.0, requested_scale);
    PreferredMemoryColorGuardDecision {
        scale: guarded_scale,
        family_index: Some(family_index),
        support_weight,
        scale_reduction: (requested_scale - guarded_scale).max(0.0),
    }
}

#[cfg(test)]
fn preferred_memory_color_guard_scale(
    before_lab: [f64; 3],
    tentative_lab: [f64; 3],
    requested_scale: f64,
) -> PreferredMemoryColorGuardDecision {
    preferred_memory_color_guard_scale_with_match(
        before_lab,
        tentative_lab,
        requested_scale,
        preferred_memory_color_match(before_lab),
    )
}

#[derive(Debug, Clone, Copy, Default)]
struct PreferredMemoryColorFamilyAccumulator {
    matched: usize,
    limited: usize,
    scale_reduction_sum: f64,
    scale_reduction_max: f64,
}

#[derive(Debug, Clone, Default)]
struct PreferredMemoryColorGuardAccumulator {
    evaluated: usize,
    matched: usize,
    limited: usize,
    scale_reduction_sum: f64,
    scale_reduction_max: f64,
    families: [PreferredMemoryColorFamilyAccumulator; 3],
}

impl PreferredMemoryColorGuardAccumulator {
    fn record(&mut self, decision: PreferredMemoryColorGuardDecision) {
        self.evaluated += 1;
        let Some(family_index) = decision.family_index else {
            return;
        };
        if decision.support_weight <= 0.0 {
            return;
        }
        self.matched += 1;
        let family = &mut self.families[family_index];
        family.matched += 1;
        if decision.scale_reduction <= 1e-12 {
            return;
        }
        self.limited += 1;
        self.scale_reduction_sum += decision.scale_reduction;
        self.scale_reduction_max = self.scale_reduction_max.max(decision.scale_reduction);
        family.limited += 1;
        family.scale_reduction_sum += decision.scale_reduction;
        family.scale_reduction_max = family.scale_reduction_max.max(decision.scale_reduction);
    }

    fn finish(self, denominator: f64) -> AdaptiveVibrancePreferredMemoryColorGuardDiagnostics {
        let mut diagnostics = AdaptiveVibrancePreferredMemoryColorGuardDiagnostics::empty(true);
        diagnostics.evaluated_pixel_ratio = self.evaluated as f64 / denominator;
        diagnostics.matched_pixel_ratio = self.matched as f64 / denominator;
        diagnostics.limited_pixel_ratio = self.limited as f64 / denominator;
        diagnostics.mean_scale_reduction = if self.limited == 0 {
            0.0
        } else {
            self.scale_reduction_sum / self.limited as f64
        };
        diagnostics.max_scale_reduction = self.scale_reduction_max;
        for (family_diagnostics, accumulator) in diagnostics.families.iter_mut().zip(self.families)
        {
            family_diagnostics.matched_pixel_ratio = accumulator.matched as f64 / denominator;
            family_diagnostics.limited_pixel_ratio = accumulator.limited as f64 / denominator;
            family_diagnostics.mean_scale_reduction = if accumulator.limited == 0 {
                0.0
            } else {
                accumulator.scale_reduction_sum / accumulator.limited as f64
            };
            family_diagnostics.max_scale_reduction = accumulator.scale_reduction_max;
        }
        diagnostics
    }

    fn merge(&mut self, other: Self) {
        self.evaluated += other.evaluated;
        self.matched += other.matched;
        self.limited += other.limited;
        self.scale_reduction_sum += other.scale_reduction_sum;
        self.scale_reduction_max = self.scale_reduction_max.max(other.scale_reduction_max);
        for (left, right) in self.families.iter_mut().zip(other.families) {
            left.matched += right.matched;
            left.limited += right.limited;
            left.scale_reduction_sum += right.scale_reduction_sum;
            left.scale_reduction_max = left.scale_reduction_max.max(right.scale_reduction_max);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PreferredSkinRenderingDecision {
    output_rgb: [f64; 3],
    output_lab: [f64; 3],
    matched: bool,
    outside_preferred_core: bool,
    adjusted: bool,
    gamut_limited: bool,
    delta_e_ab: f64,
    abs_hue_shift_degrees: f64,
    chroma_delta: f64,
}

impl PreferredSkinRenderingDecision {
    fn unchanged(
        rgb: [f64; 3],
        lab: [f64; 3],
        matched: bool,
        outside_preferred_core: bool,
    ) -> Self {
        Self {
            output_rgb: rgb,
            output_lab: lab,
            matched,
            outside_preferred_core,
            adjusted: false,
            gamut_limited: false,
            delta_e_ab: 0.0,
            abs_hue_shift_degrees: 0.0,
            chroma_delta: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct PreferredSkinRenderingAccumulator {
    evaluated: usize,
    matched: usize,
    outside_preferred_core: usize,
    adjusted: usize,
    gamut_limited: usize,
    delta_e_ab_sum: f64,
    delta_e_ab_max: f64,
    abs_hue_shift_degrees_sum: f64,
    abs_hue_shift_degrees_max: f64,
    chroma_delta_sum: f64,
    abs_chroma_delta_max: f64,
}

impl PreferredSkinRenderingAccumulator {
    fn record(&mut self, decision: PreferredSkinRenderingDecision) {
        self.evaluated += 1;
        if decision.matched {
            self.matched += 1;
        }
        if decision.outside_preferred_core {
            self.outside_preferred_core += 1;
        }
        if !decision.adjusted {
            return;
        }
        self.adjusted += 1;
        self.gamut_limited += usize::from(decision.gamut_limited);
        self.delta_e_ab_sum += decision.delta_e_ab;
        self.delta_e_ab_max = self.delta_e_ab_max.max(decision.delta_e_ab);
        self.abs_hue_shift_degrees_sum += decision.abs_hue_shift_degrees;
        self.abs_hue_shift_degrees_max = self
            .abs_hue_shift_degrees_max
            .max(decision.abs_hue_shift_degrees);
        self.chroma_delta_sum += decision.chroma_delta;
        self.abs_chroma_delta_max = self.abs_chroma_delta_max.max(decision.chroma_delta.abs());
    }

    fn finish(self, denominator: f64) -> PreferredSkinRenderingDiagnostics {
        let mut diagnostics = PreferredSkinRenderingDiagnostics::empty(
            true,
            "trusted modern-clean render applied a bounded one-way excess-chroma preference-ellipse shoulder while preserving CIELAB lightness",
        );
        diagnostics.evaluated_pixel_ratio = self.evaluated as f64 / denominator;
        diagnostics.matched_pixel_ratio = self.matched as f64 / denominator;
        diagnostics.outside_preferred_core_ratio = self.outside_preferred_core as f64 / denominator;
        diagnostics.adjusted_pixel_ratio = self.adjusted as f64 / denominator;
        diagnostics.gamut_limited_pixel_ratio = self.gamut_limited as f64 / denominator;
        if self.adjusted > 0 {
            let adjusted = self.adjusted as f64;
            diagnostics.mean_delta_e_ab = self.delta_e_ab_sum / adjusted;
            diagnostics.max_delta_e_ab = self
                .delta_e_ab_max
                .min(PREFERRED_SKIN_RENDERING_MAX_DELTA_E_AB);
            diagnostics.mean_abs_hue_shift_degrees = self.abs_hue_shift_degrees_sum / adjusted;
            diagnostics.max_abs_hue_shift_degrees = self.abs_hue_shift_degrees_max;
            diagnostics.mean_chroma_delta = self.chroma_delta_sum / adjusted;
            diagnostics.max_abs_chroma_delta = self.abs_chroma_delta_max;
        }
        diagnostics
    }

    fn merge(&mut self, other: Self) {
        self.evaluated += other.evaluated;
        self.matched += other.matched;
        self.outside_preferred_core += other.outside_preferred_core;
        self.adjusted += other.adjusted;
        self.gamut_limited += other.gamut_limited;
        self.delta_e_ab_sum += other.delta_e_ab_sum;
        self.delta_e_ab_max = self.delta_e_ab_max.max(other.delta_e_ab_max);
        self.abs_hue_shift_degrees_sum += other.abs_hue_shift_degrees_sum;
        self.abs_hue_shift_degrees_max = self
            .abs_hue_shift_degrees_max
            .max(other.abs_hue_shift_degrees_max);
        self.chroma_delta_sum += other.chroma_delta_sum;
        self.abs_chroma_delta_max = self.abs_chroma_delta_max.max(other.abs_chroma_delta_max);
    }
}

#[derive(Debug, Clone, Default)]
struct AdaptiveVibranceAccumulator {
    applied: usize,
    texture_limited: usize,
    gamut_limited: usize,
    skin_memory_evaluated: usize,
    skin_memory_protected: usize,
    skin_memory_protection_sum: f64,
    skin_memory_protection_max: f64,
    preferred_memory_color_guard: PreferredMemoryColorGuardAccumulator,
    preferred_skin_rendering: PreferredSkinRenderingAccumulator,
    sum_scale: f64,
    max_applied_scale: f64,
}

impl AdaptiveVibranceAccumulator {
    fn merge(&mut self, other: Self) {
        self.applied += other.applied;
        self.texture_limited += other.texture_limited;
        self.gamut_limited += other.gamut_limited;
        self.skin_memory_evaluated += other.skin_memory_evaluated;
        self.skin_memory_protected += other.skin_memory_protected;
        self.skin_memory_protection_sum += other.skin_memory_protection_sum;
        self.skin_memory_protection_max = self
            .skin_memory_protection_max
            .max(other.skin_memory_protection_max);
        self.preferred_memory_color_guard
            .merge(other.preferred_memory_color_guard);
        self.preferred_skin_rendering
            .merge(other.preferred_skin_rendering);
        self.sum_scale += other.sum_scale;
        self.max_applied_scale = self.max_applied_scale.max(other.max_applied_scale);
    }
}

fn lab_to_prophoto_rgb(
    lab: [f64; 3],
    xyz_to_prophoto: &nalgebra::Matrix3<f64>,
) -> Option<[f64; 3]> {
    if lab.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let xyz = Vector3::from(crate::colorspace::lab_to_xyz_d50(lab));
    let rgb = xyz_to_prophoto * xyz;
    let rgb = [rgb[0], rgb[1], rgb[2]];
    rgb.iter().all(|value| value.is_finite()).then_some(rgb)
}

fn unit_rgb_in_gamut(rgb: [f64; 3]) -> bool {
    rgb.iter()
        .all(|value| value.is_finite() && *value >= -1e-10 && *value <= 1.0 + 1e-10)
}

fn shortest_hue_delta_degrees(from: f64, to: f64) -> f64 {
    (to - from + 180.0).rem_euclid(360.0) - 180.0
}

fn preferred_skin_rendering_decision(
    rgb: [f64; 3],
    lab: [f64; 3],
    xyz_to_prophoto: &nalgebra::Matrix3<f64>,
) -> PreferredSkinRenderingDecision {
    let support_weight = skin_memory_protection_weight_from_lab(lab);
    let matched = support_weight >= PREFERRED_SKIN_RENDERING_MIN_SUPPORT_WEIGHT;
    if !matched {
        return PreferredSkinRenderingDecision::unchanged(rgb, lab, false, false);
    }

    let radius = preferred_memory_color_normalized_radius(lab, PREFERRED_SKIN_RENDERING_MODEL);
    let outside_preferred_core = radius > PREFERRED_SKIN_RENDERING_CORE_RADIUS + 1e-12;
    if !outside_preferred_core || !radius.is_finite() {
        return PreferredSkinRenderingDecision::unchanged(rgb, lab, true, false);
    }

    let center = PREFERRED_SKIN_RENDERING_MODEL.preferred_center_lab();
    let before_chroma = lab[1].hypot(lab[2]);
    if before_chroma <= PREFERRED_SKIN_RENDERING_MODEL.preferred_center_lch[1] + 1e-12 {
        return PreferredSkinRenderingDecision::unchanged(rgb, lab, true, true);
    }
    let delta_a = center[1] - lab[1];
    let delta_b = center[2] - lab[2];
    let distance_ab = delta_a.hypot(delta_b);
    if distance_ab <= 1e-12 {
        return PreferredSkinRenderingDecision::unchanged(rgb, lab, true, true);
    }

    let fraction_to_core =
        ((radius - PREFERRED_SKIN_RENDERING_CORE_RADIUS) / radius).clamp(0.0, 1.0);
    let mut requested_fraction =
        PREFERRED_SKIN_RENDERING_RADIAL_EXCESS_REDUCTION * support_weight * fraction_to_core;
    let requested_delta_e_ab = distance_ab * requested_fraction;
    if requested_delta_e_ab > PREFERRED_SKIN_RENDERING_MAX_DELTA_E_AB {
        requested_fraction *= PREFERRED_SKIN_RENDERING_MAX_DELTA_E_AB / requested_delta_e_ab;
    }
    if distance_ab * requested_fraction < PREFERRED_SKIN_RENDERING_MIN_DELTA_E_AB {
        return PreferredSkinRenderingDecision::unchanged(rgb, lab, true, true);
    }

    let candidate_for_fraction = |fraction: f64| {
        let candidate_lab = [
            lab[0],
            lab[1] + delta_a * fraction,
            lab[2] + delta_b * fraction,
        ];
        lab_to_prophoto_rgb(candidate_lab, xyz_to_prophoto)
            .map(|candidate| (candidate_lab, candidate))
    };

    let mut applied_fraction = requested_fraction;
    let requested_candidate = candidate_for_fraction(requested_fraction);
    let requested_in_gamut = requested_candidate
        .as_ref()
        .is_some_and(|(_, candidate)| unit_rgb_in_gamut(*candidate));
    if !requested_in_gamut {
        let mut low = 0.0;
        let mut high = requested_fraction;
        for _ in 0..22 {
            let mid = (low + high) * 0.5;
            if candidate_for_fraction(mid)
                .as_ref()
                .is_some_and(|(_, candidate)| unit_rgb_in_gamut(*candidate))
            {
                low = mid;
            } else {
                high = mid;
            }
        }
        applied_fraction = low * 0.999_999;
    }

    let Some((output_lab, output_rgb)) = candidate_for_fraction(applied_fraction) else {
        return PreferredSkinRenderingDecision::unchanged(rgb, lab, true, true);
    };
    let delta_e_ab = distance_ab * applied_fraction;
    if delta_e_ab < PREFERRED_SKIN_RENDERING_MIN_DELTA_E_AB {
        return PreferredSkinRenderingDecision::unchanged(rgb, lab, true, true);
    }

    let after_chroma = output_lab[1].hypot(output_lab[2]);
    if after_chroma > before_chroma + 1e-10 {
        return PreferredSkinRenderingDecision::unchanged(rgb, lab, true, true);
    }
    let before_hue = lab[2].atan2(lab[1]).to_degrees().rem_euclid(360.0);
    let after_hue = output_lab[2]
        .atan2(output_lab[1])
        .to_degrees()
        .rem_euclid(360.0);
    PreferredSkinRenderingDecision {
        output_rgb: output_rgb.map(|value| value.clamp(0.0, 1.0)),
        output_lab,
        matched: true,
        outside_preferred_core: true,
        adjusted: true,
        gamut_limited: applied_fraction + 1e-9 < requested_fraction,
        delta_e_ab,
        abs_hue_shift_degrees: shortest_hue_delta_degrees(before_hue, after_hue).abs(),
        chroma_delta: after_chroma - before_chroma,
    }
}

fn apply_preferred_skin_rendering(
    mut img: Array3<f64>,
    color_protection: &ToneColorProtection,
) -> (Array3<f64>, PreferredSkinRenderingDiagnostics) {
    let (height, width, channels) = img.dim();
    if color_protection.color_trust_state() != "trusted" {
        return (
            img,
            PreferredSkinRenderingDiagnostics::empty(
                false,
                format!(
                    "disabled because tone color trust state is {}",
                    color_protection.color_trust_state()
                ),
            ),
        );
    }
    if channels < 3 {
        return (
            img,
            PreferredSkinRenderingDiagnostics::empty(
                false,
                "preferred-skin rendering requires at least three channels",
            ),
        );
    }

    let prophoto_to_xyz = crate::colorspace::prophoto_to_xyz_d50_matrix();
    let xyz_to_prophoto = crate::colorspace::xyz_d50_to_prophoto_matrix();
    let accumulator = img
        .axis_chunks_iter_mut(Axis(0), 128)
        .into_par_iter()
        .map(|mut chunk| {
            let mut accumulator = PreferredSkinRenderingAccumulator::default();
            let chunk_height = chunk.dim().0;
            for y in 0..chunk_height {
                for x in 0..width {
                    let rgb = [
                        chunk[[y, x, 0]].clamp(0.0, 1.0),
                        chunk[[y, x, 1]].clamp(0.0, 1.0),
                        chunk[[y, x, 2]].clamp(0.0, 1.0),
                    ];
                    let luminance = linear_luminance(rgb[0], rgb[1], rgb[2]);
                    if !(0.015..=0.82).contains(&luminance) || rgb_saturation(rgb) < 0.015 {
                        continue;
                    }
                    let Some(lab) = prophoto_rgb_to_lab(rgb, &prophoto_to_xyz) else {
                        continue;
                    };
                    let decision = preferred_skin_rendering_decision(rgb, lab, &xyz_to_prophoto);
                    accumulator.record(decision);
                    if decision.adjusted {
                        for channel in 0..3 {
                            chunk[[y, x, channel]] = decision.output_rgb[channel];
                        }
                    }
                }
            }
            accumulator
        })
        .reduce(
            PreferredSkinRenderingAccumulator::default,
            |mut left, right| {
                left.merge(right);
                left
            },
        );

    let denominator = height.saturating_mul(width).max(1) as f64;
    (img, accumulator.finish(denominator))
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
    img: Array3<f64>,
    color_protection: &ToneColorProtection,
) -> (Array3<f64>, AdaptiveVibranceDiagnostics) {
    let (img, diagnostics, _) = apply_adaptive_vibrance_internal(img, color_protection, false);
    (img, diagnostics)
}

fn apply_adaptive_vibrance_with_preferred_skin(
    img: Array3<f64>,
    color_protection: &ToneColorProtection,
) -> (
    Array3<f64>,
    AdaptiveVibranceDiagnostics,
    PreferredSkinRenderingDiagnostics,
) {
    apply_adaptive_vibrance_internal(img, color_protection, true)
}

fn apply_adaptive_vibrance_internal(
    mut img: Array3<f64>,
    color_protection: &ToneColorProtection,
    preferred_skin_requested: bool,
) -> (
    Array3<f64>,
    AdaptiveVibranceDiagnostics,
    PreferredSkinRenderingDiagnostics,
) {
    let (height, width, channels) = img.dim();
    if color_protection.color_trust_state() != "trusted" {
        return (
            img,
            AdaptiveVibranceDiagnostics::disabled(format!(
                "disabled because tone color trust state is {}",
                color_protection.color_trust_state()
            )),
            PreferredSkinRenderingDiagnostics::empty(
                false,
                format!(
                    "disabled because tone color trust state is {}",
                    color_protection.color_trust_state()
                ),
            ),
        );
    }
    if height < ADAPTIVE_VIBRANCE_MIN_DIMENSION || width < ADAPTIVE_VIBRANCE_MIN_DIMENSION {
        let adaptive = AdaptiveVibranceDiagnostics::disabled(
            "image is smaller than the adaptive vibrance minimum dimension",
        );
        if preferred_skin_requested {
            let (img, preferred_skin) = apply_preferred_skin_rendering(img, color_protection);
            return (img, adaptive, preferred_skin);
        }
        return (
            img,
            adaptive,
            PreferredSkinRenderingDiagnostics::empty(
                false,
                "preferred-skin rendering was not requested",
            ),
        );
    }
    if channels < 3 {
        return (
            img,
            AdaptiveVibranceDiagnostics::disabled(
                "adaptive vibrance requires at least three channels",
            ),
            PreferredSkinRenderingDiagnostics::empty(
                false,
                "preferred-skin rendering requires at least three channels",
            ),
        );
    }

    let pixel_count = height * width;
    let prophoto_to_xyz = crate::colorspace::prophoto_to_xyz_d50_matrix();
    let xyz_to_prophoto = crate::colorspace::xyz_d50_to_prophoto_matrix();
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

    let chunk_accumulators: Vec<AdaptiveVibranceAccumulator> = img
        .axis_chunks_iter_mut(Axis(0), 128)
        .into_par_iter()
        .enumerate()
        .map(|(chunk_index, mut chunk)| {
            let row_start = chunk_index * 128;
            let chunk_height = chunk.dim().0;
            let mut accumulator = AdaptiveVibranceAccumulator::default();
            for local_y in 0..chunk_height {
                let y = row_start + local_y;
                for x in 0..width {
                    let idx = y * width + x;
                    let source_rgb = [
                        chunk[[local_y, x, 0]].clamp(0.0, 1.0),
                        chunk[[local_y, x, 1]].clamp(0.0, 1.0),
                        chunk[[local_y, x, 2]].clamp(0.0, 1.0),
                    ];
                    let luminance = luminance[idx] as f64;
                    let source_saturation = rgb_saturation(source_rgb);
                    let texture = (luminance - blurred_luminance[idx] as f64).abs();
                    let texture_gate = 1.0
                        - smoothstep_range(
                            ADAPTIVE_VIBRANCE_TEXTURE_START,
                            ADAPTIVE_VIBRANCE_TEXTURE_END,
                            texture,
                        );
                    let source_neutral_gate = smoothstep_range(
                        ADAPTIVE_VIBRANCE_NEUTRAL_START,
                        ADAPTIVE_VIBRANCE_NEUTRAL_FULL,
                        source_saturation,
                    );
                    let source_saturation_gate = 1.0
                        - smoothstep_range(
                            ADAPTIVE_VIBRANCE_SATURATION_START,
                            ADAPTIVE_VIBRANCE_SATURATION_END,
                            source_saturation,
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
                    let source_unprotected_boost = ADAPTIVE_VIBRANCE_AMOUNT
                        * source_neutral_gate
                        * source_saturation_gate
                        * luminance_gate
                        * texture_gate;
                    let preferred_skin_candidate = preferred_skin_requested
                        && (0.015..=0.82).contains(&luminance)
                        && source_saturation >= 0.015;
                    let source_lab = if source_unprotected_boost > 1e-12 || preferred_skin_candidate
                    {
                        prophoto_rgb_to_lab(source_rgb, &prophoto_to_xyz)
                    } else {
                        None
                    };
                    let preferred_skin_decision = if preferred_skin_candidate {
                        source_lab.map(|lab| {
                            preferred_skin_rendering_decision(source_rgb, lab, &xyz_to_prophoto)
                        })
                    } else {
                        None
                    };
                    if let Some(decision) = preferred_skin_decision {
                        accumulator.preferred_skin_rendering.record(decision);
                    }
                    let (rgb, before_lab) = preferred_skin_decision
                        .filter(|decision| decision.adjusted)
                        .map(|decision| (decision.output_rgb, Some(decision.output_lab)))
                        .unwrap_or((source_rgb, source_lab));
                    let unprotected_boost =
                        if preferred_skin_decision.is_some_and(|decision| decision.adjusted) {
                            let saturation = rgb_saturation(rgb);
                            ADAPTIVE_VIBRANCE_AMOUNT
                                * smoothstep_range(
                                    ADAPTIVE_VIBRANCE_NEUTRAL_START,
                                    ADAPTIVE_VIBRANCE_NEUTRAL_FULL,
                                    saturation,
                                )
                                * (1.0
                                    - smoothstep_range(
                                        ADAPTIVE_VIBRANCE_SATURATION_START,
                                        ADAPTIVE_VIBRANCE_SATURATION_END,
                                        saturation,
                                    ))
                                * luminance_gate
                                * texture_gate
                        } else {
                            source_unprotected_boost
                        };
                    if unprotected_boost > 1e-12 {
                        accumulator.skin_memory_evaluated += 1;
                    }
                    let skin_memory_weight = before_lab
                        .filter(|_| unprotected_boost > 1e-12)
                        .map(skin_memory_protection_weight_from_lab)
                        .unwrap_or(0.0);
                    if skin_memory_weight > 1e-12 {
                        accumulator.skin_memory_protected += 1;
                        accumulator.skin_memory_protection_sum += skin_memory_weight;
                        accumulator.skin_memory_protection_max = accumulator
                            .skin_memory_protection_max
                            .max(skin_memory_weight);
                    }
                    let skin_memory_gate =
                        1.0 - ADAPTIVE_VIBRANCE_SKIN_MAXIMUM_REDUCTION * skin_memory_weight;
                    let requested_scale = (1.0 + unprotected_boost * skin_memory_gate)
                        .min(ADAPTIVE_VIBRANCE_MAX_SCALE);
                    let gamut_scale = max_chroma_scale_inside_gamut(rgb, luminance);
                    let gamut_bounded_scale = requested_scale.min(gamut_scale);
                    let scale = if gamut_bounded_scale > 1.0 + 1e-12 {
                        before_lab
                            .and_then(|before_lab| {
                                let matched = preferred_memory_color_match(before_lab);
                                let decision = if matched.is_some() {
                                    let tentative_rgb = std::array::from_fn(|c| {
                                        (luminance + (rgb[c] - luminance) * gamut_bounded_scale)
                                            .clamp(0.0, 1.0)
                                    });
                                    let tentative_lab =
                                        prophoto_rgb_to_lab(tentative_rgb, &prophoto_to_xyz)?;
                                    preferred_memory_color_guard_scale_with_match(
                                        before_lab,
                                        tentative_lab,
                                        gamut_bounded_scale,
                                        matched,
                                    )
                                } else {
                                    PreferredMemoryColorGuardDecision {
                                        scale: gamut_bounded_scale,
                                        family_index: None,
                                        support_weight: 0.0,
                                        scale_reduction: 0.0,
                                    }
                                };
                                accumulator.preferred_memory_color_guard.record(decision);
                                Some(decision.scale)
                            })
                            .unwrap_or(gamut_bounded_scale)
                    } else {
                        gamut_bounded_scale
                    };
                    if texture_gate < 0.5 {
                        accumulator.texture_limited += 1;
                    }
                    if requested_scale > gamut_scale + 1e-12 {
                        accumulator.gamut_limited += 1;
                    }
                    if scale > 1.002 {
                        accumulator.applied += 1;
                        accumulator.sum_scale += scale;
                        accumulator.max_applied_scale = accumulator.max_applied_scale.max(scale);
                    }

                    for c in 0..channels {
                        if c < 3 {
                            chunk[[local_y, x, c]] =
                                (luminance + (rgb[c] - luminance) * scale).clamp(0.0, 1.0);
                        }
                    }
                }
            }
            accumulator
        })
        .collect();
    let mut accumulator = AdaptiveVibranceAccumulator::default();
    for chunk_accumulator in chunk_accumulators {
        accumulator.merge(chunk_accumulator);
    }

    let denom = height.saturating_mul(width).max(1) as f64;
    let preferred_skin_diagnostics = if preferred_skin_requested {
        accumulator.preferred_skin_rendering.finish(denom)
    } else {
        PreferredSkinRenderingDiagnostics::empty(
            false,
            "preferred-skin rendering was not requested",
        )
    };
    (
        img,
        AdaptiveVibranceDiagnostics {
            enabled: true,
            reason: "trusted color path received bounded hue-preserving adaptive vibrance with feathered D50 CIELAB skin-memory protection and a one-way published sky/grass preference-ellipse overshoot guard".to_string(),
            amount: ADAPTIVE_VIBRANCE_AMOUNT,
            max_scale: ADAPTIVE_VIBRANCE_MAX_SCALE,
            applied_ratio: accumulator.applied as f64 / denom,
            mean_scale: if accumulator.applied == 0 {
                1.0
            } else {
                accumulator.sum_scale / accumulator.applied as f64
            },
            max_applied_scale: accumulator.max_applied_scale.max(1.0),
            texture_limited_ratio: accumulator.texture_limited as f64 / denom,
            gamut_limited_ratio: accumulator.gamut_limited as f64 / denom,
            skin_memory_protection: AdaptiveVibranceSkinMemoryProtectionDiagnostics {
                enabled: true,
                evaluated_pixel_ratio: accumulator.skin_memory_evaluated as f64 / denom,
                protected_pixel_ratio: accumulator.skin_memory_protected as f64 / denom,
                mean_protection_weight: if accumulator.skin_memory_protected == 0 {
                    0.0
                } else {
                    accumulator.skin_memory_protection_sum
                        / accumulator.skin_memory_protected as f64
                },
                max_protection_weight: accumulator.skin_memory_protection_max,
                ..AdaptiveVibranceSkinMemoryProtectionDiagnostics::empty(true)
            },
            preferred_memory_color_guard: accumulator
                .preferred_memory_color_guard
                .finish(denom),
        },
        preferred_skin_diagnostics,
    )
}

#[derive(Debug, Clone, Copy)]
enum GrainDetailAxis {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy)]
struct GrainLuminanceProbe {
    y: usize,
    x: usize,
    radius: usize,
    axis: GrainDetailAxis,
    pre_delta: f64,
}

#[derive(Debug, Clone, Copy)]
struct GrainChromaProbe {
    y: usize,
    x: usize,
    radius: usize,
    axis: GrainDetailAxis,
    pre_delta: [f64; 2],
}

fn scalar_axis_delta(
    values: &[f32],
    width: usize,
    y: usize,
    x: usize,
    radius: usize,
    axis: GrainDetailAxis,
) -> f64 {
    match axis {
        GrainDetailAxis::Horizontal => {
            values[y * width + x + radius] as f64 - values[y * width + x - radius] as f64
        }
        GrainDetailAxis::Vertical => {
            values[(y + radius) * width + x] as f64 - values[(y - radius) * width + x] as f64
        }
    }
}

fn opponent_axis_delta(
    opponent_a: &[f32],
    opponent_b: &[f32],
    width: usize,
    y: usize,
    x: usize,
    radius: usize,
    axis: GrainDetailAxis,
) -> [f64; 2] {
    [
        scalar_axis_delta(opponent_a, width, y, x, radius, axis),
        scalar_axis_delta(opponent_b, width, y, x, radius, axis),
    ]
}

fn opponent_norm(value: [f64; 2]) -> f64 {
    ((value[0] * value[0] + value[1] * value[1]) * 0.5).sqrt()
}

fn scalar_multiscale_coherent(inner: f64, outer: f64) -> bool {
    let inner_abs = inner.abs();
    let outer_abs = outer.abs();
    inner * outer > 0.0
        && inner_abs.min(outer_abs) >= inner_abs.max(outer_abs) * GRAIN_DETAIL_SCALE_AGREEMENT_MIN
}

fn opponent_multiscale_coherent(inner: [f64; 2], outer: [f64; 2]) -> bool {
    let inner_norm = opponent_norm(inner);
    let outer_norm = opponent_norm(outer);
    if inner_norm <= 1e-12 || outer_norm <= 1e-12 {
        return false;
    }
    let cosine = (inner[0] * outer[0] + inner[1] * outer[1])
        / ((inner[0].hypot(inner[1])) * (outer[0].hypot(outer[1]))).max(1e-12);
    cosine >= GRAIN_DETAIL_COHERENCE_MIN
        && inner_norm.min(outer_norm)
            >= inner_norm.max(outer_norm) * GRAIN_DETAIL_SCALE_AGREEMENT_MIN
}

fn populate_coherent_chroma_structure(
    opponent_a: &[f32],
    opponent_b: &[f32],
    structure: &mut [f32],
    height: usize,
    width: usize,
    radius: usize,
) {
    let outer_radius = radius.saturating_mul(2);
    if outer_radius == 0 || height <= outer_radius * 2 || width <= outer_radius * 2 {
        structure.fill(0.0);
        return;
    }
    structure.fill(0.0);
    for y in outer_radius..(height - outer_radius) {
        for x in outer_radius..(width - outer_radius) {
            let mut coherent_structure = 0.0f64;
            for axis in [GrainDetailAxis::Horizontal, GrainDetailAxis::Vertical] {
                let inner = opponent_axis_delta(opponent_a, opponent_b, width, y, x, radius, axis);
                let outer =
                    opponent_axis_delta(opponent_a, opponent_b, width, y, x, outer_radius, axis);
                if opponent_multiscale_coherent(inner, outer) {
                    coherent_structure = coherent_structure.max(opponent_norm(inner));
                }
            }
            structure[y * width + x] = coherent_structure as f32;
        }
    }
}

fn dilate_structure_max(
    source: &[f32],
    horizontal: &mut [f32],
    destination: &mut [f32],
    height: usize,
    width: usize,
    radius: usize,
) {
    if radius == 0 {
        destination.copy_from_slice(source);
        return;
    }
    for y in 0..height {
        for x in 0..width {
            let start = x.saturating_sub(radius);
            let end = (x + radius + 1).min(width);
            horizontal[y * width + x] = (start..end)
                .map(|sample_x| source[y * width + sample_x])
                .fold(0.0f32, f32::max);
        }
    }
    for y in 0..height {
        let start = y.saturating_sub(radius);
        let end = (y + radius + 1).min(height);
        for x in 0..width {
            destination[y * width + x] = (start..end)
                .map(|sample_y| horizontal[sample_y * width + x])
                .fold(0.0f32, f32::max);
        }
    }
}

fn collect_grain_detail_probes(
    luminance: &[f32],
    opponent_a: &[f32],
    opponent_b: &[f32],
    height: usize,
    width: usize,
    radius: usize,
) -> (usize, Vec<GrainLuminanceProbe>, Vec<GrainChromaProbe>) {
    let outer_radius = radius.saturating_mul(2);
    if outer_radius == 0 || height <= outer_radius * 2 || width <= outer_radius * 2 {
        return (1, Vec::new(), Vec::new());
    }
    let interior_height = height - outer_radius * 2;
    let interior_width = width - outer_radius * 2;
    let interior_pixels = interior_height.saturating_mul(interior_width);
    let sample_stride = interior_pixels.div_ceil(GRAIN_DETAIL_MAX_PROBES).max(1);
    let capacity = interior_pixels
        .div_ceil(sample_stride)
        .min(GRAIN_DETAIL_MAX_PROBES);
    let mut luminance_probes = Vec::with_capacity(capacity / 2);
    let mut chroma_probes = Vec::with_capacity(capacity / 2);

    for y in outer_radius..(height - outer_radius) {
        for x in outer_radius..(width - outer_radius) {
            let sample_index = (y - outer_radius) * interior_width + (x - outer_radius);
            if !sample_index.is_multiple_of(sample_stride) {
                continue;
            }

            let horizontal_luma =
                scalar_axis_delta(luminance, width, y, x, radius, GrainDetailAxis::Horizontal);
            let vertical_luma =
                scalar_axis_delta(luminance, width, y, x, radius, GrainDetailAxis::Vertical);
            let (luma_axis, luma_inner) = if horizontal_luma.abs() >= vertical_luma.abs() {
                (GrainDetailAxis::Horizontal, horizontal_luma)
            } else {
                (GrainDetailAxis::Vertical, vertical_luma)
            };
            let luma_outer = scalar_axis_delta(luminance, width, y, x, outer_radius, luma_axis);
            if luma_inner.abs() >= GRAIN_DETAIL_LUMA_CONTRAST_MIN
                && luma_outer.abs() >= GRAIN_DETAIL_LUMA_CONTRAST_MIN
                && scalar_multiscale_coherent(luma_inner, luma_outer)
            {
                luminance_probes.push(GrainLuminanceProbe {
                    y,
                    x,
                    radius,
                    axis: luma_axis,
                    pre_delta: luma_inner,
                });
            }

            let horizontal_chroma = opponent_axis_delta(
                opponent_a,
                opponent_b,
                width,
                y,
                x,
                radius,
                GrainDetailAxis::Horizontal,
            );
            let vertical_chroma = opponent_axis_delta(
                opponent_a,
                opponent_b,
                width,
                y,
                x,
                radius,
                GrainDetailAxis::Vertical,
            );
            let (chroma_axis, chroma_inner) =
                if opponent_norm(horizontal_chroma) >= opponent_norm(vertical_chroma) {
                    (GrainDetailAxis::Horizontal, horizontal_chroma)
                } else {
                    (GrainDetailAxis::Vertical, vertical_chroma)
                };
            let chroma_outer = opponent_axis_delta(
                opponent_a,
                opponent_b,
                width,
                y,
                x,
                outer_radius,
                chroma_axis,
            );
            if opponent_norm(chroma_inner) >= GRAIN_DETAIL_CHROMA_CONTRAST_MIN
                && opponent_norm(chroma_outer) >= GRAIN_DETAIL_CHROMA_CONTRAST_MIN
                && opponent_multiscale_coherent(chroma_inner, chroma_outer)
            {
                chroma_probes.push(GrainChromaProbe {
                    y,
                    x,
                    radius,
                    axis: chroma_axis,
                    pre_delta: chroma_inner,
                });
            }
        }
    }

    (sample_stride, luminance_probes, chroma_probes)
}

fn image_luminance_at(img: &Array3<f64>, y: usize, x: usize) -> f64 {
    linear_luminance(
        img[[y, x, 0]].clamp(0.0, 1.0),
        img[[y, x, 1]].clamp(0.0, 1.0),
        img[[y, x, 2]].clamp(0.0, 1.0),
    )
}

fn image_opponent_at(img: &Array3<f64>, y: usize, x: usize) -> [f64; 2] {
    let red = img[[y, x, 0]].clamp(0.0, 1.0);
    let green = img[[y, x, 1]].clamp(0.0, 1.0);
    let blue = img[[y, x, 2]].clamp(0.0, 1.0);
    [red - green, blue - 0.5 * (red + green)]
}

fn post_luminance_probe_delta(img: &Array3<f64>, probe: GrainLuminanceProbe) -> f64 {
    match probe.axis {
        GrainDetailAxis::Horizontal => {
            image_luminance_at(img, probe.y, probe.x + probe.radius)
                - image_luminance_at(img, probe.y, probe.x - probe.radius)
        }
        GrainDetailAxis::Vertical => {
            image_luminance_at(img, probe.y + probe.radius, probe.x)
                - image_luminance_at(img, probe.y - probe.radius, probe.x)
        }
    }
}

fn post_chroma_probe_delta(img: &Array3<f64>, probe: GrainChromaProbe) -> [f64; 2] {
    let (positive, negative) = match probe.axis {
        GrainDetailAxis::Horizontal => (
            image_opponent_at(img, probe.y, probe.x + probe.radius),
            image_opponent_at(img, probe.y, probe.x - probe.radius),
        ),
        GrainDetailAxis::Vertical => (
            image_opponent_at(img, probe.y + probe.radius, probe.x),
            image_opponent_at(img, probe.y - probe.radius, probe.x),
        ),
    };
    [positive[0] - negative[0], positive[1] - negative[1]]
}

fn evaluate_grain_detail_retention(
    img: &Array3<f64>,
    sample_stride: usize,
    probe_radius: usize,
    luminance_probes: &[GrainLuminanceProbe],
    chroma_probes: &[GrainChromaProbe],
) -> GrainDetailRetentionDiagnostics {
    let mut luminance_retention = luminance_probes
        .iter()
        .map(|probe| {
            (post_luminance_probe_delta(img, *probe) / probe.pre_delta)
                .clamp(0.0, GRAIN_DETAIL_RETENTION_RATIO_MAX)
        })
        .collect::<Vec<_>>();
    let mut chroma_retention = chroma_probes
        .iter()
        .map(|probe| {
            let post = post_chroma_probe_delta(img, *probe);
            let pre_norm_squared =
                probe.pre_delta[0] * probe.pre_delta[0] + probe.pre_delta[1] * probe.pre_delta[1];
            ((post[0] * probe.pre_delta[0] + post[1] * probe.pre_delta[1])
                / pre_norm_squared.max(1e-12))
            .clamp(0.0, GRAIN_DETAIL_RETENTION_RATIO_MAX)
        })
        .collect::<Vec<_>>();
    luminance_retention.sort_by(|a, b| a.total_cmp(b));
    chroma_retention.sort_by(|a, b| a.total_cmp(b));

    let luminance_decision_supported = luminance_retention.len() >= GRAIN_DETAIL_MIN_PROBE_COUNT;
    let chroma_decision_supported = chroma_retention.len() >= GRAIN_DETAIL_MIN_PROBE_COUNT;
    let decision_supported = luminance_decision_supported || chroma_decision_supported;
    let luminance_median_retention = if luminance_retention.is_empty() {
        1.0
    } else {
        percentile_from_sorted_values(&luminance_retention, 0.50)
    };
    let luminance_p10_retention = if luminance_retention.is_empty() {
        1.0
    } else {
        percentile_from_sorted_values(&luminance_retention, 0.10)
    };
    let chroma_median_retention = if chroma_retention.is_empty() {
        1.0
    } else {
        percentile_from_sorted_values(&chroma_retention, 0.50)
    };
    let chroma_p10_retention = if chroma_retention.is_empty() {
        1.0
    } else {
        percentile_from_sorted_values(&chroma_retention, 0.10)
    };

    let mut failures = Vec::<String>::new();
    if luminance_decision_supported
        && (luminance_median_retention < GRAIN_DETAIL_MEDIAN_RETENTION_MIN
            || luminance_p10_retention < GRAIN_DETAIL_P10_RETENTION_MIN)
    {
        failures.push(format!(
            "coherent luminance detail retention fell to median {:.3}, p10 {:.3}",
            luminance_median_retention, luminance_p10_retention
        ));
    }
    if chroma_decision_supported
        && (chroma_median_retention < GRAIN_DETAIL_MEDIAN_RETENTION_MIN
            || chroma_p10_retention < GRAIN_DETAIL_P10_RETENTION_MIN)
    {
        failures.push(format!(
            "coherent opponent-color detail retention fell to median {:.3}, p10 {:.3}",
            chroma_median_retention, chroma_p10_retention
        ));
    }
    let review_required = !failures.is_empty();
    let reason = if review_required {
        format!(
            "grain reduction exceeded conservative structured-detail loss limits: {}",
            failures.join("; ")
        )
    } else if decision_supported {
        format!(
            "coherent pre/post detail probes passed; luminance support={}, chroma support={}",
            luminance_decision_supported, chroma_decision_supported
        )
    } else {
        "no sufficiently strong multiscale-coherent edge population was present; detail retention is reported as unsupported rather than trusted".to_string()
    };

    GrainDetailRetentionDiagnostics {
        method: "coherent_multiscale_opponent_edge_retention_v1",
        evaluated: true,
        decision_supported,
        sample_stride,
        probe_radius,
        minimum_probe_count: GRAIN_DETAIL_MIN_PROBE_COUNT,
        luminance_probe_count: luminance_retention.len(),
        chroma_probe_count: chroma_retention.len(),
        luminance_decision_supported,
        chroma_decision_supported,
        luminance_median_retention,
        luminance_p10_retention,
        chroma_median_retention,
        chroma_p10_retention,
        luminance_contrast_threshold: GRAIN_DETAIL_LUMA_CONTRAST_MIN,
        chroma_contrast_threshold: GRAIN_DETAIL_CHROMA_CONTRAST_MIN,
        coherence_threshold: GRAIN_DETAIL_COHERENCE_MIN,
        median_retention_threshold: GRAIN_DETAIL_MEDIAN_RETENTION_MIN,
        p10_retention_threshold: GRAIN_DETAIL_P10_RETENTION_MIN,
        review_required,
        reason: reason.clone(),
        review_reason: review_required.then_some(reason),
    }
}

fn apply_render_noise_reduction(
    mut img: Array3<f64>,
    settings: GrainReductionSettings,
    pre_noise: &RenderGrainDiagnostics,
) -> (Array3<f64>, RenderNoiseReductionDiagnostics) {
    let settings = settings.normalized();
    if !settings.enabled {
        return (
            img,
            RenderNoiseReductionDiagnostics::disabled(
                settings,
                "disabled by the independent film-grain control",
            ),
        );
    }
    if settings.strength <= f64::EPSILON {
        return (
            img,
            RenderNoiseReductionDiagnostics::disabled(
                settings,
                "enabled with zero strength, so no grain reduction was applied",
            ),
        );
    }
    let (height, width, channels) = img.dim();
    if channels < 3 {
        return (
            img,
            RenderNoiseReductionDiagnostics::disabled(
                settings,
                "render noise reduction requires at least three channels",
            ),
        );
    }
    let luminance_radius = settings.effective_radius();
    let chroma_radius = ((RENDER_NOISE_REDUCTION_CHROMA_RADIUS as f64 * settings.scale).round()
        as usize)
        .clamp(1, 8);
    let required_radius = luminance_radius.max(chroma_radius);
    if height < required_radius * 2 + 1 || width < required_radius * 2 + 1 {
        return (
            img,
            RenderNoiseReductionDiagnostics::disabled(
                settings,
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

    let frame_chroma_grain = pre_noise
        .flat_chroma_residual_p95
        .max(pre_noise.chroma_residual_p95 * 0.65);
    let frame_chroma_damping = smoothstep_range(
        RENDER_NOISE_REDUCTION_FRAME_CHROMA_GRAIN_START,
        RENDER_NOISE_REDUCTION_FRAME_CHROMA_GRAIN_FULL,
        frame_chroma_grain,
    );

    let blurred_luminance = box_blur_luminance(&luminance, height, width, luminance_radius);
    let mut adjusted_luminance = vec![0.0f32; pixel_count];
    let mut chroma_strength = vec![0.0f32; pixel_count];
    let mut chroma_damping = vec![0.0f32; pixel_count];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let red = img[[y, x, 0]].clamp(0.0, 1.0);
            let green = img[[y, x, 1]].clamp(0.0, 1.0);
            let blue = img[[y, x, 2]].clamp(0.0, 1.0);
            chroma_strength[idx] = (red - green) as f32;
            chroma_damping[idx] = (blue - 0.5 * (red + green)) as f32;
        }
    }
    populate_coherent_chroma_structure(
        &chroma_strength,
        &chroma_damping,
        &mut adjusted_luminance,
        height,
        width,
        luminance_radius,
    );
    let (detail_sample_stride, luminance_detail_probes, chroma_detail_probes) =
        collect_grain_detail_probes(
            &luminance,
            &chroma_strength,
            &chroma_damping,
            height,
            width,
            luminance_radius,
        );
    dilate_structure_max(
        &adjusted_luminance,
        &mut chroma_strength,
        &mut chroma_damping,
        height,
        width,
        required_radius,
    );
    std::mem::swap(&mut adjusted_luminance, &mut chroma_damping);
    let mut applied = 0usize;
    let mut structure_excluded = 0usize;
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
            let coherent_chroma_structure = adjusted_luminance[idx] as f64;
            let structure = smoothed_luminance_structure(
                &blurred_luminance,
                height,
                width,
                y,
                x,
                luminance_radius,
            )
            .unwrap_or(detail_residual);
            let chroma_texture = detail_residual
                .min(
                    structure
                        + detail_residual * RENDER_NOISE_REDUCTION_CHROMA_TEXTURE_RESIDUAL_WEIGHT,
                )
                .max(coherent_chroma_structure);
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
            let coherent_chroma_protection = smoothstep_range(
                RENDER_NOISE_REDUCTION_TEXTURE_START,
                RENDER_NOISE_REDUCTION_TEXTURE_END,
                coherent_chroma_structure,
            );
            let structure_gate = 1.0
                - smoothstep_range(
                    RENDER_NOISE_REDUCTION_STRUCTURE_GATE_START,
                    RENDER_NOISE_REDUCTION_STRUCTURE_GATE_END,
                    structure.max(coherent_chroma_structure),
                );
            if chroma_texture_gate < 0.5 || luma_texture_gate < 0.5 {
                texture_limited += 1;
            }
            if effective_saturation_gate < 0.5 {
                saturation_limited += 1;
            }
            let complete_filter_window = y >= required_radius
                && y + required_radius < height
                && x >= required_radius
                && x + required_radius < width;
            if !complete_filter_window || structure_gate <= f64::EPSILON {
                // The negative sentinel prevents even f32 luminance/residual round-tripping from
                // changing a protected output pixel or an incompletely supported boundary pixel
                // in the channel reconstruction loop below.
                structure_excluded += 1;
                adjusted_luminance[idx] = luminance[idx];
                chroma_strength[idx] = -1.0;
                chroma_damping[idx] = 0.0;
                continue;
            }
            let chroma_texture_for_chroma = chroma_texture_gate.max(
                RENDER_NOISE_REDUCTION_NEUTRAL_TEXTURE_FLOOR
                    * effective_saturation_gate
                    * neutral_floor_luma_gate
                    * (1.0 - coherent_chroma_protection),
            );
            let chroma_gate = chroma_texture_for_chroma
                * (RENDER_NOISE_REDUCTION_SATURATION_GATE_FLOOR
                    + (1.0 - RENDER_NOISE_REDUCTION_SATURATION_GATE_FLOOR)
                        * effective_saturation_gate)
                * (0.70 + 0.30 * shadow_gate)
                * structure_gate;
            let luma_texture_for_luma =
                luma_texture_gate.max(RENDER_NOISE_REDUCTION_LUMA_TEXTURE_FLOOR * luma_noise_gate);
            let luma_gate = luma_texture_for_luma * (0.35 + 0.65 * shadow_gate) * structure_gate;
            let chroma = RENDER_NOISE_REDUCTION_CHROMA_AMOUNT * settings.strength * chroma_gate;
            let luma = RENDER_NOISE_REDUCTION_LUMA_AMOUNT * settings.strength * luma_gate;
            let damp = frame_chroma_damping
                * RENDER_NOISE_REDUCTION_CHROMA_DAMP_AMOUNT
                * settings.strength
                * chroma_texture_for_chroma
                * (0.70 + 0.30 * shadow_gate)
                * (RENDER_NOISE_REDUCTION_SATURATION_GATE_FLOOR
                    + (1.0 - RENDER_NOISE_REDUCTION_SATURATION_GATE_FLOOR)
                        * effective_saturation_gate)
                * structure_gate;
            let new_lum = lum * (1.0 - luma) + local_lum * luma;
            let luma_delta = (new_lum - lum).abs();

            if chroma > 0.02 || luma > 0.01 {
                applied += 1;
            }
            luma_delta_sum += luma_delta;
            luma_delta_max = luma_delta_max.max(luma_delta);
            adjusted_luminance[idx] = new_lum as f32;
            chroma_strength[idx] = chroma as f32;
            chroma_damping[idx] = damp as f32;
        }
    }

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
        let blurred_residual = box_blur_luminance(&residual, height, width, chroma_radius);
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let strength = chroma_strength[idx] as f64;
                if strength < 0.0 {
                    continue;
                }
                let base_residual = residual[idx] as f64;
                let smoothed_residual =
                    base_residual * (1.0 - strength) + blurred_residual[idx] as f64 * strength;
                // Frame-level grain evidence may increase suppression of the local high-frequency
                // residual, but it must never scale the locally blurred chroma base. Scaling the
                // whole opponent residual would be global desaturation rather than denoising.
                let local_chroma_base = blurred_residual[idx] as f64;
                let damped_residual = local_chroma_base
                    + (smoothed_residual - local_chroma_base)
                        * (1.0 - chroma_damping[idx] as f64).clamp(0.0, 1.0);
                let chroma_delta = (damped_residual - base_residual).abs();
                chroma_delta_sum += chroma_delta;
                chroma_delta_max = chroma_delta_max.max(chroma_delta);
                img[[y, x, channel]] =
                    (adjusted_luminance[idx] as f64 + damped_residual).clamp(0.0, 1.0);
            }
        }
    }

    let denom = pixel_count.max(1) as f64;
    let chroma_denom = (pixel_count.max(1) * 3) as f64;
    let detail_retention = evaluate_grain_detail_retention(
        &img,
        detail_sample_stride,
        luminance_radius,
        &luminance_detail_probes,
        &chroma_detail_probes,
    );
    (
        img,
        RenderNoiseReductionDiagnostics {
            enabled: true,
            requested_enabled: true,
            reason: "independently requested edge-aware film-grain reduction smoothed selected chroma residuals and mild shadow luminance noise, exactly excluded multiscale structure and incomplete boundary windows, and reduced chroma smoothing in saturated non-shadow regions".to_string(),
            requested_strength: settings.strength,
            requested_scale: settings.scale,
            radius: required_radius,
            chroma_amount: RENDER_NOISE_REDUCTION_CHROMA_AMOUNT * settings.strength,
            luma_amount: RENDER_NOISE_REDUCTION_LUMA_AMOUNT * settings.strength,
            applied_ratio: applied as f64 / denom,
            structure_gate_start: RENDER_NOISE_REDUCTION_STRUCTURE_GATE_START,
            structure_gate_end: RENDER_NOISE_REDUCTION_STRUCTURE_GATE_END,
            structure_excluded_ratio: structure_excluded as f64 / denom,
            texture_limited_ratio: texture_limited as f64 / denom,
            saturation_limited_ratio: saturation_limited as f64 / denom,
            mean_abs_chroma_delta: chroma_delta_sum / chroma_denom,
            max_abs_chroma_delta: chroma_delta_max,
            mean_abs_luma_delta: luma_delta_sum / denom,
            max_abs_luma_delta: luma_delta_max,
            detail_retention,
        },
    )
}

fn apply_local_luminance_detail(
    mut img: Array3<f64>,
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
        return (img, base_diagnostics);
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
                img[[y, x, c]] = (img[[y, x, c]].clamp(0.0, 1.0) * scale).clamp(0.0, 1.0);
            }
        }
    }

    let denom = pixel_count.max(1) as f64;
    (
        img,
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

fn apply_highlight_neutral_chroma_cleanup(mut img: Array3<f64>) -> (Array3<f64>, usize) {
    let (_, width, channels) = img.dim();
    if channels < 3 {
        return (img, 0);
    }

    let compressed_pixels = AtomicU64::new(0);
    img.axis_chunks_iter_mut(Axis(0), 512)
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

    (img, compressed_pixels.load(Ordering::Relaxed) as usize)
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

fn apply_midtone_neutral_chroma_cleanup(mut img: Array3<f64>) -> (Array3<f64>, usize) {
    let (_, width, channels) = img.dim();
    if channels < 3 {
        return (img, 0);
    }

    let compressed_pixels = AtomicU64::new(0);
    img.axis_chunks_iter_mut(Axis(0), 512)
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

    (img, compressed_pixels.load(Ordering::Relaxed) as usize)
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

fn apply_shadow_saturation_guard(mut img: Array3<f64>) -> (Array3<f64>, usize) {
    let (_, width, channels) = img.dim();
    if channels < 3 {
        return (img, 0);
    }

    let limited_pixels = AtomicU64::new(0);
    img.axis_chunks_iter_mut(Axis(0), 512)
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

    (img, limited_pixels.load(Ordering::Relaxed) as usize)
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
        let p95 = input_linear_percentiles[2];
        // Open compressed or broad-headroom shadow negatives, but keep midrange
        // shadow scans conservative to avoid amplifying grain texture.
        if p95 <= NEGATIVE_SHADOW_LOW_RANGE_MAX_P95 || p95 >= NEGATIVE_SHADOW_BROAD_HEADROOM_MIN_P95
        {
            Some(NEGATIVE_SHADOW_TARGET_LINEAR_MEDIAN)
        } else {
            Some(NEGATIVE_SHADOW_MIDRANGE_TARGET_LINEAR_MEDIAN)
        }
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

/// Pre-tone linear luminance percentiles `[p5, p50, p95]` of a scene-referred buffer —
/// the statistics that drive auto exposure, surfaced for diagnostics.
pub fn tone_input_linear_percentiles(img: &Array3<f64>) -> [f64; 3] {
    let (_, input_linear_percentiles, _) = tone_input_percentiles(img);
    input_linear_percentiles
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
    apply_tonemap_with_params_color_protection_style_and_grain_diagnostics(
        img,
        params,
        color_protection,
        render_style,
        GrainReductionSettings::default(),
    )
}

pub fn apply_tonemap_with_params_color_protection_style_and_grain_diagnostics(
    img: &Array3<f64>,
    params: &ToneCurveParams,
    color_protection: &ToneColorProtection,
    render_style: RenderStyle,
    grain_reduction: GrainReductionSettings,
) -> TonemapApplyResult {
    let result = Array3::<f64>::zeros(img.raw_dim());
    apply_tonemap_to_owned_buffer(
        Some(img),
        result,
        params,
        color_protection,
        render_style,
        grain_reduction,
    )
}

pub fn apply_tonemap_owned_with_params_color_protection_style_and_grain_diagnostics(
    img: Array3<f64>,
    params: &ToneCurveParams,
    color_protection: &ToneColorProtection,
    render_style: RenderStyle,
    grain_reduction: GrainReductionSettings,
) -> TonemapApplyResult {
    apply_tonemap_to_owned_buffer(
        None,
        img,
        params,
        color_protection,
        render_style,
        grain_reduction,
    )
}

fn apply_tonemap_to_owned_buffer(
    source: Option<&Array3<f64>>,
    mut result: Array3<f64>,
    params: &ToneCurveParams,
    color_protection: &ToneColorProtection,
    render_style: RenderStyle,
    grain_reduction: GrainReductionSettings,
) -> TonemapApplyResult {
    let shape = result.shape();
    let h = shape[0];
    let w = shape[1];
    let processing_config = render_style.processing_config();

    let compressed_pixels = AtomicU64::new(0);
    let gamut_chroma_scale_sum = AtomicU64::new(0);
    let gamut_chroma_scale_min = AtomicU64::new(u64::MAX);
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
                    let (r, g, b) = if let Some(img) = source {
                        (img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]])
                    } else {
                        (
                            out_chunk[[local_y, x, 0]],
                            out_chunk[[local_y, x, 1]],
                            out_chunk[[local_y, x, 2]],
                        )
                    };
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

                    let (mapped, compressed, perceptual_chroma_scale) =
                        compress_highlight_chroma(scaled, mapped_linear, params.shoulder_max);
                    if compressed {
                        compressed_pixels.fetch_add(1, Ordering::Relaxed);
                        let quantized =
                            (perceptual_chroma_scale.clamp(0.0, 1.0) * 1e9).round() as u64;
                        gamut_chroma_scale_sum.fetch_add(quantized, Ordering::Relaxed);
                        gamut_chroma_scale_min.fetch_min(quantized, Ordering::Relaxed);
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
        apply_local_luminance_detail(result)
    } else {
        (result, LocalLuminanceDetailDiagnostics::disabled())
    };
    let (result, adaptive_vibrance, preferred_skin_rendering) =
        if processing_config.adaptive_vibrance && processing_config.preferred_skin_rendering {
            apply_adaptive_vibrance_with_preferred_skin(result, color_protection)
        } else if processing_config.adaptive_vibrance {
            let (result, adaptive_vibrance) = apply_adaptive_vibrance(result, color_protection);
            (
                result,
                adaptive_vibrance,
                PreferredSkinRenderingDiagnostics::empty(
                    false,
                    format!("disabled by {} render intent", render_style.as_str()),
                ),
            )
        } else {
            (
                result,
                AdaptiveVibranceDiagnostics::disabled(format!(
                    "disabled by {} render intent",
                    render_style.as_str()
                )),
                PreferredSkinRenderingDiagnostics::empty(
                    false,
                    format!("disabled by {} render intent", render_style.as_str()),
                ),
            )
        };
    let (result, post_highlight_neutral_compressed_pixels) =
        if color_protection.highlight_neutral_chroma_enabled {
            apply_highlight_neutral_chroma_cleanup(result)
        } else {
            (result, 0)
        };
    let (result, shadow_saturation_guard_pixels) = if color_protection.shadow_chroma_enabled {
        apply_shadow_saturation_guard(result)
    } else {
        (result, 0)
    };
    let (result, midtone_neutral_compressed_pixels) =
        if color_protection.midtone_neutral_chroma_enabled {
            apply_midtone_neutral_chroma_cleanup(result)
        } else {
            (result, 0)
        };
    let noise_reduction_pre_grain = render_grain_diagnostics(&result);
    let (result, noise_reduction) =
        apply_render_noise_reduction(result, grain_reduction, &noise_reduction_pre_grain);
    let noise_reduction_post_grain = if noise_reduction.enabled {
        render_grain_diagnostics(&result)
    } else {
        noise_reduction_pre_grain.clone()
    };
    let flat_luma_before = noise_reduction_pre_grain.flat_luma_residual_p95;
    let flat_chroma_before = noise_reduction_pre_grain.flat_chroma_residual_p95;
    let noise_reduction_flat_luma_p95_reduction_ratio = if flat_luma_before > 1e-12 {
        (flat_luma_before - noise_reduction_post_grain.flat_luma_residual_p95) / flat_luma_before
    } else {
        0.0
    };
    let noise_reduction_flat_chroma_p95_reduction_ratio = if flat_chroma_before > 1e-12 {
        (flat_chroma_before - noise_reduction_post_grain.flat_chroma_residual_p95)
            / flat_chroma_before
    } else {
        0.0
    };
    let midtone_neutral_chroma_compressed_ratio =
        midtone_neutral_compressed_pixels as f64 / total_pixels;
    let perceptual_gamut_mapped_pixels = compressed_pixels.load(Ordering::Relaxed);
    let perceptual_gamut_mean_chroma_scale = if perceptual_gamut_mapped_pixels == 0 {
        1.0
    } else {
        gamut_chroma_scale_sum.load(Ordering::Relaxed) as f64
            / perceptual_gamut_mapped_pixels as f64
            / 1e9
    };
    let perceptual_gamut_min_chroma_scale = if perceptual_gamut_mapped_pixels == 0 {
        1.0
    } else {
        gamut_chroma_scale_min.load(Ordering::Relaxed) as f64 / 1e9
    };
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
            perceptual_gamut_mapping_space: "CIELAB_D50_constant_lightness_and_hue",
            perceptual_gamut_mapped_ratio: perceptual_gamut_mapped_pixels as f64 / total_pixels,
            perceptual_gamut_mean_chroma_scale,
            perceptual_gamut_min_chroma_scale,
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
            adaptive_vibrance_skin_memory_protection: adaptive_vibrance.skin_memory_protection,
            adaptive_vibrance_preferred_memory_color_guard: adaptive_vibrance
                .preferred_memory_color_guard,
            preferred_skin_rendering,
            noise_reduction_enabled: noise_reduction.enabled,
            noise_reduction_requested_enabled: noise_reduction.requested_enabled,
            noise_reduction_reason: noise_reduction.reason,
            noise_reduction_requested_strength: noise_reduction.requested_strength,
            noise_reduction_requested_scale: noise_reduction.requested_scale,
            noise_reduction_radius: noise_reduction.radius,
            noise_reduction_chroma_amount: noise_reduction.chroma_amount,
            noise_reduction_luma_amount: noise_reduction.luma_amount,
            noise_reduction_applied_ratio: noise_reduction.applied_ratio,
            noise_reduction_structure_gate_start: noise_reduction.structure_gate_start,
            noise_reduction_structure_gate_end: noise_reduction.structure_gate_end,
            noise_reduction_structure_excluded_ratio: noise_reduction.structure_excluded_ratio,
            noise_reduction_texture_limited_ratio: noise_reduction.texture_limited_ratio,
            noise_reduction_saturation_limited_ratio: noise_reduction.saturation_limited_ratio,
            noise_reduction_mean_abs_chroma_delta: noise_reduction.mean_abs_chroma_delta,
            noise_reduction_max_abs_chroma_delta: noise_reduction.max_abs_chroma_delta,
            noise_reduction_mean_abs_luma_delta: noise_reduction.mean_abs_luma_delta,
            noise_reduction_max_abs_luma_delta: noise_reduction.max_abs_luma_delta,
            noise_reduction_pre_grain,
            noise_reduction_post_grain,
            noise_reduction_flat_luma_p95_reduction_ratio,
            noise_reduction_flat_chroma_p95_reduction_ratio,
            noise_reduction_detail_retention: noise_reduction.detail_retention,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prophoto_lab(rgb: [f64; 3]) -> [f64; 3] {
        let xyz =
            crate::colorspace::prophoto_to_xyz_d50_matrix() * Vector3::new(rgb[0], rgb[1], rgb[2]);
        crate::colorspace::xyz_d50_to_lab([xyz[0], xyz[1], xyz[2]])
    }

    fn lab_prophoto(lab: [f64; 3]) -> [f64; 3] {
        let xyz = Vector3::from(crate::colorspace::lab_to_xyz_d50(lab));
        let rgb = crate::colorspace::xyz_d50_to_prophoto_matrix() * xyz;
        [rgb[0], rgb[1], rgb[2]]
    }

    fn lch_prophoto(lightness: f64, chroma: f64, hue_degrees: f64) -> [f64; 3] {
        let hue = hue_degrees.to_radians();
        lab_prophoto([lightness, chroma * hue.cos(), chroma * hue.sin()])
    }

    #[test]
    fn perceptual_gamut_mapping_preserves_cielab_lightness_and_hue() {
        let source = [1.18, -0.08, 0.64];
        let source_lab = prophoto_lab(source);
        let (mapped, changed, chroma_scale) = compress_highlight_chroma(source, 0.5, 0.99);
        let mapped_lab = prophoto_lab(mapped);

        assert!(changed);
        assert!((0.0..1.0).contains(&chroma_scale));
        assert!(mapped
            .iter()
            .all(|value| *value >= -1e-8 && *value <= 0.99 + 1e-8));
        assert!((mapped_lab[0] - source_lab[0]).abs() < 1e-6);
        let source_hue = source_lab[2].atan2(source_lab[1]);
        let mapped_hue = mapped_lab[2].atan2(mapped_lab[1]);
        assert!((mapped_hue - source_hue).abs() < 1e-6);
    }

    #[test]
    fn perceptual_gamut_mapping_is_identity_inside_render_gamut() {
        let source = [0.18, 0.42, 0.73];
        let (mapped, changed, chroma_scale) = compress_highlight_chroma(source, 0.4, 0.99);
        assert_eq!(mapped, source);
        assert!(!changed);
        assert_eq!(chroma_scale, 1.0);
    }

    #[test]
    fn skin_memory_protection_uses_measured_core_and_feathered_support() {
        let matrix = crate::colorspace::prophoto_to_xyz_d50_matrix();
        let core = skin_memory_protection_weight(lch_prophoto(55.0, 20.0, 50.0), &matrix);
        let feathered_hue = skin_memory_protection_weight(lch_prophoto(55.0, 20.0, 20.0), &matrix);
        let neutral = skin_memory_protection_weight(lch_prophoto(55.0, 2.0, 50.0), &matrix);
        let cool_hue = skin_memory_protection_weight(lch_prophoto(55.0, 20.0, 220.0), &matrix);
        let outside_lightness =
            skin_memory_protection_weight(lch_prophoto(95.0, 20.0, 50.0), &matrix);

        assert!((core - 1.0).abs() < 1e-10, "core weight={core}");
        assert!(
            feathered_hue > 0.0 && feathered_hue < 1.0,
            "feathered hue weight={feathered_hue}"
        );
        assert_eq!(neutral, 0.0);
        assert_eq!(cool_hue, 0.0);
        assert_eq!(outside_lightness, 0.0);
    }

    #[test]
    fn preferred_memory_color_ellipse_uses_published_core_and_feathered_support() {
        let model = ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_MODELS[0];
        let center = model.preferred_center_lab();
        let theta = model.ellipse_rotation_degrees.to_radians();
        let along_major = |radius: f64| {
            [
                center[0],
                center[1] + theta.cos() * model.semi_major_axis_ab * radius,
                center[2] + theta.sin() * model.semi_major_axis_ab * radius,
            ]
        };

        let center_radius = preferred_memory_color_normalized_radius(center, model);
        let feathered_radius = preferred_memory_color_normalized_radius(along_major(1.5), model);
        let outside_radius = preferred_memory_color_normalized_radius(along_major(2.1), model);
        assert!(center_radius < 1e-12);
        assert!((feathered_radius - 1.5).abs() < 1e-10);
        assert!((outside_radius - 2.1).abs() < 1e-10);
        assert_eq!(preferred_memory_color_support_weight(center_radius), 1.0);
        assert!((0.0..1.0).contains(&preferred_memory_color_support_weight(feathered_radius)));
        assert_eq!(preferred_memory_color_support_weight(outside_radius), 0.0);
    }

    fn preferred_skin_lab_at_minor_radius(radius: f64) -> [f64; 3] {
        let model = PREFERRED_SKIN_RENDERING_MODEL;
        let center = model.preferred_center_lab();
        let theta = model.ellipse_rotation_degrees.to_radians();
        [
            center[0],
            center[1] - theta.sin() * model.semi_minor_axis_ab() * radius,
            center[2] + theta.cos() * model.semi_minor_axis_ab() * radius,
        ]
    }

    #[test]
    fn preferred_skin_rendering_preserves_core_and_reduces_only_high_chroma_radius_excess() {
        let model = PREFERRED_SKIN_RENDERING_MODEL;
        let xyz_to_prophoto = crate::colorspace::xyz_d50_to_prophoto_matrix();
        let center = model.preferred_center_lab();
        let center_rgb = lab_prophoto(center);
        let center_decision =
            preferred_skin_rendering_decision(center_rgb, center, &xyz_to_prophoto);
        assert!(center_decision.matched);
        assert!(!center_decision.outside_preferred_core);
        assert!(!center_decision.adjusted);
        assert_eq!(center_decision.output_rgb, center_rgb);

        let low_chroma_outlier_lab = preferred_skin_lab_at_minor_radius(1.5);
        let low_chroma_outlier_rgb = lab_prophoto(low_chroma_outlier_lab);
        let low_chroma_decision = preferred_skin_rendering_decision(
            low_chroma_outlier_rgb,
            low_chroma_outlier_lab,
            &xyz_to_prophoto,
        );
        assert!(low_chroma_decision.matched);
        assert!(low_chroma_decision.outside_preferred_core);
        assert!(!low_chroma_decision.adjusted);
        assert_eq!(low_chroma_decision.output_rgb, low_chroma_outlier_rgb);

        let source_lab = preferred_skin_lab_at_minor_radius(-1.5);
        let source_rgb = lab_prophoto(source_lab);
        assert!(unit_rgb_in_gamut(source_rgb));
        assert!(
            skin_memory_protection_weight_from_lab(source_lab)
                >= PREFERRED_SKIN_RENDERING_MIN_SUPPORT_WEIGHT
        );
        let decision = preferred_skin_rendering_decision(source_rgb, source_lab, &xyz_to_prophoto);
        let output_lab = prophoto_lab(decision.output_rgb);
        let source_radius = preferred_memory_color_normalized_radius(source_lab, model);
        let output_radius = preferred_memory_color_normalized_radius(output_lab, model);

        assert!(decision.matched);
        assert!(decision.outside_preferred_core);
        assert!(decision.adjusted);
        assert!(decision.delta_e_ab > 0.0);
        assert!(decision.delta_e_ab <= PREFERRED_SKIN_RENDERING_MAX_DELTA_E_AB + 1e-8);
        assert!(output_radius < source_radius);
        assert!(output_radius > PREFERRED_SKIN_RENDERING_CORE_RADIUS);
        assert!((output_lab[0] - source_lab[0]).abs() < 1e-6);
        assert!(output_lab[1].hypot(output_lab[2]) < source_lab[1].hypot(source_lab[2]));
        assert!(decision.chroma_delta < 0.0);
        assert!(unit_rgb_in_gamut(decision.output_rgb));

        let cool_lab = [55.0, -20.0, -10.0];
        let cool_rgb = lab_prophoto(cool_lab);
        let cool_decision = preferred_skin_rendering_decision(cool_rgb, cool_lab, &xyz_to_prophoto);
        assert!(!cool_decision.matched);
        assert!(!cool_decision.adjusted);
        assert_eq!(cool_decision.output_rgb, cool_rgb);
    }

    #[test]
    fn preferred_skin_rendering_preserves_variation_and_requires_trusted_color() {
        let model = PREFERRED_SKIN_RENDERING_MODEL;
        let inner_lab = preferred_skin_lab_at_minor_radius(-1.3);
        let outer_lab = preferred_skin_lab_at_minor_radius(-1.8);
        let image = Array3::from_shape_fn((9, 18, 3), |(_, x, channel)| {
            if x < 9 {
                lab_prophoto(inner_lab)[channel]
            } else {
                lab_prophoto(outer_lab)[channel]
            }
        });
        let (rendered, diagnostics) =
            apply_preferred_skin_rendering(image.clone(), &ToneColorProtection::default());
        let rendered_inner = prophoto_lab([
            rendered[[4, 4, 0]],
            rendered[[4, 4, 1]],
            rendered[[4, 4, 2]],
        ]);
        let rendered_outer = prophoto_lab([
            rendered[[4, 13, 0]],
            rendered[[4, 13, 1]],
            rendered[[4, 13, 2]],
        ]);
        let rendered_inner_radius = preferred_memory_color_normalized_radius(rendered_inner, model);
        let rendered_outer_radius = preferred_memory_color_normalized_radius(rendered_outer, model);

        assert!(diagnostics.enabled);
        assert!(diagnostics.adjusted_pixel_ratio > 0.99);
        assert!(diagnostics.mean_chroma_delta < 0.0);
        assert!(rendered_inner_radius > PREFERRED_SKIN_RENDERING_CORE_RADIUS);
        assert!(rendered_outer_radius > rendered_inner_radius);
        assert!((rendered_inner[0] - inner_lab[0]).abs() < 1e-6);
        assert!((rendered_outer[0] - outer_lab[0]).abs() < 1e-6);

        let untrusted = ToneColorProtection {
            policy: ToneColorProtectionPolicy::DisabledColorCandidateReview,
            highlight_neutral_chroma_enabled: false,
            midtone_neutral_chroma_enabled: false,
            shadow_chroma_enabled: false,
            reason: "synthetic untrusted color".to_string(),
        };
        let (unchanged, disabled) = apply_preferred_skin_rendering(image.clone(), &untrusted);
        assert!(!disabled.enabled);
        assert_eq!(unchanged, image);
    }

    #[test]
    fn preferred_skin_rendering_fuses_with_adaptive_vibrance_without_undoing_reduction() {
        let source_lab = preferred_skin_lab_at_minor_radius(-1.8);
        let source_rgb = lab_prophoto(source_lab);
        let image = Array3::from_shape_fn((25, 25, 3), |(_, _, channel)| source_rgb[channel]);
        let allocation = image.as_ptr() as usize;

        let (rendered, adaptive, preferred_skin) =
            apply_adaptive_vibrance_with_preferred_skin(image, &ToneColorProtection::default());
        let rendered_lab = prophoto_lab([
            rendered[[12, 12, 0]],
            rendered[[12, 12, 1]],
            rendered[[12, 12, 2]],
        ]);

        assert_eq!(rendered.as_ptr() as usize, allocation);
        assert!(adaptive.enabled);
        assert!(preferred_skin.enabled);
        assert!(preferred_skin.adjusted_pixel_ratio > 0.99);
        assert!(preferred_skin.mean_chroma_delta < 0.0);
        assert!(rendered_lab[1].hypot(rendered_lab[2]) < source_lab[1].hypot(source_lab[2]));
        assert!((rendered_lab[0] - source_lab[0]).abs() < 1e-6);
    }

    #[test]
    fn adaptive_vibrance_parallel_chunk_reduction_is_deterministic() {
        let skin = lab_prophoto(preferred_skin_lab_at_minor_radius(-1.8));
        let cool = lab_prophoto([55.0, -20.0, -10.0]);
        let neutral = [0.24, 0.23, 0.22];
        let image =
            Array3::from_shape_fn((385, 33, 3), |(y, x, channel)| match (y / 17 + x / 5) % 3 {
                0 => skin[channel],
                1 => cool[channel],
                _ => neutral[channel],
            });

        let (first, first_adaptive, first_skin) = apply_adaptive_vibrance_with_preferred_skin(
            image.clone(),
            &ToneColorProtection::default(),
        );
        let (second, second_adaptive, second_skin) =
            apply_adaptive_vibrance_with_preferred_skin(image, &ToneColorProtection::default());

        assert_eq!(first, second);
        assert_eq!(first_adaptive.applied_ratio, second_adaptive.applied_ratio);
        assert_eq!(first_adaptive.mean_scale, second_adaptive.mean_scale);
        assert_eq!(
            first_adaptive.skin_memory_protection.protected_pixel_ratio,
            second_adaptive.skin_memory_protection.protected_pixel_ratio
        );
        assert_eq!(
            first_adaptive
                .preferred_memory_color_guard
                .mean_scale_reduction,
            second_adaptive
                .preferred_memory_color_guard
                .mean_scale_reduction
        );
        assert_eq!(
            first_skin.adjusted_pixel_ratio,
            second_skin.adjusted_pixel_ratio
        );
        assert_eq!(first_skin.mean_delta_e_ab, second_skin.mean_delta_e_ab);
        assert_eq!(first_skin.mean_chroma_delta, second_skin.mean_chroma_delta);
    }

    #[test]
    fn preferred_memory_color_guard_only_limits_paths_away_from_a_supported_center() {
        let model = ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_MODELS[0];
        let center = model.preferred_center_lab();
        let outward = [center[0], center[1] * 1.05, center[2] * 1.05];
        let outward_decision = preferred_memory_color_guard_scale(center, outward, 1.05);
        assert_eq!(outward_decision.family_index, Some(0));
        assert!((outward_decision.support_weight - 1.0).abs() < 1e-12);
        assert!((outward_decision.scale - 1.0).abs() < 1e-12);
        assert!((outward_decision.scale_reduction - 0.05).abs() < 1e-12);

        let undersaturated = [center[0], center[1] * 0.70, center[2] * 0.70];
        let toward = [center[0], center[1] * 0.75, center[2] * 0.75];
        let toward_decision = preferred_memory_color_guard_scale(undersaturated, toward, 1.05);
        assert_eq!(toward_decision.family_index, Some(0));
        assert!((toward_decision.scale - 1.05).abs() < 1e-12);
        assert_eq!(toward_decision.scale_reduction, 0.0);

        let unrelated = [55.0, 45.0, 20.0];
        let unrelated_after = [55.0, 47.0, 21.0];
        let unrelated_decision =
            preferred_memory_color_guard_scale(unrelated, unrelated_after, 1.05);
        assert_eq!(unrelated_decision.family_index, None);
        assert_eq!(unrelated_decision.scale, 1.05);
        assert_eq!(unrelated_decision.scale_reduction, 0.0);
    }

    #[test]
    fn adaptive_vibrance_preference_guard_prevents_sky_center_overshoot_without_luma_shift() {
        let model = ADAPTIVE_VIBRANCE_PREFERRED_MEMORY_MODELS[0];
        let sky = lab_prophoto(model.preferred_center_lab());
        assert!(sky.iter().all(|value| (0.0..=1.0).contains(value)));
        let image = Array3::from_shape_fn((25, 25, 3), |(_, _, c)| sky[c]);
        let (after, diagnostics) = apply_adaptive_vibrance(image, &ToneColorProtection::default());
        let after_rgb = [after[[12, 12, 0]], after[[12, 12, 1]], after[[12, 12, 2]]];
        let before_luma = linear_luminance(sky[0], sky[1], sky[2]);
        let after_luma = linear_luminance(after_rgb[0], after_rgb[1], after_rgb[2]);
        let guard = &diagnostics.preferred_memory_color_guard;

        assert!(guard.enabled);
        assert!(guard.evaluated_pixel_ratio > 0.99);
        assert!(guard.matched_pixel_ratio > 0.99);
        assert!(guard.limited_pixel_ratio > 0.99);
        assert!(guard.mean_scale_reduction > 0.0);
        assert_eq!(guard.families[0].family, "sky");
        assert!(guard.families[0].limited_pixel_ratio > 0.99);
        assert!(after_rgb
            .iter()
            .zip(sky)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-6));
        assert!((after_luma - before_luma).abs() < 2e-6);
    }

    #[test]
    fn adaptive_vibrance_limits_skin_memory_colors_without_luminance_shift() {
        let skin = lch_prophoto(55.0, 20.0, 50.0);
        let cool = lch_prophoto(55.0, 20.0, 220.0);
        assert!(skin.iter().all(|value| (0.0..=1.0).contains(value)));
        assert!(cool.iter().all(|value| (0.0..=1.0).contains(value)));

        let skin_image = Array3::from_shape_fn((25, 25, 3), |(_, _, c)| skin[c]);
        let cool_image = Array3::from_shape_fn((25, 25, 3), |(_, _, c)| cool[c]);
        let (skin_after, skin_diagnostics) =
            apply_adaptive_vibrance(skin_image, &ToneColorProtection::default());
        let (cool_after, cool_diagnostics) =
            apply_adaptive_vibrance(cool_image, &ToneColorProtection::default());

        let skin_after_rgb = [
            skin_after[[12, 12, 0]],
            skin_after[[12, 12, 1]],
            skin_after[[12, 12, 2]],
        ];
        let cool_after_rgb = [
            cool_after[[12, 12, 0]],
            cool_after[[12, 12, 1]],
            cool_after[[12, 12, 2]],
        ];
        let skin_luminance = linear_luminance(skin[0], skin[1], skin[2]);
        let cool_luminance = linear_luminance(cool[0], cool[1], cool[2]);
        let skin_scale = (skin_after_rgb[0] - skin_luminance) / (skin[0] - skin_luminance);
        let cool_scale = (cool_after_rgb[2] - cool_luminance) / (cool[2] - cool_luminance);

        assert!(skin_diagnostics.skin_memory_protection.enabled);
        assert!(
            skin_diagnostics
                .skin_memory_protection
                .protected_pixel_ratio
                > 0.99
        );
        assert!(
            skin_diagnostics
                .skin_memory_protection
                .mean_protection_weight
                > 0.99
        );
        assert_eq!(
            cool_diagnostics
                .skin_memory_protection
                .protected_pixel_ratio,
            0.0
        );
        assert!(
            skin_scale > 1.0 && skin_scale <= 1.011,
            "skin scale={skin_scale}"
        );
        assert!(cool_scale > 1.07, "cool scale={cool_scale}");
        let skin_luminance_drift =
            (linear_luminance(skin_after_rgb[0], skin_after_rgb[1], skin_after_rgb[2])
                - skin_luminance)
                .abs();
        let cool_luminance_drift =
            (linear_luminance(cool_after_rgb[0], cool_after_rgb[1], cool_after_rgb[2])
                - cool_luminance)
                .abs();
        assert!(
            skin_luminance_drift < 2e-6,
            "skin luminance drift={skin_luminance_drift}"
        );
        assert!(
            cool_luminance_drift < 2e-6,
            "cool luminance drift={cool_luminance_drift}"
        );
    }

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

    #[test]
    fn grain_detail_retention_flags_adversarial_structured_edge_erasure() {
        let size = 81usize;
        let mut luminance = vec![0.0f32; size * size];
        let mut opponent_a = vec![0.0f32; size * size];
        let mut opponent_b = vec![0.0f32; size * size];
        for y in 0..size {
            for x in 0..size {
                let rgb = if y < size / 2 {
                    if x < size / 2 {
                        [0.70, 0.20, 0.20]
                    } else {
                        [0.20, 0.4025, 0.70]
                    }
                } else {
                    let level = if x < size / 2 { 0.18 } else { 0.72 };
                    [level, level, level]
                };
                let idx = y * size + x;
                luminance[idx] = linear_luminance(rgb[0], rgb[1], rgb[2]) as f32;
                opponent_a[idx] = (rgb[0] - rgb[1]) as f32;
                opponent_b[idx] = (rgb[2] - 0.5 * (rgb[0] + rgb[1])) as f32;
            }
        }
        let (sample_stride, luminance_probes, chroma_probes) =
            collect_grain_detail_probes(&luminance, &opponent_a, &opponent_b, size, size, 2);
        assert!(luminance_probes.len() >= GRAIN_DETAIL_MIN_PROBE_COUNT);
        assert!(chroma_probes.len() >= GRAIN_DETAIL_MIN_PROBE_COUNT);

        let erased = Array3::<f64>::from_elem((size, size, 3), 0.4);
        let diagnostics = evaluate_grain_detail_retention(
            &erased,
            sample_stride,
            2,
            &luminance_probes,
            &chroma_probes,
        );

        assert!(diagnostics.decision_supported, "{diagnostics:?}");
        assert!(diagnostics.review_required, "{diagnostics:?}");
        assert_eq!(diagnostics.luminance_p10_retention, 0.0);
        assert_eq!(diagnostics.chroma_p10_retention, 0.0);
        assert!(diagnostics
            .review_reason
            .as_deref()
            .is_some_and(
                |reason| reason.contains("luminance") && reason.contains("opponent-color")
            ));
    }

    #[test]
    fn post_tone_render_passes_reuse_the_owned_working_buffer() {
        let mut image = Array3::<f64>::zeros((96, 128, 3));
        for y in 0..96 {
            for x in 0..128 {
                image[[y, x, 0]] = 0.08 + x as f64 / 512.0;
                image[[y, x, 1]] = 0.10 + y as f64 / 480.0;
                image[[y, x, 2]] = 0.07 + (x + y) as f64 / 900.0;
            }
        }
        let allocation = image.as_ptr() as usize;
        let (image, _) = apply_local_luminance_detail(image);
        assert_eq!(image.as_ptr() as usize, allocation);
        let (image, _) = apply_adaptive_vibrance(image, &ToneColorProtection::default());
        assert_eq!(image.as_ptr() as usize, allocation);
        let (image, _) = apply_highlight_neutral_chroma_cleanup(image);
        assert_eq!(image.as_ptr() as usize, allocation);
        let (image, _) = apply_shadow_saturation_guard(image);
        assert_eq!(image.as_ptr() as usize, allocation);
        let (image, _) = apply_midtone_neutral_chroma_cleanup(image);
        assert_eq!(image.as_ptr() as usize, allocation);
        let pre_grain = render_grain_diagnostics(&image);
        let (image, _) =
            apply_render_noise_reduction(image, GrainReductionSettings::default(), &pre_grain);
        assert_eq!(image.as_ptr() as usize, allocation);

        let enabled = GrainReductionSettings {
            enabled: true,
            strength: 0.5,
            scale: 1.0,
        };
        let pre_grain = render_grain_diagnostics(&image);
        let (image, _) = apply_render_noise_reduction(image, enabled, &pre_grain);
        assert_eq!(image.as_ptr() as usize, allocation);
    }

    #[test]
    fn owned_batch_tonemap_is_pixel_identical_and_reuses_the_scene_buffer() {
        let mut image = Array3::<f64>::zeros((96, 128, 3));
        for y in 0..96 {
            for x in 0..128 {
                image[[y, x, 0]] = 0.04 + x as f64 / 170.0;
                image[[y, x, 1]] = 0.03 + y as f64 / 150.0;
                image[[y, x, 2]] = 0.02 + (x + y) as f64 / 310.0;
            }
        }
        let allocation = image.as_ptr() as usize;
        let params = ToneCurveParams::default();
        let protection = ToneColorProtection::default();
        let grain = GrainReductionSettings {
            enabled: true,
            strength: 0.5,
            scale: 1.0,
        };
        let borrowed = apply_tonemap_with_params_color_protection_style_and_grain_diagnostics(
            &image,
            &params,
            &protection,
            RenderStyle::ModernClean,
            grain,
        );
        let owned = apply_tonemap_owned_with_params_color_protection_style_and_grain_diagnostics(
            image,
            &params,
            &protection,
            RenderStyle::ModernClean,
            grain,
        );

        assert_eq!(owned.image.as_ptr() as usize, allocation);
        assert_eq!(owned.image, borrowed.image);
        assert_eq!(
            owned.diagnostics.noise_reduction_applied_ratio,
            borrowed.diagnostics.noise_reduction_applied_ratio
        );
        assert_eq!(
            owned
                .diagnostics
                .noise_reduction_detail_retention
                .luminance_p10_retention,
            borrowed
                .diagnostics
                .noise_reduction_detail_retention
                .luminance_p10_retention
        );
    }
}

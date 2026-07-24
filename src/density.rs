use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use ndarray::parallel::prelude::*;
use ndarray::{Array3, Axis};
use serde::{Deserialize, Serialize};

use crate::constants::{EPSILON_T, MAX_14BIT, MAX_16BIT};
use crate::streaming;

const HISTOGRAM_BINS: usize = 4096;
const DENSITY_DMAX_PERCENTILE: f64 = 0.995;
const LINEAR_DIVISION_PERCENTILE: f64 = 0.995;
const RESPONSE_ESTIMATION_MAX_SAMPLES: usize = 120_000;
const RESPONSE_ESTIMATION_TRIM_LOW: f64 = 0.01;
const RESPONSE_ESTIMATION_TRIM_HIGH: f64 = 0.99;
const RESPONSE_MIN_SAMPLES: usize = 1_024;
const RESPONSE_MIN_EXPLAINED_VARIANCE: f64 = 0.72;
const RESPONSE_MIN_PAIR_CORRELATION: f64 = 0.45;
const RESPONSE_MAX_RAW_SLOPE_CONDITION: f64 = 4.0;
const RESPONSE_MAX_FRAME_SHRINKAGE: f64 = 0.55;
const RESPONSE_MIN_SLOPE: f64 = 0.67;
const RESPONSE_MAX_SLOPE: f64 = 1.50;
const MEASURED_RESPONSE_MIN_CONFIDENCE: f64 = 0.75;
const MEASURED_RESPONSE_MIN_HELD_OUT_PATCHES: usize = 12;
const MEASURED_RESPONSE_MAX_DELTA_E00_RMS: f64 = 6.0;
const MEASURED_RESPONSE_MAX_DELTA_E00_MAX: f64 = 15.0;
const MEASURED_RESPONSE_MIN_BASELINE_IMPROVEMENT: f64 = 0.25;
const MEASURED_RESPONSE_MAX_MATRIX_CONDITION: f64 = 20.0;
const MEASURED_RESPONSE_MIN_CURVE_POINTS: usize = 4;
const MEASURED_RESPONSE_MAX_NOISE_GAIN: f64 = 4.0;
const MEASURED_RESPONSE_MAX_TRUSTED_EXTRAPOLATION_RATIO: f64 = 0.01;
/// Until a measured film response curve is available, direct-density reconstruction uses a
/// unit density slope. Keeping this explicit makes the physical assumption visible and avoids
/// normalizing every frame to an arbitrary per-image contrast range.
pub const DEFAULT_NEGATIVE_DENSITY_RESPONSE_SCALE: f64 = 1.0;

fn default_response_anchor_percentile() -> f64 {
    DENSITY_DMAX_PERCENTILE
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NegativeResponseValidation {
    pub held_out_patch_count: usize,
    pub delta_e00_rms: f64,
    pub delta_e00_max: f64,
    pub unit_slope_delta_e00_rms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worst_hue_family: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NegativeCharacteristicCurves {
    /// Each point is `[separated_layer_density, scene_log_exposure]`.
    pub red: Vec<[f64; 2]>,
    pub green: Vec<[f64; 2]>,
    pub blue: Vec<[f64; 2]>,
}

impl NegativeCharacteristicCurves {
    fn channels(&self) -> [&[[f64; 2]]; 3] {
        [&self.red, &self.green, &self.blue]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MeasuredNegativeResponseCalibration {
    pub model_id: String,
    /// Maps base-subtracted scanner optical density into separated film-layer density.
    pub scanner_density_to_layer_density: [[f64; 3]; 3],
    /// Monotone inverse characteristic curves from separated layer density to scene log exposure.
    pub characteristic_curves: NegativeCharacteristicCurves,
    #[serde(default = "default_response_anchor_percentile")]
    pub white_anchor_percentile: f64,
    pub confidence: f64,
    pub validation: NegativeResponseValidation,
}

#[derive(Debug, Clone, Serialize)]
pub struct NegativeResponseModelDiagnostics {
    pub model: &'static str,
    pub source: &'static str,
    pub accepted: bool,
    pub review_required: bool,
    pub reason: String,
    pub sampled_pixels: usize,
    pub retained_pixels: usize,
    pub explained_variance_ratio: Option<f64>,
    pub minimum_pair_correlation: Option<f64>,
    pub raw_density_slopes: [f64; 3],
    pub applied_density_slopes: [f64; 3],
    pub shrinkage_toward_frame_estimate: f64,
    pub raw_slope_condition_number: f64,
    pub applied_slope_condition_number: f64,
    pub maximum_density_noise_gain: f64,
    pub crosstalk_model: &'static str,
    pub characteristic_curve_model: &'static str,
    pub measured_model_id: Option<String>,
    pub measured_confidence: Option<f64>,
    pub held_out_delta_e00_rms: Option<f64>,
    pub held_out_delta_e00_max: Option<f64>,
    pub unit_slope_delta_e00_rms: Option<f64>,
    pub scanner_density_to_layer_density: Option<[[f64; 3]; 3]>,
}

#[derive(Debug, Clone)]
pub struct NegativeResponseModel {
    pub diagnostics: NegativeResponseModelDiagnostics,
    measured: Option<MeasuredNegativeResponseCalibration>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NegativeResponseReconstructionDiagnostics {
    pub model: NegativeResponseModelDiagnostics,
    pub scene_white_log_exposure_anchor: f64,
    pub channel_white_log_exposure_anchors: [f64; 3],
    pub highlight_headroom_samples: [u64; 3],
    pub scene_linear_high_percentile: [f64; 3],
    pub maximum_scene_linear_value: [f64; 3],
    pub signed_headroom_preserved: bool,
    pub curve_extrapolated_low_samples: [u64; 3],
    pub curve_extrapolated_high_samples: [u64; 3],
    pub curve_extrapolated_any_samples: u64,
    pub curve_extrapolated_any_ratio: f64,
    pub curve_interpolation: &'static str,
    pub review_required: bool,
    pub review_reason: String,
}

pub struct NegativeResponseReconstructionResult {
    pub scene_linear: Array3<f64>,
    pub diagnostics: NegativeResponseReconstructionDiagnostics,
}

#[derive(Debug, Clone)]
pub struct ChannelPercentileSummary {
    pub min_value: f64,
    pub max_value: f64,
    pub high_percentile: f64,
}

#[derive(Debug, Clone)]
pub struct LinearDivisionDiagnostics {
    pub exact_max: f64,
    pub robust_high_percentile: f64,
    pub percentile: f64,
}

#[derive(Debug, Clone)]
pub struct DensityDiagnostics {
    pub base_transmittance: [f64; 3],
    pub base_density: [f64; 3],
    pub robust_d_max: [f64; 3],
    pub shared_robust_d_max: f64,
    pub exact_d_max: [f64; 3],
    pub d_max_percentile: f64,
    pub histogram_bins: usize,
    /// Samples denser than the shared robust white anchor. These are retained as values above
    /// scene-linear 1.0 instead of being clipped to the anchor.
    pub highlight_headroom_samples: [u64; 3],
    pub epsilon_t: f64,
    pub density_stats: [ChannelPercentileSummary; 3],
    pub linear_division: [LinearDivisionDiagnostics; 3],
}

pub struct Phase3Result {
    pub positive_density: Array3<f64>,
    pub diagnostics: DensityDiagnostics,
}

/// Convert a u16 image to f64 in [EPSILON_T, 1.0], clamping zeros to EPSILON_T.
pub fn normalize_to_float(img: &Array3<u16>, bit_depth: u8) -> Array3<f64> {
    let max_val = match bit_depth {
        14 => MAX_14BIT,
        _ => MAX_16BIT,
    };
    img.mapv(|v| normalize_sample(v, max_val))
}

/// Convert an already-positive RGB scan to linear [0, 1] without density inversion.
pub fn normalize_positive_scan_to_linear_rgb(img: &Array3<u16>, bit_depth: u8) -> Array3<f64> {
    let (height, width, channels) = img.dim();
    assert_eq!(channels, 3, "Expected 3-channel image");

    let max_val = bit_depth_max(bit_depth);
    let mut out = Array3::<f64>::zeros((height, width, channels));
    out.axis_chunks_iter_mut(Axis(0), streaming::DEFAULT_TILE_ROWS)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut out_chunk)| {
            let row_start = chunk_idx * streaming::DEFAULT_TILE_ROWS;
            let chunk_h = out_chunk.dim().0;
            for local_y in 0..chunk_h {
                let y = row_start + local_y;
                for x in 0..width {
                    for c in 0..3 {
                        out_chunk[[local_y, x, c]] =
                            (img[[y, x, c]] as f64 / max_val).clamp(0.0, 1.0);
                    }
                }
            }
        });
    out
}

/// Convert transmittance to optical density: D = -log10(T).
pub fn transmittance_to_density(t: f64) -> f64 {
    -normalize_transmittance(t).log10()
}

/// Convert optical density back to transmittance: T = 10^(-D).
pub fn density_to_transmittance(d: f64) -> f64 {
    if !d.is_finite() {
        return EPSILON_T;
    }
    normalize_transmittance(10.0f64.powf(-d.max(0.0)))
}

/// Convert signed positive-image density to scene-linear light.
///
/// Unlike physical transmittance, a reconstructed scene may legitimately exceed 1.0 above its
/// robust white anchor. Negative density therefore remains negative and becomes highlight
/// headroom instead of being clamped away.
fn signed_density_to_scene_linear(d: f64) -> f64 {
    if !d.is_finite() {
        return EPSILON_T;
    }
    let value = 10.0f64.powf(-d);
    if value.is_finite() {
        value.max(EPSILON_T)
    } else {
        EPSILON_T
    }
}

/// Convert a density-domain image to linear transmittance.
pub fn density_image_to_transmittance(img: &Array3<f64>) -> Array3<f64> {
    let (h, w, c) = img.dim();
    let mut out = Array3::<f64>::zeros((h, w, c));

    out.axis_chunks_iter_mut(Axis(0), streaming::DEFAULT_TILE_ROWS)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut out_chunk)| {
            let row_start = chunk_idx * streaming::DEFAULT_TILE_ROWS;
            let chunk_h = out_chunk.dim().0;
            for local_y in 0..chunk_h {
                let y = row_start + local_y;
                for x in 0..w {
                    for ch in 0..c {
                        out_chunk[[local_y, x, ch]] = density_to_transmittance(img[[y, x, ch]]);
                    }
                }
            }
        });

    out
}

/// Convert a signed positive-density image to scene-linear light.
///
/// `density_response_scale` is the slope of recorded film density versus scene log exposure.
/// A measured film/roll response should supply it; the current default is the explicit unit
/// slope in [`DEFAULT_NEGATIVE_DENSITY_RESPONSE_SCALE`]. The conversion intentionally does not
/// normalize by the frame's own D-max and does not clamp values above 1.0.
pub fn density_image_to_normalized_transmittance(
    img: &Array3<f64>,
    density_response_scale: f64,
) -> Array3<f64> {
    let (h, w, c) = img.dim();
    let mut out = Array3::<f64>::zeros((h, w, c));
    let density_response_scale = density_response_scale.max(1e-6);

    out.axis_chunks_iter_mut(Axis(0), streaming::DEFAULT_TILE_ROWS)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut out_chunk)| {
            let row_start = chunk_idx * streaming::DEFAULT_TILE_ROWS;
            let chunk_h = out_chunk.dim().0;
            for local_y in 0..chunk_h {
                let y = row_start + local_y;
                for x in 0..w {
                    for ch in 0..c {
                        let scene_density = img[[y, x, ch]] / density_response_scale;
                        out_chunk[[local_y, x, ch]] = signed_density_to_scene_linear(scene_density);
                    }
                }
            }
        });

    out
}

fn slope_condition_number(slopes: &[f64; 3]) -> f64 {
    let minimum = slopes.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = slopes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if minimum.is_finite() && maximum.is_finite() && minimum > 1e-12 {
        maximum / minimum
    } else {
        f64::INFINITY
    }
}

fn sorted_percentile(values: &mut [f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let index = ((values.len() - 1) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    values[index]
}

pub fn unit_negative_response_model() -> NegativeResponseModel {
    NegativeResponseModel {
        diagnostics: NegativeResponseModelDiagnostics {
            model: "shared_unit_density_slope",
            source: "explicit_uncalibrated_fallback",
            accepted: true,
            review_required: true,
            reason: "no measured film characteristic curve or dye-crosstalk model is available; preserving recorded density relationships with an explicit unit-slope fallback"
                .to_string(),
            sampled_pixels: 0,
            retained_pixels: 0,
            explained_variance_ratio: None,
            minimum_pair_correlation: None,
            raw_density_slopes: [1.0; 3],
            applied_density_slopes: [1.0; 3],
            shrinkage_toward_frame_estimate: 0.0,
            raw_slope_condition_number: 1.0,
            applied_slope_condition_number: 1.0,
            maximum_density_noise_gain: 1.0,
            crosstalk_model: "identity_unmeasured",
            characteristic_curve_model: "linear_unit_slope_unmeasured",
            measured_model_id: None,
            measured_confidence: None,
            held_out_delta_e00_rms: None,
            held_out_delta_e00_max: None,
            unit_slope_delta_e00_rms: None,
            scanner_density_to_layer_density: None,
        },
        measured: None,
    }
}

fn rejected_frame_response_model(
    sampled_pixels: usize,
    retained_pixels: usize,
    explained_variance_ratio: Option<f64>,
    minimum_pair_correlation: Option<f64>,
    raw_density_slopes: [f64; 3],
    reason: String,
) -> NegativeResponseModel {
    NegativeResponseModel {
        diagnostics: NegativeResponseModelDiagnostics {
            model: "regularized_frame_luminance_axis",
            source: "frame_derived_candidate",
            accepted: false,
            review_required: true,
            reason,
            sampled_pixels,
            retained_pixels,
            explained_variance_ratio,
            minimum_pair_correlation,
            raw_density_slopes,
            applied_density_slopes: [1.0; 3],
            shrinkage_toward_frame_estimate: 0.0,
            raw_slope_condition_number: slope_condition_number(&raw_density_slopes),
            applied_slope_condition_number: 1.0,
            maximum_density_noise_gain: 1.0,
            crosstalk_model: "identity_unmeasured",
            characteristic_curve_model: "linear_regularized_frame_estimate",
            measured_model_id: None,
            measured_confidence: None,
            held_out_delta_e00_rms: None,
            held_out_delta_e00_max: None,
            unit_slope_delta_e00_rms: None,
            scanner_density_to_layer_density: None,
        },
        measured: None,
    }
}

/// Estimate a conservative per-layer density response from the dominant correlated variation in
/// a frame. The estimate is deliberately shrunk toward unit slope because scene colors are not a
/// calibration target and cannot identify dye crosstalk or nonlinear characteristic curves.
pub fn estimate_regularized_frame_response(
    positive_density: &Array3<f64>,
    shared_robust_d_max: f64,
) -> NegativeResponseModel {
    let (height, width, channels) = positive_density.dim();
    assert_eq!(channels, 3, "Expected 3-channel density image");
    let total_pixels = height.saturating_mul(width);
    let stride = total_pixels
        .div_ceil(RESPONSE_ESTIMATION_MAX_SAMPLES)
        .max(1);
    let mut samples = Vec::<[f64; 3]>::with_capacity(
        total_pixels
            .div_ceil(stride)
            .min(RESPONSE_ESTIMATION_MAX_SAMPLES),
    );
    for flat in (0..total_pixels).step_by(stride) {
        let y = flat / width;
        let x = flat % width;
        let density =
            std::array::from_fn(|channel| shared_robust_d_max - positive_density[[y, x, channel]]);
        if density.iter().all(|value| value.is_finite()) {
            samples.push(density);
        }
    }
    let sampled_pixels = samples.len();
    if sampled_pixels < RESPONSE_MIN_SAMPLES {
        return rejected_frame_response_model(
            sampled_pixels,
            0,
            None,
            None,
            [1.0; 3],
            format!(
                "only {sampled_pixels} finite samples were available; at least {RESPONSE_MIN_SAMPLES} are required"
            ),
        );
    }

    let mut channel_values: [Vec<f64>; 3] =
        std::array::from_fn(|channel| samples.iter().map(|sample| sample[channel]).collect());
    let low: [f64; 3] = std::array::from_fn(|channel| {
        sorted_percentile(&mut channel_values[channel], RESPONSE_ESTIMATION_TRIM_LOW)
    });
    let high: [f64; 3] = std::array::from_fn(|channel| {
        sorted_percentile(&mut channel_values[channel], RESPONSE_ESTIMATION_TRIM_HIGH)
    });
    let retained = samples
        .iter()
        .copied()
        .filter(|sample| {
            (0..3)
                .all(|channel| sample[channel] >= low[channel] && sample[channel] <= high[channel])
        })
        .collect::<Vec<_>>();
    let retained_pixels = retained.len();
    if retained_pixels < RESPONSE_MIN_SAMPLES {
        return rejected_frame_response_model(
            sampled_pixels,
            retained_pixels,
            None,
            None,
            [1.0; 3],
            "robust 1%-99% trimming left too few samples for a stable density response estimate"
                .to_string(),
        );
    }

    let means: [f64; 3] = std::array::from_fn(|channel| {
        retained.iter().map(|sample| sample[channel]).sum::<f64>() / retained_pixels as f64
    });
    let mut covariance = Matrix3::<f64>::zeros();
    for sample in &retained {
        let centered = Vector3::new(
            sample[0] - means[0],
            sample[1] - means[1],
            sample[2] - means[2],
        );
        covariance += centered * centered.transpose();
    }
    covariance /= (retained_pixels - 1) as f64;
    let total_variance = covariance.trace();
    if !total_variance.is_finite() || total_variance <= 1e-10 {
        return rejected_frame_response_model(
            sampled_pixels,
            retained_pixels,
            None,
            None,
            [1.0; 3],
            "trimmed density samples contain insufficient variation to estimate film-layer response"
                .to_string(),
        );
    }

    let correlations = [
        covariance[(0, 1)] / (covariance[(0, 0)] * covariance[(1, 1)]).sqrt().max(1e-12),
        covariance[(0, 2)] / (covariance[(0, 0)] * covariance[(2, 2)]).sqrt().max(1e-12),
        covariance[(1, 2)] / (covariance[(1, 1)] * covariance[(2, 2)]).sqrt().max(1e-12),
    ];
    let minimum_pair_correlation = correlations.iter().copied().fold(f64::INFINITY, f64::min);

    let eigensystem = SymmetricEigen::new(covariance);
    let (principal_index, principal_value) = eigensystem
        .eigenvalues
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap_or((0, 0.0));
    let explained_variance_ratio = principal_value / total_variance;
    let mut principal = eigensystem
        .eigenvectors
        .column(principal_index)
        .into_owned();
    if principal.iter().sum::<f64>() < 0.0 {
        principal *= -1.0;
    }
    if principal
        .iter()
        .any(|value| !value.is_finite() || *value <= 1e-8)
    {
        return rejected_frame_response_model(
            sampled_pixels,
            retained_pixels,
            Some(explained_variance_ratio),
            Some(minimum_pair_correlation),
            [1.0; 3],
            "the dominant density axis did not increase in all three film layers; scene chroma is too strong for a frame-derived response estimate"
                .to_string(),
        );
    }
    let geometric_mean = (principal[0] * principal[1] * principal[2]).cbrt();
    let raw_density_slopes = std::array::from_fn(|channel| principal[channel] / geometric_mean);
    let raw_condition = slope_condition_number(&raw_density_slopes);

    let rejection = if explained_variance_ratio < RESPONSE_MIN_EXPLAINED_VARIANCE {
        Some(format!(
            "dominant density variation explained only {:.1}% of trimmed variance; {:.1}% is required",
            explained_variance_ratio * 100.0,
            RESPONSE_MIN_EXPLAINED_VARIANCE * 100.0
        ))
    } else if minimum_pair_correlation < RESPONSE_MIN_PAIR_CORRELATION {
        Some(format!(
            "minimum inter-layer density correlation was {minimum_pair_correlation:.3}; {RESPONSE_MIN_PAIR_CORRELATION:.3} is required"
        ))
    } else if raw_condition > RESPONSE_MAX_RAW_SLOPE_CONDITION {
        Some(format!(
            "unregularized density slope condition number {raw_condition:.3} exceeded the {RESPONSE_MAX_RAW_SLOPE_CONDITION:.3} safety limit"
        ))
    } else {
        None
    };
    if let Some(reason) = rejection {
        return rejected_frame_response_model(
            sampled_pixels,
            retained_pixels,
            Some(explained_variance_ratio),
            Some(minimum_pair_correlation),
            raw_density_slopes,
            reason,
        );
    }

    let variance_strength = ((explained_variance_ratio - 0.65) / 0.30).clamp(0.0, 1.0);
    let correlation_strength = ((minimum_pair_correlation - 0.35) / 0.55).clamp(0.0, 1.0);
    let shrinkage = RESPONSE_MAX_FRAME_SHRINKAGE * variance_strength.min(correlation_strength);
    let applied_density_slopes = std::array::from_fn(|channel| {
        raw_density_slopes[channel]
            .ln()
            .mul_add(shrinkage, 0.0)
            .exp()
            .clamp(RESPONSE_MIN_SLOPE, RESPONSE_MAX_SLOPE)
    });
    let applied_condition = slope_condition_number(&applied_density_slopes);
    let maximum_density_noise_gain = applied_density_slopes
        .iter()
        .map(|slope| 1.0 / slope)
        .fold(0.0, f64::max);

    NegativeResponseModel {
        diagnostics: NegativeResponseModelDiagnostics {
            model: "regularized_frame_luminance_axis",
            source: "frame_derived_candidate",
            accepted: true,
            review_required: true,
            reason: format!(
                "a robust principal density axis explained {:.1}% of trimmed variation with minimum pair correlation {:.3}; only {:.1}% of its per-layer slope estimate was applied and dye crosstalk/nonlinearity remain unmeasured",
                explained_variance_ratio * 100.0,
                minimum_pair_correlation,
                shrinkage * 100.0
            ),
            sampled_pixels,
            retained_pixels,
            explained_variance_ratio: Some(explained_variance_ratio),
            minimum_pair_correlation: Some(minimum_pair_correlation),
            raw_density_slopes,
            applied_density_slopes,
            shrinkage_toward_frame_estimate: shrinkage,
            raw_slope_condition_number: raw_condition,
            applied_slope_condition_number: applied_condition,
            maximum_density_noise_gain,
            crosstalk_model: "identity_unmeasured",
            characteristic_curve_model: "linear_regularized_frame_estimate",
            measured_model_id: None,
            measured_confidence: None,
            held_out_delta_e00_rms: None,
            held_out_delta_e00_max: None,
            unit_slope_delta_e00_rms: None,
            scanner_density_to_layer_density: None,
        },
        measured: None,
    }
}

fn rows_to_matrix3(rows: &[[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::new(
        rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0],
        rows[2][1], rows[2][2],
    )
}

fn matrix_condition_number(matrix: &Matrix3<f64>) -> f64 {
    let singular_values = matrix.svd(false, false).singular_values;
    let maximum = singular_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let minimum = singular_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if maximum.is_finite() && minimum.is_finite() && minimum > 1e-12 {
        maximum / minimum
    } else {
        f64::INFINITY
    }
}

fn measured_response_noise_gain(calibration: &MeasuredNegativeResponseCalibration) -> f64 {
    let matrix = rows_to_matrix3(&calibration.scanner_density_to_layer_density);
    calibration
        .characteristic_curves
        .channels()
        .iter()
        .enumerate()
        .map(|(channel, curve)| {
            let curve_gain = curve
                .windows(2)
                .map(|points| {
                    (points[1][1] - points[0][1]).abs()
                        / (points[1][0] - points[0][0]).abs().max(1e-12)
                })
                .fold(0.0, f64::max);
            let matrix_gain = matrix.row(channel).norm();
            curve_gain * matrix_gain
        })
        .fold(0.0, f64::max)
}

pub fn validate_measured_negative_response(
    calibration: &MeasuredNegativeResponseCalibration,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if calibration.model_id.trim().is_empty() {
        errors.push("negative_response.model_id must not be empty".to_string());
    }
    if !calibration.confidence.is_finite()
        || !(MEASURED_RESPONSE_MIN_CONFIDENCE..=1.0).contains(&calibration.confidence)
    {
        errors.push(format!(
            "negative_response.confidence must be finite and between {MEASURED_RESPONSE_MIN_CONFIDENCE:.2} and 1.0"
        ));
    }
    if !calibration.white_anchor_percentile.is_finite()
        || !(0.95..=0.9999).contains(&calibration.white_anchor_percentile)
    {
        errors.push(
            "negative_response.white_anchor_percentile must be between 0.95 and 0.9999".to_string(),
        );
    }

    let matrix = rows_to_matrix3(&calibration.scanner_density_to_layer_density);
    if matrix.iter().any(|value| !value.is_finite()) {
        errors.push(
            "negative_response.scanner_density_to_layer_density contains non-finite values"
                .to_string(),
        );
    }
    let matrix_condition = matrix_condition_number(&matrix);
    if !matrix_condition.is_finite() || matrix_condition > MEASURED_RESPONSE_MAX_MATRIX_CONDITION {
        errors.push(format!(
            "negative_response scanner-density separation matrix condition number {matrix_condition:.3} exceeds {MEASURED_RESPONSE_MAX_MATRIX_CONDITION:.3}"
        ));
    }

    for (channel, curve) in ["red", "green", "blue"]
        .into_iter()
        .zip(calibration.characteristic_curves.channels())
    {
        if curve.len() < MEASURED_RESPONSE_MIN_CURVE_POINTS {
            errors.push(format!(
                "negative_response characteristic curve `{channel}` requires at least {MEASURED_RESPONSE_MIN_CURVE_POINTS} measured points"
            ));
            continue;
        }
        for (index, point) in curve.iter().enumerate() {
            if point.iter().any(|value| !value.is_finite()) {
                errors.push(format!(
                    "negative_response characteristic curve `{channel}` point {index} contains non-finite values"
                ));
            }
        }
        for (index, points) in curve.windows(2).enumerate() {
            if points[1][0] <= points[0][0] || points[1][1] <= points[0][1] {
                errors.push(format!(
                    "negative_response characteristic curve `{channel}` must be strictly increasing in layer density and scene log exposure (points {index} and {})",
                    index + 1
                ));
            }
        }
    }

    let validation = &calibration.validation;
    if validation.held_out_patch_count < MEASURED_RESPONSE_MIN_HELD_OUT_PATCHES {
        errors.push(format!(
            "negative_response validation requires at least {MEASURED_RESPONSE_MIN_HELD_OUT_PATCHES} held-out patches"
        ));
    }
    for (label, value) in [
        ("delta_e00_rms", validation.delta_e00_rms),
        ("delta_e00_max", validation.delta_e00_max),
        (
            "unit_slope_delta_e00_rms",
            validation.unit_slope_delta_e00_rms,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            errors.push(format!(
                "negative_response validation `{label}` must be finite and non-negative"
            ));
        }
    }
    if validation.delta_e00_rms > MEASURED_RESPONSE_MAX_DELTA_E00_RMS {
        errors.push(format!(
            "negative_response held-out DeltaE00 RMS {:.3} exceeds {:.3}",
            validation.delta_e00_rms, MEASURED_RESPONSE_MAX_DELTA_E00_RMS
        ));
    }
    if validation.delta_e00_max > MEASURED_RESPONSE_MAX_DELTA_E00_MAX {
        errors.push(format!(
            "negative_response held-out maximum DeltaE00 {:.3} exceeds {:.3}",
            validation.delta_e00_max, MEASURED_RESPONSE_MAX_DELTA_E00_MAX
        ));
    }
    if validation.unit_slope_delta_e00_rms - validation.delta_e00_rms
        < MEASURED_RESPONSE_MIN_BASELINE_IMPROVEMENT
    {
        errors.push(format!(
            "negative_response must improve held-out DeltaE00 RMS over the unit-slope baseline by at least {MEASURED_RESPONSE_MIN_BASELINE_IMPROVEMENT:.2}"
        ));
    }
    let noise_gain = measured_response_noise_gain(calibration);
    if !noise_gain.is_finite() || noise_gain > MEASURED_RESPONSE_MAX_NOISE_GAIN {
        errors.push(format!(
            "negative_response combined curve/matrix density-noise gain {noise_gain:.3} exceeds {MEASURED_RESPONSE_MAX_NOISE_GAIN:.3}"
        ));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

pub fn measured_negative_response_model(
    calibration: &MeasuredNegativeResponseCalibration,
) -> Result<NegativeResponseModel, Vec<String>> {
    validate_measured_negative_response(calibration)?;
    let matrix = rows_to_matrix3(&calibration.scanner_density_to_layer_density);
    let condition = matrix_condition_number(&matrix);
    let noise_gain = measured_response_noise_gain(calibration);
    Ok(NegativeResponseModel {
        diagnostics: NegativeResponseModelDiagnostics {
            model: "measured_nonlinear_dye_separation",
            source: "measured_roll_target_with_held_out_validation",
            accepted: true,
            review_required: false,
            reason: format!(
                "measured 3x3 scanner-density/dye separation and monotone nonlinear film curves passed held-out validation (DeltaE00 RMS {:.3} versus unit-slope {:.3}, {} patches)",
                calibration.validation.delta_e00_rms,
                calibration.validation.unit_slope_delta_e00_rms,
                calibration.validation.held_out_patch_count
            ),
            sampled_pixels: 0,
            retained_pixels: 0,
            explained_variance_ratio: None,
            minimum_pair_correlation: None,
            raw_density_slopes: [1.0; 3],
            applied_density_slopes: [1.0; 3],
            shrinkage_toward_frame_estimate: 1.0,
            raw_slope_condition_number: condition,
            applied_slope_condition_number: condition,
            maximum_density_noise_gain: noise_gain,
            crosstalk_model: "measured_3x3_scanner_density_to_film_layers",
            characteristic_curve_model: "measured_monotone_pchip_density_to_scene_log_exposure",
            measured_model_id: Some(calibration.model_id.clone()),
            measured_confidence: Some(calibration.confidence),
            held_out_delta_e00_rms: Some(calibration.validation.delta_e00_rms),
            held_out_delta_e00_max: Some(calibration.validation.delta_e00_max),
            unit_slope_delta_e00_rms: Some(calibration.validation.unit_slope_delta_e00_rms),
            scanner_density_to_layer_density: Some(
                calibration.scanner_density_to_layer_density,
            ),
        },
        measured: Some(calibration.clone()),
    })
}

#[derive(Debug, Clone)]
struct PreparedMonotoneCurve {
    x: Vec<f64>,
    y: Vec<f64>,
    tangent: Vec<f64>,
}

impl PreparedMonotoneCurve {
    fn new(points: &[[f64; 2]]) -> Self {
        let x = points.iter().map(|point| point[0]).collect::<Vec<_>>();
        let y = points.iter().map(|point| point[1]).collect::<Vec<_>>();
        let intervals = x
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect::<Vec<_>>();
        let secants = y
            .windows(2)
            .zip(&intervals)
            .map(|(window, interval)| (window[1] - window[0]) / interval)
            .collect::<Vec<_>>();
        let mut tangent = vec![0.0; points.len()];
        if points.len() == 2 {
            tangent.fill(secants[0]);
        } else {
            for index in 1..points.len() - 1 {
                let previous = secants[index - 1];
                let next = secants[index];
                if previous * next > 0.0 {
                    let w1 = 2.0 * intervals[index] + intervals[index - 1];
                    let w2 = intervals[index] + 2.0 * intervals[index - 1];
                    tangent[index] = (w1 + w2) / (w1 / previous + w2 / next);
                }
            }
            let endpoint = |h0: f64, h1: f64, d0: f64, d1: f64| {
                let estimate = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
                if estimate.signum() != d0.signum() {
                    0.0
                } else if d0.signum() != d1.signum() && estimate.abs() > 3.0 * d0.abs() {
                    3.0 * d0
                } else {
                    estimate
                }
            };
            tangent[0] = endpoint(intervals[0], intervals[1], secants[0], secants[1]);
            let last = points.len() - 1;
            tangent[last] = endpoint(
                intervals[last - 1],
                intervals[last - 2],
                secants[last - 1],
                secants[last - 2],
            );
        }
        Self { x, y, tangent }
    }

    fn evaluate(&self, value: f64) -> (f64, bool, bool) {
        let last = self.x.len() - 1;
        if value <= self.x[0] {
            return (
                self.y[0] + self.tangent[0] * (value - self.x[0]),
                value < self.x[0],
                false,
            );
        }
        if value >= self.x[last] {
            return (
                self.y[last] + self.tangent[last] * (value - self.x[last]),
                false,
                value > self.x[last],
            );
        }
        let upper = self.x.partition_point(|x| *x <= value);
        let lower = upper - 1;
        let interval = self.x[upper] - self.x[lower];
        let t = (value - self.x[lower]) / interval;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        (
            h00 * self.y[lower]
                + h10 * interval * self.tangent[lower]
                + h01 * self.y[upper]
                + h11 * interval * self.tangent[upper],
            false,
            false,
        )
    }
}

struct PreparedMeasuredResponse {
    density_to_layer: Matrix3<f64>,
    curves: [PreparedMonotoneCurve; 3],
    anchor_percentile: f64,
}

impl PreparedMeasuredResponse {
    fn new(calibration: &MeasuredNegativeResponseCalibration) -> Self {
        let channels = calibration.characteristic_curves.channels();
        Self {
            density_to_layer: rows_to_matrix3(&calibration.scanner_density_to_layer_density),
            curves: std::array::from_fn(|channel| PreparedMonotoneCurve::new(channels[channel])),
            anchor_percentile: calibration.white_anchor_percentile,
        }
    }

    fn evaluate(&self, measured_density: [f64; 3]) -> ([f64; 3], [bool; 3], [bool; 3]) {
        let layer = self.density_to_layer
            * Vector3::new(
                measured_density[0],
                measured_density[1],
                measured_density[2],
            );
        let evaluated =
            std::array::from_fn::<_, 3, _>(|channel| self.curves[channel].evaluate(layer[channel]));
        (
            std::array::from_fn(|channel| evaluated[channel].0),
            std::array::from_fn(|channel| evaluated[channel].1),
            std::array::from_fn(|channel| evaluated[channel].2),
        )
    }
}

/// Evaluate a measured response at one base-subtracted scanner-density triplet. Calibration tools
/// use this to score held-out patches before the completed model is admitted by the stricter
/// runtime validator.
pub fn evaluate_measured_negative_response_log_exposure(
    calibration: &MeasuredNegativeResponseCalibration,
    scanner_density: [f64; 3],
) -> Result<[f64; 3], String> {
    if scanner_density.iter().any(|value| !value.is_finite()) {
        return Err("scanner density must contain three finite values".to_string());
    }
    if calibration
        .characteristic_curves
        .channels()
        .iter()
        .any(|curve| curve.len() < 2)
    {
        return Err(
            "each negative-response curve requires at least two points for evaluation".to_string(),
        );
    }
    let prepared = PreparedMeasuredResponse::new(calibration);
    Ok(prepared.evaluate(scanner_density).0)
}

/// Reconstruct scene-linear light from a shared-anchor positive density image using the selected
/// bounded response model. A common log-exposure anchor preserves cross-channel relationships and
/// negative values before exponentiation preserve highlight headroom.
pub fn reconstruct_with_negative_response(
    positive_density: &Array3<f64>,
    density_diagnostics: &DensityDiagnostics,
    model: &NegativeResponseModel,
) -> NegativeResponseReconstructionResult {
    let (height, width, channels) = positive_density.dim();
    assert_eq!(channels, 3, "Expected 3-channel density image");
    let total_pixels = height.saturating_mul(width);
    let stride = total_pixels
        .div_ceil(RESPONSE_ESTIMATION_MAX_SAMPLES)
        .max(1);
    let slopes = model.diagnostics.applied_density_slopes;
    let prepared_measured = model.measured.as_ref().map(PreparedMeasuredResponse::new);
    let channel_anchors: [f64; 3] = if let Some(prepared) = prepared_measured.as_ref() {
        let mut exposure_samples: [Vec<f64>; 3] =
            std::array::from_fn(|_| Vec::with_capacity(total_pixels.div_ceil(stride)));
        for flat in (0..total_pixels).step_by(stride) {
            let y = flat / width;
            let x = flat % width;
            let measured_density = std::array::from_fn(|channel| {
                density_diagnostics.shared_robust_d_max - positive_density[[y, x, channel]]
            });
            let (scene_log_exposure, _, _) = prepared.evaluate(measured_density);
            for channel in 0..3 {
                if scene_log_exposure[channel].is_finite() {
                    exposure_samples[channel].push(scene_log_exposure[channel]);
                }
            }
        }
        std::array::from_fn(|channel| {
            sorted_percentile(&mut exposure_samples[channel], prepared.anchor_percentile)
        })
    } else {
        std::array::from_fn(|channel| density_diagnostics.robust_d_max[channel] / slopes[channel])
    };
    let shared_anchor = channel_anchors
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);
    let mut scene_linear = Array3::<f64>::zeros((height, width, channels));
    let mut highlight_headroom_samples = [0u64; 3];
    let mut maximum_scene_linear_value = [f64::NEG_INFINITY; 3];
    let mut curve_extrapolated_low_samples = [0u64; 3];
    let mut curve_extrapolated_high_samples = [0u64; 3];
    let mut curve_extrapolated_any_samples = 0u64;

    for (y0, y1) in streaming::tile_ranges(height, streaming::DEFAULT_TILE_ROWS) {
        for y in y0..y1 {
            for x in 0..width {
                let measured_density = std::array::from_fn(|channel| {
                    density_diagnostics.shared_robust_d_max - positive_density[[y, x, channel]]
                });
                let (scene_log_exposure, extrapolated_low, extrapolated_high) =
                    if let Some(prepared) = prepared_measured.as_ref() {
                        prepared.evaluate(measured_density)
                    } else {
                        (
                            std::array::from_fn(|channel| {
                                measured_density[channel] / slopes[channel]
                            }),
                            [false; 3],
                            [false; 3],
                        )
                    };
                if (0..3).any(|channel| extrapolated_low[channel] || extrapolated_high[channel]) {
                    curve_extrapolated_any_samples += 1;
                }
                for channel in 0..3 {
                    curve_extrapolated_low_samples[channel] += extrapolated_low[channel] as u64;
                    curve_extrapolated_high_samples[channel] += extrapolated_high[channel] as u64;
                    let positive_scene_density = shared_anchor - scene_log_exposure[channel];
                    if positive_scene_density < 0.0 {
                        highlight_headroom_samples[channel] += 1;
                    }
                    let value = signed_density_to_scene_linear(positive_scene_density);
                    scene_linear[[y, x, channel]] = value;
                    maximum_scene_linear_value[channel] =
                        maximum_scene_linear_value[channel].max(value);
                }
            }
        }
    }

    let mut high_samples: [Vec<f64>; 3] =
        std::array::from_fn(|_| Vec::with_capacity(total_pixels.div_ceil(stride)));
    for flat in (0..total_pixels).step_by(stride) {
        let y = flat / width;
        let x = flat % width;
        for channel in 0..3 {
            high_samples[channel].push(scene_linear[[y, x, channel]]);
        }
    }
    let scene_linear_high_percentile = std::array::from_fn(|channel| {
        sorted_percentile(&mut high_samples[channel], DENSITY_DMAX_PERCENTILE)
    });
    let curve_extrapolated_any_ratio =
        curve_extrapolated_any_samples as f64 / total_pixels.max(1) as f64;
    let coverage_review_required = prepared_measured.is_some()
        && curve_extrapolated_any_ratio > MEASURED_RESPONSE_MAX_TRUSTED_EXTRAPOLATION_RATIO;
    let review_required = model.diagnostics.review_required || coverage_review_required;
    let review_reason = if model.diagnostics.review_required {
        model.diagnostics.reason.clone()
    } else if coverage_review_required {
        format!(
            "measured negative-response curves extrapolated for {:.2}% of pixels, exceeding the {:.2}% trusted-coverage limit",
            curve_extrapolated_any_ratio * 100.0,
            MEASURED_RESPONSE_MAX_TRUSTED_EXTRAPOLATION_RATIO * 100.0
        )
    } else {
        "held-out-validated measured negative response covered the rendered frame within the trusted extrapolation limit"
            .to_string()
    };

    NegativeResponseReconstructionResult {
        scene_linear,
        diagnostics: NegativeResponseReconstructionDiagnostics {
            model: model.diagnostics.clone(),
            scene_white_log_exposure_anchor: shared_anchor,
            channel_white_log_exposure_anchors: channel_anchors,
            highlight_headroom_samples,
            scene_linear_high_percentile,
            maximum_scene_linear_value,
            signed_headroom_preserved: true,
            curve_extrapolated_low_samples,
            curve_extrapolated_high_samples,
            curve_extrapolated_any_samples,
            curve_extrapolated_any_ratio,
            curve_interpolation: if prepared_measured.is_some() {
                "monotone_piecewise_cubic_hermite_with_endpoint_tangent_extrapolation"
            } else {
                "linear_density_slope"
            },
            review_required,
            review_reason,
        },
    }
}

fn histogram_bin(value: f64, min_value: f64, max_value: f64) -> usize {
    let span = (max_value - min_value).max(1e-12);
    let t = ((value - min_value) / span).clamp(0.0, 1.0);
    ((t * (HISTOGRAM_BINS - 1) as f64).round() as usize).min(HISTOGRAM_BINS - 1)
}

fn percentile_from_histogram(hist: &[u64], min_value: f64, max_value: f64, percentile: f64) -> f64 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return min_value;
    }

    let target = ((total as f64 - 1.0) * percentile.clamp(0.0, 1.0)).round() as u64;
    let mut cumulative = 0u64;
    for (idx, count) in hist.iter().enumerate() {
        cumulative += *count;
        if cumulative > target {
            let span = (max_value - min_value).max(1e-12);
            let t = idx as f64 / (HISTOGRAM_BINS - 1) as f64;
            return min_value + t * span;
        }
    }

    max_value
}

fn bit_depth_max(bit_depth: u8) -> f64 {
    match bit_depth {
        14 => MAX_14BIT,
        _ => MAX_16BIT,
    }
}

fn normalize_transmittance(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(EPSILON_T, 1.0)
    } else {
        EPSILON_T
    }
}

fn normalize_sample(pixel: u16, max_val: f64) -> f64 {
    normalize_transmittance(pixel as f64 / max_val)
}

fn normalize_base_sample(value: f64, max_val: f64) -> f64 {
    normalize_transmittance(value / max_val)
}

fn density_after_base(pixel: u16, max_val: f64, base_density: f64) -> f64 {
    let t = normalize_sample(pixel, max_val);
    transmittance_to_density(t) - base_density
}

fn linear_division_value(pixel: u16, max_val: f64, base_transmittance: f64) -> f64 {
    let t = normalize_sample(pixel, max_val);
    base_transmittance / t
}

/// Phase 3: density-domain orange mask removal and negative inversion.
///
/// 1. Normalize u16 pixels to [EPSILON_T, 1.0]
/// 2. Subtract film-base density (orange mask) per channel
/// 3. Invert in density domain via `D_pos = D_max_robust - D'`
pub fn phase3_invert_with_diagnostics(
    img: &Array3<u16>,
    base_color: &[f64; 3],
    bit_depth: u8,
) -> Phase3Result {
    let (height, width, channels) = img.dim();
    assert_eq!(channels, 3, "Expected 3-channel image");

    let max_val = bit_depth_max(bit_depth);
    let base_transmittance: [f64; 3] =
        std::array::from_fn(|c| normalize_base_sample(base_color[c], max_val));
    let base_density: [f64; 3] =
        std::array::from_fn(|c| transmittance_to_density(base_transmittance[c]));

    let mut density_min = [f64::INFINITY; 3];
    let mut density_max = [f64::NEG_INFINITY; 3];
    let mut linear_min = [f64::INFINITY; 3];
    let mut linear_max = [f64::NEG_INFINITY; 3];

    for (y0, y1) in streaming::tile_ranges(height, streaming::DEFAULT_TILE_ROWS) {
        for y in y0..y1 {
            for x in 0..width {
                for c in 0..3 {
                    let d = density_after_base(img[[y, x, c]], max_val, base_density[c]);
                    density_min[c] = density_min[c].min(d);
                    density_max[c] = density_max[c].max(d);

                    let ratio =
                        linear_division_value(img[[y, x, c]], max_val, base_transmittance[c]);
                    linear_min[c] = linear_min[c].min(ratio);
                    linear_max[c] = linear_max[c].max(ratio);
                }
            }
        }
    }

    for c in 0..3 {
        if !density_min[c].is_finite() || !density_max[c].is_finite() {
            density_min[c] = 0.0;
            density_max[c] = 0.0;
        }
        if !linear_min[c].is_finite() || !linear_max[c].is_finite() {
            linear_min[c] = 1.0;
            linear_max[c] = 1.0;
        }
    }

    let mut density_hist = vec![vec![0u64; HISTOGRAM_BINS]; 3];
    let mut linear_hist = vec![vec![0u64; HISTOGRAM_BINS]; 3];

    for (y0, y1) in streaming::tile_ranges(height, streaming::DEFAULT_TILE_ROWS) {
        for y in y0..y1 {
            for x in 0..width {
                for c in 0..3 {
                    let d = density_after_base(img[[y, x, c]], max_val, base_density[c]);
                    density_hist[c][histogram_bin(d, density_min[c], density_max[c])] += 1;

                    let ratio =
                        linear_division_value(img[[y, x, c]], max_val, base_transmittance[c]);
                    linear_hist[c][histogram_bin(ratio, linear_min[c], linear_max[c])] += 1;
                }
            }
        }
    }

    let robust_d_max: [f64; 3] = std::array::from_fn(|c| {
        percentile_from_histogram(
            &density_hist[c],
            density_min[c],
            density_max[c],
            DENSITY_DMAX_PERCENTILE,
        )
    });
    let shared_robust_d_max = robust_d_max
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);
    let robust_linear_high: [f64; 3] = std::array::from_fn(|c| {
        percentile_from_histogram(
            &linear_hist[c],
            linear_min[c],
            linear_max[c],
            LINEAR_DIVISION_PERCENTILE,
        )
    });

    let mut positive_density = Array3::<f64>::zeros((height, width, channels));
    let mut highlight_headroom_samples = [0u64; 3];
    for (y0, y1) in streaming::tile_ranges(height, streaming::DEFAULT_TILE_ROWS) {
        for y in y0..y1 {
            for x in 0..width {
                for c in 0..3 {
                    let d = density_after_base(img[[y, x, c]], max_val, base_density[c]);
                    let inverted = shared_robust_d_max - d;
                    if inverted < 0.0 {
                        highlight_headroom_samples[c] += 1;
                    }
                    positive_density[[y, x, c]] = inverted;
                }
            }
        }
    }

    let density_stats = std::array::from_fn(|c| ChannelPercentileSummary {
        min_value: density_min[c],
        max_value: density_max[c],
        high_percentile: robust_d_max[c],
    });
    let linear_division = std::array::from_fn(|c| LinearDivisionDiagnostics {
        exact_max: linear_max[c],
        robust_high_percentile: robust_linear_high[c],
        percentile: LINEAR_DIVISION_PERCENTILE,
    });

    Phase3Result {
        positive_density,
        diagnostics: DensityDiagnostics {
            base_transmittance,
            base_density,
            robust_d_max,
            shared_robust_d_max,
            exact_d_max: density_max,
            d_max_percentile: DENSITY_DMAX_PERCENTILE,
            histogram_bins: HISTOGRAM_BINS,
            highlight_headroom_samples,
            epsilon_t: EPSILON_T,
            density_stats,
            linear_division,
        },
    }
}

/// Compatibility wrapper that returns only the positive density image.
pub fn phase3_invert(img: &Array3<u16>, base_color: &[f64; 3], bit_depth: u8) -> Array3<f64> {
    phase3_invert_with_diagnostics(img, base_color, bit_depth).positive_density
}

pub fn linear_division_diagnostic_image(
    img: &Array3<u16>,
    base_color: &[f64; 3],
    bit_depth: u8,
    robust_high_percentile: &[f64; 3],
) -> Array3<f64> {
    let (height, width, channels) = img.dim();
    assert_eq!(channels, 3, "Expected 3-channel image");

    let max_val = bit_depth_max(bit_depth);
    let base_transmittance: [f64; 3] =
        std::array::from_fn(|c| normalize_base_sample(base_color[c], max_val));
    let mut out = Array3::<f64>::zeros((height, width, channels));

    out.axis_chunks_iter_mut(Axis(0), streaming::DEFAULT_TILE_ROWS)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut out_chunk)| {
            let row_start = chunk_idx * streaming::DEFAULT_TILE_ROWS;
            let chunk_h = out_chunk.dim().0;
            for local_y in 0..chunk_h {
                let y = row_start + local_y;
                for x in 0..width {
                    for c in 0..3 {
                        let ratio =
                            linear_division_value(img[[y, x, c]], max_val, base_transmittance[c]);
                        out_chunk[[local_y, x, c]] = if robust_high_percentile[c] <= 1e-12 {
                            ratio
                        } else {
                            (ratio / robust_high_percentile[c]).clamp(0.0, 1.5)
                        };
                    }
                }
            }
        });

    out
}

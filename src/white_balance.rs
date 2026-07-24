//! Evidence-gated technical illuminant reconstruction and independent creative white balance.
//!
//! The technical stage operates on the scene-referred linear ProPhoto RGB master and uses a
//! Bradford chromatic adaptation only when spatially distributed, multi-tone neutral evidence is
//! coherent. Creative temperature/tint is a separate, reversible render transform and is never
//! baked into the technical master.

use crate::cli::{TechnicalWhiteBalanceMode, WhiteBalanceArgs};
use nalgebra::{Matrix3, Vector3};
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const D50_XY: [f64; 2] = [0.34567, 0.35850];
const D50_CCT_K: f64 = 5003.0;
const AUTO_MAX_NEUTRAL_SATURATION: f64 = 0.24;
const AUTO_MIN_CAST_LOG_CHROMA: f64 = 0.018;
const AUTO_MAX_LOG_CHROMA_MAD: f64 = 0.10;
const TECHNICAL_TINT_MAX_DUV: f64 = 0.018;
const CREATIVE_TINT_MAX_DUV: f64 = 0.014;
const CREATIVE_TEMPERATURE_MAX_MIRED_SHIFT: f64 = 70.0;

fn prophoto_to_xyz_d50() -> Matrix3<f64> {
    Matrix3::new(
        0.7976749, 0.1351917, 0.0313534, 0.2880402, 0.7118741, 0.0000857, 0.0, 0.0, 0.8252100,
    )
}

fn xyz_d50_to_prophoto() -> Matrix3<f64> {
    prophoto_to_xyz_d50()
        .try_inverse()
        .expect("ProPhoto RGB matrix is invertible")
}

fn bradford() -> Matrix3<f64> {
    Matrix3::new(
        0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296,
    )
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TechnicalWhiteBalanceDiagnostics {
    pub requested_mode: String,
    pub status: String,
    pub source: String,
    pub reason: String,
    pub applied: bool,
    pub confidence: f64,
    pub review_required: bool,
    pub sample_count: usize,
    pub sampled_pixel_count: usize,
    pub neutral_sample_fraction: f64,
    pub occupied_spatial_bin_count: usize,
    pub populated_luminance_band_count: usize,
    pub log_chroma_mad: Option<f64>,
    pub pre_adaptation_neutral_log_chroma: Option<f64>,
    pub post_adaptation_neutral_log_chroma: Option<f64>,
    pub estimated_source_white_prophoto: Option<[f64; 3]>,
    pub estimated_source_white_xy: Option<[f64; 2]>,
    pub estimated_source_cct_kelvin: Option<f64>,
    pub manual_temperature_kelvin: Option<f64>,
    pub manual_tint: Option<f64>,
    pub target_white_xy: [f64; 2],
    pub adaptation_method: String,
    pub adaptation_matrix_prophoto: [[f64; 3]; 3],
    pub preserves_signed_scene_headroom: bool,
}

#[derive(Debug, Clone)]
pub struct TechnicalWhiteBalanceResult {
    pub image: Array3<f64>,
    pub diagnostics: TechnicalWhiteBalanceDiagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CreativeWhiteBalanceDiagnostics {
    pub applied: bool,
    pub temperature: f64,
    pub tint: f64,
    pub temperature_mired_shift: f64,
    pub target_temperature_kelvin: f64,
    pub target_white_xy: [f64; 2],
    pub adaptation_method: String,
    pub adaptation_matrix_prophoto: [[f64; 3]; 3],
    pub separated_from_technical_master: bool,
}

#[derive(Debug, Clone)]
pub struct CreativeWhiteBalanceResult {
    pub image: Array3<f64>,
    pub diagnostics: CreativeWhiteBalanceDiagnostics,
}

#[derive(Debug, Clone, Copy)]
struct NeutralSample {
    x: usize,
    y: usize,
    luminance: f64,
    log_rg: f64,
    log_bg: f64,
}

#[derive(Debug, Clone)]
struct NeutralEstimate {
    sampled_pixel_count: usize,
    samples: Vec<NeutralSample>,
    occupied_spatial_bin_count: usize,
    populated_luminance_band_count: usize,
    log_chroma_mad: f64,
    source_white_prophoto: [f64; 3],
    source_white_xyz: Vector3<f64>,
    source_white_xy: [f64; 2],
    source_cct_kelvin: Option<f64>,
    pre_log_chroma: f64,
    confidence: f64,
    evidence_accepted: bool,
}

pub fn apply_technical_white_balance(
    image: Array3<f64>,
    settings: &WhiteBalanceArgs,
) -> TechnicalWhiteBalanceResult {
    match settings.technical_white_balance {
        TechnicalWhiteBalanceMode::Off => technical_identity_result(
            image,
            settings,
            "disabled",
            "technical white balance was explicitly disabled",
            false,
        ),
        TechnicalWhiteBalanceMode::Manual => {
            let source_xy = cct_tint_to_xy(
                settings.technical_temperature_kelvin,
                settings.technical_tint * TECHNICAL_TINT_MAX_DUV,
            );
            let source_xyz = xy_to_xyz(source_xy);
            let matrix = chromatic_adaptation_prophoto(source_xyz, xy_to_xyz(D50_XY));
            let source_rgb = normalize_white_rgb(xyz_d50_to_prophoto() * source_xyz);
            let post_rgb = normalize_white_rgb(matrix * source_rgb);
            let mut image = image;
            apply_matrix_in_place(&mut image, &matrix);
            TechnicalWhiteBalanceResult {
                image,
                diagnostics: TechnicalWhiteBalanceDiagnostics {
                    requested_mode: settings.technical_white_balance.as_str().to_string(),
                    status: "applied_manual".to_string(),
                    source: "manual_source_illuminant".to_string(),
                    reason: "Bradford-adapted the explicit source CCT/tint to the ProPhoto D50 technical working white"
                        .to_string(),
                    applied: true,
                    confidence: 1.0,
                    review_required: false,
                    sample_count: 0,
                    sampled_pixel_count: 0,
                    neutral_sample_fraction: 0.0,
                    occupied_spatial_bin_count: 0,
                    populated_luminance_band_count: 0,
                    log_chroma_mad: None,
                    pre_adaptation_neutral_log_chroma: Some(rgb_log_chroma(source_rgb)),
                    post_adaptation_neutral_log_chroma: Some(rgb_log_chroma(post_rgb)),
                    estimated_source_white_prophoto: None,
                    estimated_source_white_xy: Some(source_xy),
                    estimated_source_cct_kelvin: Some(settings.technical_temperature_kelvin),
                    manual_temperature_kelvin: Some(settings.technical_temperature_kelvin),
                    manual_tint: Some(settings.technical_tint),
                    target_white_xy: D50_XY,
                    adaptation_method: "Bradford_CAT_in_XYZ_composed_for_linear_ProPhoto_D50"
                        .to_string(),
                    adaptation_matrix_prophoto: matrix_to_array(&matrix),
                    preserves_signed_scene_headroom: true,
                },
            }
        }
        TechnicalWhiteBalanceMode::Auto => apply_auto_technical_white_balance(image, settings),
    }
}

fn apply_auto_technical_white_balance(
    image: Array3<f64>,
    settings: &WhiteBalanceArgs,
) -> TechnicalWhiteBalanceResult {
    let Some(estimate) = estimate_neutral_white(&image) else {
        return technical_identity_result(
            image,
            settings,
            "insufficient_evidence",
            "no finite, positive, low-chroma scene samples supported an illuminant estimate",
            true,
        );
    };

    let neutral_sample_fraction =
        estimate.samples.len() as f64 / estimate.sampled_pixel_count.max(1) as f64;
    if !estimate.evidence_accepted {
        return TechnicalWhiteBalanceResult {
            image,
            diagnostics: diagnostics_from_estimate(
                settings,
                &estimate,
                Matrix3::identity(),
                "insufficient_evidence",
                "automatic illuminant estimate was not applied because neutral support was not sufficiently spatially distributed, multi-tone, and coherent",
                false,
                estimate.pre_log_chroma > 0.05,
                neutral_sample_fraction,
                estimate.pre_log_chroma,
            ),
        };
    }
    if estimate.pre_log_chroma < AUTO_MIN_CAST_LOG_CHROMA {
        return TechnicalWhiteBalanceResult {
            image,
            diagnostics: diagnostics_from_estimate(
                settings,
                &estimate,
                Matrix3::identity(),
                "not_needed",
                "accepted neutral evidence was already consistent with the D50 technical working white",
                false,
                false,
                neutral_sample_fraction,
                estimate.pre_log_chroma,
            ),
        };
    }

    let matrix = chromatic_adaptation_prophoto(estimate.source_white_xyz, xy_to_xyz(D50_XY));
    let post_rgb = normalize_white_rgb(matrix * Vector3::from(estimate.source_white_prophoto));
    let post_log_chroma = rgb_log_chroma(post_rgb);
    if !matrix.iter().all(|value| value.is_finite())
        || post_log_chroma >= estimate.pre_log_chroma * 0.35
    {
        return TechnicalWhiteBalanceResult {
            image,
            diagnostics: diagnostics_from_estimate(
                settings,
                &estimate,
                Matrix3::identity(),
                "rejected_no_improvement",
                "proposed chromatic adaptation did not sufficiently reduce the held neutral estimate error",
                false,
                true,
                neutral_sample_fraction,
                estimate.pre_log_chroma,
            ),
        };
    }

    let mut image = image;
    apply_matrix_in_place(&mut image, &matrix);
    TechnicalWhiteBalanceResult {
        image,
        diagnostics: diagnostics_from_estimate(
            settings,
            &estimate,
            matrix,
            "applied_auto",
            "spatially distributed, multi-tone neutral evidence supported a Bradford adaptation to D50",
            true,
            false,
            neutral_sample_fraction,
            post_log_chroma,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn diagnostics_from_estimate(
    settings: &WhiteBalanceArgs,
    estimate: &NeutralEstimate,
    matrix: Matrix3<f64>,
    status: &str,
    reason: &str,
    applied: bool,
    review_required: bool,
    neutral_sample_fraction: f64,
    post_log_chroma: f64,
) -> TechnicalWhiteBalanceDiagnostics {
    TechnicalWhiteBalanceDiagnostics {
        requested_mode: settings.technical_white_balance.as_str().to_string(),
        status: status.to_string(),
        source: "scene_neutral_evidence".to_string(),
        reason: reason.to_string(),
        applied,
        confidence: estimate.confidence,
        review_required,
        sample_count: estimate.samples.len(),
        sampled_pixel_count: estimate.sampled_pixel_count,
        neutral_sample_fraction,
        occupied_spatial_bin_count: estimate.occupied_spatial_bin_count,
        populated_luminance_band_count: estimate.populated_luminance_band_count,
        log_chroma_mad: Some(estimate.log_chroma_mad),
        pre_adaptation_neutral_log_chroma: Some(estimate.pre_log_chroma),
        post_adaptation_neutral_log_chroma: Some(post_log_chroma),
        estimated_source_white_prophoto: Some(estimate.source_white_prophoto),
        estimated_source_white_xy: Some(estimate.source_white_xy),
        estimated_source_cct_kelvin: estimate.source_cct_kelvin,
        manual_temperature_kelvin: None,
        manual_tint: None,
        target_white_xy: D50_XY,
        adaptation_method: if applied {
            "Bradford_CAT_in_XYZ_composed_for_linear_ProPhoto_D50".to_string()
        } else {
            "identity".to_string()
        },
        adaptation_matrix_prophoto: matrix_to_array(&matrix),
        preserves_signed_scene_headroom: true,
    }
}

fn technical_identity_result(
    image: Array3<f64>,
    settings: &WhiteBalanceArgs,
    status: &str,
    reason: &str,
    review_required: bool,
) -> TechnicalWhiteBalanceResult {
    TechnicalWhiteBalanceResult {
        image,
        diagnostics: TechnicalWhiteBalanceDiagnostics {
            requested_mode: settings.technical_white_balance.as_str().to_string(),
            status: status.to_string(),
            source: "none".to_string(),
            reason: reason.to_string(),
            applied: false,
            confidence: 0.0,
            review_required,
            sample_count: 0,
            sampled_pixel_count: 0,
            neutral_sample_fraction: 0.0,
            occupied_spatial_bin_count: 0,
            populated_luminance_band_count: 0,
            log_chroma_mad: None,
            pre_adaptation_neutral_log_chroma: None,
            post_adaptation_neutral_log_chroma: None,
            estimated_source_white_prophoto: None,
            estimated_source_white_xy: None,
            estimated_source_cct_kelvin: None,
            manual_temperature_kelvin: None,
            manual_tint: None,
            target_white_xy: D50_XY,
            adaptation_method: "identity".to_string(),
            adaptation_matrix_prophoto: matrix_to_array(&Matrix3::identity()),
            preserves_signed_scene_headroom: true,
        },
    }
}

pub fn apply_creative_white_balance(
    image: &Array3<f64>,
    temperature: f64,
    tint: f64,
) -> CreativeWhiteBalanceResult {
    let (matrix, diagnostics) = creative_white_balance_transform(temperature, tint);
    CreativeWhiteBalanceResult {
        image: if diagnostics.applied {
            apply_matrix(image, &matrix)
        } else {
            image.clone()
        },
        diagnostics,
    }
}

pub fn apply_creative_white_balance_owned(
    mut image: Array3<f64>,
    temperature: f64,
    tint: f64,
) -> CreativeWhiteBalanceResult {
    let (matrix, diagnostics) = creative_white_balance_transform(temperature, tint);
    if diagnostics.applied {
        apply_matrix_in_place(&mut image, &matrix);
    }
    CreativeWhiteBalanceResult { image, diagnostics }
}

pub fn creative_white_balance_diagnostics(
    temperature: f64,
    tint: f64,
) -> CreativeWhiteBalanceDiagnostics {
    creative_white_balance_transform(temperature, tint).1
}

fn creative_white_balance_transform(
    temperature: f64,
    tint: f64,
) -> (Matrix3<f64>, CreativeWhiteBalanceDiagnostics) {
    let temperature = temperature.clamp(-1.0, 1.0);
    let tint = tint.clamp(-1.0, 1.0);
    let mired_shift = temperature * CREATIVE_TEMPERATURE_MAX_MIRED_SHIFT;
    let target_mired = (1_000_000.0 / D50_CCT_K + mired_shift).clamp(110.0, 335.0);
    let target_temperature_kelvin = 1_000_000.0 / target_mired;
    // Positive UI tint means a magenta creative cast, opposite to the source-illuminant
    // direction used by technical correction.
    let target_xy = cct_tint_to_xy(target_temperature_kelvin, -tint * CREATIVE_TINT_MAX_DUV);
    let applied = temperature.abs() > 1e-12 || tint.abs() > 1e-12;
    let matrix = if applied {
        chromatic_adaptation_prophoto(xy_to_xyz(D50_XY), xy_to_xyz(target_xy))
    } else {
        Matrix3::identity()
    };
    (
        matrix,
        CreativeWhiteBalanceDiagnostics {
            applied,
            temperature,
            tint,
            temperature_mired_shift: mired_shift,
            target_temperature_kelvin,
            target_white_xy: target_xy,
            adaptation_method: if applied {
                "Bradford_CAT_from_D50_to_creative_white".to_string()
            } else {
                "identity".to_string()
            },
            adaptation_matrix_prophoto: matrix_to_array(&matrix),
            separated_from_technical_master: true,
        },
    )
}

fn estimate_neutral_white(image: &Array3<f64>) -> Option<NeutralEstimate> {
    let (height, width, channels) = image.dim();
    if channels < 3 || height == 0 || width == 0 {
        return None;
    }
    let stride = (((height * width) as f64 / 200_000.0).sqrt().ceil() as usize).max(1);
    let mut candidates = Vec::<NeutralSample>::new();
    let mut sampled_pixel_count = 0usize;
    for y in (0..height).step_by(stride) {
        for x in (0..width).step_by(stride) {
            sampled_pixel_count += 1;
            let rgb = [image[[y, x, 0]], image[[y, x, 1]], image[[y, x, 2]]];
            if !rgb.iter().all(|value| value.is_finite() && *value > 1e-8) {
                continue;
            }
            let maximum = rgb[0].max(rgb[1]).max(rgb[2]);
            let minimum = rgb[0].min(rgb[1]).min(rgb[2]);
            let saturation = (maximum - minimum) / maximum.max(1e-12);
            if saturation > AUTO_MAX_NEUTRAL_SATURATION {
                continue;
            }
            let luminance = 0.288_040_2 * rgb[0] + 0.711_874_1 * rgb[1] + 0.000_085_7 * rgb[2];
            if !luminance.is_finite() || luminance <= 1e-8 {
                continue;
            }
            candidates.push(NeutralSample {
                x,
                y,
                luminance,
                log_rg: (rgb[0] / rgb[1]).ln(),
                log_bg: (rgb[2] / rgb[1]).ln(),
            });
        }
    }
    if candidates.len() < 16 {
        return None;
    }

    let mut luminances = candidates
        .iter()
        .map(|sample| sample.luminance)
        .collect::<Vec<_>>();
    luminances.sort_by(f64::total_cmp);
    let low = quantile_sorted(&luminances, 0.05)?;
    let high = quantile_sorted(&luminances, 0.95)?;
    let band1 = quantile_sorted(&luminances, 0.33)?;
    let band2 = quantile_sorted(&luminances, 0.67)?;
    candidates.retain(|sample| sample.luminance >= low && sample.luminance <= high);
    if candidates.len() < 16 {
        return None;
    }

    let mut spatial_bins = BTreeSet::<usize>::new();
    let mut luminance_bands = BTreeSet::<usize>::new();
    let mut log_rg = Vec::with_capacity(candidates.len());
    let mut log_bg = Vec::with_capacity(candidates.len());
    for sample in &candidates {
        let bin_x = (sample.x * 4 / width.max(1)).min(3);
        let bin_y = (sample.y * 4 / height.max(1)).min(3);
        spatial_bins.insert(bin_y * 4 + bin_x);
        luminance_bands.insert(if sample.luminance < band1 {
            0
        } else if sample.luminance < band2 {
            1
        } else {
            2
        });
        log_rg.push(sample.log_rg);
        log_bg.push(sample.log_bg);
    }
    let median_rg = median(&mut log_rg)?;
    let median_bg = median(&mut log_bg)?;
    let mad_rg = median_absolute_deviation(&log_rg, median_rg)?;
    let mad_bg = median_absolute_deviation(&log_bg, median_bg)?;
    let log_chroma_mad = (mad_rg * mad_rg + mad_bg * mad_bg).sqrt();
    let source_white_prophoto =
        normalize_white_rgb(Vector3::new(median_rg.exp(), 1.0, median_bg.exp()));
    let mut source_white_xyz = prophoto_to_xyz_d50() * source_white_prophoto;
    source_white_xyz /= source_white_xyz.y.max(1e-12);
    let source_white_xy = xyz_to_xy(source_white_xyz)?;
    let pre_log_chroma = (median_rg * median_rg + median_bg * median_bg).sqrt();

    let required_samples = (sampled_pixel_count / 800).clamp(32, 256);
    let sample_score = (candidates.len() as f64 / required_samples as f64).min(1.0);
    let spatial_score = (spatial_bins.len() as f64 / 8.0).min(1.0);
    let band_score = luminance_bands.len() as f64 / 3.0;
    let dispersion_score = (1.0 - log_chroma_mad / AUTO_MAX_LOG_CHROMA_MAD).clamp(0.0, 1.0);
    let confidence =
        0.30 * sample_score + 0.30 * spatial_score + 0.20 * band_score + 0.20 * dispersion_score;
    let evidence_accepted = candidates.len() >= required_samples
        && spatial_bins.len() >= 4
        && luminance_bands.len() >= 2
        && log_chroma_mad <= AUTO_MAX_LOG_CHROMA_MAD
        && confidence >= 0.62;

    Some(NeutralEstimate {
        sampled_pixel_count,
        samples: candidates,
        occupied_spatial_bin_count: spatial_bins.len(),
        populated_luminance_band_count: luminance_bands.len(),
        log_chroma_mad,
        source_white_prophoto: source_white_prophoto.into(),
        source_white_xyz,
        source_white_xy,
        source_cct_kelvin: xy_to_cct(source_white_xy),
        pre_log_chroma,
        confidence,
        evidence_accepted,
    })
}

fn chromatic_adaptation_prophoto(
    source_white_xyz: Vector3<f64>,
    target_white_xyz: Vector3<f64>,
) -> Matrix3<f64> {
    let bradford = bradford();
    let inverse = bradford
        .try_inverse()
        .expect("Bradford matrix is invertible");
    let source_lms = bradford * source_white_xyz;
    let target_lms = bradford * target_white_xyz;
    let diagonal = Matrix3::from_diagonal(&Vector3::new(
        target_lms.x / source_lms.x.max(1e-12),
        target_lms.y / source_lms.y.max(1e-12),
        target_lms.z / source_lms.z.max(1e-12),
    ));
    xyz_d50_to_prophoto() * inverse * diagonal * bradford * prophoto_to_xyz_d50()
}

fn apply_matrix(image: &Array3<f64>, matrix: &Matrix3<f64>) -> Array3<f64> {
    let (height, width, channels) = image.dim();
    let mut output = image.clone();
    if channels < 3 {
        return output;
    }
    for y in 0..height {
        for x in 0..width {
            let input = Vector3::new(image[[y, x, 0]], image[[y, x, 1]], image[[y, x, 2]]);
            let mapped = matrix * input;
            output[[y, x, 0]] = mapped.x;
            output[[y, x, 1]] = mapped.y;
            output[[y, x, 2]] = mapped.z;
        }
    }
    output
}

fn apply_matrix_in_place(image: &mut Array3<f64>, matrix: &Matrix3<f64>) {
    let (height, width, channels) = image.dim();
    if channels < 3 {
        return;
    }
    for y in 0..height {
        for x in 0..width {
            let input = Vector3::new(image[[y, x, 0]], image[[y, x, 1]], image[[y, x, 2]]);
            let mapped = matrix * input;
            image[[y, x, 0]] = mapped.x;
            image[[y, x, 1]] = mapped.y;
            image[[y, x, 2]] = mapped.z;
        }
    }
}

fn cct_tint_to_xy(temperature_kelvin: f64, tint_duv: f64) -> [f64; 2] {
    let temperature = temperature_kelvin.clamp(1667.0, 25_000.0);
    let x = if temperature <= 4000.0 {
        -0.266_123_9e9 / temperature.powi(3) - 0.234_358_0e6 / temperature.powi(2)
            + 0.877_695_6e3 / temperature
            + 0.179_910
    } else {
        -3.025_846_9e9 / temperature.powi(3)
            + 2.107_037_9e6 / temperature.powi(2)
            + 0.222_634_7e3 / temperature
            + 0.240_390
    };
    let y = if temperature <= 2222.0 {
        -1.106_381_4 * x.powi(3) - 1.348_110_20 * x.powi(2) + 2.185_558_32 * x - 0.202_196_83
    } else if temperature <= 4000.0 {
        -0.954_947_6 * x.powi(3) - 1.374_185_93 * x.powi(2) + 2.091_370_15 * x - 0.167_488_67
    } else {
        3.081_758_0 * x.powi(3) - 5.873_386_70 * x.powi(2) + 3.751_129_97 * x - 0.370_014_83
    };
    let denominator = -2.0 * x + 12.0 * y + 3.0;
    let u = 4.0 * x / denominator;
    let v = 6.0 * y / denominator + tint_duv;
    let inverse_denominator = 2.0 * u - 8.0 * v + 4.0;
    [
        (3.0 * u / inverse_denominator).clamp(0.05, 0.75),
        (2.0 * v / inverse_denominator).clamp(0.05, 0.80),
    ]
}

fn xy_to_xyz(xy: [f64; 2]) -> Vector3<f64> {
    let y = xy[1].max(1e-12);
    Vector3::new(xy[0] / y, 1.0, (1.0 - xy[0] - xy[1]) / y)
}

fn xyz_to_xy(xyz: Vector3<f64>) -> Option<[f64; 2]> {
    let sum = xyz.x + xyz.y + xyz.z;
    (sum.is_finite() && sum > 1e-12).then_some([xyz.x / sum, xyz.y / sum])
}

fn xy_to_cct(xy: [f64; 2]) -> Option<f64> {
    let denominator = 0.1858 - xy[1];
    if denominator.abs() < 1e-9 {
        return None;
    }
    let n = (xy[0] - 0.3320) / denominator;
    let cct = -449.0 * n.powi(3) + 3525.0 * n.powi(2) - 6823.3 * n + 5520.33;
    (cct.is_finite() && (1000.0..=50_000.0).contains(&cct)).then_some(cct)
}

fn normalize_white_rgb(rgb: Vector3<f64>) -> Vector3<f64> {
    let luminance = 0.288_040_2 * rgb.x + 0.711_874_1 * rgb.y + 0.000_085_7 * rgb.z;
    rgb / luminance.max(1e-12)
}

fn rgb_log_chroma(rgb: Vector3<f64>) -> f64 {
    let rg = (rgb.x.max(1e-12) / rgb.y.max(1e-12)).ln();
    let bg = (rgb.z.max(1e-12) / rgb.y.max(1e-12)).ln();
    (rg * rg + bg * bg).sqrt()
}

fn matrix_to_array(matrix: &Matrix3<f64>) -> [[f64; 3]; 3] {
    [
        [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
        [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
        [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
    ]
}

fn quantile_sorted(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let position = quantile.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    Some(values[lower] * (1.0 - fraction) + values[upper] * fraction)
}

fn median(values: &mut [f64]) -> Option<f64> {
    values.sort_by(f64::total_cmp);
    quantile_sorted(values, 0.5)
}

fn median_absolute_deviation(values: &[f64], center: f64) -> Option<f64> {
    let mut deviations = values
        .iter()
        .map(|value| (value - center).abs())
        .collect::<Vec<_>>();
    median(&mut deviations)
}

use nalgebra::{Matrix3, Vector3};
use ndarray::parallel::prelude::*;
use ndarray::{Array3, Axis};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::constants::{
    BRADFORD_LMS_TO_XYZ, BRADFORD_XYZ_TO_LMS, D50_WHITE, PROPHOTO_TO_XYZ_D50, XYZ_D50_TO_PROPHOTO,
};
use crate::streaming;

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
const MIN_CHANNEL_ANCHOR_SUPPORT: usize = 64;
const CHANNEL_NAMES: [&str; 3] = ["red", "green", "blue"];

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
    pub neutral_balance_scale: [f64; 3],
    pub image_matrix_pre_scale_clipped_low_ratio: [f64; 3],
    pub image_matrix_pre_scale_clipped_high_ratio: [f64; 3],
    pub image_matrix_exposure_scale: f64,
}

pub struct ColorspaceMappingResult {
    pub prophoto: Array3<f64>,
    pub diagnostics: ColorspaceDiagnostics,
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
        .filter_map(|(idx, low_support)| {
            low_support.then(|| {
                format!(
                    "{}={}",
                    CHANNEL_NAMES[idx], diagnostics.channel_anchor_counts[idx]
                )
            })
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

fn estimate_neutral_stats(img: &Array3<f64>) -> ([f64; 3], usize) {
    let (h, w, _) = img.dim();
    let total = h * w;
    let mut sums = [0.0f64; 3];
    let mut count = 0usize;

    for y in 0..h {
        for x in 0..w {
            let r = img[[y, x, 0]];
            let g = img[[y, x, 1]];
            let b = img[[y, x, 2]];
            let mu = (r + g + b) / 3.0;
            if mu < 1e-9 {
                continue;
            }

            let max_dev = (r - mu).abs().max((g - mu).abs()).max((b - mu).abs());
            if max_dev / mu <= NEUTRAL_THRESHOLD {
                sums[0] += r;
                sums[1] += g;
                sums[2] += b;
                count += 1;
            }
        }
    }

    if count == 0 || (count as f64) < (total as f64 * MIN_NEUTRAL_FRACTION) {
        ([1.0, 1.0, 1.0], count)
    } else {
        let n = count as f64;
        ([sums[0] / n, sums[1] / n, sums[2] / n], count)
    }
}

fn estimate_channel_anchors(img: &Array3<f64>) -> ([[f64; 3]; 3], [usize; 3]) {
    let (h, w, _) = img.dim();
    let mut sums = [[0.0f64; 3]; 3];
    let mut counts = [0usize; 3];

    for y in 0..h {
        for x in 0..w {
            let pixel = [img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]]];
            let max_v = pixel[0].max(pixel[1]).max(pixel[2]);
            let min_v = pixel[0].min(pixel[1]).min(pixel[2]);
            if max_v < 1e-6 {
                continue;
            }

            let saturation = (max_v - min_v) / max_v;
            if saturation < MIN_DOMINANT_SATURATION {
                continue;
            }

            let mut order = [(0usize, pixel[0]), (1usize, pixel[1]), (2usize, pixel[2])];
            order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if order[0].1 < order[1].1 * DOMINANT_RATIO_THRESHOLD {
                continue;
            }

            let dominant = order[0].0;
            for c in 0..3 {
                sums[dominant][c] += pixel[c];
            }
            counts[dominant] += 1;
        }
    }

    let anchors = std::array::from_fn(|dominant| {
        if counts[dominant] == 0 {
            let mut unit = [0.0f64; 3];
            unit[dominant] = 1.0;
            unit
        } else {
            let n = counts[dominant] as f64;
            [
                sums[dominant][0] / n,
                sums[dominant][1] / n,
                sums[dominant][2] / n,
            ]
        }
    });

    (anchors, counts)
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
    pre_scale_channel_max: [f64; 3],
    pre_scale_channel_high_percentile: [f64; 3],
    pre_scale_clipped_high_ratio: [f64; 3],
    pre_scale_clipped_low_ratio: [f64; 3],
    exposure_scale: f64,
}

fn evaluate_mapping_stats(img: &Array3<f64>, combined: &Matrix3<f64>) -> MappingStats {
    let (h, w, _c) = img.dim();
    let total_pixels = (h * w).max(1) as f64;
    let mut pre_scale_channel_max = [0.0f64; 3];
    let mut pre_scale_clipped_high = [0u64; 3];
    let mut pre_scale_clipped_low = [0u64; 3];

    for y in 0..h {
        for x in 0..w {
            let mapped = combined * Vector3::new(img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]]);
            for c in 0..3 {
                let value = mapped[c];
                if value.is_finite() {
                    pre_scale_channel_max[c] = pre_scale_channel_max[c].max(value.max(0.0));
                    if value > 1.0 {
                        pre_scale_clipped_high[c] += 1;
                    }
                    if value < 0.0 {
                        pre_scale_clipped_low[c] += 1;
                    }
                }
            }
        }
    }

    let histogram_max: [f64; 3] = std::array::from_fn(|c| pre_scale_channel_max[c].max(1.0));
    let mut histograms = vec![vec![0u64; HISTOGRAM_BINS]; 3];
    for y in 0..h {
        for x in 0..w {
            let mapped = combined * Vector3::new(img[[y, x, 0]], img[[y, x, 1]], img[[y, x, 2]]);
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

    MappingStats {
        pre_scale_channel_max,
        pre_scale_channel_high_percentile,
        pre_scale_clipped_high_ratio: std::array::from_fn(|c| {
            pre_scale_clipped_high[c] as f64 / total_pixels
        }),
        pre_scale_clipped_low_ratio: std::array::from_fn(|c| {
            pre_scale_clipped_low[c] as f64 / total_pixels
        }),
        exposure_scale,
    }
}

fn low_gamut_fallback_reason(stats: &MappingStats) -> Option<String> {
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
            "image-derived colorspace matrix would clamp too many negative channel values (max channel {:.1}%, total {:.1}%); using neutral-balance gamut fallback",
            max_channel_low * 100.0,
            total_low * 100.0
        ))
    } else {
        None
    }
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

pub fn direct_density_render_fallback_reason(
    ica_diagnostics: &ColorspaceDiagnostics,
    direct_diagnostics: &ColorspaceDiagnostics,
) -> Option<String> {
    if !has_destructive_gamut_fallback(ica_diagnostics) {
        return None;
    }

    let (ica_max_low, ica_total_low) = image_matrix_low_clip_summary(ica_diagnostics);
    let (direct_max_low, direct_total_low) = image_matrix_low_clip_summary(direct_diagnostics);
    let relative_limit = ica_total_low * DIRECT_RENDER_LOW_CLIP_RELATIVE_LIMIT;
    let absolute_limit = (ica_total_low - DIRECT_RENDER_LOW_CLIP_ABSOLUTE_MARGIN).max(0.0);
    let direct_is_materially_safer =
        direct_total_low <= relative_limit || direct_total_low <= absolute_limit;

    if !direct_is_materially_safer {
        return None;
    }

    Some(format!(
        "ICA-separated transmittance produced destructive colorspace negative-gamut clipping (max channel {:.1}%, total {:.1}%); direct density transmittance reduced the image-matrix low clipping to max channel {:.1}%, total {:.1}%",
        ica_max_low * 100.0,
        ica_total_low * 100.0,
        direct_max_low * 100.0,
        direct_total_low * 100.0
    ))
}

fn neutral_balance_mapping(img: &Array3<f64>) -> (Matrix3<f64>, [f64; 3]) {
    let (neutral_rgb, _) = estimate_neutral_stats(img);
    let neutral_mean = ((neutral_rgb[0] + neutral_rgb[1] + neutral_rgb[2]) / 3.0).max(1e-6);
    let scale: [f64; 3] =
        std::array::from_fn(|c| (neutral_mean / neutral_rgb[c].max(1e-6)).clamp(0.25, 4.0));
    (
        Matrix3::new(scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, scale[2]),
        scale,
    )
}

pub fn estimate_work_to_xyz(img: &Array3<f64>) -> ColorspaceDiagnostics {
    let prior = prophoto_to_xyz_d50_matrix();
    let (neutral_rgb, neutral_pixel_count) = estimate_neutral_stats(img);
    let (anchor_rgb, anchor_counts) = estimate_channel_anchors(img);

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
        mapping_strategy: "image_derived_matrix",
        neutral_balance_scale: [1.0; 3],
        image_matrix_pre_scale_clipped_low_ratio: [0.0; 3],
        image_matrix_pre_scale_clipped_high_ratio: [0.0; 3],
        image_matrix_exposure_scale: 1.0,
    }
}

/// Map an image from work RGB to ProPhoto RGB (D50) via:
/// 1. Estimate an explicit work->XYZ matrix from neutral and dominant pixels
/// 2. Estimate the source white in XYZ from the neutral set
/// 3. Bradford-adapt that white to D50
pub fn map_to_prophoto_d50_with_diagnostics(img: &Array3<f64>) -> ColorspaceMappingResult {
    let (h, w, _c) = img.dim();

    let mut diagnostics = estimate_work_to_xyz(img);
    let work_to_xyz = const_to_matrix3(&diagnostics.work_to_xyz);
    let cat = bradford_cat(&diagnostics.source_white);
    let xyz_to_pro = xyz_d50_to_prophoto_matrix();
    let image_matrix = xyz_to_pro * cat * work_to_xyz;
    let image_matrix_stats = evaluate_mapping_stats(img, &image_matrix);
    let weak_anchor_fallback_reason = weak_channel_anchor_warning(&diagnostics)
        .map(|warning| format!("{warning}; using neutral-balance fallback for rendered output"));
    let gamut_fallback_reason = low_gamut_fallback_reason(&image_matrix_stats);
    let (combined, neutral_balance_scale, mapping_strategy) =
        if weak_anchor_fallback_reason.is_some() {
            let (fallback, scale) = neutral_balance_mapping(img);
            (fallback, scale, "neutral_balance_weak_anchor_fallback")
        } else if gamut_fallback_reason.is_some() {
            let (fallback, scale) = neutral_balance_mapping(img);
            (fallback, scale, "neutral_balance_gamut_fallback")
        } else {
            (image_matrix, [1.0; 3], "image_derived_matrix")
        };
    let chosen_stats = if weak_anchor_fallback_reason.is_some() || gamut_fallback_reason.is_some() {
        evaluate_mapping_stats(img, &combined)
    } else {
        image_matrix_stats.clone()
    };
    let exposure_scale = chosen_stats.exposure_scale;

    let mut out = Array3::<f64>::zeros((h, w, 3));
    let post_scale_clipped_high = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    let post_scale_clipped_low = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
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
                    let mapped = (combined * pixel) / exposure_scale;
                    for c in 0..3 {
                        if mapped[c] > 1.0 {
                            post_scale_clipped_high[c].fetch_add(1, Ordering::Relaxed);
                        }
                        if mapped[c] < 0.0 {
                            post_scale_clipped_low[c].fetch_add(1, Ordering::Relaxed);
                        }
                        out_chunk[[local_y, x, c]] = mapped[c].clamp(0.0, 1.0);
                    }
                }
            }
        });

    let total_pixels = (h * w).max(1) as f64;
    diagnostics.pre_scale_channel_max = chosen_stats.pre_scale_channel_max;
    diagnostics.pre_scale_channel_high_percentile = chosen_stats.pre_scale_channel_high_percentile;
    diagnostics.pre_scale_clipped_high_ratio = chosen_stats.pre_scale_clipped_high_ratio;
    diagnostics.pre_scale_clipped_low_ratio = chosen_stats.pre_scale_clipped_low_ratio;
    diagnostics.post_scale_clipped_high_ratio = std::array::from_fn(|c| {
        post_scale_clipped_high[c].load(Ordering::Relaxed) as f64 / total_pixels
    });
    diagnostics.post_scale_clipped_low_ratio = std::array::from_fn(|c| {
        post_scale_clipped_low[c].load(Ordering::Relaxed) as f64 / total_pixels
    });
    diagnostics.exposure_scale = exposure_scale;
    diagnostics.weak_anchor_fallback_used = weak_anchor_fallback_reason.is_some();
    diagnostics.weak_anchor_fallback_reason = weak_anchor_fallback_reason;
    diagnostics.gamut_fallback_used = gamut_fallback_reason.is_some();
    diagnostics.gamut_fallback_reason = gamut_fallback_reason;
    diagnostics.mapping_strategy = mapping_strategy;
    diagnostics.neutral_balance_scale = neutral_balance_scale;
    diagnostics.image_matrix_pre_scale_clipped_low_ratio =
        image_matrix_stats.pre_scale_clipped_low_ratio;
    diagnostics.image_matrix_pre_scale_clipped_high_ratio =
        image_matrix_stats.pre_scale_clipped_high_ratio;
    diagnostics.image_matrix_exposure_scale = image_matrix_stats.exposure_scale;

    ColorspaceMappingResult {
        prophoto: out,
        diagnostics,
    }
}

pub fn map_to_prophoto_d50(img: &Array3<f64>) -> Array3<f64> {
    map_to_prophoto_d50_with_diagnostics(img).prophoto
}

use ndarray::parallel::prelude::*;
use ndarray::{Array3, Axis};

use crate::constants::{EPSILON_T, MAX_14BIT, MAX_16BIT};
use crate::streaming;

const HISTOGRAM_BINS: usize = 4096;
const DENSITY_DMAX_PERCENTILE: f64 = 0.998;
const LINEAR_DIVISION_PERCENTILE: f64 = 0.998;

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
    pub exact_d_max: [f64; 3],
    pub d_max_percentile: f64,
    pub histogram_bins: usize,
    pub clamped_to_zero: [u64; 3],
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
    img.mapv(|v| (v as f64 / max_val).max(EPSILON_T))
}

/// Convert transmittance to optical density: D = -log10(T).
pub fn transmittance_to_density(t: f64) -> f64 {
    -(t.max(EPSILON_T)).log10()
}

/// Convert optical density back to transmittance: T = 10^(-D).
pub fn density_to_transmittance(d: f64) -> f64 {
    10.0f64.powf(-d)
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

fn density_after_base(pixel: u16, max_val: f64, base_density: f64) -> f64 {
    let t = (pixel as f64 / max_val).max(EPSILON_T);
    transmittance_to_density(t) - base_density
}

fn linear_division_value(pixel: u16, max_val: f64, base_transmittance: f64) -> f64 {
    let t = (pixel as f64 / max_val).max(EPSILON_T);
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
        std::array::from_fn(|c| (base_color[c] / max_val).max(EPSILON_T));
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
    let robust_linear_high: [f64; 3] = std::array::from_fn(|c| {
        percentile_from_histogram(
            &linear_hist[c],
            linear_min[c],
            linear_max[c],
            LINEAR_DIVISION_PERCENTILE,
        )
    });

    let mut positive_density = Array3::<f64>::zeros((height, width, channels));
    let mut clamped_to_zero = [0u64; 3];
    for (y0, y1) in streaming::tile_ranges(height, streaming::DEFAULT_TILE_ROWS) {
        for y in y0..y1 {
            for x in 0..width {
                for c in 0..3 {
                    let d = density_after_base(img[[y, x, c]], max_val, base_density[c]);
                    let inverted = robust_d_max[c] - d;
                    if inverted <= 0.0 {
                        clamped_to_zero[c] += 1;
                        positive_density[[y, x, c]] = 0.0;
                    } else {
                        positive_density[[y, x, c]] = inverted;
                    }
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
            exact_d_max: density_max,
            d_max_percentile: DENSITY_DMAX_PERCENTILE,
            histogram_bins: HISTOGRAM_BINS,
            clamped_to_zero,
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
        std::array::from_fn(|c| (base_color[c] / max_val).max(EPSILON_T));
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

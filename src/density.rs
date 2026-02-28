use ndarray::Array3;

use crate::constants::{EPSILON_T, MAX_14BIT, MAX_16BIT};
use crate::streaming::tile_ranges;

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

/// Compute the percentile value from a sorted slice.
/// `p` is in [0, 100].
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    if lo == hi || hi >= sorted.len() {
        sorted[lo.min(sorted.len() - 1)]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Phase 3: density-domain orange mask removal and negative inversion.
///
/// 1. Normalize u16 pixels to [EPSILON_T, 1.0]
/// 2. Subtract film-base density (orange mask) per channel
/// 3. Invert in density domain and normalize output to [0, 1]
pub fn phase3_invert(
    img: &Array3<u16>,
    base_color: &[f64; 3],
    bit_depth: u8,
) -> Array3<f64> {
    let (height, width, channels) = img.dim();
    assert_eq!(channels, 3, "Expected 3-channel image");

    let max_val = match bit_depth {
        14 => MAX_14BIT,
        _ => MAX_16BIT,
    };

    // Compute base density per channel
    let base_d: [f64; 3] = std::array::from_fn(|c| {
        let base_t = (base_color[c] / max_val).max(EPSILON_T);
        transmittance_to_density(base_t)
    });

    // Step 3.1 + 3.2: normalize, convert to density, subtract base — tiled for streaming
    let mut density = Array3::<f64>::zeros((height, width, channels));
    let tiles = tile_ranges(height, 512);
    for (start, end) in &tiles {
        for y in *start..*end {
            for x in 0..width {
                for c in 0..3 {
                    let t = (img[[y, x, c]] as f64 / max_val).max(EPSILON_T);
                    let d = transmittance_to_density(t);
                    density[[y, x, c]] = d - base_d[c];
                }
            }
        }
    }

    // Step 3.3: compute D_max per channel (99.5th percentile of base-subtracted density)
    let npixels = height * width;
    let mut d_max = [0.0f64; 3];
    for c in 0..3 {
        let mut vals: Vec<f64> = Vec::with_capacity(npixels);
        for y in 0..height {
            for x in 0..width {
                vals.push(density[[y, x, c]]);
            }
        }
        vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        d_max[c] = percentile_sorted(&vals, 99.5);
    }

    // Invert: D_pos = D_max - D
    let mut inverted = Array3::<f64>::zeros((height, width, channels));
    for (start, end) in &tiles {
        for y in *start..*end {
            for x in 0..width {
                for c in 0..3 {
                    inverted[[y, x, c]] = d_max[c] - density[[y, x, c]];
                }
            }
        }
    }

    // Normalize output to [0, 1] using per-channel 0.5th and 99.5th percentiles
    for c in 0..3 {
        let mut vals: Vec<f64> = Vec::with_capacity(npixels);
        for y in 0..height {
            for x in 0..width {
                vals.push(inverted[[y, x, c]]);
            }
        }
        vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = percentile_sorted(&vals, 0.5);
        let hi = percentile_sorted(&vals, 99.5);
        let range = (hi - lo).max(EPSILON_T);

        for y in 0..height {
            for x in 0..width {
                let v = (inverted[[y, x, c]] - lo) / range;
                inverted[[y, x, c]] = v.clamp(0.0, 1.0);
            }
        }
    }

    inverted
}

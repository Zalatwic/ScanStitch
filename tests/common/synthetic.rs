#![allow(dead_code)]

use ndarray::Array3;

/// Create a constant-color image of shape (h, w, 3).
pub fn constant_image(h: usize, w: usize, rgb: [u16; 3]) -> Array3<u16> {
    let mut arr = Array3::<u16>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            arr[[y, x, 0]] = rgb[0];
            arr[[y, x, 1]] = rgb[1];
            arr[[y, x, 2]] = rgb[2];
        }
    }
    arr
}

/// Create an image with a colored border and different-colored center.
/// `border_rows` specifies how many rows at top/bottom are border color (black).
pub fn image_with_borders(
    h: usize,
    w: usize,
    border_rows: usize,
    center_rgb: [u16; 3],
) -> Array3<u16> {
    let mut arr = Array3::<u16>::zeros((h, w, 3));
    for y in border_rows..(h - border_rows) {
        for x in 0..w {
            arr[[y, x, 0]] = center_rgb[0];
            arr[[y, x, 1]] = center_rgb[1];
            arr[[y, x, 2]] = center_rgb[2];
        }
    }
    arr
}

/// Create a synthetic film negative scan image.
/// - `border_rows`: black scanner border rows at top/bottom
/// - `rebate_cols`: film rebate (sprocket area) columns at left/right filled with `base_rgb`
/// - `base_rgb`: film base / rebate color (e.g., orange mask)
/// - `content_rgb`: content area color
pub fn film_negative_image(
    h: usize,
    w: usize,
    border_rows: usize,
    rebate_cols: usize,
    base_rgb: [u16; 3],
    content_rgb: [u16; 3],
) -> Array3<u16> {
    let mut arr = Array3::<u16>::zeros((h, w, 3));
    for y in border_rows..(h - border_rows) {
        for x in 0..w {
            if x < rebate_cols || x >= (w - rebate_cols) {
                // Rebate / film base area
                arr[[y, x, 0]] = base_rgb[0];
                arr[[y, x, 1]] = base_rgb[1];
                arr[[y, x, 2]] = base_rgb[2];
            } else {
                // Content area
                arr[[y, x, 0]] = content_rgb[0];
                arr[[y, x, 1]] = content_rgb[1];
                arr[[y, x, 2]] = content_rgb[2];
            }
        }
    }
    arr
}

/// Create a horizontal gradient image from `left_rgb` to `right_rgb`.
pub fn gradient_image(h: usize, w: usize, left_rgb: [u16; 3], right_rgb: [u16; 3]) -> Array3<u16> {
    let mut arr = Array3::<u16>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            let t = x as f64 / (w.max(1) - 1).max(1) as f64;
            for c in 0..3 {
                let v = left_rgb[c] as f64 * (1.0 - t) + right_rgb[c] as f64 * t;
                arr[[y, x, c]] = v.round() as u16;
            }
        }
    }
    arr
}

/// Add deterministic pseudo-random noise to an image using a simple LCG.
/// `amplitude` is the max absolute noise value (symmetrically distributed).
pub fn add_noise(arr: &mut Array3<u16>, seed: u64, amplitude: u16) {
    let mut state = seed;
    let shape = arr.shape().to_vec();
    let amp = amplitude as i32;

    for y in 0..shape[0] {
        for x in 0..shape[1] {
            for c in 0..shape[2] {
                // LCG: state = (a * state + c) mod m
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Extract a noise value in [-amplitude, +amplitude]
                let raw = ((state >> 33) as i32) % (2 * amp + 1) - amp;
                let val = arr[[y, x, c]] as i32 + raw;
                arr[[y, x, c]] = val.clamp(0, u16::MAX as i32) as u16;
            }
        }
    }
}

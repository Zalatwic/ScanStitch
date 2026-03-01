use ndarray::{s, Array3};

/// Compute the median of a slice of f64 values. The input is modified (sorted).
fn median_f64(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    if n % 2 == 0 {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    } else {
        vals[n / 2]
    }
}

/// Compute the median luminance for each row of an (H, W, 3) u16 image.
fn compute_row_luminance(img: &Array3<u16>) -> Vec<f64> {
    let (h, w, _) = img.dim();
    let mut lum = Vec::with_capacity(h);
    for y in 0..h {
        let mut row_vals: Vec<f64> = Vec::with_capacity(w);
        for x in 0..w {
            let r = img[[y, x, 0]] as f64;
            let g = img[[y, x, 1]] as f64;
            let b = img[[y, x, 2]] as f64;
            row_vals.push((r + g + b) / 3.0);
        }
        lum.push(median_f64(&mut row_vals));
    }
    lum
}

/// Detect and remove scanner dead-zone borders from a 16-bit RGB image
/// using a discrete derivative (edge detection) approach on row-by-row
/// median luminance.
///
/// Returns `(cropped_image, top_rows_removed, bottom_rows_removed)`.
///
/// The algorithm:
/// 1. Compute median luminance for every row.
/// 2. Scan from the outside in, computing the discrete derivative
///    (absolute difference between adjacent rows).
/// 3. The scanner mask boundary produces a massive luminance spike.
/// 4. Apply safety margin inward from the spike.
/// 5. If no spike is found, assume no border is present.
pub fn remove_borders(
    img: &Array3<u16>,
    safety_margin: usize,
) -> (Array3<u16>, usize, usize) {
    let (h, w, _) = img.dim();
    if h < 4 {
        return (img.clone(), 0, 0);
    }

    // Step 1: Compute median luminance for every row
    let row_lum = compute_row_luminance(img);

    // Content reference: median luminance of the central 50% of rows
    let q1 = h / 4;
    let q3 = 3 * h / 4;
    let mut center_lums: Vec<f64> = row_lum[q1..q3].to_vec();
    let content_lum = median_f64(&mut center_lums);

    // Spike threshold: 20% of content luminance, floor of 500
    let spike_threshold = (content_lum * 0.2).max(500.0);

    log::debug!(
        "Border detection: image {}x{}, content_lum = {:.1}, spike_threshold = {:.1}",
        h, w, content_lum, spike_threshold
    );

    // Step 2 & 3: Scan from outside in using discrete derivative
    let max_top = (h / 3).min(h - 1);
    let min_bottom = 2 * h / 3;

    // Top border: scan from y=0 downward, find first spike
    let mut top_crop: usize = 0;
    for y in 0..max_top {
        let delta = (row_lum[y + 1] - row_lum[y]).abs();
        if delta > spike_threshold {
            top_crop = y + 1 + safety_margin;
            break;
        }
    }

    // Bottom border: scan from y=h-1 upward, find first spike
    let mut bottom_crop: usize = 0;
    for y in (min_bottom..h).rev() {
        if y == 0 {
            break;
        }
        let delta = (row_lum[y - 1] - row_lum[y]).abs();
        if delta > spike_threshold {
            bottom_crop = (h - y) + safety_margin;
            break;
        }
    }

    // No borders detected
    if top_crop == 0 && bottom_crop == 0 {
        log::info!("Border detection: no border spikes detected. No cropping.");
        return (img.clone(), 0, 0);
    }

    // Safety: don't consume entire image
    if top_crop + bottom_crop >= h {
        log::info!("Border detection: borders would consume entire image. No cropping.");
        return (img.clone(), 0, 0);
    }

    log::info!(
        "Border detection: removing {} top rows, {} bottom rows (from {}x{} image)",
        top_crop, bottom_crop, h, w
    );

    let end_row = h - bottom_crop;
    let cropped = img.slice(s![top_crop..end_row, .., ..]).to_owned();

    (cropped, top_crop, bottom_crop)
}

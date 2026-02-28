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

/// Per-row statistics: median luminance and MAD (median absolute deviation).
struct RowStats {
    median_lum: f64,
    mad: f64,
}

/// Compute per-row statistics for an (H, W, 3) u16 image.
fn compute_row_stats(img: &Array3<u16>) -> Vec<RowStats> {
    let (h, w, _) = img.dim();
    let mut stats = Vec::with_capacity(h);

    for y in 0..h {
        let mut lums: Vec<f64> = Vec::with_capacity(w);
        for x in 0..w {
            let r = img[[y, x, 0]] as f64;
            let g = img[[y, x, 1]] as f64;
            let b = img[[y, x, 2]] as f64;
            lums.push((r + g + b) / 3.0);
        }

        let median_lum = median_f64(&mut lums.clone());

        // MAD = median(|lum_i - median_lum|)
        let mut abs_devs: Vec<f64> = lums.iter().map(|&l| (l - median_lum).abs()).collect();
        let mad = median_f64(&mut abs_devs);

        stats.push(RowStats { median_lum, mad });
    }

    stats
}

/// Detect and remove scanner dead-zone borders from a 16-bit RGB image.
///
/// Returns `(cropped_image, top_rows_removed, bottom_rows_removed)`.
///
/// The algorithm:
/// 1. Compute per-row median luminance and MAD.
/// 2. Use the central rows' median luminance as the content reference.
/// 3. A row is a border candidate if it has low MAD AND its luminance differs
///    significantly from the content reference.
/// 4. Find contiguous border regions from top/bottom with gap tolerance.
/// 5. Apply safety margin with variance spike check.
pub fn remove_borders(
    img: &Array3<u16>,
    safety_margin: usize,
) -> (Array3<u16>, usize, usize) {
    let (h, w, c) = img.dim();
    if h < 4 {
        return (img.clone(), 0, 0);
    }

    let stats = compute_row_stats(img);

    // Determine content reference: median luminance of the central 50% of rows
    let q1 = h / 4;
    let q3 = 3 * h / 4;
    let mut center_lums: Vec<f64> = stats[q1..q3].iter().map(|s| s.median_lum).collect();
    let content_lum = median_f64(&mut center_lums);

    // Collect MAD values for content rows to understand typical content MAD
    let mut center_mads: Vec<f64> = stats[q1..q3].iter().map(|s| s.mad).collect();
    let content_mad = median_f64(&mut center_mads);

    log::debug!(
        "Border detection: image {}x{}x{}, content_lum = {:.1}, content_mad = {:.1}",
        h, w, c, content_lum, content_mad
    );

    // Adaptive MAD threshold: 10% of global median MAD, floor of 50.0
    let all_mads: Vec<f64> = stats.iter().map(|s| s.mad).collect();
    let mut all_mads_sorted = all_mads.clone();
    let global_median_mad = median_f64(&mut all_mads_sorted);
    let mad_threshold = (global_median_mad * 0.1).max(50.0);

    // Luminance difference threshold: a border row must differ from content by at least
    // this amount. Use a fraction of the content luminance, with a minimum.
    // For black borders on a ~6333 lum content: diff = 6333, well above threshold.
    // For white borders (65000) on ~6333 content: diff = ~58667, well above threshold.
    // For a constant image: diff = 0 for all rows, so nothing qualifies.
    let lum_diff_threshold = (content_lum * 0.3).max(500.0);

    log::debug!(
        "Border detection: mad_threshold = {:.1}, lum_diff_threshold = {:.1}, global_median_mad = {:.1}",
        mad_threshold, lum_diff_threshold, global_median_mad
    );

    // Classify border rows: low MAD AND luminance far from content
    let is_border_row: Vec<bool> = stats
        .iter()
        .map(|s| {
            let lum_diff = (s.median_lum - content_lum).abs();
            s.mad <= mad_threshold && lum_diff > lum_diff_threshold
        })
        .collect();

    // Check if any border rows exist at all
    let any_borders = is_border_row.iter().any(|&b| b);
    if !any_borders {
        log::info!("Border detection: no border rows detected. No cropping.");
        return (img.clone(), 0, 0);
    }

    // Find top border extent with gap tolerance of 2
    let top_border = find_border_extent(&is_border_row, true, 2);

    // Find bottom border extent with gap tolerance of 2
    let bottom_border = find_border_extent(&is_border_row, false, 2);

    log::debug!(
        "Border detection: raw top_border = {}, raw bottom_border = {}",
        top_border, bottom_border
    );

    // Safety check: ensure we don't overlap
    if top_border + bottom_border >= h {
        log::info!("Border detection: borders would consume entire image. No cropping.");
        return (img.clone(), 0, 0);
    }

    // Apply safety margin: try to extend crop by safety_margin rows beyond detected border,
    // but stop if we hit a row whose luminance is close to content (variance spike in luminance).
    let top_crop = extend_with_safety(
        top_border,
        safety_margin,
        &stats,
        true,
        h,
        content_lum,
        lum_diff_threshold,
    );
    let bottom_crop = extend_with_safety(
        bottom_border,
        safety_margin,
        &stats,
        false,
        h,
        content_lum,
        lum_diff_threshold,
    );

    // Final safety: don't consume entire image
    if top_crop + bottom_crop >= h {
        log::info!(
            "Border detection: safety-extended borders would consume entire image. No cropping."
        );
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

/// Find the extent of a border region starting from top (forward=true) or bottom (forward=false).
/// Allows small gaps of up to `gap_tolerance` non-border rows within the border region.
fn find_border_extent(is_border: &[bool], from_top: bool, gap_tolerance: usize) -> usize {
    let n = is_border.len();
    if n == 0 {
        return 0;
    }

    let mut extent = 0;
    let mut gap_run = 0;

    for i in 0..n {
        let idx = if from_top { i } else { n - 1 - i };
        if is_border[idx] {
            extent = i + 1;
            gap_run = 0;
        } else {
            gap_run += 1;
            if gap_run > gap_tolerance {
                break;
            }
        }
    }

    extent
}

/// Try to extend the crop by `margin` rows beyond `detected` border rows.
/// For each additional row, check if its luminance is close to content (meaning it IS content).
/// If it is content, stop extending.
fn extend_with_safety(
    detected: usize,
    margin: usize,
    stats: &[RowStats],
    from_top: bool,
    total_rows: usize,
    content_lum: f64,
    lum_diff_threshold: f64,
) -> usize {
    let mut crop = detected;

    for _ in 0..margin {
        let check_idx = if from_top {
            crop
        } else {
            if crop >= total_rows {
                break;
            }
            total_rows - 1 - crop
        };

        if check_idx >= total_rows {
            break;
        }

        let lum_diff = (stats[check_idx].median_lum - content_lum).abs();
        if lum_diff <= lum_diff_threshold {
            // This row looks like content — don't crop further
            break;
        }

        crop += 1;
    }

    crop
}

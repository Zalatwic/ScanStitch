use ndarray::Array3;

/// Result of film base (rebate) detection on left/right edges.
pub struct BaseDetection {
    /// Confidence that the left edge contains film base [0, 1].
    pub left_confidence: f64,
    /// Confidence that the right edge contains film base [0, 1].
    pub right_confidence: f64,
    /// Estimated base RGB color as f64 values (in u16 range).
    pub base_color: [f64; 3],
    /// Block confidence grid for the left strip (rows x cols of f64 in [0,1]).
    pub left_base_mask: Vec<Vec<f64>>,
    /// Block confidence grid for the right strip (rows x cols of f64 in [0,1]).
    pub right_base_mask: Vec<Vec<f64>>,
}

/// Per-block statistics.
struct BlockStats {
    median_rgb: [f64; 3],
    mad_lum: f64,
}

/// Detect unexposed film base (rebate) regions on left/right edges.
///
/// Operates on a cropped u16 image (h, w, 3). Returns confidence scores
/// for each side plus the estimated base color.
pub fn detect_film_base(img: &Array3<u16>) -> BaseDetection {
    let (h, w, _) = (img.shape()[0], img.shape()[1], img.shape()[2]);

    // Strip width: 5% of image width, min 10px, max half the image
    let strip_w = (w as f64 * 0.05).round().max(10.0).min((w / 2) as f64) as usize;

    // Interior region: everything not in the strips
    let interior_start = strip_w;
    let interior_end = w.saturating_sub(strip_w).max(interior_start + 1);

    // Compute interior median color for comparison
    let interior_median = region_median_color(img, h, interior_start, interior_end);

    // Process left strip
    let (left_conf, left_mask, left_color) =
        process_strip(img, h, 0, strip_w, &interior_median);

    // Process right strip
    let (right_conf, right_mask, right_color) =
        process_strip(img, h, w.saturating_sub(strip_w), w, &interior_median);

    // Determine base color from the higher-confidence side
    let base_color = if left_conf > 0.3 || right_conf > 0.3 {
        if left_conf >= right_conf {
            left_color
        } else {
            right_color
        }
    } else {
        [0.0, 0.0, 0.0]
    };

    BaseDetection {
        left_confidence: left_conf,
        right_confidence: right_conf,
        base_color,
        left_base_mask: left_mask,
        right_base_mask: right_mask,
    }
}

/// Compute the median RGB of a vertical region spanning full height.
fn region_median_color(img: &Array3<u16>, h: usize, x_start: usize, x_end: usize) -> [f64; 3] {
    let mut channels: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for y in 0..h {
        for x in x_start..x_end {
            for c in 0..3 {
                channels[c].push(img[[y, x, c]] as f64);
            }
        }
    }
    let mut result = [0.0; 3];
    for c in 0..3 {
        result[c] = fast_median(&mut channels[c]);
    }
    result
}

/// Process one edge strip and return (confidence, block_mask, base_color).
fn process_strip(
    img: &Array3<u16>,
    h: usize,
    x_start: usize,
    x_end: usize,
    interior_median: &[f64; 3],
) -> (f64, Vec<Vec<f64>>, [f64; 3]) {
    let strip_w = x_end - x_start;
    if strip_w == 0 || h == 0 {
        return (0.0, vec![], [0.0; 3]);
    }

    let grid_rows = 7;
    let grid_cols = 7;

    // Compute block stats
    let block_h = h / grid_rows;
    let block_w = strip_w / grid_cols;
    if block_h == 0 || block_w == 0 {
        return (0.0, vec![], [0.0; 3]);
    }

    let mut blocks: Vec<Vec<BlockStats>> = Vec::with_capacity(grid_rows);
    for r in 0..grid_rows {
        let mut row_blocks = Vec::with_capacity(grid_cols);
        let y0 = r * block_h;
        let y1 = if r == grid_rows - 1 { h } else { y0 + block_h };
        for c_idx in 0..grid_cols {
            let bx0 = x_start + c_idx * block_w;
            let bx1 = if c_idx == grid_cols - 1 { x_end } else { bx0 + block_w };
            row_blocks.push(compute_block_stats(img, y0, y1, bx0, bx1));
        }
        blocks.push(row_blocks);
    }

    let total_blocks = grid_rows * grid_cols;

    // Step 4: classify low-texture blocks (MAD < 300.0)
    let mad_threshold = 300.0;
    let mut low_texture: Vec<(usize, usize)> = Vec::new();
    for r in 0..grid_rows {
        for c_idx in 0..grid_cols {
            if blocks[r][c_idx].mad_lum < mad_threshold {
                low_texture.push((r, c_idx));
            }
        }
    }

    if low_texture.is_empty() {
        let mask = vec![vec![0.0; grid_cols]; grid_rows];
        return (0.0, mask, [0.0; 3]);
    }

    // Step 5: find dominant chroma cluster among low-texture blocks
    // Chroma: (R-G)/sum and (R-B)/sum
    let chromas: Vec<[f64; 2]> = low_texture
        .iter()
        .map(|&(r, c_idx)| {
            let rgb = &blocks[r][c_idx].median_rgb;
            block_chroma(rgb)
        })
        .collect();

    // Median chroma = cluster center
    let mut c0s: Vec<f64> = chromas.iter().map(|c| c[0]).collect();
    let mut c1s: Vec<f64> = chromas.iter().map(|c| c[1]).collect();
    let center = [fast_median(&mut c0s), fast_median(&mut c1s)];

    let chroma_threshold = 0.05;
    let mut base_blocks: Vec<(usize, usize)> = Vec::new();
    for (i, &(r, c_idx)) in low_texture.iter().enumerate() {
        let dist = ((chromas[i][0] - center[0]).powi(2) + (chromas[i][1] - center[1]).powi(2))
            .sqrt();
        if dist < chroma_threshold {
            base_blocks.push((r, c_idx));
        }
    }

    if base_blocks.is_empty() {
        let mask = vec![vec![0.0; grid_cols]; grid_rows];
        return (0.0, mask, [0.0; 3]);
    }

    // Base fraction and raw confidence
    let base_fraction = base_blocks.len() as f64 / total_blocks as f64;
    let mut confidence = (base_fraction / 0.4).min(1.0);

    // Compute strip base color via trimmed mean (10% trim)
    let strip_base_color = trimmed_mean_color(&blocks, &base_blocks, 0.10);

    // CRITICAL: compare strip color to interior color
    // If they're too similar, this isn't real base — it's just content
    let strip_chroma = block_chroma(&strip_base_color);
    let interior_chroma = block_chroma(interior_median);
    let chroma_dist = ((strip_chroma[0] - interior_chroma[0]).powi(2)
        + (strip_chroma[1] - interior_chroma[1]).powi(2))
    .sqrt();

    // Also compare luminance
    let strip_lum = strip_base_color[0] * 0.2126 + strip_base_color[1] * 0.7152 + strip_base_color[2] * 0.0722;
    let interior_lum = interior_median[0] * 0.2126 + interior_median[1] * 0.7152 + interior_median[2] * 0.0722;
    let lum_ratio = if strip_lum.max(interior_lum) > 0.0 {
        (strip_lum - interior_lum).abs() / strip_lum.max(interior_lum)
    } else {
        0.0
    };

    // If chroma and luminance are both similar to interior, this is not base
    if chroma_dist < 0.03 && lum_ratio < 0.15 {
        confidence *= 0.1; // Drastically reduce — strip looks like content
    } else if chroma_dist < 0.05 && lum_ratio < 0.25 {
        confidence *= 0.3; // Moderately reduce
    }

    // Build mask
    let mut mask = vec![vec![0.0; grid_cols]; grid_rows];
    for &(r, c_idx) in &base_blocks {
        mask[r][c_idx] = 1.0;
    }

    (confidence, mask, strip_base_color)
}

/// Compute chroma coordinates for an RGB triplet.
fn block_chroma(rgb: &[f64; 3]) -> [f64; 2] {
    let sum = rgb[0] + rgb[1] + rgb[2];
    if sum < 1.0 {
        [0.0, 0.0]
    } else {
        [(rgb[0] - rgb[1]) / sum, (rgb[0] - rgb[2]) / sum]
    }
}

/// Compute block statistics: median RGB and MAD of luminance.
fn compute_block_stats(
    img: &Array3<u16>,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
) -> BlockStats {
    let n = (y1 - y0) * (x1 - x0);
    if n == 0 {
        return BlockStats {
            median_rgb: [0.0; 3],
            mad_lum: 0.0,
        };
    }

    let mut r_vals = Vec::with_capacity(n);
    let mut g_vals = Vec::with_capacity(n);
    let mut b_vals = Vec::with_capacity(n);
    let mut lum_vals = Vec::with_capacity(n);

    for y in y0..y1 {
        for x in x0..x1 {
            let r = img[[y, x, 0]] as f64;
            let g = img[[y, x, 1]] as f64;
            let b = img[[y, x, 2]] as f64;
            r_vals.push(r);
            g_vals.push(g);
            b_vals.push(b);
            lum_vals.push(r * 0.2126 + g * 0.7152 + b * 0.0722);
        }
    }

    let med_r = fast_median(&mut r_vals);
    let med_g = fast_median(&mut g_vals);
    let med_b = fast_median(&mut b_vals);
    let med_lum = fast_median(&mut lum_vals);

    // MAD = median of |x - median(x)|
    let mut abs_devs: Vec<f64> = lum_vals.iter().map(|&v| (v - med_lum).abs()).collect();
    let mad = fast_median(&mut abs_devs);

    BlockStats {
        median_rgb: [med_r, med_g, med_b],
        mad_lum: mad,
    }
}

/// Trimmed mean color from identified base blocks (trim fraction from each end).
fn trimmed_mean_color(
    blocks: &[Vec<BlockStats>],
    base_blocks: &[(usize, usize)],
    trim_frac: f64,
) -> [f64; 3] {
    if base_blocks.is_empty() {
        return [0.0; 3];
    }

    let mut result = [0.0; 3];
    for c in 0..3 {
        let mut vals: Vec<f64> = base_blocks
            .iter()
            .map(|&(r, ci)| blocks[r][ci].median_rgb[c])
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = vals.len();
        let trim_count = (n as f64 * trim_frac).floor() as usize;
        let start = trim_count;
        let end = n - trim_count;
        if end <= start {
            // All trimmed, just use overall median
            result[c] = vals[n / 2];
        } else {
            let sum: f64 = vals[start..end].iter().sum();
            result[c] = sum / (end - start) as f64;
        }
    }
    result
}

/// In-place median via nth_element-style partial sort.
fn fast_median(vals: &mut Vec<f64>) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let n = vals.len();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    }
}

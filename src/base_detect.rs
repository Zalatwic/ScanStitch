use ndarray::Array3;

/// Result of film base (rebate) detection on left/right edges.
pub struct BaseDetection {
    /// Confidence that the left edge contains film base [0, 1].
    pub left_confidence: f64,
    /// Confidence that the right edge contains film base [0, 1].
    pub right_confidence: f64,
    /// Estimated base RGB color as f64 values (in u16 range).
    pub base_color: [f64; 3],
    /// Edge strip width used for left/right detection.
    pub strip_width: usize,
    /// Block confidence grid for the left strip (rows x cols of f64 in [0,1]).
    pub left_base_mask: Vec<Vec<f64>>,
    /// Block confidence grid for the right strip (rows x cols of f64 in [0,1]).
    pub right_base_mask: Vec<Vec<f64>>,
}

/// Post-stitch base estimate reconciliation against stable component priors.
#[derive(Debug, Clone)]
pub struct BaseReconciliation {
    pub base_color: [f64; 3],
    pub confidence: f64,
    pub source: &'static str,
    pub reason: String,
    pub raw_working_base_color: [f64; 3],
    pub raw_working_confidence: f64,
    pub component_consensus_base_color: Option<[f64; 3]>,
    pub component_consensus_confidence: Option<f64>,
    pub component_consensus_relative_spread: Option<[f64; 3]>,
    pub working_vs_consensus_relative_delta: Option<[f64; 3]>,
    pub edge_balance_ratio: f64,
}

/// A vertically contiguous base-like region found across image width.
#[derive(Debug, Clone)]
pub struct BaseRegion {
    pub x_start: usize,
    pub x_end: usize,
    pub confidence: f64,
    pub median_rgb: [f64; 3],
    pub mad_lum: f64,
}

/// Full-width scan for base-like vertical regions.
#[derive(Debug, Clone)]
pub struct VerticalBaseScan {
    pub block_width: usize,
    pub regions: Vec<BaseRegion>,
    pub column_mask: Vec<f64>,
    pub dominant_color: [f64; 3],
}

/// Per-block statistics.
#[derive(Debug, Clone)]
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
    let interior_median = region_median_color(img, h, interior_start, interior_end);

    let (mut left_conf, left_mask, mut left_color) =
        process_strip(img, h, 0, strip_w, &interior_median);
    let (mut right_conf, right_mask, mut right_color) =
        process_strip(img, h, w.saturating_sub(strip_w), w, &interior_median);

    // Cross-check against the full-width vertical scan so split-frame gaps and
    // edge rebates share a consistent base estimate.
    let vertical = detect_vertical_base_regions(img);
    if let Some(region) = vertical.regions.first() {
        if region.x_start <= vertical.block_width {
            left_conf = left_conf.max(region.confidence);
            left_color = region.median_rgb;
        }
    }
    if let Some(region) = vertical.regions.last() {
        if region.x_end + vertical.block_width >= w {
            right_conf = right_conf.max(region.confidence);
            right_color = region.median_rgb;
        }
    }

    let base_color = if left_conf > 0.3 || right_conf > 0.3 {
        if left_conf >= right_conf {
            left_color
        } else {
            right_color
        }
    } else {
        vertical.dominant_color
    };

    BaseDetection {
        left_confidence: left_conf,
        right_confidence: right_conf,
        base_color,
        strip_width: strip_w,
        left_base_mask: left_mask,
        right_base_mask: right_mask,
    }
}

/// Reconcile a stitched working-image base estimate against the component-base
/// consensus when the stitched crop has likely lost the true outer rebate.
pub fn reconcile_stitched_base_estimate(
    working: &BaseDetection,
    component_bases: &[[f64; 3]],
) -> BaseReconciliation {
    let raw_working_confidence = working.left_confidence.max(working.right_confidence);
    let stronger_edge = raw_working_confidence.max(1e-6);
    let edge_balance_ratio = if stronger_edge <= 1e-6 {
        1.0
    } else {
        (working.left_confidence.min(working.right_confidence) / stronger_edge).clamp(0.0, 1.0)
    };

    let component_consensus = component_consensus_base_color(component_bases);
    if let Some((consensus, spread, consensus_confidence)) = component_consensus {
        let relative_delta = relative_rgb_delta(&working.base_color, &consensus);
        let mean_delta = relative_delta.iter().sum::<f64>() / relative_delta.len() as f64;
        let max_delta = relative_delta.iter().copied().fold(0.0f64, f64::max);
        let materially_darker_channels = (0..3)
            .filter(|&c| working.base_color[c] < consensus[c] * 0.92)
            .count();
        let should_override = edge_balance_ratio < 0.75
            && materially_darker_channels >= 2
            && (max_delta >= 0.08 || mean_delta >= 0.06);

        if should_override {
            return BaseReconciliation {
                base_color: consensus,
                confidence: consensus_confidence,
                source: "component_consensus",
                reason: format!(
                    "stitched crop-edge base was unbalanced (edge-balance {:.3}) and materially darker than the component consensus (max delta {:.1}%, mean delta {:.1}%); using the pre-stitch component consensus instead",
                    edge_balance_ratio,
                    max_delta * 100.0,
                    mean_delta * 100.0
                ),
                raw_working_base_color: working.base_color,
                raw_working_confidence,
                component_consensus_base_color: Some(consensus),
                component_consensus_confidence: Some(consensus_confidence),
                component_consensus_relative_spread: Some(spread),
                working_vs_consensus_relative_delta: Some(relative_delta),
                edge_balance_ratio,
            };
        }

        return BaseReconciliation {
            base_color: working.base_color,
            confidence: raw_working_confidence,
            source: "working_edges",
            reason: format!(
                "stitched working-image edge estimate stayed close to the component consensus (max delta {:.1}%, mean delta {:.1}%)",
                max_delta * 100.0,
                mean_delta * 100.0
            ),
            raw_working_base_color: working.base_color,
            raw_working_confidence,
            component_consensus_base_color: Some(consensus),
            component_consensus_confidence: Some(consensus_confidence),
            component_consensus_relative_spread: Some(spread),
            working_vs_consensus_relative_delta: Some(relative_delta),
            edge_balance_ratio,
        };
    }

    BaseReconciliation {
        base_color: working.base_color,
        confidence: raw_working_confidence,
        source: "working_edges",
        reason: "no stable component-base consensus was available; keeping the stitched working-image edge estimate".to_string(),
        raw_working_base_color: working.base_color,
        raw_working_confidence,
        component_consensus_base_color: None,
        component_consensus_confidence: None,
        component_consensus_relative_spread: None,
        working_vs_consensus_relative_delta: None,
        edge_balance_ratio,
    }
}

/// Scan the full image width for base-like vertical regions such as edge rebate
/// or internal frame gaps.
pub fn detect_vertical_base_regions(img: &Array3<u16>) -> VerticalBaseScan {
    let (h, w, _) = img.dim();
    if h == 0 || w == 0 {
        return VerticalBaseScan {
            block_width: 0,
            regions: Vec::new(),
            column_mask: Vec::new(),
            dominant_color: [0.0; 3],
        };
    }

    let block_width = (w as f64 * 0.02)
        .round()
        .max(5.0)
        .min((w / 8).max(5) as f64) as usize;

    let mut blocks = Vec::<BlockStats>::new();
    let mut spans = Vec::<(usize, usize)>::new();
    let mut x = 0usize;
    while x < w {
        let x_end = (x + block_width).min(w);
        spans.push((x, x_end));
        blocks.push(compute_block_stats(img, 0, h, x, x_end));
        x = x_end;
    }

    if blocks.len() < 2 {
        return VerticalBaseScan {
            block_width,
            regions: Vec::new(),
            column_mask: vec![0.0; w],
            dominant_color: [0.0; 3],
        };
    }

    let mut mad_vals: Vec<f64> = blocks.iter().map(|b| b.mad_lum).collect();
    let median_block_mad = fast_median(&mut mad_vals);
    let low_texture_threshold = (median_block_mad * 0.60).clamp(80.0, 600.0);
    let grow_texture_threshold = (low_texture_threshold * 1.2).clamp(100.0, 750.0);
    let seed_chroma_threshold = 0.020;
    let grow_chroma_threshold = 0.018;

    let mut region_strength = vec![0.0f64; blocks.len()];
    for i in 0..blocks.len() {
        if blocks[i].mad_lum > low_texture_threshold {
            continue;
        }

        let chroma_i = block_chroma(&blocks[i].median_rgb);
        let mut neighbor_contrast = 0.0f64;
        let mut neighbor_lum_contrast = 0.0f64;

        if i > 0 {
            let chroma_prev = block_chroma(&blocks[i - 1].median_rgb);
            neighbor_contrast = neighbor_contrast.max(chroma_distance(chroma_i, chroma_prev));
            neighbor_lum_contrast = neighbor_lum_contrast.max(relative_luminance_delta(
                &blocks[i].median_rgb,
                &blocks[i - 1].median_rgb,
            ));
        }
        if i + 1 < blocks.len() {
            let chroma_next = block_chroma(&blocks[i + 1].median_rgb);
            neighbor_contrast = neighbor_contrast.max(chroma_distance(chroma_i, chroma_next));
            neighbor_lum_contrast = neighbor_lum_contrast.max(relative_luminance_delta(
                &blocks[i].median_rgb,
                &blocks[i + 1].median_rgb,
            ));
        }

        let seed_conf = ((neighbor_contrast - 0.010) / 0.060).clamp(0.0, 1.0) * 0.7
            + ((neighbor_lum_contrast - 0.02) / 0.25).clamp(0.0, 1.0) * 0.3;

        if seed_conf <= 0.0 || neighbor_contrast < seed_chroma_threshold {
            continue;
        }

        let mut left = i;
        while left > 0 && blocks[left - 1].mad_lum <= grow_texture_threshold {
            let candidate = block_chroma(&blocks[left - 1].median_rgb);
            if chroma_distance(candidate, chroma_i) > grow_chroma_threshold {
                break;
            }
            left -= 1;
        }

        let mut right = i;
        while right + 1 < blocks.len() && blocks[right + 1].mad_lum <= grow_texture_threshold {
            let candidate = block_chroma(&blocks[right + 1].median_rgb);
            if chroma_distance(candidate, chroma_i) > grow_chroma_threshold {
                break;
            }
            right += 1;
        }

        for strength in &mut region_strength[left..=right] {
            *strength = strength.max(seed_conf);
        }
    }

    let mut regions = Vec::<BaseRegion>::new();
    let mut i = 0usize;
    while i < blocks.len() {
        if region_strength[i] <= 0.0 {
            i += 1;
            continue;
        }

        let start_idx = i;
        let mut end_idx = i;
        while end_idx + 1 < blocks.len() && region_strength[end_idx + 1] > 0.0 {
            end_idx += 1;
        }

        let x_start = spans[start_idx].0;
        let x_end = spans[end_idx].1;
        let width_fraction = (x_end - x_start) as f64 / w as f64;
        if width_fraction < 0.01 || width_fraction > 0.85 {
            i = end_idx + 1;
            continue;
        }

        let members: Vec<(usize, usize)> = (start_idx..=end_idx).map(|idx| (0, idx)).collect();
        let mut region_conf = 0.0f64;
        for idx in start_idx..=end_idx {
            region_conf += region_strength[idx];
        }
        region_conf /= (end_idx - start_idx + 1) as f64;

        let contrast = region_neighbor_contrast(&blocks, start_idx, end_idx);
        region_conf = (region_conf * 0.7 + contrast * 0.3).clamp(0.0, 1.0);
        if region_conf < 0.2 {
            i = end_idx + 1;
            continue;
        }

        let median_rgb = trimmed_mean_color_line(&blocks, start_idx, end_idx, 0.10);
        let mad_lum = blocks[start_idx..=end_idx]
            .iter()
            .map(|b| b.mad_lum)
            .sum::<f64>()
            / (end_idx - start_idx + 1) as f64;

        // `trimmed_mean_color` expects grid coordinates; feed a single-row view.
        let _ = members;
        regions.push(BaseRegion {
            x_start,
            x_end,
            confidence: region_conf,
            median_rgb,
            mad_lum,
        });

        i = end_idx + 1;
    }

    let mut column_mask = vec![0.0f64; w];
    for region in &regions {
        for v in &mut column_mask[region.x_start..region.x_end] {
            *v = (*v).max(region.confidence);
        }
    }

    let dominant_color = regions
        .iter()
        .max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.median_rgb)
        .unwrap_or([0.0; 3]);

    VerticalBaseScan {
        block_width,
        regions,
        column_mask,
        dominant_color,
    }
}

/// Compute the median RGB of a vertical region spanning full height.
fn region_median_color(img: &Array3<u16>, h: usize, x_start: usize, x_end: usize) -> [f64; 3] {
    let n = h * (x_end - x_start);
    let mut channels: [Vec<f64>; 3] = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
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
            let bx1 = if c_idx == grid_cols - 1 {
                x_end
            } else {
                bx0 + block_w
            };
            row_blocks.push(compute_block_stats(img, y0, y1, bx0, bx1));
        }
        blocks.push(row_blocks);
    }

    let total_blocks = grid_rows * grid_cols;
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
        return (0.0, vec![vec![0.0; grid_cols]; grid_rows], [0.0; 3]);
    }

    let chromas: Vec<[f64; 2]> = low_texture
        .iter()
        .map(|&(r, c_idx)| block_chroma(&blocks[r][c_idx].median_rgb))
        .collect();

    let mut c0s: Vec<f64> = chromas.iter().map(|c| c[0]).collect();
    let mut c1s: Vec<f64> = chromas.iter().map(|c| c[1]).collect();
    let center = [fast_median(&mut c0s), fast_median(&mut c1s)];

    let chroma_threshold = 0.05;
    let mut base_blocks: Vec<(usize, usize)> = Vec::new();
    for (i, &(r, c_idx)) in low_texture.iter().enumerate() {
        let dist = chroma_distance(chromas[i], center);
        if dist < chroma_threshold {
            base_blocks.push((r, c_idx));
        }
    }

    if base_blocks.is_empty() {
        return (0.0, vec![vec![0.0; grid_cols]; grid_rows], [0.0; 3]);
    }

    let base_fraction = base_blocks.len() as f64 / total_blocks as f64;
    let mut confidence = (base_fraction / 0.4).min(1.0);
    let strip_base_color = trimmed_mean_color(&blocks, &base_blocks, 0.10);

    let strip_chroma = block_chroma(&strip_base_color);
    let interior_chroma = block_chroma(interior_median);
    let chroma_dist = chroma_distance(strip_chroma, interior_chroma);

    let strip_lum = luminance(&strip_base_color);
    let interior_lum = luminance(interior_median);
    let lum_ratio = if strip_lum.max(interior_lum) > 0.0 {
        (strip_lum - interior_lum).abs() / strip_lum.max(interior_lum)
    } else {
        0.0
    };

    if chroma_dist < 0.03 && lum_ratio < 0.15 {
        confidence *= 0.1;
    } else if chroma_dist < 0.05 && lum_ratio < 0.25 {
        confidence *= 0.3;
    }

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

fn chroma_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn luminance(rgb: &[f64; 3]) -> f64 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}

fn relative_luminance_delta(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let la = luminance(a);
    let lb = luminance(b);
    if la.max(lb) <= 1.0 {
        0.0
    } else {
        (la - lb).abs() / la.max(lb)
    }
}

fn relative_rgb_delta(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|c| {
        let denom = a[c].max(b[c]).max(1.0);
        (a[c] - b[c]).abs() / denom
    })
}

fn component_consensus_base_color(
    component_bases: &[[f64; 3]],
) -> Option<([f64; 3], [f64; 3], f64)> {
    let valid: Vec<[f64; 3]> = component_bases
        .iter()
        .copied()
        .filter(|rgb| rgb.iter().all(|v| v.is_finite() && *v > 0.0))
        .collect();
    if valid.len() < 2 {
        return None;
    }

    let consensus =
        std::array::from_fn(|c| valid.iter().map(|rgb| rgb[c]).sum::<f64>() / valid.len() as f64);
    let spread = std::array::from_fn(|c| {
        let min_v = valid.iter().map(|rgb| rgb[c]).fold(f64::INFINITY, f64::min);
        let max_v = valid
            .iter()
            .map(|rgb| rgb[c])
            .fold(f64::NEG_INFINITY, f64::max);
        if !min_v.is_finite() || !max_v.is_finite() {
            1.0
        } else {
            (max_v - min_v) / max_v.max(1.0)
        }
    });
    let max_spread = spread.iter().copied().fold(0.0f64, f64::max);
    if max_spread > 0.08 {
        return None;
    }

    let confidence = if max_spread <= 0.02 {
        1.0
    } else {
        (1.0 - ((max_spread - 0.02) / 0.06) * 0.35).clamp(0.65, 1.0)
    };

    Some((consensus, spread, confidence))
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
            lum_vals.push(luminance(&[r, g, b]));
        }
    }

    let med_r = fast_median(&mut r_vals);
    let med_g = fast_median(&mut g_vals);
    let med_b = fast_median(&mut b_vals);
    let med_lum = fast_median(&mut lum_vals);

    let mut abs_devs: Vec<f64> = lum_vals.iter().map(|&v| (v - med_lum).abs()).collect();
    let mad = fast_median(&mut abs_devs);

    BlockStats {
        median_rgb: [med_r, med_g, med_b],
        mad_lum: mad,
    }
}

fn region_neighbor_contrast(blocks: &[BlockStats], start_idx: usize, end_idx: usize) -> f64 {
    let mut contrast = 0.0f64;
    let region_color = trimmed_mean_color_line(blocks, start_idx, end_idx, 0.10);
    let region_chroma = block_chroma(&region_color);

    if start_idx > 0 {
        contrast = contrast.max(chroma_distance(
            region_chroma,
            block_chroma(&blocks[start_idx - 1].median_rgb),
        ));
    }
    if end_idx + 1 < blocks.len() {
        contrast = contrast.max(chroma_distance(
            region_chroma,
            block_chroma(&blocks[end_idx + 1].median_rgb),
        ));
    }

    ((contrast - 0.01) / 0.05).clamp(0.0, 1.0)
}

fn trimmed_mean_color_line(
    blocks: &[BlockStats],
    start_idx: usize,
    end_idx: usize,
    trim_frac: f64,
) -> [f64; 3] {
    if start_idx > end_idx || end_idx >= blocks.len() {
        return [0.0; 3];
    }

    let mut result = [0.0; 3];
    for c in 0..3 {
        let mut vals: Vec<f64> = blocks[start_idx..=end_idx]
            .iter()
            .map(|b| b.median_rgb[c])
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        let trim_count = (n as f64 * trim_frac).floor() as usize;
        let start = trim_count.min(n - 1);
        let end = n.saturating_sub(trim_count).max(start + 1);
        let sum: f64 = vals[start..end].iter().sum();
        result[c] = sum / (end - start) as f64;
    }
    result
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
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        let trim_count = (n as f64 * trim_frac).floor() as usize;
        let start = trim_count.min(n - 1);
        let end = n.saturating_sub(trim_count).max(start + 1);
        let sum: f64 = vals[start..end].iter().sum();
        result[c] = sum / (end - start) as f64;
    }
    result
}

/// In-place median via partial sort.
fn fast_median(vals: &mut Vec<f64>) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    }
}

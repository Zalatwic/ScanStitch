use ndarray::{Array2, Array3, s};
use crate::report::PhaseReport;

/// Configuration for the stitching algorithm.
#[derive(Debug, Clone)]
pub struct StitchConfig {
    pub max_overlap: usize,
    pub min_ncc_score: f64,
    pub min_overlap: usize,
    pub blend_width: usize,
}

impl Default for StitchConfig {
    fn default() -> Self {
        Self {
            max_overlap: 300,
            min_ncc_score: 0.6,
            min_overlap: 20,
            blend_width: 50,
        }
    }
}

/// Result of a stitch operation.
pub struct StitchResult {
    pub result: Option<Array3<u16>>,
    pub x_offset: i32,
    pub ncc_score: f64,
    pub order: StitchOrder,
    pub report: PhaseReport,
}

/// Detected stitch ordering.
#[derive(Debug, Clone, PartialEq)]
pub enum StitchOrder {
    LeftRight,
    RightLeft,
    NoStitch,
}

/// Convert an RGB u16 image to grayscale f64 using luminance weights.
fn to_grayscale(img: &Array3<u16>) -> Array2<f64> {
    let (h, w, _) = img.dim();
    let mut gray = Array2::<f64>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let r = img[[y, x, 0]] as f64;
            let g = img[[y, x, 1]] as f64;
            let b = img[[y, x, 2]] as f64;
            gray[[y, x]] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }
    }
    gray
}

/// Compute normalized cross-correlation between two 2D arrays of the same shape.
fn ncc(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let n = a.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean_a = a.sum() / n;
    let mean_b = b.sum() / n;

    let mut sum_ab = 0.0;
    let mut sum_aa = 0.0;
    let mut sum_bb = 0.0;

    for (va, vb) in a.iter().zip(b.iter()) {
        let da = va - mean_a;
        let db = vb - mean_b;
        sum_ab += da * db;
        sum_aa += da * da;
        sum_bb += db * db;
    }

    let denom = (sum_aa * sum_bb).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }
    sum_ab / denom
}

/// Compute a combined overlap score: NCC weighted by value similarity.
///
/// Pure NCC is invariant to linear transformations, so two strips with the same
/// shape but different offsets score identically. We penalize the score by the
/// normalized RMS difference so that only strips whose actual values match get
/// a high combined score.
fn overlap_score(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let corr = ncc(a, b);
    if corr <= 0.0 {
        return 0.0;
    }

    // Compute normalized RMS difference (relative to the data range)
    let n = a.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let mut sum_sq_diff = 0.0;
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for (va, vb) in a.iter().zip(b.iter()) {
        let diff = va - vb;
        sum_sq_diff += diff * diff;
        min_val = min_val.min(*va).min(*vb);
        max_val = max_val.max(*va).max(*vb);
    }

    let range = max_val - min_val;
    if range < 1e-10 {
        // Constant images — no meaningful overlap
        return 0.0;
    }

    let nrmse = (sum_sq_diff / n).sqrt() / range;
    // Combine: high NCC + low NRMSE = good match
    // A perfect match has nrmse=0 -> score=corr
    // A poor value match has nrmse~1 -> score~0
    let similarity = (1.0 - nrmse.min(1.0)).max(0.0);
    corr * similarity
}

/// Find the best overlap between the right edge of `left` and the left edge of `right`
/// using normalized cross-correlation combined with value similarity.
///
/// Returns `Some((x_offset, score))` where `x_offset` is the column in `left` coordinates
/// where `right` starts, or `None` if no good match is found.
pub fn find_overlap_ncc(
    left: &Array3<u16>,
    right: &Array3<u16>,
    max_overlap: usize,
) -> Option<(usize, f64)> {
    let (h_l, w_l, _) = left.dim();
    let (h_r, w_r, _) = right.dim();
    let h = h_l.min(h_r);

    let gray_l = to_grayscale(left);
    let gray_r = to_grayscale(right);

    let max_ovl = max_overlap.min(w_l).min(w_r);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_overlap = 0usize;

    for ovl in 10..=max_ovl {
        let strip_l = gray_l.slice(s![0..h, (w_l - ovl)..w_l]);
        let strip_r = gray_r.slice(s![0..h, 0..ovl]);

        let score = overlap_score(&strip_l.to_owned(), &strip_r.to_owned());

        if score > best_score {
            best_score = score;
            best_overlap = ovl;
        }
    }

    if best_score > 0.3 {
        let x_offset = w_l - best_overlap;
        Some((x_offset, best_score))
    } else {
        None
    }
}

/// Score a left-right hypothesis. Returns the NCC score or 0.0 if no overlap found.
pub fn score_hypothesis(left: &Array3<u16>, right: &Array3<u16>, max_overlap: usize) -> f64 {
    match find_overlap_ncc(left, right, max_overlap) {
        Some((_, score)) => score,
        None => 0.0,
    }
}

/// Stitch two image components together, automatically detecting the correct order.
pub fn stitch_components(
    comp1: &Array3<u16>,
    comp2: &Array3<u16>,
    config: &StitchConfig,
) -> StitchResult {
    // Test both orderings
    let result_12 = find_overlap_ncc(comp1, comp2, config.max_overlap);
    let result_21 = find_overlap_ncc(comp2, comp1, config.max_overlap);

    let score_12 = result_12.map_or(0.0, |(_, s)| s);
    let score_21 = result_21.map_or(0.0, |(_, s)| s);

    let (left, right, overlap_result, best_score, order) = if score_12 >= score_21 {
        (comp1, comp2, result_12, score_12, StitchOrder::LeftRight)
    } else {
        (comp2, comp1, result_21, score_21, StitchOrder::RightLeft)
    };

    // Check minimum score
    if best_score < config.min_ncc_score {
        return StitchResult {
            result: None,
            x_offset: 0,
            ncc_score: best_score,
            order: StitchOrder::NoStitch,
            report: PhaseReport::fail("stitch", &format!("NCC score {:.3} below threshold", best_score)),
        };
    }

    let (x_offset, _) = match overlap_result {
        Some(r) => r,
        None => {
            return StitchResult {
                result: None,
                x_offset: 0,
                ncc_score: best_score,
                order: StitchOrder::NoStitch,
                report: PhaseReport::fail("stitch", "no overlap found"),
            };
        }
    };

    let (h_l, w_l, _) = left.dim();
    let (h_r, w_r, _) = right.dim();
    let overlap = w_l - x_offset;

    if overlap < config.min_overlap {
        return StitchResult {
            result: None,
            x_offset: x_offset as i32,
            ncc_score: best_score,
            order: StitchOrder::NoStitch,
            report: PhaseReport::fail("stitch", &format!("overlap {} below minimum {}", overlap, config.min_overlap)),
        };
    }

    let h = h_l.min(h_r);
    let total_w = x_offset + w_r;
    let mut stitched = Array3::<u16>::zeros((h, total_w, 3));

    // Region 1: left-only [0..x_offset]
    stitched
        .slice_mut(s![0..h, 0..x_offset, ..])
        .assign(&left.slice(s![0..h, 0..x_offset, ..]));

    // Region 2: overlap [x_offset..w_l] with feathered blend
    let blend_w = config.blend_width.min(overlap);
    for y in 0..h {
        for x in 0..overlap {
            let abs_x = x_offset + x;
            let alpha = if blend_w == 0 {
                1.0
            } else {
                (x as f64 / blend_w as f64).min(1.0)
            };
            for c in 0..3 {
                let l_val = left[[y, abs_x, c]] as f64;
                let r_val = right[[y, x, c]] as f64;
                let blended = l_val * (1.0 - alpha) + r_val * alpha;
                stitched[[y, abs_x, c]] = blended.round().clamp(0.0, u16::MAX as f64) as u16;
            }
        }
    }

    // Region 3: right-only [w_l..total_w]
    if w_r > overlap {
        stitched
            .slice_mut(s![0..h, w_l..total_w, ..])
            .assign(&right.slice(s![0..h, overlap..w_r, ..]));
    }

    let report = PhaseReport::ok(
        "stitch",
        best_score,
        serde_json::json!({
            "x_offset": x_offset,
            "overlap": overlap,
            "ncc_score": best_score,
            "total_width": total_w,
            "order": format!("{:?}", order),
        }),
    );

    StitchResult {
        result: Some(stitched),
        x_offset: x_offset as i32,
        ncc_score: best_score,
        order,
        report,
    }
}

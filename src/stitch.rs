use ndarray::{Array2, Array3, s};
use crate::report::PhaseReport;

/// Configuration for the stitching algorithm.
#[derive(Debug, Clone)]
pub struct StitchConfig {
    pub max_overlap: usize,
    pub min_ncc_score: f64,
    pub min_overlap: usize,
    pub blend_width: usize,
    /// Maximum vertical drift in pixels to search (searches -max_y_offset..=+max_y_offset).
    pub max_y_offset: i32,
}

impl Default for StitchConfig {
    fn default() -> Self {
        Self {
            max_overlap: 300,
            min_ncc_score: 0.6,
            min_overlap: 20,
            blend_width: 50,
            max_y_offset: 15,
        }
    }
}

/// Result of a stitch operation.
pub struct StitchResult {
    pub result: Option<Array3<u16>>,
    pub x_offset: i32,
    pub y_offset: i32,
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
        return 0.0;
    }

    let nrmse = (sum_sq_diff / n).sqrt() / range;
    let similarity = (1.0 - nrmse.min(1.0)).max(0.0);
    corr * similarity
}

/// Find the best overlap between the right edge of `left` and the left edge of `right`
/// using normalized cross-correlation combined with value similarity.
///
/// Searches a 2D window: horizontal overlaps from 10..=max_overlap and vertical offsets
/// from -max_y_offset..=+max_y_offset to account for mechanical stepper drift.
///
/// Returns `Some((x_offset, y_offset, score))` where `x_offset` is the column in `left`
/// coordinates where `right` starts, `y_offset` is the vertical shift applied to `right`
/// (positive = right image shifted down), or `None` if no good match is found.
pub fn find_overlap_ncc(
    left: &Array3<u16>,
    right: &Array3<u16>,
    max_overlap: usize,
    max_y_offset: i32,
) -> Option<(usize, i32, f64)> {
    let (h_l, w_l, _) = left.dim();
    let (h_r, w_r, _) = right.dim();
    let h = h_l.min(h_r);

    let gray_l = to_grayscale(left);
    let gray_r = to_grayscale(right);

    let max_ovl = max_overlap.min(w_l).min(w_r);
    let max_dy = max_y_offset.unsigned_abs() as usize;

    // Need enough height to produce a valid comparison region
    if h <= max_dy {
        return None;
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut best_overlap = 0usize;
    let mut best_dy: i32 = 0;

    for ovl in 10..=max_ovl {
        for dy in -max_y_offset..=max_y_offset {
            let abs_dy = dy.unsigned_abs() as usize;
            let cmp_h = h - abs_dy;
            if cmp_h < 10 {
                continue;
            }

            // When dy > 0, right is shifted down relative to left:
            //   left rows  [dy..dy+cmp_h]  align with  right rows [0..cmp_h]
            // When dy < 0, right is shifted up relative to left:
            //   left rows  [0..cmp_h]      align with  right rows [|dy|..cmp_h+|dy|]
            let (l_y_start, r_y_start) = if dy >= 0 {
                (dy as usize, 0usize)
            } else {
                (0usize, abs_dy)
            };

            let strip_l = gray_l.slice(s![
                l_y_start..(l_y_start + cmp_h),
                (w_l - ovl)..w_l
            ]);
            let strip_r = gray_r.slice(s![
                r_y_start..(r_y_start + cmp_h),
                0..ovl
            ]);

            let score = overlap_score(&strip_l.to_owned(), &strip_r.to_owned());

            // Prefer smaller |dy| when scores are effectively tied (within f64 epsilon).
            // This avoids spurious drift detection in regions with no vertical variation.
            let dominated = score > best_score
                || (score == best_score && dy.unsigned_abs() < best_dy.unsigned_abs());
            if dominated {
                best_score = score;
                best_overlap = ovl;
                best_dy = dy;
            }
        }
    }

    if best_score > 0.3 {
        let x_offset = w_l - best_overlap;
        Some((x_offset, best_dy, best_score))
    } else {
        None
    }
}

/// Score a left-right hypothesis. Returns the NCC score or 0.0 if no overlap found.
pub fn score_hypothesis(
    left: &Array3<u16>,
    right: &Array3<u16>,
    max_overlap: usize,
    max_y_offset: i32,
) -> f64 {
    match find_overlap_ncc(left, right, max_overlap, max_y_offset) {
        Some((_, _, score)) => score,
        None => 0.0,
    }
}

/// Stitch two image components together, automatically detecting the correct order.
///
/// Accounts for vertical mechanical drift by searching a 2D window and placing the
/// right component at the detected y_offset on an expanded canvas.
pub fn stitch_components(
    comp1: &Array3<u16>,
    comp2: &Array3<u16>,
    config: &StitchConfig,
) -> StitchResult {
    // Test both orderings
    let result_12 = find_overlap_ncc(comp1, comp2, config.max_overlap, config.max_y_offset);
    let result_21 = find_overlap_ncc(comp2, comp1, config.max_overlap, config.max_y_offset);

    let score_12 = result_12.map_or(0.0, |(_, _, s)| s);
    let score_21 = result_21.map_or(0.0, |(_, _, s)| s);

    let (left, right, overlap_result, best_score, order) = if score_12 >= score_21 {
        (comp1, comp2, result_12, score_12, StitchOrder::LeftRight)
    } else {
        (comp2, comp1, result_21, score_21, StitchOrder::RightLeft)
    };

    if best_score < config.min_ncc_score {
        return StitchResult {
            result: None,
            x_offset: 0,
            y_offset: 0,
            ncc_score: best_score,
            order: StitchOrder::NoStitch,
            report: PhaseReport::fail("stitch", &format!("NCC score {:.3} below threshold", best_score)),
        };
    }

    let (x_offset, y_offset, _) = match overlap_result {
        Some(r) => r,
        None => {
            return StitchResult {
                result: None,
                x_offset: 0,
                y_offset: 0,
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
            y_offset,
            ncc_score: best_score,
            order: StitchOrder::NoStitch,
            report: PhaseReport::fail("stitch", &format!("overlap {} below minimum {}", overlap, config.min_overlap)),
        };
    }

    // Canvas dimensions accounting for vertical drift.
    // y_offset > 0: right is shifted down → right starts at row y_offset, left starts at row 0.
    // y_offset < 0: right is shifted up → left starts at row |y_offset|, right starts at row 0.
    let abs_dy = y_offset.unsigned_abs() as usize;
    let l_y0 = if y_offset >= 0 { 0 } else { abs_dy };
    let r_y0 = if y_offset >= 0 { abs_dy } else { 0 };
    let canvas_h = (l_y0 + h_l).max(r_y0 + h_r);
    let total_w = x_offset + w_r;

    // Initialize canvas to 0 (black for empty areas)
    let mut stitched = Array3::<u16>::zeros((canvas_h, total_w, 3));

    // Region 1: left-only columns [0..x_offset]
    stitched
        .slice_mut(s![l_y0..(l_y0 + h_l), 0..x_offset, ..])
        .assign(&left.slice(s![0..h_l, 0..x_offset, ..]));

    // Region 2: overlap columns [x_offset..w_l] with feathered blend
    // Only blend rows where both left and right have valid data.
    let blend_w = config.blend_width.min(overlap);
    let blend_top = l_y0.max(r_y0);
    let blend_bot = (l_y0 + h_l).min(r_y0 + h_r);

    if blend_bot > blend_top {
        for canvas_y in blend_top..blend_bot {
            let ly = canvas_y - l_y0;
            let ry = canvas_y - r_y0;
            for x in 0..overlap {
                let abs_x = x_offset + x;
                let alpha = if blend_w == 0 {
                    1.0
                } else {
                    (x as f64 / blend_w as f64).min(1.0)
                };
                for c in 0..3 {
                    let l_val = left[[ly, abs_x, c]] as f64;
                    let r_val = right[[ry, x, c]] as f64;
                    let blended = l_val * (1.0 - alpha) + r_val * alpha;
                    stitched[[canvas_y, abs_x, c]] = blended.round().clamp(0.0, u16::MAX as f64) as u16;
                }
            }
        }

        // Fill overlap rows that only have left data (above or below the blend zone)
        for canvas_y in l_y0..(l_y0 + h_l) {
            if canvas_y >= blend_top && canvas_y < blend_bot {
                continue;
            }
            let ly = canvas_y - l_y0;
            for x in 0..overlap {
                let abs_x = x_offset + x;
                for c in 0..3 {
                    stitched[[canvas_y, abs_x, c]] = left[[ly, abs_x, c]];
                }
            }
        }

        // Fill overlap rows that only have right data (above or below the blend zone)
        for canvas_y in r_y0..(r_y0 + h_r) {
            if canvas_y >= blend_top && canvas_y < blend_bot {
                continue;
            }
            let ry = canvas_y - r_y0;
            for x in 0..overlap {
                let abs_x = x_offset + x;
                for c in 0..3 {
                    stitched[[canvas_y, abs_x, c]] = right[[ry, x, c]];
                }
            }
        }
    }

    // Region 3: right-only columns [w_l..total_w]
    if w_r > overlap {
        stitched
            .slice_mut(s![r_y0..(r_y0 + h_r), w_l..total_w, ..])
            .assign(&right.slice(s![0..h_r, overlap..w_r, ..]));
    }

    let report = PhaseReport::ok(
        "stitch",
        best_score,
        serde_json::json!({
            "x_offset": x_offset,
            "y_offset": y_offset,
            "overlap": overlap,
            "ncc_score": best_score,
            "total_width": total_w,
            "canvas_height": canvas_h,
            "order": format!("{:?}", order),
        }),
    );

    StitchResult {
        result: Some(stitched),
        x_offset: x_offset as i32,
        y_offset,
        ncc_score: best_score,
        order,
        report,
    }
}

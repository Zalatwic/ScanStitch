use ndarray::{s, Array3};

#[derive(Debug, Clone, Copy)]
struct LineStats {
    median_lum: f64,
    mad_lum: f64,
}

#[derive(Debug, Clone, Copy)]
struct EdgeRunDiagnostics {
    crop_rows: usize,
    strong_rows: usize,
    peak_lum_gap: f64,
    peak_row_delta: f64,
    boundary_transition_delta: f64,
    confidence: f64,
    overwhelming_support_fallback_used: bool,
    unresolved_strong_candidate: bool,
}

const OVERWHELMING_EDGE_MIN_STRONG_LINES: usize = 8;
const OVERWHELMING_EDGE_MIN_LUMINANCE_GAP_MULTIPLE: f64 = 4.0;
const OVERWHELMING_EDGE_MIN_BOUNDARY_FRACTION: f64 = 0.35;
const UNRESOLVED_EDGE_MIN_STRONG_LINES: usize = 4;
const UNRESOLVED_EDGE_MIN_LUMINANCE_GAP_MULTIPLE: f64 = 3.0;
const UNRESOLVED_EDGE_MAX_RUN_SPAN_MULTIPLE: f64 = 0.75;

#[derive(Debug, Clone)]
pub struct BorderRemovalDiagnostics {
    pub evidence_evaluated: bool,
    pub vertical_crop_rejected: bool,
    pub horizontal_crop_rejected: bool,
    pub top_removed: usize,
    pub bottom_removed: usize,
    pub left_removed: usize,
    pub right_removed: usize,
    pub content_lum: f64,
    pub content_mad: f64,
    pub column_content_lum: f64,
    pub column_content_mad: f64,
    pub top_strong_rows: usize,
    pub bottom_strong_rows: usize,
    pub top_peak_lum_gap: f64,
    pub bottom_peak_lum_gap: f64,
    pub top_peak_row_delta: f64,
    pub bottom_peak_row_delta: f64,
    pub top_boundary_transition_delta: f64,
    pub bottom_boundary_transition_delta: f64,
    pub top_confidence: f64,
    pub bottom_confidence: f64,
    pub left_strong_columns: usize,
    pub right_strong_columns: usize,
    pub left_peak_lum_gap: f64,
    pub right_peak_lum_gap: f64,
    pub left_peak_column_delta: f64,
    pub right_peak_column_delta: f64,
    pub left_boundary_transition_delta: f64,
    pub right_boundary_transition_delta: f64,
    pub left_confidence: f64,
    pub right_confidence: f64,
    pub top_overwhelming_support_fallback_used: bool,
    pub bottom_overwhelming_support_fallback_used: bool,
    pub left_overwhelming_support_fallback_used: bool,
    pub right_overwhelming_support_fallback_used: bool,
    pub top_unresolved_strong_candidate: bool,
    pub bottom_unresolved_strong_candidate: bool,
    pub left_unresolved_strong_candidate: bool,
    pub right_unresolved_strong_candidate: bool,
    pub row_strong_lum_gap_threshold: f64,
    pub row_boundary_derivative_threshold: f64,
    pub column_strong_lum_gap_threshold: f64,
    pub column_boundary_derivative_threshold: f64,
    pub dead_zone_detected: bool,
    pub warnings: Vec<String>,
}

pub struct BorderRemovalResult {
    pub cropped: Array3<u16>,
    pub diagnostics: BorderRemovalDiagnostics,
}

/// Compute the median of a slice of f64 values. The input is modified (sorted).
fn median_f64(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    if n.is_multiple_of(2) {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    } else {
        vals[n / 2]
    }
}

/// Compute robust per-row statistics used for border detection.
fn compute_row_stats(img: &Array3<u16>) -> Vec<LineStats> {
    let (h, w, _) = img.dim();
    let mut stats = Vec::with_capacity(h);

    for y in 0..h {
        let mut lum_vals = Vec::<f64>::with_capacity(w);
        for x in 0..w {
            let r = img[[y, x, 0]] as f64;
            let g = img[[y, x, 1]] as f64;
            let b = img[[y, x, 2]] as f64;
            lum_vals.push((r + g + b) / 3.0);
        }

        let median_lum = median_f64(&mut lum_vals);
        let mut abs_dev: Vec<f64> = lum_vals
            .into_iter()
            .map(|v| (v - median_lum).abs())
            .collect();
        let mad_lum = median_f64(&mut abs_dev);

        stats.push(LineStats {
            median_lum,
            mad_lum,
        });
    }

    stats
}

/// Compute the same robust statistics along columns so scanner borders can be removed on all
/// four sides without rotating or resampling the source.
fn compute_column_stats(img: &Array3<u16>) -> Vec<LineStats> {
    let (h, w, _) = img.dim();
    let mut stats = Vec::with_capacity(w);

    for x in 0..w {
        let mut lum_vals = Vec::<f64>::with_capacity(h);
        for y in 0..h {
            let r = img[[y, x, 0]] as f64;
            let g = img[[y, x, 1]] as f64;
            let b = img[[y, x, 2]] as f64;
            lum_vals.push((r + g + b) / 3.0);
        }

        let median_lum = median_f64(&mut lum_vals);
        let mut abs_dev = lum_vals
            .into_iter()
            .map(|value| (value - median_lum).abs())
            .collect::<Vec<_>>();
        let mad_lum = median_f64(&mut abs_dev);
        stats.push(LineStats {
            median_lum,
            mad_lum,
        });
    }

    stats
}

/// Compute a robust median MAD over a slice of rows.
fn median_mad(stats: &[LineStats]) -> f64 {
    let mut vals: Vec<f64> = stats.iter().map(|s| s.mad_lum).collect();
    median_f64(&mut vals)
}

fn strong_lum_gap_threshold(content_lum: f64) -> f64 {
    (content_lum * 0.12).max(600.0)
}

fn boundary_derivative_threshold(content_lum: f64) -> f64 {
    (content_lum * 0.10).max(400.0)
}

/// Normalize the three independent signals that make an applied edge crop admissible. This is an
/// evidence-strength score, not an empirical probability of geometric correctness. The weakest
/// signal controls the result: four strong lines, 1.5x the minimum luminance separation, and a full
/// derivative-threshold boundary transition each saturate their respective term at one.
fn accepted_edge_confidence(
    strong_rows: usize,
    peak_lum_gap: f64,
    strong_lum_gap: f64,
    boundary_transition_delta: f64,
    derivative_threshold: f64,
) -> f64 {
    if strong_rows < 2
        || !peak_lum_gap.is_finite()
        || !strong_lum_gap.is_finite()
        || !boundary_transition_delta.is_finite()
        || !derivative_threshold.is_finite()
        || strong_lum_gap <= 0.0
        || derivative_threshold <= 0.0
    {
        return 0.0;
    }

    let run_support = (strong_rows as f64 / 4.0).clamp(0.0, 1.0);
    let luminance_separation = (peak_lum_gap / (strong_lum_gap * 1.5)).clamp(0.0, 1.0);
    let boundary_transition = (boundary_transition_delta / derivative_threshold).clamp(0.0, 1.0);
    run_support
        .min(luminance_separation)
        .min(boundary_transition)
}

/// Detect the contiguous dead-zone run on one edge using low-variance row
/// statistics plus hysteresis/run-length logic.
fn detect_edge_run(
    stats: &[LineStats],
    content_lum: f64,
    content_mad: f64,
    from_top: bool,
    safety_margin: usize,
) -> EdgeRunDiagnostics {
    let len = stats.len();
    if len < 4 {
        return EdgeRunDiagnostics {
            crop_rows: 0,
            strong_rows: 0,
            peak_lum_gap: 0.0,
            peak_row_delta: 0.0,
            boundary_transition_delta: 0.0,
            confidence: 0.0,
            overwhelming_support_fallback_used: false,
            unresolved_strong_candidate: false,
        };
    }

    let strong_mad_threshold = (content_mad * 0.35).clamp(40.0, 500.0);
    let weak_mad_threshold = (content_mad * 0.60).clamp(80.0, 900.0);
    let strong_lum_gap = strong_lum_gap_threshold(content_lum);
    let weak_lum_gap = strong_lum_gap * 0.55;
    let derivative_threshold = boundary_derivative_threshold(content_lum);
    let max_search = (len / 3).max(1);
    let curved_edge_max = ((len as f64 * 0.03).ceil() as usize).max(2);

    let mut last_candidate = None;
    let mut strong_rows = 0usize;
    let mut standard_strong_rows = 0usize;
    let mut standard_strong_min_lum = f64::INFINITY;
    let mut standard_strong_max_lum = f64::NEG_INFINITY;
    let mut weak_bridge = 0usize;
    let mut peak_lum_gap = 0.0f64;
    let mut peak_row_delta = 0.0f64;

    for step in 0..max_search {
        let idx = if from_top { step } else { len - 1 - step };
        let row = stats[idx];
        let lum_gap = (row.median_lum - content_lum).abs();
        peak_lum_gap = peak_lum_gap.max(lum_gap);
        let standard_strong = row.mad_lum <= strong_mad_threshold && lum_gap >= strong_lum_gap;
        let curved_edge_strong = step < curved_edge_max
            && lum_gap >= strong_lum_gap * 1.45
            && row.mad_lum <= (content_mad * 1.8).max(800.0);
        let standard_weak = row.mad_lum <= weak_mad_threshold && lum_gap >= weak_lum_gap;
        let curved_edge_weak = step < curved_edge_max
            && lum_gap >= strong_lum_gap * 1.10
            && row.mad_lum <= (content_mad * 2.25).max(1200.0);
        // Scanner/film edges can be curved or flare-contaminated, which raises the line MAD
        // even when most of the line remains outside the photograph. A materially larger
        // luminance gap admits that case while retaining the boundary-derivative guard below.
        let strong = standard_strong || curved_edge_strong;
        let weak = standard_weak || curved_edge_weak;

        if strong {
            strong_rows += 1;
            if standard_strong {
                standard_strong_rows += 1;
                standard_strong_min_lum = standard_strong_min_lum.min(row.median_lum);
                standard_strong_max_lum = standard_strong_max_lum.max(row.median_lum);
            }
            weak_bridge = 0;
            last_candidate = Some(idx);
            continue;
        }

        if weak && strong_rows > 0 && weak_bridge < 2 {
            weak_bridge += 1;
            last_candidate = Some(idx);
            continue;
        }

        if strong_rows >= 2 {
            let next_delta = if from_top {
                if idx + 1 < len {
                    (stats[idx + 1].median_lum - row.median_lum).abs()
                } else {
                    0.0
                }
            } else if idx > 0 {
                (stats[idx - 1].median_lum - row.median_lum).abs()
            } else {
                0.0
            };
            peak_row_delta = peak_row_delta.max(next_delta);

            if row.mad_lum > weak_mad_threshold
                || lum_gap < weak_lum_gap
                || next_delta >= derivative_threshold
            {
                break;
            }
        } else if lum_gap < weak_lum_gap {
            break;
        }
    }

    if strong_rows < 2 {
        return EdgeRunDiagnostics {
            crop_rows: 0,
            strong_rows,
            peak_lum_gap,
            peak_row_delta,
            boundary_transition_delta: 0.0,
            confidence: 0.0,
            overwhelming_support_fallback_used: false,
            unresolved_strong_candidate: false,
        };
    }

    let Some(last_idx) = last_candidate else {
        return EdgeRunDiagnostics {
            crop_rows: 0,
            strong_rows,
            peak_lum_gap,
            peak_row_delta,
            boundary_transition_delta: 0.0,
            confidence: 0.0,
            overwhelming_support_fallback_used: false,
            unresolved_strong_candidate: false,
        };
    };

    let boundary_delta = if from_top {
        if last_idx + 1 < len {
            (stats[last_idx + 1].median_lum - stats[last_idx].median_lum).abs()
        } else {
            0.0
        }
    } else if last_idx > 0 {
        (stats[last_idx - 1].median_lum - stats[last_idx].median_lum).abs()
    } else {
        0.0
    };
    peak_row_delta = peak_row_delta.max(boundary_delta);

    // A real scanner gate may leave a narrow flare/antialias ramp instead of a one-line step.
    // Confirm that case against the last confidently dead line over a bounded inward window.
    // The window scales with the scan dimension but remains too short to turn an ordinary
    // gradual scene falloff into a removable border.
    let transition_window = ((len as f64 * 0.012).ceil() as usize).clamp(2, 48);
    let boundary_reference = stats[last_idx].median_lum;
    let mut aggregate_boundary_delta = boundary_delta;
    let mut transition_extension = 0usize;
    for offset in 1..=transition_window {
        let inward_idx = if from_top {
            last_idx.saturating_add(offset)
        } else {
            match last_idx.checked_sub(offset) {
                Some(index) => index,
                None => break,
            }
        };
        if inward_idx >= len {
            break;
        }
        let delta = (stats[inward_idx].median_lum - boundary_reference).abs();
        aggregate_boundary_delta = aggregate_boundary_delta.max(delta);
        peak_row_delta = peak_row_delta.max(delta);
        let inward_lum_gap = (stats[inward_idx].median_lum - content_lum).abs();
        let reached_content = delta >= derivative_threshold * 0.5
            && (inward_lum_gap < weak_lum_gap || stats[inward_idx].mad_lum > weak_mad_threshold);
        if reached_content {
            transition_extension = offset.saturating_sub(1);
            break;
        }
    }

    let ordinary_boundary_supported = boundary_delta >= derivative_threshold * 0.25
        || aggregate_boundary_delta >= derivative_threshold * 0.5;
    let overwhelming_support_fallback_used = !ordinary_boundary_supported
        && strong_rows >= OVERWHELMING_EDGE_MIN_STRONG_LINES
        && peak_lum_gap >= strong_lum_gap * OVERWHELMING_EDGE_MIN_LUMINANCE_GAP_MULTIPLE
        && aggregate_boundary_delta
            >= derivative_threshold * OVERWHELMING_EDGE_MIN_BOUNDARY_FRACTION;
    if !ordinary_boundary_supported && !overwhelming_support_fallback_used {
        // Curved-edge admission deliberately tolerates locally textured lines near the outermost
        // edge, but that alone is not enough to block delivery. Requiring ordinary low-variance
        // support prevents a flat or high-contrast subject region from becoming an unresolved
        // scanner-border claim solely because the content reference is bimodal.
        let standard_strong_luminance_span =
            (standard_strong_max_lum - standard_strong_min_lum).max(0.0);
        let unresolved_strong_candidate = standard_strong_rows >= UNRESOLVED_EDGE_MIN_STRONG_LINES
            && peak_lum_gap >= strong_lum_gap * UNRESOLVED_EDGE_MIN_LUMINANCE_GAP_MULTIPLE
            && standard_strong_luminance_span
                <= strong_lum_gap * UNRESOLVED_EDGE_MAX_RUN_SPAN_MULTIPLE;
        return EdgeRunDiagnostics {
            crop_rows: 0,
            strong_rows,
            peak_lum_gap,
            peak_row_delta,
            boundary_transition_delta: aggregate_boundary_delta,
            confidence: 0.0,
            overwhelming_support_fallback_used: false,
            unresolved_strong_candidate,
        };
    }

    let mut crop = if from_top {
        last_idx + 1
    } else {
        len - last_idx
    }
    .saturating_add(transition_extension)
    .min(len);

    // Safety margin inward, but stop if variance rises sharply into real content.
    for _ in 0..safety_margin {
        let next_idx = if from_top {
            crop
        } else {
            match len.checked_sub(crop + 1) {
                Some(idx) => idx,
                None => break,
            }
        };

        if next_idx >= len {
            break;
        }

        let next = stats[next_idx];
        let lum_gap = (next.median_lum - content_lum).abs();
        let next_delta = if from_top {
            if next_idx > 0 {
                (next.median_lum - stats[next_idx - 1].median_lum).abs()
            } else {
                0.0
            }
        } else if next_idx + 1 < len {
            (stats[next_idx + 1].median_lum - next.median_lum).abs()
        } else {
            0.0
        };
        peak_row_delta = peak_row_delta.max(next_delta);

        let sharp_rise = next.mad_lum > content_mad.max(weak_mad_threshold * 0.8)
            && lum_gap < strong_lum_gap
            && next_delta < derivative_threshold * 0.5;

        if sharp_rise {
            break;
        }

        crop += 1;
    }

    EdgeRunDiagnostics {
        crop_rows: crop,
        strong_rows,
        peak_lum_gap,
        peak_row_delta,
        boundary_transition_delta: aggregate_boundary_delta,
        confidence: accepted_edge_confidence(
            strong_rows,
            peak_lum_gap,
            strong_lum_gap,
            aggregate_boundary_delta,
            derivative_threshold,
        ),
        overwhelming_support_fallback_used,
        unresolved_strong_candidate: false,
    }
}

/// Detect and remove scanner dead-zone borders from all four sides of a 16-bit RGB image.
/// Row and column evidence are evaluated independently, so the operation never rotates or
/// resamples source pixels.
pub fn remove_borders_with_diagnostics(
    img: &Array3<u16>,
    safety_margin: usize,
) -> BorderRemovalResult {
    let (h, w, _) = img.dim();
    if h < 4 || w < 4 {
        return BorderRemovalResult {
            cropped: img.clone(),
            diagnostics: BorderRemovalDiagnostics {
                evidence_evaluated: false,
                vertical_crop_rejected: false,
                horizontal_crop_rejected: false,
                top_removed: 0,
                bottom_removed: 0,
                left_removed: 0,
                right_removed: 0,
                content_lum: 0.0,
                content_mad: 0.0,
                column_content_lum: 0.0,
                column_content_mad: 0.0,
                top_strong_rows: 0,
                bottom_strong_rows: 0,
                top_peak_lum_gap: 0.0,
                bottom_peak_lum_gap: 0.0,
                top_peak_row_delta: 0.0,
                bottom_peak_row_delta: 0.0,
                top_boundary_transition_delta: 0.0,
                bottom_boundary_transition_delta: 0.0,
                top_confidence: 0.0,
                bottom_confidence: 0.0,
                left_strong_columns: 0,
                right_strong_columns: 0,
                left_peak_lum_gap: 0.0,
                right_peak_lum_gap: 0.0,
                left_peak_column_delta: 0.0,
                right_peak_column_delta: 0.0,
                left_boundary_transition_delta: 0.0,
                right_boundary_transition_delta: 0.0,
                left_confidence: 0.0,
                right_confidence: 0.0,
                top_overwhelming_support_fallback_used: false,
                bottom_overwhelming_support_fallback_used: false,
                left_overwhelming_support_fallback_used: false,
                right_overwhelming_support_fallback_used: false,
                top_unresolved_strong_candidate: false,
                bottom_unresolved_strong_candidate: false,
                left_unresolved_strong_candidate: false,
                right_unresolved_strong_candidate: false,
                row_strong_lum_gap_threshold: 0.0,
                row_boundary_derivative_threshold: 0.0,
                column_strong_lum_gap_threshold: 0.0,
                column_boundary_derivative_threshold: 0.0,
                dead_zone_detected: false,
                warnings: vec![
                    "image too small for four-edge border detection; leaving borders unchanged"
                        .to_string(),
                ],
            },
        };
    }

    let row_stats = compute_row_stats(img);
    let q1 = h / 4;
    let q3 = 3 * h / 4;
    let center = &row_stats[q1..q3.max(q1 + 1)];

    let mut center_lums: Vec<f64> = center.iter().map(|s| s.median_lum).collect();
    let content_lum = median_f64(&mut center_lums);
    let content_mad = median_mad(center);

    let top = detect_edge_run(&row_stats, content_lum, content_mad, true, safety_margin);
    let bottom = detect_edge_run(&row_stats, content_lum, content_mad, false, safety_margin);
    let column_stats = compute_column_stats(img);
    let column_q1 = w / 4;
    let column_q3 = 3 * w / 4;
    let column_center = &column_stats[column_q1..column_q3.max(column_q1 + 1)];
    let mut column_center_lums = column_center
        .iter()
        .map(|stats| stats.median_lum)
        .collect::<Vec<_>>();
    let column_content_lum = median_f64(&mut column_center_lums);
    let column_content_mad = median_mad(column_center);
    let left = detect_edge_run(
        &column_stats,
        column_content_lum,
        column_content_mad,
        true,
        safety_margin,
    );
    let right = detect_edge_run(
        &column_stats,
        column_content_lum,
        column_content_mad,
        false,
        safety_margin,
    );

    let mut warnings = Vec::<String>::new();
    for (edge, diagnostics) in [
        ("top", &top),
        ("bottom", &bottom),
        ("left", &left),
        ("right", &right),
    ] {
        if diagnostics.overwhelming_support_fallback_used {
            warnings.push(format!(
                "{edge} border crop used the conservative overwhelming-edge fallback because at least {} strong lines and a {:.1}x luminance gap supported the edge while its gradual boundary supplied only {:.0}% of the ordinary derivative threshold",
                OVERWHELMING_EDGE_MIN_STRONG_LINES,
                OVERWHELMING_EDGE_MIN_LUMINANCE_GAP_MULTIPLE,
                OVERWHELMING_EDGE_MIN_BOUNDARY_FRACTION * 100.0
            ));
        }
        if diagnostics.unresolved_strong_candidate {
            warnings.push(format!(
                "{edge} edge retained a strong dead-zone candidate but lacked even the bounded gradual-transition evidence required for automatic cropping; geometry review is required"
            ));
        }
    }
    let vertical_crop_rejected = top.crop_rows + bottom.crop_rows >= h;
    let (top_removed, bottom_removed) = if vertical_crop_rejected {
        warnings.push(format!(
            "vertical border crop rejected because it would consume the full image ({} top, {} bottom of {})",
            top.crop_rows, bottom.crop_rows, h
        ));
        (0, 0)
    } else {
        (top.crop_rows, bottom.crop_rows)
    };
    let horizontal_crop_rejected = left.crop_rows + right.crop_rows >= w;
    let (left_removed, right_removed) = if horizontal_crop_rejected {
        warnings.push(format!(
            "horizontal border crop rejected because it would consume the full image ({} left, {} right of {})",
            left.crop_rows, right.crop_rows, w
        ));
        (0, 0)
    } else {
        (left.crop_rows, right.crop_rows)
    };

    let dead_zone_detected = top_removed + bottom_removed + left_removed + right_removed > 0;
    if !dead_zone_detected {
        warnings.push(
            "no convincing scanner dead-zone rows or columns were detected; borders were left unchanged"
                .to_string(),
        );
        log::info!("Border detection: no dead-zone regions detected. No cropping.");
    }

    log::info!(
        "Border detection: removing {} top, {} bottom, {} left, {} right pixels from {}x{} image",
        top_removed,
        bottom_removed,
        left_removed,
        right_removed,
        h,
        w
    );
    log::debug!(
        "Border reference statistics: row content_lum={:.1}, row content_mad={:.1}, column content_lum={:.1}, column content_mad={:.1}",
        content_lum,
        content_mad,
        column_content_lum,
        column_content_mad
    );

    let end_row = h - bottom_removed;
    let end_column = w - right_removed;
    let cropped = img
        .slice(s![top_removed..end_row, left_removed..end_column, ..])
        .to_owned();
    BorderRemovalResult {
        cropped,
        diagnostics: BorderRemovalDiagnostics {
            evidence_evaluated: true,
            vertical_crop_rejected,
            horizontal_crop_rejected,
            top_removed,
            bottom_removed,
            left_removed,
            right_removed,
            content_lum,
            content_mad,
            column_content_lum,
            column_content_mad,
            top_strong_rows: top.strong_rows,
            bottom_strong_rows: bottom.strong_rows,
            top_peak_lum_gap: top.peak_lum_gap,
            bottom_peak_lum_gap: bottom.peak_lum_gap,
            top_peak_row_delta: top.peak_row_delta,
            bottom_peak_row_delta: bottom.peak_row_delta,
            top_boundary_transition_delta: top.boundary_transition_delta,
            bottom_boundary_transition_delta: bottom.boundary_transition_delta,
            top_confidence: if top_removed > 0 { top.confidence } else { 0.0 },
            bottom_confidence: if bottom_removed > 0 {
                bottom.confidence
            } else {
                0.0
            },
            left_strong_columns: left.strong_rows,
            right_strong_columns: right.strong_rows,
            left_peak_lum_gap: left.peak_lum_gap,
            right_peak_lum_gap: right.peak_lum_gap,
            left_peak_column_delta: left.peak_row_delta,
            right_peak_column_delta: right.peak_row_delta,
            left_boundary_transition_delta: left.boundary_transition_delta,
            right_boundary_transition_delta: right.boundary_transition_delta,
            left_confidence: if left_removed > 0 {
                left.confidence
            } else {
                0.0
            },
            right_confidence: if right_removed > 0 {
                right.confidence
            } else {
                0.0
            },
            top_overwhelming_support_fallback_used: top.overwhelming_support_fallback_used,
            bottom_overwhelming_support_fallback_used: bottom.overwhelming_support_fallback_used,
            left_overwhelming_support_fallback_used: left.overwhelming_support_fallback_used,
            right_overwhelming_support_fallback_used: right.overwhelming_support_fallback_used,
            top_unresolved_strong_candidate: top.unresolved_strong_candidate,
            bottom_unresolved_strong_candidate: bottom.unresolved_strong_candidate,
            left_unresolved_strong_candidate: left.unresolved_strong_candidate,
            right_unresolved_strong_candidate: right.unresolved_strong_candidate,
            row_strong_lum_gap_threshold: strong_lum_gap_threshold(content_lum),
            row_boundary_derivative_threshold: boundary_derivative_threshold(content_lum),
            column_strong_lum_gap_threshold: strong_lum_gap_threshold(column_content_lum),
            column_boundary_derivative_threshold: boundary_derivative_threshold(column_content_lum),
            dead_zone_detected,
            warnings,
        },
    }
}

pub fn remove_borders(img: &Array3<u16>, safety_margin: usize) -> (Array3<u16>, usize, usize) {
    let result = remove_borders_with_diagnostics(img, safety_margin);
    (
        result.cropped,
        result.diagnostics.top_removed,
        result.diagnostics.bottom_removed,
    )
}

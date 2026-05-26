use ndarray::{s, Array3};

#[derive(Debug, Clone, Copy)]
struct RowStats {
    median_lum: f64,
    mad_lum: f64,
}

#[derive(Debug, Clone, Copy)]
struct EdgeRunDiagnostics {
    crop_rows: usize,
    strong_rows: usize,
    peak_lum_gap: f64,
    peak_row_delta: f64,
}

#[derive(Debug, Clone)]
pub struct BorderRemovalDiagnostics {
    pub top_removed: usize,
    pub bottom_removed: usize,
    pub content_lum: f64,
    pub content_mad: f64,
    pub top_strong_rows: usize,
    pub bottom_strong_rows: usize,
    pub top_peak_lum_gap: f64,
    pub bottom_peak_lum_gap: f64,
    pub top_peak_row_delta: f64,
    pub bottom_peak_row_delta: f64,
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
fn compute_row_stats(img: &Array3<u16>) -> Vec<RowStats> {
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

        stats.push(RowStats {
            median_lum,
            mad_lum,
        });
    }

    stats
}

/// Compute a robust median MAD over a slice of rows.
fn median_mad(stats: &[RowStats]) -> f64 {
    let mut vals: Vec<f64> = stats.iter().map(|s| s.mad_lum).collect();
    median_f64(&mut vals)
}

/// Detect the contiguous dead-zone run on one edge using low-variance row
/// statistics plus hysteresis/run-length logic.
fn detect_edge_run(
    stats: &[RowStats],
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
        };
    }

    let strong_mad_threshold = (content_mad * 0.35).clamp(40.0, 500.0);
    let weak_mad_threshold = (content_mad * 0.60).clamp(80.0, 900.0);
    let strong_lum_gap = (content_lum * 0.12).max(600.0);
    let weak_lum_gap = strong_lum_gap * 0.55;
    let derivative_threshold = (content_lum * 0.10).max(400.0);
    let max_search = (len / 3).max(1);

    let mut last_candidate = None;
    let mut strong_rows = 0usize;
    let mut weak_bridge = 0usize;
    let mut peak_lum_gap = 0.0f64;
    let mut peak_row_delta = 0.0f64;

    for step in 0..max_search {
        let idx = if from_top { step } else { len - 1 - step };
        let row = stats[idx];
        let lum_gap = (row.median_lum - content_lum).abs();
        peak_lum_gap = peak_lum_gap.max(lum_gap);
        let strong = row.mad_lum <= strong_mad_threshold && lum_gap >= strong_lum_gap;
        let weak = row.mad_lum <= weak_mad_threshold && lum_gap >= weak_lum_gap;

        if strong {
            strong_rows += 1;
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
        };
    }

    let Some(last_idx) = last_candidate else {
        return EdgeRunDiagnostics {
            crop_rows: 0,
            strong_rows,
            peak_lum_gap,
            peak_row_delta,
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

    if boundary_delta < derivative_threshold * 0.25 {
        return EdgeRunDiagnostics {
            crop_rows: 0,
            strong_rows,
            peak_lum_gap,
            peak_row_delta,
        };
    }

    let mut crop = if from_top {
        last_idx + 1
    } else {
        len - last_idx
    };

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
    }
}

/// Detect and remove scanner dead-zone borders from a 16-bit RGB image using
/// robust row statistics and contiguous-region logic.
///
/// Returns `(cropped_image, top_rows_removed, bottom_rows_removed)`.
pub fn remove_borders_with_diagnostics(
    img: &Array3<u16>,
    safety_margin: usize,
) -> BorderRemovalResult {
    let (h, w, _) = img.dim();
    if h < 4 {
        return BorderRemovalResult {
            cropped: img.clone(),
            diagnostics: BorderRemovalDiagnostics {
                top_removed: 0,
                bottom_removed: 0,
                content_lum: 0.0,
                content_mad: 0.0,
                top_strong_rows: 0,
                bottom_strong_rows: 0,
                top_peak_lum_gap: 0.0,
                bottom_peak_lum_gap: 0.0,
                top_peak_row_delta: 0.0,
                bottom_peak_row_delta: 0.0,
                dead_zone_detected: false,
                warnings: vec![
                    "image too short for border detection; leaving borders unchanged".to_string(),
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
    let mut warnings = Vec::<String>::new();

    if top.crop_rows == 0 && bottom.crop_rows == 0 {
        warnings.push(
            "no convincing scanner dead-zone rows were detected; borders were left unchanged"
                .to_string(),
        );
        log::info!("Border detection: no dead-zone regions detected. No cropping.");
        return BorderRemovalResult {
            cropped: img.clone(),
            diagnostics: BorderRemovalDiagnostics {
                top_removed: 0,
                bottom_removed: 0,
                content_lum,
                content_mad,
                top_strong_rows: top.strong_rows,
                bottom_strong_rows: bottom.strong_rows,
                top_peak_lum_gap: top.peak_lum_gap,
                bottom_peak_lum_gap: bottom.peak_lum_gap,
                top_peak_row_delta: top.peak_row_delta,
                bottom_peak_row_delta: bottom.peak_row_delta,
                dead_zone_detected: false,
                warnings,
            },
        };
    }

    if top.crop_rows + bottom.crop_rows >= h {
        warnings.push(format!(
            "border crop rejected because it would consume the full image ({} top, {} bottom of {})",
            top.crop_rows, bottom.crop_rows, h
        ));
        log::warn!(
            "Border detection rejected crop because it would consume the full image ({} top, {} bottom of {}).",
            top.crop_rows,
            bottom.crop_rows,
            h
        );
        return BorderRemovalResult {
            cropped: img.clone(),
            diagnostics: BorderRemovalDiagnostics {
                top_removed: 0,
                bottom_removed: 0,
                content_lum,
                content_mad,
                top_strong_rows: top.strong_rows,
                bottom_strong_rows: bottom.strong_rows,
                top_peak_lum_gap: top.peak_lum_gap,
                bottom_peak_lum_gap: bottom.peak_lum_gap,
                top_peak_row_delta: top.peak_row_delta,
                bottom_peak_row_delta: bottom.peak_row_delta,
                dead_zone_detected: false,
                warnings,
            },
        };
    }

    log::info!(
        "Border detection: removing {} top rows, {} bottom rows (content_lum={:.1}, content_mad={:.1}) from {}x{} image",
        top.crop_rows,
        bottom.crop_rows,
        content_lum,
        content_mad,
        h,
        w
    );

    let end_row = h - bottom.crop_rows;
    let cropped = img.slice(s![top.crop_rows..end_row, .., ..]).to_owned();
    BorderRemovalResult {
        cropped,
        diagnostics: BorderRemovalDiagnostics {
            top_removed: top.crop_rows,
            bottom_removed: bottom.crop_rows,
            content_lum,
            content_mad,
            top_strong_rows: top.strong_rows,
            bottom_strong_rows: bottom.strong_rows,
            top_peak_lum_gap: top.peak_lum_gap,
            bottom_peak_lum_gap: bottom.peak_lum_gap,
            top_peak_row_delta: top.peak_row_delta,
            bottom_peak_row_delta: bottom.peak_row_delta,
            dead_zone_detected: true,
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

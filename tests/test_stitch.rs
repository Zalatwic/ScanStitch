mod common;
use common::synthetic;
use ndarray::{s, Array3};
use std::path::PathBuf;

fn make_split_pair(h: usize, total_w: usize, overlap: usize) -> (Array3<u16>, Array3<u16>) {
    let full = gradient_2d(h, total_w);
    let split_point = total_w / 2 + overlap / 2;
    let comp1 = full.slice(s![.., 0..split_point, ..]).to_owned();
    let comp2 = full
        .slice(s![.., (split_point - overlap)..total_w, ..])
        .to_owned();
    (comp1, comp2)
}

fn make_textured_split_pair(
    h: usize,
    total_w: usize,
    overlap: usize,
) -> (Array3<u16>, Array3<u16>) {
    let mut full = gradient_2d(h, total_w);
    synthetic::add_noise(&mut full, 0x5EED, 900);
    for y in 0..h {
        for x in 0..total_w {
            let stripe = ((x / 19) + (y / 23)) % 3;
            if stripe == 0 {
                full[[y, x, 0]] = full[[y, x, 0]].saturating_add(700);
            } else if stripe == 1 {
                full[[y, x, 1]] = full[[y, x, 1]].saturating_add(500);
            } else {
                full[[y, x, 2]] = full[[y, x, 2]].saturating_add(650);
            }
        }
    }

    let split_point = total_w / 2 + overlap / 2;
    let comp1 = full.slice(s![.., 0..split_point, ..]).to_owned();
    let comp2 = full
        .slice(s![.., (split_point - overlap)..total_w, ..])
        .to_owned();
    (comp1, comp2)
}

fn make_textured_three_way_sequence(
    h: usize,
    total_w: usize,
    component_w: usize,
    step: usize,
) -> Vec<Array3<u16>> {
    let mut full = gradient_2d(h, total_w);
    synthetic::add_noise(&mut full, 0xA11CE, 800);
    for y in 0..h {
        for x in 0..total_w {
            let feature = ((x / 17) ^ (y / 13) ^ ((x + y) / 29)) % 5;
            full[[y, x, feature % 3]] = full[[y, x, feature % 3]].saturating_add(900);
        }
    }
    [0, step, step * 2]
        .into_iter()
        .map(|start| {
            full.slice(s![.., start..(start + component_w), ..])
                .to_owned()
        })
        .collect()
}

fn scale_image_channels(img: &mut Array3<u16>, gain: [f64; 3]) {
    let (h, w, _) = img.dim();
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                img[[y, x, c]] = (img[[y, x, c]] as f64 * gain[c])
                    .round()
                    .clamp(0.0, 16383.0) as u16;
            }
        }
    }
}

fn offset_image_channels(img: &mut Array3<u16>, offset: [f64; 3]) {
    let (h, w, _) = img.dim();
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                img[[y, x, c]] = (img[[y, x, c]] as f64 + offset[c])
                    .round()
                    .clamp(0.0, 16383.0) as u16;
            }
        }
    }
}

fn box_blur_u16(img: &Array3<u16>, radius: usize) -> Array3<u16> {
    let (height, width, channels) = img.dim();
    let mut horizontal = Array3::<f64>::zeros((height, width, channels));
    for y in 0..height {
        for x in 0..width {
            let x_start = x.saturating_sub(radius);
            let x_end = (x + radius + 1).min(width);
            for channel in 0..channels {
                horizontal[[y, x, channel]] = (x_start..x_end)
                    .map(|sample_x| img[[y, sample_x, channel]] as f64)
                    .sum::<f64>()
                    / (x_end - x_start) as f64;
            }
        }
    }
    let mut output = Array3::<u16>::zeros((height, width, channels));
    for y in 0..height {
        let y_start = y.saturating_sub(radius);
        let y_end = (y + radius + 1).min(height);
        for x in 0..width {
            for channel in 0..channels {
                output[[y, x, channel]] = ((y_start..y_end)
                    .map(|sample_y| horizontal[[sample_y, x, channel]])
                    .sum::<f64>()
                    / (y_end - y_start) as f64)
                    .round()
                    .clamp(0.0, 16383.0) as u16;
            }
        }
    }
    output
}

fn shade_image_channels_vertically(img: &mut Array3<u16>, log_slopes: [f64; 3]) {
    let (h, w, _) = img.dim();
    for y in 0..h {
        let y_normalized = if h <= 1 {
            0.0
        } else {
            2.0 * y as f64 / (h - 1) as f64 - 1.0
        };
        for x in 0..w {
            for c in 0..3 {
                img[[y, x, c]] = (img[[y, x, c]] as f64 * (log_slopes[c] * y_normalized).exp())
                    .round()
                    .clamp(0.0, 16383.0) as u16;
            }
        }
    }
}

fn impose_vertical_affine_mismatch(
    img: &mut Array3<u16>,
    correction_center_gain: [f64; 3],
    correction_log_gain_slopes: [f64; 3],
    correction_center_offset: [f64; 3],
    correction_offset_slopes: [f64; 3],
) {
    let (h, w, _) = img.dim();
    for y in 0..h {
        let y_normalized = if h <= 1 {
            0.0
        } else {
            2.0 * y as f64 / (h - 1) as f64 - 1.0
        };
        for x in 0..w {
            for c in 0..3 {
                let correction_gain = correction_center_gain[c]
                    * (correction_log_gain_slopes[c] * y_normalized).exp();
                let correction_offset =
                    correction_center_offset[c] + correction_offset_slopes[c] * y_normalized;
                img[[y, x, c]] = ((img[[y, x, c]] as f64 - correction_offset) / correction_gain)
                    .round()
                    .clamp(0.0, 16383.0) as u16;
            }
        }
    }
}

struct AffineMismatch2d {
    center_gain: [f64; 3],
    log_gain_x_slopes: [f64; 3],
    log_gain_y_slopes: [f64; 3],
    center_offset: [f64; 3],
    offset_x_slopes: [f64; 3],
    offset_y_slopes: [f64; 3],
}

fn impose_2d_affine_mismatch(
    img: &mut Array3<u16>,
    overlap_width: usize,
    mismatch: &AffineMismatch2d,
) {
    let (h, w, _) = img.dim();
    for y in 0..h {
        let y_normalized = if h <= 1 {
            0.0
        } else {
            2.0 * y as f64 / (h - 1) as f64 - 1.0
        };
        for x in 0..w {
            let x_normalized = if overlap_width <= 1 {
                0.0
            } else {
                (2.0 * x as f64 / (overlap_width - 1) as f64 - 1.0).clamp(-1.0, 1.0)
            };
            for channel in 0..3 {
                let gain = mismatch.center_gain[channel]
                    * (mismatch.log_gain_x_slopes[channel] * x_normalized
                        + mismatch.log_gain_y_slopes[channel] * y_normalized)
                        .exp();
                let offset = mismatch.center_offset[channel]
                    + mismatch.offset_x_slopes[channel] * x_normalized
                    + mismatch.offset_y_slopes[channel] * y_normalized;
                img[[y, x, channel]] = ((img[[y, x, channel]] as f64 - offset) / gain)
                    .round()
                    .clamp(0.0, 16383.0) as u16;
            }
        }
    }
}

struct QuadraticGainMismatch2d {
    center_gain: [f64; 3],
    log_gain_x_slopes: [f64; 3],
    log_gain_y_slopes: [f64; 3],
    log_gain_xx: [f64; 3],
    log_gain_xy: [f64; 3],
    log_gain_yy: [f64; 3],
}

fn impose_quadratic_2d_gain_mismatch(
    img: &mut Array3<u16>,
    overlap_width: usize,
    mismatch: &QuadraticGainMismatch2d,
) {
    let (h, w, _) = img.dim();
    for y in 0..h {
        let y_normalized = if h <= 1 {
            0.0
        } else {
            2.0 * y as f64 / (h - 1) as f64 - 1.0
        };
        for x in 0..w {
            let x_normalized = if overlap_width <= 1 {
                0.0
            } else {
                (2.0 * x as f64 / (overlap_width - 1) as f64 - 1.0).clamp(-1.0, 1.0)
            };
            for channel in 0..3 {
                let log_gain = mismatch.log_gain_x_slopes[channel] * x_normalized
                    + mismatch.log_gain_y_slopes[channel] * y_normalized
                    + mismatch.log_gain_xx[channel] * x_normalized * x_normalized
                    + mismatch.log_gain_xy[channel] * x_normalized * y_normalized
                    + mismatch.log_gain_yy[channel] * y_normalized * y_normalized;
                let gain = mismatch.center_gain[channel] * log_gain.exp();
                img[[y, x, channel]] = (img[[y, x, channel]] as f64 / gain)
                    .round()
                    .clamp(0.0, 16383.0) as u16;
            }
        }
    }
}

fn seam_correction(metrics: &serde_json::Value) -> &serde_json::Value {
    metrics
        .get("seam_exposure_correction")
        .expect("seam exposure correction diagnostics")
}

fn sum_json_array(value: &serde_json::Value) -> f64 {
    value
        .as_array()
        .expect("numeric array")
        .iter()
        .map(|entry| entry.as_f64().expect("array value"))
        .sum()
}

/// Create a 2D gradient image that varies both horizontally and vertically.
/// This is essential for drift tests — a purely horizontal gradient has identical
/// rows and can't distinguish vertical offsets.
fn gradient_2d(h: usize, w: usize) -> Array3<u16> {
    let mut arr = Array3::<u16>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            let tx = x as f64 / (w.max(1) - 1).max(1) as f64;
            let ty = y as f64 / (h.max(1) - 1).max(1) as f64;
            // R: horizontal gradient, G: vertical gradient, B: diagonal
            arr[[y, x, 0]] = (3000.0 + 7000.0 * tx).round() as u16;
            arr[[y, x, 1]] = (3000.0 + 7000.0 * ty).round() as u16;
            arr[[y, x, 2]] = (3000.0 + 7000.0 * (tx + ty) / 2.0).round() as u16;
        }
    }
    arr
}

/// Build a split pair where the right component is shifted vertically by `dy` pixels.
fn make_split_pair_with_drift(
    h: usize,
    total_w: usize,
    overlap: usize,
    dy: i32,
) -> (Array3<u16>, Array3<u16>) {
    // Create a taller source so we can shift without losing content
    let pad = dy.unsigned_abs() as usize + 1;
    let full = gradient_2d(h + 2 * pad, total_w);
    let split_point = total_w / 2 + overlap / 2;

    // Left component: rows [pad..pad+h]
    let comp1 = full
        .slice(s![pad..(pad + h), 0..split_point, ..])
        .to_owned();

    // Right component: rows [(pad+dy)..(pad+dy+h)] — shifted vertically
    let r_start = (pad as i32 + dy) as usize;
    let comp2 = full
        .slice(s![
            r_start..(r_start + h),
            (split_point - overlap)..total_w,
            ..
        ])
        .to_owned();
    (comp1, comp2)
}

#[test]
fn test_ncc_finds_correct_offset() {
    let (comp1, comp2) = make_split_pair(200, 800, 100);
    let result = scanstitch::stitch::find_overlap_ncc(&comp1, &comp2, 150, 15);
    assert!(result.is_some(), "should find overlap");
    let (offset, y_off, score) = result.unwrap();
    assert!(
        (offset as i32 - 350).unsigned_abs() < 15,
        "offset={}",
        offset
    );
    assert_eq!(y_off, 0, "should detect zero drift for aligned pair");
    assert!(score > 0.8, "score={}", score);
}

#[test]
fn test_stitch_produces_correct_width() {
    let (comp1, comp2) = make_split_pair(200, 800, 100);
    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );
    assert!(result.result.is_some(), "should succeed");
    let stitched = result.result.unwrap();
    assert_eq!(result.y_offset, 0);
    assert!(
        (stitched.dim().1 as i32 - 800).unsigned_abs() < 20,
        "w={}",
        stitched.dim().1
    );
}

#[test]
fn test_stitch_order_detection() {
    let (comp1, comp2) = make_split_pair(200, 800, 100);
    let score_12 = scanstitch::stitch::score_hypothesis(&comp1, &comp2, 150, 15);
    let score_21 = scanstitch::stitch::score_hypothesis(&comp2, &comp1, 150, 15);
    assert!(score_12 > score_21, "12={}, 21={}", score_12, score_21);
}

#[test]
fn test_three_component_sequence_infers_order_and_preserves_union() {
    let ordered = make_textured_three_way_sequence(180, 1_000, 420, 290);
    let shuffled = [&ordered[2], &ordered[0], &ordered[1]];
    let config = scanstitch::stitch::StitchConfig::default();

    let singleton = [&ordered[0]];
    let singleton_result = scanstitch::stitch::stitch_component_sequence(&singleton, &config);
    assert!(singleton_result.result.is_some());
    assert_eq!(singleton_result.report.confidence, 0.0);
    assert_eq!(
        singleton_result.report.metrics["decision"],
        "skipped_single_input"
    );
    assert_eq!(
        singleton_result.report.metrics["stitch_evidence_evaluated"],
        false
    );
    assert_eq!(
        singleton_result.report.metrics["confidence_basis"],
        "not_applicable_single_input"
    );

    let (order, diagnostics) =
        scanstitch::stitch::infer_component_sequence_order(&shuffled, &config);
    assert_eq!(order, vec![1, 2, 0]);
    assert!(diagnostics.all_adjacencies_validated);
    assert_eq!(diagnostics.accepted_adjacent_edges, 2);

    let result = scanstitch::stitch::stitch_component_sequence(&shuffled, &config);
    assert!(result.result.is_some(), "{:?}", result.report.errors);
    assert_eq!(result.order, vec![1, 2, 0]);
    assert_eq!(result.report.metrics["decision"], "accepted_sequence");
    assert_eq!(result.report.metrics["accepted_merge_count"], 2);
    assert_eq!(result.report.metrics["preserved_full_valid_union"], true);
    let stitched = result.result.unwrap();
    assert!(
        (stitched.dim().1 as i32 - 1_000).unsigned_abs() <= 20,
        "sequence union width was {} instead of approximately 1000",
        stitched.dim().1
    );
}

#[test]
fn test_three_component_sequence_aggregates_detail_review_evidence() {
    let mut ordered = make_textured_three_way_sequence(240, 1_000, 420, 290);
    let source = ordered[2].clone();
    let low_pass = box_blur_u16(&source, 2);
    for ((output, original), blurred) in ordered[2]
        .iter_mut()
        .zip(source.iter())
        .zip(low_pass.iter())
    {
        *output = (*original as f64 + 2.0 * (*original as f64 - *blurred as f64))
            .round()
            .clamp(0.0, 16383.0) as u16;
    }
    let components = [&ordered[0], &ordered[1], &ordered[2]];

    let result = scanstitch::stitch::stitch_component_sequence(
        &components,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(result.result.is_some(), "{:?}", result.report.errors);
    assert_eq!(result.report.metrics["decision"], "accepted_sequence");
    assert_eq!(
        result.report.metrics["seam_quality_review_required"], true,
        "{}",
        result.report.metrics
    );
    assert!(result.report.metrics["seam_quality_review_reasons"]
        .as_array()
        .is_some_and(|reasons| !reasons.is_empty()));
    assert!(result.report.metrics["pair_merges"]
        .as_array()
        .expect("pair merges")
        .iter()
        .any(|merge| { merge["pair_report"]["metrics"]["seam_blend"]["review_required"] == true }));
    assert!(result.report.confidence <= 0.35);
    assert!(result
        .report
        .warnings
        .iter()
        .any(|warning| warning.contains("seam-quality review")));
}

#[test]
fn test_three_component_sequence_tracks_independent_border_crop_origins() {
    let ordered = make_textured_three_way_sequence(240, 1_000, 420, 290);
    let top_origins = [0usize, 24, 41];
    let cropped = ordered
        .iter()
        .zip(top_origins)
        .map(|(component, top)| component.slice(s![top.., .., ..]).to_owned())
        .collect::<Vec<_>>();
    let shuffled = [&cropped[2], &cropped[0], &cropped[1]];
    let shuffled_origins = [41, 0, 24];
    let config = scanstitch::stitch::StitchConfig {
        max_y_offset: 5,
        ..scanstitch::stitch::StitchConfig::default()
    };

    let (order, diagnostics) =
        scanstitch::stitch::infer_component_sequence_order_with_vertical_origins(
            &shuffled,
            &shuffled_origins,
            &config,
        );
    assert_eq!(order, vec![1, 2, 0]);
    assert!(diagnostics.all_adjacencies_validated);
    assert!(diagnostics.pairwise_edges.iter().any(|edge| {
        edge.expected_vertical_offset_px.unsigned_abs() > config.max_y_offset as u32
            && edge.vertical_offset_px.is_some()
    }));

    let result = scanstitch::stitch::stitch_component_sequence_with_vertical_origins(
        &shuffled,
        &shuffled_origins,
        &config,
    );
    assert!(result.result.is_some(), "{:?}", result.report.errors);
    assert_eq!(result.order, vec![1, 2, 0]);
    assert_eq!(result.report.metrics["decision"], "accepted_sequence");
    assert_eq!(
        result.report.metrics["vertical_crop_origins"],
        serde_json::json!([41, 0, 24])
    );
    assert_eq!(result.report.metrics["preserved_full_valid_union"], true);
    let stitched = result.result.unwrap();
    assert!(
        (stitched.dim().1 as i32 - 1_000).unsigned_abs() <= 20,
        "sequence union width was {} instead of approximately 1000",
        stitched.dim().1
    );
}

#[test]
fn test_sequence_rejects_incomplete_overlap_graph_without_accepting_partial_mosaic() {
    let ordered = make_textured_three_way_sequence(160, 1_000, 420, 290);
    let unrelated = synthetic::constant_image(160, 420, [13_000, 2_000, 9_000]);
    let components = [&ordered[0], &ordered[1], &unrelated];

    let result = scanstitch::stitch::stitch_component_sequence(
        &components,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(result.result.is_none());
    assert!(!result.report.success);
    assert_eq!(result.report.metrics["decision"], "rejected_sequence");
    assert_eq!(result.report.metrics["preserved_full_valid_union"], false);
    assert!(result
        .report
        .errors
        .iter()
        .any(|error| error.contains("no partial mosaic was accepted as complete")));
}

#[test]
fn test_no_overlap_returns_none() {
    let comp1 = synthetic::constant_image(200, 400, [5000, 5000, 5000]);
    let comp2 = synthetic::constant_image(200, 400, [10000, 10000, 10000]);
    let result = scanstitch::stitch::find_overlap_ncc(&comp1, &comp2, 150, 15);
    if let Some((_, _, score)) = result {
        assert!(score < 0.5, "score too high: {}", score);
    }
}

#[test]
fn test_feathered_blend_no_seam() {
    let (comp1, comp2) = make_split_pair(200, 800, 100);
    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );
    let stitched = result.result.unwrap();
    let mid_x = stitched.dim().1 / 2;
    let y = 100;
    for c in 0..3 {
        let diff =
            (stitched[[y, mid_x, c]] as i32 - stitched[[y, mid_x + 1, c]] as i32).unsigned_abs();
        assert!(diff < 500, "seam at x={}: diff={}", mid_x, diff);
    }
}

#[test]
fn test_vertical_drift_detected() {
    // Simulate a +3 pixel vertical drift
    let (comp1, comp2) = make_split_pair_with_drift(200, 800, 100, 3);
    let result = scanstitch::stitch::find_overlap_ncc(&comp1, &comp2, 150, 15);
    assert!(result.is_some(), "should find overlap with drift");
    let (_, y_off, score) = result.unwrap();
    assert_eq!(y_off, 3, "should detect +3 drift, got {}", y_off);
    assert!(score > 0.7, "score={}", score);
}

#[test]
fn test_negative_vertical_drift_detected() {
    // Simulate a -5 pixel vertical drift
    let (comp1, comp2) = make_split_pair_with_drift(200, 800, 100, -5);
    let result = scanstitch::stitch::find_overlap_ncc(&comp1, &comp2, 150, 15);
    assert!(result.is_some(), "should find overlap with negative drift");
    let (_, y_off, score) = result.unwrap();
    assert_eq!(y_off, -5, "should detect -5 drift, got {}", y_off);
    assert!(score > 0.7, "score={}", score);
}

#[test]
fn test_stitch_with_drift_expands_canvas() {
    let (comp1, comp2) = make_split_pair_with_drift(200, 800, 100, 4);
    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );
    assert!(result.result.is_some(), "should succeed");
    let stitched = result.result.unwrap();
    // Canvas height should be at least h + |dy|
    assert!(
        stitched.dim().0 >= 200 + 4,
        "canvas_h={} too small",
        stitched.dim().0
    );
    assert_eq!(result.y_offset, 4);
}

#[test]
fn test_expected_vertical_origin_keeps_border_shift_inside_drift_window() {
    let (comp1, comp2) = make_split_pair_with_drift(220, 900, 120, 40);
    let config = scanstitch::stitch::StitchConfig {
        expected_y_offset: 40,
        max_y_offset: 5,
        ..scanstitch::stitch::StitchConfig::default()
    };

    let result = scanstitch::stitch::stitch_components(&comp1, &comp2, &config);

    assert!(result.result.is_some(), "{:?}", result.report.errors);
    assert_eq!(result.y_offset, 40);
    assert_eq!(
        result.report.metrics["acceptance_thresholds"]["expected_y_offset"],
        40
    );
    assert!(result.report.metrics["hypotheses"]
        .as_array()
        .expect("hypotheses")
        .iter()
        .all(
            |hypothesis| hypothesis["validation"]["vertical_offset_plausibility_score"]
                .as_f64()
                .is_some_and(|score| score >= 0.0)
        ));
}

#[test]
fn test_ncc_is_robust_to_exposure_difference() {
    let (comp1, mut comp2) = make_split_pair(200, 800, 100);
    let (h, w, _) = comp2.dim();
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = comp2[[y, x, c]] as f64 * 1.2;
                comp2[[y, x, c]] = v.clamp(0.0, u16::MAX as f64) as u16;
            }
        }
    }

    let result = scanstitch::stitch::find_overlap_ncc(&comp1, &comp2, 150, 15);
    assert!(result.is_some(), "should still find overlap");
    let (offset, y_off, score) = result.unwrap();
    assert!(
        (offset as i32 - 350).unsigned_abs() < 15,
        "offset={}",
        offset
    );
    assert_eq!(y_off, 0);
    assert!(score > 0.8, "score={}", score);
}

#[test]
fn test_stitch_report_contains_both_hypotheses() {
    let (comp1, comp2) = make_split_pair(200, 800, 100);
    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    let hypotheses = result.report.metrics["hypotheses"]
        .as_array()
        .expect("hypotheses array");
    assert_eq!(hypotheses.len(), 2);
    assert!(result.report.metrics["chosen_hypothesis"].is_string());
    assert!(result.report.metrics["acceptance_thresholds"].is_object());
    assert!(result.report.metrics["transform_model_used"].is_string());
    assert!(result.report.metrics["runtime"]["total_ms"].is_number());
    assert!(result.report.metrics["runtime"]["translation_evaluation_ms"].is_number());
    assert!(result.report.metrics["runtime"]["homography_attempted"].is_boolean());
    assert_eq!(result.report.metrics["method"], "ncc_translation");
    assert_eq!(
        result.report.metrics["runtime"]["native_affine_attempted"],
        false
    );
    assert!(result.report.metrics["native_affine"].is_null());
    assert!(hypotheses
        .iter()
        .all(|hyp| hyp["objective_score"].is_number()));
    assert!(hypotheses
        .iter()
        .all(|hyp| hyp["search"]["top_candidates"].is_array()));
    assert!(hypotheses
        .iter()
        .all(|hyp| hyp["search"]["selection_reason"].is_string()));
    assert!(hypotheses
        .iter()
        .all(|hyp| hyp["search"]["evaluated_candidates"].is_array()));
    assert!(hypotheses
        .iter()
        .all(|hyp| hyp["validation"]["overlap_support_score"].is_number()));
    assert!(hypotheses.iter().all(|hyp| {
        hyp["validation"]["vertical_offset_plausibility_score"].is_number()
            && hyp["validation"]["plausibility_score"].is_number()
            && hyp["validation"]["prior_weight"].is_number()
            && hyp["validation"]["local_consistency_score"].is_number()
            && hyp["validation"]["informative_window_ratio"].is_number()
            && hyp["validation"]["row_support_ratio"].is_number()
            && hyp["validation"]["column_support_ratio"].is_number()
            && hyp["validation"]["seam_support_score"].is_number()
    }));
    assert!(hypotheses.iter().all(|hyp| {
        hyp["search"]["top_candidates"]
            .as_array()
            .expect("top candidates array")
            .iter()
            .all(|candidate| {
                candidate["vertical_offset_plausibility_score"].is_number()
                    && candidate["plausibility_score"].is_number()
                    && candidate["correspondence_score"].is_number()
                    && candidate["prior_weight"].is_number()
            })
    }));
}

#[test]
fn test_seam_exposure_report_present_and_skips_matched_exposure() {
    let (comp1, comp2) = make_textured_split_pair(180, 900, 140);
    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(result.result.is_some(), "stitch should succeed");
    let correction = seam_correction(&result.report.metrics);
    assert_eq!(correction["mode"], "auto");
    assert_eq!(correction["applied"], false);
    assert!(correction["reason"].as_str().is_some());
    assert!(correction["sample_count"].as_u64().expect("sample count") > 0);
    assert_eq!(
        correction["gain_rgb"]
            .as_array()
            .expect("gain rgb")
            .iter()
            .map(|v| v.as_f64().expect("gain"))
            .collect::<Vec<_>>(),
        vec![1.0, 1.0, 1.0]
    );
    assert!(
        correction["seam_score_after"]
            .as_f64()
            .expect("score after")
            <= correction["seam_score_before"]
                .as_f64()
                .expect("score before")
                + 1e-9
    );
    let seam_blend = &result.report.metrics["seam_blend"];
    assert_eq!(seam_blend["review_required"], false, "{seam_blend}");
    assert_eq!(
        seam_blend["detail_consistency"]["review_required"], false,
        "{seam_blend}"
    );
}

#[test]
fn test_seam_blend_reports_repeated_detail_mismatch_end_to_end() {
    let (comp1, comp2) = make_textured_split_pair(240, 960, 180);
    let comp2 = box_blur_u16(&comp2, 4);

    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(
        result.result.is_some(),
        "geometry should still be measurable"
    );
    let seam_blend = &result.report.metrics["seam_blend"];
    let detail = &seam_blend["detail_consistency"];
    assert_eq!(seam_blend["review_required"], true, "{seam_blend}");
    assert_eq!(detail["evaluated"], true, "{detail}");
    assert_eq!(detail["review_required"], true, "{detail}");
    assert!(
        detail["imbalanced_scale_count"]
            .as_u64()
            .expect("imbalanced scale count")
            >= 2,
        "{detail}"
    );
    assert!(
        detail["maximum_symmetric_energy_ratio"]
            .as_f64()
            .expect("maximum ratio")
            >= 2.0,
        "{detail}"
    );
    assert!(seam_blend["review_reason"]
        .as_str()
        .expect("review reason")
        .contains("detail"));
    assert!(result.report.confidence <= 0.35);
    assert!(result
        .report
        .warnings
        .iter()
        .any(|warning| warning.contains("seam-quality review")));
}

#[test]
fn test_seam_exposure_applies_for_global_exposure_mismatch() {
    let (comp1, mut comp2) = make_textured_split_pair(180, 900, 140);
    scale_image_channels(&mut comp2, [1.14, 1.14, 1.14]);

    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(result.result.is_some(), "stitch should succeed");
    let correction = seam_correction(&result.report.metrics);
    assert_eq!(correction["applied"], true, "{}", correction);
    let gains = correction["gain_rgb"].as_array().expect("gain rgb");
    for gain in gains {
        let gain = gain.as_f64().expect("gain");
        assert!(
            (0.84..0.92).contains(&gain),
            "expected inverse gain near 1/1.14, got {} in {}",
            gain,
            correction
        );
    }
    assert!(
        correction["seam_score_after"]
            .as_f64()
            .expect("score after")
            < correction["seam_score_before"]
                .as_f64()
                .expect("score before")
    );
    assert!(
        sum_json_array(&correction["clipped_high_after"])
            <= sum_json_array(&correction["clipped_high_before"]) + 1e-9
    );
}

#[test]
fn test_seam_exposure_uses_per_channel_gain_when_stable() {
    let (comp1, mut comp2) = make_textured_split_pair(180, 900, 140);
    scale_image_channels(&mut comp2, [1.16, 0.90, 1.04]);

    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(result.result.is_some(), "stitch should succeed");
    let correction = seam_correction(&result.report.metrics);
    assert_eq!(correction["applied"], true, "{}", correction);
    assert_eq!(correction["per_channel_gain_used"], true, "{}", correction);
    let gains = correction["gain_rgb"].as_array().expect("gain rgb");
    assert!(
        gains[0].as_f64().expect("red gain") < gains[2].as_f64().expect("blue gain")
            && gains[2].as_f64().expect("blue gain") < gains[1].as_f64().expect("green gain"),
        "expected channel-specific inverse gains, got {}",
        correction
    );
    assert!(
        correction["seam_score_after"]
            .as_f64()
            .expect("score after")
            < correction["seam_score_before"]
                .as_f64()
                .expect("score before")
                * 0.5
    );
}

#[test]
fn test_seam_exposure_applies_held_out_validated_additive_compensation() {
    let (comp1, mut comp2) = make_textured_split_pair(220, 960, 180);
    offset_image_channels(&mut comp2, [780.0, 760.0, 800.0]);

    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(result.result.is_some(), "stitch should succeed");
    let correction = seam_correction(&result.report.metrics);
    assert_eq!(correction["applied"], true, "{}", correction);
    assert_eq!(correction["model"], "gain_offset_rgb", "{}", correction);
    assert_eq!(
        correction["held_out_validation_passed"], true,
        "{}",
        correction
    );
    assert!(
        correction["held_out_gain_offset_seam_score"]
            .as_f64()
            .expect("gain+offset held-out score")
            < correction["held_out_gain_seam_score"]
                .as_f64()
                .expect("gain held-out score"),
        "{}",
        correction
    );
    let offsets = correction["offset_rgb"].as_array().expect("offset rgb");
    for (offset, expected) in offsets.iter().zip([-780.0, -760.0, -800.0]) {
        assert!(
            (offset.as_f64().expect("offset") - expected).abs() < 25.0,
            "{}",
            correction
        );
    }
}

#[test]
fn test_seam_exposure_applies_vertical_shading_field_end_to_end() {
    let (comp1, mut comp2) = make_textured_split_pair(384, 960, 180);
    let source_slopes = [0.14, 0.11, 0.08];
    shade_image_channels_vertically(&mut comp2, source_slopes);

    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(result.result.is_some(), "stitch should succeed");
    let correction = seam_correction(&result.report.metrics);
    assert_eq!(correction["applied"], true, "{}", correction);
    assert_eq!(correction["model"], "gain_spatial_y_rgb", "{}", correction);
    assert_eq!(
        correction["held_out_validation_passed"], true,
        "{}",
        correction
    );
    assert!(
        correction["held_out_spatial_gain_seam_score"]
            .as_f64()
            .expect("spatial held-out score")
            < correction["held_out_gain_seam_score"]
                .as_f64()
                .expect("constant-gain held-out score"),
        "{}",
        correction
    );
    let slopes = correction["spatial_gain_log_slope_y_rgb"]
        .as_array()
        .expect("spatial slopes");
    for (actual, source) in slopes.iter().zip(source_slopes) {
        assert!(
            (actual.as_f64().expect("slope") + source).abs() < 0.015,
            "{}",
            correction
        );
    }
}

#[test]
fn test_seam_exposure_applies_vertical_gain_offset_field_end_to_end() {
    let (comp1, mut comp2) = make_textured_split_pair(384, 960, 180);
    let center_gain = [0.98, 1.02, 1.00];
    let gain_slopes = [-0.12, -0.09, -0.07];
    let center_offset = [-500.0, -450.0, -550.0];
    let offset_slopes = [160.0, 120.0, 90.0];
    impose_vertical_affine_mismatch(
        &mut comp2,
        center_gain,
        gain_slopes,
        center_offset,
        offset_slopes,
    );

    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(result.result.is_some(), "stitch should succeed");
    let correction = seam_correction(&result.report.metrics);
    assert_eq!(correction["applied"], true, "{}", correction);
    assert_eq!(
        correction["model"], "gain_offset_spatial_y_rgb",
        "{}",
        correction
    );
    assert_eq!(
        correction["held_out_validation_passed"], true,
        "{}",
        correction
    );
    assert!(
        correction["held_out_spatial_gain_offset_seam_score"]
            .as_f64()
            .expect("spatial affine held-out score")
            < correction["held_out_spatial_gain_seam_score"]
                .as_f64()
                .expect("spatial gain held-out score"),
        "{}",
        correction
    );
    let reported_gain_slopes = correction["spatial_gain_log_slope_y_rgb"]
        .as_array()
        .expect("gain slopes");
    let reported_offset_slopes = correction["spatial_offset_slope_y_rgb"]
        .as_array()
        .expect("offset slopes");
    for channel in 0..3 {
        assert!(
            (reported_gain_slopes[channel].as_f64().expect("gain slope") - gain_slopes[channel])
                .abs()
                < 0.04,
            "{}",
            correction
        );
        assert!(
            (reported_offset_slopes[channel]
                .as_f64()
                .expect("offset slope")
                - offset_slopes[channel])
                .abs()
                < 140.0,
            "{}",
            correction
        );
    }
}

#[test]
fn test_seam_exposure_applies_2d_gain_offset_field_end_to_end() {
    let overlap = 192;
    let (comp1, mut comp2) = make_textured_split_pair(384, 960, overlap);
    let mismatch = AffineMismatch2d {
        center_gain: [0.98, 1.01, 1.00],
        log_gain_x_slopes: [-0.08, -0.06, -0.05],
        log_gain_y_slopes: [-0.05, -0.04, -0.03],
        center_offset: [-460.0, -500.0, -420.0],
        offset_x_slopes: [150.0, 120.0, 90.0],
        offset_y_slopes: [100.0, 80.0, 70.0],
    };
    impose_2d_affine_mismatch(&mut comp2, overlap, &mismatch);

    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(result.result.is_some(), "stitch should succeed");
    let correction = seam_correction(&result.report.metrics);
    assert_eq!(correction["applied"], true, "{}", correction);
    assert_eq!(
        correction["model"], "gain_offset_spatial_xy_rgb",
        "{}",
        correction
    );
    assert_eq!(
        correction["held_out_validation_passed"], true,
        "{}",
        correction
    );
    assert_eq!(
        correction["spatial_2d_validation"]["gain_offset_accepted"], true,
        "{}",
        correction
    );
    let reported_gain_x_slopes = correction["spatial_gain_log_slope_x_rgb"]
        .as_array()
        .expect("gain x slopes");
    let reported_offset_x_slopes = correction["spatial_offset_slope_x_rgb"]
        .as_array()
        .expect("offset x slopes");
    for channel in 0..3 {
        assert!(
            (reported_gain_x_slopes[channel]
                .as_f64()
                .expect("gain x slope")
                - mismatch.log_gain_x_slopes[channel])
                .abs()
                < 0.04,
            "{}",
            correction
        );
        assert!(
            (reported_offset_x_slopes[channel]
                .as_f64()
                .expect("offset x slope")
                - mismatch.offset_x_slopes[channel])
                .abs()
                < 150.0,
            "{}",
            correction
        );
    }
}

#[test]
fn test_seam_exposure_applies_quadratic_2d_gain_field_end_to_end() {
    let overlap = 192;
    let (comp1, mut comp2) = make_textured_split_pair(576, 960, overlap);
    let mismatch = QuadraticGainMismatch2d {
        center_gain: [1.00, 0.99, 1.01],
        log_gain_x_slopes: [-0.025, -0.020, -0.015],
        log_gain_y_slopes: [-0.020, -0.015, -0.010],
        log_gain_xx: [-0.090, -0.080, -0.070],
        log_gain_xy: [0.030, 0.026, 0.022],
        log_gain_yy: [-0.060, -0.052, -0.045],
    };
    impose_quadratic_2d_gain_mismatch(&mut comp2, overlap, &mismatch);

    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(result.result.is_some(), "stitch should succeed");
    let correction = seam_correction(&result.report.metrics);
    assert_eq!(correction["applied"], true, "{}", correction);
    assert_eq!(
        correction["model"], "gain_spatial_quadratic_xy_rgb",
        "{}",
        correction
    );
    assert_eq!(
        correction["held_out_validation_passed"], true,
        "{}",
        correction
    );
    assert_eq!(
        correction["spatial_2d_validation"]["quadratic"]["gain_accepted"], true,
        "{}",
        correction
    );
    assert_eq!(
        correction["spatial_2d_validation"]["quadratic"]["evaluation_grid_size"], 9,
        "{}",
        correction
    );
    let reported_xx = correction["spatial_gain_log_quadratic_xx_rgb"]
        .as_array()
        .expect("quadratic xx coefficients");
    for (channel, reported) in reported_xx.iter().enumerate() {
        assert!(
            (reported.as_f64().expect("quadratic xx coefficient") - mismatch.log_gain_xx[channel])
                .abs()
                < 0.035,
            "{}",
            correction
        );
    }
}

#[test]
fn test_stitch_report_tracks_validation_vs_search_ranking() {
    let (comp1, comp2) = make_split_pair(200, 800, 100);
    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    let hypotheses = result.report.metrics["hypotheses"]
        .as_array()
        .expect("hypotheses array");
    for hypothesis in hypotheses {
        assert!(
            hypothesis["search"]["selection_reason"].as_str().is_some(),
            "selection reason should be explicit"
        );
        let evaluated = hypothesis["search"]["evaluated_candidates"]
            .as_array()
            .expect("evaluated candidates array");
        assert!(
            !evaluated.is_empty(),
            "validated candidates should be reported for the chosen hypothesis set"
        );
        assert!(evaluated
            .iter()
            .all(|candidate| candidate["search_band"].is_string()));
        assert!(evaluated
            .iter()
            .all(|candidate| candidate["objective_score"].is_number()));
        assert!(evaluated.iter().all(|candidate| {
            candidate["vertical_offset_plausibility_score"].is_number()
                && candidate["plausibility_score"].is_number()
                && candidate["prior_weight"].is_number()
                && candidate["local_consistency_score"].is_number()
        }));
    }
}

#[test]
#[ignore = "requires local LOGAN TIFFs and runs the full real-image stitch path"]
fn test_real_sample_pair_uses_rgba8_load_path_and_accepts_narrow_overlap_stitch() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path1 = repo_root.join("LOGAN043.tif");
    let path2 = repo_root.join("LOGAN044.tif");

    let loaded1 = scanstitch::tiff_io::load_tiff_u16(&path1, 14).unwrap();
    let loaded2 = scanstitch::tiff_io::load_tiff_u16(&path2, 14).unwrap();
    assert_eq!(loaded1.diagnostics.color_type, "RGBA8");
    assert_eq!(loaded2.diagnostics.color_type, "RGBA8");
    assert_eq!(loaded1.diagnostics.source_bits_per_sample, 8);
    assert_eq!(loaded2.diagnostics.source_bits_per_sample, 8);
    assert!(loaded1.diagnostics.source_has_alpha);
    assert!(loaded2.diagnostics.source_has_alpha);
    assert_eq!(
        loaded1.diagnostics.working_range.transform.as_str(),
        "upscaled_to_working_bit_depth"
    );
    assert_eq!(
        loaded2.diagnostics.working_range.transform.as_str(),
        "upscaled_to_working_bit_depth"
    );

    let border1 = scanstitch::border::remove_borders_with_diagnostics(&loaded1.image, 2);
    let border2 = scanstitch::border::remove_borders_with_diagnostics(&loaded2.image, 2);
    assert_eq!(border1.diagnostics.top_removed, 0);
    assert_eq!(border1.diagnostics.bottom_removed, 0);
    assert_eq!(border2.diagnostics.top_removed, 39);
    assert_eq!(border2.diagnostics.bottom_removed, 0);

    let expected_y_offset =
        border2.diagnostics.top_removed as i32 - border1.diagnostics.top_removed as i32;
    assert_eq!(expected_y_offset, 39);
    let config = scanstitch::stitch::StitchConfig {
        expected_y_offset,
        ..scanstitch::stitch::StitchConfig::default()
    };
    let result = scanstitch::stitch::stitch_components(&border1.cropped, &border2.cropped, &config);

    assert!(
        result.result.is_some(),
        "expected the real sample pair to stitch successfully; metrics={}",
        result.report.metrics
    );
    assert_eq!(result.order, scanstitch::stitch::StitchOrder::LeftRight);
    assert_eq!(result.y_offset, 41);
    assert_eq!(result.report.metrics["decision"], "accepted");
    assert_eq!(result.report.metrics["method"], "ncc_translation");
    assert_eq!(result.report.metrics["transform_model_used"], "translation");
    let affine = &result.report.metrics["native_affine"];
    assert_eq!(affine["accepted"], false, "{affine}");
    assert_eq!(affine["rotation_search_boundary_hit"], true, "{affine}");
    assert_eq!(affine["rotation_search_limit_deg"], 3.0, "{affine}");
    assert!(
        affine["reason"]
            .as_str()
            .is_some_and(|reason| reason.contains("search limit")),
        "{affine}"
    );
    assert_eq!(
        result.result.as_ref().map(|image| image.dim()),
        Some((3671, 11701, 3))
    );
    let seam_correction = seam_correction(&result.report.metrics);
    assert_eq!(seam_correction["model"], "identity", "{seam_correction}");
    assert_eq!(seam_correction["applied"], false, "{seam_correction}");
    assert_eq!(
        seam_correction["offset_rgb"],
        serde_json::json!([0.0, 0.0, 0.0]),
        "{seam_correction}"
    );
    assert_eq!(
        seam_correction["held_out_validation_passed"], false,
        "{seam_correction}"
    );
    let seam_blend = &result.report.metrics["seam_blend"];
    assert_eq!(
        seam_blend["review_required"], false,
        "real LOGAN seam unexpectedly triggered quality review: {seam_blend}"
    );
    assert_eq!(
        seam_blend["detail_consistency"]["evaluated"], true,
        "{seam_blend}"
    );
    assert_eq!(
        seam_blend["detail_consistency"]["review_required"], false,
        "{seam_blend}"
    );
    assert!(
        seam_blend["detail_consistency"]["supported_scale_count"]
            .as_u64()
            .is_some_and(|count| count >= 2),
        "{seam_blend}"
    );

    let hypotheses = result.report.metrics["hypotheses"]
        .as_array()
        .expect("hypotheses array");
    let left_right = hypotheses
        .iter()
        .find(|hypothesis| hypothesis["ordering"] == "[1|2]")
        .expect("[1|2] hypothesis");

    assert_eq!(left_right["accepted"], true);
    assert!(!left_right["acceptance_reason"]
        .as_str()
        .expect("acceptance reason")
        .trim()
        .is_empty());
    assert_eq!(left_right["overlap_width"], 217);
    assert!(
        left_right["validation"]["global_ncc_score"]
            .as_f64()
            .expect("global NCC")
            >= 0.50
    );
    assert!(
        left_right["validation"]["row_support_ratio"]
            .as_f64()
            .expect("row support ratio")
            >= 0.60
    );
    assert!(
        left_right["validation"]["column_support_ratio"]
            .as_f64()
            .expect("column support ratio")
            >= 0.60
    );
    assert!(
        left_right["validation"]["seam_support_score"]
            .as_f64()
            .expect("seam support score")
            >= 0.65
    );
}

#[test]
fn test_stitch_recovers_large_overlap_beyond_primary_cap() {
    let (comp1, comp2) = make_textured_split_pair(160, 1200, 420);
    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(
        result.result.is_some(),
        "expected wider-overlap stitch to succeed; report={}",
        result.report.metrics
    );
    assert_eq!(result.order, scanstitch::stitch::StitchOrder::LeftRight);

    let hypotheses = result.report.metrics["hypotheses"]
        .as_array()
        .expect("hypotheses array");
    let left_right = hypotheses
        .iter()
        .find(|hypothesis| hypothesis["ordering"] == "[1|2]")
        .expect("[1|2] hypothesis");

    assert!(
        left_right["overlap_width"]
            .as_u64()
            .expect("overlap width should be reported")
            > 300,
        "expected recovered overlap beyond the old primary 300px cap"
    );
    assert_eq!(left_right["search"]["expanded_overlap_search_used"], true);
    assert!(
        left_right["search"]["max_overlap_considered"]
            .as_u64()
            .expect("search max overlap")
            > 300
    );
}

#[test]
fn test_stitch_rejection_reports_scored_hypotheses() {
    let comp1 = synthetic::constant_image(200, 400, [5000, 5000, 5000]);
    let comp2 = synthetic::constant_image(200, 400, [10000, 10000, 10000]);
    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(result.result.is_none(), "expected stitch rejection");
    assert_eq!(result.report.metrics["decision"], "rejected");
    assert!(result.report.metrics["rejection_reason"].is_string());
    assert!(result.report.metrics["acceptance_thresholds"].is_object());
    let hypotheses = result.report.metrics["hypotheses"]
        .as_array()
        .expect("hypotheses array");
    assert_eq!(hypotheses.len(), 2);
    assert!(
        hypotheses
            .iter()
            .all(|hyp| hyp["rejection_reason"].as_str().is_some()),
        "each hypothesis should report an explicit rejection reason"
    );
}

#[test]
fn test_stitch_handles_one_row_height_mismatch() {
    let (comp1, comp2_full) = make_split_pair(200, 800, 100);
    let comp2 = comp2_full.slice(s![0..199, .., ..]).to_owned();

    let result = scanstitch::stitch::stitch_components(
        &comp1,
        &comp2,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(
        result.result.is_some(),
        "height mismatch should still stitch"
    );
    assert_eq!(result.order, scanstitch::stitch::StitchOrder::LeftRight);
}

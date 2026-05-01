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
    assert_eq!(border2.diagnostics.top_removed, 0);
    assert_eq!(border2.diagnostics.bottom_removed, 0);

    let result = scanstitch::stitch::stitch_components(
        &border1.cropped,
        &border2.cropped,
        &scanstitch::stitch::StitchConfig::default(),
    );

    assert!(
        result.result.is_some(),
        "expected the real sample pair to stitch successfully; metrics={}",
        result.report.metrics
    );
    assert_eq!(result.order, scanstitch::stitch::StitchOrder::LeftRight);
    assert_eq!(result.report.metrics["decision"], "accepted");

    let hypotheses = result.report.metrics["hypotheses"]
        .as_array()
        .expect("hypotheses array");
    let left_right = hypotheses
        .iter()
        .find(|hypothesis| hypothesis["ordering"] == "[1|2]")
        .expect("[1|2] hypothesis");

    assert_eq!(left_right["accepted"], true);
    assert!(left_right["acceptance_reason"]
        .as_str()
        .expect("acceptance reason")
        .contains("narrow-overlap seam support"));
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

mod common;
use common::synthetic;
use scanstitch::base_detect::{
    reconcile_stitched_base_estimate, roll_consensus_base, BaseDetection, RollBaseCandidate,
};

#[test]
fn test_detects_base_both_sides() {
    let img =
        synthetic::film_negative_image(500, 1000, 0, 80, [12000, 7000, 3000], [5000, 4000, 3500]);
    let result = scanstitch::base_detect::detect_film_base(&img);
    assert!(
        result.left_confidence > 0.7,
        "left confidence: {}",
        result.left_confidence
    );
    assert!(
        result.right_confidence > 0.7,
        "right confidence: {}",
        result.right_confidence
    );
    let base = &result.base_color;
    assert!(
        (base[0] as i32 - 12000).unsigned_abs() < 500,
        "base R: {}",
        base[0]
    );
    assert!(
        (base[1] as i32 - 7000).unsigned_abs() < 500,
        "base G: {}",
        base[1]
    );
}

#[test]
fn test_detects_base_left_only() {
    let mut img = synthetic::constant_image(500, 1000, [5000, 4000, 3500]);
    for y in 0..500 {
        for x in 0..80 {
            img[[y, x, 0]] = 12000;
            img[[y, x, 1]] = 7000;
            img[[y, x, 2]] = 3000;
        }
    }
    let result = scanstitch::base_detect::detect_film_base(&img);
    assert!(
        result.left_confidence > 0.7,
        "left: {}",
        result.left_confidence
    );
    assert!(
        result.right_confidence < 0.3,
        "right: {}",
        result.right_confidence
    );
}

#[test]
fn test_detects_horizontal_base_region_when_edges_do_not_hold_rebate() {
    let mut img = synthetic::constant_image(500, 1000, [5000, 4000, 3500]);
    for y in 0..30 {
        for x in 0..1000 {
            img[[y, x, 0]] = 12000;
            img[[y, x, 1]] = 7000;
            img[[y, x, 2]] = 3000;
        }
    }

    let result = scanstitch::base_detect::detect_film_base(&img);

    assert_eq!(result.base_color_source, "horizontal_base_region");
    assert!(
        (result.base_color[0] as i32 - 12000).unsigned_abs() < 500,
        "base R: {}",
        result.base_color[0]
    );
    assert!(
        (result.base_color[1] as i32 - 7000).unsigned_abs() < 500,
        "base G: {}",
        result.base_color[1]
    );
    assert!(
        result.base_color_reason.contains("horizontal rebate"),
        "reason should mention horizontal evidence: {}",
        result.base_color_reason
    );
}

#[test]
fn test_dark_horizontal_scanner_gap_is_not_treated_as_film_base() {
    let mut img = synthetic::constant_image(500, 1000, [9000, 5200, 2800]);
    for y in 0..30 {
        for x in 0..1000 {
            img[[y, x, 0]] = 380;
            img[[y, x, 1]] = 170;
            img[[y, x, 2]] = 110;
        }
    }

    let result = scanstitch::base_detect::detect_film_base(&img);

    assert_ne!(
        result.base_color_source, "horizontal_base_region",
        "dark scanner gaps should not be promoted to film base: {:?}",
        result.base_color
    );
    assert!(
        result.base_color.iter().all(|value| *value > 2500.0),
        "fallback should stay tied to image transmittance, not the dark gap: {:?}",
        result.base_color
    );
}

#[test]
fn test_measures_orange_rebate_inside_removed_scanner_band() {
    let mut img = synthetic::constant_image(120, 240, [5200, 4100, 3400]);
    for y in 0..10 {
        for x in 0..240 {
            for c in 0..3 {
                img[[y, x, c]] = 120;
            }
        }
    }
    for y in 10..14 {
        for x in 0..240 {
            img[[y, x, 0]] = 12_000;
            img[[y, x, 1]] = 7000;
            img[[y, x, 2]] = 3000;
        }
    }

    let detection = scanstitch::base_detect::detect_film_base_in_removed_bands(&img, 14, 0, 0, 0)
        .expect("orange rebate in removed top band");

    assert_eq!(detection.base_color_source, "removed_border_rebate_band");
    assert!(detection.top_confidence >= 0.70);
    assert!((detection.base_color[0] - 12_000.0).abs() < 1.0);
    assert!((detection.base_color[1] - 7000.0).abs() < 1.0);
    assert!((detection.base_color[2] - 3000.0).abs() < 1.0);
}

#[test]
fn test_removed_neutral_scanner_board_does_not_masquerade_as_orange_base() {
    let mut img = synthetic::constant_image(120, 240, [5200, 4100, 3400]);
    for y in 0..14 {
        for x in 0..240 {
            for c in 0..3 {
                img[[y, x, c]] = 60_000;
            }
        }
    }

    assert!(
        scanstitch::base_detect::detect_film_base_in_removed_bands(&img, 14, 0, 0, 0).is_none(),
        "neutral scanner board must not be accepted as orange film base"
    );
}

#[test]
fn test_removed_dark_reddish_scanner_band_is_rejected_against_frame_envelope() {
    let mut img = synthetic::constant_image(120, 240, [12_000, 7000, 4000]);
    for y in 0..14 {
        for x in 0..240 {
            img[[y, x, 0]] = 1100;
            img[[y, x, 1]] = 700;
            img[[y, x, 2]] = 380;
        }
    }

    assert!(
        scanstitch::base_detect::detect_film_base_in_removed_bands(&img, 14, 0, 0, 0)
            .is_none(),
        "a dark reddish scanner strip must be rejected when the frame contains a much higher-transmittance envelope"
    );
}

#[test]
fn test_dark_horizontal_gap_below_high_transmittance_envelope_is_not_base() {
    let mut img = synthetic::constant_image(500, 1000, [1200, 1000, 800]);
    for y in 0..30 {
        for x in 0..1000 {
            img[[y, x, 0]] = 380;
            img[[y, x, 1]] = 170;
            img[[y, x, 2]] = 110;
        }
    }
    for y in 180..260 {
        for x in 420..500 {
            img[[y, x, 0]] = 14000;
            img[[y, x, 1]] = 12000;
            img[[y, x, 2]] = 7000;
        }
    }

    let result = scanstitch::base_detect::detect_film_base(&img);

    assert_ne!(
        result.base_color_source, "horizontal_base_region",
        "dark scanner gap should be rejected when a higher-transmittance envelope exists: {:?}",
        result.base_color
    );
    assert!(
        result.base_color.iter().all(|value| *value > 1000.0),
        "base estimate should not come from the dark gap: {:?}",
        result.base_color
    );
}

#[test]
fn test_roll_consensus_rejects_dark_horizontal_cluster() {
    let mut candidates = Vec::new();
    for idx in 0..6 {
        candidates.push(roll_candidate(
            idx,
            [8856.0, 4352.0, 2071.0],
            "horizontal_base_region",
            0.85,
        ));
    }
    for idx in 6..26 {
        candidates.push(roll_candidate(
            idx,
            [
                27200.0 + ((idx % 3) as f64 * 420.0),
                12600.0 + ((idx % 4) as f64 * 180.0),
                7720.0 + ((idx % 2) as f64 * 120.0),
            ],
            "high_transmittance_fallback",
            0.04,
        ));
    }

    let consensus = roll_consensus_base(&candidates).expect("roll consensus should resolve");

    assert_eq!(consensus.source, "roll_consensus_base");
    assert_eq!(consensus.selected_cluster_frame_count, 20);
    assert_eq!(consensus.rejected_dark_candidate_count, 6);
    assert!(
        consensus.color[0] > 26000.0 && consensus.color[1] > 12000.0 && consensus.color[2] > 7400.0,
        "selected roll base should be the high-transmittance cluster, got {:?}",
        consensus.color
    );
    assert!(
        consensus.color[0] > 12000.0,
        "roll base must not resolve to the old low cluster: {:?}",
        consensus.color
    );
}

#[test]
fn test_roll_consensus_accepts_consistent_fallback_evidence() {
    let candidates = (0..5)
        .map(|idx| {
            roll_candidate(
                idx,
                [
                    24000.0 + idx as f64 * 110.0,
                    11800.0 + idx as f64 * 45.0,
                    7200.0 + idx as f64 * 30.0,
                ],
                "high_transmittance_fallback",
                0.0,
            )
        })
        .collect::<Vec<_>>();

    let consensus = roll_consensus_base(&candidates)
        .expect("consistent fallback evidence across the roll should resolve");

    assert_eq!(consensus.source, "roll_consensus_base");
    assert_eq!(consensus.selected_cluster_frame_count, 5);
    assert!(consensus.confidence >= 0.35);
}

#[test]
fn test_roll_consensus_rejects_conflicting_same_size_clusters() {
    let mut candidates = Vec::new();
    for idx in 0..3 {
        candidates.push(roll_candidate(
            idx,
            [20000.0, 10000.0, 8000.0],
            "working_edges",
            0.8,
        ));
    }
    for idx in 3..6 {
        candidates.push(roll_candidate(
            idx,
            [12000.0, 13000.0, 9000.0],
            "high_transmittance_fallback",
            0.0,
        ));
    }

    assert!(
        roll_consensus_base(&candidates).is_none(),
        "equal conflicting roll clusters should leave the base unresolved"
    );
}

#[test]
fn test_robust_to_dust() {
    let mut img =
        synthetic::film_negative_image(500, 1000, 0, 80, [12000, 7000, 3000], [5000, 4000, 3500]);
    // Add dust specks on left edge
    for y in (0..500).step_by(20) {
        for x in 0..3 {
            for c in 0..3 {
                img[[y, x, c]] = 60000;
            }
        }
    }
    let result = scanstitch::base_detect::detect_film_base(&img);
    assert!(
        result.left_confidence > 0.5,
        "should tolerate dust: {}",
        result.left_confidence
    );
}

fn roll_candidate(idx: usize, color: [f64; 3], source: &str, confidence: f64) -> RollBaseCandidate {
    RollBaseCandidate {
        frame_id: format!("RAW_{idx:04}"),
        color,
        source: source.to_string(),
        confidence,
        support_fraction: None,
    }
}

#[test]
fn test_no_base_present() {
    let img = synthetic::constant_image(500, 1000, [5000, 5000, 5000]);
    let result = scanstitch::base_detect::detect_film_base(&img);
    assert!(
        result.left_confidence < 0.5,
        "left: {}",
        result.left_confidence
    );
    assert!(
        result.right_confidence < 0.5,
        "right: {}",
        result.right_confidence
    );
    assert!(
        result.base_color.iter().all(|value| *value > 4500.0),
        "low-confidence fallback should keep a finite base color, got {:?}",
        result.base_color
    );
    assert_eq!(result.base_color_source, "high_transmittance_fallback");
}

#[test]
fn test_fallback_base_uses_joint_pixels_when_channel_percentiles_conflict() {
    let mut img = synthetic::constant_image(100, 100, [5000, 5000, 5000]);
    for i in 0..100 {
        img[[0, i, 0]] = 60000;
        img[[1, i, 1]] = 60000;
        img[[2, i, 2]] = 60000;
    }

    let result = scanstitch::base_detect::detect_film_base(&img);

    assert_eq!(result.base_color_source, "high_transmittance_fallback");
    assert!(
        result
            .base_color_reason
            .contains("joint high-transmittance pixel colour"),
        "reason should explain joint fallback: {}",
        result.base_color_reason
    );
    assert!(
        result.base_color.iter().all(|value| *value < 20000.0),
        "fallback should not combine unrelated channel outliers into impossible white: {:?}",
        result.base_color
    );
    assert_eq!(result.base_color_proxy_confidence, Some(0.0));
    assert_eq!(result.base_color_support_fraction, Some(0.0));
}

#[test]
fn test_classify_intact_frame() {
    let img =
        synthetic::film_negative_image(500, 1000, 0, 80, [12000, 7000, 3000], [5000, 4000, 3500]);
    let detection = scanstitch::base_detect::detect_film_base(&img);
    let class = scanstitch::frame_classify::classify(&detection);
    assert_eq!(class, scanstitch::frame_classify::FrameClass::Intact);
}

#[test]
fn test_classify_candidate_split() {
    let img = synthetic::constant_image(500, 1000, [5000, 4000, 3500]);
    let detection = scanstitch::base_detect::detect_film_base(&img);
    let class = scanstitch::frame_classify::classify(&detection);
    assert_eq!(
        class,
        scanstitch::frame_classify::FrameClass::CandidateSplit
    );
}

#[test]
fn test_reconcile_stitched_base_uses_component_consensus_for_darker_asymmetric_crop_edges() {
    let working = BaseDetection {
        left_confidence: 0.4206973108382335,
        right_confidence: 1.0,
        top_confidence: 0.0,
        bottom_confidence: 0.0,
        base_color: [12664.0, 10269.0, 6670.5],
        base_color_source: "working_edges",
        base_color_reason: "test edge estimate".to_string(),
        base_color_proxy_confidence: None,
        base_color_support_fraction: None,
        strip_width: 298,
        left_base_mask: vec![],
        right_base_mask: vec![],
    };

    let reconciled = reconcile_stitched_base_estimate(
        &working,
        &[
            [14477.0, 11778.333333333334, 7795.333333333333],
            [14391.0, 11629.0, 7710.0],
        ],
    );

    assert_eq!(reconciled.source, "component_consensus");
    assert!(
        reconciled.base_color[0] > 14400.0
            && reconciled.base_color[1] > 11650.0
            && reconciled.base_color[2] > 7740.0,
        "reconciled base should return to the stable pre-stitch component consensus: {:?}",
        reconciled.base_color
    );
    assert!(
        reconciled
            .reason
            .contains("stitched crop-edge base was unbalanced"),
        "expected an explicit reconciliation reason, got: {}",
        reconciled.reason
    );
    assert!(
        reconciled.confidence >= 0.95,
        "component consensus should remain high-confidence when the component bases agree closely: {}",
        reconciled.confidence
    );
}

#[test]
fn test_reconcile_stitched_base_keeps_working_edges_when_consistent_with_components() {
    let working = BaseDetection {
        left_confidence: 0.88,
        right_confidence: 0.93,
        top_confidence: 0.0,
        bottom_confidence: 0.0,
        base_color: [14420.0, 11710.0, 7760.0],
        base_color_source: "working_edges",
        base_color_reason: "test edge estimate".to_string(),
        base_color_proxy_confidence: None,
        base_color_support_fraction: None,
        strip_width: 64,
        left_base_mask: vec![],
        right_base_mask: vec![],
    };

    let reconciled = reconcile_stitched_base_estimate(
        &working,
        &[[14477.0, 11778.0, 7795.0], [14391.0, 11629.0, 7710.0]],
    );

    assert_eq!(reconciled.source, "working_edges");
    assert_eq!(reconciled.base_color, working.base_color);
    assert!(
        reconciled
            .reason
            .contains("stayed close to the component consensus"),
        "expected a no-override diagnostic reason, got: {}",
        reconciled.reason
    );
}

fn sampled_band_median_mad(
    image: &ndarray::Array3<u16>,
    x_start: usize,
    x_end: usize,
) -> ([f64; 3], [f64; 3]) {
    let (height, width, _) = image.dim();
    let x_start = x_start.min(width);
    let x_end = x_end.min(width).max(x_start + 1);
    let mut channels: [Vec<u16>; 3] = std::array::from_fn(|_| Vec::new());
    for y in (0..height).step_by(8) {
        for x in x_start..x_end {
            for channel in 0..3 {
                channels[channel].push(image[[y, x, channel]]);
            }
        }
    }
    let median = std::array::from_fn(|channel| {
        channels[channel].sort_unstable();
        channels[channel][channels[channel].len() / 2] as f64
    });
    let mad = std::array::from_fn(|channel| {
        let mut deviations = channels[channel]
            .iter()
            .map(|value| (*value as f64 - median[channel]).abs() as u16)
            .collect::<Vec<_>>();
        deviations.sort_unstable();
        deviations[deviations.len() / 2] as f64
    });
    (median, mad)
}

#[test]
#[ignore = "requires the local OLD_TESTROLL DNG corpus"]
fn test_real_testroll_curved_rebate_diagnostics() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("OLD_TESTROLL")
        .join("RAW_0000.dng");
    if !path.exists() {
        eprintln!("local OLD_TESTROLL fixture is absent; diagnostic skipped");
        return;
    }
    let loaded = scanstitch::tiff_io::load_tiff_u16(&path, 16).unwrap();
    let deskew = scanstitch::deskew::auto_deskew(
        loaded.image,
        u16::MAX as f64,
        &scanstitch::deskew::DeskewConfig::default(),
    );
    assert_eq!(deskew.diagnostics.status, "already_aligned");
    assert!(!deskew.diagnostics.applied);
    assert!(!deskew.diagnostics.review_required);
    assert_eq!(deskew.diagnostics.horizontal_side_count, 2);
    assert_eq!(deskew.diagnostics.retained_area_ratio, 1.0);
    assert!(deskew
        .diagnostics
        .detected_source_skew_degrees
        .is_some_and(|angle| angle.abs() < 0.08));
    let border = scanstitch::border::remove_borders_with_diagnostics(&deskew.image, 2);
    let image = border.cropped;
    let (_, width, _) = image.dim();
    let detection = scanstitch::base_detect::detect_film_base(&image);
    assert_eq!(border.diagnostics.top_removed, 111);
    assert_eq!(border.diagnostics.bottom_removed, 129);
    assert_eq!(border.diagnostics.left_removed, 34);
    assert_eq!(border.diagnostics.right_removed, 0);
    assert_eq!(detection.base_color_source, "high_transmittance_fallback");
    assert_eq!(
        scanstitch::base_detect::base_detection_confidence(&detection),
        0.0,
        "the scene-derived fallback must not be promoted to measured film-base evidence"
    );
    assert!(detection
        .base_color_proxy_confidence
        .is_some_and(|confidence| confidence <= 0.01));
    assert!(detection
        .base_color_support_fraction
        .is_some_and(|support| support < 0.001));
    assert!(detection.base_color_reason.contains("low proxy confidence"));
    eprintln!(
        "border={:?} detection confidence L/R/T/B={:.3}/{:.3}/{:.3}/{:.3}, base={:?}, reason={}",
        border.diagnostics,
        detection.left_confidence,
        detection.right_confidence,
        detection.top_confidence,
        detection.bottom_confidence,
        detection.base_color,
        detection.base_color_reason,
    );
    for band_width in [8usize, 16, 32, 64, 96, 128, 192, 256, 298, 384] {
        let left = sampled_band_median_mad(&image, 0, band_width);
        let right = sampled_band_median_mad(&image, width - band_width, width);
        eprintln!(
            "band={band_width:>3}: left median={:?} mad={:?}; right median={:?} mad={:?}",
            left.0, left.1, right.0, right.1
        );
    }
}

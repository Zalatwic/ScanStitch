mod common;
use common::synthetic;
use scanstitch::base_detect::{reconcile_stitched_base_estimate, BaseDetection};

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
        base_color: [12664.0, 10269.0, 6670.5],
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
        base_color: [14420.0, 11710.0, 7760.0],
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

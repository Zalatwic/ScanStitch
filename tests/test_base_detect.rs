mod common;
use common::synthetic;

#[test]
fn test_detects_base_both_sides() {
    let img = synthetic::film_negative_image(500, 1000, 0, 80, [12000, 7000, 3000], [5000, 4000, 3500]);
    let result = scanstitch::base_detect::detect_film_base(&img);
    assert!(result.left_confidence > 0.7, "left confidence: {}", result.left_confidence);
    assert!(result.right_confidence > 0.7, "right confidence: {}", result.right_confidence);
    let base = &result.base_color;
    assert!((base[0] as i32 - 12000).unsigned_abs() < 500, "base R: {}", base[0]);
    assert!((base[1] as i32 - 7000).unsigned_abs() < 500, "base G: {}", base[1]);
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
    assert!(result.left_confidence > 0.7, "left: {}", result.left_confidence);
    assert!(result.right_confidence < 0.3, "right: {}", result.right_confidence);
}

#[test]
fn test_robust_to_dust() {
    let mut img = synthetic::film_negative_image(500, 1000, 0, 80, [12000, 7000, 3000], [5000, 4000, 3500]);
    // Add dust specks on left edge
    for y in (0..500).step_by(20) {
        for x in 0..3 {
            for c in 0..3 {
                img[[y, x, c]] = 60000;
            }
        }
    }
    let result = scanstitch::base_detect::detect_film_base(&img);
    assert!(result.left_confidence > 0.5, "should tolerate dust: {}", result.left_confidence);
}

#[test]
fn test_no_base_present() {
    let img = synthetic::constant_image(500, 1000, [5000, 5000, 5000]);
    let result = scanstitch::base_detect::detect_film_base(&img);
    assert!(result.left_confidence < 0.5, "left: {}", result.left_confidence);
    assert!(result.right_confidence < 0.5, "right: {}", result.right_confidence);
}

#[test]
fn test_classify_intact_frame() {
    let img = synthetic::film_negative_image(500, 1000, 0, 80, [12000, 7000, 3000], [5000, 4000, 3500]);
    let detection = scanstitch::base_detect::detect_film_base(&img);
    let class = scanstitch::frame_classify::classify(&detection);
    assert_eq!(class, scanstitch::frame_classify::FrameClass::Intact);
}

#[test]
fn test_classify_candidate_split() {
    let img = synthetic::constant_image(500, 1000, [5000, 4000, 3500]);
    let detection = scanstitch::base_detect::detect_film_base(&img);
    let class = scanstitch::frame_classify::classify(&detection);
    assert_eq!(class, scanstitch::frame_classify::FrameClass::CandidateSplit);
}

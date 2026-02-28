mod common;
use common::synthetic;
use ndarray::{Array3, s};

fn make_split_pair(h: usize, total_w: usize, overlap: usize) -> (Array3<u16>, Array3<u16>) {
    let full = synthetic::gradient_image(h, total_w, [3000, 5000, 8000], [10000, 8000, 4000]);
    let split_point = total_w / 2 + overlap / 2;
    let comp1 = full.slice(s![.., 0..split_point, ..]).to_owned();
    let comp2 = full.slice(s![.., (split_point - overlap)..total_w, ..]).to_owned();
    (comp1, comp2)
}

#[test]
fn test_ncc_finds_correct_offset() {
    let (comp1, comp2) = make_split_pair(200, 800, 100);
    let result = scanstitch::stitch::find_overlap_ncc(&comp1, &comp2, 150);
    assert!(result.is_some(), "should find overlap");
    let (offset, score) = result.unwrap();
    assert!((offset as i32 - 350).unsigned_abs() < 15, "offset={}", offset);
    assert!(score > 0.8, "score={}", score);
}

#[test]
fn test_stitch_produces_correct_width() {
    let (comp1, comp2) = make_split_pair(200, 800, 100);
    let result = scanstitch::stitch::stitch_components(&comp1, &comp2, &scanstitch::stitch::StitchConfig::default());
    assert!(result.result.is_some(), "should succeed");
    let stitched = result.result.unwrap();
    assert_eq!(stitched.dim().0, 200);
    assert!((stitched.dim().1 as i32 - 800).unsigned_abs() < 20, "w={}", stitched.dim().1);
}

#[test]
fn test_stitch_order_detection() {
    let (comp1, comp2) = make_split_pair(200, 800, 100);
    let score_12 = scanstitch::stitch::score_hypothesis(&comp1, &comp2, 150);
    let score_21 = scanstitch::stitch::score_hypothesis(&comp2, &comp1, 150);
    assert!(score_12 > score_21, "12={}, 21={}", score_12, score_21);
}

#[test]
fn test_no_overlap_returns_none() {
    let comp1 = synthetic::constant_image(200, 400, [5000, 5000, 5000]);
    let comp2 = synthetic::constant_image(200, 400, [10000, 10000, 10000]);
    let result = scanstitch::stitch::find_overlap_ncc(&comp1, &comp2, 150);
    if let Some((_, score)) = result {
        assert!(score < 0.5, "score too high: {}", score);
    }
}

#[test]
fn test_feathered_blend_no_seam() {
    let (comp1, comp2) = make_split_pair(200, 800, 100);
    let result = scanstitch::stitch::stitch_components(&comp1, &comp2, &scanstitch::stitch::StitchConfig::default());
    let stitched = result.result.unwrap();
    let mid_x = stitched.dim().1 / 2;
    let y = 100;
    for c in 0..3 {
        let diff = (stitched[[y, mid_x, c]] as i32 - stitched[[y, mid_x + 1, c]] as i32).unsigned_abs();
        assert!(diff < 500, "seam at x={}: diff={}", mid_x, diff);
    }
}

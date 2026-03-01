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
    let comp1 = full.slice(s![pad..(pad + h), 0..split_point, ..]).to_owned();

    // Right component: rows [(pad+dy)..(pad+dy+h)] — shifted vertically
    let r_start = (pad as i32 + dy) as usize;
    let comp2 = full.slice(s![r_start..(r_start + h), (split_point - overlap)..total_w, ..]).to_owned();
    (comp1, comp2)
}

#[test]
fn test_ncc_finds_correct_offset() {
    let (comp1, comp2) = make_split_pair(200, 800, 100);
    let result = scanstitch::stitch::find_overlap_ncc(&comp1, &comp2, 150, 15);
    assert!(result.is_some(), "should find overlap");
    let (offset, y_off, score) = result.unwrap();
    assert!((offset as i32 - 350).unsigned_abs() < 15, "offset={}", offset);
    assert_eq!(y_off, 0, "should detect zero drift for aligned pair");
    assert!(score > 0.8, "score={}", score);
}

#[test]
fn test_stitch_produces_correct_width() {
    let (comp1, comp2) = make_split_pair(200, 800, 100);
    let result = scanstitch::stitch::stitch_components(&comp1, &comp2, &scanstitch::stitch::StitchConfig::default());
    assert!(result.result.is_some(), "should succeed");
    let stitched = result.result.unwrap();
    assert_eq!(result.y_offset, 0);
    assert!((stitched.dim().1 as i32 - 800).unsigned_abs() < 20, "w={}", stitched.dim().1);
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
    let result = scanstitch::stitch::stitch_components(&comp1, &comp2, &scanstitch::stitch::StitchConfig::default());
    let stitched = result.result.unwrap();
    let mid_x = stitched.dim().1 / 2;
    let y = 100;
    for c in 0..3 {
        let diff = (stitched[[y, mid_x, c]] as i32 - stitched[[y, mid_x + 1, c]] as i32).unsigned_abs();
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
    let result = scanstitch::stitch::stitch_components(&comp1, &comp2, &scanstitch::stitch::StitchConfig::default());
    assert!(result.result.is_some(), "should succeed");
    let stitched = result.result.unwrap();
    // Canvas height should be at least h + |dy|
    assert!(stitched.dim().0 >= 200 + 4, "canvas_h={} too small", stitched.dim().0);
    assert_eq!(result.y_offset, 4);
}

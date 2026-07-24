use ndarray::Array3;
use scanstitch::deskew::{auto_deskew, manual_deskew, DeskewConfig};

fn framed_scan(height: usize, width: usize, border: usize) -> Array3<u16> {
    let mut image = Array3::<u16>::zeros((height, width, 3));
    for y in 0..height {
        for x in 0..width {
            let inside = x >= border && x < width - border && y >= border && y < height - border;
            for channel in 0..3 {
                image[[y, x, channel]] = if inside {
                    24_000 + ((x * 31 + y * 17 + channel * 101) % 2_000) as u16
                } else {
                    1_500 + channel as u16 * 120
                };
            }
        }
    }
    image
}

fn horizontal_border_scan(height: usize, width: usize, border: usize) -> Array3<u16> {
    let mut image = Array3::<u16>::zeros((height, width, 3));
    for y in 0..height {
        for x in 0..width {
            let inside = y >= border && y < height - border;
            for channel in 0..3 {
                image[[y, x, channel]] = if inside {
                    22_000 + ((x * 19 + y * 11 + channel * 71) % 1_200) as u16
                } else {
                    1_800 + channel as u16 * 90
                };
            }
        }
    }
    image
}

#[test]
fn auto_deskew_recovers_small_absolute_rotation_from_four_border_sides() {
    let config = DeskewConfig::default();
    let source = framed_scan(360, 520, 36);
    let skewed = manual_deskew(source, -1.10, u16::MAX as f64, &config);
    assert!(skewed.diagnostics.applied);

    let corrected = auto_deskew(skewed.image, u16::MAX as f64, &config);
    assert!(
        corrected.diagnostics.applied,
        "{}: {:?}",
        corrected.diagnostics.reason, corrected.diagnostics.side_estimates
    );
    assert_eq!(corrected.diagnostics.status, "applied");
    assert!(corrected.diagnostics.supporting_side_count >= 2);
    assert!(corrected.diagnostics.horizontal_side_count >= 1);
    assert!(corrected.diagnostics.vertical_side_count >= 1);
    let detected = corrected
        .diagnostics
        .detected_source_skew_degrees
        .expect("detected angle");
    assert!((detected - 1.10).abs() < 0.20, "detected {detected}");
    assert!(corrected.diagnostics.retained_area_ratio >= 0.90);
    assert_eq!(
        corrected.diagnostics.interpolation.as_deref(),
        Some("bicubic_catmull_rom_single_resample")
    );
}

#[test]
fn auto_deskew_leaves_aligned_frame_unresampled() {
    let source = framed_scan(260, 400, 28);
    let result = auto_deskew(source.clone(), u16::MAX as f64, &DeskewConfig::default());
    assert!(!result.diagnostics.applied);
    assert_eq!(result.diagnostics.status, "already_aligned");
    assert_eq!(result.diagnostics.retained_area_ratio, 1.0);
    assert_eq!(result.image, source);
}

#[test]
fn auto_deskew_accepts_two_coherent_opposing_borders_on_one_axis() {
    let config = DeskewConfig::default();
    let source = horizontal_border_scan(340, 500, 34);
    let skewed = manual_deskew(source, -0.85, u16::MAX as f64, &config);
    let corrected = auto_deskew(skewed.image, u16::MAX as f64, &config);

    assert!(
        corrected.diagnostics.applied,
        "{}: {:?}",
        corrected.diagnostics.reason, corrected.diagnostics.side_estimates
    );
    assert_eq!(corrected.diagnostics.horizontal_side_count, 2);
    assert_eq!(corrected.diagnostics.vertical_side_count, 0);
    assert!(corrected
        .diagnostics
        .detected_source_skew_degrees
        .is_some_and(|angle| (angle - 0.85).abs() < 0.20));
}

#[test]
fn auto_deskew_does_not_invent_rotation_without_border_evidence() {
    let source = Array3::from_elem((180, 240, 3), 12_000u16);
    let result = auto_deskew(source.clone(), u16::MAX as f64, &DeskewConfig::default());
    assert!(!result.diagnostics.applied);
    assert_eq!(result.diagnostics.status, "insufficient_evidence");
    assert!(!result.diagnostics.review_required);
    assert_eq!(result.image, source);
}

#[test]
fn manual_deskew_reports_signed_correction_and_retained_area() {
    let source = framed_scan(300, 450, 32);
    let result = manual_deskew(source, 0.75, u16::MAX as f64, &DeskewConfig::default());
    assert!(result.diagnostics.applied);
    assert_eq!(result.diagnostics.detected_source_skew_degrees, Some(0.75));
    assert_eq!(result.diagnostics.correction_degrees, Some(-0.75));
    assert!(result.diagnostics.retained_area_ratio < 1.0);
    assert!(result.diagnostics.retained_area_ratio >= 0.90);
    assert!(result.image.dim().0 < 300);
    assert!(result.image.dim().1 < 450);
}

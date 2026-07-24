use ndarray::{s, Array3};

fn valid_scanner_linearization(
) -> scanstitch::scanner_linearization::ScannerLinearizationCalibration {
    let curve = vec![
        [0.0, 0.0],
        [0.25, 0.0625],
        [0.5, 0.25],
        [0.75, 0.5625],
        [1.0, 1.0],
    ];
    scanstitch::scanner_linearization::ScannerLinearizationCalibration {
        model_id: "synthetic-scanner-linearization".to_string(),
        curves: scanstitch::scanner_linearization::ScannerLinearizationCurves {
            red: curve.clone(),
            green: curve.clone(),
            blue: curve,
        },
        black_level_normalized: [0.05; 3],
        white_level_normalized: [0.95; 3],
        additive_flare_normalized: [0.01; 3],
        shading_gain_polynomial: Some([
            [1.0, 0.05, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.03, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.01, 0.0, 0.0, 0.0, 0.0],
        ]),
        confidence: 0.94,
        validation: scanstitch::scanner_linearization::ScannerLinearizationValidation {
            held_out_sample_count: 24,
            transmittance_rmse: 0.002,
            transmittance_max_error: 0.008,
            identity_baseline_rmse: 0.08,
        },
    }
}

#[test]
fn test_scanner_linearization_rejects_weak_nonmonotone_calibration() {
    let mut calibration = valid_scanner_linearization();
    calibration.confidence = 0.5;
    calibration.validation.held_out_sample_count = 3;
    calibration.curves.red[2][1] = 0.01;

    let errors = scanstitch::scanner_linearization::validate_scanner_linearization(&calibration)
        .expect_err("weak nonmonotone calibration must be rejected");
    assert!(errors.iter().any(|error| error.contains("confidence")));
    assert!(errors
        .iter()
        .any(|error| error.contains("strictly increasing")));
    assert!(errors
        .iter()
        .any(|error| error.contains("held-out samples")));
}

#[test]
fn test_scanner_linearization_applies_levels_flare_curve_and_shading() {
    let calibration = valid_scanner_linearization();
    let mut image = Array3::<u16>::zeros((5, 9, 3));
    image.fill((0.50 * 65_535.0) as u16);

    let result =
        scanstitch::scanner_linearization::apply_scanner_linearization(image, 16, &calibration)
            .expect("validated scanner correction");

    assert_eq!(result.diagnostics.status, "applied");
    assert_eq!(result.diagnostics.output_bit_depth, 16);
    assert!(result.diagnostics.maximum_signal_noise_gain < 4.0);
    assert_eq!(result.diagnostics.curve_extrapolated_low_samples, [0; 3]);
    assert_eq!(result.diagnostics.curve_extrapolated_high_samples, [0; 3]);
    assert!(
        result.image[[2, 8, 0]] > result.image[[2, 0, 0]],
        "positive x shading gain should brighten the right edge"
    );
    assert!(
        result.image[[2, 8, 0]] - result.image[[2, 0, 0]]
            > result.image[[2, 8, 2]] - result.image[[2, 0, 2]],
        "per-channel shading polynomials should remain independent"
    );
    let center = result.image[[2, 4, 1]] as f64 / 65_535.0;
    assert!((center - 0.239).abs() < 0.01, "center value {center}");

    let transformed_base =
        scanstitch::scanner_linearization::transform_base_color([32_767.5; 3], 16, &calibration);
    assert!((transformed_base[0] / 65_535.0 - center).abs() < 0.01);
}

#[test]
fn test_scanner_linearization_reports_clipping_and_curve_extrapolation() {
    let calibration = valid_scanner_linearization();
    let mut image = Array3::<u16>::zeros((2, 2, 3));
    for channel in 0..3 {
        image[[0, 0, channel]] = 0;
        image[[0, 1, channel]] = 65_535;
        image[[1, 0, channel]] = 20_000;
        image[[1, 1, channel]] = 40_000;
    }
    let result =
        scanstitch::scanner_linearization::apply_scanner_linearization(image, 16, &calibration)
            .unwrap();

    for channel in 0..3 {
        assert!(result.diagnostics.curve_extrapolated_low_samples[channel] > 0);
        assert!(result.diagnostics.curve_extrapolated_high_samples[channel] > 0);
        assert_eq!(result.image[[0, 0, channel]], 0);
        assert!(result.diagnostics.clipped_high_samples[channel] > 0);
    }
}

#[test]
fn test_scanner_linearization_spatial_field_uses_original_coordinates_after_orientation() {
    let calibration = valid_scanner_linearization();
    let mut scanner_order = Array3::<u16>::zeros((5, 9, 3));
    scanner_order.fill((0.50 * 65_535.0) as u16);
    let expected_scanner_order = scanstitch::scanner_linearization::apply_scanner_linearization(
        scanner_order.clone(),
        16,
        &calibration,
    )
    .unwrap()
    .image;

    // EXIF orientation 6 maps source (5x9) to a clockwise-oriented image (9x5).
    let mut oriented = Array3::<u16>::zeros((9, 5, 3));
    let mut expected_oriented = Array3::<u16>::zeros((9, 5, 3));
    for output_y in 0..9 {
        for output_x in 0..5 {
            let source_y = 5 - 1 - output_x;
            let source_x = output_y;
            for channel in 0..3 {
                oriented[[output_y, output_x, channel]] =
                    scanner_order[[source_y, source_x, channel]];
                expected_oriented[[output_y, output_x, channel]] =
                    expected_scanner_order[[source_y, source_x, channel]];
            }
        }
    }

    let result = scanstitch::scanner_linearization::apply_scanner_linearization_with_coordinates(
        oriented,
        16,
        &calibration,
        scanstitch::scanner_linearization::ScannerCoordinateMapping {
            orientation_tag: Some(6),
            scanner_frame_width: 9,
            scanner_frame_height: 5,
        },
    )
    .unwrap();

    assert_eq!(result.image, expected_oriented);
    assert_eq!(
        result.diagnostics.coordinate_domain,
        "original_scanner_frame_via_inverse_exif_orientation"
    );
    assert_eq!(result.diagnostics.orientation_tag, Some(6));
}

#[test]
fn test_per_scan_spatial_linearization_removes_overlap_shading_mismatch() {
    let identity_curve = vec![
        [0.0, 0.0],
        [0.25, 0.25],
        [0.50, 0.50],
        [0.75, 0.75],
        [1.0, 1.0],
    ];
    let calibration = scanstitch::scanner_linearization::ScannerLinearizationCalibration {
        model_id: "synthetic-per-scan-shading".to_string(),
        curves: scanstitch::scanner_linearization::ScannerLinearizationCurves {
            red: identity_curve.clone(),
            green: identity_curve.clone(),
            blue: identity_curve,
        },
        black_level_normalized: [0.0; 3],
        white_level_normalized: [1.0; 3],
        additive_flare_normalized: [0.0; 3],
        shading_gain_polynomial: Some([
            [1.0, 0.12, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.10, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.08, 0.0, 0.0, 0.0, 0.0],
        ]),
        confidence: 0.95,
        validation: scanstitch::scanner_linearization::ScannerLinearizationValidation {
            held_out_sample_count: 24,
            transmittance_rmse: 0.002,
            transmittance_max_error: 0.008,
            identity_baseline_rmse: 0.06,
        },
    };
    let height = 64usize;
    let component_width = 240usize;
    let overlap = 80usize;
    let x_offset = component_width - overlap;
    let mut components = [
        Array3::<u16>::zeros((height, component_width, 3)),
        Array3::<u16>::zeros((height, component_width, 3)),
    ];
    for (component_index, component) in components.iter_mut().enumerate() {
        for y in 0..height {
            for x in 0..component_width {
                let global_x = component_index * x_offset + x;
                let xn = 2.0 * x as f64 / (component_width - 1) as f64 - 1.0;
                for channel in 0..3 {
                    let true_signal = 0.38
                        + 0.08 * (global_x as f64 * 0.071 + channel as f64).sin()
                        + 0.04 * (y as f64 * 0.13).cos();
                    let correction_gain = 1.0
                        + calibration.shading_gain_polynomial.as_ref().unwrap()[channel][1] * xn;
                    component[[y, x, channel]] =
                        (true_signal / correction_gain * 65_535.0).round() as u16;
                }
            }
        }
    }
    let overlap_difference = |left: &Array3<u16>, right: &Array3<u16>| {
        left.slice(s![.., x_offset.., ..])
            .iter()
            .zip(right.slice(s![.., ..overlap, ..]).iter())
            .map(|(left, right)| (*left as f64 - *right as f64).abs() / 65_535.0)
            .sum::<f64>()
            / (height * overlap * 3) as f64
    };
    let before = overlap_difference(&components[0], &components[1]);
    let corrected_left = scanstitch::scanner_linearization::apply_scanner_linearization(
        components[0].clone(),
        16,
        &calibration,
    )
    .unwrap()
    .image;
    let corrected_right = scanstitch::scanner_linearization::apply_scanner_linearization(
        components[1].clone(),
        16,
        &calibration,
    )
    .unwrap()
    .image;
    let after = overlap_difference(&corrected_left, &corrected_right);

    assert!(
        before > 0.04,
        "expected visible synthetic shading seam: {before}"
    );
    assert!(after < 0.0001, "per-scan correction left mismatch {after}");
    assert!(after < before * 0.002, "{before} -> {after}");
}

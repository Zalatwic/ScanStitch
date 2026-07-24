use approx::assert_relative_eq;
use ndarray::Array3;

fn measured_response_fixture() -> scanstitch::density::MeasuredNegativeResponseCalibration {
    scanstitch::density::MeasuredNegativeResponseCalibration {
        model_id: "synthetic-held-out-response".to_string(),
        scanner_density_to_layer_density: [
            [1.0, -0.08, 0.01],
            [-0.04, 1.0, -0.05],
            [0.01, -0.04, 1.0],
        ],
        characteristic_curves: scanstitch::density::NegativeCharacteristicCurves {
            red: vec![[0.0, 0.0], [0.25, 0.25], [0.70, 0.75], [1.30, 1.40]],
            green: vec![[0.0, 0.0], [0.30, 0.25], [0.85, 0.75], [1.45, 1.40]],
            blue: vec![[0.0, 0.0], [0.20, 0.25], [0.62, 0.75], [1.18, 1.40]],
        },
        white_anchor_percentile: 0.995,
        confidence: 0.94,
        validation: scanstitch::density::NegativeResponseValidation {
            held_out_patch_count: 24,
            delta_e00_rms: 1.8,
            delta_e00_max: 4.9,
            unit_slope_delta_e00_rms: 8.2,
            worst_hue_family: Some("deep-blue".to_string()),
        },
    }
}

#[test]
fn test_normalize_u16_to_float() {
    let mut img = Array3::<u16>::zeros((2, 2, 3));
    img[[0, 0, 0]] = 16383;
    img[[0, 0, 1]] = 8192;
    img[[0, 0, 2]] = 0;
    let float_img = scanstitch::density::normalize_to_float(&img, 14);
    assert_relative_eq!(float_img[[0, 0, 0]], 1.0, epsilon = 1e-4);
    assert_relative_eq!(float_img[[0, 0, 1]], 0.5, epsilon = 0.01);
    assert!(float_img[[0, 0, 2]] >= 1e-6);
}

#[test]
fn test_density_transmittance_paths_clamp_overrange_samples() {
    let mut img = Array3::<u16>::zeros((1, 2, 3));
    for c in 0..3 {
        img[[0, 0, c]] = 16_383;
        img[[0, 1, c]] = 65_535;
    }

    let normalized = scanstitch::density::normalize_to_float(&img, 14);
    assert_relative_eq!(normalized[[0, 0, 0]], 1.0, epsilon = 1e-12);
    assert_relative_eq!(normalized[[0, 1, 0]], 1.0, epsilon = 1e-12);

    let overrange_density = scanstitch::density::transmittance_to_density(4.0);
    assert_relative_eq!(overrange_density, 0.0, epsilon = 1e-12);
    assert_relative_eq!(
        scanstitch::density::density_to_transmittance(-2.0),
        1.0,
        epsilon = 1e-12
    );

    let result = scanstitch::density::phase3_invert_with_diagnostics(
        &img,
        &[65_535.0, 65_535.0, 65_535.0],
        14,
    );
    assert_eq!(result.diagnostics.base_transmittance, [1.0, 1.0, 1.0]);
    assert!(result
        .positive_density
        .iter()
        .all(|value| value.is_finite()));
}

#[test]
fn test_positive_scan_normalization_preserves_luminance_order_without_density_inversion() {
    let mut img = Array3::<u16>::zeros((1, 3, 3));
    for c in 0..3 {
        img[[0, 0, c]] = 0;
        img[[0, 1, c]] = 8192;
        img[[0, 2, c]] = 16383;
    }

    let positive = scanstitch::density::normalize_positive_scan_to_linear_rgb(&img, 14);

    assert_relative_eq!(positive[[0, 0, 0]], 0.0, epsilon = 1e-12);
    assert_relative_eq!(positive[[0, 2, 0]], 1.0, epsilon = 1e-12);
    assert!(
        positive[[0, 0, 0]] < positive[[0, 1, 0]] && positive[[0, 1, 0]] < positive[[0, 2, 0]],
        "positive-mode normalization must not apply density-domain inversion"
    );
}

#[test]
fn test_density_conversion_roundtrip() {
    let t = 0.5f64;
    let d = scanstitch::density::transmittance_to_density(t);
    let t2 = scanstitch::density::density_to_transmittance(d);
    assert_relative_eq!(t, t2, epsilon = 1e-10);
}

#[test]
fn test_direct_density_render_transmittance_normalizes_optical_density_range() {
    let mut density = Array3::<f64>::zeros((1, 3, 3));
    for c in 0..3 {
        density[[0, 0, c]] = 0.0;
        density[[0, 1, c]] = 2.5;
        density[[0, 2, c]] = 5.0;
    }

    let raw_transmittance = scanstitch::density::density_image_to_transmittance(&density);
    let render_transmittance =
        scanstitch::density::density_image_to_normalized_transmittance(&density, 5.0);

    assert_relative_eq!(render_transmittance[[0, 0, 0]], 1.0, epsilon = 1e-12);
    assert_relative_eq!(
        render_transmittance[[0, 1, 0]],
        10f64.powf(-0.5),
        epsilon = 1e-12
    );
    assert_relative_eq!(render_transmittance[[0, 2, 0]], 0.1, epsilon = 1e-12);
    assert!(
        raw_transmittance[[0, 2, 0]] < 0.000_02,
        "unbounded scanner optical density would collapse direct-density render shadows"
    );
}

#[test]
fn test_phase3_uses_shared_robust_dmax_for_all_channels() {
    let mut img = Array3::<u16>::zeros((1, 256, 3));
    let base = [12_000u16, 8_000, 4_000];
    for x in 0..256 {
        let t = x as f64 / 255.0;
        img[[0, x, 0]] = (base[0] as f64 * (0.80 - 0.50 * t)).round().max(1.0) as u16;
        img[[0, x, 1]] = (base[1] as f64 * (0.85 - 0.25 * t)).round().max(1.0) as u16;
        img[[0, x, 2]] = (base[2] as f64 * (0.90 - 0.10 * t)).round().max(1.0) as u16;
    }

    let result = scanstitch::density::phase3_invert_with_diagnostics(
        &img,
        &[base[0] as f64, base[1] as f64, base[2] as f64],
        14,
    );
    let max_channel_dmax = result
        .diagnostics
        .robust_d_max
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    assert_relative_eq!(
        result.diagnostics.shared_robust_d_max,
        max_channel_dmax,
        epsilon = 1e-12
    );
    assert!(
        result.diagnostics.robust_d_max[0] > result.diagnostics.robust_d_max[1]
            && result.diagnostics.robust_d_max[1] > result.diagnostics.robust_d_max[2],
        "fixture should exercise uneven per-channel density ranges: {:?}",
        result.diagnostics.robust_d_max
    );
    assert!(
        result.positive_density[[0, 0, 2]] > result.positive_density[[0, 255, 2]],
        "blue channel should be inverted against shared Dmax instead of its smaller channel-local Dmax"
    );
}

#[test]
fn test_base_subtraction_in_density() {
    let base_t = [0.7, 0.4, 0.2];
    let base_d: Vec<f64> = base_t
        .iter()
        .map(|&t| scanstitch::density::transmittance_to_density(t))
        .collect();
    let pixel_t = [0.5, 0.3, 0.15];
    let pixel_d: Vec<f64> = pixel_t
        .iter()
        .map(|&t| scanstitch::density::transmittance_to_density(t))
        .collect();
    let sub: Vec<f64> = pixel_d
        .iter()
        .zip(base_d.iter())
        .map(|(d, db)| d - db)
        .collect();
    for &s in &sub {
        assert!(s >= -0.01, "negative density: {}", s);
    }
}

#[test]
fn test_density_inversion() {
    let d_values = [0.1, 0.5, 1.0, 1.5];
    let d_max = 1.8;
    let inverted: Vec<f64> = d_values.iter().map(|&d| d_max - d).collect();
    assert!(inverted[0] > inverted[3]);
    assert_relative_eq!(inverted[0], 1.7, epsilon = 0.01);
}

#[test]
fn test_full_phase3_pipeline() {
    let mut img = Array3::<u16>::zeros((100, 100, 3));
    let base_rgb = [12000u16, 7000, 3000];
    for y in 0..100 {
        for x in 0..100 {
            if x < 10 {
                img[[y, x, 0]] = base_rgb[0];
                img[[y, x, 1]] = base_rgb[1];
                img[[y, x, 2]] = base_rgb[2];
            } else {
                img[[y, x, 0]] = 6000;
                img[[y, x, 1]] = 3500;
                img[[y, x, 2]] = 1500;
            }
        }
    }
    let base_color = [base_rgb[0] as f64, base_rgb[1] as f64, base_rgb[2] as f64];
    let result = scanstitch::density::phase3_invert(&img, &base_color, 14);
    let (h, w, c) = result.dim();
    assert_eq!((h, w, c), (100, 100, 3));
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let v = result[[y, x, ch]];
                assert!(v.is_finite(), "non-finite density: {v} at ({y},{x},{ch})");
            }
        }
    }
}

#[test]
fn test_phase3_does_not_hard_clip_dense_pixels() {
    let mut img = Array3::<u16>::zeros((1, 3, 3));
    let base = [12000u16, 7000, 3000];
    for c in 0..3 {
        img[[0, 0, c]] = base[c];
        img[[0, 1, c]] = (base[c] / 2).max(1);
        img[[0, 2, c]] = 64;
    }

    let result = scanstitch::density::phase3_invert(
        &img,
        &[base[0] as f64, base[1] as f64, base[2] as f64],
        14,
    );

    assert!(
        result.iter().copied().fold(0.0, f64::max) > 1.0,
        "phase3 should preserve density headroom instead of clipping to [0,1]"
    );
}

#[test]
fn test_phase3_uses_robust_percentile_not_exact_max() {
    let mut img = Array3::<u16>::zeros((1, 1000, 3));
    let base = [12000u16, 7000, 3000];
    for x in 0..1000 {
        for c in 0..3 {
            img[[0, x, c]] = if x == 999 { 8 } else { (base[c] / 2).max(1) };
        }
    }

    let result = scanstitch::density::phase3_invert_with_diagnostics(
        &img,
        &[base[0] as f64, base[1] as f64, base[2] as f64],
        14,
    );

    for c in 0..3 {
        assert!(
            result.diagnostics.robust_d_max[c] < result.diagnostics.exact_d_max[c],
            "channel {} robust Dmax should ignore the single extreme outlier",
            c
        );
        assert!(
            result.diagnostics.highlight_headroom_samples[c] > 0,
            "channel {} should retain the outlier above the robust white anchor",
            c
        );
        assert!(
            result.positive_density[[0, 999, c]] < 0.0,
            "channel {c} highlight density should remain signed instead of being clipped"
        );
    }
}

#[test]
fn test_phase3_dmax_percentile_ignores_sparse_clipped_border_pixels() {
    let mut img = Array3::<u16>::zeros((1, 1000, 3));
    let base = [12000u16, 7000, 3000];
    for x in 0..1000 {
        for c in 0..3 {
            img[[0, x, c]] = if x >= 995 { 8 } else { (base[c] / 2).max(1) };
        }
    }

    let result = scanstitch::density::phase3_invert_with_diagnostics(
        &img,
        &[base[0] as f64, base[1] as f64, base[2] as f64],
        14,
    );

    for c in 0..3 {
        assert!(
            result.diagnostics.robust_d_max[c] < result.diagnostics.exact_d_max[c],
            "channel {c} Dmax should ignore a 0.5% clipped-border tail"
        );
    }
}

#[test]
fn test_phase3_film_base_changes_channel_balance() {
    let mut img = Array3::<u16>::zeros((1, 8, 3));
    for x in 0..8 {
        img[[0, x, 0]] = 6000 + x as u16 * 200;
        img[[0, x, 1]] = 3600 + x as u16 * 120;
        img[[0, x, 2]] = 1800 + x as u16 * 60;
    }

    let neutral_base = [9000.0, 9000.0, 9000.0];
    let orange_base = [9000.0, 4500.0, 2200.0];
    let neutral = scanstitch::density::phase3_invert(&img, &neutral_base, 14);
    let orange = scanstitch::density::phase3_invert(&img, &orange_base, 14);

    let neutral_green_red = neutral[[0, 3, 1]] / neutral[[0, 3, 0]].max(1e-9);
    let orange_green_red = orange[[0, 3, 1]] / orange[[0, 3, 0]].max(1e-9);

    assert!(
        (neutral_green_red - orange_green_red).abs() > 0.05,
        "film-base color should affect rendered channel balance"
    );
}

#[test]
fn test_scene_linear_density_conversion_preserves_highlight_headroom() {
    let mut density = Array3::<f64>::zeros((1, 3, 3));
    for c in 0..3 {
        density[[0, 0, c]] = 1.5;
        density[[0, 1, c]] = 0.0;
        density[[0, 2, c]] = -0.3;
    }

    let linear = scanstitch::density::density_image_to_normalized_transmittance(
        &density,
        scanstitch::density::DEFAULT_NEGATIVE_DENSITY_RESPONSE_SCALE,
    );

    assert_relative_eq!(linear[[0, 0, 0]], 10f64.powf(-1.5), epsilon = 1e-12);
    assert_relative_eq!(linear[[0, 1, 0]], 1.0, epsilon = 1e-12);
    assert_relative_eq!(linear[[0, 2, 0]], 10f64.powf(0.3), epsilon = 1e-12);
    assert!(linear[[0, 2, 0]] > 1.0, "highlight headroom must survive");
}

#[test]
fn test_shared_density_anchor_preserves_recorded_channel_relationships() {
    let base = [12_000u16, 7_000, 3_000];
    let mut img = Array3::<u16>::zeros((1, 256, 3));
    for x in 0..256 {
        let scene = x as f64 / 255.0;
        let channel_density = [scene, scene * 0.75, scene * 0.5];
        for c in 0..3 {
            img[[0, x, c]] = (base[c] as f64 * 10f64.powf(-channel_density[c]))
                .round()
                .max(1.0) as u16;
        }
    }

    let inverted = scanstitch::density::phase3_invert(
        &img,
        &[base[0] as f64, base[1] as f64, base[2] as f64],
        14,
    );
    let linear = scanstitch::density::density_image_to_normalized_transmittance(
        &inverted,
        scanstitch::density::DEFAULT_NEGATIVE_DENSITY_RESPONSE_SCALE,
    );

    assert!(
        linear[[0, 180, 0]] > linear[[0, 180, 1]]
            && linear[[0, 180, 1]] > linear[[0, 180, 2]],
        "an uncalibrated density conversion must preserve recorded channel evidence instead of independently auto-leveling it"
    );
}

#[test]
fn test_unit_response_reconstruction_matches_existing_direct_density_path() {
    let base = [50_000u16, 40_000, 30_000];
    let mut img = Array3::<u16>::zeros((40, 40, 3));
    for y in 0..40 {
        for x in 0..40 {
            let density = (x + y) as f64 / 78.0;
            for channel in 0..3 {
                img[[y, x, channel]] = (base[channel] as f64 * 10f64.powf(-density))
                    .round()
                    .max(1.0) as u16;
            }
        }
    }
    let phase3 = scanstitch::density::phase3_invert_with_diagnostics(
        &img,
        &[base[0] as f64, base[1] as f64, base[2] as f64],
        16,
    );
    let legacy = scanstitch::density::density_image_to_normalized_transmittance(
        &phase3.positive_density,
        1.0,
    );
    let reconstructed = scanstitch::density::reconstruct_with_negative_response(
        &phase3.positive_density,
        &phase3.diagnostics,
        &scanstitch::density::unit_negative_response_model(),
    );

    for (actual, expected) in reconstructed.scene_linear.iter().zip(legacy.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1e-12);
    }
}

#[test]
fn test_regularized_frame_response_recovers_dominant_layer_slope_direction() {
    let base = [50_000u16, 40_000, 30_000];
    let true_slopes = [1.35, 0.90, 0.72];
    let mut img = Array3::<u16>::zeros((64, 64, 3));
    for y in 0..64 {
        for x in 0..64 {
            let exposure = 0.05 + 1.10 * (x + y) as f64 / 126.0;
            let chroma = 0.025 * ((x as f64 * 0.31).sin() + (y as f64 * 0.17).cos());
            for channel in 0..3 {
                let signed_chroma = match channel {
                    0 => chroma,
                    1 => -0.4 * chroma,
                    _ => -0.6 * chroma,
                };
                let density = (true_slopes[channel] * exposure + signed_chroma).max(0.0);
                img[[y, x, channel]] = (base[channel] as f64 * 10f64.powf(-density))
                    .round()
                    .max(1.0) as u16;
            }
        }
    }
    let phase3 = scanstitch::density::phase3_invert_with_diagnostics(
        &img,
        &[base[0] as f64, base[1] as f64, base[2] as f64],
        16,
    );
    let model = scanstitch::density::estimate_regularized_frame_response(
        &phase3.positive_density,
        phase3.diagnostics.shared_robust_d_max,
    );

    assert!(model.diagnostics.accepted, "{}", model.diagnostics.reason);
    assert!(model.diagnostics.explained_variance_ratio.unwrap() > 0.95);
    let raw = model.diagnostics.raw_density_slopes;
    assert!(raw[0] > raw[1] && raw[1] > raw[2]);
    assert_relative_eq!(
        raw[0] / raw[1],
        true_slopes[0] / true_slopes[1],
        epsilon = 0.12
    );
    assert_relative_eq!(
        raw[1] / raw[2],
        true_slopes[1] / true_slopes[2],
        epsilon = 0.12
    );
    let applied = model.diagnostics.applied_density_slopes;
    assert!(applied[0] > 1.0 && applied[2] < 1.0);
    assert!(
        model.diagnostics.applied_slope_condition_number
            < model.diagnostics.raw_slope_condition_number,
        "frame estimate must be shrunk toward unit response"
    );

    let baseline = scanstitch::density::reconstruct_with_negative_response(
        &phase3.positive_density,
        &phase3.diagnostics,
        &scanstitch::density::unit_negative_response_model(),
    );
    let corrected = scanstitch::density::reconstruct_with_negative_response(
        &phase3.positive_density,
        &phase3.diagnostics,
        &model,
    );
    let location = [32, 32];
    let baseline_log_spread = (0..3)
        .map(|channel| -baseline.scene_linear[[location[0], location[1], channel]].log10())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
            (low.min(value), high.max(value))
        });
    let corrected_log_spread = (0..3)
        .map(|channel| -corrected.scene_linear[[location[0], location[1], channel]].log10())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
            (low.min(value), high.max(value))
        });
    assert!(
        corrected_log_spread.1 - corrected_log_spread.0
            < baseline_log_spread.1 - baseline_log_spread.0,
        "regularized layer slopes should reduce a known neutral-axis imbalance"
    );
}

#[test]
fn test_frame_response_rejects_opposed_uncorrelated_channel_variation() {
    let mut positive_density = Array3::<f64>::zeros((40, 40, 3));
    for y in 0..40 {
        for x in 0..40 {
            let a = x as f64 / 39.0;
            let b = y as f64 / 39.0;
            positive_density[[y, x, 0]] = 1.0 - a;
            positive_density[[y, x, 1]] = a;
            positive_density[[y, x, 2]] = 1.0 - b;
        }
    }

    let model = scanstitch::density::estimate_regularized_frame_response(&positive_density, 1.0);
    assert!(!model.diagnostics.accepted);
    assert_eq!(model.diagnostics.applied_density_slopes, [1.0; 3]);
}

#[test]
fn test_measured_response_rejects_unvalidated_or_nonmonotone_model() {
    let mut calibration = measured_response_fixture();
    calibration.validation.held_out_patch_count = 3;
    calibration.validation.delta_e00_rms = 8.0;
    calibration.characteristic_curves.red[2][1] = 0.10;

    let errors = scanstitch::density::validate_measured_negative_response(&calibration)
        .expect_err("unsafe measured response must be rejected");
    assert!(errors
        .iter()
        .any(|error| error.contains("held-out patches")));
    assert!(errors
        .iter()
        .any(|error| error.contains("strictly increasing")));
    assert!(errors.iter().any(|error| error.contains("DeltaE00 RMS")));
    assert!(scanstitch::density::measured_negative_response_model(&calibration).is_err());
}

#[test]
fn test_measured_crosstalk_and_nonlinear_curves_recover_neutral_scene_axis() {
    let calibration = measured_response_fixture();
    let model = scanstitch::density::measured_negative_response_model(&calibration)
        .expect("held-out validated measured response");
    assert!(!model.diagnostics.review_required);
    assert_eq!(
        model.diagnostics.characteristic_curve_model,
        "measured_monotone_pchip_density_to_scene_log_exposure"
    );

    let density_to_layer =
        nalgebra::Matrix3::new(1.0, -0.08, 0.01, -0.04, 1.0, -0.05, 0.01, -0.04, 1.0);
    let layer_to_density = density_to_layer.try_inverse().unwrap();
    let base = [60_000u16, 55_000, 50_000];
    let layer_knots = [
        [0.0, 0.0, 0.0],
        [0.25, 0.30, 0.20],
        [0.70, 0.85, 0.62],
        [1.30, 1.45, 1.18],
    ];
    let mut img = Array3::<u16>::zeros((64, 64, 3));
    for y in 0..64 {
        for x in 0..64 {
            let knot = ((x + y) / 32).min(3);
            let layer = nalgebra::Vector3::new(
                layer_knots[knot][0],
                layer_knots[knot][1],
                layer_knots[knot][2],
            );
            let scanner_density = layer_to_density * layer;
            for channel in 0..3 {
                img[[y, x, channel]] = (base[channel] as f64
                    * 10f64.powf(-scanner_density[channel]))
                .round()
                .max(1.0) as u16;
            }
        }
    }

    let phase3 = scanstitch::density::phase3_invert_with_diagnostics(
        &img,
        &[base[0] as f64, base[1] as f64, base[2] as f64],
        16,
    );
    let measured = scanstitch::density::reconstruct_with_negative_response(
        &phase3.positive_density,
        &phase3.diagnostics,
        &model,
    );
    let unit = scanstitch::density::reconstruct_with_negative_response(
        &phase3.positive_density,
        &phase3.diagnostics,
        &scanstitch::density::unit_negative_response_model(),
    );

    let (y, x) = (24, 24);
    let measured_values = [
        measured.scene_linear[[y, x, 0]],
        measured.scene_linear[[y, x, 1]],
        measured.scene_linear[[y, x, 2]],
    ];
    let unit_values = [
        unit.scene_linear[[y, x, 0]],
        unit.scene_linear[[y, x, 1]],
        unit.scene_linear[[y, x, 2]],
    ];
    let measured_ratio = measured_values.iter().copied().fold(0.0, f64::max)
        / measured_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
    let unit_ratio = unit_values.iter().copied().fold(0.0, f64::max)
        / unit_values.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        measured_ratio < 1.01,
        "measured neutral ratio {measured_ratio}"
    );
    assert!(
        measured_ratio - 1.0 < (unit_ratio - 1.0) * 0.1,
        "measured response should materially outperform unit slopes ({measured_ratio} vs {unit_ratio})"
    );
    assert_eq!(
        measured.diagnostics.curve_interpolation,
        "monotone_piecewise_cubic_hermite_with_endpoint_tangent_extrapolation"
    );
}

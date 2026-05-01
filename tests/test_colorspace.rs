use approx::assert_relative_eq;
use nalgebra::Vector3;

fn diagnostics_with_image_matrix_low_clip(
    low_clip: [f64; 3],
    gamut_fallback_used: bool,
) -> scanstitch::colorspace::ColorspaceDiagnostics {
    scanstitch::colorspace::ColorspaceDiagnostics {
        source_white: [1.0, 1.0, 1.0],
        work_to_xyz: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        condition_number: 1.0,
        regularization_lambda: 0.05,
        neutral_pixel_count: 1024,
        channel_anchor_counts: [256, 256, 256],
        channel_anchor_min_count: 256,
        channel_anchor_low_support_threshold: 64,
        channel_anchor_low_support: [false, false, false],
        dominant_anchor_rgb: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        highlight_percentile: 0.995,
        pre_scale_channel_max: [1.0, 1.0, 1.0],
        pre_scale_channel_high_percentile: [1.0, 1.0, 1.0],
        pre_scale_clipped_high_ratio: [0.0, 0.0, 0.0],
        pre_scale_clipped_low_ratio: [0.0, 0.0, 0.0],
        post_scale_clipped_high_ratio: [0.0, 0.0, 0.0],
        post_scale_clipped_low_ratio: [0.0, 0.0, 0.0],
        exposure_scale: 1.0,
        fallback_used: false,
        weak_anchor_fallback_used: false,
        weak_anchor_fallback_reason: None,
        gamut_fallback_used,
        gamut_fallback_reason: if gamut_fallback_used {
            Some("synthetic destructive gamut fallback".to_string())
        } else {
            None
        },
        mapping_strategy: if gamut_fallback_used {
            "neutral_balance_gamut_fallback"
        } else {
            "image_derived_matrix"
        },
        neutral_balance_scale: [1.0, 1.0, 1.0],
        image_matrix_pre_scale_clipped_low_ratio: low_clip,
        image_matrix_pre_scale_clipped_high_ratio: [0.0, 0.0, 0.0],
        image_matrix_exposure_scale: 1.0,
    }
}

#[test]
fn test_bradford_adaptation_d65_to_d50() {
    let source_white = [0.9505, 1.0000, 1.0890];
    let cat = scanstitch::colorspace::bradford_cat(&source_white);
    let xyz_d65 = Vector3::new(0.9505, 1.0, 1.0890);
    let xyz_d50 = cat * xyz_d65;
    assert_relative_eq!(xyz_d50[0], 0.9642, epsilon = 0.01);
    assert_relative_eq!(xyz_d50[1], 1.0, epsilon = 0.01);
    assert_relative_eq!(xyz_d50[2], 0.8251, epsilon = 0.01);
}

#[test]
fn test_direct_density_render_fallback_for_destructive_ica_gamut() {
    let ica = diagnostics_with_image_matrix_low_clip([0.43, 0.16, 0.13], true);
    let direct = diagnostics_with_image_matrix_low_clip([0.0, 0.0, 0.001], false);

    let reason = scanstitch::colorspace::direct_density_render_fallback_reason(&ica, &direct);

    assert!(
        reason.is_some(),
        "expected direct density render fallback for destructive ICA colorspace diagnostics"
    );
    assert!(
        reason.unwrap().contains("direct density transmittance"),
        "fallback reason should identify the safer render input"
    );
}

#[test]
fn test_direct_density_render_fallback_requires_material_improvement() {
    let ica = diagnostics_with_image_matrix_low_clip([0.20, 0.08, 0.04], true);
    let direct = diagnostics_with_image_matrix_low_clip([0.18, 0.08, 0.03], true);

    assert!(
        scanstitch::colorspace::direct_density_render_fallback_reason(&ica, &direct).is_none(),
        "direct density candidate should not replace ICA when low clipping is not materially better"
    );
}

#[test]
fn test_prophoto_roundtrip() {
    let xyz_to_pro = scanstitch::colorspace::xyz_d50_to_prophoto_matrix();
    let pro_to_xyz = scanstitch::colorspace::prophoto_to_xyz_d50_matrix();
    let xyz = Vector3::new(0.5, 0.4, 0.3);
    let pro = xyz_to_pro * xyz;
    let xyz2 = pro_to_xyz * pro;
    for i in 0..3 {
        assert_relative_eq!(xyz[i], xyz2[i], epsilon = 1e-4);
    }
}

#[test]
fn test_output_clamped_0_1() {
    let mut img = ndarray::Array3::<f64>::zeros((10, 10, 3));
    for y in 0..10 {
        for x in 0..10 {
            img[[y, x, 0]] = 0.8;
            img[[y, x, 1]] = 0.5;
            img[[y, x, 2]] = 0.3;
        }
    }
    let result = scanstitch::colorspace::map_to_prophoto_d50(&img);
    for y in 0..10 {
        for x in 0..10 {
            for c in 0..3 {
                let v = result[[y, x, c]];
                assert!(
                    v >= 0.0 && v <= 1.0,
                    "out of range: {} at ({},{},{})",
                    v,
                    y,
                    x,
                    c
                );
            }
        }
    }
}

#[test]
fn test_estimated_work_to_xyz_is_not_hard_coded_prophoto() {
    let mut img = ndarray::Array3::<f64>::zeros((20, 20, 3));
    for y in 0..20 {
        for x in 0..20 {
            if x < 6 {
                img[[y, x, 0]] = 0.90;
                img[[y, x, 1]] = 0.22;
                img[[y, x, 2]] = 0.12;
            } else if x < 12 {
                img[[y, x, 0]] = 0.18;
                img[[y, x, 1]] = 0.82;
                img[[y, x, 2]] = 0.16;
            } else if x < 18 {
                img[[y, x, 0]] = 0.12;
                img[[y, x, 1]] = 0.20;
                img[[y, x, 2]] = 0.78;
            } else {
                img[[y, x, 0]] = 0.58;
                img[[y, x, 1]] = 0.55;
                img[[y, x, 2]] = 0.52;
            }
        }
    }

    let diagnostics = scanstitch::colorspace::estimate_work_to_xyz(&img);
    let prior = scanstitch::colorspace::prophoto_to_xyz_d50_matrix();
    let prior_rows = [
        [prior[(0, 0)], prior[(0, 1)], prior[(0, 2)]],
        [prior[(1, 0)], prior[(1, 1)], prior[(1, 2)]],
        [prior[(2, 0)], prior[(2, 1)], prior[(2, 2)]],
    ];

    let mut max_delta = 0.0f64;
    for r in 0..3 {
        for c in 0..3 {
            max_delta = max_delta.max((diagnostics.work_to_xyz[r][c] - prior_rows[r][c]).abs());
        }
    }

    assert!(diagnostics.neutral_pixel_count > 0);
    assert!(
        max_delta > 1e-3,
        "estimated matrix should differ from the fixed ProPhoto prior"
    );
}

#[test]
fn test_colorspace_reports_weak_channel_anchor_support() {
    let mut img = ndarray::Array3::<f64>::zeros((20, 20, 3));
    for y in 0..20 {
        for x in 0..20 {
            let pixel = if x < 8 {
                [0.90, 0.22, 0.12]
            } else if x < 16 {
                [0.12, 0.20, 0.78]
            } else {
                [0.58, 0.56, 0.54]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }

    let diagnostics = scanstitch::colorspace::estimate_work_to_xyz(&img);
    let warning = scanstitch::colorspace::weak_channel_anchor_warning(&diagnostics)
        .expect("expected weak anchor warning for missing green support");

    assert_eq!(diagnostics.channel_anchor_counts[1], 0);
    assert_eq!(diagnostics.channel_anchor_min_count, 0);
    assert_eq!(diagnostics.channel_anchor_low_support_threshold, 64);
    assert_eq!(diagnostics.channel_anchor_low_support, [false, true, false]);
    assert_eq!(diagnostics.regularization_lambda, 0.20);
    assert!(
        warning.contains("green=0"),
        "warning should name the weak channel and count: {}",
        warning
    );
}

#[test]
fn test_colorspace_gates_weak_single_channel_anchor_support() {
    let mut img = ndarray::Array3::<f64>::zeros((20, 20, 3));
    for y in 0..20 {
        for x in 0..20 {
            let pixel = if x < 8 {
                [0.90, 0.22, 0.12]
            } else if x < 16 {
                [0.12, 0.20, 0.78]
            } else {
                [0.58, 0.56, 0.54]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }

    let result = scanstitch::colorspace::map_to_prophoto_d50_with_diagnostics(&img);
    let diagnostics = result.diagnostics;

    assert_eq!(diagnostics.channel_anchor_low_support, [false, true, false]);
    assert_eq!(diagnostics.regularization_lambda, 0.20);
    assert!(
        diagnostics.weak_anchor_fallback_used,
        "weak anchors should gate matrix-driven rendered output: {:?}",
        diagnostics
    );
    assert_eq!(
        diagnostics.mapping_strategy,
        "neutral_balance_weak_anchor_fallback"
    );
    assert!(
        diagnostics
            .weak_anchor_fallback_reason
            .as_deref()
            .unwrap_or_default()
            .contains("green=0"),
        "fallback reason should name the weak channel: {:?}",
        diagnostics.weak_anchor_fallback_reason
    );
}

#[test]
fn test_colorspace_reports_strong_channel_anchors_without_weak_warning() {
    let mut img = ndarray::Array3::<f64>::zeros((24, 24, 3));
    for y in 0..24 {
        for x in 0..24 {
            let pixel = if x < 8 {
                [0.90, 0.22, 0.12]
            } else if x < 16 {
                [0.18, 0.82, 0.16]
            } else {
                [0.12, 0.20, 0.78]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }

    let result = scanstitch::colorspace::map_to_prophoto_d50_with_diagnostics(&img);
    let diagnostics = result.diagnostics;

    assert_eq!(diagnostics.channel_anchor_counts, [192, 192, 192]);
    assert_eq!(
        diagnostics.channel_anchor_low_support,
        [false, false, false]
    );
    assert_eq!(diagnostics.regularization_lambda, 0.05);
    assert!(
        !diagnostics.weak_anchor_fallback_used,
        "strong anchors should not trigger weak-anchor fallback: {:?}",
        diagnostics
    );
    assert_ne!(
        diagnostics.mapping_strategy,
        "neutral_balance_weak_anchor_fallback"
    );
    assert!(
        scanstitch::colorspace::weak_channel_anchor_warning(&diagnostics).is_none(),
        "strong anchors should not emit a weak-anchor warning"
    );
}

#[test]
fn test_colorspace_applies_headroom_normalization_before_clamp() {
    let mut img = ndarray::Array3::<f64>::zeros((40, 40, 3));
    for y in 0..40 {
        for x in 0..40 {
            let value = 0.80 + (x as f64 / 39.0) * 0.85 + (y as f64 / 39.0) * 0.15;
            let pixel = [value, value * 0.99, value * 1.01];
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }

    let result = scanstitch::colorspace::map_to_prophoto_d50_with_diagnostics(&img);
    let diagnostics = result.diagnostics;
    let before_clip: f64 = diagnostics.pre_scale_clipped_high_ratio.iter().sum();
    let after_clip: f64 = diagnostics.post_scale_clipped_high_ratio.iter().sum();
    let pre_scale_high = diagnostics
        .pre_scale_channel_high_percentile
        .iter()
        .copied()
        .fold(0.0f64, f64::max);

    assert!(
        diagnostics.exposure_scale > 1.0,
        "expected highlight headroom normalization to engage: {:?}",
        diagnostics
    );
    assert!(
        !diagnostics.gamut_fallback_used,
        "neutral headroom fixture should not need the negative-gamut fallback: {:?}",
        diagnostics
    );
    assert!(
        pre_scale_high > 1.0,
        "expected mapped ProPhoto highlights to overshoot before normalization: {:?}",
        diagnostics.pre_scale_channel_high_percentile
    );
    assert!(
        after_clip < before_clip,
        "expected post-scale clipping to decrease (before {}, after {})",
        before_clip,
        after_clip
    );

    for value in result.prophoto.iter() {
        assert!(
            (0.0..=1.0).contains(value),
            "mapped value out of range: {}",
            value
        );
    }
}

#[test]
fn test_colorspace_uses_gamut_fallback_for_destructive_negative_matrix_clip() {
    let mut img = ndarray::Array3::<f64>::zeros((40, 40, 3));
    for y in 0..40 {
        for x in 0..40 {
            let pixel = if x < 10 {
                [0.73, 0.41, 0.34]
            } else if x < 20 {
                [0.34, 0.63, 0.32]
            } else if x < 30 {
                [0.35, 0.38, 0.62]
            } else {
                [0.44, 0.45, 0.49]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }

    let result = scanstitch::colorspace::map_to_prophoto_d50_with_diagnostics(&img);
    let diagnostics = result.diagnostics;
    let image_matrix_low_clip: f64 = diagnostics
        .image_matrix_pre_scale_clipped_low_ratio
        .iter()
        .sum();
    let chosen_low_clip: f64 = diagnostics.pre_scale_clipped_low_ratio.iter().sum();
    let output_low_clip: f64 = diagnostics.post_scale_clipped_low_ratio.iter().sum();

    assert!(
        diagnostics.gamut_fallback_used,
        "expected a gamut fallback when the image-derived matrix clips negative values: {:?}",
        diagnostics
    );
    assert_eq!(
        diagnostics.mapping_strategy,
        "neutral_balance_gamut_fallback"
    );
    assert!(
        image_matrix_low_clip > 0.18,
        "expected rejected image matrix to have material negative clipping: {:?}",
        diagnostics.image_matrix_pre_scale_clipped_low_ratio
    );
    assert!(
        chosen_low_clip < 1e-9,
        "fallback should avoid pre-scale negative clipping: {:?}",
        diagnostics.pre_scale_clipped_low_ratio
    );
    assert!(
        output_low_clip < 1e-9,
        "fallback should avoid post-scale negative clipping: {:?}",
        diagnostics.post_scale_clipped_low_ratio
    );

    for value in result.prophoto.iter() {
        assert!(
            (0.0..=1.0).contains(value),
            "mapped value out of range: {}",
            value
        );
    }
}

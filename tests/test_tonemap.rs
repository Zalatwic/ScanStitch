use ndarray::Array3;

#[test]
fn test_sigmoid_has_toe_and_shoulder() {
    let params = scanstitch::tonemap::ToneCurveParams::default();
    let shadow = scanstitch::tonemap::apply_tone_curve(0.01, &params);
    assert!(shadow > 0.001, "toe: {}", shadow);
    let mid = scanstitch::tonemap::apply_tone_curve(0.5, &params);
    assert!(mid > 0.3 && mid < 0.7, "mid: {}", mid);
    let highlight = scanstitch::tonemap::apply_tone_curve(0.99, &params);
    assert!(highlight < params.shoulder_max, "shoulder: {}", highlight);
    assert!(highlight > 0.8, "crush: {}", highlight);
}

#[test]
fn test_tonemap_monotonic() {
    let params = scanstitch::tonemap::ToneCurveParams::default();
    let mut prev = 0.0;
    for i in 0..=1000 {
        let x = i as f64 / 1000.0;
        let y = scanstitch::tonemap::apply_tone_curve(x, &params);
        assert!(y >= prev, "non-monotonic at {}: y={}, prev={}", x, y, prev);
        prev = y;
    }
}

#[test]
fn test_tone_curve_preserves_extended_highlight_separation() {
    let params = scanstitch::tonemap::ToneCurveParams::default();
    let display_white = scanstitch::tonemap::apply_tone_curve(1.0, &params);
    let specular = scanstitch::tonemap::apply_tone_curve(2.0, &params);

    assert!(
        display_white < params.shoulder_max,
        "display white should leave shoulder room for scene-referred highlights: {}",
        display_white
    );
    assert!(
        specular > display_white,
        "extended highlights should remain distinguishable after tone mapping"
    );
    assert!(
        specular <= params.shoulder_max,
        "extended highlight should still roll off inside the display shoulder"
    );
}

#[test]
fn test_tonemap_output_range() {
    let mut img = Array3::<f64>::zeros((50, 50, 3));
    for y in 0..50 {
        for x in 0..50 {
            let t = (y * 50 + x) as f64 / 2500.0;
            img[[y, x, 0]] = t;
            img[[y, x, 1]] = t * 0.8;
            img[[y, x, 2]] = t * 0.6;
        }
    }
    let result = scanstitch::tonemap::apply_tonemap(&img);
    for y in 0..50 {
        for x in 0..50 {
            for c in 0..3 {
                let v = result[[y, x, c]];
                assert!(
                    (0.0..=1.0).contains(&v),
                    "out of range at ({},{},{}): {}",
                    y,
                    x,
                    c,
                    v
                );
            }
        }
    }
}

#[test]
fn test_tonemap_reports_local_luminance_detail_pass() {
    let mut img = Array3::<f64>::from_elem((25, 25, 3), 0.35);
    for y in 8..17 {
        for x in 8..17 {
            for c in 0..3 {
                img[[y, x, c]] = 0.58;
            }
        }
    }
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.40,
        slope: 3.0,
        toe_lift: 0.0,
        shoulder_max: 1.0,
    };

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);

    assert!(result.diagnostics.local_luminance_detail_enabled);
    assert!(result.diagnostics.local_luminance_detail_applied_ratio > 0.0);
    assert!(result.diagnostics.local_luminance_detail_mean_abs_ev > 0.0);
    assert!(result.diagnostics.local_luminance_detail_max_abs_ev <= 0.22 + 1e-12);
    assert!(
        result
            .diagnostics
            .local_luminance_detail_headroom_limited_ratio
            <= 1.0
    );
    assert!(result.image.iter().all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn test_tonemap_adaptive_vibrance_enriches_trusted_midtones_without_luma_shift() {
    let img = Array3::<f64>::from_shape_fn((25, 25, 3), |(_, _, c)| match c {
        0 => 0.42,
        1 => 0.34,
        _ => 0.28,
    });
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.40,
        slope: 2.2,
        toe_lift: 0.0,
        shoulder_max: 1.0,
    };
    let review_protection = scanstitch::tonemap::ToneColorProtection {
        policy: scanstitch::tonemap::ToneColorProtectionPolicy::DisabledColorCandidateReview,
        highlight_neutral_chroma_enabled: false,
        midtone_neutral_chroma_enabled: false,
        shadow_chroma_enabled: false,
        reason: "synthetic review color".to_string(),
    };

    let trusted = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);
    let review = scanstitch::tonemap::apply_tonemap_with_params_and_color_protection_diagnostics(
        &img,
        &params,
        &review_protection,
    );

    let trusted_rgb = [
        trusted.image[[12, 12, 0]],
        trusted.image[[12, 12, 1]],
        trusted.image[[12, 12, 2]],
    ];
    let review_rgb = [
        review.image[[12, 12, 0]],
        review.image[[12, 12, 1]],
        review.image[[12, 12, 2]],
    ];
    let trusted_lum = 0.2880 * trusted_rgb[0] + 0.7119 * trusted_rgb[1] + 0.0001 * trusted_rgb[2];
    let review_lum = 0.2880 * review_rgb[0] + 0.7119 * review_rgb[1] + 0.0001 * review_rgb[2];
    let trusted_sat = trusted_rgb
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - trusted_rgb.iter().copied().fold(f64::INFINITY, f64::min);
    let review_sat = review_rgb.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - review_rgb.iter().copied().fold(f64::INFINITY, f64::min);

    assert!(trusted.diagnostics.adaptive_vibrance_enabled);
    assert!(trusted.diagnostics.adaptive_vibrance_applied_ratio > 0.0);
    assert!(trusted.diagnostics.adaptive_vibrance_texture_limited_ratio < 0.05);
    assert!(!review.diagnostics.adaptive_vibrance_enabled);
    assert!(
        trusted_sat > review_sat,
        "trusted vibrance should increase midtone saturation: trusted={}, review={}",
        trusted_sat,
        review_sat
    );
    assert!(
        (trusted_lum - review_lum).abs() < 2e-6,
        "adaptive vibrance should preserve weighted luminance within final denoise precision: trusted={}, review={}",
        trusted_lum,
        review_lum
    );
    assert!(trusted
        .image
        .iter()
        .all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn test_tonemap_adaptive_vibrance_is_limited_on_textured_grain() {
    let mut img = Array3::<f64>::zeros((31, 31, 3));
    for y in 0..31 {
        for x in 0..31 {
            let sign: f64 = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
            img[[y, x, 0]] = (0.42f64 + 0.12 * sign).clamp(0.0, 1.0);
            img[[y, x, 1]] = (0.34f64 + 0.12 * sign).clamp(0.0, 1.0);
            img[[y, x, 2]] = (0.28f64 + 0.12 * sign).clamp(0.0, 1.0);
        }
    }
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.40,
        slope: 2.2,
        toe_lift: 0.0,
        shoulder_max: 1.0,
    };

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);

    assert!(result.diagnostics.adaptive_vibrance_enabled);
    assert!(
        result.diagnostics.adaptive_vibrance_texture_limited_ratio > 0.70,
        "high-frequency texture should gate adaptive vibrance: {:?}",
        result.diagnostics.adaptive_vibrance_texture_limited_ratio
    );
    assert!(
        result.diagnostics.adaptive_vibrance_mean_scale < 1.04,
        "texture-limited adaptive vibrance should not enrich grain strongly: {:?}",
        result.diagnostics.adaptive_vibrance_mean_scale
    );
}

#[test]
fn test_tonemap_reduces_flat_area_chroma_noise_with_diagnostics() {
    let mut img = Array3::<f64>::from_elem((31, 31, 3), 0.38);
    for y in 0..31 {
        for x in 0..31 {
            let sign: f64 = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
            img[[y, x, 0]] = (0.38f64 + 0.08 * sign).clamp(0.0, 1.0);
            img[[y, x, 1]] = 0.38;
            img[[y, x, 2]] = (0.38f64 - 0.08 * sign).clamp(0.0, 1.0);
        }
    }
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.40,
        slope: 2.2,
        toe_lift: 0.0,
        shoulder_max: 1.0,
    };

    let before = scanstitch::tonemap::render_grain_diagnostics(&img);
    let result = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);
    let after = scanstitch::tonemap::render_grain_diagnostics(&result.image);

    assert!(
        before.flat_sample_ratio > 0.80,
        "fixture should expose a flat-area chroma residual population: {:?}",
        before
    );
    assert!(
        before.flat_chroma_residual_p95 > 0.0,
        "flat-area chroma residual should be measured separately: {:?}",
        before
    );
    assert!(result.diagnostics.noise_reduction_enabled);
    assert!(result.diagnostics.noise_reduction_applied_ratio > 0.80);
    assert!(result.diagnostics.noise_reduction_mean_abs_chroma_delta > 0.0);
    assert!(result.diagnostics.noise_reduction_mean_abs_luma_delta >= 0.0);
    assert!(
        after.flat_chroma_residual_p95 < before.flat_chroma_residual_p95,
        "expected flat-area chroma residual to drop after final render denoise: before={:?} after={:?}",
        before,
        after
    );
    assert!(
        after.chroma_to_luma_p95_ratio < before.chroma_to_luma_p95_ratio,
        "expected chroma residual ratio to drop after final render denoise: before={:?} after={:?}",
        before,
        after
    );
    assert!(result.image.iter().all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn test_tonemap_denoises_flat_chroma_when_luma_grain_is_present() {
    let mut img = Array3::<f64>::zeros((31, 31, 3));
    for y in 0..31 {
        for x in 0..31 {
            let sign: f64 = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
            let luma = 0.38f64 + 0.07 * sign;
            img[[y, x, 0]] = (luma + 0.055 * sign).clamp(0.0, 1.0);
            img[[y, x, 1]] = luma.clamp(0.0, 1.0);
            img[[y, x, 2]] = (luma - 0.055 * sign).clamp(0.0, 1.0);
        }
    }
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.40,
        slope: 2.2,
        toe_lift: 0.0,
        shoulder_max: 1.0,
    };

    let before = scanstitch::tonemap::render_grain_diagnostics(&img);
    let result = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);
    let after = scanstitch::tonemap::render_grain_diagnostics(&result.image);

    assert!(
        before.flat_sample_ratio > 0.80,
        "fixture should be locally flat despite high-frequency luma grain: {:?}",
        before
    );
    assert!(
        before.flat_luma_residual_p95 > 0.05,
        "fixture should exercise luma grain in the flat subset: {:?}",
        before
    );
    assert!(
        after.flat_chroma_residual_p95 < before.flat_chroma_residual_p95 * 0.90,
        "flat chroma residual should drop even when luma grain is present: before={:?} after={:?}",
        before,
        after
    );
    assert!(
        result.diagnostics.noise_reduction_mean_abs_chroma_delta > 0.0,
        "chroma denoise should remain active in luma-grain flat areas: {:?}",
        result.diagnostics
    );
    assert!(result.image.iter().all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn test_render_grain_diagnostics_excludes_structured_luma_texture_from_flat_subset() {
    let mut img = Array3::<f64>::zeros((31, 31, 3));
    for y in 0..31 {
        for x in 0..31 {
            let lum = if (x / 2 + y / 2) % 2 == 0 { 0.18 } else { 0.72 };
            for c in 0..3 {
                img[[y, x, c]] = lum;
            }
        }
    }

    let diagnostics = scanstitch::tonemap::render_grain_diagnostics(&img);

    assert!(
        diagnostics.luma_residual_p95 > 0.10,
        "structured luma texture should remain visible in all-sample residuals: {:?}",
        diagnostics
    );
    assert!(
        diagnostics.flat_sample_ratio < 0.20,
        "structured luma texture should not be classified as flat-field grain: {:?}",
        diagnostics
    );
}

#[test]
fn test_tonemap_noise_reduction_limits_textured_and_saturated_detail() {
    let mut img = Array3::<f64>::from_elem((41, 41, 3), 0.36);
    for y in 0..41 {
        for x in 0..41 {
            let sign: f64 = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
            if x < 14 {
                img[[y, x, 0]] = 0.36 + 0.05 * sign;
                img[[y, x, 1]] = 0.36;
                img[[y, x, 2]] = 0.36 - 0.05 * sign;
            } else if x < 28 {
                let lum = if (x + y) % 2 == 0 { 0.20 } else { 0.62 };
                for c in 0..3 {
                    img[[y, x, c]] = lum;
                }
            } else {
                img[[y, x, 0]] = 0.70 + 0.04 * sign;
                img[[y, x, 1]] = 0.20;
                img[[y, x, 2]] = 0.12;
            }
        }
    }
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.40,
        slope: 2.2,
        toe_lift: 0.0,
        shoulder_max: 1.0,
    };

    let before = scanstitch::tonemap::render_grain_diagnostics(&img);
    let result = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);
    let after = scanstitch::tonemap::render_grain_diagnostics(&result.image);

    assert!(result.diagnostics.noise_reduction_enabled);
    assert!(
        result.diagnostics.noise_reduction_texture_limited_ratio > 0.14,
        "expected textured detail to limit denoise application: {:?}",
        result.diagnostics.noise_reduction_texture_limited_ratio
    );
    assert!(
        result.diagnostics.noise_reduction_saturation_limited_ratio > 0.20,
        "expected saturated detail to limit chroma denoise application: {:?}",
        result.diagnostics.noise_reduction_saturation_limited_ratio
    );
    assert!(
        result.diagnostics.noise_reduction_mean_abs_chroma_delta
            > result.diagnostics.noise_reduction_mean_abs_luma_delta,
        "final cleanup should prefer chroma residual smoothing over luma smoothing"
    );
    assert!(
        result.diagnostics.noise_reduction_max_abs_luma_delta < 0.05,
        "luma smoothing should stay bounded in mixed texture/detail scenes"
    );
    assert!(
        after.chroma_to_luma_p95_ratio < before.chroma_to_luma_p95_ratio,
        "chroma residual ratio should still improve despite detail limiting"
    );
}

#[test]
fn test_scene_referred_detail_fusion_preserves_rgb_ratios() {
    let mut base = Array3::<f64>::zeros((25, 25, 3));
    let mut guide = Array3::<f64>::zeros((25, 25, 3));
    for y in 0..25 {
        for x in 0..25 {
            base[[y, x, 0]] = 0.40;
            base[[y, x, 1]] = 0.24;
            base[[y, x, 2]] = 0.12;
            guide[[y, x, 0]] = 0.40;
            guide[[y, x, 1]] = 0.24;
            guide[[y, x, 2]] = 0.12;
        }
    }
    for y in 9..16 {
        for x in 9..16 {
            guide[[y, x, 0]] *= 1.35;
            guide[[y, x, 1]] *= 1.35;
            guide[[y, x, 2]] *= 1.35;
        }
    }

    let result = scanstitch::tonemap::fuse_scene_referred_luminance_detail(&base, &guide);

    assert!(result.diagnostics.enabled);
    assert!(result.diagnostics.applied_ratio > 0.0);
    assert!(result.diagnostics.mean_abs_ev > 0.0);
    assert!(result.diagnostics.max_abs_ev <= 0.16 + 1e-12);
    assert!(
        result.image[[12, 12, 0]] > base[[12, 12, 0]],
        "guide-local luminance detail should brighten the center feature"
    );
    let before_ratio = base[[12, 12, 0]] / base[[12, 12, 1]];
    let after_ratio = result.image[[12, 12, 0]] / result.image[[12, 12, 1]];
    assert!((after_ratio - before_ratio).abs() < 1e-12);
}

#[test]
fn test_tonemap_keeps_scene_referred_highlight_detail() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 3.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let mut img = Array3::<f64>::zeros((1, 2, 3));
    for c in 0..3 {
        img[[0, 0, c]] = 1.0;
        img[[0, 1, c]] = 2.0;
    }

    let out = scanstitch::tonemap::apply_tonemap_with_params(&img, &params);

    assert!(
        out[[0, 1, 0]] > out[[0, 0, 0]],
        "scene-referred highlight should render brighter than display white"
    );
    assert!(
        out[[0, 1, 0]] <= params.shoulder_max,
        "scene-referred highlight should stay inside the output shoulder"
    );
}

#[test]
fn test_tone_fit_measures_extended_scene_luminance() {
    let mut img = Array3::<f64>::zeros((10, 100, 3));
    for y in 0..10 {
        for x in 0..100 {
            let lum = if x >= 940 / 10 { 1.65 } else { 0.35 };
            for c in 0..3 {
                img[[y, x, c]] = lum;
            }
        }
    }

    let fit = scanstitch::tonemap::fit_tone_params_with_diagnostics(&img);

    assert!(
        fit.diagnostics.input_linear_percentiles[2] > 1.0,
        "tone fit should report scene-referred highlight percentiles above display white: {:?}",
        fit.diagnostics.input_linear_percentiles
    );
    assert!(
        fit.diagnostics.input_perceptual_percentiles[2] > 1.0,
        "perceptual fit histogram should not collapse extended highlights into the display top bin: {:?}",
        fit.diagnostics.input_perceptual_percentiles
    );
}

#[test]
fn test_fit_params_from_image() {
    let mut img = Array3::<f64>::zeros((100, 100, 3));
    for y in 0..100 {
        for x in 0..100 {
            let t = (y * 100 + x) as f64 / 10000.0;
            img[[y, x, 0]] = t;
            img[[y, x, 1]] = t;
            img[[y, x, 2]] = t;
        }
    }
    let params = scanstitch::tonemap::fit_tone_params(&img);
    assert!(params.midpoint > 0.0 && params.midpoint < 1.0);
    assert!(params.slope > 0.0);
}

#[test]
fn test_tonemap_scales_rgb_by_luminance() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 2.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let mut img = Array3::<f64>::zeros((1, 1, 3));
    img[[0, 0, 0]] = 0.40;
    img[[0, 0, 1]] = 0.20;
    img[[0, 0, 2]] = 0.10;

    let out = scanstitch::tonemap::apply_tonemap_with_params(&img, &params);
    let rg_in = img[[0, 0, 0]] / img[[0, 0, 1]];
    let rg_out = out[[0, 0, 0]] / out[[0, 0, 1]];
    let gb_in = img[[0, 0, 1]] / img[[0, 0, 2]];
    let gb_out = out[[0, 0, 1]] / out[[0, 0, 2]];

    assert!((rg_in - rg_out).abs() < 1e-6, "R/G ratio changed");
    assert!((gb_in - gb_out).abs() < 1e-6, "G/B ratio changed");
}

#[test]
fn test_tone_fit_uses_perceptual_luminance_domain_for_shadow_heavy_input() {
    let mut img = Array3::<f64>::zeros((100, 100, 3));
    for y in 0..100 {
        for x in 0..100 {
            let lum = if x < 5 {
                0.009
            } else if x < 50 {
                0.08
            } else if x < 95 {
                0.16
            } else {
                0.44
            };
            for c in 0..3 {
                img[[y, x, c]] = lum;
            }
        }
    }

    let fit = scanstitch::tonemap::fit_tone_params_with_diagnostics(&img);

    assert_eq!(fit.diagnostics.fit_domain, "log2_compressed_luminance");
    assert!(
        fit.diagnostics.input_linear_percentiles[1] < 0.15,
        "expected a shadow-heavy linear median, got {:?}",
        fit.diagnostics.input_linear_percentiles
    );
    assert!(
        fit.diagnostics.input_perceptual_percentiles[1]
            > fit.diagnostics.input_linear_percentiles[1],
        "perceptual encoding should lift the fitted midpoint out of the linear shadows"
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[1] > 0.31
            && fit.diagnostics.mapped_linear_percentiles[1] < 0.34,
        "shadow-heavy negative median should be lifted above the old dark placement: {:?}",
        fit.diagnostics.mapped_linear_percentiles
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[2] < 0.93,
        "expected the linear 95th percentile to retain highlight headroom after median lift: {:?}",
        fit.diagnostics.mapped_linear_percentiles
    );
}

#[test]
fn test_tone_fit_softens_shadow_heavy_testroll_frames() {
    let mut img = Array3::<f64>::zeros((100, 100, 3));
    for y in 0..100 {
        for x in 0..100 {
            let lum = if x < 5 {
                0.063
            } else if x < 50 {
                0.136
            } else if x < 95 {
                0.303
            } else {
                0.52
            };
            for c in 0..3 {
                img[[y, x, c]] = lum;
            }
        }
    }

    let fit = scanstitch::tonemap::fit_tone_params_with_diagnostics(&img);

    assert_eq!(fit.diagnostics.fit_domain, "log2_compressed_luminance");
    assert!(
        fit.params.slope <= 6.1 + 1e-6,
        "shadow-heavy negative fit should not amplify contrast harshly: {:?}",
        fit.params
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[0] > 0.012,
        "TESTROLL-like shadow floor should retain visible separation: input {:?}, mapped {:?}",
        fit.diagnostics.input_linear_percentiles,
        fit.diagnostics.mapped_linear_percentiles
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[2] < 0.90,
        "shadow-heavy highlights should retain headroom after softening: {:?}",
        fit.diagnostics.mapped_linear_percentiles
    );
}

#[test]
fn test_tone_fit_expands_very_dark_low_range_frames() {
    let mut img = Array3::<f64>::zeros((100, 100, 3));
    for y in 0..100 {
        for x in 0..100 {
            let lum = if x < 5 {
                0.0015
            } else if x < 50 {
                0.0025
            } else if x < 95 {
                0.0055
            } else {
                0.0080
            };
            for c in 0..3 {
                img[[y, x, c]] = lum;
            }
        }
    }

    let fit = scanstitch::tonemap::fit_tone_params_with_diagnostics(&img);

    assert_eq!(fit.diagnostics.fit_domain, "log2_compressed_luminance");
    assert!(
        fit.params.midpoint < 0.05,
        "very dark frames should fit against their actual perceptual median, not a high floor: {:?}",
        fit.params
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[1] > 0.15,
        "dark-frame median should be lifted out of the toe: input {:?}, mapped {:?}",
        fit.diagnostics.input_linear_percentiles,
        fit.diagnostics.mapped_linear_percentiles
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[2] > 0.85,
        "dark-frame highlights should be expanded toward the shoulder: input {:?}, mapped {:?}",
        fit.diagnostics.input_linear_percentiles,
        fit.diagnostics.mapped_linear_percentiles
    );
}

#[test]
fn test_tone_fit_places_low_range_negative_frames_above_dark_midtone_floor() {
    let mut img = Array3::<f64>::zeros((100, 100, 3));
    for y in 0..100 {
        for x in 0..100 {
            let lum = if x < 5 {
                0.0495
            } else if x < 50 {
                0.0535
            } else if x < 95 {
                0.0675
            } else {
                0.0750
            };
            for c in 0..3 {
                img[[y, x, c]] = lum;
            }
        }
    }

    let fit = scanstitch::tonemap::fit_tone_params_with_diagnostics(&img);

    assert_eq!(fit.diagnostics.fit_domain, "log2_compressed_luminance");
    assert!(
        fit.diagnostics.mapped_linear_percentiles[1] > 0.30,
        "low-range negative median should be lifted above the previous dark placement: {:?}",
        fit.diagnostics.mapped_linear_percentiles
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[2] > 0.70
            && fit.diagnostics.mapped_linear_percentiles[2] < 0.92,
        "low-range negative highlights should use more display range without pinning the shoulder: {:?}",
        fit.diagnostics.mapped_linear_percentiles
    );
}

#[test]
fn test_tone_fit_opens_ultra_flat_low_range_negative_frames() {
    let mut img = Array3::<f64>::zeros((100, 100, 3));
    for y in 0..100 {
        for x in 0..100 {
            let lum = if x < 5 {
                0.0595
            } else if x < 60 {
                0.0685
            } else if x < 95 {
                0.0715
            } else {
                0.0800
            };
            for c in 0..3 {
                img[[y, x, c]] = lum;
            }
        }
    }

    let fit = scanstitch::tonemap::fit_tone_params_with_diagnostics(&img);

    assert_eq!(fit.diagnostics.fit_domain, "log2_compressed_luminance");
    assert!(
        fit.params.slope > 10.0,
        "ultra-flat low-range negatives should get a bounded contrast boost: {:?}",
        fit.params
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[1] > 0.33
            && fit.diagnostics.mapped_linear_percentiles[1] < 0.37,
        "ultra-flat low-range median should keep the negative-film placement: {:?}",
        fit.diagnostics.mapped_linear_percentiles
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[2] >= 0.50,
        "ultra-flat low-range p95 should not stay muddy after median lift: input {:?}, mapped {:?}",
        fit.diagnostics.input_linear_percentiles,
        fit.diagnostics.mapped_linear_percentiles
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[0] > 0.03,
        "bounded boost should not crush the lower rendered tones: {:?}",
        fit.diagnostics.mapped_linear_percentiles
    );
}

#[test]
fn test_tone_fit_expands_testroll_like_low_range_sky_frames() {
    let mut img = Array3::<f64>::zeros((100, 100, 3));
    for y in 0..100 {
        for x in 0..100 {
            let lum = if x < 5 {
                0.0545
            } else if x < 50 {
                0.0615
            } else if x < 95 {
                0.0925
            } else {
                0.1050
            };
            for c in 0..3 {
                img[[y, x, c]] = lum;
            }
        }
    }

    let fit = scanstitch::tonemap::fit_tone_params_with_diagnostics(&img);

    assert_eq!(fit.diagnostics.fit_domain, "log2_compressed_luminance");
    assert!(
        fit.diagnostics.mapped_linear_percentiles[1] > 0.30,
        "TESTROLL-like low-range median should not stay at the previous dark placement: input {:?}, mapped {:?}",
        fit.diagnostics.input_linear_percentiles,
        fit.diagnostics.mapped_linear_percentiles
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[2] > 0.75
            && fit.diagnostics.mapped_linear_percentiles[2] < 0.93,
        "TESTROLL-like low-range highlights should use display range without pinning the shoulder: {:?}",
        fit.diagnostics.mapped_linear_percentiles
    );
}

#[test]
fn test_tone_fit_preserves_direct_density_midtones() {
    let mut img = Array3::<f64>::zeros((100, 100, 3));
    for y in 0..100 {
        for x in 0..100 {
            let lum = if x < 5 {
                0.38
            } else if x < 50 {
                0.49
            } else if x < 95 {
                0.66
            } else {
                0.82
            };
            for c in 0..3 {
                img[[y, x, c]] = lum;
            }
        }
    }

    let fit = scanstitch::tonemap::fit_tone_params_with_diagnostics(&img);

    assert_eq!(fit.diagnostics.fit_domain, "linear_luminance");
    assert!(
        fit.diagnostics.input_linear_percentiles[1] > 0.45,
        "fixture should model the post-fallback real-output midtone distribution: {:?}",
        fit.diagnostics.input_linear_percentiles
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[1]
            >= fit.diagnostics.input_linear_percentiles[1] * 0.85,
        "tone fit should not crush a healthy linear median after direct-density fallback: input {:?}, mapped {:?}",
        fit.diagnostics.input_linear_percentiles,
        fit.diagnostics.mapped_linear_percentiles
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[2] > 0.75
            && fit.diagnostics.mapped_linear_percentiles[2] < 0.95,
        "expected contrast expansion with retained highlight headroom: {:?}",
        fit.diagnostics.mapped_linear_percentiles
    );
}

#[test]
fn test_positive_scan_tone_fit_places_midtones_below_generic_negative_fit() {
    let mut img = Array3::<f64>::zeros((100, 100, 3));
    for y in 0..100 {
        for x in 0..100 {
            let lum = if x < 5 {
                0.12
            } else if x < 50 {
                0.40
            } else if x < 90 {
                0.62
            } else {
                0.86
            };
            img[[y, x, 0]] = lum * 1.04;
            img[[y, x, 1]] = lum;
            img[[y, x, 2]] = lum * 0.96;
        }
    }

    let generic = scanstitch::tonemap::fit_tone_params_with_diagnostics(&img);
    let positive = scanstitch::tonemap::fit_positive_scan_tone_params_with_diagnostics(&img);

    assert_eq!(positive.diagnostics.fit_domain, "linear_luminance");
    assert!(
        positive.diagnostics.mapped_linear_percentiles[1]
            < generic.diagnostics.mapped_linear_percentiles[1] - 0.08,
        "positive scan fit should not lift the median like the negative workflow: generic={:?}, positive={:?}",
        generic.diagnostics.mapped_linear_percentiles,
        positive.diagnostics.mapped_linear_percentiles
    );
    assert!(
        positive.diagnostics.mapped_linear_percentiles[1]
            <= positive.diagnostics.input_linear_percentiles[1] + 0.04,
        "positive scan midtone should not materially lift the scan median: input {:?}, mapped {:?}",
        positive.diagnostics.input_linear_percentiles,
        positive.diagnostics.mapped_linear_percentiles
    );
    assert!(
        (0.36..=0.44).contains(&positive.diagnostics.mapped_linear_percentiles[1]),
        "positive scan midtone should keep natural density with added visual weight: {:?}",
        positive.diagnostics.mapped_linear_percentiles
    );
    assert!(
        positive.diagnostics.mapped_linear_percentiles[2] > 0.78
            && positive.diagnostics.mapped_linear_percentiles[2] < 0.93,
        "positive scan shoulder should retain highlight detail without pushing p95 into the shoulder: {:?}",
        positive.diagnostics.mapped_linear_percentiles
    );
}

#[test]
fn test_positive_scan_auto_exposure_preserves_non_inverted_highlight_detail() {
    let mut img = Array3::<f64>::zeros((80, 120, 3));
    for y in 0..80 {
        for x in 0..120 {
            let t = x as f64 / 119.0;
            let mut lum = 0.34 + 0.62 * t;
            if (42..=58).contains(&x) && (24..=56).contains(&y) {
                lum = 0.98;
            }
            img[[y, x, 0]] = lum * (1.02 - 0.04 * t);
            img[[y, x, 1]] = lum;
            img[[y, x, 2]] = lum * (0.96 + 0.03 * t);
        }
    }

    let exposure_ev = scanstitch::tonemap::positive_scan_auto_exposure_ev(&img);
    assert!(
        exposure_ev < -0.02,
        "high-key positive scans should get a small non-brightening exposure placement: {exposure_ev}"
    );
    assert!(
        exposure_ev >= -0.45 - 1e-12,
        "positive auto exposure should stay bounded: {exposure_ev}"
    );

    let exposure_scale = 2.0f64.powf(exposure_ev);
    let exposed = img.mapv(|value| value * exposure_scale);
    let fit = scanstitch::tonemap::fit_positive_scan_tone_params_with_diagnostics(&exposed);
    let out = scanstitch::tonemap::apply_tonemap_with_params(&exposed, &fit.params);
    let quality = scanstitch::tonemap::render_quality_diagnostics(&out);

    let left = rendered_luminance_region(&out, 4, 18);
    let right = rendered_luminance_region(&out, 100, 116);
    assert!(
        right > left,
        "positive tone mapping must preserve non-inverted luminance ordering: left={left}, right={right}"
    );
    assert!(
        quality.midtone.luminance_percentiles[1] < 0.66,
        "positive tone placement should avoid overly light midtones: {:?}",
        quality.midtone.luminance_percentiles
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[2] < 0.94,
        "positive shoulder should keep highlight p95 below hard shoulder: {:?}",
        fit.diagnostics.mapped_linear_percentiles
    );
    assert!(
        quality.bright_neutral.luminance_percentiles[2] > quality.midtone.luminance_percentiles[2],
        "bright detail should remain separated from midtones: bright={:?}, mid={:?}",
        quality.bright_neutral.luminance_percentiles,
        quality.midtone.luminance_percentiles
    );
}

#[test]
fn test_tonemap_compresses_highlight_chroma_before_clipping() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 5.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let mut img = Array3::<f64>::zeros((1, 1, 3));
    img[[0, 0, 0]] = 0.20;
    img[[0, 0, 1]] = 0.90;
    img[[0, 0, 2]] = 1.00;

    let input_lum = 0.2880 * img[[0, 0, 0]] + 0.7119 * img[[0, 0, 1]] + 0.0001 * img[[0, 0, 2]];
    let mapped_lum = scanstitch::tonemap::apply_tone_curve(input_lum, &params);
    let naive_scale = mapped_lum / input_lum;
    assert!(
        img[[0, 0, 1]] * naive_scale > 1.0,
        "fixture should force a highlight channel out of gamut"
    );

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);
    let out_lum = 0.2880 * result.image[[0, 0, 0]]
        + 0.7119 * result.image[[0, 0, 1]]
        + 0.0001 * result.image[[0, 0, 2]];

    assert_eq!(result.diagnostics.highlight_chroma_compressed_ratio, 1.0);
    assert_eq!(
        result.diagnostics.pre_chroma_compression_clipped_high_ratio[1],
        1.0
    );
    assert_eq!(
        result
            .diagnostics
            .post_chroma_compression_clipped_high_ratio,
        [0.0, 0.0, 0.0]
    );
    assert!(
        (out_lum - mapped_lum).abs() < 1e-9,
        "highlight compression should preserve mapped luminance: out={}, target={}",
        out_lum,
        mapped_lum
    );
    assert!(
        result.image[[0, 0, 0]] > img[[0, 0, 0]] * naive_scale,
        "out-of-gamut highlights should desaturate toward neutral instead of only clamping high channels"
    );
    assert!(
        result.image[[0, 0, 1]] < 1.0,
        "green should be brought back inside gamut without hard clipping"
    );
    assert!(
        result.image[[0, 0, 2]] <= params.shoulder_max + 1e-12,
        "compressed highlights should roll off at the tone-curve shoulder"
    );
}

#[test]
fn test_tonemap_compresses_bright_neutral_chroma_without_moving_luminance() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 2.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let mut img = Array3::<f64>::zeros((1, 1, 3));
    img[[0, 0, 0]] = 0.55;
    img[[0, 0, 1]] = 0.70;
    img[[0, 0, 2]] = 0.80;

    let input_lum = 0.2880 * img[[0, 0, 0]] + 0.7119 * img[[0, 0, 1]] + 0.0001 * img[[0, 0, 2]];
    let mapped_lum = scanstitch::tonemap::apply_tone_curve(input_lum, &params);
    let naive_scale = mapped_lum / input_lum;
    let naive = [
        img[[0, 0, 0]] * naive_scale,
        img[[0, 0, 1]] * naive_scale,
        img[[0, 0, 2]] * naive_scale,
    ];
    assert!(
        naive.iter().all(|value| *value >= 0.0 && *value <= 1.0),
        "fixture should stay in gamut so only the neutral-highlight pass is exercised: {:?}",
        naive
    );

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);
    let out = [
        result.image[[0, 0, 0]],
        result.image[[0, 0, 1]],
        result.image[[0, 0, 2]],
    ];
    let out_lum = 0.2880 * out[0] + 0.7119 * out[1] + 0.0001 * out[2];
    let naive_sat = (naive.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - naive.iter().copied().fold(f64::INFINITY, f64::min))
        / naive.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let out_sat = (out.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - out.iter().copied().fold(f64::INFINITY, f64::min))
        / out.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(result.diagnostics.highlight_chroma_compressed_ratio, 0.0);
    assert_eq!(
        result.diagnostics.highlight_neutral_chroma_compressed_ratio,
        1.0
    );
    assert_eq!(result.diagnostics.color_protection_policy, "enabled");
    assert!(
        out_sat < naive_sat,
        "bright near-neutral chroma should be reduced: out={}, naive={}",
        out_sat,
        naive_sat
    );
    assert!(
        (out_lum - mapped_lum).abs() < 1e-9,
        "bright neutral compression should preserve mapped luminance: out={}, target={}",
        out_lum,
        mapped_lum
    );
}

#[test]
fn test_tonemap_compresses_midtone_neutral_chroma_without_moving_luminance() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 1.4,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let mut img = Array3::<f64>::zeros((1, 1, 3));
    img[[0, 0, 0]] = 0.36;
    img[[0, 0, 1]] = 0.44;
    img[[0, 0, 2]] = 0.50;

    let input_lum = 0.2880 * img[[0, 0, 0]] + 0.7119 * img[[0, 0, 1]] + 0.0001 * img[[0, 0, 2]];
    let mapped_lum = scanstitch::tonemap::apply_tone_curve(input_lum, &params);
    let naive_scale = mapped_lum / input_lum;
    let naive = [
        img[[0, 0, 0]] * naive_scale,
        img[[0, 0, 1]] * naive_scale,
        img[[0, 0, 2]] * naive_scale,
    ];

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);
    let out = [
        result.image[[0, 0, 0]],
        result.image[[0, 0, 1]],
        result.image[[0, 0, 2]],
    ];
    let out_lum = 0.2880 * out[0] + 0.7119 * out[1] + 0.0001 * out[2];
    let naive_sat = (naive.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - naive.iter().copied().fold(f64::INFINITY, f64::min))
        / naive.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let out_sat = (out.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - out.iter().copied().fold(f64::INFINITY, f64::min))
        / out.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(
        result.diagnostics.midtone_neutral_chroma_compressed_ratio,
        1.0
    );
    assert!(result.diagnostics.midtone_neutral_chroma_enabled);
    assert!(
        out_sat < naive_sat,
        "midtone near-neutral chroma should be reduced: out={}, naive={}",
        out_sat,
        naive_sat
    );
    assert!(
        out_sat <= naive_sat * 0.70,
        "midtone near-neutral chroma cleanup should be strong enough to reduce visible cast: out={}, naive={}",
        out_sat,
        naive_sat
    );
    assert!(
        (out_lum - mapped_lum).abs() < 1e-9,
        "midtone neutral compression should preserve mapped luminance: out={}, target={}",
        out_lum,
        mapped_lum
    );
}

#[test]
fn test_tonemap_compresses_lower_midtone_neutral_chroma() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 1.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let mut img = Array3::<f64>::zeros((1, 1, 3));
    img[[0, 0, 0]] = 0.12;
    img[[0, 0, 1]] = 0.14;
    img[[0, 0, 2]] = 0.16;

    let input_lum = 0.2880 * img[[0, 0, 0]] + 0.7119 * img[[0, 0, 1]] + 0.0001 * img[[0, 0, 2]];
    let mapped_lum = scanstitch::tonemap::apply_tone_curve(input_lum, &params);
    assert!(
        (0.29..=0.34).contains(&mapped_lum),
        "fixture should land in lower midtones: {mapped_lum}"
    );
    let naive_scale = mapped_lum / input_lum;
    let naive = [
        img[[0, 0, 0]] * naive_scale,
        img[[0, 0, 1]] * naive_scale,
        img[[0, 0, 2]] * naive_scale,
    ];

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);
    let out = [
        result.image[[0, 0, 0]],
        result.image[[0, 0, 1]],
        result.image[[0, 0, 2]],
    ];
    let out_lum = 0.2880 * out[0] + 0.7119 * out[1] + 0.0001 * out[2];
    let naive_sat = (naive.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - naive.iter().copied().fold(f64::INFINITY, f64::min))
        / naive.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let out_sat = (out.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - out.iter().copied().fold(f64::INFINITY, f64::min))
        / out.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(
        result.diagnostics.midtone_neutral_chroma_compressed_ratio,
        1.0
    );
    assert!(
        out_sat <= naive_sat * 0.65,
        "lower midtone neutral cleanup should reach useful strength before shadows: out={}, naive={}",
        out_sat,
        naive_sat
    );
    assert!(
        (out_lum - mapped_lum).abs() < 1e-9,
        "lower midtone neutral compression should preserve mapped luminance: out={}, target={}",
        out_lum,
        mapped_lum
    );
}

#[test]
fn test_tone_policy_weak_neutral_keeps_bounded_highlight_and_shadow_cleanup() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 2.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let protection = scanstitch::tonemap::ToneColorProtection {
        policy: scanstitch::tonemap::ToneColorProtectionPolicy::WeakNeutralBoundedNeutralCleanup,
        highlight_neutral_chroma_enabled: true,
        midtone_neutral_chroma_enabled: true,
        shadow_chroma_enabled: true,
        reason: "synthetic weak neutral support".to_string(),
    };
    let mut img = Array3::<f64>::zeros((2, 1, 3));
    img[[0, 0, 0]] = 0.55;
    img[[0, 0, 1]] = 0.70;
    img[[0, 0, 2]] = 0.80;
    img[[1, 0, 0]] = 0.36;
    img[[1, 0, 1]] = 0.44;
    img[[1, 0, 2]] = 0.50;

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_color_protection_diagnostics(
        &img,
        &params,
        &protection,
    );

    assert_eq!(
        result.diagnostics.color_protection_policy,
        "weak_neutral_bounded_neutral_cleanup"
    );
    assert!(result.diagnostics.highlight_neutral_chroma_compressed_ratio > 0.0);
    assert!(result.diagnostics.midtone_neutral_chroma_compressed_ratio > 0.0);
    assert!(result.diagnostics.highlight_neutral_chroma_enabled);
    assert!(result.diagnostics.midtone_neutral_chroma_enabled);
    assert!(result.diagnostics.shadow_chroma_enabled);
}

#[test]
fn test_tone_policy_anchor_review_keeps_bounded_highlight_and_shadow_cleanup() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 5.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let protection = scanstitch::tonemap::ToneColorProtection {
        policy:
            scanstitch::tonemap::ToneColorProtectionPolicy::ReviewBoundedNeutralAndShadowCleanup,
        highlight_neutral_chroma_enabled: true,
        midtone_neutral_chroma_enabled: true,
        shadow_chroma_enabled: true,
        reason: "synthetic anchor-support review".to_string(),
    };
    let mut img = Array3::<f64>::zeros((2, 1, 3));
    img[[0, 0, 0]] = 0.55;
    img[[0, 0, 1]] = 0.70;
    img[[0, 0, 2]] = 0.80;
    img[[1, 0, 0]] = 0.03;
    img[[1, 0, 1]] = 0.10;
    img[[1, 0, 2]] = 0.22;

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_color_protection_diagnostics(
        &img,
        &params,
        &protection,
    );

    assert_eq!(
        result.diagnostics.color_protection_policy,
        "review_bounded_neutral_shadow_cleanup"
    );
    assert!(result.diagnostics.highlight_neutral_chroma_compressed_ratio > 0.0);
    assert_eq!(result.diagnostics.shadow_chroma_compressed_ratio, 0.5);
    assert!(result.diagnostics.highlight_neutral_chroma_enabled);
    assert!(result.diagnostics.midtone_neutral_chroma_enabled);
    assert!(result.diagnostics.shadow_chroma_enabled);
    assert_eq!(result.diagnostics.color_trust_state, "review_required");
}

#[test]
fn test_tone_policy_model_review_keeps_bounded_shadow_cleanup_only() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 5.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let protection = scanstitch::tonemap::ToneColorProtection {
        policy: scanstitch::tonemap::ToneColorProtectionPolicy::ReviewBoundedShadowCleanup,
        highlight_neutral_chroma_enabled: false,
        midtone_neutral_chroma_enabled: false,
        shadow_chroma_enabled: true,
        reason: "synthetic model-plausibility review".to_string(),
    };
    let mut img = Array3::<f64>::zeros((3, 1, 3));
    img[[0, 0, 0]] = 0.55;
    img[[0, 0, 1]] = 0.70;
    img[[0, 0, 2]] = 0.80;
    img[[1, 0, 0]] = 0.34;
    img[[1, 0, 1]] = 0.42;
    img[[1, 0, 2]] = 0.50;
    img[[2, 0, 0]] = 0.03;
    img[[2, 0, 1]] = 0.10;
    img[[2, 0, 2]] = 0.22;

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_color_protection_diagnostics(
        &img,
        &params,
        &protection,
    );

    assert_eq!(
        result.diagnostics.color_protection_policy,
        "review_bounded_shadow_cleanup"
    );
    assert_eq!(
        result.diagnostics.highlight_neutral_chroma_compressed_ratio,
        0.0
    );
    assert_eq!(
        result.diagnostics.midtone_neutral_chroma_compressed_ratio,
        0.0
    );
    assert!(result.diagnostics.shadow_chroma_compressed_ratio > 0.0);
    assert!(!result.diagnostics.highlight_neutral_chroma_enabled);
    assert!(!result.diagnostics.midtone_neutral_chroma_enabled);
    assert!(result.diagnostics.shadow_chroma_enabled);
    assert_eq!(result.diagnostics.color_trust_state, "review_required");
}

#[test]
fn test_tone_policy_color_review_disables_shadow_cleanup_but_not_luminance_or_gamut_repair() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 5.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let protection = scanstitch::tonemap::ToneColorProtection {
        policy: scanstitch::tonemap::ToneColorProtectionPolicy::DisabledColorCandidateReview,
        highlight_neutral_chroma_enabled: false,
        midtone_neutral_chroma_enabled: false,
        shadow_chroma_enabled: false,
        reason: "synthetic color candidate review".to_string(),
    };
    let mut img = Array3::<f64>::zeros((2, 1, 3));
    img[[0, 0, 0]] = 0.20;
    img[[0, 0, 1]] = 0.90;
    img[[0, 0, 2]] = 1.00;
    img[[1, 0, 0]] = 0.03;
    img[[1, 0, 1]] = 0.10;
    img[[1, 0, 2]] = 0.22;

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_color_protection_diagnostics(
        &img,
        &params,
        &protection,
    );

    assert_eq!(
        result.diagnostics.color_protection_policy,
        "disabled_color_candidate_review"
    );
    assert_eq!(result.diagnostics.highlight_chroma_compressed_ratio, 0.5);
    assert_eq!(
        result.diagnostics.highlight_neutral_chroma_compressed_ratio,
        0.0
    );
    assert_eq!(result.diagnostics.shadow_chroma_compressed_ratio, 0.0);
    assert!(!result.diagnostics.highlight_neutral_chroma_enabled);
    assert!(!result.diagnostics.midtone_neutral_chroma_enabled);
    assert!(!result.diagnostics.shadow_chroma_enabled);
}

#[test]
fn test_tonemap_reduces_real_like_cool_highlight_cast() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.4965,
        slope: 5.797101449275363,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let mut img = Array3::<f64>::zeros((1, 1, 3));
    img[[0, 0, 0]] = 0.56;
    img[[0, 0, 1]] = 0.756;
    img[[0, 0, 2]] = 0.824;

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);
    let out = [
        result.image[[0, 0, 0]],
        result.image[[0, 0, 1]],
        result.image[[0, 0, 2]],
    ];
    let out_lum = 0.2880 * out[0] + 0.7119 * out[1] + 0.0001 * out[2];
    let out_sat = (out.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - out.iter().copied().fold(f64::INFINITY, f64::min))
        / out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let input_lum = 0.2880 * img[[0, 0, 0]] + 0.7119 * img[[0, 0, 1]] + 0.0001 * img[[0, 0, 2]];
    let mapped_lum = scanstitch::tonemap::apply_tone_curve(input_lum, &params);

    assert_eq!(
        result.diagnostics.highlight_neutral_chroma_compressed_ratio,
        1.0
    );
    assert!(
        out_sat < 0.12,
        "real-like cool near-neutral highlights should be pulled close to neutral; sat={}",
        out_sat
    );
    assert!(
        (out_lum - mapped_lum).abs() < 1e-9,
        "cool highlight correction should preserve mapped luminance: out={}, target={}",
        out_lum,
        mapped_lum
    );
    assert!(
        out[2] - out[0] < 0.10,
        "blue highlight separation should be reduced: {:?}",
        out
    );
}

#[test]
fn test_tonemap_does_not_neutralize_bright_saturated_colors() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 2.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let mut img = Array3::<f64>::zeros((1, 1, 3));
    img[[0, 0, 0]] = 0.78;
    img[[0, 0, 1]] = 0.68;
    img[[0, 0, 2]] = 0.10;

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);

    assert_eq!(
        result.diagnostics.highlight_neutral_chroma_compressed_ratio,
        0.0
    );
    assert_eq!(
        result.diagnostics.highlight_chroma_compressed_ratio, 0.0,
        "fixture should stay in gamut and avoid the hard highlight repair path"
    );
}

#[test]
fn test_tonemap_compresses_shadow_chroma_without_moving_luminance() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 2.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let mut img = Array3::<f64>::zeros((1, 1, 3));
    img[[0, 0, 0]] = 0.03;
    img[[0, 0, 1]] = 0.10;
    img[[0, 0, 2]] = 0.22;

    let input_lum = 0.2880 * img[[0, 0, 0]] + 0.7119 * img[[0, 0, 1]] + 0.0001 * img[[0, 0, 2]];
    let mapped_lum = scanstitch::tonemap::apply_tone_curve(input_lum, &params);
    let naive_scale = mapped_lum / input_lum;
    let naive = [
        img[[0, 0, 0]] * naive_scale,
        img[[0, 0, 1]] * naive_scale,
        img[[0, 0, 2]] * naive_scale,
    ];

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);
    let out = [
        result.image[[0, 0, 0]],
        result.image[[0, 0, 1]],
        result.image[[0, 0, 2]],
    ];
    let out_lum = 0.2880 * out[0] + 0.7119 * out[1] + 0.0001 * out[2];
    let naive_sat = (naive.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - naive.iter().copied().fold(f64::INFINITY, f64::min))
        / naive.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let out_sat = (out.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - out.iter().copied().fold(f64::INFINITY, f64::min))
        / out.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(result.diagnostics.shadow_chroma_compressed_ratio, 1.0);
    assert!(
        out_sat < naive_sat,
        "shadow chroma should be reduced: out={}, naive={}",
        out_sat,
        naive_sat
    );
    assert!(
        (out_lum - mapped_lum).abs() < 1e-9,
        "shadow chroma compression should preserve mapped luminance: out={}, target={}",
        out_lum,
        mapped_lum
    );
}

fn rendered_luminance_region(img: &Array3<f64>, x0: usize, x1: usize) -> f64 {
    let (height, width, _) = img.dim();
    let mut sum = 0.0;
    let mut count = 0usize;
    for y in 0..height {
        for x in x0.min(width)..x1.min(width) {
            sum += 0.2880 * img[[y, x, 0]] + 0.7119 * img[[y, x, 1]] + 0.0001 * img[[y, x, 2]];
            count += 1;
        }
    }
    sum / count.max(1) as f64
}

#[test]
fn test_tonemap_strongly_compresses_upper_shadow_chroma() {
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 2.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let mut img = Array3::<f64>::zeros((1, 1, 3));
    img[[0, 0, 0]] = 0.12;
    img[[0, 0, 1]] = 0.22;
    img[[0, 0, 2]] = 0.36;

    let input_lum = 0.2880 * img[[0, 0, 0]] + 0.7119 * img[[0, 0, 1]] + 0.0001 * img[[0, 0, 2]];
    let mapped_lum = scanstitch::tonemap::apply_tone_curve(input_lum, &params);
    assert!(
        mapped_lum > 0.14 && mapped_lum < 0.22,
        "fixture should land in the upper shadow compression band: {}",
        mapped_lum
    );
    let naive_scale = mapped_lum / input_lum;
    let naive = [
        img[[0, 0, 0]] * naive_scale,
        img[[0, 0, 1]] * naive_scale,
        img[[0, 0, 2]] * naive_scale,
    ];

    let result = scanstitch::tonemap::apply_tonemap_with_params_and_diagnostics(&img, &params);
    let out = [
        result.image[[0, 0, 0]],
        result.image[[0, 0, 1]],
        result.image[[0, 0, 2]],
    ];
    let naive_sat = (naive.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - naive.iter().copied().fold(f64::INFINITY, f64::min))
        / naive.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let out_sat = (out.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - out.iter().copied().fold(f64::INFINITY, f64::min))
        / out.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(result.diagnostics.shadow_chroma_compressed_ratio, 1.0);
    assert!(
        out_sat < naive_sat * 0.75,
        "upper-shadow chroma should be compressed enough to reduce speckle: out={}, naive={}",
        out_sat,
        naive_sat
    );
}

#[test]
fn test_tonemap_allows_chroma_denoise_in_saturated_shadows() {
    let mut img = Array3::<f64>::zeros((31, 31, 3));
    for y in 0..31 {
        for x in 0..31 {
            let sign: f64 = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
            img[[y, x, 0]] = (0.02f64 + 0.012 * sign).clamp(0.0, 1.0);
            img[[y, x, 1]] = (0.08f64 - 0.018 * sign).clamp(0.0, 1.0);
            img[[y, x, 2]] = (0.22f64 + 0.040 * sign).clamp(0.0, 1.0);
        }
    }
    let params = scanstitch::tonemap::ToneCurveParams {
        domain: scanstitch::tonemap::ToneFitDomain::LinearLuminance,
        midpoint: 0.5,
        slope: 2.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let protection = scanstitch::tonemap::ToneColorProtection {
        policy:
            scanstitch::tonemap::ToneColorProtectionPolicy::ReviewBoundedNeutralAndShadowCleanup,
        highlight_neutral_chroma_enabled: true,
        midtone_neutral_chroma_enabled: true,
        shadow_chroma_enabled: true,
        reason: "synthetic anchor-support review".to_string(),
    };

    let before = scanstitch::tonemap::render_grain_diagnostics(&img);
    let result = scanstitch::tonemap::apply_tonemap_with_params_and_color_protection_diagnostics(
        &img,
        &params,
        &protection,
    );
    let after = scanstitch::tonemap::render_grain_diagnostics(&result.image);
    let quality = scanstitch::tonemap::render_quality_diagnostics(&result.image);

    assert!(result.diagnostics.shadow_chroma_compressed_ratio > 0.90);
    assert!(
        result.diagnostics.noise_reduction_saturation_limited_ratio < 0.10,
        "saturated shadows should still receive chroma denoise: {:?}",
        result.diagnostics.noise_reduction_saturation_limited_ratio
    );
    assert!(
        after.chroma_residual_p95 < before.chroma_residual_p95,
        "saturated shadow chroma residual should drop after final cleanup: before={:?} after={:?}",
        before,
        after
    );
    assert!(
        quality.shadow.saturation_p95 <= 0.74,
        "extreme low-luma shadow saturation should stay bounded after final cleanup: {:?}",
        quality.shadow
    );
    assert!(result.image.iter().all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn test_render_quality_diagnostics_reports_shadow_midtone_and_bright_bands() {
    let mut img = Array3::<f64>::zeros((10, 100, 3));
    for y in 0..10 {
        for x in 0..100 {
            let i = y * 100 + x;
            let lum = 0.02 + 0.96 * (i as f64 / 999.0);
            let rgb = if i < 100 {
                [lum * 0.25, lum, lum * 1.60]
            } else if i >= 900 {
                [lum * 0.94, lum * 0.98, lum]
            } else {
                [lum, lum, lum]
            };
            for c in 0..3 {
                img[[y, x, c]] = rgb[c].clamp(0.0, 1.0);
            }
        }
    }

    let diagnostics = scanstitch::tonemap::render_quality_diagnostics(&img);

    assert_eq!(diagnostics.sample_count, 1000);
    assert_eq!(diagnostics.sample_stride, 1);
    assert_eq!(diagnostics.overall.pixel_count, 1000);
    assert_eq!(diagnostics.shadow.pixel_count, 100);
    assert_eq!(diagnostics.midtone.pixel_count, 200);
    assert_eq!(diagnostics.midtone_neutral.pixel_count, 200);
    assert_eq!(diagnostics.midtone_saturated.pixel_count, 0);
    assert_eq!(diagnostics.bright_neutral.pixel_count, 100);
    assert_eq!(diagnostics.bright_saturated.pixel_count, 0);
    assert_eq!(diagnostics.midtone_neutral_max_saturation, 0.35);
    assert_eq!(diagnostics.bright_neutral_max_saturation, 0.35);
    assert!(
        diagnostics.overall.luminance_percentiles[2] - diagnostics.overall.luminance_percentiles[0]
            > 0.80,
        "expected full-frame rendered luminance range, got {:?}",
        diagnostics.overall.luminance_percentiles
    );
    assert!(
        diagnostics.overall.luminance_percentiles[1] > 0.45
            && diagnostics.overall.luminance_percentiles[1] < 0.55,
        "expected overall luminance median near the center, got {:?}",
        diagnostics.overall.luminance_percentiles
    );
    assert!(
        diagnostics.shadow.saturation_median > 0.70,
        "expected saturated shadow fixture, got {:?}",
        diagnostics.shadow
    );
    assert!(
        diagnostics.bright_neutral.saturation_p95 < 0.07,
        "expected near-neutral bright fixture, got {:?}",
        diagnostics.bright_neutral
    );
    assert!(
        diagnostics.midtone.luminance_percentiles[1] > 0.45
            && diagnostics.midtone.luminance_percentiles[1] < 0.55,
        "expected midtone median luminance near the center, got {:?}",
        diagnostics.midtone.luminance_percentiles
    );
    assert!(
        diagnostics.midtone.rgb_median[0] > 0.45 && diagnostics.midtone.rgb_median[0] < 0.55,
        "expected midtone RGB medians to be reported, got {:?}",
        diagnostics.midtone.rgb_median
    );
}

#[test]
fn test_render_quality_diagnostics_separates_midtone_neutral_pixels() {
    let mut img = Array3::<f64>::zeros((10, 100, 3));
    for y in 0..10 {
        for x in 0..100 {
            let i = y * 100 + x;
            let lum = 0.02 + 0.96 * (i as f64 / 999.0);
            let rgb = if (400..500).contains(&i) {
                [lum * 0.96, lum * 0.98, lum]
            } else if (500..600).contains(&i) {
                [lum * 0.35, lum * 0.70, lum]
            } else {
                [lum, lum, lum]
            };
            for c in 0..3 {
                img[[y, x, c]] = rgb[c].clamp(0.0, 1.0);
            }
        }
    }

    let diagnostics = scanstitch::tonemap::render_quality_diagnostics(&img);

    assert_eq!(
        diagnostics.midtone_neutral.pixel_count + diagnostics.midtone_saturated.pixel_count,
        diagnostics.midtone.pixel_count
    );
    assert!(
        diagnostics.midtone_neutral.pixel_count > 0
            && diagnostics.midtone_saturated.pixel_count > 0,
        "midtone diagnostics should split mixed neutral/colored content: {:?} / {:?}",
        diagnostics.midtone_neutral,
        diagnostics.midtone_saturated
    );
    assert!(
        diagnostics.midtone_neutral.saturation_p95 < 0.05,
        "near-neutral midtones should be measured separately: {:?}",
        diagnostics.midtone_neutral
    );
    assert!(
        diagnostics.midtone_saturated.saturation_median > 0.60,
        "colored midtones should remain visible in their own band: {:?}",
        diagnostics.midtone_saturated
    );
}

#[test]
fn test_render_quality_diagnostics_separates_bright_saturated_pixels() {
    let mut img = Array3::<f64>::zeros((10, 100, 3));
    for y in 0..10 {
        for x in 0..100 {
            let i = y * 100 + x;
            let lum = 0.02 + 0.96 * (i as f64 / 999.0);
            let rgb = if i >= 950 {
                [lum, lum * 0.88, lum * 0.18]
            } else if i >= 900 {
                [lum * 0.96, lum * 0.99, lum]
            } else {
                [lum, lum, lum]
            };
            for c in 0..3 {
                img[[y, x, c]] = rgb[c].clamp(0.0, 1.0);
            }
        }
    }

    let diagnostics = scanstitch::tonemap::render_quality_diagnostics(&img);

    assert_eq!(
        diagnostics.bright_neutral.pixel_count + diagnostics.bright_saturated.pixel_count,
        100
    );
    assert!(
        diagnostics.bright_neutral.pixel_count > 0 && diagnostics.bright_saturated.pixel_count > 0,
        "bright diagnostics should split mixed highlights: {:?} / {:?}",
        diagnostics.bright_neutral,
        diagnostics.bright_saturated
    );
    assert!(
        diagnostics.bright_neutral.saturation_p95 < 0.05,
        "near-neutral bright pixels should be measured separately: {:?}",
        diagnostics.bright_neutral
    );
    assert!(
        diagnostics.bright_saturated.saturation_median > 0.70,
        "saturated bright pixels should remain visible in their own band: {:?}",
        diagnostics.bright_saturated
    );
}

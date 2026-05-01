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
                    v >= 0.0 && v <= 1.0,
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
        fit.diagnostics.mapped_perceptual_percentiles[1] > 0.45
            && fit.diagnostics.mapped_perceptual_percentiles[1] < 0.55,
        "expected the perceptual-domain median to stay near the curve midpoint: {:?}",
        fit.diagnostics.mapped_perceptual_percentiles
    );
    assert!(
        fit.diagnostics.mapped_linear_percentiles[2] < 0.85,
        "expected the linear 95th percentile to retain highlight headroom instead of pinning the shoulder: {:?}",
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
    assert_eq!(diagnostics.shadow.pixel_count, 100);
    assert_eq!(diagnostics.midtone.pixel_count, 200);
    assert_eq!(diagnostics.bright_neutral.pixel_count, 100);
    assert_eq!(diagnostics.bright_saturated.pixel_count, 0);
    assert_eq!(diagnostics.bright_neutral_max_saturation, 0.35);
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

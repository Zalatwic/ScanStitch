use approx::assert_relative_eq;
use ndarray::Array3;

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
                assert!(v >= 0.0, "negative density: {} at ({},{},{})", v, y, x, ch);
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
            result.diagnostics.clamped_to_zero[c] > 0,
            "channel {} should clamp the outlier against the robust Dmax",
            c
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

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
fn test_density_conversion_roundtrip() {
    let t = 0.5f64;
    let d = scanstitch::density::transmittance_to_density(t);
    let t2 = scanstitch::density::density_to_transmittance(d);
    assert_relative_eq!(t, t2, epsilon = 1e-10);
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

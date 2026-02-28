use ndarray::Array3;

#[test]
fn test_sigmoid_has_toe_and_shoulder() {
    let params = scanstitch::tonemap::ToneCurveParams::default();
    let shadow = scanstitch::tonemap::apply_tone_curve(0.01, &params);
    assert!(shadow > 0.001, "toe: {}", shadow);
    let mid = scanstitch::tonemap::apply_tone_curve(0.5, &params);
    assert!(mid > 0.3 && mid < 0.7, "mid: {}", mid);
    let highlight = scanstitch::tonemap::apply_tone_curve(0.99, &params);
    assert!(highlight < 0.99, "shoulder: {}", highlight);
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

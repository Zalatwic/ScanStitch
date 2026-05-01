mod common;
use common::synthetic;
use ndarray::Array3;

#[test]
fn test_removes_black_borders() {
    let img = synthetic::image_with_borders(100, 200, 10, [8000, 6000, 5000]);
    let (cropped, top, bot) = scanstitch::border::remove_borders(&img, 2);
    assert_eq!(top, 12);
    assert_eq!(bot, 12);
    assert_eq!(cropped.dim().0, 76);
    assert!(cropped[[0, 100, 0]] > 1000);
}

#[test]
fn test_removes_white_borders() {
    let mut img = Array3::<u16>::zeros((100, 200, 3));
    for y in 0..8 {
        for x in 0..200 {
            for c in 0..3 {
                img[[y, x, c]] = 65000;
            }
        }
    }
    for y in 92..100 {
        for x in 0..200 {
            for c in 0..3 {
                img[[y, x, c]] = 65000;
            }
        }
    }
    for y in 8..92 {
        for x in 0..200 {
            img[[y, x, 0]] = 8000;
            img[[y, x, 1]] = 6000;
            img[[y, x, 2]] = 5000;
        }
    }
    let (cropped, top, bot) = scanstitch::border::remove_borders(&img, 2);
    assert_eq!(top, 10);
    assert_eq!(bot, 10);
    assert_eq!(cropped.dim().0, 80);
}

#[test]
fn test_no_borders() {
    let img = synthetic::constant_image(100, 200, [8000, 6000, 5000]);
    let (cropped, top, bot) = scanstitch::border::remove_borders(&img, 2);
    assert_eq!(top, 0);
    assert_eq!(bot, 0);
    assert_eq!(cropped.dim().0, 100);
}

#[test]
fn test_mixed_border_variance() {
    let mut img = synthetic::image_with_borders(100, 200, 15, [8000, 6000, 5000]);
    synthetic::add_noise(&mut img, 42, 50);
    let (cropped, top, bot) = scanstitch::border::remove_borders(&img, 2);
    assert!(top >= 13 && top <= 17, "top={}", top);
    assert!(bot >= 13 && bot <= 17, "bot={}", bot);
}

#[test]
fn test_safety_margin_never_eats_content() {
    let img = synthetic::image_with_borders(100, 200, 5, [12000, 9000, 7000]);
    let (cropped, top, bot) = scanstitch::border::remove_borders(&img, 2);
    assert!(top <= 7, "top cropped too much: {}", top);
    assert!(bot <= 7, "bot cropped too much: {}", bot);
    assert!(cropped[[0, 100, 0]] > 5000, "first row should be content");
}

#[test]
fn test_border_diagnostics_warn_when_nothing_removed() {
    let img = synthetic::constant_image(100, 200, [8000, 6000, 5000]);
    let result = scanstitch::border::remove_borders_with_diagnostics(&img, 2);
    assert_eq!(result.diagnostics.top_removed, 0);
    assert_eq!(result.diagnostics.bottom_removed, 0);
    assert!(!result.diagnostics.dead_zone_detected);
    assert!(
        result
            .diagnostics
            .warnings
            .iter()
            .any(|warning| warning.contains("dead-zone")),
        "expected explicit warning when border detector no-ops"
    );
}

#[test]
fn test_gradual_top_ramp_is_not_treated_as_dead_zone() {
    let mut img = synthetic::constant_image(120, 200, [8000, 6000, 5000]);
    for y in 0..30 {
        let t = y as f64 / 29.0;
        let scale = 0.78 + 0.18 * t;
        for x in 0..200 {
            img[[y, x, 0]] = (8000.0 * scale).round() as u16;
            img[[y, x, 1]] = (6000.0 * scale).round() as u16;
            img[[y, x, 2]] = (5000.0 * scale).round() as u16;
        }
    }

    let result = scanstitch::border::remove_borders_with_diagnostics(&img, 2);

    assert_eq!(result.diagnostics.top_removed, 0);
    assert!(
        !result.diagnostics.dead_zone_detected,
        "a gradual edge falloff should not be treated as removable dead-zone rows"
    );
}

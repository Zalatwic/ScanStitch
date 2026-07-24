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
fn test_removes_left_and_right_scanner_borders() {
    let mut img = synthetic::constant_image(100, 220, [9000, 6500, 4800]);
    for y in 0..100 {
        for x in 0..9 {
            for c in 0..3 {
                img[[y, x, c]] = 0;
            }
        }
        for x in 207..220 {
            for c in 0..3 {
                img[[y, x, c]] = 0;
            }
        }
    }

    let result = scanstitch::border::remove_borders_with_diagnostics(&img, 2);

    assert_eq!(result.diagnostics.left_removed, 11);
    assert_eq!(result.diagnostics.right_removed, 15);
    assert_eq!(result.cropped.dim(), (100, 194, 3));
    assert!(result.cropped[[50, 0, 0]] > 5000);
    assert!(result.cropped[[50, 193, 0]] > 5000);
}

#[test]
fn test_removes_curved_or_flare_contaminated_side_border() {
    let mut img = synthetic::constant_image(160, 300, [10_000, 7200, 5200]);
    for y in 0..160 {
        let edge_value = 400 + ((y * 37) % 1900) as u16;
        for x in 0..8 {
            img[[y, x, 0]] = edge_value;
            img[[y, x, 1]] = edge_value / 2;
            img[[y, x, 2]] = edge_value / 3;
        }
    }

    let result = scanstitch::border::remove_borders_with_diagnostics(&img, 2);

    assert_eq!(result.diagnostics.left_removed, 10);
    assert_eq!(result.diagnostics.right_removed, 0);
    assert!(result.diagnostics.left_confidence > 0.0);
    assert_eq!(result.diagnostics.right_confidence, 0.0);
    assert_eq!(result.cropped.dim().1, 290);
    assert!(result.cropped[[80, 0, 0]] > 9000);
}

#[test]
fn test_removes_wide_dead_zone_with_gradual_inward_flare_transition() {
    let mut img = synthetic::constant_image(180, 420, [24_000, 20_000, 16_000]);
    for y in 0..180 {
        for x in 0..24 {
            for channel in 0..3 {
                img[[y, x, channel]] = 450;
            }
        }
        for x in 24..29 {
            let blend = (x - 23) as f64 / 6.0;
            let target = [24_000.0, 20_000.0, 16_000.0];
            for channel in 0..3 {
                img[[y, x, channel]] =
                    (450.0 * (1.0 - blend) + target[channel] * blend).round() as u16;
            }
        }
    }

    let result = scanstitch::border::remove_borders_with_diagnostics(&img, 2);

    assert!(
        (27..=31).contains(&result.diagnostics.left_removed),
        "expected the dead zone and bounded flare ramp to be removed: {:?}",
        result.diagnostics
    );
    assert!(result.cropped[[90, 0, 0]] > 20_000);
    assert_eq!(result.diagnostics.right_removed, 0);
}

#[test]
fn test_removes_scanner_border_on_all_four_edges() {
    let mut img = synthetic::constant_image(120, 220, [9000, 6500, 4800]);
    for y in 0..120 {
        for x in 0..220 {
            if !(7..109).contains(&y) || !(9..207).contains(&x) {
                for c in 0..3 {
                    img[[y, x, c]] = 0;
                }
            }
        }
    }

    let result = scanstitch::border::remove_borders_with_diagnostics(&img, 2);

    assert_eq!(result.diagnostics.top_removed, 9);
    assert_eq!(result.diagnostics.bottom_removed, 13);
    assert_eq!(result.diagnostics.left_removed, 11);
    assert_eq!(result.diagnostics.right_removed, 15);
    assert_eq!(result.cropped.dim(), (98, 194, 3));
    assert!(result.diagnostics.dead_zone_detected);
    assert!(result.diagnostics.evidence_evaluated);
    assert!(!result.diagnostics.vertical_crop_rejected);
    assert!(!result.diagnostics.horizontal_crop_rejected);
    for confidence in [
        result.diagnostics.top_confidence,
        result.diagnostics.bottom_confidence,
        result.diagnostics.left_confidence,
        result.diagnostics.right_confidence,
    ] {
        assert!(
            confidence >= 0.95,
            "strong four-edge crop should have strong measured evidence: {:?}",
            result.diagnostics
        );
    }
    assert!(
        result.diagnostics.top_boundary_transition_delta
            >= result.diagnostics.row_boundary_derivative_threshold
    );
    assert!(
        result.diagnostics.left_boundary_transition_delta
            >= result.diagnostics.column_boundary_derivative_threshold
    );
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
    let (_cropped, top, bot) = scanstitch::border::remove_borders(&img, 2);
    assert!((13..=17).contains(&top), "top={}", top);
    assert!((13..=17).contains(&bot), "bot={}", bot);
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
    assert_eq!(result.diagnostics.left_removed, 0);
    assert_eq!(result.diagnostics.right_removed, 0);
    assert!(!result.diagnostics.dead_zone_detected);
    assert!(result.diagnostics.evidence_evaluated);
    assert_eq!(result.diagnostics.top_confidence, 0.0);
    assert_eq!(result.diagnostics.bottom_confidence, 0.0);
    assert_eq!(result.diagnostics.left_confidence, 0.0);
    assert_eq!(result.diagnostics.right_confidence, 0.0);
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
    assert_eq!(result.diagnostics.top_confidence, 0.0);
    assert!(
        !result.diagnostics.dead_zone_detected,
        "a gradual edge falloff should not be treated as removable dead-zone rows"
    );
}

fn image_with_uniform_right_rebate_and_dark_textured_scene_edge(
    boundary_median: u16,
    inward_median: u16,
) -> Array3<u16> {
    let height = 64usize;
    let width = 6000usize;
    let scene_edge_start = 5880usize;
    let rebate_start = 5930usize;
    let mut image = synthetic::constant_image(height, width, [6000, 5000, 4000]);
    for y in 0..height {
        let inward_edge_value = if y < height / 2 {
            inward_median.saturating_sub(5000)
        } else {
            inward_median.saturating_add(5000)
        };
        for x in scene_edge_start..rebate_start {
            for channel in 0..3 {
                image[[y, x, channel]] = inward_edge_value;
            }
        }
        let boundary_value = if y < height / 2 {
            boundary_median.saturating_sub(5000)
        } else {
            boundary_median.saturating_add(5000)
        };
        for channel in 0..3 {
            image[[y, rebate_start - 1, channel]] = boundary_value;
        }
        for x in rebate_start..width {
            for channel in 0..3 {
                image[[y, x, channel]] = 30_000;
            }
        }
    }
    image
}

#[test]
fn overwhelming_uniform_edge_support_accepts_a_gradual_rebate_boundary() {
    let image = image_with_uniform_right_rebate_and_dark_textured_scene_edge(29_900, 29_800);

    let result = scanstitch::border::remove_borders_with_diagnostics(&image, 2);

    assert!(
        (70..=74).contains(&result.diagnostics.right_removed),
        "expected only the uniform rebate plus bounded safety margin to be removed: {:?}",
        result.diagnostics
    );
    assert!(
        result.diagnostics.right_overwhelming_support_fallback_used,
        "expected overwhelming-edge fallback evidence: {:?}",
        result.diagnostics
    );
    assert!(!result.diagnostics.right_unresolved_strong_candidate);
    assert!(result.diagnostics.right_confidence >= 0.35);
    assert!(result.cropped[[0, result.cropped.dim().1 - 1, 0]] < 30_000);
}

#[test]
fn overwhelming_edge_without_minimum_boundary_change_is_retained_for_review() {
    let image = image_with_uniform_right_rebate_and_dark_textured_scene_edge(29_950, 29_900);

    let result = scanstitch::border::remove_borders_with_diagnostics(&image, 2);

    assert_eq!(
        result.diagnostics.right_removed, 0,
        "{:#?}",
        result.diagnostics
    );
    assert!(!result.diagnostics.right_overwhelming_support_fallback_used);
    assert!(
        result.diagnostics.right_unresolved_strong_candidate,
        "strong-but-unresolved evidence must not silently pass: {:?}",
        result.diagnostics
    );
    assert!(result
        .diagnostics
        .warnings
        .iter()
        .any(|warning| warning.contains("geometry review is required")));
}

#[test]
fn textured_subject_band_at_image_edge_is_not_an_unresolved_scanner_border() {
    let mut image = synthetic::constant_image(128, 192, [24_900, 24_900, 24_900]);
    for y in 0..128 {
        for x in 0..192 {
            let checker = if (x + y) % 2 == 0 { 1_i32 } else { -1_i32 };
            if x < 32 {
                let neutral_base = if x < 16 { 10_000 } else { 46_000 };
                let neutral = (neutral_base + checker * 800).clamp(0, u16::MAX as i32) as u16;
                for channel in 0..3 {
                    image[[y, x, channel]] = neutral;
                }
                continue;
            }
            let luma_step = if (48..96).contains(&x) { 7_000 } else { 0 };
            let chroma_step = if (112..160).contains(&x) { 3_500 } else { 0 };
            let base = 24_900 + luma_step;
            image[[y, x, 0]] =
                (base + checker * 5_200 + chroma_step).clamp(0, u16::MAX as i32) as u16;
            image[[y, x, 1]] = base.clamp(0, u16::MAX as i32) as u16;
            image[[y, x, 2]] =
                (base - checker * 5_200 - chroma_step).clamp(0, u16::MAX as i32) as u16;
        }
    }

    let result = scanstitch::border::remove_borders_with_diagnostics(&image, 2);

    assert_eq!(
        (
            result.diagnostics.top_removed,
            result.diagnostics.bottom_removed,
            result.diagnostics.left_removed,
            result.diagnostics.right_removed,
        ),
        (0, 0, 0, 0)
    );
    assert!(!result.diagnostics.top_unresolved_strong_candidate);
    assert!(!result.diagnostics.bottom_unresolved_strong_candidate);
    assert!(!result.diagnostics.left_unresolved_strong_candidate);
    assert!(!result.diagnostics.right_unresolved_strong_candidate);
}

#[test]
fn smooth_full_frame_gradient_is_not_an_unresolved_scanner_border() {
    let image = synthetic::gradient_image(120, 200, [1800, 2000, 2200], [14500, 15000, 15500]);

    let result = scanstitch::border::remove_borders_with_diagnostics(&image, 2);

    assert_eq!(
        (
            result.diagnostics.top_removed,
            result.diagnostics.bottom_removed,
            result.diagnostics.left_removed,
            result.diagnostics.right_removed,
        ),
        (0, 0, 0, 0)
    );
    assert!(!result.diagnostics.left_unresolved_strong_candidate);
    assert!(!result.diagnostics.right_unresolved_strong_candidate);
}

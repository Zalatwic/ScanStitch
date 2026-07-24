use ndarray::Array3;
use scanstitch::cli::{TechnicalWhiteBalanceMode, WhiteBalanceArgs};
use scanstitch::white_balance::{
    apply_creative_white_balance, apply_creative_white_balance_owned, apply_technical_white_balance,
};

fn coherent_neutral_scene_with_cast(cast: [f64; 3]) -> Array3<f64> {
    let mut image = Array3::<f64>::zeros((72, 96, 3));
    for y in 0..72 {
        for x in 0..96 {
            let gradient = 0.04 + 1.18 * (0.55 * x as f64 / 95.0 + 0.45 * y as f64 / 71.0);
            let texture = 1.0 + 0.025 * ((x * 17 + y * 11) % 13) as f64 / 12.0;
            for channel in 0..3 {
                image[[y, x, channel]] = gradient * texture * cast[channel];
            }
        }
    }
    image[[0, 0, 0]] = -0.20;
    image[[0, 0, 1]] = 0.08;
    image[[0, 0, 2]] = 0.06;
    image
}

fn mean_rgb(image: &Array3<f64>) -> [f64; 3] {
    let (height, width, _) = image.dim();
    let mut sum = [0.0; 3];
    let mut count = 0usize;
    for y in 1..height {
        for x in 0..width {
            for channel in 0..3 {
                sum[channel] += image[[y, x, channel]];
            }
            count += 1;
        }
    }
    [
        sum[0] / count as f64,
        sum[1] / count as f64,
        sum[2] / count as f64,
    ]
}

fn log_chroma(rgb: [f64; 3]) -> f64 {
    let rg = (rgb[0] / rgb[1]).ln();
    let bg = (rgb[2] / rgb[1]).ln();
    (rg * rg + bg * bg).sqrt()
}

#[test]
fn automatic_technical_white_balance_requires_coherent_multitone_evidence_and_improves_it() {
    let image = coherent_neutral_scene_with_cast([1.12, 1.0, 0.88]);
    let settings = WhiteBalanceArgs::default();
    let before = log_chroma(mean_rgb(&image));

    let result = apply_technical_white_balance(image.clone(), &settings);
    let after = log_chroma(mean_rgb(&result.image));

    assert_eq!(result.diagnostics.status, "applied_auto");
    assert!(result.diagnostics.applied);
    assert!(!result.diagnostics.review_required);
    assert!(result.diagnostics.occupied_spatial_bin_count >= 8);
    assert_eq!(result.diagnostics.populated_luminance_band_count, 3);
    assert!(after < before * 0.20, "before={before} after={after}");
    assert!(result.image.iter().any(|value| *value < 0.0));
    assert!(result.image.iter().any(|value| *value > 1.0));
    assert!(result.diagnostics.preserves_signed_scene_headroom);
}

#[test]
fn automatic_technical_white_balance_does_not_force_a_saturated_scene_to_gray() {
    let mut image = Array3::<f64>::zeros((48, 64, 3));
    for y in 0..48 {
        for x in 0..64 {
            image[[y, x, 0]] = 0.55 + 0.3 * x as f64 / 63.0;
            image[[y, x, 1]] = 0.08 + 0.05 * y as f64 / 47.0;
            image[[y, x, 2]] = 0.04;
        }
    }

    let result = apply_technical_white_balance(image.clone(), &WhiteBalanceArgs::default());

    assert!(!result.diagnostics.applied);
    assert_eq!(result.image, image);
    assert_eq!(result.diagnostics.status, "insufficient_evidence");
}

#[test]
fn manual_technical_white_balance_is_explicit_and_separate_from_scene_estimation() {
    let image = Array3::<f64>::from_elem((12, 16, 3), 0.4);
    let settings = WhiteBalanceArgs {
        technical_white_balance: TechnicalWhiteBalanceMode::Manual,
        technical_temperature_kelvin: 6500.0,
        technical_tint: 0.15,
        ..WhiteBalanceArgs::default()
    };

    let result = apply_technical_white_balance(image.clone(), &settings);

    assert_eq!(result.diagnostics.status, "applied_manual");
    assert_eq!(result.diagnostics.sample_count, 0);
    assert_eq!(result.diagnostics.manual_temperature_kelvin, Some(6500.0));
    assert_eq!(result.diagnostics.manual_tint, Some(0.15));
    assert_ne!(result.image, image);
}

#[test]
fn creative_temperature_and_tint_are_reversible_render_adjustments() {
    let technical_master = Array3::<f64>::from_elem((8, 10, 3), 0.35);
    let unchanged = apply_creative_white_balance(&technical_master, 0.0, 0.0);
    assert_eq!(unchanged.image, technical_master);
    assert!(!unchanged.diagnostics.applied);

    let warm_magenta = apply_creative_white_balance(&technical_master, 0.75, 0.45);
    let rgb = mean_rgb(&warm_magenta.image);
    assert!(warm_magenta.diagnostics.applied);
    assert!(rgb[0] / rgb[2] > 1.05, "creative RGB was {rgb:?}");
    assert!(rgb[1] < (rgb[0] + rgb[2]) * 0.5, "creative RGB was {rgb:?}");
    assert!(warm_magenta.diagnostics.separated_from_technical_master);
    assert_eq!(technical_master, Array3::<f64>::from_elem((8, 10, 3), 0.35));
}

#[test]
fn owned_creative_white_balance_is_identical_and_reuses_its_input() {
    let image = coherent_neutral_scene_with_cast([0.95, 1.0, 1.08]);
    let allocation = image.as_ptr() as usize;
    let borrowed = apply_creative_white_balance(&image, 0.65, -0.30);
    let owned = apply_creative_white_balance_owned(image, 0.65, -0.30);

    assert_eq!(owned.image.as_ptr() as usize, allocation);
    assert_eq!(owned.image, borrowed.image);
    assert_eq!(owned.diagnostics, borrowed.diagnostics);
}

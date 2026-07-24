use approx::assert_relative_eq;
use nalgebra::{Matrix3, Vector3};
use ndarray::Array3;

fn matrix_from_const(values: &[[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::new(
        values[0][0],
        values[0][1],
        values[0][2],
        values[1][0],
        values[1][1],
        values[1][2],
        values[2][0],
        values[2][1],
        values[2][2],
    )
}

#[test]
fn test_prophoto_matrix_white_point_and_inverse_match_d50() {
    let rgb_white = Vector3::new(1.0, 1.0, 1.0);
    let xyz = matrix_from_const(&scanstitch::constants::PROPHOTO_TO_XYZ_D50) * rgb_white;

    for (actual, expected) in xyz.iter().zip(scanstitch::constants::D50_WHITE.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1e-10);
    }

    let forward = matrix_from_const(&scanstitch::constants::PROPHOTO_TO_XYZ_D50);
    let inverse = matrix_from_const(&scanstitch::constants::XYZ_D50_TO_PROPHOTO);
    let identity = forward * inverse;
    for row in 0..3 {
        for col in 0..3 {
            let expected = if row == col { 1.0 } else { 0.0 };
            assert_relative_eq!(identity[(row, col)], expected, epsilon = 1e-6);
        }
    }
}

#[test]
fn test_d50_xyz_lab_roundtrip_supports_perceptual_output_mapping() {
    for xyz in [
        [0.0, 0.0, 0.0],
        scanstitch::constants::D50_WHITE,
        [0.18, 0.20, 0.09],
        [0.62, 0.31, 0.12],
    ] {
        let lab = scanstitch::colorspace::xyz_d50_to_lab(xyz);
        let reconstructed = scanstitch::colorspace::lab_to_xyz_d50(lab);
        for (actual, expected) in reconstructed.iter().zip(xyz) {
            assert_relative_eq!(actual, &expected, epsilon = 1e-10);
        }
    }
}

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
        dominant_anchor_sample_rejections: scanstitch::colorspace::DominantAnchorSampleRejections {
            total_pixels: 1024,
            accepted_anchor_samples: 768,
            non_finite: 0,
            clipped: 0,
            border: 0,
            film_base_like_edge: 0,
            dust: 0,
            luma_out_of_range: 0,
            low_saturation: 0,
            weak_dominance: 0,
        },
        dominant_anchor_bands: [[64, 128, 64], [64, 128, 64], [64, 128, 64]],
        dominant_anchor_quality: scanstitch::colorspace::DominantAnchorQuality {
            score: 1.0,
            accepted: true,
            reason: "synthetic stable anchors".to_string(),
            channel_populated_band_count: [3, 3, 3],
            channel_dominant_band_fraction: [0.5, 0.5, 0.5],
            channel_mean_dominance_margin: [0.5, 0.5, 0.5],
            channel_stability_score: [1.0, 1.0, 1.0],
            channel_unstable: [false, false, false],
            unstable_channel_count: 0,
            minimum_samples_per_channel: 64,
            minimum_bands: 2,
            max_dominant_band_fraction: 0.92,
            dominance_margin_target: 0.35,
        },
        neutral_sample_bands: [128, 512, 384],
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
        selected_mapping_reason: "synthetic diagnostics".to_string(),
        candidate_scores: Vec::new(),
        candidate_acceptance: Vec::new(),
        selected_candidate: "image_derived_matrix".to_string(),
        selected_candidate_rank: None,
        selected_candidate_score: None,
        selected_quality_score: None,
        technical_safety_score: None,
        color_fidelity_score: None,
        selected_runner_up_quality_delta: None,
        selection_rejections: Vec::new(),
        candidate_risk: if gamut_fallback_used {
            "fallback_only".to_string()
        } else {
            "safe".to_string()
        },
        neutral_sample_rejections: scanstitch::colorspace::NeutralSampleRejections {
            total_pixels: 1024,
            accepted_neutral_samples: 1024,
            non_finite: 0,
            clipped: 0,
            border: 0,
            film_base_like_edge: 0,
            dust: 0,
            luma_out_of_range: 0,
            chroma_threshold: 0,
        },
        reference_patch_evaluation: None,
        neutral_estimate_quality: scanstitch::colorspace::NeutralEstimateQuality {
            score: 1.0,
            accepted: true,
            broad_support: true,
            reason: "synthetic neutral estimate".to_string(),
            sample_count: 1024,
            sample_fraction: 1.0,
            band_counts: [128, 512, 384],
            populated_band_count: 3,
            dominant_band_fraction: 0.5,
            minimum_samples: 64,
            minimum_bands: 2,
        },
        neutral_balance_scale: [1.0, 1.0, 1.0],
        neutral_trim_scale: [1.0, 1.0, 1.0],
        neutral_trim_applied: false,
        neutral_trim_before_after: scanstitch::colorspace::NeutralTrimDiagnostics {
            applied: false,
            scale: [1.0, 1.0, 1.0],
            reason: "synthetic neutral trim".to_string(),
            before: None,
            after: None,
            neutral_delta_reduced: false,
            neutral_band_delta_worsened: false,
            low_clipping_increased: false,
            high_clipping_increased: false,
            preserved_ratio_decreased: false,
        },
        calibration_acceptance: scanstitch::colorspace::CalibrationAcceptanceDiagnostics {
            status: "not_applicable".to_string(),
            reason: "synthetic diagnostics".to_string(),
            color_mode: "auto",
            preferred_candidate: None,
            preferred_candidate_quality_score: None,
            image_derived_quality_score: None,
            beats_image_derived: None,
            within_negative_gamut_limits: None,
            forced_by_color_mode: false,
        },
        neutral_safety_rescue:
            scanstitch::colorspace::NeutralSafetyRescueDiagnostics::not_evaluated(
                "synthetic diagnostics",
            ),
        nonlinear_color_model: None,
        pre_scale_preserved_ratio: 1.0,
        post_scale_preserved_ratio: 1.0,
        image_matrix_pre_scale_clipped_low_ratio: low_clip,
        image_matrix_pre_scale_clipped_high_ratio: [0.0, 0.0, 0.0],
        image_matrix_exposure_scale: 1.0,
        image_matrix_pre_scale_preserved_ratio: 1.0,
        image_matrix_neutral_balance_delta: [0.0, 0.0, 0.0],
        calibrated_profile_pre_scale_clipped_low_ratio: None,
        calibrated_profile_pre_scale_clipped_high_ratio: None,
        calibrated_profile_exposure_scale: None,
        calibrated_profile_pre_scale_preserved_ratio: None,
        calibrated_profile_neutral_balance_delta: None,
    }
}

fn synthetic_calibration_profile() -> scanstitch::color_calibration::CalibrationProfile {
    let json = serde_json::json!({
        "schema_version": 1,
        "profile_id": "synthetic-prophoto-d50",
        "source_space": { "name": "synthetic linear scanner RGB", "encoding": "linear" },
        "scanner": { "make": "Synthetic", "model": "Unit Test" },
        "film": { "stock": "Synthetic negative", "process": "C-41" },
        "target": { "type": "synthetic_color_checker", "illuminant": "D50" },
        "reference": { "dataset": "synthetic" },
        "whitepoint": [0.9642, 1.0, 0.8251],
        "work_to_xyz": [
            [0.7977, 0.1352, 0.0313],
            [0.2880, 0.7119, 0.0001],
            [0.0000, 0.0000, 0.8251]
        ],
        "confidence": 0.95
    })
    .to_string();
    scanstitch::color_calibration::parse_profile_json(&json, None)
        .profile
        .expect("synthetic calibration profile")
}

fn nonlinear_target_patches(
    count: usize,
    prefix: &str,
    seed_offset: usize,
) -> Vec<scanstitch::color_calibration::TargetPatch> {
    let coefficients = [
        [0.65, 0.25, 0.03],
        [0.14, 0.69, 0.08],
        [0.05, 0.10, 0.76],
        [0.055, -0.030, 0.010],
        [-0.025, 0.012, 0.045],
        [0.012, 0.040, -0.018],
    ];
    (0..count)
        .map(|index| {
            let seed = index + seed_offset;
            let code =
                |multiplier: usize, add: usize| ((seed * multiplier + add) % 997) as f64 / 996.0;
            let rgb = [
                0.025 + 0.95 * code(173, 31),
                0.025 + 0.95 * code(379, 97),
                0.025 + 0.95 * code(613, 211),
            ];
            let basis = scanstitch::color_calibration::root_polynomial_basis_values(2, rgb)
                .expect("degree-two basis");
            let reference_xyz = std::array::from_fn(|channel| {
                basis
                    .iter()
                    .zip(coefficients)
                    .map(|(term, coefficient)| term * coefficient[channel])
                    .sum()
            });
            scanstitch::color_calibration::TargetPatch {
                patch_id: Some(format!("{prefix}-{index:03}")),
                scanner_xy: None,
                source_rgb: rgb,
                reference_xyz,
            }
        })
        .collect()
}

fn nonlinear_calibration_profile() -> (
    scanstitch::color_calibration::CalibrationProfile,
    Vec<scanstitch::color_calibration::TargetPatch>,
) {
    let training = nonlinear_target_patches(48, "runtime-train", 0);
    let held_out = nonlinear_target_patches(36, "runtime-held", 409);
    let matrix = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training,
        &held_out,
        "runtime_matrix_baseline",
        Some(scanstitch::constants::D50_WHITE),
        None,
    )
    .expect("runtime matrix fit");
    let model =
        scanstitch::color_calibration::fit_root_polynomial_color_model_from_disjoint_patches(
            "runtime-nonlinear",
            &training,
            &held_out,
            &matrix.matrix,
        )
        .expect("runtime nonlinear fit")
        .selected_model
        .expect("runtime nonlinear model");
    let mut profile = synthetic_calibration_profile();
    profile.schema_version = 2;
    profile.work_to_xyz = matrix.matrix;
    profile.whitepoint = scanstitch::constants::D50_WHITE;
    profile.confidence = matrix.confidence;
    profile.matrix_condition_number = matrix.matrix_condition_number;
    profile.fit = Some(matrix.fit);
    profile.target_patches = held_out.clone();
    profile.application_mode =
        scanstitch::color_calibration::CalibrationApplicationMode::DirectProfile;
    profile.color_model = Some(model);
    profile.color_model_post_xyz = None;
    (profile, held_out)
}

fn residual_lut_target_patches(
    count: usize,
    prefix: &str,
    seed_offset: usize,
    include_domain_corners: bool,
) -> Vec<scanstitch::color_calibration::TargetPatch> {
    let matrix = [[0.65, 0.14, 0.05], [0.25, 0.69, 0.10], [0.03, 0.08, 0.76]];
    (0..count)
        .map(|index| {
            let seed = index + seed_offset;
            let code =
                |multiplier: usize, add: usize| ((seed * multiplier + add) % 997) as f64 / 996.0;
            let source_rgb = if include_domain_corners && index < 8 {
                [
                    if index & 1 == 0 { 0.025 } else { 0.975 },
                    if index & 2 == 0 { 0.025 } else { 0.975 },
                    if index & 4 == 0 { 0.025 } else { 0.975 },
                ]
            } else {
                [
                    0.04 + 0.92 * code(173, 31),
                    0.04 + 0.92 * code(379, 97),
                    0.04 + 0.92 * code(613, 211),
                ]
            };
            let mut reference_xyz = std::array::from_fn(|row| {
                matrix[row][0] * source_rgb[0]
                    + matrix[row][1] * source_rgb[1]
                    + matrix[row][2] * source_rgb[2]
            });
            let normalized = source_rgb.map(|value| (value - 0.025) / 0.95);
            let shape = 64.0
                * normalized[0]
                * (1.0 - normalized[0])
                * normalized[1]
                * (1.0 - normalized[1])
                * normalized[2]
                * (1.0 - normalized[2]);
            let residual = [
                0.055 * shape * (0.70 + 0.30 * (2.0 * normalized[0] - 1.0)),
                -0.040 * shape * (0.75 + 0.25 * (2.0 * normalized[1] - 1.0)),
                0.050 * shape * (0.65 + 0.35 * (2.0 * normalized[2] - 1.0)),
            ];
            for channel in 0..3 {
                reference_xyz[channel] += residual[channel];
            }
            scanstitch::color_calibration::TargetPatch {
                patch_id: Some(format!("{prefix}-{index:03}")),
                scanner_xy: None,
                source_rgb,
                reference_xyz,
            }
        })
        .collect()
}

fn residual_lut_calibration_profile() -> (
    scanstitch::color_calibration::CalibrationProfile,
    Vec<scanstitch::color_calibration::TargetPatch>,
) {
    let training = residual_lut_target_patches(180, "runtime-lut-train", 0, true);
    let held_out = residual_lut_target_patches(80, "runtime-lut-held", 431, false);
    let matrix = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training,
        &held_out,
        "runtime_lut_matrix_baseline",
        Some(scanstitch::constants::D50_WHITE),
        None,
    )
    .expect("runtime LUT matrix fit");
    let model =
        scanstitch::color_calibration::fit_residual_lut_3d_color_model_from_disjoint_patches(
            "runtime-residual-lut",
            &training,
            &held_out,
            &matrix.matrix,
            None,
        )
        .expect("runtime residual LUT fit")
        .selected_model
        .expect("runtime residual LUT model");
    let mut profile = synthetic_calibration_profile();
    profile.schema_version = 2;
    profile.work_to_xyz = matrix.matrix;
    profile.whitepoint = scanstitch::constants::D50_WHITE;
    profile.confidence = matrix.confidence;
    profile.matrix_condition_number = matrix.matrix_condition_number;
    profile.fit = Some(matrix.fit);
    profile.target_patches = held_out.clone();
    profile.application_mode =
        scanstitch::color_calibration::CalibrationApplicationMode::DirectProfile;
    profile.lut_3d_model = Some(model);
    profile.color_model_post_xyz = None;
    (profile, held_out)
}

fn prophoto_matrix_rows() -> [[f64; 3]; 3] {
    let matrix = scanstitch::colorspace::prophoto_to_xyz_d50_matrix();
    [
        [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
        [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
        [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
    ]
}

fn scanner_prior_profile() -> scanstitch::color_calibration::CalibrationProfile {
    let mut profile = synthetic_calibration_profile();
    profile.application_mode =
        scanstitch::color_calibration::CalibrationApplicationMode::ScannerConstrainedImageAdaptation;
    profile.work_to_xyz = prophoto_matrix_rows();
    profile.whitepoint = scanstitch::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;
    profile
}

#[test]
fn test_tone_protection_keeps_shadow_cleanup_for_weak_neutral_review() {
    let mut diagnostics = diagnostics_with_image_matrix_low_clip([0.0, 0.0, 0.0], false);
    diagnostics.candidate_risk = "review_neutral_support".to_string();
    diagnostics.selected_quality_score = Some(4.2);
    diagnostics.neutral_estimate_quality.accepted = false;
    diagnostics.neutral_estimate_quality.reason =
        "synthetic neutral support is too narrow".to_string();

    let protection =
        scanstitch::tonemap::ToneColorProtection::from_colorspace_diagnostics(&diagnostics);

    assert_eq!(
        protection.policy,
        scanstitch::tonemap::ToneColorProtectionPolicy::WeakNeutralBoundedNeutralCleanup
    );
    assert!(protection.highlight_neutral_chroma_enabled);
    assert!(protection.midtone_neutral_chroma_enabled);
    assert!(protection.shadow_chroma_enabled);
    assert_eq!(protection.color_trust_state(), "limited_weak_neutral");
}

#[test]
fn test_tone_protection_keeps_shadow_cleanup_for_anchor_review() {
    let mut diagnostics = diagnostics_with_image_matrix_low_clip([0.0, 0.0, 0.0], false);
    diagnostics.candidate_risk = "review_anchor_support".to_string();
    diagnostics.selected_quality_score = Some(2.2);
    diagnostics.neutral_estimate_quality.accepted = true;
    diagnostics.neutral_estimate_quality.reason =
        "synthetic neutral support is accepted".to_string();

    let protection =
        scanstitch::tonemap::ToneColorProtection::from_colorspace_diagnostics(&diagnostics);

    assert_eq!(
        protection.policy,
        scanstitch::tonemap::ToneColorProtectionPolicy::ReviewBoundedNeutralAndShadowCleanup
    );
    assert!(protection.highlight_neutral_chroma_enabled);
    assert!(protection.shadow_chroma_enabled);
    assert_eq!(protection.color_trust_state(), "review_required");
}

#[test]
fn test_tone_protection_keeps_bounded_neutral_cleanup_for_model_review() {
    let mut diagnostics = diagnostics_with_image_matrix_low_clip([0.0, 0.0, 0.0], false);
    diagnostics.candidate_risk = "review_model_plausibility".to_string();
    diagnostics.selected_quality_score = Some(2.8);
    diagnostics.neutral_estimate_quality.accepted = true;
    diagnostics.neutral_estimate_quality.reason =
        "synthetic neutral support is accepted".to_string();

    let protection =
        scanstitch::tonemap::ToneColorProtection::from_colorspace_diagnostics(&diagnostics);

    assert_eq!(
        protection.policy,
        scanstitch::tonemap::ToneColorProtectionPolicy::ReviewBoundedNeutralAndShadowCleanup
    );
    assert!(protection.highlight_neutral_chroma_enabled);
    assert!(protection.midtone_neutral_chroma_enabled);
    assert!(protection.shadow_chroma_enabled);
    assert_eq!(protection.color_trust_state(), "review_required");
}

fn reference_fit_patches_for_matrix(
    matrix_to_prophoto: nalgebra::Matrix3<f64>,
) -> Vec<scanstitch::color_calibration::TargetPatch> {
    let prophoto_to_xyz = scanstitch::colorspace::prophoto_to_xyz_d50_matrix();
    [
        [0.78, 0.12, 0.10],
        [0.14, 0.72, 0.16],
        [0.12, 0.18, 0.76],
        [0.25, 0.25, 0.25],
        [0.52, 0.52, 0.52],
        [0.82, 0.82, 0.82],
    ]
    .into_iter()
    .enumerate()
    .map(|(idx, source_rgb)| {
        let xyz = prophoto_to_xyz
            * (matrix_to_prophoto * Vector3::new(source_rgb[0], source_rgb[1], source_rgb[2]));
        scanstitch::color_calibration::TargetPatch {
            patch_id: Some(format!("patch-{idx}")),
            scanner_xy: None,
            source_rgb,
            reference_xyz: [xyz[0], xyz[1], xyz[2]],
        }
    })
    .collect()
}

fn sparse_strong_anchor_image() -> ndarray::Array3<f64> {
    let mut img = ndarray::Array3::<f64>::zeros((40, 40, 3));
    for y in 0..40 {
        for x in 0..40 {
            let pixel = if x < 2 {
                [0.90, 0.22, 0.12]
            } else if x < 4 {
                [0.18, 0.82, 0.16]
            } else if x < 6 {
                [0.12, 0.20, 0.78]
            } else {
                [0.58, 0.56, 0.54]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }
    img
}

fn broad_neutral_band_image(width: usize, height: usize) -> ndarray::Array3<f64> {
    let mut img = ndarray::Array3::<f64>::zeros((height, width, 3));
    for y in 0..height {
        let value = if y < height / 3 {
            0.18
        } else if y < 2 * height / 3 {
            0.50
        } else {
            0.82
        };
        for x in 0..width {
            for c in 0..3 {
                img[[y, x, c]] = value;
            }
        }
    }
    img
}

fn positive_rgb_detail_image(width: usize, height: usize) -> ndarray::Array3<f64> {
    let mut img = ndarray::Array3::<f64>::zeros((height, width, 3));
    for y in 0..height {
        for x in 0..width {
            let t = x as f64 / (width - 1) as f64;
            let y_t = y as f64 / (height - 1) as f64;
            img[[y, x, 0]] = 0.08 + 0.74 * t;
            img[[y, x, 1]] = 0.10 + 0.64 * y_t;
            img[[y, x, 2]] = 0.16 + 0.58 * (1.0 - t * 0.5);
        }
    }
    img
}

#[test]
fn test_unprofiled_positive_rgb_passthrough_requires_color_review() {
    let img = positive_rgb_detail_image(96, 64);
    let result =
        scanstitch::colorspace::map_positive_scan_rgb_to_prophoto_d50_with_diagnostics(&img);

    assert_eq!(
        result.diagnostics.mapping_strategy,
        "positive_rgb_passthrough"
    );
    assert_eq!(
        result.diagnostics.selected_candidate,
        "positive_scan_rgb_passthrough"
    );
    assert_eq!(result.diagnostics.candidate_risk, "review_unprofiled_input");
    assert_eq!(
        scanstitch::colorspace::tone_color_trust_state(&result.diagnostics),
        "review_required"
    );
    assert_eq!(result.diagnostics.pre_scale_preserved_ratio, 1.0);
    assert_eq!(result.diagnostics.post_scale_preserved_ratio, 1.0);
    assert_eq!(result.diagnostics.channel_anchor_low_support, [false; 3]);
    assert!(result.diagnostics.neutral_estimate_quality.accepted);
    assert!(!result.diagnostics.neutral_safety_rescue.evaluated);
    assert!(!result.diagnostics.neutral_safety_rescue.applied);

    let left =
        result.prophoto[[32, 8, 0]] + result.prophoto[[32, 8, 1]] + result.prophoto[[32, 8, 2]];
    let right =
        result.prophoto[[32, 88, 0]] + result.prophoto[[32, 88, 1]] + result.prophoto[[32, 88, 2]];
    assert!(
        right > left,
        "positive RGB passthrough must preserve non-inverted luminance ordering"
    );
}

#[test]
fn test_profiled_positive_rgb_is_a_trusted_linear_prophoto_input() {
    let img = positive_rgb_detail_image(96, 64);
    let result =
        scanstitch::colorspace::map_profiled_positive_scan_rgb_to_prophoto_d50_with_diagnostics(
            &img,
        );

    assert_eq!(
        result.diagnostics.mapping_strategy,
        "embedded_icc_to_linear_prophoto"
    );
    assert_eq!(
        result.diagnostics.selected_candidate,
        "profiled_positive_scan_rgb"
    );
    assert_eq!(result.diagnostics.candidate_risk, "safe");
    assert_eq!(
        scanstitch::colorspace::tone_color_trust_state(&result.diagnostics),
        "trusted"
    );
    assert_eq!(result.diagnostics.pre_scale_preserved_ratio, 1.0);
    assert_eq!(result.diagnostics.post_scale_preserved_ratio, 1.0);
    assert!(result
        .diagnostics
        .selected_mapping_reason
        .contains("ICC device-to-PCS"));
    assert!(!result.diagnostics.neutral_safety_rescue.evaluated);
    assert!(!result.diagnostics.neutral_safety_rescue.applied);
}

#[test]
fn test_forced_neutral_fallback_cannot_become_trusted_colour() {
    let img = broad_neutral_band_image(72, 72);
    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            None,
            scanstitch::colorspace::ColorMode::Neutral,
        )
        .unwrap();

    assert_eq!(result.diagnostics.candidate_risk, "fallback_only");
    assert!(!result.diagnostics.neutral_safety_rescue.evaluated);
    assert!(!result.diagnostics.neutral_safety_rescue.applied);
    assert!(result
        .diagnostics
        .neutral_safety_rescue
        .reason
        .contains("auto-only"));
    assert_eq!(
        scanstitch::colorspace::tone_color_trust_state(&result.diagnostics),
        "review_required"
    );
    let protection =
        scanstitch::tonemap::ToneColorProtection::from_colorspace_diagnostics(&result.diagnostics);
    assert_eq!(protection.color_trust_state(), "review_required");
    assert_eq!(
        protection.policy.as_str(),
        "disabled_color_candidate_review"
    );
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
fn test_direct_density_render_fallback_rejects_neutral_regression() {
    let ica = diagnostics_with_image_matrix_low_clip([0.43, 0.16, 0.13], true);
    let mut direct = diagnostics_with_image_matrix_low_clip([0.0, 0.0, 0.001], false);
    direct.image_matrix_neutral_balance_delta = [0.04, -0.03, 0.02];

    assert!(
        scanstitch::colorspace::direct_density_render_fallback_reason(&ica, &direct).is_none(),
        "direct density candidate should not replace ICA when neutral balance regresses"
    );
}

#[test]
fn test_direct_density_render_fallback_accepts_material_quality_win() {
    let mut ica = diagnostics_with_image_matrix_low_clip([0.01, 0.0, 0.0], false);
    ica.selected_quality_score = Some(1.20);
    ica.candidate_risk = "review_quality_score".to_string();
    let mut direct = diagnostics_with_image_matrix_low_clip([0.005, 0.0, 0.0], false);
    direct.selected_quality_score = Some(0.70);
    direct.candidate_risk = "safe".to_string();

    let reason = scanstitch::colorspace::direct_density_render_fallback_reason(&ica, &direct)
        .expect("direct density quality fallback");

    assert!(reason.contains("materially stronger colorspace candidate"));
    assert!(reason.contains("quality score"));
}

#[test]
fn test_direct_density_render_fallback_accepts_trusted_close_score() {
    let mut ica = diagnostics_with_image_matrix_low_clip([0.01, 0.0, 0.0], false);
    ica.selected_quality_score = Some(1.20);
    ica.candidate_risk = "review_anchor_support".to_string();
    let mut direct = diagnostics_with_image_matrix_low_clip([0.005, 0.0, 0.0], false);
    direct.selected_quality_score = Some(1.30);
    direct.candidate_risk = "safe".to_string();

    let reason = scanstitch::colorspace::direct_density_render_fallback_reason(&ica, &direct)
        .expect("direct density fallback for trusted close-score candidate");

    assert!(reason.contains("color review state"));
    assert!(reason.contains("review_required"));
    assert!(reason.contains("trusted"));
}

#[test]
fn test_direct_density_render_fallback_accepts_trusted_high_preserved_gamut() {
    let mut ica = diagnostics_with_image_matrix_low_clip([0.04, 0.01, 0.0], false);
    ica.selected_quality_score = Some(1.20);
    ica.candidate_risk = "review_anchor_support".to_string();
    ica.post_scale_preserved_ratio = 0.985;
    let mut direct = diagnostics_with_image_matrix_low_clip([0.01, 0.0, 0.0], false);
    direct.selected_quality_score = Some(1.30);
    direct.candidate_risk = "safe".to_string();
    direct.post_scale_preserved_ratio = 0.972;

    let reason = scanstitch::colorspace::direct_density_render_fallback_reason(&ica, &direct)
        .expect("direct density fallback should accept trusted high-preserved-gamut candidate");

    assert!(reason.contains("direct density transmittance"));
}

#[test]
fn test_direct_density_render_fallback_rejects_quality_win_with_worse_risk() {
    let mut ica = diagnostics_with_image_matrix_low_clip([0.01, 0.0, 0.0], false);
    ica.selected_quality_score = Some(1.20);
    ica.candidate_risk = "safe".to_string();
    let mut direct = diagnostics_with_image_matrix_low_clip([0.005, 0.0, 0.0], false);
    direct.selected_quality_score = Some(0.70);
    direct.candidate_risk = "review_gamut".to_string();

    assert!(
        scanstitch::colorspace::direct_density_render_fallback_reason(&ica, &direct).is_none(),
        "direct density should not replace ICA when the stronger score carries worse risk"
    );
}

#[test]
fn test_direct_density_render_prefers_safe_physical_prior_with_close_quality() {
    let mut ica = diagnostics_with_image_matrix_low_clip([0.012, 0.003, 0.0], false);
    ica.selected_quality_score = Some(1.20);
    ica.candidate_risk = "safe".to_string();
    ica.post_scale_preserved_ratio = 0.991;
    let mut direct = diagnostics_with_image_matrix_low_clip([0.013, 0.003, 0.0], false);
    direct.selected_quality_score = Some(1.85);
    direct.candidate_risk = "safe".to_string();
    direct.post_scale_preserved_ratio = 0.990;

    let reason = scanstitch::colorspace::direct_density_render_fallback_reason(&ica, &direct)
        .expect("safe physical direct-density prior");

    assert!(reason.contains("physically grounded negative-response reconstruction"));
    assert!(reason.contains("blind ICA is underconstrained"));
}

#[test]
fn test_direct_density_render_physical_prior_rejects_material_gamut_regression() {
    let mut ica = diagnostics_with_image_matrix_low_clip([0.01, 0.0, 0.0], false);
    ica.selected_quality_score = Some(1.20);
    ica.candidate_risk = "safe".to_string();
    ica.post_scale_preserved_ratio = 0.995;
    let mut direct = diagnostics_with_image_matrix_low_clip([0.01, 0.0, 0.0], false);
    direct.selected_quality_score = Some(1.30);
    direct.candidate_risk = "safe".to_string();
    direct.post_scale_preserved_ratio = 0.960;

    assert!(
        scanstitch::colorspace::direct_density_render_fallback_reason(&ica, &direct).is_none(),
        "physical prior must not override a materially safer ICA color mapping"
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
                    (0.0..=1.0).contains(&v),
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
fn test_colorspace_result_preserves_scene_referred_highlight_headroom() {
    let mut img = ndarray::Array3::<f64>::from_elem((20, 20, 3), 0.50);
    for c in 0..3 {
        img[[0, 0, c]] = 1.80;
    }
    let mut profile = synthetic_calibration_profile();
    profile.work_to_xyz = prophoto_matrix_rows();
    profile.whitepoint = scanstitch::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;

    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Calibrated,
        )
        .unwrap();

    assert!(
        result.prophoto[[0, 0, 0]] > 1.0,
        "selected ProPhoto render buffer should preserve sparse highlight headroom for tone mapping"
    );
    assert!(
        result.diagnostics.post_scale_clipped_high_ratio[0] > 0.0,
        "diagnostics should still report display-gamut highlight excursions"
    );
}

#[test]
fn test_gamut_clipping_map_reports_selected_transform_clip_pixels() {
    let mut img = ndarray::Array3::<f64>::zeros((2, 2, 3));
    img[[0, 0, 0]] = 1.2;
    img[[0, 0, 1]] = 0.5;
    img[[0, 0, 2]] = 0.5;
    img[[0, 1, 0]] = -0.1;
    img[[0, 1, 1]] = 0.2;
    img[[0, 1, 2]] = 0.2;
    img[[1, 0, 0]] = 0.5;
    img[[1, 0, 1]] = 0.5;
    img[[1, 0, 2]] = 0.5;
    img[[1, 1, 0]] = 1.2;
    img[[1, 1, 1]] = -0.2;
    img[[1, 1, 2]] = 0.5;

    let mut diagnostics = diagnostics_with_image_matrix_low_clip([0.0; 3], false);
    diagnostics.source_white = scanstitch::constants::D50_WHITE;
    diagnostics.work_to_xyz = prophoto_matrix_rows();
    diagnostics.exposure_scale = 1.0;
    diagnostics.neutral_trim_scale = [1.0; 3];

    let (map, map_diagnostics) =
        scanstitch::colorspace::build_gamut_clipping_map(&img, &diagnostics);

    assert_eq!(map.dim(), (2, 2, 3));
    assert_relative_eq!(map_diagnostics.high_clipped_ratio[0], 0.5);
    assert_relative_eq!(map_diagnostics.low_clipped_ratio[0], 0.25);
    assert_relative_eq!(map_diagnostics.low_clipped_ratio[1], 0.25);
    assert_relative_eq!(map_diagnostics.any_clipped_ratio, 0.75);
    assert_relative_eq!(map_diagnostics.preserved_ratio, 0.25);
    assert!(
        map[[0, 0, 0]] > map[[0, 0, 2]],
        "high-clipped pixel should be red-coded"
    );
    assert!(
        map[[0, 1, 2]] > map[[0, 1, 0]],
        "low-clipped pixel should be blue-coded"
    );
    assert!(
        map[[1, 0, 1]] > map[[1, 0, 0]] && map[[1, 0, 1]] > map[[1, 0, 2]],
        "safe pixel should be green-coded"
    );
}

#[test]
fn test_colorspace_prefers_valid_calibration_profile() {
    let mut img = ndarray::Array3::<f64>::zeros((24, 24, 3));
    for y in 0..24 {
        for x in 0..24 {
            let pixel = if x < 8 {
                [0.70, 0.20, 0.15]
            } else if x < 16 {
                [0.18, 0.68, 0.22]
            } else {
                [0.16, 0.24, 0.72]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }
    let profile = synthetic_calibration_profile();

    let result = scanstitch::colorspace::map_to_prophoto_d50_with_calibration_diagnostics(
        &img,
        Some(&profile),
    );
    let diagnostics = result.diagnostics;

    assert_eq!(diagnostics.mapping_strategy, "calibrated_profile");
    assert_eq!(diagnostics.source_white, profile.whitepoint);
    assert_eq!(diagnostics.work_to_xyz, profile.work_to_xyz);
    assert!(diagnostics
        .selected_mapping_reason
        .contains("external calibration profile"));
    assert!(
        diagnostics
            .calibrated_profile_exposure_scale
            .expect("calibrated candidate exposure")
            >= 1.0
    );
    assert!(diagnostics.image_matrix_exposure_scale >= 1.0);
    assert!(diagnostics.neutral_safety_rescue.evaluated);
    assert!(!diagnostics.neutral_safety_rescue.applied);
    assert_eq!(
        diagnostics
            .neutral_safety_rescue
            .matrix_candidate_kind
            .as_deref(),
        Some("calibrated_direct")
    );
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
    for (r, prior_row) in prior_rows.iter().enumerate() {
        for (c, expected_value) in prior_row.iter().enumerate() {
            max_delta = max_delta.max((diagnostics.work_to_xyz[r][c] - *expected_value).abs());
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
        !diagnostics.weak_anchor_fallback_used,
        "weak anchors should be penalized rather than automatically forcing fallback: {:?}",
        diagnostics
    );
    assert_eq!(diagnostics.mapping_strategy, "image_derived_matrix");
    assert_eq!(
        diagnostics.selected_candidate, "image_derived_matrix",
        "candidate scores: {:?}",
        diagnostics.candidate_scores
    );
    assert!(
        diagnostics.candidate_scores.iter().any(|score| {
            score.candidate == "image_derived_matrix"
                && score.channel_anchor_low_support == [false, true, false]
                && !score.rejected
                && score.rank.is_some()
        }),
        "candidate score should retain weak-anchor diagnostics: {:?}",
        diagnostics.candidate_scores
    );
}

#[test]
fn test_colorspace_reports_strong_channel_anchors_without_weak_warning() {
    let img = sparse_strong_anchor_image();

    let result = scanstitch::colorspace::map_to_prophoto_d50_with_diagnostics(&img);
    let diagnostics = result.diagnostics;

    assert_eq!(diagnostics.channel_anchor_counts, [80, 80, 80]);
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
    assert!(
        !diagnostics.gamut_fallback_used,
        "strong but sparse anchors should leave the matrix candidate inside safety limits: {:?}",
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

    let mut max_render_value = 0.0f64;
    for value in result.prophoto.iter() {
        assert!(
            value.is_finite() && *value >= 0.0,
            "mapped value should remain finite and non-negative before tone mapping: {}",
            value
        );
        max_render_value = max_render_value.max(*value);
    }
    assert!(
        max_render_value > 1.0,
        "scene-referred render buffer should preserve sparse highlight excursions for tone mapping"
    );
    let display_clamped = scanstitch::colorspace::map_to_prophoto_d50(&img);
    assert!(
        display_clamped
            .iter()
            .all(|value| (0.0..=1.0).contains(value)),
        "legacy map_to_prophoto_d50 convenience output should remain display-clamped"
    );
}

#[test]
fn test_colorspace_uses_gamut_safe_image_blend_before_neutral_fallback() {
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

    assert_eq!(
        diagnostics.mapping_strategy,
        "gamut_safe_image_matrix_blend",
        "expected the engine to preserve image chroma through a safety blend before using neutral fallback"
    );
    assert_eq!(
        diagnostics.selected_candidate,
        "gamut_safe_image_matrix_blend"
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
    assert!(
        diagnostics
            .candidate_acceptance
            .iter()
            .any(
                |candidate| candidate.candidate == "neutral_balance_fallback"
                    && candidate.status == "available_fallback"
            ),
        "neutral fallback should remain available behind the gamut-safe image blend"
    );

    for value in result.prophoto.iter() {
        assert!(
            value.is_finite() && *value >= 0.0,
            "mapped value should stay finite and non-negative after safety blend: {}",
            value
        );
    }
}

#[test]
fn test_scanner_prior_beats_weak_anchor_image_candidate_when_safe() {
    let mut img = ndarray::Array3::<f64>::zeros((24, 24, 3));
    for y in 0..24 {
        for x in 0..24 {
            let pixel = if x < 3 {
                [0.90, 0.22, 0.12]
            } else if x < 6 {
                [0.12, 0.20, 0.78]
            } else {
                [0.58, 0.56, 0.54]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }
    let profile = scanner_prior_profile();

    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Auto,
        )
        .unwrap();
    let diagnostics = result.diagnostics;

    assert_eq!(diagnostics.channel_anchor_low_support, [false, true, false]);
    assert_eq!(
        diagnostics.selected_candidate,
        "scanner_prior_image_adaptation"
    );
    assert!(diagnostics
        .candidate_acceptance
        .iter()
        .any(
            |candidate| candidate.candidate == "scanner_prior_image_adaptation"
                && candidate.status == "selected"
        ));
    assert!(diagnostics.selected_candidate_rank.is_some());
    assert_eq!(
        diagnostics.mapping_strategy,
        "scanner_constrained_image_derived_matrix"
    );
    assert!(diagnostics.neutral_safety_rescue.evaluated);
    assert!(!diagnostics.neutral_safety_rescue.applied);
    assert_eq!(
        diagnostics
            .neutral_safety_rescue
            .matrix_candidate_kind
            .as_deref(),
        Some("scanner_prior")
    );
    assert!(diagnostics
        .neutral_safety_rescue
        .reason
        .contains("protected"));
    assert!(
        !diagnostics.weak_anchor_fallback_used,
        "scanner-prior candidate should be selected without weak-anchor fallback: {:?}",
        diagnostics
    );
}

#[test]
fn test_auto_mode_ranks_neutral_fallback_behind_safe_image_matrix() {
    let img = sparse_strong_anchor_image();

    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            None,
            scanstitch::colorspace::ColorMode::Auto,
        )
        .unwrap();
    let diagnostics = result.diagnostics;

    assert_eq!(
        diagnostics.selected_candidate,
        "gamut_trusted_image_matrix_blend"
    );
    assert_eq!(diagnostics.selected_candidate_rank, Some(1));
    assert!(diagnostics.neutral_safety_rescue.evaluated);
    assert!(!diagnostics.neutral_safety_rescue.applied);
    assert!(diagnostics
        .neutral_safety_rescue
        .reason
        .contains("not applied"));
    let selected_score = diagnostics
        .candidate_scores
        .iter()
        .find(|candidate| candidate.selected)
        .expect("selected candidate");
    let neutral_score = diagnostics
        .candidate_scores
        .iter()
        .find(|candidate| candidate.candidate == "neutral_balance_fallback")
        .expect("neutral fallback candidate");
    assert!(
        neutral_score.quality_score > selected_score.quality_score,
        "neutral fallback should not outrank a safe matrix candidate: selected={:?}, neutral={:?}",
        selected_score,
        neutral_score
    );
    assert!(diagnostics.candidate_acceptance.iter().any(|candidate| {
        candidate.candidate == "image_derived_matrix" && candidate.status == "accepted_runner_up"
    }));
    assert!(diagnostics
        .candidate_acceptance
        .iter()
        .any(
            |candidate| candidate.candidate == "neutral_balance_fallback"
                && candidate.status == "available_fallback"
        ));
}

#[test]
fn test_candidate_scoring_reports_rendered_tone_evidence() {
    let img = broad_neutral_band_image(36, 36);

    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            None,
            scanstitch::colorspace::ColorMode::Auto,
        )
        .unwrap();
    let diagnostics = result.diagnostics;
    let selected_score = diagnostics
        .candidate_scores
        .iter()
        .find(|candidate| candidate.selected)
        .expect("selected candidate score");
    let rendered_tone = selected_score
        .rendered_tone_quality
        .as_ref()
        .expect("candidate rendered-tone diagnostics");

    assert!(rendered_tone.sample_count > 0);
    assert!(rendered_tone.shadow_saturation_p95.is_finite());
    assert!(rendered_tone.midtone_saturation_p95.is_finite());
    assert!(rendered_tone.bright_neutral_saturation_p95.is_finite());
    assert!(rendered_tone
        .tone_post_chroma_clipped_high_total
        .is_finite());
    assert!(selected_score.quality_components.rendered_tone_penalty >= 0.0);
    assert!(
        selected_score
            .quality_components
            .tone_chroma_cleanup_penalty
            >= 0.0
    );
    assert!(
        selected_score.color_fidelity_score
            >= selected_score.quality_components.rendered_tone_penalty
                + selected_score
                    .quality_components
                    .tone_chroma_cleanup_penalty,
        "rendered-tone evidence should be included in color fidelity score: {:?}",
        selected_score
    );
}

#[test]
fn test_calibrated_candidate_loses_when_materially_unsafe() {
    let img = sparse_strong_anchor_image();
    let mut profile = synthetic_calibration_profile();
    profile.work_to_xyz = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    profile.whitepoint = scanstitch::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;

    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Auto,
        )
        .unwrap();
    let diagnostics = result.diagnostics;

    assert_ne!(diagnostics.mapping_strategy, "calibrated_profile");
    assert!(diagnostics
        .candidate_scores
        .iter()
        .any(|score| { score.candidate == "calibrated_direct_profile" && score.rejected }));
    assert!(diagnostics
        .candidate_acceptance
        .iter()
        .any(
            |candidate| candidate.candidate == "calibrated_direct_profile"
                && candidate.status == "rejected_safety"
                && candidate.reason.contains("negative channel values")
        ));
}

#[test]
fn test_scanner_prior_candidate_loses_when_materially_unsafe() {
    let img = sparse_strong_anchor_image();
    let mut profile = scanner_prior_profile();
    profile.work_to_xyz = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    profile.whitepoint = scanstitch::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;

    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Auto,
        )
        .unwrap();
    let diagnostics = result.diagnostics;

    assert_ne!(
        diagnostics.selected_candidate,
        "scanner_prior_image_adaptation"
    );
    assert_ne!(diagnostics.selected_candidate, "neutral_balance_fallback");
    assert!(
        diagnostics.selected_candidate.contains("image_matrix")
            || diagnostics.selected_candidate == "image_derived_matrix",
        "unsafe scanner prior should fall back to an image-derived matrix candidate, got {}",
        diagnostics.selected_candidate
    );
    assert!(diagnostics
        .candidate_acceptance
        .iter()
        .any(
            |candidate| candidate.candidate == "scanner_prior_image_adaptation"
                && candidate.status == "rejected_safety"
        ));
    assert!(diagnostics
        .selection_rejections
        .iter()
        .any(|reason| reason.contains("scanner_prior_image_adaptation rejected")));
}

#[test]
fn test_auto_mode_keeps_image_candidate_when_calibration_quality_does_not_win() {
    let mut img = ndarray::Array3::<f64>::zeros((30, 30, 3));
    for y in 0..30 {
        let value = if y < 10 {
            0.18
        } else if y < 20 {
            0.50
        } else {
            0.82
        };
        for x in 0..30 {
            for c in 0..3 {
                img[[y, x, c]] = value;
            }
        }
    }
    let mut profile = synthetic_calibration_profile();
    profile.work_to_xyz = prophoto_matrix_rows();
    profile.whitepoint = scanstitch::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;
    profile.confidence = 0.0;
    profile.fit = Some(scanstitch::color_calibration::TargetFitDiagnostics {
        method: "synthetic_bad_fit".to_string(),
        patch_count: 24,
        target_residual_rms: 0.50,
        target_residual_max: 1.20,
        validation: None,
        per_hue_residuals: Vec::new(),
        worst_patches: Vec::new(),
    });

    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Auto,
        )
        .unwrap();
    let diagnostics = result.diagnostics;

    assert_eq!(
        diagnostics.selected_candidate, "image_derived_matrix",
        "candidate scores: {:?}",
        diagnostics.candidate_scores
    );
    assert_eq!(
        diagnostics.calibration_acceptance.status,
        "rejected_quality"
    );
    assert_eq!(
        diagnostics.calibration_acceptance.beats_image_derived,
        Some(false)
    );
    assert!(diagnostics
        .candidate_acceptance
        .iter()
        .any(
            |candidate| candidate.candidate == "calibrated_direct_profile"
                && candidate.status == "rejected_quality"
        ));
    assert!(diagnostics
        .selected_runner_up_quality_delta
        .is_none_or(|delta| delta > 0.0));
    assert!(diagnostics
        .selection_rejections
        .iter()
        .any(|reason| reason.contains("did not beat image-derived score")));
}

#[test]
fn test_auto_mode_rejects_calibration_that_regresses_neutral_balance() {
    let img = broad_neutral_band_image(30, 30);
    let prophoto_to_xyz = scanstitch::colorspace::prophoto_to_xyz_d50_matrix();
    let neutral_cast = nalgebra::Matrix3::new(1.12, 0.0, 0.0, 0.0, 0.88, 0.0, 0.0, 0.0, 1.0);
    let work_to_xyz = prophoto_to_xyz * neutral_cast;
    let mut profile = synthetic_calibration_profile();
    profile.work_to_xyz = [
        [
            work_to_xyz[(0, 0)],
            work_to_xyz[(0, 1)],
            work_to_xyz[(0, 2)],
        ],
        [
            work_to_xyz[(1, 0)],
            work_to_xyz[(1, 1)],
            work_to_xyz[(1, 2)],
        ],
        [
            work_to_xyz[(2, 0)],
            work_to_xyz[(2, 1)],
            work_to_xyz[(2, 2)],
        ],
    ];
    profile.whitepoint = scanstitch::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;
    profile.confidence = 1.0;
    profile.fit = None;

    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Auto,
        )
        .unwrap();
    let diagnostics = result.diagnostics;

    assert_eq!(diagnostics.selected_candidate, "image_derived_matrix");
    assert_eq!(
        diagnostics.calibration_acceptance.status,
        "rejected_neutral"
    );
    assert_eq!(
        diagnostics.calibration_acceptance.beats_image_derived,
        Some(true),
        "fixture should isolate neutral regression rather than quality rejection: {:?}",
        diagnostics.calibration_acceptance
    );
    assert!(diagnostics
        .candidate_acceptance
        .iter()
        .any(
            |candidate| candidate.candidate == "calibrated_direct_profile"
                && candidate.status == "rejected_neutral"
                && candidate.reason.contains("neutral balance")
        ));
    assert!(diagnostics
        .selection_rejections
        .iter()
        .any(|reason| reason.contains("review_neutral_support")));
}

#[test]
fn test_auto_mode_rejects_calibration_that_regresses_reference_patches() {
    let mut img = ndarray::Array3::<f64>::zeros((36, 36, 3));
    for y in 0..36 {
        for x in 0..36 {
            let pixel = if x < 9 {
                [0.78, 0.12, 0.10]
            } else if x < 18 {
                [0.14, 0.72, 0.16]
            } else if x < 27 {
                [0.12, 0.18, 0.76]
            } else if y < 12 {
                [0.25, 0.25, 0.25]
            } else if y < 24 {
                [0.52, 0.52, 0.52]
            } else {
                [0.82, 0.82, 0.82]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }

    let image_only = scanstitch::colorspace::estimate_work_to_xyz(&img);
    let image_work_to_xyz = {
        let rows = image_only.work_to_xyz;
        nalgebra::Matrix3::new(
            rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0],
            rows[2][1], rows[2][2],
        )
    };
    let image_matrix = scanstitch::colorspace::xyz_d50_to_prophoto_matrix()
        * scanstitch::colorspace::bradford_cat(&image_only.source_white)
        * image_work_to_xyz;

    let prophoto_to_xyz = scanstitch::colorspace::prophoto_to_xyz_d50_matrix();
    let uniform_underfit =
        prophoto_to_xyz * nalgebra::Matrix3::new(0.90, 0.0, 0.0, 0.0, 0.90, 0.0, 0.0, 0.0, 0.90);
    let mut profile = synthetic_calibration_profile();
    profile.work_to_xyz = [
        [
            uniform_underfit[(0, 0)],
            uniform_underfit[(0, 1)],
            uniform_underfit[(0, 2)],
        ],
        [
            uniform_underfit[(1, 0)],
            uniform_underfit[(1, 1)],
            uniform_underfit[(1, 2)],
        ],
        [
            uniform_underfit[(2, 0)],
            uniform_underfit[(2, 1)],
            uniform_underfit[(2, 2)],
        ],
    ];
    profile.whitepoint = scanstitch::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;
    profile.confidence = 1.0;
    profile.fit = None;
    profile.target_patches = reference_fit_patches_for_matrix(image_matrix);

    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Auto,
        )
        .unwrap();
    let diagnostics = result.diagnostics;

    assert_ne!(diagnostics.selected_candidate, "calibrated_direct_profile");
    assert_eq!(
        diagnostics.calibration_acceptance.status,
        "rejected_reference_fit"
    );
    assert_eq!(diagnostics.candidate_risk, "review_reference_fit");
    let reference = diagnostics
        .reference_patch_evaluation
        .expect("reference patch evaluation");
    assert_eq!(reference.patch_count, 6);
    assert!(reference
        .candidate_evaluations
        .iter()
        .any(|candidate| candidate.candidate == "calibrated_direct_profile"));
    let calibrated_evaluation = reference
        .candidate_evaluations
        .iter()
        .find(|candidate| candidate.candidate == "calibrated_direct_profile")
        .expect("calibrated candidate evaluation");
    assert_eq!(
        calibrated_evaluation.regresses_image_derived,
        Some(true),
        "calibrated reference fit should explicitly report a regression"
    );
    assert!(
        !calibrated_evaluation.hue_family_regressions.is_empty(),
        "calibrated reference fit should identify hue-family regressions: {:?}",
        calibrated_evaluation
    );
    let calibrated_score = diagnostics
        .candidate_scores
        .iter()
        .find(|score| score.candidate == "calibrated_direct_profile")
        .expect("calibrated candidate score");
    assert!(
        calibrated_score.quality_components.target_residual_penalty > 0.0,
        "reference-patch residuals should feed the candidate quality score: {:?}",
        calibrated_score
    );
    assert!(diagnostics
        .candidate_acceptance
        .iter()
        .any(
            |candidate| candidate.candidate == "calibrated_direct_profile"
                && candidate.status == "rejected_reference_fit"
        ));
    assert!(diagnostics
        .selection_rejections
        .iter()
        .any(|reason| reason.contains("review_reference_fit")));
}

#[test]
fn test_forced_calibrated_mode_fails_for_materially_unsafe_profile() {
    let img = sparse_strong_anchor_image();
    let mut profile = synthetic_calibration_profile();
    profile.work_to_xyz = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    profile.whitepoint = scanstitch::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;

    let err =
        match scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Calibrated,
        ) {
            Ok(_) => panic!("forced calibrated mode should fail unsafe calibrated profiles"),
            Err(err) => err,
        };

    assert!(
        err.contains("selected only unsafe colorspace candidate"),
        "unexpected error: {}",
        err
    );
    assert!(
        err.contains("negative channel values"),
        "error should include the gamut safety reason: {}",
        err
    );
}

#[test]
fn test_neutral_trim_requires_enough_neutral_support_and_stays_clamped() {
    let prophoto_to_xyz = scanstitch::colorspace::prophoto_to_xyz_d50_matrix();
    let cast = nalgebra::Matrix3::new(1.12, 0.0, 0.0, 0.0, 0.88, 0.0, 0.0, 0.0, 1.0);
    let work_to_xyz = prophoto_to_xyz * cast;
    let mut profile = synthetic_calibration_profile();
    profile.work_to_xyz = [
        [
            work_to_xyz[(0, 0)],
            work_to_xyz[(0, 1)],
            work_to_xyz[(0, 2)],
        ],
        [
            work_to_xyz[(1, 0)],
            work_to_xyz[(1, 1)],
            work_to_xyz[(1, 2)],
        ],
        [
            work_to_xyz[(2, 0)],
            work_to_xyz[(2, 1)],
            work_to_xyz[(2, 2)],
        ],
    ];
    profile.whitepoint = scanstitch::constants::D50_WHITE;
    profile.matrix_condition_number = 1.3;

    let mut supported = ndarray::Array3::<f64>::zeros((30, 30, 3));
    for y in 0..30 {
        let value = if y < 10 {
            0.18
        } else if y < 20 {
            0.50
        } else {
            0.82
        };
        for x in 0..30 {
            for c in 0..3 {
                supported[[y, x, c]] = value;
            }
        }
    }
    let supported_result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &supported,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Calibrated,
        )
        .unwrap();
    let supported_diag = supported_result.diagnostics;
    assert!(supported_diag.neutral_estimate_quality.accepted);
    assert_eq!(
        supported_diag.neutral_estimate_quality.populated_band_count,
        3
    );
    assert!(supported_diag.neutral_trim_applied);
    assert!(supported_diag
        .neutral_trim_scale
        .iter()
        .all(|scale| (0.85..=1.18).contains(scale)));
    let trim = supported_diag.neutral_trim_before_after;
    assert!(trim.neutral_delta_reduced);
    assert!(!trim.neutral_band_delta_worsened);
    assert!(!trim.low_clipping_increased);
    assert!(!trim.high_clipping_increased);
    assert!(
        trim.after.as_ref().unwrap().neutral_delta_magnitude
            < trim.before.as_ref().unwrap().neutral_delta_magnitude
    );
    let before_bands = trim.before.as_ref().unwrap().neutral_band_delta_magnitude;
    let after_bands = trim.after.as_ref().unwrap().neutral_band_delta_magnitude;
    assert!(
        before_bands.iter().all(Option::is_some) && after_bands.iter().all(Option::is_some),
        "broad support should report shadow/midtone/highlight trim deltas"
    );

    let weak = ndarray::Array3::<f64>::from_elem((20, 20, 3), 0.50);
    let weak_result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &weak,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Calibrated,
        )
        .unwrap();
    assert!(!weak_result.diagnostics.neutral_trim_applied);
    assert_eq!(weak_result.diagnostics.neutral_trim_scale, [1.0, 1.0, 1.0]);
    assert!(!weak_result.diagnostics.neutral_estimate_quality.accepted);
    assert_eq!(
        weak_result
            .diagnostics
            .neutral_estimate_quality
            .populated_band_count,
        1
    );

    let insufficient = ndarray::Array3::<f64>::from_elem((6, 6, 3), 0.50);
    let insufficient_result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &insufficient,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Calibrated,
        )
        .unwrap();
    assert!(!insufficient_result.diagnostics.neutral_trim_applied);
    assert!(
        !insufficient_result
            .diagnostics
            .neutral_estimate_quality
            .accepted
    );
    assert!(insufficient_result
        .diagnostics
        .neutral_estimate_quality
        .reason
        .contains("below required"));
}

#[test]
fn test_neutral_sampling_reports_rejection_reasons_and_support() {
    let mut broad = broad_neutral_band_image(30, 30);
    broad[[4, 4, 0]] = 0.9995;
    broad[[4, 4, 1]] = 0.9995;
    broad[[4, 4, 2]] = 0.9995;
    broad[[8, 8, 0]] = 0.995;
    broad[[8, 8, 1]] = 0.02;
    broad[[8, 8, 2]] = 0.02;
    broad[[12, 12, 0]] = 0.68;
    broad[[12, 12, 1]] = 0.35;
    broad[[12, 12, 2]] = 0.18;
    for x in 0..30 {
        broad[[29, x, 0]] = 0.84;
        broad[[29, x, 1]] = 0.83;
        broad[[29, x, 2]] = 0.82;
    }

    let result = scanstitch::colorspace::map_to_prophoto_d50_with_diagnostics(&broad);
    let rejections = result.diagnostics.neutral_sample_rejections;
    assert!(result.diagnostics.neutral_estimate_quality.broad_support);
    assert!(rejections.accepted_neutral_samples > 0);
    assert!(
        rejections.clipped >= 1,
        "expected clipped highlight rejection"
    );
    assert!(rejections.dust >= 1, "expected dust/outlier rejection");
    assert!(
        rejections.film_base_like_edge >= 1,
        "expected film-base-like edge rejection"
    );
    assert!(
        rejections.chroma_threshold >= 1,
        "expected non-neutral chroma rejection"
    );

    let one_band = ndarray::Array3::<f64>::from_elem((24, 24, 3), 0.50);
    let one_band_result = scanstitch::colorspace::map_to_prophoto_d50_with_diagnostics(&one_band);
    assert!(
        !one_band_result
            .diagnostics
            .neutral_estimate_quality
            .accepted
    );
    assert_eq!(
        one_band_result
            .diagnostics
            .neutral_estimate_quality
            .populated_band_count,
        1
    );
}

#[test]
fn test_dominant_anchor_sampling_rejects_edge_artifacts_and_reports_luma_bands() {
    let mut img = broad_neutral_band_image(30, 30);
    for y in 5..10 {
        for x in 5..10 {
            img[[y, x, 0]] = 0.32;
            img[[y, x, 1]] = 0.05;
            img[[y, x, 2]] = 0.05;
        }
    }
    for y in 10..15 {
        for x in 5..10 {
            img[[y, x, 0]] = 0.70;
            img[[y, x, 1]] = 0.18;
            img[[y, x, 2]] = 0.12;
        }
    }
    for y in 15..20 {
        for x in 5..10 {
            img[[y, x, 0]] = 0.95;
            img[[y, x, 1]] = 0.65;
            img[[y, x, 2]] = 0.55;
        }
    }
    img[[3, 3, 0]] = 0.9995;
    img[[3, 3, 1]] = 0.45;
    img[[3, 3, 2]] = 0.12;
    img[[4, 4, 0]] = 0.995;
    img[[4, 4, 1]] = 0.02;
    img[[4, 4, 2]] = 0.02;
    img[[5, 0, 0]] = 0.10;
    img[[5, 0, 1]] = 0.01;
    img[[5, 0, 2]] = 0.01;
    img[[29, 8, 0]] = 0.84;
    img[[29, 8, 1]] = 0.82;
    img[[29, 8, 2]] = 0.80;
    img[[2, 2, 0]] = 0.04;
    img[[2, 2, 1]] = 0.02;
    img[[2, 2, 2]] = 0.02;
    img[[20, 20, 0]] = 0.50;
    img[[20, 20, 1]] = 0.48;
    img[[20, 20, 2]] = 0.47;
    img[[21, 21, 0]] = 0.60;
    img[[21, 21, 1]] = 0.55;
    img[[21, 21, 2]] = 0.10;

    let result = scanstitch::colorspace::map_to_prophoto_d50_with_diagnostics(&img);
    let diagnostics = result.diagnostics;
    let rejections = diagnostics.dominant_anchor_sample_rejections;

    assert_eq!(
        rejections.accepted_anchor_samples,
        diagnostics.channel_anchor_counts.iter().sum::<usize>()
    );
    assert!(rejections.clipped >= 1, "expected clipped anchor rejection");
    assert!(rejections.dust >= 1, "expected dust anchor rejection");
    assert!(rejections.border >= 1, "expected border anchor rejection");
    assert!(
        rejections.film_base_like_edge >= 1,
        "expected film-base-like edge anchor rejection"
    );
    assert!(
        rejections.luma_out_of_range >= 1,
        "expected luma anchor rejection"
    );
    assert!(
        rejections.low_saturation >= 1,
        "expected low-saturation anchor rejection"
    );
    assert!(
        rejections.weak_dominance >= 1,
        "expected weak-dominance anchor rejection"
    );
    assert!(
        diagnostics.dominant_anchor_bands[0]
            .iter()
            .all(|count| *count > 0),
        "red anchors should retain shadow/midtone/highlight band support: {:?}",
        diagnostics.dominant_anchor_bands
    );
}

#[test]
fn test_dominant_anchor_instability_applies_evidence_gated_neutral_safety_rescue() {
    let mut img = broad_neutral_band_image(45, 45);
    for y in 17..29 {
        for x in 4..17 {
            img[[y, x, 0]] = 0.55;
            img[[y, x, 1]] = 0.24;
            img[[y, x, 2]] = 0.20;
        }
        for x in 17..30 {
            img[[y, x, 0]] = 0.22;
            img[[y, x, 1]] = 0.55;
            img[[y, x, 2]] = 0.22;
        }
        for x in 30..43 {
            img[[y, x, 0]] = 0.20;
            img[[y, x, 1]] = 0.25;
            img[[y, x, 2]] = 0.56;
        }
    }

    let result = scanstitch::colorspace::map_to_prophoto_d50_with_diagnostics(&img);
    let diagnostics = result.diagnostics;
    let anchor_quality = &diagnostics.dominant_anchor_quality;

    assert!(
        !anchor_quality.accepted,
        "single-band anchors should require review: {:?}",
        anchor_quality
    );
    assert_eq!(anchor_quality.channel_populated_band_count, [1, 1, 1]);
    assert_eq!(anchor_quality.channel_unstable, [true, true, true]);
    assert_eq!(diagnostics.candidate_risk, "fallback_only");
    assert_eq!(
        diagnostics.mapping_strategy,
        "neutral_balance_evidence_rescue"
    );
    assert_eq!(diagnostics.selected_candidate, "neutral_balance_fallback");
    assert!(diagnostics.neutral_safety_rescue.evaluated);
    assert!(diagnostics.neutral_safety_rescue.applied);
    assert_eq!(
        diagnostics
            .neutral_safety_rescue
            .matrix_candidate
            .as_deref(),
        Some("gamut_trusted_image_matrix_blend")
    );
    assert_eq!(
        diagnostics
            .neutral_safety_rescue
            .matrix_anchor_evidence_supported,
        Some(false)
    );
    assert!(diagnostics
        .neutral_safety_rescue
        .preserved_ratio_gain
        .is_some_and(|gain| gain >= 0.50));
    assert!(diagnostics
        .neutral_safety_rescue
        .midtone_saturation_p95_reduction
        .is_some_and(|reduction| reduction >= 0.30));
    assert_eq!(
        scanstitch::colorspace::tone_color_trust_state(&diagnostics),
        "review_required"
    );
    assert!(diagnostics.candidate_acceptance.iter().any(|candidate| {
        candidate.candidate == "neutral_balance_fallback"
            && candidate.status == "selected_evidence_rescue"
    }));
    let superseded_score = diagnostics
        .candidate_scores
        .iter()
        .find(|score| {
            Some(score.candidate)
                == diagnostics
                    .neutral_safety_rescue
                    .matrix_candidate
                    .as_deref()
        })
        .expect("superseded image-derived score");
    assert!(
        superseded_score.quality_components.anchor_stability_penalty > 0.0,
        "candidate score should include anchor stability risk: {:?}",
        superseded_score
    );
    assert_eq!(
        superseded_score.dominant_anchor_unstable_channels,
        [true, true, true]
    );
}

#[test]
fn test_candidate_scoring_reports_physical_and_perceptual_model_evidence() {
    let img = sparse_strong_anchor_image();

    let result = scanstitch::colorspace::map_to_prophoto_d50_with_diagnostics(&img);
    let diagnostics = result.diagnostics;
    let selected_score = diagnostics
        .candidate_scores
        .iter()
        .find(|score| score.selected)
        .expect("selected score");
    let model = selected_score
        .color_model_quality
        .as_ref()
        .expect("candidate model diagnostics");

    assert!(model.sample_count > 0);
    assert!(model.density_monotonicity_score.is_finite());
    assert!(model.density_monotonicity_violation_ratio >= 0.0);
    assert!(
        model.saturation_preservation_sample_count > 0,
        "saturated fixture should exercise saturation preservation diagnostics: {:?}",
        model
    );
    assert!(model.saturation_preservation_median_ratio.is_some());
    assert!(model.spatial_consistency.neutral_sample_count > 0);
    assert!(
        selected_score
            .quality_components
            .density_monotonicity_penalty
            >= 0.0
    );
    assert!(selected_score.quality_components.hue_linearity_penalty >= 0.0);
    assert!(
        selected_score
            .quality_components
            .saturation_preservation_penalty
            >= 0.0
    );
    assert!(selected_score.quality_components.memory_color_penalty >= 0.0);
    assert!(
        selected_score
            .quality_components
            .spatial_consistency_penalty
            >= 0.0
    );
}

#[test]
fn test_reference_patch_evaluation_reports_lab_delta_e_residuals() {
    let mut img = ndarray::Array3::<f64>::zeros((36, 36, 3));
    for y in 0..36 {
        for x in 0..36 {
            let pixel = if x < 9 {
                [0.78, 0.12, 0.10]
            } else if x < 18 {
                [0.14, 0.72, 0.16]
            } else if x < 27 {
                [0.12, 0.18, 0.76]
            } else if y < 12 {
                [0.25, 0.25, 0.25]
            } else if y < 24 {
                [0.52, 0.52, 0.52]
            } else {
                [0.82, 0.82, 0.82]
            };
            for c in 0..3 {
                img[[y, x, c]] = pixel[c];
            }
        }
    }

    let image_only = scanstitch::colorspace::estimate_work_to_xyz(&img);
    let image_work_to_xyz = {
        let rows = image_only.work_to_xyz;
        nalgebra::Matrix3::new(
            rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0],
            rows[2][1], rows[2][2],
        )
    };
    let image_matrix = scanstitch::colorspace::xyz_d50_to_prophoto_matrix()
        * scanstitch::colorspace::bradford_cat(&image_only.source_white)
        * image_work_to_xyz;

    let mut profile = synthetic_calibration_profile();
    let prophoto_to_xyz = scanstitch::colorspace::prophoto_to_xyz_d50_matrix();
    let underfit =
        prophoto_to_xyz * nalgebra::Matrix3::new(0.84, 0.0, 0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 1.08);
    profile.work_to_xyz = [
        [underfit[(0, 0)], underfit[(0, 1)], underfit[(0, 2)]],
        [underfit[(1, 0)], underfit[(1, 1)], underfit[(1, 2)]],
        [underfit[(2, 0)], underfit[(2, 1)], underfit[(2, 2)]],
    ];
    profile.whitepoint = scanstitch::constants::D50_WHITE;
    profile.matrix_condition_number = 1.0;
    profile.confidence = 1.0;
    profile.target_patches = reference_fit_patches_for_matrix(image_matrix);

    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &img,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Auto,
        )
        .unwrap();
    let diagnostics = result.diagnostics;
    let reference = diagnostics
        .reference_patch_evaluation
        .expect("reference patch evaluation");

    assert!(reference.selected_rms_delta_e.is_finite());
    assert!(reference.image_derived_rms_delta_e.is_finite());
    assert!(reference.selected_rms_delta_e2000.is_finite());
    assert!(reference.image_derived_rms_delta_e2000.is_finite());
    assert!(reference.candidate_evaluations.iter().all(|candidate| {
        candidate.rms_delta_e.is_finite()
            && candidate.max_delta_e.is_finite()
            && candidate.rms_delta_e2000.is_finite()
            && candidate.max_delta_e2000.is_finite()
            && candidate.mean_delta_e2000.is_finite()
    }));
    assert!(reference.per_patch.iter().all(|patch| {
        patch.selected_delta_e.is_finite()
            && patch.image_derived_delta_e.is_finite()
            && patch.selected_delta_e2000.is_finite()
            && patch.image_derived_delta_e2000.is_finite()
    }));

    let calibrated_score = diagnostics
        .candidate_scores
        .iter()
        .find(|score| score.candidate == "calibrated_direct_profile")
        .expect("calibrated candidate score");
    assert!(calibrated_score.reference_patch_rms_delta_e.is_some());
    assert!(calibrated_score.reference_patch_max_delta_e.is_some());
    assert!(calibrated_score.reference_patch_rms_delta_e2000.is_some());
    assert!(calibrated_score.reference_patch_max_delta_e2000.is_some());
    assert!(calibrated_score
        .reference_patch_max_delta_vs_image_derived
        .is_some());
    assert!(calibrated_score
        .reference_patch_delta_e_max_delta_vs_image_derived
        .is_some());
    assert!(calibrated_score
        .reference_patch_delta_e2000_max_delta_vs_image_derived
        .is_some());
    assert!(calibrated_score.quality_components.target_residual_penalty > 0.0);
}

#[test]
fn held_out_root_polynomial_is_applied_when_scene_support_and_quality_pass() {
    let (profile, held_out) = nonlinear_calibration_profile();
    let mut image = Array3::<f64>::zeros((16, 16, 3));
    for y in 0..16 {
        for x in 0..16 {
            let rgb = held_out[(y * 16 + x) % held_out.len()].source_rgb;
            for channel in 0..3 {
                image[[y, x, channel]] = rgb[channel];
            }
        }
    }
    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &image,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Calibrated,
        )
        .expect("held-out nonlinear calibrated mapping");
    assert_eq!(
        result.diagnostics.selected_candidate,
        "calibrated_root_polynomial"
    );
    let runtime = result
        .diagnostics
        .nonlinear_color_model
        .as_ref()
        .expect("nonlinear runtime diagnostics");
    assert_eq!(runtime.support_status, "accepted");
    assert!(runtime.outside_training_chromaticity_hull_ratio <= 0.35);
    assert!(!result.diagnostics.neutral_trim_applied);
    assert!(!result.diagnostics.neutral_safety_rescue.evaluated);
    assert!(!result.diagnostics.neutral_safety_rescue.applied);
    assert!(result
        .diagnostics
        .neutral_trim_before_after
        .reason
        .contains("nonlinear calibration is preserved exactly as held-out validated"));

    let model = profile.color_model.as_ref().unwrap();
    let expected_xyz =
        scanstitch::color_calibration::evaluate_root_polynomial_xyz(model, held_out[0].source_rgb);
    let expected = scanstitch::colorspace::xyz_d50_to_prophoto_matrix()
        * scanstitch::colorspace::bradford_cat(&profile.whitepoint)
        * nalgebra::Vector3::new(expected_xyz[0], expected_xyz[1], expected_xyz[2]);
    for channel in 0..3 {
        assert!(
            (result.prophoto[[0, 0, channel]]
                - expected[channel] / result.diagnostics.exposure_scale)
                .abs()
                < 1e-9
        );
    }
}

#[test]
fn root_polynomial_falls_back_to_matrix_outside_measured_scene_support() {
    let (profile, _) = nonlinear_calibration_profile();
    let mut image = Array3::<f64>::zeros((16, 16, 3));
    for y in 0..16 {
        for x in 0..16 {
            image[[y, x, 0]] = 1.0;
            image[[y, x, 1]] = 0.001;
            image[[y, x, 2]] = 0.001;
        }
    }
    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &image,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Calibrated,
        )
        .expect("matrix fallback remains a calibrated candidate");
    assert_eq!(
        result.diagnostics.selected_candidate,
        "calibrated_direct_profile"
    );
    let nonlinear = result
        .diagnostics
        .candidate_scores
        .iter()
        .find(|candidate| candidate.candidate == "calibrated_root_polynomial")
        .expect("nonlinear candidate audit");
    assert!(nonlinear.rejected);
    assert!(nonlinear
        .rejection_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("target-domain support rejected")));
}

#[test]
fn held_out_residual_lut_is_applied_inside_measured_scene_support() {
    let (profile, held_out) = residual_lut_calibration_profile();
    let model = profile.lut_3d_model.as_ref().expect("residual LUT model");
    let supported = held_out
        .iter()
        .filter(|patch| {
            scanstitch::color_calibration::residual_lut_3d_has_full_support(model, patch.source_rgb)
        })
        .collect::<Vec<_>>();
    assert!(
        supported.len() >= 64,
        "synthetic scene needs broad LUT support"
    );
    let mut image = Array3::<f64>::zeros((16, 16, 3));
    for y in 0..16 {
        for x in 0..16 {
            let rgb = supported[(y * 16 + x) % supported.len()].source_rgb;
            for channel in 0..3 {
                image[[y, x, channel]] = rgb[channel];
            }
        }
    }

    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &image,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Calibrated,
        )
        .expect("held-out residual LUT calibrated mapping");
    assert_eq!(
        result.diagnostics.selected_candidate,
        "calibrated_residual_lut_3d"
    );
    let runtime = result
        .diagnostics
        .nonlinear_color_model
        .as_ref()
        .expect("residual LUT runtime diagnostics");
    assert_eq!(runtime.model_type, "residual_lut_3d");
    assert_eq!(runtime.grid_size, Some(5));
    assert_eq!(runtime.baseline_kind, "matrix");
    assert_eq!(runtime.support_status, "accepted");
    assert!(runtime.full_model_application_ratio >= 0.65);
    assert!(!result.diagnostics.neutral_trim_applied);
    assert!(!result.diagnostics.neutral_safety_rescue.evaluated);
    assert!(!result.diagnostics.neutral_safety_rescue.applied);

    let expected_xyz = scanstitch::color_calibration::evaluate_residual_lut_3d_xyz(
        model,
        &profile.work_to_xyz,
        None,
        supported[0].source_rgb,
    );
    let expected = scanstitch::colorspace::xyz_d50_to_prophoto_matrix()
        * scanstitch::colorspace::bradford_cat(&profile.whitepoint)
        * Vector3::new(expected_xyz[0], expected_xyz[1], expected_xyz[2]);
    for channel in 0..3 {
        assert!(
            (result.prophoto[[0, 0, channel]]
                - expected[channel] / result.diagnostics.exposure_scale)
                .abs()
                < 1e-9
        );
    }
}

#[test]
fn residual_lut_falls_back_to_matrix_outside_measured_rgb_volume() {
    let (profile, _) = residual_lut_calibration_profile();
    let mut image = Array3::<f64>::zeros((16, 16, 3));
    image.fill(1.25);

    let result =
        scanstitch::colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &image,
            Some(&profile),
            scanstitch::colorspace::ColorMode::Calibrated,
        )
        .expect("matrix fallback remains available outside the LUT volume");
    assert_eq!(
        result.diagnostics.selected_candidate,
        "calibrated_direct_profile"
    );
    let lut = result
        .diagnostics
        .candidate_scores
        .iter()
        .find(|candidate| candidate.candidate == "calibrated_residual_lut_3d")
        .expect("residual LUT candidate audit");
    assert!(lut.rejected);
    assert!(lut.rejection_reason.as_deref().is_some_and(|reason| {
        reason.contains("target-domain support rejected")
            && reason.contains("outside the measured 3D LUT input domain")
    }));
    let runtime = result
        .diagnostics
        .nonlinear_color_model
        .as_ref()
        .expect("residual LUT rejection diagnostics");
    assert_eq!(runtime.model_type, "residual_lut_3d");
    assert_eq!(runtime.support_status, "rejected");
    assert_eq!(runtime.outside_training_input_domain_ratio, 1.0);
    assert_eq!(runtime.full_model_application_ratio, 0.0);
}

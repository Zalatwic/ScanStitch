fn valid_profile_json() -> String {
    serde_json::json!({
        "schema_version": 1,
        "profile_id": "synthetic-prophoto-d50",
        "source_space": {
            "name": "synthetic linear scanner RGB",
            "encoding": "linear"
        },
        "scanner": {
            "make": "Synthetic",
            "model": "Unit Test"
        },
        "film": {
            "stock": "Synthetic negative",
            "process": "C-41"
        },
        "target": {
            "type": "synthetic_color_checker",
            "illuminant": "D50"
        },
        "reference": {
            "dataset": "synthetic",
            "observer": "CIE 1931 2 degree"
        },
        "whitepoint": [0.9642, 1.0, 0.8251],
        "work_to_xyz": [
            [0.7977, 0.1352, 0.0313],
            [0.2880, 0.7119, 0.0001],
            [0.0000, 0.0000, 0.8251]
        ],
        "confidence": 0.95,
        "gamut_limits": {
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 1.0, 1.0]
        }
    })
    .to_string()
}

fn synthetic_root_polynomial_patches(
    count: usize,
    id_prefix: &str,
    seed_offset: usize,
    nonlinear: bool,
) -> Vec<scanstitch::color_calibration::TargetPatch> {
    let linear = [[0.65, 0.14, 0.05], [0.25, 0.69, 0.10], [0.03, 0.08, 0.76]];
    let nonlinear_coefficients = [
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
            let reference_xyz = if nonlinear {
                let basis = scanstitch::color_calibration::root_polynomial_basis_values(2, rgb)
                    .expect("degree-two basis");
                std::array::from_fn(|channel| {
                    basis
                        .iter()
                        .zip(nonlinear_coefficients)
                        .map(|(term, coefficient)| term * coefficient[channel])
                        .sum()
                })
            } else {
                std::array::from_fn(|channel| {
                    (0..3)
                        .map(|source_channel| linear[channel][source_channel] * rgb[source_channel])
                        .sum()
                })
            };
            scanstitch::color_calibration::TargetPatch {
                patch_id: Some(format!("{id_prefix}-{index:03}")),
                scanner_xy: None,
                source_rgb: rgb,
                reference_xyz,
            }
        })
        .collect()
}

fn synthetic_residual_lut_patches(
    count: usize,
    id_prefix: &str,
    seed_offset: usize,
    include_domain_corners: bool,
    nonlinear: bool,
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
            let mut reference_xyz = apply_rows(matrix, source_rgb);
            if nonlinear {
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
            }
            scanstitch::color_calibration::TargetPatch {
                patch_id: Some(format!("{id_prefix}-{index:03}")),
                scanner_xy: None,
                source_rgb,
                reference_xyz,
            }
        })
        .collect()
}

#[test]
fn test_root_polynomial_fitter_selects_held_out_nonlinearity_and_detects_tampering() {
    let training = synthetic_root_polynomial_patches(48, "train", 0, true);
    let held_out = synthetic_root_polynomial_patches(36, "held", 409, true);
    let matrix = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training,
        &held_out,
        "matrix_baseline",
        Some([0.9642, 1.0, 0.8251]),
        None,
    )
    .expect("matrix baseline");
    let selection =
        scanstitch::color_calibration::fit_root_polynomial_color_model_from_disjoint_patches(
            "synthetic-nonlinear",
            &training,
            &held_out,
            &matrix.matrix,
        )
        .expect("root-polynomial selection");
    let model = selection
        .selected_model
        .expect("nonlinear held-out data should select a model");
    assert_eq!(
        model.degree, 2,
        "cubic must not win without added held-out gain"
    );
    assert!(model.validation.selected_over_matrix);
    assert!(model.validation.held_out_delta_e00_rms < 0.05);
    assert!(model.validation.matrix_held_out_delta_e00_rms > 0.25);
    assert!(
        scanstitch::color_calibration::validate_root_polynomial_color_model(
            &model,
            &matrix.matrix,
            &held_out,
        )
        .is_empty()
    );

    let mut tampered = model.clone();
    tampered.coefficients[3][0] += 0.04;
    let errors = scanstitch::color_calibration::validate_root_polynomial_color_model(
        &tampered,
        &matrix.matrix,
        &held_out,
    );
    assert!(errors.iter().any(|error| {
        error.contains("inconsistent with retained held-out patches")
            || error.contains("deterministic robust fit")
            || error.contains("coefficient_max_abs")
    }));

    let mut training_tampered = model.clone();
    training_tampered.training_patches[0].reference_xyz[0] += 0.01;
    let errors = scanstitch::color_calibration::validate_root_polynomial_color_model(
        &training_tampered,
        &matrix.matrix,
        &held_out,
    );
    assert!(errors.iter().any(|error| {
        error.contains("deterministic robust fit")
            || error.contains("deterministic held-out complexity selection")
            || error.contains("retained training patches")
    }));

    let mut hull_tampered = model.clone();
    hull_tampered.training_chromaticity_hull[0][0] += 0.001;
    let errors = scanstitch::color_calibration::validate_root_polynomial_color_model(
        &hull_tampered,
        &matrix.matrix,
        &held_out,
    );
    assert!(errors
        .iter()
        .any(|error| error.contains("inconsistent with retained training_patches")));

    let mut regularization_tampered = model;
    regularization_tampered.validation.regularization_lambda =
        if (regularization_tampered.validation.regularization_lambda - 1e-6).abs() < 1e-12 {
            1e-5
        } else {
            1e-6
        };
    let errors = scanstitch::color_calibration::validate_root_polynomial_color_model(
        &regularization_tampered,
        &matrix.matrix,
        &held_out,
    );
    assert!(errors.iter().any(|error| {
        error.contains("regularization_lambda is inconsistent with retained training patches")
    }));
}

#[test]
fn test_root_polynomial_fitter_retains_matrix_for_linear_target() {
    let training = synthetic_root_polynomial_patches(48, "linear-train", 0, false);
    let held_out = synthetic_root_polynomial_patches(36, "linear-held", 409, false);
    let matrix = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training,
        &held_out,
        "matrix_baseline",
        Some([0.9642, 1.0, 0.8251]),
        None,
    )
    .expect("matrix baseline");
    let selection =
        scanstitch::color_calibration::fit_root_polynomial_color_model_from_disjoint_patches(
            "synthetic-linear",
            &training,
            &held_out,
            &matrix.matrix,
        )
        .expect("root-polynomial selection");
    assert!(selection.selected_model.is_none());
    assert_eq!(selection.diagnostics.status, "matrix_retained");
    assert!(selection
        .diagnostics
        .candidates
        .iter()
        .all(|candidate| candidate.status != "selected"));
}

#[test]
fn test_root_polynomial_fitter_refuses_underconstrained_target() {
    let training = synthetic_root_polynomial_patches(12, "small-train", 0, true);
    let held_out = synthetic_root_polynomial_patches(12, "small-held", 409, true);
    let matrix = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training,
        &held_out,
        "matrix_baseline",
        Some([0.9642, 1.0, 0.8251]),
        None,
    )
    .expect("matrix baseline");
    let selection =
        scanstitch::color_calibration::fit_root_polynomial_color_model_from_disjoint_patches(
            "synthetic-small",
            &training,
            &held_out,
            &matrix.matrix,
        )
        .expect("root-polynomial selection");
    assert!(selection.selected_model.is_none());
    assert!(selection
        .diagnostics
        .candidates
        .iter()
        .all(|candidate| { candidate.status == "not_evaluated_insufficient_samples" }));
}

#[test]
fn test_residual_lut_3d_selects_smooth_held_out_nonlinearity_and_retains_baseline_for_linear_data()
{
    let matrix = [[0.65, 0.14, 0.05], [0.25, 0.69, 0.10], [0.03, 0.08, 0.76]];
    let training = synthetic_residual_lut_patches(180, "lut-train", 0, true, true);
    let held_out = synthetic_residual_lut_patches(80, "lut-held", 431, false, true);
    let selection =
        scanstitch::color_calibration::fit_residual_lut_3d_color_model_from_disjoint_patches(
            "synthetic-lut",
            &training,
            &held_out,
            &matrix,
            None,
        )
        .unwrap();
    let model = selection
        .selected_model
        .expect("smooth held-out nonlinearity should select the 3D residual LUT");
    assert_eq!(model.grid_size, 5);
    assert_eq!(model.residual_nodes_xyz.len(), 125);
    assert!(model.validation.selected_over_baseline);
    assert!(model.validation.held_out_delta_e00_rms_improvement >= 0.20);
    assert!(model.validation.occupied_cell_fraction >= 0.35);
    assert!(
        scanstitch::color_calibration::residual_lut_3d_has_full_support(
            &model,
            held_out[0].source_rgb
        )
    );
    let mapped = scanstitch::color_calibration::evaluate_residual_lut_3d_xyz(
        &model,
        &matrix,
        None,
        held_out[0].source_rgb,
    );
    assert!(mapped.iter().all(|value| value.is_finite()));
    assert!(
        scanstitch::color_calibration::validate_residual_lut_3d_color_model(
            &model, &matrix, None, &held_out
        )
        .is_empty()
    );
    let mut tampered = model.clone();
    tampered.residual_nodes_xyz[31][0] += 0.01;
    assert!(
        scanstitch::color_calibration::validate_residual_lut_3d_color_model(
            &tampered, &matrix, None, &held_out
        )
        .iter()
        .any(|reason| reason.contains("deterministic robust fit"))
    );

    let linear_training = synthetic_residual_lut_patches(180, "lut-linear-train", 0, true, false);
    let linear_held = synthetic_residual_lut_patches(80, "lut-linear-held", 431, false, false);
    let linear =
        scanstitch::color_calibration::fit_residual_lut_3d_color_model_from_disjoint_patches(
            "synthetic-linear-lut",
            &linear_training,
            &linear_held,
            &matrix,
            None,
        )
        .unwrap();
    assert!(linear.selected_model.is_none());
    assert_eq!(linear.diagnostics.status, "baseline_retained");
}

fn scanner_profile_json(profile_id: &str, confidence: f64, auto_match: bool) -> String {
    let fingerprint = synthetic_scanner_settings_fingerprint();
    serde_json::json!({
        "schema_version": 1,
        "record_type": "scanner_profile",
        "profile_id": profile_id,
        "source_space": { "name": "synthetic scanner RGB", "encoding": "linear" },
        "scanner": { "make": "Synthetic", "model": "Library Scanner" },
        "settings": { "dpi": 3200, "mode": "positive" },
        "target": { "type": "synthetic_transmissive_target", "illuminant": "D50" },
        "reference": { "dataset": "synthetic" },
        "whitepoint": [0.9642, 1.0, 0.8251],
        "scanner_rgb_to_xyz": [
            [0.80, 0.10, 0.00],
            [0.20, 0.70, 0.10],
            [0.00, 0.20, 0.80]
        ],
        "scanner_settings_fingerprint": fingerprint,
        "confidence": confidence,
        "auto_match": auto_match
    })
    .to_string()
}

fn roll_profile_json(profile_id: &str, scanner_profile_id: &str) -> String {
    let fingerprint = synthetic_scanner_settings_fingerprint();
    serde_json::json!({
        "schema_version": 1,
        "record_type": "roll_profile",
        "profile_id": profile_id,
        "scanner_profile_id": scanner_profile_id,
        "film": { "stock": "Synthetic 200", "process": "C-41" },
        "development": { "developer": "synthetic", "time_minutes": 3.5 },
        "metadata": { "roll": "unit-test" },
        "target": { "type": "known_reference_frame" },
        "reference": { "dataset": "synthetic-roll" },
        "base_color": [12000.0, 7000.0, 3000.0],
        "correction_domain": "xyz_post_scanner",
        "correction_matrix": [
            [1.10, 0.00, 0.00],
            [0.00, 0.90, 0.00],
            [0.00, 0.00, 1.00]
        ],
        "scanner_settings_fingerprint": fingerprint,
        "confidence": 0.82
    })
    .to_string()
}

fn measured_negative_response_json() -> serde_json::Value {
    serde_json::json!({
        "model_id": "synthetic-roll-response-v1",
        "scanner_density_to_layer_density": [
            [1.0, -0.08, 0.01],
            [-0.04, 1.0, -0.05],
            [0.01, -0.04, 1.0]
        ],
        "characteristic_curves": {
            "red": [[0.0, 0.0], [0.25, 0.25], [0.70, 0.75], [1.30, 1.40]],
            "green": [[0.0, 0.0], [0.30, 0.25], [0.85, 0.75], [1.45, 1.40]],
            "blue": [[0.0, 0.0], [0.20, 0.25], [0.62, 0.75], [1.18, 1.40]]
        },
        "white_anchor_percentile": 0.995,
        "confidence": 0.94,
        "validation": {
            "held_out_patch_count": 24,
            "delta_e00_rms": 1.8,
            "delta_e00_max": 4.9,
            "unit_slope_delta_e00_rms": 8.2,
            "worst_hue_family": "deep-blue"
        }
    })
}

fn scanner_linearization_json() -> serde_json::Value {
    serde_json::json!({
        "model_id": "synthetic-scanner-linearization-v1",
        "curves": {
            "red": [[0.0, 0.0], [0.25, 0.0625], [0.5, 0.25], [0.75, 0.5625], [1.0, 1.0]],
            "green": [[0.0, 0.0], [0.25, 0.0625], [0.5, 0.25], [0.75, 0.5625], [1.0, 1.0]],
            "blue": [[0.0, 0.0], [0.25, 0.0625], [0.5, 0.25], [0.75, 0.5625], [1.0, 1.0]]
        },
        "black_level_normalized": [0.01, 0.01, 0.01],
        "white_level_normalized": [0.99, 0.99, 0.99],
        "additive_flare_normalized": [0.005, 0.004, 0.003],
        "shading_gain_polynomial": [
            [1.0, 0.02, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.01, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.00, 0.0, 0.0, 0.0, 0.0]
        ],
        "confidence": 0.95,
        "validation": {
            "held_out_sample_count": 24,
            "transmittance_rmse": 0.002,
            "transmittance_max_error": 0.008,
            "identity_baseline_rmse": 0.08
        }
    })
}

fn nonlinear_center_scanner_linearization_json() -> serde_json::Value {
    serde_json::json!({
        "model_id": "synthetic-center-nonlinear-v1",
        "curves": {
            "red": [[0.0, 0.0], [0.25, 0.0625], [0.5, 0.25], [0.75, 0.5625], [1.0, 1.0]],
            "green": [[0.0, 0.0], [0.25, 0.0625], [0.5, 0.25], [0.75, 0.5625], [1.0, 1.0]],
            "blue": [[0.0, 0.0], [0.25, 0.0625], [0.5, 0.25], [0.75, 0.5625], [1.0, 1.0]]
        },
        "black_level_normalized": [0.0, 0.0, 0.0],
        "white_level_normalized": [1.0, 1.0, 1.0],
        "additive_flare_normalized": [0.0, 0.0, 0.0],
        "confidence": 0.95,
        "validation": {
            "held_out_sample_count": 24,
            "transmittance_rmse": 0.002,
            "transmittance_max_error": 0.008,
            "identity_baseline_rmse": 0.08
        }
    })
}

fn synthetic_scanner_settings_fingerprint() -> String {
    let settings = serde_json::json!({ "dpi": 3200, "mode": "positive" });
    scanstitch::color_calibration::scanner_settings_fingerprint(Some(&settings))
        .expect("synthetic scanner settings fingerprint")
}

fn valid_disjoint_fit_json(method: &str) -> serde_json::Value {
    serde_json::json!({
        "method": method,
        "patch_count": 12,
        "target_residual_rms": 0.01,
        "target_residual_max": 0.02,
        "validation": {
            "evaluation_set": "held_out",
            "training_patch_count": 12,
            "held_out_patch_count": 12,
            "training_residual_rms": 0.009,
            "training_residual_max": 0.02,
            "identity_baseline_residual_rms": 0.1,
            "identity_baseline_residual_max": 0.2,
            "held_out_identity_improvement_fraction": 0.9
        }
    })
}

fn base_only_roll_profile_json(profile_id: &str, scanner_profile_id: &str) -> String {
    serde_json::json!({
        "schema_version": 1,
        "record_type": "roll_profile",
        "profile_id": profile_id,
        "scanner_profile_id": scanner_profile_id,
        "film": { "stock": "Synthetic 200", "process": "C-41" },
        "base_color": [12000.0, 7000.0, 3000.0],
        "confidence": 0.80
    })
    .to_string()
}

fn film_hint_json() -> String {
    serde_json::json!({
        "schema_version": 1,
        "record_type": "film_hint",
        "film_id": "synthetic-200",
        "stock": "Synthetic 200",
        "aliases": ["Synthetic Color 200"],
        "similarity_tags": ["c41", "consumer"]
    })
    .to_string()
}

fn apply_rows(matrix: [[f64; 3]; 3], source: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * source[0] + matrix[0][1] * source[1] + matrix[0][2] * source[2],
        matrix[1][0] * source[0] + matrix[1][1] * source[1] + matrix[1][2] * source[2],
        matrix[2][0] * source[0] + matrix[2][1] * source[1] + matrix[2][2] * source[2],
    ]
}

#[test]
fn test_dng_matrix_can_use_bounded_equal_camera_white_inference() {
    let metadata = scanstitch::tiff_io::DngMetadata {
        color_matrix1: Some([
            [3.133_175_611_5, -1.544_623_017_3, -0.398_035_228_3],
            [-0.877_832_353_1, 1.722_482_085_2, 0.102_752_558_9],
            [-0.023_154_357_4, 0.019_029_228_0, 0.921_009_123_3],
        ]),
        ..Default::default()
    };

    let result = scanstitch::color_calibration::dng_color_matrix1_advisory_prior(&metadata)
        .expect("ColorMatrix1 should produce diagnostics");
    assert_eq!(result.diagnostics.status, "applied");
    let profile = result.profile.expect("bounded inferred scanner prior");
    assert_eq!(profile.confidence, 0.50);
    assert_eq!(
        profile.source_space["source_white_basis"],
        "inferred equal-camera neutral"
    );
    let mapped_neutral = apply_rows(profile.work_to_xyz, [1.0, 1.0, 1.0]);
    assert!((mapped_neutral[0] / mapped_neutral[1] - 0.9642).abs() < 2e-4);
    assert!((mapped_neutral[2] / mapped_neutral[1] - 0.8251).abs() < 2e-4);
}

#[test]
fn test_dng_matrix_does_not_infer_over_explicit_unsupported_illuminant() {
    let metadata = scanstitch::tiff_io::DngMetadata {
        calibration_illuminant1: Some(255),
        color_matrix1: Some([
            [3.133_175_611_5, -1.544_623_017_3, -0.398_035_228_3],
            [-0.877_832_353_1, 1.722_482_085_2, 0.102_752_558_9],
            [-0.023_154_357_4, 0.019_029_228_0, 0.921_009_123_3],
        ]),
        ..Default::default()
    };

    let result = scanstitch::color_calibration::dng_color_matrix1_advisory_prior(&metadata)
        .expect("ColorMatrix1 should produce rejection diagnostics");
    assert!(result.profile.is_none());
    assert_eq!(result.diagnostics.status, "rejected");
    assert!(result
        .diagnostics
        .rejection_details
        .iter()
        .any(|detail| detail.contains("refusing to guess")));
}

fn scanner_fit_patches() -> Vec<scanstitch::color_calibration::TargetPatch> {
    let matrix = [
        [0.7977, 0.1352, 0.0313],
        [0.2880, 0.7119, 0.0001],
        [0.0000, 0.0000, 0.8251],
    ];
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.25, 0.75, 0.35],
        [0.85, 0.20, 0.60],
    ]
    .into_iter()
    .enumerate()
    .map(
        |(idx, source_rgb)| scanstitch::color_calibration::TargetPatch {
            patch_id: Some(format!("patch-{idx}")),
            scanner_xy: None,
            source_rgb,
            reference_xyz: apply_rows(matrix, source_rgb),
        },
    )
    .collect()
}

fn disjoint_target_sources() -> [[f64; 3]; 24] {
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.25, 0.75, 0.35],
        [0.85, 0.20, 0.60],
        [0.10, 0.20, 0.90],
        [0.90, 0.60, 0.10],
        [0.50, 0.50, 0.50],
        [0.20, 0.30, 0.40],
        [0.65, 0.80, 0.30],
        [0.40, 0.15, 0.70],
        [0.95, 0.10, 0.15],
        [0.10, 0.95, 0.20],
        [0.15, 0.20, 0.95],
        [0.80, 0.80, 0.80],
        [0.70, 0.40, 0.20],
        [0.20, 0.60, 0.75],
        [0.60, 0.20, 0.80],
        [0.35, 0.90, 0.45],
        [0.90, 0.35, 0.65],
        [0.45, 0.25, 0.15],
        [0.30, 0.70, 0.10],
        [0.75, 0.55, 0.90],
    ]
}

fn disjoint_fit_patch_sets<F>(
    prefix: &str,
    reference: F,
) -> (
    Vec<scanstitch::color_calibration::TargetPatch>,
    Vec<scanstitch::color_calibration::TargetPatch>,
)
where
    F: Fn([f64; 3]) -> [f64; 3],
{
    let patches = disjoint_target_sources()
        .into_iter()
        .enumerate()
        .map(
            |(idx, source_rgb)| scanstitch::color_calibration::TargetPatch {
                patch_id: Some(format!(
                    "{prefix}-{}-{idx}",
                    if idx < 12 { "train" } else { "heldout" }
                )),
                scanner_xy: None,
                source_rgb,
                reference_xyz: reference(source_rgb),
            },
        )
        .collect::<Vec<_>>();
    (patches[..12].to_vec(), patches[12..].to_vec())
}

fn patch_measurements_json(
    patches: &[scanstitch::color_calibration::TargetPatch],
) -> Vec<serde_json::Value> {
    patches
        .iter()
        .map(|patch| {
            serde_json::json!({
                "id": patch.patch_id,
                "source_rgb": patch.source_rgb,
                "xyz": patch.reference_xyz,
            })
        })
        .collect()
}

fn scanner_fit_measurement_json() -> String {
    let matrix = [
        [0.7977, 0.1352, 0.0313],
        [0.2880, 0.7119, 0.0001],
        [0.0000, 0.0000, 0.8251],
    ];
    let (training_patches, held_out_patches) =
        disjoint_fit_patch_sets("scanner", |source| apply_rows(matrix, source));
    serde_json::json!({
        "source_space": { "name": "synthetic scanner RGB", "encoding": "linear" },
        "scanner": { "make": "Synthetic", "model": "Fitted Scanner" },
        "settings": { "dpi": 3200, "software": "unit-test", "mode": "raw" },
        "target": { "type": "synthetic transmissive target", "illuminant": "D50" },
        "reference": { "dataset": "synthetic XYZ" },
        "training_patches": patch_measurements_json(&training_patches),
        "held_out_patches": patch_measurements_json(&held_out_patches)
    })
    .to_string()
}

fn roll_fit_measurement_json() -> String {
    let scanner_matrix = [
        [0.7977, 0.1352, 0.0313],
        [0.2880, 0.7119, 0.0001],
        [0.0000, 0.0000, 0.8251],
    ];
    let roll_correction = [[1.05, 0.02, 0.0], [0.01, 0.96, 0.0], [0.0, 0.03, 1.02]];
    let (training_patches, held_out_patches) = disjoint_fit_patch_sets("roll", |source_rgb| {
        apply_rows(roll_correction, apply_rows(scanner_matrix, source_rgb))
    });
    serde_json::json!({
        "film": { "stock": "Synthetic 200", "process": "C-41" },
        "target": { "type": "synthetic roll target" },
        "reference": { "dataset": "synthetic roll XYZ" },
        "base_color": [12000.0, 7000.0, 3000.0],
        "training_patches": patch_measurements_json(&training_patches),
        "held_out_patches": patch_measurements_json(&held_out_patches)
    })
    .to_string()
}

#[test]
fn test_valid_calibration_profile_parses() {
    let result = scanstitch::color_calibration::parse_profile_json(&valid_profile_json(), None);

    let profile = result.profile.expect("valid profile should parse");
    assert_eq!(result.diagnostics.status, "applied");
    assert_eq!(result.diagnostics.source, "external_calibration_profile");
    assert_eq!(profile.schema_version, 1);
    assert_eq!(profile.whitepoint, [0.9642, 1.0, 0.8251]);
    assert_eq!(profile.confidence, 0.95);
    assert!(profile.matrix_condition_number.is_finite());
}

#[test]
fn test_documented_calibration_examples_match_runtime_contract() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let profile_contents =
        std::fs::read_to_string(root.join("docs/color-calibration-profile.example.json")).unwrap();
    let profile = scanstitch::color_calibration::parse_profile_json(&profile_contents, None);
    assert!(
        profile.profile.is_some(),
        "documented compatibility profile was rejected: {:?}",
        profile.diagnostics.rejection_details
    );

    let examples: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("docs/color-calibration-library-record.examples.json"))
            .unwrap(),
    )
    .unwrap();
    for (name, record) in examples.as_object().unwrap() {
        scanstitch::color_calibration::validate_library_record_value(record, None).unwrap_or_else(
            |rejection| panic!("documented library example `{name}` was rejected: {rejection:?}"),
        );
    }
}

#[test]
fn test_calibrate_cli_writes_and_loader_recomputes_selected_root_polynomial() {
    let tmp = tempfile::TempDir::new().unwrap();
    let library = tmp.path().join("calibration");
    let training = synthetic_root_polynomial_patches(48, "cli-train", 0, true);
    let held_out = synthetic_root_polynomial_patches(36, "cli-held", 409, true);
    let measurements = serde_json::json!({
        "schema_version": 2,
        "scanner": { "make": "Synthetic", "model": "Nonlinear CLI" },
        "settings": { "dpi": 3200, "mode": "raw-positive", "bit_depth": 16 },
        "target": { "type": "synthetic_transmissive_target", "illuminant": "D50" },
        "reference": { "dataset": "synthetic-root-polynomial" },
        "whitepoint": [0.9642, 1.0, 0.8251],
        "confidence": 0.98,
        "training_patches": training,
        "held_out_patches": held_out,
    });
    let measurement_path = tmp.path().join("nonlinear-target.json");
    std::fs::write(&measurement_path, measurements.to_string()).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .arg("scanner-target")
        .arg("--library")
        .arg(&library)
        .arg("--profile-id")
        .arg("nonlinear-scanner")
        .arg("--measurements")
        .arg(&measurement_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "scanner calibration failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record_path = library.join("scanners/nonlinear-scanner.json");
    let mut record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&record_path).unwrap()).unwrap();
    assert_eq!(
        record["polynomial_fit"]["status"],
        "selected_higher_order_model"
    );
    assert_eq!(record["color_model"]["degree"], 2);
    assert_eq!(
        record["color_model"]["validation"]["selected_over_matrix"],
        true
    );
    assert!(record.get("training_patches").is_none());
    assert_eq!(
        record["color_model"]["training_patches"]
            .as_array()
            .map(Vec::len),
        Some(48)
    );
    assert_eq!(record["patches"].as_array().map(Vec::len), Some(36));

    let loaded = scanstitch::color_calibration::load_calibration(
        None,
        Some(&library),
        Some("nonlinear-scanner"),
        None,
        None,
        None,
    );
    let profile = loaded.profile.expect("validated nonlinear scanner profile");
    assert_eq!(
        profile.color_model.as_ref().map(|model| model.degree),
        Some(2)
    );
    assert_eq!(
        loaded
            .diagnostics
            .scanner_profile
            .as_ref()
            .and_then(|scanner| scanner.transform_type.as_deref()),
        Some("matrix_with_validated_quadratic_root_polynomial")
    );

    let model_id = profile
        .color_model
        .as_ref()
        .expect("selected nonlinear model")
        .model_id
        .clone();
    let correction = [[1.02, 0.01, 0.0], [0.0, 0.98, 0.01], [0.01, 0.0, 1.01]];
    let roll_sources = synthetic_root_polynomial_patches(24, "cli-roll-source", 733, true);
    let roll_patches = roll_sources
        .iter()
        .map(|patch| scanstitch::color_calibration::TargetPatch {
            patch_id: patch.patch_id.clone(),
            scanner_xy: None,
            source_rgb: patch.source_rgb,
            reference_xyz: apply_rows(
                correction,
                profile.calibrated_source_to_xyz(patch.source_rgb),
            ),
        })
        .collect::<Vec<_>>();
    let roll_measurements = serde_json::json!({
        "schema_version": 2,
        "film": { "stock": "Synthetic nonlinear roll", "process": "C-41" },
        "target": { "type": "synthetic roll target" },
        "reference": { "dataset": "synthetic nonlinear roll XYZ" },
        "training_patches": patch_measurements_json(&roll_patches[..12]),
        "held_out_patches": patch_measurements_json(&roll_patches[12..]),
    });
    let roll_measurement_path = tmp.path().join("nonlinear-roll-target.json");
    std::fs::write(&roll_measurement_path, roll_measurements.to_string()).unwrap();
    let roll_output = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .arg("roll-target")
        .arg("--library")
        .arg(&library)
        .arg("--profile-id")
        .arg("nonlinear-roll")
        .arg("--scanner-profile")
        .arg("nonlinear-scanner")
        .arg("--measurements")
        .arg(&roll_measurement_path)
        .output()
        .unwrap();
    assert!(
        roll_output.status.success(),
        "roll calibration failed: {}",
        String::from_utf8_lossy(&roll_output.stderr)
    );
    let roll_record_path = library.join("rolls/nonlinear-roll.json");
    let mut roll_record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&roll_record_path).unwrap()).unwrap();
    assert_eq!(
        roll_record["scanner_color_model_application"]["transform"],
        "root_polynomial_degree_2"
    );
    assert_eq!(
        roll_record["scanner_color_model_application"]["model_id"],
        model_id
    );
    let combined = scanstitch::color_calibration::load_calibration(
        None,
        Some(&library),
        Some("nonlinear-scanner"),
        Some("nonlinear-roll"),
        None,
        None,
    );
    assert!(
        combined
            .profile
            .as_ref()
            .is_some_and(|profile| profile.roll_correction_applied),
        "nonlinear scanner and fitted roll correction should compose: {:?}",
        combined.diagnostics
    );

    let valid_roll_record = roll_record.clone();
    roll_record
        .as_object_mut()
        .unwrap()
        .remove("scanner_color_model_application");
    std::fs::write(&roll_record_path, roll_record.to_string()).unwrap();
    let rejected_roll = scanstitch::color_calibration::load_calibration(
        None,
        Some(&library),
        Some("nonlinear-scanner"),
        Some("nonlinear-roll"),
        None,
        None,
    );
    assert!(rejected_roll.profile.is_none());
    assert!(rejected_roll
        .diagnostics
        .roll_profile
        .as_ref()
        .is_some_and(|roll| roll.rejection_details.iter().any(|reason| {
            reason.contains("requires scanner_color_model_application evidence")
        })));
    std::fs::write(&roll_record_path, valid_roll_record.to_string()).unwrap();

    record["color_model"]["coefficients"][3][0] = serde_json::json!(0.75);
    std::fs::write(&record_path, record.to_string()).unwrap();
    let rejected = scanstitch::color_calibration::load_calibration(
        None,
        Some(&library),
        Some("nonlinear-scanner"),
        None,
        None,
        None,
    );
    assert!(rejected.profile.is_none());
    assert!(rejected
        .diagnostics
        .library
        .as_ref()
        .unwrap()
        .invalid_entries
        .iter()
        .flat_map(|entry| &entry.reasons)
        .any(|reason| {
            reason.contains("inconsistent with retained held-out patches")
                || reason.contains("deterministic robust fit")
                || reason.contains("coefficient_max_abs")
        }));
}

#[test]
fn test_calibrate_cli_writes_executes_and_revalidates_selected_residual_lut_3d() {
    let tmp = tempfile::TempDir::new().unwrap();
    let library = tmp.path().join("calibration");
    let training = synthetic_residual_lut_patches(180, "cli-lut-train", 0, true, true);
    let held_out = synthetic_residual_lut_patches(80, "cli-lut-held", 431, false, true);
    let measurements = serde_json::json!({
        "schema_version": 2,
        "scanner": { "make": "Synthetic", "model": "Residual LUT CLI" },
        "settings": { "dpi": 3200, "mode": "raw-positive", "bit_depth": 16 },
        "target": { "type": "synthetic_transmissive_target", "illuminant": "D50" },
        "reference": { "dataset": "synthetic-smooth-residual-lut" },
        "whitepoint": [0.9642, 1.0, 0.8251],
        "confidence": 0.99,
        "training_patches": training,
        "held_out_patches": held_out.clone(),
    });
    let measurement_path = tmp.path().join("residual-lut-target.json");
    std::fs::write(&measurement_path, measurements.to_string()).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .arg("scanner-target")
        .arg("--library")
        .arg(&library)
        .arg("--profile-id")
        .arg("residual-lut-scanner")
        .arg("--measurements")
        .arg(&measurement_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "scanner residual LUT calibration failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let record_path = library.join("scanners/residual-lut-scanner.json");
    let mut record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&record_path).unwrap()).unwrap();
    assert_eq!(record["lut_3d_fit"]["status"], "selected_residual_lut_3d");
    assert_eq!(record["lut_3d_fit"]["selected_grid_size"], 5);
    assert_eq!(
        record["lut_3d_model"]["model_type"],
        "smooth_residual_lut_3d"
    );
    assert_eq!(record["lut_3d_model"]["interpolation"], "tetrahedral");
    assert_eq!(
        record["lut_3d_model"]["residual_nodes_xyz"]
            .as_array()
            .map(Vec::len),
        Some(125)
    );
    assert_eq!(
        record["lut_3d_model"]["training_patches"]
            .as_array()
            .map(Vec::len),
        Some(180)
    );
    assert_eq!(record["patches"].as_array().map(Vec::len), Some(80));

    let loaded = scanstitch::color_calibration::load_calibration(
        None,
        Some(&library),
        Some("residual-lut-scanner"),
        None,
        None,
        None,
    );
    let profile = loaded
        .profile
        .expect("validated residual LUT scanner profile");
    let model = profile
        .lut_3d_model
        .as_ref()
        .expect("selected residual LUT model");
    assert_eq!(model.grid_size, 5);
    assert!(loaded
        .diagnostics
        .scanner_profile
        .as_ref()
        .and_then(|scanner| scanner.transform_type.as_deref())
        .is_some_and(|transform| transform.contains("5x5x5_residual_lut")));

    let supported = held_out
        .iter()
        .filter(|patch| {
            scanstitch::color_calibration::residual_lut_3d_has_full_support(model, patch.source_rgb)
        })
        .take(24)
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(supported.len(), 24);
    let correction = [[1.015, 0.005, 0.0], [0.0, 0.985, 0.01], [0.005, 0.0, 1.01]];
    let roll_patches = supported
        .iter()
        .map(|patch| scanstitch::color_calibration::TargetPatch {
            patch_id: patch.patch_id.clone(),
            scanner_xy: None,
            source_rgb: patch.source_rgb,
            reference_xyz: apply_rows(
                correction,
                profile.calibrated_source_to_xyz(patch.source_rgb),
            ),
        })
        .collect::<Vec<_>>();
    let roll_measurements = serde_json::json!({
        "schema_version": 2,
        "film": { "stock": "Synthetic LUT roll", "process": "C-41" },
        "target": { "type": "synthetic roll target" },
        "reference": { "dataset": "synthetic LUT roll XYZ" },
        "training_patches": patch_measurements_json(&roll_patches[..12]),
        "held_out_patches": patch_measurements_json(&roll_patches[12..]),
    });
    let roll_measurement_path = tmp.path().join("residual-lut-roll-target.json");
    std::fs::write(&roll_measurement_path, roll_measurements.to_string()).unwrap();
    let roll_output = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .arg("roll-target")
        .arg("--library")
        .arg(&library)
        .arg("--profile-id")
        .arg("residual-lut-roll")
        .arg("--scanner-profile")
        .arg("residual-lut-scanner")
        .arg("--measurements")
        .arg(&roll_measurement_path)
        .output()
        .unwrap();
    assert!(
        roll_output.status.success(),
        "roll residual LUT calibration failed: {}",
        String::from_utf8_lossy(&roll_output.stderr)
    );
    let roll_record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(library.join("rolls/residual-lut-roll.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        roll_record["scanner_color_model_application"]["transform"],
        "residual_lut_3d_grid_5"
    );
    assert_eq!(
        roll_record["scanner_color_model_application"]["model_id"],
        model.model_id.as_str()
    );
    assert!(scanstitch::color_calibration::load_calibration(
        None,
        Some(&library),
        Some("residual-lut-scanner"),
        Some("residual-lut-roll"),
        None,
        None,
    )
    .profile
    .is_some());

    record["lut_3d_model"]["residual_nodes_xyz"][31][0] = serde_json::json!(0.125);
    std::fs::write(&record_path, record.to_string()).unwrap();
    let rejected = scanstitch::color_calibration::load_calibration(
        None,
        Some(&library),
        Some("residual-lut-scanner"),
        None,
        None,
        None,
    );
    assert!(rejected.profile.is_none());
    assert!(rejected
        .diagnostics
        .library
        .as_ref()
        .unwrap()
        .invalid_entries
        .iter()
        .flat_map(|entry| &entry.reasons)
        .any(|reason| reason.contains("deterministic robust fit")));
}

#[test]
fn test_fit_scanner_matrix_from_synthetic_target_patches() {
    let expected = [
        [0.7977, 0.1352, 0.0313],
        [0.2880, 0.7119, 0.0001],
        [0.0000, 0.0000, 0.8251],
    ];
    let (training, held_out) =
        disjoint_fit_patch_sets("matrix-fit", |source| apply_rows(expected, source));
    let fit = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training,
        &held_out,
        "least_squares_rgb_to_xyz",
        None,
        None,
    )
    .expect("fit should recover synthetic scanner matrix");
    for (row, expected_row) in expected.iter().enumerate() {
        for (col, expected_value) in expected_row.iter().enumerate() {
            assert!(
                (fit.matrix[row][col] - *expected_value).abs() < 1e-10,
                "matrix mismatch at [{row}][{col}]"
            );
        }
    }
    assert_eq!(fit.fit.patch_count, 12);
    assert!(fit.fit.target_residual_rms < 1e-12);
    assert!(
        !fit.fit.per_hue_residuals.is_empty(),
        "scanner target fit should report per-hue residual summaries"
    );
    assert_eq!(fit.fit.worst_patches.len(), 5);
    assert!(fit
        .fit
        .worst_patches
        .iter()
        .all(|patch| patch.residual_error < 1e-12));
    assert!(fit.confidence > 0.999);
    assert!((fit.whitepoint[0] - 0.9642).abs() < 1e-4);
}

#[test]
fn test_disjoint_target_fit_scores_only_held_out_patches() {
    let expected = [
        [0.7977, 0.1352, 0.0313],
        [0.2880, 0.7119, 0.0001],
        [0.0000, 0.0000, 0.8251],
    ];
    let (training, held_out) =
        disjoint_fit_patch_sets("api", |source| apply_rows(expected, source));
    let fit = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training,
        &held_out,
        "least_squares_rgb_to_xyz",
        None,
        None,
    )
    .expect("an exact matrix should generalize to disjoint target samples");

    for (row, expected_row) in expected.iter().enumerate() {
        for (column, expected_value) in expected_row.iter().enumerate() {
            assert!((fit.matrix[row][column] - expected_value).abs() < 1e-10);
        }
    }
    assert_eq!(fit.fit.patch_count, 12);
    assert!(fit.fit.target_residual_rms < 1e-12);
    let validation = fit
        .fit
        .validation
        .expect("held-out fit should carry explicit validation provenance");
    assert_eq!(validation.evaluation_set, "held_out");
    assert_eq!(validation.training_patch_count, 12);
    assert_eq!(validation.held_out_patch_count, 12);
    assert!(validation.training_residual_rms < 1e-12);
    assert!(validation.identity_baseline_residual_rms > 0.05);
    assert!(validation.held_out_identity_improvement_fraction > 0.99);
}

#[test]
fn test_disjoint_target_fit_rejects_overlap_and_held_out_failure() {
    let training_matrix = [
        [0.7977, 0.1352, 0.0313],
        [0.2880, 0.7119, 0.0001],
        [0.0000, 0.0000, 0.8251],
    ];
    let (training, mut held_out) =
        disjoint_fit_patch_sets("adversarial", |source| apply_rows(training_matrix, source));
    held_out[0].patch_id = training[0].patch_id.clone();
    let overlap = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training,
        &held_out,
        "least_squares_rgb_to_xyz",
        Some([0.9642, 1.0, 0.8251]),
        None,
    )
    .unwrap_err();
    assert!(overlap.contains("globally unique"));

    let (_, mut duplicated_values) = disjoint_fit_patch_sets("duplicate-values", |source| {
        apply_rows(training_matrix, source)
    });
    duplicated_values[0].source_rgb = training[0].source_rgb;
    duplicated_values[0].reference_xyz = training[0].reference_xyz;
    let duplicate = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training,
        &duplicated_values,
        "least_squares_rgb_to_xyz",
        Some([0.9642, 1.0, 0.8251]),
        None,
    )
    .unwrap_err();
    assert!(duplicate.contains("exactly duplicates"));

    let (_, held_out) = disjoint_fit_patch_sets("wrong-holdout", |source| {
        [source[0] * 0.05, source[1] * 0.05, source[2] * 0.05]
    });
    let error = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training,
        &held_out,
        "least_squares_rgb_to_xyz",
        Some([0.9642, 1.0, 0.8251]),
        None,
    )
    .unwrap_err();
    assert!(error.contains("held-out target confidence"));
}

#[test]
fn test_target_matrix_measurement_rejects_legacy_single_patch_set() {
    let value = serde_json::json!({
        "patches": patch_measurements_json(&scanner_fit_patches())
    });
    let error = scanstitch::color_calibration::disjoint_target_patch_sets_from_measurement(&value)
        .unwrap_err();
    assert!(error.contains("in-sample residuals are not independent"));
}

#[test]
fn test_target_fit_rejects_too_few_and_singular_patches() {
    let matrix = [
        [0.7977, 0.1352, 0.0313],
        [0.2880, 0.7119, 0.0001],
        [0.0000, 0.0000, 0.8251],
    ];
    let (training, held_out) =
        disjoint_fit_patch_sets("fit-errors", |source| apply_rows(matrix, source));
    let too_few = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training[..3],
        &held_out,
        "least_squares_rgb_to_xyz",
        None,
        None,
    )
    .unwrap_err();
    assert!(too_few.contains("at least"));

    let singular_source = (0..12)
        .map(|index| scanstitch::color_calibration::TargetPatch {
            patch_id: Some(format!("singular-train-{index}")),
            scanner_xy: None,
            source_rgb: [1.0, 1.0, 1.0],
            reference_xyz: [0.9 + index as f64 * 0.001, 1.0, 0.8],
        })
        .collect::<Vec<_>>();
    let singular = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &singular_source,
        &held_out,
        "least_squares_rgb_to_xyz",
        Some([0.9642, 1.0, 0.8251]),
        None,
    )
    .unwrap_err();
    assert!(singular.contains("singular") || singular.contains("condition number"));
}

#[test]
fn test_target_measurement_parses_lab_references() {
    let value = serde_json::json!({
        "patches": [
            { "id": "white", "rgb": [1.0, 1.0, 1.0], "lab": [100.0, 0.0, 0.0] },
            { "id": "gray", "rgb": [0.5, 0.5, 0.5], "lab": [50.0, 0.0, 0.0] },
            { "id": "red", "rgb": [1.0, 0.0, 0.0], "lab": [60.0, 60.0, 40.0] },
            { "id": "blue", "rgb": [0.0, 0.0, 1.0], "lab": [35.0, 20.0, -50.0] }
        ]
    });

    let patches = scanstitch::color_calibration::target_patches_from_measurement(&value)
        .expect("Lab references should parse");

    assert_eq!(patches.len(), 4);
    assert!((patches[0].reference_xyz[0] - 0.9642).abs() < 1e-4);
    assert!((patches[0].reference_xyz[1] - 1.0).abs() < 1e-12);
    assert!((patches[0].reference_xyz[2] - 0.8251).abs() < 1e-4);
}

#[test]
fn test_invalid_json_profile_is_rejected() {
    let result = scanstitch::color_calibration::parse_profile_json("{ not json", None);

    assert!(result.profile.is_none());
    assert_eq!(result.diagnostics.status, "rejected");
    assert!(result.diagnostics.reason.contains("failed to parse"));
}

#[test]
fn test_profile_rejects_unknown_strict_nested_calibration_fields() {
    let mut profile_with_bad_gamut: serde_json::Value =
        serde_json::from_str(&valid_profile_json()).unwrap();
    profile_with_bad_gamut["gamut_limits"]["unexpected"] = serde_json::json!(true);
    let result = scanstitch::color_calibration::parse_profile_json(
        &profile_with_bad_gamut.to_string(),
        None,
    );
    assert!(result.profile.is_none());
    assert_eq!(result.diagnostics.status, "rejected");
    assert!(result
        .diagnostics
        .rejection_details
        .iter()
        .any(|detail| detail.contains("unknown field `unexpected`")));

    let mut profile_with_bad_fit: serde_json::Value =
        serde_json::from_str(&valid_profile_json()).unwrap();
    profile_with_bad_fit["fit"] = serde_json::json!({
        "method": "least_squares_rgb_to_xyz",
        "patch_count": 4,
        "target_residual_rms": 0.01,
        "target_residual_max": 0.02,
        "ignored_metric": 1.0
    });
    let result =
        scanstitch::color_calibration::parse_profile_json(&profile_with_bad_fit.to_string(), None);
    assert!(result.profile.is_none());
    assert_eq!(result.diagnostics.status, "rejected");
    assert!(result
        .diagnostics
        .rejection_details
        .iter()
        .any(|detail| detail.contains("unknown field `ignored_metric`")));
}

#[test]
fn test_schema_v2_fitted_record_rejects_in_sample_only_diagnostics() {
    let mut record: serde_json::Value =
        serde_json::from_str(&scanner_profile_json("scanner-in-sample", 0.95, false)).unwrap();
    record["schema_version"] = serde_json::json!(2);
    record["fit"] = serde_json::json!({
        "method": "least_squares_rgb_to_xyz",
        "patch_count": 24,
        "target_residual_rms": 0.001,
        "target_residual_max": 0.003
    });

    let rejection = scanstitch::color_calibration::validate_library_record_value(&record, None)
        .expect_err("schema-v2 fit claims must prove held-out evaluation");
    assert!(rejection.reasons.iter().any(
        |reason| reason.contains("in-sample residuals cannot establish calibration confidence")
    ));
}

#[test]
fn test_schema_v2_fitted_scanner_requires_exact_runtime_signal_domain() {
    let mut record: serde_json::Value =
        serde_json::from_str(&scanner_profile_json("scanner-domain", 0.95, false)).unwrap();
    record["schema_version"] = serde_json::json!(2);
    record["fit"] = valid_disjoint_fit_json("least_squares_rgb_to_xyz");
    record["scanner_linearization"] = nonlinear_center_scanner_linearization_json();

    let missing = scanstitch::color_calibration::validate_library_record_value(&record, None)
        .expect_err("fitted scanner matrix without signal-domain provenance must fail closed");
    assert!(missing
        .reasons
        .iter()
        .any(|reason| reason.contains("target_patch_signal_domain is required")));

    record["target_patch_signal_domain"] = serde_json::json!("normalized_scanner_signal");
    let wrong_domain = scanstitch::color_calibration::validate_library_record_value(&record, None)
        .expect_err("matrix fitted before declared scanner linearization must be rejected");
    assert!(wrong_domain.reasons.iter().any(|reason| reason
        .contains("paired with scanner_linearization must use target_patch_signal_domain")));

    record["target_patch_signal_domain"] = serde_json::json!("scanner_linearized_transmittance");
    record["target_patch_linearization_model_id"] = serde_json::json!("wrong-model");
    let wrong_model = scanstitch::color_calibration::validate_library_record_value(&record, None)
        .expect_err("matrix fitted with a different linearization model must be rejected");
    assert!(wrong_model
        .reasons
        .iter()
        .any(|reason| reason.contains("must match scanner_linearization.model_id")));

    record["target_patch_linearization_model_id"] =
        serde_json::json!("synthetic-center-nonlinear-v1");
    scanstitch::color_calibration::validate_library_record_value(&record, None)
        .expect("matching runtime signal-domain provenance should validate");
}

#[test]
fn test_fitted_roll_signal_domain_must_match_selected_scanner() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    std::fs::create_dir_all(tmp.path().join("rolls")).unwrap();

    let mut scanner: serde_json::Value =
        serde_json::from_str(&scanner_profile_json("scanner-domain", 0.96, false)).unwrap();
    scanner["schema_version"] = serde_json::json!(2);
    scanner["fit"] = valid_disjoint_fit_json("least_squares_rgb_to_xyz");
    scanner["scanner_linearization"] = nonlinear_center_scanner_linearization_json();
    scanner["target_patch_signal_domain"] = serde_json::json!("scanner_linearized_transmittance");
    scanner["target_patch_linearization_model_id"] =
        serde_json::json!("synthetic-center-nonlinear-v1");
    std::fs::write(
        tmp.path().join("scanners/scanner.json"),
        scanner.to_string(),
    )
    .unwrap();

    let mut roll: serde_json::Value =
        serde_json::from_str(&roll_profile_json("roll-domain", "scanner-domain")).unwrap();
    roll["schema_version"] = serde_json::json!(2);
    roll["fit"] = valid_disjoint_fit_json("least_squares_xyz_post_scanner_correction");
    roll["target_patch_signal_domain"] = serde_json::json!("normalized_scanner_signal");
    std::fs::write(tmp.path().join("rolls/roll.json"), roll.to_string()).unwrap();

    let mismatch = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-domain"),
        Some("roll-domain"),
        None,
        None,
    );
    assert!(mismatch.profile.is_none());
    assert!(mismatch
        .diagnostics
        .roll_profile
        .as_ref()
        .expect("roll rejection diagnostics")
        .rejection_details
        .iter()
        .any(|reason| reason.contains("does not match selected scanner runtime domain")));

    roll["target_patch_signal_domain"] = serde_json::json!("scanner_linearized_transmittance");
    roll["target_patch_linearization_model_id"] =
        serde_json::json!("synthetic-center-nonlinear-v1");
    std::fs::write(tmp.path().join("rolls/roll.json"), roll.to_string()).unwrap();
    let matching = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-domain"),
        Some("roll-domain"),
        None,
        None,
    );
    assert!(
        matching.profile.is_some(),
        "matching fitted scanner/roll domains should apply: {:?}",
        matching.diagnostics
    );
}

#[test]
fn test_unsupported_schema_version_is_rejected() {
    let mut value: serde_json::Value = serde_json::from_str(&valid_profile_json()).unwrap();
    value["schema_version"] = serde_json::json!(99);

    let result = scanstitch::color_calibration::parse_profile_json(&value.to_string(), None);

    assert!(result.profile.is_none());
    assert_eq!(result.diagnostics.status, "rejected");
    assert!(result
        .diagnostics
        .rejection_details
        .iter()
        .any(|detail| detail.contains("unsupported schema_version")));
}

#[test]
fn test_singular_matrix_is_rejected() {
    let mut value: serde_json::Value = serde_json::from_str(&valid_profile_json()).unwrap();
    value["work_to_xyz"] = serde_json::json!([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);

    let result = scanstitch::color_calibration::parse_profile_json(&value.to_string(), None);

    assert!(result.profile.is_none());
    assert!(result
        .diagnostics
        .rejection_details
        .iter()
        .any(|detail| detail.contains("singular")));
}

#[test]
fn test_bad_whitepoint_is_rejected() {
    let mut value: serde_json::Value = serde_json::from_str(&valid_profile_json()).unwrap();
    value["whitepoint"] = serde_json::json!([0.1, 0.1, 8.0]);

    let result = scanstitch::color_calibration::parse_profile_json(&value.to_string(), None);

    assert!(result.profile.is_none());
    assert!(result
        .diagnostics
        .rejection_details
        .iter()
        .any(|detail| detail.contains("whitepoint")));
}

#[test]
fn test_low_confidence_profile_is_rejected() {
    let mut value: serde_json::Value = serde_json::from_str(&valid_profile_json()).unwrap();
    value["confidence"] = serde_json::json!(0.25);

    let result = scanstitch::color_calibration::parse_profile_json(&value.to_string(), None);

    assert!(result.profile.is_none());
    assert!(result
        .diagnostics
        .rejection_details
        .iter()
        .any(|detail| detail.contains("below required minimum")));
}

#[test]
fn test_library_scanner_roll_profiles_parse_and_compose_in_order() {
    let tmp = tempfile::TempDir::new().unwrap();
    let scanners = tmp.path().join("scanners");
    let rolls = tmp.path().join("rolls");
    let hints = tmp.path().join("film_hints");
    std::fs::create_dir_all(&scanners).unwrap();
    std::fs::create_dir_all(&rolls).unwrap();
    std::fs::create_dir_all(&hints).unwrap();
    std::fs::write(
        scanners.join("scanner.json"),
        scanner_profile_json("scanner-a", 0.96, false),
    )
    .unwrap();
    std::fs::write(
        rolls.join("roll.json"),
        roll_profile_json("roll-a", "scanner-a"),
    )
    .unwrap();
    std::fs::write(hints.join("hint.json"), film_hint_json()).unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-a"),
        Some("roll-a"),
        None,
        Some([11900.0, 7100.0, 3100.0]),
    );

    let profile = result.profile.expect("scanner+roll profile should apply");
    assert_eq!(result.diagnostics.status, "applied");
    assert_eq!(result.diagnostics.source, "calibration_library");
    assert_eq!(
        profile.application_mode,
        scanstitch::color_calibration::CalibrationApplicationMode::DirectProfile
    );
    assert!(profile.roll_correction_applied);
    assert_eq!(profile.scanner_profile_id.as_deref(), Some("scanner-a"));
    assert_eq!(profile.roll_profile_id.as_deref(), Some("roll-a"));
    assert_eq!(
        profile.scanner_prior_work_to_xyz,
        Some([[0.80, 0.10, 0.0], [0.20, 0.70, 0.10], [0.0, 0.20, 0.80]])
    );
    assert_eq!(profile.scanner_prior_confidence, Some(0.96));
    let expected = [[0.88, 0.11, 0.0], [0.18, 0.63, 0.09], [0.0, 0.2, 0.8]];
    for (row, expected_row) in expected.iter().enumerate() {
        for (col, expected_value) in expected_row.iter().enumerate() {
            assert!(
                (profile.work_to_xyz[row][col] - *expected_value).abs() < 1e-12,
                "matrix mismatch at [{row}][{col}]"
            );
        }
    }
    assert!(result
        .diagnostics
        .roll_profile
        .as_ref()
        .and_then(|roll| roll.base_color_delta)
        .is_some());
    assert_eq!(
        result
            .diagnostics
            .library
            .as_ref()
            .map(|library| library.film_hints_loaded),
        Some(1)
    );
}

#[test]
fn test_scanner_profile_carries_validated_signal_linearization() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    let mut scanner: serde_json::Value =
        serde_json::from_str(&scanner_profile_json("scanner-linear", 0.96, false)).unwrap();
    scanner["schema_version"] = serde_json::json!(2);
    scanner["scanner_linearization"] = scanner_linearization_json();
    std::fs::write(
        tmp.path().join("scanners/scanner.json"),
        scanner.to_string(),
    )
    .unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-linear"),
        None,
        None,
        None,
    );
    assert_eq!(result.diagnostics.status, "applied");
    let profile = result.profile.expect("scanner profile");
    assert_eq!(
        profile
            .scanner_linearization
            .as_ref()
            .map(|model| model.model_id.as_str()),
        Some("synthetic-scanner-linearization-v1")
    );
    assert_eq!(
        result
            .diagnostics
            .scanner_profile
            .as_ref()
            .and_then(|scanner| scanner.scanner_linearization.as_ref())
            .map(|model| model.validation.held_out_sample_count),
        Some(24)
    );
}

#[test]
fn test_scanner_profile_rejects_unvalidated_signal_linearization() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    let mut scanner: serde_json::Value =
        serde_json::from_str(&scanner_profile_json("scanner-linear", 0.96, false)).unwrap();
    let mut linearization = scanner_linearization_json();
    linearization["validation"]["transmittance_rmse"] = serde_json::json!(0.2);
    scanner["scanner_linearization"] = linearization;
    std::fs::write(
        tmp.path().join("scanners/scanner.json"),
        scanner.to_string(),
    )
    .unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-linear"),
        None,
        None,
        None,
    );
    assert!(result.profile.is_none());
    assert!(result
        .diagnostics
        .library
        .as_ref()
        .expect("library diagnostics")
        .invalid_entries
        .iter()
        .flat_map(|entry| &entry.reasons)
        .any(|reason| reason.contains("held-out transmittance RMSE")));
}

#[test]
fn test_v2_library_records_preserve_perfect_mode_calibration_metadata() {
    let tmp = tempfile::TempDir::new().unwrap();
    let scanners = tmp.path().join("scanners");
    let rolls = tmp.path().join("rolls");
    std::fs::create_dir_all(&scanners).unwrap();
    std::fs::create_dir_all(&rolls).unwrap();

    let mut scanner: serde_json::Value =
        serde_json::from_str(&scanner_profile_json("scanner-v2", 0.96, false)).unwrap();
    scanner["schema_version"] =
        serde_json::json!(scanstitch::color_calibration::CALIBRATION_LIBRARY_SCHEMA_VERSION);
    scanner["response_curves"] = serde_json::json!({ "red": [[0.0, 0.0], [1.0, 1.0]] });
    scanner["flare_black_white_diagnostics"] = serde_json::json!({
        "black_level": [16.0, 17.0, 18.0],
        "white_level": [65535.0, 65535.0, 65535.0],
        "flare_check": "passed"
    });
    scanner["polynomial_fit"] = serde_json::json!({ "order": 2, "status": "candidate" });
    scanner["lut_3d"] = serde_json::json!({ "size": 17, "status": "candidate" });
    scanner["delta_e00_summary"] = serde_json::json!({ "rms": 1.1, "max": 3.2 });

    let mut roll: serde_json::Value =
        serde_json::from_str(&roll_profile_json("roll-v2", "scanner-v2")).unwrap();
    roll["schema_version"] =
        serde_json::json!(scanstitch::color_calibration::CALIBRATION_LIBRARY_SCHEMA_VERSION);
    roll["hue_family_residuals"] = serde_json::json!({
        "neutral": { "rms": 0.9, "max": 2.1 }
    });
    roll["delta_e00_summary"] = serde_json::json!({ "rms": 1.8, "max": 4.4 });

    std::fs::write(scanners.join("scanner.json"), scanner.to_string()).unwrap();
    std::fs::write(rolls.join("roll.json"), roll.to_string()).unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-v2"),
        Some("roll-v2"),
        None,
        Some([12000.0, 7000.0, 3000.0]),
    );

    let profile = result
        .profile
        .expect("v2 scanner+roll profile should apply");
    assert_eq!(
        profile.schema_version,
        scanstitch::color_calibration::CALIBRATION_LIBRARY_SCHEMA_VERSION
    );
    assert_eq!(result.diagnostics.calibration_upgrade_available, None);
    let scanner = result
        .diagnostics
        .scanner_profile
        .as_ref()
        .expect("scanner diagnostics");
    assert!(scanner.response_curves.is_some());
    assert!(scanner.flare_black_white_diagnostics.is_some());
    assert_eq!(
        scanner.transform_type.as_deref(),
        Some("matrix_with_recorded_3d_lut_candidate")
    );
    assert_eq!(
        scanner
            .delta_e00_summary
            .as_ref()
            .and_then(|value| value.get("rms"))
            .and_then(serde_json::Value::as_f64),
        Some(1.1)
    );
    let roll = result
        .diagnostics
        .roll_profile
        .as_ref()
        .expect("roll diagnostics");
    assert_eq!(roll.correction_transform_type.as_deref(), Some("matrix"));
    assert!(roll.hue_family_residuals.is_some());
    assert_eq!(
        roll.delta_e00_summary
            .as_ref()
            .and_then(|value| value.get("max"))
            .and_then(serde_json::Value::as_f64),
        Some(4.4)
    );
}

#[test]
fn test_requested_film_stock_ranks_matching_roll_and_hint_evidence() {
    let tmp = tempfile::TempDir::new().unwrap();
    let scanners = tmp.path().join("scanners");
    let rolls = tmp.path().join("rolls");
    let hints = tmp.path().join("film_hints");
    std::fs::create_dir_all(&scanners).unwrap();
    std::fs::create_dir_all(&rolls).unwrap();
    std::fs::create_dir_all(&hints).unwrap();
    std::fs::write(
        rolls.join("roll.json"),
        roll_profile_json("roll-a", "scanner-a"),
    )
    .unwrap();
    std::fs::write(hints.join("hint.json"), film_hint_json()).unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        None,
        None,
        Some("Synthetic 200"),
        Some([11900.0, 7100.0, 3100.0]),
    );

    assert!(result.profile.is_none());
    assert_eq!(
        result.diagnostics.requested_film_stock.as_deref(),
        Some("Synthetic 200")
    );
    let film = result
        .diagnostics
        .film_stock
        .as_ref()
        .expect("film-stock diagnostics");
    assert_eq!(film.status, "matched");
    assert_eq!(film.matched_roll_profiles, vec!["roll-a".to_string()]);
    assert_eq!(
        result.diagnostics.nearest_film_candidates[0].film_stock_match,
        Some(true)
    );
    assert_eq!(
        result.diagnostics.nearest_roll_candidates[0].film_stock_match,
        Some(true)
    );
}

#[test]
fn test_requested_film_stock_rejects_explicit_mismatched_roll_profile() {
    let tmp = tempfile::TempDir::new().unwrap();
    let scanners = tmp.path().join("scanners");
    let rolls = tmp.path().join("rolls");
    std::fs::create_dir_all(&scanners).unwrap();
    std::fs::create_dir_all(&rolls).unwrap();
    std::fs::write(
        scanners.join("scanner.json"),
        scanner_profile_json("scanner-a", 0.96, false),
    )
    .unwrap();
    std::fs::write(
        rolls.join("roll.json"),
        roll_profile_json("roll-a", "scanner-a"),
    )
    .unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-a"),
        Some("roll-a"),
        Some("Different Film 400"),
        Some([11900.0, 7100.0, 3100.0]),
    );

    assert!(result.profile.is_none());
    assert_eq!(result.diagnostics.status, "rejected");
    let roll = result
        .diagnostics
        .roll_profile
        .as_ref()
        .expect("roll rejection diagnostics");
    assert_eq!(roll.status, "rejected");
    assert!(roll
        .rejection_details
        .iter()
        .any(|detail| detail.contains("did not match requested film stock")));
    assert_eq!(
        result
            .diagnostics
            .film_stock
            .as_ref()
            .map(|film| film.status.as_str()),
        Some("unmatched")
    );
}

#[test]
fn test_base_only_roll_profile_records_metadata_without_direct_correction() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    std::fs::create_dir_all(tmp.path().join("rolls")).unwrap();
    std::fs::write(
        tmp.path().join("scanners/scanner.json"),
        scanner_profile_json("scanner-a", 0.96, false),
    )
    .unwrap();
    std::fs::write(
        tmp.path().join("rolls/roll.json"),
        base_only_roll_profile_json("roll-base", "scanner-a"),
    )
    .unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-a"),
        Some("roll-base"),
        None,
        Some([12000.0, 7000.0, 3000.0]),
    );

    let profile = result.profile.expect("scanner profile should still apply");
    assert_eq!(
        profile.application_mode,
        scanstitch::color_calibration::CalibrationApplicationMode::ScannerConstrainedImageAdaptation
    );
    assert_eq!(
        result
            .diagnostics
            .roll_profile
            .as_ref()
            .map(|roll| roll.correction_applied),
        Some(false)
    );
}

#[test]
fn test_roll_profile_carries_validated_measured_negative_response_into_renderer_profile() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    std::fs::create_dir_all(tmp.path().join("rolls")).unwrap();
    std::fs::write(
        tmp.path().join("scanners/scanner.json"),
        scanner_profile_json("scanner-a", 0.96, false),
    )
    .unwrap();
    let mut roll: serde_json::Value =
        serde_json::from_str(&base_only_roll_profile_json("roll-response", "scanner-a")).unwrap();
    roll["scanner_settings_fingerprint"] =
        serde_json::json!(synthetic_scanner_settings_fingerprint());
    roll["negative_response"] = measured_negative_response_json();
    std::fs::write(tmp.path().join("rolls/roll.json"), roll.to_string()).unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-a"),
        Some("roll-response"),
        None,
        Some([12_000.0, 7_000.0, 3_000.0]),
    );

    assert_eq!(result.diagnostics.status, "applied");
    let response = result
        .profile
        .expect("scanner/roll profile")
        .negative_response
        .expect("validated negative response");
    assert_eq!(response.model_id, "synthetic-roll-response-v1");
    assert_eq!(response.validation.held_out_patch_count, 24);
    assert!(result.diagnostics.reason.contains("scanner"));
}

#[test]
fn test_roll_profile_rejects_negative_response_without_held_out_improvement() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    std::fs::create_dir_all(tmp.path().join("rolls")).unwrap();
    std::fs::write(
        tmp.path().join("scanners/scanner.json"),
        scanner_profile_json("scanner-a", 0.96, false),
    )
    .unwrap();
    let mut roll: serde_json::Value =
        serde_json::from_str(&base_only_roll_profile_json("roll-response", "scanner-a")).unwrap();
    roll["scanner_settings_fingerprint"] =
        serde_json::json!(synthetic_scanner_settings_fingerprint());
    let mut response = measured_negative_response_json();
    response["validation"]["delta_e00_rms"] = serde_json::json!(8.1);
    response["validation"]["unit_slope_delta_e00_rms"] = serde_json::json!(8.2);
    roll["negative_response"] = response;
    std::fs::write(tmp.path().join("rolls/roll.json"), roll.to_string()).unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-a"),
        Some("roll-response"),
        None,
        Some([12_000.0, 7_000.0, 3_000.0]),
    );

    assert!(result.profile.is_none());
    let invalid = &result
        .diagnostics
        .library
        .as_ref()
        .expect("library diagnostics")
        .invalid_entries;
    assert!(invalid.iter().any(|entry| entry
        .reasons
        .iter()
        .any(|reason| reason.contains("improve held-out DeltaE00 RMS"))));
}

#[test]
fn test_calibrate_cli_fits_measured_negative_response_and_scores_held_out_patches() {
    let tmp = tempfile::TempDir::new().unwrap();
    let library = tmp.path().join("calibration");
    std::fs::create_dir_all(library.join("scanners")).unwrap();
    std::fs::write(
        library.join("scanners/scanner.json"),
        scanner_profile_json("scanner-a", 0.96, false),
    )
    .unwrap();

    let density_to_layer =
        nalgebra::Matrix3::new(1.0, -0.08, 0.01, -0.04, 1.0, -0.05, 0.01, -0.04, 1.0);
    let layer_to_density = density_to_layer.try_inverse().unwrap();
    let build_patch = |index: usize, held_out: bool| {
        let offset = if held_out { 0.037 } else { 0.0 };
        let exposure = [
            0.10 + 0.80 * (((index * 7) % 31) as f64 / 30.0) + offset,
            0.12 + 0.74 * (((index * 11 + 3) % 31) as f64 / 30.0) + offset * 0.7,
            0.15 + 0.68 * (((index * 13 + 5) % 31) as f64 / 30.0) + offset * 0.4,
        ];
        let layer =
            nalgebra::Vector3::new(1.25 * exposure[0], 0.92 * exposure[1], 0.70 * exposure[2]);
        let scanner = layer_to_density * layer;
        if held_out {
            serde_json::json!({
                "scanner_density": [scanner[0], scanner[1], scanner[2]],
                "reference_log_exposure": exposure,
            })
        } else {
            serde_json::json!({
                "scanner_density": [scanner[0], scanner[1], scanner[2]],
                "reference_layer_density": [layer[0], layer[1], layer[2]],
                "reference_log_exposure": exposure,
            })
        }
    };
    let training = (0..31)
        .map(|index| build_patch(index, false))
        .collect::<Vec<_>>();
    let held_out = (0..24)
        .map(|index| build_patch(index + 31, true))
        .collect::<Vec<_>>();
    let measurements = serde_json::json!({
        "schema_version": 2,
        "film": { "stock": "Synthetic 200", "process": "C-41" },
        "base_color": [12000.0, 7000.0, 3000.0],
        "confidence": 0.92,
        "negative_response_fit": {
            "model_id": "fitted-synthetic-response",
            "scene_rgb_to_xyz_d50": [
                [0.7977, 0.1352, 0.0313],
                [0.2880, 0.7119, 0.0001],
                [0.0000, 0.0000, 0.8251]
            ],
            "training_patches": training,
            "held_out_patches": held_out,
            "confidence": 0.94,
            "white_anchor_percentile": 0.995
        }
    });
    let measurement_path = tmp.path().join("negative-response-measurements.json");
    std::fs::write(&measurement_path, measurements.to_string()).unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .arg("roll-target")
        .arg("--library")
        .arg(&library)
        .arg("--profile-id")
        .arg("roll-response")
        .arg("--scanner-profile")
        .arg("scanner-a")
        .arg("--measurements")
        .arg(&measurement_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "calibration failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(library.join("rolls/roll-response.json")).unwrap(),
    )
    .unwrap();
    assert!(record.get("negative_response_fit").is_none());
    assert_eq!(
        record["negative_response"]["model_id"],
        "fitted-synthetic-response"
    );
    assert_eq!(
        record["negative_response"]["validation"]["held_out_patch_count"],
        24
    );
    assert!(record["negative_response"]["validation"]["delta_e00_rms"]
        .as_f64()
        .is_some_and(|value| value < 1.0));
    assert!(
        record["negative_response"]["validation"]["unit_slope_delta_e00_rms"]
            .as_f64()
            .is_some_and(|value| value > 1.0)
    );
    assert!(record["negative_response"]["characteristic_curves"]["red"]
        .as_array()
        .is_some_and(|curve| curve.len() >= 4));
}

#[test]
fn test_calibrate_cli_fits_scanner_signal_linearization_with_held_out_error() {
    let tmp = tempfile::TempDir::new().unwrap();
    let library = tmp.path().join("calibration");
    let build_sample = |signal: f64, held_out: bool| {
        let x = if held_out { 0.4 } else { 0.0 };
        let gain = 1.0 + 0.02 * x;
        let reference = signal * signal * gain;
        serde_json::json!({
            "scanner_signal": [signal, signal, signal],
            "reference_transmittance": [reference, reference, reference],
            "x": x,
            "y": 0.0
        })
    };
    let training = (0..=30)
        .map(|index| build_sample(index as f64 / 30.0, false))
        .collect::<Vec<_>>();
    let held_out = (0..24)
        .map(|index| build_sample((index as f64 + 0.5) / 24.0, true))
        .collect::<Vec<_>>();
    let measurements = serde_json::json!({
        "schema_version": 2,
        "scanner": { "make": "Synthetic", "model": "Linearization Target" },
        "settings": { "dpi": 3200, "mode": "positive" },
        "target": { "type": "step_wedge", "illuminant": "D50" },
        "reference": { "dataset": "synthetic-transmittance" },
        "whitepoint": [0.9642, 1.0, 0.8251],
        "scanner_rgb_to_xyz": [
            [0.7977, 0.1352, 0.0313],
            [0.2880, 0.7119, 0.0001],
            [0.0, 0.0, 0.8251]
        ],
        "confidence": 0.95,
        "scanner_linearization_fit": {
            "model_id": "fitted-scanner-linearization",
            "training_samples": training,
            "held_out_samples": held_out,
            "black_level_normalized": [0.0, 0.0, 0.0],
            "white_level_normalized": [1.0, 1.0, 1.0],
            "additive_flare_normalized": [0.0, 0.0, 0.0],
            "shading_gain_polynomial": [
                [1.0, 0.02, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.02, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.02, 0.0, 0.0, 0.0, 0.0]
            ],
            "confidence": 0.95
        }
    });
    let path = tmp.path().join("scanner-linearization-measurements.json");
    std::fs::write(&path, measurements.to_string()).unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .arg("scanner-target")
        .arg("--library")
        .arg(&library)
        .arg("--profile-id")
        .arg("scanner-linear")
        .arg("--measurements")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "scanner calibration failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(library.join("scanners/scanner-linear.json")).unwrap(),
    )
    .unwrap();
    assert!(record.get("scanner_linearization_fit").is_none());
    assert_eq!(
        record["scanner_linearization"]["model_id"],
        "fitted-scanner-linearization"
    );
    assert!(
        record["scanner_linearization"]["validation"]["transmittance_rmse"]
            .as_f64()
            .is_some_and(|value| value < 0.01)
    );
    assert!(
        record["scanner_linearization"]["validation"]["identity_baseline_rmse"]
            .as_f64()
            .is_some_and(|value| value > 0.05)
    );
}

#[test]
fn test_roll_profile_rejects_mismatched_scanner_settings_fingerprint() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    std::fs::create_dir_all(tmp.path().join("rolls")).unwrap();
    std::fs::write(
        tmp.path().join("scanners/scanner.json"),
        scanner_profile_json("scanner-a", 0.96, false),
    )
    .unwrap();
    let mut roll: serde_json::Value =
        serde_json::from_str(&roll_profile_json("roll-a", "scanner-a")).unwrap();
    roll["scanner_settings_fingerprint"] = serde_json::json!("fnv1a64:0000000000000000");
    std::fs::write(tmp.path().join("rolls/roll.json"), roll.to_string()).unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-a"),
        Some("roll-a"),
        None,
        None,
    );

    assert!(result.profile.is_none());
    assert_eq!(result.diagnostics.status, "rejected");
    assert!(result
        .diagnostics
        .rejection_details
        .iter()
        .any(|detail| detail.contains("scanner_settings_fingerprint")));
}

#[test]
fn test_roll_profile_rejects_correction_without_scanner_profile_id() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    std::fs::create_dir_all(tmp.path().join("rolls")).unwrap();
    std::fs::write(
        tmp.path().join("scanners/scanner.json"),
        scanner_profile_json("scanner-a", 0.96, false),
    )
    .unwrap();
    let mut roll: serde_json::Value =
        serde_json::from_str(&roll_profile_json("roll-a", "scanner-a")).unwrap();
    roll.as_object_mut().unwrap().remove("scanner_profile_id");
    std::fs::write(tmp.path().join("rolls/roll.json"), roll.to_string()).unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-a"),
        Some("roll-a"),
        None,
        None,
    );

    assert!(result.profile.is_none());
    assert_eq!(result.diagnostics.status, "rejected");
    assert!(result
        .diagnostics
        .rejection_details
        .iter()
        .any(|detail| detail.contains("scanner_profile_id")));
}

#[test]
fn test_roll_profile_rejects_correction_without_settings_fingerprint() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    std::fs::create_dir_all(tmp.path().join("rolls")).unwrap();
    std::fs::write(
        tmp.path().join("scanners/scanner.json"),
        scanner_profile_json("scanner-a", 0.96, false),
    )
    .unwrap();
    let mut roll: serde_json::Value =
        serde_json::from_str(&roll_profile_json("roll-a", "scanner-a")).unwrap();
    roll.as_object_mut()
        .unwrap()
        .remove("scanner_settings_fingerprint");
    std::fs::write(tmp.path().join("rolls/roll.json"), roll.to_string()).unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-a"),
        Some("roll-a"),
        None,
        None,
    );

    assert!(result.profile.is_none());
    assert_eq!(result.diagnostics.status, "rejected");
    assert!(result
        .diagnostics
        .rejection_details
        .iter()
        .any(|detail| detail.contains("scanner_settings_fingerprint")));
}

#[test]
fn test_weak_scanner_auto_match_is_advisory_not_applied() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    std::fs::write(
        tmp.path().join("scanners/scanner.json"),
        scanner_profile_json("scanner-weak", 0.70, true),
    )
    .unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        None,
        None,
        None,
        None,
    );

    assert!(result.profile.is_none());
    assert_eq!(result.diagnostics.status, "not_configured");
    let scanner = result
        .diagnostics
        .scanner_profile
        .expect("advisory scanner diagnostics");
    assert_eq!(scanner.status, "advisory");
    assert!(scanner.reason.contains("advisory only"));
}

#[test]
fn test_ambiguous_scanner_auto_matches_are_advisory_not_applied() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    std::fs::write(
        tmp.path().join("scanners/scanner-a.json"),
        scanner_profile_json("scanner-a", 0.96, true),
    )
    .unwrap();
    std::fs::write(
        tmp.path().join("scanners/scanner-b.json"),
        scanner_profile_json("scanner-b", 0.97, true),
    )
    .unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        None,
        None,
        None,
        None,
    );

    assert!(result.profile.is_none());
    assert_eq!(result.diagnostics.status, "not_configured");
    let scanner = result
        .diagnostics
        .scanner_profile
        .expect("advisory scanner diagnostics");
    assert_eq!(scanner.status, "advisory");
    assert!(scanner
        .reason
        .contains("multiple high-confidence scanner profiles"));
}

#[test]
fn test_advisory_library_can_fall_back_to_compatibility_profile() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    std::fs::write(
        tmp.path().join("scanners/scanner.json"),
        scanner_profile_json("scanner-weak", 0.70, true),
    )
    .unwrap();
    let profile_path = tmp.path().join("one-off.json");
    std::fs::write(&profile_path, valid_profile_json()).unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        Some(&profile_path),
        Some(tmp.path()),
        None,
        None,
        None,
        None,
    );

    assert!(result.profile.is_some());
    assert_eq!(result.diagnostics.status, "applied");
    assert_eq!(result.diagnostics.source, "external_calibration_profile");
    assert_eq!(
        result
            .diagnostics
            .scanner_profile
            .as_ref()
            .map(|scanner| scanner.status.as_str()),
        Some("advisory")
    );
}

#[test]
fn test_roll_profile_rejects_mismatched_scanner_profile() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    std::fs::create_dir_all(tmp.path().join("rolls")).unwrap();
    std::fs::write(
        tmp.path().join("scanners/scanner.json"),
        scanner_profile_json("scanner-a", 0.96, false),
    )
    .unwrap();
    std::fs::write(
        tmp.path().join("rolls/roll.json"),
        roll_profile_json("roll-a", "scanner-b"),
    )
    .unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-a"),
        Some("roll-a"),
        None,
        None,
    );

    assert!(result.profile.is_none());
    assert_eq!(result.diagnostics.status, "rejected");
    assert!(result
        .diagnostics
        .rejection_details
        .iter()
        .any(|detail| detail.contains("expects scanner_profile_id")));
}

#[test]
fn test_invalid_library_entries_are_reported_without_blocking_valid_selection() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("scanners")).unwrap();
    std::fs::write(
        tmp.path().join("scanners/good.json"),
        scanner_profile_json("scanner-a", 0.96, false),
    )
    .unwrap();
    let mut invalid: serde_json::Value =
        serde_json::from_str(&scanner_profile_json("scanner-bad", 0.96, false)).unwrap();
    invalid["scanner_rgb_to_xyz"] =
        serde_json::json!([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    std::fs::write(tmp.path().join("scanners/bad.json"), invalid.to_string()).unwrap();

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("scanner-a"),
        None,
        None,
        None,
    );

    assert!(result.profile.is_some());
    let library = result.diagnostics.library.expect("library diagnostics");
    assert_eq!(library.scanner_profiles_loaded, 1);
    assert_eq!(library.invalid_entries.len(), 1);
    assert!(library.invalid_entries[0]
        .reasons
        .iter()
        .any(|reason| reason.contains("singular")));
}

#[test]
fn test_calibrate_cli_rejects_in_sample_and_failed_held_out_target_fits() {
    let tmp = tempfile::TempDir::new().unwrap();
    let library = tmp.path().join("calibration");

    let mut legacy: serde_json::Value =
        serde_json::from_str(&scanner_fit_measurement_json()).unwrap();
    let training = legacy
        .as_object_mut()
        .unwrap()
        .remove("training_patches")
        .unwrap();
    legacy.as_object_mut().unwrap().remove("held_out_patches");
    legacy["patches"] = training;
    let legacy_path = tmp.path().join("legacy-in-sample.json");
    std::fs::write(&legacy_path, legacy.to_string()).unwrap();
    let legacy_output = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "scanner-target",
            "--library",
            library.to_str().unwrap(),
            "--profile-id",
            "legacy-in-sample",
            "--measurements",
            legacy_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!legacy_output.status.success());
    assert!(String::from_utf8_lossy(&legacy_output.stderr)
        .contains("in-sample residuals are not independent"));
    assert!(!library.join("scanners/legacy-in-sample.json").exists());

    let mut adversarial: serde_json::Value =
        serde_json::from_str(&scanner_fit_measurement_json()).unwrap();
    for patch in adversarial["held_out_patches"].as_array_mut().unwrap() {
        let source = patch["source_rgb"].as_array().unwrap();
        let wrong_xyz = source
            .iter()
            .map(|channel| channel.as_f64().unwrap() * 0.05)
            .collect::<Vec<_>>();
        patch["xyz"] = serde_json::json!(wrong_xyz);
    }
    let adversarial_path = tmp.path().join("failed-held-out.json");
    std::fs::write(&adversarial_path, adversarial.to_string()).unwrap();
    let adversarial_output = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "scanner-target",
            "--library",
            library.to_str().unwrap(),
            "--profile-id",
            "failed-held-out",
            "--measurements",
            adversarial_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!adversarial_output.status.success());
    assert!(
        String::from_utf8_lossy(&adversarial_output.stderr).contains("held-out target confidence")
    );
    assert!(!library.join("scanners/failed-held-out.json").exists());
}

#[test]
fn test_calibrate_cli_fits_scanner_and_roll_matrices_in_runtime_linearized_domain() {
    let tmp = tempfile::TempDir::new().unwrap();
    let library = tmp.path().join("calibration");
    let scanner_matrix = [
        [0.7977, 0.1352, 0.0313],
        [0.2880, 0.7119, 0.0001],
        [0.0000, 0.0000, 0.8251],
    ];
    let linearization_value = nonlinear_center_scanner_linearization_json();
    let linearization = serde_json::from_value::<
        scanstitch::scanner_linearization::ScannerLinearizationCalibration,
    >(linearization_value.clone())
    .unwrap();
    let linearize = |signal| {
        scanstitch::scanner_linearization::evaluate_scanner_signal(&linearization, signal, 0.0, 0.0)
            .unwrap()
    };
    let (scanner_training, scanner_held_out) =
        disjoint_fit_patch_sets("linearized-scanner", |raw| {
            apply_rows(scanner_matrix, linearize(raw))
        });
    let scanner_measurements = serde_json::json!({
        "source_space": { "name": "synthetic scanner RGB", "encoding": "scanner_signal" },
        "scanner": { "make": "Synthetic", "model": "Nonlinear Scanner" },
        "settings": { "dpi": 3200, "software": "unit-test", "mode": "raw" },
        "target": { "type": "synthetic transmissive target", "illuminant": "D50" },
        "reference": { "dataset": "synthetic XYZ" },
        "scanner_linearization": linearization_value,
        "training_patches": patch_measurements_json(&scanner_training),
        "held_out_patches": patch_measurements_json(&scanner_held_out)
    });
    let scanner_path = tmp.path().join("linearized-scanner.json");
    std::fs::write(&scanner_path, scanner_measurements.to_string()).unwrap();
    let scanner_output = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "scanner-target",
            "--library",
            library.to_str().unwrap(),
            "--profile-id",
            "linearized-scanner",
            "--measurements",
            scanner_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        scanner_output.status.success(),
        "scanner calibration failed: {}",
        String::from_utf8_lossy(&scanner_output.stderr)
    );
    let scanner_record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(library.join("scanners/linearized-scanner.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        scanner_record["target_patch_signal_domain"],
        "scanner_linearized_transmittance"
    );
    assert_eq!(
        scanner_record["target_patch_linearization_model_id"],
        "synthetic-center-nonlinear-v1"
    );
    assert_eq!(
        scanner_record["target_patch_linearization_application"]["status"],
        "applied_before_matrix_fit"
    );
    for (row, expected_row) in scanner_matrix.iter().enumerate() {
        for (column, expected_value) in expected_row.iter().enumerate() {
            let actual = scanner_record["scanner_rgb_to_xyz"][row][column]
                .as_f64()
                .unwrap();
            assert!((actual - expected_value).abs() < 1e-9);
        }
    }
    let retained_source = scanner_record["patches"][0]["source_rgb"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect::<Vec<_>>();
    let expected_retained = linearize(scanner_held_out[0].source_rgb);
    for channel in 0..3 {
        assert!((retained_source[channel] - expected_retained[channel]).abs() < 1e-12);
    }

    let correction = [[1.05, 0.02, 0.0], [0.01, 0.96, 0.0], [0.0, 0.03, 1.02]];
    let (roll_training, roll_held_out) = disjoint_fit_patch_sets("linearized-roll", |raw| {
        apply_rows(correction, apply_rows(scanner_matrix, linearize(raw)))
    });
    let roll_measurements = serde_json::json!({
        "film": { "stock": "Synthetic 200", "process": "C-41" },
        "target": { "type": "synthetic roll target" },
        "reference": { "dataset": "synthetic roll XYZ" },
        "base_color": [12000.0, 7000.0, 3000.0],
        "training_patches": patch_measurements_json(&roll_training),
        "held_out_patches": patch_measurements_json(&roll_held_out)
    });
    let roll_path = tmp.path().join("linearized-roll.json");
    std::fs::write(&roll_path, roll_measurements.to_string()).unwrap();
    let roll_output = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "roll-target",
            "--library",
            library.to_str().unwrap(),
            "--profile-id",
            "linearized-roll",
            "--scanner-profile",
            "linearized-scanner",
            "--measurements",
            roll_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        roll_output.status.success(),
        "roll calibration failed: {}",
        String::from_utf8_lossy(&roll_output.stderr)
    );
    let roll_record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(library.join("rolls/linearized-roll.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        roll_record["target_patch_signal_domain"],
        "scanner_linearized_transmittance"
    );
    for (row, expected_row) in correction.iter().enumerate() {
        for (column, expected_value) in expected_row.iter().enumerate() {
            let actual = roll_record["correction_matrix"][row][column]
                .as_f64()
                .unwrap();
            assert!((actual - expected_value).abs() < 1e-9);
        }
    }
    let loaded = scanstitch::color_calibration::load_calibration(
        None,
        Some(&library),
        Some("linearized-scanner"),
        Some("linearized-roll"),
        None,
        Some([12000.0, 7000.0, 3000.0]),
    );
    assert!(
        loaded.profile.is_some(),
        "CLI-generated runtime-domain records must load together: {:?}",
        loaded.diagnostics
    );
    assert_eq!(
        loaded
            .diagnostics
            .roll_profile
            .as_ref()
            .and_then(|roll| roll.target_patch_linearization_model_id.as_deref()),
        Some("synthetic-center-nonlinear-v1")
    );
}

#[test]
fn test_calibrate_cli_requires_patch_coordinates_for_spatial_scanner_linearization() {
    let tmp = tempfile::TempDir::new().unwrap();
    let library = tmp.path().join("calibration");
    let mut measurements: serde_json::Value =
        serde_json::from_str(&scanner_fit_measurement_json()).unwrap();
    measurements["scanner_linearization"] = scanner_linearization_json();
    let path = tmp.path().join("missing-spatial-coordinates.json");
    std::fs::write(&path, measurements.to_string()).unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "scanner-target",
            "--library",
            library.to_str().unwrap(),
            "--profile-id",
            "missing-spatial-coordinates",
            "--measurements",
            path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires scanner_xy"));
    assert!(!library
        .join("scanners/missing-spatial-coordinates.json")
        .exists());
}

#[test]
fn test_calibrate_cli_fits_scanner_and_roll_target_records() {
    let tmp = tempfile::TempDir::new().unwrap();
    let library = tmp.path().join("calibration");
    let scanner_measurements = tmp.path().join("scanner-measurements.json");
    std::fs::write(&scanner_measurements, scanner_fit_measurement_json()).unwrap();

    let scanner_status = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "scanner-target",
            "--library",
            library.to_str().unwrap(),
            "--profile-id",
            "scanner-fit",
            "--measurements",
            scanner_measurements.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(scanner_status.success());
    let scanner_record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(library.join("scanners/scanner-fit.json")).unwrap(),
    )
    .unwrap();
    assert!(scanner_record.get("training_patches").is_none());
    assert!(scanner_record.get("held_out_patches").is_none());
    assert_eq!(scanner_record["patches"].as_array().map(Vec::len), Some(12));
    assert_eq!(
        scanner_record["fit"]["validation"]["evaluation_set"],
        "held_out"
    );

    let duplicate_scanner_output =
        std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
            .args([
                "scanner-target",
                "--library",
                library.to_str().unwrap(),
                "--profile-id",
                "scanner-fit",
                "--measurements",
                scanner_measurements.to_str().unwrap(),
            ])
            .output()
            .unwrap();
    assert!(
        !duplicate_scanner_output.status.success(),
        "duplicate scanner calibration ingest should require --force"
    );
    let duplicate_stderr = String::from_utf8_lossy(&duplicate_scanner_output.stderr);
    assert!(duplicate_stderr.contains("already exists"));
    assert!(duplicate_stderr.contains("--force"));

    let forced_scanner_status =
        std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
            .args([
                "scanner-target",
                "--library",
                library.to_str().unwrap(),
                "--profile-id",
                "scanner-fit",
                "--measurements",
                scanner_measurements.to_str().unwrap(),
                "--force",
            ])
            .status()
            .unwrap();
    assert!(
        forced_scanner_status.success(),
        "explicit --force should replace an existing calibration record"
    );

    let invalid_scanner_measurements = tmp.path().join("scanner-invalid-measurements.json");
    std::fs::write(
        &invalid_scanner_measurements,
        serde_json::json!({
            "scanner": { "make": "Synthetic", "model": "Invalid Prefit Scanner" },
            "target": { "type": "synthetic_transmissive_target" },
            "reference": { "dataset": "synthetic" },
            "scanner_rgb_to_xyz": [
                [0.80, 0.10, 0.00],
                [0.20, 0.70, 0.10],
                [0.00, 0.20, 0.80]
            ],
            "confidence": 0.95
        })
        .to_string(),
    )
    .unwrap();
    let invalid_scanner_output =
        std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
            .args([
                "scanner-target",
                "--library",
                library.to_str().unwrap(),
                "--profile-id",
                "scanner-invalid",
                "--measurements",
                invalid_scanner_measurements.to_str().unwrap(),
            ])
            .output()
            .unwrap();
    assert!(
        !invalid_scanner_output.status.success(),
        "calibration ingest should reject records that the library loader would reject"
    );
    let invalid_stderr = String::from_utf8_lossy(&invalid_scanner_output.stderr);
    assert!(invalid_stderr.contains("did not pass calibration library validation"));
    assert!(invalid_stderr.contains("missing whitepoint"));
    assert!(
        !library
            .join("scanners")
            .join("scanner-invalid.json")
            .exists(),
        "invalid calibration records should not be written"
    );

    let scanner_result = scanstitch::color_calibration::load_calibration(
        None,
        Some(&library),
        Some("scanner-fit"),
        None,
        None,
        None,
    );
    let scanner_profile = scanner_result
        .profile
        .expect("fitted scanner profile should load");
    assert_eq!(
        scanner_result
            .diagnostics
            .scanner_profile
            .as_ref()
            .and_then(|scanner| scanner.fit.as_ref())
            .map(|fit| fit.patch_count),
        Some(12)
    );
    assert!(
        scanner_result
            .diagnostics
            .scanner_profile
            .as_ref()
            .and_then(|scanner| scanner.fit.as_ref())
            .is_some_and(|fit| !fit.per_hue_residuals.is_empty() && !fit.worst_patches.is_empty()),
        "fitted scanner profile should retain residual summaries"
    );
    assert!(scanner_result
        .diagnostics
        .scanner_profile
        .as_ref()
        .and_then(|scanner| scanner.fit.as_ref())
        .and_then(|fit| fit.validation.as_ref())
        .is_some_and(|validation| {
            validation.training_patch_count == 12 && validation.held_out_patch_count == 12
        }));
    let scanner_fingerprint = scanner_result
        .diagnostics
        .scanner_profile
        .as_ref()
        .and_then(|scanner| scanner.scanner_settings_fingerprint.as_ref())
        .cloned()
        .expect("fitted scanner profile should report settings fingerprint");

    let prefit_roll_measurements = tmp.path().join("roll-prefit-measurements.json");
    let expected_correction = [[1.05, 0.02, 0.0], [0.01, 0.96, 0.0], [0.0, 0.03, 1.02]];
    std::fs::write(
        &prefit_roll_measurements,
        serde_json::json!({
            "film": { "stock": "Synthetic 200", "process": "C-41" },
            "correction_matrix": expected_correction,
            "confidence": 0.82
        })
        .to_string(),
    )
    .unwrap();
    let prefit_roll_status = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "roll-target",
            "--library",
            library.to_str().unwrap(),
            "--profile-id",
            "roll-prefit",
            "--scanner-profile",
            "scanner-fit",
            "--measurements",
            prefit_roll_measurements.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(
        prefit_roll_status.success(),
        "pre-fit roll correction should be stamped with selected scanner context"
    );
    let prefit_roll_result = scanstitch::color_calibration::load_calibration(
        None,
        Some(&library),
        Some("scanner-fit"),
        Some("roll-prefit"),
        None,
        None,
    );
    assert!(prefit_roll_result
        .profile
        .as_ref()
        .is_some_and(|profile| profile.roll_correction_applied));
    let prefit_roll = prefit_roll_result
        .diagnostics
        .roll_profile
        .as_ref()
        .expect("pre-fit roll diagnostics");
    assert_eq!(
        prefit_roll.scanner_settings_fingerprint.as_deref(),
        Some(scanner_fingerprint.as_str())
    );
    assert_eq!(
        prefit_roll.correction_domain.as_deref(),
        Some("xyz_post_scanner")
    );

    let roll_measurements = tmp.path().join("roll-measurements.json");
    std::fs::write(&roll_measurements, roll_fit_measurement_json()).unwrap();
    let roll_status = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "roll-target",
            "--library",
            library.to_str().unwrap(),
            "--profile-id",
            "roll-fit",
            "--scanner-profile",
            "scanner-fit",
            "--measurements",
            roll_measurements.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(roll_status.success());
    let roll_record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(library.join("rolls/roll-fit.json")).unwrap(),
    )
    .unwrap();
    assert!(roll_record.get("training_patches").is_none());
    assert!(roll_record.get("held_out_patches").is_none());
    assert_eq!(roll_record["patches"].as_array().map(Vec::len), Some(12));
    assert_eq!(roll_record["fit"]["validation"]["held_out_patch_count"], 12);
    assert_eq!(
        roll_record["scanner_color_model_application"]["transform"],
        "matrix"
    );
    assert!(roll_record["scanner_color_model_application"]
        .get("model_id")
        .is_none());
    assert_eq!(
        roll_record["scanner_color_model_application"]["output_domain"],
        "reference_xyz_d50"
    );
    assert_eq!(
        roll_record["scanner_color_model_application"]["numerical_zero_tolerance"],
        serde_json::json!(1e-12)
    );
    assert!(
        roll_record["scanner_color_model_application"]["numerical_zero_clamp_count"]
            .as_u64()
            .is_some_and(|count| count >= 1)
    );

    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(&library),
        Some("scanner-fit"),
        Some("roll-fit"),
        None,
        None,
    );
    let profile = result
        .profile
        .expect("fitted scanner+roll profile should load");
    assert_eq!(
        profile.application_mode,
        scanstitch::color_calibration::CalibrationApplicationMode::DirectProfile
    );
    assert!(profile.roll_correction_applied);
    assert_eq!(
        result
            .diagnostics
            .roll_profile
            .as_ref()
            .and_then(|roll| roll.fit.as_ref())
            .map(|fit| fit.method.as_str()),
        Some("least_squares_xyz_post_scanner_correction")
    );
    assert_eq!(
        result
            .diagnostics
            .roll_profile
            .as_ref()
            .and_then(|roll| roll.scanner_color_model_application.as_ref())
            .map(|application| application.transform.as_str()),
        Some("matrix")
    );
    let expected = [
        apply_rows(
            expected_correction,
            [
                scanner_profile.work_to_xyz[0][0],
                scanner_profile.work_to_xyz[1][0],
                scanner_profile.work_to_xyz[2][0],
            ],
        ),
        apply_rows(
            expected_correction,
            [
                scanner_profile.work_to_xyz[0][1],
                scanner_profile.work_to_xyz[1][1],
                scanner_profile.work_to_xyz[2][1],
            ],
        ),
        apply_rows(
            expected_correction,
            [
                scanner_profile.work_to_xyz[0][2],
                scanner_profile.work_to_xyz[1][2],
                scanner_profile.work_to_xyz[2][2],
            ],
        ),
    ];
    let expected = [
        [expected[0][0], expected[1][0], expected[2][0]],
        [expected[0][1], expected[1][1], expected[2][1]],
        [expected[0][2], expected[1][2], expected[2][2]],
    ];
    for (row, expected_row) in expected.iter().enumerate() {
        for (col, expected_value) in expected_row.iter().enumerate() {
            assert!(
                (profile.work_to_xyz[row][col] - *expected_value).abs() < 1e-8,
                "composed matrix mismatch at [{row}][{col}]"
            );
        }
    }
}

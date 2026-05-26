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

fn synthetic_scanner_settings_fingerprint() -> String {
    let settings = serde_json::json!({ "dpi": 3200, "mode": "positive" });
    scanstitch::color_calibration::scanner_settings_fingerprint(Some(&settings))
        .expect("synthetic scanner settings fingerprint")
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
            source_rgb,
            reference_xyz: apply_rows(matrix, source_rgb),
        },
    )
    .collect()
}

fn scanner_fit_measurement_json() -> String {
    let patches = scanner_fit_patches()
        .into_iter()
        .map(|patch| {
            serde_json::json!({
                "id": patch.patch_id,
                "rgb": patch.source_rgb,
                "xyz": patch.reference_xyz,
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "source_space": { "name": "synthetic scanner RGB", "encoding": "linear" },
        "scanner": { "make": "Synthetic", "model": "Fitted Scanner" },
        "settings": { "dpi": 3200, "software": "unit-test", "mode": "raw" },
        "target": { "type": "synthetic transmissive target", "illuminant": "D50" },
        "reference": { "dataset": "synthetic XYZ" },
        "patches": patches
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
    let sources = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.35, 0.80, 0.25],
        [0.70, 0.18, 0.62],
    ];
    let patches = sources
        .into_iter()
        .enumerate()
        .map(|(idx, source_rgb)| {
            let scanner_xyz = apply_rows(scanner_matrix, source_rgb);
            serde_json::json!({
                "id": format!("roll-patch-{idx}"),
                "scanner_rgb": source_rgb,
                "xyz": apply_rows(roll_correction, scanner_xyz),
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "film": { "stock": "Synthetic 200", "process": "C-41" },
        "target": { "type": "synthetic roll target" },
        "reference": { "dataset": "synthetic roll XYZ" },
        "base_color": [12000.0, 7000.0, 3000.0],
        "patches": patches
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
fn test_fit_scanner_matrix_from_synthetic_target_patches() {
    let fit = scanstitch::color_calibration::fit_rgb_to_xyz_from_patches(
        &scanner_fit_patches(),
        "least_squares_rgb_to_xyz",
        None,
        None,
    )
    .expect("fit should recover synthetic scanner matrix");
    let expected = [
        [0.7977, 0.1352, 0.0313],
        [0.2880, 0.7119, 0.0001],
        [0.0000, 0.0000, 0.8251],
    ];
    for (row, expected_row) in expected.iter().enumerate() {
        for (col, expected_value) in expected_row.iter().enumerate() {
            assert!(
                (fit.matrix[row][col] - *expected_value).abs() < 1e-10,
                "matrix mismatch at [{row}][{col}]"
            );
        }
    }
    assert_eq!(fit.fit.patch_count, 6);
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
fn test_target_fit_rejects_too_few_and_singular_patches() {
    let patches = scanner_fit_patches();
    let too_few = scanstitch::color_calibration::fit_rgb_to_xyz_from_patches(
        &patches[..3],
        "least_squares_rgb_to_xyz",
        None,
        None,
    )
    .unwrap_err();
    assert!(too_few.contains("at least"));

    let singular_source = vec![
        scanstitch::color_calibration::TargetPatch {
            patch_id: None,
            source_rgb: [1.0, 1.0, 1.0],
            reference_xyz: [0.9, 1.0, 0.8],
        };
        4
    ];
    let singular = scanstitch::color_calibration::fit_rgb_to_xyz_from_patches(
        &singular_source,
        "least_squares_rgb_to_xyz",
        None,
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
        Some(6)
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

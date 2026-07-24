use nalgebra::{Matrix3, Vector3};
use ndarray::Array3;

const PATCH_COLORS: [[f64; 3]; 12] = [
    [0.12, 0.18, 0.24],
    [0.22, 0.15, 0.40],
    [0.35, 0.28, 0.12],
    [0.48, 0.18, 0.22],
    [0.15, 0.42, 0.20],
    [0.28, 0.52, 0.16],
    [0.58, 0.38, 0.12],
    [0.68, 0.24, 0.40],
    [0.20, 0.28, 0.62],
    [0.38, 0.18, 0.70],
    [0.55, 0.60, 0.25],
    [0.72, 0.52, 0.58],
];

const REFERENCE_MATRIX: [[f64; 3]; 3] =
    [[0.75, 0.12, 0.03], [0.20, 0.68, 0.08], [0.02, 0.15, 0.72]];

fn matrix(rows: [[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::new(
        rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0],
        rows[2][1], rows[2][2],
    )
}

fn apply(rows: [[f64; 3]; 3], value: [f64; 3]) -> [f64; 3] {
    let mapped = matrix(rows) * Vector3::new(value[0], value[1], value[2]);
    [mapped[0], mapped[1], mapped[2]]
}

fn square_to_quad(points: [[f64; 2]; 4]) -> Matrix3<f64> {
    let [p0, p1, p2, p3] = points;
    let dx1 = p1[0] - p2[0];
    let dx2 = p3[0] - p2[0];
    let dx3 = p0[0] - p1[0] + p2[0] - p3[0];
    let dy1 = p1[1] - p2[1];
    let dy2 = p3[1] - p2[1];
    let dy3 = p0[1] - p1[1] + p2[1] - p3[1];
    let denominator = dx1 * dy2 - dx2 * dy1;
    let (g, h) = if dx3.abs() < 1e-12 && dy3.abs() < 1e-12 {
        (0.0, 0.0)
    } else {
        (
            (dx3 * dy2 - dx2 * dy3) / denominator,
            (dx1 * dy3 - dx3 * dy1) / denominator,
        )
    };
    Matrix3::new(
        p1[0] - p0[0] + g * p1[0],
        p3[0] - p0[0] + h * p3[0],
        p0[0],
        p1[1] - p0[1] + g * p1[1],
        p3[1] - p0[1] + h * p3[1],
        p0[1],
        g,
        h,
        1.0,
    )
}

fn render_chart(
    path: &std::path::Path,
    corners: [[f64; 2]; 4],
    channel_scale: [f64; 3],
    contaminate_patch: Option<usize>,
) {
    render_chart_with_code_max(path, corners, channel_scale, contaminate_patch, u16::MAX);
}

fn render_chart_with_code_max(
    path: &std::path::Path,
    corners: [[f64; 2]; 4],
    channel_scale: [f64; 3],
    contaminate_patch: Option<usize>,
    code_max: u16,
) {
    let width = 600usize;
    let height = 420usize;
    let inverse = square_to_quad(corners).try_inverse().unwrap();
    let mut image =
        Array3::<u16>::from_elem((height, width, 3), (0.023 * code_max as f64).round() as u16);
    for y in 0..height {
        for x in 0..width {
            let mapped = inverse * Vector3::new(x as f64, y as f64, 1.0);
            let u = mapped[0] / mapped[2];
            let v = mapped[1] / mapped[2];
            if (0.0..1.0).contains(&u) && (0.0..1.0).contains(&v) {
                let column = (u * 4.0).floor() as usize;
                let row = (v * 3.0).floor() as usize;
                let patch = row * 4 + column;
                let mut rgb = std::array::from_fn(|channel| {
                    PATCH_COLORS[patch][channel] * channel_scale[channel]
                });
                let variation = ((x * 17 + y * 13) % 7) as f64 * 0.000_01 - 0.000_03;
                for value in &mut rgb {
                    *value += variation;
                }
                if contaminate_patch == Some(patch)
                    && u.fract() > 0.0
                    && (u * 4.0).fract() > 0.30
                    && (u * 4.0).fract() < 0.70
                    && (v * 3.0).fract() > 0.30
                    && (v * 3.0).fract() < 0.70
                {
                    rgb = [0.90, 0.08, 0.85];
                }
                for channel in 0..3 {
                    image[[y, x, channel]] =
                        (rgb[channel].clamp(0.0, 1.0) * code_max as f64).round() as u16;
                }
            }
        }
    }
    // A small bright dust spot exercises robust outlier rejection without invalidating the patch.
    for y in 205..210 {
        for x in 295..300 {
            for channel in 0..3 {
                image[[y, x, channel]] = (0.95 * code_max as f64).round() as u16;
            }
        }
    }
    scanstitch::tiff_io::save_tiff_u16(&image, path).unwrap();
}

fn corners_json(points: [[f64; 2]; 4]) -> serde_json::Value {
    serde_json::json!({
        "top_left": points[0],
        "top_right": points[1],
        "bottom_right": points[2],
        "bottom_left": points[3]
    })
}

fn set_corner_review_approvals(
    manifest: &mut serde_json::Value,
    manifest_directory: &std::path::Path,
    working_bit_depth: u8,
) {
    manifest["scanner_signal_bit_depth"] = serde_json::json!(working_bit_depth);
    let parsed: scanstitch::calibration_target::TargetSamplingManifest =
        serde_json::from_value(manifest.clone()).unwrap();
    let result =
        scanstitch::calibration_target::sample_target_manifest(&parsed, manifest_directory)
            .expect("synthetic target manifest should produce a review draft");
    *manifest = serde_json::to_value(result.corner_review_manifest).unwrap();
    for capture in manifest["captures"].as_array_mut().unwrap() {
        capture["corner_review"]["approved"] = serde_json::json!(true);
    }
}

fn sampling_manifest(
    training_image: &str,
    held_out_image: &str,
    training_corners: [[f64; 2]; 4],
    held_out_corners: [[f64; 2]; 4],
) -> serde_json::Value {
    let patches = PATCH_COLORS
        .iter()
        .enumerate()
        .map(|(index, rgb)| {
            serde_json::json!({
                "id": format!("P{:02}", index + 1),
                "row": index / 4,
                "column": index % 4,
                "xyz": apply(REFERENCE_MATRIX, *rgb)
            })
        })
        .collect::<Vec<_>>();
    let mut manifest = serde_json::json!({
        "schema_version": 1,
        "scanner_signal_bit_depth": 16,
        "measurement": {
            "source_space": {
                "name": "synthetic normalized scanner signal",
                "encoding": "scanner_signal"
            },
            "scanner": { "make": "Synthetic", "model": "Projective Target Test" },
            "settings": { "dpi": 3200, "mode": "raw" },
            "target": { "type": "synthetic transmissive chart", "illuminant": "D50" },
            "reference": { "dataset": "synthetic XYZ D50" },
            "whitepoint": [0.9642, 1.0, 0.8251]
        },
        "grid": {
            "rows": 3,
            "columns": 4,
            "patch_inset_fraction": 0.20,
            "samples_per_axis": 25
        },
        "patches": patches,
        "captures": [
            {
                "id": "capture-training",
                "role": "training",
                "image": training_image,
                "corners": corners_json(training_corners)
            },
            {
                "id": "capture-held-out",
                "role": "held_out",
                "image": held_out_image,
                "corners": corners_json(held_out_corners)
            }
        ],
        "quality": {
            "minimum_source_bits_per_sample": 16,
            "minimum_retained_fraction": 0.70,
            "maximum_clipped_fraction": 0.01,
            "maximum_channel_robust_spread": 0.03,
            "maximum_spatial_cell_delta": 0.02,
            "outlier_z_limit": 6.0,
            "clipping_margin": 0.0001
        },
        "preview_max_dimension": 800
    });
    let reference_patches: Vec<scanstitch::calibration_target::TargetReferencePatch> =
        serde_json::from_value(manifest["patches"].clone()).unwrap();
    let reference_patch_sha256 =
        scanstitch::calibration_target::sha256_reference_patches(&reference_patches);
    manifest["reference_review"] = serde_json::json!({
        "approved": true,
        "reference_patch_sha256": reference_patch_sha256
    });
    let round_tripped: scanstitch::calibration_target::TargetSamplingManifest =
        serde_json::from_str(&serde_json::to_string(&manifest).unwrap()).unwrap();
    assert_eq!(
        round_tripped
            .reference_review
            .as_ref()
            .unwrap()
            .reference_patch_sha256,
        scanstitch::calibration_target::sha256_reference_patches(&round_tripped.patches),
        "reference review hashes must survive manifest JSON round trips"
    );
    manifest
}

fn approved_sampling_manifest(
    manifest_directory: &std::path::Path,
    training_image: &str,
    held_out_image: &str,
    training_corners: [[f64; 2]; 4],
    held_out_corners: [[f64; 2]; 4],
) -> serde_json::Value {
    let mut manifest = sampling_manifest(
        training_image,
        held_out_image,
        training_corners,
        held_out_corners,
    );
    set_corner_review_approvals(&mut manifest, manifest_directory, 16);
    manifest
}

#[test]
fn test_projective_target_sampler_feeds_disjoint_scanner_fit() {
    let tmp = tempfile::TempDir::new().unwrap();
    let training_corners = [[50.0, 45.0], [550.0, 70.0], [520.0, 380.0], [80.0, 355.0]];
    let held_out_corners = [[62.0, 38.0], [540.0, 58.0], [535.0, 370.0], [70.0, 365.0]];
    render_chart(
        &tmp.path().join("training.tiff"),
        training_corners,
        [1.0, 1.0, 1.0],
        None,
    );
    render_chart(
        &tmp.path().join("held-out.tiff"),
        held_out_corners,
        [1.001, 0.999, 1.0005],
        None,
    );
    let manifest = approved_sampling_manifest(
        tmp.path(),
        "training.tiff",
        "held-out.tiff",
        training_corners,
        held_out_corners,
    );
    let manifest_path = tmp.path().join("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let measurements = tmp.path().join("measurements.json");
    let previews = tmp.path().join("previews");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "sample-target",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--output",
            measurements.to_str().unwrap(),
            "--preview-dir",
            previews.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "target sampling failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let measurements_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&measurements).unwrap()).unwrap();
    assert_eq!(
        measurements_json["training_patches"]
            .as_array()
            .unwrap()
            .len(),
        12
    );
    assert_eq!(
        measurements_json["held_out_patches"]
            .as_array()
            .unwrap()
            .len(),
        12
    );
    assert_eq!(measurements_json["target_sampling"]["status"], "accepted");
    assert_eq!(
        measurements_json["target_sampling"]["reference_review"]["status"],
        "approved"
    );
    assert_eq!(
        measurements_json["target_sampling"]["reference_review"]["reference_patch_sha256_matched"],
        true
    );
    assert_eq!(
        measurements_json["target_sampling"]["corner_review_approved_capture_count"],
        2
    );
    assert_eq!(
        measurements_json["target_sampling"]["corner_review_required_capture_count"],
        0
    );
    assert_eq!(
        measurements_json["target_sampling"]["captures"][0]["corner_review"]["status"],
        "approved"
    );
    assert_eq!(
        measurements_json["target_sampling"]["captures"][0]["corner_review"]
            ["decoded_pixel_sha256_matched"],
        true
    );
    assert_eq!(
        measurements_json["target_sampling"]["captures"][0]["corner_review"]
            ["chart_corners_sha256_matched"],
        true
    );
    assert_eq!(
        measurements_json["target_sampling"]["captures"][0]["corner_review"]
            ["overlay_pixel_sha256_matched"],
        true
    );
    assert_eq!(
        measurements_json["target_sampling"]["captures"][0]["image_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    let sampled = measurements_json["training_patches"][0]["rgb"]
        .as_array()
        .unwrap();
    for channel in 0..3 {
        assert!(
            (sampled[channel].as_f64().unwrap() - PATCH_COLORS[0][channel]).abs() < 0.001,
            "sampled patch channel {channel} should recover projective interior"
        );
    }
    assert!(measurements_json["training_patches"][0]["scanner_xy"]
        .as_array()
        .unwrap()
        .iter()
        .all(|value| (-1.0..=1.0).contains(&value.as_f64().unwrap())));
    assert!(previews.join("capture-training.png").is_file());
    assert!(previews.join("capture-held-out.png").is_file());
    assert!(previews.join("sampling-report.json").is_file());
    assert!(previews.join("corner-review-manifest.json").is_file());

    let library = tmp.path().join("calibration");
    let fit = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "scanner-target",
            "--library",
            library.to_str().unwrap(),
            "--profile-id",
            "sampled-scanner",
            "--measurements",
            measurements.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        fit.status.success(),
        "sampled scanner fit failed: {}",
        String::from_utf8_lossy(&fit.stderr)
    );
    let record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(library.join("scanners/sampled-scanner.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        record["target_patch_signal_domain"],
        "normalized_scanner_signal"
    );
    for row in 0..3 {
        for column in 0..3 {
            assert!(
                (record["scanner_rgb_to_xyz"][row][column].as_f64().unwrap()
                    - REFERENCE_MATRIX[row][column])
                    .abs()
                    < 0.002
            );
        }
    }
    assert_eq!(record["target_sampling"]["status"], "accepted");
    let loaded = scanstitch::color_calibration::load_calibration(
        None,
        Some(&library),
        Some("sampled-scanner"),
        None,
        None,
        None,
    );
    assert_eq!(
        loaded
            .diagnostics
            .scanner_profile
            .as_ref()
            .and_then(|scanner| scanner.target_sampling.as_ref())
            .and_then(|sampling| sampling.get("status"))
            .and_then(serde_json::Value::as_str),
        Some("accepted")
    );
    let mut rejected_claim = record.clone();
    rejected_claim["target_sampling"]["status"] = serde_json::json!("rejected");
    let rejection =
        scanstitch::color_calibration::validate_library_record_value(&rejected_claim, None)
            .expect_err("rejected acquisition evidence must not load as calibration");
    assert!(rejection
        .reasons
        .iter()
        .any(|reason| reason.contains("status must be accepted")));

    let mut unapproved_corners = record.clone();
    unapproved_corners["target_sampling"]["captures"][0]["corner_review"]["status"] =
        serde_json::json!("requires_approval");
    let rejection =
        scanstitch::color_calibration::validate_library_record_value(&unapproved_corners, None)
            .expect_err("unapproved corner geometry must not survive record validation");
    assert!(rejection
        .reasons
        .iter()
        .any(|reason| reason.contains("corner_review.status must be approved")));

    let mut changed_corner_binding = record.clone();
    changed_corner_binding["target_sampling"]["captures"][0]["chart_corners_sha256"] =
        serde_json::json!("0".repeat(64));
    let rejection =
        scanstitch::color_calibration::validate_library_record_value(&changed_corner_binding, None)
            .expect_err("changed corner geometry binding must not survive record validation");
    assert!(rejection
        .reasons
        .iter()
        .any(|reason| reason
            .contains("expected_chart_corners_sha256 must equal chart_corners_sha256")));

    let mut changed_overlay_binding = record.clone();
    changed_overlay_binding["target_sampling"]["captures"][0]["overlay_pixel_sha256"] =
        serde_json::json!("0".repeat(64));
    let rejection = scanstitch::color_calibration::validate_library_record_value(
        &changed_overlay_binding,
        None,
    )
    .expect_err("changed overlay binding must not survive record validation");
    assert!(rejection
        .reasons
        .iter()
        .any(|reason| reason
            .contains("expected_overlay_pixel_sha256 must equal overlay_pixel_sha256")));

    let mut changed_retained_reference = record.clone();
    changed_retained_reference["target_sampling"]["reference_patches"][0]["reference_xyz"][0] =
        serde_json::json!(0.5);
    let rejection = scanstitch::color_calibration::validate_library_record_value(
        &changed_retained_reference,
        None,
    )
    .expect_err("changed retained reference values must not survive record validation");
    assert!(rejection
        .reasons
        .iter()
        .any(|reason| reason.contains("reference_patch_sha256 does not match reference_patches")));

    let mut changed_corner_coordinates = record.clone();
    changed_corner_coordinates["target_sampling"]["captures"][0]["chart_corners"]["top_left"][0] =
        serde_json::json!(51.0);
    let rejection = scanstitch::color_calibration::validate_library_record_value(
        &changed_corner_coordinates,
        None,
    )
    .expect_err("changed retained corner coordinates must not survive record validation");
    assert!(rejection
        .reasons
        .iter()
        .any(|reason| reason.contains("chart_corners_sha256 does not match chart_corners")));

    let mut duplicate_pixels = record;
    let first_pixel_hash =
        duplicate_pixels["target_sampling"]["captures"][0]["decoded_pixel_sha256"].clone();
    duplicate_pixels["target_sampling"]["captures"][1]["decoded_pixel_sha256"] = first_pixel_hash;
    let rejection =
        scanstitch::color_calibration::validate_library_record_value(&duplicate_pixels, None)
            .expect_err("reused decoded pixels must not survive record validation");
    assert!(rejection
        .reasons
        .iter()
        .any(|reason| reason.contains("duplicate decoded_pixel_sha256")));
}

#[test]
fn test_target_sampler_requires_hash_bound_corner_review_before_writing_measurement() {
    let tmp = tempfile::TempDir::new().unwrap();
    let training_corners = [[50.0, 45.0], [550.0, 70.0], [520.0, 380.0], [80.0, 355.0]];
    let held_out_corners = [[62.0, 38.0], [540.0, 58.0], [535.0, 370.0], [70.0, 365.0]];
    render_chart(
        &tmp.path().join("training.tiff"),
        training_corners,
        [1.0, 1.0, 1.0],
        None,
    );
    render_chart(
        &tmp.path().join("held-out.tiff"),
        held_out_corners,
        [1.001, 0.999, 1.0005],
        None,
    );
    let manifest = sampling_manifest(
        "training.tiff",
        "held-out.tiff",
        training_corners,
        held_out_corners,
    );
    let manifest_path = tmp.path().join("manifest-draft.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let measurements = tmp.path().join("measurements.json");
    let review_dir = tmp.path().join("review");
    let pending = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "sample-target",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--output",
            measurements.to_str().unwrap(),
            "--preview-dir",
            review_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!pending.status.success());
    assert!(!measurements.exists());
    assert!(String::from_utf8_lossy(&pending.stderr).contains("requires explicit approval"));
    let report: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(review_dir.join("sampling-report.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report["status"], "requires_corner_approval");
    assert_eq!(report["corner_review_approved_capture_count"], 0);
    assert_eq!(report["corner_review_required_capture_count"], 2);
    assert!(report["captures"]
        .as_array()
        .unwrap()
        .iter()
        .all(|capture| capture["corner_review"]["status"] == "not_declared"));
    assert!(review_dir.join("capture-training.png").is_file());
    assert!(review_dir.join("capture-held-out.png").is_file());

    let reviewed_manifest_path = review_dir.join("corner-review-manifest.json");
    let mut reviewed_manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&reviewed_manifest_path).unwrap()).unwrap();
    for capture in reviewed_manifest["captures"].as_array_mut().unwrap() {
        assert_eq!(capture["corner_review"]["approved"], false);
        assert_eq!(
            capture["corner_review"]["decoded_pixel_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert_eq!(
            capture["corner_review"]["chart_corners_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert_eq!(
            capture["corner_review"]["overlay_pixel_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert!(
            std::path::Path::new(capture["image"].as_str().unwrap()).is_absolute(),
            "generated review manifests must remain runnable from their output directory"
        );
        capture["corner_review"]["approved"] = serde_json::json!(true);
    }
    std::fs::write(
        &reviewed_manifest_path,
        serde_json::to_string_pretty(&reviewed_manifest).unwrap(),
    )
    .unwrap();
    let accepted_measurements = tmp.path().join("accepted-measurements.json");
    let accepted_review_dir = tmp.path().join("accepted-review");
    let accepted = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "sample-target",
            "--manifest",
            reviewed_manifest_path.to_str().unwrap(),
            "--output",
            accepted_measurements.to_str().unwrap(),
            "--preview-dir",
            accepted_review_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        accepted.status.success(),
        "approved corner review should permit sampling: {}",
        String::from_utf8_lossy(&accepted.stderr)
    );
    let accepted_measurement: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&accepted_measurements).unwrap()).unwrap();
    assert_eq!(
        accepted_measurement["target_sampling"]["status"],
        "accepted"
    );
    assert_eq!(
        accepted_measurement["target_sampling"]["corner_review_approved_capture_count"],
        2
    );

    let mut withdrawn_manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(accepted_review_dir.join("corner-review-manifest.json")).unwrap(),
    )
    .unwrap();
    withdrawn_manifest["captures"][0]["corner_review"]["approved"] = serde_json::json!(false);
    let withdrawn_manifest_path = tmp.path().join("withdrawn-review.json");
    std::fs::write(
        &withdrawn_manifest_path,
        serde_json::to_string_pretty(&withdrawn_manifest).unwrap(),
    )
    .unwrap();
    let withdrawn = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "sample-target",
            "--manifest",
            withdrawn_manifest_path.to_str().unwrap(),
            "--output",
            accepted_measurements.to_str().unwrap(),
            "--preview-dir",
            accepted_review_dir.to_str().unwrap(),
            "--force",
        ])
        .output()
        .unwrap();
    assert!(!withdrawn.status.success());
    assert!(
        !accepted_measurements.exists(),
        "a forced non-accepted rerun must remove a stale fit-ready measurement"
    );
}

#[test]
fn test_target_sampler_invalidates_corner_approval_when_decoded_pixels_change() {
    let tmp = tempfile::TempDir::new().unwrap();
    let training_corners = [[50.0, 45.0], [550.0, 70.0], [520.0, 380.0], [80.0, 355.0]];
    let held_out_corners = [[62.0, 38.0], [540.0, 58.0], [535.0, 370.0], [70.0, 365.0]];
    let training_path = tmp.path().join("training.tiff");
    render_chart(&training_path, training_corners, [1.0, 1.0, 1.0], None);
    render_chart(
        &tmp.path().join("held-out.tiff"),
        held_out_corners,
        [1.001, 0.999, 1.0005],
        None,
    );
    let manifest_value = approved_sampling_manifest(
        tmp.path(),
        "training.tiff",
        "held-out.tiff",
        training_corners,
        held_out_corners,
    );
    let mut changed = scanstitch::tiff_io::load_tiff_u16(&training_path, 16)
        .unwrap()
        .image;
    changed[[0, 0, 0]] ^= 1;
    scanstitch::tiff_io::save_tiff_u16(&changed, &training_path).unwrap();
    let manifest: scanstitch::calibration_target::TargetSamplingManifest =
        serde_json::from_value(manifest_value).unwrap();
    let result =
        scanstitch::calibration_target::sample_target_manifest(&manifest, tmp.path()).unwrap();
    assert_eq!(result.diagnostics.status, "requires_corner_approval");
    assert_eq!(result.diagnostics.corner_review_approved_capture_count, 1);
    assert_eq!(result.diagnostics.corner_review_required_capture_count, 1);
    assert_eq!(
        result.diagnostics.captures[0].corner_review.status,
        "decoded_pixel_mismatch"
    );
    assert_eq!(
        result.diagnostics.captures[0]
            .corner_review
            .decoded_pixel_sha256_matched,
        Some(false)
    );
    assert!(
        !result.corner_review_manifest.captures[0]
            .corner_review
            .as_ref()
            .unwrap()
            .approved
    );
    assert_eq!(
        result.corner_review_manifest.captures[0]
            .corner_review
            .as_ref()
            .unwrap()
            .decoded_pixel_sha256,
        result.diagnostics.captures[0].decoded_pixel_sha256
    );
}

#[test]
fn test_target_sampler_invalidates_corner_approval_when_geometry_changes() {
    let tmp = tempfile::TempDir::new().unwrap();
    let training_corners = [[50.0, 45.0], [550.0, 70.0], [520.0, 380.0], [80.0, 355.0]];
    let held_out_corners = [[62.0, 38.0], [540.0, 58.0], [535.0, 370.0], [70.0, 365.0]];
    render_chart(
        &tmp.path().join("training.tiff"),
        training_corners,
        [1.0, 1.0, 1.0],
        None,
    );
    render_chart(
        &tmp.path().join("held-out.tiff"),
        held_out_corners,
        [1.001, 0.999, 1.0005],
        None,
    );
    let approved_manifest = approved_sampling_manifest(
        tmp.path(),
        "training.tiff",
        "held-out.tiff",
        training_corners,
        held_out_corners,
    );
    let mut manifest_value = approved_manifest.clone();
    manifest_value["captures"][0]["corners"]["top_left"][0] = serde_json::json!(51.0);
    let manifest: scanstitch::calibration_target::TargetSamplingManifest =
        serde_json::from_value(manifest_value).unwrap();
    let result =
        scanstitch::calibration_target::sample_target_manifest(&manifest, tmp.path()).unwrap();
    assert_eq!(result.diagnostics.status, "requires_corner_approval");
    let review = &result.diagnostics.captures[0].corner_review;
    assert_eq!(review.status, "chart_corners_mismatch");
    assert_eq!(review.decoded_pixel_sha256_matched, Some(true));
    assert_eq!(review.chart_corners_sha256_matched, Some(false));
    let draft = result.corner_review_manifest.captures[0]
        .corner_review
        .as_ref()
        .unwrap();
    assert!(!draft.approved);
    assert_eq!(
        draft.chart_corners_sha256,
        result.diagnostics.captures[0].chart_corners_sha256
    );

    let mut changed_layout = approved_manifest.clone();
    changed_layout["grid"]["patch_inset_fraction"] = serde_json::json!(0.25);
    let manifest: scanstitch::calibration_target::TargetSamplingManifest =
        serde_json::from_value(changed_layout).unwrap();
    let result =
        scanstitch::calibration_target::sample_target_manifest(&manifest, tmp.path()).unwrap();
    assert_eq!(result.diagnostics.status, "requires_corner_approval");
    let review = &result.diagnostics.captures[0].corner_review;
    assert_eq!(review.status, "overlay_pixel_mismatch");
    assert_eq!(review.decoded_pixel_sha256_matched, Some(true));
    assert_eq!(review.chart_corners_sha256_matched, Some(true));
    assert_eq!(review.overlay_pixel_sha256_matched, Some(false));
    let draft = result.corner_review_manifest.captures[0]
        .corner_review
        .as_ref()
        .unwrap();
    assert!(!draft.approved);
    assert_eq!(
        draft.overlay_pixel_sha256,
        result.diagnostics.captures[0].overlay_pixel_sha256
    );

    let mut missing_reference_review = approved_manifest.clone();
    missing_reference_review
        .as_object_mut()
        .unwrap()
        .remove("reference_review");
    let manifest: scanstitch::calibration_target::TargetSamplingManifest =
        serde_json::from_value(missing_reference_review).unwrap();
    let result =
        scanstitch::calibration_target::sample_target_manifest(&manifest, tmp.path()).unwrap();
    assert_eq!(result.diagnostics.status, "requires_reference_approval");
    assert_eq!(result.diagnostics.reference_review.status, "not_declared");
    let reference_draft = result
        .corner_review_manifest
        .reference_review
        .as_ref()
        .unwrap();
    assert!(!reference_draft.approved);
    assert_eq!(
        reference_draft.reference_patch_sha256,
        result.diagnostics.reference_patch_sha256
    );

    let mut changed_reference = approved_manifest;
    changed_reference["patches"][0]["reference_xyz"][0] = serde_json::json!(0.5);
    let manifest: scanstitch::calibration_target::TargetSamplingManifest =
        serde_json::from_value(changed_reference).unwrap();
    let result =
        scanstitch::calibration_target::sample_target_manifest(&manifest, tmp.path()).unwrap();
    assert_eq!(result.diagnostics.status, "requires_reference_approval");
    assert_eq!(
        result.diagnostics.reference_review.status,
        "reference_patch_mismatch"
    );
    assert_eq!(
        result
            .diagnostics
            .reference_review
            .reference_patch_sha256_matched,
        Some(false)
    );
}

#[test]
fn test_target_sampler_rejects_duplicate_capture_payloads() {
    let tmp = tempfile::TempDir::new().unwrap();
    let corners = [[50.0, 45.0], [550.0, 70.0], [520.0, 380.0], [80.0, 355.0]];
    render_chart(
        &tmp.path().join("same.tiff"),
        corners,
        [1.0, 1.0, 1.0],
        None,
    );
    let manifest_value = sampling_manifest("same.tiff", "same.tiff", corners, corners);
    let manifest: scanstitch::calibration_target::TargetSamplingManifest =
        serde_json::from_value(manifest_value).unwrap();
    let error = match scanstitch::calibration_target::sample_target_manifest(&manifest, tmp.path())
    {
        Ok(_) => panic!("identical training/held-out payloads must fail"),
        Err(error) => error,
    };
    assert!(error.contains("identical SHA-256"));
}

#[test]
fn test_target_sampler_rejects_same_decoded_pixels_with_different_file_hash() {
    let tmp = tempfile::TempDir::new().unwrap();
    let corners = [[50.0, 45.0], [550.0, 70.0], [520.0, 380.0], [80.0, 355.0]];
    let first = tmp.path().join("first.tiff");
    let second = tmp.path().join("metadata-modified.tiff");
    render_chart(&first, corners, [1.0, 1.0, 1.0], None);
    let mut modified = std::fs::read(&first).unwrap();
    modified.extend_from_slice(b"ignored-trailing-container-metadata");
    std::fs::write(&second, modified).unwrap();
    let manifest_value =
        sampling_manifest("first.tiff", "metadata-modified.tiff", corners, corners);
    let manifest: scanstitch::calibration_target::TargetSamplingManifest =
        serde_json::from_value(manifest_value).unwrap();
    let error = match scanstitch::calibration_target::sample_target_manifest(&manifest, tmp.path())
    {
        Ok(_) => panic!("identical decoded pixels must fail even when file hashes differ"),
        Err(error) => error,
    };
    assert!(error.contains("identical decoded-pixel SHA-256"));
}

#[test]
fn test_target_sampler_honors_14_bit_codes_in_16_bit_tiff_container() {
    let tmp = tempfile::TempDir::new().unwrap();
    let training_corners = [[50.0, 45.0], [550.0, 70.0], [520.0, 380.0], [80.0, 355.0]];
    let held_out_corners = [[62.0, 38.0], [540.0, 58.0], [535.0, 370.0], [70.0, 365.0]];
    render_chart_with_code_max(
        &tmp.path().join("training-14-in-16.tiff"),
        training_corners,
        [1.0, 1.0, 1.0],
        None,
        16_383,
    );
    render_chart_with_code_max(
        &tmp.path().join("held-out-14-in-16.tiff"),
        held_out_corners,
        [1.001, 0.999, 1.0005],
        None,
        16_383,
    );
    let mut manifest_value = sampling_manifest(
        "training-14-in-16.tiff",
        "held-out-14-in-16.tiff",
        training_corners,
        held_out_corners,
    );
    manifest_value["scanner_signal_bit_depth"] = serde_json::json!(14);
    set_corner_review_approvals(&mut manifest_value, tmp.path(), 14);
    let manifest: scanstitch::calibration_target::TargetSamplingManifest =
        serde_json::from_value(manifest_value).unwrap();
    let result =
        scanstitch::calibration_target::sample_target_manifest(&manifest, tmp.path()).unwrap();
    assert_eq!(result.diagnostics.status, "accepted");
    assert_eq!(result.diagnostics.scanner_signal_bit_depth, 14);
    assert_eq!(result.diagnostics.scanner_signal_code_max, 16_383);
    assert_eq!(
        result.diagnostics.captures[0].source_bits_per_sample, 16,
        "TIFF container should still advertise 16 bits"
    );
    assert_eq!(
        result.diagnostics.captures[0].working_range_transform, "none",
        "14-bit codes already fit the declared working range"
    );
    let sampled = result.measurement["training_patches"][0]["rgb"]
        .as_array()
        .unwrap();
    for channel in 0..3 {
        assert!((sampled[channel].as_f64().unwrap() - PATCH_COLORS[0][channel]).abs() < 0.001);
    }
}

#[test]
fn test_target_sampler_writes_audit_but_no_measurement_for_contaminated_patch() {
    let tmp = tempfile::TempDir::new().unwrap();
    let training_corners = [[50.0, 45.0], [550.0, 70.0], [520.0, 380.0], [80.0, 355.0]];
    let held_out_corners = [[62.0, 38.0], [540.0, 58.0], [535.0, 370.0], [70.0, 365.0]];
    render_chart(
        &tmp.path().join("training.tiff"),
        training_corners,
        [1.0, 1.0, 1.0],
        None,
    );
    render_chart(
        &tmp.path().join("held-out.tiff"),
        held_out_corners,
        [1.001, 0.999, 1.0005],
        Some(5),
    );
    let manifest = sampling_manifest(
        "training.tiff",
        "held-out.tiff",
        training_corners,
        held_out_corners,
    );
    let manifest_path = tmp.path().join("manifest.json");
    std::fs::write(&manifest_path, manifest.to_string()).unwrap();
    let measurements = tmp.path().join("measurements.json");
    let previews = tmp.path().join("previews");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_scanstitch-calibrate"))
        .args([
            "sample-target",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--output",
            measurements.to_str().unwrap(),
            "--preview-dir",
            previews.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!measurements.exists());
    let report: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(previews.join("sampling-report.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report["status"], "rejected");
    assert!(report["rejected_patch_count"].as_u64().unwrap() >= 1);
    assert!(report["captures"][1]["patches"]
        .as_array()
        .unwrap()
        .iter()
        .any(|patch| patch["status"] == "rejected"));
    assert!(previews.join("capture-held-out.png").is_file());
}

#[test]
fn test_fractional_scanner_coordinate_mapping_matches_exif_orientation() {
    let mapping = scanstitch::scanner_linearization::ScannerCoordinateMapping {
        orientation_tag: Some(6),
        scanner_frame_width: 100,
        scanner_frame_height: 50,
    };
    assert_eq!(
        mapping.normalized_scanner_coordinates(0.0, 0.0).unwrap(),
        [-1.0, 1.0]
    );
    let opposite = mapping.normalized_scanner_coordinates(49.0, 99.0).unwrap();
    assert!((opposite[0] - 1.0).abs() < 1e-12);
    assert!((opposite[1] + 1.0).abs() < 1e-12);
}

#[test]
fn test_documented_target_sampling_manifest_matches_runtime_contract() {
    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/calibration-target-sampling.schema.json"
    ))
    .unwrap();
    assert_eq!(schema["properties"]["schema_version"]["const"], 1);
    assert_eq!(
        schema["properties"]["scanner_signal_bit_depth"]["minimum"],
        8
    );
    assert_eq!(schema["properties"]["patches"]["minItems"], 12);
    assert_eq!(
        schema["properties"]["captures"]["allOf"][1]["contains"]["properties"]["role"]["const"],
        "held_out"
    );
    assert_eq!(
        schema["$defs"]["capture"]["properties"]["corner_review"]["$ref"],
        "#/$defs/cornerReview"
    );
    assert_eq!(
        schema["properties"]["reference_review"]["$ref"],
        "#/$defs/referenceReview"
    );
    assert_eq!(
        schema["$defs"]["referenceReview"]["required"],
        serde_json::json!(["approved", "reference_patch_sha256"])
    );
    assert_eq!(
        schema["$defs"]["cornerReview"]["required"],
        serde_json::json!([
            "approved",
            "decoded_pixel_sha256",
            "chart_corners_sha256",
            "overlay_pixel_sha256"
        ])
    );
    let manifest: scanstitch::calibration_target::TargetSamplingManifest = serde_json::from_str(
        include_str!("../docs/calibration-target-sampling.example.json"),
    )
    .unwrap();
    scanstitch::calibration_target::validate_target_sampling_manifest(&manifest).unwrap();
    assert_eq!(manifest.patches.len(), 12);
    assert_eq!(manifest.captures.len(), 2);
    assert!(manifest.reference_review.is_none());
    assert_eq!(
        manifest.measurement["reference"]["status"],
        "template_only_zero_values_must_be_replaced"
    );
    assert!(manifest
        .patches
        .iter()
        .all(|patch| patch.reference_xyz == Some([0.0; 3])));
}

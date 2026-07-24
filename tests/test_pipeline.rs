mod common;
use common::synthetic;
use ndarray::{s, Array3};
use std::process::Command;
use tempfile::TempDir;

fn mean_luminance_region(img: &Array3<u16>, x0: usize, x1: usize) -> f64 {
    let (height, width, _) = img.dim();
    let x0 = x0.min(width);
    let x1 = x1.min(width).max(x0 + 1);
    let mut sum = 0.0;
    let mut count = 0usize;
    for y in 0..height {
        for x in x0..x1 {
            sum += 0.2126 * img[[y, x, 0]] as f64
                + 0.7152 * img[[y, x, 1]] as f64
                + 0.0722 * img[[y, x, 2]] as f64;
            count += 1;
        }
    }
    sum / count.max(1) as f64
}

fn textured_positive_panorama(height: usize, width: usize) -> Array3<u16> {
    let mut image = Array3::<u16>::zeros((height, width, 3));
    for y in 0..height {
        for x in 0..width {
            let tx = x as f64 / width.saturating_sub(1).max(1) as f64;
            let ty = y as f64 / height.saturating_sub(1).max(1) as f64;
            let feature =
                ((x.wrapping_mul(73) ^ y.wrapping_mul(151) ^ (x * y + 17)) % 997) as f64 / 997.0;
            image[[y, x, 0]] = ((0.10 + 0.55 * tx + 0.12 * feature) * 16_383.0) as u16;
            image[[y, x, 1]] = ((0.12 + 0.48 * ty + 0.10 * feature) * 16_383.0) as u16;
            image[[y, x, 2]] = ((0.14 + 0.34 * (1.0 - tx) + 0.16 * feature) * 16_383.0) as u16;
        }
    }
    image
}

fn varied_film_negative_chart(
    height: usize,
    width: usize,
    rebate_cols: usize,
    base_rgb: [u16; 3],
) -> Array3<u16> {
    let mut image = Array3::<u16>::zeros((height, width, 3));
    for y in 0..height {
        for x in 0..width {
            if x < rebate_cols || x >= width.saturating_sub(rebate_cols) {
                for channel in 0..3 {
                    image[[y, x, channel]] = base_rgb[channel];
                }
                continue;
            }

            let content_x = x - rebate_cols;
            let content_width = width.saturating_sub(2 * rebate_cols).max(1);
            let luminance_density =
                0.04 + 0.82 * content_x as f64 / content_width.saturating_sub(1).max(1) as f64;
            let patch = ((content_x / 12) + (y / 12)) % 5;
            let chroma_offsets = match patch {
                0 | 1 => [0.0, 0.0, 0.0],
                2 => [-0.035, 0.018, 0.025],
                3 => [0.028, -0.030, 0.014],
                _ => [0.018, 0.022, -0.038],
            };
            for channel in 0..3 {
                let density = (luminance_density + chroma_offsets[channel]).max(0.0);
                image[[y, x, channel]] =
                    (base_rgb[channel] as f64 * (-density).exp()).round() as u16;
            }
        }
    }
    image
}

fn box_blur_u16(img: &Array3<u16>, radius: usize) -> Array3<u16> {
    let (height, width, channels) = img.dim();
    let mut horizontal = Array3::<f64>::zeros((height, width, channels));
    for y in 0..height {
        for x in 0..width {
            let x_start = x.saturating_sub(radius);
            let x_end = (x + radius + 1).min(width);
            for channel in 0..channels {
                horizontal[[y, x, channel]] = (x_start..x_end)
                    .map(|sample_x| img[[y, sample_x, channel]] as f64)
                    .sum::<f64>()
                    / (x_end - x_start) as f64;
            }
        }
    }
    let mut output = Array3::<u16>::zeros((height, width, channels));
    for y in 0..height {
        let y_start = y.saturating_sub(radius);
        let y_end = (y + radius + 1).min(height);
        for x in 0..width {
            for channel in 0..channels {
                output[[y, x, channel]] = ((y_start..y_end)
                    .map(|sample_y| horizontal[[sample_y, x, channel]])
                    .sum::<f64>()
                    / (y_end - y_start) as f64)
                    .round()
                    .clamp(0.0, 16383.0) as u16;
            }
        }
    }
    output
}

fn positive_fast_cli(
    inputs: Vec<std::path::PathBuf>,
    output_dir: std::path::PathBuf,
) -> scanstitch::cli::Cli {
    scanstitch::cli::Cli {
        inputs,
        output_dir,
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Positive,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Fast,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: false,
        transform: "translation".to_string(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    }
}

fn write_rgb8_tiff(
    path: &std::path::Path,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::create(path)?;
    let mut encoder = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file))?;
    encoder.write_image::<tiff::encoder::colortype::RGB8>(width, height, pixels)?;
    Ok(())
}

fn write_rgb16_tiff_with_icc(
    path: &std::path::Path,
    width: u32,
    height: u32,
    pixels: &[u16],
    icc_profile: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::create(path)?;
    let mut encoder = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file))?;
    let mut image = encoder.new_image::<tiff::encoder::colortype::RGB16>(width, height)?;
    image
        .encoder()
        .write_tag(tiff::tags::Tag::Unknown(34_675), icc_profile)?;
    image.write_data(pixels)?;
    Ok(())
}

fn valid_calibration_profile_json() -> String {
    serde_json::json!({
        "schema_version": 1,
        "profile_id": "synthetic-prophoto-d50",
        "source_space": { "name": "synthetic linear scanner RGB", "encoding": "linear" },
        "scanner": { "make": "Synthetic", "model": "Pipeline Test" },
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
    .to_string()
}

fn singular_calibration_profile_json() -> String {
    serde_json::json!({
        "schema_version": 1,
        "profile_id": "singular-synthetic",
        "source_space": { "name": "synthetic linear scanner RGB", "encoding": "linear" },
        "scanner": { "make": "Synthetic", "model": "Pipeline Test" },
        "film": { "stock": "Synthetic negative", "process": "C-41" },
        "target": { "type": "synthetic_color_checker", "illuminant": "D50" },
        "reference": { "dataset": "synthetic" },
        "whitepoint": [0.9642, 1.0, 0.8251],
        "work_to_xyz": [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        "confidence": 0.95
    })
    .to_string()
}

fn scanner_library_profile_json() -> String {
    let fingerprint = synthetic_scanner_settings_fingerprint();
    serde_json::json!({
        "schema_version": 1,
        "record_type": "scanner_profile",
        "profile_id": "scanner-a",
        "source_space": { "name": "synthetic linear scanner RGB", "encoding": "linear" },
        "scanner": { "make": "Synthetic", "model": "Pipeline Library Test" },
        "settings": { "dpi": 3200, "mode": "positive" },
        "target": { "type": "synthetic_color_checker", "illuminant": "D50" },
        "reference": { "dataset": "synthetic" },
        "whitepoint": [0.9642, 1.0, 0.8251],
        "scanner_rgb_to_xyz": [
            [0.7977, 0.1352, 0.0313],
            [0.2880, 0.7119, 0.0001],
            [0.0000, 0.0000, 0.8251]
        ],
        "scanner_settings_fingerprint": fingerprint,
        "confidence": 0.96
    })
    .to_string()
}

fn scanner_library_profile_json_with_linearization() -> String {
    let mut scanner: serde_json::Value =
        serde_json::from_str(&scanner_library_profile_json()).unwrap();
    scanner["schema_version"] = serde_json::json!(2);
    scanner["scanner_linearization"] = serde_json::json!({
        "model_id": "pipeline-scanner-linearization-v1",
        "curves": {
            "red": [[0.0, 0.0], [0.25, 0.0625], [0.5, 0.25], [0.75, 0.5625], [1.0, 1.0]],
            "green": [[0.0, 0.0], [0.25, 0.0625], [0.5, 0.25], [0.75, 0.5625], [1.0, 1.0]],
            "blue": [[0.0, 0.0], [0.25, 0.0625], [0.5, 0.25], [0.75, 0.5625], [1.0, 1.0]]
        },
        "black_level_normalized": [0.01, 0.01, 0.01],
        "white_level_normalized": [0.99, 0.99, 0.99],
        "additive_flare_normalized": [0.002, 0.002, 0.002],
        "shading_gain_polynomial": [
            [1.0, 0.01, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.01, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.01, 0.0, 0.0, 0.0, 0.0]
        ],
        "confidence": 0.95,
        "validation": {
            "held_out_sample_count": 24,
            "transmittance_rmse": 0.002,
            "transmittance_max_error": 0.008,
            "identity_baseline_rmse": 0.08
        }
    });
    scanner.to_string()
}

fn roll_library_profile_json() -> String {
    let fingerprint = synthetic_scanner_settings_fingerprint();
    serde_json::json!({
        "schema_version": 1,
        "record_type": "roll_profile",
        "profile_id": "roll-a",
        "scanner_profile_id": "scanner-a",
        "film": { "stock": "Synthetic negative", "process": "C-41" },
        "development": { "developer": "synthetic" },
        "target": { "type": "synthetic_color_checker", "illuminant": "D50" },
        "reference": { "dataset": "synthetic-roll" },
        "base_color": [12000.0, 7000.0, 3000.0],
        "correction_domain": "xyz_post_scanner",
        "correction_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        "scanner_settings_fingerprint": fingerprint,
        "confidence": 0.90
    })
    .to_string()
}

fn roll_library_profile_json_with_measured_response() -> String {
    let mut roll: serde_json::Value = serde_json::from_str(&roll_library_profile_json()).unwrap();
    roll["negative_response"] = serde_json::json!({
        "model_id": "synthetic-pipeline-response-v1",
        "scanner_density_to_layer_density": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        "characteristic_curves": {
            "red": [[-0.50, -0.50], [0.0, 0.0], [0.70, 0.75], [1.50, 1.60]],
            "green": [[-0.50, -0.50], [0.0, 0.0], [0.70, 0.75], [1.50, 1.60]],
            "blue": [[-0.50, -0.50], [0.0, 0.0], [0.70, 0.75], [1.50, 1.60]]
        },
        "white_anchor_percentile": 0.995,
        "confidence": 0.94,
        "validation": {
            "held_out_patch_count": 24,
            "delta_e00_rms": 1.8,
            "delta_e00_max": 4.9,
            "unit_slope_delta_e00_rms": 8.2
        }
    });
    roll.to_string()
}

fn roll_library_profile_json_with_narrow_measured_response() -> String {
    let mut roll: serde_json::Value =
        serde_json::from_str(&roll_library_profile_json_with_measured_response()).unwrap();
    roll["negative_response"]["model_id"] =
        serde_json::json!("synthetic-pipeline-narrow-response-v1");
    roll["negative_response"]["characteristic_curves"] = serde_json::json!({
        "red": [[0.0, 0.0], [0.01, 0.016], [0.025, 0.040], [0.05, 0.080]],
        "green": [[0.0, 0.0], [0.01, 0.016], [0.025, 0.040], [0.05, 0.080]],
        "blue": [[0.0, 0.0], [0.01, 0.016], [0.025, 0.040], [0.05, 0.080]]
    });
    roll.to_string()
}

fn synthetic_scanner_settings_fingerprint() -> String {
    let settings = serde_json::json!({ "dpi": 3200, "mode": "positive" });
    scanstitch::color_calibration::scanner_settings_fingerprint(Some(&settings))
        .expect("synthetic scanner settings fingerprint")
}

fn write_library(tmp: &TempDir, include_invalid: bool) -> std::path::PathBuf {
    let library_dir = tmp.path().join("calibration");
    std::fs::create_dir_all(library_dir.join("scanners")).unwrap();
    std::fs::create_dir_all(library_dir.join("rolls")).unwrap();
    std::fs::write(
        library_dir.join("scanners/scanner.json"),
        scanner_library_profile_json(),
    )
    .unwrap();
    std::fs::write(
        library_dir.join("rolls/roll.json"),
        roll_library_profile_json(),
    )
    .unwrap();
    if include_invalid {
        std::fs::write(
            library_dir.join("rolls/invalid.json"),
            serde_json::json!({
                "schema_version": 1,
                "record_type": "roll_profile",
                "profile_id": "invalid-low-confidence",
                "film": { "stock": "Bad" },
                "base_color": [1.0, 1.0, 1.0],
                "confidence": 0.10
            })
            .to_string(),
        )
        .unwrap();
    }
    library_dir
}

#[test]
fn test_single_input_pipeline_requires_no_dummy_component() {
    let tmp = TempDir::new().unwrap();
    let input_path = tmp.path().join("single.tiff");
    let image = textured_positive_panorama(128, 420);
    scanstitch::tiff_io::save_tiff_u16(&image, &input_path).unwrap();
    let cli = positive_fast_cli(vec![input_path.clone()], tmp.path().join("output"));

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let load = report
        .phases
        .iter()
        .find(|phase| phase.name == "load")
        .expect("load phase");
    assert_eq!(load.metrics["input_count"], 1);
    assert_eq!(load.metrics["inputs"].as_array().unwrap().len(), 1);
    assert_eq!(
        load.metrics["inputs"][0].as_str(),
        Some(input_path.to_string_lossy().as_ref())
    );
    assert!(load.metrics.get("component2").is_none());

    let stitch = report
        .phases
        .iter()
        .find(|phase| phase.name == "stitch")
        .expect("stitch phase");
    assert_eq!(stitch.confidence, 0.0);
    assert_eq!(stitch.metrics["decision"], "skipped_single_input");
    assert_eq!(stitch.metrics["stitch_evidence_evaluated"], false);
    assert_eq!(
        stitch.metrics["confidence_basis"],
        "not_applicable_single_input"
    );
    assert_eq!(stitch.metrics["user_override"], false);
    assert_eq!(stitch.metrics["preserved_full_valid_union"], true);
    let working = report
        .phases
        .iter()
        .find(|phase| phase.name == "working_image_select")
        .expect("working image phase");
    assert_eq!(working.metrics["input_count"], 1);
    assert_eq!(working.metrics["stitch_review_required"], false);
    let tone = report
        .phases
        .iter()
        .find(|phase| phase.name == "tone_mapping")
        .expect("tone mapping phase");
    let skin_memory = &tone.metrics["adaptive_vibrance_skin_memory_protection"];
    let preferred_memory = &tone.metrics["adaptive_vibrance_preferred_memory_color_guard"];
    assert_eq!(tone.metrics["adaptive_vibrance_enabled"], false);
    assert_eq!(skin_memory["enabled"], false);
    assert_eq!(
        skin_memory["working_space"],
        "CIELAB_D50_from_linear_ProPhoto_RGB_D50"
    );
    assert_eq!(skin_memory["maximum_vibrance_reduction"], 0.90);
    assert!(skin_memory["evaluated_pixel_ratio"].as_f64().is_some());
    assert!(skin_memory["protected_pixel_ratio"].as_f64().is_some());
    assert_eq!(preferred_memory["enabled"], false);
    assert_eq!(
        preferred_memory["working_space"],
        "CIELAB_D50_ab_projection_from_linear_ProPhoto_RGB_D50"
    );
    assert_eq!(preferred_memory["evaluated_pixel_ratio"], 0.0);
    assert_eq!(preferred_memory["matched_pixel_ratio"], 0.0);
    assert_eq!(preferred_memory["limited_pixel_ratio"], 0.0);
    assert_eq!(preferred_memory["families"].as_array().unwrap().len(), 3);
    let summary = scanstitch::validation::summarize_report("single", &report);
    assert_eq!(
        summary
            .tone
            .adaptive_vibrance_skin_memory_protection
            .as_ref()
            .and_then(|protection| protection.enabled),
        Some(false)
    );
    assert_eq!(
        summary
            .tone
            .adaptive_vibrance_preferred_memory_color_guard
            .as_ref()
            .and_then(|guard| guard.enabled),
        Some(false)
    );
    assert!(summary.diagnostic_consistency_issues.is_empty());
    assert!(cli.output_dir.join("output.tiff").exists());
}

#[test]
fn test_pipeline_reports_supported_grain_detail_retention_without_false_review() {
    let tmp = TempDir::new().unwrap();
    let input_path = tmp.path().join("grain-detail-positive.tiff");
    let height = 192usize;
    let width = 420usize;
    let mut image = textured_positive_panorama(height, width);
    for y in 24..(height - 24) {
        for x in 80..(width - 80) {
            let rgb = if y < height / 2 {
                if x < width / 2 {
                    [11_468u16, 3_277u16, 3_277u16]
                } else {
                    [3_277u16, 6_594u16, 11_468u16]
                }
            } else {
                let level = if x < width / 2 { 3_000 } else { 11_800 };
                [level, level, level]
            };
            for c in 0..3 {
                image[[y, x, c]] = rgb[c];
            }
        }
    }
    scanstitch::tiff_io::save_tiff_u16(&image, &input_path).unwrap();
    let mut cli = positive_fast_cli(vec![input_path], tmp.path().join("output"));
    cli.grain.grain_reduction = scanstitch::cli::GrainReductionMode::On;
    cli.grain.grain_strength = 1.0;
    cli.grain.grain_scale = 1.0;

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let tone = report
        .phases
        .iter()
        .find(|phase| phase.name == "tone_mapping")
        .expect("tone mapping phase");
    let detail = &tone.metrics["grain_reduction"]["detail_retention"];
    let effective = &tone.metrics["grain_reduction"]["effective"];
    assert_eq!(tone.metrics["noise_reduction_enabled"], true);
    assert_eq!(effective["structure_gate_start"], 0.018);
    assert_eq!(effective["structure_gate_end"], 0.035);
    assert_eq!(
        effective["structure_excluded_ratio"],
        tone.metrics["noise_reduction_structure_excluded_ratio"]
    );
    let applied_ratio = effective["applied_ratio"].as_f64().unwrap();
    let excluded_ratio = effective["structure_excluded_ratio"].as_f64().unwrap();
    assert!(excluded_ratio > 0.0, "{effective:#}");
    assert!(
        applied_ratio + excluded_ratio <= 1.0 + 1e-12,
        "{effective:#}"
    );
    assert_eq!(detail["evaluated"], true);
    assert_eq!(detail["decision_supported"], true);
    assert_eq!(detail["review_required"], false, "{detail:#}");
    assert!(
        detail["luminance_probe_count"].as_u64().unwrap_or(0)
            >= detail["minimum_probe_count"].as_u64().unwrap_or(u64::MAX)
    );
    assert!(
        detail["chroma_probe_count"].as_u64().unwrap_or(0)
            >= detail["minimum_probe_count"].as_u64().unwrap_or(u64::MAX)
    );
    let save = report
        .phases
        .iter()
        .find(|phase| phase.name == "save")
        .expect("save phase");
    assert_eq!(save.metrics["grain_detail_review_required"], false);
}

#[test]
fn test_grain_render_review_binds_exact_matched_off_control() {
    let tmp = TempDir::new().unwrap();
    let input_path = tmp.path().join("profiled-grain-review.tiff");
    let height = 192usize;
    let width = 420usize;
    let mut image = textured_positive_panorama(height, width);
    for y in 24..(height - 24) {
        for x in 80..(width - 80) {
            let rgb = if y < height / 2 {
                if x < width / 2 {
                    [11_468u16, 3_277u16, 3_277u16]
                } else {
                    [3_277u16, 6_594u16, 11_468u16]
                }
            } else {
                let level = if x < width / 2 { 3_000 } else { 11_800 };
                [level, level, level]
            };
            for channel in 0..3 {
                image[[y, x, channel]] = rgb[channel];
            }
        }
    }
    let srgb_profile = moxcms::ColorProfile::new_srgb().encode().unwrap();
    let pixels = image
        .iter()
        .map(|value| value.saturating_mul(4))
        .collect::<Vec<_>>();
    write_rgb16_tiff_with_icc(
        &input_path,
        width as u32,
        height as u32,
        &pixels,
        &srgb_profile,
    )
    .unwrap();

    let control_output = tmp.path().join("grain-off");
    let mut control_cli = positive_fast_cli(vec![input_path], control_output.clone());
    control_cli.quality_mode = scanstitch::cli::QualityMode::Perfect;
    control_cli.bit_depth = 16;
    control_cli.force_no_stitch = true;
    control_cli.geometry.deskew = scanstitch::cli::DeskewMode::Off;
    let control_report = scanstitch::pipeline::run(&control_cli).unwrap();
    let control_summary =
        scanstitch::validation::summarize_report("grain-control", &control_report);
    assert_eq!(
        control_summary.render.render_reviewable,
        Some(true),
        "control status={:?} reason={:?} consistency={:?}",
        control_summary.render.render_review_status,
        control_summary.render.render_review_reason,
        control_summary.diagnostic_consistency_issues
    );
    assert_eq!(
        control_summary.render.noise_reduction_requested_enabled,
        Some(false)
    );

    let primary_output = tmp.path().join("grain-on");
    let mut primary_cli = control_cli.clone();
    primary_cli.output_dir = primary_output.clone();
    primary_cli.grain.grain_reduction = scanstitch::cli::GrainReductionMode::On;
    primary_cli.grain.grain_strength = 0.5;
    primary_cli.grain.grain_scale = 1.0;
    let primary_report = scanstitch::pipeline::run(&primary_cli).unwrap();
    let primary_summary =
        scanstitch::validation::summarize_report("grain-primary", &primary_report);
    assert_eq!(
        primary_summary.render.render_reviewable,
        Some(true),
        "primary status={:?} reason={:?} consistency={:?}",
        primary_summary.render.render_review_status,
        primary_summary.render.render_review_reason,
        primary_summary.diagnostic_consistency_issues
    );
    assert_eq!(
        primary_summary.render.noise_reduction_requested_enabled,
        Some(true)
    );

    let primary_report_path = primary_output.join("report.json");
    let control_report_path = control_output.join("report.json");
    let missing_control_error = scanstitch::render_review::write_render_review_draft(
        "grain-review",
        &primary_report_path,
        &tmp.path().join("missing-control-review"),
    )
    .unwrap_err()
    .to_string();
    assert!(missing_control_error.contains("grain-on render review requires"));

    let review_dir = tmp.path().join("grain-review");
    let draft = scanstitch::render_review::write_render_review_draft_with_grain_control(
        "grain-review",
        &primary_report_path,
        Some(&control_report_path),
        &review_dir,
    )
    .unwrap();
    assert_eq!(draft.manifest.schema_version, 2);
    assert!(draft.manifest.grain_reduction_applicable);
    let control = draft.manifest.grain_reduction_control.as_ref().unwrap();
    assert_eq!(control.purpose, "grain_reduction_off_control");
    assert!(control.technical_delivery_reviewable);
    assert_eq!(control.artifacts.len(), 3);
    let primary_master = draft
        .manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == "master_scene_referred")
        .unwrap();
    let control_master = control
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == "master_scene_referred")
        .unwrap();
    assert_eq!(primary_master.sha256, control_master.sha256);
    let markdown = std::fs::read_to_string(&draft.markdown_path).unwrap();
    assert!(markdown.contains("Exact grain-off control"));
    assert!(markdown.contains("scene-referred masters must remain byte-identical"));

    let cli_review_dir = tmp.path().join("grain-review-cli");
    let cli_review = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture")
        .arg("grain-review")
        .arg("--report")
        .arg(&primary_report_path)
        .arg("--write-render-review")
        .arg(&cli_review_dir)
        .arg("--render-review-grain-control-report")
        .arg(&control_report_path)
        .arg("--quiet")
        .output()
        .unwrap();
    assert!(
        cli_review.status.success(),
        "grain control CLI draft failed: {}",
        String::from_utf8_lossy(&cli_review.stderr)
    );

    let mut approved_manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&draft.json_path).unwrap()).unwrap();
    approved_manifest["review_status"] = serde_json::json!("approved");
    approved_manifest["reviewer"] = serde_json::json!("fixture curator");
    approved_manifest["reviewed_at"] = serde_json::json!("2026-07-21T18:00:00Z");
    approved_manifest["review_notes"] =
        serde_json::json!("Compared the exact bound grain-on and grain-off proofs at 100%.");
    for (name, decision) in approved_manifest["decisions"].as_object_mut().unwrap() {
        if decision["applicable"] == serde_json::json!(true) {
            decision["approved"] = serde_json::json!(true);
            decision["notes"] =
                serde_json::json!(format!("Reviewed {name} against the bound control."));
        }
    }
    std::fs::write(
        &draft.json_path,
        serde_json::to_string_pretty(&approved_manifest).unwrap(),
    )
    .unwrap();
    let approved =
        scanstitch::render_review::inspect_render_review_manifest(&draft.json_path, "grain-review");
    assert!(approved.approved, "review issues: {:?}", approved.issues);
    assert!(approved.grain_control_present);
    assert_eq!(approved.grain_control_report_sha256_matches, Some(true));
    assert_eq!(approved.grain_control_inputs_match, Some(true));
    assert_eq!(approved.grain_control_render_contract_matches, Some(true));
    assert_eq!(approved.grain_control_master_matches, Some(true));
    assert_eq!(approved.all_grain_control_artifact_hashes_match, Some(true));

    let original_control_report = std::fs::read_to_string(&control_report_path).unwrap();
    std::fs::write(&control_report_path, format!("{original_control_report}\n")).unwrap();
    let tampered =
        scanstitch::render_review::inspect_render_review_manifest(&draft.json_path, "grain-review");
    assert!(!tampered.approved);
    assert!(tampered
        .issues
        .contains(&"grain_control_report_sha256_mismatch".to_string()));
    std::fs::write(&control_report_path, original_control_report).unwrap();

    let control_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&control_report_path).unwrap()).unwrap();
    let mut mismatched_interactive_control = control_json.clone();
    let tone = mismatched_interactive_control["phases"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|phase| phase["name"] == "tone_mapping")
        .unwrap();
    tone["metrics"]["interactive_controls"] = serde_json::json!({
        "exposure_ev": 0.5,
        "grain_reduction_enabled": false,
        "grain_reduction_strength": 0.5,
        "grain_reduction_scale": 1.0
    });
    let mismatched_interactive_path = tmp.path().join("mismatched-interactive-report.json");
    std::fs::write(
        &mismatched_interactive_path,
        serde_json::to_string_pretty(&mismatched_interactive_control).unwrap(),
    )
    .unwrap();
    let interactive_error =
        scanstitch::render_review::write_render_review_draft_with_grain_control(
            "grain-review",
            &primary_report_path,
            Some(&mismatched_interactive_path),
            &tmp.path().join("mismatched-interactive-review"),
        )
        .unwrap_err()
        .to_string();
    assert!(interactive_error.contains("interactive_controls"));

    let mut mismatched_binary_control = control_json.clone();
    mismatched_binary_control["metadata"]["binary_sha256"] = serde_json::json!("b".repeat(64));
    let mismatched_binary_path = tmp.path().join("mismatched-binary-report.json");
    std::fs::write(
        &mismatched_binary_path,
        serde_json::to_string_pretty(&mismatched_binary_control).unwrap(),
    )
    .unwrap();
    let binary_error = scanstitch::render_review::write_render_review_draft_with_grain_control(
        "grain-review",
        &primary_report_path,
        Some(&mismatched_binary_path),
        &tmp.path().join("mismatched-binary-review"),
    )
    .unwrap_err()
    .to_string();
    assert!(binary_error.contains("binary_sha256"));

    let mut legacy_binary_control = control_json.clone();
    let metadata = legacy_binary_control["metadata"].as_object_mut().unwrap();
    metadata.remove("binary_path");
    metadata.remove("binary_sha256");
    metadata.remove("binary_file_size_bytes");
    metadata.remove("binary_identity_status");
    metadata.remove("binary_identity_error");
    let legacy_binary_path = tmp.path().join("legacy-binary-report.json");
    std::fs::write(
        &legacy_binary_path,
        serde_json::to_string_pretty(&legacy_binary_control).unwrap(),
    )
    .unwrap();
    let legacy_binary_error =
        scanstitch::render_review::write_render_review_draft_with_grain_control(
            "grain-review",
            &primary_report_path,
            Some(&legacy_binary_path),
            &tmp.path().join("legacy-binary-review"),
        )
        .unwrap_err()
        .to_string();
    assert!(legacy_binary_error.contains("binary_identity_missing_or_invalid"));

    let mut mismatched_control = control_json;
    mismatched_control["metadata"]["cli_args"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!("--different-render-contract"));
    let mismatched_control_path = tmp.path().join("mismatched-control-report.json");
    std::fs::write(
        &mismatched_control_path,
        serde_json::to_string_pretty(&mismatched_control).unwrap(),
    )
    .unwrap();
    let mismatch_error = scanstitch::render_review::write_render_review_draft_with_grain_control(
        "grain-review",
        &primary_report_path,
        Some(&mismatched_control_path),
        &tmp.path().join("mismatched-control-review"),
    )
    .unwrap_err()
    .to_string();
    assert!(mismatch_error.contains("normalized_cli_args"));
}

#[test]
fn test_pipeline_orders_and_stitches_three_shuffled_inputs() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("inputs");
    std::fs::create_dir_all(&input_dir).unwrap();
    let panorama = textured_positive_panorama(180, 1_000);
    let left = panorama.slice(s![.., 0..420, ..]).to_owned();
    let middle = panorama.slice(s![.., 290..710, ..]).to_owned();
    let right = panorama.slice(s![.., 580..1_000, ..]).to_owned();
    let right_path = input_dir.join("right.tiff");
    let left_path = input_dir.join("left.tiff");
    let middle_path = input_dir.join("middle.tiff");
    scanstitch::tiff_io::save_tiff_u16(&right, &right_path).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&left, &left_path).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&middle, &middle_path).unwrap();
    let cli = positive_fast_cli(
        vec![right_path, left_path, middle_path],
        tmp.path().join("output"),
    );

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let classify = report
        .phases
        .iter()
        .find(|phase| phase.name == "base_detect_classify")
        .expect("classification phase");
    assert_eq!(classify.confidence, 0.0);
    assert_eq!(
        classify.metrics["classification_status"],
        "not_applicable_positive_input"
    );
    assert_eq!(classify.metrics["classification_evidence_evaluated"], false);
    assert!(classify.metrics["components"]
        .as_array()
        .unwrap()
        .iter()
        .all(|component| component["classification_evidence_evaluated"] == false));
    let stitch = report
        .phases
        .iter()
        .find(|phase| phase.name == "stitch")
        .expect("stitch phase");
    assert!(stitch.success, "{:?}", stitch.errors);
    assert_eq!(stitch.metrics["decision"], "accepted_sequence");
    assert_eq!(stitch.metrics["input_count"], 3);
    assert_eq!(
        stitch.metrics["ordering"]["inferred_order"],
        serde_json::json!([2, 3, 1])
    );
    assert_eq!(stitch.metrics["accepted_merge_count"], 2);
    assert_eq!(stitch.metrics["preserved_full_valid_union"], true);
    assert_eq!(stitch.metrics["seam_quality_review_required"], false);
    assert_eq!(
        stitch.metrics["seam_quality_review_reasons"],
        serde_json::json!([])
    );
    let working = report
        .phases
        .iter()
        .find(|phase| phase.name == "working_image_select")
        .expect("working image phase");
    assert_eq!(working.metrics["used_stitched_working_image"], true);
    assert_eq!(working.metrics["stitch_review_required"], false);
    let output_width = working.metrics["working_shape"][1].as_u64().unwrap() as i32;
    assert!(
        (output_width - 1_000).unsigned_abs() <= 24,
        "working union width was {output_width}"
    );
}

#[test]
fn test_pipeline_blocks_review_for_repeated_cross_scan_detail_mismatch() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("inputs");
    std::fs::create_dir_all(&input_dir).unwrap();
    let panorama = textured_positive_panorama(240, 960);
    let left = panorama.slice(s![.., 0..570, ..]).to_owned();
    let right = panorama.slice(s![.., 390..960, ..]).to_owned();
    let right = box_blur_u16(&right, 4);
    let left_path = input_dir.join("left.tiff");
    let right_path = input_dir.join("right-blurred.tiff");
    scanstitch::tiff_io::save_tiff_u16(&left, &left_path).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&right, &right_path).unwrap();
    let mut cli = positive_fast_cli(vec![left_path, right_path], tmp.path().join("output"));
    cli.force_stitch = true;

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let stitch = report
        .phases
        .iter()
        .find(|phase| phase.name == "stitch")
        .expect("stitch phase");
    assert!(stitch.success, "{:?}", stitch.errors);
    assert_eq!(stitch.metrics["decision"], "accepted");
    assert_eq!(stitch.metrics["seam_blend"]["review_required"], true);
    assert_eq!(
        stitch.metrics["seam_blend"]["detail_consistency"]["review_required"],
        true
    );
    assert!(stitch.confidence <= 0.35);
    assert!(stitch
        .warnings
        .iter()
        .any(|warning| warning.contains("seam-quality review")));

    let working = report
        .phases
        .iter()
        .find(|phase| phase.name == "working_image_select")
        .expect("working image phase");
    assert_eq!(working.metrics["used_stitched_working_image"], true);
    assert_eq!(working.metrics["stitch_review_required"], true);
    assert!(working.metrics["stitch_review_reason"]
        .as_str()
        .expect("stitch review reason")
        .contains("detail"));

    let save = report
        .phases
        .iter()
        .find(|phase| phase.name == "save")
        .expect("save phase");
    assert_eq!(save.metrics["geometry_review_required"], true);
    assert_eq!(
        save.metrics["render_review_status"],
        "blocked_geometry_review"
    );
    assert_eq!(save.metrics["render_reviewable"], false);
    assert!(save.metrics["render_review_reason"]
        .as_str()
        .expect("render review reason")
        .contains("detail"));
}

#[test]
fn test_pipeline_separates_technical_and_creative_white_balance() {
    let tmp = TempDir::new().unwrap();
    let input_path = tmp.path().join("positive.tiff");
    let image = textured_positive_panorama(96, 160);
    scanstitch::tiff_io::save_tiff_u16(&image, &input_path).unwrap();
    let mut cli = positive_fast_cli(vec![input_path], tmp.path().join("output"));
    cli.white_balance.technical_white_balance = scanstitch::cli::TechnicalWhiteBalanceMode::Manual;
    cli.white_balance.technical_temperature_kelvin = 6500.0;
    cli.white_balance.technical_tint = 0.12;
    cli.white_balance.creative_temperature = 0.4;
    cli.white_balance.creative_tint = -0.2;

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let white_balance = report
        .phases
        .iter()
        .find(|phase| phase.name == "white_balance")
        .expect("white-balance phase");
    assert_eq!(
        white_balance.metrics["technical"]["status"],
        "applied_manual"
    );
    assert_eq!(white_balance.confidence, 1.0);
    assert_eq!(
        white_balance.metrics["technical_decision_status"],
        "user_authoritative_override"
    );
    assert_eq!(white_balance.metrics["technical_evidence_evaluated"], false);
    assert_eq!(
        white_balance.metrics["confidence_basis"],
        "user_authoritative_override"
    );
    assert_eq!(white_balance.metrics["technical"]["applied"], true);
    assert_eq!(
        white_balance.metrics["technical"]["preserves_signed_scene_headroom"],
        true
    );
    assert_eq!(
        white_balance.metrics["creative_defaults"]["excluded_from_scene_referred_master"],
        true
    );
    let tone = report
        .phases
        .iter()
        .find(|phase| phase.name == "tone_mapping")
        .expect("tone phase");
    assert_eq!(tone.metrics["creative_white_balance"]["applied"], true);
    assert_eq!(tone.metrics["creative_white_balance"]["temperature"], 0.4);
    assert_eq!(
        tone.metrics["creative_white_balance"]["separated_from_technical_master"],
        true
    );
    let summary = scanstitch::validation::summarize_report("white-balance", &report);
    assert_eq!(
        summary.white_balance.technical_status.as_deref(),
        Some("applied_manual")
    );
    assert_eq!(summary.white_balance.technical_applied, Some(true));
    assert_eq!(summary.white_balance.creative_temperature, Some(0.4));
    assert_eq!(
        summary
            .white_balance
            .creative_separated_from_technical_master,
        Some(true)
    );
}

#[test]
fn test_failed_multi_input_overlap_blocks_geometry_review() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("inputs");
    std::fs::create_dir_all(&input_dir).unwrap();
    let panorama = textured_positive_panorama(160, 710);
    let left = panorama.slice(s![.., 0..420, ..]).to_owned();
    let right = panorama.slice(s![.., 290..710, ..]).to_owned();
    let unrelated = synthetic::constant_image(160, 420, [13_000, 2_000, 9_000]);
    let paths = ["left.tiff", "right.tiff", "unrelated.tiff"]
        .into_iter()
        .map(|name| input_dir.join(name))
        .collect::<Vec<_>>();
    scanstitch::tiff_io::save_tiff_u16(&left, &paths[0]).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&right, &paths[1]).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&unrelated, &paths[2]).unwrap();
    let cli = positive_fast_cli(paths, tmp.path().join("output"));

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let stitch = report
        .phases
        .iter()
        .find(|phase| phase.name == "stitch")
        .expect("stitch phase");
    assert!(!stitch.success);
    assert_eq!(stitch.metrics["decision"], "rejected_sequence");
    let save = report
        .phases
        .iter()
        .find(|phase| phase.name == "save")
        .expect("save phase");
    assert_eq!(save.metrics["geometry_review_required"], true);
    assert_eq!(
        save.metrics["render_review_status"],
        "blocked_geometry_review"
    );
    assert_eq!(save.metrics["render_reviewable"], false);
    assert!(save.metrics["geometry_review_reason"]
        .as_str()
        .unwrap()
        .contains("not a complete frame"));
}

#[test]
fn test_full_pipeline_no_stitch() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp1 =
        synthetic::film_negative_image(200, 400, 10, 30, [12000, 7000, 3000], [5000, 4000, 3500]);
    let comp2 =
        synthetic::film_negative_image(200, 400, 10, 30, [12000, 7000, 3000], [6000, 4500, 3200]);

    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp1, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp2, &path2).unwrap();

    let output_dir = tmp.path().join("output");
    std::fs::create_dir_all(&output_dir).unwrap();
    scanstitch::tiff_io::save_tiff_u16(
        &synthetic::constant_image(4, 4, [1000, 1000, 1000]),
        &output_dir.join("old_debug_artifact.tiff"),
    )
    .unwrap();

    let cli = scanstitch::cli::Cli {
        inputs: vec![path1.clone(), path2.clone()],
        output_dir: output_dir.clone(),
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: true,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let metadata = report.metadata.as_ref().expect("run metadata");
    assert_eq!(metadata.report_schema_version, 4);
    assert_eq!(metadata.package_version, env!("CARGO_PKG_VERSION"));
    assert!(metadata.generated_at.contains('T'));
    assert_eq!(
        metadata.binary_identity_status.as_deref(),
        Some("verified_sha256")
    );
    assert!(metadata.binary_sha256.as_deref().is_some_and(
        |digest| digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    ));
    assert!(metadata.binary_file_size_bytes.is_some_and(|size| size > 0));
    assert!(metadata
        .binary_path
        .as_deref()
        .is_some_and(|path| !path.is_empty()));
    assert_eq!(metadata.binary_identity_error, None);
    assert!(!report.phases.is_empty());
    assert!(output_dir.join("output.tiff").exists());
    assert!(output_dir.join("master_scene_referred.tiff").exists());
    assert!(output_dir.join("review_srgb.png").exists());
    assert!(
        output_dir.join("phase46_gamut_clipping_map.tiff").exists(),
        "expected debug gamut/clipping map artifact"
    );
    assert!(
        output_dir
            .join("phase46_scene_referred_prophoto_float.tiff")
            .exists(),
        "expected debug scene-referred ProPhoto float artifact"
    );
    let colorspace_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "colorspace_mapping")
        .expect("colorspace phase");
    assert!(colorspace_phase.metrics["gamut_clipping_map_artifact"].is_string());
    assert!(colorspace_phase.metrics["scene_referred_prophoto_float_artifact"].is_string());
    assert_eq!(
        colorspace_phase.metrics["scene_referred_prophoto_float_artifact_diagnostics"]
            ["sample_format"],
        "IEEEFP"
    );
    assert_eq!(
        colorspace_phase.metrics["scene_referred_prophoto_float_artifact_diagnostics"]
            ["normalization"],
        "none"
    );
    assert_eq!(
        colorspace_phase.metrics["gamut_clipping_map_diagnostics"]["encoding"],
        "red=post-scale high clipping, blue=post-scale low clipping, green=in-gamut luminance; yellow/magenta indicates mixed high/low channel clipping"
    );
    assert!(
        colorspace_phase.metrics["gamut_clipping_map_diagnostics"]["preserved_ratio"].is_number()
    );
    let save_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "save")
        .expect("save phase");
    assert!(save_phase.metrics["output_width"].is_number());
    assert!(save_phase.metrics["output_height"].is_number());
    assert_eq!(
        save_phase.metrics["output_color_space"],
        "linear_prophoto_rgb_d50"
    );
    assert_eq!(
        save_phase.metrics["output_icc_profile"]["embedded"],
        serde_json::json!(true)
    );
    assert_eq!(save_phase.metrics["render_intent"], "modern-clean");
    assert_eq!(save_phase.metrics["quality_mode"], "perfect");
    let tone_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "tone_mapping")
        .expect("tone phase");
    assert_eq!(
        tone_phase.metrics["render_input_buffer_policy"],
        "owned_scene_buffer_reused_for_batch_render"
    );
    assert_eq!(
        tone_phase.metrics["scene_referred_master_preservation_policy"],
        "prewritten_before_owned_batch_render"
    );
    assert_eq!(
        save_phase.metrics["artifact_write_buffer_policy"]["strategy"],
        "sequential_bounded_tiff_strips_and_png_rows"
    );
    assert_eq!(
        save_phase.metrics["artifact_write_buffer_policy"]["full_frame_conversion_buffers"],
        false
    );
    assert!(
        save_phase.metrics["artifact_write_buffer_policy"]["peak_declared_buffer_bytes"]
            .as_u64()
            .is_some_and(|bytes| bytes > 0 && bytes < 1_200_000)
    );
    assert_eq!(
        save_phase.metrics["artifact_sha256_policy"]["algorithm"],
        "sha256"
    );
    assert_eq!(
        save_phase.metrics["artifact_sha256_policy"]["source"],
        "reopened_saved_file_bytes"
    );
    assert_eq!(
        save_phase.metrics["artifact_sha256_policy"]["buffer_bytes"],
        scanstitch::report::FILE_SHA256_BUFFER_BYTES
    );
    assert_eq!(
        save_phase.metrics["artifact_sha256_policy"]["overlaps_artifact_conversion_buffers"],
        false
    );
    assert!(
        save_phase.metrics["artifact_sha256_policy"]["peak_sequential_delivery_buffer_bytes"]
            .as_u64()
            .is_some_and(|bytes| bytes >= scanstitch::report::FILE_SHA256_BUFFER_BYTES as u64)
    );
    assert_eq!(
        save_phase.metrics["artifact_sha256_policy"]["required_for_reviewable_delivery"],
        true
    );
    assert_eq!(
        save_phase.metrics["artifact_commit_policy"]["strategy"],
        scanstitch::atomic_file::ATOMIC_COMMIT_STRATEGY
    );
    assert_eq!(
        save_phase.metrics["artifact_commit_policy"]["same_directory_staging"],
        true
    );
    assert_eq!(
        save_phase.metrics["artifact_commit_policy"]["per_file_atomic_replace"],
        true
    );
    assert_eq!(
        save_phase.metrics["artifact_commit_policy"]["cross_artifact_transaction"],
        false
    );
    assert_eq!(
        save_phase.metrics["artifact_commit_policy"]
            ["destination_visible_only_after_successful_encode"],
        true
    );
    assert_eq!(
        save_phase.metrics["artifact_commit_policy"]
            ["failed_encode_preserves_existing_destination"],
        true
    );
    assert_eq!(
        save_phase.metrics["artifact_commit_policy"]["file_contents_fsync_before_commit"],
        false
    );
    assert_eq!(
        save_phase.metrics["artifact_commit_policy"]["directory_fsync_after_commit"],
        false
    );
    assert_eq!(save_phase.metrics["master_scene_referred_requested"], true);
    assert_eq!(
        save_phase.metrics["master_write_timing"],
        "before_owned_batch_tonemap"
    );
    assert!(save_phase.metrics["master_write_duration_ms"].is_number());
    assert!(save_phase.metrics["master_scene_referred_path"].is_string());
    assert_eq!(
        save_phase.metrics["master_scene_referred_diagnostics"]["sample_format"],
        "IEEEFP"
    );
    assert_eq!(save_phase.metrics["review_srgb_requested"], true);
    assert!(save_phase.metrics["review_srgb_path"].is_string());
    assert_eq!(
        save_phase.metrics["review_srgb_icc_profile"]["description"],
        scanstitch::tiff_io::SRGB_ICC_DESCRIPTION
    );
    assert_eq!(
        save_phase.metrics["review_srgb_gamut_mapping"]["space"],
        scanstitch::tiff_io::SRGB_REVIEW_GAMUT_MAPPING_SPACE
    );
    assert_eq!(
        save_phase.metrics["review_srgb_gamut_mapping"]["post_map_out_of_gamut_pixel_count"],
        0
    );
    assert!(
        save_phase.metrics["review_srgb_gamut_mapping"]["gamut_mapped_ratio"]
            .as_f64()
            .is_some_and(|ratio| (0.0..=1.0).contains(&ratio))
    );
    assert_eq!(save_phase.metrics["overwrote_existing_output"], false);
    assert_eq!(save_phase.metrics["stale_render_artifact_count"], 1);
    assert!(save_phase
        .warnings
        .iter()
        .any(|warning| warning.contains("not overwritten")));
    for (field, path) in [
        ("output_sha256", output_dir.join("output.tiff")),
        (
            "master_scene_referred_sha256",
            output_dir.join("master_scene_referred.tiff"),
        ),
        ("review_srgb_sha256", output_dir.join("review_srgb.png")),
    ] {
        let declared = save_phase.metrics[field].as_str().expect(field);
        let (actual, size_bytes) = scanstitch::report::hash_file_sha256(&path).unwrap();
        assert_eq!(declared, actual);
        assert!(size_bytes > 0);
    }

    let summary = scanstitch::validation::summarize_report("perfect-delivery", &report);
    assert!(summary.render.artifact_sha256_binding_required);
    assert_eq!(summary.render.output_file_sha256_matches_report, Some(true));
    assert_eq!(
        summary.render.output_file_dimensions_match_report,
        Some(true)
    );
    assert_eq!(
        summary.render.output_file_storage_matches_report,
        Some(true)
    );
    assert_eq!(
        summary.render.master_scene_referred_file_status.as_deref(),
        Some("valid_scene_referred_master")
    );
    assert_eq!(
        summary.render.master_scene_referred_file_matches_report,
        Some(true)
    );
    assert_eq!(
        summary
            .render
            .master_scene_referred_file_sha256_matches_report,
        Some(true)
    );
    assert_eq!(
        summary.render.review_srgb_file_status.as_deref(),
        Some("valid_srgb_review_png")
    );
    assert_eq!(summary.render.review_srgb_file_matches_report, Some(true));
    assert_eq!(
        summary.render.review_srgb_file_sha256_matches_report,
        Some(true)
    );
    assert_eq!(
        summary.render.review_srgb_gamut_mapping_supported,
        Some(true)
    );
    assert_eq!(
        summary.render.review_srgb_post_map_out_of_gamut_pixel_count,
        Some(0)
    );

    let mut claimed_reviewable_report = report.clone();
    let claimed_save = claimed_reviewable_report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "save")
        .unwrap();
    claimed_save.metrics["render_review_status"] = serde_json::json!("reviewable");
    claimed_save.metrics["render_reviewable"] = serde_json::json!(true);
    claimed_save.metrics["stale_render_artifact_count"] = serde_json::json!(0);
    claimed_save.metrics["stale_render_artifacts"] = serde_json::json!([]);

    let claimed_report_path = output_dir.join("claimed-reviewable-report.json");
    claimed_reviewable_report
        .save(&claimed_report_path)
        .unwrap();
    let cli_review_package_dir = output_dir.join("cli-render-review-package");
    let cli_review = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture")
        .arg("logan")
        .arg("--report")
        .arg(&claimed_report_path)
        .arg("--write-render-review")
        .arg(&cli_review_package_dir)
        .arg("--quiet")
        .output()
        .unwrap();
    assert!(
        cli_review.status.success(),
        "CLI render-review draft failed: {}",
        String::from_utf8_lossy(&cli_review.stderr)
    );
    let cli_review_manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(cli_review_package_dir.join("render-review.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        cli_review_manifest["review_status"],
        "requires_human_approval"
    );
    assert_eq!(
        cli_review_manifest["artifacts"].as_array().unwrap().len(),
        3
    );
    let review_package_dir = output_dir.join("render-review-package");
    let draft = scanstitch::render_review::write_render_review_draft(
        "logan",
        &claimed_report_path,
        &review_package_dir,
    )
    .unwrap();
    assert_eq!(draft.manifest.review_status, "requires_human_approval");
    assert!(draft.manifest.technical_delivery_reviewable);
    assert_eq!(draft.manifest.inputs.len(), 2);
    assert_eq!(draft.manifest.artifacts.len(), 3);
    assert!(
        !scanstitch::render_review::inspect_render_review_manifest(&draft.json_path, "logan")
            .approved
    );

    let mut approved_manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&draft.json_path).unwrap()).unwrap();
    approved_manifest["review_status"] = serde_json::json!("approved");
    approved_manifest["reviewer"] = serde_json::json!("fixture curator");
    approved_manifest["reviewed_at"] = serde_json::json!("2026-07-19T12:00:00Z");
    approved_manifest["review_notes"] =
        serde_json::json!("Full-resolution delivery reviewed against the fixture intent.");
    for (name, decision) in approved_manifest["decisions"].as_object_mut().unwrap() {
        if decision["applicable"] == serde_json::json!(true) {
            decision["approved"] = serde_json::json!(true);
            decision["notes"] = serde_json::json!(format!(
                "Reviewed {name} at full resolution; no unacceptable defect observed."
            ));
        }
    }
    std::fs::write(
        &draft.json_path,
        serde_json::to_string_pretty(&approved_manifest).unwrap(),
    )
    .unwrap();
    let approved_review =
        scanstitch::render_review::inspect_render_review_manifest(&draft.json_path, "logan");
    assert!(
        approved_review.approved,
        "review issues: {:?}",
        approved_review.issues
    );

    let nonreviewable_report_path = output_dir.join("nonreviewable-report.json");
    report.save(&nonreviewable_report_path).unwrap();
    let nonreviewable_draft = scanstitch::render_review::write_render_review_draft(
        "logan",
        &nonreviewable_report_path,
        &output_dir.join("nonreviewable-render-review-package"),
    )
    .unwrap();
    assert!(!nonreviewable_draft.manifest.technical_delivery_reviewable);
    let mut attempted_override: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&nonreviewable_draft.json_path).unwrap())
            .unwrap();
    attempted_override["review_status"] = serde_json::json!("approved");
    attempted_override["reviewer"] = serde_json::json!("fixture curator");
    attempted_override["reviewed_at"] = serde_json::json!("2026-07-19T12:05:00Z");
    attempted_override["review_notes"] =
        serde_json::json!("Attempted approval of a technically blocked delivery.");
    for (name, decision) in attempted_override["decisions"].as_object_mut().unwrap() {
        if decision["applicable"] == serde_json::json!(true) {
            decision["approved"] = serde_json::json!(true);
            decision["notes"] = serde_json::json!(format!("Reviewed {name}."));
        }
    }
    std::fs::write(
        &nonreviewable_draft.json_path,
        serde_json::to_string_pretty(&attempted_override).unwrap(),
    )
    .unwrap();
    let rejected_override = scanstitch::render_review::inspect_render_review_manifest(
        &nonreviewable_draft.json_path,
        "logan",
    );
    assert!(!rejected_override.approved);
    assert!(rejected_override
        .issues
        .contains(&"technical_delivery_not_reviewable".to_string()));

    let claimed_summary = scanstitch::validation::summarize_report_with_source(
        "logan",
        &claimed_reviewable_report,
        Some(&claimed_report_path),
    );
    let baseline_path = output_dir.join("claimed-summary-baseline.json");
    std::fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&scanstitch::validation::tracked_baseline_from_summary(
            &claimed_summary,
        ))
        .unwrap(),
    )
    .unwrap();
    let approved_manifest_bytes = std::fs::read(&draft.json_path).unwrap();
    let render_review_sha256 = {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(&approved_manifest_bytes))
    };
    let registry_path = output_dir.join("render-review-fixtures.json");
    std::fs::write(
        &registry_path,
        serde_json::to_string_pretty(&serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "min_summary_baselines": 1,
                "min_uncalibrated_fixtures": 1,
                "min_unique_film_stocks": 1,
                "min_scene_tags": 1,
                "min_exposure_tags": 1,
                "min_calibration_cases": 1,
                "min_approved_render_review_fixtures": 1
            },
            "fixtures": {
                "logan": {
                    "component1": path1.clone(),
                    "component2": path2.clone(),
                    "input_mode": "negative",
                    "bit_depth": 14,
                    "force_no_stitch": true,
                    "summary_baseline": baseline_path.clone(),
                    "render_review": draft.json_path.clone(),
                    "render_review_sha256": render_review_sha256,
                    "film_stock": "Synthetic review stock",
                    "scene_tags": ["review-contract"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "uncalibrated-image-derived"
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let coverage_path = output_dir.join("render-review-coverage.json");
    let coverage = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-coverage")
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&coverage_path)
        .output()
        .unwrap();
    assert!(
        coverage.status.success(),
        "approved render-review coverage failed: {}",
        String::from_utf8_lossy(&coverage.stderr)
    );
    let coverage_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&coverage_path).unwrap()).unwrap();
    assert_eq!(coverage_json["approved_render_review_fixture_count"], 1);
    assert_eq!(
        coverage_json["fixtures"][0]["render_review_inspection"]["approved"],
        true
    );

    std::fs::write(
        &draft.json_path,
        format!(
            "{}\n",
            String::from_utf8(approved_manifest_bytes.clone()).unwrap()
        ),
    )
    .unwrap();
    let changed_manifest_coverage = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-coverage")
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(output_dir.join("changed-render-review-coverage.json"))
        .output()
        .unwrap();
    assert!(!changed_manifest_coverage.status.success());
    assert!(String::from_utf8_lossy(&changed_manifest_coverage.stderr)
        .contains("render_review_sha256_mismatch"));
    std::fs::write(&draft.json_path, &approved_manifest_bytes).unwrap();

    let matching_registry = std::fs::read_to_string(&registry_path).unwrap();
    let mut mismatched_registry: serde_json::Value =
        serde_json::from_str(&matching_registry).unwrap();
    mismatched_registry["fixtures"]["logan"]["component1"] = serde_json::json!(path2.clone());
    mismatched_registry["fixtures"]["logan"]["component2"] = serde_json::json!(path1.clone());
    std::fs::write(
        &registry_path,
        serde_json::to_string_pretty(&mismatched_registry).unwrap(),
    )
    .unwrap();
    let mismatched_input_coverage = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-coverage")
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(output_dir.join("mismatched-input-render-review-coverage.json"))
        .output()
        .unwrap();
    assert!(!mismatched_input_coverage.status.success());
    assert!(String::from_utf8_lossy(&mismatched_input_coverage.stderr)
        .contains("registry_input_set_mismatch"));
    std::fs::write(&registry_path, matching_registry).unwrap();

    let mut unsupported_mapping_report = claimed_reviewable_report.clone();
    unsupported_mapping_report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "save")
        .unwrap()
        .metrics["review_srgb_gamut_mapping"]["post_map_out_of_gamut_pixel_count"] =
        serde_json::json!(1);
    let unsupported_mapping = scanstitch::validation::summarize_report(
        "unsupported-review-mapping",
        &unsupported_mapping_report,
    );
    assert_eq!(
        unsupported_mapping
            .render
            .review_srgb_gamut_mapping_supported,
        Some(false)
    );
    assert!(scanstitch::validation::delivery_artifact_integrity_issues(
        &unsupported_mapping.render
    )
    .contains(&"review_srgb_gamut_mapping_not_supported"));
    assert!(
        !scanstitch::validation::final_delivery_evidence_is_reviewable(&unsupported_mapping.render)
    );

    let output_path = output_dir.join("output.tiff");
    let held_output_path = output_dir.join("output.held");
    std::fs::rename(&output_path, &held_output_path).unwrap();
    let substitute = Array3::<f64>::zeros((
        summary.render.output_height.unwrap(),
        summary.render.output_width.unwrap(),
        3,
    ));
    scanstitch::tiff_io::save_tiff_f64_linear_prophoto(&substitute, &output_path).unwrap();
    let substituted_output =
        scanstitch::validation::summarize_report("substituted-output", &claimed_reviewable_report);
    assert_eq!(
        substituted_output
            .render
            .output_file_dimensions_match_report,
        Some(true)
    );
    assert_eq!(
        substituted_output.render.output_file_storage_matches_report,
        Some(true)
    );
    assert_eq!(
        substituted_output
            .render
            .output_file_icc_profile_matches_report,
        Some(true)
    );
    assert_eq!(
        substituted_output.render.output_file_sha256_matches_report,
        Some(false)
    );
    assert!(
        scanstitch::validation::delivery_artifact_integrity_issues(&substituted_output.render)
            .contains(&"output_sha256_mismatch")
    );
    assert!(
        !scanstitch::validation::final_delivery_evidence_is_reviewable(&substituted_output.render)
    );
    let substituted_validation = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture")
        .arg("substituted-output")
        .arg("--report")
        .arg(&claimed_report_path)
        .arg("--require-reviewable")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(output_dir.join("substituted-output-validation"))
        .output()
        .unwrap();
    assert!(!substituted_validation.status.success());
    assert!(
        String::from_utf8_lossy(&substituted_validation.stderr).contains("output_sha256_mismatch")
    );
    std::fs::remove_file(&output_path).unwrap();
    std::fs::rename(&held_output_path, &output_path).unwrap();

    let review_path = output_dir.join("review_srgb.png");
    let held_valid_review_path = output_dir.join("review_srgb.valid-held");
    std::fs::rename(&review_path, &held_valid_review_path).unwrap();
    scanstitch::tiff_io::save_srgb_png_from_linear_prophoto(&substitute, &review_path).unwrap();
    let substituted_review =
        scanstitch::validation::summarize_report("substituted-review", &claimed_reviewable_report);
    assert_eq!(
        substituted_review.render.review_srgb_file_matches_report,
        Some(true)
    );
    assert_eq!(
        substituted_review
            .render
            .review_srgb_file_sha256_matches_report,
        Some(false)
    );
    assert!(
        scanstitch::validation::delivery_artifact_integrity_issues(&substituted_review.render)
            .contains(&"review_srgb_sha256_mismatch")
    );
    assert!(
        !scanstitch::validation::final_delivery_evidence_is_reviewable(&substituted_review.render)
    );
    std::fs::remove_file(&review_path).unwrap();
    std::fs::rename(&held_valid_review_path, &review_path).unwrap();

    let held_review_path = output_dir.join("review_srgb.held");
    std::fs::rename(&review_path, &held_review_path).unwrap();
    let missing_review =
        scanstitch::validation::summarize_report("missing-review", &claimed_reviewable_report);
    assert_eq!(
        missing_review.render.review_srgb_file_matches_report,
        Some(false)
    );
    assert!(
        scanstitch::validation::delivery_artifact_integrity_issues(&missing_review.render)
            .contains(&"review_srgb_artifact_mismatch")
    );
    assert!(!scanstitch::validation::final_delivery_evidence_is_reviewable(&missing_review.render));
    let changed_artifact_review =
        scanstitch::render_review::inspect_render_review_manifest(&draft.json_path, "logan");
    assert!(!changed_artifact_review.approved);
    assert!(changed_artifact_review
        .issues
        .iter()
        .any(|issue| issue.starts_with("artifact_unreadable:review_srgb")));
    std::fs::rename(&held_review_path, &review_path).unwrap();

    let master_path = output_dir.join("master_scene_referred.tiff");
    let held_valid_master_path = output_dir.join("master_scene_referred.valid-held");
    std::fs::rename(&master_path, &held_valid_master_path).unwrap();
    scanstitch::tiff_io::save_tiff_f32_linear_prophoto_scene_referred(&substitute, &master_path)
        .unwrap();
    let substituted_master =
        scanstitch::validation::summarize_report("substituted-master", &claimed_reviewable_report);
    assert_eq!(
        substituted_master
            .render
            .master_scene_referred_file_matches_report,
        Some(true)
    );
    assert_eq!(
        substituted_master
            .render
            .master_scene_referred_file_sha256_matches_report,
        Some(false)
    );
    assert!(
        scanstitch::validation::delivery_artifact_integrity_issues(&substituted_master.render)
            .contains(&"master_scene_referred_sha256_mismatch")
    );
    assert!(
        !scanstitch::validation::final_delivery_evidence_is_reviewable(&substituted_master.render)
    );
    std::fs::remove_file(&master_path).unwrap();
    std::fs::rename(&held_valid_master_path, &master_path).unwrap();

    let held_master_path = output_dir.join("master_scene_referred.held");
    std::fs::rename(&master_path, &held_master_path).unwrap();
    std::fs::copy(output_dir.join("output.tiff"), &master_path).unwrap();
    let wrong_master =
        scanstitch::validation::summarize_report("wrong-master", &claimed_reviewable_report);
    assert_eq!(
        wrong_master
            .render
            .master_scene_referred_file_matches_report,
        Some(false)
    );
    assert!(
        scanstitch::validation::delivery_artifact_integrity_issues(&wrong_master.render)
            .contains(&"master_scene_referred_artifact_mismatch")
    );
}

#[test]
fn test_positive_pipeline_preserves_luminance_order_and_skips_negative_phases() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let positive = synthetic::gradient_image(120, 200, [1800, 2000, 2200], [14500, 15000, 15500]);
    let path1 = input_dir.join("positive1.tiff");
    let path2 = input_dir.join("positive2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&positive, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&positive, &path2).unwrap();

    let calibration_path = tmp.path().join("positive-profile.json");
    std::fs::write(&calibration_path, valid_calibration_profile_json()).unwrap();
    let output_dir = tmp.path().join("positive-output");
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: output_dir.clone(),
        calibration_profile: Some(calibration_path),
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Calibrated,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Positive,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: true,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 16,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let border_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "border_removal")
        .expect("border phase");
    assert_eq!(border_phase.confidence, 0.0);
    assert_eq!(
        border_phase.metrics["border_decision_status"],
        "no_component_supported_applied_crop"
    );
    assert_eq!(border_phase.metrics["border_evidence_evaluated"], true);
    assert_eq!(
        border_phase.metrics["component1"]["border_decision_status"],
        "no_crop_no_convincing_dead_zone",
        "unexpected positive-gradient edge evidence: {}",
        border_phase.metrics["component1"]
    );
    assert_eq!(
        border_phase.metrics["component1"]["applied_crop_evidence_supported"],
        false
    );
    let classify_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "base_detect_classify")
        .expect("classification phase");
    assert_eq!(classify_phase.confidence, 0.0);
    assert_eq!(
        classify_phase.metrics["classification_status"],
        "not_applicable_positive_input"
    );
    assert_eq!(
        classify_phase.metrics["classification_evidence_evaluated"],
        false
    );
    assert_eq!(
        classify_phase.metrics["component1"]["classification_status"],
        "not_applicable_positive_input"
    );
    assert_eq!(
        classify_phase.metrics["component1"]["confidence_basis"],
        "not_evaluated"
    );
    assert_eq!(
        classify_phase.metrics["component1"]["base_color_source"],
        "not_required_positive_input"
    );
    assert_eq!(
        classify_phase.metrics["component1"]["base_measurement_stage"],
        "not_required_positive_input"
    );
    assert!(classify_phase
        .warnings
        .iter()
        .all(|warning| !warning.contains("base confidence is low")));
    let density_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "density_inversion")
        .expect("density phase");
    assert_eq!(density_phase.confidence, 0.0);
    assert_eq!(density_phase.metrics["skipped"], true);
    assert_eq!(
        density_phase.metrics["operation_status"],
        "not_applicable_positive_input"
    );
    assert_eq!(density_phase.metrics["evidence_evaluated"], false);
    assert_eq!(
        density_phase.metrics["negative_response_model_evidence_evaluated"],
        false
    );
    assert!(density_phase.metrics["film_base_evidence_confidence"].is_null());
    assert!(density_phase.metrics["negative_response_model_confidence"].is_null());
    assert_eq!(density_phase.metrics["confidence_basis"], "not_evaluated");
    assert_eq!(density_phase.metrics["input_mode"], "positive");
    assert!(density_phase.metrics.get("base_color").is_none());
    let ica_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "fastica")
        .expect("fastica phase");
    assert_eq!(ica_phase.confidence, 0.0);
    assert_eq!(ica_phase.metrics["skipped"], true);
    assert_eq!(
        ica_phase.metrics["operation_status"],
        "not_applicable_positive_input"
    );
    assert_eq!(ica_phase.metrics["evidence_evaluated"], false);
    assert_eq!(ica_phase.metrics["confidence_basis"], "not_evaluated");
    assert_eq!(ica_phase.metrics["input_mode"], "positive");
    let colorspace_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "colorspace_mapping")
        .expect("colorspace phase");
    assert_eq!(colorspace_phase.metrics["input_mode"], "positive");
    assert_eq!(
        colorspace_phase.metrics["render_input_source"],
        "positive_scan_rgb"
    );
    assert_eq!(
        colorspace_phase.metrics["render_input_reason"],
        "already-positive input normalized without density inversion"
    );
    assert_eq!(
        colorspace_phase.metrics["negative_reconstruction_confidence_status"],
        "not_applicable_positive_input"
    );
    assert_eq!(
        colorspace_phase.metrics["negative_reconstruction_evidence_evaluated"],
        false
    );
    assert!(colorspace_phase.metrics["negative_reconstruction_evidence_confidence"].is_null());
    assert_eq!(
        colorspace_phase.metrics["confidence_limited_by_negative_reconstruction_evidence"],
        false
    );
    assert!(output_dir.join("phase3_positive_passthrough.tiff").exists());
    assert!(
        !output_dir.join("phase3_linear_division.tiff").exists(),
        "positive mode must not write the base-division artifact"
    );
    let tone_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "tone_mapping")
        .expect("tone phase");
    assert_eq!(tone_phase.metrics["tone_fit_policy"], "positive_scan_rgb");
    let working_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "working_image_select")
        .expect("working image phase");
    assert_eq!(
        working_phase.metrics["base_estimate_source"],
        "not_applicable_positive_input"
    );
    assert_eq!(
        working_phase.metrics["positive_input_inspection"]["likely_negative_like"], false,
        "ordinary positive RGB fixture should not trip the orange-mask warning"
    );
    assert!(
        tone_phase.metrics["auto_exposure_ev"].as_f64().unwrap() <= 0.0,
        "positive auto exposure should not brighten already-positive scans"
    );
    assert!(
        tone_phase.metrics["mapped_linear_percentiles"][1]
            .as_f64()
            .unwrap()
            < 0.50,
        "positive tone fit should avoid an overly light median: {}",
        tone_phase.metrics["mapped_linear_percentiles"]
    );

    let rendered = scanstitch::tiff_io::load_tiff_u16(&output_dir.join("output.tiff"), 16)
        .unwrap()
        .image;
    let left = mean_luminance_region(&rendered, 10, 50);
    let right = mean_luminance_region(&rendered, 150, 190);
    assert!(
        right > left,
        "positive render should remain non-inverted: left={left} right={right}"
    );
}

#[test]
fn test_positive_pipeline_applies_matching_embedded_icc_profile() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();
    let width = 160usize;
    let height = 96usize;
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let horizontal = x as f64 / (width - 1) as f64;
            let vertical = y as f64 / (height - 1) as f64;
            pixels.push(((0.10 + 0.75 * horizontal) * 65_535.0).round() as u16);
            pixels.push(((0.12 + 0.68 * vertical) * 65_535.0).round() as u16);
            pixels.push(((0.16 + 0.58 * (1.0 - horizontal)) * 65_535.0).round() as u16);
        }
    }
    let srgb_profile = moxcms::ColorProfile::new_srgb().encode().unwrap();
    let path1 = input_dir.join("profiled1.tiff");
    let path2 = input_dir.join("profiled2.tiff");
    write_rgb16_tiff_with_icc(&path1, width as u32, height as u32, &pixels, &srgb_profile).unwrap();
    write_rgb16_tiff_with_icc(&path2, width as u32, height as u32, &pixels, &srgb_profile).unwrap();

    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: tmp.path().join("profiled-output"),
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Positive,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 16,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let load = report
        .phases
        .iter()
        .find(|phase| phase.name == "load")
        .expect("load phase");
    assert_eq!(
        load.metrics["component1_decode"]["source_icc_profile"]["status"],
        "valid_rgb_profile"
    );
    assert_eq!(load.confidence, 1.0);
    assert_eq!(
        load.metrics["decode_fidelity_status"],
        "all_components_full_declared_decode_fidelity"
    );
    assert_eq!(
        load.metrics["component1_decode"]["decode_fidelity"]["status"],
        "full_declared_decode_fidelity"
    );
    let colorspace = report
        .phases
        .iter()
        .find(|phase| phase.name == "colorspace_mapping")
        .expect("colorspace phase");
    assert_eq!(
        colorspace.metrics["input_color_transform"]["status"],
        "applied"
    );
    assert_eq!(
        colorspace.metrics["input_color_transform"]["destination_profile"],
        "linear_prophoto_rgb_d50"
    );
    assert_eq!(
        colorspace.metrics["mapping_strategy"],
        "embedded_icc_to_linear_prophoto"
    );
    assert_eq!(colorspace.metrics["candidate_risk"], "safe");
    assert_eq!(colorspace.confidence, 1.0);
    assert_eq!(
        colorspace.metrics["color_confidence_status"],
        "trusted_selected_mapping"
    );
    assert_eq!(
        colorspace.metrics["input_domain"],
        "linear_prophoto_rgb_d50_from_embedded_icc"
    );
    assert!(colorspace.metrics["render_input_reason"]
        .as_str()
        .unwrap()
        .contains("embedded ICC"));
    let working = report
        .phases
        .iter()
        .find(|phase| phase.name == "working_image_select")
        .expect("working phase");
    assert_eq!(working.confidence, 1.0);
    assert_eq!(
        working.metrics["input_mode_suitability_status"],
        "accepted_positive_input"
    );
    assert_eq!(working.metrics["input_mode_review_required"], false);
    let tone = report
        .phases
        .iter()
        .find(|phase| phase.name == "tone_mapping")
        .expect("tone phase");
    assert_eq!(tone.confidence, 1.0);
    assert_eq!(
        tone.metrics["tone_confidence_status"],
        "supported_tone_and_optional_grain_evidence"
    );
    assert_eq!(tone.metrics["tone_output_evidence_evaluated"], true);
    assert_eq!(tone.metrics["tone_output_evidence_confidence"], 1.0);
    assert_eq!(
        tone.metrics["tone_output_confidence_status"],
        "supported_render_tonal_distribution"
    );
    assert_eq!(tone.metrics["tone_output_review_required"], false);
    assert_eq!(
        tone.metrics["confidence_limited_by_tone_output_evidence"],
        false
    );
    let save = report
        .phases
        .iter()
        .find(|phase| phase.name == "save")
        .expect("save phase");
    assert_eq!(save.metrics["color_trust_state"], "trusted");
    assert_eq!(save.metrics["input_mode_review_required"], false);
    assert_eq!(save.metrics["tone_output_review_required"], false);
    assert_eq!(
        save.metrics["tone_output_confidence_status"],
        "supported_render_tonal_distribution"
    );
    assert_eq!(save.metrics["render_review_status"], "reviewable");
    assert_eq!(save.metrics["render_reviewable"], true);
    assert_eq!(save.confidence, 1.0);
    assert_eq!(
        save.metrics["delivery_confidence_status"],
        "reviewable_delivery"
    );
}

#[test]
fn test_positive_pipeline_blocks_negative_like_mode_but_accepts_warm_positive() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let negative_like =
        synthetic::gradient_image(48, 64, [22000, 14000, 5000], [36000, 22000, 9000]);
    let path1 = input_dir.join("orange1.tiff");
    let path2 = input_dir.join("orange2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&negative_like, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&negative_like, &path2).unwrap();

    let output_dir = tmp.path().join("positive-output");
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: output_dir.clone(),
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Positive,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 16,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let working_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "working_image_select")
        .expect("working image phase");
    assert_eq!(
        working_phase.metrics["positive_input_inspection"]["likely_negative_like"], true,
        "orange negative-like input should be surfaced when positive mode is selected"
    );
    assert_eq!(working_phase.confidence, 0.0);
    assert_eq!(
        working_phase.metrics["input_mode_suitability_status"],
        "review_required_negative_like_positive_input"
    );
    assert_eq!(
        working_phase.metrics["input_mode_suitability_evaluated"],
        true
    );
    assert_eq!(working_phase.metrics["input_mode_review_required"], true);
    assert_eq!(
        working_phase.metrics["confidence_basis"],
        "positive_input_channel_ratio_and_orange_mask_inspection"
    );
    assert!(
        working_phase
            .warnings
            .iter()
            .any(|warning| warning.contains("orange-mask-like channel bias")),
        "expected positive-mode warning for likely negative-like input: {:?}",
        working_phase.warnings
    );
    let density_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "density_inversion")
        .expect("density phase");
    assert_eq!(
        density_phase.metrics["skipped"], true,
        "warning should not silently switch positive mode into negative processing"
    );
    let save_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "save")
        .expect("save phase");
    assert_eq!(save_phase.metrics["color_trust_state"], "review_required");
    assert_eq!(
        save_phase.metrics["render_review_status"],
        "review_required_input_mode"
    );
    assert_eq!(save_phase.metrics["input_mode_review_required"], true);
    assert!(save_phase.metrics["input_mode_review_reason"]
        .as_str()
        .is_some_and(|reason| reason.contains("orange-mask-like channel bias")));
    assert_eq!(save_phase.metrics["render_reviewable"], false);

    let warm_path = input_dir.join("warm-positive.tiff");
    let warm_positive = synthetic::constant_image(48, 64, [27_415, 22_332, 20_330]);
    scanstitch::tiff_io::save_tiff_u16(&warm_positive, &warm_path).unwrap();
    let mut warm_cli = positive_fast_cli(vec![warm_path], tmp.path().join("warm-output"));
    warm_cli.bit_depth = 16;
    let warm_report = scanstitch::pipeline::run(&warm_cli).unwrap();
    let warm_working = warm_report
        .phases
        .iter()
        .find(|phase| phase.name == "working_image_select")
        .expect("warm positive working phase");
    assert_eq!(warm_working.confidence, 1.0);
    assert_eq!(
        warm_working.metrics["input_mode_suitability_status"],
        "accepted_warm_positive_input"
    );
    assert_eq!(warm_working.metrics["input_mode_review_required"], false);
    let warm_save = warm_report
        .phases
        .iter()
        .find(|phase| phase.name == "save")
        .expect("warm positive save phase");
    assert_eq!(warm_save.metrics["input_mode_review_required"], false);
}

#[test]
fn test_scanstitch_cli_rejects_positive_ica_render_input() {
    let tmp = TempDir::new().unwrap();
    let path1 = tmp.path().join("positive1.tiff");
    let path2 = tmp.path().join("positive2.tiff");
    let img = synthetic::constant_image(16, 16, [12000, 12000, 12000]);
    scanstitch::tiff_io::save_tiff_u16(&img, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&img, &path2).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch"))
        .arg(&path1)
        .arg(&path2)
        .arg("--input-mode")
        .arg("positive")
        .arg("--render-input")
        .arg("ica")
        .arg("--output-dir")
        .arg(tmp.path().join("out"))
        .output()
        .expect("run scanstitch");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--input-mode positive cannot be used with --render-input ica"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn test_scanstitch_cli_require_reviewable_fails_after_retaining_blocked_artifacts() {
    let tmp = TempDir::new().unwrap();
    let input = tmp.path().join("unprofiled-positive.tiff");
    let (width, height) = (64u32, 48u32);
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.extend_from_slice(&[
                (32 + 190 * x / (width - 1)) as u8,
                (28 + 180 * y / (height - 1)) as u8,
                (48 + 140 * (x + y) / (width + height - 2)) as u8,
            ]);
        }
    }
    write_rgb8_tiff(&input, width, height, &pixels).unwrap();
    let output_dir = tmp.path().join("output");

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch"))
        .arg(&input)
        .arg("--input-mode")
        .arg("positive")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--force-no-stitch")
        .arg("--technical-white-balance")
        .arg("off")
        .arg("--require-reviewable")
        .arg("--output-dir")
        .arg(&output_dir)
        .output()
        .expect("run gated scanstitch render");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--require-reviewable rejected the completed render")
            && stderr.contains("report and artifacts were retained"),
        "unexpected stderr: {stderr}"
    );
    assert!(
        !stderr.contains("saved delivery artifact integrity failed"),
        "the retained TIFF should pass independent delivery integrity even though its color decision is non-reviewable: {stderr}"
    );
    assert!(output_dir.join("output.tiff").is_file());
    let report_path = output_dir.join("report.json");
    assert!(report_path.is_file());
    let report: scanstitch::report::PipelineReport =
        serde_json::from_str(&std::fs::read_to_string(report_path).unwrap()).unwrap();
    let colorspace = report
        .phases
        .iter()
        .find(|phase| phase.name == "colorspace_mapping")
        .expect("colorspace phase");
    assert_eq!(colorspace.confidence, 0.0);
    assert_eq!(
        colorspace.metrics["color_confidence_status"],
        "review_required_selected_mapping"
    );
    let white_balance = report
        .phases
        .iter()
        .find(|phase| phase.name == "white_balance")
        .expect("white-balance phase");
    assert_eq!(white_balance.confidence, 0.0);
    assert_eq!(
        white_balance.metrics["technical_decision_status"],
        "not_applied_disabled"
    );
    assert_eq!(white_balance.metrics["technical_evidence_evaluated"], false);
    assert_eq!(
        white_balance.metrics["confidence_basis"],
        "not_applicable_disabled"
    );
    let tone = report
        .phases
        .iter()
        .find(|phase| phase.name == "tone_mapping")
        .expect("tone phase");
    assert_eq!(tone.confidence, 0.0);
    assert_eq!(
        tone.metrics["tone_confidence_status"],
        "review_required_upstream_color"
    );
    let save = report
        .phases
        .iter()
        .find(|phase| phase.name == "save")
        .expect("final save phase");
    assert_eq!(
        save.metrics["render_review_status"],
        "review_required_color"
    );
    assert_eq!(save.metrics["render_reviewable"], false);
    assert_eq!(save.confidence, 0.0);
    assert_eq!(
        save.metrics["delivery_confidence_status"],
        "diagnostic_delivery_requires_review"
    );
}

#[test]
fn test_pipeline_threads_stitch_runtime_flags_into_report() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let full = synthetic::gradient_image(180, 720, [3000, 5000, 8000], [10000, 8000, 4000]);
    let overlap = 100usize;
    let split_point = 720 / 2 + overlap / 2;
    let comp1 = full.slice(s![.., 0..split_point, ..]).to_owned();
    let comp2 = full
        .slice(s![.., (split_point - overlap)..720, ..])
        .to_owned();

    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp1, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp2, &path2).unwrap();

    let output_dir = tmp.path().join("output");
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: output_dir.clone(),
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: true,
        force_no_stitch: false,
        transform: "homography".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: true,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let stitch_phase = report
        .phases
        .iter()
        .find(|p| p.name == "stitch")
        .expect("stitch phase present");

    assert_eq!(stitch_phase.metrics["requested_transform"], "homography");
    assert_eq!(stitch_phase.metrics["opencv_requested"], true);
    assert_eq!(stitch_phase.metrics["opencv_available"], false);
    let hypotheses = stitch_phase.metrics["hypotheses"]
        .as_array()
        .expect("stitch hypotheses should be reported");
    assert!(hypotheses
        .iter()
        .all(|hyp| hyp["search"]["selection_reason"].is_string()));
    assert!(hypotheses
        .iter()
        .all(|hyp| hyp["search"]["evaluated_candidates"].is_array()));
    assert!(hypotheses.iter().all(|hyp| {
        hyp["search"]["top_candidates"]
            .as_array()
            .expect("top candidates should be reported")
            .iter()
            .all(|candidate| {
                candidate["correspondence_score"].is_number()
                    && candidate["prior_weight"].is_number()
            })
    }));
    assert!(hypotheses
        .iter()
        .all(|hyp| hyp["validation"]["prior_weight"].is_number()));
    assert!(
        stitch_phase
            .warnings
            .iter()
            .any(|w| w.contains("built without the `use-opencv` feature")),
        "expected warning about unavailable OpenCV backend"
    );
}

#[test]
fn test_ambiguous_pair_still_requests_stitch_scoring() {
    let analysis = scanstitch::frame_classify::FrameAnalysis {
        class: scanstitch::frame_classify::FrameClass::Ambiguous,
        confidence: 0.2,
        left_edge_confidence: 0.1,
        right_edge_confidence: 0.1,
        content_span: (20, 80),
        content_fraction: 0.6,
        activity_score: 0.3,
        component_score: 0.4,
        internal_base_regions: Vec::new(),
    };

    let decision =
        scanstitch::frame_classify::decide_stitch_attempt(&analysis, &analysis, false, false);

    assert!(decision.should_score());
    assert_eq!(decision.reason, "ambiguous_pair_requires_scoring");
}

#[test]
fn test_low_base_confidence_emits_working_image_warning() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp1 = synthetic::constant_image(120, 240, [6000, 6000, 6000]);
    let comp2 = synthetic::constant_image(120, 240, [6500, 6500, 6500]);

    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp1, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp2, &path2).unwrap();

    let output_dir = tmp.path().join("output");
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: output_dir.clone(),
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let working_phase = report
        .phases
        .iter()
        .find(|p| p.name == "working_image_select")
        .expect("working_image_select phase");

    assert!(
        working_phase.confidence < 0.3,
        "expected degraded working-image base confidence"
    );
    assert!(
        working_phase
            .warnings
            .iter()
            .any(|warning| warning.contains("base confidence")),
        "expected low-base-confidence warning"
    );
    assert!(
        working_phase.metrics["raw_base_proxy_confidence"].is_number(),
        "expected fallback proxy confidence diagnostic"
    );
    assert!(
        working_phase.metrics["raw_base_support_fraction"].is_number(),
        "expected fallback support fraction diagnostic"
    );

    let ica_phase = report
        .phases
        .iter()
        .find(|p| p.name == "fastica")
        .expect("fastica phase");
    assert!(
        ica_phase.confidence < 1.0,
        "expected downstream ICA confidence cap"
    );
    assert!(
        ica_phase.metrics["input_base_confidence"].is_number(),
        "expected base-confidence diagnostic to be recorded"
    );
    assert_eq!(
        ica_phase.metrics["confidence_limited_by_base_estimate"], false,
        "zero physical-separation confidence cannot also claim that the base estimate lowered it"
    );

    let colorspace_phase = report
        .phases
        .iter()
        .find(|p| p.name == "colorspace_mapping")
        .expect("colorspace_mapping phase");
    assert!(
        colorspace_phase.confidence < 1.0,
        "expected downstream colorspace confidence cap"
    );
    assert_eq!(
        colorspace_phase.metrics["direct_density_auto_selection_eligible"],
        false
    );
    assert_eq!(
        colorspace_phase.metrics["render_input_source"], "fastica_separated_transmittance",
        "fallback film-base evidence must not automatically select direct density"
    );
    assert!(
        colorspace_phase.metrics["direct_density_auto_selection_rejection_reason"]
            .as_str()
            .is_some_and(|reason| reason.contains("film-base confidence"))
    );
    assert!(colorspace_phase
        .warnings
        .iter()
        .any(|warning| warning.contains("automatic direct-density selection is disabled")));

    let save_phase = report
        .phases
        .iter()
        .find(|p| p.name == "save")
        .expect("save phase");
    assert_eq!(
        save_phase.metrics["render_review_status"],
        "blocked_low_base_confidence"
    );
    assert_eq!(save_phase.metrics["render_reviewable"], false);
    assert!(
        save_phase
            .warnings
            .iter()
            .any(|warning| warning.contains("render is not reviewable")),
        "expected save phase to block review on fallback-quality base estimates"
    );
}

#[test]
fn test_pipeline_retains_rebate_measurement_that_final_crop_removes() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let mut comp = synthetic::gradient_image(160, 320, [4000, 4000, 4000], [6500, 6500, 6500]);
    for y in 0..10 {
        for x in 0..320 {
            for c in 0..3 {
                comp[[y, x, c]] = 100;
                comp[[159 - y, x, c]] = 100;
            }
        }
    }
    for y in 10..14 {
        for x in 0..320 {
            comp[[y, x, 0]] = 12_000;
            comp[[y, x, 1]] = 7000;
            comp[[y, x, 2]] = 3000;
            let bottom = 159 - y;
            comp[[bottom, x, 0]] = 12_000;
            comp[[bottom, x, 1]] = 7000;
            comp[[bottom, x, 2]] = 3000;
        }
    }

    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: tmp.path().join("output"),
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::FilmFaithful,
        quality_mode: scanstitch::cli::QualityMode::Fast,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let border = report
        .phases
        .iter()
        .find(|phase| phase.name == "border_removal")
        .expect("border phase");
    assert!(border.confidence > 0.0);
    assert_eq!(
        border.metrics["border_decision_status"],
        "all_components_supported_applied_crop"
    );
    assert_eq!(border.metrics["supported_component_count"], 2);
    assert_eq!(
        border.metrics["component1"]["border_decision_status"],
        "crop_applied_from_measured_edge_evidence"
    );
    assert_eq!(
        border.metrics["component1"]["applied_crop_evidence_supported"],
        true
    );
    assert!(
        border.metrics["component1"]["top_removed"]
            .as_u64()
            .unwrap()
            >= 14
    );
    assert!(
        border.metrics["component1"]["bottom_removed"]
            .as_u64()
            .unwrap()
            >= 14
    );

    let component_base = report
        .phases
        .iter()
        .find(|phase| phase.name == "base_detect_classify")
        .expect("component base phase");
    assert_eq!(
        component_base.metrics["component1"]["base_color_source"],
        "removed_border_rebate_band"
    );
    assert_eq!(
        component_base.metrics["component1"]["base_measurement_stage"],
        "pre_crop_rebate_measurement"
    );

    let working = report
        .phases
        .iter()
        .find(|phase| phase.name == "working_image_select")
        .expect("working phase");
    assert_eq!(
        working.metrics["base_estimate_source"],
        "pre_crop_rebate_measurement"
    );
    assert!(working.confidence >= 0.70);
}

#[test]
fn test_manual_base_color_override_unblocks_base_but_not_negative_response_review_gate() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp = synthetic::constant_image(120, 240, [6000, 6000, 6000]);

    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let output_dir = tmp.path().join("output");
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir,
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: Some("6500,6500,6500".to_string()),
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let working_phase = report
        .phases
        .iter()
        .find(|p| p.name == "working_image_select")
        .expect("working_image_select phase");
    assert_eq!(
        working_phase.metrics["base_estimate_source"],
        "manual_base_color_override"
    );
    assert_eq!(working_phase.metrics["base_color_override_applied"], true);
    assert_eq!(working_phase.confidence, 1.0);

    let density_phase = report
        .phases
        .iter()
        .find(|p| p.name == "density_inversion")
        .expect("density_inversion phase");
    assert_eq!(density_phase.confidence, 0.0);
    assert_eq!(
        density_phase.metrics["operation_status"],
        "review_required_unmeasured_unit_slope_response"
    );
    assert_eq!(density_phase.metrics["evidence_evaluated"], false);
    assert_eq!(
        density_phase.metrics["negative_response_model_evidence_evaluated"],
        false
    );
    assert_eq!(density_phase.metrics["film_base_evidence_confidence"], 1.0);
    assert_eq!(
        density_phase.metrics["negative_response_model_confidence"],
        0.0
    );
    assert_eq!(
        density_phase.metrics["confidence_basis"],
        "minimum_film_base_and_negative_response_model_evidence"
    );
    assert_eq!(
        density_phase.metrics["direct_density_response_model"]["model"],
        "shared_unit_density_slope"
    );
    assert!(density_phase.warnings.iter().any(|warning| {
        warning
            .contains("successful arithmetic inversion does not establish film-response accuracy")
    }));

    let colorspace_phase = report
        .phases
        .iter()
        .find(|p| p.name == "colorspace_mapping")
        .expect("colorspace_mapping phase");
    assert_eq!(colorspace_phase.confidence, 0.0);
    assert_eq!(
        colorspace_phase.metrics["color_confidence_status"],
        "review_required_selected_mapping"
    );
    assert_eq!(
        colorspace_phase.metrics["negative_reconstruction_confidence_status"],
        "review_required_unmeasured_negative_response"
    );
    assert_eq!(
        colorspace_phase.metrics["negative_reconstruction_evidence_evaluated"],
        false
    );
    assert_eq!(
        colorspace_phase.metrics["negative_reconstruction_evidence_confidence"],
        0.0
    );
    assert_eq!(
        colorspace_phase.metrics["confidence_limited_by_negative_reconstruction_evidence"],
        false
    );

    let tone_phase = report
        .phases
        .iter()
        .find(|p| p.name == "tone_mapping")
        .expect("tone_mapping phase");
    assert_eq!(tone_phase.confidence, 0.0);
    assert_eq!(tone_phase.metrics["upstream_color_confidence"], 0.0);
    assert_eq!(
        tone_phase.metrics["tone_confidence_status"],
        "review_required_upstream_color"
    );

    let save_phase = report
        .phases
        .iter()
        .find(|p| p.name == "save")
        .expect("save phase");
    assert_eq!(
        save_phase.metrics["render_review_status"],
        "review_required_negative_response"
    );
    assert_eq!(save_phase.metrics["render_reviewable"], false);
    assert_eq!(
        save_phase.metrics["negative_response_review_required"],
        true
    );
    let negative_response_reason = save_phase.metrics["negative_response_review_reason"]
        .as_str()
        .expect("negative response review reason");
    assert!(
        negative_response_reason.contains("no measured film characteristic curve")
            || negative_response_reason.contains("held-out"),
        "unexpected negative-response review reason: {negative_response_reason}"
    );
    assert_eq!(
        save_phase.metrics["color_trust_state"],
        "limited_weak_neutral"
    );
}

#[test]
fn test_pipeline_reports_valid_base_transmittance_for_8bit_tiff_inputs() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let width = 180usize;
    let height = 90usize;
    let mut pixels = Vec::<u8>::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            let rgb = if x < 18 || x >= width - 18 {
                [225u8, 182u8, 120u8]
            } else {
                [100u8, 78u8, 66u8]
            };
            pixels.extend_from_slice(&rgb);
        }
    }

    let path1 = input_dir.join("comp1_rgb8.tiff");
    let path2 = input_dir.join("comp2_rgb8.tiff");
    write_rgb8_tiff(&path1, width as u32, height as u32, &pixels).unwrap();
    write_rgb8_tiff(&path2, width as u32, height as u32, &pixels).unwrap();

    let output_dir = tmp.path().join("output");
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: output_dir.clone(),
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();

    let load_phase = report
        .phases
        .iter()
        .find(|p| p.name == "load")
        .expect("load phase");
    assert_eq!(
        load_phase.metrics["component1_decode"]["source_bits_per_sample"],
        8
    );
    assert_eq!(
        load_phase.metrics["component1_decode"]["range_transform"],
        "upscaled_to_working_bit_depth"
    );
    assert!(
        (load_phase.confidence - 8.0 / 14.0).abs() <= 1e-12,
        "8-bit source expanded to 14-bit work must retain a truthful precision limit: {}",
        load_phase.confidence
    );
    assert_eq!(
        load_phase.metrics["decode_fidelity_status"],
        "no_component_full_declared_decode_fidelity"
    );
    assert_eq!(
        load_phase.metrics["component1_decode"]["decode_fidelity"]["status"],
        "limited_declared_decode_fidelity"
    );
    assert!(
        load_phase.metrics["component1_decode"]["decode_fidelity"]["limiters"]
            .as_array()
            .is_some_and(|limiters| limiters
                .iter()
                .any(|limiter| limiter == "source_precision_upscaled_without_added_information"))
    );

    let density_phase = report
        .phases
        .iter()
        .find(|p| p.name == "density_inversion")
        .expect("density phase");
    let base_transmittance = density_phase.metrics["base_transmittance"]
        .as_array()
        .expect("base_transmittance array");
    assert!(
        base_transmittance
            .iter()
            .all(|value| value.as_f64().unwrap_or(2.0) <= 1.0),
        "base transmittance should stay within [0, 1] after load normalization"
    );
}

#[test]
fn test_working_image_phase_reports_raw_and_resolved_base_diagnostics() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp1 =
        synthetic::film_negative_image(160, 320, 0, 24, [12000, 7000, 3000], [5000, 4000, 3500]);
    let comp2 =
        synthetic::film_negative_image(160, 320, 0, 24, [12000, 7000, 3000], [5200, 4100, 3400]);

    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp1, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp2, &path2).unwrap();

    let output_dir = tmp.path().join("output");
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir,
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let working_phase = report
        .phases
        .iter()
        .find(|p| p.name == "working_image_select")
        .expect("working_image_select phase");

    assert!(
        working_phase.metrics["base_estimate_source"].is_string(),
        "expected working-image base source diagnostic"
    );
    assert!(
        working_phase.metrics["base_estimate_reason"].is_string(),
        "expected working-image base reason diagnostic"
    );
    assert!(
        working_phase.metrics["raw_base_color"].is_array(),
        "expected raw working-image base color diagnostic"
    );
    assert!(
        working_phase.metrics["raw_base_confidence"].is_number(),
        "expected raw working-image base confidence diagnostic"
    );
}

#[test]
fn test_pipeline_reports_colorspace_headroom_diagnostics() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let mut comp =
        synthetic::film_negative_image(160, 320, 0, 24, [12000, 7000, 3000], [5000, 4000, 3500]);
    for y in 0..160 {
        for x in 24..296 {
            let pixel = if x < 96 {
                [4200, 10800, 9200]
            } else if x < 168 {
                [9800, 4200, 8800]
            } else if x < 240 {
                [11200, 9400, 3600]
            } else {
                [7000, 6800, 6600]
            };
            for c in 0..3 {
                comp[[y, x, c]] = pixel[c];
            }
        }
    }

    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let output_dir = tmp.path().join("output");
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: output_dir.clone(),
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let colorspace_phase = report
        .phases
        .iter()
        .find(|p| p.name == "colorspace_mapping")
        .expect("colorspace_mapping phase");

    assert!(
        colorspace_phase.metrics["highlight_percentile"].is_number(),
        "expected highlight percentile diagnostic"
    );
    assert!(
        colorspace_phase.metrics["pre_scale_channel_high_percentile"].is_array(),
        "expected pre-scale channel percentile diagnostic"
    );
    assert!(
        colorspace_phase.metrics["pre_scale_clipped_high_ratio"].is_array(),
        "expected pre-scale clipped ratio diagnostic"
    );
    assert!(
        colorspace_phase.metrics["post_scale_clipped_high_ratio"].is_array(),
        "expected post-scale clipped ratio diagnostic"
    );
    assert!(
        colorspace_phase.metrics["pre_scale_clipped_low_ratio"].is_array(),
        "expected pre-scale low-clipped ratio diagnostic"
    );
    assert!(
        colorspace_phase.metrics["post_scale_clipped_low_ratio"].is_array(),
        "expected post-scale low-clipped ratio diagnostic"
    );
    assert!(
        colorspace_phase.metrics["gamut_fallback_used"].is_boolean(),
        "expected gamut fallback diagnostic"
    );
    assert!(
        colorspace_phase.metrics["mapping_strategy"].is_string(),
        "expected colorspace mapping strategy diagnostic"
    );
    assert_eq!(colorspace_phase.metrics["color_mode"], "auto");
    assert!(
        colorspace_phase.metrics["candidate_scores"].is_array(),
        "expected colorspace candidate score diagnostics"
    );
    assert!(
        colorspace_phase.metrics["candidate_quality_scores"].is_array(),
        "expected colorspace candidate quality score diagnostics"
    );
    assert_eq!(
        colorspace_phase.metrics["candidate_score_order"],
        "lower_is_better"
    );
    assert!(
        colorspace_phase.metrics["candidate_acceptance"].is_array(),
        "expected per-candidate colorspace acceptance diagnostics"
    );
    assert!(
        colorspace_phase.metrics["selected_candidate"].is_string(),
        "expected selected colorspace candidate diagnostic"
    );
    assert!(
        colorspace_phase.metrics["selected_candidate_rank"].is_number(),
        "expected selected colorspace candidate rank diagnostic"
    );
    assert!(
        colorspace_phase.metrics["selected_candidate_score"].is_number(),
        "expected selected colorspace candidate score diagnostic"
    );
    assert!(
        colorspace_phase.metrics["selected_quality_score"].is_number(),
        "expected selected colorspace quality score diagnostic"
    );
    assert!(
        colorspace_phase.metrics["technical_safety_score"].is_number(),
        "expected selected colorspace technical safety score diagnostic"
    );
    assert!(
        colorspace_phase.metrics["color_fidelity_score"].is_number(),
        "expected selected colorspace fidelity score diagnostic"
    );
    assert!(
        colorspace_phase.metrics["quality_components"].is_object(),
        "expected selected colorspace quality component diagnostics"
    );
    assert!(
        colorspace_phase.metrics["selected_quality_components"].is_object(),
        "expected selected colorspace quality component alias"
    );
    assert!(
        colorspace_phase.metrics["quality_components"]["rendered_tone_penalty"].is_number(),
        "expected rendered-tone score component"
    );
    assert!(
        colorspace_phase.metrics["quality_components"]["tone_chroma_cleanup_penalty"].is_number(),
        "expected tone chroma cleanup risk score component"
    );
    let candidate_quality_scores = colorspace_phase.metrics["candidate_quality_scores"]
        .as_array()
        .expect("candidate quality scores");
    assert!(
        candidate_quality_scores
            .iter()
            .any(|candidate| candidate["rendered_tone_quality"].is_object()),
        "expected per-candidate rendered-tone diagnostics"
    );
    assert!(
        colorspace_phase.metrics["candidate_risk"].is_string(),
        "expected candidate risk diagnostic"
    );
    assert!(
        colorspace_phase.metrics["selected_runner_up_quality_delta"].is_number()
            || colorspace_phase.metrics["selected_runner_up_quality_delta"].is_null(),
        "expected selected-vs-runner-up quality delta diagnostic"
    );
    assert!(
        colorspace_phase.metrics["neutral_estimate_quality"].is_object(),
        "expected neutral estimate quality diagnostic"
    );
    assert!(
        colorspace_phase.metrics["neutral_sample_rejections"].is_object(),
        "expected neutral sample rejection diagnostic"
    );
    assert!(
        colorspace_phase.metrics["dominant_anchor_sample_rejections"].is_object(),
        "expected dominant anchor sample rejection diagnostic"
    );
    assert!(
        colorspace_phase.metrics["dominant_anchor_bands"].is_array(),
        "expected dominant anchor luminance-band diagnostic"
    );
    assert!(
        colorspace_phase.metrics["dominant_anchor_quality"].is_object(),
        "expected dominant anchor quality/stability diagnostic"
    );
    assert!(
        colorspace_phase.metrics["neutral_trim_before_after"].is_object(),
        "expected neutral trim before/after diagnostic"
    );
    assert!(
        colorspace_phase.metrics["calibration_acceptance"].is_object(),
        "expected calibration acceptance diagnostic"
    );
    assert!(
        colorspace_phase.metrics["color_processing_substeps"].is_object(),
        "expected explicit colorspace processing substeps"
    );
    assert!(
        colorspace_phase.metrics["selection_rejections"].is_array(),
        "expected colorspace selection rejection diagnostics"
    );
    assert!(
        colorspace_phase.metrics["neutral_trim_scale"].is_array(),
        "expected neutral trim scale diagnostic"
    );
    assert!(
        colorspace_phase.metrics["neutral_trim_applied"].is_boolean(),
        "expected neutral trim application diagnostic"
    );
    assert!(
        colorspace_phase.metrics["neutral_sample_bands"].is_array(),
        "expected neutral sample band diagnostic"
    );
    assert!(
        colorspace_phase.metrics["render_input_source"].is_string(),
        "expected render input source diagnostic"
    );
    assert!(
        colorspace_phase.metrics["render_input_reason"].is_string(),
        "expected render input selection reason diagnostic"
    );
    assert_eq!(
        colorspace_phase.metrics["calibration"]["status"],
        "not_configured"
    );
    assert!(
        colorspace_phase.metrics["direct_density_candidate_evaluated"].is_boolean(),
        "expected direct density candidate evaluation diagnostic"
    );
    assert_eq!(
        colorspace_phase.metrics["direct_density_candidate_evaluated"], true,
        "auto mode should evaluate direct-density versus ICA render input"
    );
    assert!(
        colorspace_phase.metrics["ica_candidate"].is_object(),
        "expected ICA colorspace candidate diagnostics"
    );
    assert!(
        colorspace_phase.metrics["direct_density_candidate"].is_object(),
        "auto mode should retain direct-density candidate diagnostics"
    );
    assert!(
        colorspace_phase.metrics["image_matrix_pre_scale_clipped_low_ratio"].is_array(),
        "expected rejected image-matrix low-clipping diagnostic"
    );
    assert!(
        colorspace_phase.metrics["exposure_scale"].is_number(),
        "expected colorspace exposure scale diagnostic"
    );
    assert!(
        colorspace_phase.metrics["channel_anchor_min_count"].is_number(),
        "expected minimum colorspace anchor count diagnostic"
    );
    assert!(
        colorspace_phase.metrics["channel_anchor_low_support_threshold"].is_number(),
        "expected colorspace anchor support threshold diagnostic"
    );
    assert!(
        colorspace_phase.metrics["channel_anchor_low_support"].is_array(),
        "expected weak colorspace anchor mask diagnostic"
    );
    assert!(
        colorspace_phase.metrics["weak_anchor_fallback_used"].is_boolean(),
        "expected weak-anchor fallback diagnostic"
    );
    assert!(
        colorspace_phase.metrics["ica_candidate"]["channel_anchor_low_support"].is_array(),
        "expected candidate colorspace diagnostics to include anchor support"
    );
    assert!(
        colorspace_phase.metrics["ica_candidate"]["candidate_comparison"]["image_derived"]
            ["dominant_anchor_quality"]
            .is_object(),
        "expected candidate colorspace diagnostics to include anchor quality"
    );
    assert!(
        colorspace_phase.metrics["ica_candidate"]["weak_anchor_fallback_used"].is_boolean(),
        "expected candidate colorspace diagnostics to include weak-anchor fallback state"
    );

    let pre: f64 = colorspace_phase.metrics["pre_scale_clipped_high_ratio"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .sum();
    let post: f64 = colorspace_phase.metrics["post_scale_clipped_high_ratio"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .sum();
    assert!(
        post <= pre + 1e-9,
        "expected post-scale clipping to not increase (before {}, after {})",
        pre,
        post
    );
    assert!(colorspace_phase.metrics["color_candidate_comparison_artifact"].is_null());
    assert!(colorspace_phase.metrics["gamut_clipping_map_artifact"].is_null());
    assert!(colorspace_phase.metrics["gamut_clipping_map_diagnostics"].is_null());
    assert!(colorspace_phase.metrics["scene_referred_prophoto_float_artifact"].is_null());
    assert!(
        colorspace_phase.metrics["scene_referred_prophoto_float_artifact_diagnostics"].is_null()
    );
    assert!(
        !output_dir
            .join("phase46_color_candidate_comparison.tiff")
            .exists(),
        "debug-only color comparison artifact should not be written without --debug"
    );
    assert!(
        !output_dir.join("phase46_gamut_clipping_map.tiff").exists(),
        "debug-only gamut/clipping map should not be written without --debug"
    );
    assert!(
        !output_dir
            .join("phase46_scene_referred_prophoto_float.tiff")
            .exists(),
        "debug-only scene-referred float artifact should not be written without --debug"
    );
}

#[test]
fn test_pipeline_applies_valid_calibration_profile() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp =
        synthetic::film_negative_image(120, 240, 0, 20, [12000, 7000, 3000], [5200, 4100, 3400]);
    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let profile_path = tmp.path().join("synthetic-profile.json");
    std::fs::write(&profile_path, valid_calibration_profile_json()).unwrap();

    let output_dir = tmp.path().join("output");
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir,
        calibration_profile: Some(profile_path.clone()),
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let colorspace_phase = report
        .phases
        .iter()
        .find(|p| p.name == "colorspace_mapping")
        .expect("colorspace_mapping phase");

    assert_eq!(colorspace_phase.metrics["calibration"]["status"], "applied");
    assert_eq!(
        colorspace_phase.metrics["calibration"]["source"],
        "external_calibration_profile"
    );
    assert_eq!(
        colorspace_phase.metrics["mapping_strategy"],
        "calibrated_profile"
    );
    assert_eq!(
        colorspace_phase.metrics["calibration"]["external_profile"]["path"],
        profile_path.to_string_lossy().as_ref()
    );
    assert!(colorspace_phase.metrics["candidate_comparison"]["image_derived"].is_object());
    assert!(
        colorspace_phase.metrics["candidate_comparison"]["calibrated_profile"].is_object(),
        "expected calibrated-vs-image candidate diagnostics"
    );
    assert!(
        colorspace_phase.metrics["usable_colourspace"]["post_scale_preserved_ratio"].is_number()
    );
    assert_eq!(
        colorspace_phase.metrics["calibration_acceptance"]["status"],
        "accepted"
    );
    let color_mapping_application =
        &colorspace_phase.metrics["calibration"]["color_mapping_application"];
    assert_eq!(color_mapping_application["evaluated"], true);
    assert_eq!(color_mapping_application["applied"], true);
    assert_eq!(color_mapping_application["selection_status"], "accepted");
    assert_eq!(
        color_mapping_application["selected_candidate"],
        colorspace_phase.metrics["selected_candidate"]
    );
    assert_eq!(
        color_mapping_application["preferred_candidate"],
        "calibrated_direct_profile"
    );
    assert!(color_mapping_application["reason"]
        .as_str()
        .is_some_and(|reason| reason.contains("quality score")));
    assert!(color_mapping_application["definition"]
        .as_str()
        .is_some_and(|definition| definition.contains("scanner linearization")));
    assert!(colorspace_phase.metrics["selected_quality_score"].is_number());
}

#[test]
fn test_negative_auto_uses_evidence_driven_color_and_preserves_density_headroom() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp =
        synthetic::film_negative_image(120, 240, 0, 20, [12000, 7000, 3000], [5200, 4100, 3400]);
    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let output_dir = tmp.path().join("output");
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir,
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();

    let colorspace_phase = report
        .phases
        .iter()
        .find(|p| p.name == "colorspace_mapping")
        .expect("colorspace_mapping phase");
    assert_ne!(
        colorspace_phase.metrics["mapping_strategy"], "positive_rgb_passthrough",
        "negative reconstruction must not masquerade as already-profiled positive RGB"
    );
    assert!(
        matches!(
            colorspace_phase.metrics["render_input_source"].as_str(),
            Some("fastica_separated_transmittance") | Some("direct_density_transmittance")
        ),
        "auto mode should select an evaluated negative-film reconstruction candidate"
    );
    assert_eq!(
        colorspace_phase.metrics["direct_density_candidate_evaluated"], true,
        "auto mode must compare ICA and direct-density evidence"
    );
    assert!(colorspace_phase.metrics["candidate_scores"].is_array());

    // The tone stage records the pre-tone input percentiles that drove auto exposure.
    let tone_phase = report
        .phases
        .iter()
        .find(|p| p.name == "tone_mapping")
        .expect("tone_mapping phase");
    assert!(
        tone_phase.metrics["tone_input_linear_percentiles"].is_array(),
        "expected pre-tone input percentile diagnostics"
    );

    let fastica_phase = report
        .phases
        .iter()
        .find(|p| p.name == "fastica")
        .expect("fastica phase");
    assert_eq!(fastica_phase.metrics["skipped"], false);
    assert_eq!(fastica_phase.confidence, 0.0);
    assert_eq!(fastica_phase.metrics["evidence_evaluated"], false);
    assert_eq!(
        fastica_phase.metrics["numerical_convergence_evaluated"],
        true
    );
    assert_eq!(
        fastica_phase.metrics["physical_separation_evidence_evaluated"],
        false
    );
    assert_eq!(
        fastica_phase.metrics["confidence_basis"],
        "physical_dye_separation_not_independently_validated"
    );
    assert!(matches!(
        fastica_phase.metrics["operation_status"].as_str(),
        Some("numerically_converged_physical_separation_unvalidated")
            | Some("numerical_convergence_failed_physical_separation_unvalidated")
    ));
    assert_eq!(
        fastica_phase.metrics["numerical_convergence_confidence"],
        if fastica_phase.metrics["converged"] == true {
            serde_json::json!(1.0)
        } else {
            serde_json::json!(0.0)
        }
    );
    let density_phase = report
        .phases
        .iter()
        .find(|p| p.name == "density_inversion")
        .expect("density_inversion phase");
    assert_eq!(density_phase.metrics["skipped"], false);
    assert_eq!(
        density_phase.metrics["direct_density_render_normalization"],
        "shared_robust_white_anchor_with_signed_highlight_headroom"
    );
    assert_eq!(density_phase.metrics["highlight_headroom_clipped"], false);
    assert!(density_phase.metrics["highlight_headroom_samples"].is_array());
}

#[test]
fn test_pipeline_rejects_invalid_calibration_profile_without_failing_render() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp =
        synthetic::film_negative_image(120, 240, 0, 20, [12000, 7000, 3000], [5200, 4100, 3400]);
    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let profile_path = tmp.path().join("singular-profile.json");
    std::fs::write(&profile_path, singular_calibration_profile_json()).unwrap();

    let output_dir = tmp.path().join("output");
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: output_dir.clone(),
        calibration_profile: Some(profile_path),
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    assert!(output_dir.join("output.tiff").exists());
    let colorspace_phase = report
        .phases
        .iter()
        .find(|p| p.name == "colorspace_mapping")
        .expect("colorspace_mapping phase");

    assert_eq!(
        colorspace_phase.metrics["calibration"]["status"],
        "rejected"
    );
    assert_ne!(
        colorspace_phase.metrics["mapping_strategy"],
        "calibrated_profile"
    );
    let color_mapping_application =
        &colorspace_phase.metrics["calibration"]["color_mapping_application"];
    assert_eq!(color_mapping_application["evaluated"], false);
    assert_eq!(color_mapping_application["applied"], false);
    assert_eq!(
        color_mapping_application["selection_status"],
        "not_applicable"
    );
    assert_eq!(
        color_mapping_application["preferred_candidate"],
        serde_json::Value::Null
    );
    assert_eq!(
        color_mapping_application["selected_candidate"],
        colorspace_phase.metrics["selected_candidate"]
    );
    assert!(colorspace_phase
        .warnings
        .iter()
        .any(|warning| warning.contains("calibration profile was rejected")));
}

#[test]
fn test_pipeline_color_mode_calibrated_fails_without_calibration() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp =
        synthetic::film_negative_image(120, 240, 0, 20, [12000, 7000, 3000], [5200, 4100, 3400]);
    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let output_dir = tmp.path().join("output");
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: output_dir.clone(),
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Calibrated,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let err =
        scanstitch::pipeline::run(&cli).expect_err("calibrated mode should require calibration");
    let message = err.to_string();
    assert!(
        message.contains("--color-mode calibrated requires"),
        "unexpected error: {}",
        message
    );

    let report: scanstitch::report::PipelineReport =
        serde_json::from_str(&std::fs::read_to_string(output_dir.join("report.json")).unwrap())
            .unwrap();
    let colorspace_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "colorspace_mapping")
        .expect("failed colorspace phase");
    assert!(!colorspace_phase.success);
    assert!(colorspace_phase
        .errors
        .iter()
        .any(|error| error.contains("--color-mode calibrated requires")));
}

#[test]
fn test_pipeline_applies_explicit_scanner_library_profile() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp =
        synthetic::film_negative_image(120, 240, 0, 20, [12000, 7000, 3000], [5200, 4100, 3400]);
    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();
    let library_dir = write_library(&tmp, true);

    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: tmp.path().join("output"),
        calibration_profile: None,
        calibration_library: Some(library_dir),
        scanner_profile: Some("scanner-a".to_string()),
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let colorspace_phase = report
        .phases
        .iter()
        .find(|p| p.name == "colorspace_mapping")
        .expect("colorspace_mapping phase");

    assert_eq!(colorspace_phase.metrics["calibration"]["status"], "applied");
    assert_eq!(
        colorspace_phase.metrics["calibration"]["source"],
        "calibration_library"
    );
    assert_eq!(
        colorspace_phase.metrics["calibration"]["scanner_profile"]["status"],
        "applied"
    );
    assert_eq!(
        colorspace_phase.metrics["calibration"]["scanner_profile"]["profile_id"],
        "scanner-a"
    );
    assert!(
        colorspace_phase.metrics["calibration"]["library"]["invalid_entries"]
            .as_array()
            .is_some_and(|entries| entries.len() == 1),
        "expected invalid library entry diagnostics"
    );
}

#[test]
fn test_pipeline_applies_validated_scanner_linearization_before_density() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    let library_dir = tmp.path().join("calibration");
    std::fs::create_dir_all(&input_dir).unwrap();
    std::fs::create_dir_all(library_dir.join("scanners")).unwrap();
    std::fs::write(
        library_dir.join("scanners/scanner.json"),
        scanner_library_profile_json_with_linearization(),
    )
    .unwrap();
    let comp =
        synthetic::film_negative_image(96, 192, 0, 18, [12000, 7000, 3000], [5200, 4100, 3400]);
    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: tmp.path().join("output"),
        calibration_profile: None,
        calibration_library: Some(library_dir),
        scanner_profile: Some("scanner-a".to_string()),
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::DirectDensity,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let linearization = report
        .phases
        .iter()
        .find(|phase| phase.name == "scanner_linearization")
        .expect("scanner linearization phase");
    assert_eq!(linearization.metrics["skipped"], false);
    assert_eq!(
        linearization.metrics["diagnostics"]["model_id"],
        "pipeline-scanner-linearization-v1"
    );
    assert_eq!(linearization.metrics["diagnostics"]["output_bit_depth"], 16);
    assert_eq!(
        linearization.metrics["application_stage"],
        "per_component_before_border_crop_classification_and_stitch"
    );
    assert_eq!(linearization.metrics["input_count"], 2);
    assert_eq!(
        linearization.metrics["components"]
            .as_array()
            .expect("component linearization diagnostics")
            .len(),
        2
    );
    assert_eq!(
        linearization.metrics["base_color_before"],
        serde_json::Value::Null
    );
    assert_eq!(
        linearization.metrics["base_color_after"],
        serde_json::Value::Null
    );
    let linearization_index = report
        .phases
        .iter()
        .position(|phase| phase.name == "scanner_linearization")
        .unwrap();
    let border_index = report
        .phases
        .iter()
        .position(|phase| phase.name == "border_removal")
        .unwrap();
    assert!(linearization_index < border_index);
    let density = report
        .phases
        .iter()
        .find(|phase| phase.name == "density_inversion")
        .expect("density phase");
    assert_eq!(density.metrics["bit_depth"], 16);
    assert!(density.metrics["base_estimate_source"]
        .as_str()
        .is_some_and(
            |source| source.contains("validated_scanner_linearization_per_component_pre_crop")
        ));
    let ica = report
        .phases
        .iter()
        .find(|phase| phase.name == "fastica")
        .expect("ICA phase");
    assert_eq!(ica.confidence, 0.0);
    assert_eq!(ica.metrics["skipped"], true);
    assert_eq!(ica.metrics["evidence_evaluated"], false);
    assert_eq!(
        ica.metrics["operation_status"],
        "not_evaluated_explicit_direct_density_route"
    );
    assert_eq!(
        ica.metrics["confidence_basis"],
        "not_evaluated_user_selected_alternate_route"
    );
}

#[test]
fn test_pipeline_applies_explicit_scanner_and_roll_library_profiles() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp =
        synthetic::film_negative_image(120, 240, 0, 20, [12000, 7000, 3000], [5200, 4100, 3400]);
    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();
    let library_dir = write_library(&tmp, false);

    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: tmp.path().join("output"),
        calibration_profile: None,
        calibration_library: Some(library_dir),
        scanner_profile: Some("scanner-a".to_string()),
        roll_profile: Some("roll-a".to_string()),
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let colorspace_phase = report
        .phases
        .iter()
        .find(|p| p.name == "colorspace_mapping")
        .expect("colorspace_mapping phase");

    assert_eq!(colorspace_phase.metrics["calibration"]["status"], "applied");
    assert_eq!(
        colorspace_phase.metrics["calibration"]["roll_profile"]["status"],
        "applied"
    );
    assert_eq!(
        colorspace_phase.metrics["calibration"]["roll_profile"]["correction_applied"],
        true
    );
    assert_eq!(
        colorspace_phase.metrics["mapping_strategy"],
        "calibrated_profile"
    );
    assert!(colorspace_phase.metrics["candidate_quality_scores"]
        .as_array()
        .expect("candidate scores")
        .iter()
        .any(|candidate| candidate["candidate"] == "scanner_prior_image_adaptation"));
    assert!(colorspace_phase.metrics["calibration"]["nearest_roll_candidates"].is_array());
}

#[test]
fn test_pipeline_applies_validated_measured_negative_response_before_colorspace_mapping() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    let library_dir = tmp.path().join("calibration");
    std::fs::create_dir_all(&input_dir).unwrap();
    std::fs::create_dir_all(library_dir.join("scanners")).unwrap();
    std::fs::create_dir_all(library_dir.join("rolls")).unwrap();
    std::fs::write(
        library_dir.join("scanners/scanner.json"),
        scanner_library_profile_json(),
    )
    .unwrap();
    std::fs::write(
        library_dir.join("rolls/roll.json"),
        roll_library_profile_json_with_measured_response(),
    )
    .unwrap();

    let comp = varied_film_negative_chart(96, 192, 18, [12000, 7000, 3000]);
    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir: tmp.path().join("output"),
        calibration_profile: None,
        calibration_library: Some(library_dir.clone()),
        scanner_profile: Some("scanner-a".to_string()),
        roll_profile: Some("roll-a".to_string()),
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let density = report
        .phases
        .iter()
        .find(|phase| phase.name == "density_inversion")
        .expect("density phase");
    assert_eq!(
        density.metrics["direct_density_response_candidates"]["selected_model"],
        "measured_nonlinear_dye_separation"
    );
    assert_eq!(
        density.metrics["direct_density_response_model"]["measured_model_id"],
        "synthetic-pipeline-response-v1"
    );
    assert_eq!(
        density.metrics["operation_status"],
        "held_out_measured_response_supported"
    );
    assert_eq!(density.metrics["evidence_evaluated"], true);
    assert_eq!(
        density.metrics["negative_response_model_evidence_evaluated"],
        true
    );
    assert_eq!(density.metrics["negative_response_model_confidence"], 0.94);
    assert_eq!(
        density.metrics["confidence_basis"],
        "minimum_film_base_and_negative_response_model_evidence"
    );
    let film_base_confidence = density.metrics["film_base_evidence_confidence"]
        .as_f64()
        .expect("film-base evidence confidence");
    assert!(film_base_confidence > 0.0);
    assert!(
        (density.confidence - film_base_confidence.min(0.94)).abs() < 1e-12,
        "density confidence {} did not preserve the weakest base/response evidence",
        density.confidence
    );
    assert!(
        density
            .warnings
            .iter()
            .all(|warning| !warning
                .contains("without held-out physical negative-response validation"))
    );
    let colorspace = report
        .phases
        .iter()
        .find(|phase| phase.name == "colorspace_mapping")
        .expect("colorspace phase");
    assert_eq!(
        colorspace.metrics["negative_response_reconstruction"]["model"]["model"],
        "measured_nonlinear_dye_separation"
    );
    assert_eq!(
        colorspace.metrics["render_input_source"], "direct_density_transmittance",
        "auto mode must prefer a held-out-validated measured response over blind ICA"
    );
    let measured_render_reason = colorspace.metrics["render_input_reason"]
        .as_str()
        .expect("render input reason");
    assert!(
        measured_render_reason.contains("physical evidence outranks blind frame-derived ICA"),
        "unexpected measured-response auto-selection reason: {measured_render_reason}; selected diagnostics: {}",
        colorspace.metrics["selected_candidate"]
    );
    assert_eq!(
        colorspace.metrics["negative_response_reconstruction"]["curve_interpolation"],
        "monotone_piecewise_cubic_hermite_with_endpoint_tangent_extrapolation"
    );
    assert_eq!(
        colorspace.metrics["negative_reconstruction_confidence_status"],
        "held_out_measured_response_and_runtime_coverage_supported"
    );
    assert_eq!(
        colorspace.metrics["negative_reconstruction_evidence_evaluated"],
        true
    );
    assert!(
        (colorspace.metrics["negative_reconstruction_evidence_confidence"]
            .as_f64()
            .expect("negative reconstruction evidence confidence")
            - density.confidence)
            .abs()
            < 1e-12
    );
    let selected_mapping_confidence = colorspace.metrics["selected_mapping_evidence_confidence"]
        .as_f64()
        .expect("selected mapping evidence confidence");
    assert!(
        (colorspace.confidence - selected_mapping_confidence.min(density.confidence)).abs() < 1e-12
    );
    assert_eq!(
        colorspace.metrics["color_confidence_status"],
        "limited_negative_reconstruction_evidence"
    );
    assert_eq!(colorspace.metrics["calibration"]["status"], "applied");

    let tone = report
        .phases
        .iter()
        .find(|phase| phase.name == "tone_mapping")
        .expect("tone phase");
    assert!((tone.confidence - colorspace.confidence).abs() < 1e-12);
    assert_eq!(
        tone.metrics["upstream_color_confidence"],
        colorspace.confidence
    );
    assert_eq!(
        tone.metrics["tone_confidence_status"],
        "limited_upstream_evidence"
    );

    std::fs::write(
        library_dir.join("rolls/roll.json"),
        roll_library_profile_json_with_narrow_measured_response(),
    )
    .unwrap();
    let mut out_of_support_cli = cli.clone();
    out_of_support_cli.output_dir = tmp.path().join("output-out-of-support");
    let out_of_support_report = scanstitch::pipeline::run(&out_of_support_cli).unwrap();
    let out_of_support_density = out_of_support_report
        .phases
        .iter()
        .find(|phase| phase.name == "density_inversion")
        .expect("out-of-support density phase");
    assert!(out_of_support_density.confidence > 0.0);
    assert_eq!(
        out_of_support_density.metrics["operation_status"],
        "held_out_measured_response_supported"
    );
    let out_of_support_colorspace = out_of_support_report
        .phases
        .iter()
        .find(|phase| phase.name == "colorspace_mapping")
        .expect("out-of-support colorspace phase");
    assert!(
        out_of_support_colorspace.metrics["negative_response_reconstruction"]
            ["curve_extrapolated_any_ratio"]
            .as_f64()
            .is_some_and(|ratio| ratio > 0.01)
    );
    assert_eq!(
        out_of_support_colorspace.metrics["negative_response_review_required"],
        true
    );
    assert_eq!(
        out_of_support_colorspace.metrics["negative_reconstruction_confidence_status"],
        "review_required_measured_response_outside_runtime_curve_support"
    );
    assert_eq!(
        out_of_support_colorspace.metrics["negative_reconstruction_evidence_evaluated"],
        true
    );
    assert_eq!(
        out_of_support_colorspace.metrics["negative_reconstruction_evidence_confidence"],
        0.0
    );
    assert_eq!(out_of_support_colorspace.confidence, 0.0);
    assert_eq!(
        out_of_support_colorspace.metrics["color_confidence_status"],
        "review_required_negative_reconstruction"
    );
    let out_of_support_tone = out_of_support_report
        .phases
        .iter()
        .find(|phase| phase.name == "tone_mapping")
        .expect("out-of-support tone phase");
    assert_eq!(out_of_support_tone.confidence, 0.0);
    assert_eq!(
        out_of_support_tone.metrics["tone_confidence_status"],
        "review_required_upstream_color"
    );
    let out_of_support_save = out_of_support_report
        .phases
        .iter()
        .find(|phase| phase.name == "save")
        .expect("out-of-support save phase");
    assert_eq!(
        out_of_support_save.metrics["render_review_status"],
        "review_required_negative_response",
        "unexpected geometry result: {}",
        out_of_support_report
            .phases
            .iter()
            .find(|phase| phase.name == "border_removal")
            .map(|phase| &phase.metrics)
            .expect("out-of-support border phase")
    );
    assert_eq!(out_of_support_save.confidence, 0.0);
}

#[test]
fn test_pipeline_reports_tone_fit_diagnostics() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp1 =
        synthetic::film_negative_image(160, 320, 0, 24, [12000, 7000, 3000], [5000, 4000, 3500]);
    let comp2 =
        synthetic::film_negative_image(160, 320, 0, 24, [12000, 7000, 3000], [5200, 4100, 3400]);

    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp1, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp2, &path2).unwrap();

    let output_dir = tmp.path().join("output");
    let cli = scanstitch::cli::Cli {
        inputs: vec![path1, path2],
        output_dir,
        calibration_profile: None,
        calibration_library: None,
        scanner_profile: None,
        roll_profile: None,
        film_stock: None,
        base_color: None,
        base_color_source: None,
        base_color_confidence: None,
        base_color_reason: None,
        color_mode: scanstitch::colorspace::ColorMode::Auto,
        render_input: scanstitch::cli::RenderInputMode::Auto,
        input_mode: scanstitch::cli::InputMode::Negative,
        render_intent: scanstitch::cli::RenderIntent::ModernClean,
        quality_mode: scanstitch::cli::QualityMode::Perfect,
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
        write_master: false,
        review_sidecar: None,
        write_review_sidecar: None,
        debug: false,
        force_stitch: false,
        force_no_stitch: true,
        transform: "auto".into(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
        require_reviewable: false,
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let tone_phase = report
        .phases
        .iter()
        .find(|p| p.name == "tone_mapping")
        .expect("tone_mapping phase");

    let fit_domain = tone_phase.metrics["fit_domain"]
        .as_str()
        .expect("fit domain should be a string");
    assert!(
        matches!(fit_domain, "linear_luminance" | "log2_compressed_luminance"),
        "expected an explicit tone-fit domain, got {}",
        fit_domain
    );
    assert!(
        tone_phase.metrics["perceptual_luminance_gain"].is_number(),
        "expected perceptual luminance gain diagnostic"
    );
    assert!(
        tone_phase.metrics["tone_fit_policy"].is_string(),
        "expected tone fit policy diagnostic"
    );
    assert!(
        tone_phase.metrics["auto_exposure_ev"].is_number(),
        "expected automatic exposure diagnostic"
    );
    assert!(
        tone_phase.metrics["render_exposure_ev"].is_number(),
        "expected applied render exposure diagnostic"
    );
    assert_eq!(
        tone_phase.metrics["post_tone_buffer_policy"],
        "owned_in_place_single_working_buffer"
    );
    assert_eq!(tone_phase.metrics["scene_referred_master_preserved"], true);
    assert!(
        tone_phase.metrics["input_linear_percentiles"].is_array(),
        "expected linear luminance percentiles"
    );
    assert!(
        tone_phase.metrics["input_perceptual_percentiles"].is_array(),
        "expected perceptual luminance percentiles"
    );
    assert!(
        tone_phase.metrics["mapped_linear_percentiles"].is_array(),
        "expected mapped linear luminance percentiles"
    );
    assert!(
        tone_phase.metrics["mapped_perceptual_percentiles"].is_array(),
        "expected mapped perceptual luminance percentiles"
    );
    assert!(
        tone_phase.metrics["bright_neutral_saturation_median"].is_number(),
        "expected bright neutral saturation median diagnostic"
    );
    assert!(
        tone_phase.metrics["bright_neutral_saturation_p95"].is_number(),
        "expected bright neutral saturation p95 diagnostic"
    );
    assert!(
        tone_phase.metrics["bright_neutral_max_saturation"].is_number(),
        "expected bright neutral saturation cutoff diagnostic"
    );
    assert!(
        tone_phase.metrics["bright_saturated_saturation_p95"].is_number(),
        "expected saturated bright-band saturation p95 diagnostic"
    );
    assert!(
        tone_phase.metrics["color_protection_policy"].is_string(),
        "expected tone color protection policy diagnostic"
    );
    assert!(
        tone_phase.metrics["color_trust_state"].is_string(),
        "expected tone color trust-state diagnostic"
    );
    assert!(
        tone_phase.metrics["color_protection_reason"].is_string(),
        "expected tone color protection reason diagnostic"
    );
    assert!(
        tone_phase.metrics["shadow_saturation_median"].is_number(),
        "expected shadow saturation median diagnostic"
    );
    assert!(
        tone_phase.metrics["shadow_saturation_p95"].is_number(),
        "expected shadow saturation p95 diagnostic"
    );
    assert!(
        tone_phase.metrics["midtone_luminance_percentiles"].is_array(),
        "expected midtone luminance percentile diagnostics"
    );
    assert!(
        tone_phase.metrics["midtone_saturation_median"].is_number(),
        "expected midtone saturation median diagnostic"
    );
    assert!(
        tone_phase.metrics["shadow_rgb_median"].is_array(),
        "expected shadow RGB median diagnostic"
    );
    assert!(
        tone_phase.metrics["bright_neutral_rgb_median"].is_array(),
        "expected bright neutral RGB median diagnostic"
    );
    assert!(
        tone_phase.metrics["render_quality_bands"].is_object(),
        "expected grouped render quality band diagnostics"
    );
}

#[test]
fn test_pipeline_applies_absolute_deskew_before_border_removal() {
    let tmp = TempDir::new().unwrap();
    let input = tmp.path().join("single-positive.tiff");
    let image = synthetic::image_with_borders(140, 220, 18, [7_000, 8_000, 9_000]);
    scanstitch::tiff_io::save_tiff_u16(&image, &input).unwrap();

    let mut cli = positive_fast_cli(vec![input], tmp.path().join("output"));
    cli.geometry.deskew = scanstitch::cli::DeskewMode::Manual;
    cli.geometry.deskew_angle_degrees = 0.75;
    let report = scanstitch::pipeline::run(&cli).unwrap();

    let scanner_index = report
        .phases
        .iter()
        .position(|phase| phase.name == "scanner_linearization")
        .expect("scanner phase");
    let deskew_index = report
        .phases
        .iter()
        .position(|phase| phase.name == "deskew")
        .expect("deskew phase");
    let border_index = report
        .phases
        .iter()
        .position(|phase| phase.name == "border_removal")
        .expect("border phase");
    assert!(scanner_index < deskew_index && deskew_index < border_index);

    let deskew = &report.phases[deskew_index];
    assert_eq!(deskew.metrics["requested_mode"], "manual");
    assert_eq!(deskew.metrics["status"], "applied");
    assert_eq!(deskew.metrics["applied"], true);
    assert_eq!(deskew.metrics["correction_degrees"], -0.75);
    assert!(deskew.metrics["retained_area_ratio"]
        .as_f64()
        .is_some_and(|ratio| ratio >= 0.90));
    assert_eq!(
        deskew.metrics["interpolation"],
        "bicubic_catmull_rom_single_resample"
    );
}

#[test]
fn pipeline_applies_semantic_orientation_correction_before_geometry_and_reports_effective_hash() {
    let tmp = TempDir::new().unwrap();
    let input = tmp.path().join("upside-down-positive.tiff");
    let mut image = synthetic::constant_image(120, 220, [7000, 8000, 9000]);
    for y in 0..120 {
        for x in 0..220 {
            if !(7..107).contains(&y) || !(9..205).contains(&x) {
                for channel in 0..3 {
                    image[[y, x, channel]] = 0;
                }
            }
        }
    }
    scanstitch::tiff_io::save_tiff_u16(&image, &input).unwrap();

    let mut cli = positive_fast_cli(vec![input], tmp.path().join("output"));
    cli.geometry.orientation_correction = scanstitch::cli::OrientationCorrection::Rotate180;
    let report = scanstitch::pipeline::run(&cli).unwrap();

    let load = report
        .phases
        .iter()
        .find(|phase| phase.name == "load")
        .expect("load phase");
    let decode = &load.metrics["components"][0]["decode"];
    assert_eq!(decode["orientation"]["transform"], "identity");
    assert_eq!(decode["orientation_correction"]["requested"], "rotate-180");
    assert_eq!(decode["orientation_correction"]["applied"], true);
    assert_eq!(decode["orientation_correction"]["effective_tag_value"], 3);
    assert_eq!(
        decode["orientation_correction"]["effective_transform"],
        "rotate_180"
    );
    assert_ne!(
        decode["decoded_pixel_sha256"],
        decode["source_orientation_materialized_decoded_pixel_sha256"]
    );

    let border = report
        .phases
        .iter()
        .find(|phase| phase.name == "border_removal")
        .expect("border phase");
    assert!(
        border.metrics["component1"]["top_removed"]
            .as_u64()
            .unwrap()
            >= 13
    );
    assert!(
        border.metrics["component1"]["bottom_removed"]
            .as_u64()
            .unwrap()
            >= 7
    );
    assert!(
        border.metrics["component1"]["left_removed"]
            .as_u64()
            .unwrap()
            >= 15
    );
    assert!(
        border.metrics["component1"]["right_removed"]
            .as_u64()
            .unwrap()
            >= 9
    );
}

#[test]
fn unresolved_strong_border_candidate_propagates_to_final_geometry_review() {
    let tmp = TempDir::new().unwrap();
    let input = tmp.path().join("ambiguous-right-border.tiff");
    let height = 64usize;
    let width = 1000usize;
    let rebate_start = 930usize;
    let mut image = synthetic::constant_image(height, width, [6000, 5000, 4000]);
    for y in 0..height {
        let edge = if y < height / 2 { 24_900 } else { 34_900 };
        for x in 880..rebate_start {
            for channel in 0..3 {
                image[[y, x, channel]] = edge;
            }
        }
        for x in rebate_start..width {
            for channel in 0..3 {
                image[[y, x, channel]] = 30_000;
            }
        }
    }
    scanstitch::tiff_io::save_tiff_u16(&image, &input).unwrap();

    let cli = positive_fast_cli(vec![input], tmp.path().join("output"));
    let report = scanstitch::pipeline::run(&cli).unwrap();
    let border = report
        .phases
        .iter()
        .find(|phase| phase.name == "border_removal")
        .expect("border phase");
    assert_eq!(
        border.metrics["component1"]["border_decision_status"],
        "no_crop_with_unresolved_strong_edge_candidate"
    );
    assert_eq!(
        border.metrics["component1"]["right_unresolved_strong_candidate"],
        true
    );
    assert_eq!(border.metrics["review_required"], true);

    let save = report
        .phases
        .iter()
        .find(|phase| phase.name == "save")
        .expect("save phase");
    assert_eq!(save.metrics["geometry_review_required"], true);
    assert!(save.metrics["geometry_review_reason"]
        .as_str()
        .is_some_and(|reason| reason.contains("border crop requires review")));
    assert_eq!(
        save.metrics["render_review_status"],
        "blocked_geometry_review"
    );
}

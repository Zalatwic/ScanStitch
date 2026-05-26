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
        component1: path1,
        component2: path2,
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
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let metadata = report.metadata.as_ref().expect("run metadata");
    assert_eq!(metadata.package_version, env!("CARGO_PKG_VERSION"));
    assert!(metadata.generated_at.contains('T'));
    assert!(!report.phases.is_empty());
    assert!(output_dir.join("output.tiff").exists());
    assert!(output_dir.join("master_scene_referred.tiff").exists());
    assert!(output_dir.join("review_srgb.png").exists());
    assert!(
        output_dir
            .join("phase46_color_candidate_comparison.tiff")
            .exists(),
        "expected debug color candidate comparison artifact"
    );
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
    assert!(colorspace_phase.metrics["color_candidate_comparison_artifact"].is_string());
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
    assert!(save_phase.metrics["master_scene_referred_path"].is_string());
    assert_eq!(
        save_phase.metrics["master_scene_referred_diagnostics"]["sample_format"],
        "IEEEFP"
    );
    assert!(save_phase.metrics["review_srgb_path"].is_string());
    assert_eq!(save_phase.metrics["overwrote_existing_output"], false);
    assert_eq!(save_phase.metrics["stale_render_artifact_count"], 1);
    assert!(save_phase
        .warnings
        .iter()
        .any(|warning| warning.contains("not overwritten")));
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
        component1: path1,
        component2: path2,
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
    };

    let report = scanstitch::pipeline::run(&cli).unwrap();
    let density_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "density_inversion")
        .expect("density phase");
    assert_eq!(density_phase.metrics["skipped"], true);
    assert_eq!(density_phase.metrics["input_mode"], "positive");
    assert!(density_phase.metrics.get("base_color").is_none());
    let ica_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "fastica")
        .expect("fastica phase");
    assert_eq!(ica_phase.metrics["skipped"], true);
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
fn test_positive_pipeline_warns_on_orange_negative_like_input() {
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
        component1: path1,
        component2: path2,
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
        component1: path1,
        component2: path2,
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
        component1: path1,
        component2: path2,
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
    assert!(
        ica_phase.metrics["confidence_limited_by_base_estimate"] == true,
        "expected ICA phase to record confidence limiting"
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
fn test_manual_base_color_override_unblocks_density_review_gate() {
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
        component1: path1,
        component2: path2,
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

    let save_phase = report
        .phases
        .iter()
        .find(|p| p.name == "save")
        .expect("save phase");
    assert_eq!(save_phase.metrics["render_review_status"], "reviewable");
    assert_eq!(save_phase.metrics["render_reviewable"], true);
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
        component1: path1,
        component2: path2,
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
        component1: path1,
        component2: path2,
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
        component1: path1,
        component2: path2,
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
        "expected direct-density colorspace candidate diagnostics in auto mode"
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
        component1: path1,
        component2: path2,
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
    assert!(colorspace_phase.metrics["selected_quality_score"].is_number());
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
        component1: path1,
        component2: path2,
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
        component1: path1,
        component2: path2,
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
        component1: path1,
        component2: path2,
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
        component1: path1,
        component2: path2,
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
        component1: path1,
        component2: path2,
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

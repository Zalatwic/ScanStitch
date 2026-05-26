mod common;

use common::synthetic;
use ndarray::Array3;
use scanstitch::interactive::{
    self, InteractiveRenderCache, InteractiveRenderControls, PreviewQuality,
};
use scanstitch::report::PipelineReport;
use scanstitch::tonemap::{
    ToneColorProtection, ToneCurveDiagnostics, ToneCurveParams, ToneFitDomain,
};
use std::time::SystemTime;
use tempfile::TempDir;

fn test_cli(output_dir: std::path::PathBuf) -> scanstitch::cli::Cli {
    scanstitch::cli::Cli {
        component1: output_dir.join("component1.tiff"),
        component2: output_dir.join("component2.tiff"),
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
        transform: "auto".to_string(),
        ica_max_iter: 10,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
    }
}

fn pipeline_cli(
    component1: std::path::PathBuf,
    component2: std::path::PathBuf,
    output_dir: std::path::PathBuf,
) -> scanstitch::cli::Cli {
    scanstitch::cli::Cli {
        component1,
        component2,
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
        transform: "auto".to_string(),
        ica_max_iter: 20,
        ica_tol: 1e-4,
        bit_depth: 14,
        use_opencv: false,
    }
}

fn tone_diagnostics() -> ToneCurveDiagnostics {
    ToneCurveDiagnostics {
        fit_domain: ToneFitDomain::LinearLuminance.as_str(),
        perceptual_luminance_gain: 15.0,
        input_linear_percentiles: [0.1, 0.4, 0.8],
        input_perceptual_percentiles: [0.2, 0.5, 0.9],
        mapped_linear_percentiles: [0.08, 0.45, 0.85],
        mapped_perceptual_percentiles: [0.18, 0.55, 0.92],
    }
}

fn test_cache(output_dir: std::path::PathBuf, prophoto: Array3<f64>) -> InteractiveRenderCache {
    let auto_tone_params = ToneCurveParams {
        domain: ToneFitDomain::LinearLuminance,
        midpoint: 0.45,
        slope: 2.0,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let output_path = output_dir.join("output.tiff");
    InteractiveRenderCache {
        prophoto,
        auto_exposure_ev: 0.0,
        auto_tone_params,
        tone_fit_diagnostics: tone_diagnostics(),
        tone_color_protection: ToneColorProtection::default(),
        report: PipelineReport::new(),
        cli: test_cli(output_dir),
        output_path,
        run_started_at: SystemTime::now(),
        base_confidence: 1.0,
    }
}

fn mean_luminance(img: &Array3<f64>) -> f64 {
    let (height, width, _) = img.dim();
    let mut sum = 0.0;
    for y in 0..height {
        for x in 0..width {
            sum += 0.2880 * img[[y, x, 0]] + 0.7119 * img[[y, x, 1]] + 0.0001 * img[[y, x, 2]];
        }
    }
    sum / (height * width).max(1) as f64
}

#[test]
fn control_defaults_match_auto_tone_fit() {
    let tmp = TempDir::new().unwrap();
    let prophoto = Array3::<f64>::from_elem((4, 4, 3), 0.25);
    let cache = test_cache(tmp.path().join("output"), prophoto);

    let controls = cache.default_controls();

    assert_eq!(controls.exposure_ev, 0.0);
    assert_eq!(controls.midpoint, cache.auto_tone_params.midpoint);
    assert_eq!(controls.slope, cache.auto_tone_params.slope);
    assert_eq!(controls.toe_lift, cache.auto_tone_params.toe_lift);
    assert_eq!(controls.shoulder_max, cache.auto_tone_params.shoulder_max);
}

#[test]
fn exposure_ev_scales_render_luminance_without_changing_dimensions() {
    let tmp = TempDir::new().unwrap();
    let mut prophoto = Array3::<f64>::zeros((12, 16, 3));
    for y in 0..12 {
        for x in 0..16 {
            let value = 0.12 + 0.20 * (x as f64 / 15.0);
            for c in 0..3 {
                prophoto[[y, x, c]] = value;
            }
        }
    }
    let cache = test_cache(tmp.path().join("output"), prophoto);
    let base_controls = cache.default_controls();
    let brighter_controls = InteractiveRenderControls {
        exposure_ev: 1.0,
        ..base_controls
    };

    let base = interactive::render_interactive_image(&cache, &base_controls).image;
    let brighter = interactive::render_interactive_image(&cache, &brighter_controls).image;

    assert_eq!(base.dim(), brighter.dim());
    assert!(mean_luminance(&brighter) > mean_luminance(&base));
}

#[test]
fn reset_controls_restore_auto_values() {
    let tmp = TempDir::new().unwrap();
    let cache = test_cache(
        tmp.path().join("output"),
        Array3::<f64>::from_elem((4, 4, 3), 0.25),
    );
    let mut controls = cache.default_controls();
    controls.exposure_ev = 1.5;
    controls.midpoint += 0.1;
    assert_ne!(controls, cache.default_controls());

    controls = cache.default_controls();

    assert_eq!(controls, cache.default_controls());
}

#[test]
fn preview_downscale_preserves_aspect_ratio_and_clamps_rgb() {
    let tmp = TempDir::new().unwrap();
    let mut prophoto = Array3::<f64>::zeros((120, 240, 3));
    for y in 0..120 {
        for x in 0..240 {
            prophoto[[y, x, 0]] = -0.2 + x as f64 / 80.0;
            prophoto[[y, x, 1]] = y as f64 / 40.0;
            prophoto[[y, x, 2]] = 0.5;
        }
    }
    let cache = test_cache(tmp.path().join("output"), prophoto);

    let frame = interactive::render_preview_frame(
        &cache,
        &cache.default_controls(),
        100,
        100,
        PreviewQuality::High,
        "test",
    );

    assert_eq!(frame.width, 100);
    assert_eq!(frame.height, 50);
    assert_eq!(frame.pixels.len(), frame.width * frame.height);
    assert!(frame.pixels.iter().all(|pixel| *pixel <= 0x00ff_ffff));
}

#[test]
fn explicit_interactive_save_writes_output_and_report_metrics() {
    let tmp = TempDir::new().unwrap();
    let output_dir = tmp.path().join("output");
    let mut cache = test_cache(
        output_dir.clone(),
        Array3::<f64>::from_elem((8, 10, 3), 0.25),
    );
    cache.cli.write_review_sidecar = Some(output_dir.join("review-sidecar.json"));
    let controls = InteractiveRenderControls {
        exposure_ev: 0.5,
        ..cache.default_controls()
    };

    let report = scanstitch::pipeline::save_interactive_render(&cache, &controls).unwrap();

    assert!(output_dir.join("output.tiff").exists());
    assert!(output_dir.join("master_scene_referred.tiff").exists());
    assert!(output_dir.join("review_srgb.png").exists());
    assert!(output_dir.join("review-sidecar.json").exists());
    assert!(output_dir.join("report.json").exists());
    let tone_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "tone_mapping")
        .expect("tone_mapping phase");
    assert_eq!(
        tone_phase.metrics["interactive_controls_applied"],
        serde_json::json!(true)
    );
    assert_eq!(
        tone_phase.metrics["interactive_controls"]["exposure_ev"],
        serde_json::json!(0.5)
    );
    let save_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "save")
        .expect("save phase");
    assert!(save_phase.metrics["written_review_sidecar_sha256"].is_string());
}

#[test]
fn review_sidecar_roundtrip_applies_controls_and_reports_hash() {
    let tmp = TempDir::new().unwrap();
    let output_dir = tmp.path().join("output");
    let mut cache = test_cache(
        output_dir.clone(),
        Array3::<f64>::from_elem((8, 10, 3), 0.25),
    );
    let sidecar_path = tmp.path().join("review.json");
    let written_sidecar_path = output_dir.join("saved-review.json");
    let sidecar = scanstitch::interactive::ReviewSidecar {
        schema_version: scanstitch::interactive::REVIEW_SIDECAR_SCHEMA_VERSION,
        sidecar_type: "scanstitch_guided_review".to_string(),
        render_intent: Some("modern-clean".to_string()),
        controls: Some(InteractiveRenderControls {
            exposure_ev: 0.75,
            midpoint: 0.33,
            slope: 2.4,
            toe_lift: 0.01,
            shoulder_max: 0.98,
        }),
        marks: vec![scanstitch::interactive::ReviewMark {
            kind: "neutral_grey".to_string(),
            label: Some("synthetic".to_string()),
            point: Some([0.5, 0.5]),
            bounds: None,
            weight: Some(1.0),
            note: None,
        }],
        decisions: None,
    };
    std::fs::write(
        &sidecar_path,
        serde_json::to_string_pretty(&sidecar).unwrap(),
    )
    .unwrap();

    let loaded = scanstitch::interactive::load_review_sidecar(&sidecar_path).unwrap();
    let controls = scanstitch::interactive::controls_with_review_sidecar(
        cache.default_controls(),
        &cache.auto_tone_params,
        &loaded.sidecar,
    );
    assert_eq!(controls.exposure_ev, 0.75);
    assert_eq!(controls.midpoint, 0.33);

    cache.cli.review_sidecar = Some(sidecar_path);
    cache.cli.write_review_sidecar = Some(written_sidecar_path);
    let report = scanstitch::pipeline::save_interactive_render(&cache, &controls).unwrap();
    let save_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "save")
        .expect("save phase");

    assert_eq!(save_phase.metrics["review_sidecar_applied"], true);
    assert_eq!(save_phase.metrics["review_sidecar_mark_count"], 1);
    assert!(save_phase.metrics["review_sidecar_sha256"].is_string());
    assert!(save_phase.metrics["written_review_sidecar_sha256"].is_string());
}

#[test]
fn interactive_cache_writes_nothing_until_save_and_default_render_matches_batch() {
    let tmp = TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp1 =
        synthetic::film_negative_image(120, 240, 8, 24, [12000, 7000, 3000], [5000, 4000, 3500]);
    let comp2 =
        synthetic::film_negative_image(120, 240, 8, 24, [12000, 7000, 3000], [6000, 4500, 3200]);
    let path1 = input_dir.join("comp1.tiff");
    let path2 = input_dir.join("comp2.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp1, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp2, &path2).unwrap();

    let interactive_output_dir = tmp.path().join("interactive-output");
    let interactive_cli =
        pipeline_cli(path1.clone(), path2.clone(), interactive_output_dir.clone());
    let cache = scanstitch::pipeline::build_interactive_render_cache(&interactive_cli).unwrap();

    assert!(!interactive_output_dir.join("output.tiff").exists());
    assert!(!interactive_output_dir.join("report.json").exists());

    scanstitch::pipeline::save_interactive_render(&cache, &cache.default_controls()).unwrap();

    let batch_output_dir = tmp.path().join("batch-output");
    let batch_cli = pipeline_cli(path1, path2, batch_output_dir.clone());
    scanstitch::pipeline::run(&batch_cli).unwrap();

    let interactive_output =
        scanstitch::tiff_io::load_tiff_u16(&interactive_output_dir.join("output.tiff"), 16)
            .unwrap();
    let batch_output =
        scanstitch::tiff_io::load_tiff_u16(&batch_output_dir.join("output.tiff"), 16).unwrap();

    assert_eq!(interactive_output.image, batch_output.image);
}

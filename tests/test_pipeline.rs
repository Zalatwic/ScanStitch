mod common;
use common::synthetic;
use ndarray::s;
use tempfile::TempDir;

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

    let cli = scanstitch::cli::Cli {
        component1: path1,
        component2: path2,
        output_dir: output_dir.clone(),
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
    assert!(!report.phases.is_empty());
    assert!(output_dir.join("output.tiff").exists());
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
        output_dir,
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
        output_dir,
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
        output_dir,
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
        output_dir,
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
    assert!(
        colorspace_phase.metrics["render_input_source"].is_string(),
        "expected render input source diagnostic"
    );
    assert!(
        colorspace_phase.metrics["render_input_reason"].is_string(),
        "expected render input selection reason diagnostic"
    );
    assert!(
        colorspace_phase.metrics["direct_density_candidate_evaluated"].is_boolean(),
        "expected direct density candidate evaluation diagnostic"
    );
    assert!(
        colorspace_phase.metrics["ica_candidate"].is_object(),
        "expected ICA colorspace candidate diagnostics"
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

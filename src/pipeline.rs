use crate::base_detect;
use crate::border;
use crate::cli::Cli;
use crate::colorspace;
use crate::density;
use crate::frame_classify;
use crate::ica;
use crate::report::{PhaseReport, PipelineReport};
use crate::stitch::{self, StitchConfig};
use crate::tiff_io;
use crate::tonemap;

/// Run the full scanstitch pipeline.
pub fn run(cli: &Cli) -> Result<PipelineReport, Box<dyn std::error::Error>> {
    let mut report = PipelineReport::new();

    // Create output directory
    std::fs::create_dir_all(&cli.output_dir)?;

    // ── Phase 0: Load inputs ──────────────────────────────────────────
    log::info!("Loading component 1: {}", cli.component1.display());
    let img1 = tiff_io::load_tiff_u16(&cli.component1)?;
    log::info!("  Loaded: {}x{}", img1.shape()[1], img1.shape()[0]);

    log::info!("Loading component 2: {}", cli.component2.display());
    let img2 = tiff_io::load_tiff_u16(&cli.component2)?;
    log::info!("  Loaded: {}x{}", img2.shape()[1], img2.shape()[0]);

    report.add_phase(PhaseReport::ok(
        "load",
        1.0,
        serde_json::json!({
            "component1": cli.component1.to_string_lossy(),
            "component2": cli.component2.to_string_lossy(),
            "img1_shape": [img1.shape()[0], img1.shape()[1]],
            "img2_shape": [img2.shape()[0], img2.shape()[1]],
        }),
    ));

    // ── Phase 1: Border removal ───────────────────────────────────────
    log::info!("Phase 1: Border removal (safety_margin=2)");
    let (comp1_cropped, top1, bot1) = border::remove_borders(&img1, 2);
    let (comp2_cropped, top2, bot2) = border::remove_borders(&img2, 2);
    log::info!(
        "  comp1: removed {} top, {} bottom -> {}x{}",
        top1, bot1,
        comp1_cropped.shape()[1], comp1_cropped.shape()[0]
    );
    log::info!(
        "  comp2: removed {} top, {} bottom -> {}x{}",
        top2, bot2,
        comp2_cropped.shape()[1], comp2_cropped.shape()[0]
    );

    if cli.debug {
        tiff_io::save_tiff_u16(&comp1_cropped, &cli.output_dir.join("comp1_cropped.tiff"))?;
        tiff_io::save_tiff_u16(&comp2_cropped, &cli.output_dir.join("comp2_cropped.tiff"))?;
    }

    report.add_phase(PhaseReport::ok(
        "border_removal",
        1.0,
        serde_json::json!({
            "comp1_top": top1,
            "comp1_bottom": bot1,
            "comp1_shape": [comp1_cropped.shape()[0], comp1_cropped.shape()[1]],
            "comp2_top": top2,
            "comp2_bottom": bot2,
            "comp2_shape": [comp2_cropped.shape()[0], comp2_cropped.shape()[1]],
        }),
    ));

    // ── Phase 2: Base detection, classification, conditional stitch ──
    log::info!("Phase 2: Base detection + classification");
    let det1 = base_detect::detect_film_base(&comp1_cropped);
    let det2 = base_detect::detect_film_base(&comp2_cropped);
    let base_color = det1.base_color;
    log::info!(
        "  Base color: [{:.0}, {:.0}, {:.0}]",
        base_color[0], base_color[1], base_color[2]
    );

    let class1 = frame_classify::classify(&det1);
    let class2 = frame_classify::classify(&det2);
    log::info!("  comp1 class: {:?}, comp2 class: {:?}", class1, class2);

    let should_stitch = frame_classify::should_attempt_stitch(
        &class1,
        &class2,
        cli.force_stitch,
        cli.force_no_stitch,
    );

    let working_image = if should_stitch {
        log::info!("  Attempting stitch...");
        let stitch_result = stitch::stitch_components(
            &comp1_cropped,
            &comp2_cropped,
            &StitchConfig::default(),
        );
        report.add_phase(stitch_result.report.clone());

        match stitch_result.result {
            Some(ref stitched) => {
                log::info!(
                    "  Stitch succeeded: {}x{}, NCC={:.3}",
                    stitched.shape()[1], stitched.shape()[0], stitch_result.ncc_score
                );
                if cli.debug {
                    tiff_io::save_tiff_u16(stitched, &cli.output_dir.join("stitched.tiff"))?;
                }
                stitched.clone()
            }
            None => {
                log::warn!("  Stitch failed, falling back to comp1_cropped");
                comp1_cropped.clone()
            }
        }
    } else {
        log::info!("  Skipping stitch (force_no_stitch={}, should_stitch=false)", cli.force_no_stitch);
        report.add_phase(PhaseReport::ok(
            "stitch",
            1.0,
            serde_json::json!({ "action": "skipped", "reason": "no_stitch" }),
        ));
        comp1_cropped.clone()
    };

    report.add_phase(PhaseReport::ok(
        "base_detect_classify",
        (det1.left_confidence + det1.right_confidence) / 2.0,
        serde_json::json!({
            "base_color": base_color,
            "class1": format!("{:?}", class1),
            "class2": format!("{:?}", class2),
            "should_stitch": should_stitch,
            "working_shape": [working_image.shape()[0], working_image.shape()[1]],
        }),
    ));

    // ── Phase 3: Density-domain inversion ─────────────────────────────
    log::info!("Phase 3: Density inversion (bit_depth={})", cli.bit_depth);
    let positive = density::phase3_invert(&working_image, &base_color, cli.bit_depth);
    log::info!(
        "  Output shape: {}x{}, range [0,1]",
        positive.shape()[1], positive.shape()[0]
    );

    if cli.debug {
        tiff_io::save_tiff_f64(&positive, &cli.output_dir.join("phase3_positive.tiff"))?;
    }

    report.add_phase(PhaseReport::ok(
        "density_inversion",
        1.0,
        serde_json::json!({
            "base_color": base_color,
            "bit_depth": cli.bit_depth,
            "output_shape": [positive.shape()[0], positive.shape()[1]],
        }),
    ));

    // ── Phase 4: FastICA ──────────────────────────────────────────────
    log::info!(
        "Phase 4: FastICA (max_iter={}, tol={})",
        cli.ica_max_iter, cli.ica_tol
    );
    let ica_result = ica::run_fastica(&positive, cli.ica_max_iter, cli.ica_tol);
    log::info!(
        "  Converged: {}, iterations: {}",
        ica_result.converged, ica_result.iterations
    );

    // Normalize ICA output to [0, 1] per channel
    let separated = normalize_to_unit(&ica_result.separated);

    if cli.debug {
        tiff_io::save_tiff_f64(&separated, &cli.output_dir.join("phase4_ica.tiff"))?;
    }

    let mut ica_phase = PhaseReport::ok(
        "fastica",
        if ica_result.converged { 1.0 } else { 0.5 },
        serde_json::json!({
            "converged": ica_result.converged,
            "iterations": ica_result.iterations,
            "permutation": ica_result.permutation,
            "signs": ica_result.signs,
        }),
    );
    if !ica_result.converged {
        ica_phase.warnings.push(format!(
            "ICA did not converge within {} iterations",
            cli.ica_max_iter
        ));
    }
    report.add_phase(ica_phase);

    // ── Phase 4.6: Color space mapping ────────────────────────────────
    log::info!("Phase 4.6: Color space mapping to ProPhoto D50");
    let prophoto = colorspace::map_to_prophoto_d50(&separated);

    if cli.debug {
        tiff_io::save_tiff_f64(&prophoto, &cli.output_dir.join("phase46_prophoto.tiff"))?;
    }

    report.add_phase(PhaseReport::ok(
        "colorspace_mapping",
        1.0,
        serde_json::json!({
            "target": "ProPhoto_D50",
            "output_shape": [prophoto.shape()[0], prophoto.shape()[1]],
        }),
    ));

    // ── Phase 5: Tone mapping ─────────────────────────────────────────
    log::info!("Phase 5: Tone mapping");
    let tone_params = tonemap::fit_tone_params(&prophoto);
    let tonemapped = tonemap::apply_tonemap_with_params(&prophoto, &tone_params);
    log::info!(
        "  midpoint={:.3}, slope={:.3}, toe={:.4}, shoulder={:.4}",
        tone_params.midpoint, tone_params.slope, tone_params.toe_lift, tone_params.shoulder_max
    );

    if cli.debug {
        // Save LUT as JSON: 256 evenly-spaced [input, output] pairs
        let lut: Vec<[f64; 2]> = (0..=255)
            .map(|i| {
                let x = i as f64 / 255.0;
                let y = tonemap::apply_tone_curve(x, &tone_params);
                [x, y]
            })
            .collect();
        let lut_json = serde_json::to_string_pretty(&lut)?;
        std::fs::write(cli.output_dir.join("tone_curve_lut.json"), lut_json)?;
    }

    report.add_phase(PhaseReport::ok(
        "tone_mapping",
        1.0,
        serde_json::json!({
            "midpoint": tone_params.midpoint,
            "slope": tone_params.slope,
            "toe_lift": tone_params.toe_lift,
            "shoulder_max": tone_params.shoulder_max,
        }),
    ));

    // ── Save final output ─────────────────────────────────────────────
    let output_path = cli.output_dir.join("output.tiff");
    log::info!("Saving final output to {}", output_path.display());
    tiff_io::save_tiff_f64(&tonemapped, &output_path)?;

    report.add_phase(PhaseReport::ok(
        "save",
        1.0,
        serde_json::json!({
            "output_path": output_path.to_string_lossy(),
            "output_shape": [tonemapped.shape()[0], tonemapped.shape()[1]],
        }),
    ));

    log::info!("Pipeline complete. {} phases recorded.", report.phases.len());
    Ok(report)
}

/// Normalize an f64 image to [0, 1] per channel using min-max scaling.
fn normalize_to_unit(img: &ndarray::Array3<f64>) -> ndarray::Array3<f64> {
    let (h, w, _) = img.dim();
    let mut out = img.clone();

    for c in 0..3 {
        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;
        for y in 0..h {
            for x in 0..w {
                let v = img[[y, x, c]];
                if v < min_val { min_val = v; }
                if v > max_val { max_val = v; }
            }
        }
        let range = (max_val - min_val).max(1e-12);
        for y in 0..h {
            for x in 0..w {
                out[[y, x, c]] = ((img[[y, x, c]] - min_val) / range).clamp(0.0, 1.0);
            }
        }
    }

    out
}

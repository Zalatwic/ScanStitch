use crate::base_detect;
use crate::border;
use crate::cli::Cli;
use crate::colorspace;
use crate::density;
use crate::frame_classify;
use crate::ica;
use crate::report::{PhaseReport, PipelineReport};
use crate::stitch::{self, StitchConfig, TransformMode};
use crate::tiff_io;
use crate::tonemap;
use ndarray::Array3;
use std::path::Path;
use std::time::Instant;

const BASE_CONFIDENCE_WARN: f64 = 0.30;
const BASE_CONFIDENCE_FALLBACK: f64 = 0.15;

fn save_partial_report(
    report: &PipelineReport,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    report.save(&output_dir.join("report.json"))
}

fn record_phase(
    report: &mut PipelineReport,
    output_dir: &Path,
    phase: PhaseReport,
    phase_start: Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    report.add_phase(phase.with_duration_ms(phase_start.elapsed().as_millis() as u64));
    save_partial_report(report, output_dir)
}

fn record_failure(
    report: &mut PipelineReport,
    output_dir: &Path,
    name: &str,
    message: String,
    phase_start: Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    report.add_phase(
        PhaseReport::fail(name, &message)
            .with_duration_ms(phase_start.elapsed().as_millis() as u64),
    );
    save_partial_report(report, output_dir)
}

fn downstream_base_quality_factor(base_confidence: f64) -> f64 {
    let t = (base_confidence / BASE_CONFIDENCE_WARN).clamp(0.0, 1.0);
    (0.25 + 0.75 * t).clamp(0.25, 1.0)
}

fn apply_downstream_base_confidence_limit(phase: &mut PhaseReport, base_confidence: f64) -> f64 {
    let factor = downstream_base_quality_factor(base_confidence);
    phase.confidence = phase.confidence.min(factor);
    if let serde_json::Value::Object(metrics) = &mut phase.metrics {
        metrics.insert(
            "input_base_confidence".to_string(),
            serde_json::json!(base_confidence),
        );
        metrics.insert(
            "base_confidence_quality_factor".to_string(),
            serde_json::json!(factor),
        );
        metrics.insert(
            "confidence_limited_by_base_estimate".to_string(),
            serde_json::json!(factor < 0.999),
        );
    }
    factor
}

fn mask_grid_to_image(mask: &[Vec<f64>], width: usize, height: usize) -> Array3<u16> {
    let mut out = Array3::<u16>::zeros((height.max(1), width.max(1), 3));
    if mask.is_empty() || width == 0 || height == 0 {
        return out;
    }

    let rows = mask.len();
    let cols = mask[0].len().max(1);
    for y in 0..height {
        let row = (y * rows / height).min(rows - 1);
        for x in 0..width {
            let col = (x * cols / width).min(cols - 1);
            let value = mask
                .get(row)
                .and_then(|r| r.get(col))
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let encoded = (value * u16::MAX as f64).round() as u16;
            out[[y, x, 0]] = encoded;
            out[[y, x, 1]] = encoded;
            out[[y, x, 2]] = encoded;
        }
    }
    out
}

fn column_mask_to_image(mask: &[f64], height: usize) -> Array3<u16> {
    let width = mask.len().max(1);
    let mut out = Array3::<u16>::zeros((height.max(1), width, 3));
    for x in 0..mask.len() {
        let value = mask[x].clamp(0.0, 1.0);
        let encoded = (value * u16::MAX as f64).round() as u16;
        for y in 0..height.max(1) {
            out[[y, x, 0]] = 0;
            out[[y, x, 1]] = encoded;
            out[[y, x, 2]] = encoded;
        }
    }
    out
}

fn tiff_load_metrics_json(diag: &tiff_io::TiffLoadDiagnostics) -> serde_json::Value {
    serde_json::json!({
        "width": diag.width,
        "height": diag.height,
        "color_type": diag.color_type,
        "source_bits_per_sample": diag.source_bits_per_sample,
        "source_channel_count": diag.source_channel_count,
        "source_has_alpha": diag.source_has_alpha,
        "working_bit_depth": diag.working_range.target_bit_depth,
        "range_transform": diag.working_range.transform.as_str(),
        "scale_factor": diag.working_range.scale_factor,
        "source_min": diag.working_range.source_min,
        "source_max": diag.working_range.source_max,
        "working_min": diag.working_range.working_min,
        "working_max": diag.working_range.working_max,
        "note": diag.working_range.note,
    })
}

fn tiff_load_warnings(label: &str, diag: &tiff_io::TiffLoadDiagnostics) -> Vec<String> {
    let mut warnings = Vec::new();
    if diag.source_has_alpha {
        warnings.push(format!(
            "{} TIFF decoded as {}; alpha was discarded during load",
            label, diag.color_type
        ));
    }
    if diag.source_bits_per_sample != diag.working_range.target_bit_depth
        || diag.working_range.transform != tiff_io::WorkingRangeTransform::None
    {
        let note = diag
            .working_range
            .note
            .as_deref()
            .unwrap_or("used the on-disk sample range without rescaling");
        warnings.push(format!(
            "{} TIFF decoded as {} and {}",
            label, diag.color_type, note
        ));
    }
    warnings
}

fn colorspace_candidate_metrics_json(
    diagnostics: &colorspace::ColorspaceDiagnostics,
) -> serde_json::Value {
    let (image_matrix_low_clip_max, image_matrix_low_clip_total) =
        colorspace::image_matrix_low_clip_summary(diagnostics);
    serde_json::json!({
        "mapping_strategy": diagnostics.mapping_strategy,
        "condition_number": diagnostics.condition_number,
        "neutral_pixel_count": diagnostics.neutral_pixel_count,
        "regularization_lambda": diagnostics.regularization_lambda,
        "channel_anchor_counts": diagnostics.channel_anchor_counts,
        "channel_anchor_min_count": diagnostics.channel_anchor_min_count,
        "channel_anchor_low_support_threshold": diagnostics.channel_anchor_low_support_threshold,
        "channel_anchor_low_support": diagnostics.channel_anchor_low_support,
        "weak_anchor_fallback_used": diagnostics.weak_anchor_fallback_used,
        "weak_anchor_fallback_reason": diagnostics.weak_anchor_fallback_reason.as_deref(),
        "gamut_fallback_used": diagnostics.gamut_fallback_used,
        "gamut_fallback_reason": diagnostics.gamut_fallback_reason.as_deref(),
        "image_matrix_pre_scale_clipped_low_ratio": diagnostics.image_matrix_pre_scale_clipped_low_ratio,
        "image_matrix_pre_scale_clipped_low_max": image_matrix_low_clip_max,
        "image_matrix_pre_scale_clipped_low_total": image_matrix_low_clip_total,
        "image_matrix_pre_scale_clipped_high_ratio": diagnostics.image_matrix_pre_scale_clipped_high_ratio,
        "image_matrix_exposure_scale": diagnostics.image_matrix_exposure_scale,
        "pre_scale_clipped_high_ratio": diagnostics.pre_scale_clipped_high_ratio,
        "pre_scale_clipped_low_ratio": diagnostics.pre_scale_clipped_low_ratio,
        "post_scale_clipped_high_ratio": diagnostics.post_scale_clipped_high_ratio,
        "post_scale_clipped_low_ratio": diagnostics.post_scale_clipped_low_ratio,
        "exposure_scale": diagnostics.exposure_scale,
    })
}

fn render_band_metrics_json(diagnostics: &tonemap::RenderBandDiagnostics) -> serde_json::Value {
    serde_json::json!({
        "pixel_count": diagnostics.pixel_count,
        "luminance_percentiles": diagnostics.luminance_percentiles,
        "saturation_median": diagnostics.saturation_median,
        "saturation_p95": diagnostics.saturation_p95,
        "rgb_median": diagnostics.rgb_median,
    })
}

fn save_base_debug_artifacts(
    prefix: &str,
    output_dir: &Path,
    img: &Array3<u16>,
    detection: &base_detect::BaseDetection,
) -> Result<(), Box<dyn std::error::Error>> {
    let (h, w, _) = img.dim();
    let left_mask = mask_grid_to_image(&detection.left_base_mask, detection.strip_width.max(1), h);
    let right_mask =
        mask_grid_to_image(&detection.right_base_mask, detection.strip_width.max(1), h);
    tiff_io::save_tiff_u16(
        &left_mask,
        &output_dir.join(format!("{}_base_mask_left.tiff", prefix)),
    )?;
    tiff_io::save_tiff_u16(
        &right_mask,
        &output_dir.join(format!("{}_base_mask_right.tiff", prefix)),
    )?;

    let vertical = base_detect::detect_vertical_base_regions(img);
    let columns = column_mask_to_image(&vertical.column_mask, h);
    tiff_io::save_tiff_u16(
        &columns,
        &output_dir.join(format!("{}_base_columns.tiff", prefix)),
    )?;

    if w > 0 {
        let mut overlay = img.clone();
        let strip_w = detection.strip_width.min(w);
        for y in 0..h {
            for x in 0..strip_w {
                let left_value = left_mask[[y, x, 1]] as f64 / u16::MAX as f64;
                overlay[[y, x, 1]] = ((overlay[[y, x, 1]] as f64) * (1.0 - 0.5 * left_value)
                    + u16::MAX as f64 * 0.5 * left_value)
                    .round()
                    .clamp(0.0, u16::MAX as f64) as u16;
                let rx = w - strip_w + x;
                if rx < w {
                    let right_value = right_mask[[y, x, 1]] as f64 / u16::MAX as f64;
                    overlay[[y, rx, 0]] = ((overlay[[y, rx, 0]] as f64) * (1.0 - 0.5 * right_value)
                        + u16::MAX as f64 * 0.5 * right_value)
                        .round()
                        .clamp(0.0, u16::MAX as f64)
                        as u16;
                }
            }
        }
        tiff_io::save_tiff_u16(
            &overlay,
            &output_dir.join(format!("{}_base_overlay.tiff", prefix)),
        )?;
    }

    Ok(())
}

/// Run the full scanstitch pipeline.
pub fn run(cli: &Cli) -> Result<PipelineReport, Box<dyn std::error::Error>> {
    let mut report = PipelineReport::new();
    std::fs::create_dir_all(&cli.output_dir)?;

    let load_start = Instant::now();
    log::info!("Loading component 1: {}", cli.component1.display());
    let load1 = match tiff_io::load_tiff_u16(&cli.component1, cli.bit_depth) {
        Ok(img) => img,
        Err(err) => {
            record_failure(
                &mut report,
                &cli.output_dir,
                "load",
                format!("failed to load component1: {}", err),
                load_start,
            )?;
            return Err(err);
        }
    };
    log::info!("Loading component 2: {}", cli.component2.display());
    let load2 = match tiff_io::load_tiff_u16(&cli.component2, cli.bit_depth) {
        Ok(img) => img,
        Err(err) => {
            record_failure(
                &mut report,
                &cli.output_dir,
                "load",
                format!("failed to load component2: {}", err),
                load_start,
            )?;
            return Err(err);
        }
    };
    let tiff_io::LoadedTiff {
        image: img1,
        diagnostics: load1_diag,
    } = load1;
    let tiff_io::LoadedTiff {
        image: img2,
        diagnostics: load2_diag,
    } = load2;
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let mut phase = PhaseReport::ok(
                "load",
                1.0,
                serde_json::json!({
                    "component1": cli.component1.to_string_lossy(),
                    "component2": cli.component2.to_string_lossy(),
                    "img1_shape": [img1.shape()[0], img1.shape()[1]],
                    "img2_shape": [img2.shape()[0], img2.shape()[1]],
                    "component1_decode": tiff_load_metrics_json(&load1_diag),
                    "component2_decode": tiff_load_metrics_json(&load2_diag),
                }),
            );
            phase
                .warnings
                .extend(tiff_load_warnings("component 1", &load1_diag));
            phase
                .warnings
                .extend(tiff_load_warnings("component 2", &load2_diag));
            phase
        },
        load_start,
    )?;

    let border_start = Instant::now();
    let border1 = border::remove_borders_with_diagnostics(&img1, 2);
    let border2 = border::remove_borders_with_diagnostics(&img2, 2);
    let comp1_cropped = border1.cropped;
    let comp2_cropped = border2.cropped;
    if cli.debug {
        if let Err(err) =
            tiff_io::save_tiff_u16(&comp1_cropped, &cli.output_dir.join("comp1_cropped.tiff"))
        {
            record_failure(
                &mut report,
                &cli.output_dir,
                "border_removal",
                format!("failed to save comp1_cropped debug image: {}", err),
                border_start,
            )?;
            return Err(err);
        }
        if let Err(err) =
            tiff_io::save_tiff_u16(&comp2_cropped, &cli.output_dir.join("comp2_cropped.tiff"))
        {
            record_failure(
                &mut report,
                &cli.output_dir,
                "border_removal",
                format!("failed to save comp2_cropped debug image: {}", err),
                border_start,
            )?;
            return Err(err);
        }
    }
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let mut phase = PhaseReport::ok(
                "border_removal",
                1.0,
                serde_json::json!({
                    "component1": {
                        "top_removed": border1.diagnostics.top_removed,
                        "bottom_removed": border1.diagnostics.bottom_removed,
                        "content_lum": border1.diagnostics.content_lum,
                        "content_mad": border1.diagnostics.content_mad,
                        "top_strong_rows": border1.diagnostics.top_strong_rows,
                        "bottom_strong_rows": border1.diagnostics.bottom_strong_rows,
                        "top_peak_lum_gap": border1.diagnostics.top_peak_lum_gap,
                        "bottom_peak_lum_gap": border1.diagnostics.bottom_peak_lum_gap,
                        "top_peak_row_delta": border1.diagnostics.top_peak_row_delta,
                        "bottom_peak_row_delta": border1.diagnostics.bottom_peak_row_delta,
                        "dead_zone_detected": border1.diagnostics.dead_zone_detected,
                    },
                    "comp1_shape": [comp1_cropped.shape()[0], comp1_cropped.shape()[1]],
                    "component2": {
                        "top_removed": border2.diagnostics.top_removed,
                        "bottom_removed": border2.diagnostics.bottom_removed,
                        "content_lum": border2.diagnostics.content_lum,
                        "content_mad": border2.diagnostics.content_mad,
                        "top_strong_rows": border2.diagnostics.top_strong_rows,
                        "bottom_strong_rows": border2.diagnostics.bottom_strong_rows,
                        "top_peak_lum_gap": border2.diagnostics.top_peak_lum_gap,
                        "bottom_peak_lum_gap": border2.diagnostics.bottom_peak_lum_gap,
                        "top_peak_row_delta": border2.diagnostics.top_peak_row_delta,
                        "bottom_peak_row_delta": border2.diagnostics.bottom_peak_row_delta,
                        "dead_zone_detected": border2.diagnostics.dead_zone_detected,
                    },
                    "comp2_shape": [comp2_cropped.shape()[0], comp2_cropped.shape()[1]],
                }),
            );
            phase
                .warnings
                .extend(border1.diagnostics.warnings.iter().cloned());
            phase
                .warnings
                .extend(border2.diagnostics.warnings.iter().cloned());
            phase
        },
        border_start,
    )?;

    let classify_start = Instant::now();
    let det1 = base_detect::detect_film_base(&comp1_cropped);
    let det2 = base_detect::detect_film_base(&comp2_cropped);
    let analysis1 = frame_classify::analyze_component(&comp1_cropped, &det1);
    let analysis2 = frame_classify::analyze_component(&comp2_cropped, &det2);
    let stitch_decision = frame_classify::decide_stitch_attempt(
        &analysis1,
        &analysis2,
        cli.force_stitch,
        cli.force_no_stitch,
    );
    if cli.debug {
        save_base_debug_artifacts("comp1", &cli.output_dir, &comp1_cropped, &det1)?;
        save_base_debug_artifacts("comp2", &cli.output_dir, &comp2_cropped, &det2)?;
    }
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let mut phase = PhaseReport::ok(
                "base_detect_classify",
                analysis1.confidence.max(analysis2.confidence),
                serde_json::json!({
                    "component1": {
                        "class": format!("{:?}", analysis1.class),
                        "confidence": analysis1.confidence,
                        "base_color": det1.base_color,
                        "base_strip_width": det1.strip_width,
                        "base_left_confidence": det1.left_confidence,
                        "base_right_confidence": det1.right_confidence,
                        "left_edge_confidence": analysis1.left_edge_confidence,
                        "right_edge_confidence": analysis1.right_edge_confidence,
                        "content_span": [analysis1.content_span.0, analysis1.content_span.1],
                        "content_fraction": analysis1.content_fraction,
                        "activity_score": analysis1.activity_score,
                        "component_score": analysis1.component_score,
                        "internal_base_regions": analysis1.internal_base_regions.iter().map(|r| serde_json::json!({
                            "x_start": r.x_start,
                            "x_end": r.x_end,
                            "confidence": r.confidence,
                            "mad_lum": r.mad_lum,
                        })).collect::<Vec<_>>(),
                    },
                    "component2": {
                        "class": format!("{:?}", analysis2.class),
                        "confidence": analysis2.confidence,
                        "base_color": det2.base_color,
                        "base_strip_width": det2.strip_width,
                        "base_left_confidence": det2.left_confidence,
                        "base_right_confidence": det2.right_confidence,
                        "left_edge_confidence": analysis2.left_edge_confidence,
                        "right_edge_confidence": analysis2.right_edge_confidence,
                        "content_span": [analysis2.content_span.0, analysis2.content_span.1],
                        "content_fraction": analysis2.content_fraction,
                        "activity_score": analysis2.activity_score,
                        "component_score": analysis2.component_score,
                        "internal_base_regions": analysis2.internal_base_regions.iter().map(|r| serde_json::json!({
                            "x_start": r.x_start,
                            "x_end": r.x_end,
                            "confidence": r.confidence,
                            "mad_lum": r.mad_lum,
                        })).collect::<Vec<_>>(),
                    },
                    "stitch_decision": {
                        "disposition": format!("{:?}", stitch_decision.disposition),
                        "reason": stitch_decision.reason.clone(),
                        "confidence": stitch_decision.confidence,
                    },
                }),
            );
            let comp1_base_conf = det1.left_confidence.max(det1.right_confidence);
            let comp2_base_conf = det2.left_confidence.max(det2.right_confidence);
            if comp1_base_conf < BASE_CONFIDENCE_WARN {
                phase.warnings.push(format!(
                    "component 1 base confidence is low ({:.3}); classification should be treated as provisional",
                    comp1_base_conf
                ));
            }
            if comp2_base_conf < BASE_CONFIDENCE_WARN {
                phase.warnings.push(format!(
                    "component 2 base confidence is low ({:.3}); classification should be treated as provisional",
                    comp2_base_conf
                ));
            }
            phase
        },
        classify_start,
    )?;

    let fallback_index = frame_classify::best_single_component_index(&analysis1, &analysis2);
    let fallback_component = if fallback_index == 0 {
        comp1_cropped.clone()
    } else {
        comp2_cropped.clone()
    };

    let stitch_start = Instant::now();
    let should_score_stitch = stitch_decision.should_score();
    let (working_image, used_stitched_working_image) = if should_score_stitch {
        let transform_mode = TransformMode::from_cli(&cli.transform)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let stitch_config = StitchConfig {
            transform_mode,
            use_opencv: cli.use_opencv,
            input_bit_depth: cli.bit_depth,
            target_width: Some(comp1_cropped.shape()[1].max(comp2_cropped.shape()[1])),
            target_height: Some(comp1_cropped.shape()[0].max(comp2_cropped.shape()[0])),
            debug_dir: if cli.debug {
                Some(cli.output_dir.clone())
            } else {
                None
            },
            ..StitchConfig::default()
        };
        let stitch_result =
            stitch::stitch_components(&comp1_cropped, &comp2_cropped, &stitch_config);
        let stitched_image = stitch_result.result.clone();
        record_phase(
            &mut report,
            &cli.output_dir,
            stitch_result.report,
            stitch_start,
        )?;

        match stitched_image {
            Some(stitched) => {
                if cli.debug {
                    tiff_io::save_tiff_u16(&stitched, &cli.output_dir.join("stitched.tiff"))?;
                }
                (stitched, true)
            }
            None => (fallback_component.clone(), false),
        }
    } else {
        record_phase(
            &mut report,
            &cli.output_dir,
            PhaseReport::ok(
                "stitch",
                stitch_decision.confidence,
                serde_json::json!({
                    "decision": "skipped_pre_score",
                    "reason": stitch_decision.reason.clone(),
                    "disposition": format!("{:?}", stitch_decision.disposition),
                    "fallback_component": fallback_index + 1,
                }),
            ),
            stitch_start,
        )?;
        (fallback_component.clone(), false)
    };

    let working_start = Instant::now();
    let det_final = base_detect::detect_film_base(&working_image);
    let raw_base_confidence = det_final.left_confidence.max(det_final.right_confidence);
    let base_reconciliation = if used_stitched_working_image {
        base_detect::reconcile_stitched_base_estimate(
            &det_final,
            &[det1.base_color, det2.base_color],
        )
    } else {
        base_detect::BaseReconciliation {
            base_color: det_final.base_color,
            confidence: raw_base_confidence,
            source: "working_edges",
            reason: "working image was not produced from a stitched crop; using the direct edge-base estimate".to_string(),
            raw_working_base_color: det_final.base_color,
            raw_working_confidence: raw_base_confidence,
            component_consensus_base_color: None,
            component_consensus_confidence: None,
            component_consensus_relative_spread: None,
            working_vs_consensus_relative_delta: None,
            edge_balance_ratio: if raw_base_confidence <= 1e-6 {
                1.0
            } else {
                (det_final.left_confidence.min(det_final.right_confidence) / raw_base_confidence)
                    .clamp(0.0, 1.0)
            },
        }
    };
    let base_color = base_reconciliation.base_color;
    if cli.debug {
        save_base_debug_artifacts("working", &cli.output_dir, &working_image, &det_final)?;
    }
    let base_confidence = base_reconciliation.confidence;
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let mut phase = PhaseReport::ok(
                "working_image_select",
                base_confidence,
                serde_json::json!({
                    "working_shape": [working_image.shape()[0], working_image.shape()[1]],
                    "fallback_component": fallback_index + 1,
                    "used_stitched_working_image": used_stitched_working_image,
                    "base_color": base_color,
                    "raw_base_color": base_reconciliation.raw_working_base_color,
                    "raw_base_confidence": base_reconciliation.raw_working_confidence,
                    "base_estimate_source": base_reconciliation.source,
                    "base_estimate_reason": base_reconciliation.reason,
                    "component_consensus_base_color": base_reconciliation.component_consensus_base_color,
                    "component_consensus_confidence": base_reconciliation.component_consensus_confidence,
                    "component_consensus_relative_spread": base_reconciliation.component_consensus_relative_spread,
                    "working_vs_component_consensus_relative_delta": base_reconciliation.working_vs_consensus_relative_delta,
                    "edge_balance_ratio": base_reconciliation.edge_balance_ratio,
                    "left_confidence": det_final.left_confidence,
                    "right_confidence": det_final.right_confidence,
                    "used_low_confidence_fallback": base_confidence < BASE_CONFIDENCE_FALLBACK,
                }),
            );
            if base_reconciliation.source != "working_edges" {
                phase.warnings.push(format!(
                    "working-image crop edges disagreed with the pre-stitch base prior; {}",
                    base_reconciliation.reason
                ));
            }
            if base_confidence < BASE_CONFIDENCE_FALLBACK {
                phase.warnings.push(format!(
                    "working image base confidence is very low ({:.3}); density inversion is using a fallback-quality base estimate",
                    base_confidence
                ));
            } else if base_confidence < BASE_CONFIDENCE_WARN {
                phase.warnings.push(format!(
                    "working image base confidence is low ({:.3}); downstream density/color phases should be treated cautiously",
                    base_confidence
                ));
            }
            phase
        },
        working_start,
    )?;

    let density_start = Instant::now();
    let density_result =
        density::phase3_invert_with_diagnostics(&working_image, &base_color, cli.bit_depth);
    let positive = density_result.positive_density;
    if cli.debug {
        tiff_io::save_tiff_f64(&positive, &cli.output_dir.join("phase3_positive.tiff"))?;
        let linear_diag = density::linear_division_diagnostic_image(
            &working_image,
            &base_color,
            cli.bit_depth,
            &std::array::from_fn(|c| {
                density_result.diagnostics.linear_division[c].robust_high_percentile
            }),
        );
        tiff_io::save_tiff_f64(
            &linear_diag,
            &cli.output_dir.join("phase3_linear_division.tiff"),
        )?;
    }
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let mut phase = PhaseReport::ok(
                "density_inversion",
                base_confidence,
                serde_json::json!({
                    "base_color": base_color,
                    "base_estimate_source": base_reconciliation.source,
                    "bit_depth": cli.bit_depth,
                    "base_transmittance": density_result.diagnostics.base_transmittance,
                    "base_density": density_result.diagnostics.base_density,
                    "robust_d_max": density_result.diagnostics.robust_d_max,
                    "exact_d_max": density_result.diagnostics.exact_d_max,
                    "d_max_percentile": density_result.diagnostics.d_max_percentile,
                    "histogram_bins": density_result.diagnostics.histogram_bins,
                    "clamped_to_zero": density_result.diagnostics.clamped_to_zero,
                    "epsilon_t": density_result.diagnostics.epsilon_t,
                    "density_stats": density_result
                        .diagnostics
                        .density_stats
                        .iter()
                        .map(|stats| serde_json::json!({
                            "min_value": stats.min_value,
                            "max_value": stats.max_value,
                            "high_percentile": stats.high_percentile,
                        }))
                        .collect::<Vec<_>>(),
                    "linear_division": density_result
                        .diagnostics
                        .linear_division
                        .iter()
                        .map(|stats| serde_json::json!({
                            "exact_max": stats.exact_max,
                            "robust_high_percentile": stats.robust_high_percentile,
                            "percentile": stats.percentile,
                        }))
                        .collect::<Vec<_>>(),
                    "output_shape": [positive.shape()[0], positive.shape()[1]],
                }),
            );
            apply_downstream_base_confidence_limit(&mut phase, base_confidence);
            if base_confidence < BASE_CONFIDENCE_WARN {
                phase.warnings.push(format!(
                    "density inversion proceeded with low base confidence ({:.3})",
                    base_confidence
                ));
            }
            phase
        },
        density_start,
    )?;

    let ica_start = Instant::now();
    let ica_result = ica::run_fastica(&positive, cli.ica_max_iter, cli.ica_tol);
    let separated = ica::normalize_separated_density_channels(&ica_result.separated);
    let transmittance = density::density_image_to_transmittance(&separated.normalized_density);
    if cli.debug {
        tiff_io::save_tiff_f64(
            &separated.normalized_density,
            &cli.output_dir.join("phase4_ica.tiff"),
        )?;
    }
    let mut ica_phase = PhaseReport::ok(
        "fastica",
        if ica_result.converged { 1.0 } else { 0.5 },
        serde_json::json!({
            "converged": ica_result.converged,
            "iterations": ica_result.iterations,
            "permutation": ica_result.permutation,
            "signs": ica_result.signs,
            "normalization": {
                "histogram_bins": separated.histogram_bins,
                "low_percentile": separated.low_percentile,
                "high_percentile": separated.high_percentile,
                "channels": separated.channel_stats.iter().map(|stats| serde_json::json!({
                    "min_value": stats.min_value,
                    "max_value": stats.max_value,
                    "low_percentile_value": stats.low_percentile_value,
                    "high_percentile_value": stats.high_percentile_value,
                    "clipped_low": stats.clipped_low,
                    "clipped_high": stats.clipped_high,
                })).collect::<Vec<_>>(),
            },
            "sign_resolution": "skewness-based deterministic sign flip after permutation resolution",
        }),
    );
    let base_quality_factor =
        apply_downstream_base_confidence_limit(&mut ica_phase, base_confidence);
    if !ica_result.converged {
        ica_phase.warnings.push(format!(
            "ICA did not converge within {} iterations",
            cli.ica_max_iter
        ));
    }
    if base_confidence < BASE_CONFIDENCE_WARN {
        ica_phase.warnings.push(format!(
            "ICA confidence is capped by low working-image base confidence ({:.3}; quality factor {:.3})",
            base_confidence, base_quality_factor
        ));
    }
    record_phase(&mut report, &cli.output_dir, ica_phase, ica_start)?;
    drop(ica_result);
    drop(separated);

    let colorspace_start = Instant::now();
    let mut colorspace_result = colorspace::map_to_prophoto_d50_with_diagnostics(&transmittance);
    let ica_candidate_metrics = colorspace_candidate_metrics_json(&colorspace_result.diagnostics);
    let mut direct_density_candidate_evaluated = false;
    let mut direct_density_candidate_metrics = serde_json::Value::Null;
    let mut render_input_source = "fastica_separated_transmittance";
    let mut render_input_reason =
        "ICA-separated density channels remained stable enough for colorspace mapping".to_string();
    let mut render_input_warning: Option<String> = None;

    if colorspace::has_destructive_gamut_fallback(&colorspace_result.diagnostics) {
        let direct_transmittance = density::density_image_to_transmittance(&positive);
        let direct_result = colorspace::map_to_prophoto_d50_with_diagnostics(&direct_transmittance);
        direct_density_candidate_evaluated = true;
        direct_density_candidate_metrics =
            colorspace_candidate_metrics_json(&direct_result.diagnostics);

        if let Some(reason) = colorspace::direct_density_render_fallback_reason(
            &colorspace_result.diagnostics,
            &direct_result.diagnostics,
        ) {
            if cli.debug {
                tiff_io::save_tiff_f64(
                    &colorspace_result.prophoto,
                    &cli.output_dir.join("phase46_ica_candidate.tiff"),
                )?;
            }
            render_input_source = "direct_density_transmittance";
            render_input_reason = reason.clone();
            render_input_warning = Some(reason);
            colorspace_result = direct_result;
        }
    }
    let prophoto = colorspace_result.prophoto;
    if cli.debug {
        tiff_io::save_tiff_f64(&prophoto, &cli.output_dir.join("phase46_prophoto.tiff"))?;
    }
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let mut phase = PhaseReport::ok(
                "colorspace_mapping",
                1.0,
                serde_json::json!({
                    "target": "ProPhoto_D50",
                    "input_domain": "linear_transmittance",
                    "render_input_source": render_input_source,
                    "render_input_reason": render_input_reason,
                    "direct_density_candidate_evaluated": direct_density_candidate_evaluated,
                    "ica_candidate": ica_candidate_metrics,
                    "direct_density_candidate": direct_density_candidate_metrics,
                    "source_white": colorspace_result.diagnostics.source_white,
                    "work_to_xyz": colorspace_result.diagnostics.work_to_xyz,
                    "condition_number": colorspace_result.diagnostics.condition_number,
                    "regularization_lambda": colorspace_result.diagnostics.regularization_lambda,
                    "neutral_pixel_count": colorspace_result.diagnostics.neutral_pixel_count,
                    "channel_anchor_counts": colorspace_result.diagnostics.channel_anchor_counts,
                    "channel_anchor_min_count": colorspace_result.diagnostics.channel_anchor_min_count,
                    "channel_anchor_low_support_threshold": colorspace_result.diagnostics.channel_anchor_low_support_threshold,
                    "channel_anchor_low_support": colorspace_result.diagnostics.channel_anchor_low_support,
                    "weak_anchor_fallback_used": colorspace_result.diagnostics.weak_anchor_fallback_used,
                    "weak_anchor_fallback_reason": colorspace_result.diagnostics.weak_anchor_fallback_reason.as_deref(),
                    "dominant_anchor_rgb": colorspace_result.diagnostics.dominant_anchor_rgb,
                    "highlight_percentile": colorspace_result.diagnostics.highlight_percentile,
                    "pre_scale_channel_max": colorspace_result.diagnostics.pre_scale_channel_max,
                    "pre_scale_channel_high_percentile": colorspace_result
                        .diagnostics
                        .pre_scale_channel_high_percentile,
                    "pre_scale_clipped_high_ratio": colorspace_result
                        .diagnostics
                        .pre_scale_clipped_high_ratio,
                    "pre_scale_clipped_low_ratio": colorspace_result
                        .diagnostics
                        .pre_scale_clipped_low_ratio,
                    "post_scale_clipped_high_ratio": colorspace_result
                        .diagnostics
                        .post_scale_clipped_high_ratio,
                    "post_scale_clipped_low_ratio": colorspace_result
                        .diagnostics
                        .post_scale_clipped_low_ratio,
                    "exposure_scale": colorspace_result.diagnostics.exposure_scale,
                    "fallback_used": colorspace_result.diagnostics.fallback_used,
                    "gamut_fallback_used": colorspace_result.diagnostics.gamut_fallback_used,
                    "gamut_fallback_reason": colorspace_result.diagnostics.gamut_fallback_reason.as_deref(),
                    "mapping_strategy": colorspace_result.diagnostics.mapping_strategy,
                    "neutral_balance_scale": colorspace_result.diagnostics.neutral_balance_scale,
                    "image_matrix_pre_scale_clipped_low_ratio": colorspace_result
                        .diagnostics
                        .image_matrix_pre_scale_clipped_low_ratio,
                    "image_matrix_pre_scale_clipped_high_ratio": colorspace_result
                        .diagnostics
                        .image_matrix_pre_scale_clipped_high_ratio,
                    "image_matrix_exposure_scale": colorspace_result
                        .diagnostics
                        .image_matrix_exposure_scale,
                    "output_shape": [prophoto.shape()[0], prophoto.shape()[1]],
                }),
            );
            let base_quality_factor =
                apply_downstream_base_confidence_limit(&mut phase, base_confidence);
            if colorspace_result.diagnostics.fallback_used {
                phase.warnings.push(
                    "colorspace mapping fell back to the regularized prior because the image-derived work basis was ill-conditioned"
                        .to_string(),
                );
            }
            if let Some(reason) = &colorspace_result.diagnostics.weak_anchor_fallback_reason {
                phase.warnings.push(reason.clone());
            }
            if let Some(reason) = &colorspace_result.diagnostics.gamut_fallback_reason {
                phase.warnings.push(reason.clone());
            }
            if let Some(reason) = &render_input_warning {
                phase.warnings.push(reason.clone());
            }
            if colorspace_result
                .diagnostics
                .weak_anchor_fallback_reason
                .is_none()
            {
                if let Some(warning) =
                    colorspace::weak_channel_anchor_warning(&colorspace_result.diagnostics)
                {
                    phase.warnings.push(warning);
                }
            }
            if colorspace_result.diagnostics.exposure_scale > 1.0 + 1e-6 {
                phase.warnings.push(format!(
                    "colorspace mapping applied {:.3}x highlight headroom normalization before clamp because the mapped ProPhoto channels exceeded 1.0 at the {:.1}% percentile",
                    colorspace_result.diagnostics.exposure_scale,
                    colorspace_result.diagnostics.highlight_percentile * 100.0
                ));
            }
            if base_confidence < BASE_CONFIDENCE_WARN {
                phase.warnings.push(format!(
                    "colorspace confidence is capped by low working-image base confidence ({:.3}; quality factor {:.3})",
                    base_confidence, base_quality_factor
                ));
            }
            phase
        },
        colorspace_start,
    )?;

    let tone_start = Instant::now();
    let tone_fit = tonemap::fit_tone_params_with_diagnostics(&prophoto);
    let tone_params = tone_fit.params;
    let tone_apply = tonemap::apply_tonemap_with_params_and_diagnostics(&prophoto, &tone_params);
    let tonemapped = tone_apply.image;
    let render_quality = tonemap::render_quality_diagnostics(&tonemapped);
    if cli.debug {
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
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let mut phase = PhaseReport::ok(
                "tone_mapping",
                1.0,
                serde_json::json!({
                    "midpoint": tone_params.midpoint,
                    "slope": tone_params.slope,
                    "toe_lift": tone_params.toe_lift,
                    "shoulder_max": tone_params.shoulder_max,
                    "fit_domain": tone_fit.diagnostics.fit_domain,
                    "perceptual_luminance_gain": tone_fit.diagnostics.perceptual_luminance_gain,
                    "input_linear_percentiles": tone_fit.diagnostics.input_linear_percentiles,
                    "input_perceptual_percentiles": tone_fit.diagnostics.input_perceptual_percentiles,
                    "mapped_linear_percentiles": tone_fit.diagnostics.mapped_linear_percentiles,
                    "mapped_perceptual_percentiles": tone_fit.diagnostics.mapped_perceptual_percentiles,
                    "highlight_chroma_compressed_ratio": tone_apply.diagnostics.highlight_chroma_compressed_ratio,
                    "highlight_neutral_chroma_compressed_ratio": tone_apply.diagnostics.highlight_neutral_chroma_compressed_ratio,
                    "highlight_neutral_chroma_start_luminance": tone_apply.diagnostics.highlight_neutral_chroma_start_luminance,
                    "highlight_neutral_chroma_full_luminance": tone_apply.diagnostics.highlight_neutral_chroma_full_luminance,
                    "highlight_neutral_chroma_min_scale": tone_apply.diagnostics.highlight_neutral_chroma_min_scale,
                    "highlight_neutral_chroma_max_saturation": tone_apply.diagnostics.highlight_neutral_chroma_max_saturation,
                    "shadow_chroma_compressed_ratio": tone_apply.diagnostics.shadow_chroma_compressed_ratio,
                    "shadow_chroma_start_luminance": tone_apply.diagnostics.shadow_chroma_start_luminance,
                    "shadow_chroma_full_luminance": tone_apply.diagnostics.shadow_chroma_full_luminance,
                    "shadow_chroma_min_scale": tone_apply.diagnostics.shadow_chroma_min_scale,
                    "pre_chroma_compression_clipped_high_ratio": tone_apply.diagnostics.pre_chroma_compression_clipped_high_ratio,
                    "post_chroma_compression_clipped_high_ratio": tone_apply.diagnostics.post_chroma_compression_clipped_high_ratio,
                    "post_chroma_compression_clipped_low_ratio": tone_apply.diagnostics.post_chroma_compression_clipped_low_ratio,
                }),
            );
            if let serde_json::Value::Object(metrics) = &mut phase.metrics {
                metrics.insert(
                    "render_quality_sample_count".to_string(),
                    serde_json::json!(render_quality.sample_count),
                );
                metrics.insert(
                    "render_quality_sample_stride".to_string(),
                    serde_json::json!(render_quality.sample_stride),
                );
                metrics.insert(
                    "shadow_luminance_max".to_string(),
                    serde_json::json!(render_quality.shadow_luminance_max),
                );
                metrics.insert(
                    "midtone_luminance_range".to_string(),
                    serde_json::json!(render_quality.midtone_luminance_range),
                );
                metrics.insert(
                    "bright_neutral_luminance_min".to_string(),
                    serde_json::json!(render_quality.bright_neutral_luminance_min),
                );
                metrics.insert(
                    "bright_neutral_max_saturation".to_string(),
                    serde_json::json!(render_quality.bright_neutral_max_saturation),
                );
                metrics.insert(
                    "shadow_saturation_median".to_string(),
                    serde_json::json!(render_quality.shadow.saturation_median),
                );
                metrics.insert(
                    "shadow_saturation_p95".to_string(),
                    serde_json::json!(render_quality.shadow.saturation_p95),
                );
                metrics.insert(
                    "midtone_luminance_percentiles".to_string(),
                    serde_json::json!(render_quality.midtone.luminance_percentiles),
                );
                metrics.insert(
                    "midtone_saturation_median".to_string(),
                    serde_json::json!(render_quality.midtone.saturation_median),
                );
                metrics.insert(
                    "midtone_saturation_p95".to_string(),
                    serde_json::json!(render_quality.midtone.saturation_p95),
                );
                metrics.insert(
                    "bright_neutral_saturation_median".to_string(),
                    serde_json::json!(render_quality.bright_neutral.saturation_median),
                );
                metrics.insert(
                    "bright_neutral_saturation_p95".to_string(),
                    serde_json::json!(render_quality.bright_neutral.saturation_p95),
                );
                metrics.insert(
                    "shadow_rgb_median".to_string(),
                    serde_json::json!(render_quality.shadow.rgb_median),
                );
                metrics.insert(
                    "midtone_rgb_median".to_string(),
                    serde_json::json!(render_quality.midtone.rgb_median),
                );
                metrics.insert(
                    "bright_neutral_rgb_median".to_string(),
                    serde_json::json!(render_quality.bright_neutral.rgb_median),
                );
                metrics.insert(
                    "bright_saturated_saturation_median".to_string(),
                    serde_json::json!(render_quality.bright_saturated.saturation_median),
                );
                metrics.insert(
                    "bright_saturated_saturation_p95".to_string(),
                    serde_json::json!(render_quality.bright_saturated.saturation_p95),
                );
                metrics.insert(
                    "render_quality_bands".to_string(),
                    serde_json::json!({
                        "shadow": render_band_metrics_json(&render_quality.shadow),
                        "midtone": render_band_metrics_json(&render_quality.midtone),
                        "bright_neutral": render_band_metrics_json(&render_quality.bright_neutral),
                        "bright_saturated": render_band_metrics_json(&render_quality.bright_saturated),
                    }),
                );
            }
            let base_quality_factor =
                apply_downstream_base_confidence_limit(&mut phase, base_confidence);
            if tone_apply.diagnostics.highlight_chroma_compressed_ratio > 0.001 {
                phase.warnings.push(format!(
                    "tone mapping compressed highlight chroma for {:.2}% of pixels to preserve mapped luminance without hard channel clipping",
                    tone_apply.diagnostics.highlight_chroma_compressed_ratio * 100.0
                ));
            }
            if tone_apply
                .diagnostics
                .highlight_neutral_chroma_compressed_ratio
                > 0.001
            {
                phase.warnings.push(format!(
                    "tone mapping gently compressed high-luminance near-neutral chroma for {:.2}% of pixels above {:.3} luminance to reduce residual highlight color cast",
                    tone_apply.diagnostics.highlight_neutral_chroma_compressed_ratio * 100.0,
                    tone_apply
                        .diagnostics
                        .highlight_neutral_chroma_start_luminance
                ));
            }
            if tone_apply.diagnostics.shadow_chroma_compressed_ratio > 0.001 {
                phase.warnings.push(format!(
                    "tone mapping gently compressed shadow chroma for {:.2}% of pixels below {:.3} luminance to reduce low-tone color speckle",
                    tone_apply.diagnostics.shadow_chroma_compressed_ratio * 100.0,
                    tone_apply.diagnostics.shadow_chroma_start_luminance
                ));
            }
            if tone_fit.diagnostics.mapped_linear_percentiles[2] > 0.95 {
                phase.warnings.push(format!(
                    "tone fit leaves the 95th-percentile linear luminance near the shoulder ({:.3}); highlights may still appear compressed",
                    tone_fit.diagnostics.mapped_linear_percentiles[2]
                ));
            }
            if base_confidence < BASE_CONFIDENCE_WARN {
                phase.warnings.push(format!(
                    "tone-mapping confidence is capped by low working-image base confidence ({:.3}; quality factor {:.3})",
                    base_confidence, base_quality_factor
                ));
            }
            phase
        },
        tone_start,
    )?;

    let save_start = Instant::now();
    let output_path = cli.output_dir.join("output.tiff");
    if let Err(err) = tiff_io::save_tiff_f64(&tonemapped, &output_path) {
        record_failure(
            &mut report,
            &cli.output_dir,
            "save",
            format!("failed to save final output: {}", err),
            save_start,
        )?;
        return Err(err);
    }
    record_phase(
        &mut report,
        &cli.output_dir,
        PhaseReport::ok(
            "save",
            1.0,
            serde_json::json!({
                "output_path": output_path.to_string_lossy(),
                "output_shape": [tonemapped.shape()[0], tonemapped.shape()[1]],
            }),
        ),
        save_start,
    )?;

    log::info!(
        "Pipeline complete. {} phases recorded.",
        report.phases.len()
    );
    Ok(report)
}

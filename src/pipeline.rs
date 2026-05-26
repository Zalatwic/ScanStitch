use crate::base_detect;
use crate::border;
use crate::cli::{Cli, InputMode, RenderInputMode};
use crate::color_calibration;
use crate::colorspace;
use crate::density;
use crate::frame_classify;
use crate::ica;
use crate::interactive::{self, InteractiveRenderCache, InteractiveRenderControls};
use crate::positive_input;
use crate::report::{
    format_system_time_utc, system_time_unix_ms, PhaseReport, PipelineReport, RunMetadata,
};
use crate::stitch::{self, StitchConfig, TransformMode};
use crate::tiff_io;
use crate::tonemap;
use ndarray::Array3;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

const BASE_CONFIDENCE_WARN: f64 = 0.30;
const BASE_CONFIDENCE_FALLBACK: f64 = 0.15;
const SCENE_REFERRED_ARTIFACT_DIAGNOSTIC_SAMPLE_LIMIT: usize = 1_000_000;
const POSITIVE_RENDER_INPUT_SOURCE: &str = "positive_scan_rgb";
const POSITIVE_RENDER_INPUT_REASON: &str =
    "already-positive input normalized without density inversion";

#[derive(Debug, Clone, Copy)]
struct PipelineBuildOptions {
    write_artifacts: bool,
}

#[derive(Debug, Clone)]
struct InteractiveSaveOptions {
    write_partial_report: bool,
    write_debug_artifacts: bool,
    save_report: bool,
    include_interactive_metrics: bool,
    applied_review_sidecar: Option<AppliedReviewSidecarMetadata>,
}

#[derive(Debug, Clone)]
struct AppliedReviewSidecarMetadata {
    path: String,
    sha256: String,
    controls_applied: bool,
    mark_count: usize,
}

fn save_partial_report(
    report: &PipelineReport,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    report.save(&output_dir.join("report.json"))
}

fn review_sidecar_metadata_for_cache(
    cache: &InteractiveRenderCache,
) -> Result<Option<AppliedReviewSidecarMetadata>, Box<dyn std::error::Error>> {
    let Some(path) = cache.cli.review_sidecar.as_deref() else {
        return Ok(None);
    };
    let sidecar = interactive::load_review_sidecar(path)?;
    Ok(Some(AppliedReviewSidecarMetadata {
        path: sidecar.path.to_string_lossy().to_string(),
        sha256: sidecar.sha256,
        controls_applied: sidecar.sidecar.controls.is_some(),
        mark_count: sidecar.sidecar.marks.len(),
    }))
}

fn controls_for_cache(
    cache: &InteractiveRenderCache,
) -> Result<
    (
        InteractiveRenderControls,
        Option<AppliedReviewSidecarMetadata>,
    ),
    Box<dyn std::error::Error>,
> {
    let default_controls = cache.default_controls();
    let Some(path) = cache.cli.review_sidecar.as_deref() else {
        return Ok((default_controls, None));
    };
    let sidecar = interactive::load_review_sidecar(path)?;
    let controls = interactive::controls_with_review_sidecar(
        default_controls,
        &cache.auto_tone_params,
        &sidecar.sidecar,
    );
    let metadata = AppliedReviewSidecarMetadata {
        path: sidecar.path.to_string_lossy().to_string(),
        sha256: sidecar.sha256,
        controls_applied: sidecar.sidecar.controls.is_some(),
        mark_count: sidecar.sidecar.marks.len(),
    };
    Ok((controls, Some(metadata)))
}

fn record_phase(
    report: &mut PipelineReport,
    output_dir: &Path,
    phase: PhaseReport,
    phase_start: Instant,
    write_partial_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    report.add_phase(phase.with_duration_ms(phase_start.elapsed().as_millis() as u64));
    if write_partial_report {
        save_partial_report(report, output_dir)
    } else {
        Ok(())
    }
}

fn record_failure(
    report: &mut PipelineReport,
    output_dir: &Path,
    name: &str,
    message: String,
    phase_start: Instant,
    write_partial_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    report.add_phase(
        PhaseReport::fail(name, &message)
            .with_duration_ms(phase_start.elapsed().as_millis() as u64),
    );
    if write_partial_report {
        save_partial_report(report, output_dir)
    } else {
        Ok(())
    }
}

fn require_pipeline_state<T>(
    state: Option<T>,
    report: &mut PipelineReport,
    output_dir: &Path,
    phase_name: &str,
    message: &str,
    phase_start: Instant,
    write_partial_report: bool,
) -> Result<T, Box<dyn std::error::Error>> {
    match state {
        Some(state) => Ok(state),
        None => {
            let message = format!("internal pipeline state error: {message}");
            record_failure(
                report,
                output_dir,
                phase_name,
                message.clone(),
                phase_start,
                write_partial_report,
            )?;
            Err(std::io::Error::other(message).into())
        }
    }
}

fn require_pipeline_state_ref<'a, T>(
    state: Option<&'a T>,
    report: &mut PipelineReport,
    output_dir: &Path,
    phase_name: &str,
    message: &str,
    phase_start: Instant,
    write_partial_report: bool,
) -> Result<&'a T, Box<dyn std::error::Error>> {
    match state {
        Some(state) => Ok(state),
        None => {
            let message = format!("internal pipeline state error: {message}");
            record_failure(
                report,
                output_dir,
                phase_name,
                message.clone(),
                phase_start,
                write_partial_report,
            )?;
            Err(std::io::Error::other(message).into())
        }
    }
}

fn downstream_base_quality_factor(base_confidence: f64) -> f64 {
    let t = (base_confidence / BASE_CONFIDENCE_WARN).clamp(0.0, 1.0);
    (0.25 + 0.75 * t).clamp(0.25, 1.0)
}

fn base_detection_confidence(detection: &base_detect::BaseDetection) -> f64 {
    detection
        .left_confidence
        .max(detection.right_confidence)
        .max(detection.top_confidence)
        .max(detection.bottom_confidence)
}

fn parse_base_color_override(cli: &Cli) -> Result<Option<[f64; 3]>, Box<dyn std::error::Error>> {
    let Some(raw) = cli.base_color.as_deref() else {
        return Ok(None);
    };
    let parts = raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err("--base-color must contain exactly three comma-separated numbers".into());
    }

    let mut parsed = [0.0f64; 3];
    for (idx, part) in parts.iter().enumerate() {
        parsed[idx] = part
            .parse::<f64>()
            .map_err(|_| format!("--base-color component `{part}` is not a number"))?;
        if !parsed[idx].is_finite() || parsed[idx] <= 0.0 {
            return Err("--base-color values must be finite and positive".into());
        }
    }

    let max_value = if cli.bit_depth >= 16 {
        u16::MAX as f64
    } else {
        ((1u32 << cli.bit_depth as u32) - 1) as f64
    };
    if parsed.iter().any(|value| *value > max_value) {
        return Err(format!(
            "--base-color values must fit the configured {}-bit working range (max {:.0})",
            cli.bit_depth, max_value
        )
        .into());
    }

    Ok(Some(parsed))
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

fn render_review_status(base_confidence: f64) -> &'static str {
    if base_confidence < BASE_CONFIDENCE_FALLBACK {
        "blocked_low_base_confidence"
    } else if base_confidence < BASE_CONFIDENCE_WARN {
        "caution_low_base_confidence"
    } else {
        "reviewable"
    }
}

fn render_review_reason(base_confidence: f64) -> &'static str {
    if base_confidence < BASE_CONFIDENCE_FALLBACK {
        "working-image base estimate is fallback quality; density inversion may not produce a trustworthy positive"
    } else if base_confidence < BASE_CONFIDENCE_WARN {
        "working-image base estimate is weak; compare debug density and color artifacts before judging the render"
    } else {
        "working-image base estimate is strong enough for normal visual review"
    }
}

fn div_ceil_usize(value: usize, divisor: usize) -> usize {
    if divisor == 0 {
        return value;
    }
    value.div_ceil(divisor)
}

fn ratio_option(count: usize, total: usize) -> Option<f64> {
    if total == 0 {
        None
    } else {
        Some(count as f64 / total as f64)
    }
}

fn percentile_value(sorted: &[f64], percentile: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted.get(index.min(sorted.len() - 1)).copied()
}

fn scene_referred_channel_percentiles(
    samples: &[Vec<f64>; 3],
    percentile: f64,
) -> [Option<f64>; 3] {
    [
        percentile_value(&samples[0], percentile),
        percentile_value(&samples[1], percentile),
        percentile_value(&samples[2], percentile),
    ]
}

fn scene_referred_artifact_diagnostics(img: &Array3<f64>) -> serde_json::Value {
    let shape = img.shape();
    let height = shape[0];
    let width = shape[1];
    let pixel_count = height.saturating_mul(width);
    let sample_stride =
        div_ceil_usize(pixel_count, SCENE_REFERRED_ARTIFACT_DIAGNOSTIC_SAMPLE_LIMIT).max(1);
    let sample_capacity = pixel_count.clamp(1, SCENE_REFERRED_ARTIFACT_DIAGNOSTIC_SAMPLE_LIMIT);
    let mut samples = [
        Vec::with_capacity(sample_capacity),
        Vec::with_capacity(sample_capacity),
        Vec::with_capacity(sample_capacity),
    ];
    let mut sampled_pixel_count = 0usize;
    let mut finite_channel_counts = [0usize; 3];
    let mut nonfinite_channel_counts = [0usize; 3];
    let mut negative_channel_counts = [0usize; 3];
    let mut above_white_channel_counts = [0usize; 3];
    let mut min_values = [f64::INFINITY; 3];
    let mut max_values = [f64::NEG_INFINITY; 3];
    let mut pixels_below_zero = 0usize;
    let mut pixels_above_white = 0usize;

    for y in 0..height {
        for x in 0..width {
            let pixel_index = y * width + x;
            let sample_pixel = pixel_index.is_multiple_of(sample_stride);
            let mut pixel_below_zero = false;
            let mut pixel_above_white = false;
            if sample_pixel {
                sampled_pixel_count += 1;
            }
            for c in 0..3 {
                let value = img[[y, x, c]];
                if value.is_finite() {
                    finite_channel_counts[c] += 1;
                    min_values[c] = min_values[c].min(value);
                    max_values[c] = max_values[c].max(value);
                    if value < 0.0 {
                        negative_channel_counts[c] += 1;
                        pixel_below_zero = true;
                    }
                    if value > 1.0 {
                        above_white_channel_counts[c] += 1;
                        pixel_above_white = true;
                    }
                    if sample_pixel {
                        samples[c].push(value);
                    }
                } else {
                    nonfinite_channel_counts[c] += 1;
                }
            }
            if pixel_below_zero {
                pixels_below_zero += 1;
            }
            if pixel_above_white {
                pixels_above_white += 1;
            }
        }
    }

    for channel in &mut samples {
        channel.sort_by(|a, b| a.total_cmp(b));
    }

    let channel_min = [
        (finite_channel_counts[0] > 0).then_some(min_values[0]),
        (finite_channel_counts[1] > 0).then_some(min_values[1]),
        (finite_channel_counts[2] > 0).then_some(min_values[2]),
    ];
    let channel_max = [
        (finite_channel_counts[0] > 0).then_some(max_values[0]),
        (finite_channel_counts[1] > 0).then_some(max_values[1]),
        (finite_channel_counts[2] > 0).then_some(max_values[2]),
    ];
    let channel_negative_ratio = [
        ratio_option(negative_channel_counts[0], finite_channel_counts[0]),
        ratio_option(negative_channel_counts[1], finite_channel_counts[1]),
        ratio_option(negative_channel_counts[2], finite_channel_counts[2]),
    ];
    let channel_above_white_ratio = [
        ratio_option(above_white_channel_counts[0], finite_channel_counts[0]),
        ratio_option(above_white_channel_counts[1], finite_channel_counts[1]),
        ratio_option(above_white_channel_counts[2], finite_channel_counts[2]),
    ];

    serde_json::json!({
        "encoding": "32-bit IEEE float RGB TIFF; linear ProPhoto RGB D50; scene-referred values outside [0,1] preserved without display normalization",
        "value_domain": "scene_referred_linear_prophoto_rgb_d50",
        "sample_format": "IEEEFP",
        "bits_per_sample": [32, 32, 32],
        "normalization": "none",
        "icc_profile": {
            "embedded": true,
            "tiff_tag": tiff_io::ICC_PROFILE_TAG,
            "description": tiff_io::PROPHOTO_LINEAR_ICC_DESCRIPTION,
            "transfer_function": "linear",
            "whitepoint": "D50"
        },
        "output_shape": [height, width],
        "sample_stride": sample_stride,
        "sampled_pixel_count": sampled_pixel_count,
        "finite_channel_counts": finite_channel_counts,
        "nonfinite_channel_counts": nonfinite_channel_counts,
        "channel_min": channel_min,
        "channel_max": channel_max,
        "channel_percentiles": {
            "p0_1": scene_referred_channel_percentiles(&samples, 0.001),
            "p1": scene_referred_channel_percentiles(&samples, 0.01),
            "p50": scene_referred_channel_percentiles(&samples, 0.50),
            "p95": scene_referred_channel_percentiles(&samples, 0.95),
            "p99": scene_referred_channel_percentiles(&samples, 0.99),
            "p99_9": scene_referred_channel_percentiles(&samples, 0.999)
        },
        "channel_negative_ratio": channel_negative_ratio,
        "channel_above_display_white_ratio": channel_above_white_ratio,
        "pixel_below_zero_ratio": ratio_option(pixels_below_zero, pixel_count),
        "pixel_above_display_white_ratio": ratio_option(pixels_above_white, pixel_count),
    })
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
        "dng_metadata": diag.dng_metadata.as_ref().map(dng_metadata_metrics_json),
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

fn dng_metadata_metrics_json(metadata: &tiff_io::DngMetadata) -> serde_json::Value {
    serde_json::json!({
        "make": metadata.make,
        "model": metadata.model,
        "software": metadata.software,
        "unique_camera_model": metadata.unique_camera_model,
        "calibration_illuminant1": metadata.calibration_illuminant1,
        "calibration_illuminant2": metadata.calibration_illuminant2,
        "color_matrix1": metadata.color_matrix1,
        "color_matrix1_condition_number": metadata
            .color_matrix1
            .as_ref()
            .map(color_calibration::matrix_condition_number_rows),
        "color_matrix2": metadata.color_matrix2,
        "color_matrix2_condition_number": metadata
            .color_matrix2
            .as_ref()
            .map(color_calibration::matrix_condition_number_rows),
        "as_shot_neutral": metadata.as_shot_neutral,
        "black_level": metadata.black_level,
        "white_level": metadata.white_level,
        "parse_warnings": metadata.parse_warnings,
    })
}

fn dng_matrix_close(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> bool {
    left.iter().zip(right.iter()).all(|(left_row, right_row)| {
        left_row
            .iter()
            .zip(right_row.iter())
            .all(|(left, right)| (*left - *right).abs() <= 1e-9)
    })
}

fn automatic_dng_advisory_calibration(
    cli: &Cli,
    load1_diag: &tiff_io::TiffLoadDiagnostics,
    load2_diag: &tiff_io::TiffLoadDiagnostics,
) -> Option<color_calibration::CalibrationLoadResult> {
    if cli.color_mode != colorspace::ColorMode::Auto
        || cli.calibration_profile.is_some()
        || cli.calibration_library.is_some()
        || cli.scanner_profile.is_some()
        || cli.roll_profile.is_some()
        || cli.film_stock.is_some()
    {
        return None;
    }

    let mut selected: Option<&tiff_io::DngMetadata> = None;
    for metadata in [&load1_diag.dng_metadata, &load2_diag.dng_metadata]
        .into_iter()
        .filter_map(Option::as_ref)
        .filter(|metadata| metadata.color_matrix1.is_some())
    {
        if let Some(existing) = selected {
            let existing_matrix = existing.color_matrix1.as_ref()?;
            let candidate_matrix = metadata.color_matrix1.as_ref()?;
            if !dng_matrix_close(existing_matrix, candidate_matrix) {
                return None;
            }
        } else {
            selected = Some(metadata);
        }
    }

    selected.and_then(color_calibration::dng_color_matrix1_advisory_prior)
}

fn scene_referred_detail_fusion_metrics_json(
    diagnostics: &tonemap::SceneReferredDetailFusionDiagnostics,
) -> serde_json::Value {
    serde_json::json!({
        "enabled": diagnostics.enabled,
        "reason": diagnostics.reason,
        "radius": diagnostics.radius,
        "amount": diagnostics.amount,
        "max_ev": diagnostics.max_ev,
        "applied_ratio": diagnostics.applied_ratio,
        "mean_abs_ev": diagnostics.mean_abs_ev,
        "max_abs_ev": diagnostics.max_abs_ev,
        "skipped_negative_ratio": diagnostics.skipped_negative_ratio,
        "skipped_nonfinite_ratio": diagnostics.skipped_nonfinite_ratio,
    })
}

fn colorspace_candidate_metrics_json(
    diagnostics: &colorspace::ColorspaceDiagnostics,
) -> serde_json::Value {
    let (image_matrix_low_clip_max, image_matrix_low_clip_total) =
        colorspace::image_matrix_low_clip_summary(diagnostics);
    let calibrated_profile_candidate =
        diagnostics
            .calibrated_profile_exposure_scale
            .map(|exposure_scale| {
                serde_json::json!({
                    "source": "external_calibration_profile",
                    "pre_scale_clipped_low_ratio": diagnostics.calibrated_profile_pre_scale_clipped_low_ratio,
                    "pre_scale_clipped_high_ratio": diagnostics.calibrated_profile_pre_scale_clipped_high_ratio,
                    "pre_scale_preserved_ratio": diagnostics.calibrated_profile_pre_scale_preserved_ratio,
                    "exposure_scale": exposure_scale,
                    "neutral_balance_delta": diagnostics.calibrated_profile_neutral_balance_delta,
                })
            })
            .unwrap_or(serde_json::Value::Null);
    let image_candidate_source =
        if diagnostics.mapping_strategy == "scanner_constrained_image_derived_matrix" {
            "scanner_constrained_image_derived_neutral_and_dominant_anchors"
        } else {
            "image_derived_neutral_and_dominant_anchors"
        };
    let image_derived_candidate = serde_json::json!({
        "source": image_candidate_source,
        "pre_scale_clipped_low_ratio": diagnostics.image_matrix_pre_scale_clipped_low_ratio,
        "pre_scale_clipped_low_max": image_matrix_low_clip_max,
        "pre_scale_clipped_low_total": image_matrix_low_clip_total,
        "pre_scale_clipped_high_ratio": diagnostics.image_matrix_pre_scale_clipped_high_ratio,
        "pre_scale_preserved_ratio": diagnostics.image_matrix_pre_scale_preserved_ratio,
        "exposure_scale": diagnostics.image_matrix_exposure_scale,
        "neutral_balance_delta": diagnostics.image_matrix_neutral_balance_delta,
        "channel_anchor_counts": diagnostics.channel_anchor_counts,
        "channel_anchor_min_count": diagnostics.channel_anchor_min_count,
        "channel_anchor_low_support": diagnostics.channel_anchor_low_support,
        "dominant_anchor_quality": diagnostics.dominant_anchor_quality.clone(),
    });
    let candidate_comparison = serde_json::json!({
        "selected": diagnostics.mapping_strategy,
        "selected_mapping_reason": diagnostics.selected_mapping_reason,
        "selected_candidate": diagnostics.selected_candidate,
        "selected_candidate_rank": diagnostics.selected_candidate_rank,
        "selected_candidate_score": diagnostics.selected_candidate_score,
        "selected_quality_score": diagnostics.selected_quality_score,
        "technical_safety_score": diagnostics.technical_safety_score,
        "color_fidelity_score": diagnostics.color_fidelity_score,
        "candidate_risk": diagnostics.candidate_risk,
        "selected_runner_up_quality_delta": diagnostics.selected_runner_up_quality_delta,
        "calibrated_profile": calibrated_profile_candidate,
        "image_derived": image_derived_candidate,
    });
    let usable_colourspace = serde_json::json!({
        "pre_scale_preserved_ratio": diagnostics.pre_scale_preserved_ratio,
        "post_scale_preserved_ratio": diagnostics.post_scale_preserved_ratio,
        "pre_scale_clipped_high_ratio": diagnostics.pre_scale_clipped_high_ratio,
        "pre_scale_clipped_low_ratio": diagnostics.pre_scale_clipped_low_ratio,
        "post_scale_clipped_high_ratio": diagnostics.post_scale_clipped_high_ratio,
        "post_scale_clipped_low_ratio": diagnostics.post_scale_clipped_low_ratio,
        "gamut_fallback_used": diagnostics.gamut_fallback_used,
        "gamut_fallback_reason": diagnostics.gamut_fallback_reason.as_deref(),
    });
    let candidate_quality_scores = diagnostics
        .candidate_scores
        .iter()
        .map(colorspace_candidate_score_json)
        .collect::<Vec<_>>();
    let selected_quality_components = diagnostics
        .candidate_scores
        .iter()
        .find(|candidate| candidate.selected)
        .map(|candidate| candidate.quality_components);
    let selected_acceptance = diagnostics
        .candidate_acceptance
        .iter()
        .find(|candidate| candidate.selected)
        .cloned();
    let runner_up = diagnostics
        .candidate_scores
        .iter()
        .filter(|candidate| !candidate.selected)
        .filter_map(|candidate| {
            candidate
                .selected_quality_delta
                .filter(|delta| *delta >= 0.0)
                .map(|delta| (delta, candidate))
        })
        .min_by(|(left_delta, left), (right_delta, right)| {
            left_delta
                .partial_cmp(right_delta)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.candidate.cmp(right.candidate))
        })
        .map(|(delta, candidate)| {
            serde_json::json!({
                "candidate": candidate.candidate,
                "mapping_strategy": candidate.mapping_strategy,
                "rank": candidate.rank,
                "quality_score": candidate.quality_score,
                "selected_quality_delta": delta,
                "rejected": candidate.rejected,
                "rejection_reason": candidate.rejection_reason.as_deref(),
            })
        })
        .unwrap_or(serde_json::Value::Null);
    let rejected_alternatives = diagnostics
        .candidate_acceptance
        .iter()
        .filter(|candidate| !candidate.selected && candidate.status.starts_with("rejected"))
        .cloned()
        .collect::<Vec<_>>();
    let color_decision_summary = serde_json::json!({
        "selected_candidate": diagnostics.selected_candidate,
        "selected_mapping_reason": diagnostics.selected_mapping_reason,
        "selected_quality_score": diagnostics.selected_quality_score,
        "technical_safety_score": diagnostics.technical_safety_score,
        "color_fidelity_score": diagnostics.color_fidelity_score,
        "quality_components": selected_quality_components,
        "runner_up": runner_up,
        "candidate_risk": diagnostics.candidate_risk,
        "tone_color_trust_state": colorspace::tone_color_trust_state(diagnostics),
        "selected_acceptance": selected_acceptance,
        "rejected_alternatives": rejected_alternatives,
        "selection_rejections": diagnostics.selection_rejections,
    });
    let color_processing_substeps = serde_json::json!({
        "calibration_selection": diagnostics.calibration_acceptance.clone(),
        "candidate_construction": {
            "candidate_count": diagnostics.candidate_scores.len(),
            "candidates": diagnostics
                .candidate_scores
                .iter()
                .map(|score| score.candidate)
                .collect::<Vec<_>>(),
        },
        "candidate_scoring": {
            "score_order": "lower_is_better",
            "candidate_quality_scores": candidate_quality_scores.clone(),
            "selected_candidate": diagnostics.selected_candidate,
            "selected_quality_score": diagnostics.selected_quality_score,
            "technical_safety_score": diagnostics.technical_safety_score,
            "color_fidelity_score": diagnostics.color_fidelity_score,
            "quality_components": selected_quality_components,
            "candidate_risk": diagnostics.candidate_risk,
            "selected_runner_up_quality_delta": diagnostics.selected_runner_up_quality_delta,
            "candidate_acceptance": diagnostics.candidate_acceptance.clone(),
        },
        "gamut_normalization": usable_colourspace.clone(),
        "neutral_trim": diagnostics.neutral_trim_before_after.clone(),
        "tone_map_colour_protection": {
            "status": colorspace::tone_color_trust_state(diagnostics),
            "reason": "tone mapping receives selected colorspace candidate quality diagnostics",
        },
    });
    let mut metrics = serde_json::Map::new();
    metrics.insert(
        "mapping_strategy".to_string(),
        serde_json::json!(diagnostics.mapping_strategy),
    );
    metrics.insert(
        "selected_mapping_reason".to_string(),
        serde_json::json!(diagnostics.selected_mapping_reason),
    );
    metrics.insert(
        "candidate_scores".to_string(),
        serde_json::json!(candidate_quality_scores.clone()),
    );
    metrics.insert(
        "candidate_quality_scores".to_string(),
        serde_json::json!(diagnostics
            .candidate_scores
            .iter()
            .map(colorspace_candidate_score_json)
            .collect::<Vec<_>>()),
    );
    metrics.insert(
        "candidate_score_order".to_string(),
        serde_json::json!("lower_is_better"),
    );
    metrics.insert(
        "candidate_acceptance".to_string(),
        serde_json::json!(diagnostics.candidate_acceptance.clone()),
    );
    metrics.insert(
        "selected_candidate".to_string(),
        serde_json::json!(diagnostics.selected_candidate),
    );
    metrics.insert(
        "selected_candidate_rank".to_string(),
        serde_json::json!(diagnostics.selected_candidate_rank),
    );
    metrics.insert(
        "selected_candidate_score".to_string(),
        serde_json::json!(diagnostics.selected_candidate_score),
    );
    metrics.insert(
        "selected_quality_score".to_string(),
        serde_json::json!(diagnostics.selected_quality_score),
    );
    metrics.insert(
        "technical_safety_score".to_string(),
        serde_json::json!(diagnostics.technical_safety_score),
    );
    metrics.insert(
        "color_fidelity_score".to_string(),
        serde_json::json!(diagnostics.color_fidelity_score),
    );
    metrics.insert(
        "quality_components".to_string(),
        serde_json::json!(selected_quality_components),
    );
    metrics.insert(
        "selected_quality_components".to_string(),
        serde_json::json!(selected_quality_components),
    );
    metrics.insert(
        "candidate_risk".to_string(),
        serde_json::json!(diagnostics.candidate_risk),
    );
    metrics.insert(
        "selected_runner_up_quality_delta".to_string(),
        serde_json::json!(diagnostics.selected_runner_up_quality_delta),
    );
    metrics.insert(
        "selection_rejections".to_string(),
        serde_json::json!(diagnostics.selection_rejections),
    );
    metrics.insert("color_decision_summary".to_string(), color_decision_summary);
    metrics.insert(
        "tone_color_trust_state".to_string(),
        serde_json::json!(colorspace::tone_color_trust_state(diagnostics)),
    );
    metrics.insert(
        "neutral_estimate_quality".to_string(),
        serde_json::json!(diagnostics.neutral_estimate_quality.clone()),
    );
    metrics.insert(
        "neutral_sample_rejections".to_string(),
        serde_json::json!(diagnostics.neutral_sample_rejections.clone()),
    );
    if let Some(reference_patch_evaluation) = &diagnostics.reference_patch_evaluation {
        metrics.insert(
            "reference_patch_evaluation".to_string(),
            serde_json::json!(reference_patch_evaluation),
        );
    }
    metrics.insert(
        "neutral_trim_before_after".to_string(),
        serde_json::json!(diagnostics.neutral_trim_before_after.clone()),
    );
    metrics.insert(
        "calibration_acceptance".to_string(),
        serde_json::json!(diagnostics.calibration_acceptance.clone()),
    );
    metrics.insert(
        "color_processing_substeps".to_string(),
        color_processing_substeps,
    );
    metrics.insert(
        "source_white".to_string(),
        serde_json::json!(diagnostics.source_white),
    );
    metrics.insert(
        "work_to_xyz".to_string(),
        serde_json::json!(diagnostics.work_to_xyz),
    );
    metrics.insert(
        "condition_number".to_string(),
        serde_json::json!(diagnostics.condition_number),
    );
    metrics.insert(
        "neutral_pixel_count".to_string(),
        serde_json::json!(diagnostics.neutral_pixel_count),
    );
    metrics.insert(
        "neutral_sample_bands".to_string(),
        serde_json::json!(diagnostics.neutral_sample_bands),
    );
    metrics.insert(
        "regularization_lambda".to_string(),
        serde_json::json!(diagnostics.regularization_lambda),
    );
    metrics.insert(
        "channel_anchor_counts".to_string(),
        serde_json::json!(diagnostics.channel_anchor_counts),
    );
    metrics.insert(
        "channel_anchor_min_count".to_string(),
        serde_json::json!(diagnostics.channel_anchor_min_count),
    );
    metrics.insert(
        "channel_anchor_low_support_threshold".to_string(),
        serde_json::json!(diagnostics.channel_anchor_low_support_threshold),
    );
    metrics.insert(
        "channel_anchor_low_support".to_string(),
        serde_json::json!(diagnostics.channel_anchor_low_support),
    );
    metrics.insert(
        "weak_anchor_fallback_used".to_string(),
        serde_json::json!(diagnostics.weak_anchor_fallback_used),
    );
    metrics.insert(
        "weak_anchor_fallback_reason".to_string(),
        serde_json::json!(diagnostics.weak_anchor_fallback_reason.as_deref()),
    );
    metrics.insert(
        "gamut_fallback_used".to_string(),
        serde_json::json!(diagnostics.gamut_fallback_used),
    );
    metrics.insert(
        "gamut_fallback_reason".to_string(),
        serde_json::json!(diagnostics.gamut_fallback_reason.as_deref()),
    );
    metrics.insert(
        "dominant_anchor_rgb".to_string(),
        serde_json::json!(diagnostics.dominant_anchor_rgb),
    );
    metrics.insert(
        "dominant_anchor_sample_rejections".to_string(),
        serde_json::json!(diagnostics.dominant_anchor_sample_rejections.clone()),
    );
    metrics.insert(
        "dominant_anchor_bands".to_string(),
        serde_json::json!(diagnostics.dominant_anchor_bands),
    );
    metrics.insert(
        "dominant_anchor_quality".to_string(),
        serde_json::json!(diagnostics.dominant_anchor_quality.clone()),
    );
    metrics.insert(
        "highlight_percentile".to_string(),
        serde_json::json!(diagnostics.highlight_percentile),
    );
    metrics.insert(
        "pre_scale_channel_max".to_string(),
        serde_json::json!(diagnostics.pre_scale_channel_max),
    );
    metrics.insert(
        "pre_scale_channel_high_percentile".to_string(),
        serde_json::json!(diagnostics.pre_scale_channel_high_percentile),
    );
    metrics.insert(
        "image_matrix_pre_scale_clipped_low_ratio".to_string(),
        serde_json::json!(diagnostics.image_matrix_pre_scale_clipped_low_ratio),
    );
    metrics.insert(
        "image_matrix_pre_scale_clipped_low_max".to_string(),
        serde_json::json!(image_matrix_low_clip_max),
    );
    metrics.insert(
        "image_matrix_pre_scale_clipped_low_total".to_string(),
        serde_json::json!(image_matrix_low_clip_total),
    );
    metrics.insert(
        "image_matrix_pre_scale_clipped_high_ratio".to_string(),
        serde_json::json!(diagnostics.image_matrix_pre_scale_clipped_high_ratio),
    );
    metrics.insert(
        "image_matrix_exposure_scale".to_string(),
        serde_json::json!(diagnostics.image_matrix_exposure_scale),
    );
    metrics.insert(
        "image_matrix_pre_scale_preserved_ratio".to_string(),
        serde_json::json!(diagnostics.image_matrix_pre_scale_preserved_ratio),
    );
    metrics.insert(
        "image_matrix_neutral_balance_delta".to_string(),
        serde_json::json!(diagnostics.image_matrix_neutral_balance_delta),
    );
    metrics.insert(
        "calibrated_profile_pre_scale_clipped_low_ratio".to_string(),
        serde_json::json!(diagnostics.calibrated_profile_pre_scale_clipped_low_ratio),
    );
    metrics.insert(
        "calibrated_profile_pre_scale_clipped_high_ratio".to_string(),
        serde_json::json!(diagnostics.calibrated_profile_pre_scale_clipped_high_ratio),
    );
    metrics.insert(
        "calibrated_profile_exposure_scale".to_string(),
        serde_json::json!(diagnostics.calibrated_profile_exposure_scale),
    );
    metrics.insert(
        "calibrated_profile_pre_scale_preserved_ratio".to_string(),
        serde_json::json!(diagnostics.calibrated_profile_pre_scale_preserved_ratio),
    );
    metrics.insert(
        "calibrated_profile_neutral_balance_delta".to_string(),
        serde_json::json!(diagnostics.calibrated_profile_neutral_balance_delta),
    );
    metrics.insert(
        "pre_scale_clipped_high_ratio".to_string(),
        serde_json::json!(diagnostics.pre_scale_clipped_high_ratio),
    );
    metrics.insert(
        "pre_scale_clipped_low_ratio".to_string(),
        serde_json::json!(diagnostics.pre_scale_clipped_low_ratio),
    );
    metrics.insert(
        "pre_scale_preserved_ratio".to_string(),
        serde_json::json!(diagnostics.pre_scale_preserved_ratio),
    );
    metrics.insert(
        "post_scale_clipped_high_ratio".to_string(),
        serde_json::json!(diagnostics.post_scale_clipped_high_ratio),
    );
    metrics.insert(
        "post_scale_clipped_low_ratio".to_string(),
        serde_json::json!(diagnostics.post_scale_clipped_low_ratio),
    );
    metrics.insert(
        "post_scale_preserved_ratio".to_string(),
        serde_json::json!(diagnostics.post_scale_preserved_ratio),
    );
    metrics.insert(
        "exposure_scale".to_string(),
        serde_json::json!(diagnostics.exposure_scale),
    );
    metrics.insert(
        "fallback_used".to_string(),
        serde_json::json!(diagnostics.fallback_used),
    );
    metrics.insert(
        "neutral_balance_scale".to_string(),
        serde_json::json!(diagnostics.neutral_balance_scale),
    );
    metrics.insert(
        "neutral_trim_scale".to_string(),
        serde_json::json!(diagnostics.neutral_trim_scale),
    );
    metrics.insert(
        "neutral_trim_applied".to_string(),
        serde_json::json!(diagnostics.neutral_trim_applied),
    );
    metrics.insert("candidate_comparison".to_string(), candidate_comparison);
    metrics.insert("usable_colourspace".to_string(), usable_colourspace);
    serde_json::Value::Object(metrics)
}

fn colorspace_candidate_score_json(
    score: &colorspace::ColorMappingCandidateScore,
) -> serde_json::Value {
    serde_json::to_value(score).unwrap_or(serde_json::Value::Null)
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

fn render_grain_metrics_json(diagnostics: &tonemap::RenderGrainDiagnostics) -> serde_json::Value {
    serde_json::json!({
        "sample_count": diagnostics.sample_count,
        "sample_stride": diagnostics.sample_stride,
        "flat_luma_structure_max": diagnostics.flat_luma_structure_max,
        "flat_sample_count": diagnostics.flat_sample_count,
        "flat_sample_ratio": diagnostics.flat_sample_ratio,
        "luma_residual_median": diagnostics.luma_residual_median,
        "luma_residual_p95": diagnostics.luma_residual_p95,
        "chroma_residual_median": diagnostics.chroma_residual_median,
        "chroma_residual_p95": diagnostics.chroma_residual_p95,
        "chroma_to_luma_p95_ratio": diagnostics.chroma_to_luma_p95_ratio,
        "flat_luma_residual_p95": diagnostics.flat_luma_residual_p95,
        "flat_chroma_residual_p95": diagnostics.flat_chroma_residual_p95,
        "flat_chroma_to_luma_p95_ratio": diagnostics.flat_chroma_to_luma_p95_ratio,
    })
}

fn file_modified_at(path: &Path) -> Option<SystemTime> {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
}

fn file_modified_at_string(path: &Path) -> Option<String> {
    file_modified_at(path).map(format_system_time_utc)
}

fn file_modified_at_unix_ms(path: &Path) -> Option<u64> {
    file_modified_at(path).and_then(system_time_unix_ms)
}

fn file_size_bytes(path: &Path) -> Option<u64> {
    fs::metadata(path).map(|metadata| metadata.len()).ok()
}

fn file_sha256_hex(path: &Path) -> Option<String> {
    fs::read(path)
        .ok()
        .map(|bytes| interactive::sha256_hex(&bytes))
}

fn same_existing_path(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

fn stale_render_artifacts_json(
    output_dir: &Path,
    overwritten_output: &Path,
    run_started_at: SystemTime,
) -> Vec<serde_json::Value> {
    let Ok(entries) = fs::read_dir(output_dir) else {
        return Vec::new();
    };

    let mut artifacts = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() || same_existing_path(&path, overwritten_output) {
            continue;
        }
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase());
        if !matches!(extension.as_deref(), Some("tif" | "tiff")) {
            continue;
        }
        let Some(modified_at) = file_modified_at(&path) else {
            continue;
        };
        if modified_at > run_started_at {
            continue;
        }
        artifacts.push(serde_json::json!({
            "path": path.to_string_lossy(),
            "modified_at": format_system_time_utc(modified_at),
            "modified_at_unix_ms": system_time_unix_ms(modified_at),
            "size_bytes": file_size_bytes(&path),
        }));
    }
    artifacts
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

fn build_interactive_render_cache_with_options(
    cli: &Cli,
    options: PipelineBuildOptions,
) -> Result<InteractiveRenderCache, Box<dyn std::error::Error>> {
    cli.validate()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    if options.write_artifacts {
        std::fs::create_dir_all(&cli.output_dir)?;
    }
    let run_started_at = SystemTime::now();
    let output_path = cli.output_dir.join("output.tiff");
    let base_color_supplied = cli.base_color.is_some();
    let base_color_override = if cli.input_mode == InputMode::Positive {
        None
    } else {
        parse_base_color_override(cli)?
    };
    let mut report = PipelineReport::new_with_metadata(RunMetadata::capture_current(
        &cli.output_dir,
        &output_path,
        run_started_at,
    ));

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
                options.write_artifacts,
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
                options.write_artifacts,
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
        options.write_artifacts,
    )?;

    let border_start = Instant::now();
    let border1 = border::remove_borders_with_diagnostics(&img1, 2);
    let border2 = border::remove_borders_with_diagnostics(&img2, 2);
    let comp1_cropped = border1.cropped;
    let comp2_cropped = border2.cropped;
    if options.write_artifacts && cli.debug {
        if let Err(err) =
            tiff_io::save_tiff_u16(&comp1_cropped, &cli.output_dir.join("comp1_cropped.tiff"))
        {
            record_failure(
                &mut report,
                &cli.output_dir,
                "border_removal",
                format!("failed to save comp1_cropped debug image: {}", err),
                border_start,
                options.write_artifacts,
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
                options.write_artifacts,
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
        options.write_artifacts,
    )?;
    drop(img1);
    drop(img2);

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
    if options.write_artifacts && cli.debug {
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
                        "base_color_source": det1.base_color_source,
                        "base_color_reason": det1.base_color_reason,
                        "base_color_proxy_confidence": det1.base_color_proxy_confidence,
                        "base_color_support_fraction": det1.base_color_support_fraction,
                        "base_strip_width": det1.strip_width,
                        "base_left_confidence": det1.left_confidence,
                        "base_right_confidence": det1.right_confidence,
                        "base_top_confidence": det1.top_confidence,
                        "base_bottom_confidence": det1.bottom_confidence,
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
                        "base_color_source": det2.base_color_source,
                        "base_color_reason": det2.base_color_reason,
                        "base_color_proxy_confidence": det2.base_color_proxy_confidence,
                        "base_color_support_fraction": det2.base_color_support_fraction,
                        "base_strip_width": det2.strip_width,
                        "base_left_confidence": det2.left_confidence,
                        "base_right_confidence": det2.right_confidence,
                        "base_top_confidence": det2.top_confidence,
                        "base_bottom_confidence": det2.bottom_confidence,
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
            let comp1_base_conf = base_detection_confidence(&det1);
            let comp2_base_conf = base_detection_confidence(&det2);
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
        options.write_artifacts,
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
            debug_dir: if options.write_artifacts && cli.debug {
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
            options.write_artifacts,
        )?;

        match stitched_image {
            Some(stitched) => {
                if options.write_artifacts && cli.debug {
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
            options.write_artifacts,
        )?;
        (fallback_component.clone(), false)
    };

    let working_start = Instant::now();
    let positive_input_mode = cli.input_mode == InputMode::Positive;
    let mut base_estimate_source_for_negative: Option<String> = None;
    let (base_color_for_negative, base_confidence) = if positive_input_mode {
        let positive_input_inspection =
            positive_input::inspect_u16_image(&working_image, cli.bit_depth);
        record_phase(
            &mut report,
            &cli.output_dir,
            {
                let mut phase = PhaseReport::ok(
                    "working_image_select",
                    1.0,
                    serde_json::json!({
                        "input_mode": cli.input_mode.as_str(),
                        "working_shape": [working_image.shape()[0], working_image.shape()[1]],
                        "fallback_component": fallback_index + 1,
                        "used_stitched_working_image": used_stitched_working_image,
                        "base_required": false,
                        "base_estimate_source": serde_json::Value::Null,
                        "base_estimate_reason": "already-positive input; film-base detection is not required for density inversion",
                        "base_color_override_applied": false,
                        "base_color_override_ignored": base_color_supplied,
                        "used_low_confidence_fallback": false,
                        "positive_input_inspection": positive_input::inspection_metrics_json(&positive_input_inspection),
                    }),
                );
                if base_color_supplied {
                    phase.warnings.push(
                        "--base-color was ignored because --input-mode positive skips density inversion"
                            .to_string(),
                    );
                }
                if positive_input_inspection.likely_negative_like {
                    phase
                        .warnings
                        .push(positive_input_inspection.reason.clone());
                }
                phase
            },
            working_start,
            options.write_artifacts,
        )?;
        (None, 1.0)
    } else {
        let det_final = base_detect::detect_film_base(&working_image);
        let raw_base_confidence = base_detection_confidence(&det_final);
        let detected_base_reconciliation = if used_stitched_working_image {
            base_detect::reconcile_stitched_base_estimate(
                &det_final,
                &[det1.base_color, det2.base_color],
            )
        } else {
            base_detect::BaseReconciliation {
                base_color: det_final.base_color,
                confidence: raw_base_confidence,
                source: det_final.base_color_source,
                reason: format!(
                    "working image was not produced from a stitched crop; using the direct base estimate: {}",
                    det_final.base_color_reason
                ),
                raw_working_base_color: det_final.base_color,
                raw_working_confidence: raw_base_confidence,
                component_consensus_base_color: None,
                component_consensus_confidence: None,
                component_consensus_relative_spread: None,
                working_vs_consensus_relative_delta: None,
                edge_balance_ratio: if raw_base_confidence <= 1e-6 {
                    1.0
                } else {
                    (det_final.left_confidence.min(det_final.right_confidence)
                        / raw_base_confidence)
                        .clamp(0.0, 1.0)
                },
            }
        };
        let base_reconciliation = if let Some(override_color) = base_color_override {
            let override_source = match cli.base_color_source.as_deref() {
                Some("roll_consensus_base") => "roll_consensus_base",
                _ => "manual_base_color_override",
            };
            let override_confidence = if override_source == "manual_base_color_override" {
                1.0
            } else {
                cli.base_color_confidence.unwrap_or(1.0).clamp(0.0, 1.0)
            };
            let override_reason = cli.base_color_reason.clone().unwrap_or_else(|| {
                if override_source == "roll_consensus_base" {
                    "roll-level density base override supplied by validation consensus".to_string()
                } else {
                    "manual density base override supplied via --base-color".to_string()
                }
            });
            base_detect::BaseReconciliation {
                base_color: override_color,
                confidence: override_confidence,
                source: override_source,
                reason: format!(
                    "{}; detected estimate was {} with confidence {:.3}: {}",
                    override_reason,
                    detected_base_reconciliation.source,
                    detected_base_reconciliation.confidence,
                    detected_base_reconciliation.reason
                ),
                raw_working_base_color: detected_base_reconciliation.raw_working_base_color,
                raw_working_confidence: detected_base_reconciliation.raw_working_confidence,
                component_consensus_base_color: detected_base_reconciliation
                    .component_consensus_base_color,
                component_consensus_confidence: detected_base_reconciliation
                    .component_consensus_confidence,
                component_consensus_relative_spread: detected_base_reconciliation
                    .component_consensus_relative_spread,
                working_vs_consensus_relative_delta: detected_base_reconciliation
                    .working_vs_consensus_relative_delta,
                edge_balance_ratio: detected_base_reconciliation.edge_balance_ratio,
            }
        } else {
            detected_base_reconciliation
        };
        base_estimate_source_for_negative = Some(base_reconciliation.source.to_string());
        let base_color = base_reconciliation.base_color;
        if options.write_artifacts && cli.debug {
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
                        "input_mode": cli.input_mode.as_str(),
                        "working_shape": [working_image.shape()[0], working_image.shape()[1]],
                        "fallback_component": fallback_index + 1,
                        "used_stitched_working_image": used_stitched_working_image,
                        "base_required": true,
                        "base_color": base_color,
                        "raw_base_color": base_reconciliation.raw_working_base_color,
                        "raw_base_confidence": base_reconciliation.raw_working_confidence,
                        "raw_base_proxy_confidence": det_final.base_color_proxy_confidence,
                        "raw_base_support_fraction": det_final.base_color_support_fraction,
                        "base_estimate_source": base_reconciliation.source,
                        "base_estimate_reason": base_reconciliation.reason,
                        "base_color_override_applied": base_color_override.is_some(),
                        "component_consensus_base_color": base_reconciliation.component_consensus_base_color,
                        "component_consensus_confidence": base_reconciliation.component_consensus_confidence,
                        "component_consensus_relative_spread": base_reconciliation.component_consensus_relative_spread,
                        "working_vs_component_consensus_relative_delta": base_reconciliation.working_vs_consensus_relative_delta,
                        "edge_balance_ratio": base_reconciliation.edge_balance_ratio,
                        "left_confidence": det_final.left_confidence,
                        "right_confidence": det_final.right_confidence,
                        "top_confidence": det_final.top_confidence,
                        "bottom_confidence": det_final.bottom_confidence,
                        "used_low_confidence_fallback": base_confidence < BASE_CONFIDENCE_FALLBACK,
                    }),
                );
                if base_reconciliation.source == "manual_base_color_override" {
                    phase.warnings.push(format!(
                        "manual base color override applied for density inversion; detected base confidence was {:.3}",
                        base_reconciliation.raw_working_confidence
                    ));
                } else if base_reconciliation.source == "roll_consensus_base" {
                    phase.warnings.push(format!(
                        "roll consensus base applied for density inversion with confidence {:.3}; detected frame-local base confidence was {:.3}",
                        base_reconciliation.confidence,
                        base_reconciliation.raw_working_confidence
                    ));
                } else if base_reconciliation.source != "working_edges" {
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
            options.write_artifacts,
        )?;
        (Some(base_color), base_confidence)
    };
    drop(comp1_cropped);
    drop(comp2_cropped);
    drop(fallback_component);

    let mut positive_for_direct: Option<Array3<f64>> = None;
    let mut direct_density_render_scale: Option<f64> = None;
    let (transmittance, skipped_ica_for_direct_density) = if positive_input_mode {
        let density_start = Instant::now();
        let positive_rgb =
            density::normalize_positive_scan_to_linear_rgb(&working_image, cli.bit_depth);
        if options.write_artifacts && cli.debug {
            tiff_io::save_tiff_f64(
                &positive_rgb,
                &cli.output_dir.join("phase3_positive_passthrough.tiff"),
            )?;
        }
        record_phase(
            &mut report,
            &cli.output_dir,
            PhaseReport::ok(
                "density_inversion",
                1.0,
                serde_json::json!({
                    "skipped": true,
                    "input_mode": cli.input_mode.as_str(),
                    "reason": "already-positive input normalized directly; density inversion and orange-mask removal are not required",
                    "bit_depth": cli.bit_depth,
                    "input_domain": "scanner_rgb",
                    "output_domain": "linear_rgb_0_1",
                    "base_required": false,
                    "output_shape": [positive_rgb.shape()[0], positive_rgb.shape()[1]],
                }),
            ),
            density_start,
            options.write_artifacts,
        )?;
        drop(working_image);

        let ica_start = Instant::now();
        record_phase(
            &mut report,
            &cli.output_dir,
            PhaseReport::ok(
                "fastica",
                1.0,
                serde_json::json!({
                    "skipped": true,
                    "input_mode": cli.input_mode.as_str(),
                    "reason": "already-positive input bypasses negative-film density separation",
                    "converged": serde_json::Value::Null,
                    "iterations": serde_json::Value::Null,
                    "permutation": serde_json::Value::Null,
                    "signs": serde_json::Value::Null,
                    "normalization": serde_json::Value::Null,
                    "sign_resolution": "not run",
                }),
            ),
            ica_start,
            options.write_artifacts,
        )?;
        (positive_rgb, false)
    } else {
        let density_start = Instant::now();
        let base_color = require_pipeline_state(
            base_color_for_negative,
            &mut report,
            &cli.output_dir,
            "density_inversion",
            "negative input mode reached density inversion without a film-base color",
            density_start,
            options.write_artifacts,
        )?;
        let density::Phase3Result {
            positive_density,
            diagnostics,
        } = density::phase3_invert_with_diagnostics(&working_image, &base_color, cli.bit_depth);
        direct_density_render_scale = Some(diagnostics.shared_robust_d_max.max(1e-6));
        positive_for_direct = Some(positive_density);
        if options.write_artifacts && cli.debug {
            let positive = require_pipeline_state_ref(
                positive_for_direct.as_ref(),
                &mut report,
                &cli.output_dir,
                "density_inversion",
                "positive density image was not available for debug artifact export",
                density_start,
                options.write_artifacts,
            )?;
            tiff_io::save_tiff_f64(positive, &cli.output_dir.join("phase3_positive.tiff"))?;
            let linear_diag = density::linear_division_diagnostic_image(
                &working_image,
                &base_color,
                cli.bit_depth,
                &std::array::from_fn(|c| diagnostics.linear_division[c].robust_high_percentile),
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
                        "skipped": false,
                        "input_mode": cli.input_mode.as_str(),
                        "base_color": base_color,
                        "base_estimate_source": base_estimate_source_for_negative,
                        "bit_depth": cli.bit_depth,
                        "base_transmittance": diagnostics.base_transmittance,
                        "base_density": diagnostics.base_density,
                        "robust_d_max": diagnostics.robust_d_max,
                        "shared_robust_d_max": diagnostics.shared_robust_d_max,
                        "direct_density_render_density_scale": diagnostics.shared_robust_d_max.max(1e-6),
                        "direct_density_render_normalization": "positive_density_divided_by_shared_robust_d_max_before_transmittance",
                        "exact_d_max": diagnostics.exact_d_max,
                        "d_max_percentile": diagnostics.d_max_percentile,
                        "histogram_bins": diagnostics.histogram_bins,
                        "clamped_to_zero": diagnostics.clamped_to_zero,
                        "epsilon_t": diagnostics.epsilon_t,
                        "density_stats": diagnostics
                            .density_stats
                            .iter()
                            .map(|stats| serde_json::json!({
                                "min_value": stats.min_value,
                                "max_value": stats.max_value,
                                "high_percentile": stats.high_percentile,
                            }))
                            .collect::<Vec<_>>(),
                        "linear_division": diagnostics
                            .linear_division
                            .iter()
                            .map(|stats| serde_json::json!({
                                "exact_max": stats.exact_max,
                                "robust_high_percentile": stats.robust_high_percentile,
                                "percentile": stats.percentile,
                            }))
                            .collect::<Vec<_>>(),
                        "output_shape": [
                            positive_for_direct.as_ref().map(|positive| positive.shape()[0]).unwrap_or(0),
                            positive_for_direct.as_ref().map(|positive| positive.shape()[1]).unwrap_or(0),
                        ],
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
            options.write_artifacts,
        )?;
        drop(working_image);

        let ica_start = Instant::now();
        let (transmittance, skipped_ica_for_direct_density) = if cli.render_input
            == RenderInputMode::DirectDensity
        {
            let mut ica_phase = PhaseReport::ok(
                "fastica",
                1.0,
                serde_json::json!({
                    "skipped": true,
                    "input_mode": cli.input_mode.as_str(),
                    "reason": "skipped because --render-input direct-density was selected",
                    "converged": serde_json::Value::Null,
                    "iterations": serde_json::Value::Null,
                    "permutation": serde_json::Value::Null,
                    "signs": serde_json::Value::Null,
                    "normalization": serde_json::Value::Null,
                    "sign_resolution": "not run",
                }),
            );
            apply_downstream_base_confidence_limit(&mut ica_phase, base_confidence);
            ica_phase.warnings.push(
                "ICA skipped because direct-density transmittance was explicitly selected"
                    .to_string(),
            );
            record_phase(
                &mut report,
                &cli.output_dir,
                ica_phase,
                ica_start,
                options.write_artifacts,
            )?;
            let positive = require_pipeline_state_ref(
                positive_for_direct.as_ref(),
                &mut report,
                &cli.output_dir,
                "fastica",
                "positive density image was not available for direct-density render input",
                ica_start,
                options.write_artifacts,
            )?;
            let density_scale = require_pipeline_state(
                direct_density_render_scale,
                &mut report,
                &cli.output_dir,
                "fastica",
                "direct-density render scale was not available for direct-density render input",
                ica_start,
                options.write_artifacts,
            )?;
            (
                density::density_image_to_normalized_transmittance(positive, density_scale),
                true,
            )
        } else {
            let positive = require_pipeline_state_ref(
                positive_for_direct.as_ref(),
                &mut report,
                &cli.output_dir,
                "fastica",
                "positive density image was not available for ICA",
                ica_start,
                options.write_artifacts,
            )?;
            let ica_result = ica::run_fastica(positive, cli.ica_max_iter, cli.ica_tol);
            let separated = ica::normalize_separated_density_channels(&ica_result.separated);
            let transmittance =
                density::density_image_to_transmittance(&separated.normalized_density);
            if options.write_artifacts && cli.debug {
                tiff_io::save_tiff_f64(
                    &separated.normalized_density,
                    &cli.output_dir.join("phase4_ica.tiff"),
                )?;
            }
            let mut ica_phase = PhaseReport::ok(
                "fastica",
                if ica_result.converged { 1.0 } else { 0.5 },
                serde_json::json!({
                    "skipped": false,
                    "input_mode": cli.input_mode.as_str(),
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
            record_phase(
                &mut report,
                &cli.output_dir,
                ica_phase,
                ica_start,
                options.write_artifacts,
            )?;
            drop(ica_result);
            drop(separated);
            (transmittance, false)
        };
        if skipped_ica_for_direct_density {
            drop(positive_for_direct.take());
        }
        (transmittance, skipped_ica_for_direct_density)
    };

    let colorspace_start = Instant::now();
    let mut calibration = color_calibration::load_calibration(
        cli.calibration_profile.as_deref(),
        cli.calibration_library.as_deref(),
        cli.scanner_profile.as_deref(),
        cli.roll_profile.as_deref(),
        cli.film_stock.as_deref(),
        base_color_for_negative,
    );
    if calibration.diagnostics.status == "not_configured" {
        if let Some(dng_calibration) =
            automatic_dng_advisory_calibration(cli, &load1_diag, &load2_diag)
        {
            calibration = dng_calibration;
        }
    }
    let calibration_profile = calibration.profile.as_ref();
    let positive_rgb_passthrough = positive_input_mode
        && calibration_profile.is_none()
        && cli.color_mode == colorspace::ColorMode::Auto;
    let mut calibration_diagnostics = calibration.diagnostics.clone();
    if positive_rgb_passthrough && calibration_diagnostics.status == "not_configured" {
        calibration_diagnostics.source = "positive_rgb_passthrough".to_string();
        calibration_diagnostics.reason =
            "no external calibration profile provided; positive RGB passthrough preserves the already-positive scan"
                .to_string();
    }
    let calibration_metrics = serde_json::to_value(&calibration_diagnostics)?;
    let mut colorspace_result = if positive_rgb_passthrough {
        colorspace::map_positive_scan_rgb_to_prophoto_d50_with_diagnostics(&transmittance)
    } else {
        match colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &transmittance,
            calibration_profile,
            cli.color_mode,
        ) {
            Ok(result) => result,
            Err(message) => {
                record_failure(
                    &mut report,
                    &cli.output_dir,
                    "colorspace_mapping",
                    message.clone(),
                    colorspace_start,
                    options.write_artifacts,
                )?;
                return Err(message.into());
            }
        }
    };
    let selected_input_candidate_metrics =
        colorspace_candidate_metrics_json(&colorspace_result.diagnostics);
    let ica_candidate_metrics = if positive_input_mode || skipped_ica_for_direct_density {
        serde_json::Value::Null
    } else {
        selected_input_candidate_metrics.clone()
    };
    let mut direct_density_candidate_evaluated =
        !positive_input_mode && skipped_ica_for_direct_density;
    let mut direct_density_candidate_metrics =
        if !positive_input_mode && skipped_ica_for_direct_density {
            selected_input_candidate_metrics
        } else {
            serde_json::Value::Null
        };
    let mut render_input_source = if positive_input_mode {
        POSITIVE_RENDER_INPUT_SOURCE
    } else if skipped_ica_for_direct_density {
        "direct_density_transmittance"
    } else {
        "fastica_separated_transmittance"
    };
    let mut render_input_reason = if positive_input_mode {
        POSITIVE_RENDER_INPUT_REASON.to_string()
    } else if skipped_ica_for_direct_density {
        "direct density transmittance selected by --render-input direct-density".to_string()
    } else {
        "ICA-separated density channels remained stable enough for colorspace mapping".to_string()
    };
    let mut render_input_warning: Option<String> =
        (!positive_input_mode && skipped_ica_for_direct_density).then(|| {
            "direct density transmittance selected by --render-input direct-density".to_string()
        });
    let mut selected_direct_density_transmittance: Option<Array3<f64>> = None;
    let mut direct_density_detail_guide: Option<Array3<f64>> = None;

    let should_evaluate_direct_density = !positive_input_mode
        && match cli.render_input {
            RenderInputMode::Auto => {
                cli.color_mode == colorspace::ColorMode::Auto
                    || colorspace::has_destructive_gamut_fallback(&colorspace_result.diagnostics)
            }
            RenderInputMode::DirectDensity => false,
            RenderInputMode::Ica => false,
        };
    if should_evaluate_direct_density {
        let positive = require_pipeline_state_ref(
            positive_for_direct.as_ref(),
            &mut report,
            &cli.output_dir,
            "colorspace_mapping",
            "positive density image was not available for direct-density candidate evaluation",
            colorspace_start,
            options.write_artifacts,
        )?;
        let density_scale = require_pipeline_state(
            direct_density_render_scale,
            &mut report,
            &cli.output_dir,
            "colorspace_mapping",
            "direct-density render scale was not available for candidate evaluation",
            colorspace_start,
            options.write_artifacts,
        )?;
        let direct_transmittance =
            density::density_image_to_normalized_transmittance(positive, density_scale);
        let direct_result =
            match colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
                &direct_transmittance,
                calibration_profile,
                cli.color_mode,
            ) {
                Ok(result) => result,
                Err(message) => {
                    record_failure(
                        &mut report,
                        &cli.output_dir,
                        "colorspace_mapping",
                        message.clone(),
                        colorspace_start,
                        options.write_artifacts,
                    )?;
                    return Err(message.into());
                }
            };
        direct_density_candidate_evaluated = true;
        direct_density_candidate_metrics =
            colorspace_candidate_metrics_json(&direct_result.diagnostics);

        let direct_selection_reason = match cli.render_input {
            RenderInputMode::DirectDensity => Some(
                "direct density transmittance selected by --render-input direct-density"
                    .to_string(),
            ),
            RenderInputMode::Auto => colorspace::direct_density_render_fallback_reason(
                &colorspace_result.diagnostics,
                &direct_result.diagnostics,
            ),
            RenderInputMode::Ica => None,
        };

        if let Some(reason) = direct_selection_reason {
            if options.write_artifacts && cli.debug {
                tiff_io::save_tiff_f64(
                    &colorspace_result.prophoto,
                    &cli.output_dir.join("phase46_ica_candidate.tiff"),
                )?;
            }
            render_input_source = "direct_density_transmittance";
            render_input_reason = reason.clone();
            render_input_warning = Some(reason);
            selected_direct_density_transmittance = Some(direct_transmittance);
            colorspace_result = direct_result;
        } else {
            direct_density_detail_guide = Some(direct_result.prophoto);
        }
    }
    let selected_colorspace_input = selected_direct_density_transmittance
        .as_ref()
        .unwrap_or(&transmittance);
    let comparison_thumbnail = colorspace_result.comparison_thumbnail;
    let mut prophoto = colorspace_result.prophoto;
    let detail_fusion_diagnostics = if render_input_source == "fastica_separated_transmittance" {
        if let Some(guide) = direct_density_detail_guide.as_ref() {
            let fusion = tonemap::fuse_scene_referred_luminance_detail(&prophoto, guide);
            prophoto = fusion.image;
            fusion.diagnostics
        } else if direct_density_candidate_evaluated {
            tonemap::SceneReferredDetailFusionDiagnostics::disabled(
                "direct-density candidate was selected as the render input",
            )
        } else {
            tonemap::SceneReferredDetailFusionDiagnostics::disabled(
                "direct-density detail guide was not evaluated",
            )
        }
    } else {
        let reason = if positive_input_mode {
            "selected render input is already-positive RGB"
        } else {
            "selected render input already uses direct-density transmittance"
        };
        tonemap::SceneReferredDetailFusionDiagnostics::disabled(reason)
    };
    let mut color_candidate_comparison_artifact: Option<String> = None;
    let mut gamut_clipping_map_artifact: Option<String> = None;
    let mut gamut_clipping_map_diagnostics = serde_json::Value::Null;
    let mut scene_referred_prophoto_float_artifact: Option<String> = None;
    let mut scene_referred_prophoto_float_artifact_diagnostics = serde_json::Value::Null;
    if options.write_artifacts && cli.debug {
        let path = cli
            .output_dir
            .join("phase46_scene_referred_prophoto_float.tiff");
        tiff_io::save_tiff_f32_linear_prophoto_scene_referred(&prophoto, &path)?;
        scene_referred_prophoto_float_artifact = Some(path.to_string_lossy().to_string());
        scene_referred_prophoto_float_artifact_diagnostics =
            scene_referred_artifact_diagnostics(&prophoto);
        tiff_io::save_tiff_f64_linear_prophoto(
            &prophoto,
            &cli.output_dir.join("phase46_prophoto.tiff"),
        )?;
        if let Some(comparison_thumbnail) = &comparison_thumbnail {
            let path = cli
                .output_dir
                .join("phase46_color_candidate_comparison.tiff");
            tiff_io::save_tiff_f64(comparison_thumbnail, &path)?;
            color_candidate_comparison_artifact = Some(path.to_string_lossy().to_string());
        }
        let (gamut_map, gamut_diagnostics) = colorspace::build_gamut_clipping_map(
            selected_colorspace_input,
            &colorspace_result.diagnostics,
        );
        let path = cli.output_dir.join("phase46_gamut_clipping_map.tiff");
        tiff_io::save_tiff_u16(&gamut_map, &path)?;
        gamut_clipping_map_artifact = Some(path.to_string_lossy().to_string());
        gamut_clipping_map_diagnostics = serde_json::json!(gamut_diagnostics);
    }
    let mut selected_colorspace_metrics =
        colorspace_candidate_metrics_json(&colorspace_result.diagnostics);
    if let serde_json::Value::Object(metrics) = &mut selected_colorspace_metrics {
        metrics.insert("target".to_string(), serde_json::json!("ProPhoto_D50"));
        metrics.insert(
            "input_domain".to_string(),
            serde_json::json!(if positive_input_mode {
                "linear_positive_scan_rgb"
            } else {
                "linear_transmittance"
            }),
        );
        metrics.insert(
            "render_buffer_domain".to_string(),
            serde_json::json!("scene_referred_linear_prophoto_rgb_d50"),
        );
        metrics.insert(
            "post_scale_high_headroom_preserved_for_tone_mapping".to_string(),
            serde_json::json!(true),
        );
        metrics.insert(
            "post_scale_high_headroom_ratio".to_string(),
            serde_json::json!(colorspace_result.diagnostics.post_scale_clipped_high_ratio),
        );
        metrics.insert(
            "color_mode".to_string(),
            serde_json::json!(cli.color_mode.as_str()),
        );
        metrics.insert(
            "render_input_mode".to_string(),
            serde_json::json!(cli.render_input.as_str()),
        );
        metrics.insert(
            "render_intent".to_string(),
            serde_json::json!(cli.render_intent.as_str()),
        );
        metrics.insert(
            "quality_mode".to_string(),
            serde_json::json!(cli.quality_mode.as_str()),
        );
        metrics.insert(
            "input_mode".to_string(),
            serde_json::json!(cli.input_mode.as_str()),
        );
        metrics.insert("calibration".to_string(), calibration_metrics);
        metrics.insert(
            "render_input_source".to_string(),
            serde_json::json!(render_input_source),
        );
        metrics.insert(
            "render_input_reason".to_string(),
            serde_json::json!(render_input_reason),
        );
        if let Some(summary) = metrics
            .get_mut("color_decision_summary")
            .and_then(serde_json::Value::as_object_mut)
        {
            summary.insert(
                "render_input_source".to_string(),
                serde_json::json!(render_input_source),
            );
            summary.insert(
                "render_input_reason".to_string(),
                serde_json::json!(render_input_reason),
            );
            summary.insert(
                "render_input_fallback_used".to_string(),
                serde_json::json!(render_input_source == "direct_density_transmittance"),
            );
        }
        metrics.insert(
            "direct_density_candidate_evaluated".to_string(),
            serde_json::json!(direct_density_candidate_evaluated),
        );
        metrics.insert("ica_candidate".to_string(), ica_candidate_metrics);
        metrics.insert(
            "direct_density_candidate".to_string(),
            direct_density_candidate_metrics,
        );
        metrics.insert(
            "direct_density_render_density_scale".to_string(),
            serde_json::json!(direct_density_render_scale),
        );
        metrics.insert(
            "direct_density_render_normalization".to_string(),
            serde_json::json!(direct_density_render_scale.map(|_| {
                "positive_density_divided_by_shared_robust_d_max_before_transmittance"
            })),
        );
        metrics.insert(
            "scene_referred_detail_fusion".to_string(),
            scene_referred_detail_fusion_metrics_json(&detail_fusion_diagnostics),
        );
        metrics.insert(
            "output_shape".to_string(),
            serde_json::json!([prophoto.shape()[0], prophoto.shape()[1]]),
        );
        metrics.insert(
            "color_candidate_comparison_artifact".to_string(),
            serde_json::json!(color_candidate_comparison_artifact),
        );
        metrics.insert(
            "gamut_clipping_map_artifact".to_string(),
            serde_json::json!(gamut_clipping_map_artifact),
        );
        metrics.insert(
            "gamut_clipping_map_diagnostics".to_string(),
            gamut_clipping_map_diagnostics,
        );
        metrics.insert(
            "scene_referred_prophoto_float_artifact".to_string(),
            serde_json::json!(scene_referred_prophoto_float_artifact),
        );
        metrics.insert(
            "scene_referred_prophoto_float_artifact_diagnostics".to_string(),
            scene_referred_prophoto_float_artifact_diagnostics,
        );
    }
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let mut phase = PhaseReport::ok("colorspace_mapping", 1.0, selected_colorspace_metrics);
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
            if calibration.diagnostics.status == "rejected" {
                phase.warnings.push(format!(
                    "calibration profile was rejected; using image-derived colorspace mapping: {}",
                    calibration.diagnostics.reason
                ));
            }
            let color_diagnostics = &colorspace_result.diagnostics;
            if color_diagnostics
                .calibration_acceptance
                .status
                .starts_with("rejected_")
            {
                phase.warnings.push(format!(
                    "colorspace calibration candidate was {}: {}",
                    color_diagnostics.calibration_acceptance.status,
                    color_diagnostics.calibration_acceptance.reason
                ));
            }
            for rejection in &color_diagnostics.selection_rejections {
                phase
                    .warnings
                    .push(format!("colorspace candidate rejected: {rejection}"));
            }
            if color_diagnostics.candidate_risk != "safe" {
                phase.warnings.push(format!(
                    "colorspace selected candidate `{}` has risk `{}`; tone color trust state `{}`",
                    color_diagnostics.selected_candidate,
                    color_diagnostics.candidate_risk,
                    colorspace::tone_color_trust_state(color_diagnostics)
                ));
            }
            if let Some(reference) = &color_diagnostics.reference_patch_evaluation {
                if !reference.selected_hue_family_regressions.is_empty() {
                    let hue_families = reference
                        .selected_hue_family_regressions
                        .iter()
                        .map(|regression| regression.hue_family.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    phase.warnings.push(format!(
                        "colorspace reference-patch fit regressed hue families versus image-derived mapping: {hue_families}"
                    ));
                }
            }
            if let Some(library) = &calibration.diagnostics.library {
                if !library.invalid_entries.is_empty() {
                    phase.warnings.push(format!(
                        "calibration library ignored {} invalid entr{}; see calibration.library.invalid_entries for details",
                        library.invalid_entries.len(),
                        if library.invalid_entries.len() == 1 { "y" } else { "ies" }
                    ));
                }
            }
            if calibration.diagnostics.status == "not_configured"
                && calibration
                    .diagnostics
                    .scanner_profile
                    .as_ref()
                    .is_some_and(|scanner| scanner.status == "advisory")
            {
                if let Some(scanner) = &calibration.diagnostics.scanner_profile {
                    phase.warnings.push(format!(
                        "scanner calibration profile was only advisory and was not applied: {}",
                        scanner.reason
                    ));
                }
            }
            if colorspace_result
                .diagnostics
                .weak_anchor_fallback_reason
                .is_none()
                && colorspace_result.diagnostics.mapping_strategy == "image_derived_matrix"
            {
                if let Some(warning) =
                    colorspace::weak_channel_anchor_warning(&colorspace_result.diagnostics)
                {
                    phase.warnings.push(warning);
                }
            }
            if colorspace_result.diagnostics.exposure_scale > 1.0 + 1e-6 {
                phase.warnings.push(format!(
                    "colorspace mapping applied {:.3}x highlight headroom normalization while preserving sparse scene-referred ProPhoto excursions for tone mapping because mapped channels exceeded 1.0 at the {:.1}% percentile",
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
        options.write_artifacts,
    )?;

    let auto_exposure_ev = if positive_input_mode {
        tonemap::positive_scan_auto_exposure_ev(&prophoto)
    } else {
        0.0
    };
    let tone_fit = if auto_exposure_ev.abs() > f64::EPSILON {
        let exposure_scale = 2.0f64.powf(auto_exposure_ev);
        let exposed = prophoto.mapv(|v| (v * exposure_scale).max(0.0));
        if positive_input_mode {
            tonemap::fit_positive_scan_tone_params_with_diagnostics(&exposed)
        } else {
            tonemap::fit_tone_params_with_diagnostics(&exposed)
        }
    } else if positive_input_mode {
        tonemap::fit_positive_scan_tone_params_with_diagnostics(&prophoto)
    } else {
        tonemap::fit_tone_params_with_diagnostics(&prophoto)
    };
    let tone_color_protection =
        tonemap::ToneColorProtection::from_colorspace_diagnostics(&colorspace_result.diagnostics);
    Ok(InteractiveRenderCache {
        prophoto,
        auto_exposure_ev,
        auto_tone_params: tone_fit.params,
        tone_fit_diagnostics: tone_fit.diagnostics,
        tone_color_protection,
        report,
        cli: cli.clone(),
        output_path,
        run_started_at,
        base_confidence,
    })
}

pub fn build_interactive_render_cache(
    cli: &Cli,
) -> Result<InteractiveRenderCache, Box<dyn std::error::Error>> {
    build_interactive_render_cache_with_options(
        cli,
        PipelineBuildOptions {
            write_artifacts: false,
        },
    )
}

/// Run the full scanstitch pipeline.
pub fn run(cli: &Cli) -> Result<PipelineReport, Box<dyn std::error::Error>> {
    let cache = build_interactive_render_cache_with_options(
        cli,
        PipelineBuildOptions {
            write_artifacts: true,
        },
    )?;
    let (controls, applied_review_sidecar) = controls_for_cache(&cache)?;
    let report = save_interactive_render_with_options(
        &cache,
        &controls,
        InteractiveSaveOptions {
            write_partial_report: true,
            write_debug_artifacts: true,
            save_report: true,
            include_interactive_metrics: false,
            applied_review_sidecar,
        },
    )?;

    log::info!(
        "Pipeline complete. {} phases recorded.",
        report.phases.len()
    );
    Ok(report)
}

pub fn save_interactive_render(
    cache: &InteractiveRenderCache,
    controls: &InteractiveRenderControls,
) -> Result<PipelineReport, Box<dyn std::error::Error>> {
    let applied_review_sidecar = review_sidecar_metadata_for_cache(cache)?;
    save_interactive_render_with_options(
        cache,
        controls,
        InteractiveSaveOptions {
            write_partial_report: false,
            write_debug_artifacts: false,
            save_report: true,
            include_interactive_metrics: true,
            applied_review_sidecar,
        },
    )
}

fn save_interactive_render_with_options(
    cache: &InteractiveRenderCache,
    controls: &InteractiveRenderControls,
    options: InteractiveSaveOptions,
) -> Result<PipelineReport, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&cache.cli.output_dir)?;
    let cli = &cache.cli;
    let output_path: PathBuf = cache.output_path.clone();
    let mut report = cache.report.clone();
    let tone_params = controls.to_tone_params(&cache.auto_tone_params);

    let tone_start = Instant::now();
    let tone_apply = interactive::render_interactive_image(cache, controls);
    let tonemapped = tone_apply.image;
    let render_quality = tonemap::render_quality_diagnostics(&tonemapped);
    let render_grain = tonemap::render_grain_diagnostics(&tonemapped);
    if options.write_debug_artifacts && cli.debug {
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
            let tone_fit = &cache.tone_fit_diagnostics;
            let mut phase = PhaseReport::ok(
                "tone_mapping",
                1.0,
                serde_json::json!({
                    "midpoint": tone_params.midpoint,
                    "slope": tone_params.slope,
                    "toe_lift": tone_params.toe_lift,
                    "shoulder_max": tone_params.shoulder_max,
                    "render_intent": cli.render_intent.as_str(),
                    "quality_mode": cli.quality_mode.as_str(),
                    "fit_domain": tone_fit.fit_domain,
                    "perceptual_luminance_gain": tone_fit.perceptual_luminance_gain,
                    "input_linear_percentiles": tone_fit.input_linear_percentiles,
                    "input_perceptual_percentiles": tone_fit.input_perceptual_percentiles,
                    "mapped_linear_percentiles": tone_fit.mapped_linear_percentiles,
                    "mapped_perceptual_percentiles": tone_fit.mapped_perceptual_percentiles,
                    "highlight_chroma_compressed_ratio": tone_apply.diagnostics.highlight_chroma_compressed_ratio,
                    "highlight_neutral_chroma_compressed_ratio": tone_apply.diagnostics.highlight_neutral_chroma_compressed_ratio,
                    "highlight_neutral_chroma_enabled": tone_apply.diagnostics.highlight_neutral_chroma_enabled,
                    "highlight_neutral_chroma_start_luminance": tone_apply.diagnostics.highlight_neutral_chroma_start_luminance,
                    "highlight_neutral_chroma_full_luminance": tone_apply.diagnostics.highlight_neutral_chroma_full_luminance,
                    "highlight_neutral_chroma_min_scale": tone_apply.diagnostics.highlight_neutral_chroma_min_scale,
                    "highlight_neutral_chroma_max_saturation": tone_apply.diagnostics.highlight_neutral_chroma_max_saturation,
                    "shadow_chroma_compressed_ratio": tone_apply.diagnostics.shadow_chroma_compressed_ratio,
                    "shadow_chroma_enabled": tone_apply.diagnostics.shadow_chroma_enabled,
                    "shadow_chroma_start_luminance": tone_apply.diagnostics.shadow_chroma_start_luminance,
                    "shadow_chroma_full_luminance": tone_apply.diagnostics.shadow_chroma_full_luminance,
                    "shadow_chroma_min_scale": tone_apply.diagnostics.shadow_chroma_min_scale,
                    "color_protection_policy": tone_apply.diagnostics.color_protection_policy,
                    "color_trust_state": tone_apply.diagnostics.color_trust_state,
                    "color_protection_reason": tone_apply.diagnostics.color_protection_reason.clone(),
                    "pre_chroma_compression_clipped_high_ratio": tone_apply.diagnostics.pre_chroma_compression_clipped_high_ratio,
                    "post_chroma_compression_clipped_high_ratio": tone_apply.diagnostics.post_chroma_compression_clipped_high_ratio,
                    "post_chroma_compression_clipped_low_ratio": tone_apply.diagnostics.post_chroma_compression_clipped_low_ratio,
                    "local_luminance_detail_enabled": tone_apply.diagnostics.local_luminance_detail_enabled,
                    "local_luminance_detail_radius": tone_apply.diagnostics.local_luminance_detail_radius,
                    "local_luminance_detail_amount": tone_apply.diagnostics.local_luminance_detail_amount,
                    "local_luminance_detail_max_ev": tone_apply.diagnostics.local_luminance_detail_max_ev,
                    "local_luminance_detail_applied_ratio": tone_apply.diagnostics.local_luminance_detail_applied_ratio,
                    "local_luminance_detail_mean_abs_ev": tone_apply.diagnostics.local_luminance_detail_mean_abs_ev,
                    "local_luminance_detail_max_abs_ev": tone_apply.diagnostics.local_luminance_detail_max_abs_ev,
                    "local_luminance_detail_headroom_limited_ratio": tone_apply.diagnostics.local_luminance_detail_headroom_limited_ratio,
                    "local_luminance_detail_clip_limited_ratio": tone_apply.diagnostics.local_luminance_detail_clip_limited_ratio,
                    "high_frequency_grain": render_grain_metrics_json(&render_grain),
                }),
            );
            if let serde_json::Value::Object(metrics) = &mut phase.metrics {
                metrics.insert(
                    "midtone_neutral_chroma_compressed_ratio".to_string(),
                    serde_json::json!(
                        tone_apply
                            .diagnostics
                            .midtone_neutral_chroma_compressed_ratio
                    ),
                );
                metrics.insert(
                    "midtone_neutral_chroma_enabled".to_string(),
                    serde_json::json!(tone_apply.diagnostics.midtone_neutral_chroma_enabled),
                );
                metrics.insert(
                    "midtone_neutral_chroma_start_luminance".to_string(),
                    serde_json::json!(
                        tone_apply
                            .diagnostics
                            .midtone_neutral_chroma_start_luminance
                    ),
                );
                metrics.insert(
                    "midtone_neutral_chroma_end_luminance".to_string(),
                    serde_json::json!(tone_apply.diagnostics.midtone_neutral_chroma_end_luminance),
                );
                metrics.insert(
                    "midtone_neutral_chroma_min_scale".to_string(),
                    serde_json::json!(tone_apply.diagnostics.midtone_neutral_chroma_min_scale),
                );
                metrics.insert(
                    "midtone_neutral_chroma_max_saturation".to_string(),
                    serde_json::json!(tone_apply.diagnostics.midtone_neutral_chroma_max_saturation),
                );
                metrics.insert(
                    "tone_fit_policy".to_string(),
                    serde_json::json!(if cli.input_mode == InputMode::Positive {
                        "positive_scan_rgb"
                    } else {
                        "negative_film"
                    }),
                );
                metrics.insert(
                    "auto_exposure_ev".to_string(),
                    serde_json::json!(cache.auto_exposure_ev),
                );
                metrics.insert(
                    "auto_exposure_scale".to_string(),
                    serde_json::json!(2.0f64.powf(cache.auto_exposure_ev)),
                );
                metrics.insert(
                    "render_exposure_ev".to_string(),
                    serde_json::json!(controls.exposure_ev),
                );
                metrics.insert(
                    "render_exposure_scale".to_string(),
                    serde_json::json!(2.0f64.powf(controls.exposure_ev)),
                );
                metrics.insert(
                    "adaptive_vibrance_enabled".to_string(),
                    serde_json::json!(tone_apply.diagnostics.adaptive_vibrance_enabled),
                );
                metrics.insert(
                    "adaptive_vibrance_reason".to_string(),
                    serde_json::json!(tone_apply.diagnostics.adaptive_vibrance_reason.clone()),
                );
                metrics.insert(
                    "adaptive_vibrance_amount".to_string(),
                    serde_json::json!(tone_apply.diagnostics.adaptive_vibrance_amount),
                );
                metrics.insert(
                    "adaptive_vibrance_max_scale".to_string(),
                    serde_json::json!(tone_apply.diagnostics.adaptive_vibrance_max_scale),
                );
                metrics.insert(
                    "adaptive_vibrance_applied_ratio".to_string(),
                    serde_json::json!(tone_apply.diagnostics.adaptive_vibrance_applied_ratio),
                );
                metrics.insert(
                    "adaptive_vibrance_mean_scale".to_string(),
                    serde_json::json!(tone_apply.diagnostics.adaptive_vibrance_mean_scale),
                );
                metrics.insert(
                    "adaptive_vibrance_max_applied_scale".to_string(),
                    serde_json::json!(tone_apply.diagnostics.adaptive_vibrance_max_applied_scale),
                );
                metrics.insert(
                    "adaptive_vibrance_texture_limited_ratio".to_string(),
                    serde_json::json!(
                        tone_apply
                            .diagnostics
                            .adaptive_vibrance_texture_limited_ratio
                    ),
                );
                metrics.insert(
                    "adaptive_vibrance_gamut_limited_ratio".to_string(),
                    serde_json::json!(tone_apply.diagnostics.adaptive_vibrance_gamut_limited_ratio),
                );
                metrics.insert(
                    "noise_reduction_enabled".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_enabled),
                );
                metrics.insert(
                    "noise_reduction_reason".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_reason.clone()),
                );
                metrics.insert(
                    "noise_reduction_radius".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_radius),
                );
                metrics.insert(
                    "noise_reduction_chroma_amount".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_chroma_amount),
                );
                metrics.insert(
                    "noise_reduction_luma_amount".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_luma_amount),
                );
                metrics.insert(
                    "noise_reduction_applied_ratio".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_applied_ratio),
                );
                metrics.insert(
                    "noise_reduction_texture_limited_ratio".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_texture_limited_ratio),
                );
                metrics.insert(
                    "noise_reduction_saturation_limited_ratio".to_string(),
                    serde_json::json!(
                        tone_apply
                            .diagnostics
                            .noise_reduction_saturation_limited_ratio
                    ),
                );
                metrics.insert(
                    "noise_reduction_mean_abs_chroma_delta".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_mean_abs_chroma_delta),
                );
                metrics.insert(
                    "noise_reduction_max_abs_chroma_delta".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_max_abs_chroma_delta),
                );
                metrics.insert(
                    "noise_reduction_mean_abs_luma_delta".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_mean_abs_luma_delta),
                );
                metrics.insert(
                    "noise_reduction_max_abs_luma_delta".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_max_abs_luma_delta),
                );
                metrics.insert(
                    "render_quality_sample_count".to_string(),
                    serde_json::json!(render_quality.sample_count),
                );
                metrics.insert(
                    "render_quality_sample_stride".to_string(),
                    serde_json::json!(render_quality.sample_stride),
                );
                metrics.insert(
                    "render_luminance_percentiles".to_string(),
                    serde_json::json!(render_quality.overall.luminance_percentiles),
                );
                metrics.insert(
                    "render_luminance_range_p05_p95".to_string(),
                    serde_json::json!(
                        render_quality.overall.luminance_percentiles[2]
                            - render_quality.overall.luminance_percentiles[0]
                    ),
                );
                metrics.insert(
                    "shadow_luminance_max".to_string(),
                    serde_json::json!(render_quality.shadow_luminance_max),
                );
                metrics.insert(
                    "shadow_visible_luminance_min".to_string(),
                    serde_json::json!(render_quality.shadow_visible_luminance_min),
                );
                metrics.insert(
                    "midtone_luminance_range".to_string(),
                    serde_json::json!(render_quality.midtone_luminance_range),
                );
                metrics.insert(
                    "midtone_neutral_max_saturation".to_string(),
                    serde_json::json!(render_quality.midtone_neutral_max_saturation),
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
                    "shadow_visible_pixel_count".to_string(),
                    serde_json::json!(render_quality.shadow_visible.pixel_count),
                );
                metrics.insert(
                    "shadow_visible_saturation_median".to_string(),
                    serde_json::json!(render_quality.shadow_visible.saturation_median),
                );
                metrics.insert(
                    "shadow_visible_saturation_p95".to_string(),
                    serde_json::json!(render_quality.shadow_visible.saturation_p95),
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
                    "midtone_neutral_pixel_count".to_string(),
                    serde_json::json!(render_quality.midtone_neutral.pixel_count),
                );
                metrics.insert(
                    "midtone_neutral_saturation_median".to_string(),
                    serde_json::json!(render_quality.midtone_neutral.saturation_median),
                );
                metrics.insert(
                    "midtone_neutral_saturation_p95".to_string(),
                    serde_json::json!(render_quality.midtone_neutral.saturation_p95),
                );
                metrics.insert(
                    "midtone_saturated_saturation_median".to_string(),
                    serde_json::json!(render_quality.midtone_saturated.saturation_median),
                );
                metrics.insert(
                    "midtone_saturated_saturation_p95".to_string(),
                    serde_json::json!(render_quality.midtone_saturated.saturation_p95),
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
                    "shadow_visible_rgb_median".to_string(),
                    serde_json::json!(render_quality.shadow_visible.rgb_median),
                );
                metrics.insert(
                    "midtone_rgb_median".to_string(),
                    serde_json::json!(render_quality.midtone.rgb_median),
                );
                metrics.insert(
                    "midtone_neutral_rgb_median".to_string(),
                    serde_json::json!(render_quality.midtone_neutral.rgb_median),
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
                        "overall": render_band_metrics_json(&render_quality.overall),
                        "shadow": render_band_metrics_json(&render_quality.shadow),
                        "shadow_visible": render_band_metrics_json(&render_quality.shadow_visible),
                        "midtone": render_band_metrics_json(&render_quality.midtone),
                        "midtone_neutral": render_band_metrics_json(&render_quality.midtone_neutral),
                        "midtone_saturated": render_band_metrics_json(&render_quality.midtone_saturated),
                        "bright_neutral": render_band_metrics_json(&render_quality.bright_neutral),
                        "bright_saturated": render_band_metrics_json(&render_quality.bright_saturated),
                    }),
                );
                if options.include_interactive_metrics {
                    metrics.insert(
                        "interactive_controls_applied".to_string(),
                        serde_json::json!(true),
                    );
                    metrics.insert(
                        "interactive_controls".to_string(),
                        serde_json::json!(controls),
                    );
                }
            }
            let base_quality_factor =
                apply_downstream_base_confidence_limit(&mut phase, cache.base_confidence);
            if tone_apply.diagnostics.color_protection_policy != "enabled" {
                phase.warnings.push(format!(
                    "tone color protection policy `{}` is active: {}",
                    tone_apply.diagnostics.color_protection_policy,
                    tone_apply.diagnostics.color_protection_reason
                ));
            }
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
            if tone_apply
                .diagnostics
                .midtone_neutral_chroma_compressed_ratio
                > 0.001
            {
                phase.warnings.push(format!(
                    "tone mapping gently compressed midtone near-neutral chroma for {:.2}% of pixels between {:.3} and {:.3} luminance to reduce residual snow/neutral color cast",
                    tone_apply.diagnostics.midtone_neutral_chroma_compressed_ratio * 100.0,
                    tone_apply
                        .diagnostics
                        .midtone_neutral_chroma_start_luminance,
                    tone_apply
                        .diagnostics
                        .midtone_neutral_chroma_end_luminance
                ));
            }
            if tone_apply.diagnostics.shadow_chroma_compressed_ratio > 0.001 {
                phase.warnings.push(format!(
                    "tone mapping gently compressed shadow chroma for {:.2}% of pixels below {:.3} luminance to reduce low-tone color speckle",
                    tone_apply.diagnostics.shadow_chroma_compressed_ratio * 100.0,
                    tone_apply.diagnostics.shadow_chroma_start_luminance
                ));
            }
            if cli.input_mode == InputMode::Positive && cache.auto_exposure_ev < -0.01 {
                phase.warnings.push(format!(
                    "positive-mode tone placement applied {:.2} EV pre-tone exposure to retain highlight headroom and avoid overly light midtones",
                    cache.auto_exposure_ev
                ));
            }
            if cache.tone_fit_diagnostics.mapped_linear_percentiles[2] > 0.95 {
                phase.warnings.push(format!(
                    "tone fit leaves the 95th-percentile linear luminance near the shoulder ({:.3}); highlights may still appear compressed",
                    cache.tone_fit_diagnostics.mapped_linear_percentiles[2]
                ));
            }
            if cache.base_confidence < BASE_CONFIDENCE_WARN {
                phase.warnings.push(format!(
                    "tone-mapping confidence is capped by low working-image base confidence ({:.3}; quality factor {:.3})",
                    cache.base_confidence, base_quality_factor
                ));
            }
            phase
        },
        tone_start,
        options.write_partial_report,
    )?;

    let save_start = Instant::now();
    let previous_output_modified_at = file_modified_at(&output_path);
    let overwrote_existing_output = previous_output_modified_at.is_some();
    let master_path = cli
        .should_write_master()
        .then(|| cli.output_dir.join("master_scene_referred.tiff"));
    let mut master_artifact_diagnostics = serde_json::Value::Null;
    if let Some(path) = &master_path {
        if let Err(err) =
            tiff_io::save_tiff_f32_linear_prophoto_scene_referred(&cache.prophoto, path)
        {
            record_failure(
                &mut report,
                &cli.output_dir,
                "save",
                format!("failed to save scene-referred master: {}", err),
                save_start,
                options.write_partial_report,
            )?;
            return Err(err);
        }
        master_artifact_diagnostics = scene_referred_artifact_diagnostics(&cache.prophoto);
    }
    if let Err(err) = tiff_io::save_tiff_f64_linear_prophoto(&tonemapped, &output_path) {
        record_failure(
            &mut report,
            &cli.output_dir,
            "save",
            format!("failed to save final output: {}", err),
            save_start,
            options.write_partial_report,
        )?;
        return Err(err);
    }
    let review_srgb_path = cli
        .should_write_review_proof()
        .then(|| cli.output_dir.join("review_srgb.png"));
    if let Some(path) = &review_srgb_path {
        if let Err(err) = tiff_io::save_srgb_png_from_linear_prophoto(&tonemapped, path) {
            record_failure(
                &mut report,
                &cli.output_dir,
                "save",
                format!("failed to save sRGB review proof: {}", err),
                save_start,
                options.write_partial_report,
            )?;
            return Err(err);
        }
    }
    let written_review_sidecar_path = cli.write_review_sidecar.clone();
    if let Some(path) = &written_review_sidecar_path {
        if let Err(err) = interactive::write_review_sidecar(path, cache, controls) {
            record_failure(
                &mut report,
                &cli.output_dir,
                "save",
                format!("failed to write review sidecar: {}", err),
                save_start,
                options.write_partial_report,
            )?;
            return Err(err);
        }
    }
    let stale_render_artifacts =
        stale_render_artifacts_json(&cli.output_dir, &output_path, cache.run_started_at);
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let review_status = render_review_status(cache.base_confidence);
            let review_reason = render_review_reason(cache.base_confidence);
            let mut phase = PhaseReport::ok(
                "save",
                1.0,
                serde_json::json!({
                    "output_path": output_path.to_string_lossy(),
                    "output_shape": [tonemapped.shape()[0], tonemapped.shape()[1]],
                    "output_width": tonemapped.shape()[1],
                    "output_height": tonemapped.shape()[0],
                    "output_modified_at": file_modified_at_string(&output_path),
                    "output_modified_at_unix_ms": file_modified_at_unix_ms(&output_path),
                    "output_file_size_bytes": file_size_bytes(&output_path),
                    "output_color_space": "linear_prophoto_rgb_d50",
                    "output_icc_profile": {
                        "embedded": true,
                        "tiff_tag": 34675,
                        "description": tiff_io::PROPHOTO_LINEAR_ICC_DESCRIPTION,
                        "transfer_function": "linear",
                        "whitepoint": "D50"
                    },
                    "overwrote_existing_output": overwrote_existing_output,
                    "previous_output_modified_at": previous_output_modified_at.map(format_system_time_utc),
                    "previous_output_modified_at_unix_ms": previous_output_modified_at.and_then(system_time_unix_ms),
                    "stale_render_artifact_count": stale_render_artifacts.len(),
                    "stale_render_artifacts": stale_render_artifacts,
                    "input_base_confidence": cache.base_confidence,
                    "render_review_status": review_status,
                    "render_reviewable": review_status == "reviewable",
                    "render_review_reason": review_reason,
                }),
            );
            if let serde_json::Value::Object(metrics) = &mut phase.metrics {
                metrics.insert(
                    "render_intent".to_string(),
                    serde_json::json!(cli.render_intent.as_str()),
                );
                metrics.insert(
                    "quality_mode".to_string(),
                    serde_json::json!(cli.quality_mode.as_str()),
                );
                metrics.insert(
                    "master_scene_referred_path".to_string(),
                    serde_json::json!(master_path
                        .as_ref()
                        .map(|path| path.to_string_lossy().to_string())),
                );
                metrics.insert(
                    "master_scene_referred_modified_at".to_string(),
                    serde_json::json!(master_path
                        .as_ref()
                        .and_then(|path| file_modified_at_string(path))),
                );
                metrics.insert(
                    "master_scene_referred_file_size_bytes".to_string(),
                    serde_json::json!(master_path.as_ref().and_then(|path| file_size_bytes(path))),
                );
                metrics.insert(
                    "master_scene_referred_color_space".to_string(),
                    serde_json::json!(master_path
                        .as_ref()
                        .map(|_| "scene_referred_linear_prophoto_rgb_d50")),
                );
                metrics.insert(
                    "master_scene_referred_icc_profile".to_string(),
                    master_path
                        .as_ref()
                        .map(|_| {
                            serde_json::json!({
                                "embedded": true,
                                "tiff_tag": tiff_io::ICC_PROFILE_TAG,
                                "description": tiff_io::PROPHOTO_LINEAR_ICC_DESCRIPTION,
                                "transfer_function": "linear",
                                "whitepoint": "D50"
                            })
                        })
                        .unwrap_or(serde_json::Value::Null),
                );
                metrics.insert(
                    "master_scene_referred_diagnostics".to_string(),
                    master_artifact_diagnostics.clone(),
                );
                metrics.insert(
                    "review_srgb_path".to_string(),
                    serde_json::json!(review_srgb_path
                        .as_ref()
                        .map(|path| path.to_string_lossy().to_string())),
                );
                metrics.insert(
                    "review_srgb_modified_at".to_string(),
                    serde_json::json!(review_srgb_path
                        .as_ref()
                        .and_then(|path| file_modified_at_string(path))),
                );
                metrics.insert(
                    "review_srgb_file_size_bytes".to_string(),
                    serde_json::json!(review_srgb_path
                        .as_ref()
                        .and_then(|path| file_size_bytes(path))),
                );
                metrics.insert(
                    "review_srgb_color_space".to_string(),
                    serde_json::json!(review_srgb_path.as_ref().map(|_| "srgb_display_png")),
                );
                metrics.insert(
                    "review_sidecar_applied".to_string(),
                    serde_json::json!(options.applied_review_sidecar.is_some()),
                );
                metrics.insert(
                    "review_sidecar_path".to_string(),
                    serde_json::json!(options
                        .applied_review_sidecar
                        .as_ref()
                        .map(|sidecar| sidecar.path.clone())),
                );
                metrics.insert(
                    "review_sidecar_sha256".to_string(),
                    serde_json::json!(options
                        .applied_review_sidecar
                        .as_ref()
                        .map(|sidecar| sidecar.sha256.clone())),
                );
                metrics.insert(
                    "review_sidecar_controls_applied".to_string(),
                    serde_json::json!(options
                        .applied_review_sidecar
                        .as_ref()
                        .map(|sidecar| sidecar.controls_applied)),
                );
                metrics.insert(
                    "review_sidecar_mark_count".to_string(),
                    serde_json::json!(options
                        .applied_review_sidecar
                        .as_ref()
                        .map(|sidecar| sidecar.mark_count)),
                );
                metrics.insert(
                    "written_review_sidecar_path".to_string(),
                    serde_json::json!(written_review_sidecar_path
                        .as_ref()
                        .map(|path| path.to_string_lossy().to_string())),
                );
                metrics.insert(
                    "written_review_sidecar_sha256".to_string(),
                    serde_json::json!(written_review_sidecar_path
                        .as_ref()
                        .and_then(|path| file_sha256_hex(path))),
                );
                metrics.insert(
                    "artifacts".to_string(),
                    serde_json::json!({
                        "master_scene_referred": master_path
                            .as_ref()
                            .map(|path| path.to_string_lossy().to_string()),
                        "finished_render": output_path.to_string_lossy().to_string(),
                        "review_srgb": review_srgb_path
                            .as_ref()
                            .map(|path| path.to_string_lossy().to_string()),
                        "review_sidecar": written_review_sidecar_path
                            .as_ref()
                            .map(|path| path.to_string_lossy().to_string()),
                    }),
                );
            }
            if let Some(count) = phase
                .metrics
                .get("stale_render_artifact_count")
                .and_then(serde_json::Value::as_u64)
            {
                if count > 0 {
                    phase.warnings.push(format!(
                        "output directory contains {} TIFF artifact(s) from before this run that were not overwritten; clean the directory before comparing debug renders",
                        count
                    ));
                }
            }
            if review_status == "blocked_low_base_confidence" {
                phase.warnings.push(format!(
                    "render is not reviewable: {} (base confidence {:.3}); provide a real rebate/roll-base reference or inspect debug phase artifacts before judging inversion",
                    review_reason, cache.base_confidence
                ));
            } else if review_status == "caution_low_base_confidence" {
                phase.warnings.push(format!(
                    "render needs cautious review: {} (base confidence {:.3})",
                    review_reason, cache.base_confidence
                ));
            }
            phase
        },
        save_start,
        options.write_partial_report,
    )?;

    if options.save_report {
        report.save(&cli.output_dir.join("report.json"))?;
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_state_failure_is_reported_instead_of_panicking() {
        let mut report = PipelineReport::new();
        let err = require_pipeline_state::<u8>(
            None,
            &mut report,
            Path::new("."),
            "density_inversion",
            "test missing state",
            Instant::now(),
            false,
        )
        .expect_err("missing state should return a reportable error");

        assert!(err.to_string().contains("internal pipeline state error"));
        assert_eq!(report.phases.len(), 1);
        assert_eq!(report.phases[0].name, "density_inversion");
        assert!(!report.phases[0].success);
        assert!(
            report.phases[0].errors[0].contains("test missing state"),
            "failure phase should carry the missing-state message"
        );
    }
}

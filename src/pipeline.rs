use crate::atomic_file;
use crate::base_detect;
use crate::border;
use crate::cli::{Cli, DeskewMode, InputMode, RenderInputMode};
use crate::color_calibration;
use crate::colorspace;
use crate::density;
use crate::deskew;
use crate::frame_classify;
use crate::ica;
use crate::input_color;
use crate::interactive::{self, InteractiveRenderCache, InteractiveRenderControls};
use crate::positive_input;
use crate::report::{
    format_system_time_utc, hash_file_sha256, system_time_unix_ms, PhaseReport, PipelineReport,
    RunMetadata, FILE_SHA256_BUFFER_BYTES,
};
use crate::scanner_linearization;
use crate::stitch::{self, StitchConfig, TransformMode};
use crate::tiff_io;
use crate::tonemap;
use crate::white_balance;
use ndarray::{s, Array3};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

const BASE_CONFIDENCE_WARN: f64 = 0.30;
const BASE_CONFIDENCE_FALLBACK: f64 = 0.15;
const SCENE_REFERRED_ARTIFACT_DIAGNOSTIC_SAMPLE_LIMIT: usize = 1_000_000;
const TONE_OUTPUT_MIN_EVALUABLE_RANGE: f64 = 0.005;
const TONE_OUTPUT_MIN_RENDER_RANGE: f64 = 0.002;
const TONE_OUTPUT_MIN_RANGE_RETENTION_RATIO: f64 = 0.08;
const TONE_OUTPUT_MAX_CATASTROPHIC_CHANNEL_CLIP_RATIO: f64 = 0.50;
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

struct OwnedBatchRenderContext {
    image: Array3<f64>,
    tone_input_linear_percentiles: [f64; 3],
    master_artifact_diagnostics: serde_json::Value,
    master_write_duration: Duration,
    master_prewritten: bool,
}

#[derive(Debug, Clone)]
struct PreparedComponent {
    path: PathBuf,
    load_diagnostics: tiff_io::TiffLoadDiagnostics,
    embedded_icc_profile: Option<input_color::EmbeddedIccProfile>,
    border_diagnostics: border::BorderRemovalDiagnostics,
    cropped: Array3<u16>,
    detection: base_detect::BaseDetection,
    measurement: base_detect::BaseDetection,
    measurement_stage: &'static str,
    measurement_reason: String,
    analysis: frame_classify::FrameAnalysis,
}

struct BorderedComponent {
    path: PathBuf,
    load_diagnostics: tiff_io::TiffLoadDiagnostics,
    embedded_icc_profile: Option<input_color::EmbeddedIccProfile>,
    cropped: Array3<u16>,
    border_diagnostics: border::BorderRemovalDiagnostics,
    pre_crop_base: Option<base_detect::BaseDetection>,
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

fn select_base_measurement<'a>(
    pre_crop: &'a base_detect::BaseDetection,
    post_crop: &'a base_detect::BaseDetection,
) -> (&'a base_detect::BaseDetection, &'static str, String) {
    let pre_confidence = base_detection_confidence(pre_crop);
    let post_confidence = base_detection_confidence(post_crop);
    let pre_has_direct_rebate = pre_crop.base_color_source != "high_transmittance_fallback"
        && pre_confidence >= BASE_CONFIDENCE_WARN;
    let post_has_direct_rebate = post_crop.base_color_source != "high_transmittance_fallback"
        && post_confidence >= BASE_CONFIDENCE_WARN;

    if pre_has_direct_rebate && (!post_has_direct_rebate || pre_confidence > post_confidence + 0.05)
    {
        return (
            pre_crop,
            "pre_crop_rebate_measurement",
            format!(
                "pre-crop rebate evidence retained for density inversion (confidence {:.3}, source {}); post-crop evidence was {:.3} from {}",
                pre_confidence,
                pre_crop.base_color_source,
                post_confidence,
                post_crop.base_color_source
            ),
        );
    }

    (
        post_crop,
        "post_crop_measurement",
        format!(
            "post-crop base evidence retained (confidence {:.3}, source {}); pre-crop evidence was {:.3} from {}",
            post_confidence,
            post_crop.base_color_source,
            pre_confidence,
            pre_crop.base_color_source
        ),
    )
}

fn detect_base_before_content_crop(
    img: &Array3<u16>,
    diagnostics: &border::BorderRemovalDiagnostics,
    safety_margin: usize,
) -> base_detect::BaseDetection {
    let (height, width, _) = img.dim();
    let removed_band = base_detect::detect_film_base_in_removed_bands(
        img,
        diagnostics.top_removed,
        diagnostics.bottom_removed,
        diagnostics.left_removed,
        diagnostics.right_removed,
    );
    // Border removal extends slightly into the image for safety. Base measurement uses the
    // boundary before that extension, preserving a narrow rebate immediately beside a scanner
    // dead zone while still excluding the dead zone itself.
    let top = diagnostics.top_removed.saturating_sub(safety_margin);
    let bottom = diagnostics.bottom_removed.saturating_sub(safety_margin);
    let left = diagnostics.left_removed.saturating_sub(safety_margin);
    let right = diagnostics.right_removed.saturating_sub(safety_margin);
    let boundary =
        if top + bottom >= height || left + right >= width || top + bottom + left + right == 0 {
            base_detect::detect_film_base(img)
        } else {
            let measurement = img
                .slice(s![top..height - bottom, left..width - right, ..])
                .to_owned();
            base_detect::detect_film_base(&measurement)
        };

    if let Some(removed_band) = removed_band {
        let removed_confidence = base_detection_confidence(&removed_band);
        let boundary_confidence = base_detection_confidence(&boundary);
        if boundary.base_color_source == "high_transmittance_fallback"
            || removed_confidence > boundary_confidence + 0.05
        {
            return removed_band;
        }
    }
    boundary
}

fn parse_base_color_override(
    cli: &Cli,
    working_bit_depth: u8,
) -> Result<Option<[f64; 3]>, Box<dyn std::error::Error>> {
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

    let max_value = if working_bit_depth >= 16 {
        u16::MAX as f64
    } else {
        ((1u32 << working_bit_depth as u32) - 1) as f64
    };
    if parsed.iter().any(|value| *value > max_value) {
        return Err(format!(
            "--base-color values must fit the configured {}-bit working range (max {:.0})",
            working_bit_depth, max_value
        )
        .into());
    }

    Ok(Some(parsed))
}

fn apply_downstream_base_confidence_limit(phase: &mut PhaseReport, base_confidence: f64) -> f64 {
    let factor = downstream_base_quality_factor(base_confidence);
    let confidence_before_base_limit = phase.confidence;
    let confidence_limited_by_base_estimate = confidence_before_base_limit > factor + 1e-12;
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
            serde_json::json!(confidence_limited_by_base_estimate),
        );
        metrics.insert(
            "confidence_before_base_limit".to_string(),
            serde_json::json!(confidence_before_base_limit),
        );
    }
    factor
}

fn color_decision_confidence(
    diagnostics: &colorspace::ColorspaceDiagnostics,
) -> (f64, &'static str) {
    let neutral_evidence = if diagnostics.neutral_estimate_quality.score.is_finite() {
        diagnostics.neutral_estimate_quality.score.clamp(0.0, 1.0)
    } else {
        0.0
    };
    match colorspace::tone_color_trust_state(diagnostics) {
        "trusted" => (1.0, "trusted_selected_mapping"),
        "limited_weak_neutral" => (neutral_evidence, "limited_selected_mapping"),
        _ => (0.0, "review_required_selected_mapping"),
    }
}

fn negative_response_model_phase_evidence(
    diagnostics: &density::NegativeResponseModelDiagnostics,
) -> (f64, bool, &'static str, &'static str) {
    if diagnostics.model == "measured_nonlinear_dye_separation"
        && diagnostics.source == "measured_roll_target_with_held_out_validation"
        && diagnostics.accepted
        && !diagnostics.review_required
    {
        if let Some(confidence) = diagnostics
            .measured_confidence
            .filter(|confidence| confidence.is_finite())
        {
            return (
                confidence.clamp(0.0, 1.0),
                true,
                "held_out_measured_response_supported",
                "held_out_validated_measured_negative_response",
            );
        }
        return (
            0.0,
            false,
            "review_required_incomplete_measured_response_evidence",
            "measured_negative_response_missing_confidence",
        );
    }

    if diagnostics.model == "regularized_frame_luminance_axis" {
        (
            0.0,
            false,
            "review_required_unmeasured_frame_response",
            "frame_response_dye_crosstalk_and_nonlinearity_unmeasured",
        )
    } else {
        (
            0.0,
            false,
            "review_required_unmeasured_unit_slope_response",
            "unit_slope_response_dye_crosstalk_and_nonlinearity_unmeasured",
        )
    }
}

fn negative_reconstruction_color_evidence(
    positive_input_mode: bool,
    render_input_source: &str,
    review_required: bool,
    response_model: Option<&density::NegativeResponseModel>,
    density_phase_confidence: Option<f64>,
) -> (Option<f64>, bool, &'static str, &'static str) {
    if positive_input_mode {
        return (
            None,
            false,
            "not_applicable_positive_input",
            "not_evaluated_positive_input",
        );
    }

    if render_input_source != "direct_density_transmittance" {
        return (
            Some(0.0),
            false,
            "review_required_blind_ica_separation",
            "blind_ica_has_no_independent_physical_dye_separation_evidence",
        );
    }

    let Some(response_model) = response_model else {
        return (
            Some(0.0),
            false,
            "review_required_missing_negative_response_evidence",
            "selected_direct_density_route_missing_response_model_evidence",
        );
    };
    let measured_response = response_model.diagnostics.model == "measured_nonlinear_dye_separation"
        && response_model.diagnostics.source == "measured_roll_target_with_held_out_validation"
        && response_model.diagnostics.accepted
        && !response_model.diagnostics.review_required;

    if review_required {
        if measured_response {
            return (
                Some(0.0),
                true,
                "review_required_measured_response_outside_runtime_curve_support",
                "measured_response_runtime_curve_coverage_exceeded_trusted_limit",
            );
        }
        return (
            Some(0.0),
            false,
            "review_required_unmeasured_negative_response",
            "selected_negative_response_lacks_held_out_physical_validation",
        );
    }

    if !measured_response {
        return (
            Some(0.0),
            false,
            "review_required_unmeasured_negative_response",
            "selected_negative_response_lacks_held_out_physical_validation",
        );
    }

    let confidence = density_phase_confidence
        .filter(|confidence| confidence.is_finite())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    if confidence <= f64::EPSILON {
        return (
            Some(0.0),
            true,
            "review_required_insufficient_film_base_evidence",
            "measured_response_and_runtime_coverage_supported_but_film_base_evidence_is_zero",
        );
    }
    (
        Some(confidence),
        true,
        "held_out_measured_response_and_runtime_coverage_supported",
        "density_phase_film_base_and_measured_response_with_runtime_coverage",
    )
}

fn technical_white_balance_phase_evidence(
    status: &str,
    measured_confidence: f64,
) -> (f64, bool, &'static str, &'static str) {
    match status {
        "disabled" => (
            0.0,
            false,
            "not_applicable_disabled",
            "not_applied_disabled",
        ),
        "applied_manual" => (
            1.0,
            false,
            "user_authoritative_override",
            "user_authoritative_override",
        ),
        _ => {
            let confidence = if measured_confidence.is_finite() {
                measured_confidence.clamp(0.0, 1.0)
            } else {
                0.0
            };
            (
                confidence,
                true,
                "measured_scene_neutral_evidence",
                "automatic_evidence_evaluated",
            )
        }
    }
}

#[derive(Debug, Clone)]
struct ToneOutputEvidence {
    confidence: f64,
    evaluated: bool,
    status: &'static str,
    reason: String,
    input_luminance_range_p05_p95: f64,
    mapped_luminance_range_p05_p95: f64,
    render_luminance_range_p05_p95: f64,
    render_to_mapped_range_ratio: Option<f64>,
    maximum_post_tone_high_clip_ratio: f64,
    maximum_post_tone_low_clip_ratio: f64,
}

fn assess_tone_output_evidence(
    input_linear_percentiles: [f64; 3],
    mapped_linear_percentiles: [f64; 3],
    render_luminance_percentiles: [f64; 3],
    post_tone_high_clip_ratio: [f64; 3],
    post_tone_low_clip_ratio: [f64; 3],
    sample_count: usize,
) -> ToneOutputEvidence {
    let input_range = input_linear_percentiles[2] - input_linear_percentiles[0];
    let mapped_range = mapped_linear_percentiles[2] - mapped_linear_percentiles[0];
    let render_range = render_luminance_percentiles[2] - render_luminance_percentiles[0];
    let maximum_high_clip = post_tone_high_clip_ratio
        .into_iter()
        .fold(0.0_f64, f64::max);
    let maximum_low_clip = post_tone_low_clip_ratio.into_iter().fold(0.0_f64, f64::max);
    let all_finite = input_linear_percentiles
        .into_iter()
        .chain(mapped_linear_percentiles)
        .chain(render_luminance_percentiles)
        .chain(post_tone_high_clip_ratio)
        .chain(post_tone_low_clip_ratio)
        .all(f64::is_finite);
    let range_ratio = if mapped_range > f64::EPSILON {
        Some(render_range / mapped_range)
    } else {
        None
    };

    let evidence = |confidence: f64, evaluated: bool, status: &'static str, reason: String| {
        ToneOutputEvidence {
            confidence,
            evaluated,
            status,
            reason,
            input_luminance_range_p05_p95: input_range,
            mapped_luminance_range_p05_p95: mapped_range,
            render_luminance_range_p05_p95: render_range,
            render_to_mapped_range_ratio: range_ratio,
            maximum_post_tone_high_clip_ratio: maximum_high_clip,
            maximum_post_tone_low_clip_ratio: maximum_low_clip,
        }
    };

    if sample_count == 0
        || !all_finite
        || input_range < 0.0
        || mapped_range < 0.0
        || render_range < 0.0
    {
        return evidence(
            0.0,
            false,
            "review_required_invalid_render_tone_diagnostics",
            "rendered tone diagnostics were empty, non-finite, or not monotonically ordered"
                .to_string(),
        );
    }
    if maximum_high_clip >= TONE_OUTPUT_MAX_CATASTROPHIC_CHANNEL_CLIP_RATIO
        || maximum_low_clip >= TONE_OUTPUT_MAX_CATASTROPHIC_CHANNEL_CLIP_RATIO
    {
        return evidence(
            0.0,
            true,
            "review_required_catastrophic_post_tone_clipping",
            format!(
                "post-tone channel clipping reached {:.2}% high and {:.2}% low; the catastrophic review threshold is {:.2}%",
                maximum_high_clip * 100.0,
                maximum_low_clip * 100.0,
                TONE_OUTPUT_MAX_CATASTROPHIC_CHANNEL_CLIP_RATIO * 100.0
            ),
        );
    }
    if input_range < TONE_OUTPUT_MIN_EVALUABLE_RANGE
        && mapped_range < TONE_OUTPUT_MIN_EVALUABLE_RANGE
    {
        return evidence(
            0.0,
            true,
            "review_required_insufficient_scene_tonal_range",
            format!(
                "input and fitted p05-p95 luminance spans ({input_range:.6} and {mapped_range:.6}) are both below the {:.6} minimum needed to substantiate a rendered dynamic-range claim",
                TONE_OUTPUT_MIN_EVALUABLE_RANGE
            ),
        );
    }
    if mapped_range < TONE_OUTPUT_MIN_EVALUABLE_RANGE {
        return evidence(
            0.0,
            true,
            "review_required_collapsed_fitted_tone_range",
            format!(
                "the fitted tone curve collapsed a measurable input p05-p95 span of {input_range:.6} to {mapped_range:.6}"
            ),
        );
    }
    if render_range < TONE_OUTPUT_MIN_RENDER_RANGE
        || range_ratio.is_some_and(|ratio| ratio < TONE_OUTPUT_MIN_RANGE_RETENTION_RATIO)
    {
        return evidence(
            0.0,
            true,
            "review_required_collapsed_render_luminance_range",
            format!(
                "rendered p05-p95 luminance span {render_range:.6} retained {:.2}% of the fitted span {mapped_range:.6}; minimums are {:.6} absolute and {:.2}% relative",
                range_ratio.unwrap_or(0.0) * 100.0,
                TONE_OUTPUT_MIN_RENDER_RANGE,
                TONE_OUTPUT_MIN_RANGE_RETENTION_RATIO * 100.0
            ),
        );
    }

    evidence(
        1.0,
        true,
        "supported_render_tonal_distribution",
        format!(
            "rendered p05-p95 luminance span {render_range:.6} retained {:.2}% of the fitted span with maximum high/low channel clipping {:.2}%/{:.2}%",
            range_ratio.unwrap_or(0.0) * 100.0,
            maximum_high_clip * 100.0,
            maximum_low_clip * 100.0
        ),
    )
}

#[derive(Clone, Copy)]
struct ReviewRequirement<'a> {
    required: bool,
    reason: &'a str,
}

struct RenderReviewInputs<'a> {
    base_confidence: f64,
    color_trust_state: &'a str,
    geometry: ReviewRequirement<'a>,
    input_mode: ReviewRequirement<'a>,
    negative_response: ReviewRequirement<'a>,
    technical_white_balance: ReviewRequirement<'a>,
    tone_output: ReviewRequirement<'a>,
    grain_detail: ReviewRequirement<'a>,
}

fn render_review_status(review: &RenderReviewInputs<'_>) -> &'static str {
    if review.geometry.required {
        "blocked_geometry_review"
    } else if review.input_mode.required {
        "review_required_input_mode"
    } else if review.base_confidence < BASE_CONFIDENCE_FALLBACK {
        "blocked_low_base_confidence"
    } else if review.base_confidence < BASE_CONFIDENCE_WARN {
        "caution_low_base_confidence"
    } else if review.negative_response.required {
        "review_required_negative_response"
    } else if review.color_trust_state == "review_required" {
        "review_required_color"
    } else if review.technical_white_balance.required {
        "review_required_white_balance"
    } else if review.tone_output.required {
        "review_required_tone_output"
    } else if review.grain_detail.required {
        "review_required_grain_detail"
    } else if review.color_trust_state != "trusted" {
        "caution_limited_color"
    } else {
        "reviewable"
    }
}

fn render_review_reason(review: &RenderReviewInputs<'_>) -> String {
    if review.geometry.required {
        review.geometry.reason.to_string()
    } else if review.input_mode.required {
        review.input_mode.reason.to_string()
    } else if review.base_confidence < BASE_CONFIDENCE_FALLBACK {
        "working-image base estimate is fallback quality; density inversion may not produce a trustworthy positive".to_string()
    } else if review.base_confidence < BASE_CONFIDENCE_WARN {
        "working-image base estimate is weak; compare debug density and color artifacts before judging the render".to_string()
    } else if review.negative_response.required {
        review.negative_response.reason.to_string()
    } else if review.color_trust_state == "review_required" {
        "selected color reconstruction lacks sufficient evidence for trust; inspect color diagnostics or provide a measured profile".to_string()
    } else if review.technical_white_balance.required {
        review.technical_white_balance.reason.to_string()
    } else if review.tone_output.required {
        review.tone_output.reason.to_string()
    } else if review.grain_detail.required {
        review.grain_detail.reason.to_string()
    } else if review.color_trust_state != "trusted" {
        "selected color reconstruction has limited evidence and needs cautious color review"
            .to_string()
    } else {
        "input-mode, base, negative-response, color, rendered-tone, and optional grain-detail evidence are strong enough for normal visual review".to_string()
    }
}

fn stitch_quality_review_reasons(report: &PhaseReport) -> Vec<String> {
    let mut reasons = Vec::<String>::new();
    if report
        .metrics
        .get("seam_blend")
        .and_then(|blend| blend.get("review_required"))
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        reasons.push(
            report
                .metrics
                .get("seam_blend")
                .and_then(|blend| blend.get("review_reason"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("seam detail or gradient evidence requires review")
                .to_string(),
        );
    }
    if let Some(sequence_reasons) = report
        .metrics
        .get("seam_quality_review_reasons")
        .and_then(serde_json::Value::as_array)
    {
        reasons.extend(
            sequence_reasons
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_string),
        );
    }
    if let Some(pair_merges) = report
        .metrics
        .get("pair_merges")
        .and_then(serde_json::Value::as_array)
    {
        for merge in pair_merges {
            let Some(blend) = merge
                .get("pair_report")
                .and_then(|pair_report| pair_report.get("metrics"))
                .and_then(|metrics| metrics.get("seam_blend"))
            else {
                continue;
            };
            if blend
                .get("review_required")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
            {
                continue;
            }
            let step = merge
                .get("step")
                .and_then(serde_json::Value::as_u64)
                .map(|step| step.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let reason = blend
                .get("review_reason")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("seam detail or gradient evidence requires review");
            reasons.push(format!("sequence stitch step {step}: {reason}"));
        }
    }
    reasons.sort();
    reasons.dedup();
    reasons
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

#[derive(Debug, Clone)]
struct LoadFidelityEvidence {
    status: &'static str,
    confidence: f64,
    precision_confidence: f64,
    channel_layout_confidence: f64,
    orientation_confidence: f64,
    dng_level_confidence: f64,
    limiters: Vec<&'static str>,
}

fn load_fidelity_evidence(diag: &tiff_io::TiffLoadDiagnostics) -> LoadFidelityEvidence {
    let source_bits = diag.source_bits_per_sample.max(1) as f64;
    let working_bits = diag.working_range.target_bit_depth.max(1) as f64;
    let precision_confidence =
        (source_bits.min(working_bits) / source_bits.max(working_bits)).clamp(0.0, 1.0);
    let channel_layout_confidence = if diag.source_channel_count >= 3 {
        1.0
    } else {
        0.0
    };
    let orientation_confidence =
        if diag.orientation.tag_value.unwrap_or(1) != 1 && !diag.orientation.applied {
            0.0
        } else {
            1.0
        };
    let dng_level_confidence = if diag
        .dng_level_normalization
        .as_ref()
        .is_some_and(|levels| levels.status == "invalid_metadata")
    {
        0.5
    } else {
        1.0
    };
    let confidence = precision_confidence
        .min(channel_layout_confidence)
        .min(orientation_confidence)
        .min(dng_level_confidence);
    let mut limiters = Vec::new();
    if diag.source_bits_per_sample < diag.working_range.target_bit_depth {
        limiters.push("source_precision_upscaled_without_added_information");
    } else if diag.source_bits_per_sample > diag.working_range.target_bit_depth {
        limiters.push("source_precision_downscaled_with_information_loss");
    }
    if diag.source_channel_count < 3 {
        limiters.push("non_rgb_source_channels_expanded_to_rgb");
    }
    if orientation_confidence == 0.0 {
        limiters.push("required_orientation_not_materialized");
    }
    if dng_level_confidence < 1.0 {
        limiters.push("invalid_dng_level_metadata");
    }
    let status = if confidence >= 0.999_999 {
        "full_declared_decode_fidelity"
    } else if confidence > 0.0 {
        "limited_declared_decode_fidelity"
    } else {
        "insufficient_declared_decode_fidelity"
    };
    LoadFidelityEvidence {
        status,
        confidence,
        precision_confidence,
        channel_layout_confidence,
        orientation_confidence,
        dng_level_confidence,
        limiters,
    }
}

fn tiff_load_metrics_json(diag: &tiff_io::TiffLoadDiagnostics) -> serde_json::Value {
    let fidelity = load_fidelity_evidence(diag);
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
        "decoded_pixel_sha256": diag.decoded_pixel_sha256,
        "source_orientation_materialized_decoded_pixel_sha256": diag.orientation_correction.source_orientation_materialized_decoded_pixel_sha256,
        "decode_fidelity": {
            "status": fidelity.status,
            "evidence_evaluated": true,
            "confidence": fidelity.confidence,
            "confidence_basis": "minimum_declared_precision_channel_layout_orientation_and_dng_level_evidence",
            "confidence_definition": "declared decode-fidelity evidence strength, not file-I/O success or an empirical image-quality probability",
            "precision_confidence": fidelity.precision_confidence,
            "channel_layout_confidence": fidelity.channel_layout_confidence,
            "orientation_confidence": fidelity.orientation_confidence,
            "dng_level_confidence": fidelity.dng_level_confidence,
            "limiters": fidelity.limiters,
        },
        "orientation": {
            "tag_value": diag.orientation.tag_value,
            "transform": diag.orientation.transform,
            "applied": diag.orientation.applied,
            "source_width": diag.orientation.source_width,
            "source_height": diag.orientation.source_height,
            "output_width": diag.orientation.output_width,
            "output_height": diag.orientation.output_height,
            "reason": diag.orientation.reason,
        },
        "orientation_correction": {
            "requested": diag.orientation_correction.requested,
            "transform": diag.orientation_correction.transform,
            "applied": diag.orientation_correction.applied,
            "input_width": diag.orientation_correction.input_width,
            "input_height": diag.orientation_correction.input_height,
            "output_width": diag.orientation_correction.output_width,
            "output_height": diag.orientation_correction.output_height,
            "effective_tag_value": diag.orientation_correction.effective_tag_value,
            "effective_transform": diag.orientation_correction.effective_transform,
            "source_orientation_materialized_decoded_pixel_sha256": diag.orientation_correction.source_orientation_materialized_decoded_pixel_sha256,
            "corrected_decoded_pixel_sha256": diag.orientation_correction.corrected_decoded_pixel_sha256,
            "reason": diag.orientation_correction.reason,
        },
        "effective_orientation_tag": diag.effective_orientation_tag,
        "source_icc_profile": diag.source_icc_profile,
        "dng_metadata": diag.dng_metadata.as_ref().map(dng_metadata_metrics_json),
        "dng_level_normalization": diag.dng_level_normalization.as_ref().map(|levels| serde_json::json!({
            "status": levels.status,
            "black_level": levels.black_level,
            "white_level": levels.white_level,
            "source_code_max": levels.source_code_max,
            "clipped_below_black": levels.clipped_below_black,
            "clipped_above_white": levels.clipped_above_white,
            "reason": levels.reason,
        })),
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
    if diag.source_channel_count < 3 {
        warnings.push(format!(
            "{} decoded from {} source channel(s); independent RGB colour evidence is unavailable",
            label, diag.source_channel_count
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
    if diag.orientation.tag_value.is_some()
        && diag.orientation.tag_value != Some(1)
        && !diag.orientation.applied
    {
        warnings.push(format!("{}: {}", label, diag.orientation.reason));
    }
    if diag.orientation_correction.applied {
        warnings.push(format!(
            "{}: explicit metadata-relative orientation correction `{}` was applied; final decoded-pixel SHA-256 is {}",
            label,
            diag.orientation_correction.requested,
            diag.orientation_correction.corrected_decoded_pixel_sha256
        ));
    }
    if let Some(profile) = &diag.source_icc_profile {
        if profile.status != "valid_rgb_profile" {
            warnings.push(format!(
                "{} embedded ICC profile is not usable: {}",
                label, profile.reason
            ));
        }
    }
    if let Some(levels) = &diag.dng_level_normalization {
        if levels.status == "invalid_metadata" {
            warnings.push(format!("{}: {}", label, levels.reason));
        }
    }
    warnings
}

#[derive(Debug, Clone, Copy)]
struct BorderCropEvidence {
    status: &'static str,
    evidence_evaluated: bool,
    applied_crop_evidence_supported: bool,
    applied_edge_count: usize,
    confidence: f64,
    confidence_basis: &'static str,
    review_required: bool,
    review_reason: Option<&'static str>,
}

fn border_crop_evidence(diagnostics: &border::BorderRemovalDiagnostics) -> BorderCropEvidence {
    if !diagnostics.evidence_evaluated {
        return BorderCropEvidence {
            status: "not_evaluated_image_too_small",
            evidence_evaluated: false,
            applied_crop_evidence_supported: false,
            applied_edge_count: 0,
            confidence: 0.0,
            confidence_basis: "not_evaluated",
            review_required: true,
            review_reason: Some("image was too small for four-edge border evidence evaluation"),
        };
    }

    let applied_edges = [
        (diagnostics.top_removed, diagnostics.top_confidence),
        (diagnostics.bottom_removed, diagnostics.bottom_confidence),
        (diagnostics.left_removed, diagnostics.left_confidence),
        (diagnostics.right_removed, diagnostics.right_confidence),
    ];
    let mut applied_edge_count = 0usize;
    let mut confidence = 1.0f64;
    for (removed, edge_confidence) in applied_edges {
        if removed == 0 {
            continue;
        }
        applied_edge_count += 1;
        confidence = confidence.min(edge_confidence);
    }

    if diagnostics.vertical_crop_rejected || diagnostics.horizontal_crop_rejected {
        let (rejected_status, partial_status, reason) = match (
            diagnostics.vertical_crop_rejected,
            diagnostics.horizontal_crop_rejected,
        ) {
            (true, true) => (
                "vertical_and_horizontal_crop_rejected_full_image_guard",
                "partial_crop_with_vertical_and_horizontal_rejection",
                "both axis crop proposals were rejected because they would consume the full image",
            ),
            (true, false) => (
                "vertical_crop_rejected_full_image_guard",
                "partial_crop_with_vertical_rejection",
                "the vertical crop proposal was rejected because it would consume the full image",
            ),
            (false, true) => (
                "horizontal_crop_rejected_full_image_guard",
                "partial_crop_with_horizontal_rejection",
                "the horizontal crop proposal was rejected because it would consume the full image",
            ),
            (false, false) => unreachable!("crop rejection branch requires a rejected axis"),
        };
        return BorderCropEvidence {
            status: if applied_edge_count > 0 {
                partial_status
            } else {
                rejected_status
            },
            evidence_evaluated: true,
            applied_crop_evidence_supported: false,
            applied_edge_count,
            confidence: 0.0,
            confidence_basis: "unsafe_axis_proposal_rejected",
            review_required: true,
            review_reason: Some(reason),
        };
    }

    let unresolved_edge_candidate = diagnostics.top_unresolved_strong_candidate
        || diagnostics.bottom_unresolved_strong_candidate
        || diagnostics.left_unresolved_strong_candidate
        || diagnostics.right_unresolved_strong_candidate;
    if unresolved_edge_candidate {
        return BorderCropEvidence {
            status: if applied_edge_count > 0 {
                "partial_crop_with_unresolved_strong_edge_candidate"
            } else {
                "no_crop_with_unresolved_strong_edge_candidate"
            },
            evidence_evaluated: true,
            applied_crop_evidence_supported: false,
            applied_edge_count,
            confidence: 0.0,
            confidence_basis: "unresolved_strong_edge_candidate",
            review_required: true,
            review_reason: Some(
                "one or more edges retained a strong dead-zone candidate because boundary-transition evidence was insufficient for automatic cropping",
            ),
        };
    }

    if applied_edge_count == 0 {
        return BorderCropEvidence {
            status: "no_crop_no_convincing_dead_zone",
            evidence_evaluated: true,
            applied_crop_evidence_supported: false,
            applied_edge_count: 0,
            confidence: 0.0,
            confidence_basis: "no_applied_crop_evidence",
            review_required: false,
            review_reason: None,
        };
    }

    if !confidence.is_finite() || confidence <= 0.0 {
        return BorderCropEvidence {
            status: "crop_applied_without_supported_edge_evidence",
            evidence_evaluated: true,
            applied_crop_evidence_supported: false,
            applied_edge_count,
            confidence: 0.0,
            confidence_basis: "invalid_or_zero_applied_edge_evidence",
            review_required: true,
            review_reason: Some("an applied edge crop lacks finite positive evidence strength"),
        };
    }

    BorderCropEvidence {
        status: "crop_applied_from_measured_edge_evidence",
        evidence_evaluated: true,
        applied_crop_evidence_supported: true,
        applied_edge_count,
        confidence: confidence.clamp(0.0, 1.0),
        confidence_basis: "minimum_applied_edge_detection_confidence",
        review_required: false,
        review_reason: None,
    }
}

fn border_metrics_json(diagnostics: &border::BorderRemovalDiagnostics) -> serde_json::Value {
    let evidence = border_crop_evidence(diagnostics);
    let groups = [
        serde_json::json!({
            "border_decision_status": evidence.status,
            "border_evidence_evaluated": evidence.evidence_evaluated,
            "applied_crop_evidence_supported": evidence.applied_crop_evidence_supported,
            "applied_edge_count": evidence.applied_edge_count,
            "confidence": evidence.confidence,
            "confidence_basis": evidence.confidence_basis,
            "confidence_definition": "minimum of normalized strong-line support, luminance separation, and boundary-transition evidence across applied edges; evidence strength, not an empirical correctness probability",
            "review_required": evidence.review_required,
            "review_reason": evidence.review_reason,
            "top_overwhelming_support_fallback_used": diagnostics.top_overwhelming_support_fallback_used,
            "bottom_overwhelming_support_fallback_used": diagnostics.bottom_overwhelming_support_fallback_used,
            "left_overwhelming_support_fallback_used": diagnostics.left_overwhelming_support_fallback_used,
            "right_overwhelming_support_fallback_used": diagnostics.right_overwhelming_support_fallback_used,
            "top_unresolved_strong_candidate": diagnostics.top_unresolved_strong_candidate,
            "bottom_unresolved_strong_candidate": diagnostics.bottom_unresolved_strong_candidate,
            "left_unresolved_strong_candidate": diagnostics.left_unresolved_strong_candidate,
            "right_unresolved_strong_candidate": diagnostics.right_unresolved_strong_candidate,
            "vertical_crop_rejected": diagnostics.vertical_crop_rejected,
            "horizontal_crop_rejected": diagnostics.horizontal_crop_rejected,
            "dead_zone_detected": diagnostics.dead_zone_detected,
        }),
        serde_json::json!({
            "top_removed": diagnostics.top_removed,
            "bottom_removed": diagnostics.bottom_removed,
            "left_removed": diagnostics.left_removed,
            "right_removed": diagnostics.right_removed,
            "content_lum": diagnostics.content_lum,
            "content_mad": diagnostics.content_mad,
            "column_content_lum": diagnostics.column_content_lum,
            "column_content_mad": diagnostics.column_content_mad,
            "row_strong_lum_gap_threshold": diagnostics.row_strong_lum_gap_threshold,
            "row_boundary_derivative_threshold": diagnostics.row_boundary_derivative_threshold,
            "column_strong_lum_gap_threshold": diagnostics.column_strong_lum_gap_threshold,
            "column_boundary_derivative_threshold": diagnostics.column_boundary_derivative_threshold,
        }),
        serde_json::json!({
            "top_strong_rows": diagnostics.top_strong_rows,
            "bottom_strong_rows": diagnostics.bottom_strong_rows,
            "top_peak_lum_gap": diagnostics.top_peak_lum_gap,
            "bottom_peak_lum_gap": diagnostics.bottom_peak_lum_gap,
            "top_peak_row_delta": diagnostics.top_peak_row_delta,
            "bottom_peak_row_delta": diagnostics.bottom_peak_row_delta,
            "top_boundary_transition_delta": diagnostics.top_boundary_transition_delta,
            "bottom_boundary_transition_delta": diagnostics.bottom_boundary_transition_delta,
            "top_confidence": diagnostics.top_confidence,
            "bottom_confidence": diagnostics.bottom_confidence,
        }),
        serde_json::json!({
            "left_strong_columns": diagnostics.left_strong_columns,
            "right_strong_columns": diagnostics.right_strong_columns,
            "left_peak_lum_gap": diagnostics.left_peak_lum_gap,
            "right_peak_lum_gap": diagnostics.right_peak_lum_gap,
            "left_peak_column_delta": diagnostics.left_peak_column_delta,
            "right_peak_column_delta": diagnostics.right_peak_column_delta,
            "left_boundary_transition_delta": diagnostics.left_boundary_transition_delta,
            "right_boundary_transition_delta": diagnostics.right_boundary_transition_delta,
            "left_confidence": diagnostics.left_confidence,
            "right_confidence": diagnostics.right_confidence,
        }),
    ];
    let mut metrics = serde_json::Map::new();
    for group in groups {
        let serde_json::Value::Object(group) = group else {
            unreachable!("border metric group must be a JSON object");
        };
        metrics.extend(group);
    }
    serde_json::Value::Object(metrics)
}

fn component_classification_metrics_json(component: &PreparedComponent) -> serde_json::Value {
    let analysis = &component.analysis;
    let measurement = &component.measurement;
    let detection = &component.detection;
    let classification_evidence_evaluated =
        component.measurement_stage != "not_required_positive_input";
    serde_json::json!({
        "path": component.path.to_string_lossy(),
        "classification_status": if classification_evidence_evaluated {
            "evaluated_negative_film_rebate_topology"
        } else {
            "not_applicable_positive_input"
        },
        "classification_evidence_evaluated": classification_evidence_evaluated,
        "confidence_basis": if classification_evidence_evaluated {
            "measured_edge_and_internal_rebate_evidence"
        } else {
            "not_evaluated"
        },
        "class": format!("{:?}", analysis.class),
        "confidence": analysis.confidence,
        "base_color": measurement.base_color,
        "base_color_source": measurement.base_color_source,
        "base_color_reason": measurement.base_color_reason,
        "base_measurement_stage": component.measurement_stage,
        "base_measurement_selection_reason": component.measurement_reason,
        "base_color_proxy_confidence": measurement.base_color_proxy_confidence,
        "base_color_support_fraction": measurement.base_color_support_fraction,
        "base_strip_width": measurement.strip_width,
        "base_left_confidence": measurement.left_confidence,
        "base_right_confidence": measurement.right_confidence,
        "base_top_confidence": measurement.top_confidence,
        "base_bottom_confidence": measurement.bottom_confidence,
        "post_crop_detection": {
            "base_color": detection.base_color,
            "base_color_source": detection.base_color_source,
            "confidence": base_detection_confidence(detection),
        },
        "left_edge_confidence": analysis.left_edge_confidence,
        "right_edge_confidence": analysis.right_edge_confidence,
        "content_span": [analysis.content_span.0, analysis.content_span.1],
        "content_fraction": analysis.content_fraction,
        "activity_score": analysis.activity_score,
        "component_score": analysis.component_score,
        "internal_base_regions": analysis.internal_base_regions.iter().map(|region| serde_json::json!({
            "x_start": region.x_start,
            "x_end": region.x_end,
            "confidence": region.confidence,
            "mad_lum": region.mad_lum,
        })).collect::<Vec<_>>(),
    })
}

fn best_single_component_index(components: &[PreparedComponent]) -> usize {
    components
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.analysis
                .component_score
                .partial_cmp(&right.analysis.component_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
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
        "orientation": metadata.orientation,
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
    load_diagnostics: &[&tiff_io::TiffLoadDiagnostics],
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
    for metadata in load_diagnostics
        .iter()
        .filter_map(|diagnostics| diagnostics.dng_metadata.as_ref())
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

fn calibration_color_mapping_application_diagnostics(
    diagnostics: &colorspace::ColorspaceDiagnostics,
) -> color_calibration::CalibrationColorMappingApplicationDiagnostics {
    let acceptance = &diagnostics.calibration_acceptance;
    let evaluated = acceptance.preferred_candidate.is_some();
    let applied = matches!(acceptance.status.as_str(), "accepted" | "forced");
    color_calibration::CalibrationColorMappingApplicationDiagnostics {
        evaluated,
        applied,
        selection_status: acceptance.status.clone(),
        selected_candidate: diagnostics.selected_candidate.clone(),
        preferred_candidate: acceptance.preferred_candidate.clone(),
        reason: acceptance.reason.clone(),
        definition: "whether a calibration or scanner-prior color mapping candidate was selected for the final colorspace output; scanner linearization, film-base, and negative-response calibration are separate application stages".to_string(),
    }
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
        "neutral_safety_rescue": diagnostics.neutral_safety_rescue.clone(),
        "nonlinear_color_model": diagnostics.nonlinear_color_model.clone(),
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
        "neutral_safety_rescue": diagnostics.neutral_safety_rescue.clone(),
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
        "nonlinear_model_support": diagnostics.nonlinear_color_model.clone(),
        "neutral_safety_rescue": diagnostics.neutral_safety_rescue.clone(),
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
    metrics.insert(
        "neutral_safety_rescue".to_string(),
        serde_json::json!(diagnostics.neutral_safety_rescue.clone()),
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
    if let Some(nonlinear_color_model) = &diagnostics.nonlinear_color_model {
        metrics.insert(
            "nonlinear_color_model".to_string(),
            serde_json::json!(nonlinear_color_model),
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

fn grain_detail_retention_metrics_json(
    diagnostics: &tonemap::GrainDetailRetentionDiagnostics,
) -> serde_json::Value {
    serde_json::json!({
        "method": diagnostics.method,
        "evaluated": diagnostics.evaluated,
        "decision_supported": diagnostics.decision_supported,
        "sample_stride": diagnostics.sample_stride,
        "probe_radius": diagnostics.probe_radius,
        "minimum_probe_count": diagnostics.minimum_probe_count,
        "luminance_probe_count": diagnostics.luminance_probe_count,
        "chroma_probe_count": diagnostics.chroma_probe_count,
        "luminance_decision_supported": diagnostics.luminance_decision_supported,
        "chroma_decision_supported": diagnostics.chroma_decision_supported,
        "luminance_median_retention": diagnostics.luminance_median_retention,
        "luminance_p10_retention": diagnostics.luminance_p10_retention,
        "chroma_median_retention": diagnostics.chroma_median_retention,
        "chroma_p10_retention": diagnostics.chroma_p10_retention,
        "luminance_contrast_threshold": diagnostics.luminance_contrast_threshold,
        "chroma_contrast_threshold": diagnostics.chroma_contrast_threshold,
        "coherence_threshold": diagnostics.coherence_threshold,
        "median_retention_threshold": diagnostics.median_retention_threshold,
        "p10_retention_threshold": diagnostics.p10_retention_threshold,
        "review_required": diagnostics.review_required,
        "reason": diagnostics.reason,
        "review_reason": diagnostics.review_reason,
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
    hash_file_sha256(path).ok().map(|(sha256, _)| sha256)
}

fn saved_artifact_sha256(path: &Path, label: &str) -> Result<String, String> {
    let (sha256, size_bytes) = hash_file_sha256(path)
        .map_err(|error| format!("failed to hash saved {label} {}: {error}", path.display()))?;
    if size_bytes == 0 {
        return Err(format!(
            "saved {label} {} is empty and cannot be bound for delivery",
            path.display()
        ));
    }
    Ok(sha256)
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
        let is_tiff = matches!(extension.as_deref(), Some("tif" | "tiff"));
        let is_review_proof = extension.as_deref() == Some("png")
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case("review_srgb.png"));
        if !is_tiff && !is_review_proof {
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
    let positive_input_mode = cli.input_mode == InputMode::Positive;
    let mut report = PipelineReport::new_with_metadata(RunMetadata::capture_current(
        &cli.output_dir,
        &output_path,
        run_started_at,
    ));

    let load_start = Instant::now();
    let mut source_inspections = Vec::with_capacity(cli.inputs.len());
    for (index, path) in cli.inputs.iter().enumerate() {
        match tiff_io::inspect_scan_source(path) {
            Ok(inspection) => source_inspections.push(inspection),
            Err(err) => {
                record_failure(
                    &mut report,
                    &cli.output_dir,
                    "load",
                    format!(
                        "failed to inspect input component {} ({}): {err}",
                        index + 1,
                        path.display()
                    ),
                    load_start,
                    options.write_artifacts,
                )?;
                return Err(err);
            }
        }
    }
    let mut working_bit_depth = source_inspections
        .iter()
        .fold(cli.bit_depth, |depth, inspection| {
            depth.max(inspection.source_bits_per_sample)
        })
        .min(16);
    let mut base_color_override = if cli.input_mode == InputMode::Positive {
        None
    } else {
        parse_base_color_override(cli, working_bit_depth)?
    };
    let mut loaded_components = Vec::with_capacity(cli.inputs.len());
    for (index, path) in cli.inputs.iter().enumerate() {
        log::info!("Loading input component {}: {}", index + 1, path.display());
        let mut loaded = match tiff_io::load_tiff_u16(path, working_bit_depth) {
            Ok(image) => image,
            Err(err) => {
                record_failure(
                    &mut report,
                    &cli.output_dir,
                    "load",
                    format!(
                        "failed to load input component {} ({}): {}",
                        index + 1,
                        path.display(),
                        err
                    ),
                    load_start,
                    options.write_artifacts,
                )?;
                return Err(err);
            }
        };
        if let Err(error) = tiff_io::apply_orientation_correction_u16(
            &mut loaded,
            cli.geometry.orientation_correction.exif_tag(),
            cli.geometry.orientation_correction.as_str(),
            working_bit_depth,
        ) {
            record_failure(
                &mut report,
                &cli.output_dir,
                "load",
                format!(
                    "failed to apply orientation correction to input component {} ({}): {}",
                    index + 1,
                    path.display(),
                    error
                ),
                load_start,
                options.write_artifacts,
            )?;
            return Err(error.into());
        }
        loaded_components.push((path.clone(), loaded));
    }
    let load_fidelity = loaded_components
        .iter()
        .map(|(_, loaded)| load_fidelity_evidence(&loaded.diagnostics))
        .collect::<Vec<_>>();
    let load_phase_confidence = if load_fidelity.is_empty() {
        0.0
    } else {
        load_fidelity
            .iter()
            .map(|evidence| evidence.confidence)
            .fold(1.0f64, f64::min)
    };
    let full_fidelity_component_count = load_fidelity
        .iter()
        .filter(|evidence| evidence.status == "full_declared_decode_fidelity")
        .count();
    let load_fidelity_status = if full_fidelity_component_count == load_fidelity.len() {
        "all_components_full_declared_decode_fidelity"
    } else if full_fidelity_component_count > 0 {
        "mixed_component_decode_fidelity"
    } else {
        "no_component_full_declared_decode_fidelity"
    };
    let component_load_metrics = loaded_components
        .iter()
        .enumerate()
        .map(|(index, (path, loaded))| {
            serde_json::json!({
                "index": index + 1,
                "path": path.to_string_lossy(),
                "shape": [loaded.image.shape()[0], loaded.image.shape()[1]],
                "decode": tiff_load_metrics_json(&loaded.diagnostics),
            })
        })
        .collect::<Vec<_>>();
    let mut load_metrics = serde_json::json!({
        "input_count": loaded_components.len(),
        "inputs": cli.inputs.iter().map(|path| path.to_string_lossy()).collect::<Vec<_>>(),
        "components": component_load_metrics,
        "requested_working_bit_depth": cli.bit_depth,
        "preserved_working_bit_depth": working_bit_depth,
        "decode_fidelity_status": load_fidelity_status,
        "decode_fidelity_evaluated": true,
        "full_fidelity_component_count": full_fidelity_component_count,
        "confidence_basis": "minimum_component_declared_decode_fidelity",
        "confidence_definition": "weakest component declared precision/channel-layout/orientation/DNG-level evidence; successful decoding can have confidence below one",
    });
    if let serde_json::Value::Object(metrics) = &mut load_metrics {
        if let Some((path, loaded)) = loaded_components.first() {
            metrics.insert(
                "component1".to_string(),
                serde_json::json!(path.to_string_lossy()),
            );
            metrics.insert(
                "img1_shape".to_string(),
                serde_json::json!([loaded.image.shape()[0], loaded.image.shape()[1]]),
            );
            metrics.insert(
                "component1_decode".to_string(),
                tiff_load_metrics_json(&loaded.diagnostics),
            );
        }
        if let Some((path, loaded)) = loaded_components.get(1) {
            metrics.insert(
                "component2".to_string(),
                serde_json::json!(path.to_string_lossy()),
            );
            metrics.insert(
                "img2_shape".to_string(),
                serde_json::json!([loaded.image.shape()[0], loaded.image.shape()[1]]),
            );
            metrics.insert(
                "component2_decode".to_string(),
                tiff_load_metrics_json(&loaded.diagnostics),
            );
        }
    }
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let mut phase = PhaseReport::ok("load", load_phase_confidence, load_metrics);
            for (index, (_, loaded)) in loaded_components.iter().enumerate() {
                phase.warnings.extend(tiff_load_warnings(
                    &format!("input component {}", index + 1),
                    &loaded.diagnostics,
                ));
            }
            if working_bit_depth > cli.bit_depth {
                phase.warnings.push(format!(
                    "requested {}-bit processing would discard source precision; promoted the working domain to {} bits",
                    cli.bit_depth, working_bit_depth
                ));
            }
            if load_phase_confidence < 0.999_999 {
                phase.warnings.push(format!(
                    "load fidelity confidence is limited to {:.3} by the weakest component's declared decode evidence",
                    load_phase_confidence
                ));
            }
            phase
        },
        load_start,
        options.write_artifacts,
    )?;

    // Scanner response and spatial shading are defined in each source scan's own scanner frame.
    // Select the scanner record before any crop or mosaic operation, then linearize every decoded
    // component independently. Roll selection may be refined later from the measured film base,
    // but scanner selection itself is deliberately independent of that observation.
    let scanner_linearization_start = Instant::now();
    let pre_geometry_calibration = color_calibration::load_calibration(
        cli.calibration_profile.as_deref(),
        cli.calibration_library.as_deref(),
        cli.scanner_profile.as_deref(),
        cli.roll_profile.as_deref(),
        cli.film_stock.as_deref(),
        None,
    );
    let selected_pre_geometry_linearization = pre_geometry_calibration
        .profile
        .as_ref()
        .and_then(|profile| profile.scanner_linearization.clone());
    let mut applied_scanner_linearization = None;
    if let Some(linearization) = selected_pre_geometry_linearization.as_ref() {
        let source_bit_depth = working_bit_depth;
        let base_override_before = base_color_override;
        let mut component_diagnostics = Vec::<serde_json::Value>::new();
        let mut maximum_extrapolated_ratio = 0.0f64;
        let mut maximum_clipped_ratio = 0.0f64;
        for (index, (_, loaded)) in loaded_components.iter_mut().enumerate() {
            let orientation = &loaded.diagnostics.orientation;
            let coordinate_mapping = scanner_linearization::ScannerCoordinateMapping {
                orientation_tag: loaded.diagnostics.effective_orientation_tag,
                scanner_frame_width: orientation.source_width,
                scanner_frame_height: orientation.source_height,
            };
            let source_image = std::mem::replace(&mut loaded.image, Array3::zeros((0, 0, 3)));
            let result = match scanner_linearization::apply_scanner_linearization_with_coordinates(
                source_image,
                source_bit_depth,
                linearization,
                coordinate_mapping,
            ) {
                Ok(result) => result,
                Err(errors) => {
                    let message = format!(
                        "selected scanner linearization failed on component {} before crop/stitch: {}",
                        index + 1,
                        errors.join("; ")
                    );
                    record_failure(
                        &mut report,
                        &cli.output_dir,
                        "scanner_linearization",
                        message.clone(),
                        scanner_linearization_start,
                        options.write_artifacts,
                    )?;
                    return Err(message.into());
                }
            };
            loaded.image = result.image;
            let diagnostics = &result.diagnostics;
            let total_pixels = diagnostics.input_sample_count.max(1) as f64;
            let extrapolated_ratio = (0..3)
                .map(|channel| {
                    (diagnostics.curve_extrapolated_low_samples[channel]
                        + diagnostics.curve_extrapolated_high_samples[channel])
                        as f64
                        / total_pixels
                })
                .fold(0.0, f64::max);
            let clipped_ratio = (0..3)
                .map(|channel| {
                    (diagnostics.clipped_low_samples[channel]
                        + diagnostics.clipped_high_samples[channel]) as f64
                        / total_pixels
                })
                .fold(0.0, f64::max);
            maximum_extrapolated_ratio = maximum_extrapolated_ratio.max(extrapolated_ratio);
            maximum_clipped_ratio = maximum_clipped_ratio.max(clipped_ratio);
            component_diagnostics.push(serde_json::json!({
                "index": index + 1,
                "path": cli.inputs[index].to_string_lossy(),
                "diagnostics": diagnostics,
                "maximum_channel_curve_extrapolated_ratio": extrapolated_ratio,
                "maximum_channel_clipped_ratio": clipped_ratio,
                "output_shape": [loaded.image.shape()[0], loaded.image.shape()[1]],
            }));
            if options.write_artifacts && cli.debug {
                tiff_io::save_tiff_u16(
                    &loaded.image,
                    &cli.output_dir.join(format!(
                        "component_{:02}_scanner_linearized.tiff",
                        index + 1
                    )),
                )?;
            }
        }
        working_bit_depth = 16;
        if let Some(base_color) = base_color_override {
            base_color_override = Some(scanner_linearization::transform_base_color(
                base_color,
                source_bit_depth,
                linearization,
            ));
        }
        applied_scanner_linearization = Some(linearization.clone());
        let representative_diagnostics = component_diagnostics
            .first()
            .and_then(|component| component.get("diagnostics"))
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let mut phase = PhaseReport::ok(
            "scanner_linearization",
            linearization.confidence,
            serde_json::json!({
                "skipped": false,
                "application_stage": "per_component_before_border_crop_classification_and_stitch",
                "coordinate_domain": "full_decoded_component_mapped_back_to_original_scanner_frame",
                "input_count": component_diagnostics.len(),
                "diagnostics": representative_diagnostics,
                "components": component_diagnostics,
                "base_color_before": base_override_before,
                "base_color_after": base_color_override,
                "base_transform_reference_position": "scanner_frame_center_for_manual_override_only; automatic base is measured after linearization",
                "maximum_channel_curve_extrapolated_ratio": maximum_extrapolated_ratio,
                "maximum_channel_clipped_ratio": maximum_clipped_ratio,
                "output_bit_depth": working_bit_depth,
                "selection_without_observed_base": pre_geometry_calibration.diagnostics,
            }),
        );
        if maximum_extrapolated_ratio > 0.01 {
            phase.warnings.push(format!(
                "scanner linearization extrapolated beyond measured curve support for up to {:.2}% of one component/channel",
                maximum_extrapolated_ratio * 100.0
            ));
            phase.confidence = phase.confidence.min(0.75);
        }
        if maximum_clipped_ratio > 0.001 {
            phase.warnings.push(format!(
                "scanner correction clipped up to {:.3}% of one component/channel after black/flare/curve/shading correction",
                maximum_clipped_ratio * 100.0
            ));
            phase.confidence = phase.confidence.min(0.75);
        }
        record_phase(
            &mut report,
            &cli.output_dir,
            phase,
            scanner_linearization_start,
            options.write_artifacts,
        )?;
    } else {
        record_phase(
            &mut report,
            &cli.output_dir,
            PhaseReport::ok(
                "scanner_linearization",
                0.0,
                serde_json::json!({
                    "skipped": true,
                    "application_stage": "per_component_before_border_crop_classification_and_stitch",
                    "coordinate_domain": "not_applicable",
                    "reason": pre_geometry_calibration.diagnostics.reason,
                    "selection_status": pre_geometry_calibration.diagnostics.status,
                    "selection_source": pre_geometry_calibration.diagnostics.source,
                }),
            ),
            scanner_linearization_start,
            options.write_artifacts,
        )?;
    }

    let deskew_start = Instant::now();
    let deskew_config = deskew::DeskewConfig::default();
    let deskew_maximum_value = ((1u32 << working_bit_depth.min(16)) - 1) as f64;
    let deskew_input_count = loaded_components.len();
    let mut deskew_diagnostics = Vec::with_capacity(deskew_input_count);
    match cli.geometry.deskew {
        DeskewMode::Off => {
            for (_, loaded) in &loaded_components {
                deskew_diagnostics.push(deskew::no_op_diagnostics(
                    "off",
                    "disabled",
                    "absolute deskew was disabled explicitly",
                    &loaded.image,
                ));
            }
        }
        DeskewMode::Auto if deskew_input_count != 1 => {
            for (_, loaded) in &loaded_components {
                deskew_diagnostics.push(deskew::no_op_diagnostics(
                    "auto",
                    "skipped_multi_input_auto",
                    "automatic absolute deskew currently applies only to a single scan; multi-scan relative rotation remains evidence-gated by affine overlap validation",
                    &loaded.image,
                ));
            }
        }
        DeskewMode::Auto | DeskewMode::Manual => {
            for (index, (_, loaded)) in loaded_components.iter_mut().enumerate() {
                let source = std::mem::replace(&mut loaded.image, Array3::zeros((0, 0, 3)));
                let result = if cli.geometry.deskew == DeskewMode::Auto {
                    deskew::auto_deskew(source, deskew_maximum_value, &deskew_config)
                } else {
                    deskew::manual_deskew(
                        source,
                        cli.geometry.deskew_angle_degrees,
                        deskew_maximum_value,
                        &deskew_config,
                    )
                };
                loaded.image = result.image;
                if options.write_artifacts && cli.debug && result.diagnostics.applied {
                    tiff_io::save_tiff_u16(
                        &loaded.image,
                        &cli.output_dir
                            .join(format!("component_{:02}_deskewed.tiff", index + 1)),
                    )?;
                }
                deskew_diagnostics.push(result.diagnostics);
            }
        }
    }
    let deskew_review_reasons = deskew_diagnostics
        .iter()
        .enumerate()
        .filter(|(_, diagnostics)| diagnostics.review_required)
        .map(|(index, diagnostics)| format!("component {}: {}", index + 1, diagnostics.reason))
        .collect::<Vec<_>>();
    let deskew_review_required = !deskew_review_reasons.is_empty();
    let deskew_review_reason = if deskew_review_required {
        deskew_review_reasons.join("; ")
    } else {
        "absolute deskew did not expose unresolved significant-skew evidence".to_string()
    };
    let deskew_applied_component_count = deskew_diagnostics
        .iter()
        .filter(|diagnostics| diagnostics.applied)
        .count();
    let representative_deskew = deskew_diagnostics.first();
    let deskew_component_metrics = deskew_diagnostics
        .iter()
        .enumerate()
        .map(|(index, diagnostics)| {
            serde_json::json!({
                "index": index + 1,
                "path": cli.inputs[index].to_string_lossy(),
                "diagnostics": diagnostics,
            })
        })
        .collect::<Vec<_>>();
    let deskew_confidence = representative_deskew
        .map(|diagnostics| diagnostics.confidence)
        .unwrap_or(0.0);
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let mut phase = PhaseReport::ok(
                "deskew",
                deskew_confidence,
                serde_json::json!({
                    "requested_mode": cli.geometry.deskew.as_str(),
                    "status": representative_deskew.map(|diagnostics| diagnostics.status.as_str()),
                    "applied": deskew_applied_component_count > 0,
                    "applied_component_count": deskew_applied_component_count,
                    "input_count": deskew_input_count,
                    "detected_source_skew_degrees": representative_deskew.and_then(|diagnostics| diagnostics.detected_source_skew_degrees),
                    "correction_degrees": representative_deskew.and_then(|diagnostics| diagnostics.correction_degrees),
                    "review_required": deskew_review_required,
                    "review_reason": deskew_review_reason,
                    "retained_area_ratio": representative_deskew.map(|diagnostics| diagnostics.retained_area_ratio),
                    "proposed_retained_area_ratio": representative_deskew.and_then(|diagnostics| diagnostics.proposed_retained_area_ratio),
                    "supporting_side_count": representative_deskew.map(|diagnostics| diagnostics.supporting_side_count),
                    "horizontal_side_count": representative_deskew.map(|diagnostics| diagnostics.horizontal_side_count),
                    "vertical_side_count": representative_deskew.map(|diagnostics| diagnostics.vertical_side_count),
                    "side_angle_spread_degrees": representative_deskew.and_then(|diagnostics| diagnostics.side_angle_spread_degrees),
                    "interpolation": representative_deskew.and_then(|diagnostics| diagnostics.interpolation.as_deref()),
                    "reason": representative_deskew.map(|diagnostics| diagnostics.reason.as_str()),
                    "components": deskew_component_metrics,
                    "application_stage": "after_scanner_linearization_before_border_crop_classification_and_stitch",
                }),
            );
            if deskew_review_required {
                phase.warnings.push(format!(
                    "absolute deskew requires geometry review: {deskew_review_reason}"
                ));
            }
            phase
        },
        deskew_start,
        options.write_artifacts,
    )?;

    let border_start = Instant::now();
    let mut bordered_components = Vec::with_capacity(loaded_components.len());
    for (index, (path, loaded)) in loaded_components.into_iter().enumerate() {
        let tiff_io::LoadedTiff {
            image,
            diagnostics: load_diagnostics,
            embedded_icc_profile,
        } = loaded;
        let border_result = border::remove_borders_with_diagnostics(&image, 2);
        let pre_crop_base = (!positive_input_mode)
            .then(|| detect_base_before_content_crop(&image, &border_result.diagnostics, 2));
        if options.write_artifacts && cli.debug {
            let debug_path = cli
                .output_dir
                .join(format!("component_{:02}_cropped.tiff", index + 1));
            if let Err(err) = tiff_io::save_tiff_u16(&border_result.cropped, &debug_path) {
                record_failure(
                    &mut report,
                    &cli.output_dir,
                    "border_removal",
                    format!(
                        "failed to save cropped debug image for component {}: {}",
                        index + 1,
                        err
                    ),
                    border_start,
                    options.write_artifacts,
                )?;
                return Err(err);
            }
        }
        bordered_components.push(BorderedComponent {
            path,
            load_diagnostics,
            embedded_icc_profile,
            cropped: border_result.cropped,
            border_diagnostics: border_result.diagnostics,
            pre_crop_base,
        });
    }
    let border_component_metrics = bordered_components
        .iter()
        .enumerate()
        .map(|(index, component)| {
            serde_json::json!({
                "index": index + 1,
                "path": component.path.to_string_lossy(),
                "crop": border_metrics_json(&component.border_diagnostics),
                "output_shape": [component.cropped.shape()[0], component.cropped.shape()[1]],
            })
        })
        .collect::<Vec<_>>();
    let border_component_evidence = bordered_components
        .iter()
        .map(|component| border_crop_evidence(&component.border_diagnostics))
        .collect::<Vec<_>>();
    let evaluated_component_count = border_component_evidence
        .iter()
        .filter(|evidence| evidence.evidence_evaluated)
        .count();
    let supported_component_count = border_component_evidence
        .iter()
        .filter(|evidence| evidence.applied_crop_evidence_supported)
        .count();
    let all_components_supported = !border_component_evidence.is_empty()
        && supported_component_count == border_component_evidence.len();
    let border_phase_confidence = if border_component_evidence.is_empty() {
        0.0
    } else {
        border_component_evidence
            .iter()
            .map(|evidence| evidence.confidence)
            .fold(1.0f64, f64::min)
    };
    let border_phase_status = if all_components_supported {
        "all_components_supported_applied_crop"
    } else if supported_component_count > 0 {
        "partial_components_supported_applied_crop"
    } else {
        "no_component_supported_applied_crop"
    };
    let border_phase_confidence_basis = if all_components_supported {
        "minimum_component_applied_crop_evidence_confidence"
    } else if supported_component_count > 0 {
        "weakest_component_includes_no_supported_applied_crop_evidence"
    } else {
        "no_supported_applied_crop_evidence"
    };
    let border_review_reasons = border_component_evidence
        .iter()
        .enumerate()
        .filter_map(|(index, evidence)| {
            evidence
                .review_reason
                .map(|reason| format!("component {}: {reason}", index + 1))
        })
        .collect::<Vec<_>>();
    let mut border_metrics = serde_json::json!({
        "input_count": bordered_components.len(),
        "border_decision_status": border_phase_status,
        "border_evidence_evaluated": evaluated_component_count == bordered_components.len(),
        "evaluated_component_count": evaluated_component_count,
        "supported_component_count": supported_component_count,
        "all_components_have_supported_applied_crop": all_components_supported,
        "confidence_basis": border_phase_confidence_basis,
        "confidence_definition": "minimum component applied-crop evidence strength; zero when any component has no supported applied crop or an unsafe/not-evaluated decision",
        "review_required": !border_review_reasons.is_empty(),
        "review_reasons": border_review_reasons.clone(),
        "components": border_component_metrics,
    });
    if let serde_json::Value::Object(metrics) = &mut border_metrics {
        if let Some(component) = bordered_components.first() {
            metrics.insert(
                "component1".to_string(),
                border_metrics_json(&component.border_diagnostics),
            );
            metrics.insert(
                "comp1_shape".to_string(),
                serde_json::json!([component.cropped.shape()[0], component.cropped.shape()[1]]),
            );
        }
        if let Some(component) = bordered_components.get(1) {
            metrics.insert(
                "component2".to_string(),
                border_metrics_json(&component.border_diagnostics),
            );
            metrics.insert(
                "comp2_shape".to_string(),
                serde_json::json!([component.cropped.shape()[0], component.cropped.shape()[1]]),
            );
        }
    }
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let mut phase =
                PhaseReport::ok("border_removal", border_phase_confidence, border_metrics);
            for (index, component) in bordered_components.iter().enumerate() {
                phase.warnings.extend(
                    component
                        .border_diagnostics
                        .warnings
                        .iter()
                        .map(|warning| format!("component {}: {}", index + 1, warning)),
                );
            }
            phase
        },
        border_start,
        options.write_artifacts,
    )?;

    let classify_start = Instant::now();
    let mut components = Vec::with_capacity(bordered_components.len());
    for (index, component) in bordered_components.into_iter().enumerate() {
        let (detection, measurement, measurement_stage, measurement_reason, analysis) =
            if positive_input_mode {
                let detection = base_detect::positive_input_not_applicable(&component.cropped);
                (
                    detection.clone(),
                    detection,
                    "not_required_positive_input",
                    "negative-film base detection and frame classification were skipped for already-positive input"
                        .to_string(),
                    frame_classify::positive_input_component(&component.cropped),
                )
            } else {
                let detection = base_detect::detect_film_base(&component.cropped);
                let (measurement, measurement_stage, measurement_reason) =
                    match component.pre_crop_base.as_ref() {
                        Some(pre_crop) => {
                            let (selected, stage, reason) =
                                select_base_measurement(pre_crop, &detection);
                            (selected.clone(), stage, reason)
                        }
                        None => (
                            detection.clone(),
                            "post_crop_content_edges",
                            "post-crop negative-film base detection used".to_string(),
                        ),
                    };
                let analysis = frame_classify::analyze_component(&component.cropped, &detection);
                (
                    detection,
                    measurement,
                    measurement_stage,
                    measurement_reason,
                    analysis,
                )
            };
        if options.write_artifacts && cli.debug && !positive_input_mode {
            save_base_debug_artifacts(
                &format!("component_{:02}", index + 1),
                &cli.output_dir,
                &component.cropped,
                &detection,
            )?;
        }
        components.push(PreparedComponent {
            path: component.path,
            load_diagnostics: component.load_diagnostics,
            embedded_icc_profile: component.embedded_icc_profile,
            border_diagnostics: component.border_diagnostics,
            cropped: component.cropped,
            detection,
            measurement,
            measurement_stage,
            measurement_reason,
            analysis,
        });
    }
    let stitch_decision = if components.len() == 1 {
        frame_classify::StitchDecision {
            disposition: frame_classify::StitchDisposition::Skip,
            reason: "single_input_no_stitch_required".to_string(),
            confidence: 0.0,
        }
    } else if positive_input_mode {
        if cli.force_no_stitch {
            frame_classify::StitchDecision {
                disposition: frame_classify::StitchDisposition::Skip,
                reason: "force_no_stitch".to_string(),
                confidence: 1.0,
            }
        } else if cli.force_stitch {
            frame_classify::StitchDecision {
                disposition: frame_classify::StitchDisposition::Forced,
                reason: "force_stitch_positive_sequence".to_string(),
                confidence: 1.0,
            }
        } else {
            frame_classify::StitchDecision {
                disposition: frame_classify::StitchDisposition::Attempt,
                reason: "positive_multi_input_requires_overlap_scoring".to_string(),
                confidence: 0.6,
            }
        }
    } else if components.len() == 2 {
        frame_classify::decide_stitch_attempt(
            &components[0].analysis,
            &components[1].analysis,
            cli.force_stitch,
            cli.force_no_stitch,
        )
    } else if cli.force_no_stitch {
        frame_classify::StitchDecision {
            disposition: frame_classify::StitchDisposition::Skip,
            reason: "force_no_stitch".to_string(),
            confidence: 1.0,
        }
    } else if cli.force_stitch {
        frame_classify::StitchDecision {
            disposition: frame_classify::StitchDisposition::Forced,
            reason: "force_stitch_sequence".to_string(),
            confidence: 1.0,
        }
    } else {
        frame_classify::StitchDecision {
            disposition: frame_classify::StitchDisposition::Attempt,
            reason: "multi_component_sequence_requires_overlap_graph_scoring".to_string(),
            confidence: 0.6,
        }
    };
    let stitch_decision_user_override = matches!(
        stitch_decision.reason.as_str(),
        "force_no_stitch"
            | "force_stitch"
            | "force_stitch_sequence"
            | "force_stitch_positive_sequence"
    );
    let stitch_decision_evidence_evaluated = components.len() > 1
        && !stitch_decision_user_override
        && !matches!(
            stitch_decision.reason.as_str(),
            "positive_multi_input_requires_overlap_scoring"
                | "multi_component_sequence_requires_overlap_graph_scoring"
        );
    let stitch_decision_confidence_basis = if components.len() == 1 {
        "not_applicable_single_input"
    } else if stitch_decision_user_override {
        "user_authoritative_override"
    } else if stitch_decision_evidence_evaluated {
        "measured_negative_film_component_topology"
    } else {
        "routing_prior_pending_overlap_scoring"
    };
    let classification_component_metrics = components
        .iter()
        .enumerate()
        .map(|(index, component)| {
            let mut value = component_classification_metrics_json(component);
            if let serde_json::Value::Object(metrics) = &mut value {
                metrics.insert("index".to_string(), serde_json::json!(index + 1));
            }
            value
        })
        .collect::<Vec<_>>();
    let mut classification_metrics = serde_json::json!({
        "input_count": components.len(),
        "classification_status": if positive_input_mode {
            "not_applicable_positive_input"
        } else {
            "evaluated_negative_film_rebate_topology"
        },
        "classification_evidence_evaluated": !positive_input_mode,
        "confidence_basis": if positive_input_mode {
            "not_evaluated"
        } else {
            "maximum_component_measured_rebate_classification_confidence"
        },
        "components": classification_component_metrics,
        "stitch_decision": {
            "disposition": format!("{:?}", stitch_decision.disposition),
            "reason": stitch_decision.reason.clone(),
            "confidence": stitch_decision.confidence,
            "evidence_evaluated": stitch_decision_evidence_evaluated,
            "confidence_basis": stitch_decision_confidence_basis,
            "user_override": stitch_decision_user_override,
        },
    });
    if let serde_json::Value::Object(metrics) = &mut classification_metrics {
        if let Some(component) = components.first() {
            metrics.insert(
                "component1".to_string(),
                component_classification_metrics_json(component),
            );
        }
        if let Some(component) = components.get(1) {
            metrics.insert(
                "component2".to_string(),
                component_classification_metrics_json(component),
            );
        }
    }
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let maximum_confidence = components
                .iter()
                .map(|component| component.analysis.confidence)
                .fold(0.0f64, f64::max);
            let mut phase = PhaseReport::ok(
                "base_detect_classify",
                maximum_confidence,
                classification_metrics,
            );
            for (index, component) in components.iter().enumerate() {
                let confidence = base_detection_confidence(&component.measurement);
                if !positive_input_mode && confidence < BASE_CONFIDENCE_WARN {
                    phase.warnings.push(format!(
                        "component {} base confidence is low ({:.3}); classification should be treated as provisional",
                        index + 1,
                        confidence
                    ));
                }
            }
            phase
        },
        classify_start,
        options.write_artifacts,
    )?;

    let fallback_index = best_single_component_index(&components);
    let fallback_component = components[fallback_index].cropped.clone();
    let stitch_start = Instant::now();
    let should_score_stitch = stitch_decision.should_score();
    let (
        working_image,
        used_stitched_working_image,
        stitch_completion_review_required,
        stitch_quality_review_reasons,
    ) = if should_score_stitch {
        let transform_mode = TransformMode::from_cli(&cli.transform)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
        let stitch_config = StitchConfig {
            transform_mode,
            use_opencv: cli.use_opencv,
            input_bit_depth: working_bit_depth,
            expected_y_offset: if components.len() == 2 {
                let first_top = components[0]
                    .border_diagnostics
                    .top_removed
                    .min(i32::MAX as usize) as i32;
                let second_top = components[1]
                    .border_diagnostics
                    .top_removed
                    .min(i32::MAX as usize) as i32;
                second_top.saturating_sub(first_top)
            } else {
                0
            },
            target_width: None,
            target_height: None,
            debug_dir: if options.write_artifacts && cli.debug {
                Some(cli.output_dir.clone())
            } else {
                None
            },
            ..StitchConfig::default()
        };
        if components.len() == 2 {
            let stitch_result = stitch::stitch_components(
                &components[0].cropped,
                &components[1].cropped,
                &stitch_config,
            );
            let stitch_quality_review_reasons =
                stitch_quality_review_reasons(&stitch_result.report);
            let stitched_image = stitch_result.result;
            record_phase(
                &mut report,
                &cli.output_dir,
                stitch_result.report,
                stitch_start,
                options.write_artifacts,
            )?;
            match stitched_image {
                Some(stitched) => (stitched, true, false, stitch_quality_review_reasons),
                None => (
                    fallback_component.clone(),
                    false,
                    true,
                    stitch_quality_review_reasons,
                ),
            }
        } else {
            let component_refs = components
                .iter()
                .map(|component| &component.cropped)
                .collect::<Vec<_>>();
            let vertical_crop_origins = components
                .iter()
                .map(|component| {
                    component
                        .border_diagnostics
                        .top_removed
                        .min(i32::MAX as usize) as i32
                })
                .collect::<Vec<_>>();
            let sequence_result = stitch::stitch_component_sequence_with_vertical_origins(
                &component_refs,
                &vertical_crop_origins,
                &stitch_config,
            );
            let stitch_quality_review_reasons =
                stitch_quality_review_reasons(&sequence_result.report);
            let stitched_image = sequence_result.result;
            record_phase(
                &mut report,
                &cli.output_dir,
                sequence_result.report,
                stitch_start,
                options.write_artifacts,
            )?;
            match stitched_image {
                Some(stitched) => (stitched, true, false, stitch_quality_review_reasons),
                None => (
                    fallback_component.clone(),
                    false,
                    true,
                    stitch_quality_review_reasons,
                ),
            }
        }
    } else {
        record_phase(
            &mut report,
            &cli.output_dir,
            PhaseReport::ok(
                "stitch",
                stitch_decision.confidence,
                serde_json::json!({
                    "decision": if components.len() == 1 {
                        "skipped_single_input"
                    } else {
                        "skipped_pre_score"
                    },
                    "reason": stitch_decision.reason.clone(),
                    "disposition": format!("{:?}", stitch_decision.disposition),
                    "stitch_evidence_evaluated": stitch_decision_evidence_evaluated,
                    "confidence_basis": stitch_decision_confidence_basis,
                    "user_override": stitch_decision_user_override,
                    "input_count": components.len(),
                    "fallback_component": fallback_index + 1,
                    "preserved_full_valid_union": components.len() == 1,
                }),
            ),
            stitch_start,
            options.write_artifacts,
        )?;
        (fallback_component.clone(), false, false, Vec::new())
    };
    if options.write_artifacts && cli.debug && used_stitched_working_image {
        tiff_io::save_tiff_u16(&working_image, &cli.output_dir.join("stitched.tiff"))?;
    }
    let stitch_review_required =
        stitch_completion_review_required || !stitch_quality_review_reasons.is_empty();
    let stitch_review_reason = if stitch_completion_review_required {
        let completion_reason = "one or more requested input components could not be joined through a validated overlap; the rendered fallback contains only the strongest single component and is not a complete frame";
        if stitch_quality_review_reasons.is_empty() {
            completion_reason.to_string()
        } else {
            format!(
                "{completion_reason}; {}",
                stitch_quality_review_reasons.join("; ")
            )
        }
    } else if !stitch_quality_review_reasons.is_empty() {
        format!(
            "all requested components were joined, but seam-quality evidence requires review: {}",
            stitch_quality_review_reasons.join("; ")
        )
    } else if used_stitched_working_image {
        format!(
            "all {} requested input components were joined through validated overlap evidence",
            components.len()
        )
    } else if components.len() == 1 {
        "single input required no geometric stitching".to_string()
    } else {
        "stitching was explicitly or confidently skipped and the strongest complete-looking component was selected"
            .to_string()
    };
    let border_review_required = !border_review_reasons.is_empty();
    let mut geometry_review_reasons = Vec::new();
    if border_review_required {
        geometry_review_reasons.push(format!(
            "border crop requires review: {}",
            border_review_reasons.join("; ")
        ));
    }
    if deskew_review_required {
        geometry_review_reasons.push(deskew_review_reason.clone());
    }
    if stitch_review_required {
        geometry_review_reasons.push(stitch_review_reason.clone());
    }
    let geometry_review_required = !geometry_review_reasons.is_empty();
    let geometry_review_reason = if geometry_review_required {
        geometry_review_reasons.join("; ")
    } else {
        "border crop and absolute/relative geometry completed without a significant unresolved review condition"
            .to_string()
    };

    let (working_icc_profile, source_icc_selection_status, source_icc_selection_reason) =
        if !positive_input_mode {
            (
                None,
                "not_applied_negative_input",
                "embedded ICC conversion is deferred for negative scans because density reconstruction requires scanner-channel linearization rather than a display-referred RGB conversion"
                    .to_string(),
            )
        } else if used_stitched_working_image {
            let first_profile = components
                .first()
                .and_then(|component| component.embedded_icc_profile.as_ref());
            let profile_count = components
                .iter()
                .filter(|component| component.embedded_icc_profile.is_some())
                .count();
            if profile_count == components.len() {
                let first = first_profile.expect("profile count proves a first embedded profile");
                if components.iter().all(|component| {
                    component
                        .embedded_icc_profile
                        .as_ref()
                        .is_some_and(|profile| first.has_identical_payload(profile))
                }) {
                    (
                        Some(first.clone()),
                        "matched_embedded_profiles",
                        format!(
                            "all {} stitched components carry byte-identical embedded ICC profiles",
                            components.len()
                        ),
                    )
                } else {
                    (
                        None,
                        "profile_mismatch",
                        "stitched components carry different embedded ICC profiles; a single post-stitch transform would be colorimetrically invalid"
                            .to_string(),
                    )
                }
            } else {
                (
                    None,
                    "incomplete_profile_coverage",
                    format!(
                        "only {} of {} stitched components carry embedded ICC profiles; source color cannot be transformed consistently after stitching",
                        profile_count,
                        components.len()
                    ),
                )
            }
        } else {
            match components[fallback_index].embedded_icc_profile.as_ref() {
                Some(profile) => (
                    Some(profile.clone()),
                    "selected_component_profile",
                    format!(
                        "using the embedded ICC profile from selected component {}",
                        fallback_index + 1
                    ),
                ),
                None => (
                    None,
                    "unprofiled",
                    format!(
                        "selected component {} has no embedded ICC profile",
                        fallback_index + 1
                    ),
                ),
            }
        };

    let working_start = Instant::now();
    let mut base_estimate_source_for_negative: Option<String> = None;
    let mut input_mode_review_required = false;
    let mut input_mode_review_reason =
        "negative input mode selected; positive-mode suitability inspection is not applicable"
            .to_string();
    let (base_color_for_negative, base_confidence) = if positive_input_mode {
        let positive_input_inspection =
            positive_input::inspect_u16_image(&working_image, working_bit_depth);
        input_mode_review_required = positive_input_inspection.likely_negative_like;
        input_mode_review_reason = positive_input_inspection.reason.clone();
        let input_mode_suitability_status = if input_mode_review_required {
            "review_required_negative_like_positive_input"
        } else if positive_input_inspection.accepted_high_warm_score {
            "accepted_warm_positive_input"
        } else {
            "accepted_positive_input"
        };
        record_phase(
            &mut report,
            &cli.output_dir,
            {
                let mut phase = PhaseReport::ok(
                    "working_image_select",
                    if input_mode_review_required { 0.0 } else { 1.0 },
                    serde_json::json!({
                        "input_mode": cli.input_mode.as_str(),
                        "input_mode_suitability_status": input_mode_suitability_status,
                        "input_mode_suitability_evaluated": true,
                        "input_mode_review_required": input_mode_review_required,
                        "input_mode_review_reason": input_mode_review_reason,
                        "confidence_basis": "positive_input_channel_ratio_and_orange_mask_inspection",
                        "input_count": components.len(),
                        "working_shape": [working_image.shape()[0], working_image.shape()[1]],
                        "fallback_component": fallback_index + 1,
                        "used_stitched_working_image": used_stitched_working_image,
                        "stitch_review_required": stitch_review_required,
                        "stitch_review_reason": stitch_review_reason.clone(),
                        "base_required": false,
                        "base_estimate_source": "not_applicable_positive_input",
                        "base_estimate_reason": "already-positive input; film-base detection is not required for density inversion",
                        "base_color_override_applied": false,
                        "base_color_override_ignored": base_color_supplied,
                        "used_low_confidence_fallback": false,
                        "source_icc_selection_status": source_icc_selection_status,
                        "source_icc_selection_reason": source_icc_selection_reason,
                        "source_icc_profile": working_icc_profile.as_ref().map(|profile| &profile.diagnostics),
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
                if matches!(
                    source_icc_selection_status,
                    "profile_mismatch" | "incomplete_profile_coverage"
                ) {
                    phase.warnings.push(source_icc_selection_reason.clone());
                }
                if stitch_review_required {
                    phase.warnings.push(stitch_review_reason.clone());
                    phase.confidence = phase.confidence.min(0.35);
                }
                if let Some(profile) = working_icc_profile.as_ref() {
                    if !profile.is_valid_rgb_profile() {
                        phase.warnings.push(format!(
                            "selected embedded ICC profile cannot be applied: {}",
                            profile.diagnostics.reason
                        ));
                    }
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
        let raw_edge_balance = if raw_base_confidence <= 1e-6 {
            1.0
        } else {
            (det_final.left_confidence.min(det_final.right_confidence) / raw_base_confidence)
                .clamp(0.0, 1.0)
        };
        let detected_base_reconciliation = if used_stitched_working_image {
            let component_bases = components
                .iter()
                .map(|component| &component.measurement)
                .filter(|measurement| {
                    measurement.base_color_source != "high_transmittance_fallback"
                        && base_detection_confidence(measurement) >= BASE_CONFIDENCE_WARN
                })
                .map(|measurement| measurement.base_color)
                .collect::<Vec<_>>();
            if component_bases.len() >= 2 {
                base_detect::reconcile_stitched_base_estimate(&det_final, &component_bases)
            } else {
                base_detect::BaseReconciliation {
                    base_color: det_final.base_color,
                    confidence: raw_base_confidence,
                    source: det_final.base_color_source,
                    reason: format!(
                        "stitched working image retained its own base estimate because fewer than two confident pre-crop rebate measurements were available: {}",
                        det_final.base_color_reason
                    ),
                    raw_working_base_color: det_final.base_color,
                    raw_working_confidence: raw_base_confidence,
                    component_consensus_base_color: None,
                    component_consensus_confidence: None,
                    component_consensus_relative_spread: None,
                    working_vs_consensus_relative_delta: None,
                    edge_balance_ratio: raw_edge_balance,
                }
            }
        } else {
            let component = &components[fallback_index];
            let measurement = &component.measurement;
            let measurement_confidence = base_detection_confidence(measurement);
            let measurement_is_direct = measurement.base_color_source
                != "high_transmittance_fallback"
                && measurement_confidence >= BASE_CONFIDENCE_WARN;
            if measurement_is_direct
                && (raw_base_confidence < BASE_CONFIDENCE_WARN
                    || measurement_confidence > raw_base_confidence + 0.05)
            {
                base_detect::BaseReconciliation {
                    base_color: measurement.base_color,
                    confidence: measurement_confidence,
                    source: component.measurement_stage,
                    reason: format!(
                        "using the separately retained component rebate measurement after final content selection: {}; {}",
                        component.measurement_reason, measurement.base_color_reason
                    ),
                    raw_working_base_color: det_final.base_color,
                    raw_working_confidence: raw_base_confidence,
                    component_consensus_base_color: None,
                    component_consensus_confidence: None,
                    component_consensus_relative_spread: None,
                    working_vs_consensus_relative_delta: None,
                    edge_balance_ratio: raw_edge_balance,
                }
            } else {
                base_detect::BaseReconciliation {
                    base_color: det_final.base_color,
                    confidence: raw_base_confidence,
                    source: det_final.base_color_source,
                    reason: format!(
                        "working image kept its direct base estimate because the separately retained component measurement was not stronger: {}; {}",
                        component.measurement_reason, det_final.base_color_reason
                    ),
                    raw_working_base_color: det_final.base_color,
                    raw_working_confidence: raw_base_confidence,
                    component_consensus_base_color: None,
                    component_consensus_confidence: None,
                    component_consensus_relative_spread: None,
                    working_vs_consensus_relative_delta: None,
                    edge_balance_ratio: raw_edge_balance,
                }
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
                        "input_count": components.len(),
                        "working_shape": [working_image.shape()[0], working_image.shape()[1]],
                        "fallback_component": fallback_index + 1,
                        "used_stitched_working_image": used_stitched_working_image,
                        "stitch_review_required": stitch_review_required,
                        "stitch_review_reason": stitch_review_reason.clone(),
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
                } else if base_reconciliation.source == "pre_crop_rebate_measurement" {
                    phase.warnings.push(format!(
                        "pre-crop film-base measurement retained separately from the final content crop; {}",
                        base_reconciliation.reason
                    ));
                } else if base_reconciliation.source != "working_edges" {
                    phase.warnings.push(format!(
                        "working-image crop edges disagreed with the pre-stitch base prior; {}",
                        base_reconciliation.reason
                    ));
                }
                if stitch_review_required {
                    phase.warnings.push(stitch_review_reason.clone());
                    phase.confidence = phase.confidence.min(0.35);
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

    let explicit_color_model_requested = cli.calibration_profile.is_some()
        || cli.calibration_library.is_some()
        || cli.scanner_profile.is_some()
        || cli.roll_profile.is_some()
        || cli.film_stock.is_some()
        || cli.color_mode != colorspace::ColorMode::Auto;
    let mut calibration = color_calibration::load_calibration(
        cli.calibration_profile.as_deref(),
        cli.calibration_library.as_deref(),
        cli.scanner_profile.as_deref(),
        cli.roll_profile.as_deref(),
        cli.film_stock.as_deref(),
        base_color_for_negative,
    );
    if calibration.diagnostics.status == "not_configured" {
        let load_diagnostics = components
            .iter()
            .map(|component| &component.load_diagnostics)
            .collect::<Vec<_>>();
        if let Some(dng_calibration) = automatic_dng_advisory_calibration(cli, &load_diagnostics) {
            calibration = dng_calibration;
        }
    }
    drop(components);
    drop(fallback_component);

    let selected_post_measurement_linearization = calibration
        .profile
        .as_ref()
        .and_then(|profile| profile.scanner_linearization.clone());
    if applied_scanner_linearization != selected_post_measurement_linearization {
        let consistency_start = Instant::now();
        let message = format!(
            "scanner linearization selection changed after film-base measurement (pre-geometry model {:?}, post-measurement model {:?}); refusing a late mosaic-coordinate correction",
            applied_scanner_linearization
                .as_ref()
                .map(|linearization| linearization.model_id.as_str()),
            selected_post_measurement_linearization
                .as_ref()
                .map(|linearization| linearization.model_id.as_str())
        );
        record_failure(
            &mut report,
            &cli.output_dir,
            "scanner_linearization_selection_consistency",
            message.clone(),
            consistency_start,
            options.write_artifacts,
        )?;
        return Err(message.into());
    }
    if applied_scanner_linearization.is_some() {
        if let Some(source) = base_estimate_source_for_negative.as_mut() {
            source.push_str("+validated_scanner_linearization_per_component_pre_crop");
        }
        if options.write_artifacts && cli.debug {
            tiff_io::save_tiff_u16(
                &working_image,
                &cli.output_dir
                    .join("phase25_scanner_linearized_working_image.tiff"),
            )?;
        }
    }

    let mut positive_for_direct: Option<Array3<f64>> = None;
    let mut direct_density_render_scale: Option<f64> = None;
    let mut direct_density_diagnostics: Option<density::DensityDiagnostics> = None;
    let mut direct_density_response_model: Option<density::NegativeResponseModel> = None;
    let mut negative_density_phase_confidence: Option<f64> = None;
    let mut direct_density_response_candidates = serde_json::Value::Null;
    let mut direct_density_reconstruction_diagnostics = serde_json::Value::Null;
    let (mut transmittance, skipped_ica_for_direct_density) = if positive_input_mode {
        let density_start = Instant::now();
        let positive_rgb =
            density::normalize_positive_scan_to_linear_rgb(&working_image, working_bit_depth);
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
                0.0,
                serde_json::json!({
                    "skipped": true,
                    "operation_status": "not_applicable_positive_input",
                    "evidence_evaluated": false,
                    "negative_response_model_evidence_evaluated": false,
                    "film_base_evidence_confidence": serde_json::Value::Null,
                    "negative_response_model_confidence": serde_json::Value::Null,
                    "confidence_basis": "not_evaluated",
                    "confidence_definition": "negative-density reconstruction confidence is zero because inversion and response-model evidence are not applicable to already-positive input",
                    "runtime_coverage_validation": "not applicable to already-positive input",
                    "input_mode": cli.input_mode.as_str(),
                    "reason": "already-positive input normalized directly; density inversion and orange-mask removal are not required",
                    "bit_depth": working_bit_depth,
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
                0.0,
                serde_json::json!({
                    "skipped": true,
                    "operation_status": "not_applicable_positive_input",
                    "evidence_evaluated": false,
                    "confidence_basis": "not_evaluated",
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
        } = density::phase3_invert_with_diagnostics(&working_image, &base_color, working_bit_depth);
        direct_density_render_scale = Some(density::DEFAULT_NEGATIVE_DENSITY_RESPONSE_SCALE);
        let unit_response_model = density::unit_negative_response_model();
        let frame_response_model = density::estimate_regularized_frame_response(
            &positive_density,
            diagnostics.shared_robust_d_max,
        );
        let measured_response_model = calibration
            .profile
            .as_ref()
            .and_then(|profile| profile.negative_response.as_ref())
            .and_then(|response| density::measured_negative_response_model(response).ok());
        let selected_response_model = if let Some(measured) = measured_response_model.as_ref() {
            measured.clone()
        } else if frame_response_model.diagnostics.accepted {
            frame_response_model.clone()
        } else {
            unit_response_model.clone()
        };
        let (
            negative_response_model_confidence,
            negative_response_model_evidence_evaluated,
            density_operation_status,
            negative_response_confidence_basis,
        ) = negative_response_model_phase_evidence(&selected_response_model.diagnostics);
        let density_confidence = base_confidence.min(negative_response_model_confidence);
        negative_density_phase_confidence = Some(density_confidence);
        direct_density_response_candidates = serde_json::json!({
            "unit_slope": &unit_response_model.diagnostics,
            "regularized_frame_luminance_axis": &frame_response_model.diagnostics,
            "measured_nonlinear_dye_separation": measured_response_model.as_ref().map(|model| &model.diagnostics),
            "selected_model": selected_response_model.diagnostics.model,
            "selection_reason": if selected_response_model.diagnostics.model == "measured_nonlinear_dye_separation" {
                "the scanner-matched roll response supplied measured dye-crosstalk separation and nonlinear characteristic curves that passed held-out DeltaE00 and noise-gain gates"
            } else if selected_response_model.diagnostics.model == "regularized_frame_luminance_axis" {
                "the bounded frame-derived luminance-axis candidate passed correlation, explained-variance, conditioning, and noise-gain gates; it remains review-required because no measured film target was available"
            } else {
                "the frame-derived response candidate failed evidence gates; retained the explicit unit-density-slope fallback"
            },
        });
        direct_density_diagnostics = Some(diagnostics.clone());
        direct_density_response_model = Some(selected_response_model);
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
                working_bit_depth,
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
                    density_confidence,
                    serde_json::json!({
                        "skipped": false,
                        "operation_status": density_operation_status,
                        "evidence_evaluated": negative_response_model_evidence_evaluated,
                        "negative_response_model_evidence_evaluated": negative_response_model_evidence_evaluated,
                        "film_base_evidence_confidence": base_confidence,
                        "negative_response_model_confidence": negative_response_model_confidence,
                        "confidence_basis": "minimum_film_base_and_negative_response_model_evidence",
                        "negative_response_confidence_basis": negative_response_confidence_basis,
                        "confidence_definition": "weakest pre-application physical support among the film-base estimate and selected negative-response model; successful log-density arithmetic does not establish response accuracy",
                        "runtime_coverage_validation": "colorspace_mapping.negative_response_reconstruction evaluates actual curve coverage after application and can only reduce final trust",
                        "input_mode": cli.input_mode.as_str(),
                        "base_color": base_color,
                        "base_estimate_source": base_estimate_source_for_negative,
                        "bit_depth": working_bit_depth,
                        "base_transmittance": diagnostics.base_transmittance,
                        "base_density": diagnostics.base_density,
                        "robust_d_max": diagnostics.robust_d_max,
                        "shared_robust_d_max": diagnostics.shared_robust_d_max,
                        "direct_density_response_scale": density::DEFAULT_NEGATIVE_DENSITY_RESPONSE_SCALE,
                        "direct_density_response_source": direct_density_response_model.as_ref().map(|model| model.diagnostics.source),
                        "direct_density_response_model": direct_density_response_model.as_ref().map(|model| &model.diagnostics),
                        "direct_density_response_candidates": direct_density_response_candidates,
                        "direct_density_render_normalization": "shared_robust_white_anchor_with_signed_highlight_headroom",
                        "exact_d_max": diagnostics.exact_d_max,
                        "d_max_percentile": diagnostics.d_max_percentile,
                        "histogram_bins": diagnostics.histogram_bins,
                        "highlight_headroom_samples": diagnostics.highlight_headroom_samples,
                        "highlight_headroom_clipped": false,
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
                if base_confidence < BASE_CONFIDENCE_WARN {
                    phase.warnings.push(format!(
                        "density inversion proceeded with low base confidence ({:.3})",
                        base_confidence
                    ));
                }
                if !negative_response_model_evidence_evaluated {
                    phase.warnings.push(format!(
                        "density inversion selected `{}` without held-out physical negative-response validation; successful arithmetic inversion does not establish film-response accuracy",
                        direct_density_response_model
                            .as_ref()
                            .map(|model| model.diagnostics.model)
                            .unwrap_or("unknown_negative_response")
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
                0.0,
                serde_json::json!({
                    "skipped": true,
                    "operation_status": "not_evaluated_explicit_direct_density_route",
                    "evidence_evaluated": false,
                    "confidence_basis": "not_evaluated_user_selected_alternate_route",
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
            let density_diagnostics = require_pipeline_state_ref(
                direct_density_diagnostics.as_ref(),
                &mut report,
                &cli.output_dir,
                "fastica",
                "density diagnostics were not available for direct-density render input",
                ica_start,
                options.write_artifacts,
            )?;
            let response_model = require_pipeline_state_ref(
                direct_density_response_model.as_ref(),
                &mut report,
                &cli.output_dir,
                "fastica",
                "negative response model was not available for direct-density render input",
                ica_start,
                options.write_artifacts,
            )?;
            let reconstruction = density::reconstruct_with_negative_response(
                positive,
                density_diagnostics,
                response_model,
            );
            direct_density_reconstruction_diagnostics =
                serde_json::to_value(&reconstruction.diagnostics)?;
            (reconstruction.scene_linear, true)
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
                0.0,
                serde_json::json!({
                    "skipped": false,
                    "operation_status": if ica_result.converged {
                        "numerically_converged_physical_separation_unvalidated"
                    } else {
                        "numerical_convergence_failed_physical_separation_unvalidated"
                    },
                    "evidence_evaluated": false,
                    "numerical_convergence_evaluated": true,
                    "numerical_convergence_confidence": if ica_result.converged { 1.0 } else { 0.0 },
                    "physical_separation_evidence_evaluated": false,
                    "confidence_basis": "physical_dye_separation_not_independently_validated",
                    "confidence_definition": "physical dye-separation evidence strength, not numerical optimizer convergence",
                    "downstream_candidate_validation": "colorspace_mapping_compares_ica_and_direct_density_render_safety_and_colour_quality",
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
            ica_phase.warnings.push(
                "ICA numerical convergence does not independently validate physical dye separation; inspect downstream ICA/direct-density candidate quality and measured negative-response evidence"
                    .to_string(),
            );
            if ica_phase.metrics["confidence_limited_by_base_estimate"] == true {
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
    let mut input_color_transform_metrics = serde_json::json!({
        "status": if positive_input_mode { "not_applied" } else { "not_applicable_negative_input" },
        "source_icc_selection_status": source_icc_selection_status,
        "source_icc_selection_reason": source_icc_selection_reason,
        "reason": if positive_input_mode {
            if explicit_color_model_requested {
                "an explicit color model was requested; embedded ICC conversion was not composed with that model"
            } else if working_icc_profile.is_some() {
                "selected embedded ICC profile was not usable"
            } else {
                "no single usable embedded ICC profile was selected for the working image"
            }
        } else {
            "negative-film reconstruction retains scanner device channels until density and dye reconstruction"
        },
    });
    let mut embedded_icc_transform_applied = false;
    if positive_input_mode && !explicit_color_model_requested {
        if let Some(profile) = working_icc_profile
            .as_ref()
            .filter(|profile| profile.is_valid_rgb_profile())
        {
            let source_image =
                std::mem::replace(&mut transmittance, Array3::<f64>::zeros((0, 0, 3)));
            match input_color::transform_to_linear_prophoto(source_image, profile) {
                Ok(result) => {
                    transmittance = result.image;
                    input_color_transform_metrics = serde_json::to_value(&result.diagnostics)?;
                    embedded_icc_transform_applied = true;
                }
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
        }
    }
    let calibration_profile = calibration.profile.as_ref();
    let positive_rgb_passthrough = positive_input_mode
        && calibration_profile.is_none()
        && cli.color_mode == colorspace::ColorMode::Auto;
    let mut calibration_diagnostics = calibration.diagnostics.clone();
    if positive_rgb_passthrough && calibration_diagnostics.status == "not_configured" {
        if embedded_icc_transform_applied {
            calibration_diagnostics.source = "embedded_icc_profile".to_string();
            calibration_diagnostics.reason = "no external calibration model was configured; the embedded ICC source profile nevertheless established the positive input color space and was transformed into linear ProPhoto RGB D50"
                .to_string();
        } else {
            calibration_diagnostics.source = "positive_rgb_passthrough".to_string();
            calibration_diagnostics.reason = "no external calibration profile was provided; positive RGB values are preserved, but their source color space is unverified and requires review"
                .to_string();
        }
    }
    let mut colorspace_result = if positive_rgb_passthrough {
        if embedded_icc_transform_applied {
            colorspace::map_profiled_positive_scan_rgb_to_prophoto_d50_with_diagnostics(
                &transmittance,
            )
        } else {
            colorspace::map_positive_scan_rgb_to_prophoto_d50_with_diagnostics(&transmittance)
        }
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
        if embedded_icc_transform_applied {
            "already-positive input transformed through its embedded ICC source profile into linear ProPhoto RGB D50"
                .to_string()
        } else {
            POSITIVE_RENDER_INPUT_REASON.to_string()
        }
    } else if skipped_ica_for_direct_density {
        "direct density transmittance selected by --render-input direct-density".to_string()
    } else {
        "ICA-separated density channels remained stable enough for colorspace mapping".to_string()
    };
    let mut render_input_warning: Option<String> =
        (!positive_input_mode && skipped_ica_for_direct_density).then(|| {
            "direct density transmittance selected by --render-input direct-density".to_string()
        });
    let direct_density_auto_base_eligible =
        positive_input_mode || base_confidence >= BASE_CONFIDENCE_FALLBACK;
    let direct_density_auto_base_rejection_reason = (!positive_input_mode
        && cli.render_input == RenderInputMode::Auto
        && !direct_density_auto_base_eligible)
        .then(|| {
            format!(
                "automatic direct-density selection is disabled because film-base confidence {:.3} is below the {:.3} minimum; density referenced to a fallback base is not physically trustworthy",
                base_confidence, BASE_CONFIDENCE_FALLBACK
            )
        });
    if let Some(reason) = &direct_density_auto_base_rejection_reason {
        render_input_warning = Some(reason.clone());
    }
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
        let density_diagnostics = require_pipeline_state_ref(
            direct_density_diagnostics.as_ref(),
            &mut report,
            &cli.output_dir,
            "colorspace_mapping",
            "density diagnostics were not available for direct-density candidate evaluation",
            colorspace_start,
            options.write_artifacts,
        )?;
        let response_model = require_pipeline_state_ref(
            direct_density_response_model.as_ref(),
            &mut report,
            &cli.output_dir,
            "colorspace_mapping",
            "negative response model was not available for direct-density candidate evaluation",
            colorspace_start,
            options.write_artifacts,
        )?;
        let reconstruction = density::reconstruct_with_negative_response(
            positive,
            density_diagnostics,
            response_model,
        );
        direct_density_reconstruction_diagnostics =
            serde_json::to_value(&reconstruction.diagnostics)?;
        let direct_transmittance = reconstruction.scene_linear;
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

        let measured_response_selected =
            response_model.diagnostics.model == "measured_nonlinear_dye_separation";
        let measured_response_can_force_direct = direct_density_auto_base_eligible
            && measured_response_selected
            && !colorspace::has_catastrophic_render_mapping(&direct_result.diagnostics);
        let direct_selection_reason = match cli.render_input {
            RenderInputMode::DirectDensity => Some(
                "direct density transmittance selected by --render-input direct-density"
                    .to_string(),
            ),
            RenderInputMode::Auto if measured_response_can_force_direct => Some(format!(
                "direct density transmittance selected because held-out-validated measured negative-response model `{}` supplies scanner-matched dye separation and nonlinear characteristic curves; that physical evidence outranks blind frame-derived ICA and its selected color mapping passed catastrophic clipping gates",
                response_model
                    .diagnostics
                    .measured_model_id
                    .as_deref()
                    .unwrap_or("unnamed-measured-response")
            )),
            RenderInputMode::Auto if !direct_density_auto_base_eligible => None,
            RenderInputMode::Auto => {
                colorspace::direct_density_render_fallback_reason(
                    &colorspace_result.diagnostics,
                    &direct_result.diagnostics,
                )
            }
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
        } else if direct_density_auto_base_eligible {
            direct_density_detail_guide = Some(direct_result.prophoto);
        }
    }
    let (negative_response_review_required, negative_response_review_reason) =
        if positive_input_mode {
            (
                false,
                "negative-film response reconstruction is not applicable to positive input"
                    .to_string(),
            )
        } else if render_input_source == "direct_density_transmittance" {
            (
                direct_density_reconstruction_diagnostics["review_required"]
                    .as_bool()
                    .unwrap_or(true),
                direct_density_reconstruction_diagnostics["review_reason"]
                    .as_str()
                    .unwrap_or(
                        "negative-response reconstruction diagnostics were unavailable; review is required",
                    )
                    .to_string(),
            )
        } else {
            (
                true,
                "ICA is a frame-derived blind density separation, not a scanner/film response measured against held-out reference patches"
                    .to_string(),
            )
        };
    let (
        negative_reconstruction_evidence_confidence,
        negative_reconstruction_evidence_evaluated,
        negative_reconstruction_confidence_status,
        negative_reconstruction_confidence_basis,
    ) = negative_reconstruction_color_evidence(
        positive_input_mode,
        render_input_source,
        negative_response_review_required,
        direct_density_response_model.as_ref(),
        negative_density_phase_confidence,
    );
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
    calibration_diagnostics.color_mapping_application = Some(
        calibration_color_mapping_application_diagnostics(&colorspace_result.diagnostics),
    );
    let calibration_metrics = serde_json::to_value(&calibration_diagnostics)?;
    let mut selected_colorspace_metrics =
        colorspace_candidate_metrics_json(&colorspace_result.diagnostics);
    if let serde_json::Value::Object(metrics) = &mut selected_colorspace_metrics {
        metrics.insert("target".to_string(), serde_json::json!("ProPhoto_D50"));
        metrics.insert(
            "input_domain".to_string(),
            serde_json::json!(if positive_input_mode {
                if embedded_icc_transform_applied {
                    "linear_prophoto_rgb_d50_from_embedded_icc"
                } else {
                    "unprofiled_positive_scan_rgb"
                }
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
        metrics.insert(
            "input_color_transform".to_string(),
            input_color_transform_metrics,
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
        metrics.insert(
            "direct_density_auto_selection_eligible".to_string(),
            serde_json::json!(direct_density_auto_base_eligible),
        );
        metrics.insert(
            "direct_density_auto_selection_rejection_reason".to_string(),
            serde_json::json!(direct_density_auto_base_rejection_reason),
        );
        metrics.insert("ica_candidate".to_string(), ica_candidate_metrics);
        metrics.insert(
            "direct_density_candidate".to_string(),
            direct_density_candidate_metrics,
        );
        metrics.insert(
            "direct_density_response_scale".to_string(),
            serde_json::json!(direct_density_render_scale),
        );
        metrics.insert(
            "negative_response_candidates".to_string(),
            direct_density_response_candidates,
        );
        metrics.insert(
            "negative_response_reconstruction".to_string(),
            direct_density_reconstruction_diagnostics,
        );
        metrics.insert(
            "negative_response_review_required".to_string(),
            serde_json::json!(negative_response_review_required),
        );
        metrics.insert(
            "negative_response_review_reason".to_string(),
            serde_json::json!(negative_response_review_reason),
        );
        metrics.insert(
            "negative_reconstruction_evidence_evaluated".to_string(),
            serde_json::json!(negative_reconstruction_evidence_evaluated),
        );
        metrics.insert(
            "negative_reconstruction_evidence_confidence".to_string(),
            serde_json::json!(negative_reconstruction_evidence_confidence),
        );
        metrics.insert(
            "negative_reconstruction_confidence_status".to_string(),
            serde_json::json!(negative_reconstruction_confidence_status),
        );
        metrics.insert(
            "negative_reconstruction_confidence_basis".to_string(),
            serde_json::json!(negative_reconstruction_confidence_basis),
        );
        metrics.insert(
            "direct_density_render_normalization".to_string(),
            serde_json::json!(direct_density_render_scale.map(|_| {
                "response_corrected_shared_log_exposure_anchor_with_signed_highlight_headroom"
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
            let (selected_mapping_confidence, selected_mapping_confidence_status) =
                color_decision_confidence(&colorspace_result.diagnostics);
            let negative_reconstruction_limit =
                negative_reconstruction_evidence_confidence.unwrap_or(1.0);
            let color_confidence_before_base =
                selected_mapping_confidence.min(negative_reconstruction_limit);
            let confidence_limited_by_negative_reconstruction =
                negative_reconstruction_evidence_confidence.is_some()
                    && selected_mapping_confidence > negative_reconstruction_limit + 1e-12;
            let color_confidence_status = if selected_mapping_confidence <= f64::EPSILON {
                selected_mapping_confidence_status
            } else if negative_reconstruction_evidence_confidence
                .is_some_and(|confidence| confidence <= f64::EPSILON)
            {
                "review_required_negative_reconstruction"
            } else if confidence_limited_by_negative_reconstruction {
                "limited_negative_reconstruction_evidence"
            } else {
                selected_mapping_confidence_status
            };
            let mut phase = PhaseReport::ok(
                "colorspace_mapping",
                color_confidence_before_base,
                selected_colorspace_metrics,
            );
            if let serde_json::Value::Object(metrics) = &mut phase.metrics {
                metrics.insert(
                    "color_evidence_evaluated".to_string(),
                    serde_json::json!(true),
                );
                metrics.insert(
                    "color_confidence_status".to_string(),
                    serde_json::json!(color_confidence_status),
                );
                metrics.insert(
                    "selected_mapping_evidence_confidence".to_string(),
                    serde_json::json!(selected_mapping_confidence),
                );
                metrics.insert(
                    "selected_mapping_confidence_status".to_string(),
                    serde_json::json!(selected_mapping_confidence_status),
                );
                metrics.insert(
                    "confidence_before_negative_reconstruction_limit".to_string(),
                    serde_json::json!(selected_mapping_confidence),
                );
                metrics.insert(
                    "confidence_limited_by_negative_reconstruction_evidence".to_string(),
                    serde_json::json!(confidence_limited_by_negative_reconstruction),
                );
                metrics.insert(
                    "color_confidence_before_base_limit".to_string(),
                    serde_json::json!(color_confidence_before_base),
                );
                metrics.insert(
                    "confidence_basis".to_string(),
                    serde_json::json!(
                        "minimum_selected_mapping_negative_reconstruction_and_working_image_base_evidence"
                    ),
                );
                metrics.insert(
                    "confidence_definition".to_string(),
                    serde_json::json!("weakest evidence for trusting the complete selected colour reconstruction; mapping success cannot restore blind, unmeasured, or out-of-support negative reconstruction"),
                );
            }
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
            if negative_response_review_required {
                phase.warnings.push(format!(
                    "negative-response reconstruction requires review: {negative_response_review_reason}"
                ));
            }
            if calibration.diagnostics.status == "rejected" {
                phase.warnings.push(format!(
                    "calibration profile was rejected; using image-derived colorspace mapping: {}",
                    calibration.diagnostics.reason
                ));
            }
            if positive_input_mode
                && explicit_color_model_requested
                && working_icc_profile
                    .as_ref()
                    .is_some_and(|profile| profile.is_valid_rgb_profile())
            {
                phase.warnings.push(
                    "a usable embedded ICC source profile was not composed with the explicitly requested color model; verify that the explicit calibration consumes encoded scanner RGB rather than profile-linearized RGB"
                        .to_string(),
                );
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
            if color_diagnostics.neutral_safety_rescue.applied {
                phase.warnings.push(format!(
                    "colorspace neutral safety rescue applied: {}",
                    color_diagnostics.neutral_safety_rescue.reason
                ));
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

    let white_balance_start = Instant::now();
    let technical_white_balance =
        white_balance::apply_technical_white_balance(prophoto, &cli.white_balance);
    prophoto = technical_white_balance.image;
    let mut technical_white_balance_artifact = None;
    if options.write_artifacts && cli.debug {
        let path = cli
            .output_dir
            .join("phase47_technical_white_balance_scene_referred.tiff");
        tiff_io::save_tiff_f32_linear_prophoto_scene_referred(&prophoto, &path)?;
        technical_white_balance_artifact = Some(path.to_string_lossy().to_string());
    }
    let technical_white_balance_review_required =
        technical_white_balance.diagnostics.review_required;
    let technical_white_balance_review_reason = technical_white_balance.diagnostics.reason.clone();
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let (confidence, evidence_evaluated, confidence_basis, decision_status) =
                technical_white_balance_phase_evidence(
                    &technical_white_balance.diagnostics.status,
                    technical_white_balance.diagnostics.confidence,
                );
            let mut phase = PhaseReport::ok(
                "white_balance",
                confidence,
                serde_json::json!({
                    "technical": technical_white_balance.diagnostics,
                    "technical_decision_status": decision_status,
                    "technical_evidence_evaluated": evidence_evaluated,
                    "confidence_basis": confidence_basis,
                    "confidence_definition": "measured automatic scene-neutral evidence, an explicit user-authoritative manual setting, or zero for a disabled non-evaluation",
                    "technical_master_artifact": technical_white_balance_artifact,
                    "creative_defaults": {
                        "temperature": cli.white_balance.creative_temperature,
                        "tint": cli.white_balance.creative_tint,
                        "applied_only_to_finished_render": true,
                        "excluded_from_scene_referred_master": true
                    },
                    "technical_creative_separation": true,
                    "output_domain": "scene_referred_linear_prophoto_rgb_d50",
                }),
            );
            if technical_white_balance_review_required {
                phase.warnings.push(format!(
                    "technical white balance requires review: {technical_white_balance_review_reason}"
                ));
            }
            phase
        },
        white_balance_start,
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
        geometry_review_required,
        geometry_review_reason,
        input_mode_review_required,
        input_mode_review_reason,
        negative_response_review_required,
        negative_response_review_reason,
        technical_white_balance_review_required,
        technical_white_balance_review_reason,
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

fn prepare_owned_batch_render(
    cache: &mut InteractiveRenderCache,
    write_partial_report: bool,
) -> Result<OwnedBatchRenderContext, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&cache.cli.output_dir)?;
    let tone_input_linear_percentiles = tonemap::tone_input_linear_percentiles(&cache.prophoto);
    let master_path = cache
        .cli
        .should_write_master()
        .then(|| cache.cli.output_dir.join("master_scene_referred.tiff"));
    let mut master_artifact_diagnostics = serde_json::Value::Null;
    let master_write_start = Instant::now();
    if let Some(path) = &master_path {
        if let Err(err) =
            tiff_io::save_tiff_f32_linear_prophoto_scene_referred(&cache.prophoto, path)
        {
            record_failure(
                &mut cache.report,
                &cache.cli.output_dir,
                "save",
                format!("failed to save scene-referred master: {err}"),
                master_write_start,
                write_partial_report,
            )?;
            return Err(err);
        }
        master_artifact_diagnostics = scene_referred_artifact_diagnostics(&cache.prophoto);
    }
    let master_write_duration = if master_path.is_some() {
        master_write_start.elapsed()
    } else {
        Duration::ZERO
    };
    let image = std::mem::replace(&mut cache.prophoto, Array3::zeros((0, 0, 3)));
    Ok(OwnedBatchRenderContext {
        image,
        tone_input_linear_percentiles,
        master_artifact_diagnostics,
        master_write_duration,
        master_prewritten: master_path.is_some(),
    })
}

/// Run the full scanstitch pipeline.
pub fn run(cli: &Cli) -> Result<PipelineReport, Box<dyn std::error::Error>> {
    let mut cache = build_interactive_render_cache_with_options(
        cli,
        PipelineBuildOptions {
            write_artifacts: true,
        },
    )?;
    let (controls, applied_review_sidecar) = controls_for_cache(&cache)?;
    let owned_batch = prepare_owned_batch_render(&mut cache, true)?;
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
        Some(owned_batch),
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
        None,
    )
}

fn save_interactive_render_with_options(
    cache: &InteractiveRenderCache,
    controls: &InteractiveRenderControls,
    options: InteractiveSaveOptions,
    owned_batch: Option<OwnedBatchRenderContext>,
) -> Result<PipelineReport, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&cache.cli.output_dir)?;
    let cli = &cache.cli;
    let output_path: PathBuf = cache.output_path.clone();
    let mut report = cache.report.clone();
    let tone_params = controls.to_tone_params(&cache.auto_tone_params);
    let creative_white_balance = white_balance::creative_white_balance_diagnostics(
        controls.creative_temperature,
        controls.creative_tint,
    );

    let tone_start = Instant::now();
    let (
        tone_apply,
        tone_input_linear_percentiles,
        prewritten_master_diagnostics,
        prewritten_master_duration,
        master_prewritten,
        render_input_buffer_policy,
        master_preservation_policy,
    ) = if let Some(batch) = owned_batch {
        let master_preservation_policy = if batch.master_prewritten {
            "prewritten_before_owned_batch_render"
        } else {
            "not_requested_owned_batch_source_consumed"
        };
        (
            interactive::render_owned_batch_image(cache, batch.image, controls),
            batch.tone_input_linear_percentiles,
            batch.master_artifact_diagnostics,
            batch.master_write_duration,
            batch.master_prewritten,
            "owned_scene_buffer_reused_for_batch_render",
            master_preservation_policy,
        )
    } else {
        (
            interactive::render_interactive_image(cache, controls),
            tonemap::tone_input_linear_percentiles(&cache.prophoto),
            serde_json::Value::Null,
            Duration::ZERO,
            false,
            "separate_render_buffer_from_cached_interactive_master",
            "cached_interactive_master_unchanged",
        )
    };
    let tonemapped = tone_apply.image;
    let render_quality = tonemap::render_quality_diagnostics(&tonemapped);
    let render_grain = tonemap::render_grain_diagnostics(&tonemapped);
    let tone_output_evidence = assess_tone_output_evidence(
        cache.tone_fit_diagnostics.input_linear_percentiles,
        cache.tone_fit_diagnostics.mapped_linear_percentiles,
        render_quality.overall.luminance_percentiles,
        tone_apply
            .diagnostics
            .post_chroma_compression_clipped_high_ratio,
        tone_apply
            .diagnostics
            .post_chroma_compression_clipped_low_ratio,
        render_quality.sample_count,
    );
    if options.write_debug_artifacts && cli.debug {
        let lut: Vec<[f64; 2]> = (0..=255)
            .map(|i| {
                let x = i as f64 / 255.0;
                let y = tonemap::apply_tone_curve(x, &tone_params);
                [x, y]
            })
            .collect();
        let lut_json = serde_json::to_string_pretty(&lut)?;
        atomic_file::write_bytes(
            &cli.output_dir.join("tone_curve_lut.json"),
            lut_json.as_bytes(),
        )?;
    }
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let tone_fit = &cache.tone_fit_diagnostics;
            let upstream_color_confidence = cache
                .report
                .phases
                .iter()
                .find(|phase| phase.name == "colorspace_mapping")
                .map(|phase| phase.confidence)
                .unwrap_or(0.0);
            let grain_detail = &tone_apply.diagnostics.noise_reduction_detail_retention;
            let grain_evidence_confidence =
                if !tone_apply.diagnostics.noise_reduction_requested_enabled
                    || (tone_apply.diagnostics.noise_reduction_enabled
                        && grain_detail.decision_supported
                        && !grain_detail.review_required)
                {
                    1.0
                } else {
                    0.0
                };
            let confidence_before_tone_output =
                upstream_color_confidence.min(grain_evidence_confidence);
            let confidence_limited_by_tone_output =
                confidence_before_tone_output > tone_output_evidence.confidence + 1e-12;
            let tone_confidence =
                confidence_before_tone_output.min(tone_output_evidence.confidence);
            let tone_confidence_status = if upstream_color_confidence <= f64::EPSILON {
                "review_required_upstream_color"
            } else if grain_evidence_confidence <= f64::EPSILON {
                "review_required_requested_grain_evidence"
            } else if tone_output_evidence.confidence <= f64::EPSILON {
                "review_required_tone_output_evidence"
            } else if upstream_color_confidence < 0.999_999 {
                "limited_upstream_evidence"
            } else {
                "supported_tone_and_optional_grain_evidence"
            };
            let mut phase = PhaseReport::ok(
                "tone_mapping",
                tone_confidence,
                serde_json::json!({
                    "midpoint": tone_params.midpoint,
                    "slope": tone_params.slope,
                    "toe_lift": tone_params.toe_lift,
                    "shoulder_max": tone_params.shoulder_max,
                    "render_intent": cli.render_intent.as_str(),
                    "quality_mode": cli.quality_mode.as_str(),
                    "creative_white_balance": creative_white_balance,
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
                    "tone_evidence_evaluated".to_string(),
                    serde_json::json!(true),
                );
                metrics.insert(
                    "tone_confidence_status".to_string(),
                    serde_json::json!(tone_confidence_status),
                );
                metrics.insert(
                    "upstream_color_confidence".to_string(),
                    serde_json::json!(upstream_color_confidence),
                );
                metrics.insert(
                    "requested_grain_evidence_confidence".to_string(),
                    serde_json::json!(grain_evidence_confidence),
                );
                metrics.insert(
                    "tone_output_evidence_evaluated".to_string(),
                    serde_json::json!(tone_output_evidence.evaluated),
                );
                metrics.insert(
                    "tone_output_evidence_confidence".to_string(),
                    serde_json::json!(tone_output_evidence.confidence),
                );
                metrics.insert(
                    "tone_output_confidence_status".to_string(),
                    serde_json::json!(tone_output_evidence.status),
                );
                metrics.insert(
                    "tone_output_review_required".to_string(),
                    serde_json::json!(tone_output_evidence.confidence <= f64::EPSILON),
                );
                metrics.insert(
                    "tone_output_review_reason".to_string(),
                    serde_json::json!(tone_output_evidence.reason),
                );
                metrics.insert(
                    "confidence_before_tone_output_evidence".to_string(),
                    serde_json::json!(confidence_before_tone_output),
                );
                metrics.insert(
                    "confidence_limited_by_tone_output_evidence".to_string(),
                    serde_json::json!(confidence_limited_by_tone_output),
                );
                metrics.insert(
                    "input_luminance_range_p05_p95".to_string(),
                    serde_json::json!(tone_output_evidence.input_luminance_range_p05_p95),
                );
                metrics.insert(
                    "mapped_luminance_range_p05_p95".to_string(),
                    serde_json::json!(tone_output_evidence.mapped_luminance_range_p05_p95),
                );
                metrics.insert(
                    "render_to_mapped_luminance_range_ratio".to_string(),
                    serde_json::json!(tone_output_evidence.render_to_mapped_range_ratio),
                );
                metrics.insert(
                    "maximum_post_tone_high_clip_ratio".to_string(),
                    serde_json::json!(tone_output_evidence.maximum_post_tone_high_clip_ratio),
                );
                metrics.insert(
                    "maximum_post_tone_low_clip_ratio".to_string(),
                    serde_json::json!(tone_output_evidence.maximum_post_tone_low_clip_ratio),
                );
                metrics.insert(
                    "tone_output_evidence_thresholds".to_string(),
                    serde_json::json!({
                        "minimum_evaluable_range": TONE_OUTPUT_MIN_EVALUABLE_RANGE,
                        "minimum_render_range": TONE_OUTPUT_MIN_RENDER_RANGE,
                        "minimum_render_to_mapped_range_ratio": TONE_OUTPUT_MIN_RANGE_RETENTION_RATIO,
                        "maximum_catastrophic_channel_clip_ratio": TONE_OUTPUT_MAX_CATASTROPHIC_CHANNEL_CLIP_RATIO,
                    }),
                );
                metrics.insert(
                    "confidence_basis".to_string(),
                    serde_json::json!(
                        "minimum_upstream_color_requested_grain_and_rendered_tone_evidence"
                    ),
                );
                metrics.insert(
                    "confidence_definition".to_string(),
                    serde_json::json!("tone/render evidence strength; successful tone application does not restore untrusted colour, unsupported requested grain reduction, collapsed dynamic range, or catastrophically clipped output"),
                );
                metrics.insert(
                    "post_tone_buffer_policy".to_string(),
                    serde_json::json!("owned_in_place_single_working_buffer"),
                );
                metrics.insert(
                    "render_input_buffer_policy".to_string(),
                    serde_json::json!(render_input_buffer_policy),
                );
                metrics.insert(
                    "scene_referred_master_preserved".to_string(),
                    serde_json::json!(true),
                );
                metrics.insert(
                    "scene_referred_master_preservation_policy".to_string(),
                    serde_json::json!(master_preservation_policy),
                );
                metrics.insert(
                    "perceptual_gamut_mapping_space".to_string(),
                    serde_json::json!(tone_apply.diagnostics.perceptual_gamut_mapping_space),
                );
                metrics.insert(
                    "perceptual_gamut_mapped_ratio".to_string(),
                    serde_json::json!(tone_apply.diagnostics.perceptual_gamut_mapped_ratio),
                );
                metrics.insert(
                    "perceptual_gamut_mean_chroma_scale".to_string(),
                    serde_json::json!(tone_apply.diagnostics.perceptual_gamut_mean_chroma_scale),
                );
                metrics.insert(
                    "perceptual_gamut_min_chroma_scale".to_string(),
                    serde_json::json!(tone_apply.diagnostics.perceptual_gamut_min_chroma_scale),
                );
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
                    "tone_input_linear_percentiles".to_string(),
                    serde_json::json!(tone_input_linear_percentiles),
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
                let skin_memory = &tone_apply
                    .diagnostics
                    .adaptive_vibrance_skin_memory_protection;
                metrics.insert(
                    "adaptive_vibrance_skin_memory_protection".to_string(),
                    serde_json::json!({
                        "enabled": skin_memory.enabled,
                        "method": skin_memory.method,
                        "working_space": skin_memory.working_space,
                        "reference": skin_memory.reference,
                        "core_lightness": skin_memory.core_lightness,
                        "support_lightness": skin_memory.support_lightness,
                        "core_chroma": skin_memory.core_chroma,
                        "support_chroma": skin_memory.support_chroma,
                        "core_hue_degrees": skin_memory.core_hue_degrees,
                        "support_hue_degrees": skin_memory.support_hue_degrees,
                        "maximum_vibrance_reduction": skin_memory.maximum_vibrance_reduction,
                        "evaluated_pixel_ratio": skin_memory.evaluated_pixel_ratio,
                        "protected_pixel_ratio": skin_memory.protected_pixel_ratio,
                        "mean_protection_weight": skin_memory.mean_protection_weight,
                        "max_protection_weight": skin_memory.max_protection_weight,
                    }),
                );
                let preferred_memory = &tone_apply
                    .diagnostics
                    .adaptive_vibrance_preferred_memory_color_guard;
                metrics.insert(
                    "adaptive_vibrance_preferred_memory_color_guard".to_string(),
                    serde_json::json!({
                        "enabled": preferred_memory.enabled,
                        "method": preferred_memory.method,
                        "working_space": preferred_memory.working_space,
                        "reference": preferred_memory.reference,
                        "interpretation": preferred_memory.interpretation,
                        "core_normalized_radius": preferred_memory.core_normalized_radius,
                        "support_normalized_radius": preferred_memory.support_normalized_radius,
                        "evaluated_pixel_ratio": preferred_memory.evaluated_pixel_ratio,
                        "matched_pixel_ratio": preferred_memory.matched_pixel_ratio,
                        "limited_pixel_ratio": preferred_memory.limited_pixel_ratio,
                        "mean_scale_reduction": preferred_memory.mean_scale_reduction,
                        "max_scale_reduction": preferred_memory.max_scale_reduction,
                        "families": preferred_memory.families.iter().map(|family| serde_json::json!({
                            "family": family.family,
                            "preferred_center_lab": family.preferred_center_lab,
                            "preferred_center_lch": family.preferred_center_lch,
                            "semi_major_axis_ab": family.semi_major_axis_ab,
                            "semi_minor_axis_ab": family.semi_minor_axis_ab,
                            "axis_ratio": family.axis_ratio,
                            "ellipse_rotation_degrees": family.ellipse_rotation_degrees,
                            "matched_pixel_ratio": family.matched_pixel_ratio,
                            "limited_pixel_ratio": family.limited_pixel_ratio,
                            "mean_scale_reduction": family.mean_scale_reduction,
                            "max_scale_reduction": family.max_scale_reduction,
                        })).collect::<Vec<_>>(),
                    }),
                );
                let preferred_skin = &tone_apply.diagnostics.preferred_skin_rendering;
                metrics.insert(
                    "preferred_skin_rendering".to_string(),
                    serde_json::json!({
                        "enabled": preferred_skin.enabled,
                        "reason": preferred_skin.reason,
                        "method": preferred_skin.method,
                        "working_space": preferred_skin.working_space,
                        "preference_reference": preferred_skin.preference_reference,
                        "support_reference": preferred_skin.support_reference,
                        "interpretation": preferred_skin.interpretation,
                        "preferred_center_lab": preferred_skin.preferred_center_lab,
                        "preferred_center_lch": preferred_skin.preferred_center_lch,
                        "semi_major_axis_ab": preferred_skin.semi_major_axis_ab,
                        "semi_minor_axis_ab": preferred_skin.semi_minor_axis_ab,
                        "axis_ratio": preferred_skin.axis_ratio,
                        "ellipse_rotation_degrees": preferred_skin.ellipse_rotation_degrees,
                        "core_normalized_radius": preferred_skin.core_normalized_radius,
                        "radial_excess_reduction": preferred_skin.radial_excess_reduction,
                        "maximum_delta_e_ab": preferred_skin.maximum_delta_e_ab,
                        "minimum_support_weight": preferred_skin.minimum_support_weight,
                        "evaluated_pixel_ratio": preferred_skin.evaluated_pixel_ratio,
                        "matched_pixel_ratio": preferred_skin.matched_pixel_ratio,
                        "outside_preferred_core_ratio": preferred_skin.outside_preferred_core_ratio,
                        "adjusted_pixel_ratio": preferred_skin.adjusted_pixel_ratio,
                        "gamut_limited_pixel_ratio": preferred_skin.gamut_limited_pixel_ratio,
                        "mean_delta_e_ab": preferred_skin.mean_delta_e_ab,
                        "max_delta_e_ab": preferred_skin.max_delta_e_ab,
                        "mean_abs_hue_shift_degrees": preferred_skin.mean_abs_hue_shift_degrees,
                        "max_abs_hue_shift_degrees": preferred_skin.max_abs_hue_shift_degrees,
                        "mean_chroma_delta": preferred_skin.mean_chroma_delta,
                        "max_abs_chroma_delta": preferred_skin.max_abs_chroma_delta,
                    }),
                );
                metrics.insert(
                    "noise_reduction_enabled".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_enabled),
                );
                metrics.insert(
                    "noise_reduction_requested_enabled".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_requested_enabled),
                );
                metrics.insert(
                    "noise_reduction_reason".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_reason.clone()),
                );
                metrics.insert(
                    "noise_reduction_requested_strength".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_requested_strength),
                );
                metrics.insert(
                    "noise_reduction_requested_scale".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_requested_scale),
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
                    "noise_reduction_structure_gate_start".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_structure_gate_start),
                );
                metrics.insert(
                    "noise_reduction_structure_gate_end".to_string(),
                    serde_json::json!(tone_apply.diagnostics.noise_reduction_structure_gate_end),
                );
                metrics.insert(
                    "noise_reduction_structure_excluded_ratio".to_string(),
                    serde_json::json!(
                        tone_apply
                            .diagnostics
                            .noise_reduction_structure_excluded_ratio
                    ),
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
                    "noise_reduction_flat_luma_p95_reduction_ratio".to_string(),
                    serde_json::json!(
                        tone_apply
                            .diagnostics
                            .noise_reduction_flat_luma_p95_reduction_ratio
                    ),
                );
                metrics.insert(
                    "noise_reduction_flat_chroma_p95_reduction_ratio".to_string(),
                    serde_json::json!(
                        tone_apply
                            .diagnostics
                            .noise_reduction_flat_chroma_p95_reduction_ratio
                    ),
                );
                metrics.insert(
                    "noise_reduction_detail_review_required".to_string(),
                    serde_json::json!(
                        tone_apply
                            .diagnostics
                            .noise_reduction_detail_retention
                            .review_required
                    ),
                );
                metrics.insert(
                    "noise_reduction_detail_decision_supported".to_string(),
                    serde_json::json!(
                        tone_apply
                            .diagnostics
                            .noise_reduction_detail_retention
                            .decision_supported
                    ),
                );
                metrics.insert(
                    "grain_reduction".to_string(),
                    serde_json::json!({
                        "requested": {
                            "enabled": tone_apply.diagnostics.noise_reduction_requested_enabled,
                            "strength": tone_apply.diagnostics.noise_reduction_requested_strength,
                            "scale": tone_apply.diagnostics.noise_reduction_requested_scale,
                        },
                        "effective": {
                            "enabled": tone_apply.diagnostics.noise_reduction_enabled,
                            "radius": tone_apply.diagnostics.noise_reduction_radius,
                            "chroma_amount": tone_apply.diagnostics.noise_reduction_chroma_amount,
                            "luma_amount": tone_apply.diagnostics.noise_reduction_luma_amount,
                            "applied_ratio": tone_apply.diagnostics.noise_reduction_applied_ratio,
                            "structure_gate_start": tone_apply.diagnostics.noise_reduction_structure_gate_start,
                            "structure_gate_end": tone_apply.diagnostics.noise_reduction_structure_gate_end,
                            "structure_excluded_ratio": tone_apply.diagnostics.noise_reduction_structure_excluded_ratio,
                            "texture_limited_ratio": tone_apply.diagnostics.noise_reduction_texture_limited_ratio,
                            "saturation_limited_ratio": tone_apply.diagnostics.noise_reduction_saturation_limited_ratio,
                            "reason": tone_apply.diagnostics.noise_reduction_reason.clone(),
                        },
                        "before": render_grain_metrics_json(
                            &tone_apply.diagnostics.noise_reduction_pre_grain
                        ),
                        "after": render_grain_metrics_json(
                            &tone_apply.diagnostics.noise_reduction_post_grain
                        ),
                        "detail_retention": grain_detail_retention_metrics_json(
                            &tone_apply.diagnostics.noise_reduction_detail_retention
                        ),
                        "effect": {
                            "flat_luma_residual_p95_reduction_ratio": tone_apply.diagnostics.noise_reduction_flat_luma_p95_reduction_ratio,
                            "flat_chroma_residual_p95_reduction_ratio": tone_apply.diagnostics.noise_reduction_flat_chroma_p95_reduction_ratio,
                            "mean_abs_chroma_delta": tone_apply.diagnostics.noise_reduction_mean_abs_chroma_delta,
                            "max_abs_chroma_delta": tone_apply.diagnostics.noise_reduction_max_abs_chroma_delta,
                            "mean_abs_luma_delta": tone_apply.diagnostics.noise_reduction_mean_abs_luma_delta,
                            "max_abs_luma_delta": tone_apply.diagnostics.noise_reduction_max_abs_luma_delta,
                        }
                    }),
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
                    serde_json::json!(tone_output_evidence.render_luminance_range_p05_p95),
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
            if tone_output_evidence.confidence <= f64::EPSILON {
                phase.warnings.push(format!(
                    "rendered tone output requires review: {}",
                    tone_output_evidence.reason
                ));
            }
            if grain_detail.review_required {
                phase.warnings.push(format!(
                    "optional grain reduction requires structured-detail review: {}",
                    grain_detail
                        .review_reason
                        .as_deref()
                        .unwrap_or(&grain_detail.reason)
                ));
            } else if tone_apply.diagnostics.noise_reduction_enabled
                && !grain_detail.decision_supported
            {
                phase.warnings.push(format!(
                    "optional grain reduction had no supported coherent-edge population for retention validation: {}",
                    grain_detail.reason
                ));
            }
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

    let save_start = Instant::now()
        .checked_sub(prewritten_master_duration)
        .unwrap_or_else(Instant::now);
    let previous_output_modified_at = file_modified_at(&output_path);
    let overwrote_existing_output = previous_output_modified_at.is_some();
    let master_path = cli
        .should_write_master()
        .then(|| cli.output_dir.join("master_scene_referred.tiff"));
    let review_srgb_path = cli
        .should_write_review_proof()
        .then(|| cli.output_dir.join("review_srgb.png"));
    let artifact_width = u32::try_from(tonemapped.shape()[1]).map_err(|_| {
        format!(
            "render width {} exceeds the TIFF/PNG u32 limit",
            tonemapped.shape()[1]
        )
    })?;
    let artifact_height = u32::try_from(tonemapped.shape()[0]).map_err(|_| {
        format!(
            "render height {} exceeds the TIFF/PNG u32 limit",
            tonemapped.shape()[0]
        )
    })?;
    let artifact_write_buffers = tiff_io::artifact_write_buffer_diagnostics(
        artifact_width,
        artifact_height,
        master_path.is_some(),
        review_srgb_path.is_some(),
    );
    let mut master_artifact_diagnostics = prewritten_master_diagnostics;
    let mut master_write_duration = prewritten_master_duration;
    if let Some(path) = master_path.as_ref().filter(|_| !master_prewritten) {
        let master_write_start = Instant::now();
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
        master_write_duration = master_write_start.elapsed();
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
    let mut review_srgb_gamut_mapping_diagnostics = serde_json::Value::Null;
    if let Some(path) = &review_srgb_path {
        match tiff_io::save_srgb_png_from_linear_prophoto(&tonemapped, path) {
            Ok(diagnostics) => {
                review_srgb_gamut_mapping_diagnostics = serde_json::json!({
                    "space": tiff_io::SRGB_REVIEW_GAMUT_MAPPING_SPACE,
                    "policy": "preserve D50 CIELAB lightness and hue; binary-search chroma toward the neutral axis until the D65 sRGB target is in gamut",
                    "width": diagnostics.width,
                    "height": diagnostics.height,
                    "pixel_count": diagnostics.pixel_count,
                    "nonfinite_input_pixel_count": diagnostics.nonfinite_input_pixel_count,
                    "gamut_mapped_pixel_count": diagnostics.gamut_mapped_pixel_count,
                    "gamut_mapped_ratio": diagnostics.gamut_mapped_ratio,
                    "mapped_mean_chroma_scale": diagnostics.mapped_mean_chroma_scale,
                    "mapped_min_chroma_scale": diagnostics.mapped_min_chroma_scale,
                    "post_map_out_of_gamut_pixel_count": diagnostics.post_map_out_of_gamut_pixel_count,
                });
            }
            Err(err) => {
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
    let artifact_hashes = (|| -> Result<(String, Option<String>, Option<String>), String> {
        let output_sha256 = saved_artifact_sha256(&output_path, "finished render")?;
        let master_sha256 = master_path
            .as_deref()
            .map(|path| saved_artifact_sha256(path, "scene-referred master"))
            .transpose()?;
        let review_srgb_sha256 = review_srgb_path
            .as_deref()
            .map(|path| saved_artifact_sha256(path, "sRGB review proof"))
            .transpose()?;
        Ok((output_sha256, master_sha256, review_srgb_sha256))
    })();
    let (output_sha256, master_sha256, review_srgb_sha256) = match artifact_hashes {
        Ok(hashes) => hashes,
        Err(message) => {
            record_failure(
                &mut report,
                &cli.output_dir,
                "save",
                message.clone(),
                save_start,
                options.write_partial_report,
            )?;
            return Err(std::io::Error::other(message).into());
        }
    };
    let stale_render_artifacts =
        stale_render_artifacts_json(&cli.output_dir, &output_path, cache.run_started_at);
    record_phase(
        &mut report,
        &cli.output_dir,
        {
            let color_trust_state = cache.tone_color_protection.color_trust_state();
            let review_inputs = RenderReviewInputs {
                base_confidence: cache.base_confidence,
                color_trust_state,
                geometry: ReviewRequirement {
                    required: cache.geometry_review_required,
                    reason: &cache.geometry_review_reason,
                },
                input_mode: ReviewRequirement {
                    required: cache.input_mode_review_required,
                    reason: &cache.input_mode_review_reason,
                },
                negative_response: ReviewRequirement {
                    required: cache.negative_response_review_required,
                    reason: &cache.negative_response_review_reason,
                },
                technical_white_balance: ReviewRequirement {
                    required: cache.technical_white_balance_review_required,
                    reason: &cache.technical_white_balance_review_reason,
                },
                tone_output: ReviewRequirement {
                    required: tone_output_evidence.confidence <= f64::EPSILON,
                    reason: &tone_output_evidence.reason,
                },
                grain_detail: ReviewRequirement {
                    required: tone_apply
                        .diagnostics
                        .noise_reduction_detail_retention
                        .review_required,
                    reason: tone_apply
                        .diagnostics
                        .noise_reduction_detail_retention
                        .review_reason
                        .as_deref()
                        .unwrap_or(
                            &tone_apply
                                .diagnostics
                                .noise_reduction_detail_retention
                                .reason,
                        ),
                },
            };
            let review_status = render_review_status(&review_inputs);
            let review_reason = render_review_reason(&review_inputs);
            let delivery_confidence = if review_status == "reviewable" {
                1.0
            } else {
                0.0
            };
            let mut phase = PhaseReport::ok(
                "save",
                delivery_confidence,
                serde_json::json!({
                    "output_path": output_path.to_string_lossy(),
                    "output_shape": [tonemapped.shape()[0], tonemapped.shape()[1]],
                    "output_width": tonemapped.shape()[1],
                    "output_height": tonemapped.shape()[0],
                    "output_modified_at": file_modified_at_string(&output_path),
                    "output_modified_at_unix_ms": file_modified_at_unix_ms(&output_path),
                    "output_file_size_bytes": file_size_bytes(&output_path),
                    "output_color_space": "linear_prophoto_rgb_d50",
                    "output_encoding": {
                        "container": "tiff",
                        "channels": "RGB",
                        "sample_format": "UINT",
                        "bits_per_sample": [16, 16, 16]
                    },
                    "artifact_write_buffer_policy": {
                        "strategy": artifact_write_buffers.strategy,
                        "full_frame_conversion_buffers": artifact_write_buffers.full_frame_conversion_buffers,
                        "tiff_strip_target_bytes": artifact_write_buffers.tiff_strip_target_bytes,
                        "primary_tiff_conversion_buffer_bytes": artifact_write_buffers.primary_tiff_conversion_buffer_bytes,
                        "master_tiff_conversion_buffer_bytes": artifact_write_buffers.master_tiff_conversion_buffer_bytes,
                        "review_png_row_buffer_bytes": artifact_write_buffers.review_png_row_buffer_bytes,
                        "review_png_stream_chunk_bytes": artifact_write_buffers.review_png_stream_chunk_bytes,
                        "peak_declared_buffer_bytes": artifact_write_buffers.peak_declared_buffer_bytes,
                    },
                    "master_write_timing": if master_path.is_none() {
                        "not_requested"
                    } else if master_prewritten {
                        "before_owned_batch_tonemap"
                    } else {
                        "during_interactive_save"
                    },
                    "master_write_duration_ms": master_write_duration.as_millis() as u64,
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
                    "color_trust_state": color_trust_state,
                    "geometry_review_required": cache.geometry_review_required,
                    "geometry_review_reason": cache.geometry_review_reason,
                    "input_mode_review_required": cache.input_mode_review_required,
                    "input_mode_review_reason": cache.input_mode_review_reason,
                    "negative_response_review_required": cache.negative_response_review_required,
                    "negative_response_review_reason": cache.negative_response_review_reason,
                    "technical_white_balance_review_required": cache.technical_white_balance_review_required,
                    "technical_white_balance_review_reason": cache.technical_white_balance_review_reason,
                    "tone_output_review_required": tone_output_evidence.confidence <= f64::EPSILON,
                    "tone_output_review_reason": tone_output_evidence.reason.clone(),
                    "tone_output_confidence_status": tone_output_evidence.status,
                    "tone_output_evidence_confidence": tone_output_evidence.confidence,
                    "grain_detail_review_required": tone_apply.diagnostics.noise_reduction_detail_retention.review_required,
                    "grain_detail_review_reason": tone_apply.diagnostics.noise_reduction_detail_retention.review_reason.clone(),
                    "render_review_status": review_status,
                    "render_reviewable": review_status == "reviewable",
                    "render_review_reason": review_reason.clone(),
                    "delivery_confidence_status": if review_status == "reviewable" { "reviewable_delivery" } else { "diagnostic_delivery_requires_review" },
                    "review_evidence_evaluated": true,
                    "confidence_basis": "final_render_review_gate",
                    "confidence_definition": "confidence that the saved render is reviewable for delivery, distinct from successful artifact file I/O",
                }),
            );
            if let serde_json::Value::Object(metrics) = &mut phase.metrics {
                metrics.insert(
                    "output_sha256".to_string(),
                    serde_json::json!(output_sha256),
                );
                metrics.insert(
                    "artifact_sha256_policy".to_string(),
                    serde_json::json!({
                        "algorithm": "sha256",
                        "source": "reopened_saved_file_bytes",
                        "buffer_bytes": FILE_SHA256_BUFFER_BYTES,
                        "overlaps_artifact_conversion_buffers": false,
                        "peak_sequential_delivery_buffer_bytes": artifact_write_buffers
                            .peak_declared_buffer_bytes
                            .max(FILE_SHA256_BUFFER_BYTES),
                        "required_for_reviewable_delivery": true
                    }),
                );
                metrics.insert(
                    "artifact_commit_policy".to_string(),
                    serde_json::json!({
                        "strategy": atomic_file::ATOMIC_COMMIT_STRATEGY,
                        "same_directory_staging": true,
                        "per_file_atomic_replace": true,
                        "cross_artifact_transaction": false,
                        "destination_visible_only_after_successful_encode": true,
                        "failed_encode_preserves_existing_destination": true,
                        "file_contents_fsync_before_commit": false,
                        "directory_fsync_after_commit": false,
                        "temporary_disk_overhead": "up_to_new_artifact_size_per_sequential_write",
                        "applies_to": [
                            "primary_tiff",
                            "requested_scene_master",
                            "requested_srgb_proof",
                            "report_json"
                        ]
                    }),
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
                    "master_scene_referred_requested".to_string(),
                    serde_json::json!(cli.should_write_master()),
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
                    "master_scene_referred_sha256".to_string(),
                    serde_json::json!(master_sha256),
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
                    "review_srgb_requested".to_string(),
                    serde_json::json!(cli.should_write_review_proof()),
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
                    "review_srgb_sha256".to_string(),
                    serde_json::json!(review_srgb_sha256),
                );
                metrics.insert(
                    "review_srgb_color_space".to_string(),
                    serde_json::json!(review_srgb_path.as_ref().map(|_| "srgb_display_png")),
                );
                metrics.insert(
                    "review_srgb_encoding".to_string(),
                    review_srgb_path
                        .as_ref()
                        .map(|_| {
                            serde_json::json!({
                                "container": "png",
                                "channels": "RGB",
                                "sample_format": "UINT",
                                "bits_per_sample": [8, 8, 8]
                            })
                        })
                        .unwrap_or(serde_json::Value::Null),
                );
                metrics.insert(
                    "review_srgb_icc_profile".to_string(),
                    review_srgb_path
                        .as_ref()
                        .map(|_| {
                            serde_json::json!({
                                "embedded": true,
                                "description": tiff_io::SRGB_ICC_DESCRIPTION,
                                "transfer_function": "srgb",
                                "whitepoint": "D65"
                            })
                        })
                        .unwrap_or(serde_json::Value::Null),
                );
                metrics.insert(
                    "review_srgb_gamut_mapping".to_string(),
                    review_srgb_gamut_mapping_diagnostics.clone(),
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
            if review_status == "blocked_geometry_review" {
                phase.warnings.push(format!(
                    "render is not geometrically complete: {review_reason}"
                ));
            } else if review_status == "blocked_low_base_confidence" {
                phase.warnings.push(format!(
                    "render is not reviewable: {} (base confidence {:.3}); provide a real rebate/roll-base reference or inspect debug phase artifacts before judging inversion",
                    review_reason, cache.base_confidence
                ));
            } else if review_status == "caution_low_base_confidence" {
                phase.warnings.push(format!(
                    "render needs cautious review: {} (base confidence {:.3})",
                    review_reason, cache.base_confidence
                ));
            } else if review_status == "review_required_negative_response" {
                phase.warnings.push(format!(
                    "render requires negative-response review: {review_reason}"
                ));
            } else if review_status == "review_required_color" {
                phase.warnings.push(format!(
                    "render requires color review: {} (color trust state {})",
                    review_reason, color_trust_state
                ));
            } else if review_status == "review_required_white_balance" {
                phase.warnings.push(format!(
                    "render requires technical white-balance review: {review_reason}"
                ));
            } else if review_status == "review_required_tone_output" {
                phase.warnings.push(format!(
                    "render requires tone-output review: {review_reason}"
                ));
            } else if review_status == "review_required_grain_detail" {
                phase.warnings.push(format!(
                    "render requires optional grain-detail review: {review_reason}"
                ));
            } else if review_status == "caution_limited_color" {
                phase.warnings.push(format!(
                    "render needs cautious color review: {} (color trust state {})",
                    review_reason, color_trust_state
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
    use crate::constants::D50_WHITE;

    #[test]
    fn rendered_tone_evidence_distinguishes_supported_collapsed_clipped_and_flat_outputs() {
        let supported = assess_tone_output_evidence(
            [0.05, 0.40, 0.85],
            [0.08, 0.45, 0.88],
            [0.10, 0.46, 0.82],
            [0.002, 0.003, 0.001],
            [0.001, 0.002, 0.001],
            1_000,
        );
        assert_eq!(supported.confidence, 1.0);
        assert!(supported.evaluated);
        assert_eq!(supported.status, "supported_render_tonal_distribution");

        let collapsed = assess_tone_output_evidence(
            [0.05, 0.40, 0.85],
            [0.08, 0.45, 0.88],
            [0.4000, 0.4005, 0.4010],
            [0.0; 3],
            [0.0; 3],
            1_000,
        );
        assert_eq!(collapsed.confidence, 0.0);
        assert!(collapsed.evaluated);
        assert_eq!(
            collapsed.status,
            "review_required_collapsed_render_luminance_range"
        );

        let clipped = assess_tone_output_evidence(
            [0.05, 0.40, 0.85],
            [0.08, 0.45, 0.88],
            [0.10, 0.46, 0.82],
            [0.51, 0.0, 0.0],
            [0.0; 3],
            1_000,
        );
        assert_eq!(clipped.confidence, 0.0);
        assert!(clipped.evaluated);
        assert_eq!(
            clipped.status,
            "review_required_catastrophic_post_tone_clipping"
        );

        let flat =
            assess_tone_output_evidence([0.20; 3], [0.20; 3], [0.20; 3], [0.0; 3], [0.0; 3], 1_000);
        assert_eq!(flat.confidence, 0.0);
        assert!(flat.evaluated);
        assert_eq!(
            flat.status,
            "review_required_insufficient_scene_tonal_range"
        );

        let invalid = assess_tone_output_evidence(
            [0.05, 0.40, 0.85],
            [0.08, 0.45, 0.88],
            [0.10, 0.46, 0.82],
            [0.0; 3],
            [0.0; 3],
            0,
        );
        assert_eq!(invalid.confidence, 0.0);
        assert!(!invalid.evaluated);
        assert_eq!(
            invalid.status,
            "review_required_invalid_render_tone_diagnostics"
        );
    }

    #[test]
    fn stitch_detail_review_evidence_propagates_from_pair_and_sequence_reports() {
        let pair = PhaseReport::ok(
            "stitch",
            0.9,
            serde_json::json!({
                "seam_blend": {
                    "review_required": true,
                    "review_reason": "repeated detail-energy imbalance"
                }
            }),
        );
        assert_eq!(
            stitch_quality_review_reasons(&pair),
            vec!["repeated detail-energy imbalance"]
        );

        let sequence = PhaseReport::ok(
            "stitch",
            0.9,
            serde_json::json!({
                "seam_quality_review_required": true,
                "seam_quality_review_reasons": [
                    "sequence stitch step 1, component 3: repeated detail-energy imbalance"
                ],
                "pair_merges": [{
                    "step": 1,
                    "pair_report": {
                        "metrics": {
                            "seam_blend": {
                                "review_required": true,
                                "review_reason": "repeated detail-energy imbalance"
                            }
                        }
                    }
                }]
            }),
        );
        let reasons = stitch_quality_review_reasons(&sequence);
        assert_eq!(reasons.len(), 2, "{reasons:?}");
        assert!(reasons.iter().any(|reason| reason.contains("component 3")));
        assert!(reasons
            .iter()
            .any(|reason| reason.contains("sequence stitch step 1:")));

        let review = RenderReviewInputs {
            base_confidence: 1.0,
            color_trust_state: "trusted",
            geometry: ReviewRequirement {
                required: !reasons.is_empty(),
                reason: &reasons[0],
            },
            input_mode: ReviewRequirement {
                required: false,
                reason: "accepted",
            },
            negative_response: ReviewRequirement {
                required: false,
                reason: "not applicable",
            },
            technical_white_balance: ReviewRequirement {
                required: false,
                reason: "accepted",
            },
            tone_output: ReviewRequirement {
                required: false,
                reason: "accepted",
            },
            grain_detail: ReviewRequirement {
                required: false,
                reason: "not requested",
            },
        };
        assert_eq!(render_review_status(&review), "blocked_geometry_review");
        assert!(render_review_reason(&review).contains("detail-energy"));

        let grain_review = RenderReviewInputs {
            base_confidence: 1.0,
            color_trust_state: "trusted",
            geometry: ReviewRequirement {
                required: false,
                reason: "accepted",
            },
            input_mode: ReviewRequirement {
                required: false,
                reason: "accepted",
            },
            negative_response: ReviewRequirement {
                required: false,
                reason: "accepted",
            },
            technical_white_balance: ReviewRequirement {
                required: false,
                reason: "accepted",
            },
            tone_output: ReviewRequirement {
                required: false,
                reason: "accepted",
            },
            grain_detail: ReviewRequirement {
                required: true,
                reason: "coherent opponent-color detail retention fell below threshold",
            },
        };
        assert_eq!(
            render_review_status(&grain_review),
            "review_required_grain_detail"
        );
        assert!(render_review_reason(&grain_review).contains("opponent-color"));

        let tone_review = RenderReviewInputs {
            tone_output: ReviewRequirement {
                required: true,
                reason: "rendered luminance range collapsed",
            },
            grain_detail: ReviewRequirement {
                required: false,
                reason: "not requested",
            },
            ..grain_review
        };
        assert_eq!(
            render_review_status(&tone_review),
            "review_required_tone_output"
        );
        assert!(render_review_reason(&tone_review).contains("luminance range"));
    }

    fn nonlinear_report_target_patches(
        count: usize,
        prefix: &str,
        seed_offset: usize,
    ) -> Vec<color_calibration::TargetPatch> {
        let coefficients = [
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
                let code = |multiplier: usize, add: usize| {
                    ((seed * multiplier + add) % 997) as f64 / 996.0
                };
                let source_rgb = [
                    0.025 + 0.95 * code(173, 31),
                    0.025 + 0.95 * code(379, 97),
                    0.025 + 0.95 * code(613, 211),
                ];
                let basis = color_calibration::root_polynomial_basis_values(2, source_rgb)
                    .expect("degree-two basis");
                let reference_xyz = std::array::from_fn(|channel| {
                    basis
                        .iter()
                        .zip(coefficients)
                        .map(|(term, coefficient)| term * coefficient[channel])
                        .sum()
                });
                color_calibration::TargetPatch {
                    patch_id: Some(format!("{prefix}-{index:03}")),
                    scanner_xy: None,
                    source_rgb,
                    reference_xyz,
                }
            })
            .collect()
    }

    fn nonlinear_report_profile() -> (
        color_calibration::CalibrationProfile,
        Vec<color_calibration::TargetPatch>,
    ) {
        let training = nonlinear_report_target_patches(48, "report-train", 0);
        let held_out = nonlinear_report_target_patches(36, "report-held", 409);
        let matrix = color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
            &training,
            &held_out,
            "report_matrix_baseline",
            Some(D50_WHITE),
            None,
        )
        .expect("report matrix fit");
        let model = color_calibration::fit_root_polynomial_color_model_from_disjoint_patches(
            "report-nonlinear",
            &training,
            &held_out,
            &matrix.matrix,
        )
        .expect("report nonlinear fit")
        .selected_model
        .expect("report nonlinear model");
        let raw = serde_json::json!({
            "schema_version": 1,
            "profile_id": "report-nonlinear",
            "source_space": { "name": "synthetic scanner RGB", "encoding": "linear" },
            "scanner": { "make": "Synthetic", "model": "Pipeline unit test" },
            "film": { "stock": "Synthetic" },
            "target": { "type": "synthetic target", "illuminant": "D50" },
            "reference": { "dataset": "synthetic" },
            "whitepoint": D50_WHITE,
            "work_to_xyz": matrix.matrix,
            "confidence": matrix.confidence
        });
        let mut profile = color_calibration::parse_profile_json(&raw.to_string(), None)
            .profile
            .expect("base report profile");
        profile.schema_version = 2;
        profile.work_to_xyz = matrix.matrix;
        profile.whitepoint = D50_WHITE;
        profile.confidence = matrix.confidence;
        profile.matrix_condition_number = matrix.matrix_condition_number;
        profile.fit = Some(matrix.fit);
        profile.target_patches = held_out.clone();
        profile.application_mode = color_calibration::CalibrationApplicationMode::DirectProfile;
        profile.color_model = Some(model);
        profile.color_model_post_xyz = None;
        (profile, held_out)
    }

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

    #[test]
    fn nonlinear_model_support_is_preserved_in_pipeline_metrics_for_selection_and_fallback() {
        let (profile, held_out) = nonlinear_report_profile();
        let mut in_domain = Array3::<f64>::zeros((16, 16, 3));
        for y in 0..16 {
            for x in 0..16 {
                let rgb = held_out[(y * 16 + x) % held_out.len()].source_rgb;
                for channel in 0..3 {
                    in_domain[[y, x, channel]] = rgb[channel];
                }
            }
        }
        let selected = colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &in_domain,
            Some(&profile),
            colorspace::ColorMode::Calibrated,
        )
        .expect("in-domain nonlinear mapping");
        let selected_metrics = colorspace_candidate_metrics_json(&selected.diagnostics);
        assert_eq!(
            selected_metrics["mapping_strategy"],
            "calibrated_root_polynomial"
        );
        assert_eq!(
            selected_metrics["nonlinear_color_model"]["support_status"],
            "accepted"
        );
        assert_eq!(
            selected_metrics["candidate_comparison"]["nonlinear_color_model"]["model_id"],
            profile.color_model.as_ref().unwrap().model_id
        );
        assert_eq!(
            selected_metrics["color_processing_substeps"]["nonlinear_model_support"]
                ["support_status"],
            "accepted"
        );

        let mut out_of_domain = Array3::<f64>::zeros((16, 16, 3));
        for y in 0..16 {
            for x in 0..16 {
                out_of_domain[[y, x, 0]] = 1.0;
                out_of_domain[[y, x, 1]] = 0.001;
                out_of_domain[[y, x, 2]] = 0.001;
            }
        }
        let fallback = colorspace::map_to_prophoto_d50_with_color_mode_and_calibration_diagnostics(
            &out_of_domain,
            Some(&profile),
            colorspace::ColorMode::Calibrated,
        )
        .expect("out-of-domain matrix fallback");
        let fallback_metrics = colorspace_candidate_metrics_json(&fallback.diagnostics);
        assert_eq!(fallback_metrics["mapping_strategy"], "calibrated_profile");
        assert_eq!(
            fallback_metrics["nonlinear_color_model"]["support_status"],
            "rejected"
        );
        assert!(fallback_metrics["candidate_quality_scores"]
            .as_array()
            .unwrap()
            .iter()
            .any(|candidate| {
                candidate["candidate"] == "calibrated_root_polynomial"
                    && candidate["rejected"] == true
            }));
    }
}

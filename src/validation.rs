use crate::report::{PhaseReport, PipelineReport};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationSummary {
    pub fixture: String,
    pub stitch: StitchValidationSummary,
    pub base_density: BaseDensityValidationSummary,
    pub colorspace: ColorspaceValidationSummary,
    pub tone: ToneValidationSummary,
    pub warnings: Vec<PhaseWarnings>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StitchValidationSummary {
    pub decision: Option<String>,
    pub confidence: Option<f64>,
    pub chosen_hypothesis: Option<String>,
    pub rejection_reason: Option<String>,
    pub search_selection_reason: Option<String>,
    pub evaluated_candidate_count: Option<usize>,
    pub max_overlap_considered: Option<usize>,
    pub top_candidates: Vec<StitchCandidateSummary>,
    pub evidence_score: Option<f64>,
    pub prior_weight: Option<f64>,
    pub overlap_support_score: Option<f64>,
    pub vertical_offset_plausibility_score: Option<f64>,
    pub local_consistency_score: Option<f64>,
    pub plausibility_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StitchCandidateSummary {
    pub rank: Option<usize>,
    pub overlap_width: Option<usize>,
    pub vertical_offset: Option<i64>,
    pub correspondence_score: Option<f64>,
    pub prior_weight: Option<f64>,
    pub search_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaseDensityValidationSummary {
    pub base_confidence: Option<f64>,
    pub raw_base_confidence: Option<f64>,
    pub base_estimate_source: Option<String>,
    pub density_confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColorspaceValidationSummary {
    pub render_input_source: Option<String>,
    pub render_input_reason: Option<String>,
    pub density_candidate_evaluated: Option<bool>,
    pub exposure_scale: Option<f64>,
    pub mapping_strategy: Option<String>,
    pub regularization_lambda: Option<f64>,
    pub channel_anchor_counts: Option<Vec<usize>>,
    pub channel_anchor_min_count: Option<usize>,
    pub channel_anchor_low_support: Option<Vec<bool>>,
    pub weak_anchor_fallback_used: Option<bool>,
    pub gamut_fallback_used: Option<bool>,
    pub image_matrix_pre_scale_clipped_low_ratio: Option<Vec<f64>>,
    pub image_matrix_pre_scale_clipped_high_ratio: Option<Vec<f64>>,
    pub image_matrix_exposure_scale: Option<f64>,
    pub post_scale_clipped_high_ratio: Option<Vec<f64>>,
    pub post_scale_clipped_low_ratio: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToneValidationSummary {
    pub shadow_saturation_median: Option<f64>,
    pub shadow_saturation_p95: Option<f64>,
    pub midtone_saturation_median: Option<f64>,
    pub midtone_saturation_p95: Option<f64>,
    pub midtone_luminance_percentiles: Option<Vec<f64>>,
    pub bright_neutral_saturation_median: Option<f64>,
    pub bright_neutral_saturation_p95: Option<f64>,
    pub bright_saturated_saturation_median: Option<f64>,
    pub bright_saturated_saturation_p95: Option<f64>,
    pub post_chroma_compression_clipped_high_ratio: Option<Vec<f64>>,
    pub post_chroma_compression_clipped_low_ratio: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhaseWarnings {
    pub phase: String,
    pub warnings: Vec<String>,
}

pub fn summarize_report(fixture: impl Into<String>, report: &PipelineReport) -> ValidationSummary {
    let stitch_phase = phase(report, "stitch");
    let working_phase = phase(report, "working_image_select");
    let density_phase = phase(report, "density_inversion");
    let colorspace_phase = phase(report, "colorspace_mapping");
    let tone_phase = phase(report, "tone_mapping");

    ValidationSummary {
        fixture: fixture.into(),
        stitch: summarize_stitch(stitch_phase),
        base_density: BaseDensityValidationSummary {
            base_confidence: working_phase.map(|p| p.confidence),
            raw_base_confidence: working_phase.and_then(|p| f64_metric(p, "raw_base_confidence")),
            base_estimate_source: working_phase
                .and_then(|p| string_metric(p, "base_estimate_source")),
            density_confidence: density_phase.map(|p| p.confidence),
        },
        colorspace: colorspace_phase
            .map(summarize_colorspace)
            .unwrap_or_else(empty_colorspace_summary),
        tone: tone_phase
            .map(summarize_tone)
            .unwrap_or_else(empty_tone_summary),
        warnings: report
            .phases
            .iter()
            .filter(|phase| !phase.warnings.is_empty())
            .map(|phase| PhaseWarnings {
                phase: phase.name.clone(),
                warnings: phase.warnings.clone(),
            })
            .collect(),
    }
}

pub fn summary_to_markdown(summary: &ValidationSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Validation Summary: {}\n\n", summary.fixture));
    out.push_str("| Area | Field | Value |\n");
    out.push_str("|-|-|-|\n");
    push_row(&mut out, "stitch", "decision", &summary.stitch.decision);
    push_row(
        &mut out,
        "stitch",
        "confidence",
        &format_f64(summary.stitch.confidence),
    );
    push_row(
        &mut out,
        "stitch",
        "chosen_hypothesis",
        &summary.stitch.chosen_hypothesis,
    );
    push_row(
        &mut out,
        "stitch",
        "selection_reason",
        &summary.stitch.search_selection_reason,
    );
    push_row(
        &mut out,
        "density",
        "base_confidence",
        &format_f64(summary.base_density.base_confidence),
    );
    push_row(
        &mut out,
        "density",
        "density_confidence",
        &format_f64(summary.base_density.density_confidence),
    );
    push_row(
        &mut out,
        "colorspace",
        "render_input_source",
        &summary.colorspace.render_input_source,
    );
    push_row(
        &mut out,
        "colorspace",
        "mapping_strategy",
        &summary.colorspace.mapping_strategy,
    );
    push_row(
        &mut out,
        "colorspace",
        "exposure_scale",
        &format_f64(summary.colorspace.exposure_scale),
    );
    push_row(
        &mut out,
        "colorspace",
        "channel_anchor_low_support",
        &format_bool_vec(&summary.colorspace.channel_anchor_low_support),
    );
    push_row(
        &mut out,
        "tone",
        "shadow_saturation_p95",
        &format_f64(summary.tone.shadow_saturation_p95),
    );
    push_row(
        &mut out,
        "tone",
        "midtone_saturation_p95",
        &format_f64(summary.tone.midtone_saturation_p95),
    );
    push_row(
        &mut out,
        "tone",
        "bright_neutral_saturation_p95",
        &format_f64(summary.tone.bright_neutral_saturation_p95),
    );
    push_row(
        &mut out,
        "tone",
        "bright_saturated_saturation_p95",
        &format_f64(summary.tone.bright_saturated_saturation_p95),
    );

    if !summary.warnings.is_empty() {
        out.push_str("\n## Warnings\n\n");
        for phase in &summary.warnings {
            for warning in &phase.warnings {
                out.push_str(&format!("- `{}`: {}\n", phase.phase, warning));
            }
        }
    }

    out
}

fn summarize_stitch(phase: Option<&PhaseReport>) -> StitchValidationSummary {
    let Some(phase) = phase else {
        return StitchValidationSummary {
            decision: None,
            confidence: None,
            chosen_hypothesis: None,
            rejection_reason: None,
            search_selection_reason: None,
            evaluated_candidate_count: None,
            max_overlap_considered: None,
            top_candidates: Vec::new(),
            evidence_score: None,
            prior_weight: None,
            overlap_support_score: None,
            vertical_offset_plausibility_score: None,
            local_consistency_score: None,
            plausibility_score: None,
        };
    };

    let selected = selected_hypothesis(phase);
    let selected_search = selected.and_then(|hyp| hyp.get("search"));
    let selected_validation = selected.and_then(|hyp| hyp.get("validation"));
    let top_candidates = selected_search
        .and_then(|search| search.get("top_candidates"))
        .and_then(|value| value.as_array())
        .map(|candidates| {
            candidates
                .iter()
                .take(3)
                .map(|candidate| StitchCandidateSummary {
                    rank: usize_value(candidate.get("rank")),
                    overlap_width: usize_value(candidate.get("overlap_width")),
                    vertical_offset: i64_value(candidate.get("vertical_offset")),
                    correspondence_score: f64_value(candidate.get("correspondence_score")),
                    prior_weight: f64_value(candidate.get("prior_weight")),
                    search_score: f64_value(candidate.get("search_score")),
                })
                .collect()
        })
        .unwrap_or_default();

    StitchValidationSummary {
        decision: string_metric(phase, "decision"),
        confidence: Some(phase.confidence),
        chosen_hypothesis: string_metric(phase, "chosen_hypothesis"),
        rejection_reason: string_metric(phase, "rejection_reason"),
        search_selection_reason: selected_search
            .and_then(|search| string_value(search.get("selection_reason"))),
        evaluated_candidate_count: selected_search
            .and_then(|search| search.get("evaluated_candidates"))
            .and_then(|value| value.as_array())
            .map(Vec::len),
        max_overlap_considered: selected_search
            .and_then(|search| usize_value(search.get("max_overlap_considered"))),
        top_candidates,
        evidence_score: selected_validation.and_then(|v| f64_value(v.get("evidence_score"))),
        prior_weight: selected_validation.and_then(|v| f64_value(v.get("prior_weight"))),
        overlap_support_score: selected_validation
            .and_then(|v| f64_value(v.get("overlap_support_score"))),
        vertical_offset_plausibility_score: selected_validation
            .and_then(|v| f64_value(v.get("vertical_offset_plausibility_score"))),
        local_consistency_score: selected_validation
            .and_then(|v| f64_value(v.get("local_consistency_score"))),
        plausibility_score: selected_validation
            .and_then(|v| f64_value(v.get("plausibility_score"))),
    }
}

fn summarize_colorspace(phase: &PhaseReport) -> ColorspaceValidationSummary {
    ColorspaceValidationSummary {
        render_input_source: string_metric(phase, "render_input_source"),
        render_input_reason: string_metric(phase, "render_input_reason"),
        density_candidate_evaluated: bool_metric(phase, "direct_density_candidate_evaluated"),
        exposure_scale: f64_metric(phase, "exposure_scale"),
        mapping_strategy: string_metric(phase, "mapping_strategy"),
        regularization_lambda: f64_metric(phase, "regularization_lambda"),
        channel_anchor_counts: usize_vec_metric(phase, "channel_anchor_counts"),
        channel_anchor_min_count: usize_metric(phase, "channel_anchor_min_count"),
        channel_anchor_low_support: bool_vec_metric(phase, "channel_anchor_low_support"),
        weak_anchor_fallback_used: bool_metric(phase, "weak_anchor_fallback_used"),
        gamut_fallback_used: bool_metric(phase, "gamut_fallback_used"),
        image_matrix_pre_scale_clipped_low_ratio: f64_vec_metric(
            phase,
            "image_matrix_pre_scale_clipped_low_ratio",
        ),
        image_matrix_pre_scale_clipped_high_ratio: f64_vec_metric(
            phase,
            "image_matrix_pre_scale_clipped_high_ratio",
        ),
        image_matrix_exposure_scale: f64_metric(phase, "image_matrix_exposure_scale"),
        post_scale_clipped_high_ratio: f64_vec_metric(phase, "post_scale_clipped_high_ratio"),
        post_scale_clipped_low_ratio: f64_vec_metric(phase, "post_scale_clipped_low_ratio"),
    }
}

fn summarize_tone(phase: &PhaseReport) -> ToneValidationSummary {
    ToneValidationSummary {
        shadow_saturation_median: f64_metric(phase, "shadow_saturation_median"),
        shadow_saturation_p95: f64_metric(phase, "shadow_saturation_p95"),
        midtone_saturation_median: f64_metric(phase, "midtone_saturation_median"),
        midtone_saturation_p95: f64_metric(phase, "midtone_saturation_p95"),
        midtone_luminance_percentiles: f64_vec_metric(phase, "midtone_luminance_percentiles"),
        bright_neutral_saturation_median: f64_metric(phase, "bright_neutral_saturation_median"),
        bright_neutral_saturation_p95: f64_metric(phase, "bright_neutral_saturation_p95"),
        bright_saturated_saturation_median: f64_metric(phase, "bright_saturated_saturation_median"),
        bright_saturated_saturation_p95: f64_metric(phase, "bright_saturated_saturation_p95"),
        post_chroma_compression_clipped_high_ratio: f64_vec_metric(
            phase,
            "post_chroma_compression_clipped_high_ratio",
        ),
        post_chroma_compression_clipped_low_ratio: f64_vec_metric(
            phase,
            "post_chroma_compression_clipped_low_ratio",
        ),
    }
}

fn empty_colorspace_summary() -> ColorspaceValidationSummary {
    ColorspaceValidationSummary {
        render_input_source: None,
        render_input_reason: None,
        density_candidate_evaluated: None,
        exposure_scale: None,
        mapping_strategy: None,
        regularization_lambda: None,
        channel_anchor_counts: None,
        channel_anchor_min_count: None,
        channel_anchor_low_support: None,
        weak_anchor_fallback_used: None,
        gamut_fallback_used: None,
        image_matrix_pre_scale_clipped_low_ratio: None,
        image_matrix_pre_scale_clipped_high_ratio: None,
        image_matrix_exposure_scale: None,
        post_scale_clipped_high_ratio: None,
        post_scale_clipped_low_ratio: None,
    }
}

fn empty_tone_summary() -> ToneValidationSummary {
    ToneValidationSummary {
        shadow_saturation_median: None,
        shadow_saturation_p95: None,
        midtone_saturation_median: None,
        midtone_saturation_p95: None,
        midtone_luminance_percentiles: None,
        bright_neutral_saturation_median: None,
        bright_neutral_saturation_p95: None,
        bright_saturated_saturation_median: None,
        bright_saturated_saturation_p95: None,
        post_chroma_compression_clipped_high_ratio: None,
        post_chroma_compression_clipped_low_ratio: None,
    }
}

fn selected_hypothesis(phase: &PhaseReport) -> Option<&serde_json::Value> {
    let hypotheses = phase.metrics.get("hypotheses")?.as_array()?;
    if let Some(chosen) = string_metric(phase, "chosen_hypothesis") {
        if let Some(hypothesis) = hypotheses
            .iter()
            .find(|hypothesis| string_value(hypothesis.get("ordering")).as_deref() == Some(&chosen))
        {
            return Some(hypothesis);
        }
    }
    hypotheses
        .iter()
        .find(|hypothesis| {
            hypothesis
                .get("accepted")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        })
        .or_else(|| hypotheses.first())
}

fn phase<'a>(report: &'a PipelineReport, name: &str) -> Option<&'a PhaseReport> {
    report.phases.iter().find(|phase| phase.name == name)
}

fn string_metric(phase: &PhaseReport, key: &str) -> Option<String> {
    string_value(phase.metrics.get(key))
}

fn f64_metric(phase: &PhaseReport, key: &str) -> Option<f64> {
    f64_value(phase.metrics.get(key))
}

fn usize_metric(phase: &PhaseReport, key: &str) -> Option<usize> {
    usize_value(phase.metrics.get(key))
}

fn bool_metric(phase: &PhaseReport, key: &str) -> Option<bool> {
    phase.metrics.get(key).and_then(serde_json::Value::as_bool)
}

fn f64_vec_metric(phase: &PhaseReport, key: &str) -> Option<Vec<f64>> {
    phase.metrics.get(key).and_then(f64_vec_value)
}

fn usize_vec_metric(phase: &PhaseReport, key: &str) -> Option<Vec<usize>> {
    phase.metrics.get(key).and_then(usize_vec_value)
}

fn bool_vec_metric(phase: &PhaseReport, key: &str) -> Option<Vec<bool>> {
    phase.metrics.get(key).and_then(bool_vec_value)
}

fn string_value(value: Option<&serde_json::Value>) -> Option<String> {
    value
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string)
}

fn f64_value(value: Option<&serde_json::Value>) -> Option<f64> {
    value.and_then(serde_json::Value::as_f64)
}

fn i64_value(value: Option<&serde_json::Value>) -> Option<i64> {
    value.and_then(serde_json::Value::as_i64)
}

fn usize_value(value: Option<&serde_json::Value>) -> Option<usize> {
    value
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn f64_vec_value(value: &serde_json::Value) -> Option<Vec<f64>> {
    value.as_array().map(|values| {
        values
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .collect()
    })
}

fn usize_vec_value(value: &serde_json::Value) -> Option<Vec<usize>> {
    value.as_array().map(|values| {
        values
            .iter()
            .filter_map(|value| value.as_u64().and_then(|value| usize::try_from(value).ok()))
            .collect()
    })
}

fn bool_vec_value(value: &serde_json::Value) -> Option<Vec<bool>> {
    value.as_array().map(|values| {
        values
            .iter()
            .filter_map(serde_json::Value::as_bool)
            .collect()
    })
}

fn push_row(out: &mut String, area: &str, field: &str, value: &Option<String>) {
    out.push_str(&format!(
        "| {} | `{}` | {} |\n",
        area,
        field,
        value.as_deref().unwrap_or("")
    ));
}

fn format_f64(value: Option<f64>) -> Option<String> {
    value.map(|value| format!("{value:.6}"))
}

fn format_bool_vec(value: &Option<Vec<bool>>) -> Option<String> {
    value.as_ref().map(|values| {
        values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    })
}

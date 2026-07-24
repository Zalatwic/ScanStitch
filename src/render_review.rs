//! Hash-bound human review packages for final rendered film scans.
//!
//! Automated diagnostics can establish technical safety, but they cannot decide whether a crop
//! lost meaningful content, a seam is perceptible, or a render is photographically preferred.
//! This module generates deliberately unapproved review drafts and validates later human approval
//! against the exact report and delivery-artifact bytes that were inspected.

use crate::atomic_file;
use crate::report::PipelineReport;
use crate::validation::{final_delivery_evidence_is_reviewable, summarize_report_with_source};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

pub const RENDER_REVIEW_SCHEMA_VERSION: u32 = 2;
const LEGACY_RENDER_REVIEW_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RenderReviewArtifactBinding {
    pub kind: String,
    pub path: String,
    pub sha256: String,
    pub file_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RenderReviewInputBinding {
    pub component_index: usize,
    pub path: String,
    pub sha256: String,
    pub file_size_bytes: u64,
    pub decoded_pixel_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RenderReviewGrainControlBinding {
    pub purpose: String,
    pub source_report: String,
    pub source_report_sha256: String,
    pub technical_delivery_reviewable: bool,
    pub artifacts: Vec<RenderReviewArtifactBinding>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RenderReviewDecision {
    pub applicable: bool,
    pub approved: bool,
    pub notes: Option<String>,
}

impl RenderReviewDecision {
    fn draft(applicable: bool) -> Self {
        Self {
            applicable,
            approved: false,
            notes: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RenderReviewDecisions {
    pub orientation_and_retained_content: RenderReviewDecision,
    pub border_crop_and_no_lost_content: RenderReviewDecision,
    pub stitch_geometry_and_full_union: RenderReviewDecision,
    pub seam_tone_color_texture_sharpness: RenderReviewDecision,
    pub color_and_neutrality: RenderReviewDecision,
    pub tone_and_dynamic_range: RenderReviewDecision,
    pub grain_and_fine_detail: RenderReviewDecision,
    pub overall_visual_preference: RenderReviewDecision,
}

impl RenderReviewDecisions {
    fn draft(stitch_applicable: bool, grain_applicable: bool) -> Self {
        Self {
            orientation_and_retained_content: RenderReviewDecision::draft(true),
            border_crop_and_no_lost_content: RenderReviewDecision::draft(true),
            stitch_geometry_and_full_union: RenderReviewDecision::draft(stitch_applicable),
            seam_tone_color_texture_sharpness: RenderReviewDecision::draft(stitch_applicable),
            color_and_neutrality: RenderReviewDecision::draft(true),
            tone_and_dynamic_range: RenderReviewDecision::draft(true),
            grain_and_fine_detail: RenderReviewDecision::draft(grain_applicable),
            overall_visual_preference: RenderReviewDecision::draft(true),
        }
    }

    fn named(&self) -> [(&'static str, &RenderReviewDecision); 8] {
        [
            (
                "orientation_and_retained_content",
                &self.orientation_and_retained_content,
            ),
            (
                "border_crop_and_no_lost_content",
                &self.border_crop_and_no_lost_content,
            ),
            (
                "stitch_geometry_and_full_union",
                &self.stitch_geometry_and_full_union,
            ),
            (
                "seam_tone_color_texture_sharpness",
                &self.seam_tone_color_texture_sharpness,
            ),
            ("color_and_neutrality", &self.color_and_neutrality),
            ("tone_and_dynamic_range", &self.tone_and_dynamic_range),
            ("grain_and_fine_detail", &self.grain_and_fine_detail),
            ("overall_visual_preference", &self.overall_visual_preference),
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RenderReviewManifest {
    pub schema_version: u32,
    pub fixture: String,
    pub review_status: String,
    pub instructions: String,
    pub source_report: String,
    pub source_report_sha256: String,
    pub technical_delivery_reviewable: bool,
    pub input_component_count: usize,
    pub stitch_applicable: bool,
    pub grain_reduction_applicable: bool,
    #[serde(default)]
    pub grain_reduction_control: Option<RenderReviewGrainControlBinding>,
    pub inputs: Vec<RenderReviewInputBinding>,
    pub artifacts: Vec<RenderReviewArtifactBinding>,
    pub decisions: RenderReviewDecisions,
    pub reviewer: Option<String>,
    pub reviewed_at: Option<String>,
    pub review_notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RenderReviewInspection {
    pub status: String,
    pub approved: bool,
    pub schema_version: Option<u32>,
    pub fixture_matches: Option<bool>,
    pub source_report_sha256_matches: Option<bool>,
    pub technical_delivery_reviewable: Option<bool>,
    pub input_count: usize,
    pub all_input_hashes_match: Option<bool>,
    pub registry_inputs_match: Option<bool>,
    pub artifact_count: usize,
    pub all_artifact_hashes_match: Option<bool>,
    pub grain_control_present: bool,
    pub grain_control_report_sha256_matches: Option<bool>,
    pub grain_control_technical_delivery_reviewable: Option<bool>,
    pub grain_control_inputs_match: Option<bool>,
    pub grain_control_render_contract_matches: Option<bool>,
    pub grain_control_master_matches: Option<bool>,
    pub grain_control_artifact_count: usize,
    pub all_grain_control_artifact_hashes_match: Option<bool>,
    pub decision_count: usize,
    pub applicable_decision_count: usize,
    pub approved_applicable_decision_count: usize,
    pub issues: Vec<String>,
}

impl RenderReviewInspection {
    fn unreadable(status: &str, issue: String) -> Self {
        Self {
            status: status.to_string(),
            approved: false,
            schema_version: None,
            fixture_matches: None,
            source_report_sha256_matches: None,
            technical_delivery_reviewable: None,
            input_count: 0,
            all_input_hashes_match: None,
            registry_inputs_match: None,
            artifact_count: 0,
            all_artifact_hashes_match: None,
            grain_control_present: false,
            grain_control_report_sha256_matches: None,
            grain_control_technical_delivery_reviewable: None,
            grain_control_inputs_match: None,
            grain_control_render_contract_matches: None,
            grain_control_master_matches: None,
            grain_control_artifact_count: 0,
            all_grain_control_artifact_hashes_match: None,
            decision_count: 0,
            applicable_decision_count: 0,
            approved_applicable_decision_count: 0,
            issues: vec![issue],
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderReviewDraftOutput {
    pub manifest: RenderReviewManifest,
    pub json_path: PathBuf,
    pub markdown_path: PathBuf,
}

#[derive(Debug, Clone)]
struct ExpectedReportInput {
    component_index: usize,
    path: PathBuf,
    decoded_pixel_sha256: String,
}

pub fn write_render_review_draft(
    fixture: &str,
    report_path: &Path,
    output_dir: &Path,
) -> Result<RenderReviewDraftOutput, Box<dyn std::error::Error>> {
    write_render_review_draft_with_grain_control(fixture, report_path, None, output_dir)
}

pub fn write_render_review_draft_with_grain_control(
    fixture: &str,
    report_path: &Path,
    grain_control_report_path: Option<&Path>,
    output_dir: &Path,
) -> Result<RenderReviewDraftOutput, Box<dyn std::error::Error>> {
    let source_report = absolute_path(report_path)?;
    let report_bytes = std::fs::read(&source_report)?;
    let report = serde_json::from_slice::<PipelineReport>(&report_bytes)?;
    let summary = summarize_report_with_source(fixture, &report, Some(&source_report));
    if summary.render.review_srgb_requested != Some(true) {
        return Err(
            "render review requires a requested sRGB proof; rerun in perfect quality mode".into(),
        );
    }

    let expected_inputs = expected_report_inputs(&report, &summary)?;
    let input_component_count = expected_inputs.len();
    let stitch_applicable = input_component_count > 1;
    let grain_reduction_applicable = summary.render.noise_reduction_requested_enabled == Some(true);
    let mut inputs = Vec::with_capacity(expected_inputs.len());
    for input in expected_inputs {
        let (sha256, file_size_bytes) = hash_file_sha256(&input.path)?;
        inputs.push(RenderReviewInputBinding {
            component_index: input.component_index,
            path: absolute_path(&input.path)?.to_string_lossy().to_string(),
            sha256,
            file_size_bytes,
            decoded_pixel_sha256: input.decoded_pixel_sha256,
        });
    }
    let expected_artifacts = expected_delivery_artifacts(&summary)?;
    let mut artifacts = Vec::with_capacity(expected_artifacts.len());
    for (kind, path) in expected_artifacts {
        let (sha256, file_size_bytes) = hash_file_sha256(&path)?;
        artifacts.push(RenderReviewArtifactBinding {
            kind: kind.to_string(),
            path: absolute_path(&path)?.to_string_lossy().to_string(),
            sha256,
            file_size_bytes,
        });
    }

    let grain_reduction_control = match (grain_reduction_applicable, grain_control_report_path) {
        (true, Some(control_report_path)) => Some(build_grain_control_binding(
            fixture,
            &report,
            &summary,
            &inputs,
            &artifacts,
            control_report_path,
        )?),
        (true, None) => {
            return Err("grain-on render review requires --render-review-grain-control-report from a matched grain-off perfect render".into());
        }
        (false, Some(_)) => {
            return Err("a grain-off control report is valid only when the reviewed report requested grain reduction".into());
        }
        (false, None) => None,
    };

    let instructions = if grain_reduction_applicable {
        "Inspect the exact hash-bound sRGB proof at full resolution and the technical master where relevant. Compare grain and fine detail at 100% against the bound grain-off control proof; do not use an unbound or stale comparison. Approve only after checking every applicable decision. Change review_status to approved, add reviewer/reviewed_at/review_notes, add specific notes to every applicable decision, and set only applicable approved values true. This draft never approves its own output."
    } else {
        "Inspect the exact hash-bound sRGB proof at full resolution and the technical master where relevant. Approve only after checking every applicable decision. Change review_status to approved, add reviewer/reviewed_at/review_notes, add specific notes to every applicable decision, and set only applicable approved values true. This draft never approves its own output."
    };

    let manifest = RenderReviewManifest {
        schema_version: RENDER_REVIEW_SCHEMA_VERSION,
        fixture: fixture.to_string(),
        review_status: "requires_human_approval".to_string(),
        instructions: instructions.to_string(),
        source_report: source_report.to_string_lossy().to_string(),
        source_report_sha256: sha256_bytes(&report_bytes),
        technical_delivery_reviewable: final_delivery_evidence_is_reviewable(&summary.render),
        input_component_count,
        stitch_applicable,
        grain_reduction_applicable,
        grain_reduction_control,
        inputs,
        artifacts,
        decisions: RenderReviewDecisions::draft(stitch_applicable, grain_reduction_applicable),
        reviewer: None,
        reviewed_at: None,
        review_notes: None,
    };

    std::fs::create_dir_all(output_dir)?;
    let json_path = output_dir.join("render-review.json");
    let markdown_path = output_dir.join("render-review.md");
    let json = serde_json::to_string_pretty(&manifest)?;
    let markdown = render_review_to_markdown(&manifest);
    atomic_file::write_bytes(&json_path, json.as_bytes())?;
    atomic_file::write_bytes(&markdown_path, markdown.as_bytes())?;
    Ok(RenderReviewDraftOutput {
        manifest,
        json_path,
        markdown_path,
    })
}

fn build_grain_control_binding(
    fixture: &str,
    primary_report: &PipelineReport,
    primary_summary: &crate::validation::ValidationSummary,
    primary_inputs: &[RenderReviewInputBinding],
    primary_artifacts: &[RenderReviewArtifactBinding],
    control_report_path: &Path,
) -> Result<RenderReviewGrainControlBinding, Box<dyn std::error::Error>> {
    let source_report = absolute_path(control_report_path)?;
    let report_bytes = std::fs::read(&source_report)?;
    let report = serde_json::from_slice::<PipelineReport>(&report_bytes)?;
    let summary = summarize_report_with_source(fixture, &report, Some(&source_report));

    if summary.render.noise_reduction_requested_enabled != Some(false)
        || summary.tone.noise_reduction_enabled != Some(false)
    {
        return Err(
            "grain comparison control must explicitly report requested and effective reduction off"
                .into(),
        );
    }
    if !final_delivery_evidence_is_reviewable(&summary.render) {
        return Err("grain comparison control is not a technically reviewable delivery".into());
    }
    if summary.render.master_scene_referred_requested != Some(true)
        || primary_summary.render.master_scene_referred_requested != Some(true)
    {
        return Err(
            "grain comparison requires perfect-mode primary and control scene-referred masters"
                .into(),
        );
    }

    let control_inputs = expected_report_inputs(&report, &summary)?;
    if !grain_control_inputs_match(primary_inputs, &control_inputs) {
        return Err(
            "grain comparison control inputs or orientation-materialized pixels do not match"
                .into(),
        );
    }
    let render_contract_issues =
        grain_control_render_contract_issues(primary_report, primary_summary, &report, &summary);
    if !render_contract_issues.is_empty() {
        return Err(format!(
            "grain comparison control render contract differs: {}",
            render_contract_issues.join(",")
        )
        .into());
    }

    let mut artifacts = Vec::new();
    for (kind, path) in expected_delivery_artifacts(&summary)? {
        let (sha256, file_size_bytes) = hash_file_sha256(&path)?;
        artifacts.push(RenderReviewArtifactBinding {
            kind: kind.to_string(),
            path: absolute_path(&path)?.to_string_lossy().to_string(),
            sha256,
            file_size_bytes,
        });
    }
    let primary_master = artifact_binding(primary_artifacts, "master_scene_referred")
        .ok_or("primary scene-referred master binding missing")?;
    let control_master = artifact_binding(&artifacts, "master_scene_referred")
        .ok_or("grain control scene-referred master binding missing")?;
    if !primary_master
        .sha256
        .eq_ignore_ascii_case(&control_master.sha256)
    {
        return Err(
            "grain comparison control scene-referred master is not byte-identical to the primary"
                .into(),
        );
    }

    Ok(RenderReviewGrainControlBinding {
        purpose: "grain_reduction_off_control".to_string(),
        source_report: source_report.to_string_lossy().to_string(),
        source_report_sha256: sha256_bytes(&report_bytes),
        technical_delivery_reviewable: true,
        artifacts,
    })
}

pub fn inspect_render_review_manifest(
    manifest_path: &Path,
    expected_fixture: &str,
) -> RenderReviewInspection {
    inspect_render_review_manifest_internal(manifest_path, expected_fixture, None)
}

pub fn inspect_render_review_manifest_for_inputs(
    manifest_path: &Path,
    expected_fixture: &str,
    expected_registry_inputs: &[PathBuf],
) -> RenderReviewInspection {
    inspect_render_review_manifest_internal(
        manifest_path,
        expected_fixture,
        Some(expected_registry_inputs),
    )
}

fn inspect_render_review_manifest_internal(
    manifest_path: &Path,
    expected_fixture: &str,
    expected_registry_inputs: Option<&[PathBuf]>,
) -> RenderReviewInspection {
    let manifest_text = match std::fs::read_to_string(manifest_path) {
        Ok(contents) => contents,
        Err(error) => {
            return RenderReviewInspection::unreadable(
                "unreadable",
                format!("manifest_unreadable:{error}"),
            );
        }
    };
    let manifest = match serde_json::from_str::<RenderReviewManifest>(&manifest_text) {
        Ok(manifest) => manifest,
        Err(error) => {
            return RenderReviewInspection::unreadable(
                "invalid",
                format!("manifest_invalid:{error}"),
            );
        }
    };

    let mut issues = Vec::new();
    if manifest.schema_version != RENDER_REVIEW_SCHEMA_VERSION
        && manifest.schema_version != LEGACY_RENDER_REVIEW_SCHEMA_VERSION
    {
        issues.push("schema_version_unsupported".to_string());
    }
    let fixture_matches = manifest.fixture == expected_fixture;
    if !fixture_matches {
        issues.push("fixture_mismatch".to_string());
    }
    if manifest.review_status != "approved" {
        issues.push("review_status_not_approved".to_string());
    }
    if nonempty(manifest.reviewer.as_deref()).is_none() {
        issues.push("reviewer_missing".to_string());
    }
    match nonempty(manifest.reviewed_at.as_deref()) {
        None => issues.push("reviewed_at_missing".to_string()),
        Some(value) if !looks_like_rfc3339(value) => {
            issues.push("reviewed_at_not_rfc3339".to_string())
        }
        Some(_) => {}
    }
    if nonempty(manifest.review_notes.as_deref()).is_none() {
        issues.push("review_notes_missing".to_string());
    }

    let source_report = resolve_manifest_relative_path(manifest_path, &manifest.source_report);
    let (source_report_sha256_matches, report) = match std::fs::read(&source_report) {
        Ok(bytes) => {
            let matches = sha256_bytes(&bytes).eq_ignore_ascii_case(&manifest.source_report_sha256);
            if !matches {
                issues.push("source_report_sha256_mismatch".to_string());
            }
            match serde_json::from_slice::<PipelineReport>(&bytes) {
                Ok(report) => (Some(matches), Some(report)),
                Err(error) => {
                    issues.push(format!("source_report_invalid:{error}"));
                    (Some(matches), None)
                }
            }
        }
        Err(error) => {
            issues.push(format!("source_report_unreadable:{error}"));
            (None, None)
        }
    };

    let mut technical_delivery_reviewable = None;
    let mut all_input_hashes_match = None;
    let mut registry_inputs_match = None;
    let mut all_artifact_hashes_match = None;
    let mut grain_control_evidence = GrainControlInspectionEvidence {
        present: manifest.grain_reduction_control.is_some(),
        artifact_count: manifest
            .grain_reduction_control
            .as_ref()
            .map_or(0, |control| control.artifacts.len()),
        ..GrainControlInspectionEvidence::default()
    };
    let decisions = manifest.decisions.named();
    let applicable_decision_count = decisions
        .iter()
        .filter(|(_, decision)| decision.applicable)
        .count();
    let approved_applicable_decision_count = decisions
        .iter()
        .filter(|(_, decision)| decision.applicable && decision.approved)
        .count();

    if let Some(report) = report {
        let summary = summarize_report_with_source(expected_fixture, &report, Some(&source_report));
        let technical_reviewable = final_delivery_evidence_is_reviewable(&summary.render);
        technical_delivery_reviewable = Some(technical_reviewable);
        if !technical_reviewable {
            issues.push("technical_delivery_not_reviewable".to_string());
        }
        if manifest.technical_delivery_reviewable != technical_reviewable {
            issues.push("technical_delivery_reviewability_mismatch".to_string());
        }
        if summary.render.review_srgb_requested != Some(true) {
            issues.push("srgb_review_artifact_not_requested".to_string());
        }

        let expected_inputs = expected_report_inputs(&report, &summary);
        let expected_component_count = expected_inputs
            .as_ref()
            .map(|inputs| inputs.len())
            .unwrap_or(0);
        let expected_stitch_applicable = expected_component_count > 1;
        let expected_grain_applicable =
            summary.render.noise_reduction_requested_enabled == Some(true);
        if manifest.input_component_count != expected_component_count {
            issues.push("input_component_count_mismatch".to_string());
        }
        if manifest.stitch_applicable != expected_stitch_applicable {
            issues.push("stitch_applicability_mismatch".to_string());
        }
        if manifest.grain_reduction_applicable != expected_grain_applicable {
            issues.push("grain_applicability_mismatch".to_string());
        }
        match expected_inputs {
            Ok(expected_inputs) => {
                all_input_hashes_match = Some(validate_input_bindings(
                    manifest_path,
                    &manifest.inputs,
                    &expected_inputs,
                    &mut issues,
                ));
                if let Some(registry_inputs) = expected_registry_inputs {
                    let matches = registry_inputs.len() == expected_inputs.len()
                        && registry_inputs
                            .iter()
                            .zip(&expected_inputs)
                            .all(|(registry, report)| paths_equivalent(registry, &report.path));
                    registry_inputs_match = Some(matches);
                    if !matches {
                        issues.push("registry_input_set_mismatch".to_string());
                    }
                }
            }
            Err(error) => issues.push(format!("report_input_evidence_invalid:{error}")),
        }
        validate_decisions(
            &manifest.decisions,
            expected_stitch_applicable,
            expected_grain_applicable,
            &mut issues,
        );

        match expected_delivery_artifacts(&summary) {
            Ok(expected) => {
                let hashes_match = validate_artifact_bindings(
                    manifest_path,
                    &manifest.artifacts,
                    &expected,
                    &mut issues,
                );
                all_artifact_hashes_match = Some(hashes_match);
            }
            Err(error) => issues.push(format!("delivery_artifact_evidence_invalid:{error}")),
        }
        grain_control_evidence = inspect_grain_control_binding(
            manifest_path,
            manifest.grain_reduction_control.as_ref(),
            expected_grain_applicable,
            expected_fixture,
            &report,
            &summary,
            &mut issues,
        );
    }

    let approved = issues.is_empty();
    RenderReviewInspection {
        status: if approved {
            "approved"
        } else {
            "review_required"
        }
        .to_string(),
        approved,
        schema_version: Some(manifest.schema_version),
        fixture_matches: Some(fixture_matches),
        source_report_sha256_matches,
        technical_delivery_reviewable,
        input_count: manifest.inputs.len(),
        all_input_hashes_match,
        registry_inputs_match,
        artifact_count: manifest.artifacts.len(),
        all_artifact_hashes_match,
        grain_control_present: grain_control_evidence.present,
        grain_control_report_sha256_matches: grain_control_evidence.report_sha256_matches,
        grain_control_technical_delivery_reviewable: grain_control_evidence
            .technical_delivery_reviewable,
        grain_control_inputs_match: grain_control_evidence.inputs_match,
        grain_control_render_contract_matches: grain_control_evidence.render_contract_matches,
        grain_control_master_matches: grain_control_evidence.master_matches,
        grain_control_artifact_count: grain_control_evidence.artifact_count,
        all_grain_control_artifact_hashes_match: grain_control_evidence.all_artifact_hashes_match,
        decision_count: decisions.len(),
        applicable_decision_count,
        approved_applicable_decision_count,
        issues,
    }
}

#[derive(Debug, Default)]
struct GrainControlInspectionEvidence {
    present: bool,
    report_sha256_matches: Option<bool>,
    technical_delivery_reviewable: Option<bool>,
    inputs_match: Option<bool>,
    render_contract_matches: Option<bool>,
    master_matches: Option<bool>,
    artifact_count: usize,
    all_artifact_hashes_match: Option<bool>,
}

fn inspect_grain_control_binding(
    manifest_path: &Path,
    binding: Option<&RenderReviewGrainControlBinding>,
    expected_applicable: bool,
    fixture: &str,
    primary_report: &PipelineReport,
    primary_summary: &crate::validation::ValidationSummary,
    issues: &mut Vec<String>,
) -> GrainControlInspectionEvidence {
    let mut evidence = GrainControlInspectionEvidence {
        present: binding.is_some(),
        artifact_count: binding.map_or(0, |binding| binding.artifacts.len()),
        ..GrainControlInspectionEvidence::default()
    };
    if !expected_applicable {
        if binding.is_some() {
            issues.push("grain_control_unexpected".to_string());
        }
        return evidence;
    }
    let Some(binding) = binding else {
        issues.push("grain_control_missing".to_string());
        return evidence;
    };
    if binding.purpose != "grain_reduction_off_control" {
        issues.push("grain_control_purpose_invalid".to_string());
    }

    let source_report = resolve_manifest_relative_path(manifest_path, &binding.source_report);
    let report_bytes = match std::fs::read(&source_report) {
        Ok(bytes) => bytes,
        Err(error) => {
            issues.push(format!("grain_control_report_unreadable:{error}"));
            return evidence;
        }
    };
    let report_sha256_matches =
        sha256_bytes(&report_bytes).eq_ignore_ascii_case(&binding.source_report_sha256);
    evidence.report_sha256_matches = Some(report_sha256_matches);
    if !report_sha256_matches {
        issues.push("grain_control_report_sha256_mismatch".to_string());
    }
    let report = match serde_json::from_slice::<PipelineReport>(&report_bytes) {
        Ok(report) => report,
        Err(error) => {
            issues.push(format!("grain_control_report_invalid:{error}"));
            return evidence;
        }
    };
    let summary = summarize_report_with_source(fixture, &report, Some(&source_report));
    let technical_reviewable = final_delivery_evidence_is_reviewable(&summary.render);
    evidence.technical_delivery_reviewable = Some(technical_reviewable);
    if !technical_reviewable {
        issues.push("grain_control_technical_delivery_not_reviewable".to_string());
    }
    if binding.technical_delivery_reviewable != technical_reviewable {
        issues.push("grain_control_technical_reviewability_mismatch".to_string());
    }
    if summary.render.noise_reduction_requested_enabled != Some(false)
        || summary.tone.noise_reduction_enabled != Some(false)
    {
        issues.push("grain_control_reduction_not_off".to_string());
    }

    let inputs_match = match (
        expected_report_inputs(primary_report, primary_summary),
        expected_report_inputs(&report, &summary),
    ) {
        (Ok(primary), Ok(control)) => expected_grain_control_inputs_match(&primary, &control),
        (Err(error), _) => {
            issues.push(format!(
                "grain_control_primary_input_evidence_invalid:{error}"
            ));
            false
        }
        (_, Err(error)) => {
            issues.push(format!("grain_control_input_evidence_invalid:{error}"));
            false
        }
    };
    evidence.inputs_match = Some(inputs_match);
    if !inputs_match {
        issues.push("grain_control_inputs_mismatch".to_string());
    }

    let render_contract_issues =
        grain_control_render_contract_issues(primary_report, primary_summary, &report, &summary);
    let render_contract_matches = render_contract_issues.is_empty();
    evidence.render_contract_matches = Some(render_contract_matches);
    for issue in render_contract_issues {
        issues.push(format!("grain_control_render_contract_mismatch:{issue}"));
    }

    match expected_delivery_artifacts(&summary) {
        Ok(expected) => {
            let mut control_artifact_issues = Vec::new();
            let hashes_match = validate_artifact_bindings(
                manifest_path,
                &binding.artifacts,
                &expected,
                &mut control_artifact_issues,
            );
            evidence.all_artifact_hashes_match = Some(hashes_match);
            issues.extend(
                control_artifact_issues
                    .into_iter()
                    .map(|issue| format!("grain_control_{issue}")),
            );
        }
        Err(error) => issues.push(format!("grain_control_artifact_evidence_invalid:{error}")),
    }

    let master_matches = match (
        expected_delivery_artifacts(primary_summary),
        expected_delivery_artifacts(&summary),
    ) {
        (Ok(primary), Ok(control)) => {
            let primary = primary
                .iter()
                .find(|(kind, _)| *kind == "master_scene_referred")
                .map(|(_, path)| path);
            let control = control
                .iter()
                .find(|(kind, _)| *kind == "master_scene_referred")
                .map(|(_, path)| path);
            match (primary, control) {
                (Some(primary), Some(control)) => {
                    match (hash_file_sha256(primary), hash_file_sha256(control)) {
                        (Ok((primary, _)), Ok((control, _))) => {
                            primary.eq_ignore_ascii_case(&control)
                        }
                        (Err(error), _) => {
                            issues.push(format!("grain_control_primary_master_unreadable:{error}"));
                            false
                        }
                        (_, Err(error)) => {
                            issues.push(format!("grain_control_master_unreadable:{error}"));
                            false
                        }
                    }
                }
                _ => {
                    issues.push("grain_control_master_binding_missing".to_string());
                    false
                }
            }
        }
        (Err(error), _) => {
            issues.push(format!(
                "grain_control_primary_artifact_evidence_invalid:{error}"
            ));
            false
        }
        (_, Err(error)) => {
            issues.push(format!("grain_control_artifact_evidence_invalid:{error}"));
            false
        }
    };
    evidence.master_matches = Some(master_matches);
    if !master_matches {
        issues.push("grain_control_scene_referred_master_mismatch".to_string());
    }

    evidence
}

fn validate_input_bindings(
    manifest_path: &Path,
    bindings: &[RenderReviewInputBinding],
    expected: &[ExpectedReportInput],
    issues: &mut Vec<String>,
) -> bool {
    let mut by_index = BTreeMap::new();
    for binding in bindings {
        if by_index.insert(binding.component_index, binding).is_some() {
            issues.push(format!(
                "input_component_index_duplicate:{}",
                binding.component_index
            ));
        }
    }
    let expected_indices = expected
        .iter()
        .map(|input| input.component_index)
        .collect::<BTreeSet<_>>();
    for index in by_index
        .keys()
        .filter(|index| !expected_indices.contains(index))
    {
        issues.push(format!("input_component_unexpected:{index}"));
    }

    let mut all_match = true;
    for input in expected {
        let index = input.component_index;
        let Some(binding) = by_index.get(&index) else {
            issues.push(format!("input_component_missing:{index}"));
            all_match = false;
            continue;
        };
        let bound_path = resolve_manifest_relative_path(manifest_path, &binding.path);
        if !paths_equivalent(&bound_path, &input.path) {
            issues.push(format!("input_component_path_mismatch:{index}"));
            all_match = false;
        }
        if !binding
            .decoded_pixel_sha256
            .eq_ignore_ascii_case(&input.decoded_pixel_sha256)
        {
            issues.push(format!("input_decoded_pixel_sha256_mismatch:{index}"));
            all_match = false;
        }
        match hash_file_sha256(&input.path) {
            Ok((actual_sha256, actual_size)) => {
                if !actual_sha256.eq_ignore_ascii_case(&binding.sha256) {
                    issues.push(format!("input_component_sha256_mismatch:{index}"));
                    all_match = false;
                }
                if actual_size != binding.file_size_bytes {
                    issues.push(format!("input_component_size_mismatch:{index}"));
                    all_match = false;
                }
            }
            Err(error) => {
                issues.push(format!("input_component_unreadable:{index}:{error}"));
                all_match = false;
            }
        }
    }
    all_match && by_index.len() == expected.len()
}

fn grain_control_inputs_match(
    primary: &[RenderReviewInputBinding],
    control: &[ExpectedReportInput],
) -> bool {
    primary.len() == control.len()
        && primary.iter().zip(control).all(|(primary, control)| {
            primary.component_index == control.component_index
                && paths_equivalent(Path::new(&primary.path), &control.path)
                && primary
                    .decoded_pixel_sha256
                    .eq_ignore_ascii_case(&control.decoded_pixel_sha256)
        })
}

fn expected_grain_control_inputs_match(
    primary: &[ExpectedReportInput],
    control: &[ExpectedReportInput],
) -> bool {
    primary.len() == control.len()
        && primary.iter().zip(control).all(|(primary, control)| {
            primary.component_index == control.component_index
                && paths_equivalent(&primary.path, &control.path)
                && primary
                    .decoded_pixel_sha256
                    .eq_ignore_ascii_case(&control.decoded_pixel_sha256)
        })
}

fn grain_control_render_contract_issues(
    primary_report: &PipelineReport,
    primary: &crate::validation::ValidationSummary,
    control_report: &PipelineReport,
    control: &crate::validation::ValidationSummary,
) -> Vec<&'static str> {
    let mut issues = Vec::new();
    if primary.report.report_schema_version != control.report.report_schema_version {
        issues.push("report_schema_version");
    }
    if primary.report.pipeline_schema_version != control.report.pipeline_schema_version {
        issues.push("pipeline_schema_version");
    }
    if primary.report.package_version != control.report.package_version {
        issues.push("package_version");
    }
    if primary.report.binary_name != control.report.binary_name {
        issues.push("binary_name");
    }
    let primary_binary_valid = report_binary_identity_is_valid(primary);
    let control_binary_valid = report_binary_identity_is_valid(control);
    if !primary_binary_valid || !control_binary_valid {
        issues.push("binary_identity_missing_or_invalid");
    } else {
        if primary.report.binary_sha256 != control.report.binary_sha256 {
            issues.push("binary_sha256");
        }
        if primary.report.binary_file_size_bytes != control.report.binary_file_size_bytes {
            issues.push("binary_file_size_bytes");
        }
    }
    if primary.report.working_directory != control.report.working_directory {
        issues.push("working_directory");
    }
    match (
        normalized_grain_comparison_cli_args(primary.report.cli_args.as_deref()),
        normalized_grain_comparison_cli_args(control.report.cli_args.as_deref()),
    ) {
        (Ok(primary), Ok(control)) if primary == control => {}
        (Ok(_), Ok(_)) => issues.push("normalized_cli_args"),
        _ => issues.push("cli_args_missing_or_invalid"),
    }
    if primary.render.output_width != control.render.output_width {
        issues.push("output_width");
    }
    if primary.render.output_height != control.render.output_height {
        issues.push("output_height");
    }
    if primary.render.output_color_space != control.render.output_color_space {
        issues.push("output_color_space");
    }
    if primary.render.render_intent != control.render.render_intent {
        issues.push("render_intent");
    }
    if primary.render.quality_mode != control.render.quality_mode {
        issues.push("quality_mode");
    }
    if primary.white_balance.creative_applied != control.white_balance.creative_applied {
        issues.push("creative_white_balance_application");
    }
    if primary.white_balance.creative_temperature != control.white_balance.creative_temperature {
        issues.push("creative_temperature");
    }
    if primary.white_balance.creative_tint != control.white_balance.creative_tint {
        issues.push("creative_tint");
    }
    if primary.white_balance.creative_target_temperature_kelvin
        != control.white_balance.creative_target_temperature_kelvin
    {
        issues.push("creative_target_temperature_kelvin");
    }
    if normalized_interactive_controls(primary_report)
        != normalized_interactive_controls(control_report)
    {
        issues.push("interactive_controls");
    }
    issues
}

fn report_binary_identity_is_valid(summary: &crate::validation::ValidationSummary) -> bool {
    summary.report.binary_identity_status.as_deref() == Some("verified_sha256")
        && summary
            .report
            .binary_sha256
            .as_deref()
            .is_some_and(is_sha256_hex)
        && summary
            .report
            .binary_file_size_bytes
            .is_some_and(|size| size > 0)
        && summary
            .report
            .binary_path
            .as_deref()
            .is_some_and(|path| !path.trim().is_empty())
        && summary.report.binary_identity_error.is_none()
}

fn normalized_interactive_controls(report: &PipelineReport) -> Option<serde_json::Value> {
    let mut controls = report
        .phases
        .iter()
        .find(|phase| phase.name == "tone_mapping")?
        .metrics
        .get("interactive_controls")?
        .as_object()?
        .clone();
    controls.remove("grain_reduction_enabled");
    controls.remove("grain_reduction_strength");
    controls.remove("grain_reduction_scale");
    Some(serde_json::Value::Object(controls))
}

fn normalized_grain_comparison_cli_args(args: Option<&[String]>) -> Result<Vec<String>, ()> {
    let args = args.filter(|args| !args.is_empty()).ok_or(())?;
    let mut normalized = Vec::new();
    let mut index = 1usize;
    while index < args.len() {
        let argument = &args[index];
        let removes_following_value = matches!(
            argument.as_str(),
            "--output-dir" | "-o" | "--grain-reduction" | "--grain-strength" | "--grain-scale"
        );
        if removes_following_value {
            if index + 1 >= args.len() {
                return Err(());
            }
            index += 2;
            continue;
        }
        if [
            "--output-dir=",
            "--grain-reduction=",
            "--grain-strength=",
            "--grain-scale=",
        ]
        .iter()
        .any(|prefix| argument.starts_with(prefix))
        {
            index += 1;
            continue;
        }
        if argument.starts_with("-o") && !argument.starts_with("--") && argument.len() > 2 {
            index += 1;
            continue;
        }
        normalized.push(argument.clone());
        index += 1;
    }
    Ok(normalized)
}

fn validate_decisions(
    decisions: &RenderReviewDecisions,
    stitch_applicable: bool,
    grain_applicable: bool,
    issues: &mut Vec<String>,
) {
    let expected = BTreeMap::from([
        ("orientation_and_retained_content", true),
        ("border_crop_and_no_lost_content", true),
        ("stitch_geometry_and_full_union", stitch_applicable),
        ("seam_tone_color_texture_sharpness", stitch_applicable),
        ("color_and_neutrality", true),
        ("tone_and_dynamic_range", true),
        ("grain_and_fine_detail", grain_applicable),
        ("overall_visual_preference", true),
    ]);
    for (name, decision) in decisions.named() {
        let expected_applicable = expected[name];
        if decision.applicable != expected_applicable {
            issues.push(format!("decision_applicability_mismatch:{name}"));
            continue;
        }
        if expected_applicable {
            if !decision.approved {
                issues.push(format!("decision_not_approved:{name}"));
            }
            if nonempty(decision.notes.as_deref()).is_none() {
                issues.push(format!("decision_notes_missing:{name}"));
            }
        } else if decision.approved {
            issues.push(format!("nonapplicable_decision_approved:{name}"));
        }
    }
}

fn validate_artifact_bindings(
    manifest_path: &Path,
    bindings: &[RenderReviewArtifactBinding],
    expected: &[(&'static str, PathBuf)],
    issues: &mut Vec<String>,
) -> bool {
    let mut by_kind = BTreeMap::new();
    for binding in bindings {
        if by_kind.insert(binding.kind.as_str(), binding).is_some() {
            issues.push(format!("artifact_kind_duplicate:{}", binding.kind));
        }
    }
    let expected_kinds = expected
        .iter()
        .map(|(kind, _)| *kind)
        .collect::<BTreeSet<_>>();
    for kind in by_kind
        .keys()
        .filter(|kind| !expected_kinds.contains(**kind))
    {
        issues.push(format!("artifact_unexpected:{kind}"));
    }

    let mut all_match = true;
    for (kind, expected_path) in expected {
        let Some(binding) = by_kind.get(kind) else {
            issues.push(format!("artifact_missing:{kind}"));
            all_match = false;
            continue;
        };
        let bound_path = resolve_manifest_relative_path(manifest_path, &binding.path);
        if !paths_equivalent(&bound_path, expected_path) {
            issues.push(format!("artifact_path_mismatch:{kind}"));
            all_match = false;
        }
        match hash_file_sha256(expected_path) {
            Ok((actual_sha256, actual_size)) => {
                if !actual_sha256.eq_ignore_ascii_case(&binding.sha256) {
                    issues.push(format!("artifact_sha256_mismatch:{kind}"));
                    all_match = false;
                }
                if actual_size != binding.file_size_bytes {
                    issues.push(format!("artifact_size_mismatch:{kind}"));
                    all_match = false;
                }
            }
            Err(error) => {
                issues.push(format!("artifact_unreadable:{kind}:{error}"));
                all_match = false;
            }
        }
    }
    all_match && by_kind.len() == expected.len()
}

fn artifact_binding<'a>(
    artifacts: &'a [RenderReviewArtifactBinding],
    kind: &str,
) -> Option<&'a RenderReviewArtifactBinding> {
    artifacts.iter().find(|artifact| artifact.kind == kind)
}

fn expected_delivery_artifacts(
    summary: &crate::validation::ValidationSummary,
) -> Result<Vec<(&'static str, PathBuf)>, Box<dyn std::error::Error>> {
    let mut artifacts = Vec::new();
    let output = summary
        .render
        .output_path
        .as_deref()
        .ok_or("primary output path missing from report")?;
    artifacts.push((
        "primary_output",
        resolve_report_relative_path(summary, output),
    ));
    if summary.render.master_scene_referred_requested == Some(true) {
        let path = summary
            .render
            .master_scene_referred_path
            .as_deref()
            .ok_or("requested scene-referred master path missing from report")?;
        artifacts.push((
            "master_scene_referred",
            resolve_report_relative_path(summary, path),
        ));
    }
    if summary.render.review_srgb_requested == Some(true) {
        let path = summary
            .render
            .review_srgb_path
            .as_deref()
            .ok_or("requested sRGB review path missing from report")?;
        artifacts.push(("review_srgb", resolve_report_relative_path(summary, path)));
    }
    Ok(artifacts)
}

fn expected_report_inputs(
    report: &PipelineReport,
    summary: &crate::validation::ValidationSummary,
) -> Result<Vec<ExpectedReportInput>, Box<dyn std::error::Error>> {
    let components = report
        .phases
        .iter()
        .find(|phase| phase.name == "load")
        .and_then(|phase| phase.metrics.get("components"))
        .and_then(serde_json::Value::as_array)
        .ok_or("load-phase component evidence missing")?;
    if components.is_empty() {
        return Err("load-phase component evidence is empty".into());
    }
    let mut inputs = Vec::with_capacity(components.len());
    for component in components {
        let index = component
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("load component index missing or invalid")?;
        let path = component
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or("load component path missing")?;
        let decoded_pixel_sha256 = component
            .get("decode")
            .and_then(|decode| decode.get("decoded_pixel_sha256"))
            .and_then(serde_json::Value::as_str)
            .filter(|value| is_sha256_hex(value))
            .ok_or("load component decoded-pixel SHA-256 missing or invalid")?;
        inputs.push(ExpectedReportInput {
            component_index: index,
            path: resolve_report_relative_path(summary, path),
            decoded_pixel_sha256: decoded_pixel_sha256.to_ascii_lowercase(),
        });
    }
    inputs.sort_by_key(|input| input.component_index);
    if inputs
        .iter()
        .enumerate()
        .any(|(offset, input)| input.component_index != offset + 1)
    {
        return Err("load component indices are not a complete one-based sequence".into());
    }
    Ok(inputs)
}

fn resolve_report_relative_path(
    summary: &crate::validation::ValidationSummary,
    path: &str,
) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        return candidate;
    }
    if let Some(working_directory) = summary
        .report
        .working_directory
        .as_deref()
        .filter(|directory| !directory.trim().is_empty())
    {
        return PathBuf::from(working_directory).join(candidate);
    }
    if let Some(parent) = summary
        .report
        .source_report_path
        .as_deref()
        .map(Path::new)
        .and_then(Path::parent)
    {
        return parent.join(candidate);
    }
    candidate
}

fn resolve_manifest_relative_path(manifest_path: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(candidate)
    }
}

fn paths_equivalent(left: &Path, right: &Path) -> bool {
    let left = left
        .canonicalize()
        .or_else(|_| absolute_path(left))
        .unwrap_or_else(|_| left.to_path_buf());
    let right = right
        .canonicalize()
        .or_else(|_| absolute_path(right))
        .unwrap_or_else(|_| right.to_path_buf());
    left == right
}

fn absolute_path(path: &Path) -> std::io::Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn hash_file_sha256(path: &Path) -> Result<(String, u64), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let file_size_bytes = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok((format!("{:x}", hasher.finalize()), file_size_bytes))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.trim().is_empty())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn looks_like_rfc3339(value: &str) -> bool {
    let Some((date, time)) = value.split_once('T') else {
        return false;
    };
    let date_shape = date.len() == 10
        && date.as_bytes().get(4) == Some(&b'-')
        && date.as_bytes().get(7) == Some(&b'-');
    let timezone_present = time.ends_with('Z')
        || time
            .get(8..)
            .is_some_and(|suffix| suffix.contains('+') || suffix.contains('-'));
    date_shape && time.len() >= 9 && timezone_present
}

fn render_review_to_markdown(manifest: &RenderReviewManifest) -> String {
    let mut out = String::from("# Render Review Draft\n\n");
    out.push_str(
        "Status: `requires_human_approval`. This package never approves its own output.\n\n",
    );
    out.push_str(&format!("{}\n\n", manifest.instructions));
    out.push_str(&format!(
        "- fixture: `{}`\n- technical delivery reviewable: `{}`\n- input components: `{}`\n- source report: `{}`\n- source report SHA-256: `{}`\n\n",
        manifest.fixture,
        manifest.technical_delivery_reviewable,
        manifest.input_component_count,
        manifest.source_report,
        manifest.source_report_sha256
    ));
    out.push_str("## Exact source components\n\n");
    out.push_str(
        "| Component | Path | Bytes | File SHA-256 | Decoded-pixel SHA-256 |\n|-:|-|-:|-|-|\n",
    );
    for input in &manifest.inputs {
        out.push_str(&format!(
            "| {} | `{}` | {} | `{}` | `{}` |\n",
            input.component_index,
            input.path,
            input.file_size_bytes,
            input.sha256,
            input.decoded_pixel_sha256
        ));
    }
    out.push('\n');
    out.push_str("## Exact artifacts\n\n");
    out.push_str("| Kind | Path | Bytes | SHA-256 |\n|-|-|-:|-|\n");
    for artifact in &manifest.artifacts {
        out.push_str(&format!(
            "| `{}` | `{}` | {} | `{}` |\n",
            artifact.kind, artifact.path, artifact.file_size_bytes, artifact.sha256
        ));
    }
    if let Some(control) = &manifest.grain_reduction_control {
        out.push_str("\n## Exact grain-off control\n\n");
        out.push_str(&format!(
            "- purpose: `{}`\n- technical delivery reviewable: `{}`\n- source report: `{}`\n- source report SHA-256: `{}`\n\n",
            control.purpose,
            control.technical_delivery_reviewable,
            control.source_report,
            control.source_report_sha256
        ));
        out.push_str("| Kind | Path | Bytes | SHA-256 |\n|-|-|-:|-|\n");
        for artifact in &control.artifacts {
            out.push_str(&format!(
                "| `{}` | `{}` | {} | `{}` |\n",
                artifact.kind, artifact.path, artifact.file_size_bytes, artifact.sha256
            ));
        }
        out.push_str("\nCompare the primary and control `review_srgb` artifacts at 100%. Their scene-referred masters must remain byte-identical, proving that the bound comparison differs only in the permitted grain controls and output location.\n");
    }
    out.push_str("\n## Decisions\n\n");
    out.push_str("Inspect the full-resolution `review_srgb` artifact for colour/tone/crop and the primary/master artifacts at 100% for seam and fine-detail decisions.\n\n");
    out.push_str("| Decision | Applicable | Approved in draft |\n|-|-:|-:|\n");
    for (name, decision) in manifest.decisions.named() {
        out.push_str(&format!(
            "| `{name}` | {} | {} |\n",
            decision.applicable, decision.approved
        ));
    }
    out.push_str("\nAn approved manifest is accepted only when its own registry SHA-256 is pinned, its report and every delivery artifact still match, any applicable grain-off control remains compatible and hash-complete, technical delivery remains reviewable, every applicable decision is approved with notes, and reviewer/date/overall notes are present.\n");
    out
}

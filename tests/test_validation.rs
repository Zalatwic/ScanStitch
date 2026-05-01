use scanstitch::report::{PhaseReport, PipelineReport};

fn fixture_report() -> PipelineReport {
    let mut report = PipelineReport::new();
    report.add_phase(PhaseReport::ok(
        "stitch",
        0.82,
        serde_json::json!({
            "decision": "accepted",
            "chosen_hypothesis": "[1|2]",
            "hypotheses": [
                {
                    "ordering": "[1|2]",
                    "accepted": true,
                    "search": {
                        "selection_reason": "best_evidence_score",
                        "max_overlap_considered": 420,
                        "top_candidates": [
                            {
                                "rank": 1,
                                "overlap_width": 180,
                                "vertical_offset": -2,
                                "correspondence_score": 0.91,
                                "prior_weight": 0.64,
                                "search_score": 0.88
                            }
                        ],
                        "evaluated_candidates": [{ "validation_rank": 1 }]
                    },
                    "validation": {
                        "evidence_score": 0.87,
                        "prior_weight": 0.64,
                        "overlap_support_score": 0.93,
                        "vertical_offset_plausibility_score": 0.98,
                        "local_consistency_score": 0.76,
                        "plausibility_score": 0.91
                    }
                }
            ]
        }),
    ));
    report.add_phase(PhaseReport::ok(
        "working_image_select",
        0.73,
        serde_json::json!({
            "raw_base_confidence": 0.69,
            "base_estimate_source": "working_edges"
        }),
    ));
    report.add_phase(PhaseReport::ok(
        "density_inversion",
        0.73,
        serde_json::json!({ "base_estimate_source": "working_edges" }),
    ));
    let mut colorspace = PhaseReport::ok(
        "colorspace_mapping",
        0.90,
        serde_json::json!({
            "render_input_source": "direct_density_transmittance",
            "render_input_reason": "safer low-clipping candidate",
            "direct_density_candidate_evaluated": true,
            "exposure_scale": 1.12,
            "mapping_strategy": "neutral_balance_weak_anchor_fallback",
            "regularization_lambda": 0.20,
            "channel_anchor_counts": [128, 0, 130],
            "channel_anchor_min_count": 0,
            "channel_anchor_low_support": [false, true, false],
            "weak_anchor_fallback_used": true,
            "gamut_fallback_used": false,
            "image_matrix_pre_scale_clipped_low_ratio": [0.02, 0.00, 0.01],
            "image_matrix_pre_scale_clipped_high_ratio": [0.00, 0.03, 0.00],
            "image_matrix_exposure_scale": 1.31,
            "post_scale_clipped_high_ratio": [0.0, 0.001, 0.0],
            "post_scale_clipped_low_ratio": [0.0, 0.0, 0.0]
        }),
    );
    colorspace
        .warnings
        .push("colorspace matrix has weak dominant-channel anchor support".to_string());
    report.add_phase(colorspace);
    report.add_phase(PhaseReport::ok(
        "tone_mapping",
        0.90,
        serde_json::json!({
            "shadow_saturation_median": 0.12,
            "shadow_saturation_p95": 0.31,
            "midtone_saturation_median": 0.18,
            "midtone_saturation_p95": 0.42,
            "midtone_luminance_percentiles": [0.33, 0.48, 0.61],
            "bright_neutral_saturation_median": 0.04,
            "bright_neutral_saturation_p95": 0.08,
            "bright_saturated_saturation_median": 0.63,
            "bright_saturated_saturation_p95": 0.81,
            "post_chroma_compression_clipped_high_ratio": [0.0, 0.0, 0.002],
            "post_chroma_compression_clipped_low_ratio": [0.0, 0.001, 0.0]
        }),
    ));
    report
}

#[test]
fn test_validation_summary_extracts_standard_report_fields() {
    let report = fixture_report();
    let summary = scanstitch::validation::summarize_report("logan", &report);

    assert_eq!(summary.fixture, "logan");
    assert_eq!(summary.stitch.decision.as_deref(), Some("accepted"));
    assert_eq!(summary.stitch.confidence, Some(0.82));
    assert_eq!(
        summary.stitch.search_selection_reason.as_deref(),
        Some("best_evidence_score")
    );
    assert_eq!(summary.stitch.evaluated_candidate_count, Some(1));
    assert_eq!(
        summary.stitch.top_candidates[0].correspondence_score,
        Some(0.91)
    );
    assert_eq!(summary.base_density.base_confidence, Some(0.73));
    assert_eq!(summary.base_density.density_confidence, Some(0.73));
    assert_eq!(
        summary.colorspace.render_input_source.as_deref(),
        Some("direct_density_transmittance")
    );
    assert_eq!(
        summary.colorspace.channel_anchor_low_support,
        Some(vec![false, true, false])
    );
    assert_eq!(summary.colorspace.weak_anchor_fallback_used, Some(true));
    assert_eq!(summary.tone.bright_neutral_saturation_p95, Some(0.08));
    assert_eq!(
        summary.tone.post_chroma_compression_clipped_high_ratio,
        Some(vec![0.0, 0.0, 0.002])
    );
    assert_eq!(summary.warnings.len(), 1);
}

#[test]
fn test_validation_summary_markdown_is_compact() {
    let report = fixture_report();
    let summary = scanstitch::validation::summarize_report("logan", &report);
    let markdown = scanstitch::validation::summary_to_markdown(&summary);

    assert!(markdown.contains("# Validation Summary: logan"));
    assert!(markdown.contains("neutral_balance_weak_anchor_fallback"));
    assert!(markdown.contains("bright_neutral_saturation_p95"));
    assert!(markdown.contains("colorspace matrix has weak dominant-channel anchor support"));
}

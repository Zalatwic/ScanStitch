#![recursion_limit = "512"]

mod common;
use common::synthetic;
use scanstitch::report::{PhaseReport, PipelineReport, RunMetadata};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;
use std::process::Command;

fn file_sha256_hex(path: &Path) -> String {
    let mut file = std::fs::File::open(path).unwrap();
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).unwrap();
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    format!("{:x}", hasher.finalize())
}

fn calibration_library_sha256_hex(path: &Path) -> String {
    let mut stack = vec![path.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
            {
                files.push(path);
            }
        }
    }
    files.sort_by_key(|file| {
        file.strip_prefix(path)
            .unwrap()
            .components()
            .map(|component| component.as_os_str().to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("/")
    });

    let mut hasher = Sha256::new();
    for file in files {
        let relative_path = file
            .strip_prefix(path)
            .unwrap()
            .components()
            .map(|component| component.as_os_str().to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("/");
        hasher.update(relative_path.as_bytes());
        hasher.update(b"\0");
        hasher.update(file_sha256_hex(&file).as_bytes());
        hasher.update(b"\0");
    }
    format!("{:x}", hasher.finalize())
}

fn write_rgba8_tiff(path: &Path, width: u32, height: u32) {
    let pixels = vec![128_u8; width as usize * height as usize * 4];
    let file = std::fs::File::create(path).unwrap();
    let mut encoder = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file)).unwrap();
    encoder
        .write_image::<tiff::encoder::colortype::RGBA8>(width, height, &pixels)
        .unwrap();
}

fn fixture_report() -> PipelineReport {
    let mut report = PipelineReport::new();
    report.metadata = Some(RunMetadata {
        report_schema_version: 2,
        pipeline_schema_version: 1,
        generated_at: "2026-05-01T21:00:00Z".to_string(),
        generated_at_unix_ms: 1_777_665_600_000,
        package_name: "scanstitch".to_string(),
        package_version: "0.1.0".to_string(),
        binary_name: "scanstitch".to_string(),
        working_directory: "V:\\PERSONAL\\PHOTOGRAPHY\\SCANSTITCH".to_string(),
        cli_args: vec!["scanstitch".to_string()],
        output_dir: "output".to_string(),
        output_path: "output/output.tiff".to_string(),
    });
    report.add_phase(PhaseReport::ok(
        "stitch",
        0.82,
        serde_json::json!({
            "decision": "accepted",
            "chosen_hypothesis": "[1|2]",
            "seam_exposure_correction": {
                "mode": "auto",
                "applied": true,
                "reason": "applied luma-only overlap gain",
                "sample_count": 8192,
                "valid_sample_ratio": 0.72,
                "gain_rgb": [0.91, 0.91, 0.91],
                "gain_luma": 0.91,
                "seam_score_before": 0.102,
                "seam_score_after": 0.018,
                "clipped_high_before": [0.0, 0.0, 0.0],
                "clipped_high_after": [0.0, 0.0, 0.0],
                "clipped_low_before": [0.0, 0.0, 0.0],
                "clipped_low_after": [0.0, 0.0, 0.0]
            },
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
            "raw_base_proxy_confidence": 0.12,
            "raw_base_support_fraction": 0.014,
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
            "calibration": {
                "status": "applied",
                "source": "external_calibration_profile",
                "external_profile": {
                    "path": "docs/color-calibration-profile.example.json",
                    "profile_id": "synthetic-prophoto-d50"
                },
                "scanner_profile": {
                    "status": "applied",
                    "profile_id": "scanner-a"
                },
                "roll_profile": {
                    "status": "applied",
                    "profile_id": "roll-a"
                },
                "profile_schema_version": 1,
                "confidence": 0.95,
                "reason": "valid external calibration profile selected over image-derived estimate",
                "matrix_condition_number": 2.1,
                "whitepoint": [0.9642, 1.0, 0.8251],
                "requested_film_stock": "Synthetic 200",
                "film_stock": {
                    "status": "matched",
                    "selection": "explicit_film_stock",
                    "requested_stock": "Synthetic 200",
                    "matched_roll_profiles": ["roll-a"],
                    "reason": "requested film stock matched one film hint",
                    "rejection_details": []
                },
                "rejection_details": []
            },
            "render_input_source": "direct_density_transmittance",
            "render_input_reason": "safer low-clipping candidate",
            "direct_density_candidate_evaluated": true,
            "exposure_scale": 1.12,
            "mapping_strategy": "neutral_balance_weak_anchor_fallback",
            "selected_mapping_reason": "synthetic weak-anchor fallback",
            "selected_candidate": "neutral_balance_fallback",
            "selected_candidate_rank": 1,
            "selected_candidate_score": 0.35,
            "selected_quality_score": 0.42,
            "technical_safety_score": 0.12,
            "color_fidelity_score": 0.30,
            "candidate_risk": "review_reference_fit",
            "tone_color_trust_state": "review_required",
            "selected_runner_up_quality_delta": 0.46,
            "candidate_quality_scores": [
                {
                    "candidate": "image_derived_matrix",
                    "rank": 2,
                    "selected": false,
                    "quality_score": 0.88,
                    "selected_quality_delta": 0.46,
                    "rejected": true,
                    "rejection_reason": "synthetic destructive clip",
                    "pre_scale_clipped_low_total": 0.03,
                    "pre_scale_clipped_low_max": 0.02,
                    "pre_scale_clipped_high_total": 0.03,
                    "pre_scale_clipped_high_max": 0.03,
                    "reference_patch_rms_delta_e": 3.2,
                    "reference_patch_rms_delta_e2000": 2.7,
                    "reference_patch_max_delta_vs_image_derived": 0.2,
                    "reference_patch_delta_e_delta_vs_image_derived": 1.1,
                    "reference_patch_delta_e_max_delta_vs_image_derived": 1.6,
                    "reference_patch_delta_e2000_delta_vs_image_derived": 0.8,
                    "reference_patch_delta_e2000_max_delta_vs_image_derived": 1.3,
                    "color_model_quality": {
                        "density_monotonicity_score": 0.95,
                        "hue_linearity_score": 0.62,
                        "saturation_preservation_median_ratio": 0.84,
                        "spatial_consistency": {
                            "neutral_delta_p95": 0.045
                        }
                    },
                    "quality_components": {
                        "low_gamut_clip_penalty": 0.24,
                        "high_gamut_clip_penalty": 0.045,
                        "preserved_gamut_penalty": 0.15,
                        "exposure_penalty": 0.0,
                        "neutral_balance_penalty": 0.02,
                        "neutral_estimate_penalty": 0.0,
                        "anchor_support_penalty": 0.55,
                        "anchor_stability_penalty": 0.25,
                        "condition_penalty": 0.0,
                        "calibration_confidence_penalty": 0.18,
                        "target_residual_penalty": 0.0,
                        "density_monotonicity_penalty": 0.0,
                        "hue_linearity_penalty": 0.0,
                        "saturation_preservation_penalty": 0.0,
                        "memory_color_penalty": 0.0,
                        "spatial_consistency_penalty": 0.0,
                        "fallback_penalty": 0.0
                    }
                },
                {
                    "candidate": "neutral_balance_fallback",
                    "rank": 1,
                    "selected": true,
                    "quality_score": 0.42,
                    "selected_quality_delta": 0.0,
                    "rejected": false,
                    "rejection_reason": null,
                    "pre_scale_clipped_low_total": 0.0,
                    "pre_scale_clipped_low_max": 0.0,
                    "pre_scale_clipped_high_total": 0.001,
                    "pre_scale_clipped_high_max": 0.001,
                    "color_model_quality": {
                        "density_monotonicity_score": 1.0,
                        "hue_linearity_score": 0.92,
                        "saturation_preservation_median_ratio": 0.76,
                        "spatial_consistency": {
                            "neutral_delta_p95": 0.006
                        }
                    },
                    "quality_components": {
                        "low_gamut_clip_penalty": 0.0,
                        "high_gamut_clip_penalty": 0.0015,
                        "preserved_gamut_penalty": 0.0,
                        "exposure_penalty": 0.0,
                        "neutral_balance_penalty": 0.0,
                        "neutral_estimate_penalty": 0.0,
                        "anchor_support_penalty": 0.0,
                        "anchor_stability_penalty": 0.0,
                        "condition_penalty": 0.0,
                        "calibration_confidence_penalty": 0.40,
                        "target_residual_penalty": 0.0,
                        "density_monotonicity_penalty": 0.0,
                        "hue_linearity_penalty": 0.0,
                        "saturation_preservation_penalty": 0.0,
                        "memory_color_penalty": 0.06,
                        "spatial_consistency_penalty": 0.07,
                        "fallback_penalty": 0.35
                    }
                }
            ],
            "candidate_acceptance": [
                {
                    "candidate": "image_derived_matrix",
                    "candidate_kind": "image_derived",
                    "mapping_strategy": "image_derived_matrix",
                    "source_label": "image-derived",
                    "status": "rejected_safety",
                    "reason": "synthetic destructive clip",
                    "rank": 2,
                    "selected": false,
                    "eligible_in_color_mode": true,
                    "quality_score": 0.88,
                    "selected_quality_delta": 0.46,
                    "rejected": true,
                    "rejection_reason": "synthetic destructive clip",
                    "beats_image_derived": null,
                    "within_negative_gamut_limits": false
                },
                {
                    "candidate": "neutral_balance_fallback",
                    "candidate_kind": "neutral_fallback",
                    "mapping_strategy": "neutral_balance_weak_anchor_fallback",
                    "source_label": "neutral-balance fallback",
                    "status": "selected",
                    "reason": "neutral selected",
                    "rank": 1,
                    "selected": true,
                    "eligible_in_color_mode": true,
                    "quality_score": 0.42,
                    "selected_quality_delta": 0.0,
                    "rejected": false,
                    "rejection_reason": null,
                    "beats_image_derived": null,
                    "within_negative_gamut_limits": null
                }
            ],
            "neutral_estimate_quality": {
                "score": 0.92,
                "accepted": true,
                "broad_support": true,
                "reason": "neutral estimate accepted with broad luminance-band support",
                "sample_count": 164,
                "sample_fraction": 0.08,
                "band_counts": [12, 128, 24],
                "populated_band_count": 3,
                "dominant_band_fraction": 0.780,
                "minimum_samples": 64,
                "minimum_bands": 2
            },
            "neutral_sample_rejections": {
                "total_pixels": 2048,
                "accepted_neutral_samples": 164,
                "non_finite": 0,
                "clipped": 3,
                "border": 12,
                "film_base_like_edge": 24,
                "dust": 2,
                "luma_out_of_range": 18,
                "chroma_threshold": 81
            },
            "dominant_anchor_sample_rejections": {
                "total_pixels": 2048,
                "accepted_anchor_samples": 258,
                "non_finite": 0,
                "clipped": 4,
                "border": 8,
                "film_base_like_edge": 5,
                "dust": 3,
                "luma_out_of_range": 12,
                "low_saturation": 64,
                "weak_dominance": 97
            },
            "dominant_anchor_quality": {
                "score": 0.67,
                "accepted": false,
                "reason": "dominant-channel anchors unstable: green count=0 bands=0 dominant_band=0.0% mean_margin=0.000",
                "channel_populated_band_count": [3, 0, 3],
                "channel_dominant_band_fraction": [0.50, 0.0, 0.55],
                "channel_mean_dominance_margin": [0.44, 0.0, 0.38],
                "channel_stability_score": [0.96, 0.0, 0.92],
                "channel_unstable": [false, true, false],
                "unstable_channel_count": 1,
                "minimum_samples_per_channel": 64,
                "minimum_bands": 2,
                "max_dominant_band_fraction": 0.92,
                "dominance_margin_target": 0.35
            },
            "reference_patch_evaluation": {
                "patch_count": 6,
                "selected_candidate": "neutral_balance_fallback",
                "image_derived_candidate": "image_derived_matrix",
                "selected_rms_error": 0.031,
                "selected_max_error": 0.060,
                "image_derived_rms_error": 0.020,
                "image_derived_max_error": 0.041,
                "selected_rms_delta_e": 3.4,
                "selected_max_delta_e": 6.2,
                "image_derived_rms_delta_e": 2.1,
                "image_derived_max_delta_e": 4.1,
                "selected_rms_delta_e2000": 2.9,
                "selected_max_delta_e2000": 5.2,
                "image_derived_rms_delta_e2000": 1.8,
                "image_derived_max_delta_e2000": 3.4,
                "rms_error_delta_vs_image_derived": 0.011,
                "max_error_delta_vs_image_derived": 0.019,
                "delta_e_rms_delta_vs_image_derived": 1.3,
                "delta_e_max_delta_vs_image_derived": 2.1,
                "delta_e2000_rms_delta_vs_image_derived": 1.1,
                "delta_e2000_max_delta_vs_image_derived": 1.8,
                "selected_improves_image_derived": false,
                "selected_regresses_image_derived": true,
                "worst_hue_families": [
                    { "hue_family": "red", "patch_count": 2, "rms_error": 0.05, "max_error": 0.06, "rms_delta_e": 5.4, "max_delta_e": 6.2 }
                ],
                "per_patch": [],
                "candidate_evaluations": [
                    {
                        "candidate": "calibrated_direct_profile",
                        "patch_count": 6,
                        "rms_error": 0.044,
                        "max_error": 0.081,
                        "mean_error": 0.037,
                        "rms_delta_e": 4.7,
                        "max_delta_e": 8.3,
                        "mean_delta_e": 3.6,
                        "rms_delta_e2000": 4.0,
                        "max_delta_e2000": 7.1,
                        "mean_delta_e2000": 3.0,
                        "rms_delta_vs_image_derived": 0.024,
                        "max_delta_vs_image_derived": 0.040,
                        "delta_e_rms_delta_vs_image_derived": 2.6,
                        "delta_e_max_delta_vs_image_derived": 4.2,
                        "delta_e2000_rms_delta_vs_image_derived": 2.2,
                        "delta_e2000_max_delta_vs_image_derived": 3.7,
                        "regresses_image_derived": true,
                        "worst_hue_families": [
                            { "hue_family": "red", "patch_count": 2, "rms_error": 0.070, "max_error": 0.081, "rms_delta_e": 7.8, "max_delta_e": 8.3 }
                        ],
                        "hue_family_regressions": [
                            {
                                "hue_family": "red",
                                "patch_count": 2,
                                "candidate_rms_error": 0.070,
                                "image_derived_rms_error": 0.020,
                                "rms_delta_vs_image_derived": 0.050,
                                "candidate_max_error": 0.081,
                                "image_derived_max_error": 0.041,
                                "max_delta_vs_image_derived": 0.040,
                                "candidate_rms_delta_e": 7.8,
                                "image_derived_rms_delta_e": 2.1,
                                "delta_e_rms_delta_vs_image_derived": 5.7,
                                "candidate_max_delta_e": 8.3,
                                "image_derived_max_delta_e": 4.1,
                                "delta_e_max_delta_vs_image_derived": 4.2
                            }
                        ]
                    }
                ]
            },
            "neutral_trim_before_after": {
                "applied": true,
                "scale": [0.98, 1.01, 1.0],
                "reason": "neutral trim applied: broad neutral support reduced measured neutral delta without increasing clipping",
                "before": {
                    "neutral_balance_delta": [0.02, -0.03, 0.01],
                    "neutral_delta_magnitude": 0.02,
                    "neutral_band_delta_magnitude": [0.03, 0.02, 0.025],
                    "pre_scale_clipped_low_ratio": [0.0, 0.0, 0.0],
                    "pre_scale_clipped_high_ratio": [0.0, 0.001, 0.0],
                    "pre_scale_preserved_ratio": 0.998,
                    "exposure_scale": 1.12
                },
                "after": {
                    "neutral_balance_delta": [0.004, -0.003, -0.001],
                    "neutral_delta_magnitude": 0.002667,
                    "neutral_band_delta_magnitude": [0.01, 0.002667, 0.009],
                    "pre_scale_clipped_low_ratio": [0.0, 0.0, 0.0],
                    "pre_scale_clipped_high_ratio": [0.0, 0.001, 0.0],
                    "pre_scale_preserved_ratio": 0.998,
                    "exposure_scale": 1.12
                },
                "neutral_delta_reduced": true,
                "neutral_band_delta_worsened": false,
                "low_clipping_increased": false,
                "high_clipping_increased": false,
                "preserved_ratio_decreased": false
            },
            "calibration_acceptance": {
                "status": "accepted",
                "reason": "calibrated profile beat image-derived score",
                "color_mode": "auto",
                "preferred_candidate": "calibrated_direct_profile",
                "preferred_candidate_quality_score": 0.31,
                "image_derived_quality_score": 0.88,
                "beats_image_derived": true,
                "within_negative_gamut_limits": true,
                "forced_by_color_mode": false
            },
            "selection_rejections": ["image_derived_matrix rejected: synthetic destructive clip"],
            "regularization_lambda": 0.20,
            "neutral_sample_bands": [12, 128, 24],
            "dominant_anchor_bands": [[12, 64, 52], [0, 0, 0], [16, 72, 42]],
            "neutral_trim_scale": [0.98, 1.01, 1.0],
            "neutral_trim_applied": true,
            "channel_anchor_counts": [128, 0, 130],
            "channel_anchor_min_count": 0,
            "channel_anchor_low_support": [false, true, false],
            "weak_anchor_fallback_used": true,
            "gamut_fallback_used": false,
            "image_matrix_pre_scale_clipped_low_ratio": [0.02, 0.00, 0.01],
            "image_matrix_pre_scale_clipped_high_ratio": [0.00, 0.03, 0.00],
            "image_matrix_exposure_scale": 1.31,
            "image_matrix_pre_scale_preserved_ratio": 0.97,
            "image_matrix_neutral_balance_delta": [0.01, -0.02, 0.01],
            "calibrated_profile_pre_scale_clipped_low_ratio": [0.01, 0.0, 0.0],
            "calibrated_profile_pre_scale_clipped_high_ratio": [0.0, 0.01, 0.0],
            "calibrated_profile_exposure_scale": 1.08,
            "calibrated_profile_pre_scale_preserved_ratio": 0.99,
            "calibrated_profile_neutral_balance_delta": [0.0, 0.0, 0.0],
            "pre_scale_preserved_ratio": 0.995,
            "post_scale_preserved_ratio": 0.999,
            "post_scale_clipped_high_ratio": [0.0, 0.001, 0.0],
            "post_scale_clipped_low_ratio": [0.0, 0.0, 0.0],
            "color_candidate_comparison_artifact": "output/phase46_color_candidate_comparison.tiff",
            "gamut_clipping_map_artifact": "output/phase46_gamut_clipping_map.tiff",
            "scene_referred_prophoto_float_artifact": "output/phase46_scene_referred_prophoto_float.tiff",
            "scene_referred_prophoto_float_artifact_diagnostics": {
                "encoding": "32-bit IEEE float RGB TIFF; linear ProPhoto RGB D50; scene-referred values outside [0,1] preserved without display normalization",
                "value_domain": "scene_referred_linear_prophoto_rgb_d50",
                "sample_format": "IEEEFP",
                "bits_per_sample": [32, 32, 32],
                "normalization": "none",
                "pixel_above_display_white_ratio": 0.02,
                "pixel_below_zero_ratio": 0.01
            },
            "gamut_clipping_map_diagnostics": {
                "encoding": "red=post-scale high clipping, blue=post-scale low clipping, green=in-gamut luminance; yellow/magenta indicates mixed high/low channel clipping",
                "preserved_ratio": 0.999,
                "any_clipped_ratio": 0.001
            }
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
            "highlight_chroma_compressed_ratio": 0.004,
            "highlight_neutral_chroma_compressed_ratio": 0.003,
            "highlight_neutral_chroma_enabled": true,
            "shadow_chroma_compressed_ratio": 0.002,
            "shadow_chroma_enabled": true,
            "color_protection_policy": "enabled",
            "color_trust_state": "trusted",
            "color_protection_reason": "enabled from colorspace candidate `neutral_balance_fallback` with quality score 0.420",
            "midtone_saturation_median": 0.18,
            "midtone_saturation_p95": 0.42,
            "render_luminance_percentiles": [0.04, 0.48, 0.94],
            "render_luminance_range_p05_p95": 0.90,
            "midtone_luminance_percentiles": [0.33, 0.48, 0.61],
            "bright_neutral_saturation_median": 0.04,
            "bright_neutral_saturation_p95": 0.08,
            "bright_saturated_saturation_median": 0.63,
            "bright_saturated_saturation_p95": 0.81,
            "post_chroma_compression_clipped_high_ratio": [0.0, 0.0, 0.002],
            "post_chroma_compression_clipped_low_ratio": [0.0, 0.001, 0.0],
            "high_frequency_grain": {
                "sample_count": 4096,
                "sample_stride": 2,
                "flat_luma_structure_max": 0.045,
                "flat_sample_count": 2048,
                "flat_sample_ratio": 0.5,
                "luma_residual_median": 0.001,
                "luma_residual_p95": 0.006,
                "chroma_residual_median": 0.002,
                "chroma_residual_p95": 0.009,
                "chroma_to_luma_p95_ratio": 1.5,
                "flat_luma_residual_p95": 0.004,
                "flat_chroma_residual_p95": 0.005,
                "flat_chroma_to_luma_p95_ratio": 1.25
            }
        }),
    ));
    report.add_phase(PhaseReport::ok(
        "save",
        1.0,
        serde_json::json!({
            "output_path": "output/output.tiff",
            "output_shape": [1200, 1800],
            "output_width": 1800,
            "output_height": 1200,
            "output_modified_at": "2026-05-01T21:02:00Z",
            "output_file_size_bytes": 12_345_678,
            "output_color_space": "linear_prophoto_rgb_d50",
            "output_icc_profile": {
                "embedded": true,
                "tiff_tag": 34675,
                "description": "ScanStitch linear ProPhoto RGB D50",
                "transfer_function": "linear",
                "whitepoint": "D50"
            },
            "input_base_confidence": 0.73,
            "render_review_status": "reviewable",
            "render_reviewable": true,
            "render_review_reason": "working-image base estimate is strong enough for normal visual review",
            "overwrote_existing_output": true,
            "stale_render_artifact_count": 1,
            "stale_render_artifacts": [
                {
                    "path": "output/phase46_prophoto.tiff",
                    "modified_at": "2026-02-28T20:00:00Z",
                    "size_bytes": 100
                }
            ]
        }),
    ));
    report
}

fn clear_stale_artifacts(report: &mut PipelineReport) {
    let save = report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "save")
        .expect("save phase");
    save.metrics["stale_render_artifact_count"] = serde_json::json!(0);
    save.metrics["stale_render_artifacts"] = serde_json::json!([]);
}

fn set_report_output_path(report: &mut PipelineReport, path: &Path) {
    let path = path.to_string_lossy().to_string();
    if let Some(metadata) = report.metadata.as_mut() {
        metadata.output_path = path.clone();
    }
    let save = report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "save")
        .expect("save phase");
    save.metrics["output_path"] = serde_json::json!(path);
}

fn write_report_color_debug_artifacts(report: &mut PipelineReport) -> tempfile::TempDir {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("output")).unwrap();
    std::fs::write(
        tmp.path()
            .join("output/phase46_color_candidate_comparison.tiff"),
        b"candidate-preview",
    )
    .unwrap();
    std::fs::write(
        tmp.path().join("output/phase46_gamut_clipping_map.tiff"),
        b"clipping-map",
    )
    .unwrap();
    std::fs::write(
        tmp.path()
            .join("output/phase46_scene_referred_prophoto_float.tiff"),
        b"scene-referred-float",
    )
    .unwrap();
    let metadata = report.metadata.as_mut().expect("metadata");
    metadata.working_directory = tmp.path().to_string_lossy().to_string();
    metadata.generated_at_unix_ms = 0;
    tmp
}

fn small_rgb_image() -> ndarray::Array3<f64> {
    let mut img = ndarray::Array3::<f64>::zeros((2, 2, 3));
    img[[0, 0, 0]] = 0.25;
    img[[0, 0, 1]] = 0.50;
    img[[0, 0, 2]] = 0.75;
    img
}

#[test]
fn test_validation_summary_extracts_standard_report_fields() {
    let report = fixture_report();
    let summary = scanstitch::validation::summarize_report("logan", &report);

    assert_eq!(summary.fixture, "logan");
    assert_eq!(
        summary.report.generated_at.as_deref(),
        Some("2026-05-01T21:00:00Z")
    );
    assert_eq!(summary.render.output_width, Some(1800));
    assert_eq!(summary.render.output_height, Some(1200));
    assert_eq!(
        summary.render.output_color_space.as_deref(),
        Some("linear_prophoto_rgb_d50")
    );
    assert_eq!(summary.render.output_icc_profile_embedded, Some(true));
    assert_eq!(
        summary.render.output_icc_profile_description.as_deref(),
        Some("ScanStitch linear ProPhoto RGB D50")
    );
    assert_eq!(summary.render.input_base_confidence, Some(0.73));
    assert_eq!(
        summary.render.render_review_status.as_deref(),
        Some("reviewable")
    );
    assert_eq!(summary.render.render_reviewable, Some(true));
    assert_eq!(summary.render.stale_render_artifact_count, Some(1));
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
    let seam = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .expect("seam exposure summary");
    assert_eq!(seam.applied, Some(true));
    assert_eq!(seam.gain_rgb, Some(vec![0.91, 0.91, 0.91]));
    assert_eq!(seam.seam_score_before, Some(0.102));
    assert_eq!(seam.seam_score_after, Some(0.018));
    assert_eq!(summary.base_density.base_confidence, Some(0.73));
    assert_eq!(summary.base_density.raw_base_proxy_confidence, Some(0.12));
    assert_eq!(summary.base_density.raw_base_support_fraction, Some(0.014));
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
    assert_eq!(
        summary.colorspace.calibration_status.as_deref(),
        Some("applied")
    );
    assert_eq!(
        summary
            .colorspace
            .calibration_external_profile_path
            .as_deref(),
        Some("docs/color-calibration-profile.example.json")
    );
    assert_eq!(
        summary
            .colorspace
            .calibration_scanner_profile_status
            .as_deref(),
        Some("applied")
    );
    assert_eq!(
        summary.colorspace.calibration_scanner_profile_id.as_deref(),
        Some("scanner-a")
    );
    assert_eq!(
        summary
            .colorspace
            .calibration_roll_profile_status
            .as_deref(),
        Some("applied")
    );
    assert_eq!(
        summary.colorspace.calibration_roll_profile_id.as_deref(),
        Some("roll-a")
    );
    assert_eq!(
        summary
            .colorspace
            .calibration_requested_film_stock
            .as_deref(),
        Some("Synthetic 200")
    );
    assert_eq!(
        summary.colorspace.calibration_film_stock_status.as_deref(),
        Some("matched")
    );
    assert_eq!(
        summary
            .colorspace
            .calibration_film_stock_matched_roll_profiles,
        vec!["roll-a".to_string()]
    );
    assert_eq!(summary.colorspace.post_scale_preserved_ratio, Some(0.999));
    assert_eq!(
        summary
            .colorspace
            .color_candidate_comparison_artifact
            .as_deref(),
        Some("output/phase46_color_candidate_comparison.tiff")
    );
    assert_eq!(
        summary.colorspace.gamut_clipping_map_artifact.as_deref(),
        Some("output/phase46_gamut_clipping_map.tiff")
    );
    assert_eq!(
        summary
            .colorspace
            .scene_referred_prophoto_float_artifact
            .as_deref(),
        Some("output/phase46_scene_referred_prophoto_float.tiff")
    );
    assert_eq!(
        summary.colorspace.gamut_clipping_map_encoding.as_deref(),
        Some("red=post-scale high clipping, blue=post-scale low clipping, green=in-gamut luminance; yellow/magenta indicates mixed high/low channel clipping")
    );
    assert_eq!(
        summary.colorspace.gamut_clipping_map_preserved_ratio,
        Some(0.999)
    );
    assert_eq!(
        summary.colorspace.gamut_clipping_map_any_clipped_ratio,
        Some(0.001)
    );
    assert_eq!(summary.colorspace.debug_artifacts.len(), 3);
    assert_eq!(
        summary.colorspace.debug_artifacts[0].kind,
        "candidate_comparison"
    );
    assert_eq!(
        summary.colorspace.selected_candidate.as_deref(),
        Some("neutral_balance_fallback")
    );
    assert_eq!(summary.colorspace.selected_candidate_rank, Some(1));
    assert_eq!(summary.colorspace.selected_quality_score, Some(0.42));
    assert_eq!(summary.colorspace.technical_safety_score, Some(0.12));
    assert_eq!(summary.colorspace.color_fidelity_score, Some(0.30));
    assert_eq!(
        summary.colorspace.candidate_risk.as_deref(),
        Some("review_reference_fit")
    );
    assert_eq!(
        summary.colorspace.tone_color_trust_state.as_deref(),
        Some("review_required")
    );
    assert_eq!(
        summary.render.colorspace_candidate_risk.as_deref(),
        Some("review_reference_fit")
    );
    assert_eq!(
        summary.render.colorspace_tone_color_trust_state.as_deref(),
        Some("review_required")
    );
    assert_eq!(
        summary.render.colorspace_density_monotonicity_score,
        Some(1.0)
    );
    assert_eq!(summary.render.colorspace_hue_linearity_score, Some(0.92));
    assert_eq!(
        summary
            .render
            .colorspace_saturation_preservation_median_ratio,
        Some(0.76)
    );
    assert_eq!(
        summary.render.colorspace_spatial_neutral_delta_p95,
        Some(0.006)
    );
    assert_eq!(summary.render.colorspace_memory_color_penalty, Some(0.06));
    assert_eq!(
        summary.render.colorspace_spatial_consistency_penalty,
        Some(0.07)
    );
    assert_eq!(
        summary.render.colorspace_reference_patch_rms_delta_e,
        Some(3.4)
    );
    assert_eq!(
        summary.render.colorspace_reference_patch_rms_delta_e2000,
        Some(2.9)
    );
    assert_eq!(
        summary.colorspace.selected_runner_up_quality_delta,
        Some(0.46)
    );
    assert_eq!(summary.colorspace.candidate_quality_scores.len(), 2);
    assert_eq!(summary.colorspace.candidate_quality_scores[0].rank, Some(2));
    assert_eq!(
        summary
            .colorspace
            .selected_quality_components
            .as_ref()
            .and_then(|components| components.fallback_penalty),
        Some(0.35)
    );
    assert_eq!(
        summary
            .colorspace
            .selected_quality_components
            .as_ref()
            .and_then(|components| components.memory_color_penalty),
        Some(0.06)
    );
    assert_eq!(
        summary
            .colorspace
            .neutral_sample_rejections
            .as_ref()
            .and_then(|rejections| rejections.film_base_like_edge),
        Some(24)
    );
    assert_eq!(
        summary
            .colorspace
            .dominant_anchor_sample_rejections
            .as_ref()
            .and_then(|rejections| rejections.accepted_anchor_samples),
        Some(258)
    );
    assert_eq!(
        summary.colorspace.dominant_anchor_bands,
        Some(vec![vec![12, 64, 52], vec![0, 0, 0], vec![16, 72, 42]])
    );
    assert_eq!(
        summary
            .colorspace
            .dominant_anchor_quality
            .as_ref()
            .and_then(|quality| quality.unstable_channel_count),
        Some(1)
    );
    assert_eq!(
        summary.colorspace.candidate_quality_scores[0]
            .quality_components
            .as_ref()
            .and_then(|components| components.anchor_stability_penalty),
        Some(0.25)
    );
    assert_eq!(
        summary.colorspace.candidate_quality_scores[0].reference_patch_rms_delta_e,
        Some(3.2)
    );
    assert_eq!(
        summary.colorspace.candidate_quality_scores[0].reference_patch_rms_delta_e2000,
        Some(2.7)
    );
    assert_eq!(
        summary.colorspace.candidate_quality_scores[0].reference_patch_max_delta_vs_image_derived,
        Some(0.2)
    );
    assert_eq!(
        summary.colorspace.candidate_quality_scores[0]
            .reference_patch_delta_e_max_delta_vs_image_derived,
        Some(1.6)
    );
    assert_eq!(
        summary.colorspace.candidate_quality_scores[0]
            .reference_patch_delta_e2000_delta_vs_image_derived,
        Some(0.8)
    );
    assert_eq!(
        summary.colorspace.candidate_quality_scores[0]
            .reference_patch_delta_e2000_max_delta_vs_image_derived,
        Some(1.3)
    );
    assert_eq!(
        summary.colorspace.candidate_quality_scores[0].density_monotonicity_score,
        Some(0.95)
    );
    assert_eq!(
        summary.colorspace.candidate_quality_scores[0].spatial_neutral_delta_p95,
        Some(0.045)
    );
    assert_eq!(
        summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.selected_regresses_image_derived),
        Some(true)
    );
    assert_eq!(
        summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.selected_rms_delta_e),
        Some(3.4)
    );
    assert_eq!(
        summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.selected_rms_delta_e2000),
        Some(2.9)
    );
    assert_eq!(
        summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.max_error_delta_vs_image_derived),
        Some(0.019)
    );
    assert_eq!(
        summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.delta_e_rms_delta_vs_image_derived),
        Some(1.3)
    );
    assert_eq!(
        summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.delta_e_max_delta_vs_image_derived),
        Some(2.1)
    );
    assert_eq!(
        summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.delta_e2000_rms_delta_vs_image_derived),
        Some(1.1)
    );
    assert_eq!(
        summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .and_then(|evaluation| evaluation.delta_e2000_max_delta_vs_image_derived),
        Some(1.8)
    );
    assert_eq!(
        summary
            .colorspace
            .reference_patch_evaluation
            .as_ref()
            .map(|evaluation| evaluation.hue_family_regressions.clone()),
        Some(vec!["red".to_string()])
    );
    assert_eq!(
        summary.colorspace.candidate_acceptance[0].status.as_deref(),
        Some("rejected_safety")
    );
    assert_eq!(
        summary.colorspace.candidate_acceptance[0]
            .mapping_strategy
            .as_deref(),
        Some("image_derived_matrix")
    );
    assert_eq!(
        summary.colorspace.candidate_acceptance[0]
            .source_label
            .as_deref(),
        Some("image-derived")
    );
    assert_eq!(
        summary.colorspace.candidate_acceptance[0].eligible_in_color_mode,
        Some(true)
    );
    assert_eq!(
        summary
            .colorspace
            .neutral_estimate_quality
            .as_ref()
            .and_then(|quality| quality.accepted),
        Some(true)
    );
    assert_eq!(
        summary
            .colorspace
            .calibration_acceptance
            .as_ref()
            .and_then(|acceptance| acceptance.status.as_deref()),
        Some("accepted")
    );
    assert_eq!(summary.colorspace.neutral_trim_applied, Some(true));
    assert_eq!(
        summary.colorspace.neutral_trim_scale,
        Some(vec![0.98, 1.01, 1.0])
    );
    assert_eq!(
        summary
            .colorspace
            .neutral_trim_before_after
            .as_ref()
            .and_then(|trim| trim.low_clipping_increased),
        Some(false)
    );
    assert_eq!(
        summary
            .colorspace
            .neutral_trim_before_after
            .as_ref()
            .and_then(|trim| trim.neutral_band_delta_worsened),
        Some(false)
    );
    assert_eq!(
        summary.colorspace.calibrated_profile_exposure_scale,
        Some(1.08)
    );
    assert_eq!(
        summary.tone.render_luminance_percentiles,
        Some(vec![0.04, 0.48, 0.94])
    );
    assert_eq!(summary.tone.render_luminance_range_p05_p95, Some(0.90));
    assert_eq!(summary.tone.bright_neutral_saturation_p95, Some(0.08));
    assert_eq!(
        summary.tone.color_protection_policy.as_deref(),
        Some("enabled")
    );
    assert_eq!(summary.tone.color_trust_state.as_deref(), Some("trusted"));
    assert_eq!(
        summary
            .tone
            .high_frequency_grain
            .as_ref()
            .and_then(|grain| grain.luma_residual_p95),
        Some(0.006)
    );
    assert_eq!(
        summary
            .tone
            .high_frequency_grain
            .as_ref()
            .and_then(|grain| grain.flat_sample_ratio),
        Some(0.5)
    );
    assert_eq!(
        summary
            .tone
            .high_frequency_grain
            .as_ref()
            .and_then(|grain| grain.flat_chroma_residual_p95),
        Some(0.005)
    );
    assert_eq!(
        summary.tone.post_chroma_compression_clipped_high_ratio,
        Some(vec![0.0, 0.0, 0.002])
    );
    assert_eq!(summary.warnings.len(), 1);
}

#[test]
fn test_validation_summary_inspects_color_debug_artifacts() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("output")).unwrap();
    std::fs::write(
        tmp.path()
            .join("output/phase46_color_candidate_comparison.tiff"),
        b"candidate-preview",
    )
    .unwrap();
    std::fs::write(
        tmp.path().join("output/phase46_gamut_clipping_map.tiff"),
        b"clipping-map",
    )
    .unwrap();
    std::fs::write(
        tmp.path()
            .join("output/phase46_scene_referred_prophoto_float.tiff"),
        b"scene-referred-float",
    )
    .unwrap();

    let mut report = fixture_report();
    let metadata = report.metadata.as_mut().expect("metadata");
    metadata.working_directory = tmp.path().to_string_lossy().to_string();
    metadata.generated_at_unix_ms = 0;

    let summary = scanstitch::validation::summarize_report("logan", &report);
    let artifacts = &summary.colorspace.debug_artifacts;
    assert_eq!(artifacts.len(), 3);
    assert_eq!(artifacts[0].kind, "candidate_comparison");
    assert_eq!(artifacts[0].status, "fresh");
    assert!(artifacts[0].exists);
    assert_eq!(artifacts[0].fresh_for_report, Some(true));
    assert_eq!(
        artifacts[0].file_size_bytes,
        Some("candidate-preview".len() as u64)
    );
    assert!(artifacts[0].modified_at_unix_ms.is_some());
    assert_eq!(artifacts[1].kind, "gamut_clipping_map");
    assert_eq!(artifacts[1].status, "fresh");
    assert_eq!(artifacts[2].kind, "scene_referred_prophoto_float");
    assert_eq!(artifacts[2].status, "fresh");
}

#[test]
fn test_summary_baseline_flags_invalid_named_color_debug_artifacts() {
    let report = fixture_report();
    let summary = scanstitch::validation::summarize_report("logan", &report);
    let baseline = scanstitch::validation::tracked_baseline_from_summary(&summary);

    let comparison =
        scanstitch::validation::compare_summary_baseline("baseline.json", &baseline, &summary);

    assert_eq!(comparison.status, "review_required");
    assert_eq!(comparison.colorspace_debug_artifact_invalid_count, 3);
    assert_eq!(
        comparison.colorspace_debug_artifact_issues,
        vec![
            "colorspace_debug_artifact_invalid:candidate_comparison:missing".to_string(),
            "colorspace_debug_artifact_invalid:gamut_clipping_map:missing".to_string(),
            "colorspace_debug_artifact_invalid:scene_referred_prophoto_float:missing".to_string()
        ]
    );
    assert!(comparison
        .issues
        .contains(&"colorspace_debug_artifact_invalid:candidate_comparison:missing".to_string()));
}

#[test]
fn test_tracked_baseline_from_summary_exports_compact_regression_contract() {
    let mut report = fixture_report();
    let _debug_artifacts = write_report_color_debug_artifacts(&mut report);
    let summary = scanstitch::validation::summarize_report("logan", &report);
    let baseline = scanstitch::validation::tracked_baseline_from_summary(&summary);

    assert_eq!(baseline.fixture.as_deref(), Some("logan"));
    assert_eq!(baseline.stitch.decision.as_deref(), Some("accepted"));
    assert_eq!(baseline.stitch.confidence, Some(0.82));
    assert_eq!(baseline.stitch.seam_exposure_correction_applied, Some(true));
    assert_eq!(baseline.render.output_width, Some(1800));
    assert_eq!(baseline.render.output_height, Some(1200));
    assert_eq!(baseline.render.raw_base_proxy_confidence, Some(0.12));
    assert_eq!(baseline.render.raw_base_support_fraction, Some(0.014));
    assert_eq!(
        baseline.render.colorspace_mapping_strategy.as_deref(),
        Some("neutral_balance_weak_anchor_fallback")
    );
    assert_eq!(
        baseline.colorspace.candidate_risk.as_deref(),
        Some("review_reference_fit")
    );
    assert_eq!(baseline.colorspace.selected_candidate_rank, Some(1));
    assert_eq!(baseline.colorspace.technical_safety_score, Some(0.12));
    assert_eq!(baseline.colorspace.color_fidelity_score, Some(0.30));
    assert_eq!(baseline.colorspace.calibration_confidence, Some(0.95));
    assert_eq!(
        baseline.colorspace.calibration_matrix_condition_number,
        Some(2.1)
    );
    assert_eq!(
        baseline
            .colorspace
            .calibration_scanner_profile_status
            .as_deref(),
        Some("applied")
    );
    assert_eq!(
        baseline
            .colorspace
            .calibration_scanner_profile_id
            .as_deref(),
        Some("scanner-a")
    );
    assert_eq!(
        baseline
            .colorspace
            .calibration_roll_profile_status
            .as_deref(),
        Some("applied")
    );
    assert_eq!(
        baseline.colorspace.calibration_roll_profile_id.as_deref(),
        Some("roll-a")
    );
    assert!(baseline.colorspace.calibration_rejection_details.is_empty());
    assert_eq!(
        baseline.colorspace.selected_runner_up_quality_delta,
        Some(0.46)
    );
    assert_eq!(baseline.colorspace.density_monotonicity_score, Some(1.0));
    assert_eq!(baseline.colorspace.hue_linearity_score, Some(0.92));
    assert_eq!(
        baseline.colorspace.saturation_preservation_median_ratio,
        Some(0.76)
    );
    assert_eq!(baseline.colorspace.spatial_neutral_delta_p95, Some(0.006));
    assert_eq!(baseline.colorspace.memory_color_penalty, Some(0.06));
    assert_eq!(baseline.colorspace.spatial_consistency_penalty, Some(0.07));
    assert_eq!(
        baseline
            .colorspace
            .calibration_acceptance_preferred_candidate
            .as_deref(),
        Some("calibrated_direct_profile")
    );
    assert_eq!(
        baseline
            .colorspace
            .calibration_acceptance_beats_image_derived,
        Some(true)
    );
    assert_eq!(
        baseline
            .colorspace
            .calibration_requested_film_stock
            .as_deref(),
        Some("Synthetic 200")
    );
    assert_eq!(
        baseline.colorspace.calibration_film_stock_status.as_deref(),
        Some("matched")
    );
    assert_eq!(
        baseline
            .colorspace
            .calibration_film_stock_matched_roll_profiles,
        vec!["roll-a".to_string()]
    );
    assert_eq!(
        baseline.colorspace.reference_patch_selected_rms_delta_e,
        Some(3.4)
    );
    assert_eq!(
        baseline.colorspace.reference_patch_selected_rms_delta_e2000,
        Some(2.9)
    );
    assert_eq!(
        baseline
            .colorspace
            .reference_patch_max_error_delta_vs_image_derived,
        Some(0.019)
    );
    assert_eq!(
        baseline
            .colorspace
            .reference_patch_delta_e_rms_delta_vs_image_derived,
        Some(1.3)
    );
    assert_eq!(
        baseline
            .colorspace
            .reference_patch_delta_e_max_delta_vs_image_derived,
        Some(2.1)
    );
    assert_eq!(
        baseline
            .colorspace
            .reference_patch_delta_e2000_rms_delta_vs_image_derived,
        Some(1.1)
    );
    assert_eq!(
        baseline
            .colorspace
            .reference_patch_delta_e2000_max_delta_vs_image_derived,
        Some(1.8)
    );
    assert_eq!(
        baseline
            .colorspace
            .reference_patch_selected_regresses_image_derived,
        Some(true)
    );
    assert_eq!(
        baseline.colorspace.reference_patch_worst_hue_families,
        vec!["red".to_string()]
    );
    assert_eq!(
        baseline.colorspace.reference_patch_hue_family_regressions,
        vec!["red".to_string()]
    );
    assert_eq!(
        baseline.colorspace.reference_patch_regressed_candidates,
        vec!["calibrated_direct_profile".to_string()]
    );
    assert_eq!(baseline.colorspace.neutral_estimate_score, Some(0.92));
    assert_eq!(baseline.colorspace.neutral_estimate_accepted, Some(true));
    assert_eq!(
        baseline.colorspace.neutral_estimate_populated_band_count,
        Some(3)
    );
    assert_eq!(
        baseline.colorspace.neutral_estimate_dominant_band_fraction,
        Some(0.78)
    );
    assert_eq!(baseline.colorspace.dominant_anchor_score, Some(0.67));
    assert_eq!(baseline.colorspace.dominant_anchor_accepted, Some(false));
    assert_eq!(
        baseline.colorspace.dominant_anchor_unstable_channel_count,
        Some(1)
    );
    assert_eq!(
        baseline.colorspace.dominant_anchor_channel_unstable,
        vec![false, true, false]
    );
    assert_eq!(baseline.colorspace.channel_anchor_min_count, Some(0));
    assert_eq!(
        baseline.colorspace.channel_anchor_low_support,
        vec![false, true, false]
    );
    assert_eq!(baseline.colorspace.weak_anchor_fallback_used, Some(true));
    assert_eq!(baseline.colorspace.gamut_fallback_used, Some(false));
    assert_eq!(baseline.colorspace.neutral_trim_applied, Some(true));
    assert_eq!(
        baseline.colorspace.candidate_acceptance_signatures,
        vec![
            "image_derived_matrix|kind=image_derived|strategy=image_derived_matrix|status=rejected_safety|rank=2|selected=false|eligible=true|rejected=true".to_string(),
            "neutral_balance_fallback|kind=neutral_fallback|strategy=neutral_balance_weak_anchor_fallback|status=selected|rank=1|selected=true|eligible=true|rejected=false".to_string(),
        ]
    );
    assert_eq!(baseline.tone.highlight_chroma_compressed_ratio, Some(0.004));
    assert_eq!(baseline.grain.chroma_to_luma_p95_ratio, Some(1.5));

    let comparison =
        scanstitch::validation::compare_summary_baseline("baseline.json", &baseline, &summary);
    assert_eq!(comparison.status, "comparable");
    assert!(comparison.issues.is_empty());
}

#[test]
fn test_validation_summary_markdown_is_compact() {
    let report = fixture_report();
    let summary = scanstitch::validation::summarize_report("logan", &report);
    let markdown = scanstitch::validation::summary_to_markdown(&summary);

    assert!(markdown.contains("# Validation Summary: logan"));
    assert!(markdown.contains("generated_at"));
    assert!(markdown.contains("output_dimensions"));
    assert!(markdown.contains("seam_exposure_gain_rgb"));
    assert!(markdown.contains("calibration_status"));
    assert!(markdown.contains("selected_candidate"));
    assert!(markdown.contains("selected_quality_score"));
    assert!(markdown.contains("technical_safety_score"));
    assert!(markdown.contains("color_fidelity_score"));
    assert!(markdown.contains("selected_quality_components"));
    assert!(markdown.contains("fallback=0.350000"));
    assert!(markdown.contains("selected_hue_linearity_score"));
    assert!(markdown.contains("selected_memory_color_penalty"));
    assert!(markdown.contains("selected_spatial_consistency_penalty"));
    assert!(markdown.contains("candidate_risk"));
    assert!(markdown.contains("tone_color_trust_state"));
    assert!(markdown.contains("selected_runner_up_quality_delta"));
    assert!(markdown.contains("candidate_acceptance"));
    assert!(markdown.contains("strategy=image_derived_matrix"));
    assert!(markdown.contains("eligible=true"));
    assert!(markdown.contains("neutral_estimate_quality"));
    assert!(markdown.contains("neutral_sample_rejections"));
    assert!(markdown.contains("dominant_anchor_sample_rejections"));
    assert!(markdown.contains("hue_regressions=red"));
    assert!(markdown.contains("reference_patch_fit"));
    assert!(markdown.contains("delta_e2000_rms"));
    assert!(markdown.contains("color_candidate_comparison_artifact"));
    assert!(markdown.contains("gamut_clipping_map_artifact"));
    assert!(markdown.contains("scene_referred_prophoto_float_artifact"));
    assert!(markdown.contains("debug_artifacts"));
    assert!(markdown.contains("neutral_trim_scale"));
    assert!(markdown.contains("neutral_trim_delta_before"));
    assert!(markdown.contains("color_protection_policy"));
    assert!(markdown.contains("color_trust_state"));
    assert!(markdown.contains("render_luminance_range_p05_p95"));
    assert!(markdown.contains("high_frequency_luma_residual_p95"));
    assert!(markdown.contains("high_frequency_flat_chroma_residual_p95"));
    assert!(markdown.contains("neutral_balance_weak_anchor_fallback"));
    assert!(markdown.contains("bright_neutral_saturation_p95"));
    assert!(markdown.contains("colorspace matrix has weak dominant-channel anchor support"));
}

#[test]
fn test_synthetic_color_suite_proves_required_decision_edges() {
    let suite = scanstitch::validation::run_synthetic_color_suite();

    assert_eq!(suite.status, "passed", "issues: {:?}", suite.issues);
    assert_eq!(suite.cases.len(), 19);
    assert!(suite.cases.iter().any(|case| {
        case.name == "calibrated_profile_beats_image_derived"
            && case.actual_selected_candidate.as_deref() == Some("calibrated_direct_profile")
            && case.actual_calibration_acceptance_status.as_deref() == Some("accepted")
            && case.actual_calibration_beats_image_derived == Some(true)
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "weak_calibration_rejected_quality"
            && case.actual_selected_candidate.as_deref() == Some("image_derived_matrix")
            && case.actual_calibration_acceptance_status.as_deref() == Some("rejected_quality")
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "calibration_neutral_regression_rejected"
            && case.actual_selected_candidate.as_deref() == Some("image_derived_matrix")
            && case.actual_mapping_strategy.as_deref() == Some("image_derived_matrix")
            && case.actual_calibration_acceptance_status.as_deref() == Some("rejected_neutral")
            && case.actual_calibration_beats_image_derived == Some(true)
            && case.actual_calibrated_candidate_status.as_deref() == Some("rejected_neutral")
            && case.actual_candidate_risk.as_deref() == Some("review_neutral_support")
            && case.actual_tone_color_trust_state.as_deref() == Some("review_required")
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "unsafe_calibration_rejected_auto"
            && case.actual_selected_candidate.as_deref() == Some("gamut_trusted_image_matrix_blend")
            && case.actual_mapping_strategy.as_deref() == Some("gamut_trusted_image_matrix_blend")
            && case.actual_calibration_acceptance_status.as_deref() == Some("rejected_unsafe")
            && case.actual_calibration_beats_image_derived == Some(false)
            && case.actual_calibrated_candidate_status.as_deref() == Some("rejected_safety")
            && case
                .actual_preferred_calibration_candidate_status
                .as_deref()
                == Some("rejected_safety")
            && case.actual_gamut_fallback_used == Some(false)
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "scanner_prior_beats_weak_anchor_image_candidate"
            && case.actual_selected_candidate.as_deref() == Some("scanner_prior_image_adaptation")
            && case.actual_mapping_strategy.as_deref()
                == Some("scanner_constrained_image_derived_matrix")
            && case.actual_calibration_acceptance_status.as_deref() == Some("accepted")
            && case.actual_calibration_beats_image_derived == Some(true)
            && case
                .actual_preferred_calibration_candidate_status
                .as_deref()
                == Some("selected")
            && case.actual_channel_anchor_low_support == Some([false, true, false])
            && case.actual_gamut_fallback_used == Some(false)
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "unsafe_scanner_prior_rejected_auto"
            && case.actual_selected_candidate.as_deref() == Some("gamut_trusted_image_matrix_blend")
            && case.actual_mapping_strategy.as_deref() == Some("gamut_trusted_image_matrix_blend")
            && case.actual_calibration_acceptance_status.as_deref() == Some("rejected_unsafe")
            && case.actual_calibration_beats_image_derived == Some(false)
            && case
                .actual_preferred_calibration_candidate_status
                .as_deref()
                == Some("rejected_safety")
            && case.actual_gamut_fallback_used == Some(false)
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "sparse_uncalibrated_anchor_review_required"
            && case.actual_selected_candidate.as_deref() == Some("image_derived_matrix")
            && case.actual_candidate_risk.as_deref() == Some("review_neutral_support")
            && case.actual_tone_color_trust_state.as_deref() == Some("review_required")
            && case.actual_neutral_estimate_accepted == Some(false)
            && case.actual_dominant_anchor_accepted == Some(false)
            && case.actual_channel_anchor_low_support == Some([false, true, false])
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "dirty_edge_samples_rejected"
            && case.actual_selected_candidate.as_deref() == Some("gamut_trusted_image_matrix_blend")
            && case.actual_neutral_estimate_accepted == Some(true)
            && case
                .actual_neutral_sample_rejections
                .as_ref()
                .is_some_and(|rejections| {
                    rejections.clipped.unwrap_or(0) >= 1
                        && rejections.dust.unwrap_or(0) >= 1
                        && rejections.film_base_like_edge.unwrap_or(0) >= 1
                })
            && case
                .actual_dominant_anchor_sample_rejections
                .as_ref()
                .is_some_and(|rejections| {
                    rejections.clipped.unwrap_or(0) >= 1
                        && rejections.border.unwrap_or(0) >= 1
                        && rejections.dust.unwrap_or(0) >= 1
                        && rejections.film_base_like_edge.unwrap_or(0) >= 1
                })
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "biased_scene_anchor_review_required"
            && case.actual_selected_candidate.as_deref() == Some("gamut_trusted_image_matrix_blend")
            && matches!(
                case.actual_candidate_risk.as_deref(),
                Some("review_anchor_support" | "review_gamut")
            )
            && case.actual_tone_color_trust_state.as_deref() == Some("review_required")
            && case.actual_neutral_estimate_accepted == Some(true)
            && case.actual_dominant_anchor_accepted == Some(false)
            && case.actual_dominant_anchor_channel_populated_band_count == Some([1, 1, 1])
            && case.actual_dominant_anchor_channel_unstable == Some([true, true, true])
    }));
    let selected_candidate_cases = suite
        .cases
        .iter()
        .filter(|case| case.actual_selected_candidate.is_some())
        .collect::<Vec<_>>();
    assert_eq!(selected_candidate_cases.len(), 12);
    assert!(selected_candidate_cases.iter().all(|case| {
        case.actual_density_monotonicity_score.is_some()
            && case.actual_hue_linearity_score.is_some()
            && case.actual_memory_color_penalty.is_some()
            && case.actual_spatial_consistency_penalty.is_some()
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "biased_scene_anchor_review_required"
            && case
                .actual_density_monotonicity_score
                .is_some_and(|score| (0.0..=1.0).contains(&score))
            && case
                .actual_hue_linearity_score
                .is_some_and(|score| score > 0.90)
            && case
                .actual_saturation_preservation_median_ratio
                .is_some_and(|ratio| ratio >= 1.0)
            && case
                .actual_spatial_neutral_delta_p95
                .is_some_and(|delta| delta >= 0.0)
            && case
                .actual_memory_color_penalty
                .is_some_and(|penalty| penalty >= 0.0)
            && case
                .actual_spatial_consistency_penalty
                .is_some_and(|penalty| penalty >= 0.0)
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "high_key_exposure_normalization_preserves_headroom"
            && case.actual_selected_candidate.as_deref() == Some("image_derived_matrix")
            && case.actual_exposure_scale.is_some_and(|scale| scale > 1.0)
            && case.actual_post_scale_high_clip_less_than_pre_scale == Some(true)
            && case
                .actual_post_scale_preserved_ratio
                .is_some_and(|ratio| ratio >= 0.99)
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "destructive_gamut_fallback_preserves_output"
            && case.actual_selected_candidate.as_deref() == Some("gamut_safe_image_matrix_blend")
            && case.actual_mapping_strategy.as_deref() == Some("gamut_safe_image_matrix_blend")
            && case.actual_candidate_risk.as_deref() == Some("review_neutral_support")
            && case.actual_gamut_fallback_used == Some(false)
            && case
                .actual_image_matrix_pre_scale_low_clip_total
                .is_some_and(|ratio| ratio > 0.18)
            && case
                .actual_selected_pre_scale_low_clip_total
                .is_some_and(|ratio| ratio < 1e-9)
            && case
                .actual_post_scale_low_clip_total
                .is_some_and(|ratio| ratio < 1e-9)
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "direct_density_render_fallback_for_destructive_ica_gamut"
            && case.actual_render_input_source.as_deref() == Some("direct_density_transmittance")
            && case.actual_render_input_fallback_used == Some(true)
            && case
                .actual_render_input_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("reduced the image-matrix low clipping"))
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "direct_density_render_fallback_rejects_neutral_regression"
            && case.actual_render_input_source.as_deref() == Some("fastica_separated_transmittance")
            && case.actual_render_input_fallback_used == Some(false)
            && case
                .actual_render_input_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("retained"))
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "direct_density_render_fallback_for_quality_win"
            && case.actual_render_input_source.as_deref() == Some("direct_density_transmittance")
            && case.actual_render_input_fallback_used == Some(true)
            && case
                .actual_render_input_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("materially stronger colorspace candidate"))
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "direct_density_render_fallback_rejects_worse_risk"
            && case.actual_render_input_source.as_deref() == Some("fastica_separated_transmittance")
            && case.actual_render_input_fallback_used == Some(false)
            && case
                .actual_render_input_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("retained"))
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "tone_policy_weak_neutral_keeps_bounded_neutral_cleanup"
            && case.actual_tone_policy.as_deref() == Some("weak_neutral_bounded_neutral_cleanup")
            && case.actual_tone_policy_color_trust_state.as_deref() == Some("limited_weak_neutral")
            && case.actual_tone_highlight_neutral_chroma_enabled == Some(true)
            && case.actual_tone_shadow_chroma_enabled == Some(true)
            && case
                .actual_tone_highlight_neutral_chroma_compressed_ratio
                .is_some_and(|ratio| ratio > 0.0)
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "tone_policy_color_review_disables_color_cleanup_preserves_gamut_repair"
            && case.actual_tone_policy.as_deref() == Some("disabled_color_candidate_review")
            && case.actual_tone_policy_color_trust_state.as_deref() == Some("review_required")
            && case.actual_tone_highlight_neutral_chroma_enabled == Some(false)
            && case.actual_tone_shadow_chroma_enabled == Some(false)
            && case
                .actual_tone_highlight_chroma_compressed_ratio
                .is_some_and(|ratio| ratio > 0.0)
            && case.actual_tone_highlight_neutral_chroma_compressed_ratio == Some(0.0)
            && case.actual_tone_shadow_chroma_compressed_ratio == Some(0.0)
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "reference_patch_regression_rejected"
            && case.actual_candidate_risk.as_deref() == Some("review_reference_fit")
            && case.actual_calibration_acceptance_status.as_deref()
                == Some("rejected_reference_fit")
            && case.actual_calibrated_candidate_status.as_deref() == Some("rejected_reference_fit")
            && case.actual_reference_patch_regressed_candidates
                == vec![
                    "gamut_safe_image_matrix_blend".to_string(),
                    "calibrated_direct_profile".to_string(),
                ]
            && !case
                .actual_reference_patch_hue_family_regressions
                .is_empty()
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "forced_unsafe_calibration_fails"
            && case
                .actual_error
                .as_deref()
                .is_some_and(|err| err.contains("negative channel values"))
    }));
}

#[test]
fn test_validate_cli_runs_synthetic_color_suite_strict() {
    let tmp = tempfile::TempDir::new().unwrap();
    let summary_json = tmp.path().join("synthetic-color-suite.json");
    let summary_md = tmp.path().join("synthetic-color-suite.md");

    let status = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--synthetic-color-suite")
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .status()
        .expect("run scanstitch-validate synthetic suite");

    assert!(status.success(), "synthetic suite CLI failed: {status}");
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["status"], "passed");
    assert_eq!(summary["cases"].as_array().map(Vec::len), Some(19));
    let cases = summary["cases"].as_array().expect("synthetic cases");
    assert!(cases
        .iter()
        .any(|case| case["actual_density_monotonicity_score"].is_number()));
    assert!(cases
        .iter()
        .any(|case| case["actual_memory_color_penalty"].is_number()));
    assert!(cases
        .iter()
        .any(|case| case["actual_spatial_consistency_penalty"].is_number()));
    let markdown = std::fs::read_to_string(&summary_md).unwrap();
    assert!(markdown.contains("calibrated_profile_beats_image_derived"));
    assert!(markdown.contains("calibration_neutral_regression_rejected"));
    assert!(markdown.contains("rejected_neutral"));
    assert!(markdown.contains("unsafe_calibration_rejected_auto"));
    assert!(markdown.contains("rejected_unsafe"));
    assert!(markdown.contains("scanner_prior_beats_weak_anchor_image_candidate"));
    assert!(markdown.contains("scanner_constrained_image_derived_matrix"));
    assert!(markdown.contains("unsafe_scanner_prior_rejected_auto"));
    assert!(markdown.contains("sparse_uncalibrated_anchor_review_required"));
    assert!(markdown.contains("dirty_edge_samples_rejected"));
    assert!(markdown.contains("biased_scene_anchor_review_required"));
    assert!(markdown.contains("Anchor stability"));
    assert!(markdown.contains("high_key_exposure_normalization_preserves_headroom"));
    assert!(markdown.contains("high_clip"));
    assert!(markdown.contains("destructive_gamut_fallback_preserves_output"));
    assert!(markdown.contains("gamut_safe_image_matrix_blend"));
    assert!(markdown.contains("image_low"));
    assert!(markdown.contains("Model"));
    assert!(markdown.contains("memory_penalty"));
    assert!(markdown.contains("spatial_penalty"));
    assert!(markdown.contains("Render input"));
    assert!(markdown.contains("direct_density_render_fallback_for_destructive_ica_gamut"));
    assert!(markdown.contains("direct_density_transmittance"));
    assert!(markdown.contains("direct_density_render_fallback_rejects_neutral_regression"));
    assert!(markdown.contains("direct_density_render_fallback_for_quality_win"));
    assert!(markdown.contains("direct_density_render_fallback_rejects_worse_risk"));
    assert!(markdown.contains("fastica_separated_transmittance"));
    assert!(markdown.contains("Tone policy"));
    assert!(markdown.contains("tone_policy_weak_neutral_keeps_bounded_neutral_cleanup"));
    assert!(markdown.contains("weak_neutral_bounded_neutral_cleanup"));
    assert!(
        markdown.contains("tone_policy_color_review_disables_color_cleanup_preserves_gamut_repair")
    );
    assert!(markdown.contains("disabled_color_candidate_review"));
    assert!(markdown.contains("Reference"));
    assert!(markdown.contains("regressed gamut_safe_image_matrix_blend"));
    assert!(markdown.contains("calibrated_direct_profile"));
    assert!(markdown.contains("hue_regressions"));
    assert!(markdown.contains("unstable [true, true, true]"));
    assert!(markdown.contains("Rejected samples"));
    assert!(markdown.contains("review_neutral_support"));
    assert!(markdown.contains("forced_unsafe_calibration_fails"));
}

#[test]
fn test_validate_cli_uses_fixture_registry_summary_baseline() {
    let tmp = tempfile::TempDir::new().unwrap();
    let report_path = tmp.path().join("report.json");
    let baseline_path = tmp.path().join("baseline.json");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("summary.json");
    let summary_md = tmp.path().join("summary.md");

    let output_path = tmp.path().join("output.tiff");
    scanstitch::tiff_io::save_tiff_f64_linear_prophoto(&small_rgb_image(), &output_path).unwrap();
    let mut report = fixture_report();
    set_report_output_path(&mut report, &output_path);
    let _debug_artifacts = write_report_color_debug_artifacts(&mut report);
    report.save(&report_path).unwrap();
    let mut baseline = fixture_summary_baseline();
    baseline.fixture = Some("registry-fixture".to_string());
    std::fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&baseline).unwrap(),
    )
    .unwrap();
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "fixtures": {
                "registry-fixture": {
                    "component1": "missing-left.tif",
                    "component2": "missing-right.tif",
                    "output_dir": tmp.path().join("out"),
                    "summary_baseline": baseline_path
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture")
        .arg("registry-fixture")
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--report")
        .arg(&report_path)
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .status()
        .expect("run scanstitch-validate with registry baseline");

    assert!(status.success(), "registry baseline CLI failed: {status}");
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(
        summary["summary_baseline_comparison"]["status"],
        "comparable"
    );
    assert_eq!(
        summary["summary_baseline_comparison"]["baseline_summary_path"].as_str(),
        Some(baseline_path.to_string_lossy().as_ref())
    );
}

#[test]
fn test_validate_cli_rejects_unknown_summary_baseline_fields() {
    let tmp = tempfile::TempDir::new().unwrap();
    let report_path = tmp.path().join("report.json");
    let baseline_path = tmp.path().join("baseline.json");
    let summary_json = tmp.path().join("summary.json");
    let summary_md = tmp.path().join("summary.md");

    let output_path = tmp.path().join("output.tiff");
    scanstitch::tiff_io::save_tiff_f64_linear_prophoto(&small_rgb_image(), &output_path).unwrap();
    let mut report = fixture_report();
    set_report_output_path(&mut report, &output_path);
    let _debug_artifacts = write_report_color_debug_artifacts(&mut report);
    report.save(&report_path).unwrap();

    let mut baseline = serde_json::to_value(fixture_summary_baseline()).unwrap();
    baseline["colorspace"]["selected_candidate_typo"] =
        serde_json::json!("neutral_balance_fallback");
    std::fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&baseline).unwrap(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture")
        .arg("logan")
        .arg("--report")
        .arg(&report_path)
        .arg("--compare-summary")
        .arg(&baseline_path)
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate with typoed summary baseline");

    assert!(
        !output.status.success(),
        "unknown baseline field unexpectedly passed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown field `selected_candidate_typo`"),
        "stderr missing unknown baseline field error: {stderr}"
    );
}

#[test]
fn test_validate_cli_rejects_incomplete_direct_summary_baseline_contract() {
    let tmp = tempfile::TempDir::new().unwrap();
    let report_path = tmp.path().join("report.json");
    let baseline_path = tmp.path().join("baseline.json");
    let summary_json = tmp.path().join("summary.json");
    let summary_md = tmp.path().join("summary.md");

    let output_path = tmp.path().join("output.tiff");
    scanstitch::tiff_io::save_tiff_f64_linear_prophoto(&small_rgb_image(), &output_path).unwrap();
    let mut report = fixture_report();
    set_report_output_path(&mut report, &output_path);
    let _debug_artifacts = write_report_color_debug_artifacts(&mut report);
    report.save(&report_path).unwrap();
    std::fs::write(
        &baseline_path,
        serde_json::json!({ "fixture": "logan" }).to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture")
        .arg("logan")
        .arg("--report")
        .arg(&report_path)
        .arg("--compare-summary")
        .arg(&baseline_path)
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate with incomplete summary baseline");

    assert!(
        !output.status.success(),
        "incomplete direct summary baseline unexpectedly passed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("summary_baseline_incomplete:stitch.decision"),
        "stderr missing incomplete baseline issue: {stderr}"
    );
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(
        summary["summary_baseline_comparison"]["status"],
        "review_required"
    );
    assert!(summary["summary_baseline_comparison"]["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "summary_baseline_incomplete:render.render_input_source"));
}

#[test]
fn test_validate_cli_rejects_fixture_registry_contract_violations() {
    let tmp = tempfile::TempDir::new().unwrap();
    let registry_path = tmp.path().join("fixtures.json");
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "required_scene_tags": ["skin-tone", "skin-tone"],
                "required_scanner_profiles": ["scanner-a", "scanner-a"],
                "required_roll_profiles": ["roll-a", "roll-a"],
                "required_reference_evidence": ["gray-card", "gray-card"],
                "required_scene_exposure_pairs": ["missing-separator"],
                "required_debug_artifact_kinds": [
                    "",
                    "candidate_comparison",
                    "candidate_comparison",
                    "not-a-debug-artifact"
                ]
            },
            "fixtures": {
                "both-calibration": {
                    "component1": "left.tif",
                    "component2": "right.tif",
                    "calibration_profile": "profile.json",
                    "calibration_library": "calibration",
                    "scene_tags": ["portrait", "portrait"]
                },
                "scanner-without-library": {
                    "component1": "left.tif",
                    "component2": "right.tif",
                    "scanner_profile": "scanner-a"
                },
                "roll-without-scanner": {
                    "component1": "left.tif",
                    "component2": "right.tif",
                    "calibration_library": "calibration",
                    "roll_profile": "roll-a",
                    "calibration_case": "scanner-roll-library"
                },
                "external-no-profile": {
                    "component1": "left.tif",
                    "component2": "right.tif",
                    "calibration_case": "external-profile"
                },
                "uncalibrated-with-profile": {
                    "component1": "left.tif",
                    "component2": "right.tif",
                    "calibration_profile": "profile.json",
                    "calibration_case": "uncalibrated-image-derived"
                },
                "bad-expectations": {
                    "component1": "left.tif",
                    "component2": "right.tif",
                    "component1_sha256": "not-a-sha256",
                    "summary_baseline_sha256": "not-a-sha256",
                    "calibration_library_sha256": "not-a-sha256",
                    "reference_evidence": ["gray-card", "gray-card"],
                    "expectations": {
                        "stitch_decision": "",
                        "base_estimate_source": "",
                        "output_color_space": "",
                        "render_input_source": "",
                        "render_input_reason_contains": "",
                        "selected_mapping_reason_contains": "",
                        "candidate_acceptance_signatures_required": [
                            "",
                            "signature-a",
                            "signature-a"
                        ],
                        "calibration_rejection_details_required": [
                            "",
                            "calibration-detail-a",
                            "calibration-detail-a"
                        ],
                        "calibration_confidence_min": 1.1,
                        "calibration_matrix_condition_number_max": -0.1,
                        "selection_rejections_required": [
                            "",
                            "selection-rejection-a",
                            "selection-rejection-a"
                        ],
                        "selected_quality_score_max": -0.1,
                        "technical_safety_score_max": -0.1,
                        "color_fidelity_score_max": -0.1,
                        "highlight_chroma_compressed_ratio_min": 1.1,
                        "highlight_chroma_compressed_ratio_max": 1.1,
                        "highlight_neutral_chroma_compressed_ratio_max": 1.1,
                        "shadow_chroma_compressed_ratio_max": 1.1,
                        "memory_color_penalty_max": -0.1,
                        "spatial_consistency_penalty_max": -0.1,
                        "selected_runner_up_quality_delta_min": -0.1,
                        "density_monotonicity_score_min": -0.1,
                        "hue_linearity_score_min": -0.1,
                        "saturation_preservation_median_ratio_min": -0.1,
                        "spatial_neutral_delta_p95_max": -0.1,
                        "post_scale_preserved_ratio_min": 1.1,
                        "reference_patch_rms_delta_e_max": -0.1,
                        "reference_patch_rms_delta_e2000_max": -0.1,
                        "debug_artifact_kinds_required": [
                            "",
                            "candidate_comparison",
                            "candidate_comparison",
                            "not-a-debug-artifact"
                        ]
                    }
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--list-fixtures")
        .output()
        .expect("run scanstitch-validate with invalid fixture registry");

    assert!(
        !output.status.success(),
        "invalid fixture registry unexpectedly passed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    for expected in [
        "fixture registry",
        "coverage_requirements:required_scene_tag_duplicate:skin-tone",
        "coverage_requirements:required_scanner_profile_duplicate:scanner-a",
        "coverage_requirements:required_roll_profile_duplicate:roll-a",
        "coverage_requirements:required_reference_evidence_duplicate:gray-card",
        "coverage_requirements:required_scene_exposure_pair_invalid:missing-separator",
        "coverage_requirements:required_debug_artifact_kind_empty",
        "coverage_requirements:required_debug_artifact_kind_duplicate:candidate_comparison",
        "coverage_requirements:required_debug_artifact_kind_unknown:not-a-debug-artifact",
        "both-calibration:calibration_profile_and_calibration_library_both_declared",
        "both-calibration:scene_tag_duplicate:portrait",
        "scanner-without-library:profile_id_without_calibration_library",
        "roll-without-scanner:roll_profile_without_scanner_profile",
        "roll-without-scanner:scanner_roll_library_case_without_scanner_profile",
        "external-no-profile:external_profile_case_without_calibration_profile",
        "uncalibrated-with-profile:uncalibrated_case_declares_calibration_profile",
        "bad-expectations:component1_sha256_invalid",
        "bad-expectations:component_sha256_pair_incomplete",
        "bad-expectations:summary_baseline_sha256_invalid",
        "bad-expectations:summary_baseline_sha256_without_summary_baseline",
        "bad-expectations:calibration_library_sha256_invalid",
        "bad-expectations:calibration_library_sha256_without_calibration_library",
        "bad-expectations:reference_evidence_duplicate:gray-card",
        "bad-expectations:expectations.stitch_decision_empty",
        "bad-expectations:expectations.base_estimate_source_empty",
        "bad-expectations:expectations.output_color_space_empty",
        "bad-expectations:expectations.render_input_source_empty",
        "bad-expectations:expectations.render_input_reason_contains_empty",
        "bad-expectations:expectations.selected_mapping_reason_contains_empty",
        "bad-expectations:expectations.candidate_acceptance_signatures_required_empty",
        "bad-expectations:expectations.candidate_acceptance_signatures_required_duplicate:signature-a",
        "bad-expectations:expectations.calibration_rejection_details_required_empty",
        "bad-expectations:expectations.calibration_rejection_details_required_duplicate:calibration-detail-a",
        "bad-expectations:expectations.calibration_confidence_min_out_of_range",
        "bad-expectations:expectations.calibration_matrix_condition_number_max_out_of_range",
        "bad-expectations:expectations.selection_rejections_required_empty",
        "bad-expectations:expectations.selection_rejections_required_duplicate:selection-rejection-a",
        "bad-expectations:expectations.selected_quality_score_max_out_of_range",
        "bad-expectations:expectations.technical_safety_score_max_out_of_range",
        "bad-expectations:expectations.color_fidelity_score_max_out_of_range",
        "bad-expectations:expectations.highlight_chroma_compressed_ratio_min_out_of_range",
        "bad-expectations:expectations.highlight_chroma_compressed_ratio_max_out_of_range",
        "bad-expectations:expectations.highlight_neutral_chroma_compressed_ratio_max_out_of_range",
        "bad-expectations:expectations.shadow_chroma_compressed_ratio_max_out_of_range",
        "bad-expectations:expectations.memory_color_penalty_max_out_of_range",
        "bad-expectations:expectations.spatial_consistency_penalty_max_out_of_range",
        "bad-expectations:expectations.selected_runner_up_quality_delta_min_out_of_range",
        "bad-expectations:expectations.density_monotonicity_score_min_out_of_range",
        "bad-expectations:expectations.hue_linearity_score_min_out_of_range",
        "bad-expectations:expectations.saturation_preservation_median_ratio_min_out_of_range",
        "bad-expectations:expectations.spatial_neutral_delta_p95_max_out_of_range",
        "bad-expectations:expectations.post_scale_preserved_ratio_min_out_of_range",
        "bad-expectations:expectations.reference_patch_rms_delta_e_max_out_of_range",
        "bad-expectations:expectations.reference_patch_rms_delta_e2000_max_out_of_range",
        "bad-expectations:expectations.debug_artifact_kinds_required_empty",
        "bad-expectations:expectations.debug_artifact_kinds_required_duplicate:candidate_comparison",
        "bad-expectations:expectations.debug_artifact_kinds_required_unknown:not-a-debug-artifact",
    ] {
        assert!(
            stderr.contains(expected),
            "stderr missing {expected:?}: {stderr}"
        );
    }

    let unknown_registry_path = tmp.path().join("fixtures-unknown-field.json");
    std::fs::write(
        &unknown_registry_path,
        serde_json::json!({
            "fixtures": {
                "unknown-field": {
                    "component1": "left.tif",
                    "component2": "right.tif",
                    "unexpected": true
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&unknown_registry_path)
        .arg("--list-fixtures")
        .output()
        .expect("run scanstitch-validate with unknown fixture registry field");

    assert!(
        !output.status.success(),
        "unknown fixture registry field unexpectedly passed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown field `unexpected`"),
        "stderr missing unknown-field serde error: {stderr}"
    );
}

#[test]
fn test_validate_cli_writes_summary_baseline_from_report() {
    let tmp = tempfile::TempDir::new().unwrap();
    let report_path = tmp.path().join("report.json");
    let baseline_path = tmp.path().join("baseline.json");
    let summary_json = tmp.path().join("summary.json");
    let summary_md = tmp.path().join("summary.md");

    let output_path = tmp.path().join("output.tiff");
    scanstitch::tiff_io::save_tiff_f64_linear_prophoto(&small_rgb_image(), &output_path).unwrap();
    let mut report = fixture_report();
    set_report_output_path(&mut report, &output_path);
    let _debug_artifacts = write_report_color_debug_artifacts(&mut report);
    report.save(&report_path).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture")
        .arg("logan")
        .arg("--report")
        .arg(&report_path)
        .arg("--write-summary-baseline")
        .arg(&baseline_path)
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate baseline export");

    assert!(
        output.status.success(),
        "baseline export CLI failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let baseline: scanstitch::validation::TrackedValidationBaseline =
        serde_json::from_str(&std::fs::read_to_string(&baseline_path).unwrap()).unwrap();
    assert_eq!(baseline.fixture.as_deref(), Some("logan"));
    assert_eq!(baseline.render.output_width, Some(1800));
    assert_eq!(
        baseline.render.output_color_space.as_deref(),
        Some("linear_prophoto_rgb_d50")
    );
    assert_eq!(
        baseline.render.output_file_icc_profile_matches_report,
        Some(true)
    );
    assert_eq!(baseline.render.raw_base_proxy_confidence, Some(0.12));
    assert_eq!(baseline.render.raw_base_support_fraction, Some(0.014));
    assert_eq!(
        baseline.render.render_input_source.as_deref(),
        Some("direct_density_transmittance")
    );
    assert_eq!(
        baseline.colorspace.calibration_status.as_deref(),
        Some("applied")
    );
    assert_eq!(baseline.colorspace.calibration_confidence, Some(0.95));
    assert_eq!(
        baseline.colorspace.calibration_matrix_condition_number,
        Some(2.1)
    );
    assert_eq!(
        baseline
            .colorspace
            .calibration_scanner_profile_status
            .as_deref(),
        Some("applied")
    );
    assert_eq!(
        baseline
            .colorspace
            .calibration_scanner_profile_id
            .as_deref(),
        Some("scanner-a")
    );
    assert_eq!(
        baseline
            .colorspace
            .calibration_roll_profile_status
            .as_deref(),
        Some("applied")
    );
    assert_eq!(
        baseline.colorspace.calibration_roll_profile_id.as_deref(),
        Some("roll-a")
    );
    assert_eq!(
        baseline
            .colorspace
            .calibration_requested_film_stock
            .as_deref(),
        Some("Synthetic 200")
    );
    assert_eq!(
        baseline.colorspace.calibration_film_stock_status.as_deref(),
        Some("matched")
    );
    assert_eq!(
        baseline
            .colorspace
            .calibration_film_stock_matched_roll_profiles,
        vec!["roll-a".to_string()]
    );
    assert_eq!(
        baseline.colorspace.selected_candidate.as_deref(),
        Some("neutral_balance_fallback")
    );
    assert_eq!(baseline.colorspace.selected_candidate_rank, Some(1));
    assert_eq!(
        baseline.colorspace.calibration_acceptance_status.as_deref(),
        Some("accepted")
    );
    assert_eq!(baseline.colorspace.selected_quality_score, Some(0.42));
    assert_eq!(baseline.colorspace.technical_safety_score, Some(0.12));
    assert_eq!(baseline.colorspace.color_fidelity_score, Some(0.30));
    assert_eq!(
        baseline.colorspace.selected_runner_up_quality_delta,
        Some(0.46)
    );
    assert_eq!(baseline.colorspace.hue_linearity_score, Some(0.92));
    assert_eq!(baseline.colorspace.memory_color_penalty, Some(0.06));
    assert_eq!(baseline.colorspace.spatial_consistency_penalty, Some(0.07));
    assert_eq!(
        baseline.colorspace.reference_patch_selected_rms_delta_e,
        Some(3.4)
    );
    assert_eq!(
        baseline
            .colorspace
            .reference_patch_selected_regresses_image_derived,
        Some(true)
    );
    assert_eq!(
        baseline.colorspace.reference_patch_hue_family_regressions,
        vec!["red".to_string()]
    );
    assert_eq!(baseline.colorspace.neutral_estimate_score, Some(0.92));
    assert_eq!(baseline.colorspace.neutral_estimate_accepted, Some(true));
    assert_eq!(
        baseline.colorspace.dominant_anchor_channel_unstable,
        vec![false, true, false]
    );
    assert_eq!(
        baseline.colorspace.channel_anchor_low_support,
        vec![false, true, false]
    );
    assert_eq!(baseline.colorspace.weak_anchor_fallback_used, Some(true));
    assert_eq!(baseline.colorspace.neutral_trim_applied, Some(true));
    assert_eq!(baseline.colorspace.candidate_acceptance_signatures.len(), 2);
    assert_eq!(baseline.grain.luma_residual_p95, Some(0.006));

    let summary: scanstitch::validation::ValidationSummary =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    let comparison =
        scanstitch::validation::compare_summary_baseline("baseline.json", &baseline, &summary);
    assert_eq!(comparison.status, "comparable");
}

#[test]
fn test_validate_cli_reports_fixture_coverage_strict() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path1 = tmp.path().join("left.tiff");
    let path2 = tmp.path().join("right.tiff");
    let baseline_path = tmp.path().join("baseline.json");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("coverage.json");
    let summary_md = tmp.path().join("coverage.md");

    let component = synthetic::constant_image(4, 4, [12000, 7000, 3000]);
    scanstitch::tiff_io::save_tiff_u16(&component, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&component, &path2).unwrap();
    let component1_sha256 = file_sha256_hex(&path1);
    let component2_sha256 = file_sha256_hex(&path2);
    std::fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&fixture_summary_baseline()).unwrap(),
    )
    .unwrap();
    let summary_baseline_sha256 = file_sha256_hex(&baseline_path);
    let library_dir = write_validation_fixture_library_with_reference_patches(tmp.path());
    let calibration_library_sha256 = calibration_library_sha256_hex(&library_dir);
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "min_component_pairs": 1,
                "min_component_sha256_pairs": 1,
                "min_readable_tiff_pairs": 1,
                "min_tiff_layout_consistent_pairs": 1,
                "min_tiff_dimension_matched_pairs": 1,
                "min_tiff_bits_per_sample": 14,
                "min_summary_baselines": 1,
                "min_summary_baseline_sha256_fixtures": 1,
                "min_calibrated_fixtures": 1,
                "min_calibration_sha256_fixtures": 1,
                "min_unique_scanner_profiles": 1,
                "min_unique_roll_profiles": 1,
                "min_unique_film_stocks": 1,
                "min_scene_tags": 2,
                "min_exposure_tags": 1,
                "min_calibration_cases": 1,
                "min_film_stock_calibration_pairs": 1,
                "min_scene_exposure_pairs": 2,
                "min_reference_fixtures": 1,
                "min_reference_evidence_types": 2,
                "min_reference_patch_fixtures": 1,
                "min_reference_patch_count": 4,
                "min_debug_artifact_expectation_fixtures": 1,
                "required_film_stocks": ["Synthetic negative"],
                "required_scene_tags": ["neutral-target"],
                "required_exposure_tags": ["normal-exposure"],
                "required_calibration_cases": ["scanner-roll-library"],
                "required_scanner_profiles": ["scanner-a"],
                "required_roll_profiles": ["roll-a"],
                "required_reference_evidence": ["gray-card", "colorchecker"],
                "required_film_stock_calibration_pairs": [
                    "Synthetic negative|scanner-roll-library"
                ],
                "required_scene_exposure_pairs": [
                    "neutral-target|normal-exposure",
                    "synthetic-still-life|normal-exposure"
                ],
                "required_debug_artifact_kinds": [
                    "candidate_comparison",
                    "gamut_clipping_map",
                    "scene_referred_prophoto_float"
                ]
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "component1_sha256": component1_sha256.clone(),
                    "component2_sha256": component2_sha256.clone(),
                    "output_dir": tmp.path().join("out"),
                    "summary_baseline": baseline_path,
                    "summary_baseline_sha256": summary_baseline_sha256.clone(),
                    "calibration_library": library_dir,
                    "calibration_library_sha256": calibration_library_sha256.clone(),
                    "scanner_profile": "scanner-a",
                    "roll_profile": "roll-a",
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target", "synthetic-still-life"],
                    "exposure_tags": ["normal-exposure"],
                    "reference_evidence": ["gray-card", "colorchecker"],
                    "calibration_case": "scanner-roll-library",
                    "expectations": {
                        "debug_artifacts_required": true,
                        "debug_artifact_kinds_required": [
                            "candidate_comparison",
                            "gamut_clipping_map",
                            "scene_referred_prophoto_float"
                        ]
                    }
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-coverage")
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate fixture coverage");

    assert!(
        output.status.success(),
        "fixture coverage CLI failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["status"], "passed");
    assert_eq!(summary["fixture_count"], 1);
    assert_eq!(summary["component_pair_available_count"], 1);
    assert_eq!(summary["component_sha256_declared_pair_count"], 1);
    assert_eq!(summary["component_sha256_computed_pair_count"], 1);
    assert_eq!(summary["component_sha256_pair_count"], 1);
    assert_eq!(summary["readable_tiff_pair_count"], 1);
    assert_eq!(summary["tiff_layout_consistent_pair_count"], 1);
    assert_eq!(summary["tiff_dimension_matched_pair_count"], 1);
    assert_eq!(summary["validation_ready_fixture_count"], 1);
    assert_eq!(summary["summary_baseline_declared_count"], 1);
    assert_eq!(summary["summary_baseline_file_count"], 1);
    assert_eq!(summary["summary_baseline_parseable_count"], 1);
    assert_eq!(summary["summary_baseline_contract_complete_count"], 1);
    assert_eq!(summary["summary_baseline_count"], 1);
    assert_eq!(summary["summary_baseline_sha256_declared_count"], 1);
    assert_eq!(summary["summary_baseline_sha256_computed_count"], 1);
    assert_eq!(summary["summary_baseline_sha256_count"], 1);
    assert_eq!(summary["calibration_evidence_declared_count"], 1);
    assert_eq!(summary["calibration_evidence_count"], 1);
    assert_eq!(summary["calibration_evidence_unusable_count"], 0);
    assert_eq!(summary["calibration_sha256_declared_count"], 1);
    assert_eq!(summary["calibration_sha256_computed_count"], 1);
    assert_eq!(summary["calibration_sha256_count"], 1);
    assert_eq!(summary["uncalibrated_fixture_count"], 0);
    assert_eq!(summary["scanner_profile_count"], 1);
    assert_eq!(summary["roll_profile_count"], 1);
    assert_eq!(summary["film_stock_count"], 1);
    assert_eq!(summary["scene_tag_count"], 2);
    assert_eq!(summary["exposure_tag_count"], 1);
    assert_eq!(summary["calibration_case_count"], 1);
    assert_eq!(summary["reference_fixture_count"], 1);
    assert_eq!(summary["reference_evidence_type_count"], 2);
    assert_eq!(summary["reference_patch_fixture_count"], 1);
    assert_eq!(summary["reference_patch_count"], 4);
    assert_eq!(summary["debug_artifact_expectation_fixture_count"], 1);
    assert_eq!(summary["film_stock_calibration_pair_count"], 1);
    assert_eq!(summary["scene_exposure_pair_count"], 2);
    assert_eq!(
        summary["coverage_requirements"]["min_scene_tags"],
        serde_json::json!(2)
    );
    assert_eq!(
        summary["coverage_requirements"]["min_readable_tiff_pairs"],
        serde_json::json!(1)
    );
    assert_eq!(
        summary["coverage_requirements"]["min_component_sha256_pairs"],
        serde_json::json!(1)
    );
    assert_eq!(
        summary["coverage_requirements"]["min_tiff_layout_consistent_pairs"],
        serde_json::json!(1)
    );
    assert_eq!(
        summary["coverage_requirements"]["min_tiff_dimension_matched_pairs"],
        serde_json::json!(1)
    );
    assert_eq!(
        summary["coverage_requirements"]["min_tiff_bits_per_sample"],
        serde_json::json!(14)
    );
    assert_eq!(
        summary["coverage_requirements"]["min_summary_baseline_sha256_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        summary["coverage_requirements"]["min_calibration_sha256_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        summary["coverage_requirements"]["min_debug_artifact_expectation_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        summary["unique_scanner_profiles"],
        serde_json::json!(["scanner-a"])
    );
    assert_eq!(
        summary["missing_required_scanner_profiles"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["unique_roll_profiles"],
        serde_json::json!(["roll-a"])
    );
    assert_eq!(
        summary["missing_required_roll_profiles"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["unique_film_stocks"],
        serde_json::json!(["Synthetic negative"])
    );
    assert_eq!(
        summary["missing_required_film_stocks"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["scene_tags"],
        serde_json::json!(["neutral-target", "synthetic-still-life"])
    );
    assert_eq!(
        summary["missing_required_scene_tags"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["exposure_tags"],
        serde_json::json!(["normal-exposure"])
    );
    assert_eq!(
        summary["missing_required_exposure_tags"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["calibration_cases"],
        serde_json::json!(["scanner-roll-library"])
    );
    assert_eq!(
        summary["missing_required_calibration_cases"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["reference_evidence"],
        serde_json::json!(["colorchecker", "gray-card"])
    );
    assert_eq!(
        summary["missing_required_reference_evidence"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["film_stock_calibration_pairs"],
        serde_json::json!(["Synthetic negative|scanner-roll-library"])
    );
    assert_eq!(
        summary["missing_required_film_stock_calibration_pairs"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["scene_exposure_pairs"],
        serde_json::json!([
            "neutral-target|normal-exposure",
            "synthetic-still-life|normal-exposure"
        ])
    );
    assert_eq!(
        summary["missing_required_scene_exposure_pairs"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["debug_artifact_kinds_required"],
        serde_json::json!([
            "candidate_comparison",
            "gamut_clipping_map",
            "scene_referred_prophoto_float"
        ])
    );
    assert_eq!(
        summary["missing_required_debug_artifact_kinds"],
        serde_json::json!([])
    );
    assert!(summary["issues"].as_array().unwrap().is_empty());
    assert_eq!(
        summary["fixtures"][0]["summary_baseline_parse_status"],
        "valid"
    );
    assert_eq!(summary["fixtures"][0]["summary_baseline_valid"], true);
    assert_eq!(
        summary["fixtures"][0]["summary_baseline_sha256"]["status"],
        "matched"
    );
    assert_eq!(
        summary["fixtures"][0]["summary_baseline_sha256"]["expected_sha256"],
        serde_json::json!(summary_baseline_sha256)
    );
    assert_eq!(
        summary["fixtures"][0]["summary_baseline_sha256"]["actual_sha256"],
        serde_json::json!(summary_baseline_sha256)
    );
    assert_eq!(
        summary["fixtures"][0]["calibration_library_selection_status"],
        "applied"
    );
    assert_eq!(
        summary["fixtures"][0]["calibration_library_sha256"]["status"],
        "matched"
    );
    assert_eq!(
        summary["fixtures"][0]["calibration_library_sha256"]["expected_sha256"],
        serde_json::json!(calibration_library_sha256)
    );
    assert_eq!(
        summary["fixtures"][0]["calibration_library_sha256"]["actual_sha256"],
        serde_json::json!(calibration_library_sha256)
    );
    assert!(
        summary["fixtures"][0]["calibration_library_sha256"]["file_count"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(summary["fixtures"][0]["calibration_evidence_usable"], true);
    assert_eq!(
        summary["fixtures"][0]["calibration_reference_patch_count"],
        4
    );
    assert_eq!(summary["fixtures"][0]["validation_ready"], true);
    assert_eq!(
        summary["fixtures"][0]["component1_tiff"]["status"],
        "readable"
    );
    assert_eq!(
        summary["fixtures"][0]["component1_sha256"]["status"],
        "matched"
    );
    assert_eq!(
        summary["fixtures"][0]["component1_sha256"]["expected_sha256"],
        serde_json::json!(component1_sha256)
    );
    assert_eq!(
        summary["fixtures"][0]["component1_sha256"]["actual_sha256"],
        serde_json::json!(component1_sha256)
    );
    assert_eq!(summary["fixtures"][0]["component1_sha256"]["matched"], true);
    assert!(
        summary["fixtures"][0]["component1_sha256"]["file_size_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(
        summary["fixtures"][0]["component1_tiff"]["source_bits_per_sample"],
        16
    );
    assert_eq!(
        summary["fixtures"][0]["component2_tiff"]["status"],
        "readable"
    );
    assert_eq!(
        summary["fixtures"][0]["component2_sha256"]["status"],
        "matched"
    );
    assert_eq!(
        summary["fixtures"][0]["component2_sha256"]["actual_sha256"],
        serde_json::json!(component2_sha256)
    );
    assert_eq!(
        summary["fixtures"][0]["tiff_pair"]["layout_consistent"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["tiff_pair"]["dimension_matched"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["calibration_case"],
        "scanner-roll-library"
    );
    assert_eq!(summary["fixtures"][0]["debug_artifacts_required"], true);
    assert_eq!(
        summary["fixtures"][0]["debug_artifact_kinds_required"],
        serde_json::json!([
            "candidate_comparison",
            "gamut_clipping_map",
            "scene_referred_prophoto_float"
        ])
    );
    assert_eq!(
        summary["fixtures"][0]["action_items"],
        serde_json::json!([])
    );
    assert_eq!(summary["action_items"], serde_json::json!([]));

    let markdown = std::fs::read_to_string(&summary_md).unwrap();
    assert!(markdown.contains("# Fixture Coverage"));
    assert!(markdown.contains("component SHA-256 pairs matched"));
    assert!(markdown.contains("matched/matched"));
    assert!(markdown.contains("summary baseline SHA-256 fixtures matched"));
    assert!(markdown.contains("calibration SHA-256 fixtures matched"));
    assert!(markdown.contains("readable TIFF pairs"));
    assert!(markdown.contains("TIFF layout-consistent pairs"));
    assert!(markdown.contains("TIFF dimension-matched pairs"));
    assert!(markdown.contains("layout matched; dimensions matched"));
    assert!(markdown.contains("RGB16"));
    assert!(markdown.contains("Synthetic negative"));
    assert!(markdown.contains("synthetic-still-life"));
    assert!(markdown.contains("scanner-roll-library"));
    assert!(markdown.contains("reference evidence"));
    assert!(markdown.contains("colorchecker"));
    assert!(markdown.contains("gray-card"));
    assert!(markdown.contains("reference-patch fixtures"));
    assert!(markdown.contains("reference patches"));
    assert!(markdown.contains("debug-artifact expectation fixtures"));
    assert!(markdown.contains("Actions"));
    assert!(markdown.contains("required debug artifact kinds"));
    assert!(markdown.contains("candidate_comparison"));
    assert!(markdown.contains("gamut_clipping_map"));
    assert!(markdown.contains("scanner-a"));
    assert!(markdown.contains("roll-a"));
    assert!(markdown.contains("Synthetic negative|scanner-roll-library"));
    assert!(markdown.contains("synthetic-still-life|normal-exposure"));
}

#[test]
fn test_validate_cli_fixture_coverage_can_compute_undeclared_component_hashes() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path1 = tmp.path().join("left.tiff");
    let path2 = tmp.path().join("right.tiff");
    let baseline_path = tmp.path().join("baseline.json");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("coverage.json");
    let summary_md = tmp.path().join("coverage.md");

    let component = synthetic::constant_image(4, 4, [12000, 7000, 3000]);
    scanstitch::tiff_io::save_tiff_u16(&component, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&component, &path2).unwrap();
    let component1_sha256 = file_sha256_hex(&path1);
    let component2_sha256 = file_sha256_hex(&path2);
    std::fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&fixture_summary_baseline()).unwrap(),
    )
    .unwrap();
    let summary_baseline_sha256 = file_sha256_hex(&baseline_path);
    let library_dir = write_validation_fixture_library(tmp.path());
    let calibration_library_sha256 = calibration_library_sha256_hex(&library_dir);
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "min_component_pairs": 1,
                "min_readable_tiff_pairs": 1,
                "min_tiff_layout_consistent_pairs": 1,
                "min_tiff_dimension_matched_pairs": 1,
                "min_summary_baselines": 1,
                "min_calibrated_fixtures": 1
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "summary_baseline": baseline_path,
                    "calibration_library": library_dir,
                    "scanner_profile": "scanner-a",
                    "roll_profile": "roll-a",
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "scanner-roll-library"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-coverage")
        .arg("--compute-fixture-hashes")
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate fixture coverage with computed hashes");

    assert!(
        output.status.success(),
        "fixture coverage with computed hashes failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["component_sha256_declared_pair_count"], 0);
    assert_eq!(summary["component_sha256_computed_pair_count"], 1);
    assert_eq!(summary["component_sha256_pair_count"], 0);
    assert_eq!(summary["summary_baseline_sha256_declared_count"], 0);
    assert_eq!(summary["summary_baseline_sha256_computed_count"], 1);
    assert_eq!(summary["summary_baseline_sha256_count"], 0);
    assert_eq!(summary["calibration_sha256_declared_count"], 0);
    assert_eq!(summary["calibration_sha256_computed_count"], 1);
    assert_eq!(summary["calibration_sha256_count"], 0);
    assert_eq!(
        summary["fixtures"][0]["component1_sha256"]["status"],
        "computed"
    );
    assert_eq!(
        summary["fixtures"][0]["component1_sha256"]["expected_sha256"],
        serde_json::Value::Null
    );
    assert_eq!(
        summary["fixtures"][0]["component1_sha256"]["actual_sha256"],
        serde_json::json!(component1_sha256)
    );
    assert_eq!(
        summary["fixtures"][0]["component1_sha256"]["matched"],
        false
    );
    assert_eq!(
        summary["fixtures"][0]["component2_sha256"]["status"],
        "computed"
    );
    assert_eq!(
        summary["fixtures"][0]["component2_sha256"]["actual_sha256"],
        serde_json::json!(component2_sha256)
    );
    assert_eq!(
        summary["fixtures"][0]["summary_baseline_sha256"]["status"],
        "computed"
    );
    assert_eq!(
        summary["fixtures"][0]["summary_baseline_sha256"]["expected_sha256"],
        serde_json::Value::Null
    );
    assert_eq!(
        summary["fixtures"][0]["summary_baseline_sha256"]["actual_sha256"],
        serde_json::json!(summary_baseline_sha256)
    );
    assert_eq!(
        summary["fixtures"][0]["calibration_library_sha256"]["status"],
        "computed"
    );
    assert_eq!(
        summary["fixtures"][0]["calibration_library_sha256"]["expected_sha256"],
        serde_json::Value::Null
    );
    assert_eq!(
        summary["fixtures"][0]["calibration_library_sha256"]["actual_sha256"],
        serde_json::json!(calibration_library_sha256)
    );
    assert_eq!(summary["fixtures"][0]["validation_ready"], true);
    assert!(summary["issues"].as_array().unwrap().is_empty());

    let markdown = std::fs::read_to_string(&summary_md).unwrap();
    assert!(markdown.contains("component SHA-256 pairs computed"));
    assert!(markdown.contains("computed/computed"));
    assert!(markdown.contains("summary baseline SHA-256 fixtures computed"));
    assert!(markdown.contains("calibration SHA-256 fixtures computed"));
}

#[test]
fn test_validate_cli_fixture_coverage_writes_hash_registry_snapshot() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path1 = tmp.path().join("left.tiff");
    let path2 = tmp.path().join("right.tiff");
    let baseline_path = tmp.path().join("baseline.json");
    let registry_path = tmp.path().join("fixtures.json");
    let snapshot_path = tmp.path().join("fixtures-with-hashes.json");
    let summary_json = tmp.path().join("coverage.json");
    let summary_md = tmp.path().join("coverage.md");

    let component = synthetic::constant_image(4, 4, [12000, 7000, 3000]);
    scanstitch::tiff_io::save_tiff_u16(&component, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&component, &path2).unwrap();
    let component1_sha256 = file_sha256_hex(&path1);
    let component2_sha256 = file_sha256_hex(&path2);
    std::fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&fixture_summary_baseline()).unwrap(),
    )
    .unwrap();
    let summary_baseline_sha256 = file_sha256_hex(&baseline_path);
    let library_dir = write_validation_fixture_library(tmp.path());
    let calibration_library_sha256 = calibration_library_sha256_hex(&library_dir);
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "min_component_pairs": 1,
                "min_readable_tiff_pairs": 1,
                "min_tiff_layout_consistent_pairs": 1,
                "min_tiff_dimension_matched_pairs": 1,
                "min_summary_baselines": 1,
                "min_calibrated_fixtures": 1
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "summary_baseline": baseline_path,
                    "calibration_library": library_dir,
                    "scanner_profile": "scanner-a",
                    "roll_profile": "roll-a",
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "scanner-roll-library"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-coverage")
        .arg("--write-fixture-hash-registry")
        .arg(&snapshot_path)
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate fixture coverage with hash registry writer");

    assert!(
        output.status.success(),
        "fixture coverage hash registry writer failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["component_sha256_computed_pair_count"], 1);
    assert_eq!(summary["summary_baseline_sha256_computed_count"], 1);
    assert_eq!(summary["calibration_sha256_computed_count"], 1);
    assert_eq!(summary["component_sha256_pair_count"], 0);
    assert_eq!(summary["summary_baseline_sha256_count"], 0);
    assert_eq!(summary["calibration_sha256_count"], 0);

    let snapshot: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&snapshot_path).unwrap()).unwrap();
    let fixture = &snapshot["fixtures"]["logan"];
    assert_eq!(fixture["component1_sha256"], component1_sha256);
    assert_eq!(fixture["component2_sha256"], component2_sha256);
    assert_eq!(fixture["summary_baseline_sha256"], summary_baseline_sha256);
    assert_eq!(
        fixture["calibration_library_sha256"],
        calibration_library_sha256
    );
    assert!(fixture.get("calibration_profile_sha256").is_none());
    assert!(fixture.get("expectations").is_none());
    assert_eq!(snapshot["coverage_requirements"]["min_fixtures"], 1);
    assert!(snapshot["coverage_requirements"]
        .get("min_component_sha256_pairs")
        .is_none());
    assert!(!std::fs::read_to_string(&registry_path)
        .unwrap()
        .contains("component1_sha256"));
}

#[test]
fn test_validate_cli_fixture_coverage_hash_gates_require_validation_ready_fixture() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path1 = tmp.path().join("left.tiff");
    let path2 = tmp.path().join("right.tiff");
    let baseline_path = tmp.path().join("baseline.json");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("coverage.json");

    let component = synthetic::constant_image(4, 4, [12000, 7000, 3000]);
    scanstitch::tiff_io::save_tiff_u16(&component, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&component, &path2).unwrap();
    let component1_sha256 = file_sha256_hex(&path1);
    let component2_sha256 = file_sha256_hex(&path2);
    std::fs::write(&baseline_path, b"{not valid json").unwrap();
    let summary_baseline_sha256 = file_sha256_hex(&baseline_path);
    let library_dir = write_validation_fixture_library(tmp.path());
    let calibration_library_sha256 = calibration_library_sha256_hex(&library_dir);
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "min_component_pairs": 1,
                "min_component_sha256_pairs": 1,
                "min_readable_tiff_pairs": 1,
                "min_tiff_layout_consistent_pairs": 1,
                "min_tiff_dimension_matched_pairs": 1,
                "min_summary_baselines": 1,
                "min_summary_baseline_sha256_fixtures": 1,
                "min_calibrated_fixtures": 1,
                "min_calibration_sha256_fixtures": 1
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "component1_sha256": component1_sha256,
                    "component2_sha256": component2_sha256,
                    "summary_baseline": baseline_path,
                    "summary_baseline_sha256": summary_baseline_sha256,
                    "calibration_library": library_dir,
                    "calibration_library_sha256": calibration_library_sha256,
                    "scanner_profile": "scanner-a",
                    "roll_profile": "roll-a",
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "scanner-roll-library"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-coverage")
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .output()
        .expect("run scanstitch-validate fixture coverage with matched hashes on unusable fixture");

    assert!(
        !output.status.success(),
        "fixture coverage should fail strict hash gates for an unusable fixture"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("logan:summary_baseline_invalid"));
    assert!(stderr.contains("fixture_coverage_min_component_sha256_pairs_not_met"));
    assert!(stderr.contains("fixture_coverage_min_summary_baseline_sha256_fixtures_not_met"));
    assert!(stderr.contains("fixture_coverage_min_calibration_sha256_fixtures_not_met"));
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["validation_ready_fixture_count"], 0);
    assert_eq!(summary["component_sha256_declared_pair_count"], 1);
    assert_eq!(summary["component_sha256_computed_pair_count"], 1);
    assert_eq!(summary["component_sha256_pair_count"], 0);
    assert_eq!(summary["summary_baseline_sha256_declared_count"], 1);
    assert_eq!(summary["summary_baseline_sha256_computed_count"], 1);
    assert_eq!(summary["summary_baseline_sha256_count"], 0);
    assert_eq!(summary["calibration_sha256_declared_count"], 1);
    assert_eq!(summary["calibration_sha256_computed_count"], 1);
    assert_eq!(summary["calibration_sha256_count"], 0);
    assert_eq!(
        summary["fixtures"][0]["component1_sha256"]["status"],
        "matched"
    );
    assert_eq!(
        summary["fixtures"][0]["component2_sha256"]["status"],
        "matched"
    );
    assert_eq!(
        summary["fixtures"][0]["summary_baseline_sha256"]["status"],
        "matched"
    );
    assert_eq!(
        summary["fixtures"][0]["calibration_library_sha256"]["status"],
        "matched"
    );
    assert_eq!(summary["fixtures"][0]["validation_ready"], false);
}

#[test]
fn test_validate_cli_fixture_coverage_rejects_declared_but_unusable_assets() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path1 = tmp.path().join("left.tiff");
    let path2 = tmp.path().join("right.tiff");
    let baseline_path = tmp.path().join("baseline.json");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("coverage.json");
    let missing_library = tmp.path().join("missing-calibration");

    let component = synthetic::constant_image(4, 4, [12000, 7000, 3000]);
    scanstitch::tiff_io::save_tiff_u16(&component, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&component, &path2).unwrap();
    std::fs::write(&baseline_path, b"{not valid json").unwrap();
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "min_component_pairs": 1,
                "min_summary_baselines": 1,
                "min_calibrated_fixtures": 1,
                "min_uncalibrated_fixtures": 1
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "summary_baseline": baseline_path,
                    "calibration_library": missing_library,
                    "scanner_profile": "scanner-a",
                    "roll_profile": "roll-a",
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "scanner-roll-library"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-coverage")
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .output()
        .expect("run scanstitch-validate fixture coverage");

    assert!(
        !output.status.success(),
        "fixture coverage should fail strict unusable registry assets"
    );
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["status"], "review_required");
    assert_eq!(summary["fixture_count"], 1);
    assert_eq!(summary["validation_ready_fixture_count"], 0);
    assert_eq!(summary["component_pair_available_count"], 1);
    assert_eq!(summary["summary_baseline_declared_count"], 1);
    assert_eq!(summary["summary_baseline_file_count"], 1);
    assert_eq!(summary["summary_baseline_parseable_count"], 0);
    assert_eq!(summary["summary_baseline_contract_complete_count"], 0);
    assert_eq!(summary["summary_baseline_count"], 0);
    assert_eq!(summary["calibration_evidence_declared_count"], 1);
    assert_eq!(summary["calibration_evidence_count"], 0);
    assert_eq!(summary["calibration_evidence_unusable_count"], 1);
    assert_eq!(summary["uncalibrated_fixture_count"], 0);
    assert_eq!(
        summary["fixtures"][0]["summary_baseline_parse_status"],
        "invalid"
    );
    assert_eq!(summary["fixtures"][0]["summary_baseline_valid"], false);
    assert_eq!(
        summary["fixtures"][0]["calibration_library_selection_status"],
        "missing"
    );
    assert_eq!(
        summary["fixtures"][0]["calibration_evidence_declared"],
        true
    );
    assert_eq!(summary["fixtures"][0]["calibration_evidence_usable"], false);
    assert_eq!(summary["fixtures"][0]["validation_ready"], false);
    assert_eq!(
        summary["fixtures"][0]["action_items"],
        serde_json::json!([
            "repair_summary_baseline_for_fixture:logan",
            "provide_calibration_library_for_fixture:logan"
        ])
    );
    let repair_plan = summary["fixtures"][0]["repair_plan"].as_array().unwrap();
    assert!(repair_plan.iter().any(|item| {
        item["action"] == "repair_summary_baseline_for_fixture:logan"
            && item["paths"] == serde_json::json!([baseline_path.display().to_string()])
            && item["details"]
                .as_str()
                .is_some_and(|details| details.contains("baseline"))
    }));
    assert!(repair_plan.iter().any(|item| {
        item["action"] == "provide_calibration_library_for_fixture:logan"
            && item["paths"] == serde_json::json!([missing_library.display().to_string()])
            && item["details"]
                .as_str()
                .is_some_and(|details| details.contains("calibration library"))
    }));
    let issues = summary["issues"].as_array().unwrap();
    assert!(issues
        .iter()
        .any(|issue| issue == "logan:summary_baseline_invalid"));
    assert!(issues
        .iter()
        .any(|issue| issue == "logan:calibration_library_missing"));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_no_validation_ready_fixtures"));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_min_fixtures_not_met"));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_min_summary_baselines_not_met"));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_min_calibrated_fixtures_not_met"));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_min_uncalibrated_fixtures_not_met"));
    let action_items = summary["action_items"].as_array().unwrap();
    assert!(action_items
        .iter()
        .any(|action| action == "repair_summary_baseline_for_fixture:logan"));
    assert!(action_items
        .iter()
        .any(|action| action == "provide_calibration_library_for_fixture:logan"));
}

#[test]
fn test_validate_cli_fixture_coverage_rejects_incomplete_summary_baseline_contract() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path1 = tmp.path().join("left.tiff");
    let path2 = tmp.path().join("right.tiff");
    let baseline_path = tmp.path().join("baseline.json");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("coverage.json");
    let summary_md = tmp.path().join("coverage.md");

    let component = synthetic::constant_image(4, 4, [12000, 7000, 3000]);
    scanstitch::tiff_io::save_tiff_u16(&component, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&component, &path2).unwrap();
    std::fs::write(
        &baseline_path,
        serde_json::json!({ "fixture": "logan" }).to_string(),
    )
    .unwrap();
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "min_component_pairs": 1,
                "min_summary_baselines": 1,
                "min_uncalibrated_fixtures": 1,
                "min_unique_film_stocks": 1,
                "min_scene_tags": 1,
                "min_exposure_tags": 1,
                "min_calibration_cases": 1,
                "required_film_stocks": ["Synthetic negative"],
                "required_scene_tags": ["neutral-target"],
                "required_exposure_tags": ["normal-exposure"],
                "required_calibration_cases": ["uncalibrated-image-derived"]
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "summary_baseline": baseline_path,
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "uncalibrated-image-derived"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-coverage")
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate fixture coverage with incomplete baseline");

    assert!(
        !output.status.success(),
        "fixture coverage should fail strict incomplete baseline contract"
    );
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["status"], "review_required");
    assert_eq!(summary["summary_baseline_file_count"], 1);
    assert_eq!(summary["summary_baseline_parseable_count"], 1);
    assert_eq!(summary["summary_baseline_contract_complete_count"], 0);
    assert_eq!(summary["summary_baseline_count"], 0);
    assert_eq!(summary["validation_ready_fixture_count"], 0);
    assert_eq!(summary["uncalibrated_fixture_count"], 0);
    assert_eq!(summary["film_stock_count"], 0);
    assert_eq!(summary["scene_tag_count"], 0);
    assert_eq!(summary["exposure_tag_count"], 0);
    assert_eq!(summary["calibration_case_count"], 0);
    assert_eq!(summary["unique_film_stocks"], serde_json::json!([]));
    assert_eq!(summary["scene_tags"], serde_json::json!([]));
    assert_eq!(summary["exposure_tags"], serde_json::json!([]));
    assert_eq!(summary["calibration_cases"], serde_json::json!([]));
    assert_eq!(
        summary["missing_required_film_stocks"],
        serde_json::json!(["Synthetic negative"])
    );
    assert_eq!(
        summary["missing_required_scene_tags"],
        serde_json::json!(["neutral-target"])
    );
    assert_eq!(
        summary["missing_required_exposure_tags"],
        serde_json::json!(["normal-exposure"])
    );
    assert_eq!(
        summary["missing_required_calibration_cases"],
        serde_json::json!(["uncalibrated-image-derived"])
    );
    assert_eq!(
        summary["fixtures"][0]["summary_baseline_parse_status"],
        "valid"
    );
    assert_eq!(
        summary["fixtures"][0]["summary_baseline_contract_status"],
        "incomplete"
    );
    assert_eq!(summary["fixtures"][0]["summary_baseline_valid"], false);
    assert!(
        summary["fixtures"][0]["summary_baseline_contract_missing_fields"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field == "colorspace.selected_candidate")
    );
    assert_eq!(
        summary["fixtures"][0]["action_items"],
        serde_json::json!(["complete_summary_baseline_for_fixture:logan"])
    );
    let issues = summary["issues"].as_array().unwrap();
    assert!(issues
        .iter()
        .any(|issue| { issue == "logan:summary_baseline_incomplete:render.render_input_source" }));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_min_summary_baselines_not_met"));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_min_uncalibrated_fixtures_not_met"));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_min_unique_film_stocks_not_met"));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_min_scene_tags_not_met"));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_min_exposure_tags_not_met"));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_min_calibration_cases_not_met"));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_required_film_stock_missing:Synthetic negative"));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_required_scene_tag_missing:neutral-target"));
    assert!(issues
        .iter()
        .any(|issue| issue == "fixture_coverage_required_exposure_tag_missing:normal-exposure"));
    assert!(issues.iter().any(|issue| {
        issue == "fixture_coverage_required_calibration_case_missing:uncalibrated-image-derived"
    }));
    assert!(summary["action_items"]
        .as_array()
        .unwrap()
        .iter()
        .any(|action| action == "complete_summary_baseline_for_fixture:logan"));

    let markdown = std::fs::read_to_string(&summary_md).unwrap();
    assert!(markdown.contains("valid/incomplete"));
    assert!(markdown.contains("complete_summary_baseline_for_fixture:logan"));
}

#[test]
fn test_validate_cli_fixture_coverage_fails_unmet_registry_requirements() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path1 = tmp.path().join("left.tiff");
    let path2 = tmp.path().join("right.tiff");
    let baseline_path = tmp.path().join("baseline.json");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("coverage.json");
    let summary_md = tmp.path().join("coverage.md");

    let component = synthetic::constant_image(4, 4, [12000, 7000, 3000]);
    scanstitch::tiff_io::save_tiff_u16(&component, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&component, &path2).unwrap();
    std::fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&fixture_summary_baseline()).unwrap(),
    )
    .unwrap();
    let library_dir = write_validation_fixture_library(tmp.path());
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_unique_film_stocks": 2,
                "min_unique_scanner_profiles": 2,
                "min_unique_roll_profiles": 2,
                "min_readable_tiff_pairs": 1,
                "min_tiff_layout_consistent_pairs": 1,
                "min_tiff_dimension_matched_pairs": 1,
                "min_tiff_bits_per_sample": 14,
                "min_film_stock_calibration_pairs": 2,
                "min_scene_exposure_pairs": 2,
                "min_reference_fixtures": 1,
                "min_reference_evidence_types": 1,
                "min_reference_patch_fixtures": 1,
                "min_reference_patch_count": 4,
                "min_debug_artifact_expectation_fixtures": 1,
                "required_film_stocks": ["Synthetic negative", "Kodak Gold 200"],
                "required_scene_tags": ["neutral-target", "skin-tone"],
                "required_exposure_tags": ["normal-exposure", "underexposed-negative"],
                "required_calibration_cases": ["scanner-roll-library", "uncalibrated-image-derived"],
                "required_scanner_profiles": ["scanner-a", "scanner-b"],
                "required_roll_profiles": ["roll-a", "roll-b"],
                "required_reference_evidence": ["gray-card"],
                "required_film_stock_calibration_pairs": [
                    "Synthetic negative|scanner-roll-library",
                    "Synthetic negative|uncalibrated-image-derived"
                ],
                "required_scene_exposure_pairs": [
                    "neutral-target|normal-exposure",
                    "skin-tone|normal-exposure"
                ],
                "required_debug_artifact_kinds": ["candidate_comparison"]
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "output_dir": tmp.path().join("out"),
                    "summary_baseline": baseline_path,
                    "calibration_library": library_dir,
                    "scanner_profile": "scanner-a",
                    "roll_profile": "roll-a",
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "scanner-roll-library"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-coverage")
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate fixture coverage");

    assert!(
        !output.status.success(),
        "fixture coverage should fail strict unmet requirements"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("fixture_coverage_min_unique_film_stocks_not_met"));
    assert!(stderr.contains("fixture_coverage_min_unique_scanner_profiles_not_met"));
    assert!(stderr.contains("fixture_coverage_min_unique_roll_profiles_not_met"));
    assert!(stderr.contains("fixture_coverage_min_film_stock_calibration_pairs_not_met"));
    assert!(stderr.contains("fixture_coverage_min_scene_exposure_pairs_not_met"));
    assert!(stderr.contains("fixture_coverage_min_reference_fixtures_not_met"));
    assert!(stderr.contains("fixture_coverage_min_reference_evidence_types_not_met"));
    assert!(stderr.contains("fixture_coverage_min_reference_patch_fixtures_not_met"));
    assert!(stderr.contains("fixture_coverage_min_reference_patch_count_not_met"));
    assert!(stderr.contains("fixture_coverage_min_debug_artifact_expectation_fixtures_not_met"));
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["status"], "review_required");
    assert_eq!(summary["film_stock_calibration_pair_count"], 1);
    assert_eq!(summary["scene_exposure_pair_count"], 1);
    assert_eq!(
        summary["missing_required_film_stocks"],
        serde_json::json!(["Kodak Gold 200"])
    );
    assert_eq!(
        summary["missing_required_scene_tags"],
        serde_json::json!(["skin-tone"])
    );
    assert_eq!(
        summary["missing_required_exposure_tags"],
        serde_json::json!(["underexposed-negative"])
    );
    assert_eq!(
        summary["missing_required_calibration_cases"],
        serde_json::json!(["uncalibrated-image-derived"])
    );
    assert_eq!(
        summary["missing_required_scanner_profiles"],
        serde_json::json!(["scanner-b"])
    );
    assert_eq!(
        summary["missing_required_roll_profiles"],
        serde_json::json!(["roll-b"])
    );
    assert_eq!(summary["reference_fixture_count"], 0);
    assert_eq!(summary["reference_evidence_type_count"], 0);
    assert_eq!(summary["reference_patch_fixture_count"], 0);
    assert_eq!(summary["reference_patch_count"], 0);
    assert_eq!(summary["reference_evidence"], serde_json::json!([]));
    assert_eq!(summary["debug_artifact_expectation_fixture_count"], 0);
    assert_eq!(
        summary["debug_artifact_kinds_required"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["missing_required_reference_evidence"],
        serde_json::json!(["gray-card"])
    );
    assert_eq!(
        summary["missing_required_debug_artifact_kinds"],
        serde_json::json!(["candidate_comparison"])
    );
    assert_eq!(
        summary["missing_required_film_stock_calibration_pairs"],
        serde_json::json!(["Synthetic negative|uncalibrated-image-derived"])
    );
    assert_eq!(
        summary["missing_required_scene_exposure_pairs"],
        serde_json::json!(["skin-tone|normal-exposure"])
    );
    assert_eq!(
        summary["action_items"],
        serde_json::json!([
            "add_unique_scanner_profile_coverage:1",
            "add_unique_roll_profile_coverage:1",
            "add_unique_film_stock_coverage:1",
            "add_film_stock_calibration_pairs:1",
            "add_scene_exposure_pairs:1",
            "add_reference_fixtures:1",
            "add_reference_evidence_types:1",
            "add_reference_patch_fixtures:1",
            "add_reference_patches:4",
            "add_debug_artifact_expectation_fixtures:1",
            "add_validation_ready_fixture_for_film_stock:Kodak Gold 200",
            "add_validation_ready_fixture_for_scene_tag:skin-tone",
            "add_validation_ready_fixture_for_exposure_tag:underexposed-negative",
            "add_validation_ready_fixture_for_calibration_case:uncalibrated-image-derived",
            "add_validation_ready_fixture_for_scanner_profile:scanner-b",
            "add_validation_ready_fixture_for_roll_profile:roll-b",
            "add_validation_ready_fixture_with_reference_evidence:gray-card",
            "declare_debug_artifact_expectation_for_validation_ready_fixture:candidate_comparison",
            "add_validation_ready_fixture_for_film_stock_calibration_pair:Synthetic negative|uncalibrated-image-derived",
            "add_validation_ready_fixture_for_scene_exposure_pair:skin-tone|normal-exposure"
        ])
    );
    assert!(summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_coverage_required_scene_tag_missing:skin-tone"));
    assert!(summary["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_coverage_required_calibration_case_missing:uncalibrated-image-derived"
    }));
    assert!(summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_coverage_required_scanner_profile_missing:scanner-b"));
    assert!(summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_coverage_required_roll_profile_missing:roll-b"));
    assert!(summary["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_coverage_required_reference_evidence_missing:gray-card"
    }));
    assert!(summary["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_coverage_required_debug_artifact_kind_missing:candidate_comparison"
    }));
    assert!(summary["issues"].as_array().unwrap().iter().any(|issue| {
        issue
            == "fixture_coverage_required_film_stock_calibration_pair_missing:Synthetic negative|uncalibrated-image-derived"
    }));
    assert!(summary["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_coverage_required_scene_exposure_pair_missing:skin-tone|normal-exposure"
    }));
    let markdown = std::fs::read_to_string(&summary_md).unwrap();
    assert!(markdown.contains("min unique film stocks"));
    assert!(markdown.contains("min unique scanner profiles"));
    assert!(markdown.contains("min unique roll profiles"));
    assert!(markdown.contains("min film stock + calibration pairs"));
    assert!(markdown.contains("min scene + exposure pairs"));
    assert!(markdown.contains("min TIFF layout-consistent pairs"));
    assert!(markdown.contains("min TIFF dimension-matched pairs"));
    assert!(markdown.contains("min reference fixtures"));
    assert!(markdown.contains("min reference evidence types"));
    assert!(markdown.contains("min reference-patch fixtures"));
    assert!(markdown.contains("min reference patches"));
    assert!(markdown.contains("min debug-artifact expectation fixtures"));
    assert!(markdown.contains("missing required scanner profiles"));
    assert!(markdown.contains("scanner-b"));
    assert!(markdown.contains("missing required roll profiles"));
    assert!(markdown.contains("roll-b"));
    assert!(markdown.contains("missing required film stocks"));
    assert!(markdown.contains("Kodak Gold 200"));
    assert!(markdown.contains("missing required scene tags"));
    assert!(markdown.contains("skin-tone"));
    assert!(markdown.contains("missing required exposure tags"));
    assert!(markdown.contains("underexposed-negative"));
    assert!(markdown.contains("missing required calibration cases"));
    assert!(markdown.contains("uncalibrated-image-derived"));
    assert!(markdown.contains("missing required reference evidence"));
    assert!(markdown.contains("gray-card"));
    assert!(markdown.contains("missing required film stock + calibration pairs"));
    assert!(markdown.contains("Synthetic negative|uncalibrated-image-derived"));
    assert!(markdown.contains("missing required scene + exposure pairs"));
    assert!(markdown.contains("skin-tone|normal-exposure"));
    assert!(markdown.contains("missing required debug artifact kinds"));
    assert!(markdown.contains("candidate_comparison"));
    assert!(markdown.contains("## Action Items"));
    assert!(markdown.contains("add_unique_film_stock_coverage:1"));
    assert!(markdown.contains("add_reference_patches:4"));
    assert!(markdown.contains("add_validation_ready_fixture_for_scene_tag:skin-tone"));
    assert!(markdown.contains(
        "declare_debug_artifact_expectation_for_validation_ready_fixture:candidate_comparison"
    ));
}

#[test]
fn test_validate_cli_fixture_coverage_reports_tiff_pair_mismatches() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path1 = tmp.path().join("left.tiff");
    let path2 = tmp.path().join("right.tiff");
    let baseline_path = tmp.path().join("baseline.json");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("coverage.json");

    let left = synthetic::constant_image(4, 4, [12000, 7000, 3000]);
    let right = synthetic::constant_image(5, 4, [12000, 7000, 3000]);
    scanstitch::tiff_io::save_tiff_u16(&left, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&right, &path2).unwrap();
    std::fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&fixture_summary_baseline()).unwrap(),
    )
    .unwrap();
    let library_dir = write_validation_fixture_library(tmp.path());
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "min_component_pairs": 1,
                "min_readable_tiff_pairs": 1,
                "min_tiff_layout_consistent_pairs": 1,
                "min_tiff_dimension_matched_pairs": 1,
                "min_summary_baselines": 1,
                "min_calibrated_fixtures": 1
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "summary_baseline": baseline_path,
                    "calibration_library": library_dir,
                    "scanner_profile": "scanner-a",
                    "roll_profile": "roll-a",
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "scanner-roll-library"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-coverage")
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .output()
        .expect("run scanstitch-validate fixture coverage with mismatched TIFF dimensions");

    assert!(
        !output.status.success(),
        "fixture coverage should fail strict TIFF pair mismatch"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("fixture_coverage_min_tiff_dimension_matched_pairs_not_met"));
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["readable_tiff_pair_count"], 1);
    assert_eq!(summary["tiff_layout_consistent_pair_count"], 1);
    assert_eq!(summary["tiff_dimension_matched_pair_count"], 0);
    assert_eq!(
        summary["fixtures"][0]["tiff_pair"]["layout_consistent"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["tiff_pair"]["dimension_matched"],
        false
    );
    assert_eq!(
        summary["fixtures"][0]["tiff_pair"]["dimensions_match"],
        false
    );
    assert!(summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "logan:tiff_pair_dimensions_mismatch"));
    assert_eq!(
        summary["fixtures"][0]["action_items"],
        serde_json::json!([
            "replace_component_pair_with_dimension_matched_tiffs_for_fixture:logan"
        ])
    );
    assert!(summary["fixtures"][0]["repair_plan"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| {
            item["action"]
                == "replace_component_pair_with_dimension_matched_tiffs_for_fixture:logan"
                && item["paths"]
                    == serde_json::json!([path1.display().to_string(), path2.display().to_string()])
                && item["details"]
                    .as_str()
                    .is_some_and(|details| details.contains("dimensions match"))
        }));
    assert!(summary["action_items"]
        .as_array()
        .unwrap()
        .iter()
        .any(|action| action
            == "replace_component_pair_with_dimension_matched_tiffs_for_fixture:logan"));
}

#[test]
fn test_validate_cli_fixture_coverage_reports_tiff_pair_layout_mismatches() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path1 = tmp.path().join("left.tiff");
    let path2 = tmp.path().join("right.tiff");
    let baseline_path = tmp.path().join("baseline.json");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("coverage.json");

    let left = synthetic::constant_image(4, 4, [12000, 7000, 3000]);
    scanstitch::tiff_io::save_tiff_u16(&left, &path1).unwrap();
    write_rgba8_tiff(&path2, 4, 4);
    std::fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&fixture_summary_baseline()).unwrap(),
    )
    .unwrap();
    let library_dir = write_validation_fixture_library(tmp.path());
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "min_component_pairs": 1,
                "min_readable_tiff_pairs": 1,
                "min_tiff_layout_consistent_pairs": 1,
                "min_tiff_dimension_matched_pairs": 1,
                "min_summary_baselines": 1,
                "min_calibrated_fixtures": 1
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "summary_baseline": baseline_path,
                    "calibration_library": library_dir,
                    "scanner_profile": "scanner-a",
                    "roll_profile": "roll-a",
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "scanner-roll-library"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-coverage")
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .output()
        .expect("run scanstitch-validate fixture coverage with mismatched TIFF layout");

    assert!(
        !output.status.success(),
        "fixture coverage should fail strict TIFF pair layout mismatch"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("fixture_coverage_min_tiff_layout_consistent_pairs_not_met"));
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["readable_tiff_pair_count"], 1);
    assert_eq!(summary["tiff_layout_consistent_pair_count"], 0);
    assert_eq!(summary["tiff_dimension_matched_pair_count"], 1);
    assert_eq!(
        summary["fixtures"][0]["tiff_pair"]["layout_consistent"],
        false
    );
    assert_eq!(
        summary["fixtures"][0]["tiff_pair"]["dimension_matched"],
        true
    );
    assert!(summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "logan:tiff_pair_layout_mismatch"));
    assert!(summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "logan:tiff_pair_bits_per_sample_mismatch"));
    assert_eq!(
        summary["fixtures"][0]["action_items"],
        serde_json::json!(["replace_component_pair_with_layout_matched_tiffs_for_fixture:logan"])
    );
}

#[test]
fn test_validate_cli_roll_inventory_inspects_scan_dir_and_reports_sequence_gaps() {
    let tmp = tempfile::TempDir::new().unwrap();
    let roll_dir = tmp.path().join("TESTROLL");
    std::fs::create_dir_all(&roll_dir).unwrap();
    let frame0 = roll_dir.join("RAW_0000.tif");
    let frame2 = roll_dir.join("RAW_0002.tif");
    write_rgba8_tiff(&frame0, 4, 3);
    write_rgba8_tiff(&frame2, 4, 3);
    std::fs::write(roll_dir.join("notes.txt"), "ignored").unwrap();

    let summary_json = tmp.path().join("roll-inventory.json");
    let summary_md = tmp.path().join("roll-inventory.md");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-inventory")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate roll inventory");

    assert!(
        output.status.success(),
        "roll inventory CLI failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["status"], "review_required");
    assert_eq!(summary["roll_name"], "TESTROLL");
    assert_eq!(summary["frame_count"], 2);
    assert_eq!(summary["usable_frame_count"], 2);
    assert_eq!(summary["unreadable_frame_count"], 0);
    assert_eq!(summary["sequence_gap_count"], 1);
    assert_eq!(
        summary["sequence_gaps"][0]["missing"],
        serde_json::json!(["RAW_0001"])
    );
    assert_eq!(summary["frames"][0]["name"], "RAW_0000.tif");
    assert_eq!(summary["frames"][0]["width"], 4);
    assert_eq!(summary["frames"][0]["height"], 3);
    assert_eq!(summary["frames"][0]["source_bits_per_sample"], 8);
    assert_eq!(summary["frames"][0]["source_channel_count"], 4);
    assert!(summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue
            .as_str()
            .is_some_and(|issue| issue.starts_with("roll_inventory:sequence_gap:RAW_0001"))));
    assert!(std::fs::read_to_string(&summary_md)
        .unwrap()
        .contains("# Roll Inventory: TESTROLL"));
}

#[test]
fn test_validate_cli_roll_inventory_positive_flags_negative_like_input() {
    let tmp = tempfile::TempDir::new().unwrap();
    let roll_dir = tmp.path().join("TESTROLL");
    std::fs::create_dir_all(&roll_dir).unwrap();
    for idx in 0..2 {
        let frame = synthetic::gradient_image(
            48,
            72,
            [22000 + idx as u16 * 500, 14000, 5000],
            [36000 + idx as u16 * 500, 22000, 9000],
        );
        scanstitch::tiff_io::save_tiff_u16(&frame, &roll_dir.join(format!("RAW_{idx:04}.tif")))
            .unwrap();
    }

    let summary_json = tmp.path().join("roll-inventory-positive.json");
    let summary_md = tmp.path().join("roll-inventory-positive.md");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-inventory")
        .arg("--input-mode")
        .arg("positive")
        .arg("--quiet")
        .arg("--bit-depth")
        .arg("16")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run positive roll inventory");

    assert!(
        output.status.success(),
        "positive roll inventory CLI failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["status"], "review_required");
    assert_eq!(summary["issues"].as_array().unwrap().len(), 2);
    assert!(summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .all(|issue| issue
            .as_str()
            .is_some_and(|issue| issue.contains("positive_input_negative_like"))));
    for frame in summary["frames"].as_array().unwrap() {
        assert_eq!(frame["positive_input_probe"]["likely_negative_like"], true);
        assert!(frame["positive_input_probe"]["orange_mask_score"]
            .as_f64()
            .is_some_and(|score| score > 0.9));
        assert!(frame["positive_input_probe"]["reason"]
            .as_str()
            .is_some_and(|reason| reason.contains("orange-mask-like channel bias")));
    }
    let markdown = std::fs::read_to_string(&summary_md).unwrap();
    assert!(markdown.contains("Positive Input"));
    assert!(markdown.contains("negative-like score"));

    let strict_output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-inventory")
        .arg("--input-mode")
        .arg("positive")
        .arg("--quiet")
        .arg("--strict")
        .arg("--bit-depth")
        .arg("16")
        .arg("--output-dir")
        .arg(tmp.path().join("roll-inventory-positive-strict-output"))
        .output()
        .expect("run strict positive roll inventory");
    assert!(
        !strict_output.status.success(),
        "strict positive roll inventory should fail on negative-like input"
    );
    let strict_stderr = String::from_utf8_lossy(&strict_output.stderr);
    assert!(
        strict_stderr.contains("strict roll inventory failed")
            && strict_stderr.contains("positive_input_negative_like"),
        "unexpected strict stderr: {strict_stderr}"
    );
}

#[test]
fn test_validate_cli_roll_inventory_positive_labels_accepted_warm_input() {
    let tmp = tempfile::TempDir::new().unwrap();
    let roll_dir = tmp.path().join("OLD_SLIDE_TESTROLL");
    std::fs::create_dir_all(&roll_dir).unwrap();
    let warm_positive = synthetic::constant_image(16, 16, [27415, 22332, 20330]);
    scanstitch::tiff_io::save_tiff_u16(
        &warm_positive,
        &roll_dir.join("BIRDING047_VARIED_EXPOSURE.tif"),
    )
    .unwrap();

    let summary_json = tmp.path().join("roll-inventory-warm-positive.json");
    let summary_md = tmp.path().join("roll-inventory-warm-positive.md");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-inventory")
        .arg("--input-mode")
        .arg("positive")
        .arg("--quiet")
        .arg("--bit-depth")
        .arg("16")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run warm positive roll inventory");

    assert!(
        output.status.success(),
        "warm positive roll inventory CLI failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["status"], "passed");
    assert_eq!(summary["issues"].as_array().unwrap().len(), 0);
    let probe = &summary["frames"][0]["positive_input_probe"];
    assert_eq!(probe["likely_negative_like"], false);
    assert_eq!(probe["accepted_high_warm_score"], true);
    assert!(probe["orange_mask_score"]
        .as_f64()
        .is_some_and(|score| score > 0.95));
    let reason = probe["reason"].as_str().unwrap_or_default();
    assert!(reason.contains("high warm-channel score"));
    assert!(reason.contains("mild channel ratios"));
    assert!(reason.contains("accepted as already-positive"));

    let markdown = std::fs::read_to_string(&summary_md).unwrap();
    assert!(markdown.contains("Positive Input"));
    assert!(markdown.contains("ok warm score"));
    assert!(!markdown.contains("negative-like score"));
}

#[test]
fn test_validate_cli_roll_suite_applies_roll_consensus_base_source() {
    let tmp = tempfile::TempDir::new().unwrap();
    let roll_dir = tmp.path().join("TESTROLL");
    std::fs::create_dir_all(&roll_dir).unwrap();
    for idx in 0..4 {
        let mut frame = synthetic::constant_image(120, 240, [3600, 3300, 3000]);
        for y in 40..64 {
            for x in 120..148 {
                frame[[y, x, 0]] = 24000 + idx as u16 * 80;
                frame[[y, x, 1]] = 11800 + idx as u16 * 30;
                frame[[y, x, 2]] = 7200 + idx as u16 * 20;
            }
        }
        scanstitch::tiff_io::save_tiff_u16(&frame, &roll_dir.join(format!("RAW_{idx:04}.tif")))
            .unwrap();
    }

    let summary_json = tmp.path().join("roll-suite.json");
    let summary_md = tmp.path().join("roll-suite.md");
    let output_dir = tmp.path().join("roll-output");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-suite")
        .arg("--quiet")
        .arg("--bit-depth")
        .arg("16")
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate roll suite");

    assert!(
        output.status.success(),
        "roll suite CLI failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["roll_base_source"], "roll_consensus_base");
    assert_eq!(summary["roll_base_frame_count"], 4);
    assert_eq!(summary["roll_base_candidate_count"], 4);
    assert!(
        summary["roll_base_color"][0].as_f64().unwrap() > 23_000.0,
        "roll base should come from the high-transmittance fallback cluster: {}",
        summary["roll_base_color"]
    );
    assert!(
        summary["frames"]
            .as_array()
            .unwrap()
            .iter()
            .all(|frame| frame["base_estimate_source"] == "roll_consensus_base"),
        "each rendered frame should report roll consensus base provenance"
    );
    assert!(
        summary["frames"]
            .as_array()
            .unwrap()
            .iter()
            .all(|frame| { frame["base_estimate_source"] != "manual_base_color_override" }),
        "roll consensus must not be reported as a manual override"
    );
    assert_eq!(summary["quality"]["frame_count"], 4);
    assert_eq!(summary["review"]["frame_count"], 4);
    assert!(
        summary["review"]["candidate_risk_counts"]
            .as_array()
            .is_some_and(|counts| !counts.is_empty()),
        "roll suite should aggregate candidate-risk review causes"
    );
    assert!(
        summary["review"]["tone_color_trust_state_counts"]
            .as_array()
            .is_some_and(|counts| !counts.is_empty()),
        "roll suite should aggregate tone-colour trust review causes"
    );
    assert!(
        summary["review"]["calibration_status_counts"]
            .as_array()
            .is_some_and(|counts| !counts.is_empty()),
        "roll suite should aggregate calibration status review causes"
    );
    assert!(
        summary["review"]["reference_patch_evaluation_missing_count"]
            .as_u64()
            .is_some(),
        "roll suite should aggregate missing reference-patch evidence"
    );
    assert!(
        summary["review"]["issue_counts"].as_array().is_some(),
        "roll suite should aggregate normalized issue kinds"
    );
    assert!(
        summary["quality"]["midtone_luminance_p50_mean"]
            .as_f64()
            .is_some(),
        "roll suite should aggregate final midtone brightness diagnostics"
    );
    assert!(
        summary["quality"]["midtone_luminance_p50_range"]
            .as_f64()
            .is_some(),
        "roll suite should aggregate midtone consistency diagnostics"
    );
    assert!(
        summary["quality"]["render_luminance_range_p05_p95_mean"]
            .as_f64()
            .is_some(),
        "roll suite should aggregate full-frame rendered luminance contrast diagnostics"
    );
    assert!(
        summary["quality"]["shadow_saturation_p95_max"]
            .as_f64()
            .is_some(),
        "roll suite should aggregate final shadow saturation diagnostics"
    );
    assert!(
        summary["quality"]["bright_neutral_rgb_balance_delta_max"]
            .as_f64()
            .is_some(),
        "roll suite should aggregate final color-balance diagnostics"
    );
    assert!(
        summary["quality"]
            .get("midtone_neutral_rgb_balance_delta_max")
            .is_some(),
        "roll suite should expose neutral-only midtone color-balance diagnostics"
    );
    assert!(
        summary["quality"]["high_frequency_frame_count"]
            .as_u64()
            .unwrap()
            > 0,
        "roll suite should surface per-frame high-frequency residual coverage"
    );
    assert!(
        summary["quality"]["high_frequency_flat_chroma_residual_p95_mean"]
            .as_f64()
            .is_some(),
        "roll suite should aggregate flat-area high-frequency residual coverage"
    );
    assert!(
        summary["quality"]["noise_reduction_enabled_count"]
            .as_u64()
            .unwrap()
            > 0,
        "roll suite should aggregate final render denoise diagnostics"
    );
    assert!(
        summary["frames"]
            .as_array()
            .unwrap()
            .iter()
            .all(
                |frame| frame.get("high_frequency_luma_residual_p95").is_some()
                    && frame.get("high_frequency_chroma_residual_p95").is_some()
                    && frame.get("high_frequency_flat_sample_ratio").is_some()
                    && frame
                        .get("high_frequency_flat_chroma_residual_p95")
                        .is_some()
                    && frame.get("midtone_luminance_p50").is_some()
                    && frame.get("render_luminance_p05").is_some()
                    && frame.get("render_luminance_p50").is_some()
                    && frame.get("render_luminance_p95").is_some()
                    && frame.get("render_luminance_range_p05_p95").is_some()
                    && frame.get("shadow_saturation_p95").is_some()
                    && frame.get("midtone_neutral_pixel_count").is_some()
                    && frame.get("midtone_neutral_saturation_p95").is_some()
                    && frame.get("bright_neutral_saturation_p95").is_some()
                    && frame.get("shadow_rgb_balance_delta").is_some()
                    && frame.get("midtone_rgb_balance_delta").is_some()
                    && frame.get("midtone_neutral_rgb_balance_delta").is_some()
                    && frame.get("bright_neutral_rgb_balance_delta").is_some()
                    && frame.get("noise_reduction_applied_ratio").is_some()
                    && frame.get("colorspace_post_scale_preserved_ratio").is_some()
            ),
        "each roll-suite frame should expose quality diagnostics for roll-level audit"
    );
    let summary_md = std::fs::read_to_string(&summary_md).unwrap();
    assert!(summary_md.contains("Roll Base Clusters"));
    assert!(summary_md.contains("Roll Review Audit"));
    assert!(summary_md.contains("Candidate risk counts"));
    assert!(summary_md.contains("Review issue counts"));
    assert!(summary_md.contains("Roll Quality Diagnostics"));
    assert!(summary_md.contains("Render luminance p05-p95 range"));
    assert!(summary_md.contains("Midtone luminance p50"));
    assert!(summary_md.contains("mean/min/max/range"));
    assert!(summary_md.contains("Shadow Sat p95"));
    assert!(summary_md.contains("Mid Neutral"));
    assert!(summary_md.contains("Flat Chroma"));
    assert!(summary_md.contains("RGB Delta"));
    assert!(summary_md.contains("Frame Quality Diagnostics"));

    let subset_summary_json = tmp.path().join("roll-suite-subset.json");
    let subset_summary_md = tmp.path().join("roll-suite-subset.md");
    let subset_output_dir = tmp.path().join("roll-output-subset");
    let subset_output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-suite")
        .arg("--roll-suite-frame")
        .arg("RAW_0001.tif,raw-0003")
        .arg("--quiet")
        .arg("--bit-depth")
        .arg("16")
        .arg("--output-dir")
        .arg(&subset_output_dir)
        .arg("--summary-json")
        .arg(&subset_summary_json)
        .arg("--summary-md")
        .arg(&subset_summary_md)
        .output()
        .expect("run scanstitch-validate roll suite subset");

    assert!(
        subset_output.status.success(),
        "roll suite subset CLI failed: status={} stderr={} stdout={}",
        subset_output.status,
        String::from_utf8_lossy(&subset_output.stderr),
        String::from_utf8_lossy(&subset_output.stdout)
    );
    let subset: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&subset_summary_json).unwrap()).unwrap();
    assert_eq!(subset["frame_count"], 2);
    assert_eq!(subset["inventory"]["frame_count"], 4);
    assert_eq!(subset["roll_base_frame_count"], 4);
    assert_eq!(subset["quality"]["frame_count"], 2);
    assert_eq!(subset["review"]["frame_count"], 2);
    let subset_frames = subset["frames"].as_array().unwrap();
    assert_eq!(subset_frames[0]["name"], "RAW_0001.tif");
    assert_eq!(subset_frames[1]["name"], "RAW_0003.tif");
    assert!(
        subset_frames
            .iter()
            .all(|frame| frame["base_estimate_source"] == "roll_consensus_base"),
        "subset renders should still use the full-roll consensus base"
    );
    assert!(subset_output_dir.join("raw-0001").exists());
    assert!(subset_output_dir.join("raw-0003").exists());
    assert!(!subset_output_dir.join("raw-0000").exists());
    assert!(!subset_output_dir.join("raw-0002").exists());

    let subset_compare_summary_json = tmp.path().join("roll-suite-subset-compare.json");
    let subset_compare_summary_md = tmp.path().join("roll-suite-subset-compare.md");
    let subset_compare_output_dir = tmp.path().join("roll-output-subset-compare");
    let subset_compare_output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-suite")
        .arg("--roll-suite-frame")
        .arg("RAW_0001.tif,raw-0003")
        .arg("--quiet")
        .arg("--bit-depth")
        .arg("16")
        .arg("--compare-roll-suite")
        .arg(&summary_json)
        .arg("--output-dir")
        .arg(&subset_compare_output_dir)
        .arg("--summary-json")
        .arg(&subset_compare_summary_json)
        .arg("--summary-md")
        .arg(&subset_compare_summary_md)
        .output()
        .expect("run scanstitch-validate roll suite subset comparison");

    assert!(
        subset_compare_output.status.success(),
        "roll suite subset comparison CLI failed: status={} stderr={} stdout={}",
        subset_compare_output.status,
        String::from_utf8_lossy(&subset_compare_output.stderr),
        String::from_utf8_lossy(&subset_compare_output.stdout)
    );
    let subset_compared: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&subset_compare_summary_json).unwrap())
            .unwrap();
    assert_eq!(subset_compared["comparison"]["status"], "review_required");
    assert_eq!(
        subset_compared["comparison"]["issues"],
        serde_json::json!(["roll_suite_compare:frame_set_changed"]),
        "subset comparison should report only the intentional frame-set change"
    );
    assert_eq!(
        subset_compared["comparison"]["baseline_only_frames"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        subset_compared["comparison"]["frames"]
            .as_array()
            .unwrap()
            .len(),
        4
    );

    let legacy_summary_json = tmp.path().join("roll-suite-legacy.json");
    let mut legacy_summary = summary.clone();
    let legacy_quality = legacy_summary["quality"].as_object_mut().unwrap();
    legacy_quality.remove("midtone_luminance_p50_max");
    legacy_quality.remove("midtone_luminance_p50_range");
    std::fs::write(
        &legacy_summary_json,
        serde_json::to_string_pretty(&legacy_summary).unwrap(),
    )
    .unwrap();

    let compare_summary_json = tmp.path().join("roll-suite-compare.json");
    let compare_summary_md = tmp.path().join("roll-suite-compare.md");
    let compare_output_dir = tmp.path().join("roll-output-compare");
    let compare_output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-suite")
        .arg("--quiet")
        .arg("--bit-depth")
        .arg("16")
        .arg("--compare-roll-suite")
        .arg(&legacy_summary_json)
        .arg("--output-dir")
        .arg(&compare_output_dir)
        .arg("--summary-json")
        .arg(&compare_summary_json)
        .arg("--summary-md")
        .arg(&compare_summary_md)
        .output()
        .expect("run scanstitch-validate roll suite comparison");

    assert!(
        compare_output.status.success(),
        "roll suite comparison CLI failed: status={} stderr={} stdout={}",
        compare_output.status,
        String::from_utf8_lossy(&compare_output.stderr),
        String::from_utf8_lossy(&compare_output.stdout)
    );

    let compared: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&compare_summary_json).unwrap()).unwrap();
    assert_eq!(compared["comparison"]["status"], "comparable");
    assert_eq!(
        compared["comparison"]["issues"].as_array().unwrap().len(),
        0
    );
    assert_eq!(compared["comparison"]["frame_count_delta"], 0);
    assert_eq!(
        compared["comparison"]["quality"]["midtone_luminance_p50_mean_delta"]
            .as_f64()
            .unwrap()
            .abs(),
        0.0
    );
    assert_eq!(
        compared["comparison"]["quality"]["midtone_luminance_p50_range_delta"]
            .as_f64()
            .unwrap()
            .abs(),
        0.0
    );
    assert_eq!(
        compared["comparison"]["quality"]["render_luminance_range_p05_p95_mean_delta"]
            .as_f64()
            .unwrap()
            .abs(),
        0.0
    );
    assert_eq!(
        compared["comparison"]["frames"].as_array().unwrap().len(),
        4
    );
    assert!(
        compared["comparison"]["frames"]
            .as_array()
            .unwrap()
            .iter()
            .all(|frame| frame
                .get("high_frequency_chroma_residual_p95_delta")
                .is_some()
                && frame.get("render_luminance_range_p05_p95_delta").is_some()
                && frame
                    .get("colorspace_post_scale_preserved_ratio_delta")
                    .is_some()),
        "roll comparison should expose per-frame quality and gamut deltas"
    );
    let compare_md = std::fs::read_to_string(&compare_summary_md).unwrap();
    assert!(compare_md.contains("Roll Suite Comparison"));
    assert!(compare_md.contains("render luminance range mean"));
    assert!(compare_md.contains("Render Range d"));
    assert!(compare_md.contains("midtone p50 range"));
    assert!(compare_md.contains("chroma residual p95 mean"));
    assert!(compare_md.contains("Clip High d"));
}

#[test]
fn test_validate_cli_roll_suite_positive_skips_negative_base_workflow() {
    let tmp = tempfile::TempDir::new().unwrap();
    let roll_dir = tmp.path().join("TESTROLL");
    std::fs::create_dir_all(&roll_dir).unwrap();
    for idx in 0..3 {
        let mut frame =
            synthetic::gradient_image(90, 140, [3500, 4200, 5000], [52000, 50000, 47000]);
        for y in 24..52 {
            for x in 44..84 {
                frame[[y, x, 0]] = 18000 + idx as u16 * 700;
                frame[[y, x, 1]] = 32000 + idx as u16 * 500;
                frame[[y, x, 2]] = 46000 + idx as u16 * 300;
            }
        }
        scanstitch::tiff_io::save_tiff_u16(&frame, &roll_dir.join(format!("RAW_{idx:04}.tif")))
            .unwrap();
    }

    let summary_json = tmp.path().join("roll-suite-positive.json");
    let summary_md = tmp.path().join("roll-suite-positive.md");
    let output_dir = tmp.path().join("roll-positive-output");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-suite")
        .arg("--input-mode")
        .arg("positive")
        .arg("--quiet")
        .arg("--bit-depth")
        .arg("16")
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate positive roll suite");

    assert!(
        output.status.success(),
        "positive roll suite CLI failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(
        summary["status"], "passed",
        "positive roll suite should pass without negative-base or colour-trust review issues"
    );
    assert_eq!(summary["roll_base_color"], serde_json::Value::Null);
    assert_eq!(summary["roll_base_source"], serde_json::Value::Null);
    assert_eq!(summary["roll_base_frame_count"], 0);
    assert_eq!(summary["frame_count"], 3);
    assert_eq!(summary["frames"].as_array().unwrap().len(), 3);
    assert_eq!(summary["review"]["candidate_safe_count"], 3);
    assert_eq!(summary["review"]["candidate_review_required_count"], 0);
    assert_eq!(summary["review"]["tone_color_trusted_count"], 3);
    assert_eq!(summary["review"]["tone_color_review_required_count"], 0);
    assert_eq!(
        summary["review"]["issue_counts"].as_array().unwrap().len(),
        0
    );

    let issues = summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|issue| issue.as_str())
        .collect::<Vec<_>>();
    for forbidden in [
        "base_confidence_review_required",
        "base_estimate_fallback",
        "calibration_not_applied",
        "reference_patch_evaluation_missing",
        "candidate_risk_review_required",
        "tone_color_trust_review_required",
    ] {
        assert!(
            issues.iter().all(|issue| !issue.contains(forbidden)),
            "positive roll suite emitted negative precondition issue `{forbidden}`: {issues:?}"
        );
    }

    for frame in summary["frames"].as_array().unwrap() {
        assert!(Path::new(frame["output_path"].as_str().unwrap()).exists());
        let report_path = Path::new(frame["report_path"].as_str().unwrap());
        assert!(report_path.exists());
        assert_eq!(frame["render_input_source"], "positive_scan_rgb");
        assert_eq!(
            frame["render_input_reason"],
            "already-positive input normalized without density inversion"
        );
        assert_eq!(frame["candidate_risk"], "safe");
        assert_eq!(frame["tone_color_trust_state"], "trusted");

        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(report_path).unwrap()).unwrap();
        let cli_args = report["metadata"]["cli_args"].as_array().unwrap();
        assert!(
            cli_args.iter().all(|arg| arg != "--base-color"),
            "positive roll-suite child must not receive --base-color"
        );
        let phases = report["phases"].as_array().unwrap();
        let density = phases
            .iter()
            .find(|phase| phase["name"] == "density_inversion")
            .unwrap();
        let ica = phases
            .iter()
            .find(|phase| phase["name"] == "fastica")
            .unwrap();
        let tone = phases
            .iter()
            .find(|phase| phase["name"] == "tone_mapping")
            .unwrap();
        assert_eq!(density["metrics"]["skipped"], true);
        assert_eq!(density["metrics"]["input_mode"], "positive");
        assert!(density["metrics"].get("base_color").is_none());
        assert_eq!(ica["metrics"]["skipped"], true);
        assert_eq!(ica["metrics"]["input_mode"], "positive");
        assert_eq!(tone["metrics"]["tone_fit_policy"], "positive_scan_rgb");
        assert!(
            tone["metrics"]["auto_exposure_ev"].as_f64().unwrap() <= 0.0,
            "positive roll-suite tone placement should not brighten scans"
        );
        assert!(
            tone["metrics"]["mapped_linear_percentiles"][1]
                .as_f64()
                .unwrap()
                < 0.62,
            "positive roll-suite midtones should avoid an overly light placement: {}",
            tone["metrics"]["mapped_linear_percentiles"]
        );
    }
}

#[test]
fn test_validate_cli_roll_suite_positive_flags_negative_like_input() {
    let tmp = tempfile::TempDir::new().unwrap();
    let roll_dir = tmp.path().join("TESTROLL");
    std::fs::create_dir_all(&roll_dir).unwrap();
    for idx in 0..2 {
        let frame = synthetic::gradient_image(
            64,
            96,
            [22000 + idx as u16 * 800, 14000, 5000],
            [36000 + idx as u16 * 800, 22000, 9000],
        );
        scanstitch::tiff_io::save_tiff_u16(&frame, &roll_dir.join(format!("RAW_{idx:04}.tif")))
            .unwrap();
    }

    let summary_json = tmp.path().join("roll-suite-positive-negative-like.json");
    let summary_md = tmp.path().join("roll-suite-positive-negative-like.md");
    let output_dir = tmp.path().join("roll-positive-negative-like-output");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-suite")
        .arg("--input-mode")
        .arg("positive")
        .arg("--quiet")
        .arg("--bit-depth")
        .arg("16")
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate positive roll suite");

    assert!(
        output.status.success(),
        "positive roll suite CLI failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["status"], "review_required");
    assert_eq!(summary["review_required_count"], 2);
    let issues = summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|issue| issue.as_str())
        .collect::<Vec<_>>();
    assert!(
        issues
            .iter()
            .all(|issue| issue.contains("positive_input_negative_like")),
        "unexpected roll-suite issues: {issues:?}"
    );
    let markdown = std::fs::read_to_string(&summary_md).unwrap();
    assert!(markdown.contains("positive_input_negative_like"));

    for frame in summary["frames"].as_array().unwrap() {
        assert_eq!(frame["status"], "review_required");
        assert_eq!(frame["positive_input_likely_negative_like"], true);
        assert_eq!(frame["positive_input_accepted_high_warm_score"], false);
        assert!(frame["positive_input_orange_mask_score"]
            .as_f64()
            .is_some_and(|score| score > 0.9));
        assert!(frame["positive_input_reason"]
            .as_str()
            .is_some_and(|reason| reason.contains("orange-mask-like channel bias")));

        let frame_summary_path = Path::new(frame["summary_json_path"].as_str().unwrap());
        let frame_summary: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(frame_summary_path).unwrap()).unwrap();
        assert_eq!(
            frame_summary["render"]["positive_input_likely_negative_like"],
            true
        );
        assert_eq!(
            frame_summary["render"]["positive_input_accepted_high_warm_score"],
            false
        );
        assert!(frame_summary["render"]["positive_input_orange_mask_score"]
            .as_f64()
            .is_some_and(|score| score > 0.9));
        let frame_summary_md_path = Path::new(frame["summary_md_path"].as_str().unwrap());
        let frame_summary_md = std::fs::read_to_string(frame_summary_md_path).unwrap();
        assert!(frame_summary_md.contains("positive_input_accepted_high_warm_score"));
        assert!(frame_summary["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|phase| phase["phase"] == "working_image_select"
                && phase["warnings"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|warning| warning
                        .as_str()
                        .is_some_and(|warning| warning.contains("orange-mask-like")))));
    }

    let strict_output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-suite")
        .arg("--input-mode")
        .arg("positive")
        .arg("--quiet")
        .arg("--strict")
        .arg("--bit-depth")
        .arg("16")
        .arg("--output-dir")
        .arg(tmp.path().join("roll-positive-negative-like-strict-output"))
        .output()
        .expect("run strict positive roll suite");

    assert!(
        !strict_output.status.success(),
        "strict positive roll suite should fail when input looks negative-like"
    );
    let strict_stderr = String::from_utf8_lossy(&strict_output.stderr);
    assert!(
        strict_stderr.contains("strict roll suite failed")
            && strict_stderr.contains("positive_input_negative_like"),
        "unexpected strict stderr: {strict_stderr}"
    );
}

#[test]
fn test_validate_cli_rejects_positive_ica_render_input() {
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--input-mode")
        .arg("positive")
        .arg("--render-input")
        .arg("ica")
        .output()
        .expect("run scanstitch-validate");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--input-mode positive cannot be used with --render-input ica"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn test_validate_cli_fixture_coverage_rejects_component_hash_mismatch() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path1 = tmp.path().join("left.tiff");
    let path2 = tmp.path().join("right.tiff");
    let baseline_path = tmp.path().join("baseline.json");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("coverage.json");

    let component = synthetic::constant_image(4, 4, [12000, 7000, 3000]);
    scanstitch::tiff_io::save_tiff_u16(&component, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&component, &path2).unwrap();
    let wrong_hash = "0".repeat(64);
    let component1_actual_hash = file_sha256_hex(&path1);
    let component2_hash = file_sha256_hex(&path2);
    std::fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&fixture_summary_baseline()).unwrap(),
    )
    .unwrap();
    let wrong_summary_baseline_hash = "e".repeat(64);
    let library_dir = write_validation_fixture_library(tmp.path());
    let wrong_calibration_hash = "f".repeat(64);
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "min_component_pairs": 1,
                "min_component_sha256_pairs": 1,
                "min_readable_tiff_pairs": 1,
                "min_tiff_layout_consistent_pairs": 1,
                "min_tiff_dimension_matched_pairs": 1,
                "min_summary_baselines": 1,
                "min_summary_baseline_sha256_fixtures": 1,
                "min_calibrated_fixtures": 1,
                "min_calibration_sha256_fixtures": 1
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "component1_sha256": wrong_hash,
                    "component2_sha256": component2_hash,
                    "summary_baseline": baseline_path,
                    "summary_baseline_sha256": wrong_summary_baseline_hash,
                    "calibration_library": library_dir,
                    "calibration_library_sha256": wrong_calibration_hash,
                    "scanner_profile": "scanner-a",
                    "roll_profile": "roll-a",
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "scanner-roll-library"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-coverage")
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .output()
        .expect("run scanstitch-validate fixture coverage with a hash mismatch");

    assert!(
        !output.status.success(),
        "fixture coverage should fail strict component hash mismatch"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("logan:component1_sha256_mismatch"));
    assert!(stderr.contains("logan:summary_baseline_sha256_mismatch"));
    assert!(stderr.contains("logan:calibration_library_sha256_mismatch"));
    assert!(stderr.contains("fixture_coverage_min_component_sha256_pairs_not_met"));
    assert!(stderr.contains("fixture_coverage_min_summary_baseline_sha256_fixtures_not_met"));
    assert!(stderr.contains("fixture_coverage_min_calibration_sha256_fixtures_not_met"));
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["component_sha256_declared_pair_count"], 1);
    assert_eq!(summary["component_sha256_computed_pair_count"], 1);
    assert_eq!(summary["component_sha256_pair_count"], 0);
    assert_eq!(summary["summary_baseline_sha256_declared_count"], 1);
    assert_eq!(summary["summary_baseline_sha256_computed_count"], 1);
    assert_eq!(summary["summary_baseline_sha256_count"], 0);
    assert_eq!(summary["calibration_sha256_declared_count"], 1);
    assert_eq!(summary["calibration_sha256_computed_count"], 1);
    assert_eq!(summary["calibration_sha256_count"], 0);
    assert_eq!(
        summary["fixtures"][0]["component1_sha256"]["status"],
        "mismatch"
    );
    assert_eq!(
        summary["fixtures"][0]["component1_sha256"]["expected_sha256"],
        serde_json::json!("0000000000000000000000000000000000000000000000000000000000000000")
    );
    assert_eq!(
        summary["fixtures"][0]["component1_sha256"]["actual_sha256"],
        serde_json::json!(component1_actual_hash)
    );
    assert_eq!(
        summary["fixtures"][0]["component1_sha256"]["matched"],
        false
    );
    assert_eq!(
        summary["fixtures"][0]["component2_sha256"]["status"],
        "matched"
    );
    assert_eq!(
        summary["fixtures"][0]["summary_baseline_sha256"]["status"],
        "mismatch"
    );
    assert_eq!(
        summary["fixtures"][0]["calibration_library_sha256"]["status"],
        "mismatch"
    );
    assert_eq!(summary["fixtures"][0]["validation_ready"], false);
    assert!(summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "logan:component1_sha256_mismatch"));
    assert!(summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "logan:summary_baseline_sha256_mismatch"));
    assert!(summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "logan:calibration_library_sha256_mismatch"));
}

#[test]
fn test_validate_cli_runs_fixture_suite_strict() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp =
        synthetic::film_negative_image(160, 300, 10, 30, [12000, 7000, 3000], [5400, 4300, 3500]);
    let path1 = input_dir.join("left.tiff");
    let path2 = input_dir.join("right.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let library_dir = write_validation_fixture_library(tmp.path());
    let baseline_path = tmp.path().join("baseline.json");
    let baseline_output_dir = tmp.path().join("baseline-output");
    write_fixture_suite_tracked_baseline(
        &baseline_path,
        &path1,
        &path2,
        &baseline_output_dir,
        &library_dir,
    );

    let registry_path = tmp.path().join("fixtures.json");
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "min_component_pairs": 1,
                "min_summary_baselines": 1,
                "min_calibrated_fixtures": 1,
                "min_unique_scanner_profiles": 1,
                "min_unique_roll_profiles": 1,
                "min_unique_film_stocks": 1,
                "min_scene_tags": 1,
                "min_exposure_tags": 1,
                "min_calibration_cases": 1,
                "required_scanner_profiles": ["scanner-a"],
                "required_roll_profiles": ["roll-a"],
                "required_film_stocks": ["Synthetic negative"],
                "required_scene_tags": ["neutral-target"],
                "required_exposure_tags": ["normal-exposure"],
                "required_calibration_cases": ["scanner-roll-library"]
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "output_dir": tmp.path().join("registry-output"),
                    "summary_baseline": baseline_path,
                    "calibration_library": library_dir,
                    "scanner_profile": "scanner-a",
                    "roll_profile": "roll-a",
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "reference_evidence": ["gray-card"],
                    "calibration_case": "scanner-roll-library",
                    "expectations": {
                        "stitch_decision": "skipped_pre_score",
                        "base_estimate_source": "working_edges",
                        "output_color_space": "linear_prophoto_rgb_d50",
                        "selected_candidate_rank": 1,
                        "calibration_confidence_min": 0.0,
                        "calibration_matrix_condition_number_max": 100.0,
                        "selected_quality_score_max": 10.0,
                        "technical_safety_score_max": 10.0,
                        "color_fidelity_score_max": 10.0,
                        "highlight_chroma_compressed_ratio_min": 0.0,
                        "highlight_chroma_compressed_ratio_max": 1.0,
                        "highlight_neutral_chroma_compressed_ratio_max": 1.0,
                        "shadow_chroma_compressed_ratio_max": 1.0,
                        "memory_color_penalty_max": 10.0,
                        "spatial_consistency_penalty_max": 10.0,
                        "density_monotonicity_score_min": 0.0,
                        "hue_linearity_score_min": 0.0,
                        "spatial_neutral_delta_p95_max": 10.0,
                        "debug_artifacts_required": true,
                        "debug_artifact_kinds_required": [
                            "candidate_comparison",
                            "gamut_clipping_map",
                            "scene_referred_prophoto_float"
                        ]
                    }
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let suite_output_dir = tmp.path().join("suite-output");
    let suite_json = tmp.path().join("fixture-suite.json");
    let suite_md = tmp.path().join("fixture-suite.md");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-suite")
        .arg("--strict")
        .arg("--force-no-stitch")
        .arg("--debug")
        .arg("--ica-max-iter")
        .arg("20")
        .arg("--ica-tol")
        .arg("0.0001")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(&suite_output_dir)
        .arg("--summary-json")
        .arg(&suite_json)
        .arg("--summary-md")
        .arg(&suite_md)
        .output()
        .expect("run scanstitch-validate fixture suite");

    assert!(
        output.status.success(),
        "fixture suite CLI failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let suite: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&suite_json).unwrap()).unwrap();
    assert_eq!(suite["status"], "passed");
    assert_eq!(suite["fixture_count"], 1);
    assert_eq!(suite["passed_count"], 1);
    assert_eq!(suite["coverage"]["status"], "passed");
    assert_eq!(
        suite["coverage"]["unique_scanner_profiles"],
        serde_json::json!(["scanner-a"])
    );
    assert_eq!(
        suite["coverage"]["unique_roll_profiles"],
        serde_json::json!(["roll-a"])
    );
    assert_eq!(suite["coverage"]["scene_tag_count"], 1);
    assert_eq!(suite["fixtures"][0]["name"], "logan");
    assert_eq!(suite["fixtures"][0]["status"], "passed");
    assert_eq!(suite["fixtures"][0]["coverage_validation_ready"], true);
    assert_eq!(
        suite["fixtures"][0]["coverage_issues"],
        serde_json::json!([])
    );
    assert_eq!(
        suite["fixtures"][0]["coverage_action_items"],
        serde_json::json!([])
    );
    assert_eq!(
        suite["fixtures"][0]["reference_evidence"],
        serde_json::json!(["gray-card"])
    );
    assert!(suite["fixtures"][0]["output_path"].is_string());
    assert!(suite["fixtures"][0]["output_modified_at"].is_string());
    assert_eq!(
        suite["fixtures"][0]["output_file_icc_profile_matches_report"],
        true
    );
    assert_eq!(suite["fixtures"][0]["stale_render_artifact_count"], 0);
    assert_eq!(
        suite["fixtures"][0]["summary_baseline_status"],
        "comparable"
    );
    assert_eq!(suite["fixtures"][0]["stitch_decision"], "skipped_pre_score");
    assert_eq!(
        suite["fixtures"][0]["expected_stitch_decision"],
        "skipped_pre_score"
    );
    assert_eq!(
        suite["fixtures"][0]["base_estimate_source"],
        "working_edges"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_base_estimate_source"],
        "working_edges"
    );
    assert_eq!(
        suite["fixtures"][0]["output_color_space"],
        "linear_prophoto_rgb_d50"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_output_color_space"],
        "linear_prophoto_rgb_d50"
    );
    assert!(
        suite["fixtures"][0]["render_input_source"].is_string(),
        "fixture suite should expose render-input selection"
    );
    assert!(
        suite["fixtures"][0]["render_input_reason"].is_string()
            || suite["fixtures"][0]["render_input_reason"].is_null(),
        "fixture suite should expose render-input reason"
    );
    assert!(
        suite["fixtures"][0]["selected_mapping_reason"].is_string()
            || suite["fixtures"][0]["selected_mapping_reason"].is_null(),
        "fixture suite should expose selected mapping reason"
    );
    assert_eq!(
        suite["fixtures"][0]["calibration_source"],
        "calibration_library"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_calibration_scanner_profile"],
        "scanner-a"
    );
    assert_eq!(
        suite["fixtures"][0]["calibration_scanner_profile_id"],
        "scanner-a"
    );
    assert_eq!(
        suite["fixtures"][0]["calibration_scanner_profile_status"],
        "applied"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_calibration_roll_profile"],
        "roll-a"
    );
    assert_eq!(
        suite["fixtures"][0]["calibration_roll_profile_id"],
        "roll-a"
    );
    assert_eq!(
        suite["fixtures"][0]["calibration_roll_profile_status"],
        "applied"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_calibration_source"],
        "calibration_library"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_calibration_film_stock"],
        "Synthetic negative"
    );
    assert_eq!(
        suite["fixtures"][0]["calibration_requested_film_stock"],
        "Synthetic negative"
    );
    assert!(
        suite["fixtures"][0]["calibration_acceptance_status"].is_string(),
        "fixture suite should expose calibration acceptance"
    );
    assert!(
        suite["fixtures"][0]["calibration_confidence"].is_number(),
        "fixture suite should expose calibration confidence"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_calibration_confidence_min"],
        0.0
    );
    assert!(
        suite["fixtures"][0]["calibration_matrix_condition_number"].is_number(),
        "fixture suite should expose calibration matrix condition"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_calibration_matrix_condition_number_max"],
        100.0
    );
    assert!(
        suite["fixtures"][0]["calibration_rejection_details"].is_array(),
        "fixture suite should expose calibration rejection details"
    );
    assert!(
        suite["fixtures"][0]["selection_rejections"].is_array(),
        "fixture suite should expose selection rejections"
    );
    assert_eq!(suite["fixtures"][0]["selected_candidate_rank"], 1);
    assert_eq!(suite["fixtures"][0]["expected_selected_candidate_rank"], 1);
    assert!(
        suite["fixtures"][0]["candidate_acceptance_signatures"]
            .as_array()
            .expect("candidate acceptance signatures array")
            .len()
            >= 2,
        "fixture suite should expose candidate acceptance signatures"
    );
    assert!(
        suite["fixtures"][0]["selected_quality_score"].is_number(),
        "fixture suite should expose selected color quality score"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_selected_quality_score_max"],
        10.0
    );
    assert!(
        suite["fixtures"][0]["technical_safety_score"].is_number(),
        "fixture suite should expose technical safety score"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_technical_safety_score_max"],
        10.0
    );
    assert!(
        suite["fixtures"][0]["color_fidelity_score"].is_number(),
        "fixture suite should expose color fidelity score"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_color_fidelity_score_max"],
        10.0
    );
    assert!(
        suite["fixtures"][0]["memory_color_penalty"].is_number(),
        "fixture suite should expose memory-colour penalty"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_memory_color_penalty_max"],
        10.0
    );
    assert!(
        suite["fixtures"][0]["spatial_consistency_penalty"].is_number(),
        "fixture suite should expose spatial consistency penalty"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_spatial_consistency_penalty_max"],
        10.0
    );
    assert!(
        suite["fixtures"][0]["selected_runner_up_quality_delta"].is_number()
            || suite["fixtures"][0]["selected_runner_up_quality_delta"].is_null(),
        "fixture suite should expose selected runner-up quality margin"
    );
    assert!(
        suite["fixtures"][0]["density_monotonicity_score"].is_number(),
        "fixture suite should expose density monotonicity score"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_density_monotonicity_score_min"],
        0.0
    );
    assert!(
        suite["fixtures"][0]["hue_linearity_score"].is_number(),
        "fixture suite should expose hue linearity score"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_hue_linearity_score_min"],
        0.0
    );
    assert!(
        suite["fixtures"][0]["spatial_neutral_delta_p95"].is_number(),
        "fixture suite should expose spatial neutral consistency"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_spatial_neutral_delta_p95_max"],
        10.0
    );
    assert!(
        suite["fixtures"][0]["candidate_risk"].is_string(),
        "fixture suite should expose candidate risk"
    );
    assert!(
        suite["fixtures"][0]["tone_color_trust_state"].is_string(),
        "fixture suite should expose tone-color trust state"
    );
    assert!(
        suite["fixtures"][0]["highlight_chroma_compressed_ratio"].is_number(),
        "fixture suite should expose highlight chroma compression"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_highlight_chroma_compressed_ratio_min"],
        0.0
    );
    assert_eq!(
        suite["fixtures"][0]["expected_highlight_chroma_compressed_ratio_max"],
        1.0
    );
    assert!(
        suite["fixtures"][0]["highlight_neutral_chroma_compressed_ratio"].is_number(),
        "fixture suite should expose neutral-highlight chroma compression"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_highlight_neutral_chroma_compressed_ratio_max"],
        1.0
    );
    assert!(
        suite["fixtures"][0]["shadow_chroma_compressed_ratio"].is_number(),
        "fixture suite should expose shadow chroma compression"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_shadow_chroma_compressed_ratio_max"],
        1.0
    );
    assert_eq!(
        suite["fixtures"][0]["expected_debug_artifacts_required"],
        true
    );
    assert_eq!(suite["fixtures"][0]["debug_artifact_count"], 3);
    assert_eq!(suite["fixtures"][0]["debug_artifact_invalid_count"], 0);
    assert_eq!(
        suite["fixtures"][0]["debug_artifact_kinds"],
        serde_json::json!([
            "candidate_comparison",
            "gamut_clipping_map",
            "scene_referred_prophoto_float"
        ])
    );
    assert_eq!(
        suite["fixtures"][0]["expected_debug_artifact_kinds_required"],
        serde_json::json!([
            "candidate_comparison",
            "gamut_clipping_map",
            "scene_referred_prophoto_float"
        ])
    );
    assert!(suite["issues"].as_array().unwrap().is_empty());

    let per_fixture_summary = suite_output_dir.join("logan/summary.json");
    let per_fixture_report = suite_output_dir.join("logan/report.json");
    assert!(per_fixture_summary.exists());
    assert!(per_fixture_report.exists());
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(per_fixture_summary).unwrap()).unwrap();
    assert_eq!(
        summary["summary_baseline_comparison"]["status"],
        "comparable"
    );

    let markdown = std::fs::read_to_string(&suite_md).unwrap();
    assert!(markdown.contains("# Fixture Suite"));
    assert!(markdown.contains("coverage status: `passed`"));
    assert!(!markdown.contains("## Coverage Action Items"));
    assert!(markdown.contains("Coverage"));
    assert!(markdown.contains("Coverage actions"));
    assert!(markdown.contains("Render input"));
    assert!(markdown.contains("Tone trust"));
    assert!(markdown.contains("Tone protection"));
    assert!(markdown.contains("expected skipped_pre_score; actual skipped_pre_score"));
    assert!(markdown.contains("expected working_edges; actual working_edges"));
    assert!(markdown.contains("expected linear_prophoto_rgb_d50; actual linear_prophoto_rgb_d50"));
    assert!(markdown.contains("rank expected 1; actual 1"));
    assert!(markdown.contains("acceptance signatures"));
    assert!(markdown.contains("selected max 10.000000"));
    assert!(markdown.contains("safety max 10.000000"));
    assert!(markdown.contains("fidelity max 10.000000"));
    assert!(markdown.contains("memory max 10.000000"));
    assert!(markdown.contains("spatial consistency max 10.000000"));
    assert!(markdown.contains("highlight chroma min 0.000000"));
    assert!(markdown.contains("highlight chroma max 1.000000"));
    assert!(markdown.contains("neutral highlight max 1.000000"));
    assert!(markdown.contains("shadow chroma max 1.000000"));
    assert!(markdown.contains("density min 0.000000"));
    assert!(markdown.contains("hue min 0.000000"));
    assert!(markdown.contains("spatial neutral max 10.000000"));
    assert!(markdown.contains("required; actual 3; invalid 0"));
    assert!(markdown.contains(
        "required kinds candidate_comparison, gamut_clipping_map, scene_referred_prophoto_float"
    ));
    assert!(markdown.contains(
        "actual kinds candidate_comparison, gamut_clipping_map, scene_referred_prophoto_float"
    ));
    assert!(markdown.contains("ICC"));
    assert!(markdown.contains("expected calibration_library; actual applied / calibration_library"));
    assert!(markdown.contains("confidence min 0.000000"));
    assert!(markdown.contains("condition max 100.000000"));
    assert!(markdown.contains("scanner scanner-a (expected scanner-a)"));
    assert!(markdown.contains("roll roll-a (expected roll-a)"));
    assert!(markdown.contains("logan"));
}

#[test]
fn test_validate_cli_fixture_suite_rejects_incomplete_summary_baseline_contract() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp =
        synthetic::film_negative_image(160, 300, 10, 30, [12000, 7000, 3000], [5400, 4300, 3500]);
    let path1 = input_dir.join("left.tiff");
    let path2 = input_dir.join("right.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let library_dir = write_validation_fixture_library(tmp.path());
    let baseline_path = tmp.path().join("baseline.json");
    std::fs::write(
        &baseline_path,
        serde_json::json!({ "fixture": "logan" }).to_string(),
    )
    .unwrap();

    let registry_path = tmp.path().join("fixtures.json");
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "summary_baseline": baseline_path,
                    "calibration_library": library_dir,
                    "scanner_profile": "scanner-a",
                    "roll_profile": "roll-a",
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "scanner-roll-library"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let suite_output_dir = tmp.path().join("suite-output");
    let suite_json = tmp.path().join("fixture-suite.json");
    let suite_md = tmp.path().join("fixture-suite.md");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-suite")
        .arg("--strict")
        .arg("--force-no-stitch")
        .arg("--ica-max-iter")
        .arg("20")
        .arg("--ica-tol")
        .arg("0.0001")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(&suite_output_dir)
        .arg("--summary-json")
        .arg(&suite_json)
        .arg("--summary-md")
        .arg(&suite_md)
        .output()
        .expect("run scanstitch-validate fixture suite with incomplete baseline");

    assert!(
        !output.status.success(),
        "fixture suite unexpectedly accepted incomplete baseline: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("fixture_suite:logan:summary_baseline_incomplete:stitch.decision"),
        "stderr missing incomplete baseline issue: {stderr}"
    );
    assert!(
        stderr.contains(
            "fixture_suite:logan:summary_baseline_incomplete:colorspace.selected_candidate"
        ),
        "stderr missing colorspace incomplete baseline issue: {stderr}"
    );

    let suite: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&suite_json).unwrap()).unwrap();
    assert_eq!(suite["status"], "review_required");
    assert_eq!(suite["fixtures"][0]["status"], "review_required");
    assert_eq!(suite["fixtures"][0]["coverage_validation_ready"], false);
    assert!(suite["fixtures"][0]["coverage_issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| { issue == "logan:summary_baseline_incomplete:render.render_input_source" }));
    assert_eq!(
        suite["fixtures"][0]["coverage_action_items"],
        serde_json::json!(["complete_summary_baseline_for_fixture:logan"])
    );
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| { issue == "fixture_suite:logan:coverage_not_validation_ready" }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:summary_baseline_incomplete:render.render_input_source"
    }));
}

#[test]
fn test_validate_cli_fixture_suite_fails_unmet_coverage_requirements() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp =
        synthetic::film_negative_image(160, 300, 10, 30, [12000, 7000, 3000], [5400, 4300, 3500]);
    let path1 = input_dir.join("left.tiff");
    let path2 = input_dir.join("right.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let library_dir = write_validation_fixture_library(tmp.path());
    let baseline_path = tmp.path().join("baseline.json");
    let baseline_output_dir = tmp.path().join("baseline-output");
    write_fixture_suite_tracked_baseline(
        &baseline_path,
        &path1,
        &path2,
        &baseline_output_dir,
        &library_dir,
    );

    let registry_path = tmp.path().join("fixtures.json");
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_unique_film_stocks": 2,
                "required_calibration_cases": ["scanner-roll-library", "uncalibrated-image-derived"]
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "output_dir": tmp.path().join("registry-output"),
                    "summary_baseline": baseline_path,
                    "calibration_library": library_dir,
                    "scanner_profile": "scanner-a",
                    "roll_profile": "roll-a",
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "scanner-roll-library",
                    "expectations": {
                        "stitch_decision": "not-the-stitch-decision",
                        "base_estimate_source": "not-the-base-estimate-source",
                        "output_color_space": "not-the-output-space",
                        "render_input_source": "not-the-render-input",
                        "render_input_reason_contains": "missing-render-input-reason",
                        "mapping_strategy": "not-the-mapping-strategy",
                        "selected_mapping_reason_contains": "missing-selected-mapping-reason",
                        "selected_candidate": "not-the-selected-candidate",
                        "selected_candidate_rank": 999,
                        "candidate_acceptance_signatures_required": [
                            "missing-acceptance-signature"
                        ],
                        "calibration_rejection_details_required": [
                            "missing-calibration-rejection-detail"
                        ],
                        "calibration_confidence_min": 1.0,
                        "calibration_matrix_condition_number_max": 0.0,
                        "selection_rejections_required": [
                            "missing-selection-rejection"
                        ],
                        "calibration_acceptance_status": "not-the-calibration-status",
                        "selected_quality_score_max": 0.0,
                        "technical_safety_score_max": 0.0,
                        "color_fidelity_score_max": 0.0,
                        "highlight_chroma_compressed_ratio_min": 1.0,
                        "highlight_chroma_compressed_ratio_max": 0.0,
                        "highlight_neutral_chroma_compressed_ratio_max": 0.0,
                        "shadow_chroma_compressed_ratio_max": 0.0,
                        "memory_color_penalty_max": 0.0,
                        "spatial_consistency_penalty_max": 0.0,
                        "selected_runner_up_quality_delta_min": 999.0,
                        "density_monotonicity_score_min": 2.0,
                        "hue_linearity_score_min": 2.0,
                        "saturation_preservation_median_ratio_min": 2.0,
                        "spatial_neutral_delta_p95_max": 0.0,
                        "candidate_risk": "not-the-candidate-risk",
                        "tone_color_trust_state": "not-the-tone-trust-state",
                        "reference_patch_evaluation_required": true,
                        "reference_patch_count_min": 4,
                        "reference_patch_hue_family_regression_count_max": 0,
                        "reference_patch_selected_regresses_image_derived": false,
                        "reference_patch_delta_e2000_delta_vs_image_derived_max": 0.0,
                        "reference_patch_max_delta_vs_image_derived_max": 0.0,
                        "reference_patch_delta_e_max_delta_vs_image_derived_max": 0.0,
                        "reference_patch_delta_e2000_max_delta_vs_image_derived_max": 0.0,
                        "reference_patch_rms_delta_e_max": 1.0,
                        "reference_patch_rms_delta_e2000_max": 0.75,
                        "debug_artifacts_required": true,
                        "debug_artifact_kinds_required": [
                            "candidate_comparison",
                            "gamut_clipping_map",
                            "scene_referred_prophoto_float"
                        ]
                    }
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let suite_output_dir = tmp.path().join("suite-output");
    let suite_json = tmp.path().join("fixture-suite.json");
    let suite_md = tmp.path().join("fixture-suite.md");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-suite")
        .arg("--strict")
        .arg("--force-no-stitch")
        .arg("--ica-max-iter")
        .arg("20")
        .arg("--ica-tol")
        .arg("0.0001")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(&suite_output_dir)
        .arg("--summary-json")
        .arg(&suite_json)
        .arg("--summary-md")
        .arg(&suite_md)
        .output()
        .expect("run scanstitch-validate fixture suite");

    assert!(
        !output.status.success(),
        "fixture suite should fail strict unmet coverage requirements"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("fixture_coverage_min_unique_film_stocks_not_met"));
    let suite: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&suite_json).unwrap()).unwrap();
    assert_eq!(suite["status"], "review_required");
    assert_eq!(suite["coverage"]["status"], "review_required");
    assert_eq!(suite["fixtures"][0]["status"], "review_required");
    assert_eq!(
        suite["coverage"]["fixtures"][0]["action_items"],
        serde_json::json!(["add_reference_patches_for_fixture:logan"])
    );
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_coverage_min_unique_film_stocks_not_met"));
    assert_eq!(
        suite["coverage"]["action_items"],
        serde_json::json!([
            "add_unique_film_stock_coverage:2",
            "add_validation_ready_fixture",
            "add_validation_ready_fixture_for_calibration_case:scanner-roll-library",
            "add_validation_ready_fixture_for_calibration_case:uncalibrated-image-derived",
            "add_reference_patches_for_fixture:logan"
        ])
    );
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_coverage_required_calibration_case_missing:uncalibrated-image-derived"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "logan:reference_patch_evaluation_required_without_calibration_patches"
    }));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:stitch_decision_mismatch"));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:base_estimate_source_mismatch"));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:output_color_space_mismatch"));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:render_input_source_mismatch"));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| { issue == "fixture_suite:logan:render_input_reason_missing_expected_text" }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:selected_mapping_reason_missing_expected_text"
    }));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:selected_candidate_mismatch"));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:selected_candidate_rank_mismatch"));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue
            == "fixture_suite:logan:candidate_acceptance_signature_missing:missing-acceptance-signature"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue
            == "fixture_suite:logan:calibration_rejection_detail_missing:missing-calibration-rejection-detail"
    }));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| { issue == "fixture_suite:logan:calibration_confidence_below_expected" }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:calibration_matrix_condition_number_above_expected"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:selection_rejection_missing:missing-selection-rejection"
    }));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:selected_quality_score_above_expected"));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:color_fidelity_score_above_expected"));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:selected_runner_up_quality_delta_below_expected"
    }));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:density_monotonicity_score_below_expected"));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:hue_linearity_score_below_expected"));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:saturation_preservation_median_ratio_below_expected"
    }));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:spatial_neutral_delta_p95_above_expected"));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:tone_color_trust_state_mismatch"));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:highlight_chroma_compressed_ratio_below_expected"
    }));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| { issue == "fixture_suite:logan:reference_patch_evaluation_missing" }));
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| { issue == "fixture_suite:logan:reference_patch_count_below_expected" }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:reference_patch_hue_family_regression_count_above_expected"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:reference_patch_selected_regression_mismatch"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue
            == "fixture_suite:logan:reference_patch_delta_e2000_delta_vs_image_derived_above_expected"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:reference_patch_max_delta_vs_image_derived_above_expected"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue
            == "fixture_suite:logan:reference_patch_delta_e_max_delta_vs_image_derived_above_expected"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue
            == "fixture_suite:logan:reference_patch_delta_e2000_max_delta_vs_image_derived_above_expected"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:reference_patch_rms_delta_e_above_expected"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:reference_patch_rms_delta_e2000_above_expected"
    }));
    assert_eq!(
        suite["fixtures"][0]["expected_reference_patch_count_min"],
        serde_json::json!(4)
    );
    assert_eq!(
        suite["fixtures"][0]["expected_reference_patch_hue_family_regression_count_max"],
        serde_json::json!(0)
    );
    assert_eq!(
        suite["fixtures"][0]["expected_reference_patch_selected_regresses_image_derived"],
        serde_json::json!(false)
    );
    assert_eq!(
        suite["fixtures"][0]["expected_reference_patch_delta_e2000_delta_vs_image_derived_max"],
        serde_json::json!(0.0)
    );
    assert_eq!(
        suite["fixtures"][0]["expected_reference_patch_max_delta_vs_image_derived_max"],
        serde_json::json!(0.0)
    );
    assert_eq!(
        suite["fixtures"][0]["expected_reference_patch_delta_e_max_delta_vs_image_derived_max"],
        serde_json::json!(0.0)
    );
    assert_eq!(
        suite["fixtures"][0]["expected_reference_patch_delta_e2000_max_delta_vs_image_derived_max"],
        serde_json::json!(0.0)
    );
    assert_eq!(
        suite["fixtures"][0]["expected_reference_patch_rms_delta_e2000_max"],
        serde_json::json!(0.75)
    );
    assert!(suite["fixtures"][0]["reference_patch_patch_count"].is_null());
    assert!(suite["fixtures"][0]["reference_patch_delta_e2000_delta_vs_image_derived"].is_null());
    assert!(suite["fixtures"][0]["reference_patch_max_delta_vs_image_derived"].is_null());
    assert!(suite["fixtures"][0]["reference_patch_delta_e_max_delta_vs_image_derived"].is_null());
    assert!(
        suite["fixtures"][0]["reference_patch_delta_e2000_max_delta_vs_image_derived"].is_null()
    );
    assert!(suite["fixtures"][0]["reference_patch_selected_regresses_image_derived"].is_null());
    assert_eq!(
        suite["fixtures"][0]["reference_patch_hue_family_regressions"],
        serde_json::json!([])
    );
    assert!(suite["fixtures"][0]["reference_patch_selected_rms_delta_e2000"].is_null());
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:debug_artifacts_missing"));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:debug_artifact_kind_missing:candidate_comparison"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:debug_artifact_kind_missing:gamut_clipping_map"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:debug_artifact_kind_missing:scene_referred_prophoto_float"
    }));
    let markdown = std::fs::read_to_string(&suite_md).unwrap();
    assert!(markdown.contains("coverage status: `review_required`"));
    assert!(markdown.contains("## Coverage Action Items"));
    assert!(markdown.contains("add_unique_film_stock_coverage:2"));
    assert!(markdown.contains("add_validation_ready_fixture"));
    assert!(markdown
        .contains("add_validation_ready_fixture_for_calibration_case:uncalibrated-image-derived"));
    assert!(markdown.contains("add_reference_patches_for_fixture:logan"));
    assert!(markdown.contains("expected not-the-stitch-decision"));
    assert!(markdown.contains("expected not-the-base-estimate-source"));
    assert!(markdown.contains("expected not-the-output-space"));
    assert!(markdown.contains("expected not-the-render-input"));
    assert!(markdown.contains("reason contains missing-render-input-reason"));
    assert!(markdown.contains("mapping reason contains missing-selected-mapping-reason"));
    assert!(markdown.contains("rank expected 999"));
    assert!(markdown.contains("required signatures 1"));
    assert!(markdown.contains("confidence min 1.000000"));
    assert!(markdown.contains("condition max 0.000000"));
    assert!(markdown.contains("required rejection details 1"));
    assert!(markdown.contains("required selection rejections 1"));
    assert!(markdown.contains("selected max 0.000000"));
    assert!(markdown.contains("safety max 0.000000"));
    assert!(markdown.contains("fidelity max 0.000000"));
    assert!(markdown.contains("memory max 0.000000"));
    assert!(markdown.contains("spatial consistency max 0.000000"));
    assert!(markdown.contains("highlight chroma min 1.000000"));
    assert!(markdown.contains("highlight chroma max 0.000000"));
    assert!(markdown.contains("neutral highlight max 0.000000"));
    assert!(markdown.contains("shadow chroma max 0.000000"));
    assert!(markdown.contains("runner-up min 999.000000"));
    assert!(markdown.contains("density min 2.000000"));
    assert!(markdown.contains("hue min 2.000000"));
    assert!(markdown.contains("saturation min 2.000000"));
    assert!(markdown.contains("spatial neutral max 0.000000"));
    assert!(markdown.contains("min patches 4"));
    assert!(markdown.contains("max hue regressions 0"));
    assert!(markdown.contains("max rms dE2000 vs image 0.000000"));
    assert!(markdown.contains("max rms dE 1.000000"));
    assert!(markdown.contains("max rms dE2000 0.750000"));
    assert!(markdown.contains("required; actual missing"));
    assert!(markdown.contains(
        "required kinds candidate_comparison, gamut_clipping_map, scene_referred_prophoto_float"
    ));
    assert!(markdown.contains("actual kinds missing"));
}

#[test]
fn test_validate_cli_applies_fixture_registry_calibration_defaults() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp =
        synthetic::film_negative_image(180, 320, 10, 30, [12000, 7000, 3000], [5200, 4200, 3600]);
    let path1 = input_dir.join("left.tiff");
    let path2 = input_dir.join("right.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let library_dir = write_validation_fixture_library(tmp.path());
    let output_dir = tmp.path().join("out");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("summary.json");
    let summary_md = tmp.path().join("summary.md");
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "fixtures": {
                "registry-calibrated": {
                    "component1": path1,
                    "component2": path2,
                    "output_dir": output_dir,
                    "calibration_library": library_dir,
                    "scanner_profile": "scanner-a",
                    "roll_profile": "roll-a",
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "scanner-roll-library"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture")
        .arg("registry-calibrated")
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--force-no-stitch")
        .arg("--ica-max-iter")
        .arg("20")
        .arg("--ica-tol")
        .arg("0.0001")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate with registry calibration defaults");

    assert!(
        output.status.success(),
        "registry calibration CLI failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(
        summary["colorspace"]["calibration_source"],
        "calibration_library"
    );
    assert_eq!(
        summary["colorspace"]["calibration_scanner_profile_status"],
        "applied"
    );
    assert_eq!(
        summary["colorspace"]["calibration_scanner_profile_id"],
        "scanner-a"
    );
    assert_eq!(
        summary["colorspace"]["calibration_roll_profile_status"],
        "applied"
    );
    assert_eq!(
        summary["colorspace"]["calibration_roll_profile_id"],
        "roll-a"
    );
    assert_eq!(
        summary["colorspace"]["calibration_requested_film_stock"],
        "Synthetic negative"
    );
    assert_eq!(
        summary["colorspace"]["calibration_film_stock_status"],
        "matched_roll_metadata"
    );

    let report: PipelineReport =
        serde_json::from_str(&std::fs::read_to_string(tmp.path().join("out/report.json")).unwrap())
            .unwrap();
    let colorspace_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "colorspace_mapping")
        .expect("colorspace phase");
    assert_eq!(
        colorspace_phase.metrics["calibration"]["scanner_profile"]["profile_id"],
        "scanner-a"
    );
    assert_eq!(
        colorspace_phase.metrics["calibration"]["roll_profile"]["profile_id"],
        "roll-a"
    );
    assert_eq!(
        colorspace_phase.metrics["calibration"]["requested_film_stock"],
        "Synthetic negative"
    );
}

#[test]
fn test_validate_cli_keeps_fixture_film_stock_metadata_from_requiring_library() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp =
        synthetic::film_negative_image(180, 320, 10, 30, [12000, 7000, 3000], [5200, 4200, 3600]);
    let path1 = input_dir.join("left.tiff");
    let path2 = input_dir.join("right.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let output_dir = tmp.path().join("out");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("summary.json");
    let summary_md = tmp.path().join("summary.md");
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "fixtures": {
                "registry-profile": {
                    "component1": path1,
                    "component2": path2,
                    "output_dir": output_dir,
                    "calibration_profile": "docs/color-calibration-profile.example.json",
                    "film_stock": "Metadata-only negative",
                    "scene_tags": ["neutral-target"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "external-profile"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture")
        .arg("registry-profile")
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--force-no-stitch")
        .arg("--ica-max-iter")
        .arg("20")
        .arg("--ica-tol")
        .arg("0.0001")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate with registry external profile");

    assert!(
        output.status.success(),
        "registry external profile CLI failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["colorspace"]["calibration_status"], "applied");
    assert_eq!(
        summary["colorspace"]["calibration_source"],
        "external_calibration_profile"
    );
    assert!(
        summary["colorspace"]["calibration_requested_film_stock"].is_null(),
        "registry film stock metadata should not request library film-stock selection without a calibration library"
    );

    let report: PipelineReport =
        serde_json::from_str(&std::fs::read_to_string(tmp.path().join("out/report.json")).unwrap())
            .unwrap();
    let colorspace_phase = report
        .phases
        .iter()
        .find(|phase| phase.name == "colorspace_mapping")
        .expect("colorspace phase");
    assert_eq!(
        colorspace_phase.metrics["calibration"]["source"],
        "external_calibration_profile"
    );
    assert!(colorspace_phase.metrics["calibration"]["requested_film_stock"].is_null());
}

#[test]
fn test_validation_summary_independently_inspects_output_icc_profile() {
    let tmp = tempfile::TempDir::new().unwrap();
    let output_path = tmp.path().join("output.tiff");
    scanstitch::tiff_io::save_tiff_f64_linear_prophoto(&small_rgb_image(), &output_path).unwrap();
    let mut report = fixture_report();
    set_report_output_path(&mut report, &output_path);

    let summary = scanstitch::validation::summarize_report("logan", &report);

    assert_eq!(
        summary.render.output_file_icc_profile_status.as_deref(),
        Some("valid_icc_profile")
    );
    assert_eq!(summary.render.output_file_icc_profile_embedded, Some(true));
    assert_eq!(summary.render.output_file_icc_profile_valid, Some(true));
    assert_eq!(
        summary
            .render
            .output_file_icc_profile_description
            .as_deref(),
        Some("ScanStitch linear ProPhoto RGB D50")
    );
    assert_eq!(
        summary.render.output_file_icc_profile_matches_report,
        Some(true)
    );
}

#[test]
fn test_summary_baseline_comparison_flags_missing_output_icc_profile() {
    let tmp = tempfile::TempDir::new().unwrap();
    let output_path = tmp.path().join("output.tiff");
    scanstitch::tiff_io::save_tiff_f64(&small_rgb_image(), &output_path).unwrap();
    let mut report = fixture_report();
    set_report_output_path(&mut report, &output_path);
    let summary = scanstitch::validation::summarize_report("logan", &report);

    assert_eq!(
        summary.render.output_file_icc_profile_status.as_deref(),
        Some("missing_icc_profile")
    );
    assert_eq!(
        summary.render.output_file_icc_profile_matches_report,
        Some(false)
    );

    let baseline = fixture_summary_baseline();
    let comparison =
        scanstitch::validation::compare_summary_baseline("baseline.json", &baseline, &summary);
    assert_eq!(comparison.status, "review_required");
    assert!(comparison
        .issues
        .contains(&"current_output_icc_profile_mismatch".to_string()));
}

#[test]
fn test_validation_summary_accepts_old_report_without_metadata() {
    let mut report = fixture_report();
    report.metadata = None;

    let summary = scanstitch::validation::summarize_report("old", &report);

    assert_eq!(summary.report.generated_at, None);
    assert_eq!(summary.render.source_report_generated_at, None);
    assert_eq!(
        summary.render.output_path.as_deref(),
        Some("output/output.tiff")
    );
}

#[test]
fn test_old_report_without_phase_duration_deserializes() {
    let report: PipelineReport = serde_json::from_str(
        r#"{
            "phases": [
                {
                    "name": "save",
                    "success": true,
                    "confidence": 1.0,
                    "metrics": { "output_path": "output/output.tiff" },
                    "warnings": [],
                    "errors": []
                }
            ]
        }"#,
    )
    .unwrap();

    assert_eq!(report.phases[0].duration_ms, 0);
    let summary = scanstitch::validation::summarize_report("old", &report);
    assert_eq!(
        summary.render.output_path.as_deref(),
        Some("output/output.tiff")
    );
}

#[test]
fn test_old_sparse_report_phase_defaults_deserialize() {
    let report: PipelineReport = serde_json::from_str(
        r#"{
            "phases": [
                { "name": "stitch" },
                {
                    "name": "tone_mapping",
                    "metrics": {
                        "shadow_saturation_p95": 0.2
                    }
                }
            ]
        }"#,
    )
    .unwrap();

    assert!(!report.phases[0].success);
    assert_eq!(report.phases[0].confidence, 0.0);
    assert!(report.phases[0].metrics.is_null());
    assert!(report.phases[0].warnings.is_empty());
    let summary = scanstitch::validation::summarize_report("sparse-old", &report);
    assert_eq!(summary.stitch.decision, None);
    assert_eq!(summary.tone.shadow_saturation_p95, Some(0.2));
}

#[test]
fn test_render_comparison_reports_key_deltas() {
    let mut baseline_report = fixture_report();
    let mut current_report = fixture_report();
    clear_stale_artifacts(&mut baseline_report);
    clear_stale_artifacts(&mut current_report);
    current_report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "colorspace_mapping")
        .unwrap()
        .metrics["render_input_source"] = serde_json::json!("fastica_separated_transmittance");

    let baseline = scanstitch::validation::summarize_report("baseline", &baseline_report);
    let current = scanstitch::validation::summarize_report("current", &current_report);
    let comparison = scanstitch::validation::compare_render_summaries(
        "baseline/report.json",
        &baseline.render,
        Some("current/report.json".to_string()),
        &current.render,
    );

    assert_eq!(comparison.output_dimensions_match, Some(true));
    assert_eq!(comparison.status, "review_required");
    assert!(comparison.render_input_source_changed);
    assert_eq!(comparison.issues, vec!["render_input_source_changed"]);
    assert_eq!(comparison.luma_residual_p95_ratio, Some(1.0));
}

#[test]
fn test_render_comparison_marks_identical_reports_comparable() {
    let mut report = fixture_report();
    clear_stale_artifacts(&mut report);
    let baseline = scanstitch::validation::summarize_report("baseline", &report);
    let current = scanstitch::validation::summarize_report("current", &report);
    let comparison = scanstitch::validation::compare_render_summaries(
        "baseline/report.json",
        &baseline.render,
        Some("current/report.json".to_string()),
        &current.render,
    );

    assert_eq!(comparison.status, "comparable");
    assert!(comparison.issues.is_empty());
}

#[test]
fn test_render_comparison_flags_gamut_preservation_delta() {
    let mut baseline_report = fixture_report();
    let mut current_report = fixture_report();
    clear_stale_artifacts(&mut baseline_report);
    clear_stale_artifacts(&mut current_report);
    current_report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "colorspace_mapping")
        .unwrap()
        .metrics["post_scale_preserved_ratio"] = serde_json::json!(0.94);

    let baseline = scanstitch::validation::summarize_report("baseline", &baseline_report);
    let current = scanstitch::validation::summarize_report("current", &current_report);
    let comparison = scanstitch::validation::compare_render_summaries(
        "baseline/report.json",
        &baseline.render,
        Some("current/report.json".to_string()),
        &current.render,
    );

    assert_eq!(comparison.status, "review_required");
    assert!(comparison
        .issues
        .contains(&"colorspace_gamut_preservation_changed_materially".to_string()));
}

#[test]
fn test_render_comparison_flags_colorspace_quality_score_delta() {
    let mut baseline_report = fixture_report();
    let mut current_report = fixture_report();
    clear_stale_artifacts(&mut baseline_report);
    clear_stale_artifacts(&mut current_report);
    current_report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "colorspace_mapping")
        .unwrap()
        .metrics["selected_quality_score"] = serde_json::json!(0.95);

    let baseline = scanstitch::validation::summarize_report("baseline", &baseline_report);
    let current = scanstitch::validation::summarize_report("current", &current_report);
    let comparison = scanstitch::validation::compare_render_summaries(
        "baseline/report.json",
        &baseline.render,
        Some("current/report.json".to_string()),
        &current.render,
    );

    assert_eq!(comparison.status, "review_required");
    assert!(
        (comparison
            .colorspace_selected_quality_score_delta
            .expect("quality delta")
            - 0.53)
            .abs()
            < 1e-9
    );
    assert!(comparison
        .issues
        .contains(&"colorspace_quality_score_changed_materially".to_string()));
}

#[test]
fn test_render_comparison_flags_perceptual_color_model_deltas() {
    let mut baseline_report = fixture_report();
    let mut current_report = fixture_report();
    clear_stale_artifacts(&mut baseline_report);
    clear_stale_artifacts(&mut current_report);
    let colorspace = current_report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "colorspace_mapping")
        .unwrap();
    colorspace.metrics["candidate_risk"] = serde_json::json!("review_model_plausibility");
    colorspace.metrics["tone_color_trust_state"] = serde_json::json!("review_required");
    colorspace.metrics["reference_patch_evaluation"]["selected_rms_delta_e"] =
        serde_json::json!(5.1);
    colorspace.metrics["candidate_quality_scores"][1]["color_model_quality"]
        ["density_monotonicity_score"] = serde_json::json!(0.88);
    colorspace.metrics["candidate_quality_scores"][1]["color_model_quality"]
        ["hue_linearity_score"] = serde_json::json!(0.80);
    colorspace.metrics["candidate_quality_scores"][1]["color_model_quality"]
        ["saturation_preservation_median_ratio"] = serde_json::json!(0.36);
    colorspace.metrics["candidate_quality_scores"][1]["color_model_quality"]
        ["spatial_consistency"]["neutral_delta_p95"] = serde_json::json!(0.052);
    colorspace.metrics["candidate_quality_scores"][1]["quality_components"]
        ["memory_color_penalty"] = serde_json::json!(0.13);
    colorspace.metrics["candidate_quality_scores"][1]["quality_components"]
        ["spatial_consistency_penalty"] = serde_json::json!(0.14);

    let baseline = scanstitch::validation::summarize_report("baseline", &baseline_report);
    let current = scanstitch::validation::summarize_report("current", &current_report);
    let comparison = scanstitch::validation::compare_render_summaries(
        "baseline/report.json",
        &baseline.render,
        Some("current/report.json".to_string()),
        &current.render,
    );

    assert_eq!(comparison.status, "review_required");
    assert!(comparison
        .issues
        .contains(&"colorspace_candidate_risk_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_density_monotonicity_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_hue_linearity_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_saturation_preservation_changed_materially".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_spatial_neutral_cast_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_memory_color_penalty_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_spatial_consistency_penalty_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_reference_delta_e_regressed".to_string()));
}

fn write_validation_fixture_library(root: &Path) -> std::path::PathBuf {
    let library_dir = root.join("calibration");
    std::fs::create_dir_all(library_dir.join("scanners")).unwrap();
    std::fs::create_dir_all(library_dir.join("rolls")).unwrap();
    std::fs::write(
        library_dir.join("scanners/scanner.json"),
        validation_fixture_scanner_profile_json(),
    )
    .unwrap();
    std::fs::write(
        library_dir.join("rolls/roll.json"),
        validation_fixture_roll_profile_json(),
    )
    .unwrap();
    library_dir
}

fn write_validation_fixture_library_with_reference_patches(root: &Path) -> std::path::PathBuf {
    let library_dir = root.join("calibration");
    std::fs::create_dir_all(library_dir.join("scanners")).unwrap();
    std::fs::create_dir_all(library_dir.join("rolls")).unwrap();
    let mut scanner: serde_json::Value =
        serde_json::from_str(&validation_fixture_scanner_profile_json()).unwrap();
    scanner["patches"] = serde_json::json!([
        { "id": "neutral", "rgb": [0.55, 0.55, 0.55], "xyz": [0.50, 0.52, 0.43] },
        { "id": "red", "rgb": [0.80, 0.20, 0.18], "xyz": [0.42, 0.22, 0.08] },
        { "id": "green", "rgb": [0.24, 0.72, 0.22], "xyz": [0.28, 0.48, 0.14] },
        { "id": "blue", "rgb": [0.18, 0.24, 0.80], "xyz": [0.16, 0.12, 0.48] }
    ]);
    std::fs::write(
        library_dir.join("scanners/scanner.json"),
        serde_json::to_string_pretty(&scanner).unwrap(),
    )
    .unwrap();
    std::fs::write(
        library_dir.join("rolls/roll.json"),
        validation_fixture_roll_profile_json(),
    )
    .unwrap();
    library_dir
}

fn write_fixture_suite_tracked_baseline(
    baseline_path: &Path,
    component1: &Path,
    component2: &Path,
    output_dir: &Path,
    library_dir: &Path,
) {
    let pipeline_cli = scanstitch::cli::Cli {
        component1: component1.to_path_buf(),
        component2: component2.to_path_buf(),
        output_dir: output_dir.to_path_buf(),
        calibration_profile: None,
        calibration_library: Some(library_dir.to_path_buf()),
        scanner_profile: Some("scanner-a".to_string()),
        roll_profile: Some("roll-a".to_string()),
        film_stock: Some("Synthetic negative".to_string()),
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
        ica_tol: 0.0001,
        bit_depth: 14,
        use_opencv: false,
    };
    let report = scanstitch::pipeline::run(&pipeline_cli)
        .expect("fixture suite baseline pipeline should run");
    let report_path = output_dir.join("report.json");
    report
        .save(&report_path)
        .expect("fixture suite baseline report should save");
    let summary =
        scanstitch::validation::summarize_report_with_source("logan", &report, Some(&report_path));
    let baseline = scanstitch::validation::tracked_baseline_from_summary(&summary);
    std::fs::write(
        baseline_path,
        serde_json::to_string_pretty(&baseline).unwrap(),
    )
    .expect("fixture suite tracked baseline should write");
}

fn validation_fixture_scanner_profile_json() -> String {
    let fingerprint = validation_fixture_scanner_settings_fingerprint();
    serde_json::json!({
        "schema_version": 1,
        "record_type": "scanner_profile",
        "profile_id": "scanner-a",
        "source_space": { "name": "synthetic linear scanner RGB", "encoding": "linear" },
        "scanner": { "make": "Synthetic", "model": "Validation Registry Test" },
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

fn validation_fixture_roll_profile_json() -> String {
    let fingerprint = validation_fixture_scanner_settings_fingerprint();
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

fn validation_fixture_scanner_settings_fingerprint() -> String {
    let settings = serde_json::json!({ "dpi": 3200, "mode": "positive" });
    scanstitch::color_calibration::scanner_settings_fingerprint(Some(&settings))
        .expect("synthetic scanner settings fingerprint")
}

fn fixture_summary_baseline() -> scanstitch::validation::TrackedValidationBaseline {
    serde_json::from_value(serde_json::json!({
        "fixture": "logan",
        "stitch": {
            "decision": "accepted",
            "confidence": 0.82,
            "chosen_hypothesis": "[1|2]",
            "seam_exposure_correction_applied": true
        },
        "render": {
            "output_width": 1800,
            "output_height": 1200,
            "output_color_space": "linear_prophoto_rgb_d50",
            "output_file_icc_profile_matches_report": true,
            "base_estimate_source": "working_edges",
            "raw_base_proxy_confidence": 0.12,
            "raw_base_support_fraction": 0.014,
            "render_input_source": "direct_density_transmittance",
            "colorspace_mapping_strategy": "neutral_balance_weak_anchor_fallback"
        },
        "colorspace": {
            "calibration_status": "applied",
            "calibration_source": "external_calibration_profile",
            "calibration_scanner_profile_status": "applied",
            "calibration_scanner_profile_id": "scanner-a",
            "calibration_roll_profile_status": "applied",
            "calibration_roll_profile_id": "roll-a",
            "calibration_confidence": 0.95,
            "calibration_matrix_condition_number": 2.1,
            "calibration_requested_film_stock": "Synthetic 200",
            "calibration_film_stock_status": "matched",
            "calibration_film_stock_matched_roll_profiles": ["roll-a"],
            "calibration_rejection_details": [],
            "selected_candidate": "neutral_balance_fallback",
            "selected_candidate_rank": 1,
            "calibration_acceptance_status": "accepted",
            "calibration_acceptance_preferred_candidate": "calibrated_direct_profile",
            "calibration_acceptance_beats_image_derived": true,
            "candidate_risk": "review_reference_fit",
            "tone_color_trust_state": "review_required",
            "selected_quality_score": 0.42,
            "technical_safety_score": 0.12,
            "color_fidelity_score": 0.30,
            "selected_runner_up_quality_delta": 0.46,
            "density_monotonicity_score": 1.0,
            "hue_linearity_score": 0.92,
            "saturation_preservation_median_ratio": 0.76,
            "spatial_neutral_delta_p95": 0.006,
            "memory_color_penalty": 0.06,
            "spatial_consistency_penalty": 0.07,
            "post_scale_preserved_ratio": 0.999,
            "reference_patch_selected_rms_delta_e": 3.4,
            "reference_patch_selected_rms_delta_e2000": 2.9,
            "reference_patch_max_error_delta_vs_image_derived": 0.019,
            "reference_patch_delta_e_rms_delta_vs_image_derived": 1.3,
            "reference_patch_delta_e_max_delta_vs_image_derived": 2.1,
            "reference_patch_delta_e2000_rms_delta_vs_image_derived": 1.1,
            "reference_patch_delta_e2000_max_delta_vs_image_derived": 1.8,
            "reference_patch_selected_regresses_image_derived": true,
            "reference_patch_worst_hue_families": ["red"],
            "reference_patch_hue_family_regressions": ["red"],
            "reference_patch_regressed_candidates": ["calibrated_direct_profile"],
            "neutral_estimate_score": 0.92,
            "neutral_estimate_accepted": true,
            "neutral_estimate_populated_band_count": 3,
            "neutral_estimate_dominant_band_fraction": 0.78,
            "dominant_anchor_score": 0.67,
            "dominant_anchor_accepted": false,
            "dominant_anchor_unstable_channel_count": 1,
            "dominant_anchor_channel_unstable": [false, true, false],
            "channel_anchor_min_count": 0,
            "channel_anchor_low_support": [false, true, false],
            "weak_anchor_fallback_used": true,
            "gamut_fallback_used": false,
            "neutral_trim_applied": true,
            "candidate_acceptance_signatures": [
                "image_derived_matrix|kind=image_derived|strategy=image_derived_matrix|status=rejected_safety|rank=2|selected=false|eligible=true|rejected=true",
                "neutral_balance_fallback|kind=neutral_fallback|strategy=neutral_balance_weak_anchor_fallback|status=selected|rank=1|selected=true|eligible=true|rejected=false"
            ]
        },
        "tolerances": {
            "confidence_abs": 0.02,
            "base_support_fraction_abs": 0.00005,
            "tone_ratio_abs": 0.02,
            "grain_ratio_abs": 0.02,
            "colorspace_quality_abs": 0.25,
            "colorspace_preserved_abs": 0.02,
            "colorspace_score_abs": 0.05,
            "color_ratio_abs": 0.25,
            "spatial_neutral_abs": 0.03,
            "reference_xyz_abs": 0.005,
            "reference_delta_e_abs": 1.0
        },
        "tone": {
            "highlight_chroma_compressed_ratio": 0.004,
            "highlight_neutral_chroma_compressed_ratio": 0.003,
            "shadow_chroma_compressed_ratio": 0.002
        },
        "grain": {
            "luma_residual_p95": 0.006,
            "chroma_residual_p95": 0.009,
            "chroma_to_luma_p95_ratio": 1.5
        }
    }))
    .unwrap()
}

#[test]
fn test_summary_baseline_comparison_marks_matching_tracked_fields_comparable() {
    let tmp = tempfile::TempDir::new().unwrap();
    let output_path = tmp.path().join("output.tiff");
    scanstitch::tiff_io::save_tiff_f64_linear_prophoto(&small_rgb_image(), &output_path).unwrap();
    let mut report = fixture_report();
    set_report_output_path(&mut report, &output_path);
    let _debug_artifacts = write_report_color_debug_artifacts(&mut report);
    let summary = scanstitch::validation::summarize_report("logan", &report);
    let baseline = fixture_summary_baseline();

    let comparison =
        scanstitch::validation::compare_summary_baseline("baseline.json", &baseline, &summary);

    assert_eq!(comparison.status, "comparable");
    assert!(comparison.issues.is_empty());
    assert_eq!(comparison.output_dimensions_match, Some(true));
    assert!(!comparison.output_color_space_changed);
    assert_eq!(comparison.raw_base_proxy_confidence_delta, Some(0.0));
    assert_eq!(comparison.raw_base_support_fraction_delta, Some(0.0));
    assert!(!comparison.calibration_status_changed);
    assert!(!comparison.calibration_source_changed);
    assert!(!comparison.calibration_scanner_profile_status_changed);
    assert!(!comparison.calibration_scanner_profile_id_changed);
    assert!(!comparison.calibration_roll_profile_status_changed);
    assert!(!comparison.calibration_roll_profile_id_changed);
    assert_eq!(comparison.calibration_confidence_delta, Some(0.0));
    assert_eq!(
        comparison.calibration_matrix_condition_number_delta,
        Some(0.0)
    );
    assert!(!comparison.calibration_requested_film_stock_changed);
    assert!(!comparison.calibration_film_stock_status_changed);
    assert!(!comparison.calibration_film_stock_matched_roll_profiles_changed);
    assert!(!comparison.calibration_rejection_details_changed);
    assert!(!comparison.selected_candidate_changed);
    assert!(!comparison.selected_candidate_rank_changed);
    assert!(!comparison.calibration_acceptance_status_changed);
    assert_eq!(
        comparison.colorspace_technical_safety_score_delta,
        Some(0.0)
    );
    assert_eq!(comparison.colorspace_color_fidelity_score_delta, Some(0.0));
    assert_eq!(
        comparison.colorspace_selected_runner_up_quality_delta_delta,
        Some(0.0)
    );
    assert_eq!(
        comparison.colorspace_reference_patch_selected_rms_delta_e_delta,
        Some(0.0)
    );
    assert_eq!(
        comparison.colorspace_reference_patch_selected_rms_delta_e2000_delta,
        Some(0.0)
    );
    assert_eq!(
        comparison.colorspace_reference_patch_max_error_delta_vs_image_derived_delta,
        Some(0.0)
    );
    assert_eq!(
        comparison.colorspace_reference_patch_delta_e_rms_delta_vs_image_derived_delta,
        Some(0.0)
    );
    assert_eq!(
        comparison.colorspace_reference_patch_delta_e_max_delta_vs_image_derived_delta,
        Some(0.0)
    );
    assert_eq!(
        comparison.colorspace_reference_patch_delta_e2000_rms_delta_vs_image_derived_delta,
        Some(0.0)
    );
    assert_eq!(
        comparison.colorspace_reference_patch_delta_e2000_max_delta_vs_image_derived_delta,
        Some(0.0)
    );
    assert!(!comparison.colorspace_reference_patch_selected_regresses_image_derived_changed);
    assert!(!comparison.colorspace_reference_patch_hue_family_regressions_changed);
    assert_eq!(
        comparison.colorspace_neutral_estimate_score_delta,
        Some(0.0)
    );
    assert!(!comparison.colorspace_neutral_estimate_accepted_changed);
    assert!(!comparison.colorspace_neutral_estimate_populated_band_count_changed);
    assert_eq!(comparison.colorspace_dominant_anchor_score_delta, Some(0.0));
    assert!(!comparison.colorspace_dominant_anchor_accepted_changed);
    assert!(!comparison.colorspace_dominant_anchor_channel_unstable_changed);
    assert!(!comparison.colorspace_channel_anchor_low_support_changed);
    assert!(!comparison.colorspace_weak_anchor_fallback_changed);
    assert!(!comparison.colorspace_gamut_fallback_changed);
    assert!(!comparison.colorspace_neutral_trim_applied_changed);
    assert_eq!(comparison.luma_residual_p95_ratio, Some(1.0));
}

#[test]
fn test_summary_baseline_comparison_flags_tracked_color_tone_and_grain_drift() {
    let mut report = fixture_report();
    let _debug_artifacts = write_report_color_debug_artifacts(&mut report);
    let summary = scanstitch::validation::summarize_report("logan", &report);
    let mut baseline = fixture_summary_baseline();
    baseline.render.raw_base_proxy_confidence = Some(0.20);
    baseline.render.raw_base_support_fraction = Some(0.05);
    baseline.render.colorspace_mapping_strategy = Some("image_derived_matrix".to_string());
    baseline.colorspace.selected_candidate = Some("image_derived_matrix".to_string());
    baseline.colorspace.calibration_status = Some("rejected".to_string());
    baseline.colorspace.calibration_scanner_profile_status = Some("rejected".to_string());
    baseline.colorspace.calibration_scanner_profile_id = Some("other-scanner".to_string());
    baseline.colorspace.calibration_roll_profile_status = Some("rejected".to_string());
    baseline.colorspace.calibration_roll_profile_id = Some("other-roll".to_string());
    baseline.colorspace.calibration_confidence = Some(1.0);
    baseline.colorspace.calibration_matrix_condition_number = Some(0.8);
    baseline.colorspace.calibration_requested_film_stock = Some("Different 400".to_string());
    baseline.colorspace.calibration_film_stock_status = Some("unmatched".to_string());
    baseline
        .colorspace
        .calibration_film_stock_matched_roll_profiles = vec!["other-roll".to_string()];
    baseline.colorspace.calibration_rejection_details =
        vec!["old rejected calibration detail".to_string()];
    baseline.colorspace.calibration_acceptance_status = Some("rejected_quality".to_string());
    baseline.colorspace.selected_candidate_rank = Some(2);
    baseline.colorspace.selected_quality_score = Some(0.90);
    baseline.colorspace.technical_safety_score = Some(-0.50);
    baseline.colorspace.color_fidelity_score = Some(-0.10);
    baseline.colorspace.selected_runner_up_quality_delta = Some(0.80);
    baseline.colorspace.hue_linearity_score = Some(1.0);
    baseline.colorspace.memory_color_penalty = Some(0.0);
    baseline.colorspace.spatial_consistency_penalty = Some(0.0);
    baseline.colorspace.reference_patch_selected_rms_delta_e = Some(2.0);
    baseline.colorspace.reference_patch_selected_rms_delta_e2000 = Some(1.0);
    baseline
        .colorspace
        .reference_patch_max_error_delta_vs_image_derived = Some(0.0);
    baseline
        .colorspace
        .reference_patch_delta_e_rms_delta_vs_image_derived = Some(0.0);
    baseline
        .colorspace
        .reference_patch_delta_e_max_delta_vs_image_derived = Some(0.0);
    baseline
        .colorspace
        .reference_patch_delta_e2000_rms_delta_vs_image_derived = Some(0.0);
    baseline
        .colorspace
        .reference_patch_delta_e2000_max_delta_vs_image_derived = Some(0.0);
    baseline
        .colorspace
        .reference_patch_selected_regresses_image_derived = Some(false);
    baseline.colorspace.reference_patch_worst_hue_families = vec!["blue".to_string()];
    baseline.colorspace.reference_patch_hue_family_regressions = vec!["blue".to_string()];
    baseline.colorspace.reference_patch_regressed_candidates = vec!["old".to_string()];
    baseline.colorspace.neutral_estimate_score = Some(0.99);
    baseline.colorspace.neutral_estimate_accepted = Some(false);
    baseline.colorspace.neutral_estimate_populated_band_count = Some(2);
    baseline.colorspace.neutral_estimate_dominant_band_fraction = Some(0.40);
    baseline.colorspace.dominant_anchor_score = Some(0.90);
    baseline.colorspace.dominant_anchor_accepted = Some(true);
    baseline.colorspace.dominant_anchor_unstable_channel_count = Some(0);
    baseline.colorspace.dominant_anchor_channel_unstable = vec![false, false, false];
    baseline.colorspace.channel_anchor_min_count = Some(2);
    baseline.colorspace.channel_anchor_low_support = vec![false, false, false];
    baseline.colorspace.weak_anchor_fallback_used = Some(false);
    baseline.colorspace.gamut_fallback_used = Some(true);
    baseline.colorspace.neutral_trim_applied = Some(false);
    baseline.colorspace.candidate_acceptance_signatures = vec![
        "image_derived_matrix|kind=image_derived|strategy=image_derived_matrix|status=selected|rank=1|selected=true|eligible=true|rejected=false".to_string(),
    ];
    baseline.tone.highlight_chroma_compressed_ratio = Some(0.050);
    baseline.grain.luma_residual_p95 = Some(0.003);

    let comparison =
        scanstitch::validation::compare_summary_baseline("baseline.json", &baseline, &summary);

    assert_eq!(comparison.status, "review_required");
    assert!(comparison
        .issues
        .contains(&"raw_base_proxy_confidence_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"raw_base_support_fraction_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_mapping_strategy_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_selected_candidate_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"calibration_status_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"calibration_scanner_profile_status_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"calibration_scanner_profile_id_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"calibration_roll_profile_status_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"calibration_roll_profile_id_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"calibration_confidence_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"calibration_matrix_condition_number_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"calibration_requested_film_stock_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"calibration_film_stock_status_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"calibration_film_stock_matched_roll_profiles_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"calibration_rejection_details_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"calibration_acceptance_status_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_selected_candidate_rank_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_candidate_acceptance_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_quality_score_changed_materially".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_technical_safety_score_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_color_fidelity_score_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_runner_up_margin_shrank_materially".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_hue_linearity_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_memory_color_penalty_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_spatial_consistency_penalty_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_reference_delta_e_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_reference_delta_e2000_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_reference_xyz_max_vs_image_derived_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_reference_delta_e_vs_image_derived_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_reference_delta_e_max_vs_image_derived_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_reference_delta_e2000_vs_image_derived_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_reference_delta_e2000_max_vs_image_derived_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_reference_selected_regression_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_reference_worst_hue_families_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_reference_hue_family_regressions_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_reference_regressed_candidates_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_neutral_estimate_score_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_neutral_estimate_acceptance_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_neutral_estimate_band_support_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_neutral_estimate_band_dominance_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_dominant_anchor_score_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_dominant_anchor_acceptance_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_dominant_anchor_unstable_channel_count_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_dominant_anchor_channel_unstable_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_channel_anchor_min_count_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_channel_anchor_low_support_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_weak_anchor_fallback_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_gamut_fallback_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"colorspace_neutral_trim_applied_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"highlight_chroma_compression_changed_materially".to_string()));
    assert!(comparison
        .issues
        .contains(&"luma_residual_p95_changed_materially".to_string()));
}

#[test]
fn test_tracked_logan_baseline_has_required_regression_fields() {
    let baseline: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/baselines/logan_summary_baseline.json"
    ))
    .unwrap();

    assert_eq!(baseline["fixture"], "logan");
    assert_eq!(baseline["stitch"]["decision"], "accepted");
    assert_eq!(baseline["render"]["output_width"], 5959);
    assert_eq!(
        baseline["render"]["output_color_space"],
        "linear_prophoto_rgb_d50"
    );
    assert_eq!(
        baseline["render"]["output_file_icc_profile_matches_report"],
        true
    );
    assert_eq!(
        baseline["render"]["render_input_source"],
        "fastica_separated_transmittance"
    );
    assert_eq!(
        baseline["render"]["colorspace_mapping_strategy"],
        "gamut_trusted_image_matrix_blend"
    );
    assert_eq!(
        baseline["colorspace"]["candidate_risk"],
        "review_neutral_support"
    );
    assert_eq!(
        baseline["colorspace"]["calibration_status"],
        "not_configured"
    );
    assert!(baseline["colorspace"]["calibration_confidence"].is_null());
    assert!(baseline["colorspace"]["calibration_matrix_condition_number"].is_null());
    assert_eq!(
        baseline["colorspace"]["calibration_rejection_details"],
        serde_json::json!([])
    );
    assert_eq!(
        baseline["colorspace"]["selected_candidate"],
        "gamut_trusted_image_matrix_blend"
    );
    assert_eq!(baseline["colorspace"]["selected_candidate_rank"], 1);
    assert!(
        (baseline["colorspace"]["technical_safety_score"]
            .as_f64()
            .expect("technical safety score")
            - 1.283516)
            .abs()
            < 0.000001
    );
    assert!(
        (baseline["colorspace"]["color_fidelity_score"]
            .as_f64()
            .expect("color fidelity score")
            - 0.510881)
            .abs()
            < 0.000001
    );
    assert!(
        (baseline["colorspace"]["hue_linearity_score"]
            .as_f64()
            .expect("hue linearity baseline")
            - 0.835048)
            .abs()
            < 0.000001
    );
    assert_eq!(baseline["colorspace"]["memory_color_penalty"], 0.0);
    assert_eq!(baseline["colorspace"]["spatial_consistency_penalty"], 0.0);
    assert!(
        (baseline["colorspace"]["neutral_estimate_score"]
            .as_f64()
            .expect("neutral estimate score")
            - 0.812808)
            .abs()
            < 0.000001
    );
    assert_eq!(baseline["colorspace"]["neutral_estimate_accepted"], false);
    assert_eq!(
        baseline["colorspace"]["neutral_estimate_populated_band_count"],
        3
    );
    assert_eq!(baseline["colorspace"]["dominant_anchor_accepted"], false);
    assert_eq!(
        baseline["colorspace"]["dominant_anchor_unstable_channel_count"],
        1
    );
    assert_eq!(
        baseline["colorspace"]["dominant_anchor_channel_unstable"],
        serde_json::json!([false, false, true])
    );
    assert_eq!(baseline["colorspace"]["channel_anchor_min_count"], 587336);
    assert_eq!(
        baseline["colorspace"]["channel_anchor_low_support"],
        serde_json::json!([false, false, false])
    );
    assert_eq!(baseline["colorspace"]["weak_anchor_fallback_used"], false);
    assert_eq!(baseline["colorspace"]["gamut_fallback_used"], false);
    assert_eq!(baseline["colorspace"]["neutral_trim_applied"], false);
    assert_eq!(
        baseline["colorspace"]["candidate_acceptance_signatures"],
        serde_json::json!([
            "image_derived_matrix|kind=image_derived|strategy=image_derived_matrix|status=rejected_safety|rank=3|selected=false|eligible=true|rejected=true",
            "gamut_safe_image_matrix_blend|kind=image_derived|strategy=gamut_safe_image_matrix_blend|status=accepted_runner_up|rank=2|selected=false|eligible=true|rejected=false",
            "gamut_trusted_image_matrix_blend|kind=image_derived|strategy=gamut_trusted_image_matrix_blend|status=selected|rank=1|selected=true|eligible=true|rejected=false",
            "neutral_balance_fallback|kind=neutral_fallback|strategy=neutral_balance_fallback|status=available_fallback|rank=4|selected=false|eligible=true|rejected=false"
        ])
    );
    assert_eq!(
        baseline["colorspace"]["calibration_acceptance_status"],
        "not_applicable"
    );
    assert!(
        baseline["grain"]["luma_residual_p95"]
            .as_f64()
            .expect("luma residual baseline")
            > 0.0
    );
    assert!(
        baseline["tolerances"]["grain_ratio_abs"]
            .as_f64()
            .expect("grain tolerance")
            <= 0.05
    );
    assert_eq!(
        baseline["tolerances"]["base_support_fraction_abs"]
            .as_f64()
            .expect("base support tolerance"),
        0.00005
    );
    assert_eq!(
        baseline["tolerances"]["reference_delta_e_abs"]
            .as_f64()
            .expect("reference DeltaE tolerance"),
        1.0
    );
    assert_eq!(
        baseline["tolerances"]["reference_xyz_abs"]
            .as_f64()
            .expect("reference XYZ tolerance"),
        0.005
    );
}

#[test]
fn test_report_schema_examples_deserialize() {
    for contents in [
        include_str!("../docs/report-examples/current-report-minimal.json"),
        include_str!("../docs/report-examples/old-report-sparse.json"),
    ] {
        let report: PipelineReport = serde_json::from_str(contents).unwrap();
        assert!(!report.phases.is_empty());
        let summary = scanstitch::validation::summarize_report("example", &report);
        assert!(summary.render.output_path.is_some() || summary.stitch.decision.is_some());
    }

    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../docs/report-schema.schema.json")).unwrap();
    let report_schema_doc = include_str!("../docs/report-schema.md");
    assert!(report_schema_doc.contains("CIEDE2000"));
    assert_eq!(schema["required"], serde_json::json!(["phases"]));
    assert_eq!(
        schema["properties"]["phases"]["items"]["allOf"][0]["if"]["properties"]["name"]["const"],
        "colorspace_mapping"
    );
    assert_eq!(
        schema["properties"]["phases"]["items"]["allOf"][0]["then"]["properties"]["metrics"]
            ["$ref"],
        "#/$defs/colorspaceMappingMetrics"
    );
    assert!(
        schema["$defs"]["colorspaceMappingMetrics"]["properties"]["candidate_quality_scores"]
            .is_object()
    );
    assert_eq!(
        schema["$defs"]["colorspaceMappingMetrics"]["properties"]["candidate_acceptance"]["items"]
            ["$ref"],
        "#/$defs/candidateAcceptance"
    );
    assert_eq!(
        schema["$defs"]["colorspaceMappingMetrics"]["properties"]["calibration_acceptance"]["$ref"],
        "#/$defs/calibrationAcceptance"
    );
    assert!(
        schema["$defs"]["candidateAcceptance"]["properties"]["eligible_in_color_mode"].is_object()
    );
    assert_eq!(
        schema["$defs"]["candidateAcceptance"]["properties"]["candidate_kind"]["enum"],
        serde_json::json!([
            "calibrated_direct",
            "scanner_prior",
            "image_derived",
            "neutral_fallback"
        ])
    );
    assert!(
        schema["$defs"]["candidateQualityScore"]["properties"]["pre_scale_clipped_low_total"]
            .is_object()
    );
    assert!(schema["$defs"]["candidateQualityScore"]["properties"]
        ["reference_patch_rms_delta_e2000"]
        .is_object());
    assert!(schema["$defs"]["candidateQualityScore"]["properties"]
        ["reference_patch_delta_e2000_delta_vs_image_derived"]
        .is_object());
    assert!(schema["$defs"]["candidateQualityScore"]["properties"]
        ["reference_patch_max_delta_vs_image_derived"]
        .is_object());
    assert!(schema["$defs"]["candidateQualityScore"]["properties"]
        ["reference_patch_delta_e_max_delta_vs_image_derived"]
        .is_object());
    assert!(schema["$defs"]["candidateQualityScore"]["properties"]
        ["reference_patch_delta_e2000_max_delta_vs_image_derived"]
        .is_object());
    assert!(
        schema["$defs"]["referencePatchEvaluation"]["properties"]["selected_rms_delta_e2000"]
            .is_object()
    );
    assert!(schema["$defs"]["referencePatchEvaluation"]["properties"]
        ["delta_e2000_rms_delta_vs_image_derived"]
        .is_object());
    assert!(schema["$defs"]["referencePatchEvaluation"]["properties"]
        ["delta_e2000_max_delta_vs_image_derived"]
        .is_object());
    assert!(
        schema["$defs"]["calibrationMetrics"]["properties"]["matrix_condition_number"].is_object()
    );
    assert!(
        schema["$defs"]["calibrationMetrics"]["properties"]["profile_schema_version"].is_object()
    );
    assert!(
        schema["$defs"]["neutralSampleRejections"]["properties"]["chroma_threshold"].is_object()
    );
    assert!(
        schema["$defs"]["dominantAnchorSampleRejections"]["properties"]["weak_dominance"]
            .is_object()
    );
}

#[test]
fn test_validation_summary_baseline_schema_and_logan_baseline_cover_suite_contract() {
    let baseline: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/baselines/logan_summary_baseline.json"
    ))
    .unwrap();
    let parsed: scanstitch::validation::TrackedValidationBaseline =
        serde_json::from_value(baseline.clone()).unwrap();
    assert_eq!(parsed.fixture.as_deref(), Some("logan"));

    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/validation-summary-baseline.schema.json"
    ))
    .unwrap();
    assert_eq!(
        schema["required"],
        serde_json::json!([
            "fixture",
            "stitch",
            "render",
            "colorspace",
            "tolerances",
            "tone",
            "grain"
        ])
    );
    assert_eq!(
        schema["properties"]["colorspace"]["$ref"],
        "#/$defs/colorspaceBaseline"
    );
    assert!(
        schema["$defs"]["colorspaceBaseline"]["properties"]["calibration_scanner_profile_id"]
            .is_object()
    );
    assert!(
        schema["$defs"]["colorspaceBaseline"]["properties"]["calibration_roll_profile_id"]
            .is_object()
    );
    assert!(schema["$defs"]["colorspaceBaseline"]["properties"]
        ["reference_patch_selected_rms_delta_e2000"]
        .is_object());
    assert!(schema["$defs"]["colorspaceBaseline"]["properties"]
        ["reference_patch_delta_e2000_rms_delta_vs_image_derived"]
        .is_object());
    assert!(schema["$defs"]["colorspaceBaseline"]["properties"]
        ["reference_patch_max_error_delta_vs_image_derived"]
        .is_object());
    assert!(schema["$defs"]["colorspaceBaseline"]["properties"]
        ["reference_patch_delta_e_max_delta_vs_image_derived"]
        .is_object());
    assert!(schema["$defs"]["colorspaceBaseline"]["properties"]
        ["reference_patch_delta_e2000_max_delta_vs_image_derived"]
        .is_object());
    assert_eq!(
        schema["$defs"]["candidateAcceptanceSignature"]["pattern"],
        "^[^|]+\\|kind=[^|]+\\|strategy=[^|]+\\|status=[^|]+\\|rank=[0-9]+\\|selected=(true|false)\\|eligible=(true|false)\\|rejected=(true|false)$"
    );
    assert!(schema["$defs"]["tolerances"]["properties"]["reference_xyz_abs"].is_object());
    assert!(schema["$defs"]["tolerances"]["properties"]["base_support_fraction_abs"].is_object());
    assert_eq!(
        schema["$defs"]["candidateAcceptanceSignatures"]["minItems"],
        1
    );
    assert_eq!(
        schema["$defs"]["renderBaseline"]["required"],
        serde_json::json!([
            "output_width",
            "output_height",
            "output_color_space",
            "output_file_icc_profile_matches_report",
            "base_estimate_source",
            "render_input_source",
            "colorspace_mapping_strategy"
        ])
    );
    assert!(
        schema["$defs"]["renderBaseline"]["properties"]["raw_base_proxy_confidence"].is_object()
    );
    assert!(
        schema["$defs"]["renderBaseline"]["properties"]["raw_base_support_fraction"].is_object()
    );
    assert_eq!(
        schema["$defs"]["toneBaseline"]["required"],
        serde_json::json!(["highlight_chroma_compressed_ratio"])
    );

    for pointer in [
        "/stitch/decision",
        "/render/output_width",
        "/render/output_height",
        "/render/output_color_space",
        "/render/output_file_icc_profile_matches_report",
        "/render/base_estimate_source",
        "/render/render_input_source",
        "/render/colorspace_mapping_strategy",
        "/colorspace/calibration_status",
        "/colorspace/selected_candidate",
        "/colorspace/calibration_acceptance_status",
        "/colorspace/candidate_risk",
        "/colorspace/tone_color_trust_state",
        "/colorspace/selected_quality_score",
        "/colorspace/technical_safety_score",
        "/colorspace/color_fidelity_score",
        "/colorspace/post_scale_preserved_ratio",
        "/colorspace/neutral_estimate_score",
        "/colorspace/dominant_anchor_accepted",
        "/colorspace/channel_anchor_min_count",
        "/colorspace/weak_anchor_fallback_used",
        "/colorspace/gamut_fallback_used",
        "/colorspace/neutral_trim_applied",
        "/colorspace/candidate_acceptance_signatures/0",
        "/tone/highlight_chroma_compressed_ratio",
    ] {
        let value = baseline
            .pointer(pointer)
            .unwrap_or_else(|| panic!("baseline missing {pointer}"));
        assert!(
            !value.is_null(),
            "baseline contract field is null: {pointer}"
        );
    }
}

#[test]
fn test_validation_fixture_registry_schema_and_example_cover_color_corpus_fields() {
    let registry: serde_json::Value =
        serde_json::from_str(include_str!("../docs/validation-fixtures.example.json")).unwrap();
    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../docs/validation-fixtures.schema.json")).unwrap();

    assert_eq!(schema["required"], serde_json::json!(["fixtures"]));
    assert_eq!(
        schema["properties"]["fixtures"]["additionalProperties"]["$ref"],
        "#/$defs/fixtureEntry"
    );
    assert_eq!(schema["$defs"]["pairLabel"]["pattern"], "^[^|]+\\|[^|]+$");
    assert_eq!(schema["$defs"]["sha256Hex"]["pattern"], "^[A-Fa-f0-9]{64}$");
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_readable_tiff_pairs"]
            .is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_component_sha256_pairs"]
            .is_object()
    );
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_summary_baseline_sha256_fixtures"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_calibration_sha256_fixtures"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_tiff_layout_consistent_pairs"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_tiff_dimension_matched_pairs"]
        .is_object());
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_tiff_bits_per_sample"]
            .is_object()
    );
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_film_stock_calibration_pairs"]
        .is_object());
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_unique_scanner_profiles"]
            .is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_unique_roll_profiles"]
            .is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_scene_exposure_pairs"]
            .is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_reference_fixtures"].is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_reference_evidence_types"]
            .is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_reference_patch_fixtures"]
            .is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_reference_patch_count"]
            .is_object()
    );
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_debug_artifact_expectation_fixtures"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["required_film_stock_calibration_pairs"]
        .is_object());
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["required_scanner_profiles"]
            .is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["required_roll_profiles"].is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["required_reference_evidence"]
            .is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["required_scene_exposure_pairs"]
            .is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["required_debug_artifact_kinds"]
            .is_object()
    );
    assert_eq!(
        schema["$defs"]["debugArtifactKind"]["enum"],
        serde_json::json!([
            "candidate_comparison",
            "gamut_clipping_map",
            "scene_referred_prophoto_float"
        ])
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["required"],
        serde_json::json!(["component1", "component2"])
    );
    assert!(schema["$defs"]["fixtureEntry"]["properties"]["film_stock"].is_object());
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["component1_sha256"]["$ref"],
        "#/$defs/sha256Hex"
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["component2_sha256"]["$ref"],
        "#/$defs/sha256Hex"
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["summary_baseline_sha256"]["$ref"],
        "#/$defs/sha256Hex"
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["calibration_profile_sha256"]["$ref"],
        "#/$defs/sha256Hex"
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["calibration_library_sha256"]["$ref"],
        "#/$defs/sha256Hex"
    );
    assert!(schema["$defs"]["fixtureEntry"]["properties"]["scene_tags"].is_object());
    assert!(schema["$defs"]["fixtureEntry"]["properties"]["exposure_tags"].is_object());
    assert!(schema["$defs"]["fixtureEntry"]["properties"]["reference_evidence"].is_object());
    assert!(schema["$defs"]["fixtureEntry"]["properties"]["calibration_case"].is_object());
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["expectations"]["$ref"],
        "#/$defs/fixtureExpectations"
    );
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]["stitch_decision"].is_object());
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["base_estimate_source"].is_object()
    );
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]["output_color_space"].is_object());
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["render_input_reason_contains"]
            .is_object()
    );
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["selected_mapping_reason_contains"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]["selected_candidate"].is_object());
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["selected_candidate_rank"].is_object()
    );
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["candidate_acceptance_signatures_required"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["calibration_rejection_details_required"]
        .is_object());
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["calibration_confidence_min"]
            .is_object()
    );
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["calibration_matrix_condition_number_max"]
        .is_object());
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["selection_rejections_required"]
            .is_object()
    );
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["selected_quality_score_max"]
            .is_object()
    );
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["technical_safety_score_max"]
            .is_object()
    );
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["color_fidelity_score_max"]
            .is_object()
    );
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["highlight_chroma_compressed_ratio_min"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["highlight_chroma_compressed_ratio_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["highlight_neutral_chroma_compressed_ratio_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["shadow_chroma_compressed_ratio_max"]
        .is_object());
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["memory_color_penalty_max"]
            .is_object()
    );
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["spatial_consistency_penalty_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["selected_runner_up_quality_delta_min"]
        .is_object());
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["density_monotonicity_score_min"]
            .is_object()
    );
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["hue_linearity_score_min"].is_object()
    );
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["saturation_preservation_median_ratio_min"]
        .is_object());
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["spatial_neutral_delta_p95_max"]
            .is_object()
    );
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["post_scale_preserved_ratio_min"]
            .is_object()
    );
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["reference_patch_evaluation_required"]
        .is_object());
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["reference_patch_count_min"]
            .is_object()
    );
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["reference_patch_hue_family_regression_count_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["reference_patch_selected_regresses_image_derived"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["reference_patch_delta_e2000_delta_vs_image_derived_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["reference_patch_max_delta_vs_image_derived_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["reference_patch_delta_e_max_delta_vs_image_derived_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["reference_patch_delta_e2000_max_delta_vs_image_derived_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["reference_patch_rms_delta_e_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["reference_patch_rms_delta_e2000_max"]
        .is_object());
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["debug_artifacts_required"]
            .is_object()
    );
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["debug_artifact_kinds_required"]
            .is_object()
    );
    assert!(
        schema["$defs"]["fixtureEntry"]["allOf"]
            .as_array()
            .expect("fixture entry conditionals")
            .len()
            >= 6
    );

    assert!(registry["fixtures"]["logan"]["calibration_library"].is_string());
    assert_eq!(
        registry["fixtures"]["logan"]["calibration_case"],
        "scanner-roll-library"
    );
    assert_eq!(
        registry["fixtures"]["logan"]["scanner_profile"],
        "coolscan-4000-vuescan-raw"
    );
    assert_eq!(
        registry["fixtures"]["logan"]["roll_profile"],
        "logan-roll-2026-05"
    );
    assert_eq!(
        registry["fixtures"]["logan"]["reference_evidence"],
        serde_json::json!(["gray-card"])
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["debug_artifacts_required"],
        serde_json::json!(true)
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["debug_artifact_kinds_required"],
        serde_json::json!([
            "candidate_comparison",
            "gamut_clipping_map",
            "scene_referred_prophoto_float"
        ])
    );
    assert!(registry["fixtures"]["my-split-frame"]["calibration_profile"].is_string());
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["reference_evidence"],
        serde_json::json!(["colorchecker"])
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["reference_patch_evaluation_required"],
        serde_json::json!(true)
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]["reference_patch_count_min"],
        serde_json::json!(24)
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["reference_patch_hue_family_regression_count_max"],
        serde_json::json!(0)
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["reference_patch_selected_regresses_image_derived"],
        serde_json::json!(false)
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["reference_patch_delta_e2000_delta_vs_image_derived_max"],
        serde_json::json!(0.0)
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["reference_patch_max_delta_vs_image_derived_max"],
        serde_json::json!(0.0)
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["reference_patch_delta_e_max_delta_vs_image_derived_max"],
        serde_json::json!(0.0)
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["reference_patch_delta_e2000_max_delta_vs_image_derived_max"],
        serde_json::json!(0.0)
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]["reference_patch_rms_delta_e_max"],
        serde_json::json!(6.0)
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["reference_patch_rms_delta_e2000_max"],
        serde_json::json!(4.0)
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]["debug_artifact_kinds_required"],
        serde_json::json!([
            "candidate_comparison",
            "gamut_clipping_map",
            "scene_referred_prophoto_float"
        ])
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["calibration_case"],
        "external-profile"
    );
    assert_eq!(
        registry["fixtures"]["uncalibrated-night"]["calibration_case"],
        "uncalibrated-image-derived"
    );
    assert_eq!(
        registry["fixtures"]["uncalibrated-night"]["expectations"]["debug_artifact_kinds_required"],
        serde_json::json!([
            "candidate_comparison",
            "gamut_clipping_map",
            "scene_referred_prophoto_float"
        ])
    );
    assert!(
        registry["fixtures"]["uncalibrated-night"]["calibration_profile"].is_null(),
        "uncalibrated example must not declare a calibration profile"
    );
    assert!(
        registry["fixtures"]["uncalibrated-night"]["calibration_library"].is_null(),
        "uncalibrated example must not declare a calibration library"
    );
    assert_eq!(
        registry["coverage_requirements"]["required_film_stock_calibration_pairs"],
        serde_json::json!([
            "Kodak Gold 200|scanner-roll-library",
            "Fujicolor 200|external-profile",
            "Kodak Ultramax 400|uncalibrated-image-derived"
        ])
    );
    assert_eq!(
        registry["coverage_requirements"]["min_readable_tiff_pairs"],
        serde_json::json!(3)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_tiff_layout_consistent_pairs"],
        serde_json::json!(3)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_tiff_dimension_matched_pairs"],
        serde_json::json!(3)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_tiff_bits_per_sample"],
        serde_json::json!(14)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_reference_fixtures"],
        serde_json::json!(2)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_reference_evidence_types"],
        serde_json::json!(2)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_reference_patch_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_reference_patch_count"],
        serde_json::json!(24)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_debug_artifact_expectation_fixtures"],
        serde_json::json!(3)
    );
    assert_eq!(
        registry["coverage_requirements"]["required_scene_tags"],
        serde_json::json!([
            "skin-tone",
            "foliage",
            "high-saturation",
            "high-key",
            "deep-shadow",
            "low-neutral"
        ])
    );
    assert_eq!(
        registry["coverage_requirements"]["required_scanner_profiles"],
        serde_json::json!(["coolscan-4000-vuescan-raw"])
    );
    assert_eq!(
        registry["coverage_requirements"]["required_roll_profiles"],
        serde_json::json!(["logan-roll-2026-05"])
    );
    assert_eq!(
        registry["coverage_requirements"]["required_reference_evidence"],
        serde_json::json!(["colorchecker", "gray-card"])
    );
    assert_eq!(
        registry["coverage_requirements"]["required_scene_exposure_pairs"],
        serde_json::json!([
            "skin-tone|normal-exposure",
            "foliage|overexposed-negative",
            "high-saturation|overexposed-negative",
            "high-key|overexposed-negative",
            "deep-shadow|underexposed-negative",
            "low-neutral|underexposed-negative"
        ])
    );
    assert_eq!(
        registry["coverage_requirements"]["required_debug_artifact_kinds"],
        serde_json::json!([
            "candidate_comparison",
            "gamut_clipping_map",
            "scene_referred_prophoto_float"
        ])
    );
}

#[test]
fn test_validation_docs_map_local_corpus_scaffold_to_registry_actions() {
    let validation = include_str!("../docs/validation.md");
    let gitignore = include_str!("../.gitignore");

    for ignored_path in ["/local-fixtures/", "/calibration/", "/*.tif", "/*.tiff"] {
        assert!(
            gitignore.contains(ignored_path),
            "local fixture path `{ignored_path}` must stay ignored"
        );
    }

    for required in [
        "Local Corpus Scaffold",
        "not a passing corpus by itself",
        "calibration/scanners/coolscan-4000-vuescan-raw.json",
        "calibration/rolls/logan-roll-2026-05.json",
        "calibration/film_hints/kodak-gold-200.json",
        "local-fixtures/my-split-left.tif",
        "local-fixtures/my-split-right.tif",
        "local-fixtures/baselines/my-split-frame-summary-baseline.json",
        "local-fixtures/calibration/my-split-frame-profile.json",
        "local-fixtures/uncalibrated-night-left.tif",
        "local-fixtures/uncalibrated-night-right.tif",
        "local-fixtures/baselines/uncalibrated-night-summary-baseline.json",
        "repair_component1_tiff_for_fixture:logan",
        "repair_component2_tiff_for_fixture:logan",
        "replace_component1_with_minimum_bit_depth_tiff_for_fixture:logan",
        "replace_component2_with_minimum_bit_depth_tiff_for_fixture:logan",
        "replace_component_pair_with_dimension_matched_tiffs_for_fixture:logan",
        "replace_component_pair_with_layout_matched_tiffs_for_fixture:<name>",
        "RGBA8 5959x3670",
        "RGBA8 5959x3669",
        "min_tiff_bits_per_sample: 14",
        "provide_calibration_library_for_fixture:logan",
        "Kodak Gold 200|scanner-roll-library",
        "skin-tone|normal-exposure",
        "provide_component1_for_fixture:my-split-frame",
        "provide_calibration_profile_for_fixture:my-split-frame",
        "add_reference_patches_for_fixture:my-split-frame",
        "Fujicolor 200|external-profile",
        "foliage|overexposed-negative",
        "high-saturation|overexposed-negative",
        "high-key|overexposed-negative",
        "provide_component1_for_fixture:uncalibrated-night",
        "Kodak Ultramax 400|uncalibrated-image-derived",
        "deep-shadow|underexposed-negative",
        "low-neutral|underexposed-negative",
        "--compute-fixture-hashes",
        "--write-fixture-hash-registry",
        "--fixture-coverage --strict",
        "--fixture-suite --strict --debug",
        "coverage_validation_ready",
        "coverage_issues",
        "coverage_action_items",
        "coverage_not_validation_ready",
        "Coverage actions",
        "fixtures[].repair_plan",
        "path or paths",
        "repair detail",
        "CIEDE2000",
        "reference_patch_count_min",
        "reference_patch_hue_family_regression_count_max",
        "reference_patch_selected_regresses_image_derived",
        "reference_patch_delta_e2000_delta_vs_image_derived_max",
        "reference_patch_max_delta_vs_image_derived_max",
        "reference_patch_delta_e_max_delta_vs_image_derived_max",
        "reference_patch_delta_e2000_max_delta_vs_image_derived_max",
        "reference_patch_rms_delta_e2000_max",
        "no remaining `action_items`",
    ] {
        assert!(
            validation.contains(required),
            "validation docs missing local corpus scaffold detail `{required}`"
        );
    }
}

#[test]
fn test_color_calibration_library_record_schema_and_examples_cover_contract() {
    let examples: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/color-calibration-library-record.examples.json"
    ))
    .unwrap();
    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/color-calibration-library-record.schema.json"
    ))
    .unwrap();

    assert_eq!(
        schema["oneOf"],
        serde_json::json!([
            { "$ref": "#/$defs/scannerProfile" },
            { "$ref": "#/$defs/rollProfile" },
            { "$ref": "#/$defs/filmHint" }
        ])
    );
    assert_eq!(
        schema["$defs"]["scannerProfile"]["required"],
        serde_json::json!([
            "schema_version",
            "record_type",
            "scanner",
            "target",
            "reference",
            "whitepoint",
            "confidence"
        ])
    );
    assert_eq!(
        schema["$defs"]["scannerProfile"]["oneOf"][0]["required"],
        serde_json::json!(["scanner_rgb_to_xyz"])
    );
    assert_eq!(
        schema["$defs"]["rollProfile"]["allOf"][0]["then"]["required"],
        serde_json::json!([
            "scanner_profile_id",
            "scanner_settings_fingerprint",
            "correction_domain"
        ])
    );
    assert_eq!(
        schema["$defs"]["filmHint"]["anyOf"][0]["required"],
        serde_json::json!(["film_id"])
    );
    assert_eq!(
        schema["$defs"]["calibrationConfidence"]["minimum"],
        serde_json::json!(0.5)
    );
    assert_eq!(
        schema["$defs"]["settingsFingerprint"]["pattern"],
        "^fnv1a64:[0-9a-f]{16}$"
    );
    assert_eq!(
        schema["$defs"]["fitDiagnostics"]["additionalProperties"],
        serde_json::json!(false)
    );
    assert_eq!(
        schema["$defs"]["patchFitResidual"]["additionalProperties"],
        serde_json::json!(false)
    );

    for key in ["scanner_profile", "roll_profile", "film_hint"] {
        let value = examples
            .get(key)
            .unwrap_or_else(|| panic!("missing calibration library example {key}"));
        scanstitch::color_calibration::validate_library_record_value(value, None).unwrap_or_else(
            |rejection| {
                panic!(
                    "example {key} did not satisfy loader contract: {:?}",
                    rejection.reasons
                )
            },
        );
    }

    let scanner_fingerprint = scanstitch::color_calibration::scanner_settings_fingerprint(Some(
        &examples["scanner_profile"]["settings"],
    ))
    .expect("scanner example settings fingerprint");
    assert_eq!(
        examples["scanner_profile"]["scanner_settings_fingerprint"].as_str(),
        Some(scanner_fingerprint.as_str())
    );
    assert_eq!(
        examples["roll_profile"]["scanner_settings_fingerprint"].as_str(),
        Some(scanner_fingerprint.as_str())
    );

    let tmp = tempfile::TempDir::new().unwrap();
    let scanner_dir = tmp.path().join("scanners");
    let roll_dir = tmp.path().join("rolls");
    let film_dir = tmp.path().join("film_hints");
    std::fs::create_dir_all(&scanner_dir).unwrap();
    std::fs::create_dir_all(&roll_dir).unwrap();
    std::fs::create_dir_all(&film_dir).unwrap();
    std::fs::write(
        scanner_dir.join("example-coolscan-vuescan-raw.json"),
        serde_json::to_string_pretty(&examples["scanner_profile"]).unwrap(),
    )
    .unwrap();
    std::fs::write(
        roll_dir.join("example-logan-roll.json"),
        serde_json::to_string_pretty(&examples["roll_profile"]).unwrap(),
    )
    .unwrap();
    std::fs::write(
        film_dir.join("kodak-gold-200.json"),
        serde_json::to_string_pretty(&examples["film_hint"]).unwrap(),
    )
    .unwrap();
    let result = scanstitch::color_calibration::load_calibration(
        None,
        Some(tmp.path()),
        Some("example-coolscan-vuescan-raw"),
        Some("example-logan-roll"),
        Some("Kodak Gold 200"),
        Some([12000.0, 7000.0, 3000.0]),
    );
    let profile = result
        .profile
        .expect("schema examples should compose as a calibration library");
    assert_eq!(result.diagnostics.status, "applied");
    assert_eq!(
        profile.application_mode,
        scanstitch::color_calibration::CalibrationApplicationMode::DirectProfile
    );
    assert!(profile.roll_correction_applied);
    assert_eq!(
        result
            .diagnostics
            .library
            .as_ref()
            .map(|library| library.invalid_entries.len()),
        Some(0)
    );

    assert_eq!(
        examples["scanner_profile"]["record_type"],
        "scanner_profile"
    );
    assert_eq!(examples["roll_profile"]["record_type"], "roll_profile");
    assert_eq!(examples["film_hint"]["record_type"], "film_hint");
}

#[test]
fn test_color_reconstruction_audit_maps_goal_to_artifacts_and_gap() {
    let audit = include_str!("../docs/color-reconstruction-audit.md");

    for required in [
        "Objective-To-Artifact Checklist",
        "Current Verification Gates",
        "Current Local Evidence",
        "Missing Evidence",
        "real multi-scene, multi-film-stock CoolScan validation corpus",
        "synthetic colour gate",
        "19 passed cases",
        "reference-patch regression rejection",
        "CIEDE2000",
        "unsafe-calibration failure",
        "src/color_calibration.rs",
        "src/colorspace.rs",
        "src/bin/scanstitch-validate.rs",
        "docs/validation-fixtures.schema.json",
        "tests/fixtures/baselines/logan_summary_baseline.json",
        "component*_sha256",
        "summary_baseline_sha256",
        "calibration_library_sha256",
        "min_component_sha256_pairs",
        "min_summary_baseline_sha256_fixtures",
        "min_calibration_sha256_fixtures",
        "calibration_confidence_min",
        "candidate_acceptance_signatures_required",
        "reference_patch_count_min",
        "reference_patch_hue_family_regression_count_max",
        "reference_patch_selected_regresses_image_derived",
        "reference_patch_delta_e2000_delta_vs_image_derived_max",
        "reference_patch_max_delta_vs_image_derived_max",
        "reference_patch_delta_e_max_delta_vs_image_derived_max",
        "reference_patch_delta_e2000_max_delta_vs_image_derived_max",
        "reference_patch_rms_delta_e2000_max",
        "render_input_reason_contains",
        "highlight_chroma_compressed_ratio_max",
        "debug_artifact_kinds_required",
        "debug_artifact_expectation_fixture_count",
        "coverage_validation_ready",
        "coverage_issues",
        "coverage_action_items",
        "coverage_not_validation_ready",
        "repair_plan",
        "action/path/detail",
        "action_items",
        "provide_component1_for_fixture",
        "provide_calibration_library_for_fixture",
        "add_reference_patches_for_fixture",
        "gamut-clipping",
        "--fixture-coverage --compute-fixture-hashes",
        "--write-fixture-hash-registry",
        "--fixture-coverage --strict",
        "--fixture-suite --strict",
        "summary_baseline_status=comparable",
        "colorspace.calibration_status=not_configured",
        "candidate_comparison",
        "gamut_clipping_map",
        "debug_artifact_invalid_count=0",
        "fixture coverage hash gate",
        "fixture_coverage_status=review_required",
        "validation_ready_fixture_count=0",
        "validation-fixtures.with-hashes.json",
        "Local Corpus Scaffold",
        "calibration/scanners/coolscan-4000-vuescan-raw.json",
        "local-fixtures/my-split-left.tif",
        "repair_component1_tiff_for_fixture:logan",
        "replace_component1_with_minimum_bit_depth_tiff_for_fixture:logan",
        "replace_component_pair_with_dimension_matched_tiffs_for_fixture:logan",
        "RGBA8 5959x3670",
        "fixture_suite_status=failed",
        "passed=0",
        "review_required=1",
        "failed=2",
        "fixture_suite:logan:coverage_not_validation_ready",
        "fixture_suite:logan:declared_calibration_not_applied",
        "CoolScan requirement",
    ] {
        assert!(
            audit.contains(required),
            "colour reconstruction audit missing `{required}`"
        );
    }

    assert!(
        audit.contains("not complete") && audit.contains("should remain open"),
        "audit must explicitly prevent treating proxy coverage as completion"
    );
}

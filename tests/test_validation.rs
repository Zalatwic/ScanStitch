#![recursion_limit = "512"]

mod common;
use common::synthetic;
use ndarray::{s, Array3};
use scanstitch::report::{PhaseReport, PipelineReport, RunMetadata};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;
use std::process::Command;

fn textured_positive_panorama(height: usize, width: usize) -> Array3<u16> {
    let mut image = Array3::<u16>::zeros((height, width, 3));
    for y in 0..height {
        for x in 0..width {
            let tx = x as f64 / width.saturating_sub(1).max(1) as f64;
            let ty = y as f64 / height.saturating_sub(1).max(1) as f64;
            let feature =
                ((x.wrapping_mul(73) ^ y.wrapping_mul(151) ^ (x * y + 17)) % 997) as f64 / 997.0;
            image[[y, x, 0]] = ((0.10 + 0.55 * tx + 0.12 * feature) * 16_383.0) as u16;
            image[[y, x, 1]] = ((0.12 + 0.48 * ty + 0.10 * feature) * 16_383.0) as u16;
            image[[y, x, 2]] = ((0.14 + 0.34 * (1.0 - tx) + 0.16 * feature) * 16_383.0) as u16;
        }
    }
    image
}

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

fn write_rgb16_tiff_with_orientation(path: &Path, image: &Array3<u16>, orientation: u16) {
    let (height, width, channels) = image.dim();
    assert_eq!(channels, 3);
    let pixels = image.iter().copied().collect::<Vec<_>>();
    let file = std::fs::File::create(path).unwrap();
    let mut encoder = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file)).unwrap();
    let mut output = encoder
        .new_image::<tiff::encoder::colortype::RGB16>(width as u32, height as u32)
        .unwrap();
    output
        .encoder()
        .write_tag(tiff::tags::Tag::Orientation, orientation)
        .unwrap();
    output.write_data(&pixels).unwrap();
}

fn fixture_report() -> PipelineReport {
    let mut report = PipelineReport::new();
    report.metadata = Some(RunMetadata {
        report_schema_version: 3,
        pipeline_schema_version: 1,
        generated_at: "2026-05-01T21:00:00Z".to_string(),
        generated_at_unix_ms: 1_777_665_600_000,
        package_name: "scanstitch".to_string(),
        package_version: "0.1.0".to_string(),
        binary_name: "scanstitch".to_string(),
        binary_path: Some("V:\\PERSONAL\\PHOTOGRAPHY\\SCANSTITCH\\scanstitch.exe".to_string()),
        binary_sha256: Some("a".repeat(64)),
        binary_file_size_bytes: Some(1),
        binary_identity_status: Some("verified_sha256".to_string()),
        binary_identity_error: None,
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
            "homography_feature_validation": {
                "accepted": true,
                "reason": "spatially disjoint cross-fit passed",
                "partition_method": "4x4_whole_cell_checkerboard_cross_fit",
                "match_count": 64,
                "training_match_count": 32,
                "held_out_match_count": 32,
                "training_spatial_cell_count": 8,
                "held_out_spatial_cell_count": 8,
                "training_inlier_count": 29,
                "training_inlier_ratio": 0.90625,
                "training_median_error_px": 0.42,
                "training_p95_error_px": 1.4,
                "held_out_inlier_count": 28,
                "held_out_inlier_ratio": 0.875,
                "held_out_median_error_px": 0.48,
                "held_out_p95_error_px": 1.6,
                "reverse_validation_inlier_count": 29,
                "reverse_validation_inlier_ratio": 0.90625,
                "reverse_validation_median_error_px": 0.46,
                "reverse_validation_p95_error_px": 1.5,
                "cross_fit_max_disagreement_px": 0.72
            },
            "homography_spatial_validation": {
                "accepted": true,
                "reason": "homography improved both spatial score partitions",
                "method": "two_spatial_checkerboard_ncc_splits_against_translation",
                "baseline_ncc": [0.81, 0.82],
                "candidate_ncc": [0.91, 0.92],
                "ncc_improvement": [0.10, 0.10],
                "mean_ncc_improvement": 0.10,
                "registration_error_reduction": [0.526, 0.556],
                "mean_registration_error_reduction": 0.541,
                "sample_count": [1600, 1590],
                "maximum_deviation_from_translation_px": 2.4,
                "transform_right_to_left": [[1.0, 0.0, 380.0], [0.0, 1.0, 0.0], [0.00002, 0.0, 1.0]],
                "model_selection_limit": "candidate features are fit only on whole-cell training regions"
            },
            "seam_exposure_correction": {
                "mode": "auto",
                "model": "gain_only_scalar",
                "applied": true,
                "reason": "applied luma-only overlap gain",
                "sample_count": 8192,
                "valid_sample_ratio": 0.72,
                "gain_rgb": [0.91, 0.91, 0.91],
                "gain_luma": 0.91,
                "spatial_gain_log_slope_x_rgb": [0.0, 0.0, 0.0],
                "spatial_gain_log_slope_x_luma": 0.0,
                "spatial_gain_log_slope_y_rgb": [0.0, 0.0, 0.0],
                "spatial_gain_log_slope_y_luma": 0.0,
                "spatial_gain_log_quadratic_xx_rgb": [0.0, 0.0, 0.0],
                "spatial_gain_log_quadratic_xx_luma": 0.0,
                "spatial_gain_log_quadratic_xy_rgb": [0.0, 0.0, 0.0],
                "spatial_gain_log_quadratic_xy_luma": 0.0,
                "spatial_gain_log_quadratic_yy_rgb": [0.0, 0.0, 0.0],
                "spatial_gain_log_quadratic_yy_luma": 0.0,
                "spatial_gain_top_rgb": [0.91, 0.91, 0.91],
                "spatial_gain_bottom_rgb": [0.91, 0.91, 0.91],
                "spatial_offset_slope_x_rgb": [0.0, 0.0, 0.0],
                "spatial_offset_slope_x_luma": 0.0,
                "spatial_offset_slope_x_rgb_normalized": [0.0, 0.0, 0.0],
                "spatial_offset_slope_y_rgb": [0.0, 0.0, 0.0],
                "spatial_offset_slope_y_luma": 0.0,
                "spatial_offset_slope_y_rgb_normalized": [0.0, 0.0, 0.0],
                "spatial_offset_quadratic_xx_rgb": [0.0, 0.0, 0.0],
                "spatial_offset_quadratic_xx_luma": 0.0,
                "spatial_offset_quadratic_xx_rgb_normalized": [0.0, 0.0, 0.0],
                "spatial_offset_quadratic_xy_rgb": [0.0, 0.0, 0.0],
                "spatial_offset_quadratic_xy_luma": 0.0,
                "spatial_offset_quadratic_xy_rgb_normalized": [0.0, 0.0, 0.0],
                "spatial_offset_quadratic_yy_rgb": [0.0, 0.0, 0.0],
                "spatial_offset_quadratic_yy_luma": 0.0,
                "spatial_offset_quadratic_yy_rgb_normalized": [0.0, 0.0, 0.0],
                "spatial_offset_top_rgb": [0.0, 0.0, 0.0],
                "spatial_offset_bottom_rgb": [0.0, 0.0, 0.0],
                "spatial_offset_top_rgb_normalized": [0.0, 0.0, 0.0],
                "spatial_offset_bottom_rgb_normalized": [0.0, 0.0, 0.0],
                "offset_rgb": [0.0, 0.0, 0.0],
                "offset_luma": 0.0,
                "offset_rgb_normalized": [0.0, 0.0, 0.0],
                "seam_score_before": 0.102,
                "seam_score_after": 0.018,
                "clipped_high_before": [0.0, 0.0, 0.0],
                "clipped_high_after": [0.0, 0.0, 0.0],
                "clipped_low_before": [0.0, 0.0, 0.0],
                "clipped_low_after": [0.0, 0.0, 0.0],
                "training_window_count": 12,
                "held_out_window_count": 12,
                "training_sample_count": 4096,
                "held_out_sample_count": 4096,
                "spatial_training_window_count": 12,
                "spatial_held_out_window_count": 12,
                "spatial_distinct_training_rows": 4,
                "spatial_distinct_held_out_rows": 4,
                "spatial_consistent_window_ratio": 1.0,
                "spatial_slope_agreement_ratio": 0.0,
                "spatial_affine_training_window_count": 12,
                "spatial_affine_held_out_window_count": 12,
                "spatial_affine_distinct_training_rows": 4,
                "spatial_affine_distinct_held_out_rows": 4,
                "spatial_affine_consistent_window_ratio": 1.0,
                "spatial_affine_slope_agreement_ratio": 0.0,
                "spatial_affine_center_offset_delta_normalized": 0.001,
                "held_out_identity_seam_score": 0.101,
                "held_out_gain_seam_score": 0.019,
                "held_out_gain_offset_seam_score": 0.018,
                "held_out_spatial_gain_seam_score": 0.019,
                "held_out_spatial_gain_offset_seam_score": 0.0185,
                "held_out_selected_seam_score": 0.019,
                "held_out_improvement_over_identity": 0.082,
                "held_out_improvement_over_gain": 0.001,
                "held_out_spatial_improvement_over_best_constant": 0.0,
                "held_out_spatial_gain_offset_improvement_over_best_simpler": 0.0005,
                "held_out_validation_passed": true,
                "gain_offset_rejection_reason": "gain+offset did not materially beat gain-only",
                "spatial_rejection_reason": "vertical gain field is unnecessary",
                "spatial_gain_offset_rejection_reason": "vertical gain+offset field is unnecessary",
                "spatial_2d_validation": {
                    "coordinate_system": "normalized_overlap_xy_clamped_outside_support",
                    "training_window_count": 12,
                    "held_out_window_count": 12,
                    "distinct_training_rows": 4,
                    "distinct_held_out_rows": 4,
                    "distinct_training_columns": 3,
                    "distinct_held_out_columns": 3,
                    "gain_consistent_window_ratio": 1.0,
                    "gain_slope_agreement_ratio": 0.0,
                    "gain_horizontal_slope_agreement_ratio": 0.0,
                    "gain_center_log_delta": 0.001,
                    "gain_offset_consistent_window_ratio": 1.0,
                    "gain_offset_slope_agreement_ratio": 0.0,
                    "gain_offset_horizontal_slope_agreement_ratio": 0.0,
                    "gain_offset_center_gain_log_delta": 0.001,
                    "gain_offset_center_offset_delta_normalized": 0.001,
                    "held_out_gain_seam_score": 0.019,
                    "held_out_gain_offset_seam_score": 0.0185,
                    "gain_best_simpler_model": "gain-only",
                    "gain_improvement_over_best_simpler": 0.0,
                    "gain_offset_best_simpler_model": "gain-only",
                    "gain_offset_improvement_over_best_simpler": 0.0005,
                    "gain_accepted": false,
                    "gain_offset_accepted": false,
                    "gain_rejection_reason": "2D gain field is unnecessary",
                    "gain_offset_rejection_reason": "2D gain+offset field is unnecessary",
                    "quadratic": {
                        "basis": ["1", "x", "y", "x_squared", "x_y", "y_squared"],
                        "evaluation_grid_size": 9,
                        "regularization_lambda": 0.001,
                        "minimum_windows_per_split": 18,
                        "minimum_rows_per_split": 6,
                        "minimum_columns_per_split": 5,
                        "training_window_count": 12,
                        "held_out_window_count": 12,
                        "distinct_training_rows": 4,
                        "distinct_held_out_rows": 4,
                        "distinct_training_columns": 3,
                        "distinct_held_out_columns": 3,
                        "gain_design_condition_number": 4.5,
                        "held_out_gain_design_condition_number": 4.6,
                        "gain_consistent_window_ratio": 1.0,
                        "gain_curvature_coefficient_agreement_ratio": 0.0,
                        "gain_max_validation_field_log_delta": 0.001,
                        "gain_curvature_signal": 0.0,
                        "estimated_gain_log_quadratic_xx_rgb": [0.0, 0.0, 0.0],
                        "estimated_gain_log_quadratic_xy_rgb": [0.0, 0.0, 0.0],
                        "estimated_gain_log_quadratic_yy_rgb": [0.0, 0.0, 0.0],
                        "gain_grid_min_rgb": [0.91, 0.91, 0.91],
                        "gain_grid_max_rgb": [0.91, 0.91, 0.91],
                        "gain_offset_design_condition_number": 0.0,
                        "held_out_gain_offset_design_condition_number": 0.0,
                        "gain_offset_consistent_window_ratio": 1.0,
                        "gain_offset_curvature_coefficient_agreement_ratio": 0.0,
                        "gain_offset_max_validation_gain_field_log_delta": 0.001,
                        "gain_offset_max_validation_offset_field_delta_normalized": 0.001,
                        "gain_offset_curvature_signal": 0.0,
                        "estimated_gain_offset_log_quadratic_xx_rgb": [0.0, 0.0, 0.0],
                        "estimated_gain_offset_log_quadratic_xy_rgb": [0.0, 0.0, 0.0],
                        "estimated_gain_offset_log_quadratic_yy_rgb": [0.0, 0.0, 0.0],
                        "estimated_gain_offset_quadratic_xx_rgb": [0.0, 0.0, 0.0],
                        "estimated_gain_offset_quadratic_xy_rgb": [0.0, 0.0, 0.0],
                        "estimated_gain_offset_quadratic_yy_rgb": [0.0, 0.0, 0.0],
                        "gain_offset_grid_gain_min_rgb": [0.91, 0.91, 0.91],
                        "gain_offset_grid_gain_max_rgb": [0.91, 0.91, 0.91],
                        "gain_offset_grid_offset_abs_max_normalized": 0.0,
                        "held_out_gain_seam_score": 0.019,
                        "held_out_gain_offset_seam_score": 0.0185,
                        "gain_best_simpler_model": "gain-only",
                        "gain_improvement_over_best_simpler": 0.0,
                        "gain_offset_best_simpler_model": "gain-only",
                        "gain_offset_improvement_over_best_simpler": 0.0005,
                        "gain_accepted": false,
                        "gain_offset_accepted": false,
                        "gain_rejection_reason": "insufficient quadratic support",
                        "gain_offset_rejection_reason": "insufficient quadratic support"
                    }
                }
            },
            "seam_blend": {
                "mode": "seam_aware_multiband",
                "applied": true,
                "reason": "minimum-cost seam with five frequency bands",
                "review_required": false,
                "review_reason": "seam detail and gradient gates passed",
                "overlap_width_px": 180,
                "overlap_height_px": 1180,
                "transition_width_px": 50,
                "pyramid_levels": 5,
                "seam_path_mean_normalized_cost": 0.014,
                "seam_path_p95_normalized_cost": 0.042,
                "overlap_mean_abs_difference": 0.035,
                "overlap_p95_abs_difference": 0.161,
                "output_seam_gradient_p95": 0.009,
                "source_seam_gradient_p95": 0.016,
                "output_to_source_seam_gradient_ratio": 0.576,
                "detail_consistency": {
                    "method": "multiscale_luma_highpass_energy_disjoint_checkerboard_windows",
                    "evaluated": true,
                    "reason": "no repeated severe imbalance",
                    "supported_scale_count": 3,
                    "decision_supported": true,
                    "imbalanced_scale_count": 0,
                    "maximum_symmetric_energy_ratio": 1.08,
                    "review_ratio_threshold": 2.0,
                    "minimum_direction_consistency": 0.75,
                    "maximum_cross_split_ratio": 1.35,
                    "minimum_repeated_scale_count": 2,
                    "review_required": false,
                    "review_reason": "no repeated severe detail transition"
                }
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
                "color_mapping_application": {
                    "evaluated": true,
                    "applied": false,
                    "selection_status": "rejected_reference_fit",
                    "selected_candidate": "neutral_balance_fallback",
                    "preferred_candidate": "calibrated_direct_profile",
                    "reason": "calibrated profile regressed synthetic reference patches",
                    "definition": "whether a calibration or scanner-prior color mapping candidate was selected for the final colorspace output; scanner linearization, film-base, and negative-response calibration are separate application stages"
                },
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
                "status": "rejected_reference_fit",
                "reason": "calibrated profile regressed synthetic reference patches",
                "color_mode": "auto",
                "preferred_candidate": "calibrated_direct_profile",
                "preferred_candidate_quality_score": 0.31,
                "image_derived_quality_score": 0.88,
                "beats_image_derived": true,
                "within_negative_gamut_limits": true,
                "forced_by_color_mode": false
            },
            "neutral_safety_rescue": {
                "evaluated": true,
                "applied": true,
                "matrix_candidate": "gamut_trusted_image_matrix_blend",
                "matrix_candidate_kind": "image_derived",
                "matrix_anchor_evidence_supported": false,
                "neutral_estimate_supported": true,
                "neutral_model_evidence_supported": true,
                "matrix_pre_scale_preserved_ratio": 0.74,
                "neutral_pre_scale_preserved_ratio": 0.998,
                "preserved_ratio_gain": 0.258,
                "minimum_preserved_ratio": 0.97,
                "minimum_preserved_ratio_gain": 0.10,
                "matrix_midtone_saturation_p95": 0.94,
                "neutral_midtone_saturation_p95": 0.66,
                "midtone_saturation_p95_reduction": 0.28,
                "maximum_midtone_saturation_p95": 0.85,
                "minimum_midtone_saturation_p95_reduction": 0.12,
                "matrix_memory_color_penalty": 0.013,
                "neutral_memory_color_penalty": 0.0,
                "matrix_spatial_consistency_penalty": 0.018,
                "neutral_spatial_consistency_penalty": 0.0,
                "neutral_saturation_preservation_sample_count": 4096,
                "neutral_saturation_preservation_p05_ratio": 0.72,
                "neutral_saturation_preservation_median_ratio": 0.90,
                "neutral_saturation_preservation_p95_ratio": 0.97,
                "reason": "auto neutral safety rescue replaced unsupported image-derived candidate"
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
            "tone_confidence_status": "supported_tone_and_optional_grain_evidence",
            "tone_output_evidence_evaluated": true,
            "tone_output_evidence_confidence": 1.0,
            "tone_output_confidence_status": "supported_render_tonal_distribution",
            "tone_output_review_required": false,
            "tone_output_review_reason": "rendered tonal distribution is supported",
            "confidence_limited_by_tone_output_evidence": false,
            "input_luminance_range_p05_p95": 0.80,
            "mapped_luminance_range_p05_p95": 0.85,
            "render_to_mapped_luminance_range_ratio": 1.0588235294117647,
            "maximum_post_tone_high_clip_ratio": 0.002,
            "maximum_post_tone_low_clip_ratio": 0.001,
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
            "shadow_rgb_median": [0.14, 0.13, 0.12],
            "shadow_visible_pixel_count": 120,
            "shadow_visible_rgb_median": [0.18, 0.17, 0.16],
            "midtone_saturation_median": 0.18,
            "midtone_saturation_p95": 0.42,
            "midtone_rgb_median": [0.49, 0.48, 0.47],
            "render_luminance_percentiles": [0.04, 0.48, 0.94],
            "render_luminance_range_p05_p95": 0.90,
            "midtone_luminance_percentiles": [0.33, 0.48, 0.61],
            "midtone_neutral_pixel_count": 512,
            "midtone_neutral_saturation_p95": 0.05,
            "midtone_neutral_rgb_median": [0.50, 0.49, 0.48],
            "bright_neutral_saturation_median": 0.04,
            "bright_neutral_saturation_p95": 0.08,
            "bright_neutral_rgb_median": [0.88, 0.86, 0.84],
            "bright_saturated_saturation_median": 0.63,
            "bright_saturated_saturation_p95": 0.81,
            "post_chroma_compression_clipped_high_ratio": [0.0, 0.0, 0.002],
            "post_chroma_compression_clipped_low_ratio": [0.0, 0.001, 0.0],
            "perceptual_gamut_mapping_space": "CIELAB_D50_constant_lightness_and_hue",
            "perceptual_gamut_mapped_ratio": 0.012,
            "perceptual_gamut_mean_chroma_scale": 0.83,
            "perceptual_gamut_min_chroma_scale": 0.41,
            "adaptive_vibrance_skin_memory_protection": {
                "enabled": true,
                "method": "feathered_cielab_lightness_chroma_hue_region_limits_only_creative_vibrance",
                "working_space": "CIELAB_D50_from_linear_ProPhoto_RGB_D50",
                "reference": "https://doi.org/10.1002/col.70012",
                "core_lightness": [27.12, 73.71],
                "support_lightness": [18.0, 88.0],
                "core_chroma": [9.01, 30.43],
                "support_chroma": [5.0, 45.0],
                "core_hue_degrees": [24.63, 79.64],
                "support_hue_degrees": [15.0, 90.0],
                "maximum_vibrance_reduction": 0.90,
                "evaluated_pixel_ratio": 0.53,
                "protected_pixel_ratio": 0.18,
                "mean_protection_weight": 0.72,
                "max_protection_weight": 1.0
            },
            "adaptive_vibrance_preferred_memory_color_guard": {
                "enabled": true,
                "method": "published_preference_ellipse_support_then_one_way_ab_path_projection_limits_only_creative_vibrance",
                "working_space": "CIELAB_D50_ab_projection_from_linear_ProPhoto_RGB_D50",
                "reference": "https://doi.org/10.2352/issn.2169-2629.2021.29.170",
                "interpretation": "appearance-relative proxy only; not semantic detection, calibration evidence, or an instruction to pull pixels toward a preferred center",
                "core_normalized_radius": 1.0,
                "support_normalized_radius": 2.0,
                "evaluated_pixel_ratio": 0.53,
                "matched_pixel_ratio": 0.20,
                "limited_pixel_ratio": 0.08,
                "mean_scale_reduction": 0.0175,
                "max_scale_reduction": 0.05,
                "families": [
                    {
                        "family": "sky",
                        "preferred_center_lab": [58.3, 5.930770515379112, -43.295680628602085],
                        "preferred_center_lch": [58.3, 43.7, 277.8],
                        "semi_major_axis_ab": 15.02,
                        "semi_minor_axis_ab": 8.118918918918919,
                        "axis_ratio": 1.85,
                        "ellipse_rotation_degrees": 120.0,
                        "matched_pixel_ratio": 0.10,
                        "limited_pixel_ratio": 0.05,
                        "mean_scale_reduction": 0.02,
                        "max_scale_reduction": 0.04
                    },
                    {
                        "family": "spring_grass",
                        "preferred_center_lab": [37.7, -34.50220214414507, 51.15161822664608],
                        "preferred_center_lch": [37.7, 61.7, 124.0],
                        "semi_major_axis_ab": 32.71,
                        "semi_minor_axis_ab": 11.894545454545455,
                        "axis_ratio": 2.75,
                        "ellipse_rotation_degrees": 125.77,
                        "matched_pixel_ratio": 0.06,
                        "limited_pixel_ratio": 0.02,
                        "mean_scale_reduction": 0.015,
                        "max_scale_reduction": 0.03
                    },
                    {
                        "family": "autumn_grass",
                        "preferred_center_lab": [45.3, 1.3112479885618953, 44.18054581727678],
                        "preferred_center_lch": [45.3, 44.2, 88.3],
                        "semi_major_axis_ab": 16.08,
                        "semi_minor_axis_ab": 12.864,
                        "axis_ratio": 1.25,
                        "ellipse_rotation_degrees": 96.77,
                        "matched_pixel_ratio": 0.04,
                        "limited_pixel_ratio": 0.01,
                        "mean_scale_reduction": 0.01,
                        "max_scale_reduction": 0.05
                    }
                ]
            },
            "preferred_skin_rendering": {
                "enabled": true,
                "reason": "trusted modern-clean render applied a bounded one-way excess-chroma preference-ellipse shoulder while preserving CIELAB lightness",
                "method": "published_skin_preference_ellipse_with_bounded_one_way_excess_chroma_shoulder",
                "working_space": "CIELAB_D50_ab_projection_from_linear_ProPhoto_RGB_D50",
                "preference_reference": "https://doi.org/10.2352/issn.2169-2629.2021.29.170",
                "support_reference": "https://doi.org/10.1002/col.70012",
                "interpretation": "display-only aggregate appearance proxy; not face or skin-group detection, calibration evidence, or a claim that every matched pixel is skin",
                "preferred_center_lab": [61.3, 19.52430924020962, 16.557818355467763],
                "preferred_center_lch": [61.3, 25.6, 40.3],
                "semi_major_axis_ab": 19.41,
                "semi_minor_axis_ab": 6.982014388489209,
                "axis_ratio": 2.78,
                "ellipse_rotation_degrees": 71.57,
                "core_normalized_radius": 1.0,
                "radial_excess_reduction": 0.35,
                "maximum_delta_e_ab": 3.0,
                "minimum_support_weight": 0.05,
                "evaluated_pixel_ratio": 0.70,
                "matched_pixel_ratio": 0.25,
                "outside_preferred_core_ratio": 0.15,
                "adjusted_pixel_ratio": 0.10,
                "gamut_limited_pixel_ratio": 0.01,
                "mean_delta_e_ab": 1.1,
                "max_delta_e_ab": 2.5,
                "mean_abs_hue_shift_degrees": 1.5,
                "max_abs_hue_shift_degrees": 4.0,
                "mean_chroma_delta": -0.5,
                "max_abs_chroma_delta": 2.0
            },
            "noise_reduction_requested_enabled": true,
            "noise_reduction_requested_strength": 0.7,
            "noise_reduction_requested_scale": 1.5,
            "noise_reduction_enabled": true,
            "noise_reduction_reason": "independently requested edge-aware film-grain reduction",
            "noise_reduction_radius": 3,
            "noise_reduction_chroma_amount": 0.672,
            "noise_reduction_luma_amount": 0.21,
            "noise_reduction_applied_ratio": 0.81,
            "noise_reduction_structure_gate_start": 0.018,
            "noise_reduction_structure_gate_end": 0.035,
            "noise_reduction_structure_excluded_ratio": 0.12,
            "noise_reduction_texture_limited_ratio": 0.12,
            "noise_reduction_saturation_limited_ratio": 0.04,
            "noise_reduction_mean_abs_chroma_delta": 0.003,
            "noise_reduction_max_abs_chroma_delta": 0.02,
            "noise_reduction_mean_abs_luma_delta": 0.001,
            "noise_reduction_max_abs_luma_delta": 0.008,
            "noise_reduction_flat_luma_p95_reduction_ratio": 0.18,
            "noise_reduction_flat_chroma_p95_reduction_ratio": 0.42,
            "grain_reduction": {
                "detail_retention": {
                    "method": "coherent_multiscale_opponent_edge_retention_v1",
                    "evaluated": true,
                    "decision_supported": true,
                    "sample_stride": 2,
                    "probe_radius": 3,
                    "minimum_probe_count": 64,
                    "luminance_probe_count": 512,
                    "chroma_probe_count": 384,
                    "luminance_decision_supported": true,
                    "chroma_decision_supported": true,
                    "luminance_median_retention": 0.98,
                    "luminance_p10_retention": 0.94,
                    "chroma_median_retention": 0.97,
                    "chroma_p10_retention": 0.92,
                    "luminance_contrast_threshold": 0.035,
                    "chroma_contrast_threshold": 0.05,
                    "coherence_threshold": 0.75,
                    "median_retention_threshold": 0.90,
                    "p10_retention_threshold": 0.70,
                    "review_required": false,
                    "reason": "coherent pre/post detail probes passed",
                    "review_reason": null
                }
            },
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
            "tone_output_review_required": false,
            "tone_output_review_reason": "rendered tonal distribution is supported",
            "tone_output_confidence_status": "supported_render_tonal_distribution",
            "tone_output_evidence_confidence": 1.0,
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

#[test]
fn test_final_delivery_reviewability_requires_intact_saved_artifact_evidence() {
    use scanstitch::validation::{
        delivery_artifact_integrity_issues, final_delivery_evidence_is_reviewable,
        RenderDiagnosticSummary,
    };

    let digest = "a".repeat(64);
    let render = RenderDiagnosticSummary {
        artifact_sha256_binding_required: true,
        output_path: Some("output/output.tiff".to_string()),
        output_width: Some(640),
        output_height: Some(480),
        output_sha256: Some(digest.clone()),
        output_file_sha256: Some(digest),
        output_file_sha256_matches_report: Some(true),
        output_file_dimensions_match_report: Some(true),
        output_file_storage_matches_report: Some(true),
        output_file_icc_profile_matches_report: Some(true),
        master_scene_referred_requested: Some(false),
        review_srgb_requested: Some(false),
        stale_render_artifact_count: Some(0),
        render_review_status: Some("reviewable".to_string()),
        render_reviewable: Some(true),
        ..RenderDiagnosticSummary::default()
    };
    assert!(delivery_artifact_integrity_issues(&render).is_empty());
    assert!(final_delivery_evidence_is_reviewable(&render));

    let mut legacy_structural_only = render.clone();
    legacy_structural_only.artifact_sha256_binding_required = false;
    legacy_structural_only.output_sha256 = None;
    legacy_structural_only.output_file_sha256 = None;
    legacy_structural_only.output_file_sha256_matches_report = None;
    assert!(delivery_artifact_integrity_issues(&legacy_structural_only).is_empty());
    assert!(final_delivery_evidence_is_reviewable(
        &legacy_structural_only
    ));

    let mut missing = render.clone();
    missing.output_path = None;
    missing.output_width = Some(0);
    missing.output_height = None;
    missing.output_file_dimensions_match_report = Some(false);
    missing.output_file_storage_matches_report = None;
    missing.output_file_icc_profile_matches_report = Some(false);
    missing.output_sha256 = None;
    missing.output_file_sha256 = None;
    missing.output_file_sha256_matches_report = None;
    missing.stale_render_artifact_count = None;
    let issues = delivery_artifact_integrity_issues(&missing);
    assert_eq!(
        issues,
        [
            "output_path_missing",
            "output_dimensions_missing_or_invalid",
            "output_dimensions_mismatch",
            "output_storage_evidence_missing",
            "output_icc_profile_mismatch",
            "output_sha256_declaration_missing_or_invalid",
            "stale_render_artifact_evidence_missing",
        ]
    );
    assert!(!final_delivery_evidence_is_reviewable(&missing));

    let mut missing_auxiliary = render.clone();
    missing_auxiliary.master_scene_referred_requested = Some(true);
    missing_auxiliary.review_srgb_requested = Some(true);
    assert_eq!(
        delivery_artifact_integrity_issues(&missing_auxiliary),
        [
            "master_scene_referred_path_missing",
            "master_scene_referred_artifact_evidence_missing",
            "review_srgb_path_missing",
            "review_srgb_artifact_evidence_missing",
            "master_scene_referred_sha256_declaration_missing_or_invalid",
            "review_srgb_sha256_declaration_missing_or_invalid",
            "review_srgb_gamut_mapping_evidence_missing",
        ]
    );

    let mut stale = render;
    stale.stale_render_artifact_count = Some(2);
    assert_eq!(
        delivery_artifact_integrity_issues(&stale),
        ["stale_render_artifacts"]
    );
}

#[test]
fn test_validation_summary_preserves_n_component_inferred_order() {
    let mut report = fixture_report();
    let stitch = report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "stitch")
        .expect("stitch phase");
    let mut corrected_pair = stitch.metrics.clone();
    corrected_pair["seam_exposure_correction"]["offset_rgb_normalized"] =
        serde_json::json!([0.02, -0.01, 0.005]);
    let mut identity_pair = stitch.metrics.clone();
    identity_pair["seam_exposure_correction"]["model"] = serde_json::json!("identity");
    identity_pair["seam_exposure_correction"]["applied"] = serde_json::json!(false);
    identity_pair["seam_exposure_correction"]["held_out_validation_passed"] =
        serde_json::json!(false);
    identity_pair["seam_exposure_correction"]["offset_rgb_normalized"] =
        serde_json::json!([0.0, 0.0, 0.0]);
    stitch.metrics = serde_json::json!({
        "decision": "accepted_sequence",
        "ordering": {
            "inferred_order": [2, 3, 1],
            "all_adjacencies_validated": true
        },
        "pair_merges": [
            {"pair_report": {"metrics": corrected_pair}},
            {"pair_report": {"metrics": identity_pair}}
        ]
    });

    let summary = scanstitch::validation::summarize_report("three-way", &report);
    assert_eq!(summary.stitch.inferred_order, vec![2, 3, 1]);
    let exposure = summary.stitch.seam_exposure_correction.unwrap();
    assert_eq!(exposure.model.as_deref(), Some("sequence_mixed"));
    assert_eq!(exposure.applied, Some(true));
    assert_eq!(exposure.held_out_validation_passed, Some(true));
    assert_eq!(
        exposure.offset_rgb_normalized,
        Some(vec![0.02, 0.01, 0.005])
    );
    let blend = summary.stitch.seam_blend.unwrap();
    assert_eq!(blend.merge_count, 2);
    assert_eq!(blend.review_required, Some(false));
    assert_eq!(
        blend
            .detail_consistency
            .unwrap()
            .minimum_supported_scale_count,
        Some(3)
    );
}

#[test]
fn test_validation_summary_extracts_absolute_deskew_evidence() {
    let mut report = fixture_report();
    report.phases.push(PhaseReport::ok(
        "load",
        1.0,
        serde_json::json!({
            "input_count": 1,
            "components": [{
                "index": 1,
                "decode": {
                    "decoded_pixel_sha256": "a".repeat(64),
                    "source_orientation_materialized_decoded_pixel_sha256": "b".repeat(64),
                    "orientation": {
                        "tag_value": 6,
                        "transform": "rotate_90_clockwise",
                        "applied": true,
                        "source_width": 220,
                        "source_height": 160,
                        "output_width": 160,
                        "output_height": 220
                    },
                    "orientation_correction": {
                        "requested": "rotate-180",
                        "transform": "rotate_180",
                        "applied": true,
                        "input_width": 160,
                        "input_height": 220,
                        "output_width": 160,
                        "output_height": 220,
                        "effective_tag_value": 8,
                        "effective_transform": "rotate_270_clockwise",
                        "source_orientation_materialized_decoded_pixel_sha256": "b".repeat(64),
                        "corrected_decoded_pixel_sha256": "a".repeat(64),
                        "reason": "explicit correction"
                    }
                }
            }]
        }),
    ));
    report.phases.push(PhaseReport::ok(
        "deskew",
        0.91,
        serde_json::json!({
            "requested_mode": "auto",
            "status": "applied",
            "applied": true,
            "applied_component_count": 1,
            "input_count": 1,
            "detected_source_skew_degrees": 0.72,
            "correction_degrees": -0.72,
            "review_required": false,
            "review_reason": "accepted",
            "retained_area_ratio": 0.96,
            "proposed_retained_area_ratio": 0.96,
            "supporting_side_count": 4,
            "horizontal_side_count": 2,
            "vertical_side_count": 2,
            "side_angle_spread_degrees": 0.04,
            "interpolation": "bicubic_catmull_rom_single_resample",
            "reason": "four sides agree",
            "components": [{
                "index": 1,
                "diagnostics": {
                    "applied": true,
                    "retained_area_ratio": 0.96
                }
            }]
        }),
    ));

    let summary = scanstitch::validation::summarize_report("deskewed", &report);
    assert_eq!(summary.input_orientation.input_count, Some(1));
    assert_eq!(summary.input_orientation.component_count, 1);
    assert_eq!(
        summary.input_orientation.all_components_reported,
        Some(true)
    );
    assert_eq!(summary.input_orientation.components[0].tag_value, Some(6));
    assert_eq!(
        summary.input_orientation.components[0]
            .decoded_pixel_sha256
            .as_deref(),
        Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    );
    assert_eq!(
        summary.input_orientation.components[0].transform.as_deref(),
        Some("rotate_270_clockwise")
    );
    assert_eq!(
        summary.input_orientation.components[0]
            .source_orientation_materialized_decoded_pixel_sha256
            .as_deref(),
        Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    );
    assert_eq!(
        summary.input_orientation.components[0]
            .metadata_transform
            .as_deref(),
        Some("rotate_90_clockwise")
    );
    assert_eq!(
        summary.input_orientation.components[0]
            .orientation_correction_requested
            .as_deref(),
        Some("rotate-180")
    );
    assert_eq!(
        summary.input_orientation.components[0]
            .orientation_correction_transform
            .as_deref(),
        Some("rotate_180")
    );
    assert_eq!(
        summary.input_orientation.components[0].effective_tag_value,
        Some(8)
    );
    assert_eq!(
        summary.input_orientation.components[0].output_width,
        Some(160)
    );
    assert_eq!(summary.deskew.status.as_deref(), Some("applied"));
    assert_eq!(summary.deskew.applied, Some(true));
    assert_eq!(summary.deskew.correction_degrees, Some(-0.72));
    assert_eq!(summary.deskew.supporting_side_count, Some(4));
    assert_eq!(summary.deskew.retained_area_ratio, Some(0.96));
    assert_eq!(summary.deskew.proposed_retained_area_ratio, Some(0.96));
    assert_eq!(summary.deskew.all_components_reported, Some(true));
    assert_eq!(summary.deskew.all_components_applied, Some(true));
    assert_eq!(
        summary.deskew.minimum_component_retained_area_ratio,
        Some(0.96)
    );
    assert_eq!(
        summary.deskew.interpolation.as_deref(),
        Some("bicubic_catmull_rom_single_resample")
    );
    let markdown = scanstitch::validation::summary_to_markdown(&summary);
    assert!(markdown.contains(
        "component 1: tag=6 effective_transform=rotate_270_clockwise correction=rotate-180 applied=true"
    ));
    assert!(markdown.contains("detected_source_skew_degrees"));
    assert!(markdown.contains("retained_area_ratio"));
}

#[test]
fn test_validation_summary_extracts_border_crop_and_measured_negative_reconstruction() {
    let mut report = PipelineReport::new();
    report.add_phase(PhaseReport::ok(
        "border_removal",
        1.0,
        serde_json::json!({
            "input_count": 2,
            "components": [
                {
                    "index": 1,
                    "crop": {
                        "top_removed": 8,
                        "bottom_removed": 9,
                        "left_removed": 10,
                        "right_removed": 11,
                        "dead_zone_detected": true
                    },
                    "output_shape": [103, 179]
                },
                {
                    "index": 2,
                    "crop": {
                        "top_removed": 7,
                        "bottom_removed": 8,
                        "left_removed": 9,
                        "right_removed": 10,
                        "dead_zone_detected": true
                    },
                    "output_shape": [105, 181]
                }
            ]
        }),
    ));
    report.add_phase(PhaseReport::ok(
        "density_inversion",
        0.84,
        serde_json::json!({
            "input_mode": "negative",
            "skipped": false,
            "direct_density_response_model": {
                "model": "measured_nonlinear_dye_separation",
                "source": "measured_roll_target_with_held_out_validation",
                "accepted": true,
                "review_required": false,
                "crosstalk_model": "measured_3x3_scanner_density_to_film_layers",
                "characteristic_curve_model": "measured_monotone_pchip_density_to_scene_log_exposure",
                "measured_model_id": "synthetic-response-v1",
                "measured_confidence": 0.91,
                "held_out_delta_e00_rms": 2.4,
                "held_out_delta_e00_max": 5.8,
                "unit_slope_delta_e00_rms": 4.1,
                "maximum_density_noise_gain": 1.8
            }
        }),
    ));
    report.add_phase(PhaseReport::ok(
        "colorspace_mapping",
        0.93,
        serde_json::json!({
            "negative_response_review_required": false,
            "negative_response_reconstruction": {
                "signed_headroom_preserved": true,
                "curve_extrapolated_any_ratio": 0.004,
                "curve_interpolation": "monotone_piecewise_cubic_hermite_with_endpoint_tangent_extrapolation"
            }
        }),
    ));

    let summary = scanstitch::validation::summarize_report("negative-border", &report);
    assert_eq!(summary.border_crop.all_components_reported, Some(true));
    assert_eq!(summary.border_crop.all_components_cropped, Some(true));
    assert_eq!(summary.border_crop.total_removed_edge_count, 8);
    assert_eq!(
        summary.border_crop.minimum_removed_edge_count_per_component,
        Some(4)
    );
    assert_eq!(summary.border_crop.components[0].input_width, Some(200));
    assert_eq!(summary.border_crop.components[0].input_height, Some(120));
    assert_eq!(
        summary.negative_reconstruction.response_model.as_deref(),
        Some("measured_nonlinear_dye_separation")
    );
    assert_eq!(
        summary
            .negative_reconstruction
            .held_out_improvement_over_unit_slope,
        Some(1.6999999999999997)
    );
    assert_eq!(
        summary
            .negative_reconstruction
            .reconstruction_review_required,
        Some(false)
    );
    assert_eq!(
        summary.negative_reconstruction.signed_headroom_preserved,
        Some(true)
    );
    let markdown = scanstitch::validation::summary_to_markdown(&summary);
    assert!(markdown.contains("all_components_cropped"));
    assert!(markdown.contains("held_out_improvement_over_unit_slope"));
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
    assert_eq!(
        summary.report.binary_identity_status.as_deref(),
        Some("verified_sha256")
    );
    assert_eq!(summary.report.binary_sha256, Some("a".repeat(64)));
    assert_eq!(summary.report.binary_file_size_bytes, Some(1));
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
    assert_eq!(summary.render.tone_output_review_required, Some(false));
    assert_eq!(
        summary.render.tone_output_confidence_status.as_deref(),
        Some("supported_render_tonal_distribution")
    );
    assert_eq!(summary.render.tone_output_evidence_confidence, Some(1.0));
    assert_eq!(
        summary
            .render
            .tone_output_render_to_mapped_luminance_range_ratio,
        Some(1.0588235294117647)
    );
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
    let homography_features = summary
        .stitch
        .homography_feature_validation
        .as_ref()
        .expect("homography feature validation summary");
    assert_eq!(homography_features.accepted, Some(true));
    assert_eq!(homography_features.validation_count, 1);
    assert_eq!(homography_features.accepted_count, 1);
    assert_eq!(homography_features.minimum_training_match_count, Some(32));
    assert_eq!(homography_features.minimum_held_out_match_count, Some(32));
    assert_eq!(
        homography_features.minimum_held_out_inlier_ratio,
        Some(0.875)
    );
    assert_eq!(
        homography_features.maximum_cross_fit_disagreement_px,
        Some(0.72)
    );
    let homography = summary
        .stitch
        .homography_spatial_validation
        .as_ref()
        .expect("homography spatial validation summary");
    assert_eq!(homography.accepted, Some(true));
    assert_eq!(homography.validation_count, 1);
    assert_eq!(homography.accepted_count, 1);
    assert_eq!(homography.minimum_split_ncc_improvement, Some(0.10));
    assert_eq!(homography.minimum_mean_ncc_improvement, Some(0.10));
    assert_eq!(
        homography.minimum_split_registration_error_reduction,
        Some(0.526)
    );
    assert_eq!(
        homography.minimum_mean_registration_error_reduction,
        Some(0.541)
    );
    assert_eq!(homography.minimum_split_sample_count, Some(1590));
    assert_eq!(
        homography.minimum_model_deviation_from_translation_px,
        Some(2.4)
    );
    let seam = summary
        .stitch
        .seam_exposure_correction
        .as_ref()
        .expect("seam exposure summary");
    assert_eq!(seam.applied, Some(true));
    assert_eq!(seam.model.as_deref(), Some("gain_only_scalar"));
    assert_eq!(seam.gain_rgb, Some(vec![0.91, 0.91, 0.91]));
    assert_eq!(seam.spatial_gain_log_slope_x_rgb, Some(vec![0.0, 0.0, 0.0]));
    assert_eq!(seam.spatial_gain_log_slope_y_rgb, Some(vec![0.0, 0.0, 0.0]));
    assert_eq!(
        seam.spatial_gain_log_quadratic_xx_rgb,
        Some(vec![0.0, 0.0, 0.0])
    );
    assert_eq!(seam.spatial_gain_top_rgb, Some(vec![0.91, 0.91, 0.91]));
    assert_eq!(
        seam.spatial_offset_slope_y_rgb_normalized,
        Some(vec![0.0, 0.0, 0.0])
    );
    assert_eq!(seam.offset_rgb, Some(vec![0.0, 0.0, 0.0]));
    assert_eq!(seam.seam_score_before, Some(0.102));
    assert_eq!(seam.seam_score_after, Some(0.018));
    assert_eq!(seam.held_out_validation_passed, Some(true));
    assert_eq!(seam.held_out_selected_seam_score, Some(0.019));
    assert_eq!(seam.held_out_spatial_gain_seam_score, Some(0.019));
    assert_eq!(seam.spatial_distinct_training_rows, Some(4));
    assert_eq!(seam.held_out_spatial_gain_offset_seam_score, Some(0.0185));
    assert_eq!(seam.spatial_affine_distinct_training_rows, Some(4));
    let spatial_2d = seam
        .spatial_2d_validation
        .as_ref()
        .expect("2D seam validation summary");
    assert_eq!(spatial_2d.distinct_training_columns, Some(3));
    assert_eq!(spatial_2d.gain_accepted, Some(false));
    assert_eq!(spatial_2d.gain_offset_accepted, Some(false));
    let quadratic = spatial_2d
        .quadratic
        .as_ref()
        .expect("quadratic seam validation summary");
    assert_eq!(quadratic.evaluation_grid_size, Some(9));
    assert_eq!(quadratic.minimum_windows_per_split, Some(18));
    assert_eq!(quadratic.gain_design_condition_number, Some(4.5));
    assert_eq!(quadratic.gain_accepted, Some(false));
    let seam_blend = summary
        .stitch
        .seam_blend
        .as_ref()
        .expect("seam blend summary");
    assert_eq!(seam_blend.mode.as_deref(), Some("seam_aware_multiband"));
    assert_eq!(seam_blend.applied, Some(true));
    assert_eq!(seam_blend.merge_count, 1);
    assert_eq!(seam_blend.applied_merge_count, 1);
    assert_eq!(seam_blend.review_required, Some(false));
    assert_eq!(seam_blend.review_required_merge_count, 0);
    assert_eq!(seam_blend.overlap_p95_abs_difference, Some(0.161));
    assert_eq!(seam_blend.output_to_source_seam_gradient_ratio, Some(0.576));
    let detail = seam_blend
        .detail_consistency
        .as_ref()
        .expect("seam detail summary");
    assert_eq!(detail.evaluated, Some(true));
    assert_eq!(detail.decision_supported, Some(true));
    assert_eq!(detail.review_required, Some(false));
    assert_eq!(detail.minimum_supported_scale_count, Some(3));
    assert_eq!(detail.maximum_imbalanced_scale_count, Some(0));
    assert_eq!(detail.maximum_symmetric_energy_ratio, Some(1.08));
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
        Some("rejected_reference_fit")
    );
    let calibration_color_mapping = summary
        .colorspace
        .calibration_color_mapping_application
        .as_ref()
        .expect("calibration color-mapping application summary");
    assert_eq!(calibration_color_mapping.evaluated, Some(true));
    assert_eq!(calibration_color_mapping.applied, Some(false));
    assert_eq!(
        calibration_color_mapping.selection_status.as_deref(),
        Some("rejected_reference_fit")
    );
    assert_eq!(
        calibration_color_mapping.selected_candidate.as_deref(),
        Some("neutral_balance_fallback")
    );
    assert!(summary.diagnostic_consistency_issues.is_empty());
    let neutral_rescue = summary
        .colorspace
        .neutral_safety_rescue
        .as_ref()
        .expect("neutral safety rescue summary");
    assert_eq!(neutral_rescue.evaluated, Some(true));
    assert_eq!(neutral_rescue.applied, Some(true));
    assert_eq!(
        neutral_rescue.matrix_candidate.as_deref(),
        Some("gamut_trusted_image_matrix_blend")
    );
    assert_eq!(neutral_rescue.preserved_ratio_gain, Some(0.258));
    assert_eq!(neutral_rescue.midtone_saturation_p95_reduction, Some(0.28));
    assert_eq!(
        neutral_rescue.reason.as_deref(),
        Some("auto neutral safety rescue replaced unsupported image-derived candidate")
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
    assert_eq!(
        summary.tone.tone_confidence_status.as_deref(),
        Some("supported_tone_and_optional_grain_evidence")
    );
    assert_eq!(summary.tone.tone_output_evidence_evaluated, Some(true));
    assert_eq!(summary.tone.tone_output_evidence_confidence, Some(1.0));
    assert_eq!(
        summary.tone.tone_output_confidence_status.as_deref(),
        Some("supported_render_tonal_distribution")
    );
    assert_eq!(summary.tone.tone_output_review_required, Some(false));
    assert_eq!(summary.tone.input_luminance_range_p05_p95, Some(0.80));
    assert_eq!(summary.tone.mapped_luminance_range_p05_p95, Some(0.85));
    assert_eq!(
        summary.tone.render_to_mapped_luminance_range_ratio,
        Some(1.0588235294117647)
    );
    assert_eq!(summary.tone.maximum_post_tone_high_clip_ratio, Some(0.002));
    assert_eq!(summary.tone.maximum_post_tone_low_clip_ratio, Some(0.001));
    assert_eq!(summary.tone.bright_neutral_saturation_p95, Some(0.08));
    assert_eq!(summary.tone.shadow_rgb_median, Some(vec![0.14, 0.13, 0.12]));
    assert_eq!(
        summary.tone.midtone_rgb_median,
        Some(vec![0.49, 0.48, 0.47])
    );
    assert_eq!(
        summary.tone.midtone_neutral_rgb_median,
        Some(vec![0.50, 0.49, 0.48])
    );
    assert_eq!(
        summary.tone.bright_neutral_rgb_median,
        Some(vec![0.88, 0.86, 0.84])
    );
    assert_eq!(
        summary.tone.color_protection_policy.as_deref(),
        Some("enabled")
    );
    assert_eq!(summary.tone.color_trust_state.as_deref(), Some("trusted"));
    assert_eq!(summary.tone.noise_reduction_requested_enabled, Some(true));
    assert_eq!(summary.tone.noise_reduction_requested_strength, Some(0.7));
    assert_eq!(summary.tone.noise_reduction_requested_scale, Some(1.5));
    assert_eq!(summary.tone.noise_reduction_enabled, Some(true));
    assert_eq!(summary.tone.noise_reduction_radius, Some(3));
    assert_eq!(
        summary.tone.noise_reduction_structure_gate_start,
        Some(0.018)
    );
    assert_eq!(summary.tone.noise_reduction_structure_gate_end, Some(0.035));
    assert_eq!(
        summary.tone.noise_reduction_structure_excluded_ratio,
        Some(0.12)
    );
    assert_eq!(
        summary.tone.noise_reduction_flat_chroma_p95_reduction_ratio,
        Some(0.42)
    );
    let grain_detail = summary
        .tone
        .grain_detail_retention
        .as_ref()
        .expect("grain detail retention summary");
    assert_eq!(grain_detail.decision_supported, Some(true));
    assert_eq!(grain_detail.review_required, Some(false));
    assert_eq!(grain_detail.luminance_probe_count, Some(512));
    assert_eq!(grain_detail.chroma_probe_count, Some(384));
    assert_eq!(grain_detail.luminance_p10_retention, Some(0.94));
    assert_eq!(grain_detail.chroma_p10_retention, Some(0.92));
    assert_eq!(summary.render.noise_reduction_requested_strength, Some(0.7));
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
    assert_eq!(
        summary.tone.perceptual_gamut_mapping_space.as_deref(),
        Some("CIELAB_D50_constant_lightness_and_hue")
    );
    assert_eq!(summary.tone.perceptual_gamut_mapped_ratio, Some(0.012));
    assert_eq!(summary.tone.perceptual_gamut_mean_chroma_scale, Some(0.83));
    assert_eq!(summary.tone.perceptual_gamut_min_chroma_scale, Some(0.41));
    let skin_memory = summary
        .tone
        .adaptive_vibrance_skin_memory_protection
        .as_ref()
        .expect("skin-memory protection summary");
    assert_eq!(skin_memory.enabled, Some(true));
    assert_eq!(
        skin_memory.working_space.as_deref(),
        Some("CIELAB_D50_from_linear_ProPhoto_RGB_D50")
    );
    assert_eq!(skin_memory.core_lightness, Some(vec![27.12, 73.71]));
    assert_eq!(skin_memory.protected_pixel_ratio, Some(0.18));
    assert_eq!(skin_memory.mean_protection_weight, Some(0.72));
    assert_eq!(skin_memory.max_protection_weight, Some(1.0));
    let preferred_memory = summary
        .tone
        .adaptive_vibrance_preferred_memory_color_guard
        .as_ref()
        .expect("preferred-memory-colour guard summary");
    assert_eq!(preferred_memory.enabled, Some(true));
    assert_eq!(
        preferred_memory.working_space.as_deref(),
        Some("CIELAB_D50_ab_projection_from_linear_ProPhoto_RGB_D50")
    );
    assert_eq!(preferred_memory.matched_pixel_ratio, Some(0.20));
    assert_eq!(preferred_memory.limited_pixel_ratio, Some(0.08));
    assert_eq!(preferred_memory.mean_scale_reduction, Some(0.0175));
    assert_eq!(preferred_memory.families.len(), 3);
    assert_eq!(preferred_memory.families[0].family.as_deref(), Some("sky"));
    let preferred_skin = summary
        .tone
        .preferred_skin_rendering
        .as_ref()
        .expect("preferred-skin rendering summary");
    assert_eq!(preferred_skin.enabled, Some(true));
    assert_eq!(
        preferred_skin.preference_reference.as_deref(),
        Some("https://doi.org/10.2352/issn.2169-2629.2021.29.170")
    );
    assert_eq!(preferred_skin.matched_pixel_ratio, Some(0.25));
    assert_eq!(preferred_skin.adjusted_pixel_ratio, Some(0.10));
    assert_eq!(preferred_skin.mean_delta_e_ab, Some(1.1));
    assert!(summary.diagnostic_consistency_issues.is_empty());
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
    assert_eq!(
        baseline.stitch.seam_exposure_model.as_deref(),
        Some("gain_only_scalar")
    );
    assert_eq!(
        baseline.stitch.seam_exposure_spatial_2d_gain_accepted,
        Some(false)
    );
    assert_eq!(
        baseline
            .stitch
            .seam_exposure_spatial_2d_gain_offset_accepted,
        Some(false)
    );
    assert_eq!(
        baseline
            .stitch
            .seam_exposure_spatial_quadratic_gain_accepted,
        Some(false)
    );
    assert_eq!(
        baseline
            .stitch
            .seam_exposure_spatial_quadratic_gain_offset_accepted,
        Some(false)
    );
    assert_eq!(
        baseline.stitch.seam_blend_mode.as_deref(),
        Some("seam_aware_multiband")
    );
    assert_eq!(baseline.stitch.seam_blend_applied, Some(true));
    assert_eq!(baseline.stitch.seam_blend_review_required, Some(false));
    assert_eq!(baseline.stitch.seam_detail_review_required, Some(false));
    assert_eq!(
        baseline.stitch.seam_detail_max_symmetric_energy_ratio,
        Some(1.08)
    );
    assert_eq!(baseline.stitch.seam_gradient_ratio, Some(0.576));
    assert_eq!(baseline.stitch.seam_overlap_p95_abs_difference, Some(0.161));
    assert_eq!(baseline.render.output_width, Some(1800));
    assert_eq!(baseline.render.output_height, Some(1200));
    assert_eq!(baseline.render.raw_base_proxy_confidence, Some(0.12));
    assert_eq!(baseline.render.raw_base_support_fraction, Some(0.014));
    assert_eq!(
        baseline.render.colorspace_mapping_strategy.as_deref(),
        Some("neutral_balance_weak_anchor_fallback")
    );
    assert_eq!(
        baseline.render.render_review_status.as_deref(),
        Some("reviewable")
    );
    assert_eq!(baseline.render.render_reviewable, Some(true));
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
    assert_eq!(
        baseline.tone.tone_output_confidence_status.as_deref(),
        Some("supported_render_tonal_distribution")
    );
    assert_eq!(baseline.tone.tone_output_review_required, Some(false));
    assert_eq!(baseline.tone.tone_output_evidence_confidence, Some(1.0));
    assert_eq!(baseline.tone.render_luminance_range_p05_p95, Some(0.90));
    assert_eq!(
        baseline.tone.render_to_mapped_luminance_range_ratio,
        Some(1.0588235294117647)
    );
    assert_eq!(
        baseline
            .tone
            .adaptive_vibrance_skin_memory_protection_enabled,
        Some(true)
    );
    assert_eq!(
        baseline
            .tone
            .adaptive_vibrance_skin_memory_protection_space
            .as_deref(),
        Some("CIELAB_D50_from_linear_ProPhoto_RGB_D50")
    );
    assert_eq!(
        baseline.tone.adaptive_vibrance_skin_memory_protected_ratio,
        Some(0.18)
    );
    assert_eq!(
        baseline.tone.adaptive_vibrance_skin_memory_mean_protection,
        Some(0.72)
    );
    assert_eq!(
        baseline
            .tone
            .adaptive_vibrance_preferred_memory_color_guard_enabled,
        Some(true)
    );
    assert_eq!(
        baseline
            .tone
            .adaptive_vibrance_preferred_memory_color_guard_space
            .as_deref(),
        Some("CIELAB_D50_ab_projection_from_linear_ProPhoto_RGB_D50")
    );
    assert_eq!(
        baseline
            .tone
            .adaptive_vibrance_preferred_memory_color_guard_reference
            .as_deref(),
        Some("https://doi.org/10.2352/issn.2169-2629.2021.29.170")
    );
    assert_eq!(
        baseline
            .tone
            .adaptive_vibrance_preferred_memory_color_matched_ratio,
        Some(0.20)
    );
    assert_eq!(
        baseline
            .tone
            .adaptive_vibrance_preferred_memory_color_limited_ratio,
        Some(0.08)
    );
    assert_eq!(
        baseline
            .tone
            .adaptive_vibrance_preferred_memory_color_mean_scale_reduction,
        Some(0.0175)
    );
    assert_eq!(baseline.tone.preferred_skin_rendering_enabled, Some(true));
    assert_eq!(
        baseline.tone.preferred_skin_rendering_space.as_deref(),
        Some("CIELAB_D50_ab_projection_from_linear_ProPhoto_RGB_D50")
    );
    assert_eq!(
        baseline
            .tone
            .preferred_skin_rendering_preference_reference
            .as_deref(),
        Some("https://doi.org/10.2352/issn.2169-2629.2021.29.170")
    );
    assert_eq!(
        baseline.tone.preferred_skin_rendering_adjusted_ratio,
        Some(0.10)
    );
    assert_eq!(
        baseline.tone.preferred_skin_rendering_mean_delta_e_ab,
        Some(1.1)
    );
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
    assert!(markdown.contains("seam_exposure_spatial_gain_log_slope_y_rgb"));
    assert!(markdown.contains("seam_exposure_spatial_gain_log_slope_x_rgb"));
    assert!(markdown.contains("seam_exposure_held_out_spatial_gain_score"));
    assert!(markdown.contains("seam_exposure_spatial_offset_slope_y_rgb"));
    assert!(markdown.contains("seam_exposure_held_out_spatial_gain_offset_score"));
    assert!(markdown.contains("seam_exposure_spatial_2d_gain_offset_accepted"));
    assert!(markdown.contains("seam_exposure_spatial_quadratic_gain_accepted"));
    assert!(markdown.contains("seam_exposure_spatial_quadratic_gain_offset_accepted"));
    assert!(markdown.contains("seam_blend_review_required"));
    assert!(markdown.contains("seam_detail_review_required"));
    assert!(markdown.contains("seam_detail_max_symmetric_energy_ratio"));
    assert!(markdown.contains("seam_exposure_offset_rgb"));
    assert!(markdown.contains("seam_exposure_held_out_validation_passed"));
    assert!(markdown.contains("homography_feature_validation_accepted"));
    assert!(markdown.contains("homography_minimum_held_out_feature_inlier_ratio"));
    assert!(markdown.contains("homography_spatial_validation_accepted"));
    assert!(markdown.contains("homography_minimum_split_ncc_improvement"));
    assert!(markdown.contains("calibration_record_status"));
    assert!(markdown.contains("calibration_color_mapping_evaluated"));
    assert!(markdown.contains("calibration_color_mapping_applied"));
    assert!(markdown.contains("calibration_color_mapping_status"));
    assert!(markdown.contains("calibration_color_mapping_selected_candidate"));
    assert!(markdown.contains("calibration_color_mapping_preferred_candidate"));
    assert!(markdown.contains("calibration_color_mapping_reason"));
    assert!(markdown.contains("calibration_color_mapping_consistency_issues"));
    assert!(markdown.contains("adaptive_vibrance_preferred_memory_color_guard_enabled"));
    assert!(markdown.contains("adaptive_vibrance_preferred_memory_color_matched_ratio"));
    assert!(markdown.contains("adaptive_vibrance_preferred_memory_color_limited_ratio"));
    assert!(markdown.contains("preferred_skin_rendering_enabled"));
    assert!(markdown.contains("preferred_skin_rendering_adjusted_ratio"));
    assert!(markdown.contains("preferred_skin_rendering_mean_delta_e_ab"));
    assert!(markdown.contains("noise_reduction_structure_gate_start"));
    assert!(markdown.contains("noise_reduction_structure_excluded_ratio"));
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
    assert!(markdown.contains("neutral_safety_rescue_applied"));
    assert!(markdown.contains("neutral_safety_rescue_preserved_ratio_gain"));
    assert!(markdown.contains("neutral_safety_rescue_midtone_saturation_p95_reduction"));
    assert!(markdown.contains("neutral_safety_rescue_reason"));
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
    assert!(markdown.contains("adaptive_vibrance_skin_memory_protection_enabled"));
    assert!(markdown.contains("adaptive_vibrance_skin_memory_protected_ratio"));
    assert!(markdown.contains("render_luminance_range_p05_p95"));
    assert!(markdown.contains("tone_output_confidence_status"));
    assert!(markdown.contains("tone_output_evidence_confidence"));
    assert!(markdown.contains("render_to_mapped_luminance_range_ratio"));
    assert!(markdown.contains("maximum_post_tone_high_clip_ratio"));
    assert!(markdown.contains("high_frequency_luma_residual_p95"));
    assert!(markdown.contains("high_frequency_flat_chroma_residual_p95"));
    assert!(markdown.contains("grain_detail_review_required"));
    assert!(markdown.contains("grain_detail_decision_supported"));
    assert!(markdown.contains("grain_detail_luminance_p10_retention"));
    assert!(markdown.contains("grain_detail_chroma_p10_retention"));
    assert!(markdown.contains("neutral_balance_weak_anchor_fallback"));
    assert!(markdown.contains("bright_neutral_saturation_p95"));
    assert!(markdown.contains("bright_neutral_rgb_median"));
    assert!(markdown.contains("colorspace matrix has weak dominant-channel anchor support"));
}

#[test]
fn test_validation_summary_rejects_contradictory_calibration_application_claim() {
    let mut report = fixture_report();
    let colorspace = report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "colorspace_mapping")
        .expect("colorspace phase");
    colorspace.metrics["calibration"]["color_mapping_application"]["applied"] =
        serde_json::json!(true);

    let summary = scanstitch::validation::summarize_report("fixture", &report);
    assert!(summary
        .diagnostic_consistency_issues
        .contains(&"calibration_color_mapping_applied_status_mismatch".to_string()));
}

#[test]
fn test_validation_summary_rejects_contradictory_preferred_memory_color_guard_population() {
    let mut report = fixture_report();
    let tone = report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "tone_mapping")
        .expect("tone phase");
    tone.metrics["adaptive_vibrance_preferred_memory_color_guard"]["limited_pixel_ratio"] =
        serde_json::json!(0.25);

    let summary = scanstitch::validation::summarize_report("fixture", &report);
    assert!(summary
        .diagnostic_consistency_issues
        .contains(&"preferred_memory_color_guard_limited_exceeds_matched".to_string()));
    assert!(summary
        .diagnostic_consistency_issues
        .contains(&"preferred_memory_color_guard_family_limited_sum_mismatch".to_string()));
}

#[test]
fn test_validation_summary_rejects_contradictory_preferred_skin_rendering_effect() {
    let mut report = fixture_report();
    let tone = report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "tone_mapping")
        .expect("tone phase");
    tone.metrics["preferred_skin_rendering"]["adjusted_pixel_ratio"] = serde_json::json!(0.20);
    tone.metrics["preferred_skin_rendering"]["max_delta_e_ab"] = serde_json::json!(3.5);
    tone.metrics["preferred_skin_rendering"]["mean_chroma_delta"] = serde_json::json!(0.1);

    let summary = scanstitch::validation::summarize_report("fixture", &report);
    assert!(summary
        .diagnostic_consistency_issues
        .contains(&"preferred_skin_rendering_adjusted_exceeds_outside_core".to_string()));
    assert!(summary
        .diagnostic_consistency_issues
        .contains(&"preferred_skin_rendering_delta_e_cap_exceeded".to_string()));
    assert!(summary
        .diagnostic_consistency_issues
        .contains(&"preferred_skin_rendering_chroma_increase_detected".to_string()));
}

#[test]
fn test_validation_summary_rejects_contradictory_grain_selectivity_evidence() {
    let mut report = fixture_report();
    let tone = report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "tone_mapping")
        .expect("tone phase");
    tone.metrics["noise_reduction_structure_gate_start"] = serde_json::json!(0.04);
    tone.metrics["noise_reduction_structure_excluded_ratio"] = serde_json::json!(0.30);
    tone.metrics["noise_reduction_mean_abs_chroma_delta"] = serde_json::json!(0.03);

    let summary = scanstitch::validation::summarize_report("fixture", &report);
    assert!(summary
        .diagnostic_consistency_issues
        .contains(&"grain_reduction_structure_gate_invalid".to_string()));
    assert!(summary
        .diagnostic_consistency_issues
        .contains(&"grain_reduction_applied_and_excluded_overlap".to_string()));
    assert!(summary
        .diagnostic_consistency_issues
        .contains(&"grain_reduction_chroma_delta_mean_exceeds_max".to_string()));
}

#[test]
fn test_synthetic_color_suite_proves_required_decision_edges() {
    let suite = scanstitch::validation::run_synthetic_color_suite();

    assert_eq!(suite.status, "passed", "issues: {:?}", suite.issues);
    assert_eq!(suite.cases.len(), 24);
    assert!(suite.cases.iter().any(|case| {
        case.name == "calibrated_profile_beats_image_derived"
            && case.actual_selected_candidate.as_deref() == Some("calibrated_direct_profile")
            && case.actual_calibration_acceptance_status.as_deref() == Some("accepted")
            && case.actual_calibration_beats_image_derived == Some(true)
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "root_polynomial_selected_inside_measured_support"
            && case.actual_selected_candidate.as_deref() == Some("calibrated_root_polynomial")
            && case.actual_mapping_strategy.as_deref() == Some("calibrated_root_polynomial")
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "root_polynomial_rejected_outside_measured_support"
            && case.actual_selected_candidate.as_deref() == Some("calibrated_direct_profile")
            && case.actual_mapping_strategy.as_deref() == Some("calibrated_profile")
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "residual_lut_selected_inside_measured_support"
            && case.actual_selected_candidate.as_deref() == Some("calibrated_residual_lut_3d")
            && case.actual_mapping_strategy.as_deref() == Some("calibrated_residual_lut_3d")
    }));
    assert!(suite.cases.iter().any(|case| {
        case.name == "residual_lut_rejected_outside_measured_rgb_volume"
            && case.actual_selected_candidate.as_deref() == Some("calibrated_direct_profile")
            && case.actual_mapping_strategy.as_deref() == Some("calibrated_profile")
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
            && case.actual_selected_candidate.as_deref() == Some("neutral_balance_fallback")
            && case.actual_mapping_strategy.as_deref() == Some("neutral_balance_evidence_rescue")
            && case.actual_candidate_risk.as_deref() == Some("fallback_only")
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
    assert_eq!(selected_candidate_cases.len(), 16);
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
        case.name == "tone_policy_model_review_keeps_bounded_neutral_cleanup"
            && case.actual_tone_policy.as_deref() == Some("review_bounded_neutral_shadow_cleanup")
            && case.actual_tone_policy_color_trust_state.as_deref() == Some("review_required")
            && case.actual_tone_highlight_neutral_chroma_enabled == Some(true)
            && case.actual_tone_shadow_chroma_enabled == Some(true)
            && case
                .actual_tone_highlight_neutral_chroma_compressed_ratio
                .is_some_and(|ratio| ratio > 0.0)
            && case
                .actual_tone_shadow_chroma_compressed_ratio
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
    assert_eq!(summary["cases"].as_array().map(Vec::len), Some(24));
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
    assert!(markdown.contains("root_polynomial_selected_inside_measured_support"));
    assert!(markdown.contains("root_polynomial_rejected_outside_measured_support"));
    assert!(markdown.contains("residual_lut_selected_inside_measured_support"));
    assert!(markdown.contains("residual_lut_rejected_outside_measured_rgb_volume"));
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
    assert!(markdown.contains("neutral_balance_evidence_rescue"));
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
    assert!(markdown.contains("tone_policy_model_review_keeps_bounded_neutral_cleanup"));
    assert!(markdown.contains("review_bounded_neutral_shadow_cleanup"));
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
fn test_validate_cli_strict_rejects_contradictory_calibration_application_claim() {
    let tmp = tempfile::TempDir::new().unwrap();
    let report_path = tmp.path().join("report.json");
    let summary_json = tmp.path().join("summary.json");
    let summary_md = tmp.path().join("summary.md");
    let output_path = tmp.path().join("output.tiff");
    scanstitch::tiff_io::save_tiff_f64_linear_prophoto(&small_rgb_image(), &output_path).unwrap();

    let mut report = fixture_report();
    set_report_output_path(&mut report, &output_path);
    let colorspace = report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "colorspace_mapping")
        .expect("colorspace phase");
    colorspace.metrics["calibration"]["color_mapping_application"]["applied"] =
        serde_json::json!(true);
    report.save(&report_path).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture")
        .arg("calibration-consistency")
        .arg("--report")
        .arg(&report_path)
        .arg("--strict")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run scanstitch-validate with contradictory calibration application claim");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("strict report consistency validation failed")
            && stderr.contains("calibration_color_mapping_applied_status_mismatch"),
        "unexpected stderr: {stderr}"
    );
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert!(summary["diagnostic_consistency_issues"]
        .as_array()
        .is_some_and(|issues| issues
            .iter()
            .any(|issue| issue == "calibration_color_mapping_applied_status_mismatch")));
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
                "bad-overrides": {
                    "component1": "left.tif",
                    "component2": "right.tif",
                    "input_mode": "not-a-mode",
                    "deskew": "sideways",
                    "orientation_correction": "sideways",
                    "bit_depth": 12,
                    "grain_reduction": "sideways",
                    "grain_strength": 1.1,
                    "grain_scale": 0.1,
                    "force_stitch": true,
                    "force_no_stitch": true
                },
                "grain-conflict": {
                    "component1": "left.tif",
                    "component2": "right.tif",
                    "grain_reduction": "on",
                    "expectations": {
                        "grain_reduction_enabled": false
                    }
                },
                "single-orphans": {
                    "component1": "single.tif",
                    "component2_sha256": "0".repeat(64),
                    "additional_components": [{
                        "path": "third.tif"
                    }]
                },
                "single-force-stitch": {
                    "component1": "single.tif",
                    "force_stitch": true
                },
                "bad-expectations": {
                    "component1": "left.tif",
                    "component2": "right.tif",
                    "component1_sha256": "not-a-sha256",
                    "additional_components": [{
                        "path": "left.tif",
                        "sha256": "0".repeat(64)
                    }],
                    "summary_baseline_sha256": "not-a-sha256",
                    "calibration_library_sha256": "not-a-sha256",
                    "reference_evidence": ["gray-card", "gray-card"],
                    "expectations": {
                        "deskew_status": "",
                        "deskew_retained_area_ratio_min": 1.1,
                        "orientation_components_expected": [{
                            "component_index": 4,
                            "upright_approved": false,
                            "decoded_pixel_sha256": "not-a-sha256",
                            "tag_present": false,
                            "tag_value": 6,
                            "transform": "",
                            "applied": false,
                            "source_width": 0,
                            "source_height": 0,
                            "output_width": 0,
                            "output_height": 0
                        }],
                        "stitch_decision": "",
                        "inferred_component_order": [1, 1, 4],
                        "technical_white_balance_status": "",
                        "creative_temperature": 1.1,
                        "creative_tint": -1.1,
                        "seam_exposure_model": "",
                        "seam_exposure_held_out_improvement_over_gain_min": -0.1,
                        "seam_exposure_offset_normalized_abs_max": 1.1,
                        "seam_exposure_spatial_slope_abs_min": 0.3,
                        "seam_exposure_spatial_slope_abs_max": 0.2,
                        "seam_exposure_spatial_slope_agreement_ratio_min": 1.1,
                        "seam_exposure_held_out_spatial_improvement_over_best_constant_min": -0.1,
                        "seam_exposure_spatial_offset_slope_normalized_abs_min": 0.3,
                        "seam_exposure_spatial_offset_slope_normalized_abs_max": 0.2,
                        "seam_exposure_spatial_offset_endpoint_normalized_abs_max": 1.1,
                        "seam_exposure_spatial_affine_slope_agreement_ratio_min": 1.1,
                        "seam_exposure_spatial_affine_center_offset_delta_normalized_max": 1.1,
                        "seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler_min": -0.1,
                        "seam_exposure_spatial_2d_distinct_columns_min": 0,
                        "seam_exposure_spatial_2d_horizontal_slope_agreement_ratio_min": 1.1,
                        "seam_exposure_held_out_spatial_2d_improvement_over_best_simpler_min": -0.1,
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
                        "neutral_safety_rescue_preserved_ratio_gain_min": 1.1,
                        "neutral_safety_rescue_midtone_saturation_p95_reduction_min": 1.1,
                        "neutral_safety_rescue_reason_contains": "",
                        "highlight_chroma_compressed_ratio_min": 1.1,
                        "highlight_chroma_compressed_ratio_max": 1.1,
                        "highlight_neutral_chroma_compressed_ratio_max": 1.1,
                        "shadow_chroma_compressed_ratio_max": 1.1,
                        "grain_reduction_applied_ratio_min": 1.1,
                        "grain_reduction_structure_excluded_ratio_min": 1.1,
                        "grain_reduction_flat_luma_p95_reduction_ratio_min": 1.1,
                        "grain_reduction_flat_chroma_p95_reduction_ratio_min": 1.1,
                        "memory_color_penalty_max": -0.1,
                        "spatial_consistency_penalty_max": -0.1,
                        "selected_runner_up_quality_delta_min": -0.1,
                        "density_monotonicity_score_min": -0.1,
                        "hue_linearity_score_min": -0.1,
                        "saturation_preservation_median_ratio_min": -0.1,
                        "spatial_neutral_delta_p95_max": -0.1,
                        "post_scale_preserved_ratio_min": 1.1,
                        "render_luminance_range_p05_p95_min": 1.1,
                        "render_review_status": "",
                        "tone_output_confidence_status": "",
                        "tone_output_evidence_confidence_min": 1.1,
                        "render_to_mapped_luminance_range_ratio_min": -0.1,
                        "post_chroma_compression_clipped_high_ratio_max": 1.1,
                        "post_chroma_compression_clipped_low_ratio_max": 1.1,
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
        "bad-overrides:input_mode_invalid:not-a-mode",
        "bad-overrides:deskew_invalid:sideways",
        "bad-overrides:orientation_correction_invalid:sideways",
        "bad-overrides:bit_depth_invalid",
        "bad-overrides:grain_reduction_invalid:sideways",
        "bad-overrides:grain_strength_invalid",
        "bad-overrides:grain_scale_invalid",
        "bad-overrides:force_stitch_and_force_no_stitch",
        "grain-conflict:grain_reduction_conflicts_with_enabled_expectation",
        "single-orphans:component2_sha256_without_component2",
        "single-orphans:additional_components_without_component2",
        "single-force-stitch:force_stitch_requires_multiple_components",
        "bad-expectations:component1_sha256_invalid",
        "bad-expectations:component_sha256_pair_incomplete",
        "bad-expectations:component_sha256_set_incomplete",
        "bad-expectations:component3_path_duplicate",
        "bad-expectations:summary_baseline_sha256_invalid",
        "bad-expectations:summary_baseline_sha256_without_summary_baseline",
        "bad-expectations:calibration_library_sha256_invalid",
        "bad-expectations:calibration_library_sha256_without_calibration_library",
        "bad-expectations:reference_evidence_duplicate:gray-card",
        "bad-expectations:expectations.deskew_status_empty",
        "bad-expectations:expectations.deskew_retained_area_ratio_min_out_of_range",
        "bad-expectations:expectations.orientation_components_expected_component_out_of_range:4",
        "bad-expectations:expectations.orientation_components_expected_decoded_pixel_sha256_invalid:4",
        "bad-expectations:expectations.orientation_components_expected_tag_invalid:4",
        "bad-expectations:expectations.orientation_components_expected_transform_empty:4",
        "bad-expectations:expectations.orientation_components_expected_dimensions_out_of_range:4",
        "bad-expectations:expectations.stitch_decision_empty",
        "bad-expectations:expectations.technical_white_balance_status_empty",
        "bad-expectations:expectations.inferred_component_order_duplicate:1",
        "bad-expectations:expectations.inferred_component_order_out_of_range:4",
        "bad-expectations:expectations.creative_temperature_out_of_range",
        "bad-expectations:expectations.creative_tint_out_of_range",
        "bad-expectations:expectations.seam_exposure_model_empty",
        "bad-expectations:expectations.seam_exposure_held_out_improvement_over_gain_min_out_of_range",
        "bad-expectations:expectations.seam_exposure_offset_normalized_abs_max_out_of_range",
        "bad-expectations:expectations.seam_exposure_spatial_slope_bounds_inverted",
        "bad-expectations:expectations.seam_exposure_spatial_slope_agreement_ratio_min_out_of_range",
        "bad-expectations:expectations.seam_exposure_held_out_spatial_improvement_over_best_constant_min_out_of_range",
        "bad-expectations:expectations.seam_exposure_spatial_offset_slope_bounds_inverted",
        "bad-expectations:expectations.seam_exposure_spatial_offset_endpoint_normalized_abs_max_out_of_range",
        "bad-expectations:expectations.seam_exposure_spatial_affine_slope_agreement_ratio_min_out_of_range",
        "bad-expectations:expectations.seam_exposure_spatial_affine_center_offset_delta_normalized_max_out_of_range",
        "bad-expectations:expectations.seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler_min_out_of_range",
        "bad-expectations:expectations.seam_exposure_spatial_2d_distinct_columns_min_out_of_range",
        "bad-expectations:expectations.seam_exposure_spatial_2d_horizontal_slope_agreement_ratio_min_out_of_range",
        "bad-expectations:expectations.seam_exposure_held_out_spatial_2d_improvement_over_best_simpler_min_out_of_range",
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
        "bad-expectations:expectations.neutral_safety_rescue_preserved_ratio_gain_min_out_of_range",
        "bad-expectations:expectations.neutral_safety_rescue_midtone_saturation_p95_reduction_min_out_of_range",
        "bad-expectations:expectations.neutral_safety_rescue_reason_contains_empty",
        "bad-expectations:expectations.highlight_chroma_compressed_ratio_min_out_of_range",
        "bad-expectations:expectations.highlight_chroma_compressed_ratio_max_out_of_range",
        "bad-expectations:expectations.highlight_neutral_chroma_compressed_ratio_max_out_of_range",
        "bad-expectations:expectations.shadow_chroma_compressed_ratio_max_out_of_range",
        "bad-expectations:expectations.grain_reduction_applied_ratio_min_out_of_range",
        "bad-expectations:expectations.grain_reduction_structure_excluded_ratio_min_out_of_range",
        "bad-expectations:expectations.grain_reduction_flat_luma_p95_reduction_ratio_min_out_of_range",
        "bad-expectations:expectations.grain_reduction_flat_chroma_p95_reduction_ratio_min_out_of_range",
        "bad-expectations:expectations.memory_color_penalty_max_out_of_range",
        "bad-expectations:expectations.spatial_consistency_penalty_max_out_of_range",
        "bad-expectations:expectations.selected_runner_up_quality_delta_min_out_of_range",
        "bad-expectations:expectations.density_monotonicity_score_min_out_of_range",
        "bad-expectations:expectations.hue_linearity_score_min_out_of_range",
        "bad-expectations:expectations.saturation_preservation_median_ratio_min_out_of_range",
        "bad-expectations:expectations.spatial_neutral_delta_p95_max_out_of_range",
        "bad-expectations:expectations.post_scale_preserved_ratio_min_out_of_range",
        "bad-expectations:expectations.render_luminance_range_p05_p95_min_out_of_range",
        "bad-expectations:expectations.render_review_status_empty",
        "bad-expectations:expectations.tone_output_confidence_status_empty",
        "bad-expectations:expectations.tone_output_evidence_confidence_min_out_of_range",
        "bad-expectations:expectations.render_to_mapped_luminance_range_ratio_min_out_of_range",
        "bad-expectations:expectations.post_chroma_compression_clipped_high_ratio_max_out_of_range",
        "bad-expectations:expectations.post_chroma_compression_clipped_low_ratio_max_out_of_range",
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
        Some("rejected_reference_fit")
    );
    assert_eq!(
        baseline.colorspace.calibration_color_mapping_applied,
        Some(false)
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
    assert_eq!(baseline.grain.detail_review_required, Some(false));
    assert_eq!(baseline.grain.detail_decision_supported, Some(true));
    assert_eq!(baseline.grain.luminance_p10_retention, Some(0.94));
    assert_eq!(baseline.grain.chroma_p10_retention, Some(0.92));

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
    let path3 = tmp.path().join("middle.tiff");
    let baseline_path = tmp.path().join("baseline.json");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("coverage.json");
    let summary_md = tmp.path().join("coverage.md");

    let component = synthetic::constant_image(4, 4, [12000, 7000, 3000]);
    scanstitch::tiff_io::save_tiff_u16(&component, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&component, &path2).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&component, &path3).unwrap();
    let component1_sha256 = file_sha256_hex(&path1);
    let component2_sha256 = file_sha256_hex(&path2);
    let component3_sha256 = file_sha256_hex(&path3);
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
                "min_n_component_fixtures": 1,
                "min_component_sha256_sets": 1,
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
                "min_render_dynamic_range_contract_fixtures": 1,
                "min_stitch_normalization_contract_fixtures": 1,
                "min_geometry_preparation_contract_fixtures": 1,
                "min_geometry_accuracy_contract_fixtures": 1,
                "min_orientation_accuracy_contract_fixtures": 1,
                "min_negative_reconstruction_contract_fixtures": 1,
                "min_grain_reduction_enabled_fixtures": 1,
                "min_grain_detail_contract_fixtures": 1,
                "min_grain_reduction_effect_contract_fixtures": 1,
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
                    "additional_components": [{
                        "path": path3,
                        "sha256": component3_sha256.clone()
                    }],
                    "output_dir": tmp.path().join("out"),
                    "input_mode": "negative",
                    "grain_reduction": "on",
                    "grain_strength": 0.6,
                    "grain_scale": 1.0,
                    "deskew": "manual",
                    "deskew_angle_degrees": 0.5,
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
                        "deskew_status": "applied",
                        "deskew_applied": true,
                        "deskew_review_required": false,
                        "deskew_all_components_applied": true,
                        "deskew_minimum_component_retained_area_ratio_min": 0.9,
                        "border_crop_all_components_cropped": true,
                        "border_crop_minimum_removed_edge_count_per_component_min": 4,
                        "border_crop_retained_area_ratio_min": 0.5,
                        "border_crop_retained_area_ratio_max": 0.99,
                        "border_crop_rejected": false,
                        "deskew_correction_degrees_expected": -0.5,
                        "deskew_correction_tolerance_degrees": 0.05,
                        "border_crop_components_expected": [
                            {
                                "component_index": 1,
                                "top_removed": 1,
                                "bottom_removed": 1,
                                "left_removed": 1,
                                "right_removed": 1,
                                "tolerance_px": 4
                            },
                            {
                                "component_index": 2,
                                "top_removed": 1,
                                "bottom_removed": 1,
                                "left_removed": 1,
                                "right_removed": 1,
                                "tolerance_px": 4
                            },
                            {
                                "component_index": 3,
                                "top_removed": 1,
                                "bottom_removed": 1,
                                "left_removed": 1,
                                "right_removed": 1,
                                "tolerance_px": 4
                            }
                        ],
                        "orientation_components_expected": [
                            {
                                "component_index": 1,
                                "upright_approved": true,
                                "decoded_pixel_sha256": "1".repeat(64),
                                "tag_present": false,
                                "tag_value": null,
                                "transform": "identity",
                                "applied": false,
                                "source_width": 4,
                                "source_height": 4,
                                "output_width": 4,
                                "output_height": 4
                            },
                            {
                                "component_index": 2,
                                "upright_approved": true,
                                "decoded_pixel_sha256": "2".repeat(64),
                                "tag_present": false,
                                "tag_value": null,
                                "transform": "identity",
                                "applied": false,
                                "source_width": 4,
                                "source_height": 4,
                                "output_width": 4,
                                "output_height": 4
                            },
                            {
                                "component_index": 3,
                                "upright_approved": true,
                                "decoded_pixel_sha256": "3".repeat(64),
                                "tag_present": false,
                                "tag_value": null,
                                "transform": "identity",
                                "applied": false,
                                "source_width": 4,
                                "source_height": 4,
                                "output_width": 4,
                                "output_height": 4
                            }
                        ],
                        "stitch_decision": "accepted_sequence",
                        "inferred_component_order": [1, 3, 2],
                        "seam_exposure_model": "identity",
                        "seam_exposure_held_out_validation_passed": false,
                        "seam_exposure_offset_normalized_abs_max": 0.0,
                        "seam_blend_required": true,
                        "seam_blend_mode": "seam_aware_multiband",
                        "seam_blend_review_required": false,
                        "seam_detail_review_required": false,
                        "seam_detail_supported_scale_count_min": 2,
                        "seam_detail_max_symmetric_energy_ratio_max": 2.0,
                        "seam_gradient_ratio_max": 1.05,
                        "seam_overlap_p95_abs_difference_max": 0.99,
                        "base_estimate_source": "pre_crop_rebate_measurement",
                        "base_confidence_min": 0.3,
                        "density_inversion_skipped": false,
                        "negative_response_model": "measured_nonlinear_dye_separation",
                        "negative_response_source": "measured_roll_target_with_held_out_validation",
                        "negative_response_accepted": true,
                        "negative_response_review_required": false,
                        "negative_response_crosstalk_model": "measured_3x3_scanner_density_to_film_layers",
                        "negative_response_characteristic_curve_model": "measured_monotone_pchip_density_to_scene_log_exposure",
                        "negative_response_measured_model_id": "synthetic-response-v1",
                        "negative_response_measured_confidence_min": 0.75,
                        "negative_response_held_out_delta_e00_rms_max": 6.0,
                        "negative_response_held_out_max_delta_e00_max": 15.0,
                        "negative_response_held_out_improvement_over_unit_slope_min": 0.25,
                        "negative_response_density_noise_gain_max": 4.0,
                        "negative_response_curve_extrapolated_ratio_max": 0.01,
                        "negative_response_signed_headroom_preserved": true,
                        "negative_response_curve_interpolation": "monotone_piecewise_cubic_hermite_with_endpoint_tangent_extrapolation",
                        "render_input_source": "direct_density_transmittance",
                        "calibration_acceptance_status": "accepted",
                        "calibration_color_mapping_applied": true,
                        "candidate_risk": "safe",
                        "tone_color_trust_state": "trusted",
                        "neutral_safety_rescue_applied": false,
                        "output_color_space": "linear_prophoto_rgb_d50",
                        "grain_reduction_enabled": true,
                        "grain_reduction_applied_ratio_min": 0.01,
                        "grain_reduction_structure_excluded_ratio_min": 0.01,
                        "grain_reduction_flat_luma_p95_reduction_ratio_min": 0.02,
                        "grain_reduction_flat_chroma_p95_reduction_ratio_min": 0.05,
                        "grain_detail_review_required": false,
                        "grain_detail_decision_supported": true,
                        "grain_detail_luminance_probe_count_min": 64,
                        "grain_detail_chroma_probe_count_min": 64,
                        "grain_detail_luminance_p10_retention_min": 0.7,
                        "grain_detail_chroma_p10_retention_min": 0.7,
                        "post_scale_preserved_ratio_min": 0.98,
                        "render_luminance_range_p05_p95_min": 0.4,
                        "render_review_status": "reviewable",
                        "render_reviewable": true,
                        "tone_output_confidence_status": "supported_render_tonal_distribution",
                        "tone_output_review_required": false,
                        "tone_output_evidence_confidence_min": 1.0,
                        "render_to_mapped_luminance_range_ratio_min": 0.08,
                        "post_chroma_compression_clipped_high_ratio_max": 0.01,
                        "post_chroma_compression_clipped_low_ratio_max": 0.01,
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
    assert_eq!(summary["component_file_count"], 3);
    assert_eq!(summary["max_component_count"], 3);
    assert_eq!(summary["n_component_fixture_count"], 1);
    assert_eq!(summary["n_component_validation_ready_fixture_count"], 1);
    assert_eq!(summary["component_set_available_count"], 1);
    assert_eq!(summary["component_sha256_declared_set_count"], 1);
    assert_eq!(summary["component_sha256_computed_set_count"], 1);
    assert_eq!(summary["component_sha256_set_count"], 1);
    assert_eq!(summary["readable_tiff_set_count"], 1);
    assert_eq!(summary["tiff_layout_consistent_set_count"], 1);
    assert_eq!(summary["tiff_dimension_matched_set_count"], 1);
    assert_eq!(summary["component_pair_available_count"], 1);
    assert_eq!(summary["component_sha256_declared_pair_count"], 1);
    assert_eq!(summary["component_sha256_computed_pair_count"], 1);
    assert_eq!(summary["component_sha256_pair_count"], 1);
    assert_eq!(summary["readable_tiff_pair_count"], 1);
    assert_eq!(summary["tiff_layout_consistent_pair_count"], 1);
    assert_eq!(summary["tiff_dimension_matched_pair_count"], 1);
    assert_eq!(summary["fixtures"][0]["component_count"], 3);
    assert_eq!(summary["fixtures"][0]["all_components_exist"], true);
    assert_eq!(
        summary["fixtures"][0]["additional_components"][0]["component_number"],
        3
    );
    assert_eq!(
        summary["fixtures"][0]["additional_components"][0]["sha256"]["matched"],
        true
    );
    assert_eq!(summary["validation_ready_fixture_count"], 1);
    assert_eq!(summary["render_dynamic_range_contract_fixture_count"], 1);
    assert_eq!(summary["stitch_normalization_contract_fixture_count"], 1);
    assert_eq!(summary["geometry_preparation_contract_fixture_count"], 1);
    assert_eq!(summary["geometry_accuracy_contract_fixture_count"], 1);
    assert_eq!(summary["orientation_accuracy_contract_fixture_count"], 1);
    assert_eq!(summary["negative_reconstruction_contract_fixture_count"], 1);
    assert_eq!(summary["grain_reduction_enabled_fixture_count"], 1);
    assert_eq!(summary["grain_detail_contract_fixture_count"], 1);
    assert_eq!(summary["grain_reduction_effect_contract_fixture_count"], 1);
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
        summary["coverage_requirements"]["min_render_dynamic_range_contract_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        summary["coverage_requirements"]["min_stitch_normalization_contract_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        summary["coverage_requirements"]["min_geometry_accuracy_contract_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        summary["coverage_requirements"]["min_orientation_accuracy_contract_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        summary["fixtures"][0]["geometry_accuracy_contract_complete"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["geometry_accuracy_contract_missing_fields"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["fixtures"][0]["orientation_accuracy_contract_complete"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["orientation_accuracy_contract_missing_fields"],
        serde_json::json!([])
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
        summary["fixtures"][0]["grain_reduction_enabled_declared"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["grain_detail_contract_complete"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["grain_detail_contract_missing_fields"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["fixtures"][0]["grain_reduction_effect_contract_complete"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["grain_reduction_effect_contract_missing_fields"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["fixtures"][0]["render_dynamic_range_contract_declared"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["render_dynamic_range_contract_complete"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["render_dynamic_range_contract_missing_fields"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["fixtures"][0]["stitch_normalization_contract_declared"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["stitch_normalization_contract_complete"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["stitch_normalization_contract_missing_fields"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["fixtures"][0]["geometry_preparation_contract_complete"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["geometry_preparation_contract_missing_fields"],
        serde_json::json!([])
    );
    assert_eq!(
        summary["fixtures"][0]["negative_reconstruction_contract_complete"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["negative_reconstruction_contract_missing_fields"],
        serde_json::json!([])
    );
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
    assert!(markdown.contains("complete render-dynamic-range contracts"));
    assert!(markdown.contains("complete stitch-normalization contracts"));
    assert!(markdown.contains("contract=complete"));
    assert!(markdown.contains("grain-reduction-enabled fixtures"));
    assert!(markdown.contains("complete grain-detail contracts"));
    assert!(markdown.contains("complete grain-reduction-effect contracts"));
    assert!(markdown.contains("detail-contract=complete"));
    assert!(markdown.contains("effect-contract=complete"));
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
fn test_validate_cli_fixture_coverage_requires_complete_grain_detail_contract() {
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
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_grain_reduction_enabled_fixtures": 1,
                "min_grain_detail_contract_fixtures": 1,
                "min_grain_reduction_effect_contract_fixtures": 1
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "summary_baseline": baseline_path,
                    "grain_reduction": "on",
                    "grain_strength": 0.0,
                    "grain_scale": 1.0,
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["fine-texture"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "uncalibrated-image-derived",
                    "expectations": {
                        "grain_reduction_enabled": true
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
        .expect("run scanstitch-validate incomplete grain-detail coverage");

    assert!(
        !output.status.success(),
        "an enabled denoise declaration must not substitute for a complete detail contract"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("fixture_coverage_min_grain_detail_contract_fixtures_not_met"));
    assert!(
        stderr.contains("fixture_coverage_min_grain_reduction_effect_contract_fixtures_not_met")
    );
    assert!(!stderr.contains("fixture_coverage_min_grain_reduction_enabled_fixtures_not_met"));

    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["validation_ready_fixture_count"], 1);
    assert_eq!(summary["grain_reduction_enabled_fixture_count"], 1);
    assert_eq!(summary["grain_detail_contract_fixture_count"], 0);
    assert_eq!(summary["grain_reduction_effect_contract_fixture_count"], 0);
    assert_eq!(
        summary["fixtures"][0]["grain_detail_contract_complete"],
        false
    );
    assert_eq!(
        summary["fixtures"][0]["grain_detail_contract_missing_fields"],
        serde_json::json!([
            "grain_strength>0",
            "expectations.grain_detail_review_required=false",
            "expectations.grain_detail_decision_supported=true",
            "expectations.grain_detail_luminance_probe_count_min>=64",
            "expectations.grain_detail_chroma_probe_count_min>=64",
            "expectations.grain_detail_luminance_p10_retention_min>=0.70",
            "expectations.grain_detail_chroma_p10_retention_min>=0.70"
        ])
    );
    assert_eq!(
        summary["fixtures"][0]["action_items"],
        serde_json::json!([
            "complete_grain_detail_contract_for_fixture:logan",
            "complete_grain_reduction_effect_contract_for_fixture:logan"
        ])
    );
    assert_eq!(
        summary["fixtures"][0]["grain_reduction_effect_contract_missing_fields"],
        serde_json::json!([
            "grain_detail_contract_complete",
            "expectations.grain_reduction_applied_ratio_min>0",
            "expectations.grain_reduction_structure_excluded_ratio_min>0",
            "expectations.grain_reduction_flat_luma_p95_reduction_ratio_min>0",
            "expectations.grain_reduction_flat_chroma_p95_reduction_ratio_min>0"
        ])
    );
    assert_eq!(
        summary["action_items"],
        serde_json::json!([
            "add_complete_grain_detail_contracts_for_validation_ready_fixtures:1",
            "add_complete_grain_reduction_effect_contracts_for_validation_ready_fixtures:1",
            "complete_grain_detail_contract_for_fixture:logan",
            "complete_grain_reduction_effect_contract_for_fixture:logan"
        ])
    );
    assert!(summary["fixtures"][0]["repair_plan"][0]["details"]
        .as_str()
        .unwrap()
        .contains("both-channel"));
    assert!(summary["fixtures"][0]["repair_plan"][1]["details"]
        .as_str()
        .unwrap()
        .contains("exact structure-excluded"));

    let markdown = std::fs::read_to_string(&summary_md).unwrap();
    assert!(markdown.contains("detail-contract=incomplete"));
    assert!(markdown.contains("effect-contract=incomplete"));
    assert!(markdown.contains("complete_grain_detail_contract_for_fixture:logan"));
    assert!(markdown.contains("complete_grain_reduction_effect_contract_for_fixture:logan"));
}

#[test]
fn test_validate_cli_fixture_coverage_requires_complete_render_dynamic_range_contract() {
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
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_render_dynamic_range_contract_fixtures": 1
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "summary_baseline": baseline_path,
                    "film_stock": "Synthetic positive",
                    "scene_tags": ["wide-latitude"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "uncalibrated-image-derived",
                    "expectations": {
                        "post_scale_preserved_ratio_min": 0.98
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
        .expect("run scanstitch-validate incomplete render-dynamic-range coverage");

    assert!(
        !output.status.success(),
        "one preservation floor must not substitute for a complete dynamic-range contract"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("fixture_coverage_min_render_dynamic_range_contract_fixtures_not_met"));

    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["validation_ready_fixture_count"], 1);
    assert_eq!(summary["render_dynamic_range_contract_fixture_count"], 0);
    assert_eq!(
        summary["fixtures"][0]["render_dynamic_range_contract_declared"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["render_dynamic_range_contract_complete"],
        false
    );
    assert_eq!(
        summary["fixtures"][0]["render_dynamic_range_contract_missing_fields"],
        serde_json::json!([
            "expectations.render_luminance_range_p05_p95_min>0",
            "expectations.render_review_status=reviewable",
            "expectations.render_reviewable=true",
            "expectations.tone_output_confidence_status=supported_render_tonal_distribution",
            "expectations.tone_output_review_required=false",
            "expectations.tone_output_evidence_confidence_min>0",
            "expectations.render_to_mapped_luminance_range_ratio_min>0",
            "expectations.post_chroma_compression_clipped_high_ratio_max<1",
            "expectations.post_chroma_compression_clipped_low_ratio_max<1"
        ])
    );
    assert!(summary["fixtures"][0]["action_items"]
        .as_array()
        .unwrap()
        .iter()
        .any(|action| action == "complete_render_dynamic_range_contract_for_fixture:logan"));
    assert!(summary["action_items"]
        .as_array()
        .unwrap()
        .iter()
        .any(|action| {
            action == "add_complete_render_dynamic_range_contracts_for_validation_ready_fixtures:1"
        }));
    let repair_detail = summary["fixtures"][0]["repair_plan"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["action"] == "complete_render_dynamic_range_contract_for_fixture:logan")
        .and_then(|item| item["details"].as_str())
        .unwrap();
    assert!(repair_detail.contains("p05-p95"));
    assert!(repair_detail.contains("render_reviewable"));
    assert!(repair_detail.contains("tone_output_confidence_status"));
    assert!(repair_detail.contains("render_to_mapped"));
    assert!(repair_detail.contains("high/low"));

    let markdown = std::fs::read_to_string(&summary_md).unwrap();
    assert!(markdown.contains("contract=incomplete"));
    assert!(markdown.contains("complete_render_dynamic_range_contract_for_fixture:logan"));
}

#[test]
fn test_validate_cli_fixture_coverage_requires_complete_stitch_normalization_contract() {
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
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_stitch_normalization_contract_fixtures": 1
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "summary_baseline": baseline_path,
                    "film_stock": "Synthetic positive",
                    "scene_tags": ["overlap-detail"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "uncalibrated-image-derived",
                    "expectations": {
                        "stitch_decision": "accepted"
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
        .expect("run incomplete stitch-normalization fixture coverage");

    assert!(
        !output.status.success(),
        "an accepted decision alone must not substitute for normalized-stitch evidence"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("fixture_coverage_min_stitch_normalization_contract_fixtures_not_met"));

    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["validation_ready_fixture_count"], 1);
    assert_eq!(summary["stitch_normalization_contract_fixture_count"], 0);
    assert_eq!(
        summary["fixtures"][0]["stitch_normalization_contract_declared"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["stitch_normalization_contract_complete"],
        false
    );
    assert_eq!(
        summary["fixtures"][0]["stitch_normalization_contract_missing_fields"],
        serde_json::json!([
            "expectations.seam_exposure_model=known_model",
            "expectations.seam_exposure_held_out_validation_passed={true|false}",
            "expectations.seam_exposure_offset_normalized_abs_max<=0.05",
            "expectations.seam_blend_required=true",
            "expectations.seam_blend_mode=seam_aware_multiband",
            "expectations.seam_blend_review_required=false",
            "expectations.seam_detail_review_required=false",
            "expectations.seam_detail_supported_scale_count_min>=2",
            "expectations.seam_detail_max_symmetric_energy_ratio_max<=2.00",
            "expectations.seam_gradient_ratio_max<=1.05",
            "expectations.seam_overlap_p95_abs_difference_max<1"
        ])
    );
    assert_eq!(
        summary["action_items"],
        serde_json::json!([
            "add_complete_stitch_normalization_contracts_for_validation_ready_fixtures:1",
            "complete_stitch_normalization_contract_for_fixture:logan"
        ])
    );
    let repair_detail = summary["fixtures"][0]["repair_plan"][0]["details"]
        .as_str()
        .unwrap();
    assert!(repair_detail.contains("held-out validation"));
    assert!(repair_detail.contains("seam-aware multiband"));
    assert!(repair_detail.contains("supported detail scales"));
    assert!(repair_detail.contains("overlap p95"));

    let markdown = std::fs::read_to_string(&summary_md).unwrap();
    assert!(markdown.contains("Stitch normalization"));
    assert!(markdown.contains("contract=incomplete"));
    assert!(markdown.contains("complete_stitch_normalization_contract_for_fixture:logan"));
}

#[test]
fn test_validate_cli_fixture_coverage_requires_complete_geometry_and_negative_contracts() {
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
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_geometry_preparation_contract_fixtures": 1,
                "min_geometry_accuracy_contract_fixtures": 1,
                "min_orientation_accuracy_contract_fixtures": 1,
                "min_negative_reconstruction_contract_fixtures": 1
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path2,
                    "input_mode": "negative",
                    "summary_baseline": baseline_path,
                    "film_stock": "Synthetic negative",
                    "scene_tags": ["bordered-negative"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "measured-response",
                    "expectations": {
                        "deskew_all_components_applied": true,
                        "deskew_correction_degrees_expected": -0.5,
                        "orientation_components_expected": [{
                            "component_index": 1,
                            "upright_approved": false,
                            "decoded_pixel_sha256": "4".repeat(64),
                            "tag_present": true,
                            "tag_value": 6,
                            "transform": "identity",
                            "applied": false,
                            "source_width": 4,
                            "source_height": 3,
                            "output_width": 4,
                            "output_height": 3
                        }],
                        "density_inversion_skipped": false
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
        .expect("run incomplete geometry/negative fixture coverage");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("fixture_coverage_min_geometry_preparation_contract_fixtures_not_met"));
    assert!(stderr.contains("fixture_coverage_min_geometry_accuracy_contract_fixtures_not_met"));
    assert!(stderr.contains("fixture_coverage_min_orientation_accuracy_contract_fixtures_not_met"));
    assert!(
        stderr.contains("fixture_coverage_min_negative_reconstruction_contract_fixtures_not_met")
    );

    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["validation_ready_fixture_count"], 1);
    assert_eq!(summary["geometry_preparation_contract_fixture_count"], 0);
    assert_eq!(summary["geometry_accuracy_contract_fixture_count"], 0);
    assert_eq!(summary["orientation_accuracy_contract_fixture_count"], 0);
    assert_eq!(summary["negative_reconstruction_contract_fixture_count"], 0);
    let geometry_missing = summary["fixtures"][0]["geometry_preparation_contract_missing_fields"]
        .as_array()
        .unwrap();
    assert!(geometry_missing
        .iter()
        .any(|field| field == "expectations.border_crop_all_components_cropped=true"));
    assert!(geometry_missing.iter().any(|field| {
        field == "expectations.deskew_minimum_component_retained_area_ratio_min>=0.90"
    }));
    let accuracy_missing = summary["fixtures"][0]["geometry_accuracy_contract_missing_fields"]
        .as_array()
        .unwrap();
    assert!(accuracy_missing
        .iter()
        .any(|field| { field == "expectations.deskew_correction_tolerance_degrees<=0.05" }));
    assert!(accuracy_missing
        .iter()
        .any(|field| field == "expectations.border_crop_components_expected=all_2_components"));
    let orientation_missing = summary["fixtures"][0]
        ["orientation_accuracy_contract_missing_fields"]
        .as_array()
        .unwrap();
    assert!(orientation_missing
        .iter()
        .any(|field| field == "expectations.orientation_components_expected=all_2_components"));
    assert!(orientation_missing.iter().any(|field| {
        field == "expectations.orientation_components_expected.component_1.upright_approved=true"
    }));
    assert!(orientation_missing.iter().any(|field| {
        field
            == "expectations.orientation_components_expected.component_1.transform=rotate_90_clockwise"
    }));
    assert!(orientation_missing.iter().any(|field| {
        field == "expectations.orientation_components_expected.component_1.applied=true"
    }));
    assert!(orientation_missing.iter().any(|field| {
        field == "expectations.orientation_components_expected.component_1.output_dimensions=3x4"
    }));
    let negative_missing = summary["fixtures"][0]
        ["negative_reconstruction_contract_missing_fields"]
        .as_array()
        .unwrap();
    assert!(negative_missing.iter().any(|field| {
        field == "expectations.negative_response_model=measured_nonlinear_dye_separation"
    }));
    assert!(negative_missing.iter().any(|field| {
        field == "expectations.negative_response_curve_extrapolated_ratio_max<=0.01"
    }));
    assert!(summary["action_items"]
        .as_array()
        .unwrap()
        .iter()
        .any(|action| action == "complete_geometry_preparation_contract_for_fixture:logan"));
    assert!(summary["action_items"]
        .as_array()
        .unwrap()
        .iter()
        .any(|action| action == "complete_geometry_accuracy_contract_for_fixture:logan"));
    assert!(summary["action_items"]
        .as_array()
        .unwrap()
        .iter()
        .any(|action| action == "complete_orientation_accuracy_contract_for_fixture:logan"));
    assert!(summary["action_items"]
        .as_array()
        .unwrap()
        .iter()
        .any(|action| action == "complete_negative_reconstruction_contract_for_fixture:logan"));
    let markdown = std::fs::read_to_string(&summary_md).unwrap();
    assert!(markdown.contains("Geometry preparation / orientation"));
    assert!(markdown.contains("accuracy=incomplete"));
    assert!(markdown.contains("orientation=incomplete"));
    assert!(markdown.contains("Negative reconstruction"));
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
                "min_render_dynamic_range_contract_fixtures": 1,
                "min_stitch_normalization_contract_fixtures": 1,
                "min_grain_reduction_enabled_fixtures": 1,
                "min_grain_detail_contract_fixtures": 1,
                "min_grain_reduction_effect_contract_fixtures": 1,
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
    assert!(stderr.contains("fixture_coverage_min_render_dynamic_range_contract_fixtures_not_met"));
    assert!(stderr.contains("fixture_coverage_min_stitch_normalization_contract_fixtures_not_met"));
    assert!(stderr.contains("fixture_coverage_min_grain_reduction_enabled_fixtures_not_met"));
    assert!(stderr.contains("fixture_coverage_min_grain_detail_contract_fixtures_not_met"));
    assert!(
        stderr.contains("fixture_coverage_min_grain_reduction_effect_contract_fixtures_not_met")
    );
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
    assert_eq!(summary["render_dynamic_range_contract_fixture_count"], 0);
    assert_eq!(summary["stitch_normalization_contract_fixture_count"], 0);
    assert_eq!(summary["grain_reduction_enabled_fixture_count"], 0);
    assert_eq!(summary["grain_detail_contract_fixture_count"], 0);
    assert_eq!(summary["grain_reduction_effect_contract_fixture_count"], 0);
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
            "add_complete_render_dynamic_range_contracts_for_validation_ready_fixtures:1",
            "add_complete_stitch_normalization_contracts_for_validation_ready_fixtures:1",
            "add_validation_ready_grain_reduction_enabled_fixtures:1",
            "add_complete_grain_detail_contracts_for_validation_ready_fixtures:1",
            "add_complete_grain_reduction_effect_contracts_for_validation_ready_fixtures:1",
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
    assert!(markdown.contains("min grain-reduction-enabled fixtures"));
    assert!(markdown.contains("min complete grain-detail-contract fixtures"));
    assert!(markdown.contains("min complete grain-reduction-effect-contract fixtures"));
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
    let right = synthetic::constant_image(7, 4, [12000, 7000, 3000]);
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
    assert_eq!(
        summary["fixtures"][0]["tiff_pair"]["dimensions_compatible"],
        false
    );
    assert_eq!(summary["fixtures"][0]["tiff_pair"]["height_delta"], 3);
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
fn test_validate_cli_fixture_coverage_accepts_one_row_stitch_compatible_pair() {
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
                "min_uncalibrated_fixtures": 1
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
        .output()
        .expect("run scanstitch-validate fixture coverage with one-row TIFF delta");

    assert!(
        output.status.success(),
        "one-row TIFF pair should be accepted as stitch-compatible: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["status"], "passed");
    assert_eq!(summary["tiff_dimension_matched_pair_count"], 1);
    assert_eq!(summary["validation_ready_fixture_count"], 1);
    assert_eq!(summary["uncalibrated_fixture_count"], 1);
    assert_eq!(
        summary["fixtures"][0]["tiff_pair"]["dimensions_match"],
        false
    );
    assert_eq!(
        summary["fixtures"][0]["tiff_pair"]["dimensions_compatible"],
        true
    );
    assert_eq!(
        summary["fixtures"][0]["tiff_pair"]["dimension_matched"],
        true
    );
    assert_eq!(summary["fixtures"][0]["tiff_pair"]["width_delta"], 0);
    assert_eq!(summary["fixtures"][0]["tiff_pair"]["height_delta"], 1);
    assert!(summary["issues"].as_array().unwrap().is_empty());
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
fn test_validate_cli_roll_inventory_writes_fixture_registry_scaffold() {
    let tmp = tempfile::TempDir::new().unwrap();
    let roll_dir = tmp.path().join("TESTROLL");
    std::fs::create_dir_all(&roll_dir).unwrap();
    let frame0 = roll_dir.join("RAW_0000.tif");
    let frame1 = roll_dir.join("RAW_0001.tif");
    write_rgba8_tiff(&frame0, 4, 3);
    write_rgba8_tiff(&frame1, 4, 3);

    let summary_json = tmp.path().join("roll-inventory.json");
    let registry_path = tmp.path().join("roll-fixtures.json");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-inventory")
        .arg("--quiet")
        .arg("--bit-depth")
        .arg("14")
        .arg("--grain-reduction")
        .arg("on")
        .arg("--grain-strength")
        .arg("0.61")
        .arg("--grain-scale")
        .arg("1.8")
        .arg("--roll-suite-frame")
        .arg("RAW_0000")
        .arg("--film-stock")
        .arg("Kodak Gold 200")
        .arg("--roll-fixture-scene-tag")
        .arg("outdoor,skin")
        .arg("--roll-fixture-exposure-tag")
        .arg("normal-exposure")
        .arg("--write-roll-fixture-registry")
        .arg(&registry_path)
        .arg("--summary-json")
        .arg(&summary_json)
        .output()
        .expect("run scanstitch-validate roll inventory registry scaffold writer");

    assert!(
        output.status.success(),
        "roll inventory registry writer failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let registry: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&registry_path).unwrap()).unwrap();
    let fixtures = registry["fixtures"].as_object().unwrap();
    assert_eq!(fixtures.len(), 1);
    let fixture = &registry["fixtures"]["testroll-raw-0000"];
    assert_eq!(fixture["component1"], frame0.display().to_string());
    assert!(
        fixture.get("component2").is_none(),
        "single-frame scaffolds must not invent a duplicate second component"
    );
    assert_eq!(fixture["input_mode"], "negative");
    assert_eq!(fixture["bit_depth"], 14);
    assert_eq!(fixture["grain_reduction"], "on");
    assert_eq!(fixture["grain_strength"], 0.61);
    assert_eq!(fixture["grain_scale"], 1.8);
    assert_eq!(fixture["deskew"], "auto");
    assert!(fixture.get("deskew_angle_degrees").is_none());
    assert_eq!(fixture["force_no_stitch"], true);
    assert_eq!(
        fixture["summary_baseline"],
        std::path::PathBuf::from("local-fixtures")
            .join("baselines")
            .join("testroll-raw-0000-summary-baseline.json")
            .display()
            .to_string()
    );
    assert_eq!(
        fixture["expectations"]["stitch_decision"],
        "skipped_single_input"
    );
    assert_eq!(
        fixture["expectations"]["output_color_space"],
        "linear_prophoto_rgb_d50"
    );
    assert_eq!(fixture["calibration_case"], "uncalibrated-image-derived");
    assert_eq!(fixture["film_stock"], "Kodak Gold 200");
    assert_eq!(fixture["scene_tags"][0], "outdoor");
    assert_eq!(fixture["scene_tags"][1], "skin");
    assert_eq!(fixture["exposure_tags"][0], "normal-exposure");

    let list_output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--list-fixtures")
        .output()
        .expect("load generated roll fixture registry");
    assert!(
        list_output.status.success(),
        "generated registry should load: status={} stderr={} stdout={}",
        list_output.status,
        String::from_utf8_lossy(&list_output.stderr),
        String::from_utf8_lossy(&list_output.stdout)
    );
}

#[test]
fn test_validate_cli_roll_inventory_writes_fixture_metadata_template() {
    let tmp = tempfile::TempDir::new().unwrap();
    let roll_dir = tmp.path().join("TESTROLL");
    std::fs::create_dir_all(&roll_dir).unwrap();
    write_rgba8_tiff(&roll_dir.join("RAW_0000.tif"), 4, 3);
    write_rgba8_tiff(&roll_dir.join("RAW_0001.tif"), 4, 3);

    let template_path = tmp.path().join("roll-fixture-metadata-template.json");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-inventory")
        .arg("--quiet")
        .arg("--roll-suite-frame")
        .arg("RAW_0000")
        .arg("--film-stock")
        .arg("Kodak Gold 200")
        .arg("--roll-fixture-scene-tag")
        .arg("outdoor")
        .arg("--roll-fixture-exposure-tag")
        .arg("normal-exposure")
        .arg("--write-roll-fixture-metadata-template")
        .arg(&template_path)
        .output()
        .expect("run scanstitch-validate roll fixture metadata template writer");

    assert!(
        output.status.success(),
        "roll metadata template writer failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let template: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&template_path).unwrap()).unwrap();
    let frames = template["frames"].as_object().unwrap();
    assert_eq!(frames.len(), 1);
    let frame = &template["frames"]["RAW_0000"];
    assert_eq!(frame["film_stock"], "Kodak Gold 200");
    assert_eq!(frame["scene_tags"], serde_json::json!(["outdoor"]));
    assert_eq!(
        frame["exposure_tags"],
        serde_json::json!(["normal-exposure"])
    );
    assert_eq!(frame["calibration_case"], "uncalibrated-image-derived");
    assert!(frame["description"]
        .as_str()
        .is_some_and(|description| description.contains("RAW_0000.tif")));
    assert!(template.get("coverage_requirements").is_none());
}

#[test]
fn test_validate_cli_roll_inventory_writes_contact_sheet_and_index() {
    let tmp = tempfile::TempDir::new().unwrap();
    let roll_dir = tmp.path().join("TESTROLL");
    std::fs::create_dir_all(&roll_dir).unwrap();
    write_rgba8_tiff(&roll_dir.join("RAW_0000.tif"), 4, 3);
    write_rgba8_tiff(&roll_dir.join("RAW_0001.tif"), 4, 3);

    let sheet_path = tmp.path().join("roll-contact-sheet.png");
    let index_path = tmp.path().join("roll-contact-sheet.json");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-inventory")
        .arg("--quiet")
        .arg("--roll-suite-frame")
        .arg("RAW_0000")
        .arg("--write-roll-contact-sheet")
        .arg(&sheet_path)
        .arg("--write-roll-contact-sheet-index")
        .arg(&index_path)
        .output()
        .expect("run scanstitch-validate roll contact sheet writer");

    assert!(
        output.status.success(),
        "roll contact sheet writer failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    assert_eq!(image::image_dimensions(&sheet_path).unwrap(), (196, 136));
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    assert_eq!(index["contact_sheet"], sheet_path.display().to_string());
    assert_eq!(index["roll_name"], "TESTROLL");
    assert_eq!(index["input_mode"], "negative");
    assert_eq!(index["bit_depth"], serde_json::json!(14));
    assert_eq!(index["columns"], serde_json::json!(1));
    assert_eq!(index["rows"], serde_json::json!(1));
    assert_eq!(index["thumb_width"], serde_json::json!(180));
    assert_eq!(index["thumb_height"], serde_json::json!(120));

    let frames = index["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 1);
    let frame = &frames[0];
    assert_eq!(frame["name"], "RAW_0000.tif");
    assert_eq!(frame["stem"], "RAW_0000");
    assert_eq!(frame["row"], serde_json::json!(0));
    assert_eq!(frame["column"], serde_json::json!(0));
    assert_eq!(frame["tile_x"], serde_json::json!(8));
    assert_eq!(frame["tile_y"], serde_json::json!(8));
    assert_eq!(frame["source_width"], serde_json::json!(4));
    assert_eq!(frame["source_height"], serde_json::json!(3));
    assert_eq!(
        frame["preview_transform"],
        "per_channel_stretch_inverted_gamma"
    );
}

#[test]
fn test_validate_cli_roll_fixture_metadata_template_refresh_preserves_curated_entries() {
    let tmp = tempfile::TempDir::new().unwrap();
    let roll_dir = tmp.path().join("TESTROLL");
    std::fs::create_dir_all(&roll_dir).unwrap();
    write_rgba8_tiff(&roll_dir.join("RAW_0000.tif"), 4, 3);
    write_rgba8_tiff(&roll_dir.join("RAW_0001.tif"), 4, 3);

    let existing_metadata = tmp.path().join("existing-roll-fixture-metadata.json");
    std::fs::write(
        &existing_metadata,
        serde_json::json!({
            "coverage_requirements": {
                "required_scene_tags": ["skin-tone"]
            },
            "frames": {
                "RAW_0000": {
                    "film_stock": "Kodak Portra 400",
                    "scene_tags": ["skin-tone"],
                    "exposure_tags": ["normal-exposure"],
                    "reference_evidence": ["gray-card"],
                    "calibration_case": "uncalibrated-image-derived",
                    "description": "Curated frame metadata from local notes"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let refreshed_template = tmp.path().join("refreshed-roll-fixture-metadata.json");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-inventory")
        .arg("--quiet")
        .arg("--roll-fixture-metadata")
        .arg(&existing_metadata)
        .arg("--write-roll-fixture-metadata-template")
        .arg(&refreshed_template)
        .output()
        .expect("refresh roll fixture metadata template");

    assert!(
        output.status.success(),
        "roll metadata template refresh failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let template: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&refreshed_template).unwrap()).unwrap();
    assert_eq!(
        template["coverage_requirements"]["required_scene_tags"],
        serde_json::json!(["skin-tone"])
    );
    let frames = template["frames"].as_object().unwrap();
    assert_eq!(frames.len(), 2);
    assert_eq!(
        template["frames"]["RAW_0000"]["film_stock"],
        "Kodak Portra 400"
    );
    assert_eq!(
        template["frames"]["RAW_0000"]["reference_evidence"],
        serde_json::json!(["gray-card"])
    );
    assert_eq!(
        template["frames"]["RAW_0000"]["description"],
        "Curated frame metadata from local notes"
    );
    assert_eq!(
        template["frames"]["RAW_0001"]["calibration_case"],
        "uncalibrated-image-derived"
    );
    assert!(template["frames"]["RAW_0001"]["description"]
        .as_str()
        .is_some_and(|description| description.contains("TODO")));
}

#[test]
fn test_validate_cli_roll_inventory_applies_fixture_metadata_sidecar() {
    let tmp = tempfile::TempDir::new().unwrap();
    let roll_dir = tmp.path().join("TESTROLL");
    std::fs::create_dir_all(&roll_dir).unwrap();
    let frame0 = roll_dir.join("RAW_0000.tif");
    let frame1 = roll_dir.join("RAW_0001.tif");
    write_rgba8_tiff(&frame0, 4, 3);
    write_rgba8_tiff(&frame1, 4, 3);

    let metadata_path = tmp.path().join("roll-fixture-metadata.json");
    std::fs::write(
        &metadata_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "required_scene_tags": ["skin-tone"],
                "required_exposure_tags": ["normal-exposure"],
                "required_scene_exposure_pairs": ["skin-tone|normal-exposure"],
                "required_debug_artifact_kinds": ["candidate_comparison"]
            },
            "frames": {
                "RAW_0000": {
                    "film_stock": "Kodak Portra 400",
                    "scene_tags": ["skin-tone"],
                    "exposure_tags": ["normal-exposure"],
                    "reference_evidence": ["gray-card"],
                    "calibration_case": "uncalibrated-image-derived",
                    "expectations": {
                        "candidate_risk": "safe",
                        "debug_artifacts_required": true,
                        "debug_artifact_kinds_required": ["candidate_comparison"]
                    },
                    "description": "Curated frame metadata from local notes"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let summary_json = tmp.path().join("roll-inventory.json");
    let registry_path = tmp.path().join("roll-fixtures.json");
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-inventory")
        .arg("--quiet")
        .arg("--bit-depth")
        .arg("14")
        .arg("--roll-suite-frame")
        .arg("RAW_0000")
        .arg("--roll-fixture-scene-tag")
        .arg("global-scene")
        .arg("--roll-fixture-exposure-tag")
        .arg("global-exposure")
        .arg("--roll-fixture-metadata")
        .arg(&metadata_path)
        .arg("--write-roll-fixture-registry")
        .arg(&registry_path)
        .arg("--summary-json")
        .arg(&summary_json)
        .output()
        .expect("run scanstitch-validate roll metadata registry scaffold writer");

    assert!(
        output.status.success(),
        "roll metadata registry writer failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let registry: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&registry_path).unwrap()).unwrap();
    let fixture = &registry["fixtures"]["testroll-raw-0000"];
    assert_eq!(fixture["film_stock"], "Kodak Portra 400");
    assert_eq!(fixture["scene_tags"], serde_json::json!(["skin-tone"]));
    assert_eq!(
        fixture["exposure_tags"],
        serde_json::json!(["normal-exposure"])
    );
    assert_eq!(
        fixture["reference_evidence"],
        serde_json::json!(["gray-card"])
    );
    assert_eq!(
        fixture["expectations"]["stitch_decision"],
        "skipped_single_input"
    );
    assert!(fixture.get("component2").is_none());
    assert_eq!(fixture["expectations"]["candidate_risk"], "safe");
    assert_eq!(fixture["expectations"]["debug_artifacts_required"], true);
    assert_eq!(
        fixture["expectations"]["debug_artifact_kinds_required"],
        serde_json::json!(["candidate_comparison"])
    );
    assert_eq!(
        fixture["description"],
        "Curated frame metadata from local notes"
    );
    assert_eq!(registry["coverage_requirements"]["min_fixtures"], 1);
    assert_eq!(
        registry["coverage_requirements"]["required_scene_exposure_pairs"],
        serde_json::json!(["skin-tone|normal-exposure"])
    );
}

#[test]
fn test_validate_cli_roll_fixture_metadata_rejects_unmatched_frame_key() {
    let tmp = tempfile::TempDir::new().unwrap();
    let roll_dir = tmp.path().join("TESTROLL");
    std::fs::create_dir_all(&roll_dir).unwrap();
    write_rgba8_tiff(&roll_dir.join("RAW_0000.tif"), 4, 3);

    let metadata_path = tmp.path().join("roll-fixture-metadata.json");
    std::fs::write(
        &metadata_path,
        serde_json::json!({
            "frames": {
                "RAW_9999": {
                    "scene_tags": ["skin-tone"],
                    "exposure_tags": ["normal-exposure"]
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-inventory")
        .arg("--quiet")
        .arg("--roll-fixture-metadata")
        .arg(&metadata_path)
        .arg("--write-roll-fixture-registry")
        .arg(tmp.path().join("roll-fixtures.json"))
        .output()
        .expect("run scanstitch-validate roll metadata with unmatched frame key");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--roll-fixture-metadata contains frame key(s)")
            && stderr.contains("RAW_9999"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn test_validate_cli_roll_fixture_registry_rejects_incomplete_calibration_wiring() {
    let tmp = tempfile::TempDir::new().unwrap();
    let roll_dir = tmp.path().join("TESTROLL");
    std::fs::create_dir_all(&roll_dir).unwrap();
    write_rgba8_tiff(&roll_dir.join("RAW_0000.tif"), 4, 3);

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-inventory")
        .arg("--quiet")
        .arg("--scanner-profile")
        .arg("scanner-a")
        .arg("--write-roll-fixture-registry")
        .arg(tmp.path().join("roll-fixtures.json"))
        .output()
        .expect("run scanstitch-validate invalid roll fixture registry scaffold writer");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--calibration-library"));
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
        .arg("--grain-reduction")
        .arg("on")
        .arg("--grain-strength")
        .arg("0.6")
        .arg("--grain-scale")
        .arg("1.5")
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
    assert_eq!(summary["review"]["tone_output_unknown_count"], 0);
    assert!(
        summary["review"]["render_review_status_counts"]
            .as_array()
            .is_some_and(|counts| !counts.is_empty()),
        "roll suite should aggregate the authoritative final render decision"
    );
    assert!(
        summary["review"]["tone_output_confidence_status_counts"]
            .as_array()
            .is_some_and(|counts| !counts.is_empty()),
        "roll suite should aggregate rendered-tone confidence states"
    );
    assert!(
        summary["review"]["tone_output_review_reason_counts"]
            .as_array()
            .is_some_and(|counts| !counts.is_empty()),
        "roll suite should aggregate rendered-tone decision reasons"
    );
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
    assert_eq!(
        summary["quality"]["grain_detail_evaluated_count"], 4,
        "every grain-on frame should contribute an evaluated detail decision"
    );
    assert_eq!(summary["quality"]["grain_detail_review_required_count"], 0);
    assert!(
        summary["quality"]["grain_detail_decision_supported_count"]
            .as_u64()
            .is_some_and(|count| count > 0),
        "the detail-bearing roll must provide supported grain evidence: {}",
        summary["quality"]
    );
    assert!(
        summary["quality"]["grain_detail_luminance_supported_count"]
            .as_u64()
            .is_some_and(|count| count > 0)
            && summary["quality"]["grain_detail_chroma_supported_count"]
                .as_u64()
                .is_some_and(|count| count > 0),
        "the roll must independently support luminance and opponent-colour retention: {}",
        summary["quality"]
    );
    assert!(
        summary["quality"]["grain_detail_luminance_p10_retention_min"]
            .as_f64()
            .is_some_and(|retention| retention >= 0.70)
            && summary["quality"]["grain_detail_chroma_p10_retention_min"]
                .as_f64()
                .is_some_and(|retention| retention >= 0.70),
        "supported roll detail must clear the automatic p10 retention floor: {}",
        summary["quality"]
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
                    && frame["tone_output_evidence_evaluated"].as_bool().is_some()
                    && frame["tone_output_evidence_confidence"].as_f64().is_some()
                    && frame["tone_output_confidence_status"].as_str().is_some()
                    && frame["tone_output_review_required"].as_bool().is_some()
                    && frame["tone_output_review_reason"].as_str().is_some()
                    && frame
                        .get("confidence_limited_by_tone_output_evidence")
                        .and_then(serde_json::Value::as_bool)
                        .is_some()
                    && frame["input_luminance_range_p05_p95"].as_f64().is_some()
                    && frame["mapped_luminance_range_p05_p95"].as_f64().is_some()
                    && frame
                        .get("render_to_mapped_luminance_range_ratio")
                        .is_some()
                    && frame["maximum_post_tone_high_clip_ratio"]
                        .as_f64()
                        .is_some()
                    && frame["maximum_post_tone_low_clip_ratio"].as_f64().is_some()
                    && frame.get("shadow_saturation_p95").is_some()
                    && frame.get("midtone_neutral_pixel_count").is_some()
                    && frame.get("midtone_neutral_saturation_p95").is_some()
                    && frame.get("bright_neutral_saturation_p95").is_some()
                    && frame.get("shadow_rgb_median").is_some()
                    && frame.get("midtone_rgb_median").is_some()
                    && frame.get("midtone_neutral_rgb_median").is_some()
                    && frame.get("bright_neutral_rgb_median").is_some()
                    && frame.get("shadow_rgb_balance_delta").is_some()
                    && frame.get("midtone_rgb_balance_delta").is_some()
                    && frame.get("midtone_neutral_rgb_balance_delta").is_some()
                    && frame.get("bright_neutral_rgb_balance_delta").is_some()
                    && frame.get("noise_reduction_applied_ratio").is_some()
                    && frame["grain_detail_evaluated"] == true
                    && frame["grain_detail_review_required"] == false
                    && frame.get("grain_detail_decision_supported").is_some()
                    && frame.get("grain_detail_luminance_probe_count").is_some()
                    && frame.get("grain_detail_chroma_probe_count").is_some()
                    && frame.get("grain_detail_luminance_p10_retention").is_some()
                    && frame.get("grain_detail_chroma_p10_retention").is_some()
                    && frame.get("colorspace_post_scale_preserved_ratio").is_some()
            ),
        "each roll-suite frame should expose quality diagnostics for roll-level audit: {}",
        summary["frames"]
    );
    for frame in summary["frames"].as_array().unwrap() {
        let mapped_range = frame["mapped_luminance_range_p05_p95"]
            .as_f64()
            .expect("mapped luminance range");
        if mapped_range > 0.0 {
            assert!(
                frame["render_to_mapped_luminance_range_ratio"]
                    .as_f64()
                    .is_some(),
                "a measurable fitted range must retain a numeric render:mapped ratio: {frame}"
            );
        } else {
            assert!(frame["render_to_mapped_luminance_range_ratio"].is_null());
            assert_eq!(frame["tone_output_review_required"], true);
            assert_eq!(
                frame["tone_output_confidence_status"],
                "review_required_insufficient_scene_tonal_range"
            );
        }
    }
    let summary_md = std::fs::read_to_string(&summary_md).unwrap();
    assert!(summary_md.contains("Roll Base Clusters"));
    assert!(summary_md.contains("Roll Review Audit"));
    assert!(summary_md.contains("Final render reviewable/not reviewable/unknown"));
    assert!(summary_md.contains("Tone output evaluated/review required/unknown"));
    assert!(summary_md.contains("Frame Tone Output Evidence"));
    assert!(summary_md.contains("Render:Mapped"));
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
    assert!(summary_md.contains("Bright Neutral RGB"));
    assert!(summary_md.contains("Frame Quality Diagnostics"));
    assert!(summary_md.contains("Grain detail evaluated/supported/review frames"));
    assert!(summary_md.contains("Frame Grain Detail Diagnostics"));

    let subset_summary_json = tmp.path().join("roll-suite-subset.json");
    let subset_summary_md = tmp.path().join("roll-suite-subset.md");
    let subset_output_dir = tmp.path().join("roll-output-subset");
    let subset_output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--roll-dir")
        .arg(&roll_dir)
        .arg("--roll-suite")
        .arg("--roll-suite-frame")
        .arg("RAW_0001.tif,raw-0003")
        .arg("--grain-reduction")
        .arg("on")
        .arg("--grain-strength")
        .arg("0.6")
        .arg("--grain-scale")
        .arg("1.5")
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
        .arg("--grain-reduction")
        .arg("on")
        .arg("--grain-strength")
        .arg("0.6")
        .arg("--grain-scale")
        .arg("1.5")
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
    for frame in legacy_summary["frames"].as_array_mut().unwrap() {
        let frame = frame.as_object_mut().unwrap();
        for field in [
            "tone_output_evidence_evaluated",
            "tone_output_evidence_confidence",
            "tone_output_confidence_status",
            "tone_output_review_required",
            "tone_output_review_reason",
            "confidence_limited_by_tone_output_evidence",
            "input_luminance_range_p05_p95",
            "mapped_luminance_range_p05_p95",
            "render_to_mapped_luminance_range_ratio",
            "maximum_post_tone_high_clip_ratio",
            "maximum_post_tone_low_clip_ratio",
        ] {
            frame.remove(field);
        }
    }
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
        .arg("--grain-reduction")
        .arg("on")
        .arg("--grain-strength")
        .arg("0.6")
        .arg("--grain-scale")
        .arg("1.5")
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
        compared["comparison"]["quality"]["grain_detail_decision_supported_count_delta"],
        0
    );
    assert_eq!(
        compared["comparison"]["quality"]["grain_detail_luminance_p10_retention_min_delta"]
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
                    .get("grain_detail_luminance_p10_retention_delta")
                    .is_some()
                && frame
                    .get("grain_detail_chroma_p10_retention_delta")
                    .is_some()
                && frame
                    .get("colorspace_post_scale_preserved_ratio_delta")
                    .is_some()
                && frame.get("render_review_status_changed").is_some()
                && frame.get("render_reviewable_changed").is_some()
                && frame.get("tone_output_confidence_status_changed").is_some()
                && frame.get("tone_output_review_required_changed").is_some()
                && frame.get("tone_output_evidence_confidence_delta").is_some()
                && frame
                    .get("tone_output_render_to_mapped_luminance_range_ratio_delta")
                    .is_some()),
        "roll comparison should expose per-frame quality, gamut, and tone-decision evidence"
    );
    let compare_md = std::fs::read_to_string(&compare_summary_md).unwrap();
    assert!(compare_md.contains("Roll Suite Comparison"));
    assert!(compare_md.contains("Frame Tone Output Comparison"));
    assert!(compare_md.contains("Render:Mapped d"));
    assert!(compare_md.contains("grain luma p10 retention min"));
    assert!(compare_md.contains("Grain Support (all/L/C)"));
    assert!(compare_md.contains("render luminance range mean"));
    assert!(compare_md.contains("Render Range d"));
    assert!(compare_md.contains("midtone p50 range"));
    assert!(compare_md.contains("chroma residual p95 mean"));
    assert!(compare_md.contains("Clip High d"));
}

#[test]
fn test_validate_cli_roll_suite_positive_skips_negative_base_but_requires_profiled_color() {
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
    assert_eq!(summary["status"], "review_required");
    assert_eq!(summary["roll_base_color"], serde_json::Value::Null);
    assert_eq!(summary["roll_base_source"], serde_json::Value::Null);
    assert_eq!(summary["roll_base_frame_count"], 0);
    assert_eq!(summary["frame_count"], 3);
    assert_eq!(summary["frames"].as_array().unwrap().len(), 3);
    assert_eq!(summary["review"]["candidate_safe_count"], 0);
    assert_eq!(summary["review"]["candidate_review_required_count"], 3);
    assert_eq!(summary["review"]["tone_color_trusted_count"], 0);
    assert_eq!(summary["review"]["tone_color_review_required_count"], 3);
    assert_eq!(summary["review"]["render_not_reviewable_count"], 3);
    assert_eq!(summary["review"]["tone_output_review_required_count"], 0);
    assert_eq!(
        summary["review"]["issue_counts"].as_array().unwrap().len(),
        3
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
    ] {
        assert!(
            issues.iter().all(|issue| !issue.contains(forbidden)),
            "positive roll suite emitted negative precondition issue `{forbidden}`: {issues:?}"
        );
    }
    assert_eq!(
        issues
            .iter()
            .filter(|issue| issue.contains("candidate_risk_review_required"))
            .count(),
        3
    );
    assert_eq!(
        issues
            .iter()
            .filter(|issue| issue.contains("tone_color_trust_review_required"))
            .count(),
        3
    );
    assert_eq!(
        issues
            .iter()
            .filter(|issue| issue.contains("render_review_not_supported"))
            .count(),
        3
    );

    for frame in summary["frames"].as_array().unwrap() {
        assert!(Path::new(frame["output_path"].as_str().unwrap()).exists());
        let report_path = Path::new(frame["report_path"].as_str().unwrap());
        assert!(report_path.exists());
        assert_eq!(frame["render_input_source"], "positive_scan_rgb");
        assert_eq!(
            frame["render_input_reason"],
            "already-positive input normalized without density inversion"
        );
        assert_eq!(frame["candidate_risk"], "review_unprofiled_input");
        assert_eq!(frame["tone_color_trust_state"], "review_required");
        assert_eq!(frame["render_reviewable"], false);
        assert_eq!(frame["tone_output_review_required"], false);

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
    for required in [
        "positive_input_negative_like",
        "candidate_risk_review_required",
        "tone_color_trust_review_required",
    ] {
        assert!(
            issues.iter().any(|issue| issue.contains(required)),
            "missing `{required}` roll-suite issue: {issues:?}"
        );
    }
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
fn test_validate_cli_require_reviewable_retains_non_reviewable_summary_and_artifacts() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input = tmp.path().join("unprofiled-positive.tiff");
    let image = textured_positive_panorama(48, 64);
    write_rgb16_tiff_with_orientation(&input, &image, 1);
    let output_dir = tmp.path().join("output");
    let summary_json = tmp.path().join("summary.json");
    let summary_md = tmp.path().join("summary.md");

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--component1")
        .arg(&input)
        .arg("--fixture")
        .arg("unprofiled-positive")
        .arg("--input-mode")
        .arg("positive")
        .arg("--bit-depth")
        .arg("14")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--force-no-stitch")
        .arg("--deskew")
        .arg("off")
        .arg("--technical-white-balance")
        .arg("off")
        .arg("--require-reviewable")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--summary-json")
        .arg(&summary_json)
        .arg("--summary-md")
        .arg(&summary_md)
        .output()
        .expect("run gated validation render");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--require-reviewable rejected validation output")
            && stderr.contains("report and validation summary artifacts were retained"),
        "unexpected stderr: {stderr}"
    );
    assert!(output_dir.join("output.tiff").is_file());
    assert!(output_dir.join("report.json").is_file());
    assert!(summary_json.is_file());
    assert!(summary_md.is_file());
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(summary_json).unwrap()).unwrap();
    assert_eq!(summary["render"]["render_reviewable"], false);
    assert_ne!(summary["render"]["render_review_status"], "reviewable");

    let tracked_baseline_path = tmp.path().join("tracked-baseline.json");
    let compact_summary: scanstitch::validation::ValidationSummary =
        serde_json::from_value(summary).unwrap();
    std::fs::write(
        &tracked_baseline_path,
        serde_json::to_string_pretty(&scanstitch::validation::tracked_baseline_from_summary(
            &compact_summary,
        ))
        .unwrap(),
    )
    .unwrap();

    let registry_path = tmp.path().join("fixtures.json");
    let mut registry = serde_json::json!({
        "coverage_requirements": {
            "min_fixtures": 1,
            "min_summary_baselines": 1,
            "min_uncalibrated_fixtures": 1,
            "min_unique_film_stocks": 1,
            "min_scene_tags": 1,
            "min_exposure_tags": 1,
            "min_calibration_cases": 1
        },
        "fixtures": {
            "unprofiled-positive": {
                "component1": input,
                "input_mode": "positive",
                "bit_depth": 14,
                "deskew": "off",
                "force_no_stitch": true,
                "summary_baseline": tracked_baseline_path,
                "film_stock": "Synthetic positive",
                "scene_tags": ["unprofiled-positive"],
                "exposure_tags": ["normal-exposure"],
                "calibration_case": "uncalibrated-image-derived"
            }
        }
    });
    std::fs::write(
        &registry_path,
        serde_json::to_string_pretty(&registry).unwrap(),
    )
    .unwrap();
    let undeclared_suite_json = tmp.path().join("undeclared-fixture-suite.json");
    let undeclared_suite = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-suite")
        .arg("--fixture-suite-fixture")
        .arg("unprofiled-positive")
        .arg("--strict")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--technical-white-balance")
        .arg("off")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(tmp.path().join("undeclared-suite-output"))
        .arg("--summary-json")
        .arg(&undeclared_suite_json)
        .output()
        .expect("run strict fixture suite with undeclared diagnostic delivery");
    assert!(!undeclared_suite.status.success());
    assert!(String::from_utf8_lossy(&undeclared_suite.stderr).contains(
        "fixture_suite:unprofiled-positive:non_reviewable_render_not_explicitly_expected"
    ));
    let undeclared_summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&undeclared_suite_json).unwrap()).unwrap();
    assert_eq!(undeclared_summary["status"], "review_required");

    registry["fixtures"]["unprofiled-positive"]["expectations"] =
        serde_json::json!({"render_reviewable": false});
    std::fs::write(
        &registry_path,
        serde_json::to_string_pretty(&registry).unwrap(),
    )
    .unwrap();
    let declared_suite_json = tmp.path().join("declared-fixture-suite.json");
    let declared_suite = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-suite")
        .arg("--fixture-suite-fixture")
        .arg("unprofiled-positive")
        .arg("--strict")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--technical-white-balance")
        .arg("off")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(tmp.path().join("declared-suite-output"))
        .arg("--summary-json")
        .arg(&declared_suite_json)
        .output()
        .expect("run strict fixture suite with declared diagnostic delivery");
    assert!(
        declared_suite.status.success(),
        "declared diagnostic fixture should pass strict suite: stderr={} summary={}",
        String::from_utf8_lossy(&declared_suite.stderr),
        std::fs::read_to_string(&declared_suite_json).unwrap_or_default()
    );
    let declared_summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&declared_suite_json).unwrap()).unwrap();
    assert_eq!(declared_summary["status"], "passed");
    assert_eq!(
        declared_summary["fixtures"][0]["expected_render_reviewable"],
        false
    );
    let declared_coverage_fixture = declared_summary["coverage"]["fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|fixture| fixture["name"] == "unprofiled-positive")
        .unwrap();
    assert_eq!(
        declared_coverage_fixture["render_dynamic_range_contract_declared"],
        false
    );
    assert!(!declared_coverage_fixture["action_items"]
        .as_array()
        .unwrap()
        .iter()
        .any(|action| action
            .as_str()
            .is_some_and(|action| action.starts_with("complete_render_dynamic_range_contract"))));

    let gated_suite_json = tmp.path().join("gated-fixture-suite.json");
    let gated_suite = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-suite")
        .arg("--fixture-suite-fixture")
        .arg("unprofiled-positive")
        .arg("--strict")
        .arg("--require-reviewable")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--technical-white-balance")
        .arg("off")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(tmp.path().join("gated-suite-output"))
        .arg("--summary-json")
        .arg(&gated_suite_json)
        .output()
        .expect("run production-gated diagnostic fixture suite");
    assert!(!gated_suite.status.success());
    assert!(String::from_utf8_lossy(&gated_suite.stderr)
        .contains("--require-reviewable rejected fixture-suite output(s): unprofiled-positive"));
    let gated_summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&gated_suite_json).unwrap()).unwrap();
    assert!(
        gated_summary["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue
                == "fixture_suite:unprofiled-positive:require_reviewable_not_satisfied")
    );

    let mut reported_reviewable: PipelineReport =
        serde_json::from_str(&std::fs::read_to_string(output_dir.join("report.json")).unwrap())
            .unwrap();
    let save = reported_reviewable
        .phases
        .iter_mut()
        .find(|phase| phase.name == "save")
        .unwrap();
    save.metrics["render_review_status"] = serde_json::json!("reviewable");
    save.metrics["render_reviewable"] = serde_json::json!(true);
    save.metrics["render_review_reason"] = serde_json::json!("synthetic gate probe");
    let reported_reviewable_path = tmp.path().join("reported-reviewable.json");
    reported_reviewable.save(&reported_reviewable_path).unwrap();
    std::fs::remove_file(output_dir.join("output.tiff")).unwrap();
    let missing_output_summary_json = tmp.path().join("missing-output-summary.json");
    let missing_output_gate = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--report")
        .arg(&reported_reviewable_path)
        .arg("--fixture")
        .arg("unprofiled-positive")
        .arg("--require-reviewable")
        .arg("--quiet")
        .arg("--summary-json")
        .arg(&missing_output_summary_json)
        .arg("--summary-md")
        .arg(tmp.path().join("missing-output-summary.md"))
        .output()
        .expect("gate report whose saved output disappeared");
    assert!(!missing_output_gate.status.success());
    let missing_output_stderr = String::from_utf8_lossy(&missing_output_gate.stderr);
    assert!(
        missing_output_stderr.contains("delivery_artifact_issues=")
            && missing_output_stderr.contains("output_icc_profile_mismatch"),
        "unexpected missing-output gate error: {missing_output_stderr}"
    );
    let missing_output_summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&missing_output_summary_json).unwrap())
            .unwrap();
    assert_eq!(
        missing_output_summary["render"]["render_review_status"],
        "reviewable"
    );
    assert_eq!(missing_output_summary["render"]["render_reviewable"], true);
    assert_eq!(
        missing_output_summary["render"]["output_file_icc_profile_matches_report"],
        false
    );
}

#[test]
fn test_validate_cli_runs_fixture_suite_strict() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let mut comp =
        synthetic::film_negative_image(160, 300, 10, 30, [12000, 7000, 3000], [4800, 2500, 1200]);
    // A spatial mosaic of four neutral exposure levels supplies independent neutral support
    // across both position and luminance. Each triplet inverts the measured 3x3 dye-separation
    // matrix and three characteristic curves at a shared scene-log-exposure target.
    let neutral_levels = [
        [6455, 3727, 1655],
        [3067, 1750, 811],
        [1287, 724, 353],
        [612, 340, 173],
    ];
    for y in 10..150 {
        for x in 30..270 {
            let level = ((x - 30) / 40 + (y - 10) / 35) % neutral_levels.len();
            for channel in 0..3 {
                comp[[y, x, channel]] = neutral_levels[level][channel];
            }
        }
    }
    let path1 = input_dir.join("left.tiff");
    let path2 = input_dir.join("right.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&comp, &path2).unwrap();

    let library_dir = write_validation_fixture_library_with_measured_negative_response(tmp.path());
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
                "min_negative_reconstruction_contract_fixtures": 1,
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
                    "component2": path1,
                    "output_dir": tmp.path().join("registry-output"),
                    "input_mode": "negative",
                    "bit_depth": 14,
                    "grain_reduction": "off",
                    "grain_strength": 0.42,
                    "grain_scale": 1.3,
                    "deskew": "off",
                    "force_no_stitch": true,
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
                        "deskew_status": "disabled",
                        "deskew_applied": false,
                        "deskew_review_required": false,
                        "deskew_retained_area_ratio_min": 1.0,
                        "stitch_decision": "skipped_pre_score",
                        "creative_temperature": 0.0,
                        "creative_tint": 0.0,
                        "base_estimate_source": "pre_crop_rebate_measurement",
                        "base_confidence_min": 0.3,
                        "density_inversion_skipped": false,
                        "negative_response_model": "measured_nonlinear_dye_separation",
                        "negative_response_source": "measured_roll_target_with_held_out_validation",
                        "negative_response_accepted": true,
                        "negative_response_review_required": false,
                        "negative_response_crosstalk_model": "measured_3x3_scanner_density_to_film_layers",
                        "negative_response_characteristic_curve_model": "measured_monotone_pchip_density_to_scene_log_exposure",
                        "negative_response_measured_model_id": "synthetic-validation-response-v1",
                        "negative_response_measured_confidence_min": 0.94,
                        "negative_response_held_out_delta_e00_rms_max": 1.8,
                        "negative_response_held_out_max_delta_e00_max": 4.9,
                        "negative_response_held_out_improvement_over_unit_slope_min": 6.0,
                        "negative_response_density_noise_gain_max": 4.0,
                        "negative_response_curve_extrapolated_ratio_max": 0.01,
                        "negative_response_signed_headroom_preserved": true,
                        "negative_response_curve_interpolation": "monotone_piecewise_cubic_hermite_with_endpoint_tangent_extrapolation",
                        "render_input_source": "direct_density_transmittance",
                        "calibration_acceptance_status": "accepted",
                        "calibration_color_mapping_applied": true,
                        "candidate_risk": "safe",
                        "tone_color_trust_state": "trusted",
                        "neutral_safety_rescue_applied": false,
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
                        "grain_reduction_enabled": false,
                        "grain_detail_review_required": false,
                        "grain_detail_decision_supported": true,
                        "grain_detail_luminance_p10_retention_min": 1.0,
                        "grain_detail_chroma_p10_retention_min": 1.0,
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
        .arg("--fixture-suite-fixture")
        .arg("logan")
        .arg("--strict")
        .arg("--debug")
        .arg("--grain-reduction")
        .arg("on")
        .arg("--grain-strength")
        .arg("0.9")
        .arg("--grain-scale")
        .arg("3.0")
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
        "fixture suite CLI failed: status={} stderr={} stdout={} suite_summary={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout),
        std::fs::read_to_string(&suite_json)
            .unwrap_or_else(|err| format!("<failed to read {}: {err}>", suite_json.display()))
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
    assert_eq!(
        suite["coverage"]["negative_reconstruction_contract_fixture_count"],
        1
    );
    assert_eq!(suite["fixtures"][0]["name"], "logan");
    assert_eq!(suite["fixtures"][0]["status"], "passed");
    assert_eq!(suite["fixtures"][0]["creative_temperature"], 0.0);
    assert_eq!(suite["fixtures"][0]["expected_creative_temperature"], 0.0);
    assert_eq!(suite["fixtures"][0]["creative_tint"], 0.0);
    assert_eq!(suite["fixtures"][0]["deskew_status"], "disabled");
    assert_eq!(suite["fixtures"][0]["expected_deskew_status"], "disabled");
    assert_eq!(suite["fixtures"][0]["deskew_applied"], false);
    assert_eq!(suite["fixtures"][0]["deskew_review_required"], false);
    assert_eq!(suite["fixtures"][0]["deskew_retained_area_ratio"], 1.0);
    assert_eq!(suite["fixtures"][0]["effective_grain_reduction"], "off");
    assert_eq!(suite["fixtures"][0]["effective_grain_strength"], 0.42);
    assert_eq!(suite["fixtures"][0]["effective_grain_scale"], 1.3);
    assert_eq!(suite["fixtures"][0]["grain_reduction_enabled"], false);
    assert_eq!(
        suite["fixtures"][0]["expected_grain_reduction_enabled"],
        false
    );
    assert_eq!(suite["fixtures"][0]["grain_detail_review_required"], false);
    assert_eq!(
        suite["fixtures"][0]["grain_detail_decision_supported"],
        true
    );
    assert_eq!(
        suite["fixtures"][0]["grain_detail_luminance_p10_retention"],
        1.0
    );
    assert_eq!(
        suite["fixtures"][0]["grain_detail_chroma_p10_retention"],
        1.0
    );
    assert_eq!(suite["coverage"]["fixtures"][0]["input_mode"], "negative");
    assert_eq!(suite["coverage"]["fixtures"][0]["bit_depth"], 14);
    assert_eq!(suite["coverage"]["fixtures"][0]["grain_reduction"], "off");
    assert_eq!(suite["coverage"]["fixtures"][0]["grain_strength"], 0.42);
    assert_eq!(suite["coverage"]["fixtures"][0]["grain_scale"], 1.3);
    assert_eq!(suite["coverage"]["fixtures"][0]["deskew"], "off");
    assert_eq!(suite["coverage"]["fixtures"][0]["force_no_stitch"], true);
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
        "pre_crop_rebate_measurement"
    );
    assert_eq!(
        suite["fixtures"][0]["expected_base_estimate_source"],
        "pre_crop_rebate_measurement"
    );
    assert_eq!(
        suite["fixtures"][0]["negative_reconstruction"]["response_model"],
        "measured_nonlinear_dye_separation"
    );
    assert_eq!(
        suite["fixtures"][0]["negative_reconstruction"]["measured_model_id"],
        "synthetic-validation-response-v1"
    );
    assert_eq!(
        suite["fixtures"][0]["negative_reconstruction"]["response_review_required"],
        false
    );
    assert_eq!(
        suite["fixtures"][0]["negative_reconstruction"]["signed_headroom_preserved"],
        true
    );
    assert!(
        suite["fixtures"][0]["negative_reconstruction"]["curve_extrapolated_any_ratio"]
            .as_f64()
            .is_some_and(|ratio| ratio <= 0.01)
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
    assert_eq!(
        suite["fixtures"][0]["neutral_safety_rescue_applied"], false,
        "fixture={}",
        suite["fixtures"][0]
    );
    assert_eq!(
        suite["fixtures"][0]["expected_neutral_safety_rescue_applied"],
        false
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
    assert!(markdown.contains("Geometry preparation"));
    assert!(markdown.contains("Negative reconstruction"));
    assert!(markdown.contains("Tone trust"));
    assert!(markdown.contains("Tone protection"));
    assert!(markdown.contains("expected skipped_pre_score; actual skipped_pre_score"));
    assert!(markdown
        .contains("expected pre_crop_rebate_measurement; actual pre_crop_rebate_measurement"));
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
    assert!(markdown.contains("grain config off, strength 0.420, scale 1.300"));
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
    assert!(markdown.contains(
        "response model expected measured_nonlinear_dye_separation; actual measured_nonlinear_dye_separation"
    ));
    assert!(markdown.contains("held-out dE00 RMS max 1.800000; actual 1.800000"));
    assert!(markdown.contains("logan"));

    let adversarial_registry_path = tmp.path().join("fixtures-adversarial.json");
    let mut adversarial_registry: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&registry_path).unwrap()).unwrap();
    adversarial_registry["fixtures"]["logan"]["expectations"]
        ["negative_response_measured_confidence_min"] = serde_json::json!(0.95);
    std::fs::write(
        &adversarial_registry_path,
        serde_json::to_string_pretty(&adversarial_registry).unwrap(),
    )
    .unwrap();
    let adversarial_output_dir = tmp.path().join("suite-adversarial-output");
    let adversarial_json = tmp.path().join("fixture-suite-adversarial.json");
    let adversarial_output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&adversarial_registry_path)
        .arg("--fixture-suite")
        .arg("--fixture-suite-fixture")
        .arg("logan")
        .arg("--strict")
        .arg("--debug")
        .arg("--ica-max-iter")
        .arg("20")
        .arg("--ica-tol")
        .arg("0.0001")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(&adversarial_output_dir)
        .arg("--summary-json")
        .arg(&adversarial_json)
        .output()
        .expect("run adversarial measured-response confidence fixture suite");
    assert!(
        !adversarial_output.status.success(),
        "a stricter-than-measured confidence floor must fail the strict suite"
    );
    assert!(String::from_utf8_lossy(&adversarial_output.stderr)
        .contains("fixture_suite:logan:negative_response_measured_confidence_below_expected"));
    let adversarial: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&adversarial_json).unwrap()).unwrap();
    assert_eq!(
        adversarial["coverage"]["negative_reconstruction_contract_fixture_count"], 1,
        "the registry contract remains structurally complete while runtime evidence fails"
    );
    assert_eq!(
        adversarial["fixtures"][0]["negative_reconstruction"]["measured_confidence"],
        0.94
    );
    assert_eq!(
        adversarial["fixtures"][0]["negative_reconstruction"]["expected_measured_confidence_min"],
        0.95
    );
}

#[test]
fn test_validate_cli_direct_fixture_honors_registry_pipeline_overrides() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input = tmp.path().join("positive.tiff");
    let image = synthetic::image_with_borders(96, 144, 12, [7_000, 8_000, 9_000]);
    scanstitch::tiff_io::save_tiff_u16(&image, &input).unwrap();
    let output_dir = tmp.path().join("direct-output");
    let registry_path = tmp.path().join("fixtures.json");
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "fixtures": {
                "direct-positive": {
                    "component1": input.clone(),
                    "output_dir": output_dir,
                    "input_mode": "positive",
                    "bit_depth": 16,
                    "grain_reduction": "on",
                    "grain_strength": 0.37,
                    "grain_scale": 1.7,
                    "deskew": "manual",
                    "deskew_angle_degrees": 0.5,
                    "force_no_stitch": true
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture")
        .arg("direct-positive")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--quiet")
        .output()
        .expect("run direct fixture");
    assert!(
        output.status.success(),
        "direct fixture failed: stderr={} stdout={}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(output_dir.join("report.json")).unwrap())
            .unwrap();
    let phase = |name: &str| {
        report["phases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|phase| phase["name"] == name)
            .unwrap()
    };
    assert_eq!(phase("load")["metrics"]["preserved_working_bit_depth"], 16);
    assert_eq!(phase("deskew")["metrics"]["requested_mode"], "manual");
    assert_eq!(phase("deskew")["metrics"]["status"], "applied");
    assert_eq!(phase("load")["metrics"]["input_count"], 1);
    assert_eq!(
        phase("stitch")["metrics"]["decision"],
        "skipped_single_input"
    );
    assert_eq!(
        phase("density_inversion")["metrics"]["input_mode"],
        "positive"
    );
    assert_eq!(phase("density_inversion")["metrics"]["skipped"], true);
    assert_eq!(
        phase("tone_mapping")["metrics"]["grain_reduction"]["requested"]["enabled"],
        true
    );
    assert_eq!(
        phase("tone_mapping")["metrics"]["grain_reduction"]["requested"]["strength"],
        0.37
    );
    assert_eq!(
        phase("tone_mapping")["metrics"]["grain_reduction"]["requested"]["scale"],
        1.7
    );

    let override_output_dir = tmp.path().join("single-override-output");
    let override_output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--component1")
        .arg(&input)
        .arg("--input-mode")
        .arg("positive")
        .arg("--bit-depth")
        .arg("16")
        .arg("--force-no-stitch")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--output-dir")
        .arg(&override_output_dir)
        .arg("--quiet")
        .output()
        .expect("run direct single-component override");
    assert!(
        override_output.status.success(),
        "single-component override failed: stderr={} stdout={}",
        String::from_utf8_lossy(&override_output.stderr),
        String::from_utf8_lossy(&override_output.stdout)
    );
    let override_report: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(override_output_dir.join("report.json")).unwrap(),
    )
    .unwrap();
    let override_stitch = override_report["phases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|phase| phase["name"] == "stitch")
        .unwrap();
    assert_eq!(
        override_stitch["metrics"]["decision"],
        "skipped_single_input"
    );
}

#[test]
fn test_validate_cli_single_component_fixture_is_a_set_but_not_a_pair() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input = tmp.path().join("single.tiff");
    let baseline_path = tmp.path().join("baseline.json");
    let registry_path = tmp.path().join("fixtures.json");
    let summary_json = tmp.path().join("coverage.json");
    let summary_md = tmp.path().join("coverage.md");
    let image = synthetic::constant_image(4, 5, [12_000, 7_000, 3_000]);
    scanstitch::tiff_io::save_tiff_u16(&image, &input).unwrap();
    let component1_sha256 = file_sha256_hex(&input);
    std::fs::write(
        &baseline_path,
        serde_json::to_string_pretty(&fixture_summary_baseline()).unwrap(),
    )
    .unwrap();
    let summary_baseline_sha256 = file_sha256_hex(&baseline_path);
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "min_component_sha256_sets": 1,
                "min_tiff_bits_per_sample": 14,
                "min_summary_baselines": 1,
                "min_summary_baseline_sha256_fixtures": 1,
                "min_uncalibrated_fixtures": 1
            },
            "fixtures": {
                "logan": {
                    "component1": input,
                    "component1_sha256": component1_sha256.clone(),
                    "summary_baseline": baseline_path,
                    "summary_baseline_sha256": summary_baseline_sha256,
                    "film_stock": "Synthetic positive",
                    "scene_tags": ["single-frame"],
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
        .expect("run single-component fixture coverage");
    assert!(
        output.status.success(),
        "single-component coverage failed: stderr={} stdout={}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_json).unwrap()).unwrap();
    assert_eq!(summary["status"], "passed");
    assert_eq!(summary["fixture_count"], 1);
    assert_eq!(summary["component_file_count"], 1);
    assert_eq!(summary["max_component_count"], 1);
    assert_eq!(summary["component_set_available_count"], 1);
    assert_eq!(summary["component_sha256_declared_set_count"], 1);
    assert_eq!(summary["component_sha256_computed_set_count"], 1);
    assert_eq!(summary["component_sha256_set_count"], 1);
    assert_eq!(summary["readable_tiff_set_count"], 1);
    assert_eq!(summary["tiff_layout_consistent_set_count"], 1);
    assert_eq!(summary["tiff_dimension_matched_set_count"], 1);
    assert_eq!(summary["component_pair_available_count"], 0);
    assert_eq!(summary["component_sha256_declared_pair_count"], 0);
    assert_eq!(summary["component_sha256_computed_pair_count"], 0);
    assert_eq!(summary["component_sha256_pair_count"], 0);
    assert_eq!(summary["readable_tiff_pair_count"], 0);
    assert_eq!(summary["tiff_layout_consistent_pair_count"], 0);
    assert_eq!(summary["tiff_dimension_matched_pair_count"], 0);
    assert_eq!(summary["validation_ready_fixture_count"], 1);
    let fixture = &summary["fixtures"][0];
    assert_eq!(fixture["component_count"], 1);
    assert!(fixture["component2"].is_null());
    assert_eq!(fixture["component2_exists"], false);
    assert!(fixture["component2_sha256"].is_null());
    assert!(fixture["component2_tiff"].is_null());
    assert!(fixture["tiff_pair"].is_null());
    assert_eq!(fixture["all_components_exist"], true);
    assert_eq!(fixture["all_component_sha256_matched"], true);
    assert_eq!(fixture["all_components_readable_tiff"], true);
    assert_eq!(fixture["all_component_layouts_consistent"], true);
    assert_eq!(fixture["all_component_dimensions_matched"], true);
    assert_eq!(fixture["component1_sha256"]["matched"], true);
    assert_eq!(
        fixture["component1_sha256"]["actual_sha256"],
        component1_sha256
    );
    assert_eq!(fixture["validation_ready"], true);
    assert_eq!(fixture["action_items"], serde_json::json!([]));
    let markdown = std::fs::read_to_string(&summary_md).unwrap();
    assert!(markdown.contains("| logan | yes |"));
    assert!(markdown.contains("RGB16 5x4 16bpc"));
}

#[test]
fn test_validate_cli_fixture_suite_proves_geometry_preparation_contract() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input = tmp.path().join("four-edge-board-positive.tiff");
    let mut image = synthetic::constant_image(160, 220, [0, 0, 0]);
    for y in 16..144 {
        for x in 16..204 {
            let checker = if ((x / 6) + (y / 6)) % 2 == 0 {
                1_u16
            } else {
                0_u16
            };
            image[[y, x, 0]] = 14_000 + ((x * 71 + y * 29) % 18_000) as u16;
            image[[y, x, 1]] = 12_000 + ((x * 41 + y * 53) % 16_000) as u16;
            image[[y, x, 2]] = 10_000 + ((x * 37 + y * 23) % 14_000) as u16 + checker * 1_200;
        }
    }
    write_rgb16_tiff_with_orientation(&input, &image, 6);

    let registry_path = tmp.path().join("geometry-fixtures.json");
    let direct_output_dir = tmp.path().join("geometry-direct-output");
    let baseline_path = tmp.path().join("geometry-summary-baseline.json");
    let mut registry = serde_json::json!({
        "fixtures": {
            "logan": {
                "component1": input,
                "component2": input,
                "output_dir": direct_output_dir,
                "input_mode": "positive",
                "bit_depth": 16,
                "deskew": "manual",
                "deskew_angle_degrees": 0.5,
                "orientation_correction": "rotate-180",
                "force_no_stitch": true,
                "film_stock": "Synthetic positive",
                "scene_tags": ["four-edge-scanner-board"],
                "exposure_tags": ["normal-exposure"],
                "calibration_case": "uncalibrated-image-derived"
            }
        }
    });
    std::fs::write(
        &registry_path,
        serde_json::to_string_pretty(&registry).unwrap(),
    )
    .unwrap();
    let orientation_review_dir = tmp.path().join("geometry-orientation-review");
    let orientation_review = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture")
        .arg("logan")
        .arg("--write-orientation-review")
        .arg(&orientation_review_dir)
        .arg("--quiet")
        .output()
        .expect("write orientation review package");
    assert!(
        orientation_review.status.success(),
        "orientation review package failed: {}",
        String::from_utf8_lossy(&orientation_review.stderr)
    );
    let orientation_review: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(orientation_review_dir.join("orientation-review.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(orientation_review["schema_version"], 1);
    assert_eq!(
        orientation_review["review_status"],
        "requires_human_approval"
    );
    assert_eq!(orientation_review["input_mode"], "positive");
    assert_eq!(orientation_review["working_bit_depth"], 16);
    assert_eq!(orientation_review["orientation_correction"], "rotate-180");
    assert_eq!(
        orientation_review["orientation_components_expected"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let orientation_draft = &orientation_review["orientation_components_expected"][0];
    assert_eq!(orientation_draft["upright_approved"], false);
    assert_eq!(orientation_draft["tag_value"], 6);
    assert_eq!(orientation_draft["transform"], "rotate_270_clockwise");
    assert_eq!(orientation_draft["source_width"], 220);
    assert_eq!(orientation_draft["source_height"], 160);
    assert_eq!(orientation_draft["output_width"], 160);
    assert_eq!(orientation_draft["output_height"], 220);
    let reviewed_decoded_pixel_sha256 = orientation_draft["decoded_pixel_sha256"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(reviewed_decoded_pixel_sha256.len(), 64);
    let preview_path = orientation_review_dir.join(
        orientation_review["previews"][0]["preview_path"]
            .as_str()
            .unwrap(),
    );
    let (preview_width, preview_height) = image::image_dimensions(&preview_path).unwrap();
    assert!(preview_height > preview_width);
    let orientation_review_markdown =
        std::fs::read_to_string(orientation_review_dir.join("orientation-review.md")).unwrap();
    assert!(orientation_review_markdown.contains("never approves its own output"));
    assert!(orientation_review_markdown.contains("upright_approved` value `false"));
    assert!(orientation_review_markdown.contains("orientation correction: `rotate-180`"));

    let single_orientation_review_dir = tmp.path().join("single-orientation-review");
    let single_orientation_review = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture")
        .arg("single-positive")
        .arg("--component1")
        .arg(&input)
        .arg("--input-mode")
        .arg("positive")
        .arg("--bit-depth")
        .arg("16")
        .arg("--orientation-correction")
        .arg("rotate-180")
        .arg("--write-orientation-review")
        .arg(&single_orientation_review_dir)
        .arg("--quiet")
        .output()
        .expect("write single-component orientation review package");
    assert!(
        single_orientation_review.status.success(),
        "single orientation review package failed: {}",
        String::from_utf8_lossy(&single_orientation_review.stderr)
    );
    let single_orientation_review: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(single_orientation_review_dir.join("orientation-review.json"))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        single_orientation_review["orientation_components_expected"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        single_orientation_review["orientation_correction"],
        "rotate-180"
    );
    assert_eq!(
        single_orientation_review["orientation_components_expected"][0]["decoded_pixel_sha256"],
        reviewed_decoded_pixel_sha256
    );

    let direct = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture")
        .arg("logan")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--quiet")
        .arg("--write-summary-baseline")
        .arg(&baseline_path)
        .output()
        .expect("run direct geometry fixture and write baseline");
    assert!(
        direct.status.success(),
        "direct geometry fixture failed: {}",
        String::from_utf8_lossy(&direct.stderr)
    );
    let direct_summary: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(direct_output_dir.join("summary.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(direct_summary["input_orientation"]["input_count"], 2);
    assert_eq!(
        direct_summary["input_orientation"]["all_components_reported"],
        true
    );
    let orientation_components = direct_summary["input_orientation"]["components"]
        .as_array()
        .unwrap();
    let decoded_pixel_sha256 = orientation_components[0]["decoded_pixel_sha256"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(decoded_pixel_sha256.len(), 64);
    assert_eq!(decoded_pixel_sha256, reviewed_decoded_pixel_sha256);
    for component in orientation_components {
        assert_eq!(component["decoded_pixel_sha256"], decoded_pixel_sha256);
        assert_eq!(component["tag_value"], 6);
        assert_eq!(component["metadata_transform"], "rotate_90_clockwise");
        assert_eq!(component["orientation_correction_requested"], "rotate-180");
        assert_eq!(component["orientation_correction_transform"], "rotate_180");
        assert_eq!(component["orientation_correction_applied"], true);
        assert_eq!(component["effective_tag_value"], 8);
        assert_eq!(component["transform"], "rotate_270_clockwise");
        assert_eq!(component["applied"], true);
        assert_eq!(component["source_width"], 220);
        assert_eq!(component["source_height"], 160);
        assert_eq!(component["output_width"], 160);
        assert_eq!(component["output_height"], 220);
    }
    assert_eq!(direct_summary["deskew"]["all_components_applied"], true);
    assert!(
        direct_summary["deskew"]["minimum_component_retained_area_ratio"]
            .as_f64()
            .is_some_and(|ratio| ratio >= 0.9)
    );
    assert_eq!(
        direct_summary["border_crop"]["all_components_cropped"],
        true
    );
    assert_eq!(
        direct_summary["border_crop"]["minimum_removed_edge_count_per_component"],
        4
    );
    assert!(direct_summary["border_crop"]["minimum_retained_area_ratio"]
        .as_f64()
        .is_some_and(|ratio| ratio >= 0.5));
    assert!(direct_summary["border_crop"]["maximum_retained_area_ratio"]
        .as_f64()
        .is_some_and(|ratio| ratio < 0.99));
    assert!(direct_summary["deskew"]["correction_degrees"]
        .as_f64()
        .is_some_and(|angle| (angle + 0.5).abs() <= 1e-9));
    let crop_components = direct_summary["border_crop"]["components"]
        .as_array()
        .unwrap();
    assert_eq!(crop_components.len(), 2);
    for component in crop_components {
        for edge in [
            "top_removed",
            "bottom_removed",
            "left_removed",
            "right_removed",
        ] {
            assert!(component[edge]
                .as_u64()
                .is_some_and(|removed| removed.abs_diff(16) <= 4));
        }
    }
    let border_crop_components_expected = serde_json::json!([
        {
            "component_index": 1,
            "top_removed": 16,
            "bottom_removed": 16,
            "left_removed": 16,
            "right_removed": 16,
            "tolerance_px": 4
        },
        {
            "component_index": 2,
            "top_removed": 16,
            "bottom_removed": 16,
            "left_removed": 16,
            "right_removed": 16,
            "tolerance_px": 4
        }
    ]);

    registry["coverage_requirements"] = serde_json::json!({
        "min_geometry_preparation_contract_fixtures": 1,
        "min_geometry_accuracy_contract_fixtures": 1,
        "min_orientation_accuracy_contract_fixtures": 1
    });
    registry["fixtures"]["logan"]["summary_baseline"] = serde_json::json!(baseline_path);
    registry["fixtures"]["logan"]["expectations"] = serde_json::json!({
        "deskew_status": "applied",
        "deskew_applied": true,
        "deskew_review_required": false,
        "deskew_all_components_applied": true,
        "deskew_minimum_component_retained_area_ratio_min": 0.9,
        "border_crop_all_components_cropped": true,
        "border_crop_minimum_removed_edge_count_per_component_min": 4,
        "border_crop_retained_area_ratio_min": 0.5,
        "border_crop_retained_area_ratio_max": 0.99,
        "border_crop_rejected": false,
        "render_review_status": "review_required_color",
        "render_reviewable": false,
        "deskew_correction_degrees_expected": -0.5,
        "deskew_correction_tolerance_degrees": 0.001,
        "border_crop_components_expected": border_crop_components_expected,
        "orientation_components_expected": [
            {
                "component_index": 1,
                "upright_approved": true,
                "decoded_pixel_sha256": decoded_pixel_sha256.clone(),
                "tag_present": true,
                "tag_value": 6,
                "transform": "rotate_270_clockwise",
                "applied": true,
                "source_width": 220,
                "source_height": 160,
                "output_width": 160,
                "output_height": 220
            },
            {
                "component_index": 2,
                "upright_approved": true,
                "decoded_pixel_sha256": decoded_pixel_sha256.clone(),
                "tag_present": true,
                "tag_value": 6,
                "transform": "rotate_270_clockwise",
                "applied": true,
                "source_width": 220,
                "source_height": 160,
                "output_width": 160,
                "output_height": 220
            }
        ]
    });
    std::fs::write(
        &registry_path,
        serde_json::to_string_pretty(&registry).unwrap(),
    )
    .unwrap();

    let suite_output_dir = tmp.path().join("geometry-suite-output");
    let suite_json = tmp.path().join("geometry-suite.json");
    let suite = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-suite")
        .arg("--fixture-suite-fixture")
        .arg("logan")
        .arg("--strict")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(&suite_output_dir)
        .arg("--summary-json")
        .arg(&suite_json)
        .output()
        .expect("run strict geometry fixture suite");
    assert!(
        suite.status.success(),
        "strict geometry fixture failed: {} {}",
        String::from_utf8_lossy(&suite.stderr),
        std::fs::read_to_string(&suite_json).unwrap_or_default()
    );
    let suite: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&suite_json).unwrap()).unwrap();
    assert_eq!(suite["status"], "passed");
    assert_eq!(
        suite["coverage"]["geometry_preparation_contract_fixture_count"],
        1
    );
    assert_eq!(
        suite["coverage"]["geometry_accuracy_contract_fixture_count"],
        1
    );
    assert_eq!(
        suite["coverage"]["orientation_accuracy_contract_fixture_count"],
        1
    );
    assert_eq!(
        suite["fixtures"][0]["input_orientation"]["components"][0]["tag_value"],
        6
    );
    assert_eq!(
        suite["fixtures"][0]["geometry_preparation"]["all_border_crop_components_cropped"],
        true
    );
    assert_eq!(
        suite["fixtures"][0]["geometry_preparation"]["minimum_removed_edge_count_per_component"],
        4
    );
    let suite_markdown =
        std::fs::read_to_string(suite_output_dir.join("fixture-suite.md")).unwrap();
    assert!(suite_markdown.contains("Geometry preparation / orientation"));
    assert!(suite_markdown.contains("deskew all components expected true; actual true"));
    assert!(suite_markdown.contains("removed edges minimum expected 4; actual 4"));
    assert!(suite_markdown
        .contains("deskew correction expected -0.500000 +/- 0.001000; actual -0.500000"));
    assert!(suite_markdown.contains("crop component 1 top/bottom/left/right expected"));
    assert!(suite_markdown
        .contains("orientation component 1 upright approved true; decoded pixel SHA-256 expected"));
    assert!(suite_markdown
        .contains("tag expected 6; actual 6; transform expected rotate_270_clockwise"));

    let adversarial_registry_path = tmp.path().join("geometry-fixtures-adversarial.json");
    let expected_top = registry["fixtures"]["logan"]["expectations"]
        ["border_crop_components_expected"][0]["top_removed"]
        .as_u64()
        .unwrap();
    let actual_top = crop_components[0]["top_removed"].as_u64().unwrap();
    registry["fixtures"]["logan"]["expectations"]["border_crop_components_expected"][0]
        ["top_removed"] = serde_json::json!(actual_top + 5);
    assert_ne!(expected_top, actual_top + 5);
    registry["fixtures"]["logan"]["expectations"]["orientation_components_expected"][0]
        ["tag_value"] = serde_json::json!(8);
    registry["fixtures"]["logan"]["expectations"]["orientation_components_expected"][0]
        ["transform"] = serde_json::json!("rotate_90_clockwise");
    registry["fixtures"]["logan"]["expectations"]["orientation_components_expected"][0]
        ["decoded_pixel_sha256"] = serde_json::json!("f".repeat(64));
    std::fs::write(
        &adversarial_registry_path,
        serde_json::to_string_pretty(&registry).unwrap(),
    )
    .unwrap();
    let adversarial_json = tmp.path().join("geometry-suite-adversarial.json");
    let adversarial = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&adversarial_registry_path)
        .arg("--fixture-suite")
        .arg("--fixture-suite-fixture")
        .arg("logan")
        .arg("--strict")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(tmp.path().join("geometry-suite-adversarial-output"))
        .arg("--summary-json")
        .arg(&adversarial_json)
        .output()
        .expect("run adversarial geometry fixture suite");
    assert!(!adversarial.status.success());
    assert!(String::from_utf8_lossy(&adversarial.stderr)
        .contains("fixture_suite:logan:border_crop_component1_top_outside_tolerance"));
    assert!(String::from_utf8_lossy(&adversarial.stderr)
        .contains("fixture_suite:logan:orientation_component1_tag_value_mismatch"));
    assert!(String::from_utf8_lossy(&adversarial.stderr)
        .contains("fixture_suite:logan:orientation_component1_transform_mismatch"));
    assert!(String::from_utf8_lossy(&adversarial.stderr)
        .contains("fixture_suite:logan:orientation_component1_decoded_pixel_sha256_mismatch"));
    let adversarial: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&adversarial_json).unwrap()).unwrap();
    assert_eq!(
        adversarial["coverage"]["geometry_preparation_contract_fixture_count"], 1,
        "the structurally complete registry must remain distinct from failed runtime evidence"
    );
    assert_eq!(
        adversarial["coverage"]["geometry_accuracy_contract_fixture_count"], 1,
        "a complete factual declaration must remain distinct from failed runtime accuracy"
    );
    assert_eq!(
        adversarial["coverage"]["orientation_accuracy_contract_fixture_count"], 1,
        "a coherent but factually wrong orientation declaration must fail only at runtime"
    );
}

#[test]
fn test_validate_cli_fixture_suite_proves_dynamic_range_and_grain_contracts() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input = tmp.path().join("grain-detail-positive.tiff");
    let mut image = synthetic::constant_image(128, 192, [24_900, 24_900, 24_900]);
    for y in 0..128 {
        for x in 0..192 {
            let checker = if (x + y) % 2 == 0 { 1_i32 } else { -1_i32 };
            if x < 32 {
                let neutral_base = if x < 16 { 10_000 } else { 46_000 };
                let neutral = (neutral_base + checker * 800).clamp(0, u16::MAX as i32) as u16;
                image[[y, x, 0]] = neutral;
                image[[y, x, 1]] = neutral;
                image[[y, x, 2]] = neutral;
                continue;
            }
            let luma_step = if (48..96).contains(&x) { 7_000 } else { 0 };
            let chroma_step = if (112..160).contains(&x) { 3_500 } else { 0 };
            let base = 24_900 + luma_step;
            image[[y, x, 0]] =
                (base + checker * 5_200 + chroma_step).clamp(0, u16::MAX as i32) as u16;
            image[[y, x, 1]] = base.clamp(0, u16::MAX as i32) as u16;
            image[[y, x, 2]] =
                (base - checker * 5_200 - chroma_step).clamp(0, u16::MAX as i32) as u16;
        }
    }
    scanstitch::tiff_io::save_tiff_u16(&image, &input).unwrap();

    let registry_path = tmp.path().join("fixtures.json");
    let baseline_path = tmp.path().join("grain-summary-baseline.json");
    let direct_output_dir = tmp.path().join("direct-output");
    let calibration_profile = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("docs")
        .join("color-calibration-profile.example.json");
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "fixtures": {
                "logan": {
                    "component1": input,
                    "component2": input,
                    "output_dir": direct_output_dir,
                    "input_mode": "positive",
                    "bit_depth": 16,
                    "grain_reduction": "on",
                    "grain_strength": 0.4,
                    "grain_scale": 1.0,
                    "force_no_stitch": true,
                    "calibration_profile": calibration_profile,
                    "film_stock": "Synthetic positive",
                    "scene_tags": ["fine-texture", "isoluminant-edge"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "external-profile"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let direct = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture")
        .arg("logan")
        .arg("--color-mode")
        .arg("calibrated")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--quiet")
        .arg("--write-summary-baseline")
        .arg(&baseline_path)
        .output()
        .expect("run direct grain-effect fixture baseline");
    assert!(
        direct.status.success(),
        "direct grain-effect fixture failed: stderr={} stdout={}",
        String::from_utf8_lossy(&direct.stderr),
        String::from_utf8_lossy(&direct.stdout)
    );

    let report: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(direct_output_dir.join("report.json")).unwrap(),
    )
    .unwrap();
    let tone = report["phases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|phase| phase["name"] == "tone_mapping")
        .map(|phase| &phase["metrics"])
        .unwrap();
    let applied_ratio = tone["noise_reduction_applied_ratio"].as_f64().unwrap();
    let structure_excluded_ratio = tone["noise_reduction_structure_excluded_ratio"]
        .as_f64()
        .unwrap();
    let flat_luma_reduction = tone["noise_reduction_flat_luma_p95_reduction_ratio"]
        .as_f64()
        .unwrap();
    let flat_chroma_reduction = tone["noise_reduction_flat_chroma_p95_reduction_ratio"]
        .as_f64()
        .unwrap();
    let detail = &tone["grain_reduction"]["detail_retention"];
    let direct_summary: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(direct_output_dir.join("summary.json")).unwrap(),
    )
    .unwrap();
    let post_scale_preserved_ratio = direct_summary["colorspace"]["post_scale_preserved_ratio"]
        .as_f64()
        .unwrap();
    let render_luminance_range = direct_summary["tone"]["render_luminance_range_p05_p95"]
        .as_f64()
        .unwrap();
    let tone_output_evidence_confidence = direct_summary["tone"]["tone_output_evidence_confidence"]
        .as_f64()
        .unwrap();
    let render_to_mapped_luminance_range_ratio = direct_summary["tone"]
        ["render_to_mapped_luminance_range_ratio"]
        .as_f64()
        .unwrap();
    let maximum_ratio = |value: &serde_json::Value| {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|ratio| ratio.as_f64().unwrap())
            .fold(0.0, f64::max)
    };
    let high_clipping_ratio =
        maximum_ratio(&direct_summary["tone"]["post_chroma_compression_clipped_high_ratio"]);
    let low_clipping_ratio =
        maximum_ratio(&direct_summary["tone"]["post_chroma_compression_clipped_low_ratio"]);
    assert!(
        applied_ratio > 0.0,
        "grain pass must change supported pixels"
    );
    assert!(
        structure_excluded_ratio > 0.0,
        "grain pass must exactly exclude structured or unsupported pixels"
    );
    assert!(
        flat_luma_reduction > 0.0,
        "grain pass must reduce flat-area luma residuals: {flat_luma_reduction}"
    );
    assert!(
        flat_chroma_reduction > 0.0,
        "grain pass must reduce flat-area chroma residuals: {flat_chroma_reduction}"
    );
    assert_eq!(detail["review_required"], false);
    assert_eq!(detail["decision_supported"], true);
    assert!(detail["luminance_probe_count"].as_u64().unwrap() >= 64);
    assert!(detail["chroma_probe_count"].as_u64().unwrap() >= 64);
    assert!(detail["luminance_p10_retention"].as_f64().unwrap() >= 0.70);
    assert!(detail["chroma_p10_retention"].as_f64().unwrap() >= 0.70);
    assert!(post_scale_preserved_ratio > 0.0);
    assert!(render_luminance_range > 0.0 && render_luminance_range < 1.0);
    assert_eq!(
        direct_summary["render"]["render_review_status"], "reviewable",
        "trusted dynamic-range fixture diagnostics: {}",
        direct_summary["colorspace"]
    );
    assert_eq!(direct_summary["render"]["render_reviewable"], true);
    assert_eq!(
        direct_summary["tone"]["tone_output_confidence_status"],
        "supported_render_tonal_distribution"
    );
    assert_eq!(direct_summary["tone"]["tone_output_review_required"], false);
    assert!(tone_output_evidence_confidence > 0.0);
    assert!(render_to_mapped_luminance_range_ratio > 0.0);
    assert!(high_clipping_ratio < 1.0);
    assert!(low_clipping_ratio < 1.0);

    let applied_floor = applied_ratio * 0.5;
    let structure_excluded_floor = structure_excluded_ratio * 0.5;
    let flat_luma_floor = flat_luma_reduction * 0.5;
    let flat_chroma_floor = flat_chroma_reduction * 0.5;
    let post_scale_preserved_floor = post_scale_preserved_ratio * 0.5;
    let render_luminance_range_floor = render_luminance_range * 0.5;
    let tone_output_evidence_confidence_floor = tone_output_evidence_confidence * 0.5;
    let render_to_mapped_luminance_range_ratio_floor = render_to_mapped_luminance_range_ratio * 0.5;
    let high_clipping_ceiling = (high_clipping_ratio + 1.0) * 0.5;
    let low_clipping_ceiling = (low_clipping_ratio + 1.0) * 0.5;
    let accepting_registry = serde_json::json!({
        "coverage_requirements": {
            "min_render_dynamic_range_contract_fixtures": 1,
            "min_grain_reduction_enabled_fixtures": 1,
            "min_grain_detail_contract_fixtures": 1,
            "min_grain_reduction_effect_contract_fixtures": 1
        },
        "fixtures": {
            "logan": {
                "component1": input,
                "component2": input,
                "input_mode": "positive",
                "bit_depth": 16,
                "grain_reduction": "on",
                "grain_strength": 0.4,
                "grain_scale": 1.0,
                "force_no_stitch": true,
                "summary_baseline": baseline_path,
                "calibration_profile": calibration_profile,
                "film_stock": "Synthetic positive",
                "scene_tags": ["fine-texture", "isoluminant-edge"],
                "exposure_tags": ["normal-exposure"],
                "calibration_case": "external-profile",
                "expectations": {
                    "grain_reduction_enabled": true,
                    "grain_reduction_applied_ratio_min": applied_floor,
                    "grain_reduction_structure_excluded_ratio_min": structure_excluded_floor,
                    "grain_reduction_flat_luma_p95_reduction_ratio_min": flat_luma_floor,
                    "grain_reduction_flat_chroma_p95_reduction_ratio_min": flat_chroma_floor,
                    "grain_detail_review_required": false,
                    "grain_detail_decision_supported": true,
                    "grain_detail_luminance_probe_count_min": 64,
                    "grain_detail_chroma_probe_count_min": 64,
                    "grain_detail_luminance_p10_retention_min": 0.70,
                    "grain_detail_chroma_p10_retention_min": 0.70,
                    "post_scale_preserved_ratio_min": post_scale_preserved_floor,
                    "render_luminance_range_p05_p95_min": render_luminance_range_floor,
                    "render_review_status": "reviewable",
                    "render_reviewable": true,
                    "tone_output_confidence_status": "supported_render_tonal_distribution",
                    "tone_output_review_required": false,
                    "tone_output_evidence_confidence_min": tone_output_evidence_confidence_floor,
                    "render_to_mapped_luminance_range_ratio_min": render_to_mapped_luminance_range_ratio_floor,
                    "post_chroma_compression_clipped_high_ratio_max": high_clipping_ceiling,
                    "post_chroma_compression_clipped_low_ratio_max": low_clipping_ceiling
                }
            }
        }
    });
    std::fs::write(&registry_path, accepting_registry.to_string()).unwrap();

    let suite_output_dir = tmp.path().join("suite-output");
    let suite_json = tmp.path().join("fixture-suite.json");
    let suite_md = tmp.path().join("fixture-suite.md");
    let suite_run = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-suite")
        .arg("--fixture-suite-fixture")
        .arg("logan")
        .arg("--color-mode")
        .arg("calibrated")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--strict")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(&suite_output_dir)
        .arg("--summary-json")
        .arg(&suite_json)
        .arg("--summary-md")
        .arg(&suite_md)
        .output()
        .expect("run strict grain-effect fixture suite");
    assert!(
        suite_run.status.success(),
        "strict grain-effect fixture suite failed: stderr={} stdout={}",
        String::from_utf8_lossy(&suite_run.stderr),
        String::from_utf8_lossy(&suite_run.stdout)
    );

    let suite: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&suite_json).unwrap()).unwrap();
    assert_eq!(
        suite["coverage"]["grain_reduction_enabled_fixture_count"],
        1
    );
    assert_eq!(suite["coverage"]["grain_detail_contract_fixture_count"], 1);
    assert_eq!(
        suite["coverage"]["grain_reduction_effect_contract_fixture_count"],
        1
    );
    assert_eq!(
        suite["coverage"]["render_dynamic_range_contract_fixture_count"],
        1
    );
    assert!(suite["fixtures"][0]["grain_reduction_applied_ratio"]
        .as_f64()
        .is_some_and(|value| value >= applied_floor));
    assert!(
        suite["fixtures"][0]["grain_reduction_structure_excluded_ratio"]
            .as_f64()
            .is_some_and(|value| value >= structure_excluded_floor)
    );
    assert!(
        suite["fixtures"][0]["grain_reduction_flat_luma_p95_reduction_ratio"]
            .as_f64()
            .is_some_and(|value| value >= flat_luma_floor)
    );
    assert!(
        suite["fixtures"][0]["grain_reduction_flat_chroma_p95_reduction_ratio"]
            .as_f64()
            .is_some_and(|value| value >= flat_chroma_floor)
    );
    assert!(suite["fixtures"][0]["post_scale_preserved_ratio"]
        .as_f64()
        .is_some_and(|value| value >= post_scale_preserved_floor));
    assert!(suite["fixtures"][0]["render_luminance_range_p05_p95"]
        .as_f64()
        .is_some_and(|value| value >= render_luminance_range_floor));
    assert_eq!(suite["fixtures"][0]["render_review_status"], "reviewable");
    assert_eq!(suite["fixtures"][0]["render_reviewable"], true);
    assert_eq!(
        suite["fixtures"][0]["tone_output_confidence_status"],
        "supported_render_tonal_distribution"
    );
    assert_eq!(suite["fixtures"][0]["tone_output_review_required"], false);
    assert!(suite["fixtures"][0]["tone_output_evidence_confidence"]
        .as_f64()
        .is_some_and(|value| value >= tone_output_evidence_confidence_floor));
    assert!(
        suite["fixtures"][0]["render_to_mapped_luminance_range_ratio"]
            .as_f64()
            .is_some_and(|value| value >= render_to_mapped_luminance_range_ratio_floor)
    );
    assert!(
        suite["fixtures"][0]["post_chroma_compression_clipped_high_ratio_max"]
            .as_f64()
            .is_some_and(|value| value <= high_clipping_ceiling)
    );
    assert!(
        suite["fixtures"][0]["post_chroma_compression_clipped_low_ratio_max"]
            .as_f64()
            .is_some_and(|value| value <= low_clipping_ceiling)
    );
    assert!(suite["issues"].as_array().unwrap().is_empty());

    let markdown = std::fs::read_to_string(&suite_md).unwrap();
    assert!(markdown.contains("grain applied ratio min"));
    assert!(markdown.contains("grain exact structure-excluded ratio min"));
    assert!(markdown.contains("grain flat-luma reduction min"));
    assert!(markdown.contains("grain flat-chroma reduction min"));
    assert!(markdown.contains("luma p05-p95 min"));
    assert!(markdown.contains("render status expected reviewable; actual reviewable"));
    assert!(markdown.contains("tone status expected supported_render_tonal_distribution"));
    assert!(markdown.contains("tone evidence min"));
    assert!(markdown.contains("render:mapped min"));
    assert!(markdown.contains("high clip max"));
    assert!(markdown.contains("low clip max"));

    let (expectation_field, issue_metric, measured_value) = [
        (
            "grain_reduction_applied_ratio_min",
            "grain_reduction_applied_ratio",
            applied_ratio,
        ),
        (
            "grain_reduction_structure_excluded_ratio_min",
            "grain_reduction_structure_excluded_ratio",
            structure_excluded_ratio,
        ),
        (
            "grain_reduction_flat_luma_p95_reduction_ratio_min",
            "grain_reduction_flat_luma_p95_reduction_ratio",
            flat_luma_reduction,
        ),
        (
            "grain_reduction_flat_chroma_p95_reduction_ratio_min",
            "grain_reduction_flat_chroma_p95_reduction_ratio",
            flat_chroma_reduction,
        ),
    ]
    .into_iter()
    .find(|(_, _, value)| *value < 1.0 - 1e-9)
    .expect("at least one measured grain-effect ratio must leave headroom for a rejecting floor");
    let mut rejecting_registry = accepting_registry.clone();
    rejecting_registry["fixtures"]["logan"]["expectations"][expectation_field] =
        serde_json::json!((measured_value + 1.0) * 0.5);
    std::fs::write(&registry_path, rejecting_registry.to_string()).unwrap();

    let rejecting_json = tmp.path().join("fixture-suite-rejecting.json");
    let rejecting_run = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-suite")
        .arg("--fixture-suite-fixture")
        .arg("logan")
        .arg("--color-mode")
        .arg("calibrated")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--strict")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(tmp.path().join("suite-output-rejecting"))
        .arg("--summary-json")
        .arg(&rejecting_json)
        .output()
        .expect("run rejecting grain-effect fixture suite");
    assert!(
        !rejecting_run.status.success(),
        "a grain-effect minimum above the measured result must fail"
    );
    let expected_issue = format!("fixture_suite:logan:{issue_metric}_below_expected");
    assert!(
        String::from_utf8_lossy(&rejecting_run.stderr).contains(&expected_issue),
        "rejecting suite should report {expected_issue}: {}",
        String::from_utf8_lossy(&rejecting_run.stderr)
    );
    let rejecting_summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&rejecting_json).unwrap()).unwrap();
    assert!(rejecting_summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == &expected_issue));

    let mut dynamic_range_rejecting_registry = accepting_registry;
    dynamic_range_rejecting_registry["fixtures"]["logan"]["expectations"]
        ["render_luminance_range_p05_p95_min"] =
        serde_json::json!((render_luminance_range + 1.0) * 0.5);
    dynamic_range_rejecting_registry["fixtures"]["logan"]["expectations"]
        ["render_to_mapped_luminance_range_ratio_min"] =
        serde_json::json!(render_to_mapped_luminance_range_ratio + 0.1);
    dynamic_range_rejecting_registry["fixtures"]["logan"]["expectations"]["render_review_status"] =
        serde_json::json!("review_required_tone_output");
    dynamic_range_rejecting_registry["fixtures"]["logan"]["expectations"]["render_reviewable"] =
        serde_json::json!(false);
    dynamic_range_rejecting_registry["fixtures"]["logan"]["expectations"]
        ["tone_output_confidence_status"] =
        serde_json::json!("review_required_collapsed_render_luminance_range");
    dynamic_range_rejecting_registry["fixtures"]["logan"]["expectations"]
        ["tone_output_review_required"] = serde_json::json!(true);
    std::fs::write(&registry_path, dynamic_range_rejecting_registry.to_string()).unwrap();
    let dynamic_range_rejecting_json = tmp
        .path()
        .join("fixture-suite-dynamic-range-rejecting.json");
    let dynamic_range_rejecting_run = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-suite")
        .arg("--fixture-suite-fixture")
        .arg("logan")
        .arg("--color-mode")
        .arg("calibrated")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--strict")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(tmp.path().join("suite-output-dynamic-range-rejecting"))
        .arg("--summary-json")
        .arg(&dynamic_range_rejecting_json)
        .output()
        .expect("run rejecting dynamic-range fixture suite");
    assert!(
        !dynamic_range_rejecting_run.status.success(),
        "a luminance-range minimum above the measured result must fail"
    );
    let dynamic_range_issue = "fixture_suite:logan:render_luminance_range_p05_p95_below_expected";
    let relative_range_issue =
        "fixture_suite:logan:render_to_mapped_luminance_range_ratio_below_expected";
    let decision_issues = [
        "fixture_suite:logan:render_review_status_mismatch",
        "fixture_suite:logan:render_reviewable_mismatch",
        "fixture_suite:logan:tone_output_confidence_status_mismatch",
        "fixture_suite:logan:tone_output_review_required_mismatch",
    ];
    assert!(
        String::from_utf8_lossy(&dynamic_range_rejecting_run.stderr).contains(dynamic_range_issue),
        "rejecting suite should report {dynamic_range_issue}: {}",
        String::from_utf8_lossy(&dynamic_range_rejecting_run.stderr)
    );
    let dynamic_range_rejecting_summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&dynamic_range_rejecting_json).unwrap())
            .unwrap();
    assert!(dynamic_range_rejecting_summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == dynamic_range_issue));
    assert!(dynamic_range_rejecting_summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == relative_range_issue));
    for issue in decision_issues {
        assert!(
            dynamic_range_rejecting_summary["issues"]
                .as_array()
                .unwrap()
                .iter()
                .any(|actual| actual == issue),
            "missing dynamic-range decision issue {issue}: {}",
            dynamic_range_rejecting_summary["issues"]
        );
    }
}

#[test]
fn test_validate_cli_fixture_suite_proves_three_scan_stitch_normalization_contract() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input_dir = tmp.path().join("inputs");
    std::fs::create_dir_all(&input_dir).unwrap();
    let panorama = textured_positive_panorama(180, 1_000);
    let left = panorama.slice(s![.., 0..420, ..]).to_owned();
    let middle = panorama.slice(s![.., 290..710, ..]).to_owned();
    let right = panorama.slice(s![.., 580..1_000, ..]).to_owned();
    let right_path = input_dir.join("right.tiff");
    let left_path = input_dir.join("left.tiff");
    let middle_path = input_dir.join("middle.tiff");
    scanstitch::tiff_io::save_tiff_u16(&right, &right_path).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&left, &left_path).unwrap();
    scanstitch::tiff_io::save_tiff_u16(&middle, &middle_path).unwrap();

    let registry_path = tmp.path().join("fixtures.json");
    let baseline_path = tmp.path().join("stitch-summary-baseline.json");
    let direct_output_dir = tmp.path().join("direct-output");
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "fixtures": {
                "logan": {
                    "component1": right_path,
                    "component2": left_path,
                    "additional_components": [{"path": middle_path}],
                    "output_dir": direct_output_dir,
                    "input_mode": "positive",
                    "bit_depth": 14,
                    "force_stitch": true,
                    "film_stock": "Synthetic positive",
                    "scene_tags": ["panorama", "overlap-detail"],
                    "exposure_tags": ["normal-exposure"],
                    "calibration_case": "uncalibrated-image-derived"
                }
            }
        })
        .to_string(),
    )
    .unwrap();

    let direct = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture")
        .arg("logan")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--quiet")
        .arg("--write-summary-baseline")
        .arg(&baseline_path)
        .output()
        .expect("run direct three-scan fixture baseline");
    assert!(
        direct.status.success(),
        "direct three-scan fixture failed: stderr={} stdout={}",
        String::from_utf8_lossy(&direct.stderr),
        String::from_utf8_lossy(&direct.stdout)
    );

    let direct_summary: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(direct_output_dir.join("summary.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(direct_summary["stitch"]["decision"], "accepted_sequence");
    assert_eq!(
        direct_summary["stitch"]["inferred_order"],
        serde_json::json!([2, 3, 1])
    );
    let exposure = &direct_summary["stitch"]["seam_exposure_correction"];
    let exposure_model = exposure["model"].as_str().unwrap();
    assert!(matches!(
        exposure_model,
        "sequence_mixed"
            | "identity"
            | "gain_only_scalar"
            | "gain_only_rgb"
            | "gain_offset_rgb"
            | "gain_spatial_y_rgb"
            | "gain_offset_spatial_y_rgb"
            | "gain_spatial_xy_rgb"
            | "gain_offset_spatial_xy_rgb"
            | "gain_spatial_quadratic_xy_rgb"
            | "gain_offset_spatial_quadratic_xy_rgb"
    ));
    let expected_held_out_validation = exposure_model != "identity";
    assert_eq!(
        exposure["held_out_validation_passed"].as_bool(),
        Some(expected_held_out_validation)
    );
    let offset_abs_max = exposure["offset_rgb_normalized"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap().abs())
        .fold(0.0, f64::max);
    assert!(offset_abs_max <= 0.05);

    let blend = &direct_summary["stitch"]["seam_blend"];
    assert_eq!(blend["mode"], "seam_aware_multiband");
    assert_eq!(blend["applied"], true);
    assert_eq!(blend["review_required"], false);
    let detail = &blend["detail_consistency"];
    assert_eq!(detail["review_required"], false);
    let supported_scale_count = detail["minimum_supported_scale_count"].as_u64().unwrap() as usize;
    let maximum_energy_ratio = detail["maximum_symmetric_energy_ratio"].as_f64().unwrap();
    let gradient_ratio = blend["output_to_source_seam_gradient_ratio"]
        .as_f64()
        .unwrap();
    let overlap_p95 = blend["overlap_p95_abs_difference"].as_f64().unwrap();
    assert!(supported_scale_count >= 2);
    assert!((1.0..=2.0).contains(&maximum_energy_ratio));
    assert!((0.0..=1.05).contains(&gradient_ratio));
    assert!((0.0..1.0).contains(&overlap_p95));

    let accepting_registry = serde_json::json!({
        "coverage_requirements": {
            "min_n_component_fixtures": 1,
            "min_stitch_normalization_contract_fixtures": 1
        },
        "fixtures": {
            "logan": {
                "component1": right_path,
                "component2": left_path,
                "additional_components": [{"path": middle_path}],
                "input_mode": "positive",
                "bit_depth": 14,
                "force_stitch": true,
                "summary_baseline": baseline_path,
                "film_stock": "Synthetic positive",
                "scene_tags": ["panorama", "overlap-detail"],
                "exposure_tags": ["normal-exposure"],
                "calibration_case": "uncalibrated-image-derived",
                "expectations": {
                    "stitch_decision": "accepted_sequence",
                    "inferred_component_order": [2, 3, 1],
                    "seam_exposure_model": exposure_model,
                    "seam_exposure_held_out_validation_passed": expected_held_out_validation,
                    "seam_exposure_offset_normalized_abs_max": 0.05,
                    "seam_blend_required": true,
                    "seam_blend_mode": "seam_aware_multiband",
                    "seam_blend_review_required": false,
                    "seam_detail_review_required": false,
                    "seam_detail_supported_scale_count_min": 2,
                    "seam_detail_max_symmetric_energy_ratio_max": 2.0,
                    "seam_gradient_ratio_max": 1.05,
                    "seam_overlap_p95_abs_difference_max": 0.99,
                    "render_review_status": "review_required_color",
                    "render_reviewable": false
                }
            }
        }
    });
    std::fs::write(&registry_path, accepting_registry.to_string()).unwrap();

    let suite_json = tmp.path().join("fixture-suite.json");
    let suite_run = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-suite")
        .arg("--fixture-suite-fixture")
        .arg("logan")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--strict")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(tmp.path().join("suite-output"))
        .arg("--summary-json")
        .arg(&suite_json)
        .output()
        .expect("run strict three-scan stitch-normalization fixture suite");
    assert!(
        suite_run.status.success(),
        "strict three-scan fixture suite failed: stderr={} stdout={}",
        String::from_utf8_lossy(&suite_run.stderr),
        String::from_utf8_lossy(&suite_run.stdout)
    );
    let suite: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&suite_json).unwrap()).unwrap();
    assert_eq!(
        suite["coverage"]["n_component_validation_ready_fixture_count"],
        1
    );
    assert_eq!(
        suite["coverage"]["stitch_normalization_contract_fixture_count"],
        1
    );
    assert_eq!(
        suite["coverage"]["fixtures"][0]["stitch_normalization_contract_complete"],
        true
    );
    assert_eq!(suite["fixtures"][0]["stitch_decision"], "accepted_sequence");
    assert_eq!(
        suite["fixtures"][0]["inferred_component_order"],
        serde_json::json!([2, 3, 1])
    );
    assert_eq!(suite["fixtures"][0]["seam_exposure_model"], exposure_model);
    assert_eq!(
        suite["fixtures"][0]["seam_blend_mode"],
        "seam_aware_multiband"
    );
    assert_eq!(suite["fixtures"][0]["seam_blend_review_required"], false);
    assert_eq!(suite["fixtures"][0]["seam_detail_review_required"], false);
    assert!(suite["issues"].as_array().unwrap().is_empty());

    let mut rejecting_registry = accepting_registry;
    rejecting_registry["fixtures"]["logan"]["expectations"]
        ["seam_detail_supported_scale_count_min"] = serde_json::json!(supported_scale_count + 1);
    std::fs::write(&registry_path, rejecting_registry.to_string()).unwrap();
    let rejecting_json = tmp.path().join("fixture-suite-rejecting.json");
    let rejecting_run = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-registry")
        .arg(&registry_path)
        .arg("--fixture-suite")
        .arg("--fixture-suite-fixture")
        .arg("logan")
        .arg("--quality-mode")
        .arg("fast")
        .arg("--strict")
        .arg("--quiet")
        .arg("--output-dir")
        .arg(tmp.path().join("suite-output-rejecting"))
        .arg("--summary-json")
        .arg(&rejecting_json)
        .output()
        .expect("run rejecting three-scan stitch-normalization fixture suite");
    assert!(
        !rejecting_run.status.success(),
        "a detail-scale floor above the measured N-scan result must fail"
    );
    let expected_issue = "fixture_suite:logan:seam_detail_supported_scale_count_below_expected";
    assert!(
        String::from_utf8_lossy(&rejecting_run.stderr).contains(expected_issue),
        "rejecting suite should report {expected_issue}: {}",
        String::from_utf8_lossy(&rejecting_run.stderr)
    );
    let rejecting_summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&rejecting_json).unwrap()).unwrap();
    assert!(rejecting_summary["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == expected_issue));
}

#[test]
fn test_validate_cli_fixture_suite_writes_missing_summary_baseline() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input_dir = tmp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();

    let comp =
        synthetic::film_negative_image(160, 300, 10, 30, [12000, 7000, 3000], [5400, 4300, 3500]);
    let path1 = input_dir.join("single-frame.tiff");
    scanstitch::tiff_io::save_tiff_u16(&comp, &path1).unwrap();

    let baseline_path = tmp.path().join("baselines/generated.json");
    let other_baseline_path = tmp.path().join("baselines/other-generated.json");
    let registry_path = tmp.path().join("fixtures.json");
    std::fs::write(
        &registry_path,
        serde_json::json!({
            "coverage_requirements": {
                "min_fixtures": 1,
                "min_component_pairs": 1,
                "min_summary_baselines": 1
            },
            "fixtures": {
                "logan": {
                    "component1": path1,
                    "component2": path1,
                    "output_dir": tmp.path().join("registry-output"),
                    "input_mode": "negative",
                    "bit_depth": 14,
                    "force_no_stitch": true,
                    "summary_baseline": baseline_path,
                    "expectations": {
                        "stitch_decision": "skipped_pre_score",
                        "base_estimate_source": "working_edges",
                        "output_color_space": "linear_prophoto_rgb_d50"
                    }
                },
                "other": {
                    "component1": path1,
                    "component2": path1,
                    "output_dir": tmp.path().join("registry-output-other"),
                    "input_mode": "negative",
                    "bit_depth": 14,
                    "force_no_stitch": true,
                    "summary_baseline": other_baseline_path,
                    "expectations": {
                        "stitch_decision": "skipped_pre_score",
                        "base_estimate_source": "working_edges",
                        "output_color_space": "linear_prophoto_rgb_d50"
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
        .arg("--fixture-suite-fixture")
        .arg("logan")
        .arg("--write-fixture-suite-baselines")
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
        .expect("run scanstitch-validate fixture-suite baseline writer");

    assert!(
        output.status.success(),
        "fixture suite baseline writer failed: status={} stderr={} stdout={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(baseline_path.exists());
    assert!(!other_baseline_path.exists());

    let baseline: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&baseline_path).unwrap()).unwrap();
    assert_eq!(baseline["fixture"], "logan");
    assert_eq!(baseline["stitch"]["decision"], "skipped_pre_score");
    assert_eq!(
        baseline["render"]["output_color_space"],
        "linear_prophoto_rgb_d50"
    );

    let suite: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&suite_json).unwrap()).unwrap();
    assert_eq!(suite["status"], "review_required");
    assert_eq!(suite["fixture_count"], serde_json::json!(1));
    assert_eq!(suite["coverage"]["fixtures"].as_array().unwrap().len(), 2);
    assert_eq!(suite["fixtures"][0]["status"], "review_required");
    assert_eq!(suite["fixtures"][0]["name"], "logan");
    assert_eq!(
        suite["fixtures"][0]["summary_baseline_write_status"],
        "written"
    );
    assert_eq!(
        suite["fixtures"][0]["summary_baseline_written_path"],
        baseline_path.display().to_string()
    );
    assert_eq!(
        suite["fixtures"][0]["summary_baseline_status"],
        "comparable"
    );
    assert!(suite["fixtures"][0]["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| issue == "fixture_suite:logan:coverage_not_validation_ready"));

    let per_fixture_summary = suite_output_dir.join("logan/summary.json");
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(per_fixture_summary).unwrap()).unwrap();
    assert_eq!(
        summary["summary_baseline_comparison"]["status"],
        "comparable"
    );

    let markdown = std::fs::read_to_string(&suite_md).unwrap();
    assert!(markdown.contains("write written"));
}

#[test]
fn test_validate_cli_rejects_fixture_suite_baseline_writer_in_strict_mode() {
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-suite")
        .arg("--write-fixture-suite-baselines")
        .arg("--strict")
        .arg("--quiet")
        .output()
        .expect("run scanstitch-validate invalid fixture-suite baseline writer");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--write-fixture-suite-baselines is a corpus-building mode"));
}

#[test]
fn test_validate_cli_rejects_fixture_suite_baseline_overwrite_without_writer() {
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-suite")
        .arg("--overwrite-fixture-suite-baselines")
        .arg("--quiet")
        .output()
        .expect("run scanstitch-validate invalid fixture-suite baseline overwrite");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr
        .contains("--overwrite-fixture-suite-baselines requires --write-fixture-suite-baselines"));
}

#[test]
fn test_validate_cli_rejects_unknown_fixture_suite_selector() {
    let output = Command::new(env!("CARGO_BIN_EXE_scanstitch-validate"))
        .arg("--fixture-suite")
        .arg("--fixture-suite-fixture")
        .arg("missing-fixture")
        .arg("--quiet")
        .output()
        .expect("run scanstitch-validate invalid fixture-suite selector");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr
        .contains("--fixture-suite-fixture did not match registry fixture(s): missing-fixture"));
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
                        "neutral_safety_rescue_applied": true,
                        "neutral_safety_rescue_preserved_ratio_gain_min": 1.0,
                        "neutral_safety_rescue_midtone_saturation_p95_reduction_min": 1.0,
                        "neutral_safety_rescue_reason_contains": "missing-neutral-rescue-reason",
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
    assert!(suite["issues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|issue| { issue == "fixture_suite:logan:neutral_safety_rescue_applied_mismatch" }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:neutral_safety_rescue_preserved_ratio_gain_below_expected"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue
            == "fixture_suite:logan:neutral_safety_rescue_midtone_saturation_p95_reduction_below_expected"
    }));
    assert!(suite["issues"].as_array().unwrap().iter().any(|issue| {
        issue == "fixture_suite:logan:neutral_safety_rescue_reason_missing_expected_text"
    }));
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
    let tone = current_report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "tone_mapping")
        .unwrap();
    tone.metrics["tone_output_confidence_status"] =
        serde_json::json!("review_required_collapsed_render_luminance_range");
    tone.metrics["tone_output_review_required"] = serde_json::json!(true);
    tone.metrics["tone_output_evidence_confidence"] = serde_json::json!(0.0);
    tone.metrics["render_luminance_range_p05_p95"] = serde_json::json!(0.01);
    tone.metrics["render_to_mapped_luminance_range_ratio"] = serde_json::json!(0.01);
    tone.metrics["maximum_post_tone_high_clip_ratio"] = serde_json::json!(0.60);
    tone.metrics["maximum_post_tone_low_clip_ratio"] = serde_json::json!(0.55);
    let save = current_report
        .phases
        .iter_mut()
        .find(|phase| phase.name == "save")
        .unwrap();
    save.metrics["render_review_status"] = serde_json::json!("review_required_tone_output");
    save.metrics["render_reviewable"] = serde_json::json!(false);
    save.metrics["tone_output_review_required"] = serde_json::json!(true);
    save.metrics["tone_output_confidence_status"] =
        serde_json::json!("review_required_collapsed_render_luminance_range");
    save.metrics["tone_output_evidence_confidence"] = serde_json::json!(0.0);

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
    for issue in [
        "render_input_source_changed",
        "render_review_status_changed",
        "render_reviewable_changed",
        "tone_output_confidence_status_changed",
        "tone_output_review_required_changed",
        "tone_output_evidence_confidence_regressed",
        "tone_output_render_luminance_range_regressed",
        "tone_output_range_retention_regressed",
        "tone_output_high_clipping_regressed",
        "tone_output_low_clipping_regressed",
    ] {
        assert!(comparison.issues.iter().any(|actual| actual == issue));
    }
    assert_eq!(comparison.luma_residual_p95_ratio, Some(1.0));
}

#[test]
fn test_render_comparison_marks_identical_reports_comparable() {
    let tmp = tempfile::TempDir::new().unwrap();
    let output_path = tmp.path().join("output.tiff");
    scanstitch::tiff_io::save_tiff_f64_linear_prophoto(&small_rgb_image(), &output_path).unwrap();
    let mut report = fixture_report();
    set_report_output_path(&mut report, &output_path);
    clear_stale_artifacts(&mut report);
    let baseline = scanstitch::validation::summarize_report("baseline", &report);
    let current = scanstitch::validation::summarize_report("current", &report);
    let comparison = scanstitch::validation::compare_render_summaries(
        "baseline/report.json",
        &baseline.render,
        Some("current/report.json".to_string()),
        &current.render,
    );

    assert_eq!(
        comparison.status, "comparable",
        "unexpected issues: {:?}",
        comparison.issues
    );
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

fn write_validation_fixture_library_with_measured_negative_response(
    root: &Path,
) -> std::path::PathBuf {
    let library_dir = root.join("calibration");
    std::fs::create_dir_all(library_dir.join("scanners")).unwrap();
    std::fs::create_dir_all(library_dir.join("rolls")).unwrap();
    std::fs::write(
        library_dir.join("scanners/scanner.json"),
        validation_fixture_scanner_profile_json(),
    )
    .unwrap();
    let mut roll: serde_json::Value =
        serde_json::from_str(&validation_fixture_roll_profile_json()).unwrap();
    roll["negative_response"] = serde_json::json!({
        "model_id": "synthetic-validation-response-v1",
        "scanner_density_to_layer_density": [
            [1.0, -0.08, 0.01],
            [-0.04, 1.0, -0.05],
            [0.01, -0.04, 1.0]
        ],
        "characteristic_curves": {
            "red": [[-2.0, -2.0], [0.0, 0.0], [0.25, 0.18], [0.70, 0.78], [1.30, 1.55], [5.0, 5.0]],
            "green": [[-2.0, -2.0], [0.0, 0.0], [0.25, 0.18], [0.70, 0.78], [1.30, 1.55], [5.0, 5.0]],
            "blue": [[-2.0, -2.0], [0.0, 0.0], [0.25, 0.18], [0.70, 0.78], [1.30, 1.55], [5.0, 5.0]]
        },
        "white_anchor_percentile": 0.995,
        "confidence": 0.94,
        "validation": {
            "held_out_patch_count": 24,
            "delta_e00_rms": 1.8,
            "delta_e00_max": 4.9,
            "unit_slope_delta_e00_rms": 8.2
        }
    });
    std::fs::write(
        library_dir.join("rolls/roll.json"),
        serde_json::to_string_pretty(&roll).unwrap(),
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
        inputs: vec![component1.to_path_buf(), component2.to_path_buf()],
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
        white_balance: scanstitch::cli::WhiteBalanceArgs::default(),
        geometry: scanstitch::cli::GeometryArgs::default(),
        grain: scanstitch::cli::GrainReductionArgs::default(),
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
        require_reviewable: false,
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
            "colorspace_mapping_strategy": "neutral_balance_weak_anchor_fallback",
            "render_review_status": "reviewable",
            "render_reviewable": true
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
            "calibration_acceptance_status": "rejected_reference_fit",
            "calibration_color_mapping_applied": false,
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
            "shadow_chroma_compressed_ratio": 0.002,
            "tone_output_confidence_status": "supported_render_tonal_distribution",
            "tone_output_review_required": false,
            "tone_output_evidence_confidence": 1.0,
            "render_luminance_range_p05_p95": 0.90,
            "render_to_mapped_luminance_range_ratio": 1.0588235294117647,
            "maximum_post_tone_high_clip_ratio": 0.002,
            "maximum_post_tone_low_clip_ratio": 0.001
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
    assert!(!comparison.calibration_color_mapping_applied_changed);
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
fn test_summary_baseline_comparison_flags_seam_model_and_spatial_acceptance_drift() {
    let mut report = fixture_report();
    let _debug_artifacts = write_report_color_debug_artifacts(&mut report);
    let summary = scanstitch::validation::summarize_report("logan", &report);
    let mut baseline = scanstitch::validation::tracked_baseline_from_summary(&summary);
    baseline.stitch.seam_exposure_model = Some("identity".to_string());
    baseline.stitch.seam_exposure_spatial_2d_gain_accepted = Some(true);
    baseline
        .stitch
        .seam_exposure_spatial_2d_gain_offset_accepted = Some(true);
    baseline
        .stitch
        .seam_exposure_spatial_quadratic_gain_accepted = Some(true);
    baseline
        .stitch
        .seam_exposure_spatial_quadratic_gain_offset_accepted = Some(true);
    baseline.stitch.seam_blend_review_required = Some(true);
    baseline.stitch.seam_detail_review_required = Some(true);
    baseline.stitch.seam_detail_max_symmetric_energy_ratio = Some(1.50);

    let comparison =
        scanstitch::validation::compare_summary_baseline("baseline.json", &baseline, &summary);

    assert_eq!(comparison.status, "review_required");
    assert!(comparison.seam_exposure_model_changed);
    assert!(comparison.seam_exposure_spatial_2d_gain_acceptance_changed);
    assert!(comparison.seam_exposure_spatial_2d_gain_offset_acceptance_changed);
    assert!(comparison.seam_exposure_spatial_quadratic_gain_acceptance_changed);
    assert!(comparison.seam_exposure_spatial_quadratic_gain_offset_acceptance_changed);
    assert!(comparison.seam_blend_review_required_changed);
    assert!(comparison.seam_detail_review_required_changed);
    assert_eq!(
        comparison.seam_detail_max_symmetric_energy_ratio_delta,
        Some(1.08 - 1.50)
    );
    assert!(comparison
        .issues
        .contains(&"seam_exposure_model_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"seam_exposure_spatial_2d_gain_acceptance_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"seam_exposure_spatial_2d_gain_offset_acceptance_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"seam_exposure_spatial_quadratic_gain_acceptance_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"seam_exposure_spatial_quadratic_gain_offset_acceptance_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"seam_blend_review_required_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"seam_detail_review_required_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"seam_detail_energy_ratio_changed_materially".to_string()));
}

#[test]
fn test_summary_baseline_comparison_flags_tracked_color_tone_and_grain_drift() {
    let mut report = fixture_report();
    let _debug_artifacts = write_report_color_debug_artifacts(&mut report);
    let mut summary = scanstitch::validation::summarize_report("logan", &report);
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
    baseline.colorspace.calibration_color_mapping_applied = Some(true);
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
    baseline
        .tone
        .adaptive_vibrance_skin_memory_protection_enabled = Some(false);
    baseline.tone.adaptive_vibrance_skin_memory_protection_space =
        Some("legacy_rgb_rule".to_string());
    baseline.tone.adaptive_vibrance_skin_memory_protected_ratio = Some(0.90);
    baseline.tone.adaptive_vibrance_skin_memory_mean_protection = Some(0.10);
    baseline
        .tone
        .adaptive_vibrance_preferred_memory_color_guard_enabled = Some(false);
    baseline
        .tone
        .adaptive_vibrance_preferred_memory_color_guard_space = Some("legacy_rgb_rule".to_string());
    baseline
        .tone
        .adaptive_vibrance_preferred_memory_color_guard_reference =
        Some("legacy_reference".to_string());
    baseline
        .tone
        .adaptive_vibrance_preferred_memory_color_matched_ratio = Some(0.90);
    baseline
        .tone
        .adaptive_vibrance_preferred_memory_color_limited_ratio = Some(0.70);
    baseline
        .tone
        .adaptive_vibrance_preferred_memory_color_mean_scale_reduction = Some(0.20);
    baseline.tone.preferred_skin_rendering_enabled = Some(false);
    baseline.tone.preferred_skin_rendering_space = Some("legacy_rgb_rule".to_string());
    baseline.tone.preferred_skin_rendering_preference_reference =
        Some("legacy_reference".to_string());
    baseline.tone.preferred_skin_rendering_adjusted_ratio = Some(0.90);
    baseline.tone.preferred_skin_rendering_mean_delta_e_ab = Some(2.5);
    baseline.grain.luma_residual_p95 = Some(0.003);
    baseline.grain.detail_review_required = Some(true);
    baseline.grain.detail_decision_supported = Some(false);
    baseline.grain.luminance_p10_retention = Some(1.20);
    baseline.grain.chroma_p10_retention = Some(1.20);
    summary.render.render_review_status = Some("review_required_tone_output".to_string());
    summary.render.render_reviewable = Some(false);
    summary.tone.tone_output_confidence_status =
        Some("review_required_collapsed_render_luminance_range".to_string());
    summary.tone.tone_output_review_required = Some(true);
    summary.tone.tone_output_evidence_confidence = Some(0.0);
    summary.tone.render_luminance_range_p05_p95 = Some(0.01);
    summary.tone.render_to_mapped_luminance_range_ratio = Some(0.01);
    summary.tone.maximum_post_tone_high_clip_ratio = Some(0.60);
    summary.tone.maximum_post_tone_low_clip_ratio = Some(0.55);

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
        .contains(&"calibration_color_mapping_applied_changed".to_string()));
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
        .contains(&"adaptive_vibrance_skin_memory_protection_enabled_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"adaptive_vibrance_skin_memory_protection_space_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"adaptive_vibrance_skin_memory_protected_ratio_changed_materially".to_string()));
    assert!(comparison
        .issues
        .contains(&"adaptive_vibrance_skin_memory_mean_protection_changed_materially".to_string()));
    assert!(comparison
        .issues
        .contains(&"adaptive_vibrance_preferred_memory_color_guard_enabled_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"adaptive_vibrance_preferred_memory_color_guard_space_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"adaptive_vibrance_preferred_memory_color_guard_reference_changed".to_string()));
    assert!(comparison.issues.contains(
        &"adaptive_vibrance_preferred_memory_color_matched_ratio_changed_materially".to_string()
    ));
    assert!(comparison.issues.contains(
        &"adaptive_vibrance_preferred_memory_color_limited_ratio_changed_materially".to_string()
    ));
    assert!(comparison.issues.contains(
        &"adaptive_vibrance_preferred_memory_color_mean_scale_reduction_changed_materially"
            .to_string()
    ));
    assert!(comparison
        .issues
        .contains(&"preferred_skin_rendering_enabled_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"preferred_skin_rendering_space_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"preferred_skin_rendering_preference_reference_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"preferred_skin_rendering_adjusted_ratio_changed_materially".to_string()));
    assert!(comparison
        .issues
        .contains(&"preferred_skin_rendering_mean_delta_e_ab_changed_materially".to_string()));
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
    for issue in [
        "render_review_status_changed",
        "render_reviewable_changed",
        "tone_output_confidence_status_changed",
        "tone_output_review_required_changed",
        "tone_output_evidence_confidence_regressed",
        "tone_output_render_luminance_range_regressed",
        "tone_output_range_retention_regressed",
        "tone_output_high_clipping_regressed",
        "tone_output_low_clipping_regressed",
    ] {
        assert!(comparison.issues.iter().any(|actual| actual == issue));
    }
    assert!(comparison
        .issues
        .contains(&"luma_residual_p95_changed_materially".to_string()));
    assert!(comparison
        .issues
        .contains(&"grain_detail_review_required_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"grain_detail_decision_supported_changed".to_string()));
    assert!(comparison
        .issues
        .contains(&"grain_detail_luminance_retention_regressed".to_string()));
    assert!(comparison
        .issues
        .contains(&"grain_detail_chroma_retention_regressed".to_string()));
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
            - 1.283527)
            .abs()
            < 0.000001
    );
    assert!(
        (baseline["colorspace"]["color_fidelity_score"]
            .as_f64()
            .expect("color fidelity score")
            - 0.457071)
            .abs()
            < 0.000001
    );
    assert!(
        (baseline["colorspace"]["hue_linearity_score"]
            .as_f64()
            .expect("hue linearity baseline")
            - 0.835089)
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
    assert_eq!(
        baseline["colorspace"]["calibration_color_mapping_applied"],
        false
    );
    assert!(
        baseline["grain"]["luma_residual_p95"]
            .as_f64()
            .expect("luma residual baseline")
            > 0.0
    );
    assert_eq!(baseline["grain"]["detail_review_required"], false);
    assert_eq!(baseline["grain"]["detail_decision_supported"], true);
    assert_eq!(baseline["grain"]["luminance_p10_retention"], 1.0);
    assert_eq!(baseline["grain"]["chroma_p10_retention"], 1.0);
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
    let current: PipelineReport = serde_json::from_str(include_str!(
        "../docs/report-examples/current-report-minimal.json"
    ))
    .unwrap();
    let current_metadata = current.metadata.as_ref().unwrap();
    assert_eq!(current_metadata.report_schema_version, 4);
    assert_eq!(
        current_metadata.binary_identity_status.as_deref(),
        Some("verified_sha256")
    );
    assert!(current_metadata.binary_sha256.is_some());
    let legacy: PipelineReport = serde_json::from_str(include_str!(
        "../docs/report-examples/old-report-sparse.json"
    ))
    .unwrap();
    assert!(legacy
        .metadata
        .as_ref()
        .is_none_or(|metadata| metadata.binary_sha256.is_none()));
    let report_schema_doc = include_str!("../docs/report-schema.md");
    assert!(report_schema_doc.contains("CIEDE2000"));
    assert_eq!(schema["required"], serde_json::json!(["phases"]));
    assert_eq!(
        schema["properties"]["metadata"]["properties"]["binary_sha256"]["pattern"],
        "^[A-Fa-f0-9]{64}$"
    );
    assert_eq!(
        schema["properties"]["phases"]["items"]["allOf"][0]["if"]["properties"]["name"]["const"],
        "colorspace_mapping"
    );
    assert_eq!(
        schema["properties"]["phases"]["items"]["allOf"][0]["then"]["properties"]["metrics"]
            ["$ref"],
        "#/$defs/colorspaceMappingMetrics"
    );
    assert_eq!(
        schema["properties"]["phases"]["items"]["allOf"][3]["if"]["properties"]["name"]["const"],
        "load"
    );
    assert_eq!(
        schema["properties"]["phases"]["items"]["allOf"][3]["then"]["properties"]["metrics"]
            ["$ref"],
        "#/$defs/loadMetrics"
    );
    assert_eq!(
        schema["properties"]["phases"]["items"]["allOf"][4]["if"]["properties"]["name"]["const"],
        "fastica"
    );
    assert_eq!(
        schema["properties"]["phases"]["items"]["allOf"][4]["then"]["properties"]["metrics"]
            ["$ref"],
        "#/$defs/fasticaMetrics"
    );
    assert_eq!(
        schema["properties"]["phases"]["items"]["allOf"][5]["if"]["properties"]["name"]["const"],
        "density_inversion"
    );
    assert_eq!(
        schema["properties"]["phases"]["items"]["allOf"][5]["then"]["properties"]["metrics"]
            ["$ref"],
        "#/$defs/densityInversionMetrics"
    );
    assert_eq!(
        schema["properties"]["phases"]["items"]["allOf"][6]["if"]["properties"]["name"]["const"],
        "save"
    );
    assert_eq!(
        schema["properties"]["phases"]["items"]["allOf"][6]["then"]["properties"]["metrics"]
            ["$ref"],
        "#/$defs/saveMetrics"
    );
    assert!(
        schema["$defs"]["densityInversionMetrics"]["properties"]["operation_status"]["enum"]
            .as_array()
            .is_some_and(|statuses| statuses
                .iter()
                .any(|status| status == "held_out_measured_response_supported"))
    );
    assert_eq!(
        schema["$defs"]["densityInversionMetrics"]["properties"]
            ["negative_response_model_confidence"]["maximum"],
        1
    );
    assert_eq!(
        schema["$defs"]["densityInversionMetrics"]["allOf"][2]["then"]["properties"]
            ["negative_response_model_confidence"]["const"],
        0
    );
    assert!(
        schema["$defs"]["fasticaMetrics"]["properties"]["operation_status"]["enum"]
            .as_array()
            .is_some_and(|statuses| statuses.iter().any(|status| {
                status == "numerically_converged_physical_separation_unvalidated"
            }))
    );
    assert_eq!(
        schema["$defs"]["fasticaMetrics"]["properties"]["numerical_convergence_confidence"]
            ["maximum"],
        1
    );
    assert_eq!(
        schema["$defs"]["loadMetrics"]["properties"]["components"]["items"]["$ref"],
        "#/$defs/loadComponent"
    );
    assert_eq!(
        schema["$defs"]["loadComponent"]["properties"]["decode"]["$ref"],
        "#/$defs/loadDecodeMetrics"
    );
    assert_eq!(
        schema["$defs"]["loadDecodeMetrics"]["properties"]["orientation"]["$ref"],
        "#/$defs/inputOrientationDiagnostics"
    );
    assert_eq!(
        schema["$defs"]["loadDecodeMetrics"]["properties"]["orientation_correction"]["$ref"],
        "#/$defs/orientationCorrectionDiagnostics"
    );
    assert_eq!(
        schema["$defs"]["loadDecodeMetrics"]["properties"]["decode_fidelity"]["$ref"],
        "#/$defs/loadDecodeFidelity"
    );
    assert_eq!(
        schema["$defs"]["loadDecodeFidelity"]["properties"]["confidence"]["maximum"],
        1
    );
    assert!(
        schema["$defs"]["loadDecodeFidelity"]["properties"]["status"]["enum"]
            .as_array()
            .is_some_and(|statuses| statuses
                .iter()
                .any(|status| status == "limited_declared_decode_fidelity"))
    );
    assert!(
        schema["$defs"]["loadMetrics"]["properties"]["decode_fidelity_status"]["enum"]
            .as_array()
            .is_some_and(|statuses| statuses
                .iter()
                .any(|status| status == "mixed_component_decode_fidelity"))
    );
    assert_eq!(
        schema["$defs"]["loadDecodeMetrics"]["properties"]["decoded_pixel_sha256"]["pattern"],
        "^[A-Fa-f0-9]{64}$"
    );
    for field in ["source_min", "source_max", "working_min", "working_max"] {
        assert_eq!(
            schema["$defs"]["loadDecodeMetrics"]["properties"][field]["$ref"],
            "#/$defs/integerVector3"
        );
    }
    assert_eq!(
        schema["$defs"]["loadDecodeMetrics"]["properties"]
            ["source_orientation_materialized_decoded_pixel_sha256"]["pattern"],
        "^[A-Fa-f0-9]{64}$"
    );
    assert!(
        schema["$defs"]["orientationCorrectionDiagnostics"]["properties"]["requested"]["enum"]
            .as_array()
            .is_some_and(|values| values.iter().any(|value| value == "rotate-180"))
    );
    assert_eq!(
        schema["$defs"]["orientationCorrectionDiagnostics"]["properties"]["effective_tag_value"]
            ["maximum"],
        8
    );
    assert_eq!(
        schema["$defs"]["inputOrientationDiagnostics"]["properties"]["tag_value"]["type"],
        serde_json::json!(["integer", "null"])
    );
    for field in [
        "transform",
        "applied",
        "source_width",
        "source_height",
        "output_width",
        "output_height",
        "reason",
    ] {
        assert!(schema["$defs"]["inputOrientationDiagnostics"]["properties"][field].is_object());
    }
    assert!(
        schema["$defs"]["colorspaceMappingMetrics"]["properties"]["candidate_quality_scores"]
            .is_object()
    );
    assert_eq!(
        schema["$defs"]["colorspaceMappingMetrics"]["properties"]["neutral_safety_rescue"]["$ref"],
        "#/$defs/neutralSafetyRescue"
    );
    assert!(
        schema["$defs"]["colorspaceMappingMetrics"]["properties"]["mapping_strategy"]["enum"]
            .as_array()
            .is_some_and(|strategies| strategies
                .iter()
                .any(|strategy| strategy == "neutral_balance_evidence_rescue"))
    );
    for strategy in [
        "embedded_icc_to_linear_prophoto",
        "gamut_stabilized_image_matrix_blend",
    ] {
        assert!(
            schema["$defs"]["colorspaceMappingMetrics"]["properties"]["mapping_strategy"]["enum"]
                .as_array()
                .is_some_and(|strategies| strategies.contains(&serde_json::json!(strategy)))
        );
    }
    assert_eq!(
        schema["$defs"]["neutralSafetyRescue"]["additionalProperties"],
        false
    );
    assert!(schema["$defs"]["neutralSafetyRescue"]["required"]
        .as_array()
        .is_some_and(|fields| fields
            .iter()
            .any(|field| field == "midtone_saturation_p95_reduction")));
    assert!(
        schema["$defs"]["colorspaceMappingMetrics"]["properties"]["color_confidence_status"]
            ["enum"]
            .as_array()
            .is_some_and(|statuses| statuses
                .iter()
                .any(|status| status == "review_required_selected_mapping"))
    );
    assert!(
        schema["$defs"]["colorspaceMappingMetrics"]["properties"]["color_confidence_status"]
            ["enum"]
            .as_array()
            .is_some_and(|statuses| statuses
                .iter()
                .any(|status| status == "review_required_negative_reconstruction"))
    );
    assert!(schema["$defs"]["colorspaceMappingMetrics"]["properties"]
        ["negative_reconstruction_confidence_status"]["enum"]
        .as_array()
        .is_some_and(|statuses| statuses.iter().any(|status| {
            status == "review_required_measured_response_outside_runtime_curve_support"
        })));
    assert_eq!(
        schema["$defs"]["colorspaceMappingMetrics"]["properties"]
            ["negative_reconstruction_evidence_confidence"]["maximum"],
        1
    );
    assert_eq!(
        schema["$defs"]["colorspaceMappingMetrics"]["properties"]
            ["color_confidence_before_base_limit"]["maximum"],
        1
    );
    assert!(
        schema["$defs"]["toneMappingMetrics"]["properties"]["tone_confidence_status"]["enum"]
            .as_array()
            .is_some_and(|statuses| statuses
                .iter()
                .any(|status| status == "review_required_upstream_color"))
    );
    assert_eq!(
        schema["$defs"]["toneMappingMetrics"]["properties"]["requested_grain_evidence_confidence"]
            ["maximum"],
        1
    );
    assert!(
        schema["$defs"]["toneMappingMetrics"]["properties"]["tone_confidence_status"]["enum"]
            .as_array()
            .is_some_and(|statuses| statuses
                .iter()
                .any(|status| status == "review_required_tone_output_evidence"))
    );
    assert_eq!(
        schema["$defs"]["toneMappingMetrics"]["properties"]["tone_output_confidence_status"]
            ["$ref"],
        "#/$defs/toneOutputConfidenceStatus"
    );
    assert_eq!(
        schema["$defs"]["toneMappingMetrics"]["properties"]
            ["adaptive_vibrance_preferred_memory_color_guard"]["$ref"],
        "#/$defs/adaptiveVibrancePreferredMemoryColorGuard"
    );
    assert_eq!(
        schema["$defs"]["toneMappingMetrics"]["properties"]["preferred_skin_rendering"]["$ref"],
        "#/$defs/preferredSkinRendering"
    );
    assert_eq!(
        schema["$defs"]["toneMappingMetrics"]["properties"]
            ["noise_reduction_structure_excluded_ratio"]["maximum"],
        1
    );
    assert_eq!(
        schema["$defs"]["toneMappingMetrics"]["properties"]["noise_reduction_structure_gate_end"]
            ["exclusiveMinimum"],
        0
    );
    assert_eq!(
        schema["$defs"]["adaptiveVibrancePreferredMemoryColorGuard"]["additionalProperties"],
        false
    );
    assert_eq!(
        schema["$defs"]["adaptiveVibrancePreferredMemoryColorGuard"]["properties"]["reference"]
            ["const"],
        "https://doi.org/10.2352/issn.2169-2629.2021.29.170"
    );
    assert_eq!(
        schema["$defs"]["adaptiveVibrancePreferredMemoryColorGuard"]["properties"]["families"]
            ["maxItems"],
        3
    );
    assert_eq!(
        schema["$defs"]["preferredSkinRendering"]["properties"]["maximum_delta_e_ab"]["const"],
        3
    );
    assert_eq!(
        schema["$defs"]["preferredSkinRendering"]["properties"]["radial_excess_reduction"]["const"],
        0.35
    );
    assert_eq!(
        schema["$defs"]["preferredSkinRendering"]["properties"]["method"]["const"],
        "published_skin_preference_ellipse_with_bounded_one_way_excess_chroma_shoulder"
    );
    assert_eq!(
        schema["$defs"]["preferredSkinRendering"]["properties"]["mean_chroma_delta"]["maximum"],
        0
    );
    assert_eq!(
        schema["$defs"]["preferredSkinRendering"]["properties"]["max_delta_e_ab"]["maximum"],
        3.000000001
    );
    assert!(schema["$defs"]["toneOutputConfidenceStatus"]["enum"]
        .as_array()
        .is_some_and(|statuses| statuses
            .iter()
            .any(|status| status == "review_required_collapsed_render_luminance_range")));
    assert_eq!(
        schema["$defs"]["toneMappingMetrics"]["allOf"][0]["then"]["properties"]
            ["tone_output_evidence_confidence"]["const"],
        1
    );
    assert!(
        schema["$defs"]["saveMetrics"]["properties"]["render_review_status"]["enum"]
            .as_array()
            .is_some_and(|statuses| statuses
                .iter()
                .any(|status| status == "review_required_tone_output"))
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
    assert_eq!(
        schema["$defs"]["calibrationMetrics"]["properties"]["color_mapping_application"]["$ref"],
        "#/$defs/calibrationColorMappingApplication"
    );
    assert_eq!(
        schema["$defs"]["calibrationColorMappingApplication"]["additionalProperties"],
        false
    );
    for field in [
        "evaluated",
        "applied",
        "selection_status",
        "selected_candidate",
        "preferred_candidate",
        "reason",
        "definition",
    ] {
        assert!(
            schema["$defs"]["calibrationColorMappingApplication"]["required"]
                .as_array()
                .is_some_and(|fields| fields.contains(&serde_json::json!(field)))
        );
    }
    assert!(
        schema["$defs"]["calibrationColorMappingApplication"]["properties"]["selection_status"]
            ["enum"]
            .as_array()
            .is_some_and(|statuses| {
                statuses.contains(&serde_json::json!("accepted"))
                    && statuses.contains(&serde_json::json!("rejected_unsafe"))
                    && statuses.contains(&serde_json::json!("not_applicable"))
            })
    );
    assert_eq!(
        schema["$defs"]["calibrationColorMappingApplication"]["allOf"][2]["then"]["properties"]
            ["selection_status"]["enum"],
        serde_json::json!(["accepted", "forced"])
    );
    assert_eq!(
        schema["$defs"]["calibrationColorMappingApplication"]["allOf"][3]["then"]["properties"]
            ["applied"]["const"],
        true
    );
    assert_eq!(
        schema["$defs"]["calibrationMetrics"]["properties"]["scanner_profile"]["properties"]
            ["target_patch_signal_domain"]["enum"],
        serde_json::json!([
            "normalized_scanner_signal",
            "scanner_linearized_transmittance",
            null
        ])
    );
    assert_eq!(
        schema["$defs"]["calibrationMetrics"]["properties"]["scanner_profile"]["properties"]
            ["color_model"]["$ref"],
        "#/$defs/rootPolynomialColorModel"
    );
    assert_eq!(
        schema["$defs"]["calibrationMetrics"]["properties"]["scanner_profile"]["properties"]
            ["lut_3d_model"]["$ref"],
        "#/$defs/residualLut3dColorModel"
    );
    assert_eq!(
        schema["$defs"]["calibrationMetrics"]["properties"]["scanner_profile"]["properties"]
            ["lut_3d_fit"]["$ref"],
        "#/$defs/residualLut3dSelectionDiagnostics"
    );
    assert_eq!(
        schema["$defs"]["calibrationMetrics"]["properties"]["roll_profile"]["properties"]
            ["scanner_color_model_application"]["$ref"],
        "#/$defs/scannerColorModelApplication"
    );
    assert_eq!(
        schema["$defs"]["scannerColorModelApplication"]["properties"]["output_domain"]["const"],
        "reference_xyz_d50"
    );
    assert_eq!(
        schema["$defs"]["calibrationMetrics"]["properties"]["external_profile"]["oneOf"][1]
            ["properties"]["color_model"]["$ref"],
        "#/$defs/rootPolynomialColorModel"
    );
    assert_eq!(
        schema["$defs"]["calibrationMetrics"]["properties"]["external_profile"]["oneOf"][1]
            ["properties"]["lut_3d_model"]["$ref"],
        "#/$defs/residualLut3dColorModel"
    );
    assert!(schema["$defs"]["colorspaceMappingMetrics"]["properties"]["mapping_strategy"]
        ["enum"]
        .as_array()
        .is_some_and(|values| values.contains(&serde_json::json!("calibrated_residual_lut_3d"))));
    assert_eq!(
        schema["$defs"]["nonlinearColorModelRuntimeDiagnostics"]["properties"]["model_type"]
            ["enum"],
        serde_json::json!(["root_polynomial", "residual_lut_3d"])
    );
    assert!(
        schema["$defs"]["colorspaceMappingMetrics"]["properties"]["mapping_strategy"]["enum"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("calibrated_root_polynomial"))
    );
    assert_eq!(
        schema["$defs"]["colorspaceMappingMetrics"]["properties"]["nonlinear_color_model"]["$ref"],
        "#/$defs/nonlinearColorModelRuntimeDiagnostics"
    );
    assert_eq!(
        schema["$defs"]["nonlinearColorModelRuntimeDiagnostics"]["properties"]["support_status"]
            ["enum"],
        serde_json::json!(["accepted", "rejected"])
    );
    assert_eq!(
        schema["$defs"]["rootPolynomialColorModel"]["properties"]["basis"]["const"],
        "homogeneous_root_polynomial_rgb"
    );
    assert_eq!(
        schema["$defs"]["rootPolynomialColorModel"]["properties"]["training_patches"]["items"]
            ["$ref"],
        "#/$defs/rootPolynomialTrainingPatch"
    );
    assert!(report_schema_doc.contains("nonlinear_color_model"));
    assert!(
        schema["$defs"]["candidateAcceptance"]["properties"]["eligible_in_color_mode"].is_object()
    );
    assert_eq!(
        schema["$defs"]["candidateAcceptance"]["properties"]["candidate_kind"]["enum"],
        serde_json::json!([
            "calibrated_direct",
            "scanner_prior",
            "positive_rgb",
            "image_derived",
            "neutral_fallback"
        ])
    );
    let candidate_statuses = schema["$defs"]["candidateAcceptance"]["properties"]["status"]["enum"]
        .as_array()
        .expect("candidate acceptance status enum");
    for status in ["selected_evidence_rescue", "rejected_evidence_rescue"] {
        assert!(candidate_statuses.contains(&serde_json::json!(status)));
    }
    for field in [
        "scene_referred_prophoto_float_artifact_diagnostics",
        "gamut_clipping_map_diagnostics",
    ] {
        assert_eq!(
            schema["$defs"]["colorspaceMappingMetrics"]["properties"][field]["type"],
            serde_json::json!(["object", "null"])
        );
    }
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
    assert_eq!(
        schema["$defs"]["calibrationMetrics"]["properties"]["whitepoint"]["oneOf"][0]["$ref"],
        "#/$defs/numberVector3"
    );
    assert_eq!(
        schema["$defs"]["calibrationMetrics"]["properties"]["whitepoint"]["oneOf"][1]["type"],
        "null"
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
    assert_eq!(parsed.render.render_review_status, None);
    assert_eq!(parsed.render.render_reviewable, None);
    assert_eq!(parsed.tone.tone_output_confidence_status, None);
    assert_eq!(parsed.tone.tone_output_review_required, None);
    assert_eq!(
        parsed.tone.adaptive_vibrance_skin_memory_protection_enabled,
        None
    );
    assert_eq!(
        parsed
            .tone
            .adaptive_vibrance_preferred_memory_color_guard_enabled,
        None
    );
    assert_eq!(parsed.tone.preferred_skin_rendering_enabled, None);

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
    assert!(schema["$defs"]["stitchBaseline"]["properties"]["seam_exposure_model"].is_object());
    assert!(schema["$defs"]["stitchBaseline"]["properties"]
        ["seam_exposure_spatial_2d_gain_accepted"]
        .is_object());
    assert!(schema["$defs"]["stitchBaseline"]["properties"]
        ["seam_exposure_spatial_2d_gain_offset_accepted"]
        .is_object());
    assert!(schema["$defs"]["stitchBaseline"]["properties"]
        ["seam_exposure_spatial_quadratic_gain_accepted"]
        .is_object());
    assert!(schema["$defs"]["stitchBaseline"]["properties"]
        ["seam_exposure_spatial_quadratic_gain_offset_accepted"]
        .is_object());
    assert!(
        schema["$defs"]["stitchBaseline"]["properties"]["seam_blend_review_required"].is_object()
    );
    assert!(
        schema["$defs"]["stitchBaseline"]["properties"]["seam_detail_review_required"].is_object()
    );
    assert!(schema["$defs"]["stitchBaseline"]["properties"]
        ["seam_detail_max_symmetric_energy_ratio"]
        .is_object());
    for field in [
        "detail_review_required",
        "detail_decision_supported",
        "luminance_p10_retention",
        "chroma_p10_retention",
    ] {
        assert!(
            schema["$defs"]["grainBaseline"]["properties"][field].is_object(),
            "grain baseline schema missing {field}"
        );
    }
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
    for field in ["render_review_status", "render_reviewable"] {
        assert!(
            schema["$defs"]["renderBaseline"]["properties"][field].is_object(),
            "render baseline schema missing {field}"
        );
    }
    assert_eq!(
        schema["$defs"]["toneBaseline"]["required"],
        serde_json::json!(["highlight_chroma_compressed_ratio"])
    );
    for field in [
        "tone_output_confidence_status",
        "tone_output_review_required",
        "tone_output_evidence_confidence",
        "render_luminance_range_p05_p95",
        "render_to_mapped_luminance_range_ratio",
        "maximum_post_tone_high_clip_ratio",
        "maximum_post_tone_low_clip_ratio",
        "adaptive_vibrance_skin_memory_protection_enabled",
        "adaptive_vibrance_skin_memory_protection_space",
        "adaptive_vibrance_skin_memory_protected_ratio",
        "adaptive_vibrance_skin_memory_mean_protection",
        "adaptive_vibrance_preferred_memory_color_guard_enabled",
        "adaptive_vibrance_preferred_memory_color_guard_space",
        "adaptive_vibrance_preferred_memory_color_guard_reference",
        "adaptive_vibrance_preferred_memory_color_matched_ratio",
        "adaptive_vibrance_preferred_memory_color_limited_ratio",
        "adaptive_vibrance_preferred_memory_color_mean_scale_reduction",
        "preferred_skin_rendering_enabled",
        "preferred_skin_rendering_space",
        "preferred_skin_rendering_preference_reference",
        "preferred_skin_rendering_adjusted_ratio",
        "preferred_skin_rendering_mean_delta_e_ab",
    ] {
        assert!(
            schema["$defs"]["toneBaseline"]["properties"][field].is_object(),
            "tone baseline schema missing {field}"
        );
    }
    assert!(schema["$defs"]["toneOutputStatus"]["anyOf"][0]["enum"]
        .as_array()
        .is_some_and(|statuses| statuses
            .iter()
            .any(|status| status == "review_required_catastrophic_post_tone_clipping")));

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
        "/colorspace/calibration_color_mapping_applied",
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
    let orientation_review_schema: serde_json::Value =
        serde_json::from_str(include_str!("../docs/orientation-review.schema.json")).unwrap();
    let render_review_schema: serde_json::Value =
        serde_json::from_str(include_str!("../docs/render-review.schema.json")).unwrap();

    assert_eq!(schema["required"], serde_json::json!(["fixtures"]));
    assert_eq!(
        schema["properties"]["fixtures"]["additionalProperties"]["$ref"],
        "#/$defs/fixtureEntry"
    );
    assert_eq!(schema["$defs"]["pairLabel"]["pattern"], "^[^|]+\\|[^|]+$");
    assert_eq!(schema["$defs"]["sha256Hex"]["pattern"], "^[A-Fa-f0-9]{64}$");
    assert_eq!(
        orientation_review_schema["properties"]["schema_version"]["const"],
        1
    );
    assert_eq!(
        orientation_review_schema["properties"]["review_status"]["const"],
        "requires_human_approval"
    );
    assert_eq!(
        orientation_review_schema["$defs"]["orientationExpectationDraft"]["properties"]
            ["upright_approved"]["const"],
        false
    );
    assert_eq!(
        orientation_review_schema["$defs"]["orientationExpectationDraft"]["properties"]
            ["decoded_pixel_sha256"]["pattern"],
        "^[a-f0-9]{64}$"
    );
    assert!(
        orientation_review_schema["properties"]["orientation_correction"]["enum"]
            .as_array()
            .is_some_and(|values| values.iter().any(|value| value == "rotate-180"))
    );
    assert!(
        schema["$defs"]["fixtureEntry"]["properties"]["orientation_correction"]["enum"]
            .as_array()
            .is_some_and(|values| values.iter().any(|value| value == "rotate-180"))
    );
    assert_eq!(
        render_review_schema["properties"]["schema_version"]["const"],
        2
    );
    assert_eq!(
        render_review_schema["properties"]["review_status"]["enum"],
        serde_json::json!(["requires_human_approval", "approved"])
    );
    assert_eq!(
        render_review_schema["$defs"]["artifactBinding"]["properties"]["kind"]["enum"],
        serde_json::json!(["primary_output", "master_scene_referred", "review_srgb"])
    );
    assert_eq!(
        render_review_schema["$defs"]["inputBinding"]["properties"]["decoded_pixel_sha256"]["$ref"],
        "#/$defs/sha256Hex"
    );
    assert!(render_review_schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .any(|field| field == "grain_reduction_control"));
    assert_eq!(
        render_review_schema["$defs"]["grainControlBinding"]["properties"]["purpose"]["const"],
        "grain_reduction_off_control"
    );
    assert_eq!(
        render_review_schema["$defs"]["grainControlBinding"]["properties"]["artifacts"]["minItems"],
        3
    );
    assert_eq!(
        render_review_schema["$defs"]["decisions"]["required"]
            .as_array()
            .unwrap()
            .len(),
        8
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_readable_tiff_pairs"]
            .is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_component_sha256_pairs"]
            .is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_n_component_fixtures"]
            .is_object()
    );
    assert!(
        schema["$defs"]["coverageRequirements"]["properties"]["min_component_sha256_sets"]
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
        ["min_approved_render_review_fixtures"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_debug_artifact_expectation_fixtures"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_render_dynamic_range_contract_fixtures"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_stitch_normalization_contract_fixtures"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_geometry_preparation_contract_fixtures"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_geometry_accuracy_contract_fixtures"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_orientation_accuracy_contract_fixtures"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_negative_reconstruction_contract_fixtures"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_grain_reduction_enabled_fixtures"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_grain_detail_contract_fixtures"]
        .is_object());
    assert!(schema["$defs"]["coverageRequirements"]["properties"]
        ["min_grain_reduction_effect_contract_fixtures"]
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
        serde_json::json!(["component1"])
    );
    assert!(schema["$defs"]["fixtureEntry"]["properties"]["film_stock"].is_object());
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["input_mode"]["enum"],
        serde_json::json!(["negative", "positive"])
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["bit_depth"]["enum"],
        serde_json::json!([14, 16])
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["grain_reduction"]["enum"],
        serde_json::json!(["off", "on"])
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["grain_strength"]["minimum"],
        0
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["grain_strength"]["maximum"],
        1
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["grain_scale"]["minimum"],
        0.5
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["grain_scale"]["maximum"],
        4
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["deskew"]["enum"],
        serde_json::json!(["auto", "manual", "off"])
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["deskew_angle_degrees"]["minimum"],
        -3
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["force_no_stitch"]["type"],
        "boolean"
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["component1_sha256"]["$ref"],
        "#/$defs/sha256Hex"
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["component2_sha256"]["$ref"],
        "#/$defs/sha256Hex"
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["additional_components"]["items"]["$ref"],
        "#/$defs/additionalFixtureComponent"
    );
    assert_eq!(
        schema["$defs"]["additionalFixtureComponent"]["required"],
        serde_json::json!(["path"])
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["summary_baseline_sha256"]["$ref"],
        "#/$defs/sha256Hex"
    );
    assert_eq!(
        schema["$defs"]["fixtureEntry"]["properties"]["render_review_sha256"]["$ref"],
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
        schema["$defs"]["fixtureExpectations"]["properties"]["inferred_component_order"]
            .is_object()
    );
    for field in [
        "deskew_status",
        "deskew_applied",
        "deskew_review_required",
        "deskew_retained_area_ratio_min",
        "deskew_all_components_applied",
        "deskew_minimum_component_retained_area_ratio_min",
        "border_crop_all_components_cropped",
        "border_crop_minimum_removed_edge_count_per_component_min",
        "border_crop_retained_area_ratio_min",
        "border_crop_retained_area_ratio_max",
        "border_crop_rejected",
        "deskew_correction_degrees_expected",
        "deskew_correction_tolerance_degrees",
        "border_crop_components_expected",
        "orientation_components_expected",
        "base_confidence_min",
        "density_inversion_skipped",
        "negative_response_model",
        "negative_response_source",
        "negative_response_accepted",
        "negative_response_review_required",
        "negative_response_crosstalk_model",
        "negative_response_characteristic_curve_model",
        "negative_response_measured_model_id",
        "negative_response_measured_confidence_min",
        "negative_response_held_out_delta_e00_rms_max",
        "negative_response_held_out_max_delta_e00_max",
        "negative_response_held_out_improvement_over_unit_slope_min",
        "negative_response_density_noise_gain_max",
        "negative_response_curve_extrapolated_ratio_max",
        "negative_response_signed_headroom_preserved",
        "negative_response_curve_interpolation",
        "technical_white_balance_status",
        "technical_white_balance_applied",
        "technical_white_balance_review_required",
        "creative_temperature",
        "creative_tint",
        "seam_exposure_spatial_2d_gain_accepted",
        "seam_exposure_spatial_2d_gain_offset_accepted",
        "seam_exposure_spatial_quadratic_gain_accepted",
        "seam_exposure_spatial_quadratic_gain_offset_accepted",
        "seam_exposure_spatial_2d_distinct_columns_min",
        "seam_exposure_spatial_2d_horizontal_slope_agreement_ratio_min",
        "seam_exposure_held_out_spatial_2d_improvement_over_best_simpler_min",
        "seam_blend_review_required",
        "seam_detail_review_required",
        "seam_detail_supported_scale_count_min",
        "seam_detail_max_symmetric_energy_ratio_max",
        "grain_reduction_enabled",
        "grain_reduction_applied_ratio_min",
        "grain_reduction_structure_excluded_ratio_min",
        "grain_reduction_flat_luma_p95_reduction_ratio_min",
        "grain_reduction_flat_chroma_p95_reduction_ratio_min",
        "grain_detail_review_required",
        "grain_detail_decision_supported",
        "grain_detail_luminance_probe_count_min",
        "grain_detail_chroma_probe_count_min",
        "grain_detail_luminance_p10_retention_min",
        "grain_detail_chroma_p10_retention_min",
        "calibration_color_mapping_applied",
        "neutral_safety_rescue_applied",
        "neutral_safety_rescue_preserved_ratio_gain_min",
        "neutral_safety_rescue_midtone_saturation_p95_reduction_min",
        "neutral_safety_rescue_reason_contains",
    ] {
        assert!(
            schema["$defs"]["fixtureExpectations"]["properties"][field].is_object(),
            "fixture expectations schema missing {field}"
        );
    }
    assert!(
        schema["$defs"]["fixtureExpectations"]["properties"]["base_estimate_source"].is_object()
    );
    assert_eq!(
        schema["$defs"]["fixtureExpectations"]["properties"]["border_crop_components_expected"]
            ["items"]["$ref"],
        "#/$defs/borderCropComponentExpectation"
    );
    assert_eq!(
        schema["$defs"]["borderCropComponentExpectation"]["required"],
        serde_json::json!([
            "component_index",
            "top_removed",
            "bottom_removed",
            "left_removed",
            "right_removed",
            "tolerance_px"
        ])
    );
    assert_eq!(
        schema["$defs"]["borderCropComponentExpectation"]["properties"]["tolerance_px"]["maximum"],
        64
    );
    assert_eq!(
        schema["$defs"]["fixtureExpectations"]["properties"]["orientation_components_expected"]
            ["items"]["$ref"],
        "#/$defs/orientationComponentExpectation"
    );
    assert_eq!(
        schema["$defs"]["orientationComponentExpectation"]["required"],
        serde_json::json!([
            "component_index",
            "upright_approved",
            "decoded_pixel_sha256",
            "tag_present",
            "tag_value",
            "transform",
            "applied",
            "source_width",
            "source_height",
            "output_width",
            "output_height"
        ])
    );
    assert_eq!(
        schema["$defs"]["orientationComponentExpectation"]["properties"]["tag_value"]["maximum"],
        8
    );
    assert_eq!(
        schema["$defs"]["orientationComponentExpectation"]["properties"]["decoded_pixel_sha256"]
            ["$ref"],
        "#/$defs/sha256Hex"
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
        schema["$defs"]["fixtureExpectations"]["properties"]["seam_exposure_model"].is_object()
    );
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["seam_exposure_held_out_validation_passed"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["seam_exposure_held_out_improvement_over_gain_min"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["seam_exposure_offset_normalized_abs_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["seam_exposure_spatial_slope_abs_min"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["seam_exposure_spatial_slope_abs_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["seam_exposure_spatial_slope_agreement_ratio_min"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["seam_exposure_held_out_spatial_improvement_over_best_constant_min"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["seam_exposure_spatial_offset_slope_normalized_abs_min"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["seam_exposure_spatial_offset_slope_normalized_abs_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["seam_exposure_spatial_offset_endpoint_normalized_abs_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["seam_exposure_spatial_affine_slope_agreement_ratio_min"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["seam_exposure_spatial_affine_center_offset_delta_normalized_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler_min"]
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
        ["render_luminance_range_p05_p95_min"]
        .is_object());
    for field in [
        "render_review_status",
        "render_reviewable",
        "tone_output_confidence_status",
        "tone_output_review_required",
        "tone_output_evidence_confidence_min",
        "render_to_mapped_luminance_range_ratio_min",
    ] {
        assert!(
            schema["$defs"]["fixtureExpectations"]["properties"][field].is_object(),
            "fixture expectation schema missing {field}"
        );
    }
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["post_chroma_compression_clipped_high_ratio_max"]
        .is_object());
    assert!(schema["$defs"]["fixtureExpectations"]["properties"]
        ["post_chroma_compression_clipped_low_ratio_max"]
        .is_object());
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
    assert_eq!(registry["fixtures"]["logan"]["grain_reduction"], "off");
    assert_eq!(registry["fixtures"]["logan"]["grain_strength"], 0.5);
    assert_eq!(registry["fixtures"]["logan"]["grain_scale"], 1.0);
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["debug_artifacts_required"],
        serde_json::json!(true)
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["seam_blend_required"],
        serde_json::json!(true)
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["seam_exposure_model"],
        "identity"
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["seam_exposure_held_out_validation_passed"],
        false
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["seam_blend_mode"],
        "seam_aware_multiband"
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["seam_blend_review_required"],
        false
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["seam_detail_review_required"],
        false
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["seam_detail_supported_scale_count_min"],
        2
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["seam_detail_max_symmetric_energy_ratio_max"],
        2.0
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["grain_reduction_enabled"],
        false
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["grain_detail_review_required"],
        false
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["grain_detail_decision_supported"],
        true
    );
    assert_eq!(
        registry["fixtures"]["logan"]["expectations"]["seam_gradient_ratio_max"],
        0.9
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
        registry["fixtures"]["my-split-frame"]["input_mode"],
        "negative"
    );
    assert_eq!(registry["fixtures"]["my-split-frame"]["deskew"], "manual");
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["additional_components"][0]["path"],
        "local-fixtures/my-split-middle.tif"
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["grain_reduction"],
        "on"
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["grain_strength"],
        0.6
    );
    assert_eq!(registry["fixtures"]["my-split-frame"]["grain_scale"], 1.0);
    for field in [
        "deskew_all_components_applied",
        "deskew_minimum_component_retained_area_ratio_min",
        "border_crop_all_components_cropped",
        "border_crop_minimum_removed_edge_count_per_component_min",
        "border_crop_retained_area_ratio_min",
        "border_crop_retained_area_ratio_max",
        "border_crop_rejected",
        "deskew_correction_degrees_expected",
        "deskew_correction_tolerance_degrees",
        "border_crop_components_expected",
        "orientation_components_expected",
        "base_confidence_min",
        "density_inversion_skipped",
        "negative_response_model",
        "negative_response_source",
        "negative_response_accepted",
        "negative_response_review_required",
        "negative_response_crosstalk_model",
        "negative_response_characteristic_curve_model",
        "negative_response_measured_model_id",
        "negative_response_measured_confidence_min",
        "negative_response_held_out_delta_e00_rms_max",
        "negative_response_held_out_max_delta_e00_max",
        "negative_response_held_out_improvement_over_unit_slope_min",
        "negative_response_density_noise_gain_max",
        "negative_response_curve_extrapolated_ratio_max",
        "negative_response_signed_headroom_preserved",
        "negative_response_curve_interpolation",
    ] {
        assert!(
            !registry["fixtures"]["my-split-frame"]["expectations"][field].is_null(),
            "example fixture missing {field}"
        );
    }
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["grain_detail_luminance_probe_count_min"],
        64
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]["grain_reduction_applied_ratio_min"],
        0.01
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["grain_reduction_structure_excluded_ratio_min"],
        0.01
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["grain_reduction_flat_luma_p95_reduction_ratio_min"],
        0.02
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["grain_reduction_flat_chroma_p95_reduction_ratio_min"],
        0.05
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["grain_detail_chroma_probe_count_min"],
        64
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["grain_detail_luminance_p10_retention_min"],
        0.7
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["grain_detail_chroma_p10_retention_min"],
        0.7
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]["post_scale_preserved_ratio_min"],
        0.98
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]["calibration_color_mapping_applied"],
        true
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["render_luminance_range_p05_p95_min"],
        0.45
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]["render_review_status"],
        "reviewable"
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]["render_reviewable"],
        true
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]["tone_output_confidence_status"],
        "supported_render_tonal_distribution"
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]["tone_output_review_required"],
        false
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["tone_output_evidence_confidence_min"],
        1.0
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["render_to_mapped_luminance_range_ratio_min"],
        0.08
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["post_chroma_compression_clipped_high_ratio_max"],
        0.005
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]
            ["post_chroma_compression_clipped_low_ratio_max"],
        0.005
    );
    assert_eq!(
        registry["fixtures"]["my-split-frame"]["expectations"]["inferred_component_order"],
        serde_json::json!([1, 3, 2])
    );
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
        registry["coverage_requirements"]["min_approved_render_review_fixtures"],
        serde_json::json!(3)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_debug_artifact_expectation_fixtures"],
        serde_json::json!(3)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_render_dynamic_range_contract_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_stitch_normalization_contract_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_geometry_preparation_contract_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_geometry_accuracy_contract_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_orientation_accuracy_contract_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_negative_reconstruction_contract_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_grain_reduction_enabled_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_grain_detail_contract_fixtures"],
        serde_json::json!(1)
    );
    assert_eq!(
        registry["coverage_requirements"]["min_grain_reduction_effect_contract_fixtures"],
        serde_json::json!(1)
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
fn test_validation_roll_fixture_metadata_schema_covers_sidecar_contract() {
    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/validation-roll-fixture-metadata.schema.json"
    ))
    .unwrap();
    let validation_docs = include_str!("../docs/validation.md");

    assert_eq!(
        schema["$id"],
        "https://example.invalid/scanstitch/validation-roll-fixture-metadata.schema.json"
    );
    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(
        schema["properties"]["coverage_requirements"]["$ref"],
        "https://example.invalid/scanstitch/validation-fixtures.schema.json#/$defs/coverageRequirements"
    );
    assert_eq!(
        schema["properties"]["frames"]["additionalProperties"]["$ref"],
        "#/$defs/rollFixtureMetadataEntry"
    );
    assert_eq!(schema["$defs"]["sha256Hex"]["pattern"], "^[A-Fa-f0-9]{64}$");

    let entry = &schema["$defs"]["rollFixtureMetadataEntry"];
    assert_eq!(entry["additionalProperties"], false);
    for property in [
        "film_stock",
        "scene_tags",
        "exposure_tags",
        "reference_evidence",
        "calibration_case",
        "summary_baseline",
        "summary_baseline_sha256",
        "render_review",
        "render_review_sha256",
        "calibration_profile",
        "calibration_profile_sha256",
        "calibration_library",
        "calibration_library_sha256",
        "scanner_profile",
        "roll_profile",
        "description",
    ] {
        assert!(
            entry["properties"][property].is_object(),
            "roll fixture metadata schema missing `{property}`"
        );
    }
    assert_eq!(
        entry["properties"]["expectations"]["$ref"],
        "https://example.invalid/scanstitch/validation-fixtures.schema.json#/$defs/fixtureExpectations"
    );
    assert!(
        entry["allOf"]
            .as_array()
            .expect("roll fixture metadata conditionals")
            .len()
            >= 5
    );
    assert!(validation_docs.contains("validation-roll-fixture-metadata.schema.json"));
    assert!(validation_docs.contains("--write-roll-fixture-metadata-template"));
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
        "local-fixtures/render-reviews/my-split-frame/render-review.json",
        "force_no_stitch: true",
        "An independent roll frame needs only `component1`",
        "no duplicate path is needed",
        "repair_component1_tiff_for_fixture:logan",
        "repair_component2_tiff_for_fixture:logan",
        "replace_component1_with_minimum_bit_depth_tiff_for_fixture:logan",
        "replace_component2_with_minimum_bit_depth_tiff_for_fixture:logan",
        "stitch-compatible regression evidence",
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
        "--write-roll-fixture-registry",
        "--roll-fixture-scene-tag",
        "--roll-fixture-exposure-tag",
        "--roll-fixture-metadata",
        "--write-roll-fixture-metadata-template",
        "--write-roll-contact-sheet",
        "--write-roll-contact-sheet-index",
        "contact sheet",
        "Hash-Bound Final Render Review",
        "--write-render-review",
        "render_review_sha256",
        "min_approved_render_review_fixtures",
        "complete_render_review_for_fixture",
        "validation-roll-fixture-metadata.schema.json",
        "testroll-metadata.json",
        "--write-fixture-suite-baselines",
        "--overwrite-fixture-suite-baselines",
        "--fixture-suite-fixture",
        "--fixture-coverage --strict",
        "--fixture-suite --strict --debug",
        "coverage_validation_ready",
        "coverage_issues",
        "coverage_action_items",
        "coverage_not_validation_ready",
        "summary_baseline_write_status",
        "summary_baseline_written_path",
        "skipped_exists",
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
fn test_ci_mandatorily_builds_and_tests_the_native_opencv_feature() {
    let workflow = include_str!("../.github/workflows/ci.yml");
    let cargo_manifest = include_str!("../Cargo.toml");

    assert!(workflow.contains("opencv-native:"));
    assert!(workflow.contains("runs-on: ubuntu-24.04"));
    for package in [
        "clang",
        "libopencv-dev",
        "libclang-dev",
        "llvm-dev",
        "pkg-config",
    ] {
        assert!(
            workflow.contains(package),
            "missing native package {package}"
        );
    }
    assert!(workflow.contains("LIBCLANG_PATH=$(llvm-config --libdir)"));
    assert!(
        workflow.contains("cargo test --locked --features use-opencv --lib -- --test-threads=1")
    );
    assert!(!workflow.contains("Skipping use-opencv build"));
    assert!(cargo_manifest.contains("default-features = false"));
    assert!(cargo_manifest.contains("features = [\"calib3d\", \"features2d\", \"imgproc\"]"));
}

#[test]
fn test_color_reconstruction_policy_documents_scoring_and_constant_invariants() {
    let policy = include_str!("../docs/color-reconstruction-policy.md");
    let readme = include_str!("../README.md");
    let report_schema = include_str!("../docs/report-schema.md");

    for required in [
        "shared_robust_d_max",
        "per-channel orange-mask removal",
        "Direct-density render input also divides by the same `shared_robust_d_max`",
        "Changing this to per-channel Dmax is a colour-model change",
        "lower_is_better",
        "technical_safety_score",
        "color_fidelity_score",
        "Hard gates run before score wins are accepted",
        "Neutral fallback carries a large fidelity penalty",
        "Weights in `colorspace.rs` are decision policy",
        "Physical encoding constants",
        "Density and render-domain constants",
        "Anchor and sample filters",
        "Candidate quality weights and review thresholds",
        "Tone cleanup thresholds",
        "multi-scene, multi-film-stock CoolScan corpus",
    ] {
        assert!(
            policy.contains(required),
            "colour policy missing `{required}`"
        );
    }

    assert!(
        readme.contains("docs/color-reconstruction-policy.md"),
        "README should link the colour reconstruction policy"
    );
    assert!(
        report_schema.contains("Scores are ordered `lower_is_better`"),
        "report schema should keep score ordering visible"
    );
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
    assert_eq!(
        schema["$defs"]["scannerProfile"]["allOf"][0]["then"]["required"],
        serde_json::json!(["target_patch_signal_domain"])
    );
    assert_eq!(
        schema["$defs"]["targetPatch"]["properties"]["scanner_xy"]["maxItems"],
        serde_json::json!(2)
    );
    assert_eq!(
        schema["$defs"]["scannerProfile"]["properties"]["color_model"]["$ref"],
        "#/$defs/rootPolynomialColorModel"
    );
    assert_eq!(
        schema["$defs"]["scannerProfile"]["properties"]["lut_3d_model"]["$ref"],
        "#/$defs/residualLut3dColorModel"
    );
    assert_eq!(
        schema["$defs"]["scannerProfile"]["properties"]["lut_3d_fit"]["$ref"],
        "#/$defs/residualLut3dSelectionDiagnostics"
    );
    assert_eq!(
        schema["$defs"]["rootPolynomialColorModel"]["properties"]["basis"]["const"],
        "homogeneous_root_polynomial_rgb"
    );
    assert_eq!(
        schema["$defs"]["rootPolynomialColorModel"]["properties"]["training_patches"]["items"]
            ["$ref"],
        "#/$defs/retainedRootPolynomialTrainingPatch"
    );
    assert_eq!(
        schema["$defs"]["rootPolynomialColorModel"]["allOf"][0]["then"]["properties"]
            ["coefficients"]["minItems"],
        serde_json::json!(6)
    );
    assert_eq!(
        schema["$defs"]["rootPolynomialColorModel"]["allOf"][1]["then"]["properties"]
            ["coefficients"]["minItems"],
        serde_json::json!(13)
    );
    assert_eq!(
        schema["$defs"]["rootPolynomialValidation"]["properties"]["held_out_delta_e00_rms"]
            ["maximum"],
        serde_json::json!(6)
    );
    assert_eq!(
        schema["$defs"]["residualLut3dColorModel"]["properties"]["interpolation"]["const"],
        "tetrahedral"
    );
    assert_eq!(
        schema["$defs"]["residualLut3dColorModel"]["allOf"][0]["then"]["properties"]
            ["residual_nodes_xyz"]["minItems"],
        serde_json::json!(125)
    );
    assert_eq!(
        schema["$defs"]["residualLut3dValidation"]["properties"]["held_out_inside_domain_fraction"]
            ["minimum"],
        serde_json::json!(0.98)
    );
    assert_eq!(
        schema["$defs"]["rollProfile"]["properties"]["scanner_color_model_application"]["$ref"],
        "#/$defs/scannerColorModelApplication"
    );
    assert_eq!(
        schema["$defs"]["scannerColorModelApplication"]["properties"]["numerical_zero_tolerance"]
            ["maximum"],
        serde_json::json!(1e-9)
    );

    let profile_schema: serde_json::Value = serde_json::from_str(include_str!(
        "../docs/color-calibration-profile.schema.json"
    ))
    .unwrap();
    assert_eq!(
        profile_schema["properties"]["color_model"]["$ref"],
        "#/$defs/rootPolynomialColorModel"
    );
    assert_eq!(
        profile_schema["properties"]["lut_3d_model"]["$ref"],
        "#/$defs/residualLut3dColorModel"
    );
    assert_eq!(
        profile_schema["properties"]["lut_3d_fit"]["$ref"],
        "#/$defs/residualLut3dSelectionDiagnostics"
    );
    assert_eq!(
        profile_schema["$defs"]["rootPolynomialColorModel"]["properties"]["training_patches"]
            ["items"]["$ref"],
        "#/$defs/retainedRootPolynomialTrainingPatch"
    );
    assert_eq!(
        profile_schema["allOf"][3]["then"]["properties"]["schema_version"]["const"],
        serde_json::json!(2)
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
    assert_eq!(
        examples["scanner_profile"]["target_patch_signal_domain"],
        "scanner_linearized_transmittance"
    );
    assert_eq!(
        examples["roll_profile"]["target_patch_linearization_model_id"],
        examples["scanner_profile"]["scanner_linearization"]["model_id"]
    );
    assert_eq!(
        examples["roll_profile"]["scanner_color_model_application"]["transform"],
        "matrix"
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
        "20 passed cases",
        "reference-patch regression rejection",
        "CIEDE2000",
        "unsafe-calibration failure",
        "src/color_calibration.rs",
        "src/colorspace.rs",
        "src/bin/scanstitch-validate.rs",
        "src/render_review.rs",
        "docs/validation-fixtures.schema.json",
        "docs/render-review.schema.json",
        "tests/fixtures/baselines/logan_summary_baseline.json",
        "component*_sha256",
        "summary_baseline_sha256",
        "calibration_library_sha256",
        "min_component_sha256_pairs",
        "min_summary_baseline_sha256_fixtures",
        "min_calibration_sha256_fixtures",
        "min_approved_render_review_fixtures",
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
        "complete_render_review_for_fixture",
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
        "no validation-ready fixtures",
        "built-in LOGAN fixture coverage gate now passes",
        "validation-fixtures.with-hashes.json",
        "Local Corpus Scaffold",
        "calibration/scanners/coolscan-4000-vuescan-raw.json",
        "local-fixtures/my-split-left.tif",
        "repair_component1_tiff_for_fixture:logan",
        "replace_component1_with_minimum_bit_depth_tiff_for_fixture:logan",
        "stitch-compatible by the built-in LOGAN regression gate",
        "RGBA8 5959x3670",
        "single scan with only `component1`",
        "counter remains zero",
        "corner-review-manifest.json",
        "requires_corner_approval",
        "chart_corners_sha256",
        "overlay_pixel_sha256",
        "reference_patch_sha256",
        "classification_evidence_evaluated",
        "not_applicable_positive_input",
        "applied_crop_evidence_supported",
        "no_crop_no_convincing_dead_zone",
        "not_applicable_single_input",
        "not_evaluated_explicit_direct_density_route",
        "review_required_negative_like_positive_input",
        "review_required_input_mode",
        "accepted_warm_positive_input",
        "full_declared_decode_fidelity",
        "source_precision_upscaled_without_added_information",
        "review_required_selected_mapping",
        "not_applicable_disabled",
        "diagnostic_delivery_requires_review",
        "numerically_converged_physical_separation_unvalidated",
        "physical_separation_evidence_evaluated=false",
        "confidence_before_base_limit",
        "minimum_film_base_and_negative_response_model_evidence",
        "held_out_measured_response_supported",
        "review_required_unmeasured_unit_slope_response",
        "negative_response_model_evidence_evaluated=false",
        "held_out_measured_response_and_runtime_coverage_supported",
        "review_required_measured_response_outside_runtime_curve_support",
        "confidence_limited_by_negative_reconstruction_evidence",
        "supported_render_tonal_distribution",
        "review_required_collapsed_render_luminance_range",
        "review_required_catastrophic_post_tone_clipping",
        "review_required_tone_output",
        "confidence_limited_by_tone_output_evidence",
        "tone_output_confidence_status_changed",
        "tone_output_evidence_confidence_regressed",
        "tone_output_range_retention_regressed",
        "render_review_status=reviewable",
        "tone_output_evidence_confidence_min",
        "render_to_mapped_luminance_range_ratio_min",
        "roll_suite:<frame>:render_review_not_supported",
        "roll_suite:<frame>:tone_output_review_required",
        "tone_output_status_regressed",
        "tone_output_review_newly_required",
        "tone_output_evidence_confidence_dropped",
        "tone_output_range_retention_dropped",
        "--fail-on tone",
        "tests, including 88 validator",
        "fixture_suite_status=failed",
        "fixture_suite:logan:declared_calibration_not_applied",
        "TESTROLL/` contains 12 readable RGBA16 TIFF frames",
        "OLD_TESTROLL/` contains 39 readable `DNG_LINEAR_RAW16` frames",
        "inventory alone does not prove film stock",
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

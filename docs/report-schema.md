# Report Schema

`report.json` contains a top-level object with `metadata` and a `phases` array.
[`report-schema.schema.json`](report-schema.schema.json) is the machine-readable
draft 2020-12 schema. It keeps phase metrics extensible but gives typed coverage
for the core `colorspace_mapping` diagnostics used by validation tooling.

Top-level `metadata` is stable and identifies the run that produced the report:

| Field | Notes |
|-|-|
| `report_schema_version`, `pipeline_schema_version` | Compact schema identifiers for tooling. |
| `generated_at`, `generated_at_unix_ms` | Report/run creation time. |
| `package_name`, `package_version`, `binary_name` | Build identity. |
| `working_directory` | Process working directory. |
| `cli_args` | Invocation arguments captured by the process. |
| `output_dir`, `output_path` | Intended output directory and final render path. |

Each phase has:

| Field | Stability | Notes |
|-|-|-|
| `name` | Stable | Phase identifier. |
| `success` | Stable | `false` means the phase failed or rejected its candidate. |
| `confidence` | Stable | Phase confidence in `[0, 1]`, sometimes capped by upstream base confidence. |
| `duration_ms` | Stable | Wall-clock phase duration. |
| `metrics` | Mixed | Phase-specific fields. Stable fields are listed below. |
| `warnings` | Stable | Human-readable diagnostics. Text can change; presence and phase are more stable than exact wording. |
| `errors` | Stable | Human-readable failure reasons. |

## Stable Metrics

`load`

| Field | Notes |
|-|-|
| `component1_decode`, `component2_decode` | Per-component source layout and working-range diagnostics. |
| `component*_decode.dng_metadata` | Present for DNG inputs when primary-IFD metadata is available. It records scanner/software identity (`make`, `model`, `unique_camera_model`, `software`), DNG colour tags (`color_matrix1`, `color_matrix2`, `as_shot_neutral`, `black_level`, `white_level`, calibration illuminants), matrix condition numbers, and non-fatal metadata parse warnings. |

`stitch`

| Field | Notes |
|-|-|
| `decision` | `accepted`, `rejected`, or `skipped_pre_score`. |
| `chosen_hypothesis` | Selected ordering such as `[1|2]` or `[2|1]`, when scored. |
| `requested_transform`, `transform_model_used` | Requested and actual alignment model. |
| `hypotheses[].ordering` | Hypothesis label. |
| `hypotheses[].accepted` | Candidate acceptance state. |
| `hypotheses[].rejection_reason` | Explicit rejection reason, when rejected. |
| `hypotheses[].search.selection_reason` | Why the candidate was selected for validation. |
| `hypotheses[].search.max_overlap_considered` | Largest overlap searched. |
| `hypotheses[].search.top_candidates[].correspondence_score` | Image-evidence score before priors. |
| `hypotheses[].search.top_candidates[].prior_weight` | Conservative prior applied to the candidate. |
| `hypotheses[].validation.evidence_score` | Dense validation evidence. |
| `hypotheses[].validation.prior_weight` | Validation prior. |
| `hypotheses[].validation.overlap_support_score` | Width-support score. |
| `hypotheses[].validation.vertical_offset_plausibility_score` | Vertical-offset plausibility. |
| `hypotheses[].validation.local_consistency_score` | Dense local-window consistency. |
| `hypotheses[].validation.plausibility_score` | Combined plausibility score. |
| `runtime.total_ms` | Total stitch function runtime in milliseconds. |
| `runtime.translation_evaluation_ms` | Time spent evaluating the two translation hypotheses. |
| `runtime.homography_attempted`, `runtime.homography_ms` | Whether the OpenCV/homography path was attempted and its runtime. |
| `seam_exposure_correction.mode` | Currently `auto` for conditional overlap-derived compensation. |
| `seam_exposure_correction.applied` | Whether the right-side component was gain-corrected before blending. |
| `seam_exposure_correction.reason` | Decision reason for applying or rejecting correction. |
| `seam_exposure_correction.sample_count`, `valid_sample_ratio` | Trustworthy overlap samples used after clipping, near-black, low-texture, and outlier rejection. |
| `seam_exposure_correction.gain_rgb`, `gain_luma` | Applied gain. Identity values mean no correction was applied. |
| `seam_exposure_correction.delta_luma_before`, `delta_luma_after` | Signed log-luma seam delta before and after the applied correction. |
| `seam_exposure_correction.delta_rgb_before`, `delta_rgb_after` | Signed per-channel log seam deltas before and after the applied correction. |
| `seam_exposure_correction.seam_score_before`, `seam_score_after` | Robust seam mismatch score; lower is better. |
| `seam_exposure_correction.clipped_high_before`, `clipped_high_after` | Per-channel high-clipping ratios in the overlap before and after the applied correction. |
| `seam_exposure_correction.clipped_low_before`, `clipped_low_after` | Per-channel low-clipping ratios in the overlap before and after the applied correction. |

`working_image_select` and `density_inversion`

| Field | Notes |
|-|-|
| `input_mode` | `negative` for the default negative-film workflow, or canonical `positive` for `--input-mode positive`, `slide`, or `positive-slide`. |
| `working_image_select.confidence` | Base-confidence signal used to cap downstream phases. |
| `raw_base_confidence` | Direct working-crop rebate confidence from left/right or top/bottom base evidence. |
| `raw_base_proxy_confidence`, `raw_base_support_fraction` | Low-confidence high-transmittance fallback evidence when no edge/base strip is detected. |
| `base_estimate_source` | Where the base estimate came from, such as `working_edges`, `horizontal_base_region`, `vertical_base_region`, `component_consensus`, roll-level `roll_consensus_base`, manual `manual_base_color_override`, or low-confidence `high_transmittance_fallback`. |
| `base_estimate_reason` | Human-readable reason for the selected base estimate. |
| `positive_input_inspection` | Present in positive mode. It samples channel medians before tone mapping, reports ratios such as `red_green_ratio`, `green_blue_ratio`, `red_blue_delta`, an `orange_mask_score`, `likely_negative_like`, and `accepted_high_warm_score`. `likely_negative_like` is raised when the hard channel thresholds match or when the aggregate orange-mask score is high enough with severe R/G and G/B ratios to catch dim near-threshold frames. `accepted_high_warm_score=true` means the aggregate score is high but ratios are mild enough to keep the frame accepted as already-positive. When `likely_negative_like=true`, `working_image_select.warnings` also includes a mode-suitability warning because the already-positive path appears to have been requested for orange-mask negative-like input. |
| `density_inversion.confidence` | Density phase confidence after base-confidence limiting. |
| `base_transmittance`, `base_density` | Base estimate in linear and density domains. |
| `density_inversion.skipped` | Present for compatibility. In positive mode this is `true`, `input_mode` is `positive`, no base-color fields are required, and the working RGB scan is normalized directly to linear `[0,1]`. |
| `fastica.skipped` | In positive mode this is `true` with `input_mode=positive`; ICA is not run because no density-inverted negative-film image exists. |

`colorspace_mapping`

| Field | Notes |
|-|-|
| `render_input_source` | Render input selected for color mapping. |
| `render_input_reason` | Why the render input was selected. |
| `input_mode`, `render_input_mode` | Input workflow and requested render-input selector. Positive mode rejects `render_input_mode=ica`; `auto` and `direct-density` both report `render_input_source=positive_scan_rgb` and `render_input_reason=already-positive input normalized without density inversion`. |
| `calibration.status`, `calibration.source` | Calibration state: `not_configured`, `applied`, or `rejected`; source is an external one-off profile, calibration library, DNG `ColorMatrix1` advisory prior, image-derived anchors, or `positive_rgb_passthrough` for uncalibrated already-positive auto-mode inputs. |
| `calibration.external_profile`, `profile_schema_version`, `confidence`, `reason`, `matrix_condition_number`, `whitepoint`, `rejection_details` | One-off profile identity, validation metadata, and rejection details. |
| `calibration.library`, `scanner_profile`, `roll_profile`, `requested_film_stock`, `film_stock`, `nearest_roll_candidates`, `nearest_film_candidates` | Calibration-library diagnostics, including invalid entries, selected/advisory profiles, scanner settings, settings fingerprints, fit method, patch count, target residuals, base-color deltas, requested film-stock matches/mismatches, and whether a roll correction matrix was applied. When no explicit calibration is configured, auto mode may expose DNG `ColorMatrix1` in `scanner_profile` as a weak scanner prior; it is still scored against image-derived candidates before selection. |
| `direct_density_candidate_evaluated`, `direct_density_render_density_scale`, `direct_density_render_normalization` | Whether the direct-density candidate was compared, and the density scale used when Phase 3 positive density is normalized before transmittance conversion. Auto mode evaluates direct density alongside ICA-separated transmittance and switches only when direct density is materially safer or higher quality without worse risk. |
| `color_mode` | Requested color mapping mode: `auto`, `calibrated`, `image-derived`, or `neutral`. |
| `mapping_strategy` | `positive_rgb_passthrough`, `calibrated_profile`, `scanner_constrained_image_derived_matrix`, `image_derived_matrix`, `gamut_safe_image_matrix_blend`, `gamut_trusted_image_matrix_blend`, `neutral_balance_fallback`, `neutral_balance_gamut_fallback`, `neutral_balance_forced`, or `neutral_balance_weak_anchor_fallback`. |
| `selected_mapping_reason` | Why the selected mapping or fallback was used. |
| `candidate_scores`, `candidate_quality_scores`, `candidate_score_order`, `candidate_acceptance`, `selected_candidate`, `selected_candidate_rank`, `selected_candidate_score`, `selected_quality_score`, `technical_safety_score`, `color_fidelity_score`, `quality_components`, `selected_quality_components`, `candidate_risk`, `selected_runner_up_quality_delta`, `selection_rejections` | Ranked color mapping candidate diagnostics and hard or quality-based safety rejections. Scores are ordered `lower_is_better`; `quality_score` remains the compatibility aggregate, `technical_safety_score` covers gamut/clipping/exposure stability, and `color_fidelity_score` covers neutral accuracy, target residuals, scanner confidence, anchor support/stability, rendered-tone colour, tone-map chroma cleanup risk, and physical/perceptual model plausibility. Each candidate score also includes `quality_components` for the sub-score penalties; the selected candidate is also exposed as `quality_components` / `selected_quality_components` for report auditing. `candidate_risk` is `safe`, `review_neutral_support`, `review_anchor_support`, `review_gamut`, `review_reference_fit`, `review_tone_quality`, `review_model_plausibility`, `review_quality_score`, or `fallback_only`. |
| `candidate_quality_scores[].reference_patch_*_vs_image_derived` | Candidate-level reference-patch deltas versus the image-derived candidate, including XYZ RMS/max deltas, DeltaE76 RMS/max deltas, CIEDE2000 RMS/max deltas, and the boolean `reference_patch_regresses_image_derived` decision flag used by automatic calibration rejection. |
| `candidate_quality_scores[].rendered_tone_quality` | Candidate-specific rendered-tone evidence from a bounded sample after candidate exposure normalization and tone mapping: shadow/midtone saturation, bright-neutral saturation, bright-saturated preservation, chroma-compression ratios, post-cleanup clipping, and the rendered-tone/chroma-cleanup penalties added to `quality_components`. |
| `candidate_quality_scores[].color_model_quality` | Candidate-specific physical/perceptual evidence from the same bounded decision surface: density-to-luminance monotonicity bins, hue-linearity concentration by dominant input channel, saturated-colour preservation ratios, memory-colour proxy Lab summaries, and neutral-tile spatial cast consistency. Penalties are reported as `density_monotonicity_penalty`, `hue_linearity_penalty`, `saturation_preservation_penalty`, `memory_color_penalty`, and `spatial_consistency_penalty` in `quality_components`. |
| `calibration_acceptance` | Automatic acceptance/rejection reason for calibrated or scanner-prior candidates, including whether they beat image-derived quality, stayed inside negative-gamut safety limits, and did not regress neutral or reference-patch metrics versus image-derived mapping. Auto mode prefers calibrated/scanner-prior candidates only when all conservative checks pass; unsafe forced calibrated mode fails visibly. |
| `exposure_scale` | Highlight headroom normalization scale used by the selected mapping. |
| `regularization_lambda` | Matrix anchor regularization. |
| `neutral_sample_bands`, `neutral_sample_rejections`, `neutral_estimate_quality`, `dominant_anchor_bands`, `dominant_anchor_sample_rejections`, `dominant_anchor_quality` | Neutral and dominant-anchor sample support by shadow/midtone/highlight luminance band, accepted/rejected sample counts by reason, and whether the neutral estimate and dominant-anchor set are broad/stable enough for trusted colour. For `positive_rgb_passthrough`, these samples are diagnostic only: the selected colour transform does not depend on negative-film neutral or dominant anchors, so weak anchor support does not by itself make the positive RGB candidate untrusted. Dominant-anchor quality reports per-channel band coverage, dominant-band concentration, mean dominance margin, unstable-channel masks, and the aggregate stability score used by `anchor_stability_penalty`. Rejection reasons include clipped pixels, scanner-border-like edges, film-base-like edges, dust/outliers, luma range, chroma/low-saturation, and weak dominance gates. |
| `reference_patch_evaluation` | Optional profile-patch evaluation with per-patch selected-vs-image residuals, XYZ RMS/max error, D50 Lab DeltaE76 mean/RMS/max error, CIEDE2000 mean/RMS/max error, worst hue families, explicit candidate-level hue-family regressions versus image-derived mapping, and whether the selected candidate improves or regresses. Present only when a profile or selected library record retains target patches. |
| `neutral_trim_scale`, `neutral_trim_applied`, `neutral_trim_before_after` | Conservative post-matrix neutral trim scale, whether it was applied, before/after weighted neutral delta, per-band shadow/midtone/highlight neutral deltas, and clipping checks. Trim is applied only when aggregate neutral delta improves and no populated band worsens beyond tolerance. |
| `channel_anchor_counts` | Dominant-channel anchor counts by RGB channel. |
| `channel_anchor_min_count` | Minimum dominant-channel anchor count. |
| `channel_anchor_low_support` | Per-channel weak-anchor mask. |
| `weak_anchor_fallback_used` | Whether weak anchors gated matrix-driven output. |
| `gamut_fallback_used` | Whether negative-gamut clipping triggered fallback. |
| `image_matrix_pre_scale_clipped_low_ratio` | Low clipping that the image-derived matrix would have produced. |
| `image_matrix_pre_scale_clipped_high_ratio` | High clipping that the image-derived matrix would have produced. |
| `image_matrix_exposure_scale` | Exposure scale that the image-derived matrix would have required. |
| `calibrated_profile_pre_scale_clipped_low_ratio`, `calibrated_profile_pre_scale_clipped_high_ratio`, `calibrated_profile_exposure_scale` | External-profile candidate clipping and headroom requirements, when a valid profile was loaded. |
| `pre_scale_preserved_ratio`, `post_scale_preserved_ratio` | Ratio of pixels remaining inside gamut before and after exposure normalization for the selected mapping. |
| `post_scale_clipped_high_ratio`, `post_scale_clipped_low_ratio` | Selected colorspace output clipping after exposure scaling. |
| `color_decision_summary`, `candidate_comparison`, `usable_colourspace`, `color_processing_substeps`, `tone_color_trust_state`, `scene_referred_detail_fusion`, `color_candidate_comparison_artifact`, `gamut_clipping_map_artifact`, `gamut_clipping_map_diagnostics`, `scene_referred_prophoto_float_artifact`, `scene_referred_prophoto_float_artifact_diagnostics` | Compact selected/runner-up/rejected candidate summary, calibrated-vs-image-derived comparison, selected usable-gamut summary, explicit calibration/candidate/gamut/neutral/tone-protection substep diagnostics, whether tone-map colour protection should trust the selected color result, bounded scene-referred luminance-detail fusion from a direct-density guide when ICA keeps the safer colour, the debug-only side-by-side color candidate thumbnail path, the debug-only selected-transform gamut/clipping map path plus encoding and recomputed clipping summary, and the debug-only 32-bit float scene-referred linear ProPhoto TIFF path plus headroom diagnostics. The float artifact preserves finite values outside `[0,1]` without display normalization so highlight latitude and below-zero matrix excursions can be inspected separately from the tone-mapped output. |
| `ica_candidate`, `direct_density_candidate` | Candidate-level versions of the same colorspace diagnostics. |

`tone_mapping`

| Field | Notes |
|-|-|
| `tone_fit_policy` | Tone/exposure policy used for the render. `negative_film` uses the legacy negative workflow tone fit; `positive_scan_rgb` keeps already-positive scans on a conservative non-inverting tone placement with lower median lift and a softer display shoulder. |
| `auto_exposure_ev`, `auto_exposure_scale`, `render_exposure_ev`, `render_exposure_scale` | Automatic and applied pre-tone exposure. Positive mode may apply a bounded non-brightening exposure to keep high-key scans from rendering too light while preserving highlight headroom; negative mode defaults to `0 EV`. |
| `fit_domain` | Tone-fit luminance domain. |
| `input_linear_percentiles`, `mapped_linear_percentiles` | Linear luminance percentile diagnostics. |
| `render_luminance_percentiles`, `render_luminance_range_p05_p95` | Full-frame rendered luminance p05/p50/p95 and p05-p95 range, used to detect weak or collapsed contrast independent of the midtone-only placement checks. |
| `shadow_saturation_median`, `shadow_saturation_p95` | Darkest-band saturation diagnostics. |
| `midtone_luminance_percentiles` | Midtone luminance band. |
| `midtone_saturation_median`, `midtone_saturation_p95` | Midtone saturation diagnostics. |
| `bright_neutral_saturation_median`, `bright_neutral_saturation_p95` | Bright neutral-band saturation diagnostics. |
| `bright_saturated_saturation_median`, `bright_saturated_saturation_p95` | Bright saturated-band saturation diagnostics. |
| `color_protection_policy`, `color_trust_state`, `color_protection_reason`, `highlight_neutral_chroma_enabled`, `midtone_neutral_chroma_enabled`, `shadow_chroma_enabled` | Whether colorspace-quality diagnostics allowed tone-map chroma cleanup. Policy is `enabled`, `weak_neutral_bounded_neutral_cleanup`, legacy `weak_neutral_bounded_highlight_cleanup` / `neutral_highlight_disabled_weak_neutral`, `review_bounded_neutral_shadow_cleanup`, legacy `review_shadow_cleanup_only`, or `disabled_color_candidate_review`; trust state is `trusted`, `limited_weak_neutral`, or `review_required`; luminance tone mapping and hard highlight-gamut repair still run independently. |
| `midtone_neutral_chroma_*` | Bounded post-denoise neutral-midtone chroma cleanup diagnostics: compressed ratio, enabled flag, luminance range, min scale, and max saturation. This pass reduces residual snow/gray midtone casts only when the colour candidate permits neutral chroma cleanup; it is disabled for weak-neutral and color-review paths. |
| `midtone_neutral_pixel_count`, `midtone_neutral_saturation_p95`, `midtone_neutral_rgb_median`, `midtone_saturated_saturation_p95` | Midtone-band split by saturation for colour-cast review. Use the neutral subset for midtone cast drift; the full `midtone_rgb_median` can be dominated by real scene colour. |
| `post_chroma_compression_clipped_high_ratio` | Post-tone high clipping by channel. |
| `post_chroma_compression_clipped_low_ratio` | Post-tone low clipping by channel. |
| `local_luminance_detail_*` | Bounded post-tone local luminance detail diagnostics: whether the pass ran, blur radius, amount, EV cap, applied pixel ratio, mean/max absolute applied EV, positive-detail headroom-limited ratio, and hard clip-limited ratio. The pass scales RGB channels together, so it is intended to add latitude/detail without changing hue. |
| `adaptive_vibrance_*` | Trusted-colour display vibrance diagnostics: whether the hue-preserving saturation pass ran, why it was enabled/disabled, requested amount, maximum chroma scale, applied pixel ratio, mean/max applied scale, texture-limited ratio, and gamut-limited ratio. The pass is disabled when the selected colour candidate requires review and is reduced in high-frequency texture so it does not enrich visible grain. |
| `noise_reduction_*` | Bounded final render cleanup diagnostics: whether the edge-aware denoise pass ran, blur radius, chroma/luma amounts, applied ratio, texture/saturation-limited ratios, and mean/max chroma and luminance deltas. The pass smooths chroma residuals and mild shadow luminance noise in flat areas after tone/detail/vibrance processing; chroma texture protection uses smoothed structure so grain-scale residuals in otherwise flat areas can still be cleaned up. |
| `render_quality_bands` | Grouped overall, shadow, midtone, bright-neutral, and bright-saturated band diagnostics. |
| `high_frequency_grain.sample_count`, `sample_stride` | Sampled local residual population for grain diagnostics. |
| `high_frequency_grain.luma_residual_median`, `luma_residual_p95` | Luminance high-frequency residuals after a 3x3 local mean. |
| `high_frequency_grain.chroma_residual_median`, `chroma_residual_p95` | Chroma high-frequency residuals after removing the luminance residual. |
| `high_frequency_grain.chroma_to_luma_p95_ratio` | Chroma-vs-luma residual ratio for comparing visible color grain. |
| `high_frequency_grain.flat_luma_structure_max`, `flat_sample_count`, `flat_sample_ratio` | Low-structure subset used to separate flat-field grain from real scene edges or texture. |
| `high_frequency_grain.flat_luma_residual_p95`, `flat_chroma_residual_p95`, `flat_chroma_to_luma_p95_ratio` | High-frequency residual p95 values measured only in low-structure samples. Prefer these when deciding whether smooth areas still show distracting grain. |
| `interactive_controls_applied` | Present and `true` for explicit `scanstitch-ui` saves. |
| `interactive_controls` | Applied `exposure_ev`, `midpoint`, `slope`, `toe_lift`, and `shoulder_max` for `scanstitch-ui` saves. |

`save`

| Field | Notes |
|-|-|
| `output_path` | Final TIFF path written by this run. |
| `output_shape`, `output_width`, `output_height` | Final render dimensions. |
| `output_modified_at`, `output_modified_at_unix_ms` | Filesystem modification time after save. |
| `output_file_size_bytes` | Saved TIFF byte size. |
| `output_color_space`, `output_icc_profile` | Final TIFF colour encoding claimed by the save phase. Batch and interactive saves embed a linear ProPhoto RGB D50 ICC profile in TIFF tag `34675`. Validation summaries also independently inspect the saved TIFF and report `output_file_icc_profile_*` fields so the file can be checked against this claim. |
| `render_intent`, `quality_mode` | Finished-render intent and quality/throughput mode used for this save. Perfect mode writes the archival master and review PNG by default. |
| `master_scene_referred_path`, `master_scene_referred_diagnostics` | 32-bit float linear ProPhoto RGB D50 archival master path and value-domain diagnostics. Values outside `[0,1]` are preserved for highlight headroom and matrix excursion review. |
| `review_srgb_path`, `review_srgb_color_space` | Display-ready sRGB PNG proof written in perfect mode for quick review. |
| `review_sidecar_*`, `written_review_sidecar_*` | Applied guided-review sidecar path/hash/mark counts and the sidecar written from interactive or batch controls when requested. |
| `overwrote_existing_output` | Whether `output.tiff` existed before this run saved the final render. |
| `previous_output_modified_at`, `previous_output_modified_at_unix_ms` | Previous final-output timestamp when overwritten. |
| `stale_render_artifact_count`, `stale_render_artifacts[]` | TIFF artifacts in the output directory that predate this run and were not overwritten. |
| `input_base_confidence`, `render_review_status`, `render_reviewable`, `render_review_reason` | Final visual-review gate derived from the working-image base estimate. `blocked_low_base_confidence` means the TIFF was written, but the base estimate was fallback quality and density inversion should not be trusted without a rebate/roll-base reference or debug-artifact inspection. |

## Diagnostic Fields

Fields not listed as stable are diagnostic. They are useful for investigation, but may be renamed or reshaped while the pipeline is still being tuned. The most likely fields to move are detailed stitch search internals, exact warning text, debug artifact paths, and experimental candidate-ranking scores.

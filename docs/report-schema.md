# Report Schema

`report.json` contains a top-level object with a `phases` array. Each phase has:

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

`working_image_select` and `density_inversion`

| Field | Notes |
|-|-|
| `working_image_select.confidence` | Base-confidence signal used to cap downstream phases. |
| `raw_base_confidence` | Direct working-crop edge confidence. |
| `base_estimate_source` | Where the base estimate came from. |
| `density_inversion.confidence` | Density phase confidence after base-confidence limiting. |
| `base_transmittance`, `base_density` | Base estimate in linear and density domains. |

`colorspace_mapping`

| Field | Notes |
|-|-|
| `render_input_source` | Render input selected for color mapping. |
| `render_input_reason` | Why the render input was selected. |
| `direct_density_candidate_evaluated` | Whether the direct-density candidate was compared. |
| `mapping_strategy` | `image_derived_matrix`, `neutral_balance_gamut_fallback`, or `neutral_balance_weak_anchor_fallback`. |
| `exposure_scale` | Highlight headroom normalization scale used by the selected mapping. |
| `regularization_lambda` | Matrix anchor regularization. |
| `channel_anchor_counts` | Dominant-channel anchor counts by RGB channel. |
| `channel_anchor_min_count` | Minimum dominant-channel anchor count. |
| `channel_anchor_low_support` | Per-channel weak-anchor mask. |
| `weak_anchor_fallback_used` | Whether weak anchors gated matrix-driven output. |
| `gamut_fallback_used` | Whether negative-gamut clipping triggered fallback. |
| `image_matrix_pre_scale_clipped_low_ratio` | Low clipping that the image-derived matrix would have produced. |
| `image_matrix_pre_scale_clipped_high_ratio` | High clipping that the image-derived matrix would have produced. |
| `image_matrix_exposure_scale` | Exposure scale that the image-derived matrix would have required. |
| `post_scale_clipped_high_ratio`, `post_scale_clipped_low_ratio` | Selected colorspace output clipping after exposure scaling. |
| `ica_candidate`, `direct_density_candidate` | Candidate-level versions of the same colorspace diagnostics. |

`tone_mapping`

| Field | Notes |
|-|-|
| `fit_domain` | Tone-fit luminance domain. |
| `input_linear_percentiles`, `mapped_linear_percentiles` | Linear luminance percentile diagnostics. |
| `shadow_saturation_median`, `shadow_saturation_p95` | Darkest-band saturation diagnostics. |
| `midtone_luminance_percentiles` | Midtone luminance band. |
| `midtone_saturation_median`, `midtone_saturation_p95` | Midtone saturation diagnostics. |
| `bright_neutral_saturation_median`, `bright_neutral_saturation_p95` | Bright neutral-band saturation diagnostics. |
| `bright_saturated_saturation_median`, `bright_saturated_saturation_p95` | Bright saturated-band saturation diagnostics. |
| `post_chroma_compression_clipped_high_ratio` | Post-tone high clipping by channel. |
| `post_chroma_compression_clipped_low_ratio` | Post-tone low clipping by channel. |
| `render_quality_bands` | Grouped shadow, midtone, bright-neutral, and bright-saturated band diagnostics. |

## Diagnostic Fields

Fields not listed as stable are diagnostic. They are useful for investigation, but may be renamed or reshaped while the pipeline is still being tuned. The most likely fields to move are detailed stitch search internals, exact warning text, debug artifact paths, and experimental candidate-ranking scores.

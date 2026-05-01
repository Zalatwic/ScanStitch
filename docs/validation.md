# Real-Image Validation

Use `scanstitch-validate` to rerun a known local pair or summarize an existing `report.json` into compact JSON and Markdown. The command writes summaries only; rendered TIFFs stay under ignored local output directories.

## LOGAN Fixture

The first canonical local fixture is the LOGAN pair. Keep the source scans outside version control:

```bash
cargo run --bin scanstitch-validate -- --fixture logan
```

By default this expects `LOGAN043.tif` and `LOGAN044.tif` in the repository root and writes:

```text
output/validation/logan/report.json
output/validation/logan/summary.json
output/validation/logan/summary.md
```

To run the same summary against another local pair:

```bash
cargo run --bin scanstitch-validate -- \
  --fixture my-pair \
  --component1 path/to/component1.tif \
  --component2 path/to/component2.tif \
  --output-dir output/validation/my-pair
```

To summarize a report that already exists without reprocessing images:

```bash
cargo run --bin scanstitch-validate -- \
  --fixture logan \
  --report output/validation/logan/report.json \
  --summary-json output/validation/logan/summary.json \
  --summary-md output/validation/logan/summary.md
```

## What To Compare

Compare `summary.json` between runs. These fields are the first-pass regression signals:

| Area | Fields |
|-|-|
| Stitch | `decision`, `confidence`, `chosen_hypothesis`, `search_selection_reason`, `evidence_score`, `prior_weight`, `overlap_support_score`, `vertical_offset_plausibility_score`, `local_consistency_score`, `plausibility_score` |
| Base/density | `base_confidence`, `raw_base_confidence`, `density_confidence`, `base_estimate_source` |
| Colorspace | `render_input_source`, `render_input_reason`, `mapping_strategy`, `exposure_scale`, `regularization_lambda`, `channel_anchor_counts`, `channel_anchor_low_support`, `weak_anchor_fallback_used`, `gamut_fallback_used`, image-matrix low/high clipping |
| Tone | Shadow, midtone, bright-neutral, and bright-saturated saturation medians/p95 values; post-tone high/low clipping ratios |

## Interpreting Deltas

Treat these as review triggers:

| Field | Review when |
|-|-|
| `stitch.decision` | It changes between accepted, rejected, and skipped. |
| `stitch.confidence` | It drops materially or the chosen hypothesis changes. |
| `base_density.base_confidence` | It drops enough to cap downstream confidence. |
| `colorspace.render_input_source` | It switches between ICA and direct-density render input. |
| `colorspace.mapping_strategy` | It switches into or out of a neutral-balance fallback. |
| `colorspace.channel_anchor_low_support` | A new channel becomes weak, or a weak channel unexpectedly disappears after unrelated changes. |
| `colorspace.image_matrix_pre_scale_clipped_low_ratio` | Low clipping rises materially, especially with weak anchors. |
| `tone.bright_neutral_saturation_p95` | It rises after a change that should not affect highlight neutrality. |
| `tone.bright_saturated_saturation_p95` | It falls after a change that should not desaturate saturated highlights. |
| `tone.post_chroma_compression_clipped_high_ratio` / `low_ratio` | Clipping rises from near-zero to visible levels. |

The validation harness is a repeatability check, not a replacement for visual review. For color, tone, or colorspace changes, rerun LOGAN and inspect the local render before accepting the summary delta.

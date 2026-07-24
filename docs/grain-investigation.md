# Grain Investigation Notes

## 2026-05-02 Fresh LOGAN Comparison

Fresh isolated outputs were produced without deleting existing local renders:

```powershell
cargo run --release --bin scanstitch -- LOGAN043.tif LOGAN044.tif -o output/fresh_plan_20260501-201404 --force-stitch
cargo run --release --bin scanstitch-validate -- --fixture logan --output-dir output/validation/logan_plan_20260501-201404 --force-stitch
cargo run --release --bin scanstitch-validate -- `
  --fixture logan `
  --report output/fresh_plan_20260501-201404/report.json `
  --compare-report output/validation/logan_plan_20260501-201404/report.json `
  --summary-json output/fresh_plan_20260501-201404/compare-summary.json `
  --summary-md output/fresh_plan_20260501-201404/compare-summary.md
```

The normal and validation reports matched on the important render-comparison fields:

| Field | Result |
|-|-|
| Output dimensions | `5959x3670`, matched |
| Stitch decision | `accepted`, unchanged |
| Base estimate source | `component_consensus`, unchanged |
| Render input source | `direct_density_transmittance`, unchanged |
| Colorspace strategy | `neutral_balance_weak_anchor_fallback`, unchanged |
| Stale render artifacts | `0` in both output directories |
| Luma residual p95 ratio | `1.000000` |
| Chroma residual p95 ratio | `1.000000` |

Representative current diagnostics:

| Metric | Value |
|-|-|
| `high_frequency_grain.luma_residual_p95` | `0.035027` |
| `high_frequency_grain.chroma_residual_p95` | `0.014870` |
| `high_frequency_grain.chroma_to_luma_p95_ratio` | `0.424520` |
| `highlight_chroma_compressed_ratio` | `0.013409` |
| `highlight_neutral_chroma_compressed_ratio` | `0.112689` |
| `shadow_chroma_compressed_ratio` | `0.092097` |

Conclusion: with fresh outputs and matching options, the normal CLI and validation render are
diagnostically identical for LOGAN. The earlier visible grain mismatch is consistent with comparing
a stale `output/output.tiff` against a fresh validation render, not with a normal-vs-validation
pipeline divergence.

## Policy

The comparison above predates the independent control and documents the earlier implicit-cleanup
baseline. Current policy is stricter:

- Film-grain reduction defaults **off**. `modern-clean`, `natural-neutral`, and `film-faithful` do
  not silently enable or disable it.
- Enable it explicitly with `--grain-reduction on`; tune normalized effect with
  `--grain-strength 0..1` and native-radius multiplier with `--grain-scale 0.5..4`.
- The same on/off, strength, and scale controls are available in `scanstitch-ui` and persist in
  schema-v1 review sidecars. Old sidecars omit these fields and safely deserialize with reduction
  off.
- The final pass remains edge/detail aware, prefers chroma-grain suppression over luminance
  smoothing, limits structured detail, and scales every smoothing/damping term by requested
  strength. Its live protection mask detects coherent luminance and opponent-colour structure at
  the filter scale and twice that scale, then expands protection across the filter footprint. A
  second selectivity gate feathers between 0.018 and the independent 0.035 luminance-detail
  contrast floor. At or beyond that floor, or where a complete filter window is unavailable at
  the image boundary, the channel reconstruction is bypassed, so protected pixels remain exactly
  unchanged instead of receiving an f32 round trip. This prevents
  isoluminant coloured edges from being mistaken for chroma grain and prevents a requested pass
  from becoming a near-universal low-pass blend on a structured photograph.
- Saturation limiting is a real shadow-relaxed multiplier. The frame-level chroma-grain term now
  suppresses only the local high-frequency residual around the blurred chroma base; it never
  scales that base itself. A uniform saturated field therefore remains uniform and retains its
  chroma rather than being globally desaturated by a diagnostic labelled as denoise.
- `report.json` retains flat/all-sample `high_frequency_grain` output metrics and adds a structured
  `grain_reduction` record with requested/effective settings (including structure-gate start/end
  and exact excluded ratio), full `before` and `after` residuals, effect ratios/deltas, and
  `detail_retention`. The retention record separately reports coherent
  luminance and opponent-colour probe counts plus median and p10 pre/post contrast retention. Each
  channel needs 64 probes; a supported channel must retain at least 0.90 median and 0.70 p10
  contrast. A supported harmful loss caps tone confidence and makes the final render
  `review_required_grain_detail`; the requested output remains available for diagnosis. Missing
  coherent detail is explicitly unsupported, not a false pass. Legacy flat `noise_reduction_*`
  keys remain available to validation tooling.

For a controlled comparison, render the same input and options into separate fresh directories:

```powershell
scanstitch frame.tif -o output/grain-off --grain-reduction off
scanstitch frame.tif -o output/grain-on --grain-reduction on --grain-strength 0.6 --grain-scale 1.0
```

For roll evidence, use the same grain controls on both the baseline and current run and attach the
earlier roll JSON with `--compare-roll-suite`. The roll summary retains every frame's support,
probe counts, and median/p10 retention, then aggregates only evaluated, channel-supported frames.
It also reports the exact structure-excluded ratio per frame and its roll mean/minimum. It reports
changed enablement/evaluation, lost support, newly required review, aggregate p10 drops greater
than 0.03, and per-frame drops greater than 0.05. Historical summaries without these fields remain
readable and do not manufacture a baseline delta.

Judge reduction using flat-area residuals, supported pre/post detail retention, and 100% visual
inspection of edges, hair, foliage, textiles, fine colour boundaries, and saturated detail.
All-sample residuals intentionally include real scene structure. The self-retention diagnostic can
catch destructive smoothing, but cannot by itself decide whether coherent fine structure is subject
texture or film grain; approved real regions and visual review remain necessary.
Use fresh matched renders before judging visual grain; stale-output comparisons remain invalid
evidence.

For fixture-registry evidence, `min_grain_reduction_enabled_fixtures` rejects an all-off corpus and
`min_grain_detail_contract_fixtures` requires a validation-ready grain-on case with pinned nonzero
strength/scale, no-review/support expectations, and at least 64 probes plus a 0.70 p10-retention floor
for both luminance and opponent-colour structure. This proves the preservation decision was actually
exercised. `min_grain_reduction_effect_contract_fixtures` goes further: the same fixture must pin
positive minima for the applied-pixel ratio, exact structure-excluded ratio, and flat-area luma and
chroma p95 residual reductions. Those fixture-specific floors prove that a detail-safe pass also did
measurable work without regressing to a near-universal mask; they do not replace approved real
regions or a photographic preference judgment.

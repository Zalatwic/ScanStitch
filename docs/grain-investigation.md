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

Default denoising is allowed only when it is bounded, diagnostics-backed, and visible in the report.
The current default cleanup runs after tone/detail/vibrance, smooths chroma residuals plus mild
shadow luminance noise in flat areas, and reports `noise_reduction_*` fields alongside
`high_frequency_grain`. Newer reports also include `high_frequency_grain.flat_*` metrics, which
separate low-structure flat-area grain from all-sample residuals that can include real scene detail.
Use fresh matched renders before judging visual grain; stale-output comparisons remain invalid
evidence.

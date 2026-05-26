# Image Quality Workflow

Use this workflow before changing pixel processing.

1. Produce fresh normal and validation renders in isolated output directories.
2. Run `scanstitch-validate --compare-report --strict`.
3. Review `comparison.status`, `comparison.issues`, and warnings before opening TIFFs.
4. Inspect the final TIFFs only after the reports show matching dimensions, stitch decisions,
   base source, render input source, and colorspace strategy.
5. Record the visual observation with the relevant metrics:
   - `colorspace.render_input_source`
   - `colorspace.mapping_strategy`
   - `colorspace.channel_anchor_low_support`
   - tone/chroma compression ratios
   - `tone.noise_reduction_*`
   - `tone.high_frequency_grain.*_residual_p95`
   - `tone.high_frequency_grain.flat_*` when distinguishing grain in smooth areas from real detail
   - stitch seam exposure diagnostics

Processing changes should be targeted at a measured failure mode:

| Symptom | First diagnostics to check |
|-|-|
| Color cast | anchor counts, weak-anchor fallback, calibration source |
| Highlight color speckle | highlight neutral chroma compression and high clipping |
| Shadow color speckle | shadow chroma compression and chroma residual p95 |
| Grain/noise mismatch | all-sample and flat-area luma/chroma residual p95 ratios plus `noise_reduction_*` deltas from a fresh comparison |
| Seam discontinuity | seam exposure correction and stitch overlap validation |

Do not retune denoising based on stale or unmatched outputs. Any strength change should be
supported by both a visual example and a diagnostic delta.

## Positive Roll Tone Defaults

The current positive-mode defaults were tuned against the local slide/positive BIRDING roll on
2026-05-19 using isolated output directories. In that run the positive roll lived at `TESTROLL/`;
if `TESTROLL/` has since been repointed to scanner RAW negatives, the positive roll-suite should
report `render.positive_input_likely_negative_like=true` and
`roll_suite:<frame>:positive_input_negative_like` instead of being treated as calibration evidence.
Restore the positive scans or point `--roll-dir` at the positive roll before retuning positive-mode
colour or tone. Run the inventory gate first; if it reports `positive_input_negative_like`, stop and
fix the roll selection before using the output as positive calibration evidence:

```powershell
cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-inventory --input-mode positive --bit-depth 16 --strict --output-dir output/testroll_positive_inventory_gate
```

Only after the strict inventory gate passes should a full positive roll-suite be treated as the
primary positive regression run:

```powershell
cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --input-mode positive --bit-depth 16 --output-dir output/testroll_positive_tone_density
```

Positive mode keeps the successful colour behaviour from `positive_rgb_passthrough`: uncalibrated
auto colour preserves the input RGB channel ratios, reports
`colorspace_mapping.render_input_source=positive_scan_rgb`, and does not use negative-film neutral
or dominant-anchor reconstruction. The tonal defaults are positive-specific:

| Default | Value | Intent |
|-|-:|-|
| Highlight exposure target | `p95 <= 0.90` | Preserve highlight headroom before tone fitting |
| Midtone exposure ceiling | `p50 <= 0.64` | Stop high-key positive scans from staying too light |
| Auto exposure bound | `-0.45 EV..0 EV` | Allow only bounded non-brightening placement |
| Tone fit domain | linear luminance | Preserve positive scan luminance order |
| Slope scale/range | `1.05`, clamped `1.35..3.80` | Add print-like contrast without hard shoulders |
| Target median | `0.80 * input_p50 + 0.045`, clamped `0.14..0.64` | Keep midtones near or below the scan median |
| Toe/shoulder | `0.003` / `0.988` | Keep black detail and avoid clipped white placement |

Fresh positive-roll suite metrics for the original positive run, refined run, and selected
tone-density defaults:

| Output dir | Avg auto EV | Avg mapped p50 | Avg midtone p50 | Avg mapped p95 | Avg bright-neutral p95 | Highlight comp max | Post-tone clip max |
|-|-:|-:|-:|-:|-:|-:|-:|
| `output/testroll_positive` | `0.000` | `0.555` | `0.556` | `0.940` | `0.963` | `0.094` | `0.000` |
| `output/testroll_positive_refined` | `-0.075` | `0.443` | `0.443` | `0.867` | `0.922` | `0.010` | `0.000` |
| `output/testroll_positive_tone_density` | `-0.075` | `0.408` | `0.408` | `0.846` | `0.912` | `0.003` | `0.000` |

Representative frames checked: `BIRDING038`, `BIRDING045`, `BIRDING060`, `BIRDING068`, and
`BIRDING070`. The selected defaults lowered midtone placement across normal, high-key, low-key, and
bright foliage frames without high/low post-tone clipping, while leaving `mapping_strategy` as
`positive_rgb_passthrough`, `candidate_risk=safe`, and `tone_color_trust_state=trusted`.

Remaining edge cases:

| Case | Review signal |
|-|-|
| Very high-key positives at the `-0.45 EV` cap | Confirm `mapped_linear_percentiles[1]` and `bright_neutral` bands still separate highlights from midtones. |
| Low-key positives with sparse midtone samples | Confirm darker placement did not make the subject too dense; `BIRDING060` is the current reference. |
| Frames with large shadow cleanup ratios | Review `shadow_chroma_compressed_ratio` and `shadow_saturation_p95` for shadow colour speckle versus over-neutralization. |
| Uncalibrated positive passthrough | Treat hue as scan-preserving, not colourimetric truth; use a calibration profile/library for scanner-characterized positives. |
| Interactive saves | Check `interactive_controls` because manual exposure/tone edits can intentionally override the defaults above. |

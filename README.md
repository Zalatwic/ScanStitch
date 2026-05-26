# ScanStitch

Reconstructs 35mm color negative film frames from split RAW TIFF scans (Nikon CoolScan 4000). Removes scanner borders, optionally stitches split frames, strips the orange mask in density domain, separates dye layers via FastICA, maps to ProPhoto RGB D50, and applies film-like tone mapping. Outputs 16-bit positive TIFF.

## Pipeline

| Phase | Module | What it does |
|-|-|-|
| 1 | `border.rs` | Detects/removes scanner dead zones (top/bottom) using per-row median + MAD statistics |
| 2a | `base_detect.rs` | Identifies film base (rebate) on left/right edges via block-based chroma clustering |
| 2b | `frame_classify.rs` | Classifies frame as Intact / SingleBorder / CandidateSplit / Ambiguous, then emits an explicit stitch decision |
| 2c | `stitch.rs` | Scores both `[1|2]` and `[2|1]`, widens the pure-Rust overlap sweep when the small-overlap band is weak or ambiguous, ranks distinct translation candidates with correspondence-driven scores plus conservative overlap/vertical priors, validates them with dense local windows, conditionally normalizes seam exposure only when overlap evidence supports it, optionally upgrades to feature-based homography, and reports explicit rejection reasons plus candidate-selection diagnostics |
| 3 | `density.rs` | T -> density, subtract base density, invert via robust percentile `D_max - D'`, plus linear-division diagnostics |
| 4 | `ica.rs` | Full-image streaming FastICA (tanh contrast), deterministic permutation/sign resolution, per-channel robust density normalization |
| 4.6 | `colorspace.rs` | Rank calibrated, scanner-prior, image-derived, and neutral-balance color mapping candidates with quality scores, Bradford-adapt to ProPhoto RGB (D50), then report calibration acceptance, neutral support, gamut safety, and neutral trim diagnostics |
| 5 | `tonemap.rs` | Naka-Rushton sigmoid with histogram-fitted midpoint/slope, diagnostics-driven chroma protection, bounded local detail, edge-aware render denoise, and render-quality color diagnostics |

## Build

```bash
# Default (pure Rust, no external deps beyond cargo)
cargo build --release

# With OpenCV for robust feature matching
# Requires libclang/LLVM tooling discoverable via `llvm-config` or `LIBCLANG_PATH`
cargo build --release --features use-opencv
```

## Usage

```
scanstitch <COMPONENT1> <COMPONENT2> [OPTIONS]
```

| Flag | Default | Description |
|-|-|-|
| `-o, --output-dir` | `output` | Where to write results |
| `--debug` | off | Save intermediate images + tone curve LUT |
| `--force-stitch` | off | Force stitching even if frames look intact |
| `--force-no-stitch` | off | Skip stitching entirely |
| `--bit-depth` | 14 | Input bit depth (14-bit padded to 16, or native 16) |
| `--input-mode` | negative | Input workflow: `negative` for film negatives, `positive` for slide scans or already-inverted RGB; aliases `slide` and `positive-slide` report as `positive` |
| `--calibration-profile` | none | Compatibility one-off scanner/film calibration profile JSON |
| `--calibration-library` | none | Local scanner, roll, and film-hint calibration library directory |
| `--scanner-profile` | none | Scanner/settings profile ID selected from the calibration library |
| `--roll-profile` | none | Optional roll correction/profile ID selected from the calibration library |
| `--film-stock` | none | Optional film-stock label used to rank film evidence and reject mismatched roll profiles |
| `--color-mode` | auto | Color mapping mode: auto / calibrated / image-derived / neutral |
| `--render-input` | auto | Negative-mode render input: auto / ica / direct-density. `--input-mode positive --render-input ica` is rejected. |
| `--render-intent` | modern-clean | Finished-render intent: modern-clean / natural-neutral / film-faithful |
| `--quality-mode` | perfect | Quality/throughput mode: perfect / balanced / fast. Perfect writes master and review artifacts by default. |
| `--write-master` | auto in perfect | Write `master_scene_referred.tiff` outside perfect mode too |
| `--review-sidecar` | none | Apply saved guided-review render decisions |
| `--write-review-sidecar` | none | Write guided-review render decisions after saving |
| `--ica-max-iter` | 100 | Max FastICA iterations |
| `--ica-tol` | 1e-5 | FastICA convergence threshold |
| `--transform` | auto | Transform model: auto / translation / affine / homography |
| `--use-opencv` | off | Use OpenCV backend (needs `use-opencv` feature) |

Enable logging with `RUST_LOG=info` (or `debug` for verbose).

## Interactive UI

`scanstitch-ui` runs the same pipeline up to the ProPhoto render cache, then opens a terminal control UI plus a separate native preview window:

```bash
cargo run --bin scanstitch-ui -- component1.tiff component2.tiff -o output/
```

Use `Tab` to select tone controls, arrow keys or `+`/`-` to adjust, `r` to reset to the automatic tone fit, `s` to write `output.tiff` and `report.json`, and `q` to quit. The interactive path does not write final output until `s` is pressed.

## Output

| File | Always | Description |
|-|-|-|
| `output.tiff` | yes | Final 16-bit positive TIFF (linear ProPhoto RGB D50 ICC embedded, tone-mapped) |
| `master_scene_referred.tiff` | perfect | 32-bit float scene-referred linear ProPhoto RGB D50 master with highlight headroom preserved |
| `review_srgb.png` | perfect | Display-ready sRGB proof for quick visual review |
| `report.json` | yes | Run metadata, output freshness diagnostics, per-phase confidence, metrics, warnings, errors, stitch hypotheses, candidate-ranking diagnostics, validation-vs-search selection diagnostics, overlap/local/plausibility score breakdowns, correspondence-vs-prior weights, overlap-range diagnostics, downstream base-confidence caps, and explicit rejection reasons |
| review sidecar JSON | optional | Non-destructive interactive/guided render decisions when `--write-review-sidecar` is supplied |
| `comp1_cropped.tiff` | debug | After border removal |
| `comp2_cropped.tiff` | debug | After border removal |
| `comp1_base_mask_left.tiff` / `comp1_base_mask_right.tiff` | debug | Component 1 edge-confidence grids |
| `comp2_base_mask_left.tiff` / `comp2_base_mask_right.tiff` | debug | Component 2 edge-confidence grids |
| `*_base_columns.tiff` / `*_base_overlay.tiff` | debug | Full-width base-gap confidence and overlays |
| `stitch_overlap_12_*.tiff` / `stitch_overlap_21_*.tiff` | debug | Both stitch hypotheses: aligned overlap strips and seam overlays |
| `stitch_seam_overlay.tiff` | debug | Final accepted overlap/seam visualization |
| `stitched.tiff` | debug | After stitching (if stitching occurred) |
| `phase3_positive.tiff` | debug | After density inversion |
| `phase3_positive_passthrough.tiff` | debug | Positive-mode normalized RGB passthrough before color mapping |
| `phase3_linear_division.tiff` | debug | Negative-mode diagnostic linear-division inversion view |
| `phase4_ica.tiff` | debug | After FastICA separation |
| `phase46_prophoto.tiff` | debug | After ProPhoto mapping |
| `phase46_scene_referred_prophoto_float.tiff` | debug | 32-bit float scene-referred linear ProPhoto buffer, preserving finite values below 0 and above display white |
| `phase46_color_candidate_comparison.tiff` | debug | Side-by-side selected/image-derived/calibrated candidate thumbnail when available |
| `phase46_gamut_clipping_map.tiff` | debug | Selected-transform clipping map: red high, blue low, green preserved in-gamut |
| `tone_curve_lut.json` | debug | 256-point [input, output] tone curve |

## Report Diagnostics

The `colorspace_mapping` phase reports `candidate_quality_scores`, `quality_components`,
`color_decision_summary`, `selected_quality_score`, `technical_safety_score`,
`color_fidelity_score`, `candidate_risk`, `tone_color_trust_state`, `calibration_acceptance`,
`neutral_estimate_quality`, `neutral_sample_rejections`, `dominant_anchor_sample_rejections`,
`dominant_anchor_quality`, candidate `rendered_tone_quality` / `color_model_quality`,
XYZ/DeltaE reference-patch residuals, and
`neutral_trim_before_after` in addition to the legacy `candidate_scores` fields. In `auto`,
calibrated and scanner-prior candidates are used
only when they beat image-derived quality, stay inside negative-gamut safety limits, and do not
regress neutral or reference-patch diagnostics. Candidate metrics for ICA and direct-density render
inputs include the same anchor support fields alongside gamut clipping, exposure-scale,
neutral-delta, and preserved-gamut diagnostics. Direct-density render input normalizes the Phase 3
positive-density range by `shared_robust_d_max` before transmittance conversion, matching ICA's
bounded density domain and preventing sparse clipped pixels from placing high-key negatives near
black; the report records `direct_density_render_density_scale`. Direct-density fallback is selected
only when it materially improves destructive ICA low clipping without regressing neutral balance. Debug runs write
`phase46_prophoto.tiff`, `phase46_scene_referred_prophoto_float.tiff`, and, when
candidate previews are available, `phase46_color_candidate_comparison.tiff`. They also write
`phase46_gamut_clipping_map.tiff`, where red marks post-scale high clipping, blue marks post-scale
low clipping, and green marks preserved in-gamut pixels. The float ProPhoto artifact is the
high-latitude scene-referred buffer before tone mapping; use it for exposure/colour inspection
instead of the clamped preview when highlights exceed display white.

When `--input-mode positive` is selected, the pipeline still loads, crops borders, optionally
stitches, color maps, tone maps, saves, and reports normally, but skips negative-film base
requirements, density inversion, orange-mask removal, and ICA. The selected working RGB scan is
normalized directly to linear `[0,1]`; `density_inversion` and `fastica` remain in the report with
`skipped: true` and `input_mode: "positive"`, and `colorspace_mapping.render_input_source` is
`positive_scan_rgb`. In uncalibrated auto colour mode, the colour phase now uses
`mapping_strategy=positive_rgb_passthrough`: it preserves the already-positive RGB channel ratios in
the linear ProPhoto working buffer, evaluates clipping/headroom and tone/model diagnostics, and does
not require negative-film neutral or dominant-anchor support. Explicit calibration profiles still
use the calibrated candidate path. Positive mode also uses a separate `positive_scan_rgb` tone
policy: it may apply only bounded non-brightening pre-tone exposure, places midtones more
conservatively than the negative-film fit, and uses a slightly softer shoulder so already-positive
scans keep natural density without inverting luminance order.

Calibration diagnostics live under `colorspace_mapping.metrics.calibration`. A selected scanner
profile is used as the stable scanner/settings prior, an explicit roll matrix is composed after it,
requested film-stock evidence ranks roll/film candidates and rejects mismatched explicit roll
profiles, and weak auto-matches are reported as advisory instead of being silently applied. The older
`--calibration-profile` path remains available for one-off scanner/film profiles. See
`docs/color-calibration.md` for library record examples, the calibration-library record schema, and
`scanstitch-calibrate` usage.

Create or update local calibration records with `scanstitch-calibrate`:

```powershell
cargo run --bin scanstitch-calibrate -- scanner-target --library calibration --measurements scanner-target.json --profile-id coolscan-4000-vuescan-raw
cargo run --bin scanstitch-calibrate -- roll-target --library calibration --scanner-profile coolscan-4000-vuescan-raw --measurements roll-target.json --profile-id logan-roll
cargo run --bin scanstitch-calibrate -- roll-base --library calibration --scanner-profile coolscan-4000-vuescan-raw --measurements roll-base.json --profile-id logan-roll-base
```

The `stitch` phase reports `seam_exposure_correction` for accepted stitches. In `auto` mode this
stays identity unless robust overlap samples show a meaningful exposure mismatch, the proposed gain
improves the seam score, local windows agree, and clipping does not materially increase.

The `tone_mapping` phase reports render-quality bands for shadows, midtones, bright neutral
candidates, and saturated bright pixels. Chroma protection remains luminance-based but is gated by
the selected colorspace quality diagnostics and reports `color_protection_reason`,
`color_trust_state`, `highlight_neutral_chroma_enabled`, and `shadow_chroma_enabled`. Key fields include
`tone_fit_policy`, `auto_exposure_ev`, `render_exposure_ev`, `shadow_saturation_median/p95`,
`midtone_luminance_percentiles`, `midtone_saturation_median/p95`,
`bright_neutral_saturation_median/p95`, `bright_neutral_rgb_median`, and
`bright_saturated_saturation_median/p95`. It also reports diagnostic high-frequency luma/chroma
residuals plus `noise_reduction_*` fields from the bounded final render cleanup. Weak-neutral
colour candidates keep shadow chroma cleanup enabled while disabling only neutral-highlight cleanup,
so sparse-neutral frames can still suppress shadow colour speckle without claiming full colour
trust. Reports saved from `scanstitch-ui` additionally record the applied interactive tone and
exposure controls.

The top-level report metadata records `generated_at`, binary/package version, working directory,
CLI args, and the intended output path. The `save` phase records final output dimensions,
modification time, overwrite status, and stale TIFF artifacts left from earlier runs.

See `docs/report-schema.md` for stable report fields and diagnostic fields.
See `docs/grain-investigation.md` for the current LOGAN grain comparison and render-denoise policy.

## Real-Image Validation

`scanstitch-validate` runs or summarizes local validation fixtures and emits compact JSON/Markdown
summaries without versioning rendered TIFFs.

```powershell
cargo run --bin scanstitch-validate -- --fixture logan
cargo run --bin scanstitch-validate -- --report output/validation/logan/report.json --fixture logan
cargo run --bin scanstitch-validate -- --report output/report.json --fixture logan --compare-report output/validation/logan/report.json --strict
cargo run --release --bin scanstitch-validate -- --fixture logan --compare-summary tests/fixtures/baselines/logan_summary_baseline.json --strict
cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --input-mode positive --bit-depth 16 --output-dir output/testroll_positive
```

See `docs/validation.md` for the LOGAN workflow, tracked summary-baseline gate, and delta-review fields.
Use `docs/validation-fixtures.example.json` as the template for local fixture registries;
`docs/validation-fixtures.schema.json` documents the machine-readable registry contract, and
`docs/validation-summary-baseline.schema.json` documents tracked compact summary baselines.
See `docs/color-reconstruction-audit.md` for the current objective-to-artifact checklist and the
remaining real-corpus evidence gap.
See `docs/release.md` for local release builds and install commands.

## Benchmarks

`scanstitch-bench` runs lightweight local phase timing without adding benchmark dependencies:

```powershell
cargo run --release --bin scanstitch-bench -- --width 640 --height 360 --iterations 3
cargo run --release --bin scanstitch-bench -- --real-logan
```

The real LOGAN benchmark is opt-in and requires local TIFFs.

The full LOGAN stitch test is ignored by default because it requires local private TIFFs and runs
the full real-image stitch path. Run it explicitly when needed:

```powershell
cargo test --test test_stitch test_real_sample_pair_uses_rgba8_load_path_and_accepts_narrow_overlap_stitch -- --ignored
```

## Architecture

```
src/
  main.rs                    CLI entry point
  bin/scanstitch-validate.rs Real-image validation harness
  bin/scanstitch-calibrate.rs Calibration-library record generator
  bin/scanstitch-ui.rs       Interactive render review UI
  bin/scanstitch-bench.rs    Local phase benchmark harness
  pipeline.rs                Phase orchestration and report assembly
  border.rs                  Phase 1: row-stats border detection
  base_detect.rs             Phase 2a: block chroma clustering
  frame_classify.rs          Phase 2b: intact/split classification
  stitch.rs                  Phase 2c: scoring, validation, blending, diagnostics
  density.rs                 Phase 3: density-domain inversion
  ica.rs                     Phase 4: streaming FastICA and channel normalization
  colorspace.rs              Phase 4.6: ProPhoto D50 mapping and fallback diagnostics
  tonemap.rs                 Phase 5: tone mapping and render diagnostics
  color_calibration.rs       Calibration profiles, libraries, fitting, and diagnostics
  positive_input.rs          Positive/slide input suitability checks
  validation.rs              Compact validation summary extraction and suite checks
  constants.rs               D50 white point, ProPhoto/Bradford matrices
  tiff_io.rs                 TIFF/DNG load/save and working-range normalization
  interactive.rs             Interactive render controls and sidecar support
  streaming.rs               Tiled iteration helpers
  report.rs                  JSON report structs
  cli.rs                     Clap 4 arg definitions
  debug.rs                   Debug image helpers
  cv_adapter.rs              OpenCV feature-gated adapter
```

Line and test counts are intentionally omitted here; use `rg --files src tests` and `rg "#\[test\]" src tests` for local counts.

## Tests

```bash
cargo fmt --check
cargo build --locked
cargo test -- --test-threads=1
cargo test --test test_border -- --test-threads=1
cargo test --test test_ica -- --test-threads=1
```

CI runs the serialized Rust test gate on Linux and Windows. The slow LOGAN stitch test remains
ignored by default because it requires private local TIFFs; run it explicitly as shown in the
benchmark section when validating real-scan behavior.

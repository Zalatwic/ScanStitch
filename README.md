# ScanStitch

Reconstructs 35mm color negative film frames from split RAW TIFF scans (Nikon CoolScan 4000). Removes scanner borders, optionally stitches split frames, strips the orange mask in density domain, separates dye layers via FastICA, maps to ProPhoto RGB D50, and applies film-like tone mapping. Outputs 16-bit positive TIFF.

## Pipeline

| Phase | Module | What it does |
|-|-|-|
| 1 | `border.rs` | Detects/removes scanner dead zones (top/bottom) using per-row median + MAD statistics |
| 2a | `base_detect.rs` | Identifies film base (rebate) on left/right edges via block-based chroma clustering |
| 2b | `frame_classify.rs` | Classifies frame as Intact / SingleBorder / CandidateSplit / Ambiguous, then emits an explicit stitch decision |
| 2c | `stitch.rs` | Scores both `[1|2]` and `[2|1]`, widens the pure-Rust overlap sweep when the small-overlap band is weak or ambiguous, ranks distinct translation candidates with correspondence-driven scores plus conservative overlap/vertical priors, validates them with dense local windows, optionally upgrades to feature-based homography, and reports explicit rejection reasons plus candidate-selection diagnostics |
| 3 | `density.rs` | T -> density, subtract base density, invert via robust percentile `D_max - D'`, plus linear-division diagnostics |
| 4 | `ica.rs` | Full-image streaming FastICA (tanh contrast), deterministic permutation/sign resolution, per-channel robust density normalization |
| 4.6 | `colorspace.rs` | Estimate explicit `M_work_to_XYZ`, Bradford CAT, then convert to ProPhoto RGB (D50), with weak channel-anchor diagnostics and conservative gamut fallback |
| 5 | `tonemap.rs` | Naka-Rushton sigmoid with histogram-fitted midpoint/slope plus render-quality color diagnostics |

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
| `--ica-max-iter` | 100 | Max FastICA iterations |
| `--ica-tol` | 1e-5 | FastICA convergence threshold |
| `--transform` | auto | Transform model: auto / translation / affine / homography |
| `--use-opencv` | off | Use OpenCV backend (needs `use-opencv` feature) |

Enable logging with `RUST_LOG=info` (or `debug` for verbose).

## Output

| File | Always | Description |
|-|-|-|
| `output.tiff` | yes | Final 16-bit positive TIFF (ProPhoto D50, tone-mapped) |
| `report.json` | yes | Per-phase confidence, metrics, warnings, errors, stitch hypotheses, candidate-ranking diagnostics, validation-vs-search selection diagnostics, overlap/local/plausibility score breakdowns, correspondence-vs-prior weights, overlap-range diagnostics, downstream base-confidence caps, and explicit rejection reasons |
| `comp1_cropped.tiff` | debug | After border removal |
| `comp2_cropped.tiff` | debug | After border removal |
| `comp1_base_mask_left.tiff` / `comp1_base_mask_right.tiff` | debug | Component 1 edge-confidence grids |
| `comp2_base_mask_left.tiff` / `comp2_base_mask_right.tiff` | debug | Component 2 edge-confidence grids |
| `*_base_columns.tiff` / `*_base_overlay.tiff` | debug | Full-width base-gap confidence and overlays |
| `stitch_overlap_12_*.tiff` / `stitch_overlap_21_*.tiff` | debug | Both stitch hypotheses: aligned overlap strips and seam overlays |
| `stitch_seam_overlay.tiff` | debug | Final accepted overlap/seam visualization |
| `stitched.tiff` | debug | After stitching (if stitching occurred) |
| `phase3_positive.tiff` | debug | After density inversion |
| `phase3_linear_division.tiff` | debug | Diagnostic linear-division inversion view |
| `phase4_ica.tiff` | debug | After FastICA separation |
| `phase46_prophoto.tiff` | debug | After ProPhoto mapping |
| `tone_curve_lut.json` | debug | 256-point [input, output] tone curve |

## Report Diagnostics

The `colorspace_mapping` phase reports `channel_anchor_counts`, `channel_anchor_min_count`,
`channel_anchor_low_support_threshold`, and `channel_anchor_low_support` so weak image-derived
matrix anchors are visible. Weak anchors gate matrix-driven rendered output through
`neutral_balance_weak_anchor_fallback` while still reporting the image-derived matrix diagnostics.
Candidate metrics for ICA and direct-density render inputs include the same anchor support fields
alongside gamut clipping and exposure-scale diagnostics.

The `tone_mapping` phase reports render-quality bands for shadows, midtones, bright neutral
candidates, and saturated bright pixels. Key fields include `shadow_saturation_median/p95`,
`midtone_luminance_percentiles`, `midtone_saturation_median/p95`,
`bright_neutral_saturation_median/p95`, `bright_neutral_rgb_median`, and
`bright_saturated_saturation_median/p95`.

See `docs/report-schema.md` for stable report fields and diagnostic fields.

## Real-Image Validation

`scanstitch-validate` runs or summarizes local validation fixtures and emits compact JSON/Markdown
summaries without versioning rendered TIFFs.

```bash
cargo run --bin scanstitch-validate -- --fixture logan
cargo run --bin scanstitch-validate -- --report output/validation/logan/report.json --fixture logan
```

See `docs/validation.md` for the LOGAN workflow and delta-review fields.

## Architecture

```
src/
  main.rs                    CLI entry point (29 lines)
  bin/scanstitch-validate.rs Real-image validation harness (147 lines)
  pipeline.rs                Phase orchestration (1085 lines)
  border.rs                  Phase 1: row-stats border detection (358 lines)
  base_detect.rs             Phase 2a: block chroma clustering (636 lines)
  frame_classify.rs          Phase 2b: intact/split classification (273 lines)
  stitch.rs                  Phase 2c: scoring, validation, blending, diagnostics (2904 lines)
  density.rs                 Phase 3: density-domain inversion (269 lines)
  ica.rs                     Phase 4: streaming FastICA and channel normalization (506 lines)
  colorspace.rs              Phase 4.6: ProPhoto D50 mapping and fallback diagnostics (546 lines)
  tonemap.rs                 Phase 5: tone mapping and render diagnostics (584 lines)
  validation.rs              Compact validation summary extraction (461 lines)
  constants.rs               D50 white point, ProPhoto/Bradford matrices (32 lines)
  tiff_io.rs                 TIFF load/save and working-range normalization (284 lines)
  streaming.rs               Tiled iteration helpers (44 lines)
  report.rs                  JSON report structs (70 lines)
  cli.rs                     Clap 4 arg definitions (38 lines)
  debug.rs                   Debug image helpers (38 lines)
  cv_adapter.rs              OpenCV feature-gated adapter (251 lines)
```

~8600 lines of Rust code, ~2800 lines of tests.

## Tests

```bash
cargo fmt --check
cargo build --locked
cargo test -- --test-threads=1    # all 92 tests
cargo test --test test_border -- --test-threads=1
cargo test --test test_ica -- --test-threads=1
```

Test coverage: streaming/stitch unit scoring (10), base detection + classification (8),
border detection (7), colorspace (11), density (7), ICA (7), pipeline integration (8),
stitch integration (15), TIFF I/O (2), tonemap (15), and validation summary extraction (2).
Tests use synthetic or temporary fixture data from `tests/common/synthetic.rs`.

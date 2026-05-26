# Quickstart

## Prerequisites

- Rust toolchain (`rustup` installed, stable channel)
- MSVC build tools (VS2022 on this machine; custom LIB/INCLUDE paths configured in `~/.cargo/config.toml` for onecore libs)

## Build

```bash
cargo build --release
```

Binary lands at `target/release/scanstitch.exe`.

OpenCV builds remain feature-gated. In this environment `cargo build --features use-opencv`
fails unless `llvm-config`/`libclang` are installed and discoverable via
`LLVM_CONFIG_PATH` or `LIBCLANG_PATH`.

## Run

Typical invocation for CoolScan 4000 scans:

```bash
RUST_LOG=info ./target/release/scanstitch scan_001.tiff scan_002.tiff -o output/ --debug --bit-depth 14
```

For slide scans or negatives already inverted by scanner/software, make positive mode explicit:

```bash
RUST_LOG=info ./target/release/scanstitch slide_001.tiff slide_001.tiff -o output/positive \
  --input-mode positive --force-no-stitch --debug --bit-depth 16
```

With a local scanner/roll calibration library:

```bash
RUST_LOG=info ./target/release/scanstitch scan_001.tiff scan_002.tiff -o output/ \
  --calibration-library calibration \
  --scanner-profile coolscan-4000-vuescan-raw \
  --roll-profile roll-001 \
  --film-stock "Kodak Gold 200"
```

For frames you know are split across both files:
```bash
RUST_LOG=info ./target/release/scanstitch left.tiff right.tiff -o output/ --debug --force-stitch
```

For frames you know are NOT split (just process the first one):
```bash
RUST_LOG=info ./target/release/scanstitch frame.tiff dummy.tiff -o output/ --force-no-stitch
```

## Interactive Preview

Use `scanstitch-ui` when you want to adjust exposure and tone before writing the final TIFF:

```bash
cargo run --bin scanstitch-ui -- left.tiff right.tiff -o output/
```

The preview opens in a separate native window while controls stay in the terminal. Use `Tab` to select a control, arrow keys or `+`/`-` to adjust it, `r` to reset to auto tone, `s` to save `output.tiff` and `report.json`, and `q` to quit. No final TIFF or report is written until `s`.

## Interpret Output

Look at `output/`:

1. **`output.tiff`** -- Your final 16-bit positive. Open in RawTherapee, darktable, or Photoshop. Linear ProPhoto RGB color space, D50 white point, ICC profile embedded.

2. **`master_scene_referred.tiff`** -- In the default `--quality-mode perfect`, the archival 32-bit float scene-referred linear ProPhoto RGB D50 master. It preserves values outside display `[0,1]` for highlight latitude and transform review.

3. **`review_srgb.png`** -- In perfect mode, a display-ready sRGB proof for fast visual review.

4. **`report.json`** -- Machine-readable pipeline log. Check:
   - `metadata.generated_at`, `metadata.cli_args`, and `metadata.output_path` -- confirm the report belongs to the render you are inspecting.
   - `phases[].success` -- did each phase succeed?
   - `phases[].confidence` -- how confident is the result?
   - `phases[].warnings` -- anything suspicious?
   - The `fastica` phase: did it converge? How were the separated channels normalized?
   - The `stitch` phase: what were the `[1|2]` and `[2|1]` scores, which hypothesis was chosen, what thresholds were applied, what did `search.top_candidates`, `search.selection_reason`, `search.evaluated_candidates`, `search.max_overlap_considered`, `search.top_candidates[].correspondence_score`, `search.top_candidates[].prior_weight`, `validation.evidence_score`, `validation.prior_weight`, `validation.overlap_support_score`, `validation.vertical_offset_plausibility_score`, `validation.local_consistency_score`, and `validation.plausibility_score` say, did `seam_exposure_correction` apply or stay identity, and what explicit rejection reason was recorded if stitching failed?
   - The `colorspace_mapping` phase: check `calibration.status`, `calibration.scanner_profile`, `calibration.roll_profile`, `calibration.film_stock`, `calibration.nearest_roll_candidates`, `calibration.nearest_film_candidates`, `color_decision_summary`, `candidate_quality_scores`, `quality_components`, candidate `rendered_tone_quality` and `color_model_quality`, `selected_quality_score`, `technical_safety_score`, `color_fidelity_score`, `candidate_risk`, `tone_color_trust_state`, `calibration_acceptance`, `neutral_estimate_quality`, `neutral_sample_rejections`, `dominant_anchor_sample_rejections`, `dominant_anchor_quality`, optional XYZ/DeltaE `reference_patch_evaluation`, `neutral_trim_before_after`, `mapping_strategy`, `render_input_source`, `direct_density_render_density_scale`, candidate gamut clipping, and `exposure_scale` before trusting color changes.
   - The `tone_mapping` phase: check `tone_fit_policy`, `auto_exposure_ev`, `render_exposure_ev`, `color_protection_reason`, `color_trust_state`, `highlight_neutral_chroma_enabled`, `shadow_chroma_enabled`, `shadow_saturation_median/p95`, `bright_neutral_saturation_median/p95`, `bright_saturated_saturation_median/p95`, `midtone_luminance_percentiles`, `midtone_saturation_median/p95`, RGB medians, `noise_reduction_*`, and `high_frequency_grain` for color/grain review. In positive mode, `auto_exposure_ev` should be non-positive. For `scanstitch-ui` saves, also check `interactive_controls`.
   - The `save` phase: check `output_modified_at`, `output_shape`, `output_color_space`, `output_icc_profile`, `master_scene_referred_path`, `review_srgb_path`, validation summary `output_file_icc_profile_*` inspection fields, `overwrote_existing_output`, and `stale_render_artifact_count` before comparing files in an existing output directory.
   - The `working_image_select` / `density_inversion` / `fastica` / `colorspace_mapping` / `tone_mapping` phases: was base confidence low enough to trigger warnings or cap downstream confidence?
   - For `--input-mode positive`, confirm `density_inversion.metrics.skipped=true`, `fastica.metrics.skipped=true`, `colorspace_mapping.metrics.render_input_source=positive_scan_rgb`, and, for uncalibrated auto-mode scans, `colorspace_mapping.metrics.mapping_strategy=positive_rgb_passthrough`.

5. **Debug images** (if `--debug`):
   - `comp1_cropped.tiff` / `comp2_cropped.tiff` -- verify borders were removed correctly
   - `*_base_mask_*.tiff`, `*_base_columns.tiff`, `*_base_overlay.tiff` -- inspect base-confidence masks and detected rebate/frame-gap regions
   - `stitch_overlap_12_*.tiff`, `stitch_overlap_21_*.tiff`, `stitch_seam_overlay.tiff` -- inspect both stitch hypotheses and the chosen seam
   - `stitched.tiff` -- verify stitch seam quality
   - `phase3_positive.tiff` -- after orange mask removal in optical-density units; values can exceed display white before direct-density render normalization
   - `phase3_positive_passthrough.tiff` -- positive-mode normalized RGB passthrough before color mapping
   - `phase3_linear_division.tiff` -- negative-mode diagnostic linear-domain inversion view
   - `phase4_ica.tiff` -- after dye separation (colors may look odd, that's normal)
   - `phase46_prophoto.tiff` -- after color space mapping (should look reasonable)
   - `phase46_scene_referred_prophoto_float.tiff` -- 32-bit float scene-referred ProPhoto buffer for inspecting highlight latitude before tone mapping
   - `phase46_color_candidate_comparison.tiff` -- compare selected/image-derived/calibrated color candidates when available
   - `phase46_gamut_clipping_map.tiff` -- inspect selected-transform high/low clipping and preserved in-gamut regions
   - `tone_curve_lut.json` -- the fitted S-curve, plot if curious

## Validate Real Pairs

For the local LOGAN fixture, produce a compact summary without committing rendered outputs:

```powershell
cargo run --bin scanstitch-validate -- --fixture logan
```

This writes `output/validation/logan/summary.json` and `summary.md`. See `docs/validation.md`
for custom pairs, report-only summarization, and the fields to compare between runs.

To fail fast on drift from the tracked LOGAN compact baseline:

```powershell
cargo run --release --bin scanstitch-validate -- `
  --fixture logan `
  --compare-summary tests/fixtures/baselines/logan_summary_baseline.json `
  --strict
```

For fresh normal-vs-validation comparisons, use new output directories before rerunning:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$fresh = "output/fresh_$stamp"
$validation = "output/validation/logan_$stamp"
cargo run --release --bin scanstitch -- LOGAN043.tif LOGAN044.tif -o $fresh --force-stitch
cargo run --release --bin scanstitch-validate -- --fixture logan --output-dir $validation --force-stitch
cargo run --release --bin scanstitch-validate -- `
  --fixture logan `
  --report "$fresh/report.json" `
  --compare-report "$validation/report.json" `
  --summary-json "$fresh/compare-summary.json" `
  --summary-md "$fresh/compare-summary.md" `
  --strict
```

The comparison should report `comparison.status = comparable`, matching dimensions, and no stale render artifacts before visual review.

For quick phase timing on a synthetic image:

```powershell
cargo run --release --bin scanstitch-bench -- --width 640 --height 360 --iterations 3
```

For a positive/slide TESTROLL validation pass:

```powershell
cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --input-mode positive --bit-depth 16 --output-dir output/testroll_positive
```

## Troubleshooting

**Wrong colors / muddy result**
- Check `--bit-depth`. CoolScan 4000 RAW TIFFs are 14-bit padded to 16-bit containers. Using `--bit-depth 16` on 14-bit data will miscalculate the density conversion. Default is 14, which is correct for CoolScan 4000.

**Stitch failed / "NCC score below threshold"**
- Check `report.json` for both stitch hypotheses. If one ordering scores well but is rejected, the report now includes the search ranking, `search.selection_reason`, `search.evaluated_candidates`, `search.max_overlap_considered`, `search.top_candidates[].correspondence_score`, `search.top_candidates[].prior_weight`, `validation.evidence_score`, `validation.prior_weight`, `overlap_support_score`, `vertical_offset_plausibility_score`, `local_consistency_score`, `plausibility_score`, and the explicit rejection reason.
- Look at `comp1_cropped.tiff` and `comp2_cropped.tiff` -- do they actually share content?
- Inspect `stitch_overlap_12_overlay.tiff` / `stitch_overlap_21_overlay.tiff` to confirm whether the overlap and seam are plausible.
- Try `--force-no-stitch` and process the better component alone.

**ICA did not converge**
- Check `report.json` for the iteration count and convergence status.
- Try increasing `--ica-max-iter 200` or relaxing `--ica-tol 1e-4`.
- On very uniform images (blank sky, etc.), ICA may struggle because there aren't enough independent sources to separate. The output should still be usable.

**Border removal ate real content**
- Run with `--debug` and check `comp1_cropped.tiff`. If the image looks over-cropped, the scanner border was not cleanly separated from content. This is rare with CoolScan 4000 scans.

**Out of memory**
- CoolScan 4000 scans are ~130MB each. The pipeline needs ~2-3GB RAM for the full pipeline (density arrays, ICA passes). Close other applications or process on a machine with >= 8GB RAM.

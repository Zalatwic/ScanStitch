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

For frames you know are split across both files:
```bash
RUST_LOG=info ./target/release/scanstitch left.tiff right.tiff -o output/ --debug --force-stitch
```

For frames you know are NOT split (just process the first one):
```bash
RUST_LOG=info ./target/release/scanstitch frame.tiff dummy.tiff -o output/ --force-no-stitch
```

## Interpret Output

Look at `output/`:

1. **`output.tiff`** -- Your final 16-bit positive. Open in RawTherapee, darktable, or Photoshop. ProPhoto RGB color space, D50 white point.

2. **`report.json`** -- Machine-readable pipeline log. Check:
   - `phases[].success` -- did each phase succeed?
   - `phases[].confidence` -- how confident is the result?
   - `phases[].warnings` -- anything suspicious?
   - The `fastica` phase: did it converge? How were the separated channels normalized?
   - The `stitch` phase: what were the `[1|2]` and `[2|1]` scores, which hypothesis was chosen, what thresholds were applied, what did `search.top_candidates`, `search.selection_reason`, `search.evaluated_candidates`, `search.max_overlap_considered`, `search.top_candidates[].correspondence_score`, `search.top_candidates[].prior_weight`, `validation.evidence_score`, `validation.prior_weight`, `validation.overlap_support_score`, `validation.vertical_offset_plausibility_score`, `validation.local_consistency_score`, and `validation.plausibility_score` say, and what explicit rejection reason was recorded if stitching failed?
   - The `colorspace_mapping` phase: check `channel_anchor_counts`, `channel_anchor_min_count`, `channel_anchor_low_support`, `weak_anchor_fallback_used`, `mapping_strategy`, `render_input_source`, candidate gamut clipping, and `exposure_scale` before trusting matrix-derived color changes.
   - The `tone_mapping` phase: check `shadow_saturation_median/p95`, `bright_neutral_saturation_median/p95`, `bright_saturated_saturation_median/p95`, `midtone_luminance_percentiles`, `midtone_saturation_median/p95`, and RGB medians for color-quality review.
   - The `working_image_select` / `density_inversion` / `fastica` / `colorspace_mapping` / `tone_mapping` phases: was base confidence low enough to trigger warnings or cap downstream confidence?

3. **Debug images** (if `--debug`):
   - `comp1_cropped.tiff` / `comp2_cropped.tiff` -- verify borders were removed correctly
   - `*_base_mask_*.tiff`, `*_base_columns.tiff`, `*_base_overlay.tiff` -- inspect base-confidence masks and detected rebate/frame-gap regions
   - `stitch_overlap_12_*.tiff`, `stitch_overlap_21_*.tiff`, `stitch_seam_overlay.tiff` -- inspect both stitch hypotheses and the chosen seam
   - `stitched.tiff` -- verify stitch seam quality
   - `phase3_positive.tiff` -- after orange mask removal (should look like a rough positive)
   - `phase3_linear_division.tiff` -- diagnostic linear-domain inversion view
   - `phase4_ica.tiff` -- after dye separation (colors may look odd, that's normal)
   - `phase46_prophoto.tiff` -- after color space mapping (should look reasonable)
   - `tone_curve_lut.json` -- the fitted S-curve, plot if curious

## Validate Real Pairs

For the local LOGAN fixture, produce a compact summary without committing rendered outputs:

```bash
cargo run --bin scanstitch-validate -- --fixture logan
```

This writes `output/validation/logan/summary.json` and `summary.md`. See `docs/validation.md`
for custom pairs, report-only summarization, and the fields to compare between runs.

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

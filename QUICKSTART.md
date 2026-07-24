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

To extract chart measurements before creating that library:

```bash
cargo run --bin scanstitch-calibrate -- sample-target \
  --manifest chart-sampling.json \
  --output scanner-target.json \
  --preview-dir output/chart-sampling
cargo run --bin scanstitch-calibrate -- scanner-target \
  --library calibration --profile-id coolscan-4000-vuescan-raw \
  --measurements scanner-target.json
```

Start from `docs/calibration-target-sampling.example.json`. Its zero XYZ values are deliberately
non-fit-ready placeholders: replace every value with the factual dataset for your physical target.
Use independent training and held-out chart scans. The first run exits nonzero while still writing every overlay,
`sampling-report.json`, and a paste-ready `corner-review-manifest.json`. Inspect that the outer
grid and every patch box align with the exact decoded capture; separately compare every patch ID,
position, and XYZ/Lab value with the named reference source. Then set
`corner_review.approved=true` for each capture and `reference_review.approved=true` in the generated
manifest and rerun it.
The decoded-pixel, corner-coordinate, and overlay-pixel hashes make any later pixel, orientation,
precision, geometry, grid-layout, or overlay-rendering change invalidate the approval. Rejected or
unapproved sampling deliberately does not create the measurement file. The independent
`reference_patch_sha256` likewise invalidates approval after any patch identity, position, colour
value, or XYZ/Lab representation change.
The fitter always preserves a matrix fallback, may select a root-polynomial model with 18/18 or
more disjoint patches, and evaluates a smooth residual 5x5x5 LUT only with at least 108 training and
48 held-out patches (7x7x7 requires 500/96). A typed LUT is executable only after held-out accuracy,
support, smoothness, conditioning, and deterministic-refit gates pass; ordinary `lut_3d` metadata is
audit-only.

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

3. **`review_srgb.png`** -- In perfect mode, a display-ready RGB8 proof. Out-of-sRGB ProPhoto colours are moved toward the neutral axis in D50 CIELAB while preserving lightness and hue, then encoded as sRGB D65 with an explicit standard sRGB ICC profile.

4. **`report.json`** -- Machine-readable pipeline log. Check:
   - `metadata.generated_at`, `metadata.cli_args`, and `metadata.output_path` -- confirm the report belongs to the render you are inspecting.
   - `phases[].success` -- did each phase succeed?
   - `phases[].confidence` -- how confident is the result?
   - `phases[].warnings` -- anything suspicious?
   - The `load` phase: check `decode_fidelity_status`, the weakest-component phase confidence, and every `components[].decode.decode_fidelity`. An 8-bit source expanded into a 14/16-bit buffer remains usable but cannot regain tonal levels absent from the scan, so inspect `precision_confidence` and `limiters` before judging dynamic range.
   - The `density_inversion` phase for negative input: require `operation_status=held_out_measured_response_supported` before treating density confidence as physical reconstruction evidence. `review_required_unmeasured_frame_response` and `review_required_unmeasured_unit_slope_response` deliberately report `negative_response_model_confidence=0.0`, even when the film-base estimate is strong and inversion arithmetic succeeded. The phase confidence is the minimum of film-base and response-model evidence; runtime curve coverage is checked later under `colorspace_mapping.negative_response_reconstruction`.
   - The `fastica` phase: did it converge, how were the separated channels normalized, and is `physical_separation_evidence_evaluated=false`? Numerical convergence is retained separately but does not establish dye identity; trust the eventual render only through the downstream ICA/direct-density comparison and measured negative-response/colour evidence.
   - The `stitch` phase: what were the `[1|2]` and `[2|1]` scores, which hypothesis was chosen, what thresholds were applied, what did `search.top_candidates`, `search.selection_reason`, `search.evaluated_candidates`, `search.max_overlap_considered`, `search.top_candidates[].correspondence_score`, `search.top_candidates[].prior_weight`, `validation.evidence_score`, `validation.prior_weight`, `validation.overlap_support_score`, `validation.vertical_offset_plausibility_score`, `validation.local_consistency_score`, and `validation.plausibility_score` say, did `seam_exposure_correction` apply or stay identity, and did `seam_blend.detail_consistency` or `seam_blend.review_required` identify a focus/grain/detail transition even though geometry succeeded? For three or more scans, inspect every `pair_merges[]` entry; the compact validation summary conservatively aggregates their exposure, residual, gradient, and multi-scale-detail evidence.
   - The `colorspace_mapping` phase: check `color_confidence_status`, `selected_mapping_evidence_confidence`, `negative_reconstruction_evidence_confidence`, `negative_reconstruction_confidence_status`, `calibration.status`, `calibration.color_mapping_application`, `calibration.scanner_profile`, `calibration.roll_profile`, `calibration.film_stock`, `calibration.nearest_roll_candidates`, `calibration.nearest_film_candidates`, `color_decision_summary`, `candidate_quality_scores`, `quality_components`, candidate `rendered_tone_quality` and `color_model_quality`, `selected_quality_score`, `technical_safety_score`, `color_fidelity_score`, `candidate_risk`, `tone_color_trust_state`, `calibration_acceptance`, `neutral_estimate_quality`, `neutral_sample_rejections`, `dominant_anchor_sample_rejections`, `dominant_anchor_quality`, optional XYZ/DeltaE `reference_patch_evaluation`, optional typed-model `nonlinear_color_model` scene-support decision, `neutral_trim_before_after`, `mapping_strategy`, `render_input_source`, `direct_density_render_density_scale`, candidate gamut clipping, and `exposure_scale` before trusting color changes. `calibration.status=applied` can describe a usable record or separately applied scanner/response component; only `calibration.color_mapping_application.applied=true` with `selection_status=accepted|forced` says that its colour mapping reached the final colorspace buffer. `fallback_only`, review-required mappings, blind/unmeasured negative reconstruction, and measured curves outside runtime support report complete colour confidence `0.0` even though a diagnostic render completed.
   - The `tone_mapping` phase: check `tone_confidence_status`, `upstream_color_confidence`, `requested_grain_evidence_confidence`, `tone_output_confidence_status`, `tone_output_review_required`, the input/fitted/rendered p05-p95 ranges and render-to-fitted ratio, maximum post-tone channel clipping, `tone_fit_policy`, `auto_exposure_ev`, `render_exposure_ev`, `color_protection_reason`, `color_trust_state`, `highlight_neutral_chroma_enabled`, `shadow_chroma_enabled`, saturation/luminance band diagnostics, RGB medians, `noise_reduction_*`, `grain_reduction.detail_retention`, and `high_frequency_grain`. The universal tone gate catches near-zero/relative collapse and catastrophic clipping; use approved fixture-specific floors to judge scene-class beauty. When grain reduction is on, require supported coherent probes for a detail-bearing fixture and inspect luminance/chroma median and p10 retention. In positive mode, `auto_exposure_ev` should be non-positive. For `scanstitch-ui` saves, also check `interactive_controls`.
   - The `save` phase: check `output_modified_at`, `output_shape`, `output_color_space`, `output_encoding`, `output_icc_profile`, `output_sha256`, `master_scene_referred_requested`, `master_scene_referred_path`, `master_scene_referred_sha256`, `review_srgb_requested`, `review_srgb_path`, `review_srgb_sha256`, `review_srgb_encoding`, `review_srgb_icc_profile`, `review_srgb_gamut_mapping`, `artifact_sha256_policy`, `artifact_commit_policy`, `tone_output_review_required`, `render_review_status`, `render_reviewable`, `delivery_confidence_status`, validation summary `output_file_*`, `master_scene_referred_file_*`, and `review_srgb_file_*`/`review_srgb_gamut_*` inspection fields, `overwrote_existing_output`, and `stale_render_artifact_count` before comparing files in an existing output directory. In schema v4, require every requested artifact's independently recomputed `*_file_sha256_matches_report` value to be `true`. `artifact_commit_policy` confirms per-file same-directory staged replacement, while explicitly reporting no cross-artifact transaction or `fsync` durability guarantee. Save `success` means artifact I/O completed; confidence `1.0` is reserved for a reviewable delivery. `review_required_tone_output` is diagnostic output, not an approved final image.
   - The `working_image_select` / `density_inversion` / `fastica` / `colorspace_mapping` / `tone_mapping` phases: was base confidence low enough to trigger warnings or cap downstream confidence?
   - For `--input-mode positive`, confirm `working_image_select.metrics.input_mode_suitability_status` is `accepted_positive_input` or `accepted_warm_positive_input`, `density_inversion.metrics.skipped=true`, `fastica.metrics.skipped=true`, `colorspace_mapping.metrics.render_input_source=positive_scan_rgb`, and, for uncalibrated auto-mode scans, `colorspace_mapping.metrics.mapping_strategy=positive_rgb_passthrough`. `review_required_negative_like_positive_input` keeps diagnostic output but makes final delivery `review_required_input_mode`.

For unattended delivery, add `--require-reviewable`. The pipeline still retains diagnostic output
and `report.json`, but exits nonzero unless the final review status and boolean both say the render
is reviewable; missing or inconsistent review evidence also fails. Current schema-v4 reports hash
the exact saved primary TIFF and every requested master/proof after reopening them, and the gate
recomputes those hashes independently.

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

## Validate Real Scans and Pairs

For the local LOGAN fixture, produce a compact summary without committing rendered outputs:

```powershell
cargo run --bin scanstitch-validate -- --fixture logan
```

This writes `output/validation/logan/summary.json` and `summary.md`. See `docs/validation.md`
for custom pairs, report-only summarization, and the fields to compare between runs. Current
summaries retain `tone_output_confidence_status`, review/evidence state, range retention, and
clipping maxima; `--fail-on tone` selects regressions in those fields.

For one independent scan, pass only `--component1`; a duplicate `--component2` is not needed:

```powershell
cargo run --bin scanstitch-validate -- --component1 scan.tif --input-mode positive --bit-depth 16 --output-dir output/validation/single
```

Add `--require-reviewable` to a direct run, an existing `--report`, a fixture suite, or a roll suite
when automation must accept only a final `render_review_status=reviewable` and
`render_reviewable=true`. Reports, TIFFs, and compact summaries are written before a rejected run
exits nonzero. The gate also reopens the output TIFF to require its reported dimensions, RGB16
storage, ICC profile, and schema-v4 SHA-256; reopens every promised scene-referred master and sRGB
review proof to require the reported dimensions, storage, profile, and schema-v4 SHA-256; and
requires an explicit zero stale-artifact count. These hashes bind unchanged reports to unchanged
artifacts; `report.json` is not signed, so hostile edits to both the report and artifacts—or a
schema downgrade—remain outside this delivery gate. Pin a completed hash-bound render-review
manifest in the fixture registry when authenticated human acceptance is required.
Strict fixture suites also fail an otherwise stable non-reviewable result unless the
fixture explicitly pins `expectations.render_reviewable=false` as an intentional diagnostic case;
the stronger `--require-reviewable` gate rejects even that case.

For a corpus-level audit, use `--fixture-coverage --strict`. Require
`min_stitch_normalization_contract_fixtures` so an accepted decision alone cannot claim seamless
coverage; the counted fixture must also pin exposure normalization, multiband blending, overlap and
gradient ceilings, and supported multi-scale detail continuity.

Require `min_geometry_preparation_contract_fixtures` for a rotated four-border fixture. A complete
contract pins applied deskew on every component, no geometry review, at least 0.90 retained area,
four removed border edges per component, a crop-retained range beginning at 0.50 and ending below
1.0, and no rejected crop. This distinguishes measured preparation from a no-op that happened to
produce an image.

Also require `min_geometry_accuracy_contract_fixtures` when crop correctness matters. Supply
`deskew_correction_degrees_expected` with a tolerance of at most 0.05 degrees and
`border_crop_components_expected` with one annotated top/bottom/left/right removal for every input;
each crop tolerance must be at most 4 pixels. Preparation coverage says every edge was processed;
accuracy coverage says each resulting boundary agrees with known truth.

Require `min_orientation_accuracy_contract_fixtures` to pin decoded orientation. Add one
`orientation_components_expected` record per input with `upright_approved: true`, the exact
`decoded_pixel_sha256`, source-tag presence/value, effective transform, applied flag, source
dimensions, and final dimensions. For example, EXIF orientation 6 with no explicit correction must
be `rotate_90_clockwise`, applied, and swap width/height. If metadata-correct pixels are still
semantically upside down, add `--orientation-correction rotate-180` (or the matching fixture
`orientation_correction` value); the reported transform and hash then bind the composed result. Run
once, visually approve that the decoded component is upright, then copy its digest
from `summary.json.input_orientation.components[]`; never copy the placeholder hashes from the
example registry. Strict execution binds that approval to the decoded pixels and fails if their
orientation, precision, dimensions, or samples change.

Create the review package without a full render:

```powershell
cargo run --bin scanstitch-validate -- --fixture-registry fixtures.json --fixture frame-001 --write-orientation-review output/orientation-review/frame-001
```

For one standalone positive scan, use `--component1 scan.tif --input-mode positive --bit-depth 16`.
Add `--orientation-correction rotate-180` when that is the factual metadata-relative correction.
Inspect every generated PNG, then copy `orientation_components_expected` from
`orientation-review.json` and change `upright_approved` to `true` only for previews you approve.

For negative colour acceptance, require `min_negative_reconstruction_contract_fixtures`. The
counted fixture must use a credible measured base and an accepted measured roll response with 3x3
dye-crosstalk separation, monotone nonlinear film curves, held-out DeltaE00 improvement over the
unit-slope fallback, bounded noise gain and extrapolation, signed headroom, direct-density input,
safe/trusted colour decisions, and linear ProPhoto output. Use the exact fields and conservative
limits in `docs/validation.md`; an unmeasured fallback negative cannot satisfy this contract.

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

Each roll frame preserves the final `render_review_status` / `render_reviewable` decision and the
complete rendered-tone status, review reason, evidence confidence, input/fitted/rendered range
relationship, and clipping maxima. A non-reviewable final render, a tone-output review, or missing
decision evidence makes that frame `review_required`; `--fail-on tone` selects the tone-specific
roll and roll-comparison issues. Older roll baselines without these fields remain comparable, while
a current run that loses previously present evidence fails closed.

For a controlled grain pass, add `--grain-reduction on --grain-strength 0.6 --grain-scale 1.0` and
write to a fresh directory. The roll JSON/Markdown reports evaluated, supported, and review counts;
independent luminance/chroma support and minimum probe counts; and median/p10 retention mean/min.
When using `--compare-roll-suite`, keep all grain controls identical. Lost support, new detail review,
or a material retention drop is a comparison issue and is selected by `--fail-on grain`.
For a durable acceptance corpus, pin grain mode, nonzero strength, scale, both 64-probe minima, and
both 0.70 p10 floors in the fixture itself. Also pin conservative, positive
`grain_reduction_applied_ratio_min`, `grain_reduction_flat_luma_p95_reduction_ratio_min`, and
`grain_reduction_flat_chroma_p95_reduction_ratio_min` values measured from that approved fixture.
Require `min_grain_reduction_enabled_fixtures`, `min_grain_detail_contract_fixtures`, and
`min_grain_reduction_effect_contract_fixtures` in registry coverage rather than relying on CLI
defaults or treating an effect-free pass as denoise evidence.

For dynamic-range acceptance, measure an approved render of each scene class and pin
`post_scale_preserved_ratio_min`, `render_luminance_range_p05_p95_min`, and both
`post_chroma_compression_clipped_*_ratio_max` ceilings. Also pin `render_review_status=reviewable`,
`render_reviewable=true`, `tone_output_confidence_status=supported_render_tonal_distribution`,
`tone_output_review_required=false`, and positive tone-evidence-confidence and render-to-mapped
range-retention floors. Then require
`min_render_dynamic_range_contract_fixtures`. Do not reuse one contrast floor for high-key, fog,
night, and full-sun scenes; the contract prevents regression relative to approved intent, not a
forced universal histogram.

For final visual acceptance, render in perfect mode and create a non-self-approving review draft:

```powershell
cargo run --bin scanstitch-validate -- --fixture frame-001 --report output/frame-001/report.json --write-render-review output/render-review/frame-001
```

Inspect the exact hash-listed sRGB proof and full-resolution master/output, then complete every
applicable crop, orientation, stitch/seam, colour, tone, grain/detail, and overall-preference
decision with notes. Pin the completed `render-review.json` as `render_review` and
`render_review_sha256` in the fixture and require `min_approved_render_review_fixtures`. Coverage
rejects pending decisions, a technically non-reviewable render, changed source/report/output bytes,
or a manifest copied from a different ordered input set.

## Troubleshooting

**Wrong colors / muddy result**
- Check `--bit-depth`. CoolScan 4000 RAW TIFFs are 14-bit padded to 16-bit containers. Using `--bit-depth 16` on 14-bit data will miscalculate the density conversion. Default is 14, which is correct for CoolScan 4000.

**Stitch failed / "NCC score below threshold"**
- Check `report.json` for both stitch hypotheses. If one ordering scores well but is rejected, the report now includes the search ranking, `search.selection_reason`, `search.evaluated_candidates`, `search.max_overlap_considered`, `search.top_candidates[].correspondence_score`, `search.top_candidates[].prior_weight`, `validation.evidence_score`, `validation.prior_weight`, `overlap_support_score`, `vertical_offset_plausibility_score`, `local_consistency_score`, `plausibility_score`, and the explicit rejection reason.
- Look at `comp1_cropped.tiff` and `comp2_cropped.tiff` -- do they actually share content?
- Inspect `stitch_overlap_12_overlay.tiff` / `stitch_overlap_21_overlay.tiff` to confirm whether the overlap and seam are plausible.
- Try `--force-no-stitch` and process the better component alone.

**Stitch succeeded but save status is `blocked_geometry_review`**
- Inspect `seam_blend.review_reason` and `detail_consistency.scales`. A repeated 2x high-pass-energy imbalance across both disjoint checkerboard partitions means the scans may differ in focus, scanner sharpness, or grain/noise. The union is retained for inspection, but ScanStitch does not invent a blur or sharpening correction without measured evidence.

**Save status is `review_required_grain_detail`**
- The requested grain pass measurably reduced supported coherent luminance or opponent-colour edge contrast beyond its conservative limit. Inspect `grain_reduction.detail_retention`, especially probe counts, median retention, p10 retention, and `review_reason`; compare a fresh grain-off render or reduce `--grain-strength` / `--grain-scale` rather than treating the smoothed output as reviewable.

**ICA did not converge**
- Check `report.json` for the iteration count and convergence status.
- Try increasing `--ica-max-iter 200` or relaxing `--ica-tol 1e-4`.
- On very uniform images (blank sky, etc.), ICA may struggle because there aren't enough independent sources to separate. The output should still be usable.

**Border removal ate real content**
- Run with `--debug` and check `comp1_cropped.tiff`. If the image looks over-cropped, the scanner border was not cleanly separated from content. This is rare with CoolScan 4000 scans.

**Out of memory**
- CoolScan 4000 scans are ~130MB each. The pipeline needs ~2-3GB RAM for the full pipeline (density arrays, ICA passes). Close other applications or process on a machine with >= 8GB RAM.

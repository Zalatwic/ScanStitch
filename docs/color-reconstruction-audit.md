# Colour Reconstruction Audit

Status on 2026-05-12: the implementation has substantial synthetic, schema, LOGAN baseline, and
TESTROLL smoke coverage, but the overall colour reconstruction objective is not complete. The missing evidence is a
real multi-scene, multi-film-stock CoolScan validation corpus with tracked baselines and calibration
cases.

## Objective-To-Artifact Checklist

| Requirement | Current evidence | Coverage status |
|-|-|-|
| Produce ProPhoto RGB positives from CoolScan RAW TIFF inputs | `README.md` pipeline contract, `src/tiff_io.rs`, `src/pipeline.rs`, `src/colorspace.rs`, `src/tonemap.rs`; final 16-bit linear ProPhoto output plus debug-only 32-bit float scene-referred ProPhoto artifact; `tests/test_tiff_io.rs`, `tests/test_pipeline.rs`, release LOGAN validation output | Covered for implementation and one LOGAN fixture; not proven across varied real scans |
| Physical negative-film model: orange-mask/base-density removal, dye separation, colour mapping, exposure normalization, gamut handling, tone mapping | `src/density.rs`, `src/ica.rs`, `src/colorspace.rs`, `src/tonemap.rs`; `tests/test_density.rs`, `tests/test_ica.rs`, `tests/test_colorspace.rs`, `tests/test_tonemap.rs` | Covered by unit/synthetic tests; needs broader real-image confirmation |
| Explicit scanner, roll, film-stock, and per-image decision model with precedence/confidence/rejection reasons | `src/color_calibration.rs`, `src/bin/scanstitch-calibrate.rs`, `docs/color-calibration.md`, `docs/color-calibration-library-record.schema.json`; `tests/test_color_calibration.rs`, `tests/test_pipeline.rs`; fixture-suite expectations for `calibration_confidence_min`, `calibration_matrix_condition_number_max`, `calibration_rejection_details_required`, and `selection_rejections_required` | Covered for loader, ingest, profile selection, roll matching, film-stock ranking, rejection diagnostics, and suite-level calibration-safety gates |
| Calibration-aware paths can outperform image-derived mapping when valid and are rejected when unsafe/weak | `src/colorspace.rs`, `src/color_calibration.rs`; `tests/test_colorspace.rs` cases for calibrated wins, weak quality rejection, neutral/reference regression rejection, unsafe calibrated-mode failure | Covered in synthetic decision cases; real calibrated fixture coverage is still pending |
| Robust image-derived fallback when anchors are sparse, biased, clipped, dusty, border-contaminated, or scene-dependent | `src/colorspace.rs`; `tests/test_colorspace.rs`; synthetic suite cases for dirty/clipped/border rejection, sparse anchors, biased scene anchors, high-key headroom | Covered synthetically; needs more real-scene evidence |
| Candidate selection uses measurable quality signals | `candidate_quality_scores`, `quality_components`, `technical_safety_score`, `color_fidelity_score` in `src/colorspace.rs`; `docs/report-schema.md`; synthetic-suite selected model evidence; fixture-suite expectations for `selected_quality_score_max`, `technical_safety_score_max`, `color_fidelity_score_max`, `memory_color_penalty_max`, `spatial_consistency_penalty_max`, `selected_runner_up_quality_delta_min`, `candidate_acceptance_signatures_required`, and `selected_candidate_rank`; `tests/test_colorspace.rs`, `tests/test_validation.rs` | Covered for neutral accuracy, reference-patch XYZ/DeltaE76/CIEDE2000 error reporting and decision gating, hue-family residuals, density monotonicity, saturation preservation, memory-colour plausibility, spatial cast consistency, gamut preservation, clipping risk, tone-colour stability, and candidate-rank/signature stability |
| Reports and debug artifacts make decisions inspectable | `src/report.rs`, `src/validation.rs`, `docs/report-schema.md`, `README.md` output artifact list; `phase46_color_candidate_comparison.tiff`, `phase46_gamut_clipping_map.tiff`, `phase46_scene_referred_prophoto_float.tiff`; coverage requirements for `min_debug_artifact_expectation_fixtures` and `required_debug_artifact_kinds`; fixture-suite expectations for `render_input_reason_contains`, `selected_mapping_reason_contains`, `highlight_chroma_compressed_ratio_max`, `highlight_neutral_chroma_compressed_ratio_max`, `shadow_chroma_compressed_ratio_max`, `debug_artifacts_required`, and `debug_artifact_kinds_required`; `tests/test_validation.rs` | Covered for report fields, debug artifact references, compact summaries, baseline comparisons, rationale text checks, tone-protection ratio gates, named debug-artifact freshness gates, high-latitude artifact export, and corpus-level debug-artifact expectation coverage |
| Validation harness proves synthetic edge cases and real fixtures including LOGAN without hidden regressions | `src/bin/scanstitch-validate.rs`, `src/validation.rs`, `docs/validation.md`, `docs/validation-fixtures.schema.json`, `docs/validation-summary-baseline.schema.json`, `tests/fixtures/baselines/logan_summary_baseline.json`; `tests/test_validation.rs`; fixture coverage hash gates for `component*_sha256`, `summary_baseline_sha256`, `calibration_profile_sha256`, and `calibration_library_sha256`; fixture coverage `repair_plan` entries with action/path/detail fields; fixture-suite gates for stitch/base/output-space/render-input/candidate/calibration/gamut/tone/debug expectations, including `reference_patch_count_min`, `reference_patch_hue_family_regression_count_max`, `reference_patch_selected_regresses_image_derived`, `reference_patch_delta_e2000_delta_vs_image_derived_max`, `reference_patch_max_delta_vs_image_derived_max`, `reference_patch_delta_e_max_delta_vs_image_derived_max`, `reference_patch_delta_e2000_max_delta_vs_image_derived_max`, `reference_patch_rms_delta_e_max`, and `reference_patch_rms_delta_e2000_max`; per-fixture suite `coverage_validation_ready`, `coverage_issues`, `coverage_action_items`, and `fixture_suite:<name>:coverage_not_validation_ready` for corpus readiness and repair work | Covered for synthetic suite and LOGAN baseline; not complete for a multi-scene/multi-film-stock corpus |
| Avoid silent automatic changes that cannot be justified by diagnostics | Calibration profile/library rejection details, candidate acceptance diagnostics, `candidate_risk`, `tone_color_trust_state`, strict baseline and registry schemas | Covered for implemented paths; future new paths should add report fields and validation deltas before default enablement |

## Current Verification Gates

Use these gates after colour-pipeline changes:

```powershell
$env:TEMP = (Join-Path (Get-Location) 'target\tmp'); $env:TMP = $env:TEMP; New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null; cargo fmt
$env:TEMP = (Join-Path (Get-Location) 'target\tmp'); $env:TMP = $env:TEMP; New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null; cargo test --all-targets
$env:TEMP = (Join-Path (Get-Location) 'target\tmp'); $env:TMP = $env:TEMP; New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null; cargo clippy --all-targets -- -D warnings
$env:TEMP = (Join-Path (Get-Location) 'target\tmp'); $env:TMP = $env:TEMP; New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null; cargo run --bin scanstitch-validate -- --synthetic-color-suite --strict --quiet --summary-json output/validation/synthetic-color-suite.json --summary-md output/validation/synthetic-color-suite.md
$env:TEMP = (Join-Path (Get-Location) 'target\tmp'); $env:TMP = $env:TEMP; New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null; cargo run --bin scanstitch-validate -- --fixture-registry docs\validation-fixtures.example.json --fixture-coverage --compute-fixture-hashes --write-fixture-hash-registry output/validation/validation-fixtures.with-hashes.json --summary-json output/validation/fixture-coverage-hashes.json --summary-md output/validation/fixture-coverage-hashes.md
$env:TEMP = (Join-Path (Get-Location) 'target\tmp'); $env:TMP = $env:TEMP; New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null; cargo run --release --bin scanstitch-validate -- --fixture-registry docs\validation-fixtures.example.json --fixture-suite --strict --debug --summary-json output/validation/fixture-suite.json --summary-md output/validation/fixture-suite.md
$env:TEMP = (Join-Path (Get-Location) 'target\tmp'); $env:TMP = $env:TEMP; New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null; cargo run --release --bin scanstitch-validate -- --fixture logan --compare-summary tests\fixtures\baselines\logan_summary_baseline.json --strict
git diff --check
```

## Current Local Evidence

On 2026-05-12, the synthetic colour gate
`cargo run --bin scanstitch-validate -- --synthetic-color-suite --strict --quiet --summary-json target\tmp\synthetic-color-suite.json --summary-md target\tmp\synthetic-color-suite.md`
passed with `status=passed`, 19 passed cases, and `issues=[]`. The cases cover calibrated profile
selection, weak/unsafe calibration rejection, scanner-prior selection/rejection, sparse and biased
anchor review, dirty-edge sample rejection, high-key headroom preservation, destructive-gamut render
fallback, tone colour-protection policy gates, reference-patch regression rejection, and forced
unsafe-calibration failure.

On 2026-05-08, the local LOGAN gate
`cargo run --release --bin scanstitch-validate -- --fixture logan --compare-summary tests\fixtures\baselines\logan_summary_baseline.json --strict`
passed with `summary_baseline_status=comparable` and `issues=none`. The generated summary reported
`output_color_space=linear_prophoto_rgb_d50`, an embedded matching output ICC profile, no stale
render artifacts, `stitch.decision=accepted`, `colorspace.mapping_strategy=image_derived_matrix`,
`colorspace.calibration_status=not_configured`, and `tone.color_trust_state=review_required`.
The same LOGAN command with `--debug` also passed strict comparison and emitted fresh
`candidate_comparison` and `gamut_clipping_map` debug artifacts with
`debug_artifact_invalid_count=0`.

On 2026-05-12, the supplied local `TESTROLL/` scanner roll was inspected as VueScan DNG from
`Nikon LS-4000`. The DNG loader selects the full-resolution uncompressed three-channel
`DNG_LINEAR_RAW16` SubIFD instead of the embedded RGB thumbnail, and the load phase now reports
primary-IFD scanner metadata such as Make/Model/Software/UniqueCameraModel, DNG `ColorMatrix1`,
`AsShotNeutral`, black/white levels, calibration illuminants, and matrix condition diagnostics. In
auto colour mode, when no explicit scanner/roll calibration is configured, a valid DNG
`ColorMatrix1` is used only as a weak scanner-constrained advisory prior; it becomes another scored
candidate instead of forcing the render. A representative single-frame smoke run using
`RAW_0023.dng` as both components with `--force-no-stitch --bit-depth 16 --debug` produced linear
ProPhoto RGB D50 output with a valid embedded ICC profile and
`stale_render_artifact_count=0`. The frame has no detectable left/right or top/bottom rebate/base strip, so the report records
`base_estimate_source=high_transmittance_fallback`, a finite fallback base estimate of roughly
`[26422, 12355, 7730]` in 16-bit scanner samples, `raw_base_proxy_confidence=0.0`, and
`raw_base_support_fraction=0.00009639424113108324` (0.010% simultaneous high-transmittance
support). Because the independent per-channel high percentiles barely co-occurred, the fallback
base colour is selected from the top 0.501% jointly high-transmittance pixels instead of combining
unrelated channel outliers. The selected colour decision is now `gamut_trusted_image_matrix_blend`, with the original
image-derived matrix rejected for destructive negative gamut, the trusted blend mixed 93.6% toward
neutral balance, `candidate_risk=safe`, `tone.color_protection_policy=enabled`, and
`tone.color_trust_state=trusted`. The render buffer reports
`render_buffer_domain=scene_referred_linear_prophoto_rgb_d50` and sparse high-headroom preservation
for tone mapping. The debug run emitted fresh `candidate_comparison`, `gamut_clipping_map`, and
`scene_referred_prophoto_float` artifacts with no stale render artifacts. The float artifact
`phase46_scene_referred_prophoto_float.tiff` is a 32-bit IEEE float linear ProPhoto RGB D50 TIFF
with no display normalization; on RAW_0023 it was 222,712,821 bytes, sampled 976,800 pixels for
diagnostics, preserved channel maxima of roughly `[1.024, 1.049, 1.062]`,
`pixel_above_display_white_ratio=0.009605963619121513`, and
`pixel_below_zero_ratio=0.01943790680632786`. The direct-density candidate was evaluated but not
selected because the ICA path had the better selected quality and trust state. This is real DNG ingestion and
unbordered-frame fallback evidence, but it is not calibrated roll proof without scanner/roll records
or reference target evidence, and the base estimate remains untrusted.

On 2026-05-26, roll inventory checks found additional local scan evidence but not a complete
validation corpus. `TESTROLL/` contains 12 readable RGBA16 TIFF frames at `5959x3946` with no
sequence gaps. `OLD_TESTROLL/` contains 39 readable `DNG_LINEAR_RAW16` frames at `5952x3944`, with
one sequence gap covering `RAW_0030` through `RAW_0046`. These rolls can supply more real-image
smoke coverage, but inventory alone does not prove film stock, scene/exposure labels, calibration
records, reference patches, compact baselines, or fixture-suite expectations.

The fixture coverage hash gate
`cargo run --bin scanstitch-validate -- --fixture-registry docs\validation-fixtures.example.json --fixture-coverage --compute-fixture-hashes --write-fixture-hash-registry output\validation\validation-fixtures.with-hashes.json --summary-json output\validation\fixture-coverage-hashes.json --summary-md output\validation\fixture-coverage-hashes.md`
completed with `fixture_coverage_status=review_required`. It inspected 3 registry entries, found
no validation-ready fixtures for the stricter local CoolScan corpus, computed one LOGAN component SHA-256 pair and one LOGAN summary
baseline SHA-256, computed no calibration evidence SHA-256, and emitted non-empty `action_items`
and fixture-level repair actions. The generated `validation-fixtures.with-hashes.json` records the
current LOGAN component and summary-baseline hashes, but it is not a passing strict corpus snapshot.
By contrast, the built-in LOGAN fixture coverage gate now passes as one uncalibrated,
validation-ready regression fixture with pinned component and summary-baseline hashes.
The required debug artifact kinds are now `candidate_comparison`, `gamut_clipping_map`, and
`scene_referred_prophoto_float`, and all remain missing from validation-ready fixture coverage
for the stricter example registry because there are no validation-ready local CoolScan fixtures yet.
The `Local Corpus Scaffold` section in `docs/validation.md` maps those actions to the expected
ignored local paths, including `calibration/scanners/coolscan-4000-vuescan-raw.json`,
`calibration/rolls/logan-roll-2026-05.json`, `local-fixtures/my-split-left.tif`, and
`local-fixtures/uncalibrated-night-left.tif`. It also records that the current LOGAN pair is
`RGBA8 5959x3670` and `RGBA8 5959x3669`, so `repair_component1_tiff_for_fixture:logan` and
`repair_component2_tiff_for_fixture:logan` now have narrower companion actions:
`replace_component1_with_minimum_bit_depth_tiff_for_fixture:logan`,
`replace_component2_with_minimum_bit_depth_tiff_for_fixture:logan`. The one-row height delta is
accepted as stitch-compatible by the built-in LOGAN regression gate; exact-dimension replacements
are still preferred for a strict CoolScan RAW corpus unless a registry documents the rationale.

The current registry fixture suite can be run without `--strict` to inspect all fixture outcomes:
`cargo run --release --bin scanstitch-validate -- --fixture-registry docs\validation-fixtures.example.json --fixture-suite --debug --summary-json output\validation\fixture-suite-current.json --summary-md output\validation\fixture-suite-current.md`.
That run completed with `fixture_suite_status=failed` and `coverage status=review_required`.
LOGAN produced fresh debug artifacts but remained review-required for the example registry because
the registry declares calibration evidence that is missing locally
(`fixture_suite:logan:declared_calibration_not_applied`);
`my-split-frame` and `uncalibrated-night` failed because their component TIFFs are missing.

This is useful real-fixture regression evidence for LOGAN, but it is not calibrated corpus proof and
the synthetic suite is not real-image proof; neither satisfies the multi-scene, multi-film-stock
CoolScan requirement.

## Missing Evidence

The remaining blocker is not another synthetic unit test. It is a validation corpus with enough real
CoolScan scans to exercise the target operating range:

- multiple film stocks, including at least one validation-ready calibrated scanner/roll case and
  one validation-ready uncalibrated image-derived case
- multiple scene classes, including skin, foliage, saturated colour, high-key, deep shadow, and
  mixed neutral/low-neutral scenes
- exposure variation, including normal, overexposed negative, and underexposed negative scans
- tracked compact baselines for every validation-ready fixture
- pinned SHA-256 identity for every validation-ready fixture's component TIFFs, compact baseline,
  and declared calibration profile or calibration library
- fixture-registry coverage requirements that require cross-condition pairs such as
  `film_stock|calibration_case` and `scene_tag|exposure_tag`
- fixture-registry coverage requirements for exact-file reproducibility:
  `min_component_sha256_pairs`, `min_summary_baseline_sha256_fixtures`, and
  `min_calibration_sha256_fixtures`
- fixture-coverage summaries whose `missing_required_film_stocks`,
  `missing_required_scene_tags`, `missing_required_exposure_tags`,
  `missing_required_calibration_cases`, `missing_required_scanner_profiles`,
  `missing_required_roll_profiles`, `missing_required_film_stock_calibration_pairs`, and
  `missing_required_scene_exposure_pairs` fields are empty for validation-ready fixtures
- fixture-coverage summaries whose `action_items` array is empty after all required real-corpus
  film, scene, exposure, calibration, reference, and debug-artifact conditions are satisfied
- fixture-suite entries whose `coverage_validation_ready` values are true, `coverage_issues`
  arrays are empty, and `coverage_action_items` arrays are empty after every fixture has its local
  component TIFFs, compact baseline, calibration evidence, and reference-patch evidence, with no
  remaining `fixture_suite:<name>:coverage_not_validation_ready` issues
- fixture-coverage summaries with no fixture-specific repair actions remaining, including
  `provide_component1_for_fixture`, `provide_component2_for_fixture`,
  `complete_summary_baseline_for_fixture`, `provide_calibration_library_for_fixture`, and
  `add_reference_patches_for_fixture`, plus TIFF-specific repair actions such as
  `replace_component_pair_with_dimension_matched_tiffs_for_fixture` and
  `replace_component1_with_minimum_bit_depth_tiff_for_fixture`
- fixture-coverage entries whose `repair_plan` arrays are empty, or whose remaining plan items map
  every action to explicit local paths and repair details for corpus-supply work
- fixture-coverage summaries whose validation-ready `component_sha256_pair_count`,
  `summary_baseline_sha256_count`, and `calibration_sha256_count` meet the registry requirements
- fixture-coverage summaries whose `debug_artifact_expectation_fixture_count` and
  `debug_artifact_kinds_required` satisfy the required candidate-comparison and gamut-clipping
  artifact coverage
- strict fixture-suite runs with no stale output, ICC mismatch, calibration-source mismatch,
  candidate-rank/signature mismatch, missing decision-rationale text, missing calibration rejection
  details, missing selection rejections, missing named debug artifacts, or colour/tone/gamut
  regression issues

Until that corpus exists and passes `--fixture-coverage --strict` plus `--fixture-suite --strict`,
the broad colour reconstruction goal should remain open.

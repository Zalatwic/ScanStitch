# Colour Reconstruction Audit

Status updated 2026-07-23: the implementation now also preserves signed scene headroom, fits and
held-out-validates measured scanner/roll matrices plus scanner/negative-response models, applies the
exact runtime scanner linearization to both matrix-fit partitions, and rejects ambiguous or mismatched
scanner/roll fit domains. Scanner target fitting now robustly fits and evidence-selects homogeneous
quadratic/cubic root-polynomial colour models plus smooth residual 5x5x5/7x7x7 tetrahedral LUTs,
retains enough training evidence for deterministic loader refit, applies a selected model only inside
measured chromaticity/RGB-volume support, and preserves the next simpler calibrated fallback. Roll
fitting records and verifies the exact scanner matrix/nonlinear transform used before
its XYZ correction. It compares physical direct-density and
ICA paths, prevents automatic direct-density selection on fallback film-base evidence, and uses a
D50 CIELAB lightness/hue-preserving display-gamut mapper. The overall colour reconstruction
objective now also separates evidence-gated/manual technical Bradford white balance from reversible
creative temperature/tint and provides a hash-bound, non-self-approving human final-render review
contract. It is still not complete. The missing evidence is a pinned multi-scene, multi-film-stock
scanner corpus with real target values, approved geometry, and populated calibration cases. See
`validation-corpus-status.md` and `../PROJECT_ASSESSMENT.md`; older evidence below is retained as a
chronological audit trail and must not be read as current ground truth.

## Objective-To-Artifact Checklist

| Requirement | Current evidence | Coverage status |
|-|-|-|
| Produce ProPhoto RGB positives from CoolScan RAW TIFF inputs | `README.md` pipeline contract, `src/tiff_io.rs`, `src/pipeline.rs`, `src/colorspace.rs`, `src/tonemap.rs`; final 16-bit linear ProPhoto output plus debug-only 32-bit float scene-referred ProPhoto artifact; `tests/test_tiff_io.rs`, `tests/test_pipeline.rs`, release LOGAN validation output | Covered for implementation and one LOGAN fixture; not proven across varied real scans |
| Physical negative-film model: orange-mask/base-density removal, dye separation, colour mapping, exposure normalization, gamut handling, tone mapping | `src/density.rs`, `src/ica.rs`, `src/colorspace.rs`, `src/tonemap.rs`; `tests/test_density.rs`, `tests/test_ica.rs`, `tests/test_colorspace.rs`, `tests/test_tonemap.rs` | Covered by unit/synthetic tests; needs broader real-image confirmation |
| Separate technical illuminant reconstruction from creative temperature/tint | `src/white_balance.rs`, `src/cli.rs`, `src/interactive.rs`, `src/pipeline.rs`; `tests/test_white_balance.rs`, `tests/test_interactive.rs`, `tests/test_pipeline.rs`; `white_balance` and `tone_mapping.creative_white_balance` report fields | Covered for evidence gating, manual Bradford adaptation, signed headroom, CLI/UI/sidecar controls, and technical-master separation; real target-based illuminant accuracy still needs corpus validation |
| Scene-adaptive rendering protects neutrals, familiar skin/sky/grass colour regions, saturation extremes, highlights, shadows, texture, and gamut | `src/tonemap.rs`, `src/pipeline.rs`, `src/validation.rs`; `adaptive_vibrance_skin_memory_protection`, `adaptive_vibrance_preferred_memory_color_guard`, and `preferred_skin_rendering` report/compact-summary evidence; `tests/test_tonemap.rs` plus perceptual-region, one-way path, contradiction, luminance-invariance, variation-preservation, trust-gating, and deterministic parallel-reduction tests | Covered for a conservative measured skin vibrance gate, a bounded one-way excess-chroma shoulder outside the published aggregate skin-preference ellipse, and published sky/spring-grass/autumn-grass preference ellipses. The sky/grass guard preserves boosts toward a supported centre and only limits overshoot/away motion; the skin shoulder acts only above the published preferred chroma, preserves L*, never raises chroma or crosses the core, and is capped at 3 DeltaEab. All are display-only aggregate proxies, cannot claim semantic detection or calibration, and never change the technical master; preferred strength and inclusive real-scene accuracy still need approved photographic evidence |
| Explicit scanner, roll, film-stock, and per-image decision model with precedence/confidence/rejection reasons | `src/calibration_target.rs`, `src/color_calibration.rs`, `src/bin/scanstitch-calibrate.rs`, `docs/calibration-target-sampling.schema.json`, `docs/color-calibration.md`, `docs/color-calibration-library-record.schema.json`; `tests/test_calibration_target.rs`, `tests/test_color_calibration.rs`, `tests/test_pipeline.rs`; fixture-suite expectations for `calibration_confidence_min`, `calibration_matrix_condition_number_max`, `calibration_rejection_details_required`, and `selection_rejections_required` | Covered for projective chart sampling, hash-bound human corner review, payload separation, robust patch-quality rejection, loader/ingest, disjoint matrix/root-polynomial/residual-LUT fit and held-out qualification, deterministic retained-evidence refit, profile selection, roll transform matching, film-stock ranking, rejection diagnostics, and suite-level calibration-safety gates; real independently measured target captures and calibrated fixtures remain missing |
| Calibration-aware paths can outperform image-derived mapping when valid and are rejected when unsafe/weak | `src/colorspace.rs`, `src/color_calibration.rs`; `tests/test_colorspace.rs` cases for calibrated wins, weak quality rejection, neutral/reference regression rejection, unsafe calibrated-mode failure | Covered in synthetic decision cases; real calibrated fixture coverage is still pending |
| Robust image-derived fallback when anchors are sparse, biased, clipped, dusty, border-contaminated, or scene-dependent | `src/colorspace.rs`; `tests/test_colorspace.rs`; synthetic suite cases for dirty/clipped/border rejection, sparse anchors, biased scene anchors, high-key headroom | Covered synthetically; needs more real-scene evidence |
| Candidate selection uses measurable quality signals | `candidate_quality_scores`, `quality_components`, `technical_safety_score`, `color_fidelity_score` in `src/colorspace.rs`; `docs/report-schema.md`; synthetic-suite selected model evidence; fixture-suite expectations for `selected_quality_score_max`, `technical_safety_score_max`, `color_fidelity_score_max`, `memory_color_penalty_max`, `spatial_consistency_penalty_max`, `selected_runner_up_quality_delta_min`, `candidate_acceptance_signatures_required`, and `selected_candidate_rank`; `tests/test_colorspace.rs`, `tests/test_validation.rs` | Covered for neutral accuracy, reference-patch XYZ/DeltaE76/CIEDE2000 error reporting and decision gating, hue-family residuals, density monotonicity, saturation preservation, memory-colour plausibility, spatial cast consistency, gamut preservation, clipping risk, tone-colour stability, and candidate-rank/signature stability |
| Reports and debug artifacts make decisions inspectable | `src/report.rs`, `src/validation.rs`, `docs/report-schema.md`, `README.md` output artifact list; `phase46_color_candidate_comparison.tiff`, `phase46_gamut_clipping_map.tiff`, `phase46_scene_referred_prophoto_float.tiff`; coverage requirements for `min_debug_artifact_expectation_fixtures` and `required_debug_artifact_kinds`; fixture-suite expectations for `render_input_reason_contains`, `selected_mapping_reason_contains`, `highlight_chroma_compressed_ratio_max`, `highlight_neutral_chroma_compressed_ratio_max`, `shadow_chroma_compressed_ratio_max`, `debug_artifacts_required`, and `debug_artifact_kinds_required`; `tests/test_validation.rs` | Covered for report fields, explicit evaluated/not-applicable evidence provenance, debug artifact references, compact summaries, baseline comparisons, rationale text checks, tone-protection ratio gates, named debug-artifact freshness gates, high-latitude artifact export, and corpus-level debug-artifact expectation coverage |
| Validation harness proves synthetic edge cases and real fixtures including LOGAN without hidden regressions | `src/bin/scanstitch-validate.rs`, `src/validation.rs`, `src/render_review.rs`, `docs/validation.md`, `docs/validation-fixtures.schema.json`, `docs/render-review.schema.json`, `docs/validation-summary-baseline.schema.json`, `tests/fixtures/baselines/logan_summary_baseline.json`; `tests/test_pipeline.rs`, `tests/test_validation.rs`; fixture coverage hash gates for `component*_sha256`, `summary_baseline_sha256`, `calibration_profile_sha256`, `calibration_library_sha256`, and `render_review_sha256`; complete validation-ready geometry-preparation, negative-reconstruction, stitch-normalization, dynamic-range, grain, and human visual-review contracts with exact repairs; fixture-suite gates for stitch/base/output-space/render-input/candidate/calibration/gamut/tone/debug expectations, including measured nonlinear negative response, all-component deskew/four-edge crop, reference-patch error/regression, and debug-artifact freshness | Covered for synthetic suite and LOGAN baseline; not complete for a multi-scene/multi-film-stock corpus or any approved real final-render review |
| Avoid silent automatic changes that cannot be justified by diagnostics | Calibration profile/library rejection details, candidate acceptance diagnostics, `candidate_risk`, `tone_color_trust_state`, strict baseline and registry schemas | Covered for implemented paths; future new paths should add report fields and validation deltas before default enablement |

Reference-target fixture gates remain explicit:
`reference_patch_count_min`, `reference_patch_hue_family_regression_count_max`,
`reference_patch_selected_regresses_image_derived`,
`reference_patch_delta_e2000_delta_vs_image_derived_max`,
`reference_patch_max_delta_vs_image_derived_max`,
`reference_patch_delta_e_max_delta_vs_image_derived_max`,
`reference_patch_delta_e2000_max_delta_vs_image_derived_max`,
`reference_patch_rms_delta_e_max`, and `reference_patch_rms_delta_e2000_max`. Structured fixture
repairs retain action/path/detail data so a compact checklist does not erase machine-readable corpus
work.

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

On 2026-05-27, the synthetic colour gate
`cargo run --bin scanstitch-validate -- --synthetic-color-suite --strict --quiet --summary-json target\tmp\synthetic-color-suite.json --summary-md target\tmp\synthetic-color-suite.md`
passed with `status=passed`, 20 passed cases, and `issues=[]`. The cases cover calibrated profile
selection, weak/unsafe calibration rejection, scanner-prior selection/rejection, sparse and biased
anchor review, dirty-edge sample rejection, high-key headroom preservation, destructive-gamut render
fallback, tone colour-protection policy gates, reference-patch regression rejection, and forced
unsafe-calibration failure.

On 2026-07-19, the strict debug colour gate passed with `status=passed`, 24 of 24 cases
passed, and `issues=[]`. Four nonlinear cases prove that held-out-qualified root-polynomial and
smooth residual-LUT models can be selected inside measured support and are rejected in favour of the
matrix fallback outside that support. Separate regressions prove exact per-pixel LUT evaluation,
scanner-target/roll-target LUT identity propagation, deterministic loader refit, and modified-node
tamper rejection. On 2026-07-21, `cargo fmt --all -- --check`, `git diff --check`,
`cargo check -j1 --all-targets`, `cargo clippy -j1 --all-targets -- -D warnings`, and
`cargo test -j1 --all-targets` all passed; the exact all-target inventory
contained 503 tests. The final non-incremental executable build completed in 716.4 seconds and the
full run completed in 438.9 seconds, with 501 passed, zero failed, and two intentionally ignored real-corpus tests
(OLD_TESTROLL and the
full-resolution LOGAN stitch). All 16 tracked repository JSON files
also parsed. The strict fixture-suite path passed grain enabled/review/support,
luminance/opponent-colour probe-count and p10-retention contracts, plus positive minima for applied
pixels, exact structure exclusion, and flat-area luma/chroma p95 residual reduction. A deterministic
positive-input pipeline fixture proves preservation, nonzero reduction, and a non-universal mask,
then fails strict validation when one effect floor is raised above the measured result. The same real
pipeline path now proves a complete
scene-specific dynamic-range contract—reviewable delivery, supported/no-review tone evidence,
positive evidence confidence and render-to-mapped range retention, post-scale preservation, robust
p05-p95 luminance span, and post-tone high/low clipping ceilings—and rejects absolute and relative
span floors raised above the measured render.
Strict fixture execution now also treats an undeclared non-reviewable delivery as an issue instead
of accepting it merely because it is stable. `expectations.render_reviewable=false` explicitly
marks a fixture whose purpose is diagnostic, while validator `--require-reviewable` rejects even
that declared case for production acceptance after retaining the render, report, and compact suite
summary. Delivery acceptance additionally reopens the primary TIFF for dimensions, RGB16 storage,
and ICC agreement, then reopens every promised RGB32-float ProPhoto master and RGB8 sRGB proof for
dimensions, storage, and profile agreement. Explicit zero stale artifacts are also required, so a
reviewable report cannot mask a missing or mismatched file.
The perfect-mode sRGB proof no longer uses independent target-channel clipping after the ProPhoto
matrix conversion. Out-of-target colours are binary-searched toward the neutral axis in D50
CIELAB, preserving lightness and hue, then adapted to D65 and encoded under an explicit standard
sRGB ICC profile. Save and compact-validation evidence records mapped ratios/scales and rejects
non-finite inputs or any remaining post-map gamut excursion.
Final visual review now has a separate evidence contract. `--write-render-review` emits a draft with
all decisions false and binds ordered source-file plus decoded-pixel hashes, report bytes, primary
TIFF, float master, and sRGB proof. An approved fixture is counted only when its registry inputs,
technical delivery gate, every applicable noted human decision, and pinned manifest SHA-256 agree.
For grain-on results, schema version 2 additionally requires a technically reviewable grain-off
report with identical decoded inputs, normalized non-grain render arguments, and byte-identical
scene-referred master, then binds all three control artifacts. Report schema version 3 introduced
exact executable identity; the pair must also carry the same valid executable digest and size.
Current report schema version 4 additionally hashes the exact reopened primary, master, and proof
bytes and makes those matches part of technical reviewability. The regression accepts a completed
exact pair and rejects a missing/stale control, modified
report, render-contract drift, absent legacy fingerprint, different binary, substituted input order,
or changed/missing output. No real fixture has been human-approved through this path yet.
The validator now also enforces complete stitch-normalization coverage instead of counting an
accepted decision alone. A deterministic shuffled three-scan positive fixture proves two accepted
merges, inferred order, conservative aggregate exposure normalization, a <=5% normalized offset,
seam-aware multiband application, no seam/detail review, at least two supported detail scales,
<=2.0 symmetric detail-energy imbalance, <=1.05 seam-gradient ratio, and a sub-1 overlap-p95
ceiling; strict validation then rejects a supported-scale floor above the measured result. Compact
N-input summaries now aggregate every per-merge exposure record, retain common models or report
`sequence_mixed`, use worst-case normalized offsets, and require model-appropriate held-out success
for every non-identity merge.
Positive reports now use the truthful
`base_estimate_source=not_applicable_positive_input`, allowing complete tracked baselines without
inventing negative-film base evidence. The skipped negative-film classifier likewise reports
`classification_evidence_evaluated=false`, `classification_status=not_applicable_positive_input`,
`confidence_basis=not_evaluated`, and confidence `0.0` for both single- and multi-input positive
runs; phase success records a correct bypass rather than fabricated perfect evidence. Synthetic
tests also pin truthful border-crop evidence: applied edges use the weakest normalized run-length,
luminance-gap, and boundary-transition signal, while `no_crop_no_convincing_dead_zone` reports
`applied_crop_evidence_supported=false` and phase confidence `0.0` rather than an unconditional
perfect score. Crop coordinates and the independently pinned four-edge accuracy contract are
unchanged. The same non-evaluation contract now covers `skipped_single_input` with
`confidence_basis=not_applicable_single_input`, positive density/ICA bypasses, and
`not_evaluated_explicit_direct_density_route`: successful control flow reports confidence
`0.0`, while automatic topology evidence and user-authoritative overrides remain distinguishable.
Declared-positive input now also propagates the existing orange-mask/channel-ratio probe into the
final gate: `review_required_negative_like_positive_input` becomes
`render_review_status=review_required_input_mode`, while `accepted_warm_positive_input` proves that
high warmth with mild ratios remains accepted.
The load phase now separates successful file I/O from declared decode fidelity. RGB16 retained in a
16-bit working domain reports `full_declared_decode_fidelity`; RGBA8 expanded into the default
14-bit domain reports confidence `8/14` and
`source_precision_upscaled_without_added_information`, and multi-input confidence is the weakest
component. This preserves legacy processing while making it explicit that numeric expansion cannot
restore missing source tonal levels.
Downstream confidence now follows the same evidence contract. `fallback_only` colour is
review-required instead of regaining trusted tone status; `review_required_selected_mapping` gives
colorspace confidence `0.0`, and tone confidence carries that upstream limit. Disabled technical
white balance reports `not_applicable_disabled` at `0.0`; manual adaptation is explicitly
`user_authoritative_override`. A saved diagnostic render can still have `success=true`, but
`diagnostic_delivery_requires_review` now has delivery confidence `0.0` rather than conflating file
I/O with approval.
Executed FastICA now distinguishes numerical convergence from physical dye-separation evidence.
`numerically_converged_physical_separation_unvalidated` retains the optimizer result while
`physical_separation_evidence_evaluated=false` and phase confidence `0.0`; downstream candidate
comparison can judge render safety and colour plausibility but cannot manufacture calibration truth.
The shared base limiter also records `confidence_before_base_limit` and sets
`confidence_limited_by_base_estimate=true` only when it actually lowers the phase value.
Density inversion now applies the same evidence discipline to the non-ICA path. Its
`minimum_film_base_and_negative_response_model_evidence` confidence is the weaker of explicit
film-base support and response-model support. `held_out_measured_response_supported` retains the
validated model confidence; `review_required_unmeasured_unit_slope_response` and the frame-derived
equivalent report `negative_response_model_evidence_evaluated=false` and `0.0` rather than treating
successful logarithmic arithmetic as physical dye-crosstalk or characteristic-curve validation.
Runtime curve extrapolation remains a later one-way review gate.
That later gate now propagates quantitatively instead of applying a generic `0.5` cap. A trusted
mapping plus `held_out_measured_response_and_runtime_coverage_supported` carries the density
phase's weakest film-base/model confidence into tone. Blind ICA, unmeasured reconstruction, and
`review_required_measured_response_outside_runtime_curve_support` report complete colour confidence
`0.0`; tone inherits it and diagnostic save remains non-reviewable. Mapping-only confidence,
negative-reconstruction confidence, and
`confidence_limited_by_negative_reconstruction_evidence` remain separate so reports preserve the
actual limiting cause.
Rendered output now has the same evidence discipline. `supported_render_tonal_distribution`
requires finite ordered diagnostics, a nontrivial fitted tonal span, retained rendered p05-p95
range, and no catastrophic per-channel clipping. Deterministic cases pin
`review_required_collapsed_render_luminance_range` and
`review_required_catastrophic_post_tone_clipping`, along with flat/invalid outcomes. The tone report
records `confidence_limited_by_tone_output_evidence`; failure contributes `0.0`, becomes
`review_required_tone_output_evidence`, and propagates to final
`render_review_status=review_required_tone_output`. A trusted positive ICC render remains
reviewable. These universal gates reject unusable output but do not replace the approved,
scene-specific dynamic-range contract needed to judge preferred contrast.
The compact validator now retains the same tone-output status, review reason, evidence confidence,
range relationship, and clipping maxima in JSON and Markdown. Newly written tracked baselines pin
them; generic report comparisons and baseline comparisons emit
`tone_output_confidence_status_changed`, `tone_output_evidence_confidence_regressed`,
`tone_output_range_retention_regressed`, and high/low clipping issues, and `--fail-on tone` selects
them. Missing
legacy fields remain non-asserting, so backward compatibility does not fabricate evidence.
Roll validation now carries that full decision into every frame and aggregates final-render status,
tone-output status, and review-reason counts. `roll_suite:<frame>:render_review_not_supported` and
`roll_suite:<frame>:tone_output_review_required` prevent a batch from passing merely because an
independent candidate-risk check stayed safe; missing current decision evidence also fails closed.
Matched roll comparisons emit `tone_output_status_regressed`,
`tone_output_review_newly_required`, `tone_output_evidence_confidence_dropped`,
`tone_output_range_retention_dropped`, and explicit high/low clipping issues. Historical roll JSON
without the new fields remains comparable, but removing evidence that existed in the baseline is a
regression.
Synthetic tests also prove that the enabled filter preserves
coherent luminance and isoluminant colour edges and that an adversarial structured-edge erasure
becomes `review_required_grain_detail`. A four-frame grain-on roll-suite test passed with supported
luminance and opponent-colour populations, no harmful-detail review, and zero retention drift on a
matched rerun; an adversarial comparator separately proves lost support, new review, and material
p10 regression are reported without penalizing legacy field absence. The registry path also proved
fixture-first grain mode/strength/scale propagation in direct and strict suite runs,
conflicting-global-CLI isolation, invalid/conflicting setting rejection, and generated roll-scaffold
capture. Fixture coverage now separately gates validation-ready geometry preparation, factual
geometry accuracy, exact orientation handling, measured negative reconstruction, dynamic range,
stitch normalization, grain-on declarations, complete two-channel preservation contracts, and
measurable-effect contracts; partial declarations receive field-specific repair plans. Deterministic strict fixtures
prove a -0.5-degree correction, all four edges of a known 16-pixel board, and an orientation-6
transform bound to an approved decoded-pixel digest and dimension swap, plus a measured nonlinear
dye-response path. An out-of-tolerance crop edge, a coherent-but-wrong orientation-8/wrong-digest
annotation, and an over-tight response confidence fail exact
runtime issues without conflating structural coverage with execution. Validation registries now
represent a single scan with only `component1`; `component2` is optional, roll scaffolds no longer
duplicate a path, and direct overrides accept one input. A strict single-component fixture proves
that singleton availability/hash/TIFF/layout/dimension sets are validation-ready while every pair
counter remains zero; existing pair and shuffled three-scan suites still pass. Orphaned
`component2_sha256`, `additional_components` without a real second input, and forced stitching of
a singleton are rejected. Target sampling now writes overlays and a
`corner-review-manifest.json` before it can write fit-ready measurements. Every capture must move
from `requires_corner_approval` to approved with matching `decoded_pixel_sha256` and
`chart_corners_sha256` plus the exact `overlay_pixel_sha256`; pixel, geometry, and grid-layout
tamper regressions prove all three bindings fail closed. The same manifest requires explicit
reference-data approval matching `reference_patch_sha256`; missing, edited-value, JSON-round-trip,
and retained-record tamper regressions prevent placeholder XYZ/Lab values from silently becoming
calibration truth. The
full all-target inventory is 503 tests, including 88 validator
tests. The full-resolution LOGAN stitch was
run separately and passed its pinned geometry plus multi-scale seam-detail evidence assertions. This
is synthetic decision-path and implementation evidence only; it is not independently measured
scanner/film colour or grain/detail truth and does not close the real-corpus requirement.
The standalone orientation-review mode generated current local LOGAN and BIRDING047 preview/digest
packages under `output/audit/orientation-review-*-20260718`. Their manifests remain
`requires_human_approval` with every `upright_approved` value false; generation and an automated
visual sanity check were not promoted into corpus acceptance.

An earlier release pipeline regenerated BIRDING047 under
`output/audit/positive-birding047-skin-protection-candidate-20260719/`. Its independent delivery gate
passed and the pending render-review draft binds the exact source bytes, decoded pixels, report,
primary TIFF, float master, and sRGB proof. The measured D50 CIELAB skin-memory gate evaluated
58.112% of pixels and attenuated display vibrance for 23.491%, with mean protection weight 0.614.
The technical scene master remained byte-identical to the preceding candidate; an every-eighth-pixel
sRGB-proof comparison measured mean/p95/maximum absolute RGB-code changes of 0.17/1.33/14 and no
saturation increases. Automated inspection found no near-black edge strip but still noted a
pronounced warm/magenta appearance and visible grain. The draft remains `requires_human_approval`
with zero of five applicable decisions approved, so it is actionable review material rather than
colour, skin-rendition, or aesthetic truth.

On 2026-07-20 the current release rerendered the same source under
`output/audit/positive-birding047-preferred-memory-guard-20260720/`. The published sky/spring-grass/
autumn-grass preference-ellipse guard evaluated 58.112% of pixels, matched 35.134%, and limited
7.155%; overall mean/max removed boost was 0.00167/0.08048. Spring-grass neighbourhood matches
dominated (29.093% of all pixels) but usually removed only 0.000149 mean scale, demonstrating why
the gate is reported as a non-semantic colour proxy. Strict `--require-reviewable` validation passed
with coherent family sums, valid RGB16/RGB32-float/RGB8+ICC artifacts, supported sRGB gamut mapping,
and zero stale files. The technical master retained SHA-256
`b72fb7e908aecf147c628adce8711accae21a91598c17aae9756f34e4af30625`; only 1.694% of RGB16 primary
pixels and 0.390% of RGB8 proof pixels changed after quantization, with maximum absolute ProPhoto
luminance drift 0.0000151. Visual inspection still found a strong warm/magenta rendering and visible
grain, and the scene has no clear sky/foliage truth. The old review draft cannot approve the changed
output, so this is bounded mechanics/performance evidence rather than preference validation.

On 2026-07-21 the accepted follow-up rerendered BIRDING047 under
`output/audit/positive-birding047-preferred-skin-parallel-20260721/`. Modern-clean rendering now
leaves pixels inside the published aggregate skin 50%-acceptability ellipse unchanged and considers
an outside pixel only when the broader measured L*C*h support agrees and source chroma exceeds the
published preferred-centre chroma. It removes 35% of ellipse-radius excess without crossing the
core, preserves CIELAB L*, refuses any chroma increase, caps movement at 3 DeltaEab, and backs off
at ProPhoto gamut boundaries. The fused real pass evaluated 79.126% of all pixels, matched 21.708%,
found 16.472% outside the core, and adjusted 3.137%; mean/max DeltaEab were 0.903/3.000,
mean/max absolute hue movement were 1.112/6.595 degrees, mean chroma delta was -0.604, maximum
absolute chroma change was 2.127, and no adjusted pixel required gamut backoff. Strict
`--require-reviewable` validation passed with coherent population/effect accounting and intact
RGB16/RGB32-float/RGB8 artifacts. The primary SHA-256 is
`fb567932c08923007e189642b4edcb8fc3b64f80441308cb206e52bd671a03f8`; the technical master remains
byte-identical at `b72fb7e908aecf147c628adce8711accae21a91598c17aae9756f34e4af30625`.
Against the preceding preferred-memory render, 3.135% of RGB16 pixels and 2.463% of RGB8 proof
pixels changed; changed-channel mean/p95 magnitudes were 0.00225/0.00816 in linear ProPhoto and
1.08/3 codes in sRGB. A deterministic 128-row parallel implementation produced a byte-identical
primary TIFF to the serial fused reference and pixel-identical sRGB proof while reducing tone time
from the rejected 21.255-second serial separate pass and 15.097-second parallel separate pass to
9.433 seconds (the pre-feature render was 7.080 seconds). Visual inspection suggests only a modest
reduction in excessive warm colour; the proof remains warm/magenta and has no approved reference
print or diverse-skin preference set. A new schema-valid render-review draft binds the exact source,
decoded pixels, report, primary, technical master, and proof hashes; it remains
`requires_human_approval` with all five applicable decisions false. This therefore proves bounded
mechanics, provenance, determinism, and real execution—not semantic skin identification or optimal
beauty.

The superseding matched optional-grain follow-up is
`output/audit/positive-birding047-grain-v3-on-050-retry-20260721/`, paired with
`output/audit/positive-birding047-grain-v3-control-off-retry-20260721/` and rendered by the same
9,539,584-byte release executable at strength 0.5 and scale 1.0. Both report-schema-v3 records and
an external post-render check identify it as SHA-256
`0612bd1c2c22b9a4cad36f5d05b5601ce19114785c78ebd306e4ff1c8756de96`. An initial audit candidate
exposed a 99.499% active mask and
frame-level damping that scaled the full chroma residual; that was rejected as near-universal
smoothing/global desaturation despite passing average detail retention. The corrected pass damps
only local high-frequency chroma around its blurred base, applies a real shadow-relaxed saturation
mask, feathers multiscale structure from 0.018 to the independent 0.035 detail floor, and bypasses
protected structure and incomplete boundary windows exactly. The final real mask selected 48.063%
and excluded 50.674%; median luma/chroma residual fell 6.368%/14.887%, while conservative flat p95
fell 0.928%/0.665%. Luminance/chroma p10 retention was 0.992/~1.000 over 38,647/14,622 probes with
no review flag. The technical master stayed byte-identical; against grain-off, 49.089% of RGB16
pixels changed but the channel-delta median was zero and signed mean drift stayed near 1e-5. Strict
validation accepts the new gate/population accounting and rejects the obsolete enabled report for
missing it. The review-schema-v2 draft at
`render-review-bound-control-v3/render-review.json` binds both exact report hashes and all six
delivery artifacts, reopens every hash successfully, makes grain/detail applicable, and leaves all
six applicable human decisions unapproved. This is real selective/measurable effect and provenance
evidence, not proof that the strength is preferred or that every suppressed residual is film grain.

A later artifact-streaming-only release rerendered the same grain-on BIRDING047 pixels under
`output/audit/positive-birding047-streamed-artifacts-grain-on-050-20260721/`. Its report records
`sequential_bounded_tiff_strips_and_png_rows`, no full-frame conversion buffer, and a 1,013,400-byte
declared peak versus the former 248,958,600-byte float-master conversion allocation. The monitored
process peaked at 1,599.6 MiB overall and completed in 55.0 seconds, with 22.021 seconds in save.
Primary/master SHA-256 remained
`7db2c9021e77d93089ff92e30e82c1e9792949396f757d52c9f5d2aea99ab432`/
`b72fb7e908aecf147c628adce8711accae21a91598c17aae9756f34e4af30625`; the streamed PNG had zero
changed decoded pixels and an independently valid standard-sRGB profile. A later byte audit found
that moxcms had placed the current hour/minute/second in the otherwise identical 612-byte profile,
so the earlier payload and PNG container were not byte-reproducible. Independent delivery
validation passed. This proves pixel identity and the bounded conversion design, not a controlled
wall-time gain.

The superseding release reuses the owned batch scene array after writing a requested technical
master, while interactive rendering retains its cached master and separate reversible buffer. Two
real reruns under `positive-birding047-owned-batch-canonical-icc-{a,b}-20260721`, from exact release
SHA-256 `c6442ce0bb5e94c85f0a585ef5d686b03f3e6fbb527d7f02826969d3063d8de2`, completed in
45.522/44.644 seconds and peaked at 1,124.5/1,124.6 MiB. That is about 475 MiB (29.7%) below the
streaming-only process peak and corresponds to one avoided full-frame f64 RGB array; both runs still
declare only 1,013,400 bytes of artifact-conversion buffering. Primary, scene master, and proof are
byte-identical between runs at SHA-256
`7db2c9021e77d93089ff92e30e82c1e9792949396f757d52c9f5d2aea99ab432`,
`b72fb7e908aecf147c628adce8711accae21a91598c17aae9756f34e4af30625`, and
`c4558f5594737c83055272f574c954875504f251b3d21713d342b2a88f33c47a`. The canonical 612-byte sRGB
payload has SHA-256 `0f303d7fb11d811c9751321ea472f4f024781dcdf0143ac7d94cebd1f0719323`; it uses the fixed profile
definition date 2026-05-08 00:00:00, remains standards-valid, and leaves the preceding render's IDAT
bytes unchanged. Both packages pass independent `--require-reviewable` validation. This is exact
delivery reproducibility and bounded-memory evidence, not human approval of colour or grain.

The current schema-v4 release, exact renderer SHA-256
`ff236561e2c5897c190d8f5c58b8b1803cea6a47808b224febaa9c2e22130f76`, rerendered the same case at
`positive-birding047-schema-v4-artifact-hash-20260721`. It completed in 49.173 seconds, peaked at
1,124.7 MiB working set/1,123.9 MiB private memory, and spent 19.246 seconds in save, including
reopening and hashing all three artifacts with a sequential 1,048,576-byte buffer. The report's
primary/master/proof digests exactly match independent file hashes and remain
`7db2c9021e77d93089ff92e30e82c1e9792949396f757d52c9f5d2aea99ab432`,
`b72fb7e908aecf147c628adce8711accae21a91598c17aae9756f34e4af30625`, and
`c4558f5594737c83055272f574c954875504f251b3d21713d342b2a88f33c47a`. Independent validation from
exact validator SHA-256 `763695c07a5b78bf2ee7e368685e5d0547dd9a674667842c903612bfe564b22c`
passes `--require-reviewable` with all three exact-byte matches true. Regressions replace each
artifact in turn with a structurally valid same-sized/profiled image and prove that structural
inspection still passes while the SHA mismatch rejects delivery. This binds an unchanged report
to unchanged artifacts; it is not a signature, and coordinated report/artifact edits remain outside
the technical gate's trust boundary.

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

The remaining blocker is not another synthetic unit test. It is a
real multi-scene, multi-film-stock CoolScan validation corpus large enough to exercise the target
operating range:

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
- fixture-registry `min_approved_render_review_fixtures`, with completed reviewer-authored manifests
  whose ordered source/decode/report/output/master/proof hashes and all applicable visual decisions
  still match
- fixture-registry `min_render_dynamic_range_contract_fixtures`, with validation-ready cases that
  pin `render_review_status=reviewable`, `render_reviewable=true`,
  `tone_output_confidence_status=supported_render_tonal_distribution`,
  `tone_output_review_required=false`, positive `tone_output_evidence_confidence_min` and
  `render_to_mapped_luminance_range_ratio_min` floors, scene-specific post-scale preservation and
  p05-p95 luminance-span floors, plus high/low post-tone clipping ceilings
- fixture-registry `min_geometry_preparation_contract_fixtures`, with real rotated scans that pin
  all-component application, four-edge removal, retained-area bounds, and no geometry/crop review
- fixture-registry `min_geometry_accuracy_contract_fixtures`, with approved signed deskew angles and
  per-component top/bottom/left/right crop annotations under <=0.05-degree and <=4-pixel tolerances
- fixture-registry `min_orientation_accuracy_contract_fixtures`, with every source tag/transform,
  applied flag, source/output dimensions, explicit upright approval, and decoded-pixel digest
checked against the result a human approved as upright
- fixture-registry `min_negative_reconstruction_contract_fixtures`, with a measured rebate/base and
  independently held-out roll target that pins nonlinear dye reconstruction, CIEDE2000 improvement,
  noise/extrapolation/headroom limits, and a trusted direct-density colour result
- fixture-registry `min_grain_reduction_enabled_fixtures` and
  `min_grain_detail_contract_fixtures` and
  `min_grain_reduction_effect_contract_fixtures` requirements, with at least one validation-ready
  real grain-on case that pins nonzero strength/scale, both-channel 64-probe / 0.70-p10 preservation
  evidence, and positive applied-pixel, exact structure-excluded, and flat-area luma/chroma
  residual-reduction floors
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
  `add_reference_patches_for_fixture`, `complete_render_review_for_fixture`, plus TIFF-specific repair actions such as
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

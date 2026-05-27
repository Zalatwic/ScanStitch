# Real-Image Validation

Use `scanstitch-validate` to rerun a known local pair or summarize an existing `report.json` into compact JSON and Markdown. When it reruns a pair, rendered TIFFs stay under ignored local output directories.

## LOGAN Fixture

The first canonical local fixture is the LOGAN pair. Keep the source scans outside version control:

```powershell
cargo run --bin scanstitch-validate -- --fixture logan
```

By default this expects `LOGAN043.tif` and `LOGAN044.tif` in the repository root and writes:

```text
output/validation/logan/report.json
output/validation/logan/summary.json
output/validation/logan/summary.md
```

List known fixtures:

```powershell
cargo run --bin scanstitch-validate -- --list-fixtures
```

Add local fixtures with a registry file shaped like
[`validation-fixtures.example.json`](validation-fixtures.example.json). The companion
[`validation-fixtures.schema.json`](validation-fixtures.schema.json) gives the machine-readable
draft 2020-12 schema for registry review tools.
`scanstitch-validate` also rejects unknown registry fields and calibration wiring that violates the
schema contract before listing, auditing, or running fixtures. Per-frame roll metadata sidecars use
[`validation-roll-fixture-metadata.schema.json`](validation-roll-fixture-metadata.schema.json),
which reuses the registry coverage and expectation definitions.

```powershell
cargo run --bin scanstitch-validate -- `
  --fixture-registry docs/validation-fixtures.example.json `
  --list-fixtures
```

## Local Corpus Scaffold

The example registry is a scaffold for a private validation corpus, not a passing corpus by itself.
The local source scans, calibration records, and per-fixture baselines live under ignored paths, so
they can contain large TIFFs or private roll data without entering version control:

```text
LOGAN043.tif
LOGAN044.tif
calibration/scanners/coolscan-4000-vuescan-raw.json
calibration/rolls/logan-roll-2026-05.json
calibration/film_hints/kodak-gold-200.json
local-fixtures/my-split-left.tif
local-fixtures/my-split-right.tif
local-fixtures/baselines/my-split-frame-summary-baseline.json
local-fixtures/calibration/my-split-frame-profile.json
local-fixtures/uncalibrated-night-left.tif
local-fixtures/uncalibrated-night-right.tif
local-fixtures/baselines/uncalibrated-night-summary-baseline.json
```

The current local coverage run maps directly to those paths:

- `repair_component1_tiff_for_fixture:logan` and
  `repair_component2_tiff_for_fixture:logan`: the presently available LOGAN pair is
  `RGBA8 5959x3670` and `RGBA8 5959x3669`, so it fails the example registry's
  `min_tiff_bits_per_sample: 14` requirement. Its one-row height delta is accepted by the built-in
  LOGAN gate as stitch-compatible regression evidence, but it is not high-bit CoolScan RAW input.
  For strict CoolScan
  RAW evidence, replace both components with matching high-bit scans from the same scanner/settings
  workflow. Coverage also emits the more specific
  `replace_component1_with_minimum_bit_depth_tiff_for_fixture:logan`,
  `replace_component2_with_minimum_bit_depth_tiff_for_fixture:logan`, and only emits
  `replace_component_pair_with_dimension_matched_tiffs_for_fixture:logan` when a pair is neither
  exact-dimension nor stitch-compatible. Only relax the minimum bit-depth or dimension requirements
  with an explicit rationale.
- `provide_calibration_library_for_fixture:logan`: create the `calibration/` library with scanner
  profile `coolscan-4000-vuescan-raw`, roll profile `logan-roll-2026-05`, and a Kodak Gold 200 film
  hint. This supplies the required `Kodak Gold 200|scanner-roll-library` pair and the
  `skin-tone|normal-exposure` scene/exposure pair once the components are validation-ready.
- `provide_component1_for_fixture:my-split-frame`,
  `provide_component2_for_fixture:my-split-frame`,
  `provide_summary_baseline_for_fixture:my-split-frame`,
  `provide_calibration_profile_for_fixture:my-split-frame`, and
  `add_reference_patches_for_fixture:my-split-frame`: add a matching high-bit Fujicolor 200 pair,
  an external profile with ColorChecker or equivalent D50 Lab/XYZ patches, and a compact baseline.
  This fixture is intended to cover `Fujicolor 200|external-profile`,
  `foliage|overexposed-negative`, `high-saturation|overexposed-negative`, and
  `high-key|overexposed-negative`.
- `provide_component1_for_fixture:uncalibrated-night`,
  `provide_component2_for_fixture:uncalibrated-night`, and
  `provide_summary_baseline_for_fixture:uncalibrated-night`: add a matching high-bit Kodak
  Ultramax 400 pair with no calibration profile or library. This fixture is intended to cover
  `Kodak Ultramax 400|uncalibrated-image-derived`, `deep-shadow|underexposed-negative`, and
  `low-neutral|underexposed-negative`.

After adding source files, run each fixture once and freeze its compact baseline. For example:

```powershell
cargo run --release --bin scanstitch-validate -- `
  --fixture-registry docs/validation-fixtures.example.json `
  --fixture my-split-frame `
  --debug

cargo run --release --bin scanstitch-validate -- `
  --fixture-registry docs/validation-fixtures.example.json `
  --fixture my-split-frame `
  --report output/validation/my-split-frame/report.json `
  --write-summary-baseline local-fixtures/baselines/my-split-frame-summary-baseline.json `
  --summary-json output/validation/my-split-frame/summary.json `
  --summary-md output/validation/my-split-frame/summary.md
```

Then compute the local SHA-256 snapshot and copy only the intended component, compact-baseline, and
calibration hash fields back into the registry you use for strict validation:

```powershell
cargo run --bin scanstitch-validate -- `
  --fixture-registry docs/validation-fixtures.example.json `
  --fixture-coverage `
  --compute-fixture-hashes `
  --write-fixture-hash-registry output/validation/validation-fixtures.with-hashes.json `
  --summary-json output/validation/fixture-coverage-hashes.json `
  --summary-md output/validation/fixture-coverage-hashes.md
```

The scaffold is ready only when `--fixture-coverage --strict` and `--fixture-suite --strict --debug`
both pass with no remaining `action_items`.

## VueScan DNG Scanner Rolls

VueScan DNG files from scanner RAW rolls are accepted when they contain an uncompressed
three-channel RGB or DNG LinearRaw full-resolution image directory. The loader selects the largest
supported image directory or SubIFD, so scanner DNGs with a small embedded RGB thumbnail are not
accidentally validated at thumbnail resolution. The load phase records the selected source layout,
for example `DNG_LINEAR_RAW16`, plus the full source dimensions, working bit depth, and primary-IFD
DNG metadata when present: scanner/software strings, `ColorMatrix1`/`ColorMatrix2`,
`AsShotNeutral`, black/white levels, calibration illuminants, matrix condition numbers, and
non-fatal metadata parse warnings.

When no explicit calibration profile or calibration library is configured and `--color-mode auto`
is used, a valid DNG `ColorMatrix1` is converted into a weak scanner-constrained advisory prior.
That prior does not count as roll calibration evidence and does not override user-supplied
calibration. It simply adds a `scanner_prior_image_adaptation` colorspace candidate; automatic
candidate scoring still accepts it only if it beats image-derived mapping without failing gamut,
neutral, or reference-fit checks.

The local `TESTROLL/` folder is ignored by version control and can be used as a private roll corpus.
When checking an already-positive roll, `--roll-inventory --input-mode positive` performs the same
orange-mask-like channel-bias and severe-ratio aggregate-score preflight and reports
`roll_inventory:<frame>:positive_input_negative_like` before any full-resolution render is needed.
Inventory Markdown includes a `Positive Input` column with the probe score when this preflight runs.
Accepted high-score warm positives are labeled `ok warm score` so they can be distinguished from
lower-score neutral positives and from `negative-like score` warnings. The probe JSON also exposes
`accepted_high_warm_score=true` for that accepted high-score state.
The current validation wrapper still expects two component paths; for a single full-frame DNG, pass
the same DNG as both components and force no-stitch mode:

```powershell
cargo run --release --bin scanstitch-validate -- `
  --fixture testroll-raw-0000 `
  --component1 TESTROLL/RAW_0000.dng `
  --component2 TESTROLL/RAW_0000.dng `
  --force-no-stitch `
  --bit-depth 16 `
  --output-dir output/validation/testroll/raw-0000 `
  --summary-json output/validation/testroll/raw-0000-summary.json `
  --summary-md output/validation/testroll/raw-0000-summary.md
```

To scaffold a private fixture registry from every usable frame in a roll inventory, write a roll
fixture registry:

```powershell
cargo run --release --bin scanstitch-validate -- `
  --roll-dir TESTROLL `
  --roll-inventory `
  --bit-depth 16 `
  --film-stock "Kodak Gold 200" `
  --roll-fixture-scene-tag outdoor,skin `
  --roll-fixture-exposure-tag normal-exposure `
  --write-roll-contact-sheet output/validation/testroll-contact-sheet.png `
  --write-roll-contact-sheet-index output/validation/testroll-contact-sheet.json `
  --write-roll-fixture-metadata-template local-fixtures/testroll-metadata-template.json `
  --roll-fixture-metadata local-fixtures/testroll-metadata.json `
  --write-roll-fixture-registry local-fixtures/testroll-fixtures.json `
  --summary-json output/validation/testroll-inventory.json `
  --summary-md output/validation/testroll-inventory.md
```

The writer emits same-path `component1`/`component2` entries with `force_no_stitch: true`, local
baseline paths under `local-fixtures/baselines/`, per-frame output directories, the selected
`input_mode`/`bit_depth`, and any supplied calibration profile/library, scanner profile, roll
profile, film-stock default, `--roll-fixture-scene-tag`, or `--roll-fixture-exposure-tag` values.
Those scene and exposure flags are stamped onto every generated entry, so use them only for a
curated subset whose labels are genuinely shared. The writer does not invent scene, exposure, or
reference-evidence labels; add or pass those curated fields before treating the registry as corpus
proof. Use repeated or comma-separated `--roll-suite-frame <name|stem|slug>` selectors with the
writer when only a representative subset should become fixture entries. After accepting generated
baselines with `--write-fixture-suite-baselines`, run fixture coverage with
`--compute-fixture-hashes --write-fixture-hash-registry` to pin the component and baseline hashes
for strict reruns.

For per-frame curation, keep a private ignored JSON sidecar and pass it with
`--roll-fixture-metadata`. Start one from the current inventory with
`--write-roll-fixture-metadata-template`; repeated or comma-separated `--roll-suite-frame`
selectors limit the template to a curated subset. Template frame keys use the frame stem by
default, and sidecar frame keys may be the frame filename, stem, slug, or generated fixture name.
For visual scene and exposure curation, add `--write-roll-contact-sheet` and optionally
`--write-roll-contact-sheet-index` to the inventory run. The PNG contact sheet uses a fast
per-channel stretch preview, inverting negative inputs only for visual triage; it is not color
proof or calibration evidence. The JSON index maps each tile back to its frame key, dimensions, and
preview transform so curation notes can be copied into the sidecar without guessing from filenames.
To refresh an existing partially curated sidecar, pass it with `--roll-fixture-metadata` alongside
`--write-roll-fixture-metadata-template`; matching entries are preserved, newly discovered selected
frames get `TODO` entries, and unmatched sidecar keys are rejected. Sidecar metadata overrides only
the fields it declares; unspecified entries keep the writer defaults. Unknown sidecar frame keys are
rejected so a typo cannot silently drop required scene, exposure, calibration, or reference evidence.
The generated template is not corpus proof by itself; replace the `TODO` descriptions and add only
factual film-stock, scene, exposure, calibration, and reference fields.

```json
{
  "coverage_requirements": {
    "min_fixtures": 2,
    "required_scene_tags": ["skin-tone", "foliage"],
    "required_exposure_tags": ["normal-exposure", "overexposed-negative"],
    "required_scene_exposure_pairs": [
      "skin-tone|normal-exposure",
      "foliage|overexposed-negative"
    ],
    "required_debug_artifact_kinds": ["candidate_comparison", "gamut_clipping_map"]
  },
  "frames": {
    "RAW_0000": {
      "film_stock": "Kodak Gold 200",
      "scene_tags": ["skin-tone"],
      "exposure_tags": ["normal-exposure"],
      "reference_evidence": ["gray-card"],
      "calibration_case": "uncalibrated-image-derived",
      "expectations": {
        "debug_artifacts_required": true,
        "debug_artifact_kinds_required": ["candidate_comparison"]
      },
      "description": "Curated from local roll notes"
    }
  }
}
```

Use `--debug` on a representative frame to verify candidate-comparison and gamut-clipping debug
artifacts. Unbordered full-frame DNGs may report `base_estimate_source=high_transmittance_fallback`;
that is a low-confidence finite base estimate used to avoid a zero-base density inversion, not proof
of calibrated roll-base evidence. A strict corpus should still add a scanner/roll calibration record
or reference target evidence when available.

`--roll-suite` first derives a same-roll base candidate set from all readable frames. The prepass
uses bounded-resolution samples for base detection, then renders each frame at full resolution in an
isolated worker process so large rolls do not accumulate allocator/commit pressure across frames.
Roll-suite `--render-input auto` defaults the worker render path to direct-density, avoiding the
per-frame ICA cost for no-stitch validation while still allowing `--render-input ica` when that path
is explicitly under test. Direct-density workers normalize Phase 3 positive density by
`shared_robust_d_max` before converting to transmittance; `density_inversion` and
`colorspace_mapping` report `direct_density_render_density_scale` so roll-to-roll density placement
can be audited. It clusters candidate colors by chroma/luminance consistency, rejects
candidates that are much darker than the roll high-transmittance envelope, and applies the selected
roll base as `base_estimate_source=roll_consensus_base` only when enough frames agree. If no stable
cluster exists, frames keep their local low-confidence base status and remain review-gated.

Negative-film tone fitting treats low-linear-range direct-density frames as high-key/dense review
cases instead of pinning them to the generic dark log-median placement. The fitted curve lifts the
perceptual median for frames whose linear p95 and p95-p05 span are both low, while retaining the
extended-highlight shoulder and post-tone clipping checks. Gamut candidate selection also considers
the stricter `gamut_trusted_image_matrix_blend` when the image-derived matrix is inside broad
negative-gamut safety limits but still outside display-trust low-gamut limits.

For slide scans or already-inverted negatives, pass `--input-mode positive`. Positive roll-suite
runs do not derive a roll consensus base, do not pass `--base-color` to worker renders, and do not
emit negative-film base-confidence or base-fallback precondition issues. Worker reports keep
`density_inversion` and `fastica` phases for compatibility with `skipped=true`,
`input_mode=positive`, and `colorspace_mapping.render_input_source=positive_scan_rgb`.
Uncalibrated positive auto-mode workers report `mapping_strategy=positive_rgb_passthrough`, preserve
the already-positive RGB channel ratios in the linear ProPhoto working buffer, and use neutral or
dominant-anchor samples as diagnostics rather than as negative-film colour reconstruction gates.
Positive workers also record `working_image_select.positive_input_inspection` in the full report and
promote it into compact-summary fields `render.positive_input_likely_negative_like`,
`render.positive_input_accepted_high_warm_score`, `render.positive_input_orange_mask_score`, and
`render.positive_input_reason`. When a positive roll-suite frame has orange-mask-like channel bias
or a high aggregate orange-mask score with severe R/G and G/B ratios, the frame is marked `review_required` with
`roll_suite:<frame>:positive_input_negative_like`; this is a mode-suitability warning, not an
automatic conversion to negative mode.
When `--roll-suite --debug` is used, each frame retains the final output plus the report-linked
candidate-comparison, gamut-clipping, and scene-referred ProPhoto float artifacts; extra intermediate
TIFFs are pruned after the frame summary is written to keep full-roll debug runs bounded.
The roll-suite JSON also includes a `quality` aggregate and per-frame frame fields for
`high_frequency_luma_residual_p95`, `high_frequency_chroma_residual_p95`,
`high_frequency_chroma_to_luma_p95_ratio`, flat-area high-frequency residual metrics,
`noise_reduction_*`,
`colorspace_post_scale_preserved_ratio`, positive-input suitability, tone chroma-compression ratios,
rendered luminance p05-p95 range, midtone luminance p50 mean/min/max/range, shadow/full-midtone/midtone-neutral/bright-neutral RGB balance deltas, and post-tone clipping maxima. It also includes a `review`
aggregate with candidate-risk, tone-colour-trust, calibration-status, reference-patch, and normalized
issue-kind counts so roll-wide blockers can be inspected without expanding every frame. The Markdown report
mirrors these in roll and per-frame quality tables so full-roll runs can be compared for
all-sample and flat-area chroma/luma residuals, denoise strength/limiting, mode-suitability issues, colour-balance drift, review-gate causes, and
midtone consistency, weak or collapsed full-frame contrast, and highlight/shadow headroom without opening every frame summary.
Use the flat-area residuals when judging whether smooth regions still have distracting grain; the
all-sample residuals intentionally include real scene edges and texture.
Use `--compare-roll-suite <previous-roll-suite.json>` on a rerun to attach aggregate and per-frame
before/after deltas to the current roll-suite JSON and Markdown. For older baselines that predate
the midtone range aggregate, it derives max/range from per-frame `midtone_luminance_p50` values.
The comparator reports material regressions in failed-frame count, darkened midtone placement, widened midtone range, increased chroma or flat-area grain
residuals, dropped rendered luminance range, weaker neutral balance, reduced preserved gamut, and new post-tone highlight clipping,
while still showing small improvements and unchanged fields explicitly:

```powershell
cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite `
  --bit-depth 16 `
  --render-input direct-density `
  --input-mode negative `
  --quality-mode balanced `
  --compare-roll-suite output/testroll_previous/roll-suite.json `
  --output-dir output/testroll_current `
  --summary-json output/testroll_current/roll-suite.json `
  --summary-md output/testroll_current/roll-suite.md
```

When disk space or iteration time makes a full-roll render impractical, add repeated or
comma-separated `--roll-suite-frame <name|stem|slug>` values to render a representative subset while
still deriving the negative roll base from the full readable roll inventory. This is intended for
quick before/after checks across known dark, weak-neutral, anchor-review, and safe frames; the final
acceptance run should still use the full roll when storage allows. Subset comparisons report the
intentional frame-set change and keep per-frame deltas for the overlapping frames; aggregate
regression issues are reserved for matching frame sets.

Positive TESTROLL regression gate and roll-suite commands. Run the strict inventory gate first; if
it reports `positive_input_negative_like`, fix the roll selection before using positive roll-suite
output as calibration or regression evidence:

```powershell
cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-inventory --input-mode positive --bit-depth 16 --strict --output-dir output/testroll_positive_inventory_gate
```

After the strict inventory gate passes, run the full positive roll-suite:

```powershell
cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --input-mode positive --bit-depth 16 --output-dir output/testroll_positive
```

Registry entries may include `summary_baseline`; when present, `scanstitch-validate` compares the
compact summary against that baseline automatically unless `--compare-summary` is supplied. Entries
may also declare `summary_baseline_sha256` so the regression contract itself cannot drift silently.
Entries may also carry calibration defaults: `calibration_profile`, `calibration_library`,
`scanner_profile`, `roll_profile`, and `film_stock`. These defaults are passed to the pipeline when
the fixture is run, and matching command-line flags override the registry value for that field.
Registry `film_stock` is always retained as corpus metadata; it is passed as a pipeline calibration
selector only when a calibration library is active, or when `--film-stock` is supplied explicitly.
Entries may also override pipeline shape with `input_mode`, `bit_depth`, `force_stitch`, and
`force_no_stitch`. Use `force_no_stitch: true` with the same path for `component1` and `component2`
to turn an independent roll frame into a fixture-suite entry; this is useful for building the
multi-scene CoolScan corpus from single-frame roll scans without forcing the whole suite to run in
no-stitch mode.
For corpus auditability, entries should also declare `scene_tags`, `exposure_tags`, and
`reference_evidence`, and `calibration_case` so coverage reports can show which real-image
conditions are actually tested and which fixtures include objective colour references such as
gray-card, ColorChecker, or Lab reference-patch evidence.
Entries may declare paired `component1_sha256` and `component2_sha256` values to bind the registry
to exact local scan bytes; coverage recomputes the file hashes and reports mismatches before a
fixture can be considered validation-ready. Entries may also declare `calibration_profile_sha256`
for external profile files or `calibration_library_sha256` for the deterministic digest of all JSON
records under a calibration library, so colour evidence cannot drift silently between reruns. Use
`--compute-fixture-hashes` with `--fixture-coverage` to emit actual component, summary-baseline,
and calibration evidence hashes before those expected values are added to the registry. Add
`--write-fixture-hash-registry <path>` during fixture coverage to write a separate registry
snapshot with available computed hashes filled in; the writer refuses to overwrite the input
registry path.
Entries may include an `expectations` object for fixture-suite guardrails such as
`stitch_decision`, `base_estimate_source`, `render_input_source`,
`render_input_reason_contains`, `mapping_strategy`, `selected_mapping_reason_contains`,
`output_color_space`, `selected_candidate`, `selected_candidate_rank`,
`candidate_acceptance_signatures_required`, `calibration_acceptance_status`,
`calibration_confidence_min`, `calibration_matrix_condition_number_max`,
`calibration_rejection_details_required`, `selection_rejections_required`, `candidate_risk`,
`tone_color_trust_state`, `highlight_chroma_compressed_ratio_min`,
`highlight_chroma_compressed_ratio_max`, `highlight_neutral_chroma_compressed_ratio_max`,
`shadow_chroma_compressed_ratio_max`, `selected_quality_score_max`, `technical_safety_score_max`,
`color_fidelity_score_max`, `memory_color_penalty_max`,
`spatial_consistency_penalty_max`, `selected_runner_up_quality_delta_min`,
`density_monotonicity_score_min`, `hue_linearity_score_min`,
`saturation_preservation_median_ratio_min`, `spatial_neutral_delta_p95_max`,
`post_scale_preserved_ratio_min`, `reference_patch_evaluation_required`,
`reference_patch_count_min`, `reference_patch_hue_family_regression_count_max`,
`reference_patch_selected_regresses_image_derived`,
`reference_patch_delta_e2000_delta_vs_image_derived_max`,
`reference_patch_max_delta_vs_image_derived_max`,
`reference_patch_delta_e_max_delta_vs_image_derived_max`,
`reference_patch_delta_e2000_max_delta_vs_image_derived_max`,
`reference_patch_rms_delta_e_max`, `reference_patch_rms_delta_e2000_max`,
`debug_artifacts_required`, and
`debug_artifact_kinds_required` (`candidate_comparison`, `gamut_clipping_map`, and
`scene_referred_prophoto_float`); strict suite runs report
expectation mismatches as fixture-suite issues instead of leaving the intended stitch/base decision,
output colour space, colour path, score ceiling/floor, reference-target count/error/regression,
hue-regression, or debug-artifact evidence implicit in prose.
The registry may include top-level `coverage_requirements` with minimum fixture, baseline,
calibrated/uncalibrated, scanner-profile, roll-profile, film-stock, scene-tag, exposure-tag, and
calibration-case counts plus reference-fixture/reference-evidence counts, selected calibration
reference-patch counts, debug-artifact expectation counts, matched component, summary-baseline, and
calibration SHA-256 counts, and required tag/value lists. It can also require validation-ready
cross-condition pairs such as `film_stock|calibration_case` and `scene_tag|exposure_tag`, plus
required debug artifact kinds (`candidate_comparison`, `gamut_clipping_map`, and
`scene_referred_prophoto_float`), so a
corpus cannot pass by satisfying film-stock, scene, exposure, and calibration counts only in
isolation. `min_fixtures` is evaluated against validation-ready fixtures, not merely declared
registry entries. `--fixture-coverage --strict` fails when those requirements are not met.

Audit whether a registry is ready to serve as real-image validation evidence:

```powershell
cargo run --bin scanstitch-validate -- `
  --fixture-registry docs/validation-fixtures.example.json `
  --fixture-coverage `
  --write-fixture-hash-registry output/validation/validation-fixtures.with-hashes.json `
  --strict `
  --summary-json output/validation/fixture-coverage.json `
  --summary-md output/validation/fixture-coverage.md
```

The coverage audit checks component file availability, optional component SHA-256 hashes, readable
RGB/RGBA TIFF headers, optional minimum sample precision, pair layout consistency, pair dimension
matching, compact baseline presence/parseability and contract completeness, usable and optionally
SHA-bound summary baselines and calibration evidence, scanner and roll profile IDs, film-stock
metadata, scene tags, exposure tags, reference-evidence tags, and calibration-case metadata.
For fixtures with calibration profiles or libraries, it also reports how many reference patches are
available in the selected calibration evidence, so reference-target coverage cannot be claimed from
labels alone. If a fixture requires `reference_patch_evaluation_required` but its selected
calibration evidence has no target patches, coverage reports
`reference_patch_evaluation_required_without_calibration_patches` before the suite is run.
Aggregate scanner/roll profile,
film-stock/tag/case counts and cross-condition pair counts are computed from validation-ready
fixtures only. `summary_baseline_parseable_count` counts baselines that deserialize,
`summary_baseline_contract_complete_count` counts baselines that satisfy the tracked baseline
contract, and `summary_baseline_count` is the strict readiness count for contract-complete baselines.
When `summary_baseline_sha256` is declared, mismatches report
`summary_baseline_sha256_mismatch` and are not validation-ready.
`summary_baseline_sha256_count` counts validation-ready fixtures whose declared baseline hash
matches,
`summary_baseline_sha256_computed_count` counts emitted actual baseline hashes, and
`min_summary_baseline_sha256_fixtures` can require pinned compact baselines for strict reruns.
`readable_tiff_pair_count` counts fixtures whose two component files are readable supported TIFFs,
`tiff_layout_consistent_pair_count` counts pairs with matching color type, sample precision,
channel count, and alpha layout, and `tiff_dimension_matched_pair_count` counts pairs with exact
matching component dimensions or a stitch-compatible one-row height delta with matching width.
When `min_tiff_bits_per_sample` is set, lower-precision component files report
`component*_tiff_bits_below_min` and are not validation-ready.
When `component*_sha256` values are declared, mismatched hashes report
`component*_sha256_mismatch` and are not validation-ready. `component_sha256_pair_count` counts
validation-ready fixtures whose two declared component hashes both match the local files. With
`--compute-fixture-hashes`, `component_sha256_computed_pair_count` counts fixtures whose two actual
component hashes were emitted even if the registry has not declared expected hashes yet, and
`min_component_sha256_pairs` can require that exact-file binding for a strict corpus.
When `calibration_profile_sha256` or `calibration_library_sha256` is declared, mismatches report
`calibration_profile_sha256_mismatch` or `calibration_library_sha256_mismatch` and are not
validation-ready. `calibration_sha256_count` counts validation-ready fixtures whose declared
calibration evidence hash matches, `calibration_sha256_computed_count` counts emitted actual
calibration hashes, and `min_calibration_sha256_fixtures` can require pinned calibration evidence
for strict reruns.
`calibration_evidence_count` counts calibration profiles/libraries that can be loaded and selected,
`uncalibrated_fixture_count` counts validation-ready fixtures without declared calibration evidence,
and `validation_ready_fixture_count` counts fixtures whose local files and metadata are complete
enough to rerun as validation evidence. Declared-but-missing or invalid files remain visible through
the declared/file counters and fixture-level issues. When required metadata or cross-condition pairs
are not covered, the summary also lists `missing_required_film_stocks`,
`missing_required_scene_tags`, `missing_required_exposure_tags`,
`missing_required_calibration_cases`, `missing_required_scanner_profiles`,
`missing_required_roll_profiles`, `missing_required_reference_evidence`,
`missing_required_debug_artifact_kinds`, `missing_required_film_stock_calibration_pairs`, and
`missing_required_scene_exposure_pairs` so corpus-building work has a concrete to-do list. The audit
also emits machine-readable `action_items` entries such as
`add_validation_ready_fixtures:<remaining_count>`,
`add_reference_patches:<remaining_count>`,
`add_validation_ready_fixture_for_scene_tag:<tag>`, and
`declare_debug_artifact_expectation_for_validation_ready_fixture:<kind>`. When registered fixtures
are blocked by local assets or metadata, the same list also includes fixture-specific repair actions
such as `provide_component1_for_fixture:<name>`,
`complete_summary_baseline_for_fixture:<name>`,
`provide_calibration_library_for_fixture:<name>`, and
`add_reference_patches_for_fixture:<name>`. TIFF pair failures include narrower actions such as
`replace_component_pair_with_layout_matched_tiffs_for_fixture:<name>` and
`replace_component_pair_with_dimension_matched_tiffs_for_fixture:<name>`, while low-bit component
failures include `replace_component1_with_minimum_bit_depth_tiff_for_fixture:<name>` or
`replace_component2_with_minimum_bit_depth_tiff_for_fixture:<name>`. These fixture-specific actions
are also copied onto
each fixture entry as `fixtures[].action_items` and shown in the Markdown fixture table's `Actions`
column. The JSON also includes `fixtures[].repair_plan`, a structured companion list with each
action's relevant local path or paths and a short repair detail, so local tools do not need to parse
action strings or rediscover registry paths.
The audit does not prove image quality by itself; it proves the fixture set has the local
ingredients and domain labels needed for strict real-image reruns, including whether selected
calibration data can drive reference-patch DeltaE76 and CIEDE2000 validation.

Run every registered fixture as one real-image suite:

```powershell
cargo run --release --bin scanstitch-validate -- `
  --fixture-registry docs/validation-fixtures.example.json `
  --fixture-suite `
  --strict `
  --debug `
  --summary-json output/validation/fixture-suite.json `
  --summary-md output/validation/fixture-suite.md
```

The suite writes each fixture's `report.json`, `summary.json`, and `summary.md` under its registry
`output_dir`, compares any declared compact baseline, and emits one suite-level JSON/Markdown result
with per-fixture pass, review, or failure state. It surfaces output freshness/profile checks, output
colour space, fixture-level `coverage_validation_ready`, `coverage_issues`, and
`coverage_action_items`, stitch decision, base-estimate source, render-input source, selected
candidate quality, candidate risk, tone-colour trust, tone-protection ratios, gamut preservation,
expected-vs-actual calibration source, exact scanner/roll profile IDs, requested film-stock
evidence, and any registry-declared colour-decision expectations. If a fixture declares
`debug_artifacts_required` or
`debug_artifact_kinds_required`, run the suite with `--debug`; otherwise the suite reports
`debug_artifacts_missing` or `debug_artifact_kind_missing:<kind>`.
Validation runs default to `--quality-mode balanced` to avoid writing perfect-mode master/review
artifacts during strict gates; pass `--quality-mode perfect` when those artifacts are part of the
review.
When the embedded coverage summary has unmet corpus requirements, the suite Markdown repeats the
coverage `action_items` list under `Coverage Action Items` so review runs show both fixture failures
and the concrete corpus-building steps in one report. The suite fixture table also has a
`Coverage actions` column copied from each coverage entry so missing baselines, missing components,
or TIFF repair work stay attached to the fixture that needs the repair.
When a fixture can be executed but its embedded coverage entry is not validation-ready, the suite
adds `fixture_suite:<name>:coverage_not_validation_ready`; missing component files remain hard
fixture failures.
It raises fixture-suite issues when declared calibration does not apply in the produced report, the
expected scanner/roll profile IDs are not the ones reported by the colour phase, an expected
stitch/base/output-space/render-input/candidate/risk/tone/gamut-preservation/reference-patch value
is missed, required debug artifacts are absent or stale, the output ICC profile does not
match the report, or stale render artifacts are present. Fixture-suite baselines must also contain
the core tracked stitch, render, colour-candidate, gamut, tone-trust, and candidate-acceptance
fields; a parseable but empty compact baseline is treated as incomplete rather than as proof of
comparability. Use `--output-dir some/root` to place each fixture under `some/root/<fixture-name>`
instead of the registry output directories.

To build missing compact baselines for registry fixtures, run the fixture suite without `--strict`
and add `--write-fixture-suite-baselines`. This writes only missing files declared by each fixture's
`summary_baseline`, records `summary_baseline_write_status` and
`summary_baseline_written_path` on the suite entry, then compares the freshly written baseline so
the per-fixture summary still reports baseline comparability. Existing files are left untouched and
reported as `skipped_exists`; add `--overwrite-fixture-suite-baselines` only when intentionally
refreshing accepted baselines. Baseline writing is rejected with `--strict`, so regression gates stay
read-only. For large local roll registries, add one or more `--fixture-suite-fixture <name>`
selectors to render and write baselines only for selected curated entries; the embedded coverage
summary still audits the whole registry so remaining corpus gaps stay visible. After accepting
generated baselines, pin `summary_baseline_sha256`, rerun
`--fixture-coverage --strict`, then rerun `--fixture-suite --strict --debug`.

Run deterministic synthetic colour-decision cases without local TIFFs:

```powershell
cargo run --release --bin scanstitch-validate -- `
  --synthetic-color-suite `
  --strict
```

This suite gates the candidate decision model for calibrated-profile wins, weak calibration quality
rejection, calibrated neutral-regression rejection, automatic unsafe-calibration rejection,
scanner-prior wins on weak image anchors, automatic unsafe scanner-prior rejection, sparse
uncalibrated anchor review, dirty/clipped/border sample rejection, biased scene-anchor review,
high-key exposure headroom normalization, destructive gamut fallback, reference-patch hue-family
and CIEDE2000 regression rejection, direct-density render-input fallback/rejection for destructive
gamut, neutral-regression, score, and risk decisions, and forced unsafe calibrated-mode failure.
The JSON and Markdown artifacts also surface the selected candidate's physical/perceptual model
evidence: density monotonicity, hue linearity, saturation preservation, spatial neutral Delta95,
memory-colour penalty, and spatial-consistency penalty.
It also gates tone colour-protection policies for weak neutral support and colour-candidate review,
including the guard that gamut repair stays active while creative colour cleanup is disabled.

To run the same summary against another local pair:

```powershell
cargo run --bin scanstitch-validate -- `
  --fixture my-pair `
  --component1 path/to/component1.tif `
  --component2 path/to/component2.tif `
  --output-dir output/validation/my-pair
```

To run with a calibration profile:

```powershell
cargo run --bin scanstitch-validate -- `
  --fixture logan `
  --output-dir output/validation/logan-calibrated `
  --calibration-profile path/to/scanner-film-profile.json
```

To run with a scanner/roll calibration library:

```powershell
cargo run --release --bin scanstitch-validate -- `
  --fixture logan `
  --output-dir output/validation/logan-calibrated `
  --calibration-library calibration `
  --scanner-profile coolscan-4000-vuescan-raw `
  --roll-profile logan-roll-2026-05 `
  --film-stock "Kodak Gold 200"
```

To summarize a report that already exists without reprocessing images:

```powershell
cargo run --bin scanstitch-validate -- `
  --fixture logan `
  --report output/validation/logan/report.json `
  --summary-json output/validation/logan/summary.json `
  --summary-md output/validation/logan/summary.md
```

To freeze the current compact baseline used by future strict runs:

```powershell
cargo run --release --bin scanstitch-validate -- `
  --fixture logan `
  --report output/validation/logan/report.json `
  --write-summary-baseline tests/fixtures/baselines/logan_summary_baseline.json `
  --summary-json output/validation/logan/summary.json `
  --summary-md output/validation/logan/summary.md
```

Tracked compact baselines are shaped by
[`validation-summary-baseline.schema.json`](validation-summary-baseline.schema.json). The schema
captures the fixture-suite contract for required stitch, render, base fallback evidence,
colour-candidate, gamut, tone-trust, tolerance, and candidate-acceptance evidence. `scanstitch-validate` rejects unknown
baseline fields so typoed or stale keys cannot be silently ignored, and direct `--compare-summary`
gates report `summary_baseline_incomplete:<field>` when a baseline is parseable but too sparse to
prove the tracked contract.

To compare a normal render report with a validation render report without touching either render:

```powershell
cargo run --bin scanstitch-validate -- `
  --fixture logan `
  --report output/report.json `
  --compare-report output/validation/logan/report.json `
  --summary-json output/compare-summary.json `
  --summary-md output/compare-summary.md
```

To gate a fresh LOGAN summary against the tracked compact baseline:

```powershell
cargo run --release --bin scanstitch-validate -- `
  --fixture logan `
  --compare-summary tests/fixtures/baselines/logan_summary_baseline.json `
  --strict
```

The summary includes each report timestamp/path, final dimensions, output colour/profile checks,
stitch decision/crop, base-estimate source, render input source, calibration status/source,
selected colour candidate, calibration acceptance, colorspace strategy, usable-gamut preservation,
tone/chroma compression ratios, and high-frequency luma/chroma residuals when the source report has
them. Comparison summaries also
include `comparison.status` and `comparison.issues`. Add `--strict` to make the command exit with
an error when comparison issues are present. Use `--fail-on` for selected issue groups, for example
`--fail-on stale-output,render-input-change,grain`. Supported groups are `any`, `stale-output`,
`dimensions`, `stitch-change`, `base-source`, `render-input-change`, `colorspace-strategy`,
`colorspace-quality`, `quality`, `calibration`, `gamut`, `grain`, `tone`, `debug-artifacts`, and
`summary-baseline`; exact issue names also match. Summary-baseline comparisons populate
`summary_baseline_comparison.status`, `summary_baseline_comparison.issues`, and tracked deltas for
the compact baseline fields, including calibration status/source, scanner/roll profile identity,
requested film stock, film-stock match status, matched roll-profile evidence, selected candidate
rank, raw base proxy-confidence/support-fraction deltas with a dedicated support-fraction
tolerance, technical-safety and
colour-fidelity score split, calibration confidence/matrix-condition/rejection-detail diagnostics,
runner-up decision margin, stable candidate-acceptance signatures (candidate kind, mapping
strategy, rank, status, mode eligibility, selected/rejected state), optional reference-patch
XYZ max, DeltaE76 RMS/max, and CIEDE2000 RMS/max regression evidence, hue-family regression evidence, density,
hue-linearity, saturation, memory-colour, and spatial
cast model evidence, neutral/dominant-anchor support, weak/gamut fallback use, and neutral-trim
state.

For stale-output investigations, use fresh directories so the final TIFFs and summaries come from
the same run:

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

See `docs/grain-investigation.md` for the current fresh LOGAN comparison and grain conclusion.

## What To Compare

Compare `summary.json` between runs. These fields are the first-pass regression signals:

| Area | Fields |
|-|-|
| Report/render | `report.generated_at`, `report.source_report_path`, `render.output_path`, `render.output_modified_at`, `render.output_width/height`, final save `output_color_space` / `output_icc_profile` in the full report, independent `render.output_file_icc_profile_*` TIFF inspection, `render.render_review_status`, `render.render_reviewable`, `render.input_base_confidence`, `render.positive_input_likely_negative_like`, `render.positive_input_accepted_high_warm_score`, `render.positive_input_orange_mask_score`, `render.positive_input_reason`, `render.stale_render_artifact_count` |
| Comparison | `comparison.status`, `comparison.issues`, output dimension match, render input changes, grain p95 ratios |
| Stitch | `decision`, `confidence`, `chosen_hypothesis`, `search_selection_reason`, `evidence_score`, `prior_weight`, `overlap_support_score`, `vertical_offset_plausibility_score`, `local_consistency_score`, `plausibility_score`, `seam_exposure_correction.applied`, `seam_exposure_correction.gain_rgb`, `seam_exposure_correction.seam_score_before/after`, seam clipping ratios |
| Base/density | `base_confidence`, `raw_base_confidence`, `raw_base_proxy_confidence`, `raw_base_support_fraction`, `density_confidence`, `base_estimate_source` |
| Colorspace | `calibration_status`, `calibration_source`, scanner/roll/film-stock diagnostics in the full report, `render_input_source`, `render_input_reason`, `mapping_strategy`, `selected_candidate`, `selected_candidate_rank`, `selected_quality_score`, `technical_safety_score`, `color_fidelity_score`, `selected_quality_components`, `candidate_risk`, `tone_color_trust_state`, `selected_runner_up_quality_delta`, `candidate_quality_scores`, candidate `rendered_tone_quality` and `color_model_quality`, rendered-tone, tone-chroma-cleanup, density-monotonicity, hue-linearity, saturation-preservation, memory-colour, and spatial-consistency score components, `candidate_acceptance`, `calibration_acceptance`, `neutral_estimate_quality`, `neutral_sample_rejections`, `dominant_anchor_sample_rejections`, `dominant_anchor_quality`, optional `reference_patch_evaluation` with XYZ, D50 Lab DeltaE76, and CIEDE2000 residuals, `neutral_trim_before_after`, `neutral_trim_applied`, `neutral_trim_scale`, `scene_referred_detail_fusion`, `exposure_scale`, `post_scale_preserved_ratio`, calibrated-vs-image exposure/clipping deltas, `regularization_lambda`, `channel_anchor_counts`, `channel_anchor_low_support`, `dominant_anchor_bands`, `weak_anchor_fallback_used`, `gamut_fallback_used`, image-matrix low/high clipping, debug artifact paths for `color_candidate_comparison_artifact`, `gamut_clipping_map_artifact`, and `scene_referred_prophoto_float_artifact`, `debug_artifacts` existence/freshness inspection, plus compact gamut-map and scene-referred float artifact diagnostics |
| Tone | Tone/chroma compression ratios; `color_protection_policy`, `color_trust_state`, `color_protection_reason`; shadow, midtone, midtone-neutral, bright-neutral, and bright-saturated saturation medians/p95 values; post-tone high/low clipping ratios; `local_luminance_detail_*`; `adaptive_vibrance_*`; `noise_reduction_*`; all-sample and flat-area `high_frequency_grain.*_residual_p95` values |

## Interpreting Deltas

Treat these as review triggers:

| Field | Review when |
|-|-|
| `stitch.decision` | It changes between accepted, rejected, and skipped. |
| `stitch.confidence` | It drops materially or the chosen hypothesis changes. |
| `report.generated_at` / `render.output_modified_at` | A report or TIFF is older than the render you intended to inspect. |
| `render.stale_render_artifact_count` | It is non-zero before a visual comparison that uses debug/intermediate TIFFs. |
| `colorspace.debug_artifacts` | A named color candidate preview or gamut clipping map is `missing`, `stale`, `not_file`, or has unknown freshness before review. Strict summary-baseline validation reports this as `colorspace_debug_artifact_invalid:<kind>:<status>`. |
| `render.output_file_icc_profile_matches_report` | It is `false`, meaning the saved TIFF profile tag does not match the report's claimed output profile. |
| `render.render_review_status` | It is `blocked_low_base_confidence`, meaning density inversion used a fallback-quality base estimate and the rendered TIFF should not be treated as a reviewable positive. |
| `render.positive_input_likely_negative_like` | It is `true` in positive mode, meaning the already-positive path was requested for input whose channel medians look like orange-mask negative film. Use negative mode or replace the roll with actual positive scans before treating colour/tone output as a positive regression. |
| `render.positive_input_accepted_high_warm_score` | It is `true` in positive mode when the aggregate warm-channel score is high but R/G and G/B ratios are mild enough to accept the frame as already-positive input. Use it to distinguish warm positive scenes from negative-like warnings. |
| `stitch.seam_exposure_correction.applied` | It flips unexpectedly, or correction is applied with non-identity gains on a previously consistent seam. |
| `stitch.seam_exposure_correction.seam_score_after` | It fails to improve relative to `seam_score_before`, or clipping ratios rise materially. |
| `base_density.base_confidence` | It drops enough to cap downstream confidence. |
| `colorspace.render_input_source` | It switches between ICA, direct-density, and positive RGB render input. |
| `colorspace.calibration_status` | It changes between `not_configured`, `applied`, and `rejected`. |
| `colorspace.calibration_source` | It switches between an external profile, calibration library, and image-derived anchors. |
| `colorspace.calibration_confidence` / `calibration_matrix_condition_number` / `calibration_rejection_details` | Confidence drops, the matrix condition worsens materially, or the recorded calibration failure details change. |
| `colorspace.calibration.film_stock` | Requested film-stock evidence changes match status, becomes ambiguous/unmatched, or starts rejecting an explicit roll profile. |
| `colorspace.mapping_strategy` | It switches into or out of a neutral-balance fallback. |
| `colorspace.selected_candidate` | Candidate ranking starts preferring a different color mapping family. |
| `colorspace.candidate_acceptance_signatures` | A candidate changes kind, strategy, rank, acceptance status, mode eligibility, selected state, or rejected state in a compact baseline comparison. |
| `colorspace.selected_quality_score` | It changes materially, especially after calibration or gamut-safety work. |
| `colorspace.technical_safety_score` / `color_fidelity_score` | Either score rises materially, which means safety or fidelity penalties increased even if the total candidate decision stayed stable. |
| `colorspace.candidate_risk` | It changes away from `safe`, especially to `review_reference_fit`, `review_gamut`, `review_tone_quality`, `review_model_plausibility`, `review_quality_score`, `review_anchor_support`, or `review_neutral_support`. |
| `colorspace.tone_color_trust_state` | It changes away from `trusted`, which means tone colour cleanup is limited or requires review. |
| `colorspace.selected_runner_up_quality_delta` | It shrinks toward zero, flips sign, or becomes unavailable after a color-scoring change. |
| `colorspace.candidate_acceptance` | A calibrated, scanner-prior, image-derived, or neutral candidate changes selected/rejected status or reason. |
| `colorspace.calibration_acceptance` | A calibrated/scanner-prior candidate is newly rejected for quality, safety, neutral regression, or reference-fit regression. |
| `colorspace.neutral_estimate_quality` | Neutral support becomes one-band-only or insufficient. |
| `colorspace.neutral_estimate_quality.score` / `dominant_anchor_quality.score` | Either score drops materially in a compact baseline comparison, even if the selected mapping is unchanged. |
| `colorspace.reference_patch_evaluation` | Selected XYZ RMS/max, D50 Lab DeltaE76, or CIEDE2000 residuals regress versus image-derived mapping, candidate-level hue-family regressions appear, or worst hue families concentrate in one color family. |
| `colorspace.selected_quality_components.rendered_tone_penalty` / `tone_chroma_cleanup_penalty` | They rise materially, which means candidate ranking is being influenced by rendered-tone colour or tone cleanup risk. |
| `colorspace.selected_quality_components.memory_color_penalty` / `spatial_consistency_penalty`, plus selected `hue_linearity_score` | They regress materially, which means compact baseline validation is catching physical/perceptual model drift even when candidate selection is unchanged. |
| `colorspace.candidate_quality_scores[].rendered_tone_quality` | Shadow/midtone or neutral-highlight saturation rises, bright-saturated median saturation collapses, or post-cleanup clipping appears. |
| `colorspace.candidate_quality_scores[].color_model_quality` | Density monotonicity bins regress, hue-linearity concentration collapses, saturated-colour preservation ratios collapse or explode, memory-colour proxies become broadly implausible, or neutral-tile cast p95 rises. |
| `colorspace.neutral_trim_applied` | Neutral trim unexpectedly turns on or off, or per-band trim deltas show a populated band worsened. |
| `colorspace.scene_referred_detail_fusion.*` | Detail fusion unexpectedly disables, changes strength materially, or starts skipping a large new ratio of pixels for negative or non-finite scene-referred values. |
| `colorspace.post_scale_preserved_ratio` | It drops materially, especially after applying a calibration profile. |
| `colorspace.channel_anchor_low_support` | A new channel becomes weak, or a weak channel unexpectedly disappears after unrelated changes. |
| `colorspace.dominant_anchor_quality` | Dominant-channel anchors become single-band, luma-concentrated, weak-margin, or unstable enough to add `anchor_stability_penalty` / `review_anchor_support`. |
| `colorspace.image_matrix_pre_scale_clipped_low_ratio` | Low clipping rises materially, especially with weak anchors. |
| `tone.bright_neutral_saturation_p95` | It rises after a change that should not affect highlight neutrality. |
| `tone.color_protection_policy` | It changes between `enabled`, `weak_neutral_bounded_neutral_cleanup` (or legacy `weak_neutral_bounded_highlight_cleanup` / `neutral_highlight_disabled_weak_neutral`), `review_bounded_neutral_shadow_cleanup` (or legacy `review_shadow_cleanup_only`), and `disabled_color_candidate_review`. |
| `tone.color_trust_state` | It changes between `trusted`, `limited_weak_neutral`, and `review_required`. |
| `tone.bright_saturated_saturation_p95` | It falls after a change that should not desaturate saturated highlights. |
| `tone.post_chroma_compression_clipped_high_ratio` / `low_ratio` | Clipping rises from near-zero to visible levels. |
| `tone.local_luminance_detail_*` | The applied ratio, EV strength, or clip-limited ratio changes materially between reports that should have equivalent tone/detail behavior. |
| `tone.adaptive_vibrance_*` | The trusted-colour vibrance pass unexpectedly enables/disables, changes strength materially, or becomes heavily texture/gamut-limited. |
| `tone.noise_reduction_*` | The final render denoise unexpectedly disables, changes strength materially, or starts applying to nearly all high-texture/saturated pixels. |
| `tone.high_frequency_grain.*_residual_p95` | Luma or chroma residuals change materially between reports that should be visually equivalent. |

The validation harness is a repeatability check, not a replacement for visual review. For color, tone, or colorspace changes, rerun LOGAN and inspect the local render before accepting the summary delta.

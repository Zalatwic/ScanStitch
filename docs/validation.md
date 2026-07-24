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

For an automation delivery gate, add `--require-reviewable` to a direct render, an existing
`--report`, `--fixture-suite`, or `--roll-suite`. The validator deliberately lets processing finish,
writes the report/render and compact JSON/Markdown, and then exits nonzero unless final evidence is
both present and internally consistent as `render_review_status=reviewable` plus
`render_reviewable=true`. It additionally reopens the saved TIFF to verify the reported positive
dimensions, RGB16 storage, and ICC profile; reopens every requested scene-referred master and sRGB
review proof to verify its dimensions, storage, and profile; for schema-v4 reports, independently
recomputes and requires the declared SHA-256 of the primary and every requested auxiliary artifact;
requires coherent constant-lightness/
hue sRGB gamut-map diagnostics with no non-finite inputs or remaining target excursions; and requires an explicit zero
stale-render-artifact count. The flag is rejected for inventory, coverage, listing,
orientation-review, render-review-draft, and synthetic-decision modes because those modes do not
produce final renders.

The schema-v4 hashes detect missing, changed, or structurally valid substituted artifacts while
the report is unchanged. They do not sign or authenticate `report.json`; coordinated report and
artifact edits, including a schema downgrade, are outside this gate's trust boundary. For
reviewed corpus admission, pin the exact completed render-review manifest hash in the fixture
registry and preserve reviewer provenance separately.

Current writers also report `artifact_commit_policy`: each file is fully encoded in a
same-directory temporary and atomically replaces its destination only on success. This prevents a
failed encoder from exposing a partial replacement, but is not a transaction across the
primary/master/proof/report set and does not claim file/directory `fsync` power-loss durability.
The SHA gate remains authoritative after all independent per-file commits.

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
local-fixtures/render-reviews/my-split-frame/render-review.json
local-fixtures/render-reviews/uncalibrated-night/render-review.json
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

Strict fixture execution has an additional delivery guard. A consistent non-reviewable render is
an issue unless the fixture explicitly pins `expectations.render_reviewable: false`; this keeps an
accidental regression from becoming a stable passing baseline while still allowing fixtures whose
purpose is to prove a diagnostic fallback. Pin the matching `render_review_status` as well when its
exact reason is part of the test. `--require-reviewable` is stronger: it rejects every such fixture,
even when diagnostic delivery was explicitly expected, after preserving all suite artifacts.

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
Both `scanstitch` and `scanstitch-validate` accept a single input directly. For a single
full-frame DNG, pass only `--component1`; `--component2` is optional and is used only when a
second scan really exists. The load/stitch report records one input and
`stitch.decision=skipped_single_input`:

```powershell
cargo run --release --bin scanstitch-validate -- `
  --fixture testroll-raw-0000 `
  --component1 TESTROLL/RAW_0000.dng `
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

The writer emits one real `component1` and no dummy `component2`, with
`force_no_stitch: true`, local baseline paths under `local-fixtures/baselines/`, per-frame
output directories, the selected `input_mode`/`bit_depth`, and any supplied calibration
profile/library, scanner profile, roll
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
`noise_reduction_*`, grain-detail evaluated/supported/review state, independent luminance/chroma
support and probe counts, and median/p10 retention,
`colorspace_post_scale_preserved_ratio`, positive-input suitability, tone chroma-compression ratios,
rendered luminance p05-p95 range, midtone luminance p50 mean/min/max/range, shadow/full-midtone/midtone-neutral/bright-neutral RGB balance deltas, and post-tone clipping maxima. Every frame also retains
the authoritative final render status/reviewable boolean and complete rendered-tone evaluation,
confidence/status, review flag/reason, input/fitted/rendered range relationship, relative retention,
and clipping maxima. A false or missing final review decision and a true or missing tone-output
review decision create explicit roll issues, so an otherwise-safe candidate cannot make a blocked
render pass. The `review` aggregate includes final-render and tone-output status/reason counts in
addition to candidate-risk, tone-colour-trust, calibration-status, reference-patch, and normalized
issue-kind counts, so roll-wide blockers can be inspected without expanding every frame. The Markdown report
mirrors these in roll and per-frame quality tables so full-roll runs can be compared for
all-sample and flat-area chroma/luma residuals, denoise strength/limiting, mode-suitability issues, colour-balance drift, review-gate causes, and
midtone consistency, weak or collapsed full-frame contrast, and highlight/shadow headroom without opening every frame summary.
Use the flat-area residuals when judging whether smooth regions still have distracting grain; the
all-sample residuals intentionally include real scene edges and texture. Roll aggregates exclude
grain-off identity ratios and unsupported channels from retention means/minima. Inspect
`grain_detail_evaluated_count`, `grain_detail_decision_supported_count`,
`grain_detail_review_required_count`, per-channel supported counts and minimum probe counts, and
per-channel median/p10 retention mean/min before treating a whole roll as safely denoised.
Grain reduction defaults off and is independent of render intent. For a denoise-validation run,
pass `--grain-reduction on --grain-strength <0..1> --grain-scale <0.5..4>`; the validator forwards
all three settings to isolated roll-suite child processes. Inspect the full report's
`grain_reduction.requested/effective/before/after/effect` record to distinguish a requested pass
from one that actually ran and to measure its effect. Also inspect
`grain_reduction.detail_retention`: it measures pre/post contrast at filter-scale and 2x-scale
coherent luminance and opponent-colour edges, including isoluminant coloured edges that a luma-only
gate would miss. A channel is supported only with at least 64 probes; supported median retention
must remain at least 0.90 and p10 retention at least 0.70. Missing coherent structure is reported as
unsupported rather than silently counted as a pass, while a supported failure propagates
`render_review_status=review_required_grain_detail`.
The same report exposes `noise_reduction_structure_gate_start/end` and
`noise_reduction_structure_excluded_ratio`. The end matches the independent 0.035 luminance-detail
contrast floor; structure, coherent opponent-colour support, and incomplete image-boundary windows
at or beyond the policy are exact no-ops. Strict validation rejects invalid gate ordering, a gate
end inconsistent with the detail floor, impossible overlap between active and excluded populations,
disabled nonzero effects, and invalid mean/max delta accounting. Roll summaries retain each gate
and exclusion ratio and aggregate exclusion mean/minimum so a representative structured run cannot
hide a near-universal mask behind acceptable average detail retention.
Use `--compare-roll-suite <previous-roll-suite.json>` on a rerun to attach aggregate and per-frame
before/after deltas to the current roll-suite JSON and Markdown. For older baselines that predate
the midtone range aggregate, it derives max/range from per-frame `midtone_luminance_p50` values.
The comparator reports material regressions in failed-frame count, darkened midtone placement,
widened midtone range, increased chroma or flat-area grain residuals, changed denoise enablement,
lost detail support, new grain-detail review, luminance/chroma p10 retention drops, dropped rendered
luminance range, weaker neutral balance, reduced preserved gamut, and new post-tone highlight clipping.
It also reports per-frame lost final-render reviewability, regressed tone-output status, newly
required tone review, lost tone evidence confidence or relative range retention, and increased
high/low clipping, while still showing small improvements and unchanged fields explicitly:

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

Use identical grain controls for both sides of a roll comparison. With an unchanged frame set, a
changed enabled/evaluated count, increased detail-review count, or reduced overall/luminance/chroma
support count is a comparison issue. Aggregate p10 mean/min drops greater than 0.03 and per-frame
drops greater than 0.05 are also review issues. Older roll summaries that lack the new fields remain
comparable: absent historical evidence produces no invented delta, while the current report still
exposes the new measurements. Once a baseline contains tone evidence, losing it in a current frame
is itself a comparison issue. `--fail-on grain` selects grain issues and residual-p95 drift;
`--fail-on tone` selects the rendered-tone roll issues and comparisons.

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
Entries may also override pipeline shape with `input_mode`, `bit_depth`, `deskew`,
`deskew_angle_degrees`, `orientation_correction`, `force_stitch`, and `force_no_stitch`, and independently pin
`grain_reduction` (`off`/`on`), `grain_strength` (0..1), and `grain_scale` (0.5..4). A deskew angle is valid only with
`deskew: manual`. Direct `--fixture` execution and fixture suites resolve these fields through
the same effective-settings path. A declared fixture grain field takes precedence over the matching
global CLI control; an omitted field inherits the CLI value. Fixture-suite JSON and Markdown retain
the effective mode/strength/scale, and coverage output retains the declared values. Registry
validation rejects invalid bounds and a pinned mode that contradicts
`expectations.grain_reduction_enabled`. An independent roll frame needs only `component1`;
`force_no_stitch: true` may remain explicit in a generated entry but no duplicate path is needed.
This supports a multi-scene CoolScan corpus made from independent roll frames without forcing the
whole suite into a global no-stitch mode.

`component1` is the only required input field. Add `component2` when a frame actually spans two
scans. For a frame split across three or more scans, declare `component2` and append ordered
objects under `additional_components`, each with `path` and an optional `sha256`.
`additional_components` without `component2` is rejected, preserving stable component numbering.
Both direct fixture execution and `--fixture-suite` pass the entire one-or-more-component vector
to the pipeline. The paths need not already be left-to-right; the stitcher infers that order from
overlap evidence.
For corpus auditability, entries should also declare `scene_tags`, `exposure_tags`, and
`reference_evidence`, and `calibration_case` so coverage reports can show which real-image
conditions are actually tested and which fixtures include objective colour references such as
gray-card, ColorChecker, or Lab reference-patch evidence.
Entries may declare `component1_sha256`, optional `component2_sha256`, and `sha256` on every
`additional_components` item to bind the complete component set to exact local scan bytes. A
single-component fixture may pin `component1_sha256` alone; for two or more components, hashes are
all-or-none across the actual component set. A second-component hash without `component2` is
rejected. Coverage recomputes every file hash and reports a mismatch before a fixture can be
considered validation-ready. Entries may also declare `calibration_profile_sha256`
for external profile files or `calibration_library_sha256` for the deterministic digest of all JSON
records under a calibration library, so colour evidence cannot drift silently between reruns. Use
`--compute-fixture-hashes` with `--fixture-coverage` to emit actual component, summary-baseline,
and calibration evidence hashes before those expected values are added to the registry. Add
`--write-fixture-hash-registry <path>` during fixture coverage to write a separate registry
snapshot with available computed hashes filled in; the writer refuses to overwrite the input
registry path.
Entries may include an `expectations` object for fixture-suite guardrails such as
`deskew_status`, `deskew_applied`, `deskew_review_required`,
`deskew_retained_area_ratio_min`, `deskew_all_components_applied`,
`deskew_minimum_component_retained_area_ratio_min`,
`border_crop_all_components_cropped`,
`border_crop_minimum_removed_edge_count_per_component_min`,
`border_crop_retained_area_ratio_min`, `border_crop_retained_area_ratio_max`,
`border_crop_rejected`, `deskew_correction_degrees_expected`,
`deskew_correction_tolerance_degrees`, `border_crop_components_expected`,
`orientation_components_expected`, `stitch_decision`, `inferred_component_order`,
`technical_white_balance_status`,
`technical_white_balance_applied`, `technical_white_balance_review_required`,
`creative_temperature`, `creative_tint`, `seam_exposure_model`,
`seam_exposure_held_out_validation_passed`,
`seam_exposure_held_out_improvement_over_gain_min`,
`seam_exposure_offset_normalized_abs_max`, `seam_exposure_spatial_slope_abs_min`,
`seam_exposure_spatial_slope_abs_max`,
`seam_exposure_spatial_slope_agreement_ratio_min`,
`seam_exposure_held_out_spatial_improvement_over_best_constant_min`,
`seam_exposure_spatial_offset_slope_normalized_abs_min`,
`seam_exposure_spatial_offset_slope_normalized_abs_max`,
`seam_exposure_spatial_offset_endpoint_normalized_abs_max`,
`seam_exposure_spatial_affine_slope_agreement_ratio_min`,
`seam_exposure_spatial_affine_center_offset_delta_normalized_max`,
`seam_exposure_held_out_spatial_gain_offset_improvement_over_best_simpler_min`,
`seam_blend_required`, `seam_blend_mode`,
`seam_blend_review_required`, `seam_detail_review_required`,
`seam_detail_supported_scale_count_min`, `seam_detail_max_symmetric_energy_ratio_max`,
`seam_gradient_ratio_max`, `seam_overlap_p95_abs_difference_max`,
`base_estimate_source`, `base_confidence_min`, `density_inversion_skipped`,
`negative_response_model`, `negative_response_source`, `negative_response_accepted`,
`negative_response_review_required`, `negative_response_crosstalk_model`,
`negative_response_characteristic_curve_model`, `negative_response_measured_model_id`,
`negative_response_measured_confidence_min`,
`negative_response_held_out_delta_e00_rms_max`,
`negative_response_held_out_max_delta_e00_max`,
`negative_response_held_out_improvement_over_unit_slope_min`,
`negative_response_density_noise_gain_max`,
`negative_response_curve_extrapolated_ratio_max`,
`negative_response_signed_headroom_preserved`, `negative_response_curve_interpolation`,
`render_input_source`,
`render_input_reason_contains`, `mapping_strategy`, `selected_mapping_reason_contains`,
`output_color_space`, `selected_candidate`, `selected_candidate_rank`,
`candidate_acceptance_signatures_required`, `calibration_acceptance_status`,
`calibration_color_mapping_applied`,
`calibration_confidence_min`, `calibration_matrix_condition_number_max`,
`calibration_rejection_details_required`, `selection_rejections_required`, `candidate_risk`,
`tone_color_trust_state`, `neutral_safety_rescue_applied`,
`neutral_safety_rescue_preserved_ratio_gain_min`,
`neutral_safety_rescue_midtone_saturation_p95_reduction_min`,
`neutral_safety_rescue_reason_contains`, `highlight_chroma_compressed_ratio_min`,
`highlight_chroma_compressed_ratio_max`, `highlight_neutral_chroma_compressed_ratio_max`,
`shadow_chroma_compressed_ratio_max`, `grain_reduction_enabled`,
`grain_detail_review_required`, `grain_detail_decision_supported`,
`grain_detail_luminance_probe_count_min`, `grain_detail_chroma_probe_count_min`,
`grain_detail_luminance_p10_retention_min`, `grain_detail_chroma_p10_retention_min`,
`grain_reduction_applied_ratio_min`,
`grain_reduction_structure_excluded_ratio_min`,
`grain_reduction_flat_luma_p95_reduction_ratio_min`,
`grain_reduction_flat_chroma_p95_reduction_ratio_min`,
`selected_quality_score_max`, `technical_safety_score_max`,
`color_fidelity_score_max`, `memory_color_penalty_max`,
`spatial_consistency_penalty_max`, `selected_runner_up_quality_delta_min`,
`density_monotonicity_score_min`, `hue_linearity_score_min`,
`saturation_preservation_median_ratio_min`, `spatial_neutral_delta_p95_max`,
`post_scale_preserved_ratio_min`, `render_luminance_range_p05_p95_min`,
`render_review_status`, `render_reviewable`, `tone_output_confidence_status`,
`tone_output_review_required`, `tone_output_evidence_confidence_min`,
`render_to_mapped_luminance_range_ratio_min`,
`post_chroma_compression_clipped_high_ratio_max`,
`post_chroma_compression_clipped_low_ratio_max`, `reference_patch_evaluation_required`,
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
seam mode/continuity, output colour space, colour path, score ceiling/floor, reference-target count/error/regression,
hue-regression, or debug-artifact evidence implicit in prose.

The geometry-preparation coverage contract is deliberately stronger than a generic deskew/crop
expectation. `min_geometry_preparation_contract_fixtures` counts a validation-ready fixture only
when it pins `deskew_status: applied`, deskew application for every component, no deskew review, and
a minimum per-component retained-area floor of at least 0.90. It must then pin all components as
cropped, at least four removed edges per component, a crop-retained floor of at least 0.50, an upper
bound strictly below 1.0, and `border_crop_rejected: false`. The suite compares these values with
the aggregate `deskew` and `border_crop` summaries. Partial declarations produce
`complete_geometry_preparation_contract_for_fixture:<name>` and list every missing or too-weak
field.

The separate geometry-accuracy contract prevents those aggregate checks from being mistaken for
crop truth. `min_geometry_accuracy_contract_fixtures` counts a validation-ready fixture only when
the complete preparation contract is also present, the expected signed correction is between -3
and +3 degrees, and its tolerance is at most 0.05 degrees. It also requires exactly one uniquely
indexed `border_crop_components_expected` record for every input component. Each record pins
`top_removed`, `bottom_removed`, `left_removed`, and `right_removed` to nonnegative pixel counts with
`tolerance_px` at most 4. Strict suite execution compares those annotations with
`deskew.correction_degrees` and the indexed `border_crop.components[]` summaries; a mismatch names
the exact component/edge and expected, tolerance, and actual values. Partial declarations produce
`complete_geometry_accuracy_contract_for_fixture:<name>`. The broader schema limits (3 degrees and
64 pixels) allow draft annotations to parse, but such loose declarations do not count toward this
coverage requirement.

Orientation is an independent exact contract. `min_orientation_accuracy_contract_fixtures` counts
a validation-ready fixture only when `orientation_components_expected` has exactly one unique,
one-based record for every input. Each record requires `upright_approved: true` and pins
`decoded_pixel_sha256`, source `tag_present`, nullable source `tag_value`, effective `transform`,
combined `applied`, `source_width`, `source_height`, final `output_width`, and final `output_height`.
The digest covers normalized, effectively oriented RGB samples plus dimensions, channel count, and
working bit depth. A fixture-level `orientation_correction` uses the same values as the CLI and is
composed after the EXIF/DNG transform. Runtime completeness enforces the resulting D4 mapping:
absent/1 with no correction is identity and un-applied; source metadata and explicit correction may
otherwise combine into any EXIF 1-8 transform, with dimensions swapped exactly for effective tags
5/6/7/8. Strict suite execution compares every runtime field
and digest with compact `input_orientation.components[]` evidence and emits an exact
component/field mismatch. Partial records produce
`complete_orientation_accuracy_contract_for_fixture:<name>`.

For a real fixture, first run without claiming the coverage minimum, visually approve each decoded
component as upright, then copy its `decoded_pixel_sha256` from `summary.json` and set
`upright_approved: true`. Do not copy the example's placeholder digests. This binds the reviewed
pixel result rather than trusting the source tag alone. The deterministic fixture proves the
binding and failure paths; only an actual review supplies semantic uprightness evidence.

Use the standalone review mode to produce that evidence without paying for a full reconstruction:

```powershell
cargo run --bin scanstitch-validate -- `
  --fixture-registry fixtures.json `
  --fixture frame-001 `
  --write-orientation-review output/orientation-review/frame-001
```

It writes `component-NN-orientation-preview.png`, `orientation-review.md`, and
`orientation-review.json`. The preview is per-channel stretched (and display-inverted for negative
input) strictly to make orientation legible; it is not colour evidence. The JSON contains a
paste-ready `orientation_components_expected` array, but its `upright_approved` fields are always
`false`, records the metadata-relative `orientation_correction`, and its schema enforces that
non-self-approving draft state. Reviewers must inspect every
PNG before copying the array into the fixture and changing approved entries to `true`. The schema is
`orientation-review.schema.json`. For a standalone single scan, omit the registry and pass
`--component1`, plus the factual `--input-mode`, `--bit-depth`, and, when needed,
`--orientation-correction`.

## Hash-Bound Final Render Review

Automated safety evidence cannot decide whether meaningful content was cropped, whether a seam is
visible at 100%, or whether a technically valid grade is photographically preferred. Generate a
human-review draft from a completed perfect-mode report:

```powershell
cargo run --bin scanstitch-validate -- `
  --fixture frame-001 `
  --report output/frame-001/report.json `
  --write-render-review output/render-review/frame-001
```

When the reviewed report requested grain reduction, also provide the exact matched perfect-mode
grain-off report:

```powershell
cargo run --bin scanstitch-validate -- `
  --fixture frame-001 `
  --report output/grain-on/report.json `
  --write-render-review output/render-review/frame-001 `
  --render-review-grain-control-report output/grain-off/report.json
```

The command requires a requested sRGB proof and writes `render-review.json` plus
`render-review.md`. It never approves anything: `review_status` is
`requires_human_approval`, reviewer fields are empty, and every applicable decision is false. The
draft records ordered source paths, file SHA-256 values, orientation-materialized decoded-pixel
SHA-256 values, the source-report hash, and hashes/sizes for the primary TIFF, scene-referred float
master, and sRGB proof. Stitch/seam decisions apply only to multiple inputs; grain/detail applies
only when reduction was requested. A grain-applicable schema-v2 draft is rejected unless the
control explicitly reports reduction off, is technically reviewable, has the same ordered
source/decode evidence, non-grain render command, and valid schema-v3-or-newer report executable
SHA-256, and
has a byte-identical scene-referred master. Its report, primary TIFF, master, and proof are all
separately hash-bound. Only output-directory and grain-control arguments may differ. Legacy
schema-v1 non-grain reviews remain readable; a grain-on v1 review or a legacy report without exact
binary identity cannot be approved as a matched control.

Inspect the exact sRGB proof for orientation, retained content, crop, colour, neutrality, tone, and
overall preference. Inspect the primary/master artifacts at full resolution for union bounds,
seam tone/colour/texture/sharpness, recoverable range, and grain/detail. An approved manifest must
set `review_status=approved`, name the reviewer, provide an RFC3339 `reviewed_at`, provide overall
notes, and approve every applicable decision with specific notes. Non-applicable decisions remain
unapproved. Its schema is [`render-review.schema.json`](render-review.schema.json).

Add the completed path as `render_review` and pin its exact SHA-256 as
`render_review_sha256` in the fixture. Set `min_approved_render_review_fixtures` in coverage. A
fixture counts only if the manifest hash matches; ordered inputs match the registry; source-file
and decoded-pixel hashes still match the report; report and every promised artifact still match;
any required grain-off control remains compatible and byte-exact; the technical production gate
remains reviewable; and all human decisions are complete. Changing
a source scan, orientation decode, report, output, master, proof, review text, or fixture input
order invalidates admission. Free-form `reference_evidence` strings describe corpus metadata such
as a photographed gray card or ColorChecker; they are not human approval and cannot satisfy this
contract.

The negative-reconstruction coverage contract proves a calibrated physical reconstruction, not
only an inversion call. `min_negative_reconstruction_contract_fixtures` counts only a
validation-ready `input_mode: negative` fixture with a trusted measured base source, base confidence
at least 0.30, and `density_inversion_skipped: false`. It requires the accepted
`measured_nonlinear_dye_separation` response from
`measured_roll_target_with_held_out_validation`, the measured 3x3 scanner-density-to-film-layer
matrix, monotone PCHIP density-to-scene-log-exposure curves, a nonempty model ID, confidence at least
0.75, held-out CIEDE2000 RMS at most 6 and maximum at most 15, at least 0.25 improvement over the
unit-slope RMS baseline, noise gain at most 4, and runtime curve extrapolation at most 0.01. The
fixture must also pin signed headroom, endpoint-tangent PCHIP interpolation, direct-density render
input, accepted calibration, safe candidate risk, trusted tone colour, and linear ProPhoto D50
output. Partial declarations produce
`complete_negative_reconstruction_contract_for_fixture:<name>` with exact repairs. These limits are
minimum corpus evidence; approved real fixtures may and should use tighter measured bounds.

The seam photometric contract distinguishes `identity`, `gain_only_scalar`, `gain_only_rgb`,
`gain_offset_rgb`, vertical `gain_spatial_y_rgb` / `gain_offset_spatial_y_rgb`, and planar
`gain_spatial_xy_rgb` / `gain_offset_spatial_xy_rgb`, plus quadratic
`gain_spatial_quadratic_xy_rgb` / `gain_offset_spatial_quadratic_xy_rgb`. An applied correction
should require
`seam_exposure_held_out_validation_passed: true`; an additive-flare fixture should additionally pin
`seam_exposure_held_out_improvement_over_gain_min` (the built-in automatic floor is 0.006) and a
fixture-appropriate normalized-offset ceiling no larger than the implementation's 0.05 hard bound.
A vertical-shading fixture should pin the selected model, a plausible slope range, at least the
built-in 2/3 slope-agreement floor, and held-out improvement over the best accepted constant model
(automatic floor: max of 0.004 and 10% of that model's score).
A combined vertical gain+offset fixture should additionally bound the normalized offset slope and
both offset endpoints, require at least the built-in 2/3 independently fitted slope agreement,
cap training/held-out center-offset drift at 0.012, and pin improvement over the best simpler model
(automatic floor: max of 0.0025 and 20% of that model's score). A planar fixture can additionally
pin `seam_exposure_spatial_2d_*` acceptance, require at least three columns per split, require the
built-in 2/3 horizontal slope agreement, and pin improvement over the best accepted vertical or
constant model. A quadratic fixture must additionally preserve at least 18 windows, six rows, and
five columns in each whole-cell split; pin
`seam_exposure_spatial_quadratic_gain_accepted` or
`seam_exposure_spatial_quadratic_gain_offset_accepted`;
require independently fitted curvature agreement and bounded full-field disagreement; and inspect
the reported 9x9 support grid. Exact quadratic edge/interior extrema supplement that grid for hard
bounds. The automatic selector rejects an ill-conditioned fit,
curvature below the signal floor, any interior gain outside 0.80–1.25, offsets above 5% of sample
range, or a held-out win below max(0.0025, 20% of the best accepted simpler model's score). These
fields validate overlap-derived compensation only; they do not turn scene overlap into an
independent scanner black/shading calibration, and both axes clamp outside observed support.
For N-input sequences, the compact summary reads every accepted `pair_merges[]` report instead of
silently dropping photometric evidence. A shared model remains named; differing valid models become
`sequence_mixed`. Offset vectors retain the largest absolute per-channel value, support/agreement
floors use conservative minima, and aggregate held-out validation is true only when every
non-identity merge passed. An all-identity sequence correctly reports false because no fitted
correction needed approval.
Seam-detail expectations are separate from photometric-model selection. Pin
`seam_blend_review_required: false` and `seam_detail_review_required: false` for an approved
seam; use `seam_detail_supported_scale_count_min` to require actual signal support and a
fixture-specific `seam_detail_max_symmetric_energy_ratio_max` when an approved scan establishes a
tighter ceiling. The automatic review decision requires both disjoint 6x6 checkerboard partitions
to reproduce a spatially consistent 2x high-pass-energy imbalance on at least two of the 1, 2, and
4 pixel scales. A flagged stitch is still written for diagnosis, but final reviewability is blocked
instead of silently treating focus, grain, or scanner sharpness discontinuity as seamless. Fewer
than two supported scales also block review because the multi-scale decision is under-evidenced.
Grain-detail expectations are independent of the seam-detail contract. A fixture intended to prove
an enabled denoise pass should pin `grain_reduction_enabled: true`, require no detail review, and
require both luminance and chroma probe counts of at least 64 plus fixture-approved p10 floors (the
automatic harmful-loss floor is 0.70). Requiring both counts is stronger than
`grain_detail_decision_supported: true`, which means that at least one channel had enough coherent
structure. A grain-off fixture may instead pin enabled false, review false, and supported true; its
retention ratios are the identity value 1.0 because no destructive operation ran. This no-op state
must not be used as evidence that the enabled algorithm preserves real texture.
Fixture coverage therefore reports `grain_reduction_enabled_fixture_count` separately from
`grain_detail_contract_fixture_count` and
`grain_reduction_effect_contract_fixture_count`. All counts include validation-ready fixtures only.
The detail contract requires an explicit `grain_reduction: on`, nonzero pinned `grain_strength`,
pinned `grain_scale`, enabled/support/no-review expectations, and both-channel probe and p10 floors
at least as strict as the algorithm's 64 / 0.70 decision limits. The still stronger effect contract
requires that complete detail contract plus positive fixture-measured floors for the applied-pixel
ratio, exact structure-excluded ratio, and flat-area luma and chroma p95 residual-reduction ratios.
The exclusion floor prevents a near-universal denoise mask from satisfying the effect contract
merely because it reduces residuals. These are not universal beauty defaults: derive conservative
floors from an approved fixture and retain enough margin for stable regression evidence. Missing
fields are emitted per fixture with
`complete_grain_detail_contract_for_fixture:<name>` or
`complete_grain_reduction_effect_contract_for_fixture:<name>` repair plans.
Dynamic-range expectations form a separate scene-specific contract. Pin a positive
`post_scale_preserved_ratio_min` so colour mapping retains scene-referred values, a positive
`render_luminance_range_p05_p95_min` so robust output contrast does not collapse, and maximum
post-chroma-compression high/low clipping ratios below 1.0. The contract must also explicitly pin
`render_review_status=reviewable`, `render_reviewable=true`,
`tone_output_confidence_status=supported_render_tonal_distribution`,
`tone_output_review_required=false`, a positive `tone_output_evidence_confidence_min`, and a
positive `render_to_mapped_luminance_range_ratio_min`. Thus a stable diagnostic output cannot count
as accepted dynamic-range evidence. These bounds must come from an approved
render of the same scene class: a high-key portrait, fog scene, night frame, and contrasty landscape
should not share an invented universal luminance-span floor. A fixture that declares any one of the
quantitative/tone fields, `render_review_status=reviewable`, or `render_reviewable=true` is marked as
a partial contract until all ten are present with the safe status/boolean values. A lone
`render_reviewable=false` (and optional non-reviewable status) instead declares an intentional
diagnostic delivery and does not pretend to start a dynamic-range acceptance contract. Coverage
reports exact missing
fields and `complete_render_dynamic_range_contract_for_fixture:<name>`. Only complete,
validation-ready cases contribute to `render_dynamic_range_contract_fixture_count`.
The runtime tone-output gate complements this contract with scene-agnostic safety checks: it blocks
near-zero or severe relative p05-p95 collapse and catastrophic per-channel clipping, then propagates
`review_required_tone_output` to final delivery. It intentionally does not certify that every
technically safe render has the preferred scene-specific contrast; fixture bounds remain necessary.
Compact summaries now retain the complete tone-output decision, confidence, ranges, relative
retention, and clipping maxima. Newly written tracked baselines pin those values; older baselines
remain readable and simply do not assert fields they predate.
Stitch normalization has its own complete coverage contract. A counted fixture must not force
no-stitch; it must pin `accepted` or `accepted_sequence`, a known exposure model (including the
`sequence_mixed` summary label), model-appropriate held-out validation, and a normalized offset
ceiling at or below 0.05. It must require `seam_aware_multiband`, no blend/detail review, at least
two supported detail scales, a maximum symmetric energy ratio no greater than 2.0, a seam-gradient
ratio ceiling no greater than 1.05, and an overlap-p95 ceiling below 1.0. Declaring any accepted
stitch or seam-normalization field starts the contract; missing or weak fields produce
`complete_stitch_normalization_contract_for_fixture:<name>`. Only complete, validation-ready cases
contribute to `stitch_normalization_contract_fixture_count`.
For sequence fixtures, `inferred_component_order` is a one-based permutation of every declared
component and must exactly match the order measured in `stitch.ordering.inferred_order`.
The registry may include top-level `coverage_requirements` with minimum fixture, baseline,
calibrated/uncalibrated, scanner-profile, roll-profile, film-stock, scene-tag, exposure-tag, and
calibration-case counts plus reference-fixture/reference-evidence counts, selected calibration
reference-patch counts, debug-artifact expectation counts, matched component, summary-baseline, and
calibration SHA-256 counts, and required tag/value lists. `min_n_component_fixtures` requires real
validation-ready N-way cases, while `min_component_sha256_sets` requires complete matched hashes
across every component in a fixture rather than only its legacy pair. It can also require validation-ready
cross-condition pairs such as `film_stock|calibration_case` and `scene_tag|exposure_tag`, plus
required debug artifact kinds (`candidate_comparison`, `gamut_clipping_map`, and
`scene_referred_prophoto_float`), so a
corpus cannot pass by satisfying film-stock, scene, exposure, and calibration counts only in
isolation. `min_grain_reduction_enabled_fixtures` prevents an all-grain-off corpus from claiming
optional-denoise coverage; `min_grain_detail_contract_fixtures` additionally requires complete
two-channel preservation contracts; and `min_grain_reduction_effect_contract_fixtures` requires
those preservation contracts to demonstrate a nonzero supported effect in both flat-area residual
channels. `min_render_dynamic_range_contract_fixtures` separately requires reviewable final/tone
decisions plus complete confidence, absolute/relative range, preservation, and high/low-clipping
contracts. `min_fixtures` is evaluated against
validation-ready fixtures, not merely declared registry entries.
`min_stitch_normalization_contract_fixtures` separately prevents an accepted-stitch count from
standing in for measured normalization, blending, residual, gradient, and detail-continuity
evidence. `--fixture-coverage --strict`
fails when those requirements are not met.

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
It also separates explicitly enabled denoise fixtures from enabled fixtures with complete,
reproducible luma/opponent-colour detail-retention contracts; inherited CLI settings and grain-off
identity ratios cannot satisfy those corpus gates.
It separately reports `geometry_preparation_contract_fixture_count`,
`geometry_accuracy_contract_fixture_count`, `orientation_accuracy_contract_fixture_count`, and
`negative_reconstruction_contract_fixture_count`. Only complete validation-ready declarations
count; the coverage summary exposes the corresponding minimum requirements, per-fixture booleans,
missing fields, action items, and structured repair plans. The fixture-suite JSON retains nested
`geometry_preparation`, `input_orientation`, and `negative_reconstruction` expected/actual evidence,
including the annotated angle, indexed crop edges, and exact decode transforms/dimensions. The
Markdown table renders the same evidence for review. Structural coverage and runtime success remain
distinct: a complete declaration still fails strict suite execution if measured output misses any
bound.
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
Tone-output fields are optional for backward compatibility, but the baseline writer includes them
whenever the source report supplies current evidence; changes then emit named `tone_output_*`
comparison issues instead of disappearing during summary reduction.
Current baselines likewise retain adaptive-vibrance skin-memory enablement, D50 working space,
protected-pixel ratio, and mean protection weight. Older baselines remain non-asserting, while a
current pinned baseline reports named issues if the protection is disabled, its perceptual space
changes, or its population/effect moves beyond the tone-ratio tolerance.
New baselines also retain the one-way preferred-memory-colour guard's enablement, D50 a*b*
projection space, literature reference, matched/limited ratios, and mean scale reduction. A changed
model provenance or material population/effect drift produces a named comparison issue; older
baselines that never recorded the guard remain readable and non-asserting for it. Full-report strict
validation additionally rejects missing model fields, invalid ellipse geometry, disabled-but-nonzero
effects, `limited > matched > evaluated` contradictions, unexpected/duplicate family sets, and
overall ratios that do not equal the sum of the mutually selected sky/spring-grass/autumn-grass
family ratios.
Current baselines likewise retain the modern-clean preferred-skin shoulder's enablement, D50 a*b*
projection space, preference-study reference, adjusted-pixel ratio, and mean DeltaEab. Named drift
issues expose a changed policy/provenance or material population/effect change, while older
baselines remain readable. Full-report strict validation checks the published Lab/LCh centre and
ellipse geometry, population hierarchy, trusted-colour gating, disabled/zero-effect accounting,
the 3 DeltaEab cap, and the one-way requirement that mean chroma cannot increase.
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
When `render_review` is declared, it must exist, parse, be technically and humanly approved, match
the fixture's ordered inputs, and have a matching `render_review_sha256`; otherwise the fixture is
not validation-ready and receives `complete_render_review_for_fixture` or a narrower hash/file
repair action. `approved_render_review_fixture_count` counts only those fully bound approvals, and
`min_approved_render_review_fixtures` makes visual evidence mandatory instead of accepting a
free-form label or stable baseline as beauty proof.
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
`add_hash_bound_approved_render_reviews:<remaining_count>`,
`add_validation_ready_fixture_for_scene_tag:<tag>`, and
`declare_debug_artifact_expectation_for_validation_ready_fixture:<kind>`. When registered fixtures
are blocked by local assets or metadata, the same list also includes fixture-specific repair actions
such as `provide_component1_for_fixture:<name>`,
`complete_summary_baseline_for_fixture:<name>`,
`provide_calibration_library_for_fixture:<name>`, and
`add_reference_patches_for_fixture:<name>`. Declared review drafts additionally produce
`complete_render_review_for_fixture:<name>` until their decisions and byte bindings pass. TIFF pair
failures include narrower actions such as
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
evidence, explicit `calibration_color_mapping_applied` evidence, and any registry-declared
colour-decision expectations. A validation-ready calibrated reference fixture must require both
`calibration_acceptance_status=accepted` and `calibration_color_mapping_applied=true`; availability
of a calibration record is insufficient. If a fixture declares
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
is missed, required debug artifacts are absent or stale, the primary output
dimensions/storage/ICC or schema-v4 SHA-256 do not match the report, a requested float master or
sRGB proof is missing, malformed, or hash-mismatched, or stale render artifacts are present.
Fixture-suite baselines must also contain
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
rejection, held-out root-polynomial selection inside measured scene support and matrix fallback
outside it, calibrated neutral-regression rejection, automatic unsafe-calibration rejection,
scanner-prior wins on weak image anchors, automatic unsafe scanner-prior rejection, sparse
uncalibrated anchor review, dirty/clipped/border sample rejection, biased scene-anchor review,
evidence-gated neutral rescue for an extreme unstable-anchor cast plus calibrated/scanner-prior/
positive/forced-path protection, high-key exposure headroom normalization, destructive gamut
fallback, reference-patch hue-family
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
captures the fixture-suite contract for required stitch/seam model, planar-model acceptance,
residuals, render, base fallback evidence, colour-candidate, gamut, tone trust, grain-detail
retention, tolerance, and candidate-acceptance evidence. `scanstitch-validate` rejects unknown
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
stitch decision/crop plus seam blend mode/gradient/overlap residuals, base-estimate source, render input source, calibration status/source,
selected colour candidate, calibration acceptance, colorspace strategy, usable-gamut preservation,
tone-output decision/confidence/range/clipping evidence, tone/chroma compression ratios,
high-frequency luma/chroma residuals, and the nested grain-detail
decision/probe/retention evidence when the source report has them. Comparison summaries also
include `comparison.status` and `comparison.issues`. Add `--strict` to make the command exit with
an error when comparison issues are present. Use `--fail-on` for selected issue groups, for example
`--fail-on stale-output,render-input-change,grain`. Supported groups are `any`, `stale-output`,
`dimensions`, `stitch-change`, `base-source`, `render-input-change`, `colorspace-strategy`,
`colorspace-quality`, `quality`, `calibration`, `gamut`, `grain`, `tone`, `debug-artifacts`, and
`summary-baseline`; exact issue names also match. The `tone` selector includes tone-output status,
confidence, range-retention, and clipping regressions as well as chroma-compression changes.
Summary-baseline comparisons populate
`summary_baseline_comparison.status`, `summary_baseline_comparison.issues`, and tracked deltas for
the compact baseline fields, including calibration status/source, scanner/roll profile identity,
requested film stock, film-stock match status, matched roll-profile evidence, explicit calibration
colour-mapping application, selected candidate
rank, raw base proxy-confidence/support-fraction deltas with a dedicated support-fraction
tolerance, technical-safety and
colour-fidelity score split, calibration confidence/matrix-condition/rejection-detail diagnostics,
runner-up decision margin, stable candidate-acceptance signatures (candidate kind, mapping
strategy, rank, status, mode eligibility, selected/rejected state), optional reference-patch
XYZ max, DeltaE76 RMS/max, and CIEDE2000 RMS/max regression evidence, hue-family regression evidence, density,
hue-linearity, saturation, memory-colour, and spatial
cast model evidence, neutral/dominant-anchor support, weak/gamut fallback use, neutral-trim state,
and grain-detail review/support state plus luminance/chroma p10 retention. Grain-detail baseline
regressions emit `grain_detail_review_required_changed`,
`grain_detail_decision_supported_changed`, `grain_detail_luminance_retention_regressed`, or
`grain_detail_chroma_retention_regressed`; `--fail-on grain` selects these as well as residual-p95
regressions.

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
| Report/render | `report.generated_at`, `report.source_report_path`, exact `report.binary_*` identity, `render.output_path`, `render.output_modified_at`, `render.output_width/height`, final save `output_color_space` / `output_encoding` / `output_icc_profile`, compact `render.tone_output_*`, independent `render.output_file_*` TIFF inspection, `render.master_scene_referred_requested/path/file_*`, `render.review_srgb_requested/path/file_*`, `render.review_srgb_gamut_*`, `render.render_review_status`, `render.render_reviewable`, `render.input_base_confidence`, `render.positive_input_likely_negative_like`, `render.positive_input_accepted_high_warm_score`, `render.positive_input_orange_mask_score`, `render.positive_input_reason`, `render.stale_render_artifact_count` |
| Comparison | `comparison.status`, `comparison.issues`, output dimension match, render-review/tone-output decision changes, tone-output confidence/range/retention/clipping deltas, render input changes, and grain p95 ratios |
| Stitch | `decision`, `inferred_order`, `confidence`, `chosen_hypothesis`, `search_selection_reason`, `evidence_score`, `prior_weight`, `overlap_support_score`, `vertical_offset_plausibility_score`, `local_consistency_score`, `plausibility_score`, aggregate `homography_feature_validation` acceptance, match/cell support, forward/reverse cross-inlier ratios, residuals and transform disagreement, aggregate `homography_spatial_validation` acceptance/minimum split and mean improvement/error-reduction/sample metrics, `seam_exposure_correction.model/applied`, center gain, normalized offset, x/y gain and offset slopes, quadratic gain/offset coefficients, vertical endpoints, planar corners, dense-grid curvature bounds, spatial training/held-out row/column counts, condition numbers and agreement, identity/gain/gain+offset/spatial held-out scores, validation/rejection reason, `seam_exposure_correction.seam_score_before/after`, seam clipping ratios, `seam_blend.mode/applied/review_required`, overlap and seam-gradient ratios, aggregate multi-scale detail support/imbalance/review evidence, and aggregate merge counts for N-input sequences |
| Base/density | `base_confidence`, `raw_base_confidence`, `raw_base_proxy_confidence`, `raw_base_support_fraction`, `density_confidence` (the weakest film-base/negative-response evidence in current reports), `base_estimate_source` |
| Colorspace | `calibration_record_status`, `calibration_source`, explicit `calibration_color_mapping_evaluated` / `applied` / `status` / candidate / reason and cross-field `calibration_color_mapping_consistency_issues`, scanner/roll/film-stock diagnostics in the full report, full-report selected-mapping versus negative-reconstruction confidence/status/limiter provenance, `render_input_source`, `render_input_reason`, `mapping_strategy`, `selected_candidate`, `selected_candidate_rank`, `selected_quality_score`, `technical_safety_score`, `color_fidelity_score`, `selected_quality_components`, `candidate_risk`, `tone_color_trust_state`, `neutral_safety_rescue` admission/effect/reason evidence, `selected_runner_up_quality_delta`, `candidate_quality_scores`, candidate `rendered_tone_quality` and `color_model_quality`, rendered-tone, tone-chroma-cleanup, density-monotonicity, hue-linearity, saturation-preservation, memory-colour, and spatial-consistency score components, `candidate_acceptance`, `calibration_acceptance`, `neutral_estimate_quality`, `neutral_sample_rejections`, `dominant_anchor_sample_rejections`, `dominant_anchor_quality`, optional `reference_patch_evaluation` with XYZ, D50 Lab DeltaE76, and CIEDE2000 residuals, `neutral_trim_before_after`, `neutral_trim_applied`, `neutral_trim_scale`, `scene_referred_detail_fusion`, `exposure_scale`, `post_scale_preserved_ratio`, calibrated-vs-image exposure/clipping deltas, `regularization_lambda`, `channel_anchor_counts`, `channel_anchor_low_support`, `dominant_anchor_bands`, `weak_anchor_fallback_used`, `gamut_fallback_used`, image-matrix low/high clipping, debug artifact paths for `color_candidate_comparison_artifact`, `gamut_clipping_map_artifact`, and `scene_referred_prophoto_float_artifact`, `debug_artifacts` existence/freshness inspection, plus compact gamut-map and scene-referred float artifact diagnostics |
| White balance | Technical requested mode/status/source/reason, applied/confidence/review state, neutral sample/spatial/luminance support, estimated source CCT, pre/post neutral log-chroma, plus creative temperature/tint and proof that the creative transform is excluded from the technical master |
| Tone | `tone_confidence_status`, `tone_output_confidence_status`, tone-output review/limiter evidence, input/fitted/rendered p05-p95 ranges and relative retention, maximum post-tone channel clipping; tone/chroma compression ratios; `color_protection_policy`, `color_trust_state`, `color_protection_reason`; shadow, midtone, midtone-neutral, bright-neutral, and bright-saturated saturation medians/p95 values; `local_luminance_detail_*`; `adaptive_vibrance_*`; `noise_reduction_*`; compact `grain_detail_retention` support, probe counts, median/p10 retention, decision and review reason (full report: `grain_reduction.detail_retention`); all-sample and flat-area `high_frequency_grain.*_residual_p95` values |

## Interpreting Deltas

Treat these as review triggers:

| Field | Review when |
|-|-|
| `stitch.decision` | It changes between accepted, rejected, and skipped. |
| `stitch.confidence` | It drops materially or the chosen hypothesis changes. |
| `stitch.native_affine.rotation_search_boundary_hit` | It is `true`. A bounded rotation search that ends at its limit has not demonstrated an interior optimum and must fall back instead of extrapolating a full-union warp. |
| `report.generated_at` / `render.output_modified_at` | A report or TIFF is older than the render you intended to inspect. |
| `render.stale_render_artifact_count` | It is non-zero before a visual comparison that uses debug/intermediate TIFFs. |
| `colorspace.debug_artifacts` | A named color candidate preview or gamut clipping map is `missing`, `stale`, `not_file`, or has unknown freshness before review. Strict summary-baseline validation reports this as `colorspace_debug_artifact_invalid:<kind>:<status>`. |
| `render.output_file_icc_profile_matches_report` | It is `false`, meaning the saved TIFF profile tag does not match the report's claimed output profile. |
| `render.output_file_dimensions_match_report` / `output_file_storage_matches_report` | Either is not `true`; the saved TIFF does not independently prove the dimensions and RGB16 unsigned encoding claimed by the report. |
| `render.artifact_sha256_binding_required` | `true` for current schema-v4 reports. A current report with this false indicates a schema/version inconsistency; legacy schema-v3-or-earlier summaries remain structurally inspectable without fabricating hash evidence. |
| `render.output_file_sha256_matches_report` | It is not `true` for a schema-v4 report; the exact reopened TIFF bytes do not match the digest declared by the save phase. |
| `render.master_scene_referred_file_matches_report` / `review_srgb_file_matches_report` | A requested auxiliary artifact is not `true`; its file is missing or does not independently match the promised dimensions, storage, description/profile, and colour space. |
| `render.master_scene_referred_file_sha256_matches_report` / `review_srgb_file_sha256_matches_report` | A requested schema-v4 artifact is not `true`; its exact reopened bytes do not match the report. This rejects a valid same-shape/profile substitution that structural checks alone cannot distinguish. |
| `render.review_srgb_gamut_mapping_supported` | It is not `true` for a requested proof; the report lacks the constant-lightness/hue D50 Lab-to-sRGB policy, coherent bounded scale/ratio evidence, or zero non-finite/post-map out-of-gamut counts. |
| `render.render_review_status` | It is anything other than `reviewable`. In particular, `blocked_low_base_confidence` means density inversion used a fallback-quality base estimate, while `review_required_tone_output` means the final luminance distribution collapsed or a channel clipped catastrophically. The TIFF remains diagnostic output rather than an approved render. |
| `tone.tone_output_confidence_status` / `tone_output_review_required` | Status changes, a new review requirement, reduced evidence confidence/range retention, or increased clipping produce named comparison issues. Identical current evidence remains comparable, and missing legacy baseline fields are non-asserting rather than fabricated passes. |
| `render.positive_input_likely_negative_like` | It is `true` in positive mode, meaning the already-positive path was requested for input whose channel medians look like orange-mask negative film. Use negative mode or replace the roll with actual positive scans before treating colour/tone output as a positive regression. |
| `render.positive_input_accepted_high_warm_score` | It is `true` in positive mode when the aggregate warm-channel score is high but R/G and G/B ratios are mild enough to accept the frame as already-positive input. Use it to distinguish warm positive scenes from negative-like warnings. |
| `stitch.seam_exposure_correction.model` / `applied` | It flips unexpectedly, or a flexible constant, vertical, planar, or quadratic gain+offset correction appears without the fixture requiring it. |
| `stitch.seam_exposure_correction.held_out_validation_passed` | An applied correction lacks disjoint spatial validation. For `gain_offset_rgb`, require a material held-out improvement over gain-only as well as identity. For `gain_spatial_y_rgb`, require distinct-row support, slope agreement, and material improvement over the best accepted constant model. |
| `stitch.seam_exposure_correction.seam_score_after` | It fails to improve relative to `seam_score_before`, normalized offsets exceed the fixture ceiling, or clipping ratios rise materially. |
| `stitch.homography_feature_validation` | A homography lacks both whole-cell partitions, forward or reverse cross-inlier support falls, residuals increase, the independently fitted transforms disagree, or a rejected attempt becomes accepted. This is the disjoint feature-fitting gate; composition uses only the training-cell transform. |
| `stitch.homography_spatial_validation` | A feature-validated homography fails either image-domain spatial partition, falls below the mean improvement/error-reduction gates, lacks sample support, or becomes indistinguishable from translation. This is additional model-selection evidence, not a reuse of feature residuals. |
| `stitch.seam_blend.applied` / `mode` | An accepted translation, affine, or homography stitch lacks the expected multiband blend, or a sequence contains an unblended merge. |
| `stitch.seam_blend.output_to_source_seam_gradient_ratio` | It exceeds the fixture-specific ceiling or rises above 1, indicating that composition amplified rather than reduced the measured seam discontinuity. |
| `stitch.seam_blend.overlap_p95_abs_difference` | It exceeds the pinned fixture ceiling, indicating exposure, colour, shading, or geometric normalization no longer matches the approved overlap. |
| `stitch.seam_blend.detail_consistency` / `review_required` | Both disjoint spatial partitions reproduce a severe multi-scale detail-energy mismatch, supported scales disappear, the maximum symmetric ratio regresses, or an accepted stitch becomes geometry-review blocked. This measures a possible focus/grain/texture transition; it does not authorize automatic sharpening or blur. |
| `base_density.base_confidence` | It drops enough to cap downstream confidence. |
| `colorspace.render_input_source` | It switches between ICA, direct-density, and positive RGB render input. |
| `colorspace.calibration_record_status` | It changes between `not_configured`, `applied`, and `rejected`; this is record/component state, not proof that its colour mapping was selected. |
| `colorspace.calibration_color_mapping_*` | `applied` or `selection_status` changes, selected/preferred candidates diverge from the final candidate decision, or `calibration_color_mapping_consistency_issues` is not `none`. Strict validation rejects an internal contradiction. |
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
| `colorspace.neutral_safety_rescue` | `applied` changes, the preserved-gamut gain or midtone-saturation reduction falls below the fixture floor, or the reason changes. An applied rescue is an intentional safety improvement but must remain `fallback_only` and review-required; it never validates colour accuracy. |
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
| `tone.preferred_skin_rendering` | The trusted modern-clean one-way shoulder changes model/reference/space, adjusts a materially different population, exceeds its 3 DeltaEab cap, reports positive mean chroma change, violates `adjusted <= outside <= matched <= evaluated`, or becomes nonzero while disabled. It is an aggregate display proxy, not semantic skin detection or calibration evidence. |
| `tone.noise_reduction_*` | Requested on/off, strength, or scale differs from the intended fixture; effective radius/amount changes unexpectedly; an enabled request has no measurable flat-area effect; or protection starts applying strongly to high-texture/saturated pixels. Off is the normal default. |
| `tone.grain_detail_retention` | An enabled pass lacks the fixture's required coherent luminance/opponent-colour probe support, supported median retention falls below 0.90, supported p10 retention falls below 0.70, the baseline review/support state changes, or final status becomes `review_required_grain_detail`. Unsupported evidence is not a preservation pass; inspect a suitable detail-bearing fixture. |
| `tone.post_tone_buffer_policy` / `scene_referred_master_preserved` | The render stops reusing one owned post-tone buffer, or an optimization mutates the cached scene-referred master. |
| `tone.high_frequency_grain.*_residual_p95` | Luma or chroma residuals change materially between reports that should be visually equivalent. |

The validation harness is a repeatability check, not a replacement for visual review. For color, tone, or colorspace changes, rerun LOGAN and inspect the local render before accepting the summary delta.

# Color Calibration

ScanStitch can use a local calibration library before falling back to the image-derived neutral and
dominant-channel anchor estimate. The library separates stable scanner/settings characterization
from optional per-roll film/development correction. The older one-file scanner/film profile path is
still supported for compatibility.

Default rendering remains automatic:

```powershell
cargo run --bin scanstitch -- LOGAN043.tif LOGAN044.tif -o output/logan
```

Run with a local calibration profile:

```powershell
cargo run --bin scanstitch -- LOGAN043.tif LOGAN044.tif `
  -o output/logan-calibrated `
  --calibration-profile path/to/scanner-film-profile.json
```

Run with the calibration library:

```powershell
cargo run --bin scanstitch -- LOGAN043.tif LOGAN044.tif `
  -o output/logan-calibrated `
  --calibration-library calibration `
  --scanner-profile coolscan-4000-vuescan-raw `
  --roll-profile logan-roll-2026-05 `
  --film-stock "Kodak Gold 200"
```

`--roll-profile` is optional. With only `--scanner-profile`, ScanStitch uses the scanner matrix as
the stable prior and still lets image-derived neutral/dominant anchors adapt the roll. With scanner
and roll matrix profiles selected, it composes the roll correction after the scanner transform.
`--film-stock` is optional audit evidence: it ranks film hints and roll candidates, and an explicit
roll profile is rejected if its film metadata does not match the requested stock.

The validation wrapper forwards the same profile:

```powershell
cargo run --bin scanstitch-validate -- `
  --fixture logan `
  --output-dir output/validation/logan-calibrated `
  --calibration-library calibration `
  --scanner-profile coolscan-4000-vuescan-raw `
  --roll-profile logan-roll-2026-05 `
  --film-stock "Kodak Gold 200"
```

## Library Format

A calibration library is a directory of JSON files. Files may be grouped under `scanners/`,
`rolls/`, and `film_hints/`; invalid entries are reported but do not stop rendering.
The machine-readable single-record schema is
[`color-calibration-library-record.schema.json`](color-calibration-library-record.schema.json), with
scanner, roll, and film-hint examples in
[`color-calibration-library-record.examples.json`](color-calibration-library-record.examples.json).

Scanner profile:

```json
{
  "schema_version": 2,
  "record_type": "scanner_profile",
  "profile_id": "coolscan-4000-vuescan-raw",
  "source_space": { "name": "linear scanner RGB", "encoding": "linear" },
  "scanner": { "make": "Nikon", "model": "CoolScan 4000" },
  "settings": { "software": "VueScan", "mode": "raw", "bit_depth": 14 },
  "response_curves": { "red": [[0.0, 0.0], [1.0, 1.0]] },
  "flare_black_white_diagnostics": {
    "black_level": [16.0, 17.0, 18.0],
    "white_level": [65535.0, 65535.0, 65535.0],
    "flare_check": "passed"
  },
  "target": { "type": "transmissive_target", "illuminant": "D50" },
  "reference": { "dataset": "target-reference" },
  "whitepoint": [0.9642, 1.0, 0.8251],
  "scanner_rgb_to_xyz": [[0.7977, 0.1352, 0.0313], [0.2880, 0.7119, 0.0001], [0.0, 0.0, 0.8251]],
  "fit": {
    "method": "least_squares_rgb_to_xyz",
    "patch_count": 24,
    "target_residual_rms": 0.006,
    "target_residual_max": 0.018,
    "per_hue_residuals": [
      { "hue_family": "red", "patch_count": 3, "residual_rms": 0.009, "residual_max": 0.018 }
    ],
    "worst_patches": [
      {
        "patch_id": "A03",
        "hue_family": "red",
        "source_rgb": [0.75, 0.15, 0.12],
        "reference_xyz": [0.619, 0.323, 0.099],
        "fitted_xyz": [0.617, 0.326, 0.101],
        "residual_xyz": [-0.002, 0.003, 0.002],
        "residual_error": 0.0044,
        "max_channel_error": 0.0032
      }
    ]
  },
  "delta_e00_summary": { "rms": 1.2, "max": 3.8, "patch_count": 24 },
  "scanner_settings_fingerprint": "fnv1a64:0123456789abcdef",
  "confidence": 0.95
}
```

Roll profile:

```json
{
  "schema_version": 2,
  "record_type": "roll_profile",
  "profile_id": "logan-roll-2026-05",
  "scanner_profile_id": "coolscan-4000-vuescan-raw",
  "film": { "stock": "Kodak Gold 200", "process": "C-41" },
  "development": { "developer": "manual tank" },
  "base_color": [12000.0, 7000.0, 3000.0],
  "correction_domain": "xyz_post_scanner",
  "correction_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
  "fit": {
    "method": "least_squares_xyz_post_scanner_correction",
    "patch_count": 12,
    "target_residual_rms": 0.012,
    "target_residual_max": 0.031
  },
  "hue_family_residuals": {
    "neutral": { "rms": 0.9, "max": 2.1 },
    "skin": { "rms": 1.8, "max": 4.2 }
  },
  "delta_e00_summary": { "rms": 2.0, "max": 5.4, "patch_count": 12 },
  "scanner_settings_fingerprint": "fnv1a64:0123456789abcdef",
  "confidence": 0.85
}
```

`base_color` may be stored without a correction matrix when no target frame exists. A roll
`correction_matrix` is applied only when the roll is selected explicitly and its
`scanner_profile_id` and `scanner_settings_fingerprint` match the selected scanner profile.
Schema v2 records may also carry response curves, black/white/flare diagnostics, polynomial or 3D
LUT candidates, DeltaE00 summaries, and hue-family residuals. The current renderer records these
as calibration evidence and still applies the matrix path deterministically unless a roll matrix is
available.

Film hint:

```json
{
  "schema_version": 1,
  "record_type": "film_hint",
  "film_id": "kodak-gold-200",
  "stock": "Kodak Gold 200",
  "aliases": ["Gold 200"],
  "similarity_tags": ["c41", "consumer"]
}
```

Auto-matching is conservative. A scanner profile is auto-applied only when exactly one valid
profile has `"auto_match": true` and confidence at or above the high auto-match threshold. Weak or
ambiguous matches are reported as advisory. Roll profiles and film hints are advisory unless
selected explicitly. Film-stock input does not silently apply a roll correction; it records
`film_stock` diagnostics, sorts `nearest_film_candidates` and `nearest_roll_candidates`, and blocks
a selected roll correction when `roll_profile.film.stock` or equivalent film metadata conflicts.

## Ingest Utility

`scanstitch-calibrate` creates library records:

```powershell
cargo run --bin scanstitch-calibrate -- scanner-target `
  --library calibration `
  --profile-id coolscan-4000-vuescan-raw `
  --measurements scanner-target-fit.json

cargo run --bin scanstitch-calibrate -- roll-target `
  --library calibration `
  --profile-id logan-roll-2026-05 `
  --scanner-profile coolscan-4000-vuescan-raw `
  --measurements roll-target-fit.json

cargo run --bin scanstitch-calibrate -- roll-base `
  --library calibration `
  --profile-id logan-roll-base-only `
  --scanner-profile coolscan-4000-vuescan-raw `
  --film-stock "Kodak Gold 200" `
  --process C-41 `
  --base-color 12000,7000,3000
```

The ingest utility refuses to replace an existing record by default. Re-run with `--force` only when
you intentionally want the same `profile_id` to overwrite the prior scanner or roll record.
Before writing, the utility validates the exact candidate record against the calibration-library
loader contract; records that would later be rejected are reported and left unwritten.

For `scanner-target` and `roll-target`, `--measurements` can be either a pre-fit record containing
`scanner_rgb_to_xyz`/`correction_matrix` or a patch-measurement object. Patch measurements use
`patches[]` with scanner/source RGB plus reference XYZ or D50 Lab values:

```json
{
  "scanner": { "make": "Nikon", "model": "CoolScan 4000" },
  "settings": { "software": "VueScan", "mode": "raw", "bit_depth": 14 },
  "target": { "type": "transmissive_target", "illuminant": "D50" },
  "reference": { "dataset": "target-reference" },
  "patches": [
    { "id": "A01", "rgb": [0.81, 0.78, 0.70], "xyz": [0.760, 0.780, 0.650] },
    { "id": "A02", "rgb": [0.42, 0.36, 0.30], "lab": [62.1, 1.2, 5.4] }
  ]
}
```

`roll-target` uses the selected `--scanner-profile` to transform each patch RGB through the scanner
matrix first, then fits a small `xyz_post_scanner` correction. The roll correction is written with
the selected scanner settings fingerprint and will be rejected later if selected with different
scanner settings. Pre-fit `correction_matrix` records are also stamped with the selected scanner
fingerprint before writing; an existing mismatched `scanner_settings_fingerprint` is rejected.

When `scanstitch-calibrate` fits a target, the written `fit` object includes aggregate RMS/max
residuals plus `per_hue_residuals` and `worst_patches`. Use those lists to catch hue-localized
target problems before relying on the profile in automatic color mapping.

## Compatibility Profile Format

Profiles are JSON. The schema lives at `docs/color-calibration-profile.schema.json`, and a
synthetic example lives at `docs/color-calibration-profile.example.json`.

Required profile data:

| Field | Meaning |
|-|-|
| `schema_version` | `1` or `2`; v2 can record complete perfect-mode scanner and roll evidence. |
| `source_space` | Scanner/source RGB metadata. |
| `scanner`, `film` | Scanner and film identifiers for matching the profile to the scan setup. |
| `target`, `reference` | Calibration target and reference dataset metadata. |
| `whitepoint` | Source white in XYZ, normalized around Y = 1. |
| `work_to_xyz` or `fit_matrix` | Row-major 3x3 source RGB to XYZ fit matrix. |
| `confidence` | Fit confidence in `[0, 1]`; values below the implementation threshold are rejected. |
| `gamut_limits` | Optional profile-domain limits. |

The loader validates finite matrix values, invertibility, condition number, whitepoint sanity,
schema version, and confidence. Invalid profiles do not fail the render; they are rejected and the
pipeline uses the existing image-derived color estimate.

## Report Fields

Every render records calibration state under `colorspace_mapping.metrics.calibration`:

```json
{
  "status": "applied",
  "source": "external_calibration_profile",
  "external_profile": {
    "path": "path/to/scanner-film-profile.json",
    "profile_id": "scanner-profile-id"
  },
  "profile_schema_version": 2,
  "confidence": 0.95,
  "reason": "valid external calibration profile selected over image-derived estimate",
  "matrix_condition_number": 2.1,
  "whitepoint": [0.9642, 1.0, 0.8251],
  "rejection_details": []
}
```

Library runs also include `library`, `scanner_profile`, `roll_profile`, `requested_film_stock`,
`film_stock`, `nearest_roll_candidates`, and `nearest_film_candidates`. These report applied,
rejected, and advisory selections, confidence, matrix condition numbers, scanner settings, scanner
settings fingerprints, target fit method, patch count, aggregate/per-hue residual errors, worst
patches, DeltaE00 summaries, scanner response/flare diagnostics, higher-order transform candidates,
base-color deltas, requested film-stock matches/mismatches, invalid library entries, and whether a
roll correction matrix was actually applied. When v1 evidence is used in a perfect-mode-oriented
run, `calibration_upgrade_available` explains which v2 evidence is missing.

Possible `status` values:

| Status | Meaning |
|-|-|
| `not_configured` | No applicable profile was passed, selected, or confidently auto-matched; image-derived mapping is used. |
| `applied` | A valid one-off profile or calibration-library scanner/roll profile affected color mapping. |
| `rejected` | An explicitly selected profile failed validation or matching; image-derived mapping is used. |

The same phase also records `candidate_quality_scores`, `candidate_acceptance`,
`selected_candidate_rank`, `selected_quality_score`, `technical_safety_score`,
`color_fidelity_score`, `quality_components`, `color_decision_summary`, `candidate_risk`,
`tone_color_trust_state`, `selected_runner_up_quality_delta`,
`calibration_acceptance`, `neutral_estimate_quality`, `neutral_sample_rejections`,
`dominant_anchor_sample_rejections`, `dominant_anchor_quality`,
candidate `rendered_tone_quality` / `color_model_quality`,
`neutral_trim_before_after`, optional XYZ/DeltaE
`reference_patch_evaluation`, `candidate_comparison`, and `usable_colourspace` diagnostics.
`candidate_scores` remains as a compatibility alias. Scores are ordered `lower_is_better`:
technical safety covers gamut, clipping, preserved pixels, exposure, matrix condition, and fallback
pressure; color fidelity covers neutral balance/support, target residuals, profile confidence,
anchor support/stability, rendered-tone colour, tone-map chroma cleanup risk, density monotonicity,
hue linearity, saturation preservation, memory-colour plausibility, and spatial neutral-cast
consistency.

In `--color-mode auto`, a calibrated or scanner-prior candidate must stay inside the negative-gamut
safety limits, beat the image-derived candidate on `quality_score`, and not regress neutral or
reference-patch metrics versus image-derived mapping before it is selected. If it does not,
`calibration_acceptance.status` records `rejected_quality`, `rejected_neutral`, or
`rejected_reference_fit`; if it exceeds the safety limits, it records `rejected_unsafe`. Forced
`--color-mode calibrated` still fails clearly when the only calibrated candidates are unsafe.

`neutral_estimate_quality` reports whether neutral samples were broad enough across
shadow/midtone/highlight bands; `neutral_sample_rejections` and
`dominant_anchor_sample_rejections` count clipped, border, film-base-like edge, dust, luma, and
chroma/low-saturation or weak-dominance rejections before neutral sampling and matrix-anchor
fitting. Dominant anchors are averaged with per-luminance-band caps so one bright or dark anchor
cluster cannot dominate the fit; `dominant_anchor_quality` records unstable channels when anchors
are too sparse, single-band, luma-concentrated, or only barely dominant. Neutral trim is applied only
when that estimate is accepted and
`neutral_trim_before_after` shows a measured aggregate neutral-delta reduction without worsening any
populated shadow/midtone/highlight band beyond tolerance, increasing low/high clipping, or reducing
preserved gamut. Tone mapping remains luminance-based; its `color_protection_policy` reports
whether highlight/shadow chroma cleanup is `enabled`, disabled only for weak neutral support, or
disabled because `candidate_risk` requires color review.

## Acceptance Workflow

1. Capture the target with the same scanner, film stock, processing, and scan settings as the
   negatives to render.
2. Fit a source RGB to XYZ matrix and write a schema v2 profile when response/black/white/target
   diagnostics are available; v1 records remain readable.
3. Run `scanstitch-validate` without the profile into a fresh output directory.
4. Run `scanstitch-validate` with `--calibration-profile` into a second fresh output directory.
5. Compare the summaries and inspect `colorspace.calibration_status`,
   `colorspace.mapping_strategy`, `colorspace.selected_candidate`,
   `colorspace.selected_quality_score`, `colorspace.calibration_acceptance`,
   `colorspace.neutral_estimate_quality`, `colorspace.neutral_trim_before_after`,
   `colorspace.post_scale_preserved_ratio`, `colorspace.calibrated_profile_*`, and
   `colorspace.image_matrix_*`.
6. Accept the profile only when it is selected or explicitly rejected for understandable reasons,
   lowers or preserves the color quality score, improves neutrality or usable gamut, and does not
   increase clipping enough to trigger review issues.

Hand grading, creative looks, ICC import, and LUT application are out of scope for this phase.

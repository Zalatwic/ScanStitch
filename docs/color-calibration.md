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
  "scanner_linearization": {
    "model_id": "coolscan-4000-linearization-v1",
    "curves": {
      "red": [[0.0, 0.0], [0.25, 0.0625], [0.5, 0.25], [0.75, 0.5625], [1.0, 1.0]],
      "green": [[0.0, 0.0], [0.25, 0.0625], [0.5, 0.25], [0.75, 0.5625], [1.0, 1.0]],
      "blue": [[0.0, 0.0], [0.25, 0.0625], [0.5, 0.25], [0.75, 0.5625], [1.0, 1.0]]
    },
    "black_level_normalized": [0.01, 0.01, 0.01],
    "white_level_normalized": [0.99, 0.99, 0.99],
    "additive_flare_normalized": [0.002, 0.002, 0.002],
    "shading_gain_polynomial": [
      [1.0, 0.01, 0.0, 0.0, 0.0, 0.0],
      [1.0, 0.01, 0.0, 0.0, 0.0, 0.0],
      [1.0, 0.01, 0.0, 0.0, 0.0, 0.0]
    ],
    "confidence": 0.95,
    "validation": {
      "held_out_sample_count": 24,
      "transmittance_rmse": 0.002,
      "transmittance_max_error": 0.008,
      "identity_baseline_rmse": 0.08
    }
  },
  "flare_black_white_diagnostics": {
    "black_level": [16.0, 17.0, 18.0],
    "white_level": [65535.0, 65535.0, 65535.0],
    "flare_check": "passed"
  },
  "target": { "type": "transmissive_target", "illuminant": "D50" },
  "reference": { "dataset": "target-reference" },
  "whitepoint": [0.9642, 1.0, 0.8251],
  "scanner_rgb_to_xyz": [[0.7977, 0.1352, 0.0313], [0.2880, 0.7119, 0.0001], [0.0, 0.0, 0.8251]],
  "target_patch_signal_domain": "scanner_linearized_transmittance",
  "target_patch_linearization_model_id": "coolscan-4000-linearization-v1",
  "fit": {
    "method": "least_squares_rgb_to_xyz",
    "patch_count": 24,
    "target_residual_rms": 0.006,
    "target_residual_max": 0.018,
    "validation": {
      "evaluation_set": "held_out",
      "training_patch_count": 24,
      "held_out_patch_count": 24,
      "training_residual_rms": 0.004,
      "training_residual_max": 0.012,
      "identity_baseline_residual_rms": 0.120,
      "identity_baseline_residual_max": 0.250,
      "held_out_identity_improvement_fraction": 0.950
    },
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
  "target_patch_signal_domain": "scanner_linearized_transmittance",
  "target_patch_linearization_model_id": "coolscan-4000-linearization-v1",
  "negative_response": {
    "model_id": "logan-roll-density-response-v1",
    "scanner_density_to_layer_density": [
      [1.0, -0.08, 0.01],
      [-0.04, 1.0, -0.05],
      [0.01, -0.04, 1.0]
    ],
    "characteristic_curves": {
      "red": [[0.0, 0.0], [0.25, 0.25], [0.70, 0.75], [1.30, 1.40]],
      "green": [[0.0, 0.0], [0.30, 0.25], [0.85, 0.75], [1.45, 1.40]],
      "blue": [[0.0, 0.0], [0.20, 0.25], [0.62, 0.75], [1.18, 1.40]]
    },
    "white_anchor_percentile": 0.995,
    "confidence": 0.94,
    "validation": {
      "held_out_patch_count": 24,
      "delta_e00_rms": 1.8,
      "delta_e00_max": 4.9,
      "unit_slope_delta_e00_rms": 8.2,
      "worst_hue_family": "deep-blue"
    }
  },
  "fit": {
    "method": "least_squares_xyz_post_scanner_correction",
    "patch_count": 12,
    "target_residual_rms": 0.012,
    "target_residual_max": 0.031,
    "validation": {
      "evaluation_set": "held_out",
      "training_patch_count": 18,
      "held_out_patch_count": 12,
      "training_residual_rms": 0.009,
      "training_residual_max": 0.024,
      "identity_baseline_residual_rms": 0.050,
      "identity_baseline_residual_max": 0.120,
      "held_out_identity_improvement_fraction": 0.760
    }
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

`negative_response` is the measured negative-film reconstruction path. Its 3×3 matrix maps
base-subtracted scanner optical density into separated film-layer density. Each curve then maps
`[separated_layer_density, scene_log_exposure]`; ScanStitch evaluates the curve with monotone
piecewise-cubic Hermite interpolation and preserves signed highlight headroom. The model is applied
only when all of the following hold:

- the roll and scanner profile/settings fingerprints match exactly;
- every curve has at least four finite, strictly increasing measured points;
- the separation matrix condition number is at most 20 and combined curve/matrix noise gain is at
  most 4;
- confidence is at least 0.75; and
- at least 12 held-out patches achieve DeltaE00 RMS ≤ 6, maximum DeltaE00 ≤ 15, and improve RMS over
  the unit-slope baseline by at least 0.25.

Schema v2 scanner records may also carry legacy `response_curves`, black/white/flare diagnostics,
DeltaE00 summaries, and hue-family residuals. The typed `scanner_linearization` path is applied
independently to each full decoded scan before border detection and stitching, with EXIF-oriented
pixels inverse-mapped to original scanner coordinates for spatial gains. A typed `color_model` is an
executable, held-out-selected root-polynomial RGB-to-D50-XYZ transform; `polynomial_fit` is its full
selection audit. A typed `lut_3d_model` is an executable, smooth residual 5x5x5 or 7x7x7
tetrahedral lattice over the selected polynomial or matrix baseline; `lut_3d_fit` records every LUT
candidate and rejection. Legacy opaque polynomial records and `lut_3d` remain audit evidence only.
The report identifies whether each typed nonlinear candidate was evaluated, selected, or rejected
in favour of a simpler calibrated fallback. The roll `negative_response` path above is likewise
applied and fully identified.

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
cargo run --bin scanstitch-calibrate -- sample-target `
  --manifest chart-sampling.json `
  --output scanner-target-fit.json `
  --preview-dir output/chart-sampling

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

`sample-target` removes the manual RGB-extraction step. Its manifest contract is
[`calibration-target-sampling.schema.json`](calibration-target-sampling.schema.json), with a
12-patch illustrative manifest in
[`calibration-target-sampling.example.json`](calibration-target-sampling.example.json). Replace the
example reference values, image paths, and corners with measurements for the actual chart. Twelve
patches per split can qualify the matrix fit only; the higher-order model needs the larger counts
documented below.

The manifest must identify at least one `training` and one independently captured `held_out`
TIFF/DNG. Both complete files and decoded oriented pixel payloads are SHA-256 hashed; duplicates are
rejected even when paths, IDs, or container metadata differ. `scanner_signal_bit_depth` declares the effective code range: set it to `14` when a
scanner stores 14-bit values in a TIFF whose container advertises 16 bits. The range transform and
observed source/working extrema are recorded per capture. For each capture, enter the top-left,
top-right, bottom-right, and bottom-left outer corners of the rectangular patch-grid footprint (not
the backing board) in decoded, EXIF-oriented pixel coordinates. A projective transform maps the declared grid
into the scan; only the inset interior of each patch is sampled. The sampler preserves the decoded
source precision, uses bilinear samples on a uniform chart-space lattice, rejects clipped pixels and
multichannel MAD outliers, computes a Tukey-biweight RGB mean, and checks p95-p05 spread plus 3x3
spatial-cell consistency. Patch centres are inverse-mapped through EXIF orientation to produce
`scanner_xy` in the original scanner frame.

Target geometry and reference values are explicit human-reviewed evidence contracts. An initial
manifest may omit `corner_review` and `reference_review`. That first run exits nonzero but writes one overlay PNG per capture,
`sampling-report.json`, and `corner-review-manifest.json`. The generated manifest replaces
relative image paths with canonical paths and pins each capture's orientation-materialized decoded
pixels, dimensions, channel count, and working precision through `decoded_pixel_sha256`; it also
pins the ordered top-left/top-right/bottom-right/bottom-left coordinates through
`chart_corners_sha256` and the exact rendered RGB review artifact through
`overlay_pixel_sha256`. Inspect
that the yellow outer chart polygon and every green patch interior align with the photographed
patch-grid footprint. Also verify every ordered patch ID, row/column position, and XYZ/Lab value
against the named factual target source; `reference_patch_sha256` binds that canonicalized set.
Only then set `corner_review.approved` to `true` for each capture and
`reference_review.approved` to `true`, and rerun using the generated manifest. Do not copy a hash
from a different scan or dataset, approve an overlay that misses the chart geometry, or treat the
example manifest's deliberately zero reference values as colour truth.

An accepted run requires every geometry/reference approval and hash match, zero rejected patches, and sufficient
training/held-out counts; it writes the measurement JSON plus refreshed overlays, report, and
review manifest. Missing approval produces `status=requires_corner_approval`; changing a capture's
pixels, orientation, or declared working precision produces
`corner_review.status=decoded_pixel_mismatch`, while changing the declared footprint produces
`corner_review.status=chart_corners_mismatch`. Changing the grid, inset, preview dimensions, patch
status, or overlay renderer without reapproval produces
`corner_review.status=overlay_pixel_mismatch`. None of these states writes a measurement file that could
accidentally be fitted, and retained calibration records independently require every corner review
to remain approved and hash-consistent. Missing reference approval produces
`status=requires_reference_approval` (or `requires_reference_and_corner_approval`); changed patch
identity, position, or XYZ/Lab values produce
`reference_review.status=reference_patch_mismatch`. Retained records preserve the reference patch
set and recompute its digest. Patch-quality rejection likewise writes only diagnostic
outputs. Chart corners are still entered manually; automatic target detection and an interactive
corner editor remain future acquisition improvements.

The ingest utility refuses to replace an existing record by default. Re-run with `--force` only when
you intentionally want the same `profile_id` to overwrite the prior scanner or roll record.
Before writing, the utility validates the exact candidate record against the calibration-library
loader contract; records that would later be rejected are reported and left unwritten.

For `scanner-target` and `roll-target`, `--measurements` can be either a pre-fit record containing
`scanner_rgb_to_xyz`/`correction_matrix` or a patch-measurement object. A matrix fit requires separate
`training_patches[]` and `held_out_patches[]` lists. Each item needs a globally unique sample `id`,
scanner/source RGB, and reference XYZ or D50 Lab. At least 12 samples are required in each list:

```json
{
  "scanner": { "make": "Nikon", "model": "CoolScan 4000" },
  "settings": { "software": "VueScan", "mode": "raw", "bit_depth": 14 },
  "target": { "type": "transmissive_target", "illuminant": "D50" },
  "reference": { "dataset": "target-reference" },
  "training_patches": [
    { "id": "capture-1-A01", "scanner_xy": [-0.75, -0.50], "rgb": [0.81, 0.78, 0.70], "xyz": [0.760, 0.780, 0.650] },
    { "id": "capture-1-A02", "scanner_xy": [-0.50, -0.50], "rgb": [0.42, 0.36, 0.30], "lab": [62.1, 1.2, 5.4] }
  ],
  "held_out_patches": [
    { "id": "capture-2-A01", "scanner_xy": [0.25, 0.50], "rgb": [0.80, 0.77, 0.69], "xyz": [0.760, 0.780, 0.650] },
    { "id": "capture-2-A02", "scanner_xy": [0.50, 0.50], "rgb": [0.41, 0.35, 0.29], "lab": [62.1, 1.2, 5.4] }
  ]
}
```

The abbreviated example omits the remaining samples. `rgb` is the normalized decoded scanner signal,
before any declared scanner linearization. `scanner_xy` is the patch centre in the original scanner
frame, normalized to `[-1,1]`; it is optional for non-spatial models and mandatory for every patch
when `shading_gain_polynomial` is present. Sample IDs identify measurements, so repeated captures of
the same physical chart patch still need distinct IDs; changing only an ID does not make an exactly
duplicated RGB/reference/coordinate tuple independent. Measurements at genuinely different scanner
coordinates remain distinct because they test the spatial model. The fitter never uses held-out
samples to solve the matrix or estimate its whitepoint. It derives confidence, aggregate residuals,
hue-family residuals, and worst-patch diagnostics only from held-out samples. It also records
training residuals and an identity-matrix baseline. A legacy single `patches[]` input is rejected
because it can only produce an in-sample residual. The written record removes both input lists and
retains the held-out set as top-level `patches` for later candidate comparison. If a nonlinear model
is selected, its exact training measurements are additionally retained under
`color_model.training_patches` so the fit and measured support domain remain independently
recomputable.

Before fitting either matrix, the utility applies the scanner profile's black/white, flare,
per-channel response, and spatial shading model to both training and held-out RGB values. Retained
held-out `patches` therefore use the same post-linearization signal seen by the runtime matrix.
The record declares `target_patch_signal_domain` as `normalized_scanner_signal` or
`scanner_linearized_transmittance`; the latter also requires the exact
`target_patch_linearization_model_id`. Scanner and roll records are rejected if these fields do not
match the runtime scanner model.

Schema-v2 records that contain a `fit` object must contain this held-out `fit.validation` evidence
and the signal-domain provenance above. Older schema-v2 records with in-sample-only or ambiguous-domain
fit claims are rejected and must be regenerated. A
pre-fit matrix without a `fit` object remains an explicitly authored profile; its supplied confidence
must come from independent external validation rather than this command.

### Evidence-gated root-polynomial colour

After fitting the mandatory matrix, `scanner-target` evaluates homogeneous degree-two and
degree-three root-polynomial transforms. The six degree-two basis terms are `R`, `G`, `B`,
`sqrt(RG)`, `sqrt(RB)`, and `sqrt(GB)`. Degree three appends `cbrt(R^2G)`, `cbrt(R^2B)`,
`cbrt(G^2R)`, `cbrt(G^2B)`, `cbrt(B^2R)`, `cbrt(B^2G)`, and `cbrt(RGB)`. Every term is
homogeneous of effective degree one, preserving exposure scaling while modelling cross-channel
scanner/spectral interaction.

The fitter is deliberately evidence-limited:

- degree two requires at least 18 training and 18 held-out measurements; degree three requires at
  least 39 training and 24 held-out measurements;
- regularization is selected by three-to-five-fold cross-validation using training data only, then
  a robust Huber/IRLS solve is performed on the training set;
- a candidate must have condition number at most `1e8`, absolute coefficient at most `50`, held-out
  DeltaE00 RMS at most `6`, maximum at most `15`, at least `0.25` and `5%` RMS improvement over the
  matrix, no more than `1.0` maximum-DeltaE00 regression, no more than `2%` XYZ-RMS regression, and
  no hue-family RMS regression above `0.5`;
- degree three replaces an accepted degree-two model only with at least `0.15` and `3%` further
  held-out RMS improvement and no material maximum-error regression.

Only the selected model is written as typed `color_model`; all attempts remain in
`polynomial_fit`. The typed model retains its training patches, while top-level `patches` retains the
disjoint held-out set. The loader repeats training-only cross-validation and robust fitting,
reselects degree/matrix complexity, reconstructs the chromaticity hull and condition number, and
recomputes training plus held-out XYZ/CIEDE2000/hue metrics against the exact stored matrix fallback.
A changed patch, coefficient, count, metric, hull, basis, signal domain, or false selection claim
rejects the record instead of trusting its JSON labels.

At render time the model is a separately ranked direct-calibration candidate. Up to 120,000 finite
scene pixels are checked against a 10%-expanded convex hull of the training RGB chromaticities. At
least 64 samples are required; more than 2% negative source samples or more than 35% outside the
expanded hull rejects the nonlinear candidate and retains the matrix. In-domain candidates still
must pass the same gamut, tone, perceptual, reference-patch, and automatic-selection comparisons as
the matrix and image-derived alternatives. When selected, the exact validated transform is applied
per pixel and post-model neutral trim is skipped; technical and creative white balance remain
separate later stages. Scanner-only library selection remains the conservative scanner-constrained
image-adaptation path; the nonlinear transform participates directly for one-off direct profiles or
a compatible scanner-plus-roll correction.

### Evidence-gated smooth residual 3D LUT

After polynomial selection, `scanner-target` evaluates a smooth residual LUT over the strongest
already-qualified baseline: the selected root-polynomial model when present, otherwise the mandatory
matrix. It does not fit an unconstrained RGB cube. The model stores the exact RGB minimum/maximum and
chromaticity hull of its retained training measurements, uses tetrahedral interpolation, and forces
every boundary node to zero. Those zero faces meet the simpler baseline continuously instead of
creating a discontinuity at the learned volume boundary. Chromaticities outside the measured hull
blend back to the baseline over a bounded 10% hull expansion.

The evidence burden rises sharply with lattice density:

- 5x5x5 has 125 stored nodes but only 27 fitted interior nodes and requires at least 108 training
  plus 48 disjoint held-out patches;
- 7x7x7 has 343 stored nodes but 125 fitted interior nodes and requires at least 500 training plus
  96 disjoint held-out patches;
- regularization is selected with deterministic three-to-five-fold training-only cross-validation;
  the final solve uses five robust Huber/IRLS iterations and penalizes adjacent-node roughness;
- an applied LUT requires at least 35% occupied training cells, at least 98% of held-out samples
  inside the learned RGB domain, condition number at most `1e10`, node magnitude at most `0.25`,
  node RMS and edge roughness at most `0.10`, held-out DeltaE00 RMS at most `4`, maximum at most
  `10`, and at least `0.20` plus `5%` RMS improvement over the declared baseline;
- held-out maximum DeltaE00 may regress by at most `0.5`, XYZ RMS by at most `1%`, and any
  hue-family DeltaE00 RMS by at most `0.35`; and
- 7x7x7 replaces an accepted 5x5x5 lattice only with at least `0.15` and `3%` additional held-out
  RMS gain and no material maximum-error regression.

Only the selected lattice is written as `lut_3d_model`; the complete selection is retained in
`lut_3d_fit`. The typed model carries all training patches, all nodes, its exact domain/hull, the
baseline kind/model ID, validation metrics, and held-out fit. At load time ScanStitch repeats model
selection, cross-validation, robust fitting, node generation, domain/hull construction, and every
reported metric. A changed node, patch, baseline identity, metric, count, or complexity claim rejects
the record.

At render time the LUT is separately scored against its root-polynomial and matrix fallbacks. More
than 20% of finite scene samples outside the learned RGB box, less than 65% receiving the full LUT,
more than 2% negative samples, more than 35% outside the expanded chromaticity hull, or fewer than
64 usable samples rejects the LUT candidate. The evaluator itself returns the declared baseline
outside the RGB volume and smoothly tapers the residual near chromaticity support, so unsupported
pixels are never extrapolated through a free-running lattice. An accepted support gate still does
not bypass gamut, clipping, tone, perceptual, reference-patch, or candidate-ranking checks.

To fit scanner signal linearization, include `scanner_linearization_fit` in a `scanner-target`
measurement that also contains either the disjoint target patch lists above or a pre-fit
`scanner_rgb_to_xyz` matrix:

```json
{
  "scanner_linearization_fit": {
    "model_id": "coolscan-linearization-v1",
    "black_level_normalized": [0.01, 0.01, 0.01],
    "white_level_normalized": [0.99, 0.99, 0.99],
    "additive_flare_normalized": [0.002, 0.002, 0.002],
    "shading_gain_polynomial": [
      [1.0, 0.01, 0.0, 0.0, 0.0, 0.0],
      [1.0, 0.01, 0.0, 0.0, 0.0, 0.0],
      [1.0, 0.01, 0.0, 0.0, 0.0, 0.0]
    ],
    "confidence": 0.95,
    "training_samples": [
      {
        "scanner_signal": [0.50, 0.50, 0.50],
        "reference_transmittance": [0.25, 0.25, 0.25],
        "x": 0.0,
        "y": 0.0
      }
    ],
    "held_out_samples": [
      {
        "scanner_signal": [0.75, 0.75, 0.75],
        "reference_transmittance": [0.5625, 0.5625, 0.5625],
        "x": 0.5,
        "y": -0.5
      }
    ]
  }
}
```

The example is abbreviated; at least 12 disjoint training and 12 held-out samples are required.
Signals and reference transmittance are normalized to `[0,1]`; spatial coordinates use `[-1,1]`.
The optional six-term shading rows are gains over `[1,x,y,x²,xy,y²]`. The fitter derives monotone
per-channel curves and writes a model only when confidence is at least 0.85, held-out transmittance
RMSE is at most 0.01, maximum error is at most 0.03, identity RMSE improves by at least 0.001,
shading gain stays within 0.5–2.0, and total curve/level/shading noise gain stays below 4. The raw
fit samples are removed from the written scanner record.

`roll-target` uses the selected `--scanner-profile` to transform both patch sets through that
profile's scanner linearization and then its strongest held-out-selected colour transform: the typed
residual LUT when qualified, otherwise the typed root-polynomial model when qualified, otherwise the
mandatory scanner matrix. LUT-based roll patches must have full measured LUT support. It fits a small
`xyz_post_scanner` correction on the training set and scores it on the held-out set. Finite XYZ
components in `[-1e-12, 0)` are treated as numerical zero; larger negative or non-finite results abort
the fit. The written `scanner_color_model_application` records the transform, nonlinear model ID,
D50 XYZ output domain, tolerance, and clamp count.

The loader verifies this application record against the selected scanner model. A roll correction
paired with a nonlinear scanner profile is rejected when the record is absent or mismatched, rather
than silently applying a correction fitted in another XYZ domain. Scanner settings, signal domain,
and linearization model must also match. Pre-fit `correction_matrix` records are stamped with the
selected scanner fingerprint; when the scanner has a nonlinear colour model, the input record must
already carry matching `scanner_color_model_application` evidence.

To fit the nonlinear negative reconstruction, provide `negative_response_fit` in the roll-target
measurement object. Training and held-out patches must be disjoint. A training patch supplies the
base-subtracted scanner optical density, independently known separated film-layer density, and
reference log10 scene exposure. Held-out patches supply only scanner density and reference log
exposure; they are not used in either the 3×3 least-squares crosstalk fit or the robust
binning/isotonic curve fit:

```json
{
  "schema_version": 2,
  "film": { "stock": "Kodak Gold 200", "process": "C-41" },
  "base_color": [12000.0, 7000.0, 3000.0],
  "confidence": 0.92,
  "negative_response_fit": {
    "model_id": "gold-200-response-v1",
    "scene_rgb_to_xyz_d50": [
      [0.7977, 0.1352, 0.0313],
      [0.2880, 0.7119, 0.0001],
      [0.0, 0.0, 0.8251]
    ],
    "confidence": 0.94,
    "white_anchor_percentile": 0.995,
    "training_patches": [
      {
        "scanner_density": [0.42, 0.31, 0.24],
        "reference_layer_density": [0.39, 0.30, 0.22],
        "reference_log_exposure": [0.34, 0.33, 0.32]
      }
    ],
    "held_out_patches": [
      {
        "scanner_density": [0.67, 0.52, 0.41],
        "reference_log_exposure": [0.55, 0.54, 0.53]
      }
    ]
  }
}
```

The abbreviated example shows one patch per list; the command requires at least 12 training and 12
held-out patches, and normally benefits from substantially more patches spanning neutral steps,
skin, foliage, sky, saturated dyes, toe, straight-line, and shoulder regions. It fits a regularized
scanner-density-to-layer matrix, derives up to 12 robust monotone curve knots per layer, evaluates
both the fitted model and unit-slope fallback in D50 Lab, computes held-out DeltaE00, and refuses to
write the record unless all runtime acceptance limits pass. The raw `negative_response_fit` samples
are removed from the written library record; only the fitted model and audit metrics remain.

When `scanstitch-calibrate` fits a target, the written `fit` object includes held-out aggregate
RMS/max residuals plus `per_hue_residuals`, `worst_patches`, training residuals, and identity-baseline
comparison. Use those lists to catch poor generalization and hue-localized target problems before
relying on the profile in automatic color mapping.

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
| `target_patch_signal_domain` | Required for a schema-v2 fitted matrix: `normalized_scanner_signal` or `scanner_linearized_transmittance`. |
| `target_patch_linearization_model_id` | Required with `scanner_linearized_transmittance`; must exactly identify the runtime scanner model. |
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
patches, target-patch signal domain and linearization model ID, DeltaE00 summaries, scanner
response/flare diagnostics, retained `target_sampling` capture hashes/geometry/quality evidence,
typed root-polynomial and residual-LUT models plus held-out selection audits,
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
`reference_patch_evaluation`, `nonlinear_color_model`, `candidate_comparison`, and
`usable_colourspace` diagnostics. `nonlinear_color_model` records the model identity, polynomial
degree or LUT grid size, baseline kind, held-out CIEDE2000 evidence, sampled-pixel count,
negative/outside-hull/outside-RGB-volume ratios, full-model application ratio, and the explicit
scene-support decision even when a simpler fallback wins.
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
`--color-mode calibrated` still fails clearly when the only calibrated candidates are unsafe. A
nonlinear candidate outside measured scene support is rejected before this comparison, so forcing
calibrated mode may select the valid matrix fallback but cannot force unsupported extrapolation.

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
2. Fit a source RGB to XYZ matrix and let the utility select a typed root-polynomial and, only with
   the much larger disjoint target set, smooth residual-LUT candidate. Write a schema v2 profile when
   response/black/white/target diagnostics are available; v1 records remain readable.
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

Hand grading, curator-trained creative looks, and evidence-qualified 3D-LUT application remain out
of scope for this phase.

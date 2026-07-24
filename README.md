# ScanStitch

Reconstructs positive or negative film scans from one image or an overlapping sequence. It honors source precision/orientation/color metadata, removes scanner borders on all four edges, orders and stitches split frames, reconstructs negatives in a scene-referred density domain, maps into linear ProPhoto RGB D50, and produces a tagged master plus display proof.

## Pipeline

| Phase | Module | What it does |
|-|-|-|
| 1 | `border.rs` | Detects/removes scanner dead zones on all four edges while retaining a separate pre-crop rebate measurement |
| 2a | `base_detect.rs` | Identifies film base/rebate independently of the final content crop |
| 2b | `frame_classify.rs` | Classifies frame as Intact / SingleBorder / CandidateSplit / Ambiguous, then emits an explicit stitch decision |
| 2c | `stitch.rs` | Builds a validated directed overlap graph for N inputs, infers a left-to-right path, aligns every merge, preserves the full union, conditionally normalizes overlap exposure, and rejects incomplete sequences instead of presenting a partial mosaic as complete |
| 3 | `density.rs` | Applies measured or bounded fallback negative-response reconstruction, film-base subtraction, dye-crosstalk separation, nonlinear characteristic curves, and signed scene headroom |
| 4 | `ica.rs` | Optional blind fallback candidate; automatic rendering prefers physically grounded direct-density evidence when comparably safe |
| 4.6 | `colorspace.rs` | Rank calibrated, scanner-prior, image-derived, and neutral-balance color mapping candidates with quality scores, Bradford-adapt to ProPhoto RGB (D50), then report calibration acceptance, neutral support, gamut safety, and neutral trim diagnostics |
| 5 | `tonemap.rs` | Naka-Rushton sigmoid with histogram-fitted midpoint/slope, diagnostics-driven chroma protection, bounded local detail, optional independently controlled edge-aware film-grain reduction, and render-quality color diagnostics |

## Build

```bash
# Default (pure Rust, no external deps beyond cargo)
cargo build --release

# With OpenCV for robust feature matching
# Requires libclang/LLVM tooling discoverable via `llvm-config` or `LIBCLANG_PATH`
cargo build --release --features use-opencv
```

On Ubuntu 24.04 the hosted feature gate installs `clang`, `libopencv-dev`, `libclang-dev`,
`llvm-dev`, and `pkg-config`, sets `LIBCLANG_PATH` from `llvm-config --libdir`, and treats a feature-build or
feature-test failure as a CI failure. The default pure-Rust path remains independent of these
packages, and the Rust dependency disables OpenCV's broad default module set so only the
`calib3d`, `features2d`, and `imgproc` bindings required by ScanStitch are generated. The native
library suite also executes deterministic SIFT and RGB16 projective-warp smoke paths.

## Usage

```
scanstitch <INPUT>... [OPTIONS]
```

One input is processed directly; no duplicate/dummy path is required. Two or more inputs are overlap-scored and, when justified, automatically ordered and stitched:

```bash
scanstitch full-frame.tiff -o output/
scanstitch right.tiff left.tiff middle.tiff -o output/
scanstitch negative.tiff -o output/ --grain-reduction on --grain-strength 0.6 --grain-scale 1.25
```

| Flag | Default | Description |
|-|-|-|
| `-o, --output-dir` | `output` | Where to write results |
| `--debug` | off | Save intermediate images + tone curve LUT |
| `--force-stitch` | off | Force stitching even if frames look intact |
| `--force-no-stitch` | off | Skip stitching entirely |
| `--bit-depth` | 14 | Input bit depth (14-bit padded to 16, or native 16) |
| `--input-mode` | negative | Input workflow: `negative` for film negatives, `positive` for slide scans or already-inverted RGB; aliases `slide` and `positive-slide` report as `positive` |
| `--calibration-profile` | none | Compatibility one-off scanner/film calibration profile JSON |
| `--calibration-library` | none | Local scanner, roll, and film-hint calibration library directory |
| `--scanner-profile` | none | Scanner/settings profile ID selected from the calibration library |
| `--roll-profile` | none | Optional roll correction/profile ID selected from the calibration library |
| `--film-stock` | none | Optional film-stock label used to rank film evidence and reject mismatched roll profiles |
| `--color-mode` | auto | Color mapping mode: auto / calibrated / image-derived / neutral |
| `--render-input` | auto | Negative-mode render input: auto / ica / direct-density. `--input-mode positive --render-input ica` is rejected. |
| `--render-intent` | modern-clean | Finished-render intent: modern-clean / natural-neutral / film-faithful |
| `--quality-mode` | perfect | Quality/throughput mode: perfect / balanced / fast. Perfect writes master and review artifacts by default. |
| `--deskew` | auto | Absolute deskew policy: evidence-gated single-scan auto / explicit manual / off. |
| `--orientation-correction` | none | Explicit semantic-upright correction relative to embedded metadata: none, flips, 90°/180°/270° rotations, and mirrored 90°/270° variants. Applied after EXIF/DNG orientation and before scanner linearization, deskew, crop, or stitch. |
| `--deskew-angle-degrees` | 0 | Signed clockwise source skew from -3° through 3° for manual deskew; the applied correction uses the opposite sign. |
| `--technical-white-balance` | auto | Technical scene illuminant policy: evidence-gated auto / explicit manual / off. |
| `--technical-temperature-kelvin` | 5003 | Manual source-illuminant CCT from 2000 K through 25000 K. |
| `--technical-tint` | 0 | Manual source-illuminant green-to-magenta correction from -1 through 1. |
| `--creative-temperature` | 0 | Independent finished-render cool-to-warm adjustment from -1 through 1; excluded from the technical master. |
| `--creative-tint` | 0 | Independent finished-render green-to-magenta adjustment from -1 through 1; excluded from the technical master. |
| `--grain-reduction` | off | Optional edge/detail-aware film-grain reduction: off / on. This is independent of render intent. |
| `--grain-strength` | 0.5 | Reduction strength from 0 through 1; retained while reduction is off so it can be enabled interactively. |
| `--grain-scale` | 1.0 | Grain-radius multiplier from 0.5 through 4.0. |
| `--write-master` | auto in perfect | Write `master_scene_referred.tiff` outside perfect mode too |
| `--review-sidecar` | none | Apply saved guided-review render decisions |
| `--write-review-sidecar` | none | Write guided-review render decisions after saving |
| `--ica-max-iter` | 100 | Max FastICA iterations |
| `--ica-tol` | 1e-5 | FastICA convergence threshold |
| `--transform` | auto | Transform model: auto / translation / affine / homography. Auto upgrades coherent skew to native affine only after held-out overlap improvement and rejects rotation optima that hit the bounded search edge; a homography must pass disjoint feature cross-fit and then beat translation on both image-domain spatial validation splits. |
| `--use-opencv` | off | Use the optional OpenCV backend for homography (needs `use-opencv` feature); native affine does not require OpenCV. |
| `--require-reviewable` | off | Production automation gate. Preserve the completed report/artifacts but exit nonzero unless the final save phase agrees on `render_review_status=reviewable` and `render_reviewable=true`; the saved TIFF independently matches its reported dimensions, RGB16 storage, ICC profile, and schema-v4 SHA-256; every requested master/review artifact independently matches its format, dimensions, profile, and schema-v4 SHA-256; and the stale-artifact count is explicitly zero. Missing or inconsistent evidence fails closed. |

Enable logging with `RUST_LOG=info` (or `debug` for verbose). The main `scanstitch` CLI also emits
a low-noise stderr heartbeat every 30 seconds during long work. It reports elapsed time plus the
last phase completed in the current run's partial `report.json`; it never presents an in-progress
phase as complete. Set `SCANSTITCH_PROGRESS_INTERVAL_SECONDS` to another value from 1 through 3600,
or to `0` to disable the heartbeat. The final phase summary includes each recorded duration.
For unattended production work, add `--require-reviewable`; a blocked or review-required render is
still written for diagnosis, but the command cannot be mistaken for a successful deliverable.

## Interactive UI

`scanstitch-ui` runs the same pipeline up to the ProPhoto render cache, then opens a terminal control UI plus a separate native preview window:

```bash
cargo run --bin scanstitch-ui -- scan1.tiff scan2.tiff scan3.tiff -o output/
```

Use `Tab` to select creative temperature/tint, tone, or grain controls, arrow keys or `+`/`-` to adjust, `r` to reset to the automatic technical/render defaults, `s` to write `output.tiff` and `report.json`, and `q` to quit. Grain reduction has a separate on/off control plus strength and scale. Creative white balance remains outside `master_scene_referred.tiff`; the interactive path does not write final output until `s` is pressed.

## Output

| File | Always | Description |
|-|-|-|
| `output.tiff` | yes | Final 16-bit positive TIFF (linear ProPhoto RGB D50 ICC embedded, tone-mapped) |
| `master_scene_referred.tiff` | perfect | 32-bit float scene-referred linear ProPhoto RGB D50 master with highlight headroom and embedded linear-ProPhoto ICC profile preserved |
| `review_srgb.png` | perfect | Display-ready RGB8 proof with constant-lightness/hue D50 Lab gamut mapping into sRGB D65 and an explicit standard sRGB ICC profile |
| `report.json` | yes | Run metadata including cached exact executable SHA-256/size/path, schema-v4 SHA-256 values over the reopened primary/master/proof bytes, output freshness diagnostics, per-phase confidence, metrics, warnings, errors, stitch hypotheses, candidate-ranking diagnostics, validation-vs-search selection diagnostics, overlap/local/plausibility score breakdowns, correspondence-vs-prior weights, overlap-range diagnostics, downstream base-confidence caps, and explicit rejection reasons |
| review sidecar JSON | optional | Non-destructive interactive/guided render decisions when `--write-review-sidecar` is supplied |
| `component_XX_cropped.tiff` | debug | Each input after four-edge border removal |
| `component_XX_base_mask_left.tiff` / `component_XX_base_mask_right.tiff` | debug | Per-component edge-confidence grids |
| `*_base_columns.tiff` / `*_base_overlay.tiff` | debug | Full-width base-gap confidence and overlays |
| `stitch_overlap_12_*.tiff` / `stitch_overlap_21_*.tiff` | debug | Both stitch hypotheses: aligned overlap strips and seam overlays |
| `stitch_seam_overlay.tiff` | debug | Final accepted overlap/seam visualization |
| `stitched.tiff` | debug | After stitching (if stitching occurred) |
| `phase3_positive.tiff` | debug | After density inversion |
| `phase3_positive_passthrough.tiff` | debug | Positive-mode normalized RGB passthrough before color mapping |
| `phase3_linear_division.tiff` | debug | Negative-mode diagnostic linear-division inversion view |
| `phase4_ica.tiff` | debug | After FastICA separation |
| `phase46_prophoto.tiff` | debug | After ProPhoto mapping |
| `phase46_scene_referred_prophoto_float.tiff` | debug | 32-bit float scene-referred linear ProPhoto buffer, preserving finite values below 0 and above display white |
| `phase46_color_candidate_comparison.tiff` | debug | Side-by-side selected/image-derived/calibrated candidate thumbnail when available |
| `phase46_gamut_clipping_map.tiff` | debug | Selected-transform clipping map: red high, blue low, green preserved in-gamut |
| `phase47_technical_white_balance_scene_referred.tiff` | debug | Technical scene-referred master after evidence-gated or manual Bradford adaptation, before creative grading/tone mapping |
| `tone_curve_lut.json` | debug | 256-point [input, output] tone curve |

Schema-v4 artifact hashes bind an unchanged `report.json` to the exact saved bytes and reject a
valid-looking substituted image that still matches dimensions/storage/profile. They are not a
digital signature: coordinated report/artifact edits or a schema downgrade are outside this gate's
trust boundary. Pin the completed hash-bound render-review manifest in a fixture—and preserve
reviewer provenance—when an acceptance record is required.

Primary TIFFs, requested masters/proofs, reports, sidecars, calibration records, and validator
review files are written to same-directory temporary files and then replaced per file. A failed
encode therefore leaves an existing destination intact and never exposes a partial replacement.
This is not a transaction across all artifacts and does not `fsync` file/directory metadata for
power-loss durability; an overwrite also temporarily needs space for the old and new copy of the
artifact being committed.

## Report Diagnostics

The `colorspace_mapping` phase reports `candidate_quality_scores`, `quality_components`,
`color_decision_summary`, `selected_quality_score`, `technical_safety_score`,
`color_fidelity_score`, `candidate_risk`, `tone_color_trust_state`, `calibration_acceptance`,
`calibration.color_mapping_application`, `neutral_safety_rescue`,
`neutral_estimate_quality`, `neutral_sample_rejections`, `dominant_anchor_sample_rejections`,
`dominant_anchor_quality`, candidate `rendered_tone_quality` / `color_model_quality`,
XYZ/DeltaE reference-patch residuals, typed root-polynomial or residual-3D-LUT
`nonlinear_color_model` scene-support evidence, and
`neutral_trim_before_after` in addition to the legacy `candidate_scores` fields. In `auto`,
calibrated and scanner-prior candidates are used
only when they beat image-derived quality, stay inside negative-gamut safety limits, and do not
regress neutral or reference-patch diagnostics. Candidate metrics for ICA and direct-density render
inputs include the same anchor support fields alongside gamut clipping, exposure-scale,
neutral-delta, and preserved-gamut diagnostics. When uncalibrated dominant anchors are unsupported
and the matrix produces an extreme cast, auto mode may select an evidence-supported neutral safety
rescue only after conservative gamut, saturation, chroma-retention, memory-colour, and spatial
consistency gates pass. The result remains fallback-only and review-required; calibrated,
scanner-prior, positive, nonlinear, and forced selections are never displaced by this rescue.
`calibration.status=applied` means that a calibration record or one of its independent components
was usable; it does not by itself mean the record's colour mapping reached the output. Inspect
`calibration.color_mapping_application.applied` and `selection_status` for that claim. The validator
cross-checks those fields against final candidate selection, and strict validation fails on any
contradiction.
Direct-density render input normalizes the Phase 3
positive-density range by `shared_robust_d_max` before transmittance conversion, matching ICA's
bounded density domain and preventing sparse clipped pixels from placing high-key negatives near
black; the report records `direct_density_render_density_scale`. Automatic selection treats direct
density as the physical prior when its risk, gamut, neutral, and quality diagnostics remain within
conservative tolerances; a held-out-validated measured response outranks blind ICA unless its actual
selected colour mapping is catastrophically clipped. Debug runs write
`phase46_prophoto.tiff`, `phase46_scene_referred_prophoto_float.tiff`, and, when
candidate previews are available, `phase46_color_candidate_comparison.tiff`. They also write
`phase46_gamut_clipping_map.tiff`, where red marks post-scale high clipping, blue marks post-scale
low clipping, and green marks preserved in-gamut pixels. The float ProPhoto artifact is the
high-latitude scene-referred buffer before tone mapping; use it for exposure/colour inspection
instead of the clamped preview when highlights exceed display white.

When `--input-mode positive` is selected, the pipeline still loads, crops borders, optionally
stitches, color maps, tone maps, saves, and reports normally, but skips negative-film base
requirements, density inversion, orange-mask removal, and ICA. The selected working RGB scan is
normalized directly to linear `[0,1]`; `density_inversion` and `fastica` remain in the report with
`skipped: true` and `input_mode: "positive"`, and `colorspace_mapping.render_input_source` is
`positive_scan_rgb`. In uncalibrated auto colour mode, the colour phase now uses
`mapping_strategy=positive_rgb_passthrough`: it preserves the already-positive RGB channel ratios in
the linear ProPhoto working buffer, evaluates clipping/headroom and tone/model diagnostics, and does
not require negative-film neutral or dominant-anchor support. Explicit calibration profiles still
use the calibrated candidate path. Positive mode also uses a separate `positive_scan_rgb` tone
policy: it may apply only bounded non-brightening pre-tone exposure, places midtones more
conservatively than the negative-film fit, and uses a slightly softer shoulder so already-positive
scans keep natural density without inverting luminance order.
Positive-mode suitability is still checked before reconstruction. A scan with severe orange-mask
channel ratios reports `review_required_negative_like_positive_input`, working-phase confidence
`0.0`, and final `render_review_status=review_required_input_mode`; output is retained for diagnosis
but `--require-reviewable` rejects it. A high warm score with mild ratios reports
`accepted_warm_positive_input` instead, so genuinely warm slides are not failed merely for warmth.

Calibration diagnostics live under `colorspace_mapping.metrics.calibration`. A selected scanner
profile is used as the stable scanner/settings prior, an explicit roll matrix is composed after it,
requested film-stock evidence ranks roll/film candidates and rejects mismatched explicit roll
profiles, and weak auto-matches are reported as advisory instead of being silently applied. The older
`--calibration-profile` path remains available for one-off scanner/film profiles. See
`docs/color-calibration.md` for library record examples, the calibration-library record schema, and
`scanstitch-calibrate` usage. See `docs/color-reconstruction-policy.md` for the density, candidate
scoring, and constant-change policy behind those decisions.

With at least 18 disjoint training and held-out target patches, `scanstitch-calibrate scanner-target`
also evaluates a quadratic root-polynomial transform; a cubic candidate requires 39 training and 24
held-out patches. It writes a typed model only when independent CIEDE2000/XYZ/hue evidence beats the
matrix. Rendering then checks scene support and ranks that model against the matrix and image-derived
alternatives, falling back rather than extrapolating beyond the measured colour domain.
With at least 108 training and 48 held-out patches it also evaluates a smooth residual 5x5x5
tetrahedral LUT over the selected polynomial or matrix; a 7x7x7 candidate requires 500/96. Boundary
nodes are fixed to zero, smoothness is regularized on training data, and held-out CIEDE2000, XYZ,
hue, occupancy, support, node-magnitude, roughness, and condition gates must beat the simpler
baseline. The loader deterministically refits every node from retained evidence. Runtime rejects the
LUT when scene coverage falls outside its measured RGB volume/hull and keeps the polynomial/matrix
fallback eligible.
`roll-target` fits its XYZ-domain correction after that exact selected scanner transform and records
the matrix/root-polynomial/residual-LUT identity, model ID, D50 output domain, and numerical-zero clamp audit;
nonlinear scanner/roll pairings without matching evidence are rejected.

Create or update local calibration records with `scanstitch-calibrate`:

```powershell
cargo run --bin scanstitch-calibrate -- sample-target --manifest chart-sampling.json --output scanner-target.json --preview-dir output/chart-sampling
cargo run --bin scanstitch-calibrate -- scanner-target --library calibration --measurements scanner-target.json --profile-id coolscan-4000-vuescan-raw
cargo run --bin scanstitch-calibrate -- roll-target --library calibration --scanner-profile coolscan-4000-vuescan-raw --measurements roll-target.json --profile-id logan-roll
cargo run --bin scanstitch-calibrate -- roll-base --library calibration --scanner-profile coolscan-4000-vuescan-raw --profile-id logan-roll-base --film-stock "Kodak Gold 200" --process C-41 --base-color 12000,7000,3000
```

`sample-target` perspective-maps a declared chart grid in independent training/held-out 12-16-bit
TIFF/DNG captures, rejects duplicate files or decoded pixels and contaminated/clipped/nonuniform patches, derives
original-scanner-frame coordinates through EXIF orientation, and writes overlay PNGs plus an audit
report. Its explicit `scanner_signal_bit_depth` handles 14-bit scanner codes padded into 16-bit TIFF
containers without dividing by the wrong maximum. It also writes
`corner-review-manifest.json`, binding each declared projective footprint to the exact
orientation-materialized decoded-pixel SHA-256, a SHA-256 of the ordered corner coordinates, and a
SHA-256 of the rendered review-overlay pixels. The same combined review manifest pins the ordered
patch IDs, grid positions, and reference XYZ/Lab values through `reference_patch_sha256`.
The first run deliberately writes no measurement
until a person inspects every grid overlay, verifies the colour values against the named factual
dataset, and sets the matching `corner_review.approved=true` and
`reference_review.approved=true` declarations. Changed pixels, corner geometry, grid layout,
overlay rendering, patch identity, or reference values invalidate approval. Rejected, unapproved,
or hash-mismatched sampling never writes fit-ready measurement JSON. See
`docs/calibration-target-sampling.example.json` and its schema.

Scanner and roll target matrices require at least 12 uniquely identified training patches and 12
disjoint held-out patches; confidence and residual diagnostics come only from the held-out set.
Both patch sets are transformed through the exact selected scanner linearization before matrix
fitting, and roll targets then use the selected held-out-qualified scanner colour model. Spatial
models require normalized original-frame `scanner_xy` on every patch. Fitted
schema-v2 records declare their signal domain and linearization model ID; in-sample-only,
ambiguous-domain, and scanner/roll domain-mismatch claims are rejected. `scanner-target` also fits held-out-validated
black/white, flare, nonlinear signal, and spatial shading correction when its JSON contains
`scanner_linearization_fit`. `roll-target` also fits a measured negative response when the JSON contains
`negative_response_fit`: a scanner-density crosstalk matrix, monotone nonlinear film-layer curves,
and held-out DeltaE00 comparison against the unit-slope fallback. See
`docs/color-calibration.md` for the measurement contract and acceptance gates.

Accepted scanner linearization is applied to every full component before border detection and
stitching. Spatial gains use the original scanner frame even when EXIF orientation has already been
materialized. A scanner model selected only after mosaicking is rejected instead of being applied in
the wrong coordinate system.

EXIF/DNG orientation describes pixel layout, not whether the photographed scene is semantically
upright. When a scan is still sideways, upside down, or mirrored after metadata is honored, pass an
explicit correction such as `--orientation-correction rotate-180`. ScanStitch composes that request
with the source tag, rebinds the exact decoded-pixel SHA-256, preserves original-scanner coordinates
for spatial calibration, and runs all geometry on the corrected pixels. It does not silently guess
semantic orientation; fixture acceptance still requires a hash-bound human upright review.

The `stitch` phase reports `seam_exposure_correction` for accepted stitches. In `auto` mode this
stays identity unless robust overlap samples show a meaningful mismatch. Whole spatial windows are
split into training and held-out sets; scalar gain, RGB gain, bounded RGB gain+offset, robust
per-channel vertical log-gain/gain+offset fields, planar x/y fields, and quadratic x/y gain or
gain+offset fields are compared. The vertical gain field
is fitted from whole-window medians, requires at least three distinct rows in both splits, must
reproduce its slope on unseen windows, and must beat the best accepted constant model. The more
complex affine field robustly fits four parameters per channel (`log center gain`, `gain slope`,
`center offset`, and `offset slope`) from signal-rich training pixels. It needs four distinct rows
in both independently fitted splits, direct held-out improvement across at least 65% of windows,
at least 2/3 slope agreement, stable center offsets, and an improvement over the best simpler model
of at least 0.0025 and 20%. Every gain endpoint remains within 0.80–1.25, every offset endpoint
within 5% of the sample range, and clipping growth shares the same hard limit. Spatial coordinates
clamp outside the measured overlap height, so the compositor never extrapolates an unsupported
gradient. A 2D candidate additionally needs at least four rows and three columns in both whole-cell
splits, independently fitted horizontal slope agreement, bounded gain/offset values at all four
corners, stable centers, and a material held-out win over every accepted constant or vertical model.
Both coordinates clamp outside the observed overlap rectangle. The constant additive model likewise must materially beat identity and gain-only, local
fits must agree, and each offset remains within 5% of the input range. These are conservative overlap-derived seam
corrections, not substitutes for a measured scanner shading/black profile. A quadratic candidate
uses the fixed `[1, x, y, x^2, x*y, y^2]` basis, needs at least 18 windows, six rows, and five columns
in each disjoint split, and is robustly regularized. Independently fitted curvature coefficients and
the complete fields must agree, every held-out window and clipping gate still applies, and a 9x9
support grid plus exact quadratic edge/interior extrema checks gain/offset bounds that four corners
cannot reveal. Held-out
window-consistency and clipping gates still apply. The coupled gain+offset solve is skipped when the
gain-only residual cannot clear its absolute complexity
margin. Nonlinear shading outside the observed overlap support is intentionally not inferred from
scene content.
Translation, native-affine, and accepted OpenCV-homography stitches then use an explicit validity
mask, preserve the full valid union, choose a minimum-cost low-detail seam in the largest fully
valid overlap rectangle, and blend five bounded spatial-frequency bands. Projective output uses one
pure-Rust bicubic resample; valid black pixels are never mistaken for empty canvas. `seam_blend`
reports overlap residuals and the output/source seam-gradient ratio; fixture suites can require
this mode and enforce scan-specific ceilings. It also compares luma high-pass energy at radii 1,
2, and 4 pixels in a 6x6 whole-cell checkerboard. A focus/grain/detail mismatch requests review
only when both disjoint partitions reproduce at least a 2x imbalance with consistent direction and
bounded ratio disagreement on at least two scales. Fewer than two signal-supported scales also
request review because detail continuity was not measurable. The full stitch is retained for inspection, but
the stitch phase emits a warning and caps confidence at 0.35, and the final render becomes
`blocked_geometry_review`; no unsupported blur or sharpening is applied.
Compact sequence summaries aggregate every affected merge. Seam blending uses the worst residual,
gradient, and detail imbalance plus the least supported scale count. Exposure normalization retains
a common model or reports `sequence_mixed`, uses the largest normalized offset, and treats held-out
validation as passed only when every non-identity merge passed its own disjoint gate. Fixture and
baseline contracts can pin those aggregate decisions. OpenCV supplies only Lowe-filtered SIFT
correspondences. Source-overlap cells are divided as a 4x4 whole-cell checkerboard; normalized DLT
and deterministic RANSAC fit the composition transform only on the training cells. The held-out
cells must satisfy cross-inlier and residual gates, a separately fitted held-out transform must
generalize back to the training cells, and the two fits must agree geometrically. The primary
training fit is then composed only after it also improves NCC and registration error over
translation in both independent image-domain checkerboard score partitions, with minimum sample
support and a meaningful deviation from translation. Independently cropped top borders are
retained as vertical coordinate origins for both pair and N-input ordering, so cropping cannot
silently move a real overlap outside the mechanical-drift search window.

Batch rendering writes a requested scene-referred master before tone mapping, then transfers the
owned scene array through creative white balance, exposure, tone mapping, local detail, vibrance,
neutral cleanup, shadow protection, and optional grain reduction in place. This removes the second
full-frame f64 RGB render allocation. Interactive rendering deliberately keeps its cached technical
master and uses a separate mutable render buffer so controls remain reversible. Tone-phase
`render_input_buffer_policy` and `scene_referred_master_preservation_policy` disclose which path ran.

Final artifact encoding is bounded too. RGB16 and RGB32-float TIFFs are converted and written one
approximately 1 MB strip at a time, while the sRGB proof is gamut-mapped and compressed one row at
a time through a 64 KiB PNG stream. The three artifacts remain sequential, so these conversion
buffers do not overlap. Save-phase `artifact_write_buffer_policy` records the strategy, every
declared buffer size, and confirms that no full-frame conversion buffer was used. The generated
sRGB profile also has a fixed profile-definition timestamp instead of the render clock, making
identical PNG proofs byte-reproducible while actual artifact/report timestamps remain auditable.

Modern-clean adaptive vibrance now includes a feathered D50 CIELAB skin-memory region derived from
[global spectrophotometric measurements](https://doi.org/10.1002/col.70012). It attenuates up to
90% of the optional vibrance boost inside the measured lightness/chroma/hue core, fades smoothly to
zero outside a wider support region, and reports the exact bounds plus evaluated/protected ratios.
It is intentionally not a face or semantic-skin detector: a false positive merely receives less
creative saturation, while the technical scene-referred master remains byte-independent of render
intent.

Modern-clean rendering coordinates a deliberately small preferred-skin shoulder inside the
adaptive-vibrance Lab pass, using the
aggregate skin image-quality ellipse published by [Ji, Tian, and Luo](https://doi.org/10.2352/issn.2169-2629.2021.29.170).
Pixels already inside that 50%-acceptability ellipse are unchanged. Outside pixels must also match
the broader four-continent measured L*C*h skin region and have more chroma than the published
preferred centre. The pass is one-way: it never raises chroma. It removes only 35% of
ellipse-radius excess, cannot cross the ellipse, preserves CIELAB lightness, and caps movement at
3 DeltaEab with a ProPhoto-gamut backoff. This is still an appearance proxy rather than face or
skin-group detection. It is disabled for natural-neutral, film-faithful, or untrusted-colour paths,
never changes the technical master, and reports its full population and effect accounting.

Sky and foliage use a different safeguard because psychophysical results place their preferred
reproductions at higher chroma than their naturalness centres. A second trusted-render-only guard
uses the published sky, spring-grass, and autumn-grass CIELAB image-quality centres and a*b*
acceptability ellipses from [Ji, Tian, and Luo](https://doi.org/10.2352/issn.2169-2629.2021.29.170).
It keeps boosts that travel toward the nearest supported preferred centre and only attenuates the
portion that would overshoot or move away. The gate is deliberately one-way: it never pulls a pixel
toward a memory colour, claims semantic recognition, or modifies technical reconstruction. Reports
expose overall and per-family matched/limited populations and the exact scale reductions.

The `tone_mapping` phase reports render-quality bands for shadows, midtones, bright neutral
candidates, and saturated bright pixels. Chroma protection remains luminance-based but is gated by
the selected colorspace quality diagnostics and reports `color_protection_reason`,
`color_trust_state`, `highlight_neutral_chroma_enabled`, and `shadow_chroma_enabled`. Key fields include
`tone_fit_policy`, `auto_exposure_ev`, `render_exposure_ev`, `shadow_saturation_median/p95`,
`midtone_luminance_percentiles`, `midtone_saturation_median/p95`,
`bright_neutral_saturation_median/p95`, `bright_neutral_rgb_median`, and
`bright_saturated_saturation_median/p95`. It also reports diagnostic high-frequency luma/chroma
residuals plus requested/effective `noise_reduction_*` fields and a structured `grain_reduction`
before/after/effect record when the optional bounded pass is requested. Grain reduction now protects
multiscale-coherent opponent-colour edges as well as luminance structure, and its 0.018-to-0.035
structure gate exactly bypasses pixels at the independent detail contrast floor. Reports expose the
gate and exact excluded ratio, while roll summaries aggregate the latter. Saturation protection is
an effective shadow-relaxed mask, and frame-level grain evidence damps only local high-frequency
chroma rather than the colour base. The pass then samples the same pre/post edge population to
report luminance and chroma median/p10 contrast retention. At least 64
probes support each channel decision; supported median retention below 0.90 or p10 below 0.70 lowers
tone confidence and makes the final status `review_required_grain_detail`. An image with no coherent
edge population is explicitly unsupported rather than falsely passed. Denoising defaults off and is
never implied by `render_intent`; strength/scale remain independently reproducible. Roll-suite
validation carries this evidence per frame, aggregates only evaluated/supported populations, and
flags lost support, newly harmful smoothing, or material p10-retention regression against a matched
roll baseline. Weak-neutral
colour candidates keep shadow chroma cleanup enabled while disabling only neutral-highlight cleanup,
so sparse-neutral frames can still suppress shadow colour speckle without claiming full colour
trust. The separate `white_balance` phase reports technical neutral support, spatial/luminance-band
coverage, log-chroma dispersion, estimated source white/CCT, Bradford matrix, confidence, and review
state. Creative temperature/tint is reported under `tone_mapping.creative_white_balance` and is
never baked into the scene-referred master. Reports saved from `scanstitch-ui` additionally record
the applied creative white-balance, tone, exposure, and grain controls.

The bounded display render uses a D50 CIELAB perceptual gamut mapper that preserves lightness and
hue while reducing only the chroma needed to enter the output gamut. Its affected ratio and chroma
scales are reported. The 32-bit scene-referred master remains signed and preserves highlight values
above 1.0. For negatives, automatic direct-density selection is rejected when the film-base
confidence is fallback quality; the candidate may still be reported, but cannot replace ICA or
drive detail fusion without a credible base measurement.

The top-level report metadata records `generated_at`, binary/package version, working directory,
CLI args, and the intended output path. The `save` phase records final output dimensions,
modification time, overwrite status, and stale TIFF artifacts left from earlier runs.

See `docs/report-schema.md` for stable report fields and diagnostic fields.
See `docs/grain-investigation.md` for the current LOGAN grain comparison and render-denoise policy.
See `docs/validation-corpus-status.md` for the current boundary between runnable real-scan evidence
and the ground truth still required to certify negative colour fidelity.

## Real-Image Validation

`scanstitch-validate` runs or summarizes local validation fixtures and emits compact JSON/Markdown
summaries without versioning rendered TIFFs.

```powershell
cargo run --bin scanstitch-validate -- --fixture logan
cargo run --bin scanstitch-validate -- --report output/validation/logan/report.json --fixture logan
cargo run --bin scanstitch-validate -- --report output/report.json --fixture logan --compare-report output/validation/logan/report.json --strict
cargo run --release --bin scanstitch-validate -- --fixture logan --compare-summary tests/fixtures/baselines/logan_summary_baseline.json --strict
cargo run --release --bin scanstitch-validate -- --fixture-registry fixtures.json --fixture-suite --strict --require-reviewable
cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --input-mode positive --bit-depth 16 --output-dir output/testroll_positive
cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --grain-reduction on --grain-strength 0.6 --grain-scale 1.0 --output-dir output/testroll_grain
```

Roll summaries retain each frame's authoritative final-render and rendered-tone decisions. A
non-reviewable delivery, failed tone-output gate, or missing decision evidence becomes a roll review
issue; use `--fail-on tone` to make tone-specific roll and comparison issues fail automation.
Strict fixture suites likewise refuse to call an undeclared non-reviewable render passed. A fixture
that intentionally exercises a diagnostic path must pin `expectations.render_reviewable=false`;
normal acceptance fixtures should pin the full reviewable dynamic-range contract. Add validator
`--require-reviewable` when every direct, report-only, fixture-suite, or roll-suite output must be
reviewable. The gate runs after reports and compact summaries are written, fails closed on missing
or inconsistent final evidence, reopens the primary TIFF to verify dimensions, RGB16 storage, and
ICC evidence, reopens every requested float master and sRGB proof to verify its promised storage,
dimensions, and profile, recomputes every schema-v4 requested-artifact SHA-256, requires an explicit
zero stale-artifact count, and overrides an intentional diagnostic expectation.

See `docs/validation.md` for the LOGAN workflow, tracked summary-baseline gate, and delta-review fields.
Use `docs/validation-fixtures.example.json` as the template for local fixture registries;
`docs/validation-fixtures.schema.json` documents the machine-readable registry contract, and
`docs/validation-summary-baseline.schema.json` documents tracked compact summary baselines.
`min_stitch_normalization_contract_fixtures` counts only validation-ready accepted stitches that
pin a known exposure model and model-appropriate held-out result, a normalized-offset ceiling no
larger than 0.05, applied seam-aware multiband blending, no seam/detail review, at least two
supported detail scales, at most 2.0 detail-energy imbalance, at most 1.05 seam-gradient ratio, and
an overlap-p95 ceiling below 1. Partial contracts receive exact repair actions.
`min_geometry_preparation_contract_fixtures` counts only validation-ready scans that pin applied
deskew on every component with no review and at least 90% retained area, followed by an accepted
four-edge crop on every component with at least 50% retained area and an explicit upper bound below
1.0. This proves that geometry preparation happened safely; it does not by itself prove that the
angle or crop boundary is correct. `min_geometry_accuracy_contract_fixtures` additionally requires
an annotated expected deskew correction within a tolerance no wider than 0.05 degrees and one
top/bottom/left/right crop target for every component, each with a tolerance no wider than 4 pixels.
Strict suite runs compare every target with the compact runtime summary and report the exact
component and edge that missed its annotation. `min_orientation_accuracy_contract_fixtures`
separately requires one exact orientation annotation per component: `upright_approved: true`, the
SHA-256 of the normalized effectively oriented decoded pixels, source-tag presence/value, composed
named transform, whether either remapping ran, original dimensions, and final dimensions. A fixture
may declare `orientation_correction`. The registry rejects malformed hashes and internally
inconsistent declarations (for example, source tag 6 without a clockwise 90-degree effective
transform and swapped dimensions when no correction is requested), while strict execution rejects a changed
decoded digest or a coherent but factually wrong transform. For real acceptance, set the approval
flag and copy the digest only after visually approving that component as upright; the deterministic
fixture proves this binding machinery, not human judgment.

Generate a lightweight review package without running reconstruction:

```powershell
cargo run --bin scanstitch-validate -- --fixture-registry fixtures.json --fixture frame-001 --write-orientation-review output/orientation-review/frame-001
```

The directory contains one effectively oriented preview per component plus
`orientation-review.json` and Markdown. Draft approvals are always `false`. A single standalone
input is also supported with `--component1 scan.tif`; use `--input-mode positive` when appropriate.
The draft records the exact metadata-relative correction and composed transform.
The manifest format is defined by `docs/orientation-review.schema.json`.

After a perfect-mode render, generate a separate final-image review package:

```powershell
cargo run --bin scanstitch-validate -- --fixture frame-001 --report output/frame-001/report.json --write-render-review output/render-review/frame-001
```

For a grain-on render, an exact perfect-mode grain-off control is mandatory:

```powershell
cargo run --bin scanstitch-validate -- --fixture frame-001 --report output/grain-on/report.json --write-render-review output/render-review/frame-001 --render-review-grain-control-report output/grain-off/report.json
```

`render-review.json` starts with `review_status=requires_human_approval`, no reviewer, and every
applicable decision unapproved. It binds the ordered source files, orientation-materialized decoded
pixel digests, report, bounded TIFF, scene-referred master, and sRGB proof by SHA-256. Review the
sRGB proof for crop/colour/tone and inspect the full-resolution master/output at 100% for seam,
texture, sharpness, and grain. A grain-on schema-v2 draft additionally binds the off-control report
and all three control artifacts. The validator requires identical ordered inputs/decoded pixels,
matching render arguments after removing only output-directory and grain controls, a technically
reviewable off result, identical valid executable fingerprints, and byte-identical scene-referred
masters. Only then complete the applicable decisions with specific notes,
reviewer, RFC3339 review time, and overall notes. Add the completed manifest as
`render_review` plus its exact `render_review_sha256` to the fixture. Coverage requirement
`min_approved_render_review_fixtures` counts it only while the registry input order, source bytes,
decoded pixels, report, every promised artifact, technical reviewability, and manifest hash all
still agree. A human declaration cannot override a technically blocked render. The format is
defined by `docs/render-review.schema.json`; legacy schema-v1 non-grain manifests remain readable,
while a v1 grain approval without a bound control is deliberately insufficient. Free-form
`reference_evidence` labels remain corpus
metadata and are not substitutes for this approval contract.

`min_negative_reconstruction_contract_fixtures` is stricter than merely
running inversion: it
requires a measured base, accepted held-out-validated 3x3 dye separation and nonlinear PCHIP film
curves, bounded DeltaE00/noise/extrapolation, signed headroom, direct-density rendering, safe/trusted
colour selection, and linear ProPhoto output. Partial declarations receive exact repair actions and
strict fixture suites compare every pinned value with runtime evidence.
Registry fixtures require only `component1` for an independent scan. A real second scan uses
`component2`; three or more use `component2` plus `additional_components`. Direct validation,
fixture suites, roll-registry scaffolds, coverage, and hash snapshots preserve that one-or-more
input vector without dummy duplicates. The registry also supports complete-set SHA-256 gates,
per-fixture input/bit-depth/deskew/orientation-correction/force settings, and an optional exact
`inferred_component_order` expectation for sequence evidence. Fixtures can also pin grain on/off,
strength, and scale independently of global validator defaults; suite results retain the effective
settings beside the grain/detail outcome. Direct fixture and fixture-suite runs use the same
effective registry settings. Coverage requirements can demand both validation-ready grain-on cases
and complete two-channel detail contracts, then separately demand measurable applied-pixel and
flat-area luma/chroma residual-reduction floors. An all-off corpus, an enabled fixture with inherited
settings, or a detail-safe filter that does no useful work therefore cannot be mistaken for denoise
acceptance evidence. Fixtures can also pin scene-specific post-scale preservation, robust p05-p95
render-luminance span, render-to-mapped retention, tone evidence confidence, post-tone high/low
clipping limits, and the authoritative supported/reviewable tone and delivery decisions; coverage
can require complete `min_render_dynamic_range_contract_fixtures` evidence instead of treating a
stable diagnostic, collapsed, or clipped render as successful dynamic-range maximization.
See `docs/color-reconstruction-audit.md` for the current objective-to-artifact checklist and the
remaining real-corpus evidence gap.
See `docs/release.md` for local release builds and install commands.

## Benchmarks

`scanstitch-bench` runs lightweight local phase timing without adding benchmark dependencies:

```powershell
cargo run --release --bin scanstitch-bench -- --width 640 --height 360 --iterations 3
cargo run --release --bin scanstitch-bench -- --real-logan
```

The real LOGAN benchmark is opt-in and requires local TIFFs.

The full LOGAN stitch test is ignored by default because it requires local private TIFFs and runs
the full real-image stitch path. Run it explicitly when needed:

```powershell
cargo test --locked --release --test test_stitch test_real_sample_pair_uses_rgba8_load_path_and_accepts_narrow_overlap_stitch -- --ignored --test-threads=1
cargo test --locked --release --test test_base_detect test_real_testroll_curved_rebate_diagnostics -- --ignored --test-threads=1
```

## Architecture

```
src/
  main.rs                    CLI entry point
  bin/scanstitch-validate.rs Real-image validation harness
  bin/scanstitch-calibrate.rs Calibration-library record generator
  bin/scanstitch-ui.rs       Interactive render review UI
  bin/scanstitch-bench.rs    Local phase benchmark harness
  pipeline.rs                Phase orchestration and report assembly
  border.rs                  Phase 1: row-stats border detection
  base_detect.rs             Phase 2a: block chroma clustering
  frame_classify.rs          Phase 2b: intact/split classification
  stitch.rs                  Phase 2c: scoring, validation, blending, diagnostics
  density.rs                 Phase 3: density-domain inversion
  ica.rs                     Phase 4: streaming FastICA and channel normalization
  colorspace.rs              Phase 4.6: ProPhoto D50 mapping and fallback diagnostics
  tonemap.rs                 Phase 5: tone mapping and render diagnostics
  color_calibration.rs       Calibration profiles, libraries, fitting, and diagnostics
  positive_input.rs          Positive/slide input suitability checks
  validation.rs              Compact validation summary extraction and suite checks
  constants.rs               D50 white point, ProPhoto/Bradford matrices
  tiff_io.rs                 TIFF/DNG load/save and working-range normalization
  interactive.rs             Interactive render controls and sidecar support
  streaming.rs               Tiled iteration helpers
  report.rs                  JSON report structs
  cli.rs                     Clap 4 arg definitions
  debug.rs                   Debug image helpers
  cv_adapter.rs              OpenCV feature-gated adapter
```

Line and test counts are intentionally omitted here; use `rg --files src tests` and `rg "#\[test\]" src tests` for local counts.

## Tests

```bash
cargo fmt --check
cargo build --locked
cargo test -- --test-threads=1
cargo test --test test_border -- --test-threads=1
cargo test --test test_ica -- --test-threads=1
```

CI runs the serialized Rust test gate on Linux and Windows. The slow LOGAN stitch test remains
ignored by default because it requires private local TIFFs; run it explicitly as shown in the
benchmark section when validating real-scan behavior.

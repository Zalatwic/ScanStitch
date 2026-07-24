# ScanStitch Current Assessment

Date: 2026-07-23. Scope: current working tree plus representative local positive, negative, and
split-frame scans. This supersedes the 2026-06-03 assessment and the historical
`CLAUDE_FINDINGS.MD`.

## Verdict

No: the full product goal is not yet accomplished.

The codebase is a sophisticated, instrumented reconstruction pipeline, not a production-proven
automatic “beautiful image” system. Positive, correctly profiled scans can complete successfully
with four-edge crop, signed scene-referred headroom, a tagged archival master, perceptual display
gamut mapping, an sRGB proof, and optional independent grain reduction. Translation stitching can
order any number of inputs, retain unequal crop origins, preserve the complete union, normalize a
bounded exposure mismatch, choose a low-energy seam, and multiband blend it. Negative processing
has physically motivated density, scanner, roll, and film-response models plus conservative model
selection.

The decisive limitation is evidence: there is no pinned, representative ground-truth corpus with
scanner/film characterization and target colours. On the real LOGAN pair, geometry now succeeds
and the seam metric improves, but film-base confidence is zero and the negative render is not
reviewable. Current TESTROLL rerenders now also remove a formerly missed right rebate and apply an
auditable semantic 180-degree correction before geometry. Their proofs are border-free/upright.
RAW_0006 auto selection now rejects its former fluorescent matrix result in favour of a measured
neutral safety rescue, but that visibly safer proof remains warm and non-reviewable because film
response and dye crosstalk are unmeasured; RAW_0011 still has conspicuous uncalibrated colour. The
program says so: geometry or plausibility improvement is not called beautiful or accurate colour.

| Goal | Current state | Assessment |
|-|-|-|
| One or more positive/negative inputs | TIFF/DNG precision, orientation 1–8, ICC, DNG levels/matrices, and metadata are handled; an explicit metadata-relative D4 correction composes into scanner coordinates and rebinds decoded pixels; one input needs no dummy; load confidence exposes source-precision/layout/orientation/level limitations | Substantially implemented; semantic uprightness still needs a factual user/reviewer decision |
| Scanner-border crop | Four independent edges, gradual flare transitions, measured per-edge evidence confidence, explicit no-op/unsafe-rejection/unresolved-candidate provenance, and regression tests; real BIRDING047 bar and TESTROLL RAW_0006 right rebate removed; strict synthetic all-component deskew/four-edge-crop contract | Implemented, not corpus-proven |
| Any-number stitching | Validated directed graph, exact ordering through 12 inputs, deterministic larger-set fallback, origin-aware pair merges, complete-union failure semantics, and a strict measured three-scan normalization regression | Strong translation path; approved real N-scan and warp evidence still missing |
| Seam normalization | Spatially split training/held-out selection among identity, scalar/RGB gain, bounded RGB gain+offset, vertical, planar x/y, and regularized quadratic x/y gain or gain+offset fields; validity-masked full-union composition, minimum-cost seam, five-band bounded blend, residual/gradient plus disjoint multi-scale detail diagnostics across translation, affine, and accepted homography paths | Strong conservative compositor with fail-closed detail review; shading outside observed overlap support, corrective focus matching, and real approved seam/warp evidence remain incomplete |
| Negative inversion | Separate base measurement, density inversion, ICA/direct-density comparison, measured nonlinear dye-response support, truthful confidence gates, and a strict measured-response fixture contract | Partial; still fails without credible real base/calibration |
| Beautiful colour/tone | Candidate ranking, disjoint training/held-out scanner and roll matrix fits, reference-patch DeltaE00 gates, D50 adaptation, signed ProPhoto master, exposure/tone/local contrast, restrained vibrance, hue/lightness-preserving Lab gamut mapping | Technically complex but not absolutely validated or universally “beautiful” |
| Optional grain reduction | Explicit on/off, independent strength/scale, exact multiscale luminance/opponent-colour exclusion, effective shadow-relaxed saturation protection, high-frequency-only chroma damping, pre/post median/p10 retention gates, before/after effects, off by default | Implemented with fail-closed harmful-detail review; current real candidate is measured but still needs human approval and corpus admission |
| Production validation | Broad synthetic/unit/integration gates, strict synthetic colour suite, enforceable geometry/stitch/colour/range/grain contracts, exact executable identity, and schema-v4 exact-byte delivery binding | Real ground truth, signed provenance, and private-fixture CI missing |

## What is working well

- Input handling does not silently call unknown scanner RGB “linear ProPhoto.” Embedded positive
  ICC profiles are transformed; unprofiled positives require review. DNG black/white levels,
  `ColorMatrix1`, white inference/adaptation, and EXIF/DNG orientation are reported.
- Source orientation metadata, explicit semantic correction, and the effective transform are now
  separate audit records. Corrections occur before scanner linearization/deskew/crop/stitch, compose
  back into original scanner coordinates, and bind both the metadata-oriented and final pixel
  hashes. The validator can generate an effectively oriented preview whose approval remains false
  until a person accepts it.
- Archival and display domains are separated. `master_scene_referred.tiff` is 32-bit float linear
  ProPhoto D50 and keeps finite negative values and highlights above 1.0. The bounded output is
  tagged. The review PNG maps out-of-sRGB colours by reducing D50 CIELAB chroma at constant
  lightness/hue, adapts to D65, and embeds an explicit standard sRGB ICC profile instead of relying
  on channel clipping or an implicit display assumption.
- Current schema-v4 reports reopen the finished primary, scene master, and proof after encoding and
  record exact SHA-256 values with a bounded sequential 1 MiB buffer. Independent validation
  recomputes each requested digest; structurally valid same-shape/profile substitutions are
  rejected even when dimensions, storage, and ICC checks still pass.
- Final images, requested auxiliaries, reports, sidecars, calibration records, and validator review
  files now stage beside their destination and become visible by per-file atomic replacement only
  after successful encoding. Forced mid-TIFF failure preserves the previous file byte-for-byte and
  removes the temporary. This is intentionally reported as neither a cross-artifact transaction nor
  an `fsync`-backed power-loss guarantee, and overwrite staging needs temporary duplicate disk space.
- Human visual acceptance is no longer represented by a free-form label alone. The validator can
  generate a deliberately unapproved render-review manifest that binds ordered source-file and
  decoded-pixel hashes, the report, primary TIFF, float master, and sRGB proof. Corpus coverage
  counts a completed approval only while technical delivery remains reviewable, every applicable
  crop/seam/colour/tone/grain/preference decision has reviewer notes, registry inputs agree, and the
  final manifest SHA-256 matches. No real fixture has supplied that approval yet.
- Colour selection is unusually auditable: calibrated, scanner-prior, image-derived, neutral,
  ICA, and direct-density candidates expose safety/fidelity components, reference-patch CIEDE2000,
  hue-family regressions, clipping/headroom, neutral support, model plausibility, and rejection
  reasons. A more complex measured model is accepted only when held-out evidence improves.
- Uncalibrated auto mode now has a narrowly gated neutral safety rescue for the specific case where
  unsupported/unstable image-derived anchors create an extreme cast. It requires accepted neutral,
  memory-colour, spatial-neutral, chroma-retention, preserved-gamut, and saturation-reduction
  evidence, and cannot displace calibrated, scanner-prior, nonlinear, positive, or forced paths.
  It remains fallback-only at zero colour confidence and therefore cannot launder plausibility into
  trust.
- Scanner target fitting now evaluates homogeneous quadratic/cubic root-polynomial transforms and,
  with substantially more evidence, smooth residual 5x5x5/7x7x7 tetrahedral LUTs over the selected
  polynomial or matrix. Training-only regularization/robust fitting and disjoint CIEDE2000, XYZ,
  maximum-error, hue, support/occupancy, condition, coefficient/node, and LUT-roughness gates control
  complexity. The loader refits retained evidence and recomputes every claim. Runtime applies each
  nonlinear transform per pixel only when the scene remains inside its measured chromaticity and,
  for LUTs, RGB-volume support, and ordinary gamut/tone/perceptual ranking still selects it;
  otherwise the next simpler calibrated transform remains the explicit fallback. The LUT path is
  executable in code but no current real target corpus is large enough to qualify it.
- Scanner and roll matrix fitting now requires at least 12 uniquely identified training samples and
  12 disjoint held-out samples. Matrix/whitepoint estimation uses only training data; confidence,
  hue summaries, and worst-patch claims use only held-out data. Schema-v2 in-sample-only fit records
  fail closed. Both sets are now transformed through the exact runtime scanner linearization before
  matrix fitting; roll targets then use and audit the exact selected scanner matrix, root-polynomial,
  or residual-LUT colour model. Fitted records declare the signal domain and model ID, spatial models require
  original-frame patch coordinates, and mismatched scanner/roll domains fail closed. A new
  projective sampler extracts robust patch interiors from independently hashed training/held-out
  TIFF/DNG chart captures, emits overlays, and fails closed on clipping, contamination, or spatial
  nonuniformity. Corners still have to be entered manually and real target captures remain absent.
- Automatic direct-density selection now requires a credible film base. A fallback-base candidate
  may be reported for diagnosis but cannot replace ICA or drive detail fusion.
- Calibration reporting now separates record/component availability from final colour-mapping
  application. `calibration.color_mapping_application` is written only after final render-input and
  candidate selection, records the exact acceptance/rejection class and selected/preferred
  candidates, and is true only for accepted or explicitly forced calibration mappings. Strict
  validation cross-checks this record against the independent selection diagnostics; calibrated
  reference fixtures must explicitly require application rather than merely an `applied` record.
- Optional grain reduction now protects coherent luminance and isoluminant opponent-colour edges
  across the actual filter footprint, exactly bypasses structure at the independent 0.035 contrast
  floor, and measures pre/post contrast. Saturation limiting now affects the filter rather than only
  its label, and frame-level damping acts on high-frequency chroma residual instead of scaling the
  underlying colour base. A supported median below
  0.90 or p10 below 0.70 caps tone confidence and makes the final output
  `review_required_grain_detail`; insufficient coherent probes are reported as unsupported rather
  than trusted. Synthetic flat-noise, mixed-detail, isoluminant-edge, and adversarial erasure gates
  exercise the distinction.
- Rendered-tone diagnostics now affect trust instead of remaining informational. The tone phase
  compares input, fitted, and rendered p05-p95 luminance spans, fails closed on near-flat evidence
  or severe relative collapse, and rejects catastrophic per-channel clipping. The save phase copies
  the decision as `review_required_tone_output`, so a successfully written diagnostic TIFF cannot
  retain reviewable delivery confidence. These are universal safety limits, not a single aesthetic
  contrast target for fog, night, portrait, and landscape scenes.
- Stitch failure is truthful. An unvalidated component is not presented as a complete frame.
  Independently cropped top borders now establish expected vertical origins for pair and N-input
  search, fixing a real LOGAN regression (expected 39 px, found 41 px).
- N-input compact summaries now aggregate every pair merge's exposure correction instead of
  dropping sequence photometric evidence. Shared models remain named, mixed models are explicit,
  normalized offsets use per-channel worst cases, and non-identity held-out validation must pass on
  every affected merge. A deterministic shuffled three-scan pipeline fixture passes a complete
  normalization/blend/detail contract and fails when its detail-scale floor exceeds the measured
  result. This is strong executable evidence, not an approved real panorama.
- Translation composition is measurable: LOGAN used a 217 px overlap, produced the full
  11701×3671 union, and reduced seam-gradient p95 to 0.576 of the source discontinuity. Fixture
  expectations can require the blend and cap this ratio plus overlap p95 mismatch.
- Uncalibrated overlap normalization can now correct a spatially consistent additive black/flare
  mismatch, but only after robust training-window fitting and disjoint held-out windows show a
  material improvement over both identity and gain-only. Per-channel offsets are hard-bounded to
  5% of the sample range, clipping growth is rejected, and reports/fixture contracts expose the
  model, normalized offsets, held-out scores, window agreement, and rejection reason.
- It can also correct a per-channel vertical shading gradient supported across the overlap height.
  Robust window-level log-gain lines need three distinct rows in each split, reproduce their slope
  on held-out windows, beat the best accepted constant model, keep both endpoints within 0.80–1.25,
  and avoid clipping growth. Correction clamps at measured vertical endpoints.
- When both shading and additive flare vary vertically, a four-parameter-per-channel nonlinear
  field competes against every simpler accepted model. Training and held-out fits use disjoint
  signal-rich windows, need four represented rows, agree on material gain/offset slopes and center
  offsets, improve most held-out windows, keep all endpoint and clipping bounds, and must win by
  both absolute and relative margins before application.
- Planar x/y gain and gain+offset candidates now extend that hierarchy when the overlap supplies at
  least four represented rows and three columns in both whole-window splits. Training and held-out
  planes must reproduce horizontal slopes, centers, and direct window improvement; all four gain
  and offset corners remain bounded; and each 2D model must materially beat the best accepted
  constant or vertical model. Composition clamps x and y outside the measured rectangle, so this
  does not claim knowledge of unobserved full-scan vignetting.
- A regularized quadratic `[1,x,y,x²,xy,y²]` tier can now model curved overlap shading. It needs at
  least 18 windows, six rows, and five columns in each disjoint split; independently fitted
  curvature and complete fields must agree; and it must materially beat every accepted simpler
  model. A 9x9 support grid plus exact quadratic edge/interior extrema checks all physical bounds;
  gains remain within 0.80–1.25, offsets remain
  within 5%, clipping growth is rejected, and both coordinates clamp at measured support. The
  coupled 12-parameter-per-channel solve is skipped when gain-only residual cannot clear the
  complexity margin. Synthetic positive, adversarial train-only, interior-extrema, exact runtime,
  and end-to-end report/composition tests pass; no approved real curved-shading seam exists yet.
- Seam statistics retain the same nearest-rank values without fully sorting unrelated samples, and
  constant/vertical candidates reuse their per-row fields instead of recomputing identical
  exponentials for every pixel. Quadratic gain+offset fitting uses a deterministic bounded sample
  set and an evidence-based prefilter; current real release timing must be rebaselined before a new
  performance claim is made.
- Accepted geometry no longer implies an unqualified seamless result when scan detail differs.
  Luma high-pass energy at 1, 2, and 4 pixel radii is measured in disjoint 6x6 whole-cell
  checkerboards. A review decision needs at least four signal-rich windows per split, a repeated 2x
  imbalance, 75% direction consistency, and bounded train/held-out ratio disagreement; fewer than
  two supported scales fail closed as under-evidenced. Severe
  focus/grain/texture mismatch or seam-gradient amplification blocks final geometry review while
  retaining the composed union for diagnosis. Synthetic matched, repeated-blur, train-only
  adversarial, serialized-report, and full-pipeline review-propagation tests pass. The code does not
  invent an unvalidated blur or sharpening correction.
- The private full-resolution LOGAN stitch regression also passed this gate on 2026-07-18. It
  retained the pinned 41 px placement, 217 px overlap, 11,701x3,671 union, identity photometric
  model, and non-review detail decision with at least two supported scales. The debug test took
  750.03 seconds on this machine. This is useful false-positive evidence for one real pair, not an
  approved sharpness threshold or a substitute for varied scanner/film seams.

## Current impediments

### 1. No absolute ground truth

The largest blocker is not another heuristic. The repository lacks a validation-ready set spanning
real scanners, settings, film stocks, scenes, exposures, targets, and approved geometry. Existing
private rolls prove readability and exercise code paths; they do not establish correct colour.
Without target XYZ/Lab values or an approved reference, a warm/magenta positive cannot be labelled
accurate, and a visually preferred negative cannot be distinguished from a pleasing error.

Needed: pinned source/profile/baseline hashes; scanner target captures; per-roll rebate/base;
identified stock/process; ColorChecker/gray references; approved crop/orientation/seam masks;
scene/exposure diversity; and grain/detail regions with acceptance bounds. See
`docs/validation-corpus-status.md`.

The new render-review package and `min_approved_render_review_fixtures` gate make those human
judgments reproducible and invalidate them after any source, decode, report, delivery-artifact,
input-order, or review-text change. Grain-applicable schema-v2 reviews now also require a matched,
hash-bound, technically reviewable grain-off report and artifacts, identical decoded inputs and
normalized non-grain render command, identical valid executable SHA-256/size identity, and a
byte-identical scene-referred master. They deliberately do not approve their own drafts, so the
remaining real crop/seam/beauty decisions still require a curator.

The schema-v4 delivery hashes close accidental/stale/single-artifact substitution, but not hostile
provenance. `report.json` has no digital signature or append-only external attestation, so an actor
who can rewrite the report can also replace an artifact and update—or downgrade—the declaration.
If untrusted storage or adversarial chain-of-custody is in scope, release manifests and reviewer
identities still need signing or an externally trusted transparency record.

Semantic uprightness is one of those factual decisions. EXIF/DNG tags describe storage layout and
cannot reliably infer whether a golfer, face, sign, or horizon should be upright. The explicit
correction removes the former processing limitation without pretending that content understanding
is ground truth; the RAW_0011 draft is visibly useful but remains `upright_approved: false` until a
human reviewer signs the hash-bound result.

The new grain self-retention proof closes a concrete luma-only failure mode, but it is not absolute
grain truth. Coherent fine grain and coherent subject texture can overlap spectrally; without
approved grain-on/off regions, the program cannot prove the preferred denoise strength or guarantee
that every visually important microtexture is preserved. A current hash-bound BIRDING047 pair now
demonstrates a selective 0.5-strength pass with 50.674% exact exclusion, 48.063% active mask,
6.368%/14.887% median luma/chroma-residual reduction, and 0.992/~1.000 p10 coherent-detail
retention. Its grain/detail decision remains false under `requires_human_approval`, so this closes
the earlier near-universal-mask implementation defect. Its fresh v2 draft binds the exact off/on
pair and common master, closing the stale-control approval loophole without pretending to supply
the missing preference judgment.

### 2. Negative colour remains conditional

Negative reconstruction is only physically defensible when scanner linearization, film base, dye
crosstalk/response, and output characterization are known or measured. ICA remains a blind,
frame-derived separation and is not proven by real held-out colour targets. Direct density is more
physical only when its density reference and response are credible. LOGAN demonstrates the gap:
base confidence 0.000 makes the output non-reviewable.

The correct next improvement is measured evidence, not greater unconditional model complexity.
The calibration format, robust held-out fitting machinery, and projective chart sampler exist, but
populated trustworthy scanner/roll records do not. The sampler still requires manually declared
chart corners and factual reference values. It now emits overlays plus a paste-ready review
manifest, refuses to write measurements until each footprint is explicitly approved, binds that
approval to both the exact orientation-materialized decoded-pixel hash and the ordered corner
coordinates, pins the exact rendered overlay pixels, and invalidates it after any pixel, precision,
orientation, geometry, grid-layout, or renderer change. It now separately requires an approved
digest of every patch ID, position, and reference XYZ/Lab value, retains those values in fitted
records, and recomputes the digest during loading. It still cannot manufacture the missing independent target
captures or approve their colour accuracy. The nonlinear polynomial/LUT paths are therefore
executable in code but unqualified for the available real scans; synthetic selection tests are not
colour truth.

### 3. Stitch geometry and normalization are incomplete

- Default translation is robust. Auto/affine mode now detects coherent local displacement,
  searches a bounded small-angle similarity seed, robustly fits a six-parameter affine model, and
  accepts it only when a disjoint held-out overlap improves. The accepted path uses a single
  bicubic resample, full valid-union bounds, exposure compensation, and seam-aware multiband
  blending. A candidate at the exact rotation-search boundary is rejected as unconverged; this
  prevents the real LOGAN pair's unsupported 3 degree/zero-local-inlier candidate from replacing
  its validated 11,701 by 3,671 translation union.
- Homography feature extraction is optional and still depends on native OpenCV, but OpenCV now
  supplies only Lowe-filtered SIFT correspondences. A pure-Rust normalized-DLT/deterministic-RANSAC
  estimator fits only 4x4 checkerboard training cells. Unseen whole cells must pass forward
  cross-inlier/residual gates; a reverse held-out fit must generalize to training cells; and the two
  transforms must agree before model selection can continue. The accepted compositor matches
  native affine: bounded full-union canvas, explicit warp validity, one pure-Rust bicubic resample,
  overlap-derived photometric selection, a largest fully valid multiband rectangle, and bounded
  feathering only for irregular overlap remnants. The candidate must additionally beat translation
  on both image-domain checkerboard NCC/error partitions and differ meaningfully from translation.
  Synthetic positive, outlier, determinism, and cell-specific adversarial regressions pass. The
  Ubuntu 24.04 native feature path now compiles, links, and passes the feature-enabled library suite
  against Clang/LLVM 18 and OpenCV 4.6, including runtime SIFT and RGB16 warp smoke coverage. CI has
  a mandatory matching job. No approved real projective pair yet proves the geometry or acceptance
  thresholds, and Windows still requires a separately installed compatible native toolchain.
- Relative small-angle rotation is corrected when overlapping scans provide evidence. A single
  scan now also has bounded absolute deskew: robust top/bottom/left/right border lines are fitted
  independently, auto correction requires mutually agreeing opposing or cross-axis support, and
  one bicubic resample is followed by a largest-valid-rectangle crop with retained-area metrics.
  Manual signed source-skew and off modes are explicit. Real rotated-scan ground truth is still
  missing, and automatic absolute deskew intentionally defers multi-input sets to overlap geometry.
- Full-resolution render memory is materially improved: the post-tone display passes now reuse one
  owned f64 RGB working allocation instead of cloning a roughly 1.05 GiB LOGAN frame at every pass.
  Delivery no longer allocates full-frame RGB16, RGB32-float, or RGB8 conversion images either:
  TIFFs stream approximately 1 MB strips and the PNG streams one row through a 64 KiB chunk. The
  11,701 by 3,671 LOGAN-sized regression declares a 1,123,296-byte peak instead of the former
  515,452,452-byte float-master conversion allocation, a 99.782% reduction, and crosses multiple
  strips while checking exact decoded pixels. The real 5,630 by 3,685 BIRDING rerender declared
  1,013,400 bytes instead of 248,958,600 (99.593% less); its primary and master stayed byte-identical
  and its differently chunked PNG decoded pixel-identically.
  A fresh current-code full LOGAN pipeline completed all 13 phases in 274.7 seconds and saved the
  11,701 by 3,671 output on this machine. Its separately written candidate baseline records the
  identity seam model and rejected planar models, but comparison to the tracked 5,959 by 3,670
  baseline still has 17 review-required drifts. The tracked baseline was not silently refreshed.
- When a held-out-validated scanner record exists, black/white, additive flare, nonlinear response,
  and bounded per-channel spatial shading are now applied independently to every full decoded scan
  before border detection or stitching. EXIF-oriented pixels are mapped back to original scanner
  coordinates for the spatial polynomial, and a later calibration-selection change is rejected
  rather than applied in mosaic coordinates.
- Without that measured record, overlap normalization may estimate bounded constant, vertical,
  planar x/y, or regularized quadratic x/y per-channel gain/offset relations only when spatially
  disjoint held-out seam evidence justifies each added dimension. The inferred field is clamped to
  its observed support; full-scan vignetting and every region outside the overlap remain
  scanner-calibration work.
- Detail-band diagnostics can now detect a likely focus, grain, or sharpness transition in the
  observed overlap and fail closed. They cannot reconstruct detail missing from one complete scan,
  distinguish optical blur from scanner noise without target evidence, or prove thresholds on an
  approved real seam. Corrective deconvolution/sharpening therefore remains deliberately absent.

### 4. “Most beautiful” is not a closed objective

The renderer has sophisticated exposure, tone, local-luminance detail, restrained vibrance, neutral
cleanup, and perceptual gamut mapping. Adaptive vibrance applies a feathered D50 CIELAB skin-memory
protection region based on global spectrophotometric bounds. It now also uses published
psychophysical sky, spring-grass, and autumn-grass image-quality centres/acceptability ellipses as a
one-way overshoot guard: a supported boost may continue toward the nearest preferred a*b* centre,
but the portion moving away is attenuated. A separate modern-clean skin shoulder may pull only a
high-chroma outlier toward the published aggregate skin-preference ellipse: it also requires broad
measured skin-colour support, leaves the 50%-acceptability core and lower-chroma pixels untouched,
preserves L*, cannot raise chroma or cross the core, and is capped at 3 DeltaEab. These are aggregate
colour-neighbourhood proxies, not semantic object/face/skin-group detection or calibration. They
touch only the finished display render, report their populations and effects, and strict validation
rejects contradictory provenance, hierarchy, trust, cap, or one-way-chroma claims. Those are strong
ingredients, not a proof of optimal taste.

The rendered-output gate can detect near-zero/relative tonal collapse and catastrophic clipping,
but it cannot decide the preferred contrast distribution for a scene that remains technically safe.
There is no user/curator preference set, reference print intent, or multi-style perceptual study.
Technical and creative white balance are now explicit: an evidence-gated or manual Bradford stage
produces the signed D50 technical master, while reversible creative temperature/tint lives only in
the finished render/UI/sidecar. Creative rendering is still not a complete photographic grading
model: there is no curator-trained style objective, selective HSL/secondary correction, or local
masking model validated against preferred reference prints.

### 5. Validation and operational readiness

- Real private scans are absent from hosted CI; the slow LOGAN test is ignored by default.
- Hosted CI now explicitly runs the calibration, interactive, scanner-linearization, and technical/
  creative white-balance suites in addition to the existing pipeline, stitch, and validation gates.
  The private release-mode LOGAN test passes and pins its unequal crop origins, 41 px placement,
  217 px overlap, and evidence thresholds; the private `RAW_0000` gate pins a no-op opposing-border
  deskew, strong separately retained pre-crop rebate, and zero-confidence post-crop fallback. These
  remain local evidence because the sources cannot be hosted.
- The fixture registry now represents one input with only `component1`, adds `component2` only
  when a second scan exists, and accepts complete N-component sets after that. Direct overrides and
  roll scaffolds no longer need a duplicated dummy path. Coverage treats a singleton as a real
  component set without inflating pair counts, while existing pair and shuffled three-scan suites
  retain their behavior. Every actual component can be hash-pinned, the full vector reaches direct
  and suite runs, measured one-based inferred order can be gated, and complete stitch-normalization
  contracts are required instead of counting an accepted decision alone. The remaining gap is
  evidence supply: no private three-or-more-scan real fixture is present in hosted CI yet.
- Baselines are mostly regression references, not truth. A bad render can remain stable.
- Grain-on roll suites now preserve per-frame coherent-detail evidence and aggregate only evaluated,
  channel-supported populations. Matched roll comparisons fail on changed denoise enablement, lost
  luminance/opponent-colour support, new detail review, or material p10-retention drops; older roll
  summaries remain readable. This makes a future real grain/detail corpus enforceable, but does not
  supply the missing approved regions or preferred-strength judgments.
- Fixture registries can now pin grain on/off, strength, and scale per case; direct runs and suites
  share fixture-first resolution, coverage reports preserve declared controls, suite output records
  the effective controls, and generated roll scaffolds snapshot the current values. This removes
  ambient CLI defaults as a source of grain-baseline drift and allows mixed on/off acceptance
  corpora. Coverage now separately gates validation-ready grain-on cases, complete two-channel
  preservation contracts (nonzero strength, pinned scale, support/no-review, 64 probes, and 0.70 p10
  floors), and effect contracts with positive applied-pixel, exact structure-excluded, and flat-area
  luma/chroma residual-reduction floors. Thus an all-off, under-specified, effect-free, or
  near-universally masked corpus cannot claim denoise readiness. It still does not establish which
  settings are photographically preferred.
- Strict fixtures can now turn the existing tonal-latitude diagnostics into a complete dynamic-range
  contract: an explicitly reviewable delivery with supported/no-review tone status, positive tone
  evidence confidence, positive fitted-to-rendered range retention, positive post-scale
  preservation and robust p05-p95 luminance-span floors, plus bounded post-tone high/low channel
  clipping. Coverage counts only complete validation-ready contracts and reports partial-field
  repairs. A deterministic calibrated pipeline fixture passes conservative measured bounds and
  fails over-tight absolute and relative span floors. The remaining limitation is aesthetic ground truth:
  each scene class still needs an approved reference rather than a universal contrast target.
- Stitch coverage now separately counts only complete, validation-ready normalization contracts:
  accepted geometry, known/model-appropriate exposure evidence, a <=5% normalized-offset ceiling,
  applied seam-aware multiband blending, no review, at least two detail scales, <=2.0 detail-energy
  imbalance, <=1.05 gradient ratio, and a sub-1 overlap-p95 ceiling. Partial contracts receive exact
  field repairs. The synthetic three-scan strict pass/rejection proves enforcement; real approved
  exposure/focus/sharpness seams remain absent.
- Geometry coverage now separates preparation from accuracy. A complete validation-ready
  preparation contract requires applied deskew on every component, no review, at least 90% deskew
  retention, four removed crop edges on every component, at least 50% crop retention, an explicit
  upper bound below 100%, and no rejected crop. The stricter accuracy count additionally requires an
  approved signed correction within <=0.05 degrees and every component's top/bottom/left/right crop
  target within <=4 pixels. Compact summaries expose the necessary per-component evidence; a
  deterministic -0.5-degree/16-pixel-board fixture passes and an out-of-tolerance edge adversary
  fails exactly while structural coverage remains complete. This proves enforcement mechanics, not
  accuracy on a real rotated scan with an approved crop mask.
- Orientation now has compact per-component evidence and a separate exact coverage contract for an
  explicit upright approval, deterministic decoded-pixel SHA-256, tag presence/value, named
  transform, application, and source/output dimensions. The digest includes normalized samples,
  dimensions, channels, and working precision. An orientation-6 TIFF passes end to end; a coherent
  tag-8/wrong-digest declaration remains structurally complete but fails exact runtime comparison.
  This binds code output to a review decision, but the current fixture is synthetic; semantic
  uprightness remains unproven until a real component is visually approved and its digest pinned.
  A standalone validator mode now writes orientation-materialized previews, Markdown, and a
  schema-checked paste-ready draft for one or N inputs without running reconstruction. It always
  emits `upright_approved: false`, so tooling cannot silently manufacture the remaining decision.
- Negative-response coverage now counts only negative fixtures with a credible measured base and an
  accepted held-out-validated roll response: 3x3 dye separation, monotone nonlinear PCHIP curves,
  model identity/confidence, bounded CIEDE2000/noise/extrapolation, signed headroom, direct-density
  rendering, and safe/trusted ProPhoto colour output. A deterministic strict fixture passes and an
  over-tight measured-confidence adversary fails while structural coverage remains distinct from
  runtime evidence. No current real roll target supplies this contract, so it does not approve the
  LOGAN or `OLD_TESTROLL` negatives.
- Full-roll plausibility inventory and debug builds are too slow for routine gates. The fresh
  current-code full-width LOGAN run took 274.7 seconds, with colour candidate evaluation the largest
  phase at 83.5 seconds.
  The main CLI now emits a configurable 30-second heartbeat with elapsed time and the last phase
  completed in the current run's partial report, and its final summary prints phase durations.
  An opt-in `--require-reviewable` delivery gate preserves completed artifacts but exits nonzero
  unless the final status and boolean both agree that the render is reviewable; it fails closed on
  missing or inconsistent evidence.
  Fine-grained progress inside a single long phase is still unavailable. The positive-only
  base/classification waste was removed during this audit: BIRDING047 fell from a 48.8-second phase
  total to 11.4 seconds (`base_detect_classify=0 ms`) with byte-identical TIFF/PNG outputs. That
  bypass now reports `classification_evidence_evaluated=false`,
  `classification_status=not_applicable_positive_input`, and phase confidence `0.0`; successful
  control flow is no longer misrepresented as perfect measured classification evidence.
- Border removal no longer emits unconditional phase confidence `1.0`. Applied-edge confidence is
  bounded by the weakest normalized strong-line, luminance-separation, and boundary-transition
  signal, then aggregated by the weakest applied edge/component. A clean no-op reports
  `no_crop_no_convincing_dead_zone`, `applied_crop_evidence_supported=false`, and confidence `0.0`;
  unsafe full-image proposals and too-small inputs retain explicit review provenance. Crop geometry
  and its existing strict pixel-accuracy fixtures are unchanged.
- Other not-applicable paths now follow the same rule. A one-input run reports
  `skipped_single_input` with `confidence_basis=not_applicable_single_input` and stitch confidence
  `0.0`; positive-input density inversion and ICA, plus ICA bypass under explicit direct-density
  rendering, report `evidence_evaluated=false` and confidence `0.0`. Automatic topology-based
  no-stitch decisions retain measured evidence, while forced controls identify their
  `user_authoritative_override` provenance instead of conflating it with automatic measurement.
- Positive mode now fails closed at delivery when its existing channel-median/orange-mask probe
  identifies strongly negative-like input. The requested positive render is retained for diagnosis,
  but working confidence becomes `0.0` and final status becomes `review_required_input_mode`;
  `--require-reviewable` rejects it. A separate `accepted_warm_positive_input` regression proves a
  high warm score with mild ratios remains accepted rather than treating warmth alone as a negative.
- Load confidence now represents the weakest component's declared precision, channel-layout,
  orientation, and DNG-level evidence rather than successful file I/O. RGB16 preserved in a 16-bit
  work domain reports `full_declared_decode_fidelity` and `1.0`; RGBA8 expanded for the default
  14-bit work domain reports `8/14` with
  `source_precision_upscaled_without_added_information`. Upscaling remains supported, but is no
  longer presented as recovering tonal information that was absent from the source.
- Downstream trust now remains consistent through delivery. A `fallback_only` colour candidate is
  review-required and cannot re-enter tone mapping as trusted; review-required colour gives both
  colorspace and tone confidence `0.0`. Disabled technical white balance reports non-evaluation at
  `0.0`, while a manual setting is explicitly `user_authoritative_override`. Save `success` records
  artifact I/O, but save confidence is `1.0` only for `reviewable_delivery`; retained diagnostic
  renders report `diagnostic_delivery_requires_review` and `0.0`.
- Blind FastICA no longer turns optimizer convergence into perfect physical confidence. Executed
  ICA reports numerical convergence separately, keeps
  `physical_separation_evidence_evaluated=false`, and uses phase confidence `0.0`; the later
  ICA/direct-density colour-quality comparison remains the place where render evidence is judged.
  Base-limit provenance now also says a phase was limited only when the base factor actually lowered
  its pre-limit confidence.
- Density inversion no longer lets a strong film-base estimate imply that an unmeasured negative
  response is trustworthy. Phase confidence now uses
  `minimum_film_base_and_negative_response_model_evidence`: a held-out measured response keeps its
  declared confidence, while frame-derived and unit-slope response models report `0.0` physical
  response confidence and remain diagnostic-only. Successful logarithmic inversion is retained as
  processing output, not misreported as colourimetric proof; actual measured-curve coverage can
  lower trust later during reconstruction.
- Negative-response trust now propagates through the whole colour/tone chain. A held-out measured
  response that covers the actual frame carries the density phase's weakest base/model confidence
  into colours and tone; blind ICA, an unmeasured response, or
  `review_required_measured_response_outside_runtime_curve_support` contributes `0.0`. The report
  separately preserves mapping-only confidence and says whether reconstruction actually limited
  it, so a mapping already at zero cannot assign the same failure twice. A deterministic varied
  negative chart proves both an in-support `0.94` chain and a narrow-curve diagnostic-only result.
- Tone confidence now includes rendered-output evidence. `supported_render_tonal_distribution`
  preserves otherwise supported confidence; near-flat/fitted/render collapse or catastrophic
  clipping produces `review_required_tone_output_evidence`, and final delivery becomes
  `review_required_tone_output`. A trusted positive ICC pipeline fixture remains fully reviewable,
  while deterministic helper cases pin supported, collapsed, clipped, flat, and invalid outcomes.
- Compact validation now preserves that decision rather than reducing it to a raw luminance span.
  Summary JSON/Markdown carries the tone-output status, review flag/reason, evidence confidence,
  input/fitted/rendered range relationship, and clipping maxima. New tracked baselines retain those
  fields; report and baseline comparisons emit named status/confidence/range/clipping regressions,
  while missing legacy fields remain readable and non-asserting.
- Roll validation now preserves the same tone-output record on every frame and aggregates final
  render states, tone statuses, and review reasons. A non-reviewable render, tone-output review, or
  missing decision evidence makes the frame review-required even if candidate-risk checks happen to
  pass. Matched roll comparisons report lost render reviewability, changed tone status/review,
  confidence/range-retention loss, and increased high/low clipping; old baselines remain
  non-asserting only for evidence they never contained.
- Fixture-suite delivery acceptance now fails closed without discarding evidence. A non-reviewable
  render cannot remain a strict passing fixture merely because its report is stable: the registry
  must explicitly pin `render_reviewable=false` to identify a diagnostic test case. Validator
  `--require-reviewable` is the stronger production gate for direct reports, fixture suites, and
  roll suites; it requires the authoritative reviewable status/boolean pair even for fixtures that
  intentionally expect diagnostic output, and runs only after TIFFs, reports, and compact summaries
  have been retained. The gate now independently reopens the primary TIFF for dimensions, RGB16
  storage, ICC agreement, and schema-v4 SHA-256; reopens every promised RGB32-float ProPhoto master
  and RGB8 sRGB proof for dimensions, storage, description/profile, colour-space agreement, and
  schema-v4 SHA-256; and requires explicit zero stale artifacts. A report-only reviewable claim
  cannot approve a missing, mismatched, or structurally valid substituted delivery file, master, or
  proof. This is exact-byte integrity, not authentication: the report is unsigned, so coordinated
  report/artifact edits or a schema downgrade remain outside the gate's trust boundary.
- On 2026-07-23, formatting, all 16 repository JSON documents, diff hygiene, all-target compilation,
  and warnings-as-errors Clippy passed. The final non-incremental all-target inventory listed 503
  tests; the complete run finished in 587.1 seconds, with 501 passed, zero failed, and the same two
  intentionally ignored private real-corpus tests. The validator target passed all 88 tests,
  including schema-v4 exact-byte delivery and structurally valid substitution rejection.
  External Draft 2020-12
  validation also accepted a legacy positive, the prior/current positive, two current negative,
  and tracked-baseline JSON artifacts.

## Real evidence from this audit

- Positive BIRDING047: 5959×3946 RGB16 source; output 5630×3685; crop removed 122 top, 139 bottom,
  187 left, and 142 right pixels. The prior black scanner strip is gone without obvious subject
  loss. Absolute deskew estimated a -0.0042° robust aggregate and correctly made no resample. The
  adaptive vibrance evaluated 58.112% of pixels; its measured skin gate protected 23.491%, while
  the published sky/grass preference ellipses matched 35.134% and limited 7.155%, with mean/max
  removed boost 0.00167/0.08048. The added one-way preferred-skin shoulder evaluated 79.126%,
  matched 21.708%, found 16.472% outside the published core, and adjusted 3.137%. Mean/max movement
  was 0.903/3.000 DeltaEab, mean chroma delta was -0.604, and no pixel needed gamut backoff. The
  technical scene master remained byte-identical at SHA-256
  `b72fb7e908aecf147c628adce8711accae21a91598c17aae9756f34e4af30625`; the final primary is
  `fb567932c08923007e189642b4edcb8fc3b64f80441308cb206e52bd671a03f8`. Against the preceding
  preferred-memory render, 3.135% of RGB16 primary pixels and 2.463% of RGB8 proof pixels changed;
  a deterministic parallel implementation is byte-identical to the serial fused primary and cut
  tone time from 17.041 to 9.433 seconds (7.080 seconds before this shoulder existed). Strict
  `--require-reviewable` validation retained intact primary/master/sRGB artifacts with coherent
  model accounting and zero stale files. Direct edge sampling found no near-black edge strip. The
  proof remains visibly warm/magenta and grainy and has no clear sky/foliage target. A fresh
  schema-valid render-review draft binds the final source, decoded pixels, report, primary, master,
  and proof hashes, but all five applicable human decisions remain false under
  `requires_human_approval`. Thus deskew accuracy on a truly rotated real scan, semantic object
  accuracy, colour/skin rendition, and aesthetic preference remain uncertified.
- Positive BIRDING047 grain-on control: the exact hash-bound release at strength 0.5/scale 1.0
  materially selected 48.063% of pixels and exactly excluded 50.674% for multiscale structure or
  incomplete boundary support. Median luma/chroma residual fell 6.368%/14.887%; conservative
  flat-area p95 fell 0.928%/0.665%. Luminance/chroma p10 retention was 0.992/~1.000 across
  38,647/14,622 coherent probes, with no detail review. Relative to the exact grain-off render,
  49.089% of RGB16 pixels changed but the all-channel median delta was zero, signed mean RGB drift
  stayed near 1e-5, and the technical master remained byte-identical. The largest interior change
  was an isolated chroma outlier surrounded by neutral white; exact boundary bypass removed the
  unsupported top-row extrema. Strict delivery/diagnostic validation and a schema/hash-reopened
  review draft pass, but `grain_and_fine_detail` and all other applicable human decisions remain
  false. This is a real selective-effect candidate, not an accepted denoise preference reference.
  A subsequent artifact-streaming-only release, SHA-256
  `1765caf7ae78a2d2ef72e5a673e5d9da4593eb276bbfeaa6d741d4e4dcf8379b`, rerendered the same grain-on
  case in 55.0 seconds wall time with a 1,599.6 MiB monitored peak working set and a 22.021-second
  save phase. The RGB16 primary and RGB32 master retained the exact hashes above; the PNG container
  hash changed while all 20.7 million decoded RGB pixels and standard-sRGB profile semantics
  matched. A later byte comparison found that the 612-byte ICC payload still carried the profile
  generator's current hour/minute/second, so it was valid but not byte-reproducible. Independent
  `--require-reviewable` delivery validation passed. The timing is one run, not a controlled speed
  claim; the bounded conversion allocation is the regression-backed result.
  The superseding owned-batch/canonical-profile release, executable SHA-256
  `c6442ce0bb5e94c85f0a585ef5d686b03f3e6fbb527d7f02826969d3063d8de2`, ran the identical case twice
  under `positive-birding047-owned-batch-canonical-icc-{a,b}-20260721`. The independent runs took
  45.522/44.644 seconds and peaked at 1,124.5/1,124.6 MiB, about 475 MiB (29.7%) below the streamed
  baseline by reusing the owned scene array after prewriting the master. Both report the same
  1,013,400-byte peak artifact-conversion buffer. All three deliverables are byte-identical between
  runs: primary `7db2c9021e77d93089ff92e30e82c1e9792949396f757d52c9f5d2aea99ab432`, master
  `b72fb7e908aecf147c628adce8711accae21a91598c17aae9756f34e4af30625`, and PNG
  `c4558f5594737c83055272f574c954875504f251b3d21713d342b2a88f33c47a`. The canonical 612-byte ICC
  payload is `0f303d7fb11d811c9751321ea472f4f024781dcdf0143ac7d94cebd1f0719323`; its fixed definition date
  is 2026-05-08 00:00:00 and the PNG IDAT bytes remain identical to the preceding owned render.
  Both packages independently pass `--require-reviewable`. These are reproducibility and bounded
  memory results, not a controlled speed comparison or a human aesthetic approval.
  The schema-v4 artifact-binding release, exact renderer SHA-256
  `ff236561e2c5897c190d8f5c58b8b1803cea6a47808b224febaa9c2e22130f76`, then rerendered the case
  under `positive-birding047-schema-v4-artifact-hash-20260721`. It completed in 49.173 seconds,
  peaked at 1,124.7 MiB working set/1,123.9 MiB private memory, and spent 19.246 seconds in save,
  including sequential post-encoding SHA-256 passes with a 1,048,576-byte buffer. All declared
  hashes match independent file hashes and the three artifact digests remain the exact
  primary/master/PNG values above. Exact validator SHA-256
  `763695c07a5b78bf2ee7e368685e5d0547dd9a674667842c903612bfe564b22c` independently passes
  `--require-reviewable` with all digest matches true. Integration regressions substitute valid
  same-shape/profile primary, master, and proof files one at a time: structural checks remain true
  and every altered byte set is rejected by its named SHA mismatch.
- Negative TESTROLL `RAW_0006`: the old run removed 143/116/165/0 pixels
  top/bottom/left/right and visibly retained the right scanner/rebate strip. The new conservative
  gradual-boundary fallback retained the same first three edges and removed 77 right pixels from 71
  strong columns. Its 24,575 luminance gap was 40.6× the 605 threshold; the 228.8 aggregate boundary
  change was 45.4% of the 504.4 derivative threshold. The 5710×3685 proof has no visible right strip,
  all four edges are reported, geometry review is false, and tone evidence retains a 0.863 p05-p95
  span with zero high/low clipping. The former auto result selected
  `gamut_stabilized_image_matrix_blend`, preserved only 0.741150 of pre-scale samples, and had a
  0.936909 midtone-saturation p95; visually it made grass fluorescent green and clothing neon cyan.
  A forced-neutral control was visibly safer. Current auto mode now independently reaches that
  safer result through `neutral_balance_evidence_rescue`: preserved gamut is 0.998206 (a 0.257056
  gain), midtone-saturation p95 is 0.660818 (a 0.276090 reduction), and memory-colour/spatial-neutral
  penalties improve from 0.012834/0.018247 to zero while retaining 117,624 chromatic samples with a
  0.898598 median saturation ratio. Visual comparison confirms natural-looking green/cyan
  separation instead of the fluorescent cast, although the proof remains warm. The release report
  and independently reopened TIFF/master/sRGB proof all pass dimensions, storage, profile, gamut,
  and freshness checks. A fresh reporting audit under
  `output/audit/negative-testroll-raw0006-calibration-application-20260720/` now separates the DNG
  prior's record state from final use: `calibration.status=applied` and source
  `dng_color_matrix1_advisory_prior` mean the prior was available, while
  `color_mapping_application` says `evaluated=true`, `applied=false`,
  `selection_status=rejected_unsafe`, preferred `scanner_prior_image_adaptation`, and selected
  `neutral_balance_fallback`. These values agree with the independent acceptance/final-candidate
  diagnostics and strict validation reports no consistency issue. The primary TIFF and float master
  are byte-identical to the preceding neutral-rescue audit; the sRGB PNG has identical IDAT bytes
  and decompressed scanline SHA-256
  `278ee331647f4f3d1fd3f23ca01c63e1c22144e80b9f4dd57ed2d793abee76ae` (only the compressed ICC
  chunk length changed by one byte, and independent ICC validation still passes). Final status correctly
  remains `review_required_negative_response`, `fallback_only`, and colour confidence 0.0 because
  only a regularized frame-derived response was available. This is evidence of safer fallback
  selection, not an approved colour result.
- Negative TESTROLL `RAW_0011`: source tag 1 was explicitly corrected by 180 degrees before every
  geometry phase. The final decoded-pixel digest changed from
  `f47622baf786a9bfb42d3ef2b45c20bd5225ec86f3c86847f6a1be4240cbdfd1` to
  `710369026f7f9bcae827f87626f58d000a4c108db9e2c74b2e21b163669ecbbe`; the effective tag is 3.
  Edge removals rotated exactly from 115/142/82/144 to 142/115/144/82, crop confidence is 0.941,
  and the 5726×3687 proof is visibly upright. The separate orientation draft records the correction
  and final digest but remains `requires_human_approval` / `upright_approved: false`. Tone evidence
  retains a 0.850 p05-p95 span with zero high/low clipping; conspicuous uncalibrated colour still
  blocks the delivery as `review_required_negative_response`.
- Negative LOGAN043/044: source is only RGBA8 expanded into the work domain. Stitch accepted with
  complete geometry and measurable seam reduction; the held-out seam gate retained the identity
  photometric model rather than inventing an additive correction. The fresh 11,701 by 3,671 sRGB
  proof is geometrically coherent but has a severe neon magenta/yellow cast. Negative colour is not
  an approved result: film-base confidence is zero, the report blocks review, and the candidate
  baseline/diff remains under `output/audit/logan-current-candidate/` rather than replacing truth.
- Negative OLD_TESTROLL `RAW_0000`: opposing top/bottom evidence measured a -0.00425° aggregate,
  made no deskew resample, and retained 100% of the decoded rectangle. The separately retained
  removed-border rebate reached 0.95 confidence while the post-crop proxy correctly remained 0.0.
  That still does not approve colour: no measured film curve/dye-crosstalk model exists, so the
  unit-slope response is explicitly `review_required_negative_response` and non-reviewable.
- Local inventories: 17/17 current DNG negatives readable; 39/39 older DNG negatives readable with
  one sequence gap; 6/6 selected RGB16 positives readable. These are coverage facts, not truth.

## Priority order

1. Build and pin the factual scanner/roll/film/target corpus; establish absolute crop, seam,
   CIEDE2000/hue, headroom, false-trust, and grain/detail gates.
2. Populate and validate real scanner-coordinate linearization records; validate the bounded
   overlap constant, vertical, planar x/y, and quadratic x/y gain/gain+offset paths on approved real
   seams, including multi-scale detail-review thresholds, while retaining disjoint evidence and
   hard physical limits. Populate and qualify the
   typed root-polynomial colour path first; enable a
   typed residual LUT for real work only if the required substantially larger independent target set
   proves it beats the polynomial fallback.
3. Keep the native OpenCV feature path mandatory in pinned CI, then validate homography, absolute
   single-scan deskew, and native affine on real rotated scans/pairs with approved union,
   retained-area, cross-fit, and interpolation limits.
4. Prove ICA/direct-density/measured-response choices on held-out real targets and keep the simplest
   model that wins. Never promote a fallback-base negative to “beautiful.”
5. Acquire and pin real N-input fixtures, run representative release gates locally/scheduled, and
   improve progress/performance for full-resolution work.

## Completion criterion

ScanStitch is complete only when representative positive and negative scans—including single and
N-part split frames—run end to end without manual repair and pass pinned geometry, retained-area,
seam, colour/hue, headroom, false-confidence, and grain/detail thresholds. The archival master,
tagged delivery TIFF, and sRGB proof must all be present, and unsupported or weak-evidence cases
must fail or require review rather than receive a success-quality claim.

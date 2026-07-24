# Validation Corpus Status

Last audited: 2026-07-23. This page distinguishes runnable real-image evidence from ground truth.
Local scan files and generated output remain ignored; the facts below are reproducible only on a
checkout that has those private fixtures.

## Evidence currently available

| Set / proof | Observed result | What it proves | What it does not prove |
|-|-|-|-|
| `TESTROLL` inventory | 17/17 readable, 16-bit DNG linear RAW, no sequence gaps | Negative-source decode, precision, and metadata coverage | Correct crop, inversion, colour, or beauty |
| `OLD_TESTROLL` inventory | 39/39 readable; review required for the missing `RAW_0030`–`RAW_0046` sequence. The opt-in `RAW_0000` gate pins a -0.00425° no-op absolute deskew, 111/129/34/0 crop, and post-crop zero-confidence high-transmittance fallback; the pipeline separately retains the removed-border rebate at 0.95 confidence, then blocks the unit-slope response as `review_required_negative_response`. | Additional real negative-source variety, opposing-border deskew evidence, separation of measured pre-crop base from an untrusted post-crop proxy, and truthful response-model review | A complete roll or correct negative colour; no measured film curve/dye-crosstalk target exists |
| Selected `OLD_SLIDE_TESTROLL` subset | 6/6 readable RGB16 frames at 5959×3946 | Positive-source decode and warm/cool scene diversity | Full-roll coverage or target colour accuracy |
| BIRDING047 positive release render | Current-code reviewable 5630×3685 output; 122/139/187/142 pixels removed top/bottom/left/right; auto deskew measured a -0.0042° robust aggregate and correctly made no resample; direct edge sampling found no near-black scanner strip; 1.341% of sRGB-proof pixels received constant-lightness/hue gamut mapping with zero remaining target excursions. The adaptive skin gate evaluated 58.112% and protected 23.491%; the sky/grass preference guard matched 35.134% and limited 7.155%. The one-way high-chroma preferred-skin shoulder evaluated 79.126%, matched 21.708%, and adjusted 3.137%, with mean/max movement 0.903/3.000 DeltaEab and mean chroma delta -0.604. | Real evidence-gated no-op deskew, four-edge crop, ICC-aware positive path, byte-stable tagged scene master, sRGB proof, bounded display gamut mapping, measured skin-vibrance protection, published sky/grass overshoot protection, and a bounded published skin-preference shoulder whose accounting passes strict validation | Accuracy on a genuinely rotated scan, semantic skin/sky/grass identification, or that its strong warm/magenta appearance matches the original scene. Every colour gate is an aggregate neighbourhood proxy. This frame has no colour target, inclusive diverse-skin corpus, approved reference print, or human preference approval. |
| BIRDING047 selective grain-on candidate | Exact release executable SHA-256 `0612bd1c…6de96` at strength 0.5/scale 1.0: 48.063% active mask, 50.674% exact structure/incomplete-window exclusion, 6.368%/14.887% lower all-sample median luma/chroma residual, 0.928%/0.665% lower flat-area p95 luma/chroma residual, and luminance/chroma p10 coherent-detail retention 0.992/~1.000 over 38,647/14,622 probes. The technical master is byte-identical to grain-off; 49.089% of RGB16 pixels and 36.540% of sRGB pixels changed, with median channel delta zero and negligible signed mean drift. | The independent optional control does measurable work without a near-universal mask, exactly preserves protected structure and unsupported image boundaries, retains measured coherent detail, avoids scaling the chroma base, and passes strict report/delivery consistency. | That this setting is preferred, that every suppressed residual is film grain rather than subject texture, or that it generalizes across stocks/scanners/scenes. Its hash-bound grain/detail decision remains explicitly unapproved. |
| TESTROLL RAW_0006 negative | Current code removed 143/116/165/77 pixels; the old run removed zero at right and visibly retained the scanner/rebate strip. The right crop used the recorded gradual-boundary fallback: 71 strong columns, 24,575 luminance gap versus a 605 threshold, and 228.8 aggregate transition versus a 504.4 derivative threshold. The 5710×3685 proof is border-free in visual inspection, with no geometry review. | A real missed-edge regression is fixed by evidence-gated conservative fallback without changing the other three boundaries; fallback provenance and confidence 0.454 remain explicit. | Correct negative colour or beauty. The response/dye model is unmeasured, the proof is conspicuously cast, and final status is `review_required_negative_response`. |
| TESTROLL RAW_0011 negative | Source orientation tag 1 plus explicit `rotate-180` produced effective tag 3 and changed the decoded digest from `f476…dfd1` to `7103…cbbe`; crop edges rotated from 115/142/82/144 to 142/115/144/82, confidence remained 0.941, and the 5726×3687 proof is visibly upright. | A real semantic correction runs before geometry, composes auditable scanner coordinates, rebinds pixels, and preserves four-edge crop behavior. | Human orientation approval or correct negative colour. The draft remains unapproved and the render remains `review_required_negative_response`. |
| LOGAN043/044 negative stitch | Accepted `[1|2]`; crop-origin expectation 39 px, measured 41 px; 217 px overlap; full 11701×3671 union; multiband seam-gradient ratio 0.576 and overlap p95 difference 0.161; the expanded constant/vertical/planar hierarchy retained `identity` | Real pair ordering/alignment after unequal border crops, full-union composition, measurable seam reduction, and conservative rejection of unsupported photometric complexity | Release-quality negative colour. Film-base confidence is 0, the render is not reviewable, and the fresh proof has a severe neon magenta/yellow cast |

The final-source RAW_0006 evidence bundle is retained at
`output/audit/negative-testroll-raw0006-borderfix-final-20260720/`. Its compact validator reopened
and matched the 5710x3685 RGB16 linear-ProPhoto primary TIFF, RGB32-float linear-ProPhoto master,
and RGB8 standard-sRGB proof. The production `--require-reviewable` gate then rejected the render as
expected with `delivery_artifact_issues=none`, so corrupt/missing output is not being confused with
the real blocker: absent measured negative-response evidence and visibly unacceptable colour.

A separate deterministic positive three-scan integration fixture now proves the enforceable N-input
path: shuffled ordering, two accepted merges, conservative aggregate exposure evidence,
`seam_aware_multiband`, no seam/detail review, multi-scale support, and bounded gradient/overlap
residuals. Its strict suite also rejects a detail-scale floor above the measured result. This is
synthetic implementation evidence and is therefore not listed as real corpus truth in the table.

A separate strict single-component registry fixture now proves that independent scans require only
`component1`: availability, exact file hash, readable-TIFF, singleton layout, and singleton
dimension sets become validation-ready while all pair counters remain zero. Direct validator
overrides accept the same one-input form, and roll-registry scaffolds no longer duplicate the path.
This removes dummy-input plumbing; it does not add photographic ground truth.

Two additional deterministic fixtures prove that the new corpus contracts are executable. The
geometry fixture applies a factual -0.5-degree correction to every component, retains at least 90%,
and locates every edge of a known 16-pixel scanner board within 4 pixels. Its strict adversary moves
one annotated edge beyond tolerance while leaving the structural preparation contract complete. Its
source TIFF also carries orientation 6 and the fixture applies `rotate-180`; strict evidence pins
the composed effective tag-8/counter-clockwise transform, applied flag, 220x160 to 160x220 dimension
swap, synthetic upright-approval flag, and exact final decoded-pixel digest, while a coherent
source-tag-8/effective-tag-6 wrong-digest adversary fails at runtime. The
negative fixture uses a measured nonlinear roll response with a nonidentity 3x3 dye-separation
matrix, nonlinear monotone curves, held-out CIEDE2000 evidence, bounded noise/extrapolation, and
signed headroom; it rejects an over-tight confidence floor. These are strict pass/fail plumbing
proofs, not real crop masks or film-colour truth.

Current-code real orientation-review drafts are available locally at
`output/audit/orientation-review-logan-20260718/` and
`output/audit/orientation-review-birding047-20260718/`, plus the metadata-relative RAW_0011 draft at
`output/audit/negative-testroll-raw0011-orientationfix-20260720/orientation-review/`. The previews are legible and visually
upright in a tooling sanity check. Their pinned decoded-pixel digests are
`d5f65f2efb8955a457848591d7b4d27f797e945609efddbca9cd9dcabcef7a2a` and
`e84312249639d0b77524ad97bb5c79c4200abb0c8f3ff9a18e39ebd50456ce1e` for the two LOGAN components,
and `18b45bcaafb6ef73546d26bce0b6fd27296589179f1b78b45361e2a38702811d` for BIRDING047. RAW_0011
records source tag 1, correction `rotate-180`, effective transform `rotate_180`, and final digest
`710369026f7f9bcae827f87626f58d000a4c108db9e2c74b2e21b163669ecbbe`. Every draft
still says `requires_human_approval` and `upright_approved: false`; none counts as accepted corpus
truth until a reviewer explicitly approves it.

Final-render review now has the same fail-closed acquisition path. `scanstitch-validate
--write-render-review` generates a draft whose crop/orientation, stitch/seam, colour, tone,
grain/detail, and overall-preference decisions are all unapproved. It binds ordered source-file and
decoded-pixel hashes, the report, primary TIFF, float master, and sRGB proof. Fixture coverage can
require `min_approved_render_review_fixtures` and revalidates every byte plus technical delivery
status before counting an approval. Schema version 2 requires a grain-on review to bind a matched,
technically reviewable grain-off report and all its artifacts; source/decode evidence and normalized
non-grain render arguments must match, both schema-v3-or-newer report executable fingerprints must be valid
and identical, and the two scene-referred masters must be byte-identical. No current real fixture
has a completed, reviewer-authored and
registry-hash-pinned render review, so this closes an evidence-admission loophole without supplying
the missing judgment.

The latest BIRDING047 package is available locally at
`output/audit/positive-birding047-preferred-skin-parallel-20260721/`. Its strict independent
`--require-reviewable` check passed for the exact 5630×3685 primary TIFF, float master, and sRGB
proof with no diagnostic-consistency or delivery-artifact issue and zero stale artifacts. The
primary TIFF SHA-256 is
`fb567932c08923007e189642b4edcb8fc3b64f80441308cb206e52bd671a03f8`; the sRGB proof is
`28142b686e8769cf30eeeb4e44ef0f54a2662803c5a3200a8fbc25a63929aa6e`. The technical float master
remains byte-identical to every preceding creative-render candidate at
`b72fb7e908aecf147c628adce8711accae21a91598c17aae9756f34e4af30625`. The preferred-skin shoulder
evaluated/matched/outside/adjusted 79.126%/21.708%/16.472%/3.137% of pixels. Its mean/max DeltaEab
were 0.903/3.000, mean/max absolute hue shift 1.112/6.595 degrees, mean chroma delta -0.604, maximum
absolute chroma change 2.127, and gamut-limited ratio zero. Against the preceding preferred-memory
package, 3.135% of RGB16 pixels changed; changed-channel mean/p95/max magnitudes were
0.002250/0.008164/0.083711 on a `[0,1]` scale. In the RGB8 proof, 2.463% changed, with
changed-channel mean/p95/max magnitudes 1.08/3/24 codes. The parallel and serial fused implementations
produced byte-identical primary TIFFs and pixel-identical sRGB proofs. The package includes a
schema-valid `render-review/render-review.json` binding the exact source/decode, report, and all
three delivery hashes. It remains `requires_human_approval`; all five applicable decisions are
false.

The superseding matched selective grain-on package is
`output/audit/positive-birding047-grain-v3-on-050-retry-20260721/`; its grain-off control is
`output/audit/positive-birding047-grain-v3-control-off-retry-20260721/`. Both strict
`--require-reviewable` runs passed with no diagnostic-consistency issue. Their report-schema-v3
metadata independently verifies the same 9,539,584-byte executable, SHA-256
`0612bd1c2c22b9a4cad36f5d05b5601ce19114785c78ebd306e4ff1c8756de96`, which also matches an
external post-render hash. Grain-on primary/proof SHA-256 are
`7db2c9021e77d93089ff92e30e82c1e9792949396f757d52c9f5d2aea99ab432` and
`a6c42c91ed9fd9fa3b2932126ce90d2cd084b4735f13b548db3d11c8784627b1`; the technical master remains
`b72fb7e908aecf147c628adce8711accae21a91598c17aae9756f34e4af30625` and is byte-identical to the
control master. The 0.5/1.0 pass exactly excluded 50.674% and materially selected 48.063% of pixels.
Relative to the exact grain-off render, the RGB16 changed-pixel ratio is 49.089%; all-channel
absolute p50/p95/p99/max differences are 0/0.00203/0.00435/0.09639 on `[0,1]`, and signed mean
channel drift is approximately `[1.09e-5, 1.05e-5, 2.08e-6]`. The isolated maximum is a chroma
outlier in an otherwise neutral white region; the previously larger unsupported top-row deltas were
eliminated by exact boundary bypass. The sRGB changed-pixel ratio is 36.540%, with p50/p95/p99/max
absolute differences 0/2/3/31 codes. The review-schema-v2, hash-reopened
`render-review-bound-control-v3/render-review.json` has SHA-256
`07099bcb849a9bbe43b5032bc8586748a6175b06b4bc77cae44598d4ca747d97`, makes grain/detail applicable
and unapproved, binds both exact reports plus all six delivery artifacts, and instructs a reviewer
to compare the two bound proofs at 100%. All six applicable decisions remain false. Earlier drafts
without the report-schema-v3 executable fingerprint are historical evidence only; none is an
accepted corpus result.

The previous package at
`output/audit/positive-birding047-skin-protection-candidate-20260719/` retains an older hash-bound
review draft, but that draft binds the old primary SHA-256
`b07a8ab6e0cc45c82350bfa23afcf4d1be91afdf5c358c23fef75b02c380de2e` and cannot approve the new
render. It remains `requires_human_approval` with zero of five applicable decisions approved.
Automated inspection of the new proof found no obvious crop loss or edge strip, but still noted
pronounced warm/magenta colour and visible grain. Neither the metrics nor inspection are curator
approval, and this people/stone scene cannot validate semantic sky/grass identification.

The compact inventory reports are under `output/audit/` when the local corpus is present. The two
real proof reports are `output/audit/positive-birding047-preferred-skin-parallel-20260721/report.json` and
`output/audit/logan-current-candidate/report.json`. The latter has a separately derived compact
candidate and tracked-baseline diff in the same ignored directory; these are review material, not an
accepted replacement for `tests/fixtures/baselines/logan_summary_baseline.json`.

## Ground truth still missing

No current registry entry pins all of the following factual evidence:

- scanner target captures plus scanner settings and at least 12 uniquely identified training and 12
  disjoint held-out patch measurements (the code now rejects in-sample-only matrix confidence);
- measured roll base/rebate and identified film stock/process;
- photographed reference patches or a spectrophotometric target with D50 XYZ/Lab values;
- approved crop, orientation, retained-area, stitch-union, and seam masks;
- a validation-ready real geometry preparation contract with applied all-component deskew,
  four-edge crop, retained-area bounds, and no geometry/crop review, plus an accuracy contract that
  pins the approved signed correction and each component's top/bottom/left/right crop boundary;
- a real orientation contract whose `upright_approved` declaration and pinned decoded-pixel digest
  were recorded only after a human approved each component as upright, rather than merely trusting
  source metadata;
- a real three-or-more-scan stitch with approved per-merge exposure normalization, seam residual,
  gradient, focus/sharpness continuity, and final-union bounds;
- a validation-ready measured negative-response contract with a credible rebate/base, independent
  held-out roll target, nonlinear dye reconstruction, bounded CIEDE2000/noise/extrapolation, and a
  trusted direct-density colour result;
- approved negative reference renders across skin, foliage, saturated colour, high key, deep
  shadow, mixed illumination, and exposure extremes;
- completed hash-bound render-review manifests for the exact approved positive/negative deliveries,
  with all applicable crop, seam, colour, tone, grain/detail, and preference decisions documented;
- grain/detail regions and acceptance bounds at multiple reduction strengths/scales. The harness
  can now reject an all-off corpus and incomplete two-channel contracts, but no local real fixture
  currently satisfies that factual evidence requirement.

`scanstitch-calibrate sample-target` can now turn independent high-bit-depth chart scans into these
disjoint measurements with projective sampling, duplicate decoded-pixel detection, robust patch
quality gates, and overlays. It cannot emit a measurement until every manually declared chart
footprint is visually approved against its overlay and bound to the exact orientation-materialized
decoded-pixel hash, ordered-corner-coordinate hash, and rendered-overlay-pixel hash; it writes a
paste-ready, explicitly unapproved review manifest and invalidates approval after any pixel,
precision, orientation, corner-geometry, grid-layout, or overlay-renderer change. No qualifying
measurement is fit-ready until the ordered patch IDs, positions, and XYZ/Lab values also have an
explicit approval matching `reference_patch_sha256`; edited or placeholder reference values fail
closed. No qualifying chart scans or factual XYZ/Lab reference dataset are present
locally, so this acquisition path is implementation evidence rather than new ground truth.
`scanstitch-validate --write-orientation-review` can likewise generate orientation-materialized
previews and digest-pinned, explicitly non-approved annotation drafts for the local scans. It makes
review reproducible but cannot supply the missing human approval itself.

Therefore the corpus can currently catch engineering regressions and false confidence, but cannot
certify absolute colour accuracy or choose the universally “most beautiful” rendering. Do not
refresh a stale baseline merely to make it pass: first classify the change against pinned source,
calibration, reference, and expectation hashes.

## Admission criteria for a release-quality fixture

A fixture is validation-ready only when source/profile/baseline SHA-256 values are pinned, source
TIFFs are readable at the declared precision, calibration and reference evidence are usable, and
declared expectations cover geometry, seam continuity, render reviewability, output colour space,
candidate trust, DeltaE00/hue-family limits where targets exist, headroom/clipping, and grain/detail
effects where denoising is exercised. A negative with fallback film-base confidence is a required
failure/review case, never an approved beauty reference.
Production delivery additionally requires the saved output to remain independently inspectable:
nonempty path, positive dimensions, ICC agreement with the report, and explicit zero stale
artifacts. A baseline cannot substitute for a missing or mismatched TIFF.
For denoise evidence, require `min_grain_reduction_enabled_fixtures`,
`min_grain_detail_contract_fixtures`, and `min_grain_reduction_effect_contract_fixtures`. A complete
case pins nonzero strength and scale, no-review and support expectations, at least 64 coherent probes
plus a 0.70 p10-retention floor in both luminance and opponent-colour channels, and positive
fixture-measured floors for applied pixels, exact structure-excluded pixels, and flat-area
luma/chroma p95 residual reduction. The exclusion floor prevents a near-universal mask from counting
as a complete effect contract.
For tonal-latitude evidence, require `min_render_dynamic_range_contract_fixtures`. Each counted case
must pin a positive post-scale preservation floor, a positive approved p05-p95 render-luminance span,
and fixture-specific high/low post-tone clipping ceilings below 1.0. These prove regression bounds,
not that every scene should fill the histogram.
For stitching evidence, require `min_stitch_normalization_contract_fixtures`. Each counted case must
pin accepted geometry, a known exposure model with model-appropriate held-out validation, at most a
5% normalized-offset ceiling, applied seam-aware multiband blending, no review, at least two
supported detail scales, at most 2.0 symmetric detail-energy imbalance, at most 1.05 gradient ratio,
and an overlap-p95 ceiling below 1.0. These prove regression enforcement; the values still need an
approved real seam before they become photographic truth.

## Known corpus/runtime constraints

- A full 33-frame, roughly 141 MB/frame positive-roll plausibility inventory exceeded ten minutes;
  per-commit gates should use a pinned representative subset and schedule the full roll separately.
- Debug full-resolution rendering is impractical for routine use. After positive-mode base analysis
  was made explicitly not applicable, an earlier BIRDING047 render completed in 14.6 seconds wall
  time with an 11.4-second phase total. The 2026-07-19 skin-protection perfect-mode candidate had a
  38.079-second phase total after compilation, including 15.212 seconds of tone mapping and 20.097
  seconds to write the tagged TIFF, float master, and sRGB proof; `base_detect_classify=0 ms`. Its
  primary TIFF SHA-256 is `b07a8ab6e0cc45c82350bfa23afcf4d1be91afdf5c358c23fef75b02c380de2e`.
  The 2026-07-20 preferred-memory-guard release rerender completed all phases in 19 seconds after
  compilation, including 7.080 seconds of tone mapping and 8.104 seconds of output writing. This
  single warm people/stone scene is encouraging performance evidence, not a broad benchmark. The
  accepted 2026-07-21 one-way preferred-skin rerender completed in 37 seconds, including 9.433
  seconds of tone mapping and 18.619 seconds of three-artifact output. A rejected serial prototype
  took 21.255 seconds in tone and a separate parallel pass took 15.097 seconds; fusing the two Lab
  creative decisions and parallelizing fixed 128-row chunks retained byte-identical primary output
  while limiting the accepted overhead to 2.353 seconds versus the pre-feature tone phase. A
  genuine foliage/sky fixture, inclusive skin/reference-print set, and full-roll timing are still
  required.
  The later streaming-artifact BIRDING047 run completed in 55.0 seconds with a 22.021-second save
  phase and a 1,599.6 MiB monitored process peak. Its report limits delivery conversion to 1,013,400
  bytes instead of a 248,958,600-byte full-frame float buffer; primary/master bytes and all decoded
  proof pixels match the pre-streaming result. This is a bounded-memory and identity result, not a
  controlled throughput benchmark. A byte audit subsequently found that the valid sRGB profile's
  generated header time still made those older PNG containers nondeterministic.
  The superseding owned-batch/canonical-ICC release ran the same BIRDING047 case twice from exact
  executable SHA-256 `c6442ce0bb5e94c85f0a585ef5d686b03f3e6fbb527d7f02826969d3063d8de2` in 45.522/44.644 seconds.
  Its monitored peaks were 1,124.5/1,124.6 MiB, about 475 MiB (29.7%) below the streaming-only run,
  while the declared artifact-conversion peak remained 1,013,400 bytes. The primary, float master,
  and PNG are byte-identical across both runs; the proof hash is
  `c4558f5594737c83055272f574c954875504f251b3d21713d342b2a88f33c47a`, and its canonical 612-byte
  profile hash is `0f303d7fb11d811c9751321ea472f4f024781dcdf0143ac7d94cebd1f0719323`. Both independently pass
  `--require-reviewable`. This establishes deterministic delivery and the avoided batch render
  allocation, but remains neither a controlled speed benchmark nor a human preference verdict.
  The current schema-v4 release, exact renderer SHA-256
  `ff236561e2c5897c190d8f5c58b8b1803cea6a47808b224febaa9c2e22130f76`, reran BIRDING047 under
  `positive-birding047-schema-v4-artifact-hash-20260721` in 49.173 seconds at a 1,124.7 MiB peak.
  Its sequential post-encoding hash pass uses a declared 1,048,576-byte buffer, does not overlap
  conversion buffers, and binds the unchanged primary/master/proof digests
  `7db2c9021e77d93089ff92e30e82c1e9792949396f757d52c9f5d2aea99ab432`,
  `b72fb7e908aecf147c628adce8711accae21a91598c17aae9756f34e4af30625`, and
  `c4558f5594737c83055272f574c954875504f251b3d21713d342b2a88f33c47a`. Exact validator SHA-256
  `763695c07a5b78bf2ee7e368685e5d0547dd9a674667842c903612bfe564b22c` independently reports all
  three matches true and passes `--require-reviewable`. Same-shape/profile substitutions are
  regression-rejected by named digest mismatches. This is unchanged-report artifact integrity,
  not a signed chain of custody or human approval.
  The fresh current-code accepted full-width LOGAN run took
  274.7 seconds, dominated by colour-model evaluation (83.5 seconds), base analysis (30.7 seconds),
  stitching (26.3 seconds), working-image selection (23.4 seconds), and ICA (22.3 seconds).
- The 2026-07-20 perfect-mode TESTROLL geometry rerenders completed all 13 phases in 52 seconds for
  RAW_0006 and 56 seconds for RAW_0011 after the optimized binary was built. Their compact validator
  summaries independently reopened the RGB16 ProPhoto output, RGB32-float master, and RGB8 standard
  sRGB proof successfully; both remain technically non-reviewable because negative-response evidence
  is absent, not because delivery artifacts or geometry failed. After the final border-specificity
  refinements, RAW_0006 was rebuilt and rerun from the exact audited source: release compilation took
  5 minutes 19 seconds and the 13 pipeline phases took 2 minutes 19 seconds. Its crop, border evidence,
  decoded digest, dimensions, tone range, clipping, and review status exactly matched the earlier run.
- The absolute-deskew phase reports three supported border sides, a -0.0042° aggregate, a 0.3476°
  full side-angle range, and a bounded no-op decision. Before skin-memory protection, its output was
  byte-identical to the pre-deskew release proof. The new display render intentionally differs, but
  its scene master SHA-256 remains `b72fb7e908aecf147c628adce8711accae21a91598c17aae9756f34e4af30625`,
  proving that the creative-only vibrance gate did not alter technical reconstruction. Synthetic
  rotated frames prove correction mechanics, but a real rotated scan with approved retained-area
  truth is still required.
- The real LOGAN stitch test remains ignored in default CI because private TIFFs are not tracked.
  On 2026-07-16 its release-mode opt-in gate passed with the pinned 39 px crop-origin prior, 41 px
  recovered placement, 217 px overlap, and support thresholds. The private `RAW_0000` no-op
  deskew/post-crop fallback-base gate also passed. CI can prove synthetic behavior, but these local
  strict fixture checks remain a release prerequisite.
- A fresh 2026-07-17 LOGAN run exposed and then verified removal of repeated full-frame post-tone
  allocations: the 11,701 by 3,671 f64 render previously failed while requesting another
  1,125,236,472-byte buffer, but now completes all 13 phases and saves the output by reusing one
  owned post-tone working buffer. Auto affine also now rejects the pair's unsupported candidate at
  the exact 3 degree search edge (three local correspondences, zero affine inliers), preserving the
  pinned translation union. The tracked compact baseline still describes a 5,959 by 3,670 render;
  strict comparison remains review-required and must not be refreshed without visual approval of
  the full-union and colour/tone deltas.
- After adding planar seam normalization, the same debug opt-in LOGAN stitch gate retained the
  identity model and all pinned geometry. Replacing full sorts with equivalent nearest-rank
  selection and reusing constant/vertical per-row fields reduced that gate from 550.94 to 451.26
  seconds; it remains too slow for hosted per-commit CI and does not substitute for colour truth.

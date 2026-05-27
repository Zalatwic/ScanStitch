# TESTROLL Negative Dynamic Range Notes - 2026-05-20

## Diagnosis

The overly dark positive renders came from two coupled issues in the negative path:

- Direct-density rendering originally converted scanner optical density to transmittance without normalizing the useful density range, so sparse dense pixels could push most of a high-key negative into near-black display values.
- The log-domain tone fit treated shadow-heavy negatives as if the input median should remain close to the measured scene-referred median, which kept several TESTROLL frames around a rendered midtone p50 of 0.20.

Weak or unstable colour came mainly from image-derived matrix candidates that preserved too much negative-gamut clipping. The current colour path ranks image-derived, gamut-safe blended, gamut-trusted blended, scanner/calibration, and neutral fallback candidates by measurable quality. Unsafe image-derived matrices are rejected when they would create material negative channel clipping, and the selected candidate records the reason, risk state, neutral support, anchor support, exposure scale, preserved-gamut ratio, and tone-colour trust state in `report.json`.

## Current Parameters

- Input mode: `negative`
- TESTROLL validation bit depth: `16`
- TESTROLL roll base: `56004.818681,22973.258242,12666.899267`
- Roll base source: `roll_consensus_base`
- Roll base confidence: `0.98`
- Roll base reason: selected stable cluster from 14 of 17 frame candidates; 3 darker candidates rejected against the roll high-transmittance envelope
- Render input for the final negative checks: `direct-density`
- Density inversion Dmax percentile: `0.995`
- Direct-density normalization: `positive_density / shared_robust_d_max` before density-to-transmittance conversion
- Colour candidates selected on validated TESTROLL negative outputs: `gamut_trusted_image_matrix_blend` or `image_derived_matrix`, with review state retained when neutral, anchor, or model-plausibility support is not strong enough for trusted colour
- Highlight headroom percentile for colour exposure normalization: `0.995`
- Gamut-trusted blend limits: max low clip per channel `0.01`, total low clip `0.02`
- Negative low-range tone trigger: linear p95 <= `0.12` and p95-p05 <= `0.045`
- Negative low-range target median: `0.35`
- Negative shadow tone trigger: linear p50 <= `0.22`
- Negative shadow target median: `0.32`
- Negative shadow highlight guardrail: mapped linear p95 <= `0.95`
- Tone curve: Naka-Rushton, toe `0.005`, shoulder `0.995`
- Perceptual log-domain gain: `15.0`
- Highlight chroma repair: enabled to preserve mapped luminance without hard channel clipping
- Near-neutral highlight chroma cleanup: starts at luminance `0.72`, full at `0.92`, min scale `0.40`, max saturation `0.35`
- Shadow chroma cleanup: starts at luminance `0.24`, full at `0.08`, min scale `0.25`
- Final render denoise shadow saturation relaxation: `0.90`, so saturated low-luma speckle is not over-protected by the chroma saturation gate
- Post-denoise shadow saturation guard: max saturation `0.54` in deep shadows, easing to `0.74` at the upper shadow boundary
- Post-denoise midtone-neutral cleanup: enabled only on trusted or anchor-review colour paths, luminance `0.24..0.76`, full strength `0.38..0.62`, max source saturation `0.35`, min scale `0.70`

## Validation Evidence

Full 12-frame TESTROLL tone/gamut comparison:

- Artifact: `output/current_testroll_negative_tone_gamut_metrics_20260520.md`
- Preview: `output/current_testroll_negative_visual_review_tone_gamut_gamma_20260520.png`
- Mean mapped p50 improved from `0.200784` to `0.250412`
- Minimum mapped p95 improved from `0.263041` to `0.441045`
- Mean selected colour quality score improved from `11.048713` to `3.041167`
- Mean preserved-gamut ratio improved from `0.981076` to `0.985160`
- Max high clipping remained `0.000000`
- Max low clipping was `0.000234`

Full 12-frame TESTROLL dynamic-range comparison before the final shadow-lift adjustment:

- Artifact: `output/testroll_negative_dynamic_range_20260520/roll-suite.md`
- Comparison: `output/testroll_negative_dynamic_range_20260520/dynamic-range-comparison.md`
- Preview: `output/testroll_negative_dynamic_range_20260520/visual-review-gamma.png`
- Mean mapped p50 improved from `0.250412` to `0.300041`
- Mean rendered midtone p50 improved from `0.250640` to `0.300218`
- Mean shadow chroma compression dropped from `0.361822` to `0.119430`
- Max post-tone high clipping remained `0.000000`
- Max post-tone low clipping was `0.000234`

Targeted final shadow-lift check:

- Artifact: `output/testroll_negative_dynamic_range_shadow_lift_20260520/compare-summary.md`
- Preview: `output/testroll_negative_dynamic_range_shadow_lift_20260520/preview-old-new.png`
- Frames checked: `RAW_0001`, `RAW_0008`, `RAW_0011`
- Mapped linear p50 improved to `0.3000` on all three checked frames
- Rendered midtone p50 improved from about `0.20` to about `0.30`
- Bright-neutral p95 stayed below the shoulder (`0.9617`, `0.9468`, `0.9521`)
- Post-tone high clipping remained `0.000000`

Full 12-frame rerun after anchor-support shadow cleanup:

- Artifact: `output/testroll_negative_anchor_shadow_cleanup_roll_suite_20260520/roll-suite.md`
- Status: `review_required`, with all 12 frames rendered; remaining issues are missing calibration/reference evidence and color-review risk, not render failures.
- Historical tone policy: `review_shadow_cleanup_only` for `review_anchor_support` frames kept neutral-highlight cleanup disabled and kept `color_trust_state=review_required`, while allowing bounded shadow chroma cleanup to reduce low-tone color speckle. This was superseded by the 2026-05-21 bounded neutral-highlight cleanup pass below.
- Mean rendered midtone p50 improved from `0.300218` in the prior full 12-frame run to `0.333413`, because the earlier full run did not include the final shadow-lift on `RAW_0008`, `RAW_0011`, and `RAW_0012`.
- Mean shadow saturation p95 improved from `0.431630` to `0.400680`; max shadow saturation p95 improved from `0.986142` to `0.868300`.
- Anchor-review frames improved as follows: `RAW_0008` midtone p50 `0.200848 -> 0.297141`, shadow saturation p95 `0.916834 -> 0.753094`; `RAW_0011` p50 `0.198087 -> 0.299259`, shadow saturation p95 `0.969294 -> 0.860807`; `RAW_0012` p50 `0.197884 -> 0.298904`, shadow saturation p95 `0.986142 -> 0.868300`.
- Post-tone high clipping remained `0.000000`; max post-tone low clipping was `0.000008`.

Fresh current-code verification on 2026-05-21 after the roll-suite diagnostic update:

- Artifact: `output/testroll_negative_current_fields_20260521.md`
- JSON: `output/testroll_negative_current_fields_20260521.json`
- Preview: `output/testroll_negative_current_fields_20260521_preview.png`
- Status: `review_required`, with all 12 frames rendered and `0` failed frames; remaining issues are calibration/reference evidence and colour-review warnings.
- Mean mapped linear p50: `0.332689`; minimum mapped linear p50: `0.300000`.
- Mean rendered midtone p50: `0.333413`; minimum rendered midtone p50: `0.297141`.
- Mapped linear p95 range: `0.442228` to `0.920000`, so low-range frames use more display latitude while high-contrast frames stay at the configured highlight guardrail.
- Post-tone high clipping max: `0.000000`; post-tone low clipping max: `0.000008`.
- Mean/min post-scale preserved-gamut ratio: `0.985160` / `0.979744`.
- Mean/max shadow saturation p95: `0.400680` / `0.868300`.
- Mean/max bright-neutral saturation p95: `0.237270` / `0.311912`.
- Roll-suite validation now records per-frame `midtone_luminance_p50`, `shadow_saturation_p95`, and `bright_neutral_saturation_p95` plus suite-level aggregates, so future TESTROLL brightness checks do not require parsing each frame report separately.

Fresh current-source goal verification on 2026-05-21 after the final tone-map edit:

- Artifact: `output/testroll_negative_goal_verify_20260521.md`
- JSON: `output/testroll_negative_goal_verify_20260521.json`
- Contact sheet: `output/testroll_negative_goal_verify_20260521_contact_sheet.png`, built from the 12 fresh per-frame `review_srgb.png` renders.
- Before/current visual comparison: `output/testroll_negative_before_current_visual_compare_20260521.png`, using the 2026-05-20 reference preview, the tone/gamut preview, and the current contact sheet.
- Command: `cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --bit-depth 16 --render-input direct-density --input-mode negative --output-dir output\testroll_negative_goal_verify_20260521 --summary-json output\testroll_negative_goal_verify_20260521.json --summary-md output\testroll_negative_goal_verify_20260521.md`
- Status: `review_required`, with all 12 frames rendered and `0` failed frames. The remaining issues are calibration/reference evidence and explicit uncalibrated colour-review flags, not render failures.
- Mean/min rendered midtone p50: `0.333413` / `0.297141`.
- Max post-tone high clipping: `0.000000`; max post-tone low clipping: `0.000008`.
- Mean/min post-scale preserved-gamut ratio: `0.985160` / `0.979744`.
- Mean/max shadow saturation p95: `0.393148` / `0.853650`; mean/max bright-neutral saturation p95: `0.237288` / `0.311912`.
- High-frequency residual p95 mean/max: luma `0.116374` / `0.275959`, chroma `0.059916` / `0.160221`, chroma:luma ratio `0.390983` / `0.719508`.
- Denoise ran on all 12 frames, with applied ratio mean/max `0.909395` / `0.994708`; mean chroma/luma deltas stayed bounded at `0.005489` / `0.002821`.
- Per-frame diagnostics still identify the difficult anchor-review frames (`RAW_0008`, `RAW_0011`, `RAW_0012`) instead of marking their colour trusted, but their midtone p50 values remain near `0.30` and highlight clipping remains zero.

Fresh current-source color-balance diagnostic rerun after adding roll-suite RGB balance deltas:

- Artifact: `output/testroll_negative_color_balance_verify_20260521.md`
- JSON: `output/testroll_negative_color_balance_verify_20260521.json`
- Command: `cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --bit-depth 16 --render-input direct-density --input-mode negative --quality-mode balanced --output-dir output\testroll_negative_color_balance_verify_20260521 --summary-json output\testroll_negative_color_balance_verify_20260521.json --summary-md output\testroll_negative_color_balance_verify_20260521.md`
- Status: `review_required`, with all 12 frames rendered and `0` failed frames. The remaining issues are unchanged: calibration/reference evidence and explicit uncalibrated colour-review flags.
- RGB balance delta mean/max: shadow `0.009891` / `0.020928`, midtone `0.086948` / `0.272736`, bright-neutral `0.092731` / `0.174407`.
- The highest midtone and bright-neutral balance deltas are on `RAW_0008`, `RAW_0011`, and `RAW_0012`, matching the existing `review_anchor_support` classification. This confirms that the remaining colour risk is concentrated in the difficult anchor-review frames, not a roll-wide shadow-cast drift.

Fresh bounded neutral-highlight cleanup verification on 2026-05-21:

- Artifact: `output/testroll_negative_anchor_highlight_roll_suite_20260521.md`
- JSON: `output/testroll_negative_anchor_highlight_roll_suite_20260521.json`
- Before/after preview: `output/testroll_negative_anchor_highlight_before_after_20260521.png`
- Command: `cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --bit-depth 16 --render-input direct-density --input-mode negative --quality-mode balanced --output-dir output\testroll_negative_anchor_highlight_roll_suite_20260521 --summary-json output\testroll_negative_anchor_highlight_roll_suite_20260521.json --summary-md output\testroll_negative_anchor_highlight_roll_suite_20260521.md`
- Policy change: `review_anchor_support` frames now use `review_bounded_neutral_shadow_cleanup`, which keeps bounded neutral-highlight cleanup and shadow chroma cleanup enabled while preserving `color_trust_state=review_required`.
- Status: `review_required`, with all 12 frames rendered and `0` failed frames. Remaining issues are still calibration/reference evidence and explicit uncalibrated colour-review flags.
- Mean/min rendered midtone p50 stayed stable at `0.333313` / `0.296821`; max post-tone high clipping stayed `0.000000`; max post-tone low clipping stayed `0.000008`.
- Bright-neutral saturation p95 mean/max improved from `0.237288` / `0.311912` to `0.213660` / `0.258280`.
- Bright-neutral RGB balance delta mean/max improved from `0.092731` / `0.174407` to `0.076858` / `0.124307`.
- Anchor-review frame bright-neutral RGB deltas improved as follows: `RAW_0008` `0.174407 -> 0.104399`, `RAW_0011` `0.130801 -> 0.074282`, `RAW_0012` `0.129169 -> 0.069699`.
- Added midtone-neutral diagnostics to separate neutral casts from real scene colour. The full midtone RGB delta mean/max is still `0.087` / `0.273`, but the midtone-neutral RGB delta mean/max is only `0.016` / `0.034`; anchor-review frames have neutral-midtone deltas `0.034` (`RAW_0008`), `0.023` (`RAW_0011`), and `0.011` (`RAW_0012`). This indicates the remaining high full-midtone delta is mostly saturated blue/cyan scene content rather than a broad neutral midtone cast.
- Added texture-aware adaptive vibrance after `RAW_0001` showed the highest chroma:luma residual among trusted-colour frames. On `RAW_0001`, adaptive vibrance application dropped from `0.704656` to `0.443311`, mean adaptive scale dropped from `1.082118` to `1.062999`, and `adaptive_vibrance_texture_limited_ratio=0.474401`; midtone p50 stayed `0.300805`, high clipping stayed `0.000000`, chroma residual p95 improved `0.153524 -> 0.146499`, and chroma:luma residual p95 improved `0.719508 -> 0.686585`.
- Full 12-frame TESTROLL aggregate after the texture gate stayed stable: mean/min rendered midtone p50 `0.333410` / `0.297141`, max post-tone high/low clipping `0.000000` / `0.000008`; chroma residual p95 mean improved `0.059970 -> 0.059384`, chroma:luma residual p95 mean/max improved `0.389380` / `0.719508` to `0.386637` / `0.686585`, and bright-neutral saturation p95 mean improved `0.214649 -> 0.214060`.
- Added flat-area grain diagnostics and changed final chroma denoise so low-structure grain is not treated as protected texture while the chroma texture gate is never stricter than the previous residual-only gate. On the same 12-frame TESTROLL run, all-sample chroma residual p95 mean/max improved `0.059384` / `0.160841 -> 0.057854` / `0.159560`, all-sample chroma:luma residual p95 mean/max improved `0.386637` / `0.686585 -> 0.367303` / `0.675267`, flat-area chroma residual p95 mean/max improved `0.032629` / `0.083129 -> 0.030276` / `0.077860`, and flat-area chroma:luma residual p95 mean/max improved `0.352741` / `0.678299 -> 0.326663` / `0.634887`; midtone p50 and clipping stayed stable at `0.333313` / `0.296821` and `0.000000` / `0.000008`.

Non-TESTROLL negative regression check:

- Artifact: `output/validation/logan_current_non_testroll_20260521.md`
- JSON: `output/validation/logan_current_non_testroll_20260521.json`
- Debug acceptance artifact: `output/validation/logan_current_debug_20260521.md`
- LOGAN rendered successfully with `render_review_status=reviewable`, `stitch.decision=accepted`, valid embedded ProPhoto ICC, matching output ICC inspection, and `stale_render_artifact_count=0`.
- Auto render input stayed on `fastica_separated_transmittance` because the current ICA path selected `gamut_trusted_image_matrix_blend` at quality score `1.813523`, while the evaluated direct-density candidate selected the same blended strategy at quality score `2.591787`.
- Relative to the older tracked LOGAN compact baseline, lower-is-better selected quality improved `7.596999 -> 1.813523`, technical safety improved `6.275002 -> 1.283516`, colour fidelity improved `1.321997 -> 0.530007`, and neutral estimate quality improved `0.733087 -> 0.812808`.
- The debug LOGAN run emitted fresh `candidate_comparison`, `gamut_clipping_map`, and `scene_referred_prophoto_float` artifacts with `debug_artifact_invalid_count=0`. These artifacts show the accepted non-TESTROLL colour decision path after the new gamut-trusted blend and tone cleanup.
- The tracked LOGAN compact baseline was updated to the accepted current contract in `tests/fixtures/baselines/logan_summary_baseline.json`.
- Strict current-code LOGAN comparison now passes: `output/validation/logan_strict_current_baseline_20260521.md` reports `summary_baseline_status=comparable` and `issues=none`.
- Fresh strict LOGAN rerun for this completion audit: `output/validation/logan_strict_goal_verify_20260521.md` / `.json` generated `2026-05-21T20:04:51Z`, rendered `5959x3670`, kept `render_review_status=reviewable`, `stitch.decision=accepted`, `render_input_source=fastica_separated_transmittance`, `selected_candidate=gamut_trusted_image_matrix_blend`, `selected_quality_score=1.794397`, and `summary_baseline_status=comparable` with `issues=none`.
- Fresh LOGAN regression after bounded neutral-highlight cleanup: `output/validation/logan_anchor_highlight_regression_20260521.md` / `.json` generated `2026-05-21T21:14:24Z`, rendered `5959x3670`, kept `render_review_status=reviewable`, `stitch.decision=accepted`, `render_input_source=fastica_separated_transmittance`, `selected_candidate=gamut_trusted_image_matrix_blend`, `selected_quality_score=1.777766`, and passed `--strict`.
- Fresh LOGAN smoke rerun after adding midtone-neutral diagnostics: `output/validation/logan_midtone_neutral_diagnostic_20260521.md` / `.json` generated `2026-05-21T21:45:03Z`, rendered `5959x3670`, kept `render_review_status=reviewable`, `stitch.decision=accepted`, `render_input_source=fastica_separated_transmittance`, `selected_candidate=gamut_trusted_image_matrix_blend`, `selected_quality_score=1.777766`, and passed `--strict`.
- Fresh LOGAN smoke rerun after texture-aware adaptive vibrance: `output/validation/logan_adaptive_texture_gate_20260521.md` / `.json` generated `2026-05-21T22:05:37Z`, rendered `5959x3670`, kept `render_review_status=reviewable`, `stitch.decision=accepted`, `render_input_source=fastica_separated_transmittance`, `selected_candidate=gamut_trusted_image_matrix_blend`, `selected_quality_score=1.777766`, and passed `--strict`.
- Fresh LOGAN smoke rerun after flat-area grain diagnostics and structure-capped chroma denoise: `output/validation/logan_structure_denoise_20260521.md` / `.json` generated `2026-05-21T23:00:16Z`, rendered `5959x3670`, kept `render_review_status=reviewable`, `stitch.decision=accepted`, `render_input_source=fastica_separated_transmittance`, `selected_candidate=gamut_trusted_image_matrix_blend`, `selected_quality_score=1.794397`, and passed `--strict`. Relative to `logan_adaptive_texture_gate_20260521`, LOGAN all-sample chroma residual p95 improved `0.094872 -> 0.091116` and chroma:luma residual p95 improved `0.622010 -> 0.607332`, with post-scale preservation essentially stable at `0.979469 -> 0.979114`; the new flat-area chroma residual p95 is `0.050650`.

Fresh TESTROLL shadow saturation guard verification on 2026-05-22:

- Artifact: `output/testroll_negative_shadow_guard_roll_suite_20260522.md`
- JSON: `output/testroll_negative_shadow_guard_roll_suite_20260522.json`
- Before/after contact sheet: `output/testroll_negative_shadow_guard_before_after_20260522.png`
- Command: `.\target\release\scanstitch-validate.exe --roll-dir TESTROLL --roll-suite --bit-depth 16 --render-input direct-density --input-mode negative --quality-mode balanced --output-dir output\testroll_negative_shadow_guard_roll_suite_20260522 --summary-json output\testroll_negative_shadow_guard_roll_suite_20260522.json --summary-md output\testroll_negative_shadow_guard_roll_suite_20260522.md`
- Policy change: saturated low-luma pixels now receive more chroma denoise, then a soft shadow saturation guard limits only extreme post-denoise shadow chroma. This replaced a rejected hard-cap probe that reduced saturation but raised high-frequency chroma residuals.
- Status: `review_required`, with all 12 frames rendered and `0` failed frames. Remaining issues are unchanged calibration/reference evidence and uncalibrated colour-review flags.
- Mean/min rendered midtone p50 stayed unchanged at `0.333313` / `0.296821`; max post-tone high/low clipping stayed `0.000000` / `0.000008`; mean/min post-scale preserved-gamut ratio stayed `0.985160` / `0.979744`.
- Shadow saturation p95 mean/max improved `0.443990` / `1.000000 -> 0.319541` / `0.580000`.
- High-frequency chroma residual p95 mean/max stayed essentially flat to slightly better: `0.057854` / `0.159560 -> 0.057798` / `0.158986`; chroma:luma residual ratio mean improved `0.367303 -> 0.366358`.
- Flat-area chroma residual p95 stayed effectively stable: mean/max `0.030276` / `0.077860 -> 0.030366` / `0.078055`; flat chroma:luma ratio mean/max `0.326663` / `0.634887 -> 0.327312` / `0.636523`.
- Difficult frames improved shadow saturation without brightness movement: `RAW_0001` `0.945458 -> 0.580000`, `RAW_0008` `0.867588 -> 0.580000`, `RAW_0011` `1.000000 -> 0.580000`, and `RAW_0012` `1.000000 -> 0.580000`; their midtone p50 values were unchanged.
- Non-TESTROLL LOGAN check after this change: `output/validation/logan_shadow_guard_strict_20260522.md` / `.json` rendered `5959x3670`, kept `render_review_status=reviewable`, `stitch.decision=accepted`, `render_input_source=fastica_separated_transmittance`, `selected_candidate=gamut_trusted_image_matrix_blend`, `selected_quality_score=1.794397`, and passed `--strict` against the refreshed compact baseline. The accepted LOGAN chroma residual p95 is `0.091103` and chroma:luma residual ratio is `0.607302`.

Fresh TESTROLL midtone-neutral cast refinement on 2026-05-22:

- Artifact: `output/testroll_negative_midtone_neutral_roll_suite_20260522.md`
- JSON: `output/testroll_negative_midtone_neutral_roll_suite_20260522.json`
- Before/after contact sheet: `output/testroll_negative_midtone_neutral_before_after_20260522.png`
- Command: `.\target\release\scanstitch-validate.exe --roll-dir TESTROLL --roll-suite --bit-depth 16 --render-input direct-density --input-mode negative --quality-mode balanced --output-dir output\testroll_negative_midtone_neutral_roll_suite_20260522 --summary-json output\testroll_negative_midtone_neutral_roll_suite_20260522.json --summary-md output\testroll_negative_midtone_neutral_roll_suite_20260522.md`
- Policy change: a late, luminance-preserving midtone-neutral chroma cleanup now trims low-saturation midtone cast after denoise, but only when the colour path permits neutral chroma cleanup. Weak-neutral review frames still keep this disabled.
- Status: `review_required`, with all 12 frames rendered and `0` failed frames. Remaining issues are unchanged calibration/reference evidence and uncalibrated colour-review flags.
- Mean/min rendered midtone p50 stayed unchanged at `0.333313` / `0.296821`; shadow saturation p95 mean/max stayed `0.319541` / `0.580000`; max post-tone high/low clipping stayed `0.000000` / `0.000008`; mean/min post-scale preserved-gamut ratio stayed `0.985160` / `0.979744`.
- Midtone-neutral saturation p95 mean/max improved `0.188423` / `0.341348 -> 0.184175` / `0.328998`; midtone-neutral RGB balance delta mean/max improved `0.016422` / `0.033831 -> 0.016156` / `0.033252`.
- Flat-area grain stayed natural: flat chroma residual p95 mean/max improved `0.030366` / `0.078055 -> 0.030026` / `0.077012`; all-sample chroma residual p95 mean/max stayed essentially flat at `0.057798` / `0.158986 -> 0.057846` / `0.159214`.
- Difficult frame midtone-neutral saturation improved without brightness movement: `RAW_0001` `0.341348 -> 0.328353`, `RAW_0008` `0.341131 -> 0.327014`, `RAW_0011` `0.340148 -> 0.328998`, and `RAW_0012` `0.339909 -> 0.327193`; their midtone p50 values stayed unchanged.
- Non-TESTROLL LOGAN check after this change: `output/validation/logan_midtone_neutral_regression_20260522.md` / `.json` rendered `5959x3670`, kept `render_review_status=reviewable`, `stitch.decision=accepted`, `render_input_source=fastica_separated_transmittance`, `selected_candidate=gamut_trusted_image_matrix_blend`, `selected_quality_score=1.794397`, and passed `--strict` against the compact baseline with `summary_baseline_status=comparable` and `issues=none`.

Fresh TESTROLL weak-neutral highlight cleanup on 2026-05-22:

- Artifact: `output/testroll_negative_weak_neutral_highlight_roll_suite_20260522.md`
- JSON: `output/testroll_negative_weak_neutral_highlight_roll_suite_20260522.json`
- Before/after contact sheet: `output/testroll_negative_weak_neutral_highlight_before_after_20260522.png`
- Command: `.\target\release\scanstitch-validate.exe --roll-dir TESTROLL --roll-suite --bit-depth 16 --render-input direct-density --input-mode negative --quality-mode balanced --output-dir output\testroll_negative_weak_neutral_highlight_roll_suite_20260522 --summary-json output\testroll_negative_weak_neutral_highlight_roll_suite_20260522.json --summary-md output\testroll_negative_weak_neutral_highlight_roll_suite_20260522.md`
- Policy change: weak-neutral frames now keep bounded near-neutral highlight cleanup enabled while still disabling midtone-neutral cleanup and preserving `limited_weak_neutral` trust. This trims snow/highlight casts without marking the uncalibrated colour path trusted.
- Status: `review_required`, with all 12 frames rendered and `0` failed frames. Remaining issues are unchanged calibration/reference evidence and uncalibrated colour-review flags.
- Mean/min rendered midtone p50 stayed unchanged at `0.333313` / `0.296821`; shadow saturation p95 mean/max stayed `0.319541` / `0.580000`; max post-tone high/low clipping stayed `0.000000` / `0.000008`; mean/min post-scale preserved-gamut ratio stayed `0.985160` / `0.979744`.
- Bright-neutral saturation p95 mean improved `0.213660 -> 0.174614`, with max unchanged at `0.258280`; bright-neutral RGB balance delta mean/max improved `0.076858` / `0.124307 -> 0.057376` / `0.104399`.
- Grain metrics improved on TESTROLL despite the extra highlight cleanup: all-sample chroma residual p95 mean improved `0.057846 -> 0.056346`, chroma:luma residual p95 mean improved `0.366543 -> 0.338283`, and flat-area chroma residual p95 mean stayed stable at about `0.030`.
- Non-TESTROLL LOGAN check after this change: `output/validation/logan_weak_neutral_highlight_regression_20260522.md` / `.json` rendered `5959x3670`, kept `render_review_status=reviewable`, `stitch.decision=accepted`, `render_input_source=fastica_separated_transmittance`, `selected_candidate=gamut_trusted_image_matrix_blend`, and `selected_quality_score=1.794397`. Relative to the prior compact baseline, bright-neutral saturation p95 improved `0.343126 -> 0.329039`; all-sample chroma residual p95 moved `0.091103 -> 0.093213`, a small accepted tradeoff for the intentional weak-neutral highlight policy change.
- The compact LOGAN baseline was refreshed from `output/validation/logan_weak_neutral_highlight_regression_20260522.baseline.json` into `tests/fixtures/baselines/logan_summary_baseline.json`. Strict comparison against the refreshed baseline passed in `output/validation/logan_weak_neutral_highlight_strict_refreshed_20260522.md` / `.json` with `summary_baseline_status=comparable` and `issues=none`.

Fresh TESTROLL saturated-shadow denoise relaxation on 2026-05-22:

- Artifact: `output/testroll_negative_shadow_denoise_relax_roll_suite_20260522.md`
- JSON: `output/testroll_negative_shadow_denoise_relax_roll_suite_20260522.json`
- Comparator artifact: `output/testroll_negative_shadow_denoise_relax_compare_roll_suite_20260522.md`
- Comparator JSON: `output/testroll_negative_shadow_denoise_relax_compare_roll_suite_20260522.json`
- Command: `cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --bit-depth 16 --render-input direct-density --input-mode negative --quality-mode balanced --output-dir output\testroll_negative_shadow_denoise_relax_roll_suite_20260522 --summary-json output\testroll_negative_shadow_denoise_relax_roll_suite_20260522.json --summary-md output\testroll_negative_shadow_denoise_relax_roll_suite_20260522.md`
- Comparator command: `cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --bit-depth 16 --render-input direct-density --input-mode negative --quality-mode balanced --compare-roll-suite output\testroll_negative_weak_neutral_highlight_roll_suite_20260522.json --output-dir output\testroll_negative_shadow_denoise_relax_compare_roll_suite_20260522 --summary-json output\testroll_negative_shadow_denoise_relax_compare_roll_suite_20260522.json --summary-md output\testroll_negative_shadow_denoise_relax_compare_roll_suite_20260522.md`
- Policy change: saturated low-luma pixels now relax the chroma-denoise saturation gate to `0.90` instead of `0.80`. This is deliberately conservative because the prior `0.80` setting was already stable; the goal is to reduce remaining colored speckle in the four high-residual frames without changing their tone placement.
- Status: `review_required`, with all 12 frames rendered and `0` failed frames. Remaining issues are unchanged calibration/reference evidence and uncalibrated colour-review flags.
- Mean/min rendered midtone p50 stayed unchanged at `0.333313` / `0.296821`; max post-tone high/low clipping stayed `0.000000` / `0.000008`; mean/min post-scale preserved-gamut ratio stayed `0.985160` / `0.979744`.
- Chroma residual p95 mean/max improved slightly `0.056346` / `0.159214 -> 0.056316` / `0.159068`; chroma:luma residual p95 mean improved `0.338283 -> 0.338167`; flat-area chroma residual p95 mean/max improved `0.029775` / `0.077012 -> 0.029727` / `0.076940`.
- The measurable gain stayed concentrated in the intended frames without moving brightness: `RAW_0001`, `RAW_0008`, `RAW_0011`, and `RAW_0012` all kept the same midtone p50 while their flat chroma residuals moved down slightly.
- Added `--compare-roll-suite <previous-roll-suite.json>` to `scanstitch-validate` so aggregate and per-frame TESTROLL deltas are written directly into the current roll-suite JSON/Markdown. The fresh comparator run against the previous weak-neutral-highlight suite reported `comparison.status=comparable` and `issues=none`, with mean midtone delta essentially zero (`+0.000000000020`), chroma residual p95 mean delta `-0.000030`, flat chroma residual p95 mean delta `-0.000048`, preserved-gamut min delta `0.000000`, and high-clipping delta `0.000000`.
- Non-TESTROLL LOGAN strict check after this change: `output/validation/logan_shadow_denoise_relax_strict_20260522.md` / `.json` rendered `5959x3670`, kept `render_review_status=reviewable`, `stitch.decision=accepted`, `render_input_source=fastica_separated_transmittance`, `selected_candidate=gamut_trusted_image_matrix_blend`, `selected_quality_score=1.794397`, and passed `--strict` with `summary_baseline_status=comparable` and `issues=none`. LOGAN chroma residual p95 improved slightly `0.093213 -> 0.093153`, and flat chroma residual p95 improved `0.052216 -> 0.052193`.

Fresh TESTROLL shadow-heavy midtone lift and tighter shadow guard on 2026-05-22:

- Artifact: `output/testroll_negative_shadow_lift_guard_roll_suite_20260522.md`
- JSON: `output/testroll_negative_shadow_lift_guard_roll_suite_20260522.json`
- Command: `cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --bit-depth 16 --render-input direct-density --input-mode negative --quality-mode balanced --compare-roll-suite output\testroll_negative_shadow_denoise_relax_compare_roll_suite_20260522.json --output-dir output\testroll_negative_shadow_lift_guard_roll_suite_20260522 --summary-json output\testroll_negative_shadow_lift_guard_roll_suite_20260522.json --summary-md output\testroll_negative_shadow_lift_guard_roll_suite_20260522.md`
- Policy change: shadow-heavy negative log-domain tone placement now targets a rendered median of `0.32` instead of `0.30`, while the shadow saturation guard tightens from `0.58..0.78` to `0.54..0.74`. Low-range negatives keep the brighter `0.35` target, and positive scans remain on their separate tone path.
- Status: `review_required`, with all 12 frames rendered and `0` failed frames. Remaining issues are still unchanged calibration/reference evidence and uncalibrated colour-review flags.
- Comparator against the saturated-shadow denoise baseline reported `comparison.status=comparable` and `issues=none`.
- Mean/min/max/range rendered midtone p50 is now `0.340218` / `0.317318` / `0.356560` / `0.039242`. Relative to the previous baseline, mean midtone increased `+0.006905`, minimum midtone increased `+0.020497`, and midtone range narrowed `-0.020497`, so the previously dark frames no longer sit near `0.30`.
- The four dark/high-residual frames lifted without changing clipping or gamut: `RAW_0001` midtone p50 `0.300477 -> 0.321269`, `RAW_0008` `0.296821 -> 0.317318`, `RAW_0011` `0.299126 -> 0.319893`, and `RAW_0012` `0.298447 -> 0.319249`.
- Shadow saturation p95 mean/max improved `0.319529` / `0.580000 -> 0.306196` / `0.540000`; the same four dark/high-residual frames now cap at `0.540000` instead of `0.580000`.
- Preserved gamut min and post-tone high clipping stayed unchanged at `0.979744` and `0.000000`. Chroma residual p95 mean moved only `+0.000318`, while the max improved slightly `-0.000176`; flat chroma residual p95 mean moved `+0.000974` and max `+0.003477`, both below the comparator's material regression threshold for the intentional brighter placement.
- Roll-suite validation now also records `midtone_luminance_p50_max` and `midtone_luminance_p50_range`; `--compare-roll-suite` derives these from per-frame midtone values when the baseline JSON predates the aggregate fields.
- Non-TESTROLL LOGAN strict check after this change: `output/validation/logan_shadow_lift_guard_strict_20260522.md` / `.json` rendered `5959x3670`, kept `render_review_status=reviewable`, `stitch.decision=accepted`, `render_input_source=fastica_separated_transmittance`, `selected_candidate=gamut_trusted_image_matrix_blend`, `selected_quality_score=1.794397`, `post_scale_preserved_ratio=0.979114`, and passed `--strict` with `summary_baseline_status=comparable` and `issues=none`. LOGAN shadow saturation p95 is now `0.540000`; chroma residual p95 stayed essentially unchanged at `0.093204`.

Fresh TESTROLL DNG model-review shadow cleanup on 2026-05-27:

- Baseline artifact: `output/goal_testroll_current_subset_20260527.md`
- Current artifact: `output/goal_testroll_model_shadow_cleanup_subset_20260527.md`
- Visual comparison: `output/goal_testroll_model_shadow_cleanup_before_after_20260527.png`
- Command: `cargo run --release --locked --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --roll-suite-frame RAW_0001,RAW_0005,RAW_0008,RAW_0011,RAW_0012 --bit-depth 16 --render-input direct-density --input-mode negative --quality-mode balanced --compare-roll-suite output\goal_testroll_current_subset_20260527.json --output-dir output\goal_testroll_model_shadow_cleanup_subset_20260527 --summary-json output\goal_testroll_model_shadow_cleanup_subset_20260527.json --summary-md output\goal_testroll_model_shadow_cleanup_subset_20260527.md --quiet`
- Policy change: `review_model_plausibility` frames now keep bounded shadow chroma cleanup enabled, while neutral highlight and midtone cleanup remain disabled and `color_trust_state` remains `review_required`.
- Status: `review_required`, with all 5 selected DNG frames rendered and `0` failed frames. The comparator reported `comparison.status=comparable` and `issues=[]`.
- `RAW_0011.dng` was the targeted model-plausibility frame: shadow saturation p95 improved `0.841849 -> 0.470662`, shadow RGB delta improved `0.009 -> 0.002`, and shadow chroma compression became `0.190654`.
- The selected-subset shadow saturation p95 mean/max improved `0.490247` / `0.841849 -> 0.416010` / `0.540000`.
- Midtone placement was unchanged: mean/min/max/range stayed `0.403397` / `0.317542` / `0.531507` / `0.213964`. Post-tone clipping stayed `0.000000` high and `0.000000` low, and preserved-gamut minimum stayed `0.981210`.
- The tradeoff was small and localized: `RAW_0011` chroma residual p95 moved `+0.000599`, and selected-subset chroma residual p95 mean moved `+0.000120`, both below the roll-suite comparator's material regression threshold.
- Non-TESTROLL guard: `output/validation/old_testroll_model_shadow_cleanup_raw0055_20260527.md` compared `OLD_TESTROLL` `RAW_0055.dng` against `output/validation/old_testroll_chromadamp055_raw0055_20260523.json` with `comparison.status=comparable` and `issues=[]`; the frame remained `candidate_risk=safe`, `tone_color_trust_state=trusted`.

Fresh TESTROLL DNG high-range shadow lift on 2026-05-27:

- Baseline artifact: `output/goal_testroll_highrange_shadow_baseline_subset_20260527.md`
- Current artifact: `output/goal_testroll_highrange_shadow_lift_subset_20260527.md`
- Full-roll artifact: `output/goal_testroll_full_highrange_shadow_lift_20260527.md`
- Visual comparison: `output/goal_testroll_highrange_shadow_lift_before_after_20260527.png`
- Command: `cargo run --release --locked --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --roll-suite-frame RAW_0003,RAW_0004,RAW_0005,RAW_0008,RAW_0010,RAW_0016 --bit-depth 16 --render-input direct-density --input-mode negative --quality-mode balanced --compare-roll-suite output\goal_testroll_highrange_shadow_baseline_subset_20260527.json --output-dir output\goal_testroll_highrange_shadow_lift_subset_20260527 --summary-json output\goal_testroll_highrange_shadow_lift_subset_20260527.json --summary-md output\goal_testroll_highrange_shadow_lift_subset_20260527.md --quiet`
- Policy change: high-range shadow-heavy negatives now allow the log-domain tone fit to place mapped p95 up to `0.95` instead of `0.92`, so the median target is not unnecessarily pulled back to the old dark placement when highlight clipping remains absent.
- Focused subset status: `review_required`, with all 6 selected DNG frames rendered and `0` failed frames. The focused comparator reported `comparison.status=comparable` and `issues=[]`.
- The intended dark-frame lift was localized to `RAW_0003.dng` and `RAW_0004.dng`: midtone p50 improved `0.202524 -> 0.305494` and `0.222864 -> 0.320378`; selected-subset midtone p50 min/range improved `0.202524` / `0.342382 -> 0.305494` / `0.239412`.
- Highlight detail stayed bounded: `RAW_0003` render p95 moved `0.930267 -> 0.950276`, `RAW_0004` moved `0.920479 -> 0.942079`, and post-tone high clipping stayed `0.000000`.
- Grain and colour tradeoffs remained inside the roll-suite guardrails: selected-subset chroma residual p95 mean moved `+0.001354`, flat chroma residual p95 mean moved `+0.000709`, preserved-gamut minimum stayed unchanged, and the comparator did not flag material residual, balance, gamut, or clipping regressions.
- Full 17-frame current-code rerun rendered every DNG with `0` failed frames. Relative to the earlier full baseline, midtone p50 min/range improved `0.202524` / `0.342382 -> 0.305494` / `0.239412`; post-tone high clipping stayed `0.000000`, preserved-gamut minimum stayed `0.978597`, chroma residual p95 mean moved only `+0.000299`, and flat chroma residual p95 mean moved `+0.000283`. The full comparison status remained `review_required` only because the previous full baseline had `RAW_0016.dng` failed and the new run rendered it as reviewable.
- Non-TESTROLL guards: `output/validation/logan_highrange_shadow_lift_strict_20260527.md` stayed strict-comparable against `tests/fixtures/baselines/logan_summary_baseline.json` with `summary_baseline_status=comparable` and `issues=[]`; `output/validation/old_testroll_highrange_shadow_lift_raw0055_20260527.md` compared `OLD_TESTROLL` `RAW_0055.dng` against `output/validation/old_testroll_chromadamp055_raw0055_20260523.json` with `comparison.status=comparable` and `issues=[]`.

Focused tests run after the audit, anchor-support shadow cleanup, weak-neutral highlight cleanup, saturated-shadow denoise relaxation, shadow-heavy midtone lift, and tightened shadow guard:

```powershell
cargo test --test test_density --test test_tonemap --test test_colorspace
cargo test --test test_validation
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Result: targeted density/tone/colorspace/validation tests, full `cargo test`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` passed on 2026-05-22. Because the `V:` target volume had only about 80 MB free and hit Windows PDB/disk-space errors, the successful test and clippy runs used `CARGO_TARGET_DIR=C:\Users\kszmyd\AppData\Local\Temp\scanstitch-cargo-target-test` and `RUSTFLAGS=-C debuginfo=0`. Coverage includes robust density Dmax selection, direct-density normalization, direct-density fallback selection/rejection, gamut-safe blend selection, shadow/low-range negative tone placement, highlight chroma repair, near-neutral highlight cleanup, weak-neutral bounded highlight cleanup, anchor-review bounded highlight/shadow cleanup, grain diagnostics/denoise behavior, and report validation.

## Reproducible Commands

Targeted single-frame final checks use the same settings as the successful shadow-lift artifacts:

```powershell
.\target\release\scanstitch-validate.exe --component1 TESTROLL\RAW_0001.tif --component2 TESTROLL\RAW_0001.tif --force-no-stitch --bit-depth 16 --render-input direct-density --input-mode negative --base-color 21737.428571,5322,2938 --base-color-source roll_consensus_base --base-color-confidence 0.98 --base-color-reason "selected stable roll base cluster from 9/12 frame candidates; rejected 3 darker candidate(s) against the roll high-transmittance envelope" --output-dir output\testroll_negative_dynamic_range_shadow_lift_20260520\raw-0001
```

The latest full current-code TESTROLL rerun with the shadow-heavy lift and tighter guard is:

```powershell
cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --bit-depth 16 --render-input direct-density --input-mode negative --quality-mode balanced --compare-roll-suite output\testroll_negative_shadow_denoise_relax_compare_roll_suite_20260522.json --output-dir output\testroll_negative_shadow_lift_guard_roll_suite_20260522 --summary-json output\testroll_negative_shadow_lift_guard_roll_suite_20260522.json --summary-md output\testroll_negative_shadow_lift_guard_roll_suite_20260522.md
```

The fresh 2026-05-21 TESTROLL verification used:

```powershell
cargo run --release --bin scanstitch-validate -- --roll-dir TESTROLL --roll-suite --bit-depth 16 --render-input direct-density --input-mode negative --output-dir output\testroll_negative_current_fields_20260521 --summary-json output\testroll_negative_current_fields_20260521.json --summary-md output\testroll_negative_current_fields_20260521.md
```

The fresh 2026-05-21 non-TESTROLL LOGAN check used:

```powershell
cargo run --release --bin scanstitch-validate -- --fixture logan --compare-summary tests\fixtures\baselines\logan_summary_baseline.json --output-dir output\validation\logan_current_non_testroll_20260521 --summary-json output\validation\logan_current_non_testroll_20260521.json --summary-md output\validation\logan_current_non_testroll_20260521.md
```

The accepted LOGAN debug and baseline refresh used:

```powershell
cargo run --release --bin scanstitch-validate -- --fixture logan --debug --compare-summary tests\fixtures\baselines\logan_summary_baseline.json --output-dir output\validation\logan_current_debug_20260521 --summary-json output\validation\logan_current_debug_20260521.json --summary-md output\validation\logan_current_debug_20260521.md --write-summary-baseline output\validation\logan_current_debug_20260521.baseline.json
Copy-Item -LiteralPath output\validation\logan_current_debug_20260521.baseline.json -Destination tests\fixtures\baselines\logan_summary_baseline.json
cargo run --release --bin scanstitch-validate -- --fixture logan --compare-summary tests\fixtures\baselines\logan_summary_baseline.json --strict --output-dir output\validation\logan_strict_current_baseline_20260521 --summary-json output\validation\logan_strict_current_baseline_20260521.json --summary-md output\validation\logan_strict_current_baseline_20260521.md
```

## Tradeoffs And Limits

- The final shadow-lift improves dark TESTROLL frames materially, but it can increase highlight chroma compression on high-contrast frames. The checked frames retained zero high clipping and bright-neutral p95 below the shoulder.
- Several TESTROLL frames remain `review_required` because no scanner/roll calibration or reference patches are configured. The image-derived path is safer than the initial matrix, but it is not calibrated colour proof.
- Snow-heavy scenes still show blue/cool bias in some shadows. The pipeline now surfaces this through candidate risk and tone-colour trust fields rather than silently declaring the colour final.
- The full 12-frame current-code rerun after the final shadow-lift and anchor-support cleanup completed successfully in `output/testroll_negative_anchor_shadow_cleanup_roll_suite_20260520`.
- LOGAN is now rebaselined to the safer current colour path and passes strict compact comparison. It remains an uncalibrated sample, so this is non-TESTROLL regression evidence, not calibrated colour proof.

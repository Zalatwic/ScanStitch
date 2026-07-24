# TESTROLL Assessment: does ScanStitch produce finished, perfected images?

> Current-code update, 2026-07-20: the orientation limitation described in this June audit is no
> longer a missing processing path. `--orientation-correction` now applies an explicit
> metadata-relative D4 transform before scanner linearization, crop, deskew, and stitch; reports bind
> the metadata-oriented and final pixel hashes plus their composed scanner transform. RAW_0011 was
> rerendered upright with `rotate-180`, while its review draft remains deliberately unapproved.
> RAW_0006 also rerendered with the formerly retained right scanner/rebate strip removed (77 pixels)
> by a conservative, recorded gradual-boundary fallback. Both proofs remain non-reviewable and
> visibly colour-inaccurate because real film-response/dye-crosstalk and colour-target evidence is
> still absent. The exact final-source RAW_0006 evidence is retained under
> `output/audit/negative-testroll-raw0006-borderfix-final-20260720/`. The detailed results below are
> the historical 2026-06-03 snapshot; use
> `PROJECT_ASSESSMENT.md` and `docs/validation-corpus-status.md` for the current verdict.

- Date: 2026-06-03
- Roll: `TESTROLL/` — 17 frames `RAW_0000.dng`..`RAW_0016.dng`
- Build: `cargo build --release` clean (24s, 0 warnings). Test suite green (`cargo test --release -- --test-threads=1`, exit 0; consistent with the documented 287 pass / 0 fail / 1 ignored).
- Scope: empirical, grounded in running the current code on this roll and looking at the actual rendered output. Complements the code-level `PROJECT_ASSESSMENT.md`; does not replace it.

One question: fed this real roll, how close does the pipeline get to finished, perfected images, where exactly is the gap, and how do we close it.

## 1. Executive verdict

The mechanical pipeline is **solid and reliable**: it decodes the 23.5 MP linear-raw DNGs, removes the scanner rebate, classifies frames, detects the film base with high confidence, inverts, and tone-maps to a gamut-safe positive with **zero hard clipping** on every frame. On the easier frames it produces genuinely usable colour.

It does **not** reliably produce *finished, perfected* images on this roll. Three things stand between current output and "perfected":

1. **Colour is frame-dependent and often cast.** Out of 17 frames, **2 are colour-clean** (`risk=safe`, `trust=trusted`); **15 carry a real colour-confidence flag**. Hard frames (dark/tungsten interiors with little neutral content) render with a strong cast — e.g. RAW_0001 comes out heavily magenta/pink. Easier frames (a daylight golf scene, a brighter interior) look fair-to-good with a mild cool cast.
2. **Orientation is not normalized.** Output inherits scan orientation, which is inconsistent across this roll: RAW_0006 is upright, but RAW_0001 and RAW_0014 render **upside-down** (the date-stamp and subjects are 180° rotated). A "finished" image must be the right way up.
3. **Saturation is muted.** To stay gamut-safe the tone stage compresses chroma hard (shadow chroma compression mean 0.35, up to 0.53), which desaturates rather than corrects the cast.

The localizing contrast: the same colour/tone engine renders the positive/slide reference fixtures (the "birding" set) beautifully — clean whites, natural skin, saturated colour. The bottleneck is specifically the **negative inversion → colour-matrix** stage on colour-difficult frames, not tone mapping or colourspace conversion.

Bottom line: **reliable scan mechanics and good results on easy frames; cast, rotation, and muting on hard frames.** The pipeline correctly refuses to over-commit colour when it lacks neutral support — but "safe" is not "finished."

## 2. Capability matrix (this roll)

| Capability | Verdict | Evidence |
|-|-|-|
| DNG decode (linear raw) | Works | 17/17 readable, `DNG_LINEAR_RAW16` 5952x3944 16-bit (`tiff_io.rs:391,435`) |
| Rebate / scanner-bar crop | Works | RAW_0001 height 3944 -> 3686 |
| Frame classify / stitch | Works (stitch n/a) | intact frames, stitch skipped pre-score, conf 1.0 |
| Film-base detection | Excellent (roll) / variable (single) | roll-consensus base conf 0.98, 17/17 agree; single-frame base 0.35 (RAW_0001) .. 0.99 (RAW_0014) |
| Density inversion (direct-density) | Works mechanically | confidence 1.0; colour quality depends on neutral support |
| True-colour matrix | Partial | both image-derived and DNG-prior matrices over-clip and are rejected -> neutral blend; 2/17 trusted |
| FastICA path (`--render-input ica`) | Broken here | RAW_0001 via ICA: green skin, magenta walls (permutation/sign failure) |
| Tone map / gamut safety | Works but over-mutes | 0 hard clipping; chroma compression 0.35-0.53 |
| Orientation | Not handled | RAW_0001, RAW_0014 render upside-down; RAW_0006 upright |
| Calibration gate | Structural block | no colour target -> `reference_patch_evaluation_missing` on all 17 |

## 3. The roll

17 single-frame VueScan-9 raw scans from a Nikon CoolScan 4000, `DNG_LINEAR_RAW16`, 5952x3944 (23.5 MP), 16-bit, ~141 MB each. (A naive header read reports 248x164/8-bit — that is the embedded preview IFD; `tiff_io.rs:435` decodes the real raster.)

It is a **varied personal roll**, not a uniform set: a daylight **golf outing** (e.g. RAW_0006) and an indoor **gathering/interior** (e.g. RAW_0001 dark tungsten, RAW_0014 brighter). Mixed lighting (daylight + incandescent), mixed orientation, and **no colour target frame**. Several frames are dark interiors with little neutral content — close to a worst case for image-derived colour. Frames are intact singles, so **stitching is a no-op here** (correctly skipped); stitching is assessed from the committed LOGAN split-frame evidence, not from TESTROLL.

(Note: an older contact sheet in `output/` shows a *different*, winter roll — that was a previous 12-frame TIFF version of TESTROLL, since replaced by these 17 DNGs.)

## 4. Roll-wide results (current code, direct-density, negative)

Aggregate: `output/validation/TESTROLL/roll-suite.md`. Status `review_required`; **0 passed / 17 review / 0 failed.** The harness default already selects `--render-input direct-density` (the author's preferred bypass).

### 4.1 The gate has two layers — separate them

A frame "passes" only with zero issues (`scanstitch-validate.rs:8164-8183`). Two different things drive the flags:

- **Structural (process, not quality):** `reference_patch_evaluation_missing` fires on **all 17** because no calibration target exists. `calibration_not_applied` does **not** fire — DNG input auto-applies the embedded ColorMatrix1 as an advisory prior (`calibration_status=applied:17`). So this roll can never reach "passed" via `--roll-suite` alone, regardless of image quality.
- **Real image-quality signal:** `candidate_risk` (safe 2, review 15: `review_neutral_support` 8, `review_anchor_support` 4, `review_model_plausibility` 3) and `tone_color_trust_state` (trusted 2, review_required 15).

Honest read: **2/17 colour-clean (RAW_0014, RAW_0015), 15/17 genuinely flagged** — not "17/17 broken."

### 4.2 What works well

- **Film-base detection:** roll-consensus base `13484.8,5534.0,3019.5`, confidence 0.98, 17/17 agree, 0 rejected.
- **Crop, classify, stitch-skip:** all correct (see matrix).
- **Gamut safety:** post-tone clipped high/low max 0.000 across the roll.
- **Per-frame adaptivity:** mapping is not a blanket fallback — 2 frames accept the full `image_derived_matrix` (RAW_0005, RAW_0010), several use `gamut_stabilized_image_matrix_blend`.

### 4.3 Where it falls short (quantified)

- **Colour matrices over-clip and are rejected.** RAW_0001: image-derived would clamp 20.4%/ch, 36.5% total; DNG-prior 75.7%/156% — both exceed `MAX_LOW_GAMUT_CLIP_RATIO_PER_CHANNEL 0.08` / `_TOTAL 0.18` (`colorspace.rs:22-23`) and fall back to the neutral blend.
- **Weak neutral anchoring, root-caused:** for RAW_0001, 98.2% of neutral samples sit in a single luminance band, so the matrix fit is unstable -> `review_neutral_support`. Review threshold is `selected_quality_score > 3.0` (`colorspace.rs:86`); RAW_0001 scores 3.65.
- **Chroma compression masks rather than corrects** (shadow 0.35-0.53; highlight up to 0.39) — the muted look.
- **Precision halved:** 16-bit DNG samples compressed into a 14-bit working domain by default (`--bit-depth 14`).
- **Single-frame CLI is handicapped vs the roll harness:** standalone `scanstitch a a` uses frame-local base detection (RAW_0001: conf 0.35) instead of the roll-consensus base (0.98); even forcing `--base-color` to the consensus value did not remove RAW_0001's magenta (the cast is inherent to the frame, not the base estimate).

### 4.4 Representative frames

| Frame | Content | Mapping | Risk | Trust | Notes |
|-|-|-|-|-|-|
| RAW_0001 | dark interior, people | gamut_trusted_blend | review_neutral_support | review | heavy magenta, upside-down |
| RAW_0006 | daylight golf | gamut_trusted_blend | review_model_plausibility | review | upright, green grass good, mild cool cast |
| RAW_0005 | interior | image_derived_matrix | review_neutral_support | review | full image matrix accepted |
| RAW_0014 | brighter interior | gamut_stabilized_blend | safe | trusted | natural colour, upside-down |
| RAW_0015 | interior | gamut_stabilized_blend | safe | trusted | colour-clean |

## 5. Visual verdict (rendered proofs)

Perfect-mode `review_srgb.png` proofs in `output/testroll_assess/`:

- **RAW_0014 (trusted, base conf 0.99) — GOOD colour, wrong orientation.** Brick fireplace reads brick-red, wood cabinets natural, framed art warm, skin plausible — a believable tungsten interior. Proof that the negative path *can* finish well automatically when base + neutral support are solid. But rendered 180° upside-down.
- **RAW_0006 (golf, daylight) — FAIR/GOOD.** Correct orientation; green fairway reads well; mild cyan cast in sky/shadows; slightly muted. Usable.
- **RAW_0001 (dark interior) — POOR.** Strong magenta/pink cast in lit areas, muted, upside-down. Re-rendering with the roll-consensus base (`--base-color`) did **not** fix it — the cast is inherent to this low-neutral-support frame.
- **RAW_0001 via `--render-input ica` — BROKEN.** Magenta walls and **green skin**: a FastICA channel permutation/sign failure. Far worse than direct-density; confirms the author's direct-density default is the right call (E2).

## 6. Gap analysis — root causes (code pointers)

1. **Colour-matrix instability is the primary colour gap.** Image-derived and DNG-prior matrices over-clip on low-neutral frames and are rejected, leaving a neutral blend that desaturates and leaves a residual cast. Root: no robust neutral/anchor spread to constrain the fit (`colorspace.rs` candidate ranking + `MAX_LOW_GAMUT_CLIP_RATIO_*`).
2. **E3 shadow cast — unaddressed at root.** Inversion uses a **single shared** `D_max` across channels (`density.rs:289` `shared_robust_d_max`, applied `:311`, 0.995 percentile `:8`). Per-channel black points are not set independently, so shadows inherit a cast the tone stage then compresses.
3. **E2 FastICA premise unproven and demonstrably destructive here.** ICA resolves channel order by brute-forcing correlations (`ica.rs:358`) and sign by assuming dye density is right-skewed (`ica.rs:316-323`); on RAW_0001 it produced green skin. It sits on the documented `--render-input auto` default yet is bypassed by the roll-suite and author.
4. **No orientation normalization.** Output inherits scan orientation; this roll mixes upright and 180°-rotated frames.
5. **Structural gate conflates "no calibration" with "bad image"** (`scanstitch-validate.rs:8169`), masking the 2 genuinely-clean frames.
6. **Precision loss:** 16-bit -> 14-bit working domain by default.

## 7. Can flags reach "perfected" on this roll?

Not fully, without a colour target. The casts on hard frames stem from missing neutral spread, which no render flag synthesizes; `--base-color` did not fix RAW_0001. What the levers do achieve:

- `--render-input direct-density` — already the de-facto default; far better than ICA here.
- `--bit-depth 16` — stop discarding 2 bits on a perfect master.
- A real **calibration target** (one frame of a grey/colour card on this film+scanner, via `scanstitch-calibrate` + `--calibration-library`) is the lever that would actually move `review_required` -> `trusted` and lift the structural gate. This is the highest-impact action for rolls like this.

## 8. Prioritized roadmap to finished-perfected

1. **Add orientation normalization.** Detect and correct 180°/90° rotation (date-stamp OCR, face/content cues, or a per-roll manual flag) so output is consistently upright. Cheapest visible win toward "finished."
2. **Per-channel black-point in inversion (fixes E3 at root).** Replace the single shared `D_max` (`density.rs:289,311`) with independent per-channel robust black points (or constrained shared+offset). Highest-leverage colour fix; attacks the cast the tone stage is currently papering over.
3. **Constrain the colour matrix instead of reject+blend.** When image-derived/DNG matrices over-clip (`colorspace.rs:22-23`), solve a gamut-constrained / regularized matrix rather than collapsing to a neutral blend, so saturation survives without clipping. Targets the `review_neutral_support` majority.
4. **Calibration workflow for this scanner+film.** Shoot one grey/colour-card frame; ingest via `scanstitch-calibrate`; supply `--calibration-library`. Lifts the structural gate and gives the matrix a real anchor — the biggest single jump toward "passed/perfected."
5. **Decide ICA's fate (E2).** Either prove FastICA beats direct-density on a labelled set, or demote it off the `auto` default. The green-skin failure is concrete motivation.
6. **Default to a 16-bit working domain for perfect/master output** (or warn loudly) to stop silent precision loss; and **give the single-frame CLI roll context** (or document that rolls should go through `--roll-suite` for the consensus base).
7. **Separate "needs calibration" from "low quality"** in the validate gate so colour-clean frames (0014, 0015) are not reported identically to genuinely-flagged ones.

## 9. Appendix

### Commands run
```
.\target\release\scanstitch-validate.exe --roll-dir TESTROLL --roll-inventory
.\target\release\scanstitch-validate.exe --roll-dir TESTROLL --roll-suite
# Single-frame perfect-mode proof (file passed twice; --force-no-stitch):
.\target\release\scanstitch.exe TESTROLL\RAW_0001.dng TESTROLL\RAW_0001.dng `
  --force-no-stitch --quality-mode perfect --render-input direct-density -o output\testroll_assess\0001_dd
# Variants rendered: --render-input ica; frames RAW_0006, RAW_0014; --base-color 13484.8,5534.0,3019.5
```

### Key artifacts
- Roll-wide: `output/validation/TESTROLL/roll-suite.md` (+ `.json`), per-frame `output/validation/TESTROLL/raw-*/`.
- Proofs: `output/testroll_assess/{0001_dd,0001_ica,0001_dd_consensus,0006_dd,0014_dd}/review_srgb.png`.

### Load-bearing code citations
- Review gate: `colorspace.rs:86` (score 3.0), `:22-23` (clip 0.08/0.18), `:4505-4517` (trust-state logic).
- Inversion / E3: `density.rs:8` (0.995 percentile), `:289` (shared D_max), `:311` (inversion).
- ICA / E2: `ica.rs:358` (permutation by correlation), `:316-323` (sign by skewness).
- Validate gate: `scanstitch-validate.rs:8164-8183` (issue conditions).
- DNG decode: `tiff_io.rs:391` (dispatch), `:435` (raster decode). Single-frame: `cli.rs:177`, `frame_classify.rs:264`.

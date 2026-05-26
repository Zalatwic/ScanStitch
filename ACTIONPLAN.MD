# CLAUDE_FINDINGS Actionability Review

Date: 2026-05-26
Scope: read-only triage of `CLAUDE_FINDINGS.MD` against the ScanStitch goal: stitch RAW scans, invert negatives, and preserve or improve the most capable practical color reconstruction process.

No project code was changed for this review. The original findings document is treated as a triage input, not as an implementation brief.

## Verification Snapshot

- `git ls-files`: 40 tracked files.
- `git status --short --untracked-files=all`: 131 rows; branch `MASTER` is ahead of `origin/MASTER` by 3 commits.
- `OLD_TESTROLL/` plus `OLD_SLIDE_TESTROLL/`: 72 files, 10,139,516,930 bytes, 9.44 GiB.
- `git check-ignore` confirms `output/`, `TESTROLL/`, and root-level `LOGAN043.tif` are ignored; `OLD_TESTROLL/`, nested `.dng`, `OLD_SLIDE_TESTROLL/`, and nested `.tif` files are not ignored.
- Rust counts: 26 source files, 43,577 source lines; 14 test files, 17,271 test lines.
- Test count: 265 `#[test]` attributes, with 24 inline in `src/` and 241 in `tests/`.
- CI: `.github/workflows/ci.yml` uses `ubuntu-latest` and does not pass `--test-threads=1`.
- Color matrix check: coded ProPhoto white maps to `[0.9642, 1.0, 0.8249]`; declared `D50_WHITE` is `[0.9642, 1.0, 0.8251]`.
- Validation evidence aligns with `docs/color-reconstruction-audit.md`: synthetic and LOGAN evidence are useful but not sufficient proof of color reconstruction quality.

## Rubric

- **Act now**: prevents data loss, preserves source control, fixes confirmed pipeline/color correctness defects, or prevents untrustworthy output/reporting.
- **Design/experiment**: plausible technical concern that needs A/B evidence, calibration evidence, or clearer policy before implementation.
- **Defer**: cleanup, consolidation, UI simplification, dependency simplification, or code-size work that does not block the core scan-to-color goal.
- **Reject/reframe**: claims whose main premise is "too complex" when the complexity supports ambitious color reconstruction. Reframe as a need for evidence, diagnostics, or maintainability boundaries.

## P0 Safety / Integrity

| ID | Actionability | Verified evidence | Recommended next action | Validation gate |
|-|-|-|-|-|
| C1 | Act now | Private scan dirs are untracked and not ignored; 9.44 GiB at risk. | Add defensive ignore rules for old/test roll dirs and nested RAW/TIFF/DNG files. | `git check-ignore -v OLD_TESTROLL OLD_TESTROLL\RAW_0000.dng OLD_SLIDE_TESTROLL OLD_SLIDE_TESTROLL\BIRDING039_VARIED_EXPOSURE.tif` |
| C2 | Act now | Only 40 tracked files; new subsystems are untracked; unstaged diff is very large. | Preserve current tree on a branch and commit intentional chunks after C1. | `git status --short --untracked-files=all` contains no untracked source/docs intended for version control. |
| C3 | Act now | Five commits total; current branch ahead 3; most current work is not committed. | Stabilize history with commit-per-change cadence before broad edits. | `git log --oneline --decorate --max-count=20` plus clean source/docs status. |

## P1 Confirmed Correctness

| ID | Actionability | Verified evidence | Recommended next action | Validation gate |
|-|-|-|-|-|
| E1 | Act now | `PROPHOTO_TO_XYZ_D50` maps white to Z `0.8249`; declared D50 Z is `0.8251`. | Fix the constant or derive matrix pair from published values; add white-point and inverse tests. | Targeted colorspace test asserting `RGB white -> D50` within tolerance and forward/inverse identity. |
| E3 | Design/experiment | Shared `D_max` is used for all channels; shadow cast risk is plausible but not proven. | Document the shared-vs-per-channel invariant or A/B it on real shadow/neutral scenes. | Neutral-shadow regression fixture passes under chosen policy. |
| E5 | Act now | Density transmittance paths use `.max(EPSILON_T)` without upper clamp; sibling positive helper clamps. | Clamp to `<= 1.0` or reject invalid bit-depth/input range before density conversion. | `cargo test --test test_density` with an over-range sample test. |
| F1 | Act now | Nine `expect()` state guards in `pipeline.rs`, including film-base and direct-density assumptions. | Replace recoverable pipeline panics with structured reportable errors. | Pipeline test proves hard base/density failure returns an error or reportable phase failure without panic. |

## P2 Color Evidence

| ID | Actionability | Verified evidence | Recommended next action | Validation gate |
|-|-|-|-|-|
| D3 | Design/experiment | 123 `const` scoring knobs in `colorspace.rs`; weights range from `0.005` to `25.0`. | Normalize and document scoring policy; do not delete metrics solely for complexity. | Candidate-rank tests plus real fixture deltas show perceptual metrics can affect selection. |
| E2 | Reject/reframe | ICA implementation and direct-density fallback exist; no real ground-truth proof that ICA improves color. | Reframe as A/B validation, not "remove ICA." Compare ICA, direct-density, calibrated fixed unmixing, and scanner/roll calibration paths. | Multi-scene, multi-film-stock corpus with calibrated and uncalibrated cases passes strict gates. |
| E4 | Design/experiment | Scanner/film constants lack derivation across density, ICA, tone, base detection, border detection, and positive mode. | Convert constants into documented calibration/scoring policy with sensitivity notes. | Threshold tests pin intended behavior and docs explain calibration rationale. |
| G1 | Act on evidence gap | LOGAN real test is ignored; color audit says synthetic/LOGAN evidence is insufficient. | Build a validation-ready CoolScan corpus with compact baselines and calibration/reference evidence. | `scanstitch-validate --fixture-coverage --strict` and `--fixture-suite --strict` pass for validation-ready fixtures. |

## P3 Trust / Operations

| ID | Actionability | Verified evidence | Recommended next action | Validation gate |
|-|-|-|-|-|
| B1 | Act | README says `~8600` source lines and `~2800` test lines; actual is 43,577/17,271. | Regenerate or remove static counts. | Count command and README agree, or README avoids exact counts. |
| B2 | Act | README architecture counts are stale, including `scanstitch-validate.rs` 147 vs 9,953. | Replace static per-file counts or script them. | No stale hard-coded architecture counts. |
| B3 | Act | README says 92 tests; actual count is 265. | Update or remove the count and module breakdown. | `#[test]` count matches docs, or docs avoid exact count. |
| B4 | Act | CHANGELOG omits calibration library, calibrate binary, positive mode, and validation scope. | Rewrite unreleased section from current tree. | Changelog names all new binaries and major modes. |
| B5 | Act | `scanstitch-calibrate` is only lightly referenced in README; full usage lives in docs. | Add concise README usage or explicit docs link for calibration and positive mode. | README search finds runnable calibrate examples or a clear `docs/color-calibration.md` pointer. |
| B6 | Reject/reframe | No `MEMORY.md` exists in this repo; stale agent-memory claims are externally unverifiable here. | Do not treat agent memory as authoritative; make repo docs/count scripts the source of truth. | Docs are generated or locally verifiable from repo state. |
| G2 | Act | CI runs Ubuntu while Quickstart documents Windows/MSVC setup. | Add Windows CI or document Linux as a supported validation platform. | GitHub Actions includes `windows-latest`, or docs stop implying Windows-only validation. |
| G3 | Act | README requires `--test-threads=1`; CI omits it. | Reconcile the requirement. | CI uses `--test-threads=1`, or README removes the requirement after proving tests are parallel-safe. |
| G4 | Defer/reframe | CI does not assert test count, but count assertions are lower value than accurate docs. | Prefer generated docs or a count-check script over hard CI count assertions. | Count drift cannot silently persist in documentation. |

## P4 Cleanup Later

| ID | Actionability | Verified evidence | Recommended next action | Validation gate |
|-|-|-|-|-|
| D1 | Defer/reframe | Validation code is 17,514 source lines, but it supports diagnostics and evidence. | Split only along useful ownership boundaries; do not shrink validation at the cost of color evidence. | Summary, baseline, synthetic, and fixture tests stay behaviorally unchanged. |
| D2 | Defer | Nineteen `synthetic_case_*` helpers live in `validation.rs`. | Move to test/support code when touching validation. | Synthetic suite summaries unchanged. |
| D4 | Defer | `pipeline.rs` is 3,223 lines and orchestrates many modes. | Extract after F1/state handling is clarified. | Pipeline integration tests unchanged. |
| D5 | Defer | UI uses both `ratatui` and `minifb` with channels. | Keep unless maintenance pain is proven; simplify only if workflow quality is preserved. | Interactive preview/save workflow still works after any simplification. |
| D6 | Defer | Both `image`+TIFF and `tiff` crates are present; `ndarray`+`nalgebra` are defensible but undocumented. | Consolidate only with IO coverage and no loss of DNG/TIFF support. | TIFF/DNG read-write tests pass. |
| D7 | Defer with caution | Duplicated percentile helpers and mirrored validation fields are confirmed; current clone keeps fields equal. | Collapse mirrors/helpers when changing validation; prefer one authoritative metric location. | Baseline comparison tests catch candidate-risk changes from the authoritative field. |
| F2 | No action | Confirmed: one non-test `.unwrap()` and no `panic!`, `todo!`, or `unimplemented!` in `src`. | Preserve this error-handling discipline. | `rg "unwrap\(|panic!|todo!|unimplemented!" src` remains clean except justified sites. |

## Decision Notes

- Do not simplify the color pipeline merely because it is complex.
- Preserve or increase color sophistication when it is physically grounded, measurable, documented, and diagnosable.
- ICA, direct-density rendering, calibrated fixed unmixing, and scanner/roll calibration should be judged by measured fidelity and diagnostics, not by code size.
- Unsupported "magic constants" should become documented, tested calibration/scoring policy.
- The central evidence gap is a validation-ready, multi-scene, multi-film-stock CoolScan corpus with calibration/reference evidence and tracked compact baselines.

## Suggested Execution Order

1. Patch `.gitignore` to protect private scan data, then verify with `git check-ignore`.
2. Stabilize the working tree with intentional commits or a branch checkpoint.
3. Fix ProPhoto D50 matrix behavior and density clamping/range validation.
4. Replace recoverable pipeline panics with structured errors.
5. Reconcile README, CHANGELOG, CI platform, and test invocation claims.
6. Design A/B validation for ICA vs direct-density vs calibrated/fixed unmixing.
7. Build the real CoolScan validation corpus and make strict fixture gates pass.
8. Defer cleanup/refactors until safety, correctness, and evidence gates are in place.

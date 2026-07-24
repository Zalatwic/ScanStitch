# CLAUDE_FINDINGS.MD — Forensic Review of ScanStitch

> **SUPERSEDED by `PROJECT_ASSESSMENT.md` (2026-06-03).** Retained only as the historical 2026-05-26 audit.
> Most of its top findings — C1–C3 (repo hygiene), E1 (colour matrix), E5 (transmittance clamp), F1 (pipeline
> panics) — have since been resolved. Verify against the current tree before acting on anything below.

Reviewer: Claude (Opus 4.7). Date: 2026-05-26. Scope: full working tree at `V:\PERSONAL\PHOTOGRAPHY\SCANSTITCH`.
Mandate: infer the project's goals, then catalogue every mistake as harshly and exhaustively as the
evidence supports, and finish with a prioritized remediation plan. Read-only review; no project file
was modified to produce it. All line counts are `wc -l`; all matrix claims were verified
numerically; all sub-agent claims were re-verified against primary source before inclusion (one
"critical" sub-agent finding was checked and proven false — see Appendix B).

---

## 1. Verdict

A focused ~3,000-line personal film-scanning tool (initial commit, 2026-02-28) has metastasized into
a **43,577-line, five-binary system** while its README, CHANGELOG, git history, and the agents' own
persistent memory all froze near the original and now describe a project that no longer exists.

The dominant pathologies, in order of how much they should worry you:

1. A repository so mismanaged that **9.1 GB of personal photographic scans are one `git add -A`
   away from being committed**, and only 40 of the project's files are tracked at all — the other
   ~60% of the code lives solely as uncommitted working-tree state.
2. Documentation that **understates the codebase by ~5x and one file by 67x**, claims 92 tests when
   there are 265, and omits entire subsystems from its changelog.
3. **Industrial-scale over-engineering**: a validation/scoring apparatus that is ~40% of the
   codebase, fronted by a color "quality score" built from 124 unexplained constants whose weights
   span a 5000x range — so the dozen sophisticated perceptual metrics it computes are numerically
   incapable of affecting the result.
4. A pipeline built on a **scientifically unsound premise** (FastICA as physical "dye separation")
   that shreds exactly the inter-channel color information a color pipeline exists to preserve, then
   patches the damage with heuristics — with no evidence it helps.
5. A **confirmed colorimetric defect**: the primary ProPhoto matrix is mistyped and does not map
   white to D50.

There is competent work here, and this review is careful to say so (Appendix A): the FastICA
*implementation* is textbook-correct, `stitch.rs` survived a full bug audit clean, the tone curve and
Bradford adaptation are right, and CI runs `clippy -D warnings`. That competence is exactly why the
surrounding negligence is damning — this was built by something that can write correct code and
simply did not keep the project honest. This is the signature failure of fast, unsupervised AI
iteration: the code sprints ahead and everything that makes a project trustworthy — accurate docs,
version control, scientific validation, scope discipline — is left for dead.

---

## 2. Method and confidence

Directly read in full or in depth: `Cargo.toml`, `README.md`, `QUICKSTART.md`, `CHANGELOG.md`,
`.gitignore`, `.github/workflows/ci.yml`, `src/constants.rs`, `src/density.rs`, `src/ica.rs`,
`src/colorspace.rs` (constants block + transform/scoring spot-reads), `src/color_calibration.rs`
(fit/confidence), `src/tonemap.rs` (curve), `src/validation.rs` (comparison logic), `src/pipeline.rs`
(panic guards), `tests/test_colorspace.rs`, the initial commit's README, and full git metadata.
Three deep sub-agents combed the four largest files; their findings were treated as leads and
**re-verified line-by-line** — several were corrected or rejected (Appendix B).

Verification techniques used: numerical matrix multiplication to check the color constants
(Appendix C); `wc -l` for every size claim; `git ls-files` / `git check-ignore` / `du` for hygiene;
`grep` counts for tests, panics, and duplication. Confidence is high on every item in Sections 4-9.

Not done, per the document-only constraint: no `cargo build`, no `cargo test`. "265 tests" is a
static count of `#[test]` attributes, not a count of passing tests; the project's compileability was
not confirmed in this review.

---

## 3. Inferred goals: what it was meant to be vs. what it became

The original intent is unambiguous from the first commit's own README — a clean, pure-Rust,
eight-phase CLI: remove scanner borders, detect the film base, classify/stitch split frames, invert
orange-mask density (`D_max − D'`), separate dye layers via FastICA, map to ProPhoto RGB D50, apply
a Naka-Rushton tone curve, write a 16-bit positive TIFF. One binary, one job, one line of
description per phase.

The current working tree is a different animal:

| Dimension | Initial commit `e25b29b` (2026-02-28) | Working tree (2026-05-26) | Drift |
|-|-|-|-|
| Source `.rs` files | 18 | 26 | +44% |
| Source lines (`wc -l`) | 3,031 (src + tests) | 43,577 (src) + 17,271 (tests) | ~14x src |
| Binaries | 1 | 5 | +4 |
| Largest file | — | `scanstitch-validate.rs`, 9,953 lines | — |
| Subsystems | 8-phase pipeline | + calibration-library engine, positive/slide mode, interactive TUI+GUI editor, benchmark harness, ~17.5k-line validation framework | sprawl |
| Phase docs | one line each | dense multi-clause paragraphs | accretion |

The trajectory is itself the finding. None of the added subsystems is inherently illegitimate, but
they were bolted on without ever asking whether a personal tool for one scanner needs a
calibration-library precedence engine, a dual-surface interactive editor, or a regression framework
larger than the pipeline. Nothing was ever removed or consolidated; complexity only accreted.

---

## 4. Findings: Documentation and self-tracking (severity High, pervasive)

The docs do not merely lag the code — they misstate it in the exact numbers a reader uses to judge a
codebase.

- **B1 — README understates code size ~5x. Severity: High.** `README.md:236` claims "~8600 lines of
  Rust code, ~2800 lines of tests." Actual: 43,577 src + 17,271 tests = ~60,800 lines (`wc -l`).

- **B2 — Per-file Architecture counts are frozen at the initial commit, off by up to 67x.
  Severity: High.** `README.md:210-234`, claimed vs actual:

  | File | README | Actual | Factor |
  |-|-|-|-|
  | `scanstitch-validate.rs` | 147 | 9,953 | 67.7x |
  | `validation.rs` | 461 | 7,561 | 16.4x |
  | `colorspace.rs` | 546 | 5,741 | 10.5x |
  | `tonemap.rs` | 584 | 2,746 | 4.7x |
  | `pipeline.rs` | 1,085 | 3,223 | 3.0x |
  | `stitch.rs` | 2,904 | 3,918 | 1.3x |

  The section a newcomer reads first to understand the codebase is fiction.

- **B3 — README claims "92 tests"; the real count is 265. Severity: High.** Verified 265 `#[test]`
  attributes (241 in `tests/`, 24 inline). `README.md:243-251`'s per-module breakdown is also wrong
  (e.g. tonemap "15"→40, validation "2"→53, colorspace "11"→37) and omits two test files that did
  not exist when written (`test_color_calibration.rs` 24, `test_interactive.rs` 7). See G1 for why
  the *number* is itself misleading.

- **B4 — CHANGELOG documents six bullets for work that added whole subsystems. Severity: High.**
  `CHANGELOG.md:3-10` never mentions the calibration library (`color_calibration.rs`, 3,031 lines),
  the `scanstitch-calibrate` binary, positive/slide mode (`positive_input.rs`), the interactive
  renderer (`interactive.rs`), or the 17.5k-line validation framework. A changelog omitting ~95% of
  the work is worse than none: it asserts completeness it does not deliver.

- **B5 — Undocumented binary and undocumented major features. Severity: Medium.** Five binaries
  exist; the README describes four (`scanstitch-calibrate` is invisible). The calibration-library
  flags and positive mode are listed in the flag table but have no usage documentation — a
  multi-thousand-line subsystem with no on-ramp.

- **B6 — The agents' own persistent memory is also wrong. Severity: Medium (meta).** It records "18
  modules, 44 tests, 10 commits" (actual 26/265/5), lists "no rayon parallelism used" (rayon is used
  at 12 sites in 4 modules), and cites a redundant pass in a function — `estimate_source_white` —
  that does not exist (grep confirms only `estimate_work_to_xyz*`). When docs, changelog, and the
  project's own memory all disagree with the code, the project has no reliable account of itself.

---

## 5. Findings: Repository hygiene and source control (severity up to Critical)

- **C1 — CRITICAL: 9.1 GB of personal scans are unprotected from accidental commit.**
  `OLD_TESTROLL/` (5.0 GB) and `OLD_SLIDE_TESTROLL/` (4.1 GB) hold raw TIFF scans — personal
  photographs. They are untracked **and not gitignored**. `.gitignore` ignores `/TESTROLL/` and
  root-level `/*.tiff`, but neither pattern matches these renamed directories or their nested TIFFs;
  `git check-ignore` confirms only `output/`, `docs/plans/`, and `TESTROLL/` are ignored. A single
  `git add -A` (or an agent "tidying up") commits 9.1 GB of personal images into history, where they
  are extremely painful to purge. This is the only finding that can cause irreversible harm. Fix it
  before touching anything else.

- **C2 — Only 40 files are tracked by git; the project is mostly uncommitted. Severity: High.**
  Every pre-existing source file shows Modified, and ~25 new files — the entire calibration
  subsystem, three of five binaries, the CHANGELOG, all new docs/schemas — are untracked. The
  majority of the 43,577 source lines exist only as working-tree state. This is not version control;
  it is a single point of failure sitting next to a `.git` directory.

- **C3 — The five-commit history is a facade. Severity: High.** Two commits on 2026-02-28, three on
  2026-05-01, then 25 days of unrecorded sprawl. The committed delta over the entire history is
  +12,508/−982 on top of the 3,031-line start (~14.5k committed lines); the majority of the current
  codebase has never been committed. You cannot bisect a regression, attribute a change, or recover
  prior state, because prior state was never recorded.

---

## 6. Findings: Scope explosion and over-engineering (severity High — the dominant disease)

- **D1 — The validation harness is larger than the product. Severity: High.**
  `scanstitch-validate.rs` (9,953) + `validation.rs` (7,561) = 17,514 lines, **40.2% of all source**,
  devoted to validation/regression scaffolding. The pipeline it exists to check is smaller than the
  checker. The largest single struct in `validation.rs` carries ~80 fields; the largest functions
  (`compare_summary_baseline` ~409 lines, `summarize_synthetic_result` ~527 lines) are mechanical
  field-by-field repetition.

- **D2 — Test logic lives inside library code. Severity: Medium.** `validation.rs` embeds 19
  `synthetic_case_*` functions — synthetic fixtures and assertions compiled into the shipping
  library, not confined to `tests/`. Bloats the binary, breaks encapsulation, blurs product vs. test.

- **D3 — The color "quality score" is elaborate machinery with no decision power. Severity: High.**
  `colorspace.rs:1-141` is a single 141-line wall of **124 `const` declarations** with exactly one
  explanatory comment (lines 79-80). They feed a candidate-ranking system that sums ~20 weighted
  penalties into a composite score. The fatal flaw: the penalty weights span **`0.005`
  (`REFERENCE_DELTA_E_MAX_PENALTY_WEIGHT`) to `25.0` (`NEUTRAL_FALLBACK_FIDELITY_PENALTY`) — a 5000x
  range** — all added into one scalar. So the sophisticated, expensive perceptual metrics the code
  carefully computes (CIEDE2000 RMS/max, hue linearity, saturation preservation, memory-color
  plausibility, spatial neutral consistency; weights 0.005-0.12) are arithmetically drowned out by a
  handful of large constants (25.0, 12.0, 10.0, 8.0) and **cannot change which candidate wins**. The
  system computes a dozen ways to measure color fidelity and then weights them into irrelevance. It
  is also, by its own author's admission via these constants, pure empirical tuning with no published
  basis — it will overfit the one scanner/film set it was tuned on. (Note: a sub-agent reported a
  "CRITICAL" RMS-divisor bug in the calibration fit; I verified `color_calibration.rs:976-1018` and
  it is **correct** — the `/3` divides by 3 XYZ components per patch as it should. Excluded.)

- **D4 — `pipeline.rs` is a 3,223-line orchestration god-module. Severity: Medium.** Phase
  sequencing, negative/positive branching, render-input selection, and report assembly are fused into
  one file — the natural choke point for every bug and merge conflict.

- **D5 — A dual user interface for a single-user tool. Severity: Medium.** `scanstitch-ui` runs a
  ratatui terminal UI **and** a separate minifb native preview window simultaneously, coordinated by
  crossbeam channels and worker threads. Two event loops, two render surfaces, two failure modes — to
  nudge a tone curve.

- **D6 — Redundant dependencies. Severity: Low.** Both the `image` crate (tiff feature) and the
  standalone `tiff` crate (two TIFF codecs), used together in `tiff_io.rs`; plus both `ndarray` and
  `nalgebra` (defensible but undocumented).

- **D7 — Pervasive copy-paste duplication, including in the regression gate. Severity: Medium.**
  The percentile/histogram helpers are reimplemented near-verbatim in at least four modules:
  `percentile_from_histogram` in `density.rs:155` and `ica.rs:444`; `percentile_from_sorted_values`
  in `colorspace.rs` and `tonemap.rs:664`. Worse, the validation layer stores the *same* metric in
  three redundant fields (`colorspace.candidate_risk`, `render.colorspace_candidate_risk` — a clone
  made at `validation.rs:4872` — and a flat field), and `compare_summary_baseline` reads 8 colorspace
  metrics from the `render.*` mirror (`validation.rs:3849`) while sibling lines and the baseline read
  from the authoritative `colorspace.*` (`:3562`). They are equal *only because* of the clone at
  :4872 — a latent regression-gate bug waiting for that invariant to lapse, and a vivid symptom of
  the struct duplication. (A sub-agent reported these 8 as confirmed data-corruption bugs; that
  over-claims — they are equal by construction today. But the inconsistency is real and fragile.)

---

## 7. Findings: Algorithmic and scientific correctness (severity mixed)

- **E1 — CONFIRMED DEFECT: the ProPhoto→XYZ matrix is mistyped and does not map white to D50.
  Severity: Medium.** `constants.rs:17` sets `PROPHOTO_TO_XYZ_D50[2][2] = 0.8249`. The published
  ProPhoto value is `0.82521`; the file's own `D50_WHITE` declares Z = `0.8251`. Numerical
  verification (Appendix C): the matrix maps RGB white (1,1,1) to Z = **0.8249**, an error of −0.0002
  vs. its own white point and −0.00031 vs. standard. Critically, **0.82521 rounded to four decimals
  is 0.8252, not 0.8249** — this is a transcription error, not rounding. The inverse matrix is a
  faithful inverse of the *wrong* forward matrix (round-trip identity to 3.8e-8), so the error is
  baked in consistently in both directions. Magnitude is small (~0.03%), but for a pipeline whose
  entire premise is colorimetric correctness, shipping a primary matrix that fails its own
  white-point identity is indefensible, and it proves the constants were never covered by a
  round-trip unit test.

- **E2 — QUESTIONABLE PREMISE: FastICA is the wrong tool, and for a color pipeline a destructive one.
  Severity: High.** `ica.rs` runs FastICA to "separate dye layers." Three compounding problems:
  1. *Independence is violated.* FastICA assumes the channels are linear mixtures of statistically
     independent sources. Film C/M/Y dye densities are driven by shared scene content and are
     strongly correlated, so the foundational assumption fails; the recovered "independent components"
     have no reason to be the physical dyes.
  2. *The transform destroys color information.* FastICA is ambiguous up to permutation, sign, **and
     scale**, and this implementation also removes the mean (the separated components are zero-mean —
     `ica.rs:329-332`). Permutation, sign, scale, and offset are precisely the inter-channel ratios
     and levels that *encode color*. A color-reconstruction pipeline runs its three color channels
     through a transform that discards color.
  3. *The fix-up is incoherent.* The code then spends `resolve_permutation` (`ica.rs:358-436`) —
     which picks the channel ordering that **maximizes correlation between the ICA outputs and the
     original, pre-ICA channels** — plus a skewness sign heuristic (`:276-327`, assuming "dye density
     is right-skewed", which fails on dark/high-key frames) plus per-channel renormalization, to claw
     back what ICA discarded. Re-aligning the output to the very channels you started from means: if
     ICA found something genuinely different, the re-alignment is wrong; if it didn't, ICA was
     pointless. Either way it is incoherent.

  Net: ICA here is, for color purposes, a no-op-with-extra-steps at best and an information shredder
  at worst, with **no validation anywhere in the codebase** that it improves color fidelity versus
  omitting it. The principled tool is a *fixed* unmixing matrix from scanner spectral sensitivities ×
  film dye absorptions — a calibration the project already has machinery for (`color_calibration.rs`)
  but does not use for dye unmixing. (To be fair: the FastICA *implementation* is textbook-correct —
  see Appendix A. The defect is conceptual, not in the code.)

- **E3 — Density inversion injects a shadow color cast via a shared D_max. Severity: Medium.**
  `density.rs:262-302` computes a per-channel robust `D_max`, collapses it to one
  `shared_robust_d_max` (the max across channels), and inverts every channel with it
  (`inverted = shared_robust_d_max − d`). Because the channels have different actual maxima, the
  black-clip (where `inverted ≤ 0`) is asymmetric across channels — the highest-density channel loses
  its top 0.5% to black while others may clip nothing — producing a channel-dependent cast in the
  deepest shadows. This may be a deliberate "keep inversion neutral, balance later" choice, but it is
  undocumented and unvalidated; either justify it with a comment and a neutrality test or invert
  per-channel.

- **E4 — Magic numbers without derivation, pipeline-wide. Severity: Medium (systemic).** Beyond the
  124 in colorspace (D3): density `DMAX_PERCENTILE 0.995`; ICA `0.01/0.995` percentiles and `4096`
  bins and the arbitrary init matrix `[0.7,0.5,0.3; …]` (`ica.rs:173`); tonemap
  `PERCEPTUAL_LUMINANCE_GAIN 15.0` and highlight knee `0.92`; base-detect `5%` strip,
  `MIN_BASE_REGION_LUMINANCE 512`, ratio `0.72`; border MAD multipliers `0.35/0.60` with clamps
  `40/500/80/900`; positive-mode orange-mask weights `0.35/0.25/0.25/0.15`. Each is a scanner/film-
  specific knob with no comment, citation, or sensitivity analysis.

- **E5 — No upper clamp on transmittance → silent negative densities on mislabeled bit depth.
  Severity: Low.** `density.rs:183` (`density_after_base`) and `:51` (`normalize_to_float`) compute
  `pixel/max_val` with only a lower `.max(EPSILON_T)` guard — no `.min(1.0)`. If a `--bit-depth 14`
  image actually contains values above 16383 (a plausible labeling mistake, since the positive-mode
  helper `:72` *does* clamp to `[0,1]` but this path does not), transmittance exceeds 1.0 and density
  goes negative, silently corrupting the inversion. Inconsistent clamping across sibling functions.

---

## 8. Findings: Error handling (severity Medium — narrower than it looks)

- **F1 — Pipeline orchestration panics on recoverable conditions instead of reporting them.
  Severity: Medium.** Of 63 `.expect()` calls across `src/`, most are in inline `#[cfg(test)]`
  modules. The real concern is the **9 in `pipeline.rs`** (lines 528, 532, 1759, 1769, 1877, 1879,
  1887, 2053, 2055) acting as state-machine guards. The worst is **`:1759`**:
  `.expect("negative input mode should have a film-base color before density inversion")` — film-base
  detection genuinely fails on difficult real scans, and here a recoverable "couldn't find the base"
  becomes a hard panic that produces **no `report.json`** — the diagnostic report being this tool's
  entire reason to exist. Four of the nine repeat the identical message `"positive density image
  should be available"`, evidence of a fragile `Option`-threaded state machine. These should
  propagate a structured error into the report.

- **F2 — Counterpoint (to keep the critique honest): error-handling discipline is otherwise good.**
  Exactly **1** bare `.unwrap()` in non-test code (`cv_adapter.rs`), and **zero**
  `panic!`/`todo!`/`unimplemented!` anywhere. There is no unfinished-stub problem.

---

## 9. Findings: Tests and CI (severity Medium — quantity over assurance)

- **G1 — The large test count is inflated by self-referential plumbing tests; the default suite has
  no ground-truth color validation. Severity: Medium.** `tests/test_colorspace.rs` (37 tests)
  hand-builds an ~80-field `ColorspaceDiagnostics` struct of synthetic scores and asserts that the
  *selector* routes them to the expected candidate. These verify that the heuristic scoring machine
  does what its code says — **not that the output color is correct**. The only real-image color check
  (the LOGAN stitch/render path) is `#[ignore]`-d and requires private TIFFs, so it never runs in CI.
  265 tests therefore buy high confidence in internal plumbing and ~none in output fidelity. The
  count is a vanity metric.

- **G2 — CI validates the wrong platform. Severity: Medium.** `.github/workflows/ci.yml:10` runs on
  `ubuntu-latest`, but the project is documented as Windows-only (MSVC toolchain, VS2022, onecore
  libs, custom `~/.cargo/config.toml`). A green CI does not demonstrate the documented build
  environment works, and the Windows-only toolchain is never exercised by CI.

- **G3 — CI omits the `--test-threads=1` the README says is required. Severity: Low.**
  `README.md:243` states tests must run with `--test-threads=1`; CI (`ci.yml:32-48`) runs them
  without it, and splits them across many invocations (e.g. `test_pipeline` and `test_stitch` alone).
  Either the single-thread requirement is bogus, or the tests share state and CI is racy. The
  invocation-splitting suggests the latter.

- **G4 — CI never asserts the test count; the "92" is verified nowhere. Severity: Low.** Counterpoint
  (credit where due): CI *does* run `cargo fmt --check` and `cargo clippy --locked --all-targets -D
  warnings`, which is genuinely strict and more than most personal projects bother with.

---

## 10. Severity summary

| Severity | Count | IDs |
|-|-|-|
| Critical | 1 | C1 |
| High | 9 | B1, B2, B3, B4, C2, C3, D1, D3, E2 |
| Medium | 12 | B5, B6, D2, D4, D5, D7, E1, E3, E4, F1, G1, G2 |
| Low | 4 | D6, E5, G3, G4 |

C1 is the only finding that can cause irreversible harm. The High cluster reduces to two themes: the
project lies about itself, and it grew without discipline.

---

## 11. Remediation plan

P0 — integrity / data loss (before anything else):
- Patch `.gitignore` to exclude `OLD_TESTROLL/`, `OLD_SLIDE_TESTROLL/`, a defensive `*TESTROLL*/`
  glob, and nested `**/*.tif`, `**/*.tiff`, `**/*.dng`; verify with `git status --ignored`. (C1)
- Commit or branch the working tree so ~26k lines stop existing only in an uncommitted worktree;
  adopt a commit-per-change cadence. (C2, C3)

P1 — correctness:
- Fix `constants.rs[2][2]` to `0.82521` (better: store the published ProPhoto↔XYZ D50 pair and derive
  the inverse programmatically); add a unit test asserting white→D50 within 1e-4 and forward∘inverse
  = identity. (E1)
- Resolve the ICA premise: either empirically validate it (recovered components must track real dye
  densities on a known color target) or replace it with a calibrated fixed unmixing matrix and delete
  the "dye separation" claim; document the skewness assumption's failure modes regardless. (E2)
- Document + test the `density.rs` shared-vs-per-channel `D_max` choice; add `.min(1.0)` (or a
  documented invariant) to the transmittance paths. (E3, E5)
- Re-weight or normalize the colorspace scoring so computed metrics can actually influence the result,
  or delete the metrics that cannot; move all magic constants to a documented, cited config with
  pinning tests. (D3, E4)

P2 — engineering / scope:
- Extract the 19 `synthetic_case_*` into `tests/`; split `scanstitch-validate.rs` + `validation.rs`
  into focused modules with a <1k-line orchestrator; collapse the triplicated metric fields and the
  quadruplicated percentile helpers. (D1, D2, D7)
- Replace the 9 `pipeline.rs` panic guards with `Result`/report-error propagation. (F1)
- Add at least one automated ground-truth color/render regression test that runs by default. (G1)
- Pick one UI surface; resolve the duplicate TIFF codec. (D5, D6)
- Point CI at a Windows runner (or document why Linux is sufficient) and reconcile `--test-threads=1`.
  (G2, G3)

P3 — documentation honesty:
- Regenerate README sizes/counts/binary list and the CHANGELOG from reality; script the counts so
  they cannot drift. Document the calibration library and positive mode. Update `MEMORY.md`. Relabel
  the investigation logs in `docs/` as non-authoritative. (B1-B6)

---

## Appendix A — Things that are genuinely correct or well done

Verified and sound; the harsh sections are bounded by these:
- `output/` is correctly gitignored with **zero** tracked files.
- `stitch.rs` (3,918 lines) passed a full bug audit with **no confirmed correctness bugs**: NCC
  mean-subtracts and guards `denom < 1e-10`, geometry uses saturating arithmetic and bounds checks,
  feather weights sum to 1, seam gain is clamped and clip-screened.
- The FastICA *implementation* is textbook-correct: two-pass covariance, PCA whitening with
  eigenvalue regularization, the correct symmetric fixed-point update (`g=tanh`, `g'=1−g²`), and a
  correct 6-permutation optimal assignment.
- `bradford_cat` builds the adaptation correctly (`M_LMS→XYZ · diag(D50/src) · M_XYZ→LMS`); both
  Bradford matrices match published values (round-trip identity to 1e-4).
- The Naka-Rushton tone curve is correct, including the clever `(1+σⁿ)` renormalization so input 1
  maps exactly to `shoulder_max` (`tonemap.rs:589-605`).
- The calibration RMS computation (`color_calibration.rs:1018`) is correct (per-component RMS).
- Error-handling discipline: 1 bare `unwrap`, 0 `panic!/todo!/unimplemented!`.
- `rayon` parallelism is real (12 sites). CI enforces `fmt` and `clippy -D warnings`.

## Appendix B — Sub-agent and memory claims corrected during verification

This review used sub-agents for breadth but trusted nothing without checking. Rejected/corrected:
- "CRITICAL: calibration RMS divisor is wrong" — FALSE; `/3` is the correct per-component divisor
  (`color_calibration.rs:976-1018`).
- "8 confirmed data-corruption bugs in `compare_summary_baseline`" — OVER-CLAIMED; the fields are
  equal by construction (clone at `validation.rs:4872`). Real issue is the latent inconsistency (D7).
- "CRITICAL: `output/` is tracked" (round 1) — FALSE; `git ls-files output/` = 0.
- "Read the original plan `docs/plans/2026-02-27-scanstitch.md`" (round 1) — that file does not exist
  on disk; the claim was a hallucination. Original goals here come from the initial commit's README.
- Memory "rayon unused" / "estimate_source_white redundant pass" / "92 tests" / "44 tests" — all
  stale or false; rayon is used, that function does not exist, and the count is 265.
- "`.to_owned()` copies in the NCC hot loop" (memory) — refuted; the copies are in the homography
  path, not per-pixel.
- Line counts: sub-agents' `wc -l` figures (9,953 / 7,561) are authoritative and used here; an earlier
  PowerShell `Measure-Object` pass undercounted by ~3.6% and was discarded.

## Appendix C — Numerical verification log (color constants)

Computed directly from `src/constants.rs`:
- `PROPHOTO_TO_XYZ_D50 · XYZ_D50_TO_PROPHOTO` = identity to max deviation 3.8e-8 → the coded inverse
  is a faithful inverse of the coded forward (the error in E1 is carried consistently, not a
  forward/inverse mismatch).
- `PROPHOTO_TO_XYZ_D50 · [1,1,1]ᵀ` = `[0.9642, 1.0000, 0.8249]`. Declared `D50_WHITE` =
  `[0.9642, 1.0000, 0.8251]`; published ProPhoto Z = `0.82521`. Z error: −0.0002 (vs own white),
  −0.00031 (vs standard). `0.82521` to 4 dp = `0.8252` ≠ `0.8249` → transcription error, not rounding.
- `BRADFORD_XYZ_TO_LMS · BRADFORD_LMS_TO_XYZ` = identity to max deviation 1.0e-4 → Bradford pair is
  correct at 4-decimal precision.

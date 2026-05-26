# Colour Reconstruction Policy

This document records the invariants behind the current negative-film colour pipeline. It is not a
claim that the constants are final; it is the policy that keeps changes measurable while the real
CoolScan corpus is still being built.

## Density Domain

Negative scans are normalized to transmittance, converted to optical density, and have the estimated
film base subtracted per channel. Phase 3 then computes a robust per-channel high-density percentile
and uses the maximum of those values as `shared_robust_d_max` for all three channels when inverting
to positive density.

The shared Dmax is intentional. It keeps red, green, and blue on one density scale after
per-channel orange-mask removal, so channel ratios are not independently stretched before ICA,
direct-density fallback, or colour-candidate scoring. Sparse clipped borders and dust are handled by
the robust percentile and by zero-clamp diagnostics instead of by per-channel range normalization.
Direct-density render input also divides by the same `shared_robust_d_max` before transmittance
conversion so it is bounded like ICA output without changing the Phase 3 density evidence.

Changing this to per-channel Dmax is a colour-model change, not a cleanup. It needs an A/B result on
real neutral-shadow and mixed-neutral scans, with `shared_robust_d_max`, per-channel
`robust_d_max`, `clamped_to_zero`, candidate ranking, neutral delta, and final tone/grain diagnostics
compared in the validation report.

## Candidate Scoring

Colour candidates are ordered by `lower_is_better`. The selected `quality_score` is an additive
diagnostic score split into:

- `technical_safety_score`: negative-gamut clipping, preserved gamut, exposure scale, and matrix
  condition/safety penalties.
- `color_fidelity_score`: neutral balance, calibration confidence, reference target residuals,
  anchor support/stability, rendered-tone colour risk, tone cleanup risk, density monotonicity,
  hue linearity, saturation preservation, memory-colour plausibility, spatial neutral consistency,
  and fallback penalties.

Hard gates run before score wins are accepted. Auto mode may prefer calibrated or scanner-prior
candidates only when they beat image-derived quality and do not regress gamut, neutral balance, or
reference-patch residuals. Direct-density render input can replace ICA only when it materially
improves destructive ICA clipping without increasing render-input risk or neutral regression.
Neutral fallback carries a large fidelity penalty because it preserves gamut by discarding scene
chroma; it should not outrank a safe matrix candidate.

Weights in `colorspace.rs` are decision policy. A weight change needs either a targeted synthetic
test that proves the intended rank/risk behavior or a fixture-suite baseline delta that documents the
real-image effect. Removing a metric because it is complex is not aligned with the project goal; the
metric should either stay reported, be replaced by stronger evidence, or be explicitly deprecated
after validation proves it no longer affects decisions.

## Constant Families

The constants fall into these ownership groups:

- **Physical encoding constants**: bit-depth maxima, `EPSILON_T`, D50/ProPhoto matrices, and Bradford
  adaptation matrices. These require direct mathematical tests.
- **Density and render-domain constants**: robust density percentiles, histogram sizes, direct-density
  fallback margins, and normalized-transmittance safeguards. These require density tests and report
  diagnostics.
- **Anchor and sample filters**: neutral/dominant thresholds, luma bounds, border/dust rejection, and
  minimum support counts. These require synthetic cases for sparse, biased, clipped, dusty, and
  border-contaminated anchors.
- **Candidate quality weights and review thresholds**: gamut, exposure, calibration, reference,
  rendered-tone, model-quality, memory-colour, spatial-consistency, and fallback penalties. These
  require candidate-rank tests or fixture-suite expectation updates.
- **Tone cleanup thresholds**: highlight/midtone/shadow chroma cleanup and grain/detail controls.
  These require tone tests plus validation summary comparison on representative frames.

Any new constant should expose either a report field, a validation summary field, or a test assertion
that makes its effect inspectable. If it cannot be observed, it should not silently affect default
colour decisions.

## Validation Rule

Synthetic tests can pin decision edges, but they cannot close the broad colour evidence gap. The
real completion gate remains a validation-ready, multi-scene, multi-film-stock CoolScan corpus with
calibrated and uncalibrated cases, pinned component and calibration hashes, compact baselines, and
strict `--fixture-coverage` plus `--fixture-suite` passes.

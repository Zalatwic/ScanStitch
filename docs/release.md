# Release Workflow

## Local Windows Build

```powershell
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets -- --test-threads=1
cargo run --locked --release --bin scanstitch-validate -- --synthetic-color-suite --strict --quiet
cargo build --release
target/release/scanstitch.exe --help
target/release/scanstitch-validate.exe --help
target/release/scanstitch-bench.exe --help
```

Optional OpenCV build:

```powershell
cargo build --release --features use-opencv
```

This requires `llvm-config` or `LIBCLANG_PATH` plus OpenCV development libraries.
The pinned Ubuntu 24.04 CI job installs `clang libopencv-dev libclang-dev llvm-dev pkg-config`, exports
`LIBCLANG_PATH=$(llvm-config --libdir)`, and runs the library suite with `--features use-opencv`;
it is mandatory rather than conditional on preinstalled runner state. The suite executes native
SIFT and RGB16 projective-warp smoke paths in addition to the pure-Rust geometry regressions.

## Install From The Working Tree

```powershell
cargo install --path . --locked
```

This installs the default binaries from the current checkout into Cargo's bin directory.

## Pre-Release Checklist

1. Update `CHANGELOG.md`.
2. Run `cargo fmt --check`.
3. Run Clippy with warnings denied and the complete all-target test gate.
4. Run the strict synthetic colour suite.
5. Run fixture coverage in strict mode. A release is not colour-validation-ready unless pinned
   component/profile/baseline hashes, reference evidence, scene/exposure diversity, and declared
   seam/colour expectations all pass. Any applied seam photometric correction must report disjoint
   held-out validation; `gain_offset_rgb` fixtures must also pin improvement over gain-only and a
   normalized-offset ceiling. For every orientation-accuracy fixture, regenerate
   `--write-orientation-review` previews when the decoded digest changes and require a reviewer to
   approve them before setting `upright_approved: true`; tooling-generated drafts remain false.
   For every release-quality visual fixture, generate `--write-render-review` from its perfect-mode
   report, inspect the exact hash-bound source/report/master/proof artifacts, complete all applicable
   decisions, pin `render_review_sha256`, and require
   `min_approved_render_review_fixtures`. A completed human manifest must not override a failed
   technical delivery gate. For every grain-on review, pass
   `--render-review-grain-control-report` for an otherwise matched perfect-mode grain-off render;
   require identical decoded inputs, normalized non-grain render arguments, exact executable
   SHA-256/size identity, and byte-identical scene-referred masters before comparing the two bound
   proofs at 100%.
6. Run the fresh LOGAN comparison from `docs/validation.md`; require accepted full-union geometry,
   `seam_aware_multiband`, and the fixture's seam residual ceilings.
7. Run the opt-in real LOGAN stitch test if local TIFFs are available:

```powershell
cargo test --locked --release --test test_stitch test_real_sample_pair_uses_rgba8_load_path_and_accepts_narrow_overlap_stitch -- --ignored --test-threads=1
cargo test --locked --release --test test_base_detect test_real_testroll_curved_rebate_diagnostics -- --ignored --test-threads=1
```

   The LOGAN test pins the unequal crop-origin prior, recovered vertical offset, overlap, seam
   support, and identity photometric model (no unsupported additive correction). The OLD_TESTROLL
   test pins its no-op opposing-border deskew and keeps the post-crop
   fallback base at zero evidence confidence, separately from the strong pre-crop rebate retained
   by the full pipeline.
8. Render representative positive and negative real fixtures in release mode. A negative render
   with fallback film-base confidence or `render_reviewable=false` is evidence of a blocked case,
   not a release-quality image. Add `--require-reviewable` to production render commands so these
   cases retain their diagnostics but fail the process; the gate also rejects missing or disagreeing
   `render_review_status` / `render_reviewable` evidence. For current schema-v4 reports, require
   independently recomputed exact-byte SHA-256 matches for the primary TIFF and every requested
   float master/sRGB proof in addition to dimensions, storage, ICC, gamut-map, freshness, and
   stale-artifact checks. Treat this as unchanged-report artifact integrity, not a digital
   signature: if adversarial chain-of-custody matters, sign the release/report/review manifest or
   anchor it in an externally trusted record. Confirm sufficient free space for same-directory
   staging: each artifact is replaced atomically per file, so an overwrite can temporarily retain
   the old destination while the complete new artifact is encoded. This is not a multi-artifact
   transaction or an explicit file/directory `fsync` guarantee.
9. Build release binaries.
10. Record the package version, report schema version, corpus digest, and known unsupported cases
    in release notes.

## GitHub Artifacts

The CI workflow currently builds and tests the project. A future release workflow can upload
Windows release artifacts by adding an `actions/upload-artifact` step for:

- `target/release/scanstitch.exe`
- `target/release/scanstitch-validate.exe`
- `target/release/scanstitch-bench.exe`

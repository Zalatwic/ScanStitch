# Release Workflow

## Local Windows Build

```powershell
cargo fmt --check
cargo test
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

## Install From The Working Tree

```powershell
cargo install --path . --locked
```

This installs the default binaries from the current checkout into Cargo's bin directory.

## Pre-Release Checklist

1. Update `CHANGELOG.md`.
2. Run `cargo fmt --check`.
3. Run `cargo test`.
4. Run the fresh LOGAN comparison from `docs/validation.md`.
5. Run the opt-in real LOGAN stitch test if local TIFFs are available:

```powershell
cargo test --test test_stitch test_real_sample_pair_uses_rgba8_load_path_and_accepts_narrow_overlap_stitch -- --ignored
```

6. Build release binaries.
7. Record the package version and report schema version in release notes.

## GitHub Artifacts

The CI workflow currently builds and tests the project. A future release workflow can upload
Windows release artifacts by adding an `actions/upload-artifact` step for:

- `target/release/scanstitch.exe`
- `target/release/scanstitch-validate.exe`
- `target/release/scanstitch-bench.exe`

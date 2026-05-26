# Changelog

## Unreleased

- Added run metadata and output freshness diagnostics to reports.
- Added `scanstitch-ui`, an interactive TUI plus native preview window for tone/exposure edits before explicit save.
- Added `--input-mode positive` for slide scans and already-inverted RGB, with explicit skipped negative-film phases and positive-input diagnostics.
- Added calibration-library support for scanner profiles, roll profiles, film-stock hints, advisory DNG scanner priors, and the `scanstitch-calibrate` record-generation utility.
- Added validation summary render comparison, strict comparison mode, fixture coverage/suite gates, tracked compact baseline schemas, calibration evidence checks, and grain diagnostics.
- Added a lightweight benchmark binary, `scanstitch-bench`.
- Added report schema examples, a JSON Schema draft, fixture registry docs, and release workflow docs.
- Fixed ProPhoto D50 matrix consistency and clamped density transmittance normalization to the physical `[0, 1]` range.
- Replaced recoverable pipeline state panics with reportable phase failures.
- Made the slow local LOGAN stitch test opt-in so default `cargo test` remains fast and reliable.

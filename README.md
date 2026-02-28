# ScanStitch

Reconstructs 35mm color negative film frames from split RAW TIFF scans (Nikon CoolScan 4000). Removes scanner borders, optionally stitches split frames, strips the orange mask in density domain, separates dye layers via FastICA, maps to ProPhoto RGB D50, and applies film-like tone mapping. Outputs 16-bit positive TIFF.

## Pipeline

| Phase | Module | What it does |
|-|-|-|
| 1 | `border.rs` | Detects/removes scanner dead zones (top/bottom) using per-row median + MAD statistics |
| 2a | `base_detect.rs` | Identifies film base (rebate) on left/right edges via block-based chroma clustering |
| 2b | `frame_classify.rs` | Classifies frame as Intact / SingleBorder / CandidateSplit |
| 2c | `stitch.rs` | NCC overlap matching, score-based order detection, feathered blending |
| 3 | `density.rs` | T -> density, subtract base density, invert via D_max - D' |
| 4 | `ica.rs` | Full-image streaming FastICA (tanh contrast), permutation + sign resolution |
| 4.6 | `colorspace.rs` | Work RGB -> XYZ -> Bradford CAT -> ProPhoto RGB (D50) |
| 5 | `tonemap.rs` | Naka-Rushton sigmoid with histogram-fitted midpoint/slope |

## Build

```bash
# Default (pure Rust, no external deps beyond cargo)
cargo build --release

# With OpenCV for robust feature matching (requires OpenCV 4 installed)
cargo build --release --features use-opencv
```

## Usage

```
scanstitch <COMPONENT1> <COMPONENT2> [OPTIONS]
```

| Flag | Default | Description |
|-|-|-|
| `-o, --output-dir` | `output` | Where to write results |
| `--debug` | off | Save intermediate images + tone curve LUT |
| `--force-stitch` | off | Force stitching even if frames look intact |
| `--force-no-stitch` | off | Skip stitching entirely |
| `--bit-depth` | 14 | Input bit depth (14-bit padded to 16, or native 16) |
| `--ica-max-iter` | 100 | Max FastICA iterations |
| `--ica-tol` | 1e-5 | FastICA convergence threshold |
| `--transform` | auto | Transform model: auto / affine / homography |
| `--use-opencv` | off | Use OpenCV backend (needs `use-opencv` feature) |

Enable logging with `RUST_LOG=info` (or `debug` for verbose).

## Output

| File | Always | Description |
|-|-|-|
| `output.tiff` | yes | Final 16-bit positive TIFF (ProPhoto D50, tone-mapped) |
| `report.json` | yes | Per-phase confidence, metrics, warnings, errors |
| `comp1_cropped.tiff` | debug | After border removal |
| `comp2_cropped.tiff` | debug | After border removal |
| `stitched.tiff` | debug | After stitching (if stitching occurred) |
| `phase3_positive.tiff` | debug | After density inversion |
| `phase4_ica.tiff` | debug | After FastICA separation |
| `phase46_prophoto.tiff` | debug | After ProPhoto mapping |
| `tone_curve_lut.json` | debug | 256-point [input, output] tone curve |

## Architecture

```
src/
  main.rs          CLI entry point (36 lines)
  pipeline.rs      Phase orchestration (291 lines)
  border.rs        Phase 1: row-stats border detection (242 lines)
  base_detect.rs   Phase 2a: block chroma clustering (311 lines)
  frame_classify.rs Phase 2b: intact/split classification (45 lines)
  stitch.rs        Phase 2c: NCC matching + blend (287 lines)
  density.rs       Phase 3: density-domain inversion (129 lines)
  ica.rs           Phase 4: streaming FastICA (377 lines)
  colorspace.rs    Phase 4.6: ProPhoto D50 mapping (132 lines)
  tonemap.rs       Phase 5: Naka-Rushton tone mapping (141 lines)
  constants.rs     D50 white point, ProPhoto/Bradford matrices (39 lines)
  tiff_io.rs       16-bit TIFF load/save (85 lines)
  streaming.rs     Tiled iteration helpers (52 lines)
  report.rs        JSON report structs (70 lines)
  cli.rs           Clap 4 arg definitions (49 lines)
  debug.rs         Debug image helpers (45 lines)
  cv_adapter.rs    OpenCV feature-gated stub (33 lines)
```

~2500 lines of library code, ~550 lines of tests.

## Tests

```bash
cargo test              # all 44 tests
cargo test --test test_border    # just border detection
cargo test --test test_ica       # just FastICA
```

Test coverage: border detection (5), base detection + classification (6), stitching (5), density (5), ICA (5), colorspace (3), tonemap (4), streaming (4), integration (1). All use synthetic images from `tests/common/synthetic.rs`.

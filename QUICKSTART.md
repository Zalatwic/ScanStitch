# Quickstart

## Prerequisites

- Rust toolchain (`rustup` installed, stable channel)
- MSVC build tools (VS2022 on this machine; custom LIB/INCLUDE paths configured in `~/.cargo/config.toml` for onecore libs)

## Build

```bash
cargo build --release
```

Binary lands at `target/release/scanstitch.exe`.

## Run

Typical invocation for CoolScan 4000 scans:

```bash
RUST_LOG=info ./target/release/scanstitch scan_001.tiff scan_002.tiff -o output/ --debug --bit-depth 14
```

For frames you know are split across both files:
```bash
RUST_LOG=info ./target/release/scanstitch left.tiff right.tiff -o output/ --debug --force-stitch
```

For frames you know are NOT split (just process the first one):
```bash
RUST_LOG=info ./target/release/scanstitch frame.tiff dummy.tiff -o output/ --force-no-stitch
```

## Interpret Output

Look at `output/`:

1. **`output.tiff`** -- Your final 16-bit positive. Open in RawTherapee, darktable, or Photoshop. ProPhoto RGB color space, D50 white point.

2. **`report.json`** -- Machine-readable pipeline log. Check:
   - `phases[].success` -- did each phase succeed?
   - `phases[].confidence` -- how confident is the result?
   - `phases[].warnings` -- anything suspicious?
   - The `fastica` phase: did it converge? How many iterations?
   - The `stitch` phase: what was the NCC score? Did it stitch or skip?

3. **Debug images** (if `--debug`):
   - `comp1_cropped.tiff` / `comp2_cropped.tiff` -- verify borders were removed correctly
   - `stitched.tiff` -- verify stitch seam quality
   - `phase3_positive.tiff` -- after orange mask removal (should look like a rough positive)
   - `phase4_ica.tiff` -- after dye separation (colors may look odd, that's normal)
   - `phase46_prophoto.tiff` -- after color space mapping (should look reasonable)
   - `tone_curve_lut.json` -- the fitted S-curve, plot if curious

## Troubleshooting

**Wrong colors / muddy result**
- Check `--bit-depth`. CoolScan 4000 RAW TIFFs are 14-bit padded to 16-bit containers. Using `--bit-depth 16` on 14-bit data will miscalculate the density conversion. Default is 14, which is correct for CoolScan 4000.

**Stitch failed / "NCC score below threshold"**
- Check `report.json` for the NCC scores. If both orderings score low, the frames may not overlap.
- Look at `comp1_cropped.tiff` and `comp2_cropped.tiff` -- do they actually share content?
- Try `--force-no-stitch` and process the better component alone.

**ICA did not converge**
- Check `report.json` for the iteration count and convergence status.
- Try increasing `--ica-max-iter 200` or relaxing `--ica-tol 1e-4`.
- On very uniform images (blank sky, etc.), ICA may struggle because there aren't enough independent sources to separate. The output should still be usable.

**Border removal ate real content**
- Run with `--debug` and check `comp1_cropped.tiff`. If the image looks over-cropped, the scanner border was not cleanly separated from content. This is rare with CoolScan 4000 scans.

**Out of memory**
- CoolScan 4000 scans are ~130MB each. The pipeline needs ~2-3GB RAM for the full pipeline (density arrays, ICA passes). Close other applications or process on a machine with >= 8GB RAM.

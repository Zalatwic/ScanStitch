use approx::assert_relative_eq;
use ndarray::Array3;
use tempfile::TempDir;

fn write_rgb8_tiff(
    path: &std::path::Path,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::create(path)?;
    let mut encoder = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file))?;
    encoder.write_image::<tiff::encoder::colortype::RGB8>(width, height, pixels)?;
    Ok(())
}

#[test]
fn test_load_tiff_u16_expands_8bit_samples_into_requested_14bit_domain() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("rgb8.tiff");
    let pixels: Vec<u8> = vec![
        0, 128, 255, //
        64, 32, 16, //
        255, 200, 100, //
        5, 10, 15,
    ];
    write_rgb8_tiff(&path, 2, 2, &pixels).unwrap();

    let loaded = scanstitch::tiff_io::load_tiff_u16(&path, 14).unwrap();

    assert_eq!(loaded.diagnostics.source_bits_per_sample, 8);
    assert_eq!(
        loaded.diagnostics.working_range.transform,
        scanstitch::tiff_io::WorkingRangeTransform::UpscaledToWorkingBitDepth
    );
    assert_eq!(loaded.image[[0, 0, 0]], 0);
    assert_eq!(loaded.image[[0, 0, 2]], 16383);
    assert_relative_eq!(loaded.image[[0, 0, 1]] as f64, 8224.0, epsilon = 1.0);
    assert_relative_eq!(loaded.image[[1, 0, 1]] as f64, 12850.0, epsilon = 1.0);
}

#[test]
fn test_normalize_to_working_bit_depth_downscales_16bit_container_data_for_14bit_pipeline() {
    let mut promoted = Array3::<u16>::zeros((1, 2, 3));
    promoted[[0, 0, 0]] = 4000 * 4;
    promoted[[0, 0, 1]] = 8000 * 4;
    promoted[[0, 0, 2]] = 12000 * 4;
    promoted[[0, 1, 0]] = 4095 * 4;
    promoted[[0, 1, 1]] = 16383;
    promoted[[0, 1, 2]] = 16383 * 4;

    let (normalized, diagnostics) =
        scanstitch::tiff_io::normalize_to_working_bit_depth(&promoted, 16, 14);

    assert_eq!(
        diagnostics.transform,
        scanstitch::tiff_io::WorkingRangeTransform::DownscaledToWorkingBitDepth
    );
    assert!(diagnostics.working_max.iter().all(|&v| v <= 16383));
    assert_relative_eq!(normalized[[0, 0, 0]] as f64, 4000.0, epsilon = 1.0);
    assert_relative_eq!(normalized[[0, 0, 1]] as f64, 8000.0, epsilon = 1.0);
    assert_relative_eq!(normalized[[0, 0, 2]] as f64, 12000.0, epsilon = 1.0);
    assert_relative_eq!(normalized[[0, 1, 2]] as f64, 16383.0, epsilon = 1.0);
}

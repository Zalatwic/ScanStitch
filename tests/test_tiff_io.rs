use approx::assert_relative_eq;
use ndarray::Array3;
use std::io::Write;
use tempfile::TempDir;
use tiff::tags::Tag;

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

struct IfdEntrySpec {
    tag: u16,
    value_type: u16,
    count: u32,
    value: Vec<u8>,
}

fn short_entry(tag: u16, value: u16) -> IfdEntrySpec {
    IfdEntrySpec {
        tag,
        value_type: 3,
        count: 1,
        value: value.to_le_bytes().to_vec(),
    }
}

fn short_vec_entry(tag: u16, values: &[u16]) -> IfdEntrySpec {
    IfdEntrySpec {
        tag,
        value_type: 3,
        count: values.len() as u32,
        value: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    }
}

fn long_entry(tag: u16, value: u32) -> IfdEntrySpec {
    IfdEntrySpec {
        tag,
        value_type: 4,
        count: 1,
        value: value.to_le_bytes().to_vec(),
    }
}

fn ascii_entry(tag: u16, value: &str) -> IfdEntrySpec {
    let mut bytes = value.as_bytes().to_vec();
    bytes.push(0);
    IfdEntrySpec {
        tag,
        value_type: 2,
        count: bytes.len() as u32,
        value: bytes,
    }
}

fn rational_vec_entry(tag: u16, values: &[(u32, u32)]) -> IfdEntrySpec {
    IfdEntrySpec {
        tag,
        value_type: 5,
        count: values.len() as u32,
        value: values
            .iter()
            .flat_map(|(numerator, denominator)| {
                numerator
                    .to_le_bytes()
                    .into_iter()
                    .chain(denominator.to_le_bytes())
            })
            .collect(),
    }
}

fn srational_vec_entry(tag: u16, values: &[(i32, i32)]) -> IfdEntrySpec {
    IfdEntrySpec {
        tag,
        value_type: 10,
        count: values.len() as u32,
        value: values
            .iter()
            .flat_map(|(numerator, denominator)| {
                numerator
                    .to_le_bytes()
                    .into_iter()
                    .chain(denominator.to_le_bytes())
            })
            .collect(),
    }
}

fn append_ifd(data: &mut Vec<u8>, entries: &[IfdEntrySpec]) -> u32 {
    let ifd_offset = data.len() as u32;
    data.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    let entries_start = data.len();
    data.resize(entries_start + entries.len() * 12, 0);
    data.extend_from_slice(&0u32.to_le_bytes());

    for (index, entry) in entries.iter().enumerate() {
        let entry_offset = entries_start + index * 12;
        data[entry_offset..entry_offset + 2].copy_from_slice(&entry.tag.to_le_bytes());
        data[entry_offset + 2..entry_offset + 4].copy_from_slice(&entry.value_type.to_le_bytes());
        data[entry_offset + 4..entry_offset + 8].copy_from_slice(&entry.count.to_le_bytes());
        let mut value_or_offset = [0u8; 4];
        if entry.value.len() <= 4 {
            value_or_offset[..entry.value.len()].copy_from_slice(&entry.value);
        } else {
            let value_offset = data.len() as u32;
            data.extend_from_slice(&entry.value);
            value_or_offset.copy_from_slice(&value_offset.to_le_bytes());
        }
        data[entry_offset + 8..entry_offset + 12].copy_from_slice(&value_or_offset);
    }

    ifd_offset
}

fn write_minimal_dng_with_linear_raw_subifd(
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut data = Vec::new();
    data.extend_from_slice(b"II");
    data.extend_from_slice(&42u16.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    let raw_offset = data.len() as u32;
    let raw_samples = [
        100u16, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
    ];
    for sample in raw_samples {
        data.extend_from_slice(&sample.to_le_bytes());
    }

    let thumbnail_offset = data.len() as u32;
    data.extend_from_slice(&[250, 0, 0]);
    while data.len() % 4 != 0 {
        data.push(0);
    }

    let sub_ifd_offset = append_ifd(
        &mut data,
        &[
            long_entry(254, 0),
            short_entry(256, 2),
            short_entry(257, 2),
            short_vec_entry(258, &[16, 16, 16]),
            short_entry(259, 1),
            short_entry(262, 34_892),
            long_entry(273, raw_offset),
            short_entry(277, 3),
            short_entry(278, 2),
            long_entry(279, 24),
            short_entry(284, 1),
        ],
    );
    while data.len() % 4 != 0 {
        data.push(0);
    }

    let primary_ifd_offset = append_ifd(
        &mut data,
        &[
            long_entry(254, 1),
            short_entry(256, 1),
            short_entry(257, 1),
            short_vec_entry(258, &[8, 8, 8]),
            short_entry(259, 1),
            short_entry(262, 2),
            ascii_entry(271, "Nikon"),
            ascii_entry(272, "LS-4000"),
            long_entry(273, thumbnail_offset),
            short_entry(277, 3),
            short_entry(278, 1),
            long_entry(279, 3),
            long_entry(330, sub_ifd_offset),
            ascii_entry(305, "VueScan 9 x64"),
            ascii_entry(50_708, "Nikon LS-4000"),
            short_vec_entry(50_714, &[0, 0, 0]),
            long_entry(50_717, 65_535),
            srational_vec_entry(
                50_721,
                &[
                    (3, 1),
                    (-3, 2),
                    (-2, 5),
                    (-7, 8),
                    (17, 10),
                    (1, 10),
                    (-1, 40),
                    (1, 50),
                    (9, 10),
                ],
            ),
            rational_vec_entry(50_728, &[(1, 2), (1, 1), (3, 2)]),
            short_entry(50_778, 21),
        ],
    );
    data[4..8].copy_from_slice(&primary_ifd_offset.to_le_bytes());

    let mut file = std::fs::File::create(path)?;
    file.write_all(&data)?;
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
fn test_load_dng_uses_full_linear_raw_subifd_instead_of_thumbnail() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("linear-raw.dng");
    write_minimal_dng_with_linear_raw_subifd(&path).unwrap();

    let inspection = scanstitch::tiff_io::inspect_scan_source(&path).unwrap();
    assert_eq!(inspection.width, 2);
    assert_eq!(inspection.height, 2);
    assert_eq!(inspection.color_type, "DNG_LINEAR_RAW16");
    assert_eq!(inspection.source_bits_per_sample, 16);

    let loaded = scanstitch::tiff_io::load_tiff_u16(&path, 14).unwrap();

    assert_eq!(loaded.image.dim(), (2, 2, 3));
    assert_eq!(loaded.diagnostics.width, 2);
    assert_eq!(loaded.diagnostics.height, 2);
    assert_eq!(loaded.diagnostics.color_type, "DNG_LINEAR_RAW16");
    assert_eq!(loaded.image[[0, 0, 0]], 100);
    assert_eq!(loaded.image[[1, 1, 2]], 1200);
    assert_eq!(
        loaded.diagnostics.working_range.transform,
        scanstitch::tiff_io::WorkingRangeTransform::None
    );
    let metadata = loaded
        .diagnostics
        .dng_metadata
        .as_ref()
        .expect("DNG metadata should be reported from the primary IFD");
    assert_eq!(metadata.make.as_deref(), Some("Nikon"));
    assert_eq!(metadata.model.as_deref(), Some("LS-4000"));
    assert_eq!(metadata.software.as_deref(), Some("VueScan 9 x64"));
    assert_eq!(
        metadata.unique_camera_model.as_deref(),
        Some("Nikon LS-4000")
    );
    assert_eq!(metadata.calibration_illuminant1, Some(21));
    let matrix = metadata.color_matrix1.expect("ColorMatrix1 should decode");
    assert_relative_eq!(matrix[0][0], 3.0, epsilon = 1e-12);
    assert_relative_eq!(matrix[0][1], -1.5, epsilon = 1e-12);
    assert_relative_eq!(matrix[2][2], 0.9, epsilon = 1e-12);
    assert_eq!(
        metadata.as_shot_neutral.as_deref(),
        Some(&[0.5, 1.0, 1.5][..])
    );
    assert_eq!(metadata.black_level.as_deref(), Some(&[0.0, 0.0, 0.0][..]));
    assert_eq!(metadata.white_level.as_deref(), Some(&[65_535.0][..]));

    let calibration = scanstitch::color_calibration::dng_color_matrix1_advisory_prior(metadata)
        .expect("ColorMatrix1 should build an advisory scanner prior");
    assert_eq!(calibration.diagnostics.status, "applied");
    assert_eq!(
        calibration.diagnostics.source,
        "dng_color_matrix1_advisory_prior"
    );
    let profile = calibration.profile.expect("advisory profile");
    assert_eq!(
        profile.application_mode,
        scanstitch::color_calibration::CalibrationApplicationMode::ScannerConstrainedImageAdaptation
    );
    assert_eq!(profile.work_to_xyz, matrix);
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

#[test]
fn test_save_tiff_f64_linear_prophoto_embeds_icc_profile() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("prophoto.tiff");
    let mut img = Array3::<f64>::zeros((2, 2, 3));
    img[[0, 0, 0]] = 0.25;
    img[[0, 0, 1]] = 0.50;
    img[[0, 0, 2]] = 0.75;

    scanstitch::tiff_io::save_tiff_f64_linear_prophoto(&img, &path).unwrap();

    let inspection = scanstitch::tiff_io::inspect_tiff_icc_profile(&path).unwrap();
    assert_eq!(
        inspection.image_description.as_deref(),
        Some(scanstitch::tiff_io::PROPHOTO_LINEAR_ICC_DESCRIPTION)
    );
    assert!(inspection.icc_profile_embedded);
    assert!(inspection.icc_profile_valid);
    assert_eq!(
        inspection.icc_profile_description.as_deref(),
        Some(scanstitch::tiff_io::PROPHOTO_LINEAR_ICC_DESCRIPTION)
    );

    let file = std::fs::File::open(&path).unwrap();
    let mut decoder = tiff::decoder::Decoder::new(std::io::BufReader::new(file)).unwrap();
    let description = decoder.get_tag_ascii_string(Tag::ImageDescription).unwrap();
    assert_eq!(
        description,
        scanstitch::tiff_io::PROPHOTO_LINEAR_ICC_DESCRIPTION
    );
    let icc = decoder
        .get_tag(Tag::Unknown(34_675))
        .unwrap()
        .into_u32_vec()
        .unwrap()
        .into_iter()
        .map(|value| u8::try_from(value).unwrap())
        .collect::<Vec<_>>();
    assert!(
        icc.len() > 128,
        "ICC profile should include header and tags"
    );
    assert_eq!(&icc[36..40], b"acsp");
    assert!(icc
        .windows(scanstitch::tiff_io::PROPHOTO_LINEAR_ICC_DESCRIPTION.len())
        .any(|window| window == scanstitch::tiff_io::PROPHOTO_LINEAR_ICC_DESCRIPTION.as_bytes()));
}

#[test]
fn test_save_scene_referred_prophoto_float_preserves_headroom() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("scene_referred_prophoto_float.tiff");
    let mut img = Array3::<f64>::zeros((1, 2, 3));
    img[[0, 0, 0]] = -0.125;
    img[[0, 0, 1]] = 0.50;
    img[[0, 0, 2]] = 1.75;
    img[[0, 1, 0]] = 2.50;
    img[[0, 1, 1]] = f64::NAN;
    img[[0, 1, 2]] = 0.25;

    scanstitch::tiff_io::save_tiff_f32_linear_prophoto_scene_referred(&img, &path).unwrap();

    let inspection = scanstitch::tiff_io::inspect_tiff_icc_profile(&path).unwrap();
    assert_eq!(
        inspection.image_description.as_deref(),
        Some(scanstitch::tiff_io::PROPHOTO_SCENE_REFERRED_FLOAT_DESCRIPTION)
    );
    assert!(inspection.icc_profile_embedded);
    assert!(inspection.icc_profile_valid);
    assert_eq!(
        inspection.icc_profile_description.as_deref(),
        Some(scanstitch::tiff_io::PROPHOTO_LINEAR_ICC_DESCRIPTION)
    );

    let file = std::fs::File::open(&path).unwrap();
    let mut decoder = tiff::decoder::Decoder::new(std::io::BufReader::new(file)).unwrap();
    assert_eq!(decoder.dimensions().unwrap(), (2, 1));
    assert_eq!(decoder.colortype().unwrap(), tiff::ColorType::RGB(32));
    let decoded = match decoder.read_image().unwrap() {
        tiff::decoder::DecodingResult::F32(values) => values,
        other => panic!("expected f32 TIFF data, got {other:?}"),
    };
    assert_relative_eq!(decoded[0], -0.125, epsilon = 1e-6);
    assert_relative_eq!(decoded[1], 0.50, epsilon = 1e-6);
    assert_relative_eq!(decoded[2], 1.75, epsilon = 1e-6);
    assert_relative_eq!(decoded[3], 2.50, epsilon = 1e-6);
    assert_relative_eq!(decoded[4], 0.0, epsilon = 1e-6);
    assert_relative_eq!(decoded[5], 0.25, epsilon = 1e-6);
}

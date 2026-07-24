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

fn write_rgb16_tiff_with_metadata(
    path: &std::path::Path,
    width: u32,
    height: u32,
    pixels: &[u16],
    orientation: Option<u16>,
    icc_profile: Option<&[u8]>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::create(path)?;
    let mut encoder = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file))?;
    let mut image = encoder.new_image::<tiff::encoder::colortype::RGB16>(width, height)?;
    if let Some(orientation) = orientation {
        image.encoder().write_tag(Tag::Orientation, orientation)?;
    }
    if let Some(icc_profile) = icc_profile {
        image
            .encoder()
            .write_tag(Tag::Unknown(34_675), icc_profile)?;
    }
    image.write_data(pixels)?;
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

fn long_vec_entry(tag: u16, values: &[u32]) -> IfdEntrySpec {
    IfdEntrySpec {
        tag,
        value_type: 4,
        count: values.len() as u32,
        value: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
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
    write_linear_raw_dng(
        path,
        &[
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
        ],
        &[0, 0, 0],
        &[65_535],
        None,
    )
}

fn write_linear_raw_dng(
    path: &std::path::Path,
    raw_samples: &[u16; 12],
    black_levels: &[u16],
    white_levels: &[u32],
    orientation: Option<u16>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut data = Vec::new();
    data.extend_from_slice(b"II");
    data.extend_from_slice(&42u16.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    let raw_offset = data.len() as u32;
    for sample in raw_samples {
        data.extend_from_slice(&sample.to_le_bytes());
    }

    let thumbnail_offset = data.len() as u32;
    data.extend_from_slice(&[250, 0, 0]);
    while data.len() % 4 != 0 {
        data.push(0);
    }

    let mut sub_ifd_entries = vec![
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
    ];
    if let Some(orientation) = orientation {
        sub_ifd_entries.push(short_entry(274, orientation));
        sub_ifd_entries.sort_by_key(|entry| entry.tag);
    }
    let sub_ifd_offset = append_ifd(&mut data, &sub_ifd_entries);
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
            short_vec_entry(50_714, black_levels),
            long_vec_entry(50_717, white_levels),
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
fn test_load_tiff_applies_all_exif_orientations_before_processing() {
    let tmp = TempDir::new().unwrap();
    let source_values = [1u16, 2, 3, 4, 5, 6];
    let pixels = source_values
        .iter()
        .flat_map(|value| [*value; 3])
        .collect::<Vec<_>>();
    let expected = [
        (1u16, 3usize, 2usize, vec![1, 2, 3, 4, 5, 6], false),
        (2, 3, 2, vec![3, 2, 1, 6, 5, 4], true),
        (3, 3, 2, vec![6, 5, 4, 3, 2, 1], true),
        (4, 3, 2, vec![4, 5, 6, 1, 2, 3], true),
        (5, 2, 3, vec![1, 4, 2, 5, 3, 6], true),
        (6, 2, 3, vec![4, 1, 5, 2, 6, 3], true),
        (7, 2, 3, vec![6, 3, 5, 2, 4, 1], true),
        (8, 2, 3, vec![3, 6, 2, 5, 1, 4], true),
    ];
    let mut decoded_hashes = Vec::new();

    for (orientation, expected_width, expected_height, expected_values, applied) in expected {
        let path = tmp.path().join(format!("orientation-{orientation}.tiff"));
        write_rgb16_tiff_with_metadata(&path, 3, 2, &pixels, Some(orientation), None).unwrap();

        let loaded = scanstitch::tiff_io::load_tiff_u16(&path, 16).unwrap();

        assert_eq!(loaded.image.dim(), (expected_height, expected_width, 3));
        assert_eq!(loaded.diagnostics.orientation.tag_value, Some(orientation));
        assert_eq!(loaded.diagnostics.orientation.applied, applied);
        assert_eq!(loaded.diagnostics.decoded_pixel_sha256.len(), 64);
        assert!(loaded
            .diagnostics
            .decoded_pixel_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit()));
        let reloaded = scanstitch::tiff_io::load_tiff_u16(&path, 16).unwrap();
        assert_eq!(
            reloaded.diagnostics.decoded_pixel_sha256,
            loaded.diagnostics.decoded_pixel_sha256
        );
        decoded_hashes.push(loaded.diagnostics.decoded_pixel_sha256.clone());
        let actual = loaded
            .image
            .outer_iter()
            .flat_map(|row| row.outer_iter().map(|pixel| pixel[0]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected_values, "EXIF orientation {orientation}");
    }
    assert_eq!(
        decoded_hashes
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        8,
        "asymmetric decoded pixels must bind each materialized orientation to a distinct digest"
    );
    let orientation_one_14_bit =
        scanstitch::tiff_io::load_tiff_u16(&tmp.path().join("orientation-1.tiff"), 14).unwrap();
    assert_ne!(
        orientation_one_14_bit.diagnostics.decoded_pixel_sha256, decoded_hashes[0],
        "the approval digest must bind the working precision"
    );
}

#[test]
fn metadata_relative_orientation_correction_composes_transform_and_rebinds_pixels() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("orientation-correction.tiff");
    let pixels = (1u16..=6).flat_map(|value| [value; 3]).collect::<Vec<_>>();
    write_rgb16_tiff_with_metadata(&path, 3, 2, &pixels, Some(6), None).unwrap();
    let mut loaded = scanstitch::tiff_io::load_tiff_u16(&path, 16).unwrap();
    let metadata_hash = loaded.diagnostics.decoded_pixel_sha256.clone();

    scanstitch::tiff_io::apply_orientation_correction_u16(&mut loaded, 3, "rotate-180", 16)
        .unwrap();

    assert_eq!(loaded.diagnostics.orientation.tag_value, Some(6));
    assert_eq!(
        loaded.diagnostics.orientation.transform,
        "rotate_90_clockwise"
    );
    assert_eq!(loaded.diagnostics.effective_orientation_tag, Some(8));
    assert_eq!(
        loaded
            .diagnostics
            .orientation_correction
            .effective_transform,
        "rotate_270_clockwise"
    );
    assert!(loaded.diagnostics.orientation_correction.applied);
    assert_eq!(
        loaded
            .diagnostics
            .orientation_correction
            .source_orientation_materialized_decoded_pixel_sha256,
        metadata_hash
    );
    assert_ne!(loaded.diagnostics.decoded_pixel_sha256, metadata_hash);
    assert_eq!(
        loaded
            .diagnostics
            .orientation_correction
            .corrected_decoded_pixel_sha256,
        loaded.diagnostics.decoded_pixel_sha256
    );
    assert_eq!(loaded.image.dim(), (3, 2, 3));
    let actual = loaded
        .image
        .outer_iter()
        .flat_map(|row| row.outer_iter().map(|pixel| pixel[0]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_eq!(actual, vec![3, 6, 2, 5, 1, 4]);

    assert_eq!(
        scanstitch::tiff_io::compose_orientation_tags(Some(6), 6),
        Some(3),
        "two clockwise quarter turns must compose to a half turn"
    );
}

#[test]
fn every_metadata_and_correction_pair_matches_its_composed_exif_transform() {
    let tmp = TempDir::new().unwrap();
    let pixels = (1u16..=6).flat_map(|value| [value; 3]).collect::<Vec<_>>();
    for tag in 1u16..=8 {
        write_rgb16_tiff_with_metadata(
            &tmp.path().join(format!("orientation-{tag}.tiff")),
            3,
            2,
            &pixels,
            Some(tag),
            None,
        )
        .unwrap();
    }

    for source_tag in 1u16..=8 {
        for correction_tag in 1u16..=8 {
            let mut actual = scanstitch::tiff_io::load_tiff_u16(
                &tmp.path().join(format!("orientation-{source_tag}.tiff")),
                16,
            )
            .unwrap();
            scanstitch::tiff_io::apply_orientation_correction_u16(
                &mut actual,
                correction_tag,
                scanstitch::tiff_io::orientation_transform_name(Some(correction_tag)),
                16,
            )
            .unwrap();
            let effective_tag =
                scanstitch::tiff_io::compose_orientation_tags(Some(source_tag), correction_tag)
                    .expect("D4 composition must remain an EXIF orientation");
            let expected = scanstitch::tiff_io::load_tiff_u16(
                &tmp.path().join(format!("orientation-{effective_tag}.tiff")),
                16,
            )
            .unwrap();

            assert_eq!(
                actual.image, expected.image,
                "source tag {source_tag} followed by correction tag {correction_tag} must equal effective tag {effective_tag}"
            );
            assert_eq!(
                actual.diagnostics.effective_orientation_tag,
                Some(effective_tag)
            );
            assert_eq!(
                actual.diagnostics.decoded_pixel_sha256,
                expected.diagnostics.decoded_pixel_sha256
            );
        }
    }
}

#[test]
fn test_embedded_srgb_icc_is_extracted_and_transformed_to_linear_prophoto() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("embedded-srgb.tiff");
    let profile = moxcms::ColorProfile::new_srgb().encode().unwrap();
    let pixels = vec![32_768u16; 2 * 2 * 3];
    write_rgb16_tiff_with_metadata(&path, 2, 2, &pixels, None, Some(&profile)).unwrap();

    let loaded = scanstitch::tiff_io::load_tiff_u16(&path, 16).unwrap();
    let diagnostics = loaded
        .diagnostics
        .source_icc_profile
        .as_ref()
        .expect("ICC diagnostics");
    assert_eq!(diagnostics.status, "valid_rgb_profile");
    assert_eq!(diagnostics.size_bytes, profile.len());
    let embedded = loaded.embedded_icc_profile.as_ref().expect("ICC payload");
    assert_eq!(embedded.bytes(), profile);

    let encoded = loaded.image.mapv(|sample| sample as f64 / 65_535.0);
    let transformed =
        scanstitch::input_color::transform_to_linear_prophoto(encoded, embedded).unwrap();

    assert_eq!(transformed.diagnostics.status, "applied");
    assert!(transformed.diagnostics.transfer_functions_applied);
    for channel in 0..3 {
        assert_relative_eq!(transformed.image[[0, 0, channel]], 0.214_05, epsilon = 7e-4);
    }
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
    assert_ne!(
        profile.work_to_xyz, matrix,
        "DNG ColorMatrix1 is stored as XYZ-to-camera and must not be used directly as camera-to-XYZ"
    );
    let neutral = metadata.as_shot_neutral.as_deref().unwrap();
    let mapped_white = std::array::from_fn::<_, 3, _>(|row| {
        (0..3)
            .map(|column| profile.work_to_xyz[row][column] * neutral[column])
            .sum::<f64>()
    });
    assert_relative_eq!(mapped_white[0] / mapped_white[1], 0.9642, epsilon = 2e-4);
    assert_relative_eq!(mapped_white[2] / mapped_white[1], 0.8251, epsilon = 2e-4);
    assert_eq!(
        profile.source_space["stored_matrix_direction"],
        "xyz_to_reference_camera"
    );
    assert_eq!(
        profile.source_space["derived_matrix_direction"],
        "camera_to_xyz_d50"
    );
}

#[test]
fn test_load_dng_applies_channel_black_white_levels_and_orientation() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("levels-and-orientation.dng");
    write_linear_raw_dng(
        &path,
        &[
            100, 200, 300, // source p0: black
            600, 1200, 1800, // source p1: half scale
            1100, 2200, 3300, // source p2: white
            50, 2300, 3400, // source p3: clipped low/high/high
        ],
        &[100, 200, 300],
        &[1100, 2200, 3300],
        Some(6),
    )
    .unwrap();

    let loaded = scanstitch::tiff_io::load_tiff_u16(&path, 16).unwrap();

    assert_eq!(loaded.image.dim(), (2, 2, 3));
    assert_eq!(loaded.diagnostics.orientation.tag_value, Some(6));
    assert_eq!(
        loaded.diagnostics.orientation.transform,
        "rotate_90_clockwise"
    );
    assert!(loaded.diagnostics.orientation.applied);
    let levels = loaded
        .diagnostics
        .dng_level_normalization
        .as_ref()
        .expect("DNG level diagnostics");
    assert_eq!(levels.status, "applied");
    assert_eq!(levels.black_level, [100.0, 200.0, 300.0]);
    assert_eq!(levels.white_level, [1100.0, 2200.0, 3300.0]);
    assert_eq!(levels.clipped_below_black, [1, 0, 0]);
    assert_eq!(levels.clipped_above_white, [0, 1, 1]);

    // A clockwise rotation maps source [p0 p1; p2 p3] to [p2 p0; p3 p1].
    assert_eq!(loaded.image[[0, 0, 0]], 65_535);
    assert_eq!(loaded.image[[0, 1, 0]], 0);
    assert_eq!(loaded.image[[1, 0, 0]], 0);
    for channel in 0..3 {
        assert_relative_eq!(
            loaded.image[[1, 1, channel]] as f64,
            32_768.0,
            epsilon = 1.0
        );
    }
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
    assert_eq!((inspection.width, inspection.height), (2, 2));
    assert_eq!(inspection.color_type, "RGB(16)");
    assert!(inspection
        .sample_format
        .as_deref()
        .is_some_and(|formats| formats.iter().all(|format| *format == 1)));
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
    assert_eq!((inspection.width, inspection.height), (2, 1));
    assert_eq!(inspection.color_type, "RGB(32)");
    assert!(inspection
        .sample_format
        .as_deref()
        .is_some_and(|formats| formats.iter().all(|format| *format == 3)));
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

#[test]
fn test_streamed_tiff_writers_preserve_pixels_across_strip_boundaries() {
    let tmp = TempDir::new().unwrap();
    let primary_path = tmp.path().join("streamed-primary.tiff");
    let master_path = tmp.path().join("streamed-master.tiff");
    let width = 1024usize;
    let height = 170usize;
    let image = Array3::<f64>::from_shape_fn((height, width, 3), |(y, x, channel)| {
        ((y * 131 + x * 17 + channel * 29) % 65_536) as f64 / 65_535.0
    });

    scanstitch::tiff_io::save_tiff_f64_linear_prophoto(&image, &primary_path).unwrap();
    scanstitch::tiff_io::save_tiff_f32_linear_prophoto_scene_referred(&image, &master_path)
        .unwrap();

    let primary_file = std::fs::File::open(&primary_path).unwrap();
    let mut primary = tiff::decoder::Decoder::new(std::io::BufReader::new(primary_file)).unwrap();
    let primary_rows_per_strip = primary.get_tag_u32(Tag::RowsPerStrip).unwrap() as usize;
    assert!(
        primary_rows_per_strip < height,
        "test must cross a TIFF strip"
    );
    let primary_pixels = match primary.read_image().unwrap() {
        tiff::decoder::DecodingResult::U16(values) => values,
        other => panic!("expected RGB16 TIFF data, got {other:?}"),
    };

    let master_file = std::fs::File::open(&master_path).unwrap();
    let mut master = tiff::decoder::Decoder::new(std::io::BufReader::new(master_file)).unwrap();
    let master_rows_per_strip = master.get_tag_u32(Tag::RowsPerStrip).unwrap() as usize;
    assert!(
        master_rows_per_strip < height,
        "test must cross a TIFF strip"
    );
    let master_pixels = match master.read_image().unwrap() {
        tiff::decoder::DecodingResult::F32(values) => values,
        other => panic!("expected RGB32-float TIFF data, got {other:?}"),
    };

    let rows_to_check = [
        0,
        primary_rows_per_strip - 1,
        primary_rows_per_strip,
        master_rows_per_strip - 1,
        master_rows_per_strip,
        height - 1,
    ];
    for y in rows_to_check {
        for &(x, channel) in &[(0, 0), (width / 2, 1), (width - 1, 2)] {
            let index = (y * width + x) * 3 + channel;
            let expected = image[[y, x, channel]];
            assert_eq!(primary_pixels[index], (expected * 65_535.0).round() as u16);
            assert_relative_eq!(master_pixels[index], expected as f32, epsilon = 1e-7);
        }
    }
}

#[test]
fn test_full_resolution_artifact_encoding_uses_bounded_conversion_buffers() {
    let diagnostics =
        scanstitch::tiff_io::artifact_write_buffer_diagnostics(11_701, 3_671, true, true);
    assert_eq!(
        diagnostics.strategy,
        "sequential_bounded_tiff_strips_and_png_rows"
    );
    assert!(!diagnostics.full_frame_conversion_buffers);
    assert_eq!(
        diagnostics.tiff_strip_target_bytes,
        scanstitch::tiff_io::TIFF_STREAM_TARGET_BYTES
    );
    assert!(diagnostics.primary_tiff_conversion_buffer_bytes < 1_200_000);
    assert!(diagnostics
        .master_tiff_conversion_buffer_bytes
        .is_some_and(|bytes| bytes < 1_200_000));
    assert_eq!(diagnostics.review_png_row_buffer_bytes, Some(11_701 * 3));
    assert_eq!(
        diagnostics.review_png_stream_chunk_bytes,
        Some(scanstitch::tiff_io::PNG_STREAM_CHUNK_BYTES)
    );
    assert!(diagnostics.peak_declared_buffer_bytes < 1_200_000);
    let legacy_master_conversion_bytes = 11_701usize * 3_671 * 3 * std::mem::size_of::<f32>();
    assert!(
        diagnostics.peak_declared_buffer_bytes * 100 < legacy_master_conversion_bytes,
        "bounded conversion should use less than 1% of the former full-frame master buffer"
    );
}

#[test]
fn test_artifact_writers_reject_empty_or_non_rgb_arrays_without_panicking() {
    let tmp = TempDir::new().unwrap();
    let non_rgb = Array3::<f64>::zeros((2, 2, 4));
    let non_rgb_error = scanstitch::tiff_io::save_tiff_f64_linear_prophoto(
        &non_rgb,
        &tmp.path().join("non-rgb.tiff"),
    )
    .unwrap_err()
    .to_string();
    assert!(non_rgb_error.contains("HxWx3 RGB array"), "{non_rgb_error}");

    let empty = Array3::<u16>::zeros((0, 2, 3));
    let empty_error = scanstitch::tiff_io::save_tiff_u16(&empty, &tmp.path().join("empty.tiff"))
        .unwrap_err()
        .to_string();
    assert!(empty_error.contains("empty RGB image"), "{empty_error}");
}

#[test]
fn test_save_srgb_review_png_embeds_explicit_standard_profile() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("review_srgb.png");
    let repeated_path = tmp.path().join("review_srgb_repeated.png");
    let mut img = Array3::<f64>::zeros((2, 3, 3));
    img[[0, 0, 0]] = 0.25;
    img[[0, 0, 1]] = 0.50;
    img[[0, 0, 2]] = 0.75;

    let diagnostics = scanstitch::tiff_io::save_srgb_png_from_linear_prophoto(&img, &path).unwrap();
    assert_eq!((diagnostics.width, diagnostics.height), (3, 2));
    assert_eq!(diagnostics.pixel_count, 6);
    assert_eq!(diagnostics.nonfinite_input_pixel_count, 0);
    assert_eq!(diagnostics.post_map_out_of_gamut_pixel_count, 0);
    assert!((0.0..=1.0).contains(&diagnostics.gamut_mapped_ratio));

    let inspection = scanstitch::tiff_io::inspect_srgb_png(&path).unwrap();
    assert_eq!((inspection.width, inspection.height), (3, 2));
    assert_eq!(inspection.color_type, "Rgb8");
    assert!(inspection.icc_profile_embedded);
    assert!(inspection.icc_profile_valid, "{:?}", inspection.reason);
    assert_eq!(
        inspection.icc_profile_description.as_deref(),
        Some(scanstitch::tiff_io::SRGB_ICC_DESCRIPTION)
    );
    assert!(
        inspection.icc_profile_matches_standard_srgb,
        "{:?}",
        inspection.reason
    );

    let saved = image::open(&path).unwrap().to_rgb8();
    let mapping = scanstitch::tiff_io::perceptual_srgb_from_linear_prophoto([0.25, 0.50, 0.75]);
    let encode = |value: f64| {
        let value = value.clamp(0.0, 1.0);
        let encoded = if value <= 0.003_130_8 {
            12.92 * value
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        };
        (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    assert_eq!(
        saved.get_pixel(0, 0).0,
        mapping.linear_srgb.map(encode),
        "row-streamed proof must preserve the exact display encoding"
    );
    assert_eq!(saved.get_pixel(2, 1).0, [0, 0, 0]);

    scanstitch::tiff_io::save_srgb_png_from_linear_prophoto(&img, &repeated_path).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(&repeated_path).unwrap(),
        "identical review renders must produce byte-identical PNG containers"
    );
}

#[test]
fn test_standard_srgb_profile_has_a_canonical_creation_time() {
    let first = scanstitch::tiff_io::srgb_icc_profile().unwrap();
    let second = scanstitch::tiff_io::srgb_icc_profile().unwrap();
    assert_eq!(first, second);
    assert_eq!(&first[36..40], b"acsp");
    for (encoded, expected) in first[24..36]
        .chunks_exact(2)
        .zip(scanstitch::tiff_io::CANONICAL_ICC_CREATION_DATE_TIME)
    {
        assert_eq!(u16::from_be_bytes(encoded.try_into().unwrap()), expected);
    }
    moxcms::ColorProfile::new_from_slice(&first)
        .expect("canonicalized payload must remain a valid ICC profile");
}

#[test]
fn test_srgb_proof_gamut_mapping_preserves_d50_lab_lightness_and_hue() {
    fn multiply(matrix: [[f64; 3]; 3], value: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|row| {
            matrix[row][0] * value[0] + matrix[row][1] * value[1] + matrix[row][2] * value[2]
        })
    }

    const SRGB_TO_XYZ_D65: [[f64; 3]; 3] = [
        [0.412_456_4, 0.357_576_1, 0.180_437_5],
        [0.212_672_9, 0.715_152_2, 0.072_175],
        [0.019_333_9, 0.119_192, 0.950_304_1],
    ];
    const D65_TO_D50: [[f64; 3]; 3] = [
        [1.047_811_2, 0.022_886_6, -0.050_127],
        [0.029_542_4, 0.990_484_4, -0.017_049_1],
        [-0.009_234_5, 0.015_043_6, 0.752_131_6],
    ];

    let source = [1.0, 0.0, 0.0];
    let source_xyz = multiply(scanstitch::constants::PROPHOTO_TO_XYZ_D50, source);
    let source_lab = scanstitch::colorspace::xyz_d50_to_lab(source_xyz);
    let mapping = scanstitch::tiff_io::perceptual_srgb_from_linear_prophoto(source);
    assert!(mapping.mapped);
    assert!(mapping.chroma_scale < 1.0);
    assert!(mapping
        .linear_srgb
        .iter()
        .all(|value| (0.0..=1.0).contains(value)));

    let mapped_xyz_d65 = multiply(SRGB_TO_XYZ_D65, mapping.linear_srgb);
    let mapped_lab = scanstitch::colorspace::xyz_d50_to_lab(multiply(D65_TO_D50, mapped_xyz_d65));
    assert_relative_eq!(mapped_lab[0], source_lab[0], epsilon = 2e-4);
    let source_hue = source_lab[2].atan2(source_lab[1]);
    let mapped_hue = mapped_lab[2].atan2(mapped_lab[1]);
    assert_relative_eq!(mapped_hue, source_hue, epsilon = 2e-4);

    let neutral = scanstitch::tiff_io::perceptual_srgb_from_linear_prophoto([0.4; 3]);
    assert!(!neutral.mapped);
    assert_eq!(neutral.chroma_scale, 1.0);
    for channel in neutral.linear_srgb {
        assert_relative_eq!(channel, 0.4, epsilon = 1e-4);
    }
}

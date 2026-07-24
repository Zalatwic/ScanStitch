use std::borrow::Cow;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::atomic_file::AtomicFile;
use crate::constants::{D50_WHITE, MAX_14BIT, MAX_16BIT, PROPHOTO_TO_XYZ_D50};
use crate::input_color::{EmbeddedIccProfile, IccProfileDiagnostics};
use image::codecs::png::PngDecoder;
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, RgbImage};
use ndarray::Array3;
use sha2::{Digest, Sha256};
use tiff::decoder::ifd::Value as TiffValue;
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::tags::Tag;
use tiff::ColorType;

pub const PROPHOTO_LINEAR_ICC_DESCRIPTION: &str = "ScanStitch linear ProPhoto RGB D50";
pub const PROPHOTO_SCENE_REFERRED_FLOAT_DESCRIPTION: &str =
    "ScanStitch scene-referred linear ProPhoto RGB D50 32-bit float";
pub const SRGB_ICC_DESCRIPTION: &str = "sRGB IEC61966-2.1";
pub const SRGB_REVIEW_GAMUT_MAPPING_SPACE: &str =
    "CIELAB_D50_constant_lightness_and_hue_to_sRGB_D65";
pub const ICC_PROFILE_TAG: u16 = 34_675;
pub const TIFF_STREAM_TARGET_BYTES: usize = 1_000_000;
pub const PNG_STREAM_CHUNK_BYTES: usize = 64 * 1024;
/// Creation timestamp of ScanStitch's immutable output-profile definitions.
///
/// ICC header timestamps describe the profile, not the rendered artifact. Keeping
/// this fixed makes otherwise identical output files byte-reproducible; artifact
/// creation time remains available from the report and filesystem metadata.
pub const CANONICAL_ICC_CREATION_DATE_TIME: [u16; 6] = [2026, 5, 8, 0, 0, 0];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TiffIccProfileInspection {
    pub width: u32,
    pub height: u32,
    pub color_type: String,
    pub sample_format: Option<Vec<u16>>,
    pub image_description: Option<String>,
    pub icc_profile_embedded: bool,
    pub icc_profile_size_bytes: Option<usize>,
    pub icc_profile_valid: bool,
    pub icc_profile_description: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PngSrgbProfileInspection {
    pub width: u32,
    pub height: u32,
    pub color_type: String,
    pub icc_profile_embedded: bool,
    pub icc_profile_size_bytes: Option<usize>,
    pub icc_profile_valid: bool,
    pub icc_profile_description: Option<String>,
    pub icc_profile_matches_standard_srgb: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SrgbGamutMappingResult {
    pub linear_srgb: [f64; 3],
    pub mapped: bool,
    pub chroma_scale: f64,
    pub input_finite: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SrgbReviewDiagnostics {
    pub width: usize,
    pub height: usize,
    pub pixel_count: usize,
    pub nonfinite_input_pixel_count: usize,
    pub gamut_mapped_pixel_count: usize,
    pub gamut_mapped_ratio: f64,
    pub mapped_mean_chroma_scale: f64,
    pub mapped_min_chroma_scale: f64,
    pub post_map_out_of_gamut_pixel_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactWriteBufferDiagnostics {
    pub strategy: &'static str,
    pub full_frame_conversion_buffers: bool,
    pub tiff_strip_target_bytes: usize,
    pub primary_tiff_conversion_buffer_bytes: usize,
    pub master_tiff_conversion_buffer_bytes: Option<usize>,
    pub review_png_row_buffer_bytes: Option<usize>,
    pub review_png_stream_chunk_bytes: Option<usize>,
    pub peak_declared_buffer_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkingRangeTransform {
    None,
    UpscaledToWorkingBitDepth,
    DownscaledToWorkingBitDepth,
}

impl WorkingRangeTransform {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::UpscaledToWorkingBitDepth => "upscaled_to_working_bit_depth",
            Self::DownscaledToWorkingBitDepth => "downscaled_to_working_bit_depth",
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkingRangeDiagnostics {
    pub source_bits_per_sample: u8,
    pub target_bit_depth: u8,
    pub source_min: [u16; 3],
    pub source_max: [u16; 3],
    pub working_min: [u16; 3],
    pub working_max: [u16; 3],
    pub transform: WorkingRangeTransform,
    pub scale_factor: f64,
    pub note: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TiffLoadDiagnostics {
    pub width: usize,
    pub height: usize,
    pub color_type: String,
    pub source_bits_per_sample: u8,
    pub source_channel_count: usize,
    pub source_has_alpha: bool,
    pub working_range: WorkingRangeDiagnostics,
    pub decoded_pixel_sha256: String,
    pub orientation: OrientationDiagnostics,
    pub orientation_correction: OrientationCorrectionDiagnostics,
    pub effective_orientation_tag: Option<u16>,
    pub source_icc_profile: Option<IccProfileDiagnostics>,
    pub dng_metadata: Option<DngMetadata>,
    pub dng_level_normalization: Option<DngLevelNormalizationDiagnostics>,
}

pub struct LoadedTiff {
    pub image: Array3<u16>,
    pub diagnostics: TiffLoadDiagnostics,
    pub embedded_icc_profile: Option<EmbeddedIccProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrientationDiagnostics {
    pub tag_value: Option<u16>,
    pub transform: String,
    pub applied: bool,
    pub source_width: usize,
    pub source_height: usize,
    pub output_width: usize,
    pub output_height: usize,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrientationCorrectionDiagnostics {
    pub requested: String,
    pub transform: String,
    pub applied: bool,
    pub input_width: usize,
    pub input_height: usize,
    pub output_width: usize,
    pub output_height: usize,
    pub effective_tag_value: Option<u16>,
    pub effective_transform: String,
    pub source_orientation_materialized_decoded_pixel_sha256: String,
    pub corrected_decoded_pixel_sha256: String,
    pub reason: String,
}

/// Hashes the normalized, orientation-materialized decoded RGB samples together
/// with their dimensions and working precision. The digest is stable across
/// container metadata changes but changes when decoding, range normalization,
/// orientation, dimensions, channel count, or sample values change.
pub fn sha256_decoded_pixels(image: &Array3<u16>, working_bit_depth: u8) -> String {
    let (height, width, channels) = image.dim();
    let mut hasher = Sha256::new();
    hasher.update((width as u64).to_le_bytes());
    hasher.update((height as u64).to_le_bytes());
    hasher.update((channels as u64).to_le_bytes());
    hasher.update([working_bit_depth]);

    const SAMPLE_CHUNK: usize = 32 * 1024;
    let mut bytes = Vec::with_capacity(SAMPLE_CHUNK * 2);
    for sample in image {
        bytes.extend_from_slice(&sample.to_le_bytes());
        if bytes.len() == bytes.capacity() {
            hasher.update(&bytes);
            bytes.clear();
        }
    }
    if !bytes.is_empty() {
        hasher.update(&bytes);
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone, PartialEq)]
pub struct DngLevelNormalizationDiagnostics {
    pub status: String,
    pub black_level: [f64; 3],
    pub white_level: [f64; 3],
    pub source_code_max: u16,
    pub clipped_below_black: [u64; 3],
    pub clipped_above_white: [u64; 3],
    pub reason: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DngMetadata {
    pub make: Option<String>,
    pub model: Option<String>,
    pub software: Option<String>,
    pub unique_camera_model: Option<String>,
    pub calibration_illuminant1: Option<u64>,
    pub calibration_illuminant2: Option<u64>,
    pub color_matrix1: Option<[[f64; 3]; 3]>,
    pub color_matrix2: Option<[[f64; 3]; 3]>,
    pub as_shot_neutral: Option<Vec<f64>>,
    pub black_level: Option<Vec<f64>>,
    pub white_level: Option<Vec<f64>>,
    pub orientation: Option<u16>,
    pub parse_warnings: Vec<String>,
}

impl DngMetadata {
    fn has_reportable_metadata(&self) -> bool {
        self.make.is_some()
            || self.model.is_some()
            || self.software.is_some()
            || self.unique_camera_model.is_some()
            || self.calibration_illuminant1.is_some()
            || self.calibration_illuminant2.is_some()
            || self.color_matrix1.is_some()
            || self.color_matrix2.is_some()
            || self.as_shot_neutral.is_some()
            || self.black_level.is_some()
            || self.white_level.is_some()
            || self.orientation.is_some()
            || !self.parse_warnings.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanSourceInspection {
    pub width: usize,
    pub height: usize,
    pub color_type: String,
    pub source_bits_per_sample: u8,
    pub source_channel_count: usize,
    pub source_has_alpha: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TiffByteOrder {
    Little,
    Big,
}

#[derive(Debug, Clone)]
struct ClassicIfdEntry {
    tag: u16,
    value_type: u16,
    count: u32,
    value_or_offset: [u8; 4],
}

#[derive(Debug, Clone)]
struct ClassicIfd {
    entries: Vec<ClassicIfdEntry>,
    next_offset: u64,
}

#[derive(Debug, Clone)]
struct DngLinearRawCandidate {
    ifd_offset: u64,
    width: usize,
    height: usize,
    color_type: String,
    source_bits_per_sample: u8,
    source_channel_count: usize,
    source_has_alpha: bool,
    strip_offsets: Vec<u64>,
    strip_byte_counts: Vec<u64>,
}

const TIFF_CLASSIC_MAGIC: u16 = 42;
const TIFF_BIGTIFF_MAGIC: u16 = 43;
const TIFF_TYPE_BYTE: u16 = 1;
const TIFF_TYPE_ASCII: u16 = 2;
const TIFF_TYPE_SHORT: u16 = 3;
const TIFF_TYPE_LONG: u16 = 4;
const TIFF_TYPE_RATIONAL: u16 = 5;
const TIFF_TYPE_SBYTE: u16 = 6;
const TIFF_TYPE_UNDEFINED: u16 = 7;
const TIFF_TYPE_SSHORT: u16 = 8;
const TIFF_TYPE_SLONG: u16 = 9;
const TIFF_TYPE_SRATIONAL: u16 = 10;
const TIFF_TYPE_FLOAT: u16 = 11;
const TIFF_TYPE_DOUBLE: u16 = 12;
const TIFF_TYPE_IFD: u16 = 13;
const TIFF_TAG_NEW_SUBFILE_TYPE: u16 = 254;
const TIFF_TAG_IMAGE_WIDTH: u16 = 256;
const TIFF_TAG_IMAGE_LENGTH: u16 = 257;
const TIFF_TAG_BITS_PER_SAMPLE: u16 = 258;
const TIFF_TAG_COMPRESSION: u16 = 259;
const TIFF_TAG_PHOTOMETRIC_INTERPRETATION: u16 = 262;
const TIFF_TAG_MAKE: u16 = 271;
const TIFF_TAG_MODEL: u16 = 272;
const TIFF_TAG_ORIENTATION: u16 = 274;
const TIFF_TAG_STRIP_OFFSETS: u16 = 273;
const TIFF_TAG_SAMPLES_PER_PIXEL: u16 = 277;
const TIFF_TAG_ROWS_PER_STRIP: u16 = 278;
const TIFF_TAG_STRIP_BYTE_COUNTS: u16 = 279;
const TIFF_TAG_PLANAR_CONFIGURATION: u16 = 284;
const TIFF_TAG_SOFTWARE: u16 = 305;
const TIFF_TAG_SUB_IFDS: u16 = 330;
const DNG_TAG_UNIQUE_CAMERA_MODEL: u16 = 50_708;
const DNG_TAG_BLACK_LEVEL: u16 = 50_714;
const DNG_TAG_WHITE_LEVEL: u16 = 50_717;
const DNG_TAG_COLOR_MATRIX1: u16 = 50_721;
const DNG_TAG_COLOR_MATRIX2: u16 = 50_722;
const DNG_TAG_AS_SHOT_NEUTRAL: u16 = 50_728;
const DNG_TAG_CALIBRATION_ILLUMINANT1: u16 = 50_778;
const DNG_TAG_CALIBRATION_ILLUMINANT2: u16 = 50_779;
const TIFF_PHOTOMETRIC_RGB: u16 = 2;
const TIFF_PHOTOMETRIC_DNG_LINEAR_RAW: u16 = 34_892;
const TIFF_COMPRESSION_NONE: u16 = 1;
const TIFF_PLANAR_CHUNKY: u16 = 1;

fn target_bit_depth_max(bit_depth: u8) -> u32 {
    match bit_depth {
        14 => MAX_14BIT as u32,
        _ => MAX_16BIT as u32,
    }
}

fn source_bit_depth_max(bit_depth: u8) -> u32 {
    match bit_depth {
        0 => 0,
        1..=15 => (1u32 << bit_depth) - 1,
        _ => u16::MAX as u32,
    }
}

fn rescale_sample(sample: u16, source_max: u32, target_max: u32) -> u16 {
    if source_max == 0 {
        return 0;
    }
    (((sample as u64) * (target_max as u64) + (source_max as u64 / 2)) / (source_max as u64))
        .min(u16::MAX as u64) as u16
}

fn array_min_max(img: &Array3<u16>) -> ([u16; 3], [u16; 3]) {
    let mut mins = [u16::MAX; 3];
    let mut maxes = [0u16; 3];
    let (h, w, c) = img.dim();
    if h == 0 || w == 0 || c == 0 {
        return ([0; 3], [0; 3]);
    }

    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let v = img[[y, x, ch]];
                mins[ch] = mins[ch].min(v);
                maxes[ch] = maxes[ch].max(v);
            }
        }
    }

    (mins, maxes)
}

pub fn normalize_to_working_bit_depth(
    img: &Array3<u16>,
    source_bits_per_sample: u8,
    target_bit_depth: u8,
) -> (Array3<u16>, WorkingRangeDiagnostics) {
    let (source_min, source_max) = array_min_max(img);
    let source_max_possible = source_bit_depth_max(source_bits_per_sample);
    let target_max = target_bit_depth_max(target_bit_depth);
    let observed_peak = source_max.iter().copied().max().unwrap_or(0) as u32;

    let (transform, scale_factor, note) = if source_bits_per_sample < target_bit_depth {
        (
            WorkingRangeTransform::UpscaledToWorkingBitDepth,
            target_max as f64 / source_max_possible.max(1) as f64,
            Some(format!(
                "expanded {}-bit TIFF samples into the {}-bit working domain",
                source_bits_per_sample, target_bit_depth
            )),
        )
    } else if source_bits_per_sample > target_bit_depth {
        if observed_peak <= target_max {
            (
                WorkingRangeTransform::None,
                1.0,
                Some(format!(
                    "{}-bit TIFF samples already fit within the {}-bit working range",
                    source_bits_per_sample, target_bit_depth
                )),
            )
        } else {
            (
                WorkingRangeTransform::DownscaledToWorkingBitDepth,
                target_max as f64 / source_max_possible.max(1) as f64,
                Some(format!(
                    "compressed {}-bit TIFF samples into the {}-bit working domain",
                    source_bits_per_sample, target_bit_depth
                )),
            )
        }
    } else {
        (WorkingRangeTransform::None, 1.0, None)
    };

    let working = if transform == WorkingRangeTransform::None {
        img.clone()
    } else {
        img.mapv(|sample| rescale_sample(sample, source_max_possible, target_max))
    };
    let (working_min, working_max) = array_min_max(&working);

    (
        working,
        WorkingRangeDiagnostics {
            source_bits_per_sample,
            target_bit_depth,
            source_min,
            source_max,
            working_min,
            working_max,
            transform,
            scale_factor,
            note,
        },
    )
}

fn canonical_dng_channel_levels(values: Option<&[f64]>, default: f64) -> Option<[f64; 3]> {
    let Some(values) = values else {
        return Some([default; 3]);
    };
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    if values.len() == 1 {
        return Some([values[0]; 3]);
    }
    if values.len().is_multiple_of(3) {
        let samples_per_channel = values.len() / 3;
        return Some(std::array::from_fn(|channel| {
            values.iter().skip(channel).step_by(3).sum::<f64>() / samples_per_channel as f64
        }));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Some([mean; 3])
}

fn normalize_dng_black_white_levels(
    image: &Array3<u16>,
    source_bits_per_sample: u8,
    metadata: Option<&DngMetadata>,
) -> (Array3<u16>, DngLevelNormalizationDiagnostics) {
    let source_code_max = source_bit_depth_max(source_bits_per_sample).min(u16::MAX as u32) as u16;
    let black_values = metadata.and_then(|metadata| metadata.black_level.as_deref());
    let white_values = metadata.and_then(|metadata| metadata.white_level.as_deref());
    let metadata_present = black_values.is_some() || white_values.is_some();
    let Some(black_level) = canonical_dng_channel_levels(black_values, 0.0) else {
        return (
            image.clone(),
            DngLevelNormalizationDiagnostics {
                status: "invalid_metadata".to_string(),
                black_level: [0.0; 3],
                white_level: [source_code_max as f64; 3],
                source_code_max,
                clipped_below_black: [0; 3],
                clipped_above_white: [0; 3],
                reason: "DNG BlackLevel contains no usable finite samples; raw code values were preserved"
                    .to_string(),
            },
        );
    };
    let Some(white_level) = canonical_dng_channel_levels(white_values, source_code_max as f64)
    else {
        return (
            image.clone(),
            DngLevelNormalizationDiagnostics {
                status: "invalid_metadata".to_string(),
                black_level,
                white_level: [source_code_max as f64; 3],
                source_code_max,
                clipped_below_black: [0; 3],
                clipped_above_white: [0; 3],
                reason: "DNG WhiteLevel contains no usable finite samples; raw code values were preserved"
                    .to_string(),
            },
        );
    };
    if (0..3).any(|channel| white_level[channel] <= black_level[channel]) {
        return (
            image.clone(),
            DngLevelNormalizationDiagnostics {
                status: "invalid_metadata".to_string(),
                black_level,
                white_level,
                source_code_max,
                clipped_below_black: [0; 3],
                clipped_above_white: [0; 3],
                reason: "DNG WhiteLevel must exceed BlackLevel in every RGB channel; raw code values were preserved"
                    .to_string(),
            },
        );
    }
    if !metadata_present {
        return (
            image.clone(),
            DngLevelNormalizationDiagnostics {
                status: "not_present".to_string(),
                black_level,
                white_level,
                source_code_max,
                clipped_below_black: [0; 3],
                clipped_above_white: [0; 3],
                reason: "DNG BlackLevel and WhiteLevel were absent; default full code range was preserved"
                    .to_string(),
            },
        );
    }

    let mut clipped_below_black = [0u64; 3];
    let mut clipped_above_white = [0u64; 3];
    let mut normalized = Array3::<u16>::zeros(image.dim());
    let (height, width, _) = image.dim();
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let sample = image[[y, x, channel]] as f64;
                if sample < black_level[channel] {
                    clipped_below_black[channel] += 1;
                }
                if sample > white_level[channel] {
                    clipped_above_white[channel] += 1;
                }
                let scaled = ((sample - black_level[channel])
                    / (white_level[channel] - black_level[channel]))
                    .clamp(0.0, 1.0)
                    * source_code_max as f64;
                normalized[[y, x, channel]] = scaled.round() as u16;
            }
        }
    }
    (
        normalized,
        DngLevelNormalizationDiagnostics {
            status: "applied".to_string(),
            black_level,
            white_level,
            source_code_max,
            clipped_below_black,
            clipped_above_white,
            reason: "DNG code values were black-subtracted and normalized by per-channel WhiteLevel before downstream processing"
                .to_string(),
        },
    )
}

fn orientation_name(orientation: Orientation) -> &'static str {
    match orientation {
        Orientation::NoTransforms => "identity",
        Orientation::Rotate90 => "rotate_90_clockwise",
        Orientation::Rotate180 => "rotate_180",
        Orientation::Rotate270 => "rotate_270_clockwise",
        Orientation::FlipHorizontal => "flip_horizontal",
        Orientation::FlipVertical => "flip_vertical",
        Orientation::Rotate90FlipH => "rotate_90_clockwise_then_flip_horizontal",
        Orientation::Rotate270FlipH => "rotate_270_clockwise_then_flip_horizontal",
    }
}

pub fn orientation_transform_name(tag_value: Option<u16>) -> &'static str {
    match tag_value {
        Some(2) => "flip_horizontal",
        Some(3) => "rotate_180",
        Some(4) => "flip_vertical",
        Some(5) => "rotate_90_clockwise_then_flip_horizontal",
        Some(6) => "rotate_90_clockwise",
        Some(7) => "rotate_270_clockwise_then_flip_horizontal",
        Some(8) => "rotate_270_clockwise",
        _ => "identity",
    }
}

fn orientation_linear_transform(tag_value: u16) -> [[i8; 2]; 2] {
    match tag_value {
        2 => [[-1, 0], [0, 1]],
        3 => [[-1, 0], [0, -1]],
        4 => [[1, 0], [0, -1]],
        5 => [[0, 1], [1, 0]],
        6 => [[0, -1], [1, 0]],
        7 => [[0, -1], [-1, 0]],
        8 => [[0, 1], [-1, 0]],
        _ => [[1, 0], [0, 1]],
    }
}

fn multiply_orientation_transforms(left: [[i8; 2]; 2], right: [[i8; 2]; 2]) -> [[i8; 2]; 2] {
    [
        [
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ],
        [
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ],
    ]
}

/// Compose a metadata orientation with a correction applied to the already-oriented pixels.
/// The result is the equivalent original-scanner-to-final-output EXIF transform.
pub fn compose_orientation_tags(
    source_tag_value: Option<u16>,
    correction_tag_value: u16,
) -> Option<u16> {
    if correction_tag_value == 1 {
        return source_tag_value;
    }
    let source = source_tag_value
        .filter(|value| (1..=8).contains(value))
        .unwrap_or(1);
    let correction = correction_tag_value.clamp(1, 8);
    let composed = multiply_orientation_transforms(
        orientation_linear_transform(correction),
        orientation_linear_transform(source),
    );
    (1..=8).find(|tag| orientation_linear_transform(*tag) == composed)
}

fn no_orientation_correction_diagnostics(
    orientation: &OrientationDiagnostics,
    decoded_pixel_sha256: &str,
) -> OrientationCorrectionDiagnostics {
    OrientationCorrectionDiagnostics {
        requested: "none".to_string(),
        transform: "identity".to_string(),
        applied: false,
        input_width: orientation.output_width,
        input_height: orientation.output_height,
        output_width: orientation.output_width,
        output_height: orientation.output_height,
        effective_tag_value: orientation.tag_value,
        effective_transform: orientation.transform.clone(),
        source_orientation_materialized_decoded_pixel_sha256: decoded_pixel_sha256.to_string(),
        corrected_decoded_pixel_sha256: decoded_pixel_sha256.to_string(),
        reason: "no metadata-relative semantic-orientation correction was requested".to_string(),
    }
}

/// Apply an explicit orthogonal correction after source orientation metadata and update the exact
/// decoded-pixel binding plus the effective original-scanner coordinate transform.
pub fn apply_orientation_correction_u16(
    loaded: &mut LoadedTiff,
    correction_tag_value: u16,
    requested: &str,
    working_bit_depth: u8,
) -> Result<(), String> {
    if !(1..=8).contains(&correction_tag_value) {
        return Err(format!(
            "orientation correction tag {correction_tag_value} is outside EXIF range 1-8"
        ));
    }
    let source_hash = loaded.diagnostics.decoded_pixel_sha256.clone();
    let input_width = loaded.diagnostics.width;
    let input_height = loaded.diagnostics.height;
    let source_tag = loaded.diagnostics.orientation.tag_value;
    let source = std::mem::replace(&mut loaded.image, Array3::zeros((0, 0, 3)));
    let (corrected, correction) = apply_orientation_u16(source, Some(correction_tag_value));
    let corrected_hash = sha256_decoded_pixels(&corrected, working_bit_depth);
    let effective_tag_value = compose_orientation_tags(source_tag, correction_tag_value);
    let effective_transform = if correction_tag_value == 1 {
        loaded.diagnostics.orientation.transform.clone()
    } else {
        orientation_transform_name(effective_tag_value).to_string()
    };
    loaded.image = corrected;
    loaded.diagnostics.width = correction.output_width;
    loaded.diagnostics.height = correction.output_height;
    loaded.diagnostics.decoded_pixel_sha256 = corrected_hash.clone();
    loaded.diagnostics.effective_orientation_tag = effective_tag_value;
    loaded.diagnostics.orientation_correction = OrientationCorrectionDiagnostics {
        requested: requested.to_string(),
        transform: correction.transform,
        applied: correction.applied,
        input_width,
        input_height,
        output_width: correction.output_width,
        output_height: correction.output_height,
        effective_tag_value,
        effective_transform,
        source_orientation_materialized_decoded_pixel_sha256: source_hash,
        corrected_decoded_pixel_sha256: corrected_hash,
        reason: if correction.applied {
            "explicit semantic-orientation correction was applied after EXIF/DNG orientation and before scanner linearization, border detection, deskew, and stitching".to_string()
        } else {
            "explicit semantic-orientation correction was identity after EXIF/DNG orientation"
                .to_string()
        },
    };
    Ok(())
}

fn apply_orientation_u16(
    image: Array3<u16>,
    tag_value: Option<u16>,
) -> (Array3<u16>, OrientationDiagnostics) {
    let (source_height, source_width, channels) = image.dim();
    let orientation = tag_value
        .and_then(|value| u8::try_from(value).ok())
        .and_then(Orientation::from_exif);
    let Some(orientation) = orientation else {
        let reason = if tag_value.is_some() {
            "TIFF Orientation tag was outside the supported EXIF range 1-8; pixels were left unchanged"
        } else {
            "TIFF Orientation tag was absent; pixels were assumed to be top-left oriented"
        };
        return (
            image,
            OrientationDiagnostics {
                tag_value,
                transform: "identity".to_string(),
                applied: false,
                source_width,
                source_height,
                output_width: source_width,
                output_height: source_height,
                reason: reason.to_string(),
            },
        );
    };
    if orientation == Orientation::NoTransforms {
        return (
            image,
            OrientationDiagnostics {
                tag_value,
                transform: orientation_name(orientation).to_string(),
                applied: false,
                source_width,
                source_height,
                output_width: source_width,
                output_height: source_height,
                reason: "TIFF Orientation tag already describes top-left pixel order".to_string(),
            },
        );
    }

    let swaps_dimensions = matches!(
        orientation,
        Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Rotate90FlipH
            | Orientation::Rotate270FlipH
    );
    let (output_height, output_width) = if swaps_dimensions {
        (source_width, source_height)
    } else {
        (source_height, source_width)
    };
    let mut output = Array3::<u16>::zeros((output_height, output_width, channels));
    for output_y in 0..output_height {
        for output_x in 0..output_width {
            let (source_y, source_x) = match orientation {
                Orientation::NoTransforms => (output_y, output_x),
                Orientation::FlipHorizontal => (output_y, source_width - 1 - output_x),
                Orientation::Rotate180 => {
                    (source_height - 1 - output_y, source_width - 1 - output_x)
                }
                Orientation::FlipVertical => (source_height - 1 - output_y, output_x),
                Orientation::Rotate90FlipH => (output_x, output_y),
                Orientation::Rotate90 => (source_height - 1 - output_x, output_y),
                Orientation::Rotate270FlipH => {
                    (source_height - 1 - output_x, source_width - 1 - output_y)
                }
                Orientation::Rotate270 => (output_x, source_width - 1 - output_y),
            };
            for channel in 0..channels {
                output[[output_y, output_x, channel]] = image[[source_y, source_x, channel]];
            }
        }
    }
    (
        output,
        OrientationDiagnostics {
            tag_value,
            transform: orientation_name(orientation).to_string(),
            applied: true,
            source_width,
            source_height,
            output_width,
            output_height,
            reason: "TIFF Orientation metadata was applied before border detection and stitching"
                .to_string(),
        },
    )
}

fn decode_layout(
    color_type: ColorType,
) -> Result<(u8, usize, bool, String), Box<dyn std::error::Error>> {
    match color_type {
        ColorType::RGB(bits) => Ok((bits, 3, false, format!("RGB{}", bits))),
        ColorType::RGBA(bits) => Ok((bits, 4, true, format!("RGBA{}", bits))),
        other => Err(format!("unsupported TIFF color type: {:?}", other).into()),
    }
}

fn decoder_for_scanner_tiff(
    file: File,
) -> Result<Decoder<BufReader<File>>, Box<dyn std::error::Error>> {
    Ok(Decoder::new(BufReader::new(file))?.with_limits(Limits::unlimited()))
}

fn tiff_value_into_bytes(value: TiffValue) -> Result<Vec<u8>, String> {
    fn append(value: TiffValue, output: &mut Vec<u8>) -> Result<(), String> {
        match value {
            TiffValue::Byte(value) => output.push(value),
            TiffValue::Short(value) => output.push(
                u8::try_from(value)
                    .map_err(|_| format!("TIFF tag contains non-byte SHORT value {value}"))?,
            ),
            TiffValue::Unsigned(value) => output.push(
                u8::try_from(value)
                    .map_err(|_| format!("TIFF tag contains non-byte LONG value {value}"))?,
            ),
            TiffValue::UnsignedBig(value) => output.push(
                u8::try_from(value)
                    .map_err(|_| format!("TIFF tag contains non-byte LONG8 value {value}"))?,
            ),
            TiffValue::List(values) => {
                for value in values {
                    append(value, output)?;
                }
            }
            other => {
                return Err(format!(
                    "TIFF tag has unsupported value representation {other:?}"
                ));
            }
        }
        Ok(())
    }

    let mut output = Vec::new();
    append(value, &mut output)?;
    Ok(output)
}

fn interleaved_u8_to_rgb_array(
    raw: Vec<u8>,
    width: usize,
    height: usize,
    channels: usize,
) -> Result<Array3<u16>, Box<dyn std::error::Error>> {
    let mut out = Vec::with_capacity(width * height * 3);
    for pixel in raw.chunks_exact(channels) {
        out.push(pixel[0] as u16);
        out.push(pixel[1] as u16);
        out.push(pixel[2] as u16);
    }
    Ok(Array3::from_shape_vec((height, width, 3), out)?)
}

fn interleaved_u16_to_rgb_array(
    raw: Vec<u16>,
    width: usize,
    height: usize,
    channels: usize,
) -> Result<Array3<u16>, Box<dyn std::error::Error>> {
    let mut out = Vec::with_capacity(width * height * 3);
    for pixel in raw.chunks_exact(channels) {
        out.push(pixel[0]);
        out.push(pixel[1]);
        out.push(pixel[2]);
    }
    Ok(Array3::from_shape_vec((height, width, 3), out)?)
}

pub fn inspect_scan_source(
    path: &Path,
) -> Result<ScanSourceInspection, Box<dyn std::error::Error>> {
    if is_dng_path(path) {
        let mut file = File::open(path)?;
        let (byte_order, first_ifd_offset) = read_classic_tiff_header(&mut file)?;
        let candidate = select_dng_linear_raw_candidate(&mut file, byte_order, first_ifd_offset)?;
        return Ok(candidate.inspection());
    }
    inspect_standard_tiff_source(path)
}

fn inspect_standard_tiff_source(
    path: &Path,
) -> Result<ScanSourceInspection, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut decoder = decoder_for_scanner_tiff(file)?;
    let (width, height) = decoder.dimensions()?;
    let color_type = decoder.colortype()?;
    let (source_bits_per_sample, source_channel_count, source_has_alpha, color_label) =
        decode_layout(color_type)?;
    Ok(ScanSourceInspection {
        width: width as usize,
        height: height as usize,
        color_type: color_label,
        source_bits_per_sample,
        source_channel_count,
        source_has_alpha,
    })
}

fn is_dng_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("dng"))
}

/// Load a TIFF image into the requested working bit depth, preserving actual
/// on-disk sample precision instead of silently expanding 8-bit data to 16-bit.
pub fn load_tiff_u16(
    path: &Path,
    target_bit_depth: u8,
) -> Result<LoadedTiff, Box<dyn std::error::Error>> {
    if is_dng_path(path) {
        return load_dng_linear_raw_u16(path, target_bit_depth);
    }

    let file = File::open(path)?;
    let mut decoder = decoder_for_scanner_tiff(file)?;
    let (width, height) = decoder.dimensions()?;
    let color_type = decoder.colortype()?;
    let (source_bits_per_sample, source_channel_count, source_has_alpha, color_label) =
        decode_layout(color_type)?;
    let orientation_tag = decoder
        .get_tag_u32(Tag::Orientation)
        .ok()
        .and_then(|value| u16::try_from(value).ok());
    let embedded_icc_profile = decoder
        .get_tag(Tag::Unknown(ICC_PROFILE_TAG))
        .ok()
        .and_then(|value| tiff_value_into_bytes(value).ok())
        .filter(|bytes| !bytes.is_empty())
        .map(EmbeddedIccProfile::from_bytes);

    let source = match decoder.read_image()? {
        DecodingResult::U8(raw) => {
            interleaved_u8_to_rgb_array(raw, width as usize, height as usize, source_channel_count)?
        }
        DecodingResult::U16(raw) => interleaved_u16_to_rgb_array(
            raw,
            width as usize,
            height as usize,
            source_channel_count,
        )?,
        other => {
            return Err(format!("unsupported TIFF sample type: {:?}", other).into());
        }
    };

    let (working_image, working_range) =
        normalize_to_working_bit_depth(&source, source_bits_per_sample, target_bit_depth);
    let (image, orientation) = apply_orientation_u16(working_image, orientation_tag);
    let decoded_pixel_sha256 = sha256_decoded_pixels(&image, target_bit_depth);
    let source_icc_profile = embedded_icc_profile
        .as_ref()
        .map(|profile| profile.diagnostics.clone());

    let orientation_correction =
        no_orientation_correction_diagnostics(&orientation, &decoded_pixel_sha256);
    Ok(LoadedTiff {
        image,
        diagnostics: TiffLoadDiagnostics {
            width: orientation.output_width,
            height: orientation.output_height,
            color_type: color_label,
            source_bits_per_sample,
            source_channel_count,
            source_has_alpha,
            working_range,
            decoded_pixel_sha256,
            effective_orientation_tag: orientation.tag_value,
            orientation,
            orientation_correction,
            source_icc_profile,
            dng_metadata: None,
            dng_level_normalization: None,
        },
        embedded_icc_profile,
    })
}

fn load_dng_linear_raw_u16(
    path: &Path,
    target_bit_depth: u8,
) -> Result<LoadedTiff, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let (byte_order, first_ifd_offset) = read_classic_tiff_header(&mut file)?;
    let first_ifd = read_classic_ifd(&mut file, byte_order, first_ifd_offset)?;
    let primary_metadata = parse_dng_metadata(&mut file, byte_order, &first_ifd);
    let candidate = select_dng_linear_raw_candidate(&mut file, byte_order, first_ifd_offset)?;
    let candidate_metadata = if candidate.ifd_offset == first_ifd_offset {
        None
    } else {
        let candidate_ifd = read_classic_ifd(&mut file, byte_order, candidate.ifd_offset)?;
        parse_dng_metadata(&mut file, byte_order, &candidate_ifd)
    };
    let dng_metadata = merge_dng_metadata(primary_metadata, candidate_metadata);
    let source = read_dng_linear_raw_pixels(&mut file, byte_order, &candidate)?;
    let (level_normalized, dng_level_normalization) = normalize_dng_black_white_levels(
        &source,
        candidate.source_bits_per_sample,
        dng_metadata.as_ref(),
    );
    let (working_image, working_range) = normalize_to_working_bit_depth(
        &level_normalized,
        candidate.source_bits_per_sample,
        target_bit_depth,
    );
    let orientation_tag = dng_metadata
        .as_ref()
        .and_then(|metadata| metadata.orientation);
    let (image, orientation) = apply_orientation_u16(working_image, orientation_tag);
    let decoded_pixel_sha256 = sha256_decoded_pixels(&image, target_bit_depth);

    let orientation_correction =
        no_orientation_correction_diagnostics(&orientation, &decoded_pixel_sha256);
    Ok(LoadedTiff {
        image,
        diagnostics: TiffLoadDiagnostics {
            width: orientation.output_width,
            height: orientation.output_height,
            color_type: candidate.color_type,
            source_bits_per_sample: candidate.source_bits_per_sample,
            source_channel_count: candidate.source_channel_count,
            source_has_alpha: candidate.source_has_alpha,
            working_range,
            decoded_pixel_sha256,
            effective_orientation_tag: orientation.tag_value,
            orientation,
            orientation_correction,
            source_icc_profile: None,
            dng_metadata,
            dng_level_normalization: Some(dng_level_normalization),
        },
        embedded_icc_profile: None,
    })
}

impl DngLinearRawCandidate {
    fn inspection(&self) -> ScanSourceInspection {
        ScanSourceInspection {
            width: self.width,
            height: self.height,
            color_type: self.color_type.clone(),
            source_bits_per_sample: self.source_bits_per_sample,
            source_channel_count: self.source_channel_count,
            source_has_alpha: self.source_has_alpha,
        }
    }

    fn pixel_count(&self) -> usize {
        self.width.saturating_mul(self.height)
    }
}

impl TiffByteOrder {
    fn read_u16(self, bytes: &[u8]) -> u16 {
        let bytes = [bytes[0], bytes[1]];
        match self {
            Self::Little => u16::from_le_bytes(bytes),
            Self::Big => u16::from_be_bytes(bytes),
        }
    }

    fn read_u32(self, bytes: &[u8]) -> u32 {
        let bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
        match self {
            Self::Little => u32::from_le_bytes(bytes),
            Self::Big => u32::from_be_bytes(bytes),
        }
    }

    fn read_i16(self, bytes: &[u8]) -> i16 {
        let bytes = [bytes[0], bytes[1]];
        match self {
            Self::Little => i16::from_le_bytes(bytes),
            Self::Big => i16::from_be_bytes(bytes),
        }
    }

    fn read_i32(self, bytes: &[u8]) -> i32 {
        let bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
        match self {
            Self::Little => i32::from_le_bytes(bytes),
            Self::Big => i32::from_be_bytes(bytes),
        }
    }

    fn read_f32(self, bytes: &[u8]) -> f32 {
        f32::from_bits(self.read_u32(bytes))
    }

    fn read_f64(self, bytes: &[u8]) -> f64 {
        let bytes = [
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ];
        match self {
            Self::Little => f64::from_le_bytes(bytes),
            Self::Big => f64::from_be_bytes(bytes),
        }
    }
}

fn read_classic_tiff_header(
    file: &mut File,
) -> Result<(TiffByteOrder, u64), Box<dyn std::error::Error>> {
    file.seek(SeekFrom::Start(0))?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let byte_order = match &header[0..2] {
        b"II" => TiffByteOrder::Little,
        b"MM" => TiffByteOrder::Big,
        _ => return Err("not a TIFF/DNG file".into()),
    };
    let magic = byte_order.read_u16(&header[2..4]);
    if magic == TIFF_BIGTIFF_MAGIC {
        return Err("BigTIFF DNG files are not supported by the ScanStitch raw loader".into());
    }
    if magic != TIFF_CLASSIC_MAGIC {
        return Err(format!("unsupported TIFF/DNG magic {magic}").into());
    }
    Ok((byte_order, byte_order.read_u32(&header[4..8]) as u64))
}

fn read_classic_ifd(
    file: &mut File,
    byte_order: TiffByteOrder,
    offset: u64,
) -> Result<ClassicIfd, Box<dyn std::error::Error>> {
    file.seek(SeekFrom::Start(offset))?;
    let mut count_bytes = [0u8; 2];
    file.read_exact(&mut count_bytes)?;
    let entry_count = byte_order.read_u16(&count_bytes) as usize;
    let mut entries = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let mut raw = [0u8; 12];
        file.read_exact(&mut raw)?;
        entries.push(ClassicIfdEntry {
            tag: byte_order.read_u16(&raw[0..2]),
            value_type: byte_order.read_u16(&raw[2..4]),
            count: byte_order.read_u32(&raw[4..8]),
            value_or_offset: [raw[8], raw[9], raw[10], raw[11]],
        });
    }
    let mut next_bytes = [0u8; 4];
    file.read_exact(&mut next_bytes)?;
    Ok(ClassicIfd {
        entries,
        next_offset: byte_order.read_u32(&next_bytes) as u64,
    })
}

fn optional_metadata<T>(
    warnings: &mut Vec<String>,
    label: &str,
    result: Result<Option<T>, Box<dyn std::error::Error>>,
) -> Option<T> {
    match result {
        Ok(value) => value,
        Err(err) => {
            warnings.push(format!("{label}: {err}"));
            None
        }
    }
}

fn parse_dng_metadata(
    file: &mut File,
    byte_order: TiffByteOrder,
    first_ifd: &ClassicIfd,
) -> Option<DngMetadata> {
    let mut warnings = Vec::new();
    let make = optional_metadata(
        &mut warnings,
        "Make",
        ifd_ascii_string_opt(file, byte_order, first_ifd, TIFF_TAG_MAKE),
    );
    let model = optional_metadata(
        &mut warnings,
        "Model",
        ifd_ascii_string_opt(file, byte_order, first_ifd, TIFF_TAG_MODEL),
    );
    let software = optional_metadata(
        &mut warnings,
        "Software",
        ifd_ascii_string_opt(file, byte_order, first_ifd, TIFF_TAG_SOFTWARE),
    );
    let orientation_value = optional_metadata(
        &mut warnings,
        "Orientation",
        ifd_value_u64(file, byte_order, first_ifd, TIFF_TAG_ORIENTATION),
    );
    let orientation = orientation_value.and_then(|value| match u16::try_from(value) {
        Ok(value) => Some(value),
        Err(_) => {
            warnings.push(format!(
                "Orientation: value {value} does not fit the TIFF SHORT domain"
            ));
            None
        }
    });
    let unique_camera_model = optional_metadata(
        &mut warnings,
        "UniqueCameraModel",
        ifd_ascii_string_opt(file, byte_order, first_ifd, DNG_TAG_UNIQUE_CAMERA_MODEL),
    );
    let calibration_illuminant1 = optional_metadata(
        &mut warnings,
        "CalibrationIlluminant1",
        ifd_value_u64(file, byte_order, first_ifd, DNG_TAG_CALIBRATION_ILLUMINANT1),
    );
    let calibration_illuminant2 = optional_metadata(
        &mut warnings,
        "CalibrationIlluminant2",
        ifd_value_u64(file, byte_order, first_ifd, DNG_TAG_CALIBRATION_ILLUMINANT2),
    );
    let color_matrix1 = optional_metadata(
        &mut warnings,
        "ColorMatrix1",
        ifd_matrix3_f64_opt(file, byte_order, first_ifd, DNG_TAG_COLOR_MATRIX1),
    );
    let color_matrix2 = optional_metadata(
        &mut warnings,
        "ColorMatrix2",
        ifd_matrix3_f64_opt(file, byte_order, first_ifd, DNG_TAG_COLOR_MATRIX2),
    );
    let as_shot_neutral = optional_metadata(
        &mut warnings,
        "AsShotNeutral",
        ifd_values_f64_opt(file, byte_order, first_ifd, DNG_TAG_AS_SHOT_NEUTRAL),
    );
    let black_level = optional_metadata(
        &mut warnings,
        "BlackLevel",
        ifd_values_f64_opt(file, byte_order, first_ifd, DNG_TAG_BLACK_LEVEL),
    );
    let white_level = optional_metadata(
        &mut warnings,
        "WhiteLevel",
        ifd_values_f64_opt(file, byte_order, first_ifd, DNG_TAG_WHITE_LEVEL),
    );

    let metadata = DngMetadata {
        make,
        model,
        software,
        unique_camera_model,
        calibration_illuminant1,
        calibration_illuminant2,
        color_matrix1,
        color_matrix2,
        as_shot_neutral,
        black_level,
        white_level,
        orientation,
        parse_warnings: warnings,
    };
    metadata.has_reportable_metadata().then_some(metadata)
}

fn merge_dng_metadata(
    primary: Option<DngMetadata>,
    selected_ifd: Option<DngMetadata>,
) -> Option<DngMetadata> {
    let mut merged = primary.unwrap_or_default();
    if let Some(selected) = selected_ifd {
        merged.make = selected.make.or(merged.make);
        merged.model = selected.model.or(merged.model);
        merged.software = selected.software.or(merged.software);
        merged.unique_camera_model = selected.unique_camera_model.or(merged.unique_camera_model);
        merged.calibration_illuminant1 = selected
            .calibration_illuminant1
            .or(merged.calibration_illuminant1);
        merged.calibration_illuminant2 = selected
            .calibration_illuminant2
            .or(merged.calibration_illuminant2);
        merged.color_matrix1 = selected.color_matrix1.or(merged.color_matrix1);
        merged.color_matrix2 = selected.color_matrix2.or(merged.color_matrix2);
        merged.as_shot_neutral = selected.as_shot_neutral.or(merged.as_shot_neutral);
        merged.black_level = selected.black_level.or(merged.black_level);
        merged.white_level = selected.white_level.or(merged.white_level);
        merged.orientation = selected.orientation.or(merged.orientation);
        merged.parse_warnings.extend(
            selected
                .parse_warnings
                .into_iter()
                .map(|warning| format!("selected image IFD: {warning}")),
        );
    }
    merged.has_reportable_metadata().then_some(merged)
}

fn select_dng_linear_raw_candidate(
    file: &mut File,
    byte_order: TiffByteOrder,
    first_ifd_offset: u64,
) -> Result<DngLinearRawCandidate, Box<dyn std::error::Error>> {
    let mut offsets = Vec::new();
    let mut pending = vec![first_ifd_offset];
    while let Some(offset) = pending.pop() {
        if offset == 0 || offsets.contains(&offset) || offsets.len() >= 32 {
            continue;
        }
        let ifd = read_classic_ifd(file, byte_order, offset)?;
        offsets.push(offset);
        if ifd.next_offset != 0 {
            pending.push(ifd.next_offset);
        }
        if let Ok(sub_ifds) = ifd_values_u64(file, byte_order, &ifd, TIFF_TAG_SUB_IFDS) {
            pending.extend(sub_ifds);
        }
    }

    let mut candidates = Vec::new();
    for offset in offsets {
        let ifd = read_classic_ifd(file, byte_order, offset)?;
        if let Some(candidate) = dng_candidate_from_ifd(file, byte_order, offset, &ifd)? {
            candidates.push(candidate);
        }
    }

    candidates
        .into_iter()
        .max_by_key(DngLinearRawCandidate::pixel_count)
        .ok_or_else(|| {
            "DNG does not contain a supported uncompressed 3-channel RGB/LinearRaw image directory"
                .into()
        })
}

fn dng_candidate_from_ifd(
    file: &mut File,
    byte_order: TiffByteOrder,
    ifd_offset: u64,
    ifd: &ClassicIfd,
) -> Result<Option<DngLinearRawCandidate>, Box<dyn std::error::Error>> {
    let width = required_ifd_value_u64(file, byte_order, ifd, TIFF_TAG_IMAGE_WIDTH)? as usize;
    let height = required_ifd_value_u64(file, byte_order, ifd, TIFF_TAG_IMAGE_LENGTH)? as usize;
    let compression = ifd_value_u64(file, byte_order, ifd, TIFF_TAG_COMPRESSION)?.unwrap_or(1);
    if compression != TIFF_COMPRESSION_NONE as u64 {
        return Ok(None);
    }
    let photometric =
        required_ifd_value_u64(file, byte_order, ifd, TIFF_TAG_PHOTOMETRIC_INTERPRETATION)? as u16;
    if photometric != TIFF_PHOTOMETRIC_RGB && photometric != TIFF_PHOTOMETRIC_DNG_LINEAR_RAW {
        return Ok(None);
    }
    let samples_per_pixel =
        ifd_value_u64(file, byte_order, ifd, TIFF_TAG_SAMPLES_PER_PIXEL)?.unwrap_or(1) as usize;
    if samples_per_pixel != 3 {
        return Ok(None);
    }
    let planar_config =
        ifd_value_u64(file, byte_order, ifd, TIFF_TAG_PLANAR_CONFIGURATION)?.unwrap_or(1);
    if planar_config != TIFF_PLANAR_CHUNKY as u64 {
        return Ok(None);
    }
    let bits_per_sample = ifd_values_u64(file, byte_order, ifd, TIFF_TAG_BITS_PER_SAMPLE)?;
    if bits_per_sample.is_empty()
        || !bits_per_sample
            .iter()
            .all(|bits| *bits == bits_per_sample[0])
    {
        return Ok(None);
    }
    let source_bits_per_sample = u8::try_from(bits_per_sample[0])?;
    if source_bits_per_sample != 8 && source_bits_per_sample != 16 {
        return Ok(None);
    }
    let strip_offsets = ifd_values_u64(file, byte_order, ifd, TIFF_TAG_STRIP_OFFSETS)?;
    let strip_byte_counts = ifd_values_u64(file, byte_order, ifd, TIFF_TAG_STRIP_BYTE_COUNTS)?;
    let rows_per_strip = ifd_value_u64(file, byte_order, ifd, TIFF_TAG_ROWS_PER_STRIP)?;
    if strip_offsets.is_empty()
        || strip_offsets.len() != strip_byte_counts.len()
        || rows_per_strip.unwrap_or(0) == 0
    {
        return Ok(None);
    }

    let color_type = if photometric == TIFF_PHOTOMETRIC_DNG_LINEAR_RAW {
        format!("DNG_LINEAR_RAW{source_bits_per_sample}")
    } else {
        format!("DNG_RGB{source_bits_per_sample}")
    };

    let _new_subfile_type = ifd_value_u64(file, byte_order, ifd, TIFF_TAG_NEW_SUBFILE_TYPE)?;

    Ok(Some(DngLinearRawCandidate {
        ifd_offset,
        width,
        height,
        color_type,
        source_bits_per_sample,
        source_channel_count: samples_per_pixel,
        source_has_alpha: false,
        strip_offsets,
        strip_byte_counts,
    }))
}

fn read_dng_linear_raw_pixels(
    file: &mut File,
    byte_order: TiffByteOrder,
    candidate: &DngLinearRawCandidate,
) -> Result<Array3<u16>, Box<dyn std::error::Error>> {
    let expected_samples = candidate
        .width
        .checked_mul(candidate.height)
        .and_then(|pixels| pixels.checked_mul(candidate.source_channel_count))
        .ok_or("DNG dimensions overflow the in-memory image buffer")?;
    let bytes_per_sample = usize::from(candidate.source_bits_per_sample / 8);
    let mut raw = Vec::with_capacity(expected_samples);

    for (offset, byte_count) in candidate
        .strip_offsets
        .iter()
        .zip(candidate.strip_byte_counts.iter())
    {
        let byte_count = usize::try_from(*byte_count)?;
        if byte_count % bytes_per_sample != 0 {
            return Err(format!("DNG strip byte count {byte_count} is not sample-aligned").into());
        }
        file.seek(SeekFrom::Start(*offset))?;
        let mut bytes = vec![0u8; byte_count];
        file.read_exact(&mut bytes)?;
        match candidate.source_bits_per_sample {
            8 => raw.extend(bytes.into_iter().map(u16::from)),
            16 => {
                for sample in bytes.chunks_exact(2) {
                    raw.push(byte_order.read_u16(sample));
                }
            }
            bits => return Err(format!("unsupported DNG bit depth {bits}").into()),
        }
    }

    if raw.len() != expected_samples {
        return Err(format!(
            "DNG sample count mismatch: decoded {}, expected {}",
            raw.len(),
            expected_samples
        )
        .into());
    }

    Ok(Array3::from_shape_vec(
        (
            candidate.height,
            candidate.width,
            candidate.source_channel_count,
        ),
        raw,
    )?)
}

fn ifd_entry(ifd: &ClassicIfd, tag: u16) -> Option<&ClassicIfdEntry> {
    ifd.entries.iter().find(|entry| entry.tag == tag)
}

fn required_ifd_value_u64(
    file: &mut File,
    byte_order: TiffByteOrder,
    ifd: &ClassicIfd,
    tag: u16,
) -> Result<u64, Box<dyn std::error::Error>> {
    ifd_value_u64(file, byte_order, ifd, tag)?
        .ok_or_else(|| format!("missing required TIFF/DNG tag {tag}").into())
}

fn ifd_value_u64(
    file: &mut File,
    byte_order: TiffByteOrder,
    ifd: &ClassicIfd,
    tag: u16,
) -> Result<Option<u64>, Box<dyn std::error::Error>> {
    let Some(values) = ifd_values_u64_opt(file, byte_order, ifd, tag)? else {
        return Ok(None);
    };
    values
        .first()
        .copied()
        .map(Some)
        .ok_or_else(|| format!("TIFF/DNG tag {tag} has no values").into())
}

fn ifd_values_u64(
    file: &mut File,
    byte_order: TiffByteOrder,
    ifd: &ClassicIfd,
    tag: u16,
) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    ifd_values_u64_opt(file, byte_order, ifd, tag)?
        .ok_or_else(|| format!("missing required TIFF/DNG tag {tag}").into())
}

fn ifd_values_u64_opt(
    file: &mut File,
    byte_order: TiffByteOrder,
    ifd: &ClassicIfd,
    tag: u16,
) -> Result<Option<Vec<u64>>, Box<dyn std::error::Error>> {
    let Some(entry) = ifd_entry(ifd, tag) else {
        return Ok(None);
    };
    let bytes = ifd_entry_bytes(file, byte_order, entry)?;
    let values = match entry.value_type {
        TIFF_TYPE_BYTE => bytes.into_iter().map(u64::from).collect(),
        TIFF_TYPE_SHORT => bytes
            .chunks_exact(2)
            .map(|chunk| byte_order.read_u16(chunk) as u64)
            .collect(),
        TIFF_TYPE_LONG | TIFF_TYPE_IFD => bytes
            .chunks_exact(4)
            .map(|chunk| byte_order.read_u32(chunk) as u64)
            .collect(),
        other => {
            return Err(format!(
                "unsupported TIFF/DNG tag {tag} value type {other} for unsigned integer decoding"
            )
            .into());
        }
    };
    Ok(Some(values))
}

fn ifd_ascii_string_opt(
    file: &mut File,
    byte_order: TiffByteOrder,
    ifd: &ClassicIfd,
    tag: u16,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let Some(entry) = ifd_entry(ifd, tag) else {
        return Ok(None);
    };
    if entry.value_type != TIFF_TYPE_ASCII && entry.value_type != TIFF_TYPE_BYTE {
        return Err(format!(
            "TIFF/DNG tag {tag} value type {} cannot be decoded as ASCII",
            entry.value_type
        )
        .into());
    }
    let bytes = ifd_entry_bytes(file, byte_order, entry)?;
    let text_bytes = bytes
        .split(|byte| *byte == 0)
        .next()
        .unwrap_or(&[])
        .iter()
        .copied()
        .filter(|byte| *byte != 0)
        .collect::<Vec<_>>();
    let text = String::from_utf8_lossy(&text_bytes).trim().to_string();
    if text.is_empty() {
        Ok(None)
    } else {
        Ok(Some(text))
    }
}

fn ifd_values_f64_opt(
    file: &mut File,
    byte_order: TiffByteOrder,
    ifd: &ClassicIfd,
    tag: u16,
) -> Result<Option<Vec<f64>>, Box<dyn std::error::Error>> {
    let Some(entry) = ifd_entry(ifd, tag) else {
        return Ok(None);
    };
    let bytes = ifd_entry_bytes(file, byte_order, entry)?;
    let values = match entry.value_type {
        TIFF_TYPE_BYTE | TIFF_TYPE_UNDEFINED => bytes.into_iter().map(f64::from).collect(),
        TIFF_TYPE_SBYTE => bytes
            .into_iter()
            .map(|value| (value as i8) as f64)
            .collect(),
        TIFF_TYPE_SHORT => bytes
            .chunks_exact(2)
            .map(|chunk| byte_order.read_u16(chunk) as f64)
            .collect(),
        TIFF_TYPE_SSHORT => bytes
            .chunks_exact(2)
            .map(|chunk| byte_order.read_i16(chunk) as f64)
            .collect(),
        TIFF_TYPE_LONG | TIFF_TYPE_IFD => bytes
            .chunks_exact(4)
            .map(|chunk| byte_order.read_u32(chunk) as f64)
            .collect(),
        TIFF_TYPE_SLONG => bytes
            .chunks_exact(4)
            .map(|chunk| byte_order.read_i32(chunk) as f64)
            .collect(),
        TIFF_TYPE_RATIONAL => bytes
            .chunks_exact(8)
            .map(|chunk| {
                let numerator = byte_order.read_u32(&chunk[0..4]) as f64;
                let denominator = byte_order.read_u32(&chunk[4..8]) as f64;
                if denominator == 0.0 {
                    f64::NAN
                } else {
                    numerator / denominator
                }
            })
            .collect(),
        TIFF_TYPE_SRATIONAL => bytes
            .chunks_exact(8)
            .map(|chunk| {
                let numerator = byte_order.read_i32(&chunk[0..4]) as f64;
                let denominator = byte_order.read_i32(&chunk[4..8]) as f64;
                if denominator == 0.0 {
                    f64::NAN
                } else {
                    numerator / denominator
                }
            })
            .collect(),
        TIFF_TYPE_FLOAT => bytes
            .chunks_exact(4)
            .map(|chunk| byte_order.read_f32(chunk) as f64)
            .collect(),
        TIFF_TYPE_DOUBLE => bytes
            .chunks_exact(8)
            .map(|chunk| byte_order.read_f64(chunk))
            .collect(),
        other => {
            return Err(format!(
                "unsupported TIFF/DNG tag {tag} value type {other} for numeric decoding"
            )
            .into());
        }
    };
    Ok(Some(values))
}

fn ifd_matrix3_f64_opt(
    file: &mut File,
    byte_order: TiffByteOrder,
    ifd: &ClassicIfd,
    tag: u16,
) -> Result<Option<[[f64; 3]; 3]>, Box<dyn std::error::Error>> {
    let Some(values) = ifd_values_f64_opt(file, byte_order, ifd, tag)? else {
        return Ok(None);
    };
    if values.len() != 9 {
        return Err(format!(
            "TIFF/DNG tag {tag} expected 9 matrix values, got {}",
            values.len()
        )
        .into());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("TIFF/DNG tag {tag} contains a non-finite matrix value").into());
    }
    Ok(Some([
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [values[6], values[7], values[8]],
    ]))
}

fn ifd_entry_bytes(
    file: &mut File,
    byte_order: TiffByteOrder,
    entry: &ClassicIfdEntry,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let type_size = match entry.value_type {
        TIFF_TYPE_BYTE | TIFF_TYPE_ASCII | TIFF_TYPE_SBYTE | TIFF_TYPE_UNDEFINED => 1,
        TIFF_TYPE_SHORT | TIFF_TYPE_SSHORT => 2,
        TIFF_TYPE_LONG | TIFF_TYPE_SLONG | TIFF_TYPE_FLOAT | TIFF_TYPE_IFD => 4,
        TIFF_TYPE_RATIONAL | TIFF_TYPE_SRATIONAL | TIFF_TYPE_DOUBLE => 8,
        other => return Err(format!("unsupported TIFF/DNG value type {other}").into()),
    };
    let byte_count = usize::try_from(u64::from(entry.count) * type_size)?;
    if byte_count <= 4 {
        return Ok(entry.value_or_offset[..byte_count].to_vec());
    }
    let offset = byte_order.read_u32(&entry.value_or_offset) as u64;
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0u8; byte_count];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn rgb_row_bytes(width: u32, bytes_per_sample: usize) -> usize {
    (width as usize)
        .saturating_mul(3)
        .saturating_mul(bytes_per_sample)
}

fn checked_rgb_dimensions(shape: &[usize]) -> Result<(u32, u32), Box<dyn std::error::Error>> {
    if shape.len() != 3 || shape[2] != 3 {
        return Err(format!("expected an HxWx3 RGB array, got shape {shape:?}").into());
    }
    if shape[0] == 0 || shape[1] == 0 {
        return Err(format!("cannot encode an empty RGB image with shape {shape:?}").into());
    }
    let height = u32::try_from(shape[0])
        .map_err(|_| format!("image height {} exceeds the TIFF/PNG u32 limit", shape[0]))?;
    let width = u32::try_from(shape[1])
        .map_err(|_| format!("image width {} exceeds the TIFF/PNG u32 limit", shape[1]))?;
    Ok((width, height))
}

fn tiff_rows_per_strip(width: u32, bytes_per_sample: usize) -> u32 {
    let row_bytes = rgb_row_bytes(width, bytes_per_sample).max(1);
    TIFF_STREAM_TARGET_BYTES
        .saturating_add(row_bytes - 1)
        .checked_div(row_bytes)
        .unwrap_or(1)
        .max(1)
        .min(u32::MAX as usize) as u32
}

fn tiff_conversion_buffer_bytes(width: u32, height: u32, bytes_per_sample: usize) -> usize {
    rgb_row_bytes(width, bytes_per_sample).saturating_mul(
        (height as usize).min(tiff_rows_per_strip(width, bytes_per_sample) as usize),
    )
}

pub fn artifact_write_buffer_diagnostics(
    width: u32,
    height: u32,
    write_master: bool,
    write_review_png: bool,
) -> ArtifactWriteBufferDiagnostics {
    let primary_tiff_conversion_buffer_bytes =
        tiff_conversion_buffer_bytes(width, height, std::mem::size_of::<u16>());
    let master_tiff_conversion_buffer_bytes = write_master
        .then(|| tiff_conversion_buffer_bytes(width, height, std::mem::size_of::<f32>()));
    let review_png_row_buffer_bytes =
        write_review_png.then(|| rgb_row_bytes(width, std::mem::size_of::<u8>()));
    let review_png_stream_chunk_bytes = write_review_png.then_some(PNG_STREAM_CHUNK_BYTES);
    let peak_declared_buffer_bytes = primary_tiff_conversion_buffer_bytes
        .max(master_tiff_conversion_buffer_bytes.unwrap_or(0))
        .max(
            review_png_row_buffer_bytes
                .unwrap_or(0)
                .saturating_add(review_png_stream_chunk_bytes.unwrap_or(0)),
        );
    ArtifactWriteBufferDiagnostics {
        strategy: "sequential_bounded_tiff_strips_and_png_rows",
        full_frame_conversion_buffers: false,
        tiff_strip_target_bytes: TIFF_STREAM_TARGET_BYTES,
        primary_tiff_conversion_buffer_bytes,
        master_tiff_conversion_buffer_bytes,
        review_png_row_buffer_bytes,
        review_png_stream_chunk_bytes,
        peak_declared_buffer_bytes,
    }
}

/// Save an Array3<u16> (height, width, 3) as a 16-bit TIFF.
pub fn save_tiff_u16(arr: &Array3<u16>, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let shape = arr.shape();
    let (width, height) = checked_rgb_dimensions(shape)?;
    write_rgb16_tiff_rows(width, height, path, None, None, |y, row| {
        for x in 0..shape[1] {
            row.push(arr[[y, x, 0]]);
            row.push(arr[[y, x, 1]]);
            row.push(arr[[y, x, 2]]);
        }
    })
}

/// Save an Array3<f64> (height, width, 3) as a 16-bit TIFF.
/// Values are clamped to [0, 1] and scaled to [0, 65535].
pub fn save_tiff_f64(arr: &Array3<f64>, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    save_tiff_f64_with_profile(arr, path, None, None)
}

pub fn save_tiff_f64_linear_prophoto(
    arr: &Array3<f64>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let icc_profile = prophoto_linear_icc_profile();
    save_tiff_f64_with_profile(
        arr,
        path,
        Some(PROPHOTO_LINEAR_ICC_DESCRIPTION),
        Some(&icc_profile),
    )
}

pub fn save_tiff_f32_linear_prophoto_scene_referred(
    arr: &Array3<f64>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let shape = arr.shape();
    let (width, height) = checked_rgb_dimensions(shape)?;

    let icc_profile = prophoto_linear_icc_profile();
    write_rgb32_float_tiff_rows(
        width,
        height,
        path,
        Some(PROPHOTO_SCENE_REFERRED_FLOAT_DESCRIPTION),
        Some(&icc_profile),
        |y, row| {
            for x in 0..shape[1] {
                for c in 0..3 {
                    let value = arr[[y, x, c]];
                    row.push(if value.is_finite() { value as f32 } else { 0.0 });
                }
            }
        },
    )
}

pub fn save_srgb_png_from_linear_prophoto(
    arr: &Array3<f64>,
    path: &Path,
) -> Result<SrgbReviewDiagnostics, Box<dyn std::error::Error>> {
    let shape = arr.shape();
    let (width, height) = checked_rgb_dimensions(shape)?;
    let mut nonfinite_input_pixel_count = 0usize;
    let mut gamut_mapped_pixel_count = 0usize;
    let mut mapped_chroma_scale_sum = 0.0f64;
    let mut mapped_min_chroma_scale = 1.0f64;
    let mut post_map_out_of_gamut_pixel_count = 0usize;

    let mut staged = AtomicFile::new(path)?;
    {
        let mut buffered = BufWriter::new(staged.file_mut());
        let mut info = png::Info::with_size(width, height);
        info.icc_profile = Some(Cow::Owned(srgb_icc_profile()?));
        let mut encoder = png::Encoder::with_info(&mut buffered, info)?;
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::High);
        encoder.set_filter(png::Filter::Adaptive);
        let mut writer = encoder.write_header()?;
        {
            let mut stream = writer.stream_writer_with_size(PNG_STREAM_CHUNK_BYTES)?;
            let mut row = Vec::with_capacity(rgb_row_bytes(width, std::mem::size_of::<u8>()));
            for y in 0..shape[0] {
                row.clear();
                for x in 0..shape[1] {
                    let mapping = perceptual_srgb_from_linear_prophoto([
                        arr[[y, x, 0]],
                        arr[[y, x, 1]],
                        arr[[y, x, 2]],
                    ]);
                    if !mapping.input_finite {
                        nonfinite_input_pixel_count += 1;
                    }
                    if mapping.mapped {
                        gamut_mapped_pixel_count += 1;
                        mapped_chroma_scale_sum += mapping.chroma_scale;
                        mapped_min_chroma_scale = mapped_min_chroma_scale.min(mapping.chroma_scale);
                    }
                    if !linear_srgb_is_in_gamut(&mapping.linear_srgb) {
                        post_map_out_of_gamut_pixel_count += 1;
                    }
                    row.push(encode_srgb_u8(mapping.linear_srgb[0]));
                    row.push(encode_srgb_u8(mapping.linear_srgb[1]));
                    row.push(encode_srgb_u8(mapping.linear_srgb[2]));
                }
                stream.write_all(&row)?;
            }
            stream.finish()?;
        }
        writer.finish()?;
        buffered.flush()?;
    }
    staged.commit()?;
    let pixel_count = shape[0].saturating_mul(shape[1]);
    Ok(SrgbReviewDiagnostics {
        width: shape[1],
        height: shape[0],
        pixel_count,
        nonfinite_input_pixel_count,
        gamut_mapped_pixel_count,
        gamut_mapped_ratio: if pixel_count == 0 {
            0.0
        } else {
            gamut_mapped_pixel_count as f64 / pixel_count as f64
        },
        mapped_mean_chroma_scale: if gamut_mapped_pixel_count == 0 {
            1.0
        } else {
            mapped_chroma_scale_sum / gamut_mapped_pixel_count as f64
        },
        mapped_min_chroma_scale: if gamut_mapped_pixel_count == 0 {
            1.0
        } else {
            mapped_min_chroma_scale
        },
        post_map_out_of_gamut_pixel_count,
    })
}

pub fn perceptual_srgb_from_linear_prophoto(rgb: [f64; 3]) -> SrgbGamutMappingResult {
    if rgb.iter().any(|value| !value.is_finite()) {
        return SrgbGamutMappingResult {
            linear_srgb: [0.0; 3],
            mapped: true,
            chroma_scale: 0.0,
            input_finite: false,
        };
    }

    let xyz_d50 = mul3x3_vec(PROPHOTO_TO_XYZ_D50, rgb);
    let direct = xyz_d50_to_linear_srgb(xyz_d50);
    if linear_srgb_is_in_gamut(&direct) {
        return SrgbGamutMappingResult {
            linear_srgb: direct,
            mapped: false,
            chroma_scale: 1.0,
            input_finite: true,
        };
    }

    let lab = crate::colorspace::xyz_d50_to_lab(xyz_d50);
    if lab.iter().all(|value| value.is_finite()) {
        let lightness = lab[0].clamp(0.0, 100.0);
        let chroma = lab[1].hypot(lab[2]);
        let candidate_for_scale = |scale: f64| {
            let candidate_lab = if chroma > 1e-12 {
                [lightness, lab[1] * scale, lab[2] * scale]
            } else {
                [lightness, 0.0, 0.0]
            };
            xyz_d50_to_linear_srgb(crate::colorspace::lab_to_xyz_d50(candidate_lab))
        };
        let neutral = candidate_for_scale(0.0);
        if linear_srgb_is_in_gamut_with_tolerance(&neutral) {
            if chroma <= 1e-12 {
                return SrgbGamutMappingResult {
                    linear_srgb: clamp_linear_srgb(neutral),
                    mapped: true,
                    chroma_scale: 1.0,
                    input_finite: true,
                };
            }
            let mut low = 0.0f64;
            let mut high = 1.0f64;
            for _ in 0..24 {
                let mid = (low + high) * 0.5;
                if linear_srgb_is_in_gamut_with_tolerance(&candidate_for_scale(mid)) {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            let chroma_scale = (low * 0.999_999).clamp(0.0, 1.0);
            return SrgbGamutMappingResult {
                linear_srgb: clamp_linear_srgb(candidate_for_scale(chroma_scale)),
                mapped: true,
                chroma_scale,
                input_finite: true,
            };
        }
    }

    SrgbGamutMappingResult {
        linear_srgb: [0.0; 3],
        mapped: true,
        chroma_scale: 0.0,
        input_finite: true,
    }
}

pub fn srgb_icc_profile() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut profile = moxcms::ColorProfile::new_srgb().encode()?;
    if profile.get(36..40) != Some(b"acsp") {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "generated sRGB ICC profile has no valid 128-byte ICC header",
        )
        .into());
    }
    let encoded_date = profile.get_mut(24..36).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "generated sRGB ICC profile is too short for its creation timestamp",
        )
    })?;
    for (destination, value) in encoded_date
        .chunks_exact_mut(2)
        .zip(CANONICAL_ICC_CREATION_DATE_TIME)
    {
        destination.copy_from_slice(&value.to_be_bytes());
    }
    Ok(profile)
}

pub fn inspect_srgb_png(
    path: &Path,
) -> Result<PngSrgbProfileInspection, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut decoder = PngDecoder::new(BufReader::new(file))?;
    let (width, height) = decoder.dimensions();
    let color_type = format!("{:?}", decoder.color_type());
    let icc_profile = decoder.icc_profile()?;
    let Some(icc_profile) = icc_profile else {
        return Ok(PngSrgbProfileInspection {
            width,
            height,
            color_type,
            icc_profile_embedded: false,
            icc_profile_size_bytes: None,
            icc_profile_valid: false,
            icc_profile_description: None,
            icc_profile_matches_standard_srgb: false,
            reason: Some("missing embedded sRGB ICC profile".to_string()),
        });
    };
    let validation_reason = icc_profile_validation_reason(&icc_profile);
    let icc_profile_valid = validation_reason.is_none();
    let icc_profile_description = icc_profile_description(&icc_profile);
    let expected_profile = srgb_icc_profile()?;
    let icc_profile_matches_standard_srgb = icc_profile_valid
        && icc_profiles_match_ignoring_creation_time(&icc_profile, &expected_profile);
    let reason = validation_reason.or_else(|| {
        (!icc_profile_matches_standard_srgb).then(|| {
            "embedded ICC profile does not match the standard ScanStitch sRGB profile".to_string()
        })
    });
    Ok(PngSrgbProfileInspection {
        width,
        height,
        color_type,
        icc_profile_embedded: true,
        icc_profile_size_bytes: Some(icc_profile.len()),
        icc_profile_valid,
        icc_profile_description,
        icc_profile_matches_standard_srgb,
        reason,
    })
}

const D50_TO_D65_BRADFORD: [[f64; 3]; 3] = [
    [0.9555766, -0.0230393, 0.0631636],
    [-0.0282895, 1.0099416, 0.0210077],
    [0.0122982, -0.0204830, 1.3299098],
];

const XYZ_D65_TO_SRGB: [[f64; 3]; 3] = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
];

fn xyz_d50_to_linear_srgb(xyz_d50: [f64; 3]) -> [f64; 3] {
    let xyz_d65 = mul3x3_vec(D50_TO_D65_BRADFORD, xyz_d50);
    mul3x3_vec(XYZ_D65_TO_SRGB, xyz_d65)
}

fn linear_srgb_is_in_gamut(rgb: &[f64; 3]) -> bool {
    rgb.iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
}

fn linear_srgb_is_in_gamut_with_tolerance(rgb: &[f64; 3]) -> bool {
    rgb.iter()
        .all(|value| value.is_finite() && *value >= -1e-10 && *value <= 1.0 + 1e-10)
}

fn clamp_linear_srgb(rgb: [f64; 3]) -> [f64; 3] {
    rgb.map(|value| value.clamp(0.0, 1.0))
}

fn mul3x3_vec(matrix: [[f64; 3]; 3], value: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * value[0] + matrix[0][1] * value[1] + matrix[0][2] * value[2],
        matrix[1][0] * value[0] + matrix[1][1] * value[1] + matrix[1][2] * value[2],
        matrix[2][0] * value[0] + matrix[2][1] * value[1] + matrix[2][2] * value[2],
    ]
}

fn encode_srgb_u8(value: f64) -> u8 {
    let value = value.clamp(0.0, 1.0);
    let encoded = if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn save_tiff_f64_with_profile(
    arr: &Array3<f64>,
    path: &Path,
    image_description: Option<&str>,
    icc_profile: Option<&[u8]>,
) -> Result<(), Box<dyn std::error::Error>> {
    let shape = arr.shape();
    let (width, height) = checked_rgb_dimensions(shape)?;
    write_rgb16_tiff_rows(
        width,
        height,
        path,
        image_description,
        icc_profile,
        |y, row| {
            for x in 0..shape[1] {
                for c in 0..3 {
                    let value = arr[[y, x, c]].clamp(0.0, 1.0);
                    row.push((value * 65535.0).round() as u16);
                }
            }
        },
    )
}

pub fn inspect_tiff_icc_profile(
    path: &Path,
) -> Result<TiffIccProfileInspection, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut decoder = decoder_for_scanner_tiff(file)?;
    let (width, height) = decoder.dimensions()?;
    let color_type = format!("{:?}", decoder.colortype()?);
    let sample_format = decoder.get_tag_u16_vec(Tag::SampleFormat).ok();
    let image_description = decoder.get_tag_ascii_string(Tag::ImageDescription).ok();
    let Ok(icc_value) = decoder.get_tag(Tag::Unknown(ICC_PROFILE_TAG)) else {
        return Ok(TiffIccProfileInspection {
            width,
            height,
            color_type,
            sample_format,
            image_description,
            icc_profile_embedded: false,
            icc_profile_size_bytes: None,
            icc_profile_valid: false,
            icc_profile_description: None,
            reason: Some(format!("missing ICC profile TIFF tag {ICC_PROFILE_TAG}")),
        });
    };
    let icc = match tiff_value_into_bytes(icc_value) {
        Ok(values) => values,
        Err(err) => {
            return Ok(TiffIccProfileInspection {
                width,
                height,
                color_type,
                sample_format,
                image_description,
                icc_profile_embedded: true,
                icc_profile_size_bytes: None,
                icc_profile_valid: false,
                icc_profile_description: None,
                reason: Some(format!("failed to decode ICC profile tag: {err}")),
            });
        }
    };
    let reason = icc_profile_validation_reason(&icc);
    let icc_profile_valid = reason.is_none();
    let icc_profile_description = icc_profile_description(&icc);
    Ok(TiffIccProfileInspection {
        width,
        height,
        color_type,
        sample_format,
        image_description,
        icc_profile_embedded: true,
        icc_profile_size_bytes: Some(icc.len()),
        icc_profile_valid,
        icc_profile_description,
        reason,
    })
}

fn write_rgb16_tiff_rows<F>(
    width: u32,
    height: u32,
    path: &Path,
    image_description: Option<&str>,
    icc_profile: Option<&[u8]>,
    mut append_row: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(usize, &mut Vec<u16>),
{
    let mut staged = AtomicFile::new(path)?;
    {
        let mut buffered = BufWriter::new(staged.file_mut());
        {
            let mut encoder = tiff::encoder::TiffEncoder::new(&mut buffered)?;
            let mut image = encoder.new_image::<tiff::encoder::colortype::RGB16>(width, height)?;
            image.rows_per_strip(tiff_rows_per_strip(width, std::mem::size_of::<u16>()))?;
            if let Some(description) = image_description {
                image
                    .encoder()
                    .write_tag(Tag::ImageDescription, description)?;
            }
            if let Some(profile) = icc_profile {
                image
                    .encoder()
                    .write_tag(Tag::Unknown(ICC_PROFILE_TAG), profile)?;
            }
            let row_samples = (width as usize)
                .checked_mul(3)
                .ok_or("RGB16 TIFF row sample count overflow")?;
            let mut next_y = 0usize;
            let mut strip = Vec::with_capacity(
                usize::try_from(image.next_strip_sample_count())
                    .map_err(|_| "RGB16 TIFF strip sample count exceeds usize")?,
            );
            while image.next_strip_sample_count() > 0 {
                let sample_count = usize::try_from(image.next_strip_sample_count())
                    .map_err(|_| "RGB16 TIFF strip sample count exceeds usize")?;
                let rows = sample_count
                    .checked_div(row_samples)
                    .ok_or("RGB16 TIFF row sample count is zero")?;
                strip.clear();
                for y in next_y..next_y.saturating_add(rows) {
                    append_row(y, &mut strip);
                }
                if strip.len() != sample_count {
                    return Err(format!(
                        "RGB16 TIFF row encoder produced {} samples for a {}-sample strip",
                        strip.len(),
                        sample_count
                    )
                    .into());
                }
                image.write_strip(&strip)?;
                next_y = next_y.saturating_add(rows);
            }
            image.finish()?;
        }
        buffered.flush()?;
    }
    staged.commit()?;
    Ok(())
}

fn write_rgb32_float_tiff_rows<F>(
    width: u32,
    height: u32,
    path: &Path,
    image_description: Option<&str>,
    icc_profile: Option<&[u8]>,
    mut append_row: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(usize, &mut Vec<f32>),
{
    let mut staged = AtomicFile::new(path)?;
    {
        let mut buffered = BufWriter::new(staged.file_mut());
        {
            let mut encoder = tiff::encoder::TiffEncoder::new(&mut buffered)?;
            let mut image =
                encoder.new_image::<tiff::encoder::colortype::RGB32Float>(width, height)?;
            image.rows_per_strip(tiff_rows_per_strip(width, std::mem::size_of::<f32>()))?;
            if let Some(description) = image_description {
                image
                    .encoder()
                    .write_tag(Tag::ImageDescription, description)?;
            }
            if let Some(profile) = icc_profile {
                image
                    .encoder()
                    .write_tag(Tag::Unknown(ICC_PROFILE_TAG), profile)?;
            }
            let row_samples = (width as usize)
                .checked_mul(3)
                .ok_or("RGB32 TIFF row sample count overflow")?;
            let mut next_y = 0usize;
            let mut strip = Vec::with_capacity(
                usize::try_from(image.next_strip_sample_count())
                    .map_err(|_| "RGB32 TIFF strip sample count exceeds usize")?,
            );
            while image.next_strip_sample_count() > 0 {
                let sample_count = usize::try_from(image.next_strip_sample_count())
                    .map_err(|_| "RGB32 TIFF strip sample count exceeds usize")?;
                let rows = sample_count
                    .checked_div(row_samples)
                    .ok_or("RGB32 TIFF row sample count is zero")?;
                strip.clear();
                for y in next_y..next_y.saturating_add(rows) {
                    append_row(y, &mut strip);
                }
                if strip.len() != sample_count {
                    return Err(format!(
                        "RGB32 TIFF row encoder produced {} samples for a {}-sample strip",
                        strip.len(),
                        sample_count
                    )
                    .into());
                }
                image.write_strip(&strip)?;
                next_y = next_y.saturating_add(rows);
            }
            image.finish()?;
        }
        buffered.flush()?;
    }
    staged.commit()?;
    Ok(())
}

pub fn prophoto_linear_icc_profile() -> Vec<u8> {
    let tags = [
        (*b"desc", icc_desc_tag(PROPHOTO_LINEAR_ICC_DESCRIPTION)),
        (*b"wtpt", icc_xyz_tag(D50_WHITE)),
        (
            *b"rXYZ",
            icc_xyz_tag([
                PROPHOTO_TO_XYZ_D50[0][0],
                PROPHOTO_TO_XYZ_D50[1][0],
                PROPHOTO_TO_XYZ_D50[2][0],
            ]),
        ),
        (
            *b"gXYZ",
            icc_xyz_tag([
                PROPHOTO_TO_XYZ_D50[0][1],
                PROPHOTO_TO_XYZ_D50[1][1],
                PROPHOTO_TO_XYZ_D50[2][1],
            ]),
        ),
        (
            *b"bXYZ",
            icc_xyz_tag([
                PROPHOTO_TO_XYZ_D50[0][2],
                PROPHOTO_TO_XYZ_D50[1][2],
                PROPHOTO_TO_XYZ_D50[2][2],
            ]),
        ),
        (*b"rTRC", icc_linear_curve_tag()),
        (*b"gTRC", icc_linear_curve_tag()),
        (*b"bTRC", icc_linear_curve_tag()),
    ];
    build_icc_profile(&tags)
}

fn build_icc_profile(tags: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
    let tag_table_len = 4 + tags.len() * 12;
    let mut offset = 128 + tag_table_len;
    let mut entries = Vec::with_capacity(tags.len());
    for (signature, data) in tags {
        entries.push((*signature, offset as u32, data.len() as u32));
        offset += padded_len(data.len());
    }

    let mut profile = Vec::with_capacity(offset);
    push_u32_be(&mut profile, offset as u32);
    profile.extend_from_slice(&[0; 4]);
    profile.extend_from_slice(&[0x02, 0x10, 0x00, 0x00]);
    profile.extend_from_slice(b"mntr");
    profile.extend_from_slice(b"RGB ");
    profile.extend_from_slice(b"XYZ ");
    for value in CANONICAL_ICC_CREATION_DATE_TIME {
        push_u16_be(&mut profile, value);
    }
    profile.extend_from_slice(b"acsp");
    profile.extend_from_slice(&[0; 4]);
    profile.extend_from_slice(&[0; 4]);
    profile.extend_from_slice(b"SCST");
    profile.extend_from_slice(b"PRLP");
    profile.extend_from_slice(&[0; 8]);
    profile.extend_from_slice(&[0; 4]);
    push_s15_fixed16(&mut profile, D50_WHITE[0]);
    push_s15_fixed16(&mut profile, D50_WHITE[1]);
    push_s15_fixed16(&mut profile, D50_WHITE[2]);
    profile.extend_from_slice(b"SCST");
    profile.extend_from_slice(&[0; 44]);
    debug_assert_eq!(profile.len(), 128);

    push_u32_be(&mut profile, tags.len() as u32);
    for (signature, offset, size) in &entries {
        profile.extend_from_slice(signature);
        push_u32_be(&mut profile, *offset);
        push_u32_be(&mut profile, *size);
    }

    for (_, data) in tags {
        profile.extend_from_slice(data);
        pad_to_4(&mut profile);
    }
    profile
}

fn icc_desc_tag(description: &str) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"desc");
    out.extend_from_slice(&[0; 4]);
    push_u32_be(&mut out, description.len() as u32 + 1);
    out.extend_from_slice(description.as_bytes());
    out.push(0);
    out
}

fn icc_xyz_tag(xyz: [f64; 3]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"XYZ ");
    out.extend_from_slice(&[0; 4]);
    for value in xyz {
        push_s15_fixed16(&mut out, value);
    }
    out
}

fn icc_linear_curve_tag() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"curv");
    out.extend_from_slice(&[0; 4]);
    push_u32_be(&mut out, 0);
    out
}

fn padded_len(len: usize) -> usize {
    (len + 3) & !3
}

fn pad_to_4(out: &mut Vec<u8>) {
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
}

fn icc_profile_validation_reason(profile: &[u8]) -> Option<String> {
    if profile.len() < 132 {
        return Some(format!(
            "ICC profile is too short for header and tag table: {} bytes",
            profile.len()
        ));
    }
    if &profile[36..40] != b"acsp" {
        return Some("ICC profile header signature is not `acsp`".to_string());
    }
    let Some(tag_count) = read_u32_be(profile, 128).map(|count| count as usize) else {
        return Some("ICC profile tag count is missing".to_string());
    };
    let Some(tag_table_len) = tag_count.checked_mul(12) else {
        return Some("ICC profile tag table length overflowed".to_string());
    };
    let Some(table_end) = 132usize.checked_add(tag_table_len) else {
        return Some("ICC profile tag table length overflowed".to_string());
    };
    if table_end > profile.len() {
        return Some(format!(
            "ICC profile tag table exceeds profile length: table_end={table_end} len={}",
            profile.len()
        ));
    }
    None
}

fn icc_profile_description(profile: &[u8]) -> Option<String> {
    if icc_profile_validation_reason(profile).is_some() {
        return None;
    }
    let tag_count = read_u32_be(profile, 128)? as usize;
    for idx in 0..tag_count {
        let entry_offset = 132 + idx * 12;
        if &profile[entry_offset..entry_offset + 4] != b"desc" {
            continue;
        }
        let tag_offset = read_u32_be(profile, entry_offset + 4)? as usize;
        let tag_size = read_u32_be(profile, entry_offset + 8)? as usize;
        let tag_end = tag_offset.checked_add(tag_size)?;
        if tag_end > profile.len() || tag_size < 12 {
            return None;
        }
        let tag = &profile[tag_offset..tag_end];
        if &tag[0..4] == b"desc" {
            let text_len = read_u32_be(tag, 8)? as usize;
            let text_start = 12usize;
            let text_end = text_start.checked_add(text_len)?.min(tag.len());
            let mut text = &tag[text_start..text_end];
            while text.last() == Some(&0) {
                text = &text[..text.len() - 1];
            }
            return String::from_utf8(text.to_vec()).ok();
        }
        if &tag[0..4] == b"mluc" {
            return icc_mluc_first_description(tag);
        }
        return None;
    }
    None
}

fn icc_mluc_first_description(tag: &[u8]) -> Option<String> {
    let record_count = read_u32_be(tag, 8)? as usize;
    let record_size = read_u32_be(tag, 12)? as usize;
    if record_count == 0 || record_size < 12 {
        return None;
    }
    let records_end = 16usize.checked_add(record_count.checked_mul(record_size)?)?;
    if records_end > tag.len() {
        return None;
    }

    let mut fallback = None;
    for index in 0..record_count {
        let record = 16 + index * record_size;
        let text_len = read_u32_be(tag, record + 4)? as usize;
        let text_offset = read_u32_be(tag, record + 8)? as usize;
        let text_end = text_offset.checked_add(text_len)?;
        if text_end > tag.len() || !text_len.is_multiple_of(2) {
            continue;
        }
        let units = tag[text_offset..text_end]
            .chunks_exact(2)
            .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        let Ok(text) = String::from_utf16(&units) else {
            continue;
        };
        let language = tag.get(record..record + 2);
        let country = tag.get(record + 2..record + 4);
        if language == Some(b"en") && country == Some(b"US") {
            return Some(text);
        }
        fallback.get_or_insert(text);
    }
    fallback
}

fn icc_profiles_match_ignoring_creation_time(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .enumerate()
            .all(|(index, (a, b))| (24..36).contains(&index) || a == b)
}

fn read_u32_be(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let slice = bytes.get(offset..end)?;
    Some(u32::from_be_bytes(slice.try_into().ok()?))
}

fn push_u16_be(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_u32_be(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_s15_fixed16(out: &mut Vec<u8>, value: f64) {
    let fixed = (value * 65_536.0).round() as i32;
    out.extend_from_slice(&fixed.to_be_bytes());
}

/// Helper: convert Array3<u16> to image::DynamicImage (8-bit, for debug/preview).
#[allow(dead_code)]
pub fn array_to_dynamic_image_8bit(arr: &Array3<u16>) -> DynamicImage {
    let shape = arr.shape();
    let height = shape[0] as u32;
    let width = shape[1] as u32;

    let mut img = RgbImage::new(width, height);
    for y in 0..shape[0] {
        for x in 0..shape[1] {
            let r = (arr[[y, x, 0]] >> 8) as u8;
            let g = (arr[[y, x, 1]] >> 8) as u8;
            let b = (arr[[y, x, 2]] >> 8) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }
    DynamicImage::ImageRgb8(img)
}

#[cfg(test)]
mod atomic_writer_tests {
    use super::*;

    #[test]
    fn failed_tiff_strip_encode_preserves_existing_destination() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output.tiff");
        let original = Array3::<u16>::from_shape_fn((2, 2, 3), |(y, x, c)| {
            (y * 1000 + x * 100 + c * 10) as u16
        });
        save_tiff_u16(&original, &destination).unwrap();
        let original_bytes = std::fs::read(&destination).unwrap();

        let error = write_rgb16_tiff_rows(2, 2, &destination, None, None, |_y, row| {
            row.push(1);
        })
        .unwrap_err();

        assert!(error.to_string().contains("row encoder produced"));
        assert_eq!(std::fs::read(&destination).unwrap(), original_bytes);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }
}

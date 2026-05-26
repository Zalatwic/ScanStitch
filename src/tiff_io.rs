use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::constants::{D50_WHITE, MAX_14BIT, MAX_16BIT, PROPHOTO_TO_XYZ_D50};
use image::{DynamicImage, Rgb, RgbImage};
use ndarray::Array3;
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::tags::Tag;
use tiff::ColorType;

pub const PROPHOTO_LINEAR_ICC_DESCRIPTION: &str = "ScanStitch linear ProPhoto RGB D50";
pub const PROPHOTO_SCENE_REFERRED_FLOAT_DESCRIPTION: &str =
    "ScanStitch scene-referred linear ProPhoto RGB D50 32-bit float";
pub const ICC_PROFILE_TAG: u16 = 34_675;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TiffIccProfileInspection {
    pub image_description: Option<String>,
    pub icc_profile_embedded: bool,
    pub icc_profile_size_bytes: Option<usize>,
    pub icc_profile_valid: bool,
    pub icc_profile_description: Option<String>,
    pub reason: Option<String>,
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
    pub dng_metadata: Option<DngMetadata>,
}

pub struct LoadedTiff {
    pub image: Array3<u16>,
    pub diagnostics: TiffLoadDiagnostics,
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

    let (image, working_range) =
        normalize_to_working_bit_depth(&source, source_bits_per_sample, target_bit_depth);

    Ok(LoadedTiff {
        image,
        diagnostics: TiffLoadDiagnostics {
            width: width as usize,
            height: height as usize,
            color_type: color_label,
            source_bits_per_sample,
            source_channel_count,
            source_has_alpha,
            working_range,
            dng_metadata: None,
        },
    })
}

fn load_dng_linear_raw_u16(
    path: &Path,
    target_bit_depth: u8,
) -> Result<LoadedTiff, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let (byte_order, first_ifd_offset) = read_classic_tiff_header(&mut file)?;
    let first_ifd = read_classic_ifd(&mut file, byte_order, first_ifd_offset)?;
    let dng_metadata = parse_dng_metadata(&mut file, byte_order, &first_ifd);
    let candidate = select_dng_linear_raw_candidate(&mut file, byte_order, first_ifd_offset)?;
    let source = read_dng_linear_raw_pixels(&mut file, byte_order, &candidate)?;
    let (image, working_range) =
        normalize_to_working_bit_depth(&source, candidate.source_bits_per_sample, target_bit_depth);

    Ok(LoadedTiff {
        image,
        diagnostics: TiffLoadDiagnostics {
            width: candidate.width,
            height: candidate.height,
            color_type: candidate.color_type,
            source_bits_per_sample: candidate.source_bits_per_sample,
            source_channel_count: candidate.source_channel_count,
            source_has_alpha: candidate.source_has_alpha,
            working_range,
            dng_metadata,
        },
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
        parse_warnings: warnings,
    };
    metadata.has_reportable_metadata().then_some(metadata)
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
        if let Some(candidate) = dng_candidate_from_ifd(file, byte_order, &ifd)? {
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

/// Save an Array3<u16> (height, width, 3) as a 16-bit TIFF.
pub fn save_tiff_u16(arr: &Array3<u16>, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let shape = arr.shape();
    let height = shape[0] as u32;
    let width = shape[1] as u32;

    // Build contiguous pixel buffer
    let mut buf = Vec::with_capacity((height * width * 3) as usize);
    for y in 0..shape[0] {
        for x in 0..shape[1] {
            buf.push(arr[[y, x, 0]]);
            buf.push(arr[[y, x, 1]]);
            buf.push(arr[[y, x, 2]]);
        }
    }

    write_rgb16_tiff(&buf, width, height, path, None, None)
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
    let height = shape[0] as u32;
    let width = shape[1] as u32;

    let mut buf = Vec::with_capacity((height * width * 3) as usize);
    for y in 0..shape[0] {
        for x in 0..shape[1] {
            for c in 0..3 {
                let value = arr[[y, x, c]];
                if value.is_finite() {
                    buf.push(value as f32);
                } else {
                    buf.push(0.0);
                }
            }
        }
    }

    let icc_profile = prophoto_linear_icc_profile();
    write_rgb32_float_tiff(
        &buf,
        width,
        height,
        path,
        Some(PROPHOTO_SCENE_REFERRED_FLOAT_DESCRIPTION),
        Some(&icc_profile),
    )
}

pub fn save_srgb_png_from_linear_prophoto(
    arr: &Array3<f64>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let shape = arr.shape();
    let height = shape[0] as u32;
    let width = shape[1] as u32;
    let mut img = RgbImage::new(width, height);

    for y in 0..shape[0] {
        for x in 0..shape[1] {
            let rgb = [
                arr[[y, x, 0]].clamp(0.0, 1.0),
                arr[[y, x, 1]].clamp(0.0, 1.0),
                arr[[y, x, 2]].clamp(0.0, 1.0),
            ];
            let xyz_d50 = mul3x3_vec(PROPHOTO_TO_XYZ_D50, rgb);
            let xyz_d65 = mul3x3_vec(D50_TO_D65_BRADFORD, xyz_d50);
            let linear_srgb = mul3x3_vec(XYZ_D65_TO_SRGB, xyz_d65);
            img.put_pixel(
                x as u32,
                y as u32,
                Rgb([
                    encode_srgb_u8(linear_srgb[0]),
                    encode_srgb_u8(linear_srgb[1]),
                    encode_srgb_u8(linear_srgb[2]),
                ]),
            );
        }
    }

    img.save(path)?;
    Ok(())
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
    let height = shape[0] as u32;
    let width = shape[1] as u32;

    let mut buf = Vec::with_capacity((height * width * 3) as usize);
    for y in 0..shape[0] {
        for x in 0..shape[1] {
            for c in 0..3 {
                let v = arr[[y, x, c]].clamp(0.0, 1.0);
                buf.push((v * 65535.0).round() as u16);
            }
        }
    }

    write_rgb16_tiff(&buf, width, height, path, image_description, icc_profile)
}

pub fn inspect_tiff_icc_profile(
    path: &Path,
) -> Result<TiffIccProfileInspection, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut decoder = decoder_for_scanner_tiff(file)?;
    let image_description = decoder.get_tag_ascii_string(Tag::ImageDescription).ok();
    let Ok(icc_value) = decoder.get_tag(Tag::Unknown(ICC_PROFILE_TAG)) else {
        return Ok(TiffIccProfileInspection {
            image_description,
            icc_profile_embedded: false,
            icc_profile_size_bytes: None,
            icc_profile_valid: false,
            icc_profile_description: None,
            reason: Some(format!("missing ICC profile TIFF tag {ICC_PROFILE_TAG}")),
        });
    };
    let icc_values = match icc_value.into_u32_vec() {
        Ok(values) => values,
        Err(err) => {
            return Ok(TiffIccProfileInspection {
                image_description,
                icc_profile_embedded: true,
                icc_profile_size_bytes: None,
                icc_profile_valid: false,
                icc_profile_description: None,
                reason: Some(format!("failed to decode ICC profile tag: {err}")),
            });
        }
    };
    let mut icc = Vec::with_capacity(icc_values.len());
    for value in icc_values {
        let Ok(byte) = u8::try_from(value) else {
            return Ok(TiffIccProfileInspection {
                image_description,
                icc_profile_embedded: true,
                icc_profile_size_bytes: Some(icc.len()),
                icc_profile_valid: false,
                icc_profile_description: None,
                reason: Some(format!("ICC profile tag contains non-byte value {value}")),
            });
        };
        icc.push(byte);
    }
    let reason = icc_profile_validation_reason(&icc);
    let icc_profile_valid = reason.is_none();
    let icc_profile_description = icc_profile_description(&icc);
    Ok(TiffIccProfileInspection {
        image_description,
        icc_profile_embedded: true,
        icc_profile_size_bytes: Some(icc.len()),
        icc_profile_valid,
        icc_profile_description,
        reason,
    })
}

fn write_rgb16_tiff(
    buf: &[u16],
    width: u32,
    height: u32,
    path: &Path,
    image_description: Option<&str>,
    icc_profile: Option<&[u8]>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::create(path)?;
    let mut encoder = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file))?;
    let mut image = encoder.new_image::<tiff::encoder::colortype::RGB16>(width, height)?;
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
    image.write_data(buf)?;
    Ok(())
}

fn write_rgb32_float_tiff(
    buf: &[f32],
    width: u32,
    height: u32,
    path: &Path,
    image_description: Option<&str>,
    icc_profile: Option<&[u8]>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::create(path)?;
    let mut encoder = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file))?;
    let mut image = encoder.new_image::<tiff::encoder::colortype::RGB32Float>(width, height)?;
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
    image.write_data(buf)?;
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
    for value in [2026u16, 5, 8, 0, 0, 0] {
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
        if &tag[0..4] != b"desc" {
            return None;
        }
        let text_len = read_u32_be(tag, 8)? as usize;
        let text_start = 12usize;
        let text_end = text_start.checked_add(text_len)?.min(tag.len());
        let mut text = &tag[text_start..text_end];
        while text.last() == Some(&0) {
            text = &text[..text.len() - 1];
        }
        return String::from_utf8(text.to_vec()).ok();
    }
    None
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

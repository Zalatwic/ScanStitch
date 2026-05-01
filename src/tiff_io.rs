use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::constants::{MAX_14BIT, MAX_16BIT};
use image::{DynamicImage, RgbImage};
use ndarray::Array3;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::ColorType;

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
}

pub struct LoadedTiff {
    pub image: Array3<u16>,
    pub diagnostics: TiffLoadDiagnostics,
}

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

/// Load a TIFF image into the requested working bit depth, preserving actual
/// on-disk sample precision instead of silently expanding 8-bit data to 16-bit.
pub fn load_tiff_u16(
    path: &Path,
    target_bit_depth: u8,
) -> Result<LoadedTiff, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(BufReader::new(file))?;
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
        },
    })
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

    // Use the tiff encoder directly for 16-bit output
    let file = std::fs::File::create(path)?;
    let mut encoder = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file))?;
    encoder.write_image::<tiff::encoder::colortype::RGB16>(width, height, &buf)?;

    Ok(())
}

/// Save an Array3<f64> (height, width, 3) as a 16-bit TIFF.
/// Values are clamped to [0, 1] and scaled to [0, 65535].
pub fn save_tiff_f64(arr: &Array3<f64>, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
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

    let file = std::fs::File::create(path)?;
    let mut encoder = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file))?;
    encoder.write_image::<tiff::encoder::colortype::RGB16>(width, height, &buf)?;

    Ok(())
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

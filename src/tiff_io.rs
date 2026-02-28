use std::path::Path;

use image::{DynamicImage, ImageReader, RgbImage};
use ndarray::Array3;

/// Load a 16-bit TIFF image into an Array3<u16> with shape (height, width, 3).
pub fn load_tiff_u16(path: &Path) -> Result<Array3<u16>, Box<dyn std::error::Error>> {
    let img = ImageReader::open(path)?.decode()?;

    let (width, height) = (img.width() as usize, img.height() as usize);

    let rgb16 = img.into_rgb16();
    let raw = rgb16.into_raw();

    // raw is row-major: [R, G, B, R, G, B, ...] for each pixel, row by row
    let arr = Array3::from_shape_vec((height, width, 3), raw)?;
    Ok(arr)
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

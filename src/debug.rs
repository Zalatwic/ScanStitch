use ndarray::Array3;
use std::path::Path;

use crate::tiff_io;

/// Save an intermediate debug image if debug mode is enabled.
/// No-op when debug_enabled is false.
pub fn save_debug_image(
    arr: &Array3<u16>,
    dir: &Path,
    name: &str,
    debug_enabled: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !debug_enabled {
        return Ok(());
    }

    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}.tif", name));
    tiff_io::save_tiff_u16(arr, &path)?;
    log::debug!("Saved debug image: {}", path.display());

    Ok(())
}

/// Save a floating-point debug image if debug mode is enabled.
/// No-op when debug_enabled is false.
#[allow(dead_code)]
pub fn save_debug_image_f64(
    arr: &Array3<f64>,
    dir: &Path,
    name: &str,
    debug_enabled: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !debug_enabled {
        return Ok(());
    }

    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}.tif", name));
    tiff_io::save_tiff_f64(arr, &path)?;
    log::debug!("Saved debug image (f64): {}", path.display());

    Ok(())
}

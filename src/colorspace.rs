use nalgebra::Matrix3;
use ndarray::Array3;

use crate::constants::{
    BRADFORD_LMS_TO_XYZ, BRADFORD_XYZ_TO_LMS, D50_WHITE, PROPHOTO_TO_XYZ_D50,
    XYZ_D50_TO_PROPHOTO,
};
use crate::streaming;

/// Helper to convert a `[[f64; 3]; 3]` constant into a `nalgebra::Matrix3<f64>`.
/// nalgebra stores columns, so we transpose the row-major constant.
fn const_to_matrix3(rows: &[[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::new(
        rows[0][0], rows[0][1], rows[0][2],
        rows[1][0], rows[1][1], rows[1][2],
        rows[2][0], rows[2][1], rows[2][2],
    )
}

/// ProPhoto RGB -> XYZ (D50) conversion matrix.
pub fn prophoto_to_xyz_d50_matrix() -> Matrix3<f64> {
    const_to_matrix3(&PROPHOTO_TO_XYZ_D50)
}

/// XYZ (D50) -> ProPhoto RGB conversion matrix.
pub fn xyz_d50_to_prophoto_matrix() -> Matrix3<f64> {
    const_to_matrix3(&XYZ_D50_TO_PROPHOTO)
}

/// Compute a Bradford chromatic adaptation transform from `source_white` to D50.
///
/// Steps:
/// 1. Convert source and D50 white points to LMS via the Bradford matrix.
/// 2. Build a diagonal scaling matrix: `scale[i] = lms_d50[i] / lms_source[i]`.
/// 3. Return `LMS_TO_XYZ * diag_scale * XYZ_TO_LMS`.
pub fn bradford_cat(source_white: &[f64; 3]) -> Matrix3<f64> {
    let m_to_lms = const_to_matrix3(&BRADFORD_XYZ_TO_LMS);
    let m_to_xyz = const_to_matrix3(&BRADFORD_LMS_TO_XYZ);

    let src = nalgebra::Vector3::new(source_white[0], source_white[1], source_white[2]);
    let d50 = nalgebra::Vector3::new(D50_WHITE[0], D50_WHITE[1], D50_WHITE[2]);

    let lms_src = m_to_lms * src;
    let lms_d50 = m_to_lms * d50;

    let diag = Matrix3::new(
        lms_d50[0] / lms_src[0], 0.0, 0.0,
        0.0, lms_d50[1] / lms_src[1], 0.0,
        0.0, 0.0, lms_d50[2] / lms_src[2],
    );

    m_to_xyz * diag * m_to_lms
}

/// Maximum channel deviation relative to pixel mean for a pixel to be
/// considered "neutral". Pixels whose `max(|ch - mu|) / mu` exceeds this
/// threshold are excluded from the gray-world average.
const NEUTRAL_THRESHOLD: f64 = 0.15;

/// Minimum fraction of total pixels that must pass the neutrality test.
/// If fewer than this fraction are neutral, we fall back to [1, 1, 1].
const MIN_NEUTRAL_FRACTION: f64 = 0.005;

/// Selective Gray World: average only near-neutral pixels to estimate the
/// scene illuminant. Falls back to `[1.0, 1.0, 1.0]` when the image lacks
/// enough neutral data (e.g. a solid-blue sky).
fn estimate_neutral_means(img: &Array3<f64>) -> [f64; 3] {
    let (h, w, _) = img.dim();
    let total = (h * w) as f64;
    let mut sums = [0.0f64; 3];
    let mut count = 0u64;

    for y in 0..h {
        for x in 0..w {
            let r = img[[y, x, 0]];
            let g = img[[y, x, 1]];
            let b = img[[y, x, 2]];

            let mu = (r + g + b) / 3.0;
            if mu < 1e-12 {
                continue; // skip near-black pixels
            }

            let max_dev = (r - mu).abs().max((g - mu).abs()).max((b - mu).abs());
            if max_dev / mu < NEUTRAL_THRESHOLD {
                sums[0] += r;
                sums[1] += g;
                sums[2] += b;
                count += 1;
            }
        }
    }

    if (count as f64) < total * MIN_NEUTRAL_FRACTION {
        return [1.0, 1.0, 1.0];
    }

    let n = count as f64;
    [sums[0] / n, sums[1] / n, sums[2] / n]
}

/// Estimate source white point using selective gray-world assumption.
/// Returns `[mean_r/max_mean, 1.0, mean_b/max_mean]`.
fn estimate_source_white(img: &Array3<f64>) -> [f64; 3] {
    let means = estimate_neutral_means(img);
    let max_mean = means[0].max(means[1]).max(means[2]).max(1e-12);
    [means[0] / max_mean, 1.0, means[2] / max_mean]
}

/// Estimate a diagonal work-RGB-to-XYZ matrix using selective gray-world.
/// `scale[c] = D50_WHITE[c] / (mean[c] / max_mean)`
fn estimate_work_to_xyz(img: &Array3<f64>) -> Matrix3<f64> {
    let means = estimate_neutral_means(img);
    let max_mean = means[0].max(means[1]).max(means[2]).max(1e-12);

    Matrix3::new(
        D50_WHITE[0] / (means[0] / max_mean), 0.0, 0.0,
        0.0, D50_WHITE[1] / (means[1] / max_mean), 0.0,
        0.0, 0.0, D50_WHITE[2] / (means[2] / max_mean),
    )
}

/// Map an image from work RGB to ProPhoto RGB (D50) via:
/// 1. Gray-world diagonal work->XYZ estimate
/// 2. Bradford chromatic adaptation to D50
/// 3. XYZ (D50) -> ProPhoto RGB
///
/// Pixels are clamped to `[0, 1]`. Iteration uses `streaming::tile_ranges`.
pub fn map_to_prophoto_d50(img: &Array3<f64>) -> Array3<f64> {
    let (h, w, _c) = img.dim();

    let work_to_xyz = estimate_work_to_xyz(img);
    let source_white = estimate_source_white(img);
    let cat = bradford_cat(&source_white);
    let xyz_to_pro = xyz_d50_to_prophoto_matrix();

    let combined = xyz_to_pro * cat * work_to_xyz;

    let mut out = Array3::<f64>::zeros((h, w, 3));

    for (row_start, row_end) in streaming::tile_ranges(h, streaming::DEFAULT_TILE_ROWS) {
        for y in row_start..row_end {
            for x in 0..w {
                let r = img[[y, x, 0]];
                let g = img[[y, x, 1]];
                let b = img[[y, x, 2]];

                let pixel = nalgebra::Vector3::new(r, g, b);
                let mapped = combined * pixel;

                out[[y, x, 0]] = mapped[0].clamp(0.0, 1.0);
                out[[y, x, 1]] = mapped[1].clamp(0.0, 1.0);
                out[[y, x, 2]] = mapped[2].clamp(0.0, 1.0);
            }
        }
    }

    out
}

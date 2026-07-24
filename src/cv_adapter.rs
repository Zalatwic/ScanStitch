use nalgebra::{DMatrix, Matrix3, SymmetricEigen, Vector3};
use std::collections::BTreeSet;

const HOMOGRAPHY_FEATURE_GRID_ROWS: usize = 4;
const HOMOGRAPHY_FEATURE_GRID_COLS: usize = 4;
const HOMOGRAPHY_MIN_MATCHES_PER_SPLIT: usize = 10;
const HOMOGRAPHY_MIN_CELLS_PER_SPLIT: usize = 3;
const HOMOGRAPHY_RANSAC_THRESHOLD_PX: f64 = 3.0;
const HOMOGRAPHY_MIN_CROSS_INLIER_RATIO: f64 = 0.60;
const HOMOGRAPHY_MAX_CROSS_MEDIAN_ERROR_PX: f64 = 2.5;
const HOMOGRAPHY_MAX_CROSS_P95_ERROR_PX: f64 = 2.75;
const HOMOGRAPHY_MAX_CROSS_FIT_DISAGREEMENT_PX: f64 = 3.0;

#[derive(Debug, Clone, Copy)]
pub struct FeatureCorrespondence {
    pub source: [f64; 2],
    pub destination: [f64; 2],
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub match_count: usize,
    pub training_match_count: usize,
    pub held_out_match_count: usize,
    pub training_spatial_cell_count: usize,
    pub held_out_spatial_cell_count: usize,
    pub inliers: usize,
    pub training_inlier_ratio: f64,
    pub training_median_error: f64,
    pub training_p95_error: f64,
    /// Median primary training-fit residual across every spatially disjoint held-out match.
    pub median_error: f64,
    /// P95 primary training-fit residual within the held-out 3px consensus set.
    pub p95_error: f64,
    pub held_out_inlier_count: usize,
    pub held_out_inlier_ratio: f64,
    pub reverse_validation_inlier_count: usize,
    pub reverse_validation_inlier_ratio: f64,
    pub reverse_validation_median_error: f64,
    pub reverse_validation_p95_error: f64,
    pub cross_fit_max_disagreement_px: f64,
    pub disjoint_validation_passed: bool,
    pub validation_reason: String,
    pub partition_method: &'static str,
    pub transform: [[f64; 3]; 3],
}

#[derive(Debug)]
struct RobustHomographyFit {
    transform: [[f64; 3]; 3],
    inlier_count: usize,
    inlier_ratio: f64,
    median_inlier_error: f64,
    p95_inlier_error: f64,
}

fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 1.0e9;
    }
    let position = percentile.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let fraction = position - lower as f64;
        values[lower] * (1.0 - fraction) + values[upper] * fraction
    }
}

fn error_percentiles(mut errors: Vec<f64>) -> (f64, f64) {
    errors.retain(|value| value.is_finite());
    errors.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    (
        percentile_sorted(&errors, 0.5),
        percentile_sorted(&errors, 0.95),
    )
}

fn project_point(transform: &[[f64; 3]; 3], point: [f64; 2]) -> Option<[f64; 2]> {
    let denominator = transform[2][0] * point[0] + transform[2][1] * point[1] + transform[2][2];
    if !denominator.is_finite() || denominator.abs() < 1e-10 {
        return None;
    }
    let x =
        (transform[0][0] * point[0] + transform[0][1] * point[1] + transform[0][2]) / denominator;
    let y =
        (transform[1][0] * point[0] + transform[1][1] * point[1] + transform[1][2]) / denominator;
    (x.is_finite() && y.is_finite()).then_some([x, y])
}

fn reprojection_errors(
    correspondences: &[FeatureCorrespondence],
    transform: &[[f64; 3]; 3],
) -> Vec<f64> {
    correspondences
        .iter()
        .map(|correspondence| {
            project_point(transform, correspondence.source)
                .map(|projected| {
                    ((projected[0] - correspondence.destination[0]).powi(2)
                        + (projected[1] - correspondence.destination[1]).powi(2))
                    .sqrt()
                })
                .unwrap_or(1.0e9)
        })
        .collect()
}

fn normalization_transform(points: &[[f64; 2]]) -> Result<Matrix3<f64>, String> {
    if points.len() < 4 {
        return Err("homography normalization needs at least four points".to_string());
    }
    let centroid = points.iter().fold([0.0; 2], |mut sum, point| {
        sum[0] += point[0];
        sum[1] += point[1];
        sum
    });
    let centroid = [
        centroid[0] / points.len() as f64,
        centroid[1] / points.len() as f64,
    ];
    let mean_distance = points
        .iter()
        .map(|point| ((point[0] - centroid[0]).powi(2) + (point[1] - centroid[1]).powi(2)).sqrt())
        .sum::<f64>()
        / points.len() as f64;
    if !mean_distance.is_finite() || mean_distance <= 1e-9 {
        return Err("homography points were spatially degenerate".to_string());
    }
    let scale = 2.0f64.sqrt() / mean_distance;
    Ok(Matrix3::new(
        scale,
        0.0,
        -scale * centroid[0],
        0.0,
        scale,
        -scale * centroid[1],
        0.0,
        0.0,
        1.0,
    ))
}

fn fit_homography_dlt(correspondences: &[FeatureCorrespondence]) -> Result<[[f64; 3]; 3], String> {
    if correspondences.len() < 4 {
        return Err("homography DLT needs at least four correspondences".to_string());
    }
    let source = correspondences
        .iter()
        .map(|correspondence| correspondence.source)
        .collect::<Vec<_>>();
    let destination = correspondences
        .iter()
        .map(|correspondence| correspondence.destination)
        .collect::<Vec<_>>();
    let source_normalization = normalization_transform(&source)?;
    let destination_normalization = normalization_transform(&destination)?;
    let destination_inverse = destination_normalization
        .try_inverse()
        .ok_or_else(|| "destination normalization was singular".to_string())?;
    let mut design = DMatrix::<f64>::zeros(correspondences.len() * 2, 9);
    for (index, correspondence) in correspondences.iter().enumerate() {
        let source = source_normalization
            * Vector3::new(correspondence.source[0], correspondence.source[1], 1.0);
        let destination = destination_normalization
            * Vector3::new(
                correspondence.destination[0],
                correspondence.destination[1],
                1.0,
            );
        let (x, y, u, v) = (
            source[0] / source[2],
            source[1] / source[2],
            destination[0] / destination[2],
            destination[1] / destination[2],
        );
        design[(index * 2, 0)] = -x;
        design[(index * 2, 1)] = -y;
        design[(index * 2, 2)] = -1.0;
        design[(index * 2, 6)] = u * x;
        design[(index * 2, 7)] = u * y;
        design[(index * 2, 8)] = u;
        design[(index * 2 + 1, 3)] = -x;
        design[(index * 2 + 1, 4)] = -y;
        design[(index * 2 + 1, 5)] = -1.0;
        design[(index * 2 + 1, 6)] = v * x;
        design[(index * 2 + 1, 7)] = v * y;
        design[(index * 2 + 1, 8)] = v;
    }
    // A minimal four-correspondence DLT system is 8x9. nalgebra's thin SVD
    // omits the ninth right-singular vector in that case, so solve the 9x9
    // normal-matrix eigensystem and take its smallest-eigenvalue vector.
    let normal = design.transpose() * design;
    let decomposition = SymmetricEigen::new(normal);
    let smallest = decomposition
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .ok_or_else(|| "homography DLT eigensystem was empty".to_string())?;
    let vector = decomposition.eigenvectors.column(smallest);
    let normalized = Matrix3::new(
        vector[0], vector[1], vector[2], vector[3], vector[4], vector[5], vector[6], vector[7],
        vector[8],
    );
    let mut homography = destination_inverse * normalized * source_normalization;
    let normalizer = homography[(2, 2)];
    if !normalizer.is_finite() || normalizer.abs() < 1e-10 {
        return Err("homography DLT produced an invalid scale".to_string());
    }
    homography /= normalizer;
    if !homography.iter().all(|value| value.is_finite()) || homography.determinant().abs() < 1e-12 {
        return Err("homography DLT produced a singular transform".to_string());
    }
    Ok(std::array::from_fn(|row| {
        std::array::from_fn(|column| homography[(row, column)])
    }))
}

fn next_xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn deterministic_sample_indices(state: &mut u64, count: usize) -> Option<[usize; 4]> {
    if count < 4 {
        return None;
    }
    let mut indices = [0usize; 4];
    for lane in 0..4 {
        let mut candidate = (next_xorshift64(state) as usize) % count;
        for _ in 0..count {
            if !indices[..lane].contains(&candidate) {
                break;
            }
            candidate = (candidate + 1) % count;
        }
        indices[lane] = candidate;
    }
    Some(indices)
}

fn robust_homography_fit(
    correspondences: &[FeatureCorrespondence],
) -> Result<RobustHomographyFit, String> {
    if correspondences.len() < HOMOGRAPHY_MIN_MATCHES_PER_SPLIT {
        return Err(format!(
            "homography fit has {} correspondences; need {}",
            correspondences.len(),
            HOMOGRAPHY_MIN_MATCHES_PER_SPLIT
        ));
    }
    let iterations = (correspondences.len() * 32).clamp(384, 2048);
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ correspondences.len() as u64;
    let mut best_inliers = Vec::<usize>::new();
    let mut best_median = f64::INFINITY;
    for _ in 0..iterations {
        let Some(indices) = deterministic_sample_indices(&mut state, correspondences.len()) else {
            continue;
        };
        let sample = indices
            .iter()
            .map(|index| correspondences[*index])
            .collect::<Vec<_>>();
        let Ok(transform) = fit_homography_dlt(&sample) else {
            continue;
        };
        let errors = reprojection_errors(correspondences, &transform);
        let inliers = errors
            .iter()
            .enumerate()
            .filter_map(|(index, error)| {
                (*error <= HOMOGRAPHY_RANSAC_THRESHOLD_PX).then_some(index)
            })
            .collect::<Vec<_>>();
        let median = error_percentiles(
            inliers
                .iter()
                .map(|index| errors[*index])
                .collect::<Vec<_>>(),
        )
        .0;
        if inliers.len() > best_inliers.len()
            || (inliers.len() == best_inliers.len() && median < best_median)
        {
            best_inliers = inliers;
            best_median = median;
        }
    }
    if best_inliers.len() < HOMOGRAPHY_MIN_MATCHES_PER_SPLIT {
        return Err(format!(
            "homography RANSAC retained {} inliers; need {}",
            best_inliers.len(),
            HOMOGRAPHY_MIN_MATCHES_PER_SPLIT
        ));
    }
    let inlier_correspondences = best_inliers
        .iter()
        .map(|index| correspondences[*index])
        .collect::<Vec<_>>();
    let transform = fit_homography_dlt(&inlier_correspondences)?;
    let errors = reprojection_errors(correspondences, &transform);
    let final_inlier_errors = errors
        .iter()
        .copied()
        .filter(|error| *error <= HOMOGRAPHY_RANSAC_THRESHOLD_PX)
        .collect::<Vec<_>>();
    let (median_inlier_error, p95_inlier_error) = error_percentiles(final_inlier_errors.clone());
    Ok(RobustHomographyFit {
        transform,
        inlier_count: final_inlier_errors.len(),
        inlier_ratio: final_inlier_errors.len() as f64 / correspondences.len() as f64,
        median_inlier_error,
        p95_inlier_error,
    })
}

fn feature_cell(point: [f64; 2], width: usize, height: usize) -> usize {
    let column = ((point[0].max(0.0) / width.max(1) as f64) * HOMOGRAPHY_FEATURE_GRID_COLS as f64)
        .floor()
        .clamp(0.0, (HOMOGRAPHY_FEATURE_GRID_COLS - 1) as f64) as usize;
    let row = ((point[1].max(0.0) / height.max(1) as f64) * HOMOGRAPHY_FEATURE_GRID_ROWS as f64)
        .floor()
        .clamp(0.0, (HOMOGRAPHY_FEATURE_GRID_ROWS - 1) as f64) as usize;
    row * HOMOGRAPHY_FEATURE_GRID_COLS + column
}

fn partition_correspondences(
    correspondences: &[FeatureCorrespondence],
    width: usize,
    height: usize,
) -> (
    Vec<FeatureCorrespondence>,
    Vec<FeatureCorrespondence>,
    usize,
    usize,
) {
    let mut training = Vec::new();
    let mut held_out = Vec::new();
    let mut training_cells = BTreeSet::new();
    let mut held_out_cells = BTreeSet::new();
    for correspondence in correspondences {
        let cell = feature_cell(correspondence.source, width, height);
        let row = cell / HOMOGRAPHY_FEATURE_GRID_COLS;
        let column = cell % HOMOGRAPHY_FEATURE_GRID_COLS;
        if (row + column).is_multiple_of(2) {
            training.push(*correspondence);
            training_cells.insert(cell);
        } else {
            held_out.push(*correspondence);
            held_out_cells.insert(cell);
        }
    }
    (
        training,
        held_out,
        training_cells.len(),
        held_out_cells.len(),
    )
}

fn cross_fit_disagreement(
    first: &[[f64; 3]; 3],
    second: &[[f64; 3]; 3],
    width: usize,
    height: usize,
) -> f64 {
    let mut maximum = 0.0f64;
    for row in 0..3 {
        for column in 0..3 {
            let point = [
                width as f64 * column as f64 * 0.5,
                height as f64 * row as f64 * 0.5,
            ];
            let Some(first_point) = project_point(first, point) else {
                return 1.0e9;
            };
            let Some(second_point) = project_point(second, point) else {
                return 1.0e9;
            };
            maximum = maximum.max(
                ((first_point[0] - second_point[0]).powi(2)
                    + (first_point[1] - second_point[1]).powi(2))
                .sqrt(),
            );
        }
    }
    maximum
}

pub fn fit_disjoint_homography(
    correspondences: &[FeatureCorrespondence],
    width: usize,
    height: usize,
) -> Result<MatchResult, String> {
    let (training, held_out, training_cell_count, held_out_cell_count) =
        partition_correspondences(correspondences, width, height);
    let training_fit = robust_homography_fit(&training)?;
    let partition_support_ok = held_out.len() >= HOMOGRAPHY_MIN_MATCHES_PER_SPLIT
        && training_cell_count >= HOMOGRAPHY_MIN_CELLS_PER_SPLIT
        && held_out_cell_count >= HOMOGRAPHY_MIN_CELLS_PER_SPLIT;
    let held_out_errors = reprojection_errors(&held_out, &training_fit.transform);
    let held_out_inlier_errors = held_out_errors
        .iter()
        .copied()
        .filter(|error| *error <= HOMOGRAPHY_RANSAC_THRESHOLD_PX)
        .collect::<Vec<_>>();
    let held_out_median = error_percentiles(held_out_errors.clone()).0;
    let held_out_p95 = error_percentiles(held_out_inlier_errors.clone()).1;
    let held_out_inlier_count = held_out_inlier_errors.len();
    let held_out_inlier_ratio = if held_out.is_empty() {
        0.0
    } else {
        held_out_inlier_count as f64 / held_out.len() as f64
    };
    let reverse_fit = robust_homography_fit(&held_out).ok();
    let (reverse_inlier_count, reverse_ratio, reverse_median, reverse_p95, disagreement) =
        if let Some(reverse_fit) = reverse_fit {
            let reverse_errors = reprojection_errors(&training, &reverse_fit.transform);
            let inlier_errors = reverse_errors
                .iter()
                .copied()
                .filter(|error| *error <= HOMOGRAPHY_RANSAC_THRESHOLD_PX)
                .collect::<Vec<_>>();
            let median = error_percentiles(reverse_errors).0;
            let p95 = error_percentiles(inlier_errors.clone()).1;
            let inliers = inlier_errors.len();
            (
                inliers,
                inliers as f64 / training.len().max(1) as f64,
                median,
                p95,
                cross_fit_disagreement(
                    &training_fit.transform,
                    &reverse_fit.transform,
                    width,
                    height,
                ),
            )
        } else {
            (0, 0.0, 1.0e9, 1.0e9, 1.0e9)
        };
    let forward_ok = held_out_inlier_ratio >= HOMOGRAPHY_MIN_CROSS_INLIER_RATIO
        && held_out_median <= HOMOGRAPHY_MAX_CROSS_MEDIAN_ERROR_PX
        && held_out_p95 <= HOMOGRAPHY_MAX_CROSS_P95_ERROR_PX;
    let reverse_ok = reverse_ratio >= HOMOGRAPHY_MIN_CROSS_INLIER_RATIO
        && reverse_median <= HOMOGRAPHY_MAX_CROSS_MEDIAN_ERROR_PX
        && reverse_p95 <= HOMOGRAPHY_MAX_CROSS_P95_ERROR_PX;
    let transforms_agree = disagreement <= HOMOGRAPHY_MAX_CROSS_FIT_DISAGREEMENT_PX;
    let passed = partition_support_ok && forward_ok && reverse_ok && transforms_agree;
    let reason = if held_out.len() < HOMOGRAPHY_MIN_MATCHES_PER_SPLIT {
        format!(
            "held-out feature partition has {} matches; need {}",
            held_out.len(),
            HOMOGRAPHY_MIN_MATCHES_PER_SPLIT
        )
    } else if training_cell_count < HOMOGRAPHY_MIN_CELLS_PER_SPLIT
        || held_out_cell_count < HOMOGRAPHY_MIN_CELLS_PER_SPLIT
    {
        format!(
            "feature partitions lack spatial coverage: {}/{} cells, need {} each",
            training_cell_count, held_out_cell_count, HOMOGRAPHY_MIN_CELLS_PER_SPLIT
        )
    } else if !forward_ok {
        format!(
            "training-fit homography did not generalize to held-out cells: {:.1}% inliers, median {:.2}px, p95 {:.2}px",
            held_out_inlier_ratio * 100.0,
            held_out_median,
            held_out_p95
        )
    } else if !reverse_ok {
        format!(
            "held-out-fit homography did not generalize to training cells: {:.1}% inliers, median {:.2}px, p95 {:.2}px",
            reverse_ratio * 100.0,
            reverse_median,
            reverse_p95
        )
    } else if !transforms_agree {
        format!(
            "cross-fit homographies disagree by {:.2}px, limit {:.2}px",
            disagreement, HOMOGRAPHY_MAX_CROSS_FIT_DISAGREEMENT_PX
        )
    } else {
        format!(
            "spatially disjoint cross-fit passed: {:.1}%/{:.1}% cross-inliers, {:.2}px maximum transform disagreement",
            held_out_inlier_ratio * 100.0,
            reverse_ratio * 100.0,
            disagreement
        )
    };
    Ok(MatchResult {
        match_count: correspondences.len(),
        training_match_count: training.len(),
        held_out_match_count: held_out.len(),
        training_spatial_cell_count: training_cell_count,
        held_out_spatial_cell_count: held_out_cell_count,
        inliers: training_fit.inlier_count,
        training_inlier_ratio: training_fit.inlier_ratio,
        training_median_error: training_fit.median_inlier_error,
        training_p95_error: training_fit.p95_inlier_error,
        median_error: held_out_median,
        p95_error: held_out_p95,
        held_out_inlier_count,
        held_out_inlier_ratio,
        reverse_validation_inlier_count: reverse_inlier_count,
        reverse_validation_inlier_ratio: reverse_ratio,
        reverse_validation_median_error: reverse_median,
        reverse_validation_p95_error: reverse_p95,
        cross_fit_max_disagreement_px: disagreement,
        disjoint_validation_passed: passed,
        validation_reason: reason,
        partition_method: "4x4_whole_cell_checkerboard_cross_fit",
        transform: training_fit.transform,
    })
}

#[cfg(feature = "use-opencv")]
pub mod opencv_match {
    pub use super::MatchResult;
    use super::{fit_disjoint_homography, FeatureCorrespondence, HOMOGRAPHY_MIN_MATCHES_PER_SPLIT};
    use crate::constants::{MAX_14BIT, MAX_16BIT};
    use ndarray::Array3;
    use opencv::core::{self, DMatch, KeyPoint, Mat, Scalar, Size, Vec3w, Vector, NORM_L2};
    use opencv::features2d::{BFMatcher, SIFT};
    use opencv::imgproc;
    use opencv::prelude::*;

    pub fn is_available() -> bool {
        true
    }

    fn grayscale_scale_divisor(bit_depth: u8) -> f64 {
        match bit_depth {
            14 => MAX_14BIT / 255.0,
            _ => MAX_16BIT / 255.0,
        }
    }

    fn array3_to_gray_mat(img: &Array3<u16>, bit_depth: u8) -> Result<Mat, opencv::Error> {
        let (h, w, _) = img.dim();
        let divisor = grayscale_scale_divisor(bit_depth);
        let mut buf = Vec::<u8>::with_capacity(h * w);
        for y in 0..h {
            for x in 0..w {
                let r = img[[y, x, 0]] as f64;
                let g = img[[y, x, 1]] as f64;
                let b = img[[y, x, 2]] as f64;
                let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                buf.push((lum / divisor).round().clamp(0.0, 255.0) as u8);
            }
        }
        let mat_ref = Mat::new_rows_cols_with_data(h as i32, w as i32, &buf)?;
        Ok(mat_ref.clone_pointee())
    }

    fn array3_to_bgr16_mat(img: &Array3<u16>) -> Result<Mat, opencv::Error> {
        let (h, w, _) = img.dim();
        let mut buf = Vec::<Vec3w>::with_capacity(h * w);
        for y in 0..h {
            for x in 0..w {
                buf.push(Vec3w::from([
                    img[[y, x, 2]],
                    img[[y, x, 1]],
                    img[[y, x, 0]],
                ]));
            }
        }
        let mat_ref = Mat::new_rows_cols_with_data(h as i32, w as i32, &buf)?;
        Ok(mat_ref.clone_pointee())
    }

    pub fn bgr16_mat_to_array3(mat: &Mat) -> Result<Array3<u16>, opencv::Error> {
        let h = mat.rows() as usize;
        let w = mat.cols() as usize;
        let mut arr = Array3::<u16>::zeros((h, w, 3));
        for y in 0..h {
            for x in 0..w {
                let pixel = mat.at_2d::<Vec3w>(y as i32, x as i32)?;
                arr[[y, x, 0]] = pixel[2];
                arr[[y, x, 1]] = pixel[1];
                arr[[y, x, 2]] = pixel[0];
            }
        }
        Ok(arr)
    }

    pub fn find_homography_robust(
        left: &Array3<u16>,
        right: &Array3<u16>,
        bit_depth: u8,
    ) -> Option<MatchResult> {
        let result = (|| -> Result<Option<MatchResult>, opencv::Error> {
            let gray_l = array3_to_gray_mat(left, bit_depth)?;
            let gray_r = array3_to_gray_mat(right, bit_depth)?;

            let mut sift = SIFT::create_def()?;
            let mut kp_l = Vector::<KeyPoint>::new();
            let mut kp_r = Vector::<KeyPoint>::new();
            let mut desc_l = Mat::default();
            let mut desc_r = Mat::default();
            let no_mask = Mat::default();

            sift.detect_and_compute_def(&gray_l, &no_mask, &mut kp_l, &mut desc_l)?;
            sift.detect_and_compute_def(&gray_r, &no_mask, &mut kp_r, &mut desc_r)?;

            if desc_l.rows() < 2 || desc_r.rows() < 2 {
                return Ok(None);
            }

            let matcher = BFMatcher::create(NORM_L2, false)?;
            let mut knn_matches = Vector::<Vector<DMatch>>::new();
            matcher.knn_train_match_def(&desc_l, &desc_r, &mut knn_matches, 2)?;

            let mut correspondences = Vec::<FeatureCorrespondence>::new();
            for pair in knn_matches.iter() {
                if pair.len() < 2 {
                    continue;
                }
                let m = pair.get(0)?;
                let n = pair.get(1)?;
                if m.distance < 0.75 * n.distance {
                    let source = kp_l.get(m.query_idx as usize)?.pt();
                    let destination = kp_r.get(m.train_idx as usize)?.pt();
                    correspondences.push(FeatureCorrespondence {
                        source: [source.x as f64, source.y as f64],
                        destination: [destination.x as f64, destination.y as f64],
                    });
                }
            }

            if correspondences.len() < HOMOGRAPHY_MIN_MATCHES_PER_SPLIT * 2 {
                return Ok(None);
            }
            let (_, width, _) = left.dim();
            let (height, _, _) = left.dim();
            match fit_disjoint_homography(&correspondences, width, height) {
                Ok(result) => Ok(Some(result)),
                Err(reason) => {
                    log::warn!("SIFT homography fit rejected: {}", reason);
                    Ok(None)
                }
            }
        })();

        match result {
            Ok(v) => v,
            Err(e) => {
                log::error!("OpenCV homography estimation failed: {}", e);
                None
            }
        }
    }

    pub fn warp_image(
        src: &Array3<u16>,
        h: &[[f64; 3]; 3],
        canvas_w: usize,
        canvas_h: usize,
    ) -> Option<Array3<u16>> {
        let result = (|| -> Result<Array3<u16>, opencv::Error> {
            let src_mat = array3_to_bgr16_mat(src)?;
            let h_mat = Mat::from_slice_2d(h)?;
            let mut dst_mat = Mat::default();
            imgproc::warp_perspective(
                &src_mat,
                &mut dst_mat,
                &h_mat,
                Size::new(canvas_w as i32, canvas_h as i32),
                imgproc::INTER_LANCZOS4,
                core::BORDER_CONSTANT,
                Scalar::all(0.0),
            )?;
            bgr16_mat_to_array3(&dst_mat)
        })();

        match result {
            Ok(arr) => Some(arr),
            Err(e) => {
                log::error!("OpenCV warp_perspective failed: {}", e);
                None
            }
        }
    }
}

#[cfg(not(feature = "use-opencv"))]
pub mod opencv_match {
    pub use super::MatchResult;
    use ndarray::Array3;

    pub fn is_available() -> bool {
        false
    }

    pub fn find_homography_robust(
        _left: &Array3<u16>,
        _right: &Array3<u16>,
        _bit_depth: u8,
    ) -> Option<MatchResult> {
        None
    }

    pub fn warp_image(
        _src: &Array3<u16>,
        _h: &[[f64; 3]; 3],
        _canvas_w: usize,
        _canvas_h: usize,
    ) -> Option<Array3<u16>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_WIDTH: usize = 800;
    const TEST_HEIGHT: usize = 600;

    fn synthetic_correspondences(
        training_transform: [[f64; 3]; 3],
        held_out_transform: [[f64; 3]; 3],
        include_outliers: bool,
    ) -> Vec<FeatureCorrespondence> {
        let mut correspondences = Vec::new();
        for row in 0..HOMOGRAPHY_FEATURE_GRID_ROWS {
            for column in 0..HOMOGRAPHY_FEATURE_GRID_COLS {
                let transform = if (row + column).is_multiple_of(2) {
                    training_transform
                } else {
                    held_out_transform
                };
                for sample in 0..5 {
                    let source = [
                        (column as f64 + 0.18 + sample as f64 * 0.13) * TEST_WIDTH as f64
                            / HOMOGRAPHY_FEATURE_GRID_COLS as f64,
                        (row as f64 + 0.22 + sample as f64 * 0.11) * TEST_HEIGHT as f64
                            / HOMOGRAPHY_FEATURE_GRID_ROWS as f64,
                    ];
                    let mut destination = project_point(&transform, source).unwrap();
                    destination[0] += ((row * 17 + column * 11 + sample * 5) as f64).sin() * 0.04;
                    destination[1] += ((row * 7 + column * 19 + sample * 3) as f64).cos() * 0.04;
                    if include_outliers && sample == 0 {
                        destination[0] += 80.0 + column as f64 * 9.0;
                        destination[1] -= 55.0 + row as f64 * 7.0;
                    }
                    correspondences.push(FeatureCorrespondence {
                        source,
                        destination,
                    });
                }
            }
        }
        correspondences
    }

    fn stable_transform() -> [[f64; 3]; 3] {
        [
            [1.002, 0.006, 42.0],
            [-0.004, 0.998, 11.0],
            [0.000_012, -0.000_009, 1.0],
        ]
    }

    #[test]
    fn disjoint_homography_recovers_shared_projective_geometry() {
        let expected = stable_transform();
        let correspondences = synthetic_correspondences(expected, expected, true);

        let result = fit_disjoint_homography(&correspondences, TEST_WIDTH, TEST_HEIGHT).unwrap();

        assert!(
            result.disjoint_validation_passed,
            "{}",
            result.validation_reason
        );
        assert_eq!(result.training_match_count, 40);
        assert_eq!(result.held_out_match_count, 40);
        assert_eq!(result.training_spatial_cell_count, 8);
        assert_eq!(result.held_out_spatial_cell_count, 8);
        assert!(result.training_inlier_ratio >= 0.75);
        assert!(result.held_out_inlier_ratio >= 0.75);
        assert!(result.reverse_validation_inlier_ratio >= 0.75);
        assert!(result.cross_fit_max_disagreement_px < 0.5);
        for point in [[0.0, 0.0], [400.0, 300.0], [800.0, 600.0]] {
            let expected_point = project_point(&expected, point).unwrap();
            let actual_point = project_point(&result.transform, point).unwrap();
            assert!((expected_point[0] - actual_point[0]).abs() < 0.25);
            assert!((expected_point[1] - actual_point[1]).abs() < 0.25);
        }
    }

    #[test]
    fn disjoint_homography_rejects_cell_specific_geometry() {
        let training = stable_transform();
        let mut held_out = training;
        held_out[0][2] += 18.0;
        held_out[1][2] -= 12.0;
        let correspondences = synthetic_correspondences(training, held_out, false);

        let result = fit_disjoint_homography(&correspondences, TEST_WIDTH, TEST_HEIGHT).unwrap();

        assert!(!result.disjoint_validation_passed);
        assert!(
            result.validation_reason.contains("did not generalize")
                || result.validation_reason.contains("disagree")
        );
    }

    #[test]
    fn disjoint_homography_fit_is_deterministic() {
        let transform = stable_transform();
        let correspondences = synthetic_correspondences(transform, transform, true);

        let first = fit_disjoint_homography(&correspondences, TEST_WIDTH, TEST_HEIGHT).unwrap();
        let second = fit_disjoint_homography(&correspondences, TEST_WIDTH, TEST_HEIGHT).unwrap();

        assert_eq!(first.transform, second.transform);
        assert_eq!(first.inliers, second.inliers);
        assert_eq!(first.validation_reason, second.validation_reason);
    }

    #[cfg(feature = "use-opencv")]
    #[test]
    fn opencv_backend_executes_sift_and_rgb16_warp_smoke_paths() {
        use ndarray::Array3;

        assert!(opencv_match::is_available());

        let featureless = Array3::<u16>::from_elem((64, 64, 3), 24_000);
        assert!(opencv_match::find_homography_robust(&featureless, &featureless, 16).is_none());

        let mut source = Array3::<u16>::zeros((9, 11, 3));
        for y in 0..source.dim().0 {
            for x in 0..source.dim().1 {
                source[[y, x, 0]] = (x * 1_301 + y * 211) as u16;
                source[[y, x, 1]] = (x * 719 + y * 997 + 3_000) as u16;
                source[[y, x, 2]] = (x * 313 + y * 1_511 + 7_000) as u16;
            }
        }
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let warped = opencv_match::warp_image(&source, &identity, 11, 9)
            .expect("OpenCV should execute an identity RGB16 projective warp");

        assert_eq!(warped, source);
    }
}

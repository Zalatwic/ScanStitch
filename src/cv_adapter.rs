#[cfg(feature = "use-opencv")]
pub mod opencv_match {
    use crate::constants::{MAX_14BIT, MAX_16BIT};
    use ndarray::Array3;
    use opencv::calib3d;
    use opencv::core::{
        self, DMatch, KeyPoint, Mat, Point2f, Scalar, Size, Vec3w, Vector, NORM_L2,
    };
    use opencv::features2d::{BFMatcher, SIFT};
    use opencv::imgproc;
    use opencv::prelude::*;

    pub struct MatchResult {
        pub inliers: usize,
        pub median_error: f64,
        pub p95_error: f64,
        pub transform: [[f64; 3]; 3],
    }

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

    fn extract_matrix(mat: &Mat) -> Result<[[f64; 3]; 3], opencv::Error> {
        let mut out = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                out[r][c] = *mat.at_2d::<f64>(r as i32, c as i32)?;
            }
        }
        Ok(out)
    }

    fn reprojection_errors(
        src: &Vector<Point2f>,
        dst: &Vector<Point2f>,
        h_mat: &Mat,
        inlier_mask: &Mat,
    ) -> Vec<f64> {
        let mut errors = Vec::new();
        let h = extract_matrix(h_mat).unwrap_or([[0.0; 3]; 3]);
        for i in 0..src.len() {
            if let Ok(mask_val) = inlier_mask.at_2d::<u8>(i as i32, 0) {
                if *mask_val == 0 {
                    continue;
                }
            }
            let sp = match src.get(i) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let dp = match dst.get(i) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let denom = h[2][0] * sp.x as f64 + h[2][1] * sp.y as f64 + h[2][2];
            if denom.abs() < 1e-12 {
                continue;
            }
            let px = (h[0][0] * sp.x as f64 + h[0][1] * sp.y as f64 + h[0][2]) / denom;
            let py = (h[1][0] * sp.x as f64 + h[1][1] * sp.y as f64 + h[1][2]) / denom;
            let err = ((px - dp.x as f64).powi(2) + (py - dp.y as f64).powi(2)).sqrt();
            errors.push(err);
        }
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        errors
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

            let mut good_l = Vector::<Point2f>::new();
            let mut good_r = Vector::<Point2f>::new();
            for pair in knn_matches.iter() {
                if pair.len() < 2 {
                    continue;
                }
                let m = pair.get(0)?;
                let n = pair.get(1)?;
                if m.distance < 0.75 * n.distance {
                    good_l.push(kp_l.get(m.query_idx as usize)?.pt());
                    good_r.push(kp_r.get(m.train_idx as usize)?.pt());
                }
            }

            if good_l.len() < 10 {
                return Ok(None);
            }

            let mut inlier_mask = Mat::default();
            let h_mat =
                calib3d::find_homography(&good_l, &good_r, &mut inlier_mask, calib3d::RANSAC, 3.0)?;

            if h_mat.rows() != 3 || h_mat.cols() != 3 {
                return Ok(None);
            }

            let mut inlier_count = 0usize;
            for i in 0..inlier_mask.rows() {
                if *inlier_mask.at_2d::<u8>(i, 0)? != 0 {
                    inlier_count += 1;
                }
            }
            if inlier_count < 10 {
                return Ok(None);
            }

            let errors = reprojection_errors(&good_l, &good_r, &h_mat, &inlier_mask);
            let median_error = if errors.is_empty() {
                0.0
            } else {
                errors[errors.len() / 2]
            };
            let p95_error = if errors.is_empty() {
                0.0
            } else {
                errors[((errors.len() as f64 * 0.95) as usize).min(errors.len() - 1)]
            };

            Ok(Some(MatchResult {
                inliers: inlier_count,
                median_error,
                p95_error,
                transform: extract_matrix(&h_mat)?,
            }))
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
    use ndarray::Array3;

    pub struct MatchResult {
        pub inliers: usize,
        pub median_error: f64,
        pub p95_error: f64,
        pub transform: [[f64; 3]; 3],
    }

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

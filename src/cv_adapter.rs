#[cfg(feature = "use-opencv")]
pub mod opencv_match {
    use ndarray::Array3;

    pub struct MatchResult {
        pub inliers: usize,
        pub median_error: f64,
        pub p95_error: f64,
        pub transform: [[f64; 3]; 3],
    }

    pub fn find_homography_orb(_l: &Array3<u16>, _r: &Array3<u16>) -> Option<MatchResult> {
        log::warn!("OpenCV matching not yet implemented");
        None
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

    pub fn find_homography_orb(_l: &Array3<u16>, _r: &Array3<u16>) -> Option<MatchResult> {
        log::warn!("OpenCV feature not enabled; skipping ORB-based matching");
        None
    }
}

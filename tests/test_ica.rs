use ndarray::Array3;
use approx::assert_relative_eq;
use nalgebra::Matrix3;

#[test]
fn test_streaming_mean_covariance() {
    let mut img = Array3::<f64>::zeros((2, 3, 3));
    img[[0, 0, 0]] = 1.0; img[[0, 0, 1]] = 2.0; img[[0, 0, 2]] = 3.0;
    img[[0, 1, 0]] = 4.0; img[[0, 1, 1]] = 5.0; img[[0, 1, 2]] = 6.0;
    img[[0, 2, 0]] = 7.0; img[[0, 2, 1]] = 8.0; img[[0, 2, 2]] = 9.0;
    img[[1, 0, 0]] = 2.0; img[[1, 0, 1]] = 3.0; img[[1, 0, 2]] = 4.0;
    img[[1, 1, 0]] = 5.0; img[[1, 1, 1]] = 6.0; img[[1, 1, 2]] = 7.0;
    img[[1, 2, 0]] = 8.0; img[[1, 2, 1]] = 9.0; img[[1, 2, 2]] = 10.0;
    let (mean, cov) = scanstitch::ica::compute_mean_cov(&img);
    assert_relative_eq!(mean[0], 4.5, epsilon = 1e-10);
    assert_relative_eq!(mean[1], 5.5, epsilon = 1e-10);
    assert_relative_eq!(mean[2], 6.5, epsilon = 1e-10);
    assert_relative_eq!(cov[(0, 1)], cov[(1, 0)], epsilon = 1e-10);
}

#[test]
fn test_whitening_decorrelates() {
    let n = 1000;
    let mut img = Array3::<f64>::zeros((1, n, 3));
    for x in 0..n {
        let t = x as f64 / n as f64;
        img[[0, x, 0]] = t * 2.0 + 0.1;
        img[[0, x, 1]] = t * 1.5 + 0.3;
        img[[0, x, 2]] = t * 1.0 + 0.5;
    }
    let (mean, cov) = scanstitch::ica::compute_mean_cov(&img);
    let whitening = scanstitch::ica::compute_whitening_matrix(&cov);
    let mut whitened = img.clone();
    scanstitch::ica::apply_whitening(&mut whitened, &mean, &whitening);
    let (_, cov_w) = scanstitch::ica::compute_mean_cov(&whitened);
    for i in 0..3 {
        assert_relative_eq!(cov_w[(i, i)], 1.0, epsilon = 0.1);
        for j in 0..3 {
            if i != j { assert!(cov_w[(i, j)].abs() < 0.1, "({},{})={}", i, j, cov_w[(i, j)]); }
        }
    }
}

#[test]
fn test_matrix_inverse_sqrt() {
    let m = Matrix3::new(2.0, 0.5, 0.1, 0.5, 3.0, 0.2, 0.1, 0.2, 1.5);
    let m_inv_sqrt = scanstitch::ica::matrix_inverse_sqrt(&m);
    let result = m_inv_sqrt * m * m_inv_sqrt;
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(result[(i, j)], expected, epsilon = 1e-6);
        }
    }
}

#[test]
fn test_ica_recovers_independent_sources() {
    let n = 5000;
    let mut sources = Array3::<f64>::zeros((1, n, 3));
    let mut lcg = 12345u64;
    for x in 0..n {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (lcg >> 33) as f64 / (1u64 << 31) as f64;
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (lcg >> 33) as f64 / (1u64 << 31) as f64;
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u3 = (lcg >> 33) as f64 / (1u64 << 31) as f64;
        sources[[0, x, 0]] = u1;
        sources[[0, x, 1]] = if u2 > 0.5 { 0.8 } else { 0.2 };
        sources[[0, x, 2]] = u3 * u3;
    }
    let mix = Matrix3::new(0.8, 0.3, 0.1, 0.2, 0.7, 0.3, 0.1, 0.2, 0.9);
    let mut mixed = Array3::<f64>::zeros((1, n, 3));
    for x in 0..n {
        let s = nalgebra::Vector3::new(sources[[0, x, 0]], sources[[0, x, 1]], sources[[0, x, 2]]);
        let m = mix * s;
        mixed[[0, x, 0]] = m[0]; mixed[[0, x, 1]] = m[1]; mixed[[0, x, 2]] = m[2];
    }
    let result = scanstitch::ica::run_fastica(&mixed, 50, 1e-5);
    assert!(result.converged, "ICA should converge");
    let (_, cov) = scanstitch::ica::compute_mean_cov(&result.separated);
    for i in 0..3 { for j in 0..3 {
        if i != j { assert!(cov[(i, j)].abs() < 0.15, "cov({},{})={}", i, j, cov[(i, j)]); }
    }}
}

#[test]
fn test_permutation_resolution() {
    let n = 100;
    let mut original = Array3::<f64>::zeros((1, n, 3));
    let mut separated = Array3::<f64>::zeros((1, n, 3));
    for x in 0..n {
        let t = x as f64 / n as f64;
        original[[0, x, 0]] = t;
        original[[0, x, 1]] = 1.0 - t;
        original[[0, x, 2]] = 0.5;
        separated[[0, x, 0]] = 0.5;       // was B
        separated[[0, x, 1]] = t;          // was R
        separated[[0, x, 2]] = 1.0 - t;   // was G
    }
    let perm = scanstitch::ica::resolve_permutation(&original, &separated);
    assert_eq!(perm[0], 1, "R assignment");
    assert_eq!(perm[1], 2, "G assignment");
    assert_eq!(perm[2], 0, "B assignment");
}

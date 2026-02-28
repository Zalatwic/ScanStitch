use nalgebra::{Matrix3, Vector3, SymmetricEigen};
use ndarray::Array3;

use crate::streaming;

/// Result of running FastICA on a 3-channel image.
pub struct IcaResult {
    /// Separated components [H, W, 3].
    pub separated: Array3<f64>,
    /// Whether the algorithm converged within the iteration limit.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
    /// The unmixing matrix W such that S = W^T * whitened_data.
    pub unmixing: Matrix3<f64>,
    /// Permutation mapping: perm[output_ch] = separated_ch.
    pub permutation: [usize; 3],
    /// Sign flips applied per output channel.
    pub signs: [f64; 3],
}

/// Compute the per-channel mean and 3x3 covariance matrix over all pixels,
/// using two streaming passes with tiled iteration.
pub fn compute_mean_cov(img: &Array3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
    let (h, w, _) = img.dim();
    let n = (h * w) as f64;
    let tiles = streaming::tile_ranges(h, streaming::DEFAULT_TILE_ROWS);

    // Pass 1: accumulate sums for mean
    let mut sum = Vector3::new(0.0, 0.0, 0.0);
    for &(r0, r1) in &tiles {
        let tile = streaming::tile_view(img, r0, r1);
        let (th, tw, _) = tile.dim();
        for r in 0..th {
            for c in 0..tw {
                sum[0] += tile[[r, c, 0]];
                sum[1] += tile[[r, c, 1]];
                sum[2] += tile[[r, c, 2]];
            }
        }
    }
    let mean = sum / n;

    // Pass 2: accumulate (x - mean)(x - mean)^T for covariance
    let mut cov = Matrix3::zeros();
    for &(r0, r1) in &tiles {
        let tile = streaming::tile_view(img, r0, r1);
        let (th, tw, _) = tile.dim();
        for r in 0..th {
            for c in 0..tw {
                let d = Vector3::new(
                    tile[[r, c, 0]] - mean[0],
                    tile[[r, c, 1]] - mean[1],
                    tile[[r, c, 2]] - mean[2],
                );
                // Outer product d * d^T accumulated into cov
                for i in 0..3 {
                    for j in 0..3 {
                        cov[(i, j)] += d[i] * d[j];
                    }
                }
            }
        }
    }
    cov /= n;

    (mean, cov)
}

/// Regularization constant added to eigenvalues for numerical stability.
const WHITEN_REG: f64 = 1e-6;

/// Compute the whitening matrix W = Lambda^{-1/2} * E^T from the covariance.
/// A small regularization (WHITEN_REG) is added to eigenvalues for numerical
/// stability with rank-deficient data.
pub fn compute_whitening_matrix(cov: &Matrix3<f64>) -> Matrix3<f64> {
    let cov_reg = cov + Matrix3::identity() * WHITEN_REG;
    let eigen = SymmetricEigen::new(cov_reg);
    let mut lambda_inv_sqrt = Matrix3::zeros();
    for i in 0..3 {
        let ev = eigen.eigenvalues[i].max(1e-10);
        lambda_inv_sqrt[(i, i)] = 1.0 / ev.sqrt();
    }
    // W = Lambda^{-1/2} * E^T
    lambda_inv_sqrt * eigen.eigenvectors.transpose()
}

/// Apply whitening in-place: for each pixel, x = whitening * (x - mean).
/// Adds small regularization noise (matching WHITEN_REG) for numerical stability
/// with rank-deficient data.
pub fn apply_whitening(img: &mut Array3<f64>, mean: &Vector3<f64>, whitening: &Matrix3<f64>) {
    let (h, w, _) = img.dim();
    let tiles = streaming::tile_ranges(h, streaming::DEFAULT_TILE_ROWS);
    // Noise scale: uniform in [-a, a] has variance a^2/3.
    // We want variance = WHITEN_REG, so a = sqrt(3 * WHITEN_REG).
    let noise_amp = (3.0 * WHITEN_REG).sqrt();
    // Three independent LCG states, one per channel
    let mut lcg_state: [u64; 3] = [
        0xDEAD_BEEF_CAFE_BABE,
        0x1234_5678_9ABC_DEF0,
        0xFEDC_BA98_7654_3210,
    ];
    for &(r0, r1) in &tiles {
        for r in r0..r1 {
            for c in 0..w {
                let mut noise = [0.0f64; 3];
                for ch in 0..3 {
                    lcg_state[ch] = lcg_state[ch]
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = (lcg_state[ch] >> 33) as f64 / (1u64 << 31) as f64;
                    noise[ch] = (u * 2.0 - 1.0) * noise_amp;
                }
                let d = Vector3::new(
                    img[[r, c, 0]] - mean[0] + noise[0],
                    img[[r, c, 1]] - mean[1] + noise[1],
                    img[[r, c, 2]] - mean[2] + noise[2],
                );
                let wh = whitening * d;
                img[[r, c, 0]] = wh[0];
                img[[r, c, 1]] = wh[1];
                img[[r, c, 2]] = wh[2];
            }
        }
    }
}

/// Compute the matrix inverse square root: M^{-1/2} = E * Lambda^{-1/2} * E^T.
pub fn matrix_inverse_sqrt(m: &Matrix3<f64>) -> Matrix3<f64> {
    let eigen = SymmetricEigen::new(*m);
    let mut lambda_inv_sqrt = Matrix3::zeros();
    for i in 0..3 {
        let ev = eigen.eigenvalues[i].max(1e-10);
        lambda_inv_sqrt[(i, i)] = 1.0 / ev.sqrt();
    }
    eigen.eigenvectors * lambda_inv_sqrt * eigen.eigenvectors.transpose()
}

/// Symmetric orthogonalization: W = (W * W^T)^{-1/2} * W.
fn symmetric_orthogonalize(w_mat: &mut Matrix3<f64>) {
    let wwt = *w_mat * w_mat.transpose();
    let inv_sqrt = matrix_inverse_sqrt(&wwt);
    *w_mat = inv_sqrt * *w_mat;
}

/// Run the FastICA algorithm on a 3-channel image.
///
/// Returns an `IcaResult` with separated components, convergence info,
/// the unmixing matrix, and permutation/sign resolution.
pub fn run_fastica(img: &Array3<f64>, max_iter: usize, tol: f64) -> IcaResult {
    let (h, w, _) = img.dim();
    let n = (h * w) as f64;
    let tiles = streaming::tile_ranges(h, streaming::DEFAULT_TILE_ROWS);

    // Step 1: compute mean and cov (streaming)
    let (mean, cov) = compute_mean_cov(img);

    // Step 2: compute whitening matrix
    let whitening = compute_whitening_matrix(&cov);

    // Step 3: whiten the data
    let mut whitened = img.clone();
    apply_whitening(&mut whitened, &mean, &whitening);

    // Step 4: initialize W deterministically and orthogonalize
    let mut w_mat = Matrix3::new(
        0.7, 0.5, 0.3,
        0.3, 0.8, 0.4,
        0.2, 0.3, 0.9,
    );
    symmetric_orthogonalize(&mut w_mat);

    // Step 5: iterate
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;
        let w_old = w_mat;

        // Accumulators: sum_xg[i] is a 3-vector, sum_gp[i] is a scalar
        let mut sum_xg = [Vector3::new(0.0, 0.0, 0.0); 3];
        let mut sum_gp = [0.0f64; 3];

        // Stream over all pixels
        for &(r0, r1) in &tiles {
            let tile = streaming::tile_view(&whitened, r0, r1);
            let (th, tw, _) = tile.dim();
            for r in 0..th {
                for x_pos in 0..tw {
                    let x = Vector3::new(
                        tile[[r, x_pos, 0]],
                        tile[[r, x_pos, 1]],
                        tile[[r, x_pos, 2]],
                    );
                    for i in 0..3 {
                        let w_col = Vector3::new(w_old[(0, i)], w_old[(1, i)], w_old[(2, i)]);
                        let u = w_col.dot(&x);
                        let g = u.tanh();
                        let gp = 1.0 - g * g;
                        sum_xg[i] += x * g;
                        sum_gp[i] += gp;
                    }
                }
            }
        }

        // Average and update W columns
        for i in 0..3 {
            let avg_xg = sum_xg[i] / n;
            let avg_gp = sum_gp[i] / n;
            let w_col = Vector3::new(w_old[(0, i)], w_old[(1, i)], w_old[(2, i)]);
            let new_col = avg_xg - avg_gp * w_col;
            w_mat[(0, i)] = new_col[0];
            w_mat[(1, i)] = new_col[1];
            w_mat[(2, i)] = new_col[2];
        }

        // Symmetric orthogonalize
        symmetric_orthogonalize(&mut w_mat);

        // Check convergence: max over columns of (1 - |dot(new, old)|)
        let mut max_diff = 0.0f64;
        for i in 0..3 {
            let new_col = Vector3::new(w_mat[(0, i)], w_mat[(1, i)], w_mat[(2, i)]);
            let old_col = Vector3::new(w_old[(0, i)], w_old[(1, i)], w_old[(2, i)]);
            let diff = 1.0 - new_col.dot(&old_col).abs();
            max_diff = max_diff.max(diff);
        }

        if max_diff < tol {
            converged = true;
            break;
        }
    }

    // Step 6: compute separated signals: S = W^T * whitened_pixel
    // which is equivalent to S = W^T * whitening * (original - mean)
    let mut separated = Array3::<f64>::zeros((h, w, 3));
    for &(r0, r1) in &tiles {
        for r in r0..r1 {
            for x_pos in 0..w {
                let x = Vector3::new(
                    whitened[[r, x_pos, 0]],
                    whitened[[r, x_pos, 1]],
                    whitened[[r, x_pos, 2]],
                );
                let s = w_mat.transpose() * x;
                separated[[r, x_pos, 0]] = s[0];
                separated[[r, x_pos, 1]] = s[1];
                separated[[r, x_pos, 2]] = s[2];
            }
        }
    }

    // Step 7: resolve permutation and sign
    let perm = resolve_permutation(img, &separated);

    // Apply permutation
    let mut permuted = Array3::<f64>::zeros((h, w, 3));
    for r in 0..h {
        for x_pos in 0..w {
            for ch in 0..3 {
                permuted[[r, x_pos, ch]] = separated[[r, x_pos, perm[ch]]];
            }
        }
    }

    // Resolve signs: if mean of assigned separated component is negative, flip
    let mut signs = [1.0f64; 3];
    for ch in 0..3 {
        let mut channel_sum = 0.0;
        for r in 0..h {
            for x_pos in 0..w {
                channel_sum += permuted[[r, x_pos, ch]];
            }
        }
        if channel_sum < 0.0 {
            signs[ch] = -1.0;
            for r in 0..h {
                for x_pos in 0..w {
                    permuted[[r, x_pos, ch]] *= -1.0;
                }
            }
        }
    }

    IcaResult {
        separated: permuted,
        converged,
        iterations,
        unmixing: w_mat,
        permutation: perm,
        signs,
    }
}

/// Resolve permutation between original and separated channels.
///
/// Computes a 3x3 absolute correlation matrix, then brute-forces all 6
/// permutations to find the one maximizing the sum of diagonal correlations.
/// perm[output_ch] = separated_ch.
pub fn resolve_permutation(original: &Array3<f64>, separated: &Array3<f64>) -> [usize; 3] {
    let (h, w, _) = original.dim();
    let n = (h * w) as f64;

    // Compute means
    let mut mean_o = [0.0f64; 3];
    let mut mean_s = [0.0f64; 3];
    for r in 0..h {
        for c in 0..w {
            for ch in 0..3 {
                mean_o[ch] += original[[r, c, ch]];
                mean_s[ch] += separated[[r, c, ch]];
            }
        }
    }
    for ch in 0..3 {
        mean_o[ch] /= n;
        mean_s[ch] /= n;
    }

    // Compute variances and cross-covariances
    let mut var_o = [0.0f64; 3];
    let mut var_s = [0.0f64; 3];
    let mut cross = [[0.0f64; 3]; 3]; // cross[o_ch][s_ch]
    for r in 0..h {
        for c in 0..w {
            for o_ch in 0..3 {
                let do_ = original[[r, c, o_ch]] - mean_o[o_ch];
                var_o[o_ch] += do_ * do_;
                for s_ch in 0..3 {
                    let ds = separated[[r, c, s_ch]] - mean_s[s_ch];
                    cross[o_ch][s_ch] += do_ * ds;
                }
            }
            for s_ch in 0..3 {
                let ds = separated[[r, c, s_ch]] - mean_s[s_ch];
                var_s[s_ch] += ds * ds;
            }
        }
    }

    // Compute absolute correlation matrix
    let mut abs_corr = [[0.0f64; 3]; 3];
    for o_ch in 0..3 {
        for s_ch in 0..3 {
            let denom = (var_o[o_ch] * var_s[s_ch]).sqrt();
            if denom < 1e-15 {
                abs_corr[o_ch][s_ch] = 0.0;
            } else {
                abs_corr[o_ch][s_ch] = (cross[o_ch][s_ch] / denom).abs();
            }
        }
    }

    // Brute-force all 6 permutations
    let perms: [[usize; 3]; 6] = [
        [0, 1, 2], [0, 2, 1], [1, 0, 2],
        [1, 2, 0], [2, 0, 1], [2, 1, 0],
    ];

    let mut best_perm = [0usize; 3];
    let mut best_score = f64::NEG_INFINITY;

    for p in &perms {
        // perm[output_ch] = separated_ch
        // sum of abs_corr[output_ch][perm[output_ch]]
        let score = abs_corr[0][p[0]] + abs_corr[1][p[1]] + abs_corr[2][p[2]];
        if score > best_score {
            best_score = score;
            best_perm = *p;
        }
    }

    best_perm
}

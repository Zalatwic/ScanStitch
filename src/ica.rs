use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use ndarray::Array3;

use crate::streaming;

const HISTOGRAM_BINS: usize = 4096;
const ICA_LOW_PERCENTILE: f64 = 0.01;
const ICA_HIGH_PERCENTILE: f64 = 0.995;

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

#[derive(Debug, Clone)]
pub struct IcaChannelNormalization {
    pub min_value: f64,
    pub max_value: f64,
    pub low_percentile_value: f64,
    pub high_percentile_value: f64,
    pub clipped_low: u64,
    pub clipped_high: u64,
}

pub struct IcaDensityNormalization {
    pub normalized_density: Array3<f64>,
    pub channel_stats: [IcaChannelNormalization; 3],
    pub low_percentile: f64,
    pub high_percentile: f64,
    pub histogram_bins: usize,
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
///
/// The covariance already includes `WHITEN_REG`, so no pixel noise is added
/// here. Injecting random image-space noise would be amplified by the whitening
/// transform and would permanently contaminate the separated dye channels.
pub fn apply_whitening(img: &mut Array3<f64>, mean: &Vector3<f64>, whitening: &Matrix3<f64>) {
    let (h, w, _) = img.dim();
    let tiles = streaming::tile_ranges(h, streaming::DEFAULT_TILE_ROWS);
    for &(r0, r1) in &tiles {
        for r in r0..r1 {
            for c in 0..w {
                let d = Vector3::new(
                    img[[r, c, 0]] - mean[0],
                    img[[r, c, 1]] - mean[1],
                    img[[r, c, 2]] - mean[2],
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
    let mut w_mat = Matrix3::new(0.7, 0.5, 0.3, 0.3, 0.8, 0.4, 0.2, 0.3, 0.9);
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

    // Step 6: compute separated signals (raw independent components).
    // s = W^T * whitened_pixel gives the separated sources directly.
    // Do NOT apply whitening_inv here — it is a dense 3x3 matrix that would
    // remix the independent components, defeating the entire ICA separation.
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

    // Apply permutation (streaming)
    let mut permuted = Array3::<f64>::zeros((h, w, 3));
    for &(r0, r1) in &tiles {
        for r in r0..r1 {
            for x_pos in 0..w {
                for ch in 0..3 {
                    permuted[[r, x_pos, ch]] = separated[[r, x_pos, perm[ch]]];
                }
            }
        }
    }

    // Resolve signs using skewness: physical dye density should be right-skewed
    // (most pixels near low-density film base, fewer at high-density image areas).
    // A negative skewness indicates the channel is inverted.

    // Streaming pass 1: per-channel sum
    let mut ch_sum = [0.0f64; 3];
    for &(r0, r1) in &tiles {
        let tile = streaming::tile_view(&permuted, r0, r1);
        let (th, tw, _) = tile.dim();
        for r in 0..th {
            for x_pos in 0..tw {
                for ch in 0..3 {
                    ch_sum[ch] += tile[[r, x_pos, ch]];
                }
            }
        }
    }
    let mut ch_mean = [0.0f64; 3];
    for ch in 0..3 {
        ch_mean[ch] = ch_sum[ch] / n;
    }

    // Streaming pass 2: second and third central moments for skewness
    let mut m2 = [0.0f64; 3];
    let mut m3 = [0.0f64; 3];
    for &(r0, r1) in &tiles {
        let tile = streaming::tile_view(&permuted, r0, r1);
        let (th, tw, _) = tile.dim();
        for r in 0..th {
            for x_pos in 0..tw {
                for ch in 0..3 {
                    let d = tile[[r, x_pos, ch]] - ch_mean[ch];
                    let d2 = d * d;
                    m2[ch] += d2;
                    m3[ch] += d2 * d;
                }
            }
        }
    }

    // Compute skewness and determine sign flips
    let mut signs = [1.0f64; 3];
    for ch in 0..3 {
        let variance = m2[ch] / n;
        let std_dev = variance.sqrt();
        if std_dev > 1e-15 {
            let skewness = (m3[ch] / n) / (std_dev * std_dev * std_dev);
            if skewness < 0.0 {
                signs[ch] = -1.0;
            }
        }
    }

    // Streaming pass 3: apply sign flips in zero-mean ICA space.
    // The separated components are centered, so a sign correction is simply
    // multiplication by -1. Any additive re-anchoring would introduce a large
    // channel-dependent bias and break the later global normalization.
    for &(r0, r1) in &tiles {
        for r in r0..r1 {
            for x_pos in 0..w {
                for ch in 0..3 {
                    permuted[[r, x_pos, ch]] *= signs[ch];
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
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
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

fn histogram_bin(value: f64, min_value: f64, max_value: f64) -> usize {
    let span = (max_value - min_value).max(1e-12);
    let t = ((value - min_value) / span).clamp(0.0, 1.0);
    ((t * (HISTOGRAM_BINS - 1) as f64).round() as usize).min(HISTOGRAM_BINS - 1)
}

fn percentile_from_histogram(hist: &[u64], min_value: f64, max_value: f64, percentile: f64) -> f64 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return min_value;
    }

    let target = ((total as f64 - 1.0) * percentile.clamp(0.0, 1.0)).round() as u64;
    let mut cumulative = 0u64;
    for (idx, count) in hist.iter().enumerate() {
        cumulative += *count;
        if cumulative > target {
            let span = (max_value - min_value).max(1e-12);
            let t = idx as f64 / (HISTOGRAM_BINS - 1) as f64;
            return min_value + t * span;
        }
    }

    max_value
}

/// Normalize separated ICA channels independently using robust full-image
/// percentiles so each dye channel remains in a density-like domain.
pub fn normalize_separated_density_channels(img: &Array3<f64>) -> IcaDensityNormalization {
    let (h, w, c) = img.dim();
    assert_eq!(c, 3, "Expected three ICA channels");

    let mut min_value = [f64::INFINITY; 3];
    let mut max_value = [f64::NEG_INFINITY; 3];
    for &(r0, r1) in &streaming::tile_ranges(h, streaming::DEFAULT_TILE_ROWS) {
        let tile = streaming::tile_view(img, r0, r1);
        let (th, tw, _) = tile.dim();
        for r in 0..th {
            for x in 0..tw {
                for ch in 0..3 {
                    let v = tile[[r, x, ch]];
                    min_value[ch] = min_value[ch].min(v);
                    max_value[ch] = max_value[ch].max(v);
                }
            }
        }
    }

    for ch in 0..3 {
        if !min_value[ch].is_finite() || !max_value[ch].is_finite() {
            min_value[ch] = 0.0;
            max_value[ch] = 0.0;
        }
    }

    let mut histograms = vec![vec![0u64; HISTOGRAM_BINS]; 3];
    for &(r0, r1) in &streaming::tile_ranges(h, streaming::DEFAULT_TILE_ROWS) {
        let tile = streaming::tile_view(img, r0, r1);
        let (th, tw, _) = tile.dim();
        for r in 0..th {
            for x in 0..tw {
                for ch in 0..3 {
                    let v = tile[[r, x, ch]];
                    histograms[ch][histogram_bin(v, min_value[ch], max_value[ch])] += 1;
                }
            }
        }
    }

    let low_value: [f64; 3] = std::array::from_fn(|ch| {
        percentile_from_histogram(
            &histograms[ch],
            min_value[ch],
            max_value[ch],
            ICA_LOW_PERCENTILE,
        )
    });
    let high_value: [f64; 3] = std::array::from_fn(|ch| {
        percentile_from_histogram(
            &histograms[ch],
            min_value[ch],
            max_value[ch],
            ICA_HIGH_PERCENTILE,
        )
    });

    let mut normalized_density = Array3::<f64>::zeros((h, w, 3));
    let mut clipped_low = [0u64; 3];
    let mut clipped_high = [0u64; 3];
    for &(r0, r1) in &streaming::tile_ranges(h, streaming::DEFAULT_TILE_ROWS) {
        for r in r0..r1 {
            for x in 0..w {
                for ch in 0..3 {
                    let lo = low_value[ch];
                    let hi = high_value[ch];
                    let span = (hi - lo).max(1e-12);
                    let v = img[[r, x, ch]];
                    if v <= lo {
                        clipped_low[ch] += 1;
                    }
                    if v >= hi {
                        clipped_high[ch] += 1;
                    }
                    normalized_density[[r, x, ch]] = ((v - lo) / span).clamp(0.0, 1.0);
                }
            }
        }
    }

    IcaDensityNormalization {
        normalized_density,
        channel_stats: std::array::from_fn(|ch| IcaChannelNormalization {
            min_value: min_value[ch],
            max_value: max_value[ch],
            low_percentile_value: low_value[ch],
            high_percentile_value: high_value[ch],
            clipped_low: clipped_low[ch],
            clipped_high: clipped_high[ch],
        }),
        low_percentile: ICA_LOW_PERCENTILE,
        high_percentile: ICA_HIGH_PERCENTILE,
        histogram_bins: HISTOGRAM_BINS,
    }
}

use ndarray::Array3;

/// Parameters for the Naka-Rushton / modified Michaelis-Menten tone curve.
pub struct ToneCurveParams {
    /// Input value that maps to ~0.5 output (sigma in the formula).
    pub midpoint: f64,
    /// Contrast / steepness (exponent n in the formula).
    pub slope: f64,
    /// Minimum output value (toe).
    pub toe_lift: f64,
    /// Maximum output value (shoulder).
    pub shoulder_max: f64,
}

impl Default for ToneCurveParams {
    fn default() -> Self {
        Self {
            midpoint: 0.5,
            slope: 5.0,
            toe_lift: 0.005,
            shoulder_max: 0.995,
        }
    }
}

/// Apply the Naka-Rushton tone curve to a single value.
///
/// Formula: x^n / (x^n + sigma^n), rescaled to [toe_lift, shoulder_max].
pub fn apply_tone_curve(x: f64, params: &ToneCurveParams) -> f64 {
    let x = x.clamp(0.0, 1.0);
    let n = params.slope;
    let sigma = params.midpoint;

    let xn = x.powf(n);
    let sn = sigma.powf(n);
    let raw = xn / (xn + sn);

    // Rescale from [0, 1) to [toe_lift, shoulder_max]
    params.toe_lift + raw * (params.shoulder_max - params.toe_lift)
}

/// Fit tone curve parameters from an image by analyzing its luminance histogram.
pub fn fit_tone_params(img: &Array3<f64>) -> ToneCurveParams {
    let shape = img.shape();
    let h = shape[0];
    let w = shape[1];

    // Build luminance histogram with 1000 bins over [0, 1]
    const NUM_BINS: usize = 1000;
    let mut histogram = vec![0u64; NUM_BINS];
    let mut total_pixels = 0u64;

    for y in 0..h {
        for x in 0..w {
            let r = img[[y, x, 0]];
            let g = img[[y, x, 1]];
            let b = img[[y, x, 2]];
            // Standard luminance weights
            let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let lum = lum.clamp(0.0, 1.0);
            let bin = ((lum * NUM_BINS as f64) as usize).min(NUM_BINS - 1);
            histogram[bin] += 1;
            total_pixels += 1;
        }
    }

    if total_pixels == 0 {
        return ToneCurveParams::default();
    }

    // Find percentiles from the cumulative histogram
    let p05_target = (total_pixels as f64 * 0.05) as u64;
    let p50_target = (total_pixels as f64 * 0.50) as u64;
    let p95_target = (total_pixels as f64 * 0.95) as u64;

    let mut cumulative = 0u64;
    let mut p05 = 0.0f64;
    let mut p50 = 0.0f64;
    let mut p95 = 0.0f64;
    let mut found_p05 = false;
    let mut found_p50 = false;
    let mut found_p95 = false;

    for (i, &count) in histogram.iter().enumerate() {
        cumulative += count;
        let value = (i as f64 + 0.5) / NUM_BINS as f64;
        if !found_p05 && cumulative >= p05_target {
            p05 = value;
            found_p05 = true;
        }
        if !found_p50 && cumulative >= p50_target {
            p50 = value;
            found_p50 = true;
        }
        if !found_p95 && cumulative >= p95_target {
            p95 = value;
            found_p95 = true;
        }
    }

    let midpoint = p50.clamp(0.1, 0.9);
    let dynamic_range = p95 - p05;
    let slope = if dynamic_range > 0.0 {
        (2.5 / dynamic_range).clamp(2.0, 8.0)
    } else {
        5.0
    };

    ToneCurveParams {
        midpoint,
        slope,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    }
}

/// Apply tone mapping to an image using automatically fitted parameters.
pub fn apply_tonemap(img: &Array3<f64>) -> Array3<f64> {
    let params = fit_tone_params(img);
    apply_tonemap_with_params(img, &params)
}

/// Apply tone mapping to an image with explicit parameters.
pub fn apply_tonemap_with_params(img: &Array3<f64>, params: &ToneCurveParams) -> Array3<f64> {
    let shape = img.shape();
    let h = shape[0];
    let w = shape[1];
    let c = shape[2];

    let mut result = Array3::<f64>::zeros((h, w, c));

    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                result[[y, x, ch]] = apply_tone_curve(img[[y, x, ch]], params);
            }
        }
    }

    result
}

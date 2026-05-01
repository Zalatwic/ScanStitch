use ndarray::parallel::prelude::*;
use ndarray::{Array3, Axis};
use std::sync::atomic::{AtomicU64, Ordering};

const PROPHOTO_LUMA: [f64; 3] = [0.2880, 0.7119, 0.0001];
const NUM_BINS: usize = 1000;
const PERCEPTUAL_LUMINANCE_GAIN: f64 = 15.0;
const LINEAR_FIT_MEDIAN_THRESHOLD: f64 = 0.25;
const LINEAR_FIT_SLOPE_SCALE: f64 = 1.6;
const PERCEPTUAL_FIT_SLOPE_SCALE: f64 = 2.5;
const HIGHLIGHT_NEUTRAL_CHROMA_START_LUMINANCE: f64 = 0.72;
const HIGHLIGHT_NEUTRAL_CHROMA_FULL_LUMINANCE: f64 = 0.92;
const HIGHLIGHT_NEUTRAL_CHROMA_MIN_SCALE: f64 = 0.40;
const HIGHLIGHT_NEUTRAL_CHROMA_MAX_SATURATION: f64 = 0.35;
const SHADOW_CHROMA_START_LUMINANCE: f64 = 0.24;
const SHADOW_CHROMA_FULL_LUMINANCE: f64 = 0.08;
const SHADOW_CHROMA_MIN_SCALE: f64 = 0.35;
const RENDER_QUALITY_MAX_SAMPLES: usize = 1_000_000;
const RENDER_SHADOW_BAND_FRACTION: f64 = 0.10;
const RENDER_MIDTONE_BAND_LOW_FRACTION: f64 = 0.40;
const RENDER_MIDTONE_BAND_HIGH_FRACTION: f64 = 0.60;
const RENDER_BRIGHT_BAND_FRACTION: f64 = 0.10;
const RENDER_BRIGHT_NEUTRAL_MAX_SATURATION: f64 = HIGHLIGHT_NEUTRAL_CHROMA_MAX_SATURATION;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneFitDomain {
    LinearLuminance,
    Log2CompressedLuminance,
}

impl ToneFitDomain {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LinearLuminance => "linear_luminance",
            Self::Log2CompressedLuminance => "log2_compressed_luminance",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToneCurveDiagnostics {
    pub fit_domain: &'static str,
    pub perceptual_luminance_gain: f64,
    pub input_linear_percentiles: [f64; 3],
    pub input_perceptual_percentiles: [f64; 3],
    pub mapped_linear_percentiles: [f64; 3],
    pub mapped_perceptual_percentiles: [f64; 3],
}

pub struct ToneFitResult {
    pub params: ToneCurveParams,
    pub diagnostics: ToneCurveDiagnostics,
}

#[derive(Debug, Clone)]
pub struct TonemapApplyDiagnostics {
    pub highlight_chroma_compressed_ratio: f64,
    pub highlight_neutral_chroma_compressed_ratio: f64,
    pub highlight_neutral_chroma_start_luminance: f64,
    pub highlight_neutral_chroma_full_luminance: f64,
    pub highlight_neutral_chroma_min_scale: f64,
    pub highlight_neutral_chroma_max_saturation: f64,
    pub shadow_chroma_compressed_ratio: f64,
    pub shadow_chroma_start_luminance: f64,
    pub shadow_chroma_full_luminance: f64,
    pub shadow_chroma_min_scale: f64,
    pub pre_chroma_compression_clipped_high_ratio: [f64; 3],
    pub post_chroma_compression_clipped_high_ratio: [f64; 3],
    pub post_chroma_compression_clipped_low_ratio: [f64; 3],
}

pub struct TonemapApplyResult {
    pub image: Array3<f64>,
    pub diagnostics: TonemapApplyDiagnostics,
}

#[derive(Debug, Clone)]
pub struct RenderBandDiagnostics {
    pub pixel_count: usize,
    pub luminance_percentiles: [f64; 3],
    pub saturation_median: f64,
    pub saturation_p95: f64,
    pub rgb_median: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct RenderQualityDiagnostics {
    pub sample_count: usize,
    pub sample_stride: usize,
    pub shadow_luminance_max: f64,
    pub midtone_luminance_range: [f64; 2],
    pub bright_neutral_luminance_min: f64,
    pub bright_neutral_max_saturation: f64,
    pub shadow: RenderBandDiagnostics,
    pub midtone: RenderBandDiagnostics,
    pub bright_neutral: RenderBandDiagnostics,
    pub bright_saturated: RenderBandDiagnostics,
}

#[derive(Debug, Clone, Copy)]
struct RenderQualityPixel {
    luminance: f64,
    saturation: f64,
    rgb: [f64; 3],
}

/// Parameters for the Naka-Rushton / modified Michaelis-Menten tone curve.
#[derive(Debug, Clone, Copy)]
pub struct ToneCurveParams {
    /// Luminance domain the curve is fit and applied in.
    pub domain: ToneFitDomain,
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
            domain: ToneFitDomain::Log2CompressedLuminance,
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
    let raw = (xn / (xn + sn)) * (1.0 + sn);

    // Rescale from [0, 1) to [toe_lift, shoulder_max]
    params.toe_lift + raw * (params.shoulder_max - params.toe_lift)
}

fn linear_luminance(r: f64, g: f64, b: f64) -> f64 {
    (PROPHOTO_LUMA[0] * r + PROPHOTO_LUMA[1] * g + PROPHOTO_LUMA[2] * b).clamp(0.0, 1.0)
}

fn encode_perceptual_luminance(lum: f64) -> f64 {
    let gain = PERCEPTUAL_LUMINANCE_GAIN;
    ((1.0 + gain * lum.clamp(0.0, 1.0)).log2()) / (1.0 + gain).log2()
}

fn decode_perceptual_luminance(encoded: f64) -> f64 {
    let gain = PERCEPTUAL_LUMINANCE_GAIN;
    (2.0f64.powf(encoded.clamp(0.0, 1.0) * (1.0 + gain).log2()) - 1.0) / gain
}

fn compress_highlight_chroma(
    rgb: [f64; 3],
    mapped_lum: f64,
    gamut_ceiling: f64,
) -> ([f64; 3], bool) {
    if rgb.iter().all(|v| *v >= 0.0 && *v <= 1.0) {
        return (rgb, false);
    }

    let neutral = mapped_lum.clamp(0.0, 1.0);
    let upper = gamut_ceiling.clamp(neutral, 1.0);
    let mut chroma_scale = 1.0f64;
    for value in rgb {
        let delta = value - neutral;
        if value > 1.0 && delta > 1e-12 {
            chroma_scale = chroma_scale.min((upper - neutral) / delta);
        } else if value < 0.0 && delta < -1e-12 {
            chroma_scale = chroma_scale.min((0.0 - neutral) / delta);
        }
    }

    if !chroma_scale.is_finite() {
        chroma_scale = 0.0;
    }
    let chroma_scale = chroma_scale.clamp(0.0, 1.0);
    let compressed = std::array::from_fn(|c| neutral + (rgb[c] - neutral) * chroma_scale);
    (compressed, chroma_scale < 1.0 - 1e-12)
}

fn smoothstep01(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn rgb_saturation(rgb: [f64; 3]) -> f64 {
    let max_v = rgb.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_v <= 1e-12 {
        return 0.0;
    }
    let min_v = rgb.iter().copied().fold(f64::INFINITY, f64::min);
    ((max_v - min_v) / max_v).clamp(0.0, 1.0)
}

fn percentile_from_sorted_values(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn render_band_diagnostics(band: &[RenderQualityPixel]) -> RenderBandDiagnostics {
    if band.is_empty() {
        return RenderBandDiagnostics {
            pixel_count: 0,
            luminance_percentiles: [0.0; 3],
            saturation_median: 0.0,
            saturation_p95: 0.0,
            rgb_median: [0.0; 3],
        };
    }

    let mut luminance = Vec::with_capacity(band.len());
    let mut saturation = Vec::with_capacity(band.len());
    let mut rgb = [
        Vec::with_capacity(band.len()),
        Vec::with_capacity(band.len()),
        Vec::with_capacity(band.len()),
    ];
    for pixel in band {
        luminance.push(pixel.luminance);
        saturation.push(pixel.saturation);
        for c in 0..3 {
            rgb[c].push(pixel.rgb[c]);
        }
    }
    luminance.sort_by(|a, b| a.total_cmp(b));
    saturation.sort_by(|a, b| a.total_cmp(b));
    for channel in &mut rgb {
        channel.sort_by(|a, b| a.total_cmp(b));
    }

    RenderBandDiagnostics {
        pixel_count: band.len(),
        luminance_percentiles: [
            percentile_from_sorted_values(&luminance, 0.05),
            percentile_from_sorted_values(&luminance, 0.50),
            percentile_from_sorted_values(&luminance, 0.95),
        ],
        saturation_median: percentile_from_sorted_values(&saturation, 0.50),
        saturation_p95: percentile_from_sorted_values(&saturation, 0.95),
        rgb_median: std::array::from_fn(|c| percentile_from_sorted_values(&rgb[c], 0.50)),
    }
}

/// Measure rendered-output luminance and saturation bands for quality review.
///
/// Bands are percentile-based so exposure shifts do not move the target sample
/// populations: shadows are the darkest 10%, midtones are the 40-60% luminance
/// band, and bright candidates are the brightest 10%. Bright candidates are
/// then split by saturation so neutral color-cast diagnostics do not pressure
/// intentionally saturated highlights.
pub fn render_quality_diagnostics(img: &Array3<f64>) -> RenderQualityDiagnostics {
    let shape = img.shape();
    let h = shape[0];
    let w = shape[1];
    let total_pixels = h.saturating_mul(w);
    let sample_stride = if total_pixels <= RENDER_QUALITY_MAX_SAMPLES {
        1
    } else {
        total_pixels.div_ceil(RENDER_QUALITY_MAX_SAMPLES)
    };

    let mut pixels = Vec::<RenderQualityPixel>::with_capacity(
        total_pixels
            .saturating_add(sample_stride.saturating_sub(1))
            .checked_div(sample_stride.max(1))
            .unwrap_or(0),
    );
    for y in 0..h {
        for x in 0..w {
            let pixel_index = y * w + x;
            if pixel_index % sample_stride != 0 {
                continue;
            }
            let rgb = [
                img[[y, x, 0]].clamp(0.0, 1.0),
                img[[y, x, 1]].clamp(0.0, 1.0),
                img[[y, x, 2]].clamp(0.0, 1.0),
            ];
            pixels.push(RenderQualityPixel {
                luminance: linear_luminance(rgb[0], rgb[1], rgb[2]),
                saturation: rgb_saturation(rgb),
                rgb,
            });
        }
    }
    pixels.sort_by(|a, b| a.luminance.total_cmp(&b.luminance));

    if pixels.is_empty() {
        let empty = render_band_diagnostics(&[]);
        return RenderQualityDiagnostics {
            sample_count: 0,
            sample_stride,
            shadow_luminance_max: 0.0,
            midtone_luminance_range: [0.0, 0.0],
            bright_neutral_luminance_min: 0.0,
            bright_neutral_max_saturation: RENDER_BRIGHT_NEUTRAL_MAX_SATURATION,
            shadow: empty.clone(),
            midtone: empty.clone(),
            bright_neutral: empty.clone(),
            bright_saturated: empty,
        };
    }

    let n = pixels.len();
    let shadow_end = ((n as f64 * RENDER_SHADOW_BAND_FRACTION).ceil() as usize).clamp(1, n);
    let midtone_start = ((n as f64 * RENDER_MIDTONE_BAND_LOW_FRACTION).floor() as usize).min(n - 1);
    let midtone_end = ((n as f64 * RENDER_MIDTONE_BAND_HIGH_FRACTION).ceil() as usize)
        .clamp(midtone_start + 1, n);
    let bright_start =
        ((n as f64 * (1.0 - RENDER_BRIGHT_BAND_FRACTION)).floor() as usize).min(n - 1);
    let mut bright_neutral = Vec::<RenderQualityPixel>::new();
    let mut bright_saturated = Vec::<RenderQualityPixel>::new();
    for pixel in &pixels[bright_start..] {
        if pixel.saturation <= RENDER_BRIGHT_NEUTRAL_MAX_SATURATION {
            bright_neutral.push(*pixel);
        } else {
            bright_saturated.push(*pixel);
        }
    }

    RenderQualityDiagnostics {
        sample_count: n,
        sample_stride,
        shadow_luminance_max: pixels[shadow_end - 1].luminance,
        midtone_luminance_range: [
            pixels[midtone_start].luminance,
            pixels[midtone_end - 1].luminance,
        ],
        bright_neutral_luminance_min: pixels[bright_start].luminance,
        bright_neutral_max_saturation: RENDER_BRIGHT_NEUTRAL_MAX_SATURATION,
        shadow: render_band_diagnostics(&pixels[..shadow_end]),
        midtone: render_band_diagnostics(&pixels[midtone_start..midtone_end]),
        bright_neutral: render_band_diagnostics(&bright_neutral),
        bright_saturated: render_band_diagnostics(&bright_saturated),
    }
}

fn compress_highlight_neutral_chroma(rgb: [f64; 3], mapped_lum: f64) -> ([f64; 3], bool) {
    if mapped_lum < HIGHLIGHT_NEUTRAL_CHROMA_START_LUMINANCE {
        return (rgb, false);
    }

    if rgb_saturation(rgb) > HIGHLIGHT_NEUTRAL_CHROMA_MAX_SATURATION {
        return (rgb, false);
    }

    let span = (HIGHLIGHT_NEUTRAL_CHROMA_FULL_LUMINANCE - HIGHLIGHT_NEUTRAL_CHROMA_START_LUMINANCE)
        .max(1e-12);
    let t = (mapped_lum - HIGHLIGHT_NEUTRAL_CHROMA_START_LUMINANCE) / span;
    let strength = smoothstep01(t);
    if strength <= 1e-12 {
        return (rgb, false);
    }

    let chroma_scale = 1.0 - strength * (1.0 - HIGHLIGHT_NEUTRAL_CHROMA_MIN_SCALE);
    let neutral = mapped_lum.clamp(0.0, 1.0);
    let compressed = std::array::from_fn(|c| neutral + (rgb[c] - neutral) * chroma_scale);
    (compressed, chroma_scale < 1.0 - 1e-12)
}

fn compress_shadow_chroma(rgb: [f64; 3], mapped_lum: f64) -> ([f64; 3], bool) {
    if mapped_lum >= SHADOW_CHROMA_START_LUMINANCE {
        return (rgb, false);
    }

    let span = (SHADOW_CHROMA_START_LUMINANCE - SHADOW_CHROMA_FULL_LUMINANCE).max(1e-12);
    let t = (mapped_lum - SHADOW_CHROMA_FULL_LUMINANCE) / span;
    let strength = (1.0 - smoothstep01(t)).sqrt();
    if strength <= 1e-12 {
        return (rgb, false);
    }

    let chroma_scale = 1.0 - strength * (1.0 - SHADOW_CHROMA_MIN_SCALE);
    let neutral = mapped_lum.clamp(0.0, 1.0);
    let compressed = std::array::from_fn(|c| neutral + (rgb[c] - neutral) * chroma_scale);
    (compressed, chroma_scale < 1.0 - 1e-12)
}

fn histogram_percentiles(histogram: &[u64], total_pixels: u64) -> [f64; 3] {
    let targets = [0.05, 0.50, 0.95];
    let mut values = [0.0f64; 3];
    let mut found = [false; 3];
    let mut cumulative = 0u64;

    for (i, &count) in histogram.iter().enumerate() {
        cumulative += count;
        let value = (i as f64 + 0.5) / NUM_BINS as f64;
        for (idx, percentile) in targets.iter().enumerate() {
            if !found[idx] && cumulative >= (total_pixels as f64 * percentile) as u64 {
                values[idx] = value;
                found[idx] = true;
            }
        }
    }

    values
}

/// Fit tone curve parameters from an image by analyzing a perceptually compressed
/// luminance histogram derived from linear ProPhoto RGB.
pub fn fit_tone_params_with_diagnostics(img: &Array3<f64>) -> ToneFitResult {
    let shape = img.shape();
    let h = shape[0];
    let w = shape[1];

    let mut linear_histogram = vec![0u64; NUM_BINS];
    let mut perceptual_histogram = vec![0u64; NUM_BINS];
    let mut total_pixels = 0u64;

    for y in 0..h {
        for x in 0..w {
            let r = img[[y, x, 0]];
            let g = img[[y, x, 1]];
            let b = img[[y, x, 2]];
            let linear_lum = linear_luminance(r, g, b);
            let perceptual_lum = encode_perceptual_luminance(linear_lum);
            let linear_bin = ((linear_lum * NUM_BINS as f64) as usize).min(NUM_BINS - 1);
            let perceptual_bin = ((perceptual_lum * NUM_BINS as f64) as usize).min(NUM_BINS - 1);
            linear_histogram[linear_bin] += 1;
            perceptual_histogram[perceptual_bin] += 1;
            total_pixels += 1;
        }
    }

    if total_pixels == 0 {
        return ToneFitResult {
            params: ToneCurveParams::default(),
            diagnostics: ToneCurveDiagnostics {
                fit_domain: ToneFitDomain::Log2CompressedLuminance.as_str(),
                perceptual_luminance_gain: PERCEPTUAL_LUMINANCE_GAIN,
                input_linear_percentiles: [0.0; 3],
                input_perceptual_percentiles: [0.0; 3],
                mapped_linear_percentiles: [0.0; 3],
                mapped_perceptual_percentiles: [0.0; 3],
            },
        };
    }

    let input_linear_percentiles = histogram_percentiles(&linear_histogram, total_pixels);
    let input_perceptual_percentiles = histogram_percentiles(&perceptual_histogram, total_pixels);
    let domain = if input_linear_percentiles[1] >= LINEAR_FIT_MEDIAN_THRESHOLD {
        ToneFitDomain::LinearLuminance
    } else {
        ToneFitDomain::Log2CompressedLuminance
    };

    let (midpoint, slope) = match domain {
        ToneFitDomain::LinearLuminance => {
            let dynamic_range = input_linear_percentiles[2] - input_linear_percentiles[0];
            let slope = if dynamic_range > 0.0 {
                (LINEAR_FIT_SLOPE_SCALE / dynamic_range).clamp(2.0, 6.0)
            } else {
                5.0
            };
            (input_linear_percentiles[1].clamp(0.1, 0.9), slope)
        }
        ToneFitDomain::Log2CompressedLuminance => {
            let dynamic_range = input_perceptual_percentiles[2] - input_perceptual_percentiles[0];
            let slope = if dynamic_range > 0.0 {
                (PERCEPTUAL_FIT_SLOPE_SCALE / dynamic_range).clamp(2.0, 8.0)
            } else {
                5.0
            };
            (input_perceptual_percentiles[1].clamp(0.1, 0.9), slope)
        }
    };

    let params = ToneCurveParams {
        domain,
        midpoint,
        slope,
        toe_lift: 0.005,
        shoulder_max: 0.995,
    };
    let (mapped_linear_percentiles, mapped_perceptual_percentiles) = match domain {
        ToneFitDomain::LinearLuminance => {
            let linear =
                std::array::from_fn(|idx| apply_tone_curve(input_linear_percentiles[idx], &params));
            let perceptual = std::array::from_fn(|idx| encode_perceptual_luminance(linear[idx]));
            (linear, perceptual)
        }
        ToneFitDomain::Log2CompressedLuminance => {
            let perceptual = std::array::from_fn(|idx| {
                apply_tone_curve(input_perceptual_percentiles[idx], &params)
            });
            let linear = std::array::from_fn(|idx| decode_perceptual_luminance(perceptual[idx]));
            (linear, perceptual)
        }
    };

    ToneFitResult {
        params,
        diagnostics: ToneCurveDiagnostics {
            fit_domain: domain.as_str(),
            perceptual_luminance_gain: PERCEPTUAL_LUMINANCE_GAIN,
            input_linear_percentiles,
            input_perceptual_percentiles,
            mapped_linear_percentiles,
            mapped_perceptual_percentiles,
        },
    }
}

pub fn fit_tone_params(img: &Array3<f64>) -> ToneCurveParams {
    fit_tone_params_with_diagnostics(img).params
}

/// Apply tone mapping to an image using automatically fitted parameters.
pub fn apply_tonemap(img: &Array3<f64>) -> Array3<f64> {
    let fit = fit_tone_params_with_diagnostics(img);
    apply_tonemap_with_params(img, &fit.params)
}

/// Apply tone mapping to an image with explicit parameters.
pub fn apply_tonemap_with_params(img: &Array3<f64>, params: &ToneCurveParams) -> Array3<f64> {
    apply_tonemap_with_params_and_diagnostics(img, params).image
}

pub fn apply_tonemap_with_params_and_diagnostics(
    img: &Array3<f64>,
    params: &ToneCurveParams,
) -> TonemapApplyResult {
    let shape = img.shape();
    let h = shape[0];
    let w = shape[1];
    let c = shape[2];

    let mut result = Array3::<f64>::zeros((h, w, c));
    let compressed_pixels = AtomicU64::new(0);
    let highlight_neutral_compressed_pixels = AtomicU64::new(0);
    let shadow_compressed_pixels = AtomicU64::new(0);
    let pre_high_clipped = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    let post_high_clipped = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    let post_low_clipped = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

    result
        .axis_chunks_iter_mut(Axis(0), 512)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut out_chunk)| {
            let row_start = chunk_idx * 512;
            let chunk_h = out_chunk.dim().0;
            for local_y in 0..chunk_h {
                let y = row_start + local_y;
                for x in 0..w {
                    let r = img[[y, x, 0]];
                    let g = img[[y, x, 1]];
                    let b = img[[y, x, 2]];
                    let linear_lum = linear_luminance(r, g, b);
                    let mapped_linear = match params.domain {
                        ToneFitDomain::LinearLuminance => apply_tone_curve(linear_lum, params),
                        ToneFitDomain::Log2CompressedLuminance => {
                            let perceptual_lum = encode_perceptual_luminance(linear_lum);
                            let mapped_perceptual = apply_tone_curve(perceptual_lum, params);
                            decode_perceptual_luminance(mapped_perceptual)
                        }
                    };
                    let scale = if linear_lum <= 1e-12 {
                        0.0
                    } else {
                        mapped_linear / linear_lum
                    };
                    let scaled = [r * scale, g * scale, b * scale];
                    for ch in 0..3 {
                        if scaled[ch] > 1.0 {
                            pre_high_clipped[ch].fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    let (mapped, compressed) =
                        compress_highlight_chroma(scaled, mapped_linear, params.shoulder_max);
                    if compressed {
                        compressed_pixels.fetch_add(1, Ordering::Relaxed);
                    }
                    let (mapped, highlight_neutral_compressed) =
                        compress_highlight_neutral_chroma(mapped, mapped_linear);
                    if highlight_neutral_compressed {
                        highlight_neutral_compressed_pixels.fetch_add(1, Ordering::Relaxed);
                    }
                    let (mapped, shadow_compressed) = compress_shadow_chroma(mapped, mapped_linear);
                    if shadow_compressed {
                        shadow_compressed_pixels.fetch_add(1, Ordering::Relaxed);
                    }

                    for ch in 0..3 {
                        if mapped[ch] > 1.0 {
                            post_high_clipped[ch].fetch_add(1, Ordering::Relaxed);
                        }
                        if mapped[ch] < 0.0 {
                            post_low_clipped[ch].fetch_add(1, Ordering::Relaxed);
                        }
                        out_chunk[[local_y, x, ch]] = mapped[ch].clamp(0.0, 1.0);
                    }
                }
            }
        });

    let total_pixels = (h * w).max(1) as f64;
    TonemapApplyResult {
        image: result,
        diagnostics: TonemapApplyDiagnostics {
            highlight_chroma_compressed_ratio: compressed_pixels.load(Ordering::Relaxed) as f64
                / total_pixels,
            highlight_neutral_chroma_compressed_ratio: highlight_neutral_compressed_pixels
                .load(Ordering::Relaxed)
                as f64
                / total_pixels,
            highlight_neutral_chroma_start_luminance: HIGHLIGHT_NEUTRAL_CHROMA_START_LUMINANCE,
            highlight_neutral_chroma_full_luminance: HIGHLIGHT_NEUTRAL_CHROMA_FULL_LUMINANCE,
            highlight_neutral_chroma_min_scale: HIGHLIGHT_NEUTRAL_CHROMA_MIN_SCALE,
            highlight_neutral_chroma_max_saturation: HIGHLIGHT_NEUTRAL_CHROMA_MAX_SATURATION,
            shadow_chroma_compressed_ratio: shadow_compressed_pixels.load(Ordering::Relaxed) as f64
                / total_pixels,
            shadow_chroma_start_luminance: SHADOW_CHROMA_START_LUMINANCE,
            shadow_chroma_full_luminance: SHADOW_CHROMA_FULL_LUMINANCE,
            shadow_chroma_min_scale: SHADOW_CHROMA_MIN_SCALE,
            pre_chroma_compression_clipped_high_ratio: std::array::from_fn(|ch| {
                pre_high_clipped[ch].load(Ordering::Relaxed) as f64 / total_pixels
            }),
            post_chroma_compression_clipped_high_ratio: std::array::from_fn(|ch| {
                post_high_clipped[ch].load(Ordering::Relaxed) as f64 / total_pixels
            }),
            post_chroma_compression_clipped_low_ratio: std::array::from_fn(|ch| {
                post_low_clipped[ch].load(Ordering::Relaxed) as f64 / total_pixels
            }),
        },
    }
}

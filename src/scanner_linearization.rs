use ndarray::parallel::prelude::*;
use ndarray::{Array3, Axis};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::constants::{MAX_14BIT, MAX_16BIT};
use crate::streaming;

const MIN_CONFIDENCE: f64 = 0.85;
const MIN_CURVE_POINTS: usize = 4;
const MIN_HELD_OUT_SAMPLES: usize = 12;
const MAX_HELD_OUT_RMSE: f64 = 0.01;
const MAX_HELD_OUT_MAX_ERROR: f64 = 0.03;
const MIN_IDENTITY_RMSE_IMPROVEMENT: f64 = 0.001;
const MAX_SIGNAL_NOISE_GAIN: f64 = 4.0;
const MIN_SHADING_GAIN: f64 = 0.5;
const MAX_SHADING_GAIN: f64 = 2.0;

fn zeros3() -> [f64; 3] {
    [0.0; 3]
}

fn ones3() -> [f64; 3] {
    [1.0; 3]
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScannerLinearizationValidation {
    pub held_out_sample_count: usize,
    pub transmittance_rmse: f64,
    pub transmittance_max_error: f64,
    pub identity_baseline_rmse: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScannerLinearizationCurves {
    /// Each point is `[level-corrected normalized scanner signal, linear transmittance]`.
    pub red: Vec<[f64; 2]>,
    pub green: Vec<[f64; 2]>,
    pub blue: Vec<[f64; 2]>,
}

impl ScannerLinearizationCurves {
    fn channels(&self) -> [&[[f64; 2]]; 3] {
        [&self.red, &self.green, &self.blue]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScannerLinearizationCalibration {
    pub model_id: String,
    pub curves: ScannerLinearizationCurves,
    #[serde(default = "zeros3")]
    pub black_level_normalized: [f64; 3],
    #[serde(default = "ones3")]
    pub white_level_normalized: [f64; 3],
    #[serde(default = "zeros3")]
    pub additive_flare_normalized: [f64; 3],
    /// Per-channel gain coefficients `[1, x, y, x², xy, y²]`, with x/y in [-1, 1].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shading_gain_polynomial: Option<[[f64; 6]; 3]>,
    pub confidence: f64,
    pub validation: ScannerLinearizationValidation,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScannerLinearizationDiagnostics {
    pub status: &'static str,
    pub model_id: String,
    pub source_bit_depth: u8,
    pub output_bit_depth: u8,
    pub black_level_normalized: [f64; 3],
    pub white_level_normalized: [f64; 3],
    pub additive_flare_normalized: [f64; 3],
    pub shading_gain_range: [[f64; 2]; 3],
    pub maximum_signal_noise_gain: f64,
    pub curve_extrapolated_low_samples: [u64; 3],
    pub curve_extrapolated_high_samples: [u64; 3],
    pub clipped_low_samples: [u64; 3],
    pub clipped_high_samples: [u64; 3],
    pub input_sample_count: u64,
    pub coordinate_domain: &'static str,
    pub orientation_tag: Option<u16>,
    pub scanner_frame_width: usize,
    pub scanner_frame_height: usize,
    pub interpolation: &'static str,
    pub confidence: f64,
    pub held_out_sample_count: usize,
    pub held_out_transmittance_rmse: f64,
    pub held_out_transmittance_max_error: f64,
    pub identity_baseline_rmse: f64,
    pub reason: String,
}

pub struct ScannerLinearizationResult {
    pub image: Array3<u16>,
    pub diagnostics: ScannerLinearizationDiagnostics,
}

/// Maps the already-oriented decoded image back to the original scanner pixel grid used when a
/// spatial shading calibration was measured.
#[derive(Debug, Clone, Copy)]
pub struct ScannerCoordinateMapping {
    pub orientation_tag: Option<u16>,
    pub scanner_frame_width: usize,
    pub scanner_frame_height: usize,
}

impl ScannerCoordinateMapping {
    pub fn identity(width: usize, height: usize) -> Self {
        Self {
            orientation_tag: Some(1),
            scanner_frame_width: width,
            scanner_frame_height: height,
        }
    }

    fn scanner_coordinates(self, output_x: usize, output_y: usize) -> (usize, usize) {
        let width = self.scanner_frame_width;
        let height = self.scanner_frame_height;
        match self.orientation_tag.unwrap_or(1) {
            2 => (width - 1 - output_x, output_y),
            3 => (width - 1 - output_x, height - 1 - output_y),
            4 => (output_x, height - 1 - output_y),
            5 => (output_y, output_x),
            6 => (output_y, height - 1 - output_x),
            7 => (width - 1 - output_y, height - 1 - output_x),
            8 => (width - 1 - output_y, output_x),
            _ => (output_x, output_y),
        }
    }

    /// Convert a possibly sub-pixel coordinate in the decoded, EXIF-oriented image into the
    /// normalized original scanner frame used by spatial calibration models.
    pub fn normalized_scanner_coordinates(
        self,
        output_x: f64,
        output_y: f64,
    ) -> Result<[f64; 2], String> {
        if !output_x.is_finite() || !output_y.is_finite() {
            return Err("oriented scanner coordinates must be finite".to_string());
        }
        let swaps_dimensions = matches!(self.orientation_tag, Some(5..=8));
        let (output_width, output_height) = if swaps_dimensions {
            (self.scanner_frame_height, self.scanner_frame_width)
        } else {
            (self.scanner_frame_width, self.scanner_frame_height)
        };
        if output_width == 0 || output_height == 0 {
            return Err("scanner coordinate mapping dimensions must be positive".to_string());
        }
        let maximum_x = output_width.saturating_sub(1) as f64;
        let maximum_y = output_height.saturating_sub(1) as f64;
        if !(0.0..=maximum_x).contains(&output_x) || !(0.0..=maximum_y).contains(&output_y) {
            return Err(format!(
                "oriented coordinate [{output_x:.3},{output_y:.3}] lies outside decoded image bounds [0,{maximum_x:.3}]x[0,{maximum_y:.3}]"
            ));
        }
        let width = self.scanner_frame_width as f64;
        let height = self.scanner_frame_height as f64;
        let (scanner_x, scanner_y) = match self.orientation_tag.unwrap_or(1) {
            2 => (width - 1.0 - output_x, output_y),
            3 => (width - 1.0 - output_x, height - 1.0 - output_y),
            4 => (output_x, height - 1.0 - output_y),
            5 => (output_y, output_x),
            6 => (output_y, height - 1.0 - output_x),
            7 => (width - 1.0 - output_y, height - 1.0 - output_x),
            8 => (width - 1.0 - output_y, output_x),
            _ => (output_x, output_y),
        };
        Ok([
            normalized_coordinate_f64(scanner_x, self.scanner_frame_width),
            normalized_coordinate_f64(scanner_y, self.scanner_frame_height),
        ])
    }

    fn validate_for_oriented_shape(self, width: usize, height: usize) -> Result<(), String> {
        let swaps_dimensions = matches!(self.orientation_tag, Some(5..=8));
        let expected = if swaps_dimensions {
            (self.scanner_frame_height, self.scanner_frame_width)
        } else {
            (self.scanner_frame_width, self.scanner_frame_height)
        };
        if (width, height) != expected {
            return Err(format!(
                "oriented image shape {}x{} is inconsistent with scanner frame {}x{} and EXIF orientation {:?}; expected {}x{}",
                width,
                height,
                self.scanner_frame_width,
                self.scanner_frame_height,
                self.orientation_tag,
                expected.0,
                expected.1
            ));
        }
        Ok(())
    }
}

fn normalized_coordinate_f64(coordinate: f64, extent: usize) -> f64 {
    if extent <= 1 {
        0.0
    } else {
        2.0 * coordinate / (extent - 1) as f64 - 1.0
    }
}

#[derive(Debug, Clone)]
struct MonotoneCurve {
    x: Vec<f64>,
    y: Vec<f64>,
    tangent: Vec<f64>,
}

impl MonotoneCurve {
    fn new(points: &[[f64; 2]]) -> Self {
        let x = points.iter().map(|point| point[0]).collect::<Vec<_>>();
        let y = points.iter().map(|point| point[1]).collect::<Vec<_>>();
        let h = x
            .windows(2)
            .map(|window| window[1] - window[0])
            .collect::<Vec<_>>();
        let delta = y
            .windows(2)
            .zip(&h)
            .map(|(window, h)| (window[1] - window[0]) / h)
            .collect::<Vec<_>>();
        let mut tangent = vec![0.0; points.len()];
        for index in 1..points.len() - 1 {
            let previous = delta[index - 1];
            let next = delta[index];
            if previous * next > 0.0 {
                let w1 = 2.0 * h[index] + h[index - 1];
                let w2 = h[index] + 2.0 * h[index - 1];
                tangent[index] = (w1 + w2) / (w1 / previous + w2 / next);
            }
        }
        let endpoint = |h0: f64, h1: f64, d0: f64, d1: f64| {
            let estimate = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
            if estimate.signum() != d0.signum() {
                0.0
            } else if d0.signum() != d1.signum() && estimate.abs() > 3.0 * d0.abs() {
                3.0 * d0
            } else {
                estimate
            }
        };
        tangent[0] = endpoint(h[0], h[1], delta[0], delta[1]);
        let last = points.len() - 1;
        tangent[last] = endpoint(h[last - 1], h[last - 2], delta[last - 1], delta[last - 2]);
        Self { x, y, tangent }
    }

    fn evaluate(&self, value: f64) -> (f64, bool, bool) {
        let last = self.x.len() - 1;
        if value <= self.x[0] {
            return (
                self.y[0] + self.tangent[0] * (value - self.x[0]),
                value < self.x[0],
                false,
            );
        }
        if value >= self.x[last] {
            return (
                self.y[last] + self.tangent[last] * (value - self.x[last]),
                false,
                value > self.x[last],
            );
        }
        let upper = self.x.partition_point(|x| *x <= value);
        let lower = upper - 1;
        let h = self.x[upper] - self.x[lower];
        let t = (value - self.x[lower]) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        (
            h00 * self.y[lower]
                + h10 * h * self.tangent[lower]
                + h01 * self.y[upper]
                + h11 * h * self.tangent[upper],
            false,
            false,
        )
    }
}

fn shading_gain(coefficients: &[f64; 6], x: f64, y: f64) -> f64 {
    coefficients[0]
        + coefficients[1] * x
        + coefficients[2] * y
        + coefficients[3] * x * x
        + coefficients[4] * x * y
        + coefficients[5] * y * y
}

fn shading_gain_ranges(calibration: &ScannerLinearizationCalibration) -> [[f64; 2]; 3] {
    std::array::from_fn(|channel| {
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        for yi in 0..=8 {
            for xi in 0..=8 {
                let x = -1.0 + 2.0 * xi as f64 / 8.0;
                let y = -1.0 + 2.0 * yi as f64 / 8.0;
                let gain = calibration
                    .shading_gain_polynomial
                    .as_ref()
                    .map(|coefficients| shading_gain(&coefficients[channel], x, y))
                    .unwrap_or(1.0);
                minimum = minimum.min(gain);
                maximum = maximum.max(gain);
            }
        }
        [minimum, maximum]
    })
}

fn maximum_noise_gain(calibration: &ScannerLinearizationCalibration) -> f64 {
    let shading_ranges = shading_gain_ranges(calibration);
    calibration
        .curves
        .channels()
        .iter()
        .enumerate()
        .map(|(channel, curve)| {
            let curve_gain = curve
                .windows(2)
                .map(|points| {
                    (points[1][1] - points[0][1]).abs()
                        / (points[1][0] - points[0][0]).abs().max(1e-12)
                })
                .fold(0.0, f64::max);
            let level_span = calibration.white_level_normalized[channel]
                - calibration.black_level_normalized[channel];
            curve_gain * shading_ranges[channel][1] / level_span.max(1e-12)
        })
        .fold(0.0, f64::max)
}

pub fn validate_scanner_linearization(
    calibration: &ScannerLinearizationCalibration,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if calibration.model_id.trim().is_empty() {
        errors.push("scanner_linearization.model_id must not be empty".to_string());
    }
    if !calibration.confidence.is_finite()
        || !(MIN_CONFIDENCE..=1.0).contains(&calibration.confidence)
    {
        errors.push(format!(
            "scanner_linearization.confidence must be finite and in [{MIN_CONFIDENCE:.2}, 1.0]"
        ));
    }
    for channel in 0..3 {
        let black = calibration.black_level_normalized[channel];
        let white = calibration.white_level_normalized[channel];
        let flare = calibration.additive_flare_normalized[channel];
        if !black.is_finite()
            || !white.is_finite()
            || !flare.is_finite()
            || black < 0.0
            || white > 1.0
            || flare < 0.0
            || black + flare >= white
        {
            errors.push(format!(
                "scanner_linearization channel {} requires finite 0<=black, 0<=flare, black+flare<white<=1",
                ["red", "green", "blue"][channel]
            ));
        }
    }
    for (channel, curve) in ["red", "green", "blue"]
        .into_iter()
        .zip(calibration.curves.channels())
    {
        if curve.len() < MIN_CURVE_POINTS {
            errors.push(format!(
                "scanner_linearization curve `{channel}` requires at least {MIN_CURVE_POINTS} points"
            ));
            continue;
        }
        if curve.iter().flatten().any(|value| !value.is_finite()) {
            errors.push(format!(
                "scanner_linearization curve `{channel}` contains non-finite values"
            ));
        }
        if curve
            .windows(2)
            .any(|points| points[1][0] <= points[0][0] || points[1][1] <= points[0][1])
        {
            errors.push(format!(
                "scanner_linearization curve `{channel}` must be strictly increasing in signal and transmittance"
            ));
        }
        if curve.first().is_some_and(|point| point[0] > 0.05)
            || curve.last().is_some_and(|point| point[0] < 0.95)
        {
            errors.push(format!(
                "scanner_linearization curve `{channel}` must cover normalized scanner signal from at most 0.05 through at least 0.95"
            ));
        }
    }
    let gain_ranges = shading_gain_ranges(calibration);
    for channel in 0..3 {
        if !gain_ranges[channel][0].is_finite()
            || !gain_ranges[channel][1].is_finite()
            || gain_ranges[channel][0] < MIN_SHADING_GAIN
            || gain_ranges[channel][1] > MAX_SHADING_GAIN
        {
            errors.push(format!(
                "scanner_linearization shading gain for channel {} must remain in [{MIN_SHADING_GAIN:.2}, {MAX_SHADING_GAIN:.2}] across the frame (observed {:.3}..{:.3})",
                ["red", "green", "blue"][channel],
                gain_ranges[channel][0],
                gain_ranges[channel][1]
            ));
        }
    }
    let validation = &calibration.validation;
    if validation.held_out_sample_count < MIN_HELD_OUT_SAMPLES {
        errors.push(format!(
            "scanner_linearization requires at least {MIN_HELD_OUT_SAMPLES} held-out samples"
        ));
    }
    if !validation.transmittance_rmse.is_finite()
        || validation.transmittance_rmse < 0.0
        || validation.transmittance_rmse > MAX_HELD_OUT_RMSE
    {
        errors.push(format!(
            "scanner_linearization held-out transmittance RMSE must be <= {MAX_HELD_OUT_RMSE:.3}"
        ));
    }
    if !validation.transmittance_max_error.is_finite()
        || validation.transmittance_max_error < 0.0
        || validation.transmittance_max_error > MAX_HELD_OUT_MAX_ERROR
    {
        errors.push(format!(
            "scanner_linearization held-out maximum transmittance error must be <= {MAX_HELD_OUT_MAX_ERROR:.3}"
        ));
    }
    if !validation.identity_baseline_rmse.is_finite()
        || validation.identity_baseline_rmse - validation.transmittance_rmse
            < MIN_IDENTITY_RMSE_IMPROVEMENT
    {
        errors.push(format!(
            "scanner_linearization must improve held-out RMSE over identity by at least {MIN_IDENTITY_RMSE_IMPROVEMENT:.3}"
        ));
    }
    let noise_gain = maximum_noise_gain(calibration);
    if !noise_gain.is_finite() || noise_gain > MAX_SIGNAL_NOISE_GAIN {
        errors.push(format!(
            "scanner_linearization maximum signal-noise gain {noise_gain:.3} exceeds {MAX_SIGNAL_NOISE_GAIN:.3}"
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn source_code_max(bit_depth: u8) -> f64 {
    match bit_depth {
        14 => MAX_14BIT,
        _ => MAX_16BIT,
    }
}

fn normalized_coordinate(index: usize, length: usize) -> f64 {
    if length <= 1 {
        0.0
    } else {
        -1.0 + 2.0 * index as f64 / (length - 1) as f64
    }
}

pub fn transform_base_color(
    base_color: [f64; 3],
    bit_depth: u8,
    calibration: &ScannerLinearizationCalibration,
) -> [f64; 3] {
    let curves: [MonotoneCurve; 3] = {
        let channels = calibration.curves.channels();
        std::array::from_fn(|channel| MonotoneCurve::new(channels[channel]))
    };
    let source_max = source_code_max(bit_depth);
    std::array::from_fn(|channel| {
        let signal = base_color[channel] / source_max;
        let corrected = (signal
            - calibration.black_level_normalized[channel]
            - calibration.additive_flare_normalized[channel])
            / (calibration.white_level_normalized[channel]
                - calibration.black_level_normalized[channel]);
        let (linear, _, _) = curves[channel].evaluate(corrected);
        let center_gain = calibration
            .shading_gain_polynomial
            .as_ref()
            .map(|coefficients| shading_gain(&coefficients[channel], 0.0, 0.0))
            .unwrap_or(1.0);
        (linear * center_gain).clamp(0.0, 1.0) * MAX_16BIT
    })
}

/// Evaluate one normalized scanner RGB sample at normalized frame coordinates. Calibration tools
/// use this for held-out scoring before the completed model passes the full runtime validator.
pub fn evaluate_scanner_signal(
    calibration: &ScannerLinearizationCalibration,
    signal: [f64; 3],
    x: f64,
    y: f64,
) -> Result<[f64; 3], String> {
    if signal.iter().any(|value| !value.is_finite()) || !x.is_finite() || !y.is_finite() {
        return Err("scanner signal and coordinates must be finite".to_string());
    }
    let channels = calibration.curves.channels();
    if channels.iter().any(|curve| curve.len() < 2) {
        return Err("each scanner linearization curve requires at least two points".to_string());
    }
    let curves: [MonotoneCurve; 3] =
        std::array::from_fn(|channel| MonotoneCurve::new(channels[channel]));
    Ok(std::array::from_fn(|channel| {
        let corrected = (signal[channel]
            - calibration.black_level_normalized[channel]
            - calibration.additive_flare_normalized[channel])
            / (calibration.white_level_normalized[channel]
                - calibration.black_level_normalized[channel]);
        let (linear, _, _) = curves[channel].evaluate(corrected);
        let gain = calibration
            .shading_gain_polynomial
            .as_ref()
            .map(|coefficients| shading_gain(&coefficients[channel], x, y))
            .unwrap_or(1.0);
        linear * gain
    }))
}

pub fn apply_scanner_linearization(
    image: Array3<u16>,
    bit_depth: u8,
    calibration: &ScannerLinearizationCalibration,
) -> Result<ScannerLinearizationResult, Vec<String>> {
    let (height, width, _) = image.dim();
    apply_scanner_linearization_with_coordinates(
        image,
        bit_depth,
        calibration,
        ScannerCoordinateMapping::identity(width, height),
    )
}

pub fn apply_scanner_linearization_with_coordinates(
    mut image: Array3<u16>,
    bit_depth: u8,
    calibration: &ScannerLinearizationCalibration,
    coordinate_mapping: ScannerCoordinateMapping,
) -> Result<ScannerLinearizationResult, Vec<String>> {
    validate_scanner_linearization(calibration)?;
    let (height, width, channels) = image.dim();
    if channels != 3 {
        return Err(vec![
            "scanner linearization requires a three-channel RGB image".to_string(),
        ]);
    }
    if let Err(error) = coordinate_mapping.validate_for_oriented_shape(width, height) {
        return Err(vec![error]);
    }
    let curve_channels = calibration.curves.channels();
    let curves: [MonotoneCurve; 3] =
        std::array::from_fn(|channel| MonotoneCurve::new(curve_channels[channel]));
    let source_max = source_code_max(bit_depth);
    let extrapolated_low: [AtomicU64; 3] = std::array::from_fn(|_| AtomicU64::new(0));
    let extrapolated_high: [AtomicU64; 3] = std::array::from_fn(|_| AtomicU64::new(0));
    let clipped_low: [AtomicU64; 3] = std::array::from_fn(|_| AtomicU64::new(0));
    let clipped_high: [AtomicU64; 3] = std::array::from_fn(|_| AtomicU64::new(0));

    image
        .axis_chunks_iter_mut(Axis(0), streaming::DEFAULT_TILE_ROWS)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_index, mut chunk)| {
            let y0 = chunk_index * streaming::DEFAULT_TILE_ROWS;
            for local_y in 0..chunk.dim().0 {
                let y = y0 + local_y;
                for x in 0..width {
                    let (scanner_x, scanner_y) = coordinate_mapping.scanner_coordinates(x, y);
                    let xn =
                        normalized_coordinate(scanner_x, coordinate_mapping.scanner_frame_width);
                    let yn =
                        normalized_coordinate(scanner_y, coordinate_mapping.scanner_frame_height);
                    for channel in 0..3 {
                        let signal = chunk[[local_y, x, channel]] as f64 / source_max;
                        let corrected = (signal
                            - calibration.black_level_normalized[channel]
                            - calibration.additive_flare_normalized[channel])
                            / (calibration.white_level_normalized[channel]
                                - calibration.black_level_normalized[channel]);
                        let (linear, low, high) = curves[channel].evaluate(corrected);
                        if low {
                            extrapolated_low[channel].fetch_add(1, Ordering::Relaxed);
                        }
                        if high {
                            extrapolated_high[channel].fetch_add(1, Ordering::Relaxed);
                        }
                        let gain = calibration
                            .shading_gain_polynomial
                            .as_ref()
                            .map(|coefficients| shading_gain(&coefficients[channel], xn, yn))
                            .unwrap_or(1.0);
                        let corrected = linear * gain;
                        if corrected < 0.0 {
                            clipped_low[channel].fetch_add(1, Ordering::Relaxed);
                        } else if corrected > 1.0 {
                            clipped_high[channel].fetch_add(1, Ordering::Relaxed);
                        }
                        chunk[[local_y, x, channel]] =
                            (corrected.clamp(0.0, 1.0) * MAX_16BIT).round() as u16;
                    }
                }
            }
        });

    let diagnostics = ScannerLinearizationDiagnostics {
        status: "applied",
        model_id: calibration.model_id.clone(),
        source_bit_depth: bit_depth,
        output_bit_depth: 16,
        black_level_normalized: calibration.black_level_normalized,
        white_level_normalized: calibration.white_level_normalized,
        additive_flare_normalized: calibration.additive_flare_normalized,
        shading_gain_range: shading_gain_ranges(calibration),
        maximum_signal_noise_gain: maximum_noise_gain(calibration),
        curve_extrapolated_low_samples: std::array::from_fn(|channel| {
            extrapolated_low[channel].load(Ordering::Relaxed)
        }),
        curve_extrapolated_high_samples: std::array::from_fn(|channel| {
            extrapolated_high[channel].load(Ordering::Relaxed)
        }),
        clipped_low_samples: std::array::from_fn(|channel| {
            clipped_low[channel].load(Ordering::Relaxed)
        }),
        clipped_high_samples: std::array::from_fn(|channel| {
            clipped_high[channel].load(Ordering::Relaxed)
        }),
        input_sample_count: height.saturating_mul(width) as u64,
        coordinate_domain: if matches!(coordinate_mapping.orientation_tag, Some(2..=8)) {
            "original_scanner_frame_via_inverse_exif_orientation"
        } else {
            "full_component_scanner_frame"
        },
        orientation_tag: coordinate_mapping.orientation_tag,
        scanner_frame_width: coordinate_mapping.scanner_frame_width,
        scanner_frame_height: coordinate_mapping.scanner_frame_height,
        interpolation: "monotone_piecewise_cubic_hermite_with_endpoint_tangent_extrapolation",
        confidence: calibration.confidence,
        held_out_sample_count: calibration.validation.held_out_sample_count,
        held_out_transmittance_rmse: calibration.validation.transmittance_rmse,
        held_out_transmittance_max_error: calibration.validation.transmittance_max_error,
        identity_baseline_rmse: calibration.validation.identity_baseline_rmse,
        reason: "validated scanner black/white and additive-flare correction, monotone signal linearization, and bounded spatial shading gain were applied before film-density reconstruction"
            .to_string(),
    };
    Ok(ScannerLinearizationResult { image, diagnostics })
}

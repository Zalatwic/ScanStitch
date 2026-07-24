use ndarray::Array3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const MAX_ANALYSIS_DIMENSION: usize = 1_200;
const MAX_LINE_SAMPLES: usize = 160;
const SEARCH_BAND_FRACTION: f64 = 0.28;
const MIN_POINT_STRENGTH: f64 = 5.0;
const LINE_RESIDUAL_TOLERANCE: f64 = 2.5;

#[derive(Debug, Clone)]
pub struct DeskewConfig {
    pub minimum_apply_angle_degrees: f64,
    pub maximum_angle_degrees: f64,
    pub maximum_side_disagreement_degrees: f64,
    pub minimum_side_inlier_ratio: f64,
    pub minimum_side_span_ratio: f64,
    pub minimum_retained_area_ratio: f64,
}

impl Default for DeskewConfig {
    fn default() -> Self {
        Self {
            minimum_apply_angle_degrees: 0.08,
            maximum_angle_degrees: 3.0,
            maximum_side_disagreement_degrees: 0.25,
            minimum_side_inlier_ratio: 0.45,
            minimum_side_span_ratio: 0.60,
            minimum_retained_area_ratio: 0.90,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeskewSideDiagnostics {
    pub side: String,
    pub axis: String,
    pub sample_count: usize,
    pub candidate_count: usize,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub independent_span_ratio: f64,
    pub angle_degrees: Option<f64>,
    pub residual_p95_px: Option<f64>,
    pub median_strength: Option<f64>,
    pub accepted: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeskewDiagnostics {
    pub requested_mode: String,
    pub status: String,
    pub applied: bool,
    pub detected_source_skew_degrees: Option<f64>,
    pub correction_degrees: Option<f64>,
    pub confidence: f64,
    pub review_required: bool,
    pub reason: String,
    pub input_shape: [usize; 2],
    pub output_shape: [usize; 2],
    pub supporting_side_count: usize,
    pub horizontal_side_count: usize,
    pub vertical_side_count: usize,
    pub side_angle_spread_degrees: Option<f64>,
    pub retained_area_ratio: f64,
    pub proposed_retained_area_ratio: Option<f64>,
    pub interpolation: Option<String>,
    pub side_estimates: Vec<DeskewSideDiagnostics>,
}

pub struct DeskewResult {
    pub image: Array3<u16>,
    pub diagnostics: DeskewDiagnostics,
}

#[derive(Clone, Copy)]
enum Side {
    Top,
    Bottom,
    Left,
    Right,
}

impl Side {
    fn name(self) -> &'static str {
        match self {
            Self::Top => "top",
            Self::Bottom => "bottom",
            Self::Left => "left",
            Self::Right => "right",
        }
    }

    fn axis(self) -> &'static str {
        match self {
            Self::Top | Self::Bottom => "horizontal",
            Self::Left | Self::Right => "vertical",
        }
    }

    fn is_horizontal(self) -> bool {
        matches!(self, Self::Top | Self::Bottom)
    }
}

#[derive(Debug, Clone, Copy)]
struct EdgePoint {
    independent: f64,
    dependent: f64,
    strength: f64,
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[middle - 1] + values[middle]) * 0.5)
    } else {
        Some(values[middle])
    }
}

fn percentile(values: &mut [f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    values.get(index).copied()
}

fn downsample_luminance(image: &Array3<u16>) -> (Vec<f64>, usize, usize, usize) {
    let (height, width, _) = image.dim();
    let stride = height.max(width).div_ceil(MAX_ANALYSIS_DIMENSION).max(1);
    let output_height = height.div_ceil(stride);
    let output_width = width.div_ceil(stride);
    let mut luminance = vec![0.0; output_height.saturating_mul(output_width)];
    luminance
        .par_chunks_mut(output_width)
        .enumerate()
        .for_each(|(output_y, row)| {
            let source_y = (output_y * stride).min(height.saturating_sub(1));
            for (output_x, value) in row.iter_mut().enumerate() {
                let source_x = (output_x * stride).min(width.saturating_sub(1));
                *value = 0.2126 * image[[source_y, source_x, 0]] as f64
                    + 0.7152 * image[[source_y, source_x, 1]] as f64
                    + 0.0722 * image[[source_y, source_x, 2]] as f64;
            }
        });
    (luminance, output_height, output_width, stride)
}

fn evenly_spaced_positions(start: usize, end_exclusive: usize) -> Vec<usize> {
    let available = end_exclusive.saturating_sub(start);
    if available == 0 {
        return Vec::new();
    }
    let count = available.clamp(1, MAX_LINE_SAMPLES);
    if count == 1 {
        return vec![start];
    }
    (0..count)
        .map(|index| start + index * (available - 1) / (count - 1))
        .collect()
}

fn edge_points_for_side(
    luminance: &[f64],
    height: usize,
    width: usize,
    side: Side,
) -> (Vec<EdgePoint>, usize) {
    if height < 12 || width < 12 {
        return (Vec::new(), 0);
    }
    let horizontal = side.is_horizontal();
    let independent_length = if horizontal { width } else { height };
    let dependent_length = if horizontal { height } else { width };
    let independent_margin = (independent_length as f64 * 0.04).round() as usize;
    let positions = evenly_spaced_positions(
        independent_margin.min(independent_length.saturating_sub(1)),
        independent_length.saturating_sub(independent_margin).max(1),
    );
    let band = ((dependent_length as f64 * SEARCH_BAND_FRACTION).round() as usize)
        .clamp(4, dependent_length.saturating_sub(2));
    let (search_start, search_end) = match side {
        Side::Top | Side::Left => (1usize, band),
        Side::Bottom | Side::Right => (dependent_length.saturating_sub(band), dependent_length - 1),
    };
    let mut points = Vec::with_capacity(positions.len());
    for independent in positions.iter().copied() {
        let mut gradients = Vec::with_capacity(search_end.saturating_sub(search_start));
        let mut peak = (0usize, 0.0f64);
        for dependent in search_start..search_end {
            let (before, after) = if horizontal {
                (
                    luminance[(dependent - 1) * width + independent],
                    luminance[(dependent + 1) * width + independent],
                )
            } else {
                (
                    luminance[independent * width + dependent - 1],
                    luminance[independent * width + dependent + 1],
                )
            };
            let gradient = (after - before).abs() * 0.5;
            gradients.push(gradient);
            if gradient > peak.1 {
                peak = (dependent, gradient);
            }
        }
        let mut median_values = gradients.clone();
        let background = median(&mut median_values).unwrap_or(0.0);
        let mut deviations = gradients
            .into_iter()
            .map(|value| (value - background).abs())
            .collect::<Vec<_>>();
        let mad = median(&mut deviations).unwrap_or(0.0);
        let noise = (1.4826 * mad).max(background * 0.05).max(1.0);
        let strength = (peak.1 - background).max(0.0) / noise;
        if strength >= MIN_POINT_STRENGTH {
            points.push(EdgePoint {
                independent: independent as f64,
                dependent: peak.0 as f64,
                strength: strength.min(100.0),
            });
        }
    }
    (points, positions.len())
}

fn rejected_side(
    side: Side,
    sample_count: usize,
    candidate_count: usize,
    reason: impl Into<String>,
) -> DeskewSideDiagnostics {
    DeskewSideDiagnostics {
        side: side.name().to_string(),
        axis: side.axis().to_string(),
        sample_count,
        candidate_count,
        inlier_count: 0,
        inlier_ratio: 0.0,
        independent_span_ratio: 0.0,
        angle_degrees: None,
        residual_p95_px: None,
        median_strength: None,
        accepted: false,
        reason: reason.into(),
    }
}

fn fit_side_line(
    points: &[EdgePoint],
    sample_count: usize,
    independent_length: usize,
    scale: usize,
    side: Side,
    config: &DeskewConfig,
) -> DeskewSideDiagnostics {
    if points.len() < 2 || sample_count == 0 {
        return rejected_side(
            side,
            sample_count,
            points.len(),
            "too few high-contrast border candidates",
        );
    }
    let minimum_pair_span = independent_length as f64 * 0.30;
    let mut best_slope = 0.0;
    let mut best_intercept = 0.0;
    let mut best_inlier_count = 0usize;
    let mut best_weight = 0.0f64;
    for (first_index, first) in points.iter().enumerate() {
        for second in points.iter().skip(first_index + 1) {
            let delta = second.independent - first.independent;
            if delta.abs() < minimum_pair_span {
                continue;
            }
            let slope = (second.dependent - first.dependent) / delta;
            if slope.abs() > 0.15 {
                continue;
            }
            let intercept = first.dependent - slope * first.independent;
            let mut inlier_count = 0usize;
            let mut weight = 0.0f64;
            for point in points {
                let residual = (point.dependent - (slope * point.independent + intercept)).abs();
                if residual <= LINE_RESIDUAL_TOLERANCE {
                    inlier_count += 1;
                    weight += point.strength;
                }
            }
            if inlier_count > best_inlier_count
                || (inlier_count == best_inlier_count && weight > best_weight)
            {
                best_slope = slope;
                best_intercept = intercept;
                best_inlier_count = inlier_count;
                best_weight = weight;
            }
        }
    }
    if best_inlier_count < 2 {
        return rejected_side(
            side,
            sample_count,
            points.len(),
            "border candidates did not form a coherent long line",
        );
    }

    let seed_inliers = points
        .iter()
        .filter(|point| {
            (point.dependent - (best_slope * point.independent + best_intercept)).abs()
                <= LINE_RESIDUAL_TOLERANCE
        })
        .copied()
        .collect::<Vec<_>>();
    let weight_sum = seed_inliers
        .iter()
        .map(|point| point.strength)
        .sum::<f64>()
        .max(1e-12);
    let mean_x = seed_inliers
        .iter()
        .map(|point| point.independent * point.strength)
        .sum::<f64>()
        / weight_sum;
    let mean_y = seed_inliers
        .iter()
        .map(|point| point.dependent * point.strength)
        .sum::<f64>()
        / weight_sum;
    let covariance = seed_inliers
        .iter()
        .map(|point| point.strength * (point.independent - mean_x) * (point.dependent - mean_y))
        .sum::<f64>();
    let variance = seed_inliers
        .iter()
        .map(|point| point.strength * (point.independent - mean_x).powi(2))
        .sum::<f64>();
    let slope = if variance > 1e-12 {
        covariance / variance
    } else {
        best_slope
    };
    let intercept = mean_y - slope * mean_x;
    let inliers = points
        .iter()
        .filter(|point| {
            (point.dependent - (slope * point.independent + intercept)).abs()
                <= LINE_RESIDUAL_TOLERANCE
        })
        .copied()
        .collect::<Vec<_>>();
    let inlier_count = inliers.len();
    let inlier_ratio = inlier_count as f64 / sample_count.max(1) as f64;
    let (minimum_independent, maximum_independent) = inliers.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(minimum, maximum), point| {
            (
                minimum.min(point.independent),
                maximum.max(point.independent),
            )
        },
    );
    let span_ratio = if minimum_independent.is_finite() && maximum_independent.is_finite() {
        ((maximum_independent - minimum_independent) / independent_length.max(1) as f64)
            .clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut residuals = inliers
        .iter()
        .map(|point| {
            (point.dependent - (slope * point.independent + intercept)).abs() * scale as f64
        })
        .collect::<Vec<_>>();
    let residual_p95 = percentile(&mut residuals, 0.95);
    let mut strengths = inliers
        .iter()
        .map(|point| point.strength)
        .collect::<Vec<_>>();
    let median_strength = median(&mut strengths);
    let angle = if side.is_horizontal() {
        slope.atan().to_degrees()
    } else {
        -slope.atan().to_degrees()
    };
    let minimum_inlier_count =
        ((sample_count as f64 * config.minimum_side_inlier_ratio).ceil() as usize).max(12);
    let accepted = inlier_count >= minimum_inlier_count
        && inlier_ratio >= config.minimum_side_inlier_ratio
        && span_ratio >= config.minimum_side_span_ratio
        && median_strength.is_some_and(|strength| strength >= MIN_POINT_STRENGTH)
        && angle.is_finite()
        && angle.abs() <= config.maximum_angle_degrees * 1.5;
    let reason = if accepted {
        format!(
            "coherent {} border line from {}/{} spatial samples",
            side.axis(),
            inlier_count,
            sample_count
        )
    } else {
        format!(
            "border line evidence below gate: inliers {}/{}, ratio {:.3}, span {:.3}, median strength {:.2}",
            inlier_count,
            minimum_inlier_count,
            inlier_ratio,
            span_ratio,
            median_strength.unwrap_or(0.0)
        )
    };
    DeskewSideDiagnostics {
        side: side.name().to_string(),
        axis: side.axis().to_string(),
        sample_count,
        candidate_count: points.len(),
        inlier_count,
        inlier_ratio,
        independent_span_ratio: span_ratio,
        angle_degrees: Some(angle),
        residual_p95_px: residual_p95,
        median_strength,
        accepted,
        reason,
    }
}

fn weighted_median(values: &mut [(f64, f64)]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.0.total_cmp(&right.0));
    let total_weight = values.iter().map(|(_, weight)| *weight).sum::<f64>();
    let target = total_weight * 0.5;
    let mut cumulative = 0.0;
    for (value, weight) in values.iter().copied() {
        cumulative += weight;
        if cumulative >= target {
            return Some(value);
        }
    }
    values.last().map(|(value, _)| *value)
}

fn largest_inscribed_dimensions(width: usize, height: usize, angle_degrees: f64) -> (usize, usize) {
    let angle = angle_degrees.to_radians().abs();
    if angle < 1e-9 {
        return (width, height);
    }
    let sine = angle.sin().abs();
    let cosine = angle.cos().abs();
    let width_f = width as f64;
    let height_f = height as f64;
    let width_is_longer = width_f >= height_f;
    let long_side = width_f.max(height_f);
    let short_side = width_f.min(height_f);
    let (inscribed_width, inscribed_height) =
        if short_side <= 2.0 * sine * cosine * long_side || (sine - cosine).abs() < 1e-12 {
            let half_short = 0.5 * short_side;
            if width_is_longer {
                (half_short / sine.max(1e-12), half_short / cosine.max(1e-12))
            } else {
                (half_short / cosine.max(1e-12), half_short / sine.max(1e-12))
            }
        } else {
            let cosine_double = (cosine * cosine - sine * sine).max(1e-12);
            (
                (width_f * cosine - height_f * sine) / cosine_double,
                (height_f * cosine - width_f * sine) / cosine_double,
            )
        };
    (
        (inscribed_width.floor() as usize)
            .saturating_sub(4)
            .clamp(1, width),
        (inscribed_height.floor() as usize)
            .saturating_sub(4)
            .clamp(1, height),
    )
}

fn retained_area_ratio(width: usize, height: usize, correction_degrees: f64) -> f64 {
    let (output_width, output_height) =
        largest_inscribed_dimensions(width, height, correction_degrees);
    output_width.saturating_mul(output_height) as f64 / width.saturating_mul(height).max(1) as f64
}

fn estimate_skew(image: &Array3<u16>, config: &DeskewConfig) -> DeskewDiagnostics {
    let (height, width, _) = image.dim();
    let (luminance, analysis_height, analysis_width, scale) = downsample_luminance(image);
    let sides = [Side::Top, Side::Bottom, Side::Left, Side::Right];
    let side_estimates = sides
        .into_iter()
        .map(|side| {
            let (points, sample_count) =
                edge_points_for_side(&luminance, analysis_height, analysis_width, side);
            fit_side_line(
                &points,
                sample_count,
                if side.is_horizontal() {
                    analysis_width
                } else {
                    analysis_height
                },
                scale,
                side,
                config,
            )
        })
        .collect::<Vec<_>>();
    let accepted = side_estimates
        .iter()
        .filter(|estimate| estimate.accepted)
        .collect::<Vec<_>>();
    let horizontal_count = accepted
        .iter()
        .filter(|estimate| estimate.axis == "horizontal")
        .count();
    let vertical_count = accepted
        .iter()
        .filter(|estimate| estimate.axis == "vertical")
        .count();
    let mut weighted_angles = accepted
        .iter()
        .filter_map(|estimate| {
            estimate.angle_degrees.map(|angle| {
                (
                    angle,
                    estimate.inlier_ratio
                        * estimate.independent_span_ratio
                        * estimate.median_strength.unwrap_or(1.0).min(20.0),
                )
            })
        })
        .collect::<Vec<_>>();
    let detected_angle = weighted_median(&mut weighted_angles);
    let accepted_angle_bounds = accepted
        .iter()
        .filter_map(|estimate| estimate.angle_degrees)
        .fold(None::<(f64, f64)>, |bounds, angle| {
            Some(bounds.map_or((angle, angle), |(minimum, maximum)| {
                (minimum.min(angle), maximum.max(angle))
            }))
        });
    let angle_spread = accepted_angle_bounds.map(|(minimum, maximum)| maximum - minimum);
    let maximum_supported_abs_angle = accepted
        .iter()
        .filter_map(|estimate| estimate.angle_degrees)
        .map(f64::abs)
        .fold(0.0f64, f64::max);
    let accepted_axes = horizontal_count > 0 && vertical_count > 0;
    let accepted_opposing_axis = horizontal_count >= 2 || vertical_count >= 2;
    let accepted_geometry_support = accepted_axes || accepted_opposing_axis;
    let agreement =
        angle_spread.is_some_and(|spread| spread <= config.maximum_side_disagreement_degrees);
    let angle_in_range =
        detected_angle.is_some_and(|angle| angle.abs() <= config.maximum_angle_degrees);
    let correction = detected_angle.map(|angle| -angle);
    let retained_ratio = correction
        .map(|angle| retained_area_ratio(width, height, angle))
        .unwrap_or(1.0);
    let retained_area_accepted = retained_ratio >= config.minimum_retained_area_ratio;
    let already_aligned = accepted.len() >= 2
        && accepted_geometry_support
        && detected_angle.is_some_and(|angle| angle.abs() < config.minimum_apply_angle_degrees)
        && maximum_supported_abs_angle <= config.maximum_side_disagreement_degrees;
    let evidence_accepted = !already_aligned
        && accepted.len() >= 2
        && accepted_geometry_support
        && agreement
        && angle_in_range
        && retained_area_accepted;
    let strongest_suspected_angle = accepted
        .iter()
        .filter_map(|estimate| estimate.angle_degrees.map(f64::abs))
        .fold(0.0f64, f64::max);
    let review_required = !evidence_accepted
        && !already_aligned
        && !accepted.is_empty()
        && strongest_suspected_angle >= config.minimum_apply_angle_degrees * 2.0;
    let confidence = if evidence_accepted || already_aligned {
        let mean_inlier_ratio = accepted
            .iter()
            .map(|estimate| estimate.inlier_ratio)
            .sum::<f64>()
            / accepted.len().max(1) as f64;
        let mean_span = accepted
            .iter()
            .map(|estimate| estimate.independent_span_ratio)
            .sum::<f64>()
            / accepted.len().max(1) as f64;
        let agreement_score = 1.0
            - angle_spread.unwrap_or(config.maximum_side_disagreement_degrees)
                / config.maximum_side_disagreement_degrees.max(1e-9);
        (0.40 * mean_inlier_ratio
            + 0.25 * mean_span
            + 0.20 * agreement_score.clamp(0.0, 1.0)
            + 0.15 * (accepted.len() as f64 / 4.0))
            .clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (status, reason) = if already_aligned {
        (
            "already_aligned",
            format!(
                "the robust border-line aggregate is {:.4}° source skew, below the {:.3}° correction threshold, and every supported side remains within the bounded ±{:.3}° no-op envelope",
                detected_angle.unwrap_or(0.0),
                config.minimum_apply_angle_degrees,
                config.maximum_side_disagreement_degrees
            ),
        )
    } else if evidence_accepted {
        (
            "accepted_for_correction",
            format!(
                "{} independent border sides across both axes agree on {:.4}° source skew",
                accepted.len(),
                detected_angle.unwrap_or(0.0)
            ),
        )
    } else if accepted.len() < 2 || !accepted_geometry_support {
        (
            "insufficient_evidence",
            "absolute deskew requires either two coherent opposing borders on one axis or coherent support across both axes".to_string(),
        )
    } else if !agreement {
        (
            "conflicting_evidence",
            format!(
                "border-line angle spread {:.4}° exceeds the {:.3}° agreement limit",
                angle_spread.unwrap_or(f64::INFINITY),
                config.maximum_side_disagreement_degrees
            ),
        )
    } else if !angle_in_range {
        (
            "angle_out_of_range",
            format!(
                "detected source skew {:.4}° exceeds the bounded {:.2}° automatic range",
                detected_angle.unwrap_or(0.0),
                config.maximum_angle_degrees
            ),
        )
    } else {
        (
            "retained_area_rejected",
            format!(
                "deskew would retain only {:.2}% of the valid rectangular area, below the {:.2}% gate",
                retained_ratio * 100.0,
                config.minimum_retained_area_ratio * 100.0
            ),
        )
    };
    DeskewDiagnostics {
        requested_mode: "auto".to_string(),
        status: status.to_string(),
        applied: false,
        detected_source_skew_degrees: detected_angle,
        correction_degrees: correction,
        confidence,
        review_required,
        reason,
        input_shape: [height, width],
        output_shape: [height, width],
        supporting_side_count: accepted.len(),
        horizontal_side_count: horizontal_count,
        vertical_side_count: vertical_count,
        side_angle_spread_degrees: angle_spread,
        retained_area_ratio: 1.0,
        proposed_retained_area_ratio: correction.map(|_| retained_ratio),
        interpolation: None,
        side_estimates,
    }
}

fn cubic_catmull_rom_weight(distance: f64) -> f64 {
    let distance = distance.abs();
    if distance <= 1.0 {
        1.5 * distance.powi(3) - 2.5 * distance.powi(2) + 1.0
    } else if distance < 2.0 {
        -0.5 * distance.powi(3) + 2.5 * distance.powi(2) - 4.0 * distance + 2.0
    } else {
        0.0
    }
}

fn rotate_bicubic_inscribed(
    source: &Array3<u16>,
    correction_degrees: f64,
    maximum_value: f64,
) -> Array3<u16> {
    let (source_height, source_width, _) = source.dim();
    let (output_width, output_height) =
        largest_inscribed_dimensions(source_width, source_height, correction_degrees);
    let angle = correction_degrees.to_radians();
    let cosine = angle.cos();
    let sine = angle.sin();
    let source_center_x = source_width as f64 * 0.5;
    let source_center_y = source_height as f64 * 0.5;
    let output_center_x = output_width as f64 * 0.5;
    let output_center_y = output_height as f64 * 0.5;
    let mut pixels = vec![0u16; output_height.saturating_mul(output_width).saturating_mul(3)];
    pixels
        .par_chunks_mut(output_width * 3)
        .enumerate()
        .for_each(|(output_y, row)| {
            let output_dy = output_y as f64 + 0.5 - output_center_y;
            for output_x in 0..output_width {
                let output_dx = output_x as f64 + 0.5 - output_center_x;
                let source_centered_x = cosine * output_dx + sine * output_dy;
                let source_centered_y = -sine * output_dx + cosine * output_dy;
                let source_x = (source_center_x + source_centered_x - 0.5)
                    .clamp(0.0, source_width.saturating_sub(1) as f64);
                let source_y = (source_center_y + source_centered_y - 0.5)
                    .clamp(0.0, source_height.saturating_sub(1) as f64);
                let base_x = source_x.floor() as isize;
                let base_y = source_y.floor() as isize;
                let mut accumulated = [0.0f64; 3];
                let mut total_weight = 0.0;
                for offset_y in -1isize..=2 {
                    let sample_y = (base_y + offset_y)
                        .clamp(0, source_height.saturating_sub(1) as isize)
                        as usize;
                    let weight_y = cubic_catmull_rom_weight(source_y - (base_y + offset_y) as f64);
                    for offset_x in -1isize..=2 {
                        let sample_x = (base_x + offset_x)
                            .clamp(0, source_width.saturating_sub(1) as isize)
                            as usize;
                        let weight_x =
                            cubic_catmull_rom_weight(source_x - (base_x + offset_x) as f64);
                        let weight = weight_x * weight_y;
                        total_weight += weight;
                        for channel in 0..3 {
                            accumulated[channel] +=
                                source[[sample_y, sample_x, channel]] as f64 * weight;
                        }
                    }
                }
                for channel in 0..3 {
                    row[output_x * 3 + channel] = (accumulated[channel] / total_weight.max(1e-12))
                        .round()
                        .clamp(0.0, maximum_value)
                        as u16;
                }
            }
        });
    Array3::from_shape_vec((output_height, output_width, 3), pixels)
        .expect("deskew output shape matches allocated pixel buffer")
}

pub fn auto_deskew(image: Array3<u16>, maximum_value: f64, config: &DeskewConfig) -> DeskewResult {
    let mut diagnostics = estimate_skew(&image, config);
    if diagnostics.status != "accepted_for_correction" {
        return DeskewResult { image, diagnostics };
    }
    let correction = diagnostics.correction_degrees.unwrap_or(0.0);
    let output = rotate_bicubic_inscribed(&image, correction, maximum_value);
    diagnostics.status = "applied".to_string();
    diagnostics.applied = true;
    diagnostics.retained_area_ratio = diagnostics.proposed_retained_area_ratio.unwrap_or(1.0);
    diagnostics.output_shape = [output.dim().0, output.dim().1];
    diagnostics.interpolation = Some("bicubic_catmull_rom_single_resample".to_string());
    diagnostics.reason = format!(
        "applied {:.4}° correction after mutually supporting independent border-line evidence; retained {:.2}% of the valid rectangular area",
        correction,
        diagnostics.retained_area_ratio * 100.0
    );
    DeskewResult {
        image: output,
        diagnostics,
    }
}

pub fn manual_deskew(
    image: Array3<u16>,
    source_skew_degrees: f64,
    maximum_value: f64,
    config: &DeskewConfig,
) -> DeskewResult {
    let (height, width, _) = image.dim();
    let correction = -source_skew_degrees;
    let retained_ratio = retained_area_ratio(width, height, correction);
    let mut diagnostics = DeskewDiagnostics {
        requested_mode: "manual".to_string(),
        status: "manual_rejected".to_string(),
        applied: false,
        detected_source_skew_degrees: Some(source_skew_degrees),
        correction_degrees: Some(correction),
        confidence: 1.0,
        review_required: false,
        reason: String::new(),
        input_shape: [height, width],
        output_shape: [height, width],
        supporting_side_count: 0,
        horizontal_side_count: 0,
        vertical_side_count: 0,
        side_angle_spread_degrees: None,
        retained_area_ratio: 1.0,
        proposed_retained_area_ratio: Some(retained_ratio),
        interpolation: None,
        side_estimates: Vec::new(),
    };
    if source_skew_degrees.abs() < config.minimum_apply_angle_degrees {
        diagnostics.status = "already_aligned".to_string();
        diagnostics.reason = format!(
            "manual source skew {:.4}° is below the {:.3}° correction threshold",
            source_skew_degrees, config.minimum_apply_angle_degrees
        );
        return DeskewResult { image, diagnostics };
    }
    if source_skew_degrees.abs() > config.maximum_angle_degrees {
        diagnostics.review_required = true;
        diagnostics.reason = format!(
            "manual source skew {:.4}° exceeds the bounded {:.2}° deskew range",
            source_skew_degrees, config.maximum_angle_degrees
        );
        return DeskewResult { image, diagnostics };
    }
    if retained_ratio < config.minimum_retained_area_ratio {
        diagnostics.review_required = true;
        diagnostics.reason = format!(
            "manual deskew would retain only {:.2}% of the valid rectangular area, below the {:.2}% gate",
            retained_ratio * 100.0,
            config.minimum_retained_area_ratio * 100.0
        );
        return DeskewResult { image, diagnostics };
    }
    let output = rotate_bicubic_inscribed(&image, correction, maximum_value);
    diagnostics.status = "applied".to_string();
    diagnostics.applied = true;
    diagnostics.retained_area_ratio = retained_ratio;
    diagnostics.output_shape = [output.dim().0, output.dim().1];
    diagnostics.interpolation = Some("bicubic_catmull_rom_single_resample".to_string());
    diagnostics.reason = format!(
        "applied manual {:.4}° correction for declared {:.4}° source skew; retained {:.2}% of the valid rectangular area",
        correction,
        source_skew_degrees,
        retained_ratio * 100.0
    );
    DeskewResult {
        image: output,
        diagnostics,
    }
}

pub fn no_op_diagnostics(
    mode: &str,
    status: &str,
    reason: impl Into<String>,
    image: &Array3<u16>,
) -> DeskewDiagnostics {
    DeskewDiagnostics {
        requested_mode: mode.to_string(),
        status: status.to_string(),
        applied: false,
        detected_source_skew_degrees: None,
        correction_degrees: None,
        confidence: 0.0,
        review_required: false,
        reason: reason.into(),
        input_shape: [image.dim().0, image.dim().1],
        output_shape: [image.dim().0, image.dim().1],
        supporting_side_count: 0,
        horizontal_side_count: 0,
        vertical_side_count: 0,
        side_angle_spread_degrees: None,
        retained_area_ratio: 1.0,
        proposed_retained_area_ratio: None,
        interpolation: None,
        side_estimates: Vec::new(),
    }
}

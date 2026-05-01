use crate::base_detect;
use crate::cv_adapter::opencv_match;
use crate::report::PhaseReport;
use crate::tiff_io;
use nalgebra::{Matrix3, Vector3};
use ndarray::{s, Array2, Array3, ArrayView2};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::PathBuf;

const SIGNATURE_SCREEN_LIMIT: usize = 18;
const SIGNATURE_SCREEN_NEIGHBORHOOD: usize = 2;
const EXPANDED_SIGNATURE_SCREEN_LIMIT: usize = 4;
const EXPANDED_SIGNATURE_SCREEN_NEIGHBORHOOD: usize = 1;
const CANDIDATE_OVERLAP_DIVERSITY_PX: usize = 8;
const MAX_TRANSLATION_CANDIDATES_TO_VALIDATE: usize = 6;
const MAX_REPORTED_TRANSLATION_CANDIDATES: usize = 5;
const MAX_REPORTED_EVALUATED_CANDIDATES: usize = 6;
const MIN_PLAUSIBLE_OVERLAP_FRACTION: f64 = 0.03;
const MAX_PLAUSIBLE_OVERLAP_FRACTION: f64 = 0.85;
const NARROW_SEAM_RESCUE_MAX_OVERLAP_FRACTION: f64 = 0.12;
const MIN_INFORMATIVE_LOCAL_WINDOWS: usize = 8;
const MIN_INFORMATIVE_WINDOW_RATIO: f64 = 0.25;
const INFORMATIVE_MODEL_INLIER_RESIDUAL_PX: f64 = 2.0;
const MAX_INFORMATIVE_MODEL_P95_RESIDUAL_PX: f64 = 3.0;
const MIN_NARROW_SEAM_SIGNATURE_SCORE: f64 = 0.25;
const MIN_NARROW_SEAM_SEAM_SUPPORT_SCORE: f64 = 0.65;
const MIN_NARROW_SEAM_ROW_SUPPORT_RATIO: f64 = 0.60;
const MIN_NARROW_SEAM_COLUMN_SUPPORT_RATIO: f64 = 0.60;
const NARROW_SEAM_OBJECTIVE_MARGIN: f64 = 0.15;

/// Runtime transform preference for stitching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformMode {
    Auto,
    Translation,
    Affine,
    Homography,
}

impl TransformMode {
    pub fn from_cli(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "translation" => Ok(Self::Translation),
            "affine" => Ok(Self::Affine),
            "homography" => Ok(Self::Homography),
            other => Err(format!(
                "invalid --transform value '{}'; expected auto, translation, affine, or homography",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Translation => "translation",
            Self::Affine => "affine",
            Self::Homography => "homography",
        }
    }
}

/// Configuration for the stitching algorithm.
#[derive(Debug, Clone)]
pub struct StitchConfig {
    pub max_overlap: usize,
    pub min_ncc_score: f64,
    pub min_overlap: usize,
    pub blend_width: usize,
    /// Maximum vertical drift in pixels to search (searches -max_y_offset..=+max_y_offset).
    pub max_y_offset: i32,
    pub transform_mode: TransformMode,
    pub use_opencv: bool,
    pub input_bit_depth: u8,
    pub target_width: Option<usize>,
    pub target_height: Option<usize>,
    pub min_objective_score: f64,
    pub min_local_inlier_ratio: f64,
    pub max_local_median_residual: f64,
    pub max_local_p95_residual: f64,
    pub min_validation_windows: usize,
    pub debug_dir: Option<PathBuf>,
}

impl Default for StitchConfig {
    fn default() -> Self {
        Self {
            max_overlap: 300,
            min_ncc_score: 0.6,
            min_overlap: 20,
            blend_width: 50,
            max_y_offset: 15,
            transform_mode: TransformMode::Auto,
            use_opencv: false,
            input_bit_depth: 14,
            target_width: None,
            target_height: None,
            min_objective_score: 0.60,
            min_local_inlier_ratio: 0.55,
            max_local_median_residual: 1.25,
            max_local_p95_residual: 3.0,
            min_validation_windows: 4,
            debug_dir: None,
        }
    }
}

/// Result of a stitch operation.
pub struct StitchResult {
    pub result: Option<Array3<u16>>,
    pub x_offset: i32,
    pub y_offset: i32,
    pub ncc_score: f64,
    pub order: StitchOrder,
    pub report: PhaseReport,
}

/// Detected stitch ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StitchOrder {
    LeftRight,
    RightLeft,
    NoStitch,
}

impl StitchOrder {
    fn label(self) -> &'static str {
        match self {
            Self::LeftRight => "[1|2]",
            Self::RightLeft => "[2|1]",
            Self::NoStitch => "[skip]",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CropInfo {
    x_start: usize,
    y_start: usize,
    width: usize,
    height: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct StitchValidationMetrics {
    pub overlap_width_px: usize,
    pub comparison_height_px: usize,
    pub global_ncc_score: f64,
    pub signature_score: f64,
    pub overlap_support_score: f64,
    pub vertical_offset_plausibility_score: f64,
    pub plausibility_score: f64,
    pub evidence_score: f64,
    pub prior_weight: f64,
    pub local_consistency_score: f64,
    pub local_windows_evaluated: usize,
    pub local_inlier_count: usize,
    pub local_inlier_ratio: f64,
    pub median_local_ncc: f64,
    pub informative_ncc_threshold: f64,
    pub informative_window_count: usize,
    pub informative_window_ratio: f64,
    pub informative_inlier_count: usize,
    pub informative_inlier_ratio: f64,
    pub row_support_ratio: f64,
    pub column_support_ratio: f64,
    pub seam_support_score: f64,
    pub median_residual_px: f64,
    pub p95_residual_px: f64,
    pub median_model_residual_px: f64,
    pub p95_model_residual_px: f64,
    pub mean_dx_px: f64,
    pub mean_dy_px: f64,
    pub overlap_fraction_left: f64,
    pub overlap_fraction_right: f64,
    pub plausibility_ok: bool,
    pub plausibility_reason: Option<String>,
    pub homography_inliers: Option<usize>,
    pub homography_median_error: Option<f64>,
    pub homography_p95_error: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StitchSearchCandidate {
    pub rank: usize,
    pub search_band: String,
    pub overlap_width: usize,
    pub x_offset: usize,
    pub vertical_offset: i32,
    pub global_ncc_score: f64,
    pub signature_score: f64,
    pub overlap_fraction_left: f64,
    pub overlap_fraction_right: f64,
    pub overlap_support_score: f64,
    pub vertical_offset_plausibility_score: f64,
    pub plausibility_score: f64,
    pub correspondence_score: f64,
    pub prior_weight: f64,
    pub search_score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StitchEvaluatedCandidate {
    pub validation_rank: usize,
    pub search_rank: Option<usize>,
    pub search_band: String,
    pub overlap_width: usize,
    pub x_offset: usize,
    pub vertical_offset: i32,
    pub objective_score: f64,
    pub accepted: bool,
    pub acceptance_reason: Option<String>,
    pub rejection_reason: Option<String>,
    pub global_ncc_score: f64,
    pub signature_score: f64,
    pub overlap_support_score: f64,
    pub vertical_offset_plausibility_score: f64,
    pub plausibility_score: f64,
    pub evidence_score: f64,
    pub prior_weight: f64,
    pub local_consistency_score: f64,
    pub informative_window_count: usize,
    pub informative_window_ratio: f64,
    pub informative_inlier_ratio: f64,
    pub row_support_ratio: f64,
    pub column_support_ratio: f64,
    pub seam_support_score: f64,
    pub local_inlier_ratio: f64,
    pub median_local_ncc: f64,
    pub median_residual_px: f64,
    pub p95_residual_px: f64,
    pub p95_model_residual_px: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StitchSearchDiagnostics {
    pub screening_strategy: String,
    pub screened_overlap_count: usize,
    pub expanded_screened_overlap_count: usize,
    pub evaluated_overlap_count: usize,
    pub exhaustive_fallback_used: bool,
    pub expanded_overlap_search_used: bool,
    pub primary_max_overlap_considered: usize,
    pub max_overlap_considered: usize,
    pub selected_candidate_rank: Option<usize>,
    pub selected_candidate_search_score: Option<f64>,
    pub search_score_gap_to_runner_up: Option<f64>,
    pub selection_reason: Option<String>,
    pub top_candidates: Vec<StitchSearchCandidate>,
    pub evaluated_candidates: Vec<StitchEvaluatedCandidate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StitchHypothesis {
    pub ordering: String,
    pub x_offset: Option<usize>,
    pub overlap_width: Option<usize>,
    pub vertical_offset: Option<i32>,
    pub transform_model: String,
    pub objective_score: f64,
    pub accepted: bool,
    pub acceptance_reason: Option<String>,
    pub rejection_reason: Option<String>,
    pub search: StitchSearchDiagnostics,
    pub validation: StitchValidationMetrics,
}

/// Convert an RGB u16 image to grayscale f64 using luminance weights.
fn to_grayscale(img: &Array3<u16>) -> Array2<f64> {
    let (h, w, _) = img.dim();
    let mut gray = Array2::<f64>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let r = img[[y, x, 0]] as f64;
            let g = img[[y, x, 1]] as f64;
            let b = img[[y, x, 2]] as f64;
            gray[[y, x]] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }
    }
    gray
}

/// Compute normalized cross-correlation between two 2D arrays of the same shape.
fn ncc(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
    let n = a.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean_a = a.sum() / n;
    let mean_b = b.sum() / n;

    let mut sum_ab = 0.0;
    let mut sum_aa = 0.0;
    let mut sum_bb = 0.0;

    for (va, vb) in a.iter().zip(b.iter()) {
        let da = va - mean_a;
        let db = vb - mean_b;
        sum_ab += da * db;
        sum_aa += da * da;
        sum_bb += db * db;
    }

    let denom = (sum_aa * sum_bb).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }
    sum_ab / denom
}

/// Compute overlap score using exposure-invariant NCC.
fn overlap_score(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
    ncc(a, b).max(0.0)
}

fn overlap_score_at(
    gray_l: &Array2<f64>,
    gray_r: &Array2<f64>,
    h: usize,
    w_l: usize,
    ovl: usize,
    dy: i32,
) -> Option<f64> {
    let abs_dy = dy.unsigned_abs() as usize;
    let cmp_h = h.saturating_sub(abs_dy);
    if cmp_h < 10 {
        return None;
    }

    let (l_y_start, r_y_start) = if dy >= 0 {
        (dy as usize, 0usize)
    } else {
        (0usize, abs_dy)
    };

    let strip_l = gray_l.slice(s![l_y_start..(l_y_start + cmp_h), (w_l - ovl)..w_l]);
    let strip_r = gray_r.slice(s![r_y_start..(r_y_start + cmp_h), 0..ovl]);
    Some(overlap_score(strip_l.view(), strip_r.view()))
}

fn ncc1d(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let mean_a = a.iter().take(n).sum::<f64>() / n as f64;
    let mean_b = b.iter().take(n).sum::<f64>() / n as f64;
    let mut sum_ab = 0.0;
    let mut sum_aa = 0.0;
    let mut sum_bb = 0.0;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        sum_ab += da * db;
        sum_aa += da * da;
        sum_bb += db * db;
    }

    let denom = (sum_aa * sum_bb).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        sum_ab / denom
    }
}

fn column_feature_series(gray: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    let (h, w) = gray.dim();
    let mut mean_series = vec![0.0f64; w];
    let mut grad_series = vec![0.0f64; w];

    for x in 0..w {
        let mut sum = 0.0f64;
        let mut grad = 0.0f64;
        for y in 0..h {
            let v = gray[[y, x]];
            sum += v;
            if y > 0 {
                grad += (v - gray[[y - 1, x]]).abs();
            }
        }
        mean_series[x] = sum / h.max(1) as f64;
        grad_series[x] = grad / h.max(2) as f64;
    }

    (mean_series, grad_series)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TranslationSearchResult {
    search_band: SearchBand,
    x_offset: usize,
    overlap: usize,
    y_offset: i32,
    global_ncc: f64,
    signature_score: f64,
    overlap_support_score: f64,
    vertical_offset_plausibility_score: f64,
    plausibility_score: f64,
    correspondence_score: f64,
    prior_weight: f64,
    search_score: f64,
}

#[derive(Debug, Clone)]
struct TranslationSearchOutcome {
    candidates: Vec<TranslationSearchResult>,
    screened_overlap_count: usize,
    expanded_screened_overlap_count: usize,
    evaluated_overlap_count: usize,
    exhaustive_fallback_used: bool,
    expanded_overlap_search_used: bool,
    primary_max_overlap_considered: usize,
    max_overlap_considered: usize,
}

#[derive(Debug, Clone)]
struct EvaluatedTranslationCandidate {
    search: TranslationSearchResult,
    validation: StitchValidationMetrics,
    objective_score: f64,
    acceptance_reason: Option<String>,
    rejection_reason: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct LocalWindowMatch {
    row_idx: usize,
    col_idx: usize,
    x_norm: f64,
    y_norm: f64,
    score: f64,
    dx: i32,
    dy: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchBand {
    PrimaryScreened,
    PrimaryExhaustive,
    ExpandedSignature,
}

impl SearchBand {
    fn as_str(self) -> &'static str {
        match self {
            Self::PrimaryScreened => "primary_signature_screen",
            Self::PrimaryExhaustive => "primary_exhaustive",
            Self::ExpandedSignature => "expanded_signature_guided",
        }
    }
}

fn smoothstep01(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn fit_weighted_local_offset_plane(
    matches: &[LocalWindowMatch],
    value: fn(&LocalWindowMatch) -> f64,
) -> Option<Vector3<f64>> {
    if matches.is_empty() {
        return None;
    }

    if matches.len() < 3 {
        let total_weight = matches.iter().map(|m| m.score.max(1e-3)).sum::<f64>();
        let mean = matches
            .iter()
            .map(|m| value(m) * m.score.max(1e-3))
            .sum::<f64>()
            / total_weight.max(1e-6);
        return Some(Vector3::new(mean, 0.0, 0.0));
    }

    let mut ata = Matrix3::<f64>::zeros();
    let mut atb = Vector3::<f64>::zeros();
    for m in matches {
        let weight = m.score.max(1e-3);
        let basis = Vector3::new(1.0, m.x_norm, m.y_norm);
        ata += basis * basis.transpose() * weight;
        atb += basis * (value(m) * weight);
    }

    (ata + Matrix3::identity() * 1e-6)
        .try_inverse()
        .map(|inv| inv * atb)
}

fn evaluate_local_offset_plane(model: &Vector3<f64>, m: &LocalWindowMatch) -> f64 {
    model[0] + model[1] * m.x_norm + model[2] * m.y_norm
}

fn overlap_support_score(overlap_fraction: f64) -> f64 {
    let rise = smoothstep01((overlap_fraction - 0.015) / 0.055);
    let fall = 1.0 - smoothstep01((overlap_fraction - 0.80) / 0.15);
    (rise * fall).clamp(0.0, 1.0)
}

fn vertical_offset_plausibility_score(y_offset: i32, max_y_offset: i32) -> f64 {
    let limit = max_y_offset.unsigned_abs() as f64;
    if limit <= 0.5 {
        return 1.0;
    }
    let offset_fraction = y_offset.unsigned_abs() as f64 / limit;
    (1.0 - 0.85 * smoothstep01((offset_fraction - 0.35) / 0.65)).clamp(0.15, 1.0)
}

fn translation_plausibility_score(
    overlap_support_score: f64,
    vertical_offset_plausibility_score: f64,
) -> f64 {
    (0.65 * overlap_support_score + 0.35 * vertical_offset_plausibility_score).clamp(0.0, 1.0)
}

fn weighted_geometric_mean(components: &[(f64, f64)]) -> f64 {
    let mut weighted_log_sum = 0.0;
    let mut total_weight = 0.0;
    for &(value, weight) in components {
        if weight <= 0.0 {
            continue;
        }
        weighted_log_sum += weight * value.clamp(1e-6, 1.0).ln();
        total_weight += weight;
    }
    if total_weight <= 0.0 {
        0.0
    } else {
        (weighted_log_sum / total_weight).exp().clamp(0.0, 1.0)
    }
}

fn overlap_prior_weight(overlap_support_score: f64) -> f64 {
    let support = overlap_support_score.clamp(0.0, 1.0);
    if support < 0.25 {
        // Keep aggressively penalizing tiny edge-only overlaps.
        (0.18 + 1.52 * support).clamp(0.18, 0.56)
    } else {
        // Once the overlap is in a plausible seam band, treat size as a weak prior.
        (0.82 + 0.18 * support.sqrt()).clamp(0.82, 1.0)
    }
}

fn vertical_prior_weight(vertical_offset_plausibility_score: f64) -> f64 {
    (0.55 + 0.45 * vertical_offset_plausibility_score.clamp(0.0, 1.0)).clamp(0.55, 1.0)
}

fn translation_prior_weight(
    overlap_support_score: f64,
    vertical_offset_plausibility_score: f64,
) -> f64 {
    (overlap_prior_weight(overlap_support_score)
        * vertical_prior_weight(vertical_offset_plausibility_score))
    .clamp(0.0, 1.0)
}

fn translation_search_correspondence_score(global_ncc: f64, signature_score: f64) -> f64 {
    let arithmetic = (global_ncc * 0.80 + signature_score * 0.20).clamp(0.0, 1.0);
    let geometric = weighted_geometric_mean(&[(global_ncc, 0.80), (signature_score, 0.20)]);
    (0.80 * arithmetic + 0.20 * geometric).clamp(0.0, 1.0)
}

fn translation_search_score(
    correspondence_score: f64,
    overlap_support_score: f64,
    vertical_offset_plausibility_score: f64,
) -> f64 {
    (correspondence_score
        * translation_prior_weight(overlap_support_score, vertical_offset_plausibility_score))
    .clamp(0.0, 1.0)
}

fn translation_evidence_score(
    global_ncc_score: f64,
    signature_score: f64,
    local_consistency_score: f64,
) -> f64 {
    let arithmetic =
        (global_ncc_score * 0.60 + signature_score * 0.10 + local_consistency_score * 0.30)
            .clamp(0.0, 1.0);
    let geometric = weighted_geometric_mean(&[
        (global_ncc_score, 0.60),
        (signature_score, 0.10),
        (local_consistency_score, 0.30),
    ]);
    (0.75 * arithmetic + 0.25 * geometric).clamp(0.0, 1.0)
}

fn translation_objective_score(
    evidence_score: f64,
    overlap_support_score: f64,
    vertical_offset_plausibility_score: f64,
) -> f64 {
    (evidence_score
        * translation_prior_weight(overlap_support_score, vertical_offset_plausibility_score))
    .clamp(0.0, 1.0)
}

fn compare_translation_candidates(
    candidate: &TranslationSearchResult,
    current: &TranslationSearchResult,
) -> std::cmp::Ordering {
    current
        .search_score
        .partial_cmp(&candidate.search_score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| {
            current
                .global_ncc
                .partial_cmp(&candidate.global_ncc)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            current
                .plausibility_score
                .partial_cmp(&candidate.plausibility_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            current
                .signature_score
                .partial_cmp(&candidate.signature_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            candidate
                .y_offset
                .unsigned_abs()
                .cmp(&current.y_offset.unsigned_abs())
        })
        .then_with(|| candidate.overlap.cmp(&current.overlap))
}

fn compare_raw_translation_candidates(
    candidate: &TranslationSearchResult,
    current: &TranslationSearchResult,
) -> std::cmp::Ordering {
    current
        .global_ncc
        .partial_cmp(&candidate.global_ncc)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| {
            current
                .signature_score
                .partial_cmp(&candidate.signature_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| candidate.overlap.cmp(&current.overlap))
        .then_with(|| {
            candidate
                .y_offset
                .unsigned_abs()
                .cmp(&current.y_offset.unsigned_abs())
        })
}

fn signature_overlap_score_from_series(
    mean_l: &[f64],
    grad_l: &[f64],
    mean_r: &[f64],
    grad_r: &[f64],
    overlap: usize,
) -> f64 {
    let w_l = mean_l.len();
    let w_r = mean_r.len();
    if overlap == 0 || overlap > w_l || overlap > w_r {
        return 0.0;
    }

    let score_mean = ncc1d(&mean_l[w_l - overlap..w_l], &mean_r[0..overlap]).max(0.0);
    let score_grad = ncc1d(&grad_l[w_l - overlap..w_l], &grad_r[0..overlap]).max(0.0);
    0.65 * score_mean + 0.35 * score_grad
}

fn signature_overlap_scores(
    mean_l: &[f64],
    grad_l: &[f64],
    mean_r: &[f64],
    grad_r: &[f64],
    max_overlap: usize,
) -> Vec<f64> {
    let mut scores = vec![0.0f64; max_overlap.saturating_add(1)];
    for (overlap, slot) in scores.iter_mut().enumerate().skip(10) {
        *slot = signature_overlap_score_from_series(mean_l, grad_l, mean_r, grad_r, overlap);
    }
    scores
}

fn screened_overlaps_from_scores(
    signature_scores: &[f64],
    min_overlap: usize,
    max_overlap: usize,
    limit: usize,
    neighborhood: usize,
    pin_extrema: bool,
) -> Vec<usize> {
    if max_overlap < min_overlap || max_overlap >= signature_scores.len() {
        return Vec::new();
    }

    let mut signature_ranks: Vec<(usize, f64)> = (min_overlap..=max_overlap)
        .map(|overlap| (overlap, signature_scores[overlap]))
        .collect();
    signature_ranks.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut overlaps = BTreeSet::<usize>::new();
    for (overlap, _) in signature_ranks.iter().take(limit) {
        let start = overlap.saturating_sub(neighborhood).max(min_overlap);
        let end = (*overlap + neighborhood).min(max_overlap);
        for candidate in start..=end {
            overlaps.insert(candidate);
        }
    }

    if pin_extrema {
        // Keep the search honest when signatures are flat by pinning the extrema.
        overlaps.insert(min_overlap);
        overlaps.insert(max_overlap);
    }

    overlaps.into_iter().collect()
}

fn expanded_overlap_search_limit(primary_max_overlap: usize, w_l: usize, w_r: usize) -> usize {
    let width_limit = w_l.min(w_r);
    let plausible_limit = ((width_limit as f64) * MAX_PLAUSIBLE_OVERLAP_FRACTION).floor() as usize;
    if plausible_limit <= primary_max_overlap {
        return primary_max_overlap.min(width_limit);
    }
    let target = ((width_limit as f64) * 0.72).round() as usize;
    target
        .max(primary_max_overlap)
        .min(plausible_limit)
        .min(width_limit)
}

fn select_distinct_translation_candidates(
    candidates: &[TranslationSearchResult],
    limit: usize,
) -> Vec<TranslationSearchResult> {
    let mut distinct = Vec::<TranslationSearchResult>::new();
    for candidate in candidates {
        let too_close = distinct.iter().any(|existing| {
            existing.overlap.abs_diff(candidate.overlap) <= CANDIDATE_OVERLAP_DIVERSITY_PX
                && (existing.y_offset - candidate.y_offset).unsigned_abs() <= 2
        });
        if too_close {
            continue;
        }
        distinct.push(*candidate);
        if distinct.len() >= limit {
            break;
        }
    }
    distinct
}

fn search_translation_candidates_in_overlaps(
    gray_l: &Array2<f64>,
    gray_r: &Array2<f64>,
    signature_scores: &[f64],
    max_y_offset: i32,
    overlaps: &[usize],
    search_band: SearchBand,
) -> Vec<TranslationSearchResult> {
    let h = gray_l.dim().0.min(gray_r.dim().0);
    let w_l = gray_l.dim().1;
    let w_r = gray_r.dim().1;
    let mut candidates: Vec<TranslationSearchResult> = overlaps
        .par_iter()
        .filter_map(|&overlap| {
            let signature_score = signature_scores.get(overlap).copied().unwrap_or(0.0);
            let mut best_for_overlap: Option<TranslationSearchResult> = None;

            for dy in -max_y_offset..=max_y_offset {
                let Some(score) = overlap_score_at(gray_l, gray_r, h, w_l, overlap, dy) else {
                    continue;
                };
                let candidate = TranslationSearchResult {
                    search_band,
                    x_offset: w_l - overlap,
                    overlap,
                    y_offset: dy,
                    global_ncc: score,
                    signature_score,
                    overlap_support_score: overlap_support_score(
                        (overlap as f64 / w_l.max(1) as f64)
                            .max(overlap as f64 / w_r.max(1) as f64),
                    ),
                    vertical_offset_plausibility_score: vertical_offset_plausibility_score(
                        dy,
                        max_y_offset,
                    ),
                    plausibility_score: 0.0,
                    correspondence_score: 0.0,
                    prior_weight: 0.0,
                    search_score: 0.0,
                };
                let mut candidate = candidate;
                candidate.plausibility_score = translation_plausibility_score(
                    candidate.overlap_support_score,
                    candidate.vertical_offset_plausibility_score,
                );
                candidate.correspondence_score = translation_search_correspondence_score(
                    candidate.global_ncc,
                    candidate.signature_score,
                );
                candidate.prior_weight = translation_prior_weight(
                    candidate.overlap_support_score,
                    candidate.vertical_offset_plausibility_score,
                );
                candidate.search_score = translation_search_score(
                    candidate.correspondence_score,
                    candidate.overlap_support_score,
                    candidate.vertical_offset_plausibility_score,
                );
                if best_for_overlap
                    .as_ref()
                    .map(|current| compare_translation_candidates(&candidate, current).is_lt())
                    .unwrap_or(true)
                {
                    best_for_overlap = Some(candidate);
                }
            }

            best_for_overlap
        })
        .collect();

    for candidate in &mut candidates {
        candidate.correspondence_score = translation_search_correspondence_score(
            candidate.global_ncc,
            candidate.signature_score,
        );
        candidate.prior_weight = translation_prior_weight(
            candidate.overlap_support_score,
            candidate.vertical_offset_plausibility_score,
        );
        candidate.search_score = translation_search_score(
            candidate.correspondence_score,
            candidate.overlap_support_score,
            candidate.vertical_offset_plausibility_score,
        );
    }
    candidates.sort_by(compare_translation_candidates);
    candidates
}

fn screened_search_is_sufficient(candidates: &[TranslationSearchResult]) -> bool {
    candidates.iter().take(3).any(|candidate| {
        candidate.global_ncc >= 0.55
            && candidate.plausibility_score >= 0.45
            && candidate.search_score >= 0.30
    })
}

fn should_expand_overlap_search(
    primary_candidates: &[TranslationSearchResult],
    primary_max_overlap: usize,
    expanded_max_overlap: usize,
) -> bool {
    if expanded_max_overlap <= primary_max_overlap {
        return false;
    }

    if !screened_search_is_sufficient(primary_candidates) {
        return true;
    }

    let near_primary_cap = primary_candidates
        .first()
        .map(|candidate| primary_max_overlap.saturating_sub(candidate.overlap) <= 24)
        .unwrap_or(false);
    let search_gap_is_tight = primary_candidates.len() >= 2
        && (primary_candidates[0].search_score - primary_candidates[1].search_score).abs() < 0.015;

    near_primary_cap || search_gap_is_tight
}

fn search_translation_hypotheses(
    gray_l: &Array2<f64>,
    gray_r: &Array2<f64>,
    max_overlap: usize,
    max_y_offset: i32,
) -> Option<TranslationSearchOutcome> {
    let (h_l, w_l) = gray_l.dim();
    let (h_r, w_r) = gray_r.dim();
    let h = h_l.min(h_r);
    if h < 10 || w_l == 0 || w_r == 0 {
        return None;
    }

    let primary_max_ovl = max_overlap.min(w_l).min(w_r);
    if primary_max_ovl < 10 {
        return None;
    }

    let expanded_max_ovl = expanded_overlap_search_limit(primary_max_ovl, w_l, w_r);
    let (mean_l, grad_l) = column_feature_series(gray_l);
    let (mean_r, grad_r) = column_feature_series(gray_r);
    let signature_scores =
        signature_overlap_scores(&mean_l, &grad_l, &mean_r, &grad_r, expanded_max_ovl);
    let screened = screened_overlaps_from_scores(
        &signature_scores,
        10,
        primary_max_ovl,
        SIGNATURE_SCREEN_LIMIT,
        SIGNATURE_SCREEN_NEIGHBORHOOD,
        true,
    );
    let screened_candidates = search_translation_candidates_in_overlaps(
        gray_l,
        gray_r,
        &signature_scores,
        max_y_offset,
        &screened,
        SearchBand::PrimaryScreened,
    );

    let exhaustive_fallback_used = !screened_search_is_sufficient(&screened_candidates);
    let primary_candidates = if exhaustive_fallback_used {
        let exhaustive: Vec<usize> = (10..=primary_max_ovl).collect();
        search_translation_candidates_in_overlaps(
            gray_l,
            gray_r,
            &signature_scores,
            max_y_offset,
            &exhaustive,
            SearchBand::PrimaryExhaustive,
        )
    } else {
        screened_candidates
    };
    let expanded_overlap_search_used =
        should_expand_overlap_search(&primary_candidates, primary_max_ovl, expanded_max_ovl);
    let (expanded_screened_overlap_count, mut expanded_candidates) = if expanded_overlap_search_used
    {
        let expanded_overlaps = screened_overlaps_from_scores(
            &signature_scores,
            primary_max_ovl.saturating_add(1),
            expanded_max_ovl,
            EXPANDED_SIGNATURE_SCREEN_LIMIT,
            EXPANDED_SIGNATURE_SCREEN_NEIGHBORHOOD,
            false,
        );
        let count = expanded_overlaps.len();
        let candidates = search_translation_candidates_in_overlaps(
            gray_l,
            gray_r,
            &signature_scores,
            max_y_offset,
            &expanded_overlaps,
            SearchBand::ExpandedSignature,
        );
        (count, candidates)
    } else {
        (0usize, Vec::new())
    };

    let mut final_candidates = primary_candidates;
    final_candidates.append(&mut expanded_candidates);
    final_candidates.sort_by(compare_translation_candidates);
    let distinct_candidates = select_distinct_translation_candidates(
        &final_candidates,
        MAX_TRANSLATION_CANDIDATES_TO_VALIDATE.max(8),
    );
    if distinct_candidates.is_empty() {
        return None;
    }

    Some(TranslationSearchOutcome {
        candidates: distinct_candidates,
        screened_overlap_count: screened.len(),
        expanded_screened_overlap_count,
        evaluated_overlap_count: if exhaustive_fallback_used {
            primary_max_ovl.saturating_sub(9)
        } else {
            screened.len()
        } + expanded_screened_overlap_count,
        exhaustive_fallback_used,
        expanded_overlap_search_used,
        primary_max_overlap_considered: primary_max_ovl,
        max_overlap_considered: if expanded_overlap_search_used {
            expanded_max_ovl
        } else {
            primary_max_ovl
        },
    })
}

fn percentile_from_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn overlap_plausibility_reason(
    overlap_width: usize,
    overlap_fraction_left: f64,
    overlap_fraction_right: f64,
) -> Option<String> {
    let overlap_fraction = overlap_fraction_left.max(overlap_fraction_right);
    if overlap_fraction < MIN_PLAUSIBLE_OVERLAP_FRACTION {
        Some(format!(
            "high NCC on a {}px overlap is likely an edge-only false positive; overlap fraction {:.3}/{:.3} is below the minimum plausible band {:.3}",
            overlap_width,
            overlap_fraction_left,
            overlap_fraction_right,
            MIN_PLAUSIBLE_OVERLAP_FRACTION
        ))
    } else if overlap_fraction > MAX_PLAUSIBLE_OVERLAP_FRACTION {
        Some(format!(
            "overlap fraction {:.3}/{:.3} is too large to be a plausible split-frame seam (max {:.3})",
            overlap_fraction_left,
            overlap_fraction_right,
            MAX_PLAUSIBLE_OVERLAP_FRACTION
        ))
    } else {
        None
    }
}

fn default_validation_metrics() -> StitchValidationMetrics {
    StitchValidationMetrics {
        overlap_width_px: 0,
        comparison_height_px: 0,
        global_ncc_score: 0.0,
        signature_score: 0.0,
        overlap_support_score: 0.0,
        vertical_offset_plausibility_score: 0.0,
        plausibility_score: 0.0,
        evidence_score: 0.0,
        prior_weight: 0.0,
        local_consistency_score: 0.0,
        local_windows_evaluated: 0,
        local_inlier_count: 0,
        local_inlier_ratio: 0.0,
        median_local_ncc: 0.0,
        informative_ncc_threshold: 0.0,
        informative_window_count: 0,
        informative_window_ratio: 0.0,
        informative_inlier_count: 0,
        informative_inlier_ratio: 0.0,
        row_support_ratio: 0.0,
        column_support_ratio: 0.0,
        seam_support_score: 0.0,
        median_residual_px: 0.0,
        p95_residual_px: 0.0,
        median_model_residual_px: 0.0,
        p95_model_residual_px: 0.0,
        mean_dx_px: 0.0,
        mean_dy_px: 0.0,
        overlap_fraction_left: 0.0,
        overlap_fraction_right: 0.0,
        plausibility_ok: false,
        plausibility_reason: Some("no overlap candidate".to_string()),
        homography_inliers: None,
        homography_median_error: None,
        homography_p95_error: None,
    }
}

fn validate_translation_hypothesis(
    gray_l: &Array2<f64>,
    gray_r: &Array2<f64>,
    search: TranslationSearchResult,
    config: &StitchConfig,
) -> StitchValidationMetrics {
    let (h_l, w_l) = gray_l.dim();
    let (h_r, w_r) = gray_r.dim();
    let abs_dy = search.y_offset.unsigned_abs() as usize;
    let cmp_h = h_l.min(h_r).saturating_sub(abs_dy);
    if cmp_h < 16 || search.overlap < 12 {
        let mut metrics = default_validation_metrics();
        metrics.overlap_width_px = search.overlap;
        metrics.comparison_height_px = cmp_h;
        metrics.global_ncc_score = search.global_ncc;
        metrics.signature_score = search.signature_score;
        metrics.overlap_support_score = search.overlap_support_score;
        metrics.vertical_offset_plausibility_score = search.vertical_offset_plausibility_score;
        metrics.plausibility_score = search.plausibility_score;
        metrics.prior_weight = search.prior_weight;
        metrics.informative_ncc_threshold = (search.global_ncc * 0.70).clamp(0.35, 0.8);
        metrics.plausibility_reason = Some("overlap too small for local validation".to_string());
        return metrics;
    }

    let (l_y_start, r_y_start) = if search.y_offset >= 0 {
        (search.y_offset as usize, 0usize)
    } else {
        (0usize, abs_dy)
    };
    let left_strip = gray_l.slice(s![
        l_y_start..(l_y_start + cmp_h),
        (w_l - search.overlap)..w_l
    ]);
    let right_strip = gray_r.slice(s![r_y_start..(r_y_start + cmp_h), 0..search.overlap]);

    let grid_rows = (cmp_h / 72).clamp(2, 6);
    let grid_cols = (search.overlap / 36).clamp(2, 6);
    let window_h = (cmp_h / grid_rows).max(12).min(cmp_h);
    let window_w = (search.overlap / grid_cols).max(10).min(search.overlap);
    let search_radius = 2i32;
    let inlier_ncc_threshold = (search.global_ncc * 0.70).clamp(0.35, 0.8);
    let mut local_matches = Vec::<LocalWindowMatch>::new();

    let mut local_ncc = Vec::<f64>::new();
    let mut residuals = Vec::<f64>::new();
    let mut dx_values = Vec::<f64>::new();
    let mut dy_values = Vec::<f64>::new();
    let mut inliers = 0usize;

    for row_idx in 0..grid_rows {
        let y0 = if grid_rows == 1 || cmp_h == window_h {
            0
        } else {
            row_idx * (cmp_h - window_h) / (grid_rows - 1)
        };
        for col_idx in 0..grid_cols {
            let x0 = if grid_cols == 1 || search.overlap == window_w {
                0
            } else {
                col_idx * (search.overlap - window_w) / (grid_cols - 1)
            };

            let patch_l = left_strip.slice(s![y0..(y0 + window_h), x0..(x0 + window_w)]);
            let mut best_score = f64::NEG_INFINITY;
            let mut best_dx = 0i32;
            let mut best_dy = 0i32;

            for local_dy in -search_radius..=search_radius {
                let ry = y0 as i32 + local_dy;
                if ry < 0 || ry + window_h as i32 > cmp_h as i32 {
                    continue;
                }

                for local_dx in -search_radius..=search_radius {
                    let rx = x0 as i32 + local_dx;
                    if rx < 0 || rx + window_w as i32 > search.overlap as i32 {
                        continue;
                    }

                    let patch_r = right_strip.slice(s![
                        ry as usize..(ry as usize + window_h),
                        rx as usize..(rx as usize + window_w)
                    ]);
                    let score = overlap_score(patch_l.view(), patch_r.view());
                    if score > best_score {
                        best_score = score;
                        best_dx = local_dx;
                        best_dy = local_dy;
                    }
                }
            }

            if best_score.is_finite() {
                let residual = ((best_dx * best_dx + best_dy * best_dy) as f64).sqrt();
                local_matches.push(LocalWindowMatch {
                    row_idx,
                    col_idx,
                    x_norm: if search.overlap == window_w {
                        0.0
                    } else {
                        x0 as f64 / (search.overlap - window_w) as f64
                    },
                    y_norm: if cmp_h == window_h {
                        0.0
                    } else {
                        y0 as f64 / (cmp_h - window_h) as f64
                    },
                    score: best_score,
                    dx: best_dx,
                    dy: best_dy,
                });
                local_ncc.push(best_score);
                residuals.push(residual);
                dx_values.push(best_dx as f64);
                dy_values.push(best_dy as f64);

                if best_score >= inlier_ncc_threshold && residual <= 1.5 {
                    inliers += 1;
                }
            }
        }
    }

    let mut sorted_ncc = local_ncc.clone();
    sorted_ncc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut sorted_residuals = residuals.clone();
    sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean_dx = if dx_values.is_empty() {
        0.0
    } else {
        dx_values.iter().sum::<f64>() / dx_values.len() as f64
    };
    let mean_dy = if dy_values.is_empty() {
        0.0
    } else {
        dy_values.iter().sum::<f64>() / dy_values.len() as f64
    };
    let local_inlier_ratio = if local_ncc.is_empty() {
        0.0
    } else {
        inliers as f64 / local_ncc.len() as f64
    };
    let median_local_ncc = percentile_from_sorted(&sorted_ncc, 0.5);
    let median_residual_px = percentile_from_sorted(&sorted_residuals, 0.5);
    let p95_residual_px = percentile_from_sorted(&sorted_residuals, 0.95);
    let informative_matches: Vec<LocalWindowMatch> = local_matches
        .iter()
        .copied()
        .filter(|m| m.score >= inlier_ncc_threshold)
        .collect();
    let dx_plane = fit_weighted_local_offset_plane(&informative_matches, |m| m.dx as f64)
        .unwrap_or_else(|| Vector3::new(0.0, 0.0, 0.0));
    let dy_plane = fit_weighted_local_offset_plane(&informative_matches, |m| m.dy as f64)
        .unwrap_or_else(|| Vector3::new(0.0, 0.0, 0.0));
    let informative_model_residuals: Vec<f64> = informative_matches
        .iter()
        .map(|m| {
            let dx_residual = evaluate_local_offset_plane(&dx_plane, m) - m.dx as f64;
            let dy_residual = evaluate_local_offset_plane(&dy_plane, m) - m.dy as f64;
            (dx_residual * dx_residual + dy_residual * dy_residual).sqrt()
        })
        .collect();
    let informative_window_count = informative_matches.len();
    let informative_window_ratio = if local_matches.is_empty() {
        0.0
    } else {
        informative_window_count as f64 / local_matches.len() as f64
    };
    let informative_inlier_count = informative_model_residuals
        .iter()
        .filter(|&&residual| residual <= INFORMATIVE_MODEL_INLIER_RESIDUAL_PX)
        .count();
    let informative_inlier_ratio = if informative_window_count == 0 {
        0.0
    } else {
        informative_inlier_count as f64 / informative_window_count as f64
    };
    let mut row_supported = vec![false; grid_rows];
    let mut column_supported = vec![false; grid_cols];
    for (m, &residual) in informative_matches
        .iter()
        .zip(informative_model_residuals.iter())
    {
        if residual <= INFORMATIVE_MODEL_INLIER_RESIDUAL_PX {
            row_supported[m.row_idx] = true;
            column_supported[m.col_idx] = true;
        }
    }
    let row_support_ratio = if grid_rows == 0 {
        0.0
    } else {
        row_supported.iter().filter(|&&supported| supported).count() as f64 / grid_rows as f64
    };
    let column_support_ratio = if grid_cols == 0 {
        0.0
    } else {
        column_supported
            .iter()
            .filter(|&&supported| supported)
            .count() as f64
            / grid_cols as f64
    };
    let mut sorted_informative_ncc = informative_matches
        .iter()
        .map(|m| m.score)
        .collect::<Vec<_>>();
    sorted_informative_ncc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut sorted_model_residuals = informative_model_residuals.clone();
    sorted_model_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let informative_median_local_ncc = percentile_from_sorted(&sorted_informative_ncc, 0.5);
    let median_model_residual_px = percentile_from_sorted(&sorted_model_residuals, 0.5);
    let p95_model_residual_px = percentile_from_sorted(&sorted_model_residuals, 0.95);
    let informative_window_factor =
        smoothstep01((informative_window_ratio - 0.10) / (MIN_INFORMATIVE_WINDOW_RATIO - 0.10));
    let coverage_score = (row_support_ratio * column_support_ratio)
        .sqrt()
        .clamp(0.0, 1.0);
    let informative_residual_score = if informative_window_count == 0 {
        0.0
    } else {
        (1.0 - (p95_model_residual_px / 4.0)).clamp(0.0, 1.0)
    };
    let seam_support_score = if informative_window_count == 0 {
        0.0
    } else {
        (0.25 * informative_window_factor
            + 0.25 * informative_inlier_ratio
            + 0.20 * coverage_score
            + 0.15 * informative_median_local_ncc
            + 0.15 * informative_residual_score)
            .clamp(0.0, 1.0)
    };
    let legacy_local_consistency_score = if local_ncc.is_empty() {
        0.0
    } else {
        let residual_score = (1.0 - (p95_residual_px / 4.0)).clamp(0.0, 1.0);
        (0.45 * local_inlier_ratio + 0.35 * median_local_ncc + 0.20 * residual_score)
            .clamp(0.0, 1.0)
    };
    let local_consistency_score =
        (0.40 * legacy_local_consistency_score + 0.60 * seam_support_score).clamp(0.0, 1.0);
    let overlap_fraction_left = search.overlap as f64 / w_l.max(1) as f64;
    let overlap_fraction_right = search.overlap as f64 / w_r.max(1) as f64;
    let vertical_offset_plausibility_score =
        vertical_offset_plausibility_score(search.y_offset, config.max_y_offset);
    let plausibility_score = translation_plausibility_score(
        search.overlap_support_score,
        vertical_offset_plausibility_score,
    );
    let plausibility_reason = if search.y_offset.unsigned_abs() > config.max_y_offset as u32 {
        Some(format!(
            "vertical offset {} exceeded search limit {}",
            search.y_offset, config.max_y_offset
        ))
    } else {
        overlap_plausibility_reason(
            search.overlap,
            overlap_fraction_left,
            overlap_fraction_right,
        )
    };

    let evidence_score = translation_evidence_score(
        search.global_ncc,
        search.signature_score,
        local_consistency_score,
    );
    let prior_weight = translation_prior_weight(
        search.overlap_support_score,
        vertical_offset_plausibility_score,
    );

    StitchValidationMetrics {
        overlap_width_px: search.overlap,
        comparison_height_px: cmp_h,
        global_ncc_score: search.global_ncc,
        signature_score: search.signature_score,
        overlap_support_score: search.overlap_support_score,
        vertical_offset_plausibility_score,
        plausibility_score,
        evidence_score,
        prior_weight,
        local_consistency_score,
        local_windows_evaluated: local_ncc.len(),
        local_inlier_count: inliers,
        local_inlier_ratio,
        median_local_ncc,
        informative_ncc_threshold: inlier_ncc_threshold,
        informative_window_count,
        informative_window_ratio,
        informative_inlier_count,
        informative_inlier_ratio,
        row_support_ratio,
        column_support_ratio,
        seam_support_score,
        median_residual_px,
        p95_residual_px,
        median_model_residual_px,
        p95_model_residual_px,
        mean_dx_px: mean_dx,
        mean_dy_px: mean_dy,
        overlap_fraction_left,
        overlap_fraction_right,
        plausibility_ok: plausibility_reason.is_none(),
        plausibility_reason,
        homography_inliers: None,
        homography_median_error: None,
        homography_p95_error: None,
    }
}

fn objective_score(validation: &StitchValidationMetrics) -> f64 {
    translation_objective_score(
        validation.evidence_score,
        validation.overlap_support_score,
        validation.vertical_offset_plausibility_score,
    )
}

fn narrow_seam_global_ncc_floor(config: &StitchConfig) -> f64 {
    (config.min_ncc_score - 0.10).max(0.50)
}

fn narrow_seam_objective_floor(config: &StitchConfig) -> f64 {
    (config.min_objective_score - NARROW_SEAM_OBJECTIVE_MARGIN).max(0.40)
}

fn narrow_overlap_seam_acceptance_reason(
    validation: &StitchValidationMetrics,
    config: &StitchConfig,
) -> Option<String> {
    let overlap_fraction = validation
        .overlap_fraction_left
        .max(validation.overlap_fraction_right);
    let objective = objective_score(validation);
    let min_informative_windows = MIN_INFORMATIVE_LOCAL_WINDOWS.max(config.min_validation_windows);

    if !validation.plausibility_ok
        || overlap_fraction < MIN_PLAUSIBLE_OVERLAP_FRACTION
        || overlap_fraction > NARROW_SEAM_RESCUE_MAX_OVERLAP_FRACTION
        || validation.global_ncc_score < narrow_seam_global_ncc_floor(config)
        || validation.signature_score < MIN_NARROW_SEAM_SIGNATURE_SCORE
        || validation.informative_window_count < min_informative_windows
        || validation.informative_window_ratio < MIN_INFORMATIVE_WINDOW_RATIO
        || validation.informative_inlier_ratio < config.min_local_inlier_ratio
        || validation.row_support_ratio < MIN_NARROW_SEAM_ROW_SUPPORT_RATIO
        || validation.column_support_ratio < MIN_NARROW_SEAM_COLUMN_SUPPORT_RATIO
        || validation.seam_support_score < MIN_NARROW_SEAM_SEAM_SUPPORT_SCORE
        || validation.p95_model_residual_px > MAX_INFORMATIVE_MODEL_P95_RESIDUAL_PX
        || objective < narrow_seam_objective_floor(config)
    {
        return None;
    }

    Some(format!(
        "accepted via narrow-overlap seam support: NCC {:.3} is below the default {:.3}, but {} informative windows ({:.1}% of local probes) support a coherent seam across {:.0}% of row bands and {:.0}% of column bands; seam-support score {:.3}, signature {:.3}, objective {:.3}",
        validation.global_ncc_score,
        config.min_ncc_score,
        validation.informative_window_count,
        validation.informative_window_ratio * 100.0,
        validation.row_support_ratio * 100.0,
        validation.column_support_ratio * 100.0,
        validation.seam_support_score,
        validation.signature_score,
        objective
    ))
}

fn translation_rejection_reason(
    validation: &StitchValidationMetrics,
    config: &StitchConfig,
    acceptance_reason: Option<&str>,
) -> Option<String> {
    if acceptance_reason.is_some() {
        return None;
    }
    if validation.global_ncc_score < config.min_ncc_score {
        return Some(format!(
            "global NCC {:.3} below threshold {:.3}",
            validation.global_ncc_score, config.min_ncc_score
        ));
    }
    if validation.local_windows_evaluated < config.min_validation_windows {
        return Some(format!(
            "only {} local windows validated; need at least {}",
            validation.local_windows_evaluated, config.min_validation_windows
        ));
    }
    if validation.plausibility_ok
        && validation.overlap_support_score >= 0.55
        && validation.global_ncc_score >= (config.min_ncc_score + 0.18).min(0.85)
        && validation.signature_score >= 0.70
        && validation.local_consistency_score >= 0.45
        && validation.vertical_offset_plausibility_score >= 0.55
    {
        return None;
    }
    if validation.overlap_support_score < 0.25 {
        return Some(format!(
            "high NCC on a {}px overlap is not sufficient support; overlap-support score {:.3} is too weak",
            validation.overlap_width_px, validation.overlap_support_score
        ));
    }
    if validation.local_inlier_ratio < config.min_local_inlier_ratio {
        return Some(format!(
            "local inlier ratio {:.3} below threshold {:.3}",
            validation.local_inlier_ratio, config.min_local_inlier_ratio
        ));
    }
    if validation.median_residual_px > config.max_local_median_residual {
        return Some(format!(
            "median local residual {:.2}px exceeded {:.2}px",
            validation.median_residual_px, config.max_local_median_residual
        ));
    }
    if validation.p95_residual_px > config.max_local_p95_residual {
        return Some(format!(
            "p95 local residual {:.2}px exceeded {:.2}px",
            validation.p95_residual_px, config.max_local_p95_residual
        ));
    }
    if !validation.plausibility_ok {
        return Some(
            validation
                .plausibility_reason
                .clone()
                .unwrap_or_else(|| "translation plausibility check failed".to_string()),
        );
    }
    let objective = objective_score(validation);
    if objective < config.min_objective_score {
        return Some(format!(
            "objective score {:.3} below threshold {:.3}",
            objective, config.min_objective_score
        ));
    }
    None
}

fn compare_evaluated_translation_candidates(
    candidate: &EvaluatedTranslationCandidate,
    current: &EvaluatedTranslationCandidate,
) -> std::cmp::Ordering {
    (current.rejection_reason.is_none() as u8)
        .cmp(&(candidate.rejection_reason.is_none() as u8))
        .then_with(|| {
            current
                .objective_score
                .partial_cmp(&candidate.objective_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| compare_translation_candidates(&candidate.search, &current.search))
}

fn default_search_diagnostics() -> StitchSearchDiagnostics {
    StitchSearchDiagnostics {
        screening_strategy: "no_overlap_candidate".to_string(),
        screened_overlap_count: 0,
        expanded_screened_overlap_count: 0,
        evaluated_overlap_count: 0,
        exhaustive_fallback_used: false,
        expanded_overlap_search_used: false,
        primary_max_overlap_considered: 0,
        max_overlap_considered: 0,
        selected_candidate_rank: None,
        selected_candidate_search_score: None,
        search_score_gap_to_runner_up: None,
        selection_reason: None,
        top_candidates: Vec::new(),
        evaluated_candidates: Vec::new(),
    }
}

fn build_selection_reason(
    outcome: &TranslationSearchOutcome,
    evaluated: &[EvaluatedTranslationCandidate],
    selected_candidate_rank: Option<usize>,
    selected: &EvaluatedTranslationCandidate,
) -> Option<String> {
    let gap_note = if outcome.candidates.len() >= 2 {
        let gap = (outcome.candidates[0].search_score - outcome.candidates[1].search_score).abs();
        if gap < 0.01 {
            format!(" Search-score gap to the runner-up was only {:.3}.", gap)
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    if outcome.expanded_overlap_search_used
        && matches!(selected.search.search_band, SearchBand::ExpandedSignature)
    {
        let best_primary = evaluated.iter().find(|candidate| {
            !matches!(candidate.search.search_band, SearchBand::ExpandedSignature)
        });
        if let Some(primary) = best_primary {
            return Some(format!(
                "Expanded signature-guided search widened the overlap sweep from {}px to {}px and surfaced the selected {}px candidate. It outranked the best <= {}px candidate on objective {:.3} vs {:.3}, correspondence evidence {:.3} vs {:.3}, prior weight {:.3} vs {:.3}, local consistency {:.3} vs {:.3}, and global NCC {:.3} vs {:.3}.{}",
                outcome.primary_max_overlap_considered,
                outcome.max_overlap_considered,
                selected.search.overlap,
                outcome.primary_max_overlap_considered,
                selected.objective_score,
                primary.objective_score,
                selected.validation.evidence_score,
                primary.validation.evidence_score,
                selected.validation.prior_weight,
                primary.validation.prior_weight,
                selected.validation.local_consistency_score,
                primary.validation.local_consistency_score,
                selected.validation.global_ncc_score,
                primary.validation.global_ncc_score,
                gap_note
            ));
        }
        return Some(format!(
            "Expanded signature-guided search widened the overlap sweep from {}px to {}px and surfaced the selected {}px candidate.{}",
            outcome.primary_max_overlap_considered,
            outcome.max_overlap_considered,
            selected.search.overlap,
            gap_note
        ));
    }

    if let Some(rank) = selected_candidate_rank {
        if rank > 0 {
            let search_leader = outcome.candidates.first()?;
            let search_leader_eval = evaluated
                .iter()
                .find(|candidate| candidate.search == *search_leader);
            if let Some(leader) = search_leader_eval {
                return Some(format!(
                    "Validation promoted search rank {} over search rank 1: objective {:.3} vs {:.3}, correspondence evidence {:.3} vs {:.3}, prior weight {:.3} vs {:.3}, local consistency {:.3} vs {:.3}, and global NCC {:.3} vs {:.3}.{}",
                    rank + 1,
                    selected.objective_score,
                    leader.objective_score,
                    selected.validation.evidence_score,
                    leader.validation.evidence_score,
                    selected.validation.prior_weight,
                    leader.validation.prior_weight,
                    selected.validation.local_consistency_score,
                    leader.validation.local_consistency_score,
                    selected.validation.global_ncc_score,
                    leader.validation.global_ncc_score,
                    gap_note
                ));
            }
            return Some(format!(
                "Validation promoted search rank {} over search rank 1.{}",
                rank + 1,
                gap_note
            ));
        }
    }

    Some(format!(
        "Search rank 1 remained the best-scoring candidate after validation: objective {:.3}, correspondence evidence {:.3}, prior weight {:.3}, local consistency {:.3}, global NCC {:.3}.{}",
        selected.objective_score,
        selected.validation.evidence_score,
        selected.validation.prior_weight,
        selected.validation.local_consistency_score,
        selected.validation.global_ncc_score,
        gap_note
    ))
}

fn build_search_diagnostics(
    outcome: &TranslationSearchOutcome,
    w_l: usize,
    w_r: usize,
    evaluated: &[EvaluatedTranslationCandidate],
    selected: Option<&EvaluatedTranslationCandidate>,
) -> StitchSearchDiagnostics {
    let selected_candidate_rank = selected.and_then(|needle| {
        outcome
            .candidates
            .iter()
            .position(|candidate| candidate == &needle.search)
    });
    let search_score_gap_to_runner_up = if outcome.candidates.len() >= 2 {
        Some((outcome.candidates[0].search_score - outcome.candidates[1].search_score).abs())
    } else {
        None
    };
    let selection_reason = selected.and_then(|candidate| {
        build_selection_reason(outcome, evaluated, selected_candidate_rank, candidate)
    });

    StitchSearchDiagnostics {
        screening_strategy: if outcome.exhaustive_fallback_used
            && outcome.expanded_overlap_search_used
        {
            "signature_screen_then_primary_exhaustive_plus_expanded_signature".to_string()
        } else if outcome.exhaustive_fallback_used {
            "signature_screen_then_primary_exhaustive".to_string()
        } else if outcome.expanded_overlap_search_used {
            "signature_screen_plus_expanded_signature".to_string()
        } else {
            "signature_screen_only".to_string()
        },
        screened_overlap_count: outcome.screened_overlap_count,
        expanded_screened_overlap_count: outcome.expanded_screened_overlap_count,
        evaluated_overlap_count: outcome.evaluated_overlap_count,
        exhaustive_fallback_used: outcome.exhaustive_fallback_used,
        expanded_overlap_search_used: outcome.expanded_overlap_search_used,
        primary_max_overlap_considered: outcome.primary_max_overlap_considered,
        max_overlap_considered: outcome.max_overlap_considered,
        selected_candidate_rank: selected_candidate_rank.map(|rank| rank + 1),
        selected_candidate_search_score: selected.map(|candidate| candidate.search.search_score),
        search_score_gap_to_runner_up,
        selection_reason,
        top_candidates: outcome
            .candidates
            .iter()
            .take(MAX_REPORTED_TRANSLATION_CANDIDATES)
            .enumerate()
            .map(|(rank, candidate)| StitchSearchCandidate {
                rank: rank + 1,
                search_band: candidate.search_band.as_str().to_string(),
                overlap_width: candidate.overlap,
                x_offset: candidate.x_offset,
                vertical_offset: candidate.y_offset,
                global_ncc_score: candidate.global_ncc,
                signature_score: candidate.signature_score,
                overlap_fraction_left: candidate.overlap as f64 / w_l.max(1) as f64,
                overlap_fraction_right: candidate.overlap as f64 / w_r.max(1) as f64,
                overlap_support_score: candidate.overlap_support_score,
                vertical_offset_plausibility_score: candidate.vertical_offset_plausibility_score,
                plausibility_score: candidate.plausibility_score,
                correspondence_score: candidate.correspondence_score,
                prior_weight: candidate.prior_weight,
                search_score: candidate.search_score,
            })
            .collect(),
        evaluated_candidates: evaluated
            .iter()
            .take(MAX_REPORTED_EVALUATED_CANDIDATES)
            .enumerate()
            .map(|(validation_rank, candidate)| StitchEvaluatedCandidate {
                validation_rank: validation_rank + 1,
                search_rank: outcome
                    .candidates
                    .iter()
                    .position(|search_candidate| search_candidate == &candidate.search)
                    .map(|rank| rank + 1),
                search_band: candidate.search.search_band.as_str().to_string(),
                overlap_width: candidate.search.overlap,
                x_offset: candidate.search.x_offset,
                vertical_offset: candidate.search.y_offset,
                objective_score: candidate.objective_score,
                accepted: candidate.rejection_reason.is_none(),
                acceptance_reason: candidate.acceptance_reason.clone(),
                rejection_reason: candidate.rejection_reason.clone(),
                global_ncc_score: candidate.validation.global_ncc_score,
                signature_score: candidate.validation.signature_score,
                overlap_support_score: candidate.validation.overlap_support_score,
                vertical_offset_plausibility_score: candidate
                    .validation
                    .vertical_offset_plausibility_score,
                plausibility_score: candidate.validation.plausibility_score,
                evidence_score: candidate.validation.evidence_score,
                prior_weight: candidate.validation.prior_weight,
                local_consistency_score: candidate.validation.local_consistency_score,
                informative_window_count: candidate.validation.informative_window_count,
                informative_window_ratio: candidate.validation.informative_window_ratio,
                informative_inlier_ratio: candidate.validation.informative_inlier_ratio,
                row_support_ratio: candidate.validation.row_support_ratio,
                column_support_ratio: candidate.validation.column_support_ratio,
                seam_support_score: candidate.validation.seam_support_score,
                local_inlier_ratio: candidate.validation.local_inlier_ratio,
                median_local_ncc: candidate.validation.median_local_ncc,
                median_residual_px: candidate.validation.median_residual_px,
                p95_residual_px: candidate.validation.p95_residual_px,
                p95_model_residual_px: candidate.validation.p95_model_residual_px,
            })
            .collect(),
    }
}

fn evaluate_translation_hypothesis(
    left: &Array3<u16>,
    right: &Array3<u16>,
    order: StitchOrder,
    config: &StitchConfig,
) -> StitchHypothesis {
    let (w_l, w_r) = (left.dim().1, right.dim().1);
    let gray_l = to_grayscale(left);
    let gray_r = to_grayscale(right);
    let Some(search_outcome) =
        search_translation_hypotheses(&gray_l, &gray_r, config.max_overlap, config.max_y_offset)
    else {
        return StitchHypothesis {
            ordering: order.label().to_string(),
            x_offset: None,
            overlap_width: None,
            vertical_offset: None,
            transform_model: "translation".to_string(),
            objective_score: 0.0,
            accepted: false,
            acceptance_reason: None,
            rejection_reason: Some("no overlap found".to_string()),
            search: default_search_diagnostics(),
            validation: default_validation_metrics(),
        };
    };

    let mut evaluated: Vec<EvaluatedTranslationCandidate> = search_outcome
        .candidates
        .iter()
        .take(MAX_TRANSLATION_CANDIDATES_TO_VALIDATE)
        .map(|search| {
            let validation = validate_translation_hypothesis(&gray_l, &gray_r, *search, config);
            let objective_score = objective_score(&validation);
            let acceptance_reason = narrow_overlap_seam_acceptance_reason(&validation, config);
            let rejection_reason = if search.overlap < config.min_overlap {
                Some(format!(
                    "overlap {} below minimum {}",
                    search.overlap, config.min_overlap
                ))
            } else {
                translation_rejection_reason(&validation, config, acceptance_reason.as_deref())
            };

            EvaluatedTranslationCandidate {
                search: *search,
                validation,
                objective_score,
                acceptance_reason,
                rejection_reason,
            }
        })
        .collect();
    evaluated.sort_by(compare_evaluated_translation_candidates);

    let best = evaluated
        .first()
        .expect("search outcome should contain at least one candidate");
    let search_diagnostics =
        build_search_diagnostics(&search_outcome, w_l, w_r, &evaluated, Some(best));

    StitchHypothesis {
        ordering: order.label().to_string(),
        x_offset: Some(best.search.x_offset),
        overlap_width: Some(best.search.overlap),
        vertical_offset: Some(best.search.y_offset),
        transform_model: "translation".to_string(),
        objective_score: best.objective_score,
        accepted: best.rejection_reason.is_none(),
        acceptance_reason: best.acceptance_reason.clone(),
        rejection_reason: best.rejection_reason.clone(),
        search: search_diagnostics,
        validation: best.validation.clone(),
    }
}

fn acceptance_thresholds_json(config: &StitchConfig) -> serde_json::Value {
    serde_json::json!({
        "min_ncc_score": config.min_ncc_score,
        "min_overlap": config.min_overlap,
        "min_plausible_overlap_fraction": MIN_PLAUSIBLE_OVERLAP_FRACTION,
        "max_plausible_overlap_fraction": MAX_PLAUSIBLE_OVERLAP_FRACTION,
        "min_objective_score": config.min_objective_score,
        "min_local_inlier_ratio": config.min_local_inlier_ratio,
        "max_local_median_residual": config.max_local_median_residual,
        "max_local_p95_residual": config.max_local_p95_residual,
        "min_validation_windows": config.min_validation_windows,
        "max_y_offset": config.max_y_offset,
        "narrow_overlap_rescue": {
            "max_overlap_fraction": NARROW_SEAM_RESCUE_MAX_OVERLAP_FRACTION,
            "min_global_ncc": narrow_seam_global_ncc_floor(config),
            "min_signature_score": MIN_NARROW_SEAM_SIGNATURE_SCORE,
            "min_informative_windows": MIN_INFORMATIVE_LOCAL_WINDOWS.max(config.min_validation_windows),
            "min_informative_window_ratio": MIN_INFORMATIVE_WINDOW_RATIO,
            "min_informative_inlier_ratio": config.min_local_inlier_ratio,
            "min_row_support_ratio": MIN_NARROW_SEAM_ROW_SUPPORT_RATIO,
            "min_column_support_ratio": MIN_NARROW_SEAM_COLUMN_SUPPORT_RATIO,
            "min_seam_support_score": MIN_NARROW_SEAM_SEAM_SUPPORT_SCORE,
            "max_p95_model_residual_px": MAX_INFORMATIVE_MODEL_P95_RESIDUAL_PX,
            "min_objective_score": narrow_seam_objective_floor(config),
        },
    })
}

/// Find the best overlap between the right edge of `left` and the left edge of
/// `right` using normalized cross-correlation.
///
/// Searches a 2D window: horizontal overlaps from 10..=max_overlap and vertical offsets
/// from -max_y_offset..=+max_y_offset to account for mechanical stepper drift.
///
/// Returns `Some((x_offset, y_offset, score))` where `x_offset` is the column in `left`
/// coordinates where `right` starts, `y_offset` is the vertical shift applied to `right`
/// (positive = right image shifted down), or `None` if no good match is found.
pub fn find_overlap_ncc(
    left: &Array3<u16>,
    right: &Array3<u16>,
    max_overlap: usize,
    max_y_offset: i32,
) -> Option<(usize, i32, f64)> {
    let (h_l, w_l, _) = left.dim();
    let (h_r, w_r, _) = right.dim();
    let h = h_l.min(h_r);
    if h < 10 || w_l == 0 || w_r == 0 {
        return None;
    }

    let gray_l = to_grayscale(left);
    let gray_r = to_grayscale(right);
    let (mean_l, grad_l) = column_feature_series(&gray_l);
    let (mean_r, grad_r) = column_feature_series(&gray_r);
    let max_ovl = max_overlap.min(w_l).min(w_r);
    if max_ovl < 10 {
        return None;
    }

    let signature_scores = signature_overlap_scores(&mean_l, &grad_l, &mean_r, &grad_r, max_ovl);
    let exhaustive: Vec<usize> = (10..=max_ovl).collect();
    let best = search_translation_candidates_in_overlaps(
        &gray_l,
        &gray_r,
        &signature_scores,
        max_y_offset,
        &exhaustive,
        SearchBand::PrimaryExhaustive,
    )
    .into_iter()
    .min_by(compare_raw_translation_candidates);

    best.and_then(|search| {
        if search.global_ncc > 0.3 {
            Some((search.x_offset, search.y_offset, search.global_ncc))
        } else {
            None
        }
    })
}

/// Score a left-right hypothesis. Returns the NCC score or 0.0 if no overlap found.
pub fn score_hypothesis(
    left: &Array3<u16>,
    right: &Array3<u16>,
    max_overlap: usize,
    max_y_offset: i32,
) -> f64 {
    match find_overlap_ncc(left, right, max_overlap, max_y_offset) {
        Some((_, _, score)) => score,
        None => 0.0,
    }
}

/// Stitch two image components together, automatically detecting the correct order.
///
/// Accounts for vertical mechanical drift by searching a 2D window and placing the
/// right component at the detected y_offset on an expanded canvas.
pub fn stitch_components(
    comp1: &Array3<u16>,
    comp2: &Array3<u16>,
    config: &StitchConfig,
) -> StitchResult {
    let opencv_available = opencv_match::is_available();
    let mut report_warnings = Vec::<String>::new();
    if config.use_opencv && !opencv_available {
        report_warnings.push(
            "OpenCV was requested at runtime, but this binary was built without the `use-opencv` feature."
                .to_string(),
        );
    }
    if matches!(config.transform_mode, TransformMode::Affine) {
        report_warnings.push(
            "Affine estimation is not implemented; any feature-based upgrade will be reported as a homography fallback."
                .to_string(),
        );
    }
    if !config.use_opencv
        && matches!(
            config.transform_mode,
            TransformMode::Affine | TransformMode::Homography
        )
    {
        report_warnings.push(
            "The requested transform mode needs the OpenCV backend at runtime; falling back to translation stitching."
                .to_string(),
        );
    }
    let mut hypotheses = vec![
        evaluate_translation_hypothesis(comp1, comp2, StitchOrder::LeftRight, config),
        evaluate_translation_hypothesis(comp2, comp1, StitchOrder::RightLeft, config),
    ];
    if let Some(debug_dir) = &config.debug_dir {
        save_hypothesis_debug_artifacts(comp1, comp2, &hypotheses, debug_dir);
    }

    let best_index = {
        let rank0 = (
            hypotheses[0].accepted as u8,
            hypotheses[0].objective_score,
            hypotheses[0]
                .search
                .selected_candidate_search_score
                .unwrap_or(0.0),
            hypotheses[0].validation.global_ncc_score,
            hypotheses[0]
                .vertical_offset
                .map(|dy| -(dy.unsigned_abs() as i32))
                .unwrap_or(i32::MIN),
        );
        let rank1 = (
            hypotheses[1].accepted as u8,
            hypotheses[1].objective_score,
            hypotheses[1]
                .search
                .selected_candidate_search_score
                .unwrap_or(0.0),
            hypotheses[1].validation.global_ncc_score,
            hypotheses[1]
                .vertical_offset
                .map(|dy| -(dy.unsigned_abs() as i32))
                .unwrap_or(i32::MIN),
        );
        if rank1 > rank0 {
            1
        } else {
            0
        }
    };
    let other_index = 1 - best_index;
    let order_gap =
        (hypotheses[best_index].objective_score - hypotheses[other_index].objective_score).abs();

    let best_order = if best_index == 0 {
        StitchOrder::LeftRight
    } else {
        StitchOrder::RightLeft
    };
    let (left, right) = if matches!(best_order, StitchOrder::LeftRight) {
        (comp1, comp2)
    } else {
        (comp2, comp1)
    };

    let best_x_offset = hypotheses[best_index].x_offset.unwrap_or(0);
    let best_y_offset = hypotheses[best_index].vertical_offset.unwrap_or(0);
    let best_overlap = hypotheses[best_index].overlap_width.unwrap_or(0);
    let best_ncc = hypotheses[best_index].validation.global_ncc_score;
    let translation_was_accepted = hypotheses[best_index].accepted;
    let attempt_homography = config.use_opencv
        && opencv_available
        && (matches!(
            config.transform_mode,
            TransformMode::Affine | TransformMode::Homography
        ) || !hypotheses[best_index].accepted);

    if attempt_homography && best_overlap > 0 {
        let (h_l, w_l, _) = left.dim();
        let (h_r, w_r, _) = right.dim();
        let strip_l = left.slice(s![.., (w_l - best_overlap).., ..]).to_owned();
        let strip_r = right.slice(s![.., ..best_overlap, ..]).to_owned();
        let homography =
            opencv_match::find_homography_robust(&strip_l, &strip_r, config.input_bit_depth);

        if let Some(ref mr) = homography {
            let plausibility =
                assess_transform_plausibility(mr.transform, best_overlap, h_l.min(h_r), w_r, h_r);
            hypotheses[best_index].validation.homography_inliers = Some(mr.inliers);
            hypotheses[best_index].validation.homography_median_error = Some(mr.median_error);
            hypotheses[best_index].validation.homography_p95_error = Some(mr.p95_error);

            if mr.inliers >= 12
                && mr.median_error <= 2.5
                && mr.p95_error <= 6.0
                && plausibility.is_ok()
            {
                let h_inv = match invert_3x3(mr.transform) {
                    Some(inv) => inv,
                    None => {
                        report_warnings.push(
                            "Homography matrix was singular; falling back to translation stitching."
                                .to_string(),
                        );
                        if !translation_was_accepted {
                            let reason = hypotheses[best_index]
                                .rejection_reason
                                .clone()
                                .unwrap_or_else(|| {
                                    "best stitch hypothesis rejected after scoring".to_string()
                                });
                            let report = stitch_failure_report(
                                config,
                                opencv_available,
                                &hypotheses,
                                best_index,
                                reason,
                                report_warnings,
                            );
                            return StitchResult {
                                result: None,
                                x_offset: best_x_offset as i32,
                                y_offset: best_y_offset,
                                ncc_score: best_ncc,
                                order: StitchOrder::NoStitch,
                                report,
                            };
                        }
                        return stitch_ncc_fallback(
                            left,
                            right,
                            best_x_offset,
                            best_y_offset,
                            best_overlap,
                            best_order,
                            config,
                            &hypotheses,
                            best_index,
                            report_warnings,
                        );
                    }
                };

                let t_l_inv: [[f64; 3]; 3] = [
                    [1.0, 0.0, best_x_offset as f64],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ];
                let m_right_to_canvas = mat_mul_3x3(t_l_inv, h_inv);
                let corners_r = [
                    [0.0, 0.0],
                    [w_r as f64, 0.0],
                    [w_r as f64, h_r as f64],
                    [0.0, h_r as f64],
                ];
                let mut min_x = 0.0f64;
                let mut min_y = 0.0f64;
                let mut max_x = w_l as f64;
                let mut max_y = h_l as f64;
                for c in &corners_r {
                    let (tx, ty) = transform_point(m_right_to_canvas, c[0], c[1]);
                    min_x = min_x.min(tx);
                    min_y = min_y.min(ty);
                    max_x = max_x.max(tx);
                    max_y = max_y.max(ty);
                }

                let shift_x = if min_x < 0.0 { -min_x } else { 0.0 };
                let shift_y = if min_y < 0.0 { -min_y } else { 0.0 };
                let l_x0 = shift_x.ceil() as usize;
                let l_y0 = shift_y.ceil() as usize;
                let canvas_w = ((max_x + l_x0 as f64).max(0.0).ceil() as usize).max(l_x0 + w_l);
                let canvas_h = ((max_y + l_y0 as f64).max(0.0).ceil() as usize).max(l_y0 + h_l);

                let t_shift: [[f64; 3]; 3] = [
                    [1.0, 0.0, l_x0 as f64],
                    [0.0, 1.0, l_y0 as f64],
                    [0.0, 0.0, 1.0],
                ];
                let m_warp = mat_mul_3x3(t_shift, m_right_to_canvas);

                let abs_dy = best_y_offset.unsigned_abs() as usize;
                let cmp_h = h_l.min(h_r).saturating_sub(abs_dy);
                let (gain_l_y0, gain_r_y0) = if best_y_offset >= 0 {
                    (best_y_offset as usize, 0usize)
                } else {
                    (0usize, abs_dy)
                };
                let aligned_strip_l = left
                    .slice(s![
                        gain_l_y0..(gain_l_y0 + cmp_h),
                        (w_l - best_overlap)..w_l,
                        ..
                    ])
                    .to_owned();
                let aligned_strip_r = right
                    .slice(s![gain_r_y0..(gain_r_y0 + cmp_h), 0..best_overlap, ..])
                    .to_owned();
                let gain = compute_exposure_gain(&aligned_strip_l, &aligned_strip_r);
                let right_comp = apply_gain(right, &gain);

                let warped_right =
                    match opencv_match::warp_image(&right_comp, &m_warp, canvas_w, canvas_h) {
                        Some(w) => w,
                        None => {
                            report_warnings.push(
                                "OpenCV warp failed; falling back to translation stitching."
                                    .to_string(),
                            );
                            if !translation_was_accepted {
                                let reason = hypotheses[best_index]
                                    .rejection_reason
                                    .clone()
                                    .unwrap_or_else(|| {
                                        "best stitch hypothesis rejected after scoring".to_string()
                                    });
                                let report = stitch_failure_report(
                                    config,
                                    opencv_available,
                                    &hypotheses,
                                    best_index,
                                    reason,
                                    report_warnings,
                                );
                                return StitchResult {
                                    result: None,
                                    x_offset: best_x_offset as i32,
                                    y_offset: best_y_offset,
                                    ncc_score: best_ncc,
                                    order: StitchOrder::NoStitch,
                                    report,
                                };
                            }
                            return stitch_ncc_fallback(
                                left,
                                right,
                                best_x_offset,
                                best_y_offset,
                                best_overlap,
                                best_order,
                                config,
                                &hypotheses,
                                best_index,
                                report_warnings,
                            );
                        }
                    };

                hypotheses[best_index].accepted = true;
                hypotheses[best_index].rejection_reason = None;
                hypotheses[best_index].transform_model =
                    if matches!(config.transform_mode, TransformMode::Affine) {
                        "homography_fallback_for_affine_request".to_string()
                    } else {
                        "homography".to_string()
                    };

                let mut stitched = Array3::<u16>::zeros((canvas_h, canvas_w, 3));
                for y in 0..h_l {
                    for x in 0..w_l {
                        let cy = l_y0 + y;
                        let cx = l_x0 + x;
                        if cy < canvas_h && cx < canvas_w {
                            for c in 0..3 {
                                stitched[[cy, cx, c]] = left[[y, x, c]];
                            }
                        }
                    }
                }

                let blend_w = config.blend_width.min(best_overlap);
                for y in 0..canvas_h {
                    for x in 0..canvas_w {
                        let in_left = y >= l_y0 && y < l_y0 + h_l && x >= l_x0 && x < l_x0 + w_l;
                        let right_nonzero = warped_right[[y, x, 0]] != 0
                            || warped_right[[y, x, 1]] != 0
                            || warped_right[[y, x, 2]] != 0;
                        if in_left && right_nonzero {
                            let left_col = x - l_x0;
                            if left_col >= best_x_offset {
                                let depth = left_col - best_x_offset;
                                let alpha = if blend_w == 0 {
                                    1.0
                                } else {
                                    (depth as f64 / blend_w as f64).min(1.0)
                                };
                                for c in 0..3 {
                                    let lv = stitched[[y, x, c]] as f64;
                                    let rv = warped_right[[y, x, c]] as f64;
                                    stitched[[y, x, c]] = (lv * (1.0 - alpha) + rv * alpha)
                                        .round()
                                        .clamp(0.0, 65535.0)
                                        as u16;
                                }
                            }
                        } else if !in_left && right_nonzero {
                            for c in 0..3 {
                                stitched[[y, x, c]] = warped_right[[y, x, c]];
                            }
                        }
                    }
                }

                let (stitched, crop_info) = apply_target_crop(&stitched, config);
                if let Some(debug_dir) = &config.debug_dir {
                    save_final_seam_overlay(
                        left,
                        right,
                        best_overlap,
                        best_y_offset,
                        best_order,
                        debug_dir,
                    );
                }

                let mut report = PhaseReport::ok(
                    "stitch",
                    hypotheses[best_index].objective_score,
                    serde_json::json!({
                        "decision": "accepted",
                        "method": "homography",
                        "requested_transform": config.transform_mode.as_str(),
                        "transform_model_used": hypotheses[best_index].transform_model,
                        "opencv_requested": config.use_opencv,
                        "opencv_available": opencv_available,
                        "chosen_hypothesis": hypotheses[best_index].ordering,
                        "hypotheses": hypotheses,
                        "acceptance_thresholds": acceptance_thresholds_json(config),
                        "x_offset": best_x_offset,
                        "y_offset": best_y_offset,
                        "overlap": best_overlap,
                        "objective_score": hypotheses[best_index].objective_score,
                        "order_gap": order_gap,
                        "total_width": canvas_w,
                        "canvas_height": canvas_h,
                        "exposure_gain": gain,
                        "homography_matrix": mr.transform,
                        "crop": crop_metrics_json(crop_info),
                    }),
                );
                report.warnings.extend(report_warnings);

                return StitchResult {
                    result: Some(stitched),
                    x_offset: best_x_offset as i32,
                    y_offset: best_y_offset,
                    ncc_score: best_ncc,
                    order: best_order,
                    report,
                };
            }

            if mr.inliers < 12 {
                report_warnings.push(
                    "Homography had too few inliers; falling back to translation stitching."
                        .to_string(),
                );
            }
            if mr.median_error > 2.5 || mr.p95_error > 6.0 {
                report_warnings.push(format!(
                    "Homography residuals were too large (median {:.2}, p95 {:.2}); falling back to translation stitching.",
                    mr.median_error, mr.p95_error
                ));
            }
            if let Err(reason) = plausibility {
                report_warnings.push(format!(
                    "Homography failed plausibility checks ({}); falling back to translation stitching.",
                    reason
                ));
            }
        } else {
            report_warnings.push(
                "OpenCV matching did not produce a homography; falling back to translation stitching."
                    .to_string(),
            );
        }
    }

    if !hypotheses[best_index].accepted {
        let reason = hypotheses[best_index]
            .rejection_reason
            .clone()
            .unwrap_or_else(|| "best stitch hypothesis rejected after scoring".to_string());
        let report = stitch_failure_report(
            config,
            opencv_available,
            &hypotheses,
            best_index,
            reason,
            report_warnings,
        );
        return StitchResult {
            result: None,
            x_offset: best_x_offset as i32,
            y_offset: best_y_offset,
            ncc_score: best_ncc,
            order: StitchOrder::NoStitch,
            report,
        };
    }

    stitch_ncc_fallback(
        left,
        right,
        best_x_offset,
        best_y_offset,
        best_overlap,
        best_order,
        config,
        &hypotheses,
        best_index,
        report_warnings,
    )
}

fn assess_transform_plausibility(
    m: [[f64; 3]; 3],
    overlap: usize,
    overlap_h: usize,
    full_w: usize,
    full_h: usize,
) -> Result<(), String> {
    let col0 = [m[0][0], m[1][0]];
    let col1 = [m[0][1], m[1][1]];
    let scale_x = (col0[0] * col0[0] + col0[1] * col0[1]).sqrt();
    let scale_y = (col1[0] * col1[0] + col1[1] * col1[1]).sqrt();
    let rotation_deg = col0[1].atan2(col0[0]).to_degrees().abs();
    let shear = if scale_x > 1e-6 && scale_y > 1e-6 {
        (col0[0] * col1[0] + col0[1] * col1[1]).abs() / (scale_x * scale_y)
    } else {
        1.0
    };
    let perspective = (m[2][0].abs() * overlap.max(full_w) as f64)
        .max(m[2][1].abs() * overlap_h.max(full_h) as f64);

    if !(0.95..=1.05).contains(&scale_x) || !(0.95..=1.05).contains(&scale_y) {
        return Err(format!(
            "scale out of range ({:.4}, {:.4})",
            scale_x, scale_y
        ));
    }
    if rotation_deg > 6.0 {
        return Err(format!("rotation too large ({:.2} deg)", rotation_deg));
    }
    if shear > 0.08 {
        return Err(format!("shear too large ({:.4})", shear));
    }
    if perspective > 0.02 {
        return Err(format!("perspective too large ({:.5})", perspective));
    }
    Ok(())
}

fn crop_metrics_json(crop_info: Option<CropInfo>) -> serde_json::Value {
    match crop_info {
        Some(crop) => serde_json::json!({
            "x_start": crop.x_start,
            "y_start": crop.y_start,
            "width": crop.width,
            "height": crop.height,
        }),
        None => serde_json::Value::Null,
    }
}

fn aligned_overlap_strips(
    left: &Array3<u16>,
    right: &Array3<u16>,
    overlap: usize,
    y_offset: i32,
) -> Option<(Array3<u16>, Array3<u16>)> {
    let (h_l, w_l, _) = left.dim();
    let (h_r, _, _) = right.dim();
    if overlap == 0 || overlap > w_l {
        return None;
    }

    let abs_dy = y_offset.unsigned_abs() as usize;
    let cmp_h = h_l.min(h_r).saturating_sub(abs_dy);
    if cmp_h == 0 {
        return None;
    }

    let (l_y_start, r_y_start) = if y_offset >= 0 {
        (y_offset as usize, 0usize)
    } else {
        (0usize, abs_dy)
    };

    Some((
        left.slice(s![l_y_start..(l_y_start + cmp_h), (w_l - overlap)..w_l, ..])
            .to_owned(),
        right
            .slice(s![r_y_start..(r_y_start + cmp_h), 0..overlap, ..])
            .to_owned(),
    ))
}

fn seam_overlay_from_strips(left_strip: &Array3<u16>, right_strip: &Array3<u16>) -> Array3<u16> {
    let (h, w, _) = left_strip.dim();
    let mut overlay = Array3::<u16>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            let l_luma = (left_strip[[y, x, 0]] as f64 * 0.2126
                + left_strip[[y, x, 1]] as f64 * 0.7152
                + left_strip[[y, x, 2]] as f64 * 0.0722)
                .round()
                .clamp(0.0, u16::MAX as f64) as u16;
            let r_luma = (right_strip[[y, x, 0]] as f64 * 0.2126
                + right_strip[[y, x, 1]] as f64 * 0.7152
                + right_strip[[y, x, 2]] as f64 * 0.0722)
                .round()
                .clamp(0.0, u16::MAX as f64) as u16;
            overlay[[y, x, 0]] = l_luma;
            overlay[[y, x, 1]] = r_luma;
            overlay[[y, x, 2]] = (((l_luma as u32 + r_luma as u32) / 2) as u16).max(1);
        }
    }
    overlay
}

fn save_debug_tiff(path: PathBuf, img: &Array3<u16>) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(err) = tiff_io::save_tiff_u16(img, &path) {
        log::warn!(
            "failed to save stitch debug artifact {}: {}",
            path.display(),
            err
        );
    }
}

fn save_hypothesis_debug_artifacts(
    comp1: &Array3<u16>,
    comp2: &Array3<u16>,
    hypotheses: &[StitchHypothesis],
    debug_dir: &PathBuf,
) {
    for hypothesis in hypotheses {
        let (left, right, suffix) = match hypothesis.ordering.as_str() {
            "[1|2]" => (comp1, comp2, "12"),
            "[2|1]" => (comp2, comp1, "21"),
            _ => continue,
        };
        let (Some(overlap), Some(y_offset)) =
            (hypothesis.overlap_width, hypothesis.vertical_offset)
        else {
            continue;
        };

        if let Some((left_strip, right_strip)) =
            aligned_overlap_strips(left, right, overlap, y_offset)
        {
            let overlay = seam_overlay_from_strips(&left_strip, &right_strip);
            save_debug_tiff(
                debug_dir.join(format!("stitch_overlap_{}_left.tiff", suffix)),
                &left_strip,
            );
            save_debug_tiff(
                debug_dir.join(format!("stitch_overlap_{}_right.tiff", suffix)),
                &right_strip,
            );
            save_debug_tiff(
                debug_dir.join(format!("stitch_overlap_{}_overlay.tiff", suffix)),
                &overlay,
            );
        }
    }
}

fn save_final_seam_overlay(
    left: &Array3<u16>,
    right: &Array3<u16>,
    overlap: usize,
    y_offset: i32,
    _order: StitchOrder,
    debug_dir: &PathBuf,
) {
    if let Some((left_strip, right_strip)) = aligned_overlap_strips(left, right, overlap, y_offset)
    {
        let overlay = seam_overlay_from_strips(&left_strip, &right_strip);
        save_debug_tiff(debug_dir.join("stitch_seam_overlay.tiff"), &overlay);
    }
}

fn stitch_failure_report(
    config: &StitchConfig,
    opencv_available: bool,
    hypotheses: &[StitchHypothesis],
    best_index: usize,
    message: String,
    warnings: Vec<String>,
) -> PhaseReport {
    let mut report = PhaseReport {
        name: "stitch".to_string(),
        success: false,
        confidence: hypotheses
            .get(best_index)
            .map(|hyp| hyp.objective_score)
            .unwrap_or(0.0),
        duration_ms: 0,
        metrics: serde_json::json!({
            "decision": "rejected",
            "requested_transform": config.transform_mode.as_str(),
            "opencv_requested": config.use_opencv,
            "opencv_available": opencv_available,
            "chosen_hypothesis": hypotheses.get(best_index).map(|hyp| hyp.ordering.clone()),
            "transform_model_used": hypotheses.get(best_index).map(|hyp| hyp.transform_model.clone()),
            "rejection_reason": message.clone(),
            "acceptance_thresholds": acceptance_thresholds_json(config),
            "hypotheses": hypotheses,
        }),
        warnings,
        errors: vec![message],
    };
    if !opencv_available && config.use_opencv {
        report.warnings.push(
            "OpenCV was requested at runtime, but this binary was built without the `use-opencv` feature."
                .to_string(),
        );
    }
    report
}

fn apply_target_crop(img: &Array3<u16>, config: &StitchConfig) -> (Array3<u16>, Option<CropInfo>) {
    let (h, w, _) = img.dim();
    let target_w = config.target_width.unwrap_or(w).min(w);
    let target_h = config.target_height.unwrap_or(h).min(h);
    if target_w == w && target_h == h {
        return (img.clone(), None);
    }

    let y_start = choose_vertical_crop(img, target_h);
    let x_start = choose_horizontal_crop(img, target_w);
    let cropped = img
        .slice(s![
            y_start..(y_start + target_h),
            x_start..(x_start + target_w),
            ..
        ])
        .to_owned();

    (
        cropped,
        Some(CropInfo {
            x_start,
            y_start,
            width: target_w,
            height: target_h,
        }),
    )
}

fn choose_vertical_crop(img: &Array3<u16>, target_h: usize) -> usize {
    let (h, w, _) = img.dim();
    if target_h >= h {
        return 0;
    }

    let mut top = 0usize;
    while top < h {
        let row_nonzero =
            (0..w).any(|x| img[[top, x, 0]] != 0 || img[[top, x, 1]] != 0 || img[[top, x, 2]] != 0);
        if row_nonzero {
            break;
        }
        top += 1;
    }

    let mut bottom = h.saturating_sub(1);
    while bottom > top {
        let row_nonzero = (0..w).any(|x| {
            img[[bottom, x, 0]] != 0 || img[[bottom, x, 1]] != 0 || img[[bottom, x, 2]] != 0
        });
        if row_nonzero {
            break;
        }
        bottom = bottom.saturating_sub(1);
    }

    let center = ((top + bottom) / 2).min(h.saturating_sub(1));
    center.saturating_sub(target_h / 2).min(h - target_h)
}

fn choose_horizontal_crop(img: &Array3<u16>, target_w: usize) -> usize {
    let (h, w, _) = img.dim();
    if target_w >= w {
        return 0;
    }

    let scan = base_detect::detect_vertical_base_regions(img);
    let mut base_mask = scan.column_mask;
    if base_mask.len() != w {
        base_mask = vec![0.0; w];
    }

    let mut fill = vec![0.0f64; w];
    let mut content_weight = vec![0.0f64; w];
    for x in 0..w {
        let mut filled = 0usize;
        let mut grad = 0.0f64;
        for y in 0..h {
            let nonzero = img[[y, x, 0]] != 0 || img[[y, x, 1]] != 0 || img[[y, x, 2]] != 0;
            if nonzero {
                filled += 1;
            }
            if x > 0 {
                let curr = img[[y, x, 0]] as f64 * 0.2126
                    + img[[y, x, 1]] as f64 * 0.7152
                    + img[[y, x, 2]] as f64 * 0.0722;
                let prev = img[[y, x - 1, 0]] as f64 * 0.2126
                    + img[[y, x - 1, 1]] as f64 * 0.7152
                    + img[[y, x - 1, 2]] as f64 * 0.0722;
                grad += (curr - prev).abs();
            }
        }
        fill[x] = filled as f64 / h.max(1) as f64;
        base_mask[x] *= fill[x];
        content_weight[x] = (1.0 - base_mask[x]).max(0.0) * (grad / h.max(1) as f64 + 1.0);
    }

    let total_weight: f64 = content_weight.iter().sum();
    let content_center = if total_weight > 1e-6 {
        content_weight
            .iter()
            .enumerate()
            .map(|(x, wgt)| x as f64 * *wgt)
            .sum::<f64>()
            / total_weight
    } else {
        w as f64 / 2.0
    };

    let base_prefix = prefix_sum(&base_mask);
    let fill_prefix = prefix_sum(&fill);
    let edge_band = (target_w / 18).max(16).min(target_w / 2);

    let mut best_start = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for start in 0..=(w - target_w) {
        let end = start + target_w;
        let left_base = range_average(&base_prefix, start, (start + edge_band).min(end));
        let right_base = range_average(&base_prefix, end.saturating_sub(edge_band), end);
        let fill_score = range_average(&fill_prefix, start, end);
        let center_penalty = (((start + target_w / 2) as f64 - content_center).abs()
            / target_w.max(1) as f64)
            .min(1.0);
        let score =
            left_base * 0.45 + right_base * 0.45 + fill_score * 0.20 - center_penalty * 0.25;
        if score > best_score {
            best_score = score;
            best_start = start;
        }
    }

    best_start
}

fn prefix_sum(values: &[f64]) -> Vec<f64> {
    let mut prefix = Vec::with_capacity(values.len() + 1);
    prefix.push(0.0);
    let mut sum = 0.0f64;
    for value in values {
        sum += *value;
        prefix.push(sum);
    }
    prefix
}

fn range_average(prefix: &[f64], start: usize, end: usize) -> f64 {
    if end <= start {
        return 0.0;
    }
    (prefix[end] - prefix[start]) / (end - start) as f64
}

/// Compute per-channel exposure gain from two overlap strips.
fn compute_exposure_gain(strip_l: &Array3<u16>, strip_r: &Array3<u16>) -> [f64; 3] {
    let (h_l, w_l, _) = strip_l.dim();
    let (h_r, w_r, _) = strip_r.dim();
    let h = h_l.min(h_r);
    let w = w_l.min(w_r);
    let mut sum_l = [0.0f64; 3];
    let mut sum_r = [0.0f64; 3];
    let n = (h * w) as f64;
    if n < 1.0 {
        return [1.0; 3];
    }
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                sum_l[c] += strip_l[[y, x, c]] as f64;
                sum_r[c] += strip_r[[y, x, c]] as f64;
            }
        }
    }
    let mut g = [1.0f64; 3];
    for c in 0..3 {
        let mean_l = sum_l[c] / n;
        let mean_r = sum_r[c] / n;
        g[c] = if mean_r < 1.0 {
            1.0
        } else {
            (mean_l / mean_r).clamp(0.5, 2.0)
        };
    }
    g
}

/// Apply per-channel gain to an image, returning a new owned copy.
fn apply_gain(img: &Array3<u16>, gain: &[f64; 3]) -> Array3<u16> {
    let mut out = img.clone();
    let (h, w, _) = out.dim();
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let val = out[[y, x, c]] as f64 * gain[c];
                out[[y, x, c]] = val.clamp(0.0, 65535.0) as u16;
            }
        }
    }
    out
}

/// Invert a 3x3 matrix. Returns `None` if the determinant is near zero.
fn invert_3x3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

/// Multiply two 3x3 matrices: result = a * b.
fn mat_mul_3x3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Apply a 3x3 homography to a 2D point, returning the projected (x, y).
fn transform_point(m: [[f64; 3]; 3], x: f64, y: f64) -> (f64, f64) {
    let w = m[2][0] * x + m[2][1] * y + m[2][2];
    if w.abs() < 1e-12 {
        return (f64::MAX, f64::MAX);
    }
    let px = (m[0][0] * x + m[0][1] * y + m[0][2]) / w;
    let py = (m[1][0] * x + m[1][1] * y + m[1][2]) / w;
    (px, py)
}

/// NCC-only translation stitch.
fn stitch_ncc_fallback(
    left: &Array3<u16>,
    right: &Array3<u16>,
    x_offset: usize,
    y_offset: i32,
    overlap: usize,
    order: StitchOrder,
    config: &StitchConfig,
    hypotheses: &[StitchHypothesis],
    best_index: usize,
    warnings: Vec<String>,
) -> StitchResult {
    let (h_l, w_l, _) = left.dim();
    let (h_r, w_r, _) = right.dim();

    let abs_dy = y_offset.unsigned_abs() as usize;
    let l_y0 = if y_offset >= 0 { 0 } else { abs_dy };
    let r_y0 = if y_offset >= 0 { abs_dy } else { 0 };
    let canvas_h = (l_y0 + h_l).max(r_y0 + h_r);
    let total_w = x_offset + w_r;

    let mut stitched = Array3::<u16>::zeros((canvas_h, total_w, 3));
    stitched
        .slice_mut(s![l_y0..(l_y0 + h_l), 0..x_offset, ..])
        .assign(&left.slice(s![0..h_l, 0..x_offset, ..]));

    let blend_w = config.blend_width.min(overlap);
    let blend_top = l_y0.max(r_y0);
    let blend_bot = (l_y0 + h_l).min(r_y0 + h_r);

    let mut sum_left = [0.0f64; 3];
    let mut sum_right = [0.0f64; 3];
    let mut pixel_count = 0u64;
    if blend_bot > blend_top {
        for canvas_y in blend_top..blend_bot {
            let ly = canvas_y - l_y0;
            let ry = canvas_y - r_y0;
            for x in 0..overlap {
                let abs_x = x_offset + x;
                for c in 0..3 {
                    sum_left[c] += left[[ly, abs_x, c]] as f64;
                    sum_right[c] += right[[ry, x, c]] as f64;
                }
                pixel_count += 1;
            }
        }
    }

    let gain = if pixel_count > 0 {
        let mut g = [1.0f64; 3];
        for c in 0..3 {
            let mean_l = sum_left[c] / pixel_count as f64;
            let mean_r = sum_right[c] / pixel_count as f64;
            g[c] = if mean_r < 1.0 {
                1.0
            } else {
                (mean_l / mean_r).clamp(0.5, 2.0)
            };
        }
        g
    } else {
        [1.0; 3]
    };

    let right_compensated = apply_gain(right, &gain);

    if blend_bot > blend_top {
        for canvas_y in blend_top..blend_bot {
            let ly = canvas_y - l_y0;
            let ry = canvas_y - r_y0;
            for x in 0..overlap {
                let abs_x = x_offset + x;
                let alpha = if blend_w == 0 {
                    1.0
                } else {
                    (x as f64 / blend_w as f64).min(1.0)
                };
                for c in 0..3 {
                    let l_val = left[[ly, abs_x, c]] as f64;
                    let r_val = right_compensated[[ry, x, c]] as f64;
                    stitched[[canvas_y, abs_x, c]] = (l_val * (1.0 - alpha) + r_val * alpha)
                        .round()
                        .clamp(0.0, u16::MAX as f64)
                        as u16;
                }
            }
        }

        for canvas_y in l_y0..(l_y0 + h_l) {
            if canvas_y >= blend_top && canvas_y < blend_bot {
                continue;
            }
            let ly = canvas_y - l_y0;
            for x in 0..overlap {
                let abs_x = x_offset + x;
                for c in 0..3 {
                    stitched[[canvas_y, abs_x, c]] = left[[ly, abs_x, c]];
                }
            }
        }

        for canvas_y in r_y0..(r_y0 + h_r) {
            if canvas_y >= blend_top && canvas_y < blend_bot {
                continue;
            }
            let ry = canvas_y - r_y0;
            for x in 0..overlap {
                let abs_x = x_offset + x;
                for c in 0..3 {
                    stitched[[canvas_y, abs_x, c]] = right_compensated[[ry, x, c]];
                }
            }
        }
    }

    if w_r > overlap {
        stitched
            .slice_mut(s![r_y0..(r_y0 + h_r), w_l..total_w, ..])
            .assign(&right_compensated.slice(s![0..h_r, overlap..w_r, ..]));
    }

    let (stitched, crop_info) = apply_target_crop(&stitched, config);
    if let Some(debug_dir) = &config.debug_dir {
        save_final_seam_overlay(left, right, overlap, y_offset, order, debug_dir);
    }
    let order_gap = if hypotheses.len() >= 2 {
        (hypotheses[0].objective_score - hypotheses[1].objective_score).abs()
    } else {
        0.0
    };

    let mut report = PhaseReport::ok(
        "stitch",
        hypotheses[best_index].objective_score,
        serde_json::json!({
            "decision": "accepted",
            "method": "ncc_translation",
            "requested_transform": config.transform_mode.as_str(),
            "transform_model_used": hypotheses[best_index].transform_model,
            "opencv_requested": config.use_opencv,
            "opencv_available": opencv_match::is_available(),
            "chosen_hypothesis": hypotheses[best_index].ordering,
            "acceptance_thresholds": acceptance_thresholds_json(config),
            "hypotheses": hypotheses,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "overlap": overlap,
            "objective_score": hypotheses[best_index].objective_score,
            "ncc_score": hypotheses[best_index].validation.global_ncc_score,
            "order_gap": order_gap,
            "total_width": total_w,
            "canvas_height": canvas_h,
            "order": order.label(),
            "exposure_gain": gain,
            "crop": crop_metrics_json(crop_info),
        }),
    );
    report.warnings.extend(warnings);

    StitchResult {
        result: Some(stitched),
        x_offset: x_offset as i32,
        y_offset,
        ncc_score: hypotheses[best_index].validation.global_ncc_score,
        order,
        report,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        default_validation_metrics, objective_score, overlap_plausibility_reason,
        overlap_support_score, translation_evidence_score, translation_plausibility_score,
        translation_prior_weight, translation_search_correspondence_score,
        translation_search_score, vertical_offset_plausibility_score, StitchValidationMetrics,
    };

    fn sample_validation_metrics(
        global_ncc_score: f64,
        signature_score: f64,
        overlap_support_score: f64,
        vertical_offset_plausibility_score: f64,
        local_consistency_score: f64,
    ) -> StitchValidationMetrics {
        let mut metrics = default_validation_metrics();
        metrics.global_ncc_score = global_ncc_score;
        metrics.signature_score = signature_score;
        metrics.overlap_support_score = overlap_support_score;
        metrics.vertical_offset_plausibility_score = vertical_offset_plausibility_score;
        metrics.plausibility_score = translation_plausibility_score(
            overlap_support_score,
            vertical_offset_plausibility_score,
        );
        metrics.local_consistency_score = local_consistency_score;
        metrics.evidence_score =
            translation_evidence_score(global_ncc_score, signature_score, local_consistency_score);
        metrics.prior_weight =
            translation_prior_weight(overlap_support_score, vertical_offset_plausibility_score);
        metrics.local_windows_evaluated = 36;
        metrics.plausibility_ok = true;
        metrics.plausibility_reason = None;
        metrics
    }

    #[test]
    fn tiny_overlap_reason_mentions_edge_only_false_positive() {
        let reason = overlap_plausibility_reason(77, 0.0129, 0.0129)
            .expect("tiny overlap should be implausible");
        assert!(reason.contains("edge-only false positive"));
        assert!(reason.contains("77px"));
    }

    #[test]
    fn overlap_support_score_penalizes_tiny_overlaps() {
        assert!(overlap_support_score(0.013) < 0.25);
        assert!(overlap_support_score(0.12) > 0.9);
    }

    #[test]
    fn vertical_offset_plausibility_penalizes_search_limit() {
        assert!(vertical_offset_plausibility_score(2, 15) > 0.95);
        assert!(vertical_offset_plausibility_score(15, 15) < 0.20);
    }

    #[test]
    fn prior_weight_penalizes_vertical_limit_even_for_large_overlap() {
        let expanded_prior =
            translation_prior_weight(1.0, vertical_offset_plausibility_score(15, 15));
        let compact_prior = translation_prior_weight(
            0.4440053690809217,
            vertical_offset_plausibility_score(2, 15),
        );

        assert!(
            compact_prior > expanded_prior,
            "a candidate on the vertical search limit should not retain the strongest prior just because its overlap is larger"
        );
    }

    #[test]
    fn search_score_prefers_stronger_correspondence_over_saturated_overlap_prior() {
        let expanded_candidate = translation_search_score(
            translation_search_correspondence_score(0.21954560297343648, 0.5261482372999441),
            1.0,
            vertical_offset_plausibility_score(-15, 15),
        );
        let smaller_candidate = translation_search_score(
            translation_search_correspondence_score(0.46234207388873, 0.48565055075364444),
            0.4440053690809217,
            vertical_offset_plausibility_score(2, 15),
        );

        assert!(
            smaller_candidate > expanded_candidate,
            "smaller candidate should win on search score when its correspondence evidence is much stronger"
        );
    }

    #[test]
    fn objective_score_prefers_stronger_correspondence_over_larger_overlap_prior() {
        let expanded_candidate =
            sample_validation_metrics(0.21954560297343648, 0.5261482372999441, 1.0, 0.15, 0.092);
        let smaller_candidate = sample_validation_metrics(
            0.46234207388873,
            0.48565055075364444,
            0.4440053690809217,
            1.0,
            0.199,
        );

        assert!(
            expanded_candidate.plausibility_score > smaller_candidate.plausibility_score,
            "the larger overlap should still retain the stronger plausibility prior"
        );
        assert!(
            objective_score(&smaller_candidate) > objective_score(&expanded_candidate),
            "stronger correspondence evidence should outweigh a saturated overlap prior"
        );
    }
}

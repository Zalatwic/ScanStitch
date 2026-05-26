use ndarray::Array3;
use std::collections::BTreeMap;

const BASE_FALLBACK_HISTOGRAM_BINS: usize = 4096;
const BASE_FALLBACK_PERCENTILE: f64 = 0.995;
const BASE_FALLBACK_JOINT_COLOR_PERCENTILE: f64 = 0.75;
const BASE_FALLBACK_SUPPORT_LOW: f64 = 0.0005;
const BASE_FALLBACK_SUPPORT_HIGH: f64 = 0.0100;
const BASE_FALLBACK_MAX_CONFIDENCE: f64 = 0.12;
const BASE_FALLBACK_SUPPORT_TOLERANCE: f64 = 0.98;
const MIN_BASE_REGION_LUMINANCE: f64 = 512.0;
const MIN_BASE_REGION_REFERENCE_RATIO: f64 = 0.20;
const DARK_REGION_HIGH_TRANSMITTANCE_LUMA_RATIO: f64 = 0.72;
const DARK_REGION_HIGH_TRANSMITTANCE_CHANNEL_RATIO: f64 = 0.70;
const ROLL_BASE_CLUSTER_CHROMA_THRESHOLD: f64 = 0.055;
const ROLL_BASE_CLUSTER_LUMA_THRESHOLD: f64 = 0.30;
const ROLL_BASE_MIN_SUPPORT_RATIO: f64 = 0.30;
const ROLL_BASE_RUNNER_UP_MARGIN: usize = 2;

/// Result of film base (rebate) detection on left/right edges.
#[derive(Debug, Clone)]
pub struct BaseDetection {
    /// Confidence that the left edge contains film base [0, 1].
    pub left_confidence: f64,
    /// Confidence that the right edge contains film base [0, 1].
    pub right_confidence: f64,
    /// Confidence that the top edge contains film base [0, 1].
    pub top_confidence: f64,
    /// Confidence that the bottom edge contains film base [0, 1].
    pub bottom_confidence: f64,
    /// Estimated base RGB color as f64 values (in u16 range).
    pub base_color: [f64; 3],
    /// Source of the selected base RGB estimate.
    pub base_color_source: &'static str,
    /// Human-readable reason for the selected base RGB estimate.
    pub base_color_reason: String,
    /// Low-confidence proxy confidence for percentile-derived fallback estimates.
    pub base_color_proxy_confidence: Option<f64>,
    /// Fraction of pixels simultaneously supporting the percentile-derived fallback estimate.
    pub base_color_support_fraction: Option<f64>,
    /// Edge strip width used for left/right detection.
    pub strip_width: usize,
    /// Block confidence grid for the left strip (rows x cols of f64 in [0,1]).
    pub left_base_mask: Vec<Vec<f64>>,
    /// Block confidence grid for the right strip (rows x cols of f64 in [0,1]).
    pub right_base_mask: Vec<Vec<f64>>,
}

/// Post-stitch base estimate reconciliation against stable component priors.
#[derive(Debug, Clone)]
pub struct BaseReconciliation {
    pub base_color: [f64; 3],
    pub confidence: f64,
    pub source: &'static str,
    pub reason: String,
    pub raw_working_base_color: [f64; 3],
    pub raw_working_confidence: f64,
    pub component_consensus_base_color: Option<[f64; 3]>,
    pub component_consensus_confidence: Option<f64>,
    pub component_consensus_relative_spread: Option<[f64; 3]>,
    pub working_vs_consensus_relative_delta: Option<[f64; 3]>,
    pub edge_balance_ratio: f64,
}

/// A vertically contiguous base-like region found across image width.
#[derive(Debug, Clone)]
pub struct BaseRegion {
    pub x_start: usize,
    pub x_end: usize,
    pub confidence: f64,
    pub median_rgb: [f64; 3],
    pub mad_lum: f64,
}

/// Full-width scan for base-like vertical regions.
#[derive(Debug, Clone)]
pub struct VerticalBaseScan {
    pub block_width: usize,
    pub regions: Vec<BaseRegion>,
    pub column_mask: Vec<f64>,
    pub dominant_color: [f64; 3],
}

/// Per-block statistics.
#[derive(Debug, Clone)]
struct BlockStats {
    median_rgb: [f64; 3],
    mad_lum: f64,
}

#[derive(Debug, Clone, Copy)]
struct HighTransmittanceFallback {
    color: [f64; 3],
    confidence: f64,
    support_fraction: f64,
    joint_support_fraction: Option<f64>,
    color_strategy: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct JointHighTransmittanceColor {
    color: [f64; 3],
    support_fraction: f64,
}

#[derive(Debug, Clone)]
pub struct RollBaseCandidate {
    pub frame_id: String,
    pub color: [f64; 3],
    pub source: String,
    pub confidence: f64,
    pub support_fraction: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct RollBaseCluster {
    pub color: [f64; 3],
    pub frame_count: usize,
    pub mean_luminance: f64,
    pub max_relative_luminance_spread: f64,
    pub source_counts: Vec<(String, usize)>,
}

#[derive(Debug, Clone)]
pub struct RollBaseConsensus {
    pub color: [f64; 3],
    pub source: &'static str,
    pub confidence: f64,
    pub frame_count: usize,
    pub candidate_count: usize,
    pub selected_cluster_frame_count: usize,
    pub rejected_dark_candidate_count: usize,
    pub high_transmittance_envelope: [f64; 3],
    pub clusters: Vec<RollBaseCluster>,
    pub reason: String,
}

impl RollBaseCandidate {
    pub fn from_detection(frame_id: impl Into<String>, detection: &BaseDetection) -> Self {
        Self {
            frame_id: frame_id.into(),
            color: detection.base_color,
            source: detection.base_color_source.to_string(),
            confidence: base_detection_confidence(detection),
            support_fraction: detection.base_color_support_fraction,
        }
    }
}

/// Detect unexposed film base (rebate) regions on left/right edges.
///
/// Operates on a cropped u16 image (h, w, 3). Returns confidence scores
/// for each side plus the estimated base color.
pub fn detect_film_base(img: &Array3<u16>) -> BaseDetection {
    let (h, w, _) = (img.shape()[0], img.shape()[1], img.shape()[2]);

    // Strip width: 5% of image width, min 10px, max half the image
    let strip_w = (w as f64 * 0.05).round().max(10.0).min((w / 2) as f64) as usize;

    // Interior region: everything not in the strips
    let interior_start = strip_w;
    let interior_end = w.saturating_sub(strip_w).max(interior_start + 1);
    let interior_median = region_median_color(img, h, interior_start, interior_end);

    let (mut left_conf, left_mask, mut left_color) =
        process_strip(img, h, 0, strip_w, &interior_median);
    let (mut right_conf, right_mask, mut right_color) =
        process_strip(img, h, w.saturating_sub(strip_w), w, &interior_median);
    let strip_h = (h as f64 * 0.05).round().max(10.0).min((h / 2) as f64) as usize;
    let row_interior_start = strip_h;
    let row_interior_end = h.saturating_sub(strip_h).max(row_interior_start + 1);
    let row_interior_median =
        region_median_color_rows(img, w, row_interior_start, row_interior_end);
    let (mut top_conf, top_color) = process_row_strip(img, w, 0, strip_h, &row_interior_median);
    let (mut bottom_conf, bottom_color) =
        process_row_strip(img, w, h.saturating_sub(strip_h), h, &row_interior_median);
    let high_transmittance_envelope = robust_high_transmittance_estimate(img);

    // Cross-check against the full-width vertical scan so split-frame gaps and
    // edge rebates share a consistent base estimate.
    let vertical = detect_vertical_base_regions(img);
    if let Some(region) = vertical.regions.first() {
        if region.x_start <= vertical.block_width {
            left_conf = left_conf.max(region.confidence);
            left_color = region.median_rgb;
        }
    }
    if let Some(region) = vertical.regions.last() {
        if region.x_end + vertical.block_width >= w {
            right_conf = right_conf.max(region.confidence);
            right_color = region.median_rgb;
        }
    }
    left_conf = demote_dark_base_region_against_envelope(
        left_conf,
        &left_color,
        &high_transmittance_envelope.color,
    );
    right_conf = demote_dark_base_region_against_envelope(
        right_conf,
        &right_color,
        &high_transmittance_envelope.color,
    );
    top_conf = demote_dark_base_region_against_envelope(
        top_conf,
        &top_color,
        &high_transmittance_envelope.color,
    );
    bottom_conf = demote_dark_base_region_against_envelope(
        bottom_conf,
        &bottom_color,
        &high_transmittance_envelope.color,
    );

    let (
        base_color,
        base_color_source,
        base_color_reason,
        base_color_proxy_confidence,
        base_color_support_fraction,
    ) = if left_conf > 0.3 || right_conf > 0.3 {
        if left_conf >= right_conf {
            (
                left_color,
                "working_edges",
                format!("left edge rebate selected with confidence {left_conf:.3}"),
                None,
                None,
            )
        } else {
            (
                right_color,
                "working_edges",
                format!("right edge rebate selected with confidence {right_conf:.3}"),
                None,
                None,
            )
        }
    } else if top_conf > 0.3 || bottom_conf > 0.3 {
        if top_conf >= bottom_conf {
            (
                top_color,
                "horizontal_base_region",
                format!("top horizontal rebate selected with confidence {top_conf:.3}"),
                None,
                None,
            )
        } else {
            (
                bottom_color,
                "horizontal_base_region",
                format!("bottom horizontal rebate selected with confidence {bottom_conf:.3}"),
                None,
                None,
            )
        }
    } else if vertical.dominant_color.iter().any(|value| *value > 0.0) {
        (
            vertical.dominant_color,
            "vertical_base_region",
            "full-width scan found a base-like vertical region after edge confidence was low"
                .to_string(),
            None,
            None,
        )
    } else {
        let fallback = high_transmittance_envelope;
        left_conf = left_conf.max(fallback.confidence);
        right_conf = right_conf.max(fallback.confidence);
        top_conf = top_conf.max(fallback.confidence);
        bottom_conf = bottom_conf.max(fallback.confidence);
        let joint_detail = fallback
            .joint_support_fraction
            .map(|fraction| {
                format!(
                    "; selected a joint high-transmittance pixel colour from {:.3}% of pixels because independent channel percentiles did not co-occur",
                    fraction * 100.0
                )
            })
            .unwrap_or_default();
        (
            fallback.color,
            "high_transmittance_fallback",
            format!(
                "no edge, vertical, or horizontal base region was detected; using the {} {:.1}% high-transmittance fallback with low proxy confidence {:.3} from {:.3}% simultaneous high-transmittance support{}",
                fallback.color_strategy,
                BASE_FALLBACK_PERCENTILE * 100.0,
                fallback.confidence,
                fallback.support_fraction * 100.0,
                joint_detail
            ),
            Some(fallback.confidence),
            Some(fallback.support_fraction),
        )
    };

    BaseDetection {
        left_confidence: left_conf,
        right_confidence: right_conf,
        top_confidence: top_conf,
        bottom_confidence: bottom_conf,
        base_color,
        base_color_source,
        base_color_reason,
        base_color_proxy_confidence,
        base_color_support_fraction,
        strip_width: strip_w,
        left_base_mask: left_mask,
        right_base_mask: right_mask,
    }
}

pub fn base_detection_confidence(detection: &BaseDetection) -> f64 {
    detection
        .left_confidence
        .max(detection.right_confidence)
        .max(detection.top_confidence)
        .max(detection.bottom_confidence)
}

fn demote_dark_base_region_against_envelope(
    confidence: f64,
    candidate: &[f64; 3],
    envelope: &[f64; 3],
) -> f64 {
    if confidence <= 0.0 || !base_candidate_is_much_darker_than_envelope(candidate, envelope) {
        confidence
    } else {
        0.0
    }
}

fn base_candidate_is_much_darker_than_envelope(candidate: &[f64; 3], envelope: &[f64; 3]) -> bool {
    if !candidate
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
        || !envelope
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    {
        return false;
    }

    let envelope_lum = luminance(envelope);
    if envelope_lum <= MIN_BASE_REGION_LUMINANCE {
        return false;
    }
    let candidate_lum = luminance(candidate);
    let dark_luminance = candidate_lum < envelope_lum * DARK_REGION_HIGH_TRANSMITTANCE_LUMA_RATIO;
    let dark_channels = (0..3)
        .filter(|&channel| {
            candidate[channel] < envelope[channel] * DARK_REGION_HIGH_TRANSMITTANCE_CHANNEL_RATIO
        })
        .count();
    dark_luminance && dark_channels >= 2
}

fn robust_high_transmittance_estimate(img: &Array3<u16>) -> HighTransmittanceFallback {
    let (h, w, channels) = img.dim();
    if h == 0 || w == 0 || channels < 3 {
        return HighTransmittanceFallback {
            color: [0.0; 3],
            confidence: 0.0,
            support_fraction: 0.0,
            joint_support_fraction: None,
            color_strategy: "channel-wise",
        };
    }

    let mut histograms: [Vec<u64>; 3] =
        std::array::from_fn(|_| vec![0; BASE_FALLBACK_HISTOGRAM_BINS]);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let bin = ((img[[y, x, c]] as usize * (BASE_FALLBACK_HISTOGRAM_BINS - 1))
                    / u16::MAX as usize)
                    .min(BASE_FALLBACK_HISTOGRAM_BINS - 1);
                histograms[c][bin] += 1;
            }
        }
    }

    let total = (h * w) as u64;
    let target = ((total.saturating_sub(1)) as f64 * BASE_FALLBACK_PERCENTILE) as u64;
    let color: [f64; 3] = std::array::from_fn(|c| {
        let mut cumulative = 0u64;
        for (bin, count) in histograms[c].iter().enumerate() {
            cumulative += *count;
            if cumulative > target {
                return (bin as f64 / (BASE_FALLBACK_HISTOGRAM_BINS - 1) as f64) * u16::MAX as f64;
            }
        }
        u16::MAX as f64
    });

    let support_threshold: [f64; 3] =
        std::array::from_fn(|c| (color[c] * BASE_FALLBACK_SUPPORT_TOLERANCE).max(1.0));
    let mut support = 0u64;
    for y in 0..h {
        for x in 0..w {
            if (0..3).all(|c| img[[y, x, c]] as f64 >= support_threshold[c]) {
                support += 1;
            }
        }
    }
    let support_fraction = support as f64 / total.max(1) as f64;
    let support_score = ((support_fraction - BASE_FALLBACK_SUPPORT_LOW)
        / (BASE_FALLBACK_SUPPORT_HIGH - BASE_FALLBACK_SUPPORT_LOW))
        .clamp(0.0, 1.0);

    let (color, color_strategy, joint_support_fraction) =
        if support_fraction < BASE_FALLBACK_SUPPORT_LOW {
            let joint = joint_high_transmittance_color(img, &color);
            if joint.support_fraction > 0.0 {
                (joint.color, "joint-pixel", Some(joint.support_fraction))
            } else {
                (color, "channel-wise", None)
            }
        } else {
            (color, "channel-wise", None)
        };

    HighTransmittanceFallback {
        color,
        confidence: support_score * BASE_FALLBACK_MAX_CONFIDENCE,
        support_fraction,
        joint_support_fraction,
        color_strategy,
    }
}

fn joint_high_transmittance_color(
    img: &Array3<u16>,
    independent_color: &[f64; 3],
) -> JointHighTransmittanceColor {
    let (h, w, channels) = img.dim();
    if h == 0 || w == 0 || channels < 3 {
        return JointHighTransmittanceColor {
            color: [0.0; 3],
            support_fraction: 0.0,
        };
    }

    let mut score_histogram = vec![0u64; BASE_FALLBACK_HISTOGRAM_BINS];
    for y in 0..h {
        for x in 0..w {
            let score = joint_support_score(img, y, x, independent_color);
            let bin = (score * (BASE_FALLBACK_HISTOGRAM_BINS - 1) as f64)
                .round()
                .clamp(0.0, (BASE_FALLBACK_HISTOGRAM_BINS - 1) as f64)
                as usize;
            score_histogram[bin] += 1;
        }
    }

    let total = (h * w) as u64;
    let target = ((total.saturating_sub(1)) as f64 * BASE_FALLBACK_PERCENTILE) as u64;
    let mut cumulative = 0u64;
    let mut threshold_bin = BASE_FALLBACK_HISTOGRAM_BINS - 1;
    for (bin, count) in score_histogram.iter().enumerate() {
        cumulative += *count;
        if cumulative > target {
            threshold_bin = bin;
            break;
        }
    }
    let threshold_score = threshold_bin as f64 / (BASE_FALLBACK_HISTOGRAM_BINS - 1) as f64;

    let mut selected_histograms: [Vec<u64>; 3] =
        std::array::from_fn(|_| vec![0; BASE_FALLBACK_HISTOGRAM_BINS]);
    let mut selected = 0u64;
    for y in 0..h {
        for x in 0..w {
            if joint_support_score(img, y, x, independent_color) < threshold_score {
                continue;
            }
            selected += 1;
            for c in 0..3 {
                let bin = ((img[[y, x, c]] as usize * (BASE_FALLBACK_HISTOGRAM_BINS - 1))
                    / u16::MAX as usize)
                    .min(BASE_FALLBACK_HISTOGRAM_BINS - 1);
                selected_histograms[c][bin] += 1;
            }
        }
    }

    if selected == 0 {
        return JointHighTransmittanceColor {
            color: [0.0; 3],
            support_fraction: 0.0,
        };
    }

    let selected_target =
        ((selected.saturating_sub(1)) as f64 * BASE_FALLBACK_JOINT_COLOR_PERCENTILE) as u64;
    let color = std::array::from_fn(|c| {
        let mut cumulative = 0u64;
        for (bin, count) in selected_histograms[c].iter().enumerate() {
            cumulative += *count;
            if cumulative > selected_target {
                return (bin as f64 / (BASE_FALLBACK_HISTOGRAM_BINS - 1) as f64) * u16::MAX as f64;
            }
        }
        u16::MAX as f64
    });

    JointHighTransmittanceColor {
        color,
        support_fraction: selected as f64 / total.max(1) as f64,
    }
}

fn joint_support_score(img: &Array3<u16>, y: usize, x: usize, independent_color: &[f64; 3]) -> f64 {
    (0..3)
        .map(|c| img[[y, x, c]] as f64 / independent_color[c].max(1.0))
        .fold(f64::INFINITY, f64::min)
        .clamp(0.0, 1.0)
}

pub fn roll_consensus_base(candidates: &[RollBaseCandidate]) -> Option<RollBaseConsensus> {
    let valid = candidates
        .iter()
        .filter(|candidate| {
            candidate
                .color
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        })
        .cloned()
        .collect::<Vec<_>>();
    if valid.len() < min_roll_base_consensus_frames(valid.len()) {
        return None;
    }

    let high_transmittance_envelope = roll_high_transmittance_envelope(&valid);
    let mut rejected_dark_candidate_count = 0usize;
    let mut retained = Vec::<RollBaseCandidate>::new();
    for candidate in valid {
        if base_candidate_is_much_darker_than_envelope(
            &candidate.color,
            &high_transmittance_envelope,
        ) {
            rejected_dark_candidate_count += 1;
        } else {
            retained.push(candidate);
        }
    }

    if retained.len() < min_roll_base_consensus_frames(candidates.len()) {
        return None;
    }

    retained.sort_by(|a, b| {
        luminance(&b.color)
            .partial_cmp(&luminance(&a.color))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut clusters = Vec::<RollClusterWork>::new();
    for candidate in retained {
        let best_idx = clusters
            .iter()
            .enumerate()
            .filter(|(_, cluster)| roll_candidate_matches_cluster(&candidate.color, cluster))
            .min_by(|(_, left), (_, right)| {
                roll_cluster_distance(&candidate.color, left)
                    .partial_cmp(&roll_cluster_distance(&candidate.color, right))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx);

        if let Some(idx) = best_idx {
            clusters[idx].push(candidate);
        } else {
            clusters.push(RollClusterWork::new(candidate));
        }
    }

    if clusters.is_empty() {
        return None;
    }

    clusters.sort_by(compare_roll_clusters);
    let min_support = min_roll_base_consensus_frames(candidates.len());
    let selected = &clusters[0];
    let selected_count = selected.candidates.len();
    let support_ratio = selected_count as f64 / candidates.len().max(1) as f64;
    if selected_count < min_support || support_ratio < ROLL_BASE_MIN_SUPPORT_RATIO {
        return None;
    }

    if let Some(runner_up) = clusters.get(1) {
        let runner_up_count = runner_up.candidates.len();
        let selected_is_brighter =
            luminance(&selected.color()) >= luminance(&runner_up.color()) * 1.08;
        if selected_count <= runner_up_count
            || (selected_count < runner_up_count + ROLL_BASE_RUNNER_UP_MARGIN
                && !selected_is_brighter)
        {
            return None;
        }
    }

    let cluster_summaries = clusters
        .iter()
        .map(RollClusterWork::to_summary)
        .collect::<Vec<_>>();
    let selected_summary = cluster_summaries
        .first()
        .expect("selected cluster summary should exist");
    let confidence = roll_consensus_confidence(
        selected_count,
        candidates.len(),
        selected_summary.max_relative_luminance_spread,
        selected.has_strong_single_frame_sources(),
    );

    Some(RollBaseConsensus {
        color: selected_summary.color,
        source: "roll_consensus_base",
        confidence,
        frame_count: candidates.len(),
        candidate_count: candidates.len(),
        selected_cluster_frame_count: selected_count,
        rejected_dark_candidate_count,
        high_transmittance_envelope,
        clusters: cluster_summaries,
        reason: format!(
            "selected stable roll base cluster from {selected_count}/{} frame candidates; rejected {} darker candidate(s) against the roll high-transmittance envelope",
            candidates.len(),
            rejected_dark_candidate_count
        ),
    })
}

#[derive(Debug, Clone)]
struct RollClusterWork {
    candidates: Vec<RollBaseCandidate>,
}

impl RollClusterWork {
    fn new(candidate: RollBaseCandidate) -> Self {
        Self {
            candidates: vec![candidate],
        }
    }

    fn push(&mut self, candidate: RollBaseCandidate) {
        self.candidates.push(candidate);
    }

    fn color(&self) -> [f64; 3] {
        median_color(
            &self
                .candidates
                .iter()
                .map(|candidate| candidate.color)
                .collect::<Vec<_>>(),
        )
    }

    fn max_relative_luminance_spread(&self) -> f64 {
        let luminances = self
            .candidates
            .iter()
            .map(|candidate| luminance(&candidate.color))
            .collect::<Vec<_>>();
        let min_lum = luminances.iter().copied().fold(f64::INFINITY, f64::min);
        let max_lum = luminances.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !min_lum.is_finite() || !max_lum.is_finite() || max_lum <= 1.0 {
            1.0
        } else {
            (max_lum - min_lum) / max_lum
        }
    }

    fn has_strong_single_frame_sources(&self) -> bool {
        self.candidates.iter().any(|candidate| {
            candidate.confidence >= 0.50 && candidate.source != "high_transmittance_fallback"
        })
    }

    fn to_summary(&self) -> RollBaseCluster {
        let color = self.color();
        let mut source_counts = BTreeMap::<String, usize>::new();
        for candidate in &self.candidates {
            *source_counts.entry(candidate.source.clone()).or_insert(0) += 1;
        }
        RollBaseCluster {
            color,
            frame_count: self.candidates.len(),
            mean_luminance: self
                .candidates
                .iter()
                .map(|candidate| luminance(&candidate.color))
                .sum::<f64>()
                / self.candidates.len().max(1) as f64,
            max_relative_luminance_spread: self.max_relative_luminance_spread(),
            source_counts: source_counts.into_iter().collect(),
        }
    }
}

fn roll_high_transmittance_envelope(candidates: &[RollBaseCandidate]) -> [f64; 3] {
    std::array::from_fn(|channel| {
        let mut values = candidates
            .iter()
            .map(|candidate| candidate.color[channel])
            .collect::<Vec<_>>();
        percentile(&mut values, 0.85)
    })
}

fn roll_candidate_matches_cluster(color: &[f64; 3], cluster: &RollClusterWork) -> bool {
    let cluster_color = cluster.color();
    chroma_distance(block_chroma(color), block_chroma(&cluster_color))
        <= ROLL_BASE_CLUSTER_CHROMA_THRESHOLD
        && relative_luminance_delta(color, &cluster_color) <= ROLL_BASE_CLUSTER_LUMA_THRESHOLD
}

fn roll_cluster_distance(color: &[f64; 3], cluster: &RollClusterWork) -> f64 {
    let cluster_color = cluster.color();
    chroma_distance(block_chroma(color), block_chroma(&cluster_color))
        + relative_luminance_delta(color, &cluster_color)
}

fn compare_roll_clusters(left: &RollClusterWork, right: &RollClusterWork) -> std::cmp::Ordering {
    let left_count = left.candidates.len();
    let right_count = right.candidates.len();
    right_count
        .cmp(&left_count)
        .then_with(|| {
            luminance(&right.color())
                .partial_cmp(&luminance(&left.color()))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            left.max_relative_luminance_spread()
                .partial_cmp(&right.max_relative_luminance_spread())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn min_roll_base_consensus_frames(candidate_count: usize) -> usize {
    if candidate_count >= 20 {
        5
    } else if candidate_count >= 8 {
        3
    } else {
        2
    }
}

fn roll_consensus_confidence(
    selected_count: usize,
    candidate_count: usize,
    luminance_spread: f64,
    has_strong_single_frame_sources: bool,
) -> f64 {
    let support_ratio = selected_count as f64 / candidate_count.max(1) as f64;
    let consistency = (1.0 - luminance_spread).clamp(0.0, 1.0);
    let source_floor = if has_strong_single_frame_sources {
        0.70
    } else {
        0.55
    };
    (source_floor + support_ratio * 0.30 + consistency * 0.15).clamp(0.35, 0.98)
}

fn median_color(colors: &[[f64; 3]]) -> [f64; 3] {
    std::array::from_fn(|channel| {
        let mut values = colors
            .iter()
            .map(|color| color[channel])
            .collect::<Vec<_>>();
        median(&mut values)
    })
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

fn percentile(values: &mut [f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx =
        ((values.len().saturating_sub(1)) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    values[idx.min(values.len() - 1)]
}

/// Reconcile a stitched working-image base estimate against the component-base
/// consensus when the stitched crop has likely lost the true outer rebate.
pub fn reconcile_stitched_base_estimate(
    working: &BaseDetection,
    component_bases: &[[f64; 3]],
) -> BaseReconciliation {
    let raw_working_confidence = working
        .left_confidence
        .max(working.right_confidence)
        .max(working.top_confidence)
        .max(working.bottom_confidence);
    let stronger_edge = working
        .left_confidence
        .max(working.right_confidence)
        .max(1e-6);
    let edge_balance_ratio = if stronger_edge <= 1e-6 {
        1.0
    } else {
        (working.left_confidence.min(working.right_confidence) / stronger_edge).clamp(0.0, 1.0)
    };

    let component_consensus = component_consensus_base_color(component_bases);
    if let Some((consensus, spread, consensus_confidence)) = component_consensus {
        let relative_delta = relative_rgb_delta(&working.base_color, &consensus);
        let mean_delta = relative_delta.iter().sum::<f64>() / relative_delta.len() as f64;
        let max_delta = relative_delta.iter().copied().fold(0.0f64, f64::max);
        let materially_darker_channels = (0..3)
            .filter(|&c| working.base_color[c] < consensus[c] * 0.92)
            .count();
        let should_override = edge_balance_ratio < 0.75
            && materially_darker_channels >= 2
            && (max_delta >= 0.08 || mean_delta >= 0.06);

        if should_override {
            return BaseReconciliation {
                base_color: consensus,
                confidence: consensus_confidence,
                source: "component_consensus",
                reason: format!(
                    "stitched crop-edge base was unbalanced (edge-balance {:.3}) and materially darker than the component consensus (max delta {:.1}%, mean delta {:.1}%); using the pre-stitch component consensus instead",
                    edge_balance_ratio,
                    max_delta * 100.0,
                    mean_delta * 100.0
                ),
                raw_working_base_color: working.base_color,
                raw_working_confidence,
                component_consensus_base_color: Some(consensus),
                component_consensus_confidence: Some(consensus_confidence),
                component_consensus_relative_spread: Some(spread),
                working_vs_consensus_relative_delta: Some(relative_delta),
                edge_balance_ratio,
            };
        }

        return BaseReconciliation {
            base_color: working.base_color,
            confidence: raw_working_confidence,
            source: working.base_color_source,
            reason: format!(
                "stitched working-image edge estimate stayed close to the component consensus (max delta {:.1}%, mean delta {:.1}%)",
                max_delta * 100.0,
                mean_delta * 100.0
            ),
            raw_working_base_color: working.base_color,
            raw_working_confidence,
            component_consensus_base_color: Some(consensus),
            component_consensus_confidence: Some(consensus_confidence),
            component_consensus_relative_spread: Some(spread),
            working_vs_consensus_relative_delta: Some(relative_delta),
            edge_balance_ratio,
        };
    }

    BaseReconciliation {
        base_color: working.base_color,
        confidence: raw_working_confidence,
        source: working.base_color_source,
        reason: format!(
            "no stable component-base consensus was available; keeping the working-image base estimate: {}",
            working.base_color_reason
        ),
        raw_working_base_color: working.base_color,
        raw_working_confidence,
        component_consensus_base_color: None,
        component_consensus_confidence: None,
        component_consensus_relative_spread: None,
        working_vs_consensus_relative_delta: None,
        edge_balance_ratio,
    }
}

/// Scan the full image width for base-like vertical regions such as edge rebate
/// or internal frame gaps.
pub fn detect_vertical_base_regions(img: &Array3<u16>) -> VerticalBaseScan {
    let (h, w, _) = img.dim();
    if h == 0 || w == 0 {
        return VerticalBaseScan {
            block_width: 0,
            regions: Vec::new(),
            column_mask: Vec::new(),
            dominant_color: [0.0; 3],
        };
    }

    let block_width = (w as f64 * 0.02)
        .round()
        .max(5.0)
        .min((w / 8).max(5) as f64) as usize;

    let mut blocks = Vec::<BlockStats>::new();
    let mut spans = Vec::<(usize, usize)>::new();
    let mut x = 0usize;
    while x < w {
        let x_end = (x + block_width).min(w);
        spans.push((x, x_end));
        blocks.push(compute_block_stats(img, 0, h, x, x_end));
        x = x_end;
    }

    if blocks.len() < 2 {
        return VerticalBaseScan {
            block_width,
            regions: Vec::new(),
            column_mask: vec![0.0; w],
            dominant_color: [0.0; 3],
        };
    }

    let mut mad_vals: Vec<f64> = blocks.iter().map(|b| b.mad_lum).collect();
    let median_block_mad = fast_median(&mut mad_vals);
    let mut lum_vals: Vec<f64> = blocks
        .iter()
        .map(|block| luminance(&block.median_rgb))
        .collect();
    let reference_lum = fast_median(&mut lum_vals);
    let low_texture_threshold = (median_block_mad * 0.60).clamp(80.0, 600.0);
    let grow_texture_threshold = (low_texture_threshold * 1.2).clamp(100.0, 750.0);
    let seed_chroma_threshold = 0.020;
    let grow_chroma_threshold = 0.018;

    let mut region_strength = vec![0.0f64; blocks.len()];
    for i in 0..blocks.len() {
        if blocks[i].mad_lum > low_texture_threshold {
            continue;
        }

        let chroma_i = block_chroma(&blocks[i].median_rgb);
        let mut neighbor_contrast = 0.0f64;
        let mut neighbor_lum_contrast = 0.0f64;

        if i > 0 {
            let chroma_prev = block_chroma(&blocks[i - 1].median_rgb);
            neighbor_contrast = neighbor_contrast.max(chroma_distance(chroma_i, chroma_prev));
            neighbor_lum_contrast = neighbor_lum_contrast.max(relative_luminance_delta(
                &blocks[i].median_rgb,
                &blocks[i - 1].median_rgb,
            ));
        }
        if i + 1 < blocks.len() {
            let chroma_next = block_chroma(&blocks[i + 1].median_rgb);
            neighbor_contrast = neighbor_contrast.max(chroma_distance(chroma_i, chroma_next));
            neighbor_lum_contrast = neighbor_lum_contrast.max(relative_luminance_delta(
                &blocks[i].median_rgb,
                &blocks[i + 1].median_rgb,
            ));
        }

        let seed_conf = ((neighbor_contrast - 0.010) / 0.060).clamp(0.0, 1.0) * 0.7
            + ((neighbor_lum_contrast - 0.02) / 0.25).clamp(0.0, 1.0) * 0.3;

        if seed_conf <= 0.0 || neighbor_contrast < seed_chroma_threshold {
            continue;
        }

        let mut left = i;
        while left > 0 && blocks[left - 1].mad_lum <= grow_texture_threshold {
            let candidate = block_chroma(&blocks[left - 1].median_rgb);
            if chroma_distance(candidate, chroma_i) > grow_chroma_threshold {
                break;
            }
            left -= 1;
        }

        let mut right = i;
        while right + 1 < blocks.len() && blocks[right + 1].mad_lum <= grow_texture_threshold {
            let candidate = block_chroma(&blocks[right + 1].median_rgb);
            if chroma_distance(candidate, chroma_i) > grow_chroma_threshold {
                break;
            }
            right += 1;
        }

        for strength in &mut region_strength[left..=right] {
            *strength = strength.max(seed_conf);
        }
    }

    let mut regions = Vec::<BaseRegion>::new();
    let mut i = 0usize;
    while i < blocks.len() {
        if region_strength[i] <= 0.0 {
            i += 1;
            continue;
        }

        let start_idx = i;
        let mut end_idx = i;
        while end_idx + 1 < blocks.len() && region_strength[end_idx + 1] > 0.0 {
            end_idx += 1;
        }

        let x_start = spans[start_idx].0;
        let x_end = spans[end_idx].1;
        let width_fraction = (x_end - x_start) as f64 / w as f64;
        if !(0.01..=0.85).contains(&width_fraction) {
            i = end_idx + 1;
            continue;
        }

        let members: Vec<(usize, usize)> = (start_idx..=end_idx).map(|idx| (0, idx)).collect();
        let mut region_conf = region_strength[start_idx..=end_idx].iter().sum::<f64>()
            / (end_idx - start_idx + 1) as f64;

        let contrast = region_neighbor_contrast(&blocks, start_idx, end_idx);
        region_conf = (region_conf * 0.7 + contrast * 0.3).clamp(0.0, 1.0);
        if region_conf < 0.2 {
            i = end_idx + 1;
            continue;
        }

        let median_rgb = trimmed_mean_color_line(&blocks, start_idx, end_idx, 0.10);
        if !plausible_base_luminance(&median_rgb, reference_lum) {
            i = end_idx + 1;
            continue;
        }
        let mad_lum = blocks[start_idx..=end_idx]
            .iter()
            .map(|b| b.mad_lum)
            .sum::<f64>()
            / (end_idx - start_idx + 1) as f64;

        // `trimmed_mean_color` expects grid coordinates; feed a single-row view.
        let _ = members;
        regions.push(BaseRegion {
            x_start,
            x_end,
            confidence: region_conf,
            median_rgb,
            mad_lum,
        });

        i = end_idx + 1;
    }

    let mut column_mask = vec![0.0f64; w];
    for region in &regions {
        for v in &mut column_mask[region.x_start..region.x_end] {
            *v = (*v).max(region.confidence);
        }
    }

    let dominant_color = regions
        .iter()
        .max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.median_rgb)
        .unwrap_or([0.0; 3]);

    VerticalBaseScan {
        block_width,
        regions,
        column_mask,
        dominant_color,
    }
}

/// Compute the median RGB of a vertical region spanning full height.
fn region_median_color(img: &Array3<u16>, h: usize, x_start: usize, x_end: usize) -> [f64; 3] {
    let n = h * (x_end - x_start);
    let mut channels: [Vec<f64>; 3] = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for y in 0..h {
        for x in x_start..x_end {
            for c in 0..3 {
                channels[c].push(img[[y, x, c]] as f64);
            }
        }
    }
    let mut result = [0.0; 3];
    for c in 0..3 {
        result[c] = fast_median(&mut channels[c]);
    }
    result
}

fn region_median_color_rows(img: &Array3<u16>, w: usize, y_start: usize, y_end: usize) -> [f64; 3] {
    let n = (y_end - y_start) * w;
    let mut channels: [Vec<f64>; 3] = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for y in y_start..y_end {
        for x in 0..w {
            for c in 0..3 {
                channels[c].push(img[[y, x, c]] as f64);
            }
        }
    }
    let mut result = [0.0; 3];
    for c in 0..3 {
        result[c] = fast_median(&mut channels[c]);
    }
    result
}

/// Process one edge strip and return (confidence, block_mask, base_color).
fn process_strip(
    img: &Array3<u16>,
    h: usize,
    x_start: usize,
    x_end: usize,
    interior_median: &[f64; 3],
) -> (f64, Vec<Vec<f64>>, [f64; 3]) {
    let strip_w = x_end - x_start;
    if strip_w == 0 || h == 0 {
        return (0.0, vec![], [0.0; 3]);
    }

    let grid_rows = 7;
    let grid_cols = 7;
    let block_h = h / grid_rows;
    let block_w = strip_w / grid_cols;
    if block_h == 0 || block_w == 0 {
        return (0.0, vec![], [0.0; 3]);
    }

    let mut blocks: Vec<Vec<BlockStats>> = Vec::with_capacity(grid_rows);
    for r in 0..grid_rows {
        let mut row_blocks = Vec::with_capacity(grid_cols);
        let y0 = r * block_h;
        let y1 = if r == grid_rows - 1 { h } else { y0 + block_h };
        for c_idx in 0..grid_cols {
            let bx0 = x_start + c_idx * block_w;
            let bx1 = if c_idx == grid_cols - 1 {
                x_end
            } else {
                bx0 + block_w
            };
            row_blocks.push(compute_block_stats(img, y0, y1, bx0, bx1));
        }
        blocks.push(row_blocks);
    }

    let total_blocks = grid_rows * grid_cols;
    let mad_threshold = 300.0;
    let mut low_texture: Vec<(usize, usize)> = Vec::new();
    for (r, row) in blocks.iter().enumerate().take(grid_rows) {
        for (c_idx, block) in row.iter().enumerate().take(grid_cols) {
            if block.mad_lum < mad_threshold {
                low_texture.push((r, c_idx));
            }
        }
    }

    if low_texture.is_empty() {
        return (0.0, vec![vec![0.0; grid_cols]; grid_rows], [0.0; 3]);
    }

    let chromas: Vec<[f64; 2]> = low_texture
        .iter()
        .map(|&(r, c_idx)| block_chroma(&blocks[r][c_idx].median_rgb))
        .collect();

    let mut c0s: Vec<f64> = chromas.iter().map(|c| c[0]).collect();
    let mut c1s: Vec<f64> = chromas.iter().map(|c| c[1]).collect();
    let center = [fast_median(&mut c0s), fast_median(&mut c1s)];

    let chroma_threshold = 0.05;
    let mut base_blocks: Vec<(usize, usize)> = Vec::new();
    for (i, &(r, c_idx)) in low_texture.iter().enumerate() {
        let dist = chroma_distance(chromas[i], center);
        if dist < chroma_threshold {
            base_blocks.push((r, c_idx));
        }
    }

    if base_blocks.is_empty() {
        return (0.0, vec![vec![0.0; grid_cols]; grid_rows], [0.0; 3]);
    }

    let base_fraction = base_blocks.len() as f64 / total_blocks as f64;
    let mut confidence = (base_fraction / 0.4).min(1.0);
    let strip_base_color = trimmed_mean_color(&blocks, &base_blocks, 0.10);

    let strip_chroma = block_chroma(&strip_base_color);
    let interior_chroma = block_chroma(interior_median);
    let chroma_dist = chroma_distance(strip_chroma, interior_chroma);

    let strip_lum = luminance(&strip_base_color);
    let interior_lum = luminance(interior_median);
    if !plausible_base_luminance(&strip_base_color, interior_lum) {
        return (0.0, vec![vec![0.0; grid_cols]; grid_rows], strip_base_color);
    }
    let lum_ratio = if strip_lum.max(interior_lum) > 0.0 {
        (strip_lum - interior_lum).abs() / strip_lum.max(interior_lum)
    } else {
        0.0
    };

    if chroma_dist < 0.03 && lum_ratio < 0.15 {
        confidence *= 0.1;
    } else if chroma_dist < 0.05 && lum_ratio < 0.25 {
        confidence *= 0.3;
    }

    let mut mask = vec![vec![0.0; grid_cols]; grid_rows];
    for &(r, c_idx) in &base_blocks {
        mask[r][c_idx] = 1.0;
    }

    (confidence, mask, strip_base_color)
}

fn process_row_strip(
    img: &Array3<u16>,
    w: usize,
    y_start: usize,
    y_end: usize,
    interior_median: &[f64; 3],
) -> (f64, [f64; 3]) {
    let strip_h = y_end - y_start;
    if strip_h == 0 || w == 0 {
        return (0.0, [0.0; 3]);
    }

    let grid_rows = 7;
    let grid_cols = 7;
    let block_h = strip_h / grid_rows;
    let block_w = w / grid_cols;
    if block_h == 0 || block_w == 0 {
        return (0.0, [0.0; 3]);
    }

    let mut blocks: Vec<Vec<BlockStats>> = Vec::with_capacity(grid_rows);
    for r in 0..grid_rows {
        let mut row_blocks = Vec::with_capacity(grid_cols);
        let y0 = y_start + r * block_h;
        let y1 = if r == grid_rows - 1 {
            y_end
        } else {
            y0 + block_h
        };
        for c_idx in 0..grid_cols {
            let bx0 = c_idx * block_w;
            let bx1 = if c_idx == grid_cols - 1 {
                w
            } else {
                bx0 + block_w
            };
            row_blocks.push(compute_block_stats(img, y0, y1, bx0, bx1));
        }
        blocks.push(row_blocks);
    }

    let total_blocks = grid_rows * grid_cols;
    let mad_threshold = 300.0;
    let mut low_texture: Vec<(usize, usize)> = Vec::new();
    for (r, row) in blocks.iter().enumerate().take(grid_rows) {
        for (c_idx, block) in row.iter().enumerate().take(grid_cols) {
            if block.mad_lum < mad_threshold {
                low_texture.push((r, c_idx));
            }
        }
    }

    if low_texture.is_empty() {
        return (0.0, [0.0; 3]);
    }

    let chromas: Vec<[f64; 2]> = low_texture
        .iter()
        .map(|&(r, c_idx)| block_chroma(&blocks[r][c_idx].median_rgb))
        .collect();

    let mut c0s: Vec<f64> = chromas.iter().map(|c| c[0]).collect();
    let mut c1s: Vec<f64> = chromas.iter().map(|c| c[1]).collect();
    let center = [fast_median(&mut c0s), fast_median(&mut c1s)];

    let chroma_threshold = 0.05;
    let mut base_blocks: Vec<(usize, usize)> = Vec::new();
    for (i, &(r, c_idx)) in low_texture.iter().enumerate() {
        let dist = chroma_distance(chromas[i], center);
        if dist < chroma_threshold {
            base_blocks.push((r, c_idx));
        }
    }

    if base_blocks.is_empty() {
        return (0.0, [0.0; 3]);
    }

    let base_fraction = base_blocks.len() as f64 / total_blocks as f64;
    let mut confidence = (base_fraction / 0.4).min(1.0);
    let strip_base_color = trimmed_mean_color(&blocks, &base_blocks, 0.10);

    let strip_chroma = block_chroma(&strip_base_color);
    let interior_chroma = block_chroma(interior_median);
    let chroma_dist = chroma_distance(strip_chroma, interior_chroma);

    let strip_lum = luminance(&strip_base_color);
    let interior_lum = luminance(interior_median);
    if !plausible_base_luminance(&strip_base_color, interior_lum) {
        return (0.0, strip_base_color);
    }
    let lum_ratio = if strip_lum.max(interior_lum) > 0.0 {
        (strip_lum - interior_lum).abs() / strip_lum.max(interior_lum)
    } else {
        0.0
    };

    if chroma_dist < 0.03 && lum_ratio < 0.15 {
        confidence *= 0.1;
    } else if chroma_dist < 0.05 && lum_ratio < 0.25 {
        confidence *= 0.3;
    }

    (confidence, strip_base_color)
}

/// Compute chroma coordinates for an RGB triplet.
fn block_chroma(rgb: &[f64; 3]) -> [f64; 2] {
    let sum = rgb[0] + rgb[1] + rgb[2];
    if sum < 1.0 {
        [0.0, 0.0]
    } else {
        [(rgb[0] - rgb[1]) / sum, (rgb[0] - rgb[2]) / sum]
    }
}

fn chroma_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn luminance(rgb: &[f64; 3]) -> f64 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}

fn plausible_base_luminance(rgb: &[f64; 3], reference_lum: f64) -> bool {
    let candidate_lum = luminance(rgb);
    candidate_lum >= MIN_BASE_REGION_LUMINANCE
        && (reference_lum <= MIN_BASE_REGION_LUMINANCE
            || candidate_lum >= reference_lum * MIN_BASE_REGION_REFERENCE_RATIO)
}

fn relative_luminance_delta(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let la = luminance(a);
    let lb = luminance(b);
    if la.max(lb) <= 1.0 {
        0.0
    } else {
        (la - lb).abs() / la.max(lb)
    }
}

fn relative_rgb_delta(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|c| {
        let denom = a[c].max(b[c]).max(1.0);
        (a[c] - b[c]).abs() / denom
    })
}

fn component_consensus_base_color(
    component_bases: &[[f64; 3]],
) -> Option<([f64; 3], [f64; 3], f64)> {
    let valid: Vec<[f64; 3]> = component_bases
        .iter()
        .copied()
        .filter(|rgb| rgb.iter().all(|v| v.is_finite() && *v > 0.0))
        .collect();
    if valid.len() < 2 {
        return None;
    }

    let consensus =
        std::array::from_fn(|c| valid.iter().map(|rgb| rgb[c]).sum::<f64>() / valid.len() as f64);
    let spread = std::array::from_fn(|c| {
        let min_v = valid.iter().map(|rgb| rgb[c]).fold(f64::INFINITY, f64::min);
        let max_v = valid
            .iter()
            .map(|rgb| rgb[c])
            .fold(f64::NEG_INFINITY, f64::max);
        if !min_v.is_finite() || !max_v.is_finite() {
            1.0
        } else {
            (max_v - min_v) / max_v.max(1.0)
        }
    });
    let max_spread = spread.iter().copied().fold(0.0f64, f64::max);
    if max_spread > 0.08 {
        return None;
    }

    let confidence = if max_spread <= 0.02 {
        1.0
    } else {
        (1.0 - ((max_spread - 0.02) / 0.06) * 0.35).clamp(0.65, 1.0)
    };

    Some((consensus, spread, confidence))
}

/// Compute block statistics: median RGB and MAD of luminance.
fn compute_block_stats(
    img: &Array3<u16>,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
) -> BlockStats {
    let n = (y1 - y0) * (x1 - x0);
    if n == 0 {
        return BlockStats {
            median_rgb: [0.0; 3],
            mad_lum: 0.0,
        };
    }

    let mut r_vals = Vec::with_capacity(n);
    let mut g_vals = Vec::with_capacity(n);
    let mut b_vals = Vec::with_capacity(n);
    let mut lum_vals = Vec::with_capacity(n);

    for y in y0..y1 {
        for x in x0..x1 {
            let r = img[[y, x, 0]] as f64;
            let g = img[[y, x, 1]] as f64;
            let b = img[[y, x, 2]] as f64;
            r_vals.push(r);
            g_vals.push(g);
            b_vals.push(b);
            lum_vals.push(luminance(&[r, g, b]));
        }
    }

    let med_r = fast_median(&mut r_vals);
    let med_g = fast_median(&mut g_vals);
    let med_b = fast_median(&mut b_vals);
    let med_lum = fast_median(&mut lum_vals);

    let mut abs_devs: Vec<f64> = lum_vals.iter().map(|&v| (v - med_lum).abs()).collect();
    let mad = fast_median(&mut abs_devs);

    BlockStats {
        median_rgb: [med_r, med_g, med_b],
        mad_lum: mad,
    }
}

fn region_neighbor_contrast(blocks: &[BlockStats], start_idx: usize, end_idx: usize) -> f64 {
    let mut contrast = 0.0f64;
    let region_color = trimmed_mean_color_line(blocks, start_idx, end_idx, 0.10);
    let region_chroma = block_chroma(&region_color);

    if start_idx > 0 {
        contrast = contrast.max(chroma_distance(
            region_chroma,
            block_chroma(&blocks[start_idx - 1].median_rgb),
        ));
    }
    if end_idx + 1 < blocks.len() {
        contrast = contrast.max(chroma_distance(
            region_chroma,
            block_chroma(&blocks[end_idx + 1].median_rgb),
        ));
    }

    ((contrast - 0.01) / 0.05).clamp(0.0, 1.0)
}

fn trimmed_mean_color_line(
    blocks: &[BlockStats],
    start_idx: usize,
    end_idx: usize,
    trim_frac: f64,
) -> [f64; 3] {
    if start_idx > end_idx || end_idx >= blocks.len() {
        return [0.0; 3];
    }

    let mut result = [0.0; 3];
    for (c, channel_result) in result.iter_mut().enumerate() {
        let mut vals: Vec<f64> = blocks[start_idx..=end_idx]
            .iter()
            .map(|b| b.median_rgb[c])
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        let trim_count = (n as f64 * trim_frac).floor() as usize;
        let start = trim_count.min(n - 1);
        let end = n.saturating_sub(trim_count).max(start + 1);
        let sum: f64 = vals[start..end].iter().sum();
        *channel_result = sum / (end - start) as f64;
    }
    result
}

/// Trimmed mean color from identified base blocks (trim fraction from each end).
fn trimmed_mean_color(
    blocks: &[Vec<BlockStats>],
    base_blocks: &[(usize, usize)],
    trim_frac: f64,
) -> [f64; 3] {
    if base_blocks.is_empty() {
        return [0.0; 3];
    }

    let mut result = [0.0; 3];
    for (c, channel_result) in result.iter_mut().enumerate() {
        let mut vals: Vec<f64> = base_blocks
            .iter()
            .map(|&(r, ci)| blocks[r][ci].median_rgb[c])
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        let trim_count = (n as f64 * trim_frac).floor() as usize;
        let start = trim_count.min(n - 1);
        let end = n.saturating_sub(trim_count).max(start + 1);
        let sum: f64 = vals[start..end].iter().sum();
        *channel_result = sum / (end - start) as f64;
    }
    result
}

/// In-place median via partial sort.
fn fast_median(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    }
}

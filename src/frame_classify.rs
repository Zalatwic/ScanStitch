use crate::base_detect::{self, BaseDetection, BaseRegion};
use ndarray::Array3;

const HIGH_CONFIDENCE: f64 = 0.6;
const LOW_CONFIDENCE: f64 = 0.45;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameClass {
    Intact,
    SingleBorder,
    CandidateSplit,
    Ambiguous,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StitchDisposition {
    Skip,
    Attempt,
    Forced,
}

#[derive(Debug, Clone)]
pub struct StitchDecision {
    pub disposition: StitchDisposition,
    pub reason: String,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    pub class: FrameClass,
    pub confidence: f64,
    pub left_edge_confidence: f64,
    pub right_edge_confidence: f64,
    pub content_span: (usize, usize),
    pub content_fraction: f64,
    pub activity_score: f64,
    pub component_score: f64,
    pub internal_base_regions: Vec<BaseRegion>,
}

/// Lightweight component bookkeeping for already-positive input. Negative-film rebate topology is
/// not meaningful here; multi-input positive sets are overlap-scored directly by the pipeline.
pub fn positive_input_component(img: &Array3<u16>) -> FrameAnalysis {
    let (_, width, _) = img.dim();
    FrameAnalysis {
        class: FrameClass::Ambiguous,
        // Negative-film rebate topology was not evaluated. A successful bypass is not
        // classification evidence and must not be reported as perfect confidence.
        confidence: 0.0,
        left_edge_confidence: 0.0,
        right_edge_confidence: 0.0,
        content_span: (0, width),
        content_fraction: if width == 0 { 0.0 } else { 1.0 },
        activity_score: 0.0,
        component_score: if width == 0 { 0.0 } else { 1.0 },
        internal_base_regions: Vec::new(),
    }
}

impl StitchDecision {
    pub fn should_score(&self) -> bool {
        !matches!(self.disposition, StitchDisposition::Skip)
    }
}

/// Legacy edge-only classification used by the original unit tests.
pub fn classify(detection: &BaseDetection) -> FrameClass {
    let left_high = detection.left_confidence >= HIGH_CONFIDENCE;
    let right_high = detection.right_confidence >= HIGH_CONFIDENCE;

    if left_high && right_high {
        FrameClass::Intact
    } else if left_high || right_high {
        FrameClass::SingleBorder
    } else {
        FrameClass::CandidateSplit
    }
}

/// Analyze a component using both edge rebate confidence and full-width
/// vertical base-gap detection.
pub fn analyze_component(img: &Array3<u16>, detection: &BaseDetection) -> FrameAnalysis {
    let (_, w, _) = img.dim();
    let scan = base_detect::detect_vertical_base_regions(img);

    let left_region = scan
        .regions
        .iter()
        .find(|region| region.x_start <= scan.block_width.saturating_mul(2))
        .cloned();
    let right_region = scan
        .regions
        .iter()
        .rev()
        .find(|region| region.x_end + scan.block_width.saturating_mul(2) >= w)
        .cloned();

    let internal_base_regions: Vec<BaseRegion> = scan
        .regions
        .iter()
        .filter(|region| {
            region.x_start > scan.block_width.saturating_mul(2)
                && region.x_end + scan.block_width.saturating_mul(2) < w
        })
        .cloned()
        .collect();

    let left_edge_confidence = detection
        .left_confidence
        .max(left_region.as_ref().map(|r| r.confidence).unwrap_or(0.0));
    let right_edge_confidence = detection
        .right_confidence
        .max(right_region.as_ref().map(|r| r.confidence).unwrap_or(0.0));

    let content_left = left_region
        .as_ref()
        .map(|region| region.x_end)
        .unwrap_or(0)
        .min(w);
    let content_right = right_region
        .as_ref()
        .map(|region| region.x_start)
        .unwrap_or(w)
        .max(content_left);
    let content_fraction = if w == 0 {
        0.0
    } else {
        (content_right.saturating_sub(content_left)) as f64 / w as f64
    };

    let activity_score = component_activity_score(img);
    let class = if left_edge_confidence >= HIGH_CONFIDENCE
        && right_edge_confidence >= HIGH_CONFIDENCE
        && content_fraction >= 0.82
        && internal_base_regions.len() <= 1
    {
        FrameClass::Intact
    } else if (left_edge_confidence >= HIGH_CONFIDENCE) ^ (right_edge_confidence >= HIGH_CONFIDENCE)
        && content_fraction >= 0.72
        && internal_base_regions.len() <= 1
    {
        FrameClass::SingleBorder
    } else if left_edge_confidence < LOW_CONFIDENCE
        && right_edge_confidence < LOW_CONFIDENCE
        && (internal_base_regions.len() >= 2
            || (internal_base_regions.len() == 1 && content_fraction < 0.75))
    {
        FrameClass::CandidateSplit
    } else {
        FrameClass::Ambiguous
    };

    let confidence = match class {
        FrameClass::Intact => (((left_edge_confidence + right_edge_confidence) * 0.5)
            + content_fraction)
            .mul_add(0.5, 0.0)
            .clamp(0.0, 1.0),
        FrameClass::SingleBorder => (left_edge_confidence.max(right_edge_confidence) * 0.6
            + content_fraction * 0.4)
            .clamp(0.0, 1.0),
        FrameClass::CandidateSplit => (((internal_base_regions.len() as f64) / 2.0).min(1.0) * 0.6
            + (1.0 - left_edge_confidence.max(right_edge_confidence)) * 0.4)
            .clamp(0.0, 1.0),
        FrameClass::Ambiguous => 0.35,
    };

    let class_bonus = match class {
        FrameClass::Intact => 1.0,
        FrameClass::SingleBorder => 0.75,
        FrameClass::CandidateSplit => 0.35,
        FrameClass::Ambiguous => 0.45,
    };
    let component_score = (class_bonus * 0.45
        + content_fraction * 0.25
        + activity_score * 0.15
        + left_edge_confidence * 0.075
        + right_edge_confidence * 0.075)
        .clamp(0.0, 1.0);

    FrameAnalysis {
        class,
        confidence,
        left_edge_confidence,
        right_edge_confidence,
        content_span: (content_left, content_right),
        content_fraction,
        activity_score,
        component_score,
        internal_base_regions,
    }
}

/// Decide whether two adjacent frames should be stitched together.
pub fn should_attempt_stitch(
    class1: &FrameClass,
    class2: &FrameClass,
    force_stitch: bool,
    force_no_stitch: bool,
) -> bool {
    if force_no_stitch {
        return false;
    }
    if force_stitch {
        return true;
    }
    !matches!((class1, class2), (FrameClass::Intact, FrameClass::Intact))
}

pub fn decide_stitch_attempt(
    analysis1: &FrameAnalysis,
    analysis2: &FrameAnalysis,
    force_stitch: bool,
    force_no_stitch: bool,
) -> StitchDecision {
    if force_no_stitch {
        return StitchDecision {
            disposition: StitchDisposition::Skip,
            reason: "force_no_stitch".to_string(),
            confidence: 1.0,
        };
    }

    if force_stitch {
        return StitchDecision {
            disposition: StitchDisposition::Forced,
            reason: "force_stitch".to_string(),
            confidence: 1.0,
        };
    }

    use FrameClass::{Ambiguous, CandidateSplit, Intact, SingleBorder};

    let pair = (&analysis1.class, &analysis2.class);
    let confidence = analysis1.confidence.max(analysis2.confidence);
    let min_confidence = analysis1.confidence.min(analysis2.confidence);

    match pair {
        (CandidateSplit, _) | (_, CandidateSplit) => StitchDecision {
            disposition: StitchDisposition::Attempt,
            reason: "candidate_split_present".to_string(),
            confidence: confidence.max(0.6),
        },
        (Ambiguous, Ambiguous) => StitchDecision {
            disposition: StitchDisposition::Attempt,
            reason: "ambiguous_pair_requires_scoring".to_string(),
            confidence: confidence.max(0.45),
        },
        (Ambiguous, _) | (_, Ambiguous) => StitchDecision {
            disposition: StitchDisposition::Attempt,
            reason: "ambiguous_component_requires_scoring".to_string(),
            confidence: confidence.max(0.4),
        },
        (SingleBorder, SingleBorder) => StitchDecision {
            disposition: StitchDisposition::Attempt,
            reason: "paired_single_borders".to_string(),
            confidence: confidence.max(0.55),
        },
        (SingleBorder, Intact) | (Intact, SingleBorder) => StitchDecision {
            disposition: StitchDisposition::Attempt,
            reason: "single_border_present".to_string(),
            confidence: confidence.max(0.5),
        },
        (Intact, Intact)
            if min_confidence >= 0.7
                && analysis1.content_fraction >= 0.78
                && analysis2.content_fraction >= 0.78 =>
        {
            StitchDecision {
                disposition: StitchDisposition::Skip,
                reason: "both_components_look_intact".to_string(),
                confidence: min_confidence,
            }
        }
        (Intact, Intact) => StitchDecision {
            disposition: StitchDisposition::Attempt,
            reason: "weak_intact_classification_requires_scoring".to_string(),
            confidence: confidence.max(0.35),
        },
    }
}

pub fn best_single_component_index(a: &FrameAnalysis, b: &FrameAnalysis) -> usize {
    if a.component_score >= b.component_score {
        0
    } else {
        1
    }
}

fn component_activity_score(img: &Array3<u16>) -> f64 {
    let (h, w, _) = img.dim();
    if h == 0 || w < 2 {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let mut count = 0u64;
    for y in 0..h {
        for x in 1..w {
            let prev = pixel_luma(img, y, x - 1);
            let curr = pixel_luma(img, y, x);
            sum += (curr - prev).abs();
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        (sum / count as f64 / 2500.0).clamp(0.0, 1.0)
    }
}

fn pixel_luma(img: &Array3<u16>, y: usize, x: usize) -> f64 {
    let r = img[[y, x, 0]] as f64;
    let g = img[[y, x, 1]] as f64;
    let b = img[[y, x, 2]] as f64;
    r * 0.2126 + g * 0.7152 + b * 0.0722
}

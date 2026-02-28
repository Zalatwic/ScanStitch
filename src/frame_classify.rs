use crate::base_detect::BaseDetection;

const HIGH_CONFIDENCE: f64 = 0.6;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameClass {
    Intact,
    SingleBorder,
    CandidateSplit,
}

/// Classify a cropped film frame based on base detection results.
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
    matches!(
        (class1, class2),
        (FrameClass::CandidateSplit, _)
            | (_, FrameClass::CandidateSplit)
            | (FrameClass::SingleBorder, FrameClass::SingleBorder)
    )
}

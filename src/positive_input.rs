use crate::constants::{MAX_14BIT, MAX_16BIT};
use ndarray::Array3;
use serde::Serialize;

const POSITIVE_INPUT_INSPECTION_MAX_SAMPLES: usize = 250_000;
// Hard thresholds catch obvious orange-mask-like channel medians. The aggregate score path only
// warns on high scores when R/G and G/B ratios are severe, which keeps warm positives accepted.
const POSITIVE_INPUT_ORANGE_RED_GREEN_RATIO: f64 = 1.25;
const POSITIVE_INPUT_ORANGE_GREEN_BLUE_RATIO: f64 = 1.08;
const POSITIVE_INPUT_ORANGE_RED_BLUE_DELTA: f64 = 0.08;
const POSITIVE_INPUT_ORANGE_MIN_RED_MEDIAN: f64 = 0.08;
const POSITIVE_INPUT_ORANGE_MASK_SCORE_WARNING: f64 = 0.95;
const POSITIVE_INPUT_ORANGE_SCORE_MIN_RED_GREEN_RATIO: f64 = 2.0;
const POSITIVE_INPUT_ORANGE_SCORE_MIN_GREEN_BLUE_RATIO: f64 = 1.35;

#[derive(Debug, Clone, Serialize)]
pub struct PositiveInputInspection {
    pub sample_count: usize,
    pub sample_stride: usize,
    pub channel_median: [f64; 3],
    pub red_green_ratio: f64,
    pub green_blue_ratio: f64,
    pub red_blue_delta: f64,
    pub orange_mask_score: f64,
    pub likely_negative_like: bool,
    pub accepted_high_warm_score: bool,
    pub reason: String,
}

impl PositiveInputInspection {
    pub fn accepted_high_warm_score(&self) -> bool {
        self.accepted_high_warm_score
    }
}

pub fn inspection_metrics_json(inspection: &PositiveInputInspection) -> serde_json::Value {
    serde_json::json!({
        "sample_count": inspection.sample_count,
        "sample_stride": inspection.sample_stride,
        "channel_median": inspection.channel_median,
        "red_green_ratio": inspection.red_green_ratio,
        "green_blue_ratio": inspection.green_blue_ratio,
        "red_blue_delta": inspection.red_blue_delta,
        "orange_mask_score": inspection.orange_mask_score,
        "likely_negative_like": inspection.likely_negative_like,
        "accepted_high_warm_score": inspection.accepted_high_warm_score,
        "reason": inspection.reason,
        "thresholds": {
            "red_green_ratio": POSITIVE_INPUT_ORANGE_RED_GREEN_RATIO,
            "green_blue_ratio": POSITIVE_INPUT_ORANGE_GREEN_BLUE_RATIO,
            "red_blue_delta": POSITIVE_INPUT_ORANGE_RED_BLUE_DELTA,
            "min_red_median": POSITIVE_INPUT_ORANGE_MIN_RED_MEDIAN,
            "orange_mask_score_warning": POSITIVE_INPUT_ORANGE_MASK_SCORE_WARNING,
            "score_min_red_green_ratio": POSITIVE_INPUT_ORANGE_SCORE_MIN_RED_GREEN_RATIO,
            "score_min_green_blue_ratio": POSITIVE_INPUT_ORANGE_SCORE_MIN_GREEN_BLUE_RATIO,
        }
    })
}

pub fn inspect_u16_image(img: &Array3<u16>, bit_depth: u8) -> PositiveInputInspection {
    let (height, width, channels) = img.dim();
    assert_eq!(channels, 3, "Expected 3-channel image");
    let total_pixels = height.saturating_mul(width).max(1);
    let sample_stride = ((total_pixels as f64 / POSITIVE_INPUT_INSPECTION_MAX_SAMPLES as f64)
        .sqrt()
        .ceil() as usize)
        .max(1);
    let max_value = match bit_depth {
        14 => MAX_14BIT,
        _ => MAX_16BIT,
    }
    .max(1.0);
    let mut samples = [Vec::new(), Vec::new(), Vec::new()];

    for y in (0..height).step_by(sample_stride) {
        for x in (0..width).step_by(sample_stride) {
            for c in 0..3 {
                samples[c].push((img[[y, x, c]] as f64 / max_value).clamp(0.0, 1.0));
            }
        }
    }
    let sample_count = samples[0].len();
    for channel in &mut samples {
        channel.sort_by(|a, b| a.total_cmp(b));
    }
    let channel_median =
        std::array::from_fn(|c| percentile_value(&samples[c], 0.50).unwrap_or(0.0));
    let red_green_ratio = (channel_median[0] + 1e-9) / (channel_median[1] + 1e-9);
    let green_blue_ratio = (channel_median[1] + 1e-9) / (channel_median[2] + 1e-9);
    let red_blue_delta = channel_median[0] - channel_median[2];
    let orange_mask_score = orange_mask_score(
        red_green_ratio,
        green_blue_ratio,
        red_blue_delta,
        channel_median[0],
    );
    let hard_threshold_match = channel_median[0] >= POSITIVE_INPUT_ORANGE_MIN_RED_MEDIAN
        && red_green_ratio >= POSITIVE_INPUT_ORANGE_RED_GREEN_RATIO
        && green_blue_ratio >= POSITIVE_INPUT_ORANGE_GREEN_BLUE_RATIO
        && red_blue_delta >= POSITIVE_INPUT_ORANGE_RED_BLUE_DELTA;
    let aggregate_score_match = orange_mask_score >= POSITIVE_INPUT_ORANGE_MASK_SCORE_WARNING
        && red_green_ratio >= POSITIVE_INPUT_ORANGE_SCORE_MIN_RED_GREEN_RATIO
        && green_blue_ratio >= POSITIVE_INPUT_ORANGE_SCORE_MIN_GREEN_BLUE_RATIO;
    let likely_negative_like = hard_threshold_match || aggregate_score_match;
    let accepted_high_warm_score =
        !likely_negative_like && orange_mask_score >= POSITIVE_INPUT_ORANGE_MASK_SCORE_WARNING;
    let reason = if likely_negative_like {
        format!(
            "positive-mode input has orange-mask-like channel bias (R/G {:.3}, G/B {:.3}, R-B {:.3}); verify this roll is already-positive before using positive mode",
            red_green_ratio, green_blue_ratio, red_blue_delta
        )
    } else if accepted_high_warm_score {
        format!(
            "positive-mode input has a high warm-channel score ({:.3}) but mild channel ratios (R/G {:.3}, G/B {:.3}); accepted as already-positive input",
            orange_mask_score, red_green_ratio, green_blue_ratio
        )
    } else {
        "positive-mode input channel medians do not match the orange-mask-like negative warning threshold".to_string()
    };

    PositiveInputInspection {
        sample_count,
        sample_stride,
        channel_median,
        red_green_ratio,
        green_blue_ratio,
        red_blue_delta,
        orange_mask_score,
        likely_negative_like,
        accepted_high_warm_score,
        reason,
    }
}

fn orange_mask_score(
    red_green_ratio: f64,
    green_blue_ratio: f64,
    red_blue_delta: f64,
    red_median: f64,
) -> f64 {
    let red_green_component = ((red_green_ratio - 1.0)
        / (POSITIVE_INPUT_ORANGE_RED_GREEN_RATIO - 1.0).max(1e-9))
    .clamp(0.0, 1.0);
    let green_blue_component = ((green_blue_ratio - 1.0)
        / (POSITIVE_INPUT_ORANGE_GREEN_BLUE_RATIO - 1.0).max(1e-9))
    .clamp(0.0, 1.0);
    let red_blue_component =
        (red_blue_delta / POSITIVE_INPUT_ORANGE_RED_BLUE_DELTA.max(1e-9)).clamp(0.0, 1.0);
    let red_level_component =
        (red_median / POSITIVE_INPUT_ORANGE_MIN_RED_MEDIAN.max(1e-9)).clamp(0.0, 1.0);

    red_green_component * 0.35
        + green_blue_component * 0.25
        + red_blue_component * 0.25
        + red_level_component * 0.15
}

fn percentile_value(sorted: &[f64], percentile: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted.get(index.min(sorted.len() - 1)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(rgb: [u16; 3]) -> Array3<u16> {
        Array3::from_shape_fn((8, 8, 3), |(_, _, channel)| rgb[channel])
    }

    #[test]
    fn high_orange_mask_score_flags_near_miss_negative_like_input() {
        let inspection = inspect_u16_image(&solid_image([5400, 1224, 520]), 16);

        assert!(
            inspection.orange_mask_score >= POSITIVE_INPUT_ORANGE_MASK_SCORE_WARNING,
            "expected high orange-mask score, got {}",
            inspection.orange_mask_score
        );
        assert!(
            inspection.red_blue_delta < POSITIVE_INPUT_ORANGE_RED_BLUE_DELTA,
            "fixture should exercise the aggregate-score path"
        );
        assert!(
            inspection.likely_negative_like,
            "high aggregate orange-mask score should flag negative-like input"
        );
    }

    #[test]
    fn neutral_positive_input_stays_below_negative_like_threshold() {
        let inspection = inspect_u16_image(&solid_image([12_000, 12_000, 12_000]), 16);

        assert!(!inspection.likely_negative_like);
        assert!(inspection.orange_mask_score < POSITIVE_INPUT_ORANGE_MASK_SCORE_WARNING);
    }

    #[test]
    fn warm_positive_input_with_high_score_but_mild_ratios_stays_accepted() {
        let inspection = inspect_u16_image(&solid_image([27_415, 22_332, 20_330]), 16);

        assert!(
            inspection.orange_mask_score >= POSITIVE_INPUT_ORANGE_MASK_SCORE_WARNING,
            "fixture should stay near the BIRDING047 warm-positive score"
        );
        assert!(inspection.red_green_ratio < POSITIVE_INPUT_ORANGE_RED_GREEN_RATIO);
        assert!(!inspection.likely_negative_like);
        assert!(inspection.accepted_high_warm_score());
        assert!(inspection.reason.contains("accepted as already-positive"));
    }
}

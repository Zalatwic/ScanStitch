use moxcms::{
    curve_from_gamma, ColorProfile, DataColorSpace, Layout, ProfileText, RenderingIntent,
    TransformOptions,
};
use ndarray::Array3;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::streaming;

#[derive(Debug, Clone, Serialize)]
pub struct IccProfileDiagnostics {
    pub status: String,
    pub size_bytes: usize,
    pub sha256: String,
    pub description: Option<String>,
    pub version: Option<String>,
    pub profile_class: Option<String>,
    pub data_color_space: Option<String>,
    pub profile_connection_space: Option<String>,
    pub rendering_intent: Option<String>,
    pub has_matrix_shaper: bool,
    pub has_device_to_pcs_lut: bool,
    pub has_cicp: bool,
    pub cicp_transfer_characteristics: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct EmbeddedIccProfile {
    bytes: Vec<u8>,
    pub diagnostics: IccProfileDiagnostics,
}

impl EmbeddedIccProfile {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let diagnostics = inspect_profile(&bytes);
        Self { bytes, diagnostics }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn has_identical_payload(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }

    pub fn is_valid_rgb_profile(&self) -> bool {
        self.diagnostics.status == "valid_rgb_profile"
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct InputColorTransformDiagnostics {
    pub status: String,
    pub engine: String,
    pub source: String,
    pub source_profile: IccProfileDiagnostics,
    pub destination_profile: String,
    pub rendering_intent: String,
    pub transfer_functions_applied: bool,
    pub device_to_pcs_lut_applied_when_available: bool,
    pub extended_range_preserved: bool,
    pub output_negative_sample_ratio: [f64; 3],
    pub output_above_one_sample_ratio: [f64; 3],
    pub output_nonfinite_sample_count: [u64; 3],
    pub reason: String,
}

pub struct InputColorTransformResult {
    pub image: Array3<f64>,
    pub diagnostics: InputColorTransformDiagnostics,
}

fn profile_text(text: &ProfileText) -> Option<String> {
    match text {
        ProfileText::PlainString(value) => (!value.is_empty()).then(|| value.clone()),
        ProfileText::Localizable(values) => values
            .iter()
            .find(|value| value.language.eq_ignore_ascii_case("en"))
            .or_else(|| values.first())
            .map(|value| value.value.clone())
            .filter(|value| !value.is_empty()),
        ProfileText::Description(value) => {
            (!value.ascii_string.is_empty()).then(|| value.ascii_string.clone())
        }
    }
}

fn inspect_profile(bytes: &[u8]) -> IccProfileDiagnostics {
    let sha256 = format!("{:x}", Sha256::digest(bytes));
    let parsed = ColorProfile::new_from_slice(bytes);
    match parsed {
        Ok(profile) => {
            let is_rgb = profile.color_space == DataColorSpace::Rgb;
            let has_matrix_shaper = profile.red_trc.is_some()
                && profile.green_trc.is_some()
                && profile.blue_trc.is_some();
            let has_device_to_pcs_lut = profile.lut_a_to_b_perceptual.is_some()
                || profile.lut_a_to_b_colorimetric.is_some()
                || profile.lut_a_to_b_saturation.is_some();
            let cicp_transfer_characteristics = profile
                .cicp
                .as_ref()
                .map(|cicp| format!("{:?}", cicp.transfer_characteristics));
            IccProfileDiagnostics {
                status: if is_rgb {
                    "valid_rgb_profile".to_string()
                } else {
                    "unsupported_non_rgb_profile".to_string()
                },
                size_bytes: bytes.len(),
                sha256,
                description: profile.description.as_ref().and_then(profile_text),
                version: Some(format!("{:?}", profile.version())),
                profile_class: Some(format!("{:?}", profile.profile_class)),
                data_color_space: Some(format!("{:?}", profile.color_space)),
                profile_connection_space: Some(format!("{:?}", profile.pcs)),
                rendering_intent: Some(format!("{:?}", profile.rendering_intent)),
                has_matrix_shaper,
                has_device_to_pcs_lut,
                has_cicp: profile.cicp.is_some(),
                cicp_transfer_characteristics,
                reason: if is_rgb {
                    "embedded ICC profile parsed as an RGB source profile".to_string()
                } else {
                    format!(
                        "embedded ICC profile uses {:?} device data; ScanStitch currently accepts RGB scan pixels only",
                        profile.color_space
                    )
                },
            }
        }
        Err(err) => IccProfileDiagnostics {
            status: "invalid_profile".to_string(),
            size_bytes: bytes.len(),
            sha256,
            description: None,
            version: None,
            profile_class: None,
            data_color_space: None,
            profile_connection_space: None,
            rendering_intent: None,
            has_matrix_shaper: false,
            has_device_to_pcs_lut: false,
            has_cicp: false,
            cicp_transfer_characteristics: None,
            reason: format!("embedded ICC profile could not be parsed: {err}"),
        },
    }
}

fn linear_prophoto_profile() -> ColorProfile {
    let mut profile = ColorProfile::new_pro_photo_rgb();
    let linear = curve_from_gamma(1.0);
    profile.red_trc = Some(linear.clone());
    profile.green_trc = Some(linear.clone());
    profile.blue_trc = Some(linear);
    profile
}

pub fn transform_to_linear_prophoto(
    mut image: Array3<f64>,
    source: &EmbeddedIccProfile,
) -> Result<InputColorTransformResult, String> {
    if !source.is_valid_rgb_profile() {
        return Err(source.diagnostics.reason.clone());
    }
    let source_profile = ColorProfile::new_from_slice(source.bytes())
        .map_err(|err| format!("embedded ICC profile parse failed during transform: {err}"))?;
    let destination_profile = linear_prophoto_profile();
    let transform = source_profile
        .create_transform_f64(
            Layout::Rgb,
            &destination_profile,
            Layout::Rgb,
            TransformOptions {
                rendering_intent: RenderingIntent::RelativeColorimetric,
                prefer_fixed_point: false,
                allow_extended_range_rgb_xyz: true,
                ..TransformOptions::default()
            },
        )
        .map_err(|err| format!("embedded ICC transform creation failed: {err}"))?;

    let (height, width, channels) = image.dim();
    if channels != 3 {
        return Err(format!(
            "embedded ICC transform requires three RGB channels, received {channels}"
        ));
    }
    let output_samples = image
        .as_slice_mut()
        .ok_or_else(|| "source RGB image is not contiguous".to_string())?;
    let samples_per_chunk = streaming::DEFAULT_TILE_ROWS.max(1) * width.max(1) * 3;
    for output_chunk in output_samples.chunks_mut(samples_per_chunk) {
        let source_chunk = output_chunk.to_vec();
        transform
            .transform(&source_chunk, output_chunk)
            .map_err(|err| format!("embedded ICC transform failed: {err}"))?;
    }

    let mut negative = [0u64; 3];
    let mut above_one = [0u64; 3];
    let mut nonfinite = [0u64; 3];
    for pixel in output_samples.chunks_exact(3) {
        for channel in 0..3 {
            if !pixel[channel].is_finite() {
                nonfinite[channel] += 1;
            } else if pixel[channel] < 0.0 {
                negative[channel] += 1;
            } else if pixel[channel] > 1.0 {
                above_one[channel] += 1;
            }
        }
    }
    if nonfinite.iter().any(|count| *count > 0) {
        return Err(format!(
            "embedded ICC transform produced non-finite RGB samples by channel: {:?}",
            nonfinite
        ));
    }
    let pixel_count = height.saturating_mul(width).max(1) as f64;
    Ok(InputColorTransformResult {
        image,
        diagnostics: InputColorTransformDiagnostics {
            status: "applied".to_string(),
            engine: "moxcms".to_string(),
            source: "embedded_tiff_icc_profile".to_string(),
            source_profile: source.diagnostics.clone(),
            destination_profile: "linear_prophoto_rgb_d50".to_string(),
            rendering_intent: "relative_colorimetric".to_string(),
            transfer_functions_applied: true,
            device_to_pcs_lut_applied_when_available: source
                .diagnostics
                .has_device_to_pcs_lut,
            extended_range_preserved: true,
            output_negative_sample_ratio: std::array::from_fn(|channel| {
                negative[channel] as f64 / pixel_count
            }),
            output_above_one_sample_ratio: std::array::from_fn(|channel| {
                above_one[channel] as f64 / pixel_count
            }),
            output_nonfinite_sample_count: nonfinite,
            reason: "decoded positive RGB was transformed through its embedded ICC profile into scene-linear ProPhoto RGB D50 without clipping extended output"
                .to_string(),
        },
    })
}

use crate::atomic_file;
use crate::cli::{Cli, RenderIntent};
use crate::report::PipelineReport;
use crate::tonemap::{
    self, GrainReductionSettings, RenderStyle, ToneColorProtection, ToneCurveDiagnostics,
    ToneCurveParams, TonemapApplyResult,
};
use crate::white_balance;
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub const REVIEW_SIDECAR_SCHEMA_VERSION: u32 = 1;

fn default_grain_reduction_strength() -> f64 {
    0.5
}

fn default_grain_reduction_scale() -> f64 {
    1.0
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct InteractiveRenderControls {
    pub exposure_ev: f64,
    #[serde(default)]
    pub creative_temperature: f64,
    #[serde(default)]
    pub creative_tint: f64,
    pub midpoint: f64,
    pub slope: f64,
    pub toe_lift: f64,
    pub shoulder_max: f64,
    #[serde(default)]
    pub grain_reduction_enabled: bool,
    #[serde(default = "default_grain_reduction_strength")]
    pub grain_reduction_strength: f64,
    #[serde(default = "default_grain_reduction_scale")]
    pub grain_reduction_scale: f64,
}

impl InteractiveRenderControls {
    pub fn from_tone_params(params: &ToneCurveParams) -> Self {
        Self {
            exposure_ev: 0.0,
            creative_temperature: 0.0,
            creative_tint: 0.0,
            midpoint: params.midpoint,
            slope: params.slope,
            toe_lift: params.toe_lift,
            shoulder_max: params.shoulder_max,
            grain_reduction_enabled: false,
            grain_reduction_strength: default_grain_reduction_strength(),
            grain_reduction_scale: default_grain_reduction_scale(),
        }
    }

    pub fn to_tone_params(self, auto_params: &ToneCurveParams) -> ToneCurveParams {
        ToneCurveParams {
            domain: auto_params.domain,
            midpoint: self.midpoint.clamp(0.001, 0.999),
            slope: self.slope.clamp(0.1, 16.0),
            toe_lift: self.toe_lift.clamp(0.0, 0.25),
            shoulder_max: self.shoulder_max.clamp(0.5, 1.0),
        }
    }

    pub fn grain_reduction_settings(self) -> GrainReductionSettings {
        GrainReductionSettings {
            enabled: self.grain_reduction_enabled,
            strength: self.grain_reduction_strength,
            scale: self.grain_reduction_scale,
        }
        .normalized()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewMark {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounds: Option<[f64; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewSidecar {
    pub schema_version: u32,
    pub sidecar_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_intent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub controls: Option<InteractiveRenderControls>,
    #[serde(default)]
    pub marks: Vec<ReviewMark>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decisions: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct ReviewSidecarApplication {
    pub path: PathBuf,
    pub sha256: String,
    pub sidecar: ReviewSidecar,
}

#[derive(Debug, Clone)]
pub struct InteractiveRenderCache {
    pub prophoto: Array3<f64>,
    pub auto_exposure_ev: f64,
    pub auto_tone_params: ToneCurveParams,
    pub tone_fit_diagnostics: ToneCurveDiagnostics,
    pub tone_color_protection: ToneColorProtection,
    pub report: PipelineReport,
    pub cli: Cli,
    pub output_path: PathBuf,
    pub run_started_at: SystemTime,
    pub base_confidence: f64,
    pub geometry_review_required: bool,
    pub geometry_review_reason: String,
    pub input_mode_review_required: bool,
    pub input_mode_review_reason: String,
    pub negative_response_review_required: bool,
    pub negative_response_review_reason: String,
    pub technical_white_balance_review_required: bool,
    pub technical_white_balance_review_reason: String,
}

impl InteractiveRenderCache {
    pub fn default_controls(&self) -> InteractiveRenderControls {
        let mut controls = InteractiveRenderControls::from_tone_params(&self.auto_tone_params);
        controls.exposure_ev = self.auto_exposure_ev;
        controls.creative_temperature = self.cli.white_balance.creative_temperature;
        controls.creative_tint = self.cli.white_balance.creative_tint;
        controls.grain_reduction_enabled = self.cli.grain.grain_reduction.enabled();
        controls.grain_reduction_strength = self.cli.grain.grain_strength;
        controls.grain_reduction_scale = self.cli.grain.grain_scale;
        controls
    }
}

#[derive(Debug, Clone)]
pub struct PreviewFrame {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u32>,
    pub status: String,
}

#[derive(Debug, Clone)]
pub enum PreviewCommand {
    Status(String),
    Frame(PreviewFrame),
    Exit,
}

#[derive(Debug, Clone)]
pub enum UiEvent {
    ControlsChanged(InteractiveRenderControls),
    ResetRequested,
    SaveRequested,
    QuitRequested,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewQuality {
    Fast,
    High,
}

pub fn render_style_from_intent(intent: RenderIntent) -> RenderStyle {
    match intent {
        RenderIntent::ModernClean => RenderStyle::ModernClean,
        RenderIntent::NaturalNeutral => RenderStyle::NaturalNeutral,
        RenderIntent::FilmFaithful => RenderStyle::FilmFaithful,
    }
}

pub fn load_review_sidecar(
    path: &Path,
) -> Result<ReviewSidecarApplication, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let sha256 = sha256_hex(&bytes);
    let sidecar = serde_json::from_slice::<ReviewSidecar>(&bytes)?;
    if sidecar.schema_version != REVIEW_SIDECAR_SCHEMA_VERSION {
        return Err(format!(
            "unsupported review sidecar schema_version {}; expected {}",
            sidecar.schema_version, REVIEW_SIDECAR_SCHEMA_VERSION
        )
        .into());
    }
    if sidecar.sidecar_type != "scanstitch_guided_review" {
        return Err(format!("unsupported review sidecar type `{}`", sidecar.sidecar_type).into());
    }
    Ok(ReviewSidecarApplication {
        path: path.to_path_buf(),
        sha256,
        sidecar,
    })
}

pub fn controls_with_review_sidecar(
    default_controls: InteractiveRenderControls,
    auto_params: &ToneCurveParams,
    sidecar: &ReviewSidecar,
) -> InteractiveRenderControls {
    let Some(mut controls) = sidecar.controls else {
        return default_controls;
    };
    controls.exposure_ev = controls.exposure_ev.clamp(-4.0, 4.0);
    controls.creative_temperature = controls.creative_temperature.clamp(-1.0, 1.0);
    controls.creative_tint = controls.creative_tint.clamp(-1.0, 1.0);
    let tone = controls.to_tone_params(auto_params);
    controls.midpoint = tone.midpoint;
    controls.slope = tone.slope;
    controls.toe_lift = tone.toe_lift;
    controls.shoulder_max = tone.shoulder_max;
    let grain = controls.grain_reduction_settings();
    controls.grain_reduction_strength = grain.strength;
    controls.grain_reduction_scale = grain.scale;
    controls
}

pub fn write_review_sidecar(
    path: &Path,
    cache: &InteractiveRenderCache,
    controls: &InteractiveRenderControls,
) -> Result<ReviewSidecar, Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let sidecar = ReviewSidecar {
        schema_version: REVIEW_SIDECAR_SCHEMA_VERSION,
        sidecar_type: "scanstitch_guided_review".to_string(),
        render_intent: Some(cache.cli.render_intent.as_str().to_string()),
        controls: Some(*controls),
        marks: Vec::new(),
        decisions: Some(serde_json::json!({
            "tone_domain": cache.auto_tone_params.domain.as_str(),
            "source": "interactive_controls",
            "non_destructive": true,
            "grain_reduction_independent_of_render_intent": true
            ,"creative_white_balance_separate_from_technical_master": true
        })),
    };
    let json = serde_json::to_string_pretty(&sidecar)?;
    atomic_file::write_bytes(path, json.as_bytes())?;
    Ok(sidecar)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

pub fn render_interactive_image(
    cache: &InteractiveRenderCache,
    controls: &InteractiveRenderControls,
) -> TonemapApplyResult {
    let tone_params = controls.to_tone_params(&cache.auto_tone_params);
    let render_style = render_style_from_intent(cache.cli.render_intent);
    let creative = (controls.creative_temperature.abs() > f64::EPSILON
        || controls.creative_tint.abs() > f64::EPSILON)
        .then(|| {
            white_balance::apply_creative_white_balance(
                &cache.prophoto,
                controls.creative_temperature,
                controls.creative_tint,
            )
        });
    let creative_source = creative
        .as_ref()
        .map(|result| &result.image)
        .unwrap_or(&cache.prophoto);
    if controls.exposure_ev.abs() <= f64::EPSILON {
        return tonemap::apply_tonemap_with_params_color_protection_style_and_grain_diagnostics(
            creative_source,
            &tone_params,
            &cache.tone_color_protection,
            render_style,
            controls.grain_reduction_settings(),
        );
    }

    let exposure_scale = 2.0f64.powf(controls.exposure_ev);
    let exposed = creative_source.mapv(|v| (v * exposure_scale).max(0.0));
    tonemap::apply_tonemap_with_params_color_protection_style_and_grain_diagnostics(
        &exposed,
        &tone_params,
        &cache.tone_color_protection,
        render_style,
        controls.grain_reduction_settings(),
    )
}

pub fn render_owned_batch_image(
    cache: &InteractiveRenderCache,
    mut image: Array3<f64>,
    controls: &InteractiveRenderControls,
) -> TonemapApplyResult {
    let tone_params = controls.to_tone_params(&cache.auto_tone_params);
    let render_style = render_style_from_intent(cache.cli.render_intent);
    if controls.creative_temperature.abs() > f64::EPSILON
        || controls.creative_tint.abs() > f64::EPSILON
    {
        image = white_balance::apply_creative_white_balance_owned(
            image,
            controls.creative_temperature,
            controls.creative_tint,
        )
        .image;
    }
    if controls.exposure_ev.abs() > f64::EPSILON {
        let exposure_scale = 2.0f64.powf(controls.exposure_ev);
        image.mapv_inplace(|value| (value * exposure_scale).max(0.0));
    }
    tonemap::apply_tonemap_owned_with_params_color_protection_style_and_grain_diagnostics(
        image,
        &tone_params,
        &cache.tone_color_protection,
        render_style,
        controls.grain_reduction_settings(),
    )
}

pub fn render_preview_frame(
    cache: &InteractiveRenderCache,
    controls: &InteractiveRenderControls,
    max_width: usize,
    max_height: usize,
    quality: PreviewQuality,
    status: impl Into<String>,
) -> PreviewFrame {
    let source = match quality {
        PreviewQuality::Fast => downscale_nearest(&cache.prophoto, max_width, max_height),
        PreviewQuality::High => downscale_box(&cache.prophoto, max_width, max_height),
    };
    let tone_params = controls.to_tone_params(&cache.auto_tone_params);
    let exposure_scale = 2.0f64.powf(controls.exposure_ev);
    let creative = (controls.creative_temperature.abs() > f64::EPSILON
        || controls.creative_tint.abs() > f64::EPSILON)
        .then(|| {
            white_balance::apply_creative_white_balance(
                &source,
                controls.creative_temperature,
                controls.creative_tint,
            )
        });
    let creative_source = creative
        .as_ref()
        .map(|result| &result.image)
        .unwrap_or(&source);
    let exposed = if controls.exposure_ev.abs() <= f64::EPSILON {
        creative_source.clone()
    } else {
        creative_source.mapv(|v| (v * exposure_scale).max(0.0))
    };
    let rendered = tonemap::apply_tonemap_with_params_color_protection_style_and_grain_diagnostics(
        &exposed,
        &tone_params,
        &cache.tone_color_protection,
        render_style_from_intent(cache.cli.render_intent),
        controls.grain_reduction_settings(),
    )
    .image;
    let (height, width, _) = rendered.dim();

    PreviewFrame {
        width,
        height,
        pixels: rgb_buffer_from_rendered(&rendered),
        status: status.into(),
    }
}

fn preview_dimensions(
    source_width: usize,
    source_height: usize,
    max_width: usize,
    max_height: usize,
) -> (usize, usize) {
    let source_width = source_width.max(1);
    let source_height = source_height.max(1);
    let max_width = max_width.max(1);
    let max_height = max_height.max(1);
    let scale = (max_width as f64 / source_width as f64)
        .min(max_height as f64 / source_height as f64)
        .min(1.0);
    let width = ((source_width as f64 * scale).round() as usize).max(1);
    let height = ((source_height as f64 * scale).round() as usize).max(1);
    (width, height)
}

fn downscale_nearest(img: &Array3<f64>, max_width: usize, max_height: usize) -> Array3<f64> {
    let (source_height, source_width, channels) = img.dim();
    let (width, height) = preview_dimensions(source_width, source_height, max_width, max_height);
    let mut out = Array3::<f64>::zeros((height, width, channels));

    for y in 0..height {
        let source_y = (y * source_height / height).min(source_height.saturating_sub(1));
        for x in 0..width {
            let source_x = (x * source_width / width).min(source_width.saturating_sub(1));
            for c in 0..channels {
                out[[y, x, c]] = img[[source_y, source_x, c]];
            }
        }
    }

    out
}

fn downscale_box(img: &Array3<f64>, max_width: usize, max_height: usize) -> Array3<f64> {
    let (source_height, source_width, channels) = img.dim();
    let (width, height) = preview_dimensions(source_width, source_height, max_width, max_height);
    if width == source_width && height == source_height {
        return img.clone();
    }

    let mut out = Array3::<f64>::zeros((height, width, channels));
    for y in 0..height {
        let y0 = y * source_height / height;
        let y1 = ((y + 1) * source_height)
            .div_ceil(height)
            .min(source_height);
        for x in 0..width {
            let x0 = x * source_width / width;
            let x1 = ((x + 1) * source_width).div_ceil(width).min(source_width);
            let mut sum = vec![0.0f64; channels];
            let mut count = 0usize;
            for source_y in y0..y1.max(y0 + 1).min(source_height) {
                for source_x in x0..x1.max(x0 + 1).min(source_width) {
                    for c in 0..channels {
                        sum[c] += img[[source_y, source_x, c]];
                    }
                    count += 1;
                }
            }
            let denom = count.max(1) as f64;
            for c in 0..channels {
                out[[y, x, c]] = sum[c] / denom;
            }
        }
    }

    out
}

fn rgb_buffer_from_rendered(img: &Array3<f64>) -> Vec<u32> {
    let (height, width, _) = img.dim();
    let mut pixels = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = encode_preview_channel(img[[y, x, 0]]);
            let g = encode_preview_channel(img[[y, x, 1]]);
            let b = encode_preview_channel(img[[y, x, 2]]);
            pixels.push(((r as u32) << 16) | ((g as u32) << 8) | b as u32);
        }
    }
    pixels
}

fn encode_preview_channel(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

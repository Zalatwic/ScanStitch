use image::{Rgb, RgbImage};
use nalgebra::Matrix3;
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const TARGET_SAMPLING_MANIFEST_SCHEMA_VERSION: u32 = 1;
const MIN_PATCHES_PER_VALIDATION_PARTITION: usize = 12;
const ROBUST_SCALE_FLOOR: f64 = 0.0015;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSamplingManifest {
    pub schema_version: u32,
    #[serde(default)]
    pub measurement: Map<String, Value>,
    /// Effective scanner code range. Use 14 for 14-bit samples padded into a 16-bit TIFF container.
    pub scanner_signal_bit_depth: u8,
    pub grid: TargetGrid,
    pub patches: Vec<TargetReferencePatch>,
    #[serde(default)]
    pub reference_review: Option<TargetReferenceReviewApproval>,
    pub captures: Vec<TargetCapture>,
    #[serde(default)]
    pub quality: TargetSamplingQuality,
    #[serde(default = "default_preview_max_dimension")]
    pub preview_max_dimension: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetGrid {
    pub rows: usize,
    pub columns: usize,
    #[serde(default = "default_patch_inset_fraction")]
    pub patch_inset_fraction: f64,
    #[serde(default = "default_samples_per_axis")]
    pub samples_per_axis: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetReferencePatch {
    pub id: String,
    pub row: usize,
    pub column: usize,
    #[serde(default, alias = "xyz")]
    pub reference_xyz: Option<[f64; 3]>,
    #[serde(default, alias = "lab")]
    pub reference_lab: Option<[f64; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetReferenceReviewApproval {
    #[serde(default)]
    pub approved: bool,
    pub reference_patch_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetCapture {
    pub id: String,
    pub role: TargetCaptureRole,
    pub image: PathBuf,
    pub corners: TargetChartCorners,
    #[serde(default)]
    pub corner_review: Option<TargetCornerReviewApproval>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetCornerReviewApproval {
    #[serde(default)]
    pub approved: bool,
    pub decoded_pixel_sha256: String,
    pub chart_corners_sha256: String,
    pub overlay_pixel_sha256: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetCaptureRole {
    Training,
    HeldOut,
}

impl TargetCaptureRole {
    fn label(self) -> &'static str {
        match self {
            Self::Training => "training",
            Self::HeldOut => "held_out",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetChartCorners {
    pub top_left: [f64; 2],
    pub top_right: [f64; 2],
    pub bottom_right: [f64; 2],
    pub bottom_left: [f64; 2],
}

impl TargetChartCorners {
    fn points(self) -> [[f64; 2]; 4] {
        [
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSamplingQuality {
    #[serde(default = "default_minimum_source_bits")]
    pub minimum_source_bits_per_sample: u8,
    #[serde(default = "default_minimum_retained_fraction")]
    pub minimum_retained_fraction: f64,
    #[serde(default = "default_maximum_clipped_fraction")]
    pub maximum_clipped_fraction: f64,
    #[serde(default = "default_maximum_channel_spread")]
    pub maximum_channel_robust_spread: f64,
    #[serde(default = "default_maximum_spatial_delta")]
    pub maximum_spatial_cell_delta: f64,
    #[serde(default = "default_outlier_z_limit")]
    pub outlier_z_limit: f64,
    #[serde(default = "default_clipping_margin")]
    pub clipping_margin: f64,
}

impl Default for TargetSamplingQuality {
    fn default() -> Self {
        Self {
            minimum_source_bits_per_sample: default_minimum_source_bits(),
            minimum_retained_fraction: default_minimum_retained_fraction(),
            maximum_clipped_fraction: default_maximum_clipped_fraction(),
            maximum_channel_robust_spread: default_maximum_channel_spread(),
            maximum_spatial_cell_delta: default_maximum_spatial_delta(),
            outlier_z_limit: default_outlier_z_limit(),
            clipping_margin: default_clipping_margin(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TargetSamplingDiagnostics {
    pub status: &'static str,
    pub schema_version: u32,
    pub coordinate_domain: &'static str,
    pub interpolation: &'static str,
    pub robust_estimator: &'static str,
    pub scanner_signal_bit_depth: u8,
    pub scanner_signal_code_max: u16,
    pub grid: TargetGrid,
    pub quality: TargetSamplingQuality,
    pub reference_patch_count: usize,
    pub reference_patches: Vec<TargetReferencePatch>,
    pub reference_patch_sha256: String,
    pub reference_review: TargetReferenceReviewDiagnostics,
    pub training_patch_count: usize,
    pub held_out_patch_count: usize,
    pub rejected_patch_count: usize,
    pub corner_review_approved_capture_count: usize,
    pub corner_review_required_capture_count: usize,
    pub captures: Vec<TargetCaptureDiagnostics>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TargetReferenceReviewDiagnostics {
    pub status: &'static str,
    pub approved: bool,
    pub expected_reference_patch_sha256: Option<String>,
    pub reference_patch_sha256_matched: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TargetCornerReviewDiagnostics {
    pub status: &'static str,
    pub approved: bool,
    pub expected_decoded_pixel_sha256: Option<String>,
    pub decoded_pixel_sha256_matched: Option<bool>,
    pub expected_chart_corners_sha256: Option<String>,
    pub chart_corners_sha256_matched: Option<bool>,
    pub expected_overlay_pixel_sha256: Option<String>,
    pub overlay_pixel_sha256_matched: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TargetCaptureDiagnostics {
    pub capture_id: String,
    pub role: &'static str,
    pub image_path: String,
    pub image_sha256: String,
    pub decoded_pixel_sha256: String,
    pub width: usize,
    pub height: usize,
    pub source_bits_per_sample: u8,
    pub source_color_type: String,
    pub working_bit_depth: u8,
    pub working_range_transform: &'static str,
    pub working_range_scale_factor: f64,
    pub source_code_min: [u16; 3],
    pub source_code_max: [u16; 3],
    pub working_code_min: [u16; 3],
    pub working_code_max: [u16; 3],
    pub source_icc_profile: Option<crate::input_color::IccProfileDiagnostics>,
    pub source_icc_transform_applied: bool,
    pub dng_level_normalization: Option<Value>,
    pub orientation_tag: Option<u16>,
    pub orientation_transform: String,
    pub chart_corners: TargetChartCorners,
    pub chart_corners_sha256: String,
    pub overlay_pixel_sha256: String,
    pub corner_review: TargetCornerReviewDiagnostics,
    pub unit_square_to_oriented_image: [[f64; 3]; 3],
    pub accepted_patch_count: usize,
    pub rejected_patch_count: usize,
    pub patches: Vec<TargetPatchSamplingDiagnostics>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TargetPatchSamplingDiagnostics {
    pub sample_id: String,
    pub chart_patch_id: String,
    pub row: usize,
    pub column: usize,
    pub status: &'static str,
    pub rejection_reasons: Vec<String>,
    pub projected_center: [f64; 2],
    pub projected_sample_corners: [[f64; 2]; 4],
    pub scanner_xy: [f64; 2],
    pub source_rgb: [f64; 3],
    pub requested_sample_count: usize,
    pub in_bounds_sample_count: usize,
    pub retained_sample_count: usize,
    pub outlier_sample_count: usize,
    pub clipped_sample_count: usize,
    pub clipped_channel_sample_count: [usize; 3],
    pub retained_fraction: f64,
    pub clipped_fraction: f64,
    pub channel_robust_spread: [f64; 3],
    pub maximum_spatial_cell_delta: [f64; 3],
}

pub struct TargetSamplingPreview {
    pub capture_id: String,
    pub image: RgbImage,
}

pub struct TargetSamplingResult {
    pub measurement: Value,
    pub diagnostics: TargetSamplingDiagnostics,
    pub previews: Vec<TargetSamplingPreview>,
    pub corner_review_manifest: TargetSamplingManifest,
}

#[derive(Debug, Clone)]
struct SamplePoint {
    rgb: [f64; 3],
    grid_x: usize,
    grid_y: usize,
    clipped: bool,
    clipped_channels: [bool; 3],
    robust_z: f64,
}

#[derive(Clone, Copy)]
struct PatchSamplingContext<'a> {
    image: &'a Array3<u16>,
    homography: &'a Matrix3<f64>,
    coordinate_mapping: crate::scanner_linearization::ScannerCoordinateMapping,
    grid: &'a TargetGrid,
    quality: &'a TargetSamplingQuality,
    capture: &'a TargetCapture,
    signal_code_max: f64,
}

pub fn sample_target_manifest(
    manifest: &TargetSamplingManifest,
    manifest_directory: &Path,
) -> Result<TargetSamplingResult, String> {
    validate_target_sampling_manifest(manifest)?;
    let mut training_patches = Vec::<Value>::new();
    let mut held_out_patches = Vec::<Value>::new();
    let mut capture_diagnostics = Vec::with_capacity(manifest.captures.len());
    let mut previews = Vec::with_capacity(manifest.captures.len());
    let mut seen_file_hashes = HashMap::<String, String>::new();
    let mut seen_pixel_hashes = HashMap::<String, String>::new();
    let mut corner_review_manifest = manifest.clone();
    let reference_patch_sha256 = sha256_reference_patches(&manifest.patches);
    let reference_review = target_reference_review_diagnostics(
        manifest.reference_review.as_ref(),
        reference_patch_sha256.as_str(),
    );
    corner_review_manifest.reference_review = Some(TargetReferenceReviewApproval {
        approved: reference_review.status == "approved",
        reference_patch_sha256: reference_patch_sha256.clone(),
    });

    for (capture_index, capture) in manifest.captures.iter().enumerate() {
        let image_path = if capture.image.is_absolute() {
            capture.image.clone()
        } else {
            manifest_directory.join(&capture.image)
        };
        let canonical_path = std::fs::canonicalize(&image_path).map_err(|error| {
            format!(
                "failed to resolve capture `{}` image {}: {error}",
                capture.id,
                image_path.display()
            )
        })?;
        let image_sha256 = sha256_file(&canonical_path)?;
        if let Some(previous_capture) =
            seen_file_hashes.insert(image_sha256.clone(), capture.id.clone())
        {
            return Err(format!(
                "captures `{previous_capture}` and `{}` have identical SHA-256 {}; training and held-out evidence must come from distinct image payloads",
                capture.id, image_sha256
            ));
        }
        let loaded =
            crate::tiff_io::load_tiff_u16(&canonical_path, manifest.scanner_signal_bit_depth)
                .map_err(|error| {
                    format!(
                        "failed to decode capture `{}` image {}: {error}",
                        capture.id,
                        canonical_path.display()
                    )
                })?;
        if loaded.diagnostics.source_bits_per_sample
            < manifest.quality.minimum_source_bits_per_sample
        {
            return Err(format!(
                "capture `{}` source precision {} bits is below required {} bits for colour-target measurement",
                capture.id,
                loaded.diagnostics.source_bits_per_sample,
                manifest.quality.minimum_source_bits_per_sample
            ));
        }
        let (height, width, channels) = loaded.image.dim();
        if channels != 3 {
            return Err(format!(
                "capture `{}` decoded with {channels} channels; RGB is required",
                capture.id
            ));
        }
        let decoded_pixel_sha256 =
            crate::tiff_io::sha256_decoded_pixels(&loaded.image, manifest.scanner_signal_bit_depth);
        if let Some(previous_capture) =
            seen_pixel_hashes.insert(decoded_pixel_sha256.clone(), capture.id.clone())
        {
            return Err(format!(
                "captures `{previous_capture}` and `{}` have identical decoded-pixel SHA-256 {}; metadata or container changes do not make reused pixels independent validation evidence",
                capture.id, decoded_pixel_sha256
            ));
        }
        let chart_corners_sha256 = sha256_chart_corners(capture.corners);
        validate_chart_corners(capture.corners, width, height, &capture.id)?;
        let homography = unit_square_to_quadrilateral(capture.corners.points())?;
        let coordinate_mapping = crate::scanner_linearization::ScannerCoordinateMapping {
            orientation_tag: loaded.diagnostics.orientation.tag_value,
            scanner_frame_width: loaded.diagnostics.orientation.source_width,
            scanner_frame_height: loaded.diagnostics.orientation.source_height,
        };
        let signal_code_max = bit_depth_code_max(manifest.scanner_signal_bit_depth) as f64;
        let sampling_context = PatchSamplingContext {
            image: &loaded.image,
            homography: &homography,
            coordinate_mapping,
            grid: &manifest.grid,
            quality: &manifest.quality,
            capture,
            signal_code_max,
        };
        let mut patches = Vec::with_capacity(manifest.patches.len());
        let mut capture_output = Vec::with_capacity(manifest.patches.len());
        for reference in &manifest.patches {
            let (output_patch, diagnostics) = sample_reference_patch(sampling_context, reference)?;
            if diagnostics.status == "accepted" {
                capture_output.push(output_patch);
            }
            patches.push(diagnostics);
        }
        let accepted_patch_count = patches
            .iter()
            .filter(|patch| patch.status == "accepted")
            .count();
        let rejected_patch_count = patches.len() - accepted_patch_count;
        match capture.role {
            TargetCaptureRole::Training => training_patches.extend(capture_output),
            TargetCaptureRole::HeldOut => held_out_patches.extend(capture_output),
        }
        let preview = render_sampling_preview(
            &loaded.image,
            capture.corners,
            &patches,
            manifest.preview_max_dimension,
        );
        let overlay_pixel_sha256 = sha256_review_overlay(&preview);
        let corner_review = target_corner_review_diagnostics(
            capture,
            decoded_pixel_sha256.as_str(),
            chart_corners_sha256.as_str(),
            overlay_pixel_sha256.as_str(),
        );
        corner_review_manifest.captures[capture_index].image = canonical_path.clone();
        corner_review_manifest.captures[capture_index].corner_review =
            Some(TargetCornerReviewApproval {
                approved: corner_review.status == "approved",
                decoded_pixel_sha256: decoded_pixel_sha256.clone(),
                chart_corners_sha256: chart_corners_sha256.clone(),
                overlay_pixel_sha256: overlay_pixel_sha256.clone(),
            });
        let diagnostics = TargetCaptureDiagnostics {
            capture_id: capture.id.clone(),
            role: capture.role.label(),
            image_path: canonical_path.to_string_lossy().to_string(),
            image_sha256,
            decoded_pixel_sha256,
            width,
            height,
            source_bits_per_sample: loaded.diagnostics.source_bits_per_sample,
            source_color_type: loaded.diagnostics.color_type.clone(),
            working_bit_depth: loaded.diagnostics.working_range.target_bit_depth,
            working_range_transform: loaded.diagnostics.working_range.transform.as_str(),
            working_range_scale_factor: loaded.diagnostics.working_range.scale_factor,
            source_code_min: loaded.diagnostics.working_range.source_min,
            source_code_max: loaded.diagnostics.working_range.source_max,
            working_code_min: loaded.diagnostics.working_range.working_min,
            working_code_max: loaded.diagnostics.working_range.working_max,
            source_icc_profile: loaded.diagnostics.source_icc_profile.clone(),
            source_icc_transform_applied: false,
            dng_level_normalization: loaded.diagnostics.dng_level_normalization.as_ref().map(
                |levels| {
                    serde_json::json!({
                        "status": levels.status,
                        "black_level": levels.black_level,
                        "white_level": levels.white_level,
                        "source_code_max": levels.source_code_max,
                        "clipped_below_black": levels.clipped_below_black,
                        "clipped_above_white": levels.clipped_above_white,
                        "reason": levels.reason,
                    })
                },
            ),
            orientation_tag: loaded.diagnostics.orientation.tag_value,
            orientation_transform: loaded.diagnostics.orientation.transform.clone(),
            chart_corners: capture.corners,
            chart_corners_sha256,
            overlay_pixel_sha256,
            corner_review,
            unit_square_to_oriented_image: matrix_rows(&homography),
            accepted_patch_count,
            rejected_patch_count,
            patches,
        };
        previews.push(TargetSamplingPreview {
            capture_id: capture.id.clone(),
            image: preview,
        });
        capture_diagnostics.push(diagnostics);
    }

    let rejected_patch_count = capture_diagnostics
        .iter()
        .map(|capture| capture.rejected_patch_count)
        .sum::<usize>();
    let partition_counts_valid = training_patches.len() >= MIN_PATCHES_PER_VALIDATION_PARTITION
        && held_out_patches.len() >= MIN_PATCHES_PER_VALIDATION_PARTITION;
    let corner_review_approved_capture_count = capture_diagnostics
        .iter()
        .filter(|capture| capture.corner_review.status == "approved")
        .count();
    let corner_review_required_capture_count =
        capture_diagnostics.len() - corner_review_approved_capture_count;
    let status = if rejected_patch_count > 0 || !partition_counts_valid {
        "rejected"
    } else {
        match (
            reference_review.status == "approved",
            corner_review_required_capture_count == 0,
        ) {
            (true, true) => "accepted",
            (true, false) => "requires_corner_approval",
            (false, true) => "requires_reference_approval",
            (false, false) => "requires_reference_and_corner_approval",
        }
    };
    let diagnostics = TargetSamplingDiagnostics {
        status,
        schema_version: TARGET_SAMPLING_MANIFEST_SCHEMA_VERSION,
        coordinate_domain: "normalized_original_scanner_frame_via_inverse_exif_orientation",
        interpolation: "bilinear_on_uniform_projective_patch_lattice",
        robust_estimator: "multichannel_mad_outlier_rejection_then_tukey_biweight_mean",
        scanner_signal_bit_depth: manifest.scanner_signal_bit_depth,
        scanner_signal_code_max: bit_depth_code_max(manifest.scanner_signal_bit_depth),
        grid: manifest.grid.clone(),
        quality: manifest.quality.clone(),
        reference_patch_count: manifest.patches.len(),
        reference_patches: manifest.patches.clone(),
        reference_patch_sha256,
        reference_review,
        training_patch_count: training_patches.len(),
        held_out_patch_count: held_out_patches.len(),
        rejected_patch_count,
        corner_review_approved_capture_count,
        corner_review_required_capture_count,
        captures: capture_diagnostics,
    };
    let mut measurement = manifest.measurement.clone();
    measurement.insert(
        "training_patches".to_string(),
        Value::Array(training_patches),
    );
    measurement.insert(
        "held_out_patches".to_string(),
        Value::Array(held_out_patches),
    );
    measurement.insert(
        "target_sampling".to_string(),
        serde_json::to_value(&diagnostics)
            .map_err(|error| format!("failed to serialize target sampling diagnostics: {error}"))?,
    );
    Ok(TargetSamplingResult {
        measurement: Value::Object(measurement),
        diagnostics,
        previews,
        corner_review_manifest,
    })
}

fn target_reference_review_diagnostics(
    review: Option<&TargetReferenceReviewApproval>,
    actual_reference_patch_sha256: &str,
) -> TargetReferenceReviewDiagnostics {
    let Some(review) = review else {
        return TargetReferenceReviewDiagnostics {
            status: "not_declared",
            approved: false,
            expected_reference_patch_sha256: None,
            reference_patch_sha256_matched: None,
        };
    };
    let matched = review
        .reference_patch_sha256
        .eq_ignore_ascii_case(actual_reference_patch_sha256);
    let status = if !matched {
        "reference_patch_mismatch"
    } else if review.approved {
        "approved"
    } else {
        "requires_approval"
    };
    TargetReferenceReviewDiagnostics {
        status,
        approved: review.approved,
        expected_reference_patch_sha256: Some(review.reference_patch_sha256.clone()),
        reference_patch_sha256_matched: Some(matched),
    }
}

fn target_corner_review_diagnostics(
    capture: &TargetCapture,
    actual_decoded_pixel_sha256: &str,
    actual_chart_corners_sha256: &str,
    actual_overlay_pixel_sha256: &str,
) -> TargetCornerReviewDiagnostics {
    let Some(review) = &capture.corner_review else {
        return TargetCornerReviewDiagnostics {
            status: "not_declared",
            approved: false,
            expected_decoded_pixel_sha256: None,
            decoded_pixel_sha256_matched: None,
            expected_chart_corners_sha256: None,
            chart_corners_sha256_matched: None,
            expected_overlay_pixel_sha256: None,
            overlay_pixel_sha256_matched: None,
        };
    };
    let decoded_pixels_matched = review
        .decoded_pixel_sha256
        .eq_ignore_ascii_case(actual_decoded_pixel_sha256);
    let chart_corners_matched = review
        .chart_corners_sha256
        .eq_ignore_ascii_case(actual_chart_corners_sha256);
    let overlay_pixels_matched = review
        .overlay_pixel_sha256
        .eq_ignore_ascii_case(actual_overlay_pixel_sha256);
    let status = if !decoded_pixels_matched {
        "decoded_pixel_mismatch"
    } else if !chart_corners_matched {
        "chart_corners_mismatch"
    } else if !overlay_pixels_matched {
        "overlay_pixel_mismatch"
    } else if review.approved {
        "approved"
    } else {
        "requires_approval"
    };
    TargetCornerReviewDiagnostics {
        status,
        approved: review.approved,
        expected_decoded_pixel_sha256: Some(review.decoded_pixel_sha256.clone()),
        decoded_pixel_sha256_matched: Some(decoded_pixels_matched),
        expected_chart_corners_sha256: Some(review.chart_corners_sha256.clone()),
        chart_corners_sha256_matched: Some(chart_corners_matched),
        expected_overlay_pixel_sha256: Some(review.overlay_pixel_sha256.clone()),
        overlay_pixel_sha256_matched: Some(overlay_pixels_matched),
    }
}

pub fn validate_target_sampling_manifest(manifest: &TargetSamplingManifest) -> Result<(), String> {
    if manifest.schema_version != TARGET_SAMPLING_MANIFEST_SCHEMA_VERSION {
        return Err(format!(
            "unsupported target sampling manifest schema_version {}; expected {}",
            manifest.schema_version, TARGET_SAMPLING_MANIFEST_SCHEMA_VERSION
        ));
    }
    if manifest.grid.rows == 0 || manifest.grid.columns == 0 {
        return Err("target grid rows and columns must be positive".to_string());
    }
    if !(8..=16).contains(&manifest.scanner_signal_bit_depth) {
        return Err("scanner_signal_bit_depth must be in [8,16]".to_string());
    }
    if !manifest.grid.patch_inset_fraction.is_finite()
        || !(0.05..=0.40).contains(&manifest.grid.patch_inset_fraction)
    {
        return Err("grid.patch_inset_fraction must be finite and in [0.05,0.40]".to_string());
    }
    if !(9..=101).contains(&manifest.grid.samples_per_axis)
        || manifest.grid.samples_per_axis.is_multiple_of(2)
    {
        return Err("grid.samples_per_axis must be an odd integer in [9,101]".to_string());
    }
    validate_quality(&manifest.quality)?;
    if !(512..=4096).contains(&manifest.preview_max_dimension) {
        return Err("preview_max_dimension must be in [512,4096]".to_string());
    }
    if manifest.patches.len() < MIN_PATCHES_PER_VALIDATION_PARTITION {
        return Err(format!(
            "target sampling requires at least {MIN_PATCHES_PER_VALIDATION_PARTITION} reference patches"
        ));
    }
    let mut patch_ids = HashSet::<String>::new();
    let mut patch_cells = HashSet::<(usize, usize)>::new();
    for (index, patch) in manifest.patches.iter().enumerate() {
        let id = patch.id.trim();
        if id.is_empty() {
            return Err(format!("patches[{index}].id must not be empty"));
        }
        if !patch_ids.insert(id.to_ascii_lowercase()) {
            return Err(format!("duplicate target patch id `{id}`"));
        }
        if patch.row >= manifest.grid.rows || patch.column >= manifest.grid.columns {
            return Err(format!(
                "patch `{id}` cell [{},{}] lies outside {}x{} grid",
                patch.row, patch.column, manifest.grid.rows, manifest.grid.columns
            ));
        }
        if !patch_cells.insert((patch.row, patch.column)) {
            return Err(format!(
                "multiple reference patches occupy grid cell [{},{}]",
                patch.row, patch.column
            ));
        }
        match (patch.reference_xyz, patch.reference_lab) {
            (Some(xyz), None) => validate_reference_triplet(xyz, "reference_xyz", index)?,
            (None, Some(lab)) => validate_reference_triplet(lab, "reference_lab", index)?,
            (Some(_), Some(_)) => {
                return Err(format!(
                    "patches[{index}] must declare exactly one of reference_xyz/xyz or reference_lab/lab"
                ));
            }
            (None, None) => {
                return Err(format!(
                    "patches[{index}] requires reference_xyz/xyz or reference_lab/lab"
                ));
            }
        }
    }
    if let Some(review) = &manifest.reference_review {
        if !valid_sha256(&review.reference_patch_sha256) {
            return Err(
                "reference_review.reference_patch_sha256 must be 64 hexadecimal characters"
                    .to_string(),
            );
        }
    }
    if manifest.captures.len() < 2 {
        return Err(
            "target sampling requires at least one training and one held-out capture".into(),
        );
    }
    let mut capture_ids = HashSet::<String>::new();
    let mut preview_names = HashSet::<String>::new();
    let mut training_count = 0usize;
    let mut held_out_count = 0usize;
    for (index, capture) in manifest.captures.iter().enumerate() {
        let id = capture.id.trim();
        if id.is_empty() {
            return Err(format!("captures[{index}].id must not be empty"));
        }
        if !capture_ids.insert(id.to_ascii_lowercase()) {
            return Err(format!("duplicate capture id `{id}`"));
        }
        let preview_name = safe_capture_filename(id).to_ascii_lowercase();
        if !preview_names.insert(preview_name) {
            return Err(format!(
                "capture id `{id}` collides with another capture after preview filename sanitization"
            ));
        }
        match capture.role {
            TargetCaptureRole::Training => training_count += 1,
            TargetCaptureRole::HeldOut => held_out_count += 1,
        }
        if let Some(review) = &capture.corner_review {
            for (field, value) in [
                ("decoded_pixel_sha256", review.decoded_pixel_sha256.as_str()),
                ("chart_corners_sha256", review.chart_corners_sha256.as_str()),
                ("overlay_pixel_sha256", review.overlay_pixel_sha256.as_str()),
            ] {
                if !valid_sha256(value) {
                    return Err(format!(
                        "captures[{index}].corner_review.{field} must be 64 hexadecimal characters"
                    ));
                }
            }
        }
    }
    if training_count == 0 || held_out_count == 0 {
        return Err(
            "target sampling requires both training and held_out capture roles".to_string(),
        );
    }
    for reserved in [
        "schema_version",
        "record_type",
        "profile_id",
        "training_patches",
        "held_out_patches",
        "patches",
        "fit",
        "target_sampling",
        "scanner_rgb_to_xyz",
        "work_to_xyz",
        "fit_matrix",
        "correction_matrix",
    ] {
        if manifest.measurement.contains_key(reserved) {
            return Err(format!(
                "measurement metadata must not predeclare reserved sampler/fitter field `{reserved}`"
            ));
        }
    }
    for required in ["target", "reference"] {
        if !manifest
            .measurement
            .get(required)
            .is_some_and(Value::is_object)
        {
            return Err(format!(
                "measurement.{required} must be a JSON object describing target provenance"
            ));
        }
    }
    Ok(())
}

fn validate_quality(quality: &TargetSamplingQuality) -> Result<(), String> {
    if !(8..=16).contains(&quality.minimum_source_bits_per_sample) {
        return Err("quality.minimum_source_bits_per_sample must be in [8,16]".to_string());
    }
    for (field, value, minimum, maximum) in [
        (
            "minimum_retained_fraction",
            quality.minimum_retained_fraction,
            0.5,
            1.0,
        ),
        (
            "maximum_clipped_fraction",
            quality.maximum_clipped_fraction,
            0.0,
            0.20,
        ),
        (
            "maximum_channel_robust_spread",
            quality.maximum_channel_robust_spread,
            0.005,
            0.50,
        ),
        (
            "maximum_spatial_cell_delta",
            quality.maximum_spatial_cell_delta,
            0.002,
            0.50,
        ),
        ("outlier_z_limit", quality.outlier_z_limit, 3.0, 12.0),
        ("clipping_margin", quality.clipping_margin, 0.0, 0.05),
    ] {
        if !value.is_finite() || !(minimum..=maximum).contains(&value) {
            return Err(format!(
                "quality.{field} must be finite and in [{minimum},{maximum}]"
            ));
        }
    }
    Ok(())
}

fn validate_reference_triplet(values: [f64; 3], field: &str, index: usize) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "patches[{index}].{field} must contain finite values"
        ));
    }
    if field == "reference_xyz" && values.iter().any(|value| *value < 0.0) {
        return Err(format!(
            "patches[{index}].reference_xyz values must be non-negative"
        ));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub fn sha256_reference_patches(patches: &[TargetReferencePatch]) -> String {
    let mut ordered = patches.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        (left.row, left.column, left.id.as_str()).cmp(&(right.row, right.column, right.id.as_str()))
    });
    let mut hash = Sha256::new();
    hash.update(b"scanstitch-target-reference-patches-v1");
    hash.update((ordered.len() as u64).to_le_bytes());
    for patch in ordered {
        hash.update((patch.row as u64).to_le_bytes());
        hash.update((patch.column as u64).to_le_bytes());
        hash.update((patch.id.len() as u64).to_le_bytes());
        hash.update(patch.id.as_bytes());
        match (patch.reference_xyz, patch.reference_lab) {
            (Some(xyz), None) => {
                hash.update([0]);
                for value in xyz {
                    hash_reference_value(&mut hash, value);
                }
            }
            (None, Some(lab)) => {
                hash.update([1]);
                for value in lab {
                    hash_reference_value(&mut hash, value);
                }
            }
            (Some(xyz), Some(lab)) => {
                hash.update([2]);
                for value in xyz.into_iter().chain(lab) {
                    hash_reference_value(&mut hash, value);
                }
            }
            (None, None) => hash.update([3]),
        }
    }
    format!("{:x}", hash.finalize())
}

fn hash_reference_value(hash: &mut Sha256, value: f64) {
    let canonical = format!("{value:.12e}");
    hash.update((canonical.len() as u64).to_le_bytes());
    hash.update(canonical.as_bytes());
}

pub fn sha256_chart_corners(corners: TargetChartCorners) -> String {
    let mut hash = Sha256::new();
    hash.update(b"scanstitch-target-chart-corners-v1");
    for point in corners.points() {
        for coordinate in point {
            hash.update(coordinate.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn validate_chart_corners(
    corners: TargetChartCorners,
    width: usize,
    height: usize,
    capture_id: &str,
) -> Result<(), String> {
    let points = corners.points();
    let maximum_x = width.saturating_sub(1) as f64;
    let maximum_y = height.saturating_sub(1) as f64;
    for (index, point) in points.iter().enumerate() {
        if point.iter().any(|value| !value.is_finite())
            || !(0.0..=maximum_x).contains(&point[0])
            || !(0.0..=maximum_y).contains(&point[1])
        {
            return Err(format!(
                "capture `{capture_id}` chart corner {index} {:?} lies outside oriented image bounds [0,{maximum_x}]x[0,{maximum_y}]",
                point
            ));
        }
    }
    let mut sign = 0.0f64;
    for index in 0..4 {
        let a = points[index];
        let b = points[(index + 1) % 4];
        let c = points[(index + 2) % 4];
        let cross = cross_2d(subtract(b, a), subtract(c, b));
        if cross.abs() < 1e-6 {
            return Err(format!(
                "capture `{capture_id}` chart quadrilateral has a degenerate corner"
            ));
        }
        if index == 0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return Err(format!(
                "capture `{capture_id}` chart corners must form a convex non-self-intersecting quadrilateral in top-left, top-right, bottom-right, bottom-left order"
            ));
        }
    }
    let area = polygon_area(&points).abs();
    if area < 256.0 {
        return Err(format!(
            "capture `{capture_id}` projected chart area {area:.1} px^2 is too small"
        ));
    }
    Ok(())
}

fn unit_square_to_quadrilateral(points: [[f64; 2]; 4]) -> Result<Matrix3<f64>, String> {
    let [p0, p1, p2, p3] = points;
    let dx1 = p1[0] - p2[0];
    let dx2 = p3[0] - p2[0];
    let dx3 = p0[0] - p1[0] + p2[0] - p3[0];
    let dy1 = p1[1] - p2[1];
    let dy2 = p3[1] - p2[1];
    let dy3 = p0[1] - p1[1] + p2[1] - p3[1];
    let (g, h) = if dx3.abs() < 1e-12 && dy3.abs() < 1e-12 {
        (0.0, 0.0)
    } else {
        let denominator = dx1 * dy2 - dx2 * dy1;
        if denominator.abs() < 1e-12 {
            return Err("chart quadrilateral does not define a stable projective mapping".into());
        }
        (
            (dx3 * dy2 - dx2 * dy3) / denominator,
            (dx1 * dy3 - dx3 * dy1) / denominator,
        )
    };
    let matrix = Matrix3::new(
        p1[0] - p0[0] + g * p1[0],
        p3[0] - p0[0] + h * p3[0],
        p0[0],
        p1[1] - p0[1] + g * p1[1],
        p3[1] - p0[1] + h * p3[1],
        p0[1],
        g,
        h,
        1.0,
    );
    if matrix.iter().any(|value| !value.is_finite()) || matrix.try_inverse().is_none() {
        return Err("chart quadrilateral produced a singular projective mapping".to_string());
    }
    for point in [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]] {
        project(&matrix, point)?;
    }
    Ok(matrix)
}

fn sample_reference_patch(
    context: PatchSamplingContext<'_>,
    reference: &TargetReferencePatch,
) -> Result<(Value, TargetPatchSamplingDiagnostics), String> {
    let PatchSamplingContext {
        image,
        homography,
        coordinate_mapping,
        grid,
        quality,
        capture,
        signal_code_max,
    } = context;
    let inset = grid.patch_inset_fraction;
    let u0 = (reference.column as f64 + inset) / grid.columns as f64;
    let u1 = (reference.column as f64 + 1.0 - inset) / grid.columns as f64;
    let v0 = (reference.row as f64 + inset) / grid.rows as f64;
    let v1 = (reference.row as f64 + 1.0 - inset) / grid.rows as f64;
    let projected_sample_corners = [
        project(homography, [u0, v0])?,
        project(homography, [u1, v0])?,
        project(homography, [u1, v1])?,
        project(homography, [u0, v1])?,
    ];
    let projected_area = polygon_area(&projected_sample_corners).abs();
    if projected_area < 16.0 {
        return Err(format!(
            "capture `{}` patch `{}` projected sample area {projected_area:.2} px^2 is too small",
            capture.id, reference.id
        ));
    }
    let projected_center = project(
        homography,
        [
            (reference.column as f64 + 0.5) / grid.columns as f64,
            (reference.row as f64 + 0.5) / grid.rows as f64,
        ],
    )?;
    let scanner_xy = coordinate_mapping
        .normalized_scanner_coordinates(projected_center[0], projected_center[1])?;
    let (height, width, _) = image.dim();
    let mut samples = Vec::with_capacity(grid.samples_per_axis * grid.samples_per_axis);
    for grid_y in 0..grid.samples_per_axis {
        let fy = (grid_y as f64 + 0.5) / grid.samples_per_axis as f64;
        let v = v0 + (v1 - v0) * fy;
        for grid_x in 0..grid.samples_per_axis {
            let fx = (grid_x as f64 + 0.5) / grid.samples_per_axis as f64;
            let u = u0 + (u1 - u0) * fx;
            let point = project(homography, [u, v])?;
            if point[0] < 0.0
                || point[1] < 0.0
                || point[0] > width.saturating_sub(1) as f64
                || point[1] > height.saturating_sub(1) as f64
            {
                continue;
            }
            let rgb = bilinear_rgb(image, point[0], point[1], signal_code_max);
            let clipped_channels = std::array::from_fn(|channel| {
                rgb[channel] <= quality.clipping_margin
                    || rgb[channel] >= 1.0 - quality.clipping_margin
            });
            samples.push(SamplePoint {
                rgb,
                grid_x,
                grid_y,
                clipped: clipped_channels.iter().any(|clipped| *clipped),
                clipped_channels,
                robust_z: 0.0,
            });
        }
    }
    let requested_sample_count = grid.samples_per_axis * grid.samples_per_axis;
    let in_bounds_sample_count = samples.len();
    if samples.is_empty() {
        return Err(format!(
            "capture `{}` patch `{}` has no in-bounds projective samples",
            capture.id, reference.id
        ));
    }
    let medians = std::array::from_fn(|channel| {
        quantile(
            samples.iter().map(|sample| sample.rgb[channel]).collect(),
            0.5,
        )
    });
    let scales: [f64; 3] = std::array::from_fn(|channel| {
        let mad = quantile(
            samples
                .iter()
                .map(|sample| (sample.rgb[channel] - medians[channel]).abs())
                .collect(),
            0.5,
        );
        (1.4826 * mad).max(ROBUST_SCALE_FLOOR)
    });
    for sample in &mut samples {
        sample.robust_z = (0..3)
            .map(|channel| (sample.rgb[channel] - medians[channel]).abs() / scales[channel])
            .fold(0.0, f64::max);
    }
    let retained_indices = samples
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| {
            (!sample.clipped && sample.robust_z <= quality.outlier_z_limit).then_some(index)
        })
        .collect::<Vec<_>>();
    let clipped_sample_count = samples.iter().filter(|sample| sample.clipped).count();
    let clipped_channel_sample_count = std::array::from_fn(|channel| {
        samples
            .iter()
            .filter(|sample| sample.clipped_channels[channel])
            .count()
    });
    let outlier_sample_count = samples
        .iter()
        .filter(|sample| !sample.clipped && sample.robust_z > quality.outlier_z_limit)
        .count();
    let retained_fraction = retained_indices.len() as f64 / requested_sample_count as f64;
    let clipped_fraction = clipped_sample_count as f64 / requested_sample_count as f64;
    let source_rgb = robust_weighted_mean(
        &samples,
        &retained_indices,
        quality.outlier_z_limit,
        medians,
    );
    let channel_robust_spread = std::array::from_fn(|channel| {
        let values = retained_indices
            .iter()
            .map(|index| samples[*index].rgb[channel])
            .collect::<Vec<_>>();
        if values.is_empty() {
            1.0
        } else {
            quantile(values.clone(), 0.95) - quantile(values, 0.05)
        }
    });
    let maximum_spatial_cell_delta = spatial_cell_delta(
        &samples,
        &retained_indices,
        grid.samples_per_axis,
        source_rgb,
    );
    let mut rejection_reasons = Vec::new();
    if in_bounds_sample_count != requested_sample_count {
        rejection_reasons.push(format!(
            "only {in_bounds_sample_count}/{requested_sample_count} projective samples were in bounds"
        ));
    }
    if retained_fraction < quality.minimum_retained_fraction {
        rejection_reasons.push(format!(
            "retained sample fraction {retained_fraction:.4} is below required {:.4}",
            quality.minimum_retained_fraction
        ));
    }
    if clipped_fraction > quality.maximum_clipped_fraction {
        rejection_reasons.push(format!(
            "clipped sample fraction {clipped_fraction:.4} exceeds allowed {:.4}",
            quality.maximum_clipped_fraction
        ));
    }
    for channel in 0..3 {
        if channel_robust_spread[channel] > quality.maximum_channel_robust_spread {
            rejection_reasons.push(format!(
                "channel {channel} robust p95-p05 spread {:.5} exceeds allowed {:.5}",
                channel_robust_spread[channel], quality.maximum_channel_robust_spread
            ));
        }
        if maximum_spatial_cell_delta[channel] > quality.maximum_spatial_cell_delta {
            rejection_reasons.push(format!(
                "channel {channel} 3x3-cell spatial delta {:.5} exceeds allowed {:.5}",
                maximum_spatial_cell_delta[channel], quality.maximum_spatial_cell_delta
            ));
        }
    }
    let status = if rejection_reasons.is_empty() {
        "accepted"
    } else {
        "rejected"
    };
    let sample_id = format!("{}:{}", capture.id.trim(), reference.id.trim());
    let mut output = Map::new();
    output.insert("id".to_string(), Value::String(sample_id.clone()));
    output.insert(
        "chart_patch_id".to_string(),
        Value::String(reference.id.trim().to_string()),
    );
    output.insert("scanner_xy".to_string(), serde_json::json!(scanner_xy));
    output.insert("rgb".to_string(), serde_json::json!(source_rgb));
    if let Some(xyz) = reference.reference_xyz {
        output.insert("xyz".to_string(), serde_json::json!(xyz));
    } else if let Some(lab) = reference.reference_lab {
        output.insert("lab".to_string(), serde_json::json!(lab));
    }
    Ok((
        Value::Object(output),
        TargetPatchSamplingDiagnostics {
            sample_id,
            chart_patch_id: reference.id.trim().to_string(),
            row: reference.row,
            column: reference.column,
            status,
            rejection_reasons,
            projected_center,
            projected_sample_corners,
            scanner_xy,
            source_rgb,
            requested_sample_count,
            in_bounds_sample_count,
            retained_sample_count: retained_indices.len(),
            outlier_sample_count,
            clipped_sample_count,
            clipped_channel_sample_count,
            retained_fraction,
            clipped_fraction,
            channel_robust_spread,
            maximum_spatial_cell_delta,
        },
    ))
}

fn robust_weighted_mean(
    samples: &[SamplePoint],
    retained_indices: &[usize],
    z_limit: f64,
    fallback: [f64; 3],
) -> [f64; 3] {
    let mut weighted_sum = [0.0f64; 3];
    let mut weight_sum = 0.0f64;
    for index in retained_indices {
        let sample = &samples[*index];
        let ratio = (sample.robust_z / z_limit).clamp(0.0, 1.0);
        let weight = (1.0 - ratio * ratio).powi(2);
        for (channel, total) in weighted_sum.iter_mut().enumerate() {
            *total += weight * sample.rgb[channel];
        }
        weight_sum += weight;
    }
    if weight_sum <= 1e-12 {
        fallback
    } else {
        std::array::from_fn(|channel| weighted_sum[channel] / weight_sum)
    }
}

fn spatial_cell_delta(
    samples: &[SamplePoint],
    retained_indices: &[usize],
    samples_per_axis: usize,
    center: [f64; 3],
) -> [f64; 3] {
    let mut cells = vec![Vec::<usize>::new(); 9];
    for index in retained_indices {
        let sample = &samples[*index];
        let cell_x = (sample.grid_x * 3 / samples_per_axis).min(2);
        let cell_y = (sample.grid_y * 3 / samples_per_axis).min(2);
        cells[cell_y * 3 + cell_x].push(*index);
    }
    std::array::from_fn(|channel| {
        cells
            .iter()
            .filter(|indices| indices.len() >= 3)
            .map(|indices| {
                let cell_median = quantile(
                    indices
                        .iter()
                        .map(|index| samples[*index].rgb[channel])
                        .collect(),
                    0.5,
                );
                (cell_median - center[channel]).abs()
            })
            .fold(0.0, f64::max)
    })
}

fn bilinear_rgb(image: &Array3<u16>, x: f64, y: f64, signal_code_max: f64) -> [f64; 3] {
    let (height, width, _) = image.dim();
    let x0 = x.floor().clamp(0.0, width.saturating_sub(1) as f64) as usize;
    let y0 = y.floor().clamp(0.0, height.saturating_sub(1) as f64) as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    std::array::from_fn(|channel| {
        let top =
            image[[y0, x0, channel]] as f64 * (1.0 - fx) + image[[y0, x1, channel]] as f64 * fx;
        let bottom =
            image[[y1, x0, channel]] as f64 * (1.0 - fx) + image[[y1, x1, channel]] as f64 * fx;
        (top * (1.0 - fy) + bottom * fy) / signal_code_max
    })
}

fn project(matrix: &Matrix3<f64>, point: [f64; 2]) -> Result<[f64; 2], String> {
    let denominator = matrix[(2, 0)] * point[0] + matrix[(2, 1)] * point[1] + matrix[(2, 2)];
    if !denominator.is_finite() || denominator.abs() < 1e-12 {
        return Err("projective chart mapping reached an invalid denominator".to_string());
    }
    let x = (matrix[(0, 0)] * point[0] + matrix[(0, 1)] * point[1] + matrix[(0, 2)]) / denominator;
    let y = (matrix[(1, 0)] * point[0] + matrix[(1, 1)] * point[1] + matrix[(1, 2)]) / denominator;
    if !x.is_finite() || !y.is_finite() {
        return Err("projective chart mapping produced non-finite coordinates".to_string());
    }
    Ok([x, y])
}

fn matrix_rows(matrix: &Matrix3<f64>) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[(row, column)]))
}

fn quantile(mut values: Vec<f64>, fraction: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let position = fraction.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let weight = position - lower as f64;
        values[lower] * (1.0 - weight) + values[upper] * weight
    }
}

fn render_sampling_preview(
    source: &Array3<u16>,
    chart_corners: TargetChartCorners,
    patches: &[TargetPatchSamplingDiagnostics],
    maximum_dimension: u32,
) -> RgbImage {
    let (height, width, _) = source.dim();
    let scale = (maximum_dimension as f64 / width.max(height) as f64).min(1.0);
    let preview_width = ((width as f64 * scale).round() as u32).max(1);
    let preview_height = ((height as f64 * scale).round() as u32).max(1);
    let tonal_limits = preview_tonal_limits(source);
    let mut preview = RgbImage::new(preview_width, preview_height);
    for output_y in 0..preview_height {
        let source_y = if preview_height <= 1 {
            0
        } else {
            ((output_y as f64 * (height - 1) as f64 / (preview_height - 1) as f64).round() as usize)
                .min(height - 1)
        };
        for output_x in 0..preview_width {
            let source_x = if preview_width <= 1 {
                0
            } else {
                ((output_x as f64 * (width - 1) as f64 / (preview_width - 1) as f64).round()
                    as usize)
                    .min(width - 1)
            };
            let pixel = std::array::from_fn(|channel| {
                let value = source[[source_y, source_x, channel]] as f64 / u16::MAX as f64;
                let normalized = ((value - tonal_limits[channel][0])
                    / (tonal_limits[channel][1] - tonal_limits[channel][0]).max(1e-6))
                .clamp(0.0, 1.0)
                .powf(1.0 / 2.2);
                (normalized * 255.0).round() as u8
            });
            preview.put_pixel(output_x, output_y, Rgb(pixel));
        }
    }
    let map_point = |point: [f64; 2]| {
        [
            if width <= 1 {
                0.0
            } else {
                point[0] * (preview_width - 1) as f64 / (width - 1) as f64
            },
            if height <= 1 {
                0.0
            } else {
                point[1] * (preview_height - 1) as f64 / (height - 1) as f64
            },
        ]
    };
    let chart = chart_corners.points().map(map_point);
    draw_polygon(&mut preview, &chart, Rgb([255, 210, 0]), 2);
    for patch in patches {
        let points = patch.projected_sample_corners.map(map_point);
        let color = if patch.status == "accepted" {
            Rgb([0, 255, 96])
        } else {
            Rgb([255, 32, 48])
        };
        draw_polygon(&mut preview, &points, color, 1);
        draw_marker(&mut preview, map_point(patch.projected_center), color);
    }
    preview
}

fn sha256_review_overlay(preview: &RgbImage) -> String {
    let mut hash = Sha256::new();
    hash.update(b"scanstitch-target-review-overlay-rgb-v1");
    hash.update(preview.width().to_le_bytes());
    hash.update(preview.height().to_le_bytes());
    hash.update(preview.as_raw());
    format!("{:x}", hash.finalize())
}

fn preview_tonal_limits(source: &Array3<u16>) -> [[f64; 2]; 3] {
    let (height, width, _) = source.dim();
    let stride = ((height.saturating_mul(width) / 100_000).max(1) as f64)
        .sqrt()
        .floor()
        .max(1.0) as usize;
    std::array::from_fn(|channel| {
        let mut samples = Vec::new();
        for y in (0..height).step_by(stride) {
            for x in (0..width).step_by(stride) {
                samples.push(source[[y, x, channel]] as f64 / u16::MAX as f64);
            }
        }
        [quantile(samples.clone(), 0.01), quantile(samples, 0.99)]
    })
}

fn draw_polygon(image: &mut RgbImage, points: &[[f64; 2]; 4], color: Rgb<u8>, thickness: i32) {
    for index in 0..4 {
        draw_line(
            image,
            points[index],
            points[(index + 1) % 4],
            color,
            thickness,
        );
    }
}

fn draw_line(image: &mut RgbImage, start: [f64; 2], end: [f64; 2], color: Rgb<u8>, thickness: i32) {
    let steps = (end[0] - start[0])
        .abs()
        .max((end[1] - start[1]).abs())
        .ceil()
        .max(1.0) as usize;
    for step in 0..=steps {
        let fraction = step as f64 / steps as f64;
        let x = start[0] + (end[0] - start[0]) * fraction;
        let y = start[1] + (end[1] - start[1]) * fraction;
        for dy in -thickness..=thickness {
            for dx in -thickness..=thickness {
                put_pixel_if_in_bounds(image, x.round() as i32 + dx, y.round() as i32 + dy, color);
            }
        }
    }
}

fn draw_marker(image: &mut RgbImage, point: [f64; 2], color: Rgb<u8>) {
    let x = point[0].round() as i32;
    let y = point[1].round() as i32;
    for offset in -3..=3 {
        put_pixel_if_in_bounds(image, x + offset, y, color);
        put_pixel_if_in_bounds(image, x, y + offset, color);
    }
}

fn put_pixel_if_in_bounds(image: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>) {
    if x >= 0 && y >= 0 && x < image.width() as i32 && y < image.height() as i32 {
        image.put_pixel(x as u32, y as u32, color);
    }
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("failed to open {} for hashing: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn safe_capture_filename(capture_id: &str) -> String {
    let filename = capture_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if filename.is_empty() {
        "capture".to_string()
    } else {
        filename
    }
}

fn subtract(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn cross_2d(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn polygon_area(points: &[[f64; 2]; 4]) -> f64 {
    (0..4)
        .map(|index| {
            let next = (index + 1) % 4;
            points[index][0] * points[next][1] - points[next][0] * points[index][1]
        })
        .sum::<f64>()
        * 0.5
}

fn default_patch_inset_fraction() -> f64 {
    0.20
}

fn default_samples_per_axis() -> usize {
    25
}

fn default_minimum_source_bits() -> u8 {
    12
}

fn default_minimum_retained_fraction() -> f64 {
    0.70
}

fn default_maximum_clipped_fraction() -> f64 {
    0.01
}

fn default_maximum_channel_spread() -> f64 {
    0.08
}

fn default_maximum_spatial_delta() -> f64 {
    0.04
}

fn default_outlier_z_limit() -> f64 {
    6.0
}

fn default_clipping_margin() -> f64 {
    0.0001
}

fn default_preview_max_dimension() -> u32 {
    1800
}

fn bit_depth_code_max(bit_depth: u8) -> u16 {
    if bit_depth >= 16 {
        u16::MAX
    } else {
        ((1u32 << bit_depth) - 1) as u16
    }
}

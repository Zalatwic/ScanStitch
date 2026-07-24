use clap::{Parser, Subcommand};
use nalgebra::{DMatrix, Matrix3, Vector3};
use scanstitch::atomic_file;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "scanstitch-calibrate",
    about = "Create or update local ScanStitch calibration library records"
)]
struct CalibrateCli {
    #[command(subcommand)]
    command: CalibrateCommand,
}

#[derive(Subcommand, Debug)]
enum CalibrateCommand {
    /// Sample a perspective-distorted target from independent training and held-out TIFF/DNG captures.
    SampleTarget {
        /// Sampling manifest containing chart geometry, references, and capture roles.
        #[arg(long)]
        manifest: PathBuf,

        /// Measurement JSON to write for scanner-target or roll-target.
        #[arg(long)]
        output: PathBuf,

        /// Directory for overlays, sampling-report.json, and the hash-bound combined review manifest.
        #[arg(long)]
        preview_dir: PathBuf,

        /// Replace existing measurement/audit outputs.
        #[arg(long)]
        force: bool,
    },

    /// Ingest a scanner target measurement JSON into calibration/scanners.
    ScannerTarget {
        /// Calibration library directory.
        #[arg(long)]
        library: PathBuf,

        /// Scanner profile ID to write.
        #[arg(long)]
        profile_id: String,

        /// Measurement/profile JSON containing disjoint training/held-out target patches or a pre-fit scanner_rgb_to_xyz matrix.
        #[arg(long)]
        measurements: PathBuf,

        /// Replace an existing scanner profile record with the same profile ID.
        #[arg(long)]
        force: bool,
    },

    /// Ingest a roll target measurement JSON into calibration/rolls.
    RollTarget {
        /// Calibration library directory.
        #[arg(long)]
        library: PathBuf,

        /// Roll profile ID to write.
        #[arg(long)]
        profile_id: String,

        /// Scanner profile ID this roll correction was measured against.
        #[arg(long)]
        scanner_profile: String,

        /// Measurement/profile JSON containing disjoint training/held-out target patches or a pre-fit correction_matrix.
        #[arg(long)]
        measurements: PathBuf,

        /// Replace an existing roll profile record with the same profile ID.
        #[arg(long)]
        force: bool,
    },

    /// Store a base-only roll profile when no target frame is available.
    RollBase {
        /// Calibration library directory.
        #[arg(long)]
        library: PathBuf,

        /// Roll profile ID to write.
        #[arg(long)]
        profile_id: String,

        /// Optional scanner profile ID this base measurement came from.
        #[arg(long)]
        scanner_profile: Option<String>,

        /// Film stock label.
        #[arg(long)]
        film_stock: String,

        /// Process label, such as C-41.
        #[arg(long, default_value = "unknown")]
        process: String,

        /// Comma-separated RGB base color, for example 12000,7000,3000.
        #[arg(long)]
        base_color: String,

        /// Confidence in this base-only metadata record.
        #[arg(long, default_value_t = 0.75)]
        confidence: f64,

        /// Replace an existing roll profile record with the same profile ID.
        #[arg(long)]
        force: bool,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NegativeResponseFitMeasurements {
    #[serde(default)]
    model_id: Option<String>,
    scene_rgb_to_xyz_d50: [[f64; 3]; 3],
    training_patches: Vec<NegativeResponseTrainingPatch>,
    held_out_patches: Vec<NegativeResponseHeldOutPatch>,
    confidence: f64,
    #[serde(default = "default_negative_response_anchor_percentile")]
    white_anchor_percentile: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NegativeResponseTrainingPatch {
    scanner_density: [f64; 3],
    reference_layer_density: [f64; 3],
    reference_log_exposure: [f64; 3],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NegativeResponseHeldOutPatch {
    scanner_density: [f64; 3],
    reference_log_exposure: [f64; 3],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScannerLinearizationFitMeasurements {
    #[serde(default)]
    model_id: Option<String>,
    training_samples: Vec<ScannerLinearizationSample>,
    held_out_samples: Vec<ScannerLinearizationSample>,
    #[serde(default = "zero_triplet")]
    black_level_normalized: [f64; 3],
    #[serde(default = "one_triplet")]
    white_level_normalized: [f64; 3],
    #[serde(default = "zero_triplet")]
    additive_flare_normalized: [f64; 3],
    #[serde(default)]
    shading_gain_polynomial: Option<[[f64; 6]; 3]>,
    confidence: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScannerLinearizationSample {
    scanner_signal: [f64; 3],
    reference_transmittance: [f64; 3],
    #[serde(default)]
    x: f64,
    #[serde(default)]
    y: f64,
}

fn zero_triplet() -> [f64; 3] {
    [0.0; 3]
}

fn one_triplet() -> [f64; 3] {
    [1.0; 3]
}

fn default_negative_response_anchor_percentile() -> f64 {
    0.995
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = CalibrateCli::parse();
    match cli.command {
        CalibrateCommand::SampleTarget {
            manifest,
            output,
            preview_dir,
            force,
        } => {
            sample_target(&manifest, &output, &preview_dir, force)?;
        }
        CalibrateCommand::ScannerTarget {
            library,
            profile_id,
            measurements,
            force,
        } => {
            let mut value = read_object_json(&measurements)?;
            fit_scanner_target_if_needed(&profile_id, &mut value)?;
            require_any_matrix(
                &value,
                &["scanner_rgb_to_xyz", "work_to_xyz", "fit_matrix"],
                "scanner target measurement",
            )?;
            set_common_fields(&mut value, "scanner_profile", &profile_id);
            write_record(&library, "scanners", &profile_id, &value, force)?;
        }
        CalibrateCommand::RollTarget {
            library,
            profile_id,
            scanner_profile,
            measurements,
            force,
        } => {
            let mut value = read_object_json(&measurements)?;
            fit_roll_target_if_needed(&library, &scanner_profile, &profile_id, &mut value)?;
            require_any_matrix(
                &value,
                &["correction_matrix", "lut", "negative_response"],
                "roll target measurement",
            )?;
            set_common_fields(&mut value, "roll_profile", &profile_id);
            value
                .as_object_mut()
                .expect("object checked")
                .entry("scanner_profile_id")
                .or_insert(Value::String(scanner_profile));
            write_record(&library, "rolls", &profile_id, &value, force)?;
        }
        CalibrateCommand::RollBase {
            library,
            profile_id,
            scanner_profile,
            film_stock,
            process,
            base_color,
            confidence,
            force,
        } => {
            let base_color = parse_rgb_triplet(&base_color)?;
            if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                return Err("confidence must be finite and in [0, 1]".into());
            }
            let mut object = Map::new();
            object.insert(
                "schema_version".to_string(),
                Value::from(scanstitch::color_calibration::CALIBRATION_LIBRARY_SCHEMA_VERSION),
            );
            object.insert(
                "record_type".to_string(),
                Value::String("roll_profile".to_string()),
            );
            object.insert("profile_id".to_string(), Value::String(profile_id.clone()));
            if let Some(scanner_profile) = scanner_profile {
                object.insert(
                    "scanner_profile_id".to_string(),
                    Value::String(scanner_profile),
                );
            }
            object.insert(
                "film".to_string(),
                serde_json::json!({
                    "stock": film_stock,
                    "process": process,
                }),
            );
            object.insert("base_color".to_string(), serde_json::json!(base_color));
            object.insert("confidence".to_string(), serde_json::json!(confidence));
            write_record(
                &library,
                "rolls",
                &profile_id,
                &Value::Object(object),
                force,
            )?;
        }
    }
    Ok(())
}

fn sample_target(
    manifest_path: &Path,
    output_path: &Path,
    preview_dir: &Path,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(manifest_path)?;
    let manifest =
        serde_json::from_str::<scanstitch::calibration_target::TargetSamplingManifest>(&contents)
            .map_err(|error| {
            format!(
                "failed to decode target sampling manifest {}: {error}",
                manifest_path.display()
            )
        })?;
    let manifest_directory = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let result =
        scanstitch::calibration_target::sample_target_manifest(&manifest, manifest_directory)?;
    let report_path = preview_dir.join("sampling-report.json");
    let corner_review_manifest_path = preview_dir.join("corner-review-manifest.json");
    let preview_paths = result
        .previews
        .iter()
        .map(|preview| {
            preview_dir.join(format!(
                "{}.png",
                scanstitch::calibration_target::safe_capture_filename(&preview.capture_id)
            ))
        })
        .collect::<Vec<_>>();
    let mut intended_outputs = vec![
        output_path.to_path_buf(),
        report_path.clone(),
        corner_review_manifest_path.clone(),
    ];
    intended_outputs.extend(preview_paths.iter().cloned());
    if !force {
        if let Some(existing) = intended_outputs.iter().find(|path| path.exists()) {
            return Err(format!(
                "{} already exists; rerun with --force to replace sampling outputs",
                existing.display()
            )
            .into());
        }
    }
    std::fs::create_dir_all(preview_dir)?;
    let report_contents = serde_json::to_string_pretty(&result.diagnostics)?;
    write_output_file(&report_path, report_contents.as_bytes(), force)?;
    let corner_review_manifest_contents =
        serde_json::to_string_pretty(&result.corner_review_manifest)?;
    write_output_file(
        &corner_review_manifest_path,
        corner_review_manifest_contents.as_bytes(),
        force,
    )?;
    for (preview, path) in result.previews.iter().zip(&preview_paths) {
        preview
            .image
            .save_with_format(path, image::ImageFormat::Png)
            .map_err(|error| format!("failed to write preview {}: {error}", path.display()))?;
    }
    if result.diagnostics.status != "accepted" {
        if force && output_path.exists() {
            std::fs::remove_file(output_path)?;
        }
        if result.diagnostics.status.contains("approval") {
            return Err(format!(
                "target sampling status {} requires explicit approval of the exact reference patch values and every capture's chart corners against the generated overlays and matching hashes; no measurement JSON was written (review manifest: {}; audit: {})",
                result.diagnostics.status,
                corner_review_manifest_path.display(),
                report_path.display()
            )
            .into());
        }
        return Err(format!(
            "target sampling rejected {} patch measurements; no measurement JSON was written (audit: {})",
            result.diagnostics.rejected_patch_count,
            report_path.display()
        )
        .into());
    }
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let measurement_contents = serde_json::to_string_pretty(&result.measurement)?;
    write_output_file(output_path, measurement_contents.as_bytes(), force)?;
    println!("{}", output_path.display());
    println!("{}", report_path.display());
    println!("{}", corner_review_manifest_path.display());
    for path in preview_paths {
        println!("{}", path.display());
    }
    Ok(())
}

fn write_output_file(
    path: &Path,
    contents: &[u8],
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    if force {
        atomic_file::write_bytes(path, contents)?;
    } else {
        atomic_file::write_bytes_noclobber(path, contents)?;
    }
    Ok(())
}

fn read_object_json(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)?;
    let value = serde_json::from_str::<Value>(&contents)?;
    if !value.is_object() {
        return Err(format!("{} must contain a JSON object", path.display()).into());
    }
    Ok(value)
}

fn fit_scanner_target_if_needed(
    profile_id: &str,
    value: &mut Value,
) -> Result<(), Box<dyn std::error::Error>> {
    if value.get("scanner_linearization").is_none() {
        if let Some(fit_value) = value.get("scanner_linearization_fit").cloned() {
            let fit = serde_json::from_value::<ScannerLinearizationFitMeasurements>(fit_value)
                .map_err(|err| format!("failed to decode scanner_linearization_fit: {err}"))?;
            let linearization = fit_scanner_linearization(profile_id, &fit)?;
            let object = value.as_object_mut().expect("object checked");
            object.insert(
                "scanner_linearization".to_string(),
                serde_json::to_value(linearization)?,
            );
            object.remove("scanner_linearization_fit");
        }
    }
    if has_any_field(value, &["scanner_rgb_to_xyz", "work_to_xyz", "fit_matrix"]) {
        return Ok(());
    }

    let (mut training_patches, mut held_out_patches) =
        scanstitch::color_calibration::disjoint_target_patch_sets_from_measurement(value)?;
    let scanner_linearization = value
        .get("scanner_linearization")
        .cloned()
        .map(
            serde_json::from_value::<
                scanstitch::scanner_linearization::ScannerLinearizationCalibration,
            >,
        )
        .transpose()
        .map_err(|err| format!("failed to decode scanner_linearization: {err}"))?;
    let linearization_diagnostics = linearize_target_patch_sets(
        scanner_linearization.as_ref(),
        &mut training_patches,
        &mut held_out_patches,
    )?;
    let fit = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training_patches,
        &held_out_patches,
        "least_squares_rgb_to_xyz",
        optional_triplet(value.get("whitepoint"), "whitepoint")?,
        optional_f64(value.get("confidence"), "confidence")?,
    )?;
    let polynomial_selection =
        scanstitch::color_calibration::fit_root_polynomial_color_model_from_disjoint_patches(
            profile_id,
            &training_patches,
            &held_out_patches,
            &fit.matrix,
        )?;
    let lut_selection =
        scanstitch::color_calibration::fit_residual_lut_3d_color_model_from_disjoint_patches(
            profile_id,
            &training_patches,
            &held_out_patches,
            &fit.matrix,
            polynomial_selection.selected_model.as_ref(),
        )?;
    let object = value.as_object_mut().expect("object checked");
    object.insert(
        "scanner_rgb_to_xyz".to_string(),
        serde_json::json!(fit.matrix),
    );
    object.insert("whitepoint".to_string(), serde_json::json!(fit.whitepoint));
    object.insert("confidence".to_string(), serde_json::json!(fit.confidence));
    object.insert("fit".to_string(), serde_json::to_value(&fit.fit)?);
    object.insert(
        "polynomial_fit".to_string(),
        serde_json::to_value(&polynomial_selection.diagnostics)?,
    );
    object.insert(
        "lut_3d_fit".to_string(),
        serde_json::to_value(&lut_selection.diagnostics)?,
    );
    if let Some(model) = polynomial_selection.selected_model.as_ref() {
        object.insert(
            "delta_e00_summary".to_string(),
            serde_json::json!({
                "source": "selected_root_polynomial_held_out",
                "model_id": model.model_id.clone(),
                "rms": model.validation.held_out_delta_e00_rms,
                "max": model.validation.held_out_delta_e00_max,
                "matrix_rms": model.validation.matrix_held_out_delta_e00_rms,
                "matrix_max": model.validation.matrix_held_out_delta_e00_max,
            }),
        );
        object.insert("color_model".to_string(), serde_json::to_value(model)?);
    }
    if let Some(model) = lut_selection.selected_model.as_ref() {
        object.insert(
            "delta_e00_summary".to_string(),
            serde_json::json!({
                "source": "selected_residual_lut_3d_held_out",
                "model_id": model.model_id.clone(),
                "rms": model.validation.held_out_delta_e00_rms,
                "max": model.validation.held_out_delta_e00_max,
                "baseline_rms": model.validation.baseline_held_out_delta_e00_rms,
                "baseline_max": model.validation.baseline_held_out_delta_e00_max,
            }),
        );
        object.insert("lut_3d_model".to_string(), serde_json::to_value(model)?);
    }
    object.insert(
        "patches".to_string(),
        serde_json::to_value(&held_out_patches)?,
    );
    object.remove("training_patches");
    object.remove("held_out_patches");
    stamp_target_patch_signal_domain(
        object,
        scanner_linearization.as_ref(),
        linearization_diagnostics,
    );
    if let Some(fingerprint) =
        scanstitch::color_calibration::scanner_settings_fingerprint(object.get("settings"))
    {
        object.insert(
            "scanner_settings_fingerprint".to_string(),
            Value::String(fingerprint),
        );
    }
    Ok(())
}

fn scanner_shading_gain(coefficients: &[f64; 6], x: f64, y: f64) -> f64 {
    coefficients[0]
        + coefficients[1] * x
        + coefficients[2] * y
        + coefficients[3] * x * x
        + coefficients[4] * x * y
        + coefficients[5] * y * y
}

fn linearize_target_patch_sets(
    calibration: Option<&scanstitch::scanner_linearization::ScannerLinearizationCalibration>,
    training_patches: &mut [scanstitch::color_calibration::TargetPatch],
    held_out_patches: &mut [scanstitch::color_calibration::TargetPatch],
) -> Result<Option<Value>, Box<dyn std::error::Error>> {
    let Some(calibration) = calibration else {
        return Ok(None);
    };
    scanstitch::scanner_linearization::validate_scanner_linearization(calibration).map_err(
        |errors| {
            format!(
                "target patches cannot use invalid scanner_linearization: {}",
                errors.join("; ")
            )
        },
    )?;
    let spatial_coordinates_required = calibration.shading_gain_polynomial.is_some();
    let mut input_min = [f64::INFINITY; 3];
    let mut input_max = [f64::NEG_INFINITY; 3];
    let mut output_min = [f64::INFINITY; 3];
    let mut output_max = [f64::NEG_INFINITY; 3];
    let mut maximum_absolute_channel_change = 0.0f64;
    let patch_count = training_patches.len() + held_out_patches.len();
    for (set_name, patches) in [
        ("training_patches", training_patches),
        ("held_out_patches", held_out_patches),
    ] {
        for (index, patch) in patches.iter_mut().enumerate() {
            let [x, y] = match patch.scanner_xy {
                Some(coordinates)
                    if coordinates.iter().all(|coordinate| {
                        coordinate.is_finite() && (-1.0..=1.0).contains(coordinate)
                    }) =>
                {
                    coordinates
                }
                Some(_) => {
                    return Err(format!(
                        "{set_name}[{index}].scanner_xy must contain normalized scanner-frame coordinates in [-1,1]"
                    )
                    .into());
                }
                None if spatial_coordinates_required => {
                    return Err(format!(
                        "{set_name}[{index}] requires scanner_xy because scanner_linearization contains a spatial shading model"
                    )
                    .into());
                }
                None => [0.0, 0.0],
            };
            let input = patch.source_rgb;
            let output = scanstitch::scanner_linearization::evaluate_scanner_signal(
                calibration,
                input,
                x,
                y,
            )?;
            if output
                .iter()
                .any(|channel| !channel.is_finite() || !(0.0..=1.0).contains(channel))
            {
                return Err(format!(
                    "{set_name}[{index}] scanner linearization produced out-of-range RGB {:?}; clipped target samples cannot establish a colour matrix",
                    output
                )
                .into());
            }
            for channel in 0..3 {
                input_min[channel] = input_min[channel].min(input[channel]);
                input_max[channel] = input_max[channel].max(input[channel]);
                output_min[channel] = output_min[channel].min(output[channel]);
                output_max[channel] = output_max[channel].max(output[channel]);
                maximum_absolute_channel_change =
                    maximum_absolute_channel_change.max((output[channel] - input[channel]).abs());
            }
            patch.source_rgb = output;
        }
    }
    Ok(Some(serde_json::json!({
        "status": "applied_before_matrix_fit",
        "model_id": calibration.model_id,
        "spatial_coordinates_required": spatial_coordinates_required,
        "coordinate_domain": "normalized_original_scanner_frame",
        "patch_count": patch_count,
        "input_min": input_min,
        "input_max": input_max,
        "output_min": output_min,
        "output_max": output_max,
        "maximum_absolute_channel_change": maximum_absolute_channel_change,
    })))
}

fn stamp_target_patch_signal_domain(
    object: &mut Map<String, Value>,
    calibration: Option<&scanstitch::scanner_linearization::ScannerLinearizationCalibration>,
    diagnostics: Option<Value>,
) {
    let domain = if calibration.is_some() {
        "scanner_linearized_transmittance"
    } else {
        "normalized_scanner_signal"
    };
    object.insert(
        "target_patch_signal_domain".to_string(),
        Value::String(domain.to_string()),
    );
    if let Some(calibration) = calibration {
        object.insert(
            "target_patch_linearization_model_id".to_string(),
            Value::String(calibration.model_id.clone()),
        );
    }
    if let Some(diagnostics) = diagnostics {
        object.insert(
            "target_patch_linearization_application".to_string(),
            diagnostics,
        );
    }
}

fn fit_scanner_monotone_curve(
    measurements: &ScannerLinearizationFitMeasurements,
    channel: usize,
) -> Result<Vec<[f64; 2]>, String> {
    let mut pairs = measurements
        .training_samples
        .iter()
        .map(|sample| {
            let level = (sample.scanner_signal[channel]
                - measurements.black_level_normalized[channel]
                - measurements.additive_flare_normalized[channel])
                / (measurements.white_level_normalized[channel]
                    - measurements.black_level_normalized[channel]);
            let gain = measurements
                .shading_gain_polynomial
                .as_ref()
                .map(|coefficients| {
                    scanner_shading_gain(&coefficients[channel], sample.x, sample.y)
                })
                .unwrap_or(1.0);
            (level, sample.reference_transmittance[channel] / gain)
        })
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut grouped = Vec::<(f64, f64, f64)>::new();
    let mut index = 0usize;
    while index < pairs.len() {
        let start = index;
        let reference_x = pairs[index].0;
        while index < pairs.len() && (pairs[index].0 - reference_x).abs() <= 1e-9 {
            index += 1;
        }
        let mut xs = pairs[start..index]
            .iter()
            .map(|pair| pair.0)
            .collect::<Vec<_>>();
        let mut ys = pairs[start..index]
            .iter()
            .map(|pair| pair.1)
            .collect::<Vec<_>>();
        grouped.push((
            median_sorted(&mut xs),
            median_sorted(&mut ys),
            (index - start) as f64,
        ));
    }
    let fitted = isotonic_non_decreasing(
        &grouped.iter().map(|point| point.1).collect::<Vec<_>>(),
        &grouped.iter().map(|point| point.2).collect::<Vec<_>>(),
    );
    let mut strict = Vec::<[f64; 2]>::new();
    for (point, fitted_y) in grouped.iter().zip(fitted) {
        let point = [point.0, fitted_y];
        if strict
            .last()
            .is_none_or(|previous| point[0] > previous[0] + 1e-8 && point[1] > previous[1] + 1e-8)
        {
            strict.push(point);
        }
    }
    if strict.len() > 12 {
        let last = strict.len() - 1;
        strict = (0..12)
            .map(|sample| strict[(sample * last + 5) / 11])
            .collect();
        strict.dedup_by(|left, right| left == right);
    }
    if strict.len() < 4 {
        return Err(format!(
            "scanner channel {} produced only {} strictly increasing curve knots; at least 4 are required",
            ["red", "green", "blue"][channel],
            strict.len()
        ));
    }
    Ok(strict)
}

fn fit_scanner_linearization(
    profile_id: &str,
    measurements: &ScannerLinearizationFitMeasurements,
) -> Result<
    scanstitch::scanner_linearization::ScannerLinearizationCalibration,
    Box<dyn std::error::Error>,
> {
    if measurements.training_samples.len() < 12 {
        return Err("scanner_linearization_fit requires at least 12 training samples".into());
    }
    if measurements.held_out_samples.len() < 12 {
        return Err(
            "scanner_linearization_fit requires at least 12 disjoint held-out samples".into(),
        );
    }
    if !measurements.confidence.is_finite() || !(0.85..=1.0).contains(&measurements.confidence) {
        return Err(
            "scanner_linearization_fit confidence must be finite and in [0.85, 1.0]".into(),
        );
    }
    for (label, samples) in [
        ("training_samples", &measurements.training_samples),
        ("held_out_samples", &measurements.held_out_samples),
    ] {
        for (index, sample) in samples.iter().enumerate() {
            validate_finite_triplet(
                &sample.scanner_signal,
                &format!("{label}[{index}].scanner_signal"),
            )?;
            validate_finite_triplet(
                &sample.reference_transmittance,
                &format!("{label}[{index}].reference_transmittance"),
            )?;
            if sample
                .scanner_signal
                .iter()
                .chain(&sample.reference_transmittance)
                .any(|value| !(0.0..=1.0).contains(value))
                || !(-1.0..=1.0).contains(&sample.x)
                || !(-1.0..=1.0).contains(&sample.y)
            {
                return Err(format!(
                    "{label}[{index}] signal/transmittance must be in [0,1] and x/y in [-1,1]"
                )
                .into());
            }
        }
    }
    let curves = scanstitch::scanner_linearization::ScannerLinearizationCurves {
        red: fit_scanner_monotone_curve(measurements, 0)?,
        green: fit_scanner_monotone_curve(measurements, 1)?,
        blue: fit_scanner_monotone_curve(measurements, 2)?,
    };
    let mut calibration = scanstitch::scanner_linearization::ScannerLinearizationCalibration {
        model_id: measurements
            .model_id
            .clone()
            .unwrap_or_else(|| format!("{profile_id}-linearization")),
        curves,
        black_level_normalized: measurements.black_level_normalized,
        white_level_normalized: measurements.white_level_normalized,
        additive_flare_normalized: measurements.additive_flare_normalized,
        shading_gain_polynomial: measurements.shading_gain_polynomial,
        confidence: measurements.confidence,
        validation: scanstitch::scanner_linearization::ScannerLinearizationValidation {
            held_out_sample_count: measurements.held_out_samples.len(),
            transmittance_rmse: 0.0,
            transmittance_max_error: 0.0,
            identity_baseline_rmse: 0.0,
        },
    };
    let mut error_sum_sq = 0.0;
    let mut maximum_error = 0.0f64;
    let mut identity_sum_sq = 0.0;
    for sample in &measurements.held_out_samples {
        let predicted = scanstitch::scanner_linearization::evaluate_scanner_signal(
            &calibration,
            sample.scanner_signal,
            sample.x,
            sample.y,
        )?;
        for (channel, predicted_channel) in predicted.iter().enumerate() {
            let error = (*predicted_channel - sample.reference_transmittance[channel]).abs();
            let identity_error =
                (sample.scanner_signal[channel] - sample.reference_transmittance[channel]).abs();
            error_sum_sq += error * error;
            maximum_error = maximum_error.max(error);
            identity_sum_sq += identity_error * identity_error;
        }
    }
    let count = (measurements.held_out_samples.len() * 3) as f64;
    calibration.validation.transmittance_rmse = (error_sum_sq / count).sqrt();
    calibration.validation.transmittance_max_error = maximum_error;
    calibration.validation.identity_baseline_rmse = (identity_sum_sq / count).sqrt();
    scanstitch::scanner_linearization::validate_scanner_linearization(&calibration).map_err(
        |errors| {
            format!(
                "fitted scanner linearization failed acceptance: {}",
                errors.join("; ")
            )
        },
    )?;
    Ok(calibration)
}

fn fit_roll_target_if_needed(
    library: &Path,
    scanner_profile: &str,
    roll_profile: &str,
    value: &mut Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let scanner = scanstitch::color_calibration::load_calibration(
        None,
        Some(library),
        Some(scanner_profile),
        None,
        None,
        None,
    );
    let scanner_reason = scanner.diagnostics.reason.clone();
    let scanner_fingerprint = scanner
        .diagnostics
        .scanner_profile
        .as_ref()
        .and_then(|scanner| scanner.scanner_settings_fingerprint.clone());
    let profile = scanner.profile.ok_or_else(|| {
        format!(
            "cannot prepare roll correction because scanner profile `{scanner_profile}` was not applied: {}",
            scanner_reason
        )
    })?;

    if value.get("negative_response").is_none() {
        if let Some(fit_value) = value.get("negative_response_fit").cloned() {
            let fit = serde_json::from_value::<NegativeResponseFitMeasurements>(fit_value)
                .map_err(|err| format!("failed to decode negative_response_fit: {err}"))?;
            let response = fit_measured_negative_response(roll_profile, &fit)?;
            value.as_object_mut().expect("object checked").insert(
                "negative_response".to_string(),
                serde_json::to_value(response)?,
            );
            value
                .as_object_mut()
                .expect("object checked")
                .remove("negative_response_fit");
        }
    }

    if has_any_field(value, &["correction_matrix", "lut", "negative_response"]) {
        stamp_roll_scanner_context(value, scanner_fingerprint.as_deref())?;
        return Ok(());
    }

    let (mut training_patches, mut held_out_patches) =
        scanstitch::color_calibration::disjoint_target_patch_sets_from_measurement(value)?;
    let linearization_diagnostics = linearize_target_patch_sets(
        profile.scanner_linearization.as_ref(),
        &mut training_patches,
        &mut held_out_patches,
    )?;
    let retained_held_out_patches = held_out_patches.clone();
    const XYZ_NUMERICAL_ZERO_TOLERANCE: f64 = 1e-12;
    let mut numerical_zero_clamp_count = 0usize;
    for patch in training_patches
        .iter_mut()
        .chain(held_out_patches.iter_mut())
    {
        if let Some(model) = profile.lut_3d_model.as_ref() {
            if !scanstitch::color_calibration::residual_lut_3d_has_full_support(
                model,
                patch.source_rgb,
            ) {
                return Err(format!(
                    "roll target patch {:?} is outside full support of selected scanner residual LUT `{}`; acquire matched target coverage or use a scanner profile whose simpler baseline was selected",
                    patch.patch_id, model.model_id
                )
                .into());
            }
        }
        let source = profile.calibrated_source_to_xyz(patch.source_rgb);
        if source
            .iter()
            .any(|value| !value.is_finite() || *value < -XYZ_NUMERICAL_ZERO_TOLERANCE)
        {
            return Err(format!(
                "selected scanner calibration transform produced invalid XYZ {:?} for roll target patch {:?}",
                source, patch.patch_id
            )
            .into());
        }
        patch.source_rgb = source.map(|value| {
            if value < 0.0 {
                numerical_zero_clamp_count += 1;
                0.0
            } else {
                value
            }
        });
    }

    let fit = scanstitch::color_calibration::fit_rgb_to_xyz_from_disjoint_patches(
        &training_patches,
        &held_out_patches,
        "least_squares_xyz_post_scanner_correction",
        Some(profile.whitepoint),
        optional_f64(value.get("confidence"), "confidence")?,
    )?;
    let object = value.as_object_mut().expect("object checked");
    object.insert(
        "correction_matrix".to_string(),
        serde_json::json!(fit.matrix),
    );
    object.insert(
        "correction_domain".to_string(),
        Value::String("xyz_post_scanner".to_string()),
    );
    object.insert("confidence".to_string(), serde_json::json!(fit.confidence));
    object.insert("fit".to_string(), serde_json::to_value(&fit.fit)?);
    object.insert(
        "scanner_color_model_application".to_string(),
        serde_json::to_value(
            scanstitch::color_calibration::ScannerColorModelApplicationDiagnostics {
                transform: profile.lut_3d_model.as_ref().map_or_else(
                    || {
                        profile.color_model.as_ref().map_or_else(
                            || "matrix".to_string(),
                            |model| format!("root_polynomial_degree_{}", model.degree),
                        )
                    },
                    |model| format!("residual_lut_3d_grid_{}", model.grid_size),
                ),
                model_id: profile
                    .lut_3d_model
                    .as_ref()
                    .map(|model| model.model_id.clone())
                    .or_else(|| {
                        profile
                            .color_model
                            .as_ref()
                            .map(|model| model.model_id.clone())
                    }),
                output_domain: "reference_xyz_d50".to_string(),
                numerical_zero_tolerance: XYZ_NUMERICAL_ZERO_TOLERANCE,
                numerical_zero_clamp_count,
            },
        )?,
    );
    object.insert(
        "patches".to_string(),
        serde_json::to_value(&retained_held_out_patches)?,
    );
    object.remove("training_patches");
    object.remove("held_out_patches");
    stamp_target_patch_signal_domain(
        object,
        profile.scanner_linearization.as_ref(),
        linearization_diagnostics,
    );
    stamp_roll_scanner_context(value, scanner_fingerprint.as_deref())?;
    Ok(())
}

fn validate_finite_triplet(values: &[f64; 3], label: &str) -> Result<(), String> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(format!("{label} must contain three finite values"))
    }
}

fn fit_scanner_density_to_layer_matrix(
    patches: &[NegativeResponseTrainingPatch],
) -> Result<[[f64; 3]; 3], String> {
    let rows = patches.len();
    let scanner =
        DMatrix::<f64>::from_fn(rows, 3, |row, column| patches[row].scanner_density[column]);
    let layer = DMatrix::<f64>::from_fn(rows, 3, |row, column| {
        patches[row].reference_layer_density[column]
    });
    let regularization = 1e-8;
    let normal = scanner.transpose() * &scanner + DMatrix::<f64>::identity(3, 3) * regularization;
    let right = scanner.transpose() * layer;
    let coefficients = normal.lu().solve(&right).ok_or_else(|| {
        "negative-response density-separation least-squares solve was singular".to_string()
    })?;
    let rows = std::array::from_fn(|layer_channel| {
        std::array::from_fn(|scanner_channel| coefficients[(scanner_channel, layer_channel)])
    });
    Ok(rows)
}

fn median_sorted(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}

fn isotonic_non_decreasing(values: &[f64], weights: &[f64]) -> Vec<f64> {
    #[derive(Clone, Copy)]
    struct Block {
        start: usize,
        end: usize,
        weighted_sum: f64,
        weight: f64,
    }
    let mut blocks = Vec::<Block>::new();
    for (index, (&value, &weight)) in values.iter().zip(weights).enumerate() {
        blocks.push(Block {
            start: index,
            end: index + 1,
            weighted_sum: value * weight,
            weight,
        });
        while blocks.len() >= 2 {
            let right = blocks[blocks.len() - 1];
            let left = blocks[blocks.len() - 2];
            if left.weighted_sum / left.weight < right.weighted_sum / right.weight {
                break;
            }
            blocks.truncate(blocks.len() - 2);
            blocks.push(Block {
                start: left.start,
                end: right.end,
                weighted_sum: left.weighted_sum + right.weighted_sum,
                weight: left.weight + right.weight,
            });
        }
    }
    let mut fitted = vec![0.0; values.len()];
    for block in blocks {
        let value = block.weighted_sum / block.weight;
        fitted[block.start..block.end].fill(value);
    }
    fitted
}

fn fit_monotone_characteristic_curve(
    patches: &[NegativeResponseTrainingPatch],
    channel: usize,
) -> Result<Vec<[f64; 2]>, String> {
    let mut pairs = patches
        .iter()
        .map(|patch| {
            (
                patch.reference_layer_density[channel],
                patch.reference_log_exposure[channel],
            )
        })
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    let bin_count = ((pairs.len() as f64).sqrt().round() as usize).clamp(4, 12);
    let mut binned_x = Vec::with_capacity(bin_count);
    let mut binned_y = Vec::with_capacity(bin_count);
    let mut weights = Vec::with_capacity(bin_count);
    for bin in 0..bin_count {
        let start = bin * pairs.len() / bin_count;
        let end = ((bin + 1) * pairs.len() / bin_count).max(start + 1);
        let slice = &pairs[start..end.min(pairs.len())];
        let mut x = slice.iter().map(|pair| pair.0).collect::<Vec<_>>();
        let mut y = slice.iter().map(|pair| pair.1).collect::<Vec<_>>();
        binned_x.push(median_sorted(&mut x));
        binned_y.push(median_sorted(&mut y));
        weights.push(slice.len() as f64);
    }
    let fitted_y = isotonic_non_decreasing(&binned_y, &weights);
    let mut curve = Vec::<[f64; 2]>::new();
    for (&x, &y) in binned_x.iter().zip(&fitted_y) {
        let strictly_beyond_previous = curve
            .last()
            .is_none_or(|previous| x > previous[0] + 1e-8 && y > previous[1] + 1e-8);
        if strictly_beyond_previous {
            curve.push([x, y]);
        }
    }
    if curve.len() < 4 {
        return Err(format!(
            "channel {} produced only {} strictly increasing response knots after robust binning/isotonic regression; at least 4 are required",
            ["red", "green", "blue"][channel],
            curve.len()
        ));
    }
    Ok(curve)
}

fn log_exposure_to_xyz(log_exposure: [f64; 3], scene_rgb_to_xyz: &Matrix3<f64>) -> [f64; 3] {
    let rgb = Vector3::new(
        10.0f64.powf(log_exposure[0].clamp(-12.0, 12.0)),
        10.0f64.powf(log_exposure[1].clamp(-12.0, 12.0)),
        10.0f64.powf(log_exposure[2].clamp(-12.0, 12.0)),
    );
    let xyz = scene_rgb_to_xyz * rgb;
    [xyz[0], xyz[1], xyz[2]]
}

fn delta_e00_from_log_exposure(
    predicted: [f64; 3],
    reference: [f64; 3],
    scene_rgb_to_xyz: &Matrix3<f64>,
) -> f64 {
    let predicted_lab =
        scanstitch::colorspace::xyz_d50_to_lab(log_exposure_to_xyz(predicted, scene_rgb_to_xyz));
    let reference_lab =
        scanstitch::colorspace::xyz_d50_to_lab(log_exposure_to_xyz(reference, scene_rgb_to_xyz));
    scanstitch::colorspace::lab_delta_e2000(predicted_lab, reference_lab)
}

fn fit_measured_negative_response(
    roll_profile: &str,
    measurements: &NegativeResponseFitMeasurements,
) -> Result<scanstitch::density::MeasuredNegativeResponseCalibration, Box<dyn std::error::Error>> {
    if measurements.training_patches.len() < 12 {
        return Err("negative_response_fit requires at least 12 training patches".into());
    }
    if measurements.held_out_patches.len() < 12 {
        return Err("negative_response_fit requires at least 12 held-out patches that were not used for fitting".into());
    }
    if !measurements.confidence.is_finite() || !(0.75..=1.0).contains(&measurements.confidence) {
        return Err("negative_response_fit confidence must be finite and in [0.75, 1.0]".into());
    }
    for (index, patch) in measurements.training_patches.iter().enumerate() {
        validate_finite_triplet(
            &patch.scanner_density,
            &format!("training_patches[{index}].scanner_density"),
        )?;
        validate_finite_triplet(
            &patch.reference_layer_density,
            &format!("training_patches[{index}].reference_layer_density"),
        )?;
        validate_finite_triplet(
            &patch.reference_log_exposure,
            &format!("training_patches[{index}].reference_log_exposure"),
        )?;
    }
    for (index, patch) in measurements.held_out_patches.iter().enumerate() {
        validate_finite_triplet(
            &patch.scanner_density,
            &format!("held_out_patches[{index}].scanner_density"),
        )?;
        validate_finite_triplet(
            &patch.reference_log_exposure,
            &format!("held_out_patches[{index}].reference_log_exposure"),
        )?;
    }
    if measurements
        .scene_rgb_to_xyz_d50
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err("negative_response_fit.scene_rgb_to_xyz_d50 must contain finite values".into());
    }

    let matrix = fit_scanner_density_to_layer_matrix(&measurements.training_patches)?;
    let curves = scanstitch::density::NegativeCharacteristicCurves {
        red: fit_monotone_characteristic_curve(&measurements.training_patches, 0)?,
        green: fit_monotone_characteristic_curve(&measurements.training_patches, 1)?,
        blue: fit_monotone_characteristic_curve(&measurements.training_patches, 2)?,
    };
    let mut response = scanstitch::density::MeasuredNegativeResponseCalibration {
        model_id: measurements
            .model_id
            .clone()
            .unwrap_or_else(|| format!("{roll_profile}-negative-response")),
        scanner_density_to_layer_density: matrix,
        characteristic_curves: curves,
        white_anchor_percentile: measurements.white_anchor_percentile,
        confidence: measurements.confidence,
        validation: scanstitch::density::NegativeResponseValidation {
            held_out_patch_count: measurements.held_out_patches.len(),
            delta_e00_rms: 0.0,
            delta_e00_max: 0.0,
            unit_slope_delta_e00_rms: 0.0,
            worst_hue_family: None,
        },
    };
    let scene_rgb_to_xyz = rows_to_matrix3(&measurements.scene_rgb_to_xyz_d50);
    let mut model_sum_sq = 0.0;
    let mut model_max = 0.0f64;
    let mut unit_sum_sq = 0.0;
    for patch in &measurements.held_out_patches {
        let predicted = scanstitch::density::evaluate_measured_negative_response_log_exposure(
            &response,
            patch.scanner_density,
        )?;
        let model_delta =
            delta_e00_from_log_exposure(predicted, patch.reference_log_exposure, &scene_rgb_to_xyz);
        let unit_delta = delta_e00_from_log_exposure(
            patch.scanner_density,
            patch.reference_log_exposure,
            &scene_rgb_to_xyz,
        );
        model_sum_sq += model_delta * model_delta;
        model_max = model_max.max(model_delta);
        unit_sum_sq += unit_delta * unit_delta;
    }
    let count = measurements.held_out_patches.len() as f64;
    response.validation.delta_e00_rms = (model_sum_sq / count).sqrt();
    response.validation.delta_e00_max = model_max;
    response.validation.unit_slope_delta_e00_rms = (unit_sum_sq / count).sqrt();
    scanstitch::density::validate_measured_negative_response(&response).map_err(|errors| {
        format!(
            "fitted negative response failed acceptance: {}",
            errors.join("; ")
        )
    })?;
    Ok(response)
}

fn stamp_roll_scanner_context(
    value: &mut Value,
    scanner_fingerprint: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let has_correction_matrix = value.get("correction_matrix").is_some();
    let has_scanner_specific_model =
        has_correction_matrix || value.get("negative_response").is_some();
    let object = value.as_object_mut().expect("object checked");
    if has_correction_matrix {
        object
            .entry("correction_domain")
            .or_insert(Value::String("xyz_post_scanner".to_string()));
    }

    let existing = object
        .get("scanner_settings_fingerprint")
        .and_then(Value::as_str)
        .map(str::to_string);
    match (
        existing.as_deref(),
        scanner_fingerprint,
        has_scanner_specific_model,
    ) {
        (Some(existing), Some(expected), _) if existing != expected => Err(format!(
            "roll target scanner_settings_fingerprint `{existing}` does not match selected scanner fingerprint `{expected}`"
        )
        .into()),
        (Some(_), _, _) => Ok(()),
        (None, Some(fingerprint), _) => {
            object.insert(
                "scanner_settings_fingerprint".to_string(),
                Value::String(fingerprint.to_string()),
            );
            Ok(())
        }
        (None, None, true) => Err("roll correction or negative-response reconstruction requires the selected scanner profile to have a settings fingerprint".into()),
        (None, None, false) => Ok(()),
    }
}

fn has_any_field(value: &Value, fields: &[&str]) -> bool {
    fields.iter().any(|field| value.get(*field).is_some())
}

fn set_common_fields(value: &mut Value, record_type: &str, profile_id: &str) {
    let object = value.as_object_mut().expect("object checked");
    object.entry("schema_version").or_insert(Value::from(
        scanstitch::color_calibration::CALIBRATION_LIBRARY_SCHEMA_VERSION,
    ));
    object.insert(
        "record_type".to_string(),
        Value::String(record_type.to_string()),
    );
    object.insert(
        "profile_id".to_string(),
        Value::String(profile_id.to_string()),
    );
}

fn require_any_matrix(
    value: &Value,
    fields: &[&str],
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if has_any_field(value, fields) {
        Ok(())
    } else {
        Err(format!("{label} must include one of: {}", fields.join(", ")).into())
    }
}

fn optional_f64(
    value: Option<&Value>,
    label: &str,
) -> Result<Option<f64>, Box<dyn std::error::Error>> {
    value
        .map(|value| {
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("{label} must be a finite number").into())
        })
        .transpose()
}

fn optional_triplet(
    value: Option<&Value>,
    label: &str,
) -> Result<Option<[f64; 3]>, Box<dyn std::error::Error>> {
    value
        .map(|value| {
            let values = value
                .as_array()
                .ok_or_else(|| format!("{label} must be an array of three numbers"))?;
            if values.len() != 3 {
                return Err(format!("{label} must contain exactly three numbers").into());
            }
            let mut out = [0.0f64; 3];
            for (idx, value) in values.iter().enumerate() {
                out[idx] = value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| format!("{label}[{idx}] must be a finite number"))?;
            }
            Ok(out)
        })
        .transpose()
}

fn rows_to_matrix3(rows: &[[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::new(
        rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0],
        rows[2][1], rows[2][2],
    )
}

fn parse_rgb_triplet(value: &str) -> Result<[f64; 3], Box<dyn std::error::Error>> {
    let parts = value
        .split(',')
        .map(str::trim)
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()?;
    if parts.len() != 3 {
        return Err("base_color must contain exactly three comma-separated numbers".into());
    }
    if parts
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("base_color values must be finite and positive".into());
    }
    Ok([parts[0], parts[1], parts[2]])
}

fn write_record(
    library: &Path,
    subdir: &str,
    profile_id: &str,
    value: &Value,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = library.join(subdir);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.json", safe_filename(profile_id)));
    scanstitch::color_calibration::validate_library_record_value(value, Some(&path)).map_err(
        |rejection| {
            format!(
                "{} did not pass calibration library validation: {}",
                path.display(),
                rejection.reasons.join("; ")
            )
        },
    )?;
    let contents = serde_json::to_string_pretty(value)?;
    if force {
        atomic_file::write_bytes(&path, contents.as_bytes())?;
    } else {
        atomic_file::write_bytes_noclobber(&path, contents.as_bytes()).map_err(|err| {
            if err.kind() == std::io::ErrorKind::AlreadyExists {
                format!(
                    "{} already exists; rerun with --force to replace it",
                    path.display()
                )
                .into()
            } else {
                Box::<dyn std::error::Error>::from(err)
            }
        })?;
    }
    println!("{}", path.display());
    Ok(())
}

fn safe_filename(profile_id: &str) -> String {
    profile_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

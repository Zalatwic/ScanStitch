use clap::{Parser, Subcommand};
use nalgebra::{Matrix3, Vector3};
use serde_json::{Map, Value};
use std::io::Write;
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
    /// Ingest a scanner target measurement JSON into calibration/scanners.
    ScannerTarget {
        /// Calibration library directory.
        #[arg(long)]
        library: PathBuf,

        /// Scanner profile ID to write.
        #[arg(long)]
        profile_id: String,

        /// Measurement/profile JSON containing patch RGB/reference values or a pre-fit scanner_rgb_to_xyz matrix.
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

        /// Measurement/profile JSON containing patch RGB/reference values or a pre-fit correction_matrix.
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = CalibrateCli::parse();
    match cli.command {
        CalibrateCommand::ScannerTarget {
            library,
            profile_id,
            measurements,
            force,
        } => {
            let mut value = read_object_json(&measurements)?;
            fit_scanner_target_if_needed(&mut value)?;
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
            fit_roll_target_if_needed(&library, &scanner_profile, &mut value)?;
            require_any_matrix(
                &value,
                &["correction_matrix", "lut"],
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

fn read_object_json(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)?;
    let value = serde_json::from_str::<Value>(&contents)?;
    if !value.is_object() {
        return Err(format!("{} must contain a JSON object", path.display()).into());
    }
    Ok(value)
}

fn fit_scanner_target_if_needed(value: &mut Value) -> Result<(), Box<dyn std::error::Error>> {
    if has_any_field(value, &["scanner_rgb_to_xyz", "work_to_xyz", "fit_matrix"]) {
        return Ok(());
    }

    let patches = scanstitch::color_calibration::target_patches_from_measurement(value)?;
    let fit = scanstitch::color_calibration::fit_rgb_to_xyz_from_patches(
        &patches,
        "least_squares_rgb_to_xyz",
        optional_triplet(value.get("whitepoint"), "whitepoint")?,
        optional_f64(value.get("confidence"), "confidence")?,
    )?;
    let object = value.as_object_mut().expect("object checked");
    object.insert(
        "scanner_rgb_to_xyz".to_string(),
        serde_json::json!(fit.matrix),
    );
    object.insert("whitepoint".to_string(), serde_json::json!(fit.whitepoint));
    object.insert("confidence".to_string(), serde_json::json!(fit.confidence));
    object.insert("fit".to_string(), serde_json::to_value(&fit.fit)?);
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

fn fit_roll_target_if_needed(
    library: &Path,
    scanner_profile: &str,
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

    if has_any_field(value, &["correction_matrix", "lut"]) {
        stamp_roll_scanner_context(value, scanner_fingerprint.as_deref())?;
        return Ok(());
    }

    let scanner_matrix = rows_to_matrix3(&profile.work_to_xyz);
    let mut patches = scanstitch::color_calibration::target_patches_from_measurement(value)?;
    for patch in &mut patches {
        let source = scanner_matrix
            * Vector3::new(
                patch.source_rgb[0],
                patch.source_rgb[1],
                patch.source_rgb[2],
            );
        patch.source_rgb = [source[0].max(0.0), source[1].max(0.0), source[2].max(0.0)];
    }

    let fit = scanstitch::color_calibration::fit_rgb_to_xyz_from_patches(
        &patches,
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
    stamp_roll_scanner_context(value, scanner_fingerprint.as_deref())?;
    Ok(())
}

fn stamp_roll_scanner_context(
    value: &mut Value,
    scanner_fingerprint: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let has_correction_matrix = value.get("correction_matrix").is_some();
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
    match (existing.as_deref(), scanner_fingerprint, has_correction_matrix) {
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
        (None, None, true) => Err(
            "roll correction requires the selected scanner profile to have a settings fingerprint"
                .into(),
        ),
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
        std::fs::write(&path, contents)?;
    } else {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|err| {
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
        file.write_all(contents.as_bytes())?;
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

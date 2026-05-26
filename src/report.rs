use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const REPORT_SCHEMA_VERSION: u32 = 2;
pub const PIPELINE_SCHEMA_VERSION: u32 = 1;

/// Top-level pipeline report, serialized to JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineReport {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<RunMetadata>,
    pub phases: Vec<PhaseReport>,
}

/// Immutable identity for the run that produced a report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunMetadata {
    pub report_schema_version: u32,
    pub pipeline_schema_version: u32,
    pub generated_at: String,
    pub generated_at_unix_ms: u64,
    pub package_name: String,
    pub package_version: String,
    pub binary_name: String,
    pub working_directory: String,
    pub cli_args: Vec<String>,
    pub output_dir: String,
    pub output_path: String,
}

/// Report for a single pipeline phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseReport {
    pub name: String,
    #[serde(default)]
    pub success: bool,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub duration_ms: u64,
    #[serde(default)]
    pub metrics: serde_json::Value,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub errors: Vec<String>,
}

impl PipelineReport {
    /// Create a new empty report.
    pub fn new() -> Self {
        Self {
            metadata: None,
            phases: Vec::new(),
        }
    }

    /// Create a report with run metadata already attached.
    pub fn new_with_metadata(metadata: RunMetadata) -> Self {
        Self {
            metadata: Some(metadata),
            phases: Vec::new(),
        }
    }

    /// Add a phase to the report.
    pub fn add_phase(&mut self, phase: PhaseReport) {
        self.phases.push(phase);
    }

    /// Save the report as JSON to the given path.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl Default for PipelineReport {
    fn default() -> Self {
        Self::new()
    }
}

impl RunMetadata {
    pub fn capture_current(
        output_dir: &Path,
        output_path: &Path,
        generated_at: SystemTime,
    ) -> Self {
        let cli_args = std::env::args().collect::<Vec<_>>();
        let binary_name = cli_args
            .first()
            .and_then(|arg| Path::new(arg).file_stem())
            .and_then(|stem| stem.to_str())
            .unwrap_or(env!("CARGO_PKG_NAME"))
            .to_string();
        let working_directory = std::env::current_dir()
            .map(|path| path.to_string_lossy().to_string())
            .unwrap_or_default();

        Self {
            report_schema_version: REPORT_SCHEMA_VERSION,
            pipeline_schema_version: PIPELINE_SCHEMA_VERSION,
            generated_at: format_system_time_utc(generated_at),
            generated_at_unix_ms: system_time_unix_ms(generated_at).unwrap_or(0),
            package_name: env!("CARGO_PKG_NAME").to_string(),
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            binary_name,
            working_directory,
            cli_args,
            output_dir: absolute_path_string(output_dir),
            output_path: absolute_path_string(output_path),
        }
    }
}

pub fn system_time_unix_ms(time: SystemTime) -> Option<u64> {
    time.duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
}

pub fn format_system_time_utc(time: SystemTime) -> String {
    let Ok(duration) = time.duration_since(UNIX_EPOCH) else {
        return "before_unix_epoch".to_string();
    };
    let total_seconds = duration.as_secs() as i64;
    let days = total_seconds.div_euclid(86_400);
    let seconds_of_day = total_seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn absolute_path_string(path: &Path) -> String {
    if path.is_absolute() {
        return path.to_string_lossy().to_string();
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path).to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string())
}

fn civil_from_days(days_since_unix_epoch: i64) -> (i32, u32, u32) {
    let z = days_since_unix_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    if month <= 2 {
        year += 1;
    }
    (year as i32, month as u32, day as u32)
}

impl PhaseReport {
    /// Convenience constructor for a successful phase with no warnings/errors.
    pub fn ok(name: &str, confidence: f64, metrics: serde_json::Value) -> Self {
        Self {
            name: name.to_string(),
            success: true,
            confidence,
            duration_ms: 0,
            metrics,
            warnings: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Convenience constructor for a failed phase.
    pub fn fail(name: &str, error: &str) -> Self {
        Self {
            name: name.to_string(),
            success: false,
            confidence: 0.0,
            duration_ms: 0,
            metrics: serde_json::Value::Null,
            warnings: Vec::new(),
            errors: vec![error.to_string()],
        }
    }

    /// Annotate a phase report with wall-clock runtime.
    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.duration_ms = duration_ms;
        self
    }
}

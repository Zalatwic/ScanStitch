use crate::atomic_file;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

pub const REPORT_SCHEMA_VERSION: u32 = 4;
pub const PIPELINE_SCHEMA_VERSION: u32 = 1;
pub const FILE_SHA256_BUFFER_BYTES: usize = 1024 * 1024;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_file_size_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_identity_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_identity_error: Option<String>,
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
        atomic_file::write_bytes(path, json.as_bytes())?;
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
        let binary_identity = current_executable_identity();

        Self {
            report_schema_version: REPORT_SCHEMA_VERSION,
            pipeline_schema_version: PIPELINE_SCHEMA_VERSION,
            generated_at: format_system_time_utc(generated_at),
            generated_at_unix_ms: system_time_unix_ms(generated_at).unwrap_or(0),
            package_name: env!("CARGO_PKG_NAME").to_string(),
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            binary_name,
            binary_path: binary_identity.path.clone(),
            binary_sha256: binary_identity.sha256.clone(),
            binary_file_size_bytes: binary_identity.file_size_bytes,
            binary_identity_status: Some(binary_identity.status.clone()),
            binary_identity_error: binary_identity.error.clone(),
            working_directory,
            cli_args,
            output_dir: absolute_path_string(output_dir),
            output_path: absolute_path_string(output_path),
        }
    }
}

#[derive(Debug, Clone)]
struct ExecutableIdentity {
    path: Option<String>,
    sha256: Option<String>,
    file_size_bytes: Option<u64>,
    status: String,
    error: Option<String>,
}

fn current_executable_identity() -> &'static ExecutableIdentity {
    static IDENTITY: OnceLock<ExecutableIdentity> = OnceLock::new();
    IDENTITY.get_or_init(inspect_current_executable)
}

fn inspect_current_executable() -> ExecutableIdentity {
    let path = match std::env::current_exe() {
        Ok(path) => path,
        Err(error) => {
            return ExecutableIdentity {
                path: None,
                sha256: None,
                file_size_bytes: None,
                status: "current_executable_unavailable".to_string(),
                error: Some(error.to_string()),
            };
        }
    };
    let display_path = path.to_string_lossy().to_string();
    let (sha256, file_size_bytes) = match hash_file_sha256(&path) {
        Ok(identity) => identity,
        Err(error) => {
            return ExecutableIdentity {
                path: Some(display_path),
                sha256: None,
                file_size_bytes: None,
                status: "current_executable_hash_failed".to_string(),
                error: Some(error.to_string()),
            };
        }
    };
    ExecutableIdentity {
        path: Some(display_path),
        sha256: Some(sha256),
        file_size_bytes: Some(file_size_bytes),
        status: "verified_sha256".to_string(),
        error: None,
    }
}

/// Reopen a file and compute the SHA-256 of its exact on-disk bytes with bounded memory.
pub fn hash_file_sha256(path: &Path) -> std::io::Result<(String, u64)> {
    let file = File::open(path)?;
    let file_size_bytes = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; FILE_SHA256_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok((format!("{:x}", hasher.finalize()), file_size_bytes))
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

#[cfg(test)]
mod tests {
    use super::inspect_current_executable;

    #[test]
    fn executable_identity_hashing_fits_a_small_cli_stack() {
        let identity = std::thread::Builder::new()
            .name("small-cli-stack-binary-hash".to_string())
            .stack_size(256 * 1024)
            .spawn(inspect_current_executable)
            .expect("spawn small-stack executable identity probe")
            .join()
            .expect("small-stack executable identity probe must not overflow");
        assert_eq!(identity.status, "verified_sha256");
        assert!(identity
            .sha256
            .as_deref()
            .is_some_and(|digest| digest.len() == 64));
        assert!(identity.file_size_bytes.is_some_and(|size| size > 0));
    }
}

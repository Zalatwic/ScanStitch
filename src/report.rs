use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level pipeline report, serialized to JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineReport {
    pub phases: Vec<PhaseReport>,
}

/// Report for a single pipeline phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseReport {
    pub name: String,
    pub success: bool,
    pub confidence: f64,
    pub metrics: serde_json::Value,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl PipelineReport {
    /// Create a new empty report.
    pub fn new() -> Self {
        Self { phases: Vec::new() }
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

impl PhaseReport {
    /// Convenience constructor for a successful phase with no warnings/errors.
    pub fn ok(name: &str, confidence: f64, metrics: serde_json::Value) -> Self {
        Self {
            name: name.to_string(),
            success: true,
            confidence,
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
            metrics: serde_json::Value::Null,
            warnings: Vec::new(),
            errors: vec![error.to_string()],
        }
    }
}

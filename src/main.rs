use clap::Parser;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime};

use scanstitch::cli::Cli;
use scanstitch::pipeline;
use scanstitch::report::{system_time_unix_ms, PipelineReport};
use scanstitch::validation::{delivery_artifact_integrity_issues, summarize_report_with_source};

const DEFAULT_PROGRESS_INTERVAL_SECONDS: u64 = 30;
const MAX_PROGRESS_INTERVAL_SECONDS: u64 = 3_600;

fn progress_interval_from_env() -> Option<Duration> {
    let Some(raw) = std::env::var_os("SCANSTITCH_PROGRESS_INTERVAL_SECONDS") else {
        return Some(Duration::from_secs(DEFAULT_PROGRESS_INTERVAL_SECONDS));
    };
    let value = raw.to_string_lossy();
    match value.parse::<u64>() {
        Ok(0) => None,
        Ok(seconds @ 1..=MAX_PROGRESS_INTERVAL_SECONDS) => Some(Duration::from_secs(seconds)),
        _ => {
            eprintln!(
                "WARN: SCANSTITCH_PROGRESS_INTERVAL_SECONDS must be 0 or 1..={MAX_PROGRESS_INTERVAL_SECONDS}; using {DEFAULT_PROGRESS_INTERVAL_SECONDS}"
            );
            Some(Duration::from_secs(DEFAULT_PROGRESS_INTERVAL_SECONDS))
        }
    }
}

fn format_elapsed(elapsed: Duration) -> String {
    let total_seconds = elapsed.as_secs();
    let hours = total_seconds / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

fn current_run_report(path: &Path, run_started_unix_ms: u64) -> Option<PipelineReport> {
    let contents = std::fs::read_to_string(path).ok()?;
    let report = serde_json::from_str::<PipelineReport>(&contents).ok()?;
    report
        .metadata
        .as_ref()
        .is_some_and(|metadata| metadata.generated_at_unix_ms >= run_started_unix_ms)
        .then_some(report)
}

fn heartbeat_message(elapsed: Duration, report: Option<&PipelineReport>) -> String {
    let elapsed = format_elapsed(elapsed);
    match report.and_then(|report| report.phases.last().map(|phase| (report, phase))) {
        Some((report, phase)) => format!(
            "[RUN] still processing ({elapsed} elapsed); {} phase(s) complete; last: {} ({} ms)",
            report.phases.len(),
            phase.name,
            phase.duration_ms
        ),
        None => format!(
            "[RUN] still processing ({elapsed} elapsed); no phase has completed in this run yet"
        ),
    }
}

fn final_render_review_failure(report: &PipelineReport) -> Option<String> {
    let Some(save) = report
        .phases
        .iter()
        .rev()
        .find(|phase| phase.name == "save")
    else {
        return Some("final save phase is missing from the pipeline report".to_string());
    };
    if !save.success {
        return Some("final save phase did not succeed".to_string());
    }

    let status = save
        .metrics
        .get("render_review_status")
        .and_then(serde_json::Value::as_str);
    let reviewable = save
        .metrics
        .get("render_reviewable")
        .and_then(serde_json::Value::as_bool);
    if status == Some("reviewable") && reviewable == Some(true) {
        return None;
    }

    let reason = save
        .metrics
        .get("render_review_reason")
        .and_then(serde_json::Value::as_str)
        .filter(|reason| !reason.trim().is_empty());
    let mut message = format!(
        "final render evidence is not exactly reviewable (status={}, render_reviewable={})",
        status.unwrap_or("missing"),
        reviewable
            .map(|value| value.to_string())
            .unwrap_or_else(|| "missing".to_string())
    );
    if let Some(reason) = reason {
        message.push_str(": ");
        message.push_str(reason);
    }
    Some(message)
}

struct ProgressHeartbeat {
    stop: Option<Sender<()>>,
    worker: Option<JoinHandle<()>>,
}

impl ProgressHeartbeat {
    fn start(report_path: PathBuf, run_started_unix_ms: u64, interval: Duration) -> Self {
        let (stop, receiver) = mpsc::channel::<()>();
        let worker = thread::spawn(move || {
            let started = Instant::now();
            let mut latest_report = None;
            loop {
                match receiver.recv_timeout(interval) {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                    Err(RecvTimeoutError::Timeout) => {
                        if let Some(report) = current_run_report(&report_path, run_started_unix_ms)
                        {
                            latest_report = Some(report);
                        }
                        eprintln!(
                            "{}",
                            heartbeat_message(started.elapsed(), latest_report.as_ref())
                        );
                    }
                }
            }
        });
        Self {
            stop: Some(stop),
            worker: Some(worker),
        }
    }

    fn stop(mut self) {
        self.stop.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let cli = Cli::parse();
    cli.validate()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    log::info!("scanstitch v{}", env!("CARGO_PKG_VERSION"));
    log::debug!("CLI args: {:?}", cli);

    let report_path = cli.output_dir.join("report.json");
    let run_started = Instant::now();
    let run_started_unix_ms = system_time_unix_ms(SystemTime::now()).unwrap_or(0);
    let heartbeat = progress_interval_from_env().map(|interval| {
        eprintln!(
            "[RUN] processing {} input(s); heartbeat every {}s; partial report: {}",
            cli.inputs.len(),
            interval.as_secs(),
            report_path.display()
        );
        ProgressHeartbeat::start(report_path.clone(), run_started_unix_ms, interval)
    });

    let pipeline_result = pipeline::run(&cli);
    if let Some(heartbeat) = heartbeat {
        heartbeat.stop();
    }
    let report = match pipeline_result {
        Ok(report) => report,
        Err(error) => {
            eprintln!(
                "[FAIL] pipeline stopped after {}; inspect {} for completed-phase diagnostics",
                format_elapsed(run_started.elapsed()),
                report_path.display()
            );
            return Err(error);
        }
    };

    // Save the report
    report.save(&report_path)?;
    log::info!("Report saved to {}", report_path.display());
    // Print summary
    for phase in &report.phases {
        let status = if phase.success { "OK" } else { "FAIL" };
        println!(
            "[{}] {} (confidence: {:.2}, duration: {} ms)",
            status, phase.name, phase.confidence, phase.duration_ms
        );
        for w in &phase.warnings {
            println!("  WARN: {}", w);
        }
        for e in &phase.errors {
            println!("  ERROR: {}", e);
        }
    }

    if cli.require_reviewable {
        let mut failures = final_render_review_failure(&report)
            .into_iter()
            .collect::<Vec<_>>();
        let validation_summary =
            summarize_report_with_source("require-reviewable", &report, Some(&report_path));
        let artifact_issues = delivery_artifact_integrity_issues(&validation_summary.render);
        if !artifact_issues.is_empty() {
            failures.push(format!(
                "saved delivery artifact integrity failed: {}",
                artifact_issues.join(", ")
            ));
        }
        if !failures.is_empty() {
            let message = format!(
                "--require-reviewable rejected the completed render: {}; report and artifacts were retained at {}",
                failures.join("; "),
                cli.output_dir.display()
            );
            eprintln!("[FAIL] {message}");
            return Err(std::io::Error::other(message).into());
        }
    }

    eprintln!(
        "[DONE] {} phase(s) completed in {}; report: {}",
        report.phases.len(),
        format_elapsed(run_started.elapsed()),
        report_path.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        current_run_report, final_render_review_failure, format_elapsed, heartbeat_message,
    };
    use scanstitch::report::{PhaseReport, PipelineReport, RunMetadata};
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn progress_elapsed_format_is_compact_and_unambiguous() {
        assert_eq!(format_elapsed(Duration::from_secs(9)), "00:09");
        assert_eq!(format_elapsed(Duration::from_secs(754)), "12:34");
        assert_eq!(format_elapsed(Duration::from_secs(3_723)), "1:02:03");
    }

    #[test]
    fn progress_heartbeat_reports_only_completed_phase_evidence() {
        assert_eq!(
            heartbeat_message(Duration::from_secs(30), None),
            "[RUN] still processing (00:30 elapsed); no phase has completed in this run yet"
        );

        let mut report = PipelineReport::new();
        report.add_phase(
            PhaseReport::ok("stitch", 0.82, serde_json::Value::Null).with_duration_ms(24_012),
        );
        assert_eq!(
            heartbeat_message(Duration::from_secs(90), Some(&report)),
            "[RUN] still processing (01:30 elapsed); 1 phase(s) complete; last: stitch (24012 ms)"
        );
    }

    #[test]
    fn progress_heartbeat_ignores_a_stale_partial_report() {
        let temporary = tempfile::TempDir::new().unwrap();
        let report_path = temporary.path().join("report.json");
        let metadata = RunMetadata::capture_current(
            temporary.path(),
            &temporary.path().join("output.tiff"),
            UNIX_EPOCH + Duration::from_secs(10),
        );
        let report = PipelineReport::new_with_metadata(metadata);
        report.save(&report_path).unwrap();

        assert!(current_run_report(&report_path, 10_000).is_some());
        assert!(current_run_report(&report_path, 10_001).is_none());
    }

    #[test]
    fn final_render_gate_requires_agreeing_reviewable_evidence() {
        let mut report = PipelineReport::new();
        report.add_phase(PhaseReport::ok(
            "save",
            1.0,
            serde_json::json!({
                "render_review_status": "reviewable",
                "render_reviewable": true,
            }),
        ));

        assert_eq!(final_render_review_failure(&report), None);
    }

    #[test]
    fn final_render_gate_rejects_blocked_missing_and_inconsistent_evidence() {
        let mut blocked = PipelineReport::new();
        blocked.add_phase(PhaseReport::ok(
            "save",
            1.0,
            serde_json::json!({
                "render_review_status": "blocked_low_base_confidence",
                "render_reviewable": false,
                "render_review_reason": "film-base confidence is zero",
            }),
        ));
        let failure = final_render_review_failure(&blocked).expect("blocked render must fail");
        assert!(failure.contains("blocked_low_base_confidence"));
        assert!(failure.contains("film-base confidence is zero"));

        let missing = PipelineReport::new();
        assert!(final_render_review_failure(&missing)
            .expect("missing save phase must fail")
            .contains("save phase is missing"));

        let mut inconsistent = PipelineReport::new();
        inconsistent.add_phase(PhaseReport::ok(
            "save",
            1.0,
            serde_json::json!({
                "render_review_status": "reviewable",
                "render_reviewable": false,
            }),
        ));
        assert!(final_render_review_failure(&inconsistent)
            .expect("inconsistent evidence must fail")
            .contains("render_reviewable=false"));
    }
}

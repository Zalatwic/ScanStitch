use clap::Parser;

use scanstitch::cli::Cli;
use scanstitch::pipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let cli = Cli::parse();
    cli.validate()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    log::info!("scanstitch v{}", env!("CARGO_PKG_VERSION"));
    log::debug!("CLI args: {:?}", cli);

    let report = pipeline::run(&cli)?;

    // Save the report
    let report_path = cli.output_dir.join("report.json");
    report.save(&report_path)?;
    log::info!("Report saved to {}", report_path.display());

    // Print summary
    for phase in &report.phases {
        let status = if phase.success { "OK" } else { "FAIL" };
        println!(
            "[{}] {} (confidence: {:.2})",
            status, phase.name, phase.confidence
        );
        for w in &phase.warnings {
            println!("  WARN: {}", w);
        }
        for e in &phase.errors {
            println!("  ERROR: {}", e);
        }
    }

    Ok(())
}

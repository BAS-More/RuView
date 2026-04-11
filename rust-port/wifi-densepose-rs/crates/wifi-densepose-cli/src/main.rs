//! WiFi-DensePose CLI Entry Point
//!
//! This is the main entry point for the wifi-densepose command-line tool.

use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use wifi_densepose_cli::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Mat(mat_cmd) => {
            wifi_densepose_cli::mat::execute(mat_cmd).await?;
        }
        Commands::Sense { port, nodes, calibration, format } => {
            println!("Starting RuvSense pipeline...");
            println!("  UDP port: {port}, nodes: {nodes}, format: {format}");
            if let Some(cal) = calibration {
                println!("  Calibration: {cal}");
            }
            println!("  (Pipeline implementation: connect ESP32 UDP → RuvSensePipeline::process())");
            println!("  Listening for CSI data on 0.0.0.0:{port}...");
            // TODO: Wire UDP listener → pipeline when hardware is available
        }
        Commands::Serve { host, port, config } => {
            println!("Starting API server on {host}:{port}");
            if let Some(cfg) = config {
                println!("  Config: {cfg}");
            }
            // TODO: Start Axum server from wifi-densepose-api crate
            println!("  (Server implementation: use wifi_densepose_api::create_router())");
        }
        Commands::Train { data, output, epochs, scale } => {
            println!("Training WiFlow model...");
            println!("  Data: {data}");
            println!("  Output: {output}");
            println!("  Epochs: {epochs}, Scale: {scale}");
            println!("  (Training: delegate to `node scripts/train-wiflow.js`)");
        }
        Commands::Calibrate { port, duration, output } => {
            println!("Running room calibration...");
            println!("  Port: {port}, Duration: {duration}s, Output: {output}");
            println!("  (Calibration: record {duration}s of empty-room CSI, compute noise floor)");
        }
        Commands::Config { file, generate } => {
            if generate {
                let config = wifi_densepose_config::Config::default();
                println!("{}", config.to_toml());
            } else if let Some(path) = file {
                match wifi_densepose_config::Config::from_file(&path) {
                    Ok(config) => println!("{:#?}", config),
                    Err(e) => eprintln!("Error loading config: {e}"),
                }
            } else {
                let config = wifi_densepose_config::Config::default();
                println!("{:#?}", config);
            }
        }
        Commands::Version => {
            println!("wifi-densepose {}", env!("CARGO_PKG_VERSION"));
            println!("MAT module version: {}", wifi_densepose_mat::VERSION);
        }
    }

    Ok(())
}

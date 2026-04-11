//! WiFi-DensePose CLI
//!
//! Command-line interface for WiFi-DensePose system, including the
//! Mass Casualty Assessment Tool (MAT) for disaster response.
//!
//! # Features
//!
//! - **mat**: Disaster survivor detection and triage management
//! - **version**: Display version information
//!
//! # Usage
//!
//! ```bash
//! # Start scanning for survivors
//! wifi-densepose mat scan --zone "Building A"
//!
//! # View current scan status
//! wifi-densepose mat status
//!
//! # List detected survivors
//! wifi-densepose mat survivors --sort-by triage
//!
//! # View and manage alerts
//! wifi-densepose mat alerts
//! ```

use clap::{Parser, Subcommand};

pub mod mat;

/// WiFi-DensePose Command Line Interface
#[derive(Parser, Debug)]
#[command(name = "wifi-densepose")]
#[command(author, version, about = "WiFi-based pose estimation and disaster response")]
#[command(propagate_version = true)]
pub struct Cli {
    /// Command to execute
    #[command(subcommand)]
    pub command: Commands,
}

/// Top-level commands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Mass Casualty Assessment Tool commands
    #[command(subcommand)]
    Mat(mat::MatCommand),

    /// Start the RuvSense real-time sensing pipeline
    Sense {
        /// UDP port for ESP32 CSI data (default: 5500)
        #[arg(short, long, default_value_t = 5500)]
        port: u16,

        /// Number of ESP32 nodes to expect
        #[arg(short, long, default_value_t = 3)]
        nodes: usize,

        /// Path to calibration file
        #[arg(short, long)]
        calibration: Option<String>,

        /// Output format: json, text, or csv
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Start the REST API server
    Serve {
        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Path to TOML configuration file
        #[arg(short, long)]
        config: Option<String>,
    },

    /// Train a model from CSI data
    Train {
        /// Path to CSI JSONL data file(s)
        #[arg(short, long)]
        data: String,

        /// Output directory for trained model
        #[arg(short, long, default_value = "models/")]
        output: String,

        /// Number of training epochs
        #[arg(short, long, default_value_t = 50)]
        epochs: u32,

        /// Model scale: lite, small, medium, full
        #[arg(short, long, default_value = "lite")]
        scale: String,
    },

    /// Run room calibration (empty room baseline)
    Calibrate {
        /// UDP port for ESP32 CSI data
        #[arg(short, long, default_value_t = 5500)]
        port: u16,

        /// Duration in seconds
        #[arg(short, long, default_value_t = 30)]
        duration: u32,

        /// Output calibration file
        #[arg(short, long, default_value = "data/calibration.json")]
        output: String,
    },

    /// Show or validate configuration
    Config {
        /// Path to TOML config file
        #[arg(short, long)]
        file: Option<String>,

        /// Generate default config and print to stdout
        #[arg(long)]
        generate: bool,
    },

    /// Display version information
    Version,
}

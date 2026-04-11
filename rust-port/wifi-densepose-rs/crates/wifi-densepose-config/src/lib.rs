//! Configuration management for WiFi-DensePose.
//!
//! Loads configuration from TOML files with environment variable overrides.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),
}

/// Top-level configuration for the WiFi-DensePose system.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub hardware: HardwareConfig,
    pub signal_processing: SignalConfig,
    pub neural_network: NeuralConfig,
    pub api: ApiConfig,
    pub storage: StorageConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hardware: HardwareConfig::default(),
            signal_processing: SignalConfig::default(),
            neural_network: NeuralConfig::default(),
            api: ApiConfig::default(),
            storage: StorageConfig::default(),
        }
    }
}

/// Hardware configuration for ESP32 sensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HardwareConfig {
    pub num_sensors: usize,
    pub num_subcarriers: usize,
    pub num_antennas: usize,
    pub sampling_rate_hz: f64,
    pub udp_port: u16,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            num_sensors: 3,
            num_subcarriers: 64,
            num_antennas: 3,
            sampling_rate_hz: 100.0,
            udp_port: 5500,
        }
    }
}

/// Signal processing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SignalConfig {
    pub noise_threshold: f64,
    pub human_detection_threshold: f64,
    pub smoothing_factor: f64,
    pub window_size: usize,
    pub doppler_enabled: bool,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            noise_threshold: 0.1,
            human_detection_threshold: 0.8,
            smoothing_factor: 0.9,
            window_size: 512,
            doppler_enabled: true,
        }
    }
}

/// Neural network configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NeuralConfig {
    pub model_path: Option<String>,
    pub backend: String,
    pub batch_size: usize,
    pub num_body_parts: usize,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            backend: "onnx".to_string(),
            batch_size: 1,
            num_body_parts: 24,
        }
    }
}

/// API server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub cors_origins: Vec<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            cors_origins: vec!["*".to_string()],
        }
    }
}

/// Storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    pub database_path: String,
    pub retention_days: u32,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            database_path: "data/ruview.db".to_string(),
            retention_days: 30,
        }
    }
}

impl Config {
    /// Load configuration from a TOML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Load configuration with defaults for any missing values.
    pub fn from_file_or_default(path: impl AsRef<Path>) -> Self {
        Self::from_file(path).unwrap_or_default()
    }

    /// Serialize configuration to TOML string.
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.hardware.num_sensors, 3);
        assert_eq!(config.api.port, 8080);
    }

    #[test]
    fn test_roundtrip_toml() {
        let config = Config::default();
        let toml_str = config.to_toml();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.hardware.num_subcarriers, 64);
    }
}

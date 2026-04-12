"""
Unit tests for CSIExtractor against the actual production API.

The CSIExtractor accepts a config dict with keys:
    hardware_type, sampling_rate, buffer_size, timeout
and an optional logger. It delegates parsing to ESP32CSIParser,
ESP32BinaryParser, or RouterCSIParser.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from v1.src.hardware.csi_extractor import (
    CSIExtractor,
    CSIExtractionError,
    CSIParseError,
    CSIValidationError,
    CSIData,
    ESP32CSIParser,
    ESP32BinaryParser,
    RouterCSIParser,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def esp32_config():
    return {
        "hardware_type": "esp32",
        "sampling_rate": 100,
        "buffer_size": 1024,
        "timeout": 5.0,
        "validation_enabled": True,
        "retry_attempts": 3,
    }


@pytest.fixture
def router_config():
    return {
        "hardware_type": "router",
        "sampling_rate": 50,
        "buffer_size": 512,
        "timeout": 10.0,
    }


@pytest.fixture
def binary_config():
    return {
        "hardware_type": "esp32",
        "parser_format": "binary",
        "sampling_rate": 100,
        "buffer_size": 1024,
        "timeout": 5.0,
    }


def make_csi_data(**overrides):
    """Build a valid CSIData for testing."""
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
        amplitude=np.random.rand(3, 56),
        phase=np.random.rand(3, 56),
        frequency=2.4e9,
        bandwidth=20e6,
        num_subcarriers=56,
        num_antennas=3,
        snr=15.5,
        metadata={"source": "test"},
    )
    defaults.update(overrides)
    return CSIData(**defaults)


# ===========================================================================
# Initialization
# ===========================================================================

class TestCSIExtractorInit:
    def test_creates_esp32_text_parser_by_default(self, esp32_config):
        ext = CSIExtractor(esp32_config)
        assert isinstance(ext.parser, ESP32CSIParser)
        assert ext.hardware_type == "esp32"
        assert ext.sampling_rate == 100
        assert ext.buffer_size == 1024

    def test_creates_binary_parser_when_configured(self, binary_config):
        ext = CSIExtractor(binary_config)
        assert isinstance(ext.parser, ESP32BinaryParser)

    def test_creates_router_parser(self, router_config):
        ext = CSIExtractor(router_config)
        assert isinstance(ext.parser, RouterCSIParser)

    def test_rejects_unsupported_hardware(self):
        cfg = {
            "hardware_type": "unsupported",
            "sampling_rate": 100,
            "buffer_size": 1024,
            "timeout": 5.0,
        }
        with pytest.raises(ValueError, match="Unsupported hardware type"):
            CSIExtractor(cfg)

    def test_rejects_missing_required_fields(self):
        with pytest.raises(ValueError, match="Missing required"):
            CSIExtractor({"hardware_type": "esp32"})

    def test_rejects_non_positive_sampling_rate(self):
        cfg = {
            "hardware_type": "esp32",
            "sampling_rate": 0,
            "buffer_size": 1024,
            "timeout": 5.0,
        }
        with pytest.raises(ValueError, match="sampling_rate must be positive"):
            CSIExtractor(cfg)

    def test_rejects_non_positive_buffer_size(self):
        cfg = {
            "hardware_type": "esp32",
            "sampling_rate": 100,
            "buffer_size": -1,
            "timeout": 5.0,
        }
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            CSIExtractor(cfg)

    def test_not_connected_initially(self, esp32_config):
        ext = CSIExtractor(esp32_config)
        assert ext.is_connected is False
        assert ext.is_streaming is False

    def test_accepts_custom_logger(self, esp32_config):
        logger = Mock()
        ext = CSIExtractor(esp32_config, logger=logger)
        assert ext.logger is logger


# ===========================================================================
# Connection lifecycle
# ===========================================================================

class TestCSIExtractorConnection:
    @pytest.mark.asyncio
    async def test_connect_sets_connected_flag(self, esp32_config):
        ext = CSIExtractor(esp32_config)
        result = await ext.connect()
        assert result is True
        assert ext.is_connected is True

    @pytest.mark.asyncio
    async def test_disconnect_clears_flag(self, esp32_config):
        ext = CSIExtractor(esp32_config)
        await ext.connect()
        await ext.disconnect()
        assert ext.is_connected is False

    @pytest.mark.asyncio
    async def test_extract_requires_connection(self, esp32_config):
        ext = CSIExtractor(esp32_config)
        with pytest.raises(CSIParseError, match="Not connected"):
            await ext.extract_csi()


# ===========================================================================
# Validation
# ===========================================================================

class TestCSIDataValidation:
    def test_valid_data_passes(self, esp32_config):
        ext = CSIExtractor(esp32_config)
        csi = make_csi_data()
        assert ext.validate_csi_data(csi) is True

    def test_rejects_empty_amplitude(self, esp32_config):
        ext = CSIExtractor(esp32_config)
        csi = make_csi_data(amplitude=np.array([]))
        with pytest.raises(CSIValidationError, match="Empty amplitude"):
            ext.validate_csi_data(csi)

    def test_rejects_empty_phase(self, esp32_config):
        ext = CSIExtractor(esp32_config)
        csi = make_csi_data(phase=np.array([]))
        with pytest.raises(CSIValidationError, match="Empty phase"):
            ext.validate_csi_data(csi)

    def test_rejects_invalid_frequency(self, esp32_config):
        ext = CSIExtractor(esp32_config)
        csi = make_csi_data(frequency=-1.0)
        with pytest.raises(CSIValidationError, match="Invalid frequency"):
            ext.validate_csi_data(csi)

    def test_rejects_extreme_snr(self, esp32_config):
        ext = CSIExtractor(esp32_config)
        csi = make_csi_data(snr=100.0)
        with pytest.raises(CSIValidationError, match="Invalid SNR"):
            ext.validate_csi_data(csi)


# ===========================================================================
# ESP32 text parser
# ===========================================================================

class TestESP32CSIParser:
    def test_parses_valid_frame(self):
        parser = ESP32CSIParser()
        # Build minimal valid ESP32 text frame
        n_ant, n_sub = 1, 2
        amps = "1.5,2.5"
        phases = "0.1,0.2"
        raw = f"CSI_DATA:1000000,{n_ant},{n_sub},2400,20,10.0,{amps},{phases}"
        csi = parser.parse(raw.encode())
        assert csi.num_antennas == 1
        assert csi.num_subcarriers == 2
        assert csi.amplitude.shape == (1, 2)
        assert csi.phase.shape == (1, 2)

    def test_rejects_empty_data(self):
        parser = ESP32CSIParser()
        with pytest.raises(CSIParseError, match="Empty data"):
            parser.parse(b"")

    def test_rejects_wrong_prefix(self):
        parser = ESP32CSIParser()
        with pytest.raises(CSIParseError, match="Invalid ESP32"):
            parser.parse(b"WRONG_PREFIX:data")

    def test_rejects_incomplete_iq_data(self):
        parser = ESP32CSIParser()
        # 1 antenna * 2 subcarriers * 2 (amp+phase) = 4 values needed, provide 2
        raw = b"CSI_DATA:1000000,1,2,2400,20,10.0,1.5,2.5"
        with pytest.raises(CSIExtractionError, match="incomplete"):
            parser.parse(raw)


# ===========================================================================
# Streaming
# ===========================================================================

class TestCSIExtractorStreaming:
    @pytest.mark.asyncio
    async def test_stop_streaming_sets_flag(self, esp32_config):
        ext = CSIExtractor(esp32_config)
        ext.is_streaming = True
        ext.stop_streaming()
        assert ext.is_streaming is False

"""Tests for PoseService."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

# PoseService requires torch — skip entire module if unavailable
torch = pytest.importorskip("torch")

from v1.src.services.pose_service import PoseService


class TestPoseServiceInit:
    def test_init_sets_defaults(self, mock_settings, mock_domain_config):
        svc = PoseService(mock_settings, mock_domain_config)
        assert svc.is_initialized is False
        assert svc.is_running is False
        assert svc.stats["total_processed"] == 0

    def test_stats_are_zero_on_init(self, mock_settings, mock_domain_config):
        svc = PoseService(mock_settings, mock_domain_config)
        assert svc.stats["successful_detections"] == 0
        assert svc.stats["failed_detections"] == 0
        assert svc.stats["average_confidence"] == 0.0


class TestPoseServiceLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_sets_flag(self, mock_settings, mock_domain_config):
        with patch("v1.src.services.pose_service.CSIProcessor"):
            svc = PoseService(mock_settings, mock_domain_config)
            await svc.initialize()
            assert svc.is_initialized is True

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_settings, mock_domain_config):
        with patch("v1.src.services.pose_service.CSIProcessor"):
            svc = PoseService(mock_settings, mock_domain_config)
            await svc.initialize()
            await svc.start()
            assert svc.is_running is True
            await svc.stop()
            assert svc.is_running is False


class TestPoseServiceStats:
    def test_initial_classification(self, mock_settings, mock_domain_config):
        svc = PoseService(mock_settings, mock_domain_config)
        assert svc.last_error is None

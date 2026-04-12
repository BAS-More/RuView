"""Unit tests for RouterInterface against the actual async SSH API."""

import sys
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Mock asyncssh before importing RouterInterface
sys.modules.setdefault("asyncssh", MagicMock())

from v1.src.hardware.router_interface import RouterInterface, RouterConnectionError


class TestRouterInterface:
    """Tests for RouterInterface using London School TDD (mocked SSH)."""

    @pytest.fixture
    def config(self):
        return {
            "host": "192.168.1.1",
            "port": 22,
            "username": "admin",
            "password": "password",
            "command_timeout": 30,
            "max_retries": 3,
        }

    @pytest.fixture
    def router(self, config):
        return RouterInterface(config)

    def test_init_stores_config(self, config):
        r = RouterInterface(config)
        assert r.host == "192.168.1.1"
        assert r.port == 22
        assert r.username == "admin"
        assert r.is_connected is False

    def test_rejects_missing_host(self):
        with pytest.raises(ValueError, match="Missing required"):
            RouterInterface({"port": 22, "username": "a", "password": "b"})

    def test_rejects_missing_password(self):
        with pytest.raises(ValueError, match="Missing required"):
            RouterInterface({"host": "x", "port": 22, "username": "a"})

    @pytest.mark.asyncio
    async def test_connect_success(self, router):
        mock_conn = MagicMock()
        with patch("v1.src.hardware.router_interface.asyncssh") as mock_ssh:
            mock_ssh.connect = AsyncMock(return_value=mock_conn)
            result = await router.connect()
        assert result is True
        assert router.is_connected is True
        assert router.ssh_client is mock_conn

    @pytest.mark.asyncio
    async def test_connect_failure_returns_false(self, router):
        """connect() returns False on failure (does not raise)."""
        with patch("v1.src.hardware.router_interface.asyncssh") as mock_ssh:
            mock_ssh.connect = AsyncMock(side_effect=OSError("refused"))
            result = await router.connect()
        assert result is False
        assert router.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, router):
        router.is_connected = True
        router.ssh_client = MagicMock()
        await router.disconnect()
        assert router.is_connected is False
        assert router.ssh_client is None

    @pytest.mark.asyncio
    async def test_execute_command_requires_connection(self, router):
        with pytest.raises(RouterConnectionError):
            await router.execute_command("ls")

    @pytest.mark.asyncio
    async def test_execute_command_runs(self, router):
        mock_result = MagicMock()
        mock_result.stdout = "output text"
        mock_result.stderr = ""
        mock_result.returncode = 0
        router.is_connected = True
        router.ssh_client = AsyncMock()
        router.ssh_client.run = AsyncMock(return_value=mock_result)

        result = await router.execute_command("uname -a")
        assert result == "output text"

    @pytest.mark.asyncio
    async def test_execute_command_raises_on_nonzero_exit(self, router):
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "not found"
        mock_result.returncode = 1
        router.is_connected = True
        router.ssh_client = AsyncMock()
        router.ssh_client.run = AsyncMock(return_value=mock_result)

        with pytest.raises(RouterConnectionError, match="Command failed"):
            await router.execute_command("bad_cmd")

    @pytest.mark.asyncio
    async def test_health_check_when_connected(self, router):
        mock_result = MagicMock()
        mock_result.stdout = "ping\npong\n"
        mock_result.stderr = ""
        mock_result.returncode = 0
        router.is_connected = True
        router.ssh_client = AsyncMock()
        router.ssh_client.run = AsyncMock(return_value=mock_result)

        result = await router.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_when_disconnected(self, router):
        assert await router.health_check() is False

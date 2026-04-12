"""Tests for AuthMiddleware and TokenManager."""

import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta


class TestTokenManager:
    def test_create_token(self, mock_settings):
        from v1.src.middleware.auth import TokenManager
        tm = TokenManager(mock_settings)
        token = tm.create_access_token({"sub": "user1"})
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_valid_token(self, mock_settings):
        from v1.src.middleware.auth import TokenManager
        tm = TokenManager(mock_settings)
        token = tm.create_access_token({"sub": "user1", "role": "admin"})
        payload = tm.verify_token(token)
        assert payload["sub"] == "user1"
        assert payload["role"] == "admin"

    def test_verify_invalid_token(self, mock_settings):
        from v1.src.middleware.auth import TokenManager, AuthenticationError
        tm = TokenManager(mock_settings)
        with pytest.raises(AuthenticationError):
            tm.verify_token("invalid.token.here")

    def test_decode_claims(self, mock_settings):
        from v1.src.middleware.auth import TokenManager
        tm = TokenManager(mock_settings)
        token = tm.create_access_token({"sub": "user1"})
        claims = tm.decode_token_claims(token)
        assert claims is not None
        assert claims["sub"] == "user1"

    def test_decode_claims_invalid(self, mock_settings):
        from v1.src.middleware.auth import TokenManager
        tm = TokenManager(mock_settings)
        claims = tm.decode_token_claims("bad-token")
        assert claims is None

    def test_token_has_expiry(self, mock_settings):
        from v1.src.middleware.auth import TokenManager
        tm = TokenManager(mock_settings)
        token = tm.create_access_token({"sub": "user1"})
        payload = tm.verify_token(token)
        assert "exp" in payload
        assert "iat" in payload


class TestUserManager:
    def test_create_user(self):
        from v1.src.middleware.auth import UserManager
        um = UserManager()
        assert um.get_user("nonexistent") is None

    @pytest.mark.skipif(
        __import__("sys").version_info >= (3, 14),
        reason="passlib bcrypt backend incompatible with Python 3.14+",
    )
    def test_hash_password(self):
        from v1.src.middleware.auth import UserManager
        hashed = UserManager.hash_password("secret123")
        assert hashed != "secret123"
        assert len(hashed) > 20

    @pytest.mark.skipif(
        __import__("sys").version_info >= (3, 14),
        reason="passlib bcrypt backend incompatible with Python 3.14+",
    )
    def test_verify_password(self):
        from v1.src.middleware.auth import UserManager
        hashed = UserManager.hash_password("secret123")
        assert UserManager.verify_password("secret123", hashed) is True
        assert UserManager.verify_password("wrong", hashed) is False


class TestTokenBlacklist:
    def test_add_and_check(self):
        from v1.src.api.middleware.auth import TokenBlacklist
        bl = TokenBlacklist()
        bl.add_token("tok123")
        assert bl.is_blacklisted("tok123") is True
        assert bl.is_blacklisted("tok456") is False

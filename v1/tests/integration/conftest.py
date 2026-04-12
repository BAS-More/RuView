"""Shared fixtures for integration tests."""

import os

# Set required environment variables before any Settings import
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-integration-tests-only")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-integration-tests-only")

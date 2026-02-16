"""Pytest fixtures for BlazeRPC tests."""

from __future__ import annotations

import pytest

from blazerpc.app import BlazeApp


@pytest.fixture
def app() -> BlazeApp:
    """Create a fresh BlazeApp instance for testing."""
    return BlazeApp(name="test", enable_batching=False)

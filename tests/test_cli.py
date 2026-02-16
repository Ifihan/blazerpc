"""Tests for CLI commands."""

from __future__ import annotations

import os
import tempfile

import pytest

from blazerpc.cli.serve import load_app
from blazerpc.exceptions import ConfigurationError


# -- load_app --


def test_load_app_missing_colon() -> None:
    with pytest.raises(ConfigurationError, match="Expected format"):
        load_app("myapp")


def test_load_app_bad_module() -> None:
    with pytest.raises(ConfigurationError, match="Could not import"):
        load_app("nonexistent_module_xyz:app")


def test_load_app_bad_attribute() -> None:
    # os module exists but has no 'blazeapp' attribute
    with pytest.raises(ConfigurationError, match="has no attribute"):
        load_app("os:blazeapp")


def test_load_app_not_blazeapp() -> None:
    # os.path exists but is not a BlazeApp
    with pytest.raises(ConfigurationError, match="not a BlazeApp"):
        load_app("os:path")


# -- proto export --


def test_proto_export() -> None:
    from blazerpc.app import BlazeApp
    from blazerpc.cli.proto import export_proto

    app = BlazeApp(enable_batching=False)

    @app.model("test_model")
    def predict(text: str) -> float:
        return 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_proto(app, tmpdir)
        assert os.path.exists(path)
        assert path.endswith("blaze_service.proto")
        content = open(path).read()
        assert 'syntax = "proto3";' in content
        assert "TestModel" in content

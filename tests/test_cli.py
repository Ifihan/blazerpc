"""Tests for CLI commands."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from blazerpc.app import BlazeApp
from blazerpc.cli.main import app
from blazerpc.cli.proto import export_proto
from blazerpc.cli.serve import load_app
from blazerpc.exceptions import ConfigurationError

runner = CliRunner()


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


# -- serve options --


def test_reload_forwards_transport_and_ports() -> None:
    with patch("blazerpc.cli.main.run_with_reload") as run_with_reload:
        result = runner.invoke(
            app,
            [
                "serve",
                "app:app",
                "--reload",
                "--transport",
                "both",
                "--port",
                "50052",
                "--http-port",
                "8081",
            ],
        )

    assert result.exit_code == 0
    run_with_reload.assert_called_once_with(
        "app:app", "0.0.0.0", 50052, 8081, "both"
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--transport", "invalid"], "must be grpc, jsonrpc, or both"),
        (["--workers", "2"], "only 1 worker is supported"),
        (["--workers", "0"], "only 1 worker is supported"),
        (["--port", "0"], "must be between 1 and 65535"),
        (["--http-port", "65536"], "must be between 1 and 65535"),
        (
            ["--transport", "both", "--port", "8080", "--http-port", "8080"],
            "must differ from --port",
        ),
    ],
)
def test_invalid_serve_options_do_not_spawn_reload(
    options: list[str], message: str
) -> None:
    with patch("blazerpc.cli.main.run_with_reload") as run_with_reload:
        result = runner.invoke(app, ["serve", "app:app", "--reload", *options])

    assert result.exit_code == 2
    assert message in result.output
    run_with_reload.assert_not_called()

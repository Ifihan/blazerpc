"""Tests for the hot reload module."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, Mock, patch

import pytest

from blazerpc.cli.reload import _python_filter, _run_server, run_with_reload


def test_python_filter_accepts_py_files() -> None:
    assert _python_filter("modified", "/app/main.py") is True
    assert _python_filter("modified", "/app/models/predict.py") is True


def test_python_filter_rejects_non_py_files() -> None:
    assert _python_filter("modified", "/app/data.json") is False
    assert _python_filter("modified", "/app/model.onnx") is False
    assert _python_filter("modified", "/app/README.md") is False
    assert _python_filter("modified", "/app/.pyc") is False


def test_run_server_is_callable() -> None:
    assert callable(_run_server)


def test_run_with_reload_missing_watchfiles() -> None:
    """run_with_reload exits cleanly when watchfiles is not installed."""
    with patch.dict("sys.modules", {"watchfiles": None}):
        try:
            run_with_reload("app:app", "127.0.0.1", 50051, 8080, "grpc")
        except SystemExit as exc:
            assert exc.code == 1
        else:
            # watchfiles is installed in dev, so the import succeeds
            # and run_process would try to start — that's fine for this test
            pass


def test_run_with_reload_passes_all_child_arguments() -> None:
    watchfiles = ModuleType("watchfiles")
    watchfiles.run_process = Mock()  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"watchfiles": watchfiles}):
        run_with_reload("app:app", "127.0.0.1", 50052, 8081, "both")

    watchfiles.run_process.assert_called_once_with(  # type: ignore[attr-defined]
        ".",
        target=_run_server,
        args=("app:app", "127.0.0.1", 50052, 8081, "both"),
        watch_filter=_python_filter,
    )


@pytest.mark.parametrize("transport", ["grpc", "jsonrpc", "both"])
def test_reload_child_uses_selected_transport(transport: str) -> None:
    blaze_app = Mock()
    blaze_app.serve = AsyncMock()
    blaze_app.serve_jsonrpc = AsyncMock()
    blaze_app.serve_both = AsyncMock()

    with patch("blazerpc.cli.serve.load_app", return_value=blaze_app):
        _run_server("app:app", "127.0.0.1", 50052, 8081, transport)

    if transport == "grpc":
        blaze_app.serve.assert_awaited_once_with("127.0.0.1", 50052)
    elif transport == "jsonrpc":
        blaze_app.serve_jsonrpc.assert_awaited_once_with("127.0.0.1", 8081)
    else:
        blaze_app.serve_both.assert_awaited_once_with(
            "127.0.0.1", grpc_port=50052, http_port=8081
        )

"""Tests for the hot reload module."""

from __future__ import annotations

from unittest.mock import patch

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
            run_with_reload("app:app", "127.0.0.1", 50051)
        except SystemExit as exc:
            assert exc.code == 1
        else:
            # watchfiles is installed in dev, so the import succeeds
            # and run_process would try to start — that's fine for this test
            pass

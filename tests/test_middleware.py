"""Tests for middleware / interceptors."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from grpclib.const import Status
from grpclib.events import RecvRequest, SendTrailingMetadata
from grpclib.server import Server

from blazerpc.app import BlazeApp
from blazerpc.codegen.servicer import build_servicer
from blazerpc.server.middleware import (
    ExceptionMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    Middleware,
)


# ---------------------------------------------------------------------------
# Middleware base class
# ---------------------------------------------------------------------------


def test_middleware_is_abstract() -> None:
    """Cannot instantiate Middleware directly."""
    with pytest.raises(TypeError):
        Middleware()  # type: ignore[abstract]


def test_middleware_attach_registers_listeners() -> None:
    """attach() should register event listeners on the server."""
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    servicer = build_servicer(app.registry)
    server = Server([servicer])

    mw = LoggingMiddleware()
    mw.attach(server)

    # Verify listeners were registered (grpclib stores them on __dispatch__)
    dispatch = server.__dispatch__
    assert len(dispatch._listeners[RecvRequest]) >= 1
    assert len(dispatch._listeners[SendTrailingMetadata]) >= 1


# ---------------------------------------------------------------------------
# LoggingMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logging_middleware_on_request(caplog: pytest.LogCaptureFixture) -> None:
    """LoggingMiddleware logs the method name on request."""
    mw = LoggingMiddleware()

    event = MagicMock(spec=RecvRequest)
    event.method_name = "/blazerpc.InferenceService/PredictEcho"
    event.peer = "127.0.0.1:54321"

    with caplog.at_level(logging.INFO, logger="blazerpc.middleware"):
        await mw.on_request(event)

    assert "PredictEcho" in caplog.text
    assert "127.0.0.1" in caplog.text


@pytest.mark.asyncio
async def test_logging_middleware_on_response(caplog: pytest.LogCaptureFixture) -> None:
    """LoggingMiddleware logs the status on response."""
    mw = LoggingMiddleware()

    event = MagicMock(spec=SendTrailingMetadata)
    event.status = Status.OK
    event.status_message = ""

    with caplog.at_level(logging.INFO, logger="blazerpc.middleware"):
        await mw.on_response(event)

    assert "OK" in caplog.text


# ---------------------------------------------------------------------------
# MetricsMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_middleware_records() -> None:
    """MetricsMiddleware should track request count and duration."""
    mw = MetricsMiddleware()

    # Simulate a request event
    req_event = MagicMock(spec=RecvRequest)
    req_event.method_name = "/blazerpc.InferenceService/PredictEcho"
    req_event.metadata = {}  # Use as key

    await mw.on_request(req_event)

    # Simulate response event with same metadata object
    resp_event = MagicMock(spec=SendTrailingMetadata)
    resp_event.metadata = req_event.metadata
    resp_event.status = Status.OK

    await mw.on_response(resp_event)

    # Timing entry should be consumed
    assert len(mw._timings) == 0


@pytest.mark.asyncio
async def test_metrics_middleware_missing_timing() -> None:
    """MetricsMiddleware handles missing timing gracefully."""
    mw = MetricsMiddleware()

    resp_event = MagicMock(spec=SendTrailingMetadata)
    resp_event.metadata = {"new": True}
    resp_event.status = Status.OK

    # Should not raise even with no matching request
    await mw.on_response(resp_event)


# ---------------------------------------------------------------------------
# ExceptionMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exception_middleware_is_noop() -> None:
    """ExceptionMiddleware should be a no-op by default."""
    mw = ExceptionMiddleware()

    req_event = MagicMock(spec=RecvRequest)
    resp_event = MagicMock(spec=SendTrailingMetadata)

    await mw.on_request(req_event)
    await mw.on_response(resp_event)

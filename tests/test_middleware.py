"""Tests for middleware / interceptors."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from grpclib.const import Status
from grpclib.events import RecvRequest, SendTrailingMetadata
from grpclib.exceptions import GRPCError
from grpclib.server import Server

from blazerpc.app import BlazeApp
from blazerpc.codegen.servicer import build_servicer
from blazerpc.server.middleware import (
    ExceptionMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    Middleware,
    OTelMetricsMiddleware,
)


# ---------------------------------------------------------------------------
# Middleware base class
# ---------------------------------------------------------------------------


def test_middleware_is_abstract() -> None:
    """Cannot instantiate Middleware directly."""
    with pytest.raises(TypeError):
        Middleware()  # type: ignore[abstract]


async def test_middleware_attach_registers_listeners() -> None:
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
    req_event.method_func = AsyncMock()

    await mw.on_request(req_event)
    await req_event.method_func(MagicMock())

    assert mw._current_request.get() is None


@pytest.mark.asyncio
async def test_metrics_middleware_missing_timing() -> None:
    """MetricsMiddleware handles missing timing gracefully."""
    mw = MetricsMiddleware()

    resp_event = MagicMock(spec=SendTrailingMetadata)
    resp_event.metadata = {"new": True}
    resp_event.status = Status.OK

    # Should not raise even with no matching request
    await mw.on_response(resp_event)


@pytest.mark.asyncio
async def test_metrics_middleware_correlates_concurrent_errors() -> None:
    mw = MetricsMiddleware()
    mw._record = MagicMock()  # type: ignore[method-assign]

    async def explicit_error(_stream: object) -> None:
        response = MagicMock(spec=SendTrailingMetadata)
        response.status = Status.PERMISSION_DENIED
        await mw.on_response(response)
        await asyncio.sleep(0)

    async def raised_error(_stream: object) -> None:
        await asyncio.sleep(0)
        raise GRPCError(Status.INVALID_ARGUMENT)

    first = MagicMock(spec=RecvRequest)
    first.method_name = "/service/First"
    first.method_func = explicit_error
    second = MagicMock(spec=RecvRequest)
    second.method_name = "/service/Second"
    second.method_func = raised_error
    await mw.on_request(first)
    await mw.on_request(second)

    results = await asyncio.gather(
        first.method_func(MagicMock()),
        second.method_func(MagicMock()),
        return_exceptions=True,
    )

    assert isinstance(results[1], GRPCError)
    assert mw._record.call_count == 2
    recorded = {(item.args[0], item.args[1]) for item in mw._record.call_args_list}
    assert recorded == {
        ("/service/First", Status.PERMISSION_DENIED),
        ("/service/Second", Status.INVALID_ARGUMENT),
    }
    assert all(item.args[2] >= 0 for item in mw._record.call_args_list)


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


# ---------------------------------------------------------------------------
# OTelMetricsMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_otel_metrics_middleware_records() -> None:
    """OTelMetricsMiddleware should track request count and duration."""
    mw = OTelMetricsMiddleware()

    req_event = MagicMock(spec=RecvRequest)
    req_event.method_name = "/blazerpc.InferenceService/PredictEcho"
    req_event.method_func = AsyncMock()

    await mw.on_request(req_event)
    await req_event.method_func(MagicMock())

    assert mw._current_request.get() is None


@pytest.mark.asyncio
async def test_otel_metrics_middleware_missing_timing() -> None:
    """OTelMetricsMiddleware handles missing timing gracefully."""
    mw = OTelMetricsMiddleware()

    resp_event = MagicMock(spec=SendTrailingMetadata)
    resp_event.metadata = {"new": True}
    resp_event.status = Status.OK

    await mw.on_response(resp_event)


@pytest.mark.asyncio
async def test_otel_metrics_middleware_custom_meter() -> None:
    """OTelMetricsMiddleware accepts a custom Meter."""
    mock_meter = MagicMock()
    mock_meter.create_counter.return_value = MagicMock()
    mock_meter.create_histogram.return_value = MagicMock()

    mw = OTelMetricsMiddleware(meter=mock_meter)

    mock_meter.create_counter.assert_called_once_with(
        "blazerpc.rpc.count",
        description="Total number of gRPC requests",
    )
    mock_meter.create_histogram.assert_called_once_with(
        "blazerpc.rpc.duration",
        unit="s",
        description="RPC request duration in seconds",
    )

    # Verify instruments are used on request/response
    req_event = MagicMock(spec=RecvRequest)
    req_event.method_name = "/blazerpc.InferenceService/PredictEcho"
    req_event.method_func = AsyncMock()

    await mw.on_request(req_event)
    await req_event.method_func(MagicMock())

    mw._request_count.add.assert_called_once_with(
        1,
        {"method": "/blazerpc.InferenceService/PredictEcho", "status": "0"},
    )
    mw._request_duration.record.assert_called_once()
    duration, attributes = mw._request_duration.record.call_args.args
    assert duration >= 0
    assert attributes == {"method": "/blazerpc.InferenceService/PredictEcho"}

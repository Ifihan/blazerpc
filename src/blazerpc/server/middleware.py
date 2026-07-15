"""Interceptors for logging, auth, rate limiting, etc.

Built on top of grpclib's event system.  Each middleware is an async
callback that hooks into ``RecvRequest`` / ``SendTrailingMetadata``
events.

Usage::

    from blazerpc.server.middleware import LoggingMiddleware, MetricsMiddleware

    # Attach to a grpclib Server instance:
    LoggingMiddleware().attach(grpclib_server)
    MetricsMiddleware().attach(grpclib_server)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from grpclib.const import Status
from grpclib.events import RecvRequest, SendTrailingMetadata, listen
from grpclib.exceptions import GRPCError
from grpclib.server import Server, Stream
from opentelemetry import metrics as otel_metrics
from prometheus_client import Counter, Histogram

log = logging.getLogger("blazerpc.middleware")


# ---------------------------------------------------------------------------
# Base middleware protocol
# ---------------------------------------------------------------------------


class Middleware(ABC):
    """Base class for BlazeRPC server middleware.

    Subclasses implement :meth:`on_request` and/or :meth:`on_response`
    to hook into the request lifecycle.  Call :meth:`attach` to register
    the middleware on a :class:`grpclib.server.Server`.
    """

    def attach(self, server: Server) -> None:
        """Register this middleware's event listeners on *server*."""
        listen(server, RecvRequest, self._handle_recv_request)
        listen(server, SendTrailingMetadata, self._handle_send_trailing)

    async def _handle_recv_request(self, event: RecvRequest) -> None:
        await self.on_request(event)

    async def _handle_send_trailing(self, event: SendTrailingMetadata) -> None:
        await self.on_response(event)

    @abstractmethod
    async def on_request(self, event: RecvRequest) -> None:
        """Called when a request is received."""

    @abstractmethod
    async def on_response(self, event: SendTrailingMetadata) -> None:
        """Called when a response is about to be sent."""


# ---------------------------------------------------------------------------
# Logging middleware
# ---------------------------------------------------------------------------


class LoggingMiddleware(Middleware):
    """Logs each RPC request with method name, peer, and status."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or log

    async def on_request(self, event: RecvRequest) -> None:
        peer = event.peer
        self._log.info("RPC request: %s from %s", event.method_name, peer)

    async def on_response(self, event: SendTrailingMetadata) -> None:
        self._log.info(
            "RPC response: status=%s message=%s",
            event.status,
            event.status_message or "",
        )


# ---------------------------------------------------------------------------
# Metrics middleware (Prometheus)
# ---------------------------------------------------------------------------


_REQUEST_COUNT = Counter(
    "blazerpc_requests_total",
    "Total number of gRPC requests",
    ["method", "status"],
)

_REQUEST_DURATION = Histogram(
    "blazerpc_request_duration_seconds",
    "Request duration in seconds",
    ["method"],
)


@dataclass
class _RequestTiming:
    method: str
    start: float
    status: Status | None = None


class _MetricsLifecycle(Middleware):
    def __init__(self) -> None:
        self._current_request: ContextVar[_RequestTiming | None] = ContextVar(
            f"{type(self).__name__}_current_request", default=None
        )

    async def on_request(self, event: RecvRequest) -> None:
        timing = _RequestTiming(event.method_name, time.perf_counter())
        method_func = event.method_func

        async def wrapped_method(stream: Stream[Any, Any]) -> None:
            token = self._current_request.set(timing)
            try:
                await method_func(stream)
            except GRPCError as exc:
                timing.status = timing.status or exc.status
                raise
            except asyncio.CancelledError:
                timing.status = timing.status or Status.CANCELLED
                raise
            except Exception:
                timing.status = timing.status or Status.UNKNOWN
                raise
            finally:
                try:
                    status = timing.status or Status.OK
                    self._record(
                        timing.method,
                        status,
                        time.perf_counter() - timing.start,
                    )
                finally:
                    self._current_request.reset(token)

        event.method_func = wrapped_method

    async def on_response(self, event: SendTrailingMetadata) -> None:
        timing = self._current_request.get()
        if timing is not None:
            timing.status = event.status

    @abstractmethod
    def _record(self, method: str, status: Status, duration: float) -> None:
        """Record metrics for one completed RPC."""


class MetricsMiddleware(_MetricsLifecycle):
    """Collects Prometheus metrics for every RPC call.

    Exported metrics:

    - ``blazerpc_requests_total{method, status}``
    - ``blazerpc_request_duration_seconds{method}``
    """

    def __init__(self) -> None:
        super().__init__()

    def _record(self, method: str, status: Status, duration: float) -> None:
        _REQUEST_COUNT.labels(method=method, status=str(status.value)).inc()
        _REQUEST_DURATION.labels(method=method).observe(duration)


# ---------------------------------------------------------------------------
# Metrics middleware (OpenTelemetry)
# ---------------------------------------------------------------------------


class OTelMetricsMiddleware(_MetricsLifecycle):
    """Pushes RPC metrics via the OpenTelemetry Metrics API.

    Exported instruments:

    - ``blazerpc.rpc.count`` – Counter with attributes ``method``, ``status``
    - ``blazerpc.rpc.duration`` – Histogram (seconds) with attribute ``method``

    Pass a custom :class:`opentelemetry.metrics.Meter` to control which
    ``MeterProvider`` (and therefore which exporter) is used.  When *meter*
    is ``None`` the global meter provider is used.
    """

    def __init__(self, meter: otel_metrics.Meter | None = None) -> None:
        super().__init__()
        if meter is None:
            m = otel_metrics.get_meter("blazerpc")
        else:
            m = meter
        self._request_count = m.create_counter(
            "blazerpc.rpc.count",
            description="Total number of gRPC requests",
        )
        self._request_duration = m.create_histogram(
            "blazerpc.rpc.duration",
            unit="s",
            description="RPC request duration in seconds",
        )

    def _record(self, method: str, status: Status, duration: float) -> None:
        self._request_count.add(1, {"method": method, "status": str(status.value)})
        self._request_duration.record(duration, {"method": method})


# ---------------------------------------------------------------------------
# Exception-mapping middleware
# ---------------------------------------------------------------------------


class ExceptionMiddleware(Middleware):
    """Maps Python exceptions to gRPC status codes.

    This middleware is a no-op on the event level -- the actual mapping
    is handled inside the servicer handlers.  It exists as a base for
    users who want to attach custom exception-to-status mappings via
    subclassing.
    """

    async def on_request(self, event: RecvRequest) -> None:
        pass

    async def on_response(self, event: SendTrailingMetadata) -> None:
        pass


# ---------------------------------------------------------------------------
# Transport-agnostic middleware (for JSON-RPC and future transports)
# ---------------------------------------------------------------------------


class RequestInfo:
    """Transport-agnostic request metadata."""

    __slots__ = ("method", "peer", "headers", "transport")

    def __init__(
        self,
        method: str,
        peer: str,
        headers: dict[str, str],
        transport: str,
    ) -> None:
        self.method = method
        self.peer = peer
        self.headers = headers
        self.transport = transport


class ResponseInfo:
    """Transport-agnostic response metadata."""

    __slots__ = ("method", "status", "duration", "transport")

    def __init__(
        self, status: int, duration: float, transport: str, method: str = ""
    ) -> None:
        self.method = method
        self.status = status
        self.duration = duration
        self.transport = transport


class TransportMiddleware(ABC):
    """Base class for transport-agnostic middleware.

    Works with JSON-RPC and any future transports, unlike
    :class:`Middleware` which is gRPC-specific.
    """

    @abstractmethod
    async def on_request(self, info: RequestInfo) -> None:
        """Called when a request is received."""

    @abstractmethod
    async def on_response(self, info: ResponseInfo) -> None:
        """Called when a response is about to be sent."""


class TransportLoggingMiddleware(TransportMiddleware):
    """Logs each request with method name, peer, and transport."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or log

    async def on_request(self, info: RequestInfo) -> None:
        self._log.info(
            "[%s] Request: %s from %s", info.transport, info.method, info.peer
        )

    async def on_response(self, info: ResponseInfo) -> None:
        self._log.info(
            "[%s] Response: status=%s duration=%.3fs",
            info.transport,
            info.status,
            info.duration,
        )


class TransportMetricsMiddleware(TransportMiddleware):
    """Prometheus metrics with a ``transport`` label dimension."""

    _COUNTER = Counter(
        "blazerpc_transport_requests_total",
        "Total requests by transport",
        ["method", "status", "transport"],
    )
    _HISTOGRAM = Histogram(
        "blazerpc_transport_request_duration_seconds",
        "Request duration by transport",
        ["method", "transport"],
    )

    async def on_request(self, info: RequestInfo) -> None:
        pass  # timing is handled at the server level

    async def on_response(self, info: ResponseInfo) -> None:
        self._COUNTER.labels(
            method=info.method, status=str(info.status), transport=info.transport
        ).inc()
        self._HISTOGRAM.labels(method=info.method, transport=info.transport).observe(
            info.duration
        )

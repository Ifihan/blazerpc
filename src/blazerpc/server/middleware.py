"""Middleware (interceptors) for BlazeRPC servers.

Built on top of grpclib's event system.  Each middleware is an async
callback that hooks into ``RecvRequest`` / ``SendTrailingMetadata``
events.

Usage::

    from blazerpc.server.middleware import LoggingMiddleware, MetricsMiddleware

    server = GRPCServer(handlers)
    LoggingMiddleware().attach(grpclib_server)
    MetricsMiddleware().attach(grpclib_server)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from grpclib.events import RecvRequest, SendTrailingMetadata, listen
from grpclib.server import Server
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
        self._log.info(
            "RPC request: %s from %s", event.method_name, peer
        )

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


class MetricsMiddleware(Middleware):
    """Collects Prometheus metrics for every RPC call.

    Exported metrics:

    - ``blazerpc_requests_total{method, status}``
    - ``blazerpc_request_duration_seconds{method}``
    """

    def __init__(self) -> None:
        self._timings: dict[int, tuple[str, float]] = {}

    async def on_request(self, event: RecvRequest) -> None:
        # Store start time keyed by id of the event's metadata object
        # (unique per request).
        key = id(event.metadata)
        self._timings[key] = (event.method_name, time.perf_counter())

    async def on_response(self, event: SendTrailingMetadata) -> None:
        key = id(event.metadata)
        entry = self._timings.pop(key, None)
        if entry is None:
            return
        method, start = entry
        duration = time.perf_counter() - start
        status_str = str(event.status.value) if event.status else "0"
        _REQUEST_COUNT.labels(method=method, status=status_str).inc()
        _REQUEST_DURATION.labels(method=method).observe(duration)


# ---------------------------------------------------------------------------
# Exception-mapping middleware
# ---------------------------------------------------------------------------


class ExceptionMiddleware(Middleware):
    """Maps Python exceptions to gRPC status codes.

    This middleware is a no-op on the event level — the actual mapping
    is handled inside the servicer handlers (see
    :mod:`blazerpc.codegen.servicer`).  It exists as a placeholder for
    users who want to attach custom exception-to-status mappings via
    subclassing.
    """

    async def on_request(self, event: RecvRequest) -> None:
        pass

    async def on_response(self, event: SendTrailingMetadata) -> None:
        pass

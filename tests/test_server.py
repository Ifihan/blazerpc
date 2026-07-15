"""Tests for gRPC server."""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest
from grpclib.server import Server

from blazerpc.app import BlazeApp
from blazerpc.codegen.servicer import build_servicer
from blazerpc.server.grpc import GRPCServer
from blazerpc.server.middleware import LoggingMiddleware

grpc_module = importlib.import_module("blazerpc.server.grpc")


def _make_server() -> tuple[BlazeApp, GRPCServer]:
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    servicer = build_servicer(app.registry)
    server = GRPCServer([servicer])
    return app, server


@pytest.mark.asyncio
async def test_server_start_and_stop() -> None:
    """Server can start on a port and be stopped programmatically."""
    _, server = _make_server()

    # Start the underlying grpclib server without blocking on signals.
    grpc_server = Server([build_servicer(BlazeApp(enable_batching=False).registry)])

    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    servicer = build_servicer(app.registry)
    grpc_server = Server([servicer])
    await grpc_server.start("127.0.0.1", 0)  # port 0 = random available port
    grpc_server.close()
    await grpc_server.wait_closed()


@pytest.mark.asyncio
async def test_grpc_server_stop_when_not_started() -> None:
    """Calling stop() before start() should be a no-op."""
    _, server = _make_server()
    await server.stop()  # should not raise


def test_grpc_server_accepts_middleware() -> None:
    """GRPCServer should accept a middleware list."""
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    servicer = build_servicer(app.registry)
    mw = LoggingMiddleware()
    server = GRPCServer([servicer], middleware=[mw])

    assert server._middleware == [mw]


@pytest.mark.parametrize("stage", ["attach", "bind"])
@pytest.mark.asyncio
async def test_grpc_server_cleans_partial_server_and_preserves_failure(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    instances: list[Any] = []

    class UnderlyingServer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            instances.append(self)
            self.closed = False

        async def start(self, host: str, port: int) -> None:
            if stage == "bind":
                raise RuntimeError("injected failure")

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            if stage == "attach":
                raise RuntimeError("cleanup failure")

    class Middleware:
        def attach(self, server: Any) -> None:
            if stage == "attach":
                raise RuntimeError("injected failure")

    monkeypatch.setattr(grpc_module, "Server", UnderlyingServer)
    server = GRPCServer([], middleware=[Middleware()])

    with pytest.raises(RuntimeError, match="injected failure"):
        await server.start(handle_signals=False)

    assert instances[0].closed
    assert server._server is None


@pytest.mark.asyncio
async def test_grpc_stop_is_idempotent_during_concurrent_calls() -> None:
    calls = 0

    class UnderlyingServer:
        def close(self) -> None:
            nonlocal calls
            calls += 1

        async def wait_closed(self) -> None:
            await asyncio.sleep(0)

    _, server = _make_server()
    server._server = UnderlyingServer()  # type: ignore[assignment]
    await asyncio.gather(server.stop(), server.stop(), server.stop())

    assert calls == 1


@pytest.mark.asyncio
async def test_grpc_stop_bounds_active_requests_by_grace_period() -> None:
    cancelled = asyncio.Event()

    class UnderlyingServer:
        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    app = BlazeApp(enable_batching=False)
    server = GRPCServer([build_servicer(app.registry)], grace_period=0.01)
    server._server = UnderlyingServer()  # type: ignore[assignment]
    await server.stop()

    assert cancelled.is_set()
    assert server._server is None

"""Tests for gRPC server."""

from __future__ import annotations

import asyncio

import pytest

from blazerpc.codegen.servicer import build_servicer
from blazerpc.app import BlazeApp
from blazerpc.server.grpc import GRPCServer


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
    from grpclib.server import Server

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

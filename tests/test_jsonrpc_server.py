"""Tests for the JSON-RPC HTTP server lifecycle."""

from __future__ import annotations

import pytest

pytest.importorskip("aiohttp", reason="aiohttp required for JSON-RPC tests")

from blazerpc import BlazeApp
from blazerpc.codegen.jsonrpc_handler import JsonRpcDispatcher
from blazerpc.server.jsonrpc import JsonRpcServer


async def test_server_stop_when_not_started() -> None:
    """Stopping a server that was never started should be a no-op."""
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    dispatcher = JsonRpcDispatcher(app.registry)
    server = JsonRpcServer(dispatcher)
    await server.stop()  # should not raise


async def test_server_constructor_accepts_middleware() -> None:
    from blazerpc.server.middleware import TransportLoggingMiddleware

    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    dispatcher = JsonRpcDispatcher(app.registry)
    mw = TransportLoggingMiddleware()
    server = JsonRpcServer(dispatcher, middleware=[mw])
    assert len(server._middleware) == 1

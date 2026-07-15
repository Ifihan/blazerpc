"""Tests for the JSON-RPC HTTP server lifecycle."""

from __future__ import annotations

import asyncio
from typing import Any

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


async def test_jsonrpc_stop_uses_grace_period_and_is_idempotent() -> None:
    app = BlazeApp(enable_batching=False)
    server = JsonRpcServer(JsonRpcDispatcher(app.registry), grace_period=0.01)
    calls = 0
    cancelled = asyncio.Event()

    class Runner:
        async def cleanup(self) -> None:
            nonlocal calls
            calls += 1
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    server._runner = Runner()  # type: ignore[assignment]
    await asyncio.gather(server.stop(), server.stop(), server.stop())

    assert calls == 1
    assert cancelled.is_set()
    assert server._runner is None


@pytest.mark.parametrize("stage", ["setup", "site", "bind"])
async def test_jsonrpc_start_cleans_partial_runner_and_preserves_failure(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    app = BlazeApp(enable_batching=False)
    runners: list[Any] = []

    class Runner:
        def __init__(self, application: Any, **kwargs: Any) -> None:
            runners.append(self)
            self.cleaned = False
            assert kwargs["shutdown_timeout"] == 0.1

        async def setup(self) -> None:
            if stage == "setup":
                raise RuntimeError("injected failure")

        async def cleanup(self) -> None:
            self.cleaned = True
            if stage == "setup":
                raise RuntimeError("cleanup failure")

    class Site:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if stage == "site":
                raise RuntimeError("injected failure")

        async def start(self) -> None:
            if stage == "bind":
                raise RuntimeError("injected failure")

    monkeypatch.setattr("blazerpc.server.jsonrpc.web.AppRunner", Runner)
    monkeypatch.setattr("blazerpc.server.jsonrpc.web.TCPSite", Site)
    server = JsonRpcServer(JsonRpcDispatcher(app.registry), grace_period=0.1)

    with pytest.raises(RuntimeError, match="injected failure"):
        await server.start(handle_signals=False)

    assert runners[0].cleaned
    assert server._runner is None

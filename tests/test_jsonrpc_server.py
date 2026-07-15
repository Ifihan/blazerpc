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


async def test_jsonrpc_concurrent_stop_callers_await_the_same_cleanup() -> None:
    app = BlazeApp(enable_batching=False)
    server = JsonRpcServer(JsonRpcDispatcher(app.registry), grace_period=0.01)
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    class Runner:
        async def cleanup(self) -> None:
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()

    server._runner = Runner()  # type: ignore[assignment]
    callers = [asyncio.create_task(server.stop()) for _ in range(3)]
    await started.wait()

    assert calls == 1
    assert not any(caller.done() for caller in callers)
    release.set()
    await asyncio.gather(*callers)
    assert server._runner is None


async def test_jsonrpc_outer_timeout_does_not_cancel_cleanup() -> None:
    app = BlazeApp(enable_batching=False)
    server = JsonRpcServer(JsonRpcDispatcher(app.registry), grace_period=0.01)
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = False

    class Runner:
        async def cleanup(self) -> None:
            nonlocal cancelled
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled = True
                raise

    server._runner = Runner()  # type: ignore[assignment]
    completing_caller = asyncio.create_task(server.stop())
    await started.wait()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(server.stop(), timeout=0.01)

    assert not cancelled
    assert not completing_caller.done()
    release.set()
    await completing_caller
    assert server._stop_task is not None
    assert server._stop_task.done()


async def test_jsonrpc_start_waits_for_prior_stop_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = BlazeApp(enable_batching=False)
    server = JsonRpcServer(JsonRpcDispatcher(app.registry))
    old_cleanup_started = asyncio.Event()
    release_old_cleanup = asyncio.Event()
    new_runner_setup = asyncio.Event()
    cleanups: list[str] = []

    class OldRunner:
        async def cleanup(self) -> None:
            old_cleanup_started.set()
            await release_old_cleanup.wait()
            cleanups.append("old")

    class NewRunner:
        def __init__(self, application: Any, **kwargs: Any) -> None:
            pass

        async def setup(self) -> None:
            new_runner_setup.set()

        async def cleanup(self) -> None:
            cleanups.append("new")

    class Site:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def start(self) -> None:
            pass

    server._runner = OldRunner()  # type: ignore[assignment]
    stopping = asyncio.create_task(server.stop())
    await old_cleanup_started.wait()
    monkeypatch.setattr("blazerpc.server.jsonrpc.web.AppRunner", NewRunner)
    monkeypatch.setattr("blazerpc.server.jsonrpc.web.TCPSite", Site)

    starting = asyncio.create_task(server.start(handle_signals=False))
    await asyncio.sleep(0)
    assert not new_runner_setup.is_set()

    release_old_cleanup.set()
    await stopping
    await new_runner_setup.wait()
    assert cleanups == ["old"]

    await server.stop()
    await starting
    assert cleanups == ["old", "new"]


async def test_jsonrpc_cancellation_completes_middleware_lifecycle() -> None:
    from blazerpc.server.middleware import (
        RequestInfo,
        ResponseInfo,
        TransportMiddleware,
    )

    app = BlazeApp(enable_batching=False)
    entered = asyncio.Event()
    responses: list[ResponseInfo] = []

    @app.model("wait")
    async def wait(value: str) -> str:
        entered.set()
        await asyncio.Event().wait()
        return value

    class RecordingMiddleware(TransportMiddleware):
        async def on_request(self, info: RequestInfo) -> None:
            pass

        async def on_response(self, info: ResponseInfo) -> None:
            responses.append(info)

    server = JsonRpcServer(
        JsonRpcDispatcher(app.registry), middleware=[RecordingMiddleware()]
    )
    task = asyncio.create_task(
        server._dispatch_jsonrpc(
            {
                "jsonrpc": "2.0",
                "method": "predict.wait",
                "params": {"value": "unreachable"},
                "id": 1,
            },
            peer="",
            headers={},
        )
    )
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(responses) == 1
    assert responses[0].method == "predict.wait"
    assert responses[0].status == 499


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

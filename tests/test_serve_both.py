"""Focused tests for dual-transport lifecycle coordination."""

from __future__ import annotations

import asyncio
import importlib
import signal
from typing import Any

import pytest

from blazerpc.app import BlazeApp

app_module = importlib.import_module("blazerpc.app")
jsonrpc_module = importlib.import_module("blazerpc.server.jsonrpc")


class _FakeServer:
    instances: list[_FakeServer]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.instances.append(self)
        self.args = args
        self.kwargs = kwargs
        self.started = asyncio.Event()
        self.shutdown = asyncio.Event()
        self.stopped = False
        self.handle_signals: bool | None = None

    async def start(
        self, host: str, port: int, *, handle_signals: bool = True
    ) -> None:
        self.handle_signals = handle_signals
        self.started.set()
        await self.shutdown.wait()

    async def stop(self) -> None:
        self.stopped = True
        self.shutdown.set()


class _FakeGrpcServer(_FakeServer):
    instances: list[_FakeServer] = []


class _FakeJsonRpcServer(_FakeServer):
    instances: list[_FakeServer] = []


@pytest.fixture
def fake_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeGrpcServer.instances.clear()
    _FakeJsonRpcServer.instances.clear()
    monkeypatch.setattr(app_module, "GRPCServer", _FakeGrpcServer)
    monkeypatch.setattr(jsonrpc_module, "JsonRpcServer", _FakeJsonRpcServer)


async def _wait_until_started(*servers: _FakeServer) -> None:
    await asyncio.wait_for(
        asyncio.gather(*(server.started.wait() for server in servers)), timeout=1
    )


def _make_app() -> BlazeApp:
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    return app


async def test_jsonrpc_middleware_is_wired_to_both_transport_modes(
    monkeypatch: pytest.MonkeyPatch, fake_servers: None
) -> None:
    from blazerpc.server.middleware import TransportLoggingMiddleware

    middleware = TransportLoggingMiddleware()
    app = _make_app()
    app.jsonrpc_middleware = [middleware]

    jsonrpc_task = asyncio.create_task(app.serve_jsonrpc())
    async def wait_for_jsonrpc() -> None:
        while not _FakeJsonRpcServer.instances:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_jsonrpc(), timeout=1)
    jsonrpc = _FakeJsonRpcServer.instances[0]
    await jsonrpc.stop()
    await asyncio.wait_for(jsonrpc_task, timeout=1)
    assert jsonrpc.kwargs["middleware"] == [middleware]

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, callback: None)
    monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: True)
    both_task = asyncio.create_task(app.serve_both())
    await _wait_for_instances(both_task, run=0)
    grpc = _FakeGrpcServer.instances[0]
    both_jsonrpc = _FakeJsonRpcServer.instances[1]
    await _wait_until_started(grpc, both_jsonrpc)
    await both_jsonrpc.stop()
    await asyncio.wait_for(both_task, timeout=1)

    assert both_jsonrpc.kwargs["middleware"] == [middleware]
    assert grpc.kwargs["middleware"] == []


async def _wait_for_instances(task: asyncio.Task[None], run: int = 0) -> None:
    async def wait() -> None:
        while (
            len(_FakeGrpcServer.instances) <= run
            or len(_FakeJsonRpcServer.instances) <= run
        ):
            if task.done():
                task.result()
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)


async def test_serve_both_owns_signals_and_stops_both_transports(
    monkeypatch: pytest.MonkeyPatch, fake_servers: None
) -> None:
    loop = asyncio.get_running_loop()
    handlers: dict[signal.Signals, Any] = {}
    removed: list[signal.Signals] = []
    monkeypatch.setattr(
        loop,
        "add_signal_handler",
        lambda sig, callback: handlers.__setitem__(sig, callback),
    )
    monkeypatch.setattr(
        loop, "remove_signal_handler", lambda sig: removed.append(sig) or True
    )

    task = asyncio.create_task(_make_app().serve_both())
    await _wait_for_instances(task)
    grpc = _FakeGrpcServer.instances[0]
    jsonrpc = _FakeJsonRpcServer.instances[0]
    await _wait_until_started(grpc, jsonrpc)

    assert set(handlers) == {signal.SIGINT, signal.SIGTERM}
    assert grpc.handle_signals is False
    assert jsonrpc.handle_signals is False

    handlers[signal.SIGTERM]()
    await asyncio.wait_for(task, timeout=1)

    assert grpc.stopped
    assert jsonrpc.stopped
    assert removed == [signal.SIGINT, signal.SIGTERM]


async def test_serve_both_can_be_programmatically_stopped_and_restarted(
    monkeypatch: pytest.MonkeyPatch, fake_servers: None
) -> None:
    loop = asyncio.get_running_loop()
    added: list[signal.Signals] = []
    removed: list[signal.Signals] = []
    monkeypatch.setattr(
        loop,
        "add_signal_handler",
        lambda sig, callback: added.append(sig),
    )
    monkeypatch.setattr(
        loop, "remove_signal_handler", lambda sig: removed.append(sig) or True
    )
    app = _make_app()

    for run in range(2):
        task = asyncio.create_task(app.serve_both())
        await _wait_for_instances(task, run)
        grpc = _FakeGrpcServer.instances[run]
        jsonrpc = _FakeJsonRpcServer.instances[run]
        await _wait_until_started(grpc, jsonrpc)

        await grpc.stop()
        await asyncio.wait_for(task, timeout=1)

        assert grpc.stopped
        assert jsonrpc.stopped

    expected = [signal.SIGINT, signal.SIGTERM] * 2
    assert added == expected
    assert removed == expected


async def test_serve_both_cleans_up_after_startup_failure(
    monkeypatch: pytest.MonkeyPatch, fake_servers: None
) -> None:
    class FailingGrpcServer(_FakeGrpcServer):
        async def start(
            self, host: str, port: int, *, handle_signals: bool = True
        ) -> None:
            self.handle_signals = handle_signals
            self.started.set()
            raise RuntimeError("failed to bind")

    loop = asyncio.get_running_loop()
    removed: list[signal.Signals] = []
    monkeypatch.setattr(app_module, "GRPCServer", FailingGrpcServer)
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, callback: None)
    monkeypatch.setattr(
        loop, "remove_signal_handler", lambda sig: removed.append(sig) or True
    )

    with pytest.raises(RuntimeError, match="failed to bind"):
        await asyncio.wait_for(
            _make_app().serve_both(), timeout=1
        )

    grpc = FailingGrpcServer.instances[0]
    jsonrpc = _FakeJsonRpcServer.instances[0]
    assert grpc.stopped
    assert jsonrpc.stopped
    assert removed == [signal.SIGINT, signal.SIGTERM]


async def test_serve_both_tolerates_unsupported_signal_handlers(
    monkeypatch: pytest.MonkeyPatch, fake_servers: None
) -> None:
    loop = asyncio.get_running_loop()
    removed: list[signal.Signals] = []

    def unsupported(sig: signal.Signals, callback: Any) -> None:
        raise NotImplementedError

    monkeypatch.setattr(loop, "add_signal_handler", unsupported)
    monkeypatch.setattr(
        loop, "remove_signal_handler", lambda sig: removed.append(sig) or True
    )

    task = asyncio.create_task(_make_app().serve_both())
    await _wait_for_instances(task)
    grpc = _FakeGrpcServer.instances[0]
    jsonrpc = _FakeJsonRpcServer.instances[0]
    await _wait_until_started(grpc, jsonrpc)

    await jsonrpc.stop()
    await asyncio.wait_for(task, timeout=1)

    assert grpc.stopped
    assert jsonrpc.stopped
    assert removed == []

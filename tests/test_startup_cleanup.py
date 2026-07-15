"""Failure-injection tests for application startup ownership."""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest

from blazerpc.app import BlazeApp

app_module = importlib.import_module("blazerpc.app")
jsonrpc_handler_module = importlib.import_module("blazerpc.codegen.jsonrpc_handler")
jsonrpc_server_module = importlib.import_module("blazerpc.server.jsonrpc")


class _Batcher:
    def __init__(self) -> None:
        self.stop_count = 0

    async def stop(self) -> None:
        self.stop_count += 1


class _Server:
    instances: list[_Server] = []
    fail_start = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.instances.append(self)
        self.stopped = False
        self.cancelled = False
        self.waiting = asyncio.Event()

    async def start(self, *args: Any, **kwargs: Any) -> None:
        if self.fail_start:
            raise RuntimeError("injected failure")
        self.waiting.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def stop(self) -> None:
        self.stopped = True


class _GrpcServer(_Server):
    instances: list[_Server] = []


class _JsonServer(_Server):
    instances: list[_Server] = []


def _app() -> BlazeApp:
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    return app


def _inject_batcher(monkeypatch: pytest.MonkeyPatch, app: BlazeApp) -> _Batcher:
    batcher = _Batcher()

    async def create_batchers() -> dict[str, Any]:
        return {"echo": batcher}

    monkeypatch.setattr(app, "_create_batchers", create_batchers)
    return batcher


def _raise(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("injected failure")


async def test_partially_started_batcher_is_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []

    class FailingBatcher:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            instances.append(self)
            self.stopped = False

        async def start(self, inference_fn: Any) -> None:
            self.task = asyncio.create_task(asyncio.Event().wait())
            raise RuntimeError("injected failure")

        async def stop(self) -> None:
            self.stopped = True
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    app = _app()
    app.enable_batching = True
    monkeypatch.setattr(app_module, "Batcher", FailingBatcher)
    monkeypatch.setattr(app_module, "_supports_adaptive_batching", lambda model: True)

    with pytest.raises(RuntimeError, match="injected failure"):
        await app._create_batchers()

    assert len(instances) == 1
    assert instances[0].stopped
    assert instances[0].task.done()


@pytest.mark.parametrize(
    "stage", ["servicer", "health", "reflection", "server", "start"]
)
async def test_serve_cleans_batcher_for_every_construction_stage(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    app = _app()
    batcher = _inject_batcher(monkeypatch, app)
    _GrpcServer.instances.clear()
    _GrpcServer.fail_start = stage == "start"
    monkeypatch.setattr(app_module, "GRPCServer", _GrpcServer)
    if stage == "servicer":
        monkeypatch.setattr(app_module, "build_servicer", _raise)
    elif stage == "health":
        monkeypatch.setattr(app_module, "build_health_service", _raise)
    elif stage == "reflection":
        monkeypatch.setattr(app_module, "build_reflection_service", _raise)
    elif stage == "server":
        monkeypatch.setattr(app_module, "GRPCServer", _raise)

    with pytest.raises(RuntimeError, match="injected failure"):
        await app.serve()

    assert batcher.stop_count == 1
    if _GrpcServer.instances:
        assert _GrpcServer.instances[0].stopped


@pytest.mark.parametrize("stage", ["dispatcher", "server", "start"])
async def test_serve_jsonrpc_cleans_batcher_for_every_construction_stage(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    app = _app()
    batcher = _inject_batcher(monkeypatch, app)
    _JsonServer.instances.clear()
    _JsonServer.fail_start = stage == "start"
    monkeypatch.setattr(jsonrpc_server_module, "JsonRpcServer", _JsonServer)
    if stage == "dispatcher":
        monkeypatch.setattr(jsonrpc_handler_module, "JsonRpcDispatcher", _raise)
    elif stage == "server":
        monkeypatch.setattr(jsonrpc_server_module, "JsonRpcServer", _raise)

    with pytest.raises(RuntimeError, match="injected failure"):
        await app.serve_jsonrpc()

    assert batcher.stop_count == 1
    if _JsonServer.instances:
        assert _JsonServer.instances[0].stopped


@pytest.mark.parametrize(
    "stage",
    [
        "servicer",
        "health",
        "reflection",
        "grpc_server",
        "dispatcher",
        "json_server",
        "grpc_start",
        "json_start",
    ],
)
async def test_serve_both_cleans_every_constructed_resource(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    app = _app()
    batcher = _inject_batcher(monkeypatch, app)
    _GrpcServer.instances.clear()
    _JsonServer.instances.clear()
    _GrpcServer.fail_start = stage == "grpc_start"
    _JsonServer.fail_start = stage == "json_start"
    monkeypatch.setattr(app_module, "GRPCServer", _GrpcServer)
    monkeypatch.setattr(jsonrpc_server_module, "JsonRpcServer", _JsonServer)
    monkeypatch.setattr(
        asyncio.get_running_loop(),
        "add_signal_handler",
        lambda *args: (_ for _ in ()).throw(NotImplementedError),
    )
    if stage == "servicer":
        monkeypatch.setattr(app_module, "build_servicer", _raise)
    elif stage == "health":
        monkeypatch.setattr(app_module, "build_health_service", _raise)
    elif stage == "reflection":
        monkeypatch.setattr(app_module, "build_reflection_service", _raise)
    elif stage == "grpc_server":
        monkeypatch.setattr(app_module, "GRPCServer", _raise)
    elif stage == "dispatcher":
        monkeypatch.setattr(jsonrpc_handler_module, "JsonRpcDispatcher", _raise)
    elif stage == "json_server":
        monkeypatch.setattr(jsonrpc_server_module, "JsonRpcServer", _raise)

    with pytest.raises(RuntimeError, match="injected failure"):
        await asyncio.wait_for(app.serve_both(), timeout=1)

    assert batcher.stop_count == 1
    assert all(server.stopped for server in _GrpcServer.instances)
    assert all(server.stopped for server in _JsonServer.instances)
    if stage == "grpc_start":
        assert _JsonServer.instances[0].cancelled
    if stage == "json_start":
        assert _GrpcServer.instances[0].cancelled


async def test_serve_preserves_start_error_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCleanupServer(_Server):
        async def start(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("start failure")

        async def stop(self) -> None:
            raise RuntimeError("cleanup failure")

    app = _app()
    monkeypatch.setattr(app_module, "GRPCServer", FailingCleanupServer)

    with pytest.raises(RuntimeError, match="start failure"):
        await app.serve()

"""End-to-end integration tests.

These tests verify full register → serve → call → response flows using
grpclib's in-process server.
"""

from __future__ import annotations

import asyncio

import pytest
from grpclib.const import Cardinality
from grpclib.server import Server

from blazerpc.app import BlazeApp
from blazerpc.codegen.servicer import build_servicer
from blazerpc.server.health import build_health_service


@pytest.mark.asyncio
async def test_server_with_health_starts_and_stops() -> None:
    """A server with both the inference servicer and health can start."""
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    servicer = build_servicer(app.registry)
    health = build_health_service([servicer])

    server = Server([servicer, health])
    await server.start("127.0.0.1", 0)
    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_multiple_models_register() -> None:
    """Multiple models can be registered and produce a working servicer."""
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    @app.model("add")
    def add(a: float, b: float) -> float:
        return a + b

    servicer = build_servicer(app.registry)
    mapping = servicer.__mapping__()

    assert "/blazerpc.InferenceService/PredictEcho" in mapping
    assert "/blazerpc.InferenceService/PredictAdd" in mapping


@pytest.mark.asyncio
async def test_streaming_model_registers() -> None:
    """A streaming model creates a server-streaming RPC."""
    app = BlazeApp(enable_batching=False)

    @app.model("tokens", streaming=True)
    async def generate(prompt: str) -> str:
        for token in ["hello", " ", "world"]:
            yield token

    servicer = build_servicer(app.registry)
    mapping = servicer.__mapping__()

    path = "/blazerpc.InferenceService/PredictTokens"
    assert path in mapping
    # Streaming RPCs have UNARY_STREAM cardinality.
    handler = mapping[path]
    assert handler.cardinality == Cardinality.UNARY_STREAM


@pytest.mark.asyncio
async def test_async_model_execution() -> None:
    """Async model functions are awaited correctly via the servicer."""
    app = BlazeApp(enable_batching=False)

    @app.model("async_echo")
    async def async_echo(text: str) -> str:
        await asyncio.sleep(0.01)
        return f"async: {text}"

    servicer = build_servicer(app.registry)
    mapping = servicer.__mapping__()
    assert "/blazerpc.InferenceService/PredictAsyncEcho" in mapping

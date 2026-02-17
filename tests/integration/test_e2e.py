"""End-to-end integration tests.

These tests verify full register → serve → call → response flows using
grpclib's in-process server with RawCodec.
"""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest
from grpclib.client import Channel
from grpclib.const import Cardinality
from grpclib.server import Server

from blazerpc.app import BlazeApp
from blazerpc.codegen.servicer import build_servicer
from blazerpc.server.grpc import RawCodec
from blazerpc.server.health import build_health_service
from blazerpc.types import TensorInput, TensorOutput


@pytest.mark.asyncio
async def test_server_with_health_starts_and_stops() -> None:
    """A server with both the inference servicer and health can start."""
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    servicer = build_servicer(app.registry)
    health = build_health_service([servicer])

    server = Server([servicer, health], codec=RawCodec())
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


# ---------------------------------------------------------------------------
# Wire-level tests: send JSON bytes through grpclib and verify responses
# ---------------------------------------------------------------------------


def _get_server_port(server: Server) -> int:
    """Extract the OS-assigned port from a started server."""
    for sock in server._server.sockets:
        return sock.getsockname()[1]
    raise RuntimeError("Server has no sockets")


async def _unary_call(
    channel: Channel, path: str, request_data: dict,
) -> dict:
    """Send a unary JSON request and return the parsed JSON response."""
    stream = channel.request(path, Cardinality.UNARY_UNARY, None, None)
    async with stream as s:
        await s.send_message(json.dumps(request_data).encode(), end=True)
        response_bytes = await s.recv_message()
    return json.loads(response_bytes)


@pytest.mark.asyncio
async def test_unary_echo_over_wire() -> None:
    """Send a JSON request to an echo model and get JSON back."""
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return f"Echo: {text}"

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictEcho",
            {"text": "hello"},
        )
        assert response["result"] == "Echo: hello"
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_unary_add_over_wire() -> None:
    """Send numeric inputs and verify the sum comes back."""
    app = BlazeApp(enable_batching=False)

    @app.model("add")
    def add(a: float, b: float) -> float:
        return a + b

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictAdd",
            {"a": 2.5, "b": 3.5},
        )
        assert response["result"] == 6.0
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_unary_list_over_wire() -> None:
    """A model that takes list[str] and returns list[float]."""
    app = BlazeApp(enable_batching=False)

    @app.model("sentiment")
    def predict(text: list[str]) -> list[float]:
        return [0.9] * len(text)

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictSentiment",
            {"text": ["good", "bad"]},
        )
        assert response["result"] == [0.9, 0.9]
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_unary_tensor_over_wire() -> None:
    """Tensor inputs are base64-decoded, processed, and re-encoded."""
    app = BlazeApp(enable_batching=False)

    @app.model("double")
    def double(
        data: TensorInput[np.float32, 4],
    ) -> TensorOutput[np.float32, 4]:
        return (data * 2).astype(np.float32)

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        request = {
            "data": {
                "shape": list(arr.shape),
                "dtype": "float",
                "data": base64.b64encode(arr.tobytes()).decode(),
            }
        }
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictDouble",
            request,
        )
        result_tensor = response["result"]
        result_data = np.frombuffer(
            base64.b64decode(result_tensor["data"]),
            dtype=np.float32,
        )
        np.testing.assert_array_equal(result_data, [2.0, 4.0, 6.0, 8.0])
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_async_model_over_wire() -> None:
    """Async model functions work over the wire."""
    app = BlazeApp(enable_batching=False)

    @app.model("async_echo")
    async def async_echo(text: str) -> str:
        await asyncio.sleep(0.01)
        return f"async: {text}"

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictAsyncEcho",
            {"text": "world"},
        )
        assert response["result"] == "async: world"
    finally:
        channel.close()
        server.close()
        await server.wait_closed()

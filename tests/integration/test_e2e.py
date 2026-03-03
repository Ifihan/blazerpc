"""End-to-end integration tests.

These tests verify full register → serve → call → response flows using
grpclib's in-process server with real binary Protobuf encoding via betterproto.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from grpclib.client import Channel
from grpclib.const import Cardinality
from grpclib.server import Server

from blazerpc.app import BlazeApp
from blazerpc.codegen.proto_types import _TensorProtoMsg, build_message_classes
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
# Wire-level tests: send Protobuf bytes through grpclib and verify responses
# ---------------------------------------------------------------------------


def _get_server_port(server: Server) -> int:
    """Extract the OS-assigned port from a started server."""
    for sock in server._server.sockets:
        return sock.getsockname()[1]
    raise RuntimeError("Server has no sockets")


async def _unary_call(
    channel: Channel,
    path: str,
    request_bytes: bytes,
    response_cls: type,
) -> object:
    """Send a unary Protobuf request and return the parsed response message."""
    stream = channel.request(path, Cardinality.UNARY_UNARY, None, None)
    async with stream as s:
        await s.send_message(request_bytes, end=True)
        response_bytes = await s.recv_message()
    return response_cls().parse(response_bytes)


@pytest.mark.asyncio
async def test_unary_echo_over_wire() -> None:
    """Send a Protobuf request to an echo model and get Protobuf back."""
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return f"Echo: {text}"

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    model = app.registry.get("echo")
    req_cls, resp_cls = build_message_classes(model)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        request_bytes = bytes(req_cls(text="hello"))
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictEcho",
            request_bytes,
            resp_cls,
        )
        assert response.result == "Echo: hello"  # type: ignore[union-attr]
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

    model = app.registry.get("add")
    req_cls, resp_cls = build_message_classes(model)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        request_bytes = bytes(req_cls(a=2.5, b=3.5))
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictAdd",
            request_bytes,
            resp_cls,
        )
        assert abs(response.result - 6.0) < 1e-5  # type: ignore[union-attr]
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

    model = app.registry.get("sentiment")
    req_cls, resp_cls = build_message_classes(model)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        request_bytes = bytes(req_cls(text=["good", "bad"]))
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictSentiment",
            request_bytes,
            resp_cls,
        )
        assert len(response.result) == 2  # type: ignore[union-attr]
        assert all(abs(v - 0.9) < 1e-5 for v in response.result)  # type: ignore[union-attr]
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_unary_tensor_over_wire() -> None:
    """Tensor inputs are Protobuf-encoded, processed, and decoded back."""
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

    model = app.registry.get("double")
    req_cls, resp_cls = build_message_classes(model)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        tp = _TensorProtoMsg(shape=list(arr.shape), dtype="float", data=arr.tobytes())
        request_bytes = bytes(req_cls(data=tp))
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictDouble",
            request_bytes,
            resp_cls,
        )
        result_arr = np.frombuffer(
            response.result.data, dtype=np.float32  # type: ignore[union-attr]
        )
        np.testing.assert_array_equal(result_arr, [2.0, 4.0, 6.0, 8.0])
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

    model = app.registry.get("async_echo")
    req_cls, resp_cls = build_message_classes(model)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        request_bytes = bytes(req_cls(text="world"))
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictAsyncEcho",
            request_bytes,
            resp_cls,
        )
        assert response.result == "async: world"  # type: ignore[union-attr]
    finally:
        channel.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_unary_with_batching_over_wire() -> None:
    """Batched model works end-to-end with per-model batcher lifecycle."""
    from blazerpc.app import _make_batch_inference_fn
    from blazerpc.runtime.batcher import Batcher

    app = BlazeApp(enable_batching=True, max_batch_size=4)

    @app.model("add")
    def add(a: float, b: float) -> float:
        return a + b

    model = app.registry.get("add")
    batcher = Batcher(app.max_batch_size, app.batch_timeout_ms)
    await batcher.start(_make_batch_inference_fn(model))

    servicer = build_servicer(app.registry, batchers={"add": batcher})
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    req_cls, resp_cls = build_message_classes(model)

    channel = Channel("127.0.0.1", port, codec=RawCodec())
    try:
        request_bytes = bytes(req_cls(a=10.0, b=20.0))
        response = await _unary_call(
            channel,
            "/blazerpc.InferenceService/PredictAdd",
            request_bytes,
            resp_cls,
        )
        assert abs(response.result - 30.0) < 1e-5  # type: ignore[union-attr]
    finally:
        channel.close()
        server.close()
        await server.wait_closed()
        await batcher.stop()

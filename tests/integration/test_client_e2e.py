"""End-to-end integration tests for BlazeClient.

These tests spin up an in-process gRPC server and use BlazeClient
to make real calls over the wire using binary Protobuf encoding.
"""

from __future__ import annotations

import numpy as np
import pytest
from grpclib.server import Server

from blazerpc.app import BlazeApp
from blazerpc.client import BlazeClient
from blazerpc.codegen.servicer import build_servicer
from blazerpc.server.grpc import RawCodec
from blazerpc.types import TensorInput, TensorOutput


def _get_server_port(server: Server) -> int:
    """Extract the OS-assigned port from a started server."""
    for sock in server._server.sockets:
        return sock.getsockname()[1]
    raise RuntimeError("Server has no sockets")


@pytest.mark.asyncio
async def test_client_predict_echo() -> None:
    """BlazeClient.predict() works for a simple string model."""
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return f"Echo: {text}"

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    try:
        async with BlazeClient("127.0.0.1", port, registry=app.registry) as client:
            result = await client.predict("echo", text="hello")
            assert result == "Echo: hello"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_client_predict_add() -> None:
    """BlazeClient.predict() works for numeric inputs."""
    app = BlazeApp(enable_batching=False)

    @app.model("add")
    def add(a: float, b: float) -> float:
        return a + b

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    try:
        async with BlazeClient("127.0.0.1", port, registry=app.registry) as client:
            result = await client.predict("add", a=2.5, b=3.5)
            assert abs(result - 6.0) < 1e-5
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_client_predict_tensor_roundtrip() -> None:
    """BlazeClient converts tensor inputs and outputs across the wire."""
    app = BlazeApp(enable_batching=False)

    @app.model("double")
    def double(
        values: TensorInput[np.float32, 2, 2],
    ) -> TensorOutput[np.float32, 2, 2]:
        return values * 2

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    try:
        values = np.arange(4, dtype=np.float32).reshape(2, 2)
        async with BlazeClient("127.0.0.1", port, registry=app.registry) as client:
            result = await client.predict("double", values=values)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, values * 2)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_client_stream() -> None:
    """BlazeClient.stream() yields chunks from a streaming model."""
    app = BlazeApp(enable_batching=False)

    @app.model("tokens", streaming=True)
    async def generate(prompt: str) -> str:
        for token in ["hello", " ", "world"]:
            yield token

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    try:
        async with BlazeClient("127.0.0.1", port, registry=app.registry) as client:
            chunks = []
            async for chunk in client.stream("tokens", prompt="hi"):
                chunks.append(chunk)
            assert chunks == ["hello", " ", "world"]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_client_stream_tensor_input_and_output() -> None:
    """BlazeClient converts tensor input and every streamed tensor output."""
    app = BlazeApp(enable_batching=False)

    @app.model("tensor_chunks", streaming=True)
    async def tensor_chunks(
        values: TensorInput[np.int64, 3],
    ) -> TensorOutput[np.int64, 3]:
        yield values
        yield values + 1

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    try:
        values = np.array([1, 2, 3], dtype=np.int64)
        async with BlazeClient("127.0.0.1", port, registry=app.registry) as client:
            chunks = [
                chunk
                async for chunk in client.stream("tensor_chunks", values=values)
            ]

        assert all(isinstance(chunk, np.ndarray) for chunk in chunks)
        np.testing.assert_array_equal(chunks[0], values)
        np.testing.assert_array_equal(chunks[1], values + 1)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_client_context_manager() -> None:
    """BlazeClient works as an async context manager and cleans up."""
    async with BlazeClient("127.0.0.1", 50051) as client:
        assert client._channel is not None
    assert client._channel is None

"""End-to-end integration tests for BlazeClient.

These tests spin up an in-process gRPC server and use BlazeClient
to make real calls over the wire.
"""

from __future__ import annotations

import pytest
from grpclib.server import Server

from blazerpc.app import BlazeApp
from blazerpc.client import BlazeClient
from blazerpc.codegen.servicer import build_servicer
from blazerpc.server.grpc import RawCodec


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
        async with BlazeClient("127.0.0.1", port) as client:
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
        async with BlazeClient("127.0.0.1", port) as client:
            result = await client.predict("add", a=2.5, b=3.5)
            assert result == 6.0
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_client_stream() -> None:
    """BlazeClient.stream() yields chunks from a streaming model."""
    app = BlazeApp(enable_batching=False)

    @app.model("tokens", streaming=True)
    async def generate(prompt: str):
        for token in ["hello", " ", "world"]:
            yield token

    servicer = build_servicer(app.registry)
    server = Server([servicer], codec=RawCodec())
    await server.start("127.0.0.1", 0)
    port = _get_server_port(server)

    try:
        async with BlazeClient("127.0.0.1", port) as client:
            chunks = []
            async for chunk in client.stream("tokens", prompt="hi"):
                chunks.append(chunk)
            assert chunks == ["hello", " ", "world"]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_client_context_manager() -> None:
    """BlazeClient works as an async context manager and cleans up."""
    async with BlazeClient("127.0.0.1", 50051) as client:
        assert client._channel is not None
    assert client._channel is None

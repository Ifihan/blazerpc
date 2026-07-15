"""End-to-end tests for the JSON-RPC transport.

Spins up an in-process aiohttp server, sends JSON-RPC requests
via the JsonRpcClient, and verifies full round-trip behavior.
"""

# ruff: noqa: E402
from __future__ import annotations

import pytest

aiohttp = pytest.importorskip("aiohttp", reason="aiohttp required for JSON-RPC tests")

import numpy as np

from blazerpc import BlazeApp, Context, Depends
from blazerpc.codegen.jsonrpc_handler import JsonRpcDispatcher
from blazerpc.jsonrpc_client import JsonRpcClient
from blazerpc.server.jsonrpc import JsonRpcServer
from blazerpc.types import TensorInput, TensorOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _start_test_server(
    app: BlazeApp,
) -> tuple[JsonRpcServer, int]:
    """Start a JSON-RPC server on a random port and return (server, port)."""
    dispatcher = JsonRpcDispatcher(app.registry, app_state=app.state)
    server = JsonRpcServer(dispatcher)

    # Use aiohttp internals to bind to port 0 (OS-assigned)
    from aiohttp import web

    web_app = web.Application()
    web_app.router.add_post("/jsonrpc", server._handle_jsonrpc)
    web_app.router.add_post("/jsonrpc/stream/{method:.+}", server._handle_sse)
    web_app.router.add_get("/health", server._handle_health)

    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    # Extract the actual port
    port = site._server.sockets[0].getsockname()[1]
    server._runner = runner
    return server, port


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_echo_model_e2e() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return f"Echo: {text}"

    server, port = await _start_test_server(app)
    try:
        async with JsonRpcClient(f"http://127.0.0.1:{port}/jsonrpc") as client:
            result = await client.predict("echo", text="hello")
            assert result == "Echo: hello"
    finally:
        await server.stop()


async def test_add_model_e2e() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("add")
    def add(a: float, b: float) -> float:
        return a + b

    server, port = await _start_test_server(app)
    try:
        async with JsonRpcClient(f"http://127.0.0.1:{port}/jsonrpc") as client:
            result = await client.predict("add", a=3.0, b=4.0)
            assert result == 7.0
    finally:
        await server.stop()


async def test_client_routes_multiple_model_versions() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo_v1(text: str) -> str:
        return f"v1:{text}"

    @app.model("echo", version="2")
    def echo_v2(text: str) -> str:
        return f"v2:{text}"

    server, port = await _start_test_server(app)
    try:
        async with JsonRpcClient(f"http://127.0.0.1:{port}/jsonrpc") as client:
            assert await client.predict("echo", text="one") == "v1:one"
            assert (
                await client.predict("echo", model_version="2", text="two")
                == "v2:two"
            )
    finally:
        await server.stop()


async def test_tensor_model_e2e() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("double")
    def double(arr: TensorInput[np.float32, 3]) -> TensorOutput[np.float32, 3]:
        return arr * 2

    server, port = await _start_test_server(app)
    try:
        async with JsonRpcClient(f"http://127.0.0.1:{port}/jsonrpc") as client:
            input_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            result = await client.predict("double", arr=input_arr)
            assert isinstance(result, np.ndarray)
            np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0])
    finally:
        await server.stop()


async def test_context_injection_e2e() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("info")
    def info(ctx: Context, text: str) -> str:
        return f"{ctx.method}: {text}"

    server, port = await _start_test_server(app)
    try:
        async with JsonRpcClient(f"http://127.0.0.1:{port}/jsonrpc") as client:
            result = await client.predict("info", text="hello")
            assert "predict.info" in result
            assert "hello" in result
    finally:
        await server.stop()


async def test_depends_injection_e2e() -> None:
    app = BlazeApp(enable_batching=False)
    app.state.multiplier = 10

    def get_mult(ctx: Context) -> int:
        return ctx.app_state.multiplier

    @app.model("multiply")
    def multiply(x: int, mult: int = Depends(get_mult)) -> int:
        return x * mult

    server, port = await _start_test_server(app)
    try:
        async with JsonRpcClient(f"http://127.0.0.1:{port}/jsonrpc") as client:
            result = await client.predict("multiply", x=5)
            assert result == 50
    finally:
        await server.stop()


async def test_streaming_e2e() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("gen", streaming=True)
    async def gen(prompt: str) -> str:
        for word in prompt.split():
            yield word

    server, port = await _start_test_server(app)
    try:
        async with JsonRpcClient(f"http://127.0.0.1:{port}/jsonrpc") as client:
            chunks = [c async for c in client.stream("gen", prompt="a b c")]
            assert chunks == ["a", "b", "c"]
    finally:
        await server.stop()


async def test_health_endpoint() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    server, port = await _start_test_server(app)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                assert resp.status == 200
                body = await resp.json()
                assert body["status"] == "ok"
    finally:
        await server.stop()


async def test_jsonrpc_batch_request_e2e() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return f"Echo: {text}"

    server, port = await _start_test_server(app)
    try:
        async with aiohttp.ClientSession() as session:
            batch = [
                {
                    "jsonrpc": "2.0",
                    "method": "predict.echo",
                    "params": {"text": "a"},
                    "id": 1,
                },
                {
                    "jsonrpc": "2.0",
                    "method": "predict.echo",
                    "params": {"text": "b"},
                    "id": 2,
                },
            ]
            async with session.post(
                f"http://127.0.0.1:{port}/jsonrpc", json=batch
            ) as resp:
                assert resp.status == 200
                results = await resp.json()
                assert len(results) == 2
                assert results[0]["result"] == "Echo: a"
                assert results[1]["result"] == "Echo: b"
    finally:
        await server.stop()


async def test_error_model_not_found_e2e() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    server, port = await _start_test_server(app)
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "predict.nonexistent",
                "params": {},
                "id": 1,
            }
            async with session.post(
                f"http://127.0.0.1:{port}/jsonrpc", json=payload
            ) as resp:
                body = await resp.json()
                assert "error" in body
                assert body["error"]["code"] == -32601
    finally:
        await server.stop()

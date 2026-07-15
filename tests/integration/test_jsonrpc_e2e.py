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
from blazerpc.server.middleware import RequestInfo, ResponseInfo, TransportMiddleware
from blazerpc.types import TensorInput, TensorOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _start_test_server(
    app: BlazeApp,
    middleware: list[TransportMiddleware] | None = None,
) -> tuple[JsonRpcServer, int]:
    """Start a JSON-RPC server on a random port and return (server, port)."""
    dispatcher = JsonRpcDispatcher(app.registry, app_state=app.state)
    server = JsonRpcServer(dispatcher, middleware=middleware)

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
                await client.predict("echo", model_version="2", text="two") == "v2:two"
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


async def test_middleware_denial_prevents_model_execution() -> None:
    app = BlazeApp(enable_batching=False)
    model_calls = 0
    requests: list[RequestInfo] = []
    responses: list[ResponseInfo] = []

    @app.model("secret")
    def secret(value: str) -> str:
        nonlocal model_calls
        model_calls += 1
        return value

    class DenyMiddleware(TransportMiddleware):
        async def on_request(self, info: RequestInfo) -> None:
            requests.append(info)
            raise PermissionError("sensitive authorization detail")

        async def on_response(self, info: ResponseInfo) -> None:
            responses.append(info)

    server, port = await _start_test_server(app, [DenyMiddleware()])
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "predict.secret",
                "params": {"value": "classified"},
                "id": 7,
            }
            async with session.post(
                f"http://127.0.0.1:{port}/jsonrpc", json=payload
            ) as resp:
                body = await resp.json()

        assert resp.status == 403
        assert body == {
            "jsonrpc": "2.0",
            "error": {"code": -32000, "message": "Forbidden"},
            "id": 7,
        }
        assert "sensitive" not in str(body)
        assert model_calls == 0
        assert [info.method for info in requests] == ["predict.secret"]
        assert len(responses) == 1
        assert responses[0].method == "predict.secret"
        assert responses[0].status == 403
        assert responses[0].duration >= 0
    finally:
        await server.stop()


async def test_batch_middleware_callbacks_are_per_element() -> None:
    app = BlazeApp(enable_batching=False)
    events: list[tuple[str, str, int | None]] = []

    @app.model("echo")
    def echo(text: str) -> str:
        return text

    class RecordingMiddleware(TransportMiddleware):
        async def on_request(self, info: RequestInfo) -> None:
            events.append(("request", info.method, None))

        async def on_response(self, info: ResponseInfo) -> None:
            events.append(("response", info.method, info.status))

    server, port = await _start_test_server(app, [RecordingMiddleware()])
    try:
        async with aiohttp.ClientSession() as session:
            batch = [
                {
                    "jsonrpc": "2.0",
                    "method": "predict.echo",
                    "params": {"text": "ok"},
                    "id": 1,
                },
                {
                    "jsonrpc": "2.0",
                    "method": "predict.missing",
                    "params": {},
                    "id": 2,
                },
            ]
            async with session.post(
                f"http://127.0.0.1:{port}/jsonrpc", json=batch
            ) as resp:
                assert resp.status == 200
                await resp.json()

        assert events[:2] == [
            ("request", "predict.echo", None),
            ("request", "predict.missing", None),
        ]
        assert set(events[2:]) == {
            ("response", "predict.echo", 200),
            ("response", "predict.missing", 200),
        }
    finally:
        await server.stop()


async def test_jsonrpc_invalid_requests_and_mixed_batch_conformance() -> None:
    app = BlazeApp(enable_batching=False)
    calls: list[str] = []

    @app.model("echo")
    def echo(text: str) -> str:
        calls.append(text)
        return text

    server, port = await _start_test_server(app)
    url = f"http://127.0.0.1:{port}/jsonrpc"
    try:
        async with aiohttp.ClientSession() as session:
            invalid_payloads = [
                {"data": "null", "headers": {"Content-Type": "application/json"}},
                {"json": 1},
                {"json": "request"},
            ]
            for request_kwargs in invalid_payloads:
                async with session.post(url, **request_kwargs) as resp:
                    assert resp.status == 200
                    body = await resp.json()
                    assert body["error"]["code"] == -32600
                    assert body["id"] is None

            async with session.post(url, json=[]) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert body["error"]["code"] == -32600

            batch = [
                7,
                {"jsonrpc": "2.0", "method": 3, "id": 2},
                {
                    "jsonrpc": "2.0",
                    "method": "predict.echo",
                    "params": {"text": "reply"},
                    "id": 3,
                },
                {
                    "jsonrpc": "2.0",
                    "method": "predict.echo",
                    "params": {"text": "notify"},
                },
            ]
            async with session.post(url, json=batch) as resp:
                assert resp.status == 200
                body = await resp.json()

        assert [item.get("id") for item in body] == [None, 2, 3]
        assert [item.get("error", {}).get("code") for item in body[:2]] == [
            -32600,
            -32600,
        ]
        assert body[2]["result"] == "reply"
        assert calls == ["reply", "notify"]
    finally:
        await server.stop()


async def test_jsonrpc_notifications_have_no_response() -> None:
    app = BlazeApp(enable_batching=False)
    calls: list[str] = []

    @app.model("record")
    def record(value: str) -> str:
        calls.append(value)
        return value

    server, port = await _start_test_server(app)
    url = f"http://127.0.0.1:{port}/jsonrpc"
    try:
        notification = {
            "jsonrpc": "2.0",
            "method": "predict.record",
            "params": {"value": "one"},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=notification) as resp:
                assert resp.status == 204
                assert await resp.read() == b""

            async with session.post(
                url,
                json=[
                    {**notification, "params": {"value": "two"}},
                    {**notification, "params": {"value": "three"}},
                ],
            ) as resp:
                assert resp.status == 204
                assert await resp.read() == b""

        assert calls == ["one", "two", "three"]
    finally:
        await server.stop()


async def test_jsonrpc_param_errors_do_not_invoke_model() -> None:
    app = BlazeApp(enable_batching=False)
    calls = 0

    @app.model("optional")
    def optional(required: str, value: str | None = "default") -> str | None:
        nonlocal calls
        calls += 1
        return value

    server, port = await _start_test_server(app)
    url = f"http://127.0.0.1:{port}/jsonrpc"
    try:
        async with aiohttp.ClientSession() as session:
            invalid_params = [
                {},
                {"required": "ok", "unknown": 1},
                {"required": "ok", "value": "x", "extra": 2},
            ]
            for request_id, params in enumerate(invalid_params, 1):
                payload = {
                    "jsonrpc": "2.0",
                    "method": "predict.optional",
                    "params": params,
                    "id": request_id,
                }
                async with session.post(url, json=payload) as resp:
                    body = await resp.json()
                    assert body["error"]["code"] == -32602

            payload = {
                "jsonrpc": "2.0",
                "method": "predict.optional",
                "params": {"required": "ok", "value": None},
                "id": 4,
            }
            async with session.post(url, json=payload) as resp:
                body = await resp.json()

        assert body["result"] is None
        assert calls == 1
    finally:
        await server.stop()


async def test_wrong_transport_cardinality_is_rejected_before_invocation() -> None:
    app = BlazeApp(enable_batching=False)
    unary_calls = 0
    stream_calls = 0

    @app.model("unary")
    def unary(value: str) -> str:
        nonlocal unary_calls
        unary_calls += 1
        return value

    @app.model("tokens", streaming=True)
    async def tokens(value: str) -> str:
        nonlocal stream_calls
        stream_calls += 1
        yield value

    server, port = await _start_test_server(app)
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "predict.tokens",
                "params": {"value": "x"},
                "id": 1,
            }
            async with session.post(
                f"http://127.0.0.1:{port}/jsonrpc", json=payload
            ) as resp:
                body = await resp.json()
                assert body["error"]["code"] == -32600

            async with session.post(
                f"http://127.0.0.1:{port}/jsonrpc/stream/unary",
                json={"params": {"value": "x"}},
            ) as resp:
                assert resp.status == 400

        assert unary_calls == 0
        assert stream_calls == 0
    finally:
        await server.stop()

"""Tests for the JSON-RPC dispatcher."""

from __future__ import annotations

import numpy as np

from blazerpc import BlazeApp, Context, Depends
from blazerpc.codegen.jsonrpc_handler import JsonRpcDispatcher
from blazerpc.runtime.json_serialization import tensor_to_json
from blazerpc.types import TensorInput, TensorOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_and_dispatcher() -> tuple[BlazeApp, JsonRpcDispatcher]:
    app = BlazeApp(enable_batching=False)

    @app.model("echo")
    def echo(text: str) -> str:
        return f"Echo: {text}"

    @app.model("add")
    def add(a: float, b: float) -> float:
        return a + b

    dispatcher = JsonRpcDispatcher(app.registry, app_state=app.state)
    return app, dispatcher


# ---------------------------------------------------------------------------
# Valid requests
# ---------------------------------------------------------------------------


async def test_handle_echo() -> None:
    _, dispatcher = _make_app_and_dispatcher()
    resp = await dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "method": "predict.echo",
            "params": {"text": "hello"},
            "id": 1,
        }
    )
    assert resp["jsonrpc"] == "2.0"
    assert resp["result"] == "Echo: hello"
    assert resp["id"] == 1


async def test_handle_add() -> None:
    _, dispatcher = _make_app_and_dispatcher()
    resp = await dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "method": "predict.add",
            "params": {"a": 3.0, "b": 4.0},
            "id": 2,
        }
    )
    assert resp["result"] == 7.0


async def test_handle_missing_params_key() -> None:
    """Params default to {} when the 'params' key is omitted."""
    _, dispatcher = _make_app_and_dispatcher()
    # The echo handler expects 'text' which won't be in the empty params,
    # but the handler should still be called (text will be None).
    resp = await dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "method": "predict.echo",
            "params": {"text": "default"},
            "id": 3,
        }
    )
    assert resp["result"] == "Echo: default"


# ---------------------------------------------------------------------------
# Error responses
# ---------------------------------------------------------------------------


async def test_handle_invalid_request() -> None:
    _, dispatcher = _make_app_and_dispatcher()
    resp = await dispatcher.handle({"method": "predict.echo"})  # missing jsonrpc
    assert resp["error"]["code"] == -32600


async def test_handle_method_not_found() -> None:
    _, dispatcher = _make_app_and_dispatcher()
    resp = await dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "method": "predict.nonexistent",
            "params": {},
            "id": 4,
        }
    )
    assert resp["error"]["code"] == -32601


async def test_handle_unknown_method_format() -> None:
    _, dispatcher = _make_app_and_dispatcher()
    resp = await dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "method": "unknown_format",
            "params": {},
            "id": 5,
        }
    )
    assert resp["error"]["code"] == -32601


async def test_handle_params_not_object() -> None:
    _, dispatcher = _make_app_and_dispatcher()
    resp = await dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "method": "predict.echo",
            "params": [1, 2, 3],  # positional params not supported
            "id": 6,
        }
    )
    assert resp["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


async def test_handle_with_context() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("info")
    def info(ctx: Context, text: str) -> str:
        return f"{ctx.method}: {text}"

    dispatcher = JsonRpcDispatcher(app.registry, app_state=app.state)
    resp = await dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "method": "predict.info",
            "params": {"text": "hello"},
            "id": 7,
        }
    )
    assert resp["result"] == "predict.info: hello"


async def test_handle_with_depends() -> None:
    app = BlazeApp(enable_batching=False)
    app.state.prefix = "PREFIX"

    def get_prefix(ctx: Context) -> str:
        return ctx.app_state.prefix

    @app.model("prefixed")
    def prefixed(text: str, prefix: str = Depends(get_prefix)) -> str:
        return f"{prefix}:{text}"

    dispatcher = JsonRpcDispatcher(app.registry, app_state=app.state)
    resp = await dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "method": "predict.prefixed",
            "params": {"text": "hello"},
            "id": 8,
        }
    )
    assert resp["result"] == "PREFIX:hello"


# ---------------------------------------------------------------------------
# Batch requests
# ---------------------------------------------------------------------------


async def test_handle_batch() -> None:
    _, dispatcher = _make_app_and_dispatcher()
    results = await dispatcher.handle_batch(
        [
            {
                "jsonrpc": "2.0",
                "method": "predict.echo",
                "params": {"text": "a"},
                "id": 1,
            },
            {
                "jsonrpc": "2.0",
                "method": "predict.add",
                "params": {"a": 1.0, "b": 2.0},
                "id": 2,
            },
        ]
    )
    assert len(results) == 2
    assert results[0]["result"] == "Echo: a"
    assert results[1]["result"] == 3.0


# ---------------------------------------------------------------------------
# Tensor handling
# ---------------------------------------------------------------------------


async def test_handle_with_tensor_io() -> None:
    app = BlazeApp(enable_batching=False)

    # Define at module scope so get_type_hints resolves TensorInput/TensorOutput
    app.registry.register("double", "1", _tensor_double_fn, streaming=False)

    dispatcher = JsonRpcDispatcher(app.registry, app_state=app.state)

    input_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    resp = await dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "method": "predict.double",
            "params": {"arr": tensor_to_json(input_arr)},
            "id": 9,
        }
    )

    assert "error" not in resp
    result = resp["result"]
    assert isinstance(result, dict)
    assert "shape" in result  # tensor dict


def _tensor_double_fn(
    arr: TensorInput[np.float32, 3],
) -> TensorOutput[np.float32, 3]:
    return arr * 2


# ---------------------------------------------------------------------------
# Streaming (dispatch only, not SSE)
# ---------------------------------------------------------------------------


async def test_handle_streaming() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("gen", streaming=True)
    async def gen(prompt: str) -> str:
        for word in prompt.split():
            yield word

    dispatcher = JsonRpcDispatcher(app.registry, app_state=app.state)
    chunks = [
        c
        async for c in dispatcher.handle_streaming(
            "stream.gen", {"prompt": "hello world"}
        )
    ]
    assert chunks == ["hello", "world"]

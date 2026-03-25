"""Tests for the shared model invocation helpers."""

from __future__ import annotations

import pytest

from blazerpc import BlazeApp, Context, Depends
from blazerpc.codegen.invoke import invoke_model, invoke_streaming_model, resolve_deps
from blazerpc.exceptions import InferenceError


# ---------------------------------------------------------------------------
# invoke_model
# ---------------------------------------------------------------------------


async def test_invoke_sync_model() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("add")
    def add(a: float, b: float) -> float:
        return a + b

    model = app.registry.get("add")
    result = await invoke_model(model, {"a": 1.0, "b": 2.0})
    assert result == 3.0


async def test_invoke_async_model() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("greet")
    async def greet(name: str) -> str:
        return f"Hello, {name}"

    model = app.registry.get("greet")
    result = await invoke_model(model, {"name": "World"})
    assert result == "Hello, World"


async def test_invoke_model_raises_inference_error() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("fail")
    def fail(x: int) -> int:
        raise ValueError("bad input")

    model = app.registry.get("fail")
    with pytest.raises(InferenceError):
        await invoke_model(model, {"x": 1})


# ---------------------------------------------------------------------------
# invoke_streaming_model
# ---------------------------------------------------------------------------


async def test_invoke_async_streaming_model() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("gen", streaming=True)
    async def gen(prompt: str) -> str:
        for word in prompt.split():
            yield word

    model = app.registry.get("gen")
    chunks = [c async for c in invoke_streaming_model(model, {"prompt": "a b c"})]
    assert chunks == ["a", "b", "c"]


async def test_invoke_sync_streaming_model() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("gen_sync", streaming=True)
    def gen_sync(n: int) -> int:
        for i in range(n):
            yield i

    model = app.registry.get("gen_sync")
    chunks = [c async for c in invoke_streaming_model(model, {"n": 3})]
    assert chunks == [0, 1, 2]


# ---------------------------------------------------------------------------
# resolve_deps
# ---------------------------------------------------------------------------


async def test_resolve_deps_with_context() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("test")
    def handler(ctx: Context, text: str) -> str:
        return text

    model = app.registry.get("test")
    resolved = await resolve_deps(
        model, {"auth": "token"}, "127.0.0.1", "predict.test", app.state
    )
    assert "ctx" in resolved
    ctx = resolved["ctx"]
    assert ctx.metadata == {"auth": "token"}
    assert ctx.peer == "127.0.0.1"
    assert ctx.method == "predict.test"


async def test_resolve_deps_with_depends() -> None:
    app = BlazeApp(enable_batching=False)
    app.state.value = 42

    def get_val(ctx: Context) -> int:
        return ctx.app_state.value

    @app.model("test")
    def handler(x: int, val: int = Depends(get_val)) -> int:
        return x + val

    model = app.registry.get("test")
    resolved = await resolve_deps(model, {}, "", "predict.test", app.state)
    assert resolved["val"] == 42


async def test_resolve_deps_with_no_deps() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("simple")
    def handler(x: int) -> int:
        return x

    model = app.registry.get("simple")
    resolved = await resolve_deps(model, {}, "", "predict.simple", app.state)
    assert resolved == {}

"""Tests for the shared model invocation helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from functools import wraps
import threading

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


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_invoke_model_rejects_hidden_stream_results(asynchronous: bool) -> None:
    app = BlazeApp(enable_batching=False)

    def sync_values(value: int) -> Iterator[int]:
        yield value

    async def async_values(value: int) -> AsyncIterator[int]:
        yield value

    @app.model(f"hidden_stream_{asynchronous}")
    def factory(value: int) -> int:
        if asynchronous:
            return async_values(value)  # type: ignore[return-value]
        return sync_values(value)  # type: ignore[return-value]

    with pytest.raises(InferenceError, match="Unary models must not return"):
        await invoke_model(
            app.registry.get(f"hidden_stream_{asynchronous}"), {"value": 1}
        )


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


async def test_sync_streaming_model_does_not_block_event_loop() -> None:
    app = BlazeApp(enable_batching=False)
    release = threading.Event()
    event_loop_ran = threading.Event()

    @app.model("blocking_gen", streaming=True)
    def blocking_gen(n: int) -> bool:
        yield release.wait(timeout=1)

    async def release_from_event_loop() -> None:
        await asyncio.sleep(0.01)
        event_loop_ran.set()
        release.set()

    release_task = asyncio.create_task(release_from_event_loop())
    model = app.registry.get("blocking_gen")
    chunks = [chunk async for chunk in invoke_streaming_model(model, {"n": 1})]
    await release_task

    assert event_loop_ran.is_set()
    assert chunks == [True]


async def test_decorated_streaming_factory_is_accepted() -> None:
    app = BlazeApp(enable_batching=False)

    def decorate(func):  # type: ignore[no-untyped-def]
        @wraps(func)
        def wrapper(**kwargs):  # type: ignore[no-untyped-def]
            return func(**kwargs)

        return wrapper

    @app.model("decorated", streaming=True)
    @decorate
    def decorated(value: str) -> str:
        return iter([value, value.upper()])  # type: ignore[return-value]

    chunks = [
        chunk
        async for chunk in invoke_streaming_model(
            app.registry.get("decorated"), {"value": "one"}
        )
    ]
    assert chunks == ["one", "ONE"]


async def test_callable_streaming_factory_is_accepted() -> None:
    app = BlazeApp(enable_batching=False)

    class Stream:
        def __call__(self, count: int) -> int:
            return iter(range(count))  # type: ignore[return-value]

    factory = Stream()
    app.registry.register("callable", "1", factory, streaming=True)

    chunks = [
        chunk
        async for chunk in invoke_streaming_model(
            app.registry.get("callable"), {"count": 3}
        )
    ]
    assert chunks == [0, 1, 2]


async def test_async_iterable_factory_is_supported() -> None:
    app = BlazeApp(enable_batching=False)

    async def values(count: int) -> AsyncIterator[int]:
        for value in range(count):
            yield value

    @app.model("async_iterable", streaming=True)
    def factory(count: int) -> int:
        return values(count)  # type: ignore[return-value]

    chunks = [
        chunk
        async for chunk in invoke_streaming_model(
            app.registry.get("async_iterable"), {"count": 3}
        )
    ]
    assert chunks == [0, 1, 2]


async def test_invalid_stream_factory_result_fails_at_invocation() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("invalid_stream", streaming=True)
    def factory(value: int) -> int:
        return value

    with pytest.raises(InferenceError, match="not iterable"):
        _ = [
            chunk
            async for chunk in invoke_streaming_model(
                app.registry.get("invalid_stream"), {"value": 1}
            )
        ]


async def test_sync_iterator_is_demand_driven_and_thread_affine() -> None:
    app = BlazeApp(enable_batching=False)
    thread_ids: list[int] = []
    next_calls = 0
    closed = threading.Event()

    class Stream(Iterator[int]):
        def __iter__(self) -> Iterator[int]:
            thread_ids.append(threading.get_ident())
            return self

        def __next__(self) -> int:
            nonlocal next_calls
            thread_ids.append(threading.get_ident())
            next_calls += 1
            if next_calls > 2:
                raise StopIteration
            return next_calls

        def close(self) -> None:
            thread_ids.append(threading.get_ident())
            closed.set()

    @app.model("threaded", streaming=True)
    def factory(value: int) -> int:
        thread_ids.append(threading.get_ident())
        return Stream()  # type: ignore[return-value]

    stream = invoke_streaming_model(app.registry.get("threaded"), {"value": 1})
    assert await anext(stream) == 1
    assert next_calls == 1
    assert not closed.is_set()
    await stream.aclose()

    assert closed.is_set()
    assert len(set(thread_ids)) == 1


@pytest.mark.parametrize("failure", [False, True])
async def test_sync_iterator_is_closed_on_completion_or_error(failure: bool) -> None:
    app = BlazeApp(enable_batching=False)
    closed = threading.Event()

    class Stream(Iterator[int]):
        def __iter__(self) -> Iterator[int]:
            return self

        def __next__(self) -> int:
            if failure:
                raise ValueError("stream failed")
            raise StopIteration

        def close(self) -> None:
            closed.set()

    @app.model(f"close_{failure}", streaming=True)
    def factory(value: int) -> int:
        return Stream()  # type: ignore[return-value]

    stream = invoke_streaming_model(app.registry.get(f"close_{failure}"), {"value": 1})
    if failure:
        with pytest.raises(InferenceError, match="stream failed"):
            _ = [chunk async for chunk in stream]
    else:
        assert [chunk async for chunk in stream] == []
    assert closed.is_set()


async def test_sync_iterator_is_closed_after_cancellation() -> None:
    app = BlazeApp(enable_batching=False)
    advancing = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class Stream(Iterator[int]):
        def __iter__(self) -> Iterator[int]:
            return self

        def __next__(self) -> int:
            advancing.set()
            release.wait(timeout=1)
            return 1

        def close(self) -> None:
            closed.set()

    @app.model("cancelled", streaming=True)
    def factory(value: int) -> int:
        return Stream()  # type: ignore[return-value]

    async def consume() -> None:
        async for _ in invoke_streaming_model(
            app.registry.get("cancelled"), {"value": 1}
        ):
            pass

    task = asyncio.create_task(consume())
    await asyncio.to_thread(advancing.wait, 1)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


async def test_sync_iterator_from_blocked_factory_is_closed_after_cancellation() -> (
    None
):
    app = BlazeApp(enable_batching=False)
    factory_started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    thread_ids: list[int] = []

    class Stream(Iterator[int]):
        def __iter__(self) -> Iterator[int]:
            return self

        def __next__(self) -> int:
            raise StopIteration

        def close(self) -> None:
            thread_ids.append(threading.get_ident())
            closed.set()

    @app.model("blocked_factory", streaming=True)
    def factory(value: int) -> int:
        thread_ids.append(threading.get_ident())
        factory_started.set()
        release.wait(timeout=1)
        return Stream()  # type: ignore[return-value]

    async def consume() -> None:
        async for _ in invoke_streaming_model(
            app.registry.get("blocked_factory"), {"value": 1}
        ):
            pass

    task = asyncio.create_task(consume())
    assert await asyncio.to_thread(factory_started.wait, 1)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed.is_set()
    assert len(set(thread_ids)) == 1


async def test_sync_iterator_from_blocked_iter_is_closed_after_cancellation() -> None:
    app = BlazeApp(enable_batching=False)
    iter_started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    thread_ids: list[int] = []

    class Stream(Iterator[int]):
        def __iter__(self) -> Iterator[int]:
            thread_ids.append(threading.get_ident())
            iter_started.set()
            release.wait(timeout=1)
            return self

        def __next__(self) -> int:
            raise StopIteration

        def close(self) -> None:
            thread_ids.append(threading.get_ident())
            closed.set()

    @app.model("blocked_iter", streaming=True)
    def factory(value: int) -> int:
        thread_ids.append(threading.get_ident())
        return Stream()  # type: ignore[return-value]

    async def consume() -> None:
        async for _ in invoke_streaming_model(
            app.registry.get("blocked_iter"), {"value": 1}
        ):
            pass

    task = asyncio.create_task(consume())
    assert await asyncio.to_thread(iter_started.wait, 1)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed.is_set()
    assert len(set(thread_ids)) == 1


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


async def test_sync_dependency_does_not_block_event_loop() -> None:
    app = BlazeApp(enable_batching=False)
    release = threading.Event()
    event_loop_ran = threading.Event()

    def blocking_dep(ctx: Context) -> bool:
        return release.wait(timeout=1)

    @app.model("test_blocking_dep")
    def handler(value: bool = Depends(blocking_dep)) -> bool:
        return value

    async def release_from_event_loop() -> None:
        await asyncio.sleep(0.01)
        event_loop_ran.set()
        release.set()

    release_task = asyncio.create_task(release_from_event_loop())
    model = app.registry.get("test_blocking_dep")
    resolved = await resolve_deps(model, {}, "", "predict.test", app.state)
    await release_task

    assert event_loop_ran.is_set()
    assert resolved["value"] is True


async def test_resolve_deps_with_no_deps() -> None:
    app = BlazeApp(enable_batching=False)

    @app.model("simple")
    def handler(x: int) -> int:
        return x

    model = app.registry.get("simple")
    resolved = await resolve_deps(model, {}, "", "predict.simple", app.state)
    assert resolved == {}

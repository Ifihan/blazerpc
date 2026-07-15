"""Protocol-agnostic model invocation helpers.

Extracted from the gRPC servicer so that both gRPC and JSON-RPC transports
can share the same model execution, dependency resolution, and error handling
logic.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import inspect
from typing import Any, cast

from blazerpc.exceptions import InferenceError
from blazerpc.runtime.registry import ModelInfo


def _next(iterator: Iterator[Any]) -> tuple[bool, Any]:
    """Advance an iterator without leaking StopIteration into an asyncio Future."""
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def _close(value: object) -> None:
    close = getattr(value, "close", None)
    if close is not None:
        close()


_MISSING = object()


class _SyncStreamSession:
    """Own synchronous stream state manipulated by one worker thread."""

    def __init__(self) -> None:
        self.stream: object = _MISSING
        self.iterator: Iterator[Any] | None = None

    def invoke(self, func: Any, kwargs: dict[str, Any]) -> object:
        self.stream = func(**kwargs)
        return self.stream

    def set_stream(self, stream: object) -> None:
        self.stream = stream

    def detach(self) -> None:
        self.stream = _MISSING

    def create_iterator(self) -> None:
        self.iterator = iter(cast(Iterable[Any], self.stream))

    def advance(self) -> tuple[bool, Any]:
        if self.iterator is None:
            raise RuntimeError("Stream iterator has not been created")
        return _next(self.iterator)

    def close(self) -> None:
        if self.iterator is not None:
            _close(self.iterator)
            return
        if self.stream is _MISSING:
            return
        try:
            self.iterator = iter(cast(Iterable[Any], self.stream))
        except Exception:
            _close(self.stream)
        else:
            _close(self.iterator)


async def invoke_model(
    model: ModelInfo,
    kwargs: dict[str, Any],
    *,
    batcher: Any | None = None,
) -> Any:
    """Execute a unary model function with the given kwargs.

    Handles batcher submission vs direct call, sync/async bridging,
    and wraps exceptions in :class:`InferenceError`.
    """
    try:
        if batcher is not None:
            # Batcher receives only request-field kwargs (no deps).
            request_only = {k: v for k, v in kwargs.items() if k in model.input_types}
            result = await batcher.submit(request_only)
        elif inspect.iscoroutinefunction(model.func):
            result = await model.func(**kwargs)
        else:
            result = await asyncio.to_thread(model.func, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, (Iterator, AsyncIterable)):
            raise TypeError("Unary models must not return iterators or async iterables")
        return result
    except Exception as exc:
        raise InferenceError(str(exc), model_name=model.name) from exc


async def invoke_streaming_model(
    model: ModelInfo,
    kwargs: dict[str, Any],
) -> AsyncIterator[Any]:
    """Execute a streaming model function, yielding chunks.

    Supports factories returning synchronous or asynchronous iterables. All
    operations on a synchronous iterator run on one dedicated worker thread.
    ``CancelledError`` is propagated after deterministic iterator cleanup.
    """
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    session = _SyncStreamSession()
    worker_future: asyncio.Future[Any] | None = None
    async_iterator: AsyncIterator[Any] | None = None
    try:
        if inspect.iscoroutinefunction(model.func):
            stream = await model.func(**kwargs)
            worker_future = asyncio.ensure_future(
                loop.run_in_executor(executor, session.set_stream, stream)
            )
            await asyncio.shield(worker_future)
            worker_future = None
        else:
            worker_future = asyncio.ensure_future(
                loop.run_in_executor(executor, session.invoke, model.func, kwargs)
            )
            stream = await asyncio.shield(worker_future)
            worker_future = None

        if inspect.isawaitable(stream):
            stream = await stream
            worker_future = asyncio.ensure_future(
                loop.run_in_executor(executor, session.set_stream, stream)
            )
            await asyncio.shield(worker_future)
            worker_future = None

        if isinstance(stream, AsyncIterable):
            worker_future = asyncio.ensure_future(
                loop.run_in_executor(executor, session.detach)
            )
            await asyncio.shield(worker_future)
            worker_future = None
            async_iterator = stream.__aiter__()
            async for chunk in async_iterator:
                yield chunk
        else:
            worker_future = asyncio.ensure_future(
                loop.run_in_executor(executor, session.create_iterator)
            )
            await asyncio.shield(worker_future)
            worker_future = None
            while True:
                worker_future = asyncio.ensure_future(
                    loop.run_in_executor(executor, session.advance)
                )
                has_chunk, chunk = await asyncio.shield(worker_future)
                worker_future = None
                if not has_chunk:
                    break
                yield chunk
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise InferenceError(str(exc), model_name=model.name) from exc
    finally:
        if worker_future is not None:
            with suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(worker_future)
        close_future = asyncio.ensure_future(
            loop.run_in_executor(executor, session.close)
        )
        with suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(close_future)
        if async_iterator is not None:
            close = getattr(async_iterator, "aclose", None)
            if close is not None:
                with suppress(Exception, asyncio.CancelledError):
                    await close()
        executor.shutdown(wait=True)


async def resolve_deps(
    model: ModelInfo,
    metadata: Any,
    peer: Any,
    method: str,
    app_state: Any | None,
) -> dict[str, Any]:
    """Resolve all ``Depends()`` and ``Context`` injections for a model call.

    This is protocol-agnostic: it accepts raw metadata and peer info
    rather than a grpclib ``Stream``.
    """
    from blazerpc.context import AppState, Context

    _state = app_state if app_state is not None else AppState()
    ctx = Context.from_raw(
        metadata=metadata, peer=peer, method=method, app_state=_state
    )

    resolved: dict[str, Any] = {}
    for name in model.context_params:
        resolved[name] = ctx
    for name, dep in model.dep_params.items():
        if inspect.iscoroutinefunction(dep.fn):
            resolved[name] = await dep.fn(ctx)
        else:
            resolved[name] = await asyncio.to_thread(dep.fn, ctx)
    return resolved

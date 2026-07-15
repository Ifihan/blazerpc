"""Protocol-agnostic model invocation helpers.

Extracted from the gRPC servicer so that both gRPC and JSON-RPC transports
can share the same model execution, dependency resolution, and error handling
logic.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, cast

from blazerpc.exceptions import InferenceError
from blazerpc.runtime.registry import ModelInfo


def _next(iterator: Iterator[Any]) -> tuple[bool, Any]:
    """Advance an iterator without leaking StopIteration into an asyncio Future."""
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def _call_and_iter(
    func: Callable[..., object], kwargs: dict[str, Any]
) -> Iterator[Any]:
    return iter(cast(Iterable[Any], func(**kwargs)))


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
            return await batcher.submit(request_only)
        elif asyncio.iscoroutinefunction(model.func):
            return await model.func(**kwargs)
        else:
            return await asyncio.to_thread(model.func, **kwargs)
    except Exception as exc:
        raise InferenceError(str(exc), model_name=model.name) from exc


async def invoke_streaming_model(
    model: ModelInfo,
    kwargs: dict[str, Any],
) -> AsyncIterator[Any]:
    """Execute a streaming model function, yielding chunks.

    Supports both async and sync generators.  Propagates
    ``CancelledError`` for client disconnection handling.
    """
    try:
        if inspect.isasyncgenfunction(model.func):
            async for chunk in model.func(**kwargs):
                yield chunk
        else:
            iterator = await asyncio.to_thread(_call_and_iter, model.func, kwargs)
            while True:
                has_chunk, chunk = await asyncio.to_thread(_next, iterator)
                if not has_chunk:
                    break
                yield chunk
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise InferenceError(str(exc), model_name=model.name) from exc


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
        if asyncio.iscoroutinefunction(dep.fn):
            resolved[name] = await dep.fn(ctx)
        else:
            resolved[name] = await asyncio.to_thread(dep.fn, ctx)
    return resolved

"""gRPC servicer stub generation.

Dynamically builds a grpclib-compatible servicer from a
:class:`~blazerpc.runtime.registry.ModelRegistry`.  Each registered model
becomes one RPC method on the ``InferenceService``.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from grpclib.const import Cardinality, Handler
from grpclib.server import Stream

from blazerpc.codegen.proto import _sanitize_name
from blazerpc.exceptions import InferenceError
from blazerpc.runtime.registry import ModelInfo, ModelRegistry
from blazerpc.runtime.serialization import python_to_proto


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class InferenceServicer:
    """A grpclib-compatible servicer created from a :class:`ModelRegistry`.

    Implements the ``__mapping__()`` protocol required by
    :class:`grpclib.server.Server`.
    """

    SERVICE_NAME = "blazerpc.InferenceService"

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        batcher: Any | None = None,
    ) -> None:
        self._registry = registry
        self._batcher = batcher

    def __mapping__(self) -> dict[str, Handler]:
        mapping: dict[str, Handler] = {}
        for model in self._registry.list_models():
            rpc_name = f"Predict{_sanitize_name(model.name)}"
            path = f"/{self.SERVICE_NAME}/{rpc_name}"

            if model.streaming:
                handler_fn = _make_streaming_handler(model)
                cardinality = Cardinality.UNARY_STREAM
            else:
                handler_fn = _make_unary_handler(model, batcher=self._batcher)
                cardinality = Cardinality.UNARY_UNARY

            mapping[path] = Handler(
                handler_fn,
                cardinality,
                None,  # request_type
                None,  # reply_type
            )
        return mapping


def build_servicer(
    registry: ModelRegistry,
    *,
    batcher: Any | None = None,
) -> InferenceServicer:
    """Convenience factory that returns a ready-to-use servicer."""
    return InferenceServicer(registry, batcher=batcher)


# ---------------------------------------------------------------------------
# Handler factories
# ---------------------------------------------------------------------------


def _make_unary_handler(
    model: ModelInfo,
    *,
    batcher: Any | None = None,
) -> Callable[..., Any]:
    """Return an ``async def handler(stream)`` for a unary RPC."""

    async def _handler(stream: Stream[Any, Any]) -> None:
        request_bytes = await stream.recv_message()
        kwargs = _decode_request(request_bytes, model)

        try:
            if batcher is not None:
                raw_result = await batcher.submit(kwargs)
            elif asyncio.iscoroutinefunction(model.func):
                raw_result = await model.func(**kwargs)
            else:
                raw_result = await asyncio.to_thread(model.func, **kwargs)
        except Exception as exc:
            raise InferenceError(str(exc), model_name=model.name) from exc

        response_bytes = _encode_response(raw_result, model)
        await stream.send_message(response_bytes)

    return _handler


def _make_streaming_handler(model: ModelInfo) -> Callable[..., Any]:
    """Return an ``async def handler(stream)`` for a server-streaming RPC."""

    async def _handler(stream: Stream[Any, Any]) -> None:
        request_bytes = await stream.recv_message()
        kwargs = _decode_request(request_bytes, model)

        try:
            if inspect.isasyncgenfunction(model.func):
                async for chunk in model.func(**kwargs):
                    response_bytes = _encode_response(chunk, model)
                    await stream.send_message(response_bytes)
            else:
                for chunk in model.func(**kwargs):
                    response_bytes = _encode_response(chunk, model)
                    await stream.send_message(response_bytes)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise InferenceError(str(exc), model_name=model.name) from exc

    return _handler


# ---------------------------------------------------------------------------
# Wire encode / decode helpers
# ---------------------------------------------------------------------------


def _decode_request(raw: Any, model: ModelInfo) -> dict[str, Any]:
    """Decode raw request data into keyword arguments for the model func."""
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    return {"__raw__": raw}


def _encode_response(result: Any, model: ModelInfo) -> Any:
    """Encode a model result into a wire-friendly representation."""
    if model.output_type is not None:
        return python_to_proto(result, model.output_type)
    return result

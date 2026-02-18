"""gRPC servicer stub generation.

Dynamically builds a grpclib-compatible servicer from a
:class:`~blazerpc.runtime.registry.ModelRegistry`.  Each registered model
becomes one RPC method on the ``InferenceService``.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
from typing import Any, Callable

import numpy as np
from grpclib.const import Cardinality, Handler
from grpclib.server import Stream

from blazerpc.codegen.proto import _sanitize_name
from blazerpc.exceptions import InferenceError
from blazerpc.runtime.registry import ModelInfo, ModelRegistry
from blazerpc.runtime.serialization import (
    TensorProto,
    deserialize_tensor,
    serialize_tensor,
)
from blazerpc.types import _TensorType


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
        batchers: dict[str, Any] | None = None,
    ) -> None:
        self._registry = registry
        self._batchers = batchers or {}

    def __mapping__(self) -> dict[str, Handler]:
        mapping: dict[str, Handler] = {}
        for model in self._registry.list_models():
            rpc_name = f"Predict{_sanitize_name(model.name)}"
            path = f"/{self.SERVICE_NAME}/{rpc_name}"

            if model.streaming:
                handler_fn = _make_streaming_handler(model)
                cardinality = Cardinality.UNARY_STREAM
            else:
                batcher = self._batchers.get(model.name)
                handler_fn = _make_unary_handler(model, batcher=batcher)
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
    batchers: dict[str, Any] | None = None,
) -> InferenceServicer:
    """Convenience factory that returns a ready-to-use servicer."""
    return InferenceServicer(registry, batchers=batchers)


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
    """Decode raw request data into keyword arguments for the model func.

    With the ``RawCodec``, *raw* arrives as JSON-encoded ``bytes``.
    Tensor fields are represented as ``{"shape": [...], "dtype": "...",
    "data": "<base64>"}`` and are converted back to numpy arrays.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw

    data = json.loads(raw)
    for field_name, field_type in model.input_types.items():
        if field_name not in data:
            continue
        if isinstance(field_type, _TensorType) and isinstance(data[field_name], dict):
            tensor_dict = data[field_name]
            proto = TensorProto(
                shape=tuple(tensor_dict["shape"]),
                dtype=tensor_dict["dtype"],
                data=base64.b64decode(tensor_dict["data"]),
            )
            data[field_name] = deserialize_tensor(proto)
    return data


def _encode_response(result: Any, model: ModelInfo) -> bytes:
    """Encode a model result into JSON bytes for the wire.

    Tensor results (numpy arrays) are serialized as
    ``{"shape": [...], "dtype": "...", "data": "<base64>"}``.
    """
    if isinstance(result, np.ndarray):
        proto = serialize_tensor(result)
        payload = {
            "result": {
                "shape": list(proto.shape),
                "dtype": proto.dtype,
                "data": base64.b64encode(proto.data).decode(),
            }
        }
        return json.dumps(payload).encode()

    if isinstance(result, _TensorType):
        proto = serialize_tensor(result)
        payload = {
            "result": {
                "shape": list(proto.shape),
                "dtype": proto.dtype,
                "data": base64.b64encode(proto.data).decode(),
            }
        }
        return json.dumps(payload).encode()

    return json.dumps({"result": result}).encode()

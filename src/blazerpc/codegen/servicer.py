"""gRPC servicer stub generation.

Dynamically builds a grpclib-compatible servicer from a
:class:`~blazerpc.runtime.registry.ModelRegistry`.  Each registered model
becomes one RPC method on the ``InferenceService``.

Wire format: binary Protobuf, encoded/decoded via betterproto message classes
built at startup by :mod:`blazerpc.codegen.proto_types`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from grpclib.const import Cardinality, Handler
from grpclib.server import Stream

from blazerpc.codegen.invoke import invoke_model, invoke_streaming_model, resolve_deps
from blazerpc.codegen.proto import ProtoGenerator, _rpc_name
from blazerpc.codegen.proto_types import (
    _TensorProtoMsg,
    build_message_classes,
)
from blazerpc.runtime.registry import ModelInfo, ModelRegistry, batcher_key
from blazerpc.runtime.serialization import deserialize_tensor, serialize_tensor
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
        app_state: Any | None = None,
    ) -> None:
        self._registry = registry
        self._batchers = batchers or {}
        self._app_state = app_state

    @property
    def file_descriptor(self) -> Any:
        """Return the dynamic service descriptor used by server reflection."""
        return ProtoGenerator().generate_file_descriptor(self._registry)

    def __mapping__(self) -> dict[str, Handler]:
        mapping: dict[str, Handler] = {}
        for model in self._registry.list_models():
            rpc_name = _rpc_name(model)
            path = f"/{self.SERVICE_NAME}/{rpc_name}"

            request_cls, response_cls = build_message_classes(model)

            if model.streaming:
                handler_fn = _make_streaming_handler(
                    model,
                    request_cls,
                    response_cls,
                    method=path,
                    app_state=self._app_state,
                )
                cardinality = Cardinality.UNARY_STREAM
            else:
                batcher = self._batchers.get(batcher_key(model.name, model.version))
                handler_fn = _make_unary_handler(
                    model,
                    request_cls,
                    response_cls,
                    batcher=batcher,
                    method=path,
                    app_state=self._app_state,
                )
                cardinality = Cardinality.UNARY_UNARY

            mapping[path] = Handler(
                handler_fn,
                cardinality,
                None,  # request_type — we handle raw bytes ourselves via RawCodec
                None,  # reply_type
            )
        return mapping


def build_servicer(
    registry: ModelRegistry,
    *,
    batchers: dict[str, Any] | None = None,
    app_state: Any | None = None,
) -> InferenceServicer:
    """Convenience factory that returns a ready-to-use servicer."""
    return InferenceServicer(registry, batchers=batchers, app_state=app_state)


# ---------------------------------------------------------------------------
# Handler factories
# ---------------------------------------------------------------------------


def _make_unary_handler(
    model: ModelInfo,
    request_cls: type,
    response_cls: type,
    *,
    batcher: Any | None = None,
    method: str = "",
    app_state: Any | None = None,
) -> Callable[..., Any]:
    """Return an ``async def handler(stream)`` for a unary RPC."""
    _has_deps = bool(model.dep_params or model.context_params)

    async def _handler(stream: Stream[Any, Any]) -> None:
        request_bytes = await stream.recv_message()
        kwargs = _decode_request(request_bytes, model, request_cls)

        if _has_deps:
            dep_kwargs = await resolve_deps(
                model, stream.metadata, stream.peer, method, app_state
            )
            kwargs = {**kwargs, **dep_kwargs}

        raw_result = await invoke_model(model, kwargs, batcher=batcher)

        response_bytes = _encode_response(raw_result, model, response_cls)
        await stream.send_message(response_bytes)

    return _handler


def _make_streaming_handler(
    model: ModelInfo,
    request_cls: type,
    response_cls: type,
    *,
    method: str = "",
    app_state: Any | None = None,
) -> Callable[..., Any]:
    """Return an ``async def handler(stream)`` for a server-streaming RPC."""
    _has_deps = bool(model.dep_params or model.context_params)

    async def _handler(stream: Stream[Any, Any]) -> None:
        request_bytes = await stream.recv_message()
        kwargs = _decode_request(request_bytes, model, request_cls)

        if _has_deps:
            dep_kwargs = await resolve_deps(
                model, stream.metadata, stream.peer, method, app_state
            )
            kwargs = {**kwargs, **dep_kwargs}

        async for chunk in invoke_streaming_model(model, kwargs):
            response_bytes = _encode_response(chunk, model, response_cls)
            await stream.send_message(response_bytes)

    return _handler


# ---------------------------------------------------------------------------
# Wire encode / decode helpers
# ---------------------------------------------------------------------------


def _decode_request(raw: Any, model: ModelInfo, request_cls: type) -> dict[str, Any]:
    """Deserialize binary Protobuf *raw* bytes into model function kwargs.

    Uses the betterproto ``request_cls`` generated for this model to parse
    the bytes, then converts ``TensorProto`` fields back to numpy arrays.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw

    msg = request_cls().parse(raw)
    kwargs: dict[str, Any] = {}
    for field_name, field_type in model.input_types.items():
        value = getattr(msg, field_name, None)
        if isinstance(field_type, _TensorType):
            # Nested TensorProto betterproto message → numpy array
            kwargs[field_name] = deserialize_tensor(
                _to_serialization_tensor_proto(value)
            )
        else:
            kwargs[field_name] = value
    return kwargs


def _encode_response(result: Any, model: ModelInfo, response_cls: type) -> bytes:
    """Serialize the model result into binary Protobuf bytes.

    Numpy arrays are wrapped in a betterproto ``TensorProto`` message.
    Scalars and lists are stored directly in the ``result`` field.
    """
    if model.output_type is None:
        # No return annotation — emit an empty message.
        return bytes(response_cls())

    if isinstance(result, np.ndarray):
        tensor_proto = serialize_tensor(result)
        tp_msg = _TensorProtoMsg(
            shape=list(tensor_proto.shape),
            dtype=tensor_proto.dtype,
            data=tensor_proto.data,
        )
        msg = response_cls(result=tp_msg)
        return bytes(msg)

    msg = response_cls(result=result)
    return bytes(msg)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _to_serialization_tensor_proto(msg: Any) -> Any:
    """Convert a betterproto TensorProto message to a serialization.TensorProto."""
    from blazerpc.runtime.serialization import TensorProto as _SP

    return _SP(
        shape=tuple(msg.shape),
        dtype=msg.dtype,
        data=msg.data,
    )

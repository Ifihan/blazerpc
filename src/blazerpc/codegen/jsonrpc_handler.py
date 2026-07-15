"""JSON-RPC 2.0 request dispatcher.

Maps JSON-RPC method names to registered BlazeRPC models and handles
the full request/response lifecycle including serialization, dependency
injection, and error mapping.

Method naming convention:
    - ``predict.<model_name>`` — unary model call
    - ``predict.<model_name>.v<version>`` — non-v1 unary model call
    - ``stream.<model_name>`` — server-streaming (must use the SSE endpoint)
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, AsyncIterator

import numpy as np

from blazerpc.codegen.invoke import invoke_model, invoke_streaming_model, resolve_deps
from blazerpc.exceptions import (
    InferenceError,
    ModelNotFoundError,
    SerializationError,
    ValidationError,
)
from blazerpc.runtime.json_serialization import json_to_python, tensor_to_json
from blazerpc.runtime.registry import ModelInfo, ModelRegistry, batcher_key
from blazerpc.types import _TensorType

log = logging.getLogger("blazerpc.jsonrpc")

# JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _error_response(
    req_id: Any, code: int, message: str, data: Any = None
) -> dict[str, Any]:
    resp: dict[str, Any] = {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": req_id,
    }
    if data is not None:
        resp["error"]["data"] = data
    return resp


def _success_response(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "result": result, "id": req_id}


class JsonRpcDispatcher:
    """Dispatches JSON-RPC 2.0 requests to registered BlazeRPC models."""

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

    # ------------------------------------------------------------------
    # Unary dispatch
    # ------------------------------------------------------------------

    async def handle(
        self,
        body: Any,
        *,
        peer: str = "",
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Process a single JSON-RPC request and return a response dict."""
        if not isinstance(body, dict):
            return _error_response(None, INVALID_REQUEST, "Invalid JSON-RPC request")

        req_id = body.get("id")
        is_notification = "id" not in body

        # Validate envelope
        if body.get("jsonrpc") != "2.0" or not isinstance(body.get("method"), str):
            return _error_response(req_id, INVALID_REQUEST, "Invalid JSON-RPC request")

        method: str = body["method"]
        params: dict[str, Any] = body.get("params", {})

        if not isinstance(params, dict):
            response = _error_response(
                req_id, INVALID_PARAMS, "params must be an object"
            )
            return None if is_notification else response

        # Parse method name
        model = self._resolve_method(method)
        if model is None:
            response = _error_response(
                req_id,
                METHOD_NOT_FOUND,
                f"Unknown method: {method}. Use predict.<model> or stream.<model>",
            )
            return None if is_notification else response

        # Streaming models cannot be called on the unary endpoint
        if model.streaming or method.startswith("stream."):
            response = _error_response(
                req_id,
                INVALID_REQUEST,
                f"Streaming model '{model.name}' must use the SSE endpoint",
            )
            return None if is_notification else response

        # Decode params → model kwargs
        try:
            kwargs = _decode_json_request(params, model)
        except (ValidationError, SerializationError) as exc:
            response = _error_response(req_id, INVALID_PARAMS, str(exc))
            return None if is_notification else response

        # Resolve dependencies
        _has_deps = bool(model.dep_params or model.context_params)
        if _has_deps:
            dep_kwargs = await resolve_deps(
                model, headers or {}, peer, method, self._app_state
            )
            kwargs = {**kwargs, **dep_kwargs}

        # Execute
        try:
            batcher = self._batchers.get(batcher_key(model.name, model.version))
            raw_result = await invoke_model(model, kwargs, batcher=batcher)
        except InferenceError as exc:
            response = _error_response(req_id, INTERNAL_ERROR, str(exc))
            return None if is_notification else response

        # Encode response
        try:
            result = _encode_json_response(raw_result, model)
        except SerializationError as exc:
            response = _error_response(
                req_id, INTERNAL_ERROR, f"Invalid model output: {exc}"
            )
            return None if is_notification else response
        return None if is_notification else _success_response(req_id, result)

    # ------------------------------------------------------------------
    # Batch dispatch (JSON-RPC spec allows arrays)
    # ------------------------------------------------------------------

    async def handle_batch(
        self,
        requests: list[Any],
        *,
        peer: str = "",
        headers: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Process a JSON-RPC batch (array of requests)."""
        import asyncio

        tasks = [self.handle(req, peer=peer, headers=headers) for req in requests]
        results = await asyncio.gather(*tasks)
        return [result for result in results if result is not None]

    # ------------------------------------------------------------------
    # Streaming dispatch (for SSE endpoint)
    # ------------------------------------------------------------------

    async def handle_streaming(
        self,
        method: str,
        params: dict[str, Any],
        *,
        peer: str = "",
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[Any]:
        """Yield chunks from a streaming model for SSE delivery."""
        model = self.validate_streaming(method, params)
        kwargs = _decode_json_request(params, model)

        _has_deps = bool(model.dep_params or model.context_params)
        if _has_deps:
            dep_kwargs = await resolve_deps(
                model, headers or {}, peer, method, self._app_state
            )
            kwargs = {**kwargs, **dep_kwargs}

        async for chunk in invoke_streaming_model(model, kwargs):
            yield _encode_json_response(chunk, model)

    def validate_streaming(self, method: str, params: Any) -> ModelInfo:
        """Validate an SSE call without resolving dependencies or invoking a model."""
        if not isinstance(method, str) or not method.startswith("stream."):
            raise ValidationError("Streaming method must use stream.<model>")
        if not isinstance(params, dict):
            raise ValidationError("params must be an object")

        model = self._resolve_method(method)
        if model is None:
            raise ModelNotFoundError(method.removeprefix("stream."), "")
        if not model.streaming:
            raise ValidationError(f"Model '{model.name}' is not streaming")
        _decode_json_request(params, model)
        return model

    def _resolve_method(self, method: str) -> ModelInfo | None:
        """Resolve an exact versioned method without discarding its version."""
        for model in self._registry.list_models():
            if method in {
                jsonrpc_method("predict", model.name, model.version),
                jsonrpc_method("stream", model.name, model.version),
            }:
                return model
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_method(method: str) -> str | None:
    """Extract model name from a JSON-RPC method string.

    Accepts ``predict.<name>`` or ``stream.<name>``.
    Returns ``None`` if the format is unrecognised.
    """
    for prefix in ("predict.", "stream."):
        if method.startswith(prefix):
            name = method[len(prefix) :]
            if name:
                return name
    return None


def jsonrpc_method(kind: str, name: str, version: str = "1") -> str:
    """Build a JSON-RPC method, retaining the historical v1 spelling."""
    suffix = "" if version == "1" else f".v{version}"
    return f"{kind}.{name}{suffix}"


def _decode_json_request(params: dict[str, Any], model: ModelInfo) -> dict[str, Any]:
    """Convert JSON params dict into model function kwargs.

    Tensor fields (identified by ``_TensorType`` annotations) are
    converted from base64 JSON dicts to numpy arrays.
    """
    unknown = params.keys() - model.input_types.keys()
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValidationError(f"Unknown parameter(s): {names}")

    signature = inspect.signature(model.func)
    missing = [
        name
        for name in model.input_types
        if name not in params
        and signature.parameters[name].default is inspect.Parameter.empty
    ]
    if missing:
        names = ", ".join(missing)
        raise ValidationError(f"Missing required parameter(s): {names}")

    kwargs: dict[str, Any] = {}
    for field_name, value in params.items():
        kwargs[field_name] = json_to_python(value, model.input_types[field_name])
    return kwargs


def _encode_json_response(result: Any, model: ModelInfo) -> Any:
    """Convert model output to a JSON-safe value.

    Numpy arrays are converted to base64 tensor dicts.
    Scalars and lists pass through unchanged.
    """
    if model.output_type is None:
        return None

    if isinstance(model.output_type, _TensorType):
        if not isinstance(result, np.ndarray):
            raise SerializationError(
                f"Expected numpy array for tensor output, got {type(result).__name__}"
            )
        return tensor_to_json(result, model.output_type)

    if isinstance(result, (list, tuple)):
        # Check if elements are numpy arrays
        return [
            tensor_to_json(item) if isinstance(item, np.ndarray) else item
            for item in result
        ]

    return result

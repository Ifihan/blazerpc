"""Tests for BlazeApp."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import numpy as np
import pytest

from blazerpc import TensorInput, TensorOutput
from blazerpc.app import BlazeApp, _make_batch_inference_fn
from blazerpc.exceptions import ModelNotFoundError, ValidationError
from blazerpc.codegen.invoke import invoke_model
from blazerpc.server.middleware import (
    LoggingMiddleware,
    TransportLoggingMiddleware,
)


def test_app_creation() -> None:
    app = BlazeApp()
    assert app.name == "blazerpc"
    assert app.enable_batching is True


def test_app_creation_no_batching() -> None:
    app = BlazeApp(enable_batching=False)
    assert app.enable_batching is False


def test_model_decorator(app: BlazeApp) -> None:
    @app.model("test_model")
    def predict(text: list[str]) -> list[float]:
        return [1.0]

    model_info = app.registry.get("test_model")
    assert model_info is not None
    assert model_info.name == "test_model"
    assert model_info.version == "1"
    assert model_info.streaming is False


def test_model_decorator_stores_type_info(app: BlazeApp) -> None:
    @app.model("typed_model")
    def predict(text: list[str], count: int) -> list[float]:
        return [1.0]

    model_info = app.registry.get("typed_model")
    assert "text" in model_info.input_types
    assert "count" in model_info.input_types
    assert model_info.output_type is not None


def test_model_not_found(app: BlazeApp) -> None:
    with pytest.raises(ModelNotFoundError):
        app.registry.get("nonexistent")


def test_exact_duplicate_model_registration_is_rejected(app: BlazeApp) -> None:
    @app.model("echo", version="2")
    def echo_v2(text: str) -> str:
        return text

    with pytest.raises(ValidationError, match="already registered"):

        @app.model("echo", version="2")
        def replacement(text: str) -> str:
            return text.upper()


def test_multiple_model_versions_are_registered_separately(app: BlazeApp) -> None:
    @app.model("echo")
    def echo_v1(text: str) -> str:
        return text

    @app.model("echo", version="2")
    def echo_v2(text: str) -> str:
        return text.upper()

    assert app.registry.get("echo").func is echo_v1
    assert app.registry.get("echo", "2").func is echo_v2


@pytest.mark.parametrize("name", ["", "123model", "-model", "bad.name", "model/name"])
def test_invalid_generated_model_identifiers_are_rejected(name: str) -> None:
    app = BlazeApp(enable_batching=False)

    with pytest.raises(ValidationError, match="valid Protobuf identifier"):

        @app.model(name)
        def model(value: str) -> str:
            return value


def test_invalid_parameter_identifier_is_rejected() -> None:
    app = BlazeApp(enable_batching=False)

    def model(value: str) -> str:
        return value

    unicode_name = "na\N{LATIN SMALL LETTER I WITH DIAERESIS}ve"
    model.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(unicode_name, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    model.__annotations__ = {unicode_name: str, "return": str}

    with pytest.raises(ValidationError, match="valid Protobuf field name"):
        app.registry.register("model", "1", model)


def test_sanitized_model_name_collision_is_rejected(app: BlazeApp) -> None:
    @app.model("foo-bar")
    def first(value: str) -> str:
        return value

    with pytest.raises(ValidationError, match="collides.*sanitization"):

        @app.model("foo_bar")
        def second(value: str) -> str:
            return value


def test_version_derived_model_name_collision_is_rejected(app: BlazeApp) -> None:
    @app.model("foo-v2")
    def first(value: str) -> str:
        return value

    with pytest.raises(ValidationError, match="collides.*sanitization"):

        @app.model("foo", version="2")
        def second(value: str) -> str:
            return value


@pytest.mark.parametrize(
    "annotation",
    [
        Any,
        object,
        str | None,
        dict[str, str],
        list,
        list[list[str]],
        list[TensorInput[np.float32, 1]],
    ],
)
def test_unsupported_input_annotations_are_rejected(annotation: Any) -> None:
    app = BlazeApp(enable_batching=False)

    def model(value: str) -> str:
        return value

    model.__annotations__["value"] = annotation
    with pytest.raises(
        ValidationError, match="annotation|List annotations|Nested lists"
    ):
        app.registry.register("bad_input", "1", model)


def test_unsupported_return_annotation_is_rejected() -> None:
    app = BlazeApp(enable_batching=False)

    def model(value: str) -> object:
        return value

    with pytest.raises(ValidationError, match="Unsupported Protobuf annotation"):
        app.registry.register("bad_output", "1", model)


def test_protobuf_keyword_parameter_is_rejected() -> None:
    app = BlazeApp(enable_batching=False)

    def model(value: str) -> str:
        return value

    model.__annotations__ = {"string": str, "return": str}
    model.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter("string", inspect.Parameter.POSITIONAL_OR_KEYWORD)],
        return_annotation=str,
    )
    with pytest.raises(ValidationError, match="valid Protobuf field name"):
        app.registry.register("keyword_field", "1", model)


def test_streaming_declaration_requires_generator_function() -> None:
    app = BlazeApp(enable_batching=False)

    with pytest.raises(ValidationError, match="generator or async generator"):

        @app.model("not_a_stream", streaming=True)
        def model(value: str) -> str:
            return value


def test_generator_function_requires_streaming_declaration() -> None:
    app = BlazeApp(enable_batching=False)

    with pytest.raises(ValidationError, match="must be declared streaming"):

        @app.model("undeclared_stream")
        def model(value: str) -> str:
            yield value


def test_app_accepts_middleware() -> None:
    mw = LoggingMiddleware()
    app = BlazeApp(middleware=[mw])
    assert app.middleware == [mw]


def test_app_middleware_defaults_to_empty() -> None:
    app = BlazeApp()
    assert app.middleware == []
    assert app.jsonrpc_middleware == []


def test_app_keeps_jsonrpc_middleware_separate() -> None:
    grpc_middleware = LoggingMiddleware()
    jsonrpc_middleware = TransportLoggingMiddleware()
    app = BlazeApp(
        middleware=[grpc_middleware],
        jsonrpc_middleware=[jsonrpc_middleware],
    )

    assert app.middleware == [grpc_middleware]
    assert app.jsonrpc_middleware == [jsonrpc_middleware]


def test_app_configures_batch_queue_capacity() -> None:
    app = BlazeApp(max_queue_size=17)
    assert app.max_queue_size == 17


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"batch_timeout_ms": -1}, "batch_timeout_ms"),
        ({"batch_timeout_ms": float("nan")}, "batch_timeout_ms"),
        ({"max_queue_size": 0}, "max_queue_size"),
    ],
)
def test_app_rejects_invalid_batch_configuration(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BlazeApp(**kwargs)


@pytest.mark.asyncio
async def test_tensor_requests_are_combined_into_one_model_call_and_split() -> None:
    app = BlazeApp()
    calls: list[tuple[int, ...]] = []

    @app.model("double")
    def double(
        values: TensorInput[np.float32, "batch", 2],  # noqa: F821
    ) -> TensorOutput[np.float32, "batch", 2]:  # noqa: F821
        calls.append(values.shape)
        return values * 2

    inference_fn = _make_batch_inference_fn(app.registry.get("double"))
    results = await inference_fn(
        [
            {"values": np.array([[1, 2]], dtype=np.float32)},
            {"values": np.array([[3, 4], [5, 6]], dtype=np.float32)},
        ]
    )

    assert calls == [(3, 2)]
    np.testing.assert_array_equal(results[0], [[2, 4]])
    np.testing.assert_array_equal(results[1], [[6, 8], [10, 12]])


@pytest.mark.asyncio
async def test_tensor_requests_split_list_output_by_leading_dimension() -> None:
    app = BlazeApp()

    @app.model("labels")
    def labels(
        values: TensorInput[np.float32, "batch", 1],  # noqa: F821
    ) -> list[int]:
        return [int(value) for value in values[:, 0]]

    inference_fn = _make_batch_inference_fn(app.registry.get("labels"))
    results = await inference_fn(
        [
            {"values": np.array([[1], [2]], dtype=np.float32)},
            {"values": np.array([[3]], dtype=np.float32)},
        ]
    )

    assert results == [[1, 2], [3]]


@pytest.mark.asyncio
async def test_scalar_endpoint_executes_directly_without_batcher_delay() -> None:
    app = BlazeApp(batch_timeout_ms=10_000)
    calls = 0

    @app.model("add")
    def add(a: float, b: float) -> float:
        nonlocal calls
        calls += 1
        return a + b

    batchers = await app._create_batchers()
    model = app.registry.get("add")
    result = await asyncio.wait_for(
        invoke_model(model, {"a": 2.0, "b": 3.0}, batcher=batchers.get("add")),
        timeout=0.5,
    )

    assert batchers == {}
    assert result == 5.0
    assert calls == 1


@pytest.mark.asyncio
async def test_batchers_are_keyed_and_cleaned_up_per_model_version() -> None:
    app = BlazeApp()

    @app.model("double")
    def double_v1(
        values: TensorInput[np.float32, "batch", 1],  # noqa: F821
    ) -> TensorOutput[np.float32, "batch", 1]:  # noqa: F821
        return values * 2

    @app.model("double", version="2")
    def double_v2(
        values: TensorInput[np.float32, "batch", 1],  # noqa: F821
    ) -> TensorOutput[np.float32, "batch", 1]:  # noqa: F821
        return values * 3

    batchers = await app._create_batchers()
    assert set(batchers) == {"double", "double:v2"}
    assert batchers["double"] is not batchers["double:v2"]
    tasks = [batcher._task for batcher in batchers.values()]

    for batcher in batchers.values():
        await batcher.stop()

    assert all(task is not None and task.done() for task in tasks)

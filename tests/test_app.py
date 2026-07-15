"""Tests for BlazeApp."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from blazerpc import TensorInput, TensorOutput
from blazerpc.app import BlazeApp, _make_batch_inference_fn
from blazerpc.exceptions import ModelNotFoundError, ValidationError
from blazerpc.codegen.invoke import invoke_model
from blazerpc.server.middleware import LoggingMiddleware


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


def test_app_accepts_middleware() -> None:
    mw = LoggingMiddleware()
    app = BlazeApp(middleware=[mw])
    assert app.middleware == [mw]


def test_app_middleware_defaults_to_empty() -> None:
    app = BlazeApp()
    assert app.middleware == []


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
def test_app_rejects_invalid_batch_configuration(
    kwargs: dict, message: str
) -> None:
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

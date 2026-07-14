"""Tests for BlazeApp."""

from __future__ import annotations

import pytest

from blazerpc.app import BlazeApp
from blazerpc.exceptions import ModelNotFoundError
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

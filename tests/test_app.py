"""Tests for BlazeApp."""

from __future__ import annotations

from blazerpc.app import BlazeApp


def test_app_creation() -> None:
    app = BlazeApp()
    assert app.name == "blazerpc"
    assert app.batcher is not None


def test_app_creation_no_batching() -> None:
    app = BlazeApp(enable_batching=False)
    assert app.batcher is None


def test_model_decorator(app: BlazeApp) -> None:
    @app.model("test_model")
    def predict(text: list[str]) -> list[float]:
        return [1.0]

    model_info = app.registry.get("test_model")
    assert model_info is not None
    assert model_info.name == "test_model"
    assert model_info.version == "1"
    assert model_info.streaming is False

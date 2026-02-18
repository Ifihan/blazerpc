"""Unit tests for BlazeClient."""

from __future__ import annotations

from blazerpc.client import BlazeClient, _build_path


def test_build_path_simple() -> None:
    assert _build_path("echo") == "/blazerpc.InferenceService/PredictEcho"


def test_build_path_underscore() -> None:
    assert _build_path("my_model") == "/blazerpc.InferenceService/PredictMyModel"


def test_build_path_hyphen() -> None:
    assert _build_path("my-model") == "/blazerpc.InferenceService/PredictMyModel"


def test_client_defaults() -> None:
    client = BlazeClient()
    assert client._host == "127.0.0.1"
    assert client._port == 50051
    assert client._channel is None


def test_client_close_without_connect() -> None:
    client = BlazeClient()
    client.close()  # should not raise

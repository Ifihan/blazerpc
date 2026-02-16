"""Tests for proto generation and servicer generation."""

from __future__ import annotations

import numpy as np
import pytest
from grpclib.const import Cardinality

from blazerpc.app import BlazeApp
from blazerpc.codegen.proto import ProtoGenerator, _sanitize_name, _type_to_proto_field
from blazerpc.codegen.servicer import InferenceServicer, build_servicer
from blazerpc.types import TensorInput, TensorOutput, _TensorType


# ---------------------------------------------------------------------------
# _sanitize_name
# ---------------------------------------------------------------------------


class TestSanitizeName:
    def test_simple(self) -> None:
        assert _sanitize_name("sentiment") == "Sentiment"

    def test_underscore(self) -> None:
        assert _sanitize_name("text_classifier") == "TextClassifier"

    def test_hyphen(self) -> None:
        assert _sanitize_name("my-model") == "MyModel"

    def test_already_pascal(self) -> None:
        assert _sanitize_name("Model") == "Model"


# ---------------------------------------------------------------------------
# _type_to_proto_field
# ---------------------------------------------------------------------------


class TestTypeToProtoField:
    def test_float(self) -> None:
        proto_type, repeated = _type_to_proto_field(float)
        assert proto_type == "float"
        assert repeated is False

    def test_int(self) -> None:
        proto_type, repeated = _type_to_proto_field(int)
        assert proto_type == "int64"
        assert repeated is False

    def test_str(self) -> None:
        proto_type, repeated = _type_to_proto_field(str)
        assert proto_type == "string"
        assert repeated is False

    def test_bool(self) -> None:
        proto_type, repeated = _type_to_proto_field(bool)
        assert proto_type == "bool"
        assert repeated is False

    def test_list_float(self) -> None:
        proto_type, repeated = _type_to_proto_field(list[float])
        assert proto_type == "float"
        assert repeated is True

    def test_list_str(self) -> None:
        proto_type, repeated = _type_to_proto_field(list[str])
        assert proto_type == "string"
        assert repeated is True

    def test_tensor_type(self) -> None:
        t = TensorInput[np.float32, "batch", 224, 224, 3]
        proto_type, repeated = _type_to_proto_field(t)
        assert proto_type == "TensorProto"
        assert repeated is False

    def test_unknown_type(self) -> None:
        proto_type, repeated = _type_to_proto_field(object)
        assert proto_type == "bytes"
        assert repeated is False


# ---------------------------------------------------------------------------
# ProtoGenerator
# ---------------------------------------------------------------------------


class TestProtoGenerator:
    def _make_app_with_model(self) -> BlazeApp:
        app = BlazeApp(enable_batching=False)

        @app.model("sentiment")
        def predict(text: list[str]) -> list[float]:
            return [1.0]

        return app

    def test_generate_contains_syntax(self) -> None:
        app = self._make_app_with_model()
        proto = ProtoGenerator().generate(app.registry)
        assert 'syntax = "proto3";' in proto

    def test_generate_contains_package(self) -> None:
        app = self._make_app_with_model()
        proto = ProtoGenerator().generate(app.registry)
        assert "package blazerpc;" in proto

    def test_generate_contains_tensor_proto(self) -> None:
        app = self._make_app_with_model()
        proto = ProtoGenerator().generate(app.registry)
        assert "message TensorProto {" in proto

    def test_generate_contains_request_message(self) -> None:
        app = self._make_app_with_model()
        proto = ProtoGenerator().generate(app.registry)
        assert "message SentimentRequest {" in proto
        assert "repeated string text = 1;" in proto

    def test_generate_contains_response_message(self) -> None:
        app = self._make_app_with_model()
        proto = ProtoGenerator().generate(app.registry)
        assert "message SentimentResponse {" in proto
        assert "repeated float result = 1;" in proto

    def test_generate_contains_service(self) -> None:
        app = self._make_app_with_model()
        proto = ProtoGenerator().generate(app.registry)
        assert "service InferenceService {" in proto
        assert "rpc PredictSentiment(SentimentRequest)" in proto

    def test_generate_streaming_model(self) -> None:
        app = BlazeApp(enable_batching=False)

        @app.model("llm", streaming=True)
        def generate(prompt: str) -> str:
            return "token"

        proto = ProtoGenerator().generate(app.registry)
        assert "returns (stream LlmResponse);" in proto

    def test_generate_multiple_models(self) -> None:
        app = BlazeApp(enable_batching=False)

        @app.model("sentiment")
        def predict_sentiment(text: list[str]) -> list[float]:
            return [1.0]

        @app.model("classify")
        def predict_classify(text: str) -> int:
            return 0

        proto = ProtoGenerator().generate(app.registry)
        assert "message SentimentRequest {" in proto
        assert "message ClassifyRequest {" in proto
        assert "rpc PredictSentiment" in proto
        assert "rpc PredictClassify" in proto

    def test_generate_tensor_model(self) -> None:
        app = BlazeApp(enable_batching=False)

        @app.model("image")
        def predict(
            pixels: TensorInput[np.float32, "batch", 224, 224, 3],
        ) -> TensorOutput[np.float32, "batch", 1000]:
            ...

        proto = ProtoGenerator().generate(app.registry)
        assert "TensorProto pixels = 1;" in proto
        assert "TensorProto result = 1;" in proto

    def test_generate_multiple_inputs(self) -> None:
        app = BlazeApp(enable_batching=False)

        @app.model("multi")
        def predict(text: str, count: int) -> float:
            return 1.0

        proto = ProtoGenerator().generate(app.registry)
        assert "string text = 1;" in proto
        assert "int64 count = 2;" in proto


# ---------------------------------------------------------------------------
# InferenceServicer
# ---------------------------------------------------------------------------


class TestInferenceServicer:
    def test_build_servicer(self) -> None:
        app = BlazeApp(enable_batching=False)

        @app.model("sentiment")
        def predict(text: list[str]) -> list[float]:
            return [1.0]

        servicer = build_servicer(app.registry)
        assert isinstance(servicer, InferenceServicer)

    def test_mapping_has_correct_paths(self) -> None:
        app = BlazeApp(enable_batching=False)

        @app.model("sentiment")
        def predict(text: list[str]) -> list[float]:
            return [1.0]

        servicer = build_servicer(app.registry)
        mapping = servicer.__mapping__()
        assert "/blazerpc.InferenceService/PredictSentiment" in mapping

    def test_mapping_multiple_models(self) -> None:
        app = BlazeApp(enable_batching=False)

        @app.model("sentiment")
        def predict_s(text: list[str]) -> list[float]:
            return [1.0]

        @app.model("classify")
        def predict_c(text: str) -> int:
            return 0

        servicer = build_servicer(app.registry)
        mapping = servicer.__mapping__()
        assert len(mapping) == 2
        assert "/blazerpc.InferenceService/PredictSentiment" in mapping
        assert "/blazerpc.InferenceService/PredictClassify" in mapping

    def test_streaming_model_cardinality(self) -> None:
        app = BlazeApp(enable_batching=False)

        @app.model("llm", streaming=True)
        def generate(prompt: str) -> str:
            return "token"

        servicer = build_servicer(app.registry)
        mapping = servicer.__mapping__()
        handler = mapping["/blazerpc.InferenceService/PredictLlm"]
        assert handler.cardinality == Cardinality.UNARY_STREAM

    def test_unary_model_cardinality(self) -> None:
        app = BlazeApp(enable_batching=False)

        @app.model("sentiment")
        def predict(text: list[str]) -> list[float]:
            return [1.0]

        servicer = build_servicer(app.registry)
        mapping = servicer.__mapping__()
        handler = mapping["/blazerpc.InferenceService/PredictSentiment"]
        assert handler.cardinality == Cardinality.UNARY_UNARY

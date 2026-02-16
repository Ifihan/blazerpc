""".proto file generation from registered models."""

from __future__ import annotations

from typing import Any, get_args, get_origin

from blazerpc.runtime.registry import ModelInfo, ModelRegistry
from blazerpc.types import PYTHON_TYPE_MAP, _TensorType


def _sanitize_name(name: str) -> str:
    """Convert a model name to a valid proto identifier (PascalCase)."""
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_"))


def _type_to_proto_field(py_type: Any) -> tuple[str, bool]:
    """Map a Python type annotation to a proto field type.

    Returns ``(proto_type, is_repeated)``.
    """
    # _TensorType  →  use the embedded TensorProto message
    if isinstance(py_type, _TensorType):
        return "TensorProto", False

    # list[X]  →  repeated X
    origin = get_origin(py_type)
    if origin is list:
        args = get_args(py_type)
        if args:
            inner, _ = _type_to_proto_field(args[0])
            return inner, True
        return "bytes", True

    # dict[K, V]  →  not directly supported as a field; fall back to bytes
    if origin is dict:
        return "bytes", False

    # Plain Python scalars
    if isinstance(py_type, type) and py_type in PYTHON_TYPE_MAP:
        return PYTHON_TYPE_MAP[py_type], False

    return "bytes", False


class ProtoGenerator:
    """Generates ``.proto`` file content from a :class:`ModelRegistry`."""

    def generate(self, registry: ModelRegistry) -> str:
        """Return a complete ``.proto`` file as a string."""
        lines: list[str] = [
            'syntax = "proto3";',
            "",
            "package blazerpc;",
            "",
        ]

        # Shared TensorProto message (always emitted so models can reference it)
        lines += self._tensor_proto_message()

        models = registry.list_models()
        for model in models:
            lines += self._generate_request_message(model)
            lines += self._generate_response_message(model)

        lines += self._generate_service(models)
        return "\n".join(lines) + "\n"

    # -- private helpers --------------------------------------------------

    @staticmethod
    def _tensor_proto_message() -> list[str]:
        return [
            "message TensorProto {",
            "  repeated int64 shape = 1;",
            "  string dtype = 2;",
            "  bytes data = 3;",
            "}",
            "",
        ]

    @staticmethod
    def _generate_request_message(model: ModelInfo) -> list[str]:
        name = _sanitize_name(model.name)
        lines = [f"message {name}Request {{"]
        field_num = 1
        for param_name, param_type in model.input_types.items():
            proto_type, repeated = _type_to_proto_field(param_type)
            prefix = "repeated " if repeated else ""
            lines.append(f"  {prefix}{proto_type} {param_name} = {field_num};")
            field_num += 1
        lines += ["}", ""]
        return lines

    @staticmethod
    def _generate_response_message(model: ModelInfo) -> list[str]:
        name = _sanitize_name(model.name)
        lines = [f"message {name}Response {{"]
        if model.output_type is not None:
            proto_type, repeated = _type_to_proto_field(model.output_type)
            prefix = "repeated " if repeated else ""
            lines.append(f"  {prefix}{proto_type} result = 1;")
        lines += ["}", ""]
        return lines

    @staticmethod
    def _generate_service(models: list[ModelInfo]) -> list[str]:
        lines = ["service InferenceService {"]
        for model in models:
            name = _sanitize_name(model.name)
            if model.streaming:
                lines.append(
                    f"  rpc Predict{name}({name}Request) "
                    f"returns (stream {name}Response);"
                )
            else:
                lines.append(
                    f"  rpc Predict{name}({name}Request) " f"returns ({name}Response);"
                )
        lines += ["}", ""]
        return lines

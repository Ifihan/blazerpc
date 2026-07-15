"""Model registry for tracking registered inference endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import re
from typing import Any, Callable

from blazerpc.exceptions import ModelNotFoundError, ValidationError
from blazerpc.types import _TensorType, extract_type_info, validate_tensor_type


_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_MODEL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")
_PROTO_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_PROTO_KEYWORDS = {
    "bool",
    "bytes",
    "double",
    "enum",
    "fixed32",
    "fixed64",
    "float",
    "import",
    "int32",
    "int64",
    "map",
    "message",
    "oneof",
    "option",
    "package",
    "public",
    "repeated",
    "reserved",
    "returns",
    "rpc",
    "service",
    "sfixed32",
    "sfixed64",
    "sint32",
    "sint64",
    "stream",
    "string",
    "syntax",
    "to",
    "uint32",
    "uint64",
    "weak",
}


def model_key(name: str, version: str = "1") -> str:
    """Return the internal registry key for a model version."""
    return f"{name}:{version}"


def batcher_key(name: str, version: str = "1") -> str:
    """Return a version-aware batcher key, preserving the v1 key."""
    return name if version == "1" else f"{name}:v{version}"


@dataclass
class ModelInfo:
    name: str
    version: str
    func: Callable[..., object]
    streaming: bool = False
    input_types: dict[str, Any] = field(default_factory=dict)
    output_type: Any = None
    dep_params: dict[str, Any] = field(default_factory=dict)
    context_params: list[str] = field(default_factory=list)


class ModelRegistry:
    def __init__(self) -> None:
        self.models: dict[str, ModelInfo] = {}

    def register(
        self,
        name: str,
        version: str,
        func: Callable[..., object],
        streaming: bool = False,
    ) -> None:
        from blazerpc.codegen.proto import _model_proto_name, _type_to_proto_field

        if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
            raise ValidationError(
                "Model version must start with an ASCII letter or digit and contain "
                "only ASCII letters, digits, '.', '_', or '-'",
                field="version",
            )
        if not isinstance(streaming, bool):
            raise ValidationError("streaming must be a boolean", field="streaming")

        if not isinstance(name, str):
            raise ValidationError("Model name must be a string", field="name")
        candidate = ModelInfo(name=name, version=version, func=func)
        proto_name = _model_proto_name(candidate)
        if not _MODEL_NAME_RE.fullmatch(name) or not _PROTO_IDENTIFIER_RE.fullmatch(
            proto_name
        ):
            raise ValidationError(
                f"Model name {name!r} does not produce a valid Protobuf identifier",
                field="name",
            )

        key = model_key(name, version)
        if key in self.models:
            raise ValidationError(
                f"Model '{name}' version '{version}' is already registered",
                field=name,
            )
        for registered in self.models.values():
            if _model_proto_name(registered) == proto_name:
                raise ValidationError(
                    f"Model '{name}' version '{version}' collides with model "
                    f"'{registered.name}' version '{registered.version}' after "
                    "Protobuf name sanitization",
                    field=name,
                )

        type_info = extract_type_info(func)
        if type_info["output"] is type(None):
            type_info["output"] = None
        tensor_types = [
            hint
            for hint in (*type_info["inputs"].values(), type_info["output"])
            if isinstance(hint, _TensorType)
        ]
        for tensor_type in tensor_types:
            try:
                validate_tensor_type(tensor_type)
            except ValueError as exc:
                raise ValidationError(str(exc), field=name) from exc
        for param_name, annotation in type_info["inputs"].items():
            if (
                not _PROTO_IDENTIFIER_RE.fullmatch(param_name)
                or param_name in _PROTO_KEYWORDS
            ):
                raise ValidationError(
                    f"Parameter name {param_name!r} is not a valid Protobuf field name",
                    field=param_name,
                )
            try:
                _type_to_proto_field(annotation)
            except TypeError as exc:
                raise ValidationError(str(exc), field=param_name) from exc
        if type_info["output"] is not None:
            try:
                _type_to_proto_field(type_info["output"])
            except TypeError as exc:
                raise ValidationError(str(exc), field="return") from exc

        is_generator = inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(
            func
        )
        if streaming and not is_generator:
            raise ValidationError(
                "Streaming models must be generator or async generator functions",
                field=name,
            )
        if not streaming and is_generator:
            raise ValidationError(
                "Generator and async generator functions must be declared streaming",
                field=name,
            )
        total_params = (
            len(type_info["inputs"])
            + len(type_info["deps"])
            + len(type_info["context_params"])
        )
        if total_params == 0:
            raise ValidationError(
                f"Model '{name}' function must have at least one parameter "
                "with a type annotation",
                field=name,
            )

        self.models[key] = ModelInfo(
            name=name,
            version=version,
            func=func,
            streaming=streaming,
            input_types=type_info["inputs"],
            output_type=type_info["output"],
            dep_params=type_info["deps"],
            context_params=type_info["context_params"],
        )

    def get(self, name: str, version: str = "1") -> ModelInfo:
        model = self.models.get(model_key(name, version))
        if model is None:
            raise ModelNotFoundError(name, version)
        return model

    def get_or_none(self, name: str, version: str = "1") -> ModelInfo | None:
        return self.models.get(model_key(name, version))

    def list_models(self) -> list[ModelInfo]:
        return list(self.models.values())

"""Model registry for tracking registered inference endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from blazerpc.exceptions import ModelNotFoundError, ValidationError
from blazerpc.types import extract_type_info


@dataclass
class ModelInfo:
    name: str
    version: str
    func: Callable[..., object]
    streaming: bool = False
    input_types: dict[str, Any] = field(default_factory=dict)
    output_type: Any = None


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
        type_info = extract_type_info(func)
        if not type_info["inputs"]:
            raise ValidationError(
                f"Model '{name}' function must have at least one parameter "
                "with a type annotation",
                field=name,
            )

        key = f"{name}:{version}"
        self.models[key] = ModelInfo(
            name=name,
            version=version,
            func=func,
            streaming=streaming,
            input_types=type_info["inputs"],
            output_type=type_info["output"],
        )

    def get(self, name: str, version: str = "1") -> ModelInfo:
        model = self.models.get(f"{name}:{version}")
        if model is None:
            raise ModelNotFoundError(name, version)
        return model

    def get_or_none(self, name: str, version: str = "1") -> ModelInfo | None:
        return self.models.get(f"{name}:{version}")

    def list_models(self) -> list[ModelInfo]:
        return list(self.models.values())

"""Model registry for tracking registered inference endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ModelInfo:
    name: str
    version: str
    func: Callable[..., object]
    streaming: bool = False


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
        key = f"{name}:{version}"
        self.models[key] = ModelInfo(
            name=name, version=version, func=func, streaming=streaming
        )

    def get(self, name: str, version: str = "1") -> ModelInfo | None:
        return self.models.get(f"{name}:{version}")

    def list_models(self) -> list[ModelInfo]:
        return list(self.models.values())

"""TensorInput, TensorOutput, and the BlazeRPC type system."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

import numpy as np

DType = TypeVar("DType", bound=np.generic)
Shape = TypeVar("Shape")


class _TensorType:
    """Internal representation of a tensor type with shape and dtype info."""

    def __init__(
        self, dtype: type, shape: tuple[Any, ...], *, is_input: bool
    ) -> None:
        self.dtype = dtype
        self.shape = shape
        self.is_input = is_input

    def __repr__(self) -> str:
        kind = "TensorInput" if self.is_input else "TensorOutput"
        return f"{kind}[{self.dtype.__name__}, {self.shape}]"


class TensorInput(Generic[DType, Shape]):
    """Type annotation for tensor inputs with shape info."""

    @classmethod
    def __class_getitem__(cls, params: tuple[Any, ...]) -> _TensorType:
        dtype, *shape = params
        return _TensorType(dtype, tuple(shape), is_input=True)


class TensorOutput(Generic[DType, Shape]):
    """Type annotation for tensor outputs with shape info."""

    @classmethod
    def __class_getitem__(cls, params: tuple[Any, ...]) -> _TensorType:
        dtype, *shape = params
        return _TensorType(dtype, tuple(shape), is_input=False)

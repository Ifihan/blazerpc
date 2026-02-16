"""BlazeRPC exception hierarchy."""

from __future__ import annotations


class BlazeRPCError(Exception):
    """Base exception for all BlazeRPC errors."""


class ValidationError(BlazeRPCError):
    """Raised when input validation fails (bad shapes, types, missing annotations)."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.field = field
        super().__init__(message)


class ModelNotFoundError(BlazeRPCError):
    """Raised when a requested model is not found in the registry."""

    def __init__(self, name: str, version: str = "1") -> None:
        self.name = name
        self.version = version
        super().__init__(f"Model '{name}' version '{version}' not found")


class SerializationError(BlazeRPCError):
    """Raised when tensor serialization or deserialization fails."""

    def __init__(self, message: str, *, dtype: str | None = None) -> None:
        self.dtype = dtype
        super().__init__(message)


class InferenceError(BlazeRPCError):
    """Raised when model inference fails."""

    def __init__(
        self, message: str, *, model_name: str | None = None
    ) -> None:
        self.model_name = model_name
        super().__init__(message)


class ConfigurationError(BlazeRPCError):
    """Raised for invalid configuration (bad import paths, missing settings)."""

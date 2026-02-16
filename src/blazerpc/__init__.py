"""BlazeRPC - A lightweight, framework-agnostic gRPC library for ML inference."""

from blazerpc.app import BlazeApp
from blazerpc.exceptions import (
    BlazeRPCError,
    ConfigurationError,
    InferenceError,
    ModelNotFoundError,
    SerializationError,
    ValidationError,
)
from blazerpc.types import TensorInput, TensorOutput

__version__ = "0.1.0"
__all__ = [
    "BlazeApp",
    "BlazeRPCError",
    "ConfigurationError",
    "InferenceError",
    "ModelNotFoundError",
    "SerializationError",
    "TensorInput",
    "TensorOutput",
    "ValidationError",
]

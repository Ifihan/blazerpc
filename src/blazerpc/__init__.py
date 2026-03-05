"""BlazeRPC - A lightweight, framework-agnostic gRPC library for ML inference."""

from blazerpc.app import BlazeApp
from blazerpc.client import BlazeClient
from blazerpc.context import Context, Depends
from blazerpc.exceptions import (
    BlazeRPCError,
    ConfigurationError,
    InferenceError,
    ModelNotFoundError,
    SerializationError,
    ValidationError,
)
from blazerpc.types import TensorInput, TensorOutput

__version__ = "2.0.0"
__all__ = [
    "BlazeApp",
    "BlazeClient",
    "BlazeRPCError",
    "ConfigurationError",
    "Context",
    "Depends",
    "InferenceError",
    "ModelNotFoundError",
    "SerializationError",
    "TensorInput",
    "TensorOutput",
    "ValidationError",
]

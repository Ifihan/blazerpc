"""BlazeRPC - A lightweight, framework-agnostic RPC library for ML inference."""

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

# Lazy import to avoid requiring aiohttp when only using gRPC
try:
    from blazerpc.jsonrpc_client import JsonRpcClient
except ImportError:
    pass

__version__ = "2.2.0"
__all__ = [
    "BlazeApp",
    "BlazeClient",
    "BlazeRPCError",
    "ConfigurationError",
    "Context",
    "Depends",
    "InferenceError",
    "JsonRpcClient",
    "ModelNotFoundError",
    "SerializationError",
    "TensorInput",
    "TensorOutput",
    "ValidationError",
]

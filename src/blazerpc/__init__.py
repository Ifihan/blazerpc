"""BlazeRPC - A lightweight, framework-agnostic RPC library for ML inference."""

import re
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

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

try:
    __version__ = version("blazerpc")
except PackageNotFoundError:
    # Distribution metadata is absent when importing directly from a checkout.
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    match = re.search(
        r'^\[project\].*?^version\s*=\s*"([^"]+)"',
        pyproject.read_text(encoding="utf-8"),
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError("Unable to determine BlazeRPC version from pyproject.toml")
    __version__ = match.group(1)

try:
    _has_aiohttp = find_spec("aiohttp") is not None
except ModuleNotFoundError:
    _has_aiohttp = False

if _has_aiohttp:
    from blazerpc.jsonrpc_client import JsonRpcClient as JsonRpcClient

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

if "JsonRpcClient" in globals():
    __all__.append("JsonRpcClient")


def __getattr__(name: str) -> object:
    if name == "JsonRpcClient":
        from blazerpc.jsonrpc_client import JsonRpcClient

        return JsonRpcClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

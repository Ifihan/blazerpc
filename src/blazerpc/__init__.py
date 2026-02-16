"""BlazeRPC - A lightweight, framework-agnostic gRPC library for ML inference."""

from blazerpc.app import BlazeApp
from blazerpc.types import TensorInput, TensorOutput

__version__ = "0.1.0"
__all__ = ["BlazeApp", "TensorInput", "TensorOutput"]

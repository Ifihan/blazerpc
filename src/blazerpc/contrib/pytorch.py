"""PyTorch integration for BlazeRPC.

Provides helpers to convert between PyTorch tensors and NumPy arrays,
and a ``@torch_model`` decorator that handles the conversion
automatically.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np
import torch


def torch_to_numpy(tensor: Any) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array.

    Detaches from the computation graph and moves to CPU if needed.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")
    return tensor.detach().cpu().numpy()


def numpy_to_torch(
    arr: np.ndarray,
    device: str = "cpu",
    dtype: Any = None,
) -> Any:
    """Convert a NumPy array to a PyTorch tensor.

    Parameters
    ----------
    arr:
        Source array.
    device:
        Target device (``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.).
    dtype:
        Optional torch dtype override.
    """
    tensor = torch.from_numpy(arr)
    if dtype is not None:
        tensor = tensor.to(dtype)
    if device != "cpu":
        tensor = tensor.to(device)
    return tensor


def torch_model(
    func: Callable | None = None,
    *,
    device: str = "cpu",
) -> Callable:
    """Decorator that auto-converts NumPy inputs to torch tensors and back.

    Usage::

        @app.model("classifier")
        @torch_model(device="cuda")
        def classify(image: np.ndarray) -> np.ndarray:
            # `image` is already a torch.Tensor on the specified device
            return model(image)
            # Return value is converted back to np.ndarray automatically
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            converted_args = [
                numpy_to_torch(a, device=device)
                if isinstance(a, np.ndarray)
                else a
                for a in args
            ]
            converted_kwargs = {
                k: numpy_to_torch(v, device=device)
                if isinstance(v, np.ndarray)
                else v
                for k, v in kwargs.items()
            }

            result = fn(*converted_args, **converted_kwargs)

            if isinstance(result, torch.Tensor):
                return torch_to_numpy(result)
            return result

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator

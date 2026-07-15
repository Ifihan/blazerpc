"""TensorFlow integration for BlazeRPC.

Provides helpers to convert between TensorFlow tensors and NumPy
arrays, and a ``@tf_model`` decorator that handles the conversion
automatically.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar, cast

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

_F = TypeVar("_F", bound=Callable[..., Any])


def tf_to_numpy(tensor: Any) -> NDArray[Any]:
    """Convert a TensorFlow tensor to a NumPy array."""
    if not isinstance(tensor, tf.Tensor):
        raise TypeError(f"Expected tf.Tensor, got {type(tensor).__name__}")
    return cast(NDArray[Any], tensor.numpy())


def numpy_to_tf(arr: NDArray[Any], dtype: Any = None) -> Any:
    """Convert a NumPy array to a TensorFlow tensor.

    Parameters
    ----------
    arr:
        Source array.
    dtype:
        Optional TensorFlow dtype override.
    """
    tensor = tf.convert_to_tensor(arr)
    if dtype is not None:
        tensor = tf.cast(tensor, dtype)
    return tensor


def tf_model(
    func: _F | None = None,
    *,
    dtype: Any = None,
) -> _F | Callable[[_F], _F]:
    """Decorator that auto-converts NumPy inputs to TF tensors and back.

    Usage::

        @app.model("classifier")
        @tf_model
        def classify(image: np.ndarray) -> np.ndarray:
            # `image` is already a tf.Tensor
            return model(image)
            # Return value is converted back to np.ndarray automatically
    """

    def decorator(fn: _F) -> _F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            converted_args = [
                numpy_to_tf(a, dtype=dtype) if isinstance(a, np.ndarray) else a
                for a in args
            ]
            converted_kwargs = {
                k: numpy_to_tf(v, dtype=dtype) if isinstance(v, np.ndarray) else v
                for k, v in kwargs.items()
            }

            result = fn(*converted_args, **converted_kwargs)

            if isinstance(result, tf.Tensor):
                return tf_to_numpy(result)
            return result

        return cast(_F, wrapper)

    if func is not None:
        return decorator(func)
    return decorator

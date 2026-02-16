"""TensorFlow integration for BlazeRPC.

Provides helpers to convert between TensorFlow tensors and NumPy
arrays, and a ``@tf_model`` decorator that handles the conversion
automatically.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np


def tf_to_numpy(tensor: Any) -> np.ndarray:
    """Convert a TensorFlow tensor to a NumPy array."""
    import tensorflow as tf

    if not isinstance(tensor, tf.Tensor):
        raise TypeError(f"Expected tf.Tensor, got {type(tensor).__name__}")
    return tensor.numpy()


def numpy_to_tf(arr: np.ndarray, dtype: Any = None) -> Any:
    """Convert a NumPy array to a TensorFlow tensor.

    Parameters
    ----------
    arr:
        Source array.
    dtype:
        Optional TensorFlow dtype override.
    """
    import tensorflow as tf

    tensor = tf.convert_to_tensor(arr)
    if dtype is not None:
        tensor = tf.cast(tensor, dtype)
    return tensor


def tf_model(
    func: Callable | None = None,
    *,
    dtype: Any = None,
) -> Callable:
    """Decorator that auto-converts NumPy inputs to TF tensors and back.

    Usage::

        @app.model("classifier")
        @tf_model
        def classify(image: np.ndarray) -> np.ndarray:
            # `image` is already a tf.Tensor
            return model(image)
            # Return value is converted back to np.ndarray automatically
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            converted_args = [
                numpy_to_tf(a, dtype=dtype)
                if isinstance(a, np.ndarray)
                else a
                for a in args
            ]
            converted_kwargs = {
                k: numpy_to_tf(v, dtype=dtype)
                if isinstance(v, np.ndarray)
                else v
                for k, v in kwargs.items()
            }

            result = fn(*converted_args, **converted_kwargs)

            import tensorflow as tf

            if isinstance(result, tf.Tensor):
                return tf_to_numpy(result)
            return result

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator

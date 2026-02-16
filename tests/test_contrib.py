"""Tests for framework integrations (contrib).

PyTorch, TensorFlow, and ONNX are optional dependencies so these tests
are conditional — they skip if the framework is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")

from blazerpc.contrib.pytorch import numpy_to_torch, torch_model, torch_to_numpy


class TestPyTorch:
    def test_torch_to_numpy(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = torch_to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_torch_to_numpy_bad_type(self) -> None:
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            torch_to_numpy([1, 2, 3])

    def test_numpy_to_torch(self) -> None:
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = numpy_to_torch(arr)
        assert isinstance(t, torch.Tensor)
        assert t.device.type == "cpu"

    def test_numpy_to_torch_with_dtype(self) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float32)
        t = numpy_to_torch(arr, dtype=torch.float64)
        assert t.dtype == torch.float64

    def test_torch_model_decorator(self) -> None:
        @torch_model
        def double_it(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = double_it(x=arr)
        # Result should be converted back to numpy
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0])

    def test_torch_model_decorator_with_device(self) -> None:
        @torch_model(device="cpu")
        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        arr = np.array([1.0], dtype=np.float32)
        result = identity(x=arr)
        assert isinstance(result, np.ndarray)

    def test_torch_model_passthrough_non_array(self) -> None:
        @torch_model
        def with_scalar(x: torch.Tensor, scale: float) -> torch.Tensor:
            return x * scale

        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = with_scalar(x=arr, scale=3.0)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [3.0, 6.0])

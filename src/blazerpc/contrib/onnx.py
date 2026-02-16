"""ONNX Runtime integration for BlazeRPC.

Provides an ``ONNXModel`` wrapper that manages an ONNX Runtime
inference session and exposes a simple ``predict()`` method compatible
with BlazeRPC's model registration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


class ONNXModel:
    """Wrapper around an ONNX Runtime inference session.

    Usage::

        onnx_model = ONNXModel("model.onnx")

        @app.model("classifier")
        def classify(image: np.ndarray) -> np.ndarray:
            return onnx_model.predict(image)

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` model file.
    providers:
        Execution providers (defaults to ``["CPUExecutionProvider"]``).
    session_options:
        Optional ``onnxruntime.SessionOptions`` instance.
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
        session_options: Any = None,
    ) -> None:
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            str(model_path),
            providers=providers,
            sess_options=session_options,
        )
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

    @property
    def input_names(self) -> list[str]:
        """Names of the model's input tensors."""
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        """Names of the model's output tensors."""
        return list(self._output_names)

    def predict(self, *inputs: np.ndarray) -> list[np.ndarray]:
        """Run inference on the given input arrays.

        Positional arguments are matched to input names in order.
        Returns a list of output arrays.
        """
        if len(inputs) != len(self._input_names):
            raise ValueError(
                f"Expected {len(self._input_names)} inputs "
                f"({self._input_names}), got {len(inputs)}"
            )

        feed = dict(zip(self._input_names, inputs))
        return self._session.run(self._output_names, feed)

    def predict_dict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with named inputs, returning named outputs."""
        results = self._session.run(self._output_names, inputs)
        return dict(zip(self._output_names, results))

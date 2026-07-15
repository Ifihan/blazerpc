"""Consumer-side typing fixture for the public tensor annotation API."""

from typing import Literal

import numpy as np

from blazerpc import TensorInput, TensorOutput


ImageShape = tuple[Literal["batch"], Literal[224], Literal[224], Literal[3]]
ScoresShape = tuple[Literal["batch"], Literal[1000]]


def classify(
    image: TensorInput[np.float32, ImageShape],
) -> TensorOutput[np.float32, ScoresShape]:
    return np.zeros((image.shape[0], 1000), dtype=np.float32)

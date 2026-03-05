"""
Client for the Iris classifier service.

Connects to a running BlazeRPC server and calls the ``PredictIris``
and ``PredictEcho`` RPCs using BlazeClient.

Prerequisites:
    1. pip install scikit-learn
    2. Start the server:   uv run blaze serve examples.simple.app:app
    3. Run this client:    uv run python examples/simple/client.py
"""

import asyncio

import numpy as np

from blazerpc import BlazeClient

from examples.simple.app import app

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


async def main() -> None:
    async with BlazeClient("127.0.0.1", 50051, registry=app.registry) as client:
        # Unary call: echo
        reply = await client.predict("echo", text="BlazeRPC")
        print(f"echo → {reply}")

        # Unary call: classify two iris samples
        samples = np.array(
            [
                [5.1, 3.5, 1.4, 0.2],  # typical setosa
                [6.7, 3.0, 5.2, 2.3],  # typical virginica
            ],
            dtype=np.float32,
        )
        probs = await client.predict("iris", features=samples)
        print("\niris classification:")
        for i, sample in enumerate(samples):
            predicted = IRIS_CLASSES[np.argmax(probs[i])]
            print(f"  sample {i + 1} {sample.tolist()} → {predicted} ({probs[i]})")


if __name__ == "__main__":
    asyncio.run(main())

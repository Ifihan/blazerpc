"""
Batch-optimized image classification service.

This example demonstrates BlazeRPC's adaptive batching. When batching
is enabled, individual requests are automatically grouped into batches
before being passed to the model function. This is critical for GPU
workloads where processing a batch of 32 images is far more efficient
than processing 32 images one at a time.

How it works:
    1. Client A sends a request for one image.
    2. Client B sends a request for one image 2 ms later.
    3. BlazeRPC collects both requests into a single batch.
    4. The model function receives the batch and returns results for
       both images at once.
    5. Each client receives only its own result.

Batching parameters:
    - ``max_batch_size``: Maximum number of requests in a batch (default 32).
    - ``batch_timeout_ms``: Maximum time to wait for a full batch before
      dispatching a partial one (default 10 ms).

Run the server:
    uv run blaze serve examples.batching.app:app
"""

import numpy as np

from blazerpc import BlazeApp

app = BlazeApp(
    name="batching-demo",
    enable_batching=True,
    max_batch_size=16,
    batch_timeout_ms=5.0,
)


@app.model("classify")
def classify_image(image: list[float]) -> float:
    """Classify an image and return a confidence score.

    When batching is enabled, BlazeRPC collects multiple calls to this
    function into a single batch. The function itself does not need to
    handle batching logic -- BlazeRPC manages that transparently.

    In production, ``image`` would be a NumPy array or tensor
    representing pixel data, and the function body would call
    ``model.predict(batch)`` on a real classifier.
    """
    # Stub: return a confidence score based on the sum of pixel values.
    return float(np.mean(image))

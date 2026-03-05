"""
Batch-optimized Iris classifier service.

This example demonstrates BlazeRPC's adaptive batching with a real
scikit-learn model. When batching is enabled, individual requests are
automatically grouped into batches before being passed to the model
function. This is critical for GPU workloads where processing a batch
is far more efficient than processing items one at a time.

How it works:
    1. Client A sends a request with one iris sample.
    2. Client B sends a request with one iris sample 2 ms later.
    3. BlazeRPC collects both requests into a single batch.
    4. The model function receives the batch and returns results for
       both samples at once.
    5. Each client receives only its own result.

Prerequisites:
    pip install scikit-learn

Run the server:
    uv run blaze serve examples.batching.app:app
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from blazerpc import BlazeApp, TensorInput, TensorOutput

# Train a model
iris = load_iris()
clf = LogisticRegression(max_iter=200)
clf.fit(iris.data, iris.target)

app = BlazeApp(
    name="batching-demo",
    enable_batching=True,
    max_batch_size=16,
    batch_timeout_ms=5.0,
)


@app.model("iris")
def predict_iris(
    features: TensorInput[np.float32, "batch", 4],
) -> TensorOutput[np.float32, "batch", 3]:
    """Classify iris flowers with adaptive batching.

    When batching is enabled, BlazeRPC collects multiple calls to this
    function into a single batch. The function itself does not need to
    handle batching logic -- BlazeRPC manages that transparently.
    """
    probs = clf.predict_proba(features).astype(np.float32)
    return probs

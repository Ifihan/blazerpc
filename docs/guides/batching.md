# Adaptive batching

GPU inference is significantly faster when processing a batch of inputs at once compared to processing them one at a time. BlazeRPC's adaptive batcher collects individual requests from concurrent clients and groups them into batches before calling your model function.

## Enabling batching

Batching is enabled by default. Configure it through `BlazeApp`:

```python
from blazerpc import BlazeApp

app = BlazeApp(
    enable_batching=True,
    max_batch_size=32,
    batch_timeout_ms=10.0,
)
```

| Parameter          | Default | Description                                                          |
| ------------------ | ------- | -------------------------------------------------------------------- |
| `enable_batching`  | `True`  | Set to `False` to process every request individually.                |
| `max_batch_size`   | `32`    | Maximum number of requests collected into a single batch.            |
| `batch_timeout_ms` | `10.0`  | Maximum time (in milliseconds) to wait for a full batch before dispatching a partial one. |

## How it works

Batching is fully transparent — you write a normal single-item handler, and the framework handles collecting requests into batches and distributing results.

The batcher runs as a background `asyncio.Task` with the following loop:

1. **Wait** for the first request to arrive.
2. **Collect** additional requests until either `max_batch_size` is reached or `batch_timeout_ms` elapses.
3. **Dispatch** the collected batch to the model function (called once per item in the batch).
4. **Distribute** each result back to the corresponding client's future.

This means:

- Under high load, batches fill up quickly and are dispatched at full capacity.
- Under light load, the timeout ensures that a lone request is not stuck waiting for a batch to fill. A 10 ms timeout adds negligible latency.
- Your handler signature stays the same whether batching is on or off.

## Example

This example serves a scikit-learn Iris classifier with batching enabled. When multiple clients send classification requests within a short time window, BlazeRPC automatically groups them into a single batch:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from blazerpc import BlazeApp, TensorInput, TensorOutput

iris = load_iris()
clf = LogisticRegression(max_iter=200)
clf.fit(iris.data, iris.target)

app = BlazeApp(
    enable_batching=True,
    max_batch_size=16,
    batch_timeout_ms=5.0,
)

@app.model("iris")
def predict_iris(
    features: TensorInput[np.float32, "batch", 4],
) -> TensorOutput[np.float32, "batch", 3]:
    probs = clf.predict_proba(features).astype(np.float32)
    return probs
```

When three clients call `predict_iris` within 5 ms of each other, the batcher groups all three requests into a single batch. The model function runs once, and each client receives only its own result.

## Tuning

**`max_batch_size`** controls the upper bound on batch size. Set this based on your GPU memory and model throughput characteristics. Larger batches improve throughput but use more memory.

**`batch_timeout_ms`** controls latency under light load. Lower values reduce tail latency for individual requests. Higher values give the batcher more time to collect a full batch, improving throughput.

A good starting point:

- For latency-sensitive applications (real-time APIs): `batch_timeout_ms=5.0`, `max_batch_size=8`.
- For throughput-optimized workloads (offline processing): `batch_timeout_ms=50.0`, `max_batch_size=64`.

## Partial failure handling

If the model function raises an exception, every request in the batch receives that exception. This is the "whole-batch failure" case.

The batcher also supports **per-item failure** at the infrastructure level. When the internal batch inference function returns an `Exception` instance at a specific index in the results list, only that item's request is rejected — other items in the batch still receive their results normally.

If the results list has a different length than the input batch, every request receives a `RuntimeError` explaining the mismatch.

## Disabling batching

Set `enable_batching=False` to process every request individually:

```python
app = BlazeApp(enable_batching=False)
```

This is appropriate when:

- Your model does not benefit from batched inference (e.g., it processes one item at a time internally).
- You want the simplest possible request path for debugging.

## Automatic exclusions

Even when `enable_batching=True`, BlazeRPC automatically skips batching for certain models:

- **Streaming models**: Server-streaming handlers (`streaming=True`) are always called individually. The batcher only handles unary RPCs.
- **Models using dependency injection**: Handlers that use `Context` or `Depends` parameters are excluded from batching. Each request is processed individually so that per-request context and dependencies are correctly resolved. A warning is logged at startup for each excluded model.

If you need both batching and shared resources, access them directly in the handler body (e.g., via a module-level variable) rather than through `Depends`. See the [dependency injection guide](dependency-injection.md#limitations) for details.

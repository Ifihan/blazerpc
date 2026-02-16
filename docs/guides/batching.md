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

The batcher runs as a background `asyncio.Task` with the following loop:

1. **Wait** for the first request to arrive.
2. **Collect** additional requests until either `max_batch_size` is reached or `batch_timeout_ms` elapses.
3. **Dispatch** the batch to the model function.
4. **Distribute** results back to each client's individual future.

This means:

- Under high load, batches fill up quickly and are dispatched at full capacity.
- Under light load, the timeout ensures that a lone request is not stuck waiting for a batch to fill. A 10 ms timeout adds negligible latency.

## Example

```python
import numpy as np
from blazerpc import BlazeApp

app = BlazeApp(
    enable_batching=True,
    max_batch_size=16,
    batch_timeout_ms=5.0,
)

@app.model("classify")
def classify_image(image: list[float]) -> float:
    # In production, this would be a batch-aware model call.
    return float(np.mean(image))
```

When three clients call `classify_image` within 5 ms of each other, the batcher groups all three requests into a single batch. The model function runs once, and each client receives only its own result.

## Tuning

**`max_batch_size`** controls the upper bound on batch size. Set this based on your GPU memory and model throughput characteristics. Larger batches improve throughput but use more memory.

**`batch_timeout_ms`** controls latency under light load. Lower values reduce tail latency for individual requests. Higher values give the batcher more time to collect a full batch, improving throughput.

A good starting point:

- For latency-sensitive applications (real-time APIs): `batch_timeout_ms=5.0`, `max_batch_size=8`.
- For throughput-optimized workloads (offline processing): `batch_timeout_ms=50.0`, `max_batch_size=64`.

## Partial failure handling

If the model function raises an exception, every request in the batch receives that exception. This is the "whole-batch failure" case.

BlazeRPC also supports **per-item failure**. If the model function returns an `Exception` instance at a specific index in the results list, only that item's request is rejected. Other items in the batch still receive their results:

```python
@app.model("classify")
def classify_batch(inputs: list[dict]) -> list:
    results = []
    for inp in inputs:
        try:
            results.append(model.predict(inp))
        except Exception as e:
            results.append(e)  # This item fails; others succeed.
    return results
```

If the results list has a different length than the input batch, every request receives a `RuntimeError` explaining the mismatch.

## Disabling batching

Set `enable_batching=False` to process every request individually:

```python
app = BlazeApp(enable_batching=False)
```

This is appropriate when:

- Your model does not benefit from batched inference (e.g., it processes one item at a time internally).
- You are using streaming endpoints (streaming is always unbatched).
- You want the simplest possible request path for debugging.

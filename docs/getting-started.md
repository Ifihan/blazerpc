# Getting started

This guide walks you through installing BlazeRPC, training a simple model, serving it over gRPC, and calling it from a Python client.

## Installation

```bash
uv add blazerpc
```

If you use a specific ML framework, install the corresponding extra:

```bash
uv add "blazerpc[pytorch]"      # PyTorch tensor conversion helpers
uv add "blazerpc[tensorflow]"   # TensorFlow tensor conversion helpers
uv add "blazerpc[onnx]"         # ONNX Runtime model wrapper
uv add "blazerpc[all]"          # All optional integrations
```

## Define a model

This example trains a scikit-learn Logistic Regression classifier on the [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) and serves it over gRPC. The Iris dataset ships with scikit-learn, so there are no downloads or GPUs required.

Create a file called `app.py`:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from blazerpc import BlazeApp, TensorInput, TensorOutput

# Train a simple model (in production, load a pre-trained model from disk)
iris = load_iris()
clf = LogisticRegression(max_iter=200)
clf.fit(iris.data, iris.target)

app = BlazeApp()

@app.model("iris")
def predict_iris(
    features: TensorInput[np.float32, "batch", 4],
) -> TensorOutput[np.float32, "batch", 3]:
    """Classify iris flowers. Returns class probabilities."""
    probs = clf.predict_proba(features).astype(np.float32)
    return probs
```

BlazeRPC reads the type annotations on your function to generate the gRPC request and response messages. `TensorInput` and `TensorOutput` declare the expected dtype and shape, and BlazeRPC serializes them as `TensorProto` messages on the wire.

Supported types include `str`, `int`, `float`, `bool`, `list[float]`, `list[str]`, and tensor types via `TensorInput` / `TensorOutput`.

## Start the server

```bash
blaze serve app:app
```

The import string follows the `module:attribute` convention. BlazeRPC imports the module, looks up the attribute, and starts the gRPC server.

```
⚡ BlazeRPC server starting...
  ✓ Loaded model: iris v1
  ✓ Server listening on 0.0.0.0:50051
```

The server registers three services automatically:

| Service                                    | Purpose                |
| ------------------------------------------ | ---------------------- |
| `blazerpc.InferenceService`                | Your model RPCs        |
| `grpc.health.v1.Health`                    | Standard health checks |
| `grpc.reflection.v1alpha.ServerReflection` | Service discovery      |

## Call the model from Python

Create a file called `client.py`:

```python
import asyncio
import numpy as np
from blazerpc import BlazeClient
from app import app

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]

async def main():
    async with BlazeClient("127.0.0.1", 50051, registry=app.registry) as client:
        samples = np.array(
            [[5.1, 3.5, 1.4, 0.2],   # typical setosa
             [6.7, 3.0, 5.2, 2.3]],  # typical virginica
            dtype=np.float32,
        )
        probs = await client.predict("iris", features=samples)

        for i, sample in enumerate(samples):
            predicted = IRIS_CLASSES[np.argmax(probs[i])]
            print(f"sample {i+1} → {predicted} (probs={probs[i]})")

asyncio.run(main())
```

`BlazeClient` requires a `registry` parameter so it can build the correct Protobuf message types for each model. Pass `app.registry` from your server application.

## Export the `.proto` file

```bash
blaze proto app:app --output-dir ./proto_out
```

This writes a `blaze_service.proto` file that you can compile with `protoc` or share with clients in any language. The generated proto looks like this:

```protobuf
syntax = "proto3";
package blazerpc;

message TensorProto {
  repeated int64 shape = 1;
  string dtype = 2;
  bytes data = 3;
}

message IrisRequest {
  TensorProto features = 1;
}

message IrisResponse {
  TensorProto result = 1;
}

service InferenceService {
  rpc PredictIris(IrisRequest) returns (IrisResponse);
}
```

Because BlazeRPC uses standard Protobuf encoding on the wire, this proto file works with any gRPC client -- Postman, grpcurl, or generated stubs in Go, Java, Rust, etc.

## Multiple models

Register as many models as you need on the same app. Each model becomes its own RPC method:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

# Iris classifier
@app.model("iris")
def predict_iris(
    features: TensorInput[np.float32, "batch", 4],
) -> TensorOutput[np.float32, "batch", 3]:
    return iris_clf.predict_proba(features).astype(np.float32)

# Linear regression
@app.model("housing")
def predict_housing(
    features: TensorInput[np.float32, "batch", 3],
) -> TensorOutput[np.float32, "batch", 1]:
    return reg.predict(features).astype(np.float32).reshape(-1, 1)

# Simple echo for health checks
@app.model("echo")
def echo(text: str) -> str:
    return f"You said: {text}"
```

All three models are served under the same `InferenceService` and discovered through a single reflection endpoint.

## Next steps

- [Dependency injection](guides/dependency-injection.md) -- Access gRPC metadata, share resources, and inject dependencies FastAPI-style.
- [Streaming](guides/streaming.md) -- Return tokens incrementally for LLM workloads.
- [Adaptive batching](guides/batching.md) -- Group requests into batches for GPU efficiency.
- [Framework integrations](guides/integrations.md) -- Use PyTorch, TensorFlow, or ONNX Runtime with automatic tensor conversion.
- [Configuration](configuration.md) -- Tune batch size, timeouts, and server options.

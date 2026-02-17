# API reference

This page documents every public class, function, and exception in BlazeRPC.

---

## `blazerpc.BlazeApp`

The main entry point. Creates an application, registers models, and starts the server.

```python
from blazerpc import BlazeApp

app = BlazeApp(
    name="my-service",
    enable_batching=True,
    max_batch_size=32,
    batch_timeout_ms=10.0,
)
```

### Constructor

| Parameter          | Type    | Default      | Description                                          |
| ------------------ | ------- | ------------ | ---------------------------------------------------- |
| `name`             | `str`   | `"blazerpc"` | Application name (used in logging and diagnostics).  |
| `enable_batching`  | `bool`  | `True`       | Enable adaptive request batching.                    |
| `max_batch_size`   | `int`   | `32`         | Maximum number of requests in a single batch.        |
| `batch_timeout_ms` | `float` | `10.0`       | Maximum time (ms) to wait before dispatching a partial batch. |

### `app.model(name, version="1", streaming=False)`

Decorator that registers a function as a model endpoint.

| Parameter   | Type   | Default  | Description                                    |
| ----------- | ------ | -------- | ---------------------------------------------- |
| `name`      | `str`  | required | Model name. Becomes part of the RPC method name (`PredictName`). |
| `version`   | `str`  | `"1"`    | Model version string.                          |
| `streaming` | `bool` | `False`  | If `True`, the function must be an async generator that yields responses. |

```python
@app.model("sentiment", version="2")
def predict_sentiment(text: list[str]) -> list[float]:
    return model.predict(text)
```

### `await app.serve(host="0.0.0.0", port=50051)`

Start the gRPC server and block until a shutdown signal is received. Registers the inference servicer, health service, and reflection handlers automatically.

| Parameter | Type  | Default       | Description       |
| --------- | ----- | ------------- | ----------------- |
| `host`    | `str` | `"0.0.0.0"`  | Bind address.     |
| `port`    | `int` | `50051`       | Listen port.      |

### `app.registry`

The underlying `ModelRegistry` instance. Useful for introspection:

```python
for model in app.registry.list_models():
    print(f"{model.name} v{model.version}")
```

---

## `blazerpc.TensorInput`

Type annotation for tensor-typed model inputs. Used by the codegen layer to emit `TensorProto` fields.

```python
from blazerpc import TensorInput
import numpy as np

def classify(image: TensorInput[np.float32, "batch", 224, 224, 3]) -> ...:
    ...
```

The subscript arguments are `dtype` followed by shape dimensions. Shape dimensions can be integers or strings (symbolic names like `"batch"`).

## `blazerpc.TensorOutput`

Type annotation for tensor-typed model outputs. Same subscript syntax as `TensorInput`.

```python
from blazerpc import TensorOutput
import numpy as np

def classify(...) -> TensorOutput[np.float32, "batch", 1000]:
    ...
```

---

## Exceptions

All exceptions inherit from `BlazeRPCError`.

### `BlazeRPCError`

Base exception for all BlazeRPC errors.

### `ValidationError`

Raised when input validation fails (bad shapes, types, missing annotations).

| Attribute | Type           | Description              |
| --------- | -------------- | ------------------------ |
| `field`   | `str \| None`  | The field that failed validation. |

### `ModelNotFoundError`

Raised when a requested model is not found in the registry.

| Attribute | Type  | Description       |
| --------- | ----- | ----------------- |
| `name`    | `str` | Model name.       |
| `version` | `str` | Model version.    |

### `SerializationError`

Raised when tensor serialization or deserialization fails.

| Attribute | Type           | Description            |
| --------- | -------------- | ---------------------- |
| `dtype`   | `str \| None`  | The dtype that caused the error. |

### `InferenceError`

Raised when model inference fails.

| Attribute    | Type           | Description       |
| ------------ | -------------- | ----------------- |
| `model_name` | `str \| None`  | The model that failed. |

### `ConfigurationError`

Raised for invalid configuration (bad import paths, missing settings).

---

## Serialization

### `blazerpc.runtime.serialization.TensorProto`

Wire representation of a tensor. A dataclass with `__slots__`.

| Field   | Type              | Description            |
| ------- | ----------------- | ---------------------- |
| `shape` | `tuple[int, ...]` | Tensor shape.          |
| `dtype` | `str`             | Proto type string (e.g. `"float"`, `"int64"`). |
| `data`  | `bytes`           | Raw tensor bytes.      |

### `serialize_tensor(arr: np.ndarray) -> TensorProto`

Serialize a NumPy array to a `TensorProto`. Raises `SerializationError` if the dtype is unsupported.

### `deserialize_tensor(proto: TensorProto) -> np.ndarray`

Deserialize a `TensorProto` back to a NumPy array. Uses `np.frombuffer()` for zero-copy reconstruction.

---

## Server

### `blazerpc.server.grpc.GRPCServer`

Production-ready async gRPC server. Wraps `grpclib.server.Server` with signal handling and graceful shutdown.

```python
server = GRPCServer(handlers, grace_period=5.0)
await server.start(host="0.0.0.0", port=50051)
```

| Constructor parameter | Type    | Default | Description                                |
| --------------------- | ------- | ------- | ------------------------------------------ |
| `handlers`            | `Sequence[Any]` | required | List of grpclib-compatible handlers. |
| `grace_period`        | `float` | `5.0`   | Seconds to wait for in-flight requests during shutdown. |

### `build_health_service(servicers=None) -> Health`

Create a gRPC health service. Pass servicer instances for per-service health tracking, or `None` for unconditional `SERVING` status.

### `build_reflection_service(handlers=None) -> list`

Create gRPC reflection handlers. Pass gRPC service handler objects (e.g. the servicer from `build_servicer()`) so clients can discover available RPCs.

---

## Middleware

### `blazerpc.server.middleware.Middleware`

Abstract base class for server middleware. Subclass it and implement `on_request()` and `on_response()`.

```python
class MyMiddleware(Middleware):
    async def on_request(self, event: RecvRequest) -> None:
        ...

    async def on_response(self, event: SendTrailingMetadata) -> None:
        ...
```

Call `middleware.attach(server)` to register it on a `grpclib.server.Server` instance.

### `LoggingMiddleware`

Logs every RPC call with method name, peer address, and response status.

### `MetricsMiddleware`

Exports Prometheus metrics:

- `blazerpc_requests_total{method, status}` -- Counter of total requests.
- `blazerpc_request_duration_seconds{method}` -- Histogram of request durations.

### `ExceptionMiddleware`

Base class for custom exception-to-gRPC-status mapping. A no-op by default.

---

## Runtime

### `blazerpc.runtime.batcher.Batcher`

Adaptive request batcher. Collects individual requests into batches.

| Constructor parameter | Type    | Default | Description                         |
| --------------------- | ------- | ------- | ----------------------------------- |
| `max_size`            | `int`   | `32`    | Maximum items per batch.            |
| `timeout_ms`          | `float` | `10.0`  | Max wait time (ms) before dispatching. |

Key methods:

- `await submit(request)` -- Submit a request and wait for the batched result.
- `await start(inference_fn)` -- Start the background batching loop.
- `await stop()` -- Stop the batching loop.

### `blazerpc.runtime.executor.ModelExecutor`

Wraps a registered model function with sync/async bridging.

- `await execute(kwargs)` -- Run inference with keyword arguments.
- `await execute_batch(kwargs_list)` -- Run inference on a batch of inputs.

Synchronous model functions are offloaded to a thread pool via `asyncio.to_thread()`.

---

## Code generation

### `blazerpc.codegen.proto.ProtoGenerator`

Generates `.proto` file content from a `ModelRegistry`.

```python
from blazerpc.codegen.proto import ProtoGenerator

proto_content = ProtoGenerator().generate(app.registry)
```

### `blazerpc.codegen.servicer.build_servicer(registry, batcher=None)`

Builds a grpclib-compatible `InferenceServicer` from a `ModelRegistry`.

```python
from blazerpc.codegen.servicer import build_servicer

servicer = build_servicer(app.registry, batcher=app.batcher)
```

---

## Contrib

### `blazerpc.contrib.pytorch`

| Function / Decorator                          | Description                                       |
| --------------------------------------------- | ------------------------------------------------- |
| `torch_to_numpy(tensor) -> np.ndarray`        | Detach, move to CPU, and convert to NumPy.        |
| `numpy_to_torch(arr, device, dtype) -> Tensor` | Convert NumPy array to a PyTorch tensor.          |
| `@torch_model(device="cpu")`                  | Decorator that auto-converts inputs and outputs.  |

### `blazerpc.contrib.tensorflow`

| Function / Decorator                       | Description                                       |
| ------------------------------------------ | ------------------------------------------------- |
| `tf_to_numpy(tensor) -> np.ndarray`        | Convert a TensorFlow tensor to NumPy.             |
| `numpy_to_tf(arr, dtype) -> tf.Tensor`     | Convert NumPy array to a TensorFlow tensor.       |
| `@tf_model(dtype=None)`                    | Decorator that auto-converts inputs and outputs.  |

### `blazerpc.contrib.onnx.ONNXModel`

Wrapper around an ONNX Runtime inference session.

```python
from blazerpc.contrib.onnx import ONNXModel

model = ONNXModel("model.onnx", providers=["CUDAExecutionProvider"])
results = model.predict(input_array)
```

| Constructor parameter | Type             | Default                      | Description              |
| --------------------- | ---------------- | ---------------------------- | ------------------------ |
| `model_path`          | `str \| Path`    | required                     | Path to the `.onnx` file. |
| `providers`           | `list[str] \| None` | `["CPUExecutionProvider"]` | ONNX Runtime execution providers. |
| `session_options`     | `Any`            | `None`                       | Optional `ort.SessionOptions`. |

Methods:

- `predict(*inputs) -> list[np.ndarray]` -- Run inference with positional inputs matched to input names.
- `predict_dict(inputs) -> dict[str, np.ndarray]` -- Run inference with named inputs, returning named outputs.
- `input_names -> list[str]` -- Names of the model's input tensors.
- `output_names -> list[str]` -- Names of the model's output tensors.

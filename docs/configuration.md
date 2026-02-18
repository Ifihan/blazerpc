# Configuration

This page lists every configurable parameter in BlazeRPC, organized by where it is set.

## `BlazeApp` constructor

These parameters are passed when you create the application instance.

```python
from blazerpc import BlazeApp

app = BlazeApp(
    name="my-service",
    enable_batching=True,
    max_batch_size=32,
    batch_timeout_ms=10.0,
)
```

| Parameter          | Type    | Default      | Description                                                          |
| ------------------ | ------- | ------------ | -------------------------------------------------------------------- |
| `name`             | `str`   | `"blazerpc"` | Application name. Used in logging output and diagnostics.            |
| `enable_batching`  | `bool`  | `True`       | Enable adaptive request batching. Set to `False` for unbatched processing. |
| `max_batch_size`   | `int`   | `32`         | Maximum number of requests collected into a single batch.            |
| `batch_timeout_ms` | `float` | `10.0`       | Maximum time (in milliseconds) to wait for a full batch before dispatching a partial one. Lower values reduce latency; higher values improve throughput. |

## `@app.model()` decorator

These parameters are passed per model.

```python
@app.model("sentiment", version="2", streaming=False)
def predict_sentiment(text: list[str]) -> list[float]:
    ...
```

| Parameter   | Type   | Default  | Description                                                          |
| ----------- | ------ | -------- | -------------------------------------------------------------------- |
| `name`      | `str`  | required | Model name. Converted to PascalCase for the RPC method name (e.g. `"sentiment"` becomes `PredictSentiment`). |
| `version`   | `str`  | `"1"`    | Model version string. Stored as metadata; does not affect routing.   |
| `streaming` | `bool` | `False`  | If `True`, the function must be an async generator that yields responses. Produces a `returns (stream Response)` RPC. |

## `GRPCServer` constructor

These parameters control the underlying gRPC server behavior. They are set internally by `BlazeApp.serve()` but can be used directly if you instantiate `GRPCServer` yourself.

```python
from blazerpc.server.grpc import GRPCServer

server = GRPCServer(handlers, grace_period=5.0)
await server.start(host="0.0.0.0", port=50051)
```

| Parameter      | Type             | Default | Description                                                     |
| -------------- | ---------------- | ------- | --------------------------------------------------------------- |
| `handlers`     | `Sequence[Any]`  | required | List of grpclib-compatible handler objects (servicers).          |
| `grace_period` | `float`          | `5.0`   | Seconds to wait for in-flight requests to complete during shutdown. After this period, the server shuts down forcefully. |

### `server.start()` parameters

| Parameter | Type  | Default      | Description       |
| --------- | ----- | ------------ | ----------------- |
| `host`    | `str` | `"0.0.0.0"`  | Bind address.     |
| `port`    | `int` | `50051`       | Listen port.      |

## CLI options

### `blaze serve`

```bash
blaze serve <app_path> [OPTIONS]
```

| Argument / Option | Type    | Default      | Description                                          |
| ----------------- | ------- | ------------ | ---------------------------------------------------- |
| `app_path`        | `str`   | required     | App import path in `module:attribute` format (e.g. `app:app`). |
| `--host`          | `str`   | `"0.0.0.0"`  | Host to bind to.                                     |
| `--port`          | `int`   | `50051`       | Port to listen on.                                   |
| `--workers`       | `int`   | `1`          | Number of worker processes.                          |
| `--reload`        | `bool`  | `False`       | Enable auto-reload for development. Requires `watchfiles`. |

#### Hot reload

When `--reload` is enabled, the server watches for `.py` file changes in the current directory and automatically restarts when changes are detected. This uses process-level restart (like uvicorn) for a clean reimport of all modules.

```bash
blaze serve app:app --reload
```

Install the reload dependency:

```bash
pip install blazerpc[reload]
# or
pip install watchfiles
```

The reload feature is intended for **development only** — do not use it in production.

### `blaze proto`

```bash
blaze proto <app_path> [OPTIONS]
```

| Argument / Option | Type  | Default  | Description                                          |
| ----------------- | ----- | -------- | ---------------------------------------------------- |
| `app_path`        | `str` | required | App import path in `module:attribute` format.        |
| `--output-dir`    | `str` | `"."`    | Output directory for the generated `.proto` file.    |

## Logging

BlazeRPC uses Python's standard `logging` module. The `blaze serve` command configures `INFO`-level logging by default. Logger names follow the module hierarchy:

| Logger name             | Source                       |
| ----------------------- | ---------------------------- |
| `blazerpc.server`       | Server lifecycle events.     |
| `blazerpc.batcher`      | Batch collection and dispatch. |
| `blazerpc.middleware`   | Middleware event hooks.      |
| `blazerpc.reflection`   | Reflection service setup.    |

To customize logging, configure these loggers before calling `blaze serve` or `app.serve()`:

```python
import logging

logging.getLogger("blazerpc.batcher").setLevel(logging.DEBUG)
```

## Prometheus metrics

When `MetricsMiddleware` is attached, the following metrics are exported:

| Metric name                          | Type      | Labels             | Description              |
| ------------------------------------ | --------- | ------------------ | ------------------------ |
| `blazerpc_requests_total`            | Counter   | `method`, `status` | Total number of gRPC requests. |
| `blazerpc_request_duration_seconds`  | Histogram | `method`           | Request duration in seconds. |

These are standard `prometheus_client` objects. Expose them via a Prometheus scrape endpoint (e.g., using `prometheus_client.start_http_server()`).

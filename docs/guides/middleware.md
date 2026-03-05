# Middleware

BlazeRPC provides a middleware system built on grpclib's event hooks. Middleware runs outside the handler function and observes request/response events without modifying the request data itself.

## Configuring middleware

Pass middleware instances to `BlazeApp` via the `middleware` parameter. They are automatically attached to the gRPC server on startup:

```python
from blazerpc import BlazeApp
from blazerpc.server.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    OTelMetricsMiddleware,
)

app = BlazeApp(
    name="iris-service",
    middleware=[
        LoggingMiddleware(),
        MetricsMiddleware(),        # Prometheus pull
        OTelMetricsMiddleware(),    # OpenTelemetry push
    ],
)
```

Middleware is applied in the order you provide it.

## Built-in middleware

BlazeRPC ships with four middleware classes:

### LoggingMiddleware

Logs every RPC call with the method name, peer address, and response status.

```python
from blazerpc.server.middleware import LoggingMiddleware

middleware = LoggingMiddleware()
middleware.attach(grpclib_server)
```

You can pass a custom `logging.Logger` instance:

```python
import logging

logger = logging.getLogger("my_app.rpc")
middleware = LoggingMiddleware(logger=logger)
```

### MetricsMiddleware

Exports Prometheus metrics for every RPC call:

| Metric                                 | Type      | Labels            | Description              |
| -------------------------------------- | --------- | ----------------- | ------------------------ |
| `blazerpc_requests_total`              | Counter   | `method`, `status` | Total number of requests. |
| `blazerpc_request_duration_seconds`    | Histogram | `method`          | Request duration.        |

```python
from blazerpc.server.middleware import MetricsMiddleware

middleware = MetricsMiddleware()
middleware.attach(grpclib_server)
```

The metrics are registered as module-level Prometheus objects and can be scraped by any Prometheus-compatible collector.

### OTelMetricsMiddleware

Pushes RPC metrics via the OpenTelemetry Metrics API:

| Instrument              | Type      | Attributes         | Description              |
| ----------------------- | --------- | ------------------ | ------------------------ |
| `blazerpc.rpc.count`    | Counter   | `method`, `status` | Total number of requests. |
| `blazerpc.rpc.duration` | Histogram | `method`           | Request duration (seconds). |

```python
from blazerpc.server.middleware import OTelMetricsMiddleware

app = BlazeApp(middleware=[OTelMetricsMiddleware()])
```

By default, `OTelMetricsMiddleware` uses the global meter provider. To push metrics to a specific backend, pass a custom `Meter`:

```python
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from blazerpc.server.middleware import OTelMetricsMiddleware

reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint="http://otel-collector:4317")
)
provider = MeterProvider(metric_readers=[reader])
meter = provider.get_meter("blazerpc")

app = BlazeApp(middleware=[OTelMetricsMiddleware(meter=meter)])
```

Install the OTel dependencies:

```bash
uv add "blazerpc[otel]"
```

### ExceptionMiddleware

A base class for custom exception-to-gRPC-status mapping. The default implementation is a no-op -- subclass it to add your own mapping logic.

## Writing custom middleware

Subclass `Middleware` and implement `on_request()` and `on_response()`:

```python
from grpclib.const import Status
from grpclib.events import RecvRequest, SendTrailingMetadata
from grpclib.exceptions import GRPCError

from blazerpc.server.middleware import Middleware

class AuthMiddleware(Middleware):
    """Reject requests without a valid authorization token."""

    async def on_request(self, event: RecvRequest) -> None:
        metadata = dict(event.metadata)
        token = metadata.get("authorization")
        if not token or not self._validate_token(token):
            raise GRPCError(
                Status.UNAUTHENTICATED,
                "Missing or invalid authorization token",
            )

    async def on_response(self, event: SendTrailingMetadata) -> None:
        pass  # No response-side logic needed.

    def _validate_token(self, token: str) -> bool:
        return token == "my-secret-token"
```

Attach it to the server:

```python
auth = AuthMiddleware()
auth.attach(grpclib_server)
```

## How it works

Middleware hooks into grpclib's event system using the `listen()` function. When you call `middleware.attach(server)`, two event listeners are registered:

- `RecvRequest` -- fires when a request is received, before the handler runs.
- `SendTrailingMetadata` -- fires when the response is about to be sent, after the handler completes.

This design means middleware cannot modify the request payload, but it can:

- Inspect request metadata (headers, peer address, method name).
- Reject requests by raising `GRPCError`.
- Record timing and status information for observability.
- Add trailing metadata to responses.

## Attaching multiple middleware

Middleware is applied in the order you provide it. Each middleware's `on_request()` runs before the handler, and each `on_response()` runs after:

```python
app = BlazeApp(
    middleware=[
        LoggingMiddleware(),
        MetricsMiddleware(),
        AuthMiddleware(),
    ],
)
```

In this example, logging fires first, then metrics, then auth. If auth rejects the request, the handler never runs, but the logging and metrics middleware still see the response event with the `UNAUTHENTICATED` status.

You can also attach middleware manually to a `grpclib.server.Server` instance using `middleware.attach(server)` if you are not using `BlazeApp`.

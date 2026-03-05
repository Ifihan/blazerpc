# Dependency injection

BlazeRPC provides a FastAPI-style dependency injection system that lets your model handlers access gRPC metadata, shared application state, and reusable dependencies — without coupling to the transport layer.

## Overview

Three building blocks:

| API | Purpose |
| --- | ------- |
| `app.state` | Attach shared resources (models, DB pools, config) at startup. |
| `Context` | Per-request object with gRPC metadata, peer info, method path, and `app_state`. |
| `Depends(fn)` | Mark a parameter as an injected dependency resolved at request time. |

## `app.state` — shared application state

`app.state` is a namespace you can attach anything to. It persists for the lifetime of the application and is accessible from dependency functions via `ctx.app_state`.

```python
from blazerpc import BlazeApp

app = BlazeApp()

# Attach resources at startup
app.state.classifier = load_my_model()
app.state.db_pool = create_pool()
```

## `Context` — per-request information

Add a `Context`-typed parameter to any handler to receive a per-request context object:

```python
from blazerpc import BlazeApp, Context

app = BlazeApp()

@app.model("info")
def info(text: str, ctx: Context) -> str:
    return f"method={ctx.method}, peer={ctx.peer}"
```

### Context attributes

| Attribute | Type | Description |
| --------- | ---- | ----------- |
| `metadata` | `MultiDict \| None` | gRPC invocation metadata (headers) sent by the client. |
| `peer` | `Any` | Connection peer info (address, certificate). |
| `method` | `str` | Full gRPC method path, e.g. `"/blazerpc.InferenceService/PredictIris"`. |
| `app_state` | `AppState` | Reference to `app.state`. |

`Context` parameters are **not** included in the Protobuf request message — they are injected by the framework at request time.

## `Depends()` — reusable dependencies

Use `Depends(fn)` as a parameter default to inject a value computed from the request context. The dependency function receives the `Context` and returns the value:

```python
from blazerpc import BlazeApp, Context, Depends
from sklearn.linear_model import LogisticRegression

app = BlazeApp()
app.state.classifier = train_model()

def get_classifier(ctx: Context) -> LogisticRegression:
    return ctx.app_state.classifier

@app.model("predict")
def predict(
    features: list[float],
    model: LogisticRegression = Depends(get_classifier),
) -> list[float]:
    return model.predict_proba([features])[0].tolist()
```

### Async dependencies

Dependency functions can be async:

```python
async def get_db_connection(ctx: Context):
    return await ctx.app_state.db_pool.acquire()

@app.model("lookup")
async def lookup(
    user_id: int,
    db = Depends(get_db_connection),
) -> str:
    row = await db.fetchone("SELECT name FROM users WHERE id = ?", user_id)
    return row["name"]
```

### Multiple dependencies

Handlers can use multiple dependencies and combine them with `Context`:

```python
def get_model(ctx: Context):
    return ctx.app_state.classifier

def get_class_names(ctx: Context):
    return ctx.app_state.class_names

@app.model("label")
def label(
    features: list[float],
    ctx: Context,
    model = Depends(get_model),
    names: list[str] = Depends(get_class_names),
) -> str:
    idx = int(model.predict([features])[0])
    return f"{names[idx]} (via {ctx.method})"
```

## Auth pattern

A common use case is extracting authentication tokens from gRPC metadata:

```python
from blazerpc import BlazeApp, Context, Depends
from blazerpc.exceptions import ValidationError

app = BlazeApp()

def require_auth(ctx: Context) -> str:
    """Extract and validate the auth token, or raise."""
    if not ctx.metadata:
        raise ValidationError("missing metadata")
    token = dict(ctx.metadata).get("authorization")
    if not token:
        raise ValidationError("missing authorization header")
    return token

@app.model("secure")
def secure_predict(
    text: str,
    token: str = Depends(require_auth),
) -> str:
    return f"Authenticated as {token}: {text}"
```

## Limitations

- **Batched handlers**: When batching is enabled, the batcher coalesces multiple requests. Dependencies are resolved per-request before batching, but only request-field kwargs are passed through the batcher. If your handler uses both `Depends` and batching, the dependency values come from the individual request context, not the batch.
- **No nested dependencies**: `Depends` functions receive `Context` only — they cannot declare their own `Depends` parameters. Keep dependency functions simple.

## Full example

See the complete working example in [`examples/dependency_injection/`](https://github.com/blazerpc/blazerpc/tree/main/examples/dependency_injection):

- `app.py` — Server with `app.state`, `Context`, and `Depends`
- `client.py` — Client that calls the DI-powered handlers

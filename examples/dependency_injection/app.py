"""
Dependency injection example.

Demonstrates BlazeRPC's FastAPI-style dependency injection system:
  - ``app.state`` for sharing resources across handlers
  - ``Context`` for accessing gRPC metadata, peer info, and method path
  - ``Depends(fn)`` for reusable dependency functions

Prerequisites:
    uv add scikit-learn

Run the server:
    uv run blaze serve examples.dependency_injection.app:app
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from blazerpc import BlazeApp, Context, Depends, TensorInput, TensorOutput

# ---------------------------------------------------------------------------
# Train a simple model and attach it to app.state
# ---------------------------------------------------------------------------

app = BlazeApp(name="di-demo", enable_batching=False)

iris = load_iris()
clf = LogisticRegression(max_iter=200)
clf.fit(iris.data, iris.target)

# Store the trained model on app.state so handlers can access it via Depends
app.state.classifier = clf
app.state.class_names = ["setosa", "versicolor", "virginica"]


# ---------------------------------------------------------------------------
# Dependency functions
# ---------------------------------------------------------------------------


def get_classifier(ctx: Context) -> LogisticRegression:
    """Retrieve the classifier from app state."""
    return ctx.app_state.classifier


def get_class_names(ctx: Context) -> list[str]:
    """Retrieve class names from app state."""
    return ctx.app_state.class_names


def get_auth_token(ctx: Context) -> str | None:
    """Extract an authorization token from gRPC metadata."""
    if ctx.metadata:
        return dict(ctx.metadata).get("authorization")
    return None


# ---------------------------------------------------------------------------
# Model handlers using dependency injection
# ---------------------------------------------------------------------------


@app.model("classify")
def classify(
    features: TensorInput[np.float32, 4],
    model: LogisticRegression = Depends(get_classifier),
) -> TensorOutput[np.float32, 3]:
    """Classify iris features using an injected model."""
    features_2d = features.reshape(1, -1) if features.ndim == 1 else features
    return model.predict_proba(features_2d).astype(np.float32)


@app.model("whoami")
def whoami(
    ctx: Context,
    token: str | None = Depends(get_auth_token),
) -> str:
    """Return info about the caller using Context and Depends."""
    return f"method={ctx.method} peer={ctx.peer} auth={'yes' if token else 'no'}"


@app.model("label")
def label(
    features: TensorInput[np.float32, 4],
    model: LogisticRegression = Depends(get_classifier),
    names: list[str] = Depends(get_class_names),
) -> str:
    """Classify and return the human-readable label."""
    features_2d = features.reshape(1, -1) if features.ndim == 1 else features
    idx = int(model.predict(features_2d)[0])
    return names[idx]

"""
Multi-model service.

This example registers two scikit-learn models on a single BlazeRPC
server: an Iris classifier and a simple linear regression model. Each
model becomes its own RPC method under ``blazerpc.InferenceService``,
so clients can call whichever model they need without running separate
servers.

Generated RPCs:
    - PredictIris(IrisRequest)           -> IrisResponse
    - PredictHousing(HousingRequest)     -> HousingResponse
    - PredictEcho(EchoRequest)           -> EchoResponse

Prerequisites:
    pip install scikit-learn

Run the server:
    uv run blaze serve examples.multi_model.app:app

Export all models' .proto:
    uv run blaze proto examples.multi_model.app:app --output-dir ./proto_out
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression

from blazerpc import BlazeApp, TensorInput, TensorOutput

app = BlazeApp(name="multi-model-demo", enable_batching=False)


# ---- Model 1: Iris classifier ----

iris = load_iris()
iris_clf = LogisticRegression(max_iter=200)
iris_clf.fit(iris.data, iris.target)


@app.model("iris")
def predict_iris(
    features: TensorInput[np.float32, "batch", 4],
) -> TensorOutput[np.float32, "batch", 3]:
    """Classify iris flowers. Returns class probabilities."""
    probs = iris_clf.predict_proba(features).astype(np.float32)
    return probs


# ---- Model 2: Simple linear regression ----
# Predicts a target value from 3 features (synthetic training data).

rng = np.random.default_rng(42)
X_train = rng.standard_normal((100, 3)).astype(np.float32)
y_train = X_train @ np.array([1.5, -2.0, 0.5]) + 0.1 * rng.standard_normal(100)
reg = LinearRegression()
reg.fit(X_train, y_train)


@app.model("housing")
def predict_housing(
    features: TensorInput[np.float32, "batch", 3],
) -> TensorOutput[np.float32, "batch", 1]:
    """Predict a value from 3 input features using linear regression."""
    preds = reg.predict(features).astype(np.float32).reshape(-1, 1)
    return preds


# ---- Model 3: Echo (for connectivity checks) ----


@app.model("echo")
def echo(text: str) -> str:
    """Echo the input back."""
    return f"You said: {text}"

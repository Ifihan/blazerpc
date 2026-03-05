"""
Iris classifier service.

This example trains a scikit-learn Logistic Regression model on the Iris
dataset and serves it over gRPC using BlazeRPC. The model accepts four
float features (sepal length, sepal width, petal length, petal width) and
returns class probabilities for the three Iris species.

Prerequisites:
    pip install scikit-learn

Run the server:
    uv run blaze serve examples.simple.app:app

Export the .proto file:
    uv run blaze proto examples.simple.app:app --output-dir ./proto_out
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from blazerpc import BlazeApp, TensorInput, TensorOutput

# Train a simple model (in production, load a pre-trained model from disk)
iris = load_iris()
clf = LogisticRegression(max_iter=200)
clf.fit(iris.data, iris.target)

app = BlazeApp(name="iris-demo", enable_batching=False)


@app.model("iris")
def predict_iris(
    features: TensorInput[np.float32, "batch", 4],
) -> TensorOutput[np.float32, "batch", 3]:
    """Classify iris flowers. Returns class probabilities.

    Input: a batch of samples, each with 4 features
        [sepal_length, sepal_width, petal_length, petal_width]

    Output: a batch of probability vectors over 3 classes
        [setosa, versicolor, virginica]
    """
    probs = clf.predict_proba(features).astype(np.float32)
    return probs


@app.model("echo")
def echo(text: str) -> str:
    """Echo the input back. Useful for connectivity checks."""
    return f"You said: {text}"

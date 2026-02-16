# Framework integrations

BlazeRPC provides optional helpers for PyTorch, TensorFlow, and ONNX Runtime. These handle the conversion between NumPy arrays (BlazeRPC's wire format) and framework-specific tensor types so you can write model code in the framework's native API.

## PyTorch

Install the extra:

```bash
pip install blazerpc[pytorch]
```

### `@torch_model` decorator

The `@torch_model` decorator converts NumPy inputs to PyTorch tensors before your function runs, and converts the PyTorch tensor output back to NumPy when it returns:

```python
from blazerpc import BlazeApp
from blazerpc.contrib.pytorch import torch_model

app = BlazeApp()

@app.model("classifier")
@torch_model(device="cuda")
def classify(image):
    # `image` is a torch.Tensor on CUDA
    return model(image)
    # Return value is converted back to np.ndarray automatically
```

| Parameter | Type  | Default | Description                                      |
| --------- | ----- | ------- | ------------------------------------------------ |
| `device`  | `str` | `"cpu"` | Target device (`"cpu"`, `"cuda"`, `"cuda:0"`).   |

The decorator can be used with or without arguments:

```python
@torch_model           # Uses default device="cpu"
@torch_model()         # Same as above
@torch_model(device="cuda:1")  # Specify a GPU
```

### Standalone conversion functions

If you need more control, use the conversion functions directly:

```python
from blazerpc.contrib.pytorch import torch_to_numpy, numpy_to_torch

# NumPy -> PyTorch
tensor = numpy_to_torch(arr, device="cuda", dtype=torch.float16)

# PyTorch -> NumPy (detaches from graph, moves to CPU)
array = torch_to_numpy(tensor)
```

## TensorFlow

Install the extra:

```bash
pip install blazerpc[tensorflow]
```

### `@tf_model` decorator

Works the same way as `@torch_model`, converting NumPy inputs to TensorFlow tensors and back:

```python
from blazerpc.contrib.tensorflow import tf_model

@app.model("classifier")
@tf_model
def classify(image):
    # `image` is a tf.Tensor
    return model(image)
```

| Parameter | Type  | Default | Description                                  |
| --------- | ----- | ------- | -------------------------------------------- |
| `dtype`   | `Any` | `None`  | Optional TensorFlow dtype to cast inputs to. |

### Standalone conversion functions

```python
from blazerpc.contrib.tensorflow import tf_to_numpy, numpy_to_tf

tensor = numpy_to_tf(arr, dtype=tf.float16)
array = tf_to_numpy(tensor)
```

## ONNX Runtime

Install the extra:

```bash
pip install blazerpc[onnx]
```

### `ONNXModel` wrapper

`ONNXModel` manages an ONNX Runtime inference session and exposes a simple `predict()` method:

```python
from blazerpc.contrib.onnx import ONNXModel

onnx_model = ONNXModel(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

@app.model("classifier")
def classify(image: np.ndarray) -> np.ndarray:
    return onnx_model.predict(image)[0]
```

| Constructor parameter | Type              | Default                      | Description              |
| --------------------- | ----------------- | ---------------------------- | ------------------------ |
| `model_path`          | `str \| Path`     | required                     | Path to the `.onnx` file. |
| `providers`           | `list[str] \| None` | `["CPUExecutionProvider"]` | Execution providers.     |
| `session_options`     | `Any`             | `None`                       | Optional `ort.SessionOptions`. |

#### Positional inputs

`predict()` matches positional arguments to input names in order:

```python
results = onnx_model.predict(input_1, input_2)
# Returns a list of output arrays
```

#### Named inputs

`predict_dict()` accepts a dictionary of named inputs and returns a dictionary of named outputs:

```python
results = onnx_model.predict_dict({
    "input_ids": input_ids_array,
    "attention_mask": attention_mask_array,
})
# Returns {"output_name": array, ...}
```

#### Introspection

```python
print(onnx_model.input_names)   # ["input_ids", "attention_mask"]
print(onnx_model.output_names)  # ["logits"]
```

## Installing all extras

To install all framework integrations at once:

```bash
pip install blazerpc[all]
```

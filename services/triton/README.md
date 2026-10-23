# AutoGluon → Triton Inference Server (ppc64le)

End-to-end guide for exporting the trained AutoGluon credit-default model to ONNX and serving it via Triton Inference Server on **ppc64le**, with a protobuf gRPC client.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  services/forecast/                                                  │
│  TabularPredictor (ag_forecast_model/)                              │
│    └─ WeightedEnsemble_L2                                           │
│         ├─ LightGBM  ◄── best exportable → ONNX                    │
│         ├─ XGBoost                                                  │
│         └─ CatBoost                                                 │
└──────────────────┬──────────────────────────────────────────────────┘
                   │ export_to_onnx.py
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  model_repository/                                                  │
│    credit_default/                                                  │
│      config.pbtxt   (ONNX Runtime backend, 23 FP32 inputs)         │
│      1/model.onnx                                                   │
└──────────────────┬──────────────────────────────────────────────────┘
                   │ mounted as /models
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  quay.io/powercloud/tritonserver:latest                             │
│    HTTP  :8000  (JSON)                                              │
│    gRPC  :8001  (protobuf) ◄── used by infer_client.py             │
│    Metrics :8002                                                    │
└──────────────────┬──────────────────────────────────────────────────┘
                   │ tritonclient.grpc (protobuf on wire)
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  infer_client.py / triton-client container                         │
│  Input:  float32[batch, 23]  (23 UCI Credit Card features)         │
│  Output: int64[batch,1] label + float32[batch,2] probabilities     │
└─────────────────────────────────────────────────────────────────────┘
```

> **Why ONNX + protobuf instead of JSON?**
> Triton's gRPC endpoint uses protobuf framing — zero JSON parsing overhead, efficient binary encoding of float arrays, and support for Triton's dynamic batching. For large batches this is 3–5× faster than the HTTP/JSON endpoint.

> **Why not export WeightedEnsemble?**
> ONNX has no operator for AutoGluon's stacking layer. The best *single* exportable base model (LightGBM > XGBoost > CatBoost by priority) is used instead. This trades ~1–3% accuracy for full ONNX portability and Triton compatibility.

---

## Directory layout

```
services/triton/
├── Containerfile          ← client image (export + inference)
├── export_to_onnx.py      ← Step 1: AutoGluon → ONNX + config.pbtxt
├── infer_client.py        ← Step 3: gRPC protobuf inference client
├── model_repository/      ← populated by export_to_onnx.py
│   └── .gitkeep
└── README.md
```

---

## Prerequisites

| What | Detail |
|------|--------|
| Trained AutoGluon model | `services/forecast/ag_forecast_model/` exists |
| Podman | On the ppc64le host |
| Triton ppc64le image | `quay.io/powercloud/tritonserver:latest` pulled |

---

## Step 1 — Build the client image

```bash
cd services/triton
podman build -t triton-client:latest -f Containerfile .
```

---

## Step 2 — Export AutoGluon model to ONNX

This writes `model_repository/credit_default/1/model.onnx` and `config.pbtxt`.

```bash
podman run --rm \
  -v $(pwd)/../forecast/ag_forecast_model:/ag_model:ro,Z \
  -v $(pwd)/model_repository:/models:Z \
  -e MODEL_DIR=/ag_model \
  -e TRITON_REPO=/models \
  triton-client:latest python export_to_onnx.py
```

After this runs you should see:

```
model_repository/
  credit_default/
    config.pbtxt
    1/
      model.onnx
```

---

## Step 3 — Start Triton server

```bash
podman run --rm \
  -v $(pwd)/model_repository:/models:ro,Z \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  quay.io/powercloud/tritonserver:latest \
  tritonserver --model-repository=/models
```

Wait for:
```
I tritonserver.cc] Started GRPCInferenceService at 0.0.0.0:8001
I tritonserver.cc] Started HTTPService at 0.0.0.0:8000
I tritonserver.cc] Started Metrics Service at 0.0.0.0:8002
```

---

## Step 4 — Run inference smoke-test

```bash
podman run --rm \
  -e TRITON_HOST=<triton-server-ip-or-hostname> \
  triton-client:latest python infer_client.py
```

Expected output:
```
[INFO] Triton gRPC client connected: <host>:8001  model=credit_default v1
[INFO] Server and model ready ✓
[INFO] Running batch inference on 2 samples …
[INFO] Sample 1 — prediction=1  P(default)=0.7231  P(no_default)=0.2769
[INFO] Sample 2 — prediction=0  P(default)=0.1043  P(no_default)=0.8957
[INFO] === Smoke-test complete ===
```

---

## Step 5 — Call from your own code

```python
from infer_client import TritonCreditDefaultClient

client = TritonCreditDefaultClient(host="10.20.x.x")

results = client.predict([
    {
        "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
        "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
        "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
        "BILL_AMT4": 0,    "BILL_AMT5": 0,    "BILL_AMT6": 0,
        "PAY_AMT1": 0,     "PAY_AMT2": 689,   "PAY_AMT3": 0,
        "PAY_AMT4": 0,     "PAY_AMT5": 0,     "PAY_AMT6": 0,
    }
])
# [{"prediction": 1, "probability_of_default": 0.723, "probability_of_no_default": 0.277}]
```

---

## Protobuf wire format explained

`tritonclient.grpc` handles all serialisation automatically. Internally it sends:

```
ModelInferRequest {
  model_name: "credit_default"
  model_version: "1"
  inputs: [
    InferInputTensor {
      name: "float_input"
      datatype: "FP32"
      shape: [batch_size, 23]
      contents: { fp32_contents: [...23 float values per row...] }
    }
  ]
  outputs: [
    InferRequestedOutput { name: "output_label" }
    InferRequestedOutput { name: "output_probability" }
  ]
}
```

You never write this proto manually — `infer_input.set_data_from_numpy(array)` and `response.as_numpy("output_label")` handle the conversion.

---

## Environment variable reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_HOST` | `localhost` | Triton server hostname or IP |
| `TRITON_GRPC_PORT` | `8001` | Triton gRPC port |
| `TRITON_HTTP_PORT` | `8000` | Triton HTTP port (not used by client) |
| `TRITON_MODEL_NAME` | `credit_default` | Model name in the repository |
| `TRITON_MODEL_VERSION` | `1` | Model version |
| `MODEL_DIR` | `../forecast/ag_forecast_model` | AutoGluon model path (export step) |
| `TRITON_REPO` | `./model_repository` | Output path for ONNX + config (export step) |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No ONNX-exportable model found` | Only ensemble trained | Re-train with `ALGORITHMS=GBM` or `medium_quality` preset |
| `Model is not ready` | Export step not run | Run Step 2 first |
| `StatusCode.UNAVAILABLE` | Triton not running | Start Triton (Step 3) |
| `Invalid argument: unexpected shape` | Feature count mismatch | Check `FEATURE_COLUMNS` in `infer_client.py` matches training data |
| Triton OOM on ppc64le | Model too large for CPU instance | Set `max_batch_size: 32` in `config.pbtxt` |

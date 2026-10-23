"""
infer_client.py — Inference client for the credit_default model served by
                  Triton Inference Server.

Supports two transports, selected by TRITON_TRANSPORT env var:
  - "http"  (default) — uses tritonclient.http, binary tensor encoding over
                        HTTP/1.1.  Works when gRPC port 8001 is unavailable.
  - "grpc"            — uses tritonclient.grpc with protobuf framing over
                        HTTP/2.  Faster for large batches when port 8001 is up.

Input  (23 float32 features):
    LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
    PAY_0..PAY_6, BILL_AMT1..BILL_AMT6, PAY_AMT1..PAY_AMT6

Output:
    label        — int64  [batch]    predicted class (0 or 1)
    probabilities — float32 [batch, 2]  class probabilities [P(0), P(1)]

Usage:
    # Smoke-test (HTTP transport, default)
    python infer_client.py

    # Explicit host
    TRITON_HOST=10.20.x.x python infer_client.py

    # Use gRPC transport if port 8001 is available
    TRITON_TRANSPORT=grpc TRITON_HOST=10.20.x.x python infer_client.py

    # Import and call from another service
    from infer_client import TritonCreditDefaultClient
    client = TritonCreditDefaultClient()
    results = client.predict([{"LIMIT_BAL": 20000, "SEX": 2, ...}])
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import numpy as np

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("triton.client")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRITON_HOST = os.getenv("TRITON_HOST", "localhost")
TRITON_GRPC_PORT = int(os.getenv("TRITON_GRPC_PORT", "8001"))
TRITON_HTTP_PORT = int(os.getenv("TRITON_HTTP_PORT", "8000"))
TRITON_TRANSPORT = os.getenv("TRITON_TRANSPORT", "http").lower()  # "http" or "grpc"
MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "credit_default")
MODEL_VERSION = os.getenv("TRITON_MODEL_VERSION", "")  # "" = always use latest active version

# Feature order must exactly match config.pbtxt input dims and ONNX export order
FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class TritonCreditDefaultClient:
    """Inference client for the credit_default ONNX model on Triton.

    Supports HTTP (default, port 8000) and gRPC (port 8001) transports,
    selected via the ``transport`` argument or ``TRITON_TRANSPORT`` env var.

    Args:
        host:          Triton server hostname.   (env: TRITON_HOST)
        http_port:     Triton HTTP port.         (env: TRITON_HTTP_PORT, default 8000)
        grpc_port:     Triton gRPC port.         (env: TRITON_GRPC_PORT, default 8001)
        model_name:    Triton model name.        (env: TRITON_MODEL_NAME)
        model_version: Model version string.     (env: TRITON_MODEL_VERSION, default "1")
        transport:     "http" (default) or "grpc". (env: TRITON_TRANSPORT)
    """

    def __init__(
        self,
        host: str = TRITON_HOST,
        http_port: int = TRITON_HTTP_PORT,
        grpc_port: int = TRITON_GRPC_PORT,
        model_name: str = MODEL_NAME,
        model_version: str = MODEL_VERSION,
        transport: str = TRITON_TRANSPORT,
    ) -> None:
        self._transport = transport
        self._http_url = f"{host}:{http_port}"
        self._grpc_url = f"{host}:{grpc_port}"
        self._model_name = model_name
        self._model_version = model_version
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._transport == "grpc":
            try:
                import tritonclient.grpc as t
            except ImportError as exc:
                raise RuntimeError("tritonclient[grpc] not installed") from exc
            self._client = t.InferenceServerClient(url=self._grpc_url)
            logger.info("Triton gRPC client connected: %s model=%s v%s",
                        self._grpc_url, self._model_name, self._model_version)
        else:
            try:
                import tritonclient.http as t
            except ImportError as exc:
                raise RuntimeError("tritonclient[http] not installed") from exc
            self._client = t.InferenceServerClient(url=self._http_url)
            logger.info("Triton HTTP client connected: %s model=%s v%s",
                        self._http_url, self._model_name, self._model_version)
        return self._client

    def is_ready(self) -> bool:
        """Return True if Triton server and model are ready."""
        client = self._get_client()
        return (
            client.is_server_ready()
            and client.is_model_ready(self._model_name, self._model_version)
        )

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run inference on a batch of records.

        Args:
            records: List of dicts, each with keys matching FEATURE_COLUMNS.
                     Missing keys default to 0.0.

        Returns:
            List of dicts with keys:
                prediction                — int (0 or 1)
                probability_of_default    — float  P(class=1)
                probability_of_no_default — float  P(class=0)
        """
        # Build float32 input array: shape [batch, 23]
        # max_batch_size=0 — Triton does not manage batching, full shape is sent.
        batch = np.array(
            [[float(rec.get(col, 0.0)) for col in FEATURE_COLUMNS] for rec in records],
            dtype=np.float32,
        )
        logger.debug("Input batch shape: %s  transport=%s", batch.shape, self._transport)

        if self._transport == "grpc":
            return self._predict_grpc(batch, records)
        return self._predict_http(batch, records)

    def _predict_grpc(self, batch: np.ndarray, records) -> list[dict[str, Any]]:
        import tritonclient.grpc as triton_grpc
        infer_input = triton_grpc.InferInput("float_input", list(batch.shape), "FP32")
        infer_input.set_data_from_numpy(batch)
        out_label = triton_grpc.InferRequestedOutput("label")
        out_prob = triton_grpc.InferRequestedOutput("probabilities")
        response = self._get_client().infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=[infer_input],
            outputs=[out_label, out_prob],
        )
        labels = response.as_numpy("label").flatten()
        probs = response.as_numpy("probabilities")
        return self._build_results(labels, probs, records)

    def _predict_http(self, batch: np.ndarray, records) -> list[dict[str, Any]]:
        import tritonclient.http as triton_http
        infer_input = triton_http.InferInput("float_input", list(batch.shape), "FP32")
        infer_input.set_data_from_numpy(batch)
        out_label = triton_http.InferRequestedOutput("label")
        out_prob = triton_http.InferRequestedOutput("probabilities")
        response = self._get_client().infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=[infer_input],
            outputs=[out_label, out_prob],
        )
        labels = response.as_numpy("label").flatten()
        probs = response.as_numpy("probabilities")
        return self._build_results(labels, probs, records)

    @staticmethod
    def _build_results(labels, probs, records) -> list[dict[str, Any]]:
        return [
            {
                "prediction": int(labels[i]),
                "probability_of_default": float(probs[i][1]),
                "probability_of_no_default": float(probs[i][0]),
            }
            for i in range(len(records))
        ]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

def main() -> int:
    port = TRITON_GRPC_PORT if TRITON_TRANSPORT == "grpc" else TRITON_HTTP_PORT
    logger.info("=== Triton credit_default inference smoke-test ===")
    logger.info("Server: %s:%s  transport=%s  model=%s v%s",
                TRITON_HOST, port, TRITON_TRANSPORT, MODEL_NAME, MODEL_VERSION)

    client = TritonCreditDefaultClient()

    # Check server readiness
    try:
        ready = client.is_ready()
    except Exception as exc:
        logger.error("Cannot reach Triton server at %s:%s — %s",
                     TRITON_HOST, TRITON_GRPC_PORT, exc)
        logger.error("Is the Triton server running?  "
                     "See README.md for the podman run command.")
        return 1

    if not ready:
        logger.error("Model '%s' v%s is not ready on the server.", MODEL_NAME, MODEL_VERSION)
        return 1

    logger.info("Server and model ready ✓")

    # Two sample records from the UCI Credit Card dataset
    samples = [
        {   # High-risk: large unpaid balance, delayed payments
            "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
            "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
            "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
            "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
            "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0,
            "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
        },
        {   # Low-risk: high limit, payments on time
            "LIMIT_BAL": 90000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 34,
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 29239, "BILL_AMT2": 14027, "BILL_AMT3": 13559,
            "BILL_AMT4": 14331, "BILL_AMT5": 14948, "BILL_AMT6": 15549,
            "PAY_AMT1": 1518, "PAY_AMT2": 1500, "PAY_AMT3": 1000,
            "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 5000,
        },
    ]

    logger.info("Running batch inference on %d samples …", len(samples))
    results = client.predict(samples)

    for i, (sample, result) in enumerate(zip(samples, results), 1):
        logger.info(
            "Sample %d — prediction=%d  P(default)=%.4f  P(no_default)=%.4f",
            i, result["prediction"],
            result["probability_of_default"],
            result["probability_of_no_default"],
        )

    client.close()
    logger.info("=== Smoke-test complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

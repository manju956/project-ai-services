"""
export_to_onnx.py — Export the best AutoGluon base model to ONNX for Triton.

AutoGluon's WeightedEnsemble cannot be exported to ONNX directly.
This script:
  1. Loads the trained TabularPredictor from ag_forecast_model/
  2. Walks the leaderboard to find the best ONNX-exportable base model
     (LightGBM → XGBoost → CatBoost → RandomForest, in that priority order)
  3. Exports that model to ONNX via its native converter
  4. Writes the ONNX file into the Triton model repository layout:
       model_repository/
         credit_default/
           1/
             model.onnx
           config.pbtxt          ← written by write_triton_config()

Usage (from services/forecast/):
    python ../triton/export_to_onnx.py

Or set env vars:
    MODEL_DIR=/path/to/ag_forecast_model
    TRITON_REPO=/path/to/model_repository
    python export_to_onnx.py
"""
from __future__ import annotations

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_onnx")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_FORECAST_DIR = os.path.join(_HERE, "..", "forecast")

AG_MODEL_DIR = os.getenv(
    "MODEL_DIR",
    os.path.join(_FORECAST_DIR, "ag_forecast_model"),
)
TRITON_REPO = os.getenv(
    "TRITON_REPO",
    os.path.join(_HERE, "model_repository"),
)
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "credit_default")
TRITON_MODEL_VERSION = os.getenv("TRITON_MODEL_VERSION", "1")

# Feature columns — must match ForecastInput in forecast_ag.py exactly
# (all 23 UCI Credit Card features, excluding ID and target)
FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
N_FEATURES = len(FEATURE_COLUMNS)  # 23

# Priority order for ONNX export — ensemble is excluded
_ONNX_PRIORITY = ["GBM", "XGB", "CAT", "RF", "XT", "KNN"]


def _find_exportable_candidates(predictor) -> list[tuple[str, str]]:
    """Return ordered list of (ag_model_name, algo_key) for all exportable models.

    Ordered best-first by leaderboard score so main() can try each in turn
    and skip any that fail to export (e.g. due to version mismatches).
    """
    lb = predictor.leaderboard(silent=True)
    candidates = []
    for _, row in lb.iterrows():
        name: str = row["model"]
        for key in _ONNX_PRIORITY:
            if name.upper().startswith(key) or f"_{key}_" in name.upper():
                candidates.append((name, key))
                break
    if not candidates:
        raise RuntimeError(
            "No ONNX-exportable model found in leaderboard. "
            f"Trained models: {lb['model'].tolist()}"
        )
    return candidates


def export_lightgbm(ag_model_name: str, predictor, out_path: str) -> None:
    """Export a LightGBM AutoGluon model to ONNX."""
    import numpy as np
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    # AutoGluon wraps LightGBM in a LGBMClassifier-compatible interface
    model_obj = predictor._trainer.load_model(ag_model_name)
    lgbm_model = model_obj.model  # the raw lightgbm.LGBMClassifier
    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
    onnx_model = convert_sklearn(lgbm_model, initial_types=initial_type,
                                  target_opset=17)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("LightGBM → ONNX written: %s", out_path)


def export_xgboost(ag_model_name: str, predictor, out_path: str) -> None:
    """Export an XGBoost AutoGluon model to ONNX.

    AutoGluon stores the XGBoost model as a raw .ubj (binary JSON) file on
    disk inside the model subdirectory.  We load it directly as an
    xgboost.Booster (bypassing the sklearn wrapper that causes version
    mismatch errors) and convert via onnxmltools.
    """
    import xgboost as xgb
    from onnxmltools.convert.xgboost import convert as convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    # Locate the raw XGBoost model file saved by AutoGluon
    model_subdir = os.path.join(AG_MODEL_DIR, "models", ag_model_name)
    ubj_path = os.path.join(model_subdir, "xgb.ubj")
    json_path = os.path.join(model_subdir, "xgb.json")

    booster = xgb.Booster()
    if os.path.exists(ubj_path):
        booster.load_model(ubj_path)
        logger.debug("Loaded XGBoost Booster from %s", ubj_path)
    elif os.path.exists(json_path):
        booster.load_model(json_path)
        logger.debug("Loaded XGBoost Booster from %s", json_path)
    else:
        raise FileNotFoundError(
            f"XGBoost model file not found in {model_subdir} "
            "(expected xgb.ubj or xgb.json)"
        )

    onnx_model = convert_xgboost(
        booster,
        initial_types=[("float_input", FloatTensorType([None, N_FEATURES]))],
    )
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("XGBoost → ONNX written: %s", out_path)


def export_catboost(ag_model_name: str, predictor, out_path: str) -> None:
    """Export a CatBoost AutoGluon model to ONNX."""
    model_obj = predictor._trainer.load_model(ag_model_name)
    cb_model = model_obj.model  # raw catboost.CatBoostClassifier
    cb_model.save_model(out_path, format="onnx",
                        export_parameters={"opset_domain": 17})
    logger.info("CatBoost → ONNX written: %s", out_path)


def write_triton_config(repo_model_dir: str) -> None:
    """Write config.pbtxt for ONNX Runtime backend in Triton."""
    config = f"""\
name: "{TRITON_MODEL_NAME}"
backend: "onnxruntime"
max_batch_size: 0

input [
  {{
    name: "float_input"
    data_type: TYPE_FP32
    dims: [ -1, {N_FEATURES} ]
  }}
]

output [
  {{
    name: "label"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }},
  {{
    name: "probabilities"
    data_type: TYPE_FP32
    dims: [ -1, 2 ]
  }}
]

instance_group [ {{ kind: KIND_CPU }} ]
"""
    config_path = os.path.join(repo_model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config)
    logger.info("Triton config written: %s", config_path)


def main() -> int:
    from autogluon.tabular import TabularPredictor

    logger.info("Loading AutoGluon predictor from %s", AG_MODEL_DIR)
    predictor = TabularPredictor.load(AG_MODEL_DIR)
    logger.info("Best AutoGluon model: %s", predictor.model_best)

    candidates = _find_exportable_candidates(predictor)
    exporters = {"GBM": export_lightgbm, "XGB": export_xgboost, "CAT": export_catboost}

    version_dir = os.path.join(TRITON_REPO, TRITON_MODEL_NAME, TRITON_MODEL_VERSION)
    os.makedirs(version_dir, exist_ok=True)
    onnx_path = os.path.join(version_dir, "model.onnx")

    exported = False
    for ag_model_name, algo_key in candidates:
        exporter = exporters.get(algo_key)
        if exporter is None:
            logger.warning("No exporter for algo '%s' — skipping '%s'", algo_key, ag_model_name)
            continue
        logger.info("Trying export: '%s' (algo=%s)", ag_model_name, algo_key)
        try:
            exporter(ag_model_name, predictor, onnx_path)
            exported = True
            break
        except Exception as exc:
            logger.warning(
                "Export of '%s' failed (%s: %s) — trying next candidate.",
                ag_model_name, type(exc).__name__, exc,
            )

    if not exported:
        raise RuntimeError(
            "All exportable models failed. Check warnings above for details."
        )

    repo_model_dir = os.path.join(TRITON_REPO, TRITON_MODEL_NAME)
    write_triton_config(repo_model_dir)

    logger.info("Done. Triton model repository: %s", TRITON_REPO)
    logger.info("  %s/%s/1/model.onnx", TRITON_MODEL_NAME, TRITON_MODEL_VERSION)
    logger.info("  %s/config.pbtxt", TRITON_MODEL_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
infer_from_ibmi.py — Step 5: score the IBM i holdout rows with the trained
model and report metrics.  Optionally write predictions back to IBM i.

Two inference paths:
    --mode api    (default) POST each row to the running FastAPI service
                  (POST /forecast on --api-url) — proves the deployed service.
    --mode local  load the AutoGluon model directly (no HTTP, much faster).

Input : poc_ibmi/data/holdout.parquet  (written by pull_from_ibmi.py)
Output: poc_ibmi/data/predictions.csv  (+ CCDEMO.CC_PREDS rows with --write-back)

Usage:
    python infer_from_ibmi.py --limit 200
    python infer_from_ibmi.py --mode local
    python infer_from_ibmi.py --mode local --write-back
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.request
from pathlib import Path

import pandas as pd

from poc_common import FEATURES, LABEL_ML, PRED_TABLE, SCHEMA, exec_once, get_connector

logger = logging.getLogger("poc.infer")
HERE = Path(__file__).resolve().parent
MODEL_DIR = os.getenv("MODEL_DIR", str(HERE.parent / "ag_forecast_model"))


def predict_api(df: pd.DataFrame, url: str) -> pd.DataFrame:
    out = []
    for rec in df[FEATURES].to_dict("records"):
        req = urllib.request.Request(
            url, data=json.dumps(rec).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.load(resp)
        out.append((body["prediction"], body["probability_of_default"]))
    return pd.DataFrame(out, columns=["prediction", "probability"], index=df.index)


def predict_local(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    from autogluon.tabular import TabularPredictor

    pred = TabularPredictor.load(MODEL_DIR)
    proba = pred.predict_proba(df[FEATURES])
    p1 = proba[1] if 1 in proba.columns else proba.iloc[:, 1]
    res = pd.DataFrame(
        {"prediction": (p1 >= 0.5).astype(int), "probability": p1}, index=df.index
    )
    return res, pred.model_best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["api", "local"], default="api")
    ap.add_argument("--api-url", default="http://localhost:8000/forecast")
    ap.add_argument("--holdout", type=Path, default=HERE / "data" / "holdout.parquet")
    ap.add_argument("--limit", type=int, default=0, help="score only first N rows (0 = all)")
    ap.add_argument("--write-back", action="store_true", help="INSERT results into IBM i")
    args = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")

    df = pd.read_parquet(args.holdout)
    if args.limit:
        df = df.head(args.limit)
    logger.info("Scoring %d holdout rows (mode=%s)", len(df), args.mode)

    model_name = "api"
    if args.mode == "api":
        res = predict_api(df, args.api_url)
    else:
        res, model_name = predict_local(df)

    res["ID"] = df["ID"]
    res["actual"] = df[LABEL_ML]

    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

    logger.info("ROC-AUC  : %.4f", roc_auc_score(res["actual"], res["probability"]))
    logger.info("Accuracy : %.4f", accuracy_score(res["actual"], res["prediction"]))
    logger.info("F1       : %.4f", f1_score(res["actual"], res["prediction"]))
    logger.info("Confusion matrix [[TN FP][FN TP]]:\n%s",
                confusion_matrix(res["actual"], res["prediction"]))

    out_csv = HERE / "data" / "predictions.csv"
    res[["ID", "prediction", "probability", "actual"]].to_csv(out_csv, index=False)
    logger.info("Saved %s", out_csv)

    if args.write_back:
        with get_connector() as conn:
            exec_once(conn, f"DELETE FROM {SCHEMA}.{PRED_TABLE}")
            rows = list(res[["ID", "prediction", "probability", "actual"]].itertuples(index=False))
            for i in range(0, len(rows), 500):
                vals = ", ".join(
                    f"({int(r.ID)},{int(r.prediction)},{float(r.probability)!r},"
                    f"{int(r.actual)},'{model_name[:64]}')"
                    for r in rows[i:i + 500]
                )
                exec_once(
                    conn,
                    f"INSERT INTO {SCHEMA}.{PRED_TABLE} "
                    f"(ID, PREDICTION, PROBABILITY_OF_DEFAULT, ACTUAL, MODEL_NAME) VALUES {vals}",
                )
            logger.info("Wrote %d predictions to %s.%s", len(rows), SCHEMA, PRED_TABLE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

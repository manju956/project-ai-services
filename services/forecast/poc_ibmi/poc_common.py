"""
poc_common.py — shared constants and connector bootstrap for the IBM i
credit-card end-to-end POC (load -> pull -> train -> infer).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

SCHEMA = os.getenv("POC_SCHEMA", "CCDEMO").upper()
TABLE = os.getenv("POC_TABLE", "CC_CLIENTS").upper()
PRED_TABLE = os.getenv("POC_PRED_TABLE", "CC_PREDS").upper()

# Column name used in Db2 (dots are not valid in SQL identifiers) and the name
# the trainer / API expect (matches the UCI CSV header).
LABEL_DB = "DEFAULT_PAYMENT_NEXT_MONTH"
LABEL_ML = "default.payment.next.month"

FEATURES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
DB_COLUMNS = ["ID"] + FEATURES + [LABEL_DB]

# Rows with MOD(ID, HOLDOUT_MOD) = 0 are never used for training.
HOLDOUT_MOD = int(os.getenv("POC_HOLDOUT_MOD", "5"))


def exec_once(conn, sql: str):
    """Execute a write/DDL statement exactly once and return the QueryResult.

    MapepireConnector._run_sql() re-executes the statement when the reply says
    is_done=False (its paging fallback).  mapepire-server reports is_done=False
    for statements without a result set (CREATE/INSERT/DELETE), so using
    _run_sql() for writes runs them TWICE: CREATE SCHEMA fails with SQL0601 and
    INSERT would duplicate rows.  All writes therefore go through this helper.

    Raises RuntimeError (from mapepire) if Db2 rejects the statement.  Closing
    the server-side cursor is best effort: the statement has already been
    applied, so a close error must not abort the load.
    """
    job = conn._get_job()
    q = job.query(sql)
    try:
        return q.run()
    finally:
        try:
            q.close()
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("poc.common").debug("ignoring cursor close error: %s", exc)


def get_connector():
    """Return a MapepireConnector (reads IBMI_* env vars).

    Works from the repo (services/mapepire next to services/forecast), inside the
    mapepire-client container (connector in /app) and inside forecast-poc-ibmi
    (run.sh mounts services/mapepire at /opt/mapepire).
    """
    here = Path(__file__).resolve()
    candidates = [Path("/opt/mapepire"), Path("/app")]
    if len(here.parents) > 2:
        candidates.append(here.parents[2] / "mapepire")
    for cand in candidates:
        if (cand / "mapepire_connector.py").exists():
            sys.path.insert(0, str(cand))
            break
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    from mapepire_connector import MapepireConnector

    return MapepireConnector()

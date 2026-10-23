"""
pull_from_ibmi.py — Step 3: pull the credit-card rows from IBM i Db2 via
mapepire and write ML-ready Parquet files (train + holdout).

Split is deterministic and done in SQL:
    holdout = MOD(ID, POC_HOLDOUT_MOD) = 0   (default 20 %, 6 000 rows)
    train   = everything else                (default 80 %, 24 000 rows)

Output (default dir: services/forecast/poc_ibmi/data):
    train.parquet     features + label column "default.payment.next.month"
    holdout.parquet   ID + features + label (label kept only for evaluation)

Why this script pages by itself instead of calling MapepireConnector.extract():
    _run_sql() fetches 100-row pages and, when the result is not done,
    RE-EXECUTES the query to continue paging.  Without a stable ORDER BY the
    second execution can return rows in a different order.  Here we keep ONE
    cursor open, use ORDER BY ID and large pages.

Usage:
    python pull_from_ibmi.py [--out-dir DIR] [--page-size 5000]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from poc_common import (
    FEATURES, HOLDOUT_MOD, LABEL_DB, LABEL_ML, SCHEMA, TABLE, get_connector,
)

logger = logging.getLogger("poc.pull")
HERE = Path(__file__).resolve().parent
# mapepire-python's websocket client drops messages > 1 MiB (error 1009 "message too big").
# ~413 bytes/row of JSON for this table => 2,000 rows ~0.8 MiB; keep a safety margin.
MAX_PAGE_SIZE = 2000


def fetch_all(conn, sql: str, page_size: int) -> list[dict]:
    """Run *sql* on a single cursor and page through it with fetch_more()."""
    from mapepire_python.data_types import QueryOptions

    job = conn._get_job()
    rows: list[dict] = []
    with job.query(sql, QueryOptions(isClCommand=False, parameters=None, autoClose=False)) as q:
        res = q.run(rows_to_fetch=page_size)
        rows.extend(res.data or [])
        while q.state.name != "RUN_DONE":
            page = q.fetch_more(rows_to_fetch=page_size)
            rows.extend(page.data or [])
            logger.info("  fetched %d rows so far", len(rows))
    return rows


def to_frame(rows: list[dict], keep_id: bool) -> pd.DataFrame:
    if not rows:
        raise SystemExit(f"Query returned 0 rows - is {SCHEMA}.{TABLE} loaded? (run: ./run.sh load)")
    df = pd.DataFrame(rows)
    df.columns = [str(c).upper() for c in df.columns]   # tolerate any server-side name casing
    cols = (["ID"] if keep_id else []) + FEATURES + [LABEL_DB]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns missing from result: {missing}; got {list(df.columns)}")
    df = df[cols].astype("int64")
    return df.rename(columns={LABEL_DB: LABEL_ML})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "data")
    ap.add_argument(
        "--page-size", type=int, default=1000,
        help="rows per websocket message (default 1000, max %d). The mapepire client "
             "rejects replies over 1 MiB (close code 1009); a row is ~0.4 KB as JSON." % MAX_PAGE_SIZE,
    )
    args = ap.parse_args()
    if not 1 <= args.page_size <= MAX_PAGE_SIZE:
        ap.error(f"--page-size must be between 1 and {MAX_PAGE_SIZE} "
                 "(larger pages exceed the 1 MiB websocket message limit)")
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cols = ", ".join(["ID"] + FEATURES + [LABEL_DB])
    base = f"SELECT {cols} FROM {SCHEMA}.{TABLE}"
    queries = {
        "train": (f"{base} WHERE MOD(ID, {HOLDOUT_MOD}) <> 0 ORDER BY ID", False),
        "holdout": (f"{base} WHERE MOD(ID, {HOLDOUT_MOD}) = 0 ORDER BY ID", True),
    }

    with get_connector() as conn:
        conn.test_connection()
        for name, (sql, keep_id) in queries.items():
            logger.info("Pulling %s ...", name)
            df = to_frame(fetch_all(conn, sql, args.page_size), keep_id)
            path = args.out_dir / f"{name}.parquet"
            df.to_parquet(path, index=False)
            logger.info("%s: %d rows, default rate %.3f -> %s",
                        name, len(df), df[LABEL_ML].mean(), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

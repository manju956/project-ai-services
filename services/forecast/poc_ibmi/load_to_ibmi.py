"""
load_to_ibmi.py — Step 2: create the schema/tables on IBM i and load the
UCI credit-card CSV (30,000 rows) through mapepire.

Usage (from services/forecast/poc_ibmi):
    python load_to_ibmi.py                     # create + load
    python load_to_ibmi.py --csv /path/to.csv  # custom CSV
    python load_to_ibmi.py --batch 500         # rows per INSERT (default 500)
    python load_to_ibmi.py --dry-run           # print SQL, touch nothing

Re-runnable: tables are CREATE OR REPLACE'd, so each run starts clean.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from poc_common import DB_COLUMNS, SCHEMA, TABLE, exec_once, get_connector

logger = logging.getLogger("poc.load")
HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "UCI_Credit_Card.csv"
CSV_TO_DB = {"default.payment.next.month": "DEFAULT_PAYMENT_NEXT_MONTH"}


def _to_int(v: str) -> int:
    # The Kaggle CSV contains values such as "5e+05" -> parse via float,
    # but only accept whole numbers.
    f = float(v)
    if f != int(f):
        raise ValueError(v)
    return int(f)


def _ddl_statements() -> list[str]:
    text = (HERE / "schema.sql").read_text().replace("${SCHEMA}", SCHEMA)
    text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("--"))
    return [s.strip() for s in text.split(";") if s.strip()]


def _read_rows(csv_path: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        header = [CSV_TO_DB.get(h, h).upper() for h in reader.fieldnames or []]
        if header != DB_COLUMNS:
            raise SystemExit(f"CSV header mismatch.\n got:      {header}\n expected: {DB_COLUMNS}")
        for rec in reader:
            # Whole numbers only -> safe to inline as SQL literals
            try:
                rows.append([_to_int(v.strip()) for v in rec.values()])
            except ValueError:
                raise SystemExit(f"Non-integer value in row: {rec}")
    return rows


def _insert_sql(batch: list[list[int]]) -> str:
    cols = ", ".join(DB_COLUMNS)
    values = ", ".join("(" + ",".join(map(str, r)) + ")" for r in batch)
    return f"INSERT INTO {SCHEMA}.{TABLE} ({cols}) VALUES {values}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")

    rows = _read_rows(args.csv)
    logger.info("Read %d rows from %s", len(rows), args.csv)

    if args.dry_run:
        print(f"CREATE SCHEMA {SCHEMA}")
        for s in _ddl_statements():
            print(s + ";")
        print(_insert_sql(rows[:2]) + ";")
        return 0

    with get_connector() as conn:
        conn.test_connection()

        exists = conn._run_sql(
            "SELECT SCHEMA_NAME FROM QSYS2.SYSSCHEMAS WHERE SCHEMA_NAME = ?", [SCHEMA]
        )
        if exists:
            logger.info("Schema %s already exists - reusing", SCHEMA)
        else:
            exec_once(conn, f"CREATE SCHEMA {SCHEMA}")
            logger.info("Created schema %s", SCHEMA)

        for stmt in _ddl_statements():
            exec_once(conn, stmt)
        logger.info("Tables created in %s", SCHEMA)

        for i in range(0, len(rows), args.batch):
            exec_once(conn, _insert_sql(rows[i:i + args.batch]))
            if (i // args.batch) % 10 == 0:
                logger.info("Inserted %d / %d", min(i + args.batch, len(rows)), len(rows))

        got = conn._run_sql(
            f"SELECT COUNT(*) AS N, SUM({DB_COLUMNS[-1]}) AS DEFAULTS FROM {SCHEMA}.{TABLE}"
        )[0]
        logger.info("Verify: %s", got)
        if int(got["N"]) != len(rows):
            raise SystemExit(f"Row count mismatch: db={got['N']} csv={len(rows)}")
    logger.info("Load complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
run_sql.py — run read-only SQL against IBM i Db2 through mapepire and print a table.

Usage (via the container):
    ./run.sh sql "SELECT COUNT(*) AS N FROM CCDEMO.CC_PREDS"
    ./run.sh sql "SELECT COUNT(*) AS N FROM CCDEMO.CC_PREDS; SELECT ..."   # ';' separates statements
    ./run.sh verify                                                        # canned post-inference checks

Only SELECT / WITH / VALUES statements are accepted, so this cannot modify data.
Statements are split on ';', so do not use ';' inside string literals.
"""
from __future__ import annotations

import sys

from poc_common import PRED_TABLE, SCHEMA, TABLE, get_connector

READ_ONLY = ("SELECT", "WITH", "VALUES")

VERIFY = [
    f"SELECT COUNT(*) AS PRED_ROWS FROM {SCHEMA}.{PRED_TABLE}",
    f"SELECT PREDICTION, ACTUAL, COUNT(*) AS N FROM {SCHEMA}.{PRED_TABLE} "
    f"GROUP BY PREDICTION, ACTUAL ORDER BY PREDICTION, ACTUAL",
    f"SELECT c.ID, c.LIMIT_BAL, p.PROBABILITY_OF_DEFAULT FROM {SCHEMA}.{TABLE} c "
    f"JOIN {SCHEMA}.{PRED_TABLE} p ON p.ID = c.ID "
    f"ORDER BY p.PROBABILITY_OF_DEFAULT DESC FETCH FIRST 10 ROWS ONLY",
]


def show(rows: list[dict]) -> None:
    if not rows:
        print("(0 rows)")
        return
    cols = list(rows[0].keys())
    cells = [[str(r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print(f"({len(rows)} row{'s' if len(rows) != 1 else ''})")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    sql_text = " ".join(argv[1:])
    statements = VERIFY if sql_text.strip() == "--verify" else [
        s.strip() for s in sql_text.split(";") if s.strip()
    ]
    for stmt in statements:
        if stmt.split(None, 1)[0].upper() not in READ_ONLY:
            print(f"Refusing non-read-only statement: {stmt[:60]}", file=sys.stderr)
            return 2

    with get_connector() as conn:
        for stmt in statements:
            print(f"\n> {stmt}")
            show(conn._run_sql(stmt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

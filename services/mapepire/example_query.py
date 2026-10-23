"""
example_query.py — mapepire-python smoke-test / quick-start script.

Reads connection credentials from environment variables (or a .env file).
Run this inside the container to verify end-to-end connectivity:

    podman run --rm \
        -e IBMI_HOST=my.ibmi.host \
        -e IBMI_USER=MYUSER \
        -e IBMI_PASSWORD=secret \
        mapepire-client:latest

Or with a .env file:

    podman run --rm --env-file .env mapepire-client:latest
"""
import json
import logging
import os
import sys

# Load .env file if present (useful for local dev outside the container)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not required; env vars can be set directly

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("mapepire.example")


def main() -> int:
    # ---------------------------------------------------------------
    # 1. Verify required environment variables are present
    #
    # IBMI_PASSWORD is always required — mapepire-server passes it to
    # IBM i Db2 SQL.  IBMI_SSH_KEY_PATH is optional and adds an SSH
    # tunnel for network security, but does not replace the password.
    # ---------------------------------------------------------------
    missing = [v for v in ("IBMI_HOST", "IBMI_USER", "IBMI_PASSWORD") if not os.getenv(v)]
    if missing:
        logger.error(
            "Missing required environment variable(s): %s\n"
            "Note: IBMI_PASSWORD is always required — mapepire-server uses it\n"
            "to authenticate against IBM i Db2 SQL.\n"
            "IBMI_SSH_KEY_PATH is optional (SSH tunnel for network security).\n"
            "\n"
            "Example (SSH tunnel + password):\n"
            "  podman run --rm \\\n"
            "    -e IBMI_HOST=10.20.x.x -e IBMI_USER=MYUSER -e IBMI_PASSWORD=secret \\\n"
            "    -e IBMI_SSH_KEY_PATH=/run/secrets/ibmi_key \\\n"
            "    -v ~/.ssh/id_rsa:/run/secrets/ibmi_key:ro,Z \\\n"
            "    mapepire-client:latest\n"
            "\n"
            "Example (direct, no SSH):\n"
            "  podman run --rm \\\n"
            "    -e IBMI_HOST=10.20.x.x -e IBMI_USER=MYUSER -e IBMI_PASSWORD=secret \\\n"
            "    mapepire-client:latest",
            ", ".join(missing),
        )
        return 1

    # ---------------------------------------------------------------
    # 2. Import connector (bundled alongside this script in /app)
    # ---------------------------------------------------------------
    from mapepire_connector import MapepireConnector

    logger.info("=== mapepire connectivity smoke-test ===")
    logger.info("Target: %s:%s (user=%s)", os.getenv("IBMI_HOST"), os.getenv("IBMI_PORT", "8076"), os.getenv("IBMI_USER"))

    with MapepireConnector() as conn:

        # ---------------------------------------------------------------
        # 3. Connection test
        # ---------------------------------------------------------------
        logger.info("[1/4] Testing connection …")
        ok = conn.test_connection()
        logger.info("      Connection OK: %s", ok)

        # ---------------------------------------------------------------
        # 4. List tables in a sample schema
        # ---------------------------------------------------------------
        schema = os.getenv("IBMI_TEST_SCHEMA", "QIWS")
        logger.info("[2/4] Listing tables in schema '%s' …", schema)
        tables = conn.list_tables(schema)
        logger.info("      Found %d table(s): %s", len(tables), tables[:10])

        # ---------------------------------------------------------------
        # 5. Describe a known IBM i sample table (QIWS.QCUSTCDT)
        # ---------------------------------------------------------------
        test_table = os.getenv("IBMI_TEST_TABLE", "QCUSTCDT")
        if test_table in tables:
            logger.info("[3/4] Describing %s.%s …", schema, test_table)
            columns = conn.list_columns(schema, test_table)
            for col in columns:
                logger.info(
                    "      %-20s  %-20s  nullable=%s",
                    col.name, col.type, col.nullable,
                )
        else:
            logger.info(
                "[3/4] Table %s.%s not found — skipping column listing.",
                schema, test_table,
            )
            columns = []

        # ---------------------------------------------------------------
        # 6. Run a free-form SQL query (configurable via env)
        # ---------------------------------------------------------------
        custom_sql = os.getenv("IBMI_TEST_SQL", "")
        if custom_sql:
            logger.info("[4/4] Running custom SQL: %s", custom_sql)
            rows = conn._run_sql(custom_sql)
            logger.info("      Returned %d row(s).", len(rows))
            if rows:
                logger.info("      First row: %s", json.dumps(rows[0], default=str))
        else:
            logger.info(
                "[4/4] No IBMI_TEST_SQL set — running default "
                "SELECT 1 FROM SYSIBM.SYSDUMMY1 …"
            )
            rows = conn._run_sql("SELECT 1 AS RESULT FROM SYSIBM.SYSDUMMY1")
            logger.info("      Result: %s", rows)

    logger.info("=== Smoke-test complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

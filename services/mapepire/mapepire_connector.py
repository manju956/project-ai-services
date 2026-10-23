"""
MapepireConnector — IBM i connector via mapepire-python (WebSocket/HTTP2).

mapepire-server is the RPM that runs on IBM i and exposes a WebSocket endpoint.
This connector talks to that server using the official ``mapepire-python`` client.

── Authentication architecture ───────────────────────────────────────────────

There are TWO independent authentication layers:

  Layer 1 — SSH (transport):
    The connector opens a local SSH tunnel to IBM i's sshd using a private key.
    This secures the network path and avoids exposing the WebSocket port.
    Controlled by: IBMI_SSH_KEY_PATH, IBMI_SSH_PORT, IBMI_SSH_KEY_PASSPHRASE.

  Layer 2 — Db2 SQL (application):
    mapepire-server receives the WebSocket Basic Auth credentials and passes
    them directly to IBM i's Db2 SQL engine via DriverManager.getConnection().
    IBM i Db2 ALWAYS requires a real username + password — there is no
    token or bypass mechanism.  IBMI_PASSWORD is therefore ALWAYS required,
    regardless of whether an SSH tunnel is used.

── Connection modes ──────────────────────────────────────────────────────────

Mode A — SSH tunnel + password (recommended for secure deployments):
    SSH tunnel secures the network path; password authenticates to Db2.
    Required env vars:  IBMI_HOST, IBMI_USER, IBMI_PASSWORD, IBMI_SSH_KEY_PATH
    Optional env vars:  IBMI_PORT (8076), IBMI_TUNNEL_LOCAL_PORT (18076),
                        IBMI_SSH_PORT (22), IBMI_SSH_KEY_PASSPHRASE,
                        IBMI_IGNORE_TLS

Mode B — direct password (simple, no SSH):
    Required env vars:  IBMI_HOST, IBMI_USER, IBMI_PASSWORD
    Optional env vars:  IBMI_PORT (8076), IBMI_IGNORE_TLS

── mapepire-python API (v1.x) ────────────────────────────────────────────────

The synchronous entry point is SQLJob (not Pool — Pool is async-only in v1.x):

    with SQLJob(creds=DaemonServer(...)) as job:
        result = job.query_and_run("SELECT 1 FROM SYSIBM.SYSDUMMY1")
        # result["data"]  → list of row dicts
        # result["is_done"] → bool

── Usage ─────────────────────────────────────────────────────────────────────

    # Mode A — SSH key (reads all config from env)
    with MapepireConnector() as conn:
        ok = conn.test_connection()
        tables = conn.list_tables("QIWS")

    # Mode A — explicit constructor
    with MapepireConnector(
        host="my.ibmi.host",
        username="MYUSER",
        ssh_key_path="/run/secrets/ibmi_key",
    ) as conn:
        conn.test_connection()

    # Mode B — password
    with MapepireConnector(
        host="my.ibmi.host",
        username="MYUSER",
        password="secret",
    ) as conn:
        conn.test_connection()
"""
from __future__ import annotations

import logging
import os
import socket
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8076
_DEFAULT_SSH_PORT = 22
_DEFAULT_TUNNEL_LOCAL_PORT = 18076
_DEFAULT_BATCH_SIZE = 10_000


# ---------------------------------------------------------------------------
# Inline minimal ABC — no dependency on services/common/data_connector.py
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field


@dataclass
class ColumnMeta:
    name: str
    type: str
    nullable: bool


@dataclass
class ExtractResult:
    output_path: str
    rows_written: int
    columns: List[str] = dc_field(default_factory=list)


class DataConnector(ABC):
    def __enter__(self) -> "DataConnector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @abstractmethod
    def test_connection(self) -> bool: ...

    @abstractmethod
    def list_tables(self, schema: str) -> List[str]: ...

    @abstractmethod
    def list_columns(self, schema: str, table: str) -> List[ColumnMeta]: ...

    @abstractmethod
    def extract(
        self,
        schema: str,
        table: str,
        columns: List[str],
        output_path: str,
        where_clause: Optional[str] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> ExtractResult: ...

    @abstractmethod
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# SSH tunnel helper
# ---------------------------------------------------------------------------

def _free_local_port(preferred: int) -> int:
    """Return *preferred* if it is free, otherwise find a free ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def _open_ssh_tunnel(
    ibmi_host: str,
    ibmi_remote_port: int,
    ssh_username: str,
    ssh_key_path: str,
    ssh_port: int,
    local_port: int,
    key_passphrase: Optional[str],
):
    """Open an SSHTunnelForwarder and return it (already started).

    Forwards  127.0.0.1:<local_port>  →  ibmi_host:<ibmi_remote_port>
    via SSH into ibmi_host:<ssh_port> using the private key at ssh_key_path.
    """
    try:
        from sshtunnel import SSHTunnelForwarder
    except ImportError as exc:
        raise RuntimeError(
            "sshtunnel is not installed. "
            "Rebuild the container image or run: pip install sshtunnel"
        ) from exc

    tunnel = SSHTunnelForwarder(
        (ibmi_host, ssh_port),
        ssh_username=ssh_username,
        ssh_pkey=ssh_key_path,
        ssh_private_key_password=key_passphrase or None,
        remote_bind_address=("127.0.0.1", ibmi_remote_port),
        local_bind_address=("127.0.0.1", local_port),
        logger=logging.getLogger("sshtunnel"),
    )
    tunnel.start()
    logger.info(
        "SSH tunnel open: local 127.0.0.1:%s → %s:%s (via SSH %s@%s:%s)",
        tunnel.local_bind_port, ibmi_host, ibmi_remote_port,
        ssh_username, ibmi_host, ssh_port,
    )
    return tunnel


# ---------------------------------------------------------------------------
# MapepireConnector
# ---------------------------------------------------------------------------

class MapepireConnector(DataConnector):
    """DataConnector backed by mapepire-python v1.x (SQLJob synchronous API).

    IBM i password is ALWAYS required — mapepire-server passes it to Db2 SQL.
    SSH key (optional) secures the network transport via a local port-forward;
    it does not replace the Db2 password.

    Args:
        host:               IBM i hostname or IP.         (env: IBMI_HOST)
        port:               mapepire-server port.         (env: IBMI_PORT, default 8076)
        username:           IBM i user profile.           (env: IBMI_USER)
        password:           IBM i password (always req).  (env: IBMI_PASSWORD)
        ignore_tls:         Skip TLS cert verification.   (env: IBMI_IGNORE_TLS)
        ssh_key_path:       SSH private key for tunnel.   (env: IBMI_SSH_KEY_PATH)
        ssh_port:           IBM i sshd port.              (env: IBMI_SSH_PORT, default 22)
        ssh_key_passphrase: Passphrase for encrypted key. (env: IBMI_SSH_KEY_PASSPHRASE)
        tunnel_local_port:  Local port for SSH tunnel.    (env: IBMI_TUNNEL_LOCAL_PORT,
                                                           default 18076)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ignore_tls: bool = False,
        ssh_key_path: Optional[str] = None,
        ssh_port: Optional[int] = None,
        ssh_key_passphrase: Optional[str] = None,
        tunnel_local_port: Optional[int] = None,
    ) -> None:
        self._host = host or os.environ["IBMI_HOST"]
        self._port = port or int(os.getenv("IBMI_PORT", str(_DEFAULT_PORT)))
        self._username = username or os.environ["IBMI_USER"]
        self._ignore_tls = ignore_tls or (
            os.getenv("IBMI_IGNORE_TLS", "false").lower() == "true"
        )

        # Mode A — SSH key
        self._ssh_key_path = ssh_key_path or os.getenv("IBMI_SSH_KEY_PATH")
        self._ssh_port = ssh_port or int(os.getenv("IBMI_SSH_PORT", str(_DEFAULT_SSH_PORT)))
        self._ssh_key_passphrase = ssh_key_passphrase or os.getenv("IBMI_SSH_KEY_PASSPHRASE")
        self._tunnel_local_port = tunnel_local_port or int(
            os.getenv("IBMI_TUNNEL_LOCAL_PORT", str(_DEFAULT_TUNNEL_LOCAL_PORT))
        )

        # IBM i Db2 password — always required
        self._password = password or os.getenv("IBMI_PASSWORD")
        if not self._password:
            raise ValueError(
                "IBMI_PASSWORD is required: mapepire-server passes it to IBM i Db2 SQL. "
                "SSH key auth (IBMI_SSH_KEY_PATH) secures the network tunnel but does "
                "not replace the Db2 password."
            )

        self._tunnel = None   # sshtunnel.SSHTunnelForwarder (Mode A)
        self._job = None      # mapepire_python.client.sql_job.SQLJob

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _using_ssh_tunnel(self) -> bool:
        return bool(self._ssh_key_path)

    def _ensure_tunnel(self) -> int:
        """Open the SSH tunnel if needed; return the local port to connect to."""
        if not self._using_ssh_tunnel:
            return self._port

        if self._tunnel is None or not self._tunnel.is_active:
            local_port = _free_local_port(self._tunnel_local_port)
            self._tunnel = _open_ssh_tunnel(
                ibmi_host=self._host,
                ibmi_remote_port=self._port,
                ssh_username=self._username,
                ssh_key_path=self._ssh_key_path,  # type: ignore[arg-type]
                ssh_port=self._ssh_port,
                local_port=local_port,
                key_passphrase=self._ssh_key_passphrase,
            )

        return self._tunnel.local_bind_port

    def _get_job(self):
        """Return (and lazily connect) the SQLJob."""
        if self._job is not None:
            return self._job

        try:
            from mapepire_python.client.sql_job import SQLJob
            from mapepire_python.data_types import DaemonServer
        except ImportError as exc:
            raise RuntimeError(
                "mapepire-python is not installed. "
                "Rebuild the container image or run: pip install mapepire-python"
            ) from exc

        local_port = self._ensure_tunnel()

        if self._using_ssh_tunnel:
            connect_host = "127.0.0.1"
            connect_port = local_port
            # TLS cert is issued for the IBM i hostname, not 127.0.0.1.
            # The SSH tunnel secures transport; cert verification against
            # the loopback address is structurally impossible.
            ignore_tls = True
            logger.debug(
                "Mode A (SSH tunnel) — connecting via 127.0.0.1:%s (→ %s:%s)",
                local_port, self._host, self._port,
            )
        else:
            connect_host = self._host
            connect_port = self._port
            ignore_tls = self._ignore_tls
            logger.debug(
                "Mode B (direct) — connecting to %s:%s user=%s",
                self._host, self._port, self._username,
            )

        creds = DaemonServer(
            host=connect_host,
            port=connect_port,
            user=self._username,
            password=self._password,
            ignoreUnauthorized=ignore_tls,
        )

        job = SQLJob()
        job.connect(creds)
        self._job = job
        return self._job

    def _run_sql(self, sql: str, parameters: Optional[list] = None) -> list:
        """Execute *sql* and return all rows as a list of dicts.

        Uses SQLJob.query_and_run() for the common single-page case.
        Falls back to manual fetch_more() paging for large result sets.
        """
        job = self._get_job()

        opts = {}
        if parameters:
            opts["parameters"] = parameters

        result = job.query_and_run(sql, opts=opts if opts else None)

        # query_and_run returns a QueryResult dataclass (not a dict)
        rows = list(result.data or [])

        # If the server indicated more pages, page through with fetch_more
        if not result.is_done:
            from mapepire_python.data_types import QueryOptions
            query_opts = QueryOptions(
                isClCommand=False,
                parameters=parameters or None,
                autoClose=False,
            )
            with job.query(sql, query_opts) as q:
                q.run()
                while q.state.name != "RUN_DONE":
                    page = q.fetch_more()
                    if page and page.data:
                        rows.extend(page.data)

        return rows

    # ------------------------------------------------------------------
    # DataConnector interface
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Ping mapepire-server and verify credentials.

        Returns:
            ``True`` on success.

        Raises:
            RuntimeError: on connection failure.
        """
        rows = self._run_sql("SELECT 1 AS PING FROM SYSIBM.SYSDUMMY1")
        if not rows:
            raise RuntimeError(
                f"mapepire connection test returned no rows "
                f"(host={self._host} port={self._port})"
            )
        mode = "SSH-tunnel" if self._using_ssh_tunnel else "direct"
        logger.info(
            "mapepire connection test passed [%s]: host=%s port=%s user=%s",
            mode, self._host, self._port, self._username,
        )
        return True

    def list_tables(self, schema: str) -> List[str]:
        """Return all table names in *schema* (IBM i library).

        Args:
            schema: IBM i library name (case-insensitive).

        Returns:
            Sorted list of table names.
        """
        rows = self._run_sql(
            "SELECT TABLE_NAME FROM QSYS2.SYSTABLES "
            "WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = 'T' "
            "ORDER BY TABLE_NAME",
            [schema.upper()],
        )
        return [row["TABLE_NAME"] for row in rows]

    def list_columns(self, schema: str, table: str) -> List[ColumnMeta]:
        """Return column metadata for *schema.table*.

        Args:
            schema: IBM i library name.
            table:  Table name.

        Returns:
            Ordered list of :class:`ColumnMeta` objects.

        Raises:
            LookupError: if no columns are returned (table not found / no access).
        """
        rows = self._run_sql(
            "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE "
            "FROM QSYS2.SYSCOLUMNS "
            "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? "
            "ORDER BY ORDINAL_POSITION",
            [schema.upper(), table.upper()],
        )
        if not rows:
            raise LookupError(
                f"Table '{schema}.{table}' not found or not accessible "
                f"for user '{self._username}' on {self._host}:{self._port}"
            )
        return [
            ColumnMeta(
                name=row["COLUMN_NAME"],
                type=row["DATA_TYPE"],
                nullable=str(row["IS_NULLABLE"]).strip().upper() == "Y",
            )
            for row in rows
        ]

    def extract(
        self,
        schema: str,
        table: str,
        columns: List[str],
        output_path: str,
        where_clause: Optional[str] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> ExtractResult:
        """Stream rows from *schema.table* to a Parquet file at *output_path*.

        Requires ``pyarrow`` (add it to the Containerfile pip install and rebuild).

        Args:
            schema:       IBM i library name.
            table:        Table name.
            columns:      Column names to SELECT.
            output_path:  Destination Parquet file path.
            where_clause: Optional SQL WHERE predicate (no ``WHERE`` keyword).
            batch_size:   Ignored — mapepire fetches all rows in one call.

        Returns:
            :class:`ExtractResult` with output path and total rows written.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required for extract(). "
                "Add 'pyarrow' to the pip install in Containerfile."
            ) from exc

        col_meta = self.list_columns(schema, table)
        available = {cm.name.upper() for cm in col_meta}
        missing = [c for c in columns if c.upper() not in available]
        if missing:
            raise LookupError(
                f"Column(s) not found in {schema}.{table}: {missing}"
            )

        col_list = ", ".join(f'"{c}"' for c in columns)
        sql = f'SELECT {col_list} FROM "{schema.upper()}"."{table.upper()}"'
        if where_clause:
            sql = f"{sql} WHERE {where_clause}"

        logger.info(
            "Starting extraction: %s.%s columns=%s output=%s",
            schema, table, columns, output_path,
        )

        _TYPE_MAP = {
            "SMALLINT": pa.int16(), "INTEGER": pa.int32(), "INT": pa.int32(),
            "BIGINT": pa.int64(), "REAL": pa.float32(), "FLOAT": pa.float64(),
            "DOUBLE": pa.float64(), "DOUBLE PRECISION": pa.float64(),
            "DECIMAL": pa.float64(), "NUMERIC": pa.float64(),
        }
        col_type_map = {cm.name.upper(): cm.type for cm in col_meta}
        arrow_fields = [
            pa.field(
                c,
                _TYPE_MAP.get(
                    col_type_map.get(c.upper(), "VARCHAR").split("(")[0].upper(),
                    pa.string(),
                ),
            )
            for c in columns
        ]
        arrow_schema = pa.schema(arrow_fields)

        rows = self._run_sql(sql)
        total_rows = len(rows)

        if total_rows > 0:
            col_data = {c: [row.get(c) for row in rows] for c in columns}
            pq.write_table(pa.table(col_data, schema=arrow_schema), output_path)
        else:
            pq.write_table(
                pa.table({c: [] for c in columns}, schema=arrow_schema),
                output_path,
            )

        logger.info(
            "Extraction complete: %s.%s — %d rows written to %s",
            schema, table, total_rows, output_path,
        )
        return ExtractResult(
            output_path=output_path,
            rows_written=total_rows,
            columns=list(columns),
        )

    def close(self) -> None:
        """Close the SQLJob and SSH tunnel (if open)."""
        if self._job is not None:
            try:
                self._job.close()
                logger.debug("mapepire SQLJob closed.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing mapepire SQLJob: %s", exc)
            finally:
                self._job = None

        if self._tunnel is not None:
            try:
                self._tunnel.stop()
                logger.debug("SSH tunnel closed.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing SSH tunnel: %s", exc)
            finally:
                self._tunnel = None

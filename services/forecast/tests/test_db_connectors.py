"""
Tests for DataConnector ABC, shared types, factory, Db2iConnector, and OracleConnector stub.

All DB driver calls (jaydebeapi.connect) and pyarrow Parquet writes are mocked
so no live database or filesystem is required.

Oracle tests verify only that the Phase 2 stub raises NotImplementedError
and that the factory raises NotImplementedError for db_type="oracle".
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from common.data_connector import (
    ColumnMeta,
    DataConnector,
    ExtractResult,
    get_data_connector,
)
from common.forecast_errors import (
    DBAuthError,
    DBColumnNotFoundError,
    DBConnectionError,
    DBTableNotFoundError,
)


# ===========================================================================
# Shared types — ColumnMeta and ExtractResult
# ===========================================================================

class TestColumnMeta:
    def test_fields(self):
        cm = ColumnMeta(name="UNITS_SOLD", type="DECIMAL", nullable=True)
        assert cm.name == "UNITS_SOLD"
        assert cm.type == "DECIMAL"
        assert cm.nullable is True

    def test_not_nullable(self):
        cm = ColumnMeta(name="ID", type="INTEGER", nullable=False)
        assert cm.nullable is False


class TestExtractResult:
    def test_fields(self):
        er = ExtractResult(
            output_path="/var/forecast/data/job_1.parquet",
            rows_written=5000,
            columns=["COL_A", "COL_B"],
        )
        assert er.output_path == "/var/forecast/data/job_1.parquet"
        assert er.rows_written == 5000
        assert er.columns == ["COL_A", "COL_B"]

    def test_columns_defaults_empty(self):
        er = ExtractResult(output_path="/tmp/out.parquet", rows_written=0)
        assert er.columns == []


# ===========================================================================
# Factory — get_data_connector(db_type, config)
# ===========================================================================

_DB2I_CONFIG = {
    "host": "ibmi.example.com",
    "port": 446,
    "database": "PRODDB",
    "username": "SVCACCT",
    "password": "s3cr3t",
}


class TestGetDataConnector:
    def test_returns_db2i_connector(self):
        from common.db2i_connector import Db2iConnector
        connector = get_data_connector("db2i", _DB2I_CONFIG)
        assert isinstance(connector, Db2iConnector)

    def test_db_type_case_insensitive(self):
        from common.db2i_connector import Db2iConnector
        connector = get_data_connector("DB2I", _DB2I_CONFIG)
        assert isinstance(connector, Db2iConnector)

    def test_unsupported_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported db_type"):
            get_data_connector("mysql", {"host": "h", "database": "d",
                                         "username": "u", "password": "p"})

    def test_oracle_raises_not_implemented(self):
        """Oracle is a known future type — factory raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="future phase"):
            get_data_connector("oracle", {"host": "h", "service_name": "ORCL",
                                           "username": "u", "password": "p"})

    def test_connector_is_context_manager(self):
        connector = get_data_connector("db2i", _DB2I_CONFIG)
        connector.close = MagicMock()
        with connector:
            pass
        connector.close.assert_called_once()

    def test_optional_config_keys_forwarded(self):
        from common.db2i_connector import Db2iConnector
        cfg = {**_DB2I_CONFIG, "connect_timeout": 60, "query_timeout": 7200}
        connector = get_data_connector("db2i", cfg)
        assert isinstance(connector, Db2iConnector)
        assert connector._connect_timeout == 60
        assert connector._query_timeout == 7200

    def test_port_defaults_to_446_when_omitted(self):
        from common.db2i_connector import Db2iConnector
        cfg = {k: v for k, v in _DB2I_CONFIG.items() if k != "port"}
        connector = get_data_connector("db2i", cfg)
        assert connector._port == 446


# ===========================================================================
# Helpers for Db2iConnector tests
# ===========================================================================

def _make_jaydebeapi_mock(cursor_mock):
    conn_mock = MagicMock()
    conn_mock.cursor.return_value = cursor_mock
    jdb_mock = MagicMock()
    jdb_mock.connect.return_value = conn_mock
    return jdb_mock, conn_mock


def _make_db2i_connector():
    from common.db2i_connector import Db2iConnector
    return Db2iConnector(
        host="ibmi.example.com", port=446,
        database="PRODDB", username="SVCACCT", password="s3cr3t",
    )


# ===========================================================================
# Db2iConnector — test_connection
# ===========================================================================

class TestDb2iConnectorTestConnection:
    def test_returns_true_on_success(self):
        cursor_mock = MagicMock()
        jdb_mock, _ = _make_jaydebeapi_mock(cursor_mock)

        with patch.dict("sys.modules", {"jaydebeapi": jdb_mock}):
            connector = _make_db2i_connector()
            result = connector.test_connection()

        assert result is True
        cursor_mock.execute.assert_called_once_with("SELECT 1 FROM SYSIBM.SYSDUMMY1")
        cursor_mock.close.assert_called_once()

    def test_connection_failure_raises_db_connection_error(self):
        jdb_mock = MagicMock()
        jdb_mock.connect.side_effect = Exception("Connection refused")

        with patch.dict("sys.modules", {"jaydebeapi": jdb_mock}):
            connector = _make_db2i_connector()
            with pytest.raises(DBConnectionError, match="Cannot connect to IBM i Db2"):
                connector.test_connection()

    def test_auth_failure_raises_db_auth_error(self):
        jdb_mock = MagicMock()
        jdb_mock.connect.side_effect = Exception("not authorized")

        with patch.dict("sys.modules", {"jaydebeapi": jdb_mock}):
            connector = _make_db2i_connector()
            with pytest.raises(DBAuthError, match="Authentication failed"):
                connector.test_connection()

    def test_missing_jaydebeapi_raises_db_connection_error(self):
        with patch.dict("sys.modules", {"jaydebeapi": None}):
            connector = _make_db2i_connector()
            with pytest.raises(DBConnectionError, match="jaydebeapi is not installed"):
                connector.test_connection()


# ===========================================================================
# Db2iConnector — list_tables
# ===========================================================================

class TestDb2iConnectorListTables:
    def test_returns_table_names(self):
        cursor_mock = MagicMock()
        cursor_mock.fetchall.return_value = [("ORDERS",), ("PRODUCTS",)]
        jdb_mock, _ = _make_jaydebeapi_mock(cursor_mock)

        with patch.dict("sys.modules", {"jaydebeapi": jdb_mock}):
            connector = _make_db2i_connector()
            tables = connector.list_tables("SALES")

        assert tables == ["ORDERS", "PRODUCTS"]
        cursor_mock.close.assert_called_once()

    def test_empty_schema_returns_empty_list(self):
        cursor_mock = MagicMock()
        cursor_mock.fetchall.return_value = []
        jdb_mock, _ = _make_jaydebeapi_mock(cursor_mock)

        with patch.dict("sys.modules", {"jaydebeapi": jdb_mock}):
            connector = _make_db2i_connector()
            assert connector.list_tables("EMPTY") == []


# ===========================================================================
# Db2iConnector — list_columns
# ===========================================================================

class TestDb2iConnectorListColumns:
    def test_returns_column_meta_objects(self):
        cursor_mock = MagicMock()
        cursor_mock.fetchall.return_value = [
            ("PRODUCT_ID", "INTEGER", "N"),
            ("UNITS_SOLD", "DECIMAL", "Y"),
        ]
        jdb_mock, _ = _make_jaydebeapi_mock(cursor_mock)

        with patch.dict("sys.modules", {"jaydebeapi": jdb_mock}):
            connector = _make_db2i_connector()
            cols = connector.list_columns("SALES", "DAILY_ORDERS")

        assert len(cols) == 2
        assert all(isinstance(c, ColumnMeta) for c in cols)
        assert cols[0] == ColumnMeta(name="PRODUCT_ID", type="INTEGER", nullable=False)
        assert cols[1] == ColumnMeta(name="UNITS_SOLD",  type="DECIMAL", nullable=True)

    def test_table_not_found_raises(self):
        cursor_mock = MagicMock()
        cursor_mock.fetchall.return_value = []
        jdb_mock, _ = _make_jaydebeapi_mock(cursor_mock)

        with patch.dict("sys.modules", {"jaydebeapi": jdb_mock}):
            connector = _make_db2i_connector()
            with pytest.raises(DBTableNotFoundError, match="not found or not accessible"):
                connector.list_columns("SALES", "MISSING_TABLE")


# ===========================================================================
# Db2iConnector — extract (Parquet output mocked via pyarrow patch)
# ===========================================================================

class TestDb2iConnectorExtract:
    """extract() writes Parquet via pyarrow.  We mock pq.ParquetWriter so no
    filesystem access is needed, and verify the returned ExtractResult."""

    def _run_extract(self, col_rows, data_rows, columns, where_clause=None):
        cursor_mock = MagicMock()
        cursor_mock.fetchall.return_value = col_rows
        cursor_mock.fetchmany.side_effect = [data_rows, []]

        jdb_mock, _ = _make_jaydebeapi_mock(cursor_mock)
        writer_mock = MagicMock()
        writer_cls_mock = MagicMock(return_value=writer_mock)

        with patch.dict("sys.modules", {"jaydebeapi": jdb_mock}):
            with patch("common.db2i_connector.pq.ParquetWriter", writer_cls_mock):
                connector = _make_db2i_connector()
                result = connector.extract(
                    schema="SALES",
                    table="DAILY_ORDERS",
                    columns=columns,
                    output_path="/tmp/out.parquet",
                    where_clause=where_clause,
                )
        return result, writer_mock, writer_cls_mock

    def test_returns_extract_result(self):
        col_rows = [("PRODUCT_ID", "INTEGER", "N"), ("UNITS_SOLD", "DECIMAL", "Y")]
        data_rows = [(1, 100), (2, 200)]
        result, _, _ = self._run_extract(col_rows, data_rows, ["PRODUCT_ID", "UNITS_SOLD"])

        assert isinstance(result, ExtractResult)
        assert result.output_path == "/tmp/out.parquet"
        assert result.rows_written == 2
        assert result.columns == ["PRODUCT_ID", "UNITS_SOLD"]

    def test_parquet_writer_opened_and_closed(self):
        col_rows = [("PRODUCT_ID", "INTEGER", "N")]
        data_rows = [(1,)]
        _, writer_mock, writer_cls_mock = self._run_extract(col_rows, data_rows, ["PRODUCT_ID"])
        writer_cls_mock.assert_called_once()
        writer_mock.write_batch.assert_called_once()
        writer_mock.close.assert_called_once()

    def test_zero_rows_returns_zero_rows_written(self):
        col_rows = [("PRODUCT_ID", "INTEGER", "N")]
        cursor_mock = MagicMock()
        cursor_mock.fetchall.return_value = col_rows
        cursor_mock.fetchmany.return_value = []

        jdb_mock, _ = _make_jaydebeapi_mock(cursor_mock)
        writer_cls_mock = MagicMock()

        with patch.dict("sys.modules", {"jaydebeapi": jdb_mock}):
            with patch("common.db2i_connector.pq.ParquetWriter", writer_cls_mock):
                connector = _make_db2i_connector()
                result = connector.extract("SALES", "EMPTY", ["PRODUCT_ID"],
                                           output_path="/tmp/out.parquet")

        assert result.rows_written == 0
        writer_cls_mock.assert_not_called()

    def test_missing_column_raises_db_column_not_found_error(self):
        cursor_mock = MagicMock()
        cursor_mock.fetchall.return_value = [("COL_A", "INTEGER", "N")]
        jdb_mock, _ = _make_jaydebeapi_mock(cursor_mock)

        with patch.dict("sys.modules", {"jaydebeapi": jdb_mock}):
            connector = _make_db2i_connector()
            with pytest.raises(DBColumnNotFoundError, match="not found"):
                connector.extract("SALES", "DAILY_ORDERS", ["MISSING_COL"],
                                  output_path="/tmp/out.parquet")

    def test_where_clause_appended_to_sql(self):
        col_rows = [("STATUS", "CHAR", "N")]
        cursor_mock = MagicMock()
        cursor_mock.fetchall.return_value = col_rows
        cursor_mock.fetchmany.return_value = []

        jdb_mock, _ = _make_jaydebeapi_mock(cursor_mock)
        writer_cls_mock = MagicMock()

        with patch.dict("sys.modules", {"jaydebeapi": jdb_mock}):
            with patch("common.db2i_connector.pq.ParquetWriter", writer_cls_mock):
                connector = _make_db2i_connector()
                connector.extract(
                    "SALES", "ORDERS", ["STATUS"],
                    output_path="/tmp/out.parquet",
                    where_clause="STATUS = 'OPEN'",
                )

        execute_calls = cursor_mock.execute.call_args_list
        data_sql = execute_calls[-1][0][0]
        assert "WHERE STATUS = 'OPEN'" in data_sql


# ===========================================================================
# Db2iConnector — close / idempotency
# ===========================================================================

class TestDb2iConnectorClose:
    def test_close_called_once_on_connection(self):
        cursor_mock = MagicMock()
        jdb_mock, conn_mock = _make_jaydebeapi_mock(cursor_mock)

        with patch.dict("sys.modules", {"jaydebeapi": jdb_mock}):
            connector = _make_db2i_connector()
            cursor_mock.execute.return_value = None
            connector.test_connection()
            connector.close()

        conn_mock.close.assert_called_once()

    def test_close_is_idempotent(self):
        cursor_mock = MagicMock()
        jdb_mock, conn_mock = _make_jaydebeapi_mock(cursor_mock)

        with patch.dict("sys.modules", {"jaydebeapi": jdb_mock}):
            connector = _make_db2i_connector()
            cursor_mock.execute.return_value = None
            connector.test_connection()
            connector.close()
            connector.close()  # must not raise

        conn_mock.close.assert_called_once()


# ===========================================================================
# OracleConnector stub — Phase 2 not-implemented assertions
# ===========================================================================

class TestOracleConnectorStub:
    def _make_stub(self):
        from common.oracle_connector import OracleConnector
        return OracleConnector(
            host="oracle.example.com", port=1521,
            service_name="ORCL", username="u", password="p",
        )

    def test_test_connection_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Phase 2"):
            self._make_stub().test_connection()

    def test_list_tables_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self._make_stub().list_tables("SCHEMA")

    def test_list_columns_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self._make_stub().list_columns("SCHEMA", "TABLE")

    def test_extract_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self._make_stub().extract("SCHEMA", "TABLE", ["COL"],
                                      output_path="/tmp/out.parquet")

    def test_close_does_not_raise(self):
        self._make_stub().close()  # must not raise


# ===========================================================================
# _db2i_type_to_arrow helper
# ===========================================================================

class TestDb2iTypeToArrow:
    def test_integer_maps_to_int32(self):
        import pyarrow as pa
        from common.db2i_connector import _db2i_type_to_arrow
        assert _db2i_type_to_arrow("INTEGER") == pa.int32()

    def test_decimal_with_precision_stripped(self):
        import pyarrow as pa
        from common.db2i_connector import _db2i_type_to_arrow
        assert _db2i_type_to_arrow("DECIMAL(10,2)") == pa.float64()

    def test_unknown_type_falls_back_to_string(self):
        import pyarrow as pa
        from common.db2i_connector import _db2i_type_to_arrow
        assert _db2i_type_to_arrow("CLOB") == pa.string()

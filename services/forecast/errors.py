"""
errors.py — re-exports all Forecast Service error types from the shared
common/forecast_errors module so that code inside forecast/ can import from
a local namespace.

Usage inside forecast/:
    from forecast.errors import DBConnectionError, raise_http_error
"""
from common.forecast_errors import (  # noqa: F401
    DBConnectionError,
    DBAuthError,
    DBTableNotFoundError,
    DBColumnNotFoundError,
    UnsupportedOutputTypeError,
    TrainingInProgressError,
    NoModelAvailableError,
    SchemaMismatchError,
    TrainingFailedError,
    raise_http_error,
)

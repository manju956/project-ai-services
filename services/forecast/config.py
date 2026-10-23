"""
config.py — environment-variable driven configuration for the Forecast Service.

Follows the pydantic_settings pattern used across all services in this repo
(see digitize/settings.py, extract/settings.py for reference).

All variables have sensible defaults so the service starts without any env
overrides in development.  In production, mount the values via OpenShift
ConfigMap / Secret as described in §15 of the design doc.
"""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ForecastConfig(BaseSettings):
    """Forecast Service configuration."""

    model_config = SettingsConfigDict(env_prefix="FORECAST_")

    # ------------------------------------------------------------------
    # Directory layout
    # ------------------------------------------------------------------
    model_dir: Path = Field(
        default=Path("/var/forecast/models"),
        description="Root directory for all model versions",
    )
    data_dir: Path = Field(
        default=Path("/var/forecast/data"),
        description="Temporary Parquet extraction cache; deleted after training",
    )
    jobs_dir: Path = Field(
        default=Path("/var/forecast/jobs"),
        description="Job status JSON files",
    )

    # ------------------------------------------------------------------
    # AutoML scanning
    # ------------------------------------------------------------------
    scan_sample_rows: int = Field(
        default=20_000,
        ge=1,
        description="Max rows used during AutoML scan phase",
    )
    scan_cv_folds: int = Field(
        default=3,
        ge=2,
        description="Cross-validation folds during AutoML scan",
    )
    scan_optional_algos: bool = Field(
        default=True,
        description=(
            "Include optional algorithms (Random Forest, Gradient Boosting, "
            "Ridge/Logistic) in AutoML scan. LightGBM and XGBoost are always "
            "included regardless of this setting."
        ),
    )

    # ------------------------------------------------------------------
    # Full training
    # ------------------------------------------------------------------
    hyperparam_iterations: int = Field(
        default=20,
        ge=1,
        description="RandomizedSearchCV iterations for full hyperparameter tuning",
    )
    max_training_rows: int = Field(
        default=5_000_000,
        ge=1,
        description="Row cap per training run; tune to stay within the <1 h SLA",
    )
    imbalance_ratio_threshold: int = Field(
        default=10,
        ge=1,
        description=(
            "Auto-enable class_weight='balanced' when majority/minority ratio "
            "exceeds this value"
        ),
    )


class ConnectorConfig(BaseSettings):
    """DB connector configuration (consumed by the DB Component layer)."""

    model_config = SettingsConfigDict(env_prefix="CONNECTOR_")

    batch_size: int = Field(
        default=10_000,
        ge=1,
        description="DB fetch batch size per JDBC round-trip",
    )
    connect_timeout_s: int = Field(
        default=30,
        ge=1,
        description="DB connection timeout in seconds",
    )
    query_timeout_s: int = Field(
        default=3_600,
        ge=1,
        description="Max extraction query time in seconds (1 h)",
    )


# Module-level singletons — import these in application code
forecast_config = ForecastConfig()
connector_config = ConnectorConfig()

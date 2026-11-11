"""Versioned table contracts shared by adapters and offline snapshot loading."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


SCHEMA_VERSION = "1.0.0"
BAR_COLUMNS = [
    "Instrument_ID", "Session", "Open_Time", "Close_Time", "Available_At",
    "Open", "High", "Low", "Close", "Volume", "Currency", "Source_ID",
]
ACTION_COLUMNS = [
    "Instrument_ID", "Action_ID", "Action_Type", "Effective_Time", "Ex_Date",
    "Pay_Date", "Amount_Per_Share", "Split_Ratio", "Available_At", "Source_ID",
]
OBSERVATION_COLUMNS = [
    "Observation_ID", "Rate_Date", "Available_At", "Quoted_Rate", "Rate_Units",
    "Day_Count", "Compounding", "Source_ID",
]
INTERVAL_COLUMNS = [
    "Start_Time", "End_Time", "Credited_Return", "Reference_Return",
    "Observation_IDs", "Availability_Policy",
]
SCHEDULE_COLUMNS = ["Session", "Open_Time", "Close_Time"]
TABLE_COLUMNS = {
    "bars": BAR_COLUMNS, "actions": ACTION_COLUMNS,
    "cash_observations": OBSERVATION_COLUMNS, "cash_intervals": INTERVAL_COLUMNS,
    "schedule": SCHEDULE_COLUMNS,
}
TIMESTAMP_COLUMNS = {"Open_Time", "Close_Time", "Effective_Time", "Available_At",
                     "Start_Time", "End_Time"}


class DataValidationError(ValueError):
    """An input cannot satisfy the declared research data contract."""


def empty_table(name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=TABLE_COLUMNS[name])


def utc_timestamp(value) -> pd.Timestamp:
    """Require a timezone instead of silently interpreting local time as UTC."""
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise DataValidationError(f"Expected a timezone-aware timestamp, got {value!r}")
    return result.tz_convert("UTC")


@dataclass(frozen=True)
class ValidationPolicy:
    missing_sessions: str = "reject"
    unexpected_sessions: str = "reject"
    dividend_payment: str = "reject"
    reconciliation_tolerance: float = 0.00005

    def __post_init__(self):
        if self.missing_sessions not in {"reject", "report"}:
            raise ValueError("missing_sessions must be reject or report")
        if self.unexpected_sessions not in {"reject", "report"}:
            raise ValueError("unexpected_sessions must be reject or report")
        if self.dividend_payment not in {"reject", "ex_date_credit_approximation"}:
            raise ValueError("A dividend payment-date policy is required")
        if not 0 < self.reconciliation_tolerance < 1:
            raise ValueError("Reconciliation tolerance must be between zero and one")


@dataclass(frozen=True)
class Dataset:
    bars: pd.DataFrame
    actions: pd.DataFrame
    cash_observations: pd.DataFrame
    cash_intervals: pd.DataFrame
    schedule: pd.DataFrame
    manifest: dict[str, Any]
    quality_report: dict[str, Any]
    reconciliation: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def dataset_id(self) -> str:
        return self.manifest["Dataset_ID"]


@dataclass(frozen=True)
class HistoryRequirements:
    training_years: int = 20
    feature_warmup_sessions: int = 200
    calibration_sessions: int = 252
    label_horizon_sessions: int = 5
    minimum_training_rows: int = 252
    holding_sessions: int = 1
    decision_delay_minutes: int = 1

    def __post_init__(self):
        for key in ("training_years", "feature_warmup_sessions", "label_horizon_sessions",
                    "minimum_training_rows", "holding_sessions", "decision_delay_minutes"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.calibration_sessions, int) or self.calibration_sessions < 0:
            raise ValueError("calibration_sessions must be a nonnegative integer")


@dataclass(frozen=True)
class EligibleSample:
    eligible: pd.DataFrame
    exclusions: pd.DataFrame

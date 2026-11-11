"""Fail-closed validation with explicit reports for permitted historical gaps."""

import numpy as np
import pandas as pd

from .contracts import DataValidationError, TABLE_COLUMNS, ValidationPolicy, utc_timestamp


def _require(frame, columns, name):
    absent = set(columns) - set(frame.columns)
    if absent:
        raise DataValidationError(f"{name} missing columns {sorted(absent)}")


def _dates(series, name, nullable=False):
    for value in series:
        if pd.isna(value) and nullable:
            continue
        if pd.isna(value) or str(value) != pd.Timestamp(value).strftime("%Y-%m-%d"):
            raise DataValidationError(f"{name} requires ISO session dates")


def _times(series, name, nullable=False):
    for value in series:
        if pd.isna(value) and nullable:
            continue
        try:
            utc_timestamp(value)
        except (ValueError, TypeError) as error:
            raise DataValidationError(f"Invalid {name} timestamp") from error


def _finite(frame, columns, name):
    try:
        values = frame[columns].to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise DataValidationError(f"{name} requires numeric values") from error
    if not np.isfinite(values).all():
        raise DataValidationError(f"{name} contains missing or non-finite values")


def validate_tables(tables, policy: ValidationPolicy) -> dict:
    for name, columns in TABLE_COLUMNS.items():
        _require(tables[name], columns, name)
    bars, actions, schedule = tables["bars"], tables["actions"], tables["schedule"]
    if bars.empty or schedule.empty:
        raise DataValidationError("Market bars and the exchange schedule cannot be empty")
    for frame, name in ((bars, "bars"), (schedule, "schedule")):
        _dates(frame.Session, name)
        if frame.Session.duplicated().any() or not frame.Session.is_monotonic_increasing:
            raise DataValidationError(f"{name} sessions must be unique and sorted")
        for column in ("Open_Time", "Close_Time"):
            _times(frame[column], column)
        if not (frame.Open_Time < frame.Close_Time).all():
            raise DataValidationError(f"{name} open must precede close")
    if bars.Instrument_ID.nunique(dropna=False) != 1:
        raise DataValidationError("A snapshot must contain exactly one instrument")
    for column in ("Instrument_ID", "Currency", "Source_ID"):
        if bars[column].isna().any() or bars[column].astype(str).str.strip().eq("").any():
            raise DataValidationError(f"Missing bar {column}")
    if bars.Currency.nunique() != 1:
        raise DataValidationError("A snapshot must use exactly one currency")
    _times(bars.Available_At, "bar availability")
    if not bars.Available_At.ge(bars.Close_Time).all():
        raise DataValidationError("A complete daily bar cannot be available before its close")
    _finite(bars, ["Open", "High", "Low", "Close", "Volume"], "OHLCV")
    if (bars[["Open", "High", "Low", "Close"]] <= 0).any().any() or bars.Volume.lt(0).any():
        raise DataValidationError("Prices must be positive and volume nonnegative")
    tolerance = bars.High.abs().clip(lower=1) * 1e-10
    if (bars.High + tolerance < bars[["Open", "Close", "Low"]].max(axis=1)).any():
        raise DataValidationError("High is below another OHLC price")
    if (bars.Low - tolerance > bars[["Open", "Close", "High"]].min(axis=1)).any():
        raise DataValidationError("Low is above another OHLC price")
    if "Adjusted_Close" in bars:
        _finite(bars, ["Adjusted_Close"], "adjusted close")
        if bars.Adjusted_Close.le(0).any():
            raise DataValidationError("Adjusted close must be positive")

    missing = sorted(set(schedule.Session) - set(bars.Session))
    unexpected = sorted(set(bars.Session) - set(schedule.Session))
    warnings = []
    for dates, setting, name in ((missing, policy.missing_sessions, "missing_sessions"),
                                  (unexpected, policy.unexpected_sessions, "unexpected_sessions")):
        if dates and setting == "reject":
            raise DataValidationError(f"{name} detected, first dates {dates[:5]}")
        if dates:
            warnings.append({"Code": name, "Count": len(dates), "Dates": dates})
    common = bars.merge(schedule, on="Session", suffixes=("", "_Expected"))
    if not (common.Open_Time.eq(common.Open_Time_Expected)
            & common.Close_Time.eq(common.Close_Time_Expected)).all():
        raise DataValidationError("Bar timestamps disagree with the persisted schedule")
    if actions.Action_ID.isna().any() or actions.Action_ID.duplicated().any():
        raise DataValidationError("Action IDs must be present and unique")
    instrument = bars.Instrument_ID.iloc[0]
    for action in actions.itertuples(index=False):
        if action.Instrument_ID != instrument or action.Ex_Date not in set(bars.Session):
            raise DataValidationError("An action must match the instrument and a market bar")
        if action.Action_Type not in {"split", "distribution"}:
            raise DataValidationError("Unsupported action type")
        event_open = bars.loc[bars.Session.eq(action.Ex_Date), "Open_Time"].iloc[0]
        if utc_timestamp(action.Effective_Time) != event_open:
            raise DataValidationError("Daily corporate actions must take effect at the ex-date open")
        utc_timestamp(action.Available_At)
        if pd.isna(action.Source_ID) or not str(action.Source_ID).strip():
            raise DataValidationError("Actions require a source identifier")
        if action.Action_Type == "split":
            if not np.isfinite(float(action.Split_Ratio)) or float(action.Split_Ratio) <= 0:
                raise DataValidationError("Split ratios must be positive")
        else:
            if not np.isfinite(float(action.Amount_Per_Share)) or float(action.Amount_Per_Share) < 0:
                raise DataValidationError("Distributions must be finite and nonnegative")
            if pd.isna(action.Pay_Date):
                if policy.dividend_payment == "reject":
                    raise DataValidationError("Missing dividend payment date requires an explicit approximation")
            elif str(action.Pay_Date) < action.Ex_Date:
                raise DataValidationError("Payment precedes the ex-date")
    _dates(actions.Ex_Date, "ex-date")
    _dates(actions.Pay_Date, "payment date", nullable=True)
    unknown_payments = int(actions.loc[actions.Action_Type.eq("distribution"), "Pay_Date"].isna().sum())
    if unknown_payments:
        warnings.append({"Code": "ex_date_credit_approximation", "Count": unknown_payments})

    observations = tables["cash_observations"]
    if observations.Observation_ID.isna().any() or observations.Observation_ID.duplicated().any():
        raise DataValidationError("Cash observation IDs must be unique and present")
    _dates(observations.Rate_Date, "rate date")
    _times(observations.Available_At, "cash availability")
    _finite(observations, ["Quoted_Rate"], "cash quotes")
    for row in observations.itertuples(index=False):
        if pd.isna(row.Source_ID) or not str(row.Source_ID).strip():
            raise DataValidationError("Cash observations require a source identifier")
        if pd.Timestamp(row.Rate_Date).date() > utc_timestamp(row.Available_At).date():
            raise DataValidationError("Cash observation is available before its reference date")
        if row.Rate_Units not in {"percent", "decimal"} or row.Day_Count not in {"ACT/360", "ACT/365"}:
            raise DataValidationError("Unsupported rate units or day count")
        if row.Compounding not in {"simple", "annual_effective", "continuous"}:
            raise DataValidationError("Unsupported compounding convention")
    intervals = tables["cash_intervals"]
    _times(intervals.Start_Time, "cash interval start")
    _times(intervals.End_Time, "cash interval end")
    _finite(intervals, ["Credited_Return", "Reference_Return"], "cash returns")
    if len(intervals) != len(bars) - 1:
        raise DataValidationError("Cash intervals must cover every observed open-to-open interval")
    if not intervals.Start_Time.reset_index(drop=True).equals(bars.Open_Time.iloc[:-1].reset_index(drop=True)):
        raise DataValidationError("Cash start times do not match market opens")
    if not intervals.End_Time.reset_index(drop=True).equals(bars.Open_Time.iloc[1:].reset_index(drop=True)):
        raise DataValidationError("Cash end times do not match market opens")
    if (intervals[["Credited_Return", "Reference_Return"]] <= -1).any().any():
        raise DataValidationError("A cash return cannot destroy more than all capital")
    for row in intervals.itertuples(index=False):
        if not row.Availability_Policy or pd.isna(row.Observation_IDs):
            raise DataValidationError("Cash intervals require an availability policy and observation IDs")
        if row.Observation_IDs == "constant_zero":
            if row.Credited_Return != 0 or row.Reference_Return != 0:
                raise DataValidationError("A constant-zero cash interval must have zero returns")
            continue
        selected = observations.loc[observations.Observation_ID.eq(row.Observation_IDs)]
        if len(selected) != 1 or selected.Available_At.iloc[0] > row.Start_Time:
            raise DataValidationError("Cash interval uses an absent or unavailable observation")

    zero = bars.Volume.eq(0)
    flat = bars.High.eq(bars.Low)
    for code, mask in (("zero_volume", zero), ("flat_ohlc", flat)):
        if mask.any():
            warnings.append({"Code": code, "Count": int(mask.sum()),
                             "First_Session": bars.loc[mask, "Session"].iloc[0],
                             "Last_Session": bars.loc[mask, "Session"].iloc[-1]})
    decades = pd.to_datetime(bars.Session).dt.year // 10 * 10
    quality = pd.DataFrame({"Decade": decades, "Zero_Volume": zero, "Flat_OHLC": flat})
    by_decade = quality.groupby("Decade").agg(Rows=("Decade", "size"),
                                              Zero_Volume=("Zero_Volume", "sum"),
                                              Flat_OHLC=("Flat_OHLC", "sum"))
    return {"Rows": len(bars), "Start": bars.Session.iloc[0], "End": bars.Session.iloc[-1],
            "Missing_Sessions": len(missing), "Unexpected_Sessions": len(unexpected),
            "Warnings": warnings, "By_Decade": by_decade.reset_index().to_dict("records")}

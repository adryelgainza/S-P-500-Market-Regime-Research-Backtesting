"""Explicit refresh/import adapters. Offline loading never invokes these."""

from datetime import datetime, timezone
import importlib.metadata
import io
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from .calendar import exchange_schedule, normalize_bars
from .cash import CashPolicy
from .contracts import (ACTION_COLUMNS, OBSERVATION_COLUMNS, DataValidationError,
                        ValidationPolicy, empty_table, utc_timestamp)
from .reconciliation import restore_as_traded_history
from .storage import write_snapshot


FRED_DFF_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
DFF_AVAILABILITY = "rate_date_plus_3_US_federal_business_days_at_2359_New_York_proxy"


def _now():
    return datetime.now(timezone.utc).isoformat()


def import_index_snapshot(csv_path="sp500_ohlcv.csv", *, root="datasets"):
    path = Path(csv_path)
    raw = path.read_bytes()
    original = pd.read_csv(io.BytesIO(raw))
    schedule = exchange_schedule(original.Date.iloc[0], original.Date.iloc[-1])
    bars = normalize_bars(original, schedule, instrument="^GSPC", source="local_yahoo_index_csv")
    metadata = {
        "Track": "historical_price_index", "Instrument_ID": "^GSPC", "Calendar": "NYSE",
        "Exchange_Timezone": "America/New_York", "Source": "existing_local_Yahoo_Finance_snapshot",
        "Request": {"Import_File": path.name, "Original_Download": "unknown exact retrieval time"},
        "Retrieved_UTC": None, "Imported_UTC": _now(),
        "Price_Basis": "original_price_index_levels_without_distributions",
        "Availability_Assumptions": [
            "Original retrieval time and historical vintages are unavailable.",
            "Complete daily bars are assumed available one minute after the scheduled close.",
            "Historical NYSE calendar discrepancies are reported, including omitted early Saturday sessions.",
            "Zero-volume and identical-OHLC early records are retained as source-quality limitations.",
        ],
        "Redistribution": "Yahoo Finance source terms apply; no redistribution permission is asserted.",
        "Cash_Assumption": "Explicit constant-zero historical proxy, matching the legacy study.",
    }
    return write_snapshot(bars=bars, schedule=schedule, metadata=metadata, root=root,
                          cash_policy=CashPolicy(mode="constant_zero", availability_policy="legacy_zero_cash"),
                          validation_policy=ValidationPolicy(missing_sessions="report"),
                          raw_files={"sp500_ohlcv.csv": raw})


def yahoo_history_tables(history, *, retrieved_utc, start=None, end=None):
    """Normalize a complete Yahoo SPY response, then select a date range.

    The caller must supply period=max through the current provider split basis.
    Dividend payment/announcement dates are absent from this provider response.
    """
    required = {"Open", "High", "Low", "Close", "Volume", "Adj Close", "Dividends", "Stock Splits"}
    if history.empty or not required.issubset(history.columns):
        raise DataValidationError("Incomplete Yahoo history or missing adjusted prices/actions")
    if not isinstance(history.index, pd.DatetimeIndex) or history.index.tz is None:
        raise DataValidationError("Yahoo history requires exchange-local, timezone-aware dates")
    if history.index.duplicated().any() or not history.index.is_monotonic_increasing:
        raise DataValidationError("Yahoo history must be sorted and unique")
    raw = restore_as_traded_history(history)
    raw["Session"] = raw.index.tz_convert("America/New_York").strftime("%Y-%m-%d")
    first_available = raw.Session.iloc[0]
    if start is not None:
        raw = raw.loc[raw.Session.ge(str(pd.Timestamp(start).date()))]
    if end is not None:
        raw = raw.loc[raw.Session.lt(str(pd.Timestamp(end).date()))]
    if raw.empty:
        raise DataValidationError("No bars in the requested date range")
    expected_start = max(first_available, str(pd.Timestamp(start).date())) if start is not None else first_available
    retrieval_date = utc_timestamp(retrieved_utc).tz_convert("America/New_York").date()
    expected_end = str(retrieval_date)
    if end is not None:
        expected_end = min(expected_end, str((pd.Timestamp(end) - pd.Timedelta(days=1)).date()))
    schedule = exchange_schedule(expected_start, expected_end)
    # Exclude an in-progress daily bar rather than silently treating it as final.
    schedule = schedule.loc[(schedule.Close_Time + pd.Timedelta(minutes=1))
                            .le(utc_timestamp(retrieved_utc))].reset_index(drop=True)
    raw = raw.loc[raw.Session.isin(schedule.Session)]
    if raw.empty:
        raise DataValidationError("No completed market sessions")
    bars = normalize_bars(raw[["Session", "Open", "High", "Low", "Close", "Volume",
                               "Adj Close", "Future_Split_Factor"]].rename(columns={"Adj Close": "Adjusted_Close"}),
                          schedule, instrument="SPY", source="yahoo_finance_yfinance")
    action_rows = []
    event_opens = dict(zip(bars.Session, bars.Open_Time))
    for row in raw.itertuples(index=False, name=None):
        values = dict(zip(raw.columns, row))
        session = values["Session"]
        event_open = event_opens[session]
        for action_type, amount in (("split", values["Stock Splits"]),
                                    ("distribution", values["Dividends"] + values.get("Capital Gains", 0))):
            if not np.isfinite(amount) or amount < 0:
                raise DataValidationError("Invalid provider action amount")
            if amount == 0:
                continue
            action_rows.append(dict(Instrument_ID="SPY", Action_ID=f"SPY_{session}_{action_type}",
                                    Action_Type=action_type, Effective_Time=event_open, Ex_Date=session,
                                    Pay_Date=pd.NA, Amount_Per_Share=amount if action_type == "distribution" else np.nan,
                                    Split_Ratio=amount if action_type == "split" else np.nan,
                                    Available_At=event_open, Source_ID="yahoo_finance_yfinance"))
    return bars, pd.DataFrame(action_rows, columns=ACTION_COLUMNS), schedule


def parse_fred_dff(raw: bytes):
    """Use a conservative availability proxy, never an invented vintage timestamp.

    Three federal business days at 23:59 New York is a documented modeling
    lag, not a reconstruction of the historical release calendar. The rate is
    held fixed at each interval's open until the following open.
    """
    frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    if len(frame.columns) != 2 or frame.columns[1] != "DFF":
        raise DataValidationError("Expected FRED DFF CSV")
    frame.columns = ["Rate_Date", "Quoted_Rate"]
    missing = frame.Quoted_Rate.isin(["", "."])
    excluded_dates = frame.loc[missing, "Rate_Date"].tolist()
    frame = frame.loc[~missing].copy()
    if frame.empty:
        raise DataValidationError("No FRED cash-rate observations")
    dates = pd.to_datetime(frame.Rate_Date, errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise DataValidationError("Cash observation dates must be unique and sorted")
    lag = CustomBusinessDay(3, calendar=USFederalHolidayCalendar())
    available = [(date + lag + pd.Timedelta(hours=23, minutes=59))
                 .tz_localize("America/New_York").tz_convert("UTC") for date in dates]
    observations = pd.DataFrame({
        "Observation_ID": "FRED_DFF_" + dates.dt.strftime("%Y-%m-%d"),
        "Rate_Date": dates.dt.strftime("%Y-%m-%d"), "Available_At": available,
        "Quoted_Rate": pd.to_numeric(frame.Quoted_Rate, errors="raise"),
        "Rate_Units": "percent", "Day_Count": "ACT/360", "Compounding": "simple",
        "Source_ID": "FRED_DFF",
    }).reset_index(drop=True)
    return observations[OBSERVATION_COLUMNS], excluded_dates


def refresh_spy_snapshot(*, root="datasets", start=None, end=None, credited_spread_bps=0.0):
    """Explicit network operation. No existing snapshot or legacy CSV is replaced."""
    import requests
    import yfinance as yf

    cache_dir = Path(root) / ".provider_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    request = dict(period="max", interval="1d", auto_adjust=False, back_adjust=False,
                   actions=True, repair=False, keepna=True, rounding=False,
                   raise_errors=True, timeout=30)
    history = yf.Ticker("SPY").history(**request)
    market_retrieved = _now()
    bars, actions, schedule = yahoo_history_tables(history, retrieved_utc=market_retrieved, start=start, end=end)
    # Include a buffer so the earliest market interval has a mature quote.
    cash_start = (pd.Timestamp(bars.Session.iloc[0]) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    cash_request = {"id": "DFF", "cosd": cash_start, "coed": bars.Session.iloc[-1]}
    response = requests.get(FRED_DFF_URL, params=cash_request, timeout=45)
    response.raise_for_status()
    cash_retrieved = _now()
    observations, excluded_cash_dates = parse_fred_dff(response.content)
    metadata = {
        "Track": "investable_etf", "Instrument_ID": "SPY", "Calendar": "NYSE",
        "Exchange_Timezone": "America/New_York", "Source": "Yahoo_Finance_and_FRED_DFF",
        "Request": {"Yahoo": request, "Ticker": "SPY", "Select_Start": start,
                    "Select_End_Exclusive": end, "FRED_URL": FRED_DFF_URL, "FRED_Params": cash_request},
        "Retrieved_UTC": market_retrieved, "Cash_Retrieved_UTC": cash_retrieved, "Imported_UTC": _now(),
        "Price_Basis": "as_traded_OHLC_reconstructed_from_complete_provider_split_history",
        "Availability_Assumptions": [
            "Final revised history is used; point-in-time vintages are unavailable.",
            "Daily OHLCV becomes available one minute after the scheduled close.",
            "Split-normalized OHLC and distributions are restored using future split factors before date filtering.",
            "Provider volume is assumed split-normalized; original response is preserved for audit.",
            "Actions are assumed known at the ex-date open; announcement and payment dates are unavailable.",
            "Missing payment dates use the explicitly declared ex-date-credit approximation.",
            DFF_AVAILABILITY,
        ],
        "Cash_Assumption": "Hypothetical lagged DFF-linked cash proxy, not an attainable broker or fund yield.",
        "Excluded_Cash_Observation_Dates": excluded_cash_dates,
        "Redistribution": "Yahoo Finance and FRED source terms apply; no redistribution permission is asserted.",
        "Provider_Versions": {name: importlib.metadata.version(name) for name in ["yfinance", "requests"]},
        "Source_References": [
            "https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html",
            "https://in.help.yahoo.com/kb/adjusted-close-sln28256.html",
            "https://fred.stlouisfed.org/series/DFF",
            "https://www.federalreserve.gov/releases/h15/",
        ],
    }
    return write_snapshot(bars=bars, actions=actions, schedule=schedule, metadata=metadata,
                          cash_observations=observations, root=root,
                          cash_policy=CashPolicy(availability_policy=DFF_AVAILABILITY,
                                                 credited_spread_bps=credited_spread_bps),
                          validation_policy=ValidationPolicy(dividend_payment="ex_date_credit_approximation"),
                          raw_files={"yahoo_spy_history.csv": history.to_csv(lineterminator="\n").encode("utf-8"),
                                     "fred_dff.csv": response.content})

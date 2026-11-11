"""Exchange sessions are persisted in snapshots so replay stays offline."""

import pandas as pd
import pandas_market_calendars as mcal

from .contracts import DataValidationError


def exchange_schedule(start, end, calendar_name="NYSE") -> pd.DataFrame:
    calendar = mcal.get_calendar(calendar_name)
    schedule = calendar.schedule(start_date=start, end_date=end, tz="UTC")
    if schedule.empty:
        raise DataValidationError("The requested range has no exchange sessions")
    return pd.DataFrame({
        "Session": schedule.index.strftime("%Y-%m-%d"),
        "Open_Time": schedule.market_open.to_numpy(),
        "Close_Time": schedule.market_close.to_numpy(),
    })


def normalize_bars(frame, schedule, *, instrument, source, currency="USD",
                   availability_delay_minutes=1) -> pd.DataFrame:
    """Attach exchange timestamps without filling or dropping missing prices."""
    if availability_delay_minutes < 0:
        raise DataValidationError("A price availability delay cannot be negative")
    frame = frame.copy()
    if "Session" not in frame:
        if "Date" not in frame:
            raise DataValidationError("Bars require Session or Date")
        frame = frame.rename(columns={"Date": "Session"})
    frame["Session"] = pd.to_datetime(frame.Session, errors="raise").dt.strftime("%Y-%m-%d")
    if frame.Session.duplicated().any():
        raise DataValidationError("Duplicate market sessions")
    if not frame.Session.is_monotonic_increasing:
        raise DataValidationError("Market sessions must be sorted")
    frame = frame.merge(schedule, on="Session", how="left", validate="one_to_one")
    if frame.Open_Time.isna().any():
        raise DataValidationError("Bars contain dates outside the supplied exchange schedule")
    frame["Available_At"] = frame.Close_Time + pd.Timedelta(minutes=availability_delay_minutes)
    frame["Instrument_ID"] = instrument
    frame["Source_ID"] = source
    frame["Currency"] = currency
    return frame

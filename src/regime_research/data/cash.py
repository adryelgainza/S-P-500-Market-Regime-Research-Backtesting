"""Causal cash proxies with explicit quotation, publication, and accrual rules."""

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from .contracts import DataValidationError, INTERVAL_COLUMNS, utc_timestamp


@dataclass(frozen=True)
class CashPolicy:
    mode: str = "lagged_rate_proxy"
    availability_policy: str = "latest_available_at_interval_start"
    maximum_quote_age_days: int = 10
    credited_spread_bps: float = 0.0
    accrual_timezone: str = "America/New_York"

    def __post_init__(self):
        if self.mode not in {"lagged_rate_proxy", "constant_zero"}:
            raise ValueError("Unsupported cash mode")
        if (not isinstance(self.maximum_quote_age_days, int)
                or isinstance(self.maximum_quote_age_days, bool)
                or self.maximum_quote_age_days < 0 or not np.isfinite(self.credited_spread_bps)):
            raise ValueError("Invalid cash age or spread")
        if not self.availability_policy:
            raise ValueError("Cash availability policy must be named")

    def to_dict(self):
        return asdict(self)


def accrue_quoted_rate(quoted_rate, start, end, *, units="percent", day_count="ACT/360",
                       compounding="simple", spread_bps=0.0,
                       accrual_timezone="America/New_York") -> float:
    start, end = utc_timestamp(start), utc_timestamp(end)
    if end <= start:
        raise DataValidationError("Cash accrual must have positive duration")
    if units not in {"percent", "decimal"} or day_count not in {"ACT/360", "ACT/365"}:
        raise DataValidationError("Unsupported rate units or day-count basis")
    rate = float(quoted_rate) / (100 if units == "percent" else 1) - spread_bps / 10000
    if not math.isfinite(rate):
        raise DataValidationError("Cash rates must be finite")
    # Money-market day counts use local calendar days, including DST weekends.
    local_start = start.tz_convert(accrual_timezone).tz_localize(None)
    local_end = end.tz_convert(accrual_timezone).tz_localize(None)
    days = (local_end - local_start).total_seconds() / 86400
    year_fraction = days / (360 if day_count == "ACT/360" else 365)
    if compounding == "simple":
        result = rate * year_fraction
    elif compounding == "annual_effective":
        if rate <= -1:
            raise DataValidationError("An effective annual rate must exceed -100%")
        result = math.expm1(math.log1p(rate) * year_fraction)
    elif compounding == "continuous":
        result = math.expm1(rate * year_fraction)
    else:
        raise DataValidationError("Unsupported compounding convention")
    if not math.isfinite(result) or result <= -1:
        raise DataValidationError("Invalid accrued cash return")
    return result


def build_cash_intervals(bars, observations, policy: CashPolicy) -> pd.DataFrame:
    if bars.empty:
        raise DataValidationError("Cash intervals require market bars")
    rows = []
    quotes = observations.sort_values(["Available_At", "Rate_Date", "Observation_ID"])
    if quotes.Observation_ID.duplicated().any():
        raise DataValidationError("Duplicate cash observation IDs")
    available_times = pd.DatetimeIndex([utc_timestamp(v) for v in quotes.Available_At])
    # A late publication of an older observation must not displace a newer
    # already-available reference date. Ties use the most recent publication.
    chosen_after_publication = []
    best = None
    for index in range(len(quotes)):
        if best is None or quotes.Rate_Date.iloc[index] >= quotes.Rate_Date.iloc[best]:
            best = index
        chosen_after_publication.append(best)
    for start, end in zip(bars.Open_Time.iloc[:-1], bars.Open_Time.iloc[1:]):
        start, end = utc_timestamp(start), utc_timestamp(end)
        if policy.mode == "constant_zero":
            reference = credited = 0.0
            observation_id = "constant_zero"
        else:
            index = available_times.searchsorted(start, side="right") - 1
            if index < 0:
                raise DataValidationError(f"No cash quote available by {start}")
            quote = quotes.iloc[chosen_after_publication[index]]
            rate_date = pd.Timestamp(quote.Rate_Date)
            age = (start.tz_convert(policy.accrual_timezone).tz_localize(None).normalize() - rate_date).days
            if age < 0 or age > policy.maximum_quote_age_days:
                raise DataValidationError(f"Cash quote is future-dated or stale at {start}")
            arguments = dict(units=quote.Rate_Units, day_count=quote.Day_Count,
                             compounding=quote.Compounding, accrual_timezone=policy.accrual_timezone)
            reference = accrue_quoted_rate(quote.Quoted_Rate, start, end, **arguments)
            credited = accrue_quoted_rate(quote.Quoted_Rate, start, end,
                                          spread_bps=policy.credited_spread_bps, **arguments)
            observation_id = quote.Observation_ID
        rows.append(dict(Start_Time=start, End_Time=end, Reference_Return=reference,
                         Credited_Return=credited, Observation_IDs=observation_id,
                         Availability_Policy=policy.availability_policy))
    return pd.DataFrame(rows, columns=INTERVAL_COLUMNS)

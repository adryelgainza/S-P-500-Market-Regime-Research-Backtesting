"""Execution-aligned target construction with explicit maturation timestamps."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    horizon: int = 10
    one_way_cost: float = .0005
    volatility_multiplier: float = .5
    volatility_window: int = 20

    def __post_init__(self):
        if self.horizon not in {5, 10, 20} or self.volatility_window < 2:
            raise ValueError("Unsupported target horizon or volatility window")
        if not 0 <= self.one_way_cost < 1 or not 0 <= self.volatility_multiplier <= 10:
            raise ValueError("Invalid target cost or volatility multiplier")


def build_labels(bars, config=LabelConfig()):
    """Entry at t+1, exit at t+h+1, with the exit-day low excluded."""
    if not {"Open", "Low", "Close", "Open_Time", "Available_At"}.issubset(bars):
        raise ValueError("Label construction requires executable bars and availability timestamps")
    horizon = config.horizon
    entry = bars.Open.shift(-1)
    forward = bars.Open.shift(-(horizon+1))/entry-1
    daily_volatility = bars.Close.pct_change(fill_method=None).rolling(
        config.volatility_window, min_periods=config.volatility_window).std(ddof=1)
    threshold = 2*config.one_way_cost+config.volatility_multiplier*daily_volatility*np.sqrt(horizon)
    path = pd.concat([*[bars.Low.shift(-day) for day in range(1, horizon+1)],
                      bars.Open.shift(-(horizon+1))], axis="columns")
    adverse = path.min(axis="columns", skipna=False)/entry-1
    valid = forward.notna() & threshold.notna() & adverse.notna()
    regime = pd.Series(np.select([forward < -threshold, forward > threshold], ["Bearish", "Bullish"],
                                 default="Neutral"), index=bars.index, dtype="string").where(valid)
    return pd.DataFrame({"Decision_Time": bars.Available_At, "Execution_Time": bars.Open_Time.shift(-1),
        "Return_End_Time": bars.Open_Time.shift(-2), "Label_End_Time": bars.Open_Time.shift(-(horizon+1)),
        "Forward_Return": forward, "Threshold": threshold, "Three_Class": regime,
        "Binary_Downside": forward.lt(-threshold).astype(float).where(valid),
        "Binary_Drawdown": adverse.lt(-threshold).astype(float).where(valid),
        "Return_Regression": forward.where(valid), "Downside_Risk": (-adverse).clip(lower=0).where(valid)}, index=bars.index)

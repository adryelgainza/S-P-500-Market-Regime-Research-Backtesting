"""Causal score-to-weight mappings extracted from the registered overlay rules."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AllocationConfig:
    strategy_id: str
    core: str
    mapping: str
    half_risk_percentile: float
    risk_off_percentile: float
    smoothing_days: int
    score_lookback_days: int
    hysteresis: float
    minimum_hold_days: int
    target_volatility: float | None
    volatility_window: int
    score_minimum_days: int = 252

    def __post_init__(self):
        if self.core not in {"Always_Long", "SMA_50_200"} or self.mapping not in {"Long_Flat", "Step", "Continuous"}:
            raise ValueError("Unsupported allocation core or mapping")
        if not 0 <= self.half_risk_percentile < self.risk_off_percentile <= 1 or not 0 <= self.hysteresis < 1:
            raise ValueError("Invalid score thresholds or hysteresis")
        if (self.smoothing_days < 1 or self.minimum_hold_days < 0
                or not 1 <= self.score_minimum_days <= self.score_lookback_days):
            raise ValueError("Invalid allocation warmup or hold count")
        if self.volatility_window not in {20, 60} or (self.target_volatility is not None and self.target_volatility <= 0):
            raise ValueError("Invalid volatility overlay")

    @classmethod
    def from_spec(cls, spec):
        return cls(**{name.lower(): value for name, value in spec.items() if name != "Complexity"})


def step_multiplier(percentiles, config):
    current, held_days = 1., config.minimum_hold_days
    multipliers = []
    for percentile in percentiles.fillna(.5).to_numpy(float):
        proposed = 1.
        if percentile >= config.risk_off_percentile:
            proposed = 0.
        elif config.mapping != "Long_Flat" and percentile >= config.half_risk_percentile:
            proposed = .5
        if proposed < current:
            current, held_days = proposed, 0
        elif proposed > current:
            reentry_cut = config.risk_off_percentile if current == 0 else config.half_risk_percentile
            if held_days >= config.minimum_hold_days and percentile < reentry_cut-config.hysteresis:
                current, held_days = proposed, 0
            else:
                held_days += 1
        else:
            held_days += 1
        multipliers.append(current)
    return np.asarray(multipliers)


def allocate(scores, market, config):
    """Build weights on the complete score history before applying an evaluation mask."""
    if scores.empty or scores.Decision_Time.duplicated().any() or not scores.Decision_Time.is_monotonic_increasing:
        raise ValueError("Allocation requires unique chronological scores")
    if not np.isfinite(scores.Risk_Score).all():
        raise ValueError("Risk scores must be finite")
    result = scores.copy().reset_index(drop=True)
    smoothed = result.Risk_Score.ewm(span=config.smoothing_days, adjust=False, min_periods=1).mean()
    result["Risk_Percentile"] = smoothed.rolling(config.score_lookback_days,
        min_periods=config.score_minimum_days).rank(pct=True)
    percentiles = result.Risk_Percentile.fillna(.5)
    if config.mapping == "Continuous":
        desired = ((config.risk_off_percentile-percentiles)/(config.risk_off_percentile-config.half_risk_percentile)).clip(0, 1)
        current, multipliers = 1., []
        for target in desired:
            current += float(np.clip(target-current, -.10, .10))
            multipliers.append(current)
        multiplier = np.asarray(multipliers)
    else:
        multiplier = step_multiplier(percentiles, config)
    aligned = market.loc[result.Row_ID]
    core = np.ones(len(result))
    if config.core == "SMA_50_200":
        core = aligned.SMA_50_200.fillna(0).to_numpy(float)
    scale = np.ones(len(result))
    if config.target_volatility is not None:
        volatility = aligned[f"Execution_Volatility_{config.volatility_window}"]
        scale = ((config.target_volatility/volatility).clip(0, 1)/.10).round()*.10
        scale = scale.fillna(0).clip(0, 1).to_numpy()
    result["Target_Weight"] = np.clip(core*multiplier*scale, 0, 1)
    result["Strategy_ID"] = config.strategy_id
    result["Rule_Version"] = "package-overlay-v2"
    result["Rebalance_Policy"] = "signal_change_only"
    return result


def market_overlays(bars):
    changes = bars.Close.pct_change(fill_method=None)
    short, long = bars.Close.rolling(50, min_periods=50).mean(), bars.Close.rolling(200, min_periods=200).mean()
    return pd.DataFrame({"SMA_50_200": short.gt(long).astype(float).where(long.notna()),
        **{f"Execution_Volatility_{window}": changes.ewm(span=window, adjust=False, min_periods=window).std()*np.sqrt(252)
           for window in [20, 60]}}, index=bars.index)

"""Compare explicit cash-action returns with provider adjustment conventions."""

import numpy as np
import pandas as pd

from .contracts import DataValidationError


def restore_as_traded_history(history: pd.DataFrame) -> pd.DataFrame:
    """Undo split normalization using the complete split history in the response.

    Yahoo's history must extend through the provider's current split basis. The
    downloader requests period=max before any date filtering. Adjusted close is
    retained in provider units and is never treated as an executable price.
    """
    frame = history.copy()
    if "Stock Splits" not in frame or "Dividends" not in frame:
        raise DataValidationError("Provider response is missing corporate-action columns")
    splits = pd.to_numeric(frame["Stock Splits"], errors="raise")
    if not np.isfinite(splits).all() or splits.lt(0).any():
        raise DataValidationError("Invalid provider split ratios")
    ratios = splits.where(splits.ne(0), 1.0)
    future_factor = ratios.iloc[::-1].cumprod().iloc[::-1] / ratios
    for column in ("Open", "High", "Low", "Close"):
        frame[column] = frame[column] * future_factor
    frame["Volume"] = frame.Volume / future_factor
    for column in ("Dividends", "Capital Gains"):
        if column in frame:
            frame[column] = frame[column] * future_factor
    frame["Future_Split_Factor"] = future_factor
    return frame


def reconcile_total_returns(bars, actions) -> pd.DataFrame:
    """Close-to-close diagnostics, separate from the next-open trading ledger.

    Cash_Total_Return credits each distribution at the ex-date close and
    reinvests there. Expected_Backward_Adjusted_Return uses the provider's
    backward dividend multiplier (1 - dividend / previous close). These are
    different reinvestment conventions, so both the residual and gap are saved.
    """
    if "Adjusted_Close" not in bars:
        return pd.DataFrame(columns=["Session", "Cash_Total_Return", "Provider_Adjusted_Return",
                                     "Expected_Backward_Adjusted_Return", "Adjustment_Residual",
                                     "Cash_Convention_Gap"])
    splits = actions.loc[actions.Action_Type.eq("split")].groupby("Ex_Date").Split_Ratio.prod()
    distributions = actions.loc[actions.Action_Type.eq("distribution")].groupby("Ex_Date").Amount_Per_Share.sum()
    ratio = bars.Session.map(splits).fillna(1).astype(float)
    dividend = bars.Session.map(distributions).fillna(0).astype(float)
    previous = bars.Close.shift(1)
    denominator = previous - dividend * ratio
    if denominator.iloc[1:].le(0).any():
        raise DataValidationError("Distribution invalidates the backward adjustment multiplier")
    cash_return = (bars.Close + dividend) * ratio / previous - 1
    expected = bars.Close * ratio / denominator - 1
    provider = bars.Adjusted_Close.pct_change(fill_method=None)
    return pd.DataFrame({
        "Session": bars.Session.iloc[1:], "Cash_Total_Return": cash_return.iloc[1:],
        "Provider_Adjusted_Return": provider.iloc[1:],
        "Expected_Backward_Adjusted_Return": expected.iloc[1:],
        "Adjustment_Residual": (provider - expected).iloc[1:],
        "Cash_Convention_Gap": (cash_return - provider).iloc[1:],
    }).reset_index(drop=True)

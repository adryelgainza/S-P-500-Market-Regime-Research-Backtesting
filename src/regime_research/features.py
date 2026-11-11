"""Causal 98-feature construction extracted from the corrected notebook.

The indicator formulas and column order match archived cell caed8953.
"""

import numpy as np
import pandas as pd


def build_features(bars):
    """Return the original indicators on the supplied chronological bar index."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    if bars.empty or not required.issubset(bars):
        raise ValueError("Feature inputs require nonempty OHLCV bars")
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
        raise ValueError("Feature rows must have a unique ordered index")
    clean_df = bars
    close = clean_df["Close"].astype("float64")
    high = clean_df["High"].astype("float64")
    low = clean_df["Low"].astype("float64")
    positive_volume = clean_df["Volume"].where(clean_df["Volume"] > 0)
    return_1 = close.pct_change(fill_method=None)

    TRADING_WINDOWS = (5, 10, 20, 40, 60, 100, 200)
    CROSSOVER_PAIRS = ((10, 40), (20, 100), (40, 200))
    ANNUALIZATION_DAYS = 252

    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ADX_PERIOD = 14
    MOM_PERIOD = 10
    CHANNEL_PERIOD = 20
    BB_STD_MULTIPLIER = 2.0
    KC_ATR_MULTIPLIER = 2.0
    TTM_KC_MULTIPLIER = 1.5
    NR_PERIOD = 7
    ATR_PERIOD = 14
    ATR_CONTRACTION_LOOKBACK = 20


    def wilder_average(series, period):
        return series.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()


    def true_range(high_values, low_values, close_values):
        previous_close = close_values.shift(1)
        return pd.concat(
            [
                high_values - low_values,
                (high_values - previous_close).abs(),
                (low_values - previous_close).abs(),
            ],
            axis="columns",
        ).max(axis="columns")


    def rolling_linreg_endpoint(series, period):
        x = np.arange(period, dtype="float64")
        centered_x = x - x.mean()
        denominator = np.dot(centered_x, centered_x)

        def endpoint(values):
            slope = np.dot(centered_x, values) / denominator
            return values.mean() + slope * (period - 1 - x.mean())

        return series.rolling(period, min_periods=period).apply(
            endpoint,
            raw=True,
        )


    def rolling_linreg_slope(series, period):
        x = np.arange(period, dtype="float64")
        centered_x = x - x.mean()
        denominator = np.dot(centered_x, centered_x)

        def slope(values):
            return np.dot(centered_x, values) / denominator

        return series.rolling(period, min_periods=period).apply(
            slope,
            raw=True,
        )


    feature_values = {}
    sma_cache = {}
    ema_cache = {}

    # Seven window-feature families across seven trading windows.
    for window in TRADING_WINDOWS:
        sma = close.rolling(window, min_periods=window).mean()
        ema = close.ewm(
            span=window,
            adjust=False,
            min_periods=window,
        ).mean()
        sma_cache[window] = sma
        ema_cache[window] = ema

        # Absolute SMA/EMA levels are converted to relative distance from Close.
        feature_values[f"SMA_Rel_Close_{window}"] = sma / close - 1
        feature_values[f"EMA_Rel_Close_{window}"] = ema / close - 1
        feature_values[f"Return_{window}"] = close.pct_change(
            window,
            fill_method=None,
        )
        feature_values[f"Volatility_{window}"] = (
            return_1.rolling(window, min_periods=window).std(ddof=1)
            * np.sqrt(ANNUALIZATION_DAYS)
        )
        feature_values[f"Volume_Change_{window}"] = (
            positive_volume.pct_change(window, fill_method=None)
        )
        feature_values[f"MA_Diff_{window}"] = sma / ema - 1
        feature_values[f"Price_Range_{window}"] = (
            high.rolling(window, min_periods=window).max()
            / low.rolling(window, min_periods=window).min()
            - 1
        )

    window_feature_columns = [
        feature_name
        for window in TRADING_WINDOWS
        for feature_name in (
            f"SMA_Rel_Close_{window}",
            f"EMA_Rel_Close_{window}",
            f"Return_{window}",
            f"Volatility_{window}",
            f"Volume_Change_{window}",
            f"MA_Diff_{window}",
            f"Price_Range_{window}",
        )
    ]

    # RSI.
    close_delta = close.diff()
    average_gain = wilder_average(close_delta.clip(lower=0), RSI_PERIOD)
    average_loss = wilder_average(-close_delta.clip(upper=0), RSI_PERIOD)
    relative_strength = average_gain / average_loss
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask(average_loss.eq(0) & average_gain.gt(0), 100)
    rsi = rsi.mask(average_loss.eq(0) & average_gain.eq(0), 50)
    feature_values["RSI_14"] = rsi

    # MACD, normalized by Close.
    fast_ema = close.ewm(
        span=MACD_FAST,
        adjust=False,
        min_periods=MACD_FAST,
    ).mean()
    slow_ema = close.ewm(
        span=MACD_SLOW,
        adjust=False,
        min_periods=MACD_SLOW,
    ).mean()
    macd = fast_ema - slow_ema
    macd_signal = macd.ewm(
        span=MACD_SIGNAL,
        adjust=False,
        min_periods=MACD_SIGNAL,
    ).mean()
    feature_values["MACD_Rel_Close_12_26"] = macd / close
    feature_values["MACD_Signal_Rel_Close_9"] = macd_signal / close
    feature_values["MACD_Histogram_Rel_Close"] = (
        (macd - macd_signal) / close
    )

    # Directional movement, ATR, and normalized momentum.
    tr = true_range(high, low, close)
    atr = wilder_average(tr, ATR_PERIOD)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0),
        0.0,
    )
    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0),
        0.0,
    )
    plus_di = 100 * wilder_average(plus_dm, ADX_PERIOD) / atr
    minus_di = 100 * wilder_average(minus_dm, ADX_PERIOD) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    feature_values["ADX_14"] = wilder_average(dx, ADX_PERIOD)
    feature_values["Plus_DI_14"] = plus_di
    feature_values["Minus_DI_14"] = minus_di
    feature_values["MOM_Rel_Close_10"] = (
        close.diff(MOM_PERIOD) / close.shift(MOM_PERIOD)
    )

    # Bollinger Bands expressed relative to Close.
    channel_close = close.rolling(
        CHANNEL_PERIOD,
        min_periods=CHANNEL_PERIOD,
    )
    bb_middle = channel_close.mean()
    bb_standard_deviation = channel_close.std(ddof=0)
    bb_upper = bb_middle + BB_STD_MULTIPLIER * bb_standard_deviation
    bb_lower = bb_middle - BB_STD_MULTIPLIER * bb_standard_deviation
    feature_values["BB_Middle_Rel_Close_20"] = bb_middle / close - 1
    feature_values["BB_Upper_Rel_Close_20"] = bb_upper / close - 1
    feature_values["BB_Lower_Rel_Close_20"] = bb_lower / close - 1
    feature_values["BB_Width_20"] = (bb_upper - bb_lower) / bb_middle
    feature_values["BB_PctB_20"] = (
        (close - bb_lower) / (bb_upper - bb_lower)
    )

    # Keltner Channels expressed relative to Close.
    channel_atr = wilder_average(tr, CHANNEL_PERIOD)
    kc_middle = close.ewm(
        span=CHANNEL_PERIOD,
        adjust=False,
        min_periods=CHANNEL_PERIOD,
    ).mean()
    kc_upper = kc_middle + KC_ATR_MULTIPLIER * channel_atr
    kc_lower = kc_middle - KC_ATR_MULTIPLIER * channel_atr
    feature_values["KC_Middle_Rel_Close_20"] = kc_middle / close - 1
    feature_values["KC_Upper_Rel_Close_20"] = kc_upper / close - 1
    feature_values["KC_Lower_Rel_Close_20"] = kc_lower / close - 1
    feature_values["KC_Width_20"] = (kc_upper - kc_lower) / kc_middle
    feature_values["KC_Position_20"] = (
        (close - kc_lower) / (kc_upper - kc_lower)
    )

    # TTM squeeze, narrow-range, and ATR-contraction features.
    ttm_kc_upper = kc_middle + TTM_KC_MULTIPLIER * channel_atr
    ttm_kc_lower = kc_middle - TTM_KC_MULTIPLIER * channel_atr
    squeeze_ready = bb_upper.notna() & ttm_kc_upper.notna()
    squeeze_on = (bb_upper < ttm_kc_upper) & (bb_lower > ttm_kc_lower)
    feature_values["TTM_Squeeze_On"] = (
        squeeze_on.astype("float64").where(squeeze_ready)
    )
    highest_high = high.rolling(
        CHANNEL_PERIOD,
        min_periods=CHANNEL_PERIOD,
    ).max()
    lowest_low = low.rolling(
        CHANNEL_PERIOD,
        min_periods=CHANNEL_PERIOD,
    ).min()
    momentum_basis = ((highest_high + lowest_low) / 2 + bb_middle) / 2
    feature_values["TTM_Squeeze_Momentum_Rel_Close_20"] = (
        rolling_linreg_endpoint(
            close - momentum_basis,
            CHANNEL_PERIOD,
        )
        / close
    )
    intraday_range = high - low
    nr_minimum = intraday_range.rolling(
        NR_PERIOD,
        min_periods=NR_PERIOD,
    ).min()
    feature_values["NR7"] = (
        intraday_range.eq(nr_minimum)
        .astype("float64")
        .where(nr_minimum.notna())
    )
    feature_values["ATR_Rel_Close_14"] = atr / close
    atr_mean = atr.rolling(
        ATR_CONTRACTION_LOOKBACK,
        min_periods=ATR_CONTRACTION_LOOKBACK,
    ).mean()
    atr_contraction_ratio = atr / atr_mean
    feature_values["ATR_Contraction_Ratio"] = atr_contraction_ratio
    feature_values["ATR_Contraction"] = (
        atr_contraction_ratio.lt(1)
        .astype("float64")
        .where(atr_contraction_ratio.notna())
    )

    technical_feature_columns = [
        "RSI_14",
        "MACD_Rel_Close_12_26",
        "MACD_Signal_Rel_Close_9",
        "MACD_Histogram_Rel_Close",
        "ADX_14",
        "Plus_DI_14",
        "Minus_DI_14",
        "MOM_Rel_Close_10",
        "TTM_Squeeze_On",
        "TTM_Squeeze_Momentum_Rel_Close_20",
        "NR7",
        "ATR_Rel_Close_14",
        "ATR_Contraction_Ratio",
        "ATR_Contraction",
        "BB_Middle_Rel_Close_20",
        "BB_Upper_Rel_Close_20",
        "BB_Lower_Rel_Close_20",
        "BB_Width_20",
        "BB_PctB_20",
        "KC_Middle_Rel_Close_20",
        "KC_Upper_Rel_Close_20",
        "KC_Lower_Rel_Close_20",
        "KC_Width_20",
        "KC_Position_20",
    ]

    # Trend slopes and crossover spread/state/event features.
    log_close = np.log(close)
    for window in TRADING_WINDOWS:
        feature_values[f"Trend_Slope_{window}"] = rolling_linreg_slope(
            log_close,
            window,
        )

    for average_type, average_cache in (
        ("SMA", sma_cache),
        ("EMA", ema_cache),
    ):
        for short_window, long_window in CROSSOVER_PAIRS:
            spread = (
                average_cache[short_window]
                / average_cache[long_window]
                - 1
            )
            state = spread.gt(0).astype("float64").where(spread.notna())
            stem = (
                f"{average_type}_Cross_{{}}_"
                f"{short_window}_{long_window}"
            )
            feature_values[stem.format("Spread")] = spread
            feature_values[stem.format("State")] = state
            feature_values[stem.format("Event")] = state.diff()

    trend_crossover_columns = [
        *[
            f"Trend_Slope_{window}"
            for window in TRADING_WINDOWS
        ],
        *[
            f"{average_type}_Cross_{signal}_{short_window}_{long_window}"
            for average_type in ("SMA", "EMA")
            for short_window, long_window in CROSSOVER_PAIRS
            for signal in ("Spread", "State", "Event")
        ],
    ]

    MODEL_FEATURES = [
        *window_feature_columns,
        *technical_feature_columns,
        *trend_crossover_columns,
    ]

    candidate_feature_df = pd.DataFrame(
        feature_values,
        index=clean_df.index,
    )[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)

    return candidate_feature_df

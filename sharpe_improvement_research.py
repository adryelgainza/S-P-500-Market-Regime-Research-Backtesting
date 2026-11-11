"""Nested, purged Sharpe-improvement research for the project notebook.

This module is executed from ``sp500_market_regime_classification.ipynb``
after the advanced walk-forward section.  It deliberately relies on the
notebook's already-created, close-time feature table and next-open return
contract.  All model, target, and strategy selection for an outer block uses
only earlier inner folds.

The 2007-2026 history has already been inspected, so the outputs are
retrospective nested pseudo-out-of-sample evidence, not a fresh holdout.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from research_integrity import (
    downside_deviation,
    equity_drawdown,
    executed_notebook_digest,
    file_digest,
    forward_adverse_return,
    frame_digest,
    load_predictor_cache,
    realized_selection_rows,
    source_provenance,
    write_run_manifest,
)


REQUIRED_NOTEBOOK_GLOBALS = (
    "ADVANCED_FEATURES",
    "ADVANCED_LABEL_END_COLUMNS",
    "ADVANCED_ONE_WAY_COST",
    "ADVANCED_RETURN_COLUMNS",
    "ADVANCED_TARGET_COLUMNS",
    "ADVANCED_WALK_FORWARD_END",
    "ADVANCED_WALK_FORWARD_START",
    "ADVANCED_VALIDATION_START",
    "ANNUALIZATION_DAYS",
    "CLASS_NAMES",
    "OUTPUT_DIR",
    "RANDOM_STATE",
    "advanced_direct_position_backtest",
    "advanced_final_predictions",
    "advanced_locked_backtests",
    "advanced_refit_intervals",
    "advanced_table",
    "clean_df",
    "save_and_show",
)
missing_notebook_globals = [
    name for name in REQUIRED_NOTEBOOK_GLOBALS if name not in globals()
]
if missing_notebook_globals:
    raise RuntimeError(
        "Run the notebook through the advanced walk-forward section first. "
        f"Missing globals: {missing_notebook_globals}"
    )


# ---------------------------------------------------------------------------
# 1. Predeclared research protocol
# ---------------------------------------------------------------------------

SHARPE_RESEARCH_VERSION = "1.1.0"
SHARPE_SOURCE_PROVENANCE = source_provenance(Path.cwd())
SHARPE_EXECUTED_CODE_DIGEST = executed_notebook_digest(
    Path.cwd() / "sp500_market_regime_classification.ipynb",
    get_ipython().history_manager.input_hist_raw,
    "sharpe-experiments",
)
SHARPE_RESEARCH_STATUS = "Retrospective nested pseudo-out-of-sample"
SHARPE_RESEARCH_START = pd.Timestamp(ADVANCED_VALIDATION_START)
SHARPE_OUTER_START = pd.Timestamp(ADVANCED_WALK_FORWARD_START)
SHARPE_OUTER_END = pd.Timestamp(ADVANCED_WALK_FORWARD_END)
SHARPE_OUTER_YEARS = 2
SHARPE_INNER_FOLDS = 4
SHARPE_INNER_YEARS = 2
SHARPE_SCORE_LOOKBACK_DAYS = 5 * ANNUALIZATION_DAYS
SHARPE_SCORE_MIN_DAYS = ANNUALIZATION_DAYS
SHARPE_CALIBRATION_FRACTION = 0.15
SHARPE_MIN_BASE_ROWS = 756
SHARPE_MIN_CALIBRATION_ROWS = 126
SHARPE_SELECTED_FEATURES = 16
SHARPE_RISK_FREE_ANNUAL = 0.0
SHARPE_PRIMARY_BOOTSTRAP_REPS = 10_000
SHARPE_SENSITIVITY_BOOTSTRAP_REPS = 2_000
SHARPE_MULTIPLICITY_REPS = 2_000
SHARPE_PRIMARY_BLOCK_LENGTH = 20
SHARPE_TURNOVER_PENALTY = 0.02


def sharpe_feature_group(prefixes, exact=()):
    """Resolve a deterministic feature group from the existing feature set."""
    selected = [
        feature
        for feature in ADVANCED_FEATURES
        if feature in exact or any(token in feature for token in prefixes)
    ]
    if not selected:
        raise AssertionError("A predeclared feature group resolved to no columns")
    return list(dict.fromkeys(selected))


SHARPE_FEATURE_GROUPS = {
    "Full": list(ADVANCED_FEATURES),
    "Core_Risk": sharpe_feature_group(
        (
            "Return_",
            "Volatility_",
            "Price_Range_",
            "ATR_",
            "BB_Width",
            "KC_Width",
            "RSI_",
            "MOM_",
        )
    ),
    "Trend": sharpe_feature_group(
        (
            "SMA_",
            "EMA_",
            "MA_Diff_",
            "Trend_Slope_",
            "Cross_",
            "MACD_",
            "ADX_",
            "Plus_DI_",
            "Minus_DI_",
        )
    ),
}


# The registry is intentionally not a Cartesian product.  Each recipe tests a
# named hypothesis while keeping the number of trials visible and finite.
SHARPE_PREDICTOR_RECIPES = [
    {
        "Recipe_ID": "HGB_3Class_5d_W20_Full_Q",
        "Model": "HGB",
        "Target": "Three_Class",
        "Horizon": 5,
        "Training_Years": 20,
        "Feature_Group": "Full",
        "Refit_Frequency": "QS",
        "Complexity": 3,
    },
    *[
        {
            "Recipe_ID": f"HGB_Downside_{horizon}d_W20_Full_Q",
            "Model": "HGB",
            "Target": "Binary_Downside",
            "Horizon": horizon,
            "Training_Years": 20,
            "Feature_Group": "Full",
            "Refit_Frequency": "QS",
            "Complexity": 3,
        }
        for horizon in (5, 10, 20)
    ],
    *[
        {
            "Recipe_ID": f"HGB_Downside_10d_W{years}_Full_Q",
            "Model": "HGB",
            "Target": "Binary_Downside",
            "Horizon": 10,
            "Training_Years": years,
            "Feature_Group": "Full",
            "Refit_Frequency": "QS",
            "Complexity": 3,
        }
        for years in (5, 10)
    ],
    *[
        {
            "Recipe_ID": f"HGB_Downside_10d_W20_{group}_Q",
            "Model": "HGB",
            "Target": "Binary_Downside",
            "Horizon": 10,
            "Training_Years": 20,
            "Feature_Group": group,
            "Refit_Frequency": "QS",
            "Complexity": 3,
        }
        for group in ("Core_Risk", "Trend")
    ],
    {
        "Recipe_ID": "RF_Downside_10d_W20_Full_Q",
        "Model": "RF",
        "Target": "Binary_Downside",
        "Horizon": 10,
        "Training_Years": 20,
        "Feature_Group": "Full",
        "Refit_Frequency": "QS",
        "Complexity": 4,
    },
    {
        "Recipe_ID": "HGB_Drawdown_10d_W20_Full_Q",
        "Model": "HGB",
        "Target": "Binary_Drawdown",
        "Horizon": 10,
        "Training_Years": 20,
        "Feature_Group": "Full",
        "Refit_Frequency": "QS",
        "Complexity": 4,
    },
    {
        "Recipe_ID": "HGB_Return_10d_W20_Full_Q",
        "Model": "HGB",
        "Target": "Return_Regression",
        "Horizon": 10,
        "Training_Years": 20,
        "Feature_Group": "Full",
        "Refit_Frequency": "QS",
        "Complexity": 4,
    },
    {
        "Recipe_ID": "HGB_ReturnRisk_10d_W20_Full_Q",
        "Model": "HGB",
        "Target": "Return_Risk",
        "Horizon": 10,
        "Training_Years": 20,
        "Feature_Group": "Full",
        "Refit_Frequency": "QS",
        "Complexity": 5,
    },
    {
        "Recipe_ID": "HGB_Downside_10d_W20_Full_A",
        "Model": "HGB",
        "Target": "Binary_Downside",
        "Horizon": 10,
        "Training_Years": 20,
        "Feature_Group": "Full",
        "Refit_Frequency": "YS",
        "Complexity": 2,
    },
]


SHARPE_STRATEGY_SPECS = [
    {
        "Strategy_ID": "Rank_LongFlat_P80",
        "Core": "Always_Long",
        "Mapping": "Long_Flat",
        "Half_Risk_Percentile": 0.80,
        "Risk_Off_Percentile": 0.80,
        "Smoothing_Days": 5,
        "Score_Lookback_Days": 5 * ANNUALIZATION_DAYS,
        "Hysteresis": 0.10,
        "Minimum_Hold_Days": 5,
        "Target_Volatility": None,
        "Volatility_Window": 20,
        "Complexity": 2,
    },
    {
        "Strategy_ID": "AlwaysLong_Step_P80",
        "Core": "Always_Long",
        "Mapping": "Step",
        "Half_Risk_Percentile": 0.65,
        "Risk_Off_Percentile": 0.80,
        "Smoothing_Days": 5,
        "Score_Lookback_Days": 5 * ANNUALIZATION_DAYS,
        "Hysteresis": 0.10,
        "Minimum_Hold_Days": 5,
        "Target_Volatility": None,
        "Volatility_Window": 20,
        "Complexity": 2,
    },
    {
        "Strategy_ID": "AlwaysLong_Step_P80_L2",
        "Core": "Always_Long",
        "Mapping": "Step",
        "Half_Risk_Percentile": 0.65,
        "Risk_Off_Percentile": 0.80,
        "Smoothing_Days": 5,
        "Score_Lookback_Days": 2 * ANNUALIZATION_DAYS,
        "Hysteresis": 0.10,
        "Minimum_Hold_Days": 5,
        "Target_Volatility": None,
        "Volatility_Window": 20,
        "Complexity": 3,
    },
    {
        "Strategy_ID": "AlwaysLong_Step_P80_L10",
        "Core": "Always_Long",
        "Mapping": "Step",
        "Half_Risk_Percentile": 0.65,
        "Risk_Off_Percentile": 0.80,
        "Smoothing_Days": 5,
        "Score_Lookback_Days": 10 * ANNUALIZATION_DAYS,
        "Hysteresis": 0.10,
        "Minimum_Hold_Days": 5,
        "Target_Volatility": None,
        "Volatility_Window": 20,
        "Complexity": 3,
    },
    {
        "Strategy_ID": "AlwaysLong_Step_P90",
        "Core": "Always_Long",
        "Mapping": "Step",
        "Half_Risk_Percentile": 0.75,
        "Risk_Off_Percentile": 0.90,
        "Smoothing_Days": 10,
        "Score_Lookback_Days": 5 * ANNUALIZATION_DAYS,
        "Hysteresis": 0.10,
        "Minimum_Hold_Days": 5,
        "Target_Volatility": None,
        "Volatility_Window": 20,
        "Complexity": 2,
    },
    {
        "Strategy_ID": "SMA_Step_P80",
        "Core": "SMA_50_200",
        "Mapping": "Step",
        "Half_Risk_Percentile": 0.65,
        "Risk_Off_Percentile": 0.80,
        "Smoothing_Days": 5,
        "Score_Lookback_Days": 5 * ANNUALIZATION_DAYS,
        "Hysteresis": 0.10,
        "Minimum_Hold_Days": 5,
        "Target_Volatility": None,
        "Volatility_Window": 20,
        "Complexity": 3,
    },
    {
        "Strategy_ID": "SMA_Continuous_P85",
        "Core": "SMA_50_200",
        "Mapping": "Continuous",
        "Half_Risk_Percentile": 0.20,
        "Risk_Off_Percentile": 0.85,
        "Smoothing_Days": 5,
        "Score_Lookback_Days": 5 * ANNUALIZATION_DAYS,
        "Hysteresis": 0.00,
        "Minimum_Hold_Days": 0,
        "Target_Volatility": None,
        "Volatility_Window": 20,
        "Complexity": 4,
    },
    *[
        {
            "Strategy_ID": f"SMA_Step_P80_Vol{int(target_vol * 100):02d}",
            "Core": "SMA_50_200",
            "Mapping": "Step",
            "Half_Risk_Percentile": 0.65,
            "Risk_Off_Percentile": 0.80,
            "Smoothing_Days": 5,
            "Score_Lookback_Days": 5 * ANNUALIZATION_DAYS,
            "Hysteresis": 0.10,
            "Minimum_Hold_Days": 5,
            "Target_Volatility": target_vol,
            "Volatility_Window": 20,
            "Complexity": 4,
        }
        for target_vol in (0.08, 0.12, 0.15)
    ],
    {
        "Strategy_ID": "SMA_Step_P80_Vol10",
        "Core": "SMA_50_200",
        "Mapping": "Step",
        "Half_Risk_Percentile": 0.65,
        "Risk_Off_Percentile": 0.80,
        "Smoothing_Days": 5,
        "Score_Lookback_Days": 5 * ANNUALIZATION_DAYS,
        "Hysteresis": 0.10,
        "Minimum_Hold_Days": 5,
        "Target_Volatility": 0.10,
        "Volatility_Window": 20,
        "Complexity": 4,
    },
    {
        "Strategy_ID": "SMA_Step_P80_Vol10_EWMA60",
        "Core": "SMA_50_200",
        "Mapping": "Step",
        "Half_Risk_Percentile": 0.65,
        "Risk_Off_Percentile": 0.80,
        "Smoothing_Days": 5,
        "Score_Lookback_Days": 5 * ANNUALIZATION_DAYS,
        "Hysteresis": 0.10,
        "Minimum_Hold_Days": 5,
        "Target_Volatility": 0.10,
        "Volatility_Window": 60,
        "Complexity": 5,
    },
]


# ---------------------------------------------------------------------------
# 2. Alternative labels and fold-local estimators
# ---------------------------------------------------------------------------

advanced_table["Low"] = clean_df["Low"].astype("float64")
advanced_table["Sharpe_Execution_Volatility_60"] = (
    advanced_table["Close"]
    .pct_change(fill_method=None)
    .ewm(span=60, adjust=False, min_periods=60)
    .std()
    * np.sqrt(ANNUALIZATION_DAYS)
)
for sharpe_horizon in (5, 10, 20):
    sharpe_forward_return = advanced_table[
        ADVANCED_RETURN_COLUMNS[sharpe_horizon]
    ]
    sharpe_threshold = advanced_table[f"Target_Threshold_{sharpe_horizon}d"]
    sharpe_adverse_return = forward_adverse_return(
        advanced_table["Open"], advanced_table["Low"], sharpe_horizon
    )
    valid_target = (
        sharpe_forward_return.notna()
        & sharpe_threshold.notna()
        & sharpe_adverse_return.notna()
    )
    advanced_table[f"Sharpe_Downside_Event_{sharpe_horizon}d"] = (
        sharpe_forward_return.lt(-sharpe_threshold)
        .astype("float64")
        .where(valid_target)
    )
    advanced_table[f"Sharpe_Drawdown_Event_{sharpe_horizon}d"] = (
        sharpe_adverse_return.lt(-sharpe_threshold)
        .astype("float64")
        .where(valid_target)
    )
    advanced_table[f"Sharpe_Downside_Risk_{sharpe_horizon}d"] = (
        -sharpe_adverse_return
    ).clip(lower=0.0).where(valid_target)


def sharpe_target_columns(recipe):
    """Return the target column(s) required by one predictor recipe."""
    horizon = recipe["Horizon"]
    target_type = recipe["Target"]
    if target_type == "Three_Class":
        return [ADVANCED_TARGET_COLUMNS[horizon]]
    if target_type == "Binary_Downside":
        return [f"Sharpe_Downside_Event_{horizon}d"]
    if target_type == "Binary_Drawdown":
        return [f"Sharpe_Drawdown_Event_{horizon}d"]
    if target_type == "Return_Regression":
        return [ADVANCED_RETURN_COLUMNS[horizon]]
    if target_type == "Return_Risk":
        return [
            ADVANCED_RETURN_COLUMNS[horizon],
            f"Sharpe_Downside_Risk_{horizon}d",
        ]
    raise ValueError(f"Unknown Sharpe target type: {target_type}")


def sharpe_make_pipeline(recipe, regression=False):
    """Create a compact fold-local model for the nested experiment."""
    feature_count = len(SHARPE_FEATURE_GROUPS[recipe["Feature_Group"]])
    selected_count = min(SHARPE_SELECTED_FEATURES, feature_count)
    if regression:
        selector_estimator = DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
        )
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=75,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    else:
        selector_estimator = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        if recipe["Model"] == "RF":
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_leaf=25,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        else:
            model = HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=0.05,
                max_iter=75,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=1.0,
                early_stopping=False,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
    selector = SelectFromModel(
        selector_estimator,
        threshold=-np.inf,
        max_features=selected_count,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", selector),
            ("model", model),
        ]
    )


def sharpe_eligible_training_rows(fold_start, recipe):
    """Return rows whose complete labels are known before a refit date."""
    label_end_column = ADVANCED_LABEL_END_COLUMNS[recipe["Horizon"]]
    target_columns = sharpe_target_columns(recipe)
    rolling_start = pd.Timestamp(fold_start) - pd.DateOffset(
        years=recipe["Training_Years"]
    )
    eligible = advanced_table.loc[
        advanced_table["Date"].ge(rolling_start)
        & advanced_table["Date"].lt(fold_start)
        & advanced_table[label_end_column].notna()
        & advanced_table[label_end_column].lt(fold_start)
        & advanced_table[target_columns].notna().all(axis="columns")
    ].sort_values("Date")
    if len(eligible) < SHARPE_MIN_BASE_ROWS + SHARPE_MIN_CALIBRATION_ROWS:
        raise ValueError(
            f"Insufficient training rows for {recipe['Recipe_ID']} at "
            f"{pd.Timestamp(fold_start).date()}"
        )
    return eligible


def sharpe_classifier_blocks(eligible, recipe, fold_start):
    """Split a training history into purged base-fit and calibration blocks."""
    label_end_column = ADVANCED_LABEL_END_COLUMNS[recipe["Horizon"]]
    calibration_rows = max(
        SHARPE_MIN_CALIBRATION_ROWS,
        int(len(eligible) * SHARPE_CALIBRATION_FRACTION),
    )
    calibration_rows = min(
        calibration_rows,
        len(eligible) - SHARPE_MIN_BASE_ROWS,
    )
    calibration = eligible.iloc[-calibration_rows:].copy()
    calibration_start = pd.Timestamp(calibration["Date"].min())
    base_fit = eligible.loc[
        eligible[label_end_column].lt(calibration_start)
    ].copy()
    if len(base_fit) < SHARPE_MIN_BASE_ROWS:
        raise ValueError(
            f"Purging left too few base rows for {recipe['Recipe_ID']}"
        )
    assert base_fit[label_end_column].max() < calibration_start
    assert calibration[label_end_column].max() < pd.Timestamp(fold_start)
    return base_fit, calibration


def sharpe_class_probability(estimator, feature_frame, positive_class):
    probabilities = estimator.predict_proba(feature_frame)
    class_lookup = {
        class_name: class_index
        for class_index, class_name in enumerate(estimator.classes_)
    }
    if positive_class not in class_lookup:
        raise ValueError(f"Missing fitted class: {positive_class}")
    return probabilities[:, class_lookup[positive_class]]


def sharpe_generate_predictor(recipe):
    """Generate one fully causal score stream and refit audit."""
    feature_columns = SHARPE_FEATURE_GROUPS[recipe["Feature_Group"]]
    target_columns = sharpe_target_columns(recipe)
    target_column = target_columns[0]
    label_end_column = ADVANCED_LABEL_END_COLUMNS[recipe["Horizon"]]
    prediction_parts = []
    audit_rows = []

    for fold_start, fold_end in advanced_refit_intervals(
        SHARPE_RESEARCH_START,
        SHARPE_OUTER_END,
        recipe["Refit_Frequency"],
    ):
        prediction_rows = advanced_table.loc[
            advanced_table["Date"].ge(fold_start)
            & advanced_table["Date"].lt(fold_end)
        ].copy()
        if prediction_rows.empty:
            continue
        eligible = sharpe_eligible_training_rows(fold_start, recipe)
        predicted_probability = np.full(len(prediction_rows), np.nan)
        predicted_return = np.full(len(prediction_rows), np.nan)
        predicted_risk = np.full(len(prediction_rows), np.nan)
        calibration_status = "Not_applicable"

        if recipe["Target"] in {
            "Three_Class",
            "Binary_Downside",
            "Binary_Drawdown",
        }:
            base_fit, calibration = sharpe_classifier_blocks(
                eligible,
                recipe,
                fold_start,
            )
            estimator = sharpe_make_pipeline(recipe, regression=False)
            estimator.fit(base_fit[feature_columns], base_fit[target_column])
            if recipe["Target"] == "Three_Class":
                positive_class = "Bearish"
                negative_class = "Bullish"
                raw_positive = sharpe_class_probability(
                    estimator,
                    prediction_rows[feature_columns],
                    positive_class,
                )
                raw_negative = sharpe_class_probability(
                    estimator,
                    prediction_rows[feature_columns],
                    negative_class,
                )
                required_classes = set(CLASS_NAMES)
            else:
                positive_class = 1.0
                raw_positive = sharpe_class_probability(
                    estimator,
                    prediction_rows[feature_columns],
                    positive_class,
                )
                raw_negative = None
                required_classes = {0.0, 1.0}

            fitted_classes = set(base_fit[target_column].unique())
            calibration_classes = set(calibration[target_column].unique())
            if (
                fitted_classes == required_classes
                and calibration_classes == required_classes
            ):
                calibrated = CalibratedClassifierCV(
                    estimator=FrozenEstimator(estimator),
                    method="sigmoid",
                )
                calibrated.fit(
                    calibration[feature_columns],
                    calibration[target_column],
                )
                predicted_probability = sharpe_class_probability(
                    calibrated,
                    prediction_rows[feature_columns],
                    positive_class,
                )
                if recipe["Target"] == "Three_Class":
                    predicted_negative = sharpe_class_probability(
                        calibrated,
                        prediction_rows[feature_columns],
                        negative_class,
                    )
                calibration_status = "Sigmoid"
            else:
                predicted_probability = raw_positive
                if recipe["Target"] == "Three_Class":
                    predicted_negative = raw_negative
                calibration_status = "Raw_fallback"

            risk_score = (
                predicted_probability - predicted_negative
                if recipe["Target"] == "Three_Class"
                else predicted_probability
            )
            train_rows = len(base_fit)
            calibration_rows = len(calibration)
            base_label_end = base_fit[label_end_column].max()
            calibration_start = calibration["Date"].min()
            calibration_label_end = calibration[label_end_column].max()
            if recipe["Target"] == "Three_Class":
                actual_event = prediction_rows[target_column].eq("Bearish")
            else:
                actual_event = prediction_rows[target_column]
        else:
            base_fit = eligible
            return_estimator = sharpe_make_pipeline(recipe, regression=True)
            return_estimator.fit(
                base_fit[feature_columns],
                base_fit[target_column],
            )
            predicted_return = return_estimator.predict(
                prediction_rows[feature_columns]
            )
            if recipe["Target"] == "Return_Risk":
                risk_estimator = sharpe_make_pipeline(recipe, regression=True)
                risk_estimator.fit(
                    base_fit[feature_columns],
                    base_fit[target_columns[1]],
                )
                predicted_risk = np.maximum(
                    risk_estimator.predict(prediction_rows[feature_columns]),
                    1e-4,
                )
                risk_score = -predicted_return / predicted_risk
            else:
                risk_score = -predicted_return
            actual_event = prediction_rows[
                f"Sharpe_Downside_Event_{recipe['Horizon']}d"
            ]
            train_rows = len(base_fit)
            calibration_rows = 0
            base_label_end = base_fit[label_end_column].max()
            calibration_start = pd.NaT
            calibration_label_end = pd.NaT

        fold_predictions = prediction_rows[
            [
                "Date",
                "Execution_Date",
                "Return_End_Date",
                "Tradable_Return_1d",
                "Execution_Volatility",
                "SMA_50_200_Long_Signal",
                ADVANCED_RETURN_COLUMNS[recipe["Horizon"]],
                label_end_column,
            ]
        ].copy()
        fold_predictions = fold_predictions.rename(
            columns={
                ADVANCED_RETURN_COLUMNS[recipe["Horizon"]]: "Forward_Return",
                label_end_column: "Label_End_Date",
            }
        )
        fold_predictions["Recipe_ID"] = recipe["Recipe_ID"]
        fold_predictions["Target_Type"] = recipe["Target"]
        fold_predictions["Model_Type"] = recipe["Model"]
        fold_predictions["Horizon"] = recipe["Horizon"]
        fold_predictions["Training_Years"] = recipe["Training_Years"]
        fold_predictions["Feature_Group"] = recipe["Feature_Group"]
        fold_predictions["Refit_Frequency"] = recipe["Refit_Frequency"]
        fold_predictions["Refit_Date"] = fold_start
        fold_predictions["Risk_Score"] = np.asarray(risk_score, dtype=float)
        fold_predictions["Predicted_Probability"] = predicted_probability
        fold_predictions["Predicted_Return"] = predicted_return
        fold_predictions["Predicted_Risk"] = predicted_risk
        fold_predictions["Actual_Event"] = pd.Series(
            actual_event,
            index=prediction_rows.index,
            dtype="float64",
        ).to_numpy()
        prediction_parts.append(fold_predictions)

        audit_rows.append(
            {
                "Recipe_ID": recipe["Recipe_ID"],
                "Refit_Date": fold_start,
                "Prediction_End": prediction_rows["Date"].max(),
                "Training_Start": base_fit["Date"].min(),
                "Training_End": base_fit["Date"].max(),
                "Training_Label_End": base_label_end,
                "Training_Rows": train_rows,
                "Calibration_Start": calibration_start,
                "Calibration_Label_End": calibration_label_end,
                "Calibration_Rows": calibration_rows,
                "Calibration": calibration_status,
                "Feature_Count": len(feature_columns),
            }
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    if predictions.duplicated(["Recipe_ID", "Date"]).any():
        raise AssertionError(f"Duplicate dates for {recipe['Recipe_ID']}")
    if not np.isfinite(predictions["Risk_Score"]).all():
        raise AssertionError(f"Nonfinite risk score for {recipe['Recipe_ID']}")
    if not (audit["Training_Label_End"] < audit["Refit_Date"]).all():
        raise AssertionError(f"Unpurged labels for {recipe['Recipe_ID']}")
    calibrated_rows = audit["Calibration_Rows"].gt(0)
    if calibrated_rows.any() and not (
        audit.loc[calibrated_rows, "Training_Label_End"]
        < audit.loc[calibrated_rows, "Calibration_Start"]
    ).all():
        raise AssertionError(f"Training/calibration overlap for {recipe['Recipe_ID']}")
    if calibrated_rows.any() and not (
        audit.loc[calibrated_rows, "Calibration_Label_End"]
        < audit.loc[calibrated_rows, "Refit_Date"]
    ).all():
        raise AssertionError(f"Calibration leakage for {recipe['Recipe_ID']}")
    return predictions, audit


SHARPE_PREDICTOR_ENGINE_VERSION = "1.1.0"
SHARPE_FORCE_PREDICTOR_REFIT = bool(
    globals().get("SHARPE_FORCE_PREDICTOR_REFIT", False)
)
sharpe_data_fingerprint = frame_digest(clean_df)
sharpe_model_input_fingerprint = frame_digest(advanced_table)
sharpe_predictor_signature_payload = {
    "Engine_Version": SHARPE_PREDICTOR_ENGINE_VERSION,
    "Recipes": SHARPE_PREDICTOR_RECIPES,
    "Feature_Groups": SHARPE_FEATURE_GROUPS,
    "Research_Start": str(SHARPE_RESEARCH_START.date()),
    "Research_End": str(SHARPE_OUTER_END.date()),
    "Selected_Feature_Count": SHARPE_SELECTED_FEATURES,
    "Calibration_Fraction": SHARPE_CALIBRATION_FRACTION,
    "Data_Fingerprint": sharpe_data_fingerprint,
    "Model_Input_Fingerprint": sharpe_model_input_fingerprint,
    "Source_Provenance": SHARPE_SOURCE_PROVENANCE,
    "Random_State": RANDOM_STATE,
    "Minimum_Base_Rows": SHARPE_MIN_BASE_ROWS,
    "Minimum_Calibration_Rows": SHARPE_MIN_CALIBRATION_ROWS,
}
SHARPE_PREDICTOR_CACHE_SIGNATURE = hashlib.sha256(
    json.dumps(
        sharpe_predictor_signature_payload,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
).hexdigest()
sharpe_score_cache_path = OUTPUT_DIR / "30_sharpe_predictor_scores.csv"
sharpe_audit_cache_path = OUTPUT_DIR / "29_sharpe_predictor_refit_audit.csv"
sharpe_cache_manifest_path = OUTPUT_DIR / "29_sharpe_predictor_cache.json"


def sharpe_load_validated_predictor_cache():
    """Load expensive score streams only after signature and audit checks."""
    if SHARPE_FORCE_PREDICTOR_REFIT:
        return None
    expected_dates = advanced_table.loc[
        advanced_table["Date"].ge(SHARPE_RESEARCH_START)
        & advanced_table["Date"].lt(SHARPE_OUTER_END), "Date"
    ]
    return load_predictor_cache(
        sharpe_score_cache_path, sharpe_audit_cache_path,
        sharpe_cache_manifest_path, SHARPE_PREDICTOR_CACHE_SIGNATURE,
        {recipe["Recipe_ID"] for recipe in SHARPE_PREDICTOR_RECIPES},
        expected_dates,
    )


sharpe_cached_predictors = sharpe_load_validated_predictor_cache()
if sharpe_cached_predictors is None:
    print(
        f"Running {len(SHARPE_PREDICTOR_RECIPES)} predeclared predictor recipes "
        "with historical-only quarterly/annual refits..."
    )
    sharpe_prediction_parts = []
    sharpe_audit_parts = []
    for sharpe_recipe in SHARPE_PREDICTOR_RECIPES:
        print(f"  {sharpe_recipe['Recipe_ID']}")
        recipe_predictions, recipe_audit = sharpe_generate_predictor(sharpe_recipe)
        sharpe_prediction_parts.append(recipe_predictions)
        sharpe_audit_parts.append(recipe_audit)
    sharpe_predictor_predictions = pd.concat(
        sharpe_prediction_parts,
        ignore_index=True,
    )
    sharpe_predictor_audit = pd.concat(sharpe_audit_parts, ignore_index=True)
else:
    sharpe_predictor_predictions, sharpe_predictor_audit = (
        sharpe_cached_predictors
    )
    print(
        "Loaded predictor scores from the validated cache; set "
        "SHARPE_FORCE_PREDICTOR_REFIT=True before this cell to force all refits."
    )


def sharpe_causal_percentile(series, lookback=SHARPE_SCORE_LOOKBACK_DAYS):
    """Rolling percentile rank that never uses a future score."""
    return series.rolling(
        lookback,
        min_periods=SHARPE_SCORE_MIN_DAYS,
    ).rank(pct=True)


def sharpe_make_ensemble(ensemble_id, component_ids):
    """Equal-weight causal score-rank ensemble; no fitted meta-weights."""
    component_frame = sharpe_predictor_predictions.loc[
        sharpe_predictor_predictions["Recipe_ID"].isin(component_ids),
        ["Date", "Recipe_ID", "Risk_Score"],
    ].pivot(index="Date", columns="Recipe_ID", values="Risk_Score")
    if set(component_ids) - set(component_frame.columns):
        raise AssertionError(f"Missing ensemble components for {ensemble_id}")
    ranked_components = component_frame[component_ids].apply(
        sharpe_causal_percentile
    )
    ensemble_score = ranked_components.mean(axis="columns")
    template = sharpe_predictor_predictions.loc[
        sharpe_predictor_predictions["Recipe_ID"].eq(component_ids[0])
    ].set_index("Date").copy()
    template["Recipe_ID"] = ensemble_id
    template["Target_Type"] = "Causal_Rank_Ensemble"
    template["Model_Type"] = "Ensemble"
    template["Risk_Score"] = ensemble_score.reindex(template.index)
    template["Predicted_Probability"] = np.nan
    template["Predicted_Return"] = np.nan
    template["Predicted_Risk"] = np.nan
    return template.reset_index()


SHARPE_ENSEMBLES = {
    "Ensemble_HGB_RF_Downside_10d": [
        "HGB_Downside_10d_W20_Full_Q",
        "RF_Downside_10d_W20_Full_Q",
    ],
    "Ensemble_HGB_MultiRisk_10d": [
        "HGB_Downside_10d_W20_Full_Q",
        "HGB_Drawdown_10d_W20_Full_Q",
        "HGB_ReturnRisk_10d_W20_Full_Q",
    ],
}
sharpe_ensemble_parts = [
    sharpe_make_ensemble(ensemble_id, component_ids)
    for ensemble_id, component_ids in SHARPE_ENSEMBLES.items()
]
sharpe_all_scores = pd.concat(
    [sharpe_predictor_predictions, *sharpe_ensemble_parts],
    ignore_index=True,
)


# ---------------------------------------------------------------------------
# 3. Causal rank overlays and common backtest accounting
# ---------------------------------------------------------------------------

def sharpe_step_multiplier(percentiles, strategy_spec):
    """Risk-off state machine with fast de-risking and hysteretic re-entry."""
    half_cut = strategy_spec["Half_Risk_Percentile"]
    off_cut = strategy_spec["Risk_Off_Percentile"]
    hysteresis = strategy_spec["Hysteresis"]
    minimum_hold = strategy_spec["Minimum_Hold_Days"]
    current = 1.0
    held_days = minimum_hold
    multipliers = []

    for percentile in percentiles.fillna(0.50).to_numpy(dtype=float):
        if strategy_spec["Mapping"] == "Long_Flat":
            proposed = 0.0 if percentile >= off_cut else 1.0
        else:
            proposed = 0.0 if percentile >= off_cut else (
                0.5 if percentile >= half_cut else 1.0
            )

        # De-risk immediately.  Re-risk only after the hold and after the score
        # clears a lower threshold, which avoids threshold chatter.
        if proposed < current:
            current = proposed
            held_days = 0
        elif proposed > current:
            reentry_cut = (
                off_cut - hysteresis
                if current == 0.0
                else half_cut - hysteresis
            )
            if held_days >= minimum_hold and percentile < reentry_cut:
                current = proposed
                held_days = 0
            else:
                held_days += 1
        else:
            held_days += 1
        multipliers.append(current)
    return np.asarray(multipliers, dtype=float)


def sharpe_overlay_backtest(score_frame, strategy_spec, transaction_cost):
    """Map one causal score stream into an execution-aligned long/cash overlay."""
    result = score_frame.sort_values("Date").reset_index(drop=True).copy()
    smoothed_score = result["Risk_Score"].ewm(
        span=strategy_spec["Smoothing_Days"],
        adjust=False,
        min_periods=1,
    ).mean()
    result["Risk_Percentile"] = sharpe_causal_percentile(
        smoothed_score,
        lookback=strategy_spec["Score_Lookback_Days"],
    ).fillna(0.5)

    if strategy_spec["Mapping"] == "Continuous":
        low_cut = strategy_spec["Half_Risk_Percentile"]
        high_cut = strategy_spec["Risk_Off_Percentile"]
        desired = (
            (high_cut - result["Risk_Percentile"])
            / max(high_cut - low_cut, 1e-12)
        ).clip(0.0, 1.0).to_numpy()
        risk_multiplier = []
        current = 1.0
        for target in desired:
            current += float(np.clip(target - current, -0.10, 0.10))
            risk_multiplier.append(current)
        risk_multiplier = np.asarray(risk_multiplier)
    else:
        risk_multiplier = sharpe_step_multiplier(
            result["Risk_Percentile"],
            strategy_spec,
        )

    if strategy_spec["Core"] == "Always_Long":
        core_position = np.ones(len(result))
    else:
        core_position = (
            result["SMA_50_200_Long_Signal"].fillna(0.0).to_numpy(dtype=float)
        )

    if strategy_spec["Target_Volatility"] is None:
        volatility_scale = np.ones(len(result))
    else:
        if strategy_spec["Volatility_Window"] == 20:
            volatility_estimate = result["Execution_Volatility"]
        elif strategy_spec["Volatility_Window"] == 60:
            volatility_by_date = advanced_table.set_index("Date")[
                "Sharpe_Execution_Volatility_60"
            ]
            volatility_estimate = result["Date"].map(volatility_by_date)
        else:
            raise ValueError("Unsupported volatility-estimation window")
        volatility_scale = (
            strategy_spec["Target_Volatility"]
            / volatility_estimate
        ).clip(0.0, 1.0)
        volatility_scale = (volatility_scale / 0.10).round() * 0.10
        volatility_scale = volatility_scale.clip(0.0, 1.0).to_numpy()

    result["Risk_Multiplier"] = risk_multiplier
    result["Volatility_Scale"] = volatility_scale
    result["Position"] = np.clip(
        core_position * risk_multiplier * volatility_scale,
        0.0,
        1.0,
    )
    return sharpe_recompute_returns(result, transaction_cost)


def sharpe_recompute_returns(frame, transaction_cost):
    """Recompute portfolio/cash/excess returns from positions on common dates."""
    result = frame.copy().sort_values("Date").reset_index(drop=True)
    result = result.loc[
        result["Execution_Date"].notna()
        & result["Return_End_Date"].notna()
        & result["Tradable_Return_1d"].notna()
    ].reset_index(drop=True)
    risk_free_daily = np.expm1(
        np.log1p(SHARPE_RISK_FREE_ANNUAL) / ANNUALIZATION_DAYS
    )
    result["Turnover"] = result["Position"].diff().abs().fillna(
        result["Position"].abs()
    )
    result["Gross_Return"] = (
        result["Position"] * result["Tradable_Return_1d"]
        + (1.0 - result["Position"]) * risk_free_daily
    )
    result["Transaction_Cost"] = result["Turnover"] * transaction_cost
    result["Net_Return"] = result["Gross_Return"] - result["Transaction_Cost"]
    result["Net_Excess_Return"] = (
        result["Position"]
        * (result["Tradable_Return_1d"] - risk_free_daily)
        - result["Transaction_Cost"]
    )
    result["Net_Equity"] = (1.0 + result["Net_Return"]).cumprod()
    result["Drawdown"] = equity_drawdown(result["Net_Equity"])
    if not result["Position"].between(0.0, 1.0).all():
        raise AssertionError("Sharpe overlay breached long/cash limits")
    if not np.allclose(
        result["Transaction_Cost"],
        result["Turnover"] * transaction_cost,
    ):
        raise AssertionError("Sharpe overlay costs do not reconcile")
    return result


def sharpe_metric_record(frame):
    """Metrics used in inner selection, final reporting, and promotion gates."""
    evaluated = frame.loc[frame["Net_Return"].notna()].copy()
    row_count = len(evaluated)
    excess = evaluated["Net_Excess_Return"]
    net = evaluated["Net_Return"]
    standard_deviation = excess.std(ddof=1)
    sharpe = (
        excess.mean() / standard_deviation * np.sqrt(ANNUALIZATION_DAYS)
        if standard_deviation > 0
        else 0.0
    )
    downside_risk = downside_deviation(excess)
    sortino = (
        excess.mean() / downside_risk * np.sqrt(ANNUALIZATION_DAYS)
        if downside_risk > 0
        else 0.0
    )
    total_return = (1.0 + net).prod() - 1.0
    cagr = (
        (1.0 + total_return) ** (ANNUALIZATION_DAYS / row_count) - 1.0
        if row_count and total_return > -1.0
        else -1.0
    )
    equity = (1.0 + net).cumprod()
    drawdown = equity_drawdown(equity)
    active = evaluated["Position"].abs().gt(1e-8)
    active_excess = excess.loc[active]
    active_standard_deviation = active_excess.std(ddof=1)
    active_sharpe = (
        active_excess.mean()
        / active_standard_deviation
        * np.sqrt(ANNUALIZATION_DAYS)
        if active_standard_deviation > 0
        else 0.0
    )
    material_position = evaluated["Position"].abs().gt(0.05)
    episode_starts = material_position & ~material_position.shift(1, fill_value=False)
    material_changes = evaluated["Position"].diff().abs().fillna(
        evaluated["Position"].abs()
    ).gt(0.05)
    return {
        "Rows": row_count,
        "Net_Total_Return": total_return,
        "CAGR": cagr,
        "Annualized_Volatility": standard_deviation * np.sqrt(ANNUALIZATION_DAYS),
        "Sharpe_Ratio": sharpe,
        "Sortino_Ratio": sortino,
        "Maximum_Drawdown": drawdown.min() if row_count else 0.0,
        "Annual_Turnover": evaluated["Turnover"].mean() * ANNUALIZATION_DAYS,
        "Trade_Count": int(material_changes.sum()),
        "Invested_Episodes": int(episode_starts.sum()),
        "Active_Days": int(active.sum()),
        "Active_Day_Sharpe": active_sharpe,
        "Average_Absolute_Exposure": evaluated["Position"].abs().mean(),
        "Transaction_Cost_Sum": evaluated["Transaction_Cost"].sum(),
    }


sharpe_score_frames = {
    recipe_id: group.sort_values("Date").reset_index(drop=True)
    for recipe_id, group in sharpe_all_scores.groupby("Recipe_ID")
}
sharpe_candidate_backtests = {}
sharpe_candidate_registry_rows = []
recipe_lookup = {
    recipe["Recipe_ID"]: recipe for recipe in SHARPE_PREDICTOR_RECIPES
}
for score_id, score_frame in sharpe_score_frames.items():
    recipe_complexity = recipe_lookup.get(score_id, {}).get("Complexity", 5)
    for strategy_spec in SHARPE_STRATEGY_SPECS:
        candidate_id = f"{score_id}__{strategy_spec['Strategy_ID']}"
        sharpe_candidate_backtests[candidate_id] = sharpe_overlay_backtest(
            score_frame,
            strategy_spec,
            ADVANCED_ONE_WAY_COST,
        )
        recipe_details = recipe_lookup.get(score_id, {})
        sharpe_candidate_registry_rows.append(
            {
                "Candidate_ID": candidate_id,
                "Score_ID": score_id,
                "Strategy_ID": strategy_spec["Strategy_ID"],
                "Target": recipe_details.get("Target", "Ensemble"),
                "Horizon": recipe_details.get("Horizon", 10),
                "Training_Years": recipe_details.get("Training_Years", np.nan),
                "Feature_Group": recipe_details.get("Feature_Group", "Ensemble"),
                "Refit_Frequency": recipe_details.get("Refit_Frequency", "Mixed"),
                "Complexity": recipe_complexity + strategy_spec["Complexity"],
                "Is_Benchmark": False,
            }
        )


sharpe_market = advanced_table.loc[
    advanced_table["Date"].ge(SHARPE_RESEARCH_START)
    & advanced_table["Date"].lt(SHARPE_OUTER_END)
    & advanced_table["Execution_Date"].notna()
    & advanced_table["Tradable_Return_1d"].notna(),
    [
        "Date",
        "Execution_Date",
        "Return_End_Date",
        "Tradable_Return_1d",
        "Execution_Volatility",
        "SMA_50_200_Long_Signal",
    ],
].copy()

for benchmark_id, benchmark_position, complexity in (
    ("Benchmark__Buy_and_Hold", np.ones(len(sharpe_market)), 0),
    (
        "Benchmark__SMA_50_200",
        sharpe_market["SMA_50_200_Long_Signal"].fillna(0.0).to_numpy(),
        1,
    ),
):
    benchmark_frame = sharpe_market.copy()
    benchmark_frame["Position"] = benchmark_position
    sharpe_candidate_backtests[benchmark_id] = sharpe_recompute_returns(
        benchmark_frame,
        ADVANCED_ONE_WAY_COST,
    )
    sharpe_candidate_registry_rows.append(
        {
            "Candidate_ID": benchmark_id,
            "Score_ID": "Benchmark",
            "Strategy_ID": benchmark_id.split("__", 1)[1],
            "Target": "Benchmark",
            "Horizon": np.nan,
            "Training_Years": np.nan,
            "Feature_Group": "None",
            "Refit_Frequency": "None",
            "Complexity": complexity,
            "Is_Benchmark": True,
        }
    )

sharpe_candidate_registry = pd.DataFrame(sharpe_candidate_registry_rows)


# ---------------------------------------------------------------------------
# 4. Fully nested selection and stitched outer returns
# ---------------------------------------------------------------------------

def sharpe_intervals(start_date, end_date, years):
    intervals = []
    interval_start = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    while interval_start < end_date:
        interval_end = min(interval_start + pd.DateOffset(years=years), end_date)
        intervals.append((interval_start, interval_end))
        interval_start = interval_end
    return intervals


def sharpe_select_nested(
    outer_years=SHARPE_OUTER_YEARS,
    allowed_candidates=None,
    selection_label="Primary",
):
    """Select one complete recipe from prior inner folds for every outer block."""
    candidate_ids = (
        list(sharpe_candidate_backtests)
        if allowed_candidates is None
        else list(allowed_candidates)
    )
    outer_intervals = sharpe_intervals(
        SHARPE_OUTER_START,
        SHARPE_OUTER_END,
        outer_years,
    )
    leaderboard_rows = []
    selection_rows = []
    desired_position_parts = []

    for outer_number, (outer_start, outer_end) in enumerate(outer_intervals, 1):
        first_inner_start = outer_start - pd.DateOffset(
            years=SHARPE_INNER_FOLDS * SHARPE_INNER_YEARS
        )
        inner_intervals = sharpe_intervals(
            first_inner_start,
            outer_start,
            SHARPE_INNER_YEARS,
        )[-SHARPE_INNER_FOLDS:]
        candidate_summaries = []

        for candidate_id in candidate_ids:
            candidate_frame = sharpe_candidate_backtests[candidate_id]
            fold_records = []
            for inner_number, (inner_start, inner_end) in enumerate(
                inner_intervals,
                1,
            ):
                inner_frame = realized_selection_rows(
                    candidate_frame, inner_start, inner_end, outer_start
                )
                if not inner_frame["Return_End_Date"].lt(outer_start).all():
                    raise AssertionError("Selection used unavailable daily returns")
                metrics = sharpe_metric_record(inner_frame)
                fold_record = {
                    "Selection_Label": selection_label,
                    "Outer_Fold": outer_number,
                    "Outer_Start": outer_start,
                    "Outer_End": outer_end,
                    "Inner_Fold": inner_number,
                    "Inner_Start": inner_start,
                    "Inner_End": inner_end,
                    "Return_Data_End": inner_frame["Return_End_Date"].max(),
                    "Candidate_ID": candidate_id,
                    **metrics,
                }
                fold_records.append(fold_record)
                leaderboard_rows.append(fold_record)

            fold_frame = pd.DataFrame(fold_records)
            registry_row = sharpe_candidate_registry.loc[
                sharpe_candidate_registry["Candidate_ID"].eq(candidate_id)
            ].iloc[0]
            median_sharpe = fold_frame["Sharpe_Ratio"].median()
            median_turnover = fold_frame["Annual_Turnover"].median()
            active_fold_count = int(
                fold_frame["Average_Absolute_Exposure"].ge(0.10).sum()
            )
            eligible = (
                fold_frame["Average_Absolute_Exposure"].mean() >= 0.10
                and active_fold_count >= 3
                and np.isfinite(median_sharpe)
            )
            robust_score = (
                median_sharpe - SHARPE_TURNOVER_PENALTY * median_turnover
                if eligible
                else -np.inf
            )
            candidate_summaries.append(
                {
                    "Candidate_ID": candidate_id,
                    "Complexity": registry_row["Complexity"],
                    "Median_Inner_Sharpe": median_sharpe,
                    "Mean_Inner_Exposure": fold_frame[
                        "Average_Absolute_Exposure"
                    ].mean(),
                    "Median_Inner_Turnover": median_turnover,
                    "Active_Inner_Folds": active_fold_count,
                    "Inner_Sharpe_SE": (
                        fold_frame["Sharpe_Ratio"].std(ddof=1)
                        / np.sqrt(len(fold_frame))
                    ),
                    "Eligible": eligible,
                    "Robust_Score": robust_score,
                }
            )

        summary = pd.DataFrame(candidate_summaries)
        eligible_summary = summary.loc[summary["Eligible"]].copy()
        if eligible_summary.empty:
            raise AssertionError(f"No eligible candidate before {outer_start.date()}")
        best = eligible_summary.sort_values(
            ["Robust_Score", "Median_Inner_Turnover"],
            ascending=[False, True],
        ).iloc[0]
        one_se_floor = best["Robust_Score"] - max(
            float(best["Inner_Sharpe_SE"]),
            0.0,
        )
        selected = eligible_summary.loc[
            eligible_summary["Robust_Score"].ge(one_se_floor)
        ].sort_values(
            ["Complexity", "Median_Inner_Turnover", "Robust_Score"],
            ascending=[True, True, False],
        ).iloc[0]
        selected_id = selected["Candidate_ID"]
        selection_rows.append(
            {
                "Selection_Label": selection_label,
                "Outer_Fold": outer_number,
                "Outer_Start": outer_start,
                "Outer_End": outer_end,
                "Inner_Data_End": max(
                    row["Return_Data_End"] for row in leaderboard_rows
                    if row["Outer_Fold"] == outer_number
                ),
                "Selected_Candidate": selected_id,
                "Selected_Complexity": selected["Complexity"],
                "Selected_Robust_Score": selected["Robust_Score"],
                "Best_Robust_Score": best["Robust_Score"],
                "One_SE_Floor": one_se_floor,
                "Median_Inner_Sharpe": selected["Median_Inner_Sharpe"],
                "Mean_Inner_Exposure": selected["Mean_Inner_Exposure"],
                "Active_Inner_Folds": selected["Active_Inner_Folds"],
                "Eligible_Candidates": len(eligible_summary),
            }
        )
        outer_source = sharpe_candidate_backtests[selected_id].loc[
            sharpe_candidate_backtests[selected_id]["Date"].ge(outer_start)
            & sharpe_candidate_backtests[selected_id]["Date"].lt(outer_end)
        ].copy()
        outer_source["Outer_Fold"] = outer_number
        outer_source["Selected_Candidate"] = selected_id
        desired_position_parts.append(
            outer_source[
                [
                    "Date",
                    "Execution_Date",
                    "Return_End_Date",
                    "Tradable_Return_1d",
                    "Execution_Volatility",
                    "Position",
                    "Outer_Fold",
                    "Selected_Candidate",
                ]
            ]
        )

    selections = pd.DataFrame(selection_rows)
    if not (selections["Inner_Data_End"] < selections["Outer_Start"]).all():
        raise AssertionError("An outer selection used future inner data")
    stitched = pd.concat(desired_position_parts, ignore_index=True)
    stitched = sharpe_recompute_returns(stitched, ADVANCED_ONE_WAY_COST)
    stitched["Selection_Label"] = selection_label
    if stitched["Date"].duplicated().any():
        raise AssertionError("Nested outer returns contain duplicate dates")
    return pd.DataFrame(leaderboard_rows), selections, stitched


(
    sharpe_inner_fold_leaderboard,
    sharpe_outer_selections,
    sharpe_nested_backtest,
) = sharpe_select_nested()


def sharpe_standardize_existing(frame, series_name):
    standardized = frame.loc[
        frame["Date"].ge(SHARPE_OUTER_START)
        & frame["Date"].lt(SHARPE_OUTER_END)
    ].copy()
    standardized = standardized[
        [
            "Date",
            "Execution_Date",
            "Return_End_Date",
            "Tradable_Return_1d",
            "Position",
        ]
    ]
    standardized = sharpe_recompute_returns(
        standardized,
        ADVANCED_ONE_WAY_COST,
    )
    standardized["Series"] = series_name
    return standardized


sharpe_hgb_baseline = sharpe_standardize_existing(
    advanced_locked_backtests["Hist_Gradient_Boosting"],
    "Locked_HGB",
)
sharpe_sma_baseline = sharpe_standardize_existing(
    sharpe_candidate_backtests["Benchmark__SMA_50_200"],
    "SMA_50_200",
)
sharpe_buy_hold_baseline = sharpe_standardize_existing(
    sharpe_candidate_backtests["Benchmark__Buy_and_Hold"],
    "Buy_and_Hold",
)
sharpe_nested_backtest["Series"] = "Nested_Selected"


# ---------------------------------------------------------------------------
# 5. Predictive quality, uncertainty, multiplicity, and promotion gates
# ---------------------------------------------------------------------------

predictor_quality_rows = []
for recipe_id, group in sharpe_predictor_predictions.loc[
    sharpe_predictor_predictions["Date"].ge(SHARPE_OUTER_START)
].groupby("Recipe_ID"):
    evaluated = group.loc[
        group["Actual_Event"].notna() & group["Risk_Score"].notna()
    ].copy()
    actual = evaluated["Actual_Event"].astype(int)
    quality = {
        "Recipe_ID": recipe_id,
        "Rows": len(evaluated),
        "Event_Rate": actual.mean(),
        "ROC_AUC_Risk_Score": (
            roc_auc_score(actual, evaluated["Risk_Score"])
            if actual.nunique() == 2
            else np.nan
        ),
        "PR_AUC": np.nan,
        "Brier": np.nan,
        "Return_Rank_IC": evaluated["Risk_Score"].corr(
            -evaluated["Forward_Return"],
            method="spearman",
        ),
        "Return_RMSE": np.nan,
    }
    probability_rows = evaluated["Predicted_Probability"].notna()
    if probability_rows.any():
        quality["PR_AUC"] = average_precision_score(
            actual.loc[probability_rows],
            evaluated.loc[probability_rows, "Predicted_Probability"],
        )
        quality["Brier"] = brier_score_loss(
            actual.loc[probability_rows],
            evaluated.loc[probability_rows, "Predicted_Probability"],
        )
    return_rows = evaluated["Predicted_Return"].notna()
    if return_rows.any():
        quality["Return_RMSE"] = np.sqrt(
            np.mean(
                np.square(
                    evaluated.loc[return_rows, "Predicted_Return"]
                    - evaluated.loc[return_rows, "Forward_Return"]
                )
            )
        )
    predictor_quality_rows.append(quality)
sharpe_predictor_quality = pd.DataFrame(predictor_quality_rows)


series_frames = {
    "Nested_Selected": sharpe_nested_backtest,
    "Locked_HGB": sharpe_hgb_baseline,
    "SMA_50_200": sharpe_sma_baseline,
    "Buy_and_Hold": sharpe_buy_hold_baseline,
}
sharpe_nested_metrics = pd.DataFrame(
    [
        {"Series": series_name, **sharpe_metric_record(frame)}
        for series_name, frame in series_frames.items()
    ]
).set_index("Series")


def sharpe_ratio_from_array(values):
    standard_deviation = np.std(values, ddof=1)
    return (
        np.mean(values) / standard_deviation * np.sqrt(ANNUALIZATION_DAYS)
        if standard_deviation > 0
        else 0.0
    )


def sharpe_paired_circular_bootstrap(
    candidate,
    benchmark,
    repetitions,
    block_length,
    seed,
):
    """Paired circular-block confidence interval for Sharpe and delta-Sharpe."""
    candidate = np.asarray(candidate, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)
    if len(candidate) != len(benchmark):
        raise ValueError("Paired bootstrap series must have identical lengths")
    row_count = len(candidate)
    blocks_needed = math.ceil(row_count / block_length)
    offsets = np.arange(block_length)
    generator = np.random.default_rng(seed)
    candidate_sharpes = np.empty(repetitions)
    benchmark_sharpes = np.empty(repetitions)
    for repetition in range(repetitions):
        starts = generator.integers(0, row_count, size=blocks_needed)
        indices = ((starts[:, None] + offsets) % row_count).ravel()[:row_count]
        candidate_sharpes[repetition] = sharpe_ratio_from_array(candidate[indices])
        benchmark_sharpes[repetition] = sharpe_ratio_from_array(benchmark[indices])
    delta = candidate_sharpes - benchmark_sharpes
    return {
        "Candidate_Sharpe_Lower": np.quantile(candidate_sharpes, 0.025),
        "Candidate_Sharpe_Upper": np.quantile(candidate_sharpes, 0.975),
        "Delta_Sharpe_Lower": np.quantile(delta, 0.025),
        "Delta_Sharpe_Upper": np.quantile(delta, 0.975),
        "Probability_Delta_Positive": np.mean(delta > 0),
        "Candidate_Sharpe_Samples": candidate_sharpes,
        "Delta_Sharpe_Samples": delta,
    }


bootstrap_rows = []
sharpe_bootstrap_samples = {}
for benchmark_number, benchmark_name in enumerate(("Locked_HGB", "SMA_50_200")):
    for block_length, repetitions in (
        (5, SHARPE_SENSITIVITY_BOOTSTRAP_REPS),
        (20, SHARPE_PRIMARY_BOOTSTRAP_REPS),
        (63, SHARPE_SENSITIVITY_BOOTSTRAP_REPS),
    ):
        bootstrap = sharpe_paired_circular_bootstrap(
            sharpe_nested_backtest["Net_Excess_Return"],
            series_frames[benchmark_name]["Net_Excess_Return"],
            repetitions,
            block_length,
            RANDOM_STATE + benchmark_number * 100 + block_length,
        )
        sharpe_bootstrap_samples[(benchmark_name, block_length)] = bootstrap
        bootstrap_rows.append(
            {
                "Candidate": "Nested_Selected",
                "Benchmark": benchmark_name,
                "Block_Length": block_length,
                "Replications": repetitions,
                **{
                    key: value
                    for key, value in bootstrap.items()
                    if not key.endswith("Samples")
                },
            }
        )
sharpe_bootstrap_ci = pd.DataFrame(bootstrap_rows)


def sharpe_hac_mean_test(values, maximum_lag=20):
    """Newey-West/HAC mean estimate and 95% interval."""
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    row_count = len(values)
    long_run_variance = np.dot(centered, centered) / row_count
    for lag in range(1, min(maximum_lag, row_count - 1) + 1):
        weight = 1.0 - lag / (maximum_lag + 1.0)
        covariance = np.dot(centered[lag:], centered[:-lag]) / row_count
        long_run_variance += 2.0 * weight * covariance
    standard_error = np.sqrt(max(long_run_variance, 0.0) / row_count)
    mean = values.mean()
    return {
        "Mean_Daily_Difference": mean,
        "HAC_Standard_Error": standard_error,
        "HAC_t": mean / standard_error if standard_error > 0 else 0.0,
        "HAC_Lower": mean - 1.96 * standard_error,
        "HAC_Upper": mean + 1.96 * standard_error,
    }


def sharpe_max_t_adjustment(candidate_return_frame, benchmark_returns):
    """Block-bootstrap max-T family-wise adjustment for all declared trials."""
    candidate_names = list(candidate_return_frame.columns)
    difference = candidate_return_frame.to_numpy() - np.asarray(
        benchmark_returns,
        dtype=float,
    )[:, None]
    means = difference.mean(axis=0)
    scales = difference.std(axis=0, ddof=1) / np.sqrt(len(difference))
    valid_scales = np.where(scales > 0, scales, np.inf)
    observed_t = means / valid_scales
    centered = difference - means
    row_count = len(centered)
    block_length = SHARPE_PRIMARY_BLOCK_LENGTH
    block_count = math.ceil(row_count / block_length)
    circular = np.vstack([centered, centered[: block_length - 1]])
    block_sums = np.vstack(
        [
            circular[start : start + block_length].sum(axis=0)
            for start in range(row_count)
        ]
    )
    generator = np.random.default_rng(RANDOM_STATE + 900)
    bootstrap_max_t = np.empty(SHARPE_MULTIPLICITY_REPS)
    for repetition in range(SHARPE_MULTIPLICITY_REPS):
        starts = generator.integers(0, row_count, size=block_count)
        bootstrap_mean = block_sums[starts].sum(axis=0) / (
            block_count * block_length
        )
        bootstrap_t = bootstrap_mean / valid_scales
        bootstrap_max_t[repetition] = np.nanmax(bootstrap_t)
    adjusted_p = np.asarray(
        [
            (1.0 + np.sum(bootstrap_max_t >= statistic))
            / (SHARPE_MULTIPLICITY_REPS + 1.0)
            for statistic in observed_t
        ]
    )
    return pd.DataFrame(
        {
            "Candidate_ID": candidate_names,
            "Mean_Daily_Difference": means,
            "Observed_t": observed_t,
            "MaxT_Adjusted_P": adjusted_p,
        }
    ).sort_values("MaxT_Adjusted_P")


evaluation_dates = sharpe_nested_backtest["Date"]
fixed_candidate_returns = {}
for candidate_id, candidate_frame in sharpe_candidate_backtests.items():
    aligned = candidate_frame.set_index("Date").reindex(evaluation_dates)
    fixed_candidate_returns[candidate_id] = aligned["Net_Excess_Return"].to_numpy()
fixed_candidate_returns["Nested_Selected"] = sharpe_nested_backtest[
    "Net_Excess_Return"
].to_numpy()
fixed_candidate_return_frame = pd.DataFrame(fixed_candidate_returns)
sharpe_multiple_testing = sharpe_max_t_adjustment(
    fixed_candidate_return_frame,
    sharpe_hgb_baseline["Net_Excess_Return"],
)


def sharpe_deflated_probability(returns, trial_count):
    """Deflated Sharpe probability using skew, kurtosis, and declared trials."""
    returns = pd.Series(returns).dropna()
    row_count = len(returns)
    daily_sharpe = returns.mean() / returns.std(ddof=1)
    euler_gamma = 0.5772156649015329
    expected_maximum = (
        (1.0 - euler_gamma)
        * norm.ppf(1.0 - 1.0 / max(trial_count, 2))
        + euler_gamma
        * norm.ppf(1.0 - 1.0 / (max(trial_count, 2) * math.e))
    )
    sharpe_standard_error = np.sqrt(
        max(
            1e-12,
            (
                1.0
                - returns.skew() * daily_sharpe
                + (returns.kurt() + 2.0) * daily_sharpe**2 / 4.0
            )
            / max(row_count - 1, 1),
        )
    )
    benchmark_daily_sharpe = expected_maximum / np.sqrt(max(row_count - 1, 1))
    probability = norm.cdf(
        (daily_sharpe - benchmark_daily_sharpe) / sharpe_standard_error
    )
    return {
        "Declared_Trials": trial_count,
        "Daily_Sharpe": daily_sharpe,
        "Expected_Max_Daily_Sharpe": benchmark_daily_sharpe,
        "Deflated_Sharpe_Probability": probability,
    }


sharpe_deflated = sharpe_deflated_probability(
    sharpe_nested_backtest["Net_Excess_Return"],
    len(sharpe_candidate_backtests),
)


def sharpe_run_restriction(label, registry_mask, outer_years=2):
    allowed = sharpe_candidate_registry.loc[
        registry_mask | sharpe_candidate_registry["Is_Benchmark"],
        "Candidate_ID",
    ]
    _, selections, backtest = sharpe_select_nested(
        outer_years=outer_years,
        allowed_candidates=allowed,
        selection_label=label,
    )
    return selections, backtest


robustness_rows = []
for cost_bps in (0, 5, 10, 20):
    nested_cost = sharpe_recompute_returns(
        sharpe_nested_backtest,
        cost_bps / 10_000,
    )
    hgb_cost = sharpe_recompute_returns(
        sharpe_hgb_baseline,
        cost_bps / 10_000,
    )
    sma_cost = sharpe_recompute_returns(
        sharpe_sma_baseline,
        cost_bps / 10_000,
    )
    nested_metrics = sharpe_metric_record(nested_cost)
    hgb_metrics = sharpe_metric_record(hgb_cost)
    sma_metrics = sharpe_metric_record(sma_cost)
    robustness_rows.append(
        {
            "Test": "Transaction_Cost",
            "Setting": f"{cost_bps}_bps",
            "Sharpe": nested_metrics["Sharpe_Ratio"],
            "Delta_vs_HGB": (
                nested_metrics["Sharpe_Ratio"] - hgb_metrics["Sharpe_Ratio"]
            ),
            "Delta_vs_SMA": (
                nested_metrics["Sharpe_Ratio"] - sma_metrics["Sharpe_Ratio"]
            ),
            "Exposure": nested_metrics["Average_Absolute_Exposure"],
        }
    )

restriction_specs = [
    (
        "Training_Window_10",
        sharpe_candidate_registry["Training_Years"].eq(10),
        2,
    ),
    (
        "Training_Window_20",
        sharpe_candidate_registry["Training_Years"].eq(20),
        2,
    ),
    (
        "Quarterly_Refits",
        sharpe_candidate_registry["Refit_Frequency"].eq("QS"),
        2,
    ),
    (
        "Annual_Refits",
        sharpe_candidate_registry["Refit_Frequency"].eq("YS"),
        2,
    ),
    (
        "Annual_Outer_Folds",
        ~sharpe_candidate_registry["Candidate_ID"].isna(),
        1,
    ),
]
sharpe_robustness_selections = []
for label, mask, outer_years in restriction_specs:
    restricted_selections, restricted_backtest = sharpe_run_restriction(
        label,
        mask,
        outer_years,
    )
    sharpe_robustness_selections.append(restricted_selections)
    restricted_metrics = sharpe_metric_record(restricted_backtest)
    robustness_rows.append(
        {
            "Test": "Selection_Design",
            "Setting": label,
            "Sharpe": restricted_metrics["Sharpe_Ratio"],
            "Delta_vs_HGB": (
                restricted_metrics["Sharpe_Ratio"]
                - sharpe_nested_metrics.loc["Locked_HGB", "Sharpe_Ratio"]
            ),
            "Delta_vs_SMA": (
                restricted_metrics["Sharpe_Ratio"]
                - sharpe_nested_metrics.loc["SMA_50_200", "Sharpe_Ratio"]
            ),
            "Exposure": restricted_metrics["Average_Absolute_Exposure"],
        }
    )
sharpe_robustness = pd.DataFrame(robustness_rows)
sharpe_robustness_selection_table = pd.concat(
    sharpe_robustness_selections,
    ignore_index=True,
)


outer_fold_rows = []
for outer_fold, nested_fold in sharpe_nested_backtest.groupby("Outer_Fold"):
    fold_start = nested_fold["Date"].min()
    fold_end = nested_fold["Date"].max() + pd.Timedelta(days=1)
    fold_record = {
        "Outer_Fold": outer_fold,
        "Fold_Start": fold_start,
        "Fold_End": fold_end,
        "Selected_Candidate": nested_fold["Selected_Candidate"].iloc[0],
        **sharpe_metric_record(nested_fold),
    }
    for benchmark_name in ("Locked_HGB", "SMA_50_200"):
        benchmark_fold = series_frames[benchmark_name].loc[
            series_frames[benchmark_name]["Date"].ge(fold_start)
            & series_frames[benchmark_name]["Date"].lt(fold_end)
        ]
        benchmark_metrics = sharpe_metric_record(benchmark_fold)
        fold_record[f"{benchmark_name}_Sharpe"] = benchmark_metrics[
            "Sharpe_Ratio"
        ]
        fold_record[f"Delta_Sharpe_vs_{benchmark_name}"] = (
            fold_record["Sharpe_Ratio"] - benchmark_metrics["Sharpe_Ratio"]
        )
    nested_complement = sharpe_nested_backtest.loc[
        ~sharpe_nested_backtest["Outer_Fold"].eq(outer_fold)
    ]
    complement_metrics = sharpe_metric_record(nested_complement)
    fold_record["Leave_One_Fold_Out_Sharpe"] = complement_metrics[
        "Sharpe_Ratio"
    ]
    for benchmark_name in ("Locked_HGB", "SMA_50_200"):
        benchmark_complement = series_frames[benchmark_name].loc[
            ~series_frames[benchmark_name]["Date"].ge(fold_start)
            | ~series_frames[benchmark_name]["Date"].lt(fold_end)
        ]
        benchmark_complement_metrics = sharpe_metric_record(
            benchmark_complement
        )
        fold_record[
            f"{benchmark_name}_Leave_One_Fold_Out_Sharpe"
        ] = benchmark_complement_metrics["Sharpe_Ratio"]
        fold_record[
            f"Leave_One_Fold_Out_Delta_vs_{benchmark_name}"
        ] = (
            complement_metrics["Sharpe_Ratio"]
            - benchmark_complement_metrics["Sharpe_Ratio"]
        )
    outer_fold_rows.append(fold_record)
sharpe_outer_fold_metrics = pd.DataFrame(outer_fold_rows)


# Sparse-strategy diagnostic: treat a material invested run plus its first exit
# day as one independent episode, then resample complete episodes.
episode_source = sharpe_nested_backtest.copy()
episode_source["Materially_Invested"] = episode_source["Position"].gt(0.05)
episode_source["Episode_Start"] = (
    episode_source["Materially_Invested"]
    & ~episode_source["Materially_Invested"].shift(1, fill_value=False)
)
episode_source["Episode_ID"] = episode_source["Episode_Start"].cumsum()
episode_source["Episode_Member"] = (
    episode_source["Materially_Invested"]
    | episode_source["Materially_Invested"].shift(1, fill_value=False)
)
episode_rows = []
for episode_id, episode in episode_source.loc[
    episode_source["Episode_Member"] & episode_source["Episode_ID"].gt(0)
].groupby("Episode_ID"):
    episode_rows.append(
        {
            "Episode_ID": int(episode_id),
            "Start_Date": episode["Date"].min(),
            "End_Date": episode["Date"].max(),
            "Calendar_Rows": len(episode),
            "Invested_Days": int(episode["Materially_Invested"].sum()),
            "Compounded_Net_Return": (
                (1.0 + episode["Net_Return"]).prod() - 1.0
            ),
            "Log_Return_Contribution": np.log1p(
                episode["Net_Return"]
            ).sum(),
            "Transaction_Cost_Sum": episode["Transaction_Cost"].sum(),
            "Average_Exposure": episode["Position"].mean(),
            "Maximum_Exposure": episode["Position"].max(),
        }
    )
sharpe_episode_log = pd.DataFrame(episode_rows)
if len(sharpe_episode_log):
    episode_generator = np.random.default_rng(RANDOM_STATE + 1_700)
    episode_contributions = sharpe_episode_log[
        "Log_Return_Contribution"
    ].to_numpy()
    episode_bootstrap_means = np.empty(SHARPE_PRIMARY_BOOTSTRAP_REPS)
    for repetition in range(SHARPE_PRIMARY_BOOTSTRAP_REPS):
        episode_bootstrap_means[repetition] = episode_generator.choice(
            episode_contributions,
            size=len(episode_contributions),
            replace=True,
        ).mean()
    sharpe_episode_bootstrap = pd.DataFrame(
        [
            {
                "Episodes": len(episode_contributions),
                "Replications": SHARPE_PRIMARY_BOOTSTRAP_REPS,
                "Mean_Episode_Log_Return": episode_contributions.mean(),
                "Median_Episode_Log_Return": np.median(episode_contributions),
                "Bootstrap_Mean_Lower": np.quantile(
                    episode_bootstrap_means,
                    0.025,
                ),
                "Bootstrap_Mean_Upper": np.quantile(
                    episode_bootstrap_means,
                    0.975,
                ),
                "Probability_Mean_Positive": np.mean(
                    episode_bootstrap_means > 0
                ),
                "Largest_Absolute_Episode_Share": (
                    np.abs(episode_contributions).max()
                    / np.abs(episode_contributions).sum()
                ),
            }
        ]
    )
else:
    sharpe_episode_bootstrap = pd.DataFrame(
        [
            {
                "Episodes": 0,
                "Replications": SHARPE_PRIMARY_BOOTSTRAP_REPS,
                "Mean_Episode_Log_Return": np.nan,
                "Median_Episode_Log_Return": np.nan,
                "Bootstrap_Mean_Lower": np.nan,
                "Bootstrap_Mean_Upper": np.nan,
                "Probability_Mean_Positive": np.nan,
                "Largest_Absolute_Episode_Share": np.nan,
            }
        ]
    )


primary_metrics = sharpe_nested_metrics.loc["Nested_Selected"]
hgb_metrics = sharpe_nested_metrics.loc["Locked_HGB"]
sma_metrics = sharpe_nested_metrics.loc["SMA_50_200"]
primary_hgb_ci = sharpe_bootstrap_ci.loc[
    sharpe_bootstrap_ci["Benchmark"].eq("Locked_HGB")
    & sharpe_bootstrap_ci["Block_Length"].eq(20)
].iloc[0]
primary_sma_ci = sharpe_bootstrap_ci.loc[
    sharpe_bootstrap_ci["Benchmark"].eq("SMA_50_200")
    & sharpe_bootstrap_ci["Block_Length"].eq(20)
].iloc[0]
nested_adjusted_p = sharpe_multiple_testing.loc[
    sharpe_multiple_testing["Candidate_ID"].eq("Nested_Selected"),
    "MaxT_Adjusted_P",
].iloc[0]
positive_fold_share = (
    sharpe_outer_fold_metrics["Delta_Sharpe_vs_Locked_HGB"].gt(0).mean()
)
active_outer_folds = int(
    sharpe_outer_fold_metrics["Average_Absolute_Exposure"].ge(0.10).sum()
)
fold_log_profits = sharpe_nested_backtest.assign(
    Log_Return=np.log1p(sharpe_nested_backtest["Net_Return"])
).groupby("Outer_Fold")["Log_Return"].sum()
positive_log_profits = fold_log_profits.clip(lower=0.0)
largest_fold_profit_share = (
    positive_log_profits.max() / positive_log_profits.sum()
    if positive_log_profits.sum() > 0
    else 1.0
)
cost_10_row = sharpe_robustness.loc[
    sharpe_robustness["Setting"].eq("10_bps")
].iloc[0]

promotion_gate_rows = [
    ("Integrity", "All timing and cost assertions pass", True, "required"),
    ("Statistical", "Net Sharpe >= 0.50", primary_metrics["Sharpe_Ratio"] >= 0.50, primary_metrics["Sharpe_Ratio"]),
    ("Statistical", "Delta Sharpe vs HGB >= 0.15", primary_metrics["Sharpe_Ratio"] - hgb_metrics["Sharpe_Ratio"] >= 0.15, primary_metrics["Sharpe_Ratio"] - hgb_metrics["Sharpe_Ratio"]),
    ("Statistical", "95% bootstrap lower delta vs HGB > 0", primary_hgb_ci["Delta_Sharpe_Lower"] > 0, primary_hgb_ci["Delta_Sharpe_Lower"]),
    ("Statistical", "Max-T adjusted p < 0.05", nested_adjusted_p < 0.05, nested_adjusted_p),
    ("Statistical", "Deflated Sharpe probability > 0.95", sharpe_deflated["Deflated_Sharpe_Probability"] > 0.95, sharpe_deflated["Deflated_Sharpe_Probability"]),
    ("Economic", "Net CAGR > 0", primary_metrics["CAGR"] > 0, primary_metrics["CAGR"]),
    ("Economic", "Average exposure >= 10%", primary_metrics["Average_Absolute_Exposure"] >= 0.10, primary_metrics["Average_Absolute_Exposure"]),
    ("Economic", "At least 30 invested episodes", primary_metrics["Invested_Episodes"] >= 30, primary_metrics["Invested_Episodes"]),
    ("Robustness", "Active in at least 7 of 10 outer folds", active_outer_folds >= 7, active_outer_folds),
    ("Robustness", "Positive delta in at least 70% of outer folds", positive_fold_share >= 0.70, positive_fold_share),
    ("Robustness", "Positive delta at 10 bps cost", cost_10_row["Delta_vs_HGB"] > 0, cost_10_row["Delta_vs_HGB"]),
    ("Robustness", "No outer fold supplies >50% of positive profit", largest_fold_profit_share <= 0.50, largest_fold_profit_share),
    ("Strong_SMA", "95% bootstrap lower delta vs SMA > 0", primary_sma_ci["Delta_Sharpe_Lower"] > 0, primary_sma_ci["Delta_Sharpe_Lower"]),
]
sharpe_promotion_gates = pd.DataFrame(
    promotion_gate_rows,
    columns=["Gate_Group", "Criterion", "Passed", "Observed"],
)
required_gate_mask = ~sharpe_promotion_gates["Gate_Group"].eq("Strong_SMA")
SHARPE_PROMOTION_READY = bool(
    sharpe_promotion_gates.loc[required_gate_mask, "Passed"].all()
)
SHARPE_BEATS_SMA_STRONGLY = bool(
    sharpe_promotion_gates.loc[
        sharpe_promotion_gates["Gate_Group"].eq("Strong_SMA"),
        "Passed",
    ].all()
)


# Causal state assertion: changing the final score cannot change an earlier position.
causality_score = next(iter(sharpe_score_frames.values())).head(400).copy()
causality_spec = SHARPE_STRATEGY_SPECS[1]
causality_base = sharpe_overlay_backtest(
    causality_score,
    causality_spec,
    ADVANCED_ONE_WAY_COST,
)
causality_score.loc[causality_score.index[-1], "Risk_Score"] += 1_000.0
causality_changed = sharpe_overlay_backtest(
    causality_score,
    causality_spec,
    ADVANCED_ONE_WAY_COST,
)
assert np.allclose(
    causality_base["Position"].iloc[:-1],
    causality_changed["Position"].iloc[:-1],
)
assert tuple(sharpe_nested_backtest["Date"]) == tuple(sharpe_hgb_baseline["Date"])
assert tuple(sharpe_nested_backtest["Date"]) == tuple(sharpe_sma_baseline["Date"])


# ---------------------------------------------------------------------------
# 6. Reproducible artifacts and notebook report
# ---------------------------------------------------------------------------

experiment_protocol = {
    "Version": SHARPE_RESEARCH_VERSION,
    "Predictor_Engine_Version": SHARPE_PREDICTOR_ENGINE_VERSION,
    "Predictor_Cache_Signature": SHARPE_PREDICTOR_CACHE_SIGNATURE,
    "Source_Provenance": SHARPE_SOURCE_PROVENANCE,
    "Predictor_Signature_Payload": sharpe_predictor_signature_payload,
    "Status": SHARPE_RESEARCH_STATUS,
    "Research_Start": str(SHARPE_RESEARCH_START.date()),
    "Outer_Start": str(SHARPE_OUTER_START.date()),
    "Outer_End": str(SHARPE_OUTER_END.date()),
    "Outer_Fold_Years": SHARPE_OUTER_YEARS,
    "Inner_Folds": SHARPE_INNER_FOLDS,
    "Inner_Fold_Years": SHARPE_INNER_YEARS,
    "Purge_Days": max(recipe["Horizon"] for recipe in SHARPE_PREDICTOR_RECIPES),
    "One_Way_Cost": ADVANCED_ONE_WAY_COST,
    "Risk_Free_Annual_Assumption": SHARPE_RISK_FREE_ANNUAL,
    "Predictor_Recipes": SHARPE_PREDICTOR_RECIPES,
    "Ensembles": SHARPE_ENSEMBLES,
    "Strategy_Specs": SHARPE_STRATEGY_SPECS,
    "Candidate_Trial_Count": len(sharpe_candidate_backtests),
    "Primary_Benchmark": "Locked_HGB",
    "Harder_Benchmark": "SMA_50_200",
    "Primary_Bootstrap_Replications": SHARPE_PRIMARY_BOOTSTRAP_REPS,
    "Multiplicity_Replications": SHARPE_MULTIPLICITY_REPS,
    "Limitations": [
        "2007-2026 was previously inspected and is not a fresh holdout.",
        "The source is a price index, so dividends are unavailable.",
        "No historical cash-rate series is present; cash and risk-free are set to zero.",
        "Macro, breadth, credit, and cross-asset features require separately versioned point-in-time data and are not fabricated here.",
    ],
}
protocol_for_hash = json.dumps(
    experiment_protocol,
    sort_keys=True,
    default=str,
).encode("utf-8")
experiment_protocol["Config_SHA256"] = hashlib.sha256(protocol_for_hash).hexdigest()

with (OUTPUT_DIR / "28_sharpe_experiment_registry.json").open(
    "w",
    encoding="utf-8",
) as protocol_file:
    json.dump(experiment_protocol, protocol_file, indent=2, default=str)

sharpe_predictor_audit.to_csv(
    OUTPUT_DIR / "29_sharpe_predictor_refit_audit.csv",
    index=False,
)
sharpe_predictor_predictions.to_csv(
    OUTPUT_DIR / "30_sharpe_predictor_scores.csv",
    index=False,
)
with sharpe_cache_manifest_path.open("w", encoding="utf-8") as cache_file:
    json.dump(
        {
            "Predictor_Cache_Signature": SHARPE_PREDICTOR_CACHE_SIGNATURE,
            "Engine_Version": SHARPE_PREDICTOR_ENGINE_VERSION,
            "Data_Fingerprint": sharpe_data_fingerprint,
            "Model_Input_Fingerprint": sharpe_model_input_fingerprint,
            "Recipe_Count": len(SHARPE_PREDICTOR_RECIPES),
            "Score_Rows": len(sharpe_predictor_predictions),
            "Audit_Rows": len(sharpe_predictor_audit),
            "Score_SHA256": file_digest(sharpe_score_cache_path),
            "Audit_SHA256": file_digest(sharpe_audit_cache_path),
        },
        cache_file,
        indent=2,
    )
sharpe_predictor_quality.to_csv(
    OUTPUT_DIR / "31_sharpe_predictor_quality.csv",
    index=False,
)
sharpe_candidate_registry.to_csv(
    OUTPUT_DIR / "32_sharpe_candidate_registry.csv",
    index=False,
)
sharpe_inner_fold_leaderboard.to_csv(
    OUTPUT_DIR / "33_sharpe_inner_fold_leaderboard.csv",
    index=False,
)
sharpe_outer_selections.to_csv(
    OUTPUT_DIR / "34_sharpe_outer_selections.csv",
    index=False,
)
sharpe_nested_backtest.to_csv(
    OUTPUT_DIR / "35_sharpe_nested_daily_returns.csv",
    index=False,
)
sharpe_nested_metrics.to_csv(
    OUTPUT_DIR / "36_sharpe_nested_metrics.csv",
)
sharpe_bootstrap_ci.to_csv(
    OUTPUT_DIR / "37_sharpe_bootstrap_ci.csv",
    index=False,
)
sharpe_multiple_testing.to_csv(
    OUTPUT_DIR / "38_sharpe_multiple_testing.csv",
    index=False,
)
pd.DataFrame(
    [
        {
            **sharpe_deflated,
            **sharpe_hac_mean_test(
                sharpe_nested_backtest["Net_Excess_Return"]
                - sharpe_hgb_baseline["Net_Excess_Return"]
            ),
        }
    ]
).to_csv(OUTPUT_DIR / "39_sharpe_deflated_and_hac.csv", index=False)
sharpe_robustness.to_csv(
    OUTPUT_DIR / "40_sharpe_robustness.csv",
    index=False,
)
sharpe_robustness_selection_table.to_csv(
    OUTPUT_DIR / "41_sharpe_robustness_selections.csv",
    index=False,
)
sharpe_outer_fold_metrics.to_csv(
    OUTPUT_DIR / "42_sharpe_outer_fold_metrics.csv",
    index=False,
)
sharpe_promotion_gates.to_csv(
    OUTPUT_DIR / "43_sharpe_promotion_gates.csv",
    index=False,
)
sharpe_episode_log.to_csv(
    OUTPUT_DIR / "44_sharpe_episode_log.csv",
    index=False,
)
sharpe_episode_bootstrap.to_csv(
    OUTPUT_DIR / "45_sharpe_episode_bootstrap.csv",
    index=False,
)


figure, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
for series_name, frame, color, linestyle in (
    ("Nested selected", sharpe_nested_backtest, "#e15759", "-"),
    ("Locked HGB", sharpe_hgb_baseline, "#4e79a7", "--"),
    ("SMA 50/200", sharpe_sma_baseline, "#59a14f", ":"),
    ("Buy and hold", sharpe_buy_hold_baseline, "black", "-."),
):
    axes[0].plot(
        frame["Execution_Date"],
        frame["Net_Equity"],
        label=series_name,
        color=color,
        linestyle=linestyle,
        linewidth=1.6,
    )
axes[0].set_yscale("log")
axes[0].set_ylabel("Growth of $1, log scale")
axes[0].set_title("Nested pseudo-OOS equity: selected recipe versus benchmarks")
axes[0].legend(ncol=2)
axes[0].grid(alpha=0.25)
axes[1].step(
    sharpe_nested_backtest["Execution_Date"],
    sharpe_nested_backtest["Position"],
    where="post",
    color="#e15759",
    linewidth=1.1,
)
axes[1].set_ylim(-0.05, 1.05)
axes[1].set_ylabel("Executed exposure")
axes[1].set_xlabel("Execution date")
axes[1].set_title("Exposure chosen only from prior inner folds")
axes[1].grid(alpha=0.25)
sharpe_equity_figure_path = save_and_show(
    figure,
    "20_sharpe_nested_equity_and_exposure.png",
)


figure, axes = plt.subplots(1, 3, figsize=(21, 5))
x_positions = np.arange(len(sharpe_outer_fold_metrics))
bar_width = 0.26
for offset, column, label, color in (
    (-bar_width, "Sharpe_Ratio", "Nested selected", "#e15759"),
    (0.0, "Locked_HGB_Sharpe", "Locked HGB", "#4e79a7"),
    (bar_width, "SMA_50_200_Sharpe", "SMA 50/200", "#59a14f"),
):
    axes[0].bar(
        x_positions + offset,
        sharpe_outer_fold_metrics[column],
        width=bar_width,
        label=label,
        color=color,
    )
axes[0].axhline(0.0, color="black", linewidth=0.8)
axes[0].set_xticks(x_positions, sharpe_outer_fold_metrics["Outer_Fold"])
axes[0].set_xlabel("Two-year outer fold")
axes[0].set_ylabel("Annualized net excess Sharpe")
axes[0].set_title("Outer-fold stability")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.25)

for column, label, color, marker in (
    (
        "Leave_One_Fold_Out_Sharpe",
        "Nested selected",
        "#e15759",
        "o",
    ),
    (
        "Locked_HGB_Leave_One_Fold_Out_Sharpe",
        "Locked HGB",
        "#4e79a7",
        "s",
    ),
    (
        "SMA_50_200_Leave_One_Fold_Out_Sharpe",
        "SMA 50/200",
        "#59a14f",
        "^",
    ),
):
    axes[1].plot(
        sharpe_outer_fold_metrics["Outer_Fold"],
        sharpe_outer_fold_metrics[column],
        marker=marker,
        linewidth=1.3,
        color=color,
        label=label,
    )
axes[1].axhline(0.0, color="black", linewidth=0.8)
axes[1].set_xlabel("Omitted outer fold")
axes[1].set_ylabel("Sharpe on all remaining folds")
axes[1].set_title("Leave-one-outer-fold-out stability")
axes[1].set_xticks(sharpe_outer_fold_metrics["Outer_Fold"])
axes[1].legend()
axes[1].grid(alpha=0.25)

comparison_labels = ["vs locked HGB", "vs SMA 50/200"]
comparison_rows = [primary_hgb_ci, primary_sma_ci]
delta_points = [
    primary_metrics["Sharpe_Ratio"] - hgb_metrics["Sharpe_Ratio"],
    primary_metrics["Sharpe_Ratio"] - sma_metrics["Sharpe_Ratio"],
]
lower_errors = [
    point - row["Delta_Sharpe_Lower"]
    for point, row in zip(delta_points, comparison_rows)
]
upper_errors = [
    row["Delta_Sharpe_Upper"] - point
    for point, row in zip(delta_points, comparison_rows)
]
axes[2].errorbar(
    comparison_labels,
    delta_points,
    yerr=[lower_errors, upper_errors],
    fmt="o",
    color="#e15759",
    capsize=6,
)
axes[2].axhline(0.0, color="black", linewidth=0.8)
axes[2].set_ylabel("Paired delta-Sharpe")
axes[2].set_title("95% circular-block bootstrap intervals")
axes[2].grid(axis="y", alpha=0.25)
sharpe_uncertainty_figure_path = save_and_show(
    figure,
    "21_sharpe_outer_folds_and_uncertainty.png",
)


print("Nested Sharpe-improvement comparison:")
display(
    sharpe_nested_metrics[
        [
            "Net_Total_Return",
            "CAGR",
            "Annualized_Volatility",
            "Sharpe_Ratio",
            "Sortino_Ratio",
            "Maximum_Drawdown",
            "Average_Absolute_Exposure",
            "Trade_Count",
            "Invested_Episodes",
        ]
    ].style.format(
        {
            "Net_Total_Return": "{:.2%}",
            "CAGR": "{:.2%}",
            "Annualized_Volatility": "{:.2%}",
            "Sharpe_Ratio": "{:.3f}",
            "Sortino_Ratio": "{:.3f}",
            "Maximum_Drawdown": "{:.2%}",
            "Average_Absolute_Exposure": "{:.1%}",
        }
    )
)
print("Outer-block recipe selections:")
display(
    sharpe_outer_selections[
        [
            "Outer_Fold",
            "Outer_Start",
            "Outer_End",
            "Selected_Candidate",
            "Median_Inner_Sharpe",
            "Mean_Inner_Exposure",
        ]
    ]
)
print("Predeclared promotion gates:")
display(sharpe_promotion_gates)
print(
    f"Promotion ready versus locked HGB: {SHARPE_PROMOTION_READY}. "
    f"Strong evidence versus SMA 50/200: {SHARPE_BEATS_SMA_STRONGLY}."
)
print(
    "Interpretation rule: failed gates are evidence against promotion, not an "
    "invitation to tune on these outer results. Freeze one winner and collect "
    "genuinely new forward data before making a strong claim."
)
print(
    "All nested timing, purging, causal-score, position, and cost checks passed."
)
SHARPE_COMPLETED_RUN_ID = write_run_manifest(
    Path.cwd(), SHARPE_SOURCE_PROVENANCE, SHARPE_EXECUTED_CODE_DIGEST
)
print(f"Research run completed: {SHARPE_COMPLETED_RUN_ID}")

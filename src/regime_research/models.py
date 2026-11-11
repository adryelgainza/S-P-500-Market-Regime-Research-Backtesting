"""Purged recipe/refit model fitting with validated, resumable score checkpoints."""

from dataclasses import asdict, dataclass
from importlib import metadata
import hashlib
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from .archive import file_hash
from .checkpoints import CheckpointStore
from .experiments import content_hash
from .labels import LabelConfig, build_labels
from .validation import validate_refit


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 42
    selected_features: int = 16
    minimum_base_rows: int = 756
    minimum_calibration_rows: int = 126
    calibration_fraction: float = .15

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in
               [self.selected_features, self.minimum_base_rows, self.minimum_calibration_rows]):
            raise ValueError("Model row and feature counts must be positive integers")
        if not isinstance(self.seed, int) or self.seed < 0 or not 0 < self.calibration_fraction < .5:
            raise ValueError("Invalid model seed or calibration fraction")


def validate_recipe(recipe):
    required = {"Recipe_ID", "Model", "Target", "Horizon", "Training_Years", "Feature_Group", "Refit_Frequency", "Complexity"}
    if set(recipe) != required or recipe["Model"] not in {"HGB", "RF"}:
        raise ValueError("Use a complete supported predictor recipe")
    if recipe["Target"] not in {"Three_Class", "Binary_Downside", "Binary_Drawdown", "Return_Regression", "Return_Risk"}:
        raise ValueError("Unsupported predictor target")
    if recipe["Model"] == "RF" and recipe["Target"] in {"Return_Regression", "Return_Risk"}:
        raise ValueError("The registered RF recipe is a classifier")
    if recipe["Horizon"] not in {5, 10, 20} or recipe["Training_Years"] not in {5, 10, 15, 20, 30}:
        raise ValueError("Unsupported horizon or training window")
    if recipe["Refit_Frequency"] not in {"YS", "QS"} or not recipe["Recipe_ID"]:
        raise ValueError("Unsupported refit schedule or empty recipe identifier")
    return dict(recipe)


def make_pipeline(recipe, feature_count, config=ModelConfig(), regression=False):
    validate_recipe(recipe)
    tree = DecisionTreeRegressor if regression else DecisionTreeClassifier
    tree_settings = {"max_depth": 5, "min_samples_leaf": 50, "random_state": config.seed}
    if not regression:
        tree_settings["class_weight"] = "balanced"
    selector = SelectFromModel(tree(**tree_settings), threshold=-np.inf,
                               max_features=min(feature_count, config.selected_features))
    hgb = {"learning_rate": .05, "max_iter": 75, "max_leaf_nodes": 15,
           "min_samples_leaf": 25, "l2_regularization": 1., "early_stopping": False, "random_state": config.seed}
    if regression:
        model = HistGradientBoostingRegressor(loss="squared_error", **hgb)
    elif recipe["Model"] == "RF":
        model = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=25,
            max_features="sqrt", class_weight="balanced_subsample", n_jobs=1, random_state=config.seed)
    else:
        model = HistGradientBoostingClassifier(loss="log_loss", class_weight="balanced", **hgb)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("selector", selector), ("model", model)])


def frame_hash(frame):
    return content_hash({"Columns": list(frame.columns), "Dtypes": frame.dtypes.astype(str).tolist(),
        "Values": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()})


def _probability(estimator, features, positive):
    classes = list(estimator.classes_)
    if positive not in classes:
        raise ValueError("A required target class is absent from the base fit")
    return estimator.predict_proba(features)[:, classes.index(positive)]


def fit_scores(bars, features, recipe, *, start, end, checkpoint_directory, config=ModelConfig(), progress=None):
    """Return scores and fit audits using only matured labels at each refit."""
    recipe = validate_recipe(recipe)
    if not bars.index.equals(features.index) or features.columns.duplicated().any():
        raise ValueError("Features and bars require identical unique row alignment")
    labels = build_labels(bars, LabelConfig(horizon=recipe["Horizon"]))
    times = pd.to_datetime(bars.Available_At, utc=True)
    sessions = pd.to_datetime(bars.Session)
    first, stop = pd.Timestamp(start), pd.Timestamp(end)
    boundaries = [first, *pd.date_range(first.normalize()+pd.Timedelta(days=1), stop,
                                      freq=recipe["Refit_Frequency"]).tolist(), stop]
    boundaries = sorted(set(value for value in boundaries if first <= value <= stop))
    store = CheckpointStore(checkpoint_directory)
    target = "Return_Regression" if recipe["Target"] == "Return_Risk" else recipe["Target"]
    runtime = {name: metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "joblib", "threadpoolctl"]}
    runtime.update(Python=platform.python_version(), Platform_System=platform.system(), Machine=platform.machine())
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in
               ["models.py", "features.py", "labels.py", "checkpoints.py", "validation.py", "protocol.py",
                "archive.py", "experiments.py", "config.py"]}
    parts, audits = [], []
    for beginning, ending in zip(boundaries[:-1], boundaries[1:]):
        prediction_ids = bars.index[(sessions >= beginning) & (sessions < ending)]
        if len(prediction_ids) == 0:
            continue
        refit = times.loc[prediction_ids[0]]
        window = sessions >= beginning-pd.DateOffset(years=recipe["Training_Years"])
        eligible = bars.index[window & (times < refit) & (pd.to_datetime(labels.Label_End_Time, utc=True) < refit)
                              & labels[target].notna()]
        regression = recipe["Target"] in {"Return_Regression", "Return_Risk"}
        calibration_ids = bars.index[:0]
        base_ids = eligible
        if not regression:
            count = max(config.minimum_calibration_rows, int(len(eligible)*config.calibration_fraction))
            if len(eligible)-count < config.minimum_base_rows:
                raise ValueError(f"Insufficient purged history for {recipe['Recipe_ID']} at {refit}")
            calibration_ids = eligible[-count:]
            calibration_start = times.loc[calibration_ids[0]]
            base_ids = eligible[pd.to_datetime(labels.loc[eligible, "Label_End_Time"], utc=True) < calibration_start]
        if len(base_ids) < config.minimum_base_rows:
            raise ValueError("Insufficient base rows after purging calibration overlap")
        audit_rows = pd.DataFrame({"Row_ID": bars.index, "Decision_Time": times, "Available_At": bars.Available_At,
                                   "Label_End_Time": labels.Label_End_Time}, index=bars.index)
        if not regression:
            validate_refit(audit_rows.loc[base_ids], audit_rows.loc[calibration_ids], refit)
        elif (pd.to_datetime(labels.loc[base_ids, "Label_End_Time"], utc=True) >= refit).any():
            raise ValueError("Regression labels are immature at the refit")
        train_input = pd.concat([features.loc[eligible], labels.loc[eligible, [target, "Downside_Risk", "Label_End_Time"]]], axis=1)
        predict_input = pd.concat([features.loc[prediction_ids], bars.loc[prediction_ids, ["Available_At"]]], axis=1)
        key = {"Schema": 1, "Recipe": recipe, "Config": asdict(config), "Refit_UTC": refit.isoformat(),
               "Training_SHA256": frame_hash(train_input), "Prediction_SHA256": frame_hash(predict_input),
               "Runtime": runtime, "Source_SHA256": sources}
        saved = store.load(key)
        reused = saved is not None
        if saved is None:
            with threadpool_limits(limits=1):
                estimator = make_pipeline(recipe, len(features.columns), config, regression)
                estimator.fit(features.loc[base_ids], labels.loc[base_ids, target])
                status = "Not_applicable"
                if regression:
                    predicted_return = estimator.predict(features.loc[prediction_ids])
                    score = -predicted_return
                    if recipe["Target"] == "Return_Risk":
                        risk = make_pipeline(recipe, len(features.columns), config, regression=True)
                        risk.fit(features.loc[base_ids], labels.loc[base_ids, "Downside_Risk"])
                        score = -predicted_return/np.maximum(risk.predict(features.loc[prediction_ids]), 1e-4)
                else:
                    required = {"Bearish", "Neutral", "Bullish"} if target == "Three_Class" else {0., 1.}
                    if set(labels.loc[base_ids, target]) == required and set(labels.loc[calibration_ids, target]) == required:
                        calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
                        calibrated.fit(features.loc[calibration_ids], labels.loc[calibration_ids, target])
                        estimator, status = calibrated, "Sigmoid"
                    else:
                        status = "Raw_fallback"
                    positive = "Bearish" if target == "Three_Class" else 1.
                    score = _probability(estimator, features.loc[prediction_ids], positive)
                    if target == "Three_Class":
                        score = score-_probability(estimator, features.loc[prediction_ids], "Bullish")
                scores = labels.loc[prediction_ids, ["Decision_Time", "Execution_Time", "Return_End_Time"]].copy()
                scores.insert(0, "Row_ID", prediction_ids)
                scores["Risk_Score"] = score
                scores["Recipe_ID"] = recipe["Recipe_ID"]
                scores["Refit_Time"] = refit
                if not np.isfinite(scores.Risk_Score).all():
                    raise ValueError("Model produced a nonfinite risk score")
                audit = {"Recipe_ID": recipe["Recipe_ID"], "Refit_Time": refit.isoformat(),
                    "Base_IDs": base_ids.tolist(), "Calibration_IDs": calibration_ids.tolist(),
                    "Base_Rows": len(base_ids), "Calibration_Rows": len(calibration_ids),
                    "Calibration": status, "Checkpoint_Key_SHA256": content_hash(key)}
                store.save(key, scores, audit)
        else:
            scores, audit = saved
        if scores.Row_ID.tolist() != prediction_ids.tolist() or audit["Checkpoint_Key_SHA256"] != content_hash(key):
            raise ValueError("Checkpoint rows or audit do not match the requested refit")
        parts.append(scores)
        audits.append(audit)
        if progress is not None:
            progress({"Recipe_ID": recipe["Recipe_ID"], "Refit_Time": refit.isoformat(), "Reused": reused,
                      "Rows": len(scores), "Completed_Refits": len(parts)})
    if not parts:
        raise ValueError("No prediction intervals in the requested history")
    return pd.concat(parts, ignore_index=True), audits

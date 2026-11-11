"""Explicit offline replay and fixed-recipe research through the tested package."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .allocation import AllocationConfig, allocate, market_overlays
from .archive import file_hash, materialize_baseline, replay_baseline
from .data import build_eligible_sample, load_dataset
from .data.contracts import HistoryRequirements
from .execution import ExecutionConfig, portfolio_metrics, save_portfolio_result, simulate_portfolio, verify_portfolio_result
from .attribution import load_portfolio
from .experiments import atomic_write, canonical_bytes, content_hash
from .features import build_features
from .labels import LabelConfig, build_labels
from .models import ModelConfig, fit_scores, frame_hash
from .protocol import validate_protocol
from .reporting import load_study, render_figures
from .statistics import InferenceConfig, InferenceResult, infer_performance
from .statistical_research import _compare_saved_table
from .validation import validate_refit


def analyze_returns(primary, benchmarks, family, config):
    """Keep short valid samples reportable with explicitly inconclusive inference."""
    if len(primary) >= config.minimum_observations:
        return infer_performance(primary, benchmarks, family, config)
    status = "inconclusive_insufficient_observations"
    empty = pd.DataFrame()
    return InferenceResult({"Config": asdict(config), "Observations": len(primary), "Status": status},
        pd.DataFrame([{"Scenario": "raw_registered_count", "DSR": None, "Status": status}]), empty, empty,
        pd.DataFrame([{"Benchmark": name, "Status": status, "Observations": len(primary)} for name in benchmarks]),
        empty, pd.DataFrame([{"Candidate_ID": name, "MaxT_Adjusted_P": None, "Status": status} for name in family]))


def _write_tables(output, metrics, inference, predictive, daily, summary):
    for name, frame in [("metrics", metrics), ("inference", inference), ("predictive", predictive), ("chart_data", daily)]:
        frame.to_csv(output/f"{name}.csv", index=False, lineterminator="\n")
    render_figures(daily, metrics, output/"figures", summary["Chart_Caption"])
    atomic_write(output/"summary.json", canonical_bytes(summary)+b"\n")
    files = {path.relative_to(output).as_posix(): file_hash(path) for path in output.rglob("*") if path.is_file()}
    atomic_write(output/"study_manifest.json", canonical_bytes({"Schema_Version": "1.0.0", "Files": files})+b"\n")


def _replay(project, output, config):
    baseline, verified = replay_baseline(project, output)
    original = baseline/"runs/layer4_attribution"
    metrics = pd.read_csv(original/"attribution.csv")
    inference = pd.read_csv(original/"paired_comparisons.csv")
    predictive = pd.read_csv(original/"predictive_diagnostics.csv")
    names = ["Nested_Selected", "Buy_And_Hold", "SMA_Only", "Core_Without_ML"]
    daily = pd.concat([pd.read_parquet(original/"portfolios"/name/"portfolio_daily.parquet").assign(Portfolio=name)
                       for name in names], ignore_index=True)
    summary = {"Mode": "replay", "Stage": "retrospective", "Instrument": "S&P 500 price-index proxy",
        "Protocol_Path": config.protocol, "Original_Statistical_Family": 182,
        "Original_MaxT_Hypotheses": 183, "Observations": int(verified["Inference"]["Observations"]),
        "Sharpe": float(verified["Inference"]["Sharpe"]), "Forward_Observations": 0,
        "Interpretation": "The nested model's incremental value remains unestablished against simple matched controls.",
        "Chart_Caption": "S&P 500 price-index proxy | 2007–2026 retrospective | net of 5 bps one-way costs | zero cash, no distributions",
        "Baseline_Archive_SHA256": json.loads((project/"archives/layer5_baseline/manifest.json").read_text())["Archive_SHA256"]}
    _write_tables(output, metrics, inference, predictive, daily, summary)


def _fit(project, output, config, progress):
    protocol = validate_protocol(json.loads((project/config.protocol).read_text(encoding="utf-8")))
    if config.seed != protocol["Models"]["Seed"]:
        raise ValueError("Run seed differs from the registered model")
    baseline = materialize_baseline(project)
    dataset_root = project/"datasets" if (project/"datasets"/config.dataset_id).exists() else baseline/"datasets"
    dataset = load_dataset(config.dataset_id, root=dataset_root)
    if (set(dataset.bars.Instrument_ID) != {protocol["Universe"]["Primary"]}
            or set(dataset.bars.Currency) != {protocol["Universe"]["Currency"]}):
        raise ValueError("Run dataset differs from the registered primary instrument")
    reference_id = next(identifier for identifier in protocol["Data"]["Dataset_IDs"]
        if set(load_dataset(identifier, root=baseline/"datasets").bars.Instrument_ID) == {protocol["Universe"]["Primary"]})
    reference = load_dataset(reference_id, root=baseline/"datasets")
    if dataset.manifest["Cash_Policy"] != reference.manifest["Cash_Policy"]:
        raise ValueError("Refreshed snapshot changed the frozen cash policy")
    bars = dataset.bars
    features = build_features(bars)
    if list(features.columns) != protocol["Models"]["Features"]:
        raise ValueError("Feature order differs from the frozen model")
    recipe = protocol["Models"]["Recipes"][0]
    if recipe["Model"] != "HGB" or recipe["Target"] != "Binary_Downside":
        raise ValueError("The supported primary study requires its registered HGB downside classifier")
    model_config = ModelConfig(seed=config.seed, selected_features=protocol["Models"]["Preprocessing"]["Selector"]["max_features"],
        minimum_base_rows=protocol["Models"]["Fit_Window"]["Minimum_Base_Rows"],
        minimum_calibration_rows=protocol["Models"]["Calibration"]["Minimum_Rows"],
        calibration_fraction=protocol["Models"]["Calibration"]["Fraction"])
    expected_selector = {"Estimator": "DecisionTreeClassifier", "max_depth": 5, "min_samples_leaf": 50,
        "class_weight": "balanced", "random_state": config.seed, "max_features": model_config.selected_features, "threshold": "-inf"}
    expected_model = {"Estimator": "HistGradientBoostingClassifier", "loss": "log_loss", "learning_rate": .05,
        "max_iter": 75, "max_leaf_nodes": 15, "min_samples_leaf": 25, "l2_regularization": 1.,
        "early_stopping": False, "class_weight": "balanced", "random_state": config.seed}
    preprocessing = protocol["Models"]["Preprocessing"]
    if preprocessing["Selector"] != expected_selector or preprocessing["Model"] != expected_model:
        raise ValueError("Protocol preprocessing or estimator differs from the supported extracted recipe")
    if protocol["Allocation"]["One_Way_Cost"] not in protocol["Inference"]["Sensitivity"]["One_Way_Costs"]:
        raise ValueError("The cost grid must include the primary portfolio cost")
    history = build_eligible_sample(dataset, HistoryRequirements(training_years=recipe["Training_Years"],
        feature_warmup_sessions=200, calibration_sessions=126, label_horizon_sessions=recipe["Horizon"], minimum_training_rows=756))
    if history.eligible.empty:
        raise ValueError("Dataset has no complete eligible prediction history")
    first_score = history.eligible.Session.iloc[0]
    scores, audits = fit_scores(bars, features, recipe, start=first_score, end=config.end,
        checkpoint_directory=project/"checkpoints/models", config=model_config, progress=progress)
    allocation = allocate(scores, market_overlays(bars), AllocationConfig.from_spec(protocol["Allocation"]["Overlays"]))
    sessions = allocation.Decision_Time.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    mask = ((sessions >= config.start) & (sessions < config.end) & allocation.Risk_Percentile.notna()
            & allocation.Execution_Time.notna() & allocation.Return_End_Time.notna())
    evaluated = allocation.loc[mask].copy()
    if evaluated.empty:
        raise ValueError("No complete evaluation intervals after score warmup")
    features.to_parquet(output/"features.parquet")
    scores.to_parquet(output/"scores.parquet", index=False)
    labels = build_labels(bars, LabelConfig(horizon=recipe["Horizon"]))
    labels.to_parquet(output/"labels.parquet")
    atomic_write(output/"refit_audits.json", canonical_bytes(audits)+b"\n")
    evaluated.to_parquet(output/"evaluated_decisions.parquet", index=False)
    portfolios, metric_rows = {}, []
    target_columns = ["Strategy_ID", "Decision_Time", "Execution_Time", "Target_Weight", "Rebalance_Policy", "Rule_Version"]
    sma = market_overlays(bars).loc[evaluated.Row_ID, "SMA_50_200"].to_numpy(float)
    for name, weights in [("Fixed_Overlay", evaluated.Target_Weight.to_numpy()), ("Buy_And_Hold", np.ones(len(evaluated))), ("SMA_50_200", sma)]:
        targets = evaluated[target_columns].copy()
        targets["Strategy_ID"], targets["Target_Weight"] = name, weights
        for cost in protocol["Inference"]["Sensitivity"]["One_Way_Costs"]:
            result = simulate_portfolio(bars, dataset.actions, dataset.cash_intervals, targets,
                ExecutionConfig(one_way_cost=cost, dividend_payment_policy="ex_date_credit_approximation",
                                end_time=evaluated.Return_End_Time.iloc[-1].isoformat()))
            destination = output/"portfolios"/name/f"cost_{int(round(cost*10000))}bps"
            save_portfolio_result(result, destination)
            verify_portfolio_result(destination)
            metric_rows.append({"Portfolio": name, "Cost_Bps": cost*10000, **portfolio_metrics(result)})
            if cost == protocol["Allocation"]["One_Way_Cost"]:
                portfolios[name] = result
    metrics = pd.DataFrame(metric_rows)
    primary = portfolios["Fixed_Overlay"].daily.set_index("Return_End_Time")[["Net_Return", "Net_Excess_Return"]]
    benchmarks = pd.DataFrame({name: result.daily.set_index("Return_End_Time").Net_Excess_Return
                              for name, result in portfolios.items() if name != "Fixed_Overlay"})
    family = primary[["Net_Excess_Return"]].rename(columns={"Net_Excess_Return": protocol["Allocation"]["Candidates"][0]})
    inference = analyze_returns(primary, benchmarks, family, InferenceConfig(registered_trials=1,
        seed=config.seed, selection_scope="fixed_candidate"))
    inference.bootstrap_summary.to_csv(output/"bootstrap.csv", index=False)
    inference.max_t.to_csv(output/"multiple_testing.csv", index=False)
    inference.dsr_inputs.to_csv(output/"dsr_inputs.csv", index=False)
    daily = pd.concat([result.daily.assign(Portfolio=name) for name, result in portfolios.items()], ignore_index=True)
    # Prediction rows retain realized outcomes separately from causal input/cache keys.
    diagnostic = pd.DataFrame({"Risk_Score": evaluated.Risk_Score.to_numpy(),
                              "Realized_Downside": labels.loc[evaluated.Row_ID, "Binary_Downside"].to_numpy()}).dropna()
    from sklearn.metrics import roc_auc_score
    predictive = pd.DataFrame([{"Sample": "Evaluated decision rows with fully matured downside labels", "Observations": len(diagnostic),
        "Brier": float(np.mean((diagnostic.Realized_Downside-diagnostic.Risk_Score)**2)) if len(diagnostic) else None,
        "ROC_AUC": roc_auc_score(diagnostic.Realized_Downside, diagnostic.Risk_Score) if diagnostic.Realized_Downside.nunique() == 2 else None}])
    summary = {"Mode": "fit", "Stage": "development", "Instrument": protocol["Universe"]["Primary"],
        "Dataset_ID": config.dataset_id, "Dataset_Source": "project" if dataset_root == project/"datasets" else "baseline",
        "Protocol_SHA256": content_hash(protocol), "Recipe": recipe, "Candidate_ID": protocol["Allocation"]["Candidates"][0],
        "Observations": len(primary), "First_Decision": evaluated.Decision_Time.iloc[0].isoformat(),
        "Last_Return_End": evaluated.Return_End_Time.iloc[-1].isoformat(), "Model_Config": asdict(model_config),
        "Inference_Config": inference.inference["Config"], "Primary_One_Way_Cost": protocol["Allocation"]["One_Way_Cost"],
        "Feature_SHA256": frame_hash(features), "Refits": len(audits), "Forward_Observations": 0,
        "Interpretation": "Fixed-recipe package refit on inspected SPY history. Evidence is developmental.",
        "Chart_Caption": f"SPY | {evaluated.Decision_Time.iloc[0].date()}–{evaluated.Return_End_Time.iloc[-1].date()} inspected development\n"
                         "net of 5 bps one-way costs | distributions and lagged DFF cash proxy"}
    _write_tables(output, metrics, inference.bootstrap_summary, predictive, daily, summary)


def run_research(project, output, config, progress=None):
    project, output = Path(project).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh study directory")
    output.mkdir(parents=True)
    if config.mode == "replay":
        _replay(project, output, config)
    else:
        _fit(project, output, config, progress)
    verify_study(output, rtol=config.numerical_rtol, atol=config.numerical_atol, project=project)
    return load_study(output)


def verify_study(output, rtol=1e-10, atol=1e-12, project=None):
    output = Path(output)
    receipt = json.loads((output/"study_manifest.json").read_text(encoding="utf-8"))
    for name, digest in receipt["Files"].items():
        path = (output/name).resolve()
        if not path.is_relative_to(output.resolve()) or file_hash(path) != digest:
            raise ValueError(f"Study artifact changed at {name}")
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(output/"chart_data.csv")
    if summary["Mode"] == "replay":
        group = daily.loc[daily.Portfolio.eq("Nested_Selected")]
        sharpe = group.Net_Excess_Return.mean()/group.Net_Excess_Return.std(ddof=1)*np.sqrt(252)
        if not np.isclose(sharpe, summary["Sharpe"], rtol=rtol, atol=atol):
            raise ValueError("Replayed Sharpe differs from the independent saved return stream")
    else:
        project = Path(project or Path.cwd()).resolve()
        dataset_root = project/"datasets" if summary["Dataset_Source"] == "project" else materialize_baseline(project)/"datasets"
        dataset = load_dataset(summary["Dataset_ID"], root=dataset_root)
        features = build_features(dataset.bars)
        if frame_hash(features) != summary["Feature_SHA256"]:
            raise ValueError("Feature matrix differs from independent data reconstruction")
        pd.testing.assert_frame_equal(pd.read_parquet(output/"features.parquet"), features,
                                      check_exact=False, rtol=rtol, atol=atol)
        computed = []
        primary_portfolios = {}
        for directory in (output/"portfolios").glob("*/*"):
            result = load_portfolio(directory)
            cost = result.accounting_reconciliation["Config"]["one_way_cost"]
            computed.append({"Portfolio": directory.parent.name, "Cost_Bps": cost*10000, **portfolio_metrics(result)})
            if cost == summary["Primary_One_Way_Cost"]:
                primary_portfolios[directory.parent.name] = result
        actual_metrics = pd.DataFrame(computed).sort_values(["Portfolio", "Cost_Bps"]).reset_index(drop=True)
        saved_metrics = pd.read_csv(output/"metrics.csv", float_precision="round_trip").sort_values(["Portfolio", "Cost_Bps"]).reset_index(drop=True)
        _compare_saved_table(saved_metrics, actual_metrics[saved_metrics.columns], rtol=rtol, atol=atol)
        labels = pd.read_parquet(output/"labels.parquet")
        rebuilt_labels = build_labels(dataset.bars, LabelConfig(horizon=summary["Recipe"]["Horizon"]))
        pd.testing.assert_frame_equal(labels, rebuilt_labels, check_exact=False, rtol=rtol, atol=atol)
        audits = json.loads((output/"refit_audits.json").read_text(encoding="utf-8"))
        rows = labels.assign(Row_ID=labels.index, Available_At=labels.Decision_Time)
        for audit in audits:
            validate_refit(rows.loc[audit["Base_IDs"]], rows.loc[audit["Calibration_IDs"]], audit["Refit_Time"])
        primary = primary_portfolios.pop("Fixed_Overlay").daily.set_index("Return_End_Time")[["Net_Return", "Net_Excess_Return"]]
        benchmarks = pd.DataFrame({name: primary_portfolios[name].daily.set_index("Return_End_Time").Net_Excess_Return
                                  for name in ["Buy_And_Hold", "SMA_50_200"]})
        family = primary[["Net_Excess_Return"]].rename(columns={"Net_Excess_Return": summary["Candidate_ID"]})
        recomputed = analyze_returns(primary, benchmarks, family, InferenceConfig(**summary["Inference_Config"]))
        for actual, name in [(recomputed.bootstrap_summary, "bootstrap"), (recomputed.dsr_inputs, "dsr_inputs"),
                             (recomputed.max_t, "multiple_testing")]:
            saved = pd.read_csv(output/f"{name}.csv", float_precision="round_trip")
            _compare_saved_table(saved, actual, rtol=rtol, atol=atol)
    return {"Verified": True, "Mode": summary["Mode"], "Observations": summary["Observations"]}

"""Rebuild registered allocation rules from frozen scores with Layer 2 accounting."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .accounting_replay import verify_accounting_replay
from .data import load_dataset
from .execution import ExecutionConfig, simulate_portfolio


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _strategy_functions(source, predictions, market, registry):
    """Load only named function definitions from the verified historical source."""
    names = {"sharpe_causal_percentile", "sharpe_make_ensemble", "sharpe_step_multiplier",
             "sharpe_overlay_backtest", "sharpe_recompute_returns"}
    namespace = dict(np=np, pd=pd, SHARPE_SCORE_LOOKBACK_DAYS=1260, SHARPE_SCORE_MIN_DAYS=252,
        SHARPE_RISK_FREE_ANNUAL=registry["Risk_Free_Annual_Assumption"], ANNUALIZATION_DAYS=252,
        sharpe_predictor_predictions=predictions, advanced_table=market,
        equity_drawdown=lambda equity: equity/equity.cummax().clip(lower=1)-1)
    definitions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(definitions) != len(names) or {node.name for node in definitions} != names:
        raise ValueError("Historical strategy definitions do not match the supported adapter")
    for definition in definitions:
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "<verified-strategy-definition>", "exec"), namespace)
    return namespace


def _family_positions(project, registry):
    scores = pd.read_csv(project/"model_outputs/30_sharpe_predictor_scores.csv",
        parse_dates=["Date", "Execution_Date", "Return_End_Date", "Label_End_Date", "Refit_Date"])
    raw = pd.read_csv(project/"sp500_ohlcv.csv", parse_dates=["Date"])
    market = raw[["Date", "Close"]].copy()
    market["Sharpe_Execution_Volatility_60"] = raw.Close.pct_change(fill_method=None).ewm(
        span=60, adjust=False, min_periods=60).std()*np.sqrt(252)
    functions = _strategy_functions((project/"sharpe_improvement_research.py").read_text(encoding="utf-8"),
                                    scores, market, registry)
    ensemble_frames = [functions["sharpe_make_ensemble"](name, members) for name, members in registry["Ensembles"].items()]
    all_scores = pd.concat([scores, *ensemble_frames], ignore_index=True)
    groups = {name: group.sort_values("Date").reset_index(drop=True) for name, group in all_scores.groupby("Recipe_ID")}
    template = groups[registry["Predictor_Recipes"][0]["Recipe_ID"]]
    nested = pd.read_csv(project/"model_outputs/35_sharpe_nested_daily_returns.csv", parse_dates=["Date"])
    dates = pd.DatetimeIndex(nested.Date)
    candidates = pd.read_csv(project/"model_outputs/32_sharpe_candidate_registry.csv")
    if len(candidates) != registry["Candidate_Trial_Count"] or candidates.Candidate_ID.duplicated().any():
        raise ValueError("Candidate registry count or identifiers are invalid")
    specs = {spec["Strategy_ID"]: spec for spec in registry["Strategy_Specs"]}
    positions = {}
    for candidate in candidates.itertuples(index=False):
        if candidate.Score_ID == "Benchmark":
            source = template.copy()
            source["Position"] = (1.0 if candidate.Strategy_ID == "Buy_and_Hold"
                                   else source.SMA_50_200_Long_Signal.fillna(0))
        else:
            source = functions["sharpe_overlay_backtest"](groups[candidate.Score_ID], specs[candidate.Strategy_ID],
                                                          registry["One_Way_Cost"])
        values = source.set_index("Date").Position.reindex(dates)
        if values.isna().any() or not values.between(0, 1).all():
            raise ValueError(f"Incomplete reconstructed positions for {candidate.Candidate_ID}")
        positions[candidate.Candidate_ID] = values.to_numpy(float)
    positions = pd.DataFrame(positions, index=dates)
    selections = pd.read_csv(project/"model_outputs/34_sharpe_outer_selections.csv",
                             parse_dates=["Outer_Start", "Outer_End"])
    reconstructed = np.full(len(dates), np.nan)
    for row in selections.itertuples(index=False):
        mask = (dates >= row.Outer_Start) & (dates < row.Outer_End)
        reconstructed[mask] = positions.loc[mask, row.Selected_Candidate]
    if not np.allclose(reconstructed, nested.Position, rtol=0, atol=1e-12):
        raise ValueError("Reconstructed selected targets differ from the saved selector")
    return positions, nested


def load_trial_family(project_dir, directory):
    project, directory = Path(project_dir).resolve(), Path(directory).resolve()
    manifest = json.loads((directory/"completion.json").read_text(encoding="utf-8"))
    for base, hashes in [(project, manifest["Source_SHA256"]), (directory, manifest["Files"])]:
        for name, expected in hashes.items():
            path = (base/name).resolve()
            if not path.is_relative_to(base) or _digest(path) != expected:
                raise ValueError(f"Trial-family checksum mismatch for {name}")
    returns = pd.read_parquet(directory/"trial_returns.parquet")
    portfolio = pd.read_parquet(directory/"portfolio_returns.parquet")
    benchmarks = pd.read_parquet(directory/"benchmark_returns.parquet")
    if (not returns.index.equals(portfolio.index) or not returns.index.equals(benchmarks.index)
            or returns.index.has_duplicates or len(returns.columns) != manifest["Registered_Trials"]
            or not np.isfinite(returns.to_numpy(float)).all()):
        raise ValueError("Trial-family dates, coverage, or finite-return checks failed")
    return portfolio, benchmarks, returns, manifest


def rebuild_trial_family(project_dir, output, accounting_dir="runs/layer2_accounting_replay"):
    project, output = Path(project_dir).resolve(), Path(output).resolve()
    if output.exists():
        return load_trial_family(project, output)
    accounting = project/accounting_dir
    verify_accounting_replay(project, accounting)
    catalog = json.loads((project/"datasets/catalog.json").read_text(encoding="utf-8"))
    dataset_id = next(record["Dataset_ID"] for record in catalog["Snapshots"] if record["Name"] == "historical_index")
    dataset = load_dataset(dataset_id, root=project/"datasets")
    registry = json.loads((project/"model_outputs/28_sharpe_experiment_registry.json").read_text(encoding="utf-8"))
    paths = [project/"sharpe_improvement_research.py", project/"sp500_ohlcv.csv",
             project/"model_outputs/46_sharpe_run_manifest.json", project/"datasets"/dataset_id/"manifest.json",
             accounting/"completion.json", Path(__file__), Path(__file__).with_name("execution.py")]
    paths.extend(project/"model_outputs"/name for name in ["28_sharpe_experiment_registry.json",
        "30_sharpe_predictor_scores.csv", "32_sharpe_candidate_registry.csv", "34_sharpe_outer_selections.csv",
        "35_sharpe_nested_daily_returns.csv"])
    source_hashes = {str(path.relative_to(project).as_posix()): _digest(path) for path in paths}
    positions, nested = _family_positions(project, registry)
    indexed = dataset.bars.set_index("Session")
    decisions = nested.Date.dt.strftime("%Y-%m-%d").map(indexed.Available_At)
    executions = nested.Execution_Date.map(indexed.Open_Time)
    ending = nested.Return_End_Date.map(indexed.Open_Time)
    end = ending.iloc[-1]
    start_decision = nested.Date.iloc[0].strftime("%Y-%m-%d")
    bars = dataset.bars.loc[dataset.bars.Session.ge(start_decision) & dataset.bars.Open_Time.le(end)]
    # Pre-2007 scores determine causal rank and state-machine warmup. Each
    # evaluated portfolio starts in cash at the same first execution open.
    returns, errors = {}, []
    for number, (name, weights) in enumerate(positions.items(), 1):
        targets = pd.DataFrame(dict(Strategy_ID=name, Rule_Version="frozen_candidate",
            Decision_Time=decisions.to_numpy(), Execution_Time=executions.to_numpy(),
            Target_Weight=weights.to_numpy(), Rebalance_Policy="daily_target"))
        result = simulate_portfolio(bars, dataset.actions, dataset.cash_intervals, targets,
                                    ExecutionConfig(one_way_cost=registry["One_Way_Cost"], end_time=end.isoformat()))
        if not pd.DatetimeIndex(result.daily.Return_End_Time).equals(pd.DatetimeIndex(ending)):
            raise ValueError("Candidate ledger return dates differ from the common sample")
        returns[name] = result.daily.Net_Excess_Return.to_numpy()
        errors.append(result.accounting_reconciliation["Maximum_Relative_Wealth_Identity_Error"])
        if number % 10 == 0 or number == len(positions.columns):
            print(f"Rebuilt {number}/{len(positions.columns)} registered trial ledgers", flush=True)
    return_index = pd.DatetimeIndex(ending, name="Return_End_Time")
    returns = pd.DataFrame(returns, index=return_index)
    portfolio = pd.read_parquet(accounting/"index_Nested_Selected/cost_5bps/portfolio_daily.parquet").set_index("Return_End_Time")
    hgb = pd.read_parquet(accounting/"index_Hist_Gradient_Boosting/cost_5bps/portfolio_daily.parquet").set_index("Return_End_Time")
    if not portfolio.index.equals(return_index) or not hgb.index.equals(return_index):
        raise ValueError("Layer 2 portfolio dates do not match the rebuilt trial family")
    benchmarks = pd.DataFrame(dict(Locked_HGB=hgb.Net_Excess_Return.to_numpy(),
        SMA_50_200=returns["Benchmark__SMA_50_200"].to_numpy(), Market=nested.Tradable_Return_1d.to_numpy()), index=return_index)
    portfolio = portfolio[["Net_Return", "Net_Excess_Return"]]
    if any(_digest(path) != source_hashes[str(path.relative_to(project).as_posix())] for path in paths):
        raise RuntimeError("Trial-family inputs changed during reconstruction")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pending-trials-", dir=output.parent) as temporary:
        pending = Path(temporary)/output.name
        pending.mkdir()
        for name, frame in [("trial_returns", returns), ("portfolio_returns", portfolio),
                             ("benchmark_returns", benchmarks), ("trial_positions", positions)]:
            frame.to_parquet(pending/f"{name}.parquet", index=True)
        manifest = dict(Schema_Version="1.0.0", Source_SHA256=source_hashes, Dataset_ID=dataset_id,
            Registered_Trials=registry["Candidate_Trial_Count"], Rows=len(returns),
            Maximum_Relative_Ledger_Identity_Error=max(errors), Accounting_Start="common_initial_cash_at_first_evaluation_open",
            Selector_Target_Equality="validated_against_saved_outer_selections_and_positions",
            Files={path.name: _digest(path) for path in pending.iterdir()})
        (pending/"completion.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        pending.rename(output)
    return load_trial_family(project, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    _, _, returns, _ = rebuild_trial_family(arguments.project_dir, arguments.output)
    print(f"Validated {len(returns.columns)} trial return streams over {len(returns)} dates")


if __name__ == "__main__":
    main()

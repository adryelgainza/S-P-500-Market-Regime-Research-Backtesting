"""Offline Layer 2 replay of frozen targets and investable accounting checks."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from .data import HistoryRequirements, build_eligible_sample, load_dataset
from .execution import (ExecutionConfig, portfolio_metrics, save_portfolio_result,
                        simulate_portfolio, verify_portfolio_result)


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixed_targets(frame, dataset, name):
    indexed = dataset.bars.set_index("Session")
    decision = frame.Date.map(indexed.Available_At)
    execution = frame.Execution_Date.map(indexed.Open_Time)
    end = frame.Return_End_Date.map(indexed.Open_Time)
    if decision.isna().any() or execution.isna().any() or end.isna().any():
        raise ValueError("Frozen targets reference missing dataset dates")
    if not end.iloc[:-1].reset_index(drop=True).equals(execution.iloc[1:].reset_index(drop=True)):
        raise ValueError("Frozen target return intervals are discontinuous")
    versions = frame.Selected_Candidate if "Selected_Candidate" in frame else frame.Config_ID
    targets = pd.DataFrame(dict(Strategy_ID=name, Rule_Version=versions.to_numpy(),
                               Decision_Time=decision.to_numpy(), Execution_Time=execution.to_numpy(),
                               Target_Weight=frame.Position.to_numpy(), Rebalance_Policy="daily_target"))
    return targets, end.iloc[-1]


def _comparison(frame, result, cost, name):
    weights = frame.Position.to_numpy(float)
    turnover = np.abs(np.diff(weights, prepend=0.0))
    old_cost = turnover*cost
    old_returns = weights*frame.Tradable_Return_1d.to_numpy(float) - old_cost
    standard_deviation = np.std(old_returns, ddof=1)
    old_sharpe = float(np.mean(old_returns)/standard_deviation*np.sqrt(252)) if standard_deviation > 1e-15 else None
    metrics = portfolio_metrics(result)
    old_total = float(np.prod(1+old_returns)-1)
    old_turnover = float(turnover.sum()*252/len(frame))
    return dict(Series=name, One_Way_Cost_Bps=cost*10000, Periods=len(frame),
                Legacy_Total_Return=old_total, Ledger_Total_Return=metrics["Net_Total_Return"],
                Delta_Total_Return=metrics["Net_Total_Return"]-old_total,
                Legacy_Sharpe=old_sharpe, Ledger_Sharpe=metrics["Sharpe_Ratio"],
                Delta_Sharpe=metrics["Sharpe_Ratio"]-old_sharpe if old_sharpe is not None else None,
                Legacy_Annual_Turnover=old_turnover, Ledger_Annual_Turnover=metrics["Annual_Turnover"],
                Delta_Annual_Turnover=metrics["Annual_Turnover"]-old_turnover,
                Legacy_Cost_Equity_Fractions=float(old_cost.sum()),
                Ledger_Cost_Equity_Fractions=metrics["Cost_Equity_Fractions"],
                Delta_Cost_Equity_Fractions=metrics["Cost_Equity_Fractions"]-old_cost.sum())


def run_accounting_replay(project_dir, output):
    project_dir, output = Path(project_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Accounting replay output already exists at {output}")
    verification = subprocess.run([sys.executable, str(project_dir/"verify_sharpe_research.py")],
                                  cwd=project_dir, check=True, capture_output=True, text=True)
    catalog = json.loads((project_dir/"datasets/catalog.json").read_text(encoding="utf-8"))
    ids = {record["Name"]: record["Dataset_ID"] for record in catalog["Snapshots"]}
    index = load_dataset(ids["historical_index"], root=project_dir/"datasets")
    spy = load_dataset(ids["investable_spy"], root=project_dir/"datasets")
    nested_path = project_dir/"model_outputs/35_sharpe_nested_daily_returns.csv"
    models_path = project_dir/"model_outputs/23_advanced_daily_strategies.csv"
    nested = pd.read_csv(nested_path)
    advanced = pd.read_csv(models_path)
    frames = {"Nested_Selected": nested}
    for model, frame in advanced.groupby("Model", sort=True):
        if frame.Strategy.nunique() != 1 or frame.Config_ID.nunique() != 1:
            raise ValueError("Replay requires one validation-locked strategy per model")
        frames[model] = frame.reset_index(drop=True)
    comparisons, metrics_rows = [], []
    source_paths = [nested_path, models_path, project_dir/"model_outputs/46_sharpe_run_manifest.json",
                    project_dir/"datasets/catalog.json"]
    source_paths.extend(sorted((project_dir/"src/regime_research").rglob("*.py")))
    source_paths.extend(project_dir/"datasets"/dataset_id/"manifest.json" for dataset_id in ids.values())
    source_hashes = {str(path.relative_to(project_dir).as_posix()): _hash_file(path) for path in source_paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pending-accounting-", dir=output.parent) as temporary:
        pending = Path(temporary)/output.name
        pending.mkdir()
        for name, frame in frames.items():
            targets, end = _fixed_targets(frame, index, name)
            for bps in [0, 5, 10, 20]:
                result = simulate_portfolio(index.bars, index.actions, index.cash_intervals, targets,
                                            ExecutionConfig(one_way_cost=bps/10000, end_time=end.isoformat()))
                save_portfolio_result(result, pending/f"index_{name}"/f"cost_{bps}bps")
                comparisons.append(_comparison(frame, result, bps/10000, name))
                metrics_rows.append(dict(Track="historical_index_fixed_targets", Series=name,
                                          One_Way_Cost_Bps=bps, **portfolio_metrics(result)))
            print(f"Completed index accounting replay for {name}", flush=True)
        eligible = build_eligible_sample(spy, HistoryRequirements()).eligible
        if eligible.empty:
            raise ValueError("No eligible SPY accounting-check sample")
        start, end = eligible.Execution_Time.iloc[0], eligible.Return_End_Time.iloc[-1]
        bars = spy.bars
        indices = np.flatnonzero(bars.Open_Time.ge(start) & bars.Open_Time.lt(end))
        for name, weight in [("SPY_Buy_And_Hold", 1.0), ("SPY_All_Cash", 0.0)]:
            targets = pd.DataFrame(dict(Strategy_ID=name, Rule_Version="accounting_check_v1",
                Decision_Time=bars.Available_At.iloc[indices-1].to_list(),
                Execution_Time=bars.Open_Time.iloc[indices].to_list(), Target_Weight=weight,
                Rebalance_Policy="signal_change_only"))
            for bps in [0, 5, 10, 20]:
                result = simulate_portfolio(spy.bars, spy.actions, spy.cash_intervals, targets,
                    ExecutionConfig(one_way_cost=bps/10000, dividend_payment_policy="ex_date_credit_approximation",
                                    end_time=end.isoformat()))
                save_portfolio_result(result, pending/name/f"cost_{bps}bps")
                metrics_rows.append(dict(Track="investable_accounting_check", Series=name,
                                          One_Way_Cost_Bps=bps, **portfolio_metrics(result)))
            print(f"Completed investable accounting check for {name}", flush=True)
        comparison_table = pd.DataFrame(comparisons)
        comparison_table.to_csv(pending/"baseline_accounting_comparison.csv", index=False)
        pd.DataFrame(metrics_rows).to_csv(pending/"cost_sensitivity.csv", index=False)
        assumptions = dict(
            Index_Track="Frozen historical-index targets replayed on their original instrument and sample.",
            Investable_Track="Buy-and-hold and all-cash accounting checks over the default eligible SPY period.",
            Cash_Policy=spy.manifest["Cash_Policy"],
            Dividend_Policy="SPY payment dates use the registered ex-date-credit approximation.",
            Open_Fill="Recorded open price with a proportional one-way cost on actual traded notional.",
            Rebalance_Policy="Index uses daily target weights. SPY checks retain shares between signals.",
            Cost_Comparison="Cost-equity fractions sum actual dollar cost divided by pre-trade wealth on each trade.",
        )
        (pending/"assumptions.json").write_text(json.dumps(assumptions, indent=2)+"\n", encoding="utf-8")
        if any(source_hashes[str(path.relative_to(project_dir).as_posix())] != _hash_file(path) for path in source_paths):
            raise RuntimeError("Research inputs or source changed during replay")
        manifest = dict(Schema_Version="1.0.0", Completed_UTC=datetime.now(timezone.utc).isoformat(),
            Dataset_IDs=ids, Source_SHA256=source_hashes, Existing_Research_Verification=verification.stdout.strip(),
            Cost_Grid_Bps=[0,5,10,20], Portfolio_Runs=len(metrics_rows),
            Files={str(path.relative_to(pending).as_posix()): _hash_file(path)
                   for path in sorted(pending.rglob("*")) if path.is_file()})
        (pending/"completion.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        pending.rename(output)
    return comparison_table


def verify_accounting_replay(project_dir, output):
    project_dir, output = Path(project_dir).resolve(), Path(output).resolve()
    manifest = json.loads((output/"completion.json").read_text(encoding="utf-8"))
    for base, hashes in [(project_dir, manifest["Source_SHA256"]), (output, manifest["Files"])]:
        for name, expected in hashes.items():
            path = (base/name).resolve()
            if not path.is_relative_to(base) or _hash_file(path) != expected:
                raise ValueError(f"Accounting replay checksum mismatch for {name}")
    for dataset_id in manifest["Dataset_IDs"].values():
        load_dataset(dataset_id, root=project_dir/"datasets")
    receipts = sorted(output.glob("*/cost_*bps/completion.json"))
    if len(receipts) != manifest["Portfolio_Runs"]:
        raise ValueError("Accounting replay portfolio count differs from its receipt")
    for receipt in receipts:
        verify_portfolio_result(receipt.parent)
    return dict(Verified_Portfolios=len(receipts), Dataset_IDs=manifest["Dataset_IDs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output")
    action.add_argument("--verify")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_accounting_replay(args.project_dir, args.verify), indent=2))
        return
    comparison = run_accounting_replay(args.project_dir, args.output)
    print(comparison.loc[comparison.One_Way_Cost_Bps.eq(5),
                          ["Series", "Legacy_Sharpe", "Ledger_Sharpe", "Delta_Annual_Turnover"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""Run Layer 4 controls, attribution, randomized references, and notebook charts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .attribution import attribute, load_portfolio, predictive_diagnostics
from .controls import build_controls, shuffle_overlay_blocks, shuffled_score_null
from .data import load_dataset
from .execution import ExecutionConfig, portfolio_metrics, save_portfolio_result, simulate_portfolio, verify_portfolio_result
from .statistical_research import _compare_saved_table
from .trial_returns import _strategy_functions, load_trial_family


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _targets(evaluation, weights, name, provenance="past_only_control"):
    return pd.DataFrame(dict(Strategy_ID=name, Rule_Version=evaluation.Fold.map(lambda fold: f"fold_{fold}").to_numpy(),
        Decision_Time=evaluation.Decision_Time.to_numpy(), Execution_Time=evaluation.Execution_Time.to_numpy(),
        Target_Weight=np.asarray(weights, dtype=float), Rebalance_Policy="daily_target", Target_Provenance=provenance))


def _simulate(dataset, evaluation, weights, name, protocol, provenance="past_only_control"):
    start = evaluation.Date.iloc[0]
    end = evaluation.Return_End_Time.iloc[-1]
    bars = dataset.bars.loc[pd.to_datetime(dataset.bars.Session).ge(start) & dataset.bars.Open_Time.le(end)]
    return simulate_portfolio(bars, dataset.actions, dataset.cash_intervals,
        _targets(evaluation, weights, name, provenance),
        ExecutionConfig(one_way_cost=protocol["One_Way_Cost"], end_time=end.isoformat()))


def _prepare_control_data(project, dataset, protocol):
    registry = json.loads((project/"model_outputs/28_sharpe_experiment_registry.json").read_text(encoding="utf-8"))
    nested = pd.read_csv(project/"model_outputs/35_sharpe_nested_daily_returns.csv",
        parse_dates=["Date", "Execution_Date", "Return_End_Date"])
    selections = pd.read_csv(project/"model_outputs/34_sharpe_outer_selections.csv",
                              parse_dates=["Outer_Start", "Outer_End"])
    predictions = pd.read_csv(project/"model_outputs/30_sharpe_predictor_scores.csv",
        parse_dates=["Date", "Execution_Date", "Return_End_Date", "Label_End_Date", "Refit_Date"])
    bars = dataset.bars
    market = pd.DataFrame(dict(Date=pd.to_datetime(bars.Session), Close=bars.Close.to_numpy()))
    changes = market.Close.pct_change(fill_method=None)
    market["Sharpe_Execution_Volatility_60"] = changes.ewm(span=60, adjust=False, min_periods=60).std()*np.sqrt(252)
    market["Volatility"] = changes.ewm(span=protocol["Volatility_Window"], adjust=False,
                                     min_periods=protocol["Volatility_Window"]).std()*np.sqrt(252)
    market["SMA_Signal"] = market.Close.rolling(50).mean().gt(market.Close.rolling(200).mean()).astype(float)
    functions = _strategy_functions((project/"sharpe_improvement_research.py").read_text(encoding="utf-8"),
                                    predictions, market, registry)
    ensembles = [functions["sharpe_make_ensemble"](name, members) for name, members in registry["Ensembles"].items()]
    scores = pd.concat([predictions, *ensembles], ignore_index=True)
    groups = {name: frame.sort_values("Date").reset_index(drop=True) for name, frame in scores.groupby("Recipe_ID")}
    candidates = pd.read_csv(project/"model_outputs/32_sharpe_candidate_registry.csv").set_index("Candidate_ID")
    specs = {spec["Strategy_ID"]: spec for spec in registry["Strategy_Specs"]}
    template = groups[registry["Predictor_Recipes"][0]["Recipe_ID"]]
    full = {}
    cores = {}
    for candidate in selections.Selected_Candidate.unique():
        row = candidates.loc[candidate]
        if row.Score_ID == "Benchmark":
            frame = template.copy()
            frame["Position"] = 1.0 if row.Strategy_ID == "Buy_and_Hold" else frame.SMA_50_200_Long_Signal
            core = frame.Position
        else:
            spec = specs[row.Strategy_ID]
            frame = functions["sharpe_overlay_backtest"](groups[row.Score_ID], spec, protocol["One_Way_Cost"])
            core = pd.Series(1., index=frame.index) if spec["Core"] == "Always_Long" else frame.SMA_50_200_Long_Signal
            if spec["Target_Volatility"] is not None:
                volatility = frame.Execution_Volatility if spec["Volatility_Window"] == 20 else frame.Date.map(
                    market.set_index("Date").Sharpe_Execution_Volatility_60)
                scaling = (spec["Target_Volatility"]/volatility).clip(0,1)
                core = core*((scaling/.1).round()*.1).clip(0,1)
        full[candidate] = frame
        cores[candidate] = pd.Series(core.to_numpy(), index=pd.DatetimeIndex(frame.Date))
    indexed = bars.set_index(pd.to_datetime(bars.Session))
    evaluation = pd.DataFrame(dict(Date=nested.Date, Decision_Time=nested.Date.map(indexed.Available_At),
        Execution_Time=nested.Execution_Date.map(indexed.Open_Time),
        Return_End_Time=nested.Return_End_Date.map(indexed.Open_Time), Fold=nested.Outer_Fold,
        Position=nested.Position, Core_Weight=np.nan,
        Volatility=nested.Date.map(market.set_index("Date").Volatility),
        SMA_Signal=nested.Date.map(market.set_index("Date").SMA_Signal)))
    history, folds, historical_ledgers = [], [], {}
    for fold in selections.itertuples(index=False):
        boundary = fold.Outer_Start.tz_localize("America/New_York").tz_convert("UTC")
        folds.append(dict(Fold=fold.Outer_Fold, Decision_Time=boundary, Selected_Candidate=fold.Selected_Candidate))
        source = full[fold.Selected_Candidate]
        window_start = fold.Outer_Start-pd.DateOffset(years=protocol["Matching_Window_Years"])
        prior = source.loc[source.Date.ge(window_start) & source.Return_End_Date.lt(fold.Outer_Start)].copy()
        historical = pd.DataFrame(dict(Date=prior.Date, Decision_Time=prior.Date.map(indexed.Available_At),
            Execution_Time=prior.Execution_Date.map(indexed.Open_Time),
            Return_End_Time=prior.Return_End_Date.map(indexed.Open_Time), Fold=fold.Outer_Fold))
        if historical.empty:
            raise ValueError("A matching fold has no matured prior observations")
        result = _simulate(dataset, historical, prior.Position, f"matching_fold_{fold.Outer_Fold}", protocol)
        historical_ledgers[fold.Outer_Fold] = result
        history.append(pd.DataFrame(dict(Fold=fold.Outer_Fold, Candidate_ID=fold.Selected_Candidate,
            Decision_Time=historical.Decision_Time.to_numpy(), Return_End_Time=result.daily.Return_End_Time.to_numpy(),
            Target_Weight=prior.Position.to_numpy(), Net_Excess_Return=result.daily.Net_Excess_Return.to_numpy(),
            Market_Excess_Return=prior.Tradable_Return_1d.to_numpy())))
        mask = evaluation.Fold.eq(fold.Outer_Fold)
        evaluation.loc[mask, "Core_Weight"] = evaluation.loc[mask, "Date"].map(cores[fold.Selected_Candidate])
        reconstructed = evaluation.loc[mask, "Date"].map(source.set_index("Date").Position)
        if not np.allclose(reconstructed, evaluation.loc[mask, "Position"], atol=1e-12, rtol=0):
            raise ValueError("Attribution reconstruction differs from the frozen nested allocation")
    if not evaluation.Position.le(evaluation.Core_Weight+1e-12).all():
        raise ValueError("The model overlay exceeds its registered underlying core")
    return (dict(evaluation=evaluation, history=pd.concat(history, ignore_index=True), folds=pd.DataFrame(folds)),
            predictions, historical_ledgers)


def _market_environment(dataset, evaluation):
    bars = dataset.bars
    indexed = bars.set_index("Open_Time")
    starts = pd.DatetimeIndex(evaluation.Execution_Time)
    ends = pd.DatetimeIndex(evaluation.Return_End_Time, name="Return_End_Time")
    raw_return = indexed.Open.reindex(ends).to_numpy()/indexed.Open.reindex(starts).to_numpy()-1
    cash = dataset.cash_intervals.set_index("Start_Time").reindex(starts)
    return pd.DataFrame(dict(Market_Return=raw_return, Market_Excess_Return=raw_return-cash.Reference_Return.to_numpy(),
        Reference_Return=cash.Reference_Return.to_numpy(), Credited_Return=cash.Credited_Return.to_numpy(),
        Fold=evaluation.Fold.to_numpy()), index=ends)


def _prediction_tables(predictions, dataset, evaluation, protocol):
    indexed = dataset.bars.set_index(pd.to_datetime(dataset.bars.Session))
    frame = predictions.loc[predictions.Date.ge(evaluation.Date.iloc[0])
        & predictions.Date.le(evaluation.Date.iloc[-1])].copy()
    frame["Decision_Time"] = frame.Date.map(indexed.Available_At)
    frame["Label_End_Time"] = frame.Label_End_Date.map(indexed.Open_Time)
    return predictive_diagnostics(frame, protocol, evaluation_end_time=evaluation.Return_End_Time.iloc[-1])


def _plots(output, portfolios, results, market, reference_name):
    directory = output/"figures"
    directory.mkdir()
    with plt.rc_context({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        primary = results.attribution.set_index("Portfolio").loc[reference_name]
        components = [primary.Initial_Capital, primary.Price_PnL, primary.Distribution_PnL,
                      primary.Cash_Interest, -primary.Trading_Cost, primary.Final_Wealth]
        bottoms = [0, components[0], sum(components[:2]), sum(components[:3]), sum(components[:4]), 0]
        figure, axis = plt.subplots(figsize=(10,4.5))
        axis.bar(np.arange(6), components, bottom=bottoms, color=["#555555","#487698","#65917b","#65917b","#b77555","#555555"])
        axis.set_xticks(np.arange(6), ["Initial\nwealth","Equity price\nP&L","Distributions","Cash interest","Trading costs","Final\nwealth"])
        for number, (value, bottom) in enumerate(zip(components, bottoms)):
            axis.text(number, bottom+max(value,0)+.04, f"{value:.4f}", ha="center")
        highest = max(bottom+max(value,0) for value, bottom in zip(components,bottoms))
        axis.set_ylim(0,highest*1.18)
        axis.set_ylabel("Wealth per initial dollar")
        axis.set_title("Nested selector wealth bridge · historical index · 5 bps one-way costs")
        figure.tight_layout()
        figure.savefig(directory/"pnl_bridge.png", dpi=160)
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(10,4.5))
        for name, color in [(reference_name,"#315f85"),("Core_Without_ML","#ba794b"),
                            ("Past_Exposure_Match","#5f8c75"),("SMA_Only","#777777")]:
            data = portfolios[name].daily.set_index("Return_End_Time").Realized_Weight.resample("ME").mean()
            axis.plot(data.index, data, label=name.replace("_"," "), linewidth=1.3, color=color)
        axis.set_ylim(-.03,1.05)
        axis.set_ylabel("Monthly mean equity weight")
        axis.legend(loc="lower left", fontsize=8, ncol=2)
        axis.set_title("Exposure through time · common next-open execution")
        figure.tight_layout()
        figure.savefig(directory/"exposure.png", dpi=160)
        plt.close(figure)
        comparison = results.paired_comparisons.dropna(subset=["Delta_Sharpe", "Delta_Sharpe_Lower", "Delta_Sharpe_Upper"])
        figure, axis = plt.subplots(figsize=(9,5))
        for number, row in enumerate(comparison.itertuples(index=False)):
            axis.plot([row.Delta_Sharpe_Lower,row.Delta_Sharpe_Upper], [number,number], color="#666666")
            axis.scatter(row.Delta_Sharpe, number, color="#315f85", s=30)
        axis.set_yticks(np.arange(len(comparison)), comparison.Control.str.replace("_"," "))
        axis.axvline(0, color="#777777", linewidth=.8)
        axis.set_xlabel("Nested-selector Sharpe minus control Sharpe")
        axis.set_title("Paired 95% conditional intervals · 20-session blocks")
        figure.tight_layout()
        figure.savefig(directory/"paired_differences.png", dpi=160)
        plt.close(figure)
        figure, axes = plt.subplots(1,2,figsize=(11,4.4))
        folds = results.fold_contributions.loc[results.fold_contributions.Portfolio.eq(reference_name)]
        axes[0].bar(folds.Fold, folds.Log_Contribution, color="#487698")
        axes[0].set_xlabel("Outer fold")
        axes[0].set_ylabel("Log-return contribution")
        episodes = results.episodes.loc[results.episodes.Portfolio.eq(reference_name) & results.episodes.Episode_ID.gt(0)]
        axes[1].bar(episodes.Episode_ID, episodes.Log_Contribution, color="#65917b")
        axes[1].set_xlabel("Invested episode, including its exit cost")
        axes[1].set_ylabel("Log-return contribution")
        figure.suptitle("Nested selector fold and episode contributions")
        figure.tight_layout()
        figure.savefig(directory/"contributions.png", dpi=160)
        plt.close(figure)


def _notebook_code_digest(path):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = [{"id": cell["id"], "source": "".join(cell["source"]).rstrip()}
             for cell in notebook["cells"] if cell["cell_type"] == "code"]
    return hashlib.sha256(json.dumps(cells, sort_keys=True, default=str).encode()).hexdigest()


def _update_notebook(project, output):
    path = project/"sp500_market_regime_classification.ipynb"
    manifest = json.loads((output/"completion.json").read_text(encoding="utf-8"))
    if _notebook_code_digest(path) != manifest["Notebook_Code_SHA256"]:
        raise RuntimeError("Notebook code differs from the attribution run's recorded code")
    attribution = pd.read_csv(output/"attribution.csv").set_index("Portfolio")
    factors = pd.read_csv(output/"factor_regression.csv").set_index("Portfolio")
    comparison = pd.read_csv(output/"paired_comparisons.csv").set_index("Control")
    model, core = attribution.loc["Nested_Selected"], attribution.loc["Core_Without_ML"]
    factor = factors.loc["Nested_Selected"]
    relative = output.relative_to(project).as_posix()
    text = f"""## Layer 4 — Attribution and matched controls

This retrospective analysis replays the saved nested allocations and matched controls on the historical index. Each run uses the same 4,915 next-open intervals, zero historical cash rate, and 5 bps one-way trading cost. Matching parameters use only matured observations before each outer fold.

| Quantity | Result |
| --- | --- |
| Nested-selector Sharpe | {model.Sharpe_Ratio:.6f} |
| Core without ML Sharpe | {core.Sharpe_Ratio:.6f} |
| Nested mean equity exposure | {model.Average_Exposure:.2%} |
| Market beta | {factor.Beta:.4f} |
| Annualized regression intercept | {factor.Annualized_Alpha:.2%} |
| Intercept HAC t-statistic | {factor.Alpha_HAC_t:.3f} |
| Delta-Sharpe versus core, 95% interval | [{comparison.loc['Core_Without_ML','Delta_Sharpe_Lower']:.4f}, {comparison.loc['Core_Without_ML','Delta_Sharpe_Upper']:.4f}] |

Intervals condition on the saved decisions. Block-shuffled overlay results are retrospective reference distributions. The control registry records matching windows, seeds, accounting assumptions, and randomization scope. [Attribution guide](docs/ATTRIBUTION.md)

![Wealth bridge]({relative}/figures/pnl_bridge.png)

The bridge reconciles initial wealth, equity price changes, distributions, cash interest, and actual trading costs to final wealth.

![Exposure through time]({relative}/figures/exposure.png)

Each line shows monthly mean realized equity exposure under the common execution convention.

![Paired performance differences]({relative}/figures/paired_differences.png)

Points show nested-selector Sharpe minus control Sharpe. Lines show pointwise paired 95% intervals. The cash comparison uses mean returns because its excess-return variance is zero.

![Fold and episode contributions]({relative}/figures/contributions.png)

Log contributions add across folds and episodes. Each invested episode includes the first exit-cost period, and episode zero records aggregate uninvested gaps.
"""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cell = dict(cell_type="markdown", id="layer4-attribution-report", metadata={}, source=text.splitlines(keepends=True))
    existing = next((index for index, item in enumerate(notebook["cells"]) if item.get("id") == cell["id"]), None)
    if existing is None:
        notebook["cells"].append(cell)
    else:
        if notebook["cells"][existing]["cell_type"] != "markdown":
            raise ValueError("The attribution report ID belongs to a code cell")
        notebook["cells"][existing] = cell
    pending = path.with_suffix(".ipynb.pending")
    pending.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    if _notebook_code_digest(pending) != manifest["Notebook_Code_SHA256"]:
        raise RuntimeError("Attribution report update altered notebook code")
    pending.replace(path)


def run_attribution_research(project_dir, output, protocol_path="protocols/layer4_controls.json"):
    project, output = Path(project_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Attribution output already exists at {output}")
    protocol_file = project/protocol_path
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    _, _, _, family = load_trial_family(project, project/"runs/layer3_candidate_family")
    dataset = load_dataset(family["Dataset_ID"], root=project/"datasets")
    source_paths = [protocol_file, Path(__file__), Path(__file__).with_name("attribution.py"),
        Path(__file__).with_name("controls.py"), Path(__file__).with_name("execution.py"),
        Path(__file__).with_name("statistics.py"), Path(__file__).with_name("statistical_research.py"),
        Path(__file__).with_name("trial_returns.py"), project/"runs/layer3_candidate_family/completion.json",
        project/"runs/layer2_accounting_replay/completion.json", project/"sharpe_improvement_research.py",
        project/"model_outputs/30_sharpe_predictor_scores.csv", project/"model_outputs/28_sharpe_experiment_registry.json",
        project/"model_outputs/32_sharpe_candidate_registry.csv", project/"model_outputs/34_sharpe_outer_selections.csv",
        project/"model_outputs/35_sharpe_nested_daily_returns.csv", project/"datasets"/dataset.dataset_id/"manifest.json"]
    source_hashes = {str(path.relative_to(project).as_posix()): _digest(path) for path in source_paths}
    notebook_digest = _notebook_code_digest(project/"sp500_market_regime_classification.ipynb")
    data, predictions, historical_ledgers = _prepare_control_data(project, dataset, protocol)
    controls = build_controls(protocol, data)
    evaluation = data["evaluation"]
    null_draws, null_summary = shuffled_score_null(protocol)
    if not null_summary["Passed"]:
        raise ArithmeticError(f"Shuffled-score null failed its declared acceptance rule: {null_summary}")
    portfolios = {}
    for name in ["Nested_Selected", "Hist_Gradient_Boosting", "KNN", "Logistic", "Random_Forest"]:
        portfolios[name] = load_portfolio(project/"runs/layer2_accounting_replay"/f"index_{name}"/"cost_5bps")
    for name, weights in controls.weights.items():
        portfolios[name] = _simulate(dataset, evaluation, weights, name, protocol)
    market = _market_environment(dataset, evaluation)
    runtime_config = dict(protocol, Reference_Portfolio="Nested_Selected")
    results = attribute(portfolios, market, runtime_config)
    predictive, calibration, score_quantiles = _prediction_tables(predictions, dataset, evaluation, protocol)
    multiplier = np.divide(evaluation.Position, evaluation.Core_Weight,
                           out=np.ones(len(evaluation)), where=evaluation.Core_Weight.to_numpy() > 0)
    reference_metrics = portfolio_metrics(portfolios["Nested_Selected"])
    random_metrics, random_weights, random_returns = [], {}, {}
    for replication in range(protocol["Random_Replications"]):
        seed = protocol["Random_Seed"]+replication
        randomized = shuffle_overlay_blocks(multiplier, evaluation.Fold, block_length=protocol["Random_Block_Length"], seed=seed)
        weights = evaluation.Core_Weight.to_numpy()*randomized
        name = f"random_{replication:04d}"
        result = _simulate(dataset, evaluation, weights, name, protocol, "retrospective_block_randomized_overlay")
        metrics = portfolio_metrics(result)
        random_metrics.append(dict(Replication=replication, Seed=seed, **metrics,
            Average_Target_Exposure=float(weights.mean()),
            Target_Exposure_Mismatch=float(weights.mean()-evaluation.Position.mean()),
            Realized_Exposure_Mismatch=metrics["Average_Exposure"]-reference_metrics["Average_Exposure"],
            Annual_Turnover_Mismatch=metrics["Annual_Turnover"]-reference_metrics["Annual_Turnover"],
            Cost_Mismatch=metrics["Total_Cost_Dollars"]-reference_metrics["Total_Cost_Dollars"],
            Maximum_Ledger_Identity_Error=result.accounting_reconciliation["Maximum_Relative_Wealth_Identity_Error"]))
        random_weights[name] = weights
        random_returns[name] = result.daily.Net_Excess_Return.to_numpy()
        if (replication+1) % 20 == 0:
            print(f"Completed {replication+1}/{protocol['Random_Replications']} randomized overlay ledgers", flush=True)
    random_metrics = pd.DataFrame(random_metrics)
    random_summary = dict(Replications=len(random_metrics), Observed_Sharpe=reference_metrics["Sharpe_Ratio"],
        Upper_Tail_Reference_Fraction=float((1+random_metrics.Sharpe_Ratio.ge(reference_metrics["Sharpe_Ratio"]).sum())/(len(random_metrics)+1)),
        Median_Random_Sharpe=float(random_metrics.Sharpe_Ratio.median()),
        Interpretation="retrospective_reference_distribution_conditional_on_saved_overlay_blocks")
    if (any(_digest(path) != source_hashes[str(path.relative_to(project).as_posix())] for path in source_paths)
            or _notebook_code_digest(project/"sp500_market_regime_classification.ipynb") != notebook_digest):
        raise RuntimeError("Attribution inputs changed during execution")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pending-attribution-", dir=output.parent) as temporary:
        pending = Path(temporary)/output.name
        pending.mkdir()
        for name, portfolio in portfolios.items():
            save_portfolio_result(portfolio, pending/"portfolios"/name)
        for fold, portfolio in historical_ledgers.items():
            save_portfolio_result(portfolio, pending/"matching_ledgers"/f"fold_{fold}")
        for name, table in [("evaluation",evaluation), ("matching_history",data["history"]), ("folds",data["folds"]),
            ("control_weights",controls.weights), ("instrument_returns",market)]:
            table.to_parquet(pending/f"{name}.parquet", index=True)
        for name, table in [("attribution",results.attribution), ("factor_regression",results.factor_regression),
            ("paired_comparisons",results.paired_comparisons), ("episode_contributions",results.episodes),
            ("fold_contributions",results.fold_contributions), ("matching_audit",controls.matching_audit),
            ("predictive_diagnostics",predictive), ("probability_calibration",calibration),
            ("score_quantiles",score_quantiles), ("shuffled_score_null",null_draws)]:
            table.to_csv(pending/f"{name}.csv", index=False)
        random_metrics.to_parquet(pending/"random_control_distribution.parquet", index=False)
        pd.DataFrame(random_weights, index=controls.weights.index).to_parquet(pending/"random_control_weights.parquet")
        pd.DataFrame(random_returns, index=market.index).to_parquet(pending/"random_control_returns.parquet")
        registry = dict(Protocol=protocol, Dataset_ID=dataset.dataset_id, Reference_Portfolio="Nested_Selected",
            Track="historical_price_index", Cash_Convention="constant_zero", Rebalance_Policy="daily_target",
            Controls=list(controls.weights.columns), Selection_Use="retrospective_attribution_diagnostics",
            Random_Reference=random_summary, Shuffled_Score_Null=null_summary)
        (pending/"control_registry.json").write_text(json.dumps(registry, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        _plots(pending, portfolios, results, market, "Nested_Selected")
        manifest = dict(Schema_Version="1.0.0", Completed_UTC=datetime.now(timezone.utc).isoformat(),
            Source_SHA256=source_hashes, Dataset_ID=dataset.dataset_id, Notebook_Code_SHA256=notebook_digest,
            Files={str(path.relative_to(pending).as_posix()):_digest(path) for path in sorted(pending.rglob("*")) if path.is_file()})
        (pending/"completion.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        pending.rename(output)
    _update_notebook(project, output)
    return results.attribution


def _compare_attribution_table(expected, actual):
    keys = [column for column in ["Portfolio","Control","Episode_ID","Fold"] if column in expected]
    if not keys:
        raise ValueError("Attribution comparison needs portfolio or contribution identifiers")
    _compare_saved_table(expected.sort_values(keys).reset_index(drop=True),
                         actual.sort_values(keys).reset_index(drop=True))


def verify_attribution_research(project_dir, output):
    project, output = Path(project_dir).resolve(), Path(output).resolve()
    manifest = json.loads((output/"completion.json").read_text(encoding="utf-8"))
    for base, hashes in [(project,manifest["Source_SHA256"]),(output,manifest["Files"])]:
        for name, expected in hashes.items():
            path = (base/name).resolve()
            if not path.is_relative_to(base) or _digest(path) != expected:
                raise ValueError(f"Attribution checksum mismatch for {name}")
    if _notebook_code_digest(project/"sp500_market_regime_classification.ipynb") != manifest["Notebook_Code_SHA256"]:
        raise ValueError("Notebook code differs from the attribution receipt")
    registry = json.loads((output/"control_registry.json").read_text(encoding="utf-8"))
    protocol = registry["Protocol"]
    dataset = load_dataset(manifest["Dataset_ID"], root=project/"datasets")
    data = {name: pd.read_parquet(output/f"{filename}.parquet") for name, filename in [
        ("evaluation","evaluation"),("history","matching_history"),("folds","folds")]}
    controls = build_controls(protocol, data)
    pd.testing.assert_frame_equal(controls.weights, pd.read_parquet(output/"control_weights.parquet"))
    _compare_saved_table(pd.read_csv(output/"matching_audit.csv").astype({column:str for column in
        ["Matching_Decision_Time","Estimation_Start","Estimation_End","First_Evaluation_Decision"]}),
        controls.matching_audit.assign(**{column:controls.matching_audit[column].astype(str) for column in
        ["Matching_Decision_Time","Estimation_Start","Estimation_End","First_Evaluation_Decision"]}))
    portfolios = {path.name:load_portfolio(path) for path in (output/"portfolios").iterdir() if path.is_dir()}
    for path in (output/"matching_ledgers").iterdir():
        verify_portfolio_result(path)
    market = pd.read_parquet(output/"instrument_returns.parquet")
    results = attribute(portfolios, market, dict(protocol, Reference_Portfolio="Nested_Selected"))
    for name, table in [("attribution",results.attribution),("factor_regression",results.factor_regression),
        ("paired_comparisons",results.paired_comparisons),("episode_contributions",results.episodes),
        ("fold_contributions",results.fold_contributions)]:
        expected = pd.read_csv(output/f"{name}.csv", float_precision="round_trip")
        actual = table.copy()
        for column in actual.select_dtypes(include=["datetimetz"]).columns:
            actual[column] = actual[column].astype(str)
        _compare_attribution_table(expected, actual)
    weights = pd.read_parquet(output/"random_control_weights.parquet")
    returns = pd.read_parquet(output/"random_control_returns.parquet")
    metrics = pd.read_parquet(output/"random_control_distribution.parquet")
    evaluation = data["evaluation"]
    multiplier = np.divide(evaluation.Position, evaluation.Core_Weight,
                           out=np.ones(len(evaluation)), where=evaluation.Core_Weight.to_numpy()>0)
    for replication in range(protocol["Random_Replications"]):
        name = f"random_{replication:04d}"
        shuffled = shuffle_overlay_blocks(multiplier, evaluation.Fold,
            block_length=protocol["Random_Block_Length"], seed=protocol["Random_Seed"]+replication)
        np.testing.assert_allclose(weights[name], evaluation.Core_Weight.to_numpy()*shuffled, rtol=0, atol=1e-12)
    for replication in sorted({0,protocol["Random_Replications"]//2,protocol["Random_Replications"]-1}):
        name = f"random_{replication:04d}"
        replay = _simulate(dataset, evaluation, weights[name], name, protocol, "retrospective_block_randomized_overlay")
        np.testing.assert_allclose(replay.daily.Net_Excess_Return, returns[name], rtol=0, atol=1e-12)
        for field, value in portfolio_metrics(replay).items():
            if value is not None:
                np.testing.assert_allclose(value, metrics.loc[metrics.Replication.eq(replication),field].iloc[0], rtol=1e-10, atol=1e-12)
    draws, summary = shuffled_score_null(protocol)
    _compare_saved_table(pd.read_csv(output/"shuffled_score_null.csv", float_precision="round_trip"), draws)
    if not summary["Passed"]:
        raise ValueError("Shuffled-score null failed on reproduction")
    return dict(Verified=True, Portfolios=len(portfolios), Randomized_Draws=len(metrics), Randomized_Ledgers_Replayed=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--protocol", default="protocols/layer4_controls.json")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output")
    action.add_argument("--verify")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_attribution_research(args.project_dir,args.verify),indent=2))
    else:
        table = run_attribution_research(args.project_dir,args.output,args.protocol)
        print(table[["Portfolio","Sharpe_Ratio","Average_Exposure","Annual_Turnover"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""Portfolio P&L, exposure, factor, episode, and predictive-score diagnostics."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score

from .data.contracts import utc_timestamp
from .execution import PortfolioResult, portfolio_metrics, verify_portfolio_result
from .statistics import hac_long_run_covariance, hac_market_regression, hac_mean_test, paired_sharpe_bootstrap


@dataclass(frozen=True)
class AttributionResult:
    attribution: pd.DataFrame
    factor_regression: pd.DataFrame
    paired_comparisons: pd.DataFrame
    episodes: pd.DataFrame
    fold_contributions: pd.DataFrame


def load_portfolio(directory):
    directory = Path(directory)
    verify_portfolio_result(directory)
    tables = {name: pd.read_parquet(directory/f"{filename}.parquet") for name, filename in [
        ("daily", "portfolio_daily"), ("valuations", "portfolio_valuations"), ("trades", "trades"),
        ("cash_events", "cash_events"), ("targets", "targets")]}
    receipt = json.loads((directory/"accounting_reconciliation.json").read_text(encoding="utf-8"))
    return PortfolioResult(accounting_reconciliation=receipt, **tables)


def _expected_cash_interest(portfolio, credited_returns):
    daily = portfolio.daily
    expected = daily.Cash.to_numpy(float)*np.asarray(credited_returns, dtype=float)
    payments = portfolio.cash_events.loc[portfolio.cash_events.Event_Type.eq("distribution_payment")]
    starts, ends = pd.DatetimeIndex(daily.Execution_Time), pd.DatetimeIndex(daily.Return_End_Time)
    timezone = portfolio.accounting_reconciliation["Config"]["exchange_timezone"]
    for payment in payments.itertuples(index=False):
        index = starts.searchsorted(payment.Event_Time, side="left")-1
        if index < 0 or payment.Event_Time > ends[index]:
            continue
        local_start = starts[index].tz_convert(timezone).tz_localize(None)
        local_end = ends[index].tz_convert(timezone).tz_localize(None)
        local_payment = payment.Event_Time.tz_convert(timezone).tz_localize(None)
        fraction = (local_end-local_payment).total_seconds()/(local_end-local_start).total_seconds()
        expected[index] += payment.Amount*credited_returns.iloc[index]*fraction
    return expected


def _episode_contributions(daily, name, threshold):
    active = daily.Realized_Weight.gt(threshold).to_numpy()
    prior = np.r_[False, active[:-1]]
    identifiers = np.cumsum(active & ~prior)
    identifiers[~active & ~prior] = 0
    records = []
    for identifier in np.unique(identifiers):
        selected = np.flatnonzero(identifiers == identifier)
        group = daily.iloc[selected]
        records.append(dict(Portfolio=name, Episode_ID=int(identifier),
            First_Execution=group.Execution_Time.iloc[0], Accounting_End=group.Return_End_Time.iloc[-1],
            Holding_Periods=int(active[selected].sum()), Accounting_Periods=len(group),
            Log_Contribution=float(np.log1p(group.Net_Return).sum()),
            Price_PnL=float(group.Price_PnL.sum()), Distribution_PnL=float(group.Distribution_PnL.sum()),
            Cash_Interest=float(group.Cash_Interest.sum()), Trading_Cost=float(group.Transaction_Cost.sum())))
    table = pd.DataFrame(records)
    absolute_total = table.Log_Contribution.abs().sum()
    table["Absolute_Log_Contribution_Share"] = (table.Log_Contribution.abs()/absolute_total if absolute_total > 0 else np.nan)
    return table


def _factor_regression(excess, market, lag, confidence):
    if len(excess) <= max(lag, 3):
        return dict(Status="insufficient_observations", Daily_Alpha=None, Beta=None, Alpha_HAC_SE=None)
    result = hac_market_regression(excess, market, lag, confidence)
    if result["Beta"] is None:
        return result
    design = np.column_stack([np.ones(len(excess)), market])
    coefficients = np.linalg.lstsq(design, excess, rcond=None)[0]
    residuals = excess-design@coefficients
    bread = np.linalg.inv(design.T@design)
    covariance = bread@(hac_long_run_covariance(design*residuals[:, None], lag)*len(excess))@bread
    beta_error = float(np.sqrt(max(covariance[1, 1], 0)))
    critical = norm.ppf((1+confidence)/2)
    result.update(Beta_HAC_SE=beta_error, Beta_Lower=result["Beta"]-critical*beta_error,
                  Beta_Upper=result["Beta"]+critical*beta_error)
    return result


def _holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return adjusted


def attribute(portfolios, instrument_returns, control_config):
    protocol = control_config
    reference_name = protocol["Reference_Portfolio"]
    reference = portfolios[reference_name]
    required = {"Market_Return", "Market_Excess_Return", "Reference_Return", "Credited_Return", "Fold"}
    if not required.issubset(instrument_returns) or not np.isfinite(instrument_returns.to_numpy(float)).all():
        raise ValueError("Attribution requires finite common market, cash, and fold data")
    index = pd.DatetimeIndex(reference.daily.Return_End_Time)
    if index.has_duplicates or not index.equals(instrument_returns.index):
        raise ValueError("Attribution requires an identical chronological return-date mask")
    rows, factors, episodes, folds, comparisons = [], [], [], [], []
    base_config = reference.accounting_reconciliation["Config"]
    compare_fields = ["initial_capital", "one_way_cost", "dividend_payment_policy", "exchange_timezone", "terminal_liquidation"]
    for name, result in portfolios.items():
        daily, receipt = result.daily, result.accounting_reconciliation
        if (not pd.DatetimeIndex(daily.Return_End_Time).equals(index)
                or not daily.Execution_Time.equals(reference.daily.Execution_Time)):
            raise ValueError("Portfolios have different execution or return dates")
        if any(receipt["Config"][key] != base_config[key] for key in compare_fields):
            raise ValueError("Portfolio accounting assumptions differ")
        if receipt["Config"]["one_way_cost"] != protocol["One_Way_Cost"]:
            raise ValueError("Portfolio costs differ from the control protocol")
        if (receipt.get("Dataset_ID") != reference.accounting_reconciliation.get("Dataset_ID")
                or receipt["Instrument_ID"] != reference.accounting_reconciliation["Instrument_ID"]):
            raise ValueError("Portfolios use different datasets or instruments")
        if not np.array_equal(result.valuations.Price.to_numpy(), reference.valuations.Price.to_numpy()):
            raise ValueError("Portfolio valuation prices differ")
        if not np.allclose(daily.Reference_Return, instrument_returns.Reference_Return, rtol=0, atol=1e-14):
            raise ValueError("Portfolio reference cash returns differ")
        expected_interest = _expected_cash_interest(result, instrument_returns.Credited_Return)
        if not np.allclose(expected_interest, daily.Cash_Interest, rtol=1e-10, atol=1e-12):
            raise ValueError("Portfolio interest differs from the common credited cash model")
        initial = receipt["Initial_Capital"]
        bridge = initial+daily.Price_PnL.sum()+daily.Distribution_PnL.sum()+daily.Cash_Interest.sum()-daily.Transaction_Cost.sum()
        if not np.isclose(bridge, daily.End_Wealth.iloc[-1], rtol=1e-10, atol=1e-12):
            raise ValueError("Portfolio P&L components fail the wealth bridge")
        episode_table = _episode_contributions(daily, name, protocol["Episode_Exposure_Threshold"])
        episodes.append(episode_table)
        invested = episode_table.loc[episode_table.Episode_ID.gt(0)]
        down = instrument_returns.Market_Return.lt(0).to_numpy()
        denominator = instrument_returns.Market_Return.to_numpy()[down].mean() if down.any() else np.nan
        capture = float(daily.Net_Return.to_numpy()[down].mean()/denominator) if down.any() and denominator != 0 else None
        rows.append(dict(Portfolio=name, **portfolio_metrics(result), Initial_Capital=initial,
            Final_Wealth=float(daily.End_Wealth.iloc[-1]), Price_PnL=float(daily.Price_PnL.sum()),
            Distribution_PnL=float(daily.Distribution_PnL.sum()), Cash_Interest=float(daily.Cash_Interest.sum()),
            Trading_Cost=float(daily.Transaction_Cost.sum()), Pnl_Bridge_Error=float(bridge-daily.End_Wealth.iloc[-1]),
            Average_Target_Exposure=float(daily.Target_Weight.mean()), Invested_Episodes=len(invested),
            Average_Holding_Periods=float(invested.Holding_Periods.mean()) if len(invested) else None,
            Largest_Absolute_Log_Contribution_Share=float(invested.Absolute_Log_Contribution_Share.max()) if len(invested) else None,
            Downside_Capture_Ratio=capture))
        factor = _factor_regression(daily.Net_Excess_Return.to_numpy(), instrument_returns.Market_Excess_Return.to_numpy(),
                                    protocol["HAC_Lag"], protocol["Confidence_Level"])
        factor.update(Portfolio=name, Observations=len(daily), HAC_Lag=protocol["HAC_Lag"],
            Annualized_Alpha=factor["Daily_Alpha"]*252 if factor["Daily_Alpha"] is not None else None,
            Annualized_Alpha_Lower=factor.get("Alpha_Lower")*252 if factor.get("Alpha_Lower") is not None else None,
            Annualized_Alpha_Upper=factor.get("Alpha_Upper")*252 if factor.get("Alpha_Upper") is not None else None)
        factors.append(factor)
        for fold_id, positions in instrument_returns.groupby("Fold", sort=True).indices.items():
            group = daily.iloc[positions]
            folds.append(dict(Portfolio=name, Fold=int(fold_id), Periods=len(group),
                Start=group.Execution_Time.iloc[0], End=group.Return_End_Time.iloc[-1],
                Log_Contribution=float(np.log1p(group.Net_Return).sum()),
                Fold_Total_Return=float(group.End_Wealth.iloc[-1]/group.Start_Wealth.iloc[0]-1),
                Price_PnL=float(group.Price_PnL.sum()), Distribution_PnL=float(group.Distribution_PnL.sum()),
                Cash_Interest=float(group.Cash_Interest.sum()), Trading_Cost=float(group.Transaction_Cost.sum())))
    for number, name in enumerate(protocol["Primary_Comparisons"]):
        result = portfolios[name]
        difference = reference.daily.Net_Excess_Return.to_numpy()-result.daily.Net_Excess_Return.to_numpy()
        mean_test = hac_mean_test(difference, protocol["HAC_Lag"], protocol["Confidence_Level"])
        bootstrap = dict(Status="undefined_control_sharpe", Delta_Sharpe_Lower=None, Delta_Sharpe_Upper=None)
        if result.daily.Net_Excess_Return.std(ddof=1) > 1e-15:
            bootstrap = paired_sharpe_bootstrap(reference.daily.Net_Excess_Return, result.daily.Net_Excess_Return,
                repetitions=protocol["Bootstrap_Replications"], block_length=protocol["Bootstrap_Block_Length"],
                seed=protocol["Random_Seed"]+10000+number, confidence_level=protocol["Confidence_Level"])
        candidate_metrics, control_metrics = portfolio_metrics(reference), portfolio_metrics(result)
        comparisons.append(dict(Portfolio=reference_name, Control=name,
            Delta_Total_Return=candidate_metrics["Net_Total_Return"]-control_metrics["Net_Total_Return"],
            Delta_Sharpe=(candidate_metrics["Sharpe_Ratio"]-control_metrics["Sharpe_Ratio"]
                          if control_metrics["Sharpe_Ratio"] is not None else None),
            Mean_Difference=mean_test["Mean_Daily_Difference"], HAC_SE=mean_test["HAC_SE"],
            HAC_t=mean_test["HAC_t"], Mean_Lower=mean_test["Lower"], Mean_Upper=mean_test["Upper"],
            One_Sided_HAC_P=float(norm.sf(mean_test["HAC_t"])) if mean_test["HAC_t"] is not None else None,
            Bootstrap_Status=bootstrap["Status"], Delta_Sharpe_Lower=bootstrap["Delta_Sharpe_Lower"],
            Delta_Sharpe_Upper=bootstrap["Delta_Sharpe_Upper"],
            Interval_Scope="conditional_pointwise_percentile", Mean_Test_Status=mean_test["Status"]))
    comparisons = pd.DataFrame(comparisons)
    valid = comparisons.One_Sided_HAC_P.notna()
    comparisons["Holm_Adjusted_HAC_P"] = np.nan
    adjusted = _holm(comparisons.One_Sided_HAC_P.fillna(1))
    comparisons.loc[valid, "Holm_Adjusted_HAC_P"] = adjusted[valid.to_numpy()]
    return AttributionResult(pd.DataFrame(rows), pd.DataFrame(factors), comparisons,
                              pd.concat(episodes, ignore_index=True), pd.DataFrame(folds))


def predictive_diagnostics(predictions, protocol, *, evaluation_end_time):
    """Descriptive diagnostics for mature forward labels on held-out score rows."""
    required = {"Recipe_ID", "Decision_Time", "Label_End_Time", "Actual_Event", "Risk_Score",
                "Predicted_Probability", "Forward_Return"}
    if not required.issubset(predictions):
        raise ValueError("Prediction diagnostics need score, probability, and label-time fields")
    end = utc_timestamp(evaluation_end_time)
    predictions = predictions.loc[predictions.Label_End_Time.notna() & predictions.Label_End_Time.le(end)]
    rows, calibration, quantiles = [], [], []
    for recipe, group in predictions.groupby("Recipe_ID", sort=True):
        valid = group.Actual_Event.notna() & group.Forward_Return.notna() & group.Risk_Score.notna()
        group = group.loc[valid].copy()
        if group.empty:
            continue
        if not group.Actual_Event.isin([0, 1]).all() or not group.Decision_Time.lt(group.Label_End_Time).all():
            raise ValueError("Invalid prediction labels or chronology")
        actual, score = group.Actual_Event.to_numpy(int), group.Risk_Score.to_numpy(float)
        if not np.isfinite(group[["Risk_Score", "Forward_Return"]].to_numpy(float)).all():
            raise ValueError("Nonfinite score diagnostics")
        probability_mask = group.Predicted_Probability.notna()
        probability = group.loc[probability_mask, "Predicted_Probability"].to_numpy(float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("Probabilities must lie between zero and one")
        row = dict(Recipe_ID=recipe, Rows=len(group), Event_Rate=float(actual.mean()),
            ROC_AUC=float(roc_auc_score(actual, score)) if len(np.unique(actual)) == 2 else None,
            PR_AUC=float(average_precision_score(actual, score)) if actual.sum() else None,
            Rank_IC_Negative_Forward_Return=float(spearmanr(score, -group.Forward_Return).statistic)
                if np.std(score)>0 and group.Forward_Return.std()>0 else None,
            Probability_Rows=len(probability), Brier=None, Precision=None, Recall=None,
            Interpretation="descriptive_scores_with_overlapping_forward_horizons")
        if len(probability):
            labels = group.loc[probability_mask, "Actual_Event"].to_numpy(int)
            row.update(Brier=float(brier_score_loss(labels, probability)),
                Precision=float(precision_score(labels, probability >= protocol["Probability_Cutoff"], zero_division=0)),
                Recall=float(recall_score(labels, probability >= protocol["Probability_Cutoff"], zero_division=0)))
            bins = np.minimum((probability*protocol["Calibration_Bins"]).astype(int), protocol["Calibration_Bins"]-1)
            for bin_id in np.unique(bins):
                selected = bins == bin_id
                calibration.append(dict(Recipe_ID=recipe, Bin=int(bin_id), Rows=int(selected.sum()),
                    Mean_Predicted_Probability=float(probability[selected].mean()), Event_Rate=float(labels[selected].mean())))
        ranks = group.Risk_Score.rank(method="average")
        groups = pd.qcut(ranks, q=protocol["Score_Quantiles"], labels=False, duplicates="drop")
        for bin_id, selected in group.groupby(groups, observed=True):
            quantiles.append(dict(Recipe_ID=recipe, Score_Quantile=int(bin_id)+1, Rows=len(selected),
                Mean_Risk_Score=float(selected.Risk_Score.mean()), Event_Rate=float(selected.Actual_Event.mean()),
                Mean_Forward_Return=float(selected.Forward_Return.mean())))
        rows.append(row)
    return (pd.DataFrame(rows, columns=["Recipe_ID", "Rows", "Event_Rate", "ROC_AUC", "PR_AUC",
                "Rank_IC_Negative_Forward_Return", "Probability_Rows", "Brier", "Precision", "Recall", "Interpretation"]),
            pd.DataFrame(calibration, columns=["Recipe_ID", "Bin", "Rows", "Mean_Predicted_Probability", "Event_Rate"]),
            pd.DataFrame(quantiles, columns=["Recipe_ID", "Score_Quantile", "Rows", "Mean_Risk_Score", "Event_Rate", "Mean_Forward_Return"]))

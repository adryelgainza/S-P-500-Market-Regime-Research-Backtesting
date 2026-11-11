"""Past-only matching and retrospective block controls for allocation attribution."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class ControlTargets:
    weights: pd.DataFrame
    matching_audit: pd.DataFrame


def build_controls(protocol, prior_data):
    """Build weights from market rows, fold metadata, and mature prior returns.

    prior_data contains evaluation, history, and folds frames. History has one
    candidate ledger per outer fold, with return-end and decision timestamps.
    All evaluation feature rows represent data available at their own decisions.
    """
    evaluation = prior_data["evaluation"].copy()
    history, folds = prior_data["history"], prior_data["folds"]
    required = {"Decision_Time", "Execution_Time", "Return_End_Time", "Fold", "SMA_Signal",
                "Volatility", "Core_Weight"}
    if not required.issubset(evaluation) or evaluation.Execution_Time.duplicated().any():
        raise ValueError("Control evaluation rows need unique times and the required causal features")
    if not (evaluation.Decision_Time.lt(evaluation.Execution_Time)
            & evaluation.Execution_Time.lt(evaluation.Return_End_Time)).all():
        raise ValueError("Invalid control decision and return chronology")
    if not np.isfinite(evaluation[["SMA_Signal", "Volatility", "Core_Weight"]].to_numpy(float)).all():
        raise ValueError("Control features must be finite")
    if (evaluation.Volatility <= 0).any() or not evaluation.Core_Weight.between(0, 1).all():
        raise ValueError("Invalid core weights or volatility estimates")
    weights = pd.DataFrame(index=pd.DatetimeIndex(evaluation.Execution_Time, name="Execution_Time"))
    weights["Cash"] = 0.0
    weights["Buy_And_Hold"] = 1.0
    weights["Past_Exposure_Match"] = np.nan
    weights["Past_Volatility_Match"] = np.nan
    scaling = np.minimum(1, protocol["Volatility_Target"]/evaluation.Volatility.to_numpy())
    step = protocol["Volatility_Rebalance_Step"]
    weights["Volatility_Only"] = np.clip(np.round(scaling/step)*step, 0, 1)
    weights["SMA_Only"] = evaluation.SMA_Signal.to_numpy(float)
    weights["Core_Without_ML"] = evaluation.Core_Weight.to_numpy(float)
    audit = []
    if set(evaluation.Fold) != set(folds.Fold) or folds.Fold.duplicated().any():
        raise ValueError("Control folds must cover the evaluation exactly")
    for fold in folds.itertuples(index=False):
        start = fold.Decision_Time-pd.DateOffset(years=protocol["Matching_Window_Years"])
        eligible = history.loc[history.Fold.eq(fold.Fold) & history.Decision_Time.ge(start)
            & history.Return_End_Time.lt(fold.Decision_Time)].copy()
        if len(eligible) < protocol["Minimum_Matching_Rows"]:
            raise ValueError(f"Insufficient mature prior observations for fold {fold.Fold}")
        if not eligible.Candidate_ID.eq(fold.Selected_Candidate).all():
            raise ValueError("Matching history must belong to the fold's selected candidate")
        columns = ["Target_Weight", "Net_Excess_Return", "Market_Excess_Return"]
        if not np.isfinite(eligible[columns].to_numpy(float)).all() or not eligible.Target_Weight.between(0, 1).all():
            raise ValueError("Invalid historical matching weights or returns")
        market_volatility = float(eligible.Market_Excess_Return.std(ddof=1))
        if market_volatility <= 1e-15:
            raise ValueError("Historical market volatility must be measurable")
        exposure = float(eligible.Target_Weight.mean())
        volatility_match = float(np.clip(eligible.Net_Excess_Return.std(ddof=1)/market_volatility, 0, 1))
        mask = evaluation.Fold.eq(fold.Fold).to_numpy()
        first_decision = evaluation.loc[mask, "Decision_Time"].min()
        if fold.Decision_Time > first_decision:
            raise ValueError("A matching decision occurs after evaluation has begun")
        weights.loc[mask, "Past_Exposure_Match"] = exposure
        weights.loc[mask, "Past_Volatility_Match"] = volatility_match
        audit.append(dict(Fold=fold.Fold, Selected_Candidate=fold.Selected_Candidate,
            Matching_Decision_Time=fold.Decision_Time, Estimation_Start=eligible.Decision_Time.min(),
            Estimation_End=eligible.Return_End_Time.max(), Observations=len(eligible),
            Fixed_Exposure_Weight=exposure, Fixed_Volatility_Weight=volatility_match,
            First_Evaluation_Decision=first_decision))
    if not np.isfinite(weights.to_numpy()).all() or not ((weights >= 0) & (weights <= 1)).all().all():
        raise ValueError("Controls have incomplete coverage or invalid weights")
    return ControlTargets(weights, pd.DataFrame(audit))


def shuffle_overlay_blocks(multiplier, folds, *, block_length, seed):
    """Permute complete contiguous blocks within folds, preserving all values."""
    values = np.asarray(multiplier, dtype=float)
    labels = np.asarray(folds)
    if (values.ndim != 1 or len(values) != len(labels) or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any() or not isinstance(block_length, int) or block_length < 1):
        raise ValueError("Block randomization needs bounded multipliers, folds, and a positive block length")
    generator = np.random.default_rng(seed)
    randomized = values.copy()
    for fold in pd.unique(labels):
        indices = np.flatnonzero(labels == fold)
        if len(indices) == 0 or not np.all(np.diff(indices) == 1):
            raise ValueError("Randomization folds must be contiguous and present")
        blocks = [values[indices[start:start+block_length]] for start in range(0, len(indices), block_length)]
        randomized[indices] = np.concatenate([blocks[index] for index in generator.permutation(len(blocks))])
    return randomized


def shuffled_score_null(protocol):
    """Independent returns and persistent scores with an exact zero-cost ledger identity."""
    settings = protocol["Shuffled_Score_Null"]
    records = []
    for replication in range(settings["Replications"]):
        generator = np.random.default_rng(settings["Seed"]+replication)
        market = generator.normal(0, settings["Market_Daily_Volatility"], settings["Observations"])
        innovations = generator.normal(size=settings["Observations"])
        scores = np.zeros(len(market))
        for index in range(1, len(scores)):
            scores[index] = settings["Score_AR1_Phi"]*scores[index-1]+innovations[index]
        percentiles = pd.Series(scores).rolling(settings["Score_Lookback"],
            min_periods=settings["Score_Minimum"]).rank(pct=True).fillna(.5)
        weights = percentiles.lt(settings["Risk_Off_Percentile"]).to_numpy(float)
        randomized = shuffle_overlay_blocks(weights, np.zeros(len(weights)),
            block_length=settings["Block_Length"], seed=settings["Seed"]+100000+replication)
        # With zero costs and cash interest, the daily-target ledger earns w*r.
        advantage = float(np.mean((randomized-1)*market))
        records.append(dict(Replication=replication, Seed=settings["Seed"]+replication,
                            Mean_Daily_Timing_Advantage=advantage))
    draws = pd.DataFrame(records)
    mean = float(draws.Mean_Daily_Timing_Advantage.mean())
    error = float(draws.Mean_Daily_Timing_Advantage.std(ddof=1)/np.sqrt(len(draws)))
    statistic = mean/error if error > 0 else None
    critical = float(norm.ppf(1-settings["Family_Alpha"]/2))
    summary = dict(Replications=len(draws), Expected_Mean=0.0, Mean_Daily_Timing_Advantage=mean,
        Monte_Carlo_SE=error, Z=statistic, Critical_Absolute_Z=critical,
        Passed=statistic is not None and abs(statistic) <= critical)
    return draws, summary

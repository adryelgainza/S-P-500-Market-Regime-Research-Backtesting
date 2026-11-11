"""Auditable inference for aligned net and excess portfolio return streams."""

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


@dataclass(frozen=True)
class InferenceConfig:
    registered_trials: int
    seed: int = 20261109
    annualization: int = 252
    primary_block_length: int = 20
    sensitivity_block_lengths: tuple = (5, 63)
    primary_bootstrap_reps: int = 10000
    sensitivity_bootstrap_reps: int = 2000
    max_t_reps: int = 2000
    hac_lags: tuple = (5, 20, 63)
    primary_hac_lag: int = 20
    confidence_level: float = .95
    minimum_observations: int = 100
    minimum_acceptable_excess_return: float = 0.0
    selection_scope: str = "conditional_saved_selector_stream"

    def __post_init__(self):
        for name in ["registered_trials", "annualization", "primary_block_length", "primary_bootstrap_reps",
                     "sensitivity_bootstrap_reps", "max_t_reps", "minimum_observations"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not .5 < self.confidence_level < 1 or self.minimum_observations < 4:
            raise ValueError("Invalid inference confidence or minimum sample size")
        if self.primary_hac_lag not in self.hac_lags or any(not isinstance(lag, int) or lag < 0 for lag in self.hac_lags):
            raise ValueError("HAC lags must include the primary nonnegative lag")
        if any(not isinstance(length, int) or length < 1 for length in self.sensitivity_block_lengths):
            raise ValueError("Bootstrap blocks must have positive integer lengths")
        if not isinstance(self.seed, int) or self.seed < 0 or not np.isfinite(self.minimum_acceptable_excess_return):
            raise ValueError("Invalid seed or minimum acceptable return")
        if self.selection_scope not in {"fixed_candidate", "conditional_saved_selector_stream"}:
            raise ValueError("Choose fixed-candidate or conditional saved-stream inference")


@dataclass(frozen=True)
class InferenceResult:
    inference: dict
    dsr_inputs: pd.DataFrame
    trial_dependence: pd.DataFrame
    trial_diagnostics: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    hac_summary: pd.DataFrame
    max_t: pd.DataFrame


def _finite_array(values, minimum=2):
    result = np.asarray(values, dtype=float)
    if result.ndim not in {1, 2} or len(result) < minimum or not np.isfinite(result).all():
        raise ValueError("Returns must be finite with enough aligned observations")
    return result


def annualized_sharpe(values, annualization=252):
    values = _finite_array(values)
    scale = np.std(values, ddof=1, axis=0)
    return np.divide(np.mean(values, axis=0)*np.sqrt(annualization), scale,
                     out=np.full(np.shape(scale), np.nan), where=scale > 1e-15)


def initial_capital_drawdown(equity):
    equity = pd.Series(equity, copy=False)
    return equity/equity.cummax().clip(lower=1.0)-1.0


def deflated_sharpe_from_moments(daily_sharpe, observations, skewness, nonexcess_kurtosis,
                                trial_sharpe_variance, independent_trials):
    """Bailey and Lopez de Prado (2014), equation 2, in daily Sharpe units."""
    row = dict(Daily_Sharpe=float(daily_sharpe), Observations=int(observations), Skewness=float(skewness),
               Nonexcess_Kurtosis=float(nonexcess_kurtosis), Trial_Sharpe_Variance=trial_sharpe_variance,
               Independent_Trials=independent_trials, Expected_Max_Daily_Sharpe=None,
               Sharpe_Standard_Error=None, DSR=None, Status="ok")
    numeric = [daily_sharpe, skewness, nonexcess_kurtosis, independent_trials]
    if (not np.isfinite(numeric).all() or observations < 4 or independent_trials < 1
            or int(independent_trials) != independent_trials or nonexcess_kurtosis < 0):
        row["Status"] = "invalid_moments_or_sample"
        return row
    if independent_trials == 1:
        threshold = 0.0
    elif trial_sharpe_variance is None or not np.isfinite(trial_sharpe_variance) or trial_sharpe_variance < 0:
        row["Status"] = "unavailable_trial_dispersion"
        return row
    else:
        n = independent_trials
        expected_normal_maximum = ((1-np.euler_gamma)*norm.ppf(1-1/n)
                                   + np.euler_gamma*norm.ppf(1-1/(n*math.e)))
        threshold = float(np.sqrt(trial_sharpe_variance)*expected_normal_maximum)
    variance_term = 1-skewness*daily_sharpe+(nonexcess_kurtosis-1)*daily_sharpe**2/4
    row["Expected_Max_Daily_Sharpe"] = threshold
    if not np.isfinite(variance_term) or variance_term <= 0:
        row["Status"] = "nonpositive_sharpe_variance"
        return row
    standard_error = float(np.sqrt(variance_term/(observations-1)))
    row.update(Sharpe_Standard_Error=standard_error,
               DSR=float(norm.cdf((daily_sharpe-threshold)/standard_error)))
    return row


def trial_family_diagnostics(trial_returns, registered_trials):
    if not isinstance(trial_returns, pd.DataFrame) or trial_returns.columns.duplicated().any():
        raise ValueError("Trial returns need uniquely named registered columns")
    if registered_trials < len(trial_returns.columns):
        raise ValueError("Registered trial count is smaller than the supplied family")
    usable, diagnostics, fingerprints = [], [], {}
    for name, series in trial_returns.items():
        values = series.to_numpy(float)
        status = "ok"
        if not np.isfinite(values).all():
            status = "missing_or_nonfinite_returns"
        elif len(values) < 4:
            status = "insufficient_observations"
        elif np.std(values, ddof=1) <= 1e-15:
            status = "zero_return_variance"
        daily_sharpe = float(np.mean(values)/np.std(values, ddof=1)) if status == "ok" else None
        duplicate_of = None
        if status == "ok":
            fingerprint = values.astype("<f8").tobytes()
            duplicate_of = fingerprints.get(fingerprint)
            fingerprints.setdefault(fingerprint, str(name))
            usable.append(name)
        diagnostics.append(dict(Candidate_ID=name, Status=status, Daily_Sharpe=daily_sharpe,
                                Duplicate_Of=duplicate_of, Observations=len(values)))
    table = pd.DataFrame(diagnostics, columns=["Candidate_ID", "Status", "Daily_Sharpe", "Duplicate_Of", "Observations"])
    sharpes = table.loc[table.Status.eq("ok"), "Daily_Sharpe"].to_numpy(float)
    variance = float(np.var(sharpes, ddof=1)) if len(sharpes) >= 2 else None
    dependence = []
    average_correlation, implied = None, None
    if len(usable) == 1:
        implied = 1.0
        matrix = np.ones((1, 1))
    elif len(usable) > 1:
        matrix = np.corrcoef(trial_returns[usable].to_numpy(float), rowvar=False)
        average_correlation = float((matrix.sum()-len(usable))/(len(usable)*(len(usable)-1)))
        if average_correlation >= -1e-12:
            rho = float(np.clip(average_correlation, 0, 1))
            implied = rho+(1-rho)*len(usable)
    else:
        matrix = np.empty((0, 0))
    for index, name in enumerate(usable):
        dependence.append(dict(Candidate_ID=name,
            Mean_Correlation_With_Other_Trials=float((matrix[index].sum()-1)/(len(usable)-1)) if len(usable)>1 else None))
    if implied is not None and abs(implied-round(implied)) < 1e-10:
        implied = float(round(implied))
    summary = dict(Registered_Trials=registered_trials, Supplied_Trials=len(table), Usable_Trials=len(usable),
        Distinct_Usable_Return_Streams=len(fingerprints), Unavailable_Registered_Trials=registered_trials-len(usable),
        Trial_Sharpe_Variance=variance, Dispersion_Estimator="sample_variance_ddof_1_of_daily_trial_Sharpes",
        Mean_Offdiagonal_Correlation=average_correlation, Implied_Independent_Trials=implied,
        Rounded_Independent_Trials=math.ceil(implied) if implied is not None else None,
        Dependence_Status="ok" if implied is not None else "unsupported_negative_or_unavailable_correlation")
    return summary, table, pd.DataFrame(dependence, columns=["Candidate_ID", "Mean_Correlation_With_Other_Trials"])


def deflated_sharpe(returns, trial_returns, registered_trials):
    family, diagnostics, dependence = trial_family_diagnostics(trial_returns, registered_trials)
    values = np.asarray(returns, dtype=float)
    valid = values.ndim == 1 and len(values) >= 4 and np.isfinite(values).all() and np.std(values, ddof=1) > 1e-15
    rows = []
    for scenario, count in [("raw_registered_count", registered_trials),
                             ("average_correlation_sensitivity", family["Rounded_Independent_Trials"])]:
        if not valid or count is None:
            rows.append(dict(Scenario=scenario, Status="undefined_selected_series" if not valid else family["Dependence_Status"], DSR=None))
            continue
        row = deflated_sharpe_from_moments(np.mean(values)/np.std(values, ddof=1), len(values),
            skew(values, bias=False), kurtosis(values, fisher=False, bias=False),
            family["Trial_Sharpe_Variance"], count)
        rows.append(dict(Scenario=scenario, **row))
    return pd.DataFrame(rows), family, diagnostics, dependence


def hac_long_run_covariance(values, maximum_lag=20):
    matrix = _finite_array(values)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    n = len(matrix)
    if not isinstance(maximum_lag, int) or not 0 <= maximum_lag < n:
        raise ValueError("HAC lag must be a nonnegative integer smaller than sample size")
    centered = matrix-matrix.mean(axis=0)
    covariance = centered.T@centered/n
    for lag in range(1, maximum_lag+1):
        lagged = centered[lag:].T@centered[:-lag]/n
        covariance += (1-lag/(maximum_lag+1))*(lagged+lagged.T)
    return covariance


def hac_mean_test(values, maximum_lag=20, confidence_level=.95):
    values = _finite_array(values)
    if values.ndim != 1:
        raise ValueError("Mean inference requires one return-difference series")
    variance = float(hac_long_run_covariance(values, maximum_lag)[0, 0])
    mean = float(np.mean(values))
    if variance <= 1e-30:
        return dict(Status="zero_hac_variance", Mean_Daily_Difference=mean, HAC_SE=None, HAC_t=None, Lower=None, Upper=None)
    error = np.sqrt(variance/len(values))
    critical = norm.ppf((1+confidence_level)/2)
    return dict(Status="ok", Mean_Daily_Difference=mean, HAC_SE=float(error), HAC_t=float(mean/error),
                Lower=float(mean-critical*error), Upper=float(mean+critical*error))


def hac_market_regression(portfolio_excess, market_excess, maximum_lag=20, confidence_level=.95):
    y, market = _finite_array(portfolio_excess, 4), _finite_array(market_excess, 4)
    if y.shape != market.shape or y.ndim != 1:
        raise ValueError("Regression returns must be paired one-dimensional series")
    design = np.column_stack([np.ones(len(y)), market])
    if np.linalg.matrix_rank(design) < 2:
        return dict(Status="rank_deficient_market", Daily_Alpha=None, Beta=None, Alpha_HAC_SE=None)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y-design@coefficients
    scores = design*residual[:, None]
    bread = np.linalg.inv(design.T@design)
    covariance = bread@(hac_long_run_covariance(scores, maximum_lag)*len(y))@bread
    variance = float(covariance[0, 0])
    if variance <= 1e-30:
        return dict(Status="zero_alpha_variance", Daily_Alpha=float(coefficients[0]), Beta=float(coefficients[1]), Alpha_HAC_SE=None)
    error = float(np.sqrt(variance))
    critical = norm.ppf((1+confidence_level)/2)
    return dict(Status="ok", Daily_Alpha=float(coefficients[0]), Beta=float(coefficients[1]), Alpha_HAC_SE=error,
                Alpha_HAC_t=float(coefficients[0]/error), Alpha_Lower=float(coefficients[0]-critical*error),
                Alpha_Upper=float(coefficients[0]+critical*error), Residual_Daily_Volatility=float(np.std(residual, ddof=2)))


def paired_sharpe_bootstrap(candidate, benchmark, *, repetitions=10000, block_length=20,
                            seed=20261109, confidence_level=.95, annualization=252):
    candidate, benchmark = _finite_array(candidate), _finite_array(benchmark)
    if candidate.ndim != 1 or candidate.shape != benchmark.shape:
        raise ValueError("Paired bootstrap requires equal one-dimensional return series")
    n = len(candidate)
    if not isinstance(block_length, int) or not 1 <= block_length <= n or repetitions < 1:
        raise ValueError("Invalid bootstrap block or replication count")
    generator = np.random.default_rng(seed)
    candidate_samples, differences = [], []
    for start in range(0, repetitions, 128):
        count = min(128, repetitions-start)
        starts = generator.integers(0, n, size=(count, math.ceil(n/block_length)))
        indices = ((starts[:, :, None]+np.arange(block_length)) % n).reshape(count, -1)[:, :n]
        portfolio_samples, baseline_samples = candidate[indices], benchmark[indices]
        portfolio_scale = portfolio_samples.std(axis=1, ddof=1)
        benchmark_scale = baseline_samples.std(axis=1, ddof=1)
        portfolio_sharpe = np.divide(portfolio_samples.mean(axis=1)*np.sqrt(annualization), portfolio_scale,
                                     out=np.full(count, np.nan), where=portfolio_scale > 1e-15)
        benchmark_sharpe = np.divide(baseline_samples.mean(axis=1)*np.sqrt(annualization), benchmark_scale,
                                     out=np.full(count, np.nan), where=benchmark_scale > 1e-15)
        candidate_samples.extend(portfolio_sharpe)
        differences.extend(portfolio_sharpe-benchmark_sharpe)
    candidate_samples, differences = np.asarray(candidate_samples), np.asarray(differences)
    valid = np.isfinite(candidate_samples) & np.isfinite(differences)
    row = dict(Block_Length=block_length, Replications=repetitions, Seed=seed, Valid_Replications=int(valid.sum()),
        Status="ok" if valid.sum() >= .95*repetitions else "insufficient_valid_bootstrap_samples",
        Candidate_Sharpe_Lower=None, Candidate_Sharpe_Upper=None, Delta_Sharpe_Lower=None,
        Delta_Sharpe_Upper=None, Bootstrap_Fraction_Delta_Positive=None, Fraction_MC_SE=None)
    if row["Status"] == "ok":
        tail = (1-confidence_level)/2
        candidate_bounds = np.quantile(candidate_samples[valid], [tail, 1-tail])
        difference_bounds = np.quantile(differences[valid], [tail, 1-tail])
        fraction = float(np.mean(differences[valid] > 0))
        row.update(Candidate_Sharpe_Lower=float(candidate_bounds[0]), Candidate_Sharpe_Upper=float(candidate_bounds[1]),
            Delta_Sharpe_Lower=float(difference_bounds[0]), Delta_Sharpe_Upper=float(difference_bounds[1]),
            Bootstrap_Fraction_Delta_Positive=fraction, Fraction_MC_SE=float(np.sqrt(fraction*(1-fraction)/valid.sum())))
    return row


def _hac_mean_scales(matrix, maximum_lag):
    centered = matrix-matrix.mean(axis=0)
    variance = np.mean(centered**2, axis=0)
    for lag in range(1, maximum_lag+1):
        variance += 2*(1-lag/(maximum_lag+1))*np.sum(centered[lag:]*centered[:-lag], axis=0)/len(matrix)
    return np.sqrt(np.maximum(variance, 0)/len(matrix))


def max_t_adjustment(candidate_returns, benchmark, *, repetitions=2000, block_length=20,
                     maximum_lag=20, seed=20261109):
    """Single-step, one-sided circular-block max-T with fixed HAC studentization."""
    if not isinstance(candidate_returns, pd.DataFrame) or candidate_returns.columns.duplicated().any():
        raise ValueError("max-T requires uniquely named candidate columns")
    baseline = _finite_array(benchmark)
    if baseline.ndim != 1 or len(candidate_returns) != len(baseline):
        raise ValueError("Candidate and benchmark dates must be aligned")
    n = len(baseline)
    if (not isinstance(block_length, int) or not 1 <= block_length <= n
            or not isinstance(maximum_lag, int) or not 0 <= maximum_lag < n or repetitions < 1):
        raise ValueError("Invalid max-T resampling settings")
    supplied = candidate_returns.to_numpy(float)
    finite = np.isfinite(supplied).all(axis=0)
    differences = supplied[:, finite]-baseline[:, None]
    scales = _hac_mean_scales(differences, maximum_lag)
    valid_scale = scales > 1e-15
    valid_positions = np.flatnonzero(finite)[valid_scale]
    differences, scales = differences[:, valid_scale], scales[valid_scale]
    output = pd.DataFrame(dict(Candidate_ID=candidate_returns.columns,
        Status=np.where(finite, "zero_hac_variance", "missing_or_nonfinite_returns"),
        Mean_Daily_Difference=np.nan, HAC_SE=np.nan, Observed_t=np.nan, MaxT_Adjusted_P=np.nan,
        Unadjusted_Bootstrap_P=np.nan, P_MC_SE=np.nan))
    if not len(valid_positions):
        return output
    means = differences.mean(axis=0)
    observed = means/scales
    centered = differences-means
    extended = np.vstack([centered, centered[:block_length-1]])
    cumulative = np.vstack([np.zeros((1, centered.shape[1])), np.cumsum(extended, axis=0)])
    starts = np.arange(n)
    block_sums = cumulative[starts+block_length]-cumulative[starts]
    blocks_needed = math.ceil(n/block_length)
    remainder = n-(blocks_needed-1)*block_length
    last_sums = cumulative[starts+remainder]-cumulative[starts]
    generator = np.random.default_rng(seed)
    maxima, point_counts = [], np.zeros(len(observed), dtype=int)
    for start in range(0, repetitions, 64):
        count = min(64, repetitions-start)
        sampled_starts = generator.integers(0, n, size=(count, blocks_needed))
        sums = block_sums[sampled_starts[:, :-1]].sum(axis=1)+last_sums[sampled_starts[:, -1]]
        statistics = sums/n/scales
        maxima.extend(np.max(statistics, axis=1))
        point_counts += np.sum(statistics >= observed, axis=0)
    maxima = np.asarray(maxima)
    adjusted = (1+np.sum(maxima[:, None] >= observed, axis=0))/(repetitions+1)
    output.loc[valid_positions, "Status"] = "ok"
    output.loc[valid_positions, ["Mean_Daily_Difference", "HAC_SE", "Observed_t",
        "MaxT_Adjusted_P", "Unadjusted_Bootstrap_P", "P_MC_SE"]] = np.column_stack([
            means, scales, observed, adjusted,
            (point_counts+1)/(repetitions+1), np.sqrt(adjusted*(1-adjusted)/(repetitions+1))])
    for column in output.columns.difference(["Candidate_ID", "Status"]):
        output[column] = pd.to_numeric(output[column], errors="raise")
    return output


def infer_performance(portfolio_returns, benchmark_returns, trial_returns, inference_config):
    config = inference_config
    if not isinstance(portfolio_returns, pd.DataFrame) or not {"Net_Return", "Net_Excess_Return"}.issubset(portfolio_returns):
        raise ValueError("Portfolio returns require net and excess return columns")
    if not isinstance(benchmark_returns, pd.DataFrame) or benchmark_returns.empty:
        raise ValueError("At least one named benchmark excess-return series is required")
    index = portfolio_returns.index
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Inference dates must be unique and chronological")
    if not index.equals(benchmark_returns.index) or not index.equals(trial_returns.index):
        raise ValueError("Inference uses an explicit identical date mask for every series")
    if benchmark_returns.columns.duplicated().any() or "Reported_Portfolio" in trial_returns:
        raise ValueError("Inference return columns must have distinct identifiers")
    net = _finite_array(portfolio_returns.Net_Return, config.minimum_observations)
    excess = _finite_array(portfolio_returns.Net_Excess_Return, config.minimum_observations)
    if (net <= -1).any():
        raise ValueError("Net returns must preserve positive portfolio wealth")
    if max((*config.hac_lags, config.primary_hac_lag)) >= len(net) or max((config.primary_block_length, *config.sensitivity_block_lengths)) > len(net):
        raise ValueError("Inference sample is too short for the registered lags and blocks")
    _finite_array(benchmark_returns, config.minimum_observations)
    dsr, family, diagnostics, dependence = deflated_sharpe(excess, trial_returns, config.registered_trials)
    bootstrap, hac = [], []
    blocks = [config.primary_block_length, *config.sensitivity_block_lengths]
    for number, (name, benchmark) in enumerate(benchmark_returns.items()):
        for length in dict.fromkeys(blocks):
            repetitions = config.primary_bootstrap_reps if length == config.primary_block_length else config.sensitivity_bootstrap_reps
            bootstrap.append(dict(Benchmark=name, Is_Primary_Block=length == config.primary_block_length,
                **paired_sharpe_bootstrap(excess, benchmark, repetitions=repetitions, block_length=length,
                    seed=config.seed+100*number+length, confidence_level=config.confidence_level,
                    annualization=config.annualization)))
        for lag in config.hac_lags:
            hac.append(dict(Benchmark=name, HAC_Lag=lag, Is_Primary_Lag=lag == config.primary_hac_lag,
                            **hac_mean_test(excess-benchmark.to_numpy(), lag, config.confidence_level)))
    tested_family = trial_returns.copy()
    tested_family["Reported_Portfolio"] = excess
    primary_name = benchmark_returns.columns[0]
    multiplicity = max_t_adjustment(tested_family, benchmark_returns.iloc[:, 0],
        repetitions=config.max_t_reps, block_length=config.primary_block_length,
        maximum_lag=config.primary_hac_lag, seed=config.seed+900)
    equity = np.cumprod(1+net)
    shortfall = np.minimum(excess-config.minimum_acceptable_excess_return, 0)
    downside = float(np.sqrt(np.mean(shortfall**2)))
    sharpe_value = float(annualized_sharpe(excess, config.annualization))
    summary = dict(Config=asdict(config), Observations=len(net), Sample_Start=str(index[0]), Sample_End=str(index[-1]),
        Net_Total_Return=float(equity[-1]-1), Sharpe=sharpe_value if np.isfinite(sharpe_value) else None,
        Maximum_Drawdown=float(initial_capital_drawdown(equity).min()),
        Sortino=float(np.mean(excess-config.minimum_acceptable_excess_return)/downside*np.sqrt(config.annualization)) if downside>1e-15 else None,
        Trial_Family=family, Multiplicity_Benchmark=primary_name, Multiplicity_Hypotheses=len(tested_family.columns),
        MaxT_Method="single_step_one_sided_circular_blocks_with_fixed_Bartlett_HAC_studentization",
        Selector_Inference_Scope=config.selection_scope,
        DSR_Assumptions="Analytical moment approximation with a normal cross-trial Sharpe model and IID selected-return sampling formula.",
        Conditional_Claim="Resampling applies to the supplied realized return streams and holds model fitting and selection decisions fixed.",
        Dispersion_Zero=family["Trial_Sharpe_Variance"] == 0)
    if "Market" in benchmark_returns:
        summary["Descriptive_Market_Regression"] = dict(Annualization=config.annualization,
            **hac_market_regression(excess, benchmark_returns.Market, config.primary_hac_lag, config.confidence_level))
    return InferenceResult(summary, dsr, dependence, diagnostics, pd.DataFrame(bootstrap), pd.DataFrame(hac), multiplicity)

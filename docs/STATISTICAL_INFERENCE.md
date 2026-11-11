# Layer 3 — Statistical inference

## Implemented analysis

Layer 3 evaluates the saved nested-selector return stream using the Layer 2 ledger. It rebuilds the 182 registered candidates from saved scores, applies the same portfolio accounting, and evaluates every series over the same 4,915 return intervals.

The implementation provides empirical trial-dispersion Deflated Sharpe, an average-correlation sensitivity, paired circular-block Sharpe intervals, HAC mean-difference and market-regression inference, and a joint max-T multiplicity adjustment. Every statistic records its inputs, sample, seed, assumptions, and diagnostic status.

## Reproduction

Run from the project directory with the project's environment active.

```powershell
python -m pip install -e ".[execution,statistics]"
python -m regime_research.statistics_calibration --output runs/layer3_null_calibration
python -m regime_research.trial_returns --output runs/layer3_candidate_family
python -m regime_research.statistical_research --output runs/layer3_statistical_inference
python -m regime_research.statistical_research --verify runs/layer3_statistical_inference
python -m unittest tests.unit.test_statistics -v
```

Calibration and inference publication require a fresh output directory. Trial reconstruction reuses an existing family only after verifying its source and artifact checksums. The reconstruction validates the Layer 2 run and confirms that the recovered selected positions match the saved outer-fold decisions.

Saved scores supply the causal rank and allocation-state warmup before the evaluation period. Each evaluated candidate then begins with cash at the same first execution open. This establishes a common accounting start for the trial family and reported portfolio.

## Deflated Sharpe

The implementation uses Equation 2 from Bailey and López de Prado. All Sharpe quantities enter in daily units. The rejection threshold uses the sample variance of daily Sharpe estimates across the usable registered trials. Selected-series inputs include its daily Sharpe, sample size, skewness, and non-excess kurtosis. The analytical formula assumes a normal cross-trial Sharpe model and an IID selected-return sampling approximation. [The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

The numerical fixture transcribes pages 9–10 of the paper. Its 250-session annualization convention converts an annual Sharpe of 2.5 and annual trial-Sharpe variance of 0.5 to daily units. The implementation reproduces the published DSR values 0.9004 for 100 trials, 0.9505 for 46 trials, and 0.9505 for 88 trials with normal moments. The project reporting convention uses 252 sessions.

The raw-count scenario uses all 182 registered attempts. The dependence sensitivity uses the Appendix A.3 equal-weight correlation approximation

$$N_{effective}=\bar\rho+(1-\bar\rho)M.$$

Its unrounded count and the upward-rounded count are saved. Duplicate return streams remain in the dependence calculation, with duplicate identifiers disclosed. Negative or undefined average correlations produce an unsupported-sensitivity status. Constant or incomplete trial streams retain a diagnostic row and their place in the registered attempt count.

The effective count is a sample-based sensitivity estimate. Linear correlation, dependence beyond correlation, and correlation-estimation uncertainty affect its interpretation. The two reported scenarios expose the dependence assumption, while the raw-count scenario serves as the primary DSR calculation.

A DSR value is an analytical CDF-based significance measure under these assumptions. Its evaluation concerns Sharpe relative to a search-adjusted threshold. Paired benchmark comparisons address incremental performance.

## Paired Sharpe intervals

The primary interval uses 10,000 circular-block replications with 20-session blocks. Sensitivities use 2,000 replications each with 5- and 63-session blocks. Portfolio and benchmark returns receive identical sampled row indices within each replication. The final block is truncated to preserve the exact original sample size.

The output contains percentile intervals for portfolio Sharpe and delta-Sharpe, the fraction of bootstrap delta estimates above zero, Monte Carlo precision for that fraction, seeds, and valid replication counts. Replicates with undefined ratios are counted explicitly. An inadequate valid-replicate fraction produces an inconclusive status.

Sharpe uses daily net excess returns. Sortino uses RMS downside shortfall over all periods relative to the declared acceptable excess return. Drawdown starts with initial capital of one. Undefined denominators produce diagnostic results.

## Dependence-aware mean and multiplicity tests

HAC mean inference uses the Bartlett/Newey–West covariance estimator. The primary lag is 20 sessions, with 5- and 63-session sensitivity lags. The market regression separately estimates daily alpha and beta with a HAC sandwich covariance and saves the alpha interval and residual volatility.

The primary max-T family contains the 182 registered fixed candidates and the reported selector stream. Every candidate's mean excess-return difference is measured against Locked HGB and divided by its original-sample HAC standard error.

The procedure jointly resamples centered circular blocks across the entire candidate family and benchmark. It uses 2,000 replications, 20-session blocks, the exact original row count, and fixed original-sample HAC denominators. Its one-sided adjusted p-value is `(1 + exceedances) / (replications + 1)`. Zero-variance differences and unavailable series carry explicit statuses.

Mean-difference tests and Sharpe-difference intervals answer distinct statistical questions. Their result tables preserve that distinction.

## Calibration design and outcomes

The saved calibration protocol fixes seeds, simulation settings, Monte Carlo bands, and power requirements before running the simulations. It tests eight-trial families with 1,250 observations, a 500-observation burn-in, 400 datasets per scenario, and 399 bootstrap draws per dataset.

At a nominal 5% familywise level, the simultaneous Monte Carlo acceptance range is 8–34 rejections among 400 datasets for each of four null scenarios. The planted-edge check requires its one-sided 95% exact power lower bound to exceed 60%.

| Scenario | Rejections | Rate | Declared gate |
| --- | --- | --- | --- |
| Independent normal returns | 30 / 400 | 7.50% | Passed |
| AR(1) returns with coefficient 0.3 | 30 / 400 | 7.50% | Passed |
| GARCH returns with coefficients 0.08 and 0.90 | 21 / 400 | 5.25% | Passed |
| Correlated trials with correlation 0.7 | 22 / 400 | 5.50% | Passed |
| A planted 0.001 daily edge in the first trial | 346 / 400 | 86.50% | Passed |

The planted-power lower bound is 83.36%. The observed null rates quantify finite-sample behavior under the stated simulation designs. Calibration covers those eight-trial designs, while the empirical study has 183 tested return streams. The empirical p-value retains its bootstrap approximation and Monte Carlo uncertainty.

## Inference scope

Bootstrap and max-T calculations condition on the supplied realized return streams. Model fits, selected recipes, and outer-fold decisions stay fixed during these resampling calculations. Inference about the full adaptive selector would require a separate experiment that repeats the fitting and selection steps with valid chronology.

The 2007–2026 history remains retrospective nested pseudo-out-of-sample evidence. The DSR search scope is the 182-candidate registered family. The project-wide search history and genuine forward-evaluation protocol are specified for Layer 5.

## Current results

| Quantity | Result |
| --- | --- |
| Ledger-accounted nested-selector Sharpe | 0.452225 |
| Empirical variance across daily trial Sharpe estimates | 0.0000615476 |
| Mean off-diagonal trial correlation | 0.680108 |
| Implied independent count | 58.900493, rounded upward to 59 |
| Raw-count DSR | 0.688054 |
| Dependence-sensitivity DSR | 0.760103 |
| Primary adjusted max-T p-value | 0.234383 |
| Primary 95% delta-Sharpe interval versus Locked HGB | [−0.316454, 0.841369] |
| Primary 95% delta-Sharpe interval versus SMA 50/200 | [−0.554239, 0.164385] |

The incremental-performance evidence is inconclusive under the primary adjusted test and paired intervals. The study records the exact input dispersion and dependence assumptions behind both DSR estimates.

## Interfaces and artifacts

`infer_performance(portfolio_returns, benchmark_returns, trial_returns, inference_config)` returns an `InferenceResult`.

The portfolio frame contains `Net_Return` and `Net_Excess_Return`. Benchmark columns contain named excess-return series. Trial columns contain the registered candidates' excess returns. All three frames share an identical, unique, chronological index. The first benchmark is the registered primary comparator. A column named `Market` additionally enables descriptive alpha/beta inference.

| Artifact | Contents |
| --- | --- |
| `inference.json` | Sample, configuration, point estimates, assumptions, family summary, and calibration outcomes |
| `dsr_inputs.csv` | Every moment, dispersion, independent-count assumption, threshold, standard error, and DSR |
| `trial_dependence.csv` | Each usable trial's mean correlation with the remaining trials |
| `trial_diagnostics.csv` | Usable, duplicate, constant, and missing trial classifications |
| `bootstrap_summary.csv` | Paired Sharpe and delta-Sharpe intervals, seeds, and valid replication counts |
| `hac_summary.csv` | Mean-difference inference by benchmark and lag |
| `multiple_testing.csv` | HAC-studentized max-T statistics and adjusted p-values |
| `null_calibration.csv` | Frozen simulation outcomes, acceptance bands, precision, and gates |
| `published_example_check.json` | Reproduction of the paper's three numerical cases |
| `completion.json` | Source, runtime, parent artifacts, settings, and output checksums |

The verifier checks those signatures, reloads the trial family, verifies calibration provenance, and reproduces the statistical tables using their recorded settings.

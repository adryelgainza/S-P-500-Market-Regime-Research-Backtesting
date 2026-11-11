# Layer 4 — Attribution and matched controls

## Implemented analysis

Layer 4 evaluates the saved nested allocation alongside seven controls and the four locked model portfolios. The study uses the historical S&P 500 index snapshot, the same 4,915 next-open intervals, the Layer 2 ledger, an explicit zero cash rate, and 5 bps one-way costs.

The report separates equity price P&L, distributions, cash interest, and trading costs. It reports exposure, risk, turnover, invested episodes, return concentration, market beta, and HAC alpha uncertainty. Separate tables describe the predictive scores and their calibration.

## Reproduction

Run these commands from the project directory with the project environment active.

```powershell
python -m pip install -e ".[execution,statistics,attribution]"
python -m regime_research.attribution_research --output runs/layer4_attribution
python -m regime_research.attribution_research --verify runs/layer4_attribution
python -m unittest tests.unit.test_attribution -v
```

Choose a fresh output directory for a new run. The command verifies its parent trial family, reconstructs the selected recipes, estimates matching parameters, replays controls and randomized overlays, publishes a receipt, and adds the charts and results to the notebook's Layer 4 Markdown section.

## Controls and matching

| Control | Construction |
| --- | --- |
| Cash | Zero equity exposure with the common cash model |
| Buy and hold | Full equity target under the common daily-target execution policy |
| Past exposure match | Each fold fixes equity weight to its selected recipe's mean target exposure over the preceding eight years |
| Past volatility match | Each fold fixes weight to the ratio of its selected recipe's historical net-return volatility to market volatility, capped at one |
| Volatility only | A causal 10% volatility target using 20-session-span EWMA volatility, exposure capped at one, and 0.1 weight increments |
| SMA only | The existing 50/200 closing-price trend rule with next-open execution |
| Core without ML | The saved selected core and its volatility scaling, with the score multiplier set to one |

Matching uses the eight-year lookback specified in the control protocol. Historical recipe ledgers start in cash at the lookback window's first execution. Matching observations must have fully realized returns ending strictly before the nominal outer-fold decision boundary. The minimum matching sample is 252 observations.

`matching_audit.csv` saves the candidate, first and last estimation timestamps, row count, fixed weights, and first evaluation decision. Ten matching ledgers and their source rows are saved for inspection. Matching estimates are held fixed within each fold.

The core ablation retains the saved schedule of core choices. In a benchmark-selected fold, the core equals that benchmark. In an ML-selected fold, it retains the recipe's always-long or SMA core and any registered volatility scaling. This comparison isolates the score overlay conditional on those choices.

## Common accounting checks

Every portfolio shares the same instrument, dataset ID, prices, execution dates, return-end dates, initial capital, fees, distribution-payment convention, and reference cash rate. Attribution independently checks credited cash interest, including payments within an interval, against the common credited-rate series.

The wealth bridge uses dollar P&L components and reconciles them to final wealth. Fold and episode contributions use log returns so that their sums equal the portfolio's total log return. The first flat period after an invested episode carries that episode's exit cost. Episode zero aggregates uninvested gaps.

Downside capture is the mean portfolio net return on negative-market-return intervals divided by the mean market total return on those intervals. Exposure is the realized post-trade equity weight. Annual turnover is absolute traded notional divided by pre-trade wealth, summed and multiplied by 252 divided by the number of return intervals.

## Factor and paired inference

The descriptive factor regression uses strategy net excess return as the response and instrument excess return as the market factor. It reports beta with a HAC interval, daily and annualized alpha, the alpha HAC t-statistic and interval, residual volatility, sample size, and the 20-session HAC lag.

Paired Sharpe comparisons use 10,000 common circular-block draws of length 20. These are pointwise conditional 95% intervals. The seven one-sided HAC mean-difference tests receive a Holm adjustment within their declared comparison family. Cash has zero excess-return variance in this study, so its comparison uses mean-return inference.

The comparisons condition on the recorded return streams and saved fold decisions. They belong to retrospective attribution diagnostics. The frozen forward-validation design remains part of Layer 5.

## Randomized overlay reference

The protocol specifies 200 draws with a fixed seed sequence. Each draw permutes contiguous 63-session blocks of the model multiplier separately within each outer fold. The saved core remains attached to its original dates.

This preserves the multiplier values and within-block order in each fold. New block boundaries can change holding durations, turnover, and costs. A varying core can also change final exposure after the shuffled multiplier is applied. Every draw is replayed through the same ledger, and its realized exposure, target exposure, turnover, and cost mismatches are recorded.

The complete randomized weights, excess returns, and metric distribution are saved. The reported upper-tail fraction uses `(1 + draws at least as large as observed) / (draw count + 1)` as a retrospective reference measure.

For the current study, the observed Sharpe is 0.452225, the median randomized Sharpe is 0.396603, and the upper-tail reference fraction is 0.119403.

## Shuffled-score null check

The declared null simulation uses 400 independent datasets of 756 observations. Market returns have zero mean and 1% daily volatility. Persistent scores follow an AR(1) process with coefficient 0.8. Causal 126-session ranks with a 63-session minimum map scores to an 80th-percentile risk-off rule, followed by 63-session block randomization.

The check uses zero costs and cash interest, under which the daily-target ledger return is exactly equity weight times market return. Across datasets, the mean timing advantage relative to full exposure is tested against zero using the predeclared two-sided 99% Monte Carlo criterion.

The observed daily advantage is 0.000003679 with Monte Carlo standard error 0.000008448. Its z-statistic is 0.435, within the declared absolute limit of 2.576.

## Predictive diagnostics

The score analysis evaluates the 13 saved predictor recipes separately from portfolio performance. A label enters only after its recorded end time has matured by the final valuation. Tables include ROC AUC, average precision, rank correlation with negative forward returns, probability Brier score, downside-event precision and recall at the fixed 0.5 cutoff, equal-width probability calibration bins, and risk-score quantiles with conditional forward returns.

Regression scores receive ranking and conditional-return diagnostics. Probability diagnostics use the recipes that supply predicted probabilities. The target definitions and horizons differ across recipes, and forward horizons overlap. These tables describe score behavior for their respective targets.

## Current results

| Portfolio or control | Net excess-return Sharpe | Mean equity exposure |
| --- | --- | --- |
| Nested selector | 0.452225 | 80.63% |
| Core without ML | 0.549293 | 96.03% |
| Past exposure match | 0.653032 | 78.31% |
| Past volatility match | 0.647293 | 80.63% |
| Volatility only | 0.650172 | 72.05% |
| SMA only | 0.642479 | 76.66% |
| Buy and hold | 0.562082 | 100.00% |

The nested selector's market beta is 0.831538, with a HAC interval of [0.755314, 0.907762]. Its annualized regression intercept is −1.00%, with a HAC interval of [−4.17%, 2.17%] and a t-statistic of −0.619.

The point estimate of nested-selector Sharpe minus core-only Sharpe is −0.097068. Its paired 95% interval is [−0.226758, 0.034593]. The incremental-performance evidence remains uncertain under these conditional diagnostics.

## Interfaces and artifacts

`build_controls(protocol, prior_data)` returns a control-weight frame and matching audit. The input dictionary contains evaluation features and timestamps, mature historical recipe returns, and fold decisions. The function filters matching observations by their return-end times and checks that each history belongs to the selected candidate.

`attribute(portfolios, instrument_returns, control_config)` accepts named Layer 2 results and an aligned market frame containing `Market_Return`, `Market_Excess_Return`, `Reference_Return`, `Credited_Return`, and `Fold`. It returns attribution, factor, paired-comparison, episode, and fold tables.

| Artifact | Contents |
| --- | --- |
| `control_registry.json` | Protocol, controls, study scope, random reference, and shuffled-score null outcomes |
| `attribution.csv` | P&L bridge, risk, exposure, turnover, downside capture, and episode concentration |
| `factor_regression.csv` | Market beta, alpha, HAC uncertainty, and residual volatility |
| `paired_comparisons.csv` | Paired Sharpe intervals and Holm-adjusted HAC mean tests |
| `matching_audit.csv` | Fold-local estimation windows and fixed matching parameters |
| `matching_history.parquet` | Historical recipe observations used by the matching estimator |
| `matching_ledgers/` | Reproducible historical ledgers for the ten matching windows |
| `episode_contributions.csv` | Invested episodes, exit costs, cash gaps, and log contributions |
| `fold_contributions.csv` | Outer-fold performance and log contributions |
| `predictive_diagnostics.csv` | Per-recipe score and probability diagnostics |
| `probability_calibration.csv` | Predicted probability and realized event rate by bin |
| `score_quantiles.csv` | Conditional forward returns and event rates by score quantile |
| `random_control_distribution.parquet` | All 200 randomized metrics and accounting mismatches |
| `random_control_weights.parquet` | Every randomized target path |
| `random_control_returns.parquet` | Every randomized net excess-return path |
| `portfolios/` | Full ledger, target, trade, event, and reconciliation files for the 12 portfolios |
| `figures/` | Wealth bridge, exposure, paired differences, and fold/episode charts |
| `completion.json` | Source, data, notebook-code identity, and artifact checksums |

Verification checks signatures, all saved portfolio and matching ledgers, recomputed matching weights and attribution tables, every randomized weight path, and fresh ledger replays of the first, middle, and last randomized draws. The notebook section embeds the four figures and the main results.

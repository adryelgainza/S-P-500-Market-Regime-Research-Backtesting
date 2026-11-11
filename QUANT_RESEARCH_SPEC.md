# Quant Research Layer Specification

## Purpose and status

This specification defines the next version of the market-regime project. The research question is whether a model-driven allocation rule adds repeatable value after accounting for investable returns, cash interest, market exposure, portfolio drift, trading costs, and the search over strategies.

Layers 1–6 have engineering implementations. Data contracts and identified snapshots are documented in `docs/DATA_LAYER.md`. Self-financing accounting is documented in `docs/PORTFOLIO_ACCOUNTING.md`. Calibrated inference and the reconstructed historical trial family are documented in `docs/STATISTICAL_INFERENCE.md`. Matched controls and P&L attribution are documented in `docs/ATTRIBUTION.md`. Frozen protocols and experiment governance are documented in `docs/VALIDATION_GOVERNANCE.md`. Layer 6 adds the Git baseline, exact runtime lock, extracted feature/label/model/allocation modules, resumable per-refit checkpoints, strict fresh-kernel runner, immutable historical replay, independently verified completion manifests, CI tiers and explanatory notebook. Its supported reproduction path is in `docs/REPRODUCIBILITY.md`. The original corrected research is preserved in `archives/layer5_baseline/`. Transfer and forward performance await their registered empirical evaluations. A higher backtest Sharpe remains an empirical outcome.

The initial implementation will support daily, single-instrument, USD, long/cash portfolios with fractional shares and target equity weights between zero and one. SPY is the proposed first investable instrument. The historical S&P 500 price-index study remains a separately identified research track. Additional markets require a registered instrument list and availability rules before their evaluation begins.

## Existing baseline to preserve

The current corrected study is version 1.1.0. Its authoritative inputs and results are identified by the existing experiment registry and completed-run manifest.

| Baseline item | Current location or value |
| --- | --- |
| Notebook | `sp500_market_regime_classification.ipynb` |
| Historical price-index snapshot | `sp500_ohlcv.csv` |
| Experiment registry | `model_outputs/28_sharpe_experiment_registry.json` |
| Completed-run manifest | `model_outputs/46_sharpe_run_manifest.json` |
| Net nested-selector Sharpe | Approximately 0.452 |
| SMA 50/200 benchmark Sharpe | Approximately 0.642 |
| Interpretation | Retrospective nested pseudo-out-of-sample research |

The following corrections are already implemented and must remain covered by regression tests.

- Inner selection uses only returns whose realization ends strictly before the outer decision date.
- Labels carry explicit end dates and are purged across fitting and calibration boundaries.
- Forward adverse excursions include only prices encountered during the specified holding interval.
- Drawdown includes initial capital, and Sortino uses root-mean-square downside shortfall.
- Predictor caches validate provenance, content, date coverage, and purge audits.
- Completed outputs are bound to data, source, runtime, and notebook execution receipts.

The next version must archive this baseline before generating replacement results. Corrected accounting and different instruments will legitimately change performance figures.

## Architecture and common contracts

```text
Immutable market snapshots -> Features and labels -> Causal model scores
           |                                           |
           v                                           v
   Cash and action events                    Registered allocation rule
           |                                           |
           +----------> Self-financing execution <-----+
                                  |
                         Daily portfolio ledger
                                  |
                  Attribution and statistical inference
                                  |
                   Notebook, result tables, and report

Frozen protocols and experiment records govern every evaluation.
A clean runner and provenance manifest bind every completed run.
```

| Layer | Main responsibility | Proposed implementation |
| --- | --- | --- |
| 1. Data | Immutable, investable, availability-aware inputs | `src/regime_research/data/` |
| 2. Portfolio accounting | Shares, cash, actions, drift, and actual trading costs | `src/regime_research/execution.py` |
| 3. Statistical inference | Sharpe uncertainty, selection adjustment, and calibration | `src/regime_research/statistics.py` |
| 4. Attribution and controls | Separate model timing from exposure and generic risk management | `src/regime_research/attribution.py`, `controls.py` |
| 5. Validation governance | Frozen hypotheses, chronological testing, and experiment history | `src/regime_research/protocol.py`, `experiments.py` |
| 6. Reproducibility and presentation | Clean execution, reusable modules, verification, and explanation | `src/regime_research/runner.py`, notebook, CI |

### Shared conventions

- Every table has a versioned schema. Identifiers include dataset, instrument, strategy, protocol, and run where relevant.
- Store timestamps in UTC and preserve exchange timezone and session identifiers. Session dates alone cannot establish intraday availability.
- Features and fitted transformations may use only information available by the decision timestamp.
- A decision after close on session t executes at the open of session t+1. The next ordinary portfolio return ends at the open of session t+2.
- Store decision time, execution time, return-end time, label-end time, and refit time separately.
- Reject duplicate keys, non-finite prices or weights, invalid chronology, and missing required provenance. Record intentional exclusions explicitly.
- Evaluation uses a registered common date mask. A missing observation cannot silently become a zero return.
- Every randomized operation receives a recorded seed. Every performance figure identifies its sample, costs, cash convention, and uncertainty method.

## Layer 1 — Investable data and immutable snapshots

### Purpose

Provide auditable market inputs and distinguish the century-long price-index experiment from a strategy evaluated on an instrument that could have been held.

### Required components

1. Create separate adapters for the existing index snapshot, investable OHLCV, corporate actions, and historical cash rates.
2. Save raw downloads under a content-addressed dataset directory. Refreshing data creates a new snapshot and preserves earlier snapshots.
3. Record source, retrieval timestamp, request settings, provider adjustment settings, currency, exchange calendar, coverage, checksums, and redistribution constraints.
4. Use raw executable prices with explicit split and distribution events in the primary ledger. Keep adjusted prices in separately named fields for diagnostics and carefully defined features.
5. Reconcile raw-plus-actions total returns against a provider's adjusted-return series on suitable matching valuation intervals. Explain material differences and avoid adding distributions twice.
6. Separate data known at a decision time from final revised history. When true historical vintages are unavailable, record that limitation and the conservative availability rule used.
7. Validate OHLC relationships, positive prices, nonnegative volume, duplicates, missing sessions, action dates, and suspicious flat histories. Do not forward-fill missing tradable prices.
8. Convert cash-rate quotations using their actual units, day-count basis, compounding convention, publication time, and calendar accrual interval. A quoted annual yield is an input to a cash model, rather than a realized daily return.
9. Distinguish a reference risk-free rate from an attainable cash-crediting assumption. Save both if they differ and document any spread or fee.

### Minimum schemas

| Table | Required fields |
| --- | --- |
| Market bars | `Dataset_ID`, `Instrument_ID`, `Session`, `Open_Time`, `Close_Time`, raw `Open`, `High`, `Low`, `Close`, `Volume`, `Currency`, `Source_ID` |
| Actions | `Instrument_ID`, `Action_ID`, `Action_Type`, `Effective_Time`, `Ex_Date`, `Pay_Date`, `Amount_Per_Share`, `Split_Ratio`, `Available_At`, `Source_ID` |
| Cash observations | `Rate_Date`, `Available_At`, `Quoted_Rate`, `Rate_Units`, `Day_Count`, `Compounding`, `Source_ID` |
| Cash intervals | `Start_Time`, `End_Time`, `Credited_Return`, `Reference_Return`, `Observation_IDs`, `Availability_Policy` |
| Dataset manifest | `Dataset_ID`, `Schema_Version`, source requests, retrieval time, file hashes, calendar, adjustment policy, quality findings |

Fields that do not apply to an action type may be null under an explicit schema rule. Missing required dividend payment dates must trigger either rejection or a declared ex-date-credit approximation. Runs using that approximation must carry its identifier in their results.

### Interfaces and artifacts

- `load_dataset(dataset_id)` returns validated bars, actions, cash intervals, and a manifest without contacting the network.
- `build_eligible_sample(dataset, history_requirements)` returns eligible decision dates and an exclusion log.
- Store snapshots under `datasets/<dataset_id>/` and quality reports under the run directory.
- Derive each evaluation start from instrument inception, feature warmup, training window, calibration, and label purging. A 20-year SPY window cannot be satisfied with invented pre-inception prices.
- If index history supplies training features for an ETF evaluation, register this as a separate proxy-training experiment. Record the instrument change and basis risk.

### Tests and acceptance criteria

- A saved snapshot reloads offline with matching hashes and schema validation.
- Synthetic split and dividend fixtures reconcile across raw and adjusted representations without duplicate returns.
- Missing sessions and stale cash observations fail according to a registered policy.
- Weekend, holiday, publication-lag, and percentage-to-decimal cash-accrual fixtures pass.
- Changing future bars cannot change previously available raw features. Price-level and adjusted-feature policies receive separate revision tests.
- Every evaluated date has sufficient real history or a recorded exclusion reason.
- The data-quality report identifies the limitations of the early index history.

## Layer 2 — Self-financing portfolio accounting

### Purpose

Replace turnover based solely on target-weight changes with a ledger that accounts for holdings, cash, portfolio drift, distributions, and trade financing.

### Timing and accounting contract

At each execution open, process applicable splits, dividend entitlements, cash accrual, and settled cash flows in a documented order. Then value the existing holdings and execute the target generated at the preceding decision time.

Let P be the execution price, q the shares held before trading, C cash, and A the value of distribution receivables. Pre-trade wealth is

$$V^- = qP + C + A.$$

For a signed trade of x dollars and one-way proportional cost c,

$$q^+ = q + x/P, \qquad C^+ = C - x - c|x|, \qquad V^+ = V^- - c|x|.$$

The default target is a post-cost equity weight. Solve

$$\frac{qP+x}{V^- - c|x|}=w^*$$

subject to available cash, receivable restrictions, and the portfolio bounds. A trade solver must account for its own costs. If unsettled receivables prevent the target from being funded, execute the feasible amount and record the shortfall.

With zero cash interest and no actions, a 50% equity allocation followed by a 10% asset gain drifts to

$$w^- = \frac{0.5(1.10)}{0.5(1.10)+0.5} = 0.5238095.$$

Returning to a 50% target therefore requires a trade even when consecutive target weights are equal.

### Required behavior

- Hold explicit shares, cash, and receivables through time. Fractional shares are permitted in the first version.
- Support `daily_target` and `signal_change_only` policies. Register the policy per strategy. A daily fractional allocation rebalances drift, while a signal-change strategy may retain drift between signals.
- Attribute dividend entitlement to holdings established before the ex-date event. A purchase at the ex-date open receives no entitlement to that distribution.
- Recognize receivables on the ex-date and credit cash on the payment date when dates are available. Receivables do not earn cash interest.
- Apply splits consistently to shares, prices, and per-share action amounts.
- Charge costs on actual traded dollars. Define one-way turnover as absolute traded notional divided by pre-trade wealth, and disclose the annualization rule.
- Report entry, rebalance, switch, and optional terminal-liquidation costs. Mark final holdings at the final valuation time by default and report liquidation as a separate registered variant.
- Separate gross asset P&L, distribution P&L, cash interest, and trading costs. ETF expenses already embedded in observed returns must not be deducted again.
- Begin with the existing 5 bps one-way assumption. Run a fixed 0, 5, 10, and 20 bps sensitivity grid and document what each cost includes.
- Describe open-price fills as an execution assumption. Daily OHLCV alone cannot establish spread, opening-auction capacity, or realized market impact.
- Keep cash and share holdings nonnegative within numerical tolerance. Do not silently borrow to satisfy a target.

### Interfaces and artifacts

- `simulate_portfolio(bars, actions, cash_intervals, target_weights, execution_config)` returns a portfolio result.
- Targets include `Strategy_ID`, `Decision_Time`, `Execution_Time`, `Target_Weight`, `Rebalance_Policy`, and `Rule_Version`.
- Save `portfolio_daily.parquet`, `trades.parquet`, `cash_events.parquet`, and `accounting_reconciliation.json` per run.
- Daily rows include wealth before and after trading, shares, cash, receivables, target and realized weights, traded notional, turnover, cost, P&L components, valuation time, and return-end time.
- Period returns span pre-trade wealth at consecutive execution opens and include the first open's trading costs exactly once. The last valuation has no new trade unless liquidation is requested.
- Compute excess returns against the reference cash return over precisely the same interval.

### Tests and acceptance criteria

- Hand-worked fixtures cover all-cash, buy-and-hold, constant 50% targets, drift without rebalancing, entry/exit, and strategy switches.
- Distribution entitlement, payment, and split fixtures preserve the expected economic wealth.
- Flat prices and zero cash rates leave only the expected cost loss.
- The trade solver achieves the target to tolerance when funding permits and logs a feasible shortfall otherwise.
- No-cost wealth identities hold to relative tolerance 1e-10 on synthetic fixtures.
- An independently written small reference ledger agrees on randomized short paths to relative tolerance 1e-8.
- Replaying fixed baseline targets isolates the accounting change before any model refitting. Save the difference in costs, turnover, returns, and Sharpe.
- Strategy switches across outer folds use the actual continuing holdings. They cannot reset capital or erase transition costs.

## Layer 3 — Statistical inference and selection adjustment

### Purpose

Make uncertainty estimates reproducible and distinguish evidence about a fixed return stream from evidence about an adaptive selection procedure.

### Deflated Sharpe correction

Implement the published Deflated Sharpe construction using variance across trial Sharpe estimates and an explicit independent-trial assumption. Equation 2 defines the adjustment, and Appendix A.3 discusses dependent trials. See [Bailey and López de Prado, The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

Use daily Sharpe units throughout the calculation. For trial Sharpe dispersion s, independent-trial count N, Euler's constant gamma, and standard normal CDF Phi, the reference threshold is

$$SR_0 = s\left[(1-\gamma)\Phi^{-1}(1-1/N)+\gamma\Phi^{-1}(1-1/(Ne))\right].$$

With sample Sharpe SR, sample length T, skewness g3, and non-excess kurtosis g4,

$$DSR = \Phi\left(\frac{(SR-SR_0)\sqrt{T-1}}{\sqrt{1-g_3 SR+(g_4-1)SR^2/4}}\right).$$

### Required behavior

- Derive cross-trial Sharpe dispersion from the registered, aligned candidate family. Record its variance estimator, sample dates, and excluded degenerate series.
- Report the raw registered trial count and the count of distinct usable return streams. Track failed or unmeasurable trials separately without erasing their existence.
- Report a raw-count DSR scenario and a dependence sensitivity scenario. For the latter, use the documented Appendix A.3 equal-weight average-correlation approximation and save the correlation estimate and implied count. Treat the raw-count scenario as conservative relative to this nonnegative-correlation approximation.
- Compute dependence over usable registered trial streams, including duplicate streams, and disclose any trials unavailable for that calculation. Round a noninteger implied count upward for the expected-maximum approximation, preserving the unrounded estimate. If average correlation is negative or undefined, mark this dependence scenario unsupported and retain the raw-count result with its assumptions. Do not present an implied count as exact.
- Special-case N=1, zero dispersion, insufficient observations, zero return variance, invalid moments, and nonpositive variance expressions. Return a diagnostic status where the statistic is undefined.
- Keep the analytical DSR's assumptions visible. Do not interpret its CDF value as a Bayesian probability of profitable future trading.
- Preserve paired block bootstrap comparisons on common dates. Resample the same time blocks jointly across strategies and benchmarks.
- Keep the existing 20-session primary block length and 5/63-session sensitivity lengths unless a new protocol freezes a replacement before evaluation.
- Retain 10,000 primary paired-bootstrap replications and 2,000 per sensitivity scenario as initial defaults. Record seeds, Monte Carlo precision, and any changed replication count in the protocol.
- Use dependence-aware studentization for the max-T procedure. Record the statistic, null centering, family membership, block construction, and finite-replicate p-value rule.
- Keep HAC mean-return and alpha inference distinct from Sharpe inference. Start with the existing 20-session lag and register lag sensitivity before evaluation.
- A bootstrap of saved selected returns measures uncertainty conditional on that stream. Claims about the full selector require a separately specified resampling experiment that repeats the selection steps with valid chronology, or an explicitly conditional claim.
- Define annualized arithmetic excess Sharpe using 252 sessions. Label it as the reporting convention and accompany it with dependence-aware uncertainty. Use elapsed calendar time for CAGR.
- Preserve initial-capital drawdown and RMS downside-deviation definitions. Record the minimum acceptable return used by Sortino, and mark ratios with zero denominators as undefined.

### Interfaces and artifacts

- `infer_performance(portfolio_returns, benchmark_returns, trial_returns, inference_config)` returns estimates, intervals, tests, and diagnostics.
- Save `inference.json`, `dsr_inputs.csv`, `trial_dependence.csv`, `bootstrap_summary.csv`, and `null_calibration.csv`.
- Store every numerical input needed to recompute DSR, including daily Sharpe, moments, T, dispersion, N, and scenario.
- Identify whether each result concerns a fixed candidate, a saved selector stream, or a repeated selection process.

### Tests and acceptance criteria

- Reproduce the paper's numerical example from a fixture transcribed with its page reference and original annualization convention. Agree within the paper's published rounding.
- Verify daily-to-annual Sharpe unit conversions independently.
- Duplicate-candidate and highly correlated-candidate fixtures exercise the dependence policy.
- Joint simulations cover independent returns, serial dependence, volatility clustering, and correlated trial families under a known no-edge null.
- Before inspecting simulation results, register replication counts and Monte Carlo acceptance bands. A nominal 5% familywise rejection rule must stay within those bands under supported nulls.
- Include a planted-edge simulation to check that the pipeline can detect a sufficiently strong signal.
- Undefined or unstable statistics appear with explanations and cannot silently satisfy promotion gates.

## Layer 4 — Attribution and matched controls

### Purpose

Measure the contribution of model timing after separating broad market exposure, cash returns, and generic trend or volatility management.

### Required controls

| Control | Question it addresses | Construction rule |
| --- | --- | --- |
| Cash | What does remaining uninvested earn? | Same cash-crediting and reference-rate conventions |
| Buy-and-hold | Does allocation improve upon passive ownership? | Same instrument, entry date, actions, and ledger |
| Constant equity allocation | Is average market exposure sufficient? | Weight fixed from development data or earlier realized folds |
| Volatility-only allocation | Does generic risk scaling explain the result? | Same causal volatility estimator, limits, and target as the model rule |
| SMA-only allocation | Does a simple trend filter explain the result? | Registered 50/200 rule with the same timing and costs |
| Core without ML overlay | What does adding the model change? | Identical core, rebalance policy, and risk controls |
| Randomized timing overlays | Is the timing better than comparable random timing? | Registered block randomization preserving relevant allocation structure |

### Required analysis

- Produce a P&L bridge from equity price moves, distributions, cash interest, and costs to total wealth change.
- Compare the model overlay to each applicable control using common dates, the same ledger, and the same reporting definitions.
- Fit the descriptive regression of strategy excess return on instrument total excess return. Report beta, annualized intercept, HAC uncertainty, residual volatility, and sample size.
- Report average and realized exposure, volatility, maximum drawdown, downside capture, turnover, invested episodes, and concentration of returns across episodes and folds.
- Evaluate predictive scores separately through calibration, ranking, downside-event precision/recall, and conditional forward returns. Distinguish descriptive conditioning from a tradable rule.
- Estimate prospective matching weights or scaling parameters using development data or information available before each fold. A full-evaluation exposure or volatility match is permitted only as a labeled descriptive diagnostic.
- For random controls, shuffle whole allocation blocks or invested/cash episodes rather than individual independent days. Preserve exposure and holding-duration structure where feasible, rerun the ledger, and report the actual turnover and cost mismatch.
- Randomized overlays are retrospective reference distributions. They cannot be presented as signals that were available in real time.
- Save fixed seeds and the full distribution of randomized comparison metrics. Do not choose a favorable random-control draw.

### Interfaces and artifacts

- `build_controls(protocol, prior_data)` produces causal control targets.
- `attribute(portfolios, instrument_returns, control_config)` produces comparison tables and diagnostics.
- Save `control_registry.json`, `attribution.csv`, `factor_regression.csv`, `paired_comparisons.csv`, and `random_control_distribution.parquet`.
- Add notebook charts for the wealth/P&L bridge, exposure through time, paired performance differences, and fold/episode contributions.

### Tests and acceptance criteria

- A constant-beta synthetic strategy has the expected beta and near-zero alpha within simulation uncertainty.
- A model with shuffled scores produces no systematic timing advantage under a calibrated null.
- Every prospective matching parameter has an estimation-end timestamp earlier than the evaluation decision.
- Controls cannot gain an advantage through different dividends, cash rates, costs, or missing-date handling.
- Attribution components reconcile to the ledger. Compounded returns are reconciled through wealth rather than summed as percentages.
- The notebook answers whether ML adds value beyond its underlying core, with uncertainty and explicit economic size.

## Layer 5 — Frozen validation and experiment governance

### Purpose

Prevent later choices from being described as earlier commitments and make the complete research search reviewable.

### Protocol requirements

Create a machine-readable protocol under `protocols/` with the following fields.

| Field group | Required content |
| --- | --- |
| Hypothesis | Economic rationale, predicted behavior, primary estimand, and falsification condition |
| Universe | Instruments, currencies, inception rules, calendars, historical availability, and market inclusion rationale |
| Data | Dataset IDs, adjustment and cash policies, quality exclusions, and availability assumptions |
| Models | Features, targets, fit windows, preprocessing, calibration, random seeds, refit schedule, and admissible recipes |
| Allocation | Score mapping, portfolio bounds, rebalance policy, trend/volatility overlays, and benchmark eligibility |
| Evaluation | Development dates, inner/outer schedules, purge logic, external-market tests, and forward start |
| Inference | Primary comparison, family membership, uncertainty method, multiplicity adjustment, thresholds, and sensitivity analyses |
| Operations | Runtime lock, source revision, monitoring frequency, missing-data policy, stopping rules, and reporting schedule |

### Experiment register

- Use an append-only event log under `experiments/events.jsonl` with unique event IDs and linked previous-event hashes.
- Record hypothesis, configuration hash, source revision, dataset ID, parent experiment, stage, start/end times, seed, outcome, failure, and artifact references.
- Append status transitions and corrections as new events. Preserve abandoned and failed experiments.
- Import the existing 182-candidate study as a historical family with its known scope. Disclose that earlier informal research attempts may not be fully reconstructable.
- Distinguish the registered family count used by a statistical calculation from the broader project search history.
- Hash chaining detects alteration relative to a trusted saved head. Version-controlled or archived heads provide the audit anchor.

### Validation stages

1. Use already inspected history for implementation, debugging, and retrospective development.
2. Freeze a small economically justified candidate set and the selection algorithm before any new evaluation. Benchmark strategies may be eligible choices only when explicitly registered. Report how often the selector chooses them.
3. Evaluate the unchanged rule on a registered set of additional markets. Report per-market results and an aggregate with predetermined weights. Avoid per-market parameter tuning and dates selected after seeing performance.
4. Treat external-market results as evidence of transfer, with a record of which histories were previously inspected. Reusing contemporaneous markets does not create independent future observations.
5. Start forward evaluation at the first eligible decision after the protocol's actual freeze timestamp. Existing history cannot be relabeled as a future holdout.
6. Follow the registered review and stopping schedule. Any redesign creates a new protocol and a new forward-evaluation period.

Training and calibration may update during forward evaluation only according to the frozen algorithm. Each training row must be available by the refit, and every target must have fully matured before it is used. Inner selections must use realized returns ending strictly before their decision time. Truncating all data after a decision must leave that decision unchanged.

### Evidence and promotion gates

Maintain two separate gate sets.

- **Research validity gates** cover chronology, accounting, provenance, reproducibility, trial disclosure, and calibrated tests. These are required for a completed research release.
- **Strategy evidence gates** cover incremental net performance versus the registered primary control, uncertainty, risk, cost sensitivity, and enough independent episodes or evaluation time. Freeze numerical thresholds before evaluation. A failed strategy gate is a reportable result.

The protocol must state a minimum economically meaningful improvement, a primary confidence level, the relevant multiple-testing family, and the minimum evaluation length or episode count. Do not import arbitrary thresholds merely because they let the current strategy pass. Undefined inference or inadequate sample size produces an inconclusive result.

### Tests and acceptance criteria

- Editing a frozen protocol changes its hash and requires a new version.
- Chronology tests catch immature labels, late cash observations, and returns that cross a selection boundary.
- A repeated run links to the same hypothesis without being disguised as a new independent idea.
- All registered candidates and failures appear in the experiment history and final comparison table.
- The historical, external-market, and forward-evaluation result sets are clearly distinguishable.
- A freeze receipt includes timestamp, source revision, dependency lock, datasets, protocol hash, and the anchored experiment-log head.

## Layer 6 — Reproducibility, packaging, and notebook presentation

### Purpose

Make the result independently runnable, auditable, and understandable without relying on an interactive notebook's hidden state.

### Required engineering

- Establish a functioning Git repository, appropriate ignore rules, and a recorded baseline commit. Handle large data and generated outputs through documented snapshot storage or release artifacts.
- Add `pyproject.toml` and a dependency lock for the supported Python version. Pin the notebook execution environment as well as numerical libraries.
- Extract data preparation, features, labels, models, allocation, execution, inference, and reporting into importable modules in stages. Preserve public behavior during each extraction.
- Replace notebook-global configuration with explicit validated configuration objects. Functions return typed or schema-validated results rather than reading mutable notebook state.
- Provide an offline replay command and an explicit data-refresh command. A result replay cannot silently download different inputs.
- Provide a supported runner that creates a new kernel for notebook execution, executes the saved code cells in order, and records each cell ID and source hash.
- Replace the current ordered-subsequence history check with a strict execution receipt for the supported runner. Unexpected code execution, reordered cells, or an untracked mutation invalidates the receipt.
- Bind the runner's package revision, protocol, data, environment, outputs, notebook source, and executed-cell record in one completed-run manifest.
- Keep scientific cache keys separate from presentation-only code where dependencies can be established safely. Unknown dependencies cause invalidation.
- Checkpoint expensive fits per recipe/refit with validated hashes and atomic writes. An interrupted run may reuse only complete, compatible checkpoints.
- Publish a completed manifest only after all required outputs reconcile and verification passes. Failed or partial runs remain visibly incomplete.
- Provide concise progress records for long refits, including completed recipes and reusable checkpoints.

### Proposed layout

```text
src/regime_research/
    data/
    features.py
    labels.py
    models.py
    allocation.py
    execution.py
    statistics.py
    attribution.py
    controls.py
    protocol.py
    experiments.py
    runner.py
    reporting.py
tests/
    fixtures/
    unit/
    integration/
    simulations/
protocols/
experiments/
datasets/<dataset_id>/
runs/<run_id>/
pyproject.toml
<dependency lockfile>
.github/workflows/tests.yml
sp500_market_regime_classification.ipynb
README.md
```

Use one chosen lockfile format during implementation. Existing sidecar scripts can remain compatibility entry points while their functions move into the package.

### Verification tiers

| Tier | When it runs | Required coverage |
| --- | --- | --- |
| Fast offline checks | Each change and CI | Schemas, chronology, ledger identities, DSR fixtures, cache validation, and small reference examples |
| Small integration run | Each research-code change | Synthetic or fixed compact data, all pipeline stages, manifest and notebook receipts |
| Statistical calibration suite | Inference changes and research releases | Registered null/planted-edge simulations and Monte Carlo acceptance bands |
| Full research replay | Scientific changes and releases | All registered candidates, artifacts, attribution, inference, and independent verification |
| Forward append | Registered evaluation schedule | New observations, locked decisions, event log, and matching manifest |

Do not require a complete multi-hour model refit for a documentation-only edit. Numerical reproducibility tolerances must be registered for supported platforms, while source/configuration and artifact checksums remain exact within a completed run.

### Notebook requirements

The notebook becomes the explanatory interface to the tested package. It must include the following sections in reading order.

1. Research question, economic hypothesis, data tracks, and known sample limitations.
2. Signal, execution, return, label, and refit timeline with a worked date example.
3. Model objective and derivation, target definitions, calibration, and causal feature construction.
4. Self-financing ledger example showing drift, cash interest, a distribution, and transaction costs.
5. Registered validation design and the difference between retrospective and forward evidence.
6. Predictive diagnostics, portfolio results, and matched-control attribution.
7. Dependence-aware uncertainty, trial adjustment, and gate outcomes.
8. Robustness, failure cases, limitations, and an evidence-based conclusion.
9. Reproduction instructions, protocol IDs, and run manifest links.

Charts must identify the instrument, sample, cash and cost assumptions, and whether the series is gross or net. Save chart data alongside figures. Keep legacy static classification and full-sample clustering clearly identified as historical or descriptive analyses.

### Tests and acceptance criteria

- A fresh environment installs from the lock and runs the offline verification commands documented in README.
- A fresh-kernel notebook execution produces a valid completion receipt. An injected extra code cell invalidates the receipt or starts a differently identified run.
- Cold-cache and warm-cache runs agree on outputs within declared numerical tolerances.
- Data changes, model-code changes, and incompatible environments invalidate affected caches. Pure explanatory Markdown changes do not trigger model refitting.
- Termination during fitting leaves usable completed checkpoints and no completed-run receipt.
- The independent verifier recomputes key accounting and metrics from saved inputs rather than trusting summary tables.
- README provides one clear reproduction path, interpretation of results, runtime expectations, and data-access requirements.

## Implementation sequence

| Phase | Work | Completion condition |
| --- | --- | --- |
| A. Preserve and extract | Archive the current baseline, establish source control and lock, define schemas, expose existing inputs and fixed targets through package interfaces | The current corrected study remains reproducible and its archived manifest is intact |
| B. Correct the core | Implement drift-aware accounting, faithful DSR inputs, dependence-aware test calibration, and strict clean-kernel receipts | Toy/reference fixtures and focused regression checks pass, with isolated baseline-change reports |
| C. Add investable inputs | Implement immutable SPY/action/cash snapshots, quality checks, and eligibility rules | Raw-plus-actions and cash fixtures reconcile, with no fabricated history |
| D. Explain performance | Build matched controls, overlay ablations, P&L decomposition, and beta/alpha diagnostics | Every control shares the accounting contract and incremental-value comparisons are reproducible |
| E. Freeze and evaluate | Register the small candidate set, external-market design, inference, gates, and actual forward start | A complete freeze receipt exists before new evaluation outcomes are inspected |
| F. Release the project | Complete module extraction, notebook explanations, CI tiers, verification, and reproducibility instructions | A fresh installation can reproduce the release and inspect its complete experiment history |

Each phase starts with the smallest valid slice of the required contracts. Package scaffolding and tests begin in Phase A and expand with each layer. A migration changes one scientific assumption at a time where possible, with a saved before/after explanation.

## Final completion checklist

- [x] The historical index study is archived with its original provenance.
- [x] Investable data include explicit actions, cash conventions, availability policies, and quality reports.
- [x] Portfolio wealth reconciles through shares, cash, receivables, and actual trading costs.
- [x] Statistical adjustments expose their inputs, assumptions, dependencies, and simulation calibration.
- [x] Controls establish whether the model contributes beyond exposure, trend, and volatility management.
- [x] Protocols and experiment events preserve the complete known search history.
- [x] Fresh forward observations are distinguished from all inspected historical evidence.
- [x] A clean environment and fresh kernel reproduce the released results.
- [x] The notebook explains the economics and mathematics and supports its conclusions with saved evidence.
- [x] Research validity and strategy-performance outcomes are reported separately.

Transfer and forward performance are pending empirical results. The historical search disclosure explicitly records incompletely reconstructable informal earlier attempts.

A strong finished project can conclude that the tested model has limited incremental predictive value. Its contribution is the rigor and clarity with which that conclusion is established.

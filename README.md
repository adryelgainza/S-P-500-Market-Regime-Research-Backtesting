# Market-regime research

An auditable study of whether model-driven allocation adds value beyond market exposure, trend and risk management. The project combines causal feature/model pipelines, self-financing execution, dependence-aware inference, matched controls, frozen protocols and strict notebook execution.

## Findings and evidence status

The corrected historical nested strategy has net excess-return Sharpe approximately **0.452**, below the SMA 50/200 result of approximately **0.642**. Matched exposure/volatility controls and the invested core also have stronger point estimates. The paired model increment remains unestablished. These are retrospective S&P 500 price-index proxy results with zero cash return and omitted distributions.

The investable SPY track uses identified executable-price/action snapshots, a lagged DFF cash proxy and explicit dividend-payment assumptions. New package refits on that inspected history are developmental. The fixed transfer and forward studies await their registered observations and contemporaneous decisions.

## Reproduce the project

Use **CPython 3.13.7 on Windows x86-64** and run from the project directory. The release inputs include the identified `archives/layer5_baseline/baseline.zip` alongside the Git-tracked manifest. Large data and result snapshots are stored as release artifacts outside Git.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m regime_research.runner run --mode replay --output runs/release_replay
.\.venv\Scripts\python.exe -m regime_research.runner verify runs/release_replay
```

The offline replay verifies the archive, its historical source, all 182 trial streams, accounting, inference, attribution and experiment anchors. It then executes the explanatory notebook in a private fresh kernel. Inspect `executed_notebook.ipynb`, `completion.json`, `execution_receipt.json` and the saved tables/figures inside the run directory. Allow tens of seconds to a few minutes, with additional disk work during first materialization.

The full workspace data/result snapshot is identified by `archives/layer6_release/manifest.json`. Extract its `workspace_inputs_and_results.zip` into the matching checkout to restore the saved ledgers, historical reports and all recorded development/release evidence. The verified default replay needs the companion baseline ZIP. The active source and version 2 freeze bundle are tracked in Git.

An explicit fixed-recipe refit uses the same package entry point.

```powershell
.\.venv\Scripts\python.exe -m regime_research.runner run --mode fit --output runs/fixed_recipe_development
.\.venv\Scripts\python.exe -m regime_research.runner verify runs/fixed_recipe_development
```

Per-recipe/refit checkpoints preserve completed work across interruptions. A second compatible fit reuses those blocks. This development study rebuilds the annual downside HGB recipe, allocation and three same-ledger portfolios across the registered 0, 5, 10 and 20 bps cost grid.

## Package and notebook

| Component | Responsibility |
| --- | --- |
| `data/` | Immutable snapshots, exchange calendars, action/cash conventions, quality checks and eligibility |
| `features.py`, `labels.py` | The original 98 causal indicators and execution-aligned matured targets |
| `config.py`, `models.py` | Validated configuration, purged HGB/RF fitting, feature selection and calibration |
| `allocation.py`, `execution.py` | Causal state machines and shares/cash/receivable accounting |
| `statistics.py`, `attribution.py`, `controls.py` | Dependence-aware inference, P&L decomposition and matched references |
| `protocol.py`, `experiments.py`, `validation.py` | Frozen commitments, audit history, chronology and separate evidence gates |
| `checkpoints.py` | Content-addressed, checksummed, atomic per-refit score checkpoints |
| `runner.py`, `kernel.py` | Exact saved-cell execution in one fresh kernel and independently verified completion |
| `research.py`, `reporting.py`, `archive.py` | Offline replay, fixed-recipe study, chart data and immutable historical source/results |
| `sp500_market_regime_classification.ipynb` | Economics, model derivations, worked timeline/ledger, results and reproduction links |

The notebook has nine sections covering the research question, timeline, model derivation, accounting, registered validation, prediction/portfolio diagnostics, inference, limitations and reproduction. The original static KNN and descriptive clustering notebook remains in the immutable baseline.

## Frozen design

`protocols/layer6_validation_v2.json` defines `downside-overlay` version `2.0.0`. It uses one annual-refit downside HGB model on SPY, the `AlwaysLong_Step_P90` allocation and buy-and-hold as the primary control. QQQ and IEF have predetermined 50/50 transfer weights. The economic minimum is 0.10 delta-Sharpe. The fixed forward review requires 756 completed scheduled intervals and at least 10 invested episodes, with registered uncertainty, multiplicity, drawdown and cost gates.

The new version binds the extracted implementation, explicit UTC refit boundaries and strict runner. Its actual freeze starts its own future-evidence window. The source-addressed version 1 receipt remains archived. Known historical searches, failures, repeated attempts and corrections remain in `experiments/events.jsonl`.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m unittest test_research_integrity -v
python -m regime_research.runner verify-archive
python -m regime_research.validation_research verify --frozen experiments/frozen/downside-overlay/2.0.0
```

The Windows CI workflow installs the same pins and runs unit and real-kernel integration tests. Its manual calibration option runs the registered null/planted-edge Monte Carlo suite. Tests cover extraction golden values, chronology, accounting, statistical fixtures, cache corruption/invalidation, interrupted fits, strict execution, source mutation and incomplete publication.

The local release passed **141 tests** and all **five calibration scenarios**. The fresh-install and clean-checkout replays passed. The full cold and warm SPY runs agree across **71 scientific tables**, including all 14 refits and 12 portfolio-cost combinations. The release summary and completed notebook are available under `runs/layer6_release/verification.json` and `runs/layer6_release_replay/executed_notebook.ipynb` after restoring the result snapshot.

## Explicit data acquisition

```powershell
python data.py import-index
python data.py refresh-spy --end 2026-07-24
python data.py verify <dataset_id>
```

`refresh-spy` makes network requests and publishes a new identified snapshot. Replay reads exact local inputs. Current limitations include revised historical data, approximate distribution payment dates, a DFF-linked cash proxy, omitted taxes/market impact, correlated transfer histories and incompletely reconstructable informal earlier search history.

## Guides

- [Reproducibility and supported runner](docs/REPRODUCIBILITY.md)
- [Data contracts and assumptions](docs/DATA_LAYER.md)
- [Portfolio accounting](docs/PORTFOLIO_ACCOUNTING.md)
- [Statistical inference](docs/STATISTICAL_INFERENCE.md)
- [Attribution and controls](docs/ATTRIBUTION.md)
- [Validation governance](docs/VALIDATION_GOVERNANCE.md)
- [Six-layer specification and status](QUANT_RESEARCH_SPEC.md)

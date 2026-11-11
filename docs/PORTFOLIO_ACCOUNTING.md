# Layer 2 — Portfolio accounting

## Implemented scope

The execution engine tracks a single USD instrument, fractional shares, settled cash, and distribution receivables. Equity targets range from zero to one. It records every fill, entitlement, payment, interest credit, valuation, and portfolio return.

Run the fixed-target replay and its independent verification from the project directory.

```powershell
python -m pip install -e ".[execution]"
python -m regime_research.accounting_replay --output runs/layer2_accounting_replay
python -m regime_research.accounting_replay --verify runs/layer2_accounting_replay
python -m unittest tests.unit.test_execution -v
```

Choose a fresh output directory for each replay. Publication is atomic, and a completed directory has a receipt binding its inputs, source, datasets, and artifacts.

## Signal and valuation timeline

For a decision after session t's completed bar, the target executes at the next session's open. The supplied bars include the preceding decision session so the engine can check availability and execution timing. Simulation begins at the first target execution and ends at the configured final valuation open.

Daily-target strategies supply a target for every execution session. Signal-change strategies may supply sparse decisions. Between those decisions, they retain their shares and cash. A change of strategy ID, rule version, allocation, or rebalance policy counts as a new signal.

Each daily return spans pre-trade wealth at consecutive opens. It includes the first open's trading cost and the ending open's price, distribution, and cash effects. A regular trade at that ending open belongs to the following return interval. Optional final liquidation belongs to the last return interval.

## Financing and portfolio drift

Pre-trade wealth includes equity, cash, and outstanding receivables. A trade of signed dollar notional x changes shares by x divided by the open price and reduces cash by x plus the proportional trading cost.

The trade solver chooses x to achieve the requested equity weight after its own costs. It constrains purchases to settled cash and sales to existing shares. Any funding shortfall appears in the valuation ledger.

A 50% equity allocation followed by a 10% equity gain drifts to 52.38% equity. A daily 50% target sells enough shares to restore 50%. A signal-change strategy continues holding the existing shares until a new signal arrives.

The cost grid uses 0, 5, 10, and 20 bps per one-way traded dollar. Turnover is absolute traded notional divided by wealth immediately before that trade. Annual turnover multiplies total one-way turnover by 252 divided by the number of return intervals. Fractional holdings and the cost convention avoid a minimum account-size assumption.

## Corporate actions and cash interest

At an ex-date open, splits first adjust the existing shares. Distribution entitlement then uses those post-split shares and the stated post-split per-share amount. A purchase at the ex-date open receives its first entitlement on a subsequent applicable ex-date. A sale at the ex-date open retains the distribution earned by the prior holding.

Entitlement adds a receivable at face value. Selling shares or subsequently splitting them leaves the earned dollar receivable intact. Payment transfers that asset to cash.

Known date-only payments settle at midnight in the configured exchange timezone, or at the ex-date open if that occurs later. This convention permits payments during closed sessions. The SPY snapshot uses its registered ex-date-credit approximation for missing payment dates.

Cash interest follows the Layer 1 interval's credited return. For a payment within an interval, the engine prorates that return by the remaining local calendar time and credits the interest at the next open. Interest on starting cash spans the whole interval. Unpaid receivables contribute zero cash interest. This simple interpolation is the implementation choice for intermediate payments, including intervals that cross daylight-saving transitions.

An optional terminal liquidation sells the equity holding at the final open and charges the configured cost. Outstanding receivables retain their face value in final wealth.

## API and target schema

The main API is `simulate_portfolio(bars, actions, cash_intervals, target_weights, execution_config)`.

| Target field | Contract |
| --- | --- |
| `Strategy_ID` | Identifier of the allocation strategy active at execution |
| `Decision_Time` | UTC time at or after the preceding bar's availability and before execution |
| `Execution_Time` | Exact next-session open in the supplied bars |
| `Target_Weight` | Finite post-cost equity weight in [0, 1] |
| `Rebalance_Policy` | `daily_target` or `signal_change_only` |
| `Rule_Version` | Recipe or rule identifier used to detect strategy switches |

Use Layer 1 tables directly, with a target frame following the schema above.

```python
from regime_research.execution import (
    ExecutionConfig,
    portfolio_metrics,
    save_portfolio_result,
    simulate_portfolio,
    verify_portfolio_result,
)

config = ExecutionConfig(
    one_way_cost=0.0005,
    dividend_payment_policy="ex_date_credit_approximation",
)
result = simulate_portfolio(
    dataset.bars, dataset.actions, dataset.cash_intervals, targets, config
)
metrics = portfolio_metrics(result)
save_portfolio_result(result, "runs/my_accounting_run")
verify_portfolio_result("runs/my_accounting_run")
```

The configuration supports initial capital, exchange timezone, numerical tolerance, an explicit final open, and optional terminal liquidation. The default final valuation marks the remaining holdings at the last supplied open.

## Artifacts and verification

| Artifact | Contents |
| --- | --- |
| `portfolio_daily.parquet` | Interval returns, reference cash returns, pre/post trading wealth, weights, holdings, turnover, costs, and P&L components |
| `portfolio_valuations.parquet` | Every open's pre/post trade balances, drifted weight, trades, funding shortfalls, and terminal valuation |
| `trades.parquet` | Actual shares, notional, costs, turnover, target weights, and fill classifications |
| `cash_events.parquet` | Entitlements, payments, interest, and split share changes |
| `targets.parquet` | Exact target decisions used by the ledger |
| `accounting_reconciliation.json` | P&L bridge, identity errors, outstanding receivables, and accounting assumptions |
| `metrics.json` | Performance, risk, exposure, trade counts, and cost summaries |
| `completion.json` | Artifact checksums and execution-source identity |

Verification reconstructs cash, shares, and receivables from event and trade flows, checks the saved wealth and return sequence, and recomputes the whole-run P&L bridge. The batch verifier also checks research inputs and loads the referenced immutable datasets.

Sharpe uses daily net excess returns with a 252-session annualization convention. CAGR uses elapsed calendar time. Drawdown includes initial capital, and Sortino uses RMS downside shortfall over all periods. Gross returns describe the P&L of the holdings actually financed in that run. A separate zero-cost run supplies the zero-cost performance comparison.

## Fixed-target replay

The batch replays the nested selector and four validation-locked model strategies over their saved 2007–2026 historical-index sample. It compares the original target-change accounting with the self-financing ledger at each cost setting. The comparison table reports changes in total return, Sharpe, annual turnover, and summed cost-equity fractions.

The SPY track runs buy-and-hold and all-cash accounting checks over its default eligible sample beginning with the execution after the first eligible decision in November 2013. Buy-and-hold retains shares and accumulates distributions as interest-bearing cash under its signal-change policy.

The batch summary files are `baseline_accounting_comparison.csv`, `cost_sensitivity.csv`, and `assumptions.json`. Per-strategy directories contain the full portfolio artifacts above.

The regression checks include hand-worked cash and share paths, drift, funding constraints, switches, split/distribution timing, intermediate payments, optional liquidation, invalid input timing, atomic output publication, and corrupted artifacts. Sixty randomized paths compare the engine with an independently written share-space bisection ledger.

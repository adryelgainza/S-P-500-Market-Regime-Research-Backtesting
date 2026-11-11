# Layer 1 — Data and snapshot guide

## Implemented scope

The data package provides separate historical-index and investable-SPY tracks. Each snapshot contains original source files, normalized market bars, corporate actions, cash observations, cash accrual intervals, an exchange schedule, a return-reconciliation table, and a quality report.

The notebook's historical-index study reads its verified original inputs. The data package supplies market and cash inputs to the Layer 2 portfolio ledger. Its reconciliation series describe close-to-close data consistency.

## Installation and commands

Run these commands from the project directory with its Python environment active.

```powershell
python -m pip install -e ".[download]"
python data.py import-index
python data.py refresh-spy --end 2026-07-24
python data.py verify <dataset_id>
python data.py eligible <dataset_id> --output runs/data_eligibility
python -m unittest discover -s tests -v
```

The end date is exclusive. The refresh adapter requests full provider history through retrieval before selecting the evaluation range, so historical split normalization can be reversed using the complete available split history. Only completed market sessions enter the normalized bars.

The equivalent package command is `python -m regime_research.data`. Supply `--root <directory>` before the subcommand to use another snapshot directory. Loading, verification, and eligibility calculations operate entirely offline.

## Snapshot identity and publication

Directories have names of the form `sha256-<64 hexadecimal characters>`. The address hashes a canonical manifest payload containing metadata, policies, runtime, data-code hashes, and file checksums. A changed input or provenance record produces a new identity.

Normalized CSV files inherit Dataset_ID from their manifest. The loader adds that column to every returned table. Omitting it from backing files avoids a circular hash definition.

The writer validates both in-memory and serialized data in a temporary directory before atomically publishing the completed directory. An existing identical snapshot is verified and reused. A damaged existing snapshot raises an error and is never silently repaired or overwritten. New downloads may leave provider caches under `datasets/.provider_cache/`, outside snapshot identity.

The loader verifies the content address, file sizes, file hashes, schema version, required tables, safe paths, instrument metadata, chronology, and cash-model reconciliation. A content address detects alteration relative to the referenced address. It does not authenticate a provider or reconstruct missing historical vintages.

`datasets/catalog.json` identifies the initial validated snapshots for convenience. Always record an explicit dataset ID in research configurations instead of silently following a changing latest-data pointer.

## Tables

| File | Contents |
| --- | --- |
| `bars.csv` | Instrument, session, UTC open/close/availability timestamps, OHLCV, currency, and source |
| `actions.csv` | Split/distribution events with ex-date, effective time, availability, payment date when known, amounts, and source |
| `cash_observations.csv` | Quoted rates, reference dates, modeled availability timestamps, units, day-count basis, and compounding |
| `cash_intervals.csv` | Reference and credited returns for each observed open-to-open interval, selected quote ID, and availability policy |
| `schedule.csv` | Persisted exchange session dates and UTC open/close timestamps |
| `reconciliation.csv` | Explicit-action close returns, provider adjusted returns, and differences attributable to adjustment conventions |
| `quality_report.json` | Coverage, missing sessions, suspicious histories, decade summaries, and reconciliation diagnostics |
| `raw/` | Unmodified imported CSV bytes or the original provider response before normalization |
| `manifest.json` | Identity, schema, source requests, assumptions, policies, runtime, and checksums |

Timestamps use UTC. Session labels retain exchange-local dates. The persisted NYSE calendar includes holidays, early closes, historical opening hours, and daylight-saving transitions using pandas-market-calendars. It is an explicit calendar proxy for this first SPY implementation.

## Price and corporate-action policies

Yahoo `auto_adjust=False` preserves a distinction between Close and Adj Close, but historical OHLC still use a split-normalized basis. The adapter reconstructs as-traded OHLC by multiplying earlier observations by the product of subsequent split ratios. It applies the matching basis conversion to per-share distributions and the inverse conversion to volume, retaining original provider bytes and split factors for audit.

SPY has no split events in the initial downloaded sample. Synthetic split and distribution fixtures verify the conversion and prevent counting a split both in adjusted prices and in share holdings. Use of this adapter for a different instrument requires validating its provider-specific price, action, and volume basis.

Adjusted_Close remains a separately named diagnostic field. It is never an execution price. Historical adjusted price levels can be revised by future distributions or splits, so their point-in-time availability cannot be inferred from the session label.

A future feature pipeline must compute split-aware returns using events known by each decision, or apply a documented point-in-time adjustment. Taking an ordinary percentage change of as-traded prices across a split would create a false return.

The action adapter does not have reliable announcement or payment dates. It assigns an explicit assumed availability time at the ex-date open and retains missing payment dates. The default generic validation policy rejects those dates. The SPY adapter explicitly selects `ex_date_credit_approximation`, which the manifest and quality report disclose. Exact receivable and payment accounting requires another action source.

The reconciliation compares two conventions on matching close-to-close intervals.

- Explicit cash-action return credits distributions at the ex-date close and reinvests at that close.
- Provider adjustment consistency uses the backward dividend multiplier based on the preceding close.

Those conventions can produce small economic differences. The saved table separates that difference from an unexplained adjustment residual. Residuals above the declared tolerance reject publication of a new investable snapshot. The initial tolerance is 0.00005 in return units, or 0.5 basis points per interval.

Provider behavior and adjustment settings are documented in [the yfinance history reference](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html) and [Yahoo's adjustment explanation](https://in.help.yahoo.com/kb/adjusted-close-sln28256.html).

## Historical cash policy

FRED DFF is treated as a percent quotation of an annual simple rate using ACT/360. A quotation of 5 means 0.05 annually. A four-calendar-day interval earns `0.05 * 4 / 360` before any declared credited-rate spread.

Rate observations receive a conservative modeled availability time three US federal business days after their reference date at 23:59 New York time. This is an availability proxy. The downloaded series contains revised historical values and does not provide actual historical release timestamps or vintages.

At each execution open, the model chooses the newest reference date among available observations and holds that rate fixed through the next execution open. A rate published during the interval can first affect the following interval. An older observation published late cannot displace a newer available reference date. Observations more than ten calendar days old are rejected by default.

Accrual uses New York calendar days, including weekends and holidays. A Friday-to-Monday interval over a daylight-saving transition still accrues three calendar days. The cash-credit spread is subtracted from the annual reference rate before accrual, and both resulting returns are stored.

The zero-spread default is a hypothetical DFF-linked cash proxy. It does not assert that a retail broker or cash fund would have paid DFF. The century-long index track uses an explicit constant-zero cash convention to preserve its original assumptions.

The source series and release context are available from [FRED DFF](https://fred.stlouisfed.org/series/DFF) and [the Federal Reserve H.15 release](https://www.federalreserve.gov/releases/h15/). The modeled lag is a project assumption.

## History eligibility and purging

The default eligibility configuration requires 20 calendar years of usable training history, 200 sessions of feature warmup, 252 calibration sessions, five-session target horizons, and at least 252 purged training rows. Each target enters at the next open and ends at the open after its holding horizon.

Training labels must mature before the first calibration decision. Calibration labels must mature before the current decision. Missing required sessions, late feature availability, inadequate real history, and unrealized forward returns produce explicit exclusion reasons.

The first eligible SPY date is derived from its real available history and these requirements. The package creates no pre-inception rows and never silently substitutes index training history. Proxy training would require a separately registered experiment.

For the saved SPY sample and default requirements, the first eligible decision is 2013-11-11. A strategy beginning in 2007 would need a separately declared shorter training requirement or proxy-training design.

```python
from regime_research.data import HistoryRequirements, build_eligible_sample, load_dataset

dataset = load_dataset("sha256-<dataset hash>")
sample = build_eligible_sample(dataset, HistoryRequirements())
bars = dataset.bars
eligible_decisions = sample.eligible
excluded_decisions = sample.exclusions
```

## Quality and acceptance checks

Tests cover offline loading, content identity, retained prior snapshots, damaged files, altered manifests, invalid OHLCV, duplicate/missing sessions, early historical data quality, dividend policy, split invariance, total-return reconciliation, cash quotation units, publication lags, stale observations, holidays, daylight saving, and training/calibration chronology.

The initial historical-index audit reports 1,085 missing NYSE calendar sessions, 5,497 zero-volume records, and 8,547 flat OHLC records. The full dates and decade breakdown are retained in its quality report. These observations are preserved for the historical study and excluded from eligible windows when required history spans a gap.

The initial SPY snapshot contains 8,427 sessions from 1993-01-29 through 2026-07-23 and 135 distribution events. Its schedule coverage is complete within that observed range. Missing dividend payment dates are disclosed through the approximation policy.

The current data package has dependency ranges in pyproject.toml. Layer 2 now provides the portfolio ledger and fixed-target accounting replay. The full research-environment lock and clean notebook runner are specified in Layer 6. [Portfolio accounting guide](PORTFOLIO_ACCOUNTING.md)
